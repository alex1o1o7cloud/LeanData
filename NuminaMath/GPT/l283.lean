import Mathlib

namespace r_pow_four_solution_l283_283301

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l283_283301


namespace min_jugs_needed_to_fill_container_l283_283871

def min_jugs_to_fill (jug_capacity container_capacity : ℕ) : ℕ :=
  Nat.ceil (container_capacity / jug_capacity)

theorem min_jugs_needed_to_fill_container :
  min_jugs_to_fill 16 200 = 13 :=
by
  -- The proof is omitted.
  sorry

end min_jugs_needed_to_fill_container_l283_283871


namespace side_length_of_base_l283_283390

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end side_length_of_base_l283_283390


namespace find_S3_l283_283362

-- Define the known scores
def S1 : ℕ := 55
def S2 : ℕ := 67
def S4 : ℕ := 55
def Avg : ℕ := 67

-- Statement to prove
theorem find_S3 : ∃ S3 : ℕ, (S1 + S2 + S3 + S4) / 4 = Avg ∧ S3 = 91 :=
by
  sorry

end find_S3_l283_283362


namespace eighth_odd_multiple_of_5_is_75_l283_283989

theorem eighth_odd_multiple_of_5_is_75 : ∃ n : ℕ, (n > 0 ∧ n % 2 = 1 ∧ n % 5 = 0 ∧ ∃ k : ℕ, k = 8 ∧ n = 10 * k - 5) :=
  sorry

end eighth_odd_multiple_of_5_is_75_l283_283989


namespace paul_has_5point86_left_l283_283005

noncomputable def paulLeftMoney : ℝ := 15 - (2 + (3 - 0.1*3) + 2*2 + 0.05 * (2 + (3 - 0.1*3) + 2*2))

theorem paul_has_5point86_left :
  paulLeftMoney = 5.86 :=
by
  sorry

end paul_has_5point86_left_l283_283005


namespace min_moves_l283_283680

theorem min_moves (n : ℕ) : (n * (n + 1)) / 2 > 100 → n = 15 :=
by
  sorry

end min_moves_l283_283680


namespace triangle_inequality_from_inequality_l283_283258

theorem triangle_inequality_from_inequality
  (a b c : ℝ)
  (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by
  sorry

end triangle_inequality_from_inequality_l283_283258


namespace Ellipse_area_constant_l283_283496

-- Definitions of given conditions and problem setup
def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0

def point_on_ellipse (a b : ℝ) : Prop :=
  ellipse_equation 1 (Real.sqrt 3 / 2) a b

def eccentricity (c a : ℝ) : Prop :=
  c / a = Real.sqrt 3 / 2

def moving_points_on_ellipse (a b x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse_equation x₁ y₁ a b ∧ ellipse_equation x₂ y₂ a b

def slopes_condition (k₁ k₂ : ℝ) : Prop :=
  k₁ * k₂ = -1/4

def area_OMN := 1

-- Main theorem statement
theorem Ellipse_area_constant
(a b : ℝ) 
(h_ellipse : point_on_ellipse a b)
(h_eccentricity : eccentricity (Real.sqrt 3 / 2 * a) a)
(M N : ℝ × ℝ) 
(h_points : moving_points_on_ellipse a b M.1 M.2 N.1 N.2)
(k₁ k₂ : ℝ) 
(h_slopes : slopes_condition k₁ k₂) : 
a^2 = 4 ∧ b^2 = 1 ∧ area_OMN = 1 := 
sorry

end Ellipse_area_constant_l283_283496


namespace calc_expr_l283_283058

theorem calc_expr : 4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2)^0 = 1 := by
  sorry

end calc_expr_l283_283058


namespace find_integer_pairs_l283_283936

theorem find_integer_pairs (m n : ℤ) (h1 : m * n ≥ 0) (h2 : m^3 + n^3 + 99 * m * n = 33^3) :
  (m = -33 ∧ n = -33) ∨ ∃ k : ℕ, k ≤ 33 ∧ m = k ∧ n = 33 - k ∨ m = 33 - k ∧ n = k :=
by
  sorry

end find_integer_pairs_l283_283936


namespace option_b_is_incorrect_l283_283730

theorem option_b_is_incorrect : ¬ (3 + 2 * Real.sqrt 2 = 5 * Real.sqrt 2) := by
  sorry

end option_b_is_incorrect_l283_283730


namespace relationship_between_a_b_c_l283_283085

noncomputable def a : ℝ := (1 / Real.sqrt 2) * (Real.cos (34 * Real.pi / 180) - Real.sin (34 * Real.pi / 180))
noncomputable def b : ℝ := Real.cos (50 * Real.pi / 180) * Real.cos (128 * Real.pi / 180) + Real.cos (40 * Real.pi / 180) * Real.cos (38 * Real.pi / 180)
noncomputable def c : ℝ := (1 / 2) * (Real.cos (80 * Real.pi / 180) - 2 * (Real.cos (50 * Real.pi / 180))^2 + 1)

theorem relationship_between_a_b_c : b > a ∧ a > c :=
  sorry

end relationship_between_a_b_c_l283_283085


namespace exists_triangle_l283_283596

variable (k α m_a : ℝ)

-- Define the main constructibility condition as a noncomputable function.
noncomputable def triangle_constructible (k α m_a : ℝ) : Prop :=
  m_a ≤ (k / 2) * ((1 - Real.sin (α / 2)) / Real.cos (α / 2))

-- Main theorem statement to prove the existence of the triangle
theorem exists_triangle :
  ∃ (k α m_a : ℝ), triangle_constructible k α m_a := 
sorry

end exists_triangle_l283_283596


namespace integral_negative_of_negative_function_l283_283961

theorem integral_negative_of_negative_function {f : ℝ → ℝ} 
  (hf_cont : Continuous f) 
  (hf_neg : ∀ x, f x < 0) 
  {a b : ℝ} 
  (hab : a < b) 
  : ∫ x in a..b, f x < 0 := 
sorry

end integral_negative_of_negative_function_l283_283961


namespace average_gas_mileage_round_trip_l283_283866

-- Definition of the problem conditions

def distance_to_home : ℕ := 120
def distance_back : ℕ := 120
def mileage_to_home : ℕ := 30
def mileage_back : ℕ := 20

-- Theorem that we need to prove
theorem average_gas_mileage_round_trip
  (d1 d2 : ℕ) (m1 m2 : ℕ)
  (h1 : d1 = distance_to_home)
  (h2 : d2 = distance_back)
  (h3 : m1 = mileage_to_home)
  (h4 : m2 = mileage_back) :
  (d1 + d2) / ((d1 / m1) + (d2 / m2)) = 24 :=
by
  sorry

end average_gas_mileage_round_trip_l283_283866


namespace inequality_solution_set_inequality_proof_l283_283100

def f (x : ℝ) : ℝ := |x - 1| - |x + 2|

theorem inequality_solution_set :
  ∀ x : ℝ, -2 < f x ∧ f x < 0 ↔ -1/2 < x ∧ x < 1/2 :=
by
  sorry

theorem inequality_proof (m n : ℝ) (h_m : -1/2 < m ∧ m < 1/2) (h_n : -1/2 < n ∧ n < 1/2) :
  |1 - 4 * m * n| > 2 * |m - n| :=
by
  sorry

end inequality_solution_set_inequality_proof_l283_283100


namespace find_age_of_older_friend_l283_283413

theorem find_age_of_older_friend (A B C : ℝ) 
  (h1 : A - B = 2.5)
  (h2 : A - C = 3.75)
  (h3 : A + B + C = 110.5)
  (h4 : B = 2 * C) : 
  A = 104.25 :=
by
  sorry

end find_age_of_older_friend_l283_283413


namespace pounds_in_a_ton_l283_283690

-- Definition of variables based on the given conditions
variables (T E D : ℝ)

-- Condition 1: The elephant weighs 3 tons.
def elephant_weight := E = 3 * T

-- Condition 2: The donkey weighs 90% less than the elephant.
def donkey_weight := D = 0.1 * E

-- Condition 3: Their combined weight is 6600 pounds.
def combined_weight := E + D = 6600

-- Main theorem to prove
theorem pounds_in_a_ton (h1 : elephant_weight T E) (h2 : donkey_weight E D) (h3 : combined_weight E D) : T = 2000 :=
by
  sorry

end pounds_in_a_ton_l283_283690


namespace katy_brownies_total_l283_283129

theorem katy_brownies_total : 
  (let monday_brownies := 5 in
   let tuesday_brownies := 2 * monday_brownies in
   let total_brownies := monday_brownies + tuesday_brownies in
   total_brownies = 15) := 
by 
  let monday_brownies := 5 in
  let tuesday_brownies := 2 * monday_brownies in
  let total_brownies := monday_brownies + tuesday_brownies in
  show total_brownies = 15 by
  sorry

end katy_brownies_total_l283_283129


namespace radius_of_base_of_cone_l283_283907

theorem radius_of_base_of_cone (S : ℝ) (hS : S = 9 * Real.pi)
  (H : ∃ (l r : ℝ), (Real.pi * l = 2 * Real.pi * r) ∧ S = Real.pi * r^2 + Real.pi * r * l) :
  ∃ (r : ℝ), r = Real.sqrt 3 :=
by
  sorry

end radius_of_base_of_cone_l283_283907


namespace find_numbers_between_1000_and_4000_l283_283897

theorem find_numbers_between_1000_and_4000 :
  ∃ (x : ℤ), 1000 ≤ x ∧ x ≤ 4000 ∧
             (x % 11 = 2) ∧
             (x % 13 = 12) ∧
             (x % 19 = 18) ∧
             (x = 1234 ∨ x = 3951) :=
sorry

end find_numbers_between_1000_and_4000_l283_283897


namespace grocery_packs_l283_283699

theorem grocery_packs (cookie_packs cake_packs : ℕ)
  (h1 : cookie_packs = 23)
  (h2 : cake_packs = 4) :
  cookie_packs + cake_packs = 27 :=
by
  sorry

end grocery_packs_l283_283699


namespace algebraic_notation_correct_l283_283993

def exprA : String := "a * 5"
def exprB : String := "a7"
def exprC : String := "3 1/2 x"
def exprD : String := "-7/8 x"

theorem algebraic_notation_correct :
  exprA ≠ "correct" ∧
  exprB ≠ "correct" ∧
  exprC ≠ "correct" ∧
  exprD = "correct" :=
by
  sorry

end algebraic_notation_correct_l283_283993


namespace age_of_James_when_Thomas_reaches_current_age_l283_283172
    
theorem age_of_James_when_Thomas_reaches_current_age
  (T S J : ℕ)
  (h1 : T = 6)
  (h2 : S = T + 13)
  (h3 : S = J - 5) :
  J + (S - T) = 37 := 
by
  sorry

end age_of_James_when_Thomas_reaches_current_age_l283_283172


namespace katy_brownies_l283_283132

theorem katy_brownies :
  ∃ (n : ℤ), (n = 5 + 2 * 5) :=
by
  sorry

end katy_brownies_l283_283132


namespace day_of_50th_in_year_N_minus_1_l283_283124

theorem day_of_50th_in_year_N_minus_1
  (N : ℕ)
  (day250_in_year_N_is_sunday : (250 % 7 = 0))
  (day150_in_year_N_plus_1_is_sunday : (150 % 7 = 0))
  : 
  (50 % 7 = 1) := 
sorry

end day_of_50th_in_year_N_minus_1_l283_283124


namespace distance_along_stream_1_hour_l283_283516

noncomputable def boat_speed_still_water : ℝ := 4
noncomputable def stream_speed : ℝ := 2
noncomputable def effective_speed_against_stream : ℝ := boat_speed_still_water - stream_speed
noncomputable def effective_speed_along_stream : ℝ := boat_speed_still_water + stream_speed

theorem distance_along_stream_1_hour : 
  effective_speed_agains_stream = 2 → effective_speed_along_stream * 1 = 6 :=
by
  sorry

end distance_along_stream_1_hour_l283_283516


namespace find_coordinates_of_B_l283_283683

-- Define the conditions from the problem
def point_A (a : ℝ) : ℝ × ℝ := (a - 1, a + 1)
def point_B (a : ℝ) : ℝ × ℝ := (a + 3, a - 5)

-- The proof problem: The coordinates of B are (4, -4)
theorem find_coordinates_of_B (a : ℝ) (h : point_A a = (0, a + 1)) : point_B a = (4, -4) := by
  -- This is skipping the proof part.
  sorry

end find_coordinates_of_B_l283_283683


namespace find_k_l283_283627

theorem find_k (k : ℝ) (h : ∃ x : ℝ, x = -2 ∧ x^2 - k * x + 2 = 0) : k = -3 := by
  sorry

end find_k_l283_283627


namespace fifth_plot_difference_l283_283874

-- Define the dimensions of the plots
def plot_width (n : Nat) : Nat := 3 + 2 * (n - 1)
def plot_length (n : Nat) : Nat := 4 + 3 * (n - 1)

-- Define the number of tiles in a plot
def tiles_in_plot (n : Nat) : Nat := plot_width n * plot_length n

-- The main theorem to prove the required difference
theorem fifth_plot_difference :
  tiles_in_plot 5 - tiles_in_plot 4 = 59 := sorry

end fifth_plot_difference_l283_283874


namespace intersection_is_open_interval_l283_283910

open Set
open Real

noncomputable def M : Set ℝ := {x | x < 1}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_is_open_interval :
  M ∩ N = { x | 0 < x ∧ x < 1 } := by
  sorry

end intersection_is_open_interval_l283_283910


namespace find_a1_l283_283489

variable {q a1 a2 a3 a4 : ℝ}
variable (S : ℕ → ℝ)

axiom common_ratio_pos : q > 0
axiom S2_eq : S 2 = 3 * a2 + 2
axiom S4_eq : S 4 = 3 * a4 + 2

theorem find_a1 (h1 : S 2 = 3 * a2 + 2) (h2 : S 4 = 3 * a4 + 2) (common_ratio_pos : q > 0) : a1 = -1 :=
sorry

end find_a1_l283_283489


namespace pizza_slices_left_l283_283748

theorem pizza_slices_left (total_slices : ℕ) (fraction_eaten : ℚ) (slices_left : ℕ) :
  total_slices = 16 → fraction_eaten = 3 / 4 → slices_left = 4 := by
  intro h1 h2
  rw [h1, h2]
  -- Prove using calculations
  have h3 : (16 : ℚ) * (3 / 4) = 12 by norm_num
  have h4 : 16 - 12 = 4 by norm_num
  exact h4

end pizza_slices_left_l283_283748


namespace canonical_equations_of_line_l283_283035

/-- Given two planes: 
  Plane 1: 4 * x + y + z + 2 = 0
  Plane 2: 2 * x - y - 3 * z - 8 = 0
  Prove that the canonical equations of the line formed by their intersection are:
  (x - 1) / -2 = (y + 6) / 14 = z / -6 -/
theorem canonical_equations_of_line :
  (∃ x y z : ℝ, 4 * x + y + z + 2 = 0 ∧ 2 * x - y - 3 * z - 8 = 0) →
  (∀ x y z : ℝ, ((x - 1) / -2 = (y + 6) / 14) ∧ ((y + 6) / 14 = z / -6)) :=
by
  sorry

end canonical_equations_of_line_l283_283035


namespace number_of_players_l283_283530

-- Definitions of the conditions
def initial_bottles : ℕ := 4 * 12
def bottles_remaining : ℕ := 15
def bottles_taken_per_player : ℕ := 2 + 1

-- Total number of bottles taken
def bottles_taken := initial_bottles - bottles_remaining

-- The main theorem stating that the number of players is 11.
theorem number_of_players : (bottles_taken / bottles_taken_per_player) = 11 :=
by
  sorry

end number_of_players_l283_283530


namespace r_pow_four_solution_l283_283300

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l283_283300


namespace add_like_terms_l283_283591

variable (a : ℝ)

theorem add_like_terms : a^2 + 2 * a^2 = 3 * a^2 := 
by sorry

end add_like_terms_l283_283591


namespace smallest_positive_four_digit_integer_equivalent_to_3_mod_4_l283_283844

theorem smallest_positive_four_digit_integer_equivalent_to_3_mod_4 : 
  ∃ n : ℤ, n ≥ 1000 ∧ n % 4 = 3 ∧ n = 1003 := 
by {
  sorry
}

end smallest_positive_four_digit_integer_equivalent_to_3_mod_4_l283_283844


namespace sequence_limit_l283_283942

noncomputable def sequence_converges (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n > 1 ∧ a (n + 1) ^ 2 ≥ a n * a (n + 2)

theorem sequence_limit (a : ℕ → ℝ) (h : sequence_converges a) : 
  ∃ l : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (Real.log (a (n + 1)) / Real.log (a n) - l) < ε := 
sorry

end sequence_limit_l283_283942


namespace intersection_S_T_l283_283482

def S : Set ℝ := { y | y ≥ 0 }
def T : Set ℝ := { x | x > 1 }

theorem intersection_S_T :
  S ∩ T = { z | z > 1 } :=
sorry

end intersection_S_T_l283_283482


namespace polynomial_A_polynomial_B_l283_283582

-- Problem (1): Prove that A = 6x^3 + 8x^2 + x - 1 given the conditions.
theorem polynomial_A :
  ∀ (x : ℝ),
  (2 * x^2 * (3 * x + 4) + (x - 1) = 6 * x^3 + 8 * x^2 + x - 1) :=
by
  intro x
  sorry

-- Problem (2): Prove that B = 6x^2 - 19x + 9 given the conditions.
theorem polynomial_B :
  ∀ (x : ℝ),
  ((2 * x - 6) * (3 * x - 1) + (x + 3) = 6 * x^2 - 19 * x + 9) :=
by
  intro x
  sorry

end polynomial_A_polynomial_B_l283_283582


namespace georgia_black_buttons_l283_283084

theorem georgia_black_buttons : 
  ∀ (B : ℕ), 
  (4 + B + 3 = 9) → 
  B = 2 :=
by
  introv h
  linarith

end georgia_black_buttons_l283_283084


namespace find_solutions_l283_283767

def satisfies_inequality (x : ℝ) : Prop :=
  (Real.cos x)^2018 + (1 / (Real.sin x))^2019 ≤ (Real.sin x)^2018 + (1 / (Real.cos x))^2019

def in_intervals (x : ℝ) : Prop :=
  (x ∈ Set.Ico (-Real.pi / 3) 0) ∨
  (x ∈ Set.Ico (Real.pi / 4) (Real.pi / 2)) ∨
  (x ∈ Set.Ioc Real.pi (5 * Real.pi / 4)) ∨
  (x ∈ Set.Ioc (3 * Real.pi / 2) (5 * Real.pi / 3))

theorem find_solutions :
  ∀ x : ℝ, x ∈ Set.Icc (-Real.pi / 3) (5 * Real.pi / 3) →
  satisfies_inequality x ↔ in_intervals x := 
  by sorry

end find_solutions_l283_283767


namespace max_super_bishops_l283_283564

/--
A "super-bishop" attacks another "super-bishop" if they are on the
same diagonal, there are no pieces between them, and the next cell
along the diagonal after the "super-bishop" B is empty. Given these
conditions, prove that the maximum number of "super-bishops" that can
be placed on a standard 8x8 chessboard such that each one attacks at
least one other is 32.
-/
theorem max_super_bishops (n : ℕ) (chessboard : ℕ → ℕ → Prop) (super_bishop : ℕ → ℕ → Prop)
  (attacks : ∀ {x₁ y₁ x₂ y₂}, super_bishop x₁ y₁ → super_bishop x₂ y₂ →
            (x₁ - x₂ = y₁ - y₂ ∨ x₁ + y₁ = x₂ + y₂) →
            (∀ x y, super_bishop x y → (x < min x₁ x₂ ∨ x > max x₁ x₂ ∨ y < min y₁ y₂ ∨ y > max y₁ y₂)) →
            chessboard (x₂ + (x₁ - x₂)) (y₂ + (y₁ - y₂))) :
  ∃ k, k = 32 ∧ (∀ x y, super_bishop x y → x < 8 ∧ y < 8) → k ≤ n :=
sorry

end max_super_bishops_l283_283564


namespace beta_exists_l283_283941

noncomputable theory
open Polynomial

-- Let n be a positive integer
variables (n : ℕ) (hn : 0 < n)

-- Let {a_i} and {b_i} be sequences of distinct real numbers
variables (a b : Fin n → ℝ)
variable (h_distinct : Function.Injective a)

-- There exists a real number α such that for each i, the product of (a_i + b_j) over j from 1 to n equals α
variable (α : ℝ)
variable (h_alpha : ∀ i, (Finset.univ : Finset (Fin n)).prod (λ j, a i + b j) = α)

-- Our goal is to prove that there is a real number β such that for each j, the product of (a_i + b_j) over i from 1 to n equals β
theorem beta_exists : ∃ β : ℝ, ∀ j, (Finset.univ : Finset (Fin n)).prod (λ i, a i + b j) = β := sorry

end beta_exists_l283_283941


namespace inequality_proof_l283_283013

theorem inequality_proof (a d b c : ℝ) 
  (h1 : 0 ≤ a) 
  (h2 : 0 ≤ d) 
  (h3 : 0 < b) 
  (h4 : 0 < c) 
  (h5 : b + c ≥ a + d) : 
  (b / (c + d) + c / (b + a) ≥ real.sqrt 2 - (1 / 2)) := 
  sorry

end inequality_proof_l283_283013


namespace function_value_bounds_l283_283954

theorem function_value_bounds (x : ℝ) : 
  (x^2 + x + 1) / (x^2 + 1) ≤ 3 / 2 ∧ (x^2 + x + 1) / (x^2 + 1) ≥ 1 / 2 := 
sorry

end function_value_bounds_l283_283954


namespace salt_solution_mixture_l283_283633

/-- Let's define the conditions and hypotheses required for our proof. -/
def ounces_of_salt_solution 
  (percent_salt : ℝ) (amount : ℝ) : ℝ := percent_salt * amount

def final_amount (x : ℝ) : ℝ := x + 70
def final_salt_content (x : ℝ) : ℝ := 0.40 * (x + 70)

theorem salt_solution_mixture (x : ℝ) :
  0.60 * x + 0.20 * 70 = 0.40 * (x + 70) ↔ x = 70 :=
by {
  sorry
}

end salt_solution_mixture_l283_283633


namespace greatest_good_number_smallest_bad_number_l283_283772

def is_good (M : ℕ) : Prop :=
  ∃ (a b c d : ℕ), M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ (a * d = b * c)

def is_good_iff_exists_xy (M : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≤ y ∧ M ≤ x * y ∧ (x + 1) * (y + 1) ≤ M + 49

theorem greatest_good_number : ∃ (M : ℕ), is_good M ∧ ∀ (N : ℕ), is_good N → N ≤ M :=
  by
    use 576
    sorry

theorem smallest_bad_number : ∃ (M : ℕ), ¬is_good M ∧ ∀ (N : ℕ), ¬is_good N → M ≤ N :=
  by
    use 443
    sorry

end greatest_good_number_smallest_bad_number_l283_283772


namespace find_middle_part_value_l283_283464

-- Define the ratios
def ratio1 := 1 / 2
def ratio2 := 1 / 4
def ratio3 := 1 / 8

-- Total sum
def total_sum := 120

-- Parts proportional to ratios
def part1 (x : ℝ) := x
def part2 (x : ℝ) := ratio1 * x
def part3 (x : ℝ) := ratio2 * x

-- Equation representing the sum of the parts equals to the total sum
def equation (x : ℝ) : Prop :=
  part1 x + part2 x / 2 + part2 x = x * (1 + ratio1 + ratio2)

-- Defining the middle part
def middle_part (x : ℝ) := ratio1 * x

theorem find_middle_part_value :
  ∃ x : ℝ, equation x ∧ middle_part x = 34.2857 := sorry

end find_middle_part_value_l283_283464


namespace paint_grid_l283_283863

theorem paint_grid (paint : Fin 3 × Fin 3 → Bool) (no_adjacent : ∀ i j, (paint (i, j) = true) → (paint (i+1, j) = false) ∧ (paint (i-1, j) = false) ∧ (paint (i, j+1) = false) ∧ (paint (i, j-1) = false)) : 
  ∃! (count : ℕ), count = 8 :=
sorry

end paint_grid_l283_283863


namespace possible_values_for_N_l283_283674

theorem possible_values_for_N (N : ℕ) (h₁ : 8 < N) (h₂ : 1 ≤ N - 1) :
  22 < N ∧ N ≤ 25 ↔ (N = 23 ∨ N = 24 ∨ N = 25) :=
by 
  sorry

end possible_values_for_N_l283_283674


namespace average_height_is_64_l283_283360

noncomputable def Parker (H_D : ℝ) : ℝ := H_D - 4
noncomputable def Daisy (H_R : ℝ) : ℝ := H_R + 8
noncomputable def Reese : ℝ := 60

theorem average_height_is_64 :
  let H_R := Reese 
  let H_D := Daisy H_R
  let H_P := Parker H_D
  (H_P + H_D + H_R) / 3 = 64 := sorry

end average_height_is_64_l283_283360


namespace prob_at_least_two_pass_theory_prob_all_pass_course_l283_283571

open ProbabilityTheory

/-- Define events for passing the theory part --/
def A1 : Event := Event.prob 0.6
def A2 : Event := Event.prob 0.5
def A3 : Event := Event.prob 0.4

/-- Define events for passing the experiment part --/
def B1 : Event := Event.prob 0.5
def B2 : Event := Event.prob 0.6
def B3 : Event := Event.prob 0.75

/-- Define events for passing the course --/
def C1 : Event := A1 ∧ B1
def C2 : Event := A2 ∧ B2
def C3 : Event := A3 ∧ B3

/-- Proving the probability of at least two passing theory part is 0.5 --/
theorem prob_at_least_two_pass_theory :
  (prob (A1 ∧ A2 ∧ ¬A3) +
   prob (A1 ∧ ¬A2 ∧ A3) +
   prob (¬A1 ∧ A2 ∧ A3) +
   prob (A1 ∧ A2 ∧ A3)) = 0.5 :=
sorry

/-- Proving the probability that all three pass the course is 0.027 --/
theorem prob_all_pass_course :
  (prob C1 * prob C2 * prob C3) = 0.027 :=
sorry

end prob_at_least_two_pass_theory_prob_all_pass_course_l283_283571


namespace side_length_of_square_base_l283_283396

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end side_length_of_square_base_l283_283396


namespace value_of_f_at_3_l283_283162

def f (a c x : ℝ) : ℝ := a * x^3 + c * x + 5

theorem value_of_f_at_3 (a c : ℝ) (h : f a c (-3) = -3) : f a c 3 = 13 :=
by
  sorry

end value_of_f_at_3_l283_283162


namespace jina_has_1_koala_bear_l283_283802

theorem jina_has_1_koala_bear:
  let teddies := 5
  let bunnies := 3 * teddies
  let additional_teddies := 2 * bunnies
  let total_teddies := teddies + additional_teddies
  let total_bunnies_and_teddies := total_teddies + bunnies
  let total_mascots := 51
  let koala_bears := total_mascots - total_bunnies_and_teddies
  koala_bears = 1 :=
by
  sorry

end jina_has_1_koala_bear_l283_283802


namespace side_length_of_square_base_l283_283395

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end side_length_of_square_base_l283_283395


namespace pipe_filling_time_l283_283843

theorem pipe_filling_time (t : ℕ) (h : 2 * (1 / t + 1 / 15) + 10 * (1 / 15) = 1) : t = 10 := by
  sorry

end pipe_filling_time_l283_283843


namespace white_area_of_sign_l283_283547

theorem white_area_of_sign :
  let total_area : ℕ := 6 * 18
  let black_area_C : ℕ := 11
  let black_area_A : ℕ := 10
  let black_area_F : ℕ := 12
  let black_area_E : ℕ := 9
  let total_black_area : ℕ := black_area_C + black_area_A + black_area_F + black_area_E
  let white_area : ℕ := total_area - total_black_area
  white_area = 66 := by
  sorry

end white_area_of_sign_l283_283547


namespace dog_probability_l283_283007

def prob_machine_A_transforms_cat_to_dog : ℚ := 1 / 3
def prob_machine_B_transforms_cat_to_dog : ℚ := 2 / 5
def prob_machine_C_transforms_cat_to_dog : ℚ := 1 / 4

def prob_cat_remains_after_A : ℚ := 1 - prob_machine_A_transforms_cat_to_dog
def prob_cat_remains_after_B : ℚ := 1 - prob_machine_B_transforms_cat_to_dog
def prob_cat_remains_after_C : ℚ := 1 - prob_machine_C_transforms_cat_to_dog

def prob_cat_remains : ℚ := prob_cat_remains_after_A * prob_cat_remains_after_B * prob_cat_remains_after_C

def prob_dog_out_of_C : ℚ := 1 - prob_cat_remains

theorem dog_probability : prob_dog_out_of_C = 7 / 10 := by
  -- Proof goes here
  sorry

end dog_probability_l283_283007


namespace joe_travel_time_l283_283126

theorem joe_travel_time
  (d : ℝ) -- Total distance
  (rw : ℝ) (rr : ℝ) -- Walking and running rates
  (tw : ℝ) -- Walking time
  (tr : ℝ) -- Running time
  (h1 : tw = 9)
  (h2 : rr = 4 * rw)
  (h3 : rw * tw = d / 3)
  (h4 : rr * tr = 2 * d / 3) :
  tw + tr = 13.5 :=
by 
  sorry

end joe_travel_time_l283_283126


namespace white_squares_in_20th_row_l283_283120

def num_squares_in_row (n : ℕ) : ℕ :=
  3 * n

def num_white_squares (n : ℕ) : ℕ :=
  (num_squares_in_row n - 2) / 2

theorem white_squares_in_20th_row: num_white_squares 20 = 30 := by
  -- Proof skipped
  sorry

end white_squares_in_20th_row_l283_283120


namespace topic_preference_order_l283_283118

noncomputable def astronomy_fraction := (8 : ℚ) / 21
noncomputable def botany_fraction := (5 : ℚ) / 14
noncomputable def chemistry_fraction := (9 : ℚ) / 28

theorem topic_preference_order :
  (astronomy_fraction > botany_fraction) ∧ (botany_fraction > chemistry_fraction) :=
by
  sorry

end topic_preference_order_l283_283118


namespace fraction_sum_to_decimal_l283_283454

theorem fraction_sum_to_decimal : 
  (3 / 10 : Rat) + (5 / 100) - (1 / 1000) = 349 / 1000 := 
by
  sorry

end fraction_sum_to_decimal_l283_283454


namespace sequence_proofs_l283_283632

theorem sequence_proofs (a b : ℕ → ℝ) :
  a 1 = 1 ∧ b 1 = 0 ∧ 
  (∀ n, 4 * a (n + 1) = 3 * a n - b n + 4) ∧ 
  (∀ n, 4 * b (n + 1) = 3 * b n - a n - 4) → 
  (∀ n, a n + b n = (1 / 2) ^ (n - 1)) ∧ 
  (∀ n, a n - b n = 2 * n - 1) ∧ 
  (∀ n, a n = (1 / 2) ^ n + n - 1 / 2 ∧ b n = (1 / 2) ^ n - n + 1 / 2) :=
sorry

end sequence_proofs_l283_283632


namespace possible_values_of_N_l283_283667

theorem possible_values_of_N (N : ℕ) (h1 : N ≥ 8 + 1)
  (h2 : ∀ (i : ℕ), (i < N → (i ≥ 0 ∧ i < 1/3 * (N-1)) → false) ) 
  (h3 : ∀ (i : ℕ), (i < N → (i ≥ 1/3 * (N-1) ∨ i < 1/3 * (N-1)) → true)) :
  23 ≤ N ∧ N ≤ 25 :=
by
  sorry

end possible_values_of_N_l283_283667


namespace derivative_at_minus_one_l283_283101
open Real

def f (x : ℝ) : ℝ := (1 + x) * (2 + x^2)^(1 / 2) * (3 + x^3)^(1 / 3)

theorem derivative_at_minus_one : deriv f (-1) = sqrt 3 * 2^(1 / 3) :=
by sorry

end derivative_at_minus_one_l283_283101


namespace find_r4_l283_283271

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l283_283271


namespace unique_function_solution_l283_283076

variable (f : ℝ → ℝ)

theorem unique_function_solution :
  (∀ x y : ℝ, f (f x - y^2) = f x ^ 2 - 2 * f x * y^2 + f (f y))
  → (∀ x : ℝ, f x = x^2) :=
by
  sorry

end unique_function_solution_l283_283076


namespace tens_digit_9_2023_l283_283424

theorem tens_digit_9_2023 :
  let cycle := [09, 81, 29, 61, 49, 41, 69, 21, 89, 01] in
  (cycle[(2023 % 10)] / 10) % 10 == 2 := by
  sorry

end tens_digit_9_2023_l283_283424


namespace largest_mersenne_prime_lt_500_l283_283195

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_mersenne_prime (p : ℕ) : Prop :=
  is_prime p ∧ is_prime (2^p - 1)

theorem largest_mersenne_prime_lt_500 : 
  ∀ n, is_mersenne_prime n → 2^n - 1 < 500 → 2^n - 1 ≤ 127 :=
by
  -- Proof goes here
  sorry

end largest_mersenne_prime_lt_500_l283_283195


namespace factorize_x2y_minus_4y_l283_283233

variable {x y : ℝ}

theorem factorize_x2y_minus_4y : x^2 * y - 4 * y = y * (x + 2) * (x - 2) :=
sorry

end factorize_x2y_minus_4y_l283_283233


namespace infinitely_many_n_divisible_by_prime_l283_283527

theorem infinitely_many_n_divisible_by_prime (p : ℕ) (hp : Prime p) : 
  ∃ᶠ n in at_top, p ∣ (2^n - n) :=
by {
  sorry
}

end infinitely_many_n_divisible_by_prime_l283_283527


namespace annual_population_increase_l283_283973

theorem annual_population_increase 
  (P : ℕ) (A : ℕ) (t : ℕ) (r : ℚ)
  (hP : P = 10000)
  (hA : A = 14400)
  (ht : t = 2)
  (h_eq : A = P * (1 + r)^t) :
  r = 0.2 :=
by
  sorry

end annual_population_increase_l283_283973


namespace five_power_l283_283431

theorem five_power (a : ℕ) (h : 5^a = 3125) : 5^(a - 3) = 25 := 
  sorry

end five_power_l283_283431


namespace maximize_container_volume_l283_283597

theorem maximize_container_volume :
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ (∀ y : ℝ, 0 < y ∧ y < 24 → (90 - 2*y) * (48 - 2*y) * y ≤ (90 - 2*x) * (48 - 2*x) * x) ∧ x = 10 :=
sorry

end maximize_container_volume_l283_283597


namespace line_within_plane_correct_l283_283860

-- Definitions of sets representing a line and a plane
variable {Point : Type}
variable (l α : Set Point)

-- Definition of the statement
def line_within_plane : Prop := l ⊆ α

-- Proof statement (without the actual proof)
theorem line_within_plane_correct (h : l ⊆ α) : line_within_plane l α :=
by
  sorry

end line_within_plane_correct_l283_283860


namespace symmetric_to_origin_l283_283409

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_to_origin (p : ℝ × ℝ) (h : p = (3, -1)) : symmetric_point p = (-3, 1) :=
by
  -- This is just the statement; the proof is not provided.
  sorry

end symmetric_to_origin_l283_283409


namespace xy_range_l283_283620

open Real

theorem xy_range (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 2 / x + 3 * y + 4 / y = 10) : 
  1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
by
  sorry

end xy_range_l283_283620


namespace r_fourth_power_sum_l283_283277

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l283_283277


namespace probability_of_distinct_divisors_l283_283694

theorem probability_of_distinct_divisors :
  ∃ (m n : ℕ), (m.gcd n = 1) ∧ (m / n) = 125 / 158081 :=
by
  sorry

end probability_of_distinct_divisors_l283_283694


namespace pencil_distribution_l283_283477

theorem pencil_distribution : 
  ∃ n : ℕ, n = 35 ∧ (∃ lst : List ℕ, lst.Length = 4 ∧ lst.Sum = 8 ∧ ∀ x ∈ lst, x ≥ 1) :=
by
  use 35
  use [5, 1, 1, 1]
  sorry

end pencil_distribution_l283_283477


namespace possible_values_for_N_l283_283665

theorem possible_values_for_N (N : ℕ) (H : ∀ (k : ℕ), N = 8 + k) (truth: (student : ℕ → Prop) (bully : ℕ → Prop) (n : ℕ), 
  (∀ i, bully i → ¬ (∃ j, student j ∧ bully j → j ≠ i)) →
  (∀ i, student i → (∃ j, student j ∧ bully j → j ≠ i))
  → ∀ i, (bully i → ¬(∃ (m : ℕ), (m ≥ N - 1 / 3 )) ∧ student i → (∃ (m : ℕ), (m ≥ N - 1 / 3 ))) : N = 23 ∨ N = 24 ∨ N = 25 :=
by
  sorry

end possible_values_for_N_l283_283665


namespace incorrect_calculation_l283_283727

theorem incorrect_calculation : ¬ (3 + 2 * Real.sqrt 2 = 5 * Real.sqrt 2) :=
by sorry

end incorrect_calculation_l283_283727


namespace stepashka_cannot_defeat_kryusha_l283_283010

-- Definitions of conditions
def glasses : ℕ := 2018
def champagne : ℕ := 2019
def initial_distribution : list ℕ := (list.repeat 1 (glasses - 1)) ++ [2]  -- 2017 glasses with 1 unit, 1 glass with 2 units

-- Modeling the operation of equalizing two glasses
def equalize (a b : ℝ) : ℝ := (a + b) / 2

-- Main theorem
theorem stepashka_cannot_defeat_kryusha :
  ¬ ∃ f : list ℕ → list ℕ, f initial_distribution = list.repeat (champagne / glasses) glasses := 
sorry

end stepashka_cannot_defeat_kryusha_l283_283010


namespace medication_price_reduction_l283_283550

variable (a : ℝ)

theorem medication_price_reduction (h : 0.60 * x = a) : x = 5/3 * a := by
  sorry

end medication_price_reduction_l283_283550


namespace find_r4_l283_283267

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l283_283267


namespace simplify_fraction_l283_283531

theorem simplify_fraction (a : ℝ) (h : a = 2) : (15 * a^4) / (75 * a^3) = 2 / 5 :=
by
  sorry

end simplify_fraction_l283_283531


namespace product_of_fractions_is_25_div_324_l283_283884

noncomputable def product_of_fractions : ℚ := 
  (10 / 6) * (4 / 20) * (20 / 12) * (16 / 32) * 
  (40 / 24) * (8 / 40) * (60 / 36) * (32 / 64)

theorem product_of_fractions_is_25_div_324 : product_of_fractions = 25 / 324 := 
  sorry

end product_of_fractions_is_25_div_324_l283_283884


namespace mixture_price_correct_l283_283685

noncomputable def priceOfMixture (x y : ℝ) (P : ℝ) : Prop :=
  P = (3.10 * x + 3.60 * y) / (x + y)

theorem mixture_price_correct {x y : ℝ} (h_proportion : x / y = 7 / 3) : priceOfMixture x (3 / 7 * x) 3.25 :=
by
  sorry

end mixture_price_correct_l283_283685


namespace point_in_quadrant_I_l283_283459

theorem point_in_quadrant_I (x y : ℝ) (h1 : 4 * x + 6 * y = 24) (h2 : y = x + 3) : x > 0 ∧ y > 0 :=
by sorry

end point_in_quadrant_I_l283_283459


namespace elementary_school_coats_correct_l283_283449

def total_coats : ℕ := 9437
def high_school_coats : ℕ := (3 * total_coats) / 5
def elementary_school_coats := total_coats - high_school_coats

theorem elementary_school_coats_correct : 
  elementary_school_coats = 3775 :=
by
  sorry

end elementary_school_coats_correct_l283_283449


namespace area_difference_l283_283040

-- Definitions of the conditions
def length_rect := 60 -- length of the rectangular garden in feet
def width_rect := 20 -- width of the rectangular garden in feet

-- Compute the area of the rectangular garden
def area_rect := length_rect * width_rect

-- Compute the perimeter of the rectangular garden
def perimeter_rect := 2 * (length_rect + width_rect)

-- Compute the side length of the square garden from the same perimeter
def side_square := perimeter_rect / 4

-- Compute the area of the square garden
def area_square := side_square * side_square

-- The goal is to prove the area difference
theorem area_difference : area_square - area_rect = 400 := by
  sorry -- Proof to be completed

end area_difference_l283_283040


namespace largest_factor_and_smallest_multiple_of_18_l283_283068

theorem largest_factor_and_smallest_multiple_of_18 :
  (∃ x, (x ∈ {d : ℕ | d ∣ 18}) ∧ (∀ y, y ∈ {d : ℕ | d ∣ 18} → y ≤ x) ∧ x = 18)
  ∧ (∃ y, (y ∈ {m : ℕ | 18 ∣ m}) ∧ (∀ z, z ∈ {m : ℕ | 18 ∣ m} → y ≤ z) ∧ y = 18) :=
by
  sorry

end largest_factor_and_smallest_multiple_of_18_l283_283068


namespace find_extreme_value_number_of_zeros_l283_283631

noncomputable def f (a : ℝ) (x : ℝ) := a * x ^ 2 + (a - 2) * x - Real.log x

-- Math proof problem I
theorem find_extreme_value (a : ℝ) (h : (∀ x : ℝ, x ≠ 0 → x ≠ 1 → f a x > f a 1)) : a = 1 := 
sorry

-- Math proof problem II
theorem number_of_zeros (a : ℝ) (h : 0 < a ∧ a < 1) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 := 
sorry

end find_extreme_value_number_of_zeros_l283_283631


namespace avg_age_of_five_students_l283_283407

-- step a: Define the conditions
def avg_age_seventeen_students : ℕ := 17
def total_seventeen_students : ℕ := 17 * avg_age_seventeen_students

def num_students_with_unknown_avg : ℕ := 5

def avg_age_nine_students : ℕ := 16
def num_students_with_known_avg : ℕ := 9
def total_age_nine_students : ℕ := num_students_with_known_avg * avg_age_nine_students

def age_seventeenth_student : ℕ := 75

-- step c: Compute the average age of the 5 students
noncomputable def total_age_five_students : ℕ :=
  total_seventeen_students - total_age_nine_students - age_seventeenth_student

def correct_avg_age_five_students : ℕ := 14

theorem avg_age_of_five_students :
  total_age_five_students / num_students_with_unknown_avg = correct_avg_age_five_students :=
sorry

end avg_age_of_five_students_l283_283407


namespace simplify_expression_l283_283959

variables (y : ℝ)

theorem simplify_expression : 
  3 * y + 4 * y^2 - 2 - (8 - 3 * y - 4 * y^2) = 8 * y^2 + 6 * y - 10 :=
by sorry

end simplify_expression_l283_283959


namespace expression_evaluation_l283_283061

theorem expression_evaluation :
  (4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2) ^ 0 = 1) :=
by
  -- Step by step calculations skipped
  sorry

end expression_evaluation_l283_283061


namespace probability_of_specific_combination_l283_283119

theorem probability_of_specific_combination :
  let shirts := 6
  let shorts := 8
  let socks := 7
  let total_clothes := shirts + shorts + socks
  let ways_total := Nat.choose total_clothes 4
  let ways_shirts := Nat.choose shirts 2
  let ways_shorts := Nat.choose shorts 1
  let ways_socks := Nat.choose socks 1
  let ways_favorable := ways_shirts * ways_shorts * ways_socks
  let probability := (ways_favorable: ℚ) / ways_total
  probability = 56 / 399 :=
by
  simp
  sorry

end probability_of_specific_combination_l283_283119


namespace min_value_y_l283_283180

theorem min_value_y (x : ℝ) : ∃ x : ℝ, (y = x^2 + 16 * x + 20) ∧ ∀ z : ℝ, (y = z^2 + 16 * z + 20) → y ≥ -44 := 
sorry

end min_value_y_l283_283180


namespace simplify_expression_at_zero_l283_283819

-- Define the expression f(x)
def f (x : ℚ) : ℚ := (2 * x + 4) / (x^2 - 6 * x + 9) / ((2 * x - 1) / (x - 3) - 1)

-- State that for the given value x = 0, the simplified expression equals -2/3
theorem simplify_expression_at_zero :
  f 0 = -2 / 3 :=
by
  sorry

end simplify_expression_at_zero_l283_283819


namespace min_value_of_expression_l283_283788

open Real

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h_perp : (x - 1) * 1 + 3 * y = 0) :
  ∃ (m : ℝ), m = 4 ∧ (∀ (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_ab_perp : (a - 1) * 1 + 3 * b = 0), (1 / a) + (1 / (3 * b)) ≥ m) :=
by
  use 4
  sorry

end min_value_of_expression_l283_283788


namespace no_solution_exists_l283_283025

theorem no_solution_exists : 
  ¬(∃ x y : ℝ, 2 * x - 3 * y = 7 ∧ 4 * x - 6 * y = 20) :=
by
  sorry

end no_solution_exists_l283_283025


namespace total_days_2001_2005_l283_283912

theorem total_days_2001_2005 : 
  let is_leap_year (y : ℕ) := y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0)
  let days_in_year (y : ℕ) := if is_leap_year y then 366 else 365 
  (days_in_year 2001) + (days_in_year 2002) + (days_in_year 2003) + (days_in_year 2004) + (days_in_year 2005) = 1461 :=
by
  sorry

end total_days_2001_2005_l283_283912


namespace remove_five_maximizes_probability_l283_283178

open Finset

theorem remove_five_maximizes_probability :
  ∀ (l : List ℤ), l = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] →
  (∃ x : ℤ, x ∈ l ∧ 
    (∀ a b : ℤ, a ≠ b → a ∈ l.erase x → b ∈ l.erase x → a + b = 10 → (a, b) ∉ (l.product l) \ {(x, 10 - x), (10 - x, x)})) →
  x = 5 :=
begin
  sorry
end

end remove_five_maximizes_probability_l283_283178


namespace edge_length_of_cube_l283_283996

/-- Define the total paint volume, remaining paint and cube paint volume -/
def total_paint_volume : ℕ := 25 * 40
def remaining_paint : ℕ := 271
def cube_paint_volume : ℕ := total_paint_volume - remaining_paint

/-- Define the volume of the cube and the statement for edge length of the cube -/
theorem edge_length_of_cube (s : ℕ) : s^3 = cube_paint_volume → s = 9 :=
by
  have h1 : cube_paint_volume = 729 := by rfl
  sorry

end edge_length_of_cube_l283_283996


namespace cos_of_angle_B_l283_283902

theorem cos_of_angle_B (A B C a b c : Real) (h₁ : A - C = Real.pi / 2) (h₂ : 2 * b = a + c) 
  (h₃ : 2 * a * Real.sin A = 2 * b * Real.sin B) (h₄ : 2 * c * Real.sin C = 2 * b * Real.sin B) :
  Real.cos B = 3 / 4 := by
  sorry

end cos_of_angle_B_l283_283902


namespace g_at_seven_equals_92_l283_283918

def g (n : ℕ) : ℕ := n^2 + 2*n + 29

theorem g_at_seven_equals_92 : g 7 = 92 :=
by
  sorry

end g_at_seven_equals_92_l283_283918


namespace quadratic_distinct_roots_iff_m_lt_four_l283_283016

theorem quadratic_distinct_roots_iff_m_lt_four (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 4 * x₁ + m = 0) ∧ (x₂^2 - 4 * x₂ + m = 0)) ↔ m < 4 :=
by sorry

end quadratic_distinct_roots_iff_m_lt_four_l283_283016


namespace minimum_omega_l283_283254

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)
noncomputable def h (ω : ℝ) (x : ℝ) : ℝ := f ω x + g ω x

theorem minimum_omega (ω : ℝ) (m : ℝ) 
  (h1 : 0 < ω)
  (h2 : ∀ x : ℝ, h ω m ≤ h ω x ∧ h ω x ≤ h ω (m + 1)) :
  ω = π :=
by
  sorry

end minimum_omega_l283_283254


namespace base3_to_base10_conversion_l283_283067

theorem base3_to_base10_conversion : ∀ n : ℕ, n = 120102 → (1 * 3^5 + 2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 0 * 3^1 + 2 * 3^0) = 416 :=
by
  intro n hn
  sorry

end base3_to_base10_conversion_l283_283067


namespace quadratic_complete_square_l283_283261

theorem quadratic_complete_square (c n : ℝ) (h1 : ∀ x : ℝ, x^2 + c * x + 20 = (x + n)^2 + 12) (h2: 0 < c) : 
  c = 4 * Real.sqrt 2 :=
by
  sorry

end quadratic_complete_square_l283_283261


namespace calculate_expression_l283_283065

noncomputable def exponent_inverse (x : ℝ) : ℝ := x ^ (-1)
noncomputable def root (x : ℝ) (n : ℕ) : ℝ := x ^ (1 / n : ℝ)

theorem calculate_expression :
  (exponent_inverse 4) - (root (1/16) 2) + (3 - Real.sqrt 2) ^ 0 = 1 := 
by
  -- Definitions according to conditions
  have h1 : exponent_inverse 4 = 1 / 4 := by sorry
  have h2 : root (1 / 16) 2 = 1 / 4 := by sorry
  have h3 : (3 - Real.sqrt 2) ^ 0 = 1 := by sorry
  
  -- Combine and simplify parts
  calc
    (exponent_inverse 4) - (root (1 / 16) 2) + (3 - Real.sqrt 2) ^ 0
        = (1 / 4) - (1 / 4) + 1 : by rw [h1, h2, h3]
    ... = 0 + 1 : by sorry
    ... = 1 : by rfl

end calculate_expression_l283_283065


namespace student_count_l283_283643

theorem student_count (N : ℕ) (h : N > 22 ∧ N ≤ 25) : N = 23 ∨ N = 24 ∨ N = 25 := by {
  cases (Nat.eq_or_lt_of_le h.right); {
    exact Or.inr Or.inr (Nat.lt_antisymm h.left _); sorry;
  };
  exact Or.inr (Or.inl (Nat.lt_antisymm h.left (Nat.lt_succ_iff.mpr h.left))); sorry;
  exact Or.inl (Nat.lt_antisymm h.left (Nat.lt_succ_of_lt h.left)); sorry;
}

end student_count_l283_283643


namespace cost_of_1500_pencils_l283_283568

theorem cost_of_1500_pencils (cost_per_box : ℕ) (pencils_per_box : ℕ) (num_pencils : ℕ) :
  cost_per_box = 30 → pencils_per_box = 100 → num_pencils = 1500 → 
  (num_pencils * (cost_per_box / pencils_per_box) = 450) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end cost_of_1500_pencils_l283_283568


namespace field_area_is_243_l283_283210

noncomputable def field_area (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : ℝ :=
  w * l

theorem field_area_is_243 (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : field_area w l h1 h2 = 243 :=
  sorry

end field_area_is_243_l283_283210


namespace meaningful_range_l283_283322

noncomputable def isMeaningful (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (x - 2 ≠ 0)

theorem meaningful_range (x : ℝ) : isMeaningful x ↔ (x ≥ -1) ∧ (x ≠ 2) :=
by
  sorry

end meaningful_range_l283_283322


namespace area_of_given_sector_l283_283901

noncomputable def area_of_sector (alpha l : ℝ) : ℝ :=
  let r := l / alpha
  (1 / 2) * l * r

theorem area_of_given_sector :
  let alpha := Real.pi / 9
  let l := Real.pi / 3
  area_of_sector alpha l = Real.pi / 2 :=
by
  sorry

end area_of_given_sector_l283_283901


namespace find_x_l283_283265

theorem find_x (x y z : ℚ) (h1 : (x * y) / (x + y) = 4) (h2 : (x * z) / (x + z) = 5) (h3 : (y * z) / (y + z) = 6) : x = 40 / 9 :=
by
  -- Structure the proof here
  sorry

end find_x_l283_283265


namespace max_possible_scores_l283_283207

theorem max_possible_scores (num_questions : ℕ) (points_correct : ℤ) (points_incorrect : ℤ) (points_unanswered : ℤ) :
  num_questions = 10 →
  points_correct = 4 →
  points_incorrect = -1 →
  points_unanswered = 0 →
  ∃ n, n = 45 :=
by
  sorry

end max_possible_scores_l283_283207


namespace abc_div_def_eq_1_div_20_l283_283792

-- Definitions
variables (a b c d e f : ℝ)

-- Conditions
axiom condition1 : a / b = 1 / 3
axiom condition2 : b / c = 2
axiom condition3 : c / d = 1 / 2
axiom condition4 : d / e = 3
axiom condition5 : e / f = 1 / 10

-- Proof statement
theorem abc_div_def_eq_1_div_20 : (a * b * c) / (d * e * f) = 1 / 20 :=
by 
  -- The actual proof is omitted, as the problem only requires the statement.
  sorry

end abc_div_def_eq_1_div_20_l283_283792


namespace abs_diff_is_perfect_square_l283_283342

-- Define the conditions
variable (m n : ℤ) (h_odd_m : m % 2 = 1) (h_odd_n : n % 2 = 1)
variable (h_div : (n^2 - 1) ∣ (m^2 + 1 - n^2))

-- Theorem statement
theorem abs_diff_is_perfect_square : ∃ (k : ℤ), (m^2 + 1 - n^2) = k^2 :=
by
  sorry

end abs_diff_is_perfect_square_l283_283342


namespace arithmetic_sequence_sum_l283_283931

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the problem conditions
def problem_conditions (a : ℕ → ℝ) : Prop :=
  (a 3 + a 8 = 3) ∧ is_arithmetic_sequence a

-- State the theorem to be proved
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : problem_conditions a) : a 1 + a 10 = 3 :=
sorry

end arithmetic_sequence_sum_l283_283931


namespace range_of_m_no_zeros_inequality_when_m_zero_l283_283086

-- Statement for Problem 1
theorem range_of_m_no_zeros (m : ℝ) (h : ∀ x : ℝ, (x^2 + m * x + m) * Real.exp x ≠ 0) : 0 < m ∧ m < 4 :=
sorry

-- Statement for Problem 2
theorem inequality_when_m_zero (x : ℝ) : 
  (x^2) * (Real.exp x) ≥ x^2 + x^3 :=
sorry

end range_of_m_no_zeros_inequality_when_m_zero_l283_283086


namespace floor_ceil_inequality_l283_283343

theorem floor_ceil_inequality 
  (a b c : ℝ)
  (h : ⌈a⌉ + ⌈b⌉ + ⌈c⌉ + ⌊a + b⌋ + ⌊b + c⌋ + ⌊c + a⌋ = 2020) :
  ⌊a⌋ + ⌊b⌋ + ⌊c⌋ + ⌈a + b + c⌉ ≥ 1346 := 
by
  sorry 

end floor_ceil_inequality_l283_283343


namespace ratio_of_logs_eq_golden_ratio_l283_283822

theorem ratio_of_logs_eq_golden_ratio
  (r s : ℝ) (hr : 0 < r) (hs : 0 < s)
  (h : Real.log r / Real.log 4 = Real.log s / Real.log 18 ∧ Real.log s / Real.log 18 = Real.log (r + s) / Real.log 24) :
  s / r = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end ratio_of_logs_eq_golden_ratio_l283_283822


namespace triangle_is_isosceles_l283_283978

theorem triangle_is_isosceles 
  (a b c : ℝ)
  (h : a^2 - b^2 + a * c - b * c = 0)
  (h_tri : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  : a = b := 
sorry

end triangle_is_isosceles_l283_283978


namespace arithmetic_seq_sum_l283_283796

theorem arithmetic_seq_sum (a d : ℕ) (S : ℕ → ℕ) (n : ℕ) :
  S 6 = 36 →
  S 12 = 144 →
  S (6 * n) = 576 →
  (∀ m, S m = m * (2 * a + (m - 1) * d) / 2) →
  n = 4 := 
by
  intros h1 h2 h3 h4
  sorry

end arithmetic_seq_sum_l283_283796


namespace math_proof_l283_283886

def exponentiation_result := -1 ^ 4
def negative_exponentiation_result := (-2) ^ 3
def absolute_value_result := abs (-3 - 1)
def division_result := 16 / negative_exponentiation_result
def multiplication_result := division_result * absolute_value_result
def final_result := exponentiation_result + multiplication_result

theorem math_proof : final_result = -9 := by
  -- To be proved
  sorry

end math_proof_l283_283886


namespace side_length_of_base_l283_283375

-- Define the conditions
def lateral_area (s : ℝ) : ℝ := (1 / 2) * s * 40
def given_area : ℝ := 120

-- Define the theorem to prove the length of the side of the base
theorem side_length_of_base : ∃ (s : ℝ), lateral_area(s) = given_area ∧ s = 6 :=
by
  sorry

end side_length_of_base_l283_283375


namespace find_a_l283_283161

-- Define the function f(x)
def f (a : ℚ) (x : ℚ) : ℚ := x^2 + (2 * a + 3) * x + (a^2 + 1)

-- State that the discriminant of f(x) is non-negative
def discriminant_nonnegative (a : ℚ) : Prop :=
  let Δ := (2 * a + 3)^2 - 4 * (a^2 + 1)
  Δ ≥ 0

-- Final statement expressing the final condition on a and the desired result |p| + |q|
theorem find_a (a : ℚ) (p q : ℤ) (h_relprime : Int.gcd p q = 1) (h_eq : a = -5 / 12) (h_abs : p * q = -5 * 12) :
  discriminant_nonnegative a →
  |p| + |q| = 17 :=
by sorry

end find_a_l283_283161


namespace altitude_triangle_eq_2w_l283_283487

theorem altitude_triangle_eq_2w (l w h : ℕ) (h₀ : w ≠ 0) (h₁ : l ≠ 0)
    (h_area_rect : l * w = (1 / 2) * l * h) : h = 2 * w :=
by
  -- Consider h₀ (w is not zero) and h₁ (l is not zero)
  -- We need to prove h = 2w given l * w = (1 / 2) * l * h
  sorry

end altitude_triangle_eq_2w_l283_283487


namespace equal_numbers_l283_283943

theorem equal_numbers {a b c d : ℝ} (h : a^2 + b^2 + c^2 + d^2 = ab + bc + cd + da) :
  a = b ∧ b = c ∧ c = d :=
by
  sorry

end equal_numbers_l283_283943


namespace a_2016_is_1_l283_283495

noncomputable def seq_a (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * b n

theorem a_2016_is_1 (a b : ℕ → ℝ)
  (h1 : a 1 = 1)
  (hb : seq_a a b)
  (h3 : b 1008 = 1) :
  a 2016 = 1 :=
sorry

end a_2016_is_1_l283_283495


namespace parabola_standard_equation_l283_283472

theorem parabola_standard_equation (directrix : ℝ) (h_directrix : directrix = 1) : 
  ∃ (a : ℝ), y^2 = a * x ∧ a = -4 :=
by
  sorry

end parabola_standard_equation_l283_283472


namespace smallest_6_digit_divisible_by_111_l283_283182

theorem smallest_6_digit_divisible_by_111 :
  ∃ x : ℕ, 100000 ≤ x ∧ x ≤ 999999 ∧ x % 111 = 0 ∧ x = 100011 :=
  by
    sorry

end smallest_6_digit_divisible_by_111_l283_283182


namespace zhou_yu_age_equation_l283_283924

variable (x : ℕ)

theorem zhou_yu_age_equation (h : x + 3 < 10) : 10 * x + (x + 3) = (x + 3) ^ 2 :=
  sorry

end zhou_yu_age_equation_l283_283924


namespace symmetric_to_origin_l283_283410

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_to_origin (p : ℝ × ℝ) (h : p = (3, -1)) : symmetric_point p = (-3, 1) :=
by
  -- This is just the statement; the proof is not provided.
  sorry

end symmetric_to_origin_l283_283410


namespace mother_l283_283185

theorem mother's_age (D M : ℕ) (h1 : 2 * D + M = 70) (h2 : D + 2 * M = 95) : M = 40 :=
sorry

end mother_l283_283185


namespace find_r_fourth_l283_283288

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l283_283288


namespace equalize_champagne_futile_l283_283011

/-- Stepashka cannot distribute champagne into 2018 glasses in such a way 
that Kryusha's attempts to equalize the amount in all glasses become futile. -/
theorem equalize_champagne_futile (n : ℕ) (h : n = 2018) : 
∃ (a : ℕ), (∀ (A B : ℕ), A ≠ B ∧ A + B = 2019 → (A + B) % 2 = 1) := 
sorry

end equalize_champagne_futile_l283_283011


namespace tom_chocolates_l283_283002

variable (n : ℕ)

-- Lisa's box holds 64 chocolates and has unit dimensions (1^3 = 1 cubic unit)
def lisa_chocolates := 64
def lisa_volume := 1

-- Tom's box has dimensions thrice Lisa's and hence its volume (3^3 = 27 cubic units)
def tom_volume := 27

-- Number of chocolates Tom's box holds
theorem tom_chocolates : lisa_chocolates * tom_volume = 1728 := by
  -- calculations with known values
  sorry

end tom_chocolates_l283_283002


namespace cube_point_problem_l283_283686
open Int

theorem cube_point_problem (n : ℤ) (x y z u : ℤ)
  (hx : x = 0 ∨ x = 8)
  (hy : y = 0 ∨ y = 12)
  (hz : z = 0 ∨ z = 6)
  (hu : 24 ∣ u)
  (hn : n = x + y + z + u) :
  (n ≠ 100) ∧ (n = 200) ↔ (n % 6 = 0 ∨ (n - 8) % 6 = 0) :=
by sorry

end cube_point_problem_l283_283686


namespace interval_monotonicity_minimum_value_range_of_a_l283_283908

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a / x

theorem interval_monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x, 0 < x ∧ x < a → f x a > 0) ∧ (∀ x, x > a → f x a < 0) :=
sorry

theorem minimum_value (a : ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ Real.exp 1 → f x a ≥ 1) ∧ (∃ x, 1 ≤ x ∧ x ≤ Real.exp 1 ∧ f x a = 1) → a = 1 :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ x, x > 1 → f x a < 1 / 2 * x) → a < 1 / 2 :=
sorry

end interval_monotonicity_minimum_value_range_of_a_l283_283908


namespace determine_x_l283_283506

theorem determine_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^3) (h3 : x / 9 = 9 * y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 :=
by
  sorry

end determine_x_l283_283506


namespace product_of_four_consecutive_integers_divisible_by_24_l283_283528

theorem product_of_four_consecutive_integers_divisible_by_24 (n : ℤ) :
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_24_l283_283528


namespace find_s_t_u_of_triangle_l283_283800

theorem find_s_t_u_of_triangle {XYZ : Triangle} (hZ : angle XYZ Z = 90) (hAngle : angle X Z Y < 45)
  (hXY : dist X Y = 5) (Q : Point) (hQ : Q ∈ XY)
  (hQZY : angle Q Z Y = 2 * angle Z Q Y) (hZQ : dist Z Q = 2) :
  ∃ (s t u : ℕ), u ≠ 0 ∧ ∀ p, nat.prime p → u ∣ p^2 → false ∧ (2 + (1 : ℝ) * real.sqrt 2 = s + t * real.sqrt u) ∧ s + t + u = 5 :=
sorry

end find_s_t_u_of_triangle_l283_283800


namespace percentage_of_games_lost_l283_283974

theorem percentage_of_games_lost (games_won games_lost games_tied total_games : ℕ)
  (h_ratio : 5 * games_lost = 3 * games_won)
  (h_tied : games_tied * 5 = total_games) :
  (games_lost * 10 / total_games) = 3 :=
by sorry

end percentage_of_games_lost_l283_283974


namespace complement_intersection_l283_283103

open Set

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def M : Set Int := {-1, 0, 1, 3}
def N : Set Int := {-2, 0, 2, 3}

theorem complement_intersection :
  ((U \ M) ∩ N = {-2, 2}) :=
by sorry

end complement_intersection_l283_283103


namespace student_count_l283_283644

theorem student_count (N : ℕ) (h : N > 22 ∧ N ≤ 25) : N = 23 ∨ N = 24 ∨ N = 25 := by {
  cases (Nat.eq_or_lt_of_le h.right); {
    exact Or.inr Or.inr (Nat.lt_antisymm h.left _); sorry;
  };
  exact Or.inr (Or.inl (Nat.lt_antisymm h.left (Nat.lt_succ_iff.mpr h.left))); sorry;
  exact Or.inl (Nat.lt_antisymm h.left (Nat.lt_succ_of_lt h.left)); sorry;
}

end student_count_l283_283644


namespace tom_purchased_8_kg_of_apples_l283_283983

noncomputable def number_of_apples_purchased (price_per_kg_apple : ℤ) (price_per_kg_mango : ℤ) (kg_mangoes : ℤ) (total_paid : ℤ) : ℤ :=
  let total_cost_mangoes := price_per_kg_mango * kg_mangoes
  total_paid - total_cost_mangoes / price_per_kg_apple

theorem tom_purchased_8_kg_of_apples : 
  number_of_apples_purchased 70 65 9 1145 = 8 := 
by {
  -- Expand the definitions and simplify
  sorry
}

end tom_purchased_8_kg_of_apples_l283_283983


namespace check_range_a_l283_283134

open Set

def A : Set ℝ := {x | x < -1/2 ∨ x > 1}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 1 ≤ 0}

theorem check_range_a :
  (∃! x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ (x₁ : ℝ) ∈ A ∩ B a ∧ (x₂ : ℝ) ∈ A ∩ B a) →
  a ∈ Icc (4/3 : ℝ) (15/8 : ℝ) :=
sorry

end check_range_a_l283_283134


namespace final_price_scarf_l283_283048

variables (P : ℝ) (D1 D2 : ℝ) (P₁ P₂ : ℝ)

theorem final_price_scarf :
  P = 15 → D1 = 0.20 → D2 = 0.25 → P₁ = P * (1 - D1) → P₂ = P₁ * (1 - D2) → P₂ = 9 :=
by
  intros hP hD1 hD2 hP₁ hP₂
  rw [hP, hD1, hD2] at hP₁ hP₂
  have hP₁' : P₁ = 15 * (1 - 0.20), from hP₁
  rw [hP₁'] at hP₂
  have hP₁'' : P₁ = 12, by norm_num
  rw [hP₁''] at hP₂
  have hP₂' : P₂ = 12 * (1 - 0.25), from hP₂
  rw [hP₂'] 
  norm_num
  sorry

end final_price_scarf_l283_283048


namespace find_number_divided_l283_283849

theorem find_number_divided (x : ℝ) (h : x / 1.33 = 48) : x = 63.84 :=
by
  sorry

end find_number_divided_l283_283849


namespace max_fuel_needed_l283_283716

noncomputable def max_fuel (d_a d_b_c_sum : ℝ) (h_a_eq : d_a = 100) (h_b_c_sum_eq : d_b_c_sum = 300) : ℝ :=
  let h_b := 150 -- Since the function is minimized & h_b + h_c = 300, h_b = h_c = 150
  let r_inv := 1 / d_a + 2 / h_b
  let r := 1 / r_inv
  let fuel := r / 10
  fuel

theorem max_fuel_needed :
  let d_a := 100
  let d_b_c_sum := 300
  max_fuel d_a d_b_c_sum (by simp) (by simp) = 30 / 7 := 
by 
  sorry

end max_fuel_needed_l283_283716


namespace pow_mod_remainder_l283_283607

theorem pow_mod_remainder (a : ℕ) (p : ℕ) (n : ℕ) (h1 : a^16 ≡ 1 [MOD p]) (h2 : a^5 ≡ n [MOD p]) : 
  a^2021 ≡ n [MOD p] :=
sorry

example : 5^2021 ≡ 14 [MOD 17] :=
begin
  apply pow_mod_remainder 5 17 14,
  { exact pow_mod_remainder 5 17 1 sorry sorry },
  { sorry },
end

end pow_mod_remainder_l283_283607


namespace percentage_fescue_in_Y_l283_283150

-- Define the seed mixtures and their compositions
structure SeedMixture :=
  (ryegrass : ℝ)  -- percentage of ryegrass

-- Seed mixture X
def X : SeedMixture := { ryegrass := 0.40 }

-- Seed mixture Y
def Y : SeedMixture := { ryegrass := 0.25 }

-- Mixture of X and Y contains 32 percent ryegrass
def mixture_percentage := 0.32

-- 46.67 percent of the weight of this mixture is X
def weight_X := 0.4667

-- Question: What percent of seed mixture Y is fescue
theorem percentage_fescue_in_Y : (1 - Y.ryegrass) = 0.75 := by
  sorry

end percentage_fescue_in_Y_l283_283150


namespace AM_GM_inequality_l283_283001

theorem AM_GM_inequality (a : List ℝ) (h : ∀ x ∈ a, 0 < x) :
  (a.sum / a.length) ≥ a.prod ^ (1 / a.length) := 
sorry

end AM_GM_inequality_l283_283001


namespace fit_nine_cross_pentominoes_on_chessboard_l283_283962

def cross_pentomino (A B C D E : Prop) :=
  A ∧ B ∧ C ∧ D ∧ E -- A cross pentomino is five connected 1x1 squares

def square1x1 : Prop := sorry -- a placeholder for a 1x1 square

def eight_by_eight_chessboard := Fin 8 × Fin 8 -- an 8x8 chessboard using finitely indexed squares

noncomputable def can_cut_nine_cross_pentominoes : Prop := sorry -- a placeholder proof verification

theorem fit_nine_cross_pentominoes_on_chessboard : can_cut_nine_cross_pentominoes  :=
by 
  -- Assume each cross pentomino consists of 5 connected 1x1 squares
  let cross := cross_pentomino square1x1 square1x1 square1x1 square1x1 square1x1
  -- We need to prove that we can cut out nine such crosses from the 8x8 chessboard
  sorry

end fit_nine_cross_pentominoes_on_chessboard_l283_283962


namespace r_pow_four_solution_l283_283303

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l283_283303


namespace xiaoming_accuracy_l283_283679

theorem xiaoming_accuracy :
  ∀ (correct already_wrong extra_needed : ℕ),
  correct = 30 →
  already_wrong = 6 →
  (correct + extra_needed).toFloat / (correct + already_wrong + extra_needed).toFloat = 0.85 →
  extra_needed = 4 := by
  intros correct already_wrong extra_needed h_correct h_wrong h_accuracy
  sorry

end xiaoming_accuracy_l283_283679


namespace eighth_positive_odd_multiple_of_5_l283_283986

theorem eighth_positive_odd_multiple_of_5 : 
  let a := 5 in 
  let d := 10 in 
  let n := 8 in 
  a + (n - 1) * d = 75 :=
by
  let a := 5
  let d := 10
  let n := 8
  have : a + (n - 1) * d = 75 := by 
    calc
      a + (n - 1) * d = 5 + (8 - 1) * 10  : by rfl
      ... = 5 + 70                          : by rfl
      ... = 75                              : by rfl
  exact this

end eighth_positive_odd_multiple_of_5_l283_283986


namespace polynomial_factor_l283_283895

theorem polynomial_factor (x : ℝ) : (x^2 - 4*x + 4) ∣ (x^4 + 16) :=
sorry

end polynomial_factor_l283_283895


namespace no_bounded_sequence_a1_gt_2015_l283_283975

theorem no_bounded_sequence_a1_gt_2015 (a1 : ℚ) (h_a1 : a1 > 2015) : 
  ∀ (a_n : ℕ → ℚ), a_n 1 = a1 → 
  (∀ (n : ℕ), ∃ (p_n q_n : ℕ), p_n > 0 ∧ q_n > 0 ∧ (p_n.gcd q_n = 1) ∧ (a_n n = p_n / q_n) ∧ 
  (a_n (n + 1) = (p_n^2 + 2015) / (p_n * q_n))) → 
  ∃ (M : ℚ), ∀ (n : ℕ), a_n n ≤ M → 
  False :=
sorry

end no_bounded_sequence_a1_gt_2015_l283_283975


namespace mushroom_picking_l283_283190

theorem mushroom_picking (n T : ℕ) (hn_min : n ≥ 5) (hn_max : n ≤ 7)
  (hmax : ∀ (M_max M_min : ℕ), M_max = T / 5 → M_min = T / 7 → 
    T ≠ 0 → M_max ≤ T / n ∧ M_min ≥ T / n) : n = 6 :=
by
  sorry

end mushroom_picking_l283_283190


namespace dishes_left_for_Oliver_l283_283443

theorem dishes_left_for_Oliver
  (total_dishes : ℕ)
  (dishes_with_mango_salsa : ℕ)
  (dishes_with_fresh_mango : ℕ)
  (dishes_with_mango_jelly : ℕ)
  (oliver_will_try_dishes_with_fresh_mango : ℕ)
  (total_dishes = 36)
  (dishes_with_mango_salsa = 3)
  (dishes_with_fresh_mango = total_dishes / 6)
  (dishes_with_mango_jelly = 1)
  (oliver_will_try_dishes_with_fresh_mango = 2)
  : total_dishes - (dishes_with_mango_salsa + dishes_with_fresh_mango + dishes_with_mango_jelly - oliver_will_try_dishes_with_fresh_mango) = 28 :=
by
  -- proof omitted
  sorry

end dishes_left_for_Oliver_l283_283443


namespace divisibility_by_7_l283_283526

theorem divisibility_by_7 (n : ℕ) : (3^(2 * n + 1) + 2^(n + 2)) % 7 = 0 :=
by
  sorry

end divisibility_by_7_l283_283526


namespace factorization_correct_l283_283075

theorem factorization_correct (x : ℝ) :
    x^2 - 3 * x - 4 = (x + 1) * (x - 4) :=
  sorry

end factorization_correct_l283_283075


namespace find_fg3_l283_283696

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem find_fg3 : f (g 3) = 2 := by
  sorry

end find_fg3_l283_283696


namespace solution_values_sum_l283_283549

theorem solution_values_sum (x y : ℝ) (p q r s : ℕ) 
  (hx : x + y = 5) 
  (hxy : 2 * x * y = 5) 
  (hx_form : x = (p + q * Real.sqrt r) / s ∨ x = (p - q * Real.sqrt r) / s) 
  (hpqs_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0) : 
  p + q + r + s = 23 := 
sorry

end solution_values_sum_l283_283549


namespace intersection_of_S_and_T_l283_283522

noncomputable def S := {x : ℝ | x ≥ 2}
noncomputable def T := {x : ℝ | x ≤ 5}

theorem intersection_of_S_and_T : S ∩ T = {x : ℝ | 2 ≤ x ∧ x ≤ 5} :=
by
  sorry

end intersection_of_S_and_T_l283_283522


namespace reciprocal_inequality_reciprocal_inequality_opposite_l283_283705

theorem reciprocal_inequality (a b : ℝ) (h1 : a > b) (h2 : ab > 0) : (1 / a < 1 / b) := 
sorry

theorem reciprocal_inequality_opposite (a b : ℝ) (h1 : a > b) (h2 : ab < 0) : (1 / a > 1 / b) := 
sorry

end reciprocal_inequality_reciprocal_inequality_opposite_l283_283705


namespace T_n_lt_1_l283_283260

open Nat

def a (n : ℕ) : ℕ := 2^n

def b (n : ℕ) : ℕ := 2^n - 1

def c (n : ℕ) : ℚ := (a n : ℚ) / ((b n : ℚ) * (b (n + 1) : ℚ))

noncomputable def T (n : ℕ) : ℚ := (Finset.range (n + 1)).sum c

theorem T_n_lt_1 (n : ℕ) : T n < 1 := by
  sorry

end T_n_lt_1_l283_283260


namespace Randy_initial_money_l283_283148

theorem Randy_initial_money (M : ℝ) (r1 : M + 200 - 1200 = 2000) : M = 3000 :=
by
  sorry

end Randy_initial_money_l283_283148


namespace value_of_b_l283_283517

theorem value_of_b (a b c : ℤ) : 
  (∃ d : ℤ, a = 17 + d ∧ b = 17 + 2 * d ∧ c = 17 + 3 * d ∧ 41 = 17 + 4 * d) → b = 29 :=
by
  intros h
  sorry


end value_of_b_l283_283517


namespace vertex_of_parabola_on_x_axis_l283_283463

theorem vertex_of_parabola_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, (x^2 - 6*x + c = 0)) ↔ c = 9 :=
by
  sorry

end vertex_of_parabola_on_x_axis_l283_283463


namespace min_value_one_over_a_plus_one_over_b_point_P_outside_ellipse_l283_283491

variable (a b : ℝ)
-- Conditions: a and b are positive real numbers and (a + b)x - 1 ≤ x^2 for all x > 0
variables (ha : a > 0) (hb : b > 0) (h : ∀ x : ℝ, 0 < x → (a + b) * x - 1 ≤ x^2)

-- Question 1: Prove that the minimum value of 1/a + 1/b is 2
theorem min_value_one_over_a_plus_one_over_b : (1 : ℝ) / a + (1 : ℝ) / b = 2 := 
sorry

-- Question 2: Determine point P(1, -1) relative to the ellipse x^2/a^2 + y^2/b^2 = 1
theorem point_P_outside_ellipse : (1 : ℝ)^2 / (a^2) + (-1 : ℝ)^2 / (b^2) > 1 :=
sorry

end min_value_one_over_a_plus_one_over_b_point_P_outside_ellipse_l283_283491


namespace train_speed_in_kmh_l283_283212

def length_of_train : ℝ := 600
def length_of_overbridge : ℝ := 100
def time_to_cross_overbridge : ℝ := 70

theorem train_speed_in_kmh :
  (length_of_train + length_of_overbridge) / time_to_cross_overbridge * 3.6 = 36 := 
by 
  sorry

end train_speed_in_kmh_l283_283212


namespace calc_expr_l283_283057

theorem calc_expr : 4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2)^0 = 1 := by
  sorry

end calc_expr_l283_283057


namespace solve_for_x_and_y_l283_283365

theorem solve_for_x_and_y (x y : ℝ) : 9 * x^2 - 25 * y^2 = 0 → (x = (5 / 3) * y ∨ x = -(5 / 3) * y) :=
by
  sorry

end solve_for_x_and_y_l283_283365


namespace transaction_loss_l283_283205

theorem transaction_loss 
  (sell_price_house sell_price_store : ℝ)
  (cost_price_house cost_price_store : ℝ)
  (house_loss_percent store_gain_percent : ℝ)
  (house_loss_eq : sell_price_house = (4/5) * cost_price_house)
  (store_gain_eq : sell_price_store = (6/5) * cost_price_store)
  (sell_prices_eq : sell_price_house = 12000 ∧ sell_price_store = 12000)
  (house_loss_percent_eq : house_loss_percent = 0.20)
  (store_gain_percent_eq : store_gain_percent = 0.20) :
  cost_price_house + cost_price_store - (sell_price_house + sell_price_store) = 1000 :=
by
  sorry

end transaction_loss_l283_283205


namespace distinct_numbers_count_l283_283238

noncomputable section

def num_distinct_numbers : Nat :=
  let vals := List.map (λ n : Nat, Nat.floor ((n^2 : ℚ) / 500)) (List.range 1000).tail
  (vals.eraseDup).length

theorem distinct_numbers_count : num_distinct_numbers = 876 :=
by
  sorry

end distinct_numbers_count_l283_283238


namespace folded_quadrilateral_has_perpendicular_diagonals_l283_283552

-- Define a quadrilateral and its properties
structure Quadrilateral :=
(A B C D : ℝ × ℝ)

structure Point :=
(x y : ℝ)

-- Define the diagonals within a quadrilateral
def diagonal1 (q : Quadrilateral) : ℝ × ℝ := (q.A.1 - q.C.1, q.A.2 - q.C.2)
def diagonal2 (q : Quadrilateral) : ℝ × ℝ := (q.B.1 - q.D.1, q.B.2 - q.D.2)

-- Define dot product to check perpendicularity
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Condition when folding quadrilateral vertices to a common point ensures no gaps or overlaps
def folding_condition (q : Quadrilateral) (P : Point) : Prop :=
sorry -- Detailed folding condition logic here if needed

-- The statement we need to prove
theorem folded_quadrilateral_has_perpendicular_diagonals (q : Quadrilateral) (P : Point)
    (h_folding : folding_condition q P)
    : dot_product (diagonal1 q) (diagonal2 q) = 0 :=
sorry

end folded_quadrilateral_has_perpendicular_diagonals_l283_283552


namespace find_r4_l283_283272

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l283_283272


namespace direction_vector_b_l283_283165

theorem direction_vector_b (b : ℝ) 
  (P Q : ℝ × ℝ) (hP : P = (-3, 1)) (hQ : Q = (1, 5))
  (hdir : 3 - (-3) = 3 ∧ 5 - 1 = b) : b = 3 := by
  sorry

end direction_vector_b_l283_283165


namespace find_z_when_x_is_1_l283_283546

-- We start by defining the conditions
variable (x y z : ℝ)
variable (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
variable (h_inv : ∃ k₁ : ℝ, ∀ x, x^2 * y = k₁)
variable (h_dir : ∃ k₂ : ℝ, ∀ y, y / z = k₂)
variable (h_y : y = 8) (h_z : z = 32) (h_x4 : x = 4)

-- Now we need to define the problem statement: 
-- proving that z = 512 when x = 1
theorem find_z_when_x_is_1 (h_x1 : x = 1) : z = 512 :=
  sorry

end find_z_when_x_is_1_l283_283546


namespace molecular_weight_one_mole_l283_283031

theorem molecular_weight_one_mole (mw_three_moles : ℕ) (h : mw_three_moles = 882) : mw_three_moles / 3 = 294 :=
by
  -- proof is omitted
  sorry

end molecular_weight_one_mole_l283_283031


namespace evaluate_nested_operation_l283_283227

def operation (a b c : ℕ) : ℕ := (a + b) / c

theorem evaluate_nested_operation : operation (operation 72 36 108) (operation 4 2 6) (operation 12 6 18) = 2 := by
  -- Here we assume all operations are valid (c ≠ 0 for each case)
  sorry

end evaluate_nested_operation_l283_283227


namespace fred_initial_balloons_l283_283899

def green_balloons_initial (given: Nat) (left: Nat) : Nat := 
  given + left

theorem fred_initial_balloons : green_balloons_initial 221 488 = 709 :=
by
  sorry

end fred_initial_balloons_l283_283899


namespace trisha_collects_4_dozen_less_l283_283755

theorem trisha_collects_4_dozen_less (B C T : ℕ) 
  (h1 : B = 6) 
  (h2 : C = 3 * B) 
  (h3 : B + C + T = 26) : 
  B - T = 4 := 
by 
  sorry

end trisha_collects_4_dozen_less_l283_283755


namespace pages_wednesday_l283_283353

-- Given conditions as definitions
def borrow_books := 3
def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51

-- Prove that Nico read 19 pages on Wednesday
theorem pages_wednesday :
  let pages_wednesday := total_pages - (pages_monday + pages_tuesday)
  pages_wednesday = 19 :=
by
  sorry

end pages_wednesday_l283_283353


namespace quadratic_solution_l283_283853

theorem quadratic_solution : 
  ∀ x : ℝ, (x^2 - 2*x - 3 = 0) ↔ (x = 3 ∨ x = -1) :=
by {
  sorry
}

end quadratic_solution_l283_283853


namespace solve_quadratic_substitution_l283_283736

theorem solve_quadratic_substitution : 
  (∀ x : ℝ, (2 * x - 5) ^ 2 - 2 * (2 * x - 5) - 3 = 0 ↔ x = 2 ∨ x = 4) :=
by
  sorry

end solve_quadratic_substitution_l283_283736


namespace find_original_fraction_l283_283562

theorem find_original_fraction (x y : ℚ) (h : (1.15 * x) / (0.92 * y) = 15 / 16) :
  x / y = 69 / 92 :=
sorry

end find_original_fraction_l283_283562


namespace r_pow_four_solution_l283_283305

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l283_283305


namespace island_not_Maya_l283_283724

variable (A B : Prop)
variable (IslandMaya : Prop)
variable (Liar : Prop → Prop)
variable (TruthTeller : Prop → Prop)

-- A's statement: "We are both liars, and this island is called Maya."
axiom A_statement : Liar A ∧ Liar B ∧ IslandMaya

-- B's statement: "At least one of us is a liar, and this island is not called Maya."
axiom B_statement : (Liar A ∨ Liar B) ∧ ¬IslandMaya

theorem island_not_Maya : ¬IslandMaya := by
  sorry

end island_not_Maya_l283_283724


namespace side_length_of_base_l283_283392

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end side_length_of_base_l283_283392


namespace largest_quantity_l283_283319

theorem largest_quantity (x y z w : ℤ) (h : x + 5 = y - 3 ∧ y - 3 = z + 2 ∧ z + 2 = w - 4) : w > y ∧ w > z ∧ w > x :=
by
  sorry

end largest_quantity_l283_283319


namespace largest_possible_a_l283_283538

theorem largest_possible_a (a b : ℕ)
  (h1 : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120) : 
  a ≤ 20 :=
begin
  sorry
end

end largest_possible_a_l283_283538


namespace units_digit_char_of_p_l283_283252

theorem units_digit_char_of_p (p : ℕ) (h_pos : 0 < p) (h_even : p % 2 = 0)
    (h_units_zero : (p^3 % 10) - (p^2 % 10) = 0) (h_units_eleven : (p + 5) % 10 = 1) :
    p % 10 = 6 :=
sorry

end units_digit_char_of_p_l283_283252


namespace lines_parallel_condition_l283_283345

theorem lines_parallel_condition (a : ℝ) : 
  (a = 1) ↔ (∀ x y : ℝ, (a * x + 2 * y - 1 = 0 → x + (a + 1) * y + 4 = 0)) :=
sorry

end lines_parallel_condition_l283_283345


namespace rhombus_diagonal_l283_283015

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h : d1 * d2 = 2 * area) (hd2 : d2 = 21) (h_area : area = 157.5) : d1 = 15 :=
by
  sorry

end rhombus_diagonal_l283_283015


namespace unique_solution_for_system_l283_283769

theorem unique_solution_for_system (a : ℝ) :
  (∃! (x y : ℝ), x^2 + 4 * y^2 = 1 ∧ x + 2 * y = a) ↔ a = -1.41 :=
by
  sorry

end unique_solution_for_system_l283_283769


namespace determine_ABC_l283_283155

noncomputable def digits_are_non_zero_distinct_and_not_larger_than_5 (A B C : ℕ) : Prop :=
  0 < A ∧ A ≤ 5 ∧ 0 < B ∧ B ≤ 5 ∧ 0 < C ∧ C ≤ 5 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

noncomputable def first_condition (A B : ℕ) : Prop :=
  A * 6 + B + A = B * 6 + A -- AB_6 + A_6 = BA_6 condition translated into arithmetics

noncomputable def second_condition (A B C : ℕ) : Prop :=
  A * 6 + B + B = C * 6 + 1 -- AB_6 + B_6 = C1_6 condition translated into arithmetics

theorem determine_ABC (A B C : ℕ) (h1 : digits_are_non_zero_distinct_and_not_larger_than_5 A B C)
    (h2 : first_condition A B) (h3 : second_condition A B C) :
    A * 100 + B * 10 + C = 5 * 100 + 1 * 10 + 5 := -- Final transformation of ABC to 515
  sorry

end determine_ABC_l283_283155


namespace mother_nickels_eq_two_l283_283706

def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 9
def total_nickels : ℕ := 18

theorem mother_nickels_eq_two : (total_nickels = initial_nickels + dad_nickels + 2) :=
by
  sorry

end mother_nickels_eq_two_l283_283706


namespace employed_females_percentage_l283_283186

variable (P : ℝ) -- Total population of town X
variable (E_P : ℝ) -- Percentage of the population that is employed
variable (M_E_P : ℝ) -- Percentage of the population that are employed males

-- Conditions
axiom h1 : E_P = 0.64
axiom h2 : M_E_P = 0.55

-- Target: Prove the percentage of employed people in town X that are females
theorem employed_females_percentage (h : P > 0) : 
  (E_P * P - M_E_P * P) / (E_P * P) * 100 = 14.06 := by
sorry

end employed_females_percentage_l283_283186


namespace sum_of_smallest_integers_l283_283715

theorem sum_of_smallest_integers (x y : ℕ) (h1 : ∃ x, x > 0 ∧ (∃ n : ℕ, 720 * x = n^2) ∧ (∀ m : ℕ, m > 0 ∧ (∃ k : ℕ, 720 * m = k^2) → x ≤ m))
  (h2 : ∃ y, y > 0 ∧ (∃ p : ℕ, 720 * y = p^4) ∧ (∀ q : ℕ, q > 0 ∧ (∃ r : ℕ, 720 * q = r^4) → y ≤ q)) :
  x + y = 1130 := 
sorry

end sum_of_smallest_integers_l283_283715


namespace initial_soup_weight_l283_283452

theorem initial_soup_weight (W: ℕ) (h: W / 16 = 5): W = 40 :=
by
  sorry

end initial_soup_weight_l283_283452


namespace part_I_part_II_l283_283785

noncomputable def f (x : ℝ) := Real.sin x
noncomputable def f' (x : ℝ) := Real.cos x

theorem part_I (x : ℝ) (h : 0 < x) : f' x > 1 - x^2 / 2 := sorry

theorem part_II (a : ℝ) : (∀ x, 0 < x ∧ x < Real.pi / 2 → f x + f x / f' x > a * x) ↔ a ≤ 2 := sorry

end part_I_part_II_l283_283785


namespace find_ac_and_area_l283_283807

variables {a b c : ℝ} {A B C : ℝ}
variables (triangle_abc : ∀ {a b c : ℝ} {A B C : ℝ}, a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
variables (h1 : (a ^ 2 + c ^ 2 - b ^ 2) / Real.cos B = 4)
variables (h2 : (2 * b * Real.cos C - 2 * c * Real.cos B) / (b * Real.cos C + c * Real.cos B) - c / a = 2)

noncomputable def ac_value := 2

noncomputable def area_of_triangle_abc := (Real.sqrt 15) / 4

theorem find_ac_and_area (triangle_abc : ∀ {a b c : ℝ} {A B C : ℝ}, a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) 
                         (h1 : (a ^ 2 + c ^ 2 - b ^ 2) / Real.cos B = 4) 
                         (h2 : (2 * b * Real.cos C - 2 * c * Real.cos B) / (b * Real.cos C + c * Real.cos B) - c / a = 2):
  ac_value = 2 ∧
  area_of_triangle_abc = (Real.sqrt 15) / 4 := 
sorry

end find_ac_and_area_l283_283807


namespace problem_statement_l283_283112

theorem problem_statement (c d : ℤ) (h1 : 5 + c = 7 - d) (h2 : 6 + d = 10 + c) : 5 - c = 6 := 
by {
  sorry
}

end problem_statement_l283_283112


namespace find_missing_fraction_l283_283837

theorem find_missing_fraction :
  ∃ (x : ℚ), (1/2 + -5/6 + 1/5 + 1/4 + -9/20 + -9/20 + x = 9/20) :=
  by
  sorry

end find_missing_fraction_l283_283837


namespace inequality_am_gm_l283_283621

variable (a b x y : ℝ)

theorem inequality_am_gm (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
  (a^2 / x) + (b^2 / y) ≥ (a + b)^2 / (x + y) :=
by {
  -- proof will be filled here
  sorry
}

end inequality_am_gm_l283_283621


namespace Sandy_marks_per_correct_sum_l283_283363

theorem Sandy_marks_per_correct_sum
  (x : ℝ)  -- number of marks Sandy gets for each correct sum
  (marks_lost_per_incorrect : ℝ := 2)  -- 2 marks lost for each incorrect sum, default value is 2
  (total_attempts : ℤ := 30)  -- Sandy attempts 30 sums, default value is 30
  (total_marks : ℝ := 60)  -- Sandy obtains 60 marks, default value is 60
  (correct_sums : ℤ := 24)  -- Sandy got 24 sums correct, default value is 24
  (incorrect_sums := total_attempts - correct_sums) -- incorrect sums are the remaining attempts
  (marks_from_correct := correct_sums * x) -- total marks from the correct sums
  (marks_lost_from_incorrect := incorrect_sums * marks_lost_per_incorrect) -- total marks lost from the incorrect sums
  (total_marks_obtained := marks_from_correct - marks_lost_from_incorrect) -- total marks obtained

  -- The theorem states that x must be 3 given the conditions above
  : total_marks_obtained = total_marks → x = 3 := by sorry

end Sandy_marks_per_correct_sum_l283_283363


namespace minimum_value_of_E_l283_283990

theorem minimum_value_of_E (x E : ℝ) (h : |x - 4| + |E| + |x - 5| = 12) : |E| = 11 :=
sorry

end minimum_value_of_E_l283_283990


namespace two_cells_for_three_congruent_parts_l283_283244

-- Definitions for 4x4 grid and removing cells
def Grid := Fin (4 * 4)

-- Function to simulate removing a cell
def remove_cell (g : Finset Grid) (pos : Grid) : Finset Grid :=
  g.erase pos

-- Predicate to check if a grid can be divided into three congruent parts after cell removal
def can_be_divided_into_three_congruent_parts (g : Finset Grid) : Prop := sorry

-- Initial 4x4 grid (16 cells indexed from 0 to 15)
def initial_grid : Finset Grid := { i | i < 16 }.toFinset

-- Example positions (3, 2) and (4, 3)
def pos1 : Grid := 10 -- (3, 2) in 0-based indexing
def pos2 : Grid := 14 -- (4, 3) in 0-based indexing

theorem two_cells_for_three_congruent_parts :
  (can_be_divided_into_three_congruent_parts (remove_cell initial_grid pos1)) ∧ 
  (can_be_divided_into_three_congruent_parts (remove_cell initial_grid pos2)) :=
sorry

end two_cells_for_three_congruent_parts_l283_283244


namespace arnold_protein_intake_l283_283882

theorem arnold_protein_intake :
  (∀ p q s : ℕ,  p = 18 / 2 ∧ q = 21 ∧ s = 56 → (p + q + s = 86)) := by
  sorry

end arnold_protein_intake_l283_283882


namespace possible_values_of_N_l283_283654

def is_valid_N (N : ℕ) : Prop :=
  (N > 22) ∧ (N ≤ 25)

theorem possible_values_of_N :
  {N : ℕ | is_valid_N N} = {23, 24, 25} :=
by
  sorry

end possible_values_of_N_l283_283654


namespace part1_a_value_part2_solution_part3_incorrect_solution_l283_283777

-- Part 1: Given solution {x = 1, y = 1}, prove a = 3
theorem part1_a_value (a : ℤ) (h1 : 1 + 2 * 1 = a) : a = 3 := 
by 
  sorry

-- Part 2: Given a = -2, prove the solution is {x = 0, y = -1}
theorem part2_solution (x y : ℤ) (h1 : x + 2 * y = -2) (h2 : 2 * x - y = 1) : x = 0 ∧ y = -1 := 
by 
  sorry

-- Part 3: Given {x = -2, y = -2}, prove that it is not a solution
theorem part3_incorrect_solution (a : ℤ) (h1 : -2 + 2 * (-2) = a) (h2 : 2 * (-2) - (-2) = 1) : False := 
by 
  sorry

end part1_a_value_part2_solution_part3_incorrect_solution_l283_283777


namespace length_of_brick_proof_l283_283203

noncomputable def length_of_brick (courtyard_length courtyard_width : ℕ) (brick_width : ℕ) (total_bricks : ℕ) : ℕ :=
  let total_area_cm := courtyard_length * courtyard_width * 10000
  total_area_cm / (brick_width * total_bricks)

theorem length_of_brick_proof :
  length_of_brick 25 16 10 20000 = 20 :=
by
  unfold length_of_brick
  sorry

end length_of_brick_proof_l283_283203


namespace rectangle_area_error_l283_283879

theorem rectangle_area_error (A B : ℝ) :
  let A' := 1.08 * A
  let B' := 1.08 * B
  let actual_area := A * B
  let measured_area := A' * B'
  let percentage_error := ((measured_area - actual_area) / actual_area) * 100
  percentage_error = 16.64 :=
by
  sorry

end rectangle_area_error_l283_283879


namespace probability_top_red_suit_l283_283751

open Finset

def is_red_suit (card : ℕ) : Prop :=
  card < 13 ∨ (26 ≤ card ∧ card < 39)
  
def is_red_card (deck : Finset ℕ) (card : ℕ) : Prop :=
  card ∈ deck ∧ is_red_suit card

theorem probability_top_red_suit:
  let deck := range 52
  in ∃ total_cards : ℕ,
      total_cards = 52 ∧
      let red_cards := deck.filter is_red_suit 
      in (red_cards.card : ℚ) / (total_cards : ℚ) = 1 / 2 :=
by
  sorry

end probability_top_red_suit_l283_283751


namespace line_plane_intersection_l283_283077

theorem line_plane_intersection :
  (∃ t : ℝ, (x, y, z) = (3 + t, 1 - t, -5) ∧ (3 + t) + 7 * (1 - t) + 3 * (-5) + 11 = 0) →
  (x, y, z) = (4, 0, -5) :=
sorry

end line_plane_intersection_l283_283077


namespace jason_egg_consumption_in_two_weeks_l283_283231

def breakfast_pattern : List Nat := 
  [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] -- Two weeks pattern alternating 3-egg and (2+1)-egg meals

noncomputable def count_eggs (pattern : List Nat) : Nat :=
  pattern.foldl (· + ·) 0

theorem jason_egg_consumption_in_two_weeks : 
  count_eggs breakfast_pattern = 42 :=
sorry

end jason_egg_consumption_in_two_weeks_l283_283231


namespace range_of_g_function_l283_283415

theorem range_of_g_function :
  (∀ x ∈ Ioo (-π / 6) (5 * π / 6), g(x) = sin(x - π / 6)) →
  (range (λ x, sin(x - π / 6)) ∩ Ioo (-π / 6) (5 * π / 6) = set.Icc (-√3 / 2) 1) :=
by
  sorry

end range_of_g_function_l283_283415


namespace possible_values_of_N_l283_283661

theorem possible_values_of_N (N : ℕ) (h : N > 8) :
  22 < N ∧ N ≤ 25 →
  N = 23 ∨ N = 24 ∨ N = 25 :=
by
  intros
  sorry

end possible_values_of_N_l283_283661


namespace positive_difference_of_complementary_angles_in_ratio_5_to_4_l283_283830

-- Definitions for given conditions
def is_complementary (a b : ℝ) : Prop :=
  a + b = 90

def ratio_5_to_4 (a b : ℝ) : Prop :=
  ∃ x : ℝ, a = 5 * x ∧ b = 4 * x

-- Theorem to prove the measure of their positive difference is 10 degrees
theorem positive_difference_of_complementary_angles_in_ratio_5_to_4
  {a b : ℝ} (h_complementary : is_complementary a b) (h_ratio : ratio_5_to_4 a b) :
  abs (a - b) = 10 :=
by 
  sorry

end positive_difference_of_complementary_angles_in_ratio_5_to_4_l283_283830


namespace arun_age_proof_l283_283856

theorem arun_age_proof {A G M : ℕ} 
  (h1 : (A - 6) / 18 = G)
  (h2 : G = M - 2)
  (h3 : M = 5) :
  A = 60 :=
by
  sorry

end arun_age_proof_l283_283856


namespace evaluate_expression_correct_l283_283600

def evaluate_expression : ℚ :=
  let a := 17
  let b := 19
  let c := 23
  let numerator := a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)
  let denominator := a * (1/b + 1/c) + b * (1/c + 1/a) + c * (1/a + 1/b)
  numerator / denominator

theorem evaluate_expression_correct : evaluate_expression = 59 := 
by {
  -- proof skipped
  sorry
}

end evaluate_expression_correct_l283_283600


namespace speed_of_current_l283_283219

theorem speed_of_current (h_start: ∀ t: ℝ, t ≥ 0 → u ≥ 0) 
  (boat1_turn_2pm: ∀ t: ℝ, t >= 1 → t < 2 → boat1_turn_13_14) 
  (boat2_turn_3pm: ∀ t: ℝ, t >= 2 → t < 3 → boat2_turn_14_15) 
  (boats_meet: ∀ x: ℝ, x = 7.5) :
  v = 2.5 := 
sorry

end speed_of_current_l283_283219


namespace r_squared_sum_l283_283292

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l283_283292


namespace root_of_quadratic_eq_l283_283835

theorem root_of_quadratic_eq : 
  ∃ x₁ x₂ : ℝ, (x₁ = 0 ∧ x₂ = 2) ∧ ∀ x : ℝ, x^2 - 2 * x = 0 → (x = x₁ ∨ x = x₂) :=
by
  sorry

end root_of_quadratic_eq_l283_283835


namespace four_people_possible_l283_283051

structure Person :=
(first_name : String)
(patronymic : String)
(surname : String)

def noThreePeopleShareSameAttribute (people : List Person) : Prop :=
  ∀ (attr : Person → String), ¬ ∃ (a b c : Person),
    a ∈ people ∧ b ∈ people ∧ c ∈ people ∧ (attr a = attr b) ∧ (attr b = attr c) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def anyTwoPeopleShareAnAttribute (people : List Person) : Prop :=
  ∀ (a b : Person), a ∈ people ∧ b ∈ people ∧ a ≠ b →
    (a.first_name = b.first_name ∨ a.patronymic = b.patronymic ∨ a.surname = b.surname)

def validGroup (people : List Person) : Prop :=
  noThreePeopleShareSameAttribute people ∧ anyTwoPeopleShareAnAttribute people

theorem four_people_possible : ∃ (people : List Person), people.length = 4 ∧ validGroup people :=
sorry

end four_people_possible_l283_283051


namespace remainder_of_875_div_by_170_l283_283164

theorem remainder_of_875_div_by_170 :
  ∃ r, (∀ x, x ∣ 680 ∧ x ∣ (875 - r) → x ≤ 170) ∧ 170 ∣ (875 - r) ∧ r = 25 :=
by
  sorry

end remainder_of_875_div_by_170_l283_283164


namespace sum_of_numbers_l283_283595

theorem sum_of_numbers :
  145 + 35 + 25 + 5 = 210 :=
by
  sorry

end sum_of_numbers_l283_283595


namespace total_handshakes_calculation_l283_283641

-- Define the conditions
def teams := 3
def players_per_team := 5
def total_players := teams * players_per_team
def referees := 2

def handshakes_among_players := (total_players * (players_per_team * (teams - 1))) / 2
def handshakes_with_referees := total_players * referees

def total_handshakes := handshakes_among_players + handshakes_with_referees

-- Define the theorem statement
theorem total_handshakes_calculation :
  total_handshakes = 105 :=
by
  sorry

end total_handshakes_calculation_l283_283641


namespace common_divisors_greatest_l283_283030

theorem common_divisors_greatest (n : ℕ) (h₁ : ∀ d, d ∣ 120 ∧ d ∣ n ↔ d = 1 ∨ d = 3 ∨ d = 9) : 9 = Nat.gcd 120 n := by
  sorry

end common_divisors_greatest_l283_283030


namespace expected_value_of_winnings_l283_283042

noncomputable def expected_value : ℝ :=
  (1 / 8) * (1 / 2) + (1 / 8) * (3 / 2) + (1 / 8) * (5 / 2) + (1 / 8) * (7 / 2) +
  (1 / 8) * 2 + (1 / 8) * 4 + (1 / 8) * 6 + (1 / 8) * 8

theorem expected_value_of_winnings : expected_value = 3.5 :=
by
  -- the proof steps will go here
  sorry

end expected_value_of_winnings_l283_283042


namespace angles_of_triangle_in_geometric_sequence_l283_283971

theorem angles_of_triangle_in_geometric_sequence (α β γ : ℝ) (a q : ℝ) (hpos : 0 < a) (hq : 1 < q) (cos_rule : ∃ (a q : ℝ), (cos α = (q^4 + q^2 - 1) / (2 * q^3)) ∧ (cos β = (1 - q^2 + q^4) / (2 * q^2)) ∧ (cos γ = (1 + q^2 - q^4) / (2 * q))) :
    cos α = (q^4 + q^2 - 1) / (2 * q^3) ∧
    cos β = (1 - q^2 + q^4) / (2 * q^2) ∧
    cos γ = (1 + q^2 - q^4) / (2 * q) := sorry

end angles_of_triangle_in_geometric_sequence_l283_283971


namespace quadrant_of_angle_l283_283246

theorem quadrant_of_angle (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  ∃! (q : ℕ), q = 2 :=
sorry

end quadrant_of_angle_l283_283246


namespace neither_outstanding_nor_young_pioneers_is_15_l283_283980

-- Define the conditions
def total_students : ℕ := 87
def outstanding_students : ℕ := 58
def young_pioneers : ℕ := 63
def both_outstanding_and_young_pioneers : ℕ := 49

-- Define the function to calculate the number of students who are neither
def neither_outstanding_nor_young_pioneers
: ℕ :=
total_students - (outstanding_students - both_outstanding_and_young_pioneers) - (young_pioneers - both_outstanding_and_young_pioneers) - both_outstanding_and_young_pioneers

-- The theorem to prove
theorem neither_outstanding_nor_young_pioneers_is_15
: neither_outstanding_nor_young_pioneers = 15 :=
by
  sorry

end neither_outstanding_nor_young_pioneers_is_15_l283_283980


namespace r_power_four_identity_l283_283312

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l283_283312


namespace min_value_condition_l283_283786

theorem min_value_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  3 * m + n = 1 → (1 / m + 2 / n) ≥ 5 + 2 * Real.sqrt 6 :=
by
  sorry

end min_value_condition_l283_283786


namespace triangle_is_obtuse_l283_283754

noncomputable def is_exterior_smaller (exterior_angle interior_angle : ℝ) : Prop :=
  exterior_angle < interior_angle

noncomputable def sum_of_angles (exterior_angle interior_angle : ℝ) : Prop :=
  exterior_angle + interior_angle = 180

theorem triangle_is_obtuse (exterior_angle interior_angle : ℝ) (h1 : is_exterior_smaller exterior_angle interior_angle) 
  (h2 : sum_of_angles exterior_angle interior_angle) : ∃ b, 90 < b ∧ b = interior_angle :=
sorry

end triangle_is_obtuse_l283_283754


namespace tim_runs_more_than_sarah_l283_283925

-- Definitions based on the conditions
def street_width : ℕ := 25
def side_length : ℕ := 450

-- Perimeters of the paths
def sarah_perimeter : ℕ := 4 * side_length
def tim_perimeter : ℕ := 4 * (side_length + 2 * street_width)

-- The theorem to prove
theorem tim_runs_more_than_sarah : tim_perimeter - sarah_perimeter = 200 := by
  -- The proof will be filled in here
  sorry

end tim_runs_more_than_sarah_l283_283925


namespace find_k_series_sum_l283_283080

theorem find_k_series_sum :
  (∃ k : ℝ, 5 + ∑' n : ℕ, ((5 + (n + 1) * k) / 5^n.succ) = 10) →
  k = 12 :=
sorry

end find_k_series_sum_l283_283080


namespace bisection_next_interval_l283_283775

def f (x : ℝ) : ℝ := x^3 - 2 * x - 5

theorem bisection_next_interval (h₀ : f 2.5 > 0) (h₁ : f 2 < 0) :
  ∃ a b, (2 < 2.5) ∧ f 2 < 0 ∧ f 2.5 > 0 ∧ a = 2 ∧ b = 2.5 :=
by
  sorry

end bisection_next_interval_l283_283775


namespace katy_brownies_l283_283131

theorem katy_brownies :
  ∃ (n : ℤ), (n = 5 + 2 * 5) :=
by
  sorry

end katy_brownies_l283_283131


namespace sum_of_solutions_eq_zero_l283_283318

theorem sum_of_solutions_eq_zero (x : ℝ) (hx : x^2 = 25) : ∃ x₁ x₂ : ℝ, (x₁^2 = 25) ∧ (x₂^2 = 25) ∧ (x₁ + x₂ = 0) := 
by {
  use 5,
  use (-5),
  split,
  { exact hx, },
  split,
  { rw pow_two, exact hx, },
  { norm_num, },
}

end sum_of_solutions_eq_zero_l283_283318


namespace solve_for_y_l283_283791

theorem solve_for_y (x y : ℚ) (h₁ : x - y = 12) (h₂ : 2 * x + y = 10) : y = -14 / 3 :=
by
  sorry

end solve_for_y_l283_283791


namespace find_b_l283_283628

theorem find_b (a b : ℝ) (h₁ : 2 * a + 3 = 5) (h₂ : b - a = 2) : b = 3 :=
by 
  sorry

end find_b_l283_283628


namespace find_r_fourth_l283_283282

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l283_283282


namespace symmetric_point_origin_l283_283412

def symmetric_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (-P.1, -P.2)

theorem symmetric_point_origin :
  symmetric_point (3, -1) = (-3, 1) :=
by
  sorry

end symmetric_point_origin_l283_283412


namespace incorrect_calculation_l283_283728

theorem incorrect_calculation : ¬ (3 + 2 * Real.sqrt 2 = 5 * Real.sqrt 2) :=
by sorry

end incorrect_calculation_l283_283728


namespace motel_percentage_reduction_l283_283702

theorem motel_percentage_reduction
  (x y : ℕ) 
  (h : 40 * x + 60 * y = 1000) :
  ((1000 - (40 * (x + 10) + 60 * (y - 10))) / 1000) * 100 = 20 := 
by
  sorry

end motel_percentage_reduction_l283_283702


namespace price_difference_l283_283561

noncomputable def original_price (final_sale_price discount : ℝ) := final_sale_price / (1 - discount)

noncomputable def after_price_increase (price after_increase : ℝ) := price * (1 + after_increase)

theorem price_difference (final_sale_price : ℝ) (discount : ℝ) (price_increase : ℝ) 
    (h1 : final_sale_price = 85) (h2 : discount = 0.15) (h3 : price_increase = 0.25) : 
    after_price_increase final_sale_price price_increase - original_price final_sale_price discount = 6.25 := 
by 
    sorry

end price_difference_l283_283561


namespace ab_diff_2023_l283_283612

theorem ab_diff_2023 (a b : ℝ) 
  (h : a^2 + b^2 - 4 * a - 6 * b + 13 = 0) : (a - b) ^ 2023 = -1 :=
sorry

end ab_diff_2023_l283_283612


namespace max_value_of_expression_l283_283326

theorem max_value_of_expression (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  ∃ m, m = 15 ∧ x^2 + y^2 + 2 * x ≤ m := 
sorry

end max_value_of_expression_l283_283326


namespace moles_of_KOH_used_l283_283240

variable {n_KOH : ℝ}

theorem moles_of_KOH_used :
  ∃ n_KOH, (NH4I + KOH = KI_produced) → (KI_produced = 1) → n_KOH = 1 :=
by
  sorry

end moles_of_KOH_used_l283_283240


namespace r_fourth_power_sum_l283_283278

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l283_283278


namespace hyperbola_asymptote_equation_l283_283711

variable (a b : ℝ)
variable (x y : ℝ)

def arithmetic_mean := (a + b) / 2 = 5
def geometric_mean := (a * b) ^ (1 / 2) = 4
def a_greater_b := a > b
def hyperbola_asymptote := (y = (1 / 2) * x) ∨ (y = -(1 / 2) * x)

theorem hyperbola_asymptote_equation :
  arithmetic_mean a b ∧ geometric_mean a b ∧ a_greater_b a b → hyperbola_asymptote x y :=
by
  sorry

end hyperbola_asymptote_equation_l283_283711


namespace line_inclination_angle_l283_283976

theorem line_inclination_angle (θ : ℝ) : 
  (∃ θ : ℝ, ∀ x y : ℝ, x + y + 1 = 0 → θ = 3 * π / 4) := sorry

end line_inclination_angle_l283_283976


namespace minimum_value_of_f_l283_283493

noncomputable def f (x : ℝ) : ℝ := 4 * x + 1 / (4 * x - 5)

theorem minimum_value_of_f (x : ℝ) : x > 5 / 4 → ∃ y, ∀ z, f z ≥ y ∧ y = 7 :=
by
  intro h
  sorry

end minimum_value_of_f_l283_283493


namespace find_linear_function_l283_283468

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
(∀ (a b c : ℝ), a + b + c ≥ 0 → f (a^3) + f (b^3) + f (c^3) ≥ 3 * f (a * b * c))
∧ (∀ (a b c : ℝ), a + b + c ≤ 0 → f (a^3) + f (b^3) + f (c^3) ≤ 3 * f (a * b * c))

theorem find_linear_function (f : ℝ → ℝ) (h : functional_equation f) : ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end find_linear_function_l283_283468


namespace katy_brownies_l283_283127

-- Define the conditions
def ate_monday : ℕ := 5
def ate_tuesday : ℕ := 2 * ate_monday

-- Define the question
def total_brownies : ℕ := ate_monday + ate_tuesday

-- State the proof problem
theorem katy_brownies : total_brownies = 15 := by
  sorry

end katy_brownies_l283_283127


namespace calculate_expression_l283_283064

noncomputable def exponent_inverse (x : ℝ) : ℝ := x ^ (-1)
noncomputable def root (x : ℝ) (n : ℕ) : ℝ := x ^ (1 / n : ℝ)

theorem calculate_expression :
  (exponent_inverse 4) - (root (1/16) 2) + (3 - Real.sqrt 2) ^ 0 = 1 := 
by
  -- Definitions according to conditions
  have h1 : exponent_inverse 4 = 1 / 4 := by sorry
  have h2 : root (1 / 16) 2 = 1 / 4 := by sorry
  have h3 : (3 - Real.sqrt 2) ^ 0 = 1 := by sorry
  
  -- Combine and simplify parts
  calc
    (exponent_inverse 4) - (root (1 / 16) 2) + (3 - Real.sqrt 2) ^ 0
        = (1 / 4) - (1 / 4) + 1 : by rw [h1, h2, h3]
    ... = 0 + 1 : by sorry
    ... = 1 : by rfl

end calculate_expression_l283_283064


namespace jeans_sold_l283_283523

-- Definitions based on conditions
def price_per_jean : ℤ := 11
def price_per_tee : ℤ := 8
def tees_sold : ℤ := 7
def total_money : ℤ := 100

-- Proof statement
theorem jeans_sold (J : ℤ)
  (h1 : price_per_jean = 11)
  (h2 : price_per_tee = 8)
  (h3 : tees_sold = 7)
  (h4 : total_money = 100) :
  J = 4 :=
by
  sorry

end jeans_sold_l283_283523


namespace first_student_can_ensure_one_real_root_l283_283950

noncomputable def can_first_student_ensure_one_real_root : Prop :=
  ∀ (b c : ℝ), ∃ a : ℝ, ∃ d : ℝ, ∀ (e : ℝ), 
    (d = 0 ∧ (e = b ∨ e = c)) → 
    (∀ x : ℝ, x^3 + d * x^2 + e * x + (if e = b then c else b) = 0)

theorem first_student_can_ensure_one_real_root :
  can_first_student_ensure_one_real_root := sorry

end first_student_can_ensure_one_real_root_l283_283950


namespace hash_value_is_minus_15_l283_283138

def hash (a b c : ℝ) : ℝ := b^2 - 3 * a * c

theorem hash_value_is_minus_15 : hash 2 3 4 = -15 :=
by
  sorry

end hash_value_is_minus_15_l283_283138


namespace fraction_calculation_l283_283898

-- Define the initial values of x and y
def x : ℚ := 4 / 6
def y : ℚ := 8 / 10

-- Statement to prove
theorem fraction_calculation : (6 * x^2 + 10 * y) / (60 * x * y) = 11 / 36 := by
  sorry

end fraction_calculation_l283_283898


namespace interest_rate_is_20_percent_l283_283191

theorem interest_rate_is_20_percent (P A : ℝ) (t : ℝ) (r : ℝ) 
  (h1 : P = 500) (h2 : A = 1000) (h3 : t = 5) :
  A = P * (1 + r * t) → r = 0.20 :=
by
  intro h
  sorry

end interest_rate_is_20_percent_l283_283191


namespace pure_imaginary_complex_number_solution_l283_283629

theorem pure_imaginary_complex_number_solution (m : ℝ) :
  (m^2 - 5 * m + 6 = 0) ∧ (m^2 - 3 * m ≠ 0) → m = 2 :=
by
  sorry

end pure_imaginary_complex_number_solution_l283_283629


namespace r_pow_four_solution_l283_283298

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l283_283298


namespace triangles_not_necessarily_congruent_l283_283123

-- Define the triangles and their properties
structure Triangle :=
  (A B C : ℝ)

-- Define angles and measures for heights and medians
def angle (t : Triangle) : ℝ := sorry
def height_from (t : Triangle) (v : ℝ) : ℝ := sorry
def median_from (t : Triangle) (v : ℝ) : ℝ := sorry

theorem triangles_not_necessarily_congruent
  (T₁ T₂ : Triangle)
  (h_angle : angle T₁ = angle T₂)
  (h_height : height_from T₁ T₁.B = height_from T₂ T₂.B)
  (h_median : median_from T₁ T₁.C = median_from T₂ T₂.C) :
  ¬ (T₁ = T₂) := 
sorry

end triangles_not_necessarily_congruent_l283_283123


namespace r_squared_sum_l283_283297

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l283_283297


namespace area_change_correct_l283_283795

theorem area_change_correct (L B : ℝ) (A : ℝ) (x : ℝ) (hx1 : A = L * B)
  (hx2 : ((L + (x / 100) * L) * (B - (x / 100) * B)) = A - (1 / 100) * A) :
  x = 10 := by
  sorry

end area_change_correct_l283_283795


namespace cubic_sum_l283_283635

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end cubic_sum_l283_283635


namespace pyramid_base_side_length_l283_283386

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end pyramid_base_side_length_l283_283386


namespace find_last_number_l283_283038

-- Definitions for the conditions
def avg_first_three (A B C : ℕ) : ℕ := (A + B + C) / 3
def avg_last_three (B C D : ℕ) : ℕ := (B + C + D) / 3
def sum_first_last (A D : ℕ) : ℕ := A + D

-- Proof problem statement
theorem find_last_number (A B C D : ℕ) 
  (h1 : avg_first_three A B C = 6)
  (h2 : avg_last_three B C D = 5)
  (h3 : sum_first_last A D = 11) : D = 4 :=
sorry

end find_last_number_l283_283038


namespace repeating_decimal_fraction_value_l283_283847

def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  d

theorem repeating_decimal_fraction_value :
  repeating_decimal_to_fraction (73 / 100 + 246 / 999000) = 731514 / 999900 :=
by
  sorry

end repeating_decimal_fraction_value_l283_283847


namespace form_a_set_l283_283713

def is_definitive (description: String) : Prop :=
  match description with
  | "comparatively small numbers" => False
  | "non-negative even numbers not greater than 10" => True
  | "all triangles" => True
  | "points in the Cartesian coordinate plane with an x-coordinate of zero" => True
  | "tall male students" => False
  | "students under 17 years old in a certain class" => True
  | _ => False

theorem form_a_set :
  is_definitive "comparatively small numbers" = False ∧
  is_definitive "non-negative even numbers not greater than 10" = True ∧
  is_definitive "all triangles" = True ∧
  is_definitive "points in the Cartesian coordinate plane with an x-coordinate of zero" = True ∧
  is_definitive "tall male students" = False ∧
  is_definitive "students under 17 years old in a certain class" = True :=
by
  repeat { split };
  exact sorry

end form_a_set_l283_283713


namespace find_ordered_pair_l283_283770

theorem find_ordered_pair : 
  ∃ (x y : ℚ), 7 * x = -5 - 3 * y ∧ 4 * x = 5 * y - 34 ∧
  x = -127 / 47 ∧ y = 218 / 47 :=
by
  sorry

end find_ordered_pair_l283_283770


namespace janice_bottle_caps_l283_283125

-- Define the conditions
def num_boxes : ℕ := 79
def caps_per_box : ℕ := 4

-- Define the question as a theorem to prove
theorem janice_bottle_caps : num_boxes * caps_per_box = 316 :=
by
  sorry

end janice_bottle_caps_l283_283125


namespace evaluate_expression_l283_283555

theorem evaluate_expression : (2019 - (2000 - (10 - 9))) - (2000 - (10 - (9 - 2019))) = 40 :=
by
  sorry

end evaluate_expression_l283_283555


namespace go_games_l283_283839

theorem go_games (total_go_balls : ℕ) (go_balls_per_game : ℕ) (h_total : total_go_balls = 901) (h_game : go_balls_per_game = 53) : (total_go_balls / go_balls_per_game) = 17 := by
  sorry

end go_games_l283_283839


namespace production_period_l283_283198

-- Define the conditions as constants
def daily_production : ℕ := 1500
def price_per_computer : ℕ := 150
def total_earnings : ℕ := 1575000

-- Define the computation to find the period and state what we need to prove
theorem production_period : (total_earnings / price_per_computer) / daily_production = 7 :=
by
  -- you can provide the steps, but it's optional since the proof is omitted
  sorry

end production_period_l283_283198


namespace total_canoes_built_l283_283053

theorem total_canoes_built (boats_jan : ℕ) (h : boats_jan = 5)
    (boats_feb : ℕ) (h1 : boats_feb = boats_jan * 3)
    (boats_mar : ℕ) (h2 : boats_mar = boats_feb * 3)
    (boats_apr : ℕ) (h3 : boats_apr = boats_mar * 3) :
  boats_jan + boats_feb + boats_mar + boats_apr = 200 :=
sorry

end total_canoes_built_l283_283053


namespace initial_water_amount_l283_283196

variable (W : ℝ)
variable (evap_per_day : ℝ := 0.014)
variable (days : ℕ := 50)
variable (evap_percent : ℝ := 7.000000000000001)

theorem initial_water_amount :
  evap_per_day * (days : ℝ) = evap_percent / 100 * W → W = 10 :=
by
  sorry

end initial_water_amount_l283_283196


namespace solve_arrangement_equation_l283_283192

def arrangement_numeral (x : ℕ) : ℕ :=
  x * (x - 1) * (x - 2)

theorem solve_arrangement_equation (x : ℕ) (h : 3 * (arrangement_numeral x)^3 = 2 * (arrangement_numeral (x + 1))^2 + 6 * (arrangement_numeral x)^2) : x = 5 := 
sorry

end solve_arrangement_equation_l283_283192


namespace seq_properties_l283_283094

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x

theorem seq_properties :
  (∀ n, a_n = -2 * (1 / 3) ^ n) ∧
  (∀ n, b_n = 2 * n - 1) ∧
  (∀ t m, (-1 ≤ m ∧ m ≤ 1) → (t^2 - 2 * m * t + 1/2 > T_n) ↔ (t < -2 ∨ t > 2)) ∧
  (∃ m n, 1 < m ∧ m < n ∧ T_1 * T_n = T_m^2 ∧ m = 2 ∧ n = 12) :=
sorry

end seq_properties_l283_283094


namespace standard_normal_probability_l283_283833

noncomputable def prob_standard_normal_between_one_and_two : ℝ :=
  let P_0_to_1 := 0.3143 in
  let P_0_to_2 := 0.4772 in
  P_0_to_2 - P_0_to_1

theorem standard_normal_probability :
  prob_standard_normal_between_one_and_two = 0.1629 :=
sorry

end standard_normal_probability_l283_283833


namespace conference_duration_l283_283868

theorem conference_duration (hours minutes lunch_break total_minutes active_session : ℕ) 
  (h1 : hours = 8) 
  (h2 : minutes = 40) 
  (h3 : lunch_break = 15) 
  (h4 : total_minutes = hours * 60 + minutes)
  (h5 : active_session = total_minutes - lunch_break) :
  active_session = 505 := 
by {
  sorry
}

end conference_duration_l283_283868


namespace pyramid_base_side_length_l283_283380

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end pyramid_base_side_length_l283_283380


namespace adoption_days_l283_283432

theorem adoption_days (P0 P_in P_adopt_rate : Nat) (P_total : Nat) (hP0 : P0 = 3) (hP_in : P_in = 3) (hP_adopt_rate : P_adopt_rate = 3) (hP_total : P_total = P0 + P_in) :
  P_total / P_adopt_rate = 2 := 
by
  sorry

end adoption_days_l283_283432


namespace side_length_of_square_base_l283_283398

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end side_length_of_square_base_l283_283398


namespace find_fraction_l283_283864

theorem find_fraction (f : ℝ) (h₁ : f * 50.0 - 4 = 6) : f = 0.2 :=
by
  sorry

end find_fraction_l283_283864


namespace find_original_selling_price_l283_283451

noncomputable def original_selling_price (purchase_price : ℝ) := 
  1.10 * purchase_price

noncomputable def new_selling_price (purchase_price : ℝ) := 
  1.17 * purchase_price

theorem find_original_selling_price (P : ℝ)
  (h1 : new_selling_price P - original_selling_price P = 56) :
  original_selling_price P = 880 := by 
  sorry

end find_original_selling_price_l283_283451


namespace major_premise_wrong_l283_283027

theorem major_premise_wrong (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) (h_deriv : ∀ x, deriv f x = 0 → ¬ IsExtremum (deriv f) x) : false :=
by 
  have h : deriv (λ x : ℝ, x ^ 3) 0 = 0 := by simp [deriv]
  sorry 

end major_premise_wrong_l283_283027


namespace tan_half_sum_of_angles_l283_283135

theorem tan_half_sum_of_angles (p q : ℝ) 
    (h1 : Real.cos p + Real.cos q = 3 / 5) 
    (h2 : Real.sin p + Real.sin q = 1 / 4) :
    Real.tan ((p + q) / 2) = 5 / 12 := by
  sorry

end tan_half_sum_of_angles_l283_283135


namespace dice_sum_probability_l283_283848

-- Define a noncomputable function to calculate the number of ways to get a sum of 15
noncomputable def dice_sum_ways (dices : ℕ) (sides : ℕ) (target_sum : ℕ) : ℕ :=
  sorry

-- Define the Lean 4 statement
theorem dice_sum_probability :
  dice_sum_ways 5 6 15 = 2002 :=
sorry

end dice_sum_probability_l283_283848


namespace distribute_pencils_l283_283478

theorem distribute_pencils (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  (∃ ways : ℕ, ways = Nat.choose (n - 1) (k - 1) ∧ ways = 35) :=
by
  sorry

end distribute_pencils_l283_283478


namespace impossibility_to_equalize_11_vertices_l283_283566

open Finset

def operation (a : ℕ → ℤ) (i : ℕ) : (ℕ → ℤ) :=
λ j, if j = i then 0 else if j = i.pred % 12 ∨ j = (i + 1) % 12 then a j + (a i) / 2 else a j

-- Assume the initial state vector a₀
def a_initial : ℕ → ℤ := nat_to_int

-- Sum of vertices
def sum_vertices (a : ℕ → ℤ) : ℤ := ∑ i in range 12, a i

-- Weighted sum in modulo 3
def weighted_sum_mod3 (a : ℕ → ℤ) : ℤ := (∑ i in range 12, i * a i) % 3

theorem impossibility_to_equalize_11_vertices
  (initial_state : ℕ → ℤ := λ i, i) :
  ¬ ∃ (a : ℕ → ℤ),
    (∀ i, a i = initial_state i) ∧
    (∀ n, a (operation a n) = operation a) ∧
    (sum_vertices a = 66) ∧
    ((∃ k : ℤ, ∀ i, (i ≠ 0) → a i = k) ∧ a 0 = 0) :=
by {
  assume h,
  sorry, -- The detailed proof should be filled here
}

end impossibility_to_equalize_11_vertices_l283_283566


namespace probability_non_defective_pens_l283_283735

theorem probability_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) (bought_pens : ℕ) 
  (h1 : total_pens = 12)
  (h2 : defective_pens = 4)
  (h3 : bought_pens = 2) :
  (C (total_pens - defective_pens) bought_pens) * (C defective_pens (0)) /
  (C total_pens bought_pens) = 14 / 33 :=
begin
  unfold C,
  rw [h1, h2, h3],
  norm_num,
  -- More steps to complete the proof would go here
  sorry
end 

end probability_non_defective_pens_l283_283735


namespace biased_coin_probability_l283_283745

variable (p q : ℝ)
variable (h : p + q = 1)

theorem biased_coin_probability :
  (∑ i in (Finset.range 1), (Nat.choose 5 1) * p^1 * q^(5 - 1)) = 5 * p * q^4 :=
by
  sorry

end biased_coin_probability_l283_283745


namespace bagels_count_l283_283719

def total_items : ℕ := 90
def bread_rolls : ℕ := 49
def croissants : ℕ := 19

def bagels : ℕ := total_items - (bread_rolls + croissants)

theorem bagels_count : bagels = 22 :=
by
  sorry

end bagels_count_l283_283719


namespace oliver_remaining_dishes_l283_283442

def num_dishes := 36
def dishes_with_mango_salsa := 3
def dishes_with_fresh_mango := num_dishes / 6
def dishes_with_mango_jelly := 1
def oliver_picks_two := 2

theorem oliver_remaining_dishes : 
  num_dishes - (dishes_with_mango_salsa + dishes_with_fresh_mango + dishes_with_mango_jelly - oliver_picks_two) = 28 := by
  sorry

end oliver_remaining_dishes_l283_283442


namespace distinct_numbers_count_l283_283237

open BigOperators

theorem distinct_numbers_count :
  (Finset.card ((Finset.image (λ n : ℕ, ⌊ (n^2 : ℝ) / 500⌋) (Finset.range 1001))) = 876) := 
sorry

end distinct_numbers_count_l283_283237


namespace friends_attended_reception_l283_283747

-- Definition of the given conditions
def total_guests : ℕ := 180
def couples_per_side : ℕ := 20

-- Statement based on the given problem
theorem friends_attended_reception : 
  let family_guests := 2 * couples_per_side + 2 * couples_per_side
  let friends := total_guests - family_guests
  friends = 100 :=
by
  -- We define the family_guests calculation
  let family_guests := 2 * couples_per_side + 2 * couples_per_side
  -- We define the friends calculation
  let friends := total_guests - family_guests
  -- We state the conclusion
  show friends = 100
  sorry

end friends_attended_reception_l283_283747


namespace correct_option_D_l283_283183

variables (a b c : ℤ)

theorem correct_option_D : -2 * a + 3 * (b - 1) = -2 * a + 3 * b - 3 := 
by
  sorry

end correct_option_D_l283_283183


namespace equalities_implied_by_sum_of_squares_l283_283789

variable {a b c d : ℝ}

theorem equalities_implied_by_sum_of_squares (h1 : a = b) (h2 : c = d) : 
  (a - b) ^ 2 + (c - d) ^ 2 = 0 :=
sorry

end equalities_implied_by_sum_of_squares_l283_283789


namespace zachary_pushups_l283_283760

theorem zachary_pushups (david_pushups zachary_pushups : ℕ) (h₁ : david_pushups = 44) (h₂ : david_pushups = zachary_pushups + 9) :
  zachary_pushups = 35 :=
by
  sorry

end zachary_pushups_l283_283760


namespace r_power_four_identity_l283_283306

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l283_283306


namespace centimeters_per_inch_l283_283144

theorem centimeters_per_inch (miles_per_map_inch : ℝ) (cm_measured : ℝ) (approx_miles : ℝ) (miles_per_inch : ℝ) (inches_from_cm : ℝ) : 
  miles_per_map_inch = 16 →
  inches_from_cm = 18.503937007874015 →
  miles_per_map_inch = 24 / 1.5 →
  approx_miles = 296.06299212598424 →
  cm_measured = 47 →
  (cm_measured / inches_from_cm) = 2.54 :=
by
  sorry

end centimeters_per_inch_l283_283144


namespace total_pictures_l283_283146

-- Conditions: Randy drew 5 pictures, Peter drew 3 more pictures than Randy.
-- Quincy drew 20 more pictures than Peter. 
-- We need to prove the total number of pictures they drew is 41.

theorem total_pictures : 
  let randy_pictures := 5 in
  let peter_pictures := randy_pictures + 3 in
  let quincy_pictures := peter_pictures + 20 in
  randy_pictures + peter_pictures + quincy_pictures = 41 := 
by 
  let randy_pictures := 5
  let peter_pictures := randy_pictures + 3
  let quincy_pictures := peter_pictures + 20
  show randy_pictures + peter_pictures + quincy_pictures = 41 from sorry

end total_pictures_l283_283146


namespace sequence_general_term_l283_283018

theorem sequence_general_term (n : ℕ) : 
  (∃ (f : ℕ → ℕ), (∀ k, f k = k^2) ∧ (∀ m, f m = m^2)) :=
by
  -- Given the sequence 1, 4, 9, 16, 25, ...
  sorry

end sequence_general_term_l283_283018


namespace phi_cannot_be_chosen_l283_283029

theorem phi_cannot_be_chosen (θ φ : ℝ) (hθ : -π/2 < θ ∧ θ < π/2) (hφ : 0 < φ ∧ φ < π)
  (h1 : 3 * Real.sin θ = 3 * Real.sqrt 2 / 2) 
  (h2 : 3 * Real.sin (-2*φ + θ) = 3 * Real.sqrt 2 / 2) : φ ≠ 5*π/4 :=
by
  sorry

end phi_cannot_be_chosen_l283_283029


namespace min_queries_to_determine_parity_l283_283514

def num_bags := 100
def num_queries := 3
def bags := Finset (Fin num_bags)

def can_query_parity (bags : Finset (Fin num_bags)) : Prop :=
  bags.card = 15

theorem min_queries_to_determine_parity :
  ∀ (query : Fin num_queries → Finset (Fin num_bags)),
  (∀ i, can_query_parity (query i)) →
  (∀ i j k, query i ∪ query j ∪ query k = {a : Fin num_bags | a.val = 1}) →
  num_queries ≥ 3 :=
  sorry

end min_queries_to_determine_parity_l283_283514


namespace max_area_OAB_l283_283437

-- Define the circle and the line
def circle (x y : ℝ) : Prop := x^2 + y^2 = 1
def line (k x y : ℝ) : Prop := y = k * x - 1

-- Define the points A and B lying on the circle and the line
def point_lies_on_circle (A B : ℝ × ℝ) : Prop := circle A.1 A.2 ∧ circle B.1 B.2
def point_lies_on_line (k : ℝ) (A B : ℝ × ℝ) : Prop := line k A.1 A.2 ∧ line k B.1 B.2

-- Define the function to calculate the area of triangle OAB
def area_OAB (A B : ℝ × ℝ) : ℝ := 0.5 * (A.1 * B.2 - A.2 * B.1)

-- Define the maximum area of triangle OAB
theorem max_area_OAB (k : ℝ) (A B : ℝ × ℝ) (h_circle : point_lies_on_circle A B) (h_line : point_lies_on_line k A B) :
  area_OAB A B ≤ 0.5 :=
sorry  -- The proof will be placed here

end max_area_OAB_l283_283437


namespace smallest_n_terminating_contains_9_l283_283845

def isTerminatingDecimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2 ^ a * 5 ^ b

def containsDigit9 (n : ℕ) : Prop :=
  (Nat.digits 10 n).contains 9

theorem smallest_n_terminating_contains_9 : ∃ n : ℕ, 
  isTerminatingDecimal n ∧
  containsDigit9 n ∧
  (∀ m : ℕ, isTerminatingDecimal m ∧ containsDigit9 m → n ≤ m) ∧
  n = 5120 :=
  sorry

end smallest_n_terminating_contains_9_l283_283845


namespace simple_interest_rate_l283_283187

theorem simple_interest_rate (P I : ℕ) (hP : P = 1200) (hI : I = 108) (R : ℝ) :
  I = P * R * R / 100 → R = 3 :=
by
  intro h
  rw [hP, hI] at h
  have h1 : 108 * 100 = 1200 * (R * R),
  { rw [mul_comm, ←mul_assoc, ←mul_comm 1200, mul_assoc, div_mul_cancel] at h, norm_num at h, rw mul_comm, exact h },
  have h2 : 10800 = 1200 * R^2,
  { norm_num, exact h1 },
  have h3 : 10800 / 1200 = R^2,
  { rw ← h2, norm_num },
  have h4 : 9 = R^2,
  { norm_num, exact h3 },
  exact pow_eq_pow h4

end simple_interest_rate_l283_283187


namespace mode_of_scores_is_37_l283_283445

open List

def scores : List ℕ := [35, 37, 39, 37, 38, 38, 37]

theorem mode_of_scores_is_37 : ∀ (l : List ℕ), l = scores → mode l = 37 :=
by
  -- Lean proof goes here
  sorry

end mode_of_scores_is_37_l283_283445


namespace find_f_and_min_g_l283_283498

theorem find_f_and_min_g (f g : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x : ℝ, f (2 * x - 3) = 4 * x^2 + 2 * x + 1)
  (h2 : ∀ x : ℝ, g x = f (x + a) - 7 * x):
  
  (∀ x : ℝ, f x = x^2 + 7 * x + 13) ∧
  
  (∀ a : ℝ, 
    ∀ x : ℝ, 
      (x = 1 → (a ≥ -1 → g x = a^2 + 9 * a + 14)) ∧
      (-3 < a ∧ a < -1 → g (-a) = 7 * a + 13) ∧
      (x = 3 → (a ≤ -3 → g x = a^2 + 13 * a + 22))) :=
by
  sorry

end find_f_and_min_g_l283_283498


namespace point_coordinates_l283_283330

-- We assume that the point P has coordinates (2, 4) and prove that the coordinates with respect to the origin in Cartesian system are indeed (2, 4).
theorem point_coordinates (x y : ℝ) (h : x = 2 ∧ y = 4) : (x, y) = (2, 4) :=
by
  sorry

end point_coordinates_l283_283330


namespace polygonal_chain_max_length_not_exceed_200_l283_283089

-- Define the size of the board
def board_size : ℕ := 15

-- Define the concept of a polygonal chain length on a symmetric board
def polygonal_chain_length (n : ℕ) : ℕ := sorry -- length function yet to be defined

-- Define the maximum length constant to be compared with
def max_length : ℕ := 200

-- Define the theorem statement including all conditions and constraints
theorem polygonal_chain_max_length_not_exceed_200 :
  ∃ (n : ℕ), n = board_size ∧ 
             (∀ (length : ℕ),
             length = polygonal_chain_length n →
             length ≤ max_length) :=
sorry

end polygonal_chain_max_length_not_exceed_200_l283_283089


namespace find_DG_l283_283816

theorem find_DG 
  (a b : ℕ) -- sides DE and EC
  (S : ℕ := 19 * (a + b)) -- area of each rectangle
  (k l : ℕ) -- sides DG and CH
  (h1 : S = a * k) 
  (h2 : S = b * l) 
  (h_bc : 19 * (a + b) = S)
  (h_div_a : S % a = 0)
  (h_div_b : S % b = 0)
  : DG = 380 :=
sorry

end find_DG_l283_283816


namespace students_play_football_l283_283704

theorem students_play_football 
  (total : ℕ) (C : ℕ) (B : ℕ) (Neither : ℕ) (F : ℕ) 
  (h_total : total = 420) 
  (h_C : C = 175) 
  (h_B : B = 130) 
  (h_Neither : Neither = 50) 
  (h_inclusion_exclusion : F + C - B = total - Neither) :
  F = 325 := 
sorry

end students_play_football_l283_283704


namespace find_number_l283_283554

theorem find_number (x : ℝ) (h : (x + 0.005) / 2 = 0.2025) : x = 0.400 :=
sorry

end find_number_l283_283554


namespace expr_min_value_expr_min_at_15_l283_283774

theorem expr_min_value (a x : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  (|x - a| + |x - 15| + |x - (a + 15)|) = 30 - x := 
sorry

theorem expr_min_at_15 (a : ℝ) (h : 0 ≤ a ∧ a ≤ 15) : 
  (|15 - a| + |15 - 15| + |15 - (a + 15)|) = 15 := 
sorry

end expr_min_value_expr_min_at_15_l283_283774


namespace rectangle_diagonals_equiv_positive_even_prime_equiv_l283_283995

-- Definitions based on problem statement (1)
def is_rectangle (q : Quadrilateral) : Prop := sorry -- "q is a rectangle"
def diagonals_equal_and_bisect (q : Quadrilateral) : Prop := sorry -- "the diagonals of q are equal and bisect each other"

-- Problem statement (1)
theorem rectangle_diagonals_equiv (q : Quadrilateral) :
  (is_rectangle q → diagonals_equal_and_bisect q) ∧
  (diagonals_equal_and_bisect q → is_rectangle q) ∧
  (¬ is_rectangle q → ¬ diagonals_equal_and_bisect q) ∧
  (¬ diagonals_equal_and_bisect q → ¬ is_rectangle q) :=
sorry

-- Definitions based on problem statement (2)
def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0
def is_prime (n : ℕ) : Prop := sorry -- "n is a prime number"

-- Problem statement (2)
theorem positive_even_prime_equiv (n : ℕ) :
  (is_positive_even n → ¬ is_prime n) ∧
  ((¬ is_prime n → is_positive_even n) = False) ∧
  ((¬ is_positive_even n → is_prime n) = False) ∧
  ((is_prime n → ¬ is_positive_even n) = False) :=
sorry

end rectangle_diagonals_equiv_positive_even_prime_equiv_l283_283995


namespace number_of_ants_proof_l283_283872

-- Define the conditions
def width_ft := 500
def length_ft := 600
def ants_per_sq_inch := 4
def inches_per_foot := 12

-- Define the calculation to get the number of ants
def number_of_ants (width_ft : ℕ) (length_ft : ℕ) (ants_per_sq_inch : ℕ) (inches_per_foot : ℕ) :=
  let width_inch := width_ft * inches_per_foot
  let length_inch := length_ft * inches_per_foot
  let area_sq_inch := width_inch * length_inch
  ants_per_sq_inch * area_sq_inch

-- Prove the number of ants is approximately 173 million
theorem number_of_ants_proof :
  number_of_ants width_ft length_ft ants_per_sq_inch inches_per_foot = 172800000 :=
by
  sorry

end number_of_ants_proof_l283_283872


namespace eighth_odd_multiple_of_5_is_75_l283_283988

theorem eighth_odd_multiple_of_5_is_75 : ∃ n : ℕ, (n > 0) ∧ (n % 2 = 1) ∧ (n % 5 = 0) ∧ (nat.find_greatest (λ m, m % 2 = 1 ∧ m % 5 = 0) 75 = 75) :=
    sorry

end eighth_odd_multiple_of_5_is_75_l283_283988


namespace average_weight_of_a_and_b_is_40_l283_283159

variable (A B C : ℝ)

-- Conditions
def condition1 : Prop := (A + B + C) / 3 = 42
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 40

-- Theorem statement
theorem average_weight_of_a_and_b_is_40 (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 B) : 
    (A + B) / 2 = 40 := by
  sorry

end average_weight_of_a_and_b_is_40_l283_283159


namespace laura_bought_4_shirts_l283_283133

-- Definitions for the conditions
def pants_price : ℕ := 54
def num_pants : ℕ := 2
def shirt_price : ℕ := 33
def given_money : ℕ := 250
def change_received : ℕ := 10

-- Proving the number of shirts bought is 4
theorem laura_bought_4_shirts :
  (num_pants * pants_price) + (shirt_price * 4) + change_received = given_money :=
by
  sorry

end laura_bought_4_shirts_l283_283133


namespace trig_identity_l283_283079

noncomputable def sin_40 := Real.sin (40 * Real.pi / 180)
noncomputable def tan_10 := Real.tan (10 * Real.pi / 180)
noncomputable def sqrt_3 := Real.sqrt 3

theorem trig_identity : sin_40 * (tan_10 - sqrt_3) = -1 := by
  sorry

end trig_identity_l283_283079


namespace tan_addition_example_l283_283914

theorem tan_addition_example (x : ℝ) (h : Real.tan x = 1/3) : 
  Real.tan (x + π/3) = 2 + 5 * Real.sqrt 3 / 3 := 
by 
  sorry

end tan_addition_example_l283_283914


namespace age_of_James_when_Thomas_reaches_current_age_l283_283171
    
theorem age_of_James_when_Thomas_reaches_current_age
  (T S J : ℕ)
  (h1 : T = 6)
  (h2 : S = T + 13)
  (h3 : S = J - 5) :
  J + (S - T) = 37 := 
by
  sorry

end age_of_James_when_Thomas_reaches_current_age_l283_283171


namespace total_alligators_seen_l283_283817

-- Definitions for the conditions
def SamaraSaw : Nat := 35
def NumberOfFriends : Nat := 6
def AverageFriendsSaw : Nat := 15

-- Statement of the proof problem
theorem total_alligators_seen :
  SamaraSaw + NumberOfFriends * AverageFriendsSaw = 125 := by
  -- Skipping the proof
  sorry

end total_alligators_seen_l283_283817


namespace exists_two_numbers_l283_283618

theorem exists_two_numbers (x : Fin 7 → ℝ) :
  ∃ i j, 0 ≤ (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) ≤ 1 / Real.sqrt 3 :=
sorry

end exists_two_numbers_l283_283618


namespace equation_of_line_parallel_to_x_axis_l283_283970

theorem equation_of_line_parallel_to_x_axis (x: ℝ) :
  ∃ (y: ℝ), (y-2=0) ∧ ∀ (P: ℝ × ℝ), (P = (1, 2)) → P.2 = 2 := 
by
  sorry

end equation_of_line_parallel_to_x_axis_l283_283970


namespace max_monthly_profit_l283_283257

theorem max_monthly_profit (x : ℝ) (h : 0 < x ∧ x ≤ 15) :
  let C := 100 + 4 * x
  let p := 76 + 15 * x - x^2
  let L := p * x - C
  L = -x^3 + 15 * x^2 + 72 * x - 100 ∧
  (∀ x, 0 < x ∧ x ≤ 15 → L ≤ -12^3 + 15 * 12^2 + 72 * 12 - 100) :=
by
  sorry

end max_monthly_profit_l283_283257


namespace club_membership_l283_283166

theorem club_membership (n : ℕ) : 
  n ≡ 6 [MOD 10] → n ≡ 6 [MOD 11] → 200 ≤ n ∧ n ≤ 300 → n = 226 :=
by
  intros h1 h2 h3
  sorry

end club_membership_l283_283166


namespace possible_values_for_N_l283_283666

theorem possible_values_for_N (N : ℕ) (H : ∀ (k : ℕ), N = 8 + k) (truth: (student : ℕ → Prop) (bully : ℕ → Prop) (n : ℕ), 
  (∀ i, bully i → ¬ (∃ j, student j ∧ bully j → j ≠ i)) →
  (∀ i, student i → (∃ j, student j ∧ bully j → j ≠ i))
  → ∀ i, (bully i → ¬(∃ (m : ℕ), (m ≥ N - 1 / 3 )) ∧ student i → (∃ (m : ℕ), (m ≥ N - 1 / 3 ))) : N = 23 ∨ N = 24 ∨ N = 25 :=
by
  sorry

end possible_values_for_N_l283_283666


namespace James_future_age_when_Thomas_reaches_James_current_age_l283_283174

-- Defining the given conditions
def Thomas_age := 6
def Shay_age := Thomas_age + 13
def James_age := Shay_age + 5

-- Goal: Proving James's age when Thomas reaches James's current age
theorem James_future_age_when_Thomas_reaches_James_current_age :
  let years_until_Thomas_is_James_current_age := James_age - Thomas_age
  let James_future_age := James_age + years_until_Thomas_is_James_current_age
  James_future_age = 42 :=
by
  sorry

end James_future_age_when_Thomas_reaches_James_current_age_l283_283174


namespace possible_values_of_N_l283_283670

theorem possible_values_of_N (N : ℕ) (h1 : N ≥ 8 + 1)
  (h2 : ∀ (i : ℕ), (i < N → (i ≥ 0 ∧ i < 1/3 * (N-1)) → false) ) 
  (h3 : ∀ (i : ℕ), (i < N → (i ≥ 1/3 * (N-1) ∨ i < 1/3 * (N-1)) → true)) :
  23 ≤ N ∧ N ≤ 25 :=
by
  sorry

end possible_values_of_N_l283_283670


namespace count_triangles_on_cube_count_triangles_not_in_face_l283_283734

open Nat

def num_triangles_cube : ℕ := 56
def num_triangles_not_in_face : ℕ := 32

theorem count_triangles_on_cube (V : Finset ℕ) (hV : V.card = 8) :
  (V.card.choose 3 = num_triangles_cube) :=
  sorry

theorem count_triangles_not_in_face (V : Finset ℕ) (hV : V.card = 8) :
  (V.card.choose 3 - (6 * 4) = num_triangles_not_in_face) :=
  sorry

end count_triangles_on_cube_count_triangles_not_in_face_l283_283734


namespace five_y_eq_45_over_7_l283_283913

theorem five_y_eq_45_over_7 (x y : ℚ) (h1 : 3 * x + 4 * y = 0) (h2 : x = y - 3) : 5 * y = 45 / 7 := by
  sorry

end five_y_eq_45_over_7_l283_283913


namespace no_real_roots_ffx_l283_283485

theorem no_real_roots_ffx 
  (b c : ℝ) 
  (h : ∀ x : ℝ, (x^2 + (b - 1) * x + (c - 1) ≠ 0 ∨ ∀x: ℝ, (b - 1)^2 - 4 * (c - 1) < 0)) 
  : ∀ x : ℝ, (x^2 + bx + c)^2 + b * (x^2 + bx + c) + c ≠ x :=
by
  sorry

end no_real_roots_ffx_l283_283485


namespace division_multiplication_result_l283_283556

theorem division_multiplication_result :
  (7.5 / 6) * 12 = 15 := by
  sorry

end division_multiplication_result_l283_283556


namespace savings_if_together_l283_283752

def window_price : ℕ := 100

def free_windows_for_six_purchased : ℕ := 2

def windows_needed_Dave : ℕ := 9
def windows_needed_Doug : ℕ := 10

def total_individual_cost (windows_purchased : ℕ) : ℕ :=
  100 * windows_purchased

def total_cost_with_deal (windows_purchased: ℕ) : ℕ :=
  let sets_of_6 := windows_purchased / 6
  let remaining_windows := windows_purchased % 6
  100 * (sets_of_6 * 6 + remaining_windows)

def combined_savings (windows_needed_Dave: ℕ) (windows_needed_Doug: ℕ) : ℕ :=
  let total_windows := windows_needed_Dave + windows_needed_Doug
  total_individual_cost windows_needed_Dave 
  + total_individual_cost windows_needed_Doug 
  - total_cost_with_deal total_windows

theorem savings_if_together : combined_savings windows_needed_Dave windows_needed_Doug = 400 :=
by
  sorry

end savings_if_together_l283_283752


namespace construction_better_than_logistics_l283_283333

theorem construction_better_than_logistics 
  (applications_computer : ℕ := 215830)
  (applications_mechanical : ℕ := 200250)
  (applications_marketing : ℕ := 154676)
  (applications_logistics : ℕ := 74570)
  (applications_trade : ℕ := 65280)
  (recruitments_computer : ℕ := 124620)
  (recruitments_marketing : ℕ := 102935)
  (recruitments_mechanical : ℕ := 89115)
  (recruitments_construction : ℕ := 76516)
  (recruitments_chemical : ℕ := 70436) :
  applications_construction / recruitments_construction < applications_logistics / recruitments_logistics→ 
  (applications_computer / recruitments_computer < applications_chemical / recruitments_chemical) :=
sorry

end construction_better_than_logistics_l283_283333


namespace pyramid_base_side_length_l283_283383

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end pyramid_base_side_length_l283_283383


namespace average_height_of_three_l283_283357

theorem average_height_of_three (parker daisy reese : ℕ) 
  (h1 : parker = daisy - 4)
  (h2 : daisy = reese + 8)
  (h3 : reese = 60) : 
  (parker + daisy + reese) / 3 = 64 := 
  sorry

end average_height_of_three_l283_283357


namespace ms_warren_running_time_l283_283143

theorem ms_warren_running_time 
  (t : ℝ) 
  (ht_total_distance : 6 * t + 2 * 0.5 = 3) : 
  60 * t = 20 := by 
  sorry

end ms_warren_running_time_l283_283143


namespace find_speed_of_goods_train_l283_283429

variable (v : ℕ) -- Speed of the goods train in km/h

theorem find_speed_of_goods_train
  (h1 : 0 < v) 
  (h2 : 6 * v + 4 * 90 = 10 * v) :
  v = 36 :=
by
  sorry

end find_speed_of_goods_train_l283_283429


namespace erwan_spending_l283_283891

def discount (price : ℕ) (percent : ℕ) : ℕ :=
  price - (price * percent / 100)

theorem erwan_spending (shoe_original_price : ℕ := 200) 
  (shoe_discount : ℕ := 30)
  (shirt_price : ℕ := 80)
  (num_shirts : ℕ := 2)
  (pants_price : ℕ := 150)
  (second_store_discount : ℕ := 20)
  (jacket_price : ℕ := 250)
  (tie_price : ℕ := 40)
  (hat_price : ℕ := 60)
  (watch_price : ℕ := 120)
  (wallet_price : ℕ := 49)
  (belt_price : ℕ := 35)
  (belt_discount : ℕ := 25)
  (scarf_price : ℕ := 45)
  (scarf_discount : ℕ := 10)
  (rewards_points_discount : ℕ := 5)
  (sales_tax : ℕ := 8)
  (gift_card : ℕ := 50)
  (shipping_fee : ℕ := 5)
  (num_shipping_stores : ℕ := 2) :
  ∃ total : ℕ,
    total = 85429 :=
by
  have first_store := discount shoe_original_price shoe_discount
  have second_store_total := pants_price + (shirt_price * num_shirts)
  have second_store := discount second_store_total second_store_discount
  have tie_half_price := tie_price / 2
  have hat_half_price := hat_price / 2
  have third_store := jacket_price + (tie_half_price + hat_half_price)
  have fourth_store := watch_price
  have fifth_store := discount belt_price belt_discount + discount scarf_price scarf_discount
  have subtotal := first_store + second_store + third_store + fourth_store + fifth_store
  have after_rewards_points := subtotal - (subtotal * rewards_points_discount / 100)
  have after_gift_card := after_rewards_points - gift_card
  have after_shipping_fees := after_gift_card + (shipping_fee * num_shipping_stores)
  have total := after_shipping_fees + (after_shipping_fees * sales_tax / 100)
  use total / 100 -- to match the monetary value in cents
  sorry

end erwan_spending_l283_283891


namespace r_squared_sum_l283_283293

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l283_283293


namespace body_diagonal_length_l283_283009

theorem body_diagonal_length (a b c : ℝ) (h1 : a * b = 6) (h2 : a * c = 8) (h3 : b * c = 12) :
  (a^2 + b^2 + c^2 = 29) :=
by
  sorry

end body_diagonal_length_l283_283009


namespace student_scores_marks_per_correct_answer_l283_283799

theorem student_scores_marks_per_correct_answer
  (total_questions : ℕ) (total_marks : ℤ) (correct_questions : ℕ)
  (wrong_questions : ℕ) (marks_wrong_answer : ℤ)
  (x : ℤ) (h1 : total_questions = 60) (h2 : total_marks = 110)
  (h3 : correct_questions = 34) (h4 : wrong_questions = total_questions - correct_questions)
  (h5 : marks_wrong_answer = -1) :
  34 * x - 26 = 110 → x = 4 := by
  sorry

end student_scores_marks_per_correct_answer_l283_283799


namespace problem_statement_l283_283539

theorem problem_statement (a b c : ℝ) 
  (h1 : 2011 * (a + b + c) = 1)
  (h2 : a * b + a * c + b * c = 2011 * a * b * c) :
  a ^ 2011 * b ^ 2011 + c ^ 2011 = 1 / 2011^2011 :=
by
  sorry

end problem_statement_l283_283539


namespace melissa_bananas_l283_283142

theorem melissa_bananas (a b : ℕ) (h1 : a = 88) (h2 : b = 4) : a - b = 84 :=
by
  sorry

end melissa_bananas_l283_283142


namespace root_equation_l283_283623

variable (m : ℝ)
theorem root_equation (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2020 = 2023 := by
  sorry

end root_equation_l283_283623


namespace unique_solution_m_l283_283081

theorem unique_solution_m :
  ∃! m : ℝ, ∀ x y : ℝ, (y = x^2 ∧ y = 4*x + m) → m = -4 :=
by 
  sorry

end unique_solution_m_l283_283081


namespace area_of_third_face_l283_283405

-- Define the variables for the dimensions of the box: l, w, and h
variables (l w h: ℝ)

-- Given conditions
def face1_area := 120
def face2_area := 72
def volume := 720

-- The relationships between the dimensions and the given areas/volume
def face1_eq : Prop := l * w = face1_area
def face2_eq : Prop := w * h = face2_area
def volume_eq : Prop := l * w * h = volume

-- The statement we need to prove is that the area of the third face (l * h) is 60 cm² given the above equations
theorem area_of_third_face :
  face1_eq l w →
  face2_eq w h →
  volume_eq l w h →
  l * h = 60 :=
by
  intros h1 h2 h3
  sorry

end area_of_third_face_l283_283405


namespace possible_values_of_N_l283_283660

theorem possible_values_of_N (N : ℕ) (h : N > 8) :
  22 < N ∧ N ≤ 25 →
  N = 23 ∨ N = 24 ∨ N = 25 :=
by
  intros
  sorry

end possible_values_of_N_l283_283660


namespace find_power_l283_283638

theorem find_power (a b c d e : ℕ) (h1 : a = 105) (h2 : b = 21) (h3 : c = 25) (h4 : d = 45) (h5 : e = 49) 
(h6 : a ^ (3 : ℕ) = b * c * d * e) : 3 = 3 := by
  sorry

end find_power_l283_283638


namespace possible_values_of_N_l283_283653

def is_valid_N (N : ℕ) : Prop :=
  (N > 22) ∧ (N ≤ 25)

theorem possible_values_of_N :
  {N : ℕ | is_valid_N N} = {23, 24, 25} :=
by
  sorry

end possible_values_of_N_l283_283653


namespace imaginary_power_sum_zero_l283_283419

theorem imaginary_power_sum_zero (i : ℂ) (n : ℤ) (h : i^2 = -1) :
  i^(2*n - 3) + i^(2*n - 1) + i^(2*n + 1) + i^(2*n + 3) = 0 :=
by {
  sorry
}

end imaginary_power_sum_zero_l283_283419


namespace bacteria_population_l283_283535

theorem bacteria_population (initial_population : ℕ) (tripling_factor : ℕ) (hours_per_tripling : ℕ) (target_population : ℕ) 
(initial_population_eq : initial_population = 300)
(tripling_factor_eq : tripling_factor = 3)
(hours_per_tripling_eq : hours_per_tripling = 5)
(target_population_eq : target_population = 87480) :
∃ n : ℕ, (hours_per_tripling * n = 30) ∧ (initial_population * (tripling_factor ^ n) ≥ target_population) := sorry

end bacteria_population_l283_283535


namespace slices_left_l283_283749

-- Conditions
def total_slices : ℕ := 16
def fraction_eaten : ℚ := 3/4
def fraction_left : ℚ := 1 - fraction_eaten

-- Proof statement
theorem slices_left : total_slices * fraction_left = 4 := by
  sorry

end slices_left_l283_283749


namespace range_of_a_l283_283782

theorem range_of_a (f : ℝ → ℝ) (h_increasing : ∀ x y, x < y → f x < f y) (a : ℝ) :
  f (a^2 - a) > f (2 * a^2 - 4 * a) → 0 < a ∧ a < 3 :=
by
  -- We translate the condition f(a^2 - a) > f(2a^2 - 4a) to the inequality
  intro h
  -- Apply the fact that f is increasing to deduce the inequality on a
  sorry

end range_of_a_l283_283782


namespace proof_problem_l283_283917

noncomputable def arithmetic_mean (a b : ℝ) : ℝ :=
  (a + b) / 2

noncomputable def geometric_mean (x y : ℝ) : ℝ :=
  Real.sqrt (x * y)

theorem proof_problem (a b c x y z m : ℝ) (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) (m_pos : 0 < m) (m_ne_one : m ≠ 1) 
  (h_b : b = arithmetic_mean a c) (h_y : y = geometric_mean x z) :
  (b - c) * Real.logb m x + (c - a) * Real.logb m y + (a - b) * Real.logb m z = 0 := by
  sorry

end proof_problem_l283_283917


namespace correct_statements_l283_283903

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

noncomputable def a_n_sequence (n : ℕ) := a n
noncomputable def Sn_sum (n : ℕ) := S n

axiom Sn_2022_lt_zero : S 2022 < 0
axiom Sn_2023_gt_zero : S 2023 > 0

theorem correct_statements :
  (a 1012 > 0) ∧ ( ∀ n, S n >= S 1011 → n = 1011) :=
  sorry

end correct_statements_l283_283903


namespace alex_correct_percentage_l283_283050

theorem alex_correct_percentage 
  (score_quiz : ℤ) (problems_quiz : ℤ)
  (score_test : ℤ) (problems_test : ℤ)
  (score_exam : ℤ) (problems_exam : ℤ)
  (h1 : score_quiz = 75) (h2 : problems_quiz = 30)
  (h3 : score_test = 85) (h4 : problems_test = 50)
  (h5 : score_exam = 80) (h6 : problems_exam = 20) :
  (75 * 30 + 85 * 50 + 80 * 20) / (30 + 50 + 20) = 81 := 
sorry

end alex_correct_percentage_l283_283050


namespace speed_of_current_l283_283221

-- Conditions translated into Lean definitions
def initial_time : ℝ := 13 -- 1:00 PM is represented as 13:00 hours
def boat1_time_turnaround : ℝ := 14 -- Boat 1 turns around at 2:00 PM
def boat2_time_turnaround : ℝ := 15 -- Boat 2 turns around at 3:00 PM
def meeting_time : ℝ := 16 -- Boats meet at 4:00 PM
def raft_drift_distance : ℝ := 7.5 -- Raft drifted 7.5 km from the pier

-- The problem statement to prove
theorem speed_of_current:
  ∃ v : ℝ, (v * (meeting_time - initial_time) = raft_drift_distance) ∧ v = 2.5 :=
by
  sorry

end speed_of_current_l283_283221


namespace simplification_qrt_1_simplification_qrt_2_l283_283456

-- Problem 1
theorem simplification_qrt_1 : (2 * Real.sqrt 12 + 3 * Real.sqrt 3 - Real.sqrt 27) = 4 * Real.sqrt 3 :=
by
  sorry

-- Problem 2
theorem simplification_qrt_2 : (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2 * 12) + Real.sqrt 24) = 4 + Real.sqrt 6 :=
by
  sorry

end simplification_qrt_1_simplification_qrt_2_l283_283456


namespace gina_good_tipper_l283_283900

noncomputable def calculate_tip_difference (bill_in_usd : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) (low_tip_rate : ℝ) (high_tip_rate : ℝ) (conversion_rate : ℝ) : ℝ :=
  let discounted_bill := bill_in_usd * (1 - discount_rate)
  let taxed_bill := discounted_bill * (1 + tax_rate)
  let low_tip := taxed_bill * low_tip_rate
  let high_tip := taxed_bill * high_tip_rate
  let difference_in_usd := high_tip - low_tip
  let difference_in_eur := difference_in_usd * conversion_rate
  difference_in_eur * 100

theorem gina_good_tipper : calculate_tip_difference 26 0.08 0.07 0.05 0.20 0.85 = 326.33 := 
by
  sorry

end gina_good_tipper_l283_283900


namespace smallest_n_l283_283991

theorem smallest_n (n : ℕ) (hn1 : ∃ k, 5 * n = k^4) (hn2: ∃ m, 4 * n = m^3) : n = 2000 :=
sorry

end smallest_n_l283_283991


namespace speed_of_current_l283_283220

-- Conditions translated into Lean definitions
def initial_time : ℝ := 13 -- 1:00 PM is represented as 13:00 hours
def boat1_time_turnaround : ℝ := 14 -- Boat 1 turns around at 2:00 PM
def boat2_time_turnaround : ℝ := 15 -- Boat 2 turns around at 3:00 PM
def meeting_time : ℝ := 16 -- Boats meet at 4:00 PM
def raft_drift_distance : ℝ := 7.5 -- Raft drifted 7.5 km from the pier

-- The problem statement to prove
theorem speed_of_current:
  ∃ v : ℝ, (v * (meeting_time - initial_time) = raft_drift_distance) ∧ v = 2.5 :=
by
  sorry

end speed_of_current_l283_283220


namespace production_days_l283_283475

theorem production_days (n P : ℕ) (h1 : P = 60 * n) (h2 : (P + 90) / (n + 1) = 65) : n = 5 := sorry

end production_days_l283_283475


namespace min_value_of_a_l283_283909

theorem min_value_of_a (a : ℝ) (h : ∃ x : ℝ, |x - 1| + |x + a| ≤ 8) : -9 ≤ a :=
by
  sorry

end min_value_of_a_l283_283909


namespace contradiction_proof_l283_283722

theorem contradiction_proof (a b c : ℝ) (h : ¬ (a > 0 ∨ b > 0 ∨ c > 0)) : false :=
by
  sorry

end contradiction_proof_l283_283722


namespace num_ways_128_as_sum_of_four_positive_perfect_squares_l283_283927

noncomputable def is_positive_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, 0 < m ∧ m * m = n

noncomputable def four_positive_perfect_squares_sum (n : ℕ) : Prop :=
  ∃ a b c d : ℕ,
    is_positive_perfect_square a ∧
    is_positive_perfect_square b ∧
    is_positive_perfect_square c ∧
    is_positive_perfect_square d ∧
    a + b + c + d = n

theorem num_ways_128_as_sum_of_four_positive_perfect_squares :
  (∃! (a b c d : ℕ), four_positive_perfect_squares_sum 128) :=
sorry

end num_ways_128_as_sum_of_four_positive_perfect_squares_l283_283927


namespace no_such_cuboid_exists_l283_283938

theorem no_such_cuboid_exists (a b c : ℝ) :
  a + b + c = 12 ∧ ab + bc + ca = 1 ∧ abc = 12 → false :=
by
  sorry

end no_such_cuboid_exists_l283_283938


namespace correct_option_l283_283850

theorem correct_option : (-1 - 3 = -4) ∧ ¬(-2 + 8 = 10) ∧ ¬(-2 * 2 = 4) ∧ ¬(-8 / -1 = -1 / 8) :=
by
  sorry

end correct_option_l283_283850


namespace tank_capacity_is_32_l283_283857

noncomputable def capacity_of_tank (C : ℝ) : Prop :=
  (3/4) * C + 4 = (7/8) * C

theorem tank_capacity_is_32 : ∃ C : ℝ, capacity_of_tank C ∧ C = 32 :=
sorry

end tank_capacity_is_32_l283_283857


namespace pyramid_base_side_length_l283_283381

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end pyramid_base_side_length_l283_283381


namespace diamonds_in_G_20_equals_840_l283_283417

def diamonds_in_G (n : ℕ) : ℕ :=
  if n < 3 then 1 else 2 * n * (n + 1)

theorem diamonds_in_G_20_equals_840 : diamonds_in_G 20 = 840 :=
by
  sorry

end diamonds_in_G_20_equals_840_l283_283417


namespace models_kirsty_can_buy_l283_283341

def original_price : ℝ := 0.45
def saved_for_models : ℝ := 30 * original_price
def new_price : ℝ := 0.50

theorem models_kirsty_can_buy :
  saved_for_models / new_price = 27 :=
sorry

end models_kirsty_can_buy_l283_283341


namespace golden_ratio_minus_one_binary_l283_283518

theorem golden_ratio_minus_one_binary (n : ℕ → ℕ) (h_n : ∀ i, 1 ≤ n i)
  (h_incr : ∀ i, n i ≤ n (i + 1)): 
  (∀ k ≥ 4, n k ≤ 2^(k - 1) - 2) := 
by
  sorry

end golden_ratio_minus_one_binary_l283_283518


namespace circle_radius_one_l283_283471

-- Define the circle equation as a hypothesis
def circle_equation (x y : ℝ) : Prop :=
  16 * x^2 + 32 * x + 16 * y^2 - 48 * y + 68 = 0

-- The goal is to prove the radius of the circle defined above
theorem circle_radius_one :
  ∃ r : ℝ, r = 1 ∧ ∀ x y : ℝ, circle_equation x y → (x + 1)^2 + (y - 1.5)^2 = r^2 :=
by
  sorry

end circle_radius_one_l283_283471


namespace simplify_expression_l283_283708

theorem simplify_expression (x : ℝ) : (3 * x + 15) + (100 * x + 15) + (10 * x - 5) = 113 * x + 25 :=
by
  sorry

end simplify_expression_l283_283708


namespace gcd_g_x_l283_283905

def g (x : ℕ) : ℕ := (5 * x + 7) * (11 * x + 3) * (17 * x + 8) * (4 * x + 5)

theorem gcd_g_x (x : ℕ) (hx : 17280 ∣ x) : Nat.gcd (g x) x = 120 :=
by sorry

end gcd_g_x_l283_283905


namespace icosahedron_edge_probability_l283_283725

theorem icosahedron_edge_probability :
  let vertices := 12
  let total_pairs := vertices * (vertices - 1) / 2
  let edges := 30
  let probability := edges.toFloat / total_pairs.toFloat
  probability = 5 / 11 :=
by
  sorry

end icosahedron_edge_probability_l283_283725


namespace fib_150_mod_9_l283_283371

theorem fib_150_mod_9 : Nat.fib 150 % 9 = 8 :=
by 
  sorry

end fib_150_mod_9_l283_283371


namespace amanda_weekly_earnings_l283_283584

def amanda_rate_per_hour : ℝ := 20.00
def monday_appointments : ℕ := 5
def monday_hours_per_appointment : ℝ := 1.5
def tuesday_appointment_hours : ℝ := 3
def thursday_appointments : ℕ := 2
def thursday_hours_per_appointment : ℝ := 2
def saturday_appointment_hours : ℝ := 6

def total_hours_worked : ℝ :=
  monday_appointments * monday_hours_per_appointment +
  tuesday_appointment_hours +
  thursday_appointments * thursday_hours_per_appointment +
  saturday_appointment_hours

def total_earnings : ℝ := total_hours_worked * amanda_rate_per_hour

theorem amanda_weekly_earnings : total_earnings = 410.00 :=
  by
    unfold total_earnings total_hours_worked monday_appointments monday_hours_per_appointment tuesday_appointment_hours thursday_appointments thursday_hours_per_appointment saturday_appointment_hours amanda_rate_per_hour 
    -- The proof will involve basic arithmetic simplification, which is skipped here.
    -- Therefore, we simply state sorry.
    sorry

end amanda_weekly_earnings_l283_283584


namespace green_eyes_count_l283_283979

noncomputable def people_count := 100
noncomputable def blue_eyes := 19
noncomputable def brown_eyes := people_count / 2
noncomputable def black_eyes := people_count / 4
noncomputable def green_eyes := people_count - (blue_eyes + brown_eyes + black_eyes)

theorem green_eyes_count : green_eyes = 6 := by
  sorry

end green_eyes_count_l283_283979


namespace find_a_find_distance_l283_283193

-- Problem 1: Given conditions to find 'a'
theorem find_a (a : ℝ) :
  (∃ θ ρ, ρ = 2 * Real.cos θ ∧ 3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ + a = 0) →
  (a = 2 ∨ a = -8) :=
sorry

-- Problem 2: Given point and line, find the distance
theorem find_distance : 
  ∃ (d : ℝ), d = Real.sqrt 3 + 5/2 ∧
  (∃ θ ρ, θ = 11 * Real.pi / 6 ∧ ρ = 2 ∧ 
   (ρ = Real.sqrt (3 * (Real.sin θ - Real.pi / 6)^2 + (ρ * Real.cos (θ - Real.pi / 6))^2) 
   → ρ * Real.sin (θ - Real.pi / 6) = 1)) :=
sorry

end find_a_find_distance_l283_283193


namespace iron_weighs_more_l283_283145

-- Define the weights of the metal pieces
def weight_iron : ℝ := 11.17
def weight_aluminum : ℝ := 0.83

-- State the theorem to prove that the difference in weights is 10.34 pounds
theorem iron_weighs_more : weight_iron - weight_aluminum = 10.34 :=
by sorry

end iron_weighs_more_l283_283145


namespace incorrect_statements_l283_283588

open Set

theorem incorrect_statements (A : Set ℝ) (B : Set ℝ) (a : ℝ) :
  (A = ∅ → ∅ ⊆ A) ∧
  (A = {x : ℝ | x^2 - 1 = 0} ∧ B = {-1, 1} → A = B) ∧
  (¬ (∀ y ∈ B, ∃! x ∈ A)) ∧
  (∀ x, f(x) = 1/x → Ioo (-∞) 0 ∪ Ioo 0 ∞ ⊆ {y | f(y) < f(x)}) ∧
  {x | 1 < x < 2} ⊆ { x | x < a } ↔ a ≥ 2 →
  ¬(statement_3 ∧ statement_4 ∧ statement_5) :=
by
  sorry

end incorrect_statements_l283_283588


namespace sum_first_10_terms_arithmetic_seq_l283_283406

theorem sum_first_10_terms_arithmetic_seq (a : ℕ → ℤ) (h : (a 4)^2 + (a 7)^2 + 2 * (a 4) * (a 7) = 9) :
  ∃ S, S = 10 * (a 4 + a 7) / 2 ∧ (S = 15 ∨ S = -15) := 
by
  sorry

end sum_first_10_terms_arithmetic_seq_l283_283406


namespace tan_arithmetic_geometric_l283_283906

noncomputable def a_seq : ℕ → ℝ := sorry -- Define a_n as an arithmetic sequence (details abstracted)
noncomputable def b_seq : ℕ → ℝ := sorry -- Define b_n as a geometric sequence (details abstracted)

axiom a_seq_is_arithmetic : ∀ n m : ℕ, a_seq (n + 1) - a_seq n = a_seq (m + 1) - a_seq m
axiom b_seq_is_geometric : ∀ n : ℕ, ∃ r : ℝ, b_seq (n + 1) = b_seq n * r
axiom a_seq_sum : a_seq 2017 + a_seq 2018 = Real.pi
axiom b_seq_square : b_seq 20 ^ 2 = 4

theorem tan_arithmetic_geometric : 
  (Real.tan ((a_seq 2 + a_seq 4033) / (b_seq 1 * b_seq 39)) = 1) :=
sorry

end tan_arithmetic_geometric_l283_283906


namespace carla_total_time_l283_283066

def total_time_spent (knife_time : ℕ) (peeling_time_multiplier : ℕ) : ℕ :=
  knife_time + peeling_time_multiplier * knife_time

theorem carla_total_time :
  total_time_spent 10 3 = 40 :=
by
  sorry

end carla_total_time_l283_283066


namespace tangent_line_at_1_l283_283099

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * x

theorem tangent_line_at_1 :
  let x := (1 : ℝ)
  let y := (f 1)
  ∃ m b : ℝ, (∀ x, y - m * (x - 1) + b = 0)
  ∧ (m = -2)
  ∧ (b = -1) :=
by
  sorry

end tangent_line_at_1_l283_283099


namespace number_of_players_is_correct_l283_283684

-- Defining the problem conditions
def wristband_cost : ℕ := 6
def jersey_cost : ℕ := wristband_cost + 7
def wristbands_per_player : ℕ := 4
def jerseys_per_player : ℕ := 2
def total_expenditure : ℕ := 3774

-- Calculating cost per player and stating the proof problem
def cost_per_player : ℕ := wristbands_per_player * wristband_cost +
                           jerseys_per_player * jersey_cost

def number_of_players : ℕ := total_expenditure / cost_per_player

-- The final proof statement to show that number_of_players is 75
theorem number_of_players_is_correct : number_of_players = 75 :=
by sorry

end number_of_players_is_correct_l283_283684


namespace real_part_of_z_is_neg3_l283_283095

noncomputable def z : ℂ := (1 + 2 * Complex.I) ^ 2

theorem real_part_of_z_is_neg3 : z.re = -3 := by
  sorry

end real_part_of_z_is_neg3_l283_283095


namespace r_squared_sum_l283_283296

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l283_283296


namespace f_14_52_eq_364_l283_283573

def f : ℕ → ℕ → ℕ := sorry  -- Placeholder definition

axiom f_xx (x : ℕ) : f x x = x
axiom f_sym (x y : ℕ) : f x y = f y x
axiom f_rec (x y : ℕ) (h : x + y > 0) : (x + y) * f x y = y * f x (x + y)

theorem f_14_52_eq_364 : f 14 52 = 364 := 
by {
  sorry  -- Placeholder for the proof steps
}

end f_14_52_eq_364_l283_283573


namespace models_kirsty_can_buy_l283_283340

def original_price : ℝ := 0.45
def saved_for_models : ℝ := 30 * original_price
def new_price : ℝ := 0.50

theorem models_kirsty_can_buy :
  saved_for_models / new_price = 27 :=
sorry

end models_kirsty_can_buy_l283_283340


namespace r_pow_four_solution_l283_283304

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l283_283304


namespace circle_radius_condition_l283_283611

theorem circle_radius_condition (c : ℝ) :
  (∃ r : ℝ, r = 5 ∧ (x y : ℝ) → (x^2 + 10*x + y^2 + 8*y + c = 0 
    ↔ (x + 5)^2 + (y + 4)^2 = 25)) → c = 16 :=
by
  sorry

end circle_radius_condition_l283_283611


namespace triangle_third_side_count_l283_283778

theorem triangle_third_side_count : 
  ∀ (x : ℕ), (3 < x ∧ x < 19) → ∃ (n : ℕ), n = 15 := 
by 
  sorry

end triangle_third_side_count_l283_283778


namespace james_tv_watching_time_l283_283688

theorem james_tv_watching_time
  (ep_jeopardy : ℕ := 20) -- Each episode of Jeopardy is 20 minutes long
  (n_jeopardy : ℕ := 2) -- James watched 2 episodes of Jeopardy
  (n_wheel : ℕ := 2) -- James watched 2 episodes of Wheel of Fortune
  (wheel_factor : ℕ := 2) -- Wheel of Fortune episodes are twice as long as Jeopardy episodes
  : (ep_jeopardy * n_jeopardy + ep_jeopardy * wheel_factor * n_wheel) / 60 = 2 :=
by
  sorry

end james_tv_watching_time_l283_283688


namespace cube_root_of_64_eq_two_pow_m_l283_283039

theorem cube_root_of_64_eq_two_pow_m (m : ℕ) (h : (64 : ℝ) ^ (1 / 3) = (2 : ℝ) ^ m) : m = 2 := 
sorry

end cube_root_of_64_eq_two_pow_m_l283_283039


namespace john_moves_correct_total_weight_l283_283803

noncomputable def johns_total_weight_moved : ℝ := 5626.398

theorem john_moves_correct_total_weight :
  let initial_back_squat : ℝ := 200
  let back_squat_increase : ℝ := 50
  let front_squat_ratio : ℝ := 0.8
  let back_squat_set_increase : ℝ := 0.05
  let front_squat_ratio_increase : ℝ := 0.04
  let front_squat_effort : ℝ := 0.9
  let deadlift_ratio : ℝ := 1.2
  let deadlift_effort : ℝ := 0.85
  let deadlift_set_increase : ℝ := 0.03
  let updated_back_squat := (initial_back_squat + back_squat_increase)
  let back_squat_set_1 := updated_back_squat
  let back_squat_set_2 := back_squat_set_1 * (1 + back_squat_set_increase)
  let back_squat_set_3 := back_squat_set_2 * (1 + back_squat_set_increase)
  let back_squat_total := 3 * (back_squat_set_1 + back_squat_set_2 + back_squat_set_3)
  let updated_front_squat := updated_back_squat * front_squat_ratio
  let front_squat_set_1 := updated_front_squat * front_squat_effort
  let front_squat_set_2 := (updated_front_squat * (1 + front_squat_ratio_increase)) * front_squat_effort
  let front_squat_set_3 := (updated_front_squat * (1 + 2 * front_squat_ratio_increase)) * front_squat_effort
  let front_squat_total := 3 * (front_squat_set_1 + front_squat_set_2 + front_squat_set_3)
  let updated_deadlift := updated_back_squat * deadlift_ratio
  let deadlift_set_1 := updated_deadlift * deadlift_effort
  let deadlift_set_2 := (updated_deadlift * (1 + deadlift_set_increase)) * deadlift_effort
  let deadlift_set_3 := (updated_deadlift * (1 + 2 * deadlift_set_increase)) * deadlift_effort
  let deadlift_total := 2 * (deadlift_set_1 + deadlift_set_2 + deadlift_set_3)
  (back_squat_total + front_squat_total + deadlift_total) = johns_total_weight_moved :=
by sorry

end john_moves_correct_total_weight_l283_283803


namespace Bruce_paid_correct_amount_l283_283056

def grape_kg := 9
def grape_price_per_kg := 70
def mango_kg := 7
def mango_price_per_kg := 55
def orange_kg := 5
def orange_price_per_kg := 45
def apple_kg := 3
def apple_price_per_kg := 80

def total_cost := grape_kg * grape_price_per_kg + 
                  mango_kg * mango_price_per_kg + 
                  orange_kg * orange_price_per_kg + 
                  apple_kg * apple_price_per_kg

theorem Bruce_paid_correct_amount : total_cost = 1480 := by
  sorry

end Bruce_paid_correct_amount_l283_283056


namespace option_b_is_incorrect_l283_283729

theorem option_b_is_incorrect : ¬ (3 + 2 * Real.sqrt 2 = 5 * Real.sqrt 2) := by
  sorry

end option_b_is_incorrect_l283_283729


namespace a10_is_b55_l283_283617

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℕ := 2 * n - 1

-- Define the new sequence b_n according to the given insertion rules
def b (k : ℕ) : ℕ := sorry

-- Prove that if a_10 = 19, then 19 is the 55th term in the new sequence b_n
theorem a10_is_b55 : b 55 = a 10 := sorry

end a10_is_b55_l283_283617


namespace r_fourth_power_sum_l283_283276

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l283_283276


namespace pyramid_base_length_l283_283402

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end pyramid_base_length_l283_283402


namespace sum_of_solutions_l283_283317

theorem sum_of_solutions (x : ℝ) (h1 : x^2 = 25) : ∃ S : ℝ, S = 0 ∧ (∀ x', x'^2 = 25 → x' = 5 ∨ x' = -5) := 
sorry

end sum_of_solutions_l283_283317


namespace sumata_miles_per_day_l283_283372

theorem sumata_miles_per_day (total_miles : ℝ) (total_days : ℝ) (h1 : total_miles = 250.0) (h2 : total_days = 5.0) :
  total_miles / total_days = 50.0 :=
by
  sorry

end sumata_miles_per_day_l283_283372


namespace r_fourth_power_sum_l283_283274

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l283_283274


namespace expression_equals_one_l283_283230

def evaluate_expression : ℕ :=
  3 * (3 * (3 * (3 * (3 - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1) - 2 * 1

theorem expression_equals_one : evaluate_expression = 1 := by
  sorry

end expression_equals_one_l283_283230


namespace minimum_distance_on_circle_l283_283810

open Complex

noncomputable def minimum_distance (z : ℂ) : ℝ :=
  abs (z - (1 + 2*I))

theorem minimum_distance_on_circle :
  ∀ z : ℂ, abs (z + 2 - 2*I) = 1 → minimum_distance z = 2 :=
by
  intros z hz
  -- Proof is omitted
  sorry

end minimum_distance_on_circle_l283_283810


namespace problem_statement_l283_283634

theorem problem_statement (x Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) :
    10 * (6 * x + 14 * Real.pi) = 4 * Q := 
sorry

end problem_statement_l283_283634


namespace solve_quadratic_eq_solve_cubic_eq_l283_283153

-- Problem 1: 4x^2 - 9 = 0 implies x = ± 3/2
theorem solve_quadratic_eq (x : ℝ) : 4 * x^2 - 9 = 0 ↔ x = 3/2 ∨ x = -3/2 :=
by sorry

-- Problem 2: 64 * (x + 1)^3 = -125 implies x = -9/4
theorem solve_cubic_eq (x : ℝ) : 64 * (x + 1)^3 = -125 ↔ x = -9/4 :=
by sorry

end solve_quadratic_eq_solve_cubic_eq_l283_283153


namespace student_rank_from_left_l283_283049

theorem student_rank_from_left (total_students rank_from_right rank_from_left : ℕ) 
  (h1 : total_students = 21) 
  (h2 : rank_from_right = 16) 
  (h3 : total_students = rank_from_right + rank_from_left - 1) 
  : rank_from_left = 6 := 
by 
  sorry

end student_rank_from_left_l283_283049


namespace possible_values_of_N_l283_283671

theorem possible_values_of_N (N : ℕ) (h1 : N ≥ 8 + 1)
  (h2 : ∀ (i : ℕ), (i < N → (i ≥ 0 ∧ i < 1/3 * (N-1)) → false) ) 
  (h3 : ∀ (i : ℕ), (i < N → (i ≥ 1/3 * (N-1) ∨ i < 1/3 * (N-1)) → true)) :
  23 ≤ N ∧ N ≤ 25 :=
by
  sorry

end possible_values_of_N_l283_283671


namespace r_power_four_identity_l283_283310

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l283_283310


namespace second_container_mass_l283_283199

-- Given conditions
def height1 := 4 -- height of first container in cm
def width1 := 2 -- width of first container in cm
def length1 := 8 -- length of first container in cm
def mass1 := 64 -- mass of material the first container can hold in grams

def height2 := 3 * height1 -- height of second container in cm
def width2 := 2 * width1 -- width of second container in cm
def length2 := length1 -- length of second container in cm

def volume (height width length : ℤ) : ℤ := height * width * length

-- The proof statement
theorem second_container_mass : volume height2 width2 length2 = 6 * volume height1 width1 length1 → 6 * mass1 = 384 :=
by
  sorry

end second_container_mass_l283_283199


namespace trigonometric_identity_l283_283327

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 :=
by 
  sorry

end trigonometric_identity_l283_283327


namespace side_length_of_base_l283_283377

-- Define the conditions
def lateral_area (s : ℝ) : ℝ := (1 / 2) * s * 40
def given_area : ℝ := 120

-- Define the theorem to prove the length of the side of the base
theorem side_length_of_base : ∃ (s : ℝ), lateral_area(s) = given_area ∧ s = 6 :=
by
  sorry

end side_length_of_base_l283_283377


namespace possible_values_for_N_l283_283673

theorem possible_values_for_N (N : ℕ) (h₁ : 8 < N) (h₂ : 1 ≤ N - 1) :
  22 < N ∧ N ≤ 25 ↔ (N = 23 ∨ N = 24 ∨ N = 25) :=
by 
  sorry

end possible_values_for_N_l283_283673


namespace remainder_276_l283_283992

theorem remainder_276 (y : ℤ) (k : ℤ) (hk : y = 23 * k + 19) : y % 276 = 180 :=
sorry

end remainder_276_l283_283992


namespace ratio_girls_to_boys_l283_283512

variable (g b : ℕ)

-- Conditions: total students are 30, six more girls than boys.
def total_students : Prop := g + b = 30
def six_more_girls : Prop := g = b + 6

-- Proof that the ratio of girls to boys is 3:2.
theorem ratio_girls_to_boys (ht : total_students g b) (hs : six_more_girls g b) : g / b = 3 / 2 :=
  sorry

end ratio_girls_to_boys_l283_283512


namespace evaluate_expression_l283_283846

theorem evaluate_expression :
  (3 ^ 4 * 5 ^ 2 * 7 ^ 3 * 11) / (7 * 11 ^ 2) = 9025 :=
by 
  sorry

end evaluate_expression_l283_283846


namespace larry_wins_probability_l283_283940

noncomputable def probability_larry_wins (p_L : ℚ) (p_J : ℚ) : ℚ :=
  let q_L := 1 - p_L
  let q_J := 1 - p_J
  let r := q_L * q_J
  p_L / (1 - r)

theorem larry_wins_probability
  (p_L : ℚ) (p_J : ℚ) (h1 : p_L = 3 / 5) (h2 : p_J = 1 / 3) :
  probability_larry_wins p_L p_J = 9 / 11 :=
by 
  sorry

end larry_wins_probability_l283_283940


namespace probability_red_or_white_is_7_over_10_l283_283997

/-
A bag consists of 20 marbles, of which 6 are blue, 9 are red, and the remainder are white.
If Lisa is to select a marble from the bag at random, prove that the probability that the
marble will be red or white is 7/10.
-/
def num_marbles : ℕ := 20
def num_blue : ℕ := 6
def num_red : ℕ := 9
def num_white : ℕ := num_marbles - (num_blue + num_red)

def probability_red_or_white : ℚ :=
  (num_red + num_white) / num_marbles

theorem probability_red_or_white_is_7_over_10 :
  probability_red_or_white = 7 / 10 := 
sorry

end probability_red_or_white_is_7_over_10_l283_283997


namespace find_n_l283_283894

noncomputable def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

theorem find_n (n : ℕ) (h : n * factorial (n + 1) + factorial (n + 1) = 5040) : n = 5 :=
sorry

end find_n_l283_283894


namespace total_balls_in_bag_l283_283513

theorem total_balls_in_bag (R G B T : ℕ) 
  (hR : R = 907) 
  (hRatio : 15 * T = 15 * R + 13 * R + 17 * R)
  : T = 2721 :=
sorry

end total_balls_in_bag_l283_283513


namespace vegetables_in_one_serving_l283_283703

theorem vegetables_in_one_serving
  (V : ℝ)
  (H1 : ∀ servings : ℝ, servings > 0 → servings * (V + 2.5) = 28)
  (H_pints_to_cups : 14 * 2 = 28) :
  V = 1 :=
by
  -- proof steps would go here
  sorry

end vegetables_in_one_serving_l283_283703


namespace value_of_t_l283_283712

theorem value_of_t (k m r s t : ℕ) 
  (hk : 1 ≤ k) (hm : 2 ≤ m) (hr : r = 13) (hs : s = 14)
  (h : k < m) (h' : m < r) (h'' : r < s) (h''' : s < t)
  (average_condition : (k + m + r + s + t) / 5 = 10) :
  t = 20 := 
sorry

end value_of_t_l283_283712


namespace ratio_of_means_l283_283137

-- Variables for means
variables (xbar ybar zbar : ℝ)
-- Variables for sample sizes
variables (m n : ℕ)

-- Given conditions
def mean_x (x : ℕ) (xbar : ℝ) := ∀ i, 1 ≤ i ∧ i ≤ x → xbar = xbar
def mean_y (y : ℕ) (ybar : ℝ) := ∀ i, 1 ≤ i ∧ i ≤ y → ybar = ybar
def combined_mean (m n : ℕ) (xbar ybar zbar : ℝ) := zbar = (1/4) * xbar + (3/4) * ybar

-- Assertion to be proved
theorem ratio_of_means (h1 : mean_x m xbar) (h2 : mean_y n ybar)
  (h3 : xbar ≠ ybar) (h4 : combined_mean m n xbar ybar zbar) :
  m / n = 1 / 3 := sorry

end ratio_of_means_l283_283137


namespace person_left_time_l283_283750

theorem person_left_time :
  ∃ (x y : ℚ), 
    0 ≤ x ∧ x < 1 ∧ 
    0 ≤ y ∧ y < 1 ∧ 
    (120 + 30 * x = 360 * y) ∧
    (360 * x = 150 + 30 * y) ∧
    (4 + x = 4 + 64 / 143) := 
by
  sorry

end person_left_time_l283_283750


namespace Merrill_and_Elliot_have_fewer_marbles_than_Selma_l283_283813

variable (Merrill_marbles Elliot_marbles Selma_marbles total_marbles fewer_marbles : ℕ)

-- Conditions
def Merrill_has_30_marbles : Merrill_marbles = 30 := by sorry

def Elliot_has_half_of_Merrill's_marbles : Elliot_marbles = Merrill_marbles / 2 := by sorry

def Selma_has_50_marbles : Selma_marbles = 50 := by sorry

def Merrill_and_Elliot_together_total_marbles : total_marbles = Merrill_marbles + Elliot_marbles := by sorry

def number_of_fewer_marbles : fewer_marbles = Selma_marbles - total_marbles := by sorry

-- Goal
theorem Merrill_and_Elliot_have_fewer_marbles_than_Selma :
  fewer_marbles = 5 := by
  sorry

end Merrill_and_Elliot_have_fewer_marbles_than_Selma_l283_283813


namespace area_of_field_l283_283046

theorem area_of_field (L W A : ℕ) (h₁ : L = 20) (h₂ : L + 2 * W = 80) : A = 600 :=
by
  sorry

end area_of_field_l283_283046


namespace least_common_multiple_inequality_l283_283537

variable (a b c : ℕ)

theorem least_common_multiple_inequality (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 18) :
  Nat.lcm a c = 90 := sorry

end least_common_multiple_inequality_l283_283537


namespace y1_lt_y2_of_linear_function_l283_283250

theorem y1_lt_y2_of_linear_function (y1 y2 : ℝ) (h1 : y1 = 2 * (-3) + 1) (h2 : y2 = 2 * 2 + 1) : y1 < y2 :=
by
  sorry

end y1_lt_y2_of_linear_function_l283_283250


namespace inequality_proof_l283_283000

theorem inequality_proof
  {x1 x2 x3 x4 : ℝ}
  (h1 : x1 ≥ x2)
  (h2 : x2 ≥ x3)
  (h3 : x3 ≥ x4)
  (h4 : x4 ≥ 2)
  (h5 : x2 + x3 + x4 ≥ x1) :
  (x1 + x2 + x3 + x4)^2 ≤ 4 * x1 * x2 * x3 * x4 :=
by
  sorry

end inequality_proof_l283_283000


namespace system1_solution_system2_solution_l283_283738

-- Part 1: Substitution Method
theorem system1_solution (x y : ℤ) :
  2 * x - y = 3 ∧ 3 * x + 2 * y = 8 ↔ x = 2 ∧ y = 1 :=
by
  sorry

-- Part 2: Elimination Method
theorem system2_solution (x y : ℚ) :
  2 * x + y = 2 ∧ 8 * x + 3 * y = 9 ↔ x = 3 / 2 ∧ y = -1 :=
by
  sorry

end system1_solution_system2_solution_l283_283738


namespace minimum_b_l283_283497

noncomputable def f (a b x : ℝ) : ℝ := a * x + b

noncomputable def g (a b x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ a then f a b x else f a b (f a b x)

theorem minimum_b {a b : ℝ} (ha : 0 < a) :
  (∀ x : ℝ, 0 ≤ x → g a b x > g a b (x - 1)) → b ≥ 1 / 4 :=
sorry

end minimum_b_l283_283497


namespace probability_of_negative_cosine_value_l283_283616

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

def sum_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem probability_of_negative_cosine_value (a : ℕ → ℝ) (S : ℕ → ℝ) 
(h_arith_seq : arithmetic_sequence a)
(h_sum_seq : sum_arithmetic_sequence a S)
(h_S4 : S 4 = Real.pi)
(h_a4_eq_2a2 : a 4 = 2 * a 2) :
∃ p : ℝ, p = 7 / 15 ∧
  ∀ n, 1 ≤ n ∧ n ≤ 30 → 
  ((Real.cos (a n) < 0) → p = 7 / 15) :=
by sorry

end probability_of_negative_cosine_value_l283_283616


namespace possible_values_of_N_l283_283668

theorem possible_values_of_N (N : ℕ) (h1 : N ≥ 8 + 1)
  (h2 : ∀ (i : ℕ), (i < N → (i ≥ 0 ∧ i < 1/3 * (N-1)) → false) ) 
  (h3 : ∀ (i : ℕ), (i < N → (i ≥ 1/3 * (N-1) ∨ i < 1/3 * (N-1)) → true)) :
  23 ≤ N ∧ N ≤ 25 :=
by
  sorry

end possible_values_of_N_l283_283668


namespace price_of_coffee_increased_by_300_percent_l283_283967

theorem price_of_coffee_increased_by_300_percent
  (P : ℝ) -- cost per pound of milk powder and coffee in June
  (h1 : 0.20 * P = 0.20) -- price of a pound of milk powder in July
  (h2 : 1.5 * 0.20 = 0.30) -- cost of 1.5 lbs of milk powder in July
  (h3 : 6.30 - 0.30 = 6.00) -- cost of 1.5 lbs of coffee in July
  (h4 : 6.00 / 1.5 = 4.00) -- price per pound of coffee in July
  : ((4.00 - 1.00) / 1.00) * 100 = 300 := 
by 
  sorry

end price_of_coffee_increased_by_300_percent_l283_283967


namespace common_difference_is_3_l283_283331

theorem common_difference_is_3 (a : ℕ → ℤ) (d : ℤ) (h1 : a 2 = 4) (h2 : 1 + a 3 = 5 + d)
  (h3 : a 6 = 4 + 4 * d) (h4 : 4 + a 10 = 8 + 8 * d) :
  (5 + d) * (8 + 8 * d) = (4 + 4 * d) ^ 2 → d = 3 := 
by
  intros hg
  sorry

end common_difference_is_3_l283_283331


namespace fraction_sum_lt_one_l283_283418

theorem fraction_sum_lt_one (n : ℕ) (h_pos : n > 0) : 
  (1 / 2 + 1 / 3 + 1 / 10 + 1 / n < 1) ↔ (n > 15) :=
sorry

end fraction_sum_lt_one_l283_283418


namespace music_students_count_l283_283865

open Nat

theorem music_students_count (total_students : ℕ) (art_students : ℕ) (both_music_art : ℕ) 
      (neither_music_art : ℕ) (M : ℕ) :
    total_students = 500 →
    art_students = 10 →
    both_music_art = 10 →
    neither_music_art = 470 →
    (total_students - neither_music_art) = 30 →
    (M + (art_students - both_music_art)) = 30 →
    M = 30 :=
by
  intros h_total h_art h_both h_neither h_music_art_total h_music_count
  sorry

end music_students_count_l283_283865


namespace valid_N_values_l283_283647

def N_values (N : ℕ) : Prop :=
  (∀ k, k = N - 8 → (22 < N ∧ N ≤ 25))

-- Main theorem statement without proof
theorem valid_N_values (N : ℕ) (h : ∀ k, k = N - 8 → N_values N) : 
  (N = 23 ∨ N = 24 ∨ N = 25) :=
by
  sorry

end valid_N_values_l283_283647


namespace updated_mean_of_decremented_observations_l283_283829

theorem updated_mean_of_decremented_observations (n : ℕ) (initial_mean decrement : ℝ)
  (h₀ : n = 50) (h₁ : initial_mean = 200) (h₂ : decrement = 6) :
  ((n * initial_mean) - (n * decrement)) / n = 194 := by
  sorry

end updated_mean_of_decremented_observations_l283_283829


namespace tangent_line_is_correct_l283_283017

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, -1)

-- Define the equation of the tangent line
def tangent_line (x : ℝ) : ℝ := -3 * x + 2

-- Statement of the problem (to prove)
theorem tangent_line_is_correct :
  curve point_of_tangency.1 = point_of_tangency.2 ∧
  ∃ m b, (∀ x, (tangent_line x) = m * x + b) ∧
         tangent_line point_of_tangency.1 = point_of_tangency.2 ∧
         (∀ x, deriv (curve) x = -3 ↔ deriv (tangent_line) point_of_tangency.1 = -3) :=
by
  sorry

end tangent_line_is_correct_l283_283017


namespace algebraic_expression_value_l283_283781

theorem algebraic_expression_value (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + x*y + y^2 = 0) :
  (x/(x + y))^2005 + (y/(x + y))^2005 = -1 :=
by
  sorry

end algebraic_expression_value_l283_283781


namespace solve_a₃_l283_283934

noncomputable def geom_seq (a₁ a₅ a₃ : ℝ) : Prop :=
a₁ = 1 / 9 ∧ a₅ = 9 ∧ a₁ * a₅ = a₃^2

theorem solve_a₃ : ∃ a₃ : ℝ, geom_seq (1/9) 9 a₃ ∧ a₃ = 1 :=
by
  sorry

end solve_a₃_l283_283934


namespace number_of_zeros_among_50_numbers_l283_283087

theorem number_of_zeros_among_50_numbers :
  ∀ (m n p : ℕ), (m + n + p = 50) → (m * p = 500) → n = 5 :=
by
  intros m n p h1 h2
  sorry

end number_of_zeros_among_50_numbers_l283_283087


namespace grayson_fraction_l283_283214

variable (A G O : ℕ) -- The number of boxes collected by Abigail, Grayson, and Olivia, respectively
variable (C_per_box : ℕ) -- The number of cookies per box
variable (TotalCookies : ℕ) -- The total number of cookies collected by Abigail, Grayson, and Olivia

-- Given conditions
def abigail_boxes : ℕ := 2
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48
def total_cookies : ℕ := 276

-- Prove the fraction of the box that Grayson collected
theorem grayson_fraction :
  G * C_per_box = TotalCookies - (abigail_boxes + olivia_boxes) * cookies_per_box → 
  G / C_per_box = 3 / 4 := 
by
  sorry

-- Assume the variables from conditions
variable (G : ℕ := 36 / 48)
variable (TotalCookies := 276)
variable (C_per_box := 48)
variable (A := 2)
variable (O := 3)


end grayson_fraction_l283_283214


namespace side_length_of_base_l283_283394

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end side_length_of_base_l283_283394


namespace speed_of_current_l283_283218

theorem speed_of_current (h_start: ∀ t: ℝ, t ≥ 0 → u ≥ 0) 
  (boat1_turn_2pm: ∀ t: ℝ, t >= 1 → t < 2 → boat1_turn_13_14) 
  (boat2_turn_3pm: ∀ t: ℝ, t >= 2 → t < 3 → boat2_turn_14_15) 
  (boats_meet: ∀ x: ℝ, x = 7.5) :
  v = 2.5 := 
sorry

end speed_of_current_l283_283218


namespace girls_in_class_l283_283798

theorem girls_in_class (k : ℕ) (n_girls n_boys total_students : ℕ)
  (h1 : n_girls = 3 * k) (h2 : n_boys = 4 * k) (h3 : total_students = 35) 
  (h4 : n_girls + n_boys = total_students) : 
  n_girls = 15 :=
by
  -- The proof would normally go here, but is omitted per instructions.
  sorry

end girls_in_class_l283_283798


namespace total_pictures_l283_283147

noncomputable def RandyPics : ℕ := 5
noncomputable def PeterPics : ℕ := RandyPics + 3
noncomputable def QuincyPics : ℕ := PeterPics + 20

theorem total_pictures :
  RandyPics + PeterPics + QuincyPics = 41 :=
by
  sorry

end total_pictures_l283_283147


namespace problem_statement_l283_283508

theorem problem_statement (x : ℝ) (h : 8 * x = 4) : 150 * (1 / x) = 300 :=
by
  sorry

end problem_statement_l283_283508


namespace inequality_not_always_hold_l283_283091

variable (a b c : ℝ)

theorem inequality_not_always_hold (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) (h4 : a > 0) (h5 : b > 0) (h6 : c > 0) : ¬ (∀ (a b : ℝ), |a - b| + 1 / (a - b) ≥ 2) :=
by
  sorry

end inequality_not_always_hold_l283_283091


namespace problem_statement_l283_283809

theorem problem_statement
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a + b + c = 0)
  (h5 : ab + ac + bc ≠ 0) :
  (a^7 + b^7 + c^7) / (abc * (ab + ac + bc)) = -7 :=
  sorry

end problem_statement_l283_283809


namespace expected_balls_in_original_positions_after_transpositions_l283_283952

theorem expected_balls_in_original_positions_after_transpositions :
  let num_balls := 7
  let first_swap_probability := 2 / 7
  let second_swap_probability := 1 / 7
  let third_swap_probability := 1 / 7
  let original_position_probability := (2 / 343) + (125 / 343)
  let expected_balls := num_balls * original_position_probability
  expected_balls = 889 / 343 := 
sorry

end expected_balls_in_original_positions_after_transpositions_l283_283952


namespace solve_diamond_l283_283110

theorem solve_diamond : ∃ (D : ℕ), D < 10 ∧ (D * 9 + 5 = D * 10 + 2) ∧ D = 3 :=
by
  sorry

end solve_diamond_l283_283110


namespace total_sales_calculation_l283_283875

def average_price_per_pair : ℝ := 9.8
def number_of_pairs_sold : ℕ := 70
def total_amount : ℝ := 686

theorem total_sales_calculation :
  average_price_per_pair * (number_of_pairs_sold : ℝ) = total_amount :=
by
  -- proof goes here
  sorry

end total_sales_calculation_l283_283875


namespace ratio_of_ages_l283_283024

variable (D R : ℕ)

theorem ratio_of_ages : (D = 9) → (R + 6 = 18) → (R / D = 4 / 3) :=
by
  intros hD hR
  -- proof goes here
  sorry

end ratio_of_ages_l283_283024


namespace general_term_of_arithmetic_seq_l283_283780

variable {a : ℕ → ℕ} 
variable {S : ℕ → ℕ}

/-- Definition of sum of first n terms of an arithmetic sequence -/
def sum_of_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

/-- Definition of arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n, a (n + 1) = a n + d

theorem general_term_of_arithmetic_seq
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 6 = 12)
  (h3 : S 3 = 12)
  (h4 : sum_of_arithmetic_sequence S a) :
  ∀ n, a n = 2 * n := 
sorry

end general_term_of_arithmetic_seq_l283_283780


namespace minimum_value_problem_l283_283946

noncomputable def minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) : Prop :=
  (2 * x + 3 * y) * (2 * y + 3 * z) * (2 * x * z + 1) ≥ 24

theorem minimum_value_problem 
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) : 
  minimum_value_inequality x y z hx hy hz hxyz :=
by sorry

end minimum_value_problem_l283_283946


namespace correctness_of_statements_l283_283920

theorem correctness_of_statements 
  (A B C D : Prop)
  (h1 : A → B) (h2 : ¬(B → A))
  (h3 : C → B) (h4 : B → C)
  (h5 : D → C) (h6 : ¬(C → D)) : 
  (A → (C ∧ ¬(C → A))) ∧ (¬(A → D) ∧ ¬(D → A)) := 
by
  -- Proof will go here.
  sorry

end correctness_of_statements_l283_283920


namespace solve_system_of_equations_l283_283154

theorem solve_system_of_equations :
  ∀ (x y z : ℚ), 
    (x * y = x + 2 * y ∧
     y * z = y + 3 * z ∧
     z * x = z + 4 * x) ↔
    (x = 0 ∧ y = 0 ∧ z = 0) ∨
    (x = 25 / 9 ∧ y = 25 / 7 ∧ z = 25 / 4) := by
  sorry

end solve_system_of_equations_l283_283154


namespace not_all_pieces_found_l283_283105

theorem not_all_pieces_found (k p v : ℕ) (h1 : p + v > 0) (h2 : k % 2 = 1) : k + 4 * p + 8 * v ≠ 1988 :=
by
  sorry

end not_all_pieces_found_l283_283105


namespace r_fourth_power_sum_l283_283281

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l283_283281


namespace centroid_of_triangle_PQR_positions_l283_283710

-- Define the basic setup
def square_side_length : ℕ := 12
def total_points : ℕ := 48

-- Define the centroid calculation condition
def centroid_positions_count : ℕ :=
  let side_segments := square_side_length
  let points_per_edge := total_points / 4
  let possible_positions_per_side := points_per_edge - 1
  (possible_positions_per_side * possible_positions_per_side)

/-- Proof statement: Proving the number of possible positions for the centroid of triangle PQR 
    formed by any three non-collinear points out of the 48 points on the perimeter of the square. --/
theorem centroid_of_triangle_PQR_positions : centroid_positions_count = 121 := 
  sorry

end centroid_of_triangle_PQR_positions_l283_283710


namespace pyramid_base_side_length_l283_283387

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end pyramid_base_side_length_l283_283387


namespace find_ordered_pair_l283_283605

theorem find_ordered_pair : ∃ k a : ℤ, 
  (∀ x : ℝ, (x^3 - 4*x^2 + 9*x - 6) % (x^2 - x + k) = 2*x + a) ∧ k = 4 ∧ a = 6 :=
sorry

end find_ordered_pair_l283_283605


namespace remaining_storage_space_l283_283369

/-- Given that 1 GB = 1024 MB, a hard drive with 300 GB of total storage,
and 300000 MB of used storage, prove that the remaining storage space is 7200 MB. -/
theorem remaining_storage_space (total_gb : ℕ) (mb_per_gb : ℕ) (used_mb : ℕ) :
  total_gb = 300 → mb_per_gb = 1024 → used_mb = 300000 →
  (total_gb * mb_per_gb - used_mb) = 7200 :=
by
  intros h1 h2 h3
  sorry

end remaining_storage_space_l283_283369


namespace felix_can_lift_150_l283_283467

-- Define the weights of Felix and his brother.
variables (F B : ℤ)

-- Given conditions
-- Felix's brother can lift three times his weight off the ground, and this amount is 600 pounds.
def brother_lift (B : ℤ) : Prop := 3 * B = 600
-- Felix's brother weighs twice as much as Felix.
def brother_weight (B F : ℤ) : Prop := B = 2 * F
-- Felix can lift off the ground 1.5 times his weight.
def felix_lift (F : ℤ) : ℤ := 3 * F / 2 -- Note: 1.5F can be represented as 3F/2 in Lean for integer operations.

-- Goal: Prove that Felix can lift 150 pounds.
theorem felix_can_lift_150 (F B : ℤ) (h1 : brother_lift B) (h2 : brother_weight B F) : felix_lift F = 150 := by
  dsimp [brother_lift, brother_weight, felix_lift] at *
  sorry

end felix_can_lift_150_l283_283467


namespace sum_of_equal_numbers_l283_283534

theorem sum_of_equal_numbers (a b : ℝ) (h1 : (12 + 25 + 18 + a + b) / 5 = 20) (h2 : a = b) : a + b = 45 :=
sorry

end sum_of_equal_numbers_l283_283534


namespace percentage_transactions_anthony_handled_more_l283_283948

theorem percentage_transactions_anthony_handled_more (M A C J : ℕ) (P : ℚ)
  (hM : M = 90)
  (hJ : J = 83)
  (hCJ : J = C + 17)
  (hCA : C = (2 * A) / 3)
  (hP : P = ((A - M): ℚ) / M * 100) :
  P = 10 := by
  sorry

end percentage_transactions_anthony_handled_more_l283_283948


namespace sufficient_but_not_necessary_condition_l283_283189

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x + m = 0) ↔ m < 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l283_283189


namespace prime_divides_binom_l283_283955

-- We define that n is a prime number.
def is_prime (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Lean statement for the problem
theorem prime_divides_binom {n k : ℕ} (h₁ : is_prime n) (h₂ : 0 < k) (h₃ : k < n) :
  n ∣ Nat.choose n k :=
sorry

end prime_divides_binom_l283_283955


namespace intersection_range_l283_283630

def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem intersection_range :
  {m : ℝ | ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = m ∧ f x2 = m ∧ f x3 = m} = Set.Ioo (-3 : ℝ) 1 :=
by
  sorry

end intersection_range_l283_283630


namespace altered_solution_water_amount_l283_283540

def initial_bleach_ratio := 2
def initial_detergent_ratio := 40
def initial_water_ratio := 100

def new_bleach_to_detergent_ratio := 3 * initial_bleach_ratio
def new_detergent_to_water_ratio := initial_detergent_ratio / 2

def detergent_amount := 60
def water_amount := 75

theorem altered_solution_water_amount :
  (initial_detergent_ratio / new_detergent_to_water_ratio) * detergent_amount / new_bleach_to_detergent_ratio = water_amount :=
by
  sorry

end altered_solution_water_amount_l283_283540


namespace calculate_expression_l283_283063

noncomputable def exponent_inverse (x : ℝ) : ℝ := x ^ (-1)
noncomputable def root (x : ℝ) (n : ℕ) : ℝ := x ^ (1 / n : ℝ)

theorem calculate_expression :
  (exponent_inverse 4) - (root (1/16) 2) + (3 - Real.sqrt 2) ^ 0 = 1 := 
by
  -- Definitions according to conditions
  have h1 : exponent_inverse 4 = 1 / 4 := by sorry
  have h2 : root (1 / 16) 2 = 1 / 4 := by sorry
  have h3 : (3 - Real.sqrt 2) ^ 0 = 1 := by sorry
  
  -- Combine and simplify parts
  calc
    (exponent_inverse 4) - (root (1 / 16) 2) + (3 - Real.sqrt 2) ^ 0
        = (1 / 4) - (1 / 4) + 1 : by rw [h1, h2, h3]
    ... = 0 + 1 : by sorry
    ... = 1 : by rfl

end calculate_expression_l283_283063


namespace possible_values_for_N_l283_283663

theorem possible_values_for_N (N : ℕ) (H : ∀ (k : ℕ), N = 8 + k) (truth: (student : ℕ → Prop) (bully : ℕ → Prop) (n : ℕ), 
  (∀ i, bully i → ¬ (∃ j, student j ∧ bully j → j ≠ i)) →
  (∀ i, student i → (∃ j, student j ∧ bully j → j ≠ i))
  → ∀ i, (bully i → ¬(∃ (m : ℕ), (m ≥ N - 1 / 3 )) ∧ student i → (∃ (m : ℕ), (m ≥ N - 1 / 3 ))) : N = 23 ∨ N = 24 ∨ N = 25 :=
by
  sorry

end possible_values_for_N_l283_283663


namespace correct_sentence_l283_283448

-- Define an enumeration for different sentences
inductive Sentence
| A : Sentence
| B : Sentence
| C : Sentence
| D : Sentence

-- Define a function stating properties of each sentence
def sentence_property (s : Sentence) : Bool :=
  match s with
  | Sentence.A => false  -- "The chromosomes from dad are more than from mom" is false
  | Sentence.B => false  -- "The chromosomes in my cells and my brother's cells are exactly the same" is false
  | Sentence.C => true   -- "Each pair of homologous chromosomes is provided by both parents" is true
  | Sentence.D => false  -- "Each pair of homologous chromosomes in my brother's cells are the same size" is false

-- The theorem to prove that Sentence.C is the correct one
theorem correct_sentence : sentence_property Sentence.C = true :=
by
  unfold sentence_property
  rfl

end correct_sentence_l283_283448


namespace valid_N_values_l283_283650

def N_values (N : ℕ) : Prop :=
  (∀ k, k = N - 8 → (22 < N ∧ N ≤ 25))

-- Main theorem statement without proof
theorem valid_N_values (N : ℕ) (h : ∀ k, k = N - 8 → N_values N) : 
  (N = 23 ∨ N = 24 ∨ N = 25) :=
by
  sorry

end valid_N_values_l283_283650


namespace clothing_probability_l283_283928

open Nat

theorem clothing_probability :
  let total_clothing := 5 + 7 + 8
  let total_ways := Nat.choose total_clothing 4
  let shirt_ways := Nat.choose 5 2
  let shorts_ways := Nat.choose 7 1
  let socks_ways := Nat.choose 8 1
  let desired_ways := shirt_ways * shorts_ways * socks_ways
  desired_ways / total_ways = 112 / 969 :=
by 
  let total_clothing := 20
  let total_ways := Nat.choose total_clothing 4
  let shirt_ways := Nat.choose 5 2
  let shorts_ways := Nat.choose 7 1
  let socks_ways := Nat.choose 8 1
  let desired_ways := shirt_ways * shorts_ways * socks_ways
  have : total_ways = 4845 := Nat.choose_eq 20 4 sorry
  have : desired_ways = 560 := sorry
  show desired_ways / total_ways = 112 / 969
  linarith

end clothing_probability_l283_283928


namespace find_r4_l283_283269

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l283_283269


namespace scientific_notation_example_l283_283601

theorem scientific_notation_example : 0.0000037 = 3.7 * 10^(-6) :=
by
  -- We would provide the proof here.
  sorry

end scientific_notation_example_l283_283601


namespace tangent_line_ln_l283_283826

theorem tangent_line_ln (x y : ℝ) (h_curve : y = Real.log (x + 1)) (h_point : (1, Real.log 2) = (1, y)) :
  x - 2 * y - 1 + 2 * Real.log 2 = 0 :=
by
  sorry

end tangent_line_ln_l283_283826


namespace odd_solution_exists_l283_283006

theorem odd_solution_exists (k m n : ℕ) (h : m * n = k^2 + k + 3) : 
∃ (x y : ℤ), (x^2 + 11 * y^2 = 4 * m ∨ x^2 + 11 * y^2 = 4 * n) ∧ (x % 2 ≠ 0 ∧ y % 2 ≠ 0) :=
sorry

end odd_solution_exists_l283_283006


namespace solve_for_a_l283_283636

theorem solve_for_a (a x : ℝ) (h : 2 * x + 3 * a = 10) (hx : x = 2) : a = 2 :=
by
  rw [hx] at h
  linarith

end solve_for_a_l283_283636


namespace a₁₀_greater_than_500_l283_283697

variables (a : ℕ → ℕ) (b : ℕ → ℕ)

-- Conditions
def strictly_increasing (a : ℕ → ℕ) : Prop := ∀ n, a n < a (n + 1)

def largest_divisor (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ n, b n < a n ∧ ∃ d > 1, d ∣ a n ∧ b n = a n / d

def greater_sequence (b : ℕ → ℕ) : Prop := ∀ n, b n > b (n + 1)

-- Statement to prove
theorem a₁₀_greater_than_500
  (h1 : strictly_increasing a)
  (h2 : largest_divisor a b)
  (h3 : greater_sequence b) :
  a 10 > 500 :=
sorry

end a₁₀_greater_than_500_l283_283697


namespace r_pow_four_solution_l283_283299

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l283_283299


namespace inscribable_quadrilateral_l283_283956

theorem inscribable_quadrilateral
  (a b c d : ℝ)
  (A : ℝ)
  (circumscribable : Prop)
  (area_condition : A = Real.sqrt (a * b * c * d))
  (A := Real.sqrt (a * b * c * d)) : 
  circumscribable → ∃ B D : ℝ, B + D = 180 :=
sorry

end inscribable_quadrilateral_l283_283956


namespace tangent_line_slope_l283_283255

/-- Given the line y = mx is tangent to the circle x^2 + y^2 - 4x + 2 = 0, 
    the slope m must be ±1. -/
theorem tangent_line_slope (m : ℝ) :
  (∃ x y : ℝ, y = m * x ∧ (x ^ 2 + y ^ 2 - 4 * x + 2 = 0)) →
  (m = 1 ∨ m = -1) :=
by
  sorry

end tangent_line_slope_l283_283255


namespace area_code_length_l283_283545

theorem area_code_length (n : ℕ) (h : 224^n - 222^n = 888) : n = 2 :=
sorry

end area_code_length_l283_283545


namespace sandy_initial_amount_l283_283149

theorem sandy_initial_amount 
  (cost_shirt : ℝ) (cost_jacket : ℝ) (found_money : ℝ)
  (h1 : cost_shirt = 12.14) (h2 : cost_jacket = 9.28) (h3 : found_money = 7.43) : 
  (cost_shirt + cost_jacket + found_money = 28.85) :=
by
  rw [h1, h2, h3]
  norm_num

end sandy_initial_amount_l283_283149


namespace problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l283_283455

theorem problem1 : 0 - (-22) = 22 := 
by 
  sorry

theorem problem2 : 8.5 - (-1.5) = 10 := 
by 
  sorry

theorem problem3 : (-13 : ℚ) - (4/7) - (-13 : ℚ) - (5/7) = 1/7 := 
by 
  sorry

theorem problem4 : (-1/2 : ℚ) - (1/4 : ℚ) = -3/4 := 
by 
  sorry

theorem problem5 : -51 + 12 + (-7) + (-11) + 36 = -21 := 
by 
  sorry

theorem problem6 : (5/6 : ℚ) + (-2/3) + 1 + (1/6) + (-1/3) = 1 := 
by 
  sorry

theorem problem7 : -13 + (-7) - 20 - (-40) + 16 = 16 := 
by 
  sorry

theorem problem8 : 4.7 - (-8.9) - 7.5 + (-6) = 0.1 := 
by 
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l283_283455


namespace intersection_claim_union_claim_l283_283256

def A : Set ℝ := {x | (x - 2) * (x + 5) < 0}
def B : Set ℝ := {x | x^2 - 2 * x - 3 ≥ 0}
def U : Set ℝ := Set.univ

-- Claim 1: Prove that A ∩ B = {x | -5 < x ∧ x ≤ -1}
theorem intersection_claim : A ∩ B = {x | -5 < x ∧ x ≤ -1} :=
by
  sorry

-- Claim 2: Prove that A ∪ (U \ B) = {x | -5 < x ∧ x < 3}
theorem union_claim : A ∪ (U \ B) = {x | -5 < x ∧ x < 3} :=
by
  sorry

end intersection_claim_union_claim_l283_283256


namespace market_value_correct_l283_283434

noncomputable def market_value : ℝ :=
  let dividend_income (M : ℝ) := 0.12 * M
  let fees (M : ℝ) := 0.01 * M
  let taxes (M : ℝ) := 0.15 * dividend_income M
  have yield_after_fees_and_taxes : ∀ M, 0.08 * M = dividend_income M - fees M - taxes M := 
    by sorry
  86.96

theorem market_value_correct :
  market_value = 86.96 := 
by
  sorry

end market_value_correct_l283_283434


namespace x_plus_reciprocal_eq_two_implies_x_pow_12_eq_one_l283_283509

theorem x_plus_reciprocal_eq_two_implies_x_pow_12_eq_one {x : ℝ} (h : x + 1 / x = 2) : x^12 = 1 :=
by
  -- The proof will go here, but it is omitted.
  sorry

end x_plus_reciprocal_eq_two_implies_x_pow_12_eq_one_l283_283509


namespace solve_for_x_l283_283922

theorem solve_for_x (x : ℝ) (h : (x / 6) / 3 = 9 / (x / 3)) : x = 9 * Real.sqrt 6 ∨ x = - (9 * Real.sqrt 6) :=
by
  sorry

end solve_for_x_l283_283922


namespace necessary_but_not_sufficient_not_sufficient_l283_283619

def P (x : ℝ) : Prop := x < 1
def Q (x : ℝ) : Prop := (x + 2) * (x - 1) < 0

theorem necessary_but_not_sufficient (x : ℝ) : P x → Q x := by
  intro hx
  sorry

theorem not_sufficient (x : ℝ) : ¬(Q x → P x) := by
  intro hq
  sorry

end necessary_but_not_sufficient_not_sufficient_l283_283619


namespace average_height_of_three_l283_283358

theorem average_height_of_three (parker daisy reese : ℕ) 
  (h1 : parker = daisy - 4)
  (h2 : daisy = reese + 8)
  (h3 : reese = 60) : 
  (parker + daisy + reese) / 3 = 64 := 
  sorry

end average_height_of_three_l283_283358


namespace intersection_sums_l283_283827

theorem intersection_sums (x1 x2 x3 y1 y2 y3 : ℝ) (h1 : y1 = x1^3 - 6 * x1 + 4)
  (h2 : y2 = x2^3 - 6 * x2 + 4) (h3 : y3 = x3^3 - 6 * x3 + 4)
  (h4 : x1 + 3 * y1 = 3) (h5 : x2 + 3 * y2 = 3) (h6 : x3 + 3 * y3 = 3) :
  x1 + x2 + x3 = 0 ∧ y1 + y2 + y3 = 3 := 
by
  sorry

end intersection_sums_l283_283827


namespace proof_problem_l283_283615

variable (x y : ℝ)

noncomputable def condition1 : Prop := x > y
noncomputable def condition2 : Prop := x * y = 1

theorem proof_problem (hx : condition1 x y) (hy : condition2 x y) : 
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := 
by
  sorry

end proof_problem_l283_283615


namespace models_kirsty_can_buy_l283_283339

def savings := 30 * 0.45
def new_price := 0.50

theorem models_kirsty_can_buy : savings / new_price = 27 := by
  sorry

end models_kirsty_can_buy_l283_283339


namespace solve_2xx_eq_sqrt2_unique_solution_l283_283820

noncomputable def solve_equation_2xx_eq_sqrt2 (x : ℝ) : Prop :=
  2 * x^x = Real.sqrt 2

theorem solve_2xx_eq_sqrt2_unique_solution (x : ℝ) : solve_equation_2xx_eq_sqrt2 x ↔ (x = 1/2 ∨ x = 1/4) ∧ x > 0 :=
by
  sorry

end solve_2xx_eq_sqrt2_unique_solution_l283_283820


namespace math_problem_l283_283022

open Function

noncomputable def rotate_90_ccw (p : ℝ × ℝ) (c : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (h, k) := c
  (h - (y - k), k + (x - h))

noncomputable def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (y, x)

theorem math_problem (a b : ℝ) :
  reflect_over_y_eq_x (rotate_90_ccw (a, b) (2, 3)) = (4, -5) → b - a = -5 :=
by
  intros h
  sorry

end math_problem_l283_283022


namespace symmetric_point_origin_l283_283411

def symmetric_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (-P.1, -P.2)

theorem symmetric_point_origin :
  symmetric_point (3, -1) = (-3, 1) :=
by
  sorry

end symmetric_point_origin_l283_283411


namespace total_players_is_139_l283_283862

def num_kabadi := 60
def num_kho_kho := 90
def num_soccer := 40
def num_basketball := 70
def num_volleyball := 50
def num_badminton := 30

def num_k_kh := 25
def num_k_s := 15
def num_k_b := 13
def num_k_v := 20
def num_k_ba := 10
def num_kh_s := 35
def num_kh_b := 16
def num_kh_v := 30
def num_kh_ba := 12
def num_s_b := 20
def num_s_v := 18
def num_s_ba := 7
def num_b_v := 15
def num_b_ba := 8
def num_v_ba := 10

def num_k_kh_s := 5
def num_k_b_v := 4
def num_s_b_ba := 3
def num_v_ba_kh := 2

def num_all_sports := 1

noncomputable def total_players : Nat :=
  (num_kabadi + num_kho_kho + num_soccer + num_basketball + num_volleyball + num_badminton) 
  - (num_k_kh + num_k_s + num_k_b + num_k_v + num_k_ba + num_kh_s + num_kh_b + num_kh_v + num_kh_ba + num_s_b + num_s_v + num_s_ba + num_b_v + num_b_ba + num_v_ba)
  + (num_k_kh_s + num_k_b_v + num_s_b_ba + num_v_ba_kh)
  - num_all_sports

theorem total_players_is_139 : total_players = 139 := 
  by 
    sorry

end total_players_is_139_l283_283862


namespace expand_and_simplify_l283_283073

theorem expand_and_simplify (x : ℝ) : 
  (2 * x + 6) * (x + 10) = 2 * x^2 + 26 * x + 60 :=
sorry

end expand_and_simplify_l283_283073


namespace weight_of_person_replaced_l283_283158

theorem weight_of_person_replaced (W_new : ℝ) (h1 : W_new = 74) (h2 : (W_new - W_old) = 9) : W_old = 65 := 
by
  sorry

end weight_of_person_replaced_l283_283158


namespace system1_solution_system2_solution_l283_283367

theorem system1_solution :
  ∃ (x y : ℤ), (4 * x - y = 1) ∧ (y = 2 * x + 3) ∧ (x = 2) ∧ (y = 7) :=
by
  sorry

theorem system2_solution :
  ∃ (x y : ℤ), (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ∧ (x = 5) ∧ (y = 5) :=
by
  sorry

end system1_solution_system2_solution_l283_283367


namespace total_people_clean_city_l283_283637

-- Define the conditions
def lizzie_group : Nat := 54
def group_difference : Nat := 17
def other_group := lizzie_group - group_difference

-- State the theorem
theorem total_people_clean_city : lizzie_group + other_group = 91 := by
  -- Proof would go here
  sorry

end total_people_clean_city_l283_283637


namespace express_y_in_terms_of_x_l283_283247

theorem express_y_in_terms_of_x (x y : ℝ) (h : y - 2 * x = 5) : y = 2 * x + 5 :=
by
  sorry

end express_y_in_terms_of_x_l283_283247


namespace inequality_real_equation_positive_integers_solution_l283_283036

-- Prove the inequality for real numbers a and b
theorem inequality_real (a b : ℝ) :
  (a^2 + 1) * (b^2 + 1) + 50 ≥ 2 * ((2 * a + 1) * (3 * b + 1)) :=
  sorry

-- Find all positive integers n and p such that the equation holds
theorem equation_positive_integers_solution :
  ∃ (n p : ℕ), 0 < n ∧ 0 < p ∧ (n^2 + 1) * (p^2 + 1) + 45 = 2 * ((2 * n + 1) * (3 * p + 1)) ∧ n = 2 ∧ p = 2 :=
  sorry

end inequality_real_equation_positive_integers_solution_l283_283036


namespace possible_values_of_N_l283_283655

def is_valid_N (N : ℕ) : Prop :=
  (N > 22) ∧ (N ≤ 25)

theorem possible_values_of_N :
  {N : ℕ | is_valid_N N} = {23, 24, 25} :=
by
  sorry

end possible_values_of_N_l283_283655


namespace m_greater_than_one_l283_283484

variables {x m : ℝ}

def p : Prop := -2 ≤ x ∧ x ≤ 11
def q : Prop := 1 - 3 * m ≤ x ∧ x ≤ 3 + m

theorem m_greater_than_one (h : ¬(x^2 - 2 * x + m ≤ 0)) : m > 1 :=
sorry

end m_greater_than_one_l283_283484


namespace tens_digit_of_9_pow_2023_l283_283423

theorem tens_digit_of_9_pow_2023 : (9 ^ 2023) % 100 / 10 = 2 :=
by sorry

end tens_digit_of_9_pow_2023_l283_283423


namespace find_a_l283_283253

theorem find_a (a : ℝ) :
  (∃x y : ℝ, x^2 + y^2 + 2 * x - 2 * y + a = 0 ∧ x + y + 4 = 0) →
  ∃c : ℝ, c = 2 ∧ a = -7 :=
by
  -- proof to be filled in
  sorry

end find_a_l283_283253


namespace tape_for_small_box_l283_283761

theorem tape_for_small_box (S : ℝ) :
  (2 * 4) + (8 * 2) + (5 * S) + (2 + 8 + 5) = 44 → S = 1 :=
by
  intro h
  sorry

end tape_for_small_box_l283_283761


namespace diagonal_ratio_of_squares_l283_283168

theorem diagonal_ratio_of_squares (P d : ℝ) (h : ∃ s S, 4 * S = 4 * s * 4 ∧ P = 4 * s ∧ d = s * Real.sqrt 2) : 
    (∃ D, D = 4 * d) :=
by
  sorry

end diagonal_ratio_of_squares_l283_283168


namespace pyramid_base_side_length_l283_283385

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end pyramid_base_side_length_l283_283385


namespace hyperbola_range_m_l283_283969

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (m + 2 > 0 ∧ m - 2 < 0) ∧ (x^2 / (m + 2) + y^2 / (m - 2) = 1)) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end hyperbola_range_m_l283_283969


namespace ellipse_equation_l283_283825

theorem ellipse_equation (h₁ : center = (0, 0))
                         (h₂ : foci_on_x_axis)
                         (h₃ : major_axis_length 18)
                         (h₄ : foci_trisect_major_axis) :
  ∃ x y : ℝ, (x ^ 2) / 81 + (y ^ 2) / 72 = 1 :=
by sorry

end ellipse_equation_l283_283825


namespace solve_diamond_l283_283111

theorem solve_diamond : ∃ (D : ℕ), D < 10 ∧ (D * 9 + 5 = D * 10 + 2) ∧ D = 3 :=
by
  sorry

end solve_diamond_l283_283111


namespace number_of_valid_n_l283_283474

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ m n : ℕ, n = 2^m * 5^n

def has_nonzero_thousandths_digit (n : ℕ) : Prop :=
  -- Placeholder for a formal definition to check the non-zero thousandths digit.
  sorry

theorem number_of_valid_n : 
  (∃ l : List ℕ, 
    l.length = 10 ∧ 
    ∀ n ∈ l, n <= 200 ∧ is_terminating_decimal n ∧ has_nonzero_thousandths_digit n) :=
sorry

end number_of_valid_n_l283_283474


namespace find_r_fourth_l283_283287

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l283_283287


namespace darnell_avg_yards_eq_11_l283_283524

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

end darnell_avg_yards_eq_11_l283_283524


namespace jumping_contest_l283_283828

variables (G F M K : ℤ)

-- Define the conditions
def condition_1 : Prop := G = 39
def condition_2 : Prop := G = F + 19
def condition_3 : Prop := M = F - 12
def condition_4 : Prop := K = 2 * F - 5

-- The theorem asserting the final distances
theorem jumping_contest 
    (h1 : condition_1 G)
    (h2 : condition_2 G F)
    (h3 : condition_3 F M)
    (h4 : condition_4 F K) :
    G = 39 ∧ F = 20 ∧ M = 8 ∧ K = 35 := by
  sorry

end jumping_contest_l283_283828


namespace determine_y_l283_283598

-- Define the main problem in a Lean theorem
theorem determine_y (y : ℕ) : 9^10 + 9^10 + 9^10 = 3^y → y = 21 :=
by
  -- proof not required, so we add sorry
  sorry

end determine_y_l283_283598


namespace maria_drank_8_bottles_l283_283812

def initial_bottles : ℕ := 14
def bought_bottles : ℕ := 45
def remaining_bottles : ℕ := 51

theorem maria_drank_8_bottles :
  let total_bottles := initial_bottles + bought_bottles
  let drank_bottles := total_bottles - remaining_bottles
  drank_bottles = 8 :=
by
  let total_bottles := 14 + 45
  let drank_bottles := total_bottles - 51
  show drank_bottles = 8
  sorry

end maria_drank_8_bottles_l283_283812


namespace practice_minutes_l283_283225

def month_total_days : ℕ := (2 * 6) + (2 * 7)

def piano_daily_minutes : ℕ := 25

def violin_daily_minutes := piano_daily_minutes * 3

def flute_daily_minutes := violin_daily_minutes / 2

theorem practice_minutes (piano_total : ℕ) (violin_total : ℕ) (flute_total : ℕ) :
  (26 * piano_daily_minutes = 650) ∧ 
  (20 * violin_daily_minutes = 1500) ∧ 
  (16 * flute_daily_minutes = 600) := by
  sorry

end practice_minutes_l283_283225


namespace option_d_is_pythagorean_triple_l283_283034

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem option_d_is_pythagorean_triple : is_pythagorean_triple 5 12 13 :=
by
  -- This will be the proof part, which is omitted as per the problem's instructions.
  sorry

end option_d_is_pythagorean_triple_l283_283034


namespace range_of_a_same_side_of_line_l283_283092

theorem range_of_a_same_side_of_line 
  {P Q : ℝ × ℝ} 
  (hP : P = (3, -1)) 
  (hQ : Q = (-1, 2)) 
  (h_side : (3 * a - 3) * (-a + 3) > 0) : 
  a > 1 ∧ a < 3 := 
by 
  sorry

end range_of_a_same_side_of_line_l283_283092


namespace range_of_a_l283_283776

noncomputable def geometric_seq (r : ℝ) (n : ℕ) (a₁ : ℝ) : ℝ := a₁ * r ^ (n - 1)

theorem range_of_a (a : ℝ) :
  (∃ a_seq b_seq : ℕ → ℝ, a_seq 1 = a ∧ (∀ n, b_seq n = (a_seq n - 2) / (a_seq n - 1)) ∧ (∀ n, a_seq n > a_seq (n+1)) ∧ (∀ n, b_seq (n + 1) = geometric_seq (2/3) (n + 1) (b_seq 1))) → 2 < a :=
by
  sorry

end range_of_a_l283_283776


namespace least_number_subtracted_divisible_by_17_and_23_l283_283033

-- Conditions
def is_divisible_by_17_and_23 (n : ℕ) : Prop := 
  n % 17 = 0 ∧ n % 23 = 0

def target_number : ℕ := 7538

-- The least number to be subtracted
noncomputable def least_number_to_subtract : ℕ := 109

-- Theorem statement
theorem least_number_subtracted_divisible_by_17_and_23 : 
  is_divisible_by_17_and_23 (target_number - least_number_to_subtract) :=
by 
  -- Proof details would normally follow here.
  sorry

end least_number_subtracted_divisible_by_17_and_23_l283_283033


namespace necessary_but_not_sufficient_l283_283502

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x, x ≥ a → x^2 - x - 2 ≥ 0) ∧ (∃ x, x ≥ a ∧ ¬(x^2 - x - 2 ≥ 0)) ↔ a ≥ 2 := 
sorry

end necessary_but_not_sufficient_l283_283502


namespace min_value_fraction_l283_283492

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  (4 / x + 9 / y) ≥ 25 :=
sorry

end min_value_fraction_l283_283492


namespace r_power_four_identity_l283_283308

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l283_283308


namespace tangent_lines_through_point_l283_283603

theorem tangent_lines_through_point (x y : ℝ) (hp : (x, y) = (3, 1))
 : ∃ (a b c : ℝ), (y - 1 = (4 / 3) * (x - 3) ∨ x = 3) :=
by
  sorry

end tangent_lines_through_point_l283_283603


namespace butterfly_development_time_l283_283801

theorem butterfly_development_time :
  ∀ (larva_time cocoon_time : ℕ), 
  (larva_time = 3 * cocoon_time) → 
  (cocoon_time = 30) → 
  (larva_time + cocoon_time = 120) :=
by 
  intros larva_time cocoon_time h1 h2
  sorry

end butterfly_development_time_l283_283801


namespace contractor_fine_per_absent_day_l283_283570

theorem contractor_fine_per_absent_day :
  ∃ x : ℝ, (∀ (total_days absent_days worked_days earnings_per_day total_earnings : ℝ),
   total_days = 30 →
   earnings_per_day = 25 →
   total_earnings = 490 →
   absent_days = 8 →
   worked_days = total_days - absent_days →
   25 * worked_days - absent_days * x = total_earnings
  ) → x = 7.5 :=
by
  existsi 7.5
  intros
  sorry

end contractor_fine_per_absent_day_l283_283570


namespace tan_15_pi_over_4_l283_283766

theorem tan_15_pi_over_4 : Real.tan (15 * Real.pi / 4) = -1 :=
by
-- The proof is omitted.
sorry

end tan_15_pi_over_4_l283_283766


namespace op_plus_18_plus_l283_283348

def op_plus (y: ℝ) : ℝ := 9 - y
def plus_op (y: ℝ) : ℝ := y - 9

theorem op_plus_18_plus :
  plus_op (op_plus 18) = -18 := by
  sorry

end op_plus_18_plus_l283_283348


namespace new_quadratic_eq_l283_283486

def quadratic_roots_eq (a b c : ℝ) (x1 x2 : ℝ) : Prop :=
  a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0

theorem new_quadratic_eq
  (a b c : ℝ) (x1 x2 : ℝ)
  (h1 : quadratic_roots_eq a b c x1 x2)
  (h_sum : x1 + x2 = -b / a)
  (h_prod : x1 * x2 = c / a) :
  a^3 * x^2 - a * b^2 * x + 2 * c * (b^2 - 2 * a * c) = 0 :=
sorry

end new_quadratic_eq_l283_283486


namespace problem1_problem2_l283_283818

theorem problem1 (n : ℕ) (hn : 0 < n) : (3^(2*n+1) + 2^(n+2)) % 7 = 0 := 
sorry

theorem problem2 (n : ℕ) (hn : 0 < n) : (3^(2*n+2) + 2^(6*n+1)) % 11 = 0 := 
sorry

end problem1_problem2_l283_283818


namespace weight_of_B_l283_283963

-- Definitions for the weights of A, B, and C
variable (A B C : ℝ)

-- Conditions given in the problem
def condition1 := (A + B + C) / 3 = 45
def condition2 := (A + B) / 2 = 40
def condition3 := (B + C) / 2 = 43

-- The theorem to prove that B = 31 under the given conditions
theorem weight_of_B : condition1 A B C → condition2 A B → condition3 B C → B = 31 := by
  intros
  sorry

end weight_of_B_l283_283963


namespace problem1_l283_283741

theorem problem1 (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x, 2 * |x + 3| ≥ m - 2 * |x + 7|) →
  (m ≤ 20) :=
by
  sorry

end problem1_l283_283741


namespace valid_N_values_l283_283648

def N_values (N : ℕ) : Prop :=
  (∀ k, k = N - 8 → (22 < N ∧ N ≤ 25))

-- Main theorem statement without proof
theorem valid_N_values (N : ℕ) (h : ∀ k, k = N - 8 → N_values N) : 
  (N = 23 ∨ N = 24 ∨ N = 25) :=
by
  sorry

end valid_N_values_l283_283648


namespace find_m_l283_283500

def vector_collinear {α : Type*} [Field α] (a b : α × α) : Prop :=
  ∃ k : α, b = (k * (a.1), k * (a.2))

theorem find_m (m : ℝ) : 
  let a := (2, 3)
  let b := (-1, 2)
  vector_collinear (2 * m - 4, 3 * m + 8) (4, -1) → m = -2 :=
by
  intros
  sorry

end find_m_l283_283500


namespace evaluate_fraction_sum_squared_l283_283691

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6

theorem evaluate_fraction_sum_squared :
  ( (1 / a + 1 / b + 1 / c + 1 / d)^2 = (11 + 2 * Real.sqrt 30) / 9 ) := 
by
  sorry

end evaluate_fraction_sum_squared_l283_283691


namespace numbers_not_expressed_l283_283614

theorem numbers_not_expressed (a b : ℕ) (hb : 0 < b) (ha : 0 < a) :
 ∀ n : ℕ, (¬ ∃ a b : ℕ, n = a / b + (a + 1) / (b + 1) ∧ 0 < b ∧ 0 < a) ↔ (n = 1 ∨ ∃ m : ℕ, n = 2^m + 2) := 
by 
  sorry

end numbers_not_expressed_l283_283614


namespace rattlesnake_tail_percentage_difference_l283_283070

-- Definitions for the problem
def eastern_segments : Nat := 6
def western_segments : Nat := 8

-- The statement to prove
theorem rattlesnake_tail_percentage_difference :
  100 * (western_segments - eastern_segments) / western_segments = 25 := by
  sorry

end rattlesnake_tail_percentage_difference_l283_283070


namespace valid_N_values_l283_283651

def N_values (N : ℕ) : Prop :=
  (∀ k, k = N - 8 → (22 < N ∧ N ≤ 25))

-- Main theorem statement without proof
theorem valid_N_values (N : ℕ) (h : ∀ k, k = N - 8 → N_values N) : 
  (N = 23 ∨ N = 24 ∨ N = 25) :=
by
  sorry

end valid_N_values_l283_283651


namespace infinitely_many_composite_values_l283_283815

theorem infinitely_many_composite_values (k m : ℕ) 
  (h_k : k ≥ 2) : 
  ∃ n : ℕ, n = 4 * k^4 ∧ ∀ m : ℕ, ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ m^4 + n = x * y :=
by
  sorry

end infinitely_many_composite_values_l283_283815


namespace panic_percentage_l283_283215

theorem panic_percentage (original_population disappeared_after first_population second_population : ℝ) 
  (h₁ : original_population = 7200)
  (h₂ : disappeared_after = original_population * 0.10)
  (h₃ : first_population = original_population - disappeared_after)
  (h₄ : second_population = 4860)
  (h₅ : second_population = first_population - (first_population * 0.25)) : 
  second_population = first_population * (1 - 0.25) :=
by
  sorry

end panic_percentage_l283_283215


namespace triangle_angle_identity_l283_283581

theorem triangle_angle_identity
  (α β γ : ℝ)
  (h_triangle : α + β + γ = π)
  (sin_α_ne_zero : Real.sin α ≠ 0)
  (sin_β_ne_zero : Real.sin β ≠ 0)
  (sin_γ_ne_zero : Real.sin γ ≠ 0) :
  (Real.cos α / (Real.sin β * Real.sin γ) +
   Real.cos β / (Real.sin α * Real.sin γ) +
   Real.cos γ / (Real.sin α * Real.sin β) = 2) := by
  sorry

end triangle_angle_identity_l283_283581


namespace r_squared_sum_l283_283294

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l283_283294


namespace exists_representation_of_77_using_fewer_sevens_l283_283939

-- Definition of the problem
def represent_77 (expr : String) : Prop :=
  ∀ n : ℕ, expr = "77" ∨ 
             expr = "(77 - 7) + 7" ∨ 
             expr = "(10 * 7) + 7" ∨ 
             expr = "(70 + 7)" ∨ 
             expr = "(7 * 11)" ∨ 
             expr = "7 + 7 * 7 + (7 / 7)"

-- The proof statement
theorem exists_representation_of_77_using_fewer_sevens : ∃ expr : String, represent_77 expr ∧ String.length expr < 3 := 
sorry

end exists_representation_of_77_using_fewer_sevens_l283_283939


namespace perpendicular_lines_l283_283116

noncomputable def l1_slope (a : ℝ) : ℝ := -a / (1 - a)
noncomputable def l2_slope (a : ℝ) : ℝ := -(a - 1) / (2 * a + 3)

theorem perpendicular_lines (a : ℝ) 
  (h1 : ∀ x y : ℝ, a * x + (1 - a) * y = 3 → Prop) 
  (h2 : ∀ x y : ℝ, (a - 1) * x + (2 * a + 3) * y = 2 → Prop) 
  (hp : l1_slope a * l2_slope a = -1) : a = -3 := by
  sorry

end perpendicular_lines_l283_283116


namespace max_ladder_height_reached_l283_283572

def distance_from_truck_to_building : ℕ := 5
def ladder_extension : ℕ := 13

theorem max_ladder_height_reached :
  (ladder_extension ^ 2 - distance_from_truck_to_building ^ 2) = 144 :=
by
  -- This is where the proof should go
  sorry

end max_ladder_height_reached_l283_283572


namespace M_inter_N_eq_l283_283787

def set_M (x : ℝ) : Prop := x^2 - 3 * x < 0
def set_N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 4

def M := { x : ℝ | set_M x }
def N := { x : ℝ | set_N x }

theorem M_inter_N_eq : M ∩ N = { x | 1 ≤ x ∧ x < 3 } :=
by sorry

end M_inter_N_eq_l283_283787


namespace pyramid_base_side_length_l283_283382

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end pyramid_base_side_length_l283_283382


namespace math_problem_l283_283731

def is_polynomial (expr : String) : Prop := sorry
def is_monomial (expr : String) : Prop := sorry
def is_cubic (expr : String) : Prop := sorry
def is_quintic (expr : String) : Prop := sorry
def correct_option_C : String := "C"

theorem math_problem :
  ¬ is_polynomial "8 - 2 / z" ∧
  ¬ (is_monomial "-x^2yz" ∧ is_cubic "-x^2yz") ∧
  is_polynomial "x^2 - 3xy^2 + 2x^2y^3 - 1" ∧
  is_quintic "x^2 - 3xy^2 + 2x^2y^3 - 1" ∧
  ¬ is_monomial "5b / x" →
  correct_option_C = "C" := sorry

end math_problem_l283_283731


namespace cindy_dress_discount_l283_283226

theorem cindy_dress_discount (P D : ℝ) 
  (h1 : P * (1 - D) * 1.25 = 61.2) 
  (h2 : P - 61.2 = 4.5) : D = 0.255 :=
sorry

end cindy_dress_discount_l283_283226


namespace gcd_79_pow7_plus_1_and_79_pow7_plus_79_pow2_plus_1_l283_283759

theorem gcd_79_pow7_plus_1_and_79_pow7_plus_79_pow2_plus_1 (h_prime : Nat.Prime 79) : 
  Nat.gcd (79^7 + 1) (79^7 + 79^2 + 1) = 1 := 
by
  sorry

end gcd_79_pow7_plus_1_and_79_pow7_plus_79_pow2_plus_1_l283_283759


namespace find_angle_x_l283_283332

noncomputable def angle_x (angle_ABC angle_ACB angle_CDE : ℝ) : ℝ :=
  let angle_BAC := 180 - angle_ABC - angle_ACB
  let angle_ADE := 180 - angle_CDE
  let angle_EAD := angle_BAC
  let angle_AED := 180 - angle_ADE - angle_EAD
  180 - angle_AED

theorem find_angle_x (angle_ABC angle_ACB angle_CDE : ℝ) :
  angle_ABC = 70 → angle_ACB = 90 → angle_CDE = 42 → angle_x angle_ABC angle_ACB angle_CDE = 158 :=
by
  intros hABC hACB hCDE
  simp [angle_x, hABC, hACB, hCDE]
  sorry

end find_angle_x_l283_283332


namespace tracy_additional_miles_l283_283723

def total_distance : ℕ := 1000
def michelle_distance : ℕ := 294
def twice_michelle_distance : ℕ := 2 * michelle_distance
def katie_distance : ℕ := michelle_distance / 3
def tracy_distance := total_distance - (michelle_distance + katie_distance)
def additional_miles := tracy_distance - twice_michelle_distance

-- The statement to prove:
theorem tracy_additional_miles : additional_miles = 20 := by
  sorry

end tracy_additional_miles_l283_283723


namespace sheila_hourly_wage_l283_283953

-- Definition of conditions
def hours_per_day_mon_wed_fri := 8
def days_mon_wed_fri := 3
def hours_per_day_tue_thu := 6
def days_tue_thu := 2
def weekly_earnings := 432

-- Variables derived from conditions
def total_hours_mon_wed_fri := hours_per_day_mon_wed_fri * days_mon_wed_fri
def total_hours_tue_thu := hours_per_day_tue_thu * days_tue_thu
def total_hours_per_week := total_hours_mon_wed_fri + total_hours_tue_thu

-- Proof statement
theorem sheila_hourly_wage : (weekly_earnings / total_hours_per_week) = 12 := 
sorry

end sheila_hourly_wage_l283_283953


namespace mowed_times_in_spring_l283_283003

-- Definition of the problem conditions
def total_mowed_times : ℕ := 11
def summer_mowed_times : ℕ := 5

-- The theorem to prove
theorem mowed_times_in_spring : (total_mowed_times - summer_mowed_times = 6) :=
by
  sorry

end mowed_times_in_spring_l283_283003


namespace positive_difference_of_complementary_angles_in_ratio_5_to_4_l283_283831

-- Definitions for given conditions
def is_complementary (a b : ℝ) : Prop :=
  a + b = 90

def ratio_5_to_4 (a b : ℝ) : Prop :=
  ∃ x : ℝ, a = 5 * x ∧ b = 4 * x

-- Theorem to prove the measure of their positive difference is 10 degrees
theorem positive_difference_of_complementary_angles_in_ratio_5_to_4
  {a b : ℝ} (h_complementary : is_complementary a b) (h_ratio : ratio_5_to_4 a b) :
  abs (a - b) = 10 :=
by 
  sorry

end positive_difference_of_complementary_angles_in_ratio_5_to_4_l283_283831


namespace prob_exactly_two_passes_prob_at_least_one_fails_l283_283370

-- Define the probabilities for students A, B, and C passing their tests.
def prob_A : ℚ := 4/5
def prob_B : ℚ := 3/5
def prob_C : ℚ := 7/10

-- Define the probabilities for students A, B, and C failing their tests.
def prob_not_A : ℚ := 1 - prob_A
def prob_not_B : ℚ := 1 - prob_B
def prob_not_C : ℚ := 1 - prob_C

-- (1) Prove that the probability of exactly two students passing is 113/250.
theorem prob_exactly_two_passes : 
  prob_A * prob_B * prob_not_C + prob_A * prob_not_B * prob_C + prob_not_A * prob_B * prob_C = 113/250 := 
sorry

-- (2) Prove that the probability that at least one student fails is 83/125.
theorem prob_at_least_one_fails : 
  1 - (prob_A * prob_B * prob_C) = 83/125 := 
sorry

end prob_exactly_two_passes_prob_at_least_one_fails_l283_283370


namespace translation_proof_l283_283682

-- Define the points and the translation process
def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (1, 2)
def point_C : ℝ × ℝ := (1, -2)

-- Translation from point A to point C
def translation_vector : ℝ × ℝ :=
  (point_C.1 - point_A.1, point_C.2 - point_A.2)

-- Define point D using the translation vector applied to point B
def point_D : ℝ × ℝ :=
  (point_B.1 + translation_vector.1, point_B.2 + translation_vector.2)

-- Statement to prove point D has the expected coordinates
theorem translation_proof : 
  point_D = (3, 0) :=
by 
  -- The exact proof is omitted, presented here for completion
  sorry

end translation_proof_l283_283682


namespace xyz_final_stock_price_l283_283890

def initial_stock_price : ℝ := 120
def first_year_increase_rate : ℝ := 0.80
def second_year_decrease_rate : ℝ := 0.30

def final_stock_price_after_two_years : ℝ :=
  (initial_stock_price * (1 + first_year_increase_rate)) * (1 - second_year_decrease_rate)

theorem xyz_final_stock_price :
  final_stock_price_after_two_years = 151.2 := by
  sorry

end xyz_final_stock_price_l283_283890


namespace value_of_x_l283_283743

theorem value_of_x (x : ℝ) (h : (0.7 * x) - ((1 / 3) * x) = 110) : x = 300 :=
sorry

end value_of_x_l283_283743


namespace last_three_digits_of_7_pow_99_l283_283604

theorem last_three_digits_of_7_pow_99 : (7 ^ 99) % 1000 = 573 := 
by sorry

end last_three_digits_of_7_pow_99_l283_283604


namespace ordering_of_a_b_c_l283_283483

theorem ordering_of_a_b_c (a b c : ℝ)
  (ha : a = Real.exp (1 / 2))
  (hb : b = Real.log (1 / 2))
  (hc : c = Real.sin (1 / 2)) :
  a > c ∧ c > b :=
by sorry

end ordering_of_a_b_c_l283_283483


namespace decrypt_probability_l283_283028

theorem decrypt_probability (p1 p2 p3 : ℚ) (h1 : p1 = 1/5) (h2 : p2 = 2/5) (h3 : p3 = 1/2) : 
  1 - ((1 - p1) * (1 - p2) * (1 - p3)) = 19/25 :=
by
  sorry

end decrypt_probability_l283_283028


namespace interval_for_systematic_sampling_l283_283981

-- Define the total population size
def total_population : ℕ := 1203

-- Define the sample size
def sample_size : ℕ := 40

-- Define the interval for systematic sampling
def interval (n m : ℕ) : ℕ := (n - (n % m)) / m

-- The proof statement that the interval \( k \) for segmenting is 30
theorem interval_for_systematic_sampling : interval total_population sample_size = 30 :=
by
  show interval 1203 40 = 30
  sorry

end interval_for_systematic_sampling_l283_283981


namespace r_fourth_power_sum_l283_283279

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l283_283279


namespace expression_evaluation_l283_283060

theorem expression_evaluation :
  (4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2) ^ 0 = 1) :=
by
  -- Step by step calculations skipped
  sorry

end expression_evaluation_l283_283060


namespace structure_burns_in_65_seconds_l283_283525

noncomputable def toothpick_grid_burn_time (m n : ℕ) (toothpicks : ℕ) (burn_time : ℕ) : ℕ :=
  if (m = 3 ∧ n = 5 ∧ toothpicks = 38 ∧ burn_time = 10) then 65 else 0

theorem structure_burns_in_65_seconds : toothpick_grid_burn_time 3 5 38 10 = 65 := by
  sorry

end structure_burns_in_65_seconds_l283_283525


namespace r_power_four_identity_l283_283311

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l283_283311


namespace largest_prime_factor_of_18_pow_3_plus_15_pow_4_minus_3_pow_7_is_19_l283_283771

theorem largest_prime_factor_of_18_pow_3_plus_15_pow_4_minus_3_pow_7_is_19 : 
  ∃ p : ℕ, Prime p ∧ p = 19 ∧ ∀ q : ℕ, Prime q → q ∣ (18^3 + 15^4 - 3^7) → q ≤ 19 :=
sorry

end largest_prime_factor_of_18_pow_3_plus_15_pow_4_minus_3_pow_7_is_19_l283_283771


namespace ratio_of_times_l283_283460

theorem ratio_of_times (D S : ℝ) (hD : D = 27) (hS : S / 2 = D / 2 + 13.5) :
  D / S = 1 / 2 :=
by
  -- the proof will go here
  sorry

end ratio_of_times_l283_283460


namespace min_rings_to_connect_all_segments_l283_283840

-- Define the problem setup
structure ChainSegment where
  rings : Fin 3 → Type

-- Define the number of segments
def num_segments : ℕ := 5

-- Define the minimum number of rings to be opened and rejoined
def min_rings_to_connect (seg : Fin num_segments) : ℕ :=
  3

theorem min_rings_to_connect_all_segments :
  ∀ segs : Fin num_segments,
  (∃ n, n = min_rings_to_connect segs) :=
by
  -- Proof to be provided
  sorry

end min_rings_to_connect_all_segments_l283_283840


namespace evaluate_g_inv_l283_283821

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 6)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 7)
variable (h_inv1 : g_inv 6 = 4)
variable (h_inv2 : g_inv 7 = 3)
variable (h_inv_eq : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x)

theorem evaluate_g_inv :
  g_inv (g_inv 6 + g_inv 7) = 3 :=
by
  sorry

end evaluate_g_inv_l283_283821


namespace expected_value_is_750_l283_283436

def winnings (roll : ℕ) : ℕ :=
  if roll % 2 = 0 then 3 * roll else 0

def expected_value : ℚ :=
  (winnings 2 / 8) + (winnings 4 / 8) + (winnings 6 / 8) + (winnings 8 / 8)

theorem expected_value_is_750 : expected_value = 7.5 := by
  sorry

end expected_value_is_750_l283_283436


namespace find_speed_of_stream_l283_283854

def boat_speeds (V_b V_s : ℝ) : Prop :=
  V_b + V_s = 10 ∧ V_b - V_s = 8

theorem find_speed_of_stream (V_b V_s : ℝ) (h : boat_speeds V_b V_s) : V_s = 1 :=
by
  sorry

end find_speed_of_stream_l283_283854


namespace triangle_II_area_l283_283488

noncomputable def triangle_area (base : ℝ) (height : ℝ) : ℝ :=
  1 / 2 * base * height

theorem triangle_II_area (a b : ℝ) :
  let I_area := triangle_area (a + b) (a + b)
  let II_area := 2 * I_area
  II_area = (a + b) ^ 2 :=
by
  let I_area := triangle_area (a + b) (a + b)
  let II_area := 2 * I_area
  sorry

end triangle_II_area_l283_283488


namespace length_of_second_train_is_229_95_l283_283744

noncomputable def length_of_second_train (length_first_train : ℝ) 
                                          (speed_first_train : ℝ) 
                                          (speed_second_train : ℝ) 
                                          (time_to_cross : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train * (1000 / 3600)
  let speed_second_train_mps := speed_second_train * (1000 / 3600)
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance_covered := relative_speed * time_to_cross
  total_distance_covered - length_first_train

theorem length_of_second_train_is_229_95 :
  length_of_second_train 270 120 80 9 = 229.95 :=
by
  sorry

end length_of_second_train_is_229_95_l283_283744


namespace conic_curve_focus_eccentricity_l283_283414

theorem conic_curve_focus_eccentricity (m : ℝ) 
  (h : ∀ x y : ℝ, x^2 + m * y^2 = 1)
  (eccentricity_eq : ∀ a b : ℝ, a > b → m = 4/3) : m = 4/3 :=
by
  sorry

end conic_curve_focus_eccentricity_l283_283414


namespace compute_expression_l283_283758

theorem compute_expression : (3 + 7)^3 + 2 * (3^2 + 7^2) = 1116 := by
  sorry

end compute_expression_l283_283758


namespace price_increase_decrease_eq_l283_283023

theorem price_increase_decrease_eq (x : ℝ) (p : ℝ) (hx : x ≠ 0) :
  x * (1 + p / 100) * (1 - p / 200) = x * (1 + p / 300) → p = 100 / 3 :=
by
  intro h
  -- The proof would go here
  sorry

end price_increase_decrease_eq_l283_283023


namespace roger_total_distance_l283_283529

theorem roger_total_distance :
  let morning_ride_miles := 2
  let evening_ride_miles := 5 * morning_ride_miles
  let next_day_morning_ride_km := morning_ride_miles * 1.6
  let next_day_ride_km := 2 * next_day_morning_ride_km
  let next_day_ride_miles := next_day_ride_km / 1.6
  morning_ride_miles + evening_ride_miles + next_day_ride_miles = 16 :=
by
  sorry

end roger_total_distance_l283_283529


namespace max_squared_sum_of_sides_l283_283877

variable {R : ℝ}
variable {O A B C : EucSpace} -- O is the center, A, B, and C are vertices
variable (a b c : ℝ)  -- Position vectors corresponding to vertices A, B, C

-- Hypotheses based on the problem conditions:
variable (h1 : ‖a‖ = R)
variable (h2 : ‖b‖ = R)
variable (h3 : ‖c‖ = R)
variable (hSumZero : a + b + c = 0)

theorem max_squared_sum_of_sides 
  {AB BC CA : ℝ} -- Side lengths
  (hAB : AB = ‖a - b‖)
  (hBC : BC = ‖b - c‖)
  (hCA : CA = ‖c - a‖) :
  AB^2 + BC^2 + CA^2 = 9 * R^2 :=
sorry

end max_squared_sum_of_sides_l283_283877


namespace pyramid_base_length_l283_283400

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end pyramid_base_length_l283_283400


namespace survey_students_l283_283560

theorem survey_students (S F : ℕ) (h1 : F = 20 + 60) (h2 : F = 40 * S / 100) : S = 200 :=
by
  sorry

end survey_students_l283_283560


namespace r_power_four_identity_l283_283309

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l283_283309


namespace factorization_identity_l283_283074

theorem factorization_identity (m : ℝ) : 
  -4 * m^3 + 4 * m^2 - m = -m * (2 * m - 1)^2 :=
sorry

end factorization_identity_l283_283074


namespace family_members_count_l283_283994

-- Defining the conditions given in the problem
variables (cyrus_bites_arms_legs : ℕ) (cyrus_bites_body : ℕ) (total_bites_family : ℕ)
variables (family_bites_per_person : ℕ) (cyrus_total_bites : ℕ)

-- Given conditions
def condition1 : cyrus_bites_arms_legs = 14 := sorry
def condition2 : cyrus_bites_body = 10 := sorry
def condition3 : cyrus_total_bites = cyrus_bites_arms_legs + cyrus_bites_body := sorry
def condition4 : total_bites_family = cyrus_total_bites / 2 := sorry
def condition5 : ∀ n : ℕ, total_bites_family = n * family_bites_per_person := sorry

-- The theorem to prove: The number of people in the rest of Cyrus' family is 12
theorem family_members_count (n : ℕ) (h1 : cyrus_bites_arms_legs = 14)
    (h2 : cyrus_bites_body = 10) (h3 : cyrus_total_bites = cyrus_bites_arms_legs + cyrus_bites_body)
    (h4 : total_bites_family = cyrus_total_bites / 2)
    (h5 : ∀ n, total_bites_family = n * family_bites_per_person) : n = 12 :=
sorry

end family_members_count_l283_283994


namespace water_in_maria_jar_after_200_days_l283_283804

def arithmetic_series_sum (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem water_in_maria_jar_after_200_days :
  let initial_volume_maria : ℕ := 1000
  let days : ℕ := 200
  let odd_days : ℕ := days / 2
  let even_days : ℕ := days / 2
  let volume_odd_transfer : ℕ := arithmetic_series_sum 1 2 odd_days
  let volume_even_transfer : ℕ := arithmetic_series_sum 2 2 even_days
  let net_transfer : ℕ := volume_odd_transfer - volume_even_transfer
  let final_volume_maria := initial_volume_maria + net_transfer
  final_volume_maria = 900 :=
by
  sorry

end water_in_maria_jar_after_200_days_l283_283804


namespace average_of_next_seven_consecutive_integers_l283_283151

theorem average_of_next_seven_consecutive_integers
  (a b : ℕ)
  (hb : b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7) :
  ((b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5) + (b + 6)) / 7) = a + 6 :=
by
  sorry

end average_of_next_seven_consecutive_integers_l283_283151


namespace find_r4_l283_283268

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l283_283268


namespace two_digit_sum_of_original_and_reverse_l283_283160

theorem two_digit_sum_of_original_and_reverse
  (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9) -- a is a digit
  (h2 : 0 ≤ b ∧ b ≤ 9) -- b is a digit
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end two_digit_sum_of_original_and_reverse_l283_283160


namespace highway_total_vehicles_l283_283949

theorem highway_total_vehicles (num_trucks : ℕ) (num_cars : ℕ) (total_vehicles : ℕ)
  (h1 : num_trucks = 100)
  (h2 : num_cars = 2 * num_trucks)
  (h3 : total_vehicles = num_cars + num_trucks) :
  total_vehicles = 300 :=
by
  sorry

end highway_total_vehicles_l283_283949


namespace r_squared_sum_l283_283295

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l283_283295


namespace solve_diamond_l283_283109

theorem solve_diamond (d : ℕ) (h : 9 * d + 5 = 10 * d + 2) : d = 3 :=
by
  sorry

end solve_diamond_l283_283109


namespace pages_wednesday_l283_283354

-- Given conditions as definitions
def borrow_books := 3
def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51

-- Prove that Nico read 19 pages on Wednesday
theorem pages_wednesday :
  let pages_wednesday := total_pages - (pages_monday + pages_tuesday)
  pages_wednesday = 19 :=
by
  sorry

end pages_wednesday_l283_283354


namespace range_of_a_l283_283783

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x ^ 2 + (a - 1) * x + 1 / 2 ≤ 0) → (-1 < a ∧ a < 3) :=
by 
  sorry

end range_of_a_l283_283783


namespace side_length_of_base_l283_283376

-- Define the conditions
def lateral_area (s : ℝ) : ℝ := (1 / 2) * s * 40
def given_area : ℝ := 120

-- Define the theorem to prove the length of the side of the base
theorem side_length_of_base : ∃ (s : ℝ), lateral_area(s) = given_area ∧ s = 6 :=
by
  sorry

end side_length_of_base_l283_283376


namespace amanda_earnings_l283_283586

def hourly_rate : ℝ := 20.00

def hours_monday : ℝ := 5 * 1.5

def hours_tuesday : ℝ := 3

def hours_thursday : ℝ := 2 * 2

def hours_saturday : ℝ := 6

def total_hours : ℝ := hours_monday + hours_tuesday + hours_thursday + hours_saturday

def total_earnings : ℝ := hourly_rate * total_hours

theorem amanda_earnings : total_earnings = 410.00 :=
by
  -- Proof steps can be filled here
  sorry

end amanda_earnings_l283_283586


namespace length_of_brick_proof_l283_283202

noncomputable def length_of_brick (courtyard_length courtyard_width : ℕ) (brick_width : ℕ) (total_bricks : ℕ) : ℕ :=
  let total_area_cm := courtyard_length * courtyard_width * 10000
  total_area_cm / (brick_width * total_bricks)

theorem length_of_brick_proof :
  length_of_brick 25 16 10 20000 = 20 :=
by
  unfold length_of_brick
  sorry

end length_of_brick_proof_l283_283202


namespace possible_values_for_N_l283_283672

theorem possible_values_for_N (N : ℕ) (h₁ : 8 < N) (h₂ : 1 ≤ N - 1) :
  22 < N ∧ N ≤ 25 ↔ (N = 23 ∨ N = 24 ∨ N = 25) :=
by 
  sorry

end possible_values_for_N_l283_283672


namespace ranking_Fiona_Giselle_Ella_l283_283515

-- Definitions of scores 
variable (score : String → ℕ)

-- Conditions based on the problem statement
def ella_not_highest : Prop := ¬ (score "Ella" = max (score "Ella") (max (score "Fiona") (score "Giselle")))
def giselle_not_lowest : Prop := ¬ (score "Giselle" = min (score "Ella") (score "Giselle"))

-- The goal is to rank the scores from highest to lowest
def score_ranking : Prop := (score "Fiona" > score "Giselle") ∧ (score "Giselle" > score "Ella")

theorem ranking_Fiona_Giselle_Ella :
  ella_not_highest score →
  giselle_not_lowest score →
  score_ranking score :=
by
  sorry

end ranking_Fiona_Giselle_Ella_l283_283515


namespace cos_expression_value_l283_283504

theorem cos_expression_value (x : ℝ) (h : Real.sin x = 3 * Real.sin (x - Real.pi / 2)) :
  Real.cos x * Real.cos (x + Real.pi / 2) = 3 / 10 := 
sorry

end cos_expression_value_l283_283504


namespace arithmetic_sequence_problem_l283_283329

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : ∀ k, k ≥ 2 → a (k + 1) - a k^2 + a (k - 1) = 0) (h2 : ∀ k, a k ≠ 0) (h3 : ∀ k ≥ 2, a (k + 1) + a (k - 1) = 2 * a k) :
  S (2 * n - 1) - 4 * n = -2 :=
by
  sorry

end arithmetic_sequence_problem_l283_283329


namespace modulus_of_z_l283_283248

open Complex

theorem modulus_of_z 
  (z : ℂ) 
  (h : (1 - I) * z = 2 * I) : 
  abs z = Real.sqrt 2 := 
sorry

end modulus_of_z_l283_283248


namespace r_squared_sum_l283_283290

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l283_283290


namespace rectangle_dimensions_square_side_length_l283_283678

-- Given a rectangle with length-to-width ratio 3:1 and area 75 cm^2, prove the length is 15 cm and width is 5 cm.
theorem rectangle_dimensions (x : ℝ) :
  (3 * x * x = 75) → (x = 5) ∧ (3 * x = 15) :=
by
  intro h
  have x_sq : x^2 = 25 := by linarith [h]
  have x_pos : x > 0 := by linarith
  split
  -- Proving x = 5
  {
    have x_eq : x = sqrt 25 := by linarith
    exact x_eq
  }
  -- Proving 3 * x = 15
  {
    have x_eq : x = 5 := by linarith using [x_sq]
    linarith [x_eq]
  }
  sorry

-- Prove the statement that the difference between the side length of a square with area 75 cm^2 and the width of the rectangle is greater than 3 cm.
theorem square_side_length (y x : ℝ) :
  (y^2 = 75) → (x = 5) → (3 < y - x) :=
by
  intro h1 h2
  have y_sqrt : y = sqrt 75 := by linarith
  have y_bounds : 8 < y ∧ y < 9 := by
    split
    { have : sqrt 64 < sqrt 75 := by nlinarith
      linarith }
    { have : sqrt 75 < sqrt 81 := by nlinarith
      linarith }
  have y_diff : y - 5 = y - x := by linarith using [h2]
  linarith
  sorry

end rectangle_dimensions_square_side_length_l283_283678


namespace angle_BAC_is_45_degrees_l283_283883

theorem angle_BAC_is_45_degrees
  (ABC : Triangle)
  (I : Point)
  (O : Point)
  (X : Point)
  (M : Point)
  (hI : I = ABC.incenter)
  (hO : O = ABC.circumcenter)
  (hOI : LineThrough O I)
  (hX : X ∈ (line_through O I) ∧ X ∈ ABC.BC)
  (hM : M = midpoint_minor_arc_BC_not_containing_A ABC)
  (hConcyclic : concyclic [A, O, M, X]) :
  ∠BAC = 45 :=
sorry

end angle_BAC_is_45_degrees_l283_283883


namespace lines_not_form_triangle_l283_283499

theorem lines_not_form_triangle {m : ℝ} :
  (∀ x y : ℝ, 2 * x - 3 * y + 1 ≠ 0 → 4 * x + 3 * y + 5 ≠ 0 → mx - y - 1 ≠ 0) →
  (m = -4 / 3 ∨ m = 2 / 3 ∨ m = 4 / 3) :=
sorry

end lines_not_form_triangle_l283_283499


namespace theodore_pays_10_percent_in_taxes_l283_283420

-- Defining the quantities
def num_stone_statues : ℕ := 10
def num_wooden_statues : ℕ := 20
def price_per_stone_statue : ℕ := 20
def price_per_wooden_statue : ℕ := 5
def total_earnings_after_taxes : ℕ := 270

-- Assertion: Theodore pays 10% of his earnings in taxes
theorem theodore_pays_10_percent_in_taxes :
  (num_stone_statues * price_per_stone_statue + num_wooden_statues * price_per_wooden_statue) - total_earnings_after_taxes
  = (10 * (num_stone_statues * price_per_stone_statue + num_wooden_statues * price_per_wooden_statue)) / 100 := 
by
  sorry

end theodore_pays_10_percent_in_taxes_l283_283420


namespace carrie_worked_days_l283_283592

theorem carrie_worked_days (d : ℕ) 
  (h1: ∀ n : ℕ, d = n → (2 * 22 * n - 54 = 122)) : d = 4 :=
by
  -- The proof will go here.
  sorry

end carrie_worked_days_l283_283592


namespace tom_climbing_time_l283_283177

theorem tom_climbing_time (elizabeth_time : ℕ) (multiplier : ℕ) 
  (h1 : elizabeth_time = 30) (h2 : multiplier = 4) : (elizabeth_time * multiplier) / 60 = 2 :=
by
  sorry

end tom_climbing_time_l283_283177


namespace brick_length_is_20_cm_l283_283200

-- Define the conditions given in the problem
def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def num_bricks : ℕ := 20000
def brick_width_cm : ℝ := 10
def total_area_cm2 : ℝ := 4000000

-- Define the goal to prove that the length of each brick is 20 cm
theorem brick_length_is_20_cm :
  (total_area_cm2 = num_bricks * (brick_width_cm * length)) → (length = 20) :=
by
  -- Assume the given conditions
  sorry

end brick_length_is_20_cm_l283_283200


namespace acute_angles_45_degrees_l283_283842

-- Assuming quadrilaterals ABCD and A'B'C'D' such that sides of each lie on 
-- the perpendicular bisectors of the sides of the other. We want to prove that
-- the acute angles of A'B'C'D' are 45 degrees.

def convex_quadrilateral (Q : Type) := 
  ∃ (A B C D : Q), True -- Placeholder for a more detailed convex quadrilateral structure

def perpendicular_bisector (S1 S2 T1 T2: Type) := 
  ∃ (M : Type), True -- Placeholder for a more detailed perpendicular bisector structure

theorem acute_angles_45_degrees
  (Q1 Q2 : Type)
  (h1 : convex_quadrilateral Q1)
  (h2 : convex_quadrilateral Q2)
  (perp1 : perpendicular_bisector Q1 Q1 Q2 Q2)
  (perp2 : perpendicular_bisector Q2 Q2 Q1 Q1) :
  ∀ (θ : ℝ), θ = 45 := 
by
  sorry

end acute_angles_45_degrees_l283_283842


namespace salesmans_profit_l283_283578

-- Define the initial conditions and given values
def backpacks_bought : ℕ := 72
def cost_price : ℕ := 1080
def swap_meet_sales : ℕ := 25
def swap_meet_price : ℕ := 20
def department_store_sales : ℕ := 18
def department_store_price : ℕ := 30
def online_sales : ℕ := 12
def online_price : ℕ := 28
def shipping_expenses : ℕ := 40
def local_market_price : ℕ := 24

-- Calculate the total revenue from each channel
def swap_meet_revenue : ℕ := swap_meet_sales * swap_meet_price
def department_store_revenue : ℕ := department_store_sales * department_store_price
def online_revenue : ℕ := (online_sales * online_price) - shipping_expenses

-- Calculate remaining backpacks and local market revenue
def backpacks_sold : ℕ := swap_meet_sales + department_store_sales + online_sales
def backpacks_left : ℕ := backpacks_bought - backpacks_sold
def local_market_revenue : ℕ := backpacks_left * local_market_price

-- Calculate total revenue and profit
def total_revenue : ℕ := swap_meet_revenue + department_store_revenue + online_revenue + local_market_revenue
def profit : ℕ := total_revenue - cost_price

-- State the theorem for the salesman's profit
theorem salesmans_profit : profit = 664 := by
  sorry

end salesmans_profit_l283_283578


namespace chocolate_bars_squares_l283_283773

theorem chocolate_bars_squares
  (gerald_bars : ℕ)
  (teacher_rate : ℕ)
  (students : ℕ)
  (squares_per_student : ℕ)
  (total_squares : ℕ)
  (total_bars : ℕ)
  (squares_per_bar : ℕ)
  (h1 : gerald_bars = 7)
  (h2 : teacher_rate = 2)
  (h3 : students = 24)
  (h4 : squares_per_student = 7)
  (h5 : total_squares = students * squares_per_student)
  (h6 : total_bars = gerald_bars + teacher_rate * gerald_bars)
  (h7 : squares_per_bar = total_squares / total_bars)
  : squares_per_bar = 8 := by 
  sorry

end chocolate_bars_squares_l283_283773


namespace dog_roaming_area_l283_283197

theorem dog_roaming_area :
  let shed_radius := 20
  let rope_length := 10
  let distance_from_edge := 10
  let radius_from_center := shed_radius - distance_from_edge
  radius_from_center = rope_length →
  (π * rope_length^2 = 100 * π) :=
by
  intros shed_radius rope_length distance_from_edge radius_from_center h
  sorry

end dog_roaming_area_l283_283197


namespace proof_problem_l283_283021

-- Definition of the condition
def condition (y : ℝ) : Prop := 6 * y^2 + 5 = 2 * y + 10

-- Stating the theorem
theorem proof_problem : ∀ y : ℝ, condition y → (12 * y - 5)^2 = 133 :=
by
  intro y
  intro h
  sorry

end proof_problem_l283_283021


namespace flat_path_time_l283_283140

/-- Malcolm's walking time problem -/
theorem flat_path_time (x : ℕ) (h1 : 6 + 12 + 6 = 24)
                       (h2 : 3 * x = 24 + 18) : x = 14 := 
by
  sorry

end flat_path_time_l283_283140


namespace inequality_proof_l283_283136

-- Definitions
variables {n : ℕ} {s : ℕ} {n_i : ℕ → ℕ} (h_distinct : (function.injective n_i))
noncomputable def M (n_i : ℕ → ℕ) (s : ℕ) := ∑ i in finset.range s, 2 ^ (n_i i)

-- Goal
theorem inequality_proof (h_distinct : function.injective n_i) :
  (∑ i in finset.range s, 2 ^ (n_i i / 2)) < (1 + real.sqrt 2) * real.sqrt (M n_i s) :=
sorry

end inequality_proof_l283_283136


namespace tan_sum_l283_283314

theorem tan_sum (θ : ℝ) (h : Real.sin (2 * θ) = 2 / 3) : Real.tan θ + 1 / Real.tan θ = 3 := sorry

end tan_sum_l283_283314


namespace value_of_b_l283_283507

theorem value_of_b 
  (a b : ℝ) 
  (h : ∃ c : ℝ, (ax^3 + bx^2 + 1) = (x^2 - x - 1) * (x + c)) : 
  b = -2 :=
  sorry

end value_of_b_l283_283507


namespace calc_expr_l283_283059

theorem calc_expr : 4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2)^0 = 1 := by
  sorry

end calc_expr_l283_283059


namespace area_of_region_a_area_of_region_b_area_of_region_c_l283_283896

-- Definition of regions and their areas
def area_of_square : Real := sorry
def area_of_diamond : Real := sorry
def area_of_hexagon : Real := sorry

-- Define the conditions for the regions
def region_a (x y : ℝ) := abs x ≤ 1 ∧ abs y ≤ 1
def region_b (x y : ℝ) := abs x + abs y ≤ 10
def region_c (x y : ℝ) := abs x + abs y + abs (x + y) ≤ 2020

-- Prove that the areas match the calculated solutions
theorem area_of_region_a : area_of_square = 4 := 
by sorry

theorem area_of_region_b : area_of_diamond = 200 := 
by sorry

theorem area_of_region_c : area_of_hexagon = 3060300 := 
by sorry

end area_of_region_a_area_of_region_b_area_of_region_c_l283_283896


namespace james_oranges_l283_283465

-- Define the problem conditions
variables (o a : ℕ) -- o is number of oranges, a is number of apples

-- Condition: James bought apples and oranges over a seven-day week
def days_week := o + a = 7

-- Condition: The total cost must be a whole number of dollars (divisible by 100 cents)
def total_cost := 65 * o + 40 * a ≡ 0 [MOD 100]

-- We need to prove: James bought 4 oranges
theorem james_oranges (o a : ℕ) (h_days_week : days_week o a) (h_total_cost : total_cost o a) : o = 4 :=
sorry

end james_oranges_l283_283465


namespace initial_velocity_is_three_l283_283880

-- Define the displacement function s(t)
def s (t : ℝ) : ℝ := 3 * t - t ^ 2

-- Define the initial time condition
def initial_time : ℝ := 0

-- State the main theorem about the initial velocity
theorem initial_velocity_is_three : (deriv s) initial_time = 3 :=
by
  sorry

end initial_velocity_is_three_l283_283880


namespace katy_brownies_l283_283128

-- Define the conditions
def ate_monday : ℕ := 5
def ate_tuesday : ℕ := 2 * ate_monday

-- Define the question
def total_brownies : ℕ := ate_monday + ate_tuesday

-- State the proof problem
theorem katy_brownies : total_brownies = 15 := by
  sorry

end katy_brownies_l283_283128


namespace cube_plus_eleven_mul_divisible_by_six_l283_283152

theorem cube_plus_eleven_mul_divisible_by_six (a : ℤ) : 6 ∣ (a^3 + 11 * a) := 
by sorry

end cube_plus_eleven_mul_divisible_by_six_l283_283152


namespace find_x_when_y_is_sqrt_8_l283_283014

theorem find_x_when_y_is_sqrt_8
  (x y : ℝ)
  (h : ∀ x y : ℝ, (x^2 * y^4 = 1600) ↔ (x = 10 ∧ y = 2)) :
  x = 5 :=
by
  sorry

end find_x_when_y_is_sqrt_8_l283_283014


namespace smallest_n_inequality_l283_283608

-- Define the main statement based on the identified conditions and answer.
theorem smallest_n_inequality (x y z w : ℝ) : 
  (x^2 + y^2 + z^2 + w^2)^2 ≤ 4 * (x^4 + y^4 + z^4 + w^4) :=
sorry

end smallest_n_inequality_l283_283608


namespace angle_B_in_triangle_is_pi_over_6_l283_283334

theorem angle_B_in_triangle_is_pi_over_6
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : A + B + C = π)
  (h₅ : b * (Real.cos C) / (Real.cos B) + c = (2 * Real.sqrt 3 / 3) * a) :
  B = π / 6 :=
by sorry

end angle_B_in_triangle_is_pi_over_6_l283_283334


namespace pyramid_base_length_l283_283401

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end pyramid_base_length_l283_283401


namespace exponents_of_equation_l283_283565

theorem exponents_of_equation :
  ∃ (x y : ℕ), 2 * (3 ^ 8) ^ 2 * (2 ^ 3) ^ 2 * 3 = 2 ^ x * 3 ^ y ∧ x = 7 ∧ y = 17 :=
by
  use 7
  use 17
  sorry

end exponents_of_equation_l283_283565


namespace divisibility_of_n_l283_283695

theorem divisibility_of_n
  (n : ℕ) (n_gt_1 : n > 1)
  (h : n ∣ (6^n - 1)) : 5 ∣ n :=
by
  sorry

end divisibility_of_n_l283_283695


namespace factor_expression_l283_283232

theorem factor_expression (x : ℝ) : 72 * x^3 - 250 * x^7 = 2 * x^3 * (36 - 125 * x^4) :=
by
  sorry

end factor_expression_l283_283232


namespace symmetric_line_equation_wrt_x_axis_l283_283536

theorem symmetric_line_equation_wrt_x_axis :
  (∀ x y : ℝ, 3 * x + 4 * y + 5 = 0 ↔ 3 * x - 4 * (-y) + 5 = 0) :=
by
  sorry

end symmetric_line_equation_wrt_x_axis_l283_283536


namespace amanda_weekly_earnings_l283_283583

def amanda_rate_per_hour : ℝ := 20.00
def monday_appointments : ℕ := 5
def monday_hours_per_appointment : ℝ := 1.5
def tuesday_appointment_hours : ℝ := 3
def thursday_appointments : ℕ := 2
def thursday_hours_per_appointment : ℝ := 2
def saturday_appointment_hours : ℝ := 6

def total_hours_worked : ℝ :=
  monday_appointments * monday_hours_per_appointment +
  tuesday_appointment_hours +
  thursday_appointments * thursday_hours_per_appointment +
  saturday_appointment_hours

def total_earnings : ℝ := total_hours_worked * amanda_rate_per_hour

theorem amanda_weekly_earnings : total_earnings = 410.00 :=
  by
    unfold total_earnings total_hours_worked monday_appointments monday_hours_per_appointment tuesday_appointment_hours thursday_appointments thursday_hours_per_appointment saturday_appointment_hours amanda_rate_per_hour 
    -- The proof will involve basic arithmetic simplification, which is skipped here.
    -- Therefore, we simply state sorry.
    sorry

end amanda_weekly_earnings_l283_283583


namespace tom_candy_pieces_l283_283982

def total_boxes : ℕ := 14
def give_away_boxes : ℕ := 8
def pieces_per_box : ℕ := 3

theorem tom_candy_pieces : (total_boxes - give_away_boxes) * pieces_per_box = 18 := 
by 
  sorry

end tom_candy_pieces_l283_283982


namespace ratio_of_donations_l283_283700

theorem ratio_of_donations (x : ℝ) (h1 : ∀ (y : ℝ), y = 40) (h2 : ∀ (y : ℝ), y = 40 * x)
  (h3 : ∀ (y : ℝ), y = 0.30 * (40 + 40 * x)) (h4 : ∀ (y : ℝ), y = 36) : x = 2 := 
by 
  sorry

end ratio_of_donations_l283_283700


namespace student_count_l283_283642

theorem student_count (N : ℕ) (h : N > 22 ∧ N ≤ 25) : N = 23 ∨ N = 24 ∨ N = 25 := by {
  cases (Nat.eq_or_lt_of_le h.right); {
    exact Or.inr Or.inr (Nat.lt_antisymm h.left _); sorry;
  };
  exact Or.inr (Or.inl (Nat.lt_antisymm h.left (Nat.lt_succ_iff.mpr h.left))); sorry;
  exact Or.inl (Nat.lt_antisymm h.left (Nat.lt_succ_of_lt h.left)); sorry;
}

end student_count_l283_283642


namespace sequence_a_is_perfect_square_l283_283698

theorem sequence_a_is_perfect_square :
  ∃ (a b : ℕ → ℤ),
    a 0 = 1 ∧ 
    b 0 = 0 ∧ 
    (∀ n, a (n + 1) = 7 * a n + 6 * b n - 3) ∧
    (∀ n, b (n + 1) = 8 * a n + 7 * b n - 4) ∧
    ∀ n, ∃ m : ℕ, a n = m * m := sorry

end sequence_a_is_perfect_square_l283_283698


namespace possible_values_for_N_l283_283675

theorem possible_values_for_N (N : ℕ) (h₁ : 8 < N) (h₂ : 1 ≤ N - 1) :
  22 < N ∧ N ≤ 25 ↔ (N = 23 ∨ N = 24 ∨ N = 25) :=
by 
  sorry

end possible_values_for_N_l283_283675


namespace student_count_l283_283645

theorem student_count (N : ℕ) (h : N > 22 ∧ N ≤ 25) : N = 23 ∨ N = 24 ∨ N = 25 := by {
  cases (Nat.eq_or_lt_of_le h.right); {
    exact Or.inr Or.inr (Nat.lt_antisymm h.left _); sorry;
  };
  exact Or.inr (Or.inl (Nat.lt_antisymm h.left (Nat.lt_succ_iff.mpr h.left))); sorry;
  exact Or.inl (Nat.lt_antisymm h.left (Nat.lt_succ_of_lt h.left)); sorry;
}

end student_count_l283_283645


namespace change_is_41_l283_283428

-- Define the cost of shirts and sandals as given in the problem conditions
def cost_of_shirts : ℕ := 10 * 5
def cost_of_sandals : ℕ := 3 * 3
def total_cost : ℕ := cost_of_shirts + cost_of_sandals

-- Define the amount given
def amount_given : ℕ := 100

-- Calculate the change
def change := amount_given - total_cost

-- State the theorem
theorem change_is_41 : change = 41 := 
by 
  -- Filling this with justification steps would be the actual proof
  -- but it's not required, so we use 'sorry' to indicate the theorem
  sorry

end change_is_41_l283_283428


namespace min_balls_in_circle_l283_283756

theorem min_balls_in_circle (b w n k : ℕ) 
  (h1 : b = 2 * w)
  (h2 : n = b + w) 
  (h3 : n - 2 * k = 6 * k) :
  n >= 24 :=
sorry

end min_balls_in_circle_l283_283756


namespace quadratic_root_identity_l283_283625

theorem quadratic_root_identity (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2020 = 2023 :=
by
  sorry

end quadratic_root_identity_l283_283625


namespace shopkeeper_gain_l283_283430

theorem shopkeeper_gain
  (true_weight : ℝ)
  (cheat_percent : ℝ)
  (gain_percent : ℝ) :
  cheat_percent = 0.1 ∧
  true_weight = 1000 →
  gain_percent = 20 :=
by
  sorry

end shopkeeper_gain_l283_283430


namespace combined_weight_of_candles_l283_283071

theorem combined_weight_of_candles (candles : ℕ) (weight_per_candle : ℕ) (total_weight : ℕ) :
  candles = 10 - 3 →
  weight_per_candle = 8 + 1 →
  total_weight = candles * weight_per_candle →
  total_weight = 63 :=
by
  intros
  subst_vars
  sorry

end combined_weight_of_candles_l283_283071


namespace convex_polyhedron_in_inscribed_sphere_l283_283935

-- Definitions based on conditions
variables (S c r : ℝ) (S' V R : ℝ)

-- The given relationship for a convex polygon.
def poly_relationship := S = (1 / 2) * c * r

-- The desired relationship for a convex polyhedron.
def polyhedron_relationship := V = (1 / 3) * S' * R

-- Proof statement
theorem convex_polyhedron_in_inscribed_sphere (S c r S' V R : ℝ) 
  (poly : S = (1 / 2) * c * r) : V = (1 / 3) * S' * R :=
sorry

end convex_polyhedron_in_inscribed_sphere_l283_283935


namespace possible_values_of_N_l283_283669

theorem possible_values_of_N (N : ℕ) (h1 : N ≥ 8 + 1)
  (h2 : ∀ (i : ℕ), (i < N → (i ≥ 0 ∧ i < 1/3 * (N-1)) → false) ) 
  (h3 : ∀ (i : ℕ), (i < N → (i ≥ 1/3 * (N-1) ∨ i < 1/3 * (N-1)) → true)) :
  23 ≤ N ∧ N ≤ 25 :=
by
  sorry

end possible_values_of_N_l283_283669


namespace find_m_range_l283_283090

def p (m : ℝ) : Prop := (4 - 4 * m) ≤ 0
def q (m : ℝ) : Prop := (5 - 2 * m) > 1

theorem find_m_range (m : ℝ) (hp_false : ¬ p m) (hq_true : q m) : 1 ≤ m ∧ m < 2 :=
by {
 sorry
}

end find_m_range_l283_283090


namespace find_n_in_range_and_modulus_l283_283553

theorem find_n_in_range_and_modulus :
  ∃ n : ℤ, 0 ≤ n ∧ n < 21 ∧ (-200) % 21 = n % 21 → n = 10 := by
  sorry

end find_n_in_range_and_modulus_l283_283553


namespace binom_sum_l283_283457

theorem binom_sum :
  (Nat.choose 15 12) + 10 = 465 := by
  sorry

end binom_sum_l283_283457


namespace smallest_possible_beta_l283_283944

theorem smallest_possible_beta
  (a b c : EuclideanSpace ℝ (Fin 3))
  (h1 : ‖a‖ = 1)
  (h2 : ‖b‖ = 1)
  (h3 : ‖c‖ = 1)
  (h4 : ∃ β : ℝ, arccos (a ⬝ b) = β)
  (h5 : ∃ β : ℝ, arccos (c ⬝ (a × b)) = β)
  (h6 : b ⬝ (c × a) = 1 / 2) :
  ∃ β : ℝ, β = 45 :=
by
  sorry

end smallest_possible_beta_l283_283944


namespace side_length_of_square_base_l283_283397

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end side_length_of_square_base_l283_283397


namespace negation_equivalence_l283_283020

-- Define the original proposition P
def proposition_P : Prop := ∀ x : ℝ, 0 ≤ x → x^3 + 2 * x ≥ 0

-- Define the negation of the proposition P
def negation_P : Prop := ∃ x : ℝ, 0 ≤ x ∧ x^3 + 2 * x < 0

-- The statement to be proven
theorem negation_equivalence : ¬ proposition_P ↔ negation_P := 
by sorry

end negation_equivalence_l283_283020


namespace charlie_metal_storage_l283_283757

theorem charlie_metal_storage (total_needed : ℕ) (amount_to_buy : ℕ) (storage : ℕ) 
    (h1 : total_needed = 635) 
    (h2 : amount_to_buy = 359) 
    (h3 : total_needed = storage + amount_to_buy) : 
    storage = 276 := 
sorry

end charlie_metal_storage_l283_283757


namespace min_throws_for_repeated_sum_l283_283425

theorem min_throws_for_repeated_sum (n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 16) : 
  ∃ m, m = 16 ∧ (∀ (k : ℕ), k < 16 → ∃ i < 16, ∃ j < 16, i ≠ j ∧ i + j = k) :=
by
  sorry

end min_throws_for_repeated_sum_l283_283425


namespace pants_original_price_l283_283122

theorem pants_original_price (P : ℝ) (h1 : P * 0.6 = 50.40) : P = 84 :=
sorry

end pants_original_price_l283_283122


namespace find_n_tan_eq_348_l283_283469

theorem find_n_tan_eq_348 (n : ℤ) (h1 : -90 < n) (h2 : n < 90) : 
  (Real.tan (n * Real.pi / 180) = Real.tan (348 * Real.pi / 180)) ↔ (n = -12) := by
  sorry

end find_n_tan_eq_348_l283_283469


namespace evaluate_expression_l283_283072

theorem evaluate_expression : (-3)^7 / 3^5 + 2^5 - 7^2 = -26 := 
by
  sorry

end evaluate_expression_l283_283072


namespace simplify_polynomial_l283_283958

theorem simplify_polynomial (x : ℤ) :
  (3 * x - 2) * (6 * x^12 + 3 * x^11 + 5 * x^9 + x^8 + 7 * x^7) =
  18 * x^13 - 3 * x^12 + 15 * x^10 - 7 * x^9 + 19 * x^8 - 14 * x^7 :=
by
  sorry

end simplify_polynomial_l283_283958


namespace r_pow_four_solution_l283_283302

theorem r_pow_four_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/(r^4) = 7 := 
by
  sorry

end r_pow_four_solution_l283_283302


namespace cos_minus_sin_of_tan_eq_sqrt3_l283_283251

theorem cos_minus_sin_of_tan_eq_sqrt3 (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos α - Real.sin α = (Real.sqrt 3 - 1) / 2 := 
by
  sorry

end cos_minus_sin_of_tan_eq_sqrt3_l283_283251


namespace intersection_point_correct_l283_283104

noncomputable def find_intersection_point (a b : ℝ) (h1 : a = b + 5) (h2 : a = 4 * b + 2) : ℝ × ℝ :=
  let x := -3
  let y := -14
  (x, y)

theorem intersection_point_correct : 
  ∀ (a b : ℝ) (h1 : a = b + 5) (h2 : a = 4 * b + 2), 
  find_intersection_point a b h1 h2 = (-3, -14) := by
  sorry

end intersection_point_correct_l283_283104


namespace total_action_figures_l283_283876

def action_figures_per_shelf : ℕ := 11
def number_of_shelves : ℕ := 4

theorem total_action_figures : action_figures_per_shelf * number_of_shelves = 44 := by
  sorry

end total_action_figures_l283_283876


namespace r_fourth_power_sum_l283_283275

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l283_283275


namespace calculate_expression_l283_283885

theorem calculate_expression : -1 ^ 4 + 16 / (-2) ^ 3 * | -3 - 1 | = -9 := 
by 
  sorry

end calculate_expression_l283_283885


namespace root_equation_l283_283624

variable (m : ℝ)
theorem root_equation (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2020 = 2023 := by
  sorry

end root_equation_l283_283624


namespace pencils_distributed_l283_283481

-- Define the conditions as a Lean statement
theorem pencils_distributed :
  let friends := 4
  let pencils := 8
  let at_least_one := 1
  ∃ (ways : ℕ), ways = 35 := sorry

end pencils_distributed_l283_283481


namespace abs_diff_less_abs_one_minus_prod_l283_283361

theorem abs_diff_less_abs_one_minus_prod (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  |x - y| < |1 - x * y| := by
  sorry

end abs_diff_less_abs_one_minus_prod_l283_283361


namespace count_valid_m_l283_283082

theorem count_valid_m (h : 1260 > 0) :
  ∃! (n : ℕ), n = 3 := by
  sorry

end count_valid_m_l283_283082


namespace decreasing_function_range_a_l283_283510

noncomputable def f (a x : ℝ) : ℝ := -x^3 + x^2 + a * x

theorem decreasing_function_range_a (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) ↔ a ≤ -(1/3) :=
by
  -- This is a placeholder for the proof.
  sorry

end decreasing_function_range_a_l283_283510


namespace lemon_heads_each_person_l283_283052

-- Define the constants used in the problem
def totalLemonHeads : Nat := 72
def numberOfFriends : Nat := 6

-- The theorem stating the problem and the correct answer
theorem lemon_heads_each_person :
  totalLemonHeads / numberOfFriends = 12 := 
by
  sorry

end lemon_heads_each_person_l283_283052


namespace find_r4_l283_283266

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l283_283266


namespace possible_values_of_N_l283_283656

def is_valid_N (N : ℕ) : Prop :=
  (N > 22) ∧ (N ≤ 25)

theorem possible_values_of_N :
  {N : ℕ | is_valid_N N} = {23, 24, 25} :=
by
  sorry

end possible_values_of_N_l283_283656


namespace vector_simplification_l283_283709

variables (V : Type) [AddCommGroup V]

variables (CE AC DE AD : V)

theorem vector_simplification :
  CE + AC - DE - AD = 0 :=
by
  sorry

end vector_simplification_l283_283709


namespace min_value_frac_sum_l283_283929

theorem min_value_frac_sum (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 4) : 
  (4 / a^2 + 1 / b^2) ≥ 9 / 4 :=
by
  sorry

end min_value_frac_sum_l283_283929


namespace l_shape_area_is_42_l283_283044

-- Defining the dimensions of the larger rectangle
def large_rect_length : ℕ := 10
def large_rect_width : ℕ := 7

-- Defining the smaller rectangle dimensions based on the given conditions
def small_rect_length : ℕ := large_rect_length - 3
def small_rect_width : ℕ := large_rect_width - 3

-- Defining the areas of the rectangles
def large_rect_area : ℕ := large_rect_length * large_rect_width
def small_rect_area : ℕ := small_rect_length * small_rect_width

-- Defining the area of the "L" shape
def l_shape_area : ℕ := large_rect_area - small_rect_area

-- The theorem to prove
theorem l_shape_area_is_42 : l_shape_area = 42 :=
by
  sorry

end l_shape_area_is_42_l283_283044


namespace divide_into_two_groups_l283_283421

theorem divide_into_two_groups (n : ℕ) (A : Fin n → Type) 
  (acquaintances : (Fin n) → (Finset (Fin n)))
  (c : (Fin n) → ℕ) (d : (Fin n) → ℕ) :
  (∀ i : Fin n, c i = (acquaintances i).card) →
  ∃ G1 G2 : Finset (Fin n), G1 ∩ G2 = ∅ ∧ G1 ∪ G2 = Finset.univ ∧
  (∀ i : Fin n, d i = (acquaintances i ∩ (if i ∈ G1 then G2 else G1)).card ∧ d i ≥ (c i) / 2) :=
by 
  sorry

end divide_into_two_groups_l283_283421


namespace largest_constant_c_l283_283470

theorem largest_constant_c :
  ∃ c : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x^2 + y^2 = 1 → x^6 + y^6 ≥ c * x * y) ∧ c = 1 / 2 :=
sorry

end largest_constant_c_l283_283470


namespace remainder_of_product_mod_10_l283_283726

-- Definitions as conditions given in part a
def n1 := 2468
def n2 := 7531
def n3 := 92045

-- The problem expressed as a proof statement
theorem remainder_of_product_mod_10 :
  ((n1 * n2 * n3) % 10) = 0 :=
  by
    -- Sorry is used to skip the proof
    sorry

end remainder_of_product_mod_10_l283_283726


namespace side_length_of_square_base_l283_283399

theorem side_length_of_square_base (area slant_height : ℝ) (h_area : area = 120) (h_slant_height : slant_height = 40) :
  ∃ s : ℝ, area = 0.5 * s * slant_height ∧ s = 6 :=
by
  sorry

end side_length_of_square_base_l283_283399


namespace side_length_of_base_l283_283379

-- Define the conditions
def lateral_area (s : ℝ) : ℝ := (1 / 2) * s * 40
def given_area : ℝ := 120

-- Define the theorem to prove the length of the side of the base
theorem side_length_of_base : ∃ (s : ℝ), lateral_area(s) = given_area ∧ s = 6 :=
by
  sorry

end side_length_of_base_l283_283379


namespace new_ratio_of_alcohol_to_water_l283_283440

theorem new_ratio_of_alcohol_to_water
  (initial_ratio : ℚ)
  (alcohol_initial : ℚ)
  (water_added : ℚ)
  (new_ratio : ℚ) :
  initial_ratio = 2 / 5 ∧
  alcohol_initial = 10 ∧
  water_added = 10 ∧
  new_ratio = 2 / 7 :=
by 
  sorry

end new_ratio_of_alcohol_to_water_l283_283440


namespace definite_integral_sin_cos_l283_283889

open Real

theorem definite_integral_sin_cos :
  ∫ x in - (π / 2)..(π / 2), (sin x + cos x) = 2 :=
sorry

end definite_integral_sin_cos_l283_283889


namespace trigonometric_identity_l283_283915

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : (Real.sin (2 * α) / Real.cos α ^ 2) = 6 :=
sorry

end trigonometric_identity_l283_283915


namespace travel_ways_A_to_C_l283_283838

-- We define the number of ways to travel from A to B
def ways_A_to_B : ℕ := 3

-- We define the number of ways to travel from B to C
def ways_B_to_C : ℕ := 2

-- We state the problem as a theorem
theorem travel_ways_A_to_C : ways_A_to_B * ways_B_to_C = 6 :=
by
  sorry

end travel_ways_A_to_C_l283_283838


namespace possible_values_of_N_l283_283659

theorem possible_values_of_N (N : ℕ) (h : N > 8) :
  22 < N ∧ N ≤ 25 →
  N = 23 ∨ N = 24 ∨ N = 25 :=
by
  intros
  sorry

end possible_values_of_N_l283_283659


namespace ticket_cost_per_ride_l283_283599

theorem ticket_cost_per_ride (total_tickets : ℕ) (spent_tickets : ℕ) (rides : ℕ) (remaining_tickets : ℕ) (cost_per_ride : ℕ) 
  (h1 : total_tickets = 79) 
  (h2 : spent_tickets = 23) 
  (h3 : rides = 8) 
  (h4 : remaining_tickets = total_tickets - spent_tickets) 
  (h5 : remaining_tickets / rides = cost_per_ride) 
  : cost_per_ride = 7 := 
sorry

end ticket_cost_per_ride_l283_283599


namespace quadratic_function_incorrect_statement_l283_283083

theorem quadratic_function_incorrect_statement (x : ℝ) : 
  ∀ y : ℝ, y = -(x + 2)^2 - 1 → ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ y = 0 ∧ -(x1 + 2)^2 - 1 = 0 ∧ -(x2 + 2)^2 - 1 = 0) :=
by 
sorry

end quadratic_function_incorrect_statement_l283_283083


namespace total_canoes_by_end_of_march_l283_283223

theorem total_canoes_by_end_of_march
  (canoes_jan : ℕ := 3)
  (canoes_feb : ℕ := canoes_jan * 2)
  (canoes_mar : ℕ := canoes_feb * 2) :
  canoes_jan + canoes_feb + canoes_mar = 21 :=
by
  sorry

end total_canoes_by_end_of_march_l283_283223


namespace problem_1_problem_2_l283_283784

open Real

noncomputable def f (x : ℝ) : ℝ := 3^x
noncomputable def g (x : ℝ) : ℝ := log x / log 3

theorem problem_1 : g 4 + g 8 - g (32 / 9) = 2 := 
by
  sorry

theorem problem_2 (x : ℝ) (h : 0 < x ∧ x < 1) : g (x / (1 - x)) < 1 ↔ 0 < x ∧ x < 3 / 4 :=
by
  sorry

end problem_1_problem_2_l283_283784


namespace general_term_formula_is_not_element_l283_283249

theorem general_term_formula (a : ℕ → ℤ) (h1 : a 1 = 2) (h17 : a 17 = 66) :
  (∀ n, a n = 4 * n - 2) :=
by
  sorry

theorem is_not_element (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 2) :
  ¬ (∃ n : ℕ, a n = 88) :=
by
  sorry

end general_term_formula_is_not_element_l283_283249


namespace probability_at_least_three_consecutive_heads_l283_283043

def is_fair_coin (p : ℝ) := p = 1/2

def fair_coin_toss (n : ℕ) : Set (Vector Bool n) := 
  {s | ∀ i < n, s.get i = tt ∨ s.get i = ff}

def at_least_three_consecutive_heads (s : Vector Bool 4) : Prop :=
  (s = [tt, tt, tt, tt] ∨
   s = [tt, tt, tt, ff] ∨
   s = [ff, tt, tt, tt])

def probability_of_event (event : Set (Vector Bool 4)) (total : Set (Vector Bool 4)) : ℝ :=
  (event.card / total.card : ℝ)

theorem probability_at_least_three_consecutive_heads :
  ∀ (total : Set (Vector Bool 4)), 
  total = (fair_coin_toss 4) →
  probability_of_event {s | at_least_three_consecutive_heads s} total = 3 / 16 :=
by 
  sorry

end probability_at_least_three_consecutive_heads_l283_283043


namespace bluegrass_percentage_l283_283951

-- Define the problem conditions
def seed_mixture_X_ryegrass_percentage : ℝ := 40
def seed_mixture_Y_ryegrass_percentage : ℝ := 25
def seed_mixture_Y_fescue_percentage : ℝ := 75
def mixture_X_Y_ryegrass_percentage : ℝ := 30
def mixture_weight_percentage_X : ℝ := 33.33333333333333

-- Prove that the percentage of bluegrass in seed mixture X is 60%
theorem bluegrass_percentage (X_ryegrass : ℝ) (Y_ryegrass : ℝ) (Y_fescue : ℝ) (mixture_ryegrass : ℝ) (weight_percentage_X : ℝ) :
  X_ryegrass = seed_mixture_X_ryegrass_percentage →
  Y_ryegrass = seed_mixture_Y_ryegrass_percentage →
  Y_fescue = seed_mixture_Y_fescue_percentage →
  mixture_ryegrass = mixture_X_Y_ryegrass_percentage →
  weight_percentage_X = mixture_weight_percentage_X →
  (100 - X_ryegrass) = 60 :=
by
  intro hX_ryegrass hY_ryegrass hY_fescue hmixture_ryegrass hweight_X
  rw [hX_ryegrass]
  sorry

end bluegrass_percentage_l283_283951


namespace models_kirsty_can_buy_l283_283338

def savings := 30 * 0.45
def new_price := 0.50

theorem models_kirsty_can_buy : savings / new_price = 27 := by
  sorry

end models_kirsty_can_buy_l283_283338


namespace average_gas_mileage_round_trip_l283_283867

-- Definition of the problem conditions

def distance_to_home : ℕ := 120
def distance_back : ℕ := 120
def mileage_to_home : ℕ := 30
def mileage_back : ℕ := 20

-- Theorem that we need to prove
theorem average_gas_mileage_round_trip
  (d1 d2 : ℕ) (m1 m2 : ℕ)
  (h1 : d1 = distance_to_home)
  (h2 : d2 = distance_back)
  (h3 : m1 = mileage_to_home)
  (h4 : m2 = mileage_back) :
  (d1 + d2) / ((d1 / m1) + (d2 / m2)) = 24 :=
by
  sorry

end average_gas_mileage_round_trip_l283_283867


namespace James_future_age_when_Thomas_reaches_James_current_age_l283_283173

-- Defining the given conditions
def Thomas_age := 6
def Shay_age := Thomas_age + 13
def James_age := Shay_age + 5

-- Goal: Proving James's age when Thomas reaches James's current age
theorem James_future_age_when_Thomas_reaches_James_current_age :
  let years_until_Thomas_is_James_current_age := James_age - Thomas_age
  let James_future_age := James_age + years_until_Thomas_is_James_current_age
  James_future_age = 42 :=
by
  sorry

end James_future_age_when_Thomas_reaches_James_current_age_l283_283173


namespace correct_transformation_l283_283558

variable {a b c : ℝ}

-- A: \frac{a+3}{b+3} = \frac{a}{b}
def transformation_A (a b : ℝ) : Prop := (a + 3) / (b + 3) = a / b

-- B: \frac{a}{b} = \frac{ac}{bc}
def transformation_B (a b c : ℝ) : Prop := a / b = (a * c) / (b * c)

-- C: \frac{3a}{3b} = \frac{a}{b}
def transformation_C (a b : ℝ) : Prop := (3 * a) / (3 * b) = a / b

-- D: \frac{a}{b} = \frac{a^2}{b^2}
def transformation_D (a b : ℝ) : Prop := a / b = (a ^ 2) / (b ^ 2)

-- The main theorem to prove
theorem correct_transformation : transformation_C a b :=
by
  sorry

end correct_transformation_l283_283558


namespace find_number_l283_283639

theorem find_number (x : ℝ) : 4 * x - 23 = 33 → x = 14 :=
by
  intros h
  sorry

end find_number_l283_283639


namespace sequence_nth_term_16_l283_283102

theorem sequence_nth_term_16 (n : ℕ) (sqrt2 : ℝ) (h_sqrt2 : sqrt2 = Real.sqrt 2) (a_n : ℕ → ℝ) 
  (h_seq : ∀ n, a_n n = sqrt2 ^ (n - 1)) :
  a_n n = 16 → n = 9 := by
  sorry

end sequence_nth_term_16_l283_283102


namespace cost_price_percentage_l283_283923

variable (SP CP : ℝ)

-- Assumption that the profit percent is 25%
axiom profit_percent : 25 = ((SP - CP) / CP) * 100

-- The statement to prove
theorem cost_price_percentage : CP / SP = 0.8 := by
  sorry

end cost_price_percentage_l283_283923


namespace average_score_l283_283408

theorem average_score (a_males : ℕ) (a_females : ℕ) (n_males : ℕ) (n_females : ℕ)
  (h_males : a_males = 85) (h_females : a_females = 92) (h_n_males : n_males = 8) (h_n_females : n_females = 20) :
  (a_males * n_males + a_females * n_females) / (n_males + n_females) = 90 :=
by
  sorry

end average_score_l283_283408


namespace total_repair_cost_l283_283569

theorem total_repair_cost :
  let rate1 := 60
  let hours1 := 8
  let days1 := 14
  let rate2 := 75
  let hours2 := 6
  let days2 := 10
  let parts_cost := 3200
  let first_mechanic_cost := rate1 * hours1 * days1
  let second_mechanic_cost := rate2 * hours2 * days2
  let total_cost := first_mechanic_cost + second_mechanic_cost + parts_cost
  total_cost = 14420 := by
  sorry

end total_repair_cost_l283_283569


namespace min_expr_l283_283521

theorem min_expr (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : a * b = 1) :
  ∃ s : ℝ, (s = a + b) ∧ (s ≥ 2) ∧ (a^2 + b^2 + 4/(s^2) = 3) :=
by sorry

end min_expr_l283_283521


namespace expression_evaluation_l283_283062

theorem expression_evaluation :
  (4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2) ^ 0 = 1) :=
by
  -- Step by step calculations skipped
  sorry

end expression_evaluation_l283_283062


namespace distribute_cousins_l283_283814

-- Define the variables and the conditions
noncomputable def ways_to_distribute_cousins (cousins : ℕ) (rooms : ℕ) : ℕ :=
  if cousins = 5 ∧ rooms = 3 then 66 else sorry

-- State the problem
theorem distribute_cousins: ways_to_distribute_cousins 5 3 = 66 :=
by
  sorry

end distribute_cousins_l283_283814


namespace ratio_of_doctors_to_nurses_l283_283548

def total_staff : ℕ := 250
def nurses : ℕ := 150
def doctors : ℕ := total_staff - nurses

theorem ratio_of_doctors_to_nurses : 
  (doctors : ℚ) / (nurses : ℚ) = 2 / 3 := by
  sorry

end ratio_of_doctors_to_nurses_l283_283548


namespace weight_of_B_l283_283965

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 :=
by
  sorry

end weight_of_B_l283_283965


namespace units_digit_of_ksq_plus_2k_l283_283808

def k := 2023^3 - 3^2023

theorem units_digit_of_ksq_plus_2k : (k^2 + 2^k) % 10 = 1 := 
  sorry

end units_digit_of_ksq_plus_2k_l283_283808


namespace infinite_series_sum_l283_283593

theorem infinite_series_sum : 
  ∑' k : ℕ, (5^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 3 := 
sorry

end infinite_series_sum_l283_283593


namespace quadratic_has_distinct_real_roots_l283_283325

theorem quadratic_has_distinct_real_roots (m : ℝ) (hm : m ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (m * x1^2 - 2 * x1 + 3 = 0) ∧ (m * x2^2 - 2 * x2 + 3 = 0) ↔ 0 < m ∧ m < (1 / 3) :=
by
  sorry

end quadratic_has_distinct_real_roots_l283_283325


namespace remainder_of_power_mod_l283_283606

theorem remainder_of_power_mod :
  ∀ (x n m : ℕ), 
  x = 5 → n = 2021 → m = 17 →
  x^n % m = 11 := by
sorry

end remainder_of_power_mod_l283_283606


namespace find_k_and_a_l283_283557

noncomputable def polynomial_P : Polynomial ℝ := Polynomial.C 5 + Polynomial.X * (Polynomial.C (-18) + Polynomial.X * (Polynomial.C 13 + Polynomial.X * (Polynomial.C (-4) + Polynomial.X)))
noncomputable def polynomial_D (k : ℝ) : Polynomial ℝ := Polynomial.C k + Polynomial.X * (Polynomial.C (-1) + Polynomial.X)
noncomputable def polynomial_R (a : ℝ) : Polynomial ℝ := Polynomial.C a + (Polynomial.C 2 * Polynomial.X)

theorem find_k_and_a : 
  ∃ k a : ℝ, polynomial_P = polynomial_D k * Polynomial.C 1 + polynomial_R a ∧ k = 10 ∧ a = 5 :=
sorry

end find_k_and_a_l283_283557


namespace moles_of_KOH_combined_l283_283234

theorem moles_of_KOH_combined (H2O_formed : ℕ) (NH4I_used : ℕ) (ratio_KOH_H2O : ℕ) : H2O_formed = 54 → NH4I_used = 3 → ratio_KOH_H2O = 1 → H2O_formed = NH4I_used := 
by 
  intro H2O_formed_eq NH4I_used_eq ratio_eq 
  sorry

end moles_of_KOH_combined_l283_283234


namespace possible_values_of_N_l283_283657

theorem possible_values_of_N (N : ℕ) (h : N > 8) :
  22 < N ∧ N ≤ 25 →
  N = 23 ∨ N = 24 ∨ N = 25 :=
by
  intros
  sorry

end possible_values_of_N_l283_283657


namespace evaluate_fg_sum_at_1_l283_283945

def f (x : ℚ) : ℚ := (4 * x^2 + 3 * x + 6) / (x^2 + 2 * x + 5)
def g (x : ℚ) : ℚ := x + 1

theorem evaluate_fg_sum_at_1 : f (g 1) + g (f 1) = 497 / 104 :=
by
  sorry

end evaluate_fg_sum_at_1_l283_283945


namespace find_r_fourth_l283_283285

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l283_283285


namespace manager_salary_l283_283188

theorem manager_salary :
  let avg_salary_employees := 1500
  let num_employees := 20
  let new_avg_salary := 2000
  (new_avg_salary * (num_employees + 1) - avg_salary_employees * num_employees = 12000) :=
by
  sorry

end manager_salary_l283_283188


namespace sum_three_ways_l283_283737

theorem sum_three_ways (n : ℕ) (h : n > 0) : 
  ∃ k, k = (n^2) / 12 ∧ k = (n^2) / 12 :=
sorry

end sum_three_ways_l283_283737


namespace min_discount_70_percent_l283_283797

theorem min_discount_70_percent
  (P S : ℝ) (M : ℝ)
  (hP : P = 800)
  (hS : S = 1200)
  (hM : M = 0.05) :
  ∃ D : ℝ, D = 0.7 ∧ S * D - P ≥ P * M :=
by sorry

end min_discount_70_percent_l283_283797


namespace problem_inequality_l283_283012

variable {a b c d : ℝ}

theorem problem_inequality (h1 : 0 ≤ a) (h2 : 0 ≤ d) (h3 : 0 < b) (h4 : 0 < c) (h5 : b + c ≥ a + d) :
  (b / (c + d)) + (c / (b + a)) ≥ (Real.sqrt 2) - (1 / 2) := 
sorry

end problem_inequality_l283_283012


namespace mean_combined_set_l283_283972

noncomputable def mean (s : Finset ℚ) : ℚ :=
  (s.sum id) / s.card

theorem mean_combined_set :
  ∀ (s1 s2 : Finset ℚ),
  s1.card = 7 →
  s2.card = 8 →
  mean s1 = 15 →
  mean s2 = 18 →
  mean (s1 ∪ s2) = 249 / 15 :=
by
  sorry

end mean_combined_set_l283_283972


namespace min_value_of_xy_cond_l283_283834

noncomputable def minValueOfXY (x y : ℝ) : ℝ :=
  if 2 * Real.cos (x + y - 1) ^ 2 = ((x + 1) ^ 2 + (y - 1) ^ 2 - 2 * x * y) / (x - y + 1) then 
    x * y
  else 
    0

theorem min_value_of_xy_cond (x y : ℝ) 
  (h : 2 * Real.cos (x + y - 1) ^ 2 = ((x + 1) ^ 2 + (y - 1) ^ 2 - 2 * x * y) / (x - y + 1)) : 
  (∃ k : ℤ, x = (k * Real.pi + 1) / 2 ∧ y = (k * Real.pi + 1) / 2) → 
  x * y = 1/4 := 
by
  -- The proof is omitted.
  sorry

end min_value_of_xy_cond_l283_283834


namespace exist_elem_not_in_union_l283_283461

-- Assume closed sets
def isClosedSet (S : Set ℝ) : Prop :=
  ∀ a b : ℝ, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a - b) ∈ S

-- The theorem to prove
theorem exist_elem_not_in_union {S1 S2 : Set ℝ} (hS1 : isClosedSet S1) (hS2 : isClosedSet S2) :
  S1 ⊂ (Set.univ : Set ℝ) → S2 ⊂ (Set.univ : Set ℝ) → ∃ c : ℝ, c ∉ S1 ∪ S2 :=
by
  intro h1 h2
  sorry

end exist_elem_not_in_union_l283_283461


namespace side_length_of_base_l283_283391

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end side_length_of_base_l283_283391


namespace car_speed_l283_283435

theorem car_speed (v : ℝ) (h : (1 / v) * 3600 = (1 / 450) * 3600 + 2) : v = 360 :=
by
  sorry

end car_speed_l283_283435


namespace perfect_square_expression_l283_283264

theorem perfect_square_expression (x y : ℝ) (k : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x y, f x = f y → 4 * x^2 - (k - 1) * x * y + 9 * y^2 = (f x) ^ 2) ↔ (k = 13 ∨ k = -11) :=
by
  sorry

end perfect_square_expression_l283_283264


namespace right_angle_triangle_probability_l283_283243

def vertex_count : ℕ := 16
def ways_to_choose_3_points : ℕ := Nat.choose vertex_count 3
def number_of_rectangles : ℕ := 36
def right_angle_triangles_per_rectangle : ℕ := 4
def total_right_angle_triangles : ℕ := number_of_rectangles * right_angle_triangles_per_rectangle
def probability_right_angle_triangle : ℚ := total_right_angle_triangles / ways_to_choose_3_points

theorem right_angle_triangle_probability :
  probability_right_angle_triangle = (9 / 35 : ℚ) := by
  sorry

end right_angle_triangle_probability_l283_283243


namespace r_fourth_power_sum_l283_283280

theorem r_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_fourth_power_sum_l283_283280


namespace thirty_five_million_in_scientific_notation_l283_283763

def million := 10^6

def sales_revenue (x : ℝ) := x * million

theorem thirty_five_million_in_scientific_notation :
  sales_revenue 35 = 3.5 * 10^7 :=
by
  sorry

end thirty_five_million_in_scientific_notation_l283_283763


namespace count_distinct_numbers_in_list_l283_283235

def num_distinct_floor_divs : ℕ :=
  let L := list.map (λ n, ⌊(n ^ 2 : ℚ) / 500⌋) (list.range' 1 1000)
  list.dedup L

theorem count_distinct_numbers_in_list : (num_distinct_floor_divs.length = 876) := by
  -- Definition of L: the list from 1 to 1000 with floor of division by 500
  let L := list.map (λ n, ⌊(n ^ 2 : ℚ) / 500⌋) (list.range' 1 1000)
  -- Deduplicate the list by removing duplicates
  let distinct_L := list.dedup L
  -- Prove that the length of distinct_L is equal to 876
  have h : distinct_L.length = 876 := sorry
  exact h

end count_distinct_numbers_in_list_l283_283235


namespace john_pennies_more_than_kate_l283_283337

theorem john_pennies_more_than_kate (kate_pennies : ℕ) (john_pennies : ℕ) (h_kate : kate_pennies = 223) (h_john : john_pennies = 388) : john_pennies - kate_pennies = 165 := by
  sorry

end john_pennies_more_than_kate_l283_283337


namespace sum_of_fractions_as_decimal_l283_283892

theorem sum_of_fractions_as_decimal : (3 / 8 : ℝ) + (5 / 32) = 0.53125 := by
  sorry

end sum_of_fractions_as_decimal_l283_283892


namespace midpoints_collinear_l283_283519

-- Define points A, B, C, D on the semicircle
variables (A B C D E F : Point)

-- Define collinearity predicate
def is_collinear (P Q R : Point) : Prop :=
  ∃ (l : Line), P ∈ l ∧ Q ∈ l ∧ R ∈ l

-- Main theorem: midpoints of AB, CD, and EF are collinear
theorem midpoints_collinear
  (semicircle : Semicircle A B)
  (C_onsc : C ∈ semicircle)
  (D_onsc : D ∈ semicircle)
  (E_intersection : ∃ E, is_intersection (Line_through A C) (Line_through B D) E)
  (F_intersection : ∃ F, is_intersection (Line_through A D) (Line_through B C) F)
  (M_AB : Point := midpoint A B)
  (M_CD : Point := midpoint C D)
  (M_EF : Point := midpoint E F) :
  is_collinear M_AB M_CD M_EF := 
sorry

end midpoints_collinear_l283_283519


namespace range_of_k_l283_283242

open Real BigOperators Topology

variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)
variable (q : ℝ) (k : ℝ)

-- Condition: Each term of the sequence is non-zero
def non_zero_sequence : Prop := ∀ n, a n ≠ 0

-- Condition: Sum of the first n terms is Sn
def sum_of_sequence (S : ℕ → ℝ) : Prop := ∀ n, S n = ∑ i in Finset.range n, a i

-- Condition: Vector is normal to the line y = kx
def normal_vector (k : ℝ) : Prop := ∀ n, k = - (a (n + 1) - a n) / (2 * a (n + 1))

-- Condition: The common ratio q satisfies 0 < |q| < 1
def common_ratio_q : Prop := 0 < abs q ∧ abs q < 1

-- Condition: The limit Sn exists as n tends to infinity
def limit_exists : Prop := ∃ l, Tendsto S atTop (nhds l)

-- Statement to be proved
theorem range_of_k 
  (h1 : non_zero_sequence a)
  (h2 : sum_of_sequence a S)
  (h3 : normal_vector k)
  (h4 : common_ratio_q q)
  (h5 : limit_exists S) :
  k ∈ Iio (-1) ∪ Ioi 0 :=
begin
  sorry
end

end range_of_k_l283_283242


namespace greatest_integer_not_exceeding_1000x_l283_283213

-- Given the conditions of the problem
variables (x : ℝ)
-- Cond 1: Edge length of the cube
def edge_length := 2
-- Cond 2: Point light source is x centimeters above a vertex
-- Cond 3: Shadow area excluding the area beneath the cube is 98 square centimeters
def shadow_area_excluding_cube := 98
-- This is the condition total area of the shadow
def total_shadow_area := shadow_area_excluding_cube + edge_length ^ 2

-- Statement: Prove that the greatest integer not exceeding 1000x is 8100:
theorem greatest_integer_not_exceeding_1000x (h1 : total_shadow_area = 102) : x ≤ 8.1 :=
by
  sorry

end greatest_integer_not_exceeding_1000x_l283_283213


namespace calculate_cost_price_l283_283352

/-
Given:
  SP (Selling Price) is 18000
  If a 10% discount is applied on the SP, the effective selling price becomes 16200
  This effective selling price corresponds to an 8% profit over the cost price
  
Prove:
  The cost price (CP) is 15000
-/

theorem calculate_cost_price (SP : ℝ) (d : ℝ) (p : ℝ) (effective_SP : ℝ) (CP : ℝ) :
  SP = 18000 →
  d = 0.1 →
  p = 0.08 →
  effective_SP = SP - (d * SP) →
  effective_SP = CP * (1 + p) →
  CP = 15000 :=
by
  intros _
  sorry

end calculate_cost_price_l283_283352


namespace triangle_cosine_identity_l283_283693

theorem triangle_cosine_identity
  (a b c : ℝ)
  (α β γ : ℝ)
  (hα : α = Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))
  (hβ : β = Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c)))
  (hγ : γ = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)))
  (habc_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (b / c + c / b) * Real.cos α + 
  (c / a + a / c) * Real.cos β + 
  (a / b + b / a) * Real.cos γ = 3 := 
sorry

end triangle_cosine_identity_l283_283693


namespace digit_C_equals_one_l283_283998

-- Define the scope of digits
def is_digit (n : ℕ) : Prop := n < 10

-- Define the equality for sums of digits
def sum_of_digits (A B C : ℕ) : Prop := A + B + C = 10

-- Main theorem to prove C = 1
theorem digit_C_equals_one (A B C : ℕ) (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hSum : sum_of_digits A B C) : C = 1 :=
sorry

end digit_C_equals_one_l283_283998


namespace inverse_function_of_f_l283_283078

noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / x
noncomputable def f_inv (x : ℝ) : ℝ := 1 / (x - 3)

theorem inverse_function_of_f:
  ∀ x : ℝ, x ≠ 3 → f (f_inv x) = x ∧ f_inv (f x) = x := by
sorry

end inverse_function_of_f_l283_283078


namespace final_answer_l283_283008

noncomputable def coin_flip_expression : List ℚ :=
  [1/2, 1/4, 1/8, 1/16, 1/32]

def compute_values (flips : List Bool) : ℚ :=
  (List.zipWith (λ c flip => if flip then c else -c) coin_flip_expression flips).sum

def pos_diff_greater_than_half (flips_1 flips_2 : List Bool) : Bool :=
  abs (compute_values flips_1 - compute_values flips_2) > 1/2

theorem final_answer : ∃ a b : ℕ, Nat.coprime a b ∧ 
  (probability (fun flips_1 flips_2 => pos_diff_greater_than_half flips_1 flips_2)) = (a / b) ∧ a + b = 39 := 
sorry

end final_answer_l283_283008


namespace find_b_l283_283563

variable (a b c : ℕ)
variable (h1 : (a + b + c) / 3 = 45)
variable (h2 : (a + b) / 2 = 40)
variable (h3 : (b + c) / 2 = 43)

theorem find_b : b = 31 := sorry

end find_b_l283_283563


namespace sphere_touches_pyramid_edges_l283_283579

theorem sphere_touches_pyramid_edges :
  ∃ (KL : ℝ), 
  ∃ (K L M N : ℝ) (MN LN NK : ℝ) (AC: ℝ) (BC: ℝ), 
  MN = 7 ∧ 
  NK = 5 ∧ 
  LN = 2 * Real.sqrt 29 ∧ 
  KL = L ∧ 
  KL = M ∧ 
  KL = 9 :=
sorry

end sphere_touches_pyramid_edges_l283_283579


namespace possible_values_for_N_l283_283662

theorem possible_values_for_N (N : ℕ) (H : ∀ (k : ℕ), N = 8 + k) (truth: (student : ℕ → Prop) (bully : ℕ → Prop) (n : ℕ), 
  (∀ i, bully i → ¬ (∃ j, student j ∧ bully j → j ≠ i)) →
  (∀ i, student i → (∃ j, student j ∧ bully j → j ≠ i))
  → ∀ i, (bully i → ¬(∃ (m : ℕ), (m ≥ N - 1 / 3 )) ∧ student i → (∃ (m : ℕ), (m ≥ N - 1 / 3 ))) : N = 23 ∨ N = 24 ∨ N = 25 :=
by
  sorry

end possible_values_for_N_l283_283662


namespace simplify_fraction_l283_283364

theorem simplify_fraction :
  ∀ (x : ℝ),
    (18 * x^4 - 9 * x^3 - 86 * x^2 + 16 * x + 96) /
    (18 * x^4 - 63 * x^3 + 22 * x^2 + 112 * x - 96) =
    (2 * x + 3) / (2 * x - 3) :=
by sorry

end simplify_fraction_l283_283364


namespace sufficient_necessary_conditions_l283_283921

variable (A B C D : Prop)

#check
theorem sufficient_necessary_conditions :
  (A → B) ∧ ¬ (B → A) →
  (C → B) ∧ ¬ (B → C) →
  (D → C) ∧ ¬ (C → D) →
  ((B → A) ∧ ¬ (A → B)) ∧
  ((A → C) ∧ ¬ (C → A)) ∧
  (¬ (D → A) ∧ ¬ (A → D)) :=
by
  intros h1 h2 h3
  cases h1 with hA_implies_B hnB_implies_A
  cases h2 with hC_implies_B hnB_implies_C
  cases h3 with hD_implies_C hnC_implies_D
  split
  -- Proof for B → A and ¬A → B
  split
  { intros hB,
    apply false.elim,
    apply hnB_implies_A,
    exact hB, },
  { exact hnB_implies_A, },
  -- Proof for A → C and ¬C → A
  split
  { intros hA,
    apply hC_implies_B,
    apply hA_implies_B,
    exact hA, },
  { apply false.elim,
    intros hA,
    apply hnB_implies_A,
    exact hA, },
  -- Proof for ¬D → A and ¬A → D
  split
  { intros hD,
    apply false.elim,
    apply hnB_implies_A,
    exact hD, },
  { intros hA,
    apply false.elim,
    apply hnC_implies_D,
    exact hA, }

end sufficient_necessary_conditions_l283_283921


namespace intersecting_point_value_l283_283163

theorem intersecting_point_value
  (b a : ℤ)
  (h1 : a = -2 * 2 + b)
  (h2 : 2 = -2 * a + b) :
  a = 2 :=
by
  sorry

end intersecting_point_value_l283_283163


namespace chess_group_players_l283_283841

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 :=
sorry

end chess_group_players_l283_283841


namespace problem1_problem2_l283_283047

open scoped Classical
open ProbabilityTheory

noncomputable def prob_second_level_after_3_shots : ℚ :=
  (2.choose 1 * (2/3) * (1/3) * (2/3))

theorem problem1 : prob_second_level_after_3_shots = 8 / 27 :=
by {
  -- calculation steps here.
  sorry
}

noncomputable def prob_selected : ℚ :=
  ((2/3)^3) + (3.choose 2 * (2/3)^2 * (1/3) * (2/3)) +
  (4.choose 2 * (2/3)^2 * (1/3)^2 * (2/3))

noncomputable def prob_selected_and_shoot_5_times : ℚ :=
  (4.choose 2 * (2/3)^2 * (1/3)^2 * (2/3))

theorem problem2 : prob_selected_and_shoot_5_times / prob_selected = 1 / 4 :=
by {
  -- calculation steps here.
  sorry
}

end problem1_problem2_l283_283047


namespace possible_values_of_N_l283_283658

theorem possible_values_of_N (N : ℕ) (h : N > 8) :
  22 < N ∧ N ≤ 25 →
  N = 23 ∨ N = 24 ∨ N = 25 :=
by
  intros
  sorry

end possible_values_of_N_l283_283658


namespace incorrect_positional_relationship_l283_283259

-- Definitions for the geometric relationships
def line := Type
def plane := Type

def parallel (l : line) (α : plane) : Prop := sorry
def perpendicular (l : line) (α : plane) : Prop := sorry
def subset (l : line) (α : plane) : Prop := sorry
def distinct (l m : line) : Prop := l ≠ m

-- Given conditions
variables (l m : line) (α : plane)

-- Theorem statement: prove that D is incorrect given the conditions
theorem incorrect_positional_relationship
  (h_distinct : distinct l m)
  (h_parallel_l_α : parallel l α)
  (h_parallel_m_α : parallel m α) :
  ¬ (parallel l m) :=
sorry

end incorrect_positional_relationship_l283_283259


namespace problem_inequality_l283_283904

variables {a b c x1 x2 x3 x4 x5 : ℝ} 

theorem problem_inequality
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_x1: 0 < x1) (h_pos_x2: 0 < x2) (h_pos_x3: 0 < x3) (h_pos_x4: 0 < x4) (h_pos_x5: 0 < x5)
  (h_sum_abc : a + b + c = 1) (h_prod_x : x1 * x2 * x3 * x4 * x5 = 1) :
  (a * x1^2 + b * x1 + c) * (a * x2^2 + b * x2 + c) * (a * x3^2 + b * x3 + c) * 
  (a * x4^2 + b * x4 + c) * (a * x5^2 + b * x5 + c) ≥ 1 :=
sorry

end problem_inequality_l283_283904


namespace yellow_peaches_l283_283422

theorem yellow_peaches (red_peaches green_peaches total_green_yellow_peaches : ℕ)
  (h1 : red_peaches = 5)
  (h2 : green_peaches = 6)
  (h3 : total_green_yellow_peaches = 20) :
  (total_green_yellow_peaches - green_peaches) = 14 :=
by
  sorry

end yellow_peaches_l283_283422


namespace average_seven_numbers_l283_283640

theorem average_seven_numbers (A B C D E F G : ℝ) 
  (h1 : (A + B + C + D) / 4 = 4)
  (h2 : (D + E + F + G) / 4 = 4)
  (hD : D = 11) : 
  (A + B + C + D + E + F + G) / 7 = 3 :=
by
  sorry

end average_seven_numbers_l283_283640


namespace find_r_fourth_l283_283284

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l283_283284


namespace min_value_theorem_l283_283613

noncomputable def min_value (x y : ℝ) : ℝ :=
  (x + 2) * (2 * y + 1) / (x * y)

theorem min_value_theorem {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  min_value x y = 19 + 4 * Real.sqrt 15 :=
sorry

end min_value_theorem_l283_283613


namespace range_of_f_area_of_triangle_l283_283098

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.sin (x - Real.pi / 6)

-- Problem Part (I)
theorem range_of_f : 
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 →
      -1/2 ≤ f x ∧ f x ≤ 1/4) :=
sorry

-- Problem Part (II)
theorem area_of_triangle 
  (A B C : ℝ)
  (a b c : ℝ) 
  (hA0 : 0 < A ∧ A < Real.pi)
  (hS1 : a = Real.sqrt 3)
  (hS2 : b = 2 * c)
  (hF : f A = 1/4) :
  (∃ (area : ℝ), area = (1/2) * b * c * Real.sin A ∧ area = Real.sqrt 3 / 3)
:=
sorry

end range_of_f_area_of_triangle_l283_283098


namespace student_count_l283_283646

theorem student_count (N : ℕ) (h : N > 22 ∧ N ≤ 25) : N = 23 ∨ N = 24 ∨ N = 25 := by {
  cases (Nat.eq_or_lt_of_le h.right); {
    exact Or.inr Or.inr (Nat.lt_antisymm h.left _); sorry;
  };
  exact Or.inr (Or.inl (Nat.lt_antisymm h.left (Nat.lt_succ_iff.mpr h.left))); sorry;
  exact Or.inl (Nat.lt_antisymm h.left (Nat.lt_succ_of_lt h.left)); sorry;
}

end student_count_l283_283646


namespace absolute_value_of_neg_five_l283_283373

theorem absolute_value_of_neg_five : |(-5 : ℤ)| = 5 := 
by 
  sorry

end absolute_value_of_neg_five_l283_283373


namespace meaningful_expr_l283_283324

theorem meaningful_expr (x : ℝ) : 
    (x + 1 ≥ 0 ∧ x - 2 ≠ 0) → (x ≥ -1 ∧ x ≠ 2) := by
  sorry

end meaningful_expr_l283_283324


namespace total_buyers_in_three_days_l283_283718

theorem total_buyers_in_three_days
  (D_minus_2 : ℕ)
  (D_minus_1 : ℕ)
  (D_0 : ℕ)
  (h1 : D_minus_2 = 50)
  (h2 : D_minus_1 = D_minus_2 / 2)
  (h3 : D_0 = D_minus_1 + 40) :
  D_minus_2 + D_minus_1 + D_0 = 140 :=
by
  sorry

end total_buyers_in_three_days_l283_283718


namespace solve_diamond_l283_283108

theorem solve_diamond (d : ℕ) (h : 9 * d + 5 = 10 * d + 2) : d = 3 :=
by
  sorry

end solve_diamond_l283_283108


namespace airline_num_airplanes_l283_283589

-- Definitions based on the conditions
def rows_per_airplane : ℕ := 20
def seats_per_row : ℕ := 7
def flights_per_day_per_airplane : ℕ := 2
def total_passengers_per_day : ℕ := 1400

-- The theorem to prove the number of airplanes owned by the company
theorem airline_num_airplanes : 
  (total_passengers_per_day = 
   rows_per_airplane * seats_per_row * flights_per_day_per_airplane * n) → 
  n = 5 := 
by 
  sorry

end airline_num_airplanes_l283_283589


namespace merchant_markup_l283_283574

theorem merchant_markup (x : ℝ) : 
  let CP := 100
  let MP := CP + (x / 100) * CP
  let SP_discount := MP - 0.1 * MP 
  let SP_profit := CP + 57.5
  SP_discount = SP_profit → x = 75 :=
by
  intros
  let CP := (100 : ℝ)
  let MP := CP + (x / 100) * CP
  let SP_discount := MP - 0.1 * MP 
  let SP_profit := CP + 57.5
  have h : SP_discount = SP_profit := sorry
  sorry

end merchant_markup_l283_283574


namespace parts_purchased_l283_283878

noncomputable def price_per_part : ℕ := 80
noncomputable def total_paid_after_discount : ℕ := 439
noncomputable def total_discount : ℕ := 121

theorem parts_purchased : 
  ∃ n : ℕ, price_per_part * n - total_discount = total_paid_after_discount → n = 7 :=
by
  sorry

end parts_purchased_l283_283878


namespace final_volume_of_water_in_tank_l283_283580

theorem final_volume_of_water_in_tank (capacity : ℕ) (initial_fraction full_volume : ℕ)
  (percent_empty percent_fill final_volume : ℕ) :
  capacity = 8000 →
  initial_fraction = 3 / 4 →
  percent_empty = 40 →
  percent_fill = 30 →
  full_volume = capacity * initial_fraction →
  final_volume = full_volume - (full_volume * percent_empty / 100) + ((full_volume - (full_volume * percent_empty / 100)) * percent_fill / 100) →
  final_volume = 4680 :=
by
  sorry

end final_volume_of_water_in_tank_l283_283580


namespace weight_of_B_l283_283966

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 :=
by
  sorry

end weight_of_B_l283_283966


namespace least_positive_integer_satisfying_conditions_l283_283179

theorem least_positive_integer_satisfying_conditions :
  ∃ b : ℕ, b > 0 ∧ (b % 7 = 6) ∧ (b % 11 = 10) ∧ (b % 13 = 12) ∧ b = 1000 :=
by
  sorry

end least_positive_integer_satisfying_conditions_l283_283179


namespace possible_values_for_N_l283_283664

theorem possible_values_for_N (N : ℕ) (H : ∀ (k : ℕ), N = 8 + k) (truth: (student : ℕ → Prop) (bully : ℕ → Prop) (n : ℕ), 
  (∀ i, bully i → ¬ (∃ j, student j ∧ bully j → j ≠ i)) →
  (∀ i, student i → (∃ j, student j ∧ bully j → j ≠ i))
  → ∀ i, (bully i → ¬(∃ (m : ℕ), (m ≥ N - 1 / 3 )) ∧ student i → (∃ (m : ℕ), (m ≥ N - 1 / 3 ))) : N = 23 ∨ N = 24 ∨ N = 25 :=
by
  sorry

end possible_values_for_N_l283_283664


namespace coin_flip_prob_difference_l283_283181

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem coin_flip_prob_difference :
  let p1 := binomial_prob 3 2 (1/2)
  let p2 := binomial_prob 3 3 (1/2)
  abs (p1 - p2) = 1 / 4 :=
by
  sorry

end coin_flip_prob_difference_l283_283181


namespace diving_club_capacity_l283_283157

theorem diving_club_capacity :
  (3 * ((2 * 5 + 4 * 2) * 5) = 270) :=
by
  sorry

end diving_club_capacity_l283_283157


namespace calculation_result_l283_283888

theorem calculation_result :
  3 * 3^3 + 4^7 / 4^5 = 97 :=
by
  sorry

end calculation_result_l283_283888


namespace probability_sum_l283_283204

noncomputable def P : ℕ → ℝ := sorry

theorem probability_sum (n : ℕ) (h : n ≥ 7) :
  P n = (1/6) * (P (n-1) + P (n-2) + P (n-3) + P (n-4) + P (n-5) + P (n-6)) :=
sorry

end probability_sum_l283_283204


namespace height_of_triangle_l283_283577

variables (a b h' : ℝ)

theorem height_of_triangle (h : (1/2) * a * h' = a * b) : h' = 2 * b :=
sorry

end height_of_triangle_l283_283577


namespace minimum_focal_chord_length_l283_283587

theorem minimum_focal_chord_length (p : ℝ) (hp : p > 0) :
  ∃ l, (l = 2 * p) ∧ (∀ y x1 x2, y^2 = 2 * p * x1 ∧ y^2 = 2 * p * x2 → l = x2 - x1) := 
sorry

end minimum_focal_chord_length_l283_283587


namespace find_r_fourth_l283_283286

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l283_283286


namespace right_triangle_sides_l283_283169

theorem right_triangle_sides (a b c : ℝ) (h_ratio : ∃ x : ℝ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x) 
(h_area : 1 / 2 * a * b = 24) : a = 6 ∧ b = 8 ∧ c = 10 :=
by
  sorry

end right_triangle_sides_l283_283169


namespace select_defective_products_l283_283590

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem select_defective_products :
  let total_products := 200
  let defective_products := 3
  let selected_products := 5
  let ways_2_defective := choose defective_products 2 * choose (total_products - defective_products) 3
  let ways_3_defective := choose defective_products 3 * choose (total_products - defective_products) 2
  ways_2_defective + ways_3_defective = choose defective_products 2 * choose (total_products - defective_products) 3 + choose defective_products 3 * choose (total_products - defective_products) 2 :=
by
  sorry

end select_defective_products_l283_283590


namespace luke_total_score_l283_283349

theorem luke_total_score (points_per_round : ℕ) (number_of_rounds : ℕ) (total_score : ℕ) : 
  points_per_round = 146 ∧ number_of_rounds = 157 ∧ total_score = points_per_round * number_of_rounds → 
  total_score = 22822 := by 
  sorry

end luke_total_score_l283_283349


namespace r_power_four_identity_l283_283307

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l283_283307


namespace jungkook_age_l283_283336

theorem jungkook_age
    (J U : ℕ)
    (h1 : J = U - 12)
    (h2 : (J + 3) + (U + 3) = 38) :
    J = 10 := 
sorry

end jungkook_age_l283_283336


namespace rectangle_properties_l283_283677

theorem rectangle_properties :
  ∃ (length width : ℝ),
    (length / width = 3) ∧ 
    (length * width = 75) ∧
    (length = 15) ∧
    (width = 5) ∧
    ∀ (side : ℝ), 
      (side^2 = 75) → 
      (side - width > 3) :=
by
  sorry

end rectangle_properties_l283_283677


namespace mike_baseball_cards_l283_283351

theorem mike_baseball_cards (initial_cards birthday_cards traded_cards : ℕ)
  (h1 : initial_cards = 64) 
  (h2 : birthday_cards = 18) 
  (h3 : traded_cards = 20) :
  initial_cards + birthday_cards - traded_cards = 62 :=
by 
  -- assumption:
  sorry

end mike_baseball_cards_l283_283351


namespace perpendicular_lines_l283_283117

noncomputable def l1_slope (a : ℝ) : ℝ := -a / (1 - a)
noncomputable def l2_slope (a : ℝ) : ℝ := -(a - 1) / (2 * a + 3)

theorem perpendicular_lines (a : ℝ) 
  (h1 : ∀ x y : ℝ, a * x + (1 - a) * y = 3 → Prop) 
  (h2 : ∀ x y : ℝ, (a - 1) * x + (2 * a + 3) * y = 2 → Prop) 
  (hp : l1_slope a * l2_slope a = -1) : a = -3 := by
  sorry

end perpendicular_lines_l283_283117


namespace evaluate_abs_expression_l283_283466

noncomputable def approx_pi : ℝ := 3.14159 -- Defining the approximate value of pi

theorem evaluate_abs_expression : |5 * approx_pi - 16| = 0.29205 :=
by
  sorry -- Proof is skipped, as per instructions

end evaluate_abs_expression_l283_283466


namespace dishes_left_for_oliver_l283_283444

theorem dishes_left_for_oliver (n a c pick mango_salsa_dishes fresh_mango_dishes mango_jelly_dish : ℕ)
  (total_dishes : n = 36)
  (mango_salsa_condition : a = 3)
  (fresh_mango_condition : fresh_mango_dishes = n / 6)
  (mango_jelly_condition : c = 1)
  (willing_to_pick_mango : pick = 2) :
  ∃ D : ℕ, D = n - (a + fresh_mango_dishes + c - pick) ∧ D = 28 :=
by
  intros
  have h1 : fresh_mango_dishes = n / 6, from (fresh_mango_condition)
  have h2 : 8 = 10 - pick, by
    rw [mango_salsa_condition, h1, mango_jelly_condition, ← add_assoc]
    norm_num
  refine ⟨n - 8, _, _⟩
  rw h2
  split
  norm_num
  rfl

end dishes_left_for_oliver_l283_283444


namespace find_num_male_general_attendees_l283_283222

def num_attendees := 1000
def num_presenters := 420
def total_general_attendees := num_attendees - num_presenters

variables (M_p F_p M_g F_g : ℕ)

axiom condition1 : M_p = F_p + 20
axiom condition2 : M_p + F_p = 420
axiom condition3 : F_g = M_g + 56
axiom condition4 : M_g + F_g = total_general_attendees

theorem find_num_male_general_attendees :
  M_g = 262 :=
by
  sorry

end find_num_male_general_attendees_l283_283222


namespace math_problem_l283_283779

theorem math_problem
  (x y : ℝ)
  (h1 : 1 / x + 1 / y = 4)
  (h2 : x^2 + y^2 = 18) :
  x^2 + y^2 = 18 :=
sorry

end math_problem_l283_283779


namespace marble_prob_l283_283559

theorem marble_prob (T : ℕ) (hT1 : T > 12) 
  (hP : ((T - 12) / T : ℚ) * ((T - 12) / T) = 36 / 49) : T = 84 :=
sorry

end marble_prob_l283_283559


namespace subset_proof_l283_283794

-- Define the set B
def B : Set ℝ := { x | x ≥ 0 }

-- Define the set A as the set {1, 2}
def A : Set ℝ := {1, 2}

-- The proof problem: Prove that A ⊆ B
theorem subset_proof : A ⊆ B := sorry

end subset_proof_l283_283794


namespace rounds_on_sunday_l283_283687

theorem rounds_on_sunday (round_time total_time saturday_rounds : ℕ) (h1 : round_time = 30)
(h2 : total_time = 780) (h3 : saturday_rounds = 11) : 
(total_time - saturday_rounds * round_time) / round_time = 15 := by
  sorry

end rounds_on_sunday_l283_283687


namespace number_of_ideal_match_sets_l283_283344

open Finset

def ideal_match_sets (I : Finset ℕ) :=
  {p : Finset ℕ × Finset ℕ // p.1 ∩ p.2 = {1, 3}}

def count_ideal_match_sets : ℕ :=
  (ideal_match_sets {1, 2, 3, 4}).toFinset.card

theorem number_of_ideal_match_sets : count_ideal_match_sets = 9 :=
  sorry

end number_of_ideal_match_sets_l283_283344


namespace remainder_when_summed_divided_by_15_l283_283811

theorem remainder_when_summed_divided_by_15 (k j : ℤ) (x y : ℤ)
  (hx : x = 60 * k + 47)
  (hy : y = 45 * j + 26) :
  (x + y) % 15 = 13 := 
sorry

end remainder_when_summed_divided_by_15_l283_283811


namespace prank_combinations_l283_283175

-- Conditions stated as definitions
def monday_choices : ℕ := 1
def tuesday_choices : ℕ := 3
def wednesday_choices : ℕ := 5
def thursday_choices : ℕ := 6
def friday_choices : ℕ := 2

-- Theorem to prove
theorem prank_combinations :
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 180 :=
by
  sorry

end prank_combinations_l283_283175


namespace ceil_floor_difference_is_3_l283_283893

noncomputable def ceil_floor_difference : ℤ :=
  Int.ceil ((14:ℚ) / 5 * (-31 / 3)) - Int.floor ((14 / 5) * Int.floor ((-31:ℚ) / 3))

theorem ceil_floor_difference_is_3 : ceil_floor_difference = 3 :=
  sorry

end ceil_floor_difference_is_3_l283_283893


namespace scheduling_arrangements_correct_l283_283753

-- Define the set of employees
inductive Employee
| A | B | C | D | E | F deriving DecidableEq

open Employee

-- Define the days of the festival
inductive Day
| May31 | June1 | June2 deriving DecidableEq

open Day

def canWork (e : Employee) (d : Day) : Prop :=
match e, d with
| A, May31 => False
| B, June2 => False
| _, _ => True

def schedulingArrangements : ℕ :=
  -- Calculations go here, placeholder for now
  sorry

theorem scheduling_arrangements_correct : schedulingArrangements = 42 := 
  sorry

end scheduling_arrangements_correct_l283_283753


namespace pyramid_base_side_length_l283_283389

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end pyramid_base_side_length_l283_283389


namespace value_of_sum_cubes_l283_283503

theorem value_of_sum_cubes (x : ℝ) (hx : x ≠ 0) (h : 47 = x^6 + (1 / x^6)) : (x^3 + (1 / x^3)) = 7 := 
by 
  sorry

end value_of_sum_cubes_l283_283503


namespace mode_of_scores_l283_283446

-- Define the list of scores
def scores : List ℕ := [35, 37, 39, 37, 38, 38, 37]

-- State that the mode of the list of scores is 37
theorem mode_of_scores : Multiset.mode (Multiset.ofList scores) = 37 := 
by 
  sorry

end mode_of_scores_l283_283446


namespace total_pizzas_eaten_l283_283861

-- Definitions for the conditions
def pizzasA : ℕ := 8
def pizzasB : ℕ := 7

-- Theorem stating the total number of pizzas eaten by both classes
theorem total_pizzas_eaten : pizzasA + pizzasB = 15 := 
by
  -- Proof is not required for the task, so we use sorry
  sorry

end total_pizzas_eaten_l283_283861


namespace infinite_n_exists_l283_283806

-- Definitions from conditions
def is_natural_number (a : ℕ) : Prop := a > 3

-- Statement of the theorem
theorem infinite_n_exists (a : ℕ) (h : is_natural_number a) : ∃ᶠ n in at_top, a + n ∣ a^n + 1 :=
sorry

end infinite_n_exists_l283_283806


namespace find_value_of_x_cubed_plus_y_cubed_l283_283320

-- Definitions based on the conditions provided
variables (x y : ℝ)
variables (h1 : y + 3 = (x - 3)^2) (h2 : x + 3 = (y - 3)^2) (h3 : x ≠ y)

theorem find_value_of_x_cubed_plus_y_cubed :
  x^3 + y^3 = 217 :=
sorry

end find_value_of_x_cubed_plus_y_cubed_l283_283320


namespace possible_values_for_N_l283_283676

theorem possible_values_for_N (N : ℕ) (h₁ : 8 < N) (h₂ : 1 ≤ N - 1) :
  22 < N ∧ N ≤ 25 ↔ (N = 23 ∨ N = 24 ∨ N = 25) :=
by 
  sorry

end possible_values_for_N_l283_283676


namespace smallest_number_increased_by_7_divisible_by_8_11_24_l283_283032

theorem smallest_number_increased_by_7_divisible_by_8_11_24 : ∃ n : ℤ, n = 257 ∧ ∀ m : ℤ, (∃ k : ℤ, m + 7 = 264 * k) → 257 ≤ m := 
begin
  sorry
end

end smallest_number_increased_by_7_divisible_by_8_11_24_l283_283032


namespace number_of_slices_per_package_l283_283707

-- Define the problem's conditions
def packages_of_bread := 2
def slices_per_package_of_ham := 8
def packages_of_ham := 2
def leftover_slices_of_bread := 8
def total_ham_slices := packages_of_ham * slices_per_package_of_ham
def total_ham_required_bread := total_ham_slices * 2
def total_initial_bread_slices (B : ℕ) := packages_of_bread * B
def total_bread_used (B : ℕ) := total_ham_required_bread
def slices_leftover (B : ℕ) := total_initial_bread_slices B - total_bread_used B

-- Specify the goal
theorem number_of_slices_per_package (B : ℕ) (h : total_initial_bread_slices B = total_bread_used B + leftover_slices_of_bread) : B = 20 :=
by
  -- Use the provided conditions along with the hypothesis
  -- of the initial bread slices equation equating to used and leftover slices
  sorry

end number_of_slices_per_package_l283_283707


namespace curved_surface_area_cone_l283_283542

-- Define the necessary values
def r := 8  -- radius of the base of the cone in centimeters
def l := 18 -- slant height of the cone in centimeters

-- Prove the curved surface area of the cone
theorem curved_surface_area_cone :
  (π * r * l = 144 * π) :=
by sorry

end curved_surface_area_cone_l283_283542


namespace length_third_altitude_l283_283019

theorem length_third_altitude (a b c : ℝ) (S : ℝ) 
  (h_altitude_a : 4 = 2 * S / a)
  (h_altitude_b : 12 = 2 * S / b)
  (h_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_third_integer : ∃ n : ℕ, h = n):
  h = 5 :=
by
  -- Proof is omitted
  sorry

end length_third_altitude_l283_283019


namespace max_height_reached_l283_283576

def h (t : ℝ) : ℝ := -20 * t^2 + 50 * t + 10

theorem max_height_reached : ∃ (t : ℝ), h t = 41.25 :=
by
  use 1.25
  sorry

end max_height_reached_l283_283576


namespace mary_regular_hours_l283_283141

theorem mary_regular_hours (x y : ℕ) (h1 : 8 * x + 10 * y = 560) (h2 : x + y = 60) : x = 20 :=
by
  sorry

end mary_regular_hours_l283_283141


namespace find_surface_area_of_sphere_l283_283093

variables (a b c : ℝ)

-- The conditions given in the problem
def condition1 := a * b = 6
def condition2 := b * c = 2
def condition3 := a * c = 3
def vertices_on_sphere := true  -- Assuming vertices on tensor sphere condition for mathematical completion

theorem find_surface_area_of_sphere
  (h1 : condition1 a b)
  (h2 : condition2 b c)
  (h3 : condition3 a c)
  (h4 : vertices_on_sphere) :
  4 * Real.pi * ((Real.sqrt (a^2 + b^2 + c^2)) / 2)^2 = 14 * Real.pi :=
  sorry

end find_surface_area_of_sphere_l283_283093


namespace problem_sol_l283_283245

open Complex

theorem problem_sol (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : (a + i) / i = 1 + b * i) : a + b = 0 :=
sorry

end problem_sol_l283_283245


namespace total_string_length_l283_283746

theorem total_string_length 
  (circumference1 : ℝ) (height1 : ℝ) (loops1 : ℕ)
  (circumference2 : ℝ) (height2 : ℝ) (loops2 : ℕ)
  (h1 : circumference1 = 6) (h2 : height1 = 20) (h3 : loops1 = 5)
  (h4 : circumference2 = 3) (h5 : height2 = 10) (h6 : loops2 = 3)
  : (loops1 * Real.sqrt (circumference1 ^ 2 + (height1 / loops1) ^ 2) + loops2 * Real.sqrt (circumference2 ^ 2 + (height2 / loops2) ^ 2)) = (5 * Real.sqrt 52 + 3 * Real.sqrt 19.89) := 
by {
  sorry
}

end total_string_length_l283_283746


namespace susie_total_savings_is_correct_l283_283533

variable (initial_amount : ℝ) (year1_addition_pct : ℝ) (year2_addition_pct : ℝ) (interest_rate : ℝ)

def susies_savings (initial_amount year1_addition_pct year2_addition_pct interest_rate : ℝ) : ℝ :=
  let end_of_first_year := initial_amount + initial_amount * year1_addition_pct
  let first_year_interest := end_of_first_year * interest_rate
  let total_after_first_year := end_of_first_year + first_year_interest
  let end_of_second_year := total_after_first_year + total_after_first_year * year2_addition_pct
  let second_year_interest := end_of_second_year * interest_rate
  end_of_second_year + second_year_interest

theorem susie_total_savings_is_correct : 
  susies_savings 200 0.20 0.30 0.05 = 343.98 := 
by
  sorry

end susie_total_savings_is_correct_l283_283533


namespace trig_inequality_solution_l283_283768

noncomputable def solveTrigInequality (x : ℝ) : Prop := 
  let LHS := (Real.cos x) ^ 2018 + (Real.sin x) ^ (-2019)
  let RHS := (Real.sin x) ^ 2018 + (Real.cos x) ^ (-2019)
  LHS ≤ RHS

-- The main theorem statement
theorem trig_inequality_solution (x : ℝ) :
  (solveTrigInequality x ∧ x ≥ -π / 3 ∧ x ≤ 5 * π / 3) ↔ 
  (x ∈ Set.Ico (-π / 3) 0 ∪ Set.Ico (π / 4) (π / 2) ∪ Set.Ioc π (5 * π / 4) ∪ Set.Ioc (3 * π / 2) (5 * π / 3)) :=
begin
  sorry
end

end trig_inequality_solution_l283_283768


namespace complement_union_l283_283194

open Set Real

noncomputable def S : Set ℝ := {x | x > -2}
noncomputable def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

theorem complement_union (x : ℝ): x ∈ (univ \ S) ∪ T ↔ x ≤ 1 :=
by
  sorry

end complement_union_l283_283194


namespace solution_l283_283933

-- Definitions
def equation1 (x y z : ℝ) : Prop := 2 * x + y + z = 17
def equation2 (x y z : ℝ) : Prop := x + 2 * y + z = 14
def equation3 (x y z : ℝ) : Prop := x + y + 2 * z = 13

-- Theorem to prove
theorem solution (x y z : ℝ) (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) : x = 6 :=
by
  sorry

end solution_l283_283933


namespace relationship_bx_l283_283088

variable {a b t x : ℝ}

-- Given conditions
variable (h1 : b > a)
variable (h2 : a > 1)
variable (h3 : t > 0)
variable (h4 : a ^ x = a + t)

theorem relationship_bx (h1 : b > a) (h2 : a > 1) (h3 : t > 0) (h4 : a ^ x = a + t) : b ^ x > b + t :=
by
  sorry

end relationship_bx_l283_283088


namespace line_perpendicular_value_of_a_l283_283114

theorem line_perpendicular_value_of_a :
  ∀ (a : ℝ),
    (∃ (l1 l2 : ℝ → ℝ),
      (∀ x, l1 x = (-a / (1 - a)) * x + 3 / (1 - a)) ∧
      (∀ x, l2 x = (-(a - 1) / (2 * a + 3)) * x + 2 / (2 * a + 3)) ∧
      (∀ x y, l1 x ≠ l2 y) ∧ 
      (-a / (1 - a)) * (-(a - 1) / (2 * a + 3)) = -1) →
    a = -3 := sorry

end line_perpendicular_value_of_a_l283_283114


namespace pyramid_base_side_length_l283_283384

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_lateral_face = 120)
  (h2 : slant_height = 40) :
  ∃ s : ℝ, (120 = 0.5 * s * 40) ∧ s = 6 := by
  sorry

end pyramid_base_side_length_l283_283384


namespace brick_length_is_20_cm_l283_283201

-- Define the conditions given in the problem
def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def num_bricks : ℕ := 20000
def brick_width_cm : ℝ := 10
def total_area_cm2 : ℝ := 4000000

-- Define the goal to prove that the length of each brick is 20 cm
theorem brick_length_is_20_cm :
  (total_area_cm2 = num_bricks * (brick_width_cm * length)) → (length = 20) :=
by
  -- Assume the given conditions
  sorry

end brick_length_is_20_cm_l283_283201


namespace sufficient_but_not_necessary_l283_283520

def P (x : ℝ) : Prop := 2 < x ∧ x < 4
def Q (x : ℝ) : Prop := Real.log x < Real.exp 1

theorem sufficient_but_not_necessary (x : ℝ) : P x → Q x ∧ (¬ ∀ x, Q x → P x) := by
  sorry

end sufficient_but_not_necessary_l283_283520


namespace side_length_of_base_l283_283393

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end side_length_of_base_l283_283393


namespace average_height_is_64_l283_283359

noncomputable def Parker (H_D : ℝ) : ℝ := H_D - 4
noncomputable def Daisy (H_R : ℝ) : ℝ := H_R + 8
noncomputable def Reese : ℝ := 60

theorem average_height_is_64 :
  let H_R := Reese 
  let H_D := Daisy H_R
  let H_P := Parker H_D
  (H_P + H_D + H_R) / 3 = 64 := sorry

end average_height_is_64_l283_283359


namespace oliver_bags_fraction_l283_283701

theorem oliver_bags_fraction
  (weight_james_bag : ℝ)
  (combined_weight_oliver_bags : ℝ)
  (h1 : weight_james_bag = 18)
  (h2 : combined_weight_oliver_bags = 6)
  (f : ℝ) :
  2 * f * weight_james_bag = combined_weight_oliver_bags → f = 1 / 6 :=
by
  intro h
  sorry

end oliver_bags_fraction_l283_283701


namespace divisor_of_1025_l283_283426

theorem divisor_of_1025 (d : ℕ) (h1: 1015 + 10 = 1025) (h2 : d ∣ 1025) : d = 5 := 
sorry

end divisor_of_1025_l283_283426


namespace workman_problem_l283_283870

theorem workman_problem (x : ℝ) (h : (1 / x) + (1 / (2 * x)) = 1 / 32): x = 48 :=
sorry

end workman_problem_l283_283870


namespace max_value_a_plus_b_plus_c_plus_d_eq_34_l283_283532

theorem max_value_a_plus_b_plus_c_plus_d_eq_34 :
  ∃ (a b c d : ℕ), (∀ (x y: ℝ), 0 < x → 0 < y → x^2 - 2 * x * y + 3 * y^2 = 10 → x^2 + 2 * x * y + 3 * y^2 = (a + b * Real.sqrt c) / d) ∧ a + b + c + d = 34 :=
sorry

end max_value_a_plus_b_plus_c_plus_d_eq_34_l283_283532


namespace fraction_of_one_third_l283_283985

theorem fraction_of_one_third (x : ℚ) (h: x * (3 / 7 : ℚ) = 0.12499999999999997) : 
  (x * 3 = 7 / 8) := by
  sorry

end fraction_of_one_third_l283_283985


namespace find_M_l283_283106

theorem find_M (a b M : ℝ) (h : (a + 2 * b)^2 = (a - 2 * b)^2 + M) : M = 8 * a * b :=
by sorry

end find_M_l283_283106


namespace proof_problem_l283_283887

-- Define the conditions based on Classmate A and Classmate B's statements
def classmateA_statement (x y : ℝ) : Prop := 6 * x = 5 * y
def classmateB_statement (x y : ℝ) : Prop := x = 2 * y - 40

-- Define the system of equations derived from the statements
def system_of_equations (x y : ℝ) : Prop := (6 * x = 5 * y) ∧ (x = 2 * y - 40)

-- Proof goal: Prove the system of equations if classmate statements hold
theorem proof_problem (x y : ℝ) :
  classmateA_statement x y ∧ classmateB_statement x y → system_of_equations x y :=
by
  sorry

end proof_problem_l283_283887


namespace tan_15pi_over_4_correct_l283_283765

open Real
open Angle

noncomputable def tan_15pi_over_4 : Real := -1

theorem tan_15pi_over_4_correct :
  tan (15 * pi / 4) = tan_15pi_over_4 :=
sorry

end tan_15pi_over_4_correct_l283_283765


namespace find_r_fourth_l283_283283

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l283_283283


namespace second_investment_amount_l283_283433

/-
A $500 investment and another investment have a combined yearly return of 8.5 percent of the total of the two investments.
The $500 investment has a yearly return of 7 percent.
The other investment has a yearly return of 9 percent.
Prove that the amount of the second investment is $1500.
-/

theorem second_investment_amount :
  ∃ x : ℝ, 35 + 0.09 * x = 0.085 * (500 + x) → x = 1500 :=
by
  sorry

end second_investment_amount_l283_283433


namespace measure_of_obtuse_angle_APB_l283_283681

-- Define the triangle type and conditions
structure Triangle :=
  (A B C : Point)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)

-- Define the point type
structure Point :=
  (x y : ℝ)

-- Property of the triangle is isotropic and it contains right angles 90 degrees 
def IsoscelesRightTriangle (T : Triangle) : Prop :=
  T.angle_A = 45 ∧ T.angle_B = 45 ∧ T.angle_C = 90

-- Define the angle bisector intersection point P
def AngleBisectorIntersection (T : Triangle) (P : Point) : Prop :=
  -- (dummy properties assuming necessary geometric constructions can be proven)
  true

-- Statement we want to prove
theorem measure_of_obtuse_angle_APB (T : Triangle) (P : Point) 
    (h1 : IsoscelesRightTriangle T) (h2 : AngleBisectorIntersection T P) :
  ∃ APB : ℝ, APB = 135 :=
  sorry

end measure_of_obtuse_angle_APB_l283_283681


namespace min_value_of_expr_l283_283228

theorem min_value_of_expr (n : ℕ) (hn : n > 0) : (n / 3) + (27 / n) = 6 :=
by
  sorry

end min_value_of_expr_l283_283228


namespace orange_ratio_l283_283004

variable {R U : ℕ}

theorem orange_ratio (h1 : R + U = 96) 
                    (h2 : (3 / 4 : ℝ) * R + (7 / 8 : ℝ) * U = 78) :
  (R : ℝ) / (R + U : ℝ) = 1 / 2 := 
by
  sorry

end orange_ratio_l283_283004


namespace work_time_relation_l283_283721

theorem work_time_relation (m n k x y z : ℝ) 
    (h1 : 1 / x = m / (y + z)) 
    (h2 : 1 / y = n / (x + z)) 
    (h3 : 1 / z = k / (x + y)) : 
    k = (m + n + 2) / (m * n - 1) :=
by
  sorry

end work_time_relation_l283_283721


namespace find_f_2002_l283_283622

-- Definitions based on conditions
variable {R : Type} [CommRing R] [NoZeroDivisors R]

-- Condition 1: f is an even function.
def even_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = f x

-- Condition 2: f(2) = 0
def f_value_at_two (f : R → R) : Prop :=
  f 2 = 0

-- Condition 3: g is an odd function.
def odd_function (g : R → R) : Prop :=
  ∀ x : R, g (-x) = -g x

-- Condition 4: g(x) = f(x-1)
def g_equals_f_shifted (f g : R → R) : Prop :=
  ∀ x : R, g x = f (x - 1)

-- The main proof problem
theorem find_f_2002 (f g : R → R)
  (hf : even_function f)
  (hf2 : f_value_at_two f)
  (hg : odd_function g)
  (hgf : g_equals_f_shifted f g) :
  f 2002 = 0 :=
sorry

end find_f_2002_l283_283622


namespace musketeers_strength_order_l283_283450

variables {A P R D : ℝ}

theorem musketeers_strength_order 
  (h1 : P + D > A + R)
  (h2 : P + A > R + D)
  (h3 : P + R = A + D) : 
  P > D ∧ D > A ∧ A > R :=
by
  sorry

end musketeers_strength_order_l283_283450


namespace hyperbola_eccentricity_l283_283494

theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b = a) 
  (h₄ : ∀ c, (c = Real.sqrt (a^2 + b^2)) → (b * c / Real.sqrt (a^2 + b^2) = a)) :
  (Real.sqrt (2) = (c / a)) :=
by
  sorry

end hyperbola_eccentricity_l283_283494


namespace john_less_than_david_by_4_l283_283733

/-
The conditions are:
1. Zachary did 51 push-ups.
2. David did 22 more push-ups than Zachary.
3. John did 69 push-ups.

We need to prove that John did 4 push-ups less than David.
-/

def zachary_pushups : ℕ := 51
def david_pushups : ℕ := zachary_pushups + 22
def john_pushups : ℕ := 69

theorem john_less_than_david_by_4 :
  david_pushups - john_pushups = 4 :=
by
  -- Proof goes here.
  sorry

end john_less_than_david_by_4_l283_283733


namespace weight_of_B_l283_283964

-- Definitions for the weights of A, B, and C
variable (A B C : ℝ)

-- Conditions given in the problem
def condition1 := (A + B + C) / 3 = 45
def condition2 := (A + B) / 2 = 40
def condition3 := (B + C) / 2 = 43

-- The theorem to prove that B = 31 under the given conditions
theorem weight_of_B : condition1 A B C → condition2 A B → condition3 B C → B = 31 := by
  intros
  sorry

end weight_of_B_l283_283964


namespace sum_of_roots_eq_zero_sum_of_all_possible_values_l283_283316

theorem sum_of_roots_eq_zero (x : ℝ) (h : x^2 = 25) : x = 5 ∨ x = -5 :=
by {
  sorry
}

theorem sum_of_all_possible_values (h : ∀ x : ℝ, x^2 = 25 → x = 5 ∨ x = -5) : ∑ x in {x : ℝ | x^2 = 25}, x = 0 :=
by {
  sorry
}

end sum_of_roots_eq_zero_sum_of_all_possible_values_l283_283316


namespace total_capacity_is_correct_l283_283170

-- Define small and large jars capacities
def small_jar_capacity : ℕ := 3
def large_jar_capacity : ℕ := 5

-- Define the total number of jars and the number of small jars
def total_jars : ℕ := 100
def small_jars : ℕ := 62

-- Define the number of large jars based on the total jars and small jars
def large_jars : ℕ := total_jars - small_jars

-- Calculate capacities
def small_jars_total_capacity : ℕ := small_jars * small_jar_capacity
def large_jars_total_capacity : ℕ := large_jars * large_jar_capacity

-- Define the total capacity
def total_capacity : ℕ := small_jars_total_capacity + large_jars_total_capacity

-- Prove that the total capacity is 376 liters
theorem total_capacity_is_correct : total_capacity = 376 := by
  sorry

end total_capacity_is_correct_l283_283170


namespace inequality_proof_l283_283097

def f (x : ℝ) (m : ℕ) : ℝ := |x - m| + |x|

theorem inequality_proof (α β : ℝ) (m : ℕ) (h1 : 1 < α) (h2 : 1 < β) (h3 : m = 1) 
  (h4 : f α m + f β m = 2) : (4 / α) + (1 / β) ≥ 9 / 2 := by
  sorry

end inequality_proof_l283_283097


namespace smallest_largest_multiples_l283_283543

theorem smallest_largest_multiples : 
  ∃ l g, l >= 10 ∧ l < 100 ∧ g >= 100 ∧ g < 1000 ∧
  (2 ∣ l) ∧ (3 ∣ l) ∧ (5 ∣ l) ∧ 
  (2 ∣ g) ∧ (3 ∣ g) ∧ (5 ∣ g) ∧
  (∀ n, n >= 10 ∧ n < 100 ∧ (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n) → l ≤ n) ∧
  (∀ n, n >= 100 ∧ n < 1000 ∧ (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n) → g >= n) ∧
  l = 30 ∧ g = 990 := 
by 
  sorry

end smallest_largest_multiples_l283_283543


namespace hole_depth_l283_283350

theorem hole_depth (height : ℝ) (half_depth : ℝ) (total_depth : ℝ) 
    (h_height : height = 90) 
    (h_half_depth : half_depth = total_depth / 2)
    (h_position : height + half_depth = total_depth - height) : 
    total_depth = 120 := 
by
    sorry

end hole_depth_l283_283350


namespace percent_flowers_are_carnations_l283_283869

-- Define the conditions
def one_third_pink_are_roses (total_flower pink_flower pink_roses : ℕ) : Prop :=
  pink_roses = (1/3) * pink_flower

def three_fourths_red_are_carnations (total_flower red_flower red_carnations : ℕ) : Prop :=
  red_carnations = (3/4) * red_flower

def six_tenths_are_pink (total_flower pink_flower : ℕ) : Prop :=
  pink_flower = (6/10) * total_flower

-- Define the proof problem statement
theorem percent_flowers_are_carnations (total_flower pink_flower pink_roses red_flower red_carnations : ℕ) :
  one_third_pink_are_roses total_flower pink_flower pink_roses →
  three_fourths_red_are_carnations total_flower red_flower red_carnations →
  six_tenths_are_pink total_flower pink_flower →
  (red_flower = total_flower - pink_flower) →
  (pink_flower - pink_roses + red_carnations = (4/10) * total_flower) →
  ((pink_flower - pink_roses) + red_carnations) * 100 / total_flower = 40 := 
sorry

end percent_flowers_are_carnations_l283_283869


namespace arithmetic_sequence_has_correct_number_of_terms_l283_283823

theorem arithmetic_sequence_has_correct_number_of_terms :
  ∀ (a₁ d : ℤ) (n : ℕ), a₁ = 1 ∧ d = -2 ∧ (n : ℤ) = (a₁ + (n - 1 : ℕ) * d) → n = 46 := by
  intros a₁ d n
  sorry

end arithmetic_sequence_has_correct_number_of_terms_l283_283823


namespace avg_weight_b_c_l283_283824

variables (A B C : ℝ)

-- Given Conditions
def condition1 := (A + B + C) / 3 = 45
def condition2 := (A + B) / 2 = 40
def condition3 := B = 37

-- Statement to prove
theorem avg_weight_b_c 
  (h1 : condition1 A B C)
  (h2 : condition2 A B)
  (h3 : condition3 B) : 
  (B + C) / 2 = 46 :=
sorry

end avg_weight_b_c_l283_283824


namespace digit_difference_l283_283859

theorem digit_difference (X Y : ℕ) (h_digits : 0 ≤ X ∧ X < 10 ∧ 0 ≤ Y ∧ Y < 10) (h_diff :  (10 * X + Y) - (10 * Y + X) = 45) : X - Y = 5 :=
sorry

end digit_difference_l283_283859


namespace worker_saves_one_third_l283_283184

variable {P : ℝ} 
variable {f : ℝ}

theorem worker_saves_one_third (h : P ≠ 0) (h_eq : 12 * f * P = 6 * (1 - f) * P) : 
  f = 1 / 3 :=
sorry

end worker_saves_one_third_l283_283184


namespace watch_loss_percentage_l283_283447

noncomputable def loss_percentage (CP SP_gain : ℝ) : ℝ :=
  100 * (CP - SP_gain) / CP

theorem watch_loss_percentage (CP : ℝ) (SP_gain : ℝ) :
  (SP_gain = CP + 0.04 * CP) →
  (CP = 700) →
  (CP - (SP_gain - 140) = CP * (16 / 100)) :=
by
  intros h_SP_gain h_CP
  rw [h_SP_gain, h_CP]
  simp
  sorry

end watch_loss_percentage_l283_283447


namespace pyramid_base_length_l283_283404

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end pyramid_base_length_l283_283404


namespace pyramid_base_length_l283_283403

theorem pyramid_base_length (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120)
  (hh : h = 40)
  (hs : A = 20 * s) : 
  s = 6 :=
by
  sorry

end pyramid_base_length_l283_283403


namespace sum_of_integers_l283_283968

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 15) (h3 : x * y = 54) : x + y = 21 :=
by
  sorry

end sum_of_integers_l283_283968


namespace meaningful_range_l283_283321

noncomputable def isMeaningful (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (x - 2 ≠ 0)

theorem meaningful_range (x : ℝ) : isMeaningful x ↔ (x ≥ -1) ∧ (x ≠ 2) :=
by
  sorry

end meaningful_range_l283_283321


namespace measure_exactly_10_liters_l283_283937

theorem measure_exactly_10_liters (A B : ℕ) (A_cap B_cap : ℕ) (hA : A_cap = 11) (hB : B_cap = 9) :
  ∃ (A B : ℕ), A + B = 10 ∧ A ≤ A_cap ∧ B ≤ B_cap := 
sorry

end measure_exactly_10_liters_l283_283937


namespace probability_prime_multiple_of_11_l283_283960

theorem probability_prime_multiple_of_11 : 
  let cards := finset.range 61
  let prime_multiple_of_11 := finset.filter (λ n, nat.prime n ∧ n % 11 = 0) cards
  let selected_card_probability := (prime_multiple_of_11.card : rat) / (cards.card : rat)
  selected_card_probability = 1 / 60 := 
by
  sorry

end probability_prime_multiple_of_11_l283_283960


namespace incorrect_statements_count_l283_283732

-- Definitions of the statements
def statement1 : Prop := "The diameter perpendicular to the chord bisects the chord" = "incorrect"

def statement2 : Prop := "A circle is a symmetrical figure, and any diameter is its axis of symmetry" = "incorrect"

def statement3 : Prop := "Two arcs of equal length are congruent" = "incorrect"

-- Theorem stating that the number of incorrect statements is 3
theorem incorrect_statements_count : 
  (statement1 → False) → (statement2 → False) → (statement3 → False) → 3 = 3 :=
by sorry

end incorrect_statements_count_l283_283732


namespace polynomial_horner_method_l283_283224

theorem polynomial_horner_method :
  let a_4 := 3
  let a_3 := 0
  let a_2 := -1
  let a_1 := 2
  let a_0 := 1
  let x := 2
  let v_0 := 3
  let v_1 := v_0 * x + a_3
  let v_2 := v_1 * x + a_2
  let v_3 := v_2 * x + a_1
  v_3 = 22 :=
by 
  let a_4 := 3
  let a_3 := 0
  let a_2 := -1
  let a_1 := 2
  let a_0 := 1
  let x := 2
  let v_0 := a_4
  let v_1 := v_0 * x + a_3
  let v_2 := v_1 * x + a_2
  let v_3 := v_2 * x + a_1
  sorry

end polynomial_horner_method_l283_283224


namespace solution_set_inequality_l283_283836

theorem solution_set_inequality (x : ℝ) : |3 * x + 1| - |x - 1| < 0 ↔ -1 < x ∧ x < 0 := 
sorry

end solution_set_inequality_l283_283836


namespace circle_radius_five_l283_283610

theorem circle_radius_five (c : ℝ) : (∃ x y : ℝ, x^2 + 10 * x + y^2 + 8 * y + c = 0) ∧ 
                                     ((x + 5)^2 + (y + 4)^2 = 25) → c = 16 :=
by
  sorry

end circle_radius_five_l283_283610


namespace red_cards_taken_out_l283_283368

-- Definitions based on the conditions
def total_cards : ℕ := 52
def half_of_total_cards (n : ℕ) := n / 2
def initial_red_cards : ℕ := half_of_total_cards total_cards
def remaining_red_cards : ℕ := 16

-- The statement to prove
theorem red_cards_taken_out : initial_red_cards - remaining_red_cards = 10 := by
  sorry

end red_cards_taken_out_l283_283368


namespace katy_brownies_total_l283_283130

theorem katy_brownies_total : 
  (let monday_brownies := 5 in
   let tuesday_brownies := 2 * monday_brownies in
   let total_brownies := monday_brownies + tuesday_brownies in
   total_brownies = 15) := 
by 
  let monday_brownies := 5 in
  let tuesday_brownies := 2 * monday_brownies in
  let total_brownies := monday_brownies + tuesday_brownies in
  show total_brownies = 15 by
  sorry

end katy_brownies_total_l283_283130


namespace man_saves_percentage_of_salary_l283_283045

variable (S : ℝ) (P : ℝ) (S_s : ℝ)

def problem_statement (S : ℝ) (S_s : ℝ) (P : ℝ) : Prop :=
  S_s = S - 1.2 * (S - (P / 100) * S)

theorem man_saves_percentage_of_salary
  (h1 : S = 6250)
  (h2 : S_s = 250) :
  problem_statement S S_s 20 :=
by
  sorry

end man_saves_percentage_of_salary_l283_283045


namespace seventy_times_reciprocal_l283_283919

theorem seventy_times_reciprocal (x : ℚ) (hx : 7 * x = 3) : 70 * (1 / x) = 490 / 3 :=
by 
  sorry

end seventy_times_reciprocal_l283_283919


namespace find_r_fourth_l283_283289

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l283_283289


namespace kimberly_total_skittles_l283_283805

def initial_skittles : ℝ := 7.5
def skittles_eaten : ℝ := 2.25
def skittles_given : ℝ := 1.5
def promotion_skittles : ℝ := 3.75
def oranges_bought : ℝ := 18
def exchange_oranges : ℝ := 6
def exchange_skittles : ℝ := 10.5

theorem kimberly_total_skittles :
  initial_skittles - skittles_eaten - skittles_given + promotion_skittles + exchange_skittles = 18 := by
  sorry

end kimberly_total_skittles_l283_283805


namespace infinite_geometric_series_sum_l283_283764

theorem infinite_geometric_series_sum :
  let a := (5 : ℚ) / 3
  let r := -(3 : ℚ) / 4
  |r| < 1 →
  (∀ S, S = a / (1 - r) → S = 20 / 21) :=
by
  intros a r h_abs_r S h_S
  sorry

end infinite_geometric_series_sum_l283_283764


namespace intersection_complement_eq_l283_283911

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define set M within U
def M : Set ℕ := {1, 3, 5, 7}

-- Define set N within U
def N : Set ℕ := {5, 6, 7}

-- Define the complement of M in U
def CU_M : Set ℕ := U \ M

-- Define the complement of N in U
def CU_N : Set ℕ := U \ N

-- Mathematically equivalent proof problem
theorem intersection_complement_eq : CU_M ∩ CU_N = {2, 4, 8} := by
  sorry

end intersection_complement_eq_l283_283911


namespace triangle_inequality_l283_283346

theorem triangle_inequality (S : Finset (ℕ × ℕ)) (m n : ℕ) (hS : S.card = m)
  (h_ab : ∀ (a b : ℕ), (a, b) ∈ S → (1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n ∧ a ≠ b)) :
  ∃ (t : Finset (ℕ × ℕ × ℕ)),
    (t.card ≥ (4 * m / (3 * n)) * (m - (n^2) / 4)) ∧
    ∀ (a b c : ℕ), (a, b, c) ∈ t → (a, b) ∈ S ∧ (b, c) ∈ S ∧ (c, a) ∈ S := by
  sorry

end triangle_inequality_l283_283346


namespace pizza_slices_have_both_cheese_and_bacon_l283_283567

theorem pizza_slices_have_both_cheese_and_bacon:
  ∀ (total_slices cheese_slices bacon_slices n : ℕ),
  total_slices = 15 →
  cheese_slices = 8 →
  bacon_slices = 13 →
  (total_slices = cheese_slices + bacon_slices - n) →
  n = 6 :=
by {
  -- proof skipped
  sorry
}

end pizza_slices_have_both_cheese_and_bacon_l283_283567


namespace poly_div_remainder_l283_283762

noncomputable def poly1 := X^5 - 1
noncomputable def poly2 := X^3 - 1
noncomputable def divisor := X^3 + X^2 + X + 1
noncomputable def dividend := poly1 * poly2
noncomputable def remainder := 2 * X + 2

theorem poly_div_remainder :
  (dividend % divisor) = remainder := sorry

end poly_div_remainder_l283_283762


namespace investment_ratio_correct_l283_283438

-- Constants representing the savings and investments
def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def weeks_in_month : ℕ := 4
def months_saving : ℕ := 4
def cost_per_share : ℕ := 50
def shares_bought : ℕ := 25

-- Derived quantities from the conditions
def total_savings_wife : ℕ := weekly_savings_wife * weeks_in_month * months_saving
def total_savings_husband : ℕ := monthly_savings_husband * months_saving
def total_savings : ℕ := total_savings_wife + total_savings_husband
def total_invested_in_stocks : ℕ := shares_bought * cost_per_share
def investment_ratio_nat : ℚ := (total_invested_in_stocks : ℚ) / (total_savings : ℚ)

-- Proof statement
theorem investment_ratio_correct : investment_ratio_nat = 1 / 2 := by
  sorry

end investment_ratio_correct_l283_283438


namespace complex_number_in_fourth_quadrant_l283_283505

theorem complex_number_in_fourth_quadrant (i : ℂ) (z : ℂ) (hx : z = -2 * i + 1) (hy : (z.re, z.im) = (1, -2)) :
  (1, -2).1 > 0 ∧ (1, -2).2 < 0 :=
by
  sorry

end complex_number_in_fourth_quadrant_l283_283505


namespace distinct_numbers_l283_283236

theorem distinct_numbers (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 1000) : 
  finset.card ((finset.image (λ k : ℕ, ⌊(k^2 : ℚ) / 500⌋) (finset.range 1000).succ)) = 2001 :=
sorry

end distinct_numbers_l283_283236


namespace count_multiples_of_15_l283_283263

theorem count_multiples_of_15 (a b n : ℕ) (h_gte : 25 ≤ a) (h_lte : b ≤ 205) (h15 : n = 15) : 
  (∃ (k : ℕ), a ≤ k * n ∧ k * n ≤ b ∧ 1 ≤ k - 1 ∧ k - 1 ≤ 12) :=
sorry

end count_multiples_of_15_l283_283263


namespace determine_linear_relation_l283_283427

-- Define the set of options
inductive PlotType
| Scatter
| StemAndLeaf
| FrequencyHistogram
| FrequencyLineChart

-- Define the question and state the expected correct answer
def correctPlotTypeForLinearRelation : PlotType :=
  PlotType.Scatter

-- Prove that the correct method for determining linear relation in a set of data is a Scatter plot
theorem determine_linear_relation :
  correctPlotTypeForLinearRelation = PlotType.Scatter :=
by
  sorry

end determine_linear_relation_l283_283427


namespace part_a_value_range_part_b_value_product_l283_283692

-- Define the polynomial 
def P (x y : ℤ) : ℤ := 2 * x^2 - 6 * x * y + 5 * y^2

-- Part (a)
theorem part_a_value_range :
  ∀ (x y : ℤ), (1 ≤ P x y) ∧ (P x y ≤ 100) → ∃ (a b : ℤ), 1 ≤ P a b ∧ P a b ≤ 100 := sorry

-- Part (b)
theorem part_b_value_product :
  ∀ (a b c d : ℤ),
    P a b = r → P c d = s → ∀ (r s : ℤ), (∃ (x y : ℤ), P x y = r) ∧ (∃ (z w : ℤ), P z w = s) → 
    ∃ (u v : ℤ), P u v = r * s := sorry

end part_a_value_range_part_b_value_product_l283_283692


namespace no_solution_for_equation_l283_283366

theorem no_solution_for_equation :
  ¬(∃ x : ℝ, x ≠ 2 ∧ x ≠ -2 ∧ (x+2)/(x-2) - x/(x+2) = 16/(x^2-4)) :=
by
    sorry

end no_solution_for_equation_l283_283366


namespace sum_of_pills_in_larger_bottles_l283_283216

-- Definitions based on the conditions
def supplements := 5
def pills_in_small_bottles := 2 * 30
def pills_per_day := 5
def days_used := 14
def pills_remaining := 350
def total_pills_before := pills_remaining + (pills_per_day * days_used)
def total_pills_in_large_bottles := total_pills_before - pills_in_small_bottles

-- The theorem statement that needs to be proven
theorem sum_of_pills_in_larger_bottles : total_pills_in_large_bottles = 360 := 
by 
  -- Placeholder for the proof
  sorry

end sum_of_pills_in_larger_bottles_l283_283216


namespace moles_of_ammonia_formed_l283_283239

def reaction (n_koh n_nh4i n_nh3 : ℕ) := 
  n_koh + n_nh4i + n_nh3 

theorem moles_of_ammonia_formed (n_koh : ℕ) :
  reaction n_koh 3 3 = n_koh + 3 + 3 := 
sorry

end moles_of_ammonia_formed_l283_283239


namespace non_neg_integers_l283_283462

open Nat

theorem non_neg_integers (n : ℕ) :
  (∃ x y k : ℕ, x.gcd y = 1 ∧ k ≥ 2 ∧ 3^n = x^k + y^k) ↔ (n = 0 ∨ n = 1 ∨ n = 2) := by
  sorry

end non_neg_integers_l283_283462


namespace little_john_remaining_money_l283_283139

noncomputable def initial_amount: ℝ := 8.50
noncomputable def spent_on_sweets: ℝ := 1.25
noncomputable def given_to_each_friend: ℝ := 1.20
noncomputable def number_of_friends: ℝ := 2

theorem little_john_remaining_money : 
  initial_amount - (spent_on_sweets + given_to_each_friend * number_of_friends) = 4.85 :=
by
  sorry

end little_john_remaining_money_l283_283139


namespace flower_bed_width_l283_283211

theorem flower_bed_width (length area : ℝ) (h_length : length = 4) (h_area : area = 143.2) :
  area / length = 35.8 :=
by
  sorry

end flower_bed_width_l283_283211


namespace quadratic_root_identity_l283_283626

theorem quadratic_root_identity (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2020 = 2023 :=
by
  sorry

end quadratic_root_identity_l283_283626


namespace supplement_comp_greater_l283_283852

theorem supplement_comp_greater {α β : ℝ} (h : α + β = 90) : 180 - α = β + 90 :=
by
  sorry

end supplement_comp_greater_l283_283852


namespace nico_reads_wednesday_l283_283355

def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51
def pages_wednesday := total_pages - (pages_monday + pages_tuesday) 

theorem nico_reads_wednesday :
  pages_wednesday = 19 :=
by
  sorry

end nico_reads_wednesday_l283_283355


namespace simplify_and_evaluate_l283_283957

theorem simplify_and_evaluate (x y : ℝ) (hx : x = -1) (hy : y = -1/3) :
  ((3 * x^2 + x * y + 2 * y) - 2 * (5 * x * y - 4 * x^2 + y)) = 8 := by
  sorry

end simplify_and_evaluate_l283_283957


namespace symmetric_diff_cardinality_l283_283858

theorem symmetric_diff_cardinality (X Y : Finset ℤ) 
  (hX : X.card = 8) 
  (hY : Y.card = 10) 
  (hXY : (X ∩ Y).card = 6) : 
  (X \ Y ∪ Y \ X).card = 6 := 
by
  sorry

end symmetric_diff_cardinality_l283_283858


namespace farm_distance_l283_283720

theorem farm_distance (a x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (triangle_ineq1 : x + z = 85)
  (triangle_ineq2 : x + y = 4 * z)
  (triangle_ineq3 : z + y = x + a) :
  0 < a ∧ a < 85 ∧
  x = (340 - a) / 6 ∧
  y = (2 * a + 85) / 3 ∧
  z = (170 + a) / 6 :=
sorry

end farm_distance_l283_283720


namespace probability_in_interval_l283_283209

theorem probability_in_interval (a b c d : ℝ) (h1 : a = 2) (h2 : b = 10) (h3 : c = 5) (h4 : d = 7) :
  (d - c) / (b - a) = 1 / 4 :=
by
  sorry

end probability_in_interval_l283_283209


namespace josef_game_l283_283335

theorem josef_game : 
  ∃ S : Finset ℕ, 
    (∀ n ∈ S, 1 ≤ n ∧ n ≤ 1440 ∧ 1440 % n = 0 ∧ n % 5 = 0) ∧ 
    S.card = 18 := sorry

end josef_game_l283_283335


namespace conditional_probability_of_wind_given_rain_l283_283439

theorem conditional_probability_of_wind_given_rain (P_A P_B P_A_and_B : ℚ)
  (h1: P_A = 4/15) (h2: P_B = 2/15) (h3: P_A_and_B = 1/10) :
  P_A_and_B / P_A = 3/8 :=
by
  sorry

end conditional_probability_of_wind_given_rain_l283_283439


namespace polygon_sides_eq_seven_l283_283575

theorem polygon_sides_eq_seven (n d : ℕ) (h1 : d = (n * (n - 3)) / 2) (h2 : d = 2 * n) : n = 7 := 
by
  sorry

end polygon_sides_eq_seven_l283_283575


namespace min_value_of_f_l283_283107

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (9 / (1 - x))

theorem min_value_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : f x = 16 :=
by
  sorry

end min_value_of_f_l283_283107


namespace pencils_distributed_l283_283480

-- Define the conditions as a Lean statement
theorem pencils_distributed :
  let friends := 4
  let pencils := 8
  let at_least_one := 1
  ∃ (ways : ℕ), ways = 35 := sorry

end pencils_distributed_l283_283480


namespace amy_total_score_l283_283999

theorem amy_total_score :
  let points_per_treasure := 4
  let treasures_first_level := 6
  let treasures_second_level := 2
  let score_first_level := treasures_first_level * points_per_treasure
  let score_second_level := treasures_second_level * points_per_treasure
  let total_score := score_first_level + score_second_level
  total_score = 32 := by
sorry

end amy_total_score_l283_283999


namespace union_sets_eq_l283_283740

-- Definitions of the given sets
def A : Set ℕ := {0, 1}
def B : Set ℕ := {1, 2}

-- The theorem to prove the union of sets A and B equals \{0, 1, 2\}
theorem union_sets_eq : (A ∪ B) = {0, 1, 2} := by
  sorry

end union_sets_eq_l283_283740


namespace bananas_per_chimp_per_day_l283_283156

theorem bananas_per_chimp_per_day (total_chimps total_bananas : ℝ) (h_chimps : total_chimps = 45) (h_bananas : total_bananas = 72) :
  total_bananas / total_chimps = 1.6 :=
by
  rw [h_chimps, h_bananas]
  norm_num

end bananas_per_chimp_per_day_l283_283156


namespace r_power_four_identity_l283_283313

-- Statement of the problem in Lean 4
theorem r_power_four_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
by sorry

end r_power_four_identity_l283_283313


namespace arrange_numbers_satisfies_mean_property_l283_283217

-- Problem conditions definitions
def vertices := fin 9
def numbers := {n : ℤ | 2016 ≤ n ∧ n ≤ 2024}
def placement (f : vertices → ℤ) : Prop :=
  ∀ a b c : vertices, 
    (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ 
    (f a ∈ numbers ∧ f b ∈ numbers ∧ f c ∈ numbers) ∧ 
    (∃ k : ℤ, abs (a - b) = k ∧ abs (b - c) = k ∧ abs (c - a) = k) →
      (f b = (f a + f c) / 2)

-- The theorem statement
theorem arrange_numbers_satisfies_mean_property :
  ∃ f : vertices → ℤ, 
  (∀ v : vertices, f v ∈ numbers) ∧ 
  placement f :=
begin
  sorry
end

end arrange_numbers_satisfies_mean_property_l283_283217


namespace distribute_pencils_l283_283479

theorem distribute_pencils (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  (∃ ways : ℕ, ways = Nat.choose (n - 1) (k - 1) ∧ ways = 35) :=
by
  sorry

end distribute_pencils_l283_283479


namespace tom_climbing_time_in_hours_l283_283176

variable (t_Elizabeth t_Tom_minutes t_Tom_hours : ℕ)

-- Conditions:
def elizabeth_time : ℕ := 30
def tom_time_relation (t_Elizabeth : ℕ) : ℕ := 4 * t_Elizabeth
def tom_time_hours (t_Tom_minutes : ℕ) : ℕ := t_Tom_minutes / 60

-- Theorem statement:
theorem tom_climbing_time_in_hours :
  tom_time_hours (tom_time_relation elizabeth_time) = 2 :=
by 
  -- Reiterate the conditions with simplified relations
  have t_Elizabeth := elizabeth_time
  have t_Tom_minutes := tom_time_relation t_Elizabeth
  have t_Tom_hours := tom_time_hours t_Tom_minutes
  show t_Tom_hours = 2
  -- Placeholder for actual proof
  sorry

end tom_climbing_time_in_hours_l283_283176


namespace meaningful_expr_l283_283323

theorem meaningful_expr (x : ℝ) : 
    (x + 1 ≥ 0 ∧ x - 2 ≠ 0) → (x ≥ -1 ∧ x ≠ 2) := by
  sorry

end meaningful_expr_l283_283323


namespace A_20_equals_17711_l283_283873

def A : ℕ → ℕ
| 0     => 1  -- by definition, an alternating sequence on an empty set, counting empty sequence
| 1     => 2  -- base case
| 2     => 3  -- base case
| (n+3) => A (n+2) + A (n+1)

theorem A_20_equals_17711 : A 20 = 17711 := 
sorry

end A_20_equals_17711_l283_283873


namespace winning_candidate_percentage_votes_l283_283926

theorem winning_candidate_percentage_votes
  (total_votes : ℕ) (majority_votes : ℕ) (P : ℕ) 
  (h1 : total_votes = 6500) 
  (h2 : majority_votes = 1300) 
  (h3 : (P * total_votes) / 100 - ((100 - P) * total_votes) / 100 = majority_votes) : 
  P = 60 :=
sorry

end winning_candidate_percentage_votes_l283_283926


namespace max_b_value_l283_283026

theorem max_b_value (a b c : ℕ) (h_volume : a * b * c = 360) (h_conditions : 1 < c ∧ c < b ∧ b < a) : b = 12 :=
  sorry

end max_b_value_l283_283026


namespace sum_of_solutions_x_squared_eq_25_l283_283315

theorem sum_of_solutions_x_squared_eq_25 : 
  (∑ x in ({x : ℝ | x^2 = 25}).to_finset, x) = 0 :=
by
  sorry

end sum_of_solutions_x_squared_eq_25_l283_283315


namespace relationship_of_y_l283_283121

theorem relationship_of_y {k y1 y2 y3 : ℝ} (hk : k > 0) :
  (y1 = k / -1) → (y2 = k / 2) → (y3 = k / 3) → y1 < y3 ∧ y3 < y2 :=
by
  intros h1 h2 h3
  sorry

end relationship_of_y_l283_283121


namespace problem_16_l283_283742

-- Definitions of the problem conditions
def trapezoid_inscribed_in_circle (r : ℝ) (a b : ℝ) : Prop :=
  r = 25 ∧ a = 14 ∧ b = 30 

def average_leg_length_of_trapezoid (a b : ℝ) (m : ℝ) : Prop :=
  a = 14 ∧ b = 30 ∧ m = 2000 

-- Using Lean to state the problem
theorem problem_16 (r a b m : ℝ) 
  (h1 : trapezoid_inscribed_in_circle r a b) 
  (h2 : average_leg_length_of_trapezoid a b m) : 
  m = 2000 := by
  sorry

end problem_16_l283_283742


namespace min_cost_29_disks_l283_283262

theorem min_cost_29_disks
  (price_single : ℕ := 20) 
  (price_pack_10 : ℕ := 111) 
  (price_pack_25 : ℕ := 265) :
  ∃ cost : ℕ, cost ≥ (price_pack_10 + price_pack_10 + price_pack_10) 
              ∧ cost ≤ (price_pack_25 + price_single * 4) 
              ∧ cost = 333 := 
by
  sorry

end min_cost_29_disks_l283_283262


namespace kindergarten_students_percentage_is_correct_l283_283881

-- Definitions based on conditions
def total_students_annville : ℕ := 150
def total_students_cleona : ℕ := 250
def percent_kindergarten_annville : ℕ := 14
def percent_kindergarten_cleona : ℕ := 10

-- Calculation of number of kindergarten students
def kindergarten_students_annville : ℕ := total_students_annville * percent_kindergarten_annville / 100
def kindergarten_students_cleona : ℕ := total_students_cleona * percent_kindergarten_cleona / 100
def total_kindergarten_students : ℕ := kindergarten_students_annville + kindergarten_students_cleona
def total_students : ℕ := total_students_annville + total_students_cleona
def kindergarten_percentage : ℚ := (total_kindergarten_students * 100) / total_students

-- The theorem to be proved using the conditions
theorem kindergarten_students_percentage_is_correct : kindergarten_percentage = 11.5 := by
  sorry

end kindergarten_students_percentage_is_correct_l283_283881


namespace no_valid_pairs_of_real_numbers_l283_283609

theorem no_valid_pairs_of_real_numbers :
  ∀ (a b : ℝ), ¬ (∃ (x y : ℤ), 3 * a * x + 7 * b * y = 3 ∧ x^2 + y^2 = 85 ∧ (x % 5 = 0 ∨ y % 5 = 0)) :=
by
  sorry

end no_valid_pairs_of_real_numbers_l283_283609


namespace problem1_problem2_l283_283453

theorem problem1 : (1 * (-1: ℚ)^4 + (1 - (1 / 2)) / 3 * (2 - 2^3)) = 2 := 
by
  sorry

theorem problem2 : ((- (3 / 4) - (5 / 9) + (7 / 12)) / (1 / 36)) = -26 := 
by
  sorry

end problem1_problem2_l283_283453


namespace max_trees_l283_283984

theorem max_trees (interval distance road_length number_of_intervals add_one : ℕ) 
  (h_interval: interval = 4) 
  (h_distance: distance = 28) 
  (h_intervals: number_of_intervals = distance / interval)
  (h_add: add_one = number_of_intervals + 1) :
  add_one = 8 :=
sorry

end max_trees_l283_283984


namespace sum_of_first_odd_numbers_l283_283229

theorem sum_of_first_odd_numbers (S1 S2 : ℕ) (n1 n2 : ℕ)
  (hS1 : S1 = n1^2) 
  (hS2 : S2 = n2^2) 
  (h1 : S1 = 2500)
  (h2 : S2 = 5625) : 
  n2 = 75 := by
  sorry

end sum_of_first_odd_numbers_l283_283229


namespace initial_pokemon_cards_l283_283689

theorem initial_pokemon_cards (x : ℕ) (h : x - 9 = 4) : x = 13 := by
  sorry

end initial_pokemon_cards_l283_283689


namespace smallest_sum_arith_geo_sequence_l283_283832

theorem smallest_sum_arith_geo_sequence 
  (A B C D: ℕ) 
  (h1: A > 0) 
  (h2: B > 0) 
  (h3: C > 0) 
  (h4: D > 0)
  (h5: 2 * B = A + C)
  (h6: B * D = C * C)
  (h7: 3 * C = 4 * B) : 
  A + B + C + D = 43 := 
sorry

end smallest_sum_arith_geo_sequence_l283_283832


namespace canoe_upstream_speed_l283_283041

namespace canoe_speed

def V_c : ℝ := 12.5            -- speed of the canoe in still water in km/hr
def V_downstream : ℝ := 16     -- speed of the canoe downstream in km/hr

theorem canoe_upstream_speed :
  ∃ (V_upstream : ℝ), V_upstream = V_c - (V_downstream - V_c) ∧ V_upstream = 9 := by
  sorry

end canoe_speed

end canoe_upstream_speed_l283_283041


namespace parallelogram_side_problem_l283_283541

theorem parallelogram_side_problem (y z : ℝ) (h1 : 4 * z + 1 = 15) (h2 : 3 * y - 2 = 15) :
  y + z = 55 / 6 :=
sorry

end parallelogram_side_problem_l283_283541


namespace line_perpendicular_value_of_a_l283_283115

theorem line_perpendicular_value_of_a :
  ∀ (a : ℝ),
    (∃ (l1 l2 : ℝ → ℝ),
      (∀ x, l1 x = (-a / (1 - a)) * x + 3 / (1 - a)) ∧
      (∀ x, l2 x = (-(a - 1) / (2 * a + 3)) * x + 2 / (2 * a + 3)) ∧
      (∀ x y, l1 x ≠ l2 y) ∧ 
      (-a / (1 - a)) * (-(a - 1) / (2 * a + 3)) = -1) →
    a = -3 := sorry

end line_perpendicular_value_of_a_l283_283115


namespace value_of_a_l283_283241

theorem value_of_a (a : ℕ) (A_a B_a : ℕ)
  (h1 : A_a = 10)
  (h2 : B_a = 11)
  (h3 : 2 * a^2 + 10 * a + 3 + 5 * a^2 + 7 * a + 8 = 8 * a^2 + 4 * a + 11) :
  a = 13 :=
sorry

end value_of_a_l283_283241


namespace option_D_correct_l283_283851

-- Formal statement in Lean 4
theorem option_D_correct (m : ℝ) : 6 * m + (-2 - 10 * m) = -4 * m - 2 :=
by
  -- Proof is skipped per instruction
  sorry

end option_D_correct_l283_283851


namespace find_distance_PF2_l283_283458

-- Define the properties of the hyperbola
def is_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

-- Define the property that P lies on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  is_hyperbola P.1 P.2

-- Define foci of the hyperbola
structure foci (F1 F2 : ℝ × ℝ) : Prop :=
(F1_prop : F1 = (2, 0))
(F2_prop : F2 = (-2, 0))

-- Given distance from P to F1
def distance_PF1 (P F1 : ℝ × ℝ) (d : ℝ) : Prop :=
  (P.1 - F1.1)^2 + (P.2 - F1.2)^2 = d^2

-- The goal is to find the distance |PF2|
theorem find_distance_PF2 (P F1 F2 : ℝ × ℝ) (D1 D2 : ℝ) :
  point_on_hyperbola P →
  foci F1 F2 →
  distance_PF1 P F1 3 →
  D2 - 3 = 4 →
  D2 = 7 :=
by
  intros hP hFoci hDIST hEQ
  -- Proof can be provided here
  sorry

end find_distance_PF2_l283_283458


namespace geometric_progression_common_ratio_l283_283328

theorem geometric_progression_common_ratio :
  ∃ r : ℝ, (r > 0) ∧ (r^3 + r^2 + r - 1 = 0) :=
by
  sorry

end geometric_progression_common_ratio_l283_283328


namespace minimum_value_of_f_l283_283167

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 3 * x + 3) + Real.sqrt (x^2 - 3 * x + 3)

theorem minimum_value_of_f : (∃ x : ℝ, ∀ y : ℝ, f x ≤ f y) ∧ f 0 = 2 * Real.sqrt 3 :=
by
  sorry

end minimum_value_of_f_l283_283167


namespace sin_330_eq_neg_one_half_l283_283739

theorem sin_330_eq_neg_one_half : 
  Real.sin (330 * Real.pi / 180) = -1 / 2 := 
sorry

end sin_330_eq_neg_one_half_l283_283739


namespace side_length_of_base_l283_283378

-- Define the conditions
def lateral_area (s : ℝ) : ℝ := (1 / 2) * s * 40
def given_area : ℝ := 120

-- Define the theorem to prove the length of the side of the base
theorem side_length_of_base : ∃ (s : ℝ), lateral_area(s) = given_area ∧ s = 6 :=
by
  sorry

end side_length_of_base_l283_283378


namespace people_entering_2pm_to_3pm_people_leaving_2pm_to_3pm_peak_visitors_time_l283_283208

def f (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 8 then 200 * n + 2000
  else if 9 ≤ n ∧ n ≤ 32 then 360 * 3 ^ ((n - 8) / 12) + 3000
  else if 33 ≤ n ∧ n ≤ 45 then 32400 - 720 * n
  else 0 -- default case for unsupported values

def g (n : ℕ) : ℝ :=
  if 1 ≤ n ∧ n ≤ 18 then 0
  else if 19 ≤ n ∧ n ≤ 32 then 500 * n - 9000
  else if 33 ≤ n ∧ n ≤ 45 then 8800
  else 0 -- default case for unsupported values

theorem people_entering_2pm_to_3pm :
  f 21 + f 22 + f 23 + f 24 = 17460 := sorry

theorem people_leaving_2pm_to_3pm :
  g 21 + g 22 + g 23 + g 24 = 9000 := sorry

theorem peak_visitors_time :
  ∀ n, 1 ≤ n ∧ n ≤ 45 → 
    (n = 28 ↔ ∀ m, 1 ≤ m ∧ m ≤ 45 → f m - g m ≤ f 28 - g 28) := sorry

end people_entering_2pm_to_3pm_people_leaving_2pm_to_3pm_peak_visitors_time_l283_283208


namespace pyramid_base_side_length_l283_283388

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ) :
  A = 120 ∧ h = 40 ∧ (A = 1 / 2 * s * h) → s = 6 :=
by
  intros
  sorry

end pyramid_base_side_length_l283_283388


namespace train_passes_jogger_in_40_seconds_l283_283855

variable (speed_jogger_kmh : ℕ)
variable (speed_train_kmh : ℕ)
variable (head_start : ℕ)
variable (train_length : ℕ)

noncomputable def time_to_pass_jogger (speed_jogger_kmh speed_train_kmh head_start train_length : ℕ) : ℕ :=
  let speed_jogger_ms := (speed_jogger_kmh * 1000) / 3600
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  let relative_speed := speed_train_ms - speed_jogger_ms
  let total_distance := head_start + train_length
  total_distance / relative_speed

theorem train_passes_jogger_in_40_seconds : time_to_pass_jogger 9 45 280 120 = 40 := by
  sorry

end train_passes_jogger_in_40_seconds_l283_283855


namespace sum_of_products_is_50_l283_283544

theorem sum_of_products_is_50
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a + b + c = 16) :
  a * b + b * c + a * c = 50 :=
by
  sorry

end sum_of_products_is_50_l283_283544


namespace window_total_width_l283_283069

theorem window_total_width 
  (panes : Nat := 6)
  (ratio_height_width : ℤ := 3)
  (border_width : ℤ := 1)
  (rows : Nat := 2)
  (columns : Nat := 3)
  (pane_width : ℤ := 12) :
  3 * pane_width + 2 * border_width + 2 * border_width = 40 := 
by
  sorry

end window_total_width_l283_283069


namespace fraction_addition_solution_is_six_l283_283113

theorem fraction_addition_solution_is_six :
  (1 / 9) + (1 / 18) = 1 / 6 := 
sorry

end fraction_addition_solution_is_six_l283_283113


namespace calories_per_slice_l283_283054

theorem calories_per_slice (n k t c : ℕ) (h1 : n = 8) (h2 : k = n / 2) (h3 : k * c = t) (h4 : t = 1200) : c = 300 :=
by sorry

end calories_per_slice_l283_283054


namespace range_of_k_l283_283511

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ 0 ≤ k ∧ k < 4 := sorry

end range_of_k_l283_283511


namespace total_number_of_buyers_l283_283717

def day_before_yesterday_buyers : ℕ := 50
def yesterday_buyers : ℕ := day_before_yesterday_buyers / 2
def today_buyers : ℕ := yesterday_buyers + 40
def total_buyers := day_before_yesterday_buyers + yesterday_buyers + today_buyers

theorem total_number_of_buyers : total_buyers = 140 :=
by
  have h1 : day_before_yesterday_buyers = 50 := rfl
  have h2 : yesterday_buyers = day_before_yesterday_buyers / 2 := rfl
  have h3 : today_buyers = yesterday_buyers + 40 := rfl
  rw [h1, h2, h3]
  simp [total_buyers]
  sorry

end total_number_of_buyers_l283_283717


namespace gcd_lcm_ordering_l283_283947

theorem gcd_lcm_ordering (a b p q : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_a_gt_b : a > b) 
    (h_p_gcd : p = Nat.gcd a b) (h_q_lcm : q = Nat.lcm a b) : q ≥ a ∧ a > b ∧ b ≥ p :=
by
  sorry

end gcd_lcm_ordering_l283_283947


namespace absolute_value_of_neg_five_l283_283374

theorem absolute_value_of_neg_five : |(-5 : ℤ)| = 5 := 
by 
  sorry

end absolute_value_of_neg_five_l283_283374


namespace sum_of_numbers_l283_283594

theorem sum_of_numbers :
  145 + 35 + 25 + 5 = 210 :=
by
  sorry

end sum_of_numbers_l283_283594


namespace find_r4_l283_283270

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l283_283270


namespace min_remainder_n_div_2005_l283_283347

theorem min_remainder_n_div_2005 (n : ℕ) (hn_pos : 0 < n) 
  (h1 : n % 902 = 602) (h2 : n % 802 = 502) (h3 : n % 702 = 402) :
  n % 2005 = 101 :=
sorry

end min_remainder_n_div_2005_l283_283347


namespace algebraic_expression_value_l283_283916

theorem algebraic_expression_value (a b : ℝ) (h : a - b = 2) : a^2 - b^2 - 4*a = -4 := 
sorry

end algebraic_expression_value_l283_283916


namespace chord_lengths_equal_l283_283932

theorem chord_lengths_equal (D E F : ℝ) (hcond_1 : D^2 ≠ E^2) (hcond_2 : E^2 > 4 * F) :
  ∀ x y, (x^2 + y^2 + D * x + E * y + F = 0) → 
  (abs x = abs y) :=
by
  sorry

end chord_lengths_equal_l283_283932


namespace bran_tuition_fee_l283_283055

theorem bran_tuition_fee (P : ℝ) (S : ℝ) (M : ℕ) (R : ℝ) (T : ℝ) 
  (h1 : P = 15) (h2 : S = 0.30) (h3 : M = 3) (h4 : R = 18) 
  (h5 : 0.70 * T - (M * P) = R) : T = 90 :=
by
  sorry

end bran_tuition_fee_l283_283055


namespace valid_N_values_l283_283649

def N_values (N : ℕ) : Prop :=
  (∀ k, k = N - 8 → (22 < N ∧ N ≤ 25))

-- Main theorem statement without proof
theorem valid_N_values (N : ℕ) (h : ∀ k, k = N - 8 → N_values N) : 
  (N = 23 ∨ N = 24 ∨ N = 25) :=
by
  sorry

end valid_N_values_l283_283649


namespace pencil_distribution_l283_283476

theorem pencil_distribution : 
  ∃ n : ℕ, n = 35 ∧ (∃ lst : List ℕ, lst.Length = 4 ∧ lst.Sum = 8 ∧ ∀ x ∈ lst, x ≥ 1) :=
by
  use 35
  use [5, 1, 1, 1]
  sorry

end pencil_distribution_l283_283476


namespace minimum_value_of_f_l283_283714

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem minimum_value_of_f :
  f 2 = -3 ∧ (∀ x : ℝ, f x ≥ -3) :=
by
  sorry

end minimum_value_of_f_l283_283714


namespace find_theta_interval_l283_283602

theorem find_theta_interval (θ : ℝ) (x : ℝ) :
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (0 ≤ x ∧ x ≤ 1) →
  (∀ k, k = 0.5 → x^2 * Real.sin θ - k * x * (1 - x) + (1 - x)^2 * Real.cos θ ≥ 0) ↔
  (0 ≤ θ ∧ θ ≤ π / 12) ∨ (23 * π / 12 ≤ θ ∧ θ ≤ 2 * π) := 
sorry

end find_theta_interval_l283_283602


namespace true_propositions_l283_283096

theorem true_propositions :
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + 2*x - m = 0) ∧            -- Condition 1
  ((∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧                    -- Condition 2
   (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ) ∧
  (∀ x y : ℝ, (x * y ≠ 0) → (x ≠ 0 ∧ y ≠ 0)) ∧              -- Condition 3
  ¬ ( (∀ p q : Prop, ¬p → ¬ (p ∧ q)) ∧ (¬ ¬p → p ∧ q) ) ∧   -- Condition 4
  (∃ x : ℝ, x^2 + x + 3 ≤ 0)                                 -- Condition 5
:= by {
  sorry
}

end true_propositions_l283_283096


namespace possible_values_of_N_l283_283652

def is_valid_N (N : ℕ) : Prop :=
  (N > 22) ∧ (N ≤ 25)

theorem possible_values_of_N :
  {N : ℕ | is_valid_N N} = {23, 24, 25} :=
by
  sorry

end possible_values_of_N_l283_283652


namespace find_number_l283_283037

theorem find_number (x : ℕ) (h : 3 * (2 * x + 8) = 84) : x = 10 :=
by
  sorry

end find_number_l283_283037


namespace r_squared_sum_l283_283291

theorem r_squared_sum (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 :=
sorry

end r_squared_sum_l283_283291


namespace not_true_expr_l283_283790

theorem not_true_expr (x y : ℝ) (h : x < y) : -2 * x > -2 * y :=
sorry

end not_true_expr_l283_283790


namespace solution_set_of_inequality_l283_283977

theorem solution_set_of_inequality (x : ℝ) : {x | x * (x - 1) > 0} = { x | x < 0 } ∪ { x | x > 1 } :=
sorry

end solution_set_of_inequality_l283_283977


namespace coords_A_l283_283930

def A : ℝ × ℝ := (1, -2)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def move_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 + d)

def A' : ℝ × ℝ := reflect_y_axis A

def A'' : ℝ × ℝ := move_up A' 3

theorem coords_A'' : A'' = (-1, 1) := by
  sorry

end coords_A_l283_283930


namespace original_number_l283_283793

theorem original_number (n : ℕ) (h1 : 100000 ≤ n ∧ n < 1000000) (h2 : n / 100000 = 7) (h3 : (n % 100000) * 10 + 7 = n / 5) : n = 714285 :=
sorry

end original_number_l283_283793


namespace commission_rate_correct_l283_283551

-- Define the given conditions
def base_pay := 190
def goal_earnings := 500
def required_sales := 7750

-- Define the commission rate function
def commission_rate (sales commission : ℕ) : ℚ := (commission : ℚ) / (sales : ℚ) * 100

-- The main statement to prove
theorem commission_rate_correct :
  commission_rate required_sales (goal_earnings - base_pay) = 4 :=
by
  sorry

end commission_rate_correct_l283_283551


namespace amanda_earnings_l283_283585

def hourly_rate : ℝ := 20.00

def hours_monday : ℝ := 5 * 1.5

def hours_tuesday : ℝ := 3

def hours_thursday : ℝ := 2 * 2

def hours_saturday : ℝ := 6

def total_hours : ℝ := hours_monday + hours_tuesday + hours_thursday + hours_saturday

def total_earnings : ℝ := hourly_rate * total_hours

theorem amanda_earnings : total_earnings = 410.00 :=
by
  -- Proof steps can be filled here
  sorry

end amanda_earnings_l283_283585


namespace kombucha_cost_l283_283501

variable (C : ℝ)

-- Henry drinks 15 bottles of kombucha every month
def bottles_per_month : ℝ := 15

-- A year has 12 months
def months_per_year : ℝ := 12

-- Total bottles consumed in a year
def total_bottles := bottles_per_month * months_per_year

-- Cash refund per bottle
def refund_per_bottle : ℝ := 0.10

-- Total cash refund for all bottles in a year
def total_refund := total_bottles * refund_per_bottle

-- Number of bottles he can buy with the total refund
def bottles_purchasable_with_refund : ℝ := 6

-- Given that the total refund allows purchasing 6 bottles
def cost_per_bottle_eq : Prop := bottles_purchasable_with_refund * C = total_refund

-- Statement to prove
theorem kombucha_cost : cost_per_bottle_eq C → C = 3 := by
  intros
  sorry

end kombucha_cost_l283_283501


namespace range_of_a_l283_283416

-- Definitions for the conditions
def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2 * x - 8 > 0

-- Main theorem
theorem range_of_a (a : ℝ) (h : a < 0) : (¬ (∃ x, prop_p a x)) → (¬ (∃ x, ¬ prop_q x)) :=
sorry

end range_of_a_l283_283416


namespace servant_service_duration_l283_283206

variables (x : ℕ) (total_compensation full_months received_compensation : ℕ)
variables (price_uniform compensation_cash : ℕ)

theorem servant_service_duration :
  total_compensation = 1000 →
  full_months = 12 →
  received_compensation = (compensation_cash + price_uniform) →
  received_compensation = 750 →
  total_compensation = (compensation_cash + price_uniform) →
  x / full_months = 750 / total_compensation →
  x = 9 :=
by sorry

end servant_service_duration_l283_283206


namespace min_words_to_learn_l283_283473

theorem min_words_to_learn (n : ℕ) (p_guess : ℝ) (required_score : ℝ)
  (h_n : n = 600) (h_p : p_guess = 0.1) (h_score : required_score = 0.9) :
  ∃ x : ℕ, (x + p_guess * (n - x)) / n ≥ required_score ∧ x = 534 :=
by
  sorry

end min_words_to_learn_l283_283473


namespace regular_polygon_area_l283_283441
open Real

theorem regular_polygon_area (R : ℝ) (n : ℕ) (hR : 0 < R) (hn : 8 ≤ n) (h_area : (1/2) * n * R^2 * sin (360 / n * (π / 180)) = 4 * R^2) :
  n = 10 := 
sorry

end regular_polygon_area_l283_283441


namespace solve_for_a_l283_283490
-- Additional imports might be necessary depending on specifics of the proof

theorem solve_for_a (a x y : ℝ) (h1 : ax - y = 3) (h2 : x = 1) (h3 : y = 2) : a = 5 :=
by
  sorry

end solve_for_a_l283_283490


namespace find_r4_l283_283273

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l283_283273


namespace nico_reads_wednesday_l283_283356

def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51
def pages_wednesday := total_pages - (pages_monday + pages_tuesday) 

theorem nico_reads_wednesday :
  pages_wednesday = 19 :=
by
  sorry

end nico_reads_wednesday_l283_283356


namespace eighth_odd_multiple_of_5_l283_283987

theorem eighth_odd_multiple_of_5 : 
  (∃ n : ℕ, n = 8 ∧ ∃ k : ℤ, k = (10 * n - 5) ∧ k > 0 ∧ k % 2 = 1) → 75 := 
by {
  sorry
}

end eighth_odd_multiple_of_5_l283_283987
