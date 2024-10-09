import Mathlib

namespace double_root_values_l1509_150929

theorem double_root_values (c : ℝ) :
  (∃ a : ℝ, (a^5 - 5 * a + c = 0) ∧ (5 * a^4 - 5 = 0)) ↔ (c = 4 ∨ c = -4) :=
by
  sorry

end double_root_values_l1509_150929


namespace gcd_of_sum_of_cubes_and_increment_l1509_150927

theorem gcd_of_sum_of_cubes_and_increment {n : ℕ} (h : n > 3) : Nat.gcd (n^3 + 27) (n + 4) = 1 :=
by sorry

end gcd_of_sum_of_cubes_and_increment_l1509_150927


namespace lcm_of_two_numbers_l1509_150938

-- Define the given conditions: Two numbers a and b, their HCF, and their product.
variables (a b : ℕ)
def hcf : ℕ := 55
def product := 82500

-- Define the concept of HCF and LCM, using the provided relationship in the problem
def gcd_ab := hcf
def lcm_ab := (product / gcd_ab)

-- State the main theorem to prove: The LCM of the two numbers is 1500
theorem lcm_of_two_numbers : lcm_ab = 1500 := by
  -- This is the place where the actual proof steps would go
  sorry

end lcm_of_two_numbers_l1509_150938


namespace partition_diff_l1509_150903

theorem partition_diff {A : Type} (S : Finset ℕ) (S_card : S.card = 67)
  (P : Finset (Finset ℕ)) (P_card : P.card = 4) :
  ∃ (U : Finset ℕ) (hU : U ∈ P), ∃ (a b c : ℕ) (ha : a ∈ U) (hb : b ∈ U) (hc : c ∈ U),
  a = b - c ∧ (1 ≤ a ∧ a ≤ 67) :=
by sorry

end partition_diff_l1509_150903


namespace min_value_a_b_c_l1509_150942

def A_n (a : ℕ) (n : ℕ) : ℕ := a * ((10^n - 1) / 9)
def B_n (b : ℕ) (n : ℕ) : ℕ := b * ((10^n - 1) / 9)
def C_n (c : ℕ) (n : ℕ) : ℕ := c * ((10^(2*n) - 1) / 9)

theorem min_value_a_b_c (a b c : ℕ) (Ha : 0 < a ∧ a < 10) (Hb : 0 < b ∧ b < 10) (Hc : 0 < c ∧ c < 10) :
  (∃ n1 n2 : ℕ, (n1 ≠ n2) ∧ (C_n c n1 - A_n a n1 = B_n b n1 ^ 2) ∧ (C_n c n2 - A_n a n2 = B_n b n2 ^ 2)) →
  a + b + c = 5 :=
by
  sorry

end min_value_a_b_c_l1509_150942


namespace phi_value_l1509_150980

theorem phi_value (phi : ℝ) (h : 0 < phi ∧ phi < π) 
  (hf : ∀ x : ℝ, 3 * Real.sin (2 * abs x - π / 3 + phi) = 3 * Real.sin (2 * x - π / 3 + phi)) 
  : φ = 5 * π / 6 :=
by 
  sorry

end phi_value_l1509_150980


namespace scrooge_share_l1509_150954

def leftover_pie : ℚ := 8 / 9

def share_each (x : ℚ) : Prop :=
  2 * x + 3 * x = leftover_pie

theorem scrooge_share (x : ℚ):
  share_each x → (2 * x = 16 / 45) := by
  sorry

end scrooge_share_l1509_150954


namespace find_4digit_number_l1509_150981

theorem find_4digit_number (a b c d n n' : ℕ) :
  n = 1000 * a + 100 * b + 10 * c + d →
  n' = 1000 * d + 100 * c + 10 * b + a →
  n = n' - 7182 →
  n = 1909 :=
by
  intros h1 h2 h3
  sorry

end find_4digit_number_l1509_150981


namespace purchase_price_mobile_l1509_150955

-- Definitions of the given conditions
def purchase_price_refrigerator : ℝ := 15000
def loss_percent_refrigerator : ℝ := 0.05
def profit_percent_mobile : ℝ := 0.10
def overall_profit : ℝ := 50

-- Defining the statement to prove
theorem purchase_price_mobile (P : ℝ)
  (h1 : purchase_price_refrigerator = 15000)
  (h2 : loss_percent_refrigerator = 0.05)
  (h3 : profit_percent_mobile = 0.10)
  (h4 : overall_profit = 50) :
  (15000 * (1 - 0.05) + P * (1 + 0.10)) - (15000 + P) = 50 → P = 8000 :=
by {
  -- Proof is omitted
  sorry
}

end purchase_price_mobile_l1509_150955


namespace monotonicity_f_on_interval_l1509_150975

def f (x : ℝ) : ℝ := |x + 2|

theorem monotonicity_f_on_interval :
  ∀ x1 x2 : ℝ, x1 < x2 → x1 < -4 → x2 < -4 → f x1 ≥ f x2 :=
by
  sorry

end monotonicity_f_on_interval_l1509_150975


namespace computer_price_increase_l1509_150974

theorem computer_price_increase
  (P : ℝ)
  (h1 : 1.30 * P = 351) :
  (P + 1.30 * P) / P = 2.3 := by
  sorry

end computer_price_increase_l1509_150974


namespace tangents_quadrilateral_cyclic_l1509_150973

variables {A B C D K L O1 O2 : Point}
variable (r : ℝ)
variable (AB_cut_circles : ∀ {A B : Point} {O1 O2 : Point}, is_intersect AB O1 O2)
variable (parallel_AB_O1O2 : is_parallel AB O1O2)
variable (tangents_formed_quadrilateral : is_quadrilateral C D K L)
variable (quadrilateral_contains_circles : contains C D K L O1 O2)

theorem tangents_quadrilateral_cyclic
  (h1: AB_cut_circles)
  (h2: parallel_AB_O1O2) 
  (h3: tangents_formed_quadrilateral)
  (h4: quadrilateral_contains_circles)
  : ∃ O : Circle, is_inscribed O C D K L :=
sorry

end tangents_quadrilateral_cyclic_l1509_150973


namespace candy_count_correct_l1509_150922

-- Define initial count of candy
def initial_candy : ℕ := 47

-- Define number of pieces of candy eaten
def eaten_candy : ℕ := 25

-- Define number of pieces of candy received
def received_candy : ℕ := 40

-- The final count of candy is what we are proving
theorem candy_count_correct : initial_candy - eaten_candy + received_candy = 62 :=
by
  sorry

end candy_count_correct_l1509_150922


namespace range_of_m_l1509_150952

/-- Given the conditions:
- \( \left|1 - \frac{x - 2}{3}\right| \leq 2 \)
- \( x^2 - 2x + 1 - m^2 \leq 0 \) where \( m > 0 \)
- \( \neg \left( \left|1 - \frac{x - 2}{3}\right| \leq 2 \right) \) is a necessary but not sufficient condition for \( x^2 - 2x + 1 - m^2 \leq 0 \)

Prove that the range of \( m \) is \( m \geq 10 \).
-/
theorem range_of_m (m : ℝ) (x : ℝ)
  (h1 : ∀ x, ¬(abs (1 - (x - 2) / 3) ≤ 2) → x < -1 ∨ x > 11)
  (h2 : ∀ x, ∀ m > 0, x^2 - 2 * x + 1 - m^2 ≤ 0)
  : m ≥ 10 :=
sorry

end range_of_m_l1509_150952


namespace constant_term_in_expansion_l1509_150982

noncomputable def P (x : ℕ) : ℕ := x^4 + 2 * x + 7
noncomputable def Q (x : ℕ) : ℕ := 2 * x^3 + 3 * x^2 + 10

theorem constant_term_in_expansion :
  (P 0) * (Q 0) = 70 := 
sorry

end constant_term_in_expansion_l1509_150982


namespace find_common_difference_l1509_150987

def is_arithmetic_sequence (a : (ℕ → ℝ)) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def is_arithmetic_sequence_with_sum (a : (ℕ → ℝ)) (S : (ℕ → ℝ)) (d : ℝ) : Prop :=
  S 0 = a 0 ∧
  ∀ n, S (n + 1) = S n + a (n + 1) ∧
        ∀ n, (S (n + 1) / a (n + 1) - S n / a n) = d

theorem find_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a →
  is_arithmetic_sequence_with_sum a S d →
  (d = 1 ∨ d = 1 / 2) :=
sorry

end find_common_difference_l1509_150987


namespace find_PB_l1509_150959

noncomputable def PA : ℝ := 5
noncomputable def PT (AB : ℝ) : ℝ := 2 * (AB - PA) + 1
noncomputable def PB (AB : ℝ) : ℝ := PA + AB

theorem find_PB (AB : ℝ) (AB_condition : AB = PB AB - PA) :
  PB AB = (81 + Real.sqrt 5117) / 8 :=
by
  sorry

end find_PB_l1509_150959


namespace find_number_l1509_150913

-- Define the condition
def condition : Prop := ∃ x : ℝ, x / 0.02 = 50

-- State the theorem to prove
theorem find_number (x : ℝ) (h : x / 0.02 = 50) : x = 1 :=
sorry

end find_number_l1509_150913


namespace no_solution_range_of_a_l1509_150968

theorem no_solution_range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (|x - 5| + |x + 3| < a)) → a ≤ 8 :=
by
  sorry

end no_solution_range_of_a_l1509_150968


namespace union_complement_A_B_eq_U_l1509_150985

-- Define the universal set U, set A, and set B
def U : Set ℕ := {1, 2, 3, 4, 5, 7}
def A : Set ℕ := {4, 7}
def B : Set ℕ := {1, 3, 4, 7}

-- Define the complement of A with respect to U (C_U A)
def C_U_A : Set ℕ := U \ A
-- Define the complement of B with respect to U (C_U B)
def C_U_B : Set ℕ := U \ B

-- The theorem to prove
theorem union_complement_A_B_eq_U : (C_U_A ∪ B) = U := by
  sorry

end union_complement_A_B_eq_U_l1509_150985


namespace quadratic_complete_square_l1509_150993

theorem quadratic_complete_square (x : ℝ) (m t : ℝ) :
  (4 * x^2 - 16 * x - 448 = 0) → ((x + m) ^ 2 = t) → (t = 116) :=
by
  sorry

end quadratic_complete_square_l1509_150993


namespace find_multiple_of_games_l1509_150916

-- declaring the number of video games each person has
def Tory_videos := 6
def Theresa_videos := 11
def Julia_videos := Tory_videos / 3

-- declaring the multiple we need to find
def multiple_of_games := Theresa_videos - Julia_videos * 5

-- Theorem stating the problem
theorem find_multiple_of_games : ∃ m : ℕ, Julia_videos * m + 5 = Theresa_videos :=
by
  sorry

end find_multiple_of_games_l1509_150916


namespace find_line_equation_l1509_150908

theorem find_line_equation : 
  ∃ c : ℝ, (∀ x y : ℝ, 2*x + 4*y + c = 0 ↔ x + 2*y - 8 = 0) ∧ (2*2 + 4*3 + c = 0) :=
sorry

end find_line_equation_l1509_150908


namespace set_of_integers_between_10_and_16_l1509_150930

theorem set_of_integers_between_10_and_16 :
  {x : ℤ | 10 < x ∧ x < 16} = {11, 12, 13, 14, 15} :=
by
  sorry

end set_of_integers_between_10_and_16_l1509_150930


namespace number_of_buses_l1509_150902

theorem number_of_buses (total_supervisors : ℕ) (supervisors_per_bus : ℕ) (h1 : total_supervisors = 21) (h2 : supervisors_per_bus = 3) : total_supervisors / supervisors_per_bus = 7 :=
by
  sorry

end number_of_buses_l1509_150902


namespace volume_of_right_prism_correct_l1509_150956

variables {α β l : ℝ}

noncomputable def volume_of_right_prism (α β l : ℝ) : ℝ :=
  (1 / 4) * l^3 * (Real.tan β)^2 * (Real.sin (2 * α))

theorem volume_of_right_prism_correct
  (α β l : ℝ)
  (α_gt0 : 0 < α) (α_lt90 : α < Real.pi / 2)
  (l_pos : 0 < l)
  : volume_of_right_prism α β l = (1 / 4) * l^3 * (Real.tan β)^2 * (Real.sin (2 * α)) :=
sorry

end volume_of_right_prism_correct_l1509_150956


namespace probability_participation_on_both_days_l1509_150940

-- Definitions based on conditions
def total_students := 5
def total_combinations := 2^total_students
def same_day_scenarios := 2
def favorable_outcomes := total_combinations - same_day_scenarios

-- Theorem statement
theorem probability_participation_on_both_days :
  (favorable_outcomes / total_combinations : ℚ) = 15 / 16 :=
by
  sorry

end probability_participation_on_both_days_l1509_150940


namespace remainder_3001_3005_mod_23_l1509_150997

theorem remainder_3001_3005_mod_23 : 
  (3001 * 3002 * 3003 * 3004 * 3005) % 23 = 9 :=
by {
  sorry
}

end remainder_3001_3005_mod_23_l1509_150997


namespace final_point_P_after_transformations_l1509_150970

noncomputable def point := (ℝ × ℝ)

def rotate_90_clockwise (p : point) : point :=
  (-p.2, p.1)

def reflect_across_x (p : point) : point :=
  (p.1, -p.2)

def P : point := (3, -5)

def Q : point := (5, -2)

def R : point := (5, -5)

theorem final_point_P_after_transformations : reflect_across_x (rotate_90_clockwise P) = (-5, 3) :=
by 
  sorry

end final_point_P_after_transformations_l1509_150970


namespace min_value_l1509_150921

theorem min_value (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) (h_sum : x1 + x2 = 1) :
  ∃ m, (∀ x1 x2, x1 > 0 ∧ x2 > 0 ∧ x1 + x2 = 1 → (3 * x1 / x2 + 1 / (x1 * x2)) ≥ m) ∧ m = 6 :=
by
  sorry

end min_value_l1509_150921


namespace number_of_possible_lengths_of_diagonal_l1509_150988

theorem number_of_possible_lengths_of_diagonal :
  ∃ n : ℕ, n = 13 ∧
  (∀ y : ℕ, (5 ≤ y ∧ y ≤ 17) ↔ (y = 5 ∨ y = 6 ∨ y = 7 ∨ y = 8 ∨ y = 9 ∨
   y = 10 ∨ y = 11 ∨ y = 12 ∨ y = 13 ∨ y = 14 ∨ y = 15 ∨ y = 16 ∨ y = 17)) :=
by
  exists 13
  sorry

end number_of_possible_lengths_of_diagonal_l1509_150988


namespace mary_cut_roses_l1509_150918

-- Definitions from conditions
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- The theorem to prove
theorem mary_cut_roses : (final_roses - initial_roses) = 10 :=
by
  sorry

end mary_cut_roses_l1509_150918


namespace hundredth_odd_integer_l1509_150958

theorem hundredth_odd_integer : (2 * 100 - 1) = 199 := 
by
  sorry

end hundredth_odd_integer_l1509_150958


namespace find_z_coordinate_of_point_on_line_l1509_150905

theorem find_z_coordinate_of_point_on_line (x1 y1 z1 x2 y2 z2 x_target : ℝ) 
(h1 : x1 = 1) (h2 : y1 = 3) (h3 : z1 = 2) 
(h4 : x2 = 4) (h5 : y2 = 4) (h6 : z2 = -1)
(h_target : x_target = 7) : 
∃ z_target : ℝ, z_target = -4 := 
by {
  sorry
}

end find_z_coordinate_of_point_on_line_l1509_150905


namespace train_passing_platform_time_l1509_150900

theorem train_passing_platform_time :
  (500 : ℝ) / (50 : ℝ) > 0 →
  (500 : ℝ) + (500 : ℝ) / ((500 : ℝ) / (50 : ℝ)) = 100 := by
  sorry

end train_passing_platform_time_l1509_150900


namespace sqrt_expression_l1509_150932

open Real

theorem sqrt_expression :
  3 * sqrt 12 / (3 * sqrt (1 / 3)) - 2 * sqrt 3 = 6 - 2 * sqrt 3 :=
by
  sorry

end sqrt_expression_l1509_150932


namespace total_tweets_is_correct_l1509_150904

-- Define the conditions of Polly's tweeting behavior and durations
def happy_tweets := 18
def hungry_tweets := 4
def mirror_tweets := 45
def duration := 20

-- Define the total tweets calculation
def total_tweets := duration * happy_tweets + duration * hungry_tweets + duration * mirror_tweets

-- Prove that the total number of tweets is 1340
theorem total_tweets_is_correct : total_tweets = 1340 := by
  sorry

end total_tweets_is_correct_l1509_150904


namespace least_n_value_l1509_150963

open Nat

theorem least_n_value (n : ℕ) (h : 1 / (n * (n + 1)) < 1 / 15) : n = 4 :=
sorry

end least_n_value_l1509_150963


namespace solve_for_y_l1509_150947

theorem solve_for_y (y : ℤ) : 7 * (4 * y + 5) - 4 = -3 * (2 - 9 * y) → y = -37 :=
by
  intro h
  sorry

end solve_for_y_l1509_150947


namespace point_slope_form_of_perpendicular_line_l1509_150951

theorem point_slope_form_of_perpendicular_line :
  ∀ (l1 l2 : ℝ → ℝ) (P : ℝ × ℝ),
    (l2 x = x + 1) →
    (P = (2, 1)) →
    (∀ x, l2 x = -1 * l1 x) →
    (∀ x, l1 x = -x + 3) :=
by
  intros l1 l2 P h1 h2 h3
  sorry

end point_slope_form_of_perpendicular_line_l1509_150951


namespace part_I_part_II_l1509_150924

noncomputable def f (x : ℝ) := (Real.sin x) * (Real.cos x) + (Real.sin x)^2

-- Part I: Prove that f(π / 4) = 1
theorem part_I : f (Real.pi / 4) = 1 := sorry

-- Part II: Prove that the maximum value of f(x) for x ∈ [0, π / 2] is (√2 + 1) / 2
theorem part_II : ∃ x ∈ Set.Icc 0 (Real.pi / 2), (∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f x) ∧ f x = (Real.sqrt 2 + 1) / 2 := sorry

end part_I_part_II_l1509_150924


namespace fraction_power_calc_l1509_150919

theorem fraction_power_calc : 
  (0.5 ^ 4) / (0.05 ^ 3) = 500 := 
sorry

end fraction_power_calc_l1509_150919


namespace min_x2_y2_z2_l1509_150906

theorem min_x2_y2_z2 (x y z : ℝ) (h : 2 * x + 3 * y + 3 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 22 :=
by
  sorry

end min_x2_y2_z2_l1509_150906


namespace typists_initial_group_l1509_150941

theorem typists_initial_group
  (T : ℕ) 
  (h1 : 0 < T) 
  (h2 : T * (240 / 40 * 20) = 2400) : T = 10 :=
by
  sorry

end typists_initial_group_l1509_150941


namespace least_number_to_add_l1509_150909

theorem least_number_to_add (n d : ℕ) (h : n = 1024) (h_d : d = 25) :
  ∃ x : ℕ, (n + x) % d = 0 ∧ x = 1 :=
by sorry

end least_number_to_add_l1509_150909


namespace total_surface_area_of_resulting_structure_l1509_150964

-- Definitions for the conditions
def bigCube := 12 * 12 * 12
def smallCube := 2 * 2 * 2
def totalSmallCubes := 64
def removedCubes := 7
def remainingCubes := totalSmallCubes - removedCubes
def surfaceAreaPerSmallCube := 24
def extraExposedSurfaceArea := 6
def effectiveSurfaceAreaPerSmallCube := surfaceAreaPerSmallCube + extraExposedSurfaceArea

-- Definition and the main statement of the proof problem.
def totalSurfaceArea := remainingCubes * effectiveSurfaceAreaPerSmallCube

theorem total_surface_area_of_resulting_structure : totalSurfaceArea = 1710 :=
by
  sorry

end total_surface_area_of_resulting_structure_l1509_150964


namespace minimum_shots_required_l1509_150907

noncomputable def minimum_shots_to_sink_boat : ℕ := 4000

-- Definitions for the problem conditions.
structure Boat :=
(square_side : ℕ)
(base1 : ℕ)
(base2 : ℕ)
(rotatable : Bool)

def boat : Boat := { square_side := 1, base1 := 1, base2 := 3, rotatable := true }

def grid_size : ℕ := 100

def shot_covers_triangular_half : Prop := sorry -- Assumption: Define this appropriately

-- Problem statement in Lean 4
theorem minimum_shots_required (boat_within_grid : Bool) : 
  Boat → grid_size = 100 → boat_within_grid → minimum_shots_to_sink_boat = 4000 :=
by
  -- Here you would do the full proof which we assume is "sorry" for now
  sorry

end minimum_shots_required_l1509_150907


namespace painting_time_equation_l1509_150950

theorem painting_time_equation (t : ℝ) :
  let Doug_rate := (1 : ℝ) / 5
  let Dave_rate := (1 : ℝ) / 7
  let combined_rate := Doug_rate + Dave_rate
  (combined_rate * (t - 1) = 1) :=
sorry

end painting_time_equation_l1509_150950


namespace halfway_between_one_eighth_and_one_third_l1509_150926

theorem halfway_between_one_eighth_and_one_third : (1/8 + 1/3) / 2 = 11/48 :=
by
  sorry

end halfway_between_one_eighth_and_one_third_l1509_150926


namespace paintable_fence_l1509_150948

theorem paintable_fence :
  ∃ h t u : ℕ,  h > 1 ∧ t > 1 ∧ u > 1 ∧ 
  (∀ n, 4 + (n * h) ≠ 5 + (m * (2 * t))) ∧
  (∀ n, 4 + (n * h) ≠ 6 + (l * (3 * u))) ∧ 
  (∀ m l, 5 + (m * (2 * t)) ≠ 6 + (l * (3 * u))) ∧
  (100 * h + 20 * t + 2 * u = 390) :=
by 
  sorry

end paintable_fence_l1509_150948


namespace positive_difference_l1509_150960

theorem positive_difference (y : ℤ) (h : (46 + y) / 2 = 52) : |y - 46| = 12 := by
  sorry

end positive_difference_l1509_150960


namespace reciprocal_fraction_addition_l1509_150994

theorem reciprocal_fraction_addition (a b c : ℝ) (h : a ≠ b) :
  (a + c) / (b + c) = b / a ↔ c = - (a + b) := 
by
  sorry

end reciprocal_fraction_addition_l1509_150994


namespace fraction_traditionalists_l1509_150914

theorem fraction_traditionalists {P T : ℕ} (h1 : ∀ (i : ℕ), i < 5 → T = P / 15) (h2 : T = P / 15) :
  (5 * T : ℚ) / (P + 5 * T : ℚ) = 1 / 4 :=
by
  sorry

end fraction_traditionalists_l1509_150914


namespace lines_intersection_l1509_150966

theorem lines_intersection :
  ∃ (x y : ℝ), 
    (x - y = 0) ∧ (3 * x + 2 * y - 5 = 0) ∧ (x = 1) ∧ (y = 1) :=
by
  sorry

end lines_intersection_l1509_150966


namespace f_of_2_l1509_150995

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem f_of_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end f_of_2_l1509_150995


namespace mt_product_l1509_150967

noncomputable def g (x : ℝ) : ℝ := sorry

theorem mt_product
  (hg : ∀ (x y : ℝ), g (g x + y) = g x + g (g y + g (-x)) - x) : 
  ∃ m t : ℝ, m = 1 ∧ t = -5 ∧ m * t = -5 := 
by
  sorry

end mt_product_l1509_150967


namespace total_carrots_l1509_150920

theorem total_carrots (carrots_sandy carrots_mary : ℕ) (h1 : carrots_sandy = 8) (h2 : carrots_mary = 6) :
  carrots_sandy + carrots_mary = 14 :=
by
  sorry

end total_carrots_l1509_150920


namespace verify_statements_l1509_150925

noncomputable def f (x : ℝ) : ℝ := 10 ^ x

theorem verify_statements (x1 x2 : ℝ) (h : x1 ≠ x2) :
  (f (x1 + x2) = f x1 * f x2) ∧
  (f x1 - f x2) / (x1 - x2) > 0 :=
by
  sorry

end verify_statements_l1509_150925


namespace back_parking_lot_filled_fraction_l1509_150911

theorem back_parking_lot_filled_fraction
    (front_spaces : ℕ) (back_spaces : ℕ) (cars_parked : ℕ) (spaces_available : ℕ)
    (h1 : front_spaces = 52)
    (h2 : back_spaces = 38)
    (h3 : cars_parked = 39)
    (h4 : spaces_available = 32) :
    (back_spaces - (front_spaces + back_spaces - cars_parked - spaces_available)) / back_spaces = 1 / 2 :=
by
  sorry

end back_parking_lot_filled_fraction_l1509_150911


namespace Jill_age_l1509_150910

variable (J R : ℕ) -- representing Jill's current age and Roger's current age

theorem Jill_age :
  (R = 2 * J + 5) →
  (R - J = 25) →
  J = 20 :=
by
  intros h1 h2
  sorry

end Jill_age_l1509_150910


namespace card_drawing_ways_l1509_150978

theorem card_drawing_ways :
  (30 * 20 = 600) :=
by
  sorry

end card_drawing_ways_l1509_150978


namespace intersection_of_sets_l1509_150989

def set_M := { y : ℝ | y ≥ 0 }
def set_N := { y : ℝ | ∃ x : ℝ, y = -x^2 + 1 }

theorem intersection_of_sets : set_M ∩ set_N = { y : ℝ | 0 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end intersection_of_sets_l1509_150989


namespace total_number_of_students_l1509_150984

theorem total_number_of_students (T G : ℕ) (h1 : 50 + G = T) (h2 : G = 50 * T / 100) : T = 100 :=
  sorry

end total_number_of_students_l1509_150984


namespace max_groups_l1509_150965

def eggs : ℕ := 20
def marbles : ℕ := 6
def eggs_per_group : ℕ := 5
def marbles_per_group : ℕ := 2

def groups_of_eggs := eggs / eggs_per_group
def groups_of_marbles := marbles / marbles_per_group

theorem max_groups (h1 : eggs = 20) (h2 : marbles = 6) 
                    (h3 : eggs_per_group = 5) (h4 : marbles_per_group = 2) : 
                    min (groups_of_eggs) (groups_of_marbles) = 3 :=
by
  sorry

end max_groups_l1509_150965


namespace range_f_in_interval_l1509_150977

-- Define the function f and the interval
def f (x : ℝ) (f_deriv_neg1 : ℝ) := x^3 + 2 * x * f_deriv_neg1
def interval := Set.Icc (-2 : ℝ) (3 : ℝ)

-- State the theorem
theorem range_f_in_interval :
  ∃ (f_deriv_neg1 : ℝ),
  (∀ x ∈ interval, f x f_deriv_neg1 ∈ Set.Icc (-4 * Real.sqrt 2) 9) :=
sorry

end range_f_in_interval_l1509_150977


namespace minimum_value_of_f_l1509_150962

-- Define the function
def f (a b x : ℝ) := x^2 + (a + 2) * x + b

-- Condition that ensures the graph is symmetric about x = 1
def symmetric_about_x1 (a : ℝ) : Prop := a + 2 = -2

-- Minimum value of the function f(x) in terms of the constant c
theorem minimum_value_of_f (a b : ℝ) (h : symmetric_about_x1 a) : ∃ c : ℝ, ∀ x : ℝ, f a b x ≥ c :=
by sorry

end minimum_value_of_f_l1509_150962


namespace smaller_circle_circumference_l1509_150937

noncomputable def circumference_of_smaller_circle :=
  let π := Real.pi
  let R := 352 / (2 * π)
  let area_difference := 4313.735577562732
  let R_squared_minus_r_squared := area_difference / π
  let r_squared := R ^ 2 - R_squared_minus_r_squared
  let r := Real.sqrt r_squared
  2 * π * r

theorem smaller_circle_circumference : 
  let circumference_larger := 352
  let area_difference := 4313.735577562732
  circumference_of_smaller_circle = 263.8934 := sorry

end smaller_circle_circumference_l1509_150937


namespace range_of_y_div_x_l1509_150933

theorem range_of_y_div_x (x y : ℝ) (h : x^2 + y^2 + 4*x + 3 = 0) :
  - (Real.sqrt 3) / 3 <= y / x ∧ y / x <= (Real.sqrt 3) / 3 :=
sorry

end range_of_y_div_x_l1509_150933


namespace excircle_inequality_l1509_150998

variables {a b c : ℝ} -- The sides of the triangle

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2 -- Definition of semiperimeter

noncomputable def excircle_distance (p a : ℝ) : ℝ := p - a -- Distance from vertices to tangency points

theorem excircle_inequality (a b c : ℝ) (p : ℝ) 
    (h1 : p = semiperimeter a b c) : 
    (excircle_distance p a) + (excircle_distance p b) > p := 
by
    -- Placeholder for proof
    sorry

end excircle_inequality_l1509_150998


namespace units_digit_8th_group_l1509_150935

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_8th_group (t k : ℕ) (ht : t = 7) (hk : k = 8) : 
  units_digit (t + k) = 5 := 
by
  -- Proof step will go here.
  sorry

end units_digit_8th_group_l1509_150935


namespace complex_power_identity_l1509_150923

theorem complex_power_identity (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 :=
by
  sorry

end complex_power_identity_l1509_150923


namespace cos_five_pi_over_four_l1509_150969

theorem cos_five_pi_over_four : Real.cos (5 * Real.pi / 4) = -1 / Real.sqrt 2 := 
by
  sorry

end cos_five_pi_over_four_l1509_150969


namespace number_of_ways_to_draw_balls_l1509_150979

def draw_balls : ℕ :=
  let balls := 15
  let draws := 4
  balls * (balls - 1) * (balls - 2) * (balls - 3)

theorem number_of_ways_to_draw_balls
  : draw_balls = 32760 :=
by
  sorry

end number_of_ways_to_draw_balls_l1509_150979


namespace stable_table_configurations_l1509_150999

noncomputable def numberOfStableConfigurations (n : ℕ) : ℕ :=
  1 / 3 * (n + 1) * (2 * n ^ 2 + 4 * n + 3)

theorem stable_table_configurations (n : ℕ) (hn : 0 < n) :
  numberOfStableConfigurations n = 
    (1 / 3 * (n + 1) * (2 * n ^ 2 + 4 * n + 3)) :=
by
  sorry

end stable_table_configurations_l1509_150999


namespace solve_a_plus_b_l1509_150946

theorem solve_a_plus_b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : 143 * a + 500 * b = 2001) : a + b = 9 :=
by
  -- Add proof here
  sorry

end solve_a_plus_b_l1509_150946


namespace value_of_expression_l1509_150991

theorem value_of_expression (x1 x2 : ℝ) 
  (h1 : x1 ^ 2 - 3 * x1 - 4 = 0) 
  (h2 : x2 ^ 2 - 3 * x2 - 4 = 0)
  (h3 : x1 + x2 = 3) 
  (h4 : x1 * x2 = -4) : 
  x1 ^ 2 - 4 * x1 - x2 + 2 * x1 * x2 = -7 := by
  sorry

end value_of_expression_l1509_150991


namespace solution_set_of_inequality_l1509_150901

variable {R : Type*} [LinearOrder R] [OrderedAddCommGroup R]

def odd_function (f : R → R) := ∀ x, f (-x) = -f x

def monotonic_increasing_on (f : R → R) (s : Set R) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h_odd : odd_function f)
  (h_mono_inc : monotonic_increasing_on f (Set.Ioi 0))
  (h_f_neg1 : f (-1) = 2) : 
  {x : ℝ | 0 < x ∧ f (x-1) + 2 ≤ 0 } = Set.Ioc 1 2 :=
by
  sorry

end solution_set_of_inequality_l1509_150901


namespace area_rectangle_around_right_triangle_l1509_150945

theorem area_rectangle_around_right_triangle (AB BC : ℕ) (hAB : AB = 5) (hBC : BC = 6) :
    let ADE_area := AB * BC
    ADE_area = 30 := by
  sorry

end area_rectangle_around_right_triangle_l1509_150945


namespace hallway_width_equals_four_l1509_150915

-- Define the conditions: dimensions of the areas and total installed area.
def centralAreaLength : ℝ := 10
def centralAreaWidth : ℝ := 10
def centralArea : ℝ := centralAreaLength * centralAreaWidth

def totalInstalledArea : ℝ := 124
def hallwayLength : ℝ := 6

-- Total area minus central area's area yields hallway's area
def hallwayArea : ℝ := totalInstalledArea - centralArea

-- Statement to prove: the width of the hallway given its area and length.
theorem hallway_width_equals_four :
  (hallwayArea / hallwayLength) = 4 := 
by
  sorry

end hallway_width_equals_four_l1509_150915


namespace tan_half_angle_inequality_l1509_150917

theorem tan_half_angle_inequality (a b c : ℝ) (α β : ℝ)
  (h : a + b < 3 * c)
  (h_tan_identity : Real.tan (α / 2) * Real.tan (β / 2) = (a + b - c) / (a + b + c)) :
  Real.tan (α / 2) * Real.tan (β / 2) < 1 / 2 :=
by
  sorry

end tan_half_angle_inequality_l1509_150917


namespace average_score_in_5_matches_l1509_150972

theorem average_score_in_5_matches 
  (avg1 avg2 : ℕ)
  (total_matches1 total_matches2 : ℕ)
  (h1 : avg1 = 27) 
  (h2 : avg2 = 32)
  (h3 : total_matches1 = 2) 
  (h4 : total_matches2 = 3) 
  : 
  (avg1 * total_matches1 + avg2 * total_matches2) / (total_matches1 + total_matches2) = 30 :=
by 
  sorry

end average_score_in_5_matches_l1509_150972


namespace solve_z_l1509_150931

noncomputable def complex_equation (z : ℂ) := (1 + 3 * Complex.I) * z = Complex.I - 3

theorem solve_z (z : ℂ) (h : complex_equation z) : z = Complex.I :=
by
  sorry

end solve_z_l1509_150931


namespace C_investment_is_20000_l1509_150996

-- Definitions of investments and profits
def A_investment : ℕ := 12000
def B_investment : ℕ := 16000
def total_profit : ℕ := 86400
def C_share_of_profit : ℕ := 36000

-- The proof problem statement
theorem C_investment_is_20000 (X : ℕ) (hA : A_investment = 12000) (hB : B_investment = 16000)
  (h_total_profit : total_profit = 86400) (h_C_share_of_profit : C_share_of_profit = 36000) :
  X = 20000 :=
sorry

end C_investment_is_20000_l1509_150996


namespace quadratic_solution_l1509_150928

theorem quadratic_solution (x : ℝ) : (x - 1)^2 = 4 ↔ (x = 3 ∨ x = -1) :=
sorry

end quadratic_solution_l1509_150928


namespace find_constants_l1509_150986

open BigOperators

theorem find_constants (a b c : ℕ) :
  (∀ n : ℕ, n > 0 → (∑ k in Finset.range n, k.succ * (k.succ + 1) ^ 2) = (n * (n + 1) * (a * n^2 + b * n + c)) / 12) →
  (a = 3 ∧ b = 11 ∧ c = 10) :=
by
  sorry

end find_constants_l1509_150986


namespace problem_min_value_l1509_150976

noncomputable def min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : ℝ :=
  1 / x^2 + 1 / y^2 + 1 / (x * y)

theorem problem_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  min_value x y hx hy hxy = 3 := 
sorry

end problem_min_value_l1509_150976


namespace larry_spent_on_lunch_l1509_150912

noncomputable def starting_amount : ℕ := 22
noncomputable def ending_amount : ℕ := 15
noncomputable def amount_given_to_brother : ℕ := 2

theorem larry_spent_on_lunch : 
  (starting_amount - (ending_amount + amount_given_to_brother)) = 5 :=
by
  -- The conditions and the proof structure would be elaborated here
  sorry

end larry_spent_on_lunch_l1509_150912


namespace distance_from_Bangalore_l1509_150949

noncomputable def calculate_distance (speed : ℕ) (start_hour start_minute end_hour end_minute halt_minutes : ℕ) : ℕ :=
  let total_travel_minutes := (end_hour * 60 + end_minute) - (start_hour * 60 + start_minute) - halt_minutes
  let total_travel_hours := total_travel_minutes / 60
  speed * total_travel_hours

theorem distance_from_Bangalore (speed : ℕ) (start_hour start_minute end_hour end_minute halt_minutes : ℕ) :
  speed = 87 ∧ start_hour = 9 ∧ start_minute = 0 ∧ end_hour = 13 ∧ end_minute = 45 ∧ halt_minutes = 45 →
  calculate_distance speed start_hour start_minute end_hour end_minute halt_minutes = 348 := by
  sorry

end distance_from_Bangalore_l1509_150949


namespace bags_on_wednesday_l1509_150961

theorem bags_on_wednesday (charge_per_bag money_per_bag monday_bags tuesday_bags total_money wednesday_bags : ℕ)
  (h_charge : charge_per_bag = 4)
  (h_money_per_dollar : money_per_bag = charge_per_bag)
  (h_monday : monday_bags = 5)
  (h_tuesday : tuesday_bags = 3)
  (h_total : total_money = 68) :
  wednesday_bags = (total_money - (monday_bags + tuesday_bags) * money_per_bag) / charge_per_bag :=
by
  sorry

end bags_on_wednesday_l1509_150961


namespace arithmetic_seq_first_term_l1509_150939

theorem arithmetic_seq_first_term (S : ℕ → ℚ) (n : ℕ) (a : ℚ)
  (h₁ : ∀ n, S n = n * (2 * a + (n - 1) * 5) / 2)
  (h₂ : ∀ n, S (3 * n) / S n = 9) :
  a = 5 / 2 :=
by
  sorry

end arithmetic_seq_first_term_l1509_150939


namespace prime_numbers_in_list_l1509_150953

noncomputable def list_numbers : ℕ → ℕ
| 0       => 43
| (n + 1) => 43 * ((10 ^ (2 * n + 2) - 1) / 99) 

theorem prime_numbers_in_list : ∃ n:ℕ, (∀ m, (m > n) → ¬ Prime (list_numbers m)) ∧ Prime (list_numbers 0) := 
by
  sorry

end prime_numbers_in_list_l1509_150953


namespace xiao_wang_conjecture_incorrect_l1509_150971

theorem xiao_wang_conjecture_incorrect : ∃ n : ℕ, n > 0 ∧ (n^2 - 8 * n + 7 > 0) := by
  sorry

end xiao_wang_conjecture_incorrect_l1509_150971


namespace average_speed_third_hour_l1509_150992

theorem average_speed_third_hour
  (total_distance : ℝ)
  (total_time : ℝ)
  (speed_first_hour : ℝ)
  (speed_second_hour : ℝ)
  (speed_third_hour : ℝ) :
  total_distance = 150 →
  total_time = 3 →
  speed_first_hour = 45 →
  speed_second_hour = 55 →
  (speed_first_hour + speed_second_hour + speed_third_hour) / total_time = 50 →
  speed_third_hour = 50 :=
sorry

end average_speed_third_hour_l1509_150992


namespace g_inv_undefined_at_1_l1509_150934

noncomputable def g (x : ℝ) : ℝ := (x - 3) / (x - 5)

noncomputable def g_inv (x : ℝ) : ℝ := (5 * x - 3) / (x - 1)

theorem g_inv_undefined_at_1 : ∀ x : ℝ, (g_inv x) = g_inv 1 → x = 1 :=
by
  intro x h
  sorry

end g_inv_undefined_at_1_l1509_150934


namespace quadratic_has_two_distinct_real_roots_l1509_150943

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁^2 + m * x₁ - 8 = 0) ∧ (x₂^2 + m * x₂ - 8 = 0) :=
by
  let Δ := m^2 + 32
  have hΔ : Δ > 0 := by
    simp [Δ]
    exact add_pos_of_nonneg_of_pos (sq_nonneg m) (by norm_num)
  sorry

end quadratic_has_two_distinct_real_roots_l1509_150943


namespace probability_all_red_or_all_white_l1509_150957

theorem probability_all_red_or_all_white :
  let red_marbles := 5
  let white_marbles := 4
  let blue_marbles := 6
  let total_marbles := red_marbles + white_marbles + blue_marbles
  let probability_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
  let probability_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
  (probability_red + probability_white) = (14 / 455) :=
by
  sorry

end probability_all_red_or_all_white_l1509_150957


namespace find_base_b_l1509_150983

theorem find_base_b (b : ℕ) (h : (3 * b + 4) ^ 2 = b ^ 3 + 2 * b ^ 2 + 9 * b + 6) : b = 10 :=
sorry

end find_base_b_l1509_150983


namespace find_overlap_length_l1509_150990

-- Definitions of the given conditions
def total_length_of_segments := 98 -- cm
def edge_to_edge_distance := 83 -- cm
def number_of_overlaps := 6

-- Theorem stating the value of x in centimeters
theorem find_overlap_length (x : ℝ) 
  (h1 : total_length_of_segments = 98) 
  (h2 : edge_to_edge_distance = 83) 
  (h3 : number_of_overlaps = 6) 
  (h4 : total_length_of_segments = edge_to_edge_distance + number_of_overlaps * x) : 
  x = 2.5 :=
  sorry

end find_overlap_length_l1509_150990


namespace total_cost_backpacks_l1509_150936

theorem total_cost_backpacks:
  let original_price := 20.00
  let discount := 0.20
  let monogram_cost := 12.00
  let coupon := 5.00
  let state_tax : List Real := [0.06, 0.08, 0.055, 0.0725, 0.04]
  let discounted_price := original_price * (1 - discount)
  let pre_tax_cost := discounted_price + monogram_cost
  let final_costs := state_tax.map (λ tax_rate => pre_tax_cost * (1 + tax_rate))
  let total_cost_before_coupon := final_costs.sum
  total_cost_before_coupon - coupon = 143.61 := by
    sorry

end total_cost_backpacks_l1509_150936


namespace point_P_distance_to_y_axis_l1509_150944

-- Define the coordinates of point P
def point_P : ℝ × ℝ := (-2, 3)

-- The distance from point P to the y-axis
def distance_to_y_axis (pt : ℝ × ℝ) : ℝ :=
  abs pt.1

-- Statement to prove
theorem point_P_distance_to_y_axis :
  distance_to_y_axis point_P = 2 :=
by
  sorry

end point_P_distance_to_y_axis_l1509_150944
