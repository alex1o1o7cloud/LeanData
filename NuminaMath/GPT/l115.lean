import Mathlib

namespace find_common_difference_l115_115670

-- Definitions for arithmetic sequences and sums
def S (a1 d : ℕ) (n : ℕ) := (n * (2 * a1 + (n - 1) * d)) / 2
def a (a1 d : ℕ) (n : ℕ) := a1 + (n - 1) * d

theorem find_common_difference (a1 d : ℕ) :
  S a1 d 3 = 6 → a a1 d 3 = 4 → d = 2 :=
by
  intros S3_eq a3_eq
  sorry

end find_common_difference_l115_115670


namespace question_1_solution_question_2_solution_l115_115216

def f (m x : ℝ) := m*x^2 - (m^2 + 1)*x + m

theorem question_1_solution (x : ℝ) :
  (f 2 x ≤ 0) ↔ (1 / 2 ≤ x ∧ x ≤ 2) :=
sorry

theorem question_2_solution (x m : ℝ) :
  (m > 0) → 
  ((0 < m ∧ m < 1 → f m x > 0 ↔ x < m ∨ x > 1 / m) ∧
  (m = 1 → f m x > 0 ↔ x ≠ 1) ∧
  (m > 1 → f m x > 0 ↔ x < 1 / m ∨ x > m)) :=
sorry

end question_1_solution_question_2_solution_l115_115216


namespace visited_neither_l115_115480

def people_total : ℕ := 90
def visited_iceland : ℕ := 55
def visited_norway : ℕ := 33
def visited_both : ℕ := 51

theorem visited_neither :
  people_total - (visited_iceland + visited_norway - visited_both) = 53 := by
  sorry

end visited_neither_l115_115480


namespace total_population_milburg_l115_115036

def num_children : ℕ := 2987
def num_adults : ℕ := 2269

theorem total_population_milburg : num_children + num_adults = 5256 := by
  sorry

end total_population_milburg_l115_115036


namespace part_I_part_II_part_III_l115_115986

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (1 / 2) * a * x^2

-- Part (Ⅰ)
theorem part_I (x : ℝ) : (0 < x) → (f 1 x < f 1 (x+1)) := sorry

-- Part (Ⅱ)
theorem part_II (f_has_two_distinct_extreme_values : ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ (f a x = f a y))) : 0 < a ∧ a < 1 := sorry

-- Part (Ⅲ)
theorem part_III (f_has_two_distinct_zeros : ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0)) : 0 < a ∧ a < (2 / Real.exp 1) := sorry

end part_I_part_II_part_III_l115_115986


namespace no_arithmetic_seq_with_sum_n_cubed_l115_115538

theorem no_arithmetic_seq_with_sum_n_cubed (a1 d : ℕ) :
  ¬ (∀ (n : ℕ), (n > 0) → (n / 2) * (2 * a1 + (n - 1) * d) = n^3) :=
sorry

end no_arithmetic_seq_with_sum_n_cubed_l115_115538


namespace inequality_holds_for_all_x_l115_115021

theorem inequality_holds_for_all_x (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 4) * x - k + 8 > 0) ↔ (-2 < k ∧ k < 6) :=
by
  sorry

end inequality_holds_for_all_x_l115_115021


namespace simplify_fraction_product_l115_115926

theorem simplify_fraction_product : 
  (256 / 20 : ℚ) * (10 / 160) * ((16 / 6) ^ 2) = 256 / 45 :=
by norm_num

end simplify_fraction_product_l115_115926


namespace sum_of_digits_of_greatest_prime_divisor_of_4095_l115_115061

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def greatest_prime_divisor_of_4095 : ℕ := 13

theorem sum_of_digits_of_greatest_prime_divisor_of_4095 :
  sum_of_digits greatest_prime_divisor_of_4095 = 4 := by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_4095_l115_115061


namespace dots_not_visible_l115_115742

theorem dots_not_visible (visible_sum : ℕ) (total_faces_sum : ℕ) (num_dice : ℕ) (total_visible_faces : ℕ)
  (h1 : total_faces_sum = 21)
  (h2 : visible_sum = 22) 
  (h3 : num_dice = 3)
  (h4 : total_visible_faces = 7) :
  (num_dice * total_faces_sum - visible_sum) = 41 :=
sorry

end dots_not_visible_l115_115742


namespace intersection_A_B_l115_115694

open Set

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def B : Set ℕ := {x | 2 ≤ x ∧ x < 6}

theorem intersection_A_B : A ∩ B = {2, 3, 4, 5} := by
  sorry

end intersection_A_B_l115_115694


namespace seq_a10_eq_90_l115_115722

noncomputable def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + 2 * n

theorem seq_a10_eq_90 {a : ℕ → ℕ} (h : seq a) : a 10 = 90 :=
  sorry

end seq_a10_eq_90_l115_115722


namespace f_monotonic_non_overlapping_domains_domain_of_sum_l115_115661

axiom f : ℝ → ℝ
axiom f_decreasing : ∀ x₁ x₂ : ℝ, -1 ≤ x₁ → x₁ ≤ 1 → -1 ≤ x₂ → x₂ ≤ 1 → x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0

theorem f_monotonic : ∀ x₁ x₂ : ℝ, -1 ≤ x₁ → x₁ ≤ 1 → -1 ≤ x₂ → x₂ ≤ 1 → x₁ ≤ x₂ → f x₁ ≥ f x₂ := sorry

theorem non_overlapping_domains : ∀ c : ℝ, (c - 1 > c^2 + 1 → c > 2) ∧ (c^2 - 1 > c + 1 → c < -1) := sorry

theorem domain_of_sum : 
  ∀ c : ℝ,
  -1 ≤ c ∧ c ≤ 2 →
  (∃ a b : ℝ, 
    ((-1 ≤ c ∧ c ≤ 0) ∨ (1 ≤ c ∧ c ≤ 2) → a = c^2 - 1 ∧ b = c + 1) ∧ 
    (0 < c ∧ c < 1 → a = c - 1 ∧ b = c^2 + 1)
  ) := sorry

end f_monotonic_non_overlapping_domains_domain_of_sum_l115_115661


namespace trapezoid_area_ratio_l115_115867

theorem trapezoid_area_ratio (AD AO OB BC AB DO OC : ℝ) (h_eq1 : AD = 15) (h_eq2 : AO = 15) (h_eq3 : OB = 15) (h_eq4 : BC = 15)
  (h_eq5 : AB = 20) (h_eq6 : DO = 20) (h_eq7 : OC = 20) (is_trapezoid : true) (OP_perp_to_AB : true) 
  (X_mid_AD : true) (Y_mid_BC : true) : (5 + 7 = 12) :=
by
  sorry

end trapezoid_area_ratio_l115_115867


namespace max_value_of_a_l115_115518

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a ≥ 0) → a ≤ 4 := 
by {
  sorry
}

end max_value_of_a_l115_115518


namespace Ted_has_15_bags_l115_115852

-- Define the parameters
def total_candy_bars : ℕ := 75
def candy_per_bag : ℝ := 5.0

-- Define the assertion to be proved
theorem Ted_has_15_bags : total_candy_bars / candy_per_bag = 15 := 
by
  sorry

end Ted_has_15_bags_l115_115852


namespace compelling_quadruples_l115_115577
   
   def isCompellingQuadruple (a b c d : ℕ) : Prop :=
     1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 10 ∧ a + d < b + c 

   def compellingQuadruplesCount (count : ℕ) : Prop :=
     count = 80
   
   theorem compelling_quadruples :
     ∃ count, compellingQuadruplesCount count :=
   by
     use 80
     sorry
   
end compelling_quadruples_l115_115577


namespace volume_OABC_is_l115_115306

noncomputable def volume_tetrahedron_ABC (a b c : ℝ) (hx : a^2 + b^2 = 36) (hy : b^2 + c^2 = 25) (hz : c^2 + a^2 = 16) : ℝ :=
  1 / 6 * a * b * c

theorem volume_OABC_is (a b c : ℝ) (hx : a^2 + b^2 = 36) (hy : b^2 + c^2 = 25) (hz : c^2 + a^2 = 16) :
  volume_tetrahedron_ABC a b c hx hy hz = (5 / 6) * Real.sqrt 30.375 :=
by
  sorry

end volume_OABC_is_l115_115306


namespace alcohol_percentage_calculation_l115_115213

-- Define the conditions as hypothesis
variables (original_solution_volume : ℝ) (original_alcohol_percent : ℝ)
          (added_alcohol_volume : ℝ) (added_water_volume : ℝ)

-- Assume the given values in the problem
variables (h1 : original_solution_volume = 40) (h2 : original_alcohol_percent = 5)
          (h3 : added_alcohol_volume = 2.5) (h4 : added_water_volume = 7.5)

-- Define the proof goal
theorem alcohol_percentage_calculation :
  let original_alcohol_volume := original_solution_volume * (original_alcohol_percent / 100)
  let total_alcohol_volume := original_alcohol_volume + added_alcohol_volume
  let total_solution_volume := original_solution_volume + added_alcohol_volume + added_water_volume
  let new_alcohol_percent := (total_alcohol_volume / total_solution_volume) * 100
  new_alcohol_percent = 9 :=
by {
  sorry
}

end alcohol_percentage_calculation_l115_115213


namespace complex_solution_l115_115586

theorem complex_solution (z : ℂ) (h : z^2 = -5 - 12 * Complex.I) :
  z = 2 - 3 * Complex.I ∨ z = -2 + 3 * Complex.I := 
sorry

end complex_solution_l115_115586


namespace heather_walked_distance_l115_115493

theorem heather_walked_distance {H S : ℝ} (hH : H = 5) (hS : S = H + 1) (total_distance : ℝ) (time_delay_stacy : ℝ) (time_heather_meet : ℝ) :
  (total_distance = 30) → (time_delay_stacy = 0.4) → (time_heather_meet = (total_distance - S * time_delay_stacy) / (H + S)) →
  (H * time_heather_meet = 12.55) :=
by
  sorry

end heather_walked_distance_l115_115493


namespace absolute_difference_distance_l115_115619

/-- Renaldo drove 15 kilometers, Ernesto drove 7 kilometers more than one-third of Renaldo's distance, 
Marcos drove -5 kilometers. Prove that the absolute difference between the total distances driven by 
Renaldo and Ernesto combined, and the distance driven by Marcos is 22 kilometers. -/
theorem absolute_difference_distance :
  let renaldo_distance := 15
  let ernesto_distance := 7 + (1 / 3) * renaldo_distance
  let marcos_distance := -5
  abs ((renaldo_distance + ernesto_distance) - marcos_distance) = 22 := by
  sorry

end absolute_difference_distance_l115_115619


namespace sin_2phi_l115_115431

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l115_115431


namespace factor_54x5_135x9_l115_115894

theorem factor_54x5_135x9 (x : ℝ) :
  54 * x ^ 5 - 135 * x ^ 9 = -27 * x ^ 5 * (5 * x ^ 4 - 2) :=
by 
  sorry

end factor_54x5_135x9_l115_115894


namespace find_d_l115_115155

-- Definitions for the functions f and g
def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

-- Statement to prove the value of d
theorem find_d (c d : ℝ) (h1 : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  -- inserting custom logic for proof
  sorry

end find_d_l115_115155


namespace min_value_of_expression_l115_115416

theorem min_value_of_expression (a b : ℝ) (h₁ : 0 < a) (h₂ : 1 < b) (h₃ : a + b = 2) :
  4 / a + 1 / (b - 1) = 9 := 
sorry

end min_value_of_expression_l115_115416


namespace parallelepiped_properties_l115_115325

/--
In an oblique parallelepiped with the following properties:
- The height is 12 dm,
- The projection of the lateral edge on the base plane is 5 dm,
- A cross-section perpendicular to the lateral edge is a rhombus with:
  - An area of 24 dm²,
  - A diagonal of 8 dm,
Prove that:
1. The lateral surface area is 260 dm².
2. The volume is 312 dm³.
-/
theorem parallelepiped_properties
    (height : ℝ)
    (projection_lateral_edge : ℝ)
    (area_rhombus : ℝ)
    (diagonal_rhombus : ℝ)
    (lateral_surface_area : ℝ)
    (volume : ℝ) :
  height = 12 ∧
  projection_lateral_edge = 5 ∧
  area_rhombus = 24 ∧
  diagonal_rhombus = 8 ∧
  lateral_surface_area = 260 ∧
  volume = 312 :=
by
  sorry

end parallelepiped_properties_l115_115325


namespace min_possible_value_of_coefficient_x_l115_115067

theorem min_possible_value_of_coefficient_x 
  (c d : ℤ) 
  (h1 : c * d = 15) 
  (h2 : ∃ (C : ℤ), C = c + d) 
  (h3 : c ≠ d ∧ c ≠ 34 ∧ d ≠ 34) :
  (∃ (C : ℤ), C = c + d ∧ C = 34) :=
sorry

end min_possible_value_of_coefficient_x_l115_115067


namespace z_share_per_rupee_x_l115_115363

-- Definitions according to the conditions
def x_gets (r : ℝ) : ℝ := r
def y_gets_for_x (r : ℝ) : ℝ := 0.45 * r
def y_share : ℝ := 18
def total_amount : ℝ := 78

-- Problem statement to prove z gets 0.5 rupees for each rupee x gets.
theorem z_share_per_rupee_x (r : ℝ) (hx : x_gets r = 40) (hy : y_gets_for_x r = 18) (ht : total_amount = 78) :
  (total_amount - (x_gets r + y_share)) / x_gets r = 0.5 := by
  sorry

end z_share_per_rupee_x_l115_115363


namespace common_intersection_implies_cd_l115_115172

theorem common_intersection_implies_cd (a b c d : ℝ) (h : a ≠ b) (x y : ℝ) 
  (H1 : y = a * x + a) (H2 : y = b * x + b) (H3 : y = c * x + d) : c = d := by
  sorry

end common_intersection_implies_cd_l115_115172


namespace UF_opponent_score_l115_115583

theorem UF_opponent_score 
  (total_points : ℕ)
  (games_played : ℕ)
  (previous_points_avg : ℕ)
  (championship_score : ℕ)
  (opponent_score : ℕ)
  (total_points_condition : total_points = 720)
  (games_played_condition : games_played = 24)
  (previous_points_avg_condition : previous_points_avg = total_points / games_played)
  (championship_score_condition : championship_score = previous_points_avg / 2 - 2)
  (loss_by_condition : opponent_score = championship_score - 2) :
  opponent_score = 11 :=
by
  sorry

end UF_opponent_score_l115_115583


namespace average_mark_excluded_students_l115_115358

variables (N A E A_R A_E : ℕ)

theorem average_mark_excluded_students:
    N = 56 → A = 80 → E = 8 → A_R = 90 →
    N * A = E * A_E + (N - E) * A_R →
    A_E = 20 :=
by
  intros hN hA hE hAR hEquation
  rw [hN, hA, hE, hAR] at hEquation
  have h : 4480 = 8 * A_E + 4320 := hEquation
  sorry

end average_mark_excluded_students_l115_115358


namespace construct_trihedral_angle_l115_115281

-- Define the magnitudes of dihedral angles
variables (α β γ : ℝ)

-- Problem statement
theorem construct_trihedral_angle (h₀ : 0 < α) (h₁ : 0 < β) (h₂ : 0 < γ) :
  ∃ (trihedral_angle : Type), true := 
sorry

end construct_trihedral_angle_l115_115281


namespace measure_angle_Z_l115_115842

-- Given conditions
def triangle_condition (X Y Z : ℝ) :=
   X = 78 ∧ Y = 4 * Z - 14

-- Triangle angle sum property
def triangle_angle_sum (X Y Z : ℝ) :=
   X + Y + Z = 180

-- Prove the measure of angle Z
theorem measure_angle_Z (X Y Z : ℝ) (h1 : triangle_condition X Y Z) (h2 : triangle_angle_sum X Y Z) : 
  Z = 23.2 :=
by
  -- Lean will expect proof steps here, ‘sorry’ is used to denote unproven parts.
  sorry

end measure_angle_Z_l115_115842


namespace days_passed_before_cows_ran_away_l115_115109

def initial_cows := 1000
def initial_days := 50
def cows_left := 800
def cows_run_away := initial_cows - cows_left
def total_food := initial_cows * initial_days
def remaining_food (x : ℕ) := total_food - initial_cows * x
def food_needed := cows_left * initial_days

theorem days_passed_before_cows_ran_away (x : ℕ) :
  (remaining_food x = food_needed) → (x = 10) :=
by
  sorry

end days_passed_before_cows_ran_away_l115_115109


namespace isosceles_triangle_area_l115_115594

-- Definitions
def isosceles_triangle (b h : ℝ) : Prop :=
∃ a : ℝ, a * b / 2 = a * h

def square_of_area_one (a : ℝ) : Prop :=
a = 1

def centroids_coincide (g_triangle g_square : ℝ × ℝ) : Prop :=
g_triangle = g_square

-- The statement of the problem
theorem isosceles_triangle_area
  (b h : ℝ)
  (s : ℝ)
  (triangle_centroid : ℝ × ℝ)
  (square_centroid : ℝ × ℝ)
  (H1 : isosceles_triangle b h)
  (H2 : square_of_area_one s)
  (H3 : centroids_coincide triangle_centroid square_centroid)
  : b * h / 2 = 9 / 4 :=
by
  sorry

end isosceles_triangle_area_l115_115594


namespace sin_eq_sqrt3_div_2_range_l115_115530

theorem sin_eq_sqrt3_div_2_range :
  {x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin x ≥ Real.sqrt 3 / 2} = 
  {x | Real.pi / 3 ≤ x ∧ x ≤ 2 * Real.pi / 3} :=
sorry

end sin_eq_sqrt3_div_2_range_l115_115530


namespace henri_drove_more_miles_l115_115100

-- Defining the conditions
def Gervais_average_miles_per_day := 315
def Gervais_days_driven := 3
def Henri_total_miles := 1250

-- Total miles driven by Gervais
def Gervais_total_miles := Gervais_average_miles_per_day * Gervais_days_driven

-- The proof problem statement
theorem henri_drove_more_miles : Henri_total_miles - Gervais_total_miles = 305 := 
by 
  sorry

end henri_drove_more_miles_l115_115100


namespace ratio_of_division_of_chord_l115_115227

theorem ratio_of_division_of_chord (R AP PB O: ℝ) (radius_given: R = 11) (chord_length: AP + PB = 18) (point_distance: O = 7) : 
  (AP / PB = 2 ∨ PB / AP = 2) :=
by 
  -- Proof goes here, to be filled in later
  sorry

end ratio_of_division_of_chord_l115_115227


namespace cost_of_chairs_l115_115906

-- Given conditions
def total_spent : ℕ := 56
def cost_of_table : ℕ := 34
def number_of_chairs : ℕ := 2

-- The target definition
def cost_of_one_chair : ℕ := 11

-- Statement to prove
theorem cost_of_chairs (N : ℕ) (T : ℕ) (C : ℕ) (x : ℕ) (hN : N = total_spent) (hT : T = cost_of_table) (hC : C = number_of_chairs) : x = cost_of_one_chair ↔ N - T = C * x :=
by
  sorry

end cost_of_chairs_l115_115906


namespace proof_theorem_l115_115041

noncomputable def proof_problem (y1 y2 y3 y4 y5 : ℝ) :=
  y1 + 8*y2 + 27*y3 + 64*y4 + 125*y5 = 7 ∧
  8*y1 + 27*y2 + 64*y3 + 125*y4 + 216*y5 = 100 ∧
  27*y1 + 64*y2 + 125*y3 + 216*y4 + 343*y5 = 1000 →
  64*y1 + 125*y2 + 216*y3 + 343*y4 + 512*y5 = -5999

theorem proof_theorem : ∀ (y1 y2 y3 y4 y5 : ℝ), proof_problem y1 y2 y3 y4 y5 :=
  by intros y1 y2 y3 y4 y5
     unfold proof_problem
     intro h
     sorry

end proof_theorem_l115_115041


namespace sum_of_primes_product_166_l115_115181

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m < n → m > 0 → n % m ≠ 0

theorem sum_of_primes_product_166
    (p1 p2 : ℕ)
    (prime_p1 : is_prime p1)
    (prime_p2 : is_prime p2)
    (product_condition : p1 * p2 = 166) :
    p1 + p2 = 85 :=
    sorry

end sum_of_primes_product_166_l115_115181


namespace solve_equation_l115_115819

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) :
  (x / (x - 1) - 2 / x = 1) ↔ x = 2 :=
sorry

end solve_equation_l115_115819


namespace rectangle_vertices_complex_plane_l115_115560

theorem rectangle_vertices_complex_plane (b : ℝ) :
  (∀ (z : ℂ), z^4 - 10*z^3 + (16*b : ℂ)*z^2 - 2*(3*b^2 - 5*b + 4 : ℂ)*z + 6 = 0 →
    (∃ (w₁ w₂ : ℂ), z = w₁ ∨ z = w₂)) →
  (b = 5 / 3 ∨ b = 2) :=
sorry

end rectangle_vertices_complex_plane_l115_115560


namespace xy_sufficient_but_not_necessary_l115_115756

theorem xy_sufficient_but_not_necessary (x y : ℝ) : (x > 0 ∧ y > 0) → (xy > 0) ∧ ¬(xy > 0 → (x > 0 ∧ y > 0)) :=
by
  intros h
  sorry

end xy_sufficient_but_not_necessary_l115_115756


namespace time_to_pay_back_l115_115320

-- Definitions for conditions
def initial_cost : ℕ := 25000
def monthly_revenue : ℕ := 4000
def monthly_expenses : ℕ := 1500
def monthly_profit : ℕ := monthly_revenue - monthly_expenses

-- Theorem statement
theorem time_to_pay_back : initial_cost / monthly_profit = 10 := by
  -- Skipping the proof here
  sorry

end time_to_pay_back_l115_115320


namespace solve_r_l115_115625

-- Definitions related to the problem
def satisfies_equation (r : ℝ) : Prop := ⌊r⌋ + 2 * r = 16

-- Theorem statement
theorem solve_r : ∃ (r : ℝ), satisfies_equation r ∧ r = 5.5 :=
by
  sorry

end solve_r_l115_115625


namespace robin_spent_on_leftover_drinks_l115_115162

-- Define the number of each type of drink bought and consumed
def sodas_bought : Nat := 30
def sodas_price : Nat := 2
def sodas_consumed : Nat := 10

def energy_drinks_bought : Nat := 20
def energy_drinks_price : Nat := 3
def energy_drinks_consumed : Nat := 14

def smoothies_bought : Nat := 15
def smoothies_price : Nat := 4
def smoothies_consumed : Nat := 5

-- Define the total cost calculation
def total_spent_on_leftover_drinks : Nat :=
  (sodas_bought * sodas_price - sodas_consumed * sodas_price) +
  (energy_drinks_bought * energy_drinks_price - energy_drinks_consumed * energy_drinks_price) +
  (smoothies_bought * smoothies_price - smoothies_consumed * smoothies_price)

theorem robin_spent_on_leftover_drinks : total_spent_on_leftover_drinks = 98 := by
  -- Provide the proof steps here (not required for this task)
  sorry

end robin_spent_on_leftover_drinks_l115_115162


namespace area_of_rotated_squares_l115_115083

noncomputable def side_length : ℝ := 8
noncomputable def rotation_middle : ℝ := 45
noncomputable def rotation_top : ℝ := 75

-- Theorem: The area of the resulting 24-sided polygon.
theorem area_of_rotated_squares :
  (∃ (polygon_area : ℝ), polygon_area = 96) :=
sorry

end area_of_rotated_squares_l115_115083


namespace find_x_l115_115396

theorem find_x (x : ℝ) : 
  0.65 * x = 0.20 * 682.50 → x = 210 := 
by 
  sorry

end find_x_l115_115396


namespace abc_sum_leq_three_l115_115240

open Real

theorem abc_sum_leq_three {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 + c^2 + a * b * c = 4) :
  a + b + c ≤ 3 :=
sorry

end abc_sum_leq_three_l115_115240


namespace train_probability_at_station_l115_115831

-- Define time intervals
def t0 := 0 -- Train arrival start time in minutes after 1:00 PM
def t1 := 60 -- Train arrival end time in minutes after 1:00 PM
def a0 := 0 -- Alex arrival start time in minutes after 1:00 PM
def a1 := 120 -- Alex arrival end time in minutes after 1:00 PM

-- Define the probability calculation problem
theorem train_probability_at_station :
  let total_area := (t1 - t0) * (a1 - a0)
  let overlap_area := (1/2 * 50 * 50) + (10 * 55)
  (overlap_area / total_area) = 1/4 := 
by
  sorry

end train_probability_at_station_l115_115831


namespace monthly_fee_for_second_plan_l115_115579

theorem monthly_fee_for_second_plan 
  (monthly_fee_first_plan : ℝ) 
  (rate_first_plan : ℝ) 
  (rate_second_plan : ℝ) 
  (minutes : ℕ) 
  (monthly_fee_second_plan : ℝ) :
  monthly_fee_first_plan = 22 -> 
  rate_first_plan = 0.13 -> 
  rate_second_plan = 0.18 -> 
  minutes = 280 -> 
  (22 + 0.13 * 280 = monthly_fee_second_plan + 0.18 * 280) -> 
  monthly_fee_second_plan = 8 := 
by
  intros h_fee_first_plan h_rate_first_plan h_rate_second_plan h_minutes h_equal_costs
  sorry

end monthly_fee_for_second_plan_l115_115579


namespace rectangular_prism_width_l115_115833

theorem rectangular_prism_width 
  (l : ℝ) (h : ℝ) (d : ℝ) (w : ℝ)
  (hl : l = 5) (hh : h = 7) (hd : d = 14) :
  d = Real.sqrt (l^2 + w^2 + h^2) → w = Real.sqrt 122 :=
by 
  sorry

end rectangular_prism_width_l115_115833


namespace find_a5_l115_115884

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)

-- Given conditions
variable (a : ℕ → ℝ)
variable (h_arith : arithmetic_sequence a)
variable (h_a1 : a 0 = 2)
variable (h_sum : a 1 + a 3 = 8)

-- The target question
theorem find_a5 : a 4 = 6 :=
by
  sorry

end find_a5_l115_115884


namespace algebra_expression_evaluation_l115_115873

theorem algebra_expression_evaluation (a b : ℝ) (h : a + 3 * b = 4) : 2 * a + 6 * b - 1 = 7 := by
  sorry

end algebra_expression_evaluation_l115_115873


namespace even_function_coeff_l115_115639

theorem even_function_coeff (a : ℝ) (h : ∀ x : ℝ, (a-2)*x^2 + (a-1)*x + 3 = (a-2)*(-x)^2 + (a-1)*(-x) + 3) : a = 1 :=
by {
  -- Proof here
  sorry
}

end even_function_coeff_l115_115639


namespace capacity_of_second_bucket_l115_115621

theorem capacity_of_second_bucket (c1 : ∃ (tank_capacity : ℕ), tank_capacity = 12 * 49) (c2 : ∃ (bucket_count : ℕ), bucket_count = 84) :
  ∃ (bucket_capacity : ℕ), bucket_capacity = 7 :=
by
  -- Extract the total capacity of the tank from condition 1
  obtain ⟨tank_capacity, htank⟩ := c1
  -- Extract the number of buckets from condition 2
  obtain ⟨bucket_count, hcount⟩ := c2
  -- Use the given relations to calculate the capacity of each bucket
  use tank_capacity / bucket_count
  -- Provide the necessary calculations
  sorry

end capacity_of_second_bucket_l115_115621


namespace cricket_team_players_l115_115541

theorem cricket_team_players (P N : ℕ) (h1 : 37 = 37) 
  (h2 : (57 - 37) = 20) 
  (h3 : ∀ N, (2 / 3 : ℚ) * N = 20 → N = 30) 
  (h4 : P = 37 + 30) : P = 67 := 
by
  -- Proof steps will go here
  sorry

end cricket_team_players_l115_115541


namespace interval_of_x_l115_115027

theorem interval_of_x (x : ℝ) (h : x = ((-x)^2 / x) + 3) : 3 < x ∧ x ≤ 6 :=
by
  sorry

end interval_of_x_l115_115027


namespace jasons_shelves_l115_115598

theorem jasons_shelves (total_books : ℕ) (number_of_shelves : ℕ) (h_total_books : total_books = 315) (h_number_of_shelves : number_of_shelves = 7) : (total_books / number_of_shelves) = 45 := 
by
  sorry

end jasons_shelves_l115_115598


namespace new_difference_l115_115460

theorem new_difference (x y a : ℝ) (h : x - y = a) : (x + 0.5) - y = a + 0.5 := 
sorry

end new_difference_l115_115460


namespace exists_irrational_an_l115_115704

theorem exists_irrational_an (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n ≥ 1, a (n + 1)^2 = a n + 1) :
  ∃ n, ¬ ∃ q : ℚ, a n = q :=
sorry

end exists_irrational_an_l115_115704


namespace slower_speed_is_l115_115977

def slower_speed_problem
  (faster_speed : ℝ)
  (additional_distance : ℝ)
  (actual_distance : ℝ)
  (v : ℝ) :
  Prop :=
  actual_distance / v = (actual_distance + additional_distance) / faster_speed

theorem slower_speed_is
  (h1 : faster_speed = 25)
  (h2 : additional_distance = 20)
  (h3 : actual_distance = 13.333333333333332)
  : ∃ v : ℝ,  slower_speed_problem faster_speed additional_distance actual_distance v ∧ v = 10 :=
by {
  sorry
}

end slower_speed_is_l115_115977


namespace range_of_a_l115_115570

noncomputable def p (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^2 * x^2 + a * x - 2 = 0

noncomputable def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → (1 < a ∨ -1 < a ∧ a < 1) :=
by sorry

end range_of_a_l115_115570


namespace combined_selling_price_l115_115671

theorem combined_selling_price 
  (cost_price1 cost_price2 cost_price3 : ℚ)
  (profit_percentage1 profit_percentage2 profit_percentage3 : ℚ)
  (h1 : cost_price1 = 1200) (h2 : profit_percentage1 = 0.4)
  (h3 : cost_price2 = 800)  (h4 : profit_percentage2 = 0.3)
  (h5 : cost_price3 = 600)  (h6 : profit_percentage3 = 0.5) : 
  cost_price1 * (1 + profit_percentage1) +
  cost_price2 * (1 + profit_percentage2) +
  cost_price3 * (1 + profit_percentage3) = 3620 := by 
  sorry

end combined_selling_price_l115_115671


namespace golden_ratio_in_range_l115_115146

noncomputable def golden_ratio := (Real.sqrt 5 - 1) / 2

theorem golden_ratio_in_range : 0.6 < golden_ratio ∧ golden_ratio < 0.7 :=
by
  sorry

end golden_ratio_in_range_l115_115146


namespace trig_identity_l115_115674

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 :=
by
  sorry

end trig_identity_l115_115674


namespace rectangle_area_correct_l115_115999

-- Definitions of side lengths
def sideOne : ℝ := 5.9
def sideTwo : ℝ := 3

-- Definition of the area calculation for a rectangle
def rectangleArea (a b : ℝ) : ℝ :=
  a * b

-- The main theorem stating the area is as calculated
theorem rectangle_area_correct :
  rectangleArea sideOne sideTwo = 17.7 := by
  sorry

end rectangle_area_correct_l115_115999


namespace cost_of_dozen_pens_l115_115526

theorem cost_of_dozen_pens 
  (x : ℝ)
  (hx_pos : 0 < x)
  (h1 : 3 * (5 * x) + 5 * x = 150)
  (h2 : 5 * x / x = 5): 
  12 * (5 * x) = 450 :=
by
  sorry

end cost_of_dozen_pens_l115_115526


namespace students_scoring_80_percent_l115_115975

theorem students_scoring_80_percent
  (x : ℕ)
  (h1 : 10 * 90 + x * 80 = 25 * 84)
  (h2 : x + 10 = 25) : x = 15 := 
by {
  -- Proof goes here
  sorry
}

end students_scoring_80_percent_l115_115975


namespace sally_spent_eur_l115_115705

-- Define the given conditions
def coupon_value : ℝ := 3
def peaches_total_usd : ℝ := 12.32
def cherries_original_usd : ℝ := 11.54
def discount_rate : ℝ := 0.1
def conversion_rate : ℝ := 0.85

-- Define the intermediate calculations
def cherries_discount_usd : ℝ := cherries_original_usd * discount_rate
def cherries_final_usd : ℝ := cherries_original_usd - cherries_discount_usd
def total_usd : ℝ := peaches_total_usd + cherries_final_usd
def total_eur : ℝ := total_usd * conversion_rate

-- The final statement to be proven
theorem sally_spent_eur : total_eur = 19.30 := by
  sorry

end sally_spent_eur_l115_115705


namespace train_constant_speed_is_48_l115_115376

theorem train_constant_speed_is_48 
  (d_12_00 d_12_15 d_12_45 : ℝ)
  (h1 : 72.5 ≤ d_12_00 ∧ d_12_00 < 73.5)
  (h2 : 61.5 ≤ d_12_15 ∧ d_12_15 < 62.5)
  (h3 : 36.5 ≤ d_12_45 ∧ d_12_45 < 37.5)
  (constant_speed : ℝ → ℝ): 
  (constant_speed d_12_15 - constant_speed d_12_00 = 48) ∧
  (constant_speed d_12_45 - constant_speed d_12_15 = 48) :=
by
  sorry

end train_constant_speed_is_48_l115_115376


namespace decagon_diagonals_l115_115762

-- Number of diagonals calculation definition
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Proving the number of diagonals in a decagon
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l115_115762


namespace sum_of_squares_of_roots_l115_115087

theorem sum_of_squares_of_roots (x1 x2 : ℝ) (h1 : 2 * x1^2 + 5 * x1 - 12 = 0) (h2 : 2 * x2^2 + 5 * x2 - 12 = 0) (h3 : x1 ≠ x2) :
  x1^2 + x2^2 = 73 / 4 :=
sorry

end sum_of_squares_of_roots_l115_115087


namespace triangle_with_consecutive_sides_and_angle_property_l115_115938

theorem triangle_with_consecutive_sides_and_angle_property :
  ∃ (a b c : ℕ), (b = a + 1) ∧ (c = b + 1) ∧
    (∃ (α β γ : ℝ), 2 * α = γ ∧
      (a * a + b * b = c * c + 2 * a * b * α.cos) ∧
      (b * b + c * c = a * a + 2 * b * c * β.cos) ∧
      (c * c + a * a = b * b + 2 * c * a * γ.cos) ∧
      (a = 4) ∧ (b = 5) ∧ (c = 6) ∧
      (γ.cos = 1 / 8)) :=
sorry

end triangle_with_consecutive_sides_and_angle_property_l115_115938


namespace average_speed_approx_l115_115876

noncomputable def average_speed : ℝ :=
  let distance1 := 7
  let speed1 := 10
  let distance2 := 10
  let speed2 := 7
  let distance3 := 5
  let speed3 := 12
  let distance4 := 8
  let speed4 := 6
  let total_distance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let time4 := distance4 / speed4
  let total_time := time1 + time2 + time3 + time4
  total_distance / total_time

theorem average_speed_approx : abs (average_speed - 7.73) < 0.01 := by
  -- The necessary definitions fulfill the conditions and hence we put sorry here
  sorry

end average_speed_approx_l115_115876


namespace right_triangle_area_semi_perimeter_inequality_l115_115231

theorem right_triangle_area_semi_perimeter_inequality 
  (x y : ℝ) (h : x > 0 ∧ y > 0) 
  (p : ℝ := (x + y + Real.sqrt (x^2 + y^2)) / 2)
  (S : ℝ := x * y / 2) 
  (hypotenuse : ℝ := Real.sqrt (x^2 + y^2)) 
  (right_triangle : hypotenuse ^ 2 = x ^ 2 + y ^ 2) : 
  S <= p^2 / 5.5 := 
sorry

end right_triangle_area_semi_perimeter_inequality_l115_115231


namespace face_value_of_shares_l115_115847

theorem face_value_of_shares (investment : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) (dividend_received : ℝ) (F : ℝ)
  (h1 : investment = 14400)
  (h2 : premium_rate = 0.20)
  (h3 : dividend_rate = 0.06)
  (h4 : dividend_received = 720) :
  (1.20 * F = investment) ∧ (0.06 * F = dividend_received) ∧ (F = 12000) :=
by
  sorry

end face_value_of_shares_l115_115847


namespace base_subtraction_proof_l115_115253

def convert_base8_to_base10 (n : Nat) : Nat :=
  5 * 8^4 + 4 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1

def convert_base9_to_base10 (n : Nat) : Nat :=
  4 * 9^3 + 3 * 9^2 + 2 * 9^1 + 1

theorem base_subtraction_proof :
  convert_base8_to_base10 54321 - convert_base9_to_base10 4321 = 19559 :=
by
  sorry

end base_subtraction_proof_l115_115253


namespace find_range_of_a_l115_115399

-- Define the operation ⊗ on ℝ: x ⊗ y = x(1 - y)
def tensor (x y : ℝ) : ℝ := x * (1 - y)

-- State the inequality condition for all real numbers x
def inequality_condition (a : ℝ) : Prop :=
  ∀ (x : ℝ), tensor (x - a) (x + 1) < 1

-- State the theorem to prove the range of a
theorem find_range_of_a (a : ℝ) (h : inequality_condition a) : -2 < a ∧ a < 2 :=
  sorry

end find_range_of_a_l115_115399


namespace f_xh_sub_f_x_l115_115485

def f (x : ℝ) (k : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + k * x - 4

theorem f_xh_sub_f_x (x h : ℝ) (k : ℝ := -5) : 
    f (x + h) k - f x k = h * (6 * x^2 - 6 * x + 6 * x * h + 2 * h^2 - 3 * h - 5) := by
  sorry

end f_xh_sub_f_x_l115_115485


namespace max_product_sum_300_l115_115138

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l115_115138


namespace sally_picked_3_plums_l115_115082

theorem sally_picked_3_plums (melanie_picked : ℕ) (dan_picked : ℕ) (total_picked : ℕ) 
    (h1 : melanie_picked = 4) (h2 : dan_picked = 9) (h3 : total_picked = 16) : 
    total_picked - (melanie_picked + dan_picked) = 3 := 
by 
  -- proof steps go here
  sorry

end sally_picked_3_plums_l115_115082


namespace factor_quadratic_l115_115678

theorem factor_quadratic : ∀ (x : ℝ), 4 * x^2 - 20 * x + 25 = (2 * x - 5)^2 :=
by
  intro x
  sorry

end factor_quadratic_l115_115678


namespace find_length_AE_l115_115542

theorem find_length_AE (AB BC CD DE AC CE AE : ℕ) 
  (h1 : AB = 2) 
  (h2 : BC = 2) 
  (h3 : CD = 5) 
  (h4 : DE = 7)
  (h5 : AC > 2) 
  (h6 : AC < 4) 
  (h7 : CE > 2) 
  (h8 : CE < 5)
  (h9 : AC ≠ CE)
  (h10 : AC ≠ AE)
  (h11 : CE ≠ AE)
  : AE = 5 :=
sorry

end find_length_AE_l115_115542


namespace plate_729_driving_days_l115_115259

def plate (n : ℕ) : Prop := n >= 0 ∧ n <= 999

def monday (n : ℕ) : Prop := n % 2 = 1

def sum_digits (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 + d2 + d3

def tuesday (n : ℕ) : Prop := sum_digits n >= 11

def wednesday (n : ℕ) : Prop := n % 3 = 0

def thursday (n : ℕ) : Prop := sum_digits n <= 14

def count_digits (n : ℕ) : ℕ × ℕ × ℕ :=
  (n / 100, (n / 10) % 10, n % 10)

def friday (n : ℕ) : Prop :=
  let (d1, d2, d3) := count_digits n
  d1 = d2 ∨ d2 = d3 ∨ d1 = d3

def saturday (n : ℕ) : Prop := n < 500

def sunday (n : ℕ) : Prop := 
  let (d1, d2, d3) := count_digits n
  d1 <= 5 ∧ d2 <= 5 ∧ d3 <= 5

def can_drive (n : ℕ) (day : String) : Prop :=
  plate n ∧ 
  (day = "Monday" → monday n) ∧ 
  (day = "Tuesday" → tuesday n) ∧ 
  (day = "Wednesday" → wednesday n) ∧ 
  (day = "Thursday" → thursday n) ∧ 
  (day = "Friday" → friday n) ∧ 
  (day = "Saturday" → saturday n) ∧ 
  (day = "Sunday" → sunday n)

theorem plate_729_driving_days :
  can_drive 729 "Monday" ∧
  can_drive 729 "Tuesday" ∧
  can_drive 729 "Wednesday" ∧
  ¬ can_drive 729 "Thursday" ∧
  ¬ can_drive 729 "Friday" ∧
  ¬ can_drive 729 "Saturday" ∧
  ¬ can_drive 729 "Sunday" :=
by
  sorry

end plate_729_driving_days_l115_115259


namespace largest_integer_satisfying_l115_115111

theorem largest_integer_satisfying (x : ℤ) : 
  (∃ x, (2/7 : ℝ) < (x / 6 : ℝ) ∧ (x / 6 : ℝ) < 3/4) → x = 4 := 
by 
  sorry

end largest_integer_satisfying_l115_115111


namespace sum_of_b_and_c_base7_l115_115469

theorem sum_of_b_and_c_base7 (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
(h4 : A < 7) (h5 : B < 7) (h6 : C < 7) 
(h7 : 7^2 * A + 7 * B + C + 7^2 * B + 7 * C + A + 7^2 * C + 7 * A + B = 7^3 * A + 7^2 * A + 7 * A + 1) 
: B + C = 6 ∨ B + C = 12 := sorry

end sum_of_b_and_c_base7_l115_115469


namespace travel_time_seattle_to_lasvegas_l115_115695

def distance_seattle_boise : ℝ := 640
def distance_boise_saltlakecity : ℝ := 400
def distance_saltlakecity_phoenix : ℝ := 750
def distance_phoenix_lasvegas : ℝ := 300

def speed_highway_seattle_boise : ℝ := 80
def speed_city_seattle_boise : ℝ := 35

def speed_highway_boise_saltlakecity : ℝ := 65
def speed_city_boise_saltlakecity : ℝ := 25

def speed_highway_saltlakecity_denver : ℝ := 75
def speed_city_saltlakecity_denver : ℝ := 30

def speed_highway_denver_phoenix : ℝ := 70
def speed_city_denver_phoenix : ℝ := 20

def speed_highway_phoenix_lasvegas : ℝ := 50
def speed_city_phoenix_lasvegas : ℝ := 30

def city_distance_estimate : ℝ := 10

noncomputable def total_time : ℝ :=
  let time_seattle_boise := ((distance_seattle_boise - city_distance_estimate) / speed_highway_seattle_boise) + (city_distance_estimate / speed_city_seattle_boise)
  let time_boise_saltlakecity := ((distance_boise_saltlakecity - city_distance_estimate) / speed_highway_boise_saltlakecity) + (city_distance_estimate / speed_city_boise_saltlakecity)
  let time_saltlakecity_phoenix := ((distance_saltlakecity_phoenix - city_distance_estimate) / speed_highway_saltlakecity_denver) + (city_distance_estimate / speed_city_saltlakecity_denver)
  let time_phoenix_lasvegas := ((distance_phoenix_lasvegas - city_distance_estimate) / speed_highway_phoenix_lasvegas) + (city_distance_estimate / speed_city_phoenix_lasvegas)
  time_seattle_boise + time_boise_saltlakecity + time_saltlakecity_phoenix + time_phoenix_lasvegas

theorem travel_time_seattle_to_lasvegas :
  total_time = 30.89 :=
sorry

end travel_time_seattle_to_lasvegas_l115_115695


namespace ratio_of_intercepts_l115_115535

theorem ratio_of_intercepts (b s t : ℝ) (h1 : s = -2 * b / 5) (h2 : t = -3 * b / 7) :
  s / t = 14 / 15 :=
by
  sorry

end ratio_of_intercepts_l115_115535


namespace number_of_girls_in_colins_class_l115_115001

variables (g b : ℕ)

theorem number_of_girls_in_colins_class
  (h1 : g / b = 3 / 4)
  (h2 : g + b = 35)
  (h3 : b > 15) :
  g = 15 :=
sorry

end number_of_girls_in_colins_class_l115_115001


namespace part1_min_value_part2_max_value_k_lt_part2_max_value_k_geq_l115_115823

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem part1_min_value : ∀ (x : ℝ), x > 0 → f x ≥ -1 / Real.exp 1 := 
by sorry

noncomputable def g (x k : ℝ) : ℝ := f x - k * (x - 1)

theorem part2_max_value_k_lt : ∀ (k : ℝ), k < Real.exp 1 / (Real.exp 1 - 1) → 
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → g x k ≤ Real.exp 1 - k * Real.exp 1 + k :=
by sorry

theorem part2_max_value_k_geq : ∀ (k : ℝ), k ≥ Real.exp 1 / (Real.exp 1 - 1) → 
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → g x k ≤ 0 :=
by sorry

end part1_min_value_part2_max_value_k_lt_part2_max_value_k_geq_l115_115823


namespace train_length_is_correct_l115_115394

-- Define the given conditions and the expected result.
def train_speed_kmph : ℝ := 270
def time_seconds : ℝ := 5
def expected_length_meters : ℝ := 375

-- State the theorem to be proven.
theorem train_length_is_correct :
  (train_speed_kmph * 1000 / 3600) * time_seconds = expected_length_meters := by
  sorry -- Proof is not required, so we use 'sorry'

end train_length_is_correct_l115_115394


namespace at_least_one_nonnegative_l115_115095

theorem at_least_one_nonnegative (a b c d e f g h : ℝ) :
  ac + bd ≥ 0 ∨ ae + bf ≥ 0 ∨ ag + bh ≥ 0 ∨ ce + df ≥ 0 ∨ cg + dh ≥ 0 ∨ eg + fh ≥ 0 :=
sorry

end at_least_one_nonnegative_l115_115095


namespace distinct_patterns_4x4_3_shaded_l115_115243

def num_distinct_patterns (n : ℕ) (shading : ℕ) : ℕ :=
  if n = 4 ∧ shading = 3 then 15
  else 0 -- Placeholder for other cases, not relevant for our problem

theorem distinct_patterns_4x4_3_shaded :
  num_distinct_patterns 4 3 = 15 :=
by {
  -- The proof would go here
  sorry
}

end distinct_patterns_4x4_3_shaded_l115_115243


namespace count_valid_a_values_l115_115033

def is_perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

def valid_a_values (a : ℕ) : Prop :=
1 ≤ a ∧ a ≤ 100 ∧ is_perfect_square (16 * a + 9)

theorem count_valid_a_values :
  ∃ N : ℕ, N = Nat.card {a : ℕ | valid_a_values a} := sorry

end count_valid_a_values_l115_115033


namespace cannot_determine_position_l115_115357

-- Define the conditions
def east_longitude_122_north_latitude_43_6 : Prop := true
def row_6_seat_3_in_cinema : Prop := true
def group_1_in_classroom : Prop := false
def island_50_nautical_miles_north_northeast_another : Prop := true

-- Define the theorem
theorem cannot_determine_position :
  ¬ ((east_longitude_122_north_latitude_43_6 = false) ∧
     (row_6_seat_3_in_cinema = false) ∧
     (island_50_nautical_miles_north_northeast_another = false) ∧
     (group_1_in_classroom = true)) :=
by
  sorry

end cannot_determine_position_l115_115357


namespace find_M_coordinates_l115_115614

-- Definition of the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y ^ 2 = 2 * p * x

-- Definition to check if point M lies according to given conditions
def matchesCondition
  (p : ℝ) (M P O F : ℝ × ℝ) : Prop :=
  let xO := O.1
  let yO := O.2
  let xP := P.1
  let yP := P.2
  let xM := M.1
  let yM := M.2
  let xF := F.1
  let yF := F.2
  (xP = 2) ∧ (yP = 2 * p) ∧
  (xO = 0) ∧ (yO = 0) ∧
  (xF = p / 2) ∧ (yF = 0) ∧
  (Real.sqrt ((xM - xP) ^ 2 + (yM - yP) ^ 2) =
  Real.sqrt ((xM - xO) ^ 2 + (yM - yO) ^ 2)) ∧
  (Real.sqrt ((xM - xP) ^ 2 + (yM - yP) ^ 2) =
  Real.sqrt ((xM - xF) ^ 2 + (yM - yF) ^ 2))

-- Prove the coordinates of M satisfy the conditions
theorem find_M_coordinates :
  ∀ p : ℝ, p > 0 →
  matchesCondition p (1/4, 7/4) (2, 2 * p) (0, 0) (p / 2, 0) :=
by
  intros p hp
  simp [parabola, matchesCondition]
  sorry

end find_M_coordinates_l115_115614


namespace magnitude_z_is_sqrt_2_l115_115939

open Complex

noncomputable def z (x y : ℝ) : ℂ := x + y * I

theorem magnitude_z_is_sqrt_2 (x y : ℝ) (h1 : (2 * x) / (1 - I) = 1 + y * I) : abs (z x y) = Real.sqrt 2 :=
by
  -- You would fill in the proof steps here based on the problem's solution.
  sorry

end magnitude_z_is_sqrt_2_l115_115939


namespace prove_f_of_increasing_l115_115962

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def strictly_increasing_on_positives (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0

theorem prove_f_of_increasing {f : ℝ → ℝ}
  (h_odd : odd_function f)
  (h_incr : strictly_increasing_on_positives f) :
  f (-3) > f (-5) :=
by
  sorry

end prove_f_of_increasing_l115_115962


namespace lcm_of_two_numbers_l115_115448

-- Definitions based on the conditions
variable (a b l : ℕ)

-- The conditions from the problem
def hcf_ab : Nat := 9
def prod_ab : Nat := 1800

-- The main statement to prove
theorem lcm_of_two_numbers : Nat.lcm a b = 200 :=
by
  -- Skipping the proof implementation
  sorry

end lcm_of_two_numbers_l115_115448


namespace cannot_be_the_lengths_l115_115970

theorem cannot_be_the_lengths (x y z : ℝ) (h1 : x^2 + y^2 = 16) (h2 : x^2 + z^2 = 25) (h3 : y^2 + z^2 = 49) : false :=
by
  sorry

end cannot_be_the_lengths_l115_115970


namespace james_earnings_l115_115683

theorem james_earnings :
  let jan_earn : ℕ := 4000
  let feb_earn := 2 * jan_earn
  let total_earnings : ℕ := 18000
  let earnings_jan_feb := jan_earn + feb_earn
  let mar_earn := total_earnings - earnings_jan_feb
  (feb_earn - mar_earn) = 2000 := by
  sorry

end james_earnings_l115_115683


namespace sum_s_r_values_l115_115879

def r_values : List ℤ := [-2, -1, 0, 1, 3]
def r_range : List ℤ := [-1, 0, 1, 3, 5]

def s (x : ℤ) : ℤ := if 1 ≤ x then 2 * x + 1 else 0

theorem sum_s_r_values :
  (s 1) + (s 3) + (s 5) = 21 :=
by
  sorry

end sum_s_r_values_l115_115879


namespace topping_cost_l115_115961

noncomputable def cost_of_topping (ic_cost sundae_cost number_of_toppings: ℝ) : ℝ :=
(sundae_cost - ic_cost) / number_of_toppings

theorem topping_cost
  (ic_cost : ℝ)
  (sundae_cost : ℝ)
  (number_of_toppings : ℕ)
  (h_ic_cost : ic_cost = 2)
  (h_sundae_cost : sundae_cost = 7)
  (h_number_of_toppings : number_of_toppings = 10) :
  cost_of_topping ic_cost sundae_cost number_of_toppings = 0.5 :=
  by
  -- Proof will be here
  sorry

end topping_cost_l115_115961


namespace total_emails_in_april_is_675_l115_115634

-- Define the conditions
def daily_emails : ℕ := 20
def additional_emails : ℕ := 5
def april_days : ℕ := 30
def half_april_days : ℕ := april_days / 2

-- Define the total number of emails received
def total_emails : ℕ :=
  (daily_emails * half_april_days) +
  ((daily_emails + additional_emails) * half_april_days)

-- Define the statement to be proven
theorem total_emails_in_april_is_675 : total_emails = 675 :=
  by
  sorry

end total_emails_in_april_is_675_l115_115634


namespace radius_of_circle_with_square_and_chord_l115_115771

theorem radius_of_circle_with_square_and_chord :
  ∃ (r : ℝ), 
    (∀ (chord_length square_side_length : ℝ), chord_length = 6 ∧ square_side_length = 2 → 
    (r = Real.sqrt 10)) :=
by
  sorry

end radius_of_circle_with_square_and_chord_l115_115771


namespace range_of_f_l115_115507

noncomputable def f (x : ℝ) : ℝ := (Real.arccos x) ^ 3 + (Real.arcsin x) ^ 3

theorem range_of_f : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 
           ∃ y : ℝ, y = f x ∧ (y ≥ (Real.pi ^ 3) / 32) ∧ (y ≤ (7 * (Real.pi ^ 3)) / 8) :=
sorry

end range_of_f_l115_115507


namespace solve_quadratic1_solve_quadratic2_l115_115009

theorem solve_quadratic1 (x : ℝ) :
  x^2 + 10 * x + 16 = 0 ↔ (x = -2 ∨ x = -8) :=
by
  sorry

theorem solve_quadratic2 (x : ℝ) :
  x * (x + 4) = 8 * x + 12 ↔ (x = -2 ∨ x = 6) :=
by
  sorry

end solve_quadratic1_solve_quadratic2_l115_115009


namespace angle_B_l115_115131

theorem angle_B (a b c A B : ℝ) (h : a * Real.cos B - b * Real.cos A = c) (C : ℝ) (hC : C = Real.pi / 5) (h_triangle : A + B + C = Real.pi) : B = 3 * Real.pi / 10 :=
sorry

end angle_B_l115_115131


namespace find_x_cube_plus_reciprocal_cube_l115_115066

variable {x : ℝ}

theorem find_x_cube_plus_reciprocal_cube (hx : x + 1/x = 10) : x^3 + 1/x^3 = 970 :=
sorry

end find_x_cube_plus_reciprocal_cube_l115_115066


namespace vector_expression_result_l115_115773

structure Vector2 :=
(x : ℝ)
(y : ℝ)

def vector_dot_product (v1 v2 : Vector2) : ℝ :=
  v1.x * v1.y + v2.x * v2.y

def vector_scalar_mul (c : ℝ) (v : Vector2) : Vector2 :=
  { x := c * v.x, y := c * v.y }

def vector_sub (v1 v2 : Vector2) : Vector2 :=
  { x := v1.x - v2.x, y := v1.y - v2.y }

noncomputable def a : Vector2 := { x := 2, y := -1 }
noncomputable def b : Vector2 := { x := 3, y := -2 }

theorem vector_expression_result :
  vector_dot_product
    (vector_sub (vector_scalar_mul 3 a) b)
    (vector_sub a (vector_scalar_mul 2 b)) = -15 := by
  sorry

end vector_expression_result_l115_115773


namespace harriet_siblings_product_l115_115158

-- Definitions based on conditions
def Harry_sisters : ℕ := 6
def Harry_brothers : ℕ := 3
def Harriet_sisters : ℕ := Harry_sisters - 1
def Harriet_brothers : ℕ := Harry_brothers

-- Statement to prove
theorem harriet_siblings_product : Harriet_sisters * Harriet_brothers = 15 := by
  -- Proof is skipped
  sorry

end harriet_siblings_product_l115_115158


namespace triangle_two_acute_angles_l115_115652

theorem triangle_two_acute_angles (A B C : ℝ) (h_triangle : A + B + C = 180) (h_pos : A > 0 ∧ B > 0 ∧ C > 0)
  (h_acute_triangle: A < 90 ∨ B < 90 ∨ C < 90): A < 90 ∧ B < 90 ∨ A < 90 ∧ C < 90 ∨ B < 90 ∧ C < 90 :=
by
  sorry

end triangle_two_acute_angles_l115_115652


namespace find_v_plus_z_l115_115656

variable (x u v w z : ℂ)
variable (y : ℂ)
variable (condition1 : y = 2)
variable (condition2 : w = -x - u)
variable (condition3 : x + y * Complex.I + u + v * Complex.I + w + z * Complex.I = -2 * Complex.I)

theorem find_v_plus_z : v + z = -4 :=
by
  have h1 : y = 2 := condition1
  have h2 : w = -x - u := condition2
  have h3 : x + y * Complex.I + u + v * Complex.I + w + z * Complex.I = -2 * Complex.I := condition3
  sorry

end find_v_plus_z_l115_115656


namespace polynomial_satisfies_conditions_l115_115182

noncomputable def f (x y z : ℝ) : ℝ :=
  (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

theorem polynomial_satisfies_conditions :
  (f x (z^2) y + f x (y^2) z = 0) ∧ (f (z^3) y x + f (x^3) y z = 0) :=
by
  sorry

end polynomial_satisfies_conditions_l115_115182


namespace sin_minus_cos_eq_l115_115228

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l115_115228


namespace correct_average_marks_l115_115959

def incorrect_average := 100
def number_of_students := 10
def incorrect_mark := 60
def correct_mark := 10
def difference := incorrect_mark - correct_mark
def incorrect_total := incorrect_average * number_of_students
def correct_total := incorrect_total - difference

theorem correct_average_marks : correct_total / number_of_students = 95 := by
  sorry

end correct_average_marks_l115_115959


namespace area_enclosed_curve_l115_115633

-- The proof statement
theorem area_enclosed_curve (x y : ℝ) : (x^2 + y^2 = 2 * (|x| + |y|)) → 
  (area_of_enclosed_region = 2 * π + 8) :=
sorry

end area_enclosed_curve_l115_115633


namespace find_quotient_l115_115550

theorem find_quotient (dividend divisor remainder quotient : ℕ) 
  (h1 : dividend = 23) (h2 : divisor = 4) (h3 : remainder = 3)
  (h4 : dividend = (divisor * quotient) + remainder) : quotient = 5 :=
sorry

end find_quotient_l115_115550


namespace correct_money_calculation_l115_115990

structure BootSale :=
(initial_money : ℕ)
(price_per_boot : ℕ)
(total_taken : ℕ)
(total_returned : ℕ)
(money_spent : ℕ)
(remaining_money_to_return : ℕ)

theorem correct_money_calculation (bs : BootSale) :
  bs.initial_money = 25 →
  bs.price_per_boot = 12 →
  bs.total_taken = 25 →
  bs.total_returned = 5 →
  bs.money_spent = 3 →
  bs.remaining_money_to_return = 2 →
  bs.total_taken - bs.total_returned + bs.money_spent = 23 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end correct_money_calculation_l115_115990


namespace necessary_and_sufficient_condition_l115_115466

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 0 < x → x + (1 / x) > a) ↔ a < 2 :=
sorry

end necessary_and_sufficient_condition_l115_115466


namespace eduardo_frankie_classes_total_l115_115421

theorem eduardo_frankie_classes_total (eduardo_classes : ℕ) (h₁ : eduardo_classes = 3) 
                                       (h₂ : ∀ frankie_classes, frankie_classes = 2 * eduardo_classes) :
  ∃ total_classes : ℕ, total_classes = eduardo_classes + 2 * eduardo_classes := 
by
  use 3 + 2 * 3
  sorry

end eduardo_frankie_classes_total_l115_115421


namespace time_A_to_complete_work_alone_l115_115374

theorem time_A_to_complete_work_alone :
  ∃ (x : ℝ), (1 / x) + (1 / 20) = (1 / 8.571428571428571) ∧ x = 15 :=
by
  sorry

end time_A_to_complete_work_alone_l115_115374


namespace necessary_french_woman_l115_115505

structure MeetingConditions where
  total_money_women : ℝ
  total_money_men : ℝ
  total_money_french : ℝ
  total_money_russian : ℝ

axiom no_other_representatives : Prop
axiom money_french_vs_russian (conditions : MeetingConditions) : conditions.total_money_french > conditions.total_money_russian
axiom money_women_vs_men (conditions : MeetingConditions) : conditions.total_money_women > conditions.total_money_men

theorem necessary_french_woman (conditions : MeetingConditions) :
  ∃ w_f : ℝ, w_f > 0 ∧ conditions.total_money_french > w_f ∧ w_f + conditions.total_money_men > conditions.total_money_women :=
by
  sorry

end necessary_french_woman_l115_115505


namespace perfect_square_expression_l115_115047

theorem perfect_square_expression (n : ℕ) (h : 7 ≤ n) : ∃ k : ℤ, (n + 2) ^ 2 = k ^ 2 :=
by 
  sorry

end perfect_square_expression_l115_115047


namespace divide_decimals_l115_115436

theorem divide_decimals : (0.24 / 0.006) = 40 := by
  sorry

end divide_decimals_l115_115436


namespace difference_in_students_and_guinea_pigs_l115_115713

def num_students (classrooms : ℕ) (students_per_classroom : ℕ) : ℕ := classrooms * students_per_classroom
def num_guinea_pigs (classrooms : ℕ) (guinea_pigs_per_classroom : ℕ) : ℕ := classrooms * guinea_pigs_per_classroom
def difference_students_guinea_pigs (students : ℕ) (guinea_pigs : ℕ) : ℕ := students - guinea_pigs

theorem difference_in_students_and_guinea_pigs :
  ∀ (classrooms : ℕ) (students_per_classroom : ℕ) (guinea_pigs_per_classroom : ℕ),
  classrooms = 6 →
  students_per_classroom = 24 →
  guinea_pigs_per_classroom = 3 →
  difference_students_guinea_pigs (num_students classrooms students_per_classroom) (num_guinea_pigs classrooms guinea_pigs_per_classroom) = 126 :=
by
  intros
  sorry

end difference_in_students_and_guinea_pigs_l115_115713


namespace original_numbers_product_l115_115173

theorem original_numbers_product (a b c d x : ℕ) 
  (h1 : a + b + c + d = 243)
  (h2 : a + 8 = x)
  (h3 : b - 8 = x)
  (h4 : c * 8 = x)
  (h5 : d / 8 = x) : 
  (min (min a (min b (min c d))) * max a (max b (max c d))) = 576 :=
by 
  sorry

end original_numbers_product_l115_115173


namespace first_six_divisors_l115_115025

theorem first_six_divisors (a b : ℤ) (h : 5 * b = 14 - 3 * a) : 
  ∃ n, n = 5 ∧ ∀ k ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ), (3 * b + 18) % k = 0 ↔ k ∈ ({1, 2, 3, 5, 6} : Finset ℕ) :=
by
  sorry

end first_six_divisors_l115_115025


namespace function_periodicity_even_l115_115218

theorem function_periodicity_even (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_period : ∀ x : ℝ, x ≥ 0 → f (x + 2) = -f x)
  (h_def : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2^x - 1) :
  f (-2017) + f 2018 = 1 :=
sorry

end function_periodicity_even_l115_115218


namespace quadratic_integers_pairs_l115_115116

theorem quadratic_integers_pairs (m n : ℕ) :
  (0 < m ∧ m < 9) ∧ (0 < n ∧ n < 9) ∧ (m^2 > 9 * n) ↔ ((m = 4 ∧ n = 1) ∨ (m = 5 ∧ n = 2)) :=
by {
  -- Insert proof here
  sorry
}

end quadratic_integers_pairs_l115_115116


namespace bruce_bank_savings_l115_115765

def aunt_gift : ℕ := 75
def grandfather_gift : ℕ := 150
def total_gift : ℕ := aunt_gift + grandfather_gift
def fraction_saved : ℚ := 1/5
def amount_saved : ℚ := total_gift * fraction_saved

theorem bruce_bank_savings : amount_saved = 45 := by
  sorry

end bruce_bank_savings_l115_115765


namespace initial_hamburgers_correct_l115_115373

-- Define the initial problem conditions
def initial_hamburgers (H : ℝ) : Prop := H + 3.0 = 12

-- State the proof problem
theorem initial_hamburgers_correct (H : ℝ) (h : initial_hamburgers H) : H = 9.0 :=
sorry

end initial_hamburgers_correct_l115_115373


namespace real_roots_of_polynomial_l115_115235

theorem real_roots_of_polynomial :
  (∀ x : ℝ, (x^10 + 36 * x^6 + 13 * x^2 = 13 * x^8 + x^4 + 36) ↔ 
    (x = 1 ∨ x = -1 ∨ x = 3 ∨ x = -3 ∨ x = 2 ∨ x = -2)) :=
by 
  sorry

end real_roots_of_polynomial_l115_115235


namespace students_play_neither_l115_115442

-- Define the conditions
def total_students : ℕ := 39
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def both_players : ℕ := 17

-- Define a theorem that states the equivalent proof problem
theorem students_play_neither : 
  total_students - (football_players + long_tennis_players - both_players) = 10 := by
  sorry

end students_play_neither_l115_115442


namespace functional_equation_solution_l115_115687

theorem functional_equation_solution :
  ∀ (f : ℚ → ℝ), (∀ x y : ℚ, f (x + y) = f x + f y + 2 * x * y) →
  ∃ k : ℝ, ∀ x : ℚ, f x = x^2 + k * x :=
by
  sorry

end functional_equation_solution_l115_115687


namespace molecular_weight_N2O_correct_l115_115637

-- Define the atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of N2O
def molecular_weight_N2O : ℝ := 2 * atomic_weight_N + atomic_weight_O

-- Prove the statement
theorem molecular_weight_N2O_correct : molecular_weight_N2O = 44.02 := by
  -- We leave the proof as an exercise (or assumption)
  sorry

end molecular_weight_N2O_correct_l115_115637


namespace bobby_total_l115_115176

-- Define the conditions
def initial_candy : ℕ := 33
def additional_candy : ℕ := 4
def chocolate : ℕ := 14

-- Define the total pieces of candy Bobby ate
def total_candy : ℕ := initial_candy + additional_candy

-- Define the total pieces of candy and chocolate Bobby ate
def total_candy_and_chocolate : ℕ := total_candy + chocolate

-- Theorem to prove the total pieces of candy and chocolate Bobby ate
theorem bobby_total : total_candy_and_chocolate = 51 :=
by sorry

end bobby_total_l115_115176


namespace jungkook_has_smallest_collection_l115_115645

-- Define the collections
def yoongi_collection : ℕ := 7
def jungkook_collection : ℕ := 6
def yuna_collection : ℕ := 9

-- State the theorem
theorem jungkook_has_smallest_collection : 
  jungkook_collection = min yoongi_collection (min jungkook_collection yuna_collection) := 
by
  sorry

end jungkook_has_smallest_collection_l115_115645


namespace music_store_cellos_l115_115468

/-- 
A certain music store stocks 600 violas. 
There are 100 cello-viola pairs, such that a cello and a viola were both made with wood from the same tree. 
The probability that the two instruments are made with wood from the same tree is 0.00020833333333333335. 
Prove that the store stocks 800 cellos.
-/
theorem music_store_cellos (V : ℕ) (P : ℕ) (Pr : ℚ) (C : ℕ) 
  (h1 : V = 600) 
  (h2 : P = 100) 
  (h3 : Pr = 0.00020833333333333335) 
  (h4 : Pr = P / (C * V)): C = 800 :=
by
  sorry

end music_store_cellos_l115_115468


namespace pages_per_day_l115_115558

def notebooks : Nat := 5
def pages_per_notebook : Nat := 40
def total_days : Nat := 50

theorem pages_per_day (H1 : notebooks = 5) (H2 : pages_per_notebook = 40) (H3 : total_days = 50) : 
  (notebooks * pages_per_notebook / total_days) = 4 := by
  sorry

end pages_per_day_l115_115558


namespace exists_x0_lt_l115_115980

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d
noncomputable def Q (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem exists_x0_lt {a b c d p q r s : ℝ} (h1 : r < s) (h2 : s - r > 2)
  (h3 : ∀ x, r < x ∧ x < s → P x a b c d < 0 ∧ Q x p q < 0)
  (h4 : ∀ x, x < r ∨ x > s → P x a b c d >= 0 ∧ Q x p q >= 0) :
  ∃ x0, r < x0 ∧ x0 < s ∧ P x0 a b c d < Q x0 p q :=
sorry

end exists_x0_lt_l115_115980


namespace fraction_between_stops_l115_115315

/-- Prove that the fraction of the remaining distance traveled between Maria's first and second stops is 1/4. -/
theorem fraction_between_stops (total_distance first_stop_distance remaining_distance final_leg_distance : ℝ)
  (h_total : total_distance = 400)
  (h_first_stop : first_stop_distance = total_distance / 2)
  (h_remaining : remaining_distance = total_distance - first_stop_distance)
  (h_final_leg : final_leg_distance = 150)
  (h_second_leg : remaining_distance - final_leg_distance = 50) :
  50 / remaining_distance = 1 / 4 :=
by
  { sorry }

end fraction_between_stops_l115_115315


namespace arithmetic_sequence_a_find_p_q_find_c_minus_a_find_y_values_l115_115668

-- Problem (a)
theorem arithmetic_sequence_a (x1 x2 x3 x4 x5 : ℕ) (h : (x1 = 2 ∧ x2 = 5 ∧ x3 = 10 ∧ x4 = 13 ∧ x5 = 15)) : 
  ∃ a b c, (a = 5 ∧ b = 10 ∧ c = 15 ∧ b - a = c - b ∧ b - a > 0) := 
sorry

-- Problem (b)
theorem find_p_q (p q : ℕ) (h : ∃ d, (7 - p = d ∧ q - 7 = d ∧ 13 - q = d)) : 
  p = 4 ∧ q = 10 :=
sorry

-- Problem (c)
theorem find_c_minus_a (a b c : ℕ) (h : ∃ d, (b - a = d ∧ c - b = d ∧ (a + 21) - c = d)) :
  c - a = 14 :=
sorry

-- Problem (d)
theorem find_y_values (y : ℤ) (h : ∃ d, ((2*y + 3) - (y - 6) = d ∧ (y*y + 2) - (2*y + 3) = d) ) :
  y = 5 ∨ y = -2 :=
sorry

end arithmetic_sequence_a_find_p_q_find_c_minus_a_find_y_values_l115_115668


namespace non_square_solution_equiv_l115_115519

theorem non_square_solution_equiv 
  (a b : ℤ) (h1 : ¬∃ k : ℤ, a = k^2) (h2 : ¬∃ k : ℤ, b = k^2) :
  (∃ x y z w : ℤ, x^2 - a * y^2 - b * z^2 + a * b * w^2 = 0 ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) ↔
  (∃ x y z : ℤ, x^2 - a * y^2 - b * z^2 = 0 ∧ (x, y, z) ≠ (0, 0, 0)) :=
by sorry

end non_square_solution_equiv_l115_115519


namespace jelly_bean_problem_l115_115118

variable (b c : ℕ)

theorem jelly_bean_problem (h1 : b = 3 * c) (h2 : b - 15 = 4 * (c - 15)) : b = 135 :=
sorry

end jelly_bean_problem_l115_115118


namespace ions_electron_shell_structure_l115_115022

theorem ions_electron_shell_structure
  (a b n m : ℤ) 
  (same_electron_shell_structure : a + n = b - m) :
  a + m = b - n :=
by
  sorry

end ions_electron_shell_structure_l115_115022


namespace katie_new_games_l115_115136

theorem katie_new_games (K : ℕ) (h : K + 8 = 92) : K = 84 :=
by
  sorry

end katie_new_games_l115_115136


namespace sequence_general_formula_l115_115783

theorem sequence_general_formula (a : ℕ → ℚ) (h₁ : a 1 = 2 / 3)
  (h₂ : ∀ n : ℕ, a (n + 1) = a n + a n * a (n + 1)) : 
  ∀ n : ℕ, a n = 2 / (5 - 2 * n) :=
by 
  sorry

end sequence_general_formula_l115_115783


namespace daniel_initial_noodles_l115_115016

variable (give : ℕ)
variable (left : ℕ)
variable (initial : ℕ)

theorem daniel_initial_noodles (h1 : give = 12) (h2 : left = 54) (h3 : initial = left + give) : initial = 66 := by
  sorry

end daniel_initial_noodles_l115_115016


namespace binary_1101_to_decimal_l115_115677

theorem binary_1101_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 13 := by
  -- To convert a binary number to its decimal equivalent, we multiply each digit by its corresponding power of 2 based on its position and then sum the results.
  sorry

end binary_1101_to_decimal_l115_115677


namespace composite_rate_proof_l115_115417

noncomputable def composite_rate (P A : ℝ) (T : ℕ) (X Y Z : ℝ) (R : ℝ) : Prop :=
  let factor := (1 + X / 100) * (1 + Y / 100) * (1 + Z / 100)
  1.375 = factor ∧ (A = P * (1 + R / 100) ^ T)

theorem composite_rate_proof :
  composite_rate 4000 5500 3 X Y Z 11.1 :=
by sorry

end composite_rate_proof_l115_115417


namespace polygon_side_intersections_l115_115350

theorem polygon_side_intersections :
  let m6 := 6
  let m7 := 7
  let m8 := 8
  let m9 := 9
  let pairs := [(m6, m7), (m6, m8), (m6, m9), (m7, m8), (m7, m9), (m8, m9)]
  let count_intersections (m n : ℕ) : ℕ := 2 * min m n
  let total_intersections := pairs.foldl (fun total pair => total + count_intersections pair.1 pair.2) 0
  total_intersections = 80 :=
by
  sorry

end polygon_side_intersections_l115_115350


namespace common_face_sum_is_9_l115_115203

noncomputable def common_sum (vertices : Fin 9 → ℕ) : ℕ :=
  let total_sum := (Finset.sum (Finset.univ : Finset (Fin 9)) vertices)
  let additional_sum := 9
  let total_with_addition := total_sum + additional_sum
  total_with_addition / 6

theorem common_face_sum_is_9 :
  ∀ (vertices : Fin 9 → ℕ), (∀ i, 1 ≤ vertices i ∧ vertices i ≤ 9) →
  Finset.sum (Finset.univ : Finset (Fin 9)) vertices = 45 →
  common_sum vertices = 9 := 
by
  intros vertices h1 h_sum
  unfold common_sum
  sorry

end common_face_sum_is_9_l115_115203


namespace molecular_weight_H2O_7_moles_l115_115260

noncomputable def atomic_weight_H : ℝ := 1.008
noncomputable def atomic_weight_O : ℝ := 16.00
noncomputable def num_atoms_H_in_H2O : ℝ := 2
noncomputable def num_atoms_O_in_H2O : ℝ := 1
noncomputable def moles_H2O : ℝ := 7

theorem molecular_weight_H2O_7_moles :
  (num_atoms_H_in_H2O * atomic_weight_H + num_atoms_O_in_H2O * atomic_weight_O) * moles_H2O = 126.112 := by
  sorry

end molecular_weight_H2O_7_moles_l115_115260


namespace difference_cubics_divisible_by_24_l115_115508

theorem difference_cubics_divisible_by_24 
    (a b : ℤ) (h : ∃ k : ℤ, a - b = 3 * k) : 
    ∃ k : ℤ, (2 * a + 1)^3 - (2 * b + 1)^3 = 24 * k :=
by
  sorry

end difference_cubics_divisible_by_24_l115_115508


namespace expand_and_simplify_l115_115539

theorem expand_and_simplify (x : ℝ) : (2*x + 6)*(x + 9) = 2*x^2 + 24*x + 54 :=
by
  sorry

end expand_and_simplify_l115_115539


namespace find_lambda_l115_115971

variables {a b : ℝ} (lambda : ℝ)

-- Conditions
def orthogonal (x y : ℝ) : Prop := x * y = 0
def magnitude_a : ℝ := 2
def magnitude_b : ℝ := 3
def is_perpendicular (x y : ℝ) : Prop := x * y = 0

-- Proof statement
theorem find_lambda (h₁ : orthogonal a b)
  (h₂ : magnitude_a = 2)
  (h₃ : magnitude_b = 3)
  (h₄ : is_perpendicular (3 * a + 2 * b) (lambda * a - b)) :
  lambda = 3 / 2 :=
sorry

end find_lambda_l115_115971


namespace correct_decimal_multiplication_l115_115706

theorem correct_decimal_multiplication : 0.085 * 3.45 = 0.29325 := 
by 
  sorry

end correct_decimal_multiplication_l115_115706


namespace curve_symmetry_l115_115590

theorem curve_symmetry :
  ∃ θ : ℝ, θ = 5 * Real.pi / 6 ∧
  ∀ (ρ θ' : ℝ), ρ = 4 * Real.sin (θ' - Real.pi / 3) ↔ ρ = 4 * Real.sin ((θ - θ') - Real.pi / 3) :=
sorry

end curve_symmetry_l115_115590


namespace am_gm_inequality_l115_115477

variable (a : ℝ) (h : a > 0) -- Variables and condition

theorem am_gm_inequality (a : ℝ) (h : a > 0) : a + 1 / a ≥ 2 := 
sorry -- Proof is not provided according to instructions.

end am_gm_inequality_l115_115477


namespace hens_count_l115_115641

theorem hens_count (H C : ℕ) (h1 : H + C = 46) (h2 : 2 * H + 4 * C = 140) : H = 22 :=
by
  sorry

end hens_count_l115_115641


namespace outfit_count_correct_l115_115692

def total_shirts : ℕ := 8
def total_pants : ℕ := 4
def total_hats : ℕ := 6
def shirt_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def hat_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def conflict_free_outfits (total_shirts total_pants total_hats : ℕ) : ℕ :=
  let total_outfits := total_shirts * total_pants * total_hats
  let matching_outfits := (2 * 1 * 4) * total_pants
  total_outfits - matching_outfits

theorem outfit_count_correct :
  conflict_free_outfits total_shirts total_pants total_hats = 160 :=
by
  unfold conflict_free_outfits
  norm_num
  sorry

end outfit_count_correct_l115_115692


namespace correct_option_l115_115226

-- Define the options as propositions
def OptionA (a : ℕ) := a ^ 3 * a ^ 5 = a ^ 15
def OptionB (a : ℕ) := a ^ 8 / a ^ 2 = a ^ 4
def OptionC (a : ℕ) := a ^ 2 + a ^ 3 = a ^ 5
def OptionD (a : ℕ) := 3 * a - a = 2 * a

-- Prove that Option D is the only correct statement
theorem correct_option (a : ℕ) : OptionD a ∧ ¬OptionA a ∧ ¬OptionB a ∧ ¬OptionC a :=
by
  sorry

end correct_option_l115_115226


namespace possible_sets_B_l115_115562

def A : Set ℤ := {-1}

def isB (B : Set ℤ) : Prop :=
  A ∪ B = {-1, 3}

theorem possible_sets_B : ∀ B : Set ℤ, isB B → B = {3} ∨ B = {-1, 3} :=
by
  intros B hB
  sorry

end possible_sets_B_l115_115562


namespace distinct_terms_count_l115_115074

/-!
  Proving the number of distinct terms in the expansion of (x + 2y)^12
-/

theorem distinct_terms_count (x y : ℕ) : 
  (x + 2 * y) ^ 12 = 13 :=
by sorry

end distinct_terms_count_l115_115074


namespace cuboid_surface_area_correct_l115_115271

-- Define the dimensions of the cuboid
def l : ℕ := 4
def w : ℕ := 5
def h : ℕ := 6

-- Define the function to calculate the surface area of the cuboid
def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + w * h + h * l)

-- The theorem stating that the surface area of the cuboid is 148 cm²
theorem cuboid_surface_area_correct : surface_area l w h = 148 := by
  sorry

end cuboid_surface_area_correct_l115_115271


namespace fran_travel_time_l115_115546

theorem fran_travel_time (joann_speed fran_speed : ℝ) (joann_time joann_distance : ℝ) :
  joann_speed = 15 → joann_time = 4 → joann_distance = joann_speed * joann_time →
  fran_speed = 20 → fran_time = joann_distance / fran_speed →
  fran_time = 3 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end fran_travel_time_l115_115546


namespace sum_of_angles_FC_correct_l115_115268

noncomputable def circleGeometry (A B C D E F : Point)
  (onCircle : ∀ (P : Point), P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E)
  (arcAB : ℝ) (arcDE : ℝ) : Prop :=
  let arcFull := 360;
  let angleF := 6;  -- Derived from the intersecting chords theorem
  let angleC := 36; -- Derived from the inscribed angle theorem
  arcAB = 60 ∧ arcDE = 72 ∧
  0 ≤ angleF ∧ 0 ≤ angleC ∧
  angleF + angleC = 42

theorem sum_of_angles_FC_correct (A B C D E F : Point) 
  (onCircle : ∀ (P : Point), P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E) :
  circleGeometry A B C D E F onCircle 60 72 :=
by
  sorry  -- Proof to be filled

end sum_of_angles_FC_correct_l115_115268


namespace cakes_sold_to_baked_ratio_l115_115813

theorem cakes_sold_to_baked_ratio
  (cakes_per_day : ℕ) 
  (days : ℕ)
  (cakes_left : ℕ)
  (total_cakes : ℕ := cakes_per_day * days)
  (cakes_sold : ℕ := total_cakes - cakes_left) :
  cakes_per_day = 20 → 
  days = 9 → 
  cakes_left = 90 → 
  cakes_sold * 2 = total_cakes := 
by 
  intros 
  sorry

end cakes_sold_to_baked_ratio_l115_115813


namespace sqrt_range_l115_115250

theorem sqrt_range (x : ℝ) : 3 - 2 * x ≥ 0 ↔ x ≤ 3 / 2 := 
    sorry

end sqrt_range_l115_115250


namespace y_is_multiple_of_3_y_is_multiple_of_9_y_is_multiple_of_27_y_is_multiple_of_81_l115_115571

noncomputable def y : ℕ := 81 + 243 + 729 + 1458 + 2187 + 6561 + 19683

theorem y_is_multiple_of_3 : y % 3 = 0 :=
sorry

theorem y_is_multiple_of_9 : y % 9 = 0 :=
sorry

theorem y_is_multiple_of_27 : y % 27 = 0 :=
sorry

theorem y_is_multiple_of_81 : y % 81 = 0 :=
sorry

end y_is_multiple_of_3_y_is_multiple_of_9_y_is_multiple_of_27_y_is_multiple_of_81_l115_115571


namespace ant_impossibility_l115_115862

-- Define the vertices and edges of a cube
structure Cube :=
(vertices : Finset ℕ) -- Representing a finite set of vertices
(edges : Finset (ℕ × ℕ)) -- Representing a finite set of edges between vertices
(valid_edge : ∀ e ∈ edges, ∃ v1 v2, (v1, v2) = e ∨ (v2, v1) = e)
(starting_vertex : ℕ)

-- Ant behavior on the cube
structure AntOnCube (C : Cube) :=
(is_path_valid : List ℕ → Prop) -- A property that checks the path is valid

-- Problem conditions translated: 
-- No retracing and specific visit numbers
noncomputable def ant_problem (C : Cube) (A : AntOnCube C) : Prop :=
  ∀ (path : List ℕ), A.is_path_valid path → ¬ (
    (path.count C.starting_vertex = 25) ∧ 
    (∀ v ∈ C.vertices, v ≠ C.starting_vertex → path.count v = 20)
  )

-- The final theorem statement
theorem ant_impossibility (C : Cube) (A : AntOnCube C) : ant_problem C A :=
by
  -- providing the theorem framework; proof omitted with sorry
  sorry

end ant_impossibility_l115_115862


namespace ratio_of_cube_volumes_l115_115997

theorem ratio_of_cube_volumes (a b : ℕ) (ha : a = 10) (hb : b = 25) :
  (a^3 : ℚ) / (b^3 : ℚ) = 8 / 125 := by
  sorry

end ratio_of_cube_volumes_l115_115997


namespace MeganSavingsExceed500_l115_115855

theorem MeganSavingsExceed500 :
  ∃ n : ℕ, n ≥ 7 ∧ ((3^n - 1) / 2 > 500) :=
sorry

end MeganSavingsExceed500_l115_115855


namespace sequence_property_l115_115973

theorem sequence_property (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h_rec : ∀ m n : ℕ, a (m + n) = a m + a n + m * n) :
  a 10 = 55 :=
sorry

end sequence_property_l115_115973


namespace polynomial_has_root_of_multiplicity_2_l115_115112

theorem polynomial_has_root_of_multiplicity_2 (r s k : ℝ)
  (h1 : x^3 + k * x - 128 = (x - r)^2 * (x - s)) -- polynomial has a root of multiplicity 2
  (h2 : -2 * r - s = 0)                         -- relationship from coefficient of x²
  (h3 : r^2 + 2 * r * s = k)                    -- relationship from coefficient of x
  (h4 : r^2 * s = 128)                          -- relationship from constant term
  : k = -48 := 
sorry

end polynomial_has_root_of_multiplicity_2_l115_115112


namespace root_quadratic_expression_value_l115_115599

theorem root_quadratic_expression_value (m : ℝ) (h : m^2 - m - 3 = 0) : 2023 - m^2 + m = 2020 := 
by 
  sorry

end root_quadratic_expression_value_l115_115599


namespace inequality_sol_set_a_eq_2_inequality_sol_set_general_l115_115045

theorem inequality_sol_set_a_eq_2 :
  ∀ x : ℝ, (x^2 - x + 2 - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 2) :=
by sorry

theorem inequality_sol_set_general (a : ℝ) :
  (∀ x : ℝ, (x^2 - x + a - a^2 ≤ 0) ↔
    (if a < 1/2 then a ≤ x ∧ x ≤ 1 - a
    else if a > 1/2 then 1 - a ≤ x ∧ x ≤ a
    else x = 1/2)) :=
by sorry

end inequality_sol_set_a_eq_2_inequality_sol_set_general_l115_115045


namespace cost_of_patent_is_correct_l115_115254

-- Defining the conditions
def c_parts : ℕ := 3600
def p : ℕ := 180
def n : ℕ := 45

-- Calculation of total revenue
def total_revenue : ℕ := n * p

-- Calculation of cost of patent
def cost_of_patent (total_revenue c_parts : ℕ) : ℕ := total_revenue - c_parts

-- The theorem to be proved
theorem cost_of_patent_is_correct (R : ℕ) (H : R = total_revenue) : cost_of_patent R c_parts = 4500 :=
by
  -- this is where your proof will go
  sorry

end cost_of_patent_is_correct_l115_115254


namespace seq_a3_eq_1_l115_115927

theorem seq_a3_eq_1 (a : ℕ → ℤ) (h₁ : ∀ n ≥ 1, a (n + 1) = a n - 3) (h₂ : a 1 = 7) : a 3 = 1 :=
by
  sorry

end seq_a3_eq_1_l115_115927


namespace total_cost_correct_l115_115264

-- Defining the conditions
def charges_per_week : ℕ := 3
def weeks_per_year : ℕ := 52
def cost_per_charge : ℝ := 0.78

-- Defining the total cost proof statement
theorem total_cost_correct : (charges_per_week * weeks_per_year : ℝ) * cost_per_charge = 121.68 :=
by
  sorry

end total_cost_correct_l115_115264


namespace find_AB_value_l115_115068

theorem find_AB_value :
  ∃ A B : ℕ, (A + B = 5 ∧ (A - B) % 11 = 5 % 11) ∧
           990 * 991 * 992 * 993 = 966428 * 100000 + A * 9100 + B * 40 :=
sorry

end find_AB_value_l115_115068


namespace star_value_example_l115_115467

def my_star (a b : ℝ) : ℝ := (a + b)^2 + (a - b)^2

theorem star_value_example : my_star 3 5 = 68 := 
by
  sorry

end star_value_example_l115_115467


namespace solve_inequalities_l115_115636

theorem solve_inequalities (x : ℝ) :
  (2 * x + 1 < 3) ∧ ((x / 2) + ((1 - 3 * x) / 4) ≤ 1) → -3 ≤ x ∧ x < 1 := 
by
  sorry

end solve_inequalities_l115_115636


namespace polygon_sides_l115_115650

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 :=
by sorry

end polygon_sides_l115_115650


namespace soda_cans_purchasable_l115_115178

theorem soda_cans_purchasable (S Q : ℕ) (t D : ℝ) (hQ_pos : Q > 0) :
    let quarters_from_dollars := 4 * D
    let total_quarters_with_tax := quarters_from_dollars * (1 + t)
    (total_quarters_with_tax / Q) * S = (4 * D * S * (1 + t)) / Q :=
sorry

end soda_cans_purchasable_l115_115178


namespace solution_set_of_inequality_l115_115006

theorem solution_set_of_inequality (x : ℝ) : x * (2 - x) > 0 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l115_115006


namespace probability_four_of_eight_show_three_l115_115967

def probability_exactly_four_show_three : ℚ :=
  let num_ways := Nat.choose 8 4
  let prob_four_threes := (1 / 6) ^ 4
  let prob_four_not_threes := (5 / 6) ^ 4
  (num_ways * prob_four_threes * prob_four_not_threes)

theorem probability_four_of_eight_show_three :
  probability_exactly_four_show_three = 43750 / 1679616 :=
by 
  sorry

end probability_four_of_eight_show_three_l115_115967


namespace lauras_european_stamps_cost_l115_115414

def stamp_cost (count : ℕ) (cost_per_stamp : ℚ) : ℚ :=
  count * cost_per_stamp

def total_stamps_cost (stamps80 : ℕ) (stamps90 : ℕ) (cost_per_stamp : ℚ) : ℚ :=
  stamp_cost stamps80 cost_per_stamp + stamp_cost stamps90 cost_per_stamp

def european_stamps_cost_80_90 :=
  total_stamps_cost 10 12 0.09 + total_stamps_cost 18 16 0.07

theorem lauras_european_stamps_cost : european_stamps_cost_80_90 = 4.36 :=
by
  sorry

end lauras_european_stamps_cost_l115_115414


namespace susie_rooms_l115_115718

theorem susie_rooms
  (house_vacuum_time_hours : ℕ)
  (room_vacuum_time_minutes : ℕ)
  (total_vacuum_time_minutes : ℕ)
  (total_vacuum_time_computed : house_vacuum_time_hours * 60 = total_vacuum_time_minutes)
  (rooms_count : ℕ)
  (rooms_count_computed : total_vacuum_time_minutes / room_vacuum_time_minutes = rooms_count) :
  house_vacuum_time_hours = 2 →
  room_vacuum_time_minutes = 20 →
  rooms_count = 6 :=
by
  intros h1 h2
  sorry

end susie_rooms_l115_115718


namespace tan_triple_angle_l115_115709

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l115_115709


namespace circle_properties_intercept_length_l115_115557

theorem circle_properties (a r : ℝ) (h1 : a^2 + 16 = r^2) (h2 : (6 - a)^2 + 16 = r^2) (h3 : r > 0) :
  a = 3 ∧ r = 5 :=
by
  sorry

theorem intercept_length (m : ℝ) (h : |24 + m| / 5 = 3) :
  m = -4 ∨ m = -44 :=
by
  sorry

end circle_properties_intercept_length_l115_115557


namespace factor_correct_l115_115295

theorem factor_correct (x : ℝ) : 36 * x^2 + 24 * x = 12 * x * (3 * x + 2) := by
  sorry

end factor_correct_l115_115295


namespace cost_price_of_apple_l115_115075

variable (CP SP: ℝ)
variable (loss: ℝ)
variable (h1: SP = 18)
variable (h2: loss = CP / 6)
variable (h3: SP = CP - loss)

theorem cost_price_of_apple : CP = 21.6 :=
by
  sorry

end cost_price_of_apple_l115_115075


namespace max_sum_of_distinct_integers_l115_115739

theorem max_sum_of_distinct_integers (A B C : ℕ) (hABC_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (hProduct : A * B * C = 1638) :
  A + B + C ≤ 126 :=
sorry

end max_sum_of_distinct_integers_l115_115739


namespace num_classes_received_basketballs_l115_115000

theorem num_classes_received_basketballs (total_basketballs left_basketballs : ℕ) 
  (h : total_basketballs = 54) (h_left : left_basketballs = 5) : 
  (total_basketballs - left_basketballs) / 7 = 7 :=
by
  sorry

end num_classes_received_basketballs_l115_115000


namespace emily_orange_count_l115_115099

theorem emily_orange_count
  (betty_oranges : ℕ)
  (h1 : betty_oranges = 12)
  (sandra_oranges : ℕ)
  (h2 : sandra_oranges = 3 * betty_oranges)
  (emily_oranges : ℕ)
  (h3 : emily_oranges = 7 * sandra_oranges) :
  emily_oranges = 252 :=
by
  sorry

end emily_orange_count_l115_115099


namespace find_value_of_y_l115_115861

variable (p y : ℝ)
variable (h1 : p > 45)
variable (h2 : p * p / 100 = (2 * p / 300) * (p + y))

theorem find_value_of_y (h1 : p > 45) (h2 : p * p / 100 = (2 * p / 300) * (p + y)) : y = p / 2 :=
sorry

end find_value_of_y_l115_115861


namespace unit_digit_of_12_pow_100_l115_115307

def unit_digit_pow (a: ℕ) (n: ℕ) : ℕ :=
  (a ^ n) % 10

theorem unit_digit_of_12_pow_100 : unit_digit_pow 12 100 = 6 := by
  sorry

end unit_digit_of_12_pow_100_l115_115307


namespace moles_of_water_formed_l115_115282

-- Defining the relevant constants
def NH4Cl_moles : ℕ := sorry  -- Some moles of Ammonium chloride (NH4Cl)
def NaOH_moles : ℕ := 3       -- 3 moles of Sodium hydroxide (NaOH)
def H2O_moles : ℕ := 3        -- The total moles of Water (H2O) formed

-- Statement of the problem
theorem moles_of_water_formed :
  NH4Cl_moles ≥ NaOH_moles → H2O_moles = 3 :=
sorry

end moles_of_water_formed_l115_115282


namespace green_pill_cost_l115_115945

-- Definitions for the problem conditions
def number_of_days : ℕ := 21
def total_cost : ℚ := 819
def daily_cost : ℚ := total_cost / number_of_days
def cost_green_pill (x : ℚ) : ℚ := x
def cost_pink_pill (x : ℚ) : ℚ := x - 1
def total_daily_pill_cost (x : ℚ) : ℚ := cost_green_pill x + 2 * cost_pink_pill x

-- Theorem to be proven
theorem green_pill_cost : ∃ x : ℚ, total_daily_pill_cost x = daily_cost ∧ x = 41 / 3 :=
sorry

end green_pill_cost_l115_115945


namespace all_equal_l115_115929

theorem all_equal (xs xsp : Fin 2011 → ℝ) (h : ∀ i : Fin 2011, xs i + xs ((i + 1) % 2011) = 2 * xsp i) (perm : ∃ σ : Fin 2011 ≃ Fin 2011, ∀ i, xsp i = xs (σ i)) :
  ∀ i j : Fin 2011, xs i = xs j := 
sorry

end all_equal_l115_115929


namespace range_of_a_for_monotonically_decreasing_function_l115_115316

theorem range_of_a_for_monotonically_decreasing_function {a : ℝ} :
    (∀ x y : ℝ, (x > 2 → y > 2 → (ax^2 + x - 1) ≤ (a*y^2 + y - 1)) ∧
                (x ≤ 2 → y ≤ 2 → (-x + 1) ≤ (-y + 1)) ∧
                (x > 2 → y ≤ 2 → (ax^2 + x - 1) ≤ (-y + 1)) ∧
                (x ≤ 2 → y > 2 → (-x + 1) ≤ (a*y^2 + y - 1))) →
    (a < 0 ∧ - (1 / (2 * a)) ≤ 2 ∧ 4 * a + 1 ≤ -1) →
    a ≤ -1 / 2 :=
by
  intro hmonotone hconditions
  sorry

end range_of_a_for_monotonically_decreasing_function_l115_115316


namespace intersection_M_N_l115_115651

open Set

def M : Set ℤ := {-1, 0, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = 2 * x + 1}

theorem intersection_M_N : M ∩ N = {-1, 1} :=
by
  sorry

end intersection_M_N_l115_115651


namespace price_of_33_kgs_l115_115878

theorem price_of_33_kgs (l q : ℝ) 
  (h1 : l * 20 = 100) 
  (h2 : l * 30 + q * 6 = 186) : 
  l * 30 + q * 3 = 168 := 
by
  sorry

end price_of_33_kgs_l115_115878


namespace ball_hits_ground_l115_115446

theorem ball_hits_ground (t : ℝ) (y : ℝ) : 
  (y = -8 * t^2 - 12 * t + 72) → 
  (y = 0) → 
  t = 3 := 
by
  sorry

end ball_hits_ground_l115_115446


namespace temperature_on_Friday_l115_115659

variable (M T W Th F : ℝ)

def avg_M_T_W_Th := (M + T + W + Th) / 4 = 48
def avg_T_W_Th_F := (T + W + Th + F) / 4 = 46
def temp_Monday := M = 42

theorem temperature_on_Friday
  (h1 : avg_M_T_W_Th M T W Th)
  (h2 : avg_T_W_Th_F T W Th F) 
  (h3 : temp_Monday M) : F = 34 := by
  sorry

end temperature_on_Friday_l115_115659


namespace positive_number_is_nine_l115_115987

theorem positive_number_is_nine (x : ℝ) (n : ℝ) (hx : x > 0) (hn : n > 0)
  (sqrt1 : x^2 = n) (sqrt2 : (x - 6)^2 = n) : 
  n = 9 :=
by
  sorry

end positive_number_is_nine_l115_115987


namespace cost_of_green_lettuce_l115_115801

-- Definitions based on the conditions given in the problem
def cost_per_pound := 2
def weight_red_lettuce := 6 / cost_per_pound
def total_weight := 7
def weight_green_lettuce := total_weight - weight_red_lettuce

-- Problem statement: Prove that the cost of green lettuce is $8
theorem cost_of_green_lettuce : (weight_green_lettuce * cost_per_pound) = 8 :=
by
  sorry

end cost_of_green_lettuce_l115_115801


namespace find_angle_measure_l115_115140

theorem find_angle_measure (x : ℝ) (h : x = 2 * (90 - x) + 30) : x = 70 :=
by
  exact sorry

end find_angle_measure_l115_115140


namespace find_January_salary_l115_115058

-- Definitions and conditions
variables (J F M A May : ℝ)
def avg_Jan_to_Apr : Prop := (J + F + M + A) / 4 = 8000
def avg_Feb_to_May : Prop := (F + M + A + May) / 4 = 8300
def May_salary : Prop := May = 6500

-- Theorem statement
theorem find_January_salary (h1 : avg_Jan_to_Apr J F M A) 
                            (h2 : avg_Feb_to_May F M A May) 
                            (h3 : May_salary May) : 
                            J = 5300 :=
sorry

end find_January_salary_l115_115058


namespace div_1959_l115_115525

theorem div_1959 (n : ℕ) : ∃ k : ℤ, 5^(8 * n) - 2^(4 * n) * 7^(2 * n) = k * 1959 := 
by 
  sorry

end div_1959_l115_115525


namespace fraction_of_menu_items_i_can_eat_l115_115287

def total_dishes (vegan_dishes non_vegan_dishes : ℕ) : ℕ := vegan_dishes + non_vegan_dishes

def vegan_dishes_without_soy (vegan_dishes vegan_with_soy : ℕ) : ℕ := vegan_dishes - vegan_with_soy

theorem fraction_of_menu_items_i_can_eat (vegan_dishes non_vegan_dishes vegan_with_soy : ℕ)
  (h_vegan_dishes : vegan_dishes = 6)
  (h_menu_total : vegan_dishes = (total_dishes vegan_dishes non_vegan_dishes) / 3)
  (h_vegan_with_soy : vegan_with_soy = 4)
  : (vegan_dishes_without_soy vegan_dishes vegan_with_soy) / (total_dishes vegan_dishes non_vegan_dishes) = 1 / 9 :=
by
  sorry

end fraction_of_menu_items_i_can_eat_l115_115287


namespace num_female_students_l115_115600

theorem num_female_students (F : ℕ) (h1: 8 * 85 + F * 92 = (8 + F) * 90) : F = 20 := 
by
  sorry

end num_female_students_l115_115600


namespace sum_of_cubes_of_ages_l115_115186

noncomputable def dick_age : ℕ := 2
noncomputable def tom_age : ℕ := 5
noncomputable def harry_age : ℕ := 6

theorem sum_of_cubes_of_ages :
  4 * dick_age + 2 * tom_age = 3 * harry_age ∧ 
  3 * harry_age^2 = 2 * dick_age^2 + 4 * tom_age^2 ∧ 
  Nat.gcd (Nat.gcd dick_age tom_age) harry_age = 1 → 
  dick_age^3 + tom_age^3 + harry_age^3 = 349 :=
by
  intros h
  sorry

end sum_of_cubes_of_ages_l115_115186


namespace part_one_part_two_l115_115175

noncomputable def M := Set.Ioo (-(1 : ℝ)/2) (1/2)

namespace Problem

variables {a b : ℝ}
def in_M (x : ℝ) := x ∈ M

theorem part_one (ha : in_M a) (hb : in_M b) :
  |(1/3 : ℝ) * a + (1/6) * b| < 1/4 :=
sorry

theorem part_two (ha : in_M a) (hb : in_M b) :
  |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end Problem

end part_one_part_two_l115_115175


namespace rob_travel_time_to_park_l115_115149

theorem rob_travel_time_to_park : 
  ∃ R : ℝ, 
    (∀ Tm : ℝ, Tm = 3 * R) ∧ -- Mark's travel time is three times Rob's travel time
    (∀ Tr : ℝ, Tm - 2 = R) → -- Considering Mark's head start of 2 hours
    R = 1 :=
sorry

end rob_travel_time_to_park_l115_115149


namespace inequality_triangle_areas_l115_115054

theorem inequality_triangle_areas (a b c α β γ : ℝ) (hα : α = 2 * Real.sqrt (b * c)) (hβ : β = 2 * Real.sqrt (a * c)) (hγ : γ = 2 * Real.sqrt (a * b)) : 
  a / α + b / β + c / γ ≥ 3 / 2 := 
by
  sorry

end inequality_triangle_areas_l115_115054


namespace rice_yield_prediction_l115_115456

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 5 * x + 250

-- Define the specific condition for x = 80
def fertilizer_amount : ℝ := 80

-- State the theorem for the expected rice yield
theorem rice_yield_prediction : regression_line fertilizer_amount = 650 :=
by
  sorry

end rice_yield_prediction_l115_115456


namespace discount_percentage_l115_115811

theorem discount_percentage (coach_cost sectional_cost other_cost paid : ℕ) 
  (h1 : coach_cost = 2500) 
  (h2 : sectional_cost = 3500) 
  (h3 : other_cost = 2000) 
  (h4 : paid = 7200) : 
  ((coach_cost + sectional_cost + other_cost - paid) * 100) / (coach_cost + sectional_cost + other_cost) = 10 :=
by
  sorry

end discount_percentage_l115_115811


namespace greatest_integer_jo_thinking_of_l115_115013

theorem greatest_integer_jo_thinking_of :
  ∃ n : ℕ, n < 150 ∧ (∃ k : ℕ, n = 9 * k - 1) ∧ (∃ m : ℕ, n = 5 * m - 2) ∧ n = 143 :=
by
  sorry

end greatest_integer_jo_thinking_of_l115_115013


namespace neg_univ_prop_l115_115658

-- Translate the original mathematical statement to a Lean 4 statement.
theorem neg_univ_prop :
  (¬(∀ x : ℝ, x^2 ≠ x)) ↔ (∃ x : ℝ, x^2 = x) :=
by
  sorry

end neg_univ_prop_l115_115658


namespace emily_remainder_l115_115512

theorem emily_remainder (c d : ℤ) (h1 : c % 60 = 53) (h2 : d % 42 = 35) : (c + d) % 21 = 4 :=
by
  sorry

end emily_remainder_l115_115512


namespace infinite_series_sum_l115_115210

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l115_115210


namespace percentage_exceed_l115_115251

theorem percentage_exceed (x y : ℝ) (h : y = x + 0.2 * x) :
  (y - x) / x * 100 = 20 :=
by
  -- Proof goes here
  sorry

end percentage_exceed_l115_115251


namespace find_n_divisible_by_11_l115_115170

theorem find_n_divisible_by_11 : ∃ n : ℕ, 0 < n ∧ n < 11 ∧ (18888 - n) % 11 = 0 :=
by
  use 1
  -- proof steps would go here, but we're only asked for the statement
  sorry

end find_n_divisible_by_11_l115_115170


namespace average_speed_last_segment_l115_115622

variable (total_distance : ℕ := 120)
variable (total_minutes : ℕ := 120)
variable (first_segment_minutes : ℕ := 40)
variable (first_segment_speed : ℕ := 50)
variable (second_segment_minutes : ℕ := 40)
variable (second_segment_speed : ℕ := 55)
variable (third_segment_speed : ℕ := 75)

theorem average_speed_last_segment :
  let total_hours := total_minutes / 60
  let average_speed := total_distance / total_hours
  let speed_first_segment := first_segment_speed * (first_segment_minutes / 60)
  let speed_second_segment := second_segment_speed * (second_segment_minutes / 60)
  let speed_third_segment := third_segment_speed * (third_segment_minutes / 60)
  average_speed = (speed_first_segment + speed_second_segment + speed_third_segment) / 3 →
  third_segment_speed = 75 :=
by
  sorry

end average_speed_last_segment_l115_115622


namespace nine_otimes_three_l115_115693

def otimes (a b : ℤ) : ℤ := a + (4 * a) / (3 * b)

theorem nine_otimes_three : otimes 9 3 = 13 := by
  sorry

end nine_otimes_three_l115_115693


namespace solve_for_y_l115_115972

theorem solve_for_y (y : ℝ) (h : 3 / y + 4 / y / (6 / y) = 1.5) : y = 3.6 :=
sorry

end solve_for_y_l115_115972


namespace seating_arrangement_l115_115406

def numWaysCableCars (adults children cars capacity : ℕ) : ℕ := 
  sorry 

theorem seating_arrangement :
  numWaysCableCars 4 2 3 3 = 348 :=
by {
  sorry
}

end seating_arrangement_l115_115406


namespace total_red_and_green_peaches_l115_115384

-- Define the number of red peaches and green peaches.
def red_peaches : ℕ := 6
def green_peaches : ℕ := 16

-- Theorem stating the sum of red and green peaches is 22.
theorem total_red_and_green_peaches : red_peaches + green_peaches = 22 := 
by
  -- Proof would go here but is not required
  sorry

end total_red_and_green_peaches_l115_115384


namespace factorize_one_factorize_two_l115_115517

variable (x a b : ℝ)

-- Problem 1: Prove that 4x^2 - 64 = 4(x + 4)(x - 4)
theorem factorize_one : 4 * x^2 - 64 = 4 * (x + 4) * (x - 4) :=
sorry

-- Problem 2: Prove that 4ab^2 - 4a^2b - b^3 = -b(2a - b)^2
theorem factorize_two : 4 * a * b^2 - 4 * a^2 * b - b^3 = -b * (2 * a - b)^2 :=
sorry

end factorize_one_factorize_two_l115_115517


namespace solve_equation_l115_115163

theorem solve_equation (Y : ℝ) : (3.242 * 10 * Y) / 100 = 0.3242 * Y := 
by 
  sorry

end solve_equation_l115_115163


namespace count_primes_with_squares_in_range_l115_115085

theorem count_primes_with_squares_in_range : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, Prime n ∧ 5000 < n^2 ∧ n^2 < 9000) ∧ 
    S.card = 5 :=
by
  sorry

end count_primes_with_squares_in_range_l115_115085


namespace James_gold_bars_l115_115921

theorem James_gold_bars (P : ℝ) (h_condition1 : 60 - P / 100 * 60 = 54) : P = 10 := 
  sorry

end James_gold_bars_l115_115921


namespace num_roots_of_unity_satisfy_cubic_l115_115963

def root_of_unity (z : ℂ) (n : ℕ) : Prop :=
  z ^ n = 1

def cubic_eqn_root (z : ℂ) (a b c : ℤ) : Prop :=
  z^3 + (a:ℂ) * z^2 + (b:ℂ) * z + (c:ℂ) = 0

theorem num_roots_of_unity_satisfy_cubic (a b c : ℤ) (n : ℕ) 
    (h_n : n ≥ 1) : ∃! z : ℂ, root_of_unity z n ∧ cubic_eqn_root z a b c := sorry

end num_roots_of_unity_satisfy_cubic_l115_115963


namespace vertex_of_parabola_l115_115059

theorem vertex_of_parabola (c d : ℝ) (h : ∀ x : ℝ, -x^2 + c * x + d ≤ 0 ↔ (x ≤ -4 ∨ x ≥ 6)) :
  ∃ v : ℝ × ℝ, v = (1, 25) :=
sorry

end vertex_of_parabola_l115_115059


namespace div_by_66_l115_115724

theorem div_by_66 :
  (43 ^ 23 + 23 ^ 43) % 66 = 0 := 
sorry

end div_by_66_l115_115724


namespace next_term_geometric_sequence_l115_115284

theorem next_term_geometric_sequence (y : ℝ) : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  a₀ = 3 ∧ 
  a₁ = 9 * y ∧ 
  a₂ = 27 * y^2 ∧ 
  a₃ = 81 * y^3 ∧ 
  a₄ = a₃ * 3 * y 
  → a₄ = 243 * y^4 := by
  sorry

end next_term_geometric_sequence_l115_115284


namespace average_headcount_correct_l115_115174

def avg_headcount_03_04 : ℕ := 11500
def avg_headcount_04_05 : ℕ := 11600
def avg_headcount_05_06 : ℕ := 11300

noncomputable def average_headcount : ℕ :=
  (avg_headcount_03_04 + avg_headcount_04_05 + avg_headcount_05_06) / 3

theorem average_headcount_correct :
  average_headcount = 11467 :=
by
  sorry

end average_headcount_correct_l115_115174


namespace time_for_B_work_alone_l115_115449

def work_rate_A : ℚ := 1 / 6
def work_rate_combined : ℚ := 1 / 3
def work_share_C : ℚ := 1 / 8

theorem time_for_B_work_alone : 
  ∃ x : ℚ, (work_rate_A + 1 / x = work_rate_combined - work_share_C) → x = 24 := 
sorry

end time_for_B_work_alone_l115_115449


namespace egg_distribution_l115_115918

-- Definitions of the conditions
def total_eggs := 10.0
def large_eggs := 6.0
def small_eggs := 4.0

def box_A_capacity := 5.0
def box_B_capacity := 4.0
def box_C_capacity := 6.0

def at_least_one_small_egg (box_A_small box_B_small box_C_small : Float) := 
  box_A_small >= 1.0 ∧ box_B_small >= 1.0 ∧ box_C_small >= 1.0

-- Problem statement
theorem egg_distribution : 
  ∃ (box_A_small box_A_large box_B_small box_B_large box_C_small box_C_large : Float),
  box_A_small + box_A_large <= box_A_capacity ∧
  box_B_small + box_B_large <= box_B_capacity ∧
  box_C_small + box_C_large <= box_C_capacity ∧
  box_A_small + box_B_small + box_C_small = small_eggs ∧
  box_A_large + box_B_large + box_C_large = large_eggs ∧
  at_least_one_small_egg box_A_small box_B_small box_C_small :=
sorry

end egg_distribution_l115_115918


namespace sequence_a_n_l115_115774

noncomputable def a_n (n : ℕ) : ℚ :=
if n = 1 then 1 else (1 : ℚ) / (2 * n - 1)

theorem sequence_a_n (n : ℕ) (hn : n ≥ 1) : 
  (a_n 1 = 1) ∧ 
  (∀ n, a_n n ≠ 0) ∧ 
  (∀ n, n ≥ 2 → a_n n + 2 * a_n n * a_n (n - 1) - a_n (n - 1) = 0) →
  a_n n = 1 / (2 * n - 1) :=
by
  sorry

end sequence_a_n_l115_115774


namespace rectangle_exists_l115_115427

theorem rectangle_exists (n : ℕ) (h_n : 0 < n)
  (marked : Finset (Fin n × Fin n))
  (h_marked : marked.card ≥ n * (Real.sqrt n + 0.5)) :
  ∃ (r1 r2 : Fin n) (c1 c2 : Fin n), r1 ≠ r2 ∧ c1 ≠ c2 ∧ 
    ((r1, c1) ∈ marked ∧ (r1, c2) ∈ marked ∧ (r2, c1) ∈ marked ∧ (r2, c2) ∈ marked) :=
  sorry

end rectangle_exists_l115_115427


namespace length_of_adjacent_side_l115_115529

variable (a b : ℝ)

theorem length_of_adjacent_side (area : ℝ) (side : ℝ) :
  area = 6 * a^3 + 9 * a^2 - 3 * a * b →
  side = 3 * a →
  (area / side = 2 * a^2 + 3 * a - b) :=
by
  intro h_area
  intro h_side
  sorry

end length_of_adjacent_side_l115_115529


namespace polynomial_roots_l115_115632

theorem polynomial_roots :
  Polynomial.roots (Polynomial.C 4 * Polynomial.X ^ 5 +
                    Polynomial.C 13 * Polynomial.X ^ 4 +
                    Polynomial.C (-30) * Polynomial.X ^ 3 +
                    Polynomial.C 8 * Polynomial.X ^ 2) =
  {0, 0, 1 / 2, -2 + 2 * Real.sqrt 2, -2 - 2 * Real.sqrt 2} :=
by
  sorry

end polynomial_roots_l115_115632


namespace find_f_10_l115_115946

-- Defining the function f as an odd, periodic function with period 2
def odd_func_periodic (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) = - f x) ∧ (∀ x : ℝ, f (x + 2) = f x)

-- Stating the theorem that f(10) is 0 given the conditions
theorem find_f_10 (f : ℝ → ℝ) (h1 : odd_func_periodic f) : f 10 = 0 :=
sorry

end find_f_10_l115_115946


namespace cone_radius_correct_l115_115883

noncomputable def cone_radius (CSA l : ℝ) : ℝ := CSA / (Real.pi * l)

theorem cone_radius_correct :
  cone_radius 1539.3804002589986 35 = 13.9 :=
by
  -- Proof omitted
  sorry

end cone_radius_correct_l115_115883


namespace Robert_has_taken_more_photos_l115_115629

variables (C L R : ℕ) -- Claire's, Lisa's, and Robert's photos

-- Conditions definitions:
def ClairePhotos : Prop := C = 8
def LisaPhotos : Prop := L = 3 * C
def RobertPhotos : Prop := R > C

-- The proof problem statement:
theorem Robert_has_taken_more_photos (h1 : ClairePhotos C) (h2 : LisaPhotos C L) : RobertPhotos C R :=
by { sorry }

end Robert_has_taken_more_photos_l115_115629


namespace find_a_l115_115364
-- Import necessary Lean libraries

-- Define the function and its maximum value condition
def f (a x : ℝ) := -x^2 + 2*a*x + 1 - a

def has_max_value (f : ℝ → ℝ) (M : ℝ) (interval : Set ℝ) : Prop :=
  ∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = M

theorem find_a (a : ℝ) :
  has_max_value (f a) 2 (Set.Icc 0 1) → (a = -1 ∨ a = 2) :=
by
  sorry

end find_a_l115_115364


namespace isabel_weekly_distance_l115_115760

def circuit_length : ℕ := 365
def morning_runs : ℕ := 7
def afternoon_runs : ℕ := 3
def days_per_week : ℕ := 7

def morning_distance := morning_runs * circuit_length
def afternoon_distance := afternoon_runs * circuit_length
def daily_distance := morning_distance + afternoon_distance
def weekly_distance := daily_distance * days_per_week

theorem isabel_weekly_distance : weekly_distance = 25550 := by
  sorry

end isabel_weekly_distance_l115_115760


namespace mark_final_buttons_l115_115800

def mark_initial_buttons : ℕ := 14
def shane_factor : ℚ := 3.5
def lent_to_anna : ℕ := 7
def lost_fraction : ℚ := 0.5
def sam_fraction : ℚ := 2 / 3

theorem mark_final_buttons : 
  let shane_buttons := mark_initial_buttons * shane_factor
  let before_anna := mark_initial_buttons + shane_buttons
  let after_lending_anna := before_anna - lent_to_anna
  let anna_returned := lent_to_anna * (1 - lost_fraction)
  let after_anna_return := after_lending_anna + anna_returned
  let after_sam := after_anna_return - (after_anna_return * sam_fraction)
  round after_sam = 20 := 
by
  sorry

end mark_final_buttons_l115_115800


namespace tax_increase_proof_l115_115664

variables (old_tax_rate new_tax_rate : ℝ) (old_income new_income : ℝ)

def old_taxes_paid (old_tax_rate old_income : ℝ) : ℝ := old_tax_rate * old_income

def new_taxes_paid (new_tax_rate new_income : ℝ) : ℝ := new_tax_rate * new_income

def increase_in_taxes (old_tax_rate new_tax_rate old_income new_income : ℝ) : ℝ :=
  new_taxes_paid new_tax_rate new_income - old_taxes_paid old_tax_rate old_income

theorem tax_increase_proof :
  increase_in_taxes 0.20 0.30 1000000 1500000 = 250000 := by
  sorry

end tax_increase_proof_l115_115664


namespace equal_areas_of_parts_l115_115966

theorem equal_areas_of_parts :
  ∀ (S1 S2 S3 S4 : ℝ), 
    S1 = S2 → S2 = S3 → 
    (S1 + S2 = S3 + S4) → 
    (S2 + S3 = S1 + S4) → 
    S1 = S2 ∧ S2 = S3 ∧ S3 = S4 :=
by
  intros S1 S2 S3 S4 h1 h2 h3 h4
  sorry

end equal_areas_of_parts_l115_115966


namespace no_positive_integer_n_such_that_14n_plus_19_is_prime_l115_115949

theorem no_positive_integer_n_such_that_14n_plus_19_is_prime :
  ∀ n : Nat, 0 < n → ¬ Nat.Prime (14^n + 19) :=
by
  intro n hn
  sorry

end no_positive_integer_n_such_that_14n_plus_19_is_prime_l115_115949


namespace find_m_if_f_even_l115_115128

variable (m : ℝ)

def f (x : ℝ) : ℝ := x^2 + (m + 2) * x + 3

theorem find_m_if_f_even (h : ∀ x, f m x = f m (-x)) : m = -2 :=
by
  sorry

end find_m_if_f_even_l115_115128


namespace find_a_l115_115858

-- Definitions
def parabola (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c
def vertex_property (a b c : ℤ) := 
  ∃ x y, x = 2 ∧ y = 5 ∧ y = parabola a b c x
def point_on_parabola (a b c : ℤ) := 
  ∃ x y, x = 1 ∧ y = 2 ∧ y = parabola a b c x

-- The main statement
theorem find_a {a b c : ℤ} (h_vertex : vertex_property a b c) (h_point : point_on_parabola a b c) : a = -3 :=
by {
  sorry
}

end find_a_l115_115858


namespace minimum_value_of_m_minus_n_l115_115395

def f (x : ℝ) : ℝ := (x - 1) ^ 2

theorem minimum_value_of_m_minus_n 
  (f_even : ∀ x : ℝ, f x = f (-x))
  (condition1 : n ≤ f (-2))
  (condition2 : n ≤ f (-1 / 2))
  (condition3 : f (-2) ≤ m)
  (condition4 : f (-1 / 2) ≤ m)
  : ∃ n m, m - n = 1 :=
by
  sorry

end minimum_value_of_m_minus_n_l115_115395


namespace no_play_students_count_l115_115094

theorem no_play_students_count :
  let total_students := 420
  let football_players := 325
  let cricket_players := 175
  let both_players := 130
  total_students - (football_players + cricket_players - both_players) = 50 :=
by
  sorry

end no_play_students_count_l115_115094


namespace percentage_of_loss_is_15_percent_l115_115065

/-- 
Given:
  SP₁ = 168 -- Selling price when gaining 20%
  Gain = 20% 
  SP₂ = 119 -- Selling price when calculating loss

Prove:
  The percentage of loss when the article is sold for Rs. 119 is 15%
--/

noncomputable def percentage_loss (CP SP₂: ℝ) : ℝ :=
  ((CP - SP₂) / CP) * 100

theorem percentage_of_loss_is_15_percent (CP SP₂ SP₁: ℝ) (Gain: ℝ):
  CP = 140 ∧ SP₁ = 168 ∧ SP₂ = 119 ∧ Gain = 20 → percentage_loss CP SP₂ = 15 :=
by
  intro h
  sorry

end percentage_of_loss_is_15_percent_l115_115065


namespace percentage_problem_l115_115717

theorem percentage_problem (P : ℝ) (h : (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 42) : P = 35 := 
by
  -- Proof goes here
  sorry

end percentage_problem_l115_115717


namespace Jane_mom_jars_needed_l115_115144

theorem Jane_mom_jars_needed : 
  ∀ (total_tomatoes jar_capacity : ℕ), 
  total_tomatoes = 550 → 
  jar_capacity = 14 → 
  ⌈(total_tomatoes: ℚ) / jar_capacity⌉ = 40 := 
by 
  intros total_tomatoes jar_capacity htotal hcapacity
  sorry

end Jane_mom_jars_needed_l115_115144


namespace inequality_must_hold_l115_115957

variable (a b c : ℝ)

theorem inequality_must_hold (h1 : a > b) (h2 : c < 0) : a * (c - 1) < b * (c - 1) := 
sorry

end inequality_must_hold_l115_115957


namespace pages_revised_twice_l115_115573

theorem pages_revised_twice
  (x : ℕ)
  (h1 : ∀ x, x > 30 → 1000 + 100 + 10 * x ≠ 1400)
  (h2 : ∀ x, x < 30 → 1000 + 100 + 10 * x ≠ 1400)
  (h3 : 1000 + 100 + 10 * 30 = 1400) :
  x = 30 :=
by
  sorry

end pages_revised_twice_l115_115573


namespace right_triangle_AB_CA_BC_l115_115960

namespace TriangleProof

def point := ℝ × ℝ

def dist (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

def A : point := (5, -2)
def B : point := (1, 5)
def C : point := (-1, 2)

def AB2 := dist A B
def BC2 := dist B C
def CA2 := dist C A

theorem right_triangle_AB_CA_BC : CA2 + BC2 = AB2 :=
by 
  -- proof will be filled here
  sorry

end TriangleProof

end right_triangle_AB_CA_BC_l115_115960


namespace eleven_pow_2048_mod_17_l115_115686

theorem eleven_pow_2048_mod_17 : 11^2048 % 17 = 1 := by
  sorry

end eleven_pow_2048_mod_17_l115_115686


namespace point_to_line_distance_l115_115336

theorem point_to_line_distance :
  let circle_center : ℝ×ℝ := (0, 1)
  let A : ℝ := -1
  let B : ℝ := 1
  let C : ℝ := -2
  let line_eq (x y : ℝ) := A * x + B * y + C == 0
  ∀ (x0 : ℝ) (y0 : ℝ),
    circle_center = (x0, y0) →
    (|A * x0 + B * y0 + C| / Real.sqrt (A^2 + B^2)) = (Real.sqrt 2 / 2) := 
by 
  intros
  -- Proof goes here
  sorry -- Placeholder for the proof.

end point_to_line_distance_l115_115336


namespace loss_percentage_l115_115936

theorem loss_percentage
  (CP : ℝ := 1166.67)
  (SP : ℝ)
  (H : SP + 140 = CP + 0.02 * CP) :
  ((CP - SP) / CP) * 100 = 10 := 
by 
  sorry

end loss_percentage_l115_115936


namespace inequality_preservation_l115_115238

theorem inequality_preservation (a b x : ℝ) (h : a > b) : a * 2^x > b * 2^x :=
sorry

end inequality_preservation_l115_115238


namespace least_positive_integer_l115_115928

theorem least_positive_integer (a : ℕ) :
  (a % 2 = 0) ∧ (a % 5 = 1) ∧ (a % 4 = 2) → a = 6 :=
by
  sorry

end least_positive_integer_l115_115928


namespace number_of_blue_candles_l115_115908

-- Conditions
def grandfather_age : ℕ := 79
def yellow_candles : ℕ := 27
def red_candles : ℕ := 14
def total_candles : ℕ := grandfather_age
def yellow_red_candles : ℕ := yellow_candles + red_candles
def blue_candles : ℕ := total_candles - yellow_red_candles

-- Proof statement
theorem number_of_blue_candles : blue_candles = 38 :=
by
  -- sorry indicates the proof is omitted
  sorry

end number_of_blue_candles_l115_115908


namespace line_intersects_x_axis_at_point_l115_115515

theorem line_intersects_x_axis_at_point (x1 y1 x2 y2 : ℝ) 
  (h1 : (x1, y1) = (7, -3))
  (h2 : (x2, y2) = (3, 1)) : 
  ∃ x, (x, 0) = (4, 0) :=
by
  -- sorry serves as a placeholder for the actual proof
  sorry

end line_intersects_x_axis_at_point_l115_115515


namespace triangle_perimeter_ABC_l115_115177

noncomputable def perimeter_triangle (AP PB r : ℕ) (hAP : AP = 23) (hPB : PB = 27) (hr : r = 21) : ℕ :=
  2 * (50 + 245 / 2)

theorem triangle_perimeter_ABC (AP PB r : ℕ) 
  (hAP : AP = 23) 
  (hPB : PB = 27) 
  (hr : r = 21) : 
  perimeter_triangle AP PB r hAP hPB hr = 345 :=
by
  sorry

end triangle_perimeter_ABC_l115_115177


namespace probability_samantha_in_sam_not_l115_115603

noncomputable def probability_in_picture_but_not (time_samantha : ℕ) (lap_samantha : ℕ) (time_sam : ℕ) (lap_sam : ℕ) : ℚ :=
  let seconds_raced := 900
  let samantha_laps := seconds_raced / time_samantha
  let sam_laps := seconds_raced / time_sam
  let start_line_samantha := (samantha_laps - (samantha_laps % 1)) * time_samantha + ((samantha_laps % 1) * lap_samantha)
  let start_line_sam := (sam_laps - (sam_laps % 1)) * time_sam + ((sam_laps % 1) * lap_sam)
  let in_picture_duration := 80
  let overlapping_time := 30
  overlapping_time / in_picture_duration

theorem probability_samantha_in_sam_not : probability_in_picture_but_not 120 60 75 25 = 3 / 8 := by
  sorry

end probability_samantha_in_sam_not_l115_115603


namespace cannot_be_right_angle_triangle_l115_115544

-- Definition of the converse of the Pythagorean theorem
def is_right_angle_triangle (a b c : ℕ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2

-- Definition to check if a given set of sides cannot form a right-angled triangle
def cannot_form_right_angle_triangle (a b c : ℕ) : Prop :=
  ¬ is_right_angle_triangle a b c

-- Given sides of the triangle option D
theorem cannot_be_right_angle_triangle : cannot_form_right_angle_triangle 3 4 6 :=
  by sorry

end cannot_be_right_angle_triangle_l115_115544


namespace closest_perfect_square_to_350_l115_115283

theorem closest_perfect_square_to_350 : 
  ∃ (n : ℤ), n^2 = 361 ∧ ∀ (k : ℤ), (k^2 ≠ 361 → |350 - n^2| < |350 - k^2|) :=
by
  sorry

end closest_perfect_square_to_350_l115_115283


namespace proposition_C_l115_115433

-- Given conditions
variables {a b : ℝ}

-- Proposition C is the correct one
theorem proposition_C (h : a^3 > b^3) : a > b := by
  sorry

end proposition_C_l115_115433


namespace original_time_to_complete_book_l115_115887

-- Define the problem based on the given conditions
variables (n : ℕ) (T : ℚ)

-- Define the conditions
def condition1 : Prop := 
  ∃ (n T : ℚ), 
  n / T = (n + 3) / (0.75 * T) ∧
  n / T = (n - 3) / (T + 5 / 6)

-- State the theorem with the correct answer
theorem original_time_to_complete_book : condition1 → T = 5 / 3 :=
by sorry

end original_time_to_complete_book_l115_115887


namespace winning_candidate_votes_l115_115587

theorem winning_candidate_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 336): 0.62 * V = 868 :=
by
  sorry

end winning_candidate_votes_l115_115587


namespace television_screen_horizontal_length_l115_115758

theorem television_screen_horizontal_length :
  ∀ (d : ℝ) (r_l : ℝ) (r_h : ℝ), r_l / r_h = 4 / 3 → d = 27 → 
  let h := (3 / 5) * d
  let l := (4 / 5) * d
  l = 21.6 := by
  sorry

end television_screen_horizontal_length_l115_115758


namespace certain_number_x_l115_115540

theorem certain_number_x :
  ∃ x : ℤ, (287 * 287 + 269 * 269 - x * (287 * 269) = 324) ∧ (x = 2) := 
by {
  use 2,
  sorry
}

end certain_number_x_l115_115540


namespace first_tier_tax_rate_l115_115839

theorem first_tier_tax_rate (price : ℕ) (total_tax : ℕ) (tier1_limit : ℕ) (tier2_rate : ℝ) (tier1_tax_rate : ℝ) :
  price = 18000 →
  total_tax = 1950 →
  tier1_limit = 11000 →
  tier2_rate = 0.09 →
  ((price - tier1_limit) * tier2_rate + tier1_tax_rate * tier1_limit = total_tax) →
  tier1_tax_rate = 0.12 :=
by
  intros hprice htotal htier1 hrate htax_eq
  sorry

end first_tier_tax_rate_l115_115839


namespace unique_function_l115_115902

theorem unique_function (f : ℝ → ℝ) (hf : ∀ x : ℝ, 0 ≤ x → 0 ≤ f x)
  (cond1 : ∀ x : ℝ, 0 ≤ x → 4 * f x ≥ 3 * x)
  (cond2 : ∀ x : ℝ, 0 ≤ x → f (4 * f x - 3 * x) = x) :
  ∀ x : ℝ, 0 ≤ x → f x = x :=
by
  sorry

end unique_function_l115_115902


namespace intervals_of_monotonicity_and_extreme_values_l115_115143

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem intervals_of_monotonicity_and_extreme_values :
  (∀ x : ℝ, x < 1 → deriv f x > 0) ∧
  (∀ x : ℝ, x > 1 → deriv f x < 0) ∧
  (∀ x : ℝ, f 1 = 1 / Real.exp 1) :=
by
  sorry

end intervals_of_monotonicity_and_extreme_values_l115_115143


namespace car_distance_l115_115339

variable (T_initial : ℕ) (T_new : ℕ) (S : ℕ) (D : ℕ)

noncomputable def calculate_distance (T_initial T_new S : ℕ) : ℕ :=
  S * T_new

theorem car_distance :
  T_initial = 6 →
  T_new = (3 / 2) * T_initial →
  S = 16 →
  D = calculate_distance T_initial T_new S →
  D = 144 :=
by
  sorry

end car_distance_l115_115339


namespace rachel_homework_l115_115193

theorem rachel_homework : 5 + 2 = 7 := by
  sorry

end rachel_homework_l115_115193


namespace flu_epidemic_infection_rate_l115_115521

theorem flu_epidemic_infection_rate : 
  ∃ x : ℝ, 1 + x + x * (1 + x) = 100 ∧ x = 9 := 
by
  sorry

end flu_epidemic_infection_rate_l115_115521


namespace number_of_ways_to_assign_volunteers_l115_115974

/-- Theorem: The number of ways to assign 5 volunteers to 3 venues such that each venue has at least one volunteer is 150. -/
theorem number_of_ways_to_assign_volunteers :
  let total_ways := 3^5
  let subtract_one_empty := 3 * 2^5
  let add_back_two_empty := 3 * 1^5
  (total_ways - subtract_one_empty + add_back_two_empty) = 150 :=
by
  sorry

end number_of_ways_to_assign_volunteers_l115_115974


namespace class_trip_contributions_l115_115810

theorem class_trip_contributions (x y : ℕ) :
  (x + 5) * (y + 6) = x * y + 792 ∧ (x - 4) * (y + 4) = x * y - 388 → x = 213 ∧ y = 120 := 
by
  sorry

end class_trip_contributions_l115_115810


namespace harriet_trip_time_l115_115723

theorem harriet_trip_time :
  ∀ (t1 : ℝ) (s1 s2 t2 d : ℝ), 
  t1 = 2.8 ∧ 
  s1 = 110 ∧ 
  s2 = 140 ∧ 
  d = s1 * t1 ∧ 
  t2 = d / s2 → 
  t1 + t2 = 5 :=
by intros t1 s1 s2 t2 d
   sorry

end harriet_trip_time_l115_115723


namespace regular_polygon_sides_l115_115400

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l115_115400


namespace sum_of_reciprocal_squares_l115_115534

theorem sum_of_reciprocal_squares
  (p q r : ℝ)
  (h1 : p + q + r = 9)
  (h2 : p * q + q * r + r * p = 8)
  (h3 : p * q * r = -2) :
  (1 / p ^ 2 + 1 / q ^ 2 + 1 / r ^ 2) = 25 := by
  sorry

end sum_of_reciprocal_squares_l115_115534


namespace expression_divisibility_l115_115827

theorem expression_divisibility (x y : ℤ) (k_1 k_2 : ℤ) (h1 : 2 * x + 3 * y = 17 * k_1) :
    ∃ k_2 : ℤ, 9 * x + 5 * y = 17 * k_2 :=
by
  sorry

end expression_divisibility_l115_115827


namespace total_monsters_l115_115969

theorem total_monsters (a1 a2 a3 a4 a5 : ℕ) 
  (h1 : a1 = 2) 
  (h2 : a2 = 2 * a1) 
  (h3 : a3 = 2 * a2) 
  (h4 : a4 = 2 * a3) 
  (h5 : a5 = 2 * a4) : 
  a1 + a2 + a3 + a4 + a5 = 62 :=
by
  sorry

end total_monsters_l115_115969


namespace polynomial_evaluation_l115_115575

theorem polynomial_evaluation (p : Polynomial ℚ) 
  (hdeg : p.degree = 7)
  (h : ∀ n : ℕ, n ≤ 7 → p.eval (2^n) = 1 / 2^(n + 1)) : 
  p.eval 0 = 255 / 2^28 := 
sorry

end polynomial_evaluation_l115_115575


namespace find_k_for_sum_of_cubes_l115_115154

theorem find_k_for_sum_of_cubes (k : ℝ) (r s : ℝ)
  (h1 : r + s = -2)
  (h2 : r * s = k / 3)
  (h3 : r^3 + s^3 = r + s) : k = 3 :=
by
  -- Sorry will be replaced by the actual proof
  sorry

end find_k_for_sum_of_cubes_l115_115154


namespace part1_part2_l115_115802

variable (a b : ℝ)

-- Conditions
axiom abs_a_eq_4 : |a| = 4
axiom abs_b_eq_6 : |b| = 6

-- Part 1: If ab > 0, find the value of a - b
theorem part1 (h : a * b > 0) : a - b = 2 ∨ a - b = -2 := 
by
  -- Proof will go here
  sorry

-- Part 2: If |a + b| = -(a + b), find the value of a + b
theorem part2 (h : |a + b| = -(a + b)) : a + b = -10 ∨ a + b = -2 := 
by
  -- Proof will go here
  sorry

end part1_part2_l115_115802


namespace lizzy_final_amount_l115_115880

-- Define constants
def m : ℕ := 80   -- cents from mother
def f : ℕ := 40   -- cents from father
def s : ℕ := 50   -- cents spent on candy
def u : ℕ := 70   -- cents from uncle
def t : ℕ := 90   -- cents for the toy
def c : ℕ := 110  -- cents change she received

-- Define the final amount calculation
def final_amount : ℕ := m + f - s + u - t + c

-- Prove the final amount is 160
theorem lizzy_final_amount : final_amount = 160 := by
  sorry

end lizzy_final_amount_l115_115880


namespace factorization_correct_l115_115688

theorem factorization_correct (x y : ℝ) : x^2 * y - x * y^2 = x * y * (x - y) :=
by
  sorry

end factorization_correct_l115_115688


namespace intersection_M_N_l115_115665

def M : Set ℝ := { x | -4 < x ∧ x < 2 }

def N : Set ℝ := { x | x^2 - x - 6 < 0 }

theorem intersection_M_N : M ∩ N = { x | -2 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l115_115665


namespace probability_y_gt_x_l115_115910

-- Define the uniform distribution and the problem setup
def uniform_distribution (a b : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ b }

-- Define the variables
variables (x : ℝ) (hx : x ∈ uniform_distribution 0 3000) (y : ℝ) (hy : y ∈ uniform_distribution 0 6000)

-- Define the probability calculation function (assuming some proper definition for probability)
noncomputable def probability_event (E : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the event that Laurent's number is greater than Chloe's number
def event_y_gt_x : Set (ℝ × ℝ) := {p | p.2 > p.1}

-- State the theorem
theorem probability_y_gt_x (x : ℝ) (hx : x ∈ uniform_distribution 0 3000) (y : ℝ) (hy : y ∈ uniform_distribution 0 6000) :
  probability_event event_y_gt_x = 3/4 :=
sorry

end probability_y_gt_x_l115_115910


namespace value_of_f_at_pi_over_12_l115_115371

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x - ω * Real.pi)

theorem value_of_f_at_pi_over_12 (ω : ℝ) (hω_pos : ω > 0) 
(h_period : ∀ x, f ω (x + Real.pi) = f ω x) : 
  f ω (Real.pi / 12) = 1 / 2 := 
sorry

end value_of_f_at_pi_over_12_l115_115371


namespace circle_center_and_radius_l115_115110

def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 4

theorem circle_center_and_radius :
  (∀ x y : ℝ, circle_eq x y ↔ (x - 2) ^ 2 + y ^ 2 = 4) →
  (exists (h k r : ℝ), (h, k) = (2, 0) ∧ r = 2) :=
by
  sorry

end circle_center_and_radius_l115_115110


namespace angle_Z_90_l115_115348

-- Definitions and conditions from step a)
def Triangle (X Y Z : ℝ) : Prop :=
  X + Y + Z = 180

def in_triangle_XYZ (X Y Z : ℝ) : Prop :=
  Triangle X Y Z ∧ (X + Y = 90)

-- Proof problem from step c)
theorem angle_Z_90 (X Y Z : ℝ) (h : in_triangle_XYZ X Y Z) : Z = 90 :=
  by
  sorry

end angle_Z_90_l115_115348


namespace four_digit_even_numbers_divisible_by_4_l115_115732

noncomputable def number_of_4_digit_even_numbers_divisible_by_4 : Nat :=
  500

theorem four_digit_even_numbers_divisible_by_4 : 
  (∃ count : Nat, count = number_of_4_digit_even_numbers_divisible_by_4) :=
sorry

end four_digit_even_numbers_divisible_by_4_l115_115732


namespace correct_statements_count_l115_115380

theorem correct_statements_count :
  (∃ n : ℕ, odd_positive_integer = 4 * n + 1 ∨ odd_positive_integer = 4 * n + 3) ∧
  (∀ k : ℕ, k = 3 * m ∨ k = 3 * m + 1 ∨ k = 3 * m + 2) ∧
  (∀ s : ℕ, odd_positive_integer ^ 2 = 8 * p + 1) ∧
  (∀ t : ℕ, perfect_square = 3 * q ∨ perfect_square = 3 * q + 1) →
  num_correct_statements = 2 :=
by
  sorry

end correct_statements_count_l115_115380


namespace alcohol_water_ratio_l115_115434

theorem alcohol_water_ratio (V : ℝ) (hV_pos : V > 0) :
  let jar1_alcohol := (2 / 3) * V
  let jar1_water := (1 / 3) * V
  let jar2_alcohol := (3 / 2) * V
  let jar2_water := (1 / 2) * V
  let total_alcohol := jar1_alcohol + jar2_alcohol
  let total_water := jar1_water + jar2_water
  (total_alcohol / total_water) = (13 / 5) :=
by
  -- Placeholder for the proof
  sorry

end alcohol_water_ratio_l115_115434


namespace ratio_p_q_l115_115383

theorem ratio_p_q 
  (total_amount : ℕ) 
  (amount_r : ℕ) 
  (ratio_q_r : ℕ × ℕ) 
  (total_amount_eq : total_amount = 1210) 
  (amount_r_eq : amount_r = 400) 
  (ratio_q_r_eq : ratio_q_r = (9, 10)) :
  ∃ (amount_p amount_q : ℕ), 
    total_amount = amount_p + amount_q + amount_r ∧ 
    (amount_q : ℕ) = 9 * (amount_r / 10) ∧ 
    (amount_p : ℕ) / (amount_q : ℕ) = 5 / 4 := 
by sorry

end ratio_p_q_l115_115383


namespace system_solution_l115_115767

theorem system_solution (x y z : ℝ) 
  (h1 : x - y ≥ z)
  (h2 : x^2 + 4 * y^2 + 5 = 4 * z) :
  (x = 2 ∧ y = -0.5 ∧ z = 2.5) :=
sorry

end system_solution_l115_115767


namespace unique_solution_for_a_l115_115462

theorem unique_solution_for_a (a : ℝ) :
  (∃! (x y : ℝ), 
    (x * Real.cos a + y * Real.sin a = 5 * Real.cos a + 2 * Real.sin a) ∧
    (-3 ≤ x + 2 * y ∧ x + 2 * y ≤ 7) ∧
    (-9 ≤ 3 * x - 4 * y ∧ 3 * x - 4 * y ≤ 1)) ↔ 
  (∃ k : ℤ, a = Real.arctan 4 + k * Real.pi ∨ a = -Real.arctan 2 + k * Real.pi) :=
sorry

end unique_solution_for_a_l115_115462


namespace pyramid_on_pentagonal_prism_l115_115647

-- Define the structure of a pentagonal prism
structure PentagonalPrism where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

-- Initial pentagonal prism properties
def initialPrism : PentagonalPrism := {
  faces := 7,
  vertices := 10,
  edges := 15
}

-- Assume we add a pyramid on top of one pentagonal face
def addPyramid (prism : PentagonalPrism) : PentagonalPrism := {
  faces := prism.faces - 1 + 5, -- 1 face covered, 5 new faces
  vertices := prism.vertices + 1, -- 1 new vertex
  edges := prism.edges + 5 -- 5 new edges
}

-- The resulting shape after adding the pyramid
def resultingShape : PentagonalPrism := addPyramid initialPrism

-- Calculating the sum of faces, vertices, and edges
def sumFacesVerticesEdges (shape : PentagonalPrism) : ℕ :=
  shape.faces + shape.vertices + shape.edges

-- Statement of the problem in Lean 4
theorem pyramid_on_pentagonal_prism : sumFacesVerticesEdges resultingShape = 42 := by
  sorry

end pyramid_on_pentagonal_prism_l115_115647


namespace percentage_of_students_in_grade_8_combined_l115_115123

theorem percentage_of_students_in_grade_8_combined (parkwood_students maplewood_students : ℕ)
  (parkwood_percentages maplewood_percentages : ℕ → ℕ) 
  (H_parkwood : parkwood_students = 150)
  (H_maplewood : maplewood_students = 120)
  (H_parkwood_percent : parkwood_percentages 8 = 18)
  (H_maplewood_percent : maplewood_percentages 8 = 25):
  (57 / 270) * 100 = 21.11 := 
by
  sorry  -- Proof omitted

end percentage_of_students_in_grade_8_combined_l115_115123


namespace sqrt_ac_bd_le_sqrt_ef_l115_115161

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem sqrt_ac_bd_le_sqrt_ef
  (a b c d e f : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ 0 ≤ f)
  (h1 : a + b ≤ e)
  (h2 : c + d ≤ f) :
  sqrt (a * c) + sqrt (b * d) ≤ sqrt (e * f) :=
by
  sorry

end sqrt_ac_bd_le_sqrt_ef_l115_115161


namespace repair_time_and_earnings_l115_115663

-- Definitions based on given conditions
def cars : ℕ := 10
def cars_repair_50min : ℕ := 6
def repair_time_50min : ℕ := 50 -- minutes per car
def longer_percentage : ℕ := 80 -- 80% longer for the remaining cars
def wage_per_hour : ℕ := 30 -- dollars per hour

-- Remaining cars to repair
def remaining_cars : ℕ := cars - cars_repair_50min

-- Calculate total repair time for each type of cars and total repair time
def repair_time_remaining_cars : ℕ := repair_time_50min + (repair_time_50min * longer_percentage) / 100
def total_repair_time : ℕ := (cars_repair_50min * repair_time_50min) + (remaining_cars * repair_time_remaining_cars)

-- Convert total repair time from minutes to hours
def total_repair_hours : ℕ := total_repair_time / 60

-- Calculate total earnings
def total_earnings : ℕ := wage_per_hour * total_repair_hours

-- The theorem to be proved: total_repair_time == 660 and total_earnings == 330
theorem repair_time_and_earnings :
  total_repair_time = 660 ∧ total_earnings = 330 := by
  sorry

end repair_time_and_earnings_l115_115663


namespace xiao_hua_correct_answers_l115_115753

theorem xiao_hua_correct_answers :
  ∃ (correct_answers wrong_answers : ℕ), 
    correct_answers + wrong_answers = 15 ∧
    8 * correct_answers - 4 * wrong_answers = 72 ∧
    correct_answers = 11 :=
by
  sorry

end xiao_hua_correct_answers_l115_115753


namespace range_of_m_l115_115377

def one_root_condition (m : ℝ) : Prop :=
  (4 - 4 * m) * (2 * m + 4) ≤ 0 ∧ m ≠ 0

theorem range_of_m : {m : ℝ | (4 - 4 * m) * (2 * m + 4) ≤ 0 ∧ m ≠ 0} = {m | m ≤ -2 ∨ m ≥ 1} :=
by
  sorry

end range_of_m_l115_115377


namespace average_all_results_l115_115794

theorem average_all_results (s₁ s₂ : ℤ) (n₁ n₂ : ℤ) (h₁ : n₁ = 60) (h₂ : n₂ = 40) (avg₁ : s₁ / n₁ = 40) (avg₂ : s₂ / n₂ = 60) : 
  ((s₁ + s₂) / (n₁ + n₂) = 48) :=
sorry

end average_all_results_l115_115794


namespace arctan_sum_is_pi_over_4_l115_115199

open Real

theorem arctan_sum_is_pi_over_4 (a b c : ℝ) (h1 : b = c) (h2 : c / (a + b) + a / (b + c) = 1) :
  arctan (c / (a + b)) + arctan (a / (b + c)) = π / 4 :=
by 
  sorry

end arctan_sum_is_pi_over_4_l115_115199


namespace double_inequality_l115_115504

variable (a b c : ℝ)

def triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem double_inequality (h : triangle_sides a b c) : 
  3 * (a * b + b * c + c * a) ≤ (a + b + c) ^ 2 ∧ (a + b + c) ^ 2 < 4 * (a * b + b * c + c * a) :=
by
  sorry

end double_inequality_l115_115504


namespace slope_of_parallel_line_l115_115232

-- Given condition: the equation of the line
def line_equation (x y : ℝ) : Prop := 2 * x - 4 * y = 9

-- Goal: the slope of any line parallel to the given line is 1/2
theorem slope_of_parallel_line (x y : ℝ) (m : ℝ) :
  (∀ x y, line_equation x y) → m = 1 / 2 := by
  sorry

end slope_of_parallel_line_l115_115232


namespace avg_of_7_consecutive_integers_l115_115653

theorem avg_of_7_consecutive_integers (c d : ℝ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 5.5 := by
  sorry

end avg_of_7_consecutive_integers_l115_115653


namespace novel_corona_high_students_l115_115914

theorem novel_corona_high_students (students_know_it_all students_karen_high total_students students_novel_corona : ℕ)
  (h1 : students_know_it_all = 50)
  (h2 : students_karen_high = 3 / 5 * students_know_it_all)
  (h3 : total_students = 240)
  (h4 : students_novel_corona = total_students - (students_know_it_all + students_karen_high))
  : students_novel_corona = 160 :=
sorry

end novel_corona_high_students_l115_115914


namespace greatest_number_of_quarters_l115_115532

def eva_has_us_coins : ℝ := 4.80
def quarters_and_dimes_have_same_count (q : ℕ) : Prop := (0.25 * q + 0.10 * q = eva_has_us_coins)

theorem greatest_number_of_quarters : ∃ (q : ℕ), quarters_and_dimes_have_same_count q ∧ q = 13 :=
sorry

end greatest_number_of_quarters_l115_115532


namespace problem1_problem2a_problem2b_l115_115019

noncomputable def x : ℝ := Real.sqrt 6 - Real.sqrt 2
noncomputable def a : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem problem1 : x * (Real.sqrt 6 - x) + (x + Real.sqrt 5) * (x - Real.sqrt 5) = 1 - 2 * Real.sqrt 3 := 
by
  sorry

theorem problem2a : a - b = 2 * Real.sqrt 2 := 
by 
  sorry

theorem problem2b : a^2 - 2 * a * b + b^2 = 8 := 
by 
  sorry

end problem1_problem2a_problem2b_l115_115019


namespace smallest_four_digit_divisible_by_3_and_8_l115_115299

theorem smallest_four_digit_divisible_by_3_and_8 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 3 = 0 ∧ n % 8 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 3 = 0 ∧ m % 8 = 0 → n ≤ m := by
  sorry

end smallest_four_digit_divisible_by_3_and_8_l115_115299


namespace last_number_is_two_l115_115616

theorem last_number_is_two (A B C D : ℝ)
  (h1 : A + B + C = 18)
  (h2 : B + C + D = 9)
  (h3 : A + D = 13) :
  D = 2 :=
sorry

end last_number_is_two_l115_115616


namespace tree_height_at_2_years_l115_115531

theorem tree_height_at_2_years (h : ℕ → ℕ) 
  (h_growth : ∀ n, h (n + 1) = 3 * h n) 
  (h_5 : h 5 = 243) : 
  h 2 = 9 := 
sorry

end tree_height_at_2_years_l115_115531


namespace train_average_speed_l115_115818

theorem train_average_speed :
  let start_time := 9.0 -- Start time in hours (9:00 am)
  let end_time := 13.75 -- End time in hours (1:45 pm)
  let total_distance := 348.0 -- Total distance in km
  let halt_time := 0.75 -- Halt time in hours (45 minutes)
  let scheduled_time := end_time - start_time -- Total scheduled time in hours
  let actual_travel_time := scheduled_time - halt_time -- Actual travel time in hours
  let average_speed := total_distance / actual_travel_time -- Average speed formula
  average_speed = 87.0 := sorry

end train_average_speed_l115_115818


namespace candy_bars_eaten_l115_115554

theorem candy_bars_eaten (calories_per_candy : ℕ) (total_calories : ℕ) (h1 : calories_per_candy = 31) (h2 : total_calories = 341) :
  total_calories / calories_per_candy = 11 :=
by
  sorry

end candy_bars_eaten_l115_115554


namespace range_of_years_of_service_l115_115349

theorem range_of_years_of_service : 
  let years := [15, 10, 9, 17, 6, 3, 14, 16]
  ∃ min max, (min ∈ years ∧ max ∈ years ∧ (max - min = 14)) :=
by 
  let years := [15, 10, 9, 17, 6, 3, 14, 16]
  use 3, 17 
  sorry

end range_of_years_of_service_l115_115349


namespace eighteen_women_time_l115_115509

theorem eighteen_women_time (h : ∀ (n : ℕ), n = 6 → ∀ (t : ℕ), t = 60 → true) : ∀ (n : ℕ), n = 18 → ∀ (t : ℕ), t = 20 → true :=
by
  sorry

end eighteen_women_time_l115_115509


namespace product_of_five_consecutive_not_square_l115_115007

theorem product_of_five_consecutive_not_square (n : ℤ) :
  ¬ ∃ k : ℤ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = k^2) :=
by
  sorry

end product_of_five_consecutive_not_square_l115_115007


namespace find_number_l115_115445

theorem find_number (x : ℝ) (h : x - (3/5) * x = 62) : x = 155 :=
by
  sorry

end find_number_l115_115445


namespace joe_speed_l115_115313

theorem joe_speed (pete_speed : ℝ) (joe_speed : ℝ) (time_run : ℝ) (distance : ℝ) 
  (h1 : joe_speed = 2 * pete_speed)
  (h2 : time_run = 2 / 3)
  (h3 : distance = 16)
  (h4 : distance = 3 * pete_speed * time_run) :
  joe_speed = 16 :=
by sorry

end joe_speed_l115_115313


namespace triangle_height_l115_115266

theorem triangle_height (base area height : ℝ)
    (h_base : base = 4)
    (h_area : area = 16)
    (h_area_formula : area = (base * height) / 2) :
    height = 8 :=
by
  sorry

end triangle_height_l115_115266


namespace ratio_sum_eq_l115_115048

variable {x y z : ℝ}

-- Conditions: 3x, 4y, 5z form a geometric sequence
def geom_sequence (x y z : ℝ) : Prop :=
  (∃ r : ℝ, 4 * y = 3 * x * r ∧ 5 * z = 4 * y * r)

-- Conditions: 1/x, 1/y, 1/z form an arithmetic sequence
def arith_sequence (x y z : ℝ) : Prop :=
  2 * x * z = y * z + x * y

-- Conclude: x/z + z/x = 34/15
theorem ratio_sum_eq (h1 : geom_sequence x y z) (h2 : arith_sequence x y z) : 
  (x / z + z / x) = (34 / 15) :=
sorry

end ratio_sum_eq_l115_115048


namespace main_theorem_l115_115428

noncomputable def f : ℝ → ℝ := sorry

axiom h_even : ∀ x : ℝ, f (-x) = f x
axiom h_decreasing : ∀ x1 x2 : ℝ, x1 ≠ x2 → 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → 
  (x1 < x2 ↔ (f x2 < f x1))

theorem main_theorem : f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end main_theorem_l115_115428


namespace kiera_total_envelopes_l115_115003

-- Define the number of blue envelopes
def blue_envelopes : ℕ := 14

-- Define the number of yellow envelopes as 6 fewer than the number of blue envelopes
def yellow_envelopes : ℕ := blue_envelopes - 6

-- Define the number of green envelopes as 3 times the number of yellow envelopes
def green_envelopes : ℕ := 3 * yellow_envelopes

-- The total number of envelopes is the sum of blue, yellow, and green envelopes
def total_envelopes : ℕ := blue_envelopes + yellow_envelopes + green_envelopes

-- Prove that the total number of envelopes is 46
theorem kiera_total_envelopes : total_envelopes = 46 := by
  sorry

end kiera_total_envelopes_l115_115003


namespace complement_union_eq_l115_115077

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_union_eq :
  U \ (A ∪ B) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end complement_union_eq_l115_115077


namespace cell_chain_length_l115_115797

theorem cell_chain_length (d n : ℕ) (h₁ : d = 5 * 10^2) (h₂ : n = 2 * 10^3) : d * n = 10^6 :=
by
  sorry

end cell_chain_length_l115_115797


namespace average_of_ABC_l115_115795

theorem average_of_ABC (A B C : ℝ) 
  (h1 : 2002 * C - 1001 * A = 8008) 
  (h2 : 2002 * B + 3003 * A = 7007) 
  (h3 : A = 2) : (A + B + C) / 3 = 2.33 := 
by 
  sorry

end average_of_ABC_l115_115795


namespace contrapositive_statement_l115_115365

theorem contrapositive_statement 
  (a : ℝ) (b : ℝ) 
  (h1 : a > 0) 
  (h3 : a + b < 0) : 
  b < 0 :=
sorry

end contrapositive_statement_l115_115365


namespace smaller_circle_area_l115_115733

theorem smaller_circle_area (r R : ℝ) (hR : R = 3 * r)
  (hTangentLines : ∀ (P A B A' B' : ℝ), P = 5 ∧ A = 5 ∧ PA = 5 ∧ A' = 5 ∧ PA' = 5 ∧ AB = 5 ∧ A'B' = 5 ) :
  π * r^2 = 25 / 3 * π := by
  sorry

end smaller_circle_area_l115_115733


namespace equivalent_operation_l115_115582

theorem equivalent_operation (x : ℚ) : 
  (x * (2 / 3)) / (4 / 7) = x * (7 / 6) :=
by sorry

end equivalent_operation_l115_115582


namespace solve_base7_addition_problem_l115_115104

noncomputable def base7_addition_problem : Prop :=
  ∃ (X Y: ℕ), 
    (5 * 7^2 + X * 7 + Y) + (3 * 7^1 + 2) = 6 * 7^2 + 2 * 7 + X ∧
    X + Y = 10 

theorem solve_base7_addition_problem : base7_addition_problem :=
by sorry

end solve_base7_addition_problem_l115_115104


namespace no_equilateral_triangle_OAB_exists_l115_115102

theorem no_equilateral_triangle_OAB_exists :
  ∀ (A B : ℝ × ℝ), 
  ((∃ a : ℝ, A = (a, (3 / 2) ^ a)) ∧ B.1 > 0 ∧ B.2 = 0) → 
  ¬ (∃ k : ℝ, k = (A.2 / A.1) ∧ k > (3 ^ (1 / 2)) / 3) := 
by 
  intro A B h
  sorry

end no_equilateral_triangle_OAB_exists_l115_115102


namespace find_y_value_l115_115937

def op (a b : ℤ) : ℤ := 4 * a + 2 * b

theorem find_y_value : ∃ y : ℤ, op 3 (op 4 y) = -14 ∧ y = -29 / 2 := sorry

end find_y_value_l115_115937


namespace square_area_l115_115107

theorem square_area (x : ℝ) (side1 side2 : ℝ) 
  (h_side1 : side1 = 6 * x - 27) 
  (h_side2 : side2 = 30 - 2 * x) 
  (h_equiv : side1 = side2) : 
  (side1 * side1 = 248.0625) := 
by
  sorry

end square_area_l115_115107


namespace harry_basketball_points_l115_115604

theorem harry_basketball_points :
  ∃ (x y : ℕ), 
    (x < 15) ∧ 
    (y < 15) ∧ 
    (62 + x) % 11 = 0 ∧ 
    (62 + x + y) % 12 = 0 ∧ 
    (x * y = 24) :=
by
  sorry

end harry_basketball_points_l115_115604


namespace room_length_l115_115627

theorem room_length (width : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) (length : ℝ)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 28875)
  (h_cost_per_sqm : cost_per_sqm = 1400)
  (h_length : length = total_cost / cost_per_sqm / width) :
  length = 5.5 := by
  sorry

end room_length_l115_115627


namespace solve_for_x_l115_115882

theorem solve_for_x (x y : ℝ) (h1 : 3 * x - 2 * y = 8) (h2 : 2 * x + 3 * y = 1) : x = 2 := 
by 
  sorry

end solve_for_x_l115_115882


namespace solve_for_a_l115_115915

theorem solve_for_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 13) (h3 : 13 ∣ 51^2016 - a) : a = 1 :=
by {
  sorry
}

end solve_for_a_l115_115915


namespace geometric_sequence_sum_eq_five_l115_115728

/-- Given that {a_n} is a geometric sequence where each a_n > 0
    and the equation a_2 * a_4 + 2 * a_3 * a_5 + a_4 * a_6 = 25 holds,
    we want to prove that a_3 + a_5 = 5. -/
theorem geometric_sequence_sum_eq_five
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a n = a 1 * r ^ (n - 1))
  (h_pos : ∀ n, a n > 0)
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : a 3 + a 5 = 5 :=
sorry

end geometric_sequence_sum_eq_five_l115_115728


namespace equality_of_expressions_l115_115559

theorem equality_of_expressions :
  (2^3 ≠ 2 * 3) ∧
  (-(-2)^2 ≠ (-2)^2) ∧
  (-3^2 ≠ 3^2) ∧
  (-2^3 = (-2)^3) :=
by
  sorry

end equality_of_expressions_l115_115559


namespace one_quarter_between_l115_115139

def one_quarter_way (a b : ℚ) : ℚ :=
  a + 1 / 4 * (b - a)

theorem one_quarter_between :
  one_quarter_way (1 / 7) (1 / 4) = 23 / 112 :=
by
  sorry

end one_quarter_between_l115_115139


namespace probability_genuine_given_equal_weight_l115_115165

noncomputable def total_coins : ℕ := 15
noncomputable def genuine_coins : ℕ := 12
noncomputable def counterfeit_coins : ℕ := 3

def condition_A : Prop := true
def condition_B (weights : Fin 6 → ℝ) : Prop :=
  weights 0 + weights 1 = weights 2 + weights 3 ∧
  weights 0 + weights 1 = weights 4 + weights 5

noncomputable def P_A_and_B : ℚ := (44 / 70) * (15 / 26) * (28 / 55)
noncomputable def P_B : ℚ := 44 / 70

theorem probability_genuine_given_equal_weight :
  P_A_and_B / P_B = 264 / 443 :=
by
  sorry

end probability_genuine_given_equal_weight_l115_115165


namespace D_72_eq_93_l115_115726

def D (n : ℕ) : ℕ :=
-- The function definition of D would go here, but we leave it abstract for now.
sorry

theorem D_72_eq_93 : D 72 = 93 :=
sorry

end D_72_eq_93_l115_115726


namespace Sam_balloon_count_l115_115787

theorem Sam_balloon_count:
  ∀ (F M S : ℕ), F = 5 → M = 7 → (F + M + S = 18) → S = 6 :=
by
  intros F M S hF hM hTotal
  rw [hF, hM] at hTotal
  linarith

end Sam_balloon_count_l115_115787


namespace initial_fish_count_l115_115849

theorem initial_fish_count (F T : ℕ) 
  (h1 : T = 3 * F)
  (h2 : T / 2 = (F - 7) + 32) : F = 50 :=
by
  sorry

end initial_fish_count_l115_115849


namespace students_brought_apples_l115_115836

theorem students_brought_apples (A B C D : ℕ) (h1 : B = 8) (h2 : C = 10) (h3 : D = 5) (h4 : A - D + B - D = C) : A = 12 :=
by {
  sorry
}

end students_brought_apples_l115_115836


namespace initial_spiders_correct_l115_115072

-- Define the initial number of each type of animal
def initial_birds : Nat := 12
def initial_puppies : Nat := 9
def initial_cats : Nat := 5

-- Conditions about the changes in the number of animals
def birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_loose : Nat := 7

-- Number of animals left in the store
def total_animals_left : Nat := 25

-- Define the remaining animals after sales and adoptions
def remaining_birds : Nat := initial_birds - birds_sold
def remaining_puppies : Nat := initial_puppies - puppies_adopted
def remaining_cats : Nat := initial_cats

-- Define the remaining animals excluding spiders
def animals_without_spiders : Nat := remaining_birds + remaining_puppies + remaining_cats

-- Define the number of remaining spiders
def remaining_spiders : Nat := total_animals_left - animals_without_spiders

-- Prove the initial number of spiders
def initial_spiders : Nat := remaining_spiders + spiders_loose

theorem initial_spiders_correct :
  initial_spiders = 15 := by 
  sorry

end initial_spiders_correct_l115_115072


namespace coin_flip_sequences_l115_115454

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l115_115454


namespace eleven_place_unamed_racer_l115_115779

theorem eleven_place_unamed_racer
  (Rand Hikmet Jack Marta David Todd : ℕ)
  (positions : Fin 15)
  (C_1 : Rand = Hikmet + 6)
  (C_2 : Marta = Jack + 1)
  (C_3 : David = Hikmet + 3)
  (C_4 : Jack = Todd + 3)
  (C_5 : Todd = Rand + 1)
  (C_6 : Marta = 8) :
  ∃ (x : Fin 15), (x ≠ Rand) ∧ (x ≠ Hikmet) ∧ (x ≠ Jack) ∧ (x ≠ Marta) ∧ (x ≠ David) ∧ (x ≠ Todd) ∧ x = 11 := 
sorry

end eleven_place_unamed_racer_l115_115779


namespace bake_cookies_l115_115461

noncomputable def scale_factor (original_cookies target_cookies : ℕ) : ℕ :=
  target_cookies / original_cookies

noncomputable def required_flour (original_flour : ℕ) (scale : ℕ) : ℕ :=
  original_flour * scale

noncomputable def adjusted_sugar (original_sugar : ℕ) (scale : ℕ) (reduction_percent : ℚ) : ℚ :=
  original_sugar * scale * (1 - reduction_percent)

theorem bake_cookies 
  (original_cookies : ℕ)
  (target_cookies : ℕ)
  (original_flour : ℕ)
  (original_sugar : ℕ)
  (reduction_percent : ℚ)
  (h_original_cookies : original_cookies = 40)
  (h_target_cookies : target_cookies = 80)
  (h_original_flour : original_flour = 3)
  (h_original_sugar : original_sugar = 1)
  (h_reduction_percent : reduction_percent = 0.25) :
  required_flour original_flour (scale_factor original_cookies target_cookies) = 6 ∧ 
  adjusted_sugar original_sugar (scale_factor original_cookies target_cookies) reduction_percent = 1.5 := by
    sorry

end bake_cookies_l115_115461


namespace min_value_ratio_l115_115337

noncomputable def min_ratio (a : ℝ) (h : a > 0) : ℝ :=
  let x_A := 4^(-a)
  let x_B := 4^(a)
  let x_C := 4^(- (18 / (2*a + 1)))
  let x_D := 4^((18 / (2*a + 1)))
  let m := abs (x_A - x_C)
  let n := abs (x_B - x_D)
  n / m

theorem min_value_ratio (a : ℝ) (h : a > 0) : 
  ∃ c : ℝ, c = 2^11 := sorry

end min_value_ratio_l115_115337


namespace inequality_problem_l115_115412

theorem inequality_problem
  (a b c d e : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : c ≤ d)
  (h4 : d ≤ e)
  (h5 : a + b + c + d + e = 1) :
  a * d + d * c + c * b + b * e + e * a ≤ 1 / 5 :=
by
  sorry

end inequality_problem_l115_115412


namespace salt_concentration_l115_115115

theorem salt_concentration (volume_water volume_solution concentration_solution : ℝ)
  (h1 : volume_water = 1)
  (h2 : volume_solution = 0.5)
  (h3 : concentration_solution = 0.45) :
  (volume_solution * concentration_solution) / (volume_water + volume_solution) = 0.15 :=
by
  sorry

end salt_concentration_l115_115115


namespace nine_x_five_y_multiple_l115_115654

theorem nine_x_five_y_multiple (x y : ℤ) (h : 2 * x + 3 * y ≡ 0 [ZMOD 17]) : 
  9 * x + 5 * y ≡ 0 [ZMOD 17] := 
by
  sorry

end nine_x_five_y_multiple_l115_115654


namespace expected_score_particular_player_l115_115848

-- Define types of dice
inductive DiceType : Type
| A | B | C

-- Define the faces of each dice type
def DiceFaces : DiceType → List ℕ
| DiceType.A => [2, 2, 4, 4, 9, 9]
| DiceType.B => [1, 1, 6, 6, 8, 8]
| DiceType.C => [3, 3, 5, 5, 7, 7]

-- Define a function to calculate the score of a player given their roll and opponents' rolls
def player_score (p_roll : ℕ) (opp_rolls : List ℕ) : ℕ :=
  opp_rolls.foldl (λ acc roll => if roll < p_roll then acc + 1 else acc) 0

-- Define a function to calculate the expected score of a player
noncomputable def expected_score (dice_choice : DiceType) : ℚ :=
  let rolls := DiceFaces dice_choice
  let total_possibilities := (rolls.length : ℚ) ^ 3
  let score_sum := rolls.foldl (λ acc p_roll =>
    acc + rolls.foldl (λ acc1 opp1_roll =>
        acc1 + rolls.foldl (λ acc2 opp2_roll =>
            acc2 + player_score p_roll [opp1_roll, opp2_roll]
          ) 0
      ) 0
    ) 0
  score_sum / total_possibilities

-- The main theorem statement
theorem expected_score_particular_player : (expected_score DiceType.A + expected_score DiceType.B + expected_score DiceType.C) / 3 = 
(8 : ℚ) / 9 := sorry

end expected_score_particular_player_l115_115848


namespace sum_reciprocals_of_roots_l115_115822

theorem sum_reciprocals_of_roots (p q x₁ x₂ : ℝ) (h₀ : x₁ + x₂ = -p) (h₁ : x₁ * x₂ = q) :
  (1 / x₁ + 1 / x₂) = -p / q :=
by 
  sorry

end sum_reciprocals_of_roots_l115_115822


namespace crayons_allocation_correct_l115_115300

noncomputable def crayons_allocation : Prop :=
  ∃ (F B J S : ℕ), 
    F + B + J + S = 96 ∧ 
    F = 2 * B ∧ 
    J = 3 * S ∧ 
    B = 12 ∧ 
    F = 24 ∧ 
    J = 45 ∧ 
    S = 15

theorem crayons_allocation_correct : crayons_allocation :=
  sorry

end crayons_allocation_correct_l115_115300


namespace volume_after_increasing_edges_l115_115333

-- Defining the initial conditions and the theorem to prove regarding the volume.
theorem volume_after_increasing_edges {a b c : ℝ} 
  (h1 : a * b * c = 8) 
  (h2 : (a + 1) * (b + 1) * (c + 1) = 27) : 
  (a + 2) * (b + 2) * (c + 2) = 64 :=
sorry

end volume_after_increasing_edges_l115_115333


namespace team_E_has_not_played_against_team_B_l115_115410

-- We begin by defining the teams as an enumeration
inductive Team
| A | B | C | D | E | F

open Team

-- Define the total number of matches each team has played
def matches_played (t : Team) : Nat :=
  match t with
  | A => 5
  | B => 4
  | C => 3
  | D => 2
  | E => 1
  | F => 0 -- Note: we assume F's matches are not provided; this can be adjusted if needed

-- Prove that team E has not played against team B
theorem team_E_has_not_played_against_team_B :
  ∃ t : Team, matches_played B = 4 ∧ matches_played E < matches_played B ∧
  (t = E) :=
by
  sorry

end team_E_has_not_played_against_team_B_l115_115410


namespace hypotenuse_length_l115_115721

theorem hypotenuse_length (a b c : ℝ) (h1 : a + b + c = 32) (h2 : a * b = 40) (h3 : a^2 + b^2 = c^2) : 
  c = 59 / 4 :=
by
  sorry

end hypotenuse_length_l115_115721


namespace min_value_expression_l115_115524

theorem min_value_expression (x : ℝ) (h : x > 10) : (x^2) / (x - 10) ≥ 40 :=
sorry

end min_value_expression_l115_115524


namespace coordinate_plane_line_l115_115245

theorem coordinate_plane_line (m n p : ℝ) (h1 : m = n / 5 - 2 / 5) (h2 : m + p = (n + 15) / 5 - 2 / 5) : p = 3 := by
  sorry

end coordinate_plane_line_l115_115245


namespace man_speed_is_correct_l115_115092

noncomputable def speed_of_man (train_length : ℝ) (train_speed : ℝ) (cross_time : ℝ) : ℝ :=
  let train_speed_m_s := train_speed * (1000 / 3600)
  let relative_speed := train_length / cross_time
  let man_speed_m_s := relative_speed - train_speed_m_s
  man_speed_m_s * (3600 / 1000)

theorem man_speed_is_correct :
  speed_of_man 210 25 28 = 2 := by
  sorry

end man_speed_is_correct_l115_115092


namespace gumball_machine_total_gumballs_l115_115430

/-- A gumball machine has red, green, and blue gumballs. Given the following conditions:
1. The machine has half as many blue gumballs as red gumballs.
2. For each blue gumball, the machine has 4 times as many green gumballs.
3. The machine has 16 red gumballs.
Prove that the total number of gumballs in the machine is 56. -/
theorem gumball_machine_total_gumballs :
  ∀ (red blue green : ℕ),
    (blue = red / 2) →
    (green = blue * 4) →
    red = 16 →
    (red + blue + green = 56) :=
by
  intros red blue green h_blue h_green h_red
  sorry

end gumball_machine_total_gumballs_l115_115430


namespace problem_graph_empty_l115_115081

open Real

theorem problem_graph_empty : ∀ x y : ℝ, ¬ (x^2 + 3 * y^2 - 4 * x - 12 * y + 28 = 0) :=
by
  intro x y
  -- Apply the contradiction argument based on the conditions given
  sorry


end problem_graph_empty_l115_115081


namespace distance_between_points_l115_115086

theorem distance_between_points (x y : ℝ) (h : x + y = 10 / 3) : 
  4 * (x + y) = 40 / 3 :=
sorry

end distance_between_points_l115_115086


namespace find_a_and_b_l115_115812

-- Define the two numbers a and b and the given conditions
variables (a b : ℕ)
variables (h1 : a - b = 831) (h2 : a = 21 * b + 11)

-- State the theorem to find the values of a and b
theorem find_a_and_b (a b : ℕ) (h1 : a - b = 831) (h2 : a = 21 * b + 11) : a = 872 ∧ b = 41 :=
by
  sorry

end find_a_and_b_l115_115812


namespace profit_per_meter_l115_115443

theorem profit_per_meter 
  (total_meters : ℕ)
  (cost_price_per_meter : ℝ)
  (total_selling_price : ℝ)
  (h1 : total_meters = 92)
  (h2 : cost_price_per_meter = 83.5)
  (h3 : total_selling_price = 9890) : 
  (total_selling_price - total_meters * cost_price_per_meter) / total_meters = 24.1 :=
by
  sorry

end profit_per_meter_l115_115443


namespace no_such_geometric_sequence_exists_l115_115484

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ q : ℝ, a (n + 1) = q * a n

noncomputable def satisfies_conditions (a : ℕ → ℝ) : Prop :=
(a 1 + a 6 = 11) ∧
(a 3 * a 4 = 32 / 9) ∧
(∀ n : ℕ, a (n + 1) > a n) ∧
(∃ m : ℕ, m > 4 ∧ (2 * a m^2 = (2 / 3 * a (m - 1) + (a (m + 1) + 4 / 9))))

theorem no_such_geometric_sequence_exists : 
  ¬ ∃ a : ℕ → ℝ, geometric_sequence a ∧ satisfies_conditions a := 
sorry

end no_such_geometric_sequence_exists_l115_115484


namespace choir_females_correct_l115_115411

noncomputable def number_of_females_in_choir : ℕ :=
  let orchestra_males := 11
  let orchestra_females := 12
  let orchestra_musicians := orchestra_males + orchestra_females
  let band_males := 2 * orchestra_males
  let band_females := 2 * orchestra_females
  let band_musicians := 2 * orchestra_musicians
  let total_musicians := 98
  let choir_males := 12
  let choir_musicians := total_musicians - (orchestra_musicians + band_musicians)
  let choir_females := choir_musicians - choir_males
  choir_females

theorem choir_females_correct : number_of_females_in_choir = 17 := by
  sorry

end choir_females_correct_l115_115411


namespace purely_imaginary_sol_l115_115314

theorem purely_imaginary_sol (x : ℝ) 
  (h1 : (x^2 - 1) = 0)
  (h_imag : (x^2 + 3 * x + 2) ≠ 0) :
  x = 1 :=
sorry

end purely_imaginary_sol_l115_115314


namespace decreased_amount_l115_115920

theorem decreased_amount {N A : ℝ} (h₁ : 0.20 * N - A = 6) (h₂ : N = 50) : A = 4 := by
  sorry

end decreased_amount_l115_115920


namespace probability_of_triangle_or_circle_l115_115459

-- Definitions (conditions)
def total_figures : ℕ := 12
def triangles : ℕ := 4
def circles : ℕ := 3
def squares : ℕ := 5
def figures : ℕ := triangles + circles + squares

-- Probability calculation
def probability_triangle_circle := (triangles + circles) / total_figures

-- Theorem statement (problem)
theorem probability_of_triangle_or_circle : probability_triangle_circle = 7 / 12 :=
by
  -- The proof is omitted, insert the proof here when necessary.
  sorry

end probability_of_triangle_or_circle_l115_115459


namespace maria_total_baggies_l115_115700

def choc_chip_cookies := 33
def oatmeal_cookies := 2
def cookies_per_bag := 5

def total_cookies := choc_chip_cookies + oatmeal_cookies

def total_baggies (total_cookies : Nat) (cookies_per_bag : Nat) : Nat :=
  total_cookies / cookies_per_bag

theorem maria_total_baggies : total_baggies total_cookies cookies_per_bag = 7 :=
  by
    -- Steps proving the equivalence can be done here
    sorry

end maria_total_baggies_l115_115700


namespace imaginary_number_m_l115_115856

theorem imaginary_number_m (m : ℝ) : 
  (∀ Z, Z = (m + 2 * Complex.I) / (1 + Complex.I) → Z.im = 0 → Z.re = 0) → m = -2 :=
by
  sorry

end imaginary_number_m_l115_115856


namespace exists_irrationals_floor_neq_l115_115498

-- Define irrationality of a number
def irrational (x : ℝ) : Prop :=
  ¬ ∃ (r : ℚ), x = r

theorem exists_irrationals_floor_neq :
  ∃ (a b : ℝ), irrational a ∧ irrational b ∧ 1 < a ∧ 1 < b ∧ 
  ∀ (m n : ℕ), ⌊a ^ m⌋ ≠ ⌊b ^ n⌋ :=
by
  sorry

end exists_irrationals_floor_neq_l115_115498


namespace fraction_equality_l115_115556

theorem fraction_equality : (16 : ℝ) / (8 * 17) = (1.6 : ℝ) / (0.8 * 17) := 
sorry

end fraction_equality_l115_115556


namespace exists_ratios_eq_l115_115326

theorem exists_ratios_eq (a b z : ℕ) (ha : 0 < a) (hb : 0 < b) (hz : 0 < z) (h : a * b = z^2 + 1) :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ (a : ℚ) / b = (x^2 + 1) / (y^2 + 1) :=
by
  sorry

end exists_ratios_eq_l115_115326


namespace distance_between_points_l115_115886

open Complex Real

def joe_point : ℂ := 2 + 3 * I
def gracie_point : ℂ := -2 + 2 * I

theorem distance_between_points : abs (joe_point - gracie_point) = sqrt 17 := by
  sorry

end distance_between_points_l115_115886


namespace tina_total_income_is_correct_l115_115452

-- Definitions based on the conditions
def hourly_wage : ℝ := 18.0
def regular_hours_per_day : ℝ := 8
def overtime_hours_per_day_weekday : ℝ := 2
def double_overtime_hours_per_day_weekend : ℝ := 2

def overtime_rate : ℝ := hourly_wage + 0.5 * hourly_wage
def double_overtime_rate : ℝ := 2 * hourly_wage

def weekday_hours_per_day : ℝ := 10
def weekend_hours_per_day : ℝ := 12

def regular_pay_per_day : ℝ := hourly_wage * regular_hours_per_day
def overtime_pay_per_day_weekday : ℝ := overtime_rate * overtime_hours_per_day_weekday
def double_overtime_pay_per_day_weekend : ℝ := double_overtime_rate * double_overtime_hours_per_day_weekend

def total_weekday_pay_per_day : ℝ := regular_pay_per_day + overtime_pay_per_day_weekday
def total_weekend_pay_per_day : ℝ := regular_pay_per_day + overtime_pay_per_day_weekday + double_overtime_pay_per_day_weekend

def number_of_weekdays : ℝ := 5
def number_of_weekends : ℝ := 2

def total_weekday_income : ℝ := total_weekday_pay_per_day * number_of_weekdays
def total_weekend_income : ℝ := total_weekend_pay_per_day * number_of_weekends

def total_weekly_income : ℝ := total_weekday_income + total_weekend_income

-- The theorem we need to prove
theorem tina_total_income_is_correct : total_weekly_income = 1530 := by
  sorry

end tina_total_income_is_correct_l115_115452


namespace negation_of_proposition_l115_115060

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2 * x + 4 > 0) :=
by sorry

end negation_of_proposition_l115_115060


namespace quadratic_inequality_condition_l115_115593

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) → 0 ≤ a ∧ a < 4 :=
sorry

end quadratic_inequality_condition_l115_115593


namespace solve_for_three_times_x_plus_ten_l115_115424

theorem solve_for_three_times_x_plus_ten (x : ℝ) (h_eq : 5 * x - 7 = 15 * x + 21) : 3 * (x + 10) = 21.6 := by
  sorry

end solve_for_three_times_x_plus_ten_l115_115424


namespace speed_of_boat_in_still_water_l115_115297

variable (x : ℝ)

theorem speed_of_boat_in_still_water (h : 10 = (x + 5) * 0.4) : x = 20 :=
sorry

end speed_of_boat_in_still_water_l115_115297


namespace number_of_trips_l115_115310

theorem number_of_trips (bags_per_trip : ℕ) (weight_per_bag : ℕ) (total_weight : ℕ)
  (h1 : bags_per_trip = 10)
  (h2 : weight_per_bag = 50)
  (h3 : total_weight = 10000) : 
  total_weight / (bags_per_trip * weight_per_bag) = 20 :=
by
  sorry

end number_of_trips_l115_115310


namespace necessary_and_sufficient_condition_l115_115565

variables {a : ℕ → ℝ}
-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Define the monotonically increasing condition
def is_monotonically_increasing (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

-- Define the specific statement
theorem necessary_and_sufficient_condition (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 < a 3 ↔ is_monotonically_increasing a) :=
by sorry

end necessary_and_sufficient_condition_l115_115565


namespace solution_set_equivalence_l115_115191

theorem solution_set_equivalence (a : ℝ) : 
    (-1 < a ∧ a < 1) ∧ (3 * a^2 - 2 * a - 5 < 0) → 
    (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) :=
by
    sorry

end solution_set_equivalence_l115_115191


namespace Kiera_envelopes_total_l115_115368

-- Define variables for different colored envelopes
def E_b : ℕ := 120
def E_y : ℕ := E_b - 25
def E_g : ℕ := 5 * E_y
def E_r : ℕ := (E_b + E_y) / 2  -- integer division in lean automatically rounds down
def E_p : ℕ := E_r + 71
def E_total : ℕ := E_b + E_y + E_g + E_r + E_p

-- The statement to be proven
theorem Kiera_envelopes_total : E_total = 975 := by
  -- intentionally put the sorry to mark the proof as unfinished
  sorry

end Kiera_envelopes_total_l115_115368


namespace krishan_money_l115_115205

theorem krishan_money 
  (x y : ℝ)
  (hx1 : 7 * x * 1.185 = 699.8)
  (hx2 : 10 * x * 0.8 = 800)
  (hy : 17 * x = 8 * y) : 
  16 * y = 3400 := 
by
  -- It's acceptable to leave the proof incomplete due to the focus being on the statement.
  sorry

end krishan_money_l115_115205


namespace equivalent_single_discount_l115_115741

theorem equivalent_single_discount (x : ℝ) : 
  (1 - 0.15) * (1 - 0.20) * (1 - 0.10) = 1 - 0.388 :=
by
  sorry

end equivalent_single_discount_l115_115741


namespace unique_array_count_l115_115552

theorem unique_array_count (n m : ℕ) (h_conds : n * m = 49 ∧ n ≥ 2 ∧ m ≥ 2 ∧ n = m) :
  ∃! (n m : ℕ), (n * m = 49 ∧ n ≥ 2 ∧ m ≥ 2 ∧ n = m) :=
by
  sorry

end unique_array_count_l115_115552


namespace no_positive_integer_solutions_l115_115789

theorem no_positive_integer_solutions (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 0) :
  ¬ ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * (x + 1) = p^(2 * n) * y * (y + 1) :=
by
  sorry

end no_positive_integer_solutions_l115_115789


namespace minimize_transport_cost_l115_115934

noncomputable def total_cost (v : ℝ) (a : ℝ) : ℝ :=
  if v > 0 ∧ v ≤ 80 then
    1000 * (v / 4 + a / v)
  else
    0

theorem minimize_transport_cost :
  ∀ v a : ℝ, a = 400 → (0 < v ∧ v ≤ 80) → total_cost v a = 20000 → v = 40 :=
by
  intros v a ha h_dom h_cost
  sorry

end minimize_transport_cost_l115_115934


namespace missing_number_evaluation_l115_115673

theorem missing_number_evaluation (x : ℝ) (h : |4 + 9 * x| - 6 = 70) : x = 8 :=
sorry

end missing_number_evaluation_l115_115673


namespace sufficient_but_not_necessary_condition_l115_115223

variable (x : ℝ)

def p := x > 2
def q := x^2 > 4

theorem sufficient_but_not_necessary_condition : (p x) → (q x) ∧ ¬((q x) → (p x)) := 
by
  sorry

end sufficient_but_not_necessary_condition_l115_115223


namespace solve_for_x_l115_115952

theorem solve_for_x (x : ℝ) (h : 1 = 1 / (4 * x^2 + 2 * x + 1)) : 
  x = 0 ∨ x = -1 / 2 := 
by sorry

end solve_for_x_l115_115952


namespace pieces_not_chewed_l115_115989

theorem pieces_not_chewed : 
  (8 * 7 - 54) = 2 := 
by 
  sorry

end pieces_not_chewed_l115_115989


namespace leah_birds_duration_l115_115458

-- Define the conditions
def boxes_bought : ℕ := 3
def boxes_existing : ℕ := 5
def parrot_weekly_consumption : ℕ := 100
def cockatiel_weekly_consumption : ℕ := 50
def grams_per_box : ℕ := 225

-- Define the question as a theorem
theorem leah_birds_duration : 
  (boxes_bought + boxes_existing) * grams_per_box / 
  (parrot_weekly_consumption + cockatiel_weekly_consumption) = 12 :=
by
  -- Proof would go here
  sorry

end leah_birds_duration_l115_115458


namespace exists_divisible_by_2021_l115_115062

def concatenated_number (n m : ℕ) : ℕ := 
  -- This function should concatenate the digits from n to m inclusively
  sorry

theorem exists_divisible_by_2021 : ∃ (n m : ℕ), n > m ∧ m ≥ 1 ∧ 2021 ∣ concatenated_number n m :=
by
  sorry

end exists_divisible_by_2021_l115_115062


namespace hall_width_to_length_ratio_l115_115889

def width (w l : ℝ) : Prop := w * l = 578
def length_width_difference (w l : ℝ) : Prop := l - w = 17

theorem hall_width_to_length_ratio (w l : ℝ) (hw : width w l) (hl : length_width_difference w l) : (w / l = 1 / 2) :=
by
  sorry

end hall_width_to_length_ratio_l115_115889


namespace new_student_weight_l115_115895

theorem new_student_weight : 
  ∀ (w_new : ℕ), 
    (∀ (sum_weight: ℕ), 80 + sum_weight - w_new = sum_weight - 18) → 
      w_new = 62 := 
by
  intros w_new h
  sorry

end new_student_weight_l115_115895


namespace find_N_l115_115520

-- Definitions and conditions directly appearing in the problem
variable (X Y Z N : ℝ)

axiom condition1 : 0.15 * X = 0.25 * N + Y
axiom condition2 : X + Y = Z

-- The theorem to prove
theorem find_N : N = 4.6 * X - 4 * Z := by
  sorry

end find_N_l115_115520


namespace smallest_positive_period_maximum_f_B_l115_115991

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi / 3) - Real.sqrt 3 / 2

theorem smallest_positive_period (x : ℝ) : 
  (∀ T, (f (x + T) = f x) → (T ≥ 0) → T = Real.pi) := 
sorry

variable {a b c : ℝ}

lemma cos_law_cos_B (h : b^2 = a * c) : 
  ∀ (B : ℝ), (B > 0) ∧ (B < Real.pi) → 
  (1 / 2) ≤ Real.cos B ∧ Real.cos B < 1 := 
sorry

theorem maximum_f_B (h : b^2 = a * c) :
  ∀ (B : ℝ), (B > 0) ∧ (B < Real.pi) → 
  f B ≤ 1 := 
sorry

end smallest_positive_period_maximum_f_B_l115_115991


namespace team_testing_equation_l115_115209

variable (x : ℝ)

theorem team_testing_equation (h : x > 15) : (600 / x = 500 / (x - 15) * 0.9) :=
sorry

end team_testing_equation_l115_115209


namespace annulus_area_l115_115435

variables {R r d : ℝ}
variables (h1 : R > r) (h2 : d < R)

theorem annulus_area :
  π * (R^2 - r^2 - d^2 / (R - r)) = π * ((R - r)^2 - d^2) :=
sorry

end annulus_area_l115_115435


namespace methane_needed_l115_115142

theorem methane_needed (total_benzene_g : ℝ) (molar_mass_benzene : ℝ) (toluene_moles : ℝ) : 
  total_benzene_g = 156 ∧ molar_mass_benzene = 78 ∧ toluene_moles = 2 → 
  toluene_moles = total_benzene_g / molar_mass_benzene := 
by
  intros
  sorry

end methane_needed_l115_115142


namespace combined_number_of_fasteners_l115_115988

def lorenzo_full_cans_total_fasteners
  (thumbtacks_cans : ℕ)
  (pushpins_cans : ℕ)
  (staples_cans : ℕ)
  (thumbtacks_per_board : ℕ)
  (pushpins_per_board : ℕ)
  (staples_per_board : ℕ)
  (boards_tested : ℕ)
  (thumbtacks_remaining : ℕ)
  (pushpins_remaining : ℕ)
  (staples_remaining : ℕ) :
  ℕ :=
  let thumbtacks_used := thumbtacks_per_board * boards_tested
  let pushpins_used := pushpins_per_board * boards_tested
  let staples_used := staples_per_board * boards_tested
  let thumbtacks_per_can := thumbtacks_used + thumbtacks_remaining
  let pushpins_per_can := pushpins_used + pushpins_remaining
  let staples_per_can := staples_used + staples_remaining
  let total_thumbtacks := thumbtacks_per_can * thumbtacks_cans
  let total_pushpins := pushpins_per_can * pushpins_cans
  let total_staples := staples_per_can * staples_cans
  total_thumbtacks + total_pushpins + total_staples

theorem combined_number_of_fasteners :
  lorenzo_full_cans_total_fasteners 5 3 2 3 2 4 150 45 35 25 = 4730 :=
  by
  sorry

end combined_number_of_fasteners_l115_115988


namespace probability_neither_test_l115_115137

theorem probability_neither_test (P_hist : ℚ) (P_geo : ℚ) (indep : Prop) 
  (H1 : P_hist = 5/9) (H2 : P_geo = 1/3) (H3 : indep) :
  (1 - P_hist) * (1 - P_geo) = 8/27 := by
  sorry

end probability_neither_test_l115_115137


namespace rectangle_ratio_l115_115954

theorem rectangle_ratio (s : ℝ) (x y : ℝ) 
  (h_outer_area : x * y * 4 + s^2 = 9 * s^2)
  (h_inner_outer_relation : s + 2 * y = 3 * s) :
  x / y = 2 :=
by {
  sorry
}

end rectangle_ratio_l115_115954


namespace lemons_needed_l115_115397

theorem lemons_needed (initial_lemons : ℝ) (initial_gallons : ℝ) 
  (reduced_ratio : ℝ) (first_gallons : ℝ) (total_gallons : ℝ) :
  initial_lemons / initial_gallons * first_gallons 
  + (initial_lemons / initial_gallons * reduced_ratio) * (total_gallons - first_gallons) = 56.25 :=
by 
  let initial_ratio := initial_lemons / initial_gallons
  let reduced_ratio_amount := initial_ratio * reduced_ratio 
  let lemons_first := initial_ratio * first_gallons
  let lemons_remaining := reduced_ratio_amount * (total_gallons - first_gallons)
  let total_lemons := lemons_first + lemons_remaining
  show total_lemons = 56.25
  sorry

end lemons_needed_l115_115397


namespace one_positive_real_solution_l115_115981

theorem one_positive_real_solution : 
    ∃! x : ℝ, 0 < x ∧ (x ^ 10 + 7 * x ^ 9 + 14 * x ^ 8 + 1729 * x ^ 7 - 1379 * x ^ 6 = 0) :=
sorry

end one_positive_real_solution_l115_115981


namespace monkeys_and_bananas_l115_115778

theorem monkeys_and_bananas (m1 m2 t b1 b2 : ℕ) (h1 : m1 = 8) (h2 : t = 8) (h3 : b1 = 8) (h4 : b2 = 3) : m2 = 3 :=
by
  -- Here we will include the formal proof steps
  sorry

end monkeys_and_bananas_l115_115778


namespace average_community_age_l115_115194

variable (num_women num_men : Nat)
variable (avg_age_women avg_age_men : Nat)

def ratio_women_men := num_women = 7 * num_men / 8
def average_age_women := avg_age_women = 30
def average_age_men := avg_age_men = 35

theorem average_community_age (k : Nat) 
  (h_ratio : ratio_women_men (7 * k) (8 * k)) 
  (h_avg_women : average_age_women 30)
  (h_avg_men : average_age_men 35) : 
  (30 * (7 * k) + 35 * (8 * k)) / (15 * k) = 32 + (2 / 3) := 
sorry

end average_community_age_l115_115194


namespace sum_kml_l115_115408

theorem sum_kml (k m l : ℤ) (b : ℤ → ℤ)
  (h_seq : ∀ n, ∃ k, b n = k * (Int.floor (Real.sqrt (n + m : ℝ))) + l)
  (h_b1 : b 1 = 2) :
  k + m + l = 3 := by
  sorry

end sum_kml_l115_115408


namespace ramola_rank_from_first_l115_115220

-- Conditions definitions
def total_students : ℕ := 26
def ramola_rank_from_last : ℕ := 13

-- Theorem statement
theorem ramola_rank_from_first : total_students - (ramola_rank_from_last - 1) = 14 := 
by 
-- We use 'by' to begin the proof block
sorry 
-- We use 'sorry' to indicate the proof is omitted

end ramola_rank_from_first_l115_115220


namespace profit_functions_properties_l115_115425

noncomputable def R (x : ℝ) : ℝ := 3000 * x - 20 * x^2
noncomputable def C (x : ℝ) : ℝ := 500 * x + 4000
noncomputable def P (x : ℝ) : ℝ := R x - C x
noncomputable def MP (x : ℝ) : ℝ := P (x + 1) - P x

theorem profit_functions_properties :
  (P x = -20 * x^2 + 2500 * x - 4000) ∧ 
  (MP x = -40 * x + 2480) ∧ 
  (∃ x_max₁, ∀ x, P x_max₁ ≥ P x) ∧ 
  (∃ x_max₂, ∀ x, MP x_max₂ ≥ MP x) ∧ 
  P x_max₁ ≠ MP x_max₂ := by
  sorry

end profit_functions_properties_l115_115425


namespace jessica_seashells_l115_115821

theorem jessica_seashells (joan jessica total : ℕ) (h1 : joan = 6) (h2 : total = 14) (h3 : total = joan + jessica) : jessica = 8 :=
by
  -- proof steps would go here
  sorry

end jessica_seashells_l115_115821


namespace min_value_sum_reciprocal_squares_l115_115105

open Real

theorem min_value_sum_reciprocal_squares 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :  
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ 27 := 
sorry

end min_value_sum_reciprocal_squares_l115_115105


namespace initial_pipes_count_l115_115455

theorem initial_pipes_count (n : ℕ) (r : ℝ) :
  n * r = 1 / 16 → (n + 15) * r = 1 / 4 → n = 5 :=
by
  intro h1 h2
  sorry

end initial_pipes_count_l115_115455


namespace integer_solution_abs_lt_sqrt2_l115_115859

theorem integer_solution_abs_lt_sqrt2 (x : ℤ) (h : |x| < Real.sqrt 2) : x = -1 ∨ x = 0 ∨ x = 1 :=
sorry

end integer_solution_abs_lt_sqrt2_l115_115859


namespace find_rate_of_new_machine_l115_115117

noncomputable def rate_of_new_machine (R : ℝ) : Prop :=
  let old_rate := 100
  let total_bolts := 350
  let time_in_hours := 84 / 60
  let bolts_by_old_machine := old_rate * time_in_hours
  let bolts_by_new_machine := total_bolts - bolts_by_old_machine
  R = bolts_by_new_machine / time_in_hours

theorem find_rate_of_new_machine : rate_of_new_machine 150 :=
by
  sorry

end find_rate_of_new_machine_l115_115117


namespace rectangle_diagonal_length_l115_115951

theorem rectangle_diagonal_length :
  ∀ (length width diagonal : ℝ), length = 6 ∧ length * width = 48 ∧ diagonal = Real.sqrt (length^2 + width^2) → diagonal = 10 :=
by
  intro length width diagonal
  rintro ⟨hl, area_eq, diagonal_eq⟩
  sorry

end rectangle_diagonal_length_l115_115951


namespace exists_perfect_square_sum_l115_115206

theorem exists_perfect_square_sum (n : ℕ) (h : n > 2) : ∃ m : ℕ, ∃ k : ℕ, n^2 + m^2 = k^2 :=
by
  sorry

end exists_perfect_square_sum_l115_115206


namespace A_work_days_l115_115340

variables (r_A r_B r_C : ℝ) (h1 : r_A + r_B = (1 / 3)) (h2 : r_B + r_C = (1 / 3)) (h3 : r_A + r_C = (5 / 24))

theorem A_work_days :
  1 / r_A = 9.6 := 
sorry

end A_work_days_l115_115340


namespace part1_part2_l115_115351

-- Part 1: Inequality solution
theorem part1 (x : ℝ) :
  (1 / 3 * x - (3 * x + 4) / 6 ≤ 2 / 3) → (x ≥ -8) := 
by
  intro h
  sorry

-- Part 2: System of inequalities solution
theorem part2 (x : ℝ) :
  (4 * (x + 1) ≤ 7 * x + 13) ∧ ((x + 2) / 3 - x / 2 > 1) → (-3 ≤ x ∧ x < -2) := 
by
  intro h
  sorry

end part1_part2_l115_115351


namespace abs_difference_of_squares_l115_115982

theorem abs_difference_of_squares : abs ((102: ℤ) ^ 2 - (98: ℤ) ^ 2) = 800 := by
  sorry

end abs_difference_of_squares_l115_115982


namespace minimize_sum_find_c_l115_115924

theorem minimize_sum_find_c (a b c d e f : ℕ) (h : a + 2 * b + 6 * c + 30 * d + 210 * e + 2310 * f = 2 ^ 15) 
  (h_min : ∀ a' b' c' d' e' f' : ℕ, a' + 2 * b' + 6 * c' + 30 * d' + 210 * e' + 2310 * f' = 2 ^ 15 → 
  a' + b' + c' + d' + e' + f' ≥ a + b + c + d + e + f) :
  c = 1 :=
sorry

end minimize_sum_find_c_l115_115924


namespace orangeade_price_second_day_l115_115275

theorem orangeade_price_second_day :
  ∀ (X O : ℝ), (2 * X * 0.60 = 3 * X * E) → (E = 2 * 0.60 / 3) →
  E = 0.40 := by
  intros X O h₁ h₂
  sorry

end orangeade_price_second_day_l115_115275


namespace min_value_expression_l115_115233

theorem min_value_expression : 
  ∀ (x y : ℝ), (3 * x * x + 4 * x * y + 4 * y * y - 12 * x - 8 * y ≥ -28) ∧ 
  (3 * ((8:ℝ)/3) * ((8:ℝ)/3) + 4 * ((8:ℝ)/3) * -1 + 4 * -1 * -1 - 12 * ((8:ℝ)/3) - 8 * -1 = -28) := 
by sorry

end min_value_expression_l115_115233


namespace exists_sum_or_diff_divisible_by_1000_l115_115190

theorem exists_sum_or_diff_divisible_by_1000 (nums : Fin 502 → Nat) :
  ∃ a b : Nat, (∃ i j : Fin 502, nums i = a ∧ nums j = b ∧ i ≠ j) ∧
  (a - b) % 1000 = 0 ∨ (a + b) % 1000 = 0 :=
by
  sorry

end exists_sum_or_diff_divisible_by_1000_l115_115190


namespace balls_into_boxes_all_ways_balls_into_boxes_one_empty_l115_115276

/-- There are 4 different balls and 4 different boxes. -/
def balls : ℕ := 4
def boxes : ℕ := 4

/-- The number of ways to put 4 different balls into 4 different boxes is 256. -/
theorem balls_into_boxes_all_ways : (balls ^ boxes) = 256 := by
  sorry

/-- The number of ways to put 4 different balls into 4 different boxes such that exactly one box remains empty is 144. -/
theorem balls_into_boxes_one_empty : (boxes.choose 1 * (balls ^ (boxes - 1))) = 144 := by
  sorry

end balls_into_boxes_all_ways_balls_into_boxes_one_empty_l115_115276


namespace max_possible_intersections_l115_115585

theorem max_possible_intersections : 
  let num_x := 12
  let num_y := 6
  let intersections := (num_x * (num_x - 1) / 2) * (num_y * (num_y - 1) / 2)
  intersections = 990 := 
by 
  sorry

end max_possible_intersections_l115_115585


namespace probability_of_point_in_smaller_square_l115_115293

-- Definitions
def A_large : ℝ := 5 * 5
def A_small : ℝ := 2 * 2

-- Theorem statement
theorem probability_of_point_in_smaller_square 
  (side_large : ℝ) (side_small : ℝ)
  (hle : side_large = 5) (hse : side_small = 2) :
  (side_large * side_large ≠ 0) ∧ (side_small * side_small ≠ 0) → 
  (A_small / A_large = 4 / 25) :=
sorry

end probability_of_point_in_smaller_square_l115_115293


namespace product_of_triangle_areas_not_end_2014_l115_115438

theorem product_of_triangle_areas_not_end_2014
  (T1 T2 T3 T4 : ℤ)
  (h1 : T1 > 0)
  (h2 : T2 > 0)
  (h3 : T3 > 0)
  (h4 : T4 > 0) :
  (T1 * T2 * T3 * T4) % 10000 ≠ 2014 := by
sorry

end product_of_triangle_areas_not_end_2014_l115_115438


namespace problem_solution_l115_115735

theorem problem_solution (x y z w : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : w > 0) 
  (h5 : x^2 + y^2 + z^2 + w^2 = 1) : 
  x^2 * y * z * w + x * y^2 * z * w + x * y * z^2 * w + x * y * z * w^2 ≤ 1 / 8 := 
by
  sorry

end problem_solution_l115_115735


namespace max_value_l115_115495

-- Definitions for conditions
variables {a b : ℝ}
variables (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (1 / b) = 2)

-- Statement of the theorem
theorem max_value : (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (1 / a) + (1 / b) = 2 ∧ ∀ y : ℝ,
  (1 / y) * ((2 / (y * (3 * y - 1)⁻¹)) + 1) ≤ 25 / 8) :=
sorry

end max_value_l115_115495


namespace range_of_a_l115_115499

def f (a x : ℝ) : ℝ := -x^3 + a * x

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → -3 * x^2 + a ≥ 0) → a ≥ 3 := 
by
  sorry

end range_of_a_l115_115499


namespace isosceles_triangle_area_l115_115947

theorem isosceles_triangle_area (a b c : ℝ) (h: a = 5 ∧ b = 5 ∧ c = 6)
  (altitude_splits_base : ∀ (h : 3^2 + x^2 = 25), x = 4) : 
  ∃ (area : ℝ), area = 12 := 
by
  sorry

end isosceles_triangle_area_l115_115947


namespace Clare_has_more_pencils_than_Jeanine_l115_115327

def Jeanine_initial_pencils : ℕ := 250
def Clare_initial_pencils : ℤ := (-3 : ℤ) * Jeanine_initial_pencils / 5
def Jeanine_pencils_given_Abby : ℕ := (2 : ℕ) * Jeanine_initial_pencils / 7
def Jeanine_pencils_given_Lea : ℕ := (5 : ℕ) * Jeanine_initial_pencils / 11
def Clare_pencils_after_squaring : ℤ := Clare_initial_pencils ^ 2
def Clare_pencils_after_Jeanine_share : ℤ := Clare_pencils_after_squaring + (-1) * Jeanine_initial_pencils / 4

def Jeanine_final_pencils : ℕ := Jeanine_initial_pencils - Jeanine_pencils_given_Abby - Jeanine_pencils_given_Lea

theorem Clare_has_more_pencils_than_Jeanine :
  Clare_pencils_after_Jeanine_share - Jeanine_final_pencils = 22372 :=
sorry

end Clare_has_more_pencils_than_Jeanine_l115_115327


namespace projectile_reaches_35m_first_at_10_over_7_l115_115241

theorem projectile_reaches_35m_first_at_10_over_7 :
  ∃ (t : ℝ), (y : ℝ) = -4.9 * t^2 + 30 * t ∧ y = 35 ∧ t = 10 / 7 :=
by
  sorry

end projectile_reaches_35m_first_at_10_over_7_l115_115241


namespace point_A_coordinates_l115_115872

noncomputable def f (a x : ℝ) : ℝ := a * x - 1

theorem point_A_coordinates (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 1 :=
sorry

end point_A_coordinates_l115_115872


namespace parabola_vertex_coordinates_l115_115330

theorem parabola_vertex_coordinates :
  ∀ (x y : ℝ), y = -3 * (x + 1)^2 - 2 → (x, y) = (-1, -2) := by
  sorry

end parabola_vertex_coordinates_l115_115330


namespace remaining_statue_weight_l115_115076

theorem remaining_statue_weight (w_initial w1 w2 w_discarded w_remaining : ℕ) 
    (h_initial : w_initial = 80)
    (h_w1 : w1 = 10)
    (h_w2 : w2 = 18)
    (h_discarded : w_discarded = 22) :
    2 * w_remaining = w_initial - w_discarded - w1 - w2 :=
by
  sorry

end remaining_statue_weight_l115_115076


namespace cheryl_material_usage_l115_115084

theorem cheryl_material_usage:
  let bought := (3 / 8) + (1 / 3)
  let left := (15 / 40)
  let used := bought - left
  used = (1 / 3) := 
by
  sorry

end cheryl_material_usage_l115_115084


namespace train_cross_pole_time_l115_115628

variable (L : Real) (V : Real)

theorem train_cross_pole_time (hL : L = 110) (hV : V = 144) : 
  (110 / (144 * 1000 / 3600) = 2.75) := 
by
  sorry

end train_cross_pole_time_l115_115628


namespace simplify_eval_expression_l115_115278

variables (a b : ℝ)

theorem simplify_eval_expression :
  a = Real.sqrt 3 →
  b = Real.sqrt 3 - 1 →
  ((3 * a) / (2 * a - b) - 1) / ((a + b) / (4 * a^2 - b^2)) = 3 * Real.sqrt 3 - 1 :=
by
  sorry

end simplify_eval_expression_l115_115278


namespace problem_conditions_l115_115785

noncomputable def f (x : ℝ) := x^2 - 2 * x * Real.log x
noncomputable def g (x : ℝ) := Real.exp x - (Real.exp 2 * x^2) / 4

theorem problem_conditions :
  (∀ x > 0, deriv f x > 0) ∧ 
  (∃! x, g x = 0) ∧ 
  (∃ x, f x = g x) :=
by
  sorry

end problem_conditions_l115_115785


namespace sum_of_solutions_eq_eight_l115_115444

theorem sum_of_solutions_eq_eight : 
  ∀ x : ℝ, (x^2 - 6 * x + 5 = 2 * x - 7) → (∃ a b : ℝ, (a = 6) ∧ (b = 2) ∧ (a + b = 8)) :=
by
  sorry

end sum_of_solutions_eq_eight_l115_115444


namespace max_trees_l115_115265

theorem max_trees (interval distance road_length number_of_intervals add_one : ℕ) 
  (h_interval: interval = 4) 
  (h_distance: distance = 28) 
  (h_intervals: number_of_intervals = distance / interval)
  (h_add: add_one = number_of_intervals + 1) :
  add_one = 8 :=
sorry

end max_trees_l115_115265


namespace tickets_sold_second_half_l115_115457

-- Definitions from conditions
def total_tickets := 9570
def first_half_tickets := 3867

-- Theorem to prove the number of tickets sold in the second half of the season
theorem tickets_sold_second_half : total_tickets - first_half_tickets = 5703 :=
by sorry

end tickets_sold_second_half_l115_115457


namespace merchant_markup_l115_115979

theorem merchant_markup (C : ℝ) (M : ℝ) (h1 : (1 + M / 100 - 0.40 * (1 + M / 100)) * C = 1.05 * C) : 
  M = 75 := sorry

end merchant_markup_l115_115979


namespace least_product_xy_l115_115343

theorem least_product_xy : ∀ (x y : ℕ), 0 < x → 0 < y →
  (1 : ℚ) / x + (1 : ℚ) / (3 * y) = 1 / 6 → x * y = 48 :=
by
  intros x y x_pos y_pos h
  sorry

end least_product_xy_l115_115343


namespace find_y_l115_115381

variable (x y z : ℝ)

theorem find_y
    (h₀ : x + y + z = 150)
    (h₁ : x + 10 = y - 10)
    (h₂ : y - 10 = 3 * z) :
    y = 74.29 :=
by
    sorry

end find_y_l115_115381


namespace price_per_pound_of_rocks_l115_115549

def number_of_rocks : ℕ := 10
def average_weight_per_rock : ℝ := 1.5
def total_amount_made : ℝ := 60

theorem price_per_pound_of_rocks:
  (total_amount_made / (number_of_rocks * average_weight_per_rock)) = 4 := 
by
  sorry

end price_per_pound_of_rocks_l115_115549


namespace fraction_computation_l115_115734

theorem fraction_computation :
  (2 + 4 - 8 + 16 + 32 - 64) / (4 + 8 - 16 + 32 + 64 - 128) = 1 / 2 :=
by
  sorry

end fraction_computation_l115_115734


namespace find_intersection_A_B_find_range_t_l115_115230

-- Define sets A, B, C
def A : Set ℝ := {y | ∃ x, (1 ≤ x ∧ x ≤ 2) ∧ y = 2^x}
def B : Set ℝ := {x | 0 < Real.log x ∧ Real.log x < 1}
def C (t : ℝ) : Set ℝ := {x | t + 1 < x ∧ x < 2 * t}

-- Theorem 1: Finding A ∩ B
theorem find_intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < Real.exp 1} := 
by
  sorry

-- Theorem 2: If A ∩ C = C, find the range of values for t
theorem find_range_t (t : ℝ) (h : A ∩ C t = C t) : t ≤ 2 :=
by
  sorry

end find_intersection_A_B_find_range_t_l115_115230


namespace cost_of_paving_l115_115247

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sqm : ℝ := 1400
def expected_cost : ℝ := 28875

theorem cost_of_paving (l w r : ℝ) (h_l : l = length) (h_w : w = width) (h_r : r = rate_per_sqm) :
  (l * w * r) = expected_cost := by
  sorry

end cost_of_paving_l115_115247


namespace lydia_age_when_planted_l115_115422

-- Definition of the conditions
def years_to_bear_fruit : ℕ := 7
def lydia_age_when_fruit_bears : ℕ := 11

-- Lean 4 statement to prove Lydia's age when she planted the tree
theorem lydia_age_when_planted (a : ℕ) : a = lydia_age_when_fruit_bears - years_to_bear_fruit :=
by
  have : a = 4 := by sorry
  exact this

end lydia_age_when_planted_l115_115422


namespace integer_not_in_range_of_f_l115_115196

noncomputable def f (x : ℝ) : ℤ :=
  if x > -1 then ⌈1 / (x + 1)⌉ else ⌊1 / (x + 1)⌋

theorem integer_not_in_range_of_f :
  ¬ ∃ x : ℝ, x ≠ -1 ∧ f x = 0 :=
by
  sorry

end integer_not_in_range_of_f_l115_115196


namespace star_operation_l115_115405

def star (a b : ℚ) : ℚ := 2 * a - b + 1

theorem star_operation :
  star 1 (star 2 (-3)) = -5 :=
by
  -- Calcualtion follows the steps given in the solution, 
  -- but this line is here just to satisfy the 'rewrite the problem' instruction.
  sorry

end star_operation_l115_115405


namespace shadow_building_length_l115_115089

-- Define the basic parameters
def height_flagpole : ℕ := 18
def shadow_flagpole : ℕ := 45
def height_building : ℕ := 20

-- Define the condition on similar conditions
def similar_conditions (h₁ s₁ h₂ s₂ : ℕ) : Prop :=
  h₁ * s₂ = h₂ * s₁

-- Theorem statement
theorem shadow_building_length :
  similar_conditions height_flagpole shadow_flagpole height_building 50 := 
sorry

end shadow_building_length_l115_115089


namespace pairings_count_l115_115983

-- Define the problem's conditions explicitly
def number_of_bowls : Nat := 6
def number_of_glasses : Nat := 6

-- The theorem stating that the number of pairings is 36
theorem pairings_count : number_of_bowls * number_of_glasses = 36 := by
  sorry

end pairings_count_l115_115983


namespace mul_72518_9999_eq_725107482_l115_115156

theorem mul_72518_9999_eq_725107482 : 72518 * 9999 = 725107482 := by
  sorry

end mul_72518_9999_eq_725107482_l115_115156


namespace tan_expression_l115_115201

theorem tan_expression (a : ℝ) (h₀ : 45 = 2 * a) (h₁ : Real.tan 45 = 1) 
  (h₂ : Real.tan (2 * a) = 2 * Real.tan a / (1 - Real.tan a * Real.tan a)) :
  Real.tan a / (1 - Real.tan a * Real.tan a) = 1 / 2 :=
by 
  sorry

end tan_expression_l115_115201


namespace students_on_field_trip_l115_115740

theorem students_on_field_trip (vans: ℕ) (capacity_per_van: ℕ) (adults: ℕ) 
  (H_vans: vans = 3) 
  (H_capacity_per_van: capacity_per_van = 5) 
  (H_adults: adults = 3) : 
  (vans * capacity_per_van - adults = 12) :=
by
  sorry

end students_on_field_trip_l115_115740


namespace diameter_of_larger_sphere_l115_115035

theorem diameter_of_larger_sphere (r : ℝ) (a b : ℕ) (hr : r = 9)
    (h1 : 3 * (4/3) * π * r^3 = (4/3) * π * ((2 * a * b^(1/3)) / 2)^3) 
    (h2 : ¬∃ c : ℕ, c^3 = b) : a + b = 21 :=
sorry

end diameter_of_larger_sphere_l115_115035


namespace cos_C_value_l115_115188

namespace Triangle

theorem cos_C_value (A B C : ℝ)
  (h_triangle : A + B + C = Real.pi)
  (sin_A : Real.sin A = 2/3)
  (cos_B : Real.cos B = 1/2) :
  Real.cos C = (2 * Real.sqrt 3 - Real.sqrt 5) / 6 := 
sorry

end Triangle

end cos_C_value_l115_115188


namespace calculate_m_squared_l115_115113

-- Define the conditions
def pizza_diameter := 16
def pizza_radius := pizza_diameter / 2
def num_slices := 4

-- Define the question
def longest_segment_length_in_piece := 2 * pizza_radius
def m := longest_segment_length_in_piece -- Length of the longest line segment in one piece

-- Rewrite the math proof problem
theorem calculate_m_squared :
  m^2 = 256 := 
by 
  -- Proof goes here
  sorry

end calculate_m_squared_l115_115113


namespace triangle_angle_C_30_degrees_l115_115319

theorem triangle_angle_C_30_degrees 
  (A B C : ℝ) 
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) 
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) 
  (h3 : A + B + C = 180) 
  : C = 30 :=
  sorry

end triangle_angle_C_30_degrees_l115_115319


namespace eggs_used_afternoon_l115_115004

theorem eggs_used_afternoon (eggs_pumpkin eggs_apple eggs_cherry eggs_total : ℕ)
  (h_pumpkin : eggs_pumpkin = 816)
  (h_apple : eggs_apple = 384)
  (h_cherry : eggs_cherry = 120)
  (h_total : eggs_total = 1820) :
  eggs_total - (eggs_pumpkin + eggs_apple + eggs_cherry) = 500 :=
by
  sorry

end eggs_used_afternoon_l115_115004


namespace first_year_payment_l115_115904

theorem first_year_payment (x : ℝ) 
  (second_year : ℝ := x + 2)
  (third_year : ℝ := x + 5)
  (fourth_year : ℝ := x + 9)
  (total_payment : ℝ := x + second_year + third_year + fourth_year)
  (h : total_payment = 96) : x = 20 := 
by
  sorry

end first_year_payment_l115_115904


namespace probability_odd_sum_is_correct_l115_115786

-- Define the set of the first twelve prime numbers.
def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

-- Define the problem statement.
noncomputable def probability_odd_sum : ℚ :=
  let even_prime_count := 1
  let odd_prime_count := 11
  let ways_to_pick_1_even_and_4_odd := (Nat.choose odd_prime_count 4)
  let total_ways := Nat.choose 12 5
  (ways_to_pick_1_even_and_4_odd : ℚ) / total_ways

theorem probability_odd_sum_is_correct :
  probability_odd_sum = 55 / 132 :=
by
  sorry

end probability_odd_sum_is_correct_l115_115786


namespace smaller_angle_between_clock_hands_3_40_pm_l115_115372

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l115_115372


namespace lisa_needs_additional_marbles_l115_115298

theorem lisa_needs_additional_marbles
  (friends : ℕ) (initial_marbles : ℕ) (total_required_marbles : ℕ) :
  friends = 12 ∧ initial_marbles = 40 ∧ total_required_marbles = (friends * (friends + 1)) / 2 →
  total_required_marbles - initial_marbles = 38 :=
by
  sorry

end lisa_needs_additional_marbles_l115_115298


namespace least_integer_greater_than_sqrt_500_l115_115208

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℤ, n^2 < 500 ∧ (n + 1)^2 > 500 ∧ n = 23 := by
  sorry

end least_integer_greater_than_sqrt_500_l115_115208


namespace function_passes_through_point_l115_115751

noncomputable def special_function (a : ℝ) (x : ℝ) := a^(x - 1) + 1

theorem function_passes_through_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  special_function a 1 = 2 :=
by
  -- skip the proof
  sorry

end function_passes_through_point_l115_115751


namespace find_foci_l115_115341

def hyperbolaFoci : Prop :=
  let eq := ∀ x y, 2 * x^2 - 3 * y^2 + 8 * x - 12 * y - 23 = 0
  ∃ foci : ℝ × ℝ, foci = (-2 - Real.sqrt (5 / 6), -2) ∨ foci = (-2 + Real.sqrt (5 / 6), -2)

theorem find_foci : hyperbolaFoci :=
by
  sorry

end find_foci_l115_115341


namespace min_value_fraction_l115_115703

theorem min_value_fraction (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + 2 * b + 3 * c = 1) : 
  (1/a + 2/b + 3/c) ≥ 36 := 
sorry

end min_value_fraction_l115_115703


namespace negation_of_forall_inequality_l115_115898

theorem negation_of_forall_inequality :
  (¬ (∀ x : ℝ, x > 0 → x * Real.sin x < 2^x - 1)) ↔ (∃ x : ℝ, x > 0 ∧ x * Real.sin x ≥ 2^x - 1) :=
by sorry

end negation_of_forall_inequality_l115_115898


namespace sum_of_inradii_eq_height_l115_115610

variables (a b c h b1 a1 : ℝ)
variables (r r1 r2 : ℝ)

-- Assume CH is the height of the right-angled triangle ABC from the vertex of the right angle.
-- r, r1, r2 are the radii of the incircles of triangles ABC, AHC, and BHC respectively.
-- Given definitions:
-- BC = a
-- AC = b
-- AB = c
-- AH = b1
-- BH = a1
-- CH = h

-- Formulas for the radii of the respective triangles:
-- r : radius of incircle of triangle ABC = (a + b - h) / 2
-- r1 : radius of incircle of triangle AHC = (h + b1 - b) / 2
-- r2 : radius of incircle of triangle BHC = (h + a1 - a) / 2

theorem sum_of_inradii_eq_height 
  (H₁ : r = (a + b - h) / 2)
  (H₂ : r1 = (h + b1 - b) / 2) 
  (H₃ : r2 = (h + a1 - a) / 2) 
  (H₄ : b1 = b - h) 
  (H₅ : a1 = a - h) : 
  r + r1 + r2 = h :=
by
  sorry

end sum_of_inradii_eq_height_l115_115610


namespace proof_cost_A_B_schools_proof_renovation_plans_l115_115817

noncomputable def cost_A_B_schools : Prop :=
  ∃ (x y : ℝ), 2 * x + 3 * y = 78 ∧ 3 * x + y = 54 ∧ x = 12 ∧ y = 18

noncomputable def renovation_plans : Prop :=
  ∃ (a : ℕ), 3 ≤ a ∧ a ≤ 5 ∧ 
    (1200 - 300) * a + (1800 - 500) * (10 - a) ≤ 11800 ∧
    300 * a + 500 * (10 - a) ≥ 4000

theorem proof_cost_A_B_schools : cost_A_B_schools :=
sorry

theorem proof_renovation_plans : renovation_plans :=
sorry

end proof_cost_A_B_schools_proof_renovation_plans_l115_115817


namespace sufficient_not_necessary_condition_l115_115023

theorem sufficient_not_necessary_condition
  (x : ℝ) : 
  x^2 - 4*x - 5 > 0 → (x > 5 ∨ x < -1) ∧ (x > 5 → x^2 - 4*x - 5 > 0) ∧ ¬(x^2 - 4*x - 5 > 0 → x > 5) := 
sorry

end sufficient_not_necessary_condition_l115_115023


namespace sum_of_ages_l115_115607

-- Definitions for Robert's and Maria's current ages
variables (R M : ℕ)

-- Conditions based on the problem statement
theorem sum_of_ages
  (h1 : R = M + 8)
  (h2 : R + 5 = 3 * (M - 3)) :
  R + M = 30 :=
by
  sorry

end sum_of_ages_l115_115607


namespace min_value_of_ellipse_l115_115522

noncomputable def min_m_plus_n (a b : ℝ) (h_ab_nonzero : a * b ≠ 0) (h_abs_diff : |a| ≠ |b|) : ℝ :=
(a ^ (2/3) + b ^ (2/3)) ^ (3/2)

theorem min_value_of_ellipse (m n a b : ℝ) (h1 : m > n) (h2 : n > 0) (h_ellipse : (a^2 / m^2) + (b^2 / n^2) = 1) (h_ab_nonzero : a * b ≠ 0) (h_abs_diff : |a| ≠ |b|) :
  (m + n) = min_m_plus_n a b h_ab_nonzero h_abs_diff :=
sorry

end min_value_of_ellipse_l115_115522


namespace arithmetic_sequence_count_l115_115437

theorem arithmetic_sequence_count :
  ∃! (n a d : ℕ), n ≥ 3 ∧ (n * (2 * a + (n - 1) * d) = 2 * 97^2) :=
sorry

end arithmetic_sequence_count_l115_115437


namespace adam_clothing_ratio_l115_115864

-- Define the initial amount of clothing Adam took out
def initial_clothing_adam : ℕ := 4 + 4 + 8 + 20

-- Define the number of friends donating the same amount of clothing as Adam
def number_of_friends : ℕ := 3

-- Define the total number of clothes being donated
def total_donated_clothes : ℕ := 126

-- Define the ratio of the clothes Adam is keeping to the clothes he initially took out
def ratio_kept_to_initial (initial_clothing: ℕ) (total_donated: ℕ) (kept: ℕ) : Prop :=
  kept * initial_clothing = 0

-- Theorem statement
theorem adam_clothing_ratio :
  ratio_kept_to_initial initial_clothing_adam total_donated_clothes 0 :=
by 
  sorry

end adam_clothing_ratio_l115_115864


namespace log_base_10_of_2_bounds_l115_115514

theorem log_base_10_of_2_bounds :
  (10^3 = 1000) ∧ (10^4 = 10000) ∧ (2^11 = 2048) ∧ (2^14 = 16384) →
  (3 / 11 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (2 / 7 : ℝ) :=
by
  sorry

end log_base_10_of_2_bounds_l115_115514


namespace sum_abc_l115_115807

noncomputable def polynomial : Polynomial ℝ :=
  Polynomial.C (-6) + Polynomial.X * (Polynomial.C 11 + Polynomial.X * (Polynomial.C (-6) + Polynomial.X))

def t (k : ℕ) : ℝ :=
  match k with
  | 0 => 3
  | 1 => 6
  | 2 => 14
  | _ => 0 -- placeholder, as only t_0, t_1, t_2 are given explicitly

def a := 6
def b := -11
def c := 18

def t_rec (k : ℕ) : ℝ :=
  match k with
  | 0 => 3
  | 1 => 6
  | 2 => 14
  | n + 3 => a * t (n + 2) + b * t (n + 1) + c * t n

theorem sum_abc : a + b + c = 13 := by
  sorry

end sum_abc_l115_115807


namespace Paco_cookies_left_l115_115217

/-
Problem: Paco had 36 cookies. He gave 14 cookies to his friend and ate 10 cookies. How many cookies did Paco have left?
Solution: Paco has 12 cookies left.

To formally state this in Lean:
-/

def initial_cookies := 36
def cookies_given_away := 14
def cookies_eaten := 10

theorem Paco_cookies_left : initial_cookies - (cookies_given_away + cookies_eaten) = 12 :=
by
  sorry

/-
This theorem states that Paco has 12 cookies left given initial conditions.
-/

end Paco_cookies_left_l115_115217


namespace largest_n_unique_k_l115_115390

theorem largest_n_unique_k :
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, (3 : ℚ) / 7 < (n : ℚ) / ((n + k : ℕ) : ℚ) ∧ 
  (n : ℚ) / ((n + k : ℕ) : ℚ) < (8 : ℚ) / 19 → k = 1 := by
sorry

end largest_n_unique_k_l115_115390


namespace smallest_model_length_l115_115506

theorem smallest_model_length 
  (full_size_length : ℕ)
  (mid_size_ratio : ℚ)
  (smallest_size_ratio : ℚ)
  (H1 : full_size_length = 240)
  (H2 : mid_size_ratio = 1/10)
  (H3 : smallest_size_ratio = 1/2) 
  : full_size_length * mid_size_ratio * smallest_size_ratio = 12 :=
by
  sorry

end smallest_model_length_l115_115506


namespace A_is_sufficient_but_not_necessary_for_D_l115_115488

variable {A B C D : Prop}

-- Defining the conditions
axiom h1 : A → B
axiom h2 : B ↔ C
axiom h3 : C → D

-- Statement to be proven
theorem A_is_sufficient_but_not_necessary_for_D : (A → D) ∧ ¬(D → A) :=
  by
  sorry

end A_is_sufficient_but_not_necessary_for_D_l115_115488


namespace value_of_expression_l115_115617

theorem value_of_expression (x : ℤ) (h : x = -2) : (3 * x - 4)^2 = 100 :=
by
  -- Given the hypothesis h: x = -2
  -- Need to show: (3 * x - 4)^2 = 100
  sorry

end value_of_expression_l115_115617


namespace relationship_between_a_and_b_l115_115868

def ellipse_touching_hyperbola (a b : ℝ) :=
  ∀ x y : ℝ, ( (x / a) ^ 2 + (y / b) ^ 2 = 1 ∧ y = 1 / x → False )

  theorem relationship_between_a_and_b (a b : ℝ) :
  ellipse_touching_hyperbola a b →
  a * b = 2 :=
by
  sorry

end relationship_between_a_and_b_l115_115868


namespace largest_possible_A_l115_115792

theorem largest_possible_A : ∃ A B : ℕ, 13 = 4 * A + B ∧ B < A ∧ A = 3 := by
  sorry

end largest_possible_A_l115_115792


namespace pascal_28_25_eq_2925_l115_115835

-- Define the Pascal's triangle nth-row function
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the theorem to prove that the 25th element in the 28 element row is 2925
theorem pascal_28_25_eq_2925 :
  pascal 27 24 = 2925 :=
by
  sorry

end pascal_28_25_eq_2925_l115_115835


namespace average_speed_sf_l115_115708

variables
  (v d t : ℝ)  -- Representing the average speed to SF, the distance, and time to SF
  (h1 : 42 = (2 * d) / (3 * t))  -- Condition: Average speed of the round trip is 42 mph
  (h2 : t = d / v)  -- Definition of time t in terms of distance and speed

theorem average_speed_sf : v = 63 :=
by
  sorry

end average_speed_sf_l115_115708


namespace find_y_l115_115150

open Real

def vecV (y : ℝ) : ℝ × ℝ := (1, y)
def vecW : ℝ × ℝ := (6, 4)

noncomputable def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (dotProduct v w) / (dotProduct w w)
  (scalar * w.1, scalar * w.2)

theorem find_y (y : ℝ) (h : projection (vecV y) vecW = (3, 2)) : y = 5 := by
  sorry

end find_y_l115_115150


namespace degree_f_x2_g_x3_l115_115166

open Polynomial

noncomputable def degree_of_composite_polynomials (f g : Polynomial ℝ) : ℕ :=
  let f_degree := Polynomial.degree f
  let g_degree := Polynomial.degree g
  match (f_degree, g_degree) with
  | (some 3, some 6) => 24
  | _ => 0

theorem degree_f_x2_g_x3 (f g : Polynomial ℝ) (h_f : Polynomial.degree f = 3) (h_g : Polynomial.degree g = 6) :
  Polynomial.degree (Polynomial.comp f (X^2) * Polynomial.comp g (X^3)) = 24 := by
  -- content Logic Here
  sorry

end degree_f_x2_g_x3_l115_115166


namespace range_of_a_l115_115755

theorem range_of_a (a : ℝ) :
  (a + 1 > 0 ∧ 3 - 2 * a > 0 ∧ a + 1 > 3 - 2 * a) ∨ (a + 1 < 0 ∧ 3 - 2 * a < 0 ∧ a + 1 > 3 - 2 * a) ∨ (a + 1 < 0 ∧ 3 - 2 * a > 0)
  → (2 / 3 < a ∧ a < 3 / 2) ∨ (a < -1) :=
by
  sorry

end range_of_a_l115_115755


namespace cookie_sales_l115_115866

theorem cookie_sales (n : ℕ) (h1 : 1 ≤ n - 11) (h2 : 1 ≤ n - 2) (h3 : (n - 11) + (n - 2) < n) : n = 12 :=
sorry

end cookie_sales_l115_115866


namespace equilateral_triangle_l115_115777

theorem equilateral_triangle (a b c : ℝ) (h1 : a + b - c = 2) (h2 : 2 * a * b - c^2 = 4) : a = b ∧ b = c ∧ a = c := 
by
  sorry

end equilateral_triangle_l115_115777


namespace find_number_of_appliances_l115_115328

-- Declare the constants related to the problem.
def commission_per_appliance : ℝ := 50
def commission_percent : ℝ := 0.1
def total_selling_price : ℝ := 3620
def total_commission : ℝ := 662

-- Define the theorem to solve for the number of appliances sold.
theorem find_number_of_appliances (n : ℝ) 
  (H : n * commission_per_appliance + commission_percent * total_selling_price = total_commission) : 
  n = 6 := 
sorry

end find_number_of_appliances_l115_115328


namespace eccentricity_of_ellipse_l115_115017

theorem eccentricity_of_ellipse {a b c e : ℝ} 
  (h1 : b^2 = 3) 
  (h2 : c = 1 / 4)
  (h3 : a^2 = b^2 + c^2)
  (h4 : a = 7 / 4) 
  : e = c / a → e = 1 / 7 :=
by 
  intros
  sorry

end eccentricity_of_ellipse_l115_115017


namespace find_m_l115_115426

theorem find_m :
  ∃ m : ℕ, 264 * 391 % 100 = m ∧ 0 ≤ m ∧ m < 100 ∧ m = 24 :=
by
  sorry

end find_m_l115_115426


namespace factorize_expression_l115_115229

-- Define the variables m and n
variables (m n : ℝ)

-- The statement to prove
theorem factorize_expression : -8 * m^2 + 2 * m * n = -2 * m * (4 * m - n) :=
sorry

end factorize_expression_l115_115229


namespace geometric_sequence_general_term_formula_no_arithmetic_sequence_l115_115814

-- Assume we have a sequence {a_n} and its sum of the first n terms S_n where S_n = 2a_n - n (for n ∈ ℕ*)
variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}

-- Condition 1: S_n = 2a_n - n
axiom Sn_condition (n : ℕ) (h : n > 0) : S_n n = 2 * a_n n - n

-- 1. Prove that the sequence {a_n + 1} is a geometric sequence with first term and common ratio equal to 2
theorem geometric_sequence (n : ℕ) (h : n > 0) : ∃ r : ℕ, r = 2 ∧ ∀ m : ℕ, a_n (m + 1) + 1 = r * (a_n m + 1) :=
by
  sorry

-- 2. Prove the general term formula an = 2^n - 1
theorem general_term_formula (n : ℕ) (h : n > 0) : a_n n = 2^n - 1 :=
by
  sorry

-- 3. Prove that there do not exist three consecutive terms in {a_n} that form an arithmetic sequence
theorem no_arithmetic_sequence (n k : ℕ) (h : n > 0 ∧ k > 0 ∧ k + 2 < n) : ¬(a_n k + a_n (k + 2) = 2 * a_n (k + 1)) :=
by
  sorry

end geometric_sequence_general_term_formula_no_arithmetic_sequence_l115_115814


namespace symmetric_points_add_l115_115513

theorem symmetric_points_add (a b : ℝ) : 
  (P : ℝ × ℝ) → (Q : ℝ × ℝ) →
  P = (a-1, 5) →
  Q = (2, b-1) →
  (P.fst = Q.fst) →
  P.snd = -Q.snd →
  a + b = -1 :=
by
  sorry

end symmetric_points_add_l115_115513


namespace hannah_practice_hours_l115_115803

theorem hannah_practice_hours (weekend_hours : ℕ) (total_weekly_hours : ℕ) (more_weekday_hours : ℕ)
  (h1 : weekend_hours = 8)
  (h2 : total_weekly_hours = 33)
  (h3 : more_weekday_hours = 17) :
  (total_weekly_hours - weekend_hours) - weekend_hours = more_weekday_hours :=
by
  sorry

end hannah_practice_hours_l115_115803


namespace number_of_green_pens_l115_115257

theorem number_of_green_pens
  (black_pens : ℕ := 6)
  (red_pens : ℕ := 7)
  (green_pens : ℕ)
  (probability_black : (black_pens : ℚ) / (black_pens + red_pens + green_pens : ℚ) = 1 / 3) :
  green_pens = 5 := 
sorry

end number_of_green_pens_l115_115257


namespace simplify_tan_cot_expr_l115_115752

theorem simplify_tan_cot_expr :
  let tan_45 := 1
  let cot_45 := 1
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 :=
by
  let tan_45 := 1
  let cot_45 := 1
  sorry

end simplify_tan_cot_expr_l115_115752


namespace quadrilateral_side_squares_inequality_l115_115121

theorem quadrilateral_side_squares_inequality :
  ∀ (x1 y1 x2 y2 : ℝ),
    0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ y1 ∧ y1 ≤ 1 ∧
    0 ≤ x2 ∧ x2 ≤ 1 ∧ 0 ≤ y2 ∧ y2 ≤ 1 →
    2 ≤ (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ∧ 
          (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ≤ 4 :=
by
  intro x1 y1 x2 y2 h
  sorry

end quadrilateral_side_squares_inequality_l115_115121


namespace reggie_games_lost_l115_115923

-- Define the necessary conditions
def initial_marbles : ℕ := 100
def bet_per_game : ℕ := 10
def marbles_after_games : ℕ := 90
def total_games : ℕ := 9

-- Define the proof problem statement
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / bet_per_game = 1 := by
  sorry

end reggie_games_lost_l115_115923


namespace ratio_of_15th_term_l115_115510

theorem ratio_of_15th_term (a d b e : ℤ) :
  (∀ n : ℕ, (n * (2 * a + (n - 1) * d)) / (n * (2 * b + (n - 1) * e)) = (7 * n^2 + 1) / (4 * n^2 + 27)) →
  (a + 14 * d) / (b + 14 * e) = 7 / 4 :=
by sorry

end ratio_of_15th_term_l115_115510


namespace seq_b_is_geometric_l115_115413

variable {α : Type*} [LinearOrderedField α]

-- Define the geometric sequence {a_n} with first term a_1 and common ratio q
def a_n (a₁ q : α) (n : ℕ) : α := a₁ * q^(n-1)

-- Define the sequence {b_n}
def b_n (a₁ q : α) (n : ℕ) : α :=
  a_n a₁ q (3*n - 2) + a_n a₁ q (3*n - 1) + a_n a₁ q (3*n)

-- Theorem stating {b_n} is a geometric sequence with common ratio q^3
theorem seq_b_is_geometric (a₁ q : α) (h : q ≠ 1) :
  ∀ n : ℕ, b_n a₁ q (n + 1) = q^3 * b_n a₁ q n :=
by
  sorry

end seq_b_is_geometric_l115_115413


namespace is_hexagonal_number_2016_l115_115159

theorem is_hexagonal_number_2016 :
  ∃ (n : ℕ), 2 * n^2 - n = 2016 :=
sorry

end is_hexagonal_number_2016_l115_115159


namespace geometric_series_cubes_sum_l115_115564

theorem geometric_series_cubes_sum (b s : ℝ) (h : -1 < s ∧ s < 1) :
  ∑' n : ℕ, (b * s^n)^3 = b^3 / (1 - s^3) := 
sorry

end geometric_series_cubes_sum_l115_115564


namespace value_of_f_neg1_l115_115948

def f (x : ℤ) : ℤ := x^2 - 2 * x

theorem value_of_f_neg1 : f (-1) = 3 := by
  sorry

end value_of_f_neg1_l115_115948


namespace alpha_sin_beta_lt_beta_sin_alpha_l115_115204

variable {α β : ℝ}

theorem alpha_sin_beta_lt_beta_sin_alpha (h1 : 0 < α) (h2 : α < β) (h3 : β < Real.pi / 2) : 
  α * Real.sin β < β * Real.sin α := 
by
  sorry

end alpha_sin_beta_lt_beta_sin_alpha_l115_115204


namespace four_consecutive_integers_product_plus_one_is_square_l115_115002

theorem four_consecutive_integers_product_plus_one_is_square (n : ℤ) :
  (n - 1) * n * (n + 1) * (n + 2) + 1 = (n^2 + n - 1)^2 := by
  sorry

end four_consecutive_integers_product_plus_one_is_square_l115_115002


namespace lines_intersect_sum_c_d_l115_115108

theorem lines_intersect_sum_c_d (c d : ℝ) 
    (h1 : ∃ x y : ℝ, x = (1/3) * y + c ∧ y = (1/3) * x + d) 
    (h2 : ∀ x y : ℝ, x = 3 ∧ y = 3) : 
    c + d = 4 :=
by sorry

end lines_intersect_sum_c_d_l115_115108


namespace ursula_purchases_total_cost_l115_115576

variable (T C B Br : ℝ)
variable (hT : T = 10) (hTC : T = 2 * C) (hB : B = 0.8 * C) (hBr : Br = B / 2)

theorem ursula_purchases_total_cost : T + C + B + Br = 21 := by
  sorry

end ursula_purchases_total_cost_l115_115576


namespace evaluate_g_at_neg2_l115_115581

def g (x : ℝ) : ℝ := x^3 - 3 * x^2 + 4

theorem evaluate_g_at_neg2 : g (-2) = -16 := by
  sorry

end evaluate_g_at_neg2_l115_115581


namespace scout_troop_profit_l115_115195

theorem scout_troop_profit :
  let bars_bought := 1200
  let cost_per_bar := 1 / 3
  let bars_per_dollar := 3
  let total_cost := bars_bought * cost_per_bar
  let selling_price_per_bar := 3 / 5
  let bars_per_three_dollars := 5
  let total_revenue := bars_bought * selling_price_per_bar
  let profit := total_revenue - total_cost
  profit = 320 := by
  let bars_bought := 1200
  let cost_per_bar := 1 / 3
  let total_cost := bars_bought * cost_per_bar
  let selling_price_per_bar := 3 / 5
  let total_revenue := bars_bought * selling_price_per_bar
  let profit := total_revenue - total_cost
  sorry

end scout_troop_profit_l115_115195


namespace quadratic_real_roots_l115_115660

theorem quadratic_real_roots (K : ℝ) :
  ∃ x : ℝ, K^2 * x^2 + (K^2 - 1) * x - 2 * K^2 = 0 :=
sorry

end quadratic_real_roots_l115_115660


namespace total_pens_l115_115324

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l115_115324


namespace intersection_A_B_l115_115846

def A := {x : ℝ | 2 * x - 1 ≤ 0}
def B := {x : ℝ | 1 / x > 1}

theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1 / 2} :=
  sorry

end intersection_A_B_l115_115846


namespace find_original_percentage_of_acid_l115_115447

noncomputable def percentage_of_acid (a w : ℕ) : ℚ :=
  (a : ℚ) / (a + w : ℚ) * 100

theorem find_original_percentage_of_acid (a w : ℕ) 
  (h1 : (a : ℚ) / (a + w + 2 : ℚ) = 1 / 4)
  (h2 : (a + 2 : ℚ) / (a + w + 4 : ℚ) = 2 / 5) : 
  percentage_of_acid a w = 33.33 :=
by 
  sorry

end find_original_percentage_of_acid_l115_115447


namespace part_a_impossibility_l115_115015

-- Define the number of rows and columns
def num_rows : ℕ := 20
def num_columns : ℕ := 15

-- Define a function that checks if the sum of the counts in rows and columns match the conditions
def is_possible_configuration : Prop :=
  (num_rows % 2 = 0) ∧ (num_columns % 2 = 1)

theorem part_a_impossibility : ¬ is_possible_configuration :=
by
  -- The proof for the contradiction will go here
  sorry

end part_a_impossibility_l115_115015


namespace unfolded_paper_has_four_symmetrical_holes_l115_115487

structure Paper :=
  (width : ℤ) (height : ℤ) (hole_x : ℤ) (hole_y : ℤ)

structure Fold :=
  (direction : String) (fold_line : ℤ)

structure UnfoldedPaper :=
  (holes : List (ℤ × ℤ))

-- Define the initial paper, folds, and punching
def initial_paper : Paper := {width := 4, height := 6, hole_x := 2, hole_y := 1}
def folds : List Fold := 
  [{direction := "bottom_to_top", fold_line := initial_paper.height / 2}, 
   {direction := "left_to_right", fold_line := initial_paper.width / 2}]
def punch : (ℤ × ℤ) := (initial_paper.hole_x, initial_paper.hole_y)

-- The theorem to prove the resulting unfolded paper
theorem unfolded_paper_has_four_symmetrical_holes (p : Paper) (fs : List Fold) (punch : ℤ × ℤ) :
  UnfoldedPaper :=
  { holes := [(1, 1), (1, 5), (3, 1), (3, 5)] } -- Four symmetrically placed holes.

end unfolded_paper_has_four_symmetrical_holes_l115_115487


namespace pairs_satisfying_int_l115_115030

theorem pairs_satisfying_int (a b : ℕ) :
  ∃ n : ℕ, a = 2 * n^2 + 1 ∧ b = n ↔ (2 * a * b^2 + 1) ∣ (a^3 + 1) := by
  sorry

end pairs_satisfying_int_l115_115030


namespace ratio_and_equation_imp_value_of_a_l115_115816

theorem ratio_and_equation_imp_value_of_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 20 - 7 * a) :
  a = 20 / 11 :=
by
  sorry

end ratio_and_equation_imp_value_of_a_l115_115816


namespace pow_calculation_l115_115028

-- We assume a is a non-zero real number or just a variable
variable (a : ℝ)

theorem pow_calculation : (2 * a^2)^3 = 8 * a^6 := 
by
  sorry

end pow_calculation_l115_115028


namespace num_congruent_2_mod_11_l115_115891

theorem num_congruent_2_mod_11 : 
  ∃ (n : ℕ), n = 28 ∧ ∀ k : ℤ, 1 ≤ 11 * k + 2 ∧ 11 * k + 2 ≤ 300 ↔ 0 ≤ k ∧ k ≤ 27 :=
sorry

end num_congruent_2_mod_11_l115_115891


namespace nth_equation_l115_115451

-- Define the product of a list of integers
def prod_list (lst : List ℕ) : ℕ :=
  lst.foldl (· * ·) 1

-- Define the product of first n odd numbers
def prod_odds (n : ℕ) : ℕ :=
  prod_list (List.map (λ i => 2 * i - 1) (List.range n))

-- Define the product of the range from n+1 to 2n
def prod_range (n : ℕ) : ℕ :=
  prod_list (List.range' (n + 1) n)

-- The theorem to prove
theorem nth_equation (n : ℕ) (hn : 0 < n) : prod_range n = 2^n * prod_odds n := 
  sorry

end nth_equation_l115_115451


namespace number_of_floors_l115_115236

-- Definitions
def height_regular_floor : ℝ := 3
def height_last_floor : ℝ := 3.5
def total_height : ℝ := 61

-- Theorem statement
theorem number_of_floors (n : ℕ) : 
  (n ≥ 2) →
  (2 * height_last_floor + (n - 2) * height_regular_floor = total_height) →
  n = 20 :=
sorry

end number_of_floors_l115_115236


namespace average_of_25_results_l115_115698

theorem average_of_25_results (first12_avg : ℕ -> ℕ -> ℕ)
                             (last12_avg : ℕ -> ℕ -> ℕ) 
                             (res13 : ℕ)
                             (avg_of_25 : ℕ) :
                             first12_avg 12 10 = 120
                             ∧ last12_avg 12 20 = 240
                             ∧ res13 = 90
                             ∧ avg_of_25 = (first12_avg 12 10 + last12_avg 12 20 + res13) / 25
                             → avg_of_25 = 18 := by
  sorry

end average_of_25_results_l115_115698


namespace moles_of_Cu_CN_2_is_1_l115_115010

def moles_of_HCN : Nat := 2
def moles_of_CuSO4 : Nat := 1
def moles_of_Cu_CN_2_formed (hcn : Nat) (cuso4 : Nat) : Nat :=
  if hcn = 2 ∧ cuso4 = 1 then 1 else 0

theorem moles_of_Cu_CN_2_is_1 : moles_of_Cu_CN_2_formed moles_of_HCN moles_of_CuSO4 = 1 :=
by
  sorry

end moles_of_Cu_CN_2_is_1_l115_115010


namespace smallest_value_among_options_l115_115280

theorem smallest_value_among_options (x : ℕ) (h : x = 9) :
    min (8/x) (min (8/(x+2)) (min (8/(x-2)) (min ((x+3)/8) ((x-3)/8)))) = (3/4) :=
by
  sorry

end smallest_value_among_options_l115_115280


namespace smallest_number_of_students_l115_115029

theorem smallest_number_of_students 
    (n : ℕ) 
    (attended := n / 4)
    (both := n / 40)
    (cheating_hint_ratio : ℚ := 3 / 2)
    (hinting := cheating_hint_ratio * (attended - both)) :
    n ≥ 200 :=
by sorry

end smallest_number_of_students_l115_115029


namespace factorization_of_x4_plus_16_l115_115806

theorem factorization_of_x4_plus_16 :
  (x : ℝ) → x^4 + 16 = (x^2 + 2 * x + 2) * (x^2 - 2 * x + 2) :=
by
  intro x
  -- Placeholder for the proof
  sorry

end factorization_of_x4_plus_16_l115_115806


namespace oliver_shelves_needed_l115_115034

-- Definitions based on conditions
def total_books : ℕ := 46
def books_taken_by_librarian : ℕ := 10
def books_remaining (total books_taken : ℕ) : ℕ := total - books_taken
def books_per_shelf : ℕ := 4

-- Theorem statement
theorem oliver_shelves_needed :
  books_remaining total_books books_taken_by_librarian / books_per_shelf = 9 := by
  sorry

end oliver_shelves_needed_l115_115034


namespace average_marks_of_failed_boys_l115_115635

def total_boys : ℕ := 120
def average_marks_all_boys : ℝ := 35
def number_of_passed_boys : ℕ := 100
def average_marks_passed_boys : ℝ := 39
def number_of_failed_boys : ℕ := total_boys - number_of_passed_boys

noncomputable def total_marks_all_boys : ℝ := average_marks_all_boys * total_boys
noncomputable def total_marks_passed_boys : ℝ := average_marks_passed_boys * number_of_passed_boys
noncomputable def total_marks_failed_boys : ℝ := total_marks_all_boys - total_marks_passed_boys
noncomputable def average_marks_failed_boys : ℝ := total_marks_failed_boys / number_of_failed_boys

theorem average_marks_of_failed_boys :
  average_marks_failed_boys = 15 :=
by
  -- The proof can be filled in here
  sorry

end average_marks_of_failed_boys_l115_115635


namespace calculate_expression_l115_115234

theorem calculate_expression : 15 * 30 + 45 * 15 + 90 = 1215 := 
by 
  sorry

end calculate_expression_l115_115234


namespace middle_number_of_consecutive_numbers_sum_of_squares_eq_2030_l115_115874

theorem middle_number_of_consecutive_numbers_sum_of_squares_eq_2030 :
  ∃ n : ℕ, n^2 + (n+1)^2 + (n+2)^2 = 2030 ∧ (n + 1) = 26 :=
by sorry

end middle_number_of_consecutive_numbers_sum_of_squares_eq_2030_l115_115874


namespace cos_value_l115_115272

theorem cos_value (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) : Real.cos (2 * π / 3 - α) = 1 / 3 :=
by
  sorry

end cos_value_l115_115272


namespace base_of_parallelogram_l115_115940

variable (Area Height Base : ℝ)

def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem base_of_parallelogram
  (h_area : Area = 200)
  (h_height : Height = 20)
  (h_area_def : parallelogram_area Base Height = Area) :
  Base = 10 :=
by sorry

end base_of_parallelogram_l115_115940


namespace total_tickets_sold_l115_115020

theorem total_tickets_sold
  (advanced_ticket_cost : ℕ)
  (door_ticket_cost : ℕ)
  (total_collected : ℕ)
  (advanced_tickets_sold : ℕ)
  (door_tickets_sold : ℕ) :
  advanced_ticket_cost = 8 →
  door_ticket_cost = 14 →
  total_collected = 1720 →
  advanced_tickets_sold = 100 →
  total_collected = (advanced_tickets_sold * advanced_ticket_cost) + (door_tickets_sold * door_ticket_cost) →
  100 + door_tickets_sold = 165 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_tickets_sold_l115_115020


namespace sum_of_consecutive_integers_l115_115716

theorem sum_of_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 7) : a + b + c = 18 := by
  sorry

end sum_of_consecutive_integers_l115_115716


namespace fixed_monthly_fee_l115_115169

theorem fixed_monthly_fee :
  ∀ (x y : ℝ), 
  x + y = 20.00 → 
  x + 2 * y = 30.00 → 
  x + 3 * y = 40.00 → 
  x = 10.00 :=
by
  intros x y H1 H2 H3
  -- Proof can be filled out here
  sorry

end fixed_monthly_fee_l115_115169


namespace gcd_sequence_property_l115_115994

theorem gcd_sequence_property (a : ℕ → ℕ) (m n : ℕ) (h : ∀ m n, m > n → Nat.gcd (a m) (a n) = Nat.gcd (a (m - n)) (a n)) : 
  Nat.gcd (a m) (a n) = a (Nat.gcd m n) :=
by
  sorry

end gcd_sequence_property_l115_115994


namespace linear_eq_conditions_l115_115696

theorem linear_eq_conditions (m : ℤ) (h : abs m = 1) (h₂ : m + 1 ≠ 0) : m = 1 :=
by
  sorry

end linear_eq_conditions_l115_115696


namespace b_share_of_payment_l115_115401

def work_fraction (d : ℕ) : ℚ := 1 / d

def total_one_day_work (a_days b_days c_days : ℕ) : ℚ :=
  work_fraction a_days + work_fraction b_days + work_fraction c_days

def share_of_work (b_days : ℕ) (total_work : ℚ) : ℚ :=
  work_fraction b_days / total_work

def share_of_payment (total_payment : ℚ) (work_share : ℚ) : ℚ :=
  total_payment * work_share

theorem b_share_of_payment 
  (a_days b_days c_days : ℕ) (total_payment : ℚ):
  a_days = 6 → b_days = 8 → c_days = 12 → total_payment = 1800 →
  share_of_payment total_payment (share_of_work b_days (total_one_day_work a_days b_days c_days)) = 600 :=
by
  intros ha hb hc hp
  unfold total_one_day_work work_fraction share_of_work share_of_payment
  rw [ha, hb, hc, hp]
  -- Simplify the fractions and the multiplication
  sorry

end b_share_of_payment_l115_115401


namespace expression_meaningful_range_l115_115237

theorem expression_meaningful_range (a : ℝ) : (∃ x, x = (a + 3) ^ (1/2) / (a - 1)) ↔ (a ≥ -3 ∧ a ≠ 1) :=
by
  sorry

end expression_meaningful_range_l115_115237


namespace expression_evaluation_l115_115738

theorem expression_evaluation : 4 * (9 - 6) / 2 - 3 = 3 := 
by
  sorry

end expression_evaluation_l115_115738


namespace algebraic_expression_domain_l115_115103

theorem algebraic_expression_domain (x : ℝ) : 
  (x + 2 ≥ 0) ∧ (x - 3 ≠ 0) ↔ (x ≥ -2) ∧ (x ≠ 3) := by
  sorry

end algebraic_expression_domain_l115_115103


namespace num_possible_values_a_l115_115537

theorem num_possible_values_a (a : ℕ) :
  (9 ∣ a) ∧ (a ∣ 18) ∧ (0 < a) → ∃ n : ℕ, n = 2 :=
by
  sorry

end num_possible_values_a_l115_115537


namespace cos_sum_to_9_l115_115051

open Real

theorem cos_sum_to_9 {x y z : ℝ} (h1 : cos x + cos y + cos z = 3) (h2 : sin x + sin y + sin z = 0) :
  cos (2 * x) + cos (2 * y) + cos (2 * z) = 9 := 
sorry

end cos_sum_to_9_l115_115051


namespace deepak_current_age_l115_115618

variable (A D : ℕ)

def ratio_condition : Prop := A * 5 = D * 2
def arun_future_age (A : ℕ) : Prop := A + 10 = 30

theorem deepak_current_age (h1 : ratio_condition A D) (h2 : arun_future_age A) : D = 50 := sorry

end deepak_current_age_l115_115618


namespace sum_of_areas_is_72_l115_115221

def base : ℕ := 2
def length1 : ℕ := 1
def length2 : ℕ := 8
def length3 : ℕ := 27

theorem sum_of_areas_is_72 : base * length1 + base * length2 + base * length3 = 72 :=
by
  sorry

end sum_of_areas_is_72_l115_115221


namespace log_product_computation_l115_115301

theorem log_product_computation : 
  (Real.log 32 / Real.log 2) * (Real.log 27 / Real.log 3) = 15 := 
by
  -- The proof content, which will be skipped with 'sorry'.
  sorry

end log_product_computation_l115_115301


namespace james_hours_per_year_l115_115402

def hours_per_day (trainings_per_day : Nat) (hours_per_training : Nat) : Nat :=
  trainings_per_day * hours_per_training

def days_per_week (total_days : Nat) (rest_days : Nat) : Nat :=
  total_days - rest_days

def hours_per_week (hours_day : Nat) (days_week : Nat) : Nat :=
  hours_day * days_week

def hours_per_year (hours_week : Nat) (weeks_year : Nat) : Nat :=
  hours_week * weeks_year

theorem james_hours_per_year :
  let trainings_per_day := 2
  let hours_per_training := 4
  let total_days_per_week := 7
  let rest_days_per_week := 2
  let weeks_per_year := 52
  hours_per_year 
    (hours_per_week 
      (hours_per_day trainings_per_day hours_per_training) 
      (days_per_week total_days_per_week rest_days_per_week)
    ) weeks_per_year
  = 2080 := by
  sorry

end james_hours_per_year_l115_115402


namespace shop_length_l115_115167

def monthly_rent : ℝ := 2244
def width : ℝ := 18
def annual_rent_per_sqft : ℝ := 68

theorem shop_length : 
  (monthly_rent * 12 / annual_rent_per_sqft / width) = 22 := 
by
  -- Proof omitted
  sorry

end shop_length_l115_115167


namespace girls_left_class_l115_115096

variable (G B G₂ B₁ : Nat)

theorem girls_left_class (h₁ : 5 * B = 6 * G) 
                         (h₂ : B = 120)
                         (h₃ : 2 * B₁ = 3 * G₂)
                         (h₄ : B₁ = B) : 
                         G - G₂ = 20 :=
by
  sorry

end girls_left_class_l115_115096


namespace intersection_of_A_and_B_l115_115043

def A : Set ℤ := {-1, 0, 3, 5}
def B : Set ℤ := {x | x - 2 > 0}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := 
by 
  sorry

end intersection_of_A_and_B_l115_115043


namespace chess_tournament_participants_l115_115334

/-- If each participant of a chess tournament plays exactly one game with each of the remaining participants, and 231 games are played during the tournament, then the number of participants is 22. -/
theorem chess_tournament_participants (n : ℕ) (h : (n - 1) * n / 2 = 231) : n = 22 :=
sorry

end chess_tournament_participants_l115_115334


namespace stack_map_A_front_view_l115_115369

def column1 : List ℕ := [3, 1]
def column2 : List ℕ := [2, 2, 1]
def column3 : List ℕ := [1, 4, 2]
def column4 : List ℕ := [5]

def tallest (l : List ℕ) : ℕ :=
  l.foldl max 0

theorem stack_map_A_front_view :
  [tallest column1, tallest column2, tallest column3, tallest column4] = [3, 2, 4, 5] := by
  sorry

end stack_map_A_front_view_l115_115369


namespace new_cost_relation_l115_115597

def original_cost (k t b : ℝ) : ℝ :=
  k * (t * b)^4

def new_cost (k t b : ℝ) : ℝ :=
  k * ((2 * b) * (0.75 * t))^4

theorem new_cost_relation (k t b : ℝ) (C : ℝ) 
  (hC : C = original_cost k t b) :
  new_cost k t b = 25.63 * C := sorry

end new_cost_relation_l115_115597


namespace train_length_l115_115187

theorem train_length (L V : ℝ) 
  (h1 : V = L / 10) 
  (h2 : V = (L + 870) / 39) 
  : L = 300 :=
by
  sorry

end train_length_l115_115187


namespace cos_double_angle_l115_115689

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 3) : Real.cos (2 * α) = -4 / 5 := 
  sorry

end cos_double_angle_l115_115689


namespace relationship_among_a_b_c_l115_115091

noncomputable def a := Real.sqrt 0.5
noncomputable def b := Real.sqrt 0.3
noncomputable def c := Real.log 0.2 / Real.log 0.3

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l115_115091


namespace a_100_correct_l115_115106

variable (a_n : ℕ → ℕ) (S₉ : ℕ) (a₁₀ : ℕ)

def is_arth_seq (a_n : ℕ → ℕ) := ∃ a d, ∀ n, a_n n = a + n * d

noncomputable def a_100 (a₅ d : ℕ) : ℕ := a₅ + 95 * d

theorem a_100_correct
  (h1 : ∃ S₉, 9 * a_n 4 = S₉)
  (h2 : a_n 9 = 8)
  (h3 : is_arth_seq a_n) :
  a_100 (a_n 4) 1 = 98 :=
by
  sorry

end a_100_correct_l115_115106


namespace total_participants_l115_115909

theorem total_participants (freshmen sophomores : ℕ) (h1 : freshmen = 8) (h2 : sophomores = 5 * freshmen) : freshmen + sophomores = 48 := 
by
  sorry

end total_participants_l115_115909


namespace trig_expression_value_l115_115179

theorem trig_expression_value (θ : ℝ) (h1 : Real.tan (2 * θ) = -2 * Real.sqrt 2)
  (h2 : 2 * θ > Real.pi / 2 ∧ 2 * θ < Real.pi) : 
  (2 * Real.cos θ / 2 ^ 2 - Real.sin θ - 1) / (Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) = 2 * Real.sqrt 2 - 3 :=
by
  sorry

end trig_expression_value_l115_115179


namespace units_digit_G100_l115_115403

def G (n : ℕ) := 3 * 2 ^ (2 ^ n) + 2

theorem units_digit_G100 : (G 100) % 10 = 0 :=
by
  sorry

end units_digit_G100_l115_115403


namespace proof_cos_2x_cos_2y_l115_115772

variable {θ x y : ℝ}

-- Conditions
def is_arith_seq (a b c : ℝ) := b = (a + c) / 2
def is_geom_seq (a b c : ℝ) := b^2 = a * c

-- Proving the given statement with the provided conditions
theorem proof_cos_2x_cos_2y (h_arith : is_arith_seq (Real.sin θ) (Real.sin x) (Real.cos θ))
                            (h_geom : is_geom_seq (Real.sin θ) (Real.sin y) (Real.cos θ)) :
  2 * Real.cos (2 * x) = Real.cos (2 * y) :=
sorry

end proof_cos_2x_cos_2y_l115_115772


namespace t_is_perfect_square_l115_115014

variable (n : ℕ) (hpos : 0 < n)
variable (t : ℕ) (ht : t = 2 + 2 * Nat.sqrt (1 + 12 * n^2))

theorem t_is_perfect_square (n : ℕ) (hpos : 0 < n) (t : ℕ) (ht : t = 2 + 2 * Nat.sqrt (1 + 12 * n^2)) : 
  ∃ k : ℕ, t = k * k := 
sorry

end t_is_perfect_square_l115_115014


namespace construct_line_through_points_l115_115489

-- Definitions of the conditions
def points_on_sheet (A B : ℝ × ℝ) : Prop := A ≠ B
def tool_constraints (ruler_length compass_max_opening distance_A_B : ℝ) : Prop :=
  distance_A_B > 2 * ruler_length ∧ distance_A_B > 2 * compass_max_opening

-- The main theorem statement
theorem construct_line_through_points (A B : ℝ × ℝ) (ruler_length compass_max_opening : ℝ) 
  (h_points : points_on_sheet A B) 
  (h_constraints : tool_constraints ruler_length compass_max_opening (dist A B)) : 
  ∃ line : ℝ × ℝ → Prop, line A ∧ line B :=
sorry

end construct_line_through_points_l115_115489


namespace total_height_increase_in_4_centuries_l115_115829

def height_increase_per_decade : ℕ := 75
def years_per_century : ℕ := 100
def years_per_decade : ℕ := 10
def centuries : ℕ := 4

theorem total_height_increase_in_4_centuries :
  height_increase_per_decade * (centuries * years_per_century / years_per_decade) = 3000 := by
  sorry

end total_height_increase_in_4_centuries_l115_115829


namespace remainder_abc_mod9_l115_115038

open Nat

-- Define the conditions for the problem
variables (a b c : ℕ)

-- Assume conditions: a, b, c are non-negative and less than 9, and the given congruences
theorem remainder_abc_mod9 (h1 : a < 9) (h2 : b < 9) (h3 : c < 9)
  (h4 : (a + 3 * b + 2 * c) % 9 = 3)
  (h5 : (2 * a + 2 * b + 3 * c) % 9 = 6)
  (h6 : (3 * a + b + 2 * c) % 9 = 1) :
  (a * b * c) % 9 = 4 :=
sorry

end remainder_abc_mod9_l115_115038


namespace square_land_plot_area_l115_115905

theorem square_land_plot_area (side_length : ℕ) (h1 : side_length = 40) : side_length * side_length = 1600 :=
by
  sorry

end square_land_plot_area_l115_115905


namespace population_percentage_l115_115643

theorem population_percentage (total_population : ℕ) (percentage : ℕ) (result : ℕ) :
  total_population = 25600 → percentage = 90 → result = (percentage * total_population) / 100 → result = 23040 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end population_percentage_l115_115643


namespace range_of_m_l115_115805

theorem range_of_m (h : ¬ (∀ x : ℝ, ∃ m : ℝ, 4 ^ x - 2 ^ (x + 1) + m = 0) → false) : 
  ∀ m : ℝ, m ≤ 1 :=
by
  sorry

end range_of_m_l115_115805


namespace find_m_l115_115302

def A (m : ℤ) : Set ℤ := {2, 5, m ^ 2 - m}
def B (m : ℤ) : Set ℤ := {2, m + 3}

theorem find_m (m : ℤ) : A m ∩ B m = B m → m = 3 := by
  sorry

end find_m_l115_115302


namespace quadrilateral_ABCD_pq_sum_l115_115533

noncomputable def AB_pq_sum : ℕ :=
  let p : ℕ := 9
  let q : ℕ := 141
  p + q

theorem quadrilateral_ABCD_pq_sum (BC CD AD : ℕ) (m_angle_A m_angle_B : ℕ) (hBC : BC = 8) (hCD : CD = 12) (hAD : AD = 10) (hAngleA : m_angle_A = 60) (hAngleB : m_angle_B = 60) : AB_pq_sum = 150 := by sorry

end quadrilateral_ABCD_pq_sum_l115_115533


namespace point_divides_segment_l115_115984

theorem point_divides_segment (x₁ y₁ x₂ y₂ m n : ℝ) (h₁ : (x₁, y₁) = (3, 7)) (h₂ : (x₂, y₂) = (5, 1)) (h₃ : m = 1) (h₄ : n = 3) :
  ( (m * x₂ + n * x₁) / (m + n), (m * y₂ + n * y₁) / (m + n) ) = (3.5, 5.5) :=
by
  sorry

end point_divides_segment_l115_115984


namespace stamps_on_last_page_l115_115239

theorem stamps_on_last_page
  (B : ℕ) (P_b : ℕ) (S_p : ℕ) (S_p_star : ℕ) 
  (B_comp : ℕ) (P_last : ℕ) 
  (stamps_total : ℕ := B * P_b * S_p) 
  (pages_total : ℕ := stamps_total / S_p_star)
  (pages_comp : ℕ := B_comp * P_b)
  (pages_filled : ℕ := pages_total - pages_comp) :
  stamps_total - (pages_total - 1) * S_p_star = 8 :=
by
  -- Proof steps would follow here.
  sorry

end stamps_on_last_page_l115_115239


namespace sqrt_eighteen_simplifies_l115_115788

open Real

theorem sqrt_eighteen_simplifies :
  sqrt 18 = 3 * sqrt 2 :=
by
  sorry

end sqrt_eighteen_simplifies_l115_115788


namespace opposite_of_neg_2022_eq_2022_l115_115612

-- Define what it means to find the opposite of a number
def opposite (n : Int) : Int := -n

-- State the theorem that needs to be proved
theorem opposite_of_neg_2022_eq_2022 : opposite (-2022) = 2022 :=
by
  -- Proof would go here but we skip it with sorry
  sorry

end opposite_of_neg_2022_eq_2022_l115_115612


namespace odd_and_periodic_function_l115_115871

noncomputable def f : ℝ → ℝ := sorry

lemma given_conditions (x : ℝ) : 
  (f (10 + x) = f (10 - x)) ∧ (f (20 - x) = -f (20 + x)) :=
  sorry

theorem odd_and_periodic_function (x : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f (x + 40) = f x) :=
  sorry

end odd_and_periodic_function_l115_115871


namespace Josanna_min_avg_score_l115_115222

theorem Josanna_min_avg_score (scores : List ℕ) (cur_avg target_avg : ℚ)
  (next_test_bonus : ℚ) (additional_avg_points : ℚ) : ℚ :=
  let cur_avg := (92 + 81 + 75 + 65 + 88) / 5
  let target_avg := cur_avg + 6
  let needed_total := target_avg * 7
  let additional_points := 401 + 5
  let needed_sum := needed_total - additional_points
  needed_sum / 2

noncomputable def min_avg_score : ℚ :=
  Josanna_min_avg_score [92, 81, 75, 65, 88] 80.2 86.2 5 6

example : min_avg_score = 99 :=
by
  sorry

end Josanna_min_avg_score_l115_115222


namespace average_children_in_families_with_children_l115_115321

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l115_115321


namespace intersection_of_A_and_B_l115_115563

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l115_115563


namespace train_passes_jogger_in_40_seconds_l115_115790

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

end train_passes_jogger_in_40_seconds_l115_115790


namespace value_of_expression_l115_115064

theorem value_of_expression (x : ℕ) (h : x = 2) : x + x * x^x = 10 := by
  rw [h] -- Substituting x = 2
  sorry

end value_of_expression_l115_115064


namespace non_congruent_rectangles_unique_l115_115185

theorem non_congruent_rectangles_unique (P : ℕ) (w : ℕ) (h : ℕ) :
  P = 72 ∧ w = 14 ∧ 2 * (w + h) = P → 
  (∃ h, w = 14 ∧ 2 * (w + h) = 72 ∧ 
  ∀ w' h', w' = w → 2 * (w' + h') = 72 → (h' = h)) :=
by
  sorry

end non_congruent_rectangles_unique_l115_115185


namespace value_of_x_l115_115168

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 := 
by
  sorry

end value_of_x_l115_115168


namespace domain_of_f_l115_115312

noncomputable def f (x : ℝ) := Real.log x / Real.log 6

noncomputable def g (x : ℝ) := Real.log x / Real.log 5

noncomputable def h (x : ℝ) := Real.log x / Real.log 3

open Set

theorem domain_of_f :
  (∀ x, x > 7776 → ∃ y, y = (h ∘ g ∘ f) x) :=
by
  sorry

end domain_of_f_l115_115312


namespace inequality_proof_l115_115922

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (1 + a + a * b)) + (b / (1 + b + b * c)) + (c / (1 + c + c * a)) ≤ 1 :=
by
  sorry

end inequality_proof_l115_115922


namespace cost_of_candy_l115_115630

theorem cost_of_candy (initial_amount remaining_amount : ℕ) (h_init : initial_amount = 4) (h_remaining : remaining_amount = 3) : initial_amount - remaining_amount = 1 :=
by
  sorry

end cost_of_candy_l115_115630


namespace chen_recording_l115_115516

variable (standard xia_steps chen_steps : ℕ)
variable (xia_record : ℤ)

-- Conditions: 
-- standard = 5000
-- Xia walked 6200 steps, recorded as +1200 steps
def met_standard (s : ℕ) : Prop :=
  s >= 5000

def xia_condition := (xia_steps = 6200) ∧ (xia_record = 1200) ∧ (xia_record = (xia_steps : ℤ) - 5000)

-- Question and solution combined into a statement: 
-- Chen walked 4800 steps, recorded as -200 steps
def chen_condition := (chen_steps = 4800) ∧ (met_standard chen_steps = false) → (((standard : ℤ) - chen_steps) * -1 = -200)

-- Proof goal:
theorem chen_recording (h₁ : standard = 5000) (h₂ : xia_condition xia_steps xia_record):
  chen_condition standard chen_steps :=
by
  sorry

end chen_recording_l115_115516


namespace vacation_cost_correct_l115_115132

namespace VacationCost

-- Define constants based on conditions
def starting_charge_per_dog : ℝ := 2
def charge_per_block : ℝ := 1.25
def number_of_dogs : ℕ := 20
def total_blocks : ℕ := 128
def family_members : ℕ := 5

-- Define total earnings from walking dogs
def total_earnings : ℝ :=
  (number_of_dogs * starting_charge_per_dog) + (total_blocks * charge_per_block)

-- Define the total cost of the vacation
noncomputable def total_cost_of_vacation : ℝ :=
  total_earnings / family_members * family_members

-- Proof statement: The total cost of the vacation is $200
theorem vacation_cost_correct : total_cost_of_vacation = 200 := by
  sorry

end VacationCost

end vacation_cost_correct_l115_115132


namespace solve_for_x_l115_115589

theorem solve_for_x :
  ∃ x : ℝ, 5 ^ (Real.logb 5 15) = 7 * x + 2 ∧ x = 13 / 7 :=
by
  sorry

end solve_for_x_l115_115589


namespace average_weight_of_Arun_l115_115869

theorem average_weight_of_Arun :
  ∃ avg_weight : Real,
    (avg_weight = (65 + 68) / 2) ∧
    ∀ w : Real, (65 < w ∧ w < 72) ∧ (60 < w ∧ w < 70) ∧ (w ≤ 68) → avg_weight = 66.5 :=
by
  -- we will fill the details of the proof here
  sorry

end average_weight_of_Arun_l115_115869


namespace product_of_roots_eq_neg_14_l115_115985

theorem product_of_roots_eq_neg_14 :
  ∀ (x : ℝ), 25 * x^2 + 60 * x - 350 = 0 → ((-350) / 25) = -14 :=
by
  intros x h
  sorry

end product_of_roots_eq_neg_14_l115_115985


namespace boxes_needed_l115_115151

-- Define Marilyn's total number of bananas
def num_bananas : Nat := 40

-- Define the number of bananas per box
def bananas_per_box : Nat := 5

-- Calculate the number of boxes required for the given number of bananas and bananas per box
def num_boxes (total_bananas : Nat) (bananas_each_box : Nat) : Nat :=
  total_bananas / bananas_each_box

-- Statement to be proved: given the specific conditions, the result should be 8
theorem boxes_needed : num_boxes num_bananas bananas_per_box = 8 :=
sorry

end boxes_needed_l115_115151


namespace sqrt_neg_sq_eq_two_l115_115870

theorem sqrt_neg_sq_eq_two : Real.sqrt ((-2 : ℝ)^2) = 2 := by
  -- Proof intentionally omitted.
  sorry

end sqrt_neg_sq_eq_two_l115_115870


namespace solve_rational_equation_l115_115764

theorem solve_rational_equation (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 4/5) :
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 20*x - 40)/(5*x - 4) = -5 ↔ x = -3 :=
by 
  sorry

end solve_rational_equation_l115_115764


namespace solve_inequality_l115_115290

def inequality_solution (x : ℝ) : Prop := |2 * x - 1| - x ≥ 2 

theorem solve_inequality (x : ℝ) : 
  inequality_solution x ↔ (x ≥ 3 ∨ x ≤ -1/3) :=
by sorry

end solve_inequality_l115_115290


namespace interest_rate_same_l115_115269

theorem interest_rate_same (initial_amount: ℝ) (interest_earned: ℝ) 
  (time_period1: ℝ) (time_period2: ℝ) (principal: ℝ) (initial_rate: ℝ) : 
  initial_amount * initial_rate * time_period2 = interest_earned * 100 ↔ initial_rate = 12 
  :=
by
  sorry

end interest_rate_same_l115_115269


namespace not_chosen_rate_l115_115768

theorem not_chosen_rate (sum : ℝ) (interest_15_percent : ℝ) (extra_interest : ℝ) : 
  sum = 7000 ∧ interest_15_percent = 2100 ∧ extra_interest = 420 →
  ∃ R : ℝ, (sum * 0.15 * 2 = interest_15_percent) ∧ 
           (interest_15_percent - (sum * R / 100 * 2) = extra_interest) ∧ 
           R = 12 := 
by {
  sorry
}

end not_chosen_rate_l115_115768


namespace Victoria_money_left_l115_115998

noncomputable def Victoria_initial_money : ℝ := 10000
noncomputable def jacket_price : ℝ := 250
noncomputable def trousers_price : ℝ := 180
noncomputable def purse_price : ℝ := 450
noncomputable def jackets_bought : ℕ := 8
noncomputable def trousers_bought : ℕ := 15
noncomputable def purses_bought : ℕ := 4
noncomputable def discount_rate : ℝ := 0.15
noncomputable def dinner_bill_inclusive : ℝ := 552.50
noncomputable def dinner_service_charge_rate : ℝ := 0.15

theorem Victoria_money_left : 
  Victoria_initial_money - 
  ((jackets_bought * jacket_price + trousers_bought * trousers_price) * (1 - discount_rate) + 
   purses_bought * purse_price + 
   dinner_bill_inclusive / (1 + dinner_service_charge_rate)) = 3725 := 
by 
  sorry

end Victoria_money_left_l115_115998


namespace quadratic_positive_imp_ineq_l115_115761

theorem quadratic_positive_imp_ineq (b c : ℤ) :
  (∀ x : ℤ, x^2 + b * x + c > 0) → b^2 - 4 * c ≤ 0 :=
by 
  sorry

end quadratic_positive_imp_ineq_l115_115761


namespace real_solutions_l115_115379

theorem real_solutions (x : ℝ) :
  (x ≠ 3 ∧ x ≠ 7) →
  ((x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)) /
  ((x - 3) * (x - 7) * (x - 3)) = 1 →
  x = 3 + Real.sqrt 3 ∨ x = 3 - Real.sqrt 3 ∨ x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5 :=
by
  sorry

end real_solutions_l115_115379


namespace hotel_room_assignment_even_hotel_room_assignment_odd_l115_115727

def smallest_n_even (k : ℕ) (m : ℕ) (h1 : k = 2 * m) : ℕ :=
  100 * (m + 1)

def smallest_n_odd (k : ℕ) (m : ℕ) (h1 : k = 2 * m + 1) : ℕ :=
  100 * (m + 1) + 1

theorem hotel_room_assignment_even (k m : ℕ) (h1 : k = 2 * m) :
  ∃ n, n = smallest_n_even k m h1 ∧ n >= 100 :=
  by
  sorry

theorem hotel_room_assignment_odd (k m : ℕ) (h1 : k = 2 * m + 1) :
  ∃ n, n = smallest_n_odd k m h1 ∧ n >= 100 :=
  by
  sorry

end hotel_room_assignment_even_hotel_room_assignment_odd_l115_115727


namespace cost_per_dozen_l115_115899

theorem cost_per_dozen (total_cost : ℝ) (total_rolls dozens : ℝ) (cost_per_dozen : ℝ) (h₁ : total_cost = 15) (h₂ : total_rolls = 36) (h₃ : dozens = total_rolls / 12) (h₄ : cost_per_dozen = total_cost / dozens) : cost_per_dozen = 5 :=
by
  sorry

end cost_per_dozen_l115_115899


namespace eval_oplus_otimes_l115_115311

-- Define the operations ⊕ and ⊗
def my_oplus (a b : ℕ) := a + b + 1
def my_otimes (a b : ℕ) := a * b - 1

-- Statement of the proof problem
theorem eval_oplus_otimes : my_oplus (my_oplus 5 7) (my_otimes 2 4) = 21 :=
by
  sorry

end eval_oplus_otimes_l115_115311


namespace find_two_numbers_l115_115046

theorem find_two_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = 5) (harmonic_mean : 2 * a * b / (a + b) = 5 / 3) :
  (a = (15 + Real.sqrt 145) / 4 ∧ b = (15 - Real.sqrt 145) / 4) ∨
  (a = (15 - Real.sqrt 145) / 4 ∧ b = (15 + Real.sqrt 145) / 4) :=
by
  sorry

end find_two_numbers_l115_115046


namespace ratio_children_to_adults_l115_115808

variable (f m c : ℕ)

-- Conditions
def average_age_female (f : ℕ) := 35
def average_age_male (m : ℕ) := 30
def average_age_child (c : ℕ) := 10
def overall_average_age (f m c : ℕ) := 25

-- Total age sums based on given conditions
def total_age_sum_female (f : ℕ) := 35 * f
def total_age_sum_male (m : ℕ) := 30 * m
def total_age_sum_child (c : ℕ) := 10 * c

-- Total sum and average conditions
def total_age_sum (f m c : ℕ) := total_age_sum_female f + total_age_sum_male m + total_age_sum_child c
def total_members (f m c : ℕ) := f + m + c

theorem ratio_children_to_adults (f m c : ℕ) (h : (total_age_sum f m c) / (total_members f m c) = 25) :
  (c : ℚ) / (f + m) = 2 / 3 := sorry

end ratio_children_to_adults_l115_115808


namespace calculate_expression_l115_115932

theorem calculate_expression : 
  let x := 7.5
  let y := 2.5
  (x ^ y + Real.sqrt x + y ^ x) - (x ^ 2 + y ^ y + Real.sqrt y) = 679.2044 :=
by
  sorry

end calculate_expression_l115_115932


namespace petya_friends_l115_115056

theorem petya_friends (x : ℕ) (total_stickers : ℕ)
  (h1 : total_stickers = 5 * x + 8)
  (h2 : total_stickers = 6 * x - 11) :
  x = 19 :=
by {
  sorry
}

end petya_friends_l115_115056


namespace solution_m_plus_n_l115_115391

variable (m n : ℝ)

theorem solution_m_plus_n 
  (h₁ : m ≠ 0)
  (h₂ : m^2 + m * n - m = 0) :
  m + n = 1 := by
  sorry

end solution_m_plus_n_l115_115391


namespace seventy_fifth_elem_in_s_l115_115551

-- Define the set s
def s : Set ℕ := {x | ∃ n : ℕ, x = 8 * n + 5}

-- State the main theorem
theorem seventy_fifth_elem_in_s : (∃ n : ℕ, n = 74 ∧ (8 * n + 5) = 597) :=
by
  -- The proof is skipped using sorry
  sorry

end seventy_fifth_elem_in_s_l115_115551


namespace b_alone_days_l115_115649

-- Definitions from the conditions
def work_rate_b (W_b : ℝ) : ℝ := W_b
def work_rate_a (W_b : ℝ) : ℝ := 2 * W_b
def work_rate_c (W_b : ℝ) : ℝ := 6 * W_b
def combined_work_rate (W_b : ℝ) : ℝ := work_rate_a W_b + work_rate_b W_b + work_rate_c W_b
def total_days_together : ℝ := 10
def total_work (W_b : ℝ) : ℝ := combined_work_rate W_b * total_days_together

-- The proof problem
theorem b_alone_days (W_b : ℝ) : 90 = total_work W_b / work_rate_b W_b :=
by
  sorry

end b_alone_days_l115_115649


namespace inequality_am_gm_l115_115090

variable (a b x y : ℝ)

theorem inequality_am_gm (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
  (a^2 / x) + (b^2 / y) ≥ (a + b)^2 / (x + y) :=
by {
  -- proof will be filled here
  sorry
}

end inequality_am_gm_l115_115090


namespace imaginary_part_of_z_l115_115133

def z : ℂ := 1 - 2 * Complex.I

theorem imaginary_part_of_z : Complex.im z = -2 := by
  sorry

end imaginary_part_of_z_l115_115133


namespace original_number_of_matchsticks_l115_115134

-- Define the conditions
def matchsticks_per_house : ℕ := 10
def houses_created : ℕ := 30
def total_matchsticks_used := houses_created * matchsticks_per_house

-- Define the question and the proof goal
theorem original_number_of_matchsticks (h : total_matchsticks_used = (Michael's_original_matchsticks / 2)) :
  (Michael's_original_matchsticks = 600) :=
by
  sorry

end original_number_of_matchsticks_l115_115134


namespace g_of_5_l115_115184

variable {g : ℝ → ℝ}
variable (h1 : ∀ x y : ℝ, 2 * x * g y = 3 * y * g x)
variable (h2 : g 10 = 15)

theorem g_of_5 : g 5 = 45 / 4 :=
  sorry

end g_of_5_l115_115184


namespace air_quality_probability_l115_115715

variable (p_good_day : ℝ) (p_good_two_days : ℝ)

theorem air_quality_probability
  (h1 : p_good_day = 0.75)
  (h2 : p_good_two_days = 0.6) :
  (p_good_two_days / p_good_day = 0.8) :=
by
  rw [h1, h2]
  norm_num

end air_quality_probability_l115_115715


namespace equipment_value_decrease_l115_115097

theorem equipment_value_decrease (a : ℝ) (b : ℝ) (n : ℕ) :
  (a * (1 - b / 100)^n) = a * (1 - b/100)^n :=
sorry

end equipment_value_decrease_l115_115097


namespace tyler_bird_pairs_l115_115122

theorem tyler_bird_pairs (n_species : ℕ) (pairs_per_species : ℕ) (total_pairs : ℕ)
  (h1 : n_species = 29)
  (h2 : pairs_per_species = 7)
  (h3 : total_pairs = n_species * pairs_per_species) : total_pairs = 203 :=
by
  sorry

end tyler_bird_pairs_l115_115122


namespace find_range_a_l115_115578

theorem find_range_a (x y a : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 2 ≤ y ∧ y ≤ 3) :
  (∀ x y, (1 ≤ x ∧ x ≤ 2) → (2 ≤ y ∧ y ≤ 3) → (xy ≤ a*x^2 + 2*y^2)) ↔ (-1/2 ≤ a) :=
sorry

end find_range_a_l115_115578


namespace two_connected_iff_constructible_with_H_paths_l115_115329

-- A graph is represented as a structure with vertices and edges
structure Graph where
  vertices : Type
  edges : vertices → vertices → Prop

-- Function to check if a graph is 2-connected
noncomputable def isTwoConnected (G : Graph) : Prop := sorry

-- Function to check if a graph can be constructed by adding H-paths
noncomputable def constructibleWithHPaths (G H : Graph) : Prop := sorry

-- Given a graph G and subgraph H, we need to prove the equivalence
theorem two_connected_iff_constructible_with_H_paths (G H : Graph) :
  (isTwoConnected G) ↔ (constructibleWithHPaths G H) := sorry

end two_connected_iff_constructible_with_H_paths_l115_115329


namespace tan_inequality_l115_115418

open Real

theorem tan_inequality {x1 x2 : ℝ} 
  (h1 : 0 < x1 ∧ x1 < π / 2) 
  (h2 : 0 < x2 ∧ x2 < π / 2) 
  (h3 : x1 ≠ x2) : 
  (1 / 2 * (tan x1 + tan x2) > tan ((x1 + x2) / 2)) :=
sorry

end tan_inequality_l115_115418


namespace value_of_polynomial_l115_115595

variable {R : Type} [CommRing R]

theorem value_of_polynomial 
  (m : R) 
  (h : 2 * m^2 - 3 * m - 1 = 0) : 
  6 * m^2 - 9 * m + 2019 = 2022 := by
  sorry

end value_of_polynomial_l115_115595


namespace cos_120_eq_neg_half_l115_115322

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_120_eq_neg_half_l115_115322


namespace cos_triple_angle_l115_115407

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 :=
by
  sorry

end cos_triple_angle_l115_115407


namespace lorry_empty_weight_l115_115804

-- Define variables for the weights involved
variable (lw : ℕ)  -- weight of the lorry when empty
variable (bl : ℕ)  -- number of bags of apples
variable (bw : ℕ)  -- weight of each bag of apples
variable (total_weight : ℕ)  -- total loaded weight of the lorry

-- Given conditions
axiom lorry_loaded_weight : bl = 20 ∧ bw = 60 ∧ total_weight = 1700

-- The theorem we want to prove
theorem lorry_empty_weight : (∀ lw bw, total_weight - bl * bw = lw) → lw = 500 :=
by
  intro h
  rw [←h lw bw]
  sorry

end lorry_empty_weight_l115_115804


namespace quadratic_root_ratio_eq_l115_115840

theorem quadratic_root_ratio_eq (k : ℝ) :
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ (x = 3 * y ∨ y = 3 * x) ∧ x + y = -10 ∧ x * y = k) → k = 18.75 := by
  sorry

end quadratic_root_ratio_eq_l115_115840


namespace find_function_l115_115830

theorem find_function (f : ℝ → ℝ) (c : ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) →
  (∀ x : ℝ, f x = x ^ 3 + c * x) :=
by
  -- The proof details will be filled here.
  sorry

end find_function_l115_115830


namespace smallest_three_digit_divisible_l115_115710

theorem smallest_three_digit_divisible :
  ∃ (A B C : Nat), A ≠ 0 ∧ 100 ≤ (100 * A + 10 * B + C) ∧ (100 * A + 10 * B + C) < 1000 ∧
  (10 * A + B) > 9 ∧ (10 * B + C) > 9 ∧ 
  (100 * A + 10 * B + C) % (10 * A + B) = 0 ∧ (100 * A + 10 * B + C) % (10 * B + C) = 0 ∧
  (100 * A + 10 * B + C) = 110 :=
by
  sorry

end smallest_three_digit_divisible_l115_115710


namespace find_q_l115_115574

theorem find_q (q x : ℝ) (h1 : x = 2) (h2 : q * x - 3 = 11) : q = 7 :=
by
  sorry

end find_q_l115_115574


namespace planar_figure_area_l115_115476

noncomputable def side_length : ℝ := 10
noncomputable def area_of_square : ℝ := side_length * side_length
noncomputable def number_of_squares : ℕ := 6
noncomputable def total_area_of_planar_figure : ℝ := number_of_squares * area_of_square

theorem planar_figure_area : total_area_of_planar_figure = 600 :=
by
  sorry

end planar_figure_area_l115_115476


namespace fraction_division_l115_115841

theorem fraction_division :
  (1/4) / 2 = 1/8 :=
by
  sorry

end fraction_division_l115_115841


namespace rationalize_denominator_sum_l115_115749

theorem rationalize_denominator_sum :
  let expr := 1 / (Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 11)
  ∃ (A B C D E F G H I : ℤ), 
    I > 0 ∧
    expr * (Real.sqrt 5 + Real.sqrt 3 - Real.sqrt 11) /
    ((Real.sqrt 5 + Real.sqrt 3)^2 - (Real.sqrt 11)^2) = 
        (A * Real.sqrt B + C * Real.sqrt D + E * Real.sqrt F + 
         G * Real.sqrt H) / I ∧
    (A + B + C + D + E + F + G + H + I) = 225 :=
by
  sorry

end rationalize_denominator_sum_l115_115749


namespace shenille_points_l115_115737

def shenille_total_points (x y : ℕ) : ℝ :=
  0.6 * x + 0.6 * y

theorem shenille_points (x y : ℕ) (h : x + y = 30) : 
  shenille_total_points x y = 18 := by
  sorry

end shenille_points_l115_115737


namespace pond_water_after_45_days_l115_115941

theorem pond_water_after_45_days :
  let initial_amount := 300
  let daily_evaporation := 1
  let rain_every_third_day := 2
  let total_days := 45
  let non_third_days := total_days - (total_days / 3)
  let third_days := total_days / 3
  let total_net_change := (non_third_days * (-daily_evaporation)) + (third_days * (rain_every_third_day - daily_evaporation))
  let final_amount := initial_amount + total_net_change
  final_amount = 285 :=
by
  sorry

end pond_water_after_45_days_l115_115941


namespace roots_of_polynomial_l115_115680

theorem roots_of_polynomial : {x : ℝ | (x^2 - 5*x + 6)*(x - 1)*(x - 6) = 0} = {1, 2, 3, 6} :=
by
  -- proof goes here
  sorry

end roots_of_polynomial_l115_115680


namespace ellen_bakes_6_balls_of_dough_l115_115725

theorem ellen_bakes_6_balls_of_dough (rising_time baking_time total_time : ℕ) (h_rise : rising_time = 3) (h_bake : baking_time = 2) (h_total : total_time = 20) :
  ∃ n : ℕ, (rising_time + baking_time) + rising_time * (n - 1) = total_time ∧ n = 6 :=
by sorry

end ellen_bakes_6_balls_of_dough_l115_115725


namespace determine_numbers_l115_115346

theorem determine_numbers (n : ℕ) (m : ℕ) (x y z u v : ℕ) (h₁ : 10000 <= n ∧ n < 100000)
(h₂ : n = 10000 * x + 1000 * y + 100 * z + 10 * u + v)
(h₃ : m = 1000 * x + 100 * y + 10 * u + v)
(h₄ : x ≠ 0)
(h₅ : n % m = 0) :
∃ a : ℕ, (10 <= a ∧ a <= 99 ∧ n = a * 1000) :=
sorry

end determine_numbers_l115_115346


namespace exponentiation_rule_l115_115896

theorem exponentiation_rule (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

end exponentiation_rule_l115_115896


namespace simplify_radical_product_l115_115770

theorem simplify_radical_product : 
  (32^(1/5)) * (8^(1/3)) * (4^(1/2)) = 8 := 
by
  sorry

end simplify_radical_product_l115_115770


namespace flip_ratio_l115_115049

theorem flip_ratio (jen_triple_flips tyler_double_flips : ℕ)
  (hjen : jen_triple_flips = 16)
  (htyler : tyler_double_flips = 12)
  : 2 * tyler_double_flips / 3 * jen_triple_flips = 1 / 2 := 
by
  rw [hjen, htyler]
  norm_num
  sorry

end flip_ratio_l115_115049


namespace div_eq_210_over_79_l115_115308

def a_at_b (a b : ℕ) : ℤ := a^2 * b - a * (b^2)
def a_hash_b (a b : ℕ) : ℤ := a^2 + b^2 - a * b

theorem div_eq_210_over_79 : (a_at_b 10 3) / (a_hash_b 10 3) = 210 / 79 :=
by
  -- This is a placeholder and needs to be filled with the actual proof.
  sorry

end div_eq_210_over_79_l115_115308


namespace determine_alpha_l115_115640

variables (m n : ℝ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_mn : m + n = 1)
variables (α : ℝ)

-- Defining the minimum value condition
def minimum_value_condition : Prop :=
  (1 / m + 16 / n) = 25

-- Defining the curve passing through point P
def passes_through_P : Prop :=
  (m / 5) ^ α = (m / 4)

theorem determine_alpha
  (h_min_value : minimum_value_condition m n)
  (h_passes_through : passes_through_P m α) :
  α = 1 / 2 :=
sorry

end determine_alpha_l115_115640


namespace dana_pencils_more_than_jayden_l115_115793

theorem dana_pencils_more_than_jayden :
  ∀ (Jayden_has_pencils : ℕ) (Marcus_has_pencils : ℕ) (Dana_has_pencils : ℕ),
    Jayden_has_pencils = 20 →
    Marcus_has_pencils = Jayden_has_pencils / 2 →
    Dana_has_pencils = Marcus_has_pencils + 25 →
    Dana_has_pencils - Jayden_has_pencils = 15 :=
by
  intros Jayden_has_pencils Marcus_has_pencils Dana_has_pencils
  intro h1
  intro h2
  intro h3
  sorry

end dana_pencils_more_than_jayden_l115_115793


namespace toby_steps_l115_115482

theorem toby_steps (sunday tuesday wednesday thursday friday_saturday monday : ℕ) :
    sunday = 9400 →
    tuesday = 8300 →
    wednesday = 9200 →
    thursday = 8900 →
    friday_saturday = 9050 →
    7 * 9000 = 63000 →
    monday = 63000 - (sunday + tuesday + wednesday + thursday + 2 * friday_saturday) → monday = 9100 :=
by
  intros hs ht hw hth hfs htc hnm
  sorry

end toby_steps_l115_115482


namespace sum_of_interior_angles_n_plus_2_l115_115040

-- Define the sum of the interior angles formula for a convex polygon
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the degree measure of the sum of the interior angles of a convex polygon with n sides being 1800
def sum_of_n_sides_is_1800 (n : ℕ) : Prop := sum_of_interior_angles n = 1800

-- Translate the proof problem as a theorem statement in Lean
theorem sum_of_interior_angles_n_plus_2 (n : ℕ) (h: sum_of_n_sides_is_1800 n) : 
  sum_of_interior_angles (n + 2) = 2160 :=
sorry

end sum_of_interior_angles_n_plus_2_l115_115040


namespace parabola_points_count_l115_115875

theorem parabola_points_count :
  ∃ n : ℕ, n = 8 ∧ 
    (∀ x y : ℕ, (y = -((x^2 : ℤ) / 3) + 7 * (x : ℤ) + 54) → 1 ≤ x ∧ x ≤ 26 ∧ x % 3 = 0) :=
by
  sorry

end parabola_points_count_l115_115875


namespace solve_inequality_l115_115601

-- Define the conditions
def condition_inequality (x : ℝ) : Prop := abs x + abs (2 * x - 3) ≥ 6

-- Define the solution set form
def solution_set (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 3

-- State the theorem
theorem solve_inequality (x : ℝ) : condition_inequality x → solution_set x := 
by 
  sorry

end solve_inequality_l115_115601


namespace math_problem_l115_115497

theorem math_problem : -5 * (-6) - 2 * (-3 * (-7) + (-8)) = 4 := 
  sorry

end math_problem_l115_115497


namespace smallest_digit_to_correct_l115_115423

def incorrect_sum : ℕ := 2104
def correct_sum : ℕ := 738 + 625 + 841
def difference : ℕ := correct_sum - incorrect_sum

theorem smallest_digit_to_correct (d : ℕ) (h : difference = 100) :
  d = 6 := 
sorry

end smallest_digit_to_correct_l115_115423


namespace find_distance_between_stripes_l115_115317

-- Define the problem conditions
def parallel_curbs (a b : ℝ) := ∀ g : ℝ, g * a = b
def crosswalk_conditions (curb_distance curb_length stripe_length : ℝ) := 
  curb_distance = 60 ∧ curb_length = 22 ∧ stripe_length = 65

-- State the theorem
theorem find_distance_between_stripes (curb_distance curb_length stripe_length : ℝ) 
  (h : ℝ) (H : crosswalk_conditions curb_distance curb_length stripe_length) :
  h = 264 / 13 :=
sorry

end find_distance_between_stripes_l115_115317


namespace golden_section_length_l115_115759

theorem golden_section_length (MN : ℝ) (MP NP : ℝ) (hMN : MN = 1) (hP : MP + NP = MN) (hgolden : MN / MP = MP / NP) (hMP_gt_NP : MP > NP) : MP = (Real.sqrt 5 - 1) / 2 :=
by sorry

end golden_section_length_l115_115759


namespace max_sum_x_y_l115_115745

theorem max_sum_x_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x^3 + y^3 + (x + y)^3 + 36 * x * y = 3456) : x + y ≤ 12 :=
sorry

end max_sum_x_y_l115_115745


namespace calculate_expression_l115_115815

theorem calculate_expression :
  ((12 ^ 12 / 12 ^ 11) ^ 2 * 4 ^ 2) / 2 ^ 4 = 144 :=
by
  sorry

end calculate_expression_l115_115815


namespace sum_GCF_LCM_l115_115124

-- Definitions of GCD and LCM for the numbers 18, 27, and 36
def GCF : ℕ := Nat.gcd (Nat.gcd 18 27) 36
def LCM : ℕ := Nat.lcm (Nat.lcm 18 27) 36

-- Theorem statement proof
theorem sum_GCF_LCM : GCF + LCM = 117 := by
  sorry

end sum_GCF_LCM_l115_115124


namespace ms_warren_running_time_l115_115475

theorem ms_warren_running_time 
  (t : ℝ) 
  (ht_total_distance : 6 * t + 2 * 0.5 = 3) : 
  60 * t = 20 := by 
  sorry

end ms_warren_running_time_l115_115475


namespace problem_l115_115347

theorem problem (θ : ℝ) (htan : Real.tan θ = 1 / 3) : Real.cos θ ^ 2 + 2 * Real.sin θ = 6 / 5 := 
by
  sorry

end problem_l115_115347


namespace inequality_square_l115_115566

theorem inequality_square (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 :=
by
  sorry

end inequality_square_l115_115566


namespace slower_train_time_to_pass_driver_faster_one_l115_115415

noncomputable def convert_speed (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def relative_speed (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let speed1 := convert_speed speed1_kmh
  let speed2 := convert_speed speed2_kmh
  speed1 + speed2

noncomputable def time_to_pass (length1_m length2_m speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let relative_speed := relative_speed speed1_kmh speed2_kmh
  (length1_m + length2_m) / relative_speed

theorem slower_train_time_to_pass_driver_faster_one :
  ∀ (length1 length2 speed1 speed2 : ℝ),
    length1 = 900 → length2 = 900 →
    speed1 = 45 → speed2 = 30 →
    time_to_pass length1 length2 speed1 speed2 = 86.39 :=
by
  intros
  simp only [time_to_pass, relative_speed, convert_speed]
  sorry

end slower_train_time_to_pass_driver_faster_one_l115_115415


namespace solve_for_x_l115_115917

theorem solve_for_x (q r x : ℚ)
  (h1 : 5 / 6 = q / 90)
  (h2 : 5 / 6 = (q + r) / 102)
  (h3 : 5 / 6 = (x - r) / 150) :
  x = 135 :=
by sorry

end solve_for_x_l115_115917


namespace distance_A_to_focus_l115_115088

noncomputable def focus_of_parabola (a b c : ℝ) : ℝ × ℝ :=
  ((b^2 - 4*a*c) / (4*a), 0)

theorem distance_A_to_focus 
  (P : ℝ × ℝ) (parabola : ℝ → ℝ → Prop)
  (A B : ℝ × ℝ)
  (hP : P = (-2, 0))
  (hPar : ∀ x y, parabola x y ↔ y^2 = 4 * x)
  (hLine : ∃ m b, ∀ x y, y = m * x + b ∧ y^2 = 4 * x → (x, y) = A ∨ (x, y) = B)
  (hDist : dist P A = (1 / 2) * dist A B)
  (hFocus : focus_of_parabola 1 0 (-1) = (1, 0)) :
  dist A (1, 0) = 5 / 3 :=
sorry

end distance_A_to_focus_l115_115088


namespace angle_bisector_slope_l115_115274

theorem angle_bisector_slope (k : ℚ) : 
  (∀ x : ℚ, (y = 2 * x ∧ y = 4 * x) → (y = k * x)) → k = -12 / 7 :=
sorry

end angle_bisector_slope_l115_115274


namespace complement_A_in_U_l115_115851

open Set

-- Definitions for sets
def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

-- The proof goal: prove that the complement of A in U is {4}
theorem complement_A_in_U : (U \ A) = {4} := by
  sorry

end complement_A_in_U_l115_115851


namespace exists_linear_function_second_quadrant_l115_115419

theorem exists_linear_function_second_quadrant (k b : ℝ) (h1 : k > 0) (h2 : b > 0) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = k * x + b) ∧ (∀ x, x < 0 → f x > 0) :=
by
  -- Prove there exists a linear function of the form f(x) = kx + b with given conditions
  -- Skip the proof for now
  sorry

end exists_linear_function_second_quadrant_l115_115419


namespace max_value_of_x0_l115_115157

noncomputable def sequence_max_value (seq : Fin 1996 → ℝ) (pos_seq : ∀ i, seq i > 0) : Prop :=
  seq 0 = seq 1995 ∧
  (∀ i : Fin 1995, seq i + 2 / seq i = 2 * seq (i + 1) + 1 / seq (i + 1)) ∧
  (seq 0 ≤ 2^997)

theorem max_value_of_x0 :
  ∃ seq : Fin 1996 → ℝ, ∀ pos_seq : ∀ i, seq i > 0, sequence_max_value seq pos_seq :=
sorry

end max_value_of_x0_l115_115157


namespace surface_area_inequality_l115_115746

theorem surface_area_inequality
  (a b c d e f S : ℝ) :
  S ≤ (Real.sqrt 3 / 6) * (a^2 + b^2 + c^2 + d^2 + e^2 + f^2) :=
sorry

end surface_area_inequality_l115_115746


namespace power_function_value_l115_115491

/-- Given a power function passing through a certain point, find the value at a specific point -/
theorem power_function_value (α : ℝ) (f : ℝ → ℝ) (h : f x = x ^ α) 
  (h_passes : f (1/4) = 4) : f 2 = 1/2 :=
sorry

end power_function_value_l115_115491


namespace circle_center_coordinates_l115_115691

theorem circle_center_coordinates (h k r : ℝ) :
  (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = 1 → (x - h)^2 + (y - k)^2 = r^2) →
  (h, k) = (2, -3) :=
by
  intro H
  sorry

end circle_center_coordinates_l115_115691


namespace cricket_team_members_l115_115057

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℚ) (wk_keeper_age : ℚ) 
  (avg_whole_team : ℚ) (avg_remaining_players : ℚ)
  (h1 : captain_age = 25)
  (h2 : wk_keeper_age = 28)
  (h3 : avg_whole_team = 22)
  (h4 : avg_remaining_players = 21)
  (h5 : 22 * n = 25 + 28 + 21 * (n - 2)) :
  n = 11 :=
by sorry

end cricket_team_members_l115_115057


namespace selling_price_correct_l115_115626

-- Define the conditions
def purchase_price : ℝ := 12000
def repair_costs : ℝ := 5000
def transportation_charges : ℝ := 1000
def profit_percentage : ℝ := 0.50

-- Calculate total cost
def total_cost : ℝ := purchase_price + repair_costs + transportation_charges

-- Define the selling price and the proof goal
def selling_price : ℝ := total_cost + (profit_percentage * total_cost)

-- Prove that the selling price equals Rs 27000
theorem selling_price_correct : selling_price = 27000 := 
by 
  -- Proof is not required, so we use sorry
  sorry

end selling_price_correct_l115_115626


namespace probability_both_heads_l115_115135

-- Define the sample space and the probability of each outcome
def sample_space : List (Bool × Bool) := [(true, true), (true, false), (false, true), (false, false)]

-- Define the function to check for both heads
def both_heads (outcome : Bool × Bool) : Bool :=
  outcome = (true, true)

-- Calculate the probability of both heads
theorem probability_both_heads :
  (sample_space.filter both_heads).length / sample_space.length = 1 / 4 := sorry

end probability_both_heads_l115_115135


namespace london_to_baglmintster_distance_l115_115296

variable (D : ℕ) -- distance from London to Baglmintster

-- Conditions
def meeting_point_condition_1 := D ≥ 40
def meeting_point_condition_2 := D ≥ 48
def initial_meeting := D - 40
def return_meeting := D - 48

theorem london_to_baglmintster_distance :
  (D - 40) + 48 = D + 8 ∧ 40 + (D - 48) = D - 8 → D = 72 :=
by
  intros h
  sorry

end london_to_baglmintster_distance_l115_115296


namespace tiling_impossible_l115_115352

theorem tiling_impossible (T2 T14 : ℕ) :
  let S_before := 2 * T2
  let S_after := 2 * (T2 - 1) + 1 
  S_after ≠ S_before :=
sorry

end tiling_impossible_l115_115352


namespace train_arrival_problem_shooting_problem_l115_115127

-- Define trials and outcome types
inductive OutcomeTrain : Type
| onTime
| notOnTime

inductive OutcomeShooting : Type
| hitTarget
| missTarget

-- Scenario 1: Train Arrival Problem
def train_arrival_trials_refers_to (n : Nat) : Prop := 
  ∃ trials : List OutcomeTrain, trials.length = 3

-- Scenario 2: Shooting Problem
def shooting_trials_refers_to (n : Nat) : Prop :=
  ∃ trials : List OutcomeShooting, trials.length = 2

theorem train_arrival_problem : train_arrival_trials_refers_to 3 :=
by
  sorry

theorem shooting_problem : shooting_trials_refers_to 2 :=
by
  sorry

end train_arrival_problem_shooting_problem_l115_115127


namespace union_of_M_N_l115_115183

def M : Set ℝ := { x | x^2 + 2*x = 0 }

def N : Set ℝ := { x | x^2 - 2*x = 0 }

theorem union_of_M_N : M ∪ N = {0, -2, 2} := sorry

end union_of_M_N_l115_115183


namespace find_principal_l115_115114

noncomputable def compoundPrincipal (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem find_principal :
  let A := 3969
  let r := 0.05
  let n := 1
  let t := 2
  compoundPrincipal A r n t = 3600 :=
by
  sorry

end find_principal_l115_115114


namespace min_value_frac_ineq_l115_115249

theorem min_value_frac_ineq (a b : ℝ) (h1 : a > 1) (h2 : b > 2) (h3 : a + b = 5) : 
  (1 / (a - 1) + 9 / (b - 2)) = 8 :=
sorry

end min_value_frac_ineq_l115_115249


namespace lunchroom_tables_l115_115978

/-- Given the total number of students and the number of students per table, 
    prove the number of tables in the lunchroom. -/
theorem lunchroom_tables (total_students : ℕ) (students_per_table : ℕ) 
  (h_total : total_students = 204) (h_per_table : students_per_table = 6) : 
  total_students / students_per_table = 34 := 
by
  sorry

end lunchroom_tables_l115_115978


namespace solution_set_max_value_l115_115147

-- Given function f(x)
def f (x : ℝ) : ℝ := |2 * x - 1| + |x - 1|

-- (I) Prove the solution set of f(x) ≤ 4 is {x | -2/3 ≤ x ≤ 2}
theorem solution_set : {x : ℝ | f x ≤ 4} = {x : ℝ | -2/3 ≤ x ∧ x ≤ 2} :=
sorry

-- (II) Given m is the minimum value of f(x)
def m := 1 / 2

-- Given a, b, c ∈ ℝ^+ and a + b + c = m
variables (a b c : ℝ)
variable (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h2 : a + b + c = m)

-- Prove the maximum value of √(2a + 1) + √(2b + 1) + √(2c + 1) is 2√3
theorem max_value : (Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) + Real.sqrt (2 * c + 1)) ≤ 2 * Real.sqrt 3 :=
sorry

end solution_set_max_value_l115_115147


namespace possible_values_of_k_l115_115101

theorem possible_values_of_k (n : ℕ) (h : n ≥ 3) :
  ∃ t : ℕ, k = 2 ^ t ∧ 2 ^ t ≥ n :=
sorry

end possible_values_of_k_l115_115101


namespace curve_C_straight_line_curve_C_not_tangent_curve_C_fixed_point_curve_C_intersect_l115_115769

noncomputable def curve_C (a x y : ℝ) := a * x ^ 2 + a * y ^ 2 - 2 * x - 2 * y = 0

theorem curve_C_straight_line (a : ℝ) : a = 0 → ∃ x y : ℝ, curve_C a x y :=
by
  intro ha
  use (-1), 1
  rw [curve_C, ha]
  simp

theorem curve_C_not_tangent (a : ℝ) : a = 1 → ¬ ∀ x y, 3 * x + y = 0 → curve_C a x y :=
by
  sorry

theorem curve_C_fixed_point (x y a : ℝ) : curve_C a 0 0 :=
by
  rw [curve_C]
  simp

theorem curve_C_intersect (a : ℝ) : a = 1 → ∃ x y : ℝ, (x + 2 * y = 0) ∧ curve_C a x y :=
by
  sorry

end curve_C_straight_line_curve_C_not_tangent_curve_C_fixed_point_curve_C_intersect_l115_115769


namespace number_of_sets_without_perfect_squares_l115_115219

/-- Define the set T_i of all integers n such that 200i ≤ n < 200(i + 1). -/
def T (i : ℕ) : Set ℕ := {n | 200 * i ≤ n ∧ n < 200 * (i + 1)}

/-- The total number of sets T_i from T_0 to T_{499}. -/
def total_sets : ℕ := 500

/-- The number of sets from T_0 to T_{499} that contain at least one perfect square. -/
def sets_with_perfect_squares : ℕ := 317

/-- The number of sets from T_0 to T_{499} that do not contain any perfect squares. -/
def sets_without_perfect_squares : ℕ := total_sets - sets_with_perfect_squares

/-- Proof that the number of sets T_0, T_1, T_2, ..., T_{499} that do not contain a perfect square is 183. -/
theorem number_of_sets_without_perfect_squares : sets_without_perfect_squares = 183 :=
by
  sorry

end number_of_sets_without_perfect_squares_l115_115219


namespace solve_for_x_l115_115964

theorem solve_for_x : ∃ x : ℝ, (6 * x) / 1.5 = 3.8 ∧ x = 0.95 := by
  use 0.95
  exact ⟨by norm_num, by norm_num⟩

end solve_for_x_l115_115964


namespace madison_classes_l115_115890

/-- Madison's classes -/
def total_bell_rings : ℕ := 9

/-- Each class requires two bell rings (one to start, one to end) -/
def bell_rings_per_class : ℕ := 2

/-- The number of classes Madison has on Monday -/
theorem madison_classes (total_bell_rings bell_rings_per_class : ℕ) (last_class_start_only : total_bell_rings % bell_rings_per_class = 1) : 
  (total_bell_rings - 1) / bell_rings_per_class + 1 = 5 :=
by
  sorry

end madison_classes_l115_115890


namespace strawberries_picking_problem_l115_115675

noncomputable def StrawberriesPicked : Prop :=
  let kg_to_lb := 2.2
  let marco_pounds := 1 + 3 * kg_to_lb
  let sister_pounds := 1.5 * marco_pounds
  let father_pounds := 2 * sister_pounds
  marco_pounds = 7.6 ∧ sister_pounds = 11.4 ∧ father_pounds = 22.8

theorem strawberries_picking_problem : StrawberriesPicked :=
  sorry

end strawberries_picking_problem_l115_115675


namespace number_of_men_in_first_group_l115_115624

theorem number_of_men_in_first_group 
    (x : ℕ) (H1 : ∀ (work_rate : ℕ → ℕ → ℚ), work_rate x 5 = 1 / (5 * x))
    (H2 : ∀ (work_rate : ℕ → ℕ → ℚ), work_rate 15 12 = 1 / (15 * 12))
    (H3 : ∀ (work_rate : ℕ → ℕ → ℚ), work_rate x 5 = work_rate 15 12) 
    : x = 36 := 
by {
    sorry
}

end number_of_men_in_first_group_l115_115624


namespace spherical_coordinates_standard_equivalence_l115_115854

def std_spherical_coords (ρ θ φ: ℝ) : Prop :=
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi

theorem spherical_coordinates_standard_equivalence :
  std_spherical_coords 5 (11 * Real.pi / 6) (2 * Real.pi - 5 * Real.pi / 3) :=
by
  sorry

end spherical_coordinates_standard_equivalence_l115_115854


namespace book_donation_growth_rate_l115_115309

theorem book_donation_growth_rate (x : ℝ) : 
  400 + 400 * (1 + x) + 400 * (1 + x)^2 = 1525 :=
sorry

end book_donation_growth_rate_l115_115309


namespace largest_n_binomial_l115_115766

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l115_115766


namespace adam_apples_count_l115_115958

variable (Jackie_apples : ℕ)
variable (extra_apples : ℕ)
variable (Adam_apples : ℕ)

theorem adam_apples_count (h1 : Jackie_apples = 9) (h2 : extra_apples = 5) (h3 : Adam_apples = Jackie_apples + extra_apples) :
  Adam_apples = 14 := 
by 
  sorry

end adam_apples_count_l115_115958


namespace min_width_of_garden_l115_115386

theorem min_width_of_garden (w : ℝ) (h : 0 < w) (h1 : w * (w + 20) ≥ 120) : w ≥ 4 :=
sorry

end min_width_of_garden_l115_115386


namespace digit_sum_10_pow_93_minus_937_l115_115596

-- Define a function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem digit_sum_10_pow_93_minus_937 :
  sum_of_digits (10^93 - 937) = 819 :=
by
  sorry

end digit_sum_10_pow_93_minus_937_l115_115596


namespace age_ratio_is_4_over_3_l115_115776

-- Define variables for ages
variable (R D : ℕ)

-- Conditions
axiom key_condition_R : R + 10 = 26
axiom key_condition_D : D = 12

-- Theorem statement: The ratio of Rahul's age to Deepak's age is 4/3
theorem age_ratio_is_4_over_3 (hR : R + 10 = 26) (hD : D = 12) : R / D = 4 / 3 :=
sorry

end age_ratio_is_4_over_3_l115_115776


namespace triangle_area_triangle_perimeter_l115_115528

noncomputable def area_of_triangle (A B C : ℝ) (a b c : ℝ) := 
  1/2 * b * c * (Real.sin A)

theorem triangle_area (A B C a b c : ℝ) 
  (h1 : b^2 + c^2 - a^2 = bc) 
  (h2 : A = Real.pi / 3) : 
  area_of_triangle A B C a b c = Real.sqrt 3 / 4 := 
  sorry

theorem triangle_perimeter (A B C a b c : ℝ) 
  (h1 : b^2 + c^2 - a^2 = bc) 
  (h2 : 4 * Real.cos B * Real.cos C - 1 = 0) 
  (h3 : b + c = 2)
  (h4 : a = 1) :
  a + b + c = 3 :=
  sorry

end triangle_area_triangle_perimeter_l115_115528


namespace negation_proof_l115_115171

open Classical

variable {x : ℝ}

theorem negation_proof :
  (∀ x : ℝ, (x + 1) ≥ 0 ∧ (x^2 - x) ≤ 0) ↔ ¬ (∃ x_0 : ℝ, (x_0 + 1) < 0 ∨ (x_0^2 - x_0) > 0) := 
by
  sorry

end negation_proof_l115_115171


namespace lino_shells_total_l115_115798

def picked_up_shells : Float := 324.0
def put_back_shells : Float := 292.0

theorem lino_shells_total : picked_up_shells - put_back_shells = 32.0 :=
by
  sorry

end lino_shells_total_l115_115798


namespace second_polygon_sides_l115_115344

theorem second_polygon_sides (s : ℝ) (h : s ≠ 0)
  (perimeter_eq : ∀ (p1 p2 : ℝ), p1 = p2)
  (first_sides : ℕ) (second_sides : ℕ)
  (first_polygon_side_length : ℝ) (second_polygon_side_length : ℝ)
  (first_sides_eq : first_sides = 50)
  (side_length_relation : first_polygon_side_length = 3 * second_polygon_side_length)
  (perimeter_relation : first_sides * first_polygon_side_length = second_sides * second_polygon_side_length) :
  second_sides = 150 := by
  sorry

end second_polygon_sides_l115_115344


namespace box_surface_area_is_276_l115_115361

-- Define the dimensions of the box
variables {l w h : ℝ}

-- Define the pricing function
def pricing (x y z : ℝ) : ℝ := 0.30 * x + 0.40 * y + 0.50 * z

-- Define the condition for the box fee
def box_fee (x y z : ℝ) (fee : ℝ) := pricing x y z = fee

-- Define the constraint that no faces are squares
def no_square_faces (l w h : ℝ) : Prop := 
  l ≠ w ∧ w ≠ h ∧ h ≠ l

-- Define the surface area calculation
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

-- The main theorem stating the problem
theorem box_surface_area_is_276 (l w h : ℝ) 
  (H1 : box_fee l w h 8.10 ∧ box_fee w h l 8.10)
  (H2 : box_fee l w h 8.70 ∧ box_fee w h l 8.70)
  (H3 : no_square_faces l w h) : 
  surface_area l w h = 276 := 
sorry

end box_surface_area_is_276_l115_115361


namespace direction_vector_of_line_m_l115_115729

noncomputable def projectionMatrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![ 5 / 21, -2 / 21, -2 / 7 ],
    ![ -2 / 21, 1 / 42, 1 / 14 ],
    ![ -2 / 7,  1 / 14, 4 / 7 ]
  ]

noncomputable def vectorI : Fin 3 → ℚ
  | 0 => 1
  | _ => 0

noncomputable def projectedVector : Fin 3 → ℚ :=
  fun i => (projectionMatrix.mulVec vectorI) i

theorem direction_vector_of_line_m :
  (projectedVector 0 = 5 / 21) ∧ 
  (projectedVector 1 = -2 / 21) ∧
  (projectedVector 2 = -6 / 21) ∧
  Nat.gcd (Nat.gcd 5 2) 6 = 1 :=
by
  sorry

end direction_vector_of_line_m_l115_115729


namespace sum_of_solutions_eq_9_l115_115750

theorem sum_of_solutions_eq_9 (x_1 x_2 : ℝ) (h : x^2 - 9 * x + 20 = 0) :
  x_1 + x_2 = 9 :=
sorry

end sum_of_solutions_eq_9_l115_115750


namespace find_x_l115_115996

theorem find_x 
  (x : ℝ)
  (h : 120 + 80 + x + x = 360) : 
  x = 80 :=
sorry

end find_x_l115_115996


namespace intersection_eq_l115_115976

noncomputable def A := {x : ℝ | x^2 - 4*x + 3 < 0 }
noncomputable def B := {x : ℝ | 2*x - 3 > 0 }

theorem intersection_eq : (A ∩ B) = {x : ℝ | (3 / 2) < x ∧ x < 3} := by
  sorry

end intersection_eq_l115_115976


namespace contrapositive_l115_115569

theorem contrapositive (x y : ℝ) : (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
by
  intro h
  sorry

end contrapositive_l115_115569


namespace value_of_b_div_a_l115_115701

theorem value_of_b_div_a (a b : ℝ) (h : |5 - a| + (b + 3)^2 = 0) : b / a = -3 / 5 :=
by
  sorry

end value_of_b_div_a_l115_115701


namespace area_of_grey_region_l115_115258

open Nat

theorem area_of_grey_region
  (a1 a2 b : ℕ)
  (h1 : a1 = 8 * 10)
  (h2 : a2 = 9 * 12)
  (hb : b = 37)
  : (a2 - (a1 - b) = 65) := by
  sorry

end area_of_grey_region_l115_115258


namespace aiyanna_cookies_l115_115370

theorem aiyanna_cookies (a b : ℕ) (h₁ : a = 129) (h₂ : b = a + 11) : b = 140 := by
  sorry

end aiyanna_cookies_l115_115370


namespace negation_example_l115_115409

theorem negation_example :
  (¬ (∃ n : ℕ, n^2 ≥ 2^n)) → (∀ n : ℕ, n^2 < 2^n) :=
by
  sorry

end negation_example_l115_115409


namespace correct_survey_method_l115_115863

-- Definitions for the conditions
def visionStatusOfMiddleSchoolStudentsNationwide := "Comprehensive survey is impractical for this large population."
def batchFoodContainsPreservatives := "Comprehensive survey is unnecessary, sampling survey would suffice."
def airQualityOfCity := "Comprehensive survey is impractical due to vast area, sampling survey is appropriate."
def passengersCarryProhibitedItems := "Comprehensive survey is necessary for security reasons."

-- Theorem stating that option C is the correct and reasonable choice
theorem correct_survey_method : airQualityOfCity = "Comprehensive survey is impractical due to vast area, sampling survey is appropriate." := by
  sorry

end correct_survey_method_l115_115863


namespace original_price_l115_115073

theorem original_price (selling_price profit_percent : ℝ) (h_sell : selling_price = 63) (h_profit : profit_percent = 5) : 
  selling_price / (1 + profit_percent / 100) = 60 :=
by sorry

end original_price_l115_115073


namespace Liz_team_deficit_l115_115362

theorem Liz_team_deficit :
  ∀ (initial_deficit liz_free_throws liz_three_pointers liz_jump_shots opponent_points : ℕ),
    initial_deficit = 20 →
    liz_free_throws = 5 →
    liz_three_pointers = 3 →
    liz_jump_shshots = 4 →
    opponent_points = 10 →
    (initial_deficit - (liz_free_throws * 1 + liz_three_pointers * 3 + liz_jump_shshots * 2 - opponent_points)) = 8 := by
  intros initial_deficit liz_free_throws liz_three_pointers liz_jump_shots opponent_points
  intros h_initial_deficit h_liz_free_throws h_liz_three_pointers h_liz_jump_shots h_opponent_points
  sorry

end Liz_team_deficit_l115_115362


namespace chord_length_of_circle_intersected_by_line_l115_115527

open Real

-- Definitions for the conditions given in the problem
def line_eqn (x y : ℝ) : Prop := x - y - 1 = 0
def circle_eqn (x y : ℝ) : Prop := x^2 - 4 * x + y^2 = 4

-- The proof statement (problem) in Lean 4
theorem chord_length_of_circle_intersected_by_line :
  ∀ (x y : ℝ), circle_eqn x y → line_eqn x y → ∃ L : ℝ, L = sqrt 17 := by
  sorry

end chord_length_of_circle_intersected_by_line_l115_115527


namespace distinct_rectangles_l115_115606

theorem distinct_rectangles :
  ∃! (l w : ℝ), l * w = 100 ∧ l + w = 24 :=
sorry

end distinct_rectangles_l115_115606


namespace ratio_of_rectangles_l115_115214

theorem ratio_of_rectangles (p q : ℝ) (h1 : q ≠ 0) 
    (h2 : q^2 = 1/4 * (2 * p * q  - q^2)) : p / q = 5 / 2 := 
sorry

end ratio_of_rectangles_l115_115214


namespace inverse_89_mod_90_l115_115644

theorem inverse_89_mod_90 : (89 * 89) % 90 = 1 := 
by
  -- Mathematical proof is skipped
  sorry

end inverse_89_mod_90_l115_115644


namespace range_of_x_l115_115200

theorem range_of_x (x : ℝ) (h : Real.log (x - 1) < 1) : 1 < x ∧ x < Real.exp 1 + 1 :=
by
  sorry

end range_of_x_l115_115200


namespace minimum_people_who_like_both_l115_115893

theorem minimum_people_who_like_both
    (total_people : ℕ)
    (vivaldi_likers : ℕ)
    (chopin_likers : ℕ)
    (people_surveyed : total_people = 150)
    (like_vivaldi : vivaldi_likers = 120)
    (like_chopin : chopin_likers = 90) :
    ∃ (both_likers : ℕ), both_likers = 60 ∧
                            vivaldi_likers + chopin_likers - both_likers ≤ total_people :=
by 
  sorry

end minimum_people_who_like_both_l115_115893


namespace coprime_odd_sum_of_floors_l115_115881

theorem coprime_odd_sum_of_floors (p q : ℕ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h_coprime : Nat.gcd p q = 1) : 
  (List.sum (List.map (λ i => Nat.floor ((i • q : ℚ) / p)) ((List.range (p / 2 + 1)).tail)) +
   List.sum (List.map (λ i => Nat.floor ((i • p : ℚ) / q)) ((List.range (q / 2 + 1)).tail))) =
  (p - 1) * (q - 1) / 4 :=
by
  sorry

end coprime_odd_sum_of_floors_l115_115881


namespace number_of_male_students_drawn_l115_115080

theorem number_of_male_students_drawn (total_students : ℕ) (total_male_students : ℕ) (total_female_students : ℕ) (sample_size : ℕ)
    (H1 : total_students = 350)
    (H2 : total_male_students = 70)
    (H3 : total_female_students = 280)
    (H4 : sample_size = 50) :
    total_male_students * sample_size / total_students = 10 :=
by
  sorry

end number_of_male_students_drawn_l115_115080


namespace smallest_x_for_multiple_l115_115289

theorem smallest_x_for_multiple (x : ℕ) (h1 : 450 = 2^1 * 3^2 * 5^2) (h2 : 640 = 2^7 * 5^1) :
  (450 * x) % 640 = 0 ↔ x = 64 :=
sorry

end smallest_x_for_multiple_l115_115289


namespace hotel_elevator_cubic_value_l115_115262

noncomputable def hotel_elevator_cubic : ℚ → ℚ := sorry

theorem hotel_elevator_cubic_value :
  hotel_elevator_cubic 11 = 11 ∧
  hotel_elevator_cubic 12 = 12 ∧
  hotel_elevator_cubic 13 = 14 ∧
  hotel_elevator_cubic 14 = 15 →
  hotel_elevator_cubic 15 = 13 :=
sorry

end hotel_elevator_cubic_value_l115_115262


namespace intersecting_lines_solution_l115_115711

theorem intersecting_lines_solution (x y b : ℝ) 
  (h₁ : y = 2 * x - 5)
  (h₂ : y = 3 * x + b)
  (hP : x = 1 ∧ y = -3) : 
  b = -6 ∧ x = 1 ∧ y = -3 := by
  sorry

end intersecting_lines_solution_l115_115711


namespace largest_four_digit_number_divisible_by_33_l115_115707

theorem largest_four_digit_number_divisible_by_33 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (33 ∣ n) ∧ ∀ m : ℕ, (1000 ≤ m ∧ m < 10000 ∧ 33 ∣ m → m ≤ 9999) :=
by
  sorry

end largest_four_digit_number_divisible_by_33_l115_115707


namespace value_of_f_l115_115486

def B : Set ℚ := {x | x ≠ 0 ∧ x ≠ 1}

def f (x : ℚ) : ℝ := sorry

noncomputable def h (x : ℚ) : ℚ :=
  1 / (1 - x)

lemma cyclic_of_h :
  ∀ x ∈ B, h (h (h x)) = x :=
sorry

lemma functional_property (x : ℚ) (hx : x ∈ B) :
  f x + f (h x) = 2 * Real.log (|x|) :=
sorry

theorem value_of_f :
  f 2023 = Real.log 2023 :=
sorry

end value_of_f_l115_115486


namespace quadratic_min_value_l115_115588

theorem quadratic_min_value (p r : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x^2 + 2 * p * x + r) (h₁ : ∃ x₀, f x₀ = 1 ∧ ∀ x, f x₀ ≤ f x) : r = p^2 + 1 :=
by
  sorry

end quadratic_min_value_l115_115588


namespace exterior_angle_of_regular_octagon_l115_115501

theorem exterior_angle_of_regular_octagon (sum_of_exterior_angles : ℝ) (n_sides : ℕ) (is_regular : n_sides = 8 ∧ sum_of_exterior_angles = 360) :
  sum_of_exterior_angles / n_sides = 45 := by
  sorry

end exterior_angle_of_regular_octagon_l115_115501


namespace bingley_bracelets_final_l115_115796

-- Definitions
def initial_bingley_bracelets : Nat := 5
def kelly_bracelets_given : Nat := 16 / 4
def bingley_bracelets_after_kelly : Nat := initial_bingley_bracelets + kelly_bracelets_given
def bingley_bracelets_given_to_sister : Nat := bingley_bracelets_after_kelly / 3
def bingley_remaining_bracelets : Nat := bingley_bracelets_after_kelly - bingley_bracelets_given_to_sister

-- Theorem
theorem bingley_bracelets_final : bingley_remaining_bracelets = 6 := by
  sorry

end bingley_bracelets_final_l115_115796


namespace tetrahedron_face_area_inequality_l115_115225

theorem tetrahedron_face_area_inequality
  (T_ABC T_ABD T_ACD T_BCD : ℝ)
  (h : T_ABC ≥ 0 ∧ T_ABD ≥ 0 ∧ T_ACD ≥ 0 ∧ T_BCD ≥ 0) :
  T_ABC < T_ABD + T_ACD + T_BCD :=
sorry

end tetrahedron_face_area_inequality_l115_115225


namespace meaningful_domain_of_function_l115_115005

theorem meaningful_domain_of_function : ∀ x : ℝ, (∃ y : ℝ, y = 3 / Real.sqrt (x - 2)) → x > 2 :=
by
  intros x h
  sorry

end meaningful_domain_of_function_l115_115005


namespace solve_for_y_l115_115453

-- Given condition
def equation (y : ℚ) := (8 * y^2 + 90 * y + 5) / (3 * y^2 + 4 * y + 49) = 4 * y + 1

-- Prove the resulting polynomial equation
theorem solve_for_y (y : ℚ) (h : equation y) : 12 * y^3 + 11 * y^2 + 110 * y + 44 = 0 :=
sorry

end solve_for_y_l115_115453


namespace packaging_combinations_l115_115481

theorem packaging_combinations :
  let wraps := 10
  let ribbons := 4
  let cards := 5
  let stickers := 6
  wraps * ribbons * cards * stickers = 1200 :=
by
  rfl

end packaging_combinations_l115_115481


namespace evaluate_expression_l115_115338

theorem evaluate_expression (b : ℕ) (h : b = 4) : (b ^ b - b * (b - 1) ^ b) ^ b = 21381376 := by
  sorry

end evaluate_expression_l115_115338


namespace find_y_in_interval_l115_115463

theorem find_y_in_interval :
  { y : ℝ | y^2 + 7 * y < 12 } = { y : ℝ | -9 < y ∧ y < 2 } :=
sorry

end find_y_in_interval_l115_115463


namespace necessary_but_not_sufficient_condition_l115_115547

-- Define the condition p: x^2 - x < 0
def p (x : ℝ) : Prop := x^2 - x < 0

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (x : ℝ) : Prop := -1 < x ∧ x < 1

-- State the theorem
theorem necessary_but_not_sufficient_condition :
  ∀ x : ℝ, p x → necessary_but_not_sufficient x :=
sorry

end necessary_but_not_sufficient_condition_l115_115547


namespace committeeFormation_l115_115098

-- Establish the given problem conditions in Lean

open Classical

-- Noncomputable because we are working with combinations and products
noncomputable def numberOfWaysToFormCommittee (numSchools : ℕ) (membersPerSchool : ℕ) (hostSchools : ℕ) (hostReps : ℕ) (nonHostReps : ℕ) : ℕ :=
  let totalSchools := numSchools
  let chooseHostSchools := Nat.choose totalSchools hostSchools
  let chooseHostRepsPerSchool := Nat.choose membersPerSchool hostReps
  let allHostRepsChosen := chooseHostRepsPerSchool ^ hostSchools
  let chooseNonHostRepsPerSchool := Nat.choose membersPerSchool nonHostReps
  let allNonHostRepsChosen := chooseNonHostRepsPerSchool ^ (totalSchools - hostSchools)
  chooseHostSchools * allHostRepsChosen * allNonHostRepsChosen

-- We now state our theorem
theorem committeeFormation : numberOfWaysToFormCommittee 4 6 2 3 1 = 86400 :=
by
  -- This is the lemma we need to prove
  sorry

end committeeFormation_l115_115098


namespace investment_value_after_five_years_l115_115224

theorem investment_value_after_five_years :
  let initial_investment := 10000
  let year1 := initial_investment * (1 - 0.05) * (1 + 0.02)
  let year2 := year1 * (1 + 0.10) * (1 + 0.02)
  let year3 := year2 * (1 + 0.04) * (1 + 0.02)
  let year4 := year3 * (1 - 0.03) * (1 + 0.02)
  let year5 := year4 * (1 + 0.08) * (1 + 0.02)
  year5 = 12570.99 :=
  sorry

end investment_value_after_five_years_l115_115224


namespace inequality_proof_l115_115141

variable (a b c : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_abc : a * b * c = 1)

theorem inequality_proof :
  (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b))) 
  ≥ (3 / 2) + (1 / 4) * (a * (c - b) ^ 2 / (c + b) + b * (c - a) ^ 2 / (c + a) + c * (b - a) ^ 2 / (b + a)) :=
by
  sorry

end inequality_proof_l115_115141


namespace standard_colony_condition_l115_115784

noncomputable def StandardBacterialColony : Prop := sorry

theorem standard_colony_condition (visible_mass_of_microorganisms : Prop) 
                                   (single_mother_cell : Prop) 
                                   (solid_culture_medium : Prop) 
                                   (not_multiple_types : Prop) 
                                   : StandardBacterialColony :=
sorry

end standard_colony_condition_l115_115784


namespace range_of_m_l115_115012

theorem range_of_m (x m : ℝ) (h₁ : x^2 - 3 * x + 2 > 0) (h₂ : ¬(x^2 - 3 * x + 2 > 0) → x < m) : 2 < m :=
by
  sorry

end range_of_m_l115_115012


namespace distinct_m_value_l115_115838

theorem distinct_m_value (a b : ℝ) (m : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
    (h_b_eq_2a : b = 2 * a) (h_m_eq_neg2a_b : m = -2 * a / b) : 
    ∃! (m : ℝ), m = -1 :=
by sorry

end distinct_m_value_l115_115838


namespace problem_statement_l115_115623

-- Mathematical Definitions
def num_students : ℕ := 6
def num_boys : ℕ := 4
def num_girls : ℕ := 2
def num_selected : ℕ := 3

def event_A : Prop := ∃ (boyA : ℕ), boyA < num_boys
def event_B : Prop := ∃ (girlB : ℕ), girlB < num_girls

def C (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of ways to select 3 out of 6 students
def total_ways : ℕ := C num_students num_selected

-- Probability of event A
def P_A : ℚ := C (num_students - 1) (num_selected - 1) / total_ways

-- Probability of events A and B
def P_AB : ℚ := C (num_students - 2) (num_selected - 2) / total_ways

-- Conditional probability P(B|A)
def P_B_given_A : ℚ := P_AB / P_A

theorem problem_statement : P_B_given_A = 2 / 5 := sorry

end problem_statement_l115_115623


namespace part1_part2_l115_115679

theorem part1 (n : Nat) (hn : 0 < n) : 
  (∃ k, -5^4 + 5^5 + 5^n = k^2) -> n = 5 :=
by
  sorry

theorem part2 (n : Nat) (hn : 0 < n) : 
  (∃ m, 2^4 + 2^7 + 2^n = m^2) -> n = 8 :=
by
  sorry

end part1_part2_l115_115679


namespace tammy_total_miles_l115_115591

noncomputable def miles_per_hour : ℝ := 1.527777778
noncomputable def hours_driven : ℝ := 36.0
noncomputable def total_miles := miles_per_hour * hours_driven

theorem tammy_total_miles : abs (total_miles - 55.0) < 1e-5 :=
by
  sorry

end tammy_total_miles_l115_115591


namespace total_heads_l115_115382

def number_of_heads := 1
def number_of_feet_hen := 2
def number_of_feet_cow := 4
def total_feet := 144

theorem total_heads (H C : ℕ) (h_hens : H = 24) (h_feet : number_of_feet_hen * H + number_of_feet_cow * C = total_feet) :
  H + C = 48 :=
sorry

end total_heads_l115_115382


namespace math_problem_l115_115592

open Nat

-- Given conditions
def S (n : ℕ) : ℕ := n * (n + 1)

-- Definitions for the terms a_n, b_n, c_n, and the sum T_n
def a_n (n : ℕ) (h : n ≠ 0) : ℕ := if n = 1 then 2 else 2 * n
def b_n (n : ℕ) (h : n ≠ 0) : ℕ := 2 * (3^n + 1)
def c_n (n : ℕ) (h : n ≠ 0) : ℕ := a_n n h * b_n n h / 4
def T (n : ℕ) (h : 0 < n) : ℕ := 
  (2 * n - 1) * 3^(n + 1) / 4 + 3 / 4 + n * (n + 1) / 2

-- Main theorem to establish the solution
theorem math_problem (n : ℕ) (h : n ≠ 0) : 
  S n = n * (n + 1) →
  a_n n h = 2 * n ∧ 
  b_n n h = 2 * (3^n + 1) ∧ 
  T n (Nat.pos_of_ne_zero h) = (2 * n - 1) * 3^(n + 1) / 4 + 3 / 4 + n * (n + 1) / 2 := 
by
  intros hS
  sorry

end math_problem_l115_115592


namespace numeric_value_of_BAR_l115_115666

variable (b a t c r : ℕ)

-- Conditions from the problem
axiom h1 : b + a + t = 6
axiom h2 : c + a + t = 8
axiom h3 : c + a + r = 12

-- Required to prove
theorem numeric_value_of_BAR : b + a + r = 10 :=
by
  -- Proof goes here
  sorry

end numeric_value_of_BAR_l115_115666


namespace dihedral_angle_equivalence_l115_115145

namespace CylinderGeometry

variables {α β γ : ℝ} 

-- Given conditions
axiom axial_cross_section : Type
axiom point_on_circumference (C : axial_cross_section) : Prop
axiom dihedral_angle (α: ℝ) : Prop
axiom angle_CAB (β : ℝ) : Prop
axiom angle_CA1B (γ : ℝ) : Prop

-- Proven statement
theorem dihedral_angle_equivalence
    (hx : point_on_circumference C)
    (hα : dihedral_angle α)
    (hβ : angle_CAB β)
    (hγ : angle_CA1B γ):
  α = Real.arcsin (Real.cos β / Real.cos γ) :=
sorry

end CylinderGeometry

end dihedral_angle_equivalence_l115_115145


namespace balloons_remaining_each_friend_l115_115042

def initial_balloons : ℕ := 250
def number_of_friends : ℕ := 5
def balloons_taken_back : ℕ := 11

theorem balloons_remaining_each_friend :
  (initial_balloons / number_of_friends) - balloons_taken_back = 39 :=
by
  sorry

end balloons_remaining_each_friend_l115_115042


namespace find_divisor_l115_115911

theorem find_divisor (d x k j : ℤ) (h₁ : x = k * d + 5) (h₂ : 7 * x = j * d + 8) : d = 11 :=
sorry

end find_divisor_l115_115911


namespace middle_aged_employees_participating_l115_115568

-- Define the total number of employees and the ratio
def total_employees : ℕ := 1200
def ratio_elderly : ℕ := 1
def ratio_middle_aged : ℕ := 5
def ratio_young : ℕ := 6

-- Define the number of employees chosen for the performance
def chosen_employees : ℕ := 36

-- Calculate the number of middle-aged employees participating in the performance
theorem middle_aged_employees_participating : (36 * ratio_middle_aged / (ratio_elderly + ratio_middle_aged + ratio_young)) = 15 :=
by
  sorry

end middle_aged_employees_participating_l115_115568


namespace range_of_m_l115_115843

-- Define the conditions
theorem range_of_m (m : ℝ) : 
    (∀ x : ℝ, (m-1) * x^2 + 2 * x + 1 = 0 → 
     (m-1 ≠ 0) ∧ 
     (4 - 4 * (m - 1) > 0)) ↔ 
    (m < 2 ∧ m ≠ 1) :=
sorry

end range_of_m_l115_115843


namespace probability_same_color_is_one_third_l115_115011

-- Define a type for colors
inductive Color 
| red 
| white 
| blue 

open Color

-- Define the function to calculate the probability of the same color selection
def sameColorProbability : ℚ :=
  let total_outcomes := 3 * 3
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

-- Theorem stating that the probability is 1/3
theorem probability_same_color_is_one_third : sameColorProbability = 1 / 3 :=
by
  -- Steps of proof will be provided here
  sorry

end probability_same_color_is_one_third_l115_115011


namespace comprehensive_survey_is_C_l115_115303

def option (label : String) (description : String) := (label, description)

def A := option "A" "Investigating the current mental health status of middle school students nationwide"
def B := option "B" "Investigating the compliance of food in our city"
def C := option "C" "Investigating the physical and mental conditions of classmates in the class"
def D := option "D" "Investigating the viewership ratings of Nanjing TV's 'Today's Life'"

theorem comprehensive_survey_is_C (suitable: (String × String → Prop)) :
  suitable C :=
sorry

end comprehensive_survey_is_C_l115_115303


namespace minimum_ticket_cost_l115_115602

-- Definitions of the conditions in Lean
def southern_cities : ℕ := 4
def northern_cities : ℕ := 5
def one_way_ticket_cost (N : ℝ) : ℝ := N
def round_trip_ticket_cost (N : ℝ) : ℝ := 1.6 * N

-- The main theorem to prove
theorem minimum_ticket_cost (N : ℝ) : 
  (∀ (Y1 Y2 Y3 Y4 : ℕ), 
  (∀ (S1 S2 S3 S4 S5 : ℕ), 
  southern_cities = 4 → northern_cities = 5 →
  one_way_ticket_cost N = N →
  round_trip_ticket_cost N = 1.6 * N →
  ∃ (total_cost : ℝ), total_cost = 6.4 * N)) :=
sorry

end minimum_ticket_cost_l115_115602


namespace no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two_l115_115714

theorem no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two :
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), n^2012 - 1 = 2^k :=
by
  sorry  

end no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two_l115_115714


namespace eight_b_equals_neg_eight_l115_115248

theorem eight_b_equals_neg_eight (a b : ℤ) (h1 : 6 * a + 3 * b = 3) (h2 : a = 2 * b + 3) : 8 * b = -8 := 
by
  sorry

end eight_b_equals_neg_eight_l115_115248


namespace max_sum_of_cubes_l115_115828

open Real

theorem max_sum_of_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * sqrt 5 :=
by
  sorry

end max_sum_of_cubes_l115_115828


namespace option_D_correct_l115_115279

variable (x : ℝ)

theorem option_D_correct : (2 * x^7) / x = 2 * x^6 := sorry

end option_D_correct_l115_115279


namespace minimum_value_expr_l115_115388

noncomputable def expr (x : ℝ) : ℝ := 9 * x + 3 / (x ^ 3)

theorem minimum_value_expr : (∀ x : ℝ, x > 0 → expr x ≥ 12) ∧ (∃ x : ℝ, x > 0 ∧ expr x = 12) :=
by
  sorry

end minimum_value_expr_l115_115388


namespace bus_length_is_200_l115_115736

def length_of_bus (distance_km distance_secs passing_secs : ℕ) : ℕ :=
  let speed_kms := distance_km / distance_secs
  let speed_ms := speed_kms * 1000
  speed_ms * passing_secs

theorem bus_length_is_200 
  (distance_km : ℕ) (distance_secs : ℕ) (passing_secs : ℕ)
  (h1 : distance_km = 12) (h2 : distance_secs = 300) (h3 : passing_secs = 5) : 
  length_of_bus distance_km distance_secs passing_secs = 200 := 
  by
    sorry

end bus_length_is_200_l115_115736


namespace range_of_a_l115_115304

open Set Real

theorem range_of_a (a : ℝ) (α : ℝ → Prop) (β : ℝ → Prop) (hα : ∀ x, α x ↔ x ≥ a) (hβ : ∀ x, β x ↔ |x - 1| < 1)
  (h : ∀ x, (β x → α x) ∧ (∃ x, α x ∧ ¬β x)) : a ≤ 0 :=
by
  sorry

end range_of_a_l115_115304


namespace woman_lawyer_probability_l115_115129

theorem woman_lawyer_probability (total_members women_count lawyer_prob : ℝ) 
  (h1: total_members = 100) 
  (h2: women_count = 0.70 * total_members) 
  (h3: lawyer_prob = 0.40) : 
  (0.40 * 0.70) = 0.28 := by sorry

end woman_lawyer_probability_l115_115129


namespace apple_cost_l115_115018

theorem apple_cost (l q : ℕ)
  (h1 : 30 * l + 6 * q = 366)
  (h2 : 15 * l = 150)
  (h3 : 30 * l + (333 - 30 * l) / q * q = 333) :
  30 + (333 - 30 * l) / q = 33 := 
sorry

end apple_cost_l115_115018


namespace correct_statements_arithmetic_seq_l115_115180

/-- For an arithmetic sequence {a_n} with a1 > 0 and common difference d ≠ 0, 
    the correct statements among options A, B, C, and D are B and C. -/
theorem correct_statements_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h_seq : ∀ n, a (n + 1) = a n + d) 
  (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2) (h_a1_pos : a 1 > 0) (h_d_ne_0 : d ≠ 0) : 
  (S 5 = S 9 → 
   S 7 = (10 * a 4) / 2) ∧ 
  (S 6 > S 7 → S 7 > S 8) := 
sorry

end correct_statements_arithmetic_seq_l115_115180


namespace sector_area_l115_115955

theorem sector_area (θ r a : ℝ) (hθ : θ = 2) (haarclength : r * θ = 4) : 
  (1/2) * r * r * θ = 4 :=
by {
  -- Proof goes here
  sorry
}

end sector_area_l115_115955


namespace find_number_of_students_l115_115744

theorem find_number_of_students
  (n : ℕ)
  (average_marks : ℕ → ℚ)
  (wrong_mark_corrected : ℕ → ℕ → ℚ)
  (correct_avg_marks_pred : ℕ → ℚ → Prop)
  (h1 : average_marks n = 60)
  (h2 : wrong_mark_corrected 90 15 = 75)
  (h3 : correct_avg_marks_pred n 57.5) :
  n = 30 :=
sorry

end find_number_of_students_l115_115744


namespace initial_balance_l115_115956

theorem initial_balance (X : ℝ) : 
  (X - 60 - 30 - 0.25 * (X - 60 - 30) - 10 = 100) ↔ (X = 236.67) := 
  by
    sorry

end initial_balance_l115_115956


namespace next_month_eggs_l115_115824

-- Given conditions definitions
def eggs_left_last_month : ℕ := 27
def eggs_after_buying : ℕ := 58
def eggs_eaten_this_month : ℕ := 48

-- Calculate number of eggs mother buys each month
def eggs_bought_each_month : ℕ := eggs_after_buying - eggs_left_last_month

-- Remaining eggs before next purchase
def eggs_left_before_next_purchase : ℕ := eggs_after_buying - eggs_eaten_this_month

-- Final amount of eggs after mother buys next month's supply
def total_eggs_next_month : ℕ := eggs_left_before_next_purchase + eggs_bought_each_month

-- Prove the total number of eggs next month equals 41
theorem next_month_eggs : total_eggs_next_month = 41 := by
  sorry

end next_month_eggs_l115_115824


namespace find_number_l115_115743

theorem find_number (x : ℤ) (h : 5 * x - 28 = 232) : x = 52 :=
by
  sorry

end find_number_l115_115743


namespace part1_part2_l115_115885

-- Define the predicate for the inequality
def prop (x m : ℝ) : Prop := x^2 - 2 * m * x - 3 * m^2 < 0

-- Define the set A
def A (m : ℝ) : Prop := m < -2 ∨ m > 2 / 3

-- Define the predicate for the other inequality
def prop_B (x a : ℝ) : Prop := x^2 - 2 * a * x + a^2 - 1 < 0

-- Define the set B in terms of a
def B (x a : ℝ) : Prop := a - 1 < x ∧ x < a + 1

-- Define the propositions required in the problem
theorem part1 (m : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → prop x m) ↔ A m :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, B x a → A x) ∧ (∃ x, A x ∧ ¬ B x a) ↔ (a ≤ -3 ∨ a ≥ 5 / 3) :=
sorry

end part1_part2_l115_115885


namespace syllogistic_reasoning_l115_115483

theorem syllogistic_reasoning (a b c : Prop) (h1 : b → c) (h2 : a → b) : a → c :=
by sorry

end syllogistic_reasoning_l115_115483


namespace defeated_candidate_percentage_l115_115780

noncomputable def percentage_defeated_candidate (total_votes diff_votes invalid_votes : ℕ) : ℕ :=
  let valid_votes := total_votes - invalid_votes
  let P := 100 * (valid_votes - diff_votes) / (2 * valid_votes)
  P

theorem defeated_candidate_percentage (total_votes : ℕ) (diff_votes : ℕ) (invalid_votes : ℕ) :
  total_votes = 12600 ∧ diff_votes = 5000 ∧ invalid_votes = 100 → percentage_defeated_candidate total_votes diff_votes invalid_votes = 30 :=
by
  intros
  sorry

end defeated_candidate_percentage_l115_115780


namespace least_number_subtracted_l115_115242

theorem least_number_subtracted (a b c : ℕ) (h1 : a = 7) (h2 : b = 9) (h3 : c = 11) :
  ∃ x, 0 ≤ x ∧ x < 1398 ∧ (1398 - x) % a = 5 ∧ (1398 - x) % b = 5 ∧ (1398 - x) % c = 5 ∧ x = 22 :=
by {
  sorry
}

end least_number_subtracted_l115_115242


namespace fraction_addition_l115_115837

theorem fraction_addition (a b c d : ℚ) (ha : a = 2/5) (hb : b = 3/8) (hc : c = 31/40) :
  a + b = c :=
by
  rw [ha, hb, hc]
  -- The proof part is skipped here as per instructions
  sorry

end fraction_addition_l115_115837


namespace train_cross_bridge_in_56_seconds_l115_115642

noncomputable def train_pass_time (length_train length_bridge : ℝ) (speed_train_kmh : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_train_ms := speed_train_kmh * (1000 / 3600)
  total_distance / speed_train_ms

theorem train_cross_bridge_in_56_seconds :
  train_pass_time 560 140 45 = 56 :=
by
  -- The proof can be added here
  sorry

end train_cross_bridge_in_56_seconds_l115_115642


namespace mr_rainwater_chickens_l115_115903

theorem mr_rainwater_chickens :
  ∃ (Ch : ℕ), (∀ (C G : ℕ), C = 9 ∧ G = 4 * C ∧ G = 2 * Ch → Ch = 18) :=
by
  sorry

end mr_rainwater_chickens_l115_115903


namespace boat_stream_speed_l115_115024

theorem boat_stream_speed :
  ∀ (v : ℝ), (∀ (downstream_speed boat_speed : ℝ), boat_speed = 22 ∧ downstream_speed = 54/2 ∧ downstream_speed = boat_speed + v) -> v = 5 :=
by
  sorry

end boat_stream_speed_l115_115024


namespace number_of_trees_in_garden_l115_115126

def total_yard_length : ℕ := 600
def distance_between_trees : ℕ := 24
def tree_at_each_end : ℕ := 1

theorem number_of_trees_in_garden : (total_yard_length / distance_between_trees) + tree_at_each_end = 26 := by
  sorry

end number_of_trees_in_garden_l115_115126


namespace maximum_y_coordinate_l115_115825

variable (x y b : ℝ)

def hyperbola (x y b : ℝ) : Prop := (x^2) / 4 - (y^2) / b = 1

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def op_condition (x y b : ℝ) : Prop := (x^2 + y^2) = 4 + b

noncomputable def eccentricity (b : ℝ) : ℝ := (Real.sqrt (4 + b)) / 2

theorem maximum_y_coordinate (hb : b > 0) 
                            (h_ec : 1 < eccentricity b ∧ eccentricity b ≤ 2) 
                            (h_hyp : hyperbola x y b) 
                            (h_first : first_quadrant x y) 
                            (h_op : op_condition x y b) 
                            : y ≤ 3 :=
sorry

end maximum_y_coordinate_l115_115825


namespace doubling_n_constant_C_l115_115672

theorem doubling_n_constant_C (e n R r : ℝ) (h_pos_e : 0 < e) (h_pos_n : 0 < n) (h_pos_R : 0 < R) (h_pos_r : 0 < r)
  (C : ℝ) (hC : C = e^2 * n / (R + n * r^2)) :
  C = (2 * e^2 * n) / (R + 2 * n * r^2) := 
sorry

end doubling_n_constant_C_l115_115672


namespace quadratic_condition_not_necessary_and_sufficient_l115_115189

theorem quadratic_condition_not_necessary_and_sufficient (a b c : ℝ) :
  ¬((∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ (b^2 - 4 * a * c < 0)) :=
sorry

end quadratic_condition_not_necessary_and_sufficient_l115_115189


namespace marbles_jack_gave_l115_115031

-- Definitions based on conditions
def initial_marbles : ℕ := 22
def final_marbles : ℕ := 42

-- Theorem stating that the difference between final and initial marbles Josh collected is the marbles Jack gave
theorem marbles_jack_gave :
  final_marbles - initial_marbles = 20 :=
  sorry

end marbles_jack_gave_l115_115031


namespace xy_product_eq_two_l115_115523

theorem xy_product_eq_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 2 / x = y + 2 / y) : x * y = 2 := 
sorry

end xy_product_eq_two_l115_115523


namespace product_of_fractions_l115_115913

theorem product_of_fractions :
  (2 / 3) * (5 / 8) * (1 / 4) = 5 / 48 := by
  sorry

end product_of_fractions_l115_115913


namespace spaghetti_manicotti_ratio_l115_115669

-- Define the number of students who were surveyed and their preferences
def total_students := 800
def students_prefer_spaghetti := 320
def students_prefer_manicotti := 160

-- The ratio of students who prefer spaghetti to those who prefer manicotti is 2
theorem spaghetti_manicotti_ratio :
  students_prefer_spaghetti / students_prefer_manicotti = 2 :=
by
  sorry

end spaghetti_manicotti_ratio_l115_115669


namespace zach_babysitting_hours_l115_115039

theorem zach_babysitting_hours :
  ∀ (bike_cost weekly_allowance mowing_pay babysitting_rate saved_amount needed_additional_amount hours_needed : ℕ),
    bike_cost = 100 →
    weekly_allowance = 5 →
    mowing_pay = 10 →
    babysitting_rate = 7 →
    saved_amount = 65 →
    needed_additional_amount = 6 →
    saved_amount + weekly_allowance + mowing_pay + hours_needed * babysitting_rate = bike_cost - needed_additional_amount →
    hours_needed = 2 :=
by
  intros bike_cost weekly_allowance mowing_pay babysitting_rate saved_amount needed_additional_amount hours_needed
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end zach_babysitting_hours_l115_115039


namespace cube_volume_and_surface_area_l115_115992

theorem cube_volume_and_surface_area (e : ℕ) (h : 12 * e = 72) :
  (e^3 = 216) ∧ (6 * e^2 = 216) := by
  sorry

end cube_volume_and_surface_area_l115_115992


namespace diagonal_ratio_of_squares_l115_115470

theorem diagonal_ratio_of_squares (P d : ℝ) (h : ∃ s S, 4 * S = 4 * s * 4 ∧ P = 4 * s ∧ d = s * Real.sqrt 2) : 
    (∃ D, D = 4 * d) :=
by
  sorry

end diagonal_ratio_of_squares_l115_115470


namespace complement_union_l115_115775

open Set

variable (U : Set ℕ := {0, 1, 2, 3, 4}) (A : Set ℕ := {1, 2, 3}) (B : Set ℕ := {2, 4})

theorem complement_union (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) : 
  (U \ A ∪ B) = {0, 2, 4} :=
by
  sorry

end complement_union_l115_115775


namespace find_y_l115_115494

theorem find_y (x y : ℤ) (h₁ : x = 4) (h₂ : 3 * x + 2 * y = 30) : y = 9 := 
by
  sorry

end find_y_l115_115494


namespace find_a_l115_115286

theorem find_a (r s a : ℚ) (h1 : s^2 = 16) (h2 : 2 * r * s = 15) (h3 : a = r^2) : a = 225/64 := by
  sorry

end find_a_l115_115286


namespace A_minus_B_l115_115252

theorem A_minus_B (x y m n A B : ℤ) (hx : x > y) (hx1 : x + y = 7) (hx2 : x * y = 12)
                  (hm : m > n) (hm1 : m + n = 13) (hm2 : m^2 + n^2 = 97)
                  (hA : A = x - y) (hB : B = m - n) :
                  A - B = -4 := by
  sorry

end A_minus_B_l115_115252


namespace tan_15_degree_l115_115078

theorem tan_15_degree : 
  let a := 45 * (Real.pi / 180)
  let b := 30 * (Real.pi / 180)
  Real.tan (a - b) = 2 - Real.sqrt 3 :=
by
  sorry

end tan_15_degree_l115_115078


namespace pamphlet_cost_l115_115008

theorem pamphlet_cost (p : ℝ) 
  (h1 : 9 * p < 10)
  (h2 : 10 * p > 11) : p = 1.11 :=
sorry

end pamphlet_cost_l115_115008


namespace hockey_league_teams_l115_115192

theorem hockey_league_teams (n : ℕ) (h : (n * (n - 1) * 10) / 2 = 1710) : n = 19 :=
by {
  sorry
}

end hockey_league_teams_l115_115192


namespace number_of_morse_code_symbols_l115_115901

-- Define the number of sequences for different lengths
def sequences_of_length (n : Nat) : Nat :=
  2 ^ n

theorem number_of_morse_code_symbols : 
  (sequences_of_length 1) + (sequences_of_length 2) + (sequences_of_length 3) + (sequences_of_length 4) + (sequences_of_length 5) = 62 := by
  sorry

end number_of_morse_code_symbols_l115_115901


namespace Q_coordinates_l115_115615

structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def P : Point := ⟨0, 3⟩
def R : Point := ⟨5, 0⟩

def isRectangle (A B C D : Point) : Prop :=
  -- replace this with the actual implementation of rectangle properties
  sorry

theorem Q_coordinates :
  ∃ Q : Point, isRectangle O P Q R ∧ Q.x = 5 ∧ Q.y = 3 :=
by
  -- replace this with the actual proof
  sorry

end Q_coordinates_l115_115615


namespace max_vouchers_with_680_l115_115719

def spend_to_voucher (spent : ℕ) : ℕ := (spent / 100) * 20

theorem max_vouchers_with_680 : spend_to_voucher 680 = 160 := by
  sorry

end max_vouchers_with_680_l115_115719


namespace joan_spent_on_trucks_l115_115950

-- Define constants for the costs
def cost_cars : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def total_toys : ℝ := 25.62
def cost_trucks : ℝ := 25.62 - (14.88 + 4.88)

-- Statement to prove
theorem joan_spent_on_trucks : cost_trucks = 5.86 := by
  sorry

end joan_spent_on_trucks_l115_115950


namespace faster_speed_l115_115246

theorem faster_speed (D : ℝ) (v : ℝ) (h₁ : D = 33.333333333333336) 
                      (h₂ : 10 * (D + 20) = v * D) : v = 16 :=
by
  sorry

end faster_speed_l115_115246


namespace largest_four_digit_perfect_square_l115_115492

theorem largest_four_digit_perfect_square :
  ∃ (n : ℕ), n = 9261 ∧ (∃ k : ℕ, k * k = n) ∧ ∀ (m : ℕ), m < 10000 → (∃ x, x * x = m) → m ≤ n := 
by 
  sorry

end largest_four_digit_perfect_square_l115_115492


namespace sum_of_first_seven_terms_l115_115638

variable (a : ℕ → ℝ) -- a sequence of real numbers (can be adapted to other types if needed)

-- Given conditions
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a n = a 0 + n * d

def sum_of_three_terms (a : ℕ → ℝ) (sum : ℝ) : Prop :=
  a 2 + a 3 + a 4 = sum

-- Theorem to prove
theorem sum_of_first_seven_terms (a : ℕ → ℝ) (h1 : is_arithmetic_progression a) (h2 : sum_of_three_terms a 12) :
  (a 0) + (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) = 28 :=
sorry

end sum_of_first_seven_terms_l115_115638


namespace exists_linear_function_intersecting_negative_axes_l115_115942

theorem exists_linear_function_intersecting_negative_axes :
  ∃ (k b : ℝ), k < 0 ∧ b < 0 ∧ (∃ x, k * x + b = 0 ∧ x < 0) ∧ (k * 0 + b < 0) :=
by
  sorry

end exists_linear_function_intersecting_negative_axes_l115_115942


namespace final_result_is_four_l115_115844

theorem final_result_is_four (x : ℕ) (h1 : x = 208) (y : ℕ) (h2 : y = x / 2) (z : ℕ) (h3 : z = y - 100) : z = 4 :=
by {
  sorry
}

end final_result_is_four_l115_115844


namespace recycling_points_l115_115472

-- Define the statement
theorem recycling_points : 
  ∀ (C H L I : ℝ) (points_per_six_pounds : ℝ), 
  C = 28 → H = 4.5 → L = 3.25 → I = 8.75 → points_per_six_pounds = 1 / 6 →
  (⌊ C * points_per_six_pounds ⌋ + ⌊ I * points_per_six_pounds ⌋  + ⌊ H * points_per_six_pounds ⌋ + ⌊ L * points_per_six_pounds ⌋ = 5) :=
by
  intros C H L I pps hC hH hL hI hpps
  rw [hC, hH, hL, hI, hpps]
  simp
  sorry

end recycling_points_l115_115472


namespace total_truck_loads_l115_115367

-- Using definitions from conditions in (a)
def sand : ℝ := 0.16666666666666666
def dirt : ℝ := 0.3333333333333333
def cement : ℝ := 0.16666666666666666

-- The proof statement based on the correct answer in (b)
theorem total_truck_loads : sand + dirt + cement = 0.6666666666666666 := 
by
  sorry

end total_truck_loads_l115_115367


namespace projectile_first_reach_height_56_l115_115360

theorem projectile_first_reach_height_56 (t : ℝ) (h1 : ∀ t, y = -16 * t^2 + 60 * t) :
    (∃ t : ℝ, y = 56 ∧ t = 1.75 ∧ (∀ t', t' < 1.75 → y ≠ 56)) :=
by
  sorry

end projectile_first_reach_height_56_l115_115360


namespace largest_b_l115_115071

def max_b (a b c : ℕ) : ℕ := b -- Define max_b function which outputs b

theorem largest_b (a b c : ℕ)
  (h1 : a * b * c = 360)
  (h2 : 1 < c)
  (h3 : c < b)
  (h4 : b < a) :
  max_b a b c = 10 :=
sorry

end largest_b_l115_115071


namespace arithmetic_expression_evaluation_l115_115580

theorem arithmetic_expression_evaluation :
  1325 + (180 / 60) * 3 - 225 = 1109 :=
by
  sorry -- To be filled with the proof steps

end arithmetic_expression_evaluation_l115_115580


namespace sum_of_roots_eq_two_l115_115148

theorem sum_of_roots_eq_two {b x1 x2 : ℝ} 
  (h : x1 ^ 2 - 2 * x1 + b = 0) 
  (k : x2 ^ 2 - 2 * x2 + b = 0) 
  (neq : x1 ≠ x2) : 
  x1 + x2 = 2 := 
sorry

end sum_of_roots_eq_two_l115_115148


namespace probability_first_number_greater_l115_115496

noncomputable def probability_first_greater_second : ℚ :=
  let total_outcomes := 8 * 8
  let favorable_outcomes := 7 + 6 + 5 + 4 + 3 + 2 + 1
  favorable_outcomes / total_outcomes

theorem probability_first_number_greater :
  probability_first_greater_second = 7 / 16 :=
sorry

end probability_first_number_greater_l115_115496


namespace additional_payment_is_65_l115_115548

def installments (n : ℕ) : ℤ := 65
def first_payment : ℕ := 20
def first_amount : ℤ := 410
def remaining_payment (x : ℤ) : ℕ := 45
def remaining_amount (x : ℤ) : ℤ := 410 + x
def average_amount : ℤ := 455

-- Define the total amount paid using both methods
def total_amount (x : ℤ) : ℤ := (20 * 410) + (45 * (410 + x))
def total_average : ℤ := 65 * 455

theorem additional_payment_is_65 :
  total_amount 65 = total_average :=
sorry

end additional_payment_is_65_l115_115548


namespace gcd_of_g_y_l115_115968

def g (y : ℕ) : ℕ := (3 * y + 4) * (8 * y + 3) * (11 * y + 5) * (y + 11)

theorem gcd_of_g_y (y : ℕ) (hy : ∃ k, y = 30492 * k) : Nat.gcd (g y) y = 660 :=
by
  sorry

end gcd_of_g_y_l115_115968


namespace kevin_hops_7_times_l115_115690

noncomputable def distance_hopped_after_n_hops (n : ℕ) : ℚ :=
  4 * (1 - (3 / 4) ^ n)

theorem kevin_hops_7_times :
  distance_hopped_after_n_hops 7 = 7086 / 2048 := 
by
  sorry

end kevin_hops_7_times_l115_115690


namespace backyard_area_l115_115120

-- Definitions from conditions
def length : ℕ := 1000 / 25
def perimeter : ℕ := 1000 / 10
def width : ℕ := (perimeter - 2 * length) / 2

-- Theorem statement: Given the conditions, the area of the backyard is 400 square meters
theorem backyard_area : length * width = 400 :=
by 
  -- Sorry to skip the proof as instructed
  sorry

end backyard_area_l115_115120


namespace polynomial_solution_l115_115093

theorem polynomial_solution (P : Polynomial ℝ) (h1 : P.eval 0 = 0) (h2 : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) : 
  ∀ x : ℝ, P.eval x = x :=
by
  sorry

end polynomial_solution_l115_115093


namespace solve_system_of_equations_l115_115385

theorem solve_system_of_equations (x y : ℚ)
  (h1 : 15 * x + 24 * y = 18)
  (h2 : 24 * x + 15 * y = 63) :
  x = 46 / 13 ∧ y = -19 / 13 := 
sorry

end solve_system_of_equations_l115_115385


namespace passengers_got_off_l115_115305

theorem passengers_got_off :
  ∀ (initial_boarded new_boarded final_left got_off : ℕ),
    initial_boarded = 28 →
    new_boarded = 7 →
    final_left = 26 →
    got_off = initial_boarded + new_boarded - final_left →
    got_off = 9 :=
by
  intros initial_boarded new_boarded final_left got_off h_initial h_new h_final h_got_off
  rw [h_initial, h_new, h_final] at h_got_off
  exact h_got_off

end passengers_got_off_l115_115305


namespace walking_rate_ratio_l115_115572

theorem walking_rate_ratio (R R' : ℝ) (usual_time early_time : ℝ) (H1 : usual_time = 42) (H2 : early_time = 36) 
(H3 : R * usual_time = R' * early_time) : (R' / R = 7 / 6) :=
by
  -- proof to be completed
  sorry

end walking_rate_ratio_l115_115572


namespace inequality_of_four_numbers_l115_115277

theorem inequality_of_four_numbers 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a ≤ 3 * b) (h2 : b ≤ 3 * a) (h3 : a ≤ 3 * c)
  (h4 : c ≤ 3 * a) (h5 : a ≤ 3 * d) (h6 : d ≤ 3 * a)
  (h7 : b ≤ 3 * c) (h8 : c ≤ 3 * b) (h9 : b ≤ 3 * d)
  (h10 : d ≤ 3 * b) (h11 : c ≤ 3 * d) (h12 : d ≤ 3 * c) : 
  a^2 + b^2 + c^2 + d^2 < 2 * (ab + ac + ad + bc + bd + cd) :=
sorry

end inequality_of_four_numbers_l115_115277


namespace number_of_shares_is_25_l115_115799

def wife_weekly_savings := 100
def husband_monthly_savings := 225
def duration_months := 4
def cost_per_share := 50

def total_savings : ℕ :=
  (wife_weekly_savings * 4 * duration_months) + (husband_monthly_savings * duration_months)

def amount_invested := total_savings / 2

def number_of_shares := amount_invested / cost_per_share

theorem number_of_shares_is_25 : number_of_shares = 25 := by
  sorry

end number_of_shares_is_25_l115_115799


namespace arithmetic_problem_l115_115933

theorem arithmetic_problem : 90 + 5 * 12 / (180 / 3) = 91 := by
  sorry

end arithmetic_problem_l115_115933


namespace gcd_13924_32451_eq_one_l115_115050

-- Define the two given integers.
def x : ℕ := 13924
def y : ℕ := 32451

-- State and prove that the greatest common divisor of x and y is 1.
theorem gcd_13924_32451_eq_one : Nat.gcd x y = 1 := by
  sorry

end gcd_13924_32451_eq_one_l115_115050


namespace workshop_total_workers_l115_115892

noncomputable def average_salary_of_all (W : ℕ) : ℝ := 8000
noncomputable def average_salary_of_technicians : ℝ := 12000
noncomputable def average_salary_of_non_technicians : ℝ := 6000

theorem workshop_total_workers
    (W : ℕ)
    (T : ℕ := 7)
    (N : ℕ := W - T)
    (h1 : (T + N) = W)
    (h2 : average_salary_of_all W = 8000)
    (h3 : average_salary_of_technicians = 12000)
    (h4 : average_salary_of_non_technicians = 6000)
    (h5 : (7 * 12000) + (N * 6000) = (7 + N) * 8000) :
  W = 21 :=
by
  sorry


end workshop_total_workers_l115_115892


namespace trapezoid_perimeter_l115_115291

noncomputable def isosceles_trapezoid_perimeter (R : ℝ) (α : ℝ) (hα : α < π / 2) : ℝ :=
  8 * R / (Real.sin α)

theorem trapezoid_perimeter (R : ℝ) (α : ℝ) (hα : α < π / 2) :
  ∃ (P : ℝ), P = isosceles_trapezoid_perimeter R α hα := by
    sorry

end trapezoid_perimeter_l115_115291


namespace average_visitors_per_day_l115_115294

/-- A library has different visitor numbers depending on the day of the week.
  - On Sundays, the library has an average of 660 visitors.
  - On Mondays through Thursdays, there are 280 visitors on average.
  - Fridays and Saturdays see an increase to an average of 350 visitors.
  - This month has a special event on the third Saturday, bringing an extra 120 visitors that day.
  - The month has 30 days and begins with a Sunday.
  We want to calculate the average number of visitors per day for the entire month. -/
theorem average_visitors_per_day
  (num_days : ℕ) (starts_on_sunday : Bool)
  (sundays_visitors : ℕ) (weekdays_visitors : ℕ) (weekend_visitors : ℕ)
  (special_event_extra_visitors : ℕ) (sundays : ℕ) (mondays : ℕ)
  (tuesdays : ℕ) (wednesdays : ℕ) (thursdays : ℕ) (fridays : ℕ)
  (saturdays : ℕ) :
  num_days = 30 → starts_on_sunday = true →
  sundays_visitors = 660 → weekdays_visitors = 280 → weekend_visitors = 350 →
  special_event_extra_visitors = 120 →
  sundays = 4 → mondays = 5 →
  tuesdays = 4 → wednesdays = 4 → thursdays = 4 → fridays = 4 → saturdays = 4 →
  ((sundays * sundays_visitors +
    mondays * weekdays_visitors +
    tuesdays * weekdays_visitors +
    wednesdays * weekdays_visitors +
    thursdays * weekdays_visitors +
    fridays * weekend_visitors +
    saturdays * weekend_visitors +
    special_event_extra_visitors) / num_days = 344) :=
by
  intros
  sorry

end average_visitors_per_day_l115_115294


namespace det_matrix_4x4_l115_115198

def matrix_4x4 : Matrix (Fin 4) (Fin 4) ℤ :=
  ![
    ![3, 0, 2, 0],
    ![2, 3, -1, 4],
    ![0, 4, -2, 3],
    ![5, 2, 0, 1]
  ]

theorem det_matrix_4x4 : Matrix.det matrix_4x4 = -84 :=
by
  sorry

end det_matrix_4x4_l115_115198


namespace sum_of_two_numbers_l115_115543

-- Define the two numbers and conditions
variables {x y : ℝ}
axiom prod_eq : x * y = 120
axiom sum_squares_eq : x^2 + y^2 = 289

-- The statement we want to prove
theorem sum_of_two_numbers (x y : ℝ) (prod_eq : x * y = 120) (sum_squares_eq : x^2 + y^2 = 289) : x + y = 23 :=
sorry

end sum_of_two_numbers_l115_115543


namespace min_shirts_to_save_l115_115502

theorem min_shirts_to_save (x : ℕ) :
  (75 + 10 * x < if x < 30 then 15 * x else 14 * x) → x = 20 :=
by
  sorry

end min_shirts_to_save_l115_115502


namespace num_distinguishable_octahedrons_l115_115404

-- Define the given conditions
def num_faces : ℕ := 8
def num_colors : ℕ := 8
def total_permutations : ℕ := Nat.factorial num_colors
def distinct_orientations : ℕ := 24

-- Prove the main statement
theorem num_distinguishable_octahedrons : total_permutations / distinct_orientations = 1680 :=
by
  sorry

end num_distinguishable_octahedrons_l115_115404


namespace max_african_team_wins_max_l115_115354

-- Assume there are n African teams and (n + 9) European teams.
-- Each pair of teams plays exactly once.
-- European teams won nine times as many matches as African teams.
-- Prove that the maximum number of matches that a single African team might have won is 11.

theorem max_african_team_wins_max (n : ℕ) (k : ℕ) (n_african_wins : ℕ) (n_european_wins : ℕ)
  (h1 : n_african_wins = (n * (n - 1)) / 2) 
  (h2 : n_european_wins = ((n + 9) * (n + 8)) / 2 + k)
  (h3 : n_european_wins = 9 * (n_african_wins + (n * (n + 9) - k))) :
  ∃ max_wins, max_wins = 11 := by
  sorry

end max_african_team_wins_max_l115_115354


namespace arithmetic_identity_l115_115845

theorem arithmetic_identity :
  65 * 1515 - 25 * 1515 + 1515 = 62115 :=
by
  sorry

end arithmetic_identity_l115_115845


namespace original_price_of_article_l115_115153

theorem original_price_of_article (x : ℝ) (h : 0.80 * x = 620) : x = 775 := 
by 
  sorry

end original_price_of_article_l115_115153


namespace simplify_expression_l115_115450

theorem simplify_expression (a b c : ℝ) : a - (a - b + c) = b - c :=
by sorry

end simplify_expression_l115_115450


namespace find_a_l115_115731

noncomputable def ab (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a {a : ℝ} : ab a 6 = -3 → a = 23 :=
by
  sorry

end find_a_l115_115731


namespace katy_summer_reading_total_l115_115490

def katy_books_in_summer (june_books july_books august_books : ℕ) : ℕ := june_books + july_books + august_books

theorem katy_summer_reading_total (june_books : ℕ) (july_books : ℕ) (august_books : ℕ) 
  (h1 : june_books = 8)
  (h2 : july_books = 2 * june_books)
  (h3 : august_books = july_books - 3) :
  katy_books_in_summer june_books july_books august_books = 37 :=
by
  sorry

end katy_summer_reading_total_l115_115490


namespace eval_expr_l115_115676

def a := -1
def b := 1 / 7
def expr := (3 * a^3 - 2 * a * b + b^2) - 2 * (-a^3 - a * b + 4 * b^2)

theorem eval_expr : expr = -36 / 7 := by
  -- Inserting the proof using the original mathematical solution steps is not required here.
  sorry

end eval_expr_l115_115676


namespace problem1_problem2_problem3_problem4_l115_115152

theorem problem1 : 
  (3 / 5 : ℚ) - ((2 / 15) + (1 / 3)) = (2 / 15) := 
  by 
  sorry

theorem problem2 : 
  (-2 : ℤ) - 12 * ((1 / 3 : ℚ) - (1 / 4 : ℚ) + (1 / 2 : ℚ)) = -8 := 
  by 
  sorry

theorem problem3 : 
  (2 : ℤ) * (-3) ^ 2 - (6 / (-2) : ℚ) * (-1 / 3) = 17 := 
  by 
  sorry

theorem problem4 : 
  (-1 ^ 4 : ℤ) + ((abs (2 ^ 3 - 10)) : ℤ) - ((-3 : ℤ) / (-1) ^ 2019) = -2 := 
  by 
  sorry

end problem1_problem2_problem3_problem4_l115_115152


namespace smallest_k_for_factorial_divisibility_l115_115393

theorem smallest_k_for_factorial_divisibility : 
  ∃ (k : ℕ), (∀ n : ℕ, n < k → ¬(2040 ∣ n!)) ∧ (2040 ∣ k!) ∧ k = 17 :=
by
  -- We skip the actual proof steps and provide a placeholder for the proof
  sorry

end smallest_k_for_factorial_divisibility_l115_115393


namespace arithmetic_and_geometric_sequence_l115_115865

-- Definitions based on given conditions
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

-- Main statement to prove
theorem arithmetic_and_geometric_sequence :
  ∀ (x y a b c : ℝ), 
  is_arithmetic_sequence 1 x y 4 →
  is_geometric_sequence (-2) a b c (-8) →
  (y - x) / b = -1 / 4 :=
by
  sorry

end arithmetic_and_geometric_sequence_l115_115865


namespace expression_value_l115_115757

-- Define the problem statement
theorem expression_value (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : (x + y) / z = (y + z) / x) (h5 : (y + z) / x = (z + x) / y) :
  ∃ k : ℝ, k = 8 ∨ k = -1 := 
sorry

end expression_value_l115_115757


namespace heidi_zoe_paint_fraction_l115_115244

theorem heidi_zoe_paint_fraction (H_period : ℝ) (HZ_period : ℝ) :
  (H_period = 60 → HZ_period = 40 → (8 / 40) = (1 / 5)) :=
by intros H_period_eq HZ_period_eq
   sorry

end heidi_zoe_paint_fraction_l115_115244


namespace symmetric_point_x_axis_l115_115318

theorem symmetric_point_x_axis (x y z : ℝ) : 
    (x, -y, -z) = (-2, -1, -9) :=
by 
  sorry

end symmetric_point_x_axis_l115_115318


namespace jeff_total_cabinets_l115_115323

def initial_cabinets : ℕ := 3
def cabinets_per_counter : ℕ := 2 * initial_cabinets
def total_cabinets_installed : ℕ := 3 * cabinets_per_counter + 5
def total_cabinets (initial : ℕ) (installed : ℕ) : ℕ := initial + installed

theorem jeff_total_cabinets : total_cabinets initial_cabinets total_cabinets_installed = 26 :=
by
  sorry

end jeff_total_cabinets_l115_115323


namespace find_alpha_l115_115392

variable (α β k : ℝ)

axiom h1 : α * β = k
axiom h2 : α = -4
axiom h3 : β = -8
axiom k_val : k = 32
axiom β_val : β = 12

theorem find_alpha (h1 : α * β = k) (h2 : α = -4) (h3 : β = -8) (k_val : k = 32) (β_val : β = 12) :
  α = 8 / 3 :=
sorry

end find_alpha_l115_115392


namespace correct_equation_l115_115853

def initial_count_A : ℕ := 54
def initial_count_B : ℕ := 48
def new_count_A (x : ℕ) : ℕ := initial_count_A + x
def new_count_B (x : ℕ) : ℕ := initial_count_B - x

theorem correct_equation (x : ℕ) : new_count_A x = 2 * new_count_B x := 
sorry

end correct_equation_l115_115853


namespace intersection_with_x_axis_l115_115555

theorem intersection_with_x_axis (a : ℝ) (h : 2 * a - 4 = 0) : a = 2 := by
  sorry

end intersection_with_x_axis_l115_115555


namespace quadratic_root_relationship_l115_115439

theorem quadratic_root_relationship
  (m1 m2 : ℝ)
  (x1 x2 x3 x4 : ℝ)
  (h_eq1 : m1 * x1^2 + (1 / 3) * x1 + 1 = 0)
  (h_eq2 : m1 * x2^2 + (1 / 3) * x2 + 1 = 0)
  (h_eq3 : m2 * x3^2 + (1 / 3) * x3 + 1 = 0)
  (h_eq4 : m2 * x4^2 + (1 / 3) * x4 + 1 = 0)
  (h_order : x1 < x3 ∧ x3 < x4 ∧ x4 < x2 ∧ x2 < 0) :
  m2 > m1 ∧ m1 > 0 :=
sorry

end quadratic_root_relationship_l115_115439


namespace avg_salary_increase_l115_115791

def initial_avg_salary : ℝ := 1700
def num_employees : ℕ := 20
def manager_salary : ℝ := 3800

theorem avg_salary_increase :
  ((num_employees * initial_avg_salary + manager_salary) / (num_employees + 1)) - initial_avg_salary = 100 :=
by
  sorry

end avg_salary_increase_l115_115791


namespace fraction_of_number_l115_115342

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end fraction_of_number_l115_115342


namespace smallest_spherical_triangle_angle_l115_115261

-- Define the conditions
def is_ratio (a b c : ℕ) : Prop := a = 4 ∧ b = 5 ∧ c = 6
def sum_of_angles (α β γ : ℕ) : Prop := α + β + γ = 270

-- Define the problem statement
theorem smallest_spherical_triangle_angle 
  (a b c α β γ : ℕ)
  (h1 : is_ratio a b c)
  (h2 : sum_of_angles (a * α) (b * β) (c * γ)) :
  a * α = 72 := 
sorry

end smallest_spherical_triangle_angle_l115_115261


namespace damage_in_usd_correct_l115_115944

def exchange_rate := (125 : ℚ) / 100
def damage_CAD := 45000000
def damage_USD := damage_CAD / exchange_rate

theorem damage_in_usd_correct (CAD_to_USD : exchange_rate = (125 : ℚ) / 100) (damage_in_cad : damage_CAD = 45000000) : 
  damage_USD = 36000000 :=
by
  sorry

end damage_in_usd_correct_l115_115944


namespace books_assigned_total_l115_115366

-- Definitions for the conditions.
def Mcgregor_books := 34
def Floyd_books := 32
def remaining_books := 23

-- The total number of books assigned.
def total_books := Mcgregor_books + Floyd_books + remaining_books

-- The theorem that needs to be proven.
theorem books_assigned_total : total_books = 89 :=
by
  sorry

end books_assigned_total_l115_115366


namespace remainder_div_l115_115782

theorem remainder_div (N : ℤ) (k : ℤ) (h : N = 35 * k + 25) : N % 15 = 10 := by
  sorry

end remainder_div_l115_115782


namespace remainder_of_number_divisor_l115_115441

-- Define the interesting number and the divisor
def number := 2519
def divisor := 9
def expected_remainder := 8

-- State the theorem to prove the remainder condition
theorem remainder_of_number_divisor :
  number % divisor = expected_remainder := by
  sorry

end remainder_of_number_divisor_l115_115441


namespace kaleb_tickets_l115_115536

variable (T : Nat)
variable (tickets_left : Nat) (ticket_cost : Nat) (total_spent : Nat)

theorem kaleb_tickets : tickets_left = 3 → ticket_cost = 9 → total_spent = 27 → T = 6 :=
by
  sorry

end kaleb_tickets_l115_115536


namespace coin_flip_sequences_l115_115063

theorem coin_flip_sequences : 
  let flips := 10
  let choices := 2
  let total_sequences := choices ^ flips
  total_sequences = 1024 :=
by
  sorry

end coin_flip_sequences_l115_115063


namespace triangle_ABC_problem_l115_115545

noncomputable def perimeter_of_triangle (a b c : ℝ) : ℝ := a + b + c

theorem triangle_ABC_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : a = 3) 
  (h2 : B = π / 3) 
  (area : ℝ)
  (h3 : (1/2) * a * c * Real.sin B = 6 * Real.sqrt 3) :

  perimeter_of_triangle a b c = 18 ∧ 
  Real.sin (2 * A) = 39 * Real.sqrt 3 / 98 := 
by 
  sorry

end triangle_ABC_problem_l115_115545


namespace right_triangle_x_value_l115_115916

theorem right_triangle_x_value (x Δ : ℕ) (h₁ : x > 0) (h₂ : Δ > 0) :
  ((x + 2 * Δ)^2 = x^2 + (x + Δ)^2) → 
  x = (Δ * (-1 + 2 * Real.sqrt 7)) / 2 := 
sorry

end right_triangle_x_value_l115_115916


namespace sequence_first_term_l115_115850

theorem sequence_first_term (a : ℕ → ℤ) 
  (h1 : a 3 = 5) 
  (h2 : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n - 1) : 
  a 1 = 2 := 
sorry

end sequence_first_term_l115_115850


namespace probability_both_in_picture_l115_115747

-- Define the conditions
def completes_lap (laps_time: ℕ) (time: ℕ) : ℕ := time / laps_time

def position_into_lap (laps_time: ℕ) (time: ℕ) : ℕ := time % laps_time

-- Define the positions of Rachel and Robert
def rachel_position (time: ℕ) : ℚ :=
  let rachel_lap_time := 100
  let laps_completed := completes_lap rachel_lap_time time
  let time_into_lap := position_into_lap rachel_lap_time time
  (laps_completed : ℚ) + (time_into_lap : ℚ) / rachel_lap_time

def robert_position (time: ℕ) : ℚ :=
  let robert_lap_time := 70
  let laps_completed := completes_lap robert_lap_time time
  let time_into_lap := position_into_lap robert_lap_time time
  (laps_completed : ℚ) + (time_into_lap : ℚ) / robert_lap_time

-- Define the probability that both are in the picture
theorem probability_both_in_picture :
  let rachel_lap_time := 100
  let robert_lap_time := 70
  let start_time := 720
  let end_time := 780
  ∃ (overlap_time: ℚ) (total_time: ℚ),
    overlap_time / total_time = 1 / 16 :=
sorry

end probability_both_in_picture_l115_115747


namespace max_q_minus_r_839_l115_115263

theorem max_q_minus_r_839 : ∃ (q r : ℕ), (839 = 19 * q + r) ∧ (0 ≤ r ∧ r < 19) ∧ q - r = 41 :=
by
  sorry

end max_q_minus_r_839_l115_115263


namespace space_diagonals_Q_l115_115479

-- Definitions based on the conditions
def vertices (Q : Type) : ℕ := 30
def edges (Q : Type) : ℕ := 70
def faces (Q : Type) : ℕ := 40
def triangular_faces (Q : Type) : ℕ := 20
def quadrilateral_faces (Q : Type) : ℕ := 15
def pentagon_faces (Q : Type) : ℕ := 5

-- Problem Statement
theorem space_diagonals_Q :
  ∀ (Q : Type),
  vertices Q = 30 →
  edges Q = 70 →
  faces Q = 40 →
  triangular_faces Q = 20 →
  quadrilateral_faces Q = 15 →
  pentagon_faces Q = 5 →
  ∃ d : ℕ, d = 310 := 
by
  -- At this point only the structure of the proof is set up.
  sorry

end space_diagonals_Q_l115_115479


namespace sequence_monotonic_and_bounded_l115_115503

theorem sequence_monotonic_and_bounded :
  ∀ (a : ℕ → ℝ), (a 1 = 1 / 2) → (∀ n, a (n + 1) = 1 / 2 + (a n)^2 / 2) →
    (∀ n, a n < 2) ∧ (∀ n, a n < a (n + 1)) :=
by
  sorry

end sequence_monotonic_and_bounded_l115_115503


namespace stutterer_square_number_unique_l115_115912

-- Definitions based on problem conditions
def is_stutterer (n : ℕ) : Prop :=
  (1000 ≤ n ∧ n < 10000) ∧ (n / 100 = (n % 1000) / 100) ∧ ((n % 1000) % 100 = n % 10 * 10 + n % 10)

def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- The theorem statement
theorem stutterer_square_number_unique : ∃ n, is_stutterer n ∧ is_square n ∧ n = 7744 :=
by
  sorry

end stutterer_square_number_unique_l115_115912


namespace functional_equation_solution_l115_115609

theorem functional_equation_solution :
  ∀ (f : ℤ → ℤ), (∀ (m n : ℤ), f (m + f (f n)) = -f (f (m + 1)) - n) → (∀ (p : ℤ), f p = 1 - p) :=
by
  intro f h
  sorry

end functional_equation_solution_l115_115609


namespace find_valid_pairs_l115_115561

def satisfies_condition (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ (a ^ 2017 + b) % (a * b) = 0

theorem find_valid_pairs : 
  ∀ (a b : ℕ), satisfies_condition a b → (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 2 ^ 2017) := 
by
  sorry

end find_valid_pairs_l115_115561


namespace molecular_weight_2N_5O_l115_115037

def molecular_weight (num_N num_O : ℕ) (atomic_weight_N atomic_weight_O : ℝ) : ℝ :=
  (num_N * atomic_weight_N) + (num_O * atomic_weight_O)

theorem molecular_weight_2N_5O :
  molecular_weight 2 5 14.01 16.00 = 108.02 :=
by
  -- proof goes here
  sorry

end molecular_weight_2N_5O_l115_115037


namespace prove_p_or_q_l115_115877

-- Define propositions p and q
def p : Prop := ∃ n : ℕ, 0 = 2 * n
def q : Prop := ∃ m : ℕ, 3 = 2 * m

-- The Lean statement to prove
theorem prove_p_or_q : p ∨ q := by
  sorry

end prove_p_or_q_l115_115877


namespace chess_club_boys_count_l115_115212

theorem chess_club_boys_count (B G : ℕ) 
  (h1 : B + G = 30)
  (h2 : (2/3 : ℝ) * G + B = 18) : 
  B = 6 :=
by
  sorry

end chess_club_boys_count_l115_115212


namespace smallest_n_for_three_nested_rectangles_l115_115398

/-- Rectangle represented by its side lengths -/
structure Rectangle where
  x : ℕ
  y : ℕ
  h1 : 1 ≤ x
  h2 : x ≤ y
  h3 : y ≤ 100

/-- Define the nesting relation between rectangles -/
def nested (R1 R2 : Rectangle) : Prop :=
  R1.x < R2.x ∧ R1.y < R2.y

/-- Prove the smallest n such that there exist 3 nested rectangles out of n rectangles where n = 101 -/
theorem smallest_n_for_three_nested_rectangles (n : ℕ) (h : n ≥ 101) :
  ∀ (rectangles : Fin n → Rectangle), 
    ∃ (R1 R2 R3 : Fin n), nested (rectangles R1) (rectangles R2) ∧ nested (rectangles R2) (rectangles R3) :=
  sorry

end smallest_n_for_three_nested_rectangles_l115_115398


namespace rachelle_meat_needed_l115_115832

-- Define the ratio of meat per hamburger
def meat_per_hamburger (pounds : ℕ) (hamburgers : ℕ) : ℚ :=
  pounds / hamburgers

-- Define the total meat needed for a given number of hamburgers
def total_meat (meat_per_hamburger : ℚ) (hamburgers : ℕ) : ℚ :=
  meat_per_hamburger * hamburgers

-- Prove that Rachelle needs 15 pounds of meat to make 36 hamburgers
theorem rachelle_meat_needed : total_meat (meat_per_hamburger 5 12) 36 = 15 := by
  sorry

end rachelle_meat_needed_l115_115832


namespace opposite_neg_two_l115_115697

theorem opposite_neg_two : -(-2) = 2 := by
  sorry

end opposite_neg_two_l115_115697


namespace real_roots_approx_correct_to_4_decimal_places_l115_115032

noncomputable def f (x : ℝ) : ℝ := x^4 - (2 * 10^10 + 1) * x^2 - x + 10^20 + 10^10 - 1

theorem real_roots_approx_correct_to_4_decimal_places :
  ∃ x1 x2 : ℝ, 
  abs (x1 - 99999.9997) ≤ 0.0001 ∧ 
  abs (x2 - 100000.0003) ≤ 0.0001 ∧ 
  f x1 = 0 ∧ 
  f x2 = 0 :=
sorry

end real_roots_approx_correct_to_4_decimal_places_l115_115032


namespace baseball_cards_start_count_l115_115681

theorem baseball_cards_start_count (X : ℝ) 
  (h1 : ∃ (x : ℝ), x = (X + 1) / 2)
  (h2 : ∃ (x' : ℝ), x' = X - ((X + 1) / 2) - 1)
  (h3 : ∃ (y : ℝ), y = 3 * (X - ((X + 1) / 2) - 1))
  (h4 : ∃ (z : ℝ), z = 18) : 
  X = 15 :=
by
  sorry

end baseball_cards_start_count_l115_115681


namespace find_c_minus_a_l115_115935

theorem find_c_minus_a (a b c : ℝ) (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 50) : c - a = 10 :=
sorry

end find_c_minus_a_l115_115935


namespace butterflies_equal_distribution_l115_115164

theorem butterflies_equal_distribution (N : ℕ) : (∃ t : ℕ, 
    (N - t) % 8 = 0 ∧ (N - t) / 8 > 0) ↔ ∃ k : ℕ, N = 45 * k :=
by sorry

end butterflies_equal_distribution_l115_115164


namespace percentage_x_minus_y_l115_115353

variable (x y : ℝ)

theorem percentage_x_minus_y (P : ℝ) :
  P / 100 * (x - y) = 20 / 100 * (x + y) ∧ y = 20 / 100 * x → P = 30 :=
by
  intros h
  sorry

end percentage_x_minus_y_l115_115353


namespace store_owner_oil_l115_115387

noncomputable def liters_of_oil (volume_per_bottle : ℕ) (number_of_bottles : ℕ) : ℕ :=
  (volume_per_bottle * number_of_bottles) / 1000

theorem store_owner_oil : liters_of_oil 200 20 = 4 := by
  sorry

end store_owner_oil_l115_115387


namespace kids_difference_l115_115375

def kidsPlayedOnMonday : Nat := 11
def kidsPlayedOnTuesday : Nat := 12

theorem kids_difference :
  kidsPlayedOnTuesday - kidsPlayedOnMonday = 1 := by
  sorry

end kids_difference_l115_115375


namespace no_three_digit_number_exists_l115_115069

theorem no_three_digit_number_exists (a b c : ℕ) (h₁ : 0 ≤ a ∧ a < 10) (h₂ : 0 ≤ b ∧ b < 10) (h₃ : 0 ≤ c ∧ c < 10) (h₄ : a ≠ 0) :
  ¬ ∃ k : ℕ, k^2 = 99 * (a - c) :=
by
  sorry

end no_three_digit_number_exists_l115_115069


namespace NumberOfStudentsEnrolledOnlyInEnglish_l115_115584

-- Definition of the problem's variables and conditions
variables (TotalStudents BothEnglishAndGerman TotalGerman OnlyEnglish OnlyGerman : ℕ)
variables (h1 : TotalStudents = 52)
variables (h2 : BothEnglishAndGerman = 12)
variables (h3 : TotalGerman = 22)
variables (h4 : TotalStudents = OnlyEnglish + OnlyGerman + BothEnglishAndGerman)
variables (h5 : OnlyGerman = TotalGerman - BothEnglishAndGerman)

-- Theorem to prove the number of students enrolled only in English
theorem NumberOfStudentsEnrolledOnlyInEnglish : OnlyEnglish = 30 :=
by
  -- Insert the necessary proof steps here to derive the number of students enrolled only in English from the given conditions
  sorry

end NumberOfStudentsEnrolledOnlyInEnglish_l115_115584


namespace pipes_height_l115_115925

theorem pipes_height (d : ℝ) (h : ℝ) (r : ℝ) (s : ℝ)
  (hd : d = 12)
  (hs : s = d)
  (hr : r = d / 2)
  (heq : h = 6 * Real.sqrt 3 + r) :
  h = 6 * Real.sqrt 3 + 6 :=
by
  sorry

end pipes_height_l115_115925


namespace division_problem_l115_115026

theorem division_problem (x y n : ℕ) 
  (h1 : x = n * y + 4) 
  (h2 : 2 * x = 14 * y + 1) 
  (h3 : 5 * y - x = 3) : n = 4 := 
sorry

end division_problem_l115_115026


namespace balls_in_jar_l115_115474

theorem balls_in_jar (total_balls initial_blue_balls balls_after_taking_out : ℕ) (probability_blue : ℚ) :
  initial_blue_balls = 6 →
  balls_after_taking_out = initial_blue_balls - 3 →
  probability_blue = 1 / 5 →
  (balls_after_taking_out : ℚ) / (total_balls - 3 : ℚ) = probability_blue →
  total_balls = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end balls_in_jar_l115_115474


namespace actual_speed_of_valentin_l115_115356

theorem actual_speed_of_valentin
  (claimed_speed : ℕ := 50) -- Claimed speed in m/min
  (wrong_meter : ℕ := 60)   -- Valentin thought 1 meter = 60 cm
  (wrong_minute : ℕ := 100) -- Valentin thought 1 minute = 100 seconds
  (correct_speed : ℕ := 18) -- The actual speed in m/min
  : (claimed_speed * wrong_meter / wrong_minute) * 60 / 100 = correct_speed :=
by
  sorry

end actual_speed_of_valentin_l115_115356


namespace total_vegetables_l115_115702

-- Definitions for the conditions in the problem
def cucumbers := 58
def carrots := cucumbers - 24
def tomatoes := cucumbers + 49
def radishes := carrots

-- Statement for the proof problem
theorem total_vegetables :
  cucumbers + carrots + tomatoes + radishes = 233 :=
by sorry

end total_vegetables_l115_115702


namespace bananas_count_l115_115888

/-- Elias bought some bananas and ate 1 of them. 
    After eating, he has 11 bananas left.
    Prove that Elias originally bought 12 bananas. -/
theorem bananas_count (x : ℕ) (h1 : x - 1 = 11) : x = 12 := by
  sorry

end bananas_count_l115_115888


namespace nacho_will_be_three_times_older_in_future_l115_115420

variable (N D x : ℕ)
variable (h1 : D = 5)
variable (h2 : N + D = 40)
variable (h3 : N + x = 3 * (D + x))

theorem nacho_will_be_three_times_older_in_future :
  x = 10 :=
by {
  -- Given conditions
  sorry
}

end nacho_will_be_three_times_older_in_future_l115_115420


namespace production_difference_correct_l115_115712

variable (w t M T : ℕ)

-- Condition: w = 2t
def condition_w := w = 2 * t

-- Widgets produced on Monday
def widgets_monday := M = w * t

-- Widgets produced on Tuesday
def widgets_tuesday := T = (w + 5) * (t - 3)

-- Difference in production
def production_difference := M - T = t + 15

theorem production_difference_correct
  (h1 : condition_w w t)
  (h2 : widgets_monday M w t)
  (h3 : widgets_tuesday T w t) :
  production_difference M T t :=
sorry

end production_difference_correct_l115_115712


namespace sequence_properties_l115_115197

noncomputable def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
a1 + d * (n - 1)

theorem sequence_properties (d a1 : ℤ) (h_d_ne_zero : d ≠ 0)
(h1 : arithmetic_sequence a1 d 2 + arithmetic_sequence a1 d 4 = 10)
(h2 : (arithmetic_sequence a1 d 2)^2 = (arithmetic_sequence a1 d 1) * (arithmetic_sequence a1 d 5)) :
a1 = 1 ∧ ∀ n : ℕ, n > 0 → arithmetic_sequence 1 2 n = 2 * n - 1 :=
by
  sorry

end sequence_properties_l115_115197


namespace clover_walk_distance_l115_115919

theorem clover_walk_distance (total_distance days walks_per_day : ℝ) (h1 : total_distance = 90) (h2 : days = 30) (h3 : walks_per_day = 2) :
  (total_distance / days / walks_per_day = 1.5) :=
by
  sorry

end clover_walk_distance_l115_115919


namespace no_natural_number_exists_l115_115667

theorem no_natural_number_exists 
  (n : ℕ) : ¬ ∃ x y : ℕ, (2 * n * (n + 1) * (n + 2) * (n + 3) + 12) = x^2 + y^2 := 
by sorry

end no_natural_number_exists_l115_115667


namespace total_metal_rods_needed_l115_115332

def metal_rods_per_sheet : ℕ := 10
def sheets_per_panel : ℕ := 3
def metal_rods_per_beam : ℕ := 4
def beams_per_panel : ℕ := 2
def panels : ℕ := 10

theorem total_metal_rods_needed : 
  (sheets_per_panel * metal_rods_per_sheet + beams_per_panel * metal_rods_per_beam) * panels = 380 :=
by
  exact rfl

end total_metal_rods_needed_l115_115332


namespace b_should_pay_l115_115620

-- Definitions for the number of horses and their duration in months
def horses_of_a := 12
def months_of_a := 8

def horses_of_b := 16
def months_of_b := 9

def horses_of_c := 18
def months_of_c := 6

-- Total rent
def total_rent := 870

-- Shares in horse-months for each person
def share_of_a := horses_of_a * months_of_a
def share_of_b := horses_of_b * months_of_b
def share_of_c := horses_of_c * months_of_c

-- Total share in horse-months
def total_share := share_of_a + share_of_b + share_of_c

-- Fraction for b
def fraction_for_b := share_of_b / total_share

-- Amount b should pay
def amount_for_b := total_rent * fraction_for_b

-- Theorem to verify the amount b should pay
theorem b_should_pay : amount_for_b = 360 := by
  -- The steps of the proof would go here
  sorry

end b_should_pay_l115_115620


namespace inequality_proof_l115_115781

theorem inequality_proof (a b c : ℝ) (hab : a > b) : a * |c| ≥ b * |c| := by
  sorry

end inequality_proof_l115_115781


namespace range_of_m_l115_115860

theorem range_of_m (α β m : ℝ) (hαβ : 0 < α ∧ α < 1 ∧ 1 < β ∧ β < 2)
  (h_eq : ∀ x, x^2 - 2*(m-1)*x + (m-1) = 0 ↔ (x = α ∨ x = β)) :
  2 < m ∧ m < 7 / 3 := by
  sorry

end range_of_m_l115_115860


namespace inequality_transform_l115_115215

theorem inequality_transform {a b c d e : ℝ} (hab : a > b) (hb0 : b > 0) 
  (hcd : c < d) (hd0 : d < 0) (he : e < 0) : 
  e / (a - c)^2 > e / (b - d)^2 :=
by 
  sorry

end inequality_transform_l115_115215


namespace exists_c_gt_zero_l115_115754

theorem exists_c_gt_zero (a b : ℝ) (h : a < b) : ∃ c > 0, a < b + c := 
sorry

end exists_c_gt_zero_l115_115754


namespace intersection_M_N_l115_115662

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | ∃ y ∈ M, |y| = x}

-- The main theorem to prove M ∩ N = {0, 1, 2}
theorem intersection_M_N : M ∩ N = {0, 1, 2} :=
by
  sorry

end intersection_M_N_l115_115662


namespace determinant_example_l115_115335

def det_2x2 (a b c d : ℤ) : ℤ := a * d - b * c

theorem determinant_example : det_2x2 7 (-2) (-3) 6 = 36 := by
  sorry

end determinant_example_l115_115335


namespace picnic_adults_children_difference_l115_115202

theorem picnic_adults_children_difference :
  ∃ (M W A C : ℕ),
    (M = 65) ∧
    (M = W + 20) ∧
    (A = M + W) ∧
    (C = 200 - A) ∧
    ((A - C) = 20) :=
by
  sorry

end picnic_adults_children_difference_l115_115202


namespace ratio_of_pond_to_field_area_l115_115511

theorem ratio_of_pond_to_field_area
  (l w : ℕ)
  (field_area pond_area : ℕ)
  (h1 : l = 2 * w)
  (h2 : l = 36)
  (h3 : pond_area = 9 * 9)
  (field_area_def : field_area = l * w)
  (pond_area_def : pond_area = 81) :
  pond_area / field_area = 1 / 8 := 
sorry

end ratio_of_pond_to_field_area_l115_115511


namespace balls_picked_at_random_eq_two_l115_115953

-- Define the initial conditions: number of balls of each color
def num_red_balls : ℕ := 5
def num_blue_balls : ℕ := 4
def num_green_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_blue_balls + num_green_balls

-- Define the given probability
def given_probability : ℚ := 0.15151515151515152

-- Define the probability calculation for picking two red balls
def probability_two_reds : ℚ :=
  (num_red_balls / total_balls) * ((num_red_balls - 1) / (total_balls - 1))

-- The theorem to prove
theorem balls_picked_at_random_eq_two :
  probability_two_reds = given_probability → n = 2 :=
by
  sorry

end balls_picked_at_random_eq_two_l115_115953


namespace rectangle_placement_l115_115613

theorem rectangle_placement (a b c d : ℝ)
  (h1 : a < c)
  (h2 : c < d)
  (h3 : d < b)
  (h4 : a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b * d - a * c)^2 + (b * c - a * d)^2 :=
sorry

end rectangle_placement_l115_115613


namespace solve_for_x_l115_115657

theorem solve_for_x (x : ℝ) (h : ⌈x⌉ * x = 156) : x = 12 :=
sorry

end solve_for_x_l115_115657


namespace fraction_day_crew_loaded_l115_115608

variable (D W : ℕ)  -- D: Number of boxes loaded by each worker on the day crew, W: Number of workers on the day crew

-- Condition 1: Each worker on the night crew loaded 3/4 as many boxes as each worker on the day crew
def boxes_loaded_night_worker : ℕ := 3 * D / 4
-- Condition 2: The night crew has 5/6 as many workers as the day crew
def workers_night : ℕ := 5 * W / 6

-- Question: Fraction of all the boxes loaded by the day crew
theorem fraction_day_crew_loaded :
  (D * W : ℚ) / ((D * W) + (3 * D / 4) * (5 * W / 6)) = (8 / 13) := by
  sorry

end fraction_day_crew_loaded_l115_115608


namespace find_x_in_sequence_l115_115359

theorem find_x_in_sequence :
  ∃ x y z : Int, (z + 3 = 5) ∧ (y + z = 5) ∧ (x + y = 2) ∧ (x = -1) :=
by
  use -1, 3, 2
  sorry

end find_x_in_sequence_l115_115359


namespace sum_of_two_rationals_negative_l115_115631

theorem sum_of_two_rationals_negative (a b : ℚ) (h : a + b < 0) : a < 0 ∨ b < 0 := sorry

end sum_of_two_rationals_negative_l115_115631


namespace people_visited_on_Sunday_l115_115429

theorem people_visited_on_Sunday (ticket_price : ℕ) 
                                 (people_per_day_week : ℕ) 
                                 (people_on_Saturday : ℕ) 
                                 (total_revenue : ℕ) 
                                 (days_week : ℕ)
                                 (total_days : ℕ) 
                                 (people_per_day_mf : ℕ) 
                                 (people_on_other_days : ℕ) 
                                 (revenue_other_days : ℕ)
                                 (revenue_Sunday : ℕ)
                                 (people_Sunday : ℕ) :
    ticket_price = 3 →
    people_per_day_week = 100 →
    people_on_Saturday = 200 →
    total_revenue = 3000 →
    days_week = 5 →
    total_days = 7 →
    people_per_day_mf = people_per_day_week * days_week →
    people_on_other_days = people_per_day_mf + people_on_Saturday →
    revenue_other_days = people_on_other_days * ticket_price →
    revenue_Sunday = total_revenue - revenue_other_days →
    people_Sunday = revenue_Sunday / ticket_price →
    people_Sunday = 300 := 
by 
  sorry

end people_visited_on_Sunday_l115_115429


namespace count_divisible_by_8_l115_115053

theorem count_divisible_by_8 (a b k : ℕ) (h1 : a = 200) (h2 : b = 400) (h3 : k = 8) :
  ∃ n : ℕ, n = 26 ∧ (∀ x, a ≤ x ∧ x ≤ b → x % k = 0 → x = a + (n - 1) * k) → True :=
by {
  sorry
}

end count_divisible_by_8_l115_115053


namespace negation_of_p_is_neg_p_l115_115160

-- Define the original proposition p
def p : Prop := ∃ n : ℕ, 2^n > 100

-- Define what it means for the negation of p to be satisfied
def neg_p := ∀ n : ℕ, 2^n ≤ 100

-- Statement to prove the logical equivalence between the negation of p and neg_p
theorem negation_of_p_is_neg_p : ¬ p ↔ neg_p := by
  sorry

end negation_of_p_is_neg_p_l115_115160


namespace annual_decrease_rate_l115_115255

def initial_population : ℝ := 8000
def population_after_two_years : ℝ := 3920

theorem annual_decrease_rate :
  ∃ r : ℝ, (0 < r ∧ r < 1) ∧ (initial_population * (1 - r)^2 = population_after_two_years) ∧ r = 0.3 :=
by
  sorry

end annual_decrease_rate_l115_115255


namespace boat_distance_against_water_flow_l115_115389

variable (a : ℝ) -- speed of the boat in still water

theorem boat_distance_against_water_flow 
  (speed_boat_still_water : ℝ := a)
  (speed_water_flow : ℝ := 3)
  (time_travel : ℝ := 3) :
  (speed_boat_still_water - speed_water_flow) * time_travel = 3 * (a - 3) := 
by
  sorry

end boat_distance_against_water_flow_l115_115389


namespace range_of_a_l115_115465

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ a = x^2 - x - 1) ↔ -1 ≤ a ∧ a ≤ 1 := 
by 
  sorry

end range_of_a_l115_115465


namespace div_rule_2701_is_37_or_73_l115_115471

theorem div_rule_2701_is_37_or_73 (a b x : ℕ) (h1 : 10 * a + b = x) (h2 : a^2 + b^2 = 58) : 
  (x = 37 ∨ x = 73) ↔ 2701 % x = 0 :=
by
  sorry

end div_rule_2701_is_37_or_73_l115_115471


namespace M_inter_N_l115_115473

def M : Set ℝ := {x | abs (x - 1) < 2}
def N : Set ℝ := {x | x * (x - 3) < 0}

theorem M_inter_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 3} :=
by
  sorry

end M_inter_N_l115_115473


namespace quadratic_eq_one_solution_m_eq_49_div_12_l115_115288

theorem quadratic_eq_one_solution_m_eq_49_div_12 (m : ℝ) : 
  (∃ m, ∀ x, 3 * x ^ 2 - 7 * x + m = 0 → (b^2 - 4 * a * c = 0) → m = 49 / 12) :=
by
  sorry

end quadratic_eq_one_solution_m_eq_49_div_12_l115_115288


namespace line_product_l115_115648

theorem line_product (b m : ℝ) (h1: b = -1) (h2: m = 2) : m * b = -2 :=
by
  rw [h1, h2]
  norm_num


end line_product_l115_115648


namespace sum_c_eq_l115_115857

-- Definitions and conditions
def a_n : ℕ → ℝ := λ n => 2 ^ n
def b_n : ℕ → ℝ := λ n => 2 * n
def c_n (n : ℕ) : ℝ := a_n n * b_n n

-- Sum of the first n terms of sequence {c_n}
def sum_c (n : ℕ) : ℝ := (Finset.range n).sum c_n

-- Theorem statement
theorem sum_c_eq (n : ℕ) : sum_c n = (n - 1) * 2 ^ (n + 2) + 4 :=
sorry

end sum_c_eq_l115_115857


namespace internet_usage_minutes_l115_115900

-- Define the given conditions
variables (M P E : ℕ)

-- Problem statement
theorem internet_usage_minutes (h : P ≠ 0) : 
  (∀ M P E : ℕ, ∃ y : ℕ, y = (100 * E * M) / P) :=
by {
  sorry
}

end internet_usage_minutes_l115_115900


namespace remainder_a83_l115_115478

def a_n (n : ℕ) : ℕ := 6^n + 8^n

theorem remainder_a83 (n : ℕ) : 
  a_n 83 % 49 = 35 := sorry

end remainder_a83_l115_115478


namespace terminal_zeros_75_480_l115_115055

theorem terminal_zeros_75_480 :
  let x := 75
  let y := 480
  let fact_x := 5^2 * 3
  let fact_y := 2^5 * 3 * 5
  let product := fact_x * fact_y
  let num_zeros := min (3) (5)
  num_zeros = 3 :=
by
  sorry

end terminal_zeros_75_480_l115_115055


namespace correct_functional_relationship_water_amount_20th_minute_water_amount_supply_days_l115_115270

-- Define the constants k and b
variables (k b : ℝ)

-- Define the function y = k * t + b
def linear_func (t : ℝ) : ℝ := k * t + b

-- Define the data points as conditions
axiom data_point1 : linear_func k b 1 = 7
axiom data_point2 : linear_func k b 2 = 12
axiom data_point3 : linear_func k b 3 = 17
axiom data_point4 : linear_func k b 4 = 22
axiom data_point5 : linear_func k b 5 = 27

-- Define the water consumption rate and total minutes in a day
def daily_water_consumption : ℝ := 1500
def minutes_in_one_day : ℝ := 1440
def days_in_month : ℝ := 30

-- The expression y = 5t + 2
theorem correct_functional_relationship : (k = 5) ∧ (b = 2) :=
by
  sorry

-- Estimated water amount at the 20th minute
theorem water_amount_20th_minute (t : ℝ) (ht : t = 20) : linear_func 5 2 t = 102 :=
by
  sorry

-- The water leaked in a month (30 days) can supply the number of days
theorem water_amount_supply_days : (linear_func 5 2 (minutes_in_one_day * days_in_month)) / daily_water_consumption = 144 :=
by
  sorry

end correct_functional_relationship_water_amount_20th_minute_water_amount_supply_days_l115_115270


namespace compute_a1d1_a2d2_a3d3_l115_115684

theorem compute_a1d1_a2d2_a3d3
  (a1 a2 a3 d1 d2 d3 : ℝ)
  (h : ∀ x : ℝ, x^6 + 2 * x^5 + x^4 + x^3 + x^2 + 2 * x + 1 = (x^2 + a1*x + d1) * (x^2 + a2*x + d2) * (x^2 + a3*x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 2 :=
by
  sorry

end compute_a1d1_a2d2_a3d3_l115_115684


namespace bags_already_made_l115_115748

def bags_per_batch : ℕ := 10
def customer_order : ℕ := 60
def days_to_fulfill : ℕ := 4
def batches_per_day : ℕ := 1

theorem bags_already_made :
  (customer_order - (days_to_fulfill * batches_per_day * bags_per_batch)) = 20 :=
by
  sorry

end bags_already_made_l115_115748


namespace fraction_still_missing_l115_115567

theorem fraction_still_missing (x : ℕ) (hx : x > 0) :
  let lost := (1/3 : ℚ) * x
  let found := (2/3 : ℚ) * lost
  let remaining := x - lost + found
  (x - remaining) / x = (1/9 : ℚ) :=
by
  let lost := (1/3 : ℚ) * x
  let found := (2/3 : ℚ) * lost
  let remaining := x - lost + found
  have h_fraction_still_missing : (x - remaining) / x = (1/9 : ℚ) := sorry
  exact h_fraction_still_missing

end fraction_still_missing_l115_115567


namespace solve_for_x_l115_115820

def f (x : ℝ) : ℝ := 3 * x - 4

noncomputable def f_inv (x : ℝ) : ℝ := (x + 4) / 3

theorem solve_for_x : ∃ x : ℝ, f x = f_inv x ∧ x = 2 := by
  sorry

end solve_for_x_l115_115820


namespace value_of_a5_l115_115256

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n m : ℕ, a n * r ^ (m - n) = a m

theorem value_of_a5 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h : a 3 * a 7 = 64) :
  a 5 = 8 ∨ a 5 = -8 :=
by
  sorry

end value_of_a5_l115_115256


namespace total_delegates_l115_115125

theorem total_delegates 
  (D: ℕ) 
  (h1: 16 ≤ D)
  (h2: (D - 16) % 2 = 0)
  (h3: 10 ≤ D - 16) : D = 36 := 
sorry

end total_delegates_l115_115125


namespace rows_seating_exactly_10_people_exists_l115_115897

theorem rows_seating_exactly_10_people_exists :
  ∃ y x : ℕ, 73 = 10 * y + 9 * x ∧ (73 - 10 * y) % 9 = 0 := 
sorry

end rows_seating_exactly_10_people_exists_l115_115897


namespace percentage_discount_is_12_l115_115682

noncomputable def cost_price : ℝ := 47.50
noncomputable def list_price : ℝ := 67.47
noncomputable def desired_selling_price : ℝ := cost_price + 0.25 * cost_price
noncomputable def actual_selling_price : ℝ := 59.375

theorem percentage_discount_is_12 :
  ∃ D : ℝ, desired_selling_price = list_price - (list_price * D) ∧ D = 0.12 := 
by 
  sorry

end percentage_discount_is_12_l115_115682


namespace distance_traveled_l115_115931

-- Define the variables for speed of slower and faster bike
def slower_speed := 60
def faster_speed := 64

-- Define the condition that slower bike takes 1 hour more than faster bike
def condition (D : ℝ) : Prop := (D / slower_speed) = (D / faster_speed) + 1

-- The theorem we need to prove
theorem distance_traveled : ∃ (D : ℝ), condition D ∧ D = 960 := 
by
  sorry

end distance_traveled_l115_115931


namespace linda_total_distance_l115_115930

theorem linda_total_distance :
  ∃ x : ℕ, (60 % x = 0) ∧ ((75 % (x + 3)) = 0) ∧ ((90 % (x + 6)) = 0) ∧
  (60 / x + 75 / (x + 3) + 90 / (x + 6) = 15) :=
sorry

end linda_total_distance_l115_115930


namespace cos_C_equal_two_thirds_l115_115464

variable {A B C : ℝ}
variable {a b c : ℝ}

-- Define the conditions
def condition1 : a > 0 ∧ b > 0 ∧ c > 0 := sorry
def condition2 : (a / b) + (b / a) = 4 * Real.cos C := sorry
def condition3 : Real.cos (A - B) = 1 / 6 := sorry

-- Statement to prove
theorem cos_C_equal_two_thirds 
  (h1: a > 0 ∧ b > 0 ∧ c > 0) 
  (h2: (a / b) + (b / a) = 4 * Real.cos C) 
  (h3: Real.cos (A - B) = 1 / 6) 
  : Real.cos C = 2 / 3 :=
  sorry

end cos_C_equal_two_thirds_l115_115464


namespace negation_p_l115_115763

def nonneg_reals := { x : ℝ // 0 ≤ x }

def p := ∀ x : nonneg_reals, Real.exp x.1 ≥ 1

theorem negation_p :
  ¬ p ↔ ∃ x : nonneg_reals, Real.exp x.1 < 1 :=
by
  sorry

end negation_p_l115_115763


namespace domain_of_f_l115_115432

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (-x^2 + 9 * x + 10)) / Real.log (x - 1)

theorem domain_of_f :
  {x : ℝ | -x^2 + 9 * x + 10 ≥ 0 ∧ x - 1 > 0 ∧ Real.log (x - 1) ≠ 0} =
  {x : ℝ | (1 < x ∧ x < 2) ∨ (2 < x ∧ x ≤ 10)} :=
by
  sorry

end domain_of_f_l115_115432


namespace distance_focus_directrix_l115_115730

theorem distance_focus_directrix (p : ℝ) :
  (∀ (x y : ℝ), y^2 = 2 * p * x ∧ x = 6 ∧ dist (x, y) (p/2, 0) = 10) →
  abs (p) = 8 :=
by
  sorry

end distance_focus_directrix_l115_115730


namespace min_value_l115_115119

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 1/(a-1) + 4/(b-1) ≥ 4 :=
by
  sorry

end min_value_l115_115119


namespace number_of_boys_in_second_group_l115_115699

noncomputable def daily_work_done_by_man (M : ℝ) (B : ℝ) : Prop :=
  M = 2 * B

theorem number_of_boys_in_second_group
  (M B : ℝ)
  (h1 : (12 * M + 16 * B) * 5 = (13 * M + 24 * B) * 4)
  (h2 : daily_work_done_by_man M B) :
  24 = 24 :=
by
  -- The proof is omitted.
  sorry

end number_of_boys_in_second_group_l115_115699


namespace angle_C_of_triangle_l115_115809

theorem angle_C_of_triangle (A B C : ℝ) (hA : A = 90) (hB : B = 50) (h_sum : A + B + C = 180) : C = 40 := 
by
  sorry

end angle_C_of_triangle_l115_115809


namespace no_pairs_satisfy_equation_l115_115079

theorem no_pairs_satisfy_equation :
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → ¬ (2 / a + 2 / b = 1 / (a + b)) :=
by
  intros a b ha hb h
  -- the proof would go here
  sorry

end no_pairs_satisfy_equation_l115_115079


namespace pineapples_sold_l115_115553

/-- 
There were initially 86 pineapples in the store. After selling some pineapples,
9 of the remaining pineapples were rotten and were discarded. Given that there 
are 29 fresh pineapples left, prove that the number of pineapples sold is 48.
-/
theorem pineapples_sold (initial_pineapples : ℕ) (rotten_pineapples : ℕ) (remaining_fresh_pineapples : ℕ)
  (h_init : initial_pineapples = 86)
  (h_rotten : rotten_pineapples = 9)
  (h_fresh : remaining_fresh_pineapples = 29) :
  initial_pineapples - (remaining_fresh_pineapples + rotten_pineapples) = 48 :=
sorry

end pineapples_sold_l115_115553


namespace find_m_l115_115685

theorem find_m (a b c m x : ℂ) :
  ( (2 * m + 1) * (x^2 - (b + 1) * x) = (2 * m - 3) * (2 * a * x - c) )
  →
  (x = (b + 1)) 
  →
  m = 1.5 := by
  sorry

end find_m_l115_115685


namespace initial_seashells_l115_115378

-- Definitions for the conditions
def seashells_given_to_Tim : ℕ := 13
def seashells_now : ℕ := 36

-- Proving the number of initially found seashells
theorem initial_seashells : seashells_now + seashells_given_to_Tim = 49 :=
by
  -- we omit the proof steps with sorry
  sorry

end initial_seashells_l115_115378


namespace system_of_inequalities_solutions_l115_115611

theorem system_of_inequalities_solutions (x : ℤ) :
  (3 * x - 2 ≥ 2 * x - 5) ∧ ((x / 2 - (x - 2) / 3 < 1 / 2)) →
  (x = -3 ∨ x = -2) :=
by sorry

end system_of_inequalities_solutions_l115_115611


namespace value_of_x_plus_y_l115_115907

theorem value_of_x_plus_y (x y : ℚ) (h1 : 1 / x + 1 / y = 5) (h2 : 1 / x - 1 / y = -9) : x + y = -5 / 14 := sorry

end value_of_x_plus_y_l115_115907


namespace total_area_of_hexagon_is_693_l115_115995

-- Conditions
def hexagon_side1_length := 3
def hexagon_side2_length := 2
def angle_between_length3_sides := 120
def all_internal_triangles_are_equilateral := true
def number_of_triangles := 6

-- Define the problem statement
theorem total_area_of_hexagon_is_693 
  (a1 : hexagon_side1_length = 3)
  (a2 : hexagon_side2_length = 2)
  (a3 : angle_between_length3_sides = 120)
  (a4 : all_internal_triangles_are_equilateral = true)
  (a5 : number_of_triangles = 6) :
  total_area_of_hexagon = 693 :=
by
  sorry

end total_area_of_hexagon_is_693_l115_115995


namespace two_faucets_fill_60_gallons_l115_115993

def four_faucets_fill (tub_volume : ℕ) (time_minutes : ℕ) : Prop :=
  4 * (tub_volume / time_minutes) = 120 / 5

def two_faucets_fill (tub_volume : ℕ) (time_minutes : ℕ) : Prop :=
  2 * (tub_volume / time_minutes) = 60 / time_minutes

theorem two_faucets_fill_60_gallons :
  (four_faucets_fill 120 5) → ∃ t: ℕ, two_faucets_fill 60 t ∧ t = 5 :=
by {
  sorry
}

end two_faucets_fill_60_gallons_l115_115993


namespace avg10_students_correct_l115_115355

-- Definitions for the conditions
def avg15_students : ℝ := 70
def num15_students : ℕ := 15
def num10_students : ℕ := 10
def num25_students : ℕ := num15_students + num10_students
def avg25_students : ℝ := 80

-- Total percentage calculation based on conditions
def total_perc25_students := num25_students * avg25_students
def total_perc15_students := num15_students * avg15_students

-- The average percent of the 10 students, based on the conditions and given average for 25 students.
theorem avg10_students_correct : 
  (total_perc25_students - total_perc15_students) / (num10_students : ℝ) = 95 := by
  sorry

end avg10_students_correct_l115_115355


namespace steve_needs_28_feet_of_wood_l115_115130

-- Define the required lengths
def lengths_4_feet : Nat := 6
def lengths_2_feet : Nat := 2

-- Define the wood length in feet for each type
def wood_length_4 : Nat := 4
def wood_length_2 : Nat := 2

-- Total feet of wood required
def total_wood : Nat := lengths_4_feet * wood_length_4 + lengths_2_feet * wood_length_2

-- The theorem to prove that the total amount of wood required is 28 feet
theorem steve_needs_28_feet_of_wood : total_wood = 28 :=
by
  sorry

end steve_needs_28_feet_of_wood_l115_115130


namespace Deepak_age_l115_115500

-- Define the current ages of Arun and Deepak
variable (A D : ℕ)

-- Define the conditions
def ratio_condition := A / D = 4 / 3
def future_age_condition := A + 6 = 26

-- Define the proof statement
theorem Deepak_age (h1 : ratio_condition A D) (h2 : future_age_condition A) : D = 15 :=
  sorry

end Deepak_age_l115_115500


namespace solve_for_x_l115_115273

theorem solve_for_x (y : ℝ) (x : ℝ) 
  (h : x / (x - 1) = (y^2 + 3 * y - 2) / (y^2 + 3 * y - 3)) : 
  x = (y^2 + 3 * y - 2) / 2 := 
by 
  sorry

end solve_for_x_l115_115273


namespace triple_integral_value_l115_115943

theorem triple_integral_value :
  (∫ x in (-1 : ℝ)..1, ∫ y in (x^2 : ℝ)..1, ∫ z in (0 : ℝ)..y, (4 + z) ) = (16 / 3 : ℝ) :=
by
  sorry

end triple_integral_value_l115_115943


namespace sum_of_ten_distinct_numbers_lt_75_l115_115044

theorem sum_of_ten_distinct_numbers_lt_75 :
  ∃ (S : Finset ℕ), S.card = 10 ∧
  (∃ (S_div_5 : Finset ℕ), S_div_5 ⊆ S ∧ S_div_5.card = 3 ∧ ∀ x ∈ S_div_5, 5 ∣ x) ∧
  (∃ (S_div_4 : Finset ℕ), S_div_4 ⊆ S ∧ S_div_4.card = 4 ∧ ∀ x ∈ S_div_4, 4 ∣ x) ∧
  S.sum id < 75 :=
by { 
  sorry 
}

end sum_of_ten_distinct_numbers_lt_75_l115_115044


namespace certain_number_l115_115211

theorem certain_number (p q : ℝ) (h1 : 3 / p = 6) (h2 : p - q = 0.3) : 3 / q = 15 :=
by
  sorry

end certain_number_l115_115211


namespace ratio_of_chris_to_amy_l115_115052

-- Definitions based on the conditions in the problem
def combined_age (Amy_age Jeremy_age Chris_age : ℕ) : Prop :=
  Amy_age + Jeremy_age + Chris_age = 132

def amy_is_one_third_jeremy (Amy_age Jeremy_age : ℕ) : Prop :=
  Amy_age = Jeremy_age / 3

def jeremy_age : ℕ := 66

-- The main theorem we need to prove
theorem ratio_of_chris_to_amy (Amy_age Chris_age : ℕ) (h1 : combined_age Amy_age jeremy_age Chris_age)
  (h2 : amy_is_one_third_jeremy Amy_age jeremy_age) : Chris_age / Amy_age = 2 :=
sorry

end ratio_of_chris_to_amy_l115_115052


namespace walnut_trees_planted_l115_115834

theorem walnut_trees_planted (initial_trees : ℕ) (final_trees : ℕ) (num_trees_planted : ℕ) : initial_trees = 107 → final_trees = 211 → num_trees_planted = final_trees - initial_trees → num_trees_planted = 104 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end walnut_trees_planted_l115_115834


namespace max_percentage_l115_115965

def total_students : ℕ := 100
def group_size : ℕ := 66
def min_percentage (scores : Fin 100 → ℝ) : Prop :=
  ∀ (S : Finset (Fin 100)), S.card = 66 → (S.sum scores) / (Finset.univ.sum scores) ≥ 0.5

theorem max_percentage (scores : Fin 100 → ℝ) (h : min_percentage scores) :
  ∃ (x : ℝ), ∀ i : Fin 100, scores i <= x ∧ x <= 0.25 * (Finset.univ.sum scores) := sorry

end max_percentage_l115_115965


namespace roots_of_polynomial_l115_115720

noncomputable def polynomial (m z : ℝ) : ℝ :=
  z^3 - (m^2 - m + 7) * z - (3 * m^2 - 3 * m - 6)

theorem roots_of_polynomial (m z : ℝ) (h : polynomial m (-1) = 0) :
  (m = 3 ∧ z = 4 ∨ z = -3) ∨ (m = -2 ∧ sorry) :=
sorry

end roots_of_polynomial_l115_115720


namespace determine_q_l115_115207

-- Define the polynomial p(x) and its square
def p (x : ℝ) : ℝ := x^2 + x + 1
def p_squared (x : ℝ) : ℝ := (x^2 + x + 1)^2

-- Define the identity condition
def identity_condition (x : ℝ) (q : ℝ → ℝ) : Prop := 
  p_squared x - 2 * p x * q x + (q x)^2 - 4 * p x + 3 * q x + 3 = 0

-- Ellaboration on the required solution
def correct_q (q : ℝ → ℝ) : Prop :=
  (∀ x, q x = x^2 + 2 * x) ∨ (∀ x, q x = x^2 - 1)

-- The theorem statement
theorem determine_q :
  ∀ q : ℝ → ℝ, (∀ x : ℝ, identity_condition x q) → correct_q q :=
by
  intros
  sorry

end determine_q_l115_115207


namespace calculate_integral_cos8_l115_115070

noncomputable def integral_cos8 : ℝ :=
  ∫ x in (Real.pi / 2)..(2 * Real.pi), 2^8 * (Real.cos x)^8

theorem calculate_integral_cos8 :
  integral_cos8 = 219 * Real.pi :=
by
  sorry

end calculate_integral_cos8_l115_115070


namespace fraction_not_integer_l115_115285

theorem fraction_not_integer (a b : ℤ) : ¬ (∃ k : ℤ, (a^2 + b^2) = k * (a^2 - b^2)) :=
sorry

end fraction_not_integer_l115_115285


namespace travel_ways_l115_115646

theorem travel_ways (buses : Nat) (trains : Nat) (boats : Nat) 
  (hb : buses = 5) (ht : trains = 6) (hb2 : boats = 2) : 
  buses + trains + boats = 13 := by
  sorry

end travel_ways_l115_115646


namespace product_of_five_consecutive_integers_divisible_by_120_l115_115605

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_integers_divisible_by_120_l115_115605


namespace chocolate_bars_squares_l115_115345

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

end chocolate_bars_squares_l115_115345


namespace final_spent_l115_115440

-- Define all the costs.
def albertoExpenses : ℤ := 2457 + 374 + 520 + 129 + 799
def albertoDiscountExhaust : ℤ := (799 * 5) / 100
def albertoTotalBeforeLoyaltyDiscount : ℤ := albertoExpenses - albertoDiscountExhaust
def albertoLoyaltyDiscount : ℤ := (albertoTotalBeforeLoyaltyDiscount * 7) / 100
def albertoFinal : ℤ := albertoTotalBeforeLoyaltyDiscount - albertoLoyaltyDiscount

def samaraExpenses : ℤ := 25 + 467 + 79 + 175 + 599 + 225
def samaraSalesTax : ℤ := (samaraExpenses * 6) / 100
def samaraFinal : ℤ := samaraExpenses + samaraSalesTax

def difference : ℤ := albertoFinal - samaraFinal

theorem final_spent (h : difference = 2278) : true :=
  sorry

end final_spent_l115_115440


namespace min_b_minus_2c_over_a_l115_115655

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (h1 : a ≤ b + c ∧ b + c ≤ 3 * a)
variable (h2 : 3 * b^2 ≤ a * (a + c) ∧ a * (a + c) ≤ 5 * b^2)

theorem min_b_minus_2c_over_a : (∃ u : ℝ, (u = (b - 2 * c) / a) ∧ (∀ v : ℝ, (v = (b - 2 * c) / a) → u ≤ v)) :=
  sorry

end min_b_minus_2c_over_a_l115_115655


namespace minimal_q_for_fraction_l115_115826

theorem minimal_q_for_fraction :
  ∃ p q : ℕ, 0 < p ∧ 0 < q ∧ 
  (3/5 : ℚ) < p / q ∧ p / q < (5/8 : ℚ) ∧
  (∀ r : ℕ, 0 < r ∧ (3/5 : ℚ) < p / r ∧ p / r < (5/8 : ℚ) → q ≤ r) ∧
  p + q = 21 :=
by
  sorry

end minimal_q_for_fraction_l115_115826


namespace tommy_balloons_l115_115292

/-- Tommy had some balloons. He received 34 more balloons from his mom,
gave away 15 balloons, and exchanged the remaining balloons for teddy bears
at a rate of 3 balloons per teddy bear. After these transactions, he had 30 teddy bears.
Prove that Tommy started with 71 balloons -/
theorem tommy_balloons : 
  ∃ B : ℕ, (B + 34 - 15) = 3 * 30 ∧ B = 71 := 
by
  have h : (71 + 34 - 15) = 3 * 30 := by norm_num
  exact ⟨71, h, rfl⟩

end tommy_balloons_l115_115292


namespace inscribed_sphere_radius_l115_115267

theorem inscribed_sphere_radius (b d : ℝ) : 
  (b * Real.sqrt d - b = 15 * (Real.sqrt 5 - 1) / 4) → 
  b + d = 11.75 :=
by
  intro h
  sorry

end inscribed_sphere_radius_l115_115267


namespace volume_ratio_l115_115331

theorem volume_ratio (a b : ℝ) (h : a^2 / b^2 = 9 / 25) : b^3 / a^3 = 125 / 27 :=
by
  -- Skipping the proof by adding 'sorry'
  sorry

end volume_ratio_l115_115331
