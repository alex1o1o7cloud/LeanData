import Mathlib

namespace lines_coplanar_l289_28953

/-
Given:
- Line 1 parameterized as (2 + s, 4 - k * s, -1 + k * s)
- Line 2 parameterized as (2 * t, 2 + t, 3 - t)
Prove: If these lines are coplanar, then k = -1/2
-/
theorem lines_coplanar (k : ℚ) (s t : ℚ)
  (line1 : ℚ × ℚ × ℚ := (2 + s, 4 - k * s, -1 + k * s))
  (line2 : ℚ × ℚ × ℚ := (2 * t, 2 + t, 3 - t))
  (coplanar : ∃ (s t : ℚ), line1 = line2) :
  k = -1 / 2 := 
sorry

end lines_coplanar_l289_28953


namespace sufficient_condition_a_gt_1_l289_28954

variable (a : ℝ)

theorem sufficient_condition_a_gt_1 (h : a > 1) : a^2 > 1 :=
by sorry

end sufficient_condition_a_gt_1_l289_28954


namespace sequence_property_l289_28974

noncomputable def seq (n : ℕ) : ℕ := 
if n = 0 then 1 else 
if n = 1 then 3 else 
seq (n-2) + 3 * 2^(n-2)

theorem sequence_property {n : ℕ} (h_pos : n > 0) :
(∀ n : ℕ, n > 0 → seq (n + 2) ≤ seq n + 3 * 2^n) →
(∀ n : ℕ, n > 0 → seq (n + 1) ≥ 2 * seq n + 1) →
seq n = 2^n - 1 := 
sorry

end sequence_property_l289_28974


namespace inequality_abc_l289_28976

theorem inequality_abc (a b c : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) (h5 : 2 ≤ n) :
  (a / (b + c)^(1/(n:ℝ)) + b / (c + a)^(1/(n:ℝ)) + c / (a + b)^(1/(n:ℝ)) ≥ 3 / 2^(1/(n:ℝ))) :=
by sorry

end inequality_abc_l289_28976


namespace player_match_count_l289_28923

open Real

theorem player_match_count (n : ℕ) : 
  (∃ T, T = 32 * n ∧ (T + 98) / (n + 1) = 38) → n = 10 :=
by
  sorry

end player_match_count_l289_28923


namespace positional_relationship_l289_28964

variables {Point Line Plane : Type}
variables (a b : Line) (α : Plane)

-- Condition 1: Line a is parallel to Plane α
def line_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry

-- Condition 2: Line b is contained within Plane α
def line_contained_within_plane (b : Line) (α : Plane) : Prop := sorry

-- The positional relationship between line a and line b is either parallel or skew
def lines_parallel_or_skew (a b : Line) : Prop := sorry

theorem positional_relationship (ha : line_parallel_to_plane a α) (hb : line_contained_within_plane b α) :
  lines_parallel_or_skew a b :=
sorry

end positional_relationship_l289_28964


namespace points_same_color_separed_by_two_l289_28904

theorem points_same_color_separed_by_two (circle : Fin 239 → Bool) : 
  ∃ i j : Fin 239, i ≠ j ∧ (i + 2) % 239 = j ∧ circle i = circle j :=
by
  sorry

end points_same_color_separed_by_two_l289_28904


namespace ribbon_per_box_l289_28952

theorem ribbon_per_box (ribbon_total ribbon_each : ℚ) (n : ℕ) (hn : n = 5) (h : ribbon_total = 5 / 12) :
  ribbon_each = ribbon_total / n ↔ ribbon_each = 1 / 12 :=
by
  sorry

end ribbon_per_box_l289_28952


namespace find_second_number_l289_28936

theorem find_second_number (a b c : ℕ) 
  (h1 : a + b + c = 550) 
  (h2 : a = 2 * b) 
  (h3 : c = a / 3) :
  b = 150 :=
by
  sorry

end find_second_number_l289_28936


namespace number_of_parallelograms_l289_28935

-- Problem's condition
def side_length (n : ℕ) : Prop := n > 0

-- Required binomial coefficient (combination formula)
def binom (n k : ℕ) : ℕ := n.choose k

-- Total number of parallelograms in the tiling
theorem number_of_parallelograms (n : ℕ) (h : side_length n) : 
  3 * binom (n + 2) 4 = 3 * (n+2).choose 4 :=
by
  sorry

end number_of_parallelograms_l289_28935


namespace binary_multiplication_l289_28968

theorem binary_multiplication :
  let a := 0b1101
  let b := 0b111
  let product := 0b1000111
  a * b = product :=
by 
  let a := 0b1101
  let b := 0b111
  let product := 0b1000111
  sorry

end binary_multiplication_l289_28968


namespace cody_initial_tickets_l289_28987

def initial_tickets (lost : ℝ) (spent : ℝ) (left : ℝ) : ℝ :=
  lost + spent + left

theorem cody_initial_tickets : initial_tickets 6.0 25.0 18.0 = 49.0 := by
  sorry

end cody_initial_tickets_l289_28987


namespace probability_XiaoYu_group_A_l289_28900

theorem probability_XiaoYu_group_A :
  ∀ (students : Fin 48) (groups : Fin 4) (groupAssignment : Fin 48 → Fin 4)
    (student : Fin 48) (groupA : Fin 4),
    (∀ (s : Fin 48), ∃ (g : Fin 4), groupAssignment s = g) → 
    (∀ (g : Fin 4), ∃ (count : ℕ), (0 < count ∧ count ≤ 12) ∧
       (∃ (groupMembers : List (Fin 48)), groupMembers.length = count ∧
        (∀ (m : Fin 48), m ∈ groupMembers → groupAssignment m = g))) →
    (groupAssignment student = groupA) →
  ∃ (p : ℚ), p = (1/4) ∧ ∀ (s : Fin 48), groupAssignment s = groupA → p = (1/4) :=
by
  sorry

end probability_XiaoYu_group_A_l289_28900


namespace total_trees_now_l289_28998

-- Definitions from conditions
def initial_mahogany_trees : ℕ := 50
def initial_narra_trees : ℕ := 30
def total_fallen_trees : ℕ := 5

-- Additional definitions capturing relations
def fell_narra_trees (x : ℕ) : Prop := x + (x + 1) = total_fallen_trees
def new_narra_trees_planted (x : ℕ) : ℕ := 2 * x
def new_mahogany_trees_planted (x : ℕ) : ℕ := 3 * (x + 1)

-- Final goal
theorem total_trees_now (x : ℕ) (h : fell_narra_trees x) :
  initial_mahogany_trees + initial_narra_trees
  - total_fallen_trees
  + new_narra_trees_planted x
  + new_mahogany_trees_planted x = 88 := by
  sorry

end total_trees_now_l289_28998


namespace cost_price_percentage_l289_28938

variable (SP CP : ℝ)

-- Assumption that the profit percent is 25%
axiom profit_percent : 25 = ((SP - CP) / CP) * 100

-- The statement to prove
theorem cost_price_percentage : CP / SP = 0.8 := by
  sorry

end cost_price_percentage_l289_28938


namespace train_stop_duration_l289_28940

theorem train_stop_duration (speed_without_stoppages speed_with_stoppages : ℕ) (h1 : speed_without_stoppages = 45) (h2 : speed_with_stoppages = 42) :
  ∃ t : ℕ, t = 4 :=
by
  sorry

end train_stop_duration_l289_28940


namespace find_S2017_l289_28920

-- Setting up the given conditions and sequences
def a1 : ℤ := -2014
def S (n : ℕ) : ℚ := n * a1 + (n * (n - 1) / 2) * 2 -- Using the provided sum formula

theorem find_S2017
  (h1 : a1 = -2014)
  (h2 : (S 2014) / 2014 - (S 2008) / 2008 = 6) :
  S 2017 = 4034 := 
sorry

end find_S2017_l289_28920


namespace vanessa_savings_remaining_l289_28982

-- Conditions
def initial_investment : ℝ := 50000
def annual_interest_rate : ℝ := 0.035
def investment_duration : ℕ := 3
def conversion_rate : ℝ := 0.85
def cost_per_toy : ℝ := 75

-- Given the above conditions, prove the remaining amount in euros after buying as many toys as possible is 16.9125
theorem vanessa_savings_remaining
  (P : ℝ := initial_investment)
  (r : ℝ := annual_interest_rate)
  (t : ℕ := investment_duration)
  (c : ℝ := conversion_rate)
  (e : ℝ := cost_per_toy) :
  (((P * (1 + r)^t) * c) - (e * (⌊(P * (1 + r)^3 * 0.85) / e⌋))) = 16.9125 :=
sorry

end vanessa_savings_remaining_l289_28982


namespace range_of_a_l289_28926

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + 4 * x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a (f a x) ≥ 0) ↔ a ≥ 3 :=
sorry

end range_of_a_l289_28926


namespace fraction_squared_0_0625_implies_value_l289_28915

theorem fraction_squared_0_0625_implies_value (x : ℝ) (hx : x^2 = 0.0625) : x = 0.25 :=
sorry

end fraction_squared_0_0625_implies_value_l289_28915


namespace paint_faces_l289_28937

def cuboid_faces : ℕ := 6
def number_of_cuboids : ℕ := 8 
def total_faces_painted : ℕ := cuboid_faces * number_of_cuboids

theorem paint_faces (h1 : cuboid_faces = 6) (h2 : number_of_cuboids = 8) : total_faces_painted = 48 := by
  -- conditions are defined above
  sorry

end paint_faces_l289_28937


namespace total_dogs_equation_l289_28913

/-- Definition of the number of boxes and number of dogs per box. --/
def num_boxes : ℕ := 7
def dogs_per_box : ℕ := 4

/-- The total number of dogs --/
theorem total_dogs_equation : num_boxes * dogs_per_box = 28 := by 
  sorry

end total_dogs_equation_l289_28913


namespace product_of_two_numbers_l289_28979

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.lcm a b = 72) (h2 : Nat.gcd a b = 8) :
  a * b = 576 :=
by
  sorry

end product_of_two_numbers_l289_28979


namespace Jeanine_has_more_pencils_than_Clare_l289_28927

def number_pencils_Jeanine_bought : Nat := 18
def number_pencils_Clare_bought := number_pencils_Jeanine_bought / 2
def number_pencils_given_to_Abby := number_pencils_Jeanine_bought / 3
def number_pencils_Jeanine_now := number_pencils_Jeanine_bought - number_pencils_given_to_Abby 

theorem Jeanine_has_more_pencils_than_Clare :
  number_pencils_Jeanine_now - number_pencils_Clare_bought = 3 := by
  sorry

end Jeanine_has_more_pencils_than_Clare_l289_28927


namespace find_positive_n_l289_28912

def arithmetic_sequence (a d : ℤ) (n : ℤ) := a + (n - 1) * d

def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := (n * (2 * a + (n - 1) * d)) / 2

theorem find_positive_n :
  ∃ (n : ℕ), n > 0 ∧ ∀ a d : ℤ, a = -12 → sum_of_first_n_terms a d 13 = 0 → arithmetic_sequence a d n > 0 ∧ n = 8 := 
sorry

end find_positive_n_l289_28912


namespace fraction_meaningful_l289_28947

-- Define the condition about the denominator not being zero.
def denominator_condition (x : ℝ) : Prop := x + 2 ≠ 0

-- The proof problem statement.
theorem fraction_meaningful (x : ℝ) : denominator_condition x ↔ x ≠ -2 :=
by
  -- Ensure that the Lean environment is aware this is a theorem statement.
  sorry -- Proof is omitted as instructed.

end fraction_meaningful_l289_28947


namespace line_intersects_circle_l289_28944

theorem line_intersects_circle (α : ℝ) (r : ℝ) (hα : true) (hr : r > 0) :
  (∃ x y : ℝ, (x * Real.cos α + y * Real.sin α = 1) ∧ (x^2 + y^2 = r^2)) → r > 1 :=
by
  sorry

end line_intersects_circle_l289_28944


namespace max_police_officers_needed_l289_28957

theorem max_police_officers_needed : 
  let streets := 10
  let non_parallel := true
  let curved_streets := 2
  let additional_intersections_per_curved := 3 
  streets = 10 ∧ 
  non_parallel = true ∧ 
  curved_streets = 2 ∧ 
  additional_intersections_per_curved = 3 → 
  ( (streets * (streets - 1) / 2) + (curved_streets * additional_intersections_per_curved) ) = 51 :=
by
  intros
  sorry

end max_police_officers_needed_l289_28957


namespace product_of_two_real_numbers_sum_three_times_product_l289_28970

variable (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)

theorem product_of_two_real_numbers_sum_three_times_product
    (h : x + y = 3 * x * y) :
  x * y = (x + y) / 3 :=
sorry

end product_of_two_real_numbers_sum_three_times_product_l289_28970


namespace parabola_directrix_is_x_eq_1_l289_28975

noncomputable def parabola_directrix (y : ℝ) : ℝ :=
  -1 / 4 * y^2

theorem parabola_directrix_is_x_eq_1 :
  ∀ x y, x = parabola_directrix y → x = 1 :=
by
  sorry

end parabola_directrix_is_x_eq_1_l289_28975


namespace problem1_problem2_l289_28916

-- Problem 1
theorem problem1 (x: ℚ) (h: x + 1 / 4 = 7 / 4) : x = 3 / 2 :=
by sorry

-- Problem 2
theorem problem2 (x: ℚ) (h: 2 / 3 + x = 3 / 4) : x = 1 / 12 :=
by sorry

end problem1_problem2_l289_28916


namespace hypotenuse_length_l289_28993

open Real

-- Definitions corresponding to the conditions
def right_triangle_vertex_length (ADC_length : ℝ) (AEC_length : ℝ) (x : ℝ) : Prop :=
  0 < x ∧ x < π / 2 ∧ ADC_length = sqrt 3 * sin x ∧ AEC_length = sin x

def trisect_hypotenuse (BD : ℝ) (DE : ℝ) (EC : ℝ) (c : ℝ) : Prop :=
  BD = c / 3 ∧ DE = c / 3 ∧ EC = c / 3

-- Main theorem definition
theorem hypotenuse_length (x hypotenuse ADC_length AEC_length : ℝ) :
  right_triangle_vertex_length ADC_length AEC_length x →
  trisect_hypotenuse (hypotenuse / 3) (hypotenuse / 3) (hypotenuse / 3) hypotenuse →
  hypotenuse = sqrt 3 * sin x :=
by
  intros h₁ h₂
  sorry

end hypotenuse_length_l289_28993


namespace S_shaped_growth_curve_varied_growth_rate_l289_28997

theorem S_shaped_growth_curve_varied_growth_rate :
  ∀ (population_growth : ℝ → ℝ), 
    (∃ t1 t2 : ℝ, t1 < t2 ∧ 
      (∃ r : ℝ, r = population_growth t1 / t1 ∧ r ≠ population_growth t2 / t2)) 
    → 
    ∀ t3 t4 : ℝ, t3 < t4 → (population_growth t3 / t3) ≠ (population_growth t4 / t4) :=
by
  sorry

end S_shaped_growth_curve_varied_growth_rate_l289_28997


namespace exponent_evaluation_problem_l289_28911

theorem exponent_evaluation_problem (m : ℕ) : 
  (m^2 * m^3 ≠ m^6) → 
  (m^2 + m^4 ≠ m^6) → 
  ((m^3)^3 ≠ m^6) → 
  (m^7 / m = m^6) :=
by
  intros hA hB hC
  -- Provide the proof here
  sorry

end exponent_evaluation_problem_l289_28911


namespace percent_university_diploma_no_job_choice_l289_28980

theorem percent_university_diploma_no_job_choice
    (total_people : ℕ)
    (P1 : 10 * total_people / 100 = total_people / 10)
    (P2 : 20 * total_people / 100 = total_people / 5)
    (P3 : 30 * total_people / 100 = 3 * total_people / 10) :
  25 = (20 * total_people / (80 * total_people / 100)) :=
by
  sorry

end percent_university_diploma_no_job_choice_l289_28980


namespace total_wheels_eq_90_l289_28924

def total_wheels (num_bicycles : Nat) (wheels_per_bicycle : Nat) (num_tricycles : Nat) (wheels_per_tricycle : Nat) :=
  num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle

theorem total_wheels_eq_90 : total_wheels 24 2 14 3 = 90 :=
by
  sorry

end total_wheels_eq_90_l289_28924


namespace volume_sphere_gt_cube_l289_28972

theorem volume_sphere_gt_cube (a r : ℝ) (h : 6 * a^2 = 4 * π * r^2) : 
  (4 / 3) * π * r^3 > a^3 :=
by sorry

end volume_sphere_gt_cube_l289_28972


namespace find_R_value_l289_28981

noncomputable def x (Q : ℝ) : ℝ := Real.sqrt (Q / 2 + Real.sqrt (Q / 2))
noncomputable def y (Q : ℝ) : ℝ := Real.sqrt (Q / 2 - Real.sqrt (Q / 2))
noncomputable def R (Q : ℝ) : ℝ := (x Q)^6 + (y Q)^6 / 40

theorem find_R_value (Q : ℝ) : R Q = 10 :=
sorry

end find_R_value_l289_28981


namespace abs_inequality_solution_l289_28903

theorem abs_inequality_solution (x : ℝ) :
  |x + 2| + |x - 2| ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end abs_inequality_solution_l289_28903


namespace xyz_values_l289_28965

theorem xyz_values (x y z : ℝ)
  (h1 : x * y - 5 * y = 20)
  (h2 : y * z - 5 * z = 20)
  (h3 : z * x - 5 * x = 20) :
  x * y * z = 340 ∨ x * y * z = -62.5 := 
by sorry

end xyz_values_l289_28965


namespace parabola_directrix_l289_28955

theorem parabola_directrix (y : ℝ) : (∃ p : ℝ, x = (1 / (4 * p)) * y^2 ∧ p = 2) → x = -2 :=
by
  sorry

end parabola_directrix_l289_28955


namespace cos_135_eq_neg_sqrt2_div_2_l289_28984

theorem cos_135_eq_neg_sqrt2_div_2 : 
  Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_135_eq_neg_sqrt2_div_2_l289_28984


namespace find_f2_of_conditions_l289_28989

theorem find_f2_of_conditions (f g : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) 
                              (h_g : ∀ x, g x = f x + 9) 
                              (h_g_val : g (-2) = 3) : 
                              f 2 = 6 :=
by 
  sorry

end find_f2_of_conditions_l289_28989


namespace problem_statement_l289_28909

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem problem_statement :
  f (5 * Real.pi / 24) = Real.sqrt 2 ∧
  ∀ x, f x ≥ 1 ↔ ∃ k : ℤ, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 2 :=
by
  sorry

end problem_statement_l289_28909


namespace find_ratio_l289_28902

-- Define the geometric sequence properties and conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
 ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions stated in the problem
axiom h₁ : a 5 * a 11 = 3
axiom h₂ : a 3 + a 13 = 4

-- The goal is to find the values of a_15 / a_5
theorem find_ratio (h₁ : a 5 * a 11 = 3) (h₂ : a 3 + a 13 = 4) :
  ∃ r : ℝ, r = a 15 / a 5 ∧ (r = 3 ∨ r = 1 / 3) :=
sorry

end find_ratio_l289_28902


namespace grandson_age_l289_28977

theorem grandson_age (M S G : ℕ) (h1 : M = 2 * S) (h2 : S = 2 * G) (h3 : M + S + G = 140) : G = 20 :=
by 
  sorry

end grandson_age_l289_28977


namespace part1_part2_l289_28990

def setA (a : ℝ) := {x : ℝ | a - 1 ≤ x ∧ x ≤ 3 - 2 * a}
def setB := {x : ℝ | x^2 - 2 * x - 8 ≤ 0}

theorem part1 (a : ℝ) : (setA a ∪ setB = setB) ↔ (-(1 / 2) ≤ a) :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, x ∈ setB ↔ x ∈ setA a) ↔ (a ≤ -1) :=
sorry

end part1_part2_l289_28990


namespace jelly_bean_remaining_l289_28946

theorem jelly_bean_remaining (J : ℕ) (P : ℕ) (taken_last_4_each : ℕ) (taken_first_each : ℕ) 
 (taken_last_total : ℕ) (taken_first_total : ℕ) (taken_total : ℕ) (remaining : ℕ) :
  J = 8000 →
  P = 10 →
  taken_last_4_each = 400 →
  taken_first_each = 2 * taken_last_4_each →
  taken_last_total = 4 * taken_last_4_each →
  taken_first_total = 6 * taken_first_each →
  taken_total = taken_last_total + taken_first_total →
  remaining = J - taken_total →
  remaining = 1600 :=
by
  intros
  sorry  

end jelly_bean_remaining_l289_28946


namespace parking_monthly_charge_l289_28917

theorem parking_monthly_charge :
  ∀ (M : ℕ), (52 * 10 - 12 * M = 100) → M = 35 :=
by
  intro M h
  sorry

end parking_monthly_charge_l289_28917


namespace solve_equations_l289_28956

theorem solve_equations (x : ℝ) :
  (3 * x^2 = 27 → x = 3 ∨ x = -3) ∧
  (2 * x^2 + x = 55 → x = 5 ∨ x = -5.5) ∧
  (2 * x^2 + 18 = 15 * x → x = 6 ∨ x = 1.5) :=
by
  sorry

end solve_equations_l289_28956


namespace find_a_for_perfect_square_trinomial_l289_28963

theorem find_a_for_perfect_square_trinomial (a : ℝ) :
  (∃ b : ℝ, x^2 - 8*x + a = (x - b)^2) ↔ a = 16 :=
by sorry

end find_a_for_perfect_square_trinomial_l289_28963


namespace simplify_expression_l289_28967

theorem simplify_expression :
  (∃ (a b c d : ℝ), 
   a = 14 * Real.sqrt 2 ∧ 
   b = 12 * Real.sqrt 2 ∧ 
   c = 8 * Real.sqrt 2 ∧ 
   d = 12 * Real.sqrt 2 ∧ 
   ((a / b) + (c / d) = 11 / 6)) :=
by 
  use 14 * Real.sqrt 2, 12 * Real.sqrt 2, 8 * Real.sqrt 2, 12 * Real.sqrt 2
  simp
  sorry

end simplify_expression_l289_28967


namespace log_sqrt2_bounds_l289_28929

theorem log_sqrt2_bounds :
  10^3 = 1000 →
  10^4 = 10000 →
  2^11 = 2048 →
  2^12 = 4096 →
  2^13 = 8192 →
  2^14 = 16384 →
  3 / 22 < Real.log 2 / Real.log 10 / 2 ∧ Real.log 2 / Real.log 10 / 2 < 1 / 7 :=
by
  sorry

end log_sqrt2_bounds_l289_28929


namespace part1_l289_28930

theorem part1 (f : ℝ → ℝ) (m n : ℝ) (cond1 : m + n > 0) (cond2 : ∀ x, f x = |x - m| + |x + n|) (cond3 : ∀ x, f x ≥ m + n) (minimum : ∃ x, f x = 2) :
    m + n = 2 := sorry

end part1_l289_28930


namespace initial_water_percentage_l289_28905

noncomputable def initial_percentage_of_water : ℚ :=
  20

theorem initial_water_percentage
  (initial_volume : ℚ := 125)
  (added_water : ℚ := 8.333333333333334)
  (final_volume : ℚ := initial_volume + added_water)
  (desired_percentage : ℚ := 25)
  (desired_amount_of_water : ℚ := desired_percentage / 100 * final_volume)
  (initial_amount_of_water : ℚ := desired_amount_of_water - added_water) :
  (initial_amount_of_water / initial_volume * 100 = initial_percentage_of_water) :=
by
  sorry

end initial_water_percentage_l289_28905


namespace coprime_gcd_l289_28950

theorem coprime_gcd (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (2 * a + b) (a * (a + b)) = 1 := 
sorry

end coprime_gcd_l289_28950


namespace original_six_digit_number_l289_28908

theorem original_six_digit_number :
  ∃ a b c d e : ℕ, 
  (100000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e = 142857) ∧ 
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + 1 = 64 * (100000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e)) :=
by
  sorry

end original_six_digit_number_l289_28908


namespace distance_from_A_to_y_axis_is_2_l289_28928

-- Define the point A
def point_A : ℝ × ℝ := (-2, 1)

-- Define the distance function to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

-- The theorem to prove
theorem distance_from_A_to_y_axis_is_2 : distance_to_y_axis point_A = 2 :=
by
  sorry

end distance_from_A_to_y_axis_is_2_l289_28928


namespace logan_television_hours_l289_28985

-- Definitions
def minutes_in_an_hour : ℕ := 60
def logan_minutes_watched : ℕ := 300
def logan_hours_watched : ℕ := logan_minutes_watched / minutes_in_an_hour

-- Theorem statement
theorem logan_television_hours : logan_hours_watched = 5 := by
  sorry

end logan_television_hours_l289_28985


namespace dara_jane_age_ratio_l289_28961

theorem dara_jane_age_ratio :
  ∀ (min_age : ℕ) (jane_current_age : ℕ) (dara_years_til_min_age : ℕ) (d : ℕ) (j : ℕ),
  min_age = 25 →
  jane_current_age = 28 →
  dara_years_til_min_age = 14 →
  d = 17 →
  j = 34 →
  d = dara_years_til_min_age - 14 + 6 →
  j = jane_current_age + 6 →
  (d:ℚ) / j = 1 / 2 := 
by
  intros
  sorry

end dara_jane_age_ratio_l289_28961


namespace positive_difference_between_two_numbers_l289_28978

variable (x y : ℝ)

theorem positive_difference_between_two_numbers 
  (h₁ : x + y = 40)
  (h₂ : 3 * y - 4 * x = 20) :
  |y - x| = 100 / 7 :=
by
  sorry

end positive_difference_between_two_numbers_l289_28978


namespace line_intersects_x_axis_at_l289_28943

theorem line_intersects_x_axis_at (a b : ℝ) (h1 : a = 12) (h2 : b = 2)
  (c d : ℝ) (h3 : c = 6) (h4 : d = 6) : 
  ∃ x : ℝ, (x, 0) = (15, 0) := 
by
  -- proof needed here
  sorry

end line_intersects_x_axis_at_l289_28943


namespace sum_of_reciprocals_of_roots_l289_28960

theorem sum_of_reciprocals_of_roots (r1 r2 : ℚ) (h_sum : r1 + r2 = 17) (h_prod : r1 * r2 = 6) :
  1 / r1 + 1 / r2 = 17 / 6 :=
sorry

end sum_of_reciprocals_of_roots_l289_28960


namespace nine_chapters_problem_l289_28919

variable (m n : ℕ)

def horses_condition_1 : Prop := m + n = 100
def horses_condition_2 : Prop := 3 * m + n / 3 = 100

theorem nine_chapters_problem (h1 : horses_condition_1 m n) (h2 : horses_condition_2 m n) :
  (m + n = 100 ∧ 3 * m + n / 3 = 100) :=
by
  exact ⟨h1, h2⟩

end nine_chapters_problem_l289_28919


namespace square_area_from_isosceles_triangle_l289_28969

theorem square_area_from_isosceles_triangle:
  ∀ (b h : ℝ) (Side_of_Square : ℝ), b = 2 ∧ h = 3 ∧ Side_of_Square = (6 / 5) 
  → (Side_of_Square ^ 2) = (36 / 25) := 
by
  intro b h Side_of_Square
  rintro ⟨hb, hh, h_side⟩
  sorry

end square_area_from_isosceles_triangle_l289_28969


namespace time_to_fill_tank_with_leak_l289_28995

-- Definitions based on the given conditions:
def rate_of_pipe_A := 1 / 6 -- Pipe A fills the tank in 6 hours
def rate_of_leak := 1 / 12 -- The leak empties the tank in 12 hours
def combined_rate := rate_of_pipe_A - rate_of_leak -- Combined rate with leak

-- The proof problem: Prove the time taken to fill the tank with the leak present is 12 hours.
theorem time_to_fill_tank_with_leak : 
  (1 / combined_rate) = 12 := by
    -- Proof goes here...
    sorry

end time_to_fill_tank_with_leak_l289_28995


namespace SunshinePumpkinsCount_l289_28971

def MoonglowPumpkins := 14
def SunshinePumpkins := 3 * MoonglowPumpkins + 12

theorem SunshinePumpkinsCount : SunshinePumpkins = 54 :=
by
  -- proof goes here
  sorry

end SunshinePumpkinsCount_l289_28971


namespace find_actual_average_height_l289_28958

noncomputable def actualAverageHeight (avg_height : ℕ) (num_boys : ℕ) (wrong_height : ℕ) (actual_height : ℕ) : Float :=
  let incorrect_total := avg_height * num_boys
  let difference := wrong_height - actual_height
  let correct_total := incorrect_total - difference
  (Float.ofInt correct_total) / (Float.ofNat num_boys)

theorem find_actual_average_height (avg_height num_boys wrong_height actual_height : ℕ) :
  avg_height = 185 ∧ num_boys = 35 ∧ wrong_height = 166 ∧ actual_height = 106 →
  actualAverageHeight avg_height num_boys wrong_height actual_height = 183.29 := by
  intros h
  have h_avg := h.1
  have h_num := h.2.1
  have h_wrong := h.2.2.1
  have h_actual := h.2.2.2
  rw [h_avg, h_num, h_wrong, h_actual]
  sorry

end find_actual_average_height_l289_28958


namespace smallest_positive_period_of_f_minimum_value_of_f_in_interval_l289_28942

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x), Real.sin (2 * x))
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3, 1)
noncomputable def f (x m : ℝ) : ℝ := (vec_a x).1 * vec_b.1 + (vec_a x).2 * vec_b.2 + m

theorem smallest_positive_period_of_f :
  ∀ (x : ℝ) (m : ℝ), ∀ p : ℝ, p > 0 → (∀ x : ℝ, f (x + p) m = f x m) → p = Real.pi := 
sorry

theorem minimum_value_of_f_in_interval :
  ∀ (x m : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) → ∃ m : ℝ, (∀ x : ℝ, f x m ≥ 5) ∧ m = 5 + Real.sqrt 3 :=
sorry

end smallest_positive_period_of_f_minimum_value_of_f_in_interval_l289_28942


namespace total_yardage_progress_l289_28983

def teamA_moves : List Int := [-5, 8, -3, 6]
def teamB_moves : List Int := [4, -2, 9, -7]

theorem total_yardage_progress :
  (teamA_moves.sum + teamB_moves.sum) = 10 :=
by
  sorry

end total_yardage_progress_l289_28983


namespace game_ends_in_65_rounds_l289_28951

noncomputable def player_tokens_A : Nat := 20
noncomputable def player_tokens_B : Nat := 19
noncomputable def player_tokens_C : Nat := 18
noncomputable def player_tokens_D : Nat := 17

def rounds_until_game_ends (A B C D : Nat) : Nat :=
  -- Implementation to count the rounds will go here, but it is skipped for this statement-only task
  sorry

theorem game_ends_in_65_rounds : rounds_until_game_ends player_tokens_A player_tokens_B player_tokens_C player_tokens_D = 65 :=
  sorry

end game_ends_in_65_rounds_l289_28951


namespace condition_1_condition_2_l289_28906

theorem condition_1 (m : ℝ) : (m^2 - 2*m - 15 > 0) ↔ (m < -3 ∨ m > 5) :=
sorry

theorem condition_2 (m : ℝ) : (2*m^2 + 3*m - 9 = 0) ∧ (7*m + 21 ≠ 0) ↔ (m = 3/2) :=
sorry

end condition_1_condition_2_l289_28906


namespace probability_of_no_shaded_square_l289_28931

noncomputable def rectangles_without_shaded_square_probability : ℚ :=
  let n := 502 * 1003
  let m := 502 ^ 2
  1 - (m : ℚ) / n 

theorem probability_of_no_shaded_square : rectangles_without_shaded_square_probability = 501 / 1003 :=
  sorry

end probability_of_no_shaded_square_l289_28931


namespace S7_eq_14_l289_28999

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variables (a : ℕ → ℤ) (S : ℕ → ℤ) (h_arith_seq : arithmetic_sequence a)
variables (h_a3 : a 3 = 0) (h_a6_plus_a7 : a 6 + a 7 = 14)

theorem S7_eq_14 : S 7 = 14 := sorry

end S7_eq_14_l289_28999


namespace fill_cistern_l289_28994

theorem fill_cistern (p_rate q_rate : ℝ) (total_time first_pipe_time : ℝ) (remaining_fraction : ℝ): 
  p_rate = 1/12 → q_rate = 1/15 → total_time = 2 → remaining_fraction = 7/10 → 
  (remaining_fraction / q_rate) = 10.5 :=
by
  sorry

end fill_cistern_l289_28994


namespace emily_weight_l289_28945

theorem emily_weight (H_weight : ℝ) (difference : ℝ) (h : H_weight = 87) (d : difference = 78) : 
  ∃ E_weight : ℝ, E_weight = 9 := 
by
  sorry

end emily_weight_l289_28945


namespace percentage_problem_l289_28918

theorem percentage_problem (X : ℝ) (h : 0.28 * X + 0.45 * 250 = 224.5) : X = 400 :=
sorry

end percentage_problem_l289_28918


namespace median_ratio_within_bounds_l289_28933

def median_ratio_limits (α : ℝ) (hα : 0 < α ∧ α < π) : Prop :=
  ∀ (s_c s_b : ℝ), s_b = 1 → (1 / 2) ≤ (s_c / s_b) ∧ (s_c / s_b) ≤ 2

theorem median_ratio_within_bounds (α : ℝ) (hα : 0 < α ∧ α < π) : 
  median_ratio_limits α hα :=
by
  sorry

end median_ratio_within_bounds_l289_28933


namespace number_of_female_democrats_l289_28932

variables (F M D_f : ℕ)

def total_participants := F + M = 660
def female_democrats := D_f = F / 2
def male_democrats := (F / 2) + (M / 4) = 220

theorem number_of_female_democrats 
  (h1 : total_participants F M) 
  (h2 : female_democrats F D_f) 
  (h3 : male_democrats F M) : 
  D_f = 110 := by
  sorry

end number_of_female_democrats_l289_28932


namespace find_angle_C_find_side_c_l289_28941

noncomputable def triangle_angle_C (a b c : ℝ) (C : ℝ) (A : ℝ) : Prop := 
a * Real.cos C = c * Real.sin A

theorem find_angle_C (a b c : ℝ) (C : ℝ) (A : ℝ)
  (h1 : triangle_angle_C a b c C A)
  (h2 : 0 < A) : C = Real.pi / 3 := 
sorry

noncomputable def triangle_side_c (a b c : ℝ) (C : ℝ) : Prop := 
(∃ (area : ℝ), area = 6 ∧ b = 4 ∧ c * c = a * a + b * b - 2 * a * b * Real.cos C)

theorem find_side_c (a b c : ℝ) (C : ℝ) 
  (h1 : triangle_side_c a b c C) : c = 2 * Real.sqrt 7 :=
sorry

end find_angle_C_find_side_c_l289_28941


namespace find_a_b_l289_28914

def z := Complex.ofReal 3 + Complex.I * 4
def z_conj := Complex.ofReal 3 - Complex.I * 4

theorem find_a_b 
  (a b : ℝ) 
  (h : z + Complex.ofReal a * z_conj + Complex.I * b = Complex.ofReal 9) : 
  a = 2 ∧ b = 4 := 
by 
  sorry

end find_a_b_l289_28914


namespace min_pos_int_k_l289_28939

noncomputable def minimum_k (x0 : ℝ) : ℝ := (x0 * (Real.log x0 + 1)) / (x0 - 2)

theorem min_pos_int_k : ∃ k : ℝ, (∀ x0 : ℝ, x0 > 2 → k > minimum_k x0) ∧ k = 5 := 
by
  sorry

end min_pos_int_k_l289_28939


namespace distance_between_centers_of_intersecting_circles_l289_28922

theorem distance_between_centers_of_intersecting_circles
  {r R d : ℝ} (hrR : r < R) (hr : 0 < r) (hR : 0 < R)
  (h_intersect : d < r + R ∧ d > R - r) :
  R - r < d ∧ d < r + R := by
  sorry

end distance_between_centers_of_intersecting_circles_l289_28922


namespace milk_percentage_after_adding_water_l289_28949

theorem milk_percentage_after_adding_water
  (initial_total_volume : ℚ) (initial_milk_percentage : ℚ)
  (additional_water_volume : ℚ) :
  initial_total_volume = 60 → initial_milk_percentage = 0.84 → additional_water_volume = 18.75 →
  (50.4 / (initial_total_volume + additional_water_volume) * 100 = 64) :=
by
  intros h1 h2 h3
  rw [h1, h3]
  simp
  sorry

end milk_percentage_after_adding_water_l289_28949


namespace totalMoney_l289_28988

noncomputable def totalAmount (x : ℝ) : ℝ := 15 * x

theorem totalMoney (x : ℝ) (h : 1.8 * x = 9) : totalAmount x = 75 :=
by sorry

end totalMoney_l289_28988


namespace find_radius_l289_28934

theorem find_radius (r : ℝ) :
  (135 * r * Real.pi) / 180 = 3 * Real.pi → r = 4 :=
by
  sorry

end find_radius_l289_28934


namespace commute_time_absolute_difference_l289_28921

theorem commute_time_absolute_difference 
  (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) :
  |x - y| = 4 :=
by sorry

end commute_time_absolute_difference_l289_28921


namespace net_rate_25_dollars_per_hour_l289_28962

noncomputable def net_rate_of_pay (hours : ℕ) (speed : ℕ) (mileage : ℕ) (rate_per_mile : ℚ) (diesel_cost_per_gallon : ℚ) : ℚ :=
  let distance := hours * speed
  let diesel_used := distance / mileage
  let earnings := rate_per_mile * distance
  let diesel_cost := diesel_cost_per_gallon * diesel_used
  let net_earnings := earnings - diesel_cost
  net_earnings / hours

theorem net_rate_25_dollars_per_hour :
  net_rate_of_pay 4 45 15 (0.75 : ℚ) (3.00 : ℚ) = 25 :=
by
  -- Proof is omitted
  sorry

end net_rate_25_dollars_per_hour_l289_28962


namespace calc_g_f_neg_2_l289_28986

def f (x : ℝ) : ℝ := x^3 - 4 * x + 3
def g (x : ℝ) : ℝ := 2 * x^2 + 2 * x + 1

theorem calc_g_f_neg_2 : g (f (-2)) = 25 := by
  sorry

end calc_g_f_neg_2_l289_28986


namespace bruce_can_buy_11_bags_l289_28925

-- Defining the total initial amount
def initial_amount : ℕ := 200

-- Defining the quantities and prices of items
def packs_crayons   : ℕ := 5
def price_crayons   : ℕ := 5
def total_crayons   : ℕ := packs_crayons * price_crayons

def books          : ℕ := 10
def price_books    : ℕ := 5
def total_books    : ℕ := books * price_books

def calculators    : ℕ := 3
def price_calc     : ℕ := 5
def total_calc     : ℕ := calculators * price_calc

-- Total cost of all items
def total_cost : ℕ := total_crayons + total_books + total_calc

-- Calculating the change Bruce will have after buying the items
def change : ℕ := initial_amount - total_cost

-- Cost of each bag
def price_bags : ℕ := 10

-- Number of bags Bruce can buy with the change
def num_bags : ℕ := change / price_bags

-- Proposition stating the main problem
theorem bruce_can_buy_11_bags : num_bags = 11 := by
  sorry

end bruce_can_buy_11_bags_l289_28925


namespace find_a_b_sum_l289_28901

def star (a b : ℕ) : ℕ := a^b + a * b

theorem find_a_b_sum (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 24) : a + b = 6 :=
  sorry

end find_a_b_sum_l289_28901


namespace max_type_a_workers_l289_28948

theorem max_type_a_workers (x y : ℕ) (h1 : x + y = 150) (h2 : y ≥ 3 * x) : x ≤ 37 :=
sorry

end max_type_a_workers_l289_28948


namespace min_value_f_l289_28992

noncomputable def f (x : ℝ) : ℝ := (x^2 - x + 3) / (x - 1)

theorem min_value_f : ∀ (x : ℝ), x ≥ 3 → ∃ m : ℝ, m = 9/2 ∧ ∀ y : ℝ, f y ≥ m :=
by
  sorry

end min_value_f_l289_28992


namespace sum_of_second_and_third_of_four_consecutive_even_integers_l289_28991

-- Definitions of conditions
variables (n : ℤ)  -- Assume n is an integer

-- Statement of problem
theorem sum_of_second_and_third_of_four_consecutive_even_integers (h : 2 * n + 6 = 160) :
  (n + 2) + (n + 4) = 160 :=
by
  sorry

end sum_of_second_and_third_of_four_consecutive_even_integers_l289_28991


namespace participants_in_sports_activities_l289_28996

theorem participants_in_sports_activities:
  ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x + y + z = 3 ∧
  let a := 10 * x + 6
  let b := 10 * y + 6
  let c := 10 * z + 6
  a + b + c = 48 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a = 6 ∧ b = 16 ∧ c = 26 ∨ a = 6 ∧ b = 26 ∧ c = 16 ∨ a = 16 ∧ b = 6 ∧ c = 26 ∨ a = 16 ∧ b = 26 ∧ c = 6 ∨ a = 26 ∧ b = 6 ∧ c = 16 ∨ a = 26 ∧ b = 16 ∧ c = 6)
  :=
by {
  sorry
}

end participants_in_sports_activities_l289_28996


namespace functional_eq_is_odd_function_l289_28907

theorem functional_eq_is_odd_function (f : ℝ → ℝ)
  (hf_nonzero : ∃ x : ℝ, f x ≠ 0)
  (hf_eq : ∀ a b : ℝ, f (a * b) = a * f b + b * f a) :
  ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end functional_eq_is_odd_function_l289_28907


namespace alice_bob_meet_after_six_turns_l289_28959

/-
Alice and Bob play a game involving a circle whose circumference
is divided by 12 equally-spaced points. The points are numbered
clockwise, from 1 to 12. Both start on point 12. Alice moves clockwise
and Bob, counterclockwise. In a turn of the game, Alice moves 5 points 
clockwise and Bob moves 9 points counterclockwise. The game ends when they stop on
the same point. 
-/
theorem alice_bob_meet_after_six_turns (k : ℕ) :
  (5 * k) % 12 = (12 - (9 * k) % 12) % 12 -> k = 6 :=
by
  sorry

end alice_bob_meet_after_six_turns_l289_28959


namespace evaluate_expression_l289_28910

theorem evaluate_expression :
  (↑(2 ^ (6 / 4))) ^ 8 = 4096 :=
by sorry

end evaluate_expression_l289_28910


namespace sequence_solution_l289_28973

theorem sequence_solution :
  ∃ (a : ℕ → ℕ) (b : ℕ → ℝ),
    a 1 = 2 ∧
    (∀ n, b n = (a (n + 1)) / (a n)) ∧
    b 10 * b 11 = 2 →
    a 21 = 2 ^ 11 :=
by
  sorry

end sequence_solution_l289_28973


namespace rectangular_solid_surface_area_l289_28966

noncomputable def is_prime (n : ℕ) : Prop := sorry

theorem rectangular_solid_surface_area (l w h : ℕ) (hl : is_prime l) (hw : is_prime w) (hh : is_prime h) (volume_eq_437 : l * w * h = 437) :
  2 * (l * w + w * h + h * l) = 958 :=
sorry

end rectangular_solid_surface_area_l289_28966
