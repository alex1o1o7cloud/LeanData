import Mathlib

namespace NUMINAMATH_GPT_series_items_increase_l1809_180936

theorem series_items_increase (n : ℕ) (hn : n ≥ 2) :
  (2^n + 1) - 2^(n-1) - 1 = 2^(n-1) :=
by
  sorry

end NUMINAMATH_GPT_series_items_increase_l1809_180936


namespace NUMINAMATH_GPT_grocer_initial_stock_l1809_180947

theorem grocer_initial_stock 
  (x : ℝ) 
  (h1 : 0.20 * x + 70 = 0.30 * (x + 100)) : 
  x = 400 := by
  sorry

end NUMINAMATH_GPT_grocer_initial_stock_l1809_180947


namespace NUMINAMATH_GPT_roots_exist_range_k_l1809_180904

theorem roots_exist_range_k (k : ℝ) : 
  (∃ x1 x2 : ℝ, (2 * k * x1^2 + (8 * k + 1) * x1 + 8 * k = 0) ∧ 
                 (2 * k * x2^2 + (8 * k + 1) * x2 + 8 * k = 0)) ↔ 
  (k ≥ -1/16 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_GPT_roots_exist_range_k_l1809_180904


namespace NUMINAMATH_GPT_corrected_mean_is_40_point_6_l1809_180941

theorem corrected_mean_is_40_point_6 
  (mean_original : ℚ) (num_observations : ℕ) (wrong_observation : ℚ) (correct_observation : ℚ) :
  num_observations = 50 → mean_original = 40 → wrong_observation = 15 → correct_observation = 45 →
  ((mean_original * num_observations + (correct_observation - wrong_observation)) / num_observations = 40.6 : Prop) :=
by intros; sorry

end NUMINAMATH_GPT_corrected_mean_is_40_point_6_l1809_180941


namespace NUMINAMATH_GPT_range_of_z_l1809_180975

theorem range_of_z (x y : ℝ) (h1 : -4 ≤ x - y ∧ x - y ≤ -1) (h2 : -1 ≤ 4 * x - y ∧ 4 * x - y ≤ 5) :
  ∃ (z : ℝ), z = 9 * x - y ∧ -1 ≤ z ∧ z ≤ 20 :=
sorry

end NUMINAMATH_GPT_range_of_z_l1809_180975


namespace NUMINAMATH_GPT_diminishing_allocation_proof_l1809_180922

noncomputable def diminishing_allocation_problem : Prop :=
  ∃ (a b m : ℝ), 
  a = 0.2 ∧
  b * (1 - a)^2 = 80 ∧
  b * (1 - a) + b * (1 - a)^3 = 164 ∧
  b + 80 + 164 = m ∧
  m = 369

theorem diminishing_allocation_proof : diminishing_allocation_problem :=
by
  sorry

end NUMINAMATH_GPT_diminishing_allocation_proof_l1809_180922


namespace NUMINAMATH_GPT_total_animals_l1809_180921

def pigs : ℕ := 10

def cows : ℕ := 2 * pigs - 3

def goats : ℕ := cows + 6

theorem total_animals : pigs + cows + goats = 50 := by
  sorry

end NUMINAMATH_GPT_total_animals_l1809_180921


namespace NUMINAMATH_GPT_intersection_A_B_l1809_180938

-- Definition of sets A and B
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y > 0 }

-- The proof goal
theorem intersection_A_B : A ∩ B = { x | x > 1 } :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_l1809_180938


namespace NUMINAMATH_GPT_smallest_distance_l1809_180985

noncomputable def a : Complex := 2 + 4 * Complex.I
noncomputable def b : Complex := 5 + 2 * Complex.I

theorem smallest_distance 
  (z w : Complex) 
  (hz : Complex.abs (z - a) = 2) 
  (hw : Complex.abs (w - b) = 4) : 
  Complex.abs (z - w) ≥ 6 - Real.sqrt 13 :=
sorry

end NUMINAMATH_GPT_smallest_distance_l1809_180985


namespace NUMINAMATH_GPT_reflection_line_slope_intercept_l1809_180993

theorem reflection_line_slope_intercept (m b : ℝ) :
  let P1 := (2, 3)
  let P2 := (10, 7)
  let midpoint := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)
  midpoint = (6, 5) ∧
  ∃(m b : ℝ), 
    m = -2 ∧
    b = 17 ∧
    P2 = (2 * midpoint.1 - P1.1, 2 * midpoint.2 - P1.2)
→ m + b = 15 := by
  intros
  sorry

end NUMINAMATH_GPT_reflection_line_slope_intercept_l1809_180993


namespace NUMINAMATH_GPT_values_of_z_l1809_180994

theorem values_of_z (x z : ℝ) 
  (h1 : 3 * x^2 + 9 * x + 7 * z + 2 = 0)
  (h2 : 3 * x + z + 4 = 0) : 
  z^2 + 20 * z - 14 = 0 := 
sorry

end NUMINAMATH_GPT_values_of_z_l1809_180994


namespace NUMINAMATH_GPT_steve_needs_28_feet_of_wood_l1809_180908

theorem steve_needs_28_feet_of_wood :
  (6 * 4) + (2 * 2) = 28 := by
  sorry

end NUMINAMATH_GPT_steve_needs_28_feet_of_wood_l1809_180908


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1809_180943

-- Define conditions P and Q
def P (x : ℝ) : Prop := x < 1
def Q (x : ℝ) : Prop := (x + 2) * (x - 1) < 0

-- Statement to prove
theorem necessary_but_not_sufficient (x : ℝ) : P x → Q x ∧ ¬ (Q x → P x) :=
by {
  sorry
}

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1809_180943


namespace NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l1809_180949

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1 / 3) :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l1809_180949


namespace NUMINAMATH_GPT_no_n_exists_l1809_180925

theorem no_n_exists (n : ℕ) : ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_no_n_exists_l1809_180925


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_m_l1809_180900

noncomputable def f (x : ℝ) : ℝ := abs (x + 2) * abs (x - 3)

theorem part1_solution_set :
  {x : ℝ | f x > 7 - x} = {x : ℝ | x < -6 ∨ x > 2} :=
sorry

theorem part2_range_of_m (m : ℝ) :
  (∃ x : ℝ, f x ≤ abs (3 * m - 2)) → m ∈ Set.Iic (-1) ∪ Set.Ici (7 / 3) :=
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_m_l1809_180900


namespace NUMINAMATH_GPT_find_k_l1809_180933

-- Define the number and compute the sum of its digits
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the problem
theorem find_k :
  ∃ k : ℕ, sum_of_digits (9 * (10^k - 1)) = 1111 ∧ k = 124 :=
sorry

end NUMINAMATH_GPT_find_k_l1809_180933


namespace NUMINAMATH_GPT_same_type_as_target_l1809_180945

-- Definitions of the polynomials
def optionA (a b : ℝ) : ℝ := a^2 * b
def optionB (a b : ℝ) : ℝ := -2 * a * b^2
def optionC (a b : ℝ) : ℝ := a * b
def optionD (a b c : ℝ) : ℝ := a * b^2 * c

-- Definition of the target polynomial type
def target (a b : ℝ) : ℝ := a * b^2

-- Statement: Option B is of the same type as target
theorem same_type_as_target (a b : ℝ) : optionB a b = -2 * target a b := 
sorry

end NUMINAMATH_GPT_same_type_as_target_l1809_180945


namespace NUMINAMATH_GPT_total_books_l1809_180978

theorem total_books (D Loris Lamont : ℕ) 
  (h1 : Loris + 3 = Lamont)
  (h2 : Lamont = 2 * D)
  (h3 : D = 20) : D + Loris + Lamont = 97 := 
by 
  sorry

end NUMINAMATH_GPT_total_books_l1809_180978


namespace NUMINAMATH_GPT_weight_of_new_person_l1809_180946

theorem weight_of_new_person 
  (avg_weight_increase : ℝ)
  (old_weight : ℝ) 
  (num_people : ℕ)
  (new_weight_increase : ℝ)
  (total_weight_increase : ℝ)  
  (W : ℝ)
  (h1 : avg_weight_increase = 1.8)
  (h2 : old_weight = 69)
  (h3 : num_people = 6) 
  (h4 : new_weight_increase = num_people * avg_weight_increase) 
  (h5 : total_weight_increase = new_weight_increase)
  (h6 : W = old_weight + total_weight_increase)
  : W = 79.8 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_new_person_l1809_180946


namespace NUMINAMATH_GPT_region_area_l1809_180964

-- Let x and y be real numbers
variables (x y : ℝ)

-- Define the inequality condition
def region_condition (x y : ℝ) : Prop := abs (4 * x - 20) + abs (3 * y + 9) ≤ 6

-- The statement that needs to be proved
theorem region_area : (∃ x y : ℝ, region_condition x y) → ∃ A : ℝ, A = 6 :=
by
  sorry

end NUMINAMATH_GPT_region_area_l1809_180964


namespace NUMINAMATH_GPT_cone_water_volume_percentage_l1809_180915

theorem cone_water_volume_percentage
  (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let full_volume := (1 / 3) * π * r^2 * h
  let water_height := (2 / 3) * h
  let water_radius := (2 / 3) * r
  let water_volume := (1 / 3) * π * water_radius^2 * water_height
  let percentage := (water_volume / full_volume) * 100
  abs (percentage - 29.6296) < 0.0001 :=
by
  let full_volume := (1 / 3) * π * r^2 * h
  let water_height := (2 / 3) * h
  let water_radius := (2 / 3) * r
  let water_volume := (1 / 3) * π * water_radius^2 * water_height
  let percentage := (water_volume / full_volume) * 100
  sorry

end NUMINAMATH_GPT_cone_water_volume_percentage_l1809_180915


namespace NUMINAMATH_GPT_product_of_two_numbers_l1809_180986

-- Define HCF function
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM function
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Define the conditions for the problem
def problem_conditions (x y : ℕ) : Prop :=
  HCF x y = 55 ∧ LCM x y = 1500

-- State the theorem that should be proven
theorem product_of_two_numbers (x y : ℕ) (h_conditions : problem_conditions x y) :
  x * y = 82500 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1809_180986


namespace NUMINAMATH_GPT_compute_c_plus_d_l1809_180932

variable {c d : ℝ}

-- Define the given polynomial equations
def poly_c (c : ℝ) := c^3 - 21*c^2 + 28*c - 70
def poly_d (d : ℝ) := 10*d^3 - 75*d^2 - 350*d + 3225

theorem compute_c_plus_d (hc : poly_c c = 0) (hd : poly_d d = 0) : c + d = 21 / 2 := sorry

end NUMINAMATH_GPT_compute_c_plus_d_l1809_180932


namespace NUMINAMATH_GPT_least_trees_l1809_180968

theorem least_trees (N : ℕ) (h1 : N % 7 = 0) (h2 : N % 6 = 0) (h3 : N % 4 = 0) (h4 : N ≥ 100) : N = 168 :=
sorry

end NUMINAMATH_GPT_least_trees_l1809_180968


namespace NUMINAMATH_GPT_remainingAreaCalculation_l1809_180962

noncomputable def totalArea : ℝ := 9500.0
noncomputable def lizzieGroupArea : ℝ := 2534.1
noncomputable def hilltownTeamArea : ℝ := 2675.95
noncomputable def greenValleyCrewArea : ℝ := 1847.57

theorem remainingAreaCalculation :
  (totalArea - (lizzieGroupArea + hilltownTeamArea + greenValleyCrewArea) = 2442.38) :=
by
  sorry

end NUMINAMATH_GPT_remainingAreaCalculation_l1809_180962


namespace NUMINAMATH_GPT_sum_of_a_and_b_l1809_180934

-- Define conditions
def population_size : ℕ := 55
def sample_size : ℕ := 5
def interval : ℕ := population_size / sample_size
def sample_indices : List ℕ := [6, 28, 50]

-- Assume a and b are such that the systematic sampling is maintained
variable (a b : ℕ)
axiom a_idx : a = sample_indices.head! + interval
axiom b_idx : b = sample_indices.getLast! - interval

-- Define Lean 4 statement to prove
theorem sum_of_a_and_b :
  (a + b) = 56 :=
by
  -- This will be the place where the proof is inserted
  sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l1809_180934


namespace NUMINAMATH_GPT_train_avg_speed_l1809_180940

variable (x : ℝ)

def avg_speed_of_train (x : ℝ) : ℝ := 3

theorem train_avg_speed (h : x > 0) : avg_speed_of_train x / (x / 7.5) = 22.5 :=
  sorry

end NUMINAMATH_GPT_train_avg_speed_l1809_180940


namespace NUMINAMATH_GPT_tangent_line_eq_at_x_is_1_range_of_sum_extreme_values_l1809_180989

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * Real.log x - x
noncomputable def g (x m : ℝ) : ℝ := f x + m * x^2
noncomputable def tangentLineEq (x y : ℝ) : Prop := x + 2 * y + 1 = 0
noncomputable def rangeCondition (x₁ x₂ m : ℝ) : Prop := g x₁ m + g x₂ m < -3 / 2

theorem tangent_line_eq_at_x_is_1 :
  tangentLineEq 1 (f 1) := 
sorry

theorem range_of_sum_extreme_values (h : 0 < m ∧ m < 1 / 4) (x₁ x₂ : ℝ) :
  rangeCondition x₁ x₂ m := 
sorry

end NUMINAMATH_GPT_tangent_line_eq_at_x_is_1_range_of_sum_extreme_values_l1809_180989


namespace NUMINAMATH_GPT_boys_play_theater_with_Ocho_l1809_180927

variables (Ocho_friends : ℕ) (half_girls : Ocho_friends / 2 = 4)

theorem boys_play_theater_with_Ocho : (Ocho_friends / 2) = 4 := by
  -- Ocho_friends is the total number of Ocho's friends
  -- half_girls is given as a condition that half of Ocho's friends are girls
  -- thus, we directly use this to conclude that the number of boys is 4
  sorry

end NUMINAMATH_GPT_boys_play_theater_with_Ocho_l1809_180927


namespace NUMINAMATH_GPT_sum_solutions_eq_l1809_180980

theorem sum_solutions_eq : 
  let a := 12
  let b := -19
  let c := -21
  (4 * x + 3) * (3 * x - 7) = 0 → (b/a) = 19/12 :=
by
  sorry

end NUMINAMATH_GPT_sum_solutions_eq_l1809_180980


namespace NUMINAMATH_GPT_length_of_tube_l1809_180906

/-- Prove that the length of the tube is 1.5 meters given the initial conditions -/
theorem length_of_tube (h1 : ℝ) (m_water : ℝ) (rho : ℝ) (g : ℝ) (p_ratio : ℝ) :
  h1 = 1.5 ∧ m_water = 1000 ∧ rho = 1000 ∧ g = 9.8 ∧ p_ratio = 2 → 
  ∃ h2 : ℝ, h2 = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_length_of_tube_l1809_180906


namespace NUMINAMATH_GPT_regular_polygon_perimeter_l1809_180996

def exterior_angle (n : ℕ) := 360 / n

theorem regular_polygon_perimeter
  (side_length : ℕ)
  (exterior_angle_deg : ℕ)
  (polygon_perimeter : ℕ)
  (h1 : side_length = 8)
  (h2 : exterior_angle_deg = 72)
  (h3 : ∃ n : ℕ, exterior_angle n = exterior_angle_deg)
  (h4 : ∀ n : ℕ, exterior_angle n = exterior_angle_deg → polygon_perimeter = n * side_length) :
  polygon_perimeter = 40 :=
sorry

end NUMINAMATH_GPT_regular_polygon_perimeter_l1809_180996


namespace NUMINAMATH_GPT_sequence_98th_term_l1809_180979

-- Definitions of the rules
def rule1 (n : ℕ) : ℕ := n * 9
def rule2 (n : ℕ) : ℕ := n / 2
def rule3 (n : ℕ) : ℕ := n - 5

-- Function to compute the next term in the sequence based on the current term
def next_term (n : ℕ) : ℕ :=
  if n < 10 then rule1 n
  else if n % 2 = 0 then rule2 n
  else rule3 n

-- Function to compute the nth term of the sequence starting with the initial term
def nth_term (start : ℕ) (n : ℕ) : ℕ :=
  Nat.iterate next_term n start

-- Theorem to prove that the 98th term of the sequence starting at 98 is 27
theorem sequence_98th_term : nth_term 98 98 = 27 := by
  sorry

end NUMINAMATH_GPT_sequence_98th_term_l1809_180979


namespace NUMINAMATH_GPT_problem_statement_l1809_180990

-- Define what it means for a number's tens and ones digits to have a sum of 13
def sum_of_tens_and_ones_equals (n : ℕ) (s : ℕ) : Prop :=
  let tens_digit := (n / 10) % 10
  let ones_digit := n % 10
  tens_digit + ones_digit = s

-- State the theorem with the given conditions and correct answer
theorem problem_statement : sum_of_tens_and_ones_equals (6^11) 13 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1809_180990


namespace NUMINAMATH_GPT_find_z_l1809_180958

theorem find_z (x y z : ℝ) 
  (h1 : y = 2 * x + 3) 
  (h2 : x + 1 / x = 3.5 + (Real.sin (z * Real.exp (-z)))) :
  z = x^2 + 1 / x^2 := 
sorry

end NUMINAMATH_GPT_find_z_l1809_180958


namespace NUMINAMATH_GPT_leftmost_digit_base9_l1809_180972

theorem leftmost_digit_base9 (x : ℕ) (h : x = 3^19 + 2*3^18 + 1*3^17 + 1*3^16 + 2*3^15 + 2*3^14 + 1*3^13 + 1*3^12 + 1*3^11 + 2*3^10 + 2*3^9 + 2*3^8 + 1*3^7 + 1*3^6 + 1*3^5 + 1*3^4 + 2*3^3 + 2*3^2 + 2*3^1 + 2) : ℕ :=
by
  sorry

end NUMINAMATH_GPT_leftmost_digit_base9_l1809_180972


namespace NUMINAMATH_GPT_total_pencils_l1809_180931

def pencils_in_rainbow_box : ℕ := 7
def total_people : ℕ := 8

theorem total_pencils : pencils_in_rainbow_box * total_people = 56 := by
  sorry

end NUMINAMATH_GPT_total_pencils_l1809_180931


namespace NUMINAMATH_GPT_dot_product_min_value_in_triangle_l1809_180944

noncomputable def dot_product_min_value (a b c : ℝ) (angleA : ℝ) : ℝ :=
  b * c * Real.cos angleA

theorem dot_product_min_value_in_triangle (b c : ℝ) (hyp1 : 0 ≤ b) (hyp2 : 0 ≤ c) 
  (hyp3 : b^2 + c^2 + b * c = 16) (hyp4 : Real.cos (2 * Real.pi / 3) = -1 / 2) : 
  ∃ (p : ℝ), p = dot_product_min_value 4 b c (2 * Real.pi / 3) ∧ p = -8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_min_value_in_triangle_l1809_180944


namespace NUMINAMATH_GPT_height_of_fourth_person_l1809_180991

theorem height_of_fourth_person
  (h : ℝ)
  (cond : (h + (h + 2) + (h + 4) + (h + 10)) / 4 = 79) :
  (h + 10) = 85 :=
by 
  sorry

end NUMINAMATH_GPT_height_of_fourth_person_l1809_180991


namespace NUMINAMATH_GPT_augmented_wedge_volume_proof_l1809_180954

open Real

noncomputable def sphere_radius (circumference : ℝ) : ℝ :=
  circumference / (2 * π)

noncomputable def sphere_volume (r : ℝ) : ℝ :=
  (4/3) * π * r^3

noncomputable def wedge_volume (volume_sphere : ℝ) (number_of_wedges : ℕ) : ℝ :=
  volume_sphere / number_of_wedges

noncomputable def augmented_wedge_volume (original_wedge_volume : ℝ) : ℝ :=
  2 * original_wedge_volume

theorem augmented_wedge_volume_proof (circumference : ℝ) (number_of_wedges : ℕ) 
  (volume : ℝ) (augmented_volume : ℝ) :
  circumference = 18 * π →
  number_of_wedges = 6 →
  volume = sphere_volume (sphere_radius circumference) →
  augmented_volume = augmented_wedge_volume (wedge_volume volume number_of_wedges) →
  augmented_volume = 324 * π :=
by
  intros h_circ h_wedges h_vol h_aug_vol
  -- This is where the proof steps would go
  sorry

end NUMINAMATH_GPT_augmented_wedge_volume_proof_l1809_180954


namespace NUMINAMATH_GPT_find_apron_cost_l1809_180912

-- Definitions used in the conditions
variables (hand_mitts cost small_knife utensils apron : ℝ)
variables (nieces : ℕ)
variables (total_cost_before_discount total_cost_after_discount : ℝ)

-- Conditions given
def conditions := 
  hand_mitts = 14 ∧ 
  utensils = 10 ∧ 
  small_knife = 2 * utensils ∧
  (total_cost_before_discount : ℝ) = (3 * hand_mitts + 3 * utensils + 3 * small_knife + 3 * apron) ∧
  (total_cost_after_discount : ℝ) = 135 ∧
  total_cost_before_discount * 0.75 = total_cost_after_discount ∧
  nieces = 3

-- Theorem statement (proof problem)
theorem find_apron_cost (h : conditions hand_mitts utensils small_knife apron nieces total_cost_before_discount total_cost_after_discount) : 
  apron = 16 :=
by 
  sorry

end NUMINAMATH_GPT_find_apron_cost_l1809_180912


namespace NUMINAMATH_GPT_x_squared_minus_y_squared_l1809_180988

theorem x_squared_minus_y_squared {x y : ℚ} 
    (h1 : x + y = 3/8) 
    (h2 : x - y = 5/24) 
    : x^2 - y^2 = 5/64 := 
by 
    -- The proof would go here
    sorry

end NUMINAMATH_GPT_x_squared_minus_y_squared_l1809_180988


namespace NUMINAMATH_GPT_distance_midpoint_chord_AB_to_y_axis_l1809_180998

theorem distance_midpoint_chord_AB_to_y_axis
  (k : ℝ)
  (A B : ℝ × ℝ)
  (hA : A.2 = k * A.1 - k)
  (hB : B.2 = k * B.1 - k)
  (hA_on_parabola : A.2 ^ 2 = 4 * A.1)
  (hB_on_parabola : B.2 ^ 2 = 4 * B.1)
  (h_distance_AB : dist A B = 4) :
  (abs ((A.1 + B.1) / 2)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_distance_midpoint_chord_AB_to_y_axis_l1809_180998


namespace NUMINAMATH_GPT_length_of_first_platform_l1809_180939

-- Definitions corresponding to conditions
def length_train := 310
def time_first_platform := 15
def length_second_platform := 250
def time_second_platform := 20

-- Time-speed relationship
def speed_first_platform (L : ℕ) : ℚ := (length_train + L) / time_first_platform
def speed_second_platform : ℚ := (length_train + length_second_platform) / time_second_platform

-- Theorem to prove length of first platform
theorem length_of_first_platform (L : ℕ) (h : speed_first_platform L = speed_second_platform) : L = 110 :=
by
  sorry

end NUMINAMATH_GPT_length_of_first_platform_l1809_180939


namespace NUMINAMATH_GPT_problem_solution_l1809_180937

theorem problem_solution (x : ℝ) :
    (x^2 / (x - 2) ≥ (3 / (x + 2)) + (7 / 5)) →
    (x ∈ Set.Ioo (-2 : ℝ) 2 ∪ Set.Ioi (2 : ℝ)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_problem_solution_l1809_180937


namespace NUMINAMATH_GPT_percentage_of_600_equals_150_is_25_l1809_180967

theorem percentage_of_600_equals_150_is_25 : (150 / 600 * 100) = 25 := by
  sorry

end NUMINAMATH_GPT_percentage_of_600_equals_150_is_25_l1809_180967


namespace NUMINAMATH_GPT_kiwis_to_apples_l1809_180911

theorem kiwis_to_apples :
  (1 / 4) * 20 = 10 → (3 / 4) * 12 * (2 / 5) = 18 :=
by
  sorry

end NUMINAMATH_GPT_kiwis_to_apples_l1809_180911


namespace NUMINAMATH_GPT_sum_of_first_n_natural_numbers_l1809_180919

theorem sum_of_first_n_natural_numbers (n : ℕ) (h : n * (n + 1) / 2 = 190) : n = 19 :=
sorry

end NUMINAMATH_GPT_sum_of_first_n_natural_numbers_l1809_180919


namespace NUMINAMATH_GPT_train_speed_l1809_180955

theorem train_speed (length1 length2 speed2 : ℝ) (time_seconds speed1 : ℝ)
    (h_length1 : length1 = 111)
    (h_length2 : length2 = 165)
    (h_speed2 : speed2 = 90)
    (h_time : time_seconds = 6.623470122390208)
    (h_speed1 : speed1 = 60) :
    (length1 / 1000.0) + (length2 / 1000.0) / (time_seconds / 3600) = speed1 + speed2 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1809_180955


namespace NUMINAMATH_GPT_hourly_wage_l1809_180907

theorem hourly_wage (reps : ℕ) (hours_per_day : ℕ) (days : ℕ) (total_payment : ℕ) :
  reps = 50 →
  hours_per_day = 8 →
  days = 5 →
  total_payment = 28000 →
  (total_payment / (reps * hours_per_day * days) : ℕ) = 14 :=
by
  intros h_reps h_hours_per_day h_days h_total_payment
  -- Now the proof steps can be added here
  sorry

end NUMINAMATH_GPT_hourly_wage_l1809_180907


namespace NUMINAMATH_GPT_LCM_of_fractions_l1809_180969

theorem LCM_of_fractions (x : ℕ) (h : x > 0) : 
  lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (9 * x))) = 1 / (36 * x) :=
by
  sorry

end NUMINAMATH_GPT_LCM_of_fractions_l1809_180969


namespace NUMINAMATH_GPT_area_increase_factor_l1809_180956

theorem area_increase_factor (s : ℝ) :
  let A_original := s^2
  let A_new := (3 * s)^2
  A_new / A_original = 9 := by
  sorry

end NUMINAMATH_GPT_area_increase_factor_l1809_180956


namespace NUMINAMATH_GPT_not_divisible_by_4_l1809_180901

theorem not_divisible_by_4 (n : Int) : ¬ (1 + n + n^2 + n^3 + n^4) % 4 = 0 := by
  sorry

end NUMINAMATH_GPT_not_divisible_by_4_l1809_180901


namespace NUMINAMATH_GPT_students_in_both_clubs_l1809_180960

theorem students_in_both_clubs (total_students drama_club art_club drama_or_art in_both_clubs : ℕ)
  (H1 : total_students = 300)
  (H2 : drama_club = 120)
  (H3 : art_club = 150)
  (H4 : drama_or_art = 220) :
  in_both_clubs = drama_club + art_club - drama_or_art :=
by
  -- this is the proof space
  sorry

end NUMINAMATH_GPT_students_in_both_clubs_l1809_180960


namespace NUMINAMATH_GPT_candidates_appeared_l1809_180963

theorem candidates_appeared (x : ℝ) (h1 : 0.07 * x = 0.06 * x + 82) : x = 8200 :=
by
  sorry

end NUMINAMATH_GPT_candidates_appeared_l1809_180963


namespace NUMINAMATH_GPT_abs_diff_of_two_numbers_l1809_180982

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 34) (h2 : x * y = 240) : abs (x - y) = 14 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_of_two_numbers_l1809_180982


namespace NUMINAMATH_GPT_marbles_sum_l1809_180917

variable {K M : ℕ}

theorem marbles_sum (hFabian_kyle : 15 = 3 * K) (hFabian_miles : 15 = 5 * M) :
  K + M = 8 :=
by
  sorry

end NUMINAMATH_GPT_marbles_sum_l1809_180917


namespace NUMINAMATH_GPT_price_difference_correct_l1809_180987

-- Define the list price of Camera Y
def list_price : ℚ := 52.50

-- Define the discount at Mega Deals
def mega_deals_discount : ℚ := 12

-- Define the discount rate at Budget Buys
def budget_buys_discount_rate : ℚ := 0.30

-- Calculate the sale prices
def mega_deals_price : ℚ := list_price - mega_deals_discount
def budget_buys_price : ℚ := (1 - budget_buys_discount_rate) * list_price

-- Calculate the price difference in dollars and convert to cents
def price_difference_in_cents : ℚ := (mega_deals_price - budget_buys_price) * 100

-- Theorem to prove the computed price difference in cents equals 375
theorem price_difference_correct : price_difference_in_cents = 375 := by
  sorry

end NUMINAMATH_GPT_price_difference_correct_l1809_180987


namespace NUMINAMATH_GPT_arith_to_geom_l1809_180976

noncomputable def a (n : ℕ) (d : ℝ) : ℝ := 1 + (n - 1) * d

theorem arith_to_geom (m n : ℕ) (d : ℝ) 
  (h_pos : d > 0)
  (h_arith_seq : ∀ k : ℕ, a k d > 0)
  (h_geo_seq : (a 4 d + 5 / 2)^2 = (a 3 d) * (a 11 d))
  (h_mn : m - n = 8) : 
  a m d - a n d = 12 := 
sorry

end NUMINAMATH_GPT_arith_to_geom_l1809_180976


namespace NUMINAMATH_GPT_num_of_laborers_is_24_l1809_180997

def average_salary_all (L S : Nat) (avg_salary_ls : Nat) (avg_salary_l : Nat) (avg_salary_s : Nat) : Prop :=
  (avg_salary_l * L + avg_salary_s * S) / (L + S) = avg_salary_ls

def average_salary_supervisors (S : Nat) (avg_salary_s : Nat) : Prop :=
  (avg_salary_s * S) / S = avg_salary_s

theorem num_of_laborers_is_24 :
  ∀ (L S : Nat) (avg_salary_ls avg_salary_l avg_salary_s : Nat),
    average_salary_all L S avg_salary_ls avg_salary_l avg_salary_s →
    average_salary_supervisors S avg_salary_s →
    S = 6 → avg_salary_ls = 1250 → avg_salary_l = 950 → avg_salary_s = 2450 →
    L = 24 :=
by
  intros L S avg_salary_ls avg_salary_l avg_salary_s h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_num_of_laborers_is_24_l1809_180997


namespace NUMINAMATH_GPT_anne_carries_16point5_kg_l1809_180918

theorem anne_carries_16point5_kg :
  let w1 := 2
  let w2 := 1.5 * w1
  let w3 := 2 * w1
  let w4 := w1 + w2
  let w5 := (w1 + w2) / 2
  w1 + w2 + w3 + w4 + w5 = 16.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_anne_carries_16point5_kg_l1809_180918


namespace NUMINAMATH_GPT_inequality_solution_intervals_l1809_180913

theorem inequality_solution_intervals (x : ℝ) (h : x > 2) : 
  (x-2)^(x^2 - 6 * x + 8) > 1 ↔ (2 < x ∧ x < 3) ∨ x > 4 := 
sorry

end NUMINAMATH_GPT_inequality_solution_intervals_l1809_180913


namespace NUMINAMATH_GPT_area_of_right_triangle_l1809_180926

theorem area_of_right_triangle (m k : ℝ) (hm : 0 < m) (hk : 0 < k) : 
  ∃ A : ℝ, A = (k^2) / (2 * m) :=
by
  sorry

end NUMINAMATH_GPT_area_of_right_triangle_l1809_180926


namespace NUMINAMATH_GPT_find_t_l1809_180971

theorem find_t : ∀ (p j t x y a b c : ℝ),
  j = 0.75 * p →
  j = 0.80 * t →
  t = p - (t/100) * p →
  x = 0.10 * t →
  y = 0.50 * j →
  x + y = 12 →
  a = x + y →
  b = 0.15 * a →
  c = 2 * b →
  t = 24 := 
by
  intros p j t x y a b c hjp hjt htp hxt hyy hxy ha hb hc
  sorry

end NUMINAMATH_GPT_find_t_l1809_180971


namespace NUMINAMATH_GPT_privateer_overtakes_at_6_08_pm_l1809_180935

noncomputable def time_of_overtake : Bool :=
  let initial_distance := 12 -- miles
  let initial_time := 10 -- 10:00 a.m.
  let privateer_speed_initial := 10 -- mph
  let merchantman_speed := 7 -- mph
  let time_to_sail_initial := 3 -- hours
  let distance_covered_privateer := privateer_speed_initial * time_to_sail_initial
  let distance_covered_merchantman := merchantman_speed * time_to_sail_initial
  let relative_distance_after_three_hours := initial_distance + distance_covered_merchantman - distance_covered_privateer
  let privateer_speed_modified := 13 -- new speed
  let merchantman_speed_modified := 12 -- corresponding merchantman speed

  -- Calculating the new relative speed after the privateer's speed is reduced
  let privateer_new_speed := (13 / 12) * merchantman_speed
  let relative_speed_after_damage := privateer_new_speed - merchantman_speed
  let time_to_overtake_remainder := relative_distance_after_three_hours / relative_speed_after_damage
  let total_time := time_to_sail_initial + time_to_overtake_remainder -- in hours

  let final_time := initial_time + total_time -- converting into the final time of the day
  final_time == 18.1333 -- This should convert to 6:08 p.m., approximately 18 hours and 8 minutes in a 24-hour format

theorem privateer_overtakes_at_6_08_pm : time_of_overtake = true :=
  by
    -- Proof will be provided here
    sorry

end NUMINAMATH_GPT_privateer_overtakes_at_6_08_pm_l1809_180935


namespace NUMINAMATH_GPT_max_cylinder_volume_in_cone_l1809_180973

theorem max_cylinder_volume_in_cone :
  ∃ x, (0 < x ∧ x < 1) ∧ ∀ y, (0 < y ∧ y < 1 → y ≠ x → ((π * (-2 * y^3 + 2 * y^2)) ≤ (π * (-2 * x^3 + 2 * x^2)))) ∧ 
  (π * (-2 * x^3 + 2 * x^2) = 8 * π / 27) := sorry

end NUMINAMATH_GPT_max_cylinder_volume_in_cone_l1809_180973


namespace NUMINAMATH_GPT_find_z2_l1809_180992

theorem find_z2 (z1 z2 : ℂ) (h1 : z1 = 1 - I) (h2 : z1 * z2 = 1 + I) : z2 = I :=
sorry

end NUMINAMATH_GPT_find_z2_l1809_180992


namespace NUMINAMATH_GPT_movie_screening_guests_l1809_180929

theorem movie_screening_guests
  (total_guests : ℕ)
  (women_percentage : ℝ)
  (men_count : ℕ)
  (men_left_fraction : ℝ)
  (children_left_percentage : ℝ)
  (children_count : ℕ)
  (people_left : ℕ) :
  total_guests = 75 →
  women_percentage = 0.40 →
  men_count = 25 →
  men_left_fraction = 1/3 →
  children_left_percentage = 0.20 →
  children_count = total_guests - (round (women_percentage * total_guests) + men_count) →
  people_left = (round (men_left_fraction * men_count)) + (round (children_left_percentage * children_count)) →
  (total_guests - people_left) = 63 :=
by
  intros ht hw hm hf hc hc_count hl
  sorry

end NUMINAMATH_GPT_movie_screening_guests_l1809_180929


namespace NUMINAMATH_GPT_prime_triplets_satisfy_condition_l1809_180910

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_triplets_satisfy_condition :
  ∀ p q r : ℕ,
    is_prime p → is_prime q → is_prime r →
    (p * (r - 1) = q * (r + 7)) →
    (p = 3 ∧ q = 2 ∧ r = 17) ∨ 
    (p = 7 ∧ q = 3 ∧ r = 7) ∨
    (p = 5 ∧ q = 3 ∧ r = 13) :=
by
  sorry

end NUMINAMATH_GPT_prime_triplets_satisfy_condition_l1809_180910


namespace NUMINAMATH_GPT_range_m_l1809_180902

open Set

noncomputable def A : Set ℝ := { x : ℝ | -5 ≤ x ∧ x ≤ 3 }
noncomputable def B (m : ℝ) : Set ℝ := { x : ℝ | m + 1 < x ∧ x < 2 * m + 3 }

theorem range_m (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ↔ m ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_m_l1809_180902


namespace NUMINAMATH_GPT_knives_percentage_l1809_180948

-- Definitions based on conditions
def initial_knives : ℕ := 6
def initial_forks : ℕ := 12
def initial_spoons : ℕ := 3 * initial_knives
def traded_knives : ℕ := 10
def traded_spoons : ℕ := 6

-- Definitions for calculations
def final_knives : ℕ := initial_knives + traded_knives
def final_spoons : ℕ := initial_spoons - traded_spoons
def total_silverware : ℕ := final_knives + final_spoons + initial_forks

-- Theorem to prove the percentage of knives
theorem knives_percentage : (final_knives * 100) / total_silverware = 40 := by
  sorry

end NUMINAMATH_GPT_knives_percentage_l1809_180948


namespace NUMINAMATH_GPT_problem1_problem2_l1809_180909

-- Let's define the first problem statement in Lean
theorem problem1 : 2 - 7 * (-3) + 10 + (-2) = 31 := sorry

-- Let's define the second problem statement in Lean
theorem problem2 : -1^2022 + 24 + (-2)^3 - 3^2 * (-1/3)^2 = 14 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1809_180909


namespace NUMINAMATH_GPT_cost_price_of_table_l1809_180970

theorem cost_price_of_table (SP : ℝ) (CP : ℝ) (h1 : SP = 1.20 * CP) (h2 : SP = 3600) : CP = 3000 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_table_l1809_180970


namespace NUMINAMATH_GPT_solve_for_x_l1809_180914

theorem solve_for_x (x : ℚ) : (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ↔ x = -5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1809_180914


namespace NUMINAMATH_GPT_cos_C_of_triangle_l1809_180930

theorem cos_C_of_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : a = 2)
  (hb : b = 3)
  (hc : c = 4)
  (h_sine_relation : 3 * Real.sin A = 2 * Real.sin B)
  (h_cosine_law : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  Real.cos C = -1/4 :=
by
  sorry

end NUMINAMATH_GPT_cos_C_of_triangle_l1809_180930


namespace NUMINAMATH_GPT_sum_of_factors_636405_l1809_180977

theorem sum_of_factors_636405 :
  ∃ (a b c : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 10 ≤ c ∧ c < 100 ∧
    a * b * c = 636405 ∧ a + b + c = 259 :=
sorry

end NUMINAMATH_GPT_sum_of_factors_636405_l1809_180977


namespace NUMINAMATH_GPT_decrease_in_profit_when_one_loom_idles_l1809_180984

def num_looms : ℕ := 125
def total_sales_value : ℕ := 500000
def total_manufacturing_expenses : ℕ := 150000
def monthly_establishment_charges : ℕ := 75000
def sales_value_per_loom : ℕ := total_sales_value / num_looms
def manufacturing_expense_per_loom : ℕ := total_manufacturing_expenses / num_looms
def decrease_in_sales_value : ℕ := sales_value_per_loom
def decrease_in_manufacturing_expenses : ℕ := manufacturing_expense_per_loom
def net_decrease_in_profit : ℕ := decrease_in_sales_value - decrease_in_manufacturing_expenses

theorem decrease_in_profit_when_one_loom_idles : net_decrease_in_profit = 2800 := by
  sorry

end NUMINAMATH_GPT_decrease_in_profit_when_one_loom_idles_l1809_180984


namespace NUMINAMATH_GPT_circumscribed_sphere_radius_l1809_180981

theorem circumscribed_sphere_radius (a b c : ℝ) : 
  R = (1/2) * Real.sqrt (a^2 + b^2 + c^2) := sorry

end NUMINAMATH_GPT_circumscribed_sphere_radius_l1809_180981


namespace NUMINAMATH_GPT_bruce_purchased_mangoes_l1809_180923

noncomputable def calculate_mango_quantity (grapes_quantity : ℕ) (grapes_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : ℕ :=
  let cost_of_grapes := grapes_quantity * grapes_rate
  let cost_of_mangoes := total_paid - cost_of_grapes
  cost_of_mangoes / mango_rate

theorem bruce_purchased_mangoes :
  calculate_mango_quantity 8 70 55 1055 = 9 :=
by
  sorry

end NUMINAMATH_GPT_bruce_purchased_mangoes_l1809_180923


namespace NUMINAMATH_GPT_price_of_red_car_l1809_180920

noncomputable def car_price (total_amount loan_amount interest_rate : ℝ) : ℝ :=
  loan_amount + (total_amount - loan_amount) / (1 + interest_rate)

theorem price_of_red_car :
  car_price 38000 20000 0.15 = 35000 :=
by sorry

end NUMINAMATH_GPT_price_of_red_car_l1809_180920


namespace NUMINAMATH_GPT_fill_tank_time_l1809_180951

theorem fill_tank_time (t_A t_B : ℕ) (hA : t_A = 20) (hB : t_B = t_A / 4) :
  t_B = 4 := by
  sorry

end NUMINAMATH_GPT_fill_tank_time_l1809_180951


namespace NUMINAMATH_GPT_integer_count_between_sqrt8_and_sqrt78_l1809_180959

theorem integer_count_between_sqrt8_and_sqrt78 :
  ∃ (n : ℕ), n = 6 ∧ ∀ (x : ℤ), (⌈Real.sqrt 8⌉ ≤ x ∧ x ≤ ⌊Real.sqrt 78⌋) ↔ (3 ≤ x ∧ x ≤ 8) := by
  sorry

end NUMINAMATH_GPT_integer_count_between_sqrt8_and_sqrt78_l1809_180959


namespace NUMINAMATH_GPT_minimum_stool_height_l1809_180974

def ceiling_height : ℤ := 280
def alice_height : ℤ := 150
def reach : ℤ := alice_height + 30
def light_bulb_height : ℤ := ceiling_height - 15

theorem minimum_stool_height : 
  ∃ h : ℤ, reach + h = light_bulb_height ∧ h = 85 :=
by
  sorry

end NUMINAMATH_GPT_minimum_stool_height_l1809_180974


namespace NUMINAMATH_GPT_largest_share_received_l1809_180952

noncomputable def largest_share (total_profit : ℝ) (ratio : List ℝ) : ℝ :=
  let total_parts := ratio.foldl (· + ·) 0
  let part_value := total_profit / total_parts
  let max_part := ratio.foldl max 0
  max_part * part_value

theorem largest_share_received
  (total_profit : ℝ)
  (h_total_profit : total_profit = 42000)
  (ratio : List ℝ)
  (h_ratio : ratio = [2, 3, 4, 4, 6]) :
  largest_share total_profit ratio = 12600 :=
by
  sorry

end NUMINAMATH_GPT_largest_share_received_l1809_180952


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1809_180999

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 ≥ 0 } = {x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1809_180999


namespace NUMINAMATH_GPT_missed_questions_l1809_180983

theorem missed_questions (F M : ℕ) (h1 : M = 5 * F) (h2 : M + F = 216) : M = 180 :=
by
  sorry

end NUMINAMATH_GPT_missed_questions_l1809_180983


namespace NUMINAMATH_GPT_total_distinguishable_triangles_l1809_180950

-- Define number of colors
def numColors : Nat := 8

-- Define center colors
def centerColors : Nat := 3

-- Prove the total number of distinguishable large equilateral triangles
theorem total_distinguishable_triangles : 
  numColors * (numColors + numColors * (numColors - 1) + (numColors.choose 3)) * centerColors = 360 := by
  sorry

end NUMINAMATH_GPT_total_distinguishable_triangles_l1809_180950


namespace NUMINAMATH_GPT_conditional_probability_l1809_180966

variables (A B : Prop)
variables (P : Prop → ℚ)
variables (h₁ : P A = 8 / 30) (h₂ : P (A ∧ B) = 7 / 30)

theorem conditional_probability : P (A → B) = 7 / 8 :=
by sorry

end NUMINAMATH_GPT_conditional_probability_l1809_180966


namespace NUMINAMATH_GPT_part1_part2_l1809_180995

noncomputable def x : ℝ := 1 - Real.sqrt 2
noncomputable def y : ℝ := 1 + Real.sqrt 2

theorem part1 : x^2 + 3 * x * y + y^2 = 3 := by
  sorry

theorem part2 : (y / x) - (x / y) = -4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1809_180995


namespace NUMINAMATH_GPT_find_e_l1809_180965

theorem find_e (b e : ℝ) (f g : ℝ → ℝ)
    (h1 : ∀ x, f x = 5 * x + b)
    (h2 : ∀ x, g x = b * x + 3)
    (h3 : ∀ x, f (g x) = 15 * x + e) : e = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_e_l1809_180965


namespace NUMINAMATH_GPT_safe_security_system_l1809_180916

theorem safe_security_system (commission_members : ℕ) 
                            (majority_access : ℕ)
                            (max_inaccess_members : ℕ) 
                            (locks : ℕ)
                            (keys_per_member : ℕ) :
  commission_members = 11 →
  majority_access = 6 →
  max_inaccess_members = 5 →
  locks = (Nat.choose 11 5) →
  keys_per_member = (locks * 6) / 11 →
  locks = 462 ∧ keys_per_member = 252 :=
by
  intros
  sorry

end NUMINAMATH_GPT_safe_security_system_l1809_180916


namespace NUMINAMATH_GPT_equation_solution_l1809_180924

variable (x y : ℝ)

theorem equation_solution
  (h1 : x * y + x + y = 17)
  (h2 : x^2 * y + x * y^2 = 66):
  x^4 + x^3 * y + x^2 * y^2 + x * y^3 + y^4 = 12499 :=
  by sorry

end NUMINAMATH_GPT_equation_solution_l1809_180924


namespace NUMINAMATH_GPT_average_molecular_weight_benzoic_acid_l1809_180928

def atomic_mass_C : ℝ := (12 * 0.9893) + (13 * 0.0107)
def atomic_mass_H : ℝ := (1 * 0.99985) + (2 * 0.00015)
def atomic_mass_O : ℝ := (16 * 0.99762) + (17 * 0.00038) + (18 * 0.00200)

theorem average_molecular_weight_benzoic_acid :
  (7 * atomic_mass_C) + (6 * atomic_mass_H) + (2 * atomic_mass_O) = 123.05826 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_molecular_weight_benzoic_acid_l1809_180928


namespace NUMINAMATH_GPT_count_white_balls_l1809_180903

variable (W B : ℕ)

theorem count_white_balls
  (h_total : W + B = 30)
  (h_white : ∀ S : Finset ℕ, S.card = 12 → ∃ w ∈ S, w < W)
  (h_black : ∀ S : Finset ℕ, S.card = 20 → ∃ b ∈ S, b < B) :
  W = 19 :=
sorry

end NUMINAMATH_GPT_count_white_balls_l1809_180903


namespace NUMINAMATH_GPT_ratio_Rachel_Sara_l1809_180905

-- Define Sara's spending
def Sara_shoes_spending : ℝ := 50
def Sara_dress_spending : ℝ := 200

-- Define Rachel's budget
def Rachel_budget : ℝ := 500

-- Calculate Sara's total spending
def Sara_total_spending : ℝ := Sara_shoes_spending + Sara_dress_spending

-- Define the theorem to prove the ratio
theorem ratio_Rachel_Sara : (Rachel_budget / Sara_total_spending) = 2 := by
  -- Proof is omitted (you would fill in the proof here)
  sorry

end NUMINAMATH_GPT_ratio_Rachel_Sara_l1809_180905


namespace NUMINAMATH_GPT_balloons_kept_by_Andrew_l1809_180953

theorem balloons_kept_by_Andrew :
  let blue := 303
  let purple := 453
  let red := 165
  let yellow := 324
  let blue_kept := (2/3 : ℚ) * blue
  let purple_kept := (3/5 : ℚ) * purple
  let red_kept := (4/7 : ℚ) * red
  let yellow_kept := (1/3 : ℚ) * yellow
  let total_kept := blue_kept.floor + purple_kept.floor + red_kept.floor + yellow_kept
  total_kept = 675 := by
  sorry

end NUMINAMATH_GPT_balloons_kept_by_Andrew_l1809_180953


namespace NUMINAMATH_GPT_emily_total_spent_l1809_180961

def total_cost (art_supplies_cost skirt_cost : ℕ) (number_of_skirts : ℕ) : ℕ :=
  art_supplies_cost + (skirt_cost * number_of_skirts)

theorem emily_total_spent :
  total_cost 20 15 2 = 50 :=
by
  sorry

end NUMINAMATH_GPT_emily_total_spent_l1809_180961


namespace NUMINAMATH_GPT_inequality_proof_l1809_180957

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
    a * (b^2 + c^2) + b * (c^2 + a^2) ≥ 4 * a * b * c :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1809_180957


namespace NUMINAMATH_GPT_sum_a_b_neg1_l1809_180942

-- Define the problem using the given condition
theorem sum_a_b_neg1 (a b : ℝ) (h : |a + 3| + (b - 2) ^ 2 = 0) : a + b = -1 := 
by
  sorry

end NUMINAMATH_GPT_sum_a_b_neg1_l1809_180942
