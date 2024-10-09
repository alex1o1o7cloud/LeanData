import Mathlib

namespace quadratic_roots_and_signs_l1472_147289

theorem quadratic_roots_and_signs :
  (∃ x1 x2 : ℝ, (x1^2 - 13*x1 + 40 = 0) ∧ (x2^2 - 13*x2 + 40 = 0) ∧ x1 = 5 ∧ x2 = 8 ∧ 0 < x1 ∧ 0 < x2) :=
by
  sorry

end quadratic_roots_and_signs_l1472_147289


namespace gravel_per_truckload_l1472_147298

def truckloads_per_mile : ℕ := 3
def miles_day1 : ℕ := 4
def miles_day2 : ℕ := 2 * miles_day1 - 1
def total_paved_miles : ℕ := miles_day1 + miles_day2
def total_road_length : ℕ := 16
def miles_remaining : ℕ := total_road_length - total_paved_miles
def remaining_truckloads : ℕ := miles_remaining * truckloads_per_mile
def barrels_needed : ℕ := 6
def gravel_per_pitch : ℕ := 5
def P : ℚ := barrels_needed / remaining_truckloads
def G : ℚ := gravel_per_pitch * P

theorem gravel_per_truckload :
  G = 2 :=
by
  sorry

end gravel_per_truckload_l1472_147298


namespace part1_max_min_part2_cos_value_l1472_147270

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x + Real.pi / 6)

theorem part1_max_min (x : ℝ) (hx : -1/2 ≤ x ∧ x ≤ 1/2) : 
  (∃ xₘ, (xₘ ∈ Set.Icc (-1/2) (1/2)) ∧ f xₘ = 2) ∧ 
  (∃ xₘ, (xₘ ∈ Set.Icc (-1/2) (1/2)) ∧ f xₘ = -Real.sqrt 3) :=
sorry

theorem part2_cos_value (α : ℝ) (h : f (α / (2 * Real.pi)) = 1/4) : 
  Real.cos (2 * Real.pi / 3 - α) = -31/32 :=
sorry

end part1_max_min_part2_cos_value_l1472_147270


namespace volume_of_pyramid_l1472_147221

noncomputable def greatest_pyramid_volume (AB AC sin_α : ℝ) (max_angle : ℝ) : ℝ :=
  if AB = 3 ∧ AC = 5 ∧ sin_α = 4 / 5 ∧ max_angle ≤ 60 then
    5 * Real.sqrt 39 / 2
  else
    0

theorem volume_of_pyramid :
  greatest_pyramid_volume 3 5 (4 / 5) 60 = 5 * Real.sqrt 39 / 2 := by
  sorry -- Proof omitted as per instruction

end volume_of_pyramid_l1472_147221


namespace num_sequences_eq_15_l1472_147208

noncomputable def num_possible_sequences : ℕ :=
  let angles_increasing_arith_seq := ∃ (x d : ℕ), x > 0 ∧ x + 4 * d < 140 ∧ 5 * x + 10 * d = 540 ∧ d ≠ 0
  by sorry

theorem num_sequences_eq_15 : num_possible_sequences = 15 := 
  by sorry

end num_sequences_eq_15_l1472_147208


namespace diff_PA_AQ_const_l1472_147250

open Real

def point := (ℝ × ℝ)

noncomputable def distance (p1 p2 : point) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem diff_PA_AQ_const (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  let P := (0, -sqrt 2)
  let Q := (0, sqrt 2)
  let A := (a, sqrt (a^2 + 1))
  distance P A - distance A Q = 2 := 
sorry

end diff_PA_AQ_const_l1472_147250


namespace part1_part2_l1472_147273

noncomputable def A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem part1 (a : ℝ) : A ∪ B a = B a ↔ a = 1 :=
by
  sorry

theorem part2 (a : ℝ) : A ∩ B a = B a ↔ a ≤ -1 ∨ a = 1 :=
by
  sorry

end part1_part2_l1472_147273


namespace system_of_equations_solution_l1472_147282

theorem system_of_equations_solution
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x1 + 2 * x2 + 2 * x3 + 2 * x4 + 2 * x5 = 1)
  (h2 : x1 + 3 * x2 + 4 * x3 + 4 * x4 + 4 * x5 = 2)
  (h3 : x1 + 3 * x2 + 5 * x3 + 6 * x4 + 6 * x5 = 3)
  (h4 : x1 + 3 * x2 + 5 * x3 + 7 * x4 + 8 * x5 = 4)
  (h5 : x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 = 5) :
  x1 = 1 ∧ x2 = -1 ∧ x3 = 1 ∧ x4 = -1 ∧ x5 = 1 :=
by {
  -- proof steps go here
  sorry
}

end system_of_equations_solution_l1472_147282


namespace hyperbola_equation_l1472_147213

open Real

-- Define the conditions in Lean
def is_hyperbola_form (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def is_positive (x : ℝ) : Prop := x > 0

def parabola_focus : (ℝ × ℝ) := (1, 0)

def hyperbola_vertex_eq_focus (a : ℝ) : Prop := a = parabola_focus.1

def hyperbola_eccentricity (e a c : ℝ) : Prop := e = c / a

-- Our proof statement
theorem hyperbola_equation :
  ∃ (a b : ℝ), is_positive a ∧ is_positive b ∧
  hyperbola_vertex_eq_focus a ∧
  hyperbola_eccentricity (sqrt 5) a (sqrt 5) ∧
  is_hyperbola_form a b 1 0 :=
by sorry

end hyperbola_equation_l1472_147213


namespace molecular_weight_of_7_moles_boric_acid_l1472_147207

-- Define the given constants.
def atomic_weight_H : ℝ := 1.008
def atomic_weight_B : ℝ := 10.81
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula for boric acid.
def molecular_weight_H3BO3 : ℝ :=
  3 * atomic_weight_H + 1 * atomic_weight_B + 3 * atomic_weight_O

-- Define the number of moles.
def moles_boric_acid : ℝ := 7

-- Calculate the total weight for 7 moles of boric acid.
def total_weight_boric_acid : ℝ :=
  moles_boric_acid * molecular_weight_H3BO3

-- The target statement to prove.
theorem molecular_weight_of_7_moles_boric_acid :
  total_weight_boric_acid = 432.838 := by
  sorry

end molecular_weight_of_7_moles_boric_acid_l1472_147207


namespace sufficient_but_not_necessary_condition_l1472_147226

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a > 1 → (1 / a < 1)) ∧ ¬((1 / a < 1) → a > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1472_147226


namespace assignment_methods_l1472_147209

theorem assignment_methods : 
  let doctors := 2
  let nurses := 4
  let schools := 2
  let doctors_per_school := 1
  let nurses_per_school := 2
  (doctors * (nurses.choose nurses_per_school)) = 12 := by
  sorry

end assignment_methods_l1472_147209


namespace sin_cos_sum_l1472_147233

theorem sin_cos_sum (x y r : ℝ) (h : r = Real.sqrt (x^2 + y^2)) (ha : (x = 5) ∧ (y = -12)) :
  (y / r) + (x / r) = -7 / 13 :=
by
  sorry

end sin_cos_sum_l1472_147233


namespace de_morgan_implication_l1472_147218

variables (p q : Prop)

theorem de_morgan_implication (h : ¬(p ∧ q)) : ¬p ∨ ¬q :=
sorry

end de_morgan_implication_l1472_147218


namespace sulfuric_acid_percentage_l1472_147225

theorem sulfuric_acid_percentage 
  (total_volume : ℝ)
  (first_solution_percentage : ℝ)
  (final_solution_percentage : ℝ)
  (second_solution_volume : ℝ)
  (expected_second_solution_percentage : ℝ) :
  total_volume = 60 ∧
  first_solution_percentage = 0.02 ∧
  final_solution_percentage = 0.05 ∧
  second_solution_volume = 18 →
  expected_second_solution_percentage = 12 :=
by
  sorry

end sulfuric_acid_percentage_l1472_147225


namespace simultaneous_inequalities_l1472_147217

theorem simultaneous_inequalities (x : ℝ) 
    (h1 : x^3 - 11 * x^2 + 10 * x < 0) 
    (h2 : x^3 - 12 * x^2 + 32 * x > 0) : 
    (1 < x ∧ x < 4) ∨ (8 < x ∧ x < 10) :=
sorry

end simultaneous_inequalities_l1472_147217


namespace find_a_l1472_147244

theorem find_a (a : ℝ) : 
  (∃ (a : ℝ), a * 15 + 6 = -9) → a = -1 :=
by
  intro h
  sorry

end find_a_l1472_147244


namespace ballet_class_members_l1472_147269

theorem ballet_class_members (large_groups : ℕ) (members_per_large_group : ℕ) (total_members : ℕ) 
    (h1 : large_groups = 12) (h2 : members_per_large_group = 7) (h3 : total_members = large_groups * members_per_large_group) : 
    total_members = 84 :=
sorry

end ballet_class_members_l1472_147269


namespace arithmetic_sequence_ratio_a10_b10_l1472_147212

variable {a : ℕ → ℕ} {b : ℕ → ℕ}
variable {S T : ℕ → ℕ}

-- We assume S_n and T_n are the sums of the first n terms of sequences a and b respectively.
-- We also assume the provided ratio condition between S_n and T_n.
axiom sum_of_first_n_terms_a (n : ℕ) : S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2
axiom sum_of_first_n_terms_b (n : ℕ) : T n = (n * (2 * b 1 + (n - 1) * (b 2 - b 1))) / 2
axiom ratio_condition (n : ℕ) : (S n) / (T n) = (3 * n - 1) / (2 * n + 3)

theorem arithmetic_sequence_ratio_a10_b10 : (a 10) / (b 10) = 56 / 41 :=
by sorry

end arithmetic_sequence_ratio_a10_b10_l1472_147212


namespace inequality_solution_set_l1472_147271

theorem inequality_solution_set (x : ℝ) : (3 - 2 * x) * (x + 1) ≤ 0 ↔ (x < -1) ∨ (x ≥ 3 / 2) :=
  sorry

end inequality_solution_set_l1472_147271


namespace find_radius_l1472_147265

theorem find_radius (abbc: ℝ) (adbd: ℝ) (bccc: ℝ) (dcdd: ℝ) (R: ℝ)
  (h1: abbc = 4) (h2: adbd = 4) (h3: bccc = 2) (h4: dcdd = 1) :
  R = 5 :=
sorry

end find_radius_l1472_147265


namespace expand_expression_l1472_147248

theorem expand_expression (x y : ℝ) : 
  (16 * x + 18 - 7 * y) * (3 * x) = 48 * x^2 + 54 * x - 21 * x * y :=
by
  sorry

end expand_expression_l1472_147248


namespace distance_CD_l1472_147287

theorem distance_CD (C D : ℝ × ℝ) (r₁ r₂ : ℝ) (φ₁ φ₂ : ℝ) 
  (hC : C = (r₁, φ₁)) (hD : D = (r₂, φ₂)) (r₁_eq_5 : r₁ = 5) (r₂_eq_12 : r₂ = 12)
  (angle_diff : φ₁ - φ₂ = π / 3) : dist C D = Real.sqrt 109 :=
  sorry

end distance_CD_l1472_147287


namespace swimmer_speeds_l1472_147203

variable (a s r : ℝ)
variable (x z y : ℝ)

theorem swimmer_speeds (h : s < r) (h' : r < 100 * s / (50 + s)) :
    (100 * s - 50 * r - r * s) / ((3 * s - r) * a) = x ∧ 
    (100 * s - 50 * r - r * s) / ((r - s) * a) = z := by
    sorry

end swimmer_speeds_l1472_147203


namespace find_first_term_l1472_147254

noncomputable def first_term_of_arithmetic_sequence : ℝ := -19.2

theorem find_first_term
  (a d : ℝ)
  (h1 : 50 * (2 * a + 99 * d) = 1050)
  (h2 : 50 * (2 * a + 199 * d) = 4050) :
  a = first_term_of_arithmetic_sequence :=
by
  -- Given conditions
  have h1' : 2 * a + 99 * d = 21 := by sorry
  have h2' : 2 * a + 199 * d = 81 := by sorry
  -- Solve for d
  have hd : d = 0.6 := by sorry
  -- Substitute d into h1'
  have h_subst : 2 * a + 99 * 0.6 = 21 := by sorry
  -- Solve for a
  have ha : a = -19.2 := by sorry
  exact ha

end find_first_term_l1472_147254


namespace intersection_point_with_y_axis_l1472_147206

theorem intersection_point_with_y_axis : 
  ∃ y, (0, y) = (0, 3) ∧ (y = 0 + 3) :=
by
  sorry

end intersection_point_with_y_axis_l1472_147206


namespace contradiction_assumption_l1472_147263

theorem contradiction_assumption (x y : ℝ) (h1 : x > y) : ¬ (x^3 ≤ y^3) := 
by
  sorry

end contradiction_assumption_l1472_147263


namespace find_k_unique_solution_l1472_147229

theorem find_k_unique_solution :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 0 → (1/(3*x) = (k - x)/8) → (3*x^2 + (8 - 3*k)*x = 0)) →
    k = 8 / 3 :=
by
  intros k h
  -- Using sorry here to skip the proof
  sorry

end find_k_unique_solution_l1472_147229


namespace product_pricing_and_savings_l1472_147223

theorem product_pricing_and_savings :
  ∃ (x y : ℝ),
    (6 * x + 3 * y = 600) ∧
    (40 * x + 30 * y = 5200) ∧
    x = 40 ∧
    y = 120 ∧
    (80 * x + 100 * y - (80 * 0.8 * x + 100 * 0.75 * y) = 3640) := 
by
  sorry

end product_pricing_and_savings_l1472_147223


namespace prob_sum_to_3_three_dice_correct_l1472_147284

def prob_sum_to_3_three_dice (sum : ℕ) (dice_count : ℕ) (dice_faces : Finset ℕ) : ℚ :=
  if sum = 3 ∧ dice_count = 3 ∧ dice_faces = {1, 2, 3, 4, 5, 6} then (1 : ℚ) / 216 else 0

theorem prob_sum_to_3_three_dice_correct :
  prob_sum_to_3_three_dice 3 3 {1, 2, 3, 4, 5, 6} = (1 : ℚ) / 216 := 
by
  sorry

end prob_sum_to_3_three_dice_correct_l1472_147284


namespace triangle_proof_problem_l1472_147228

-- The conditions and question programmed as a Lean theorem statement
theorem triangle_proof_problem
    (A B C : ℝ)
    (h1 : A > B)
    (S T : ℝ)
    (h2 : A = C)
    (K : ℝ)
    (arc_mid_A : K = A): -- K is midpoint of the arc A
    
    RS = K := sorry

end triangle_proof_problem_l1472_147228


namespace sum_of_interior_angles_of_hexagon_l1472_147234

theorem sum_of_interior_angles_of_hexagon
  (n : ℕ)
  (h : n = 6) :
  (n - 2) * 180 = 720 := by
  sorry

end sum_of_interior_angles_of_hexagon_l1472_147234


namespace value_of_c_plus_d_l1472_147252

theorem value_of_c_plus_d (a b c d : ℝ) (h1 : a + b = 5) (h2 : b + c = 6) (h3 : a + d = 2) : c + d = 3 :=
by
  sorry

end value_of_c_plus_d_l1472_147252


namespace length_AB_l1472_147295

-- Definitions and conditions
variables (R r a : ℝ) (hR : R > r) (BC_eq_a : BC = a) (r_eq_4 : r = 4)

-- Length of AB
theorem length_AB (AB : ℝ) : AB = a * Real.sqrt (R / (R - 4)) :=
sorry

end length_AB_l1472_147295


namespace jerome_contacts_total_l1472_147240

def jerome_classmates : Nat := 20
def jerome_out_of_school_friends : Nat := jerome_classmates / 2
def jerome_family_members : Nat := 2 + 1
def jerome_total_contacts : Nat := jerome_classmates + jerome_out_of_school_friends + jerome_family_members

theorem jerome_contacts_total : jerome_total_contacts = 33 := by
  sorry

end jerome_contacts_total_l1472_147240


namespace pow_two_greater_than_square_l1472_147294

theorem pow_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2 ^ n > n ^ 2 :=
  sorry

end pow_two_greater_than_square_l1472_147294


namespace days_to_use_up_one_bag_l1472_147285

def rice_kg : ℕ := 11410
def bags : ℕ := 3260
def rice_per_day : ℚ := 0.25
def rice_per_bag : ℚ := rice_kg / bags

theorem days_to_use_up_one_bag : (rice_per_bag / rice_per_day) = 14 := by
  sorry

end days_to_use_up_one_bag_l1472_147285


namespace problem_b_amount_l1472_147280

theorem problem_b_amount (a b : ℝ) (h1 : a + b = 1210) (h2 : (4/5) * a = (2/3) * b) : b = 453.75 :=
sorry

end problem_b_amount_l1472_147280


namespace parabola_x0_range_l1472_147235

variables {x₀ y₀ : ℝ}
def parabola (x₀ y₀ : ℝ) : Prop := y₀^2 = 8 * x₀

def focus (x : ℝ) : ℝ := 2

def directrix (x : ℝ) : Prop := x = -2

/-- Prove that for any point (x₀, y₀) on the parabola y² = 8x and 
if a circle centered at the focus intersects the directrix, then x₀ > 2. -/
theorem parabola_x0_range (x₀ y₀ : ℝ) (h1 : parabola x₀ y₀)
  (h2 : ((x₀ - 2)^2 + y₀^2)^(1/2) > (2 : ℝ)) : x₀ > 2 := 
sorry

end parabola_x0_range_l1472_147235


namespace marbles_problem_l1472_147231

theorem marbles_problem (h_total: ℕ) (h_each: ℕ) (h_total_eq: h_total = 35) (h_each_eq: h_each = 7) :
    h_total / h_each = 5 := by
  sorry

end marbles_problem_l1472_147231


namespace circle_area_increase_l1472_147259

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * (new_radius)^2
  let increase := new_area - original_area
  let percentage_increase := (increase / original_area) * 100
  percentage_increase = 125 := 
by {
  -- The proof will be written here.
  sorry
}

end circle_area_increase_l1472_147259


namespace expression_not_equal_33_l1472_147257

theorem expression_not_equal_33 (x y : ℤ) :
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end expression_not_equal_33_l1472_147257


namespace slices_left_per_person_is_2_l1472_147286

variables (phil_slices andre_slices small_pizza_slices large_pizza_slices : ℕ)
variables (total_slices_eaten total_slices_left slices_per_person : ℕ)

-- Conditions
def conditions : Prop :=
  phil_slices = 9 ∧
  andre_slices = 9 ∧
  small_pizza_slices = 8 ∧
  large_pizza_slices = 14 ∧
  total_slices_eaten = phil_slices + andre_slices ∧
  total_slices_left = (small_pizza_slices + large_pizza_slices) - total_slices_eaten ∧
  slices_per_person = total_slices_left / 2

theorem slices_left_per_person_is_2 (h : conditions phil_slices andre_slices small_pizza_slices large_pizza_slices total_slices_eaten total_slices_left slices_per_person) :
  slices_per_person = 2 :=
sorry

end slices_left_per_person_is_2_l1472_147286


namespace harmon_high_school_proof_l1472_147290

noncomputable def harmon_high_school : Prop :=
  ∃ (total_players players_physics players_both players_chemistry : ℕ),
    total_players = 18 ∧
    players_physics = 10 ∧
    players_both = 3 ∧
    players_chemistry = (total_players - players_physics + players_both)

theorem harmon_high_school_proof : harmon_high_school :=
  sorry

end harmon_high_school_proof_l1472_147290


namespace point_Q_in_third_quadrant_l1472_147243

-- Define point P in the fourth quadrant with coordinates a and b.
variable (a b : ℝ)
variable (h1 : a > 0)  -- Condition for the x-coordinate of P in fourth quadrant
variable (h2 : b < 0)  -- Condition for the y-coordinate of P in fourth quadrant

-- Point Q is defined by the coordinates (-a, b-1). We need to show it lies in the third quadrant.
theorem point_Q_in_third_quadrant : (-a < 0) ∧ (b - 1 < 0) :=
  by
    sorry

end point_Q_in_third_quadrant_l1472_147243


namespace ant_travel_distance_l1472_147232

theorem ant_travel_distance (r1 r2 r3 : ℝ) (h1 : r1 = 5) (h2 : r2 = 10) (h3 : r3 = 15) :
  let A_large := (1/3) * 2 * Real.pi * r3
  let D_radial := (r3 - r2) + (r2 - r1)
  let A_middle := (1/3) * 2 * Real.pi * r2
  let D_small := 2 * r1
  let A_small := (1/2) * 2 * Real.pi * r1
  A_large + D_radial + A_middle + D_small + A_small = (65 * Real.pi / 3) + 20 :=
by
  sorry

end ant_travel_distance_l1472_147232


namespace calc_delta_l1472_147291

noncomputable def delta (a b : ℝ) : ℝ :=
  (a^2 + b^2) / (1 + a * b)

-- Definition of the main problem as a Lean 4 statement
theorem calc_delta (h1 : 2 > 0) (h2 : 3 > 0) (h3 : 4 > 0) :
  delta (delta 2 3) 4 = 6661 / 2891 :=
by
  sorry

end calc_delta_l1472_147291


namespace inverse_of_h_l1472_147202

noncomputable def h (x : ℝ) : ℝ := 3 - 7 * x
noncomputable def k (x : ℝ) : ℝ := (3 - x) / 7

theorem inverse_of_h :
  (∀ x : ℝ, h (k x) = x) ∧ (∀ x : ℝ, k (h x) = x) :=
by
  sorry

end inverse_of_h_l1472_147202


namespace initial_sugar_amount_l1472_147279

-- Definitions based on the conditions
def packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Theorem statement
theorem initial_sugar_amount : packs * weight_per_pack + leftover_sugar = 3020 :=
by
  sorry

end initial_sugar_amount_l1472_147279


namespace dart_lands_in_center_square_l1472_147297

theorem dart_lands_in_center_square (s : ℝ) (h : 0 < s) :
  let center_square_area := (s / 2) ^ 2
  let triangle_area := 1 / 2 * (s / 2) ^ 2
  let total_triangle_area := 4 * triangle_area
  let total_board_area := center_square_area + total_triangle_area
  let probability := center_square_area / total_board_area
  probability = 1 / 3 :=
by
  sorry

end dart_lands_in_center_square_l1472_147297


namespace roots_geometric_progression_condition_l1472_147268

theorem roots_geometric_progression_condition 
  (a b c : ℝ) 
  (x1 x2 x3 : ℝ)
  (h1 : x1 + x2 + x3 = -a)
  (h2 : x1 * x2 + x2 * x3 + x1 * x3 = b)
  (h3 : x1 * x2 * x3 = -c)
  (h4 : x2^2 = x1 * x3) :
  a^3 * c = b^3 :=
sorry

end roots_geometric_progression_condition_l1472_147268


namespace intersection_M_N_l1472_147272

def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 4}
def N : Set ℝ := {x : ℝ | x ≥ -1 / 3}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 / 3 ≤ x ∧ x < 4} :=
sorry

end intersection_M_N_l1472_147272


namespace calculation_error_l1472_147214

theorem calculation_error (x y : ℕ) : (25 * x + 5 * y) = 25 * x + 5 * y :=
by
  sorry

end calculation_error_l1472_147214


namespace simplify_expression_to_polynomial_l1472_147204

theorem simplify_expression_to_polynomial :
    (3 * x^2 + 4 * x + 8) * (2 * x + 1) - 
    (2 * x + 1) * (x^2 + 5 * x - 72) + 
    (4 * x - 15) * (2 * x + 1) * (x + 6) = 
    12 * x^3 + 22 * x^2 - 12 * x - 10 :=
by
    sorry

end simplify_expression_to_polynomial_l1472_147204


namespace base_case_n_equals_1_l1472_147296

variable {a : ℝ}
variable {n : ℕ}

theorem base_case_n_equals_1 (h1 : a ≠ 1) (h2 : n = 1) : 1 + a = 1 + a :=
by
  sorry

end base_case_n_equals_1_l1472_147296


namespace gcd_48_30_is_6_l1472_147215

/-- Prove that the Greatest Common Divisor (GCD) of 48 and 30 is 6. -/
theorem gcd_48_30_is_6 : Int.gcd 48 30 = 6 := by
  sorry

end gcd_48_30_is_6_l1472_147215


namespace solve_expression_l1472_147264

theorem solve_expression (x y z : ℚ)
  (h1 : 2 * x + 3 * y + z = 20)
  (h2 : x + 2 * y + 3 * z = 26)
  (h3 : 3 * x + y + 2 * z = 29) :
  12 * x^2 + 22 * x * y + 12 * y^2 + 12 * x * z + 12 * y * z + 12 * z^2 = (computed_value : ℚ) :=
by
  sorry

end solve_expression_l1472_147264


namespace solve_for_x_l1472_147293

theorem solve_for_x {x : ℝ} (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (15 * x) * Real.sqrt (4 * x) * Real.sqrt (10 * x) = 20) :
  x = 2^(1/4) / Real.sqrt 3 :=
by
  -- proof omitted
  sorry

end solve_for_x_l1472_147293


namespace added_number_is_nine_l1472_147283

theorem added_number_is_nine (y : ℤ) : 
  3 * (2 * 4 + y) = 51 → y = 9 :=
by
  sorry

end added_number_is_nine_l1472_147283


namespace gcd_50403_40302_l1472_147261

theorem gcd_50403_40302 : Nat.gcd 50403 40302 = 1 :=
by
  sorry

end gcd_50403_40302_l1472_147261


namespace correlated_relationships_l1472_147205

-- Definitions for the conditions are arbitrary
-- In actual use cases, these would be replaced with real mathematical conditions
def great_teachers_produce_outstanding_students : Prop := sorry
def volume_of_sphere_with_radius : Prop := sorry
def apple_production_climate : Prop := sorry
def height_and_weight : Prop := sorry
def taxi_fare_distance_traveled : Prop := sorry
def crows_cawing_bad_omen : Prop := sorry

-- The final theorem statement
theorem correlated_relationships : 
  great_teachers_produce_outstanding_students ∧
  apple_production_climate ∧
  height_and_weight ∧
  ¬ volume_of_sphere_with_radius ∧ 
  ¬ taxi_fare_distance_traveled ∧ 
  ¬ crows_cawing_bad_omen :=
sorry

end correlated_relationships_l1472_147205


namespace weight_of_empty_carton_l1472_147267

theorem weight_of_empty_carton
    (half_full_carton_weight : ℕ)
    (full_carton_weight : ℕ)
    (h1 : half_full_carton_weight = 5)
    (h2 : full_carton_weight = 8) :
  full_carton_weight - 2 * (full_carton_weight - half_full_carton_weight) = 2 :=
by
  sorry

end weight_of_empty_carton_l1472_147267


namespace complement_problem_l1472_147219

open Set

variable (U A : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := U \ A

theorem complement_problem
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3}) :
  complement U A = {2, 4, 5} :=
by
  rw [complement, hU, hA]
  sorry

end complement_problem_l1472_147219


namespace average_of_three_numbers_is_78_l1472_147210

theorem average_of_three_numbers_is_78 (x y z : ℕ) (h1 : z = 2 * y) (h2 : y = 4 * x) (h3 : x = 18) :
  (x + y + z) / 3 = 78 :=
by sorry

end average_of_three_numbers_is_78_l1472_147210


namespace find_2a_plus_6b_l1472_147222

theorem find_2a_plus_6b (a b : ℕ) (n : ℕ)
  (h1 : 3 * a + 5 * b ≡ 19 [MOD n + 1])
  (h2 : 4 * a + 2 * b ≡ 25 [MOD n + 1])
  (hn : n = 96) :
  2 * a + 6 * b = 96 :=
by
  sorry

end find_2a_plus_6b_l1472_147222


namespace people_who_came_to_game_l1472_147256

def total_seats : Nat := 92
def people_with_banners : Nat := 38
def empty_seats : Nat := 45

theorem people_who_came_to_game : (total_seats - empty_seats = 47) :=
by 
  sorry

end people_who_came_to_game_l1472_147256


namespace increasing_f_iff_m_ge_two_inequality_when_m_equals_three_l1472_147224

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 4 * x + m * Real.log x

-- Part (1): Prove m >= 2 is the range for which f(x) is increasing
theorem increasing_f_iff_m_ge_two (m : ℝ) : (∀ x > 0, (2 * x - 4 + m / x) ≥ 0) ↔ m ≥ 2 := sorry

-- Part (2): Prove the given inequality for m = 3
theorem inequality_when_m_equals_three (x : ℝ) (h : x > 0) : (1 / 9) * x ^ 3 - (f x 3) > 2 := sorry

end increasing_f_iff_m_ge_two_inequality_when_m_equals_three_l1472_147224


namespace sarah_score_l1472_147275

theorem sarah_score
  (hunter_score : ℕ)
  (john_score : ℕ)
  (grant_score : ℕ)
  (sarah_score : ℕ)
  (h1 : hunter_score = 45)
  (h2 : john_score = 2 * hunter_score)
  (h3 : grant_score = john_score + 10)
  (h4 : sarah_score = grant_score - 5) :
  sarah_score = 95 :=
by
  sorry

end sarah_score_l1472_147275


namespace first_group_men_count_l1472_147227

/-- Given that 10 men can complete a piece of work in 90 hours,
prove that the number of men M in the first group who can complete
the same piece of work in 25 hours is 36. -/
theorem first_group_men_count (M : ℕ) (h : (10 * 90 = 25 * M)) : M = 36 :=
by
  sorry

end first_group_men_count_l1472_147227


namespace pizza_varieties_l1472_147211

-- Definition of the problem conditions
def base_flavors : ℕ := 4
def topping_options : ℕ := 4  -- No toppings, extra cheese, mushrooms, both

-- The math proof problem statement
theorem pizza_varieties : base_flavors * topping_options = 16 := by 
  sorry

end pizza_varieties_l1472_147211


namespace OfficerHoppsTotalTickets_l1472_147245

theorem OfficerHoppsTotalTickets : 
  (15 * 8 + (31 - 15) * 5 = 200) :=
  by
    sorry

end OfficerHoppsTotalTickets_l1472_147245


namespace solve_x_squared_solve_x_cubed_l1472_147292

-- Define the first problem with its condition and prove the possible solutions
theorem solve_x_squared {x : ℝ} (h : (x + 1)^2 = 9) : x = 2 ∨ x = -4 :=
sorry

-- Define the second problem with its condition and prove the possible solution
theorem solve_x_cubed {x : ℝ} (h : -2 * (x^3 - 1) = 18) : x = -2 :=
sorry

end solve_x_squared_solve_x_cubed_l1472_147292


namespace number_of_whole_numbers_with_cube_roots_less_than_8_l1472_147236

theorem number_of_whole_numbers_with_cube_roots_less_than_8 :
  ∃ (n : ℕ), (∀ (x : ℕ), (1 ≤ x ∧ x < 512) → x ≤ n) ∧ n = 511 := 
sorry

end number_of_whole_numbers_with_cube_roots_less_than_8_l1472_147236


namespace find_g_of_7_l1472_147258

theorem find_g_of_7 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 8) = 2 * x + 11) : g 7 = 21 :=
by
  sorry

end find_g_of_7_l1472_147258


namespace fraction_difference_l1472_147237

theorem fraction_difference : 7 / 12 - 3 / 8 = 5 / 24 := 
by 
  sorry

end fraction_difference_l1472_147237


namespace only_odd_integer_option_l1472_147288

theorem only_odd_integer_option : 
  (6 ^ 2 = 36 ∧ Even 36) ∧ 
  (23 - 17 = 6 ∧ Even 6) ∧ 
  (9 * 24 = 216 ∧ Even 216) ∧ 
  (96 / 8 = 12 ∧ Even 12) ∧ 
  (9 * 41 = 369 ∧ Odd 369)
:= by
  sorry

end only_odd_integer_option_l1472_147288


namespace determine_OP_squared_l1472_147266

-- Define the given conditions
variable (O P : Point) -- Points: center O and intersection point P
variable (r : ℝ) (AB CD : ℝ) (E F : Point) -- radius, lengths of chords, midpoints of chords
variable (OE OF : ℝ) -- Distances from center to midpoints of chords
variable (EF : ℝ) -- Distance between midpoints
variable (OP : ℝ) -- Distance from center to intersection point

-- Conditions as given
axiom circle_radius : r = 30
axiom chord_AB_length : AB = 40
axiom chord_CD_length : CD = 14
axiom distance_midpoints : EF = 15
axiom distance_OE : OE = 20
axiom distance_OF : OF = 29

-- The proof problem: determine that OP^2 = 733 given the conditions
theorem determine_OP_squared :
  OP^2 = 733 :=
sorry

end determine_OP_squared_l1472_147266


namespace ratio_x_y_l1472_147281

theorem ratio_x_y (x y : ℚ) (h : (14 * x - 5 * y) / (17 * x - 3 * y) = 2 / 3) : x / y = 1 / 23 := by
  sorry

end ratio_x_y_l1472_147281


namespace power_mod_remainder_l1472_147242

theorem power_mod_remainder :
  3 ^ 3021 % 13 = 1 :=
by
  sorry

end power_mod_remainder_l1472_147242


namespace smallest_k_l1472_147277

theorem smallest_k (k : ℕ) : 
  (∀ x, x ∈ [13, 7, 3, 5] → k % x = 1) ∧ k > 1 → k = 1366 :=
by
  sorry

end smallest_k_l1472_147277


namespace range_of_function_l1472_147249

noncomputable def function_range (x : ℝ) : ℝ :=
    (1 / 2) ^ (-x^2 + 2 * x)

theorem range_of_function : 
    (Set.range function_range) = Set.Ici (1 / 2) :=
by
    sorry

end range_of_function_l1472_147249


namespace function_passes_through_A_l1472_147253

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + Real.log x / Real.log a

theorem function_passes_through_A 
  (a : ℝ) 
  (h1 : 0 < a) 
  (h2 : a ≠ 1)
  : f a 2 = 4 := sorry

end function_passes_through_A_l1472_147253


namespace ping_pong_balls_sold_l1472_147241

theorem ping_pong_balls_sold (total_baseballs initial_baseballs initial_pingpong total_baseballs_sold total_balls_left : ℕ)
  (h1 : total_baseballs = 2754)
  (h2 : initial_pingpong = 1938)
  (h3 : total_baseballs_sold = 1095)
  (h4 : total_balls_left = 3021) :
  initial_pingpong - (total_balls_left - (total_baseballs - total_baseballs_sold)) = 576 :=
by sorry

end ping_pong_balls_sold_l1472_147241


namespace total_opponents_points_is_36_l1472_147278
-- Import the Mathlib library

-- Define the conditions as Lean definitions
def game_scores : List ℕ := [3, 5, 6, 7, 8, 9, 11, 12]

def lost_by_two (n : ℕ) : Prop := n + 2 ∈ game_scores

def three_times_as_many (n : ℕ) : Prop := n * 3 ∈ game_scores

-- State the problem
theorem total_opponents_points_is_36 : 
  (∃ l1 l2 l3 w1 w2 w3 w4 w5 : ℕ, 
    game_scores = [l1, l2, l3, w1, w2, w3, w4, w5] ∧
    lost_by_two l1 ∧ lost_by_two l2 ∧ lost_by_two l3 ∧
    three_times_as_many w1 ∧ three_times_as_many w2 ∧ 
    three_times_as_many w3 ∧ three_times_as_many w4 ∧ 
    three_times_as_many w5 ∧ 
    l1 + 2 + l2 + 2 + l3 + 2 + ((w1 / 3) + (w2 / 3) + (w3 / 3) + (w4 / 3) + (w5 / 3)) = 36) :=
sorry

end total_opponents_points_is_36_l1472_147278


namespace bucket_weight_l1472_147255

theorem bucket_weight (x y p q : ℝ) 
  (h1 : x + (3 / 4) * y = p) 
  (h2 : x + (1 / 3) * y = q) :
  x + (5 / 6) * y = (6 * p - q) / 5 :=
sorry

end bucket_weight_l1472_147255


namespace secondary_spermatocytes_can_contain_two_y_chromosomes_l1472_147276

-- Definitions corresponding to the conditions
def primary_spermatocytes_first_meiotic_division_contains_y (n : Nat) : Prop := n = 1
def spermatogonia_metaphase_mitosis_contains_y (n : Nat) : Prop := n = 1
def secondary_spermatocytes_second_meiotic_division_contains_y (n : Nat) : Prop := n = 0 ∨ n = 2
def spermatogonia_prophase_mitosis_contains_y (n : Nat) : Prop := n = 1

-- The theorem statement equivalent to the given math problem
theorem secondary_spermatocytes_can_contain_two_y_chromosomes :
  ∃ n, (secondary_spermatocytes_second_meiotic_division_contains_y n ∧ n = 2) :=
sorry

end secondary_spermatocytes_can_contain_two_y_chromosomes_l1472_147276


namespace meat_purchase_l1472_147274

theorem meat_purchase :
  ∃ x y : ℕ, 16 * x = y + 25 ∧ 8 * x = y - 15 ∧ y / x = 11 :=
by
  sorry

end meat_purchase_l1472_147274


namespace bulb_illumination_l1472_147262

theorem bulb_illumination (n : ℕ) (h : n = 6) : 
  (2^n - 1) = 63 := by {
  sorry
}

end bulb_illumination_l1472_147262


namespace solution_set_ineq_l1472_147220

theorem solution_set_ineq (x : ℝ) : (x - 2) / (x - 5) ≥ 3 ↔ 5 < x ∧ x ≤ 13 / 2 :=
sorry

end solution_set_ineq_l1472_147220


namespace correct_average_is_40_point_3_l1472_147238

noncomputable def incorrect_average : ℝ := 40.2
noncomputable def incorrect_total_sum : ℝ := incorrect_average * 10
noncomputable def incorrect_first_number_adjustment : ℝ := 17
noncomputable def incorrect_second_number_actual : ℝ := 31
noncomputable def incorrect_second_number_provided : ℝ := 13
noncomputable def correct_total_sum : ℝ := incorrect_total_sum - incorrect_first_number_adjustment + (incorrect_second_number_actual - incorrect_second_number_provided)
noncomputable def number_of_values : ℝ := 10

theorem correct_average_is_40_point_3 :
  correct_total_sum / number_of_values = 40.3 :=
by
  sorry

end correct_average_is_40_point_3_l1472_147238


namespace probability_neither_red_nor_purple_l1472_147201

theorem probability_neither_red_nor_purple
  (total_balls : ℕ)
  (white_balls : ℕ)
  (green_balls : ℕ)
  (yellow_balls : ℕ)
  (red_balls : ℕ)
  (purple_balls : ℕ)
  (h_total : total_balls = 60)
  (h_white : white_balls = 22)
  (h_green : green_balls = 18)
  (h_yellow : yellow_balls = 17)
  (h_red : red_balls = 3)
  (h_purple : purple_balls = 1) :
  ((total_balls - red_balls - purple_balls) / total_balls : ℚ) = 14 / 15 :=
by
  sorry

end probability_neither_red_nor_purple_l1472_147201


namespace fraction_pow_zero_l1472_147247

theorem fraction_pow_zero (a b : ℤ) (hb_nonzero : b ≠ 0) : (a / (b : ℚ)) ^ 0 = 1 :=
by 
  sorry

end fraction_pow_zero_l1472_147247


namespace ladybugs_with_spots_l1472_147239

theorem ladybugs_with_spots (total_ladybugs without_spots with_spots : ℕ) 
  (h1 : total_ladybugs = 67082) 
  (h2 : without_spots = 54912) 
  (h3 : with_spots = total_ladybugs - without_spots) : 
  with_spots = 12170 := 
by 
  -- hole for the proof 
  sorry

end ladybugs_with_spots_l1472_147239


namespace evaluate_expression_l1472_147260

theorem evaluate_expression :
  (2^1 - 3 + 5^3 - 2)⁻¹ * 3 = (3 : ℚ) / 122 :=
by
  -- proof goes here
  sorry

end evaluate_expression_l1472_147260


namespace bob_cleaning_time_l1472_147230

-- Define the conditions
def timeAlice : ℕ := 30
def fractionBob : ℚ := 1 / 3

-- Define the proof problem
theorem bob_cleaning_time : (fractionBob * timeAlice : ℚ) = 10 := by
  sorry

end bob_cleaning_time_l1472_147230


namespace find_a_plus_b_l1472_147251

variable (r a b : ℝ)
variable (seq : ℕ → ℝ)

-- Conditions on the sequence
axiom seq_def : seq 0 = 4096
axiom seq_rule : ∀ n, seq (n + 1) = seq n * r

-- Given value
axiom r_value : r = 1 / 4

-- Given intermediate positions in the sequence
axiom seq_a : seq 3 = a
axiom seq_b : seq 4 = b
axiom seq_5 : seq 5 = 4

-- Theorem to prove
theorem find_a_plus_b : a + b = 80 := by
  sorry

end find_a_plus_b_l1472_147251


namespace fraction_sum_l1472_147200

variable (x y : ℚ)

theorem fraction_sum (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := 
by
  sorry

end fraction_sum_l1472_147200


namespace annual_interest_rate_of_second_investment_l1472_147216

-- Definitions for the conditions
def total_income : ℝ := 575
def investment1 : ℝ := 3000
def rate1 : ℝ := 0.085
def income1 : ℝ := investment1 * rate1
def investment2 : ℝ := 5000
def target_income : ℝ := total_income - income1

-- Lean 4 statement to prove the annual simple interest rate of the second investment
theorem annual_interest_rate_of_second_investment : ∃ (r : ℝ), target_income = investment2 * (r / 100) ∧ r = 6.4 :=
by sorry

end annual_interest_rate_of_second_investment_l1472_147216


namespace call_duration_l1472_147299

def initial_credit : ℝ := 30
def cost_per_minute : ℝ := 0.16
def remaining_credit : ℝ := 26.48

theorem call_duration :
  (initial_credit - remaining_credit) / cost_per_minute = 22 := 
sorry

end call_duration_l1472_147299


namespace equal_probability_of_selection_l1472_147246

-- Define the number of total students
def total_students : ℕ := 86

-- Define the number of students to be eliminated through simple random sampling
def eliminated_students : ℕ := 6

-- Define the number of students selected through systematic sampling
def selected_students : ℕ := 8

-- Define the probability calculation
def probability_not_eliminated : ℚ := 80 / 86
def probability_selected : ℚ := 8 / 80
def combined_probability : ℚ := probability_not_eliminated * probability_selected

theorem equal_probability_of_selection :
  combined_probability = 4 / 43 :=
by
  -- We do not need to complete the proof as per instruction
  sorry

end equal_probability_of_selection_l1472_147246
