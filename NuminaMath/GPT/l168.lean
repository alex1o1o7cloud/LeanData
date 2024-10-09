import Mathlib

namespace fuel_A_added_l168_16839

noncomputable def total_tank_capacity : ℝ := 218

noncomputable def ethanol_fraction_A : ℝ := 0.12
noncomputable def ethanol_fraction_B : ℝ := 0.16

noncomputable def total_ethanol : ℝ := 30

theorem fuel_A_added (x : ℝ) 
    (hA : 0 ≤ x) 
    (hA_le_capacity : x ≤ total_tank_capacity) 
    (h_eq : 0.12 * x + 0.16 * (total_tank_capacity - x) = total_ethanol) : 
    x = 122 := 
sorry

end fuel_A_added_l168_16839


namespace division_example_l168_16899

theorem division_example :
  100 / 0.25 = 400 :=
by sorry

end division_example_l168_16899


namespace complement_intersection_l168_16877

open Set -- Open namespace for set operations

-- Define the universal set I
def I : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set ℕ := {1, 2, 3, 4}

-- Define set B
def B : Set ℕ := {3, 4, 5, 6}

-- Define the intersection A ∩ B
def A_inter_B : Set ℕ := A ∩ B

-- Define the complement C_I(S) as I \ S, where S is a subset of I
def complement (S : Set ℕ) : Set ℕ := I \ S

-- Prove that the complement of A ∩ B in I is {1, 2, 5, 6}
theorem complement_intersection : complement A_inter_B = {1, 2, 5, 6} :=
by
  sorry -- Proof to be provided

end complement_intersection_l168_16877


namespace sum_last_two_digits_of_powers_l168_16831

theorem sum_last_two_digits_of_powers (h₁ : 9 = 10 - 1) (h₂ : 11 = 10 + 1) :
  (9^20 + 11^20) % 100 / 10 + (9^20 + 11^20) % 10 = 2 :=
by
  sorry

end sum_last_two_digits_of_powers_l168_16831


namespace ratio_of_areas_of_concentric_circles_l168_16858

theorem ratio_of_areas_of_concentric_circles (C1 C2 : ℝ) (h1 : (60 / 360) * C1 = (45 / 360) * C2) :
  (C1 / C2) ^ 2 = (9 / 16) := by
  sorry

end ratio_of_areas_of_concentric_circles_l168_16858


namespace totalNumberOfBalls_l168_16808

def numberOfBoxes : ℕ := 3
def numberOfBallsPerBox : ℕ := 5

theorem totalNumberOfBalls : numberOfBoxes * numberOfBallsPerBox = 15 := 
by
  sorry

end totalNumberOfBalls_l168_16808


namespace dartboard_odd_sum_probability_l168_16881

theorem dartboard_odd_sum_probability :
  let innerR := 4
  let outerR := 8
  let inner_points := [3, 1, 1]
  let outer_points := [2, 3, 3]
  let total_area := π * outerR^2
  let inner_area := π * innerR^2
  let outer_area := total_area - inner_area
  let each_inner_area := inner_area / 3
  let each_outer_area := outer_area / 3
  let odd_area := 2 * each_inner_area + 2 * each_outer_area
  let even_area := each_inner_area + each_outer_area
  let P_odd := odd_area / total_area
  let P_even := even_area / total_area
  let odd_sum_prob := 2 * (P_odd * P_even)
  odd_sum_prob = 4 / 9 := by
    sorry

end dartboard_odd_sum_probability_l168_16881


namespace symmetric_slope_angle_l168_16813

theorem symmetric_slope_angle (α₁ : ℝ)
  (hα₁ : 0 ≤ α₁ ∧ α₁ < Real.pi) :
  ∃ α₂ : ℝ, (α₁ < Real.pi / 2 → α₂ = Real.pi - α₁) ∧
            (α₁ = Real.pi / 2 → α₂ = 0) :=
sorry

end symmetric_slope_angle_l168_16813


namespace find_S6_l168_16857

def arithmetic_sum (n : ℕ) : ℝ := sorry
def S_3 := 6
def S_9 := 27

theorem find_S6 : ∃ S_6 : ℝ, S_6 = 15 ∧ 
                              S_6 - S_3 = (6 + (S_9 - S_6)) / 2 :=
sorry

end find_S6_l168_16857


namespace nancy_total_money_l168_16812

def total_money (n_five n_ten n_one : ℕ) : ℕ :=
  (n_five * 5) + (n_ten * 10) + (n_one * 1)

theorem nancy_total_money :
  total_money 9 4 7 = 92 :=
by
  sorry

end nancy_total_money_l168_16812


namespace real_roots_of_system_l168_16894

theorem real_roots_of_system :
  { (x, y) : ℝ × ℝ | (x + y)^4 = 6 * x^2 * y^2 - 215 ∧ x * y * (x^2 + y^2) = -78 } =
  { (3, -2), (-2, 3), (-3, 2), (2, -3) } :=
by 
  sorry

end real_roots_of_system_l168_16894


namespace problem_bound_l168_16898

theorem problem_bound (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 1) : 
  0 ≤ y * z + z * x + x * y - 2 * (x * y * z) ∧ 
  y * z + z * x + x * y - 2 * (x * y * z) ≤ 7 / 27 :=
sorry

end problem_bound_l168_16898


namespace trees_after_planting_l168_16893

variable (x : ℕ)

theorem trees_after_planting (x : ℕ) : 
  let additional_trees := 12
  let days := 12 / 2
  let trees_removed := days * 3
  x + additional_trees - trees_removed = x - 6 :=
by
  let additional_trees := 12
  let days := 12 / 2
  let trees_removed := days * 3
  sorry

end trees_after_planting_l168_16893


namespace arithmetic_sequence_50th_term_l168_16841

-- Definitions based on the conditions stated
def first_term := 3
def common_difference := 5
def n := 50

-- Function to calculate the n-th term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- The theorem that needs to be proven
theorem arithmetic_sequence_50th_term : nth_term first_term common_difference n = 248 := 
by
  sorry

end arithmetic_sequence_50th_term_l168_16841


namespace inequality_solution_set_l168_16830

theorem inequality_solution_set :
  {x : ℝ | (3 * x + 1) / (1 - 2 * x) ≥ 0} = {x : ℝ | -1 / 3 ≤ x ∧ x < 1 / 2} := by
  sorry

end inequality_solution_set_l168_16830


namespace showUpPeopleFirstDay_l168_16843

def cansFood := 2000
def people1stDay (cansTaken_1stDay : ℕ) := cansFood - 1500 = cansTaken_1stDay
def peopleSnapped_1stDay := 500

theorem showUpPeopleFirstDay :
  (people1stDay peopleSnapped_1stDay) → (peopleSnapped_1stDay / 1) = 500 := 
by 
  sorry

end showUpPeopleFirstDay_l168_16843


namespace evaluate_expression_l168_16818

theorem evaluate_expression (x z : ℝ) (h1 : x ≠ 0) (h2 : z ≠ 0) (y : ℝ) (h3 : y = 1 / x + z) : 
    (x - 1 / x) * (y + 1 / y) = (x^2 - 1) * (1 + 2 * x * z + x^2 * z^2 + x^2) / (x^2 * (1 + x * z)) := by
  sorry

end evaluate_expression_l168_16818


namespace evaluate_expr_l168_16827

theorem evaluate_expr (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 5 * x^(y+1) + 6 * y^(x+1) = 2751 :=
by
  rw [h₁, h₂]
  rfl

end evaluate_expr_l168_16827


namespace trapezoid_height_ratios_l168_16806

theorem trapezoid_height_ratios (A B C D O M N K L : ℝ) (h : ℝ) (h_AD : D = 2 * B) 
  (h_OK : K = h / 3) (h_OL : L = (2 * h) / 3) :
  (K / h = 1 / 3) ∧ (L / h = 2 / 3) := by
  sorry

end trapezoid_height_ratios_l168_16806


namespace tony_water_drink_l168_16849

theorem tony_water_drink (W : ℝ) (h : W - 0.04 * W = 48) : W = 50 :=
sorry

end tony_water_drink_l168_16849


namespace min_value_expression_l168_16856

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 := by
  sorry

end min_value_expression_l168_16856


namespace parallelogram_theorem_l168_16809

noncomputable def parallelogram (A B C D O : Type) (θ : ℝ) :=
  let DBA := θ
  let DBC := 3 * θ
  let CAB := 9 * θ
  let ACB := 180 - (9 * θ + 3 * θ)
  let AOB := 180 - 12 * θ
  let s := ACB / AOB
  s = 4 / 5

theorem parallelogram_theorem (A B C D O : Type) (θ : ℝ) 
  (h1: θ > 0): parallelogram A B C D O θ := by
  sorry

end parallelogram_theorem_l168_16809


namespace reciprocal_pair_c_l168_16885

def is_reciprocal (a b : ℝ) : Prop :=
  a * b = 1

theorem reciprocal_pair_c :
  is_reciprocal (-2) (-1/2) :=
by sorry

end reciprocal_pair_c_l168_16885


namespace percentage_spent_on_household_items_l168_16873

theorem percentage_spent_on_household_items (monthly_income : ℝ) (savings : ℝ) (clothes_percentage : ℝ) (medicines_percentage : ℝ) (household_spent : ℝ) : 
  monthly_income = 40000 ∧ 
  savings = 9000 ∧ 
  clothes_percentage = 0.25 ∧ 
  medicines_percentage = 0.075 ∧ 
  household_spent = monthly_income - (clothes_percentage * monthly_income + medicines_percentage * monthly_income + savings)
  → (household_spent / monthly_income) * 100 = 45 :=
by
  intro h
  cases' h with h1 h_rest
  cases' h_rest with h2 h_rest
  cases' h_rest with h3 h_rest
  cases' h_rest with h4 h5
  have h_clothes := h3
  have h_medicines := h4
  have h_savings := h2
  have h_income := h1
  have h_household := h5
  sorry

end percentage_spent_on_household_items_l168_16873


namespace coord_of_point_M_in_third_quadrant_l168_16861

noncomputable def point_coordinates (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0 ∧ abs y = 1 ∧ abs x = 2

theorem coord_of_point_M_in_third_quadrant : 
  ∃ (x y : ℝ), point_coordinates x y ∧ (x, y) = (-2, -1) := 
by {
  sorry
}

end coord_of_point_M_in_third_quadrant_l168_16861


namespace abs_diff_eq_sqrt_l168_16834

theorem abs_diff_eq_sqrt (x1 x2 a b : ℝ) (h1 : x1 + x2 = a) (h2 : x1 * x2 = b) : 
  |x1 - x2| = Real.sqrt (a^2 - 4 * b) :=
by
  sorry

end abs_diff_eq_sqrt_l168_16834


namespace gerald_added_crayons_l168_16862

namespace Proof

variable (original_crayons : ℕ) (total_crayons : ℕ)

theorem gerald_added_crayons (h1 : original_crayons = 7) (h2 : total_crayons = 13) : 
  total_crayons - original_crayons = 6 := by
  sorry

end Proof

end gerald_added_crayons_l168_16862


namespace unique_zero_identity_l168_16815

theorem unique_zero_identity (n : ℤ) : (∀ z : ℤ, z + n = z ∧ z * n = 0) → n = 0 :=
by
  intro h
  have h1 : ∀ z : ℤ, z + n = z := fun z => (h z).left
  have h2 : ∀ z : ℤ, z * n = 0 := fun z => (h z).right
  sorry

end unique_zero_identity_l168_16815


namespace probability_first_two_heads_l168_16819

-- The probability of getting heads in a single flip of a fair coin
def probability_heads_single_flip : ℚ := 1 / 2

-- Independence of coin flips
def independent_flips {α : Type} (p : α → Prop) := ∀ a b : α, a ≠ b → p a ∧ p b

-- The event of getting heads on a coin flip
def heads_event : Prop := true

-- Problem statement: The probability that the first two flips are both heads
theorem probability_first_two_heads : probability_heads_single_flip * probability_heads_single_flip = 1 / 4 :=
by
  sorry

end probability_first_two_heads_l168_16819


namespace petes_average_speed_is_correct_l168_16859

-- Definition of the necessary constants
def map_distance := 5.0 -- inches
def scale := 0.023809523809523808 -- inches per mile
def travel_time := 3.5 -- hours

-- The real distance calculation based on the given map scale
def real_distance := map_distance / scale -- miles

-- Proving the average speed calculation
def average_speed := real_distance / travel_time -- miles per hour

-- Theorem statement: Pete's average speed calculation is correct
theorem petes_average_speed_is_correct : average_speed = 60 :=
by
  -- Proof outline
  -- The real distance is 5 / 0.023809523809523808 ≈ 210
  -- The average speed is 210 / 3.5 ≈ 60
  sorry

end petes_average_speed_is_correct_l168_16859


namespace B_initial_investment_l168_16850

theorem B_initial_investment (B : ℝ) :
  let A_initial := 2000
  let A_months := 12
  let A_withdraw := 1000
  let B_advanced := 1000
  let months_before_change := 8
  let months_after_change := 4
  let total_profit := 630
  let A_share := 175
  let B_share := total_profit - A_share
  let A_investment := A_initial * A_months
  let B_investment := (B * months_before_change) + ((B + B_advanced) * months_after_change)
  (B_share / A_share = B_investment / A_investment) →
  B = 4866.67 :=
sorry

end B_initial_investment_l168_16850


namespace solution_inequalities_l168_16811

theorem solution_inequalities (x : ℝ) :
  (x^2 - 12 * x + 32 > 0) ∧ (x^2 - 13 * x + 22 < 0) → 2 < x ∧ x < 4 :=
by
  intro h
  sorry

end solution_inequalities_l168_16811


namespace unique_triangle_determination_l168_16889

-- Definitions for each type of triangle and their respective conditions
def isosceles_triangle (base_angle : ℝ) (altitude : ℝ) : Type := sorry
def vertex_base_isosceles_triangle (vertex_angle : ℝ) (base : ℝ) : Type := sorry
def circ_radius_side_equilateral_triangle (radius : ℝ) (side : ℝ) : Type := sorry
def leg_radius_right_triangle (leg : ℝ) (radius : ℝ) : Type := sorry
def angles_side_scalene_triangle (angle1 : ℝ) (angle2 : ℝ) (opp_side : ℝ) : Type := sorry

-- Condition: Option A does not uniquely determine a triangle
def option_A_does_not_uniquely_determine : Prop :=
  ∀ (base_angle altitude : ℝ), 
    (∃ t1 t2 : isosceles_triangle base_angle altitude, t1 ≠ t2)

-- Condition: Options B through E uniquely determine the triangle
def options_B_to_E_uniquely_determine : Prop :=
  (∀ (vertex_angle base : ℝ), ∃! t : vertex_base_isosceles_triangle vertex_angle base, true) ∧
  (∀ (radius side : ℝ), ∃! t : circ_radius_side_equilateral_triangle radius side, true) ∧
  (∀ (leg radius : ℝ), ∃! t : leg_radius_right_triangle leg radius, true) ∧
  (∀ (angle1 angle2 opp_side : ℝ), ∃! t : angles_side_scalene_triangle angle1 angle2 opp_side, true)

-- Main theorem combining both conditions
theorem unique_triangle_determination :
  option_A_does_not_uniquely_determine ∧ options_B_to_E_uniquely_determine :=
  sorry

end unique_triangle_determination_l168_16889


namespace inequality_iff_l168_16890

theorem inequality_iff (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : (a > b) ↔ (1/a < 1/b) = false :=
by
  sorry

end inequality_iff_l168_16890


namespace possible_perimeters_l168_16869

theorem possible_perimeters (a b c: ℝ) (h1: a = 1) (h2: b = 1) 
  (h3: c = 1) (h: ∀ x y z: ℝ, x = y ∧ y = z):
  ∃ x y: ℝ, (x = 8/3 ∧ y = 5/2) := 
  by
    sorry

end possible_perimeters_l168_16869


namespace right_triangle_isosceles_l168_16837

-- Define the conditions for a right-angled triangle inscribed in a circle
variables (a b : ℝ)

-- Conditions provided in the problem
def right_triangle_inscribed (a b : ℝ) : Prop :=
  ∃ h : a ≠ 0 ∧ b ≠ 0, 2 * (a^2 + b^2) = (a + 2*b)^2 + b^2 ∧ 2 * (a^2 + b^2) = (2 * a + b)^2 + a^2

-- The theorem to prove based on the conditions
theorem right_triangle_isosceles (a b : ℝ) (h : right_triangle_inscribed a b) : a = b :=
by 
  sorry

end right_triangle_isosceles_l168_16837


namespace terrell_weight_lifting_l168_16828

theorem terrell_weight_lifting (n : ℝ) : 
  (2 * 25 * 10 = 500) → (2 * 20 * n = 500) → n = 12.5 :=
by
  intros h1 h2
  sorry

end terrell_weight_lifting_l168_16828


namespace value_a2_plus_b2_l168_16821

noncomputable def a_minus_b : ℝ := 8
noncomputable def ab : ℝ := 49.99999999999999

theorem value_a2_plus_b2 (a b : ℝ) (h1 : a - b = a_minus_b) (h2 : a * b = ab) :
  a^2 + b^2 = 164 := by
  sorry

end value_a2_plus_b2_l168_16821


namespace complement_union_eq_l168_16855

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3}
def N : Set ℕ := {3, 5}

theorem complement_union_eq :
  (U \ (M ∪ N)) = {2, 4} := by
  sorry

end complement_union_eq_l168_16855


namespace sarah_stamp_collection_value_l168_16887

theorem sarah_stamp_collection_value :
  ∀ (stamps_owned total_value_for_4_stamps : ℝ) (num_stamps_single_series : ℕ), 
  stamps_owned = 20 → 
  total_value_for_4_stamps = 10 → 
  num_stamps_single_series = 4 → 
  (stamps_owned / num_stamps_single_series) * (total_value_for_4_stamps / num_stamps_single_series) = 50 :=
by
  intros stamps_owned total_value_for_4_stamps num_stamps_single_series 
  intro h_stamps_owned
  intro h_total_value_for_4_stamps
  intro h_num_stamps_single_series
  rw [h_stamps_owned, h_total_value_for_4_stamps, h_num_stamps_single_series]
  sorry

end sarah_stamp_collection_value_l168_16887


namespace age_of_b_l168_16882

variable (a b c : ℕ)

-- Conditions
def condition1 : Prop := a = b + 2
def condition2 : Prop := b = 2 * c
def condition3 : Prop := a + b + c = 27

theorem age_of_b (h1 : condition1 a b)
                 (h2 : condition2 b c)
                 (h3 : condition3 a b c) : 
                 b = 10 := 
by sorry

end age_of_b_l168_16882


namespace find_polar_equations_and_distance_l168_16864

noncomputable def polar_equation_C1 (rho theta : ℝ) : Prop :=
  rho^2 * Real.cos (2 * theta) = 1

noncomputable def polar_equation_C2 (rho theta : ℝ) : Prop :=
  rho = 2 * Real.cos theta

theorem find_polar_equations_and_distance :
  (∀ rho theta, polar_equation_C1 rho theta ↔ rho^2 * Real.cos (2 * theta) = 1) ∧
  (∀ rho theta, polar_equation_C2 rho theta ↔ rho = 2 * Real.cos theta) ∧
  let theta := Real.pi / 6
  let rho_A := Real.sqrt 2
  let rho_B := Real.sqrt 3
  (|rho_A - rho_B| = |Real.sqrt 3 - Real.sqrt 2|) :=
  by sorry

end find_polar_equations_and_distance_l168_16864


namespace find_x_l168_16872

theorem find_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 :=
by
  sorry

end find_x_l168_16872


namespace count_base_8_digits_5_or_6_l168_16886

-- Define the conditions in Lean
def is_digit_5_or_6 (d : ℕ) : Prop := d = 5 ∨ d = 6

def count_digits_5_or_6 := 
  let total_base_8 := 512
  let total_without_5_6 := 6 * 6 * 6 -- since we exclude 2 out of 8 digits
  total_base_8 - total_without_5_6

-- The statement of the proof problem
theorem count_base_8_digits_5_or_6 : count_digits_5_or_6 = 296 :=
by {
  sorry
}

end count_base_8_digits_5_or_6_l168_16886


namespace work_done_at_4_pm_l168_16865

noncomputable def workCompletionTime (aHours : ℝ) (bHours : ℝ) (startTime : ℝ) : ℝ :=
  let aRate := 1 / aHours
  let bRate := 1 / bHours
  let cycleWork := aRate + bRate
  let cyclesNeeded := (1 : ℝ) / cycleWork
  startTime + 2 * cyclesNeeded

theorem work_done_at_4_pm :
  workCompletionTime 8 12 6 = 16 :=  -- 16 in 24-hour format is 4 pm
by 
  sorry

end work_done_at_4_pm_l168_16865


namespace choir_members_max_l168_16884

theorem choir_members_max (m y n : ℕ) (h_square : m = y^2 + 11) (h_rect : m = n * (n + 5)) : 
  m = 300 := 
sorry

end choir_members_max_l168_16884


namespace edward_lawns_forgotten_l168_16852

theorem edward_lawns_forgotten (dollars_per_lawn : ℕ) (total_lawns : ℕ) (total_earned : ℕ) (lawns_mowed : ℕ) (lawns_forgotten : ℕ) :
  dollars_per_lawn = 4 →
  total_lawns = 17 →
  total_earned = 32 →
  lawns_mowed = total_earned / dollars_per_lawn →
  lawns_forgotten = total_lawns - lawns_mowed →
  lawns_forgotten = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end edward_lawns_forgotten_l168_16852


namespace probability_mass_range_l168_16863

/-- Let ξ be a random variable representing the mass of a badminton product. 
    Suppose P(ξ < 4.8) = 0.3 and P(ξ ≥ 4.85) = 0.32. 
    We want to prove that the probability that the mass is in the range [4.8, 4.85) is 0.38. -/
theorem probability_mass_range (P : ℝ → ℝ) (h1 : P (4.8) = 0.3) (h2 : P (4.85) = 0.32) :
  P (4.8) - P (4.85) = 0.38 :=
by 
  sorry

end probability_mass_range_l168_16863


namespace mildred_weight_l168_16880

theorem mildred_weight (carol_weight mildred_is_heavier : ℕ) (h1 : carol_weight = 9) (h2 : mildred_is_heavier = 50) :
  carol_weight + mildred_is_heavier = 59 :=
by
  sorry

end mildred_weight_l168_16880


namespace inequality_of_f_l168_16842

def f (x : ℝ) : ℝ := 3 * (x - 2)^2 + 5

theorem inequality_of_f (x₁ x₂ : ℝ) (h : |x₁ - 2| > |x₂ - 2|) : f x₁ > f x₂ :=
by
  -- sorry placeholder for the actual proof
  sorry

end inequality_of_f_l168_16842


namespace find_a4_a5_l168_16823

variable {α : Type*} [LinearOrderedField α]

-- Variables representing the terms of the geometric sequence
variables (a₁ a₂ a₃ a₄ a₅ q : α)

-- Conditions given in the problem
-- Geometric sequence condition
def is_geometric_sequence (a₁ a₂ a₃ a₄ a₅ q : α) : Prop :=
  a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q ∧ a₅ = a₄ * q

-- First condition
def condition1 : Prop := a₁ + a₂ = 3

-- Second condition
def condition2 : Prop := a₂ + a₃ = 6

-- Theorem stating that a₄ + a₅ = 24 given the conditions
theorem find_a4_a5
  (h1 : condition1 a₁ a₂)
  (h2 : condition2 a₂ a₃)
  (hg : is_geometric_sequence a₁ a₂ a₃ a₄ a₅ q) :
  a₄ + a₅ = 24 := 
sorry

end find_a4_a5_l168_16823


namespace arithmetic_progression_common_difference_l168_16867

theorem arithmetic_progression_common_difference :
  ∀ (A1 An n d : ℕ), A1 = 3 → An = 103 → n = 21 → An = A1 + (n - 1) * d → d = 5 :=
by
  intros A1 An n d h1 h2 h3 h4
  sorry

end arithmetic_progression_common_difference_l168_16867


namespace N_10_first_player_wins_N_12_first_player_wins_N_15_second_player_wins_N_30_first_player_wins_l168_16816

open Nat -- Natural numbers framework

-- Definitions for game conditions would go here. We assume them to be defined as:
-- structure GameCondition (N : ℕ) :=
-- (players_take_turns_to_circle_numbers_from_1_to_N : Prop)
-- (any_two_circled_numbers_must_be_coprime : Prop)
-- (a_number_cannot_be_circled_twice : Prop)
-- (player_who_cannot_move_loses : Prop)

inductive Player
| first
| second

-- Definitions indicating which player wins for a given N
def first_player_wins (N : ℕ) : Prop := sorry
def second_player_wins (N : ℕ) : Prop := sorry

-- For N = 10
theorem N_10_first_player_wins : first_player_wins 10 := sorry

-- For N = 12
theorem N_12_first_player_wins : first_player_wins 12 := sorry

-- For N = 15
theorem N_15_second_player_wins : second_player_wins 15 := sorry

-- For N = 30
theorem N_30_first_player_wins : first_player_wins 30 := sorry

end N_10_first_player_wins_N_12_first_player_wins_N_15_second_player_wins_N_30_first_player_wins_l168_16816


namespace lucy_crayons_l168_16840

theorem lucy_crayons (W L : ℕ) (h1 : W = 1400) (h2 : W = L + 1110) : L = 290 :=
by {
  sorry
}

end lucy_crayons_l168_16840


namespace smallest_prime_divides_polynomial_l168_16814

theorem smallest_prime_divides_polynomial : 
  ∃ n : ℤ, n^2 + 5 * n + 23 = 17 := 
sorry

end smallest_prime_divides_polynomial_l168_16814


namespace simplify_expr_l168_16891

theorem simplify_expr : 3 * (4 - 2 * Complex.I) - 2 * Complex.I * (3 - 2 * Complex.I) = 8 - 12 * Complex.I :=
by
  sorry

end simplify_expr_l168_16891


namespace find_ec_l168_16832

theorem find_ec (angle_A : ℝ) (BC : ℝ) (BD_perp_AC : Prop) (CE_perp_AB : Prop)
  (angle_DBC_2_angle_ECB : Prop) :
  angle_A = 45 ∧ 
  BC = 8 ∧
  BD_perp_AC ∧
  CE_perp_AB ∧
  angle_DBC_2_angle_ECB → 
  ∃ (a b c : ℕ), a = 3 ∧ b = 2 ∧ c = 2 ∧ a + b + c = 7 :=
sorry

end find_ec_l168_16832


namespace probability_two_draws_l168_16895

def probability_first_red_second_kd (total_cards : ℕ) (red_cards : ℕ) (king_of_diamonds : ℕ) : ℚ :=
  (red_cards / total_cards) * (king_of_diamonds / (total_cards - 1))

theorem probability_two_draws :
  let total_cards := 52
  let red_cards := 26
  let king_of_diamonds := 1
  probability_first_red_second_kd total_cards red_cards king_of_diamonds = 1 / 102 :=
by {
  sorry
}

end probability_two_draws_l168_16895


namespace algebraic_expression_value_l168_16897

theorem algebraic_expression_value 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = ab + bc + ac)
  (h2 : a = 1) : 
  (a + b - c) ^ 2004 = 1 := 
by
  sorry

end algebraic_expression_value_l168_16897


namespace function_sqrt_plus_one_l168_16825

variable (f : ℝ → ℝ)
variable (x : ℝ)

theorem function_sqrt_plus_one (h1 : ∀ x : ℝ, f x = 3) (h2 : x ≥ 0) : f (Real.sqrt x) + 1 = 4 :=
by
  sorry

end function_sqrt_plus_one_l168_16825


namespace animals_in_field_l168_16866

def dog := 1
def cats := 4
def rabbits_per_cat := 2
def hares_per_rabbit := 3

def rabbits := cats * rabbits_per_cat
def hares := rabbits * hares_per_rabbit

def total_animals := dog + cats + rabbits + hares

theorem animals_in_field : total_animals = 37 := by
  sorry

end animals_in_field_l168_16866


namespace arthur_spent_on_second_day_l168_16874

variable (H D : ℝ)
variable (a1 : 3 * H + 4 * D = 10)
variable (a2 : D = 1)

theorem arthur_spent_on_second_day :
  2 * H + 3 * D = 7 :=
by
  sorry

end arthur_spent_on_second_day_l168_16874


namespace find_a_plus_b_l168_16835

theorem find_a_plus_b (a b : ℝ) (h1 : 2 * a = -6) (h2 : a^2 - b = 4) : a + b = 2 := 
by 
  sorry

end find_a_plus_b_l168_16835


namespace indolent_student_probability_l168_16826

-- Define the constants of the problem
def n : ℕ := 30  -- total number of students
def k : ℕ := 3   -- number of students selected each lesson
def m : ℕ := 10  -- number of students from the previous lesson

-- Define the probabilities
def P_asked_in_one_lesson : ℚ := 1 / k
def P_asked_twice_in_a_row : ℚ := 1 / n
def P_overall : ℚ := P_asked_in_one_lesson + P_asked_in_one_lesson - P_asked_twice_in_a_row
def P_avoid_reciting : ℚ := 1 - P_overall

theorem indolent_student_probability : P_avoid_reciting = 11 / 30 := 
  sorry

end indolent_student_probability_l168_16826


namespace projection_is_correct_l168_16846

theorem projection_is_correct :
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (4, -1)
  let p : ℝ × ℝ := (15/58, 35/58)
  let d : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)
  ∃ v : ℝ × ℝ, 
    (a.1 * v.1 + a.2 * v.2 = p.1 * v.1 + p.2 * v.2) ∧
    (b.1 * v.1 + b.2 * v.2 = p.1 * v.1 + p.2 * v.2) ∧ 
    (p.1 * d.1 + p.2 * d.2 = 0) :=
sorry

end projection_is_correct_l168_16846


namespace average_length_of_ropes_l168_16854

def length_rope_1 : ℝ := 2
def length_rope_2 : ℝ := 6

theorem average_length_of_ropes :
  (length_rope_1 + length_rope_2) / 2 = 4 :=
by
  sorry

end average_length_of_ropes_l168_16854


namespace tom_needs_495_boxes_l168_16845

-- Define the conditions
def total_chocolate_bars : ℕ := 3465
def chocolate_bars_per_box : ℕ := 7

-- Define the proof statement
theorem tom_needs_495_boxes : total_chocolate_bars / chocolate_bars_per_box = 495 :=
by
  sorry

end tom_needs_495_boxes_l168_16845


namespace solution_set_of_inequality_l168_16860

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | (x - 1) * (a * x - 4) < 0} = {x : ℝ | x > 1 ∨ x < 4 / a} :=
sorry

end solution_set_of_inequality_l168_16860


namespace alice_walks_distance_l168_16844

theorem alice_walks_distance :
  let blocks_south := 5
  let blocks_west := 8
  let distance_per_block := 1 / 4
  let total_blocks := blocks_south + blocks_west
  let total_distance := total_blocks * distance_per_block
  total_distance = 3.25 :=
by
  sorry

end alice_walks_distance_l168_16844


namespace jimmy_hostel_stay_days_l168_16879

-- Definitions based on the conditions
def nightly_hostel_charge : ℕ := 15
def nightly_cabin_charge_per_person : ℕ := 15
def total_lodging_expense : ℕ := 75
def days_in_cabin : ℕ := 2

-- The proof statement
theorem jimmy_hostel_stay_days : 
    ∃ x : ℕ, (nightly_hostel_charge * x + nightly_cabin_charge_per_person * days_in_cabin = total_lodging_expense) ∧ x = 3 := by
    sorry

end jimmy_hostel_stay_days_l168_16879


namespace day_crew_fraction_l168_16876

theorem day_crew_fraction (D W : ℝ) (h1 : D > 0) (h2 : W > 0) :
  (D * W / (D * W + (3 / 4 * D * 1 / 2 * W)) = 8 / 11) :=
by
  sorry

end day_crew_fraction_l168_16876


namespace infinite_rational_solutions_x3_y3_9_l168_16871

theorem infinite_rational_solutions_x3_y3_9 :
  ∃ (S : Set (ℚ × ℚ)), S.Infinite ∧ (∀ (x y : ℚ), (x, y) ∈ S → x^3 + y^3 = 9) :=
sorry

end infinite_rational_solutions_x3_y3_9_l168_16871


namespace find_Y_payment_l168_16836

theorem find_Y_payment 
  (P X Z : ℝ)
  (total_payment : ℝ)
  (h1 : P + X + Z = total_payment)
  (h2 : X = 1.2 * P)
  (h3 : Z = 0.96 * P) :
  P = 332.28 := by
  sorry

end find_Y_payment_l168_16836


namespace total_points_l168_16888

theorem total_points (darius_score marius_score matt_score total_points : ℕ) 
    (h1 : darius_score = 10) 
    (h2 : marius_score = darius_score + 3) 
    (h3 : matt_score = darius_score + 5) 
    (h4 : total_points = darius_score + marius_score + matt_score) : 
    total_points = 38 :=
by sorry

end total_points_l168_16888


namespace area_of_inscribed_triangle_l168_16896

theorem area_of_inscribed_triangle 
  (x : ℝ) 
  (h1 : (2:ℝ) * x ≤ (3:ℝ) * x ∧ (3:ℝ) * x ≤ (4:ℝ) * x) 
  (h2 : (4:ℝ) * x = 2 * 4) :
  ∃ (area : ℝ), area = 12.00 :=
by
  sorry

end area_of_inscribed_triangle_l168_16896


namespace find_first_offset_l168_16802

theorem find_first_offset {area diagonal offset₁ offset₂ : ℝ}
  (h_area : area = 150)
  (h_diagonal : diagonal = 20)
  (h_offset₂ : offset₂ = 6) :
  2 * area = diagonal * (offset₁ + offset₂) → offset₁ = 9 := by
  sorry

end find_first_offset_l168_16802


namespace range_of_k_l168_16800

theorem range_of_k (x : ℝ) (k : ℝ) (h_pos : x > 0) (h_ne : x ≠ 2) :
  (1 / (x - 2) + 3 = (3 - k) / (2 - x)) ↔ (k > -2 ∧ k ≠ 4) :=
by
  sorry

end range_of_k_l168_16800


namespace length_of_bridge_l168_16838

theorem length_of_bridge (length_of_train : ℕ) (speed_of_train_kmh : ℕ) (time_to_cross : ℕ)
  (lt : length_of_train = 140)
  (st : speed_of_train_kmh = 45)
  (tc : time_to_cross = 30) : 
  ∃ length_of_bridge, length_of_bridge = 235 := 
by 
  sorry

end length_of_bridge_l168_16838


namespace L_shaped_figure_area_l168_16875

noncomputable def area_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  length * width

theorem L_shaped_figure_area :
  let large_rect_length := 10
  let large_rect_width := 7
  let small_rect_length := 4
  let small_rect_width := 3
  area_rectangle large_rect_length large_rect_width - area_rectangle small_rect_length small_rect_width = 58 :=
by
  sorry

end L_shaped_figure_area_l168_16875


namespace greatest_two_digit_product_12_l168_16847

theorem greatest_two_digit_product_12 : ∃ (a b : ℕ), 10 * a + b = 62 ∧ a * b = 12 ∧ 10 ≤ 10 * a + b  ∧ 10 * a + b < 100 :=
by
  sorry

end greatest_two_digit_product_12_l168_16847


namespace max_product_of_sum_2016_l168_16851

theorem max_product_of_sum_2016 (x y : ℤ) (h : x + y = 2016) : x * y ≤ 1016064 :=
by
  -- Proof goes here, but is not needed as per instructions
  sorry

end max_product_of_sum_2016_l168_16851


namespace type1_pieces_count_l168_16848

theorem type1_pieces_count (n : ℕ) (pieces : ℕ → ℕ)  (nonNegative : ∀ i, pieces i ≥ 0) :
  pieces 1 ≥ 4 * n - 1 :=
sorry

end type1_pieces_count_l168_16848


namespace relationship_t_s_l168_16833

variable {a b : ℝ}

theorem relationship_t_s (a b : ℝ) (t : ℝ) (s : ℝ) (ht : t = a + 2 * b) (hs : s = a + b^2 + 1) :
  t ≤ s := 
sorry

end relationship_t_s_l168_16833


namespace find_k_in_geometric_sequence_l168_16868

theorem find_k_in_geometric_sequence (a : ℕ → ℕ) (k : ℕ)
  (h1 : ∀ n, a n = a 2 * 3^(n-2))
  (h2 : a 2 = 3)
  (h3 : a 3 = 9)
  (h4 : a k = 243) :
  k = 6 :=
sorry

end find_k_in_geometric_sequence_l168_16868


namespace quadratic_inequality_sum_l168_16892

theorem quadratic_inequality_sum (a b : ℝ) (h1 : 1 < 2) 
 (h2 : ∀ x : ℝ, 1 < x ∧ x < 2 → x^2 - a * x + b < 0) 
 (h3 : 1 + 2 = a)  (h4 : 1 * 2 = b) : 
 a + b = 5 := 
by 
sorry

end quadratic_inequality_sum_l168_16892


namespace numbers_identification_l168_16853

-- Definitions
def is_natural (n : ℤ) : Prop := n ≥ 0
def is_integer (n : ℤ) : Prop := True

-- Theorem
theorem numbers_identification :
  (is_natural 0 ∧ is_natural 2 ∧ is_natural 6 ∧ is_natural 7) ∧
  (is_integer (-15) ∧ is_integer (-3) ∧ is_integer 0 ∧ is_integer 4) :=
by
  sorry

end numbers_identification_l168_16853


namespace quadratic_roots_problem_l168_16804

theorem quadratic_roots_problem 
  (x y : ℤ) 
  (h1 : x + y = 10)
  (h2 : |x - y| = 12) :
  (x - 11) * (x + 1) = 0 :=
sorry

end quadratic_roots_problem_l168_16804


namespace largest_4_digit_divisible_by_12_l168_16807

theorem largest_4_digit_divisible_by_12 : ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ 12 ∣ n ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ 12 ∣ m → m ≤ n :=
sorry

end largest_4_digit_divisible_by_12_l168_16807


namespace snail_climbs_well_l168_16803

theorem snail_climbs_well (h : ℕ) (c : ℕ) (s : ℕ) (d : ℕ) (h_eq : h = 12) (c_eq : c = 3) (s_eq : s = 2) : d = 10 :=
by
  sorry

end snail_climbs_well_l168_16803


namespace max_modulus_l168_16817

open Complex

theorem max_modulus (z : ℂ) (h : abs z = 1) : ∃ M, M = 6 ∧ ∀ w, abs (z - w) ≤ M :=
by
  use 6
  sorry

end max_modulus_l168_16817


namespace minimize_transfers_l168_16883

-- Define the initial number of pieces in each supermarket
def pieces_in_A := 15
def pieces_in_B := 7
def pieces_in_C := 11
def pieces_in_D := 3
def pieces_in_E := 14

-- Define the target number of pieces in each supermarket after transfers
def target_pieces := 10

-- Define a function to compute the total number of pieces
def total_pieces := pieces_in_A + pieces_in_B + pieces_in_C + pieces_in_D + pieces_in_E

-- Define the minimum number of transfers needed
def min_transfers := 12

-- The main theorem: proving that the minimum number of transfers is 12
theorem minimize_transfers : 
  total_pieces = 5 * target_pieces → 
  ∃ (transfers : ℕ), transfers = min_transfers :=
by
  -- This represents the proof section, we leave it as sorry
  sorry

end minimize_transfers_l168_16883


namespace initial_cats_l168_16820

theorem initial_cats (C : ℕ) (h1 : 36 + 12 - 20 + C = 57) : C = 29 :=
by
  sorry

end initial_cats_l168_16820


namespace enrique_speed_l168_16810

theorem enrique_speed (distance : ℝ) (time : ℝ) (speed_diff : ℝ) (E : ℝ) :
  distance = 200 ∧ time = 8 ∧ speed_diff = 7 ∧ 
  (2 * E + speed_diff) * time = distance → 
  E = 9 :=
by
  sorry

end enrique_speed_l168_16810


namespace number_of_games_l168_16801

theorem number_of_games (total_points points_per_game : ℕ) (h1 : total_points = 21) (h2 : points_per_game = 7) : total_points / points_per_game = 3 := by
  sorry

end number_of_games_l168_16801


namespace win_sector_area_l168_16824

/-- Given a circular spinner with a radius of 8 cm and the probability of winning being 3/8,
    prove that the area of the WIN sector is 24π square centimeters. -/
theorem win_sector_area (r : ℝ) (P_win : ℝ) (area_WIN : ℝ) :
  r = 8 → P_win = 3 / 8 → area_WIN = 24 * Real.pi := by
sorry

end win_sector_area_l168_16824


namespace expression_evaluation_l168_16822

def e : Int := -(-1) + 3^2 / (1 - 4) * 2

theorem expression_evaluation : e = -5 := 
by
  unfold e
  sorry

end expression_evaluation_l168_16822


namespace compute_z_pow_7_l168_16805

namespace ComplexProof

noncomputable def z : ℂ := (Real.sqrt 3 + Complex.I) / 2

theorem compute_z_pow_7 : z ^ 7 = - (Real.sqrt 3 / 2) - (1 / 2) * Complex.I :=
by
  sorry

end ComplexProof

end compute_z_pow_7_l168_16805


namespace parabola_transform_correct_l168_16829

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := -2 * x^2 + 1

-- Define the transformation of moving the parabola one unit to the right and one unit up
def transformed_parabola (x : ℝ) : ℝ := -2 * (x - 1)^2 + 2

-- The theorem to prove
theorem parabola_transform_correct :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 1) + 1 :=
by
  intros x
  sorry

end parabola_transform_correct_l168_16829


namespace complex_number_in_second_quadrant_l168_16878

theorem complex_number_in_second_quadrant :
  let z := (2 + 4 * Complex.I) / (1 + Complex.I) 
  ∃ (im : ℂ), z = im ∧ im.re < 0 ∧ 0 < im.im := by
  sorry

end complex_number_in_second_quadrant_l168_16878


namespace math_proof_problem_l168_16870

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def complement_R (s : Set ℝ) := {x : ℝ | x ∉ s}

theorem math_proof_problem :
  (complement_R A ∩ B) = {x | 2 < x ∧ x ≤ 3} :=
sorry

end math_proof_problem_l168_16870
