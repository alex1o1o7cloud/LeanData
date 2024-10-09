import Mathlib

namespace xyz_sum_eq_eleven_l185_18589

theorem xyz_sum_eq_eleven (x y z : ℝ) (h : (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0) : x + y + z = 11 :=
sorry

end xyz_sum_eq_eleven_l185_18589


namespace sum_of_common_ratios_l185_18524

theorem sum_of_common_ratios (k p r : ℝ) (h1 : k ≠ 0) (h2 : k * (p^2) - k * (r^2) = 5 * (k * p - k * r)) (h3 : p ≠ r) : p + r = 5 :=
sorry

end sum_of_common_ratios_l185_18524


namespace smallest_sum_p_q_l185_18582

theorem smallest_sum_p_q (p q : ℕ) (h_pos : 1 < p) (h_cond : (p^2 * q - 1) = (2021 * p * q) / 2021) : p + q = 44 :=
sorry

end smallest_sum_p_q_l185_18582


namespace max_wrestlers_more_than_131_l185_18561

theorem max_wrestlers_more_than_131
  (n : ℤ)
  (total_wrestlers : ℤ := 20)
  (average_weight : ℕ := 125)
  (min_weight : ℕ := 90)
  (constraint1 : n ≥ 0)
  (constraint2 : n ≤ total_wrestlers)
  (total_weight := 2500) :
  n ≤ 17 :=
by
  sorry

end max_wrestlers_more_than_131_l185_18561


namespace problem_statement_l185_18522

theorem problem_statement (p q : ℝ)
  (α β : ℝ) (h1 : α ≠ β) (h1' : α + β = -p) (h1'' : α * β = -2)
  (γ δ : ℝ) (h2 : γ ≠ δ) (h2' : γ + δ = -q) (h2'' : γ * δ = -3) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = 3 * (q ^ 2 - p ^ 2) - 2 * q + 1 := by
  sorry

end problem_statement_l185_18522


namespace exists_nat_number_gt_1000_l185_18526

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem exists_nat_number_gt_1000 (S : ℕ → ℕ) :
  (∀ n : ℕ, S (2^n) = sum_of_digits (2^n)) →
  ∃ n : ℕ, n > 1000 ∧ S (2^n) > S (2^(n + 1)) :=
by sorry

end exists_nat_number_gt_1000_l185_18526


namespace total_dolls_l185_18536

def initial_dolls : ℕ := 6
def grandmother_dolls : ℕ := 30
def received_dolls : ℕ := grandmother_dolls / 2

theorem total_dolls : initial_dolls + grandmother_dolls + received_dolls = 51 :=
by
  -- Simplify the right hand side
  sorry

end total_dolls_l185_18536


namespace find_length_of_AC_l185_18515

theorem find_length_of_AC
  (A B C : Type)
  (AB : Real)
  (AC : Real)
  (Area : Real)
  (angle_A : Real)
  (h1 : AB = 8)
  (h2 : angle_A = (30 * Real.pi / 180)) -- converting degrees to radians
  (h3 : Area = 16) :
  AC = 8 :=
by
  -- Skipping proof as requested
  sorry

end find_length_of_AC_l185_18515


namespace find_a_l185_18547

def M : Set ℝ := {x | x^2 + x - 6 = 0}

def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem find_a (a : ℝ) : N a ⊆ M ↔ a = -1 ∨ a = 0 ∨ a = 2/3 := 
by
  sorry

end find_a_l185_18547


namespace max_proj_area_of_regular_tetrahedron_l185_18565

theorem max_proj_area_of_regular_tetrahedron (a : ℝ) (h_a : a > 0) : 
    ∃ max_area : ℝ, max_area = a^2 / 2 :=
by
  existsi (a^2 / 2)
  sorry

end max_proj_area_of_regular_tetrahedron_l185_18565


namespace water_wheel_effective_horsepower_l185_18501

noncomputable def effective_horsepower 
  (velocity : ℝ) (width : ℝ) (thickness : ℝ) (density : ℝ) 
  (diameter : ℝ) (efficiency : ℝ) (g : ℝ) (hp_conversion : ℝ) : ℝ :=
  let mass_flow_rate := velocity * width * thickness * density
  let kinetic_energy_per_second := 0.5 * mass_flow_rate * velocity^2
  let potential_energy_per_second := mass_flow_rate * diameter * g
  let indicated_power := kinetic_energy_per_second + potential_energy_per_second
  let horsepower := indicated_power / hp_conversion
  efficiency * horsepower

theorem water_wheel_effective_horsepower :
  effective_horsepower 1.4 0.5 0.13 1000 3 0.78 9.81 745.7 = 2.9 :=
by
  sorry

end water_wheel_effective_horsepower_l185_18501


namespace compass_legs_cannot_swap_l185_18571

-- Define the problem conditions: compass legs on infinite grid, constant distance d.
def on_grid (p q : ℤ × ℤ) : Prop := 
  ∃ d : ℕ, d * d = (p.1 - q.1) * (p.1 - q.1) + (p.2 - q.2) * (p.2 - q.2) ∧ d > 0

-- Define the main theorem as a Lean 4 statement
theorem compass_legs_cannot_swap (p q : ℤ × ℤ) (h : on_grid p q) : 
  ¬ ∃ r s : ℤ × ℤ, on_grid r p ∧ on_grid s p ∧ p ≠ q ∧ r = q ∧ s = p :=
sorry

end compass_legs_cannot_swap_l185_18571


namespace probability_of_two_red_two_green_l185_18575

def red_balls : ℕ := 10
def green_balls : ℕ := 8
def total_balls : ℕ := red_balls + green_balls
def drawn_balls : ℕ := 4

def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def prob_two_red_two_green : ℚ :=
  (combination red_balls 2 * combination green_balls 2 : ℚ) / combination total_balls drawn_balls

theorem probability_of_two_red_two_green :
  prob_two_red_two_green = 7 / 17 := 
sorry

end probability_of_two_red_two_green_l185_18575


namespace jason_borrowed_amount_l185_18583

def earning_per_six_hours : ℤ :=
  2 + 4 + 6 + 2 + 4 + 6

def total_hours_worked : ℤ :=
  48

def cycle_length : ℤ :=
  6

def total_cycles : ℤ :=
  total_hours_worked / cycle_length

def total_amount_borrowed : ℤ :=
  total_cycles * earning_per_six_hours

theorem jason_borrowed_amount : total_amount_borrowed = 192 :=
  by
    -- Here we use the definition and conditions to prove the equivalence
    -- of the calculation to the problem statement.
    sorry

end jason_borrowed_amount_l185_18583


namespace triangle_problem_l185_18552

/--
Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C, respectively, 
where:
1. b * (sin B - sin C) = a * sin A - c * sin C
2. a = 2 * sqrt 3
3. the area of triangle ABC is 2 * sqrt 3

Prove:
1. A = π / 3
2. The perimeter of triangle ABC is 2 * sqrt 3 + 6
-/
theorem triangle_problem 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : b * (Real.sin B - Real.sin C) = a * Real.sin A - c * Real.sin C)
  (h2 : a = 2 * Real.sqrt 3)
  (h3 : 0.5 * b * c * Real.sin A = 2 * Real.sqrt 3) :
  A = Real.pi / 3 ∧ a + b + c = 2 * Real.sqrt 3 + 6 := 
sorry

end triangle_problem_l185_18552


namespace number_of_always_true_inequalities_l185_18555

theorem number_of_always_true_inequalities (a b c d : ℝ) (h1 : a > b) (h2 : c > d) :
  (a + c > b + d) ∧
  (¬(a - c > b - d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 - 3 > -2 - (-2))) ∧
  (¬(a * c > b * d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 * 3 > -2 * (-2))) ∧
  (¬(a / c > b / d) ∨ ∃ a b c d, a = 1 ∧ b = -2 ∧ c = 3 ∧ d = -2 ∧ ¬(1 / 3 > (-2) / (-2))) :=
by
  sorry

end number_of_always_true_inequalities_l185_18555


namespace derek_age_l185_18558

theorem derek_age (aunt_beatrice_age : ℕ) (emily_age : ℕ) (derek_age : ℕ)
  (h1 : aunt_beatrice_age = 54)
  (h2 : emily_age = aunt_beatrice_age / 2)
  (h3 : derek_age = emily_age - 7) : derek_age = 20 :=
by
  sorry

end derek_age_l185_18558


namespace not_neighboring_root_equation_x2_x_2_neighboring_root_equation_k_values_l185_18543

def is_neighboring_root_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁ * x₁ + b * x₁ + c = 0 ∧ a * x₂ * x₂ + b * x₂ + c = 0 
  ∧ (x₁ - x₂ = 1 ∨ x₂ - x₁ = 1)

theorem not_neighboring_root_equation_x2_x_2 : 
  ¬ is_neighboring_root_equation 1 1 (-2) :=
sorry

theorem neighboring_root_equation_k_values (k : ℝ) : 
  is_neighboring_root_equation 1 (-(k-3)) (-3*k) ↔ k = -2 ∨ k = -4 :=
sorry

end not_neighboring_root_equation_x2_x_2_neighboring_root_equation_k_values_l185_18543


namespace plane_distance_l185_18521

variable (a b c p : ℝ)

def plane_intercept := (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧
  (p = 1 / (Real.sqrt ((1 / a^2) + (1 / b^2) + (1 / c^2))))

theorem plane_distance
  (h : plane_intercept a b c p) :
  1 / a^2 + 1 / b^2 + 1 / c^2 = 1 / p^2 := 
sorry

end plane_distance_l185_18521


namespace correct_option_is_c_l185_18586

variable {x y : ℕ}

theorem correct_option_is_c (hx : (x^2)^3 = x^6) :
  (∀ x : ℕ, x * x^2 ≠ x^2) →
  (∀ x y : ℕ, (x + y)^2 ≠ x^2 + y^2) →
  (∃ x : ℕ, x^2 + x^2 ≠ x^4) →
  (x^2)^3 = x^6 :=
by
  intros h1 h2 h3
  exact hx

end correct_option_is_c_l185_18586


namespace triangle_base_length_l185_18517

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ)
  (h_area : area = 24) (h_height : height = 8) (h_area_formula : area = (base * height) / 2) :
  base = 6 :=
by
  sorry

end triangle_base_length_l185_18517


namespace fraction_of_25_exists_l185_18598

theorem fraction_of_25_exists :
  ∃ x : ℚ, 0.60 * 40 = x * 25 + 4 ∧ x = 4 / 5 :=
by
  simp
  sorry

end fraction_of_25_exists_l185_18598


namespace problem1_problem2_problem3_l185_18505

-- Problem 1
theorem problem1
  (α : ℝ)
  (a : ℝ × ℝ := (1 / 2, - (Real.sqrt 3) / 2))
  (b : ℝ × ℝ := (Real.cos α, Real.sin α))
  (hα : 0 < α ∧ α < 2 * Real.pi / 3) :
  (a + b) • (a - b) = 0 :=
sorry

-- Problem 2
theorem problem2
  (α k : ℝ)
  (a : ℝ × ℝ := (1 / 2, - (Real.sqrt 3) / 2))
  (b : ℝ × ℝ := (Real.cos α, Real.sin α))
  (x : ℝ × ℝ := k • a + 3 • b)
  (y : ℝ × ℝ := a + (1 / k) • b)
  (hk : 0 < k)
  (hα : 0 < α ∧ α < 2 * Real.pi / 3)
  (hxy : x • y = 0) :
  k + 3 / k + 4 * Real.sin (Real.pi / 6 - α) = 0 :=
sorry

-- Problem 3
theorem problem3
  (α k : ℝ)
  (h_eq : k + 3 / k + 4 * Real.sin (Real.pi / 6 - α) = 0)
  (hα : 0 < α ∧ α < 2 * Real.pi / 3)
  (hk : 0 < k) :
  Real.pi / 2 ≤ α ∧ α < 2 * Real.pi / 3 :=
sorry

end problem1_problem2_problem3_l185_18505


namespace female_democrats_l185_18500

theorem female_democrats :
  ∀ (F M : ℕ),
  F + M = 720 →
  F/2 + M/4 = 240 →
  F / 2 = 120 :=
by
  intros F M h1 h2
  sorry

end female_democrats_l185_18500


namespace g_value_at_4_l185_18544

noncomputable def g : ℝ → ℝ := sorry -- We will define g here

def functional_condition (g : ℝ → ℝ) := ∀ x y : ℝ, x * g y = y * g x
def g_value_at_12 := g 12 = 30

theorem g_value_at_4 (g : ℝ → ℝ) (h₁ : functional_condition g) (h₂ : g_value_at_12) : g 4 = 10 := 
sorry

end g_value_at_4_l185_18544


namespace garden_length_l185_18566

theorem garden_length (w l : ℝ) (h1 : l = 2 + 3 * w) (h2 : 2 * l + 2 * w = 100) : l = 38 :=
sorry

end garden_length_l185_18566


namespace smallest_x_remainder_l185_18557

theorem smallest_x_remainder : ∃ x : ℕ, x > 0 ∧ 
    x % 6 = 5 ∧
    x % 7 = 6 ∧
    x % 8 = 7 ∧
    x = 167 :=
by
  sorry

end smallest_x_remainder_l185_18557


namespace largest_positive_real_root_l185_18529

theorem largest_positive_real_root (b2 b1 b0 : ℤ) (h2 : |b2| ≤ 3) (h1 : |b1| ≤ 3) (h0 : |b0| ≤ 3) :
  ∃ r : ℝ, (r > 0) ∧ (r^3 + (b2 : ℝ) * r^2 + (b1 : ℝ) * r + (b0 : ℝ) = 0) ∧ 3.5 < r ∧ r < 4.0 :=
sorry

end largest_positive_real_root_l185_18529


namespace cost_of_each_barbell_l185_18542

theorem cost_of_each_barbell (total_given change_received total_barbells : ℕ)
  (h1 : total_given = 850)
  (h2 : change_received = 40)
  (h3 : total_barbells = 3) :
  (total_given - change_received) / total_barbells = 270 :=
by
  sorry

end cost_of_each_barbell_l185_18542


namespace sculpture_paint_area_l185_18535

/-- An artist creates a sculpture using 15 cubes, each with a side length of 1 meter. 
The cubes are organized into a wall-like structure with three layers: 
the top layer consists of 3 cubes, 
the middle layer consists of 5 cubes, 
and the bottom layer consists of 7 cubes. 
Some of the cubes in the middle and bottom layers are spaced apart, exposing additional side faces. 
Prove that the total exposed surface area painted is 49 square meters. -/
theorem sculpture_paint_area :
  let cubes_sizes : ℕ := 15
  let layer_top : ℕ := 3
  let layer_middle : ℕ := 5
  let layer_bottom : ℕ := 7
  let side_exposed_area_layer_top : ℕ := layer_top * 5
  let side_exposed_area_layer_middle : ℕ := 2 * 3 + 3 * 2
  let side_exposed_area_layer_bottom : ℕ := layer_bottom * 1
  let exposed_side_faces : ℕ := side_exposed_area_layer_top + side_exposed_area_layer_middle + side_exposed_area_layer_bottom
  let exposed_top_faces : ℕ := layer_top * 1 + layer_middle * 1 + layer_bottom * 1
  let total_exposed_area : ℕ := exposed_side_faces + exposed_top_faces
  total_exposed_area = 49 := 
sorry

end sculpture_paint_area_l185_18535


namespace mass_percentage_of_Br_in_BaBr2_l185_18594

theorem mass_percentage_of_Br_in_BaBr2 :
  let Ba_molar_mass := 137.33
  let Br_molar_mass := 79.90
  let BaBr2_molar_mass := Ba_molar_mass + 2 * Br_molar_mass
  let mass_percentage_Br := (2 * Br_molar_mass / BaBr2_molar_mass) * 100
  mass_percentage_Br = 53.80 :=
by
  let Ba_molar_mass := 137.33
  let Br_molar_mass := 79.90
  let BaBr2_molar_mass := Ba_molar_mass + 2 * Br_molar_mass
  let mass_percentage_Br := (2 * Br_molar_mass / BaBr2_molar_mass) * 100
  sorry

end mass_percentage_of_Br_in_BaBr2_l185_18594


namespace power_division_result_l185_18511

theorem power_division_result : (-2)^(2014) / (-2)^(2013) = -2 :=
by
  sorry

end power_division_result_l185_18511


namespace coefficients_of_polynomial_l185_18551

theorem coefficients_of_polynomial (a_5 a_4 a_3 a_2 a_1 a_0 : ℝ) :
  (∀ x : ℝ, x^5 = a_5 * (2*x + 1)^5 + a_4 * (2*x + 1)^4 + a_3 * (2*x + 1)^3 + a_2 * (2*x + 1)^2 + a_1 * (2*x + 1) + a_0) →
  a_5 = 1/32 ∧ a_4 = -5/32 :=
by sorry

end coefficients_of_polynomial_l185_18551


namespace habitable_land_area_l185_18506

noncomputable def area_of_habitable_land : ℝ :=
  let length : ℝ := 23
  let diagonal : ℝ := 33
  let radius_of_pond : ℝ := 3
  let width : ℝ := Real.sqrt (diagonal ^ 2 - length ^ 2)
  let area_of_rectangle : ℝ := length * width
  let area_of_pond : ℝ := Real.pi * (radius_of_pond ^ 2)
  area_of_rectangle - area_of_pond

theorem habitable_land_area :
  abs (area_of_habitable_land - 515.91) < 0.01 :=
by
  sorry

end habitable_land_area_l185_18506


namespace inequality_bounds_of_xyz_l185_18590

theorem inequality_bounds_of_xyz
  (x y z : ℝ)
  (h1 : x < y) (h2 : y < z)
  (h3 : x + y + z = 6)
  (h4 : x * y + y * z + z * x = 9) :
  0 < x ∧ x < 1 ∧ 1 < y ∧ y < 3 ∧ 3 < z ∧ z < 4 := 
sorry

end inequality_bounds_of_xyz_l185_18590


namespace trigonometric_ratio_l185_18545

theorem trigonometric_ratio (θ : ℝ) (h : Real.sin θ + 2 * Real.cos θ = 1) :
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = -7 ∨
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 :=
sorry

end trigonometric_ratio_l185_18545


namespace waiters_hired_correct_l185_18516

noncomputable def waiters_hired (W H : ℕ) : Prop :=
  let cooks := 9
  (cooks / W = 3 / 8) ∧ (cooks / (W + H) = 1 / 4) ∧ (H = 12)

theorem waiters_hired_correct (W H : ℕ) : waiters_hired W H :=
  sorry

end waiters_hired_correct_l185_18516


namespace jack_mopping_rate_l185_18518

variable (bathroom_floor_area : ℕ) (kitchen_floor_area : ℕ) (time_mopped : ℕ)

theorem jack_mopping_rate
  (h_bathroom : bathroom_floor_area = 24)
  (h_kitchen : kitchen_floor_area = 80)
  (h_time : time_mopped = 13) :
  (bathroom_floor_area + kitchen_floor_area) / time_mopped = 8 :=
by
  sorry

end jack_mopping_rate_l185_18518


namespace smallest_interesting_number_l185_18562

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end smallest_interesting_number_l185_18562


namespace problem_equivalent_proof_statement_l185_18540

-- Definition of a line with a definite slope
def has_definite_slope (m : ℝ) : Prop :=
  ∃ slope : ℝ, slope = -m 

-- Definition of the equation of a line passing through two points being correct
def line_through_two_points (x1 y1 x2 y2 : ℝ) (h : x1 ≠ x2) : Prop :=
  ∀ x y : ℝ, (y - y1 = ((y2 - y1) / (x2 - x1)) * (x - x1)) ↔ y = ((y2 - y1) * (x - x1) / (x2 - x1)) + y1 

-- Formalizing and proving the given conditions
theorem problem_equivalent_proof_statement : 
  (∀ m : ℝ, has_definite_slope m) ∧ 
  (∀ (x1 y1 x2 y2 : ℝ) (h : x1 ≠ x2), line_through_two_points x1 y1 x2 y2 h) :=
by 
  sorry

end problem_equivalent_proof_statement_l185_18540


namespace solution_set_of_inequality_l185_18508

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x ≥ 0} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 5} := by
  sorry

end solution_set_of_inequality_l185_18508


namespace closest_pressure_reading_l185_18550

theorem closest_pressure_reading (x : ℝ) (h : 102.4 ≤ x ∧ x ≤ 102.8) :
    (|x - 102.5| > |x - 102.6| ∧ |x - 102.6| < |x - 102.7| ∧ |x - 102.6| < |x - 103.0|) → x = 102.6 :=
by
  sorry

end closest_pressure_reading_l185_18550


namespace sum_even_integers_102_to_200_l185_18564

theorem sum_even_integers_102_to_200 :
  let S := (List.range' 102 (200 - 102 + 1)).filter (λ x => x % 2 = 0)
  List.sum S = 7550 := by
{
  sorry
}

end sum_even_integers_102_to_200_l185_18564


namespace eval_composition_l185_18596

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 - 2

theorem eval_composition : f (g 2) = -7 := 
by {
  sorry
}

end eval_composition_l185_18596


namespace petya_mistake_l185_18591

theorem petya_mistake :
  (35 + 10 - 41 = 42 + 12 - 50) →
  (35 + 10 - 45 = 42 + 12 - 54) →
  (5 * (7 + 2 - 9) = 6 * (7 + 2 - 9)) →
  False :=
by
  intros h1 h2 h3
  sorry

end petya_mistake_l185_18591


namespace playground_area_l185_18567

theorem playground_area (L B : ℕ) (h1 : B = 6 * L) (h2 : B = 420)
  (A_total A_playground : ℕ) (h3 : A_total = L * B) 
  (h4 : A_playground = A_total / 7) :
  A_playground = 4200 :=
by sorry

end playground_area_l185_18567


namespace exists_equidistant_point_l185_18556

-- Define three points A, B, and C in 2D space
variables {A B C P: ℝ × ℝ}

-- Assume the points A, B, and C are not collinear
def not_collinear (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.2 - A.2) ≠ (C.1 - A.1) * (B.2 - A.2)

-- Define the concept of a point being equidistant from three given points
def equidistant (P A B C : ℝ × ℝ) : Prop :=
  dist P A = dist P B ∧ dist P B = dist P C

-- Define the intersection of the perpendicular bisectors of the sides of the triangle formed by A, B, and C
def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  sorry -- placeholder for the actual construction

-- The main theorem statement: If A, B, and C are not collinear, then there exists a unique point P that is equidistant from A, B, and C
theorem exists_equidistant_point (h: not_collinear A B C) :
  ∃! P, equidistant P A B C := 
sorry

end exists_equidistant_point_l185_18556


namespace cubes_in_fig_6_surface_area_fig_10_l185_18569

-- Define the function to calculate the number of unit cubes in Fig. n
def cubes_in_fig (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

-- Define the function to calculate the surface area of the solid figure for Fig. n
def surface_area_fig (n : ℕ) : ℕ := 6 * n * n

-- Theorem statements
theorem cubes_in_fig_6 : cubes_in_fig 6 = 91 :=
by sorry

theorem surface_area_fig_10 : surface_area_fig 10 = 600 :=
by sorry

end cubes_in_fig_6_surface_area_fig_10_l185_18569


namespace A_finishes_work_in_9_days_l185_18503

noncomputable def B_work_rate : ℝ := 1 / 15
noncomputable def B_work_10_days : ℝ := 10 * B_work_rate
noncomputable def remaining_work_by_A : ℝ := 1 - B_work_10_days

theorem A_finishes_work_in_9_days (A_days : ℝ) (B_days : ℝ) (B_days_worked : ℝ) (A_days_worked : ℝ) :
  (B_days = 15) ∧ (B_days_worked = 10) ∧ (A_days_worked = 3) ∧ 
  (remaining_work_by_A = (1 / 3)) → A_days = 9 :=
by sorry

end A_finishes_work_in_9_days_l185_18503


namespace inequality_proof_l185_18538

theorem inequality_proof (x : ℝ) (hx : 0 < x) : (1 / x) + 4 * (x ^ 2) ≥ 3 :=
by
  sorry

end inequality_proof_l185_18538


namespace three_digit_number_condition_l185_18578

theorem three_digit_number_condition (x y z : ℕ) (h₀ : 1 ≤ x ∧ x ≤ 9) (h₁ : 0 ≤ y ∧ y ≤ 9) (h₂ : 0 ≤ z ∧ z ≤ 9)
(h₃ : 100 * x + 10 * y + z = 34 * (x + y + z)) : 
100 * x + 10 * y + z = 102 ∨ 100 * x + 10 * y + z = 204 ∨ 100 * x + 10 * y + z = 306 ∨ 100 * x + 10 * y + z = 408 :=
sorry

end three_digit_number_condition_l185_18578


namespace vera_operations_impossible_l185_18502

theorem vera_operations_impossible (N : ℕ) : (N % 3 ≠ 0) → ¬(∃ k : ℕ, ((N + 3 * k) % 5 = 0) → ((N + 3 * k) / 5) = 1) :=
by
  sorry

end vera_operations_impossible_l185_18502


namespace max_togs_possible_l185_18592

def tag_cost : ℕ := 3
def tig_cost : ℕ := 4
def tog_cost : ℕ := 8
def total_budget : ℕ := 100
def min_tags : ℕ := 1
def min_tigs : ℕ := 1
def min_togs : ℕ := 1

theorem max_togs_possible : 
  ∃ (tags tigs togs : ℕ), tags ≥ min_tags ∧ tigs ≥ min_tigs ∧ togs ≥ min_togs ∧ 
  tag_cost * tags + tig_cost * tigs + tog_cost * togs = total_budget ∧ togs = 11 :=
sorry

end max_togs_possible_l185_18592


namespace calculation_of_product_l185_18576

theorem calculation_of_product : (0.09)^3 * 0.0007 = 0.0000005103 := 
by
  sorry

end calculation_of_product_l185_18576


namespace number_of_solutions_l185_18504

theorem number_of_solutions (n : ℕ) : (4 * n) = 80 ↔ n = 20 :=
by
  sorry

end number_of_solutions_l185_18504


namespace minimum_bail_rate_l185_18581

theorem minimum_bail_rate
  (distance : ℝ) (leak_rate : ℝ) (rain_rate : ℝ) (sink_threshold : ℝ) (rowing_speed : ℝ) (time_in_minutes : ℝ) (bail_rate : ℝ) : 
  (distance = 2) → 
  (leak_rate = 15) → 
  (rain_rate = 5) →
  (sink_threshold = 60) →
  (rowing_speed = 3) →
  (time_in_minutes = (2 / 3) * 60) →
  (bail_rate = sink_threshold / (time_in_minutes) - (rain_rate + leak_rate)) →
  bail_rate ≥ 18.5 :=
by
  intros h_distance h_leak_rate h_rain_rate h_sink_threshold h_rowing_speed h_time_in_minutes h_bail_rate
  sorry

end minimum_bail_rate_l185_18581


namespace no_2014_ambiguous_integer_exists_l185_18579

theorem no_2014_ambiguous_integer_exists :
  ∀ k : ℕ, (∃ m : ℤ, k^2 - 8056 = m^2) → (∃ n : ℤ, k^2 + 8056 = n^2) → false :=
by
  -- Proof is omitted as per the instructions
  sorry

end no_2014_ambiguous_integer_exists_l185_18579


namespace find_greater_number_l185_18507

theorem find_greater_number (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 6) (h3 : x * y = 216) (h4 : x > y) : x = 18 := 
sorry

end find_greater_number_l185_18507


namespace calculate_f_at_5_l185_18553

noncomputable def g (y : ℝ) : ℝ := (1 / 2) * y^2

noncomputable def f (x y : ℝ) : ℝ := 2 * x^2 + g y

theorem calculate_f_at_5 (y : ℝ) (h1 : f 2 y = 50) (h2 : y = 2*Real.sqrt 21) :
  f 5 y = 92 :=
by
  sorry

end calculate_f_at_5_l185_18553


namespace domain_of_v_l185_18539

def domain_v (x : ℝ) : Prop :=
  x ≥ 2 ∧ x ≠ 5

theorem domain_of_v :
  {x : ℝ | domain_v x} = { x | 2 < x ∧ x < 5 } ∪ { x | 5 < x }
:= by
  sorry

end domain_of_v_l185_18539


namespace train_to_platform_ratio_l185_18554

-- Define the given conditions as assumptions
def speed_kmh : ℕ := 54 -- speed of the train in km/hr
def train_length_m : ℕ := 450 -- length of the train in meters
def crossing_time_min : ℕ := 1 -- time to cross the platform in minutes

-- Conversion from km/hr to m/min
def speed_mpm : ℕ := (speed_kmh * 1000) / 60

-- Calculate the total distance covered in one minute
def total_distance_m : ℕ := speed_mpm * crossing_time_min

-- Define the length of the platform
def platform_length_m : ℕ := total_distance_m - train_length_m

-- The proof statement to show the ratio of the lengths
theorem train_to_platform_ratio : train_length_m = platform_length_m :=
by 
  -- following from the definition of platform_length_m
  sorry

end train_to_platform_ratio_l185_18554


namespace father_l185_18541

-- Definitions based on conditions in a)
def cost_MP3_player : ℕ := 120
def cost_CD : ℕ := 19
def total_cost : ℕ := cost_MP3_player + cost_CD
def savings : ℕ := 55
def amount_lacking : ℕ := 64

-- Statement of the proof problem
theorem father's_contribution : (savings + (148:ℕ) - amount_lacking = total_cost) := by
  -- Add sorry to skip the proof
  sorry

end father_l185_18541


namespace find_a_l185_18585

theorem find_a {a : ℝ} :
  (∀ x : ℝ, (ax - 1) / (x + 1) < 0 → (x < -1 ∨ x > -1 / 2)) → a = -2 :=
by 
  intros h
  sorry

end find_a_l185_18585


namespace largest_band_members_l185_18559

theorem largest_band_members :
  ∃ (r x : ℕ), r * x + 3 = 107 ∧ (r - 3) * (x + 2) = 107 ∧ r * x < 147 :=
sorry

end largest_band_members_l185_18559


namespace minimum_pizzas_needed_l185_18525

variables (p : ℕ)

def income_per_pizza : ℕ := 12
def gas_cost_per_pizza : ℕ := 4
def maintenance_cost_per_pizza : ℕ := 1
def car_cost : ℕ := 6500

theorem minimum_pizzas_needed :
  p ≥ 929 ↔ (income_per_pizza * p - (gas_cost_per_pizza + maintenance_cost_per_pizza) * p) ≥ car_cost :=
sorry

end minimum_pizzas_needed_l185_18525


namespace polynomial_quotient_l185_18595

theorem polynomial_quotient : 
  (12 * x^3 + 20 * x^2 - 7 * x + 4) / (3 * x + 4) = 4 * x^2 + (4/3) * x - 37/9 :=
by
  sorry

end polynomial_quotient_l185_18595


namespace total_theme_parks_l185_18568

-- Define the constants based on the problem's conditions
def Jamestown := 20
def Venice := Jamestown + 25
def MarinaDelRay := Jamestown + 50

-- Theorem statement: Total number of theme parks in all three towns is 135
theorem total_theme_parks : Jamestown + Venice + MarinaDelRay = 135 := by
  sorry

end total_theme_parks_l185_18568


namespace find_sum_of_numbers_l185_18584

variables (a b c : ℕ) (h_ratio : a * 7 = b * 5 ∧ b * 9 = c * 7) (h_lcm : Nat.lcm a (Nat.lcm b c) = 6300)

theorem find_sum_of_numbers (h_ratio : a * 7 = b * 5 ∧ b * 9 = c * 7) (h_lcm : Nat.lcm a (Nat.lcm b c) = 6300) :
  a + b + c = 14700 :=
sorry

end find_sum_of_numbers_l185_18584


namespace find_n_l185_18580

noncomputable def b_0 : ℝ := Real.cos (Real.pi / 18) ^ 2

noncomputable def b_n (n : ℕ) : ℝ :=
if n = 0 then b_0 else 4 * (b_n (n - 1)) * (1 - (b_n (n - 1)))

theorem find_n : ∀ n : ℕ, b_n n = b_0 → n = 24 := 
sorry

end find_n_l185_18580


namespace intersection_A_B_l185_18560

def A : Set ℤ := {x | x^2 - 3 * x - 4 < 0}
def B : Set ℤ := {-2, -1, 0, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 2, 3} :=
by sorry

end intersection_A_B_l185_18560


namespace solution_set_empty_l185_18532

theorem solution_set_empty (x : ℝ) : ¬ (|x| + |2023 - x| < 2023) :=
by
  sorry

end solution_set_empty_l185_18532


namespace total_cost_of_groceries_l185_18572

noncomputable def M (R : ℝ) : ℝ := 24 * R / 10
noncomputable def F : ℝ := 22

theorem total_cost_of_groceries (R : ℝ) (hR : 2 * R = 22) :
  10 * M R = 24 * R ∧ F = 2 * R ∧ F = 22 →
  4 * M R + 3 * R + 5 * F = 248.6 := by
  sorry

end total_cost_of_groceries_l185_18572


namespace find_b_l185_18534

variable {a b d m : ℝ}

theorem find_b (h : m = d * a * b / (a + b)) : b = m * a / (d * a - m) :=
sorry

end find_b_l185_18534


namespace smallest_prime_12_less_than_square_l185_18530

theorem smallest_prime_12_less_than_square : 
  ∃ n : ℕ, (n^2 - 12 = 13) ∧ Prime (n^2 - 12) ∧ 
  ∀ m : ℕ, (Prime (m^2 - 12) → m^2 - 12 >= 13) :=
sorry

end smallest_prime_12_less_than_square_l185_18530


namespace farmer_pays_per_acre_per_month_l185_18527

-- Define the conditions
def total_payment : ℕ := 600
def length_of_plot : ℕ := 360
def width_of_plot : ℕ := 1210
def square_feet_per_acre : ℕ := 43560

-- Define the problem to prove
theorem farmer_pays_per_acre_per_month :
  length_of_plot * width_of_plot / square_feet_per_acre > 0 ∧
  total_payment / (length_of_plot * width_of_plot / square_feet_per_acre) = 60 :=
by
  -- skipping the actual proof for now
  sorry

end farmer_pays_per_acre_per_month_l185_18527


namespace solve_equation_l185_18510

noncomputable def cube_root (x : ℝ) := x^(1 / 3)

theorem solve_equation (x : ℝ) :
  cube_root x = 15 / (8 - cube_root x) →
  x = 27 ∨ x = 125 :=
by
  sorry

end solve_equation_l185_18510


namespace sum_of_first_3n_terms_l185_18533

theorem sum_of_first_3n_terms (n : ℕ) (sn s2n s3n : ℕ) 
  (h1 : sn = 48) (h2 : s2n = 60)
  (h3 : s2n - sn = s3n - s2n) (h4 : 2 * (s2n - sn) = sn + (s3n - s2n)) :
  s3n = 36 := 
by {
  sorry
}

end sum_of_first_3n_terms_l185_18533


namespace maximum_rectangle_area_l185_18523

theorem maximum_rectangle_area (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 44) : 
  ∃ (l_max w_max : ℕ), l_max * w_max = 121 :=
by
  sorry

end maximum_rectangle_area_l185_18523


namespace rahul_and_sham_together_complete_task_in_35_days_l185_18573

noncomputable def rahul_rate (W : ℝ) : ℝ := W / 60
noncomputable def sham_rate (W : ℝ) : ℝ := W / 84
noncomputable def combined_rate (W : ℝ) := rahul_rate W + sham_rate W

theorem rahul_and_sham_together_complete_task_in_35_days (W : ℝ) :
  (W / combined_rate W) = 35 :=
by
  sorry

end rahul_and_sham_together_complete_task_in_35_days_l185_18573


namespace quadratic_inequality_solution_l185_18548

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, x^2 - (m - 4) * x - m + 7 > 0) ↔ m ∈ Set.Ioo (-2 : ℝ) 6 :=
by
  sorry

end quadratic_inequality_solution_l185_18548


namespace total_votes_l185_18546

-- Define the given conditions
def candidate_votes (V : ℝ) : ℝ := 0.35 * V
def rival_votes (V : ℝ) : ℝ := 0.35 * V + 1800

-- Prove the total number of votes cast
theorem total_votes (V : ℝ) (h : candidate_votes V + rival_votes V = V) : V = 6000 :=
by
  sorry

end total_votes_l185_18546


namespace equal_amounts_hot_and_cold_water_l185_18597

theorem equal_amounts_hot_and_cold_water (time_to_fill_cold : ℕ) (time_to_fill_hot : ℕ) (t_c : ℤ) : 
  time_to_fill_cold = 19 → 
  time_to_fill_hot = 23 → 
  t_c = 2 :=
by
  intros h_c h_h
  sorry

end equal_amounts_hot_and_cold_water_l185_18597


namespace preimages_of_f_l185_18537

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem preimages_of_f (k : ℝ) : (∃ x₁ x₂ : ℝ, f x₁ = k ∧ f x₂ = k ∧ x₁ ≠ x₂) ↔ k < 1 := by
  sorry

end preimages_of_f_l185_18537


namespace no_three_digit_numbers_divisible_by_30_l185_18599

def digits_greater_than_6 (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d > 6

theorem no_three_digit_numbers_divisible_by_30 :
  ∀ n, (100 ≤ n ∧ n < 1000 ∧ digits_greater_than_6 n ∧ n % 30 = 0) → false :=
by
  sorry

end no_three_digit_numbers_divisible_by_30_l185_18599


namespace quadratic_has_negative_root_iff_l185_18588

theorem quadratic_has_negative_root_iff (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 4 * x + 1 = 0) ↔ a ≤ 4 :=
by
  sorry

end quadratic_has_negative_root_iff_l185_18588


namespace bruce_anne_clean_in_4_hours_l185_18512

variable (B : ℝ) -- time it takes for Bruce to clean the house alone
variable (anne_rate := 1 / 12) -- Anne's rate of cleaning the house
variable (double_anne_rate := 1 / 6) -- Anne's rate if her speed is doubled
variable (combined_rate_when_doubled := 1 / 3) -- Combined rate if Anne's speed is doubled

-- Condition: Combined rate of Bruce and doubled Anne is 1/3 house per hour
axiom condition1 : (1 / B + double_anne_rate = combined_rate_when_doubled)

-- Prove that it takes Bruce and Anne together 4 hours to clean the house at their current rates
theorem bruce_anne_clean_in_4_hours (B : ℝ) (h1 : anne_rate = 1/12) (h2 : (1 / B + double_anne_rate = combined_rate_when_doubled)) :
  (1 / (1 / B + anne_rate) = 4) :=
by
  sorry

end bruce_anne_clean_in_4_hours_l185_18512


namespace curve_intersection_l185_18509

theorem curve_intersection (a m : ℝ) (a_pos : 0 < a) :
  (∀ x y : ℝ, 
     (x^2 / a^2 + y^2 = 1) ∧ (y^2 = 2 * (x + m)) 
     → 
     (1 / 2 * (a^2 + 1) = m) ∨ (-a < m ∧ m <= a))
  ∨ (a >= 1 → -a < m ∧ m < a) := 
sorry

end curve_intersection_l185_18509


namespace find_chosen_number_l185_18514

-- Define the conditions
def condition (x : ℝ) : Prop := (3 / 2) * x + 53.4 = -78.9

-- State the theorem
theorem find_chosen_number : ∃ x : ℝ, condition x ∧ x = -88.2 :=
sorry

end find_chosen_number_l185_18514


namespace hike_length_l185_18570

-- Definitions of conditions
def initial_water : ℕ := 6
def final_water : ℕ := 1
def hike_duration : ℕ := 2
def leak_rate : ℕ := 1
def last_mile_drunk : ℕ := 1
def first_part_drink_rate : ℚ := 2 / 3

-- Statement to prove
theorem hike_length (hike_duration : ℕ) (initial_water : ℕ) (final_water : ℕ) (leak_rate : ℕ) 
  (last_mile_drunk : ℕ) (first_part_drink_rate : ℚ) : 
  hike_duration = 2 → 
  initial_water = 6 → 
  final_water = 1 → 
  leak_rate = 1 → 
  last_mile_drunk = 1 → 
  first_part_drink_rate = 2 / 3 → 
  ∃ miles : ℕ, miles = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Proof placeholder
  sorry

end hike_length_l185_18570


namespace xyz_value_l185_18577

theorem xyz_value (x y z : ℝ) (h1 : 2 * x + 3 * y + z = 13) 
                              (h2 : 4 * x^2 + 9 * y^2 + z^2 - 2 * x + 15 * y + 3 * z = 82) : 
  x * y * z = 12 := 
by 
  sorry

end xyz_value_l185_18577


namespace direct_proportion_l185_18587

theorem direct_proportion : 
  ∃ k, (∀ x, y = k * x) ↔ (y = -2 * x) :=
by
  sorry

end direct_proportion_l185_18587


namespace simplify_and_evaluate_expression_l185_18549

theorem simplify_and_evaluate_expression
    (a b : ℤ)
    (h1 : a = -1/3)
    (h2 : b = -2) :
  ((3 * a + b)^2 - (3 * a + b) * (3 * a - b)) / (2 * b) = -3 :=
by
  sorry

end simplify_and_evaluate_expression_l185_18549


namespace no_integer_n_gt_1_satisfies_inequality_l185_18513

open Int

theorem no_integer_n_gt_1_satisfies_inequality :
  ∀ (n : ℤ), n > 1 → ¬ (⌊(Real.sqrt (↑n - 2) + 2 * Real.sqrt (↑n + 2))⌋ < ⌊Real.sqrt (9 * (↑n : ℝ) + 6)⌋) :=
by
  intros n hn
  sorry

end no_integer_n_gt_1_satisfies_inequality_l185_18513


namespace parabola_shift_right_by_3_l185_18519

theorem parabola_shift_right_by_3 :
  ∀ (x : ℝ), (∃ y₁ y₂ : ℝ, y₁ = 2 * x^2 ∧ y₂ = 2 * (x - 3)^2) →
  (∃ (h : ℝ), h = 3) :=
sorry

end parabola_shift_right_by_3_l185_18519


namespace bobby_total_candy_l185_18531

theorem bobby_total_candy (candy1 candy2 : ℕ) (h1 : candy1 = 26) (h2 : candy2 = 17) : candy1 + candy2 = 43 := 
by 
  sorry

end bobby_total_candy_l185_18531


namespace determinant_matrices_equivalence_l185_18528

-- Define the problem as a Lean theorem statement
theorem determinant_matrices_equivalence (p q r s : ℝ) 
  (h : p * s - q * r = 3) : 
  p * (5 * r + 4 * s) - r * (5 * p + 4 * q) = 12 := 
by 
  sorry

end determinant_matrices_equivalence_l185_18528


namespace Nigel_initial_amount_l185_18574

-- Defining the initial amount Olivia has
def Olivia_initial : ℕ := 112

-- Defining the amount left after buying the tickets
def amount_left : ℕ := 83

-- Defining the cost per ticket and the number of tickets bought
def cost_per_ticket : ℕ := 28
def number_of_tickets : ℕ := 6

-- Calculating the total cost of the tickets
def total_cost : ℕ := cost_per_ticket * number_of_tickets

-- Calculating the total amount Olivia spent
def Olivia_spent : ℕ := Olivia_initial - amount_left

-- Defining the total amount they spent
def total_spent : ℕ := total_cost

-- Main theorem to prove that Nigel initially had $139
theorem Nigel_initial_amount : ∃ (n : ℕ), (n + Olivia_initial - Olivia_spent = total_spent) → n = 139 :=
by {
  sorry
}

end Nigel_initial_amount_l185_18574


namespace freddy_age_l185_18520

theorem freddy_age
  (mat_age : ℕ)  -- Matthew's age
  (reb_age : ℕ)  -- Rebecca's age
  (fre_age : ℕ)  -- Freddy's age
  (h1 : mat_age = reb_age + 2)
  (h2 : fre_age = mat_age + 4)
  (h3 : mat_age + reb_age + fre_age = 35) :
  fre_age = 15 :=
by sorry

end freddy_age_l185_18520


namespace notebooks_if_students_halved_l185_18593

-- Definitions based on the problem conditions
def totalNotebooks: ℕ := 512
def notebooksPerStudent (students: ℕ) : ℕ := students / 8
def notebooksWhenStudentsHalved (students notebooks: ℕ) : ℕ := notebooks / (students / 2)

-- Theorem statement
theorem notebooks_if_students_halved (S : ℕ) (h : S * (S / 8) = totalNotebooks) :
    notebooksWhenStudentsHalved S totalNotebooks = 16 :=
by
  sorry

end notebooks_if_students_halved_l185_18593


namespace math_problem_l185_18563

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def g' : ℝ → ℝ := sorry

def condition1 (x : ℝ) : Prop := f (x + 3) = g (-x) + 4
def condition2 (x : ℝ) : Prop := f' x + g' (1 + x) = 0
def even_function (x : ℝ) : Prop := g (2 * x + 1) = g (- (2 * x + 1))

theorem math_problem (x : ℝ) :
  (∀ x, condition1 x) →
  (∀ x, condition2 x) →
  (∀ x, even_function x) →
  (g' 1 = 0) ∧
  (∀ x, f (1 - x) = f (x + 3)) ∧
  (∀ x, f' x = f' (-x + 2)) :=
by
  intros
  sorry

end math_problem_l185_18563
