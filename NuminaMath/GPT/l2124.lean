import Mathlib

namespace find_positive_real_number_l2124_212480

theorem find_positive_real_number (x : ℝ) (hx : x = 25 + 2 * Real.sqrt 159) :
  1 / 2 * (3 * x ^ 2 - 1) = (x ^ 2 - 50 * x - 10) * (x ^ 2 + 25 * x + 5) :=
by
  sorry

end find_positive_real_number_l2124_212480


namespace find_f_of_functions_l2124_212450

theorem find_f_of_functions
  (f g : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = - f x)
  (h_even : ∀ x, g (-x) = g x)
  (h_eq : ∀ x, f x + g x = x^3 - x^2 + x - 3) :
  ∀ x, f x = x^3 + x := 
sorry

end find_f_of_functions_l2124_212450


namespace prove_AP_BP_CP_product_l2124_212496

open Classical

-- Defines that the point P is inside the acute-angled triangle ABC
variables {A B C P: Type} [MetricSpace P] 
variables (PA1 PB1 PC1 AP BP CP : ℝ)

-- Conditions
def conditions (H₁ : PA1 = 3) (H₂ : PB1 = 3) (H₃ : PC1 = 3) (H₄ : AP + BP + CP = 43) : Prop :=
  PA1 = 3 ∧ PB1 = 3 ∧ PC1 = 3 ∧ AP + BP + CP = 43

-- Proof goal
theorem prove_AP_BP_CP_product (H₁ : PA1 = 3) (H₂ : PB1 = 3) (H₃ : PC1 = 3) (H₄ : AP + BP + CP = 43) :
  AP * BP * CP = 441 :=
by {
  -- Proof steps will be filled here
  sorry
}

end prove_AP_BP_CP_product_l2124_212496


namespace product_of_consecutive_sums_not_eq_111111111_l2124_212435

theorem product_of_consecutive_sums_not_eq_111111111 :
  ∀ (a : ℤ), (3 * a + 3) * (3 * a + 12) ≠ 111111111 := 
by
  intros a
  sorry

end product_of_consecutive_sums_not_eq_111111111_l2124_212435


namespace evaluate_fraction_l2124_212418

theorem evaluate_fraction : (5 / 6 : ℚ) / (9 / 10) - 1 = -2 / 27 := by
  sorry

end evaluate_fraction_l2124_212418


namespace dihedral_angle_proof_l2124_212414

noncomputable def angle_between_planes 
  (α β : Real) : Real :=
  Real.arcsin (Real.sin α * Real.sin β)

theorem dihedral_angle_proof 
  (α β : Real) 
  (α_non_neg : 0 ≤ α) 
  (α_non_gtr : α ≤ Real.pi / 2) 
  (β_non_neg : 0 ≤ β) 
  (β_non_gtr : β ≤ Real.pi / 2) :
  angle_between_planes α β = Real.arcsin (Real.sin α * Real.sin β) :=
by
  sorry

end dihedral_angle_proof_l2124_212414


namespace inequality_proof_l2124_212449

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_cond : a + b + c + 3 * a * b * c ≥ (a * b)^2 + (b * c)^2 + (c * a)^2 + 3) :
  (a^3 + b^3 + c^3) / 3 ≥ (a * b * c + 2021) / 2022 :=
by 
  sorry

end inequality_proof_l2124_212449


namespace average_matches_rounded_l2124_212451

def total_matches : ℕ := 6 * 1 + 3 * 2 + 3 * 3 + 2 * 4 + 6 * 5

def total_players : ℕ := 6 + 3 + 3 + 2 + 6

noncomputable def average_matches : ℚ := total_matches / total_players

theorem average_matches_rounded : Int.floor (average_matches + 0.5) = 3 :=
by
  unfold average_matches total_matches total_players
  norm_num
  sorry

end average_matches_rounded_l2124_212451


namespace new_customers_needed_l2124_212462

theorem new_customers_needed 
  (initial_customers : ℕ)
  (customers_after_some_left : ℕ)
  (first_group_left : ℕ)
  (second_group_left : ℕ)
  (new_customers : ℕ)
  (h1 : initial_customers = 13)
  (h2 : customers_after_some_left = 9)
  (h3 : first_group_left = initial_customers - customers_after_some_left)
  (h4 : second_group_left = 8)
  (h5 : new_customers = first_group_left + second_group_left) :
  new_customers = 12 :=
by
  sorry

end new_customers_needed_l2124_212462


namespace hyperbola_trajectory_center_l2124_212436

theorem hyperbola_trajectory_center :
  ∀ m : ℝ, ∃ (x y : ℝ), x^2 - y^2 - 6 * m * x - 4 * m * y + 5 * m^2 - 1 = 0 ∧ 2 * x + 3 * y = 0 :=
by
  sorry

end hyperbola_trajectory_center_l2124_212436


namespace inequalities_hold_l2124_212407

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 2) :
  a * b ≤ 1 ∧ a^2 + b^2 ≥ 2 ∧ (1 / a + 1 / b ≥ 2) :=
by
  sorry

end inequalities_hold_l2124_212407


namespace simplify_exponent_l2124_212447

theorem simplify_exponent (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
by
  sorry

end simplify_exponent_l2124_212447


namespace mean_of_five_numbers_l2124_212494

theorem mean_of_five_numbers (sum : ℚ) (h : sum = 3 / 4) : (sum / 5 = 3 / 20) :=
by
  -- Proof omitted
  sorry

end mean_of_five_numbers_l2124_212494


namespace remainder_hx10_div_hx_l2124_212433

noncomputable def h (x : ℕ) := x^6 - x^5 + x^4 - x^3 + x^2 - x + 1

theorem remainder_hx10_div_hx (x : ℕ) : (h x ^ 10) % (h x) = 7 := by
  sorry

end remainder_hx10_div_hx_l2124_212433


namespace pablo_distributed_fraction_l2124_212488

-- Definitions based on the problem statement
def mia_coins (m : ℕ) := m
def sofia_coins (m : ℕ) := 3 * m
def pablo_coins (m : ℕ) := 12 * m

-- Condition for equal distribution
def target_coins (m : ℕ) := (mia_coins m + sofia_coins m + pablo_coins m) / 3

-- Needs for redistribution
def sofia_needs (m : ℕ) := target_coins m - sofia_coins m
def mia_needs (m : ℕ) := target_coins m - mia_coins m

-- Total distributed coins by Pablo
def total_distributed_by_pablo (m : ℕ) := sofia_needs m + mia_needs m

-- Fraction of coins Pablo distributes
noncomputable def fraction_distributed_by_pablo (m : ℕ) := (total_distributed_by_pablo m) / (pablo_coins m)

-- Theorem to prove
theorem pablo_distributed_fraction (m : ℕ) : fraction_distributed_by_pablo m = 5 / 9 := by
  sorry

end pablo_distributed_fraction_l2124_212488


namespace jake_snakes_l2124_212493

theorem jake_snakes (S : ℕ) 
  (h1 : 2 * S + 1 = 6) 
  (h2 : 2250 = 5 * 250 + 1000) :
  S = 3 := 
by
  sorry

end jake_snakes_l2124_212493


namespace expand_expression_l2124_212439

theorem expand_expression (x : ℝ) : (x + 3) * (4 * x - 8) - 2 * x = 4 * x^2 + 2 * x - 24 := by
  sorry

end expand_expression_l2124_212439


namespace part1_part2_part3_l2124_212416

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x + a / x + Real.log x

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  1 - a / x^2 + 1 / x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  f' x a - x

theorem part1 (a : ℝ) (h : f' 1 a = 0) : a = 2 :=
  sorry

theorem part2 {a : ℝ} (h : ∀ x, 1 < x → x < 2 → f' x a ≥ 0) : a ≤ 2 :=
  sorry

theorem part3 (a : ℝ) :
  ((a > 1 → ∀ x, g x a ≠ 0) ∧ 
  (a = 1 ∨ a ≤ 0 → ∃ x, g x a = 0 ∧ ∀ y, g y a = 0 → y = x) ∧ 
  (0 < a ∧ a < 1 → ∃ x y, x ≠ y ∧ g x a = 0 ∧ g y a = 0)) :=
  sorry

end part1_part2_part3_l2124_212416


namespace at_least_one_nonzero_l2124_212421

theorem at_least_one_nonzero (a b : ℝ) : a^2 + b^2 ≠ 0 ↔ (a ≠ 0 ∨ b ≠ 0) := by
  sorry

end at_least_one_nonzero_l2124_212421


namespace min_f_over_f_prime_at_1_l2124_212473

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
noncomputable def quadratic_derivative (a b x : ℝ) : ℝ := 2 * a * x + b

theorem min_f_over_f_prime_at_1 (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b > 0) (h₂ : ∀ x, quadratic_function a b c x ≥ 0) :
  (∃ k, (∀ x, quadratic_function a b c x ≥ 0 → quadratic_function a b c ((-b)/(2*a)) ≤ x) ∧ k = 2) :=
by
  sorry

end min_f_over_f_prime_at_1_l2124_212473


namespace final_value_l2124_212492

variable (p q r : ℝ)

-- Conditions
axiom h1 : p + q + r = 5
axiom h2 : 1 / (p + q) + 1 / (q + r) + 1 / (r + p) = 9

-- Goal
theorem final_value : (r / (p + q)) + (p / (q + r)) + (q / (r + p)) = 42 :=
by 
  sorry

end final_value_l2124_212492


namespace best_value_l2124_212457

variables {cS qS cM qL cL : ℝ}
variables (medium_cost : cM = 1.4 * cS) (medium_quantity : qM = 0.7 * qL)
variables (large_quantity : qL = 1.5 * qS) (large_cost : cL = 1.2 * cM)

theorem best_value :
  let small_value := cS / qS
  let medium_value := cM / (0.7 * qL)
  let large_value := cL / qL
  small_value < large_value ∧ large_value < medium_value :=
sorry

end best_value_l2124_212457


namespace calculation_l2124_212417

theorem calculation :
  7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 :=
by
  sorry

end calculation_l2124_212417


namespace root_equation_alpha_beta_property_l2124_212456

theorem root_equation_alpha_beta_property {α β : ℝ} (h1 : α^2 + α - 1 = 0) (h2 : β^2 + β - 1 = 0) :
    α^2 + 2 * β^2 + β = 4 :=
by
  sorry

end root_equation_alpha_beta_property_l2124_212456


namespace find_points_l2124_212437

def acute_triangle (A B C : ℝ × ℝ × ℝ) : Prop :=
  -- Definition to ensure that the triangle formed by A, B, and C is an acute-angled triangle.
  sorry -- This would be formalized ensuring all angles are less than 90 degrees.

def no_three_collinear (A B C D E : ℝ × ℝ × ℝ) : Prop :=
  -- Definition that ensures no three points among A, B, C, D, and E are collinear.
  sorry

def line_normal_to_plane (P Q R S : ℝ × ℝ × ℝ) : Prop :=
  -- Definition to ensure that the line through any two points P, Q is normal to the plane containing R, S, and the other point.
  sorry

theorem find_points (A B C : ℝ × ℝ × ℝ) (h_acute : acute_triangle A B C) :
  ∃ (D E : ℝ × ℝ × ℝ), no_three_collinear A B C D E ∧
    (∀ (P Q R R' : ℝ × ℝ × ℝ), 
      (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E) →
      (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = E) →
      (R = A ∨ R = B ∨ R = C ∨ R = D ∨ R = E) →
      (R' = A ∨ R' = B ∨ R' = C ∨ R' = D ∨ R' = E) →
      P ≠ Q → Q ≠ R → R ≠ R' →
      line_normal_to_plane P Q R R') :=
sorry

end find_points_l2124_212437


namespace time_to_cross_trains_l2124_212484

/-- Length of the first train in meters -/
def length_train1 : ℕ := 50

/-- Length of the second train in meters -/
def length_train2 : ℕ := 120

/-- Speed of the first train in km/hr -/
def speed_train1_kmh : ℕ := 60

/-- Speed of the second train in km/hr -/
def speed_train2_kmh : ℕ := 40

/-- Relative speed in km/hr as trains are moving in opposite directions -/
def relative_speed_kmh : ℕ := speed_train1_kmh + speed_train2_kmh

/-- Convert speed from km/hr to m/s -/
def kmh_to_ms (speed_kmh : ℕ) : ℚ := (speed_kmh * 1000) / 3600

/-- Relative speed in m/s -/
def relative_speed_ms : ℚ := kmh_to_ms relative_speed_kmh

/-- Total distance to be covered in meters -/
def total_distance : ℕ := length_train1 + length_train2

/-- Time taken in seconds to cross each other -/
def time_to_cross : ℚ := total_distance / relative_speed_ms

theorem time_to_cross_trains :
  time_to_cross = 6.12 := 
sorry

end time_to_cross_trains_l2124_212484


namespace calculate_womans_haircut_cost_l2124_212487

-- Define the necessary constants and conditions
def W : ℝ := sorry
def child_haircut_cost : ℝ := 36
def tip_percentage : ℝ := 0.20
def total_tip : ℝ := 24
def number_of_children : ℕ := 2

-- Helper function to calculate total cost before the tip
def total_cost_before_tip (W : ℝ) (number_of_children : ℕ) (child_haircut_cost : ℝ) : ℝ :=
  W + number_of_children * child_haircut_cost

-- Lean statement for the main theorem
theorem calculate_womans_haircut_cost (W : ℝ) (child_haircut_cost : ℝ) (tip_percentage : ℝ)
  (total_tip : ℝ) (number_of_children : ℕ) :
  (tip_percentage * total_cost_before_tip W number_of_children child_haircut_cost) = total_tip →
  W = 48 :=
by
  sorry

end calculate_womans_haircut_cost_l2124_212487


namespace floor_diff_bounds_l2124_212432

theorem floor_diff_bounds (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  0 ≤ Int.floor (a + b) - (Int.floor a + Int.floor b) ∧ 
  Int.floor (a + b) - (Int.floor a + Int.floor b) ≤ 1 :=
by
  sorry

end floor_diff_bounds_l2124_212432


namespace total_savings_l2124_212475

noncomputable def kimmie_earnings : ℝ := 450
noncomputable def zahra_earnings : ℝ := kimmie_earnings - (1/3) * kimmie_earnings
noncomputable def kimmie_savings : ℝ := (1/2) * kimmie_earnings
noncomputable def zahra_savings : ℝ := (1/2) * zahra_earnings

theorem total_savings : kimmie_savings + zahra_savings = 375 :=
by
  -- Conditions based definitions preclude need for this proof
  sorry

end total_savings_l2124_212475


namespace find_number_l2124_212422

theorem find_number (x : ℝ) (h1 : 0.35 * x = 0.2 * 700) (h2 : 0.2 * 700 = 140) (h3 : 0.35 * x = 140) : x = 400 :=
by sorry

end find_number_l2124_212422


namespace number_of_numbers_l2124_212440

theorem number_of_numbers 
  (avg : ℚ) (avg1 : ℚ) (avg2 : ℚ) (avg3 : ℚ)
  (h_avg : avg = 4.60) 
  (h_avg1 : avg1 = 3.4) 
  (h_avg2 : avg2 = 3.8) 
  (h_avg3 : avg3 = 6.6) 
  (h_sum_eq : 2 * avg1 + 2 * avg2 + 2 * avg3 = 27.6) : 
  (27.6 / avg = 6) := 
  by sorry

end number_of_numbers_l2124_212440


namespace ratio_of_areas_of_shaded_and_white_region_l2124_212483

theorem ratio_of_areas_of_shaded_and_white_region
  (all_squares_have_vertices_in_middle: ∀ (n : ℕ), n ≠ 0 → (square_vertices_positioned_mid : Prop)) :
  ∃ (ratio : ℚ), ratio = 5 / 3 :=
by
  sorry

end ratio_of_areas_of_shaded_and_white_region_l2124_212483


namespace smallest_a_l2124_212427

-- Define the conditions and the proof goal
theorem smallest_a (a b : ℝ) (h₁ : ∀ x : ℤ, Real.sin (a * (x : ℝ) + b) = Real.sin (15 * (x : ℝ))) (h₂ : 0 ≤ a) (h₃ : 0 ≤ b) :
  a = 15 :=
sorry

end smallest_a_l2124_212427


namespace perpendicular_line_through_circle_center_l2124_212495

theorem perpendicular_line_through_circle_center :
  ∀ (x y : ℝ), (x^2 + (y-1)^2 = 4) → (3*x + 2*y + 1 = 0) → (2*x - 3*y + 3 = 0) :=
by
  intros x y h_circle h_line
  sorry

end perpendicular_line_through_circle_center_l2124_212495


namespace transform_to_A_plus_one_l2124_212479

theorem transform_to_A_plus_one (A : ℕ) (hA : A > 0) : 
  ∃ n : ℕ, (∀ i : ℕ, (i ≤ n) → ((A + 9 * i) = A + 1 ∨ ∃ j : ℕ, (A + 9 * i) = (A + 1 + 10 * j))) :=
sorry

end transform_to_A_plus_one_l2124_212479


namespace mixture_problem_l2124_212459

theorem mixture_problem
  (x : ℝ)
  (c1 c2 c_final : ℝ)
  (v1 v2 v_final : ℝ)
  (h1 : c1 = 0.60)
  (h2 : c2 = 0.75)
  (h3 : c_final = 0.72)
  (h4 : v1 = 4)
  (h5 : x = 16)
  (h6 : v2 = x)
  (h7 : v_final = v1 + v2) :
  v_final = 20 ∧ c_final * v_final = c1 * v1 + c2 * v2 :=
by
  sorry

end mixture_problem_l2124_212459


namespace max_kings_l2124_212499

theorem max_kings (initial_kings : ℕ) (kings_attacking_each_other : initial_kings = 21) 
  (no_two_kings_attack : ∀ kings_remaining, kings_remaining ≤ 16) : 
  ∃ kings_remaining, kings_remaining = 16 :=
by
  sorry

end max_kings_l2124_212499


namespace horse_food_per_day_l2124_212423

theorem horse_food_per_day (ratio_sh : ℕ) (ratio_h : ℕ) (sheep : ℕ) (total_food : ℕ) (sheep_count : sheep = 32) (ratio : ratio_sh = 4) (ratio_horses : ratio_h = 7) (total_food_need : total_food = 12880) :
  total_food / (sheep * ratio_h / ratio_sh) = 230 :=
by
  sorry

end horse_food_per_day_l2124_212423


namespace cookie_cost_l2124_212467

variables (m o c : ℝ)
variables (H1 : m = 2 * o)
variables (H2 : 3 * (3 * m + 5 * o) = 5 * m + 10 * o + 4 * c)

theorem cookie_cost (H1 : m = 2 * o) (H2 : 3 * (3 * m + 5 * o) = 5 * m + 10 * o + 4 * c) : c = (13 / 4) * o :=
by sorry

end cookie_cost_l2124_212467


namespace smallest_angle_in_scalene_triangle_l2124_212441

theorem smallest_angle_in_scalene_triangle :
  ∃ (triangle : Type) (a b c : ℝ),
    ∀ (A B C : triangle),
      a = 162 ∧
      b / c = 3 / 4 ∧
      a + b + c = 180 ∧
      a ≠ b ∧ a ≠ c ∧ b ≠ c ->
        min b c = 7.7 :=
sorry

end smallest_angle_in_scalene_triangle_l2124_212441


namespace no_such_x_exists_l2124_212477

theorem no_such_x_exists : ¬ ∃ x : ℝ, 
  (∃ x1 : ℤ, x - 1/x = x1) ∧ 
  (∃ x2 : ℤ, 1/x - 1/(x^2 + 1) = x2) ∧ 
  (∃ x3 : ℤ, 1/(x^2 + 1) - 2*x = x3) :=
by
  sorry

end no_such_x_exists_l2124_212477


namespace algebraic_expression_value_l2124_212425

variables (a b c d m : ℝ)

theorem algebraic_expression_value :
  a = -b → cd = 1 → m^2 = 1 →
  -(a + b) - cd / 2022 + m^2 / 2022 = 0 :=
by
  intros h1 h2 h3
  sorry

end algebraic_expression_value_l2124_212425


namespace advance_tickets_sold_20_l2124_212409

theorem advance_tickets_sold_20 :
  ∃ (A S : ℕ), 20 * A + 30 * S = 1600 ∧ A + S = 60 ∧ A = 20 :=
by
  sorry

end advance_tickets_sold_20_l2124_212409


namespace find_c_l2124_212404

theorem find_c (x : ℝ) (c : ℝ) (h : x = 0.3)
  (equ : (10 * x + 2) / c - (3 * x - 6) / 18 = (2 * x + 4) / 3) :
  c = 4 :=
by
  sorry

end find_c_l2124_212404


namespace quadratic_sum_r_s_l2124_212497

/-- Solve the quadratic equation and identify the sum of r and s 
from the equivalent completed square form (x + r)^2 = s. -/
theorem quadratic_sum_r_s (r s : ℤ) :
  (∃ r s : ℤ, (x - r)^2 = s → r + s = 11) :=
sorry

end quadratic_sum_r_s_l2124_212497


namespace range_of_f_when_a_eq_2_sufficient_but_not_necessary_condition_for_q_l2124_212453

-- Define the function
def f (x a : ℝ) : ℝ := x^2 - a * x + 4 - a^2

-- Problem (1): Range of the function when a = 2
theorem range_of_f_when_a_eq_2 :
  (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x 2 = (x - 1)^2 - 1) →
  Set.image (f 2) (Set.Icc (-2 : ℝ) 3) = Set.Icc (-1 : ℝ) 8 := sorry

-- Problem (2): Sufficient but not necessary condition
theorem sufficient_but_not_necessary_condition_for_q :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x 4 ≤ 0) →
  (Set.Icc (-2 : ℝ) 2 → (∃ (M : Set ℝ), Set.singleton 4 ⊆ M ∧ 
    (∀ a ∈ M, ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ 0) ∧
    (∀ a ∈ Set.Icc (-2 : ℝ) 2, ∃ a' ∉ M, ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a' ≤ 0))) := sorry

end range_of_f_when_a_eq_2_sufficient_but_not_necessary_condition_for_q_l2124_212453


namespace min_value_inequality_l2124_212491

theorem min_value_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b + b / c + c / d + d / a) ≥ 4 :=
by
  sorry

end min_value_inequality_l2124_212491


namespace julia_total_balls_l2124_212458

theorem julia_total_balls :
  (3 * 19) + (10 * 19) + (8 * 19) = 399 :=
by
  -- proof goes here
  sorry

end julia_total_balls_l2124_212458


namespace no_real_roots_iff_k_gt_2_l2124_212476

theorem no_real_roots_iff_k_gt_2 (k : ℝ) : 
  (∀ (x : ℝ), x^2 - 2 * x + k - 1 ≠ 0) ↔ k > 2 :=
by 
  sorry

end no_real_roots_iff_k_gt_2_l2124_212476


namespace minimum_k_l2124_212471

variable {a b k : ℝ}

theorem minimum_k (h_a : a > 0) (h_b : b > 0) (h : ∀ a b : ℝ, a > 0 → b > 0 → (1 / a) + (1 / b) + (k / (a + b)) ≥ 0) : k ≥ -4 :=
sorry

end minimum_k_l2124_212471


namespace inequality_min_value_l2124_212454

theorem inequality_min_value (a : ℝ) : 
  (∀ x : ℝ, abs (x - 1) + abs (x + 2) ≥ a) → (a ≤ 3) := 
by
  sorry

end inequality_min_value_l2124_212454


namespace unique_solution_for_quadratic_l2124_212485

theorem unique_solution_for_quadratic (a : ℝ) : 
  ∃! (x : ℝ), x^2 - 2 * a * x + a^2 = 0 := 
by
  sorry

end unique_solution_for_quadratic_l2124_212485


namespace no_right_angle_sequence_l2124_212446

theorem no_right_angle_sequence 
  (A B C : Type)
  (angle_A angle_B angle_C : ℝ)
  (angle_A_eq : angle_A = 59)
  (angle_B_eq : angle_B = 61)
  (angle_C_eq : angle_C = 60)
  (midpoint : A → A → A)
  (A0 B0 C0 : A) :
  ¬ ∃ n : ℕ, ∃ An Bn Cn : A, 
    (An = midpoint Bn Cn) ∧ 
    (Bn = midpoint An Cn) ∧ 
    (Cn = midpoint An Bn) ∧ 
    (angle_A = 90 ∨ angle_B = 90 ∨ angle_C = 90) :=
sorry

end no_right_angle_sequence_l2124_212446


namespace pages_copied_l2124_212489

-- Define the assumptions
def cost_per_pages (cent_per_pages: ℕ) : Prop := 
  5 * cent_per_pages = 7 * 1

def total_cents (dollars: ℕ) (cents: ℕ) : Prop :=
  cents = dollars * 100

-- The problem to prove
theorem pages_copied (dollars: ℕ) (cents: ℕ) (cent_per_pages: ℕ) : 
  cost_per_pages cent_per_pages → total_cents dollars cents → dollars = 35 → cents = 3500 → 
  3500 * (5/7 : ℚ) = 2500 :=
by
  sorry

end pages_copied_l2124_212489


namespace coefficient_x_neg_4_expansion_l2124_212428

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the function to calculate the coefficient of the term containing x^(-4)
def coeff_term_x_neg_4 : ℕ :=
  let k := 10
  binom 12 k

theorem coefficient_x_neg_4_expansion :
  coeff_term_x_neg_4 = 66 := by
  -- Calculation here would show that binom 12 10 is indeed 66
  sorry

end coefficient_x_neg_4_expansion_l2124_212428


namespace range_of_a_l2124_212412

noncomputable def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }
noncomputable def B (a : ℝ) : Set ℝ := { x | x ^ 2 + 2 * x + a ≥ 0 }

theorem range_of_a (a : ℝ) : (a > -8) → (∃ x, x ∈ A ∧ x ∈ B a) :=
by
  sorry

end range_of_a_l2124_212412


namespace arithmetic_sequence_sum_l2124_212400

-- Definitions based on problem conditions
variable (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ) -- terms of the sequence
variable (S_3 S_6 S_9 : ℤ)

-- Given conditions
variable (h1 : S_3 = 3 * a_1 + 3 * (a_2 - a_1))
variable (h2 : S_6 = 6 * a_1 + 15 * (a_2 - a_1))
variable (h3 : S_3 = 9)
variable (h4 : S_6 = 36)

-- Theorem to prove
theorem arithmetic_sequence_sum : S_9 = 81 :=
by
  -- We just state the theorem here and will provide a proof later
  sorry

end arithmetic_sequence_sum_l2124_212400


namespace roots_operation_zero_l2124_212413

def operation (a b : ℝ) : ℝ := a * b - a - b

theorem roots_operation_zero {x1 x2 : ℝ}
  (h1 : x1 + x2 = -1)
  (h2 : x1 * x2 = -1) :
  operation x1 x2 = 0 :=
by
  sorry

end roots_operation_zero_l2124_212413


namespace find_x_value_l2124_212420

variable {x : ℝ}

def opposite_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ a = (k * b.1, k * b.2)

theorem find_x_value (h : opposite_directions (x, 1) (4, x)) : x = -2 :=
sorry

end find_x_value_l2124_212420


namespace total_cars_produced_l2124_212474

theorem total_cars_produced (cars_NA cars_EU : ℕ) (h1 : cars_NA = 3884) (h2 : cars_EU = 2871) : cars_NA + cars_EU = 6755 := by
  sorry

end total_cars_produced_l2124_212474


namespace fraction_eq_l2124_212419

theorem fraction_eq {x : ℝ} (h : 1 - 6 / x + 9 / x ^ 2 - 2 / x ^ 3 = 0) :
  3 / x = 3 / 2 ∨ 3 / x = 3 / (2 + Real.sqrt 3) ∨ 3 / x = 3 / (2 - Real.sqrt 3) :=
sorry

end fraction_eq_l2124_212419


namespace find_f_half_l2124_212405

noncomputable def g (x : ℝ) : ℝ := 1 - 2 * x
noncomputable def f (y : ℝ) : ℝ := if y ≠ 0 then (1 - y^2) / y^2 else 0

theorem find_f_half :
  f (g (1 / 4)) = 15 :=
by
  have g_eq : g (1 / 4) = 1 / 2 := sorry
  rw [g_eq]
  have f_eq : f (1 / 2) = 15 := sorry
  exact f_eq

end find_f_half_l2124_212405


namespace sum_of_three_digit_numbers_l2124_212438

theorem sum_of_three_digit_numbers (a b c : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
  222 * (a + b + c) ≠ 2021 := 
sorry

end sum_of_three_digit_numbers_l2124_212438


namespace carpet_area_l2124_212465

def room_length_ft := 16
def room_width_ft := 12
def column_side_ft := 2
def ft_to_inches := 12

def room_length_in := room_length_ft * ft_to_inches
def room_width_in := room_width_ft * ft_to_inches
def column_side_in := column_side_ft * ft_to_inches

def room_area_in_sq := room_length_in * room_width_in
def column_area_in_sq := column_side_in * column_side_in

def remaining_area_in_sq := room_area_in_sq - column_area_in_sq

theorem carpet_area : remaining_area_in_sq = 27072 := by
  sorry

end carpet_area_l2124_212465


namespace intersection_l1_l2_line_parallel_to_l3_line_perpendicular_to_l3_l2124_212466

def l1 (x y : ℝ) : Prop := x + y = 2
def l2 (x y : ℝ) : Prop := x - 3 * y = -10
def l3 (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

def M : (ℝ × ℝ) := (-1, 3)

-- Part (Ⅰ): Prove that M is the intersection point of l1 and l2
theorem intersection_l1_l2 : l1 M.1 M.2 ∧ l2 M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

-- Part (Ⅱ): Prove the equation of the line passing through M and parallel to l3 is 3x - 4y + 15 = 0
def parallel_line (x y : ℝ) : Prop := 3 * x - 4 * y + 15 = 0

theorem line_parallel_to_l3 : parallel_line M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

-- Part (Ⅲ): Prove the equation of the line passing through M and perpendicular to l3 is 4x + 3y - 5 = 0
def perpendicular_line (x y : ℝ) : Prop := 4 * x + 3 * y - 5 = 0

theorem line_perpendicular_to_l3 : perpendicular_line M.1 M.2 :=
by
  -- Placeholder for the actual proof
  sorry

end intersection_l1_l2_line_parallel_to_l3_line_perpendicular_to_l3_l2124_212466


namespace total_cupcakes_l2124_212452

noncomputable def cupcakesForBonnie : ℕ := 24
noncomputable def cupcakesPerDay : ℕ := 60
noncomputable def days : ℕ := 2

theorem total_cupcakes : (cupcakesPerDay * days + cupcakesForBonnie) = 144 := 
by
  sorry

end total_cupcakes_l2124_212452


namespace radhika_christmas_games_l2124_212424

variable (C B : ℕ)

def games_on_birthday := 8
def total_games (C : ℕ) (B : ℕ) := C + B + (C + B) / 2

theorem radhika_christmas_games : 
  total_games C games_on_birthday = 30 → C = 12 :=
by
  intro h
  sorry

end radhika_christmas_games_l2124_212424


namespace lizard_ratio_l2124_212408

def lizard_problem (W S : ℕ) : Prop :=
  (S = 7 * W) ∧ (3 = S + W - 69) ∧ (W / 3 = 3)

theorem lizard_ratio (W S : ℕ) (h : lizard_problem W S) : W / 3 = 3 :=
  by
    rcases h with ⟨h1, h2, h3⟩
    exact h3

end lizard_ratio_l2124_212408


namespace book_area_correct_l2124_212463

/-- Converts inches to centimeters -/
def inch_to_cm (inches : ℚ) : ℚ :=
  inches * 2.54

/-- The length of the book given a parameter x -/
def book_length (x : ℚ) : ℚ :=
  3 * x - 4

/-- The width of the book in inches -/
def book_width_in_inches : ℚ :=
  5 / 2

/-- The width of the book in centimeters -/
def book_width : ℚ :=
  inch_to_cm book_width_in_inches

/-- The area of the book given a parameter x -/
def book_area (x : ℚ) : ℚ :=
  book_length x * book_width

/-- Proof that the area of the book with x = 5 is 69.85 cm² -/
theorem book_area_correct : book_area 5 = 69.85 := by
  sorry

end book_area_correct_l2124_212463


namespace marly_100_bills_l2124_212402

-- Define the number of each type of bill Marly has
def num_20_bills := 10
def num_10_bills := 8
def num_5_bills := 4

-- Define the values of the bills
def value_20_bill := 20
def value_10_bill := 10
def value_5_bill := 5

-- Define the total amount of money Marly has
def total_amount := num_20_bills * value_20_bill + num_10_bills * value_10_bill + num_5_bills * value_5_bill

-- Define the value of a $100 bill
def value_100_bill := 100

-- Now state the main theorem
theorem marly_100_bills : total_amount / value_100_bill = 3 := by
  sorry

end marly_100_bills_l2124_212402


namespace value_of_k_l2124_212490

theorem value_of_k (k : ℝ) (x : ℝ) (h : (k - 3) * x^2 + 6 * x + k^2 - k = 0) (r : x = -1) : 
  k = -3 := 
by
  sorry

end value_of_k_l2124_212490


namespace probability_AB_selected_l2124_212469

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l2124_212469


namespace multiply_5915581_7907_l2124_212415

theorem multiply_5915581_7907 : 5915581 * 7907 = 46757653387 := 
by
  -- sorry is used here to skip the proof
  sorry

end multiply_5915581_7907_l2124_212415


namespace cafeteria_pies_l2124_212406

theorem cafeteria_pies (initial_apples handed_out_apples apples_per_pie : ℕ)
  (h1 : initial_apples = 96)
  (h2 : handed_out_apples = 42)
  (h3 : apples_per_pie = 6) :
  (initial_apples - handed_out_apples) / apples_per_pie = 9 := by
  sorry

end cafeteria_pies_l2124_212406


namespace numbers_represented_3_units_from_A_l2124_212426

theorem numbers_represented_3_units_from_A (A : ℝ) (x : ℝ) (h : A = -2) : 
  abs (x + 2) = 3 ↔ x = 1 ∨ x = -5 := by
  sorry

end numbers_represented_3_units_from_A_l2124_212426


namespace travel_period_l2124_212443

-- Nina's travel pattern
def travels_in_one_month : ℕ := 400
def travels_in_two_months : ℕ := travels_in_one_month + 2 * travels_in_one_month

-- The total distance Nina wants to travel
def total_distance : ℕ := 14400

-- The period in months during which Nina travels the given total distance 
def required_period_in_months (d_per_2_months : ℕ) (total_d : ℕ) : ℕ := (total_d / d_per_2_months) * 2

-- Statement we need to prove
theorem travel_period : required_period_in_months travels_in_two_months total_distance = 24 := by
  sorry

end travel_period_l2124_212443


namespace ratio_of_sides_l2124_212486

theorem ratio_of_sides (s x y : ℝ) 
    (h1 : 0.1 * s^2 = 0.25 * x * y)
    (h2 : x = s / 10)
    (h3 : y = 4 * s) : x / y = 1 / 40 :=
by
  sorry

end ratio_of_sides_l2124_212486


namespace eq_curveE_eq_lineCD_l2124_212411

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def curveE (x y : ℝ) : Prop :=
  distance (x, y) (-1, 0) = Real.sqrt 3 * distance (x, y) (1, 0)

theorem eq_curveE (x y : ℝ) : curveE x y ↔ (x - 2)^2 + y^2 = 3 :=
by sorry

variables (m : ℝ)
variables (m_nonzero : m ≠ 0)
variables (A C B D : ℝ × ℝ)
variables (line1_intersect : ∀ p : ℝ × ℝ, curveE p.1 p.2 → (p = A ∨ p = C) → p.1 - m * p.2 - 1 = 0)
variables (line2_intersect : ∀ p : ℝ × ℝ, curveE p.1 p.2 → (p = B ∨ p = D) → m * p.1 + p.2 - m = 0)
variables (CD_slope : (D.2 - C.2) / (D.1 - C.1) = -1)

theorem eq_lineCD (x y : ℝ) : 
  (y = -x ∨ y = -x + 3) :=
by sorry

end eq_curveE_eq_lineCD_l2124_212411


namespace necklace_price_l2124_212461

variable (N : ℝ)

def price_of_bracelet : ℝ := 15.00
def price_of_earring : ℝ := 10.00
def num_necklaces_sold : ℝ := 5
def num_bracelets_sold : ℝ := 10
def num_earrings_sold : ℝ := 20
def num_complete_ensembles_sold : ℝ := 2
def price_of_complete_ensemble : ℝ := 45.00
def total_amount_made : ℝ := 565.0

theorem necklace_price :
  5 * N + 10 * price_of_bracelet + 20 * price_of_earring
  + 2 * price_of_complete_ensemble = total_amount_made → N = 25 :=
by
  intro h
  sorry

end necklace_price_l2124_212461


namespace irrational_number_line_representation_l2124_212430

theorem irrational_number_line_representation :
  ∀ (x : ℝ), ¬ (∃ r s : ℚ, x = r / s ∧ r ≠ 0 ∧ s ≠ 0) → ∃ p : ℝ, x = p := 
by
  sorry

end irrational_number_line_representation_l2124_212430


namespace distance_between_nails_l2124_212410

theorem distance_between_nails (banner_length : ℕ) (num_nails : ℕ) (end_distance : ℕ) :
  banner_length = 20 → num_nails = 7 → end_distance = 1 → 
  (banner_length - 2 * end_distance) / (num_nails - 1) = 3 :=
by
  intros
  sorry

end distance_between_nails_l2124_212410


namespace xyz_value_l2124_212431

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) : 
  x * y * z = 10 :=
by
  sorry

end xyz_value_l2124_212431


namespace find_S3_l2124_212468

-- Define the known scores
def S1 : ℕ := 55
def S2 : ℕ := 67
def S4 : ℕ := 55
def Avg : ℕ := 67

-- Statement to prove
theorem find_S3 : ∃ S3 : ℕ, (S1 + S2 + S3 + S4) / 4 = Avg ∧ S3 = 91 :=
by
  sorry

end find_S3_l2124_212468


namespace value_of_b_l2124_212470

theorem value_of_b 
  (a b : ℝ) 
  (h : ∃ c : ℝ, (ax^3 + bx^2 + 1) = (x^2 - x - 1) * (x + c)) : 
  b = -2 :=
  sorry

end value_of_b_l2124_212470


namespace hour_hand_rotations_l2124_212478

theorem hour_hand_rotations (degrees_per_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) (rotations_per_day : ℕ) :
  degrees_per_hour = 30 →
  hours_per_day = 24 →
  rotations_per_day = (degrees_per_hour * hours_per_day) / 360 →
  days = 6 →
  rotations_per_day * days = 12 :=
by
  intros
  sorry

end hour_hand_rotations_l2124_212478


namespace regression_line_l2124_212429

theorem regression_line (m x1 y1 : ℝ) (h_slope : m = 1.23) (h_center : (x1, y1) = (4, 5)) :
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ y = 1.23 * x + 0.08) :=
by
  use 0.08
  sorry

end regression_line_l2124_212429


namespace work_rate_l2124_212445

/-- 
A alone can finish a work in some days which B alone can finish in 15 days. 
If they work together and finish it, then out of a total wages of Rs. 3400, 
A will get Rs. 2040. Prove that A alone can finish the work in 22.5 days. 
-/
theorem work_rate (A : ℚ) (B_rate : ℚ) 
  (total_wages : ℚ) (A_wages : ℚ) 
  (total_rate : ℚ) 
  (hB : B_rate = 1 / 15) 
  (hWages : total_wages = 3400 ∧ A_wages = 2040) 
  (hTotal : total_rate = 1 / A + B_rate)
  (hWorkTogether : 
    (A_wages / (total_wages - A_wages) = 51 / 34) ↔ 
    (A / (A + 15) = 51 / 85)) : 
  A = 22.5 := 
sorry

end work_rate_l2124_212445


namespace functional_eq_solution_l2124_212434

noncomputable def functional_solution (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * y * f x

theorem functional_eq_solution (f : ℝ → ℝ) (h : functional_solution f) :
  ∀ x : ℝ, f x = 0 ∨ f x = x^2 :=
sorry

end functional_eq_solution_l2124_212434


namespace czakler_inequality_l2124_212448

variable {a b : ℕ} (ha : a > 0) (hb : b > 0)
variable {c : ℝ} (hc : c > 0)

theorem czakler_inequality (h : (a + 1 : ℝ) / (b + c) = b / a) : c ≥ 1 := by
  sorry

end czakler_inequality_l2124_212448


namespace ninth_term_of_sequence_is_4_l2124_212401

-- Definition of the first term and common ratio
def a1 : ℚ := 4
def r : ℚ := 1

-- Definition of the nth term of a geometric sequence
def a (n : ℕ) : ℚ := a1 * r^(n-1)

-- Proof that the ninth term of the sequence is 4
theorem ninth_term_of_sequence_is_4 : a 9 = 4 := by
  sorry

end ninth_term_of_sequence_is_4_l2124_212401


namespace q_join_after_days_l2124_212442

noncomputable def workRate (totalWork : ℕ) (days : ℕ) : ℚ :=
  totalWork / days

theorem q_join_after_days (W : ℕ) (days_p : ℕ) (days_q : ℕ) (total_days : ℕ) (x : ℕ) :
  days_p = 80 ∧ days_q = 48 ∧ total_days = 35 ∧ 
  ((workRate W days_p) * x + (workRate W days_p + workRate W days_q) * (total_days - x) = W) 
  → x = 8 := sorry

end q_join_after_days_l2124_212442


namespace expression_value_l2124_212472

theorem expression_value :
  3 * 12^2 - 3 * 13 + 2 * 16 * 11^2 = 4265 :=
by
  sorry

end expression_value_l2124_212472


namespace average_after_12th_innings_l2124_212498

variable (runs_11 score_12 increase_avg : ℕ)
variable (A : ℕ)

theorem average_after_12th_innings
  (h1 : score_12 = 60)
  (h2 : increase_avg = 2)
  (h3 : 11 * A = runs_11)
  (h4 : (runs_11 + score_12) / 12 = A + increase_avg) :
  (A + 2 = 38) :=
by
  sorry

end average_after_12th_innings_l2124_212498


namespace Nunzio_eats_pizza_every_day_l2124_212455

theorem Nunzio_eats_pizza_every_day
  (one_piece_fraction : ℚ := 1/8)
  (total_pizzas : ℕ := 27)
  (total_days : ℕ := 72)
  (pieces_per_pizza : ℕ := 8)
  (total_pieces : ℕ := total_pizzas * pieces_per_pizza)
  : (total_pieces / total_days = 3) :=
by
  -- We assume 1/8 as a fraction for the pieces of pizza is stated in the conditions, therefore no condition here.
  -- We need to show that Nunzio eats 3 pieces of pizza every day given the total pieces and days.
  sorry

end Nunzio_eats_pizza_every_day_l2124_212455


namespace joan_seashells_left_l2124_212481

theorem joan_seashells_left (original_seashells : ℕ) (given_seashells : ℕ) (seashells_left : ℕ)
  (h1 : original_seashells = 70) (h2 : given_seashells = 43) : seashells_left = 27 :=
by
  sorry

end joan_seashells_left_l2124_212481


namespace cos_585_eq_neg_sqrt2_div_2_l2124_212482

theorem cos_585_eq_neg_sqrt2_div_2 : Real.cos (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by sorry

end cos_585_eq_neg_sqrt2_div_2_l2124_212482


namespace f_is_periodic_l2124_212403

noncomputable def f : ℝ → ℝ := sorry

def a : ℝ := sorry

axiom exists_a_gt_zero : a > 0

axiom functional_eq (x : ℝ) : f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)

theorem f_is_periodic : ∀ x : ℝ, f (x + 2 * a) = f x := sorry

end f_is_periodic_l2124_212403


namespace solve_for_N_l2124_212464

theorem solve_for_N (a b c N : ℝ) 
  (h1 : a + b + c = 72) 
  (h2 : a - 7 = N) 
  (h3 : b + 7 = N) 
  (h4 : 2 * c = N) : 
  N = 28.8 := 
sorry

end solve_for_N_l2124_212464


namespace lcm_of_4_5_6_9_is_180_l2124_212460

theorem lcm_of_4_5_6_9_is_180 : Nat.lcm (Nat.lcm 4 5) (Nat.lcm 6 9) = 180 :=
by
  sorry

end lcm_of_4_5_6_9_is_180_l2124_212460


namespace mistaken_quotient_is_35_l2124_212444

theorem mistaken_quotient_is_35 (D : ℕ) (correct_divisor mistaken_divisor correct_quotient : ℕ) 
    (h1 : D = correct_divisor * correct_quotient)
    (h2 : correct_divisor = 21)
    (h3 : mistaken_divisor = 12)
    (h4 : correct_quotient = 20)
    : D / mistaken_divisor = 35 := by
  sorry

end mistaken_quotient_is_35_l2124_212444
