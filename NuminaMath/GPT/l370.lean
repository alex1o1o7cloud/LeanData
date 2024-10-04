import Mathlib

namespace num_valid_codes_l370_370352

def validCode (c : ℕ) : Prop :=
  c >= 0 ∧ c < 1000

def differingAtMostOnePosition (c1 c2 : ℕ) : Prop :=
  let d1 := c1 / 100
  let d2 := (c1 / 10) % 10
  let d3 := c1 % 10
  let d1' := c2 / 100
  let d2' := (c2 / 10) % 10
  let d3' := c2 % 10
  ((d1 = d1' ∧ d2 = d2' ∧ d3 ≠ d3') ∨
   (d1 = d1' ∧ d2 ≠ d2' ∧ d3 = d3') ∨
   (d1 ≠ d1' ∧ d2 = d2' ∧ d3 = d3'))

def isTranspositionOrEqual (c1 c2 : ℕ) : Prop :=
  let d1 := c1 / 100
  let d2 := (c1 / 10) % 10
  let d3 := c1 % 10
  let d1' := c2 / 100
  let d2' := (c2 / 10) % 10
  let d3' := c2 % 10
  (c1 = c2) ∨ ((d1 = d1' ∧ d2 = d3' ∧ d3 = d2') ∨ (d1 = d2' ∧ d2 = d1' ∧ d3 = d3') ∨ (d1 = d3' ∧ d2 = d2' ∧ d3 = d1'))

theorem num_valid_codes :
  let ref_code := 145 in
  (∑ x in Finset.range 1000, if validCode x ∧ ¬ differingAtMostOnePosition ref_code x ∧ ¬ isTranspositionOrEqual ref_code x then 1 else 0) = 969 :=
by
  sorry

end num_valid_codes_l370_370352


namespace sugar_price_increase_sugar_price_percentage_increase_l370_370392

theorem sugar_price_increase (P P_new : ℝ) (h1 : 25 * P_new = (30 * P) * 1.10) : 
  P_new = 1.32 * P :=
  sorry

theorem sugar_price_percentage_increase (P P_new : ℝ) (h1 : 25 * P_new = (30 * P) * 1.10) :
  ((P_new / P) * 100 - 100) = 32 :=
  by
    have h : P_new = 1.32 * P := sugar_price_increase P P_new h1
    rw [h]
    calc 
      ((1.32 * P) / P * 100 - 100)
        = (1.32 * (P / P) * 100 - 100) : by ring
        = (1.32 * 1 * 100 - 100)     : by rw [div_self (ne_of_gt (real.zero_lt_one))]
        = (1.32 * 100 - 100)         : by ring
        = 132 - 100                  : by ring
        = 32                         : by ring

end sugar_price_increase_sugar_price_percentage_increase_l370_370392


namespace juanitas_dessert_cost_is_correct_l370_370142

noncomputable def brownie_cost := 2.50
noncomputable def regular_scoop_cost := 1.00
noncomputable def premium_scoop_cost := 1.25
noncomputable def deluxe_scoop_cost := 1.50
noncomputable def syrup_cost := 0.50
noncomputable def nuts_cost := 1.50
noncomputable def whipped_cream_cost := 0.75
noncomputable def cherry_cost := 0.25
noncomputable def discount_tuesday := 0.10

noncomputable def total_cost_of_juanitas_dessert :=
    let discounted_brownie := brownie_cost * (1 - discount_tuesday)
    let ice_cream_cost := 2 * regular_scoop_cost + premium_scoop_cost
    let syrup_total := 2 * syrup_cost
    let additional_toppings := nuts_cost + whipped_cream_cost + cherry_cost
    discounted_brownie + ice_cream_cost + syrup_total + additional_toppings
   
theorem juanitas_dessert_cost_is_correct:
  total_cost_of_juanitas_dessert = 9.00 := by
  sorry

end juanitas_dessert_cost_is_correct_l370_370142


namespace desired_circle_l370_370208

variables {x y : ℝ}

def circle1 := x^2 + y^2 - x + y - 2 = 0
def circle2 := x^2 + y^2 = 5
def line := 3*x + 4*y - 1 = 0

theorem desired_circle :
  (∃ (λ : ℝ), λ ≠ -1 ∧ ∀ x y : ℝ, (circle1 ∧ circle2 ∧ line) → 
  x^2 + y^2 + 2*x - 2*y - 11 = 0) :=
sorry

end desired_circle_l370_370208


namespace correct_graph_representation_l370_370537

/--
A ship travels from point A to point B along a semicircular path, centered at Island X.
It then travels along a straight path from B to C. 
We need to prove that the correct graph which best shows the ship's distance from Island X as it moves along its course
- a horizontal line at height r from A to B and
- a curve that dips to a minimum and then rises again from B to C.
is option (B).
-/
theorem correct_graph_representation
  (A B C X : Point) (path_AB : SemicircularPath) (path_BC : LinearPath)
  (r : ℝ)
  (h1 : CenteredAt path_AB X)
  (h2 : DistanceToCenterConstant path_AB X r)
  (h3 : DistanceDecreasesAndIncreases path_BC X) :
  GraphRepresentation path_AB path_BC = GraphOption.B := sorry

end correct_graph_representation_l370_370537


namespace prism_edges_vertices_l370_370958

theorem prism_edges_vertices (lateral_faces : ℕ) (h : lateral_faces = 4) : 
  (edges vertices : ℕ) :=
by
  have edges := 12
  have vertices := 8
  sorry

end prism_edges_vertices_l370_370958


namespace g_2_pow_4_eq_4096_l370_370037

variable {X : Type} [LinearOrderedField X]

def f (x : X) : X
def g (x : X) : X

axiom f_g_x3 (x : X) (h : x ≥ 1) : f (g x) = x^3
axiom g_f_x4 (x : X) (h : x ≥ 1) : g (f x) = x^4
axiom g_16_eq_8 : g (16 : X) = 8

theorem g_2_pow_4_eq_4096 : [g (2 : X)]^4 = 4096 := by
  sorry

end g_2_pow_4_eq_4096_l370_370037


namespace ratio_of_boys_to_total_l370_370717

theorem ratio_of_boys_to_total (b : ℝ) (h1 : b = 3 / 4 * (1 - b)) : b = 3 / 7 :=
by
  {
    -- The given condition (we use it to prove the target statement)
    sorry
  }

end ratio_of_boys_to_total_l370_370717


namespace find_values_of_p_l370_370630

-- Define the system of equations
def system_of_equations (n : ℕ) (p : ℝ) (x : (Fin n) → ℝ) : Prop :=
  ∀ i : Fin n, x i ^ 4 + 2 / (x i ^ 2) = p * (x (i + 1))

-- Define the property of having at least two sets of real roots
def has_at_least_two_sets_of_real_roots (n : ℕ) (p : ℝ) : Prop :=
  ∃ x1 x2 : (Fin n) → ℝ, system_of_equations n p x1 ∧ system_of_equations n p x2 ∧ x1 ≠ x2

-- The main theorem statement
theorem find_values_of_p (n : ℕ) (h : n ≥ 2) : 
  ∃ p : ℝ, (p ∈ Ioo (-2 * Real.sqrt 2) (2 * Real.sqrt 2) ↔ has_at_least_two_sets_of_real_roots n p) :=
sorry

end find_values_of_p_l370_370630


namespace probability_of_event_l370_370013

noncomputable def interval_probability : ℝ :=
  if 0 ≤ 1 ∧ 1 ≤ 1 then (1 - (1/3)) / (1 - 0) else 0

theorem probability_of_event :
  interval_probability = 2 / 3 :=
by
  rw [interval_probability]
  sorry

end probability_of_event_l370_370013


namespace cos_beta_zero_l370_370245

theorem cos_beta_zero (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) (h3 : Real.cos α = 1 / 2) (h4 : Real.cos (α + β) = -1 / 2) : Real.cos β = 0 :=
sorry

end cos_beta_zero_l370_370245


namespace uniform_prob_correct_l370_370959

noncomputable def uniform_prob_within_interval 
  (α β γ δ : ℝ) 
  (h₁ : α ≤ β) 
  (h₂ : α ≤ γ) 
  (h₃ : γ < δ) 
  (h₄ : δ ≤ β) : ℝ :=
  (δ - γ) / (β - α)

theorem uniform_prob_correct 
  (α β γ δ : ℝ) 
  (hαβ : α ≤ β) 
  (hαγ : α ≤ γ) 
  (hγδ : γ < δ) 
  (hδβ : δ ≤ β) :
  uniform_prob_within_interval α β γ δ hαβ hαγ hγδ hδβ = (δ - γ) / (β - α) := sorry

end uniform_prob_correct_l370_370959


namespace edward_chocolate_l370_370809

theorem edward_chocolate (total_chocolate : ℚ) (num_piles : ℕ) (piles_received_by_Edward : ℕ) :
  total_chocolate = 75 / 7 → num_piles = 5 → piles_received_by_Edward = 2 → 
  (total_chocolate / num_piles) * piles_received_by_Edward = 30 / 7 := 
by
  intros ht hn hp
  sorry

end edward_chocolate_l370_370809


namespace problem1_problem2_l370_370492

theorem problem1 (x : ℝ) (h : x^(1/2) + x^(-1/2) = 3) : x + 1/x = 7 := 
  sorry

theorem problem2 : (Real.log 3 / (Real.log 4) + Real.log 3 / (Real.log 8)) * 
                   (Real.log 2 / (Real.log 3) + (3 * Real.log 2 / (2 * Real.log 3))) = 25 / 12 :=
  sorry

end problem1_problem2_l370_370492


namespace hyperbola_asymptotes_l370_370270

theorem hyperbola_asymptotes (a b : ℝ) (k : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) (e : ℝ) (h₄ : e = sqrt 6 / 2) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (sqrt 2 / 2) * x ∨ y = -(sqrt 2 / 2) * x) :=
sorry

end hyperbola_asymptotes_l370_370270


namespace power_function_passes_through_1_1_l370_370379

theorem power_function_passes_through_1_1 (a : ℝ) : (1 : ℝ) ^ a = 1 := 
by
  sorry

end power_function_passes_through_1_1_l370_370379


namespace find_K_l370_370320

def surface_area_cube (side_length : ℝ) : ℝ :=
  6 * side_length^2

def surface_area_sphere (radius : ℝ) : ℝ :=
  4 * Real.pi * radius^2

def volume_sphere (radius : ℝ) : ℝ :=
  (4 / 3) * Real.pi * radius^3

theorem find_K :
  let side_length := 3
  let surface_area := surface_area_cube side_length
  let radius := Real.sqrt (27 / (2 * Real.pi))
  let volume := volume_sphere radius
  volume = (K : ℝ) * Real.sqrt 6 / (Real.sqrt Real.pi)
  → K = 27 * Real.sqrt 6 / Real.sqrt 2 := 
by
  intros side_length surface_area radius volume h
  sorry

end find_K_l370_370320


namespace find_second_number_l370_370480

theorem find_second_number (a b c : ℝ) (h1 : a + b + c = 3.622) (h2 : a = 3.15) (h3 : c = 0.458) : b = 0.014 :=
sorry

end find_second_number_l370_370480


namespace solution_set_of_inequality_l370_370058

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3 * x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_of_inequality_l370_370058


namespace solve_inequality_system_l370_370028

theorem solve_inequality_system (x : ℝ) : 
  (2 + x > 7 - 4x ∧ x < (4 + x) / 2) → (1 < x ∧ x < 4) :=
by
  intro h
  cases h with h1 h2
  sorry

end solve_inequality_system_l370_370028


namespace some_expression_minimum_value_l370_370081

noncomputable def abs (x : ℝ) := if x ≥ 0 then x else -x

def minimum_some_expression : ℝ :=
  let a := | -1 - 4 |
  let b := | -1 + 6 |
  11 - (a + b)

theorem some_expression_minimum_value :
  minimum_some_expression = 1 := by
sorry

end some_expression_minimum_value_l370_370081


namespace value_of_expression_l370_370197

theorem value_of_expression (a b c k : ℕ) (h_a : a = 30) (h_b : b = 25) (h_c : c = 4) (h_k : k = 3) : 
  (a - (b - k * c)) - ((a - k * b) - c) = 66 :=
by
  rw [h_a, h_b, h_c, h_k]
  simp
  sorry

end value_of_expression_l370_370197


namespace lcm_12_15_18_l370_370883

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end lcm_12_15_18_l370_370883


namespace value_of_m_sub_n_l370_370642

theorem value_of_m_sub_n (m n : ℤ) (h1 : |m| = 5) (h2 : n^2 = 36) (h3 : m * n < 0) : m - n = 11 ∨ m - n = -11 := 
by 
  sorry

end value_of_m_sub_n_l370_370642


namespace lcm_12_15_18_l370_370881

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by 
  sorry

end lcm_12_15_18_l370_370881


namespace MN_length_triangle_l370_370003

variable (a b : ℝ) (h : a > b)

def is_segment_ratio (A B C : Point) (r : ℝ) : Prop :=
  ∃ P : Point, A ≠ C ∧ r = (|A - P| / |P - C|)

def MN_length (A B C D M N : Point) : ℝ :=
  if is_segment_ratio A M C (1/4) ∧ is_segment_ratio D N B (1/4)
  then (1/5) * (4 * |A - D| - |B - C|)
  else 0

theorem MN_length_triangle (A B C D M N : Point) 
  (h1 : is_segment_ratio A M C (1/4)) 
  (h2 : is_segment_ratio D N B (1/4))
  (h3 : |A - D| = a) 
  (h4 : |B - C| = b) 
  : |MN_length A B C D M N| = (1/5) * (4 * a - b) :=
by
  sorry

end MN_length_triangle_l370_370003


namespace negation_correct_l370_370053

variable {α : Type}

def honor_student (x : α) : Prop := sorry
def receives_scholarship (x : α) : Prop := sorry

theorem negation_correct :
  ¬(∀ x, honor_student x → receives_scholarship x) ↔ ∃ x, honor_student x ∧ ¬receives_scholarship x :=
sorry

end negation_correct_l370_370053


namespace monotonic_function_a_le_3_l370_370678

theorem monotonic_function_a_le_3 (a : ℝ) :
  (∀ x : ℝ, 1 < x → ∀ x₁ x₂ : ℝ, (1 < x₁) → (1 < x₂) → x₁ <= x₂ → f x₁ <= f x₂) → -- Monotonicity condition
  a <= 3 :=
by
  let f : ℝ → ℝ := λ x, -x^3 + a * x
  sorry

end monotonic_function_a_le_3_l370_370678


namespace ratio_Mandy_to_Pamela_l370_370296

-- Definitions based on conditions in the problem
def exam_items : ℕ := 100
def Lowella_correct : ℕ := (35 * exam_items) / 100  -- 35% of 100
def Pamela_correct : ℕ := Lowella_correct + (20 * Lowella_correct) / 100 -- 20% more than Lowella
def Mandy_score : ℕ := 84

-- The proof problem statement
theorem ratio_Mandy_to_Pamela : Mandy_score / Pamela_correct = 2 := by
  sorry

end ratio_Mandy_to_Pamela_l370_370296


namespace smallest_integer_a_l370_370104

theorem smallest_integer_a (a : ℤ) (b : ℤ) (h1 : a < 21) (h2 : 20 ≤ b) (h3 : b < 31) (h4 : (a : ℝ) / b < 2 / 3) : 13 < a :=
sorry

end smallest_integer_a_l370_370104


namespace sin_210_eq_neg_one_half_l370_370581

theorem sin_210_eq_neg_one_half :
  sin (Real.pi * (210 / 180)) = -1 / 2 :=
by
  have angle_eq : 210 = 180 + 30 := by norm_num
  have sin_30 : sin (Real.pi / 6) = 1 / 2 := by norm_num
  have cos_30 : cos (Real.pi / 6) = sqrt 3 / 2 := by norm_num
  sorry

end sin_210_eq_neg_one_half_l370_370581


namespace range_of_tan_theta_l370_370643

theorem range_of_tan_theta (θ : ℝ) (h : (sin θ) / (sqrt 3 * cos θ + 1) > 1) :
  (tan θ) ∈ Ioo (NegInf ℝ) (real.sqrt 2 * -1) ∪ Ioo (real.sqrt 3 / 3) (real.sqrt 2) :=
sorry

end range_of_tan_theta_l370_370643


namespace sin_210_l370_370564

theorem sin_210 : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end sin_210_l370_370564


namespace exp_monotonic_k_iff_gt_zero_l370_370292

-- Define the function
def f (k x : ℝ) : ℝ := Real.exp (k * x)

-- Define the condition that the function is monotonically increasing
def monotone_function (k : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ≤ x2 → f k x1 ≤ f k x2

-- The statement to be proved
theorem exp_monotonic_k_iff_gt_zero (k : ℝ) : (∀ x : ℝ, Real.exp (k * x) > 0) ∧ monotone_function k ↔ k > 0 :=
by
  sorry

end exp_monotonic_k_iff_gt_zero_l370_370292


namespace lcm_of_12_15_18_is_180_l370_370886

theorem lcm_of_12_15_18_is_180 :
  Nat.lcm 12 (Nat.lcm 15 18) = 180 := by
  sorry

end lcm_of_12_15_18_is_180_l370_370886


namespace no_integer_solution_l370_370878

theorem no_integer_solution :
  ∀ x : ℤ, ¬ prime (|4 * x^2 - 39 * x + 35|) :=
by sorry

end no_integer_solution_l370_370878


namespace quadratic_root_difference_l370_370738

theorem quadratic_root_difference (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ + x₂ = 2 ∧ x₁ * x₂ = a ∧ (x₁ - x₂)^2 = 20) → a = -4 := 
by
  sorry

end quadratic_root_difference_l370_370738


namespace log_expression_comparison_l370_370287

theorem log_expression_comparison : 
  let x := Real.sqrt 2,
      y := Real.cbrt 3,
      z := Real.root 5 5
  in 
  z < x ∧ x < y := 
by
  sorry

end log_expression_comparison_l370_370287


namespace lcm_12_15_18_l370_370879

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by 
  sorry

end lcm_12_15_18_l370_370879


namespace largest_difference_l370_370692

noncomputable def arithmetic_mean (a b c : ℝ) : ℝ :=
  (a + b + c) / 3

noncomputable def geometric_mean (a b c : ℝ) : ℝ :=
  (a * b * c)^(1/3 : ℝ)

noncomputable def harmonic_mean (a b c : ℝ) : ℝ :=
  3 / (1/a + 1/b + 1/c)

theorem largest_difference (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_ineq : a < b ∧ b < c) :
  let am := arithmetic_mean a b c,
      gm := geometric_mean a b c,
      hm := harmonic_mean a b c in
  (am - hm) = max (am - gm) (max (gm - hm) (am - hm)) :=
by sorry

end largest_difference_l370_370692


namespace part_one_iff_part_two_iff_l370_370793

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) - abs (x + 2)

theorem part_one_iff (x : ℝ) : f(x) > 0 ↔ (x < (-1 : ℝ) / 3 ∨ x > 3) :=
by sorry

theorem part_two_iff (m : ℝ) : (∃ (x0 : ℝ), f(x0) + 2 * m^2 < 4 * m) ↔ (-1 / 2 < m ∧ m < 5 / 2) :=
by sorry

end part_one_iff_part_two_iff_l370_370793


namespace range_of_m_l370_370792

noncomputable def max_ab (a b : ℝ) : ℝ :=
if a >= b then a else b

noncomputable def M (x y : ℝ) : ℝ :=
max_ab (abs (x - y^2 + 4)) (abs (2 * y^2 - x + 8))

theorem range_of_m (m : ℝ) :
  (∀ (x y : ℝ), M x y ≥ m^2 - 2 * m) →
  (1 - real.sqrt 7 ≤ m ∧ m ≤ 1 + real.sqrt 7) :=
sorry

end range_of_m_l370_370792


namespace cone_properties_l370_370664

noncomputable def radius (h l : ℝ) : ℝ := 
  real.sqrt (l^2 - h^2)

noncomputable def cone_volume (r h : ℝ) : ℝ := 
  (1 / 3) * real.pi * r^2 * h

noncomputable def central_angle (r l : ℝ) : ℝ :=
  2 * real.pi * r / l

noncomputable def max_area_SAB (l : ℝ) : ℝ := 
  l * real.sin (real.pi / 2)

noncomputable def sphere_volume (R : ℝ) : ℝ := 
  (4 / 3) * real.pi * R^3

theorem cone_properties : 
  let h := 1
  let l := 2 
  let r := radius h l
  let R := 2
  cone_volume r h = real.pi ∧ 
  central_angle r l = real.sqrt 3 * real.pi ∧
  max_area_SAB l = 2 ∧ 
  sphere_volume R = 32*real.pi/3 
  :=
by
  sorry

end cone_properties_l370_370664


namespace log_sum_equality_l370_370915

theorem log_sum_equality (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) :
  (∑ k in finset.range (n + 1), binomial n k * log (a ^ (n - k) * b ^ k)) = log ((a * b) ^ (n * 2 ^ (n - 1))) :=
by
  sorry

end log_sum_equality_l370_370915


namespace ticket_price_l370_370699

theorem ticket_price
  (tickets_self_and_friends : ℕ = 3)
  (extra_tickets : ℕ = 5)
  (total_spent : ℕ = 32) :
  ∃ x : ℕ, 3 * x + 5 * x = 32 ∧ x = 4 :=
by
  sorry

end ticket_price_l370_370699


namespace digits_solution_exists_l370_370078

theorem digits_solution_exists (a b : ℕ) (ha : a < 10) (hb : b < 10) 
  (h : a = (b * (10 * b)) / (10 - b)) : a = 5 ∧ b = 2 :=
by
  sorry

end digits_solution_exists_l370_370078


namespace variance_transformation_l370_370713

variable (x : ℕ → ℝ) (n : ℕ)

noncomputable def variance (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  let mean := (∑ i in finset.range n, x i) / n
  (∑ i in finset.range n, (x i - mean) ^ 2) / n

theorem variance_transformation (x : ℕ → ℝ) (n : ℕ) (h : variance x n = 2) :
  variance (λ i, -3 * x i + 1) n = 18 := by
  sorry

end variance_transformation_l370_370713


namespace price_of_adult_ticket_eq_32_l370_370499

theorem price_of_adult_ticket_eq_32 
  (num_adults : ℕ)
  (num_children : ℕ)
  (price_child_ticket : ℕ)
  (price_adult_ticket : ℕ)
  (total_collected : ℕ)
  (h1 : num_adults = 400)
  (h2 : num_children = 200)
  (h3 : price_adult_ticket = 2 * price_child_ticket)
  (h4 : total_collected = 16000)
  (h5 : total_collected = num_adults * price_adult_ticket + num_children * price_child_ticket)
  : price_adult_ticket = 32 := 
by
  sorry

end price_of_adult_ticket_eq_32_l370_370499


namespace possible_to_arrange_Xs_and_Os_on_grid_l370_370313

theorem possible_to_arrange_Xs_and_Os_on_grid : 
  ∃ (grid : ℕ → ℕ → char), 
    (∀ i j : ℕ, 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5 → 
      grid i j = 'X' ∨ grid i j = 'O') ∧
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 → 
      ¬(grid i 1 = grid i 2 ∧ grid i 2 = grid i 3) ∧
      ¬(grid i 2 = grid i 3 ∧ grid i 3 = grid i 4) ∧
      ¬(grid i 3 = grid i 4 ∧ grid i 4 = grid i 5)) ∧
    (∀ j : ℕ, 1 ≤ j ∧ j ≤ 5 → 
      ¬(grid 1 j = grid 2 j ∧ grid 2 j = grid 3 j) ∧
      ¬(grid 2 j = grid 3 j ∧ grid 3 j = grid 4 j) ∧
      ¬(grid 3 j = grid 4 j ∧ grid 4 j = grid 5 j)) ∧
    (∀ d : ℤ, 
      ¬(grid (int.to_nat d) (int.to_nat d) = grid (int.to_nat (d + 1)) (int.to_nat (d + 1)) ∧ 
        grid (int.to_nat (d + 1)) (int.to_nat (d + 1)) = grid (int.to_nat (d + 2)) (int.to_nat (d + 2))) ∧
      ¬(grid (int.to_nat d) (5 - int.to_nat d + 1) = grid (int.to_nat (d + 1)) (5 - int.to_nat (d + 1) + 1) ∧ 
        grid (int.to_nat (d + 1)) (5 - int.to_nat (d + 1) + 1) = grid (int.to_nat (d + 2)) (5 - int.to_nat (d + 2) + 1))) :=
sorry

end possible_to_arrange_Xs_and_Os_on_grid_l370_370313


namespace find_b_skew_lines_l370_370615

def line1 (b : ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 3*t, 3 + 4*t, b + 5*t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ :=
  (5 + 6*u, 6 + 3*u, 1 + 2*u)

noncomputable def lines_are_skew (b : ℝ) : Prop :=
  ∀ t u : ℝ, line1 b t ≠ line2 u

theorem find_b_skew_lines (b : ℝ) : b ≠ -12 / 5 → lines_are_skew b :=
by
  sorry

end find_b_skew_lines_l370_370615


namespace pirates_treasure_l370_370421

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l370_370421


namespace treasures_coins_count_l370_370407

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l370_370407


namespace domain_of_f_l370_370619

def cube_root_domain (x : ℝ) : ℝ :=
  real.cbrt x

def f (x : ℝ) : ℝ :=
  real.cbrt (2*x - 3) + real.cbrt (5 - 2*x)

theorem domain_of_f : ∀ x : ℝ, true :=
by
  intros x
  sorry

end domain_of_f_l370_370619


namespace ratio_of_a_to_c_l370_370056

variables {a b c d : ℚ}

theorem ratio_of_a_to_c
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) :
  a / c = 75 / 16 :=
sorry

end ratio_of_a_to_c_l370_370056


namespace gcd_lcm_find_other_number_l370_370839

theorem gcd_lcm_find_other_number {a b : ℕ} (h_gcd : Nat.gcd a b = 36) (h_lcm : Nat.lcm a b = 8820) (h_a : a = 360) : b = 882 :=
by
  sorry

end gcd_lcm_find_other_number_l370_370839


namespace part_I_part_II_l370_370677

def f (x : ℝ) := abs (x + 1)

theorem part_I (x : ℝ) : 
  (f(x + 8) ≥ 10 - f(x)) ↔ (x ≤ -10 ∨ x ≥ 0) :=
sorry

theorem part_II (x y : ℝ) (hx : abs x > 1) (hy : abs y < 1) : 
  f(y) < abs x * f(y / (x^2)) :=
sorry

end part_I_part_II_l370_370677


namespace complement_of_A_in_U_l370_370345

variable (U : Set ℝ) (A : Set ℝ)
def U := {x : ℝ | x > 1}
def A := {x : ℝ | x > 2}

theorem complement_of_A_in_U : (U \ A) = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end complement_of_A_in_U_l370_370345


namespace simplify_cubic_root_l370_370020

theorem simplify_cubic_root (a b : ℕ) (h1 : 2744000 = b) (h2 : ∛ b = 140) (h3 : 5488000 = 2 * b) :
  ∛ 5488000 = 140 * ∛ 2 :=
by
  sorry

end simplify_cubic_root_l370_370020


namespace solve_equation1_solve_equation2_solve_equation3_l370_370368

theorem solve_equation1 (x : ℝ) : 4 * (x - 1)^2 - 36 = 0 ↔ x = 4 ∨ x = -2 := sorry

theorem solve_equation2 (x : ℝ) : x^2 + 2x - 3 = 0 ↔ x = -3 ∨ x = 1 := sorry

theorem solve_equation3 (x : ℝ) : x * (x - 4) = 8 - 2x ↔ x = 4 ∨ x = -2 := sorry

end solve_equation1_solve_equation2_solve_equation3_l370_370368


namespace sin_210_l370_370567

theorem sin_210 : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end sin_210_l370_370567


namespace sin_210_eq_neg_half_l370_370560

theorem sin_210_eq_neg_half : Real.sin (210 * Real.pi / 180) = -1 / 2 := by
  -- We use the given angles and their known sine values.
  have angle_30 := Real.pi / 6
  have sin_30 := Real.sin angle_30
  -- Expression for the sine of 210 degrees in radians.
  have angle_210 := 210 * Real.pi / 180
  -- Proving the sine of 210 degrees using angle addition formula and unit circle properties.
  calc
    Real.sin angle_210 
    -- 210 degrees is 180 + 30 degrees, translating to pi and pi/6 in radians.
    = Real.sin (Real.pi + Real.pi / 6) : by rw [←Real.ofReal_nat_cast, ←Real.ofReal_nat_cast, Real.ofReal_add, Real.ofReal_div, Real.ofReal_nat_cast]
    -- Using the sine addition formula: sin(pi + x) = -sin(x).
    ... = - Real.sin (Real.pi / 6) : by exact Real.sin_add_pi_div_two angle_30
    -- Substituting the value of sin(30 degrees).
    ... = - 1 / 2 : by rw sin_30

end sin_210_eq_neg_half_l370_370560


namespace toms_next_birthday_l370_370457

-- Define the variables and conditions
variable (t j s : ℝ)
variable (hj : j = 1.2 * s)
variable (ht : t = 0.7 * j)
variable (h_sum : t + j + s = 36)

-- Prove that Tom's age on his next birthday is 11
theorem toms_next_birthday : t = 9.9474 → ceil t = 11 := by
  intros ht_value
  have h_ceil_t : ceil t = 11, from sorry
  exact h_ceil_t

end toms_next_birthday_l370_370457


namespace sin_210_eq_neg_half_l370_370571

theorem sin_210_eq_neg_half : sin (210 * real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_210_eq_neg_half_l370_370571


namespace prime_number_solution_l370_370205

theorem prime_number_solution (X Y : ℤ) (h_prime : Prime (X^4 + 4 * Y^4)) :
  (X = 1 ∧ Y = 1) ∨ (X = -1 ∧ Y = -1) :=
sorry

end prime_number_solution_l370_370205


namespace partner_q_investment_time_l370_370057

-- Define the conditions
def investment_ratio (p q : ℝ) := p / q = 7 / 5
def profit_ratio (p q : ℝ) (tp tq : ℝ) := p * tp / (q * tq) = 7 / 10
def p_investment_time := 2

-- Define the theorem to be proved
theorem partner_q_investment_time (x : ℝ) (tp tq : ℝ) (investment_p investment_q : ℝ) (profit_p profit_q : ℝ)
  (h1 : investment_ratio investment_p investment_q) (h2 : profit_ratio (investment_p * tp) (investment_q * tq))
  (h3 : tp = p_investment_time) : tq = 4 :=
  sorry

end partner_q_investment_time_l370_370057


namespace sin_210_eq_neg_half_l370_370561

theorem sin_210_eq_neg_half : Real.sin (210 * Real.pi / 180) = -1 / 2 := by
  -- We use the given angles and their known sine values.
  have angle_30 := Real.pi / 6
  have sin_30 := Real.sin angle_30
  -- Expression for the sine of 210 degrees in radians.
  have angle_210 := 210 * Real.pi / 180
  -- Proving the sine of 210 degrees using angle addition formula and unit circle properties.
  calc
    Real.sin angle_210 
    -- 210 degrees is 180 + 30 degrees, translating to pi and pi/6 in radians.
    = Real.sin (Real.pi + Real.pi / 6) : by rw [←Real.ofReal_nat_cast, ←Real.ofReal_nat_cast, Real.ofReal_add, Real.ofReal_div, Real.ofReal_nat_cast]
    -- Using the sine addition formula: sin(pi + x) = -sin(x).
    ... = - Real.sin (Real.pi / 6) : by exact Real.sin_add_pi_div_two angle_30
    -- Substituting the value of sin(30 degrees).
    ... = - 1 / 2 : by rw sin_30

end sin_210_eq_neg_half_l370_370561


namespace count_students_neither_math_physics_chemistry_l370_370353

def total_students := 150

def students_math := 90
def students_physics := 70
def students_chemistry := 40

def students_math_and_physics := 20
def students_math_and_chemistry := 15
def students_physics_and_chemistry := 10
def students_all_three := 5

theorem count_students_neither_math_physics_chemistry :
  (total_students - 
   (students_math + students_physics + students_chemistry - 
    students_math_and_physics - students_math_and_chemistry - 
    students_physics_and_chemistry + students_all_three)) = 5 := by
  sorry

end count_students_neither_math_physics_chemistry_l370_370353


namespace tunnel_length_l370_370924

def train_length : ℝ := 1.5
def exit_time_minutes : ℝ := 4
def speed_mph : ℝ := 45

theorem tunnel_length (d_train : ℝ := train_length)
                      (t_exit : ℝ := exit_time_minutes)
                      (v_mph : ℝ := speed_mph) :
  d_train + ((v_mph / 60) * t_exit - d_train) = 1.5 :=
by
  sorry

end tunnel_length_l370_370924


namespace sum_D_r_squared_le_bound_l370_370770

-- Define S as a finite set of points in general position
def S : Type := { s : ℝ × ℝ // true }
noncomputable instance : fintype S := infer_instance

-- Define general position: no three points in S are collinear
def general_position (S : set (ℝ × ℝ)) : Prop :=
  ∀ (p q r : ℝ × ℝ), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r →
  ¬ collinear ℝ {p, q, r}

-- Define Euclidean distance
def dist (p q : ℝ × ℝ) : ℝ := real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set D(S, r)
def D (S : finset (ℝ × ℝ)) (r : ℝ) : finset (finset (ℝ × ℝ)) :=
  S.powerset.filter (λ s, s.card = 2 ∧ ∃ x y ∈ s, x ≠ y ∧ dist x y = r)

theorem sum_D_r_squared_le_bound
  (S : finset (ℝ × ℝ))
  (hS : general_position S) :
  ∑ r in (finset.Ico 0 1).filter (λ x, 0 < x), (D S r).card ^ 2 ≤ 3 * S.card * (S.card - 1) / 4 :=
sorry

end sum_D_r_squared_le_bound_l370_370770


namespace cartesian_eq_of_C_distance_MN_l370_370315

noncomputable theory

-- Definitions for the given conditions
def parametric_eq_l (t : ℝ) : ℝ × ℝ :=
  (-1 + 3 / 5 * t, -1 + 4 / 5 * t)

def polar_eq_C (θ : ℝ) : ℝ :=
  sqrt 2 * Real.sin (θ + π / 4)

-- Prooving the Cartesian coordinate equation of curve C
theorem cartesian_eq_of_C {x y : ℝ} (h : x = sqrt 2 * Real.cos (π / 4) + sqrt 2 * Real.sin (π / 4)) :
  x^2 + y^2 - x - y = 0 :=
sorry

-- Prooving the distance between points M and N
theorem distance_MN (M N : ℝ × ℝ) (hM : M ∈ {x : ℝ × ℝ | x.1 ^ 2 + x.2 ^ 2 - x.1 - x.2 = 0})
  (hN : N ∈ {x : ℝ × ℝ | x.1 ^ 2 + x.2 ^ 2 - x.1 - x.2 = 0}) :
  dist M N = sqrt 41 / 5 :=
sorry

end cartesian_eq_of_C_distance_MN_l370_370315


namespace book_arrangement_l370_370729

def arrange_books (n_math n_history : ℕ) : ℕ :=
  let end_arrangements := n_math * (n_math - 1)
  let remaining_math_arrangements := fact (n_math - 2)
  let history_permutations := Nat.choose n_history 2 * Nat.choose (n_history - 2) 2 * Nat.choose (n_history - 4) 2
  let history_factorials := remaining_math_arrangements * (2! * 2! * 2!)
  end_arrangements * remaining_math_arrangements * history_permutations * history_factorials

theorem book_arrangement :
  arrange_books 4 6 = 17280 :=
by
  let end_arrangements := 4 * 3
  let remaining_math_arrangements := 2
  let history_permutations := Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2
  let history_factorials := remaining_math_arrangements * (fact 2 * fact 2 * fact 2)
  calc
    _ = end_arrangements * remaining_math_arrangements * history_permutations * history_factorials := rfl
    _ = 4 * 3 * 2 * (6! / (2! * 2! * 2!)) * (2! * 2! * 2!) := by simp
    _ = 17280 := by norm_num
  done

end book_arrangement_l370_370729


namespace exists_k_with_at_least_three_pairs_no_k_with_at_least_three_pairs_l370_370784

-- Define the sets S and T
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
def T : Set ℕ := {1, 2, 4, 7, 11, 16, 17}

-- The theorem statements
theorem exists_k_with_at_least_three_pairs (a : Set ℕ) (h : a ⊆ S ∧ a.card = 8) :
  ∃ k : ℕ, ∃ S' : Set (ℕ × ℕ), S ∘ S' = S ∧ S'.card ≥ 3 := sorry

theorem no_k_with_at_least_three_pairs (a : Set ℕ) (h : a = T) :
  ¬∃ k : ℕ, ∃ S' : Set (ℕ × ℕ), S ∘ S' = S ∧ S'.card ≥ 3 := sorry

end exists_k_with_at_least_three_pairs_no_k_with_at_least_three_pairs_l370_370784


namespace eq_x_minus_y_l370_370917

theorem eq_x_minus_y (x y : ℝ) : (x - y) * (x - y) = x^2 - 2 * x * y + y^2 :=
by
  sorry

end eq_x_minus_y_l370_370917


namespace magician_ratio_l370_370529

theorem magician_ratio (total_performances : ℕ) (no_reappearance_prob : ℚ) (total_reappearances : ℕ) (x : ℕ)
  (h1 : total_performances = 100)
  (h2 : no_reappearance_prob = 1 / 10)
  (h3 : total_reappearances = 110)
  (h4 : x = 20) :
  x / total_performances = 1 / 5 :=
by
  rw [h1, h4]
  norm_num
  sorry

end magician_ratio_l370_370529


namespace cookie_shop_problem_l370_370942

/-- 
A cookie shop sells 7 different cookies and 4 different types of milk. Gamma and Delta decide to 
purchase some items. Gamma, who is peculiar about choices, will not order more than 1 of the same type. 
Delta, on the other hand, will only order cookies, and is fine with having repeats. How many ways could 
they have left the store with 4 items collectively?
-/
theorem cookie_shop_problem :
  let cookies := 7
  let milk := 4
  -- Total number of unique items
  let total_items := cookies + milk
  -- Valid ways for Gamma and Delta to leave the store with 4 items collectively
  let valid_ways :=
    (choose total_items 4) +                     -- Gamma 4, Delta 0
    (choose total_items 3) * cookies +           -- Gamma 3, Delta 1
    (choose total_items 2) * (choose cookies 2 + cookies) * cookies + -- Gamma 2, Delta 2
    (total_items) * 
    (choose cookies 3 + cookies * (cookies - 1) + cookies) +  -- Gamma 1, Delta 3
    (choose cookies 4 + 
    cookies * (cookies - 1) + 
    choose cookies 2 + 
    cookies)  -- Gamma 0, Delta 4
  in valid_ways = 4096 :=
begin
  sorry
end

end cookie_shop_problem_l370_370942


namespace molecular_weight_chlorous_acid_l370_370889

def weight_H : ℝ := 1.01
def weight_Cl : ℝ := 35.45
def weight_O : ℝ := 16.00

def molecular_weight_HClO2 := (1 * weight_H) + (1 * weight_Cl) + (2 * weight_O)

theorem molecular_weight_chlorous_acid : molecular_weight_HClO2 = 68.46 := 
  by
    sorry

end molecular_weight_chlorous_acid_l370_370889


namespace shaded_area_square_l370_370827

theorem shaded_area_square (s : ℝ) (r : ℝ) (A : ℝ) :
  s = 4 ∧ r = 2 * Real.sqrt 2 → A = s^2 - 4 * (π * r^2 / 2) → A = 8 - 2 * π :=
by
  intros h₁ h₂
  sorry

end shaded_area_square_l370_370827


namespace problem1_problem2_l370_370494

noncomputable def arcSin (x : ℝ) : ℝ := Real.arcsin x

theorem problem1 :
  (S : ℝ) = 3 * Real.pi + 2 * Real.sqrt 2 - 6 * arcSin (Real.sqrt (2 / 3)) :=
by
  sorry

theorem problem2 :
  (S : ℝ) = 3 * arcSin (Real.sqrt (2 / 3)) - Real.sqrt 2 :=
by
  sorry

end problem1_problem2_l370_370494


namespace find_coefficients_l370_370921

theorem find_coefficients
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : ∀ x, f x = a * x^3 + b * x + c → f (-x) = -f x)
  (h3 : ∀ x, f' x = 3 * a * x^2 + b → f' 1 = -6)
  (h4 : ∀ x, f' x ≥ -12)
  : a = 2 ∧ b = -12 ∧ c = 0 := 
sorry

end find_coefficients_l370_370921


namespace Kolya_walking_speed_l370_370767

theorem Kolya_walking_speed
  (x : ℝ) 
  (h1 : x > 0) 
  (t_closing : ℝ := (3 * x) / 10) 
  (t_travel : ℝ := ((x / 10) + (x / 20))) 
  (remaining_time : ℝ := t_closing - t_travel)
  (walking_speed : ℝ := x / remaining_time)
  (correct_speed : ℝ := 20 / 3) :
  walking_speed = correct_speed := 
by 
  sorry

end Kolya_walking_speed_l370_370767


namespace smallest_value_of_ab_diff_l370_370873

def almost_equal (m n : ℕ) : Prop :=
  (m = n) ∨ (m + 1 = n) ∨ (m = n + 1)

def impossible_cut (a b : ℕ) : Prop :=
  ∀a' b', a' * b' ≠ a * b / 2 ∧ ¬almost_equal (a' * b') (a * b / 2)

theorem smallest_value_of_ab_diff 
  (a b : ℕ)
  (h1 : impossible_cut a b)
  (h2 : a ≠ b) :
  ∃ k : ℕ, k = 4 ∧ |a - b| = k :=
by
  sorry

end smallest_value_of_ab_diff_l370_370873


namespace sum_of_x_coordinates_where_g_eq_2_5_l370_370825

def g1 (x : ℝ) : ℝ := 3 * x + 6
def g2 (x : ℝ) : ℝ := -x + 2
def g3 (x : ℝ) : ℝ := 2 * x - 2
def g4 (x : ℝ) : ℝ := -2 * x + 8

def is_within (x : ℝ) (a b : ℝ) : Prop := a ≤ x ∧ x ≤ b

theorem sum_of_x_coordinates_where_g_eq_2_5 :
     (∀ x, g1 x = 2.5 → (is_within x (-4) (-2) → false)) ∧
     (∀ x, g2 x = 2.5 → (is_within x (-2) (0) → x = -0.5)) ∧
     (∀ x, g3 x = 2.5 → (is_within x 0 3 → x = 2.25)) ∧
     (∀ x, g4 x = 2.5 → (is_within x 3 5 → x = 2.75)) →
     (-0.5 + 2.25 + 2.75 = 4.5) :=
by { sorry }

end sum_of_x_coordinates_where_g_eq_2_5_l370_370825


namespace ordinate_is_residual_l370_370304

-- Assuming the definition of a residual plot in residual analysis
def ResidualPlot (ordinate: Type u) : Prop :=
  -- In a residual plot, the ordinate is the residual
  ∀ plot, ordinate = residual

theorem ordinate_is_residual {ordinate : Type u} :
  ResidualPlot ordinate → ordinate = residual :=
by
  intros h
  exact h sorry

end ordinate_is_residual_l370_370304


namespace pirate_treasure_l370_370452

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l370_370452


namespace speed_of_second_train_l370_370864

theorem speed_of_second_train
  (distance : ℝ)
  (speed_fast : ℝ)
  (time_difference : ℝ)
  (v : ℝ)
  (h_distance : distance = 425.80645161290323)
  (h_speed_fast : speed_fast = 75)
  (h_time_difference : time_difference = 4)
  (h_v : v = distance / (distance / speed_fast + time_difference)) :
  v = 44 := 
sorry

end speed_of_second_train_l370_370864


namespace mean_median_mode_equal_l370_370052

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def median (xs : List ℝ) : ℝ :=
  let sorted_xs := xs.insertionSort (≤)
  if sorted_xs.length % 2 = 1 then
    sorted_xs.get! (sorted_xs.length / 2)
  else
    (sorted_xs.get! (sorted_xs.length / 2 - 1) + sorted_xs.get! (sorted_xs.length / 2)) / 2

def mode (xs : List ℝ) : ℝ :=
  xs.groupBy id |>.map (λ l => (l.head!, l.length)) |>.maxBy (λ p => p.2) |>.1

theorem mean_median_mode_equal (x : ℝ) (h_mean : mean [70, 110, x, 40, 60, 210, 100, 50, x] = x)
                               (h_median : median [70, 110, x, 40, 60, 210, 100, 50, x] = x)
                               (h_mode : mode [70, 110, x, 40, 60, 210, 100, 50, x] = x) : x = 91 :=
by 
  sorry

end mean_median_mode_equal_l370_370052


namespace exists_special_subgrid_l370_370913

theorem exists_special_subgrid :
  ∀(grid : ℕ × ℕ → bool) (i j : ℕ), 
    ((∀ k, k = 0 ∨ k = 99 → (∀ l, grid (k, l) = tt)) ∧
    (∀ k, k = 0 ∨ k = 99 → (∀ l, grid (l, k) = tt)) ∧
    (∀ (a b : ℕ), a < 99 ∧ b < 99 → ¬ ((grid (a, b) = grid (a + 1, b) ∧ grid (a, b) = grid (a, b + 1) ∧ grid (a, b) = grid (a + 1, b + 1)) ∨ 
                                        (grid (a, b) ≠ grid (a + 1, b) ∧ grid (a, b) ≠ grid (a, b + 1) ∧ grid (a + 1, b) ≠ grid (a + 1, b + 1)))) →
    (0 ≤ i ∧ i < 99 ∧ 0 ≤ j ∧ j < 99 →
    ((grid (i, j) = grid (i + 1, j + 1) ∧ grid (i + 1, j) = grid (i, j + 1)) ∧
    (grid (i, j) ≠ grid (i + 1, j) ∧ grid (i + 1, j + 1) ≠ grid (i, j + 1)))) :=
by
  sorry

end exists_special_subgrid_l370_370913


namespace geom_seq_sum_five_terms_l370_370059

theorem geom_seq_sum_five_terms (a : ℕ → ℝ) (q : ℝ) 
    (h_pos : ∀ n, 0 < a n)
    (h_a2 : a 2 = 8) 
    (h_arith : 2 * a 4 - a 3 = a 3 - 4 * a 5) :
    a 1 * (1 - q^5) / (1 - q) = 31 :=
by
    sorry

end geom_seq_sum_five_terms_l370_370059


namespace pirates_treasure_l370_370418

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l370_370418


namespace exists_unique_m0_l370_370689

noncomputable def f : ℕ → ℤ → ℤ
| 0, x    := x
| (n+1), x := 2 * (f n x) + 1

theorem exists_unique_m0 (n ≥ 11) : ∃! (m_0 : ℤ), 0 ≤ m_0 ∧ m_0 ≤ 1991 ∧ 1993 ∣ f n m_0 :=
by
  sorry

end exists_unique_m0_l370_370689


namespace sum_reciprocal_squares_constant_l370_370341

-- Definitions that capture the conditions.
def F : Type := ℝ × ℝ -- Focal point F of the conic section.
def A (i : ℕ) : Type := ℝ × ℝ -- Points A_i on the conic section (i from 1 to n).
def FA (i : ℕ) [focal F] : ℝ := by sorry -- Distance from F to A_i.

-- Angles between consecutive points on the parabola.
def angle_eq (i j k : ℕ) : Prop := by sorry -- Angle equality condition: ∠A_i F A_j = ∠A_j F A_k.

-- Proof statement.
theorem sum_reciprocal_squares_constant (n : ℕ) (e : ℝ) (p : ℝ)
  (F : F) (A : ℕ → F)
  (h₀ : ∀ i, 1 ≤ i ∧ i ≤ n → anglex_equality_condition (A i) (F) (A (i+1)))
  (h₁ : ∀ i, 1 ≤ i ∧ i ≤ n → angle_eq i (i+1) (i+2)) :
  (∑ i in finset.range n, (1 / (FA i)^2)) = (n / (2 * e^2 * p^2)) * (2 + e^2) := by
  sorry

end sum_reciprocal_squares_constant_l370_370341


namespace brother_age_l370_370903

/-- A man is 12 years older than his brother. 
In two years, his age will be twice the age of his brother.
We need to prove that the present age of the brother is 10. --/
theorem brother_age :
  ∃ (B : ℕ), ∃ (M : ℕ), 
    (M = B + 12) ∧ 
    (M + 2 = 2 * (B + 2)) ∧ 
    (B = 10) :=
by
  -- Let B be the present age of the brother and M be the present age of the man
  use 10
  -- B = 10, then M = B + 12
  use 22
  -- Prove the conditions hold
  split
  { exact rfl }
  split
  { exact rfl }
  { exact rfl }

end brother_age_l370_370903


namespace fibonacci_last_digit_is_cyclic_l370_370182

-- Definitions based on the conditions
def fibonacci_seq (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fibonacci_seq (n - 1) + fibonacci_seq (n - 2)

def last_digit_sequence (n : ℕ) : ℕ :=
  fibonacci_seq n % 10

-- The statement to be proven
theorem fibonacci_last_digit_is_cyclic : ∃ p ∈ ℕ, ∀ n ∈ ℕ, last_digit_sequence (n + p) = last_digit_sequence n :=
by
  sorry

end fibonacci_last_digit_is_cyclic_l370_370182


namespace new_price_of_lamp_l370_370134

/-
  Given:
  - original_price: ℝ representing the original price of the lamp ($120)
  - discount1: ℝ representing the first discount rate (20%)
  - discount2: ℝ representing the second discount rate (15%)
  Prove that the new price after applying both discounts is $81.60
-/
noncomputable def original_price : ℝ := 120
noncomputable def discount1 : ℝ := 0.20
noncomputable def discount2 : ℝ := 0.15

theorem new_price_of_lamp (price : ℝ) (d1 d2 : ℝ) (h1 : price = original_price) (h2 : d1 = discount1) (h3 : d2 = discount2) : 
  let first_discount := price * d1 in
  let new_price1 := price - first_discount in
  let second_discount := new_price1 * d2 in
  let final_price := new_price1 - second_discount in
  final_price = 81.60 :=
by  
  -- Calculations and proof would go here
  sorry

end new_price_of_lamp_l370_370134


namespace carpet_covering_l370_370014

-- Definitions as per the conditions
variable (L : ℝ) -- Length of the corridor
variable I : list (ℝ × ℝ) -- List of carpet pieces represented as intervals

-- Main theorem statement
theorem carpet_covering {L : ℝ} (I : list (ℝ × ℝ)) 
    (h₁ : ∀ t, ∃ J ∈ I, J.1 ≤ t ∧ t ≤ J.2) :
    ∃ J ⊆ I, (∀ t, ∃ K ∈ J, K.1 ≤ t ∧ t ≤ K.2) ∧ (J.foldr (λ j acc, (j.2 - j.1) + acc) 0 < 2 * L) := by 
  sorry

end carpet_covering_l370_370014


namespace quadratic_solution_set_l370_370666

theorem quadratic_solution_set (a b c : ℝ) 
  (h : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ x < -2 ∨ x > 3) :
  (a > 0) ∧ 
  (∀ x : ℝ, bx + c > 0 ↔ x < 6) = false ∧ 
  (a + b + c < 0) ∧
  (∀ x : ℝ, cx^2 - bx + a < 0 ↔ x < -1 / 3 ∨ x > 1 / 2) :=
sorry

end quadratic_solution_set_l370_370666


namespace tangent_line_at_1_l370_370831

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_1 :
  ∀ (x y : ℝ), (x, y) = (1, 0) → (x - y - 1 = 0) :=
begin
  intros x y h,
  rw h,
  simp,
end

end tangent_line_at_1_l370_370831


namespace coefficient_x3_in_expansion_l370_370373

theorem coefficient_x3_in_expansion :
  let x := Polynomial.C
  let expr := (x^2 - x + 1)^10 in
  coeff expr 3 = -210 :=
by
  sorry

end coefficient_x3_in_expansion_l370_370373


namespace simplify_fraction_l370_370015

theorem simplify_fraction : (180 / 270 : ℚ) = 2 / 3 := by
  sorry

end simplify_fraction_l370_370015


namespace pirate_treasure_l370_370446

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l370_370446


namespace pirates_treasure_l370_370423

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l370_370423


namespace distance_to_x_axis_l370_370828

theorem distance_to_x_axis (P : ℝ × ℝ) (h : P = (-3, -2)) : |P.2| = 2 := 
by sorry

end distance_to_x_axis_l370_370828


namespace arccos_neg_one_eq_pi_l370_370176

theorem arccos_neg_one_eq_pi :
  ∃ θ ∈ set.Icc (0 : ℝ) real.pi, real.cos θ = -1 :=
begin
  use real.pi,
  split,
  { exact le_refl 0 },
  { exact le_of_eq rfl },
  { exact real.cos_pi }
end

end arccos_neg_one_eq_pi_l370_370176


namespace max_a3_in_arith_geo_sequences_l370_370232

theorem max_a3_in_arith_geo_sequences
  (a1 a2 a3 : ℝ) (b1 b2 b3 : ℝ)
  (h1 : a1 + a2 + a3 = 15)
  (h2 : a2 = ((a1 + a3) / 2))
  (h3 : b1 * b2 * b3 = 27)
  (h4 : (a1 + b1) * (a3 + b3) = (a2 + b2) ^ 2)
  (h5 : a1 + b1 > 0)
  (h6 : a2 + b2 > 0)
  (h7 : a3 + b3 > 0) :
  a3 ≤ 59 := sorry

end max_a3_in_arith_geo_sequences_l370_370232


namespace price_of_adult_ticket_eq_32_l370_370501

theorem price_of_adult_ticket_eq_32 
  (num_adults : ℕ)
  (num_children : ℕ)
  (price_child_ticket : ℕ)
  (price_adult_ticket : ℕ)
  (total_collected : ℕ)
  (h1 : num_adults = 400)
  (h2 : num_children = 200)
  (h3 : price_adult_ticket = 2 * price_child_ticket)
  (h4 : total_collected = 16000)
  (h5 : total_collected = num_adults * price_adult_ticket + num_children * price_child_ticket)
  : price_adult_ticket = 32 := 
by
  sorry

end price_of_adult_ticket_eq_32_l370_370501


namespace length_of_bridge_l370_370842

theorem length_of_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_s : ℝ)
  (h_train_length : train_length = 110)
  (h_train_speed_kmph : train_speed_kmph = 45)
  (h_crossing_time_s : crossing_time_s = 30) :
  let speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * crossing_time_s
  let bridge_length := total_distance - train_length
  in bridge_length = 265 :=
by
  sorry

end length_of_bridge_l370_370842


namespace transfer_equation_correct_l370_370454

theorem transfer_equation_correct (x : ℕ) :
  46 + x = 3 * (30 - x) := 
sorry

end transfer_equation_correct_l370_370454


namespace num_correct_conditions_l370_370672

-- Define the conditions as premises

def cond1 : Prop := 
  ∀ (R2 : ℝ), (R2 ≥ 0 ∧ R2 ≤ 1) → (∀ (model1 model2 : α), (R2 model1 > R2 model2) → (fitting_effect_model1 > fitting_effect_model2))

def cond2 : Prop := 
  ∃ (X : discrete_random_variable), true

def cond3 : Prop :=
  ∀ (X : random_variable), (variance(X) ≥ 0 ∧ std_dev(X) ≥ 0) → (variance(X) = 0 ↔ ∀ x, X x = mean(X))

def cond4 : Prop :=
  ∀ (A B : event_space), (independent_event (at_least_one_hits A B) (neither_hits A B))

-- The main theorem to prove the number of correct conditions
theorem num_correct_conditions : cond1 ∧ cond2 ∧ cond3 ∧ cond4 → 2 = 2 := 
by sorry

end num_correct_conditions_l370_370672


namespace toys_total_l370_370989

noncomputable def toys : ℕ := 651

variable (a m t s : ℕ)
variable h1 : a = (3.5)^2 * m
variable h2 : a = 0.60 * t
variable h3 : m = 12
variable h4 : s = t + 2

theorem toys_total : a + m + t + s = toys := 
by
  sorry

end toys_total_l370_370989


namespace all_primes_in_A_l370_370323

theorem all_primes_in_A 
  (P : Set ℕ) (prime : ℕ → Prop) 
  (is_prime : ∀ p, prime p ↔ p ∈ P)
  (A : Set ℕ) (A_subset_P : A ⊆ P) (A_card : 2 ≤ A.card)
  (cond : ∀ n ∈ ℕ, ∀ p1 p2 ... pn ∈ A, (∀ p ∈ nat.factors (p1 * p2 * ... * pn - 1), p ∈ A)) 
  : A = P := 
sorry

end all_primes_in_A_l370_370323


namespace seven_solutions_l370_370281

theorem seven_solutions: ∃ (pairs : List (ℕ × ℕ)), 
  (∀ (x y : ℕ), (x < y) → ((1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 2007) ↔ (x, y) ∈ pairs) 
  ∧ pairs.length = 7 :=
sorry

end seven_solutions_l370_370281


namespace committees_count_l370_370171

-- Define the conditions
structure Department :=
  (men : Nat)
  (women : Nat)

def mathematics : Department := { men := 3, women := 3 }
def statistics : Department := { men := 3, women := 3 }
def computer_science : Department := { men := 3, women := 3 }

-- Define the function to count combinations
noncomputable def combinations (n k : Nat) : Nat :=
  -- Continue using the binomial coefficient definition
  Nat.choose n k

-- Function to count valid committee formations
noncomputable def count_committees (d1 d2 d3 : Department) : Nat :=
  let count_department (d : Department) : Nat :=
    combinations d.men 1 * combinations d.women 1
  in count_department d1 * count_department d2 * count_department d3

-- Prove the final count of committees
theorem committees_count :
  count_committees mathematics statistics computer_science = 729 :=
by
  unfold count_committees combinations
  -- Manually substitute the known values
  have h1 : combinations 3 1 = 3 := Nat.choose_zero_succ 2
  have h2 : combinations 3 1 = 3 := Nat.choose_zero_succ 2
  rw [h1, h2] -- Apply substitution
  calc
    3 * 3 * 3 * 3 * 3 * 3 = 9 * 9 * 9 := by ring
    ... = 729 := by norm_num
 
 -- Provide a placeholder for the proof as requested
sorry

end committees_count_l370_370171


namespace race_outcomes_l370_370977

-- Definition of participants
inductive Participant
| Abe 
| Bobby
| Charles
| Devin
| Edwin
| Frank
deriving DecidableEq

open Participant

def num_participants : ℕ := 6

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Proving the number of different 1st-2nd-3rd outcomes
theorem race_outcomes : factorial 6 / factorial 3 = 120 := by
  sorry

end race_outcomes_l370_370977


namespace star_value_l370_370191

def star (x y : ℕ) : ℕ :=
  x * y - 3 * x + 1

theorem star_value : (star 5 3) - (star 3 5) = -6 := by
  unfold star
  calc
    star 5 3    = 5 * 3 - 3 * 5 + 1                   : by unfold star
            ... = 15 - 15 + 1                          : by rfl
            ... = 1                                    : by rfl
    star 3 5    = 3 * 5 - 3 * 3 + 1                   : by unfold star
            ... = 15 - 9 + 1                           : by rfl
            ... = 7                                    : by rfl
    (star 5 3) - (star 3 5)
            ... = 1 - 7                                : by rfl
            ... = -6                                   : by rfl

end star_value_l370_370191


namespace right_triangle_perimeter_l370_370962

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) (perimeter : ℝ)
  (h1 : area = 180) 
  (h2 : leg1 = 30) 
  (h3 : (1 / 2) * leg1 * leg2 = area)
  (h4 : hypotenuse^2 = leg1^2 + leg2^2)
  (h5 : leg2 = 12) 
  (h6 : hypotenuse = 2 * Real.sqrt 261) :
  perimeter = 42 + 2 * Real.sqrt 261 :=
by
  sorry

end right_triangle_perimeter_l370_370962


namespace distance_P4P5_l370_370326

noncomputable def P_point (α : Type*) [normed_add_torsor A E] :=
{P1 P2 P3 P4 P5 : α}

// Non-coplanar condition 
variables (α : Type*) [metric_space α] [normed_add_group α] [normed_space ℝ α]

open affine

-- Given conditions
variables {ell m : set α}
variables {P1 P2 P3 P4 P5 : α}

-- Points on lines and closest point conditions
variables (P1_on_ell : P1 ∈ ell)
variables (P2_on_m_closest_to_P1 : is_closest_point ℝ m P1 P2)
variables (P3_on_ell_closest_to_P2 : is_closest_point ℝ ell P2 P3)
variables (P4_on_m_closest_to_P3 : is_closest_point ℝ m P3 P4)
variables (P5_on_ell_closest_to_P4 : is_closest_point ℝ ell P4 P5)

-- Given distances
variables (d_P1P2 : dist P1 P2 = 5)
variables (d_P2P3 : dist P2 P3 = 3)
variables (d_P3P4 : dist P3 P4 = 2)

-- Goal
theorem distance_P4P5 : dist P4 P5 = real.sqrt (39) / 4 :=
sorry

end distance_P4P5_l370_370326


namespace more_polygons_of_type2_l370_370000

/-
  Given 15 white points and 1 black point on a circle:
  - Type 1 polygons consist only of white points.
  - Type 2 polygons include the black point and white points.

  Prove that the number of Type 2 polygons is 105 more than the number of Type 1 polygons.
-/
theorem more_polygons_of_type2 (w b: ℕ) (hw: w = 15) (hb: b = 1):
  let type1 := (2^15) - (1+15+105),
      type2 := (2^15) - (1 + 15) in
  type2 - type1 = 105 :=
by
  intros
  simp only [hw, hb]
  sorry

end more_polygons_of_type2_l370_370000


namespace total_slices_l370_370797

theorem total_slices (pizzas : ℕ) (slices1 slices2 slices3 slices4 : ℕ)
  (h1 : pizzas = 4)
  (h2 : slices1 = 8)
  (h3 : slices2 = 8)
  (h4 : slices3 = 10)
  (h5 : slices4 = 12) :
  slices1 + slices2 + slices3 + slices4 = 38 := by
  sorry

end total_slices_l370_370797


namespace Blair_17th_turn_l370_370318

/-
  Jo begins counting by saying "5". Blair then continues the sequence, each time saying a number that is 2 more than the last number Jo said. Jo increments by 1 each turn after Blair. They alternate turns.
  Prove that Blair says the number 55 on her 17th turn.
-/

def Jo_initial := 5
def increment_Jo := 1
def increment_Blair := 2

noncomputable def blair_sequence (n : ℕ) : ℕ :=
  Jo_initial + increment_Blair + (n - 1) * (increment_Jo + increment_Blair)

theorem Blair_17th_turn : blair_sequence 17 = 55 := by
    sorry

end Blair_17th_turn_l370_370318


namespace treasures_coins_count_l370_370409

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l370_370409


namespace length_DE_l370_370751

-- Definitions for the conditions
variables {A B C D E : Type} -- points
variables (BC : ℝ) (angleC : ℝ)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]

-- Assign the given values to the conditions
def BC_val := 20 * real.sqrt 2
def angleC_val := 45

-- Assuming midpoint condition
def is_midpoint (D : E) (B : E) (C : E) : Prop :=
  dist B D = dist D C ∧ dist B C = 2 * dist B D

-- Assuming perpendicular bisector intersects AC at E
def perp_bisector_intersect (B : E) (C : E) (A : E) (D : E) (E : E) : Prop :=
  dist D B = dist D C ∧ collinear B C D ∧ perp (B - C) (A - E)

-- Proof goal: to show that DE = 10√2 given the conditions
theorem length_DE (h1 : BC = BC_val) (h2 : angleC = angleC_val)
                 (h3 : is_midpoint D B C) (h4 : perp_bisector_intersect B C A D E) :
  dist D E = 10 * real.sqrt 2 :=
sorry

end length_DE_l370_370751


namespace harmonic_quadruple_l370_370172

-- Define the setup of the problem

variables {M N O A B C A1 B1 C1 : Type}

-- Definitions for the problem
def is_diameter (O : Type) (M N : Type) : Prop := 
  -- M and N are endpoints of the diameter of circle O
  sorry

def are_tangents (O : Type) (MK NL : Type) (M N: Type) : Prop := 
  -- MK and NL are tangents to the circle O at points M and N
  sorry

def equal_segments (AB BC: Type) : Prop := 
  -- AB and BC are equal segments
  sorry

def tangent_intersection (O : Type) (A B C A1 B1 C1: Type) (NL : Type) : Prop := 
  -- Tangents from points A, B, C to circle O intersect tangent NL at points A1, B1, C1
  sorry

-- Main theorem statement
theorem harmonic_quadruple (O : Type) (M N A B C A1 B1 C1 : Type) 
  (mk: Type) (nl: Type)
  (h1 : is_diameter O M N)
  (h2 : are_tangents O mk nl M N)
  (h3: equal_segments A B)
  (h4 : equal_segments B C)
  (h5 : tangent_intersection O A B C A1 B1 C1 nl) :
  harmonic N C1 B1 A1 :=
sorry

end harmonic_quadruple_l370_370172


namespace plane_divided_into_3_sets_plane_divided_into_9_sets_l370_370055

/-- It is impossible to divide the plane into 3 disjoint sets such that no two points in the same set are 1 unit apart. -/
theorem plane_divided_into_3_sets (P : Type) [metric_space P] [inhabited P] :
  (∃ (S₁ S₂ S₃ : set P), (∀ p1 p2 ∈ S₁, dist p1 p2 ≠ 1) ∧
                          (∀ p1 p2 ∈ S₂, dist p1 p2 ≠ 1) ∧
                          (∀ p1 p2 ∈ S₃, dist p1 p2 ≠ 1)) → false :=
sorry

/-- It is possible to divide the plane into 9 disjoint sets such that no two points in the same set are 1 unit apart. -/
theorem plane_divided_into_9_sets (P : Type) [metric_space P] [inhabited P] :
  ∃ (S₁ S₂ S₃ S₄ S₅ S₆ S₇ S₈ S₉ : set P), (∀ p1 p2 ∈ S₁, dist p1 p2 ≠ 1) ∧
                                         (∀ p1 p2 ∈ S₂, dist p1 p2 ≠ 1) ∧
                                         (∀ p1 p2 ∈ S₃, dist p1 p2 ≠ 1) ∧
                                         (∀ p1 p2 ∈ S₄, dist p1 p2 ≠ 1) ∧
                                         (∀ p1 p2 ∈ S₅, dist p1 p2 ≠ 1) ∧
                                         (∀ p1 p2 ∈ S₆, dist p1 p2 ≠ 1) ∧
                                         (∀ p1 p2 ∈ S₇, dist p1 p2 ≠ 1) ∧
                                         (∀ p1 p2 ∈ S₈, dist p1 p2 ≠ 1) ∧
                                         (∀ p1 p2 ∈ S₉, dist p1 p2 ≠ 1) :=
sorry


end plane_divided_into_3_sets_plane_divided_into_9_sets_l370_370055


namespace find_sample_size_l370_370453

def sample_size (sample : List ℕ) : ℕ :=
  sample.length

theorem find_sample_size :
  sample_size (List.replicate 500 0) = 500 :=
by
  sorry

end find_sample_size_l370_370453


namespace not_all_distinct_l370_370965

open Function

theorem not_all_distinct (a : ℕ → ℚ) (h : ∀ m n : ℕ, a m + a n = a (m * n)) : ∃ i j, i ≠ j ∧ a i = a j :=
by
  sorry

end not_all_distinct_l370_370965


namespace treasure_coins_l370_370439

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l370_370439


namespace find_a_plus_b_l370_370231

-- Define the function and conditions given in the problem
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(-x) = f(x)

variable {a b : ℝ}
def f (x : ℝ) : ℝ := a * x^2 + b * x

-- The mathematically equivalent proof problem to be stated in Lean 4
theorem find_a_plus_b (h_even : is_even_function f) (h_domain : ∀ x, x ∈ Set.Icc (a - 1) (2 * a)) :
    a + b = 1/3 :=
sorry

end find_a_plus_b_l370_370231


namespace square_perimeter_eq_16_l370_370371

theorem square_perimeter_eq_16 (s : ℕ) (h : s^2 = 4 * s) : 4 * s = 16 :=
by {
  sorry
}

end square_perimeter_eq_16_l370_370371


namespace gwen_total_books_l370_370698

def mystery_shelves : Nat := 6
def mystery_books_per_shelf : Nat := 7

def picture_shelves : Nat := 4
def picture_books_per_shelf : Nat := 5

def biography_shelves : Nat := 3
def biography_books_per_shelf : Nat := 3

def scifi_shelves : Nat := 2
def scifi_books_per_shelf : Nat := 9

theorem gwen_total_books :
    (mystery_books_per_shelf * mystery_shelves) +
    (picture_books_per_shelf * picture_shelves) +
    (biography_books_per_shelf * biography_shelves) +
    (scifi_books_per_shelf * scifi_shelves) = 89 := 
by 
    sorry

end gwen_total_books_l370_370698


namespace abs_diff_of_solutions_l370_370803

theorem abs_diff_of_solutions :
  ∀ a b : ℝ, (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ (sqrt(e), y1) = (sqrt(e), a) ∧ (sqrt(e), y2) = (sqrt(e), b) ∧
      ((y1^2 + (sqrt e)^4 = 2*(sqrt e)^2*y1 + 1) ∧ (y2^2 + (sqrt e)^4 = 2*(sqrt e)^2*y2 + 1))) →
  |a - b| = 2 :=
begin
  intros a b h,
  obtain ⟨y1, y2, h1, h2, h3, hy1_eq, hy2_eq⟩ := h,
  have h_eq : (y1^2 + (sqrt e)^4 = 2*(sqrt e)^2*y1 + 1) ∧ (y2^2 + (sqrt e)^4 = 2*(sqrt e)^2*y2 + 1),
  from hy1_eq,
  rw (sqrt(e)) at h_eq,
  sorry -- skip the proof steps
end

end abs_diff_of_solutions_l370_370803


namespace cubical_block_weight_l370_370944

-- Given conditions
variables (s : ℝ) (volume_ratio : ℝ) (weight2 : ℝ)
variable (h : volume_ratio = 8)
variable (h_weight : weight2 = 40)

-- The problem statement
theorem cubical_block_weight (weight1 : ℝ) :
  volume_ratio * weight1 = weight2 → weight1 = 5 :=
by
  -- Assume volume ratio as 8, weight of the second cube as 40 pounds
  have h1 : volume_ratio = 8 := h
  have h2 : weight2 = 40 := h_weight
  -- sorry is here to indicate we are skipping the proof
  sorry

end cubical_block_weight_l370_370944


namespace circumcircle_pqr_passes_midpoint_m_l370_370300
-- Import the general Mathlib library

-- Define the problem in Lean
theorem circumcircle_pqr_passes_midpoint_m
  (ABC : Triangle)
  (h_acute : IsAcute ABC)
  (D E F : Point)
  (hA : IsAltitudeFrom D ABC.a A)
  (hB : IsAltitudeFrom E ABC.b B)
  (hC : IsAltitudeFrom F ABC.c C)
  (M : Point)
  (hM : M = midpoint B C)
  (Q R : Point)
  (hQ : LineThrough D ∥ EF ∧ LineThrough D ∩ AC = Q)
  (hR : LineThrough D ∥ EF ∧ LineThrough D ∩ AB = R)
  (P : Point)
  (hP : EF ∩ BC = P)
  : OnCircle (circumcircle P Q R) M := sorry

end circumcircle_pqr_passes_midpoint_m_l370_370300


namespace lcm_12_15_18_l370_370884

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end lcm_12_15_18_l370_370884


namespace F_transformed_l370_370458

-- Define the coordinates of point F
def F : ℝ × ℝ := (1, 0)

-- Reflection over the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Reflection over the y-axis
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

-- Reflection over the line y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

-- Point F after all transformations
def F_final : ℝ × ℝ :=
  reflect_y_eq_x (reflect_y (reflect_x F))

-- Statement to prove
theorem F_transformed : F_final = (0, -1) :=
  sorry

end F_transformed_l370_370458


namespace octahedron_faces_incorrect_l370_370897

theorem octahedron_faces_incorrect : 
    ( ∀ (o : Octahedron), num_faces o = 8 )
    ∧ ( ∀ (t : Tetrahedron), ∃ (p1 p2 p3 p4 : Pyramid), t_is_cuts_into_4_pyramids t p1 p2 p3 p4 )
    ∧ ( ∀ (f : Frustum), extends_lateral_edges_intersect_at_point f )
    ∧ ( ∀ (r : Rectangle), rotated_around_side_forms_cylinder r ) 
    → ( "An octahedron has 10 faces" is incorrect ) :=
sorry

end octahedron_faces_incorrect_l370_370897


namespace find_total_coins_l370_370436

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l370_370436


namespace relationship_between_m_and_r_l370_370650

def sequence (a : ℕ → ℝ) (m r : ℝ) : Prop :=
  a 1 = m ∧
  (∀ k : ℕ, a (2 * k + 1 + 1) = 2 * a (2 * k + 1)) ∧
  (∀ k : ℕ, a (2 * k + 2 + 1) = a (2 * k + 2) + r) ∧
  (∀ n : ℕ, a (n + 2) = a n)

theorem relationship_between_m_and_r (a : ℕ → ℝ) (m r : ℝ) :
  sequence a m r → m + r = 0 :=
sorry

end relationship_between_m_and_r_l370_370650


namespace length_of_train_l370_370974

-- Define the given conditions
def speed_km_per_hr : ℝ := 65        -- Speed of the train in km/hr
def time_seconds : ℝ := 13.568145317605362  -- Time taken to cross the bridge in seconds
def bridge_length_m : ℝ := 145      -- Length of the bridge in meters

-- Define the conversion factor from km/hr to m/s
def kmph_to_mps (speed: ℝ) : ℝ := speed * (1000/3600)

-- Calculate the speed in m/s using the conversion factor
def speed_m_per_s : ℝ := kmph_to_mps speed_km_per_hr

-- Calculate the total distance the train travels while crossing the bridge
def total_distance_m : ℝ := speed_m_per_s * time_seconds

-- Define the theorem that represents the length of the train
theorem length_of_train : total_distance_m - bridge_length_m = 100 :=
by
  sorry

end length_of_train_l370_370974


namespace ellipse_standard_equation_l370_370263

noncomputable def standard_equation_of_ellipse (a b : ℝ) : Prop :=
  (a^2) = 13 ∧ 
  (b^2) = 4 ∧ 
  (c = 3 ∧ ( ∃ x y, (x - 0)² + y² = 4 → ( ∃ p q, (p + 3)² + q² = 4)) 

theorem ellipse_standard_equation :
  standard_equation_of_ellipse 3 2 = (\(x^2) / 13) + (\(y^2) / 4) := sorry

end ellipse_standard_equation_l370_370263


namespace stratified_sampling_third_year_students_l370_370936

theorem stratified_sampling_third_year_students
    (total_students : ℕ)
    (first_year_students : ℕ)
    (second_year_students : ℕ)
    (third_year_students : ℕ)
    (p_second_year : ℚ)
    (total_students = 2000)
    (first_year_students = 750)
    (p_second_year = 0.3) :
    second_year_students = (p_second_year * total_students).toNat →
    third_year_students = total_students - first_year_students - second_year_students →
    let sample_size := 40 in
    let proportion := (sample_size / total_students : ℚ) in
    let sampled_third_year_students := (proportion * third_year_students) in
    sampled_third_year_students.toNat = 13 :=
by
  sorry

end stratified_sampling_third_year_students_l370_370936


namespace courtyard_length_l370_370455

-- Definitions for the given conditions
def width_of_courtyard : ℝ := 16.5
def number_of_paving_stones : ℕ := 231
def area_of_each_paving_stone : ℝ := 2.5 * 2

-- Define the total covered area by paving stones
def total_area_covered_by_stones : ℝ := number_of_paving_stones * area_of_each_paving_stone

-- The length of the courtyard calculated from total_area_covered_by_stones and width_of_courtyard
def length_of_courtyard : ℝ := total_area_covered_by_stones / width_of_courtyard

-- The theorem to prove
theorem courtyard_length :
  length_of_courtyard = 70 := by
  sorry

end courtyard_length_l370_370455


namespace find_total_coins_l370_370434

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l370_370434


namespace incenter_QJ_l370_370459

noncomputable def P : Type := ℝ
def Q : Type := ℝ
def R : Type := ℝ

def PQ : ℝ := 30
def PR : ℝ := 29
def QR : ℝ := 31

def incenter (P Q R : Type) : Type := sorry
def inradius (P Q R : Type) : ℝ := sorry
def QJ (P Q R : Type) : ℝ := sorry

theorem incenter_QJ (J : Type) (r : ℝ) :
  J = incenter P Q R →
  r = inradius P Q R →
  QJ P Q R = sqrt (226 - r^2) :=
by
  -- Assuming that J is the incenter and r is the inradius
  intros hJ hr
  -- Using the given conditions and properties of triangle incenter and inradius
  sorry

end incenter_QJ_l370_370459


namespace speed_of_stream_l370_370105

theorem speed_of_stream :
  ∃ (v : ℝ), (∀ (swim_speed : ℝ), swim_speed = 1.5 → 
    (∀ (time_upstream : ℝ) (time_downstream : ℝ), 
      time_upstream = 2 * time_downstream → 
      (1.5 + v) / (1.5 - v) = 2)) → v = 0.5 :=
sorry

end speed_of_stream_l370_370105


namespace pentagon_planar_l370_370005

variable (A : Type*) [AffineSpace ℝ A]

structure Pentagon (A_1 A_2 A_3 A_4 A_5 : A) :=
(equal_sides : ∀ (i j : Fin 5), dist (points.fin (i + 1)) (points.fin (j + 1)) = dist (points.fin 0) (points.fin 1))
(equal_angles : ∀ (i j k l : Fin 5), angle (points.fin (i + 1)) (points.fin 0) (points.fin (j + 1)) 
                                     = angle (points.fin (k + 1)) (points.fin 0) (points.fin (l + 1)))

theorem pentagon_planar {A_1 A_2 A_3 A_4 A_5 : A} 
  (p : Pentagon A A_1 A_2 A_3 A_4 A_5) : 
  ∃ (P : AffineSubspace ℝ A), A_1 ∈ P ∧ A_2 ∈ P ∧ A_3 ∈ P ∧ A_4 ∈ P ∧ A_5 ∈ P :=
sorry

end pentagon_planar_l370_370005


namespace max_colors_in_colored_table_l370_370393

theorem max_colors_in_colored_table (n : ℕ) :
  let table_size := 2^n
  ∀ (color : Fin table_size → Fin table_size → ℕ),
    (∀ (i j : Fin table_size),
      color i j = color j ((i + j) % table_size)) →
    ∃ (C : ℕ), C = 2^n :=
by
  intros table_size color h
  use 2^n
  sorry

end max_colors_in_colored_table_l370_370393


namespace find_diagonal_length_l370_370530

-- Declaring the conditions as given variables.
variables (A B C D M K P : Point) (AB AC AD : Length)

-- Stating the conditions.
def condition_1 := ∃ (r : Ratio), r = 1 / 3
def condition_2 := ∃ (d : Distance), d = 2
def condition_3 := ∃ (ls : Length), ls = 4

-- Defining the main goal.
theorem find_diagonal_length 
  (h1 : condition_1)
  (h2 : condition_2)
  (h3 : condition_3) : 
  ∃ (diagonal : Length), diagonal = 8 := 
sorry

end find_diagonal_length_l370_370530


namespace calculate_fraction_l370_370468

theorem calculate_fraction :
  (2019 + 1981)^2 / 121 = 132231 := 
  sorry

end calculate_fraction_l370_370468


namespace number_of_positive_integers_l370_370629

theorem number_of_positive_integers (x : ℕ) (h : 200 ≤ x^2 ∧ x^2 ≤ 400) :
  {x : ℕ | 200 ≤ x^2 ∧ x^2 ≤ 400}.to_finset.card = 6 :=
sorry

end number_of_positive_integers_l370_370629


namespace range_of_m_l370_370275

noncomputable def f (x m : ℝ) := x^2 - 2 * m * x + 4

def P (m : ℝ) : Prop := ∀ x, 2 ≤ x → f x m ≥ f (2 : ℝ) m
def Q (m : ℝ) : Prop := ∀ x, 4 * x^2 + 4 * (m - 2) * x + 1 > 0

theorem range_of_m (m : ℝ) : (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ m ≤ 1 ∨ (2 < m ∧ m < 3) := sorry

end range_of_m_l370_370275


namespace age_difference_is_12_l370_370484

noncomputable def age_difference (x : ℕ) : ℕ :=
  let older := 3 * x
  let younger := 2 * x
  older - younger

theorem age_difference_is_12 :
  ∃ x : ℕ, 3 * x + 2 * x = 60 ∧ age_difference x = 12 :=
by
  sorry

end age_difference_is_12_l370_370484


namespace projections_collinear_iff_point_on_circumcircle_l370_370788

theorem projections_collinear_iff_point_on_circumcircle 
  (A B C : Point) 
  (Γ : Circle) 
  (hΓ : Circumcircle Γ A B C)
  (D : Point)
  (A' B' C' : Point)
  (hA' : OrthogonalProjection D (line_through B C) A')
  (hB' : OrthogonalProjection D (line_through A C) B')
  (hC' : OrthogonalProjection D (line_through A B) C') :
  Collinear A' B' C' ↔ D ∈ Γ := 
sorry

end projections_collinear_iff_point_on_circumcircle_l370_370788


namespace rodney_prob_correct_guess_l370_370806

def isTwoDigitInteger (n : ℕ) : Prop := n >= 10 ∧ n < 100

def tensDigitIsOdd (n : ℕ) : Prop := (n / 10) % 2 = 1

def unitsDigitIsEven (n : ℕ) : Prop := (n % 10) % 2 = 0

def numberGreaterThan75 (n : ℕ) : Prop := n > 75

def validNumber (n : ℕ) : Prop :=
  isTwoDigitInteger n ∧ tensDigitIsOdd n ∧ unitsDigitIsEven n ∧ numberGreaterThan75 n

def validNumbers := {n : ℕ | validNumber n}

def probabilityCorrectGuess : Rat := 1 / Rat.ofInt (Set.toFinset validNumbers).card

theorem rodney_prob_correct_guess : probabilityCorrectGuess = 1 / 7 :=
by
  -- this proof has been skipped
  sorry

end rodney_prob_correct_guess_l370_370806


namespace distance_ratio_l370_370758

-- Define the known distances
def distance_1_to_2 : ℝ := 6
def distance_2_to_3 : ℝ := 10  -- This comes from the subtraction (24 - 14)
def distance_house_to_1 : ℝ := 4
def distance_3_to_work : ℝ := 4
def total_commute : ℝ := 24

-- The ratio to prove
def ratio_distance (a b : ℝ) : ℝ := a / b

-- The main statement
theorem distance_ratio : ratio_distance distance_2_to_3 distance_1_to_2 = 5 / 3 := by
    sorry

end distance_ratio_l370_370758


namespace find_m_n_sum_l370_370234

theorem find_m_n_sum
  (sec_x_tan_x_eq : ∀ x, sec x + tan x = 22 / 7)
  (csc_x_cot_x_eq : ∀ x, ∃ (m n : ℕ), csc x + cot x = m / n ∧ Nat.coprime m n) :
  ∃ x (m n : ℕ), sec x + tan x = 22 / 7 ∧ csc x + cot x = m / n ∧ Nat.coprime m n ∧ m + n = 44 :=
by
  sorry

end find_m_n_sum_l370_370234


namespace math_proof_problem_l370_370360

noncomputable def proof_problem (a b c d : ℝ) :=
  |a| > 1 ∧ |b| > 1 ∧ |c| > 1 ∧ |d| > 1 ∧ ab * (c + d) + cd * (a + b) + a + b + c + d = 0 →
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0

-- A top-level theorem statement proving the problem
theorem math_proof_problem (a b c d : ℝ) (h1 : |a| > 1) (h2 : |b| > 1) (h3 : |c| > 1) (h4 : |d| > 1)
  (h5 : ab * (c + d) + cd * (a + b) + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 :=
sorry

end math_proof_problem_l370_370360


namespace max_digits_in_product_l370_370463

theorem max_digits_in_product (a b : ℕ) (ha : 10000 ≤ a ∧ a ≤ 99999) (hb : 1000 ≤ b ∧ b ≤ 9999) :
  ∃ prod : ℕ, prod = a * b ∧ (int.digits 10 prod).length = 9 :=
by
  sorry

end max_digits_in_product_l370_370463


namespace perimeter_even_l370_370398

-- Definitions based on the conditions
def is_integer_coordinate (p : ℕ × ℕ) : Prop := true

def is_integer_length (a b : ℕ × ℕ) : Prop := 
  ∃ (n : ℕ), ⟦(fst b - fst a)^2 + (snd b - snd a)^2 = n^2⟧

def is_convex (p : list (ℕ × ℕ)) : Prop := true

structure convex_pentagon :=
(vertices : list (ℕ × ℕ))
(length_condition : ∀ (i : fin 5), is_integer_length (vertices.nth_le i sorry) (vertices.nth_le (i + 1) mod 5 sorry))
(convex_condition : is_convex vertices)

-- Main theorem statement
theorem perimeter_even (P : convex_pentagon) : ∃ (m : ℕ), P.perimeter = 2 * m :=
sorry

end perimeter_even_l370_370398


namespace nearest_integer_to_100_times_E_is_272_l370_370041

/-- 
  Suppose we keep rolling a fair 2014-sided die (whose faces are labelled 1, 2, ..., 2014) 
  until we obtain a value less than or equal to the previous roll.
  Let E be the expected number of times we roll the die.
  Prove that the nearest integer to 100 times E is 272.
-/
theorem nearest_integer_to_100_times_E_is_272 :
  let E : ℝ := sorry,
  int.round (100 * E) = 272 :=
sorry

end nearest_integer_to_100_times_E_is_272_l370_370041


namespace find_fourth_number_l370_370909

theorem find_fourth_number
  (A B C D E F G : ℝ)
  (h1 : (A + B + C + D) / 4 = 4)
  (h2 : (D + E + F + G) / 4 = 4)
  (h3 : (A + B + C + D + E + F + G) / 7 = 3) :
  D = 5.5 :=
begin
  sorry
end

end find_fourth_number_l370_370909


namespace Q_at_3_plus_Q_at_neg3_l370_370324

-- Let Q be a cubic polynomial where the conditions are defined
def cubic_poly (a b c d : ℝ) := λ x : ℝ, a * x^3 + b * x^2 + c * x + d

-- Define the given conditions as hypotheses
variables {a b c d : ℝ}

-- Given conditions
def h1 : cubic_poly a b c d 0 = 2 * d := by
  sorry

def h2 : cubic_poly a b c d 1 = 3 * d := by
  sorry

def h3 : cubic_poly a b c d (-1) = 5 * d := by
  sorry

-- Define the main theorem that needs to be proven
theorem Q_at_3_plus_Q_at_neg3 : cubic_poly a b c d 3 + cubic_poly a b c d (-3) = 0 :=
by
  sorry

end Q_at_3_plus_Q_at_neg3_l370_370324


namespace rearrange_marked_squares_l370_370295

theorem rearrange_marked_squares (n k : ℕ) (h : n > 1) (h' : k ≤ n + 1) :
  ∃ (f g : Fin n → Fin n), true := sorry

end rearrange_marked_squares_l370_370295


namespace number_is_five_l370_370462

theorem number_is_five (n : ℕ) (h : (∑ i in finset.range 21, (i+1) * n) / 21 = 55) : n = 5 :=
sorry

end number_is_five_l370_370462


namespace sum_of_valid_n_equals_14_l370_370090

theorem sum_of_valid_n_equals_14 :
  (∑ n in {n : ℤ | ∃ (d : ℤ), d ∣ 30 ∧ 2 * n - 1 = d}, n) = 14 :=
by
  sorry

end sum_of_valid_n_equals_14_l370_370090


namespace tangent_line_at_1_is_correct_l370_370832

-- Defining the function f
def f (x : ℝ) : ℝ := 2 * Real.log x + x^2

-- Given conditions
def tangent_point : ℝ := 1

-- Value of the function at the tangent point
def f_at_point : ℝ := f tangent_point

-- The slope of the tangent line
def f_derivative (x : ℝ) : ℝ := (2 / x) + 2 * x
def slope_at_tangent_point : ℝ := f_derivative tangent_point

-- Equation of the tangent line
def tangent_line_equation (x y : ℝ) : Prop := 4 * x - y - 3 = 0

-- Proof statement: The equation of the tangent line to the function at x = 1 is 4x - y - 3 = 0
theorem tangent_line_at_1_is_correct : ∃ (x y : ℝ), tangent_line_equation x y := 
sorry

end tangent_line_at_1_is_correct_l370_370832


namespace sum_of_integer_values_l370_370086

theorem sum_of_integer_values (n : ℤ) (h : ∃ k : ℤ, 30 = k * (2 * n - 1)) : 
  ∑ n in {n | ∃ k : ℤ, 30 = k * (2 * n - 1)}, n = 14 :=
by
  sorry

end sum_of_integer_values_l370_370086


namespace dana_in_seat_2_l370_370157

def seated_positions : Type := ℕ → option string

axiom positions_filled : seated_positions 
axiom bret_in_seat_3 : positions_filled 3 = some "Bret"
axiom joe_wrong_about_bret_next_to_carl : (positions_filled 2 = some "Carl" ∨ positions_filled 4 = some "Carl") → False
axiom joe_wrong_about_abby_between_bret_and_carl : 
  (positions_filled 4 = some "Carl" ∨ positions_filled 1 = some "Carl") ∧ 
  (positions_filled 2 = some "Abby" ∨ positions_filled 4 = some "Abby") → False

theorem dana_in_seat_2 : positions_filled 2 = some "Dana" := 
by 
  sorry

end dana_in_seat_2_l370_370157


namespace pentagon_coloring_count_l370_370066

-- Define the three colors
inductive Color
| Red
| Yellow
| Green

open Color

-- Define the pentagon coloring problem
def adjacent_different (color1 color2 : Color) : Prop :=
color1 ≠ color2

-- Define a coloring for the pentagon
structure PentagonColoring :=
(A B C D E : Color)
(adjAB : adjacent_different A B)
(adjBC : adjacent_different B C)
(adjCD : adjacent_different C D)
(adjDE : adjacent_different D E)
(adjEA : adjacent_different E A)

-- The main statement to prove
theorem pentagon_coloring_count :
  ∃ (colorings : Finset PentagonColoring), colorings.card = 30 := sorry

end pentagon_coloring_count_l370_370066


namespace sin_non_increasing_on_0_2_l370_370943

theorem sin_non_increasing_on_0_2 : ∃ f : ℝ → ℝ, (∀ x ∈ (Ioo 0 (2:ℝ)), f x > f 0) ∧ ¬ (∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f x ≤ f y) :=
by
  let f := λ x:ℝ, sin x
  use f
  split
  · intros x hx
    have h_interval : 0 < x ∧ x < 2 := hx
    linarith [h_interval.1, h_interval.2]
  · sorry

end sin_non_increasing_on_0_2_l370_370943


namespace susan_well_trips_l370_370042

theorem susan_well_trips
  (r_tank : ℝ) (h_tank : ℝ) (r_bucket : ℝ) (h_bucket : ℝ)
  (V_tank : ℝ := π * r_tank^2 * h_tank)
  (V_bucket : ℝ := (1/3) * π * r_bucket^2 * h_bucket)
  (number_of_trips : ℝ := (V_tank / V_bucket).ceil)
  (h_tank : r_tank = 8) (h_tank : h_tank = 24)
  (h_bucket : r_bucket = 5) (h_bucket : h_bucket = 10) : 
  number_of_trips = 10 :=
sorry

end susan_well_trips_l370_370042


namespace rectangle_diagonal_ratio_l370_370180

theorem rectangle_diagonal_ratio (s : ℝ) :
  let d := (Real.sqrt 2) * s
  let D := (Real.sqrt 10) * s
  D / d = Real.sqrt 5 :=
by
  let d := (Real.sqrt 2) * s
  let D := (Real.sqrt 10) * s
  sorry

end rectangle_diagonal_ratio_l370_370180


namespace ratio_of_areas_l370_370132

theorem ratio_of_areas (r : ℝ) (n : ℕ) (k : ℕ) (hk : k = n / 2) (hr : r = 3) 
  (H : n = 6) (Hk : k = 3) : 
  let A_original := Real.pi * r^2 in
  let C_arc := (2 * Real.pi * r) / n in
  let r_small := C_arc / (2 * Real.pi) in
  let A_small := Real.pi * r_small^2 in
  let total_A_small := k * A_small in
  total_A_small / A_original = 1 / 12 := by {
  sorry
}

end ratio_of_areas_l370_370132


namespace point_P_trajectory_circle_l370_370693

noncomputable def trajectory_of_point_P (d h1 h2 : ℝ) (x y : ℝ) : Prop :=
  (x - d/2)^2 + y^2 = (h1^2 + h2^2) / (2 * (h2/h1)^(2/3))

theorem point_P_trajectory_circle :
  ∀ (d h1 h2 x y : ℝ),
  d = 20 →
  h1 = 15 →
  h2 = 10 →
  (∃ x y, trajectory_of_point_P d h1 h2 x y) →
  (∃ x y, (x - 16)^2 + y^2 = 24^2) :=
by
  intros d h1 h2 x y hd hh1 hh2 hxy
  sorry

end point_P_trajectory_circle_l370_370693


namespace find_linear_function_find_coordinate_l370_370649
open Real

-- Definition of points and linear function
def pointA := (3, 5) : ℝ × ℝ
def pointB := (-4, -9) : ℝ × ℝ

-- Linear function through pointA and pointB
def linear_function (x : ℝ) : ℝ := 2 * x - 1

-- Statement of the proof problem (1): Finding the linear function
theorem find_linear_function :
  ∃ k b : ℝ, (k ≠ 0) ∧
  (linear_function 3 = 5) ∧
  (linear_function (-4) = -9) :=
sorry

-- Coordinate of point C
def pointC (m : ℝ) := (m, 2) : ℝ × ℝ

-- Statement of the proof problem (2): Finding the coordinate of point C
theorem find_coordinate :
  pointC (3 / 2) = (3 / 2, 2) :=
sorry

end find_linear_function_find_coordinate_l370_370649


namespace petrol_expenses_l370_370978

-- Definitions based on the conditions stated in the problem
def salary_saved (salary : ℝ) : ℝ := 0.10 * salary
def total_known_expenses : ℝ := 5000 + 1500 + 4500 + 2500 + 3940

-- Main theorem statement that needs to be proved
theorem petrol_expenses (salary : ℝ) (petrol : ℝ) :
  salary_saved salary = 2160 ∧ salary - 2160 = 19440 ∧ 
  5000 + 1500 + 4500 + 2500 + 3940 = total_known_expenses  →
  petrol = 2000 :=
sorry

end petrol_expenses_l370_370978


namespace find_total_coins_l370_370432

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l370_370432


namespace matrix_N_exists_l370_370621

-- Definition of the matrix multiplication
def matrix_mult (N A : ℕ × ℕ → ℤ) : ℕ × ℕ → ℤ := 
λ i j, if i = 0 then if j = 0 then N (0, 0) * A (0, 0) + N (0, 1) * A (1, 0)
                  else N (0, 0) * A (0, 1) + N (0, 1) * A (1, 1)
              else if j = 0 then N (1, 0) * A (0, 0) + N (1, 1) * A (1, 0)
                           else N (1, 0) * A (0, 1) + N (1, 1) * A (1, 1)

-- The statement we want to prove
theorem matrix_N_exists : 
  ∃ (N : ℕ × ℕ → ℤ), 
    (∀ (a b c d : ℤ), matrix_mult N (λ (i j : ℕ), if i = 0 then if j = 0 then a else b else if j = 0 then c else d) = 
                      (λ (i j : ℕ), if i = 0 then if j = 0 then 3 * a else b else if j = 0 then 3 * c else d)) 
  ∨ 
    (N = (λ (i j : ℕ), 0)) := 
begin
  sorry,
end

end matrix_N_exists_l370_370621


namespace prob_sum_seven_prob_two_fours_l370_370899

-- Definitions and conditions
def total_outcomes : ℕ := 36
def outcomes_sum_seven : ℕ := 6
def outcomes_two_fours : ℕ := 1

-- Proof problem for question 1
theorem prob_sum_seven : outcomes_sum_seven / total_outcomes = 1 / 6 :=
by
  sorry

-- Proof problem for question 2
theorem prob_two_fours : outcomes_two_fours / total_outcomes = 1 / 36 :=
by
  sorry

end prob_sum_seven_prob_two_fours_l370_370899


namespace trapezoid_inequality_l370_370749

theorem trapezoid_inequality (A B C D : Point) (angleA angleD : ℝ)
  (h1 : is_trapezoid A B C D)
  (h2 : angleA < angleD)
  (h3 : angleD < 90) :
  distance A C > distance B D := 
sorry

end trapezoid_inequality_l370_370749


namespace sin_210_eq_neg_half_l370_370557

theorem sin_210_eq_neg_half : Real.sin (210 * Real.pi / 180) = -1 / 2 := by
  -- We use the given angles and their known sine values.
  have angle_30 := Real.pi / 6
  have sin_30 := Real.sin angle_30
  -- Expression for the sine of 210 degrees in radians.
  have angle_210 := 210 * Real.pi / 180
  -- Proving the sine of 210 degrees using angle addition formula and unit circle properties.
  calc
    Real.sin angle_210 
    -- 210 degrees is 180 + 30 degrees, translating to pi and pi/6 in radians.
    = Real.sin (Real.pi + Real.pi / 6) : by rw [←Real.ofReal_nat_cast, ←Real.ofReal_nat_cast, Real.ofReal_add, Real.ofReal_div, Real.ofReal_nat_cast]
    -- Using the sine addition formula: sin(pi + x) = -sin(x).
    ... = - Real.sin (Real.pi / 6) : by exact Real.sin_add_pi_div_two angle_30
    -- Substituting the value of sin(30 degrees).
    ... = - 1 / 2 : by rw sin_30

end sin_210_eq_neg_half_l370_370557


namespace ratio_of_ages_l370_370528

theorem ratio_of_ages (S M : ℕ) (h1 : M = S + 24) (h2 : M + 2 = (S + 2) * 2) (h3 : S = 22) : (M + 2) / (S + 2) = 2 := 
by {
  sorry
}

end ratio_of_ages_l370_370528


namespace speed_on_foot_l370_370004

-- Definitions based on the conditions of the problem
variables (total_distance : ℕ) (total_time : ℕ)
variables (foot_distance : ℕ) (bicycle_speed : ℕ)

-- Setting the values according to the conditions provided
def total_distance := 61
def total_time := 9
def foot_distance := 16
def bicycle_speed := 9

-- Question: Determine the speed on foot, given these conditions
theorem speed_on_foot : ∃ v : ℕ, (v * (total_time - (total_distance - foot_distance) / bicycle_speed)) = foot_distance :=
  sorry

end speed_on_foot_l370_370004


namespace orchid_bushes_total_l370_370061

def current_orchid_bushes : ℕ := 22
def new_orchid_bushes : ℕ := 13

theorem orchid_bushes_total : current_orchid_bushes + new_orchid_bushes = 35 := 
by 
  sorry

end orchid_bushes_total_l370_370061


namespace find_circle_center_l370_370511

theorem find_circle_center
  (x y : ℝ)
  (h1 : 5 * x - 4 * y = 10)
  (h2 : 3 * x - y = 0)
  : x = -10 / 7 ∧ y = -30 / 7 :=
by {
  sorry
}

end find_circle_center_l370_370511


namespace axis_of_symmetry_of_sin_l370_370676

def f (x : Real) : Real := Real.sin (2 * x + Real.pi / 3)

def is_axis_of_symmetry (x: Real) (k: Int) : Prop :=
  x = Real.pi / 12 + (k * Real.pi / 2)

theorem axis_of_symmetry_of_sin (k : Int) :
  is_axis_of_symmetry (Real.pi / 12 + (k * Real.pi / 2)) k :=
by
  sorry

end axis_of_symmetry_of_sin_l370_370676


namespace pirates_treasure_l370_370413

theorem pirates_treasure :
  ∃ m : ℕ,
    (m / 3 + 1) +
    (m / 4 + 5) +
    (m / 5 + 20) = m ∧
    m = 120 :=
by {
  sorry,
}

end pirates_treasure_l370_370413


namespace parking_lot_empty_first_time_l370_370933

theorem parking_lot_empty_first_time : 
  let initial_buses := 15
  let departure_interval := 6 -- minutes
  let entry_delay := 3 -- minutes after first departure
  let entry_interval := 8 -- minutes

  -- number of buses departed by time t
  let buses_departed (t : ℕ) := t / departure_interval
  
  -- number of buses entered by time t
  let buses_entered (t : ℕ) := if t < entry_delay then 0 else (t - entry_delay) / entry_interval

  -- t such that buses_departed(t) - buses_entered(t) = initial_buses :=
  11.5 hours :=
by
  sorry

end parking_lot_empty_first_time_l370_370933


namespace central_bank_interested_in_loyalty_program_banks_benefit_from_loyalty_program_registration_required_for_loyalty_program_l370_370121

-- Economic Incentives Conditions and Definitions
axiom loyalty_program : Type
axiom banks : Type
axiom customers : Type
axiom transactions : Type
axiom increased_transactions : loyalty_program → Prop
axiom financial_inclusion : loyalty_program → Prop
axiom customer_benefits : loyalty_program → Prop
axiom personalization : loyalty_program → Prop
axiom budget_control : loyalty_program → Prop
axiom central_bank_benefit (lp : loyalty_program) : increased_transactions lp → financial_inclusion lp → Prop
axiom bank_benefit (lp : loyalty_program) : customer_benefits lp → increased_transactions lp → Prop
axiom registration_necessity (lp : loyalty_program) : personalization lp → budget_control lp → Prop

-- Part (a): Prove the benefit to the Central Bank
theorem central_bank_interested_in_loyalty_program (lp : loyalty_program) :
  central_bank_benefit lp (increased_transactions lp) (financial_inclusion lp) :=
sorry

-- Part (b): Prove the benefit to the banks
theorem banks_benefit_from_loyalty_program (lp : loyalty_program) :
  bank_benefit lp (customer_benefits lp) (increased_transactions lp) :=
sorry

-- Part (c): Prove the necessity of registration
theorem registration_required_for_loyalty_program (lp : loyalty_program) :
  registration_necessity lp (personalization lp) (budget_control lp) :=
sorry

end central_bank_interested_in_loyalty_program_banks_benefit_from_loyalty_program_registration_required_for_loyalty_program_l370_370121


namespace alpha_tan_beta_gt_beta_tan_alpha_l370_370007

theorem alpha_tan_beta_gt_beta_tan_alpha (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) 
: α * Real.tan β > β * Real.tan α := 
sorry

end alpha_tan_beta_gt_beta_tan_alpha_l370_370007


namespace stripe_area_is_correct_l370_370504

-- Define the given conditions
def diameter : ℝ := 20
def height : ℝ := 100
def stripe_width : ℝ := 2
def revolutions : ℝ := 3

-- Define the circumference of the silo
def circumference (d : ℝ) : ℝ := π * d

-- Define the total horizontal length of the stripe
def total_horizontal_length (rev : ℝ) (circumference : ℝ) : ℝ :=
  rev * circumference

-- Define the area of the stripe
def stripe_area (width : ℝ) (length : ℝ) : ℝ :=
  width * length

-- State the theorem to be proved
theorem stripe_area_is_correct :
  stripe_area stripe_width (total_horizontal_length revolutions (circumference diameter)) = 240 * π :=
by
  sorry

end stripe_area_is_correct_l370_370504


namespace probability_red_other_side_expected_value_black_shown_l370_370861

variables {Ω : Type*} [ProbabilitySpace Ω]

/-- The probability that the other side of a card is also red given
that one side is red. -/
theorem probability_red_other_side 
  (card_red_red card_black_black card_red_black : Ω)
  (p : ProbabilitySpace Ω) :
  (p.event (λ ω, ω = card_red_red)) /
  (p.event (λ ω, ω = card_red_red ∨ ω = card_red_black && red_shown)) = 2 / 3 := sorry

/-- The expected value of the number of times black is shown in two draws. -/
theorem expected_value_black_shown 
  (card_red_red card_black_black card_red_black : Ω)
  (p : ProbabilitySpace Ω) :
  let X : ℕ → Ω → ℕ := λ n ω, if shows_black ω then 1 else 0 in
  E[X 1 + X 2] = 1 := sorry

end probability_red_other_side_expected_value_black_shown_l370_370861


namespace part1_part2_l370_370269

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.cos x) * (Real.sin (x + Real.pi / 6)) - 1

-- Part 1: Proving that f(pi/6) = 2
theorem part1 : f (Real.pi / 6) = 2 := sorry

-- Part 2: Proving the maximum and minimum values of f(x) on the interval [-pi/6, pi/4]
theorem part2 : (∀ x, x ∈ Icc (-Real.pi / 6) (Real.pi / 4) → -1 ≤ f x ∧ f x ≤ 2) := sorry

end part1_part2_l370_370269


namespace flagpole_snaps_at_x_l370_370946

noncomputable def flagpole_break_height : Real :=
  let AB : Real := 6
  let BC : Real := 2
  let AC : Real := Real.sqrt (AB^2 + BC^2)
  let x : Real := AC / 2
  x

theorem flagpole_snaps_at_x :
  let x := flagpole_break_height in
  x = Real.sqrt 10 := by
  sorry

end flagpole_snaps_at_x_l370_370946


namespace expected_score_two_free_throws_is_correct_l370_370503

noncomputable def expected_score_two_free_throws (p : ℝ) (n : ℕ) : ℝ :=
n * p

theorem expected_score_two_free_throws_is_correct : expected_score_two_free_throws 0.7 2 = 1.4 :=
by
  -- Proof will be written here.
  sorry

end expected_score_two_free_throws_is_correct_l370_370503


namespace treasures_coins_count_l370_370410

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l370_370410


namespace sin_210_eq_neg_one_half_l370_370586

theorem sin_210_eq_neg_one_half :
  ∀ (θ : ℝ), 
  θ = 210 * (π / 180) → -- angle 210 degrees
  ∃ (refθ : ℝ), 
  refθ = 30 * (π / 180) ∧ -- reference angle 30 degrees
  sin refθ = 1 / 2 → -- sin of reference angle
  sin θ = -1 / 2 := 
by
  intros θ hθ refθ hrefθ hrefθ_sin -- introduce variables and hypotheses
  sorry

end sin_210_eq_neg_one_half_l370_370586


namespace inequalities_proof_l370_370233

variables (x y z : ℝ)

def p := x + y + z
def q := x * y + y * z + z * x
def r := x * y * z

theorem inequalities_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (p x y z) ^ 2 ≥ 3 * (q x y z) ∧
  (p x y z) ^ 3 ≥ 27 * (r x y z) ∧
  (p x y z) * (q x y z) ≥ 9 * (r x y z) ∧
  (q x y z) ^ 2 ≥ 3 * (p x y z) * (r x y z) ∧
  (p x y z) ^ 2 * (q x y z) + 3 * (p x y z) * (r x y z) ≥ 4 * (q x y z) ^ 2 ∧
  (p x y z) ^ 3 + 9 * (r x y z) ≥ 4 * (p x y z) * (q x y z) ∧
  (p x y z) * (q x y z) ^ 2 ≥ 2 * (p x y z) ^ 2 * (r x y z) + 3 * (q x y z) * (r x y z) ∧
  (p x y z) * (q x y z) ^ 2 + 3 * (q x y z) * (r x y z) ≥ 4 * (p x y z) ^ 2 * (r x y z) ∧
  2 * (q x y z) ^ 3 + 9 * (r x y z) ^ 2 ≥ 7 * (p x y z) * (q x y z) * (r x y z) ∧
  (p x y z) ^ 4 + 4 * (q x y z) ^ 2 + 6 * (p x y z) * (r x y z) ≥ 5 * (p x y z) ^ 2 * (q x y z) :=
by sorry

end inequalities_proof_l370_370233


namespace parabola_symmetry_l370_370259

theorem parabola_symmetry (a b c y1 y2 : ℝ) (h_a : a > 0) 
    (h_parabola_1 : y1 = a * (-1)^2 + b * (-1) + c) 
    (h_parabola_2 : y2 = a * 2^2 + b * 2 + c) 
    (h_symmetry_axis : ∀ x y, y = a * x^2 + b * x + c → y (2 * 1 - x) = y) 
    (h_vertex_position : ∀ x y, y = a * x^2 + b * x + c → x = 1 → x) : 
  y1 > y2 := 
sorry

end parabola_symmetry_l370_370259


namespace evaluate_expression_l370_370202

theorem evaluate_expression (x : ℝ) : (1 - x^2) * (1 + x^4) = 1 - x^2 + x^4 - x^6 :=
by
  sorry

end evaluate_expression_l370_370202


namespace totalPlayers_l370_370922

def kabadiParticipants : ℕ := 50
def khoKhoParticipants : ℕ := 80
def soccerParticipants : ℕ := 30
def kabadiAndKhoKhoParticipants : ℕ := 15
def kabadiAndSoccerParticipants : ℕ := 10
def khoKhoAndSoccerParticipants : ℕ := 25
def allThreeParticipants : ℕ := 8

theorem totalPlayers : kabadiParticipants + khoKhoParticipants + soccerParticipants 
                       - kabadiAndKhoKhoParticipants - kabadiAndSoccerParticipants 
                       - khoKhoAndSoccerParticipants + allThreeParticipants = 118 :=
by 
  sorry

end totalPlayers_l370_370922


namespace work_completion_time_l370_370349

theorem work_completion_time :
  let mary_eff := (1 : ℝ) / 28
  let rosy_eff := 1.4 * mary_eff
  let tim_eff := 0.8 * mary_eff
  let combined_eff := mary_eff + rosy_eff + tim_eff
  combined_eff = 2 / 35 → 1 / combined_eff = 17.5 := by
  intros mary_eff rosy_eff tim_eff combined_eff h
  have h_mary : mary_eff = 1 / 28 := rfl
  have h_rosy : rosy_eff = 1.4 * mary_eff := rfl
  have h_tim : tim_eff = 0.8 * mary_eff := rfl
  have h_combined : combined_eff = mary_eff + rosy_eff + tim_eff := rfl
  rw [h_mary, h_rosy, h_tim, h_combined] at h
  sorry

end work_completion_time_l370_370349


namespace van_distance_l370_370541

noncomputable def distance_covered (initial_time new_time speed : ℝ) : ℝ :=
  speed * new_time

theorem van_distance :
  distance_covered 5 (5 * (3 / 2)) 60 = 450 := 
by
  sorry

end van_distance_l370_370541


namespace chosen_number_is_120_l370_370904

theorem chosen_number_is_120 (x : ℤ) (h : 2 * x - 138 = 102) : x = 120 :=
sorry

end chosen_number_is_120_l370_370904


namespace length_LN_l370_370369

theorem length_LN (N : ℝ) (LM LN : ℝ) (h1 : Math.cos N = 3 / 5) (h2 : LM = 15) : 
  LN = 25 :=
by
  sorry

end length_LN_l370_370369


namespace find_b4_l370_370599

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def factorial_base_representation (n : ℕ) : List ℕ :=
  let rec aux (n rem : ℕ) (k : ℕ) (acc : List ℕ) : List ℕ :=
    if k = 0 then acc
    else
      let coeff := rem / factorial k
      let new_rem := rem % factorial k
      aux n new_rem (k - 1) (coeff :: acc)
  aux n n (n - 1) []

theorem find_b4 (n : ℕ) (h : n = 873) : (factorial_base_representation n).nth 4 = some 1 :=
  sorry

end find_b4_l370_370599


namespace distance_from_apex_l370_370125

-- Define the areas of the two cross sections
def area1 : ℝ := 256 * Real.sqrt 2
def area2 : ℝ := 576 * Real.sqrt 2

-- Define the distance between the planes
def d : ℝ := 10

-- Prove that the distance from the apex to the larger cross section is 30 feet
theorem distance_from_apex (h : ℝ) (area1 area2 : ℝ) (d : ℝ) :
  area1 = 256 * Real.sqrt 2 → 
  area2 = 576 * Real.sqrt 2 → 
  d = 10 → 
  h = 30 :=
by
  intros h_eq_a1_area1 h_eq_a2_area2 d_eq_10
  sorry

end distance_from_apex_l370_370125


namespace quarters_difference_nickels_eq_l370_370174

variable (q : ℕ)

def charles_quarters := 7 * q + 2
def richard_quarters := 3 * q + 7
def quarters_difference := charles_quarters q - richard_quarters q
def money_difference_in_nickels := 5 * quarters_difference q

theorem quarters_difference_nickels_eq :
  money_difference_in_nickels q = 20 * (q - 5/4) :=
by
  sorry

end quarters_difference_nickels_eq_l370_370174


namespace room_length_l370_370384

theorem room_length
  (width : ℝ)
  (cost_rate : ℝ)
  (total_cost : ℝ)
  (h_width : width = 4)
  (h_cost_rate : cost_rate = 850)
  (h_total_cost : total_cost = 18700) :
  ∃ L : ℝ, L = 5.5 ∧ total_cost = cost_rate * (L * width) :=
by
  sorry

end room_length_l370_370384


namespace find_min_a_l370_370683

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := real.exp x + 0.5 * a * x^2 - 1

theorem find_min_a :
  (∀ x > 0, f'(f x a) ≥ 0) → a ≥ -real.exp 1 :=
begin
  sorry
end

end find_min_a_l370_370683


namespace find_four_digit_number_l370_370603

variable {N : ℕ} {a x y : ℕ}

theorem find_four_digit_number :
  (∃ a x y : ℕ, y < 10 ∧ 10 + a = x * y ∧ x = 9 + a ∧ N = 1000 + a + 10 * b + 100 * b ∧
  (N = 1014 ∨ N = 1035 ∨ N = 1512)) :=
by
  sorry

end find_four_digit_number_l370_370603


namespace minimum_value_of_T_l370_370687

theorem minimum_value_of_T (a b c : ℝ) (h1 : ∀ x : ℝ, (1 / a) * x^2 + b * x + c ≥ 0) (h2 : a * b > 1) :
  ∃ T : ℝ, T = 4 ∧ T = (1 / (2 * (a * b - 1))) + (a * (b + 2 * c) / (a * b - 1)) :=
by
  sorry

end minimum_value_of_T_l370_370687


namespace sqrt_div_simplification_l370_370203

theorem sqrt_div_simplification (x y : ℕ) (h: 
  ( ( (1 / 3: ℚ) ^ 2 + (1 / 4) ^ 2 + (1 / 6) ^ 2 ) / ( (1 / 5) ^ 2 + (1 / 7) ^ 2 + (1 / 8) ^ 2 ) = 37 * x / (85 * y) ) 
: (real.sqrt x / real.sqrt y) = 1737 / 857 := 
by
  sorry

end sqrt_div_simplification_l370_370203


namespace price_of_adult_ticket_eq_32_l370_370500

theorem price_of_adult_ticket_eq_32 
  (num_adults : ℕ)
  (num_children : ℕ)
  (price_child_ticket : ℕ)
  (price_adult_ticket : ℕ)
  (total_collected : ℕ)
  (h1 : num_adults = 400)
  (h2 : num_children = 200)
  (h3 : price_adult_ticket = 2 * price_child_ticket)
  (h4 : total_collected = 16000)
  (h5 : total_collected = num_adults * price_adult_ticket + num_children * price_child_ticket)
  : price_adult_ticket = 32 := 
by
  sorry

end price_of_adult_ticket_eq_32_l370_370500


namespace muffins_per_box_l370_370046

theorem muffins_per_box (total_muffins : ℕ) (available_boxes : ℕ) (needed_boxes : ℕ) (total_boxes : ℕ) (muffins_per_box : ℕ) :
  total_muffins = 95 → available_boxes = 10 → needed_boxes = 9 → total_boxes = available_boxes + needed_boxes → total_boxes * muffins_per_box = total_muffins → muffins_per_box = 5 :=
by {
  intros h1 h2 h3 h4 h5,
  sorry
}

end muffins_per_box_l370_370046


namespace quad_eq_diagonals_theorem_l370_370238

noncomputable def quad_eq_diagonals (a b c d m n : ℝ) (A C : ℝ) : Prop :=
  m^2 * n^2 = a^2 * c^2 + b^2 * d^2 - 2 * a * b * c * d * Real.cos (A + C)

theorem quad_eq_diagonals_theorem (a b c d m n A C : ℝ) :
  quad_eq_diagonals a b c d m n A C :=
by
  sorry

end quad_eq_diagonals_theorem_l370_370238


namespace pirates_treasure_l370_370431

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l370_370431


namespace pages_per_comic_l370_370546

variable {comics_initial : ℕ} -- initially 5 untorn comics in the box
variable {comics_final : ℕ}   -- now there are 11 comics in the box
variable {pages_found : ℕ}    -- found 150 pages on the floor
variable {comics_assembled : ℕ} -- comics assembled from the found pages

theorem pages_per_comic (h1 : comics_initial = 5) (h2 : comics_final = 11) 
      (h3 : pages_found = 150) (h4 : comics_assembled = comics_final - comics_initial) :
      (pages_found / comics_assembled = 25) := 
sorry

end pages_per_comic_l370_370546


namespace cubic_root_simplification_l370_370025

def simplify_cubic_root : ℚ :=
  let factorized := 2^4 * 7^3
  let simplified_inner := 2 ^ (4 / 3 : ℚ) * 7
  let integer_part_of_simplified := 2 * (2 ^ (1 / 3 : ℚ)) * 7
  in 10 * integer_part_of_simplified

theorem cubic_root_simplification : 
  simplify_cubic_root = 176.4 :=
by
  sorry

end cubic_root_simplification_l370_370025


namespace find_x_squared_add_y_squared_l370_370284

variable (x y : ℝ)
variable h : (x^2 + 1) * (y^2 + 1) + 9 = 6 * (x + y)

theorem find_x_squared_add_y_squared : x^2 + y^2 = 7 :=
sorry

end find_x_squared_add_y_squared_l370_370284


namespace find_power_l370_370495

theorem find_power (x : ℝ) (h1 : 16 = 2 ^ 4) (h2 : 4 = 2 ^ 2) : 16 ^ x = 4 ^ 10 → x = 5 :=
sorry

end find_power_l370_370495


namespace evaluate_expression_l370_370184

def f (x : ℝ) := x^3 + x^2 + 2 * Real.sqrt x

theorem evaluate_expression : 3 * f 3 - 2 * f 9 = -1524 + 6 * Real.sqrt 3 := by
  sorry

end evaluate_expression_l370_370184


namespace simplify_expr1_simplify_expr2_l370_370812

-- Define the conditions and the expressions
variable (a x : ℝ)

-- Expression 1
def expr1 := 2 * (a - 1) - (2 * a - 3) + 3
def expr1_simplified := 4

-- Expression 2
def expr2 := 3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2)
def expr2_simplified := x^2 - 3 * x + 3

-- Prove the simplifications
theorem simplify_expr1 : expr1 a = expr1_simplified :=
by sorry

theorem simplify_expr2 : expr2 x = expr2_simplified :=
by sorry

end simplify_expr1_simplify_expr2_l370_370812


namespace incorrect_option_D_l370_370895

-- definition of geometric objects and their properties
def octahedron_faces : Nat := 8
def tetrahedron_can_be_cut_into_4_pyramids : Prop := True
def frustum_extension_lines_intersect_at_a_point : Prop := True
def rectangle_rotated_around_side_forms_cylinder : Prop := True

-- incorrect identification of incorrect statement
theorem incorrect_option_D : 
  (∃ statement : String, statement = "D" ∧ ¬rectangle_rotated_around_side_forms_cylinder)  → False :=
by
  -- Proof of incorrect identification is not required per problem instructions
  sorry

end incorrect_option_D_l370_370895


namespace rectangle_length_l370_370912

theorem rectangle_length {b l : ℝ} (h1 : 2 * (l + b) = 5 * b) (h2 : l * b = 216) : l = 18 := by
    sorry

end rectangle_length_l370_370912


namespace sum_of_p_digits_l370_370774

def sum_of_digits (x : ℕ) : ℕ :=
  x.digits 10 |>.sum

def T : set ℕ :=
  {n | sum_of_digits n = 15 ∧ 0 ≤ n ∧ n < 10^8}

def p : ℕ :=
  T.to_finset.card

theorem sum_of_p_digits : sum_of_digits p = 21 := by
  sorry

end sum_of_p_digits_l370_370774


namespace total_jewelry_pieces_l370_370524

noncomputable def initial_necklaces : ℕ := 10
noncomputable def initial_earrings : ℕ := 15
noncomputable def bought_necklaces : ℕ := 10
noncomputable def bought_earrings : ℕ := 2 * initial_earrings / 3
noncomputable def extra_earrings_from_mother : ℕ := bought_earrings / 5

theorem total_jewelry_pieces : initial_necklaces + bought_necklaces + initial_earrings + bought_earrings + extra_earrings_from_mother = 47 :=
by
  have total_necklaces : ℕ := initial_necklaces + bought_necklaces
  have total_earrings : ℕ := initial_earrings + bought_earrings + extra_earrings_from_mother
  have total_jewelry : ℕ := total_necklaces + total_earrings
  exact Eq.refl 47
  
#check total_jewelry_pieces -- Check if the type is correct

end total_jewelry_pieces_l370_370524


namespace sequence_sum_l370_370746

noncomputable def sequence (n : ℕ) : ℕ :=
if n % 3 = 0 then 1
else if n % 3 = 1 then 2
else 3

theorem sequence_sum :
  let Sₙ := ∑ i in Finset.range 2010, sequence i in
  Sₙ = 4020 :=
by
  sorry

end sequence_sum_l370_370746


namespace num_solutions_50_0_l370_370329

noncomputable def f₀ (x : ℝ) : ℝ :=
if x < -50 then x + 100
else if x < 50 then -x
else x - 100

noncomputable def f (n : ℕ) : ℝ → ℝ
| 0     => f₀
| n + 1 => λ x, |f n x| - 1

def num_solutions (n : ℕ) (value : ℝ) : ℕ :=
(count {x : ℝ | f n x = value})

theorem num_solutions_50_0 : num_solutions 50 0 = 4 :=
sorry

end num_solutions_50_0_l370_370329


namespace cannot_determine_parallelogram_l370_370096

/-- Defining a quadrilateral ABCD. -/
structure Quadrilateral :=
(A B C D : ℝ × ℝ)

def is_parallelogram (q : Quadrilateral) : Prop :=
  (q.A - q.B) = (q.C - q.D)

def condition1 (q : Quadrilateral) : Prop :=
  (q.A - q.B) = (q.C - q.D) ∧ (q.A.1 - q.B.1).abs = (q.C.1 - q.D.1).abs

def condition2 (q : Quadrilateral) : Prop :=
  (q.A.2 - q.B.2).abs = (q.C.2 - q.D.2).abs

def condition3 (q : Quadrilateral) : Prop :=
  (q.B.2 - q.C.2).abs = (q.D.2 - q.A.2).abs

def condition4 (q : Quadrilateral) : Prop :=
  (q.A - q.B) = (q.A - q.D)

def condition5 (q : Quadrilateral) : Prop :=
  (q.B - q.C) = (q.C - q.D)

def condition6 (q : Quadrilateral) : Prop :=
  (q.A - q.B).abs = (q.C - q.D).abs

def condition7 (q : Quadrilateral) : Prop :=
  (q.A - q.D).abs = (q.B - q.C).abs

noncomputable def satisfies_conditions (q : Quadrilateral) : Prop :=
  condition4 q ∧ condition5 q

theorem cannot_determine_parallelogram (q : Quadrilateral) :
  satisfies_conditions q → ¬ is_parallelogram q :=
by sorry

end cannot_determine_parallelogram_l370_370096


namespace length_CF_and_area_ACF_l370_370108

open Real 

noncomputable def circle_radius := (17 : ℝ)
noncomputable def BC := (16 : ℝ)

-- Point B is on segment CD and angle CAD = 90 degrees
noncomputable def angle_CAD := π / 2

-- Given circles with radius 17 intersect at A and B
-- C on the first circle and D on the second circle
variable (A B C D F : Point)
variable (r : ℝ) (h_rad : r = circle_radius)
variable (R1 R2 : Circle) (h_inter : Circle.inter R1 R2 A) (h_inter' : Circle.inter R1 R2 B)
variable (h_on_C : on_circle C R1) (h_on_D : on_circle D R2)
variable (h_on_B_CD : lies_on B (line_segment C D))
variable (h_angle_CAD : angle C A D = angle_CAD)
variable (h_perpendicular : perpendicular (line_segment C D) (line_through B F))
variable (h_BF_BD : distance B F = distance B D)
variable (h_opposite_sides : same_line_opposite_sides A F (line_segment C D))
variable (h_BC : distance B C = BC)

theorem length_CF_and_area_ACF :
  (distance C F = 2 * circle_radius) ∧ (triangle_area A C F = circle_radius^2) :=
by
  sorry

end length_CF_and_area_ACF_l370_370108


namespace simplify_expr1_simplify_expr2_l370_370814

theorem simplify_expr1 (a : ℝ) : 2 * (a - 1) - (2 * a - 3) + 3 = 4 :=
by
  sorry

theorem simplify_expr2 (x : ℝ) : 3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = 5 * x^2 - 3 * x - 3 :=
by
  sorry

end simplify_expr1_simplify_expr2_l370_370814


namespace largest_C_is_one_third_l370_370325

noncomputable def largest_C (α : ℝ) (hα : 0 < α) : ℝ :=
  Inf {C : ℝ | ∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x * y + y * z + z * x = α →
    (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ C * (x / z + z / x + 2)}

theorem largest_C_is_one_third (α : ℝ) (hα : 0 < α) :
  largest_C α hα = 1 / 3 :=
sorry

end largest_C_is_one_third_l370_370325


namespace sin_210_l370_370568

theorem sin_210 : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end sin_210_l370_370568


namespace symmetric_points_exists_l370_370267

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a * Real.exp (-x) else Real.log (x / a)

theorem symmetric_points_exists (a : ℝ) (h_a : 0 < a) :
  (∃ x0 : ℝ, 1 < x0 ∧ f a x0 = f a (-x0)) ↔ (a > 0 ∧ a < 1 / Real.exp 1) :=
begin
  sorry
end

end symmetric_points_exists_l370_370267


namespace max_socks_display_l370_370481

theorem max_socks_display (a : ℕ → ℕ) (h_sum : (finset.range 2018).sum a = 2017) :
  ∃ (k : ℕ), k = 3^671 * 4 :=
begin
  sorry
end

end max_socks_display_l370_370481


namespace inequality_holds_for_p_in_0_to_3_l370_370206

theorem inequality_holds_for_p_in_0_to_3 (a b p : ℝ) (ha : 0 < a) (hb : 0 < b) (hp : 0 < p ∧ p ≤ 3) :
  (sqrt (a^2 + p * b^2) + sqrt (b^2 + p * a^2) ≥ a + b + (p - 1) * sqrt (a * b)) :=
begin
  sorry
end

end inequality_holds_for_p_in_0_to_3_l370_370206


namespace proposition_3_proposition_4_l370_370271

variables (p q : Prop)
def p_def : Prop := ∃ (x₀ : ℝ), x₀ ^ 2 + x₀ + 1 < 0
def q_def : Prop := ∀  (a b c : ℝ), (b > c) → (a * b > a * c)

-- Given definitions
variable h_p : ¬ p_def
variable h_q : ¬ q_def

-- Lean 4 equivalent proof problem statements
theorem proposition_3 : (¬ p_def) ∨ q_def := by
  sorry

theorem proposition_4 : (¬ p_def) ∧ (¬ q_def) := by
  sorry

end proposition_3_proposition_4_l370_370271


namespace sum_of_valid_n_equals_14_l370_370091

theorem sum_of_valid_n_equals_14 :
  (∑ n in {n : ℤ | ∃ (d : ℤ), d ∣ 30 ∧ 2 * n - 1 = d}, n) = 14 :=
by
  sorry

end sum_of_valid_n_equals_14_l370_370091


namespace sequence_integers_l370_370192

def sequence (n : ℕ) : ℤ :=
  if n = 0 then 1 else
  if n = 1 then 1 else
  if n = 2 then 1 else
  (sequence (n - 1) * sequence (n - 2) + 2) / sequence (n - 3)

theorem sequence_integers : ∀ n : ℕ, ∃ k : ℤ, sequence n = k :=
by sorry

end sequence_integers_l370_370192


namespace rectangle_area_l370_370865

-- Define the conditions
def shorter_side (rect : ℝ) : ℝ := 8
def num_rectangles : ℕ := 3

-- Define the proof problem
theorem rectangle_area : 
  let longer_side := 2 * shorter_side 8 in
  let width := shorter_side 8 in
  let length := num_rectangles * longer_side in
  length * width = 384 :=
by
  sorry

end rectangle_area_l370_370865


namespace true_propositions_count_l370_370228

theorem true_propositions_count (a b c : ℝ) : 
  let P := ∀ (a b c : ℝ), a > b → ac² > bc² ∧
      P_inv := ∀ (a b c : ℝ), ac² > bc² → a > b in
  (¬ (∀ a b c, a > b → ac² > bc²) ∧ (∀ a b c, ac² > bc² → a > b)) -> 
  2.
sorry

end true_propositions_count_l370_370228


namespace area_of_square_field_l370_370874

theorem area_of_square_field (side_length : ℕ) (h : side_length = 14) : side_length * side_length = 196 :=
by
  rw h
  norm_num

end area_of_square_field_l370_370874


namespace incorrect_average_ratio_l370_370538

def x1_x50 : list ℝ := sorry -- the list of 50 scores
def A : ℝ := (list.sum x1_x50) / 50
def B : ℝ := ((list.sum x1_x50) + 2 * A) / 51

theorem incorrect_average_ratio (h : B = A * (52 / 51)) :
  B / A = 52 / 51 :=
by sorry

end incorrect_average_ratio_l370_370538


namespace num_odd_decompositions_of_315_l370_370847

theorem num_odd_decompositions_of_315 : 
  {n : ℕ // n > 1 ∧ ∃ (x y : ℕ), odd x ∧ odd y ∧ x > 1 ∧ y > 1 ∧ x * y = 315}.card = 5 := 
sorry

end num_odd_decompositions_of_315_l370_370847


namespace triangle_cos_ratio_l370_370752

theorem triangle_cos_ratio (A B C D : Point)
  (hC_right : ∠C = 90°)
  (h_altitude : foot C (line A B) = D)
  (h_sides_int : ∀ (x : ℕ), (side_length x ∈ {length A B, length B C, length C A}))
  (h_BD : length B D = 17^3) :
  ∃ (p q : ℕ), gcd p q = 1 ∧ p + q = 3978 ∧ ∃ r s : ℕ, cos_angle B = r / s :=
by sorry

end triangle_cos_ratio_l370_370752


namespace square_of_sqrt_expression_l370_370994

theorem square_of_sqrt_expression : (sqrt (2 + sqrt (2 + sqrt 2)))^2 = 4 := 
sorry

end square_of_sqrt_expression_l370_370994


namespace general_formula_for_arithmetic_sequence_sum_of_geometric_sequence_l370_370652

noncomputable def S (p : ℝ) (n : ℕ) : ℝ := p * n^2 - 2 * n
noncomputable def a (p : ℝ) (n : ℕ) : ℝ := 8 * n - 6
noncomputable def b (n : ℕ) : ℝ := 2^(4 * n - 3)
noncomputable def T (n : ℕ) : ℝ := (2 / 15) * (16^n - 1)

theorem general_formula_for_arithmetic_sequence (p : ℝ) (h1 : 5 * p - 2 = 18) 
  : ∀ (n : ℕ), 0 < n → a p n = 8 * n - 6 
:= sorry

theorem sum_of_geometric_sequence (h2 : ∀ (n : ℕ), 0 < n → a 4 n = 2 * real.logb (b n) 2)
  : ∀ (n : ℕ), 0 < n → T n = (2 / 15) * (16^n - 1)
:= sorry

end general_formula_for_arithmetic_sequence_sum_of_geometric_sequence_l370_370652


namespace find_x_from_ratio_l370_370262

theorem find_x_from_ratio (x y k: ℚ) 
  (h1 : ∀ x y, (5 * x - 3) / (y + 20) = k) 
  (h2 : 5 * 1 - 3 = 2 * 22) (hy : y = 5) : 
  x = 58 / 55 := 
by 
  sorry

end find_x_from_ratio_l370_370262


namespace smallest_nonnegative_integer_l370_370471

theorem smallest_nonnegative_integer (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 7) (h3 : (-2222) % 7 = n) : n = 1 :=
by
  -- This would be the place to prove the theorem, but we'll skip for now.
  sorry

end smallest_nonnegative_integer_l370_370471


namespace find_x_coordinate_l370_370146

def point_is_equidistant (x y : ℝ) : Prop :=
  abs x = abs y ∧
  abs x = abs (x + 2 * y - 4) / real.sqrt (1^2 + 2^2) ∧
  abs x = abs (y - 2 * x) / real.sqrt (1^2 + (-2)^2)

theorem find_x_coordinate :
  ∃ (x : ℝ), ∃ (y : ℝ), point_is_equidistant x y ∧ x = -4 / (real.sqrt 5 - 7) :=
by sorry

end find_x_coordinate_l370_370146


namespace sin_105_mul_sin_15_eq_one_fourth_l370_370606

noncomputable def sin_105_deg := Real.sin (105 * Real.pi / 180)
noncomputable def sin_15_deg := Real.sin (15 * Real.pi / 180)

theorem sin_105_mul_sin_15_eq_one_fourth :
  sin_105_deg * sin_15_deg = 1 / 4 :=
by
  sorry

end sin_105_mul_sin_15_eq_one_fourth_l370_370606


namespace area_enclosed_F_inequality_l370_370601

noncomputable def F (x y : ℝ) : ℝ := (1 + x)^y

theorem area_enclosed (f : ℝ → ℝ) (c1_intersection_A : f 0 = 9) (c1_tangent_B : ∃ n t, (f n = t) ∧ (2 * n - 4 = t / n) ∧ (n > 0)) :
  ∫ x in 0..3, f x - 2 * x = 9 :=
by sorry

theorem F_inequality (x y : ℕ) (h1 : 0 < x) (h2 : x < y) : F x y > F y x := 
by sorry


end area_enclosed_F_inequality_l370_370601


namespace parallelogram_diagonal_problem_l370_370791

-- Defining the problem conditions
variables (A B C D O N M K : Type) [parallelogram ABCD]
variables (AC BD : Segment) (O : Point) (CO BN DM : Segment) [isDiagonal AC] [isDiagonal BD]
variables [intersect AC BD O] [median CO BCD] [median BN BCD] [median DM BCD]
variables [perpendicular DM AC]
variables (BK KN : Segment) (ratio : Nat) [ratio = 2]
variables [BN_eq : BN = (3/2 : ℚ) * CD]

-- Final proof goal
theorem parallelogram_diagonal_problem : BN = 9 :=
by
  -- Proof omitted, sorry placeholder
  sorry

end parallelogram_diagonal_problem_l370_370791


namespace unique_solution_sequence_l370_370612

open Nat

theorem unique_solution_sequence :
    ∀ {n : ℕ}, (0 < n) →
    (∃ (a : Fin n → ℕ), (∀ k : Fin (n - 2), a ⟨k.1+2, nat.lt_pred_of_lt (Fin.is_lt k)⟩ = (a ⟨k.1+1, nat.lt_pred_of_lt (Fin.is_lt k)⟩ ^ 2 + 1) / (a ⟨k.1, nat.lt_pred_of_lt (Fin.is_lt k)⟩ + 1) - 1))
    ↔ n = 4 := 
by
  sorry

end unique_solution_sequence_l370_370612


namespace sin_210_eq_neg_one_half_l370_370576

theorem sin_210_eq_neg_one_half :
  sin (Real.pi * (210 / 180)) = -1 / 2 :=
by
  have angle_eq : 210 = 180 + 30 := by norm_num
  have sin_30 : sin (Real.pi / 6) = 1 / 2 := by norm_num
  have cos_30 : cos (Real.pi / 6) = sqrt 3 / 2 := by norm_num
  sorry

end sin_210_eq_neg_one_half_l370_370576


namespace contractor_absent_days_l370_370483

theorem contractor_absent_days {x y : ℕ} (h1 : x + y = 30) (h2 : 25 * x - 7.5 * y = 425) : y = 10 :=
by
  sorry

end contractor_absent_days_l370_370483


namespace capacity_difference_l370_370531

def square_tin_volume (side height : ℝ) : ℝ :=
  side * side * height

def circular_tin_volume (diameter height : ℝ) : ℝ :=
  let radius := diameter / 2
  Math.pi * radius * radius * height

noncomputable def volume_difference (side_sq height_sq diameter_circ height_circ : ℝ) : ℝ :=
  square_tin_volume side_sq height_sq - circular_tin_volume diameter_circ height_circ

theorem capacity_difference :
  volume_difference 8 14 8 14 ≈ 192.64 :=
sorry

end capacity_difference_l370_370531


namespace find_R_eq_correct_l370_370237

theorem find_R_eq_correct :
  ∃ (R : ℝ → ℝ), 
    (∀ t : ℝ, 
      let x := Real.sin t in 
      let lhs := 7 * (x ^ 31) + 8 * (x ^ 13) - 5 * (x ^ 9) * ((1 - x ^ 2) ^ 2) - 10 * (x ^ 7) + 5 * (x ^ 5) - 2 in
      let rhs := (7 + 5 * (1 + x) * (2 - (1 - x ^ 2))) * x ^ 4 + R x in
      lhs = rhs)
    ∧ (∀ p q : ℝ, R p + R q < 4) :=
    ∃ (R : ℝ → ℝ), 
      R = (λ x, 13 * (x ^ 3) + 5 * (x ^ 2) + 12 * x + 3) ∧ function.degree_lt (λ x, R x) 4
sorry

end find_R_eq_correct_l370_370237


namespace discount_problem_l370_370935

theorem discount_problem (m : ℝ) (h : (200 * (1 - m / 100)^2 = 162)) : m = 10 :=
sorry

end discount_problem_l370_370935


namespace charlie_fraction_l370_370757

theorem charlie_fraction (J B C : ℕ) (f : ℚ) (hJ : J = 12) (hB : B = 10) 
  (h1 : B = (2 / 3) * C) (h2 : C = f * J + 9) : f = (1 / 2) := by
  sorry

end charlie_fraction_l370_370757


namespace cost_price_per_meter_l370_370485

-- Definitions for conditions
def total_length : ℝ := 9.25
def total_cost : ℝ := 416.25

-- The theorem to be proved
theorem cost_price_per_meter : total_cost / total_length = 45 := by
  sorry

end cost_price_per_meter_l370_370485


namespace sin_210_eq_neg_half_l370_370569

theorem sin_210_eq_neg_half : sin (210 * real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_210_eq_neg_half_l370_370569


namespace sleep_time_mode_and_median_l370_370867

def sleep_time_data : List (ℕ × ℕ) := [(7, 7), (8, 9), (9, 11), (10, 3)]

def mode_correct : Prop :=
  let mode := (sleep_time_data.maximum_by (λ p, p.snd)).fst
  mode = 9

def median_correct : Prop :=
  let sorted_data := List.join (List.map (λ (t : ℕ × ℕ), List.replicate t.snd t.fst) sleep_time_data)
  let n := sorted_data.length
  let median := (sorted_data.get! ((n / 2) - 1) + sorted_data.get! (n / 2)) / 2
  median = 8

theorem sleep_time_mode_and_median : mode_correct ∧ median_correct := by
  sorry

end sleep_time_mode_and_median_l370_370867


namespace calculation_part1_l370_370491

theorem calculation_part1 : (-2)^2 + Real.sqrt 12 - 2 * Real.sin (Float.pi / 3) = 4 + Real.sqrt 3 := sorry

end calculation_part1_l370_370491


namespace pirates_treasure_l370_370411

theorem pirates_treasure :
  ∃ m : ℕ,
    (m / 3 + 1) +
    (m / 4 + 5) +
    (m / 5 + 20) = m ∧
    m = 120 :=
by {
  sorry,
}

end pirates_treasure_l370_370411


namespace three_digit_numbers_distinct_base_l370_370727

theorem three_digit_numbers_distinct_base (b : ℕ) (h : (b - 1) ^ 2 * (b - 2) = 250) : b = 8 :=
sorry

end three_digit_numbers_distinct_base_l370_370727


namespace central_bank_interest_bank_benefit_registration_advantage_l370_370123

noncomputable theory

def interest_of_central_bank (increase_cashless: Prop) (promote_inclusion: Prop) : Prop :=
increase_cashless ∧ promote_inclusion

def benefit_for_banks (increase_clients: Prop) (increase_revenue: Prop) : Prop :=
increase_clients ∧ increase_revenue

def registration_requirement (personalization: Prop) (budget_control: Prop) : Prop :=
personalization ∧ budget_control

theorem central_bank_interest:
  ∀ (loyalty_program : Prop),
    (loyalty_program → interest_of_central_bank true true) :=
begin
  intros loyalty_program,
  split;
  assume h,
  { exact true.intro, },
  { exact true.intro, },
end

theorem bank_benefit:
  ∀ (loyalty_program : Prop),
    (loyalty_program → benefit_for_banks true true) :=
begin
  intros loyalty_program,
  split;
  assume h,
  { exact true.intro, },
  { exact true.intro, },
end

theorem registration_advantage:
  ∀ (loyalty_program : Prop),
    (loyalty_program → registration_requirement true true) :=
begin
  intros loyalty_program,
  split;
  assume h,
  { exact true.intro, },
  { exact true.intro, },
end

end central_bank_interest_bank_benefit_registration_advantage_l370_370123


namespace divisibility_of_fibonacci_l370_370811

theorem divisibility_of_fibonacci (m n : ℕ) (hmn : m ∣ n) : (fibonacci(m-1) ∣ fibonacci(n-1)) :=
by
  sorry

end divisibility_of_fibonacci_l370_370811


namespace last_digit_of_sum_is_four_l370_370350

theorem last_digit_of_sum_is_four (x y z : ℕ)
  (hx : 1 ≤ x ∧ x ≤ 9)
  (hy : 0 ≤ y ∧ y ≤ 9)
  (hz : 0 ≤ z ∧ z ≤ 9)
  (h : 1950 ≤ 200 * x + 11 * y + 11 * z ∧ 200 * x + 11 * y + 11 * z < 2000) :
  (200 * x + 11 * y + 11 * z) % 10 = 4 :=
sorry

end last_digit_of_sum_is_four_l370_370350


namespace sin_210_eq_neg_half_l370_370573

theorem sin_210_eq_neg_half : sin (210 * real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_210_eq_neg_half_l370_370573


namespace diameter_of_circle_C_l370_370554

theorem diameter_of_circle_C :
  (∀ (r : ℝ), (2 * 12) = 24 →
  (144 * Math.pi - Math.pi * r^2 = (4 : ℝ) • Math.pi * r^2) →
   2 * r = 24 * Real.sqrt 5 / 5) := 
begin
  intros r diameter_eq shaded_area_ratio_eq,
  sorry
end

end diameter_of_circle_C_l370_370554


namespace smallest_composite_no_prime_lt_13_l370_370214

/--
The smallest composite number that has no prime factors less than 13 is 169.
-/
theorem smallest_composite_no_prime_lt_13 : ∃ n : ℕ, n = 169 ∧ ∀ p : ℕ, p.prime → p ∣ n → 13 ≤ p :=
by
  sorry

end smallest_composite_no_prime_lt_13_l370_370214


namespace number_added_is_minus_168_l370_370893

theorem number_added_is_minus_168 (N : ℕ) (X : ℤ) (h1 : N = 180)
  (h2 : N + (1/2 : ℚ) * (1/3 : ℚ) * (1/5 : ℚ) * N = (1/15 : ℚ) * N) : X = -168 :=
by
  sorry

end number_added_is_minus_168_l370_370893


namespace benny_bought_two_cards_l370_370759

variable (initial_cards : ℕ)
variable (remaining_cards : ℕ)
variable (bought_cards : ℕ)

theorem benny_bought_two_cards 
  (h1 : initial_cards = 3) 
  (h2 : remaining_cards = 1) 
  (h3 : initial_cards - remaining_cards = bought_cards) : 
  bought_cards = 2 :=
by {
  rw [h1, h2] at h3,
  exact h3,
} 

end benny_bought_two_cards_l370_370759


namespace largest_integer_condition_l370_370875

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_integer_condition:
  ∃ x : ℤ, is_prime (|4 * x ^ 2 - 39 * x + 35|) ∧ x ≤ 6 :=
by sorry

end largest_integer_condition_l370_370875


namespace expression_eval_l370_370833

theorem expression_eval :
  (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) = 125 :=
by
  sorry

end expression_eval_l370_370833


namespace compare_logs_l370_370230

def a : ℝ := Real.logBase 5 2
def b : ℝ := Real.logBase 0.5 0.4
def c : ℝ := 2 / 5

theorem compare_logs : c < a ∧ a < b := by
  sorry

end compare_logs_l370_370230


namespace conjugate_z_l370_370667

-- Define the main problem
def find_conjugate (z : ℂ) : ℂ := conj z

-- Define the given complex number z
noncomputable def z : ℂ := (2 * complex.I) / (1 + complex.I)

-- State the theorem
theorem conjugate_z : find_conjugate z = 1 - complex.I :=
sorry

end conjugate_z_l370_370667


namespace pirates_treasure_l370_370424

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l370_370424


namespace find_z_l370_370036

theorem find_z (y z : ℝ) (k : ℝ) 
  (h1 : y = 3) (h2 : z = 16) (h3 : y ^ 2 * (z ^ (1 / 4)) = k)
  (h4 : k = 18) (h5 : y = 6) : z = 1 / 16 := by
  sorry

end find_z_l370_370036


namespace stock_quoted_rate_l370_370908

theorem stock_quoted_rate (investment dividend_rate earning : ℝ)
  (h_investment : investment = 1800)
  (h_dividend_rate : dividend_rate = 0.09)
  (h_earning : earning = 120) :
  (investment / (earning / dividend_rate)) * 100 = 135 :=
begin
  -- The proof will be done here.
  sorry
end

end stock_quoted_rate_l370_370908


namespace ice_cream_sphere_radius_l370_370164

theorem ice_cream_sphere_radius :
  let r_cone := 1
  let h_cone := 6
  let V_cone := (1/3) * π * r_cone^2 * h_cone
  let r_sphere := real.cbrt (3 / 2)
  V_cone = (4/3) * π * r_sphere^3

end ice_cream_sphere_radius_l370_370164


namespace total_students_in_class_l370_370744

-- Definitions based on the conditions
def volleyball_participants : Nat := 22
def basketball_participants : Nat := 26
def both_participants : Nat := 4

-- The theorem statement
theorem total_students_in_class : volleyball_participants + basketball_participants - both_participants = 44 :=
by
  -- Sorry to skip the proof
  sorry

end total_students_in_class_l370_370744


namespace valid_confirmation_codes_count_l370_370514

theorem valid_confirmation_codes_count : 
  -- condition 1: confirmation code is of length 5
  ∀ (codes : list ↥ (fin 10)) (h1 : codes.length = 5),
  -- condition 4: code cannot end with the sequence [0, 0, 5]
  (∀ (suffix : list ↥ (fin 10)), codes.length ≥ suffix.length → suffix = [0, 0, 5] → false) →
  -- proving the number of valid confirmation codes equals 99900
  (codes.count (λ l, l.length = 5 ∧ l != [0, 0, 5]) = 99900)
  ↔ true :=
by
  -- skipping the proof
  sorry

end valid_confirmation_codes_count_l370_370514


namespace log_conditions_l370_370706

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem log_conditions (m n : ℝ) (h₁ : log_base m 9 < log_base n 9)
  (h₂ : log_base n 9 < 0) : 0 < m ∧ m < n ∧ n < 1 :=
sorry

end log_conditions_l370_370706


namespace fuel_ethanol_problem_l370_370162

theorem fuel_ethanol_problem (x : ℝ) (h : 0.12 * x + 0.16 * (200 - x) = 28) : x = 100 := 
by
  sorry

end fuel_ethanol_problem_l370_370162


namespace education_savings_account_l370_370388

-- Definitions for conditions
def monthly_interest_rate : ℝ := 0.22 / 100
def initial_deposit : ℝ := 1000

-- The proof problem
theorem education_savings_account (x : ℕ) :
    let y := monthly_interest_rate * initial_deposit * x + initial_deposit in
    y = 2.2 * x + 1000 :=
sorry

end education_savings_account_l370_370388


namespace A_can_complete_work_in_15_days_l370_370934

noncomputable def A_days_to_complete_work (W : ℝ) : ℝ :=
  15

theorem A_can_complete_work_in_15_days (W : ℝ) (A_work_rate : ℝ) (B_work_rate : ℝ) (B_days : ℝ) :
  let B_work_rate := W / 27 in
  let A_work_rate := W / 15 in
  (5 * A_work_rate + 18 * B_work_rate = W) -> A_work_rate = W / 15 :=
by
  sorry

end A_can_complete_work_in_15_days_l370_370934


namespace quotient_of_sum_of_distinct_remainders_l370_370035

theorem quotient_of_sum_of_distinct_remainders (m : ℕ) (hm : 
  let
    sqr n := n * n
    remainders := finset.image (λ n, (sqr n % 13)) (finset.range 13).filter (λ n, 1 ≤ n ∧ n ≤ 12)
  in
  finset.sum remainders = m
) : m / 13 = 3 :=
by
  sorry

end quotient_of_sum_of_distinct_remainders_l370_370035


namespace find_total_coins_l370_370437

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l370_370437


namespace cubic_root_simplification_l370_370024

def simplify_cubic_root : ℚ :=
  let factorized := 2^4 * 7^3
  let simplified_inner := 2 ^ (4 / 3 : ℚ) * 7
  let integer_part_of_simplified := 2 * (2 ^ (1 / 3 : ℚ)) * 7
  in 10 * integer_part_of_simplified

theorem cubic_root_simplification : 
  simplify_cubic_root = 176.4 :=
by
  sorry

end cubic_root_simplification_l370_370024


namespace circle_through_point_tangent_to_given_circles_has_four_solutions_l370_370071

theorem circle_through_point_tangent_to_given_circles_has_four_solutions
    (P O1 O2 : Point)
    (circle1 : Circle O1) (circle2 : Circle O2)
    (S : Point) (homothety_center_lies_on_radical_axis : LiesOnRadicalAxis S circle1 circle2) :
    ∃ circle : Circle S, 
      PassesThrough circle P ∧ TangentTo circle circle1 ∧ TangentTo circle circle2 ∧ HasFourSolutions circle :=
sorry

end circle_through_point_tangent_to_given_circles_has_four_solutions_l370_370071


namespace log_properties_used_final_value_l370_370469

-- defining the basic properties of logarithms used in the problem
lemma log_property_one (b a : ℝ) (c : ℕ) (hb : 0 < b) (ha : 0 < a) : 
  real.log b (a ^ c) = c * real.log b a := by sorry

lemma log_property_two (c k: ℕ) (ha hb: 0 < c) :
  real.log c a / real.log c b = real.log b a := by sorry

noncomputable def simplified_first_sum : ℝ := 
  ∑ k in finset.range 20, k * real.log 5 3

noncomputable def simplified_second_sum : ℝ := 
  100 * real.log 9 25

noncomputable def converted_log : ℝ :=
  real.log 9 25

noncomputable def simplified_converted_log : ℝ := 
  real.log 3 5

theorem log_properties_used :
  (real.log 5 3 * real.log 3 5 = 1) := by sorry

theorem final_value :
  (\sum k in finset.range 20, real.log (5 ^ k) (3 ^ (k ^ 2))) * 
  (\sum k in finset.range 100, real.log (9 ^ k) (25 ^ k)) = 21000 := by 
  sorry

end log_properties_used_final_value_l370_370469


namespace lcm_of_12_15_18_is_180_l370_370887

theorem lcm_of_12_15_18_is_180 :
  Nat.lcm 12 (Nat.lcm 15 18) = 180 := by
  sorry

end lcm_of_12_15_18_is_180_l370_370887


namespace probability_C_and_D_l370_370928

theorem probability_C_and_D (P_A P_B : ℚ) (H1 : P_A = 1/4) (H2 : P_B = 1/3) :
  P_C + P_D = 5/12 :=
by
  sorry

end probability_C_and_D_l370_370928


namespace tan_beta_minus_pi_over_4_l370_370637

theorem tan_beta_minus_pi_over_4 (α β : ℝ) 
  (h1 : Real.tan (α + β) = 1/2) 
  (h2 : Real.tan (α + π/4) = -1/3) : 
  Real.tan (β - π/4) = 1 := 
sorry

end tan_beta_minus_pi_over_4_l370_370637


namespace percentage_increase_in_area_l370_370841

variables (l w : ℝ) (original_area new_area : ℝ)

-- Defining the original dimensions and their relationships
def original_length := l
def original_width := w
def original_area := original_length * original_width

-- Defining the new dimensions after the percentage increase
def new_length := 1.3 * original_length
def new_width := 1.2 * original_width
def new_area := new_length * new_width

-- The theorem proving the percentage increase in the area
theorem percentage_increase_in_area : 
  (new_area - original_area) / original_area * 100 = 56 := sorry

end percentage_increase_in_area_l370_370841


namespace inequality_system_solution_l370_370031

theorem inequality_system_solution (x : ℝ) (h1 : 2 + x > 7 - 4x) (h2 : x < (4 + x) / 2) : 1 < x ∧ x < 4 :=
by
  sorry -- Proof goes here

end inequality_system_solution_l370_370031


namespace polar_to_rectangular_intersection_distance_l370_370647

noncomputable def polar_curve := {θ : ℝ // ρ = (cos θ) / (sin θ)^2}
noncomputable def line_l := {t : ℝ // (x = (3/2) + t) ∧ (y = ( sqrt 3 * t))}

theorem polar_to_rectangular (θ : ℝ) (ρ : ℝ) : 
  ρ = (cos θ) / (sin θ)^2 → (exists x y : ℝ, y^2 = x) :=
  sorry

theorem intersection_distance : 
  (exists A B : (ℝ × ℝ), ∃ t : ℝ, A = (3/2 + t, sqrt 3 * t) ∧ B = (3/2 + -t, sqrt 3 * -t) ∧ (y^2 = x)) → 
  dist A B = 2 * sqrt 19 / 3 :=
  sorry

end polar_to_rectangular_intersection_distance_l370_370647


namespace pirates_treasure_l370_370430

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l370_370430


namespace find_asterisk_l370_370094

theorem find_asterisk : ∃ (x : ℕ), (63 / 21) * (x / 189) = 1 ∧ x = 63 :=
by
  sorry

end find_asterisk_l370_370094


namespace intersection_A_B_l370_370250

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℤ := {x | ∃ (r : ℝ), r = x ∧ r + 1 > 0 ∧ real.log (r + 1) < 1}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by
  sorry  -- The proof is omitted as per the instructions.

end intersection_A_B_l370_370250


namespace min_cut_length_isosceles_right_triangle_l370_370653

theorem min_cut_length_isosceles_right_triangle 
  (ABC : Triangle)
  (a b c : ℝ)
  (ha : a = sqrt 2)
  (hb : b = sqrt 2)
  (hc : c = sqrt 2 * sqrt 2)
  (right_angle_B : angle A B C = 90) :
  exists (cut_length : ℝ), cut_length = 0.91 :=
begin
  sorry
end

end min_cut_length_isosceles_right_triangle_l370_370653


namespace collinear_vectors_l370_370276

theorem collinear_vectors (λ : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (1, 0)) (hb : b = (1, 1))
  (h_collinear : (1 + λ, λ) = k • (λ + 1, 1)) : 
  λ = 1 ∨ λ = -1 :=
sorry

end collinear_vectors_l370_370276


namespace sum_of_digits_p_l370_370776

def sum_of_digits (x : ℕ) : ℕ :=
  x.digits 10 |>.sum

def T : set ℕ := {n | sum_of_digits n = 15 ∧ 0 ≤ n ∧ n < 10^8}

noncomputable def p : ℕ := T.to_finset.card

theorem sum_of_digits_p : sum_of_digits p = 33 :=
by sorry

end sum_of_digits_p_l370_370776


namespace pirates_treasure_l370_370414

theorem pirates_treasure :
  ∃ m : ℕ,
    (m / 3 + 1) +
    (m / 4 + 5) +
    (m / 5 + 20) = m ∧
    m = 120 :=
by {
  sorry,
}

end pirates_treasure_l370_370414


namespace exponent_multiplication_correct_l370_370474

theorem exponent_multiplication_correct (a : ℝ) : (a ^ 3) * (a ^ 3) = a ^ 6 :=
by {
  rw [<- add_assoc, mul_assoc],
  sorry
}

end exponent_multiplication_correct_l370_370474


namespace smallest_positive_period_increasing_interval_l370_370674

noncomputable def f (x : ℝ) : ℝ :=
  4 * tan x * sin (π / 2 - x) * cos (x - π / 3) - sqrt 3

theorem smallest_positive_period :
  ∀ (x : ℝ), f (x) = f (x + π) := sorry

theorem increasing_interval :
  ∀ (x : ℝ),
  -π / 4 ≤ x ∧ x ≤ π / 4 → 
  (∀ y : ℝ, -π / 12 ≤ y ∧ y ≤ π / 4 →
             (f y < f (y + ε) ∧ ε > 0 → y + ε ≤ π / 4)) := sorry

end smallest_positive_period_increasing_interval_l370_370674


namespace no_three_solutions_l370_370705

open Int

theorem no_three_solutions 
(a b c : ℤ)
(h1 : ∀ x, x ≡ a [ZMOD 14])
(h2 : ∀ x, x ≡ b [ZMOD 15])
(h3 : ∀ x, x ≡ c [ZMOD 16])
(h4 : ∀ x, 0 ≤ x ∧ x < 2000) :
  ∃ x, x ∧ x < 2000 → num_solutions h1 h2 h3 h4 ≠ 3 :=
sorry

end no_three_solutions_l370_370705


namespace find_k_l370_370074

theorem find_k (k : ℝ) (x : ℝ) :
  x^2 + k * x + 1 = 0 ∧ x^2 - x - k = 0 → k = 2 := 
sorry

end find_k_l370_370074


namespace max_sum_xyz_exist_inf_rational_triples_l370_370120

-- Part (a)
theorem max_sum_xyz (x y z: ℝ) (h1: 0 < x) (h2: 0 < y) (h3: 0 < z) 
    (h4 : 16 * x * y * z = (x + y)^2 * (x + z)^2) :  x + y + z ≤ 4 :=
by sorry

-- Part (b)
theorem exist_inf_rational_triples : 
    ∃∞ (x y z : ℚ), 16 * (x:ℝ) * (y:ℝ) * (z:ℝ) = ((x:ℝ) + (y:ℝ))^2 * ((x:ℝ) + (z:ℝ))^2 ∧ (x:ℝ) + (y:ℝ) + (z:ℝ) = 4 :=
by sorry

end max_sum_xyz_exist_inf_rational_triples_l370_370120


namespace range_g_l370_370684

noncomputable def g (x k : ℝ) : ℝ := x / (x^k + 1)

theorem range_g (k : ℝ) (hk : k > 0) : 
  set.range (λ x : ℝ, g x k) = set.Icc 0 (1 / 2) := 
by 
  sorry

end range_g_l370_370684


namespace find_m_l370_370696

def vector_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem find_m (m : ℝ) : let a := (2, m)
                          let b := (1, real.sqrt 3)
                          let ab_perp := dot_product (vector_add a b) (vector_sub a b) = 0
                        in ab_perp → m = 0 :=
by
  sorry

end find_m_l370_370696


namespace relationship_between_abc_l370_370638

def a : ℝ := (0.6 : ℝ) ^ (0.6 : ℝ)
def b : ℝ := (0.6 : ℝ) ^ (1.5 : ℝ)
def c : ℝ := (1.5 : ℝ) ^ (0.6 : ℝ)

theorem relationship_between_abc : b < a ∧ a < c := by
  sorry

end relationship_between_abc_l370_370638


namespace pieces_after_cuts_l370_370489

theorem pieces_after_cuts (n : ℕ) (h : n = 10) : (n + 1) = 11 := by
  sorry

end pieces_after_cuts_l370_370489


namespace linda_travel_distance_l370_370794

theorem linda_travel_distance
  (x : ℕ)
  (h1 : x = 1 ∨ x = 2 ∨ x = 4 ∨ x = 8 ∨ x = 16)
  (d1 := 120 / x : ℚ)
  (d2 := 120 / (2 * x) : ℚ)
  (d3 := 120 / (4 * x) : ℚ)
  (d4 := 120 / (8 * x) : ℚ)
  (d5 := 120 / (16 * x) : ℚ) :
  d1 + d2 + d3 + d4 + d5 = 232.5 := sorry

end linda_travel_distance_l370_370794


namespace smaller_squares_fit_cannot_fit_smaller_squares_l370_370967

noncomputable theory

open_locale classical

variables (K : set (ℝ × ℝ)) (side_length : ℝ) (smaller_squares : list (set (ℝ × ℝ))) (total_area : ℝ)

def is_square (s : set (ℝ × ℝ)) (l : ℝ) : Prop :=
  ∃ (x y : ℝ), s = {p : ℝ × ℝ | x ≤ p.1 ∧ p.1 ≤ x + l ∧ y ≤ p.2 ∧ p.2 ≤ y + l}

def areas_sum (sqrs : list (set (ℝ × ℝ))) : ℝ :=
  sqrs.sum (λ s, measure_theory.measure (classical.some (measure_theory.measure_space.measure s)) Icc (0, side_length))

def no_overlap (sqrs : list (set (ℝ × ℝ))) : Prop :=
  ∀ (s1 s2 ∈ sqrs), (s1 ≠ s2 → disjoint s1 s2)

theorem smaller_squares_fit : 
  is_square K 1 →
  (∀ s ∈ smaller_squares, is_square s (classical.some (measure_theory.measure_space.measure s))) →
  areas_sum smaller_squares ≤ 1/2 →
  ∃ placement : list (ℝ × ℝ), no_overlap (placement.zip smaller_squares) ∧
  (∀ p ∈ placement, ∃ q ∈ K, p.1 = q) := sorry

theorem cannot_fit_smaller_squares : 
  (∀ s ∈ smaller_squares, is_square s (classical.some (measure_theory.measure_space.measure s))) →
  areas_sum smaller_squares ≤ 1/2 →
  ∀ (a: ℝ), (a < 1 → ¬ ∃ placement : list (ℝ × ℝ), no_overlap (placement.zip smaller_squares) ∧ 
  (∀ p ∈ placement, ∃ q ∈ (set.univ : set (ℝ × ℝ)), p.1 = q)) := sorry

end smaller_squares_fit_cannot_fit_smaller_squares_l370_370967


namespace division_identity_l370_370608

theorem division_identity (abc : ℕ) (h : abc ≥ 100 ∧ abc < 1000) : 
  let n := 1001 * abc in ((n / 7) / 11) / 13 = abc := 
by 
  sorry

end division_identity_l370_370608


namespace samantha_interest_l370_370364

-- Definitions based on problem conditions
def P : ℝ := 2000
def r : ℝ := 0.08
def n : ℕ := 5

-- Compound interest calculation
noncomputable def A : ℝ := P * (1 + r) ^ n
noncomputable def Interest : ℝ := A - P

-- Theorem statement with Lean 4
theorem samantha_interest : Interest = 938.656 := 
by 
  sorry

end samantha_interest_l370_370364


namespace find_k_l370_370724

def rectangle (k : ℝ) : Prop :=
  let AB := (3, 1 : ℝ × ℝ)
  let AC := (2, k : ℝ × ℝ)
  let BC := (AC.1 - AB.1, AC.2 - AB.2)
  AB.1 * BC.1 + AB.2 * BC.2 = 0

theorem find_k (k : ℝ) (h : rectangle k) : k = 4 :=
by
  sorry

end find_k_l370_370724


namespace sodium_chloride_concentration_l370_370107

theorem sodium_chloride_concentration
  (initial_volume : ℕ)
  (initial_concentration : ℚ)
  (evaporated_volume : ℕ)
  (remaining_volume : ℕ)
  (initial_sodium_chloride : ℚ)
  (new_concentration : ℚ) :
  initial_volume = 10000 →
  initial_concentration = 5 →
  evaporated_volume = 5500 →
  remaining_volume = initial_volume - evaporated_volume →
  initial_sodium_chloride = initial_concentration / 100 * initial_volume →
  new_concentration = (initial_sodium_chloride / remaining_volume) * 100 →
  new_concentration ≈ 11.11 :=
by
  intros
  sorry

end sodium_chloride_concentration_l370_370107


namespace distance_to_point_A_farthest_distance_from_guard_post_total_fuel_consumption_l370_370131

def patrol_record := [10, -8, 6, -13, 7, -12, 3, -2] : List Int
def fuel_consumption_rate := 0.05 

theorem distance_to_point_A :
  patrol_record.sum = -9 :=
sorry

theorem farthest_distance_from_guard_post :
  patrol_record.scanl (+) 0 |>.map Int.natAbs |>.maximum = some 10 :=
sorry

theorem total_fuel_consumption :
  patrol_record.sumBy (fun x => abs x) * fuel_consumption_rate = 3.05 :=
sorry

end distance_to_point_A_farthest_distance_from_guard_post_total_fuel_consumption_l370_370131


namespace train_speed_l370_370155

def length_of_train : ℝ := 300 -- in meters
def time_to_cross_pole : ℝ := 7.4994000479961604 -- in seconds
def conversion_factor : ℝ := 3.6 -- 1 m/s to km/hr

theorem train_speed :
  length_of_train / time_to_cross_pole * conversion_factor ≈ 144.0192012 :=
by
  sorry

end train_speed_l370_370155


namespace cot_30_plus_cot_75_eq_2_l370_370907

noncomputable def cot (θ : ℝ) : ℝ := 1 / Real.tan θ

theorem cot_30_plus_cot_75_eq_2 : cot 30 + cot 75 = 2 := by sorry

end cot_30_plus_cot_75_eq_2_l370_370907


namespace sin_210_eq_neg_one_half_l370_370579

theorem sin_210_eq_neg_one_half :
  sin (Real.pi * (210 / 180)) = -1 / 2 :=
by
  have angle_eq : 210 = 180 + 30 := by norm_num
  have sin_30 : sin (Real.pi / 6) = 1 / 2 := by norm_num
  have cos_30 : cos (Real.pi / 6) = sqrt 3 / 2 := by norm_num
  sorry

end sin_210_eq_neg_one_half_l370_370579


namespace poly_sqrt_acute_triangle_l370_370333

noncomputable def is_triangular {α : Type*} [LinearOrderedField α] (a b c : α) : Prop :=
a < b + c ∧ b < a + c ∧ c < a + b

noncomputable def is_acute_angled {α : Type*} [LinearOrderedField α] (a b c : α) : Prop :=
a^2 < b^2 + c^2 ∧ b^2 < a^2 + c^2 ∧ c^2 < a^2 + b^2

theorem poly_sqrt_acute_triangle
  (P : ℝ → ℝ)
  (n : ℕ)
  (hn : n ≥ 2)
  (hP_nonneg : ∀ x, 0 ≤ P x)
  (hP_deg : ∀ x, P x = ∑ i in range (n+1), (P.coeff i) * ↑(x ^ i))
  (a b c : ℝ)
  (habc: is_acute_angled a b c)
  :
  is_acute_angled (P a)^(1/n) (P b)^(1/n) (P c)^(1/n)
:= 
sorry 

end poly_sqrt_acute_triangle_l370_370333


namespace rotation_90_deg_l370_370502

theorem rotation_90_deg (z : ℂ) (h : z = -6 - 3 * Complex.i) : Complex.i * z = 3 - 6 * Complex.i :=
by
  rw [h]
  simp
  sorry

end rotation_90_deg_l370_370502


namespace number_of_uncertain_events_is_three_l370_370049

noncomputable def cloudy_day_will_rain : Prop := sorry
noncomputable def fair_coin_heads : Prop := sorry
noncomputable def two_students_same_birth_month : Prop := sorry
noncomputable def olympics_2008_in_beijing : Prop := true

def is_uncertain (event: Prop) : Prop :=
  event ∧ ¬(event = true ∨ event = false)

theorem number_of_uncertain_events_is_three :
  is_uncertain cloudy_day_will_rain ∧
  is_uncertain fair_coin_heads ∧
  is_uncertain two_students_same_birth_month ∧
  ¬is_uncertain olympics_2008_in_beijing →
  3 = 3 :=
by sorry

end number_of_uncertain_events_is_three_l370_370049


namespace find_13th_result_l370_370910

theorem find_13th_result 
  (average_25 : ℕ → ℝ) (h1 : average_25 25 = 19)
  (average_first_12 : ℕ → ℝ) (h2 : average_first_12 12 = 14)
  (average_last_12 : ℕ → ℝ) (h3 : average_last_12 12 = 17) :
    let totalSum_25 := 25 * average_25 25
    let totalSum_first_12 := 12 * average_first_12 12
    let totalSum_last_12 := 12 * average_last_12 12
    let result_13 := totalSum_25 - totalSum_first_12 - totalSum_last_12
    result_13 = 103 :=
  by sorry

end find_13th_result_l370_370910


namespace complex_expression_l370_370998

theorem complex_expression : ( (1 + complex.i) / (1 - complex.i) ) ^ 2009 = complex.i :=
sorry

end complex_expression_l370_370998


namespace find_NC_length_l370_370034

open Real

-- Geometry definitions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def length (p1 p2 : Point) : ℝ :=
  sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def midpoint (p1 p2 : Point) : Point :=
  ⟨ (p1.x + p2.x) / 2, (p1.y + p2.y) / 2 ⟩

def square (p1 p2 p3 p4 : Point) : Prop :=
  length p1 p2 = length p2 p3 ∧ length p2 p3 = length p3 p4 ∧ length p3 p4 = length p4 p1 ∧
  (p2.x - p1.x)(p4.x - p3.x) + (p2.y - p1.y)(p4.y - p3.y) = 0 ∧
  (p3.x - p2.x)(p1.x - p4.x) + (p3.y - p2.y)(p1.y - p4.y) = 0

-- Given data
def P : Point := ⟨0, 0⟩
def Q : Point := ⟨8, 0⟩
def R : Point := ⟨8, 8⟩
def S : Point := ⟨0, 8⟩
def N : Point := ⟨4, 4 * (sqrt 2)⟩  -- This is approximated, generally needs more details
def C : Point := R

-- Definitions or statements
noncomputable def NC_length : ℝ := length N C

-- Theorem statement
theorem find_NC_length : NC_length = 4 * sqrt 5 :=
by
  sorry

end find_NC_length_l370_370034


namespace spherical_biangles_congruent_if_equal_angles_area_of_spherical_biangle_l370_370836

-- Definition of a spherical biangle
structure SphericalBiangle (R : ℝ) (α : ℝ) :=
(center : ℝ × ℝ × ℝ)
(radius : ℝ := R)
(angle : ℝ := α)

-- Problem (a): Prove that if two spherical biangles have equal angles, then they are congruent
theorem spherical_biangles_congruent_if_equal_angles :
  ∀ (R : ℝ) (α : ℝ) (Δ1 Δ2 : SphericalBiangle R α), Δ1.angle = Δ2.angle → Δ1 = Δ2 :=
by
  intros R α Δ1 Δ2 h
  sorry

-- Problem (b): Prove the formula for the area of a spherical biangle
theorem area_of_spherical_biangle :
  ∀ (R : ℝ) (α : ℝ), SphericalBiangle R α → ℝ
| R, α, biangle := 2 * R^2 * α :=
by
  intros R α biangle
  sorry

end spherical_biangles_congruent_if_equal_angles_area_of_spherical_biangle_l370_370836


namespace simple_interest_principal_l370_370851

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  let rate := r / 100
  P * (1 + rate) ^ t - P

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  (P * r * t) / 100

theorem simple_interest_principal :
  let CI := compound_interest 5000 15 3 in
  let SI := CI / 2 in
  let t := 4 in
  let r := 12.0 in
  simple_interest ?P r t = SI →
  ?P = 2712.89 :=
by
  sorry

end simple_interest_principal_l370_370851


namespace zeros_in_expansion_99999_cubed_l370_370252

theorem zeros_in_expansion_99999_cubed :
  ∀ (n: ℕ), (n > 0) → (∀ k, 99^k = 970299 → k = 2) →
  (∀ k, 999^k = 997002999 → k = 3) →
  ((99999)^3).digit_count (0) = 5 :=
by sorry

end zeros_in_expansion_99999_cubed_l370_370252


namespace increasing_function_D_on_0_2_l370_370160

def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂

def f_A (x : ℝ) : ℝ := Real.sqrt (2 - x)
def f_B (x : ℝ) : ℝ := 1 / (x - 2)
def f_C (x : ℝ) : ℝ := (1 / 2) ^ (x - 2)
def f_D (x : ℝ) : ℝ := Real.log (2 - x) / Real.log (1 / 2)

theorem increasing_function_D_on_0_2 :
  is_increasing_on f_D (set.Ioo 0 2) ∧
  ¬ is_increasing_on f_A (set.Ioo 0 2) ∧
  ¬ is_increasing_on f_B (set.Ioo 0 2) ∧
  ¬ is_increasing_on f_C (set.Ioo 0 2) :=
by sorry

end increasing_function_D_on_0_2_l370_370160


namespace part1_f_pi_by_6_part2_smallest_positive_period_part3_range_f_on_0_pi_by_2_l370_370673

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.cos x * Real.sin x + 2 * (Real.cos x) ^ 2 

theorem part1_f_pi_by_6 : f (Real.pi / 6) = 3 :=
by
  sorry

theorem part2_smallest_positive_period : 
  ∃ T > 0, (∀ x : ℝ, f(x + T) = f(x)) ∧ (∀ T' > 0, (∀ x : ℝ, f(x + T') = f(x)) → T ≤ T') :=
by
  use Real.pi
  sorry

theorem part3_range_f_on_0_pi_by_2 : 
  set.range (f ∘ λ x, x ∈ Icc 0 (Real.pi / 2)) = set.Icc 0 3 :=
by
  sorry

end part1_f_pi_by_6_part2_smallest_positive_period_part3_range_f_on_0_pi_by_2_l370_370673


namespace total_painting_cost_l370_370971

theorem total_painting_cost (n: ℕ) (south_first: ℕ) (south_diff: ℕ) (north_first: ℕ) (north_diff: ℕ) : 
  let south_adrs := fun k => south_first + south_diff * (k - 1) in
  let north_adrs := fun k => north_first + north_diff * (k - 1) in
  let south_digits := List.map (fun k => String.length (toString (south_adrs k))) [1, 20] in
  let north_digits := List.map (fun k => String.length (toString (north_adrs k))) [1, 20] in
  ∑ i, (south_digits + north_digits) = 84 :=
by
  sorry

end total_painting_cost_l370_370971


namespace sin_210_eq_neg_half_l370_370556

theorem sin_210_eq_neg_half : Real.sin (210 * Real.pi / 180) = -1 / 2 := by
  -- We use the given angles and their known sine values.
  have angle_30 := Real.pi / 6
  have sin_30 := Real.sin angle_30
  -- Expression for the sine of 210 degrees in radians.
  have angle_210 := 210 * Real.pi / 180
  -- Proving the sine of 210 degrees using angle addition formula and unit circle properties.
  calc
    Real.sin angle_210 
    -- 210 degrees is 180 + 30 degrees, translating to pi and pi/6 in radians.
    = Real.sin (Real.pi + Real.pi / 6) : by rw [←Real.ofReal_nat_cast, ←Real.ofReal_nat_cast, Real.ofReal_add, Real.ofReal_div, Real.ofReal_nat_cast]
    -- Using the sine addition formula: sin(pi + x) = -sin(x).
    ... = - Real.sin (Real.pi / 6) : by exact Real.sin_add_pi_div_two angle_30
    -- Substituting the value of sin(30 degrees).
    ... = - 1 / 2 : by rw sin_30

end sin_210_eq_neg_half_l370_370556


namespace equilateral_triangle_side_length_l370_370163

noncomputable theory

def side_length_of_equilateral_triangle_in_ellipse 
  (m n : ℕ) (hmn : Nat.Coprime m n) 
  (h_eq : (λ (x y : ℝ), x^2 + 4*y^2 = 4)) 
  (h_vertex : ∃ (x y : ℝ), x = 0 ∧ y = 1) 
  (h_altitude : ∃ (x : ℝ), x = 0) 
  : ℝ := 
  (Real.sqrt (m / n))

theorem equilateral_triangle_side_length (m n : ℕ) (hmn : Nat.Coprime m n) 
  (h_eq : ∀ (x y : ℝ), x^2 + 4*y^2 = 4)
  (h_vertex : ∃ (x y : ℝ), x = 0 ∧ y = 1)
  (h_altitude : ∃ (x : ℝ), x = 0) 
  (h_side_length : side_length_of_equilateral_triangle_in_ellipse m n hmn h_eq h_vertex h_altitude = Real.sqrt (3 / 1)) 
  : m + n = 4 
  :=
sorry

end equilateral_triangle_side_length_l370_370163


namespace perp_line_slope_l370_370466

def slope (x1 y1 x2 y2 : ℤ) : ℚ := (y2 - y1) / (x2 - x1)

def perpendicular_slope (m : ℚ) : ℚ := - (1 / m)

theorem perp_line_slope : 
  perpendicular_slope (slope 3 (-2) (-4) 3) = 7 / 5 :=
by
  sorry

end perp_line_slope_l370_370466


namespace base_n_multiple_of_five_count_l370_370628

theorem base_n_multiple_of_five_count :
  (n : ℕ) (2 ≤ n ∧ n ≤ 100) →
  (base_n_145236 : ℕ := 6 + 3 * n + 2 * n^2 + 5 * n^3 + 4 * n^4 + n^5) →
  (base_n_145236 % 5 = 0 ↔ (∃ m : ℕ, 1 ≤ m ∧ n = 5 * m)) :=
sorry

end base_n_multiple_of_five_count_l370_370628


namespace central_bank_interested_in_loyalty_program_banks_benefit_from_loyalty_program_registration_required_for_loyalty_program_l370_370122

-- Economic Incentives Conditions and Definitions
axiom loyalty_program : Type
axiom banks : Type
axiom customers : Type
axiom transactions : Type
axiom increased_transactions : loyalty_program → Prop
axiom financial_inclusion : loyalty_program → Prop
axiom customer_benefits : loyalty_program → Prop
axiom personalization : loyalty_program → Prop
axiom budget_control : loyalty_program → Prop
axiom central_bank_benefit (lp : loyalty_program) : increased_transactions lp → financial_inclusion lp → Prop
axiom bank_benefit (lp : loyalty_program) : customer_benefits lp → increased_transactions lp → Prop
axiom registration_necessity (lp : loyalty_program) : personalization lp → budget_control lp → Prop

-- Part (a): Prove the benefit to the Central Bank
theorem central_bank_interested_in_loyalty_program (lp : loyalty_program) :
  central_bank_benefit lp (increased_transactions lp) (financial_inclusion lp) :=
sorry

-- Part (b): Prove the benefit to the banks
theorem banks_benefit_from_loyalty_program (lp : loyalty_program) :
  bank_benefit lp (customer_benefits lp) (increased_transactions lp) :=
sorry

-- Part (c): Prove the necessity of registration
theorem registration_required_for_loyalty_program (lp : loyalty_program) :
  registration_necessity lp (personalization lp) (budget_control lp) :=
sorry

end central_bank_interested_in_loyalty_program_banks_benefit_from_loyalty_program_registration_required_for_loyalty_program_l370_370122


namespace q1_q2_q3_l370_370115

-- (1) Given |a| = 3, |b| = 1, and a < b, prove a + b = -2 or -4.
theorem q1 (a b : ℚ) (h1 : |a| = 3) (h2 : |b| = 1) (h3 : a < b) : a + b = -2 ∨ a + b = -4 := sorry

-- (2) Given rational numbers a and b such that ab ≠ 0, prove the value of (a/|a|) + (b/|b|) is 2, -2, or 0.
theorem q2 (a b : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) : (a / |a|) + (b / |b|) = 2 ∨ (a / |a|) + (b / |b|) = -2 ∨ (a / |a|) + (b / |b|) = 0 := sorry

-- (3) Given rational numbers a, b, c such that a + b + c = 0 and abc < 0, prove the value of (b+c)/|a| + (a+c)/|b| + (a+b)/|c| is -1.
theorem q3 (a b c : ℚ) (h1 : a + b + c = 0) (h2 : a * b * c < 0) : (b + c) / |a| + (a + c) / |b| + (a + b) / |c| = -1 := sorry

end q1_q2_q3_l370_370115


namespace find_other_number_l370_370837

def gcd (x y : Nat) : Nat := Nat.gcd x y
def lcm (x y : Nat) : Nat := Nat.lcm x y

theorem find_other_number (b : Nat) :
  gcd 360 b = 36 ∧ lcm 360 b = 8820 → b = 882 := by
  sorry

end find_other_number_l370_370837


namespace promotional_savings_l370_370966

noncomputable def y (x : ℝ) : ℝ :=
if x ≤ 500 then x
else if x ≤ 1000 then 500 + 0.8 * (x - 500)
else 500 + 400 + 0.5 * (x - 1000)

theorem promotional_savings (payment : ℝ) (hx : y 2400 = 1600) : 2400 - payment = 800 :=
by sorry

end promotional_savings_l370_370966


namespace angle_between_diagonals_is_60_degrees_l370_370147

-- Define the lengths of the sides of the quadrilateral.
def a : ℝ := 4 * real.sqrt 3
def b : ℝ := 9
def c : ℝ := real.sqrt 3

-- Define the angles between the sides in radians.
def θ_ab : ℝ := real.pi / 6 -- 30 degrees
def θ_bc : ℝ := real.pi / 2 -- 90 degrees

-- State the theorem about the angle between the diagonals of the quadrilateral.
theorem angle_between_diagonals_is_60_degrees :
  ∀ (a b c : ℝ) (θ_ab θ_bc : ℝ),
  a = 4 * real.sqrt 3 →
  b = 9 →
  c = real.sqrt 3 →
  θ_ab = real.pi / 6 →
  θ_bc = real.pi / 2 →
  ∃ (angle : ℝ), angle = real.pi / 3 := -- 60 degrees
by
  intros a b c θ_ab θ_bc ha hb hc hθ_ab hθ_bc
  use real.pi / 3
  sorry

end angle_between_diagonals_is_60_degrees_l370_370147


namespace seconds_between_flashes_l370_370141

-- Define the conditions
def flashes : ℕ := 450
def time_fraction_hour : ℚ := 3/4
def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60

-- Definition to convert fraction of an hour to seconds
def time_in_seconds (frac_hour : ℚ) : ℕ :=
  ((frac_hour * minutes_per_hour.to_rat) * seconds_per_minute.to_rat).to_nat

-- The theorem to be proved
theorem seconds_between_flashes : 
  time_in_seconds time_fraction_hour / flashes = 6 := 
sorry

end seconds_between_flashes_l370_370141


namespace neg_square_result_l370_370590

-- This definition captures the algebraic expression and its computation rule.
theorem neg_square_result (a : ℝ) : -((-3 * a) ^ 2) = -9 * (a ^ 2) := 
by
  sorry

end neg_square_result_l370_370590


namespace Maggie_bought_one_fish_book_l370_370795

-- Defining the variables and constants
def books_about_plants := 9
def science_magazines := 10
def price_book := 15
def price_magazine := 2
def total_amount_spent := 170
def cost_books_about_plants := books_about_plants * price_book
def cost_science_magazines := science_magazines * price_magazine
def cost_books_about_fish := total_amount_spent - (cost_books_about_plants + cost_science_magazines)
def books_about_fish := cost_books_about_fish / price_book

-- Theorem statement
theorem Maggie_bought_one_fish_book : books_about_fish = 1 := by
  -- Proof goes here
  sorry

end Maggie_bought_one_fish_book_l370_370795


namespace find_total_coins_l370_370435

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l370_370435


namespace cube_volume_is_27_l370_370800

theorem cube_volume_is_27 
    (a : ℕ) 
    (Vol_cube : ℕ := a^3)
    (Vol_new : ℕ := (a - 2) * a * (a + 2))
    (h : Vol_new + 12 = Vol_cube) : Vol_cube = 27 :=
by
    sorry

end cube_volume_is_27_l370_370800


namespace correct_calculation_l370_370475

-- Define the variables used in the problem
variables (a x y : ℝ)

-- The main theorem statement
theorem correct_calculation : (2 * x * y^2 - x * y^2 = x * y^2) :=
by sorry

end correct_calculation_l370_370475


namespace square_locus_arcs_l370_370490

variables {A B C D P : Type} [incidence_geometry A B C D] [plane_geometry P] -- assuming incidence_geometry and plane_geometry gives us necessary geometrical constructs

-- Define the square and the property we want to prove
def square_locus 
  (ABCD : quadrilateral P) 
  (square : is_square ABCD)

  (locus : P → Prop :=
     λ P, P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ P ≠ D ∧ (angle_sum P A B 180 ∧ angle_sum P C D 0))

theorem square_locus_arcs
  (ABCD : quadrilateral P) 
  (square : is_square ABCD) 
  (locus : P → Prop := 
    λ P, P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ P ≠ D ∧ (angle_sum P A B 180 ∧ angle_sum P C D 0))
  : ∃ locus_set, locus_set = (arc AB ∪ arc CD ∪ diagonal AC ∪ diagonal BD) \ {A, B, C, D} :=
sorry

end square_locus_arcs_l370_370490


namespace farm_problem_l370_370547

variable (H R : ℕ)

-- Conditions
def initial_relation : Prop := R = H + 6
def hens_updated : Prop := H + 8 = 20
def current_roosters (H R : ℕ) : ℕ := R + 4

-- Theorem statement
theorem farm_problem (H R : ℕ)
  (h1 : initial_relation H R)
  (h2 : hens_updated H) :
  current_roosters H R = 22 :=
by
  sorry

end farm_problem_l370_370547


namespace sin_210_eq_neg_half_l370_370572

theorem sin_210_eq_neg_half : sin (210 * real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_210_eq_neg_half_l370_370572


namespace octahedron_faces_incorrect_l370_370898

theorem octahedron_faces_incorrect : 
    ( ∀ (o : Octahedron), num_faces o = 8 )
    ∧ ( ∀ (t : Tetrahedron), ∃ (p1 p2 p3 p4 : Pyramid), t_is_cuts_into_4_pyramids t p1 p2 p3 p4 )
    ∧ ( ∀ (f : Frustum), extends_lateral_edges_intersect_at_point f )
    ∧ ( ∀ (r : Rectangle), rotated_around_side_forms_cylinder r ) 
    → ( "An octahedron has 10 faces" is incorrect ) :=
sorry

end octahedron_faces_incorrect_l370_370898


namespace math_proof_problem_l370_370509

noncomputable def problem_statement : Prop :=
  ∀ (x a b : ℕ), 
  (x + 2 = 5 ∧ x=3) ∧
  (60 / (x + 2) = 36 / x) ∧ 
  (a + b = 90) ∧ 
  (b ≥ 3 * a) ∧ 
  ( ∃ a_max : ℕ, (a_max ≤ a) ∧ (110*a_max + (30*b) = 10520))
  
theorem math_proof_problem : problem_statement := 
  by sorry

end math_proof_problem_l370_370509


namespace dihedral_angle_AD_eq_60_l370_370310

open Locale.Real

variables {V : Type*} [inner_product_space ℝ V]

/-- Definitions of the vertices A, B, C, D in the pyramid ABCD -/
variables (A B C D : V)

/-- Definitions of the conditions in the problem -/
variables (AC_perp : inner_product_space.angle ℝ (A - C) (B - C) = π / 2)
variables (AB_eq_BC : dist A B = dist B C)
variables (BC_eq_CD : dist B C = dist C D)
variables (BD_eq_AC : dist B D = dist A C)

/-- We aim to prove that the dihedral angle at the edge AD is 60 degrees -/
theorem dihedral_angle_AD_eq_60 (A B C D : V)
  (AC_perp : inner_product_space.angle ℝ (A - C) (B - C) = π / 2)
  (AB_eq_BC : dist A B = dist B C)
  (BC_eq_CD : dist B C = dist C D)
  (BD_eq_AC : dist B D = dist A C) :
  inner_product_space.angle ℝ (A - D) (B - D) = π / 3 := 
sorry

end dihedral_angle_AD_eq_60_l370_370310


namespace maximal_subset_cardinality_l370_370785

theorem maximal_subset_cardinality :
  ∃ S : Finset ℕ, S ⊆ Finset.range 101 ∧ (∀ a b ∈ S, a ≠ b → a ≠ 2 * b ∧ b ≠ 2 * a) ∧ S.card = 67 :=
sorry

end maximal_subset_cardinality_l370_370785


namespace ratio_of_areas_l370_370102

theorem ratio_of_areas (side_length : ℝ) (t q : ℝ) 
  (H1 : ∀ t, t = (sqrt 3) * side_length^2 / 16)
  (H2 : ∀ q, q = side_length^2 / 4 - (sqrt 3) * side_length^2 / 8) :
  q / t = 2 * sqrt 3 - 2 :=
by
  sorry

end ratio_of_areas_l370_370102


namespace length_AB_l370_370843

noncomputable def circle_eq (x y : ℝ) := x ^ 2 + y ^ 2 = 4
noncomputable def line_eq (x y : ℝ) := y = x + 2
noncomputable def chord_length (A B : ℝ × ℝ) := (dist A B)

theorem length_AB : ∀ A B : (ℝ × ℝ), 
  (circle_eq A.1 A.2) ∧ (circle_eq B.1 B.2) ∧ 
  (line_eq A.1 A.2) ∧ (line_eq B.1 B.2) → 
  chord_length A B = 2 * real.sqrt 2 :=
by
  intros A B h
  sorry

end length_AB_l370_370843


namespace largest_number_among_options_l370_370478

theorem largest_number_among_options :
  ∀ (x : ℝ) (y : ℝ) (z : ℝ) (w : ℝ) (v : ℝ),
  x = 0.999 -> y = 0.9099 -> z = 0.9991 -> w = 0.991 -> v = 0.9091 ->
  z > x ∧ z > y ∧ z > w ∧ z > v :=
by
  intros x y z w v hx hy hz hw hv
  rw [hx, hy, hz, hw, hv]
  split
  . exact lt_of_le_of_ne (le_of_eq rfl) (λ h, by linarith)
  split
  . exact (by linarith : 0.9991 > 0.9099)
  split
  . exact (by linarith : 0.9991 > 0.991)
  . exact (by linarith : 0.9991 > 0.9091)

end largest_number_among_options_l370_370478


namespace constant_slope_ratio_fixed_line_intersection_l370_370248

noncomputable def ellipse_c_eq : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (eccentricity a b) = (sqrt 3 / 2) ∧ right_focus a b = (sqrt 3, 0) ∧
  ellipse a b = set_of (λ p : ℝ × ℝ, (p.1^2 / 4) + p.2^2 = 1)

theorem constant_slope_ratio (a b : ℝ) (e : elliptical e a b = (sqrt 3 / 2)) (focus : elliptical e (sqrt 3, 0)) : Prop :=
  ∀ {P Q A B D : ℝ × ℝ} (k1 k2 : ℝ),
    (P ∈ ellipse a b) →
    (Q ∈ ellipse a b) →
    A = (-a, 0) →
    B = (a, 0) →
    D = (1, 0) →
    (line_intersects_c A P k1 k2) →
    (line_intersects_c B Q k2 k1) →
    (non_zero_slope A P) →
    (non_zero_slope B Q) →
    (k1 / k2) = (1 / 3)

theorem fixed_line_intersection (a b : ℝ) (e : elliptical e a b = (sqrt 3 / 2)) (focus : elliptical e (sqrt 3, 0)) : Prop :=
  ∀ {P Q A B D : ℝ × ℝ} (M : ℝ × ℝ) (k1 k2 : ℝ),
    (P ∈ ellipse a b) →
    (Q ∈ ellipse a b) →
    A = (-a, 0) →
    B = (a, 0) →
    D = (1, 0) →
    (line_intersects_c A P k1 k2) →
    (line_intersects_c B Q k2 k1) →
    (non_zero_slope A P) →
    (non_zero_slope B Q) →
    intersection_point A P B Q M →
    (M.1 = 4)

end constant_slope_ratio_fixed_line_intersection_l370_370248


namespace bridget_apples_l370_370158

variable (x : ℕ)

-- Conditions as definitions
def apples_after_splitting : ℕ := x / 2
def apples_after_giving_to_cassie : ℕ := apples_after_splitting x - 5
def apples_after_finding_hidden : ℕ := apples_after_giving_to_cassie x + 2
def final_apples : ℕ := apples_after_finding_hidden x
def bridget_keeps : ℕ := 6

-- Proof statement
theorem bridget_apples : x / 2 - 5 + 2 = bridget_keeps → x = 18 := by
  intros h
  sorry

end bridget_apples_l370_370158


namespace total_bales_stored_l370_370067

theorem total_bales_stored 
  (initial_bales : ℕ := 540) 
  (new_bales : ℕ := 2) : 
  initial_bales + new_bales = 542 :=
by
  sorry

end total_bales_stored_l370_370067


namespace interest_rate_other_investment_l370_370872

-- Define the given conditions
variables (money_earned_from_job last_year : ℝ)
variables (money_invested_in_9_percent : ℝ)
variables (money_invested_in_other_rate : ℝ)
variables (total_interest_earned : ℝ)
variables (investment_interest_rate : ℝ)

-- Given values
def walt_earnings : money_earned_from_job last_year = 9000 := by sorry
def investment_9_percent : money_invested_in_9_percent + money_invested_in_other_rate = money_earned_from_job last_year := by sorry
def investment_other_rate : money_invested_in_other_rate = 4000 := by sorry
def total_interest : total_interest_earned = 770 := by sorry

-- Prove that the interest rate for the other investment is 0.08
theorem interest_rate_other_investment : 
  0.09 * money_invested_in_9_percent + investment_interest_rate * money_invested_in_other_rate = total_interest_earned →
  investment_interest_rate = 0.08 :=
  by sorry

end interest_rate_other_investment_l370_370872


namespace min_value_of_square_sum_l370_370711

theorem min_value_of_square_sum (x y : ℝ) 
  (h1 : (x + 5) ^ 2 + (y - 12) ^ 2 = 14 ^ 2) : 
  x^2 + y^2 = 1 := 
sorry

end min_value_of_square_sum_l370_370711


namespace side_AC_of_triangle_l370_370914

theorem side_AC_of_triangle 
  {A B C : Type}
  (O K : Type)
  (r : ℝ)
  (h_radius : r = 1)
  (cos_b : ℝ)
  (h_cos_b : cos_b = 0.8)
  (H : O = center_of_inscribed_circle_in_triangle A B C )
  (K : K = point_of_tangency_on_side A B)
  (h1 : inscribed_circle_radius O r)
  (h2 : cos_angle B O = cos_b)
  (h3 : touches_median_parallel_to_side AC)
: side AC = 3 := 
sorry

end side_AC_of_triangle_l370_370914


namespace cubic_root_simplification_l370_370023

def simplify_cubic_root : ℚ :=
  let factorized := 2^4 * 7^3
  let simplified_inner := 2 ^ (4 / 3 : ℚ) * 7
  let integer_part_of_simplified := 2 * (2 ^ (1 / 3 : ℚ)) * 7
  in 10 * integer_part_of_simplified

theorem cubic_root_simplification : 
  simplify_cubic_root = 176.4 :=
by
  sorry

end cubic_root_simplification_l370_370023


namespace cos_A_proof_l370_370301

variables {A C : ℝ} {AB CD AD BC : ℝ}

-- Given conditions
def angles_equal (A C : ℝ) : Prop := A = C
def sides_equal (AB CD : ℝ) : Prop := AB = 180 ∧ CD = 180
def different_sides (AD BC : ℝ) : Prop := AD ≠ BC
def perimeter_equal (AB CD AD BC : ℝ) : Prop := AB + CD + AD + BC = 640

-- Prove that cos A = 7/9 under the given conditions
theorem cos_A_proof (A C AD BC : ℝ) 
  (h1 : angles_equal A C)
  (h2 : sides_equal AB CD)
  (h3 : different_sides AD BC)
  (h4 : perimeter_equal AB CD AD BC) :
  real.cos A = 7 / 9 := 
sorry

end cos_A_proof_l370_370301


namespace circumcircle_radius_of_right_triangle_l370_370624

theorem circumcircle_radius_of_right_triangle (r : ℝ) (BC : ℝ) (R : ℝ) 
  (h1 : r = 3) (h2 : BC = 10) : R = 7.25 := 
by
  sorry

end circumcircle_radius_of_right_triangle_l370_370624


namespace factor_expression_l370_370997

def expr1 : ℚ[X] := 20 * X^3 + 45 * X^2 - 10 - (-5 * X^3 + 15 * X^2 - 5)
def expr2 : ℚ[X] := 5 * (5 * X^3 + 6 * X^2 - 1)

theorem factor_expression : expr1 = expr2 := 
by sorry

end factor_expression_l370_370997


namespace treasures_coins_count_l370_370406

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l370_370406


namespace central_bank_interest_bank_benefit_registration_advantage_l370_370124

noncomputable theory

def interest_of_central_bank (increase_cashless: Prop) (promote_inclusion: Prop) : Prop :=
increase_cashless ∧ promote_inclusion

def benefit_for_banks (increase_clients: Prop) (increase_revenue: Prop) : Prop :=
increase_clients ∧ increase_revenue

def registration_requirement (personalization: Prop) (budget_control: Prop) : Prop :=
personalization ∧ budget_control

theorem central_bank_interest:
  ∀ (loyalty_program : Prop),
    (loyalty_program → interest_of_central_bank true true) :=
begin
  intros loyalty_program,
  split;
  assume h,
  { exact true.intro, },
  { exact true.intro, },
end

theorem bank_benefit:
  ∀ (loyalty_program : Prop),
    (loyalty_program → benefit_for_banks true true) :=
begin
  intros loyalty_program,
  split;
  assume h,
  { exact true.intro, },
  { exact true.intro, },
end

theorem registration_advantage:
  ∀ (loyalty_program : Prop),
    (loyalty_program → registration_requirement true true) :=
begin
  intros loyalty_program,
  split;
  assume h,
  { exact true.intro, },
  { exact true.intro, },
end

end central_bank_interest_bank_benefit_registration_advantage_l370_370124


namespace find_angle_x_l370_370735

-- Definitions as conditions from the problem statement
def angle_PQR := 120
def angle_PQS (x : ℝ) := 2 * x
def angle_QRS (x : ℝ) := x

-- The theorem to prove
theorem find_angle_x (x : ℝ) (h1 : angle_PQR = 120) (h2 : angle_PQS x + angle_QRS x = angle_PQR) : x = 40 :=
by
  sorry

end find_angle_x_l370_370735


namespace parabola_equation_proof_minimum_area_product_l370_370236

open Real

noncomputable def parabola_equation (p : ℝ) (h : p > 0) : Prop :=
  let F : Point := (0, p / 2)
  let P : Point := (4, 0)
  let Q : Point := (4, 8 / p)
  have QF_dist : |QF| = (8 / p) + (p / 2)
  have PQ_dist : |PQ| = 8 / p
  have PQ_dist_scaled : |QF| = 5 / 4 * |PQ|
  show p = 2, from sorry

theorem parabola_equation_proof (p : ℝ) (h : p > 0) : parabola_equation p h = (x^2 = 4 * y) := sorry

theorem minimum_area_product : 1 :=
  let F : Point := (0,1)
  let A = Point (x_A, y_A)
  let D = Point (x_D, y_D)
  let B = Point (x_B, y_B)
  let C = Point (x_C, y_C)
  let M = Point (2 * k, -1)
  let l = Line := λ x, k * x + 1
  let circle = Circle := λ x y, x^2 + (y - 1)^2 = 1
  have area_ABM := formula_area_ABM
  have area_CDM := formula_area_CDM
  have dist_M_l := distance_from_M_to_l
  show minimum_product_area = 1 from sorry

end parabola_equation_proof_minimum_area_product_l370_370236


namespace proof_problem_statement_l370_370741

noncomputable def parametric_equations (α : ℝ) : ℝ × ℝ :=
  (Math.sin α + Math.cos α, Math.sin α - Math.cos α)

noncomputable def cartesian_equation_of_curve_C (α : ℝ) : Prop :=
  let (x, y) := parametric_equations α in
  x^2 + y^2 = 2

noncomputable def line_eq_in_polar (ρ θ: ℝ) : Prop :=
  sqrt 2 * ρ * Math.sin (π / 4 - θ) + 1 / 2 = 0

noncomputable def distance_AB (A B : ℝ × ℝ) : ℝ :=
  let (xa, ya) := A in
  let (xb, yb) := B in
  Math.sqrt ((xa - xb)^2 + (ya - yb)^2)

noncomputable def AB_intersections_distance (ρ θ: ℝ) (A B: ℝ × ℝ) (hA: parametric_equations (1: ℝ) = A) (hB: parametric_equations (2: ℝ) = B) : Prop :=
  distance_AB A B = (sqrt 30) / 2

theorem proof_problem_statement :
  (∀ α : ℝ, cartesian_equation_of_curve_C α) ∧
  (∃ ρ θ A B, line_eq_in_polar ρ θ ∧ AB_intersections_distance ρ θ A B sorry sorry) :=
by sorry

end proof_problem_statement_l370_370741


namespace initial_investment_l370_370985

theorem initial_investment (A r : ℝ) (n t : ℕ) (H_A : A = 7260) (H_r : r = 0.10) (H_n : n = 1) (H_t : t = 2) : 
  ∃ (P : ℝ), P = 6000 :=
by
  -- Given: A = 7260, r = 0.1, n = 1, t = 2
  -- We will use the formula for compound interest: A = P * (1 + r / n) ^ (n * t)
  -- Substitute the given values to get the equation 7260 = P * 1.21
  let P := A / (1 + r / n) ^ (n * t)
  use P
  have : P = 7260 / 1.21, by sorry
  exact this

end initial_investment_l370_370985


namespace average_weight_women_l370_370400

variable (average_weight_men : ℕ) (number_of_men : ℕ)
variable (average_weight : ℕ) (number_of_women : ℕ)
variable (average_weight_all : ℕ) (total_people : ℕ)

theorem average_weight_women (h1 : average_weight_men = 190) 
                            (h2 : number_of_men = 8)
                            (h3 : average_weight_all = 160)
                            (h4 : total_people = 14) 
                            (h5 : number_of_women = 6):
  average_weight = 120 := 
by
  sorry

end average_weight_women_l370_370400


namespace percentage_required_to_pass_l370_370151

theorem percentage_required_to_pass :
  ∀ (marks_obtained failed_by max_marks : ℕ),
    marks_obtained = 100 →
    failed_by = 40 →
    max_marks = 400 →
    (let passing_marks := marks_obtained + failed_by,
         percentage := (passing_marks * 100) / max_marks
     in percentage = 35) :=
by
  intros marks_obtained failed_by max_marks h1 h2 h3
  let passing_marks := marks_obtained + failed_by
  let percentage := (passing_marks * 100) / max_marks
  sorry

end percentage_required_to_pass_l370_370151


namespace exists_between_elements_l370_370781

noncomputable def M : Set ℝ :=
  { x | ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ x = (m + n) / Real.sqrt (m^2 + n^2) }

theorem exists_between_elements (x y : ℝ) (hx : x ∈ M) (hy : y ∈ M) (hxy : x < y) :
  ∃ z ∈ M, x < z ∧ z < y :=
by
  sorry

end exists_between_elements_l370_370781


namespace pirates_treasure_l370_370428

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l370_370428


namespace determine_x_l370_370818

theorem determine_x (x : ℕ) (hx : x = 3^(Real.sqrt (2 + Real.log x / Real.log 3))) : x = 9 := 
by sorry

end determine_x_l370_370818


namespace exists_elem_in_union_of_min_cardinality_l370_370783

open Set

variable (A : Type) (S : Finset (Finset A))

theorem exists_elem_in_union_of_min_cardinality (n : ℕ) (h_n : 2 ≤ n) (hS : S.card = n)
  (h_distinct : ∀ s ∈ S, ∀ t ∈ S, s ≠ t → s ∪ t ∈ S )
  (h_min_card : ∃ k, (∀ s ∈ S, k ≤ s.card) ∧ 2 ≤ k) :
  ∃ x ∈ (⋃₀ S), (Finset.filter (λ s, x ∈ s) S).card ≥ n / (h_min_card.some) := 
sorry

end exists_elem_in_union_of_min_cardinality_l370_370783


namespace ginger_size_l370_370987

theorem ginger_size (anna_size : ℕ) (becky_size : ℕ) (ginger_size : ℕ) 
  (h1 : anna_size = 2) 
  (h2 : becky_size = 3 * anna_size) 
  (h3 : ginger_size = 2 * becky_size - 4) : 
  ginger_size = 8 :=
by
  -- The proof is omitted, just the theorem statement is required.
  sorry

end ginger_size_l370_370987


namespace max_value_of_y_l370_370212

noncomputable def max_value_of_function : ℝ := 1 + Real.sqrt 2

theorem max_value_of_y : ∀ x : ℝ, (2 * Real.sin x * (Real.sin x + Real.cos x)) ≤ max_value_of_function :=
by
  -- Proof goes here
  sorry

example : ∃ x : ℝ, (2 * Real.sin x * (Real.sin x + Real.cos x)) = max_value_of_function :=
by
  -- Proof goes here
  sorry

end max_value_of_y_l370_370212


namespace sin_210_eq_neg_one_half_l370_370580

theorem sin_210_eq_neg_one_half :
  sin (Real.pi * (210 / 180)) = -1 / 2 :=
by
  have angle_eq : 210 = 180 + 30 := by norm_num
  have sin_30 : sin (Real.pi / 6) = 1 / 2 := by norm_num
  have cos_30 : cos (Real.pi / 6) = sqrt 3 / 2 := by norm_num
  sorry

end sin_210_eq_neg_one_half_l370_370580


namespace probability_of_ending_at_multiple_of_3_l370_370317

noncomputable def probability_ends_at_multiple_of_3 : ℚ :=
let prob_start_multiple_3 := (5 / 15 : ℚ), -- Probability of starting at a multiple of 3
    prob_start_one_more_3 := (4 / 15 : ℚ), -- Probability of starting one more than a multiple of 3
    prob_start_one_less_3 := (5 / 15 : ℚ), -- Probability of starting one less than a multiple of 3
    prob_LL := (1 / 16 : ℚ),               -- Probability of "LL" outcome
    prob_RR := (9 / 16 : ℚ) in             -- Probability of "RR" outcome
  prob_start_multiple_3 * prob_LL +
  prob_start_one_more_3 * prob_RR +
  prob_start_one_less_3 * prob_LL

theorem probability_of_ending_at_multiple_of_3 :
  probability_ends_at_multiple_of_3 = (7 / 30 : ℚ) :=
sorry

end probability_of_ending_at_multiple_of_3_l370_370317


namespace arithmetic_sequence_solution_l370_370247

noncomputable def a (n : ℕ) : ℝ := sorry -- A placeholder for the arithmetic sequence function

def S (n : ℕ) : ℝ := (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_solution :
  (a 7 - 1)^3 + 2016 * (a 7 - 1) = -1 ∧
  (a 2010 - 1)^3 + 2016 * (a 2010 - 1) = 1 →
  S 2016 = 2016 ∧ a 2010 > a 7 :=
by {
  assume h,
  sorry -- Proof would go here
}

end arithmetic_sequence_solution_l370_370247


namespace problem_solution_l370_370931

noncomputable def sum_of_exponentiated_cards (k : ℕ) : ℕ :=
  2 * ((3^(k+1) - 1) / (3 - 1))

noncomputable def f (n : ℕ) : ℕ := 4^n - 1

noncomputable def P : ℕ := ∑ n in finset.range 1000, f(n)

theorem problem_solution : P = 6423 :=
sorry

end problem_solution_l370_370931


namespace perimeter_of_square_C_l370_370609

theorem perimeter_of_square_C :
  let side_length_A := 20 / 4  -- side length of square A is 5
  let side_length_B := 36 / 4  -- side length of square B is 9
  let side_length_C := side_length_B - side_length_A
  (4 * side_length_C = 16) := 
by
  let side_length_A := 20 / 4
  let side_length_B := 36 / 4
  let side_length_C := side_length_B - side_length_A
  show (4 * side_length_C = 16) from sorry

end perimeter_of_square_C_l370_370609


namespace side_length_of_square_l370_370361

theorem side_length_of_square (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (a b c : ℝ) (h_leg1 : a = 12) (h_leg2 : b = 9) (h_right : c^2 = a^2 + b^2) :
  ∃ s : ℝ, s = 45/8 :=
by 
  -- Given the right triangle with legs 12 cm and 9 cm, the length of the side of the square is 45/8 cm
  let s := 45/8
  use s
  sorry

end side_length_of_square_l370_370361


namespace greatest_distance_between_A_and_B_l370_370306

open Complex

def set_A : Set ℂ := {z | z^4 = 16}
def set_B : Set ℂ := {z | z^4 - 4*z^3 - 16*z^2 + 64*z = 0}

def max_distance (A B : Set ℂ) : ℝ :=
  Sup {dist a b | a ∈ A ∧ b ∈ B}

theorem greatest_distance_between_A_and_B :
  max_distance set_A set_B = Real.sqrt (28 + 16 * Real.sqrt 5) :=
by
  -- Proof should follow here.
  sorry

end greatest_distance_between_A_and_B_l370_370306


namespace calculate_EF_length_l370_370521

noncomputable def EF_length (AB CD GH : ℝ) : ℝ :=
  real.sqrt (CD * GH)

theorem calculate_EF_length (H_parallel : ∀ {A B C D E F G H : Point}, A ≠ B ∧ C ≠ D ∧ E ≠ F ∧ G ≠ H →
  AB ∥ CD ∧ CD ∥ EF ∧ EF ∥ GH)
  (h_AB : AB = 180)
  (h_CD : CD = 120)
  (h_GH : GH = 80) :
  EF = 24 * real.sqrt 5 :=
by 
  rw [EF_length, h_CD, h_GH]
  have h_EF_squared : EF * EF = 120 * 80 := by sorry
  have h_EF : EF = real.sqrt (120 * 80) := by sorry
  rw [h_EF]
  { rw [real.sqrt_mul (le_of_lt (by norm_num)), real.sqrt_eq_rpow, ←rpow_mul, inv_mul_cancel] }
  norm_num
  ring
  exact_mod_cast real.sqrt_nonneg 9600
  sorry

end calculate_EF_length_l370_370521


namespace pirates_treasure_l370_370426

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l370_370426


namespace convex_polygon_nonagon_l370_370984

theorem convex_polygon_nonagon (n : ℕ) (E I : ℝ) 
  (h1 : I = 3 * E + 180)
  (h2 : I + E = 180)
  (h3 : n * E = 360) : n = 9 :=
begin
  sorry
end

end convex_polygon_nonagon_l370_370984


namespace ellen_bakes_6_balls_of_dough_l370_370201

theorem ellen_bakes_6_balls_of_dough (rising_time baking_time total_time : ℕ) (h_rise : rising_time = 3) (h_bake : baking_time = 2) (h_total : total_time = 20) :
  ∃ n : ℕ, (rising_time + baking_time) + rising_time * (n - 1) = total_time ∧ n = 6 :=
by sorry

end ellen_bakes_6_balls_of_dough_l370_370201


namespace percentage_female_officers_on_duty_correct_l370_370801

-- Define the conditions
def total_officers_on_duty : ℕ := 144
def total_female_officers : ℕ := 400
def female_officers_on_duty : ℕ := total_officers_on_duty / 2

-- Define the percentage calculation
def percentage_female_officers_on_duty : ℕ :=
  (female_officers_on_duty * 100) / total_female_officers

-- The theorem that what we need to prove
theorem percentage_female_officers_on_duty_correct :
  percentage_female_officers_on_duty = 18 :=
by
  sorry

end percentage_female_officers_on_duty_correct_l370_370801


namespace students_neither_music_nor_art_l370_370902

theorem students_neither_music_nor_art :
  ∀ (total_students students_music students_art students_both : ℕ),
    total_students = 500 →
    students_music = 30 →
    students_art = 10 →
    students_both = 10 →
    total_students - (students_music + students_art - students_both) = 470 :=
by
  intros total_students students_music students_art students_both ht hsma hsa hsb
  simp [ht, hsma, hsa, hsb]
  sorry

end students_neither_music_nor_art_l370_370902


namespace find_a_monotonicity_l370_370682

def f (x a : ℝ) := x + a / x

-- Given: f(1, a) = 5

theorem find_a (a : ℝ) : f 1 a = 5 → a = 4 :=
by sorry

-- Given: f(x) = x + 4 / x

def g (x : ℝ) := f x 4

-- Prove: f is increasing on (2, +∞)

theorem monotonicity : ∀ x1 x2 : ℝ, 2 < x1 → x1 < x2 → g x1 < g x2 :=
by sorry

end find_a_monotonicity_l370_370682


namespace yellow_bags_count_l370_370805

theorem yellow_bags_count (R B Y : ℕ) 
  (h1 : R + B + Y = 12) 
  (h2 : 10 * R + 50 * B + 100 * Y = 500) 
  (h3 : R = B) : 
  Y = 2 := 
by 
  sorry

end yellow_bags_count_l370_370805


namespace sequence_value_l370_370242

theorem sequence_value (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, n > 0 → a n * a (n + 2) = a (n + 1) ^ 2)
  (h2 : a 7 = 16)
  (h3 : a 3 * a 5 = 4) : 
  a 3 = 1 := 
sorry

end sequence_value_l370_370242


namespace probability_of_two_points_one_unit_apart_l370_370870

open Probability

noncomputable def probability_two_points_one_unit_apart (n : ℕ) : ℚ :=
  if n = 12 then 2/11 else 0

theorem probability_of_two_points_one_unit_apart :
  probability_two_points_one_unit_apart 12 = 2 / 11 :=
by
  unfold probability_two_points_one_unit_apart
  split_ifs
  case h : h = rfl => rfl
  case h => contradiction

end probability_of_two_points_one_unit_apart_l370_370870


namespace sin_210_eq_neg_half_l370_370570

theorem sin_210_eq_neg_half : sin (210 * real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_210_eq_neg_half_l370_370570


namespace movement_left_3m_l370_370544

-- Define the condition
def movement_right_1m : ℝ := 1

-- Define the theorem stating that movement to the left by 3m should be denoted as -3
theorem movement_left_3m : movement_right_1m * (-3) = -3 :=
by
  sorry

end movement_left_3m_l370_370544


namespace quadratic_pos_implies_a_gt_1_l370_370595

theorem quadratic_pos_implies_a_gt_1 {a : ℝ} :
  (∀ x : ℝ, x^2 + 2 * x + a > 0) → a > 1 :=
by
  sorry

end quadratic_pos_implies_a_gt_1_l370_370595


namespace paper_fold_ratio_correct_l370_370149

def paper_fold_ratio (w : ℝ) (A B : ℝ) : Prop :=
  ∃ (w : ℝ) (A : ℝ) (B : ℝ),
  A = 2 * w^2 ∧ 
  B = A - (1 / 2 * (real.sqrt ((w^2 / 9) + 1)) * (1 / 2)) ∧ 
  (B / A) = 1 - (real.sqrt 10 / 24)

theorem paper_fold_ratio_correct : 
  paper_fold_ratio w (2 * w^2) (2 * w^2 - (1 / 2 * (real.sqrt ((w^2 / 9) + 1)) * (1 / 2))) :=
by
  sorry

end paper_fold_ratio_correct_l370_370149


namespace pirates_treasure_l370_370420

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l370_370420


namespace triplet_solution_l370_370614

theorem triplet_solution (x y z : ℝ) 
  (h1 : y = (x^3 + 12 * x) / (3 * x^2 + 4))
  (h2 : z = (y^3 + 12 * y) / (3 * y^2 + 4))
  (h3 : x = (z^3 + 12 * z) / (3 * z^2 + 4)) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 2 ∧ y = 2 ∧ z = 2) ∨ 
  (x = -2 ∧ y = -2 ∧ z = -2) :=
sorry

end triplet_solution_l370_370614


namespace number_of_diagonal_symmetric_configs_l370_370871

/-- A function to determine if a configuration has exact one diagonal symmetry -/
def has_one_diagonal_symmetry (config : list (ℕ × ℕ)) : Prop :=
  -- function implementation goes here
  sorry

/-- A definition for generating all configurations of 2x1 dominoes to form a 4x4 square -/
def generate_configurations : list (list (ℕ × ℕ)) :=
  -- function implementation goes here
  sorry

/-- The theorem to prove the number of distinct configurations with exactly one diagonal symmetry -/
theorem number_of_diagonal_symmetric_configs : 
  (generate_configurations.filter has_one_diagonal_symmetry).length = 8 :=
by sorry

end number_of_diagonal_symmetric_configs_l370_370871


namespace sarahs_trip_length_l370_370011

noncomputable def sarahsTrip (x : ℝ) : Prop :=
  x / 4 + 15 + x / 3 = x

theorem sarahs_trip_length : ∃ x : ℝ, sarahsTrip x ∧ x = 36 := by
  -- There should be a proof here, but it's omitted as per the task instructions
  sorry

end sarahs_trip_length_l370_370011


namespace function_has_minimum_value_in_interval_l370_370378

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + a

theorem function_has_minimum_value_in_interval (a : ℝ) :
  (∀ x ∈ Iio (1 : ℝ), ∃ y ∈ Iio (1 : ℝ), f y a < f x a) ↔ a < 1 := by 
sorry

end function_has_minimum_value_in_interval_l370_370378


namespace tan_sum_eq_double_tan_double_beta_l370_370246

-- Given conditions
variables {α β : ℝ}
variable h1 : 0 < α ∧ α < π / 2 -- α is an acute angle
variable h2 : 0 < β ∧ β < π / 2 -- β is an acute angle
variable h3 : tan (α - β) = sin (2 * β) -- tan(α - β) = sin(2β)

-- The proof goal
theorem tan_sum_eq_double_tan_double_beta (h1 : 0 < α ∧ α < π / 2)
    (h2 : 0 < β ∧ β < π / 2)
    (h3 : tan (α - β) = sin (2 * β)) :
  tan α + tan β = 2 * tan (2 * β) :=
sorry

end tan_sum_eq_double_tan_double_beta_l370_370246


namespace elmer_saves_on_fuel_l370_370607

-- Definitions of the conditions as provided in the problem
variables (x : ℝ) (c : ℝ)
-- Let x be the old car's fuel efficiency in kilometers per liter
-- Let c be the cost of fuel for the old car in dollars per liter

-- The new car's fuel efficiency is 75% better than the old car's
def new_fuel_efficiency := (7 / 4) * x
-- The new car's fuel is 40% more expensive
def new_fuel_cost := 1.4 * c
-- The distance of the trip
def distance := 300

-- Calculate the fuel cost for the old car
def old_car_fuel_cost := distance / x * c

-- Calculate the fuel cost for the new car
def new_car_fuel_cost := distance / new_fuel_efficiency * new_fuel_cost

-- Calculate the percentage savings
def savings_percentage := ((old_car_fuel_cost - new_car_fuel_cost) / old_car_fuel_cost) * 100

theorem elmer_saves_on_fuel :
  savings_percentage = 100 / 7 := by
  sorry

end elmer_saves_on_fuel_l370_370607


namespace second_player_min_n_l370_370382

theorem second_player_min_n (n : ℕ) :
  (∀ card_removed_by_first : ℕ, 1 ≤ card_removed_by_first ∧ card_removed_by_first ≤ n →
    ∃ two_consecutive_removed_by_second : ℕ × ℕ, (two_consecutive_removed_by_second.fst + 1 = two_consecutive_removed_by_second.snd) ∧ 
    (1 ≤ two_consecutive_removed_by_second.fst) ∧ (two_consecutive_removed_by_second.snd ≤ n) ∧ 
    ((∀ triplet_removed_by_first : ℕ × ℕ × ℕ, (triplet_removed_by_first.fst + 1 = triplet_removed_by_first.snd) ∧ 
       (triplet_removed_by_first.snd + 1 = triplet_removed_by_first.fst) ∧ 
       (1 ≤ triplet_removed_by_first.fst) ∧ (triplet_removed_by_first.fst ≤ n) ∧
       ∃ four_removed_by_second : ℕ × ℕ, (four_removed_by_second.fst + 1 = four_removed_by_second.snd) ∧ 
       (1 ≤ four_removed_by_second.fst) ∧ (four_removed_by_second.snd ≤ n)
    )))): 
  n = 14 := 
  sorry

end second_player_min_n_l370_370382


namespace simplify_expr1_simplify_expr2_l370_370813

-- Define the conditions and the expressions
variable (a x : ℝ)

-- Expression 1
def expr1 := 2 * (a - 1) - (2 * a - 3) + 3
def expr1_simplified := 4

-- Expression 2
def expr2 := 3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2)
def expr2_simplified := x^2 - 3 * x + 3

-- Prove the simplifications
theorem simplify_expr1 : expr1 a = expr1_simplified :=
by sorry

theorem simplify_expr2 : expr2 x = expr2_simplified :=
by sorry

end simplify_expr1_simplify_expr2_l370_370813


namespace relatively_prime_count_10_to_90_l370_370702

theorem relatively_prime_count_10_to_90 (a b n : ℕ) (h_a : a = 10) (h_b : b = 90) (h_n : n = 18) :
  let count := λ m : ℕ, (m / 2) + (m / 3) - (m / 6)
  in (b - a + 1 - (count b - count (a - 1))) = 27 := sorry

end relatively_prime_count_10_to_90_l370_370702


namespace decimal_to_base9_l370_370187

theorem decimal_to_base9 (n : ℕ) (h : n = 1729) : 
  (2 * 9^3 + 3 * 9^2 + 3 * 9^1 + 1 * 9^0) = n :=
by sorry

end decimal_to_base9_l370_370187


namespace division_addition_l370_370178

theorem division_addition :
  (-150 + 50) / (-50) = 2 := by
  sorry

end division_addition_l370_370178


namespace boxes_neither_pens_nor_pencils_l370_370010

def total_boxes : ℕ := 10
def pencil_boxes : ℕ := 6
def pen_boxes : ℕ := 3
def both_boxes : ℕ := 2

theorem boxes_neither_pens_nor_pencils : (total_boxes - (pencil_boxes + pen_boxes - both_boxes)) = 3 :=
by
  sorry

end boxes_neither_pens_nor_pencils_l370_370010


namespace roots_of_polynomial_sum_squares_l370_370777

noncomputable def a : ℂ := sorry
noncomputable def b : ℂ := sorry
noncomputable def c : ℂ := sorry

theorem roots_of_polynomial_sum_squares :
  (a + b) * (a + b) + (b + c) * (b + c) + (c + a) * (c + a) = 764 :=
by
  have h_roots : Polynomial.aeval a (Polynomial.C a + Polynomial.one) * 
                 Polynomial.aeval b (Polynomial.C b + Polynomial.one) * 
                 Polynomial.aeval c (Polynomial.C c + Polynomial.one) =
                    (Polynomial.X ^ 3 - 20 * Polynomial.X ^ 2 + 18 * Polynomial.X - 7) :=
    sorry
  
  have sum_roots : a + b + c = 20 := sorry
  have sum_roots_product_pairs : a * b + b * c + c * a = 18 := sorry

  sorry

end roots_of_polynomial_sum_squares_l370_370777


namespace min_pairs_l370_370356

noncomputable def Island :=
  structure Island where
    (resident : Type)
    (friend : resident → resident → Prop)

variables {Island}

def knight (r : Island.resident) : Prop := sorry -- define according to problem statement
def liar (r : Island.resident) : Prop := sorry -- define according to problem statement
def tells_all_friends_knights (r : Island.resident) : Prop := sorry -- define according to problem statement
def tells_all_friends_liars (r : Island.resident) : Prop := sorry -- define according to problem statement

axiom total_residents : 100 * 2 = 200
axiom counts : ∀ (I : Island),
  (∃ f, tells_all_friends_knights f) = 100 → (∃ f, tells_all_friends_liars f) = 100 

theorem min_pairs (I : Island) : ∃ P (knight_liar_pair : Island.resident → Island.resident → Prop),
  ∀ f, knight f ∨ liar f → friend f ≤ 1 :=
begin
  -- In actual theorem statement, formalize this
  sorry
end

end min_pairs_l370_370356


namespace quadratic_geometric_progression_has_two_roots_l370_370054

theorem quadratic_geometric_progression_has_two_roots 
  (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h₃ : ∃ q : ℝ, b = a * q ∧ c = a * q^2) : 
  ∃ x1 x2 : ℝ, a * x1^2 + 2 * sqrt 2 * b * x1 + c = 0 ∧ a * x2^2 + 2 * sqrt 2 * b * x2 + c = 0 ∧ x1 ≠ x2 :=
by
  sorry

end quadratic_geometric_progression_has_two_roots_l370_370054


namespace find_integer_sets_l370_370613

noncomputable def satisfy_equation (A B C : ℤ) : Prop :=
  A ^ 2 - B ^ 2 - C ^ 2 = 1 ∧ B + C - A = 3

theorem find_integer_sets :
  { (A, B, C) : ℤ × ℤ × ℤ | satisfy_equation A B C } = {(9, 8, 4), (9, 4, 8), (-3, 2, -2), (-3, -2, 2)} :=
  sorry

end find_integer_sets_l370_370613


namespace chris_and_dana_meet_probability_l370_370553

-- Define the problem conditions
def random_time_between_3_and_4 := set.Icc 0 60  -- Represent the time range as [0, 60] where 0 is 3:00 PM and 60 is 4:00 PM

-- Define a probability function
def meeting_probability (stay_duration : ℕ) (total_duration : ℕ) : ℚ :=
  let overlapping_area := (total_duration)^2 - 2 * (1/2 : ℚ) * (total_duration - stay_duration)^2 in
  overlapping_area / (total_duration^2)

-- The main theorem to prove the meeting probability is 5/9
theorem chris_and_dana_meet_probability :
  meeting_probability 20 60 = 5 / 9 :=
sorry

end chris_and_dana_meet_probability_l370_370553


namespace higher_rate_interest_diff_l370_370152

-- Define the principal, time, and rates
def principal : ℝ := 300
def time : ℕ := 10
def original_rate (R : ℝ) : ℝ := R
def higher_rate (R : ℝ) : ℝ := R + 5

-- Define the interest calculation at a given rate
def interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * time / 100

-- Define the original interest and higher interest calculations
def original_interest (R : ℝ): ℝ :=
  interest principal (original_rate R) time

def higher_interest (R : ℝ) : ℝ :=
  interest principal (higher_rate R) time

-- Define the difference in interest
def difference_in_interest (R : ℝ) : ℝ :=
  higher_interest R - original_interest R

-- The theorem to be proved
theorem higher_rate_interest_diff (R : ℝ) : difference_in_interest R = 150 :=
by
  sorry

end higher_rate_interest_diff_l370_370152


namespace geometric_sequence_common_ratio_l370_370719

-- Define the geometric sequence and conditions
variable {a : ℕ → ℝ}

def is_geometric_sequence (q : ℝ) : Prop :=
  ∀ n, a (n+1) = a n * q

def all_terms_positive : Prop :=
  ∀ n, a n > 0

def forms_arithmetic_sequence (a1 a2 a3 : ℝ) : Prop :=
  a1 + a3 = 2 * a2

noncomputable def common_ratio (q : ℝ) : Prop :=
  ∀ (a : ℕ → ℝ) (h_geom : is_geometric_sequence q) (h_pos : all_terms_positive), forms_arithmetic_sequence (3 * a 0) (2 * a 1) (1/2 * a 2) → q = 3

-- Statement of the theorem to prove
theorem geometric_sequence_common_ratio (q : ℝ) : common_ratio q := by
  sorry

end geometric_sequence_common_ratio_l370_370719


namespace pirates_treasure_l370_370417

theorem pirates_treasure :
  ∃ m : ℕ,
    (m / 3 + 1) +
    (m / 4 + 5) +
    (m / 5 + 20) = m ∧
    m = 120 :=
by {
  sorry,
}

end pirates_treasure_l370_370417


namespace polygon_n_sides_l370_370941

theorem polygon_n_sides (n : ℕ) (h : (n - 2) * 180 - x = 2000) : n = 14 :=
sorry

end polygon_n_sides_l370_370941


namespace symmetric_circle_equation_l370_370258

theorem symmetric_circle_equation :
  (∀ x y : ℝ, (x - 1) ^ 2 + y ^ 2 = 1 ↔ x ^ 2 + (y + 1) ^ 2 = 1) :=
by sorry

end symmetric_circle_equation_l370_370258


namespace solve_log_inequality_l370_370113

open Real

def valid_range (x : ℝ) : Prop :=
  (0 < x ∧ x ≠ 1 ∧ x ≠ 4 / 3)

def log_inequality (x : ℝ) : Prop :=
  log x (16 - 24 * x + 9 * x^2) < 0

theorem solve_log_inequality (x : ℝ) :
  valid_range x → log_inequality x ↔ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 4 / 3) ∨ (4 / 3 < x ∧ x < 5 / 3) :=
by
  intro h
  sorry

end solve_log_inequality_l370_370113


namespace systematic_sampling_result_l370_370536

theorem systematic_sampling_result :
  ∀ (total_students sample_size selected1_16 selected33_48 : ℕ),
  total_students = 800 →
  sample_size = 50 →
  selected1_16 = 11 →
  selected33_48 = selected1_16 + 32 →
  selected33_48 = 43 := by
  intros
  sorry

end systematic_sampling_result_l370_370536


namespace player_one_wins_l370_370858

def win_with_right_strategy (num_glasses : ℕ) (initial_cents : ℕ → ℕ) : Prop :=
  num_glasses = 100 ∧ 
  (∀ i, 0 ≤ i ∧ i < 100 → initial_cents i = 101 + i) →
  ∃ winning_strategy : (ℕ → ℕ) → ℕ → Prop, 
    (∃ first_player_wins : Prop, first_player_wins)

-- Statement proving that the first player will always win with the right strategy.
theorem player_one_wins : win_with_right_strategy 100 (λ i, 101 + i) :=
sorry

end player_one_wins_l370_370858


namespace ball_travel_distance_five_hits_l370_370153

def total_distance_traveled (h₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  let descents := List.range (n + 1) |>.map (λ i => h₀ * r ^ i)
  let ascents := List.range n |>.map (λ i => h₀ * r ^ (i + 1))
  (descents.sum + ascents.sum)

theorem ball_travel_distance_five_hits :
  total_distance_traveled 120 (3 / 4) 5 = 612.1875 :=
by
  sorry

end ball_travel_distance_five_hits_l370_370153


namespace volume_of_salt_in_flask_l370_370515

def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

def one_third_volume (v : ℝ) : ℝ :=
  v / 3

def salt_ratio : ℝ :=
  1 / 10

def approx_pi : ℝ :=
  3.14159

theorem volume_of_salt_in_flask :
  let r := 3 -- radius in inches (half of the diameter)
  let h := 9 -- height in inches
  let total_volume := volume_of_cone r h
  let full_sol_volume := one_third_volume total_volume
  let salt_volume := full_sol_volume * salt_ratio
  let result := salt_volume * approx_pi
  Float.round result 2 = 2.83 :=
by
  sorry

end volume_of_salt_in_flask_l370_370515


namespace inequality_abc_l370_370804

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end inequality_abc_l370_370804


namespace symmetric_point_l370_370732

/-- A point in a 3-dimensional Cartesian coordinate system -/
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_wrt_origin (A : Point3D) : Point3D :=
  ⟨-A.x, -A.y, -A.z⟩

theorem symmetric_point (A : Point3D) : 
  A = ⟨2, 1, -1⟩ → symmetric_wrt_origin A = ⟨-2, -1, 1⟩ := 
by
  intro h
  rw [h]
  dsimp [symmetric_wrt_origin]
  rfl

end symmetric_point_l370_370732


namespace exists_k_A_k_value_l370_370846

-- Definition of the function f and the sequence A_i
def f (A : ℕ) (digits : list ℕ) : ℕ :=
  digits.reverse.zipWith (λ (a : ℕ) (i : ℕ), a * 2^i) (list.range digits.length).sum

def A_sequence (A : ℕ) : ℕ → ℕ
| 0     := A
| (n+1) := f (A_sequence n) (nat.digits 10 (A_sequence n))

-- Problem 1: Exists k such that A_{k+1} = A_k
theorem exists_k (A : ℕ) : ∃ k, A_sequence A (k+1) = A_sequence A k :=
sorry

-- Problem 2: For A = 19^86, A_k = 19
theorem A_k_value (A : ℕ) (h : A = 19 ^ 86) : ∃ k, A_sequence A k = 19 :=
sorry

end exists_k_A_k_value_l370_370846


namespace area_of_triangle_l370_370669

open Real

-- Definitions of ellipse constants a, b, and c
def a : ℝ := 5
def b : ℝ := 3
def c : ℝ := sqrt(a^2 - b^2)

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Define distances from foci to any point P on the ellipse
def distances_from_foci (P F1 F2 : ℝ × ℝ) : ℝ × ℝ :=
  (sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2), sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))

-- Define the property that the sum of distances is constant
def sum_of_distances_property (P F1 F2 : ℝ × ℝ) : Prop :=
  let (t1, t2) := distances_from_foci P F1 F2 in
  t1 + t2 = 2 * a

-- Define the condition of the right triangle at point P
def right_triangle_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  let (t1, t2) := distances_from_foci P F1 F2 in
  t1^2 + t2^2 = (2 * c)^2

-- Define the triangle area function
def triangle_area (t1 t2 : ℝ) : ℝ :=
  (1 / 2) * t1 * t2

-- Main theorem to prove the area of the triangle is 9
theorem area_of_triangle (P F1 F2 : ℝ × ℝ)
  (h_ellipse : ellipse_equation P.1 P.2)
  (h_sum_dist : sum_of_distances_property P F1 F2)
  (h_right_angle : right_triangle_condition P F1 F2) :
  triangle_area (sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)) (sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)) = 9 :=
  sorry

end area_of_triangle_l370_370669


namespace percentage_temp_workers_l370_370112
-- Import necessary library

-- Define the conditions
variable (T : ℕ) -- Total number of workers
def P_t := 0.9 -- Percentage of technicians (90%)
def P_nt := 0.1 -- Percentage of non-technicians (10%)
def P_pt := 0.9 -- Percentage of permanent technicians (90%)
def P_pnt := 0.1 -- Percentage of permanent non-technicians (10%)

-- Define the number of permanent and temporary workers
def N_t := T * P_t -- Number of technicians
def N_nt := T * P_nt -- Number of non-technicians
def N_pt := N_t * P_pt -- Number of permanent technicians
def N_pnt := N_nt * P_pnt -- Number of permanent non-technicians
def N_perm := N_pt + N_pnt -- Total number of permanent workers
def N_temp := T - N_perm -- Total number of temporary workers

-- Calculate the percentage of temporary workers
def P_temp := (N_temp / T) * 100 -- Percentage of temporary workers

-- The proof problem: Prove that the percentage of temporary workers is 18%
theorem percentage_temp_workers : P_temp = 18 := by
  sorry

end percentage_temp_workers_l370_370112


namespace polygon_interior_angle_l370_370982

theorem polygon_interior_angle (n : ℕ) (hn : 3 * (180 - 180 * (n - 2) / n) + 180 = 180 * (n - 2) / n + 180) : n = 9 :=
by {
  sorry
}

end polygon_interior_angle_l370_370982


namespace linear_function_solution_l370_370712

-- Define the conditions 
def f (x : ℝ) : ℝ := sorry

-- Lean proof statement
theorem linear_function_solution :
  (∃ (k b : ℝ), k ≠ 0 ∧ f(x) = k*x + b ∧ (f (f x) = 4*x + 1)) →
  (f(x) = 2*x + (1 / 3) ∨ f(x) = -2*x - 1) :=
begin
  sorry
end

end linear_function_solution_l370_370712


namespace find_angle_B_l370_370347

variables {l k : Line}
variables {A B C D : Angle}

theorem find_angle_B
  (hlk : parallel l k)
  (m_angle_A : mangle A = 100)
  (m_angle_C : mangle C = 60) :
  mangle B = 100 :=
sorry

end find_angle_B_l370_370347


namespace sufficient_not_necessary_condition_l370_370655

noncomputable def setA (x : ℝ) : Prop := 
  (Real.log x / Real.log 2 - 1) * (Real.log x / Real.log 2 - 3) ≤ 0

noncomputable def setB (x : ℝ) (a : ℝ) : Prop := 
  (2 * x - a) / (x + 1) > 1

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, setA x → setB x a) ∧ (¬ ∀ x, setB x a → setA x) ↔ 
  -2 < a ∧ a < 1 := 
  sorry

end sufficient_not_necessary_condition_l370_370655


namespace angle_between_vectors_l370_370288

variable (a : ℝ × ℝ) (b : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2))
variable (theta : ℝ)

-- Given conditions
axiom magnitude_a : |a| = 2
axiom dot_product_equation : a.1 * (b.1 - a.1) + a.2 * (b.2 - a.2) + 2 = 0

-- The problem statement
theorem angle_between_vectors :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (|a| * |b|))
  in θ = Real.pi / 3 :=
sorry

end angle_between_vectors_l370_370288


namespace find_p_q_sum_l370_370970

-- Define the conditions
def is_fair_die (n : ℕ) := ∀ i, 1 ≤ i ∧ i ≤ n

def rolls_sum_to_prime (die1 : ℕ) (die2 : ℕ) (die3 : ℕ) (die4 : ℕ) : Prop :=
  Nat.Prime (die1 + die2 + die3 + die4)

-- State the theorem
theorem find_p_q_sum :
  let outcomes := [(a, b, c, d) | a <- fin 6, b <- fin 6, c <- fin 6, d <- fin 4],
      favorable_outcomes := [(a, b, c, d) ∈ outcomes | rolls_sum_to_prime a b c d],
      total_outcomes := 6^3 * 4,
      favorable_count := favorable_outcomes.length,
      p := favorable_count,
      q := total_outcomes - favorable_count 
      -- (p and q are relatively prime)
  in
  (Nat.coprime p q) → (p + q = 149) :=
by
  sorry

end find_p_q_sum_l370_370970


namespace vector_decomposition_l370_370916

def vector := ℝ × ℝ × ℝ

def x : vector := (1, -4, 4)
def p : vector := (2, 1, -1)
def q : vector := (0, 3, 2)
def r : vector := (1, -1, 1)

theorem vector_decomposition :
  x = (-1 : ℝ) • p + (0 : ℝ) • q + (3 : ℝ) • r :=
  sorry

end vector_decomposition_l370_370916


namespace degrees_of_PQR_are_even_l370_370274

open Polynomial

-- Define P, Q and R as polynomials with real coefficients, and non-zero
variable (P Q R : Polynomial ℝ)
variable (hP : P ≠ 0) (hQ : Q ≠ 0) (hR : R ≠ 0)

-- Conditions given in the problem
variable (h1 : P + Q + R = 0)
variable (h2 : P.eval₂ id Q + Q.eval₂ id R + R.eval₂ id P = 0)

-- The theorem to be proved
theorem degrees_of_PQR_are_even : 
  ∃ n m k, 
  (2 ∣ n) ∧ (2 ∣ m) ∧ (2 ∣ k) ∧ 
  (degree P = some n) ∧ 
  (degree Q = some m) ∧ 
  (degree R = some k) := 
  sorry

end degrees_of_PQR_are_even_l370_370274


namespace max_digit_occurrences_l370_370051

theorem max_digit_occurrences : 
  ∃ (n : ℕ), n = 67 ∧ 
  ∀ (a : ℕ), 
    ((100 ≤ a ∧ a ≤ 150) ∨ (200 ≤ a ∧ a ≤ 250) ∨ (300 ≤ a ∧ a ≤ 325)) →
    let digits := [10, 10, (51 + 1 + 5)] in  -- (example: digithundreds + digitens + digitunits)
    n ≥ maxdigits (frequency digits a) := 
  sorry

end max_digit_occurrences_l370_370051


namespace boys_girls_relation_l370_370991

theorem boys_girls_relation (b g : ℕ) :
  (∃ b, 3 + (b - 1) * 2 = g) → b = (g - 1) / 2 :=
by
  intro h
  sorry

end boys_girls_relation_l370_370991


namespace intensity_of_replacing_solution_l370_370033

theorem intensity_of_replacing_solution 
  (original_intensity : ℝ)
  (replaced_fraction : ℝ)
  (remaining_fraction : ℝ)
  (new_intensity : ℝ)
  (initial_intensity : ℝ) :
  original_intensity = 0.50 →
  replaced_fraction = 0.80 →
  remaining_fraction = 0.20 →
  new_intensity = 0.30 →
  initial_intensity = remaining_fraction * original_intensity + replaced_fraction * (0.25 : ℝ) →
  (0.25 : ℝ) = (initial_intensity - remaining_fraction * original_intensity) / replaced_fraction :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end intensity_of_replacing_solution_l370_370033


namespace largest_integer_condition_l370_370876

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_integer_condition:
  ∃ x : ℤ, is_prime (|4 * x ^ 2 - 39 * x + 35|) ∧ x ≤ 6 :=
by sorry

end largest_integer_condition_l370_370876


namespace initial_earning_members_l370_370822

theorem initial_earning_members (n : ℕ)
  (avg_income_initial : ℕ) (avg_income_after : ℕ) (income_deceased : ℕ)
  (h1 : avg_income_initial = 735)
  (h2 : avg_income_after = 590)
  (h3 : income_deceased = 1170)
  (h4 : 735 * n - 1170 = 590 * (n - 1)) :
  n = 4 :=
by
  sorry

end initial_earning_members_l370_370822


namespace kolya_walking_speed_is_correct_l370_370769

-- Conditions
def distance_traveled := 3 * x -- Total distance
def initial_speed := 10 -- Initial speed in km/h
def doubled_speed := 20 -- Doubled speed in km/h
def total_time_to_store_closing := distance_traveled / initial_speed -- Time to store's closing

-- Times for each segment
def time_first_segment := x / initial_speed
def time_second_segment := x / doubled_speed
def time_first_two_thirds := time_first_segment + time_second_segment
def remaining_time := total_time_to_store_closing - time_first_two_thirds

-- Prove Kolya's walking speed is 20/3 km/h
theorem kolya_walking_speed_is_correct :
  (x / remaining_time) = (20 / 3) :=
by
  sorry

end kolya_walking_speed_is_correct_l370_370769


namespace probability_A_selected_B_not_l370_370634
open BigOperators

def people : set ℕ := {1, 2, 3, 4}

theorem probability_A_selected_B_not (A B C D : ℕ) (hA : A ∈ people) (hB : B ∈ people) (hC : C ∈ people) (hD : D ∈ people) : 
  (∃ x y, x ≠ y ∧ {x, y} ⊆ people ∧ (x = A ∨ y = A) ∧ (x ≠ B ∧ y ≠ B)) →
  (∃ p, p = 1 / 3) :=
by
    sorry

end probability_A_selected_B_not_l370_370634


namespace inverse_exp_log_identity_l370_370663

theorem inverse_exp_log_identity (x : ℝ) (hx : 0 < x) : (λ (y : ℝ), Real.log (2 * y)) x = Real.log 2 + Real.log x :=
by
  sorry

end inverse_exp_log_identity_l370_370663


namespace incorrect_statement_among_ABCD_l370_370980

theorem incorrect_statement_among_ABCD :
  ¬ (-3 = Real.sqrt ((-3)^2)) :=
by
  sorry

end incorrect_statement_among_ABCD_l370_370980


namespace sum_of_p_digits_l370_370773

def sum_of_digits (x : ℕ) : ℕ :=
  x.digits 10 |>.sum

def T : set ℕ :=
  {n | sum_of_digits n = 15 ∧ 0 ≤ n ∧ n < 10^8}

def p : ℕ :=
  T.to_finset.card

theorem sum_of_p_digits : sum_of_digits p = 21 := by
  sorry

end sum_of_p_digits_l370_370773


namespace sin_210_eq_neg_one_half_l370_370589

theorem sin_210_eq_neg_one_half :
  ∀ (θ : ℝ), 
  θ = 210 * (π / 180) → -- angle 210 degrees
  ∃ (refθ : ℝ), 
  refθ = 30 * (π / 180) ∧ -- reference angle 30 degrees
  sin refθ = 1 / 2 → -- sin of reference angle
  sin θ = -1 / 2 := 
by
  intros θ hθ refθ hrefθ hrefθ_sin -- introduce variables and hypotheses
  sorry

end sin_210_eq_neg_one_half_l370_370589


namespace huang_yan_student_id_sum_l370_370101

def student_id_sum (year : Nat) (class_no : Nat) (number : Nat) (gender : Nat) : Nat :=
  let id_digits := List.digits 10 year ++ List.digits 10 class_no ++ List.digits 10 number ++ [gender]
  id_digits.foldr (· + ·) 0

/-- Given the conditions: 
   1. Zhang Chao's student ID number is 200608251.
   2. Zhang Chao enrolled in the year 2006.
   3. Zhang Chao is in class 08.
   4. Zhang Chao's number in the class is 25.
   5. Zhang Chao is male.
   6. Huang Yan enrolled this year.
   7. Huang Yan is assigned to class 12.
   8. Huang Yan's number in the class is 6.
   9. Huang Yan is female. 

Prove that the sum of the digits in Huang Yan's student ID number is 22.
-/
theorem huang_yan_student_id_sum : student_id_sum 2023 12 6 2 = 22 := by
  sorry

end huang_yan_student_id_sum_l370_370101


namespace sum_of_digits_p_l370_370775

def sum_of_digits (x : ℕ) : ℕ :=
  x.digits 10 |>.sum

def T : set ℕ := {n | sum_of_digits n = 15 ∧ 0 ≤ n ∧ n < 10^8}

noncomputable def p : ℕ := T.to_finset.card

theorem sum_of_digits_p : sum_of_digits p = 33 :=
by sorry

end sum_of_digits_p_l370_370775


namespace additional_miles_correct_l370_370507

def initial_fuel_efficiency := 33
def tank_capacity := 16
def solar_panel_efficiency := 4 / 3
def regenerative_braking_efficiency := 1 / 0.85
def hybrid_system_efficiency := 1 / 0.9

def total_efficiency_increase := solar_panel_efficiency * regenerative_braking_efficiency * hybrid_system_efficiency
def new_fuel_efficiency := initial_fuel_efficiency * total_efficiency_increase
def original_miles_per_tank := initial_fuel_efficiency * tank_capacity
def miles_per_tank_after_modifications := new_fuel_efficiency * tank_capacity

def additional_miles_per_tank := miles_per_tank_after_modifications - original_miles_per_tank
def expected_additional_miles := 391.54

theorem additional_miles_correct : additional_miles_per_tank ≈ expected_additional_miles :=
by
  sorry

end additional_miles_correct_l370_370507


namespace exists_nat_n_l370_370358

theorem exists_nat_n (l : ℕ) (hl : l > 0) : ∃ n : ℕ, n^n + 47 ≡ 0 [MOD 2^l] := by
  sorry

end exists_nat_n_l370_370358


namespace simplify_cubic_root_l370_370021

theorem simplify_cubic_root (a b : ℕ) (h1 : 2744000 = b) (h2 : ∛ b = 140) (h3 : 5488000 = 2 * b) :
  ∛ 5488000 = 140 * ∛ 2 :=
by
  sorry

end simplify_cubic_root_l370_370021


namespace arithmetic_sequence_ways_l370_370064

-- Defining the boxes and card numbers
def cards := {1, 2, 3, 4, 5, 6}

-- The question is to prove the number of ways to draw one card from each box so that the numbers form an arithmetic sequence
theorem arithmetic_sequence_ways :
  (∃ (a b c : ℕ), a ∈ cards ∧ b ∈ cards ∧ c ∈ cards ∧ (b - a = c - b ∨ a - b = b - c)) = 18 :=
sorry

end arithmetic_sequence_ways_l370_370064


namespace smallest_positive_period_cos_function_l370_370196

theorem smallest_positive_period_cos_function : 
  ∀ (x : ℝ), (1 - cos (2 * x)) = (1 - cos (2 * (x + π))) := sorry

end smallest_positive_period_cos_function_l370_370196


namespace geometric_sequence_common_ratio_l370_370721

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : 3 * a 0 + 2 * a 1 = a 2 / 0.5) :
  q = 3 :=
  sorry

end geometric_sequence_common_ratio_l370_370721


namespace smallest_n_satisfy_condition_l370_370216

open Set

namespace Problem

-- Define the main problem statement
theorem smallest_n_satisfy_condition : ∃ n : ℕ, (∀ (s : Set ℕ), (s = {i | i ∈ (Finset.range n).val}) →
  (∀ (p : Finset ℕ), p ⊆ (Finset.range n) → p ≠ ∅ → (∀ (A_1 A_2 ... A_{63} : Finset ℕ),
  (∀ i, A_i ⊆ p → disjoint A_i p) → (∀ i, A_i ≠ ∅) → 
  ∃ x y ∈ p,  x ≠ y ∧ x > y ∧ 31 * x ≤ 32 * y))) → n = 2016) :=
begin
  sorry
end

end Problem

end smallest_n_satisfy_condition_l370_370216


namespace find_b_when_remainder_is_constant_l370_370218

theorem find_b_when_remainder_is_constant (b : ℚ) :
  let p := 12 * (X ^ 3) - 9 * (X ^ 2) + (b * X) + 8
  let q := 3 * (X ^ 2) - 4 * X + 2
  let (quotient, remainder) := p.div_mod q
  remainder.degree = 0 → b = -4 / 3 :=
by
  sorry

end find_b_when_remainder_is_constant_l370_370218


namespace concurrency_ratios_product_l370_370312

theorem concurrency_ratios_product {X Y Z X' Y' Z' P : Type*}
  (h1 : Concurrent [XX', YY', ZZ']) 
  (h2 : ∀ {XP PX' YP PY' ZP PZ'}, 
        XP / PX' + YP / PY' + ZP / PZ' = 100)
  : (XP / PX') * (YP / PY') * (ZP / PZ') = 98 := by
  sorry

end concurrency_ratios_product_l370_370312


namespace ginger_size_l370_370988

theorem ginger_size (anna_size : ℕ) (becky_size : ℕ) (ginger_size : ℕ) 
  (h1 : anna_size = 2) 
  (h2 : becky_size = 3 * anna_size) 
  (h3 : ginger_size = 2 * becky_size - 4) : 
  ginger_size = 8 :=
by
  -- The proof is omitted, just the theorem statement is required.
  sorry

end ginger_size_l370_370988


namespace max_y_value_l370_370249

theorem max_y_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = (x - y) / (x + 3 * y)) : y ≤ 1 / 3 :=
by
  sorry

end max_y_value_l370_370249


namespace problem_distance_BC_l370_370716

noncomputable def length_of_BC (O B C D E : Point) (r : ℝ) : ℝ :=
  if h1 : Circle O r
  if h2 : Diameter O D E
  if h3 : Chord O B C D
  if h4 : DO = r
  if h5 : angleDBE = 90
  then 5
  else sorry

theorem problem_distance_BC (O B C D E : Point) (r : ℝ) (h_circle : Circle O r)
  (h_diameter : Diameter O D E) (h_chord : Chord O B C D) (h_radius : DO = r)
  (h_angle : angleDBE = 90) :
  length_of_BC O B C D E r = 5 :=
sorry

end problem_distance_BC_l370_370716


namespace smallest_sum_of_three_positives_l370_370396

open Finset

theorem smallest_sum_of_three_positives :
  let s := {2, 5, -7, 8, -10} : Finset ℤ
  let positive_numbers := s.filter (λ x => x > 0)
  positive_numbers.card ≥ 3 →
  positive_numbers.sum = 15 :=
by
  let s := {2, 5, -7, 8, -10} : Finset ℤ
  let positive_numbers := s.filter (λ x => x > 0)
  have h1 : positive_numbers = {2, 5, 8} := sorry
  have h2 : positive_numbers.card = 3 := sorry
  have h3 : positive_numbers.sum = 15 := by 
    simp [positive_numbers]
    sorry
  exact h3

end smallest_sum_of_three_positives_l370_370396


namespace john_sells_each_apple_for_50_cents_l370_370322

theorem john_sells_each_apple_for_50_cents
  (trees_in_row : ℕ) (trees_in_col : ℕ) (apples_per_tree : ℕ) (total_revenue : ℤ)
  (H1 : trees_in_row = 3)
  (H2 : trees_in_col = 4)
  (H3 : apples_per_tree = 5)
  (H4 : total_revenue = 30) :
  (total_revenue.to_rat / (trees_in_row * trees_in_col * apples_per_tree)) = 0.5 :=
by sorry

end john_sells_each_apple_for_50_cents_l370_370322


namespace inequality_one_inequality_system_l370_370550

-- Definition for the first problem
theorem inequality_one (x : ℝ) : 3 * x > 2 * (1 - x) ↔ x > 2 / 5 :=
by
  sorry

-- Definitions for the second problem
theorem inequality_system (x : ℝ) : 
  (3 * x - 7) / 2 ≤ x - 2 ∧ 4 * (x - 1) > 4 ↔ 2 < x ∧ x ≤ 3 :=
by
  sorry

end inequality_one_inequality_system_l370_370550


namespace sum_lengths_intervals_l370_370334

def g (x : ℝ) : ℝ := (⌊x⌋ : ℝ) * (2013 ^ (x - ⌊x⌋) - 2)

theorem sum_lengths_intervals : 
  (∑ k in finset.range 2012, (real.log 2 / real.log 2013)) = 2012 * real.log 2 / real.log 2013 :=
by
  sorry

end sum_lengths_intervals_l370_370334


namespace part1_part2i_part2ii_l370_370686

def hyperbola (x y : ℝ) := x^2 - y^2 / 3 = 1
def line (k m x y : ℝ) := y = k * x + m
def point_on_line (x y k m : ℝ) := y = k * x + m
def unique_common_point (k m : ℝ) : Prop :=
  (k ≠ sqrt 3) ∧ (k ≠ -sqrt 3) ∧ (¬∃ x y, hyperbola x y ∧ line k m x y)

def problem_1 : Prop :=
  ∃ k m, unique_common_point k m ∧ point_on_line 2 3 k m ∧ (∀ x y, line k m x y → y = 2 * x - 1)

def incenter_abscissa_constant : Prop :=
  let xM := 1
  in ∀ F1 F2 A, (A.1 > 0) ∧ (tri_incenter AF1 F2 = xM)

def slope_sum (θ : ℝ) : ℝ := 
  -tan θ + 1 / tan θ

def problem_2_ii : Prop :=
  ∀ θ : ℝ, θ ∈ (π / 6, π / 3) → 
    slope_sum θ ∈ (-2 * sqrt 3 / 3, 2 * sqrt 3 / 3)

theorem part1 : problem_1 := sorry
theorem part2i : incenter_abscissa_constant := sorry
theorem part2ii : problem_2_ii := sorry

end part1_part2i_part2ii_l370_370686


namespace valid_addends_l370_370077

noncomputable def is_valid_addend (n : ℕ) : Prop :=
  ∃ (X Y : ℕ), (100 * 9 + 10 * X + 4) = n ∧ (30 + Y) ∈ [36, 30, 20, 10]

theorem valid_addends :
  ∀ (n : ℕ),
  is_valid_addend n ↔ (n = 964 ∨ n = 974 ∨ n = 984 ∨ n = 994) :=
by
  sorry

end valid_addends_l370_370077


namespace ending_number_of_range_l370_370399

theorem ending_number_of_range (n : ℕ) (h : ∃ k, 1 ≤ k ∧ k ≤ 5 ∧ n = 29 + 11 * k) : n = 77 := by
  sorry

end ending_number_of_range_l370_370399


namespace coefficient_of_x21_in_expansion_l370_370308

theorem coefficient_of_x21_in_expansion : 
  let f := (1 + ∑ i in Finset.range 21, x^i) *
           (1 + ∑ i in Finset.range 11, x^i)^2 *
           (1 - x^3) in
  (f.coeff 21) = 63 := 
by 
  sorry

end coefficient_of_x21_in_expansion_l370_370308


namespace sum_of_integer_values_l370_370084

theorem sum_of_integer_values (n : ℤ) (h : ∃ k : ℤ, 30 = k * (2 * n - 1)) : 
  ∑ n in {n | ∃ k : ℤ, 30 = k * (2 * n - 1)}, n = 14 :=
by
  sorry

end sum_of_integer_values_l370_370084


namespace john_finishes_sixth_task_at_305pm_l370_370762

noncomputable def time_per_task (total_time_minutes : ℕ) (num_tasks : ℕ) : ℚ :=
  total_time_minutes / num_tasks

noncomputable def total_time_after_breaks (num_tasks : ℕ) (time_per_task_minutes : ℚ) (break_minutes : ℕ) : ℚ :=
  num_tasks * time_per_task_minutes + (num_tasks - 1) * break_minutes

theorem john_finishes_sixth_task_at_305pm
  (start_time : ℕ) -- 9:00 AM in minutes, e.g., 540 minutes past midnight
  (end_time : ℕ) -- 1:00 PM in minutes, e.g., 780 minutes past midnight
  (break_minutes : ℕ) -- 10-minute break
  (tasks_done : ℕ) -- Number of tasks done by 1:00 PM
  (total_breaks_done : ℕ) -- Number of breaks taken by 1:00 PM
  : (end_time - start_time - total_breaks_done * break_minutes) / tasks_done = (end_time - start_time - total_breaks_done * break_minutes) / tasks_done →
    let time_per_task := (end_time - start_time - total_breaks_done * break_minutes) / tasks_done in
    total_time_after_breaks 6 time_per_task break_minutes + start_time = 965 := -- 965 minutes is 3:05 PM
by
  sorry

end john_finishes_sixth_task_at_305pm_l370_370762


namespace scooter_value_depreciation_l370_370856

theorem scooter_value_depreciation (V0 Vn : ℝ) (rate : ℝ) (n : ℕ) 
  (hV0 : V0 = 40000) 
  (hVn : Vn = 9492.1875) 
  (hRate : rate = 3 / 4) 
  (hValue : Vn = V0 * rate ^ n) : 
  n = 5 := 
by 
  -- Conditions are set up, proof needs to be constructed.
  sorry

end scooter_value_depreciation_l370_370856


namespace find_total_coins_l370_370438

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l370_370438


namespace simple_interest_problem_l370_370109

theorem simple_interest_problem (P : ℝ) (R : ℝ) (T : ℝ) : T = 10 → 
  ((P * R * T) / 100 = (4 / 5) * P) → R = 8 :=
by
  intros hT hsi
  sorry

end simple_interest_problem_l370_370109


namespace gardener_cabbage_difference_l370_370520

theorem gardener_cabbage_difference :
  ∀ (production_last_year : ℕ), production_last_year = 81^2 →
  (∀ (production_this_year : ℕ), production_this_year = 91^2 →
  production_this_year - production_last_year = 1720) :=
by
  intros
  simp
  sorry

end gardener_cabbage_difference_l370_370520


namespace tank_capacity_is_5760_l370_370525

-- Questions and conditions definitions
def capacity_of_tank (C : ℕ) : Prop :=
  let leak_rate := C / 6
  let inlet_rate := 240 -- in liters per hour
  let net_empty_rate := C / 8
  leak_rate - inlet_rate = net_empty_rate

-- Statement to be proven
theorem tank_capacity_is_5760 : capacity_of_tank 5760 :=
by
  have leak_rate := 5760 / 6
  have inlet_rate := 240
  have net_empty_rate := 5760 / 8
  calc
    leak_rate - inlet_rate = 5760 / 6 - 240 := by rfl
    ... = 960 - 240 := by norm_num
    ... = 720 := by norm_num
    ... = 5760 / 8 := by norm_num
#align 
end tank_capacity_is_5760_l370_370525


namespace dispatch_plans_count_l370_370947

-- Definitions for the conditions in Lean
def numVehicles : ℕ := 7
def numDispatch : ℕ := 4
def vehicles := {A, B, C, D, E, F, G}

-- Defining the requirement that A and B must be dispatched, and A must be before B
def validDispatches (perm : List Char) : Prop :=
  perm.length = numDispatch ∧
  'A' ∈ perm ∧
  'B' ∈ perm ∧
  perm.indexOf 'A' < perm.indexOf 'B'

-- Calculation of valid dispatch plans
theorem dispatch_plans_count : 
  (perm : List Char) → validDispatches perm → perm.length = numDispatch → 120 :=
by sorry

end dispatch_plans_count_l370_370947


namespace probability_C_and_D_l370_370927

theorem probability_C_and_D (P_A P_B : ℚ) (H1 : P_A = 1/4) (H2 : P_B = 1/3) :
  P_C + P_D = 5/12 :=
by
  sorry

end probability_C_and_D_l370_370927


namespace stratified_sampling_example_l370_370940

theorem stratified_sampling_example 
  (N : ℕ) (S : ℕ) (D : ℕ) 
  (hN : N = 1000) (hS : S = 50) (hD : D = 200) : 
  D * (S : ℝ) / (N : ℝ) = 10 := 
by
  sorry

end stratified_sampling_example_l370_370940


namespace ratio_B_over_A_eq_one_l370_370381

theorem ratio_B_over_A_eq_one (A B : ℤ) (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 3 → 
  (A : ℝ) / (x + 3) + (B : ℝ) / (x * (x - 3)) = (x^3 - 3*x^2 + 15*x - 9) / (x^3 + x^2 - 9*x)) :
  (B : ℝ) / (A : ℝ) = 1 :=
sorry

end ratio_B_over_A_eq_one_l370_370381


namespace treasure_coins_l370_370441

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l370_370441


namespace smallest_yummy_is_minus_2013_l370_370012

-- Define a yummy integer
def is_yummy (A : ℤ) : Prop :=
  ∃ (k : ℕ), ∃ (a : ℤ), (a <= A) ∧ (a + k = A) ∧ ((k + 1) * A - k*(k + 1)/2 = 2014)

-- Define the smallest yummy integer
def smallest_yummy : ℤ :=
  -2013

-- The Lean theorem to state the proof problem
theorem smallest_yummy_is_minus_2013 : ∀ A : ℤ, is_yummy A → (-2013 ≤ A) :=
by
  sorry

end smallest_yummy_is_minus_2013_l370_370012


namespace roots_of_cubic_l370_370038

theorem roots_of_cubic (p q : ℤ) (h₀ : p ≠ 0) (h₁ : q ≠ 0)
  (h₂ : ∃ r s : ℕ, r > 0 ∧ s > 0 ∧ (x^3 + p * x^2 + q * x - 15 * p).roots = [r, r, s]) :
  |p * q| = 21 := 
sorry

end roots_of_cubic_l370_370038


namespace final_coordinates_l370_370390

-- Define the types of transformations used
def rotate180_x (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, -p.2, -p.3)
def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, -p.2, p.3)
def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (-p.1, p.2, p.3)
def rotate180_z (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (-p.1, -p.2, p.3)

-- The main proof statement
theorem final_coordinates (p : ℝ × ℝ × ℝ) :
  rotate180_x p = (2,2,2) →
  reflect_xz (rotate180_x p) = (2,2,-2) →
  reflect_yz (reflect_xz (rotate180_x p)) = (-2,2,-2) →
  rotate180_z (reflect_yz (reflect_xz (rotate180_x p))) = (2,-2,-2) →
  reflect_yz (rotate180_z (reflect_yz (reflect_xz (rotate180_x p)))) = (-2,-2,-2) :=
begin
  intros h1 h2 h3 h4,
  sorry
end

end final_coordinates_l370_370390


namespace select_doctors_probability_l370_370137

theorem select_doctors_probability
  (I S : ℕ) -- Define the number of internists and surgeons
  (hI : I = 5) -- There are 5 internists
  (hS : S = 6) -- There are 6 surgeons
  (N : ℕ) -- Define the total number of doctors to be dispatched
  (hN : N = 4) -- 4 doctors need to be dispatched
  : 
  ∑ (total_ways : ℕ) :=
  ∑(ways_internists : ℕ) :=
  ((choose (I + S) N) - ((choose I N) + (choose S N))) = 310 := -- Calculate total ways to dispatch doctors

by sorry -- The proof is not provided

end select_doctors_probability_l370_370137


namespace pentagonal_pyramid_sphere_radii_l370_370535

def radius_of_inscribed_sphere (base_edge_length : ℝ) (side_edge_property : Prop) : ℝ :=
√((5 + √5) / 40)

def radius_of_circumscribed_sphere (base_edge_length : ℝ) (side_edge_property : Prop) : ℝ :=
(1 / 4) * √(10 + 2 * √5)

theorem pentagonal_pyramid_sphere_radii
  (base_edge_length : ℝ)
  (side_edge_property : Prop) -- Represents the property that side edges form a regular star pentagon when unfolded
  (base_unit_length : base_edge_length = 1)
  (forms_star_pentagon : side_edge_property) :
  radius_of_inscribed_sphere base_edge_length side_edge_property = √((5 + √5) / 40) ∧
  radius_of_circumscribed_sphere base_edge_length side_edge_property = (1 / 4) * √(10 + 2 * √5) :=
by
  sorry

end pentagonal_pyramid_sphere_radii_l370_370535


namespace sum_of_integer_values_n_l370_370089

theorem sum_of_integer_values_n (h : ∀ n : ℤ, (30 / (2 * n - 1)) ∈ ℤ → n ∈ {1, 2, 3, 8}) : 
∑ n in {1, 2, 3, 8}, n = 14 := 
by 
     sorry

end sum_of_integer_values_n_l370_370089


namespace greater_number_is_18_l370_370855

theorem greater_number_is_18 (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : x - y = 6) 
  (h3 : y ≥ 10) : 
  x = 18 := 
by 
  sorry

end greater_number_is_18_l370_370855


namespace lcm_12_15_18_l370_370880

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by 
  sorry

end lcm_12_15_18_l370_370880


namespace total_workers_is_22_l370_370487

variable {W R : ℕ}

def average_salary_all (W : ℕ) : Prop := W * 850 = 7 * 1000 + R * 780
def total_number_of_workers (W R : ℕ) : Prop := W = 7 + R

theorem total_workers_is_22 (h1 : average_salary_all W) (h2 : total_number_of_workers W R) : W = 22 := by
  sorry

end total_workers_is_22_l370_370487


namespace distance_between_Joe_and_Gracie_l370_370697

noncomputable def JoePoint : ℂ := 3 + 4 * complex.I
noncomputable def GraciePoint : ℂ := -2 + 2 * complex.I

theorem distance_between_Joe_and_Gracie : complex.abs (JoePoint - GraciePoint) = real.sqrt 29 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end distance_between_Joe_and_Gracie_l370_370697


namespace sin_210_eq_neg_half_l370_370575

theorem sin_210_eq_neg_half : sin (210 * real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_210_eq_neg_half_l370_370575


namespace sin_210_eq_neg_one_half_l370_370583

theorem sin_210_eq_neg_one_half :
  ∀ (θ : ℝ), 
  θ = 210 * (π / 180) → -- angle 210 degrees
  ∃ (refθ : ℝ), 
  refθ = 30 * (π / 180) ∧ -- reference angle 30 degrees
  sin refθ = 1 / 2 → -- sin of reference angle
  sin θ = -1 / 2 := 
by
  intros θ hθ refθ hrefθ hrefθ_sin -- introduce variables and hypotheses
  sorry

end sin_210_eq_neg_one_half_l370_370583


namespace non_formable_triangle_sticks_l370_370975

theorem non_formable_triangle_sticks 
  (sticks : Fin 8 → ℕ) 
  (h_no_triangle : ∀ (i j k : Fin 8), i < j → j < k → sticks i + sticks j ≤ sticks k) : 
  ∃ (max_length : ℕ), (max_length = sticks (Fin.mk 7 (by norm_num))) ∧ max_length = 21 := 
by 
  sorry

end non_formable_triangle_sticks_l370_370975


namespace period_monotonicity_find_a_l370_370265

noncomputable def f (x a : ℝ) := (sqrt 3) * sin x * cos x + cos x ^ 2 + a

theorem period_monotonicity (a : ℝ) :
  ∀ k : ℤ, 
    (T = π) ∧ 
    (∀ x, x ∈ set.Icc (k * π - π / 3) (k * π + π / 6) → monotone_on (λ x, f x a) (set.Icc (k * π - π / 3) (k * π + π / 6)))
 :=
sorry

theorem find_a (a : ℝ) (h : (set.range (λ x, f x a)) ∩ {x | x ∈ [-π / 6, π / 3]} = [a, a + 3 / 2]) :
  a + (a + 3 / 2) = 1 → a = -1 / 4 :=
sorry

end period_monotonicity_find_a_l370_370265


namespace proof_f_comp_l370_370377

def f (x : ℝ) : ℝ :=
if x ≤ 1 then 2^x - 2 else Real.log x / Real.log 2 - 1

theorem proof_f_comp {x : ℝ} (h1 : x = 5/2) : f (f x) = -1/2 :=
by
  sorry

end proof_f_comp_l370_370377


namespace circle_eq_focus_tangent_directrix_l370_370830

theorem circle_eq_focus_tangent_directrix (x y : ℝ) :
  let focus := (0, 4)
  let directrix := -4
  let radius := 8
  ((x - focus.1)^2 + (y - focus.2)^2 = radius^2) :=
by
  let focus := (0, 4)
  let directrix := -4
  let radius := 8
  sorry

end circle_eq_focus_tangent_directrix_l370_370830


namespace find_a_l370_370338

theorem find_a (a b c : ℂ) (ha : a.im = 0)
  (h1 : a + b + c = 5)
  (h2 : a * b + b * c + c * a = 8)
  (h3 : a * b * c = 4) :
  a = 1 ∨ a = 2 :=
sorry

end find_a_l370_370338


namespace counting_numbers_count_l370_370279

theorem counting_numbers_count : 
  let n := { k : ℕ | 53 % k = 3 ∧ k ∣ 50 ∧ k > 3 } in
    n.card = 4 :=
by
  sorry

end counting_numbers_count_l370_370279


namespace sin_210_eq_neg_half_l370_370559

theorem sin_210_eq_neg_half : Real.sin (210 * Real.pi / 180) = -1 / 2 := by
  -- We use the given angles and their known sine values.
  have angle_30 := Real.pi / 6
  have sin_30 := Real.sin angle_30
  -- Expression for the sine of 210 degrees in radians.
  have angle_210 := 210 * Real.pi / 180
  -- Proving the sine of 210 degrees using angle addition formula and unit circle properties.
  calc
    Real.sin angle_210 
    -- 210 degrees is 180 + 30 degrees, translating to pi and pi/6 in radians.
    = Real.sin (Real.pi + Real.pi / 6) : by rw [←Real.ofReal_nat_cast, ←Real.ofReal_nat_cast, Real.ofReal_add, Real.ofReal_div, Real.ofReal_nat_cast]
    -- Using the sine addition formula: sin(pi + x) = -sin(x).
    ... = - Real.sin (Real.pi / 6) : by exact Real.sin_add_pi_div_two angle_30
    -- Substituting the value of sin(30 degrees).
    ... = - 1 / 2 : by rw sin_30

end sin_210_eq_neg_half_l370_370559


namespace find_positive_integer_solution_l370_370854

theorem find_positive_integer_solution :
  ∃ n : ℕ, 3 * n + n ^ 2 = 300 ∧ n = 16 :=
by
  use 16
  split
  { norm_num }
  { rfl }


end find_positive_integer_solution_l370_370854


namespace mark_paint_gallons_l370_370348

noncomputable def paint_required
  (num_pillars : ℕ) (height : ℝ) (diameter : ℝ) (cover_per_gallon : ℝ)
  (radius := diameter / 2)
  (lateral_surface_area_of_pillar := 2 * Real.pi * radius * height)
  (total_surface_area := lateral_surface_area_of_pillar * num_pillars)
  (num_gallons := total_surface_area / cover_per_gallon) : ℕ :=
  Real.ceil num_gallons

theorem mark_paint_gallons :
  paint_required 12 22 8 400 = 17 := sorry

end mark_paint_gallons_l370_370348


namespace sin_210_eq_neg_half_l370_370574

theorem sin_210_eq_neg_half : sin (210 * real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_210_eq_neg_half_l370_370574


namespace wire_length_l370_370824

theorem wire_length (horizontal_distance : ℝ) (height_difference : ℝ) (height_shorter : ℝ) (height_taller : ℝ)
  (h_horizontal_distance : horizontal_distance = 20)
  (h_height_difference : height_difference = 3)
  (h_height_shorter : height_shorter = 8)
  (h_height_taller : height_taller = 18) : 
  real.sqrt (horizontal_distance^2 + (height_taller - (height_shorter + height_difference))^2) = real.sqrt 449 :=
by sorry

end wire_length_l370_370824


namespace square_midpoint_quad_l370_370461

-- Define the isosceles trapezoid and the midpoints' properties
variables {AB CD AD BC : ℝ}

def is_isosceles_trapezoid (AB CD AD BC : ℝ) :=
  AB = CD ∧ AD = BC

def midpoint_quad_is_square (AB CD AD: ℝ) :=
  (AB^2 + CD^2 = 2 * AD^2)

-- The proof statement
theorem square_midpoint_quad (h : is_isosceles_trapezoid AB CD AD BC) :
  midpoint_quad_is_square AB CD AD :=
sorry

end square_midpoint_quad_l370_370461


namespace michael_exceeds_suresh_l370_370807

theorem michael_exceeds_suresh (P M S : ℝ) 
  (h_total : P + M + S = 2400)
  (h_p_m_ratio : P / 5 = M / 7)
  (h_m_s_ratio : M / 3 = S / 2) : M - S = 336 :=
by
  sorry

end michael_exceeds_suresh_l370_370807


namespace min_digs_is_three_l370_370075

/-- Represents an 8x8 board --/
structure Board :=
(dim : ℕ := 8)

/-- Each cell either contains the treasure or a plaque indicating minimum steps --/
structure Cell :=
(content : CellContent)

/-- Possible content of a cell --/
inductive CellContent
| Treasure
| Plaque (steps : ℕ)

/-- Function that returns the minimum number of cells to dig to find the treasure --/
def min_digs_to_find_treasure (board : Board) : ℕ := 3

/-- The main theorem stating the minimum number of cells needed to find the treasure on an 8x8 board --/
theorem min_digs_is_three : 
  ∀ board : Board, min_digs_to_find_treasure board = 3 := 
by 
  intro board
  sorry

end min_digs_is_three_l370_370075


namespace treasure_coins_l370_370443

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l370_370443


namespace part1_a_part1_b_part2_l370_370266

noncomputable def f (x : ℝ) (a b m : ℝ) := 2*x^3 + a*x^2 + b*x + m
noncomputable def f_prime (x : ℝ) (a b m : ℝ) := deriv (λ x, f x a b m)

theorem part1_a (h_symm : ∀ x : ℝ, f_prime x a b m = f_prime (-x - 1/2) a b m) : a = 3 :=
sorry

theorem part1_b (h_prime1 : f_prime 1 3 b m = 0) : b = -12 :=
sorry

theorem part2 (h_zeros : ∀ x : ℝ, (f x 3 (-12) m = 0 ↔ x = r ∨ x = s ∨ x = t) ∧ r ≠ s ∧ s ≠ t ∧ r ≠ t) 
  (h_max : f (-2) 3 (-12) m > 0) (h_min : f 1 3 (-12) m < 0) : -20 < m ∧ m < 7 :=
sorry

end part1_a_part1_b_part2_l370_370266


namespace total_cost_at_60_kmph_minimum_cost_proof_l370_370374

noncomputable def transportation_cost_per_hour (x : ℝ) : ℝ :=
  (1 / 19200) * x^4 - (1 / 160) * x^3 + 15 * x

noncomputable def total_transportation_cost (d : ℝ) (v : ℝ) : ℝ :=
  (d / v) * transportation_cost_per_hour(v)

theorem total_cost_at_60_kmph :
  total_transportation_cost 400 60 = 1500 := by
  sorry

noncomputable def cost_function (x : ℝ) : ℝ :=
  (400 / x) * transportation_cost_per_hour x

noncomputable def minimum_cost_speed : ℝ :=
  80

noncomputable def minimum_cost := cost_function minimum_cost_speed

theorem minimum_cost_proof :
  minimum_cost_speed = 80 ∧ minimum_cost = 2000 / 3 := by
  sorry

end total_cost_at_60_kmph_minimum_cost_proof_l370_370374


namespace raine_gets_correct_change_l370_370522

-- Define the prices of items
def price_bracelet : ℕ := 15
def price_necklace : ℕ := 10
def price_mug : ℕ := 20

-- Define the quantities bought by Raine
def qty_bracelet : ℕ := 3
def qty_necklace : ℕ := 2
def qty_mug : ℕ := 1

-- Define the total amount Raine gives
def amount_given : ℕ := 100

-- Define total cost of the items bought by Raine
def total_cost : ℕ := (qty_bracelet * price_bracelet) +
                      (qty_necklace * price_necklace) +
                      (qty_mug * price_mug)

-- Define the amount of change Raine gets back
def change : ℕ := amount_given - total_cost

-- The theorem we want to prove
theorem raine_gets_correct_change : change = 15 := by
  rw [change, total_cost]
  simp only [qty_bracelet, price_bracelet, qty_necklace, price_necklace, qty_mug, price_mug, amount_given]
  norm_num
  sorry

end raine_gets_correct_change_l370_370522


namespace weight_of_each_soda_crate_l370_370199

-- Definitions based on conditions
def bridge_weight_limit := 20000
def empty_truck_weight := 12000
def number_of_soda_crates := 20
def dryer_weight := 3000
def number_of_dryers := 3
def fully_loaded_truck_weight := 24000
def soda_weight := 1000
def produce_weight := 2 * soda_weight
def total_cargo_weight := fully_loaded_truck_weight - empty_truck_weight

-- Lean statement to prove the weight of each soda crate
theorem weight_of_each_soda_crate :
  number_of_soda_crates * ((total_cargo_weight - (number_of_dryers * dryer_weight)) / 3) / number_of_soda_crates = 50 :=
by
  sorry

end weight_of_each_soda_crate_l370_370199


namespace coeff_x3_polynomial_l370_370617

theorem coeff_x3_polynomial : 
  let p := 4 * (x^3 - 2 * x^2) + 3 * (x^2 - x^3 + 2 * x^4) - 5 * (x^4 - 2 * x^3)
  in polynomial.coeff p 3 = 11 :=
by
  let p := 4 * (x^3 - 2 * x^2) + 3 * (x^2 - x^3 + 2 * x^4) - 5 * (x^4 - 2 * x^3)
  show polynomial.coeff p 3 = 11
  sorry

end coeff_x3_polynomial_l370_370617


namespace exercise_proof_l370_370161

-- Define the four functions
def A (x : ℝ) := 2^x
def B (x : ℝ) := 2^|x|
def C (x : ℝ) := 2^x - 2^(-x)
def D (x : ℝ) := 2^x + 2^(-x)

-- Define conditions for odd and increasing functions
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_increasing (f : ℝ → ℝ) := ∀ x y, x < y → f x < f y

-- The theorem to prove
theorem exercise_proof : is_odd C ∧ is_increasing C :=
by
  -- skipping proof details
  sorry

end exercise_proof_l370_370161


namespace work_done_days_l370_370116

theorem work_done_days (a_days : ℕ) (b_days : ℕ) (together_days : ℕ) (a_work_done : ℚ) (b_work_done : ℚ) (together_work : ℚ) : 
  a_days = 12 ∧ b_days = 15 ∧ together_days = 5 ∧ 
  a_work_done = 1/12 ∧ b_work_done = 1/15 ∧ together_work = 3/4 → 
  ∃ days : ℚ, a_days > 0 ∧ b_days > 0 ∧ together_days > 0 ∧ days = 3 := 
  sorry

end work_done_days_l370_370116


namespace sum_of_valid_n_equals_14_l370_370092

theorem sum_of_valid_n_equals_14 :
  (∑ n in {n : ℤ | ∃ (d : ℤ), d ∣ 30 ∧ 2 * n - 1 = d}, n) = 14 :=
by
  sorry

end sum_of_valid_n_equals_14_l370_370092


namespace largest_of_three_l370_370477

theorem largest_of_three : 
  2 ^ (-3 : ℝ) < 3 ^ (1 / 2 : ℝ) ∧ 3 ^ (1 / 2 : ℝ) < log 2 5 ∧ log 2 5 > 2 :=
by
  sorry

end largest_of_three_l370_370477


namespace pirates_treasure_l370_370422

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l370_370422


namespace find_k_value_l370_370626

theorem find_k_value (k : ℝ) : 
  5 + ∑' n : ℕ, (5 + k + n) / 5^(n+1) = 12 → k = 18.2 :=
by 
  sorry

end find_k_value_l370_370626


namespace standard_eq_of_hyperbola_l370_370185

-- Definitions of required variables and parameters
variables (a b : ℝ) (e : ℝ) (c : ℝ)
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom e_def : e = 2
axiom hyperbola_eqn : (y : ℝ) * (y / a)^2 - (x : ℝ) * (x / b)^2 = 1

noncomputable def find_hyperbola_standard_eq : Prop :=
  hyperbola_eqn ∧ -- Hyperbola with given conditions
  x^2 = 8 * y ∧ -- Parabola equation
  e = c / a ∧  -- Eccentricity definition
  c = 2 ∧ -- Focus definition from parabola
  c^2 = a^2 + b^2 ∧ -- Relationship between a, b, and c in a hyperbola
  y^2 - x^2 / 3 = 1 -- Standard equation of hyperbola

-- The statement only - proof to be completed
theorem standard_eq_of_hyperbola :
  find_hyperbola_standard_eq :=
sorry

end standard_eq_of_hyperbola_l370_370185


namespace speed_of_man_l370_370973

/-- A train 160 m long passes a man running at a certain speed in the direction opposite to that 
of the train in 6 seconds, and the speed of the train is approximately 90 kmph. Prove that the 
speed of the man is approximately 6 kmph. -/
theorem speed_of_man (length_train meters: ℕ) (time secs: ℕ) (speed_train_kmph: ℕ)
  (length_train = 160) (time = 6) (speed_train_kmph = 90) : 
  let speed_train_mps := speed_train_kmph * 1000 / 3600 in 
  let relative_speed := length_train / time in
  let speed_of_man_mps := relative_speed - speed_train_mps in
  let speed_of_man_kmph := speed_of_man_mps * 3600 / 1000 in
  abs (speed_of_man_kmph - 6) < 0.1 :=
by 
  let speed_train_mps := 25 -- converting 90 km/h to m/s
  let relative_speed := 26.67 -- relative speed in m/s (length_train / time)
  let speed_of_man_mps := 1.67 -- man's speed in m/s (relative speed - speed_train_mps)
  let speed_of_man_kmph := 6 -- man's speed in km/h
  sorry

end speed_of_man_l370_370973


namespace constant_sum_of_inverses_l370_370782

theorem constant_sum_of_inverses
  (O P Q R S : Point)
  (ABC : Triangle)
  (h_centroid : centroid ABC = O)
  (h_equilateral : is_equilateral ABC)
  (h_plane : ∃ (plane : Plane), O ∈ plane ∧ ∀ {X}, plane ∩ (line_through P X) = point_or_line Q R S) :
  ∃ k : ℝ, ∀ (P Q R S : Point), (1 / distance P Q + 1 / distance P R + 1 / distance P S) = k :=
sorry

end constant_sum_of_inverses_l370_370782


namespace problem_statement_l370_370681

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 4))

theorem problem_statement :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc (-Real.sqrt 2 / 2) 1) ∧
  (f (Real.pi / 2) = -Real.sqrt 2 / 2) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8), 
    ∃ δ > 0, ∀ y ∈ Set.Ioc x (x + δ), f x < f y) :=
by {
  sorry
}

end problem_statement_l370_370681


namespace silvia_trip_shorter_than_jerry_by_25_percent_l370_370760

def jerry_distance : ℝ := 7
def silvia_distance : ℝ := real.sqrt (3^2 + 4^2)
def percentage_reduction : ℝ := ((jerry_distance - silvia_distance) / jerry_distance) * 100

theorem silvia_trip_shorter_than_jerry_by_25_percent :
  abs (percentage_reduction - 25) < 3 :=
by
  sorry

end silvia_trip_shorter_than_jerry_by_25_percent_l370_370760


namespace price_difference_l370_370945

/-- Given an original price, two successive price increases, and special deal prices for a fixed number of items, 
    calculate the difference between the final retail price and the average special deal price. -/
theorem price_difference
  (original_price : ℝ) (first_increase_percent: ℝ) (second_increase_percent: ℝ)
  (special_deal_percent_1: ℝ) (num_items_1: ℕ) (special_deal_percent_2: ℝ) (num_items_2: ℕ)
  (final_retail_price : ℝ) (average_special_deal_price : ℝ) :
  original_price = 50 →
  first_increase_percent = 0.30 →
  second_increase_percent = 0.15 →
  special_deal_percent_1 = 0.70 →
  num_items_1 = 50 →
  special_deal_percent_2 = 0.85 →
  num_items_2 = 100 →
  final_retail_price = original_price * (1 + first_increase_percent) * (1 + second_increase_percent) →
  average_special_deal_price = 
    (num_items_1 * (special_deal_percent_1 * final_retail_price) + 
    num_items_2 * (special_deal_percent_2 * final_retail_price)) / 
    (num_items_1 + num_items_2) →
  final_retail_price - average_special_deal_price = 14.95 :=
by
  intros
  sorry

end price_difference_l370_370945


namespace parabola_equation_slope_of_line_l_l370_370253

-- First Part
theorem parabola_equation (p : ℝ) (k : ℝ) (A B : ℝ → ℝ → Prop) (AB_dist : ℝ) (h1 : k = 1) (h2 : A (x y)) (h3 : B (x y))
  (h4 : AB_dist = 8) (h5 : x^2 = 2 * p * y) : x^2 = 4 * y := 
sorry

-- Second Part
theorem slope_of_line_l (p k : ℝ) (F A B P : ℝ → ℝ → Prop) (h1 : sum_of_slopes = -3/2) : 
  k = -2 ∨ k = 1/2 :=
sorry

end parabola_equation_slope_of_line_l_l370_370253


namespace min_value_of_expression_l370_370255

theorem min_value_of_expression (a b : ℝ) (h1 : 1 < a) (h2 : 0 < b) (h3 : a + 2 * b = 2) : 
  4 * (1 + Real.sqrt 2) ≤ (2 / (a - 1) + a / b) :=
by
  sorry

end min_value_of_expression_l370_370255


namespace common_divisors_sum_l370_370401

theorem common_divisors_sum (a b c d e : ℕ) 
  (h₁ : a = 20) 
  (h₂ : b = 40) 
  (h₃ : c = 100) 
  (h₄ : d = 80) 
  (h₅ : e = 180) : 
  ∑ x in {x | x ∣ a ∧ x ∣ b ∧ x ∣ c ∧ x ∣ d ∧ x ∣ e}, x = 12 := 
by
  sorry

end common_divisors_sum_l370_370401


namespace compute_expression_l370_370999

theorem compute_expression : 12 * (1 / 7) * 14 * 2 = 48 := 
sorry

end compute_expression_l370_370999


namespace sum_of_integer_values_n_l370_370087

theorem sum_of_integer_values_n (h : ∀ n : ℤ, (30 / (2 * n - 1)) ∈ ℤ → n ∈ {1, 2, 3, 8}) : 
∑ n in {1, 2, 3, 8}, n = 14 := 
by 
     sorry

end sum_of_integer_values_n_l370_370087


namespace correct_proportion_expression_l370_370093

def is_fraction_correctly_expressed (numerator denominator : ℕ) (expression : String) : Prop :=
  -- Define the property of a correctly expressed fraction in English
  expression = "three-fifths"

theorem correct_proportion_expression : 
  is_fraction_correctly_expressed 3 5 "three-fifths" :=
by
  sorry

end correct_proportion_expression_l370_370093


namespace correct_option_is_D_l370_370099

theorem correct_option_is_D 
  (ray_infinite : ∀ (R : Type) (r : R), extends_infinitely r)
  (line_infinite : ∀ (L : Type) (l : L), extends_infinitely l)
  (segment_extendable : ∀ (S : Type) (s : S), extendable s) :
  extendable segment :=
by sorry

-- Definitions for the conditions to make it clearer
def extends_infinitely (x : ℝ) : Prop := ∀ y : ℝ, x ≤ y
def extendable (s : ℝ) : Prop := ∃ y : ℝ, s < y

end correct_option_is_D_l370_370099


namespace choose_signs_to_sum_zero_l370_370257

theorem choose_signs_to_sum_zero (n : ℕ) (a : Fin n → ℕ) (h1 : n ≥ 2) (h2 : ∀ k : Fin n, 0 < a k ∧ a k ≤ k + 1) :
  (∃ σ : Fin n → ℤ, (∀ i, σ i = 1 ∨ σ i = -1) ∧ ∑ i, σ i * a i = 0) ↔ (∑ i, a i % 2 = 0) :=
sorry

end choose_signs_to_sum_zero_l370_370257


namespace positive_solution_count_l370_370623

noncomputable def count_positive_solutions : ℝ :=
  let x := sqrt((3 - sqrt(5)) / 2) in
  if 0 ≤ x ∧ x ≤ 1 then 1 else 0

theorem positive_solution_count :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧
   sin (arccos (sqrt (tan (arcsin x)) ^ 2)) ^ 2 = x ^ 2) :=
sorry

end positive_solution_count_l370_370623


namespace no_divisibility_after_removing_8s_l370_370545

-- Definitions for the problem
def N := List.range' 1 2018

def remove_eights (n : List ℕ) : List ℕ :=
  n.filter (λ x => ¬ x.digits.contains 8)

-- Predicates to check divisibility by 3
def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

-- Statement of the proof problem
theorem no_divisibility_after_removing_8s :
  ¬ divisible_by_3 (List.sum (remove_eights N)) :=
sorry


end no_divisibility_after_removing_8s_l370_370545


namespace license_plate_count_l370_370951

theorem license_plate_count :
  let digits := 10
  let letters := 26
  let positions := 6
  positions * digits^5 * letters^3 = 105456000 := by
  sorry

end license_plate_count_l370_370951


namespace man_speed_still_water_l370_370527

theorem man_speed_still_water : 
  ∀ (Vm Vs Vd Vu : ℝ), 
    (Vd = Vm + Vs) → (Vu = Vm - Vs) → 
    (Vd = 18) → (Vu = 6) → (Vs = 6) → 
    Vm = 12 :=
by
  assume Vm Vs Vd Vu,
  assume h1 : Vd = Vm + Vs,
  assume h2 : Vu = Vm - Vs,
  assume h3 : Vd = 18,
  assume h4 : Vu = 6,
  assume h5 : Vs = 6,
  sorry

end man_speed_still_water_l370_370527


namespace average_speed_for_remaining_part_l370_370901

theorem average_speed_for_remaining_part (D : ℝ) (v : ℝ) 
  (h1 : 0.8 * D / 80 + 0.2 * D / v = D / 50) : v = 20 :=
sorry

end average_speed_for_remaining_part_l370_370901


namespace Nigel_initial_amount_l370_370355

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

end Nigel_initial_amount_l370_370355


namespace factorization_of_polynomial_l370_370204

theorem factorization_of_polynomial (x : ℝ) : 12 * x^2 - 40 * x + 25 = (2 * Real.sqrt 3 * x - 5)^2 :=
  sorry

end factorization_of_polynomial_l370_370204


namespace sufficient_not_necessary_a_gt_4_l370_370289

variable (x a : ℝ)

def p := -2 < x ∧ x < 4
def q := (x + 2) * (x - a) < 0

theorem sufficient_not_necessary_a_gt_4 (h : ∀ x, p x → q x) (hn : ¬∀ x, q x → p x) : a > 4 :=
sorry

end sufficient_not_necessary_a_gt_4_l370_370289


namespace bread_slices_leftover_l370_370366

-- Definition of given conditions
def bread_packages : Nat := 4
def slices_per_bread_package : Nat := 24
def total_bread_slices : Nat := bread_packages * slices_per_bread_package

def ham_packages : Nat := 3
def slices_per_ham_package : Nat := 14
def total_ham_slices : Nat := ham_packages * slices_per_ham_package

def turkey_packages : Nat := 2
def slices_per_turkey_package : Nat := 18
def total_turkey_slices : Nat := turkey_packages * slices_per_turkey_package

def roast_beef_packages : Nat := 1
def slices_per_roast_beef_package : Nat := 10
def total_roast_beef_slices : Nat := roast_beef_packages * slices_per_roast_beef_package

def ham_proportion : Float := 0.40
def turkey_proportion : Float := 0.35
def roast_beef_proportion : Float := 0.25

-- Proof of leftovers
theorem bread_slices_leftover : total_bread_slices - ((Int.ofNat (total_roast_beef_slices / roast_beef_proportion.toNat)) * 2) = 16 := by
  sorry

end bread_slices_leftover_l370_370366


namespace largest_of_three_l370_370070

theorem largest_of_three (a b c : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : ab + ac + bc = -8) 
  (h3 : abc = -20) : 
  max a (max b c) = (1 + Real.sqrt 41) / 2 := 
by 
  sorry

end largest_of_three_l370_370070


namespace paths_content_729_l370_370919

theorem paths_content_729 :
  ∃ n : ℕ, (
    let C := [
      [[], [], [], [], [], [], "C", [], [], [], [], [], []],
      [[], [], [], [], "", "C", "O", "C", "", "", "", "", []],
      [[], [], [], "", "C", "O", "N", "O", "C", "", "", [], []],
      [[], [], "C", "O", "N", "T", "N", "O", "C", "", "", "", []],
      [[], "C", "O", "N", "T", "E", "T", "N", "O", "C", "", "", []],
      ["C", "O", "N", "T", "E", "N", "E", "T", "N", "O", "C", "", []],
      ["C", "O", "N", "T", "E", "N", "T", "N", "E", "T", "N", "O", "C"]
    ] in
    let valid_movements := [(0,1), (-1,1), (1,1)] in
    ∀ t : list (list string) → ℕ,
    t.extract_path_paths "CONTENT" valid_movements = 729
  ) :=
exists.intro 729 (begin 
  -- The proof is omitted. In comprising of verifying path counts.
  sorry 
end)

end paths_content_729_l370_370919


namespace common_divisors_sum_l370_370859

theorem common_divisors_sum : 
  let common_divisors := {d | d > 0 ∧ ∀ n ∈ [36, 72, -24, 120, 96], n % d = 0}
  ∑ d in common_divisors, d = 16 := 
by
  sorry

end common_divisors_sum_l370_370859


namespace total_jewelry_pieces_l370_370523

noncomputable def initial_necklaces : ℕ := 10
noncomputable def initial_earrings : ℕ := 15
noncomputable def bought_necklaces : ℕ := 10
noncomputable def bought_earrings : ℕ := 2 * initial_earrings / 3
noncomputable def extra_earrings_from_mother : ℕ := bought_earrings / 5

theorem total_jewelry_pieces : initial_necklaces + bought_necklaces + initial_earrings + bought_earrings + extra_earrings_from_mother = 47 :=
by
  have total_necklaces : ℕ := initial_necklaces + bought_necklaces
  have total_earrings : ℕ := initial_earrings + bought_earrings + extra_earrings_from_mother
  have total_jewelry : ℕ := total_necklaces + total_earrings
  exact Eq.refl 47
  
#check total_jewelry_pieces -- Check if the type is correct

end total_jewelry_pieces_l370_370523


namespace remainder_317_l370_370370

theorem remainder_317 (y : ℤ)
  (h1 : 4 + y ≡ 9 [ZMOD 16])
  (h2 : 6 + y ≡ 8 [ZMOD 81])
  (h3 : 8 + y ≡ 49 [ZMOD 625]) :
  y ≡ 317 [ZMOD 360] := 
sorry

end remainder_317_l370_370370


namespace total_cloth_length_l370_370850

theorem total_cloth_length :
  ∀ (n : ℕ) (a₁ aₙ : ℕ),
    n = 30 → a₁ = 5 → aₙ = 1 →
    let S_n := n * (a₁ + aₙ) / 2 in
    S_n = 90 := by
  intros n a₁ aₙ h1 h2 h3
  rw [h1, h2, h3]
  have h : 30 * (5 + 1) / 2 = 90 := by norm_num
  exact h

end total_cloth_length_l370_370850


namespace sara_initial_savings_l370_370365

-- Given conditions as definitions
def save_rate_sara : ℕ := 10
def save_rate_jim : ℕ := 15
def weeks : ℕ := 820

-- Prove that the initial savings of Sara is 4100 dollars given the conditions
theorem sara_initial_savings : 
  ∃ S : ℕ, S + save_rate_sara * weeks = save_rate_jim * weeks → S = 4100 := 
sorry

end sara_initial_savings_l370_370365


namespace charge_increase_percentage_l370_370372

variable (P R G : ℝ)

def charge_relation_1 : Prop := P = 0.45 * R
def charge_relation_2 : Prop := P = 0.90 * G

theorem charge_increase_percentage (h1 : charge_relation_1 P R) (h2 : charge_relation_2 P G) : 
  (R/G - 1) * 100 = 100 :=
by
  sorry

end charge_increase_percentage_l370_370372


namespace part_a_l370_370548

theorem part_a (n : ℤ) : 
  (1 + Complex.i)^n = 2^(n / 2) * (Complex.cos (n * Real.pi / 4) + Complex.i * Complex.sin (n * Real.pi / 4)) :=
sorry

end part_a_l370_370548


namespace min_value_H_interval_l370_370657

theorem min_value_H_interval {f g : ℝ → ℝ} {a b : ℝ} (h_odd_f : ∀ x, f (-x) = -f x) (h_odd_g : ∀ x, g (-x) = -g x)
    (H_max : ∀ (x : ℝ), 0 < x → H x < 5)
    (H : ℝ → ℝ := λ x, a * f x + b * g x + 1) :
    ∃ y : ℝ, y < 0 ∧ H y = -3 := sorry

end min_value_H_interval_l370_370657


namespace inequality_proof_l370_370787

variable (x y z : ℝ)

theorem inequality_proof (h : x + y + z = x * y + y * z + z * x) :
  x / (x^2 + 1) + y / (y^2 + 1) + z / (z^2 + 1) ≥ -1/2 :=
sorry

end inequality_proof_l370_370787


namespace pirate_treasure_l370_370449

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l370_370449


namespace sin_210_eq_neg_one_half_l370_370582

theorem sin_210_eq_neg_one_half :
  sin (Real.pi * (210 / 180)) = -1 / 2 :=
by
  have angle_eq : 210 = 180 + 30 := by norm_num
  have sin_30 : sin (Real.pi / 6) = 1 / 2 := by norm_num
  have cos_30 : cos (Real.pi / 6) = sqrt 3 / 2 := by norm_num
  sorry

end sin_210_eq_neg_one_half_l370_370582


namespace impossible_infinite_chessboard_condition_l370_370314

-- Define the conditions of the problem
def infinite_chessboard : Type := ℤ × ℤ
def positive_integer (n : ℕ) : Prop := 0 < n

-- Define the function on the infinite chessboard
def f : infinite_chessboard → ℕ
noncomputable def f := sorry

-- Main theorem statement
theorem impossible_infinite_chessboard_condition :
  ¬ ∃ f : infinite_chessboard → ℕ,
      (∀ (i j : ℕ), 
        (0 < i ∧ 0 < j) →
        (∀ (x y : ℤ), 
          (∃ n m : ℕ, ∑ x' in finset.range m, ∑ y' in finset.range n, f (x+x', y+y') = 0)
        )) :=
by
  sorry

end impossible_infinite_chessboard_condition_l370_370314


namespace cos_alpha_sub_beta_sin_alpha_l370_370227

open Real

variables (α β : ℝ)

-- Conditions:
-- 0 < α < π / 2
def alpha_in_first_quadrant := 0 < α ∧ α < π / 2

-- -π / 2 < β < 0
def beta_in_fourth_quadrant := -π / 2 < β ∧ β < 0

-- sin β = -5/13
def sin_beta := sin β = -5 / 13

-- tan(α - β) = 4/3
def tan_alpha_sub_beta := tan (α - β) = 4 / 3

-- Theorem statements (follows directly from the conditions and the equivalence):
theorem cos_alpha_sub_beta : alpha_in_first_quadrant α → beta_in_fourth_quadrant β → sin_beta β → tan_alpha_sub_beta α β → cos (α - β) = 3 / 5 := sorry

theorem sin_alpha : alpha_in_first_quadrant α → beta_in_fourth_quadrant β → sin_beta β → tan_alpha_sub_beta α β → sin α = 33 / 65 := sorry

end cos_alpha_sub_beta_sin_alpha_l370_370227


namespace radius_of_inscribed_circle_l370_370082

theorem radius_of_inscribed_circle (d1 d2 : ℝ) (h₁ : d1 = 10) (h₂ : d2 = 24) : 
  ∃ r : ℝ, r = 60 / 13 :=
by
  use 60 / 13
  sorry

end radius_of_inscribed_circle_l370_370082


namespace total_profit_l370_370976

theorem total_profit (A_investment : ℝ) (B_investment : ℝ) (C_investment : ℝ) 
                     (A_months : ℝ) (B_months : ℝ) (C_months : ℝ)
                     (C_share : ℝ) (A_profit_percentage : ℝ) : ℝ :=
  let A_capital_months := A_investment * A_months
  let B_capital_months := B_investment * B_months
  let C_capital_months := C_investment * C_months
  let total_capital_months := A_capital_months + B_capital_months + C_capital_months
  let P := (C_share * total_capital_months) / (C_capital_months * (1 - A_profit_percentage))
  P

example : total_profit 6500 8400 10000 6 5 3 1900 0.05 = 24667 := by
  sorry

end total_profit_l370_370976


namespace maximize_quadrilateral_area_l370_370645

noncomputable def max_area_quadrilateral (O A : Point) (is_in_circle : Point) (is_inside : Point ≠ O) : Prop :=
  ∃ B C D : Point, (B, C, and D are on the circle centered at O) ∧
  are_maximizes_area A B C D ∧
  (AC intersects BD at O) ∧
  (BD ⊥ AC)

theorem maximize_quadrilateral_area
  {O A : Point} (hA : in_circle O A) (hA_ne_O : A ≠ O) :
  ∃ B C D : Point, 
  in_circle O B ∧
  in_circle O C ∧
  in_circle O D ∧
  maximize_area ABCD ∧
  intersection O AC BD ∧
  perp BD AC :=
sorry

end maximize_quadrilateral_area_l370_370645


namespace limit_kn_half_l370_370103

open Real
open BigOperators
open IntervalIntegral

noncomputable def kn (n : ℕ) : ℝ :=
∫ x in [0,1]^n, cos^2 (π * (∑ i in finset.range n, x i) / (2 * n))

theorem limit_kn_half :
  tendsto (λ n, kn n) at_top (𝓝 (1 / 2)) :=
sorry

end limit_kn_half_l370_370103


namespace painted_cubes_at_least_two_blue_faces_l370_370948

theorem painted_cubes_at_least_two_blue_faces:
  let n := 4 in
  let total_one_inch_cubes := n * n * n in
  let corner_cubes := 8 in
  let edge_cubes := 12 * 2 in
  (corner_cubes + edge_cubes) = 32 :=
by
  let n := 4
  let total_one_inch_cubes := n * n * n
  let corner_cubes := 8
  let edge_cubes := 12 * 2
  have num_cubes_with_two_faces_or_more := corner_cubes + edge_cubes
  have num_cubes_with_two_faces_or_more_eq_32 : num_cubes_with_two_faces_or_more = 32 := by
    rw [corner_cubes, edge_cubes]
    norm_num
  exact num_cubes_with_two_faces_or_more_eq_32
sorry

end painted_cubes_at_least_two_blue_faces_l370_370948


namespace sum_of_prime_factors_l370_370771

noncomputable def gcd (a b : ℕ) : ℕ := sorry
noncomputable def factorial (n : ℕ) : ℕ := sorry
noncomputable def binomial (n k : ℕ) : ℕ := sorry

noncomputable def legendre_formula (n p : ℕ) : ℕ :=
  if h : p > 1 then
    let rec aux k :=
      if h1 : k > 0 then
        (n / (p^k)) + aux (k + 1)
      else
        0
    in
    aux 1
  else
    0

theorem sum_of_prime_factors (m n : ℕ) (h : gcd m n = 1)
  (h1 : ∑ k in finset.range 2021, (-1)^k * binomial 2020 k * cos (2020 * (arccos (k / 2020))) = m / n) :
  let prime_factors_product :=
        2^8 * 5^1517 * 101^2000 in
  let sum_of_prime_factors :=
        2 * 8 + 5 * 1517 + 101 * 2000 in
  prime_factors_product = n ∧ sum_of_prime_factors = 209601 := sorry

end sum_of_prime_factors_l370_370771


namespace pirate_treasure_l370_370450

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l370_370450


namespace min_boxes_to_eliminate_l370_370297

theorem min_boxes_to_eliminate {boxes : Fin 26 → ℕ}
  (values : List ℕ)
  (cond : values = [1, 1000, 5000, 25000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000])
  (high_value_boxes : List ℕ := [200000, 300000, 400000, 500000, 750000, 1000000])
  (high_value_count : high_value_boxes.length = 6) :
  (∃ m, m = 15 ∧ 
   (boxes_count : Fin 26 → ℕ) ×
   (eliminated_boxes : ℕ) →
   eliminated_boxes ≥ m ∧ (boxes_count - eliminated_boxes) ≤ high_value_boxes.length * 2)
↔ true := by
  sorry

end min_boxes_to_eliminate_l370_370297


namespace max_c_for_good_array_l370_370660

variables {a b : ℕ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h_lt : a < b)
          (h_good_array : Nat.lcm a b + Nat.lcm (a + 2) (b + 2) = 2 * Nat.lcm (a + 1) (b + 1))

theorem max_c_for_good_array (h_good : ∃ (a b : ℕ), h_pos_a ∧ h_pos_b ∧ h_lt ∧ h_good_array) : ∃ c : ℚ, c = 1 / 2 ∧ b > c * a^3 :=
by {
  sorry
}

end max_c_for_good_array_l370_370660


namespace plane_distance_eq_l370_370194

-- Define the planes as functions taking coordinates (x, y, z) and returning real numbers
def plane1 (x y z : ℝ) : ℝ := 2 * x - 4 * y + 6 * z - 10
def plane2 (x y z : ℝ) : ℝ := x - 2 * y + 3 * z - 4

-- Define a function to calculate the distance between a point and a plane
def point_to_plane_distance (a b c d : ℝ) (x₀ y₀ z₀ : ℝ) : ℝ :=
  abs (a * x₀ + b * y₀ + c * z₀ + d) / (sqrt (a^2 + b^2 + c^2))

theorem plane_distance_eq (a1 b1 c1 d1 a2 b2 c2 d2: ℝ) (x₀ y₀ z₀ : ℝ) :
  (a1 = 2 ∧ b1 = -4 ∧ c1 = 6 ∧ d1 = -10 ∧ a2 = 1 ∧ b2 = -2 ∧ c2 = 3 ∧ d2 = -4)
  → (plane1 x₀ y₀ z₀ = 0 ∧ plane2 x₀ y₀ z₀ = 0)
  → point_to_plane_distance a1 b1 c1 d1 4 0 0 = (1 / sqrt 14) := by
  intros _ _
  sorry

end plane_distance_eq_l370_370194


namespace splendid_count_l370_370957

def isSplendid (n : ℕ) : Prop :=
  let digits := [n / 10^6 % 10, n / 10^5 % 10, n / 10^4 % 10, n / 10^3 % 10, n / 10^2 % 10, n / 10^1 % 10, n % 10]
  n >= 1000000 ∧ n <= 9999999 ∧
  ∀ k : ℕ, k ∈ [1, 2, 3, 4, 5, 6, 7] →
    (∑ i in finset.range k, digits[i - 1] * 10^(k - 1 - i)) % k = 0

theorem splendid_count : 
  finset.univ.filter (λ n, isSplendid n).card = 2 := 
sorry

end splendid_count_l370_370957


namespace ivanka_woody_total_months_l370_370755

theorem ivanka_woody_total_months
  (woody_years : ℝ)
  (months_per_year : ℝ)
  (additional_months : ℕ)
  (woody_months : ℝ)
  (ivanka_months : ℝ)
  (total_months : ℝ)
  (h1 : woody_years = 1.5)
  (h2 : months_per_year = 12)
  (h3 : additional_months = 3)
  (h4 : woody_months = woody_years * months_per_year)
  (h5 : ivanka_months = woody_months + additional_months)
  (h6 : total_months = woody_months + ivanka_months) :
  total_months = 39 := by
  sorry

end ivanka_woody_total_months_l370_370755


namespace divisibility_by_19_l370_370611

theorem divisibility_by_19 (n : ℤ) : 
  (19 ∣ 2 ^ (3 * n + 4) + 3 ^ (2 * n + 1)) ↔ ∃ k : ℤ, n = 18 * k := by
  sorry

end divisibility_by_19_l370_370611


namespace width_of_jordan_rectangle_l370_370110

def carol_length := 5
def carol_width := 24
def jordan_length := 2
def jordan_area := carol_length * carol_width

theorem width_of_jordan_rectangle : ∃ (w : ℝ), jordan_length * w = jordan_area ∧ w = 60 :=
by
  use 60
  simp [carol_length, carol_width, jordan_length, jordan_area]
  sorry

end width_of_jordan_rectangle_l370_370110


namespace ellipse_properties_midpoint_trajectory_slope_product_const_l370_370772

noncomputable def ellipse_equation (a b : ℝ) : (ℝ × ℝ) → Prop := 
  λ p, let x := p.1 in let y := p.2 in x^2 / a^2 + y^2 / b^2 = 1

def foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-sqrt (a^2 - b^2), 0), (sqrt (a^2 - b^2), 0))

theorem ellipse_properties (a b : ℝ) (ha : a > 0) (hb: b > 0) (hab : a > b) :
  let C := ellipse_equation 2 sqrt(3) in
  let F1 := (-1, 0) in let F2 := (1, 0) in
  (C (1, 3 / 2)) ∧ 
  ( ( (1, 3 / 2).fst - F1.fst)^2  + ( (1, 3 / 2).snd - F1.snd)^2) +
  ( ( (1, 3 / 2).fst - F2.fst)^2  + ( (1, 3 / 2).snd - F2.snd)^2) = 4 :=
sorry

theorem midpoint_trajectory (x y : ℝ) :
  let C := ellipse_equation 2 sqrt(3) in
  (C (2 * x + 1, 2 * y)) →
  (x + 1 / 2)^2 + (4 * y^2 / 3) = 1 :=
sorry

theorem slope_product_const (a b : ℝ) {x y m n : ℝ}
  (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let C := ellipse_equation a b in
  (C (x, y)) ∧ (C (m, n)) → (m = -x) ∧ (n = -y) → 
  let K_PM := (y - n) / (x - m) in
  let K_PN := (y + n) / (x + m) in
  K_PM * K_PN = -(b^2) / (a^2) :=
sorry

end ellipse_properties_midpoint_trajectory_slope_product_const_l370_370772


namespace sin_210_eq_neg_one_half_l370_370577

theorem sin_210_eq_neg_one_half :
  sin (Real.pi * (210 / 180)) = -1 / 2 :=
by
  have angle_eq : 210 = 180 + 30 := by norm_num
  have sin_30 : sin (Real.pi / 6) = 1 / 2 := by norm_num
  have cos_30 : cos (Real.pi / 6) = sqrt 3 / 2 := by norm_num
  sorry

end sin_210_eq_neg_one_half_l370_370577


namespace incorrect_option_D_l370_370896

-- definition of geometric objects and their properties
def octahedron_faces : Nat := 8
def tetrahedron_can_be_cut_into_4_pyramids : Prop := True
def frustum_extension_lines_intersect_at_a_point : Prop := True
def rectangle_rotated_around_side_forms_cylinder : Prop := True

-- incorrect identification of incorrect statement
theorem incorrect_option_D : 
  (∃ statement : String, statement = "D" ∧ ¬rectangle_rotated_around_side_forms_cylinder)  → False :=
by
  -- Proof of incorrect identification is not required per problem instructions
  sorry

end incorrect_option_D_l370_370896


namespace proof_problem_l370_370335

variables (n : ℕ) (a b : Fin n → ℝ) (A B : ℝ)

-- State all given conditions
def conditions (a b : Fin n → ℝ) (A B : ℝ) : Prop :=
  (∀ i, 0 < a i ∧ 0 < b i) ∧
  (∀ i, a i ≤ b i) ∧
  (∀ i, a i ≤ A) ∧
  (∏ i, b i / a i) ≤ (B / A)

-- The statement we need to prove
theorem proof_problem (h : conditions n a b A B) :
  (∏ i, (b i + 1) / (a i + 1)) ≤ (B + 1) / (A + 1) := by
  -- The proof will be filled here
  sorry

end proof_problem_l370_370335


namespace chord_eq_MN_l370_370291

-- Definitions of the conditions
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9
def midpoint (P M N : ℝ × ℝ) : Prop := P.1 = (M.1 + N.1) / 2 ∧ P.2 = (M.2 + N.2) / 2

-- The midpoint P(1, 1) and the circle equation are the given conditions
def given_conditions (M N : ℝ × ℝ) : Prop := 
  midpoint (1, 1) M N ∧ circle_eq M.1 M.2 ∧ circle_eq N.1 N.2

-- The main theorem to prove: the equation of the line on which chord MN lies
theorem chord_eq_MN (M N : ℝ × ℝ) (h : given_conditions M N) : 
  ∃ (a b c : ℝ), a * (fst M) + b * (snd M) + c = 0 ∧ a * (fst N) + b * (snd N) + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = -1 :=
sorry

end chord_eq_MN_l370_370291


namespace countless_lines_l370_370339

-- Define point and line types
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line where
  point : Point
  direction : Point

-- Define the condition that point P is outside line L
def point_outside_line (P : Point) (L : Line) : Prop :=
  ¬ ∃ t : ℝ, P = { x := L.point.x + t * L.direction.x,
                   y := L.point.y + t * L.direction.y,
                   z := L.point.z + t * L.direction.z }

-- Define the Lean statement
theorem countless_lines (P : Point) (L : Line) (h : point_outside_line P L) :
  ∃ (lines : Set Line), lines.CountLinesPassingThrough (P) ∧ 
  (∀ l ∈ lines, angle l L = 60) ∧ (lines.card = ∞) :=
sorry

end countless_lines_l370_370339


namespace limit_difference_quotient_at_2_l370_370709

theorem limit_difference_quotient_at_2 
  (f : ℝ → ℝ) 
  (h : deriv f 2 = 3) 
  : tendsto (λ (Δx : ℝ), (f (2 + Δx) - f 2) / Δx) (𝓝 0) (𝓝 3) := 
sorry

end limit_difference_quotient_at_2_l370_370709


namespace three_lines_intersecting_at_one_point_determine_planes_l370_370403

theorem three_lines_intersecting_at_one_point_determine_planes
  (L1 L2 L3 : Line) (P : Point)
  (h1 : L1.contains P) (h2 : L2.contains P) (h3 : L3.contains P) :
  (coplanar L1 L2 L3 → (∃ plane : Plane, plane.contains L1 ∧ plane.contains L2 ∧ plane.contains L3 ∧ unique_plane plane)) ∨ 
  (¬coplanar L1 L2 L3 → (∃ plane1 plane2 plane3 : Plane, plane1.contains L1 ∧ plane1.contains P ∧ plane2.contains L2 ∧ plane2.contains P ∧ plane3.contains L3 ∧ plane3.contains P ∧ unique_plane plane1 ∧ unique_plane plane2 ∧ unique_plane plane3)) :=
sorry

end three_lines_intersecting_at_one_point_determine_planes_l370_370403


namespace min_black_squares_possible_assignment_l370_370968

-- Define the conditions
def conditions : Prop :=
  ∃ (table : Array (Array (Option Int))),
    table.size = 7 ∧
    (∀i, i < 7 → (table[i].size = 7 ∧ table[i][0].isNone)) ∧
    table[0][0] = none ∧ table[0][6] = none ∧
    table[6][0] = none ∧ table[6][6] = none

-- Part a: Number of black squares
theorem min_black_squares (h : conditions) : 23 :=
  sorry

-- Part b: Integer assignment
theorem possible_assignment (h : conditions) : ∃ (assignment : Array (Array (Option Int))),
  (∀i j, i < 7 ∧ j < 7 ∧ assignment[i][j].isSome → 
    assignment[i][j] = some ((if table[i][j].isBlack then -5 else 6)) ∧
    (∀c, Greek_cross c assignment → sum c < 0) ∧
    sum (whole_table assignment) > 0 :=
  sorry

end min_black_squares_possible_assignment_l370_370968


namespace tom_fruits_left_l370_370456

-- Conditions
def initial_oranges : ℕ := 40
def initial_apples : ℕ := 70
def fraction_oranges_sold : ℚ := 1/4
def fraction_apples_sold : ℚ := 1/2

-- Calculations embedded in the statement
def oranges_sold : ℕ := (fraction_oranges_sold * initial_oranges : ℚ).natAbs
def apples_sold : ℕ := (fraction_apples_sold * initial_apples : ℚ).natAbs
def oranges_left : ℕ := initial_oranges - oranges_sold
def apples_left : ℕ := initial_apples - apples_sold
def total_fruits_left : ℕ := oranges_left + apples_left

-- Proof statement
theorem tom_fruits_left : total_fruits_left = 65 := by sorry

end tom_fruits_left_l370_370456


namespace alex_trips_needed_l370_370159

def money_saved := 14500
def car_cost := 14600
def per_trip_charge := 1.5
def grocery_percentage := 0.05
def grocery_value := 800
def additional_money_needed := car_cost - money_saved
def per_trip_earnings := per_trip_charge + grocery_percentage * grocery_value

theorem alex_trips_needed (money_saved car_cost per_trip_charge grocery_percentage grocery_value : ℕ) :
  additional_money_needed = 100 ∧ per_trip_earnings = 41.5 → T = 3 :=
by
  sorry

end alex_trips_needed_l370_370159


namespace volume_of_solid_of_revolution_l370_370995

noncomputable def sin_curve (x : ℝ) : ℝ := Real.sin (π * x / 2)

noncomputable def square_curve (x : ℝ) : ℝ := x^2

theorem volume_of_solid_of_revolution :
  let f := sin_curve, g := square_curve in
  ∫ x in (0 : ℝ)..1, π * (f x)^2 - π * (g x)^2 = (3 * π) / 10 :=
by
  sorry

end volume_of_solid_of_revolution_l370_370995


namespace correct_permutation_count_l370_370972

noncomputable def numValidPermutations (n : ℕ) : ℕ :=
  if n = 2018 then (1009.factorial) * (1010.factorial) else 0

theorem correct_permutation_count : numValidPermutations 2018 = (1009.factorial) * (1010.factorial) :=
by
  sorry

end correct_permutation_count_l370_370972


namespace num_balls_box_l370_370860

theorem num_balls_box (n : ℕ) (balls : Fin n → ℕ) (red blue : Fin n → Prop)
  (h_colors : ∀ i, red i ∨ blue i)
  (h_constraints : ∀ i j k,  red i ∨ red j ∨ red k ∧ blue i ∨ blue j ∨ blue k) : 
  n = 4 := 
sorry

end num_balls_box_l370_370860


namespace all_roots_real_l370_370221
-- Lean 4 Statement

def a : ℕ → ℕ → ℝ
| 0, 0 => 2
| 0, n => 1
| n, 0 => 2
| m, n => if hmn : m ≥ 1 ∧ n ≥ 1 then a (m-1) n + a m (n-1) else 0

def P_k (k : ℕ) : Polynomial ℝ :=
  Polynomial.sum (Finset.range (k + 1)) (λ i, Polynomial.C (a i (2*k + 1 - 2*i)) * Polynomial.X^i)

theorem all_roots_real (k : ℕ) : 
  ∀ root ∈ (Polynomial.roots (P_k k)), root ∈ set.univ := sorry

end all_roots_real_l370_370221


namespace derivative_value_at_third_l370_370342

noncomputable def f (x : ℝ) : ℝ := Real.log (2 - 3 * x)

theorem derivative_value_at_third : (derivative f) (1 / 3) = -3 := sorry

end derivative_value_at_third_l370_370342


namespace find_moles_of_sodium_bicarbonate_l370_370213

variable (HCl : Type) (NaHCO3 : Type) (NaCl : Type) (CO2 : Type) (H2O : Type)
variables hydrochloric_acid sodium_bicarbonate sodium_chloride carbon_dioxide water : ℕ

theorem find_moles_of_sodium_bicarbonate :
  (hydrochloric_acid = 1) → (water = 1) → (sodium_bicarbonate = 1) :=
by
  intros hcl_cond h2o_cond
  sorry

end find_moles_of_sodium_bicarbonate_l370_370213


namespace poles_on_each_side_l370_370135

theorem poles_on_each_side (total_poles : ℕ) (sides_equal : ℕ)
  (h1 : total_poles = 104) (h2 : sides_equal = 4) : 
  (total_poles / sides_equal) = 26 :=
by
  sorry

end poles_on_each_side_l370_370135


namespace cost_of_softball_l370_370394

theorem cost_of_softball 
  (original_budget : ℕ)
  (dodgeball_cost : ℕ)
  (num_dodgeballs : ℕ)
  (increase_rate : ℚ)
  (num_softballs : ℕ)
  (new_budget : ℕ)
  (softball_cost : ℕ)
  (h0 : original_budget = num_dodgeballs * dodgeball_cost)
  (h1 : increase_rate = 0.20)
  (h2 : new_budget = original_budget + increase_rate * original_budget)
  (h3 : new_budget = num_softballs * softball_cost) :
  softball_cost = 9 :=
by
  sorry

end cost_of_softball_l370_370394


namespace vertical_asymptote_exists_l370_370632

theorem vertical_asymptote_exists (c : ℝ) :
  ((∀ x, f x = (x^2 - 2*x + c)/(x^2 + 2*x - 8)) 
    ∧ (x^2 + 2*x - 8 = (x - 2)*(x + 4)))
  → (c = -24 ∨ c = 0) :=
by
  sorry

where 
  f (x : ℝ) : ℝ := (x^2 - 2*x + c) / (x^2 + 2*x - 8)

end vertical_asymptote_exists_l370_370632


namespace solution_of_inequality_l370_370852

theorem solution_of_inequality : 
  {x : ℝ | x^2 - x - 2 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 2 < x} :=
by
  sorry

end solution_of_inequality_l370_370852


namespace length_of_longer_leg_of_smallest_triangle_l370_370200

theorem length_of_longer_leg_of_smallest_triangle 
  (hypotenuse_largest : ℝ) 
  (h1 : hypotenuse_largest = 10)
  (h45 : ∀ hyp, (hyp / Real.sqrt 2 / Real.sqrt 2 / Real.sqrt 2 / Real.sqrt 2) = hypotenuse_largest / 4) :
  (hypotenuse_largest / 4) = 5 / 2 := by
  sorry

end length_of_longer_leg_of_smallest_triangle_l370_370200


namespace length_of_AP_l370_370736

theorem length_of_AP 
  (YC CZ side_length : ℝ)
  (hYC : YC = 8)
  (hCZ : CZ = 15)
  (hSideLength : side_length = 9)
  (triangle_ABC_is_equilateral : (ΔABC sides := [side_length, side_length, side_length]))
  (triangle_PQR_is_equilateral : (ΔPQR sides := [side_length, side_length, side_length]))
  (B_on_CZ : B ∈ side CZ)
  (R_on_YQ : R ∈ side YQ) :
  distance A P = 10 := by
  sorry

end length_of_AP_l370_370736


namespace gcd_lcm_find_other_number_l370_370840

theorem gcd_lcm_find_other_number {a b : ℕ} (h_gcd : Nat.gcd a b = 36) (h_lcm : Nat.lcm a b = 8820) (h_a : a = 360) : b = 882 :=
by
  sorry

end gcd_lcm_find_other_number_l370_370840


namespace range_of_a_l370_370385

noncomputable def func (x a : ℝ) : ℝ := -x^2 - 2 * a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → func x a ≤ a^2) →
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ func x a = a^2) →
  -1 ≤ a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l370_370385


namespace cost_of_baguette_l370_370765

-- Define necessary conditions
def has_money (initial_money : ℕ) := initial_money = 50
def cost_per_water_bottle (cost : ℕ) := cost = 1
def num_baguettes (num : ℕ) := num = 2
def num_water_bottles (num : ℕ) := num = 2
def money_left (left_money : ℕ) := left_money = 44

-- Define what we need to prove
theorem cost_of_baguette:
  ∀ (initial_money cost_water num_baguettes num_water_bottles left_money cost_baguette : ℕ),
  has_money initial_money →
  cost_per_water_bottle cost_water →
  num_baguettes num_baguettes →
  num_water_bottles num_water_bottles →
  money_left left_money →
  initial_money - left_money = 2 * cost_baguette + num_water_bottles * cost_water →
  cost_baguette = 2 :=
by
  intros
  sorry

end cost_of_baguette_l370_370765


namespace sin_210_eq_neg_half_l370_370555

theorem sin_210_eq_neg_half : Real.sin (210 * Real.pi / 180) = -1 / 2 := by
  -- We use the given angles and their known sine values.
  have angle_30 := Real.pi / 6
  have sin_30 := Real.sin angle_30
  -- Expression for the sine of 210 degrees in radians.
  have angle_210 := 210 * Real.pi / 180
  -- Proving the sine of 210 degrees using angle addition formula and unit circle properties.
  calc
    Real.sin angle_210 
    -- 210 degrees is 180 + 30 degrees, translating to pi and pi/6 in radians.
    = Real.sin (Real.pi + Real.pi / 6) : by rw [←Real.ofReal_nat_cast, ←Real.ofReal_nat_cast, Real.ofReal_add, Real.ofReal_div, Real.ofReal_nat_cast]
    -- Using the sine addition formula: sin(pi + x) = -sin(x).
    ... = - Real.sin (Real.pi / 6) : by exact Real.sin_add_pi_div_two angle_30
    -- Substituting the value of sin(30 degrees).
    ... = - 1 / 2 : by rw sin_30

end sin_210_eq_neg_half_l370_370555


namespace correct_transformation_l370_370668

noncomputable def C1 : ℝ → ℝ := λ x, Real.cos x
noncomputable def C2 : ℝ → ℝ := λ x, Real.sin (2 * x + 2 * Real.pi / 3)

theorem correct_transformation :
  (λ x, Real.cos (2 * x - Real.pi / 6)) = (λ x, Real.sin (2 * x + 2 * Real.pi / 3)) :=
sorry

end correct_transformation_l370_370668


namespace shaded_area_after_100_iterations_l370_370869

noncomputable def triangle_area (a b : ℝ) : ℝ := (1/2) * a * b

noncomputable def total_shaded_area (initial_area : ℝ) (iterations : ℕ) : ℝ :=
  initial_area * (4/3) * (1 - (1/4) ^ iterations)

theorem shaded_area_after_100_iterations :
  let XY := 8 in
  let YZ := 8 in
  let area_XYZ := triangle_area XY YZ in
  total_shaded_area (area_XYZ / 4) 100 ≈ 10.67 :=
by
  let XY := 8
  let YZ := 8
  let area_XYZ := triangle_area XY YZ
  have h1 : area_XYZ = 32 := sorry
  have h2 : total_shaded_area (area_XYZ / 4) 100 ≈ 10.67 := sorry
  exact h2

end shaded_area_after_100_iterations_l370_370869


namespace problem1_problem2_l370_370662

noncomputable def f (a b c d : ℝ) (x : ℝ) := a * x ^ 3 + b * x ^ 2 + c * x + d
noncomputable def f_deriv (a b c : ℝ) (x : ℝ) := 3 * a * x ^ 2 + 2 * b * x + c
noncomputable def g (f f_deriv : ℝ → ℝ) (x m : ℝ) := f x - f_deriv x - 6 * x - m

theorem problem1 (a b c d : ℝ) (h₀ : a ≠ 0)
  (h₁ : f a b c d 0 = 0)
  (h₂ : f_deriv a b c 1 = 0)
  (h₃ : ∃ (x : ℝ), x = -1 ∧ f a b c d x = 2 ∧ f_deriv a b c x = 0) :
  f a b c d = (fun x => x^3 - 3 * x^2) :=
sorry

theorem problem2 (a b c d : ℝ) (h₀ : f a b c d = (fun x => x^3 - 3 * x^2)) :
  ∀ x ∈ Icc (-2 : ℝ) 4, f a b c d x ≥ f_deriv a b c x + 6 * x + (-24 : ℝ) :=
sorry

end problem1_problem2_l370_370662


namespace collinear_midpoints_of_intersects_triangle_l370_370526

variables (A B C A₁ B₁ C₁ : Point)

-- Assuming a line intersects triangle ABC at points A₁, B₁, and C₁
axiom intersects_sides_at : 
  ∃ (l : Line), intersects l (segment A B) A₁ ∧ 
                 intersects l (segment B C) B₁ ∧ 
                 intersects l (segment C A) C₁

-- Define midpoints M_A, M_B, and M_C
def M_A := midpoint (segment A A₁)
def M_B := midpoint (segment B B₁)
def M_C := midpoint (segment C C₁)

-- Prove the midpoints M_A, M_B, and M_C are collinear
theorem collinear_midpoints_of_intersects_triangle :
  collinear {M_A, M_B, M_C} :=
sorry

end collinear_midpoints_of_intersects_triangle_l370_370526


namespace train_passing_time_correct_l370_370728

-- Define the given conditions
def train_length : ℝ := 150 -- meters
def train_speed_kmh : ℝ := 36 -- km/hr

-- Define the conversion factor from km/hr to m/s
def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

-- Converted speed of the train
def train_speed_ms : ℝ := kmh_to_ms train_speed_kmh

-- Define the time it takes for the train to pass an oak tree
def passing_time (length : ℝ) (speed : ℝ) : ℝ := length / speed

-- The main statement to prove
theorem train_passing_time_correct : passing_time train_length train_speed_ms = 15 := by
  sorry

end train_passing_time_correct_l370_370728


namespace Taylor_needs_14_jars_l370_370969

noncomputable def standard_jar_volume : ℕ := 60
noncomputable def big_container_volume : ℕ := 840

theorem Taylor_needs_14_jars : big_container_volume / standard_jar_volume = 14 :=
by sorry

end Taylor_needs_14_jars_l370_370969


namespace bus_stop_time_l370_370111

-- Define the known values in the conditions
def speed_without_stops := 75  -- in kmph
def speed_with_stops := 45     -- in kmph

-- Define the goal
theorem bus_stop_time : ∃ t : ℕ, t = 24 ∧ generate_stop_time_conditions
  Δt := speed_without_stops - speed_with_stops ∧
  t = (Δt * 60) / speed_without_stops :=
begin
  -- statement of the theorem in Lean
  sorry  -- the proof goes here
end

end bus_stop_time_l370_370111


namespace max_n_for_2002_l370_370079

def exists_max_n (m : ℕ) : ℕ :=
  Max {n | ∃ (k : Fin n → ℕ), (∀ i j, i ≠ j → k i ≠ k j) ∧ (∑ i, k i^2 = m)}

theorem max_n_for_2002 :
  exists_max_n 2002 = 16 := sorry

end max_n_for_2002_l370_370079


namespace correlation_is_high_l370_370819

def linear_correlation (x y : ℝ) : Prop := x = -0.87 → y = "highly related"

theorem correlation_is_high : ∀ x, linear_correlation x "highly related" :=
by
  sorry

end correlation_is_high_l370_370819


namespace angles_of_triangle_ABC_are_45_45_90_l370_370938

theorem angles_of_triangle_ABC_are_45_45_90
    (A B C : Type)
    (ABC : triangle A B C)
    (M N K L : Type)
    (circle : Type)
    (tangent_to_BC_at_M : tangent circle (segment B C) M)
    (tangent_to_AC_at_N : tangent circle (segment A C) N)
    (intersects_AB_at_KL : intersects circle (segment A B) K L)
    (KLMN_is_square : square K L M N) :
    angles_of_triangle ABC = (45°, 45°, 90°) :=
by
    sorry

end angles_of_triangle_ABC_are_45_45_90_l370_370938


namespace simplify_cube_root_of_5488000_l370_370017

theorem simplify_cube_root_of_5488000 :
  ∛(5488000) = 140 * 2^(1/3 : ℝ) :=
by
  have h1: 5488000 = 10^3 * 5488, by norm_num
  have h2: 5488 = 2^4 * 7^3, by norm_num
  sorry

end simplify_cube_root_of_5488000_l370_370017


namespace convert_1987_to_base5_l370_370597

-- Function to convert a decimal number to base 5
def convert_to_base5 (n : ℕ) : ℕ :=
  let rec helper (n : ℕ) (base5 : ℕ) : ℕ := 
    if n = 0 then base5 
    else helper (n / 5) (base5 * 10 + (n % 5))
  helper n 0

-- The theorem we need to prove
theorem convert_1987_to_base5 : convert_to_base5 1987 = 30422 := 
by 
  -- Proof is omitted 
  -- Assertion of the fact according to the problem statement
  sorry

end convert_1987_to_base5_l370_370597


namespace find_b_l370_370743

-- Define the sample points and the conditions
variables {x : ℕ → ℝ} {y : ℕ → ℝ}
def conditions (n : ℕ) : Prop :=
  (∑ i in finset.range n, x i) = 11 ∧
  (∑ i in finset.range n, y i) = 13 ∧
  (∑ i in finset.range n, (x i)^2) = 21

-- The main theorem to demonstrate
theorem find_b (b : ℝ) (h : conditions 6) : b = 5 / 7 :=
by {
  -- Proof content would be here,
  sorry
}

end find_b_l370_370743


namespace number_of_tiles_required_l370_370150

-- Defining the room dimensions in feet
def room_length : ℝ := 15
def room_width : ℝ := 18

-- Converting tile dimensions from inches to feet
def tile_length : ℝ := 6 / 12  -- 6 inches to feet
def tile_width : ℝ := 8 / 12  -- 8 inches to feet

-- Defining the area of each tile in square feet
def tile_area : ℝ := tile_length * tile_width  -- (1/2 feet) * (2/3 feet) = 1/3 square feet

-- Defining the area of the floor in square feet
def floor_area : ℝ := room_length * room_width  -- 15 feet * 18 feet = 270 square feet

-- Theorem stating that the number of tiles required is 810
theorem number_of_tiles_required : floor_area / tile_area = 810 :=
by 
  sorry

end number_of_tiles_required_l370_370150


namespace simplify_and_evaluate_expression_l370_370367

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 5 + 1) : 
  ( ( (x^2 - 1) / x ) / (1 + 1 / x) ) = Real.sqrt 5 :=
by 
  sorry

end simplify_and_evaluate_expression_l370_370367


namespace sequence_sum_target_sum_l370_370690

noncomputable def a (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * (n + 1)

theorem sequence_sum (n : ℕ) (hn : n ≠ 0) :
  (∑ k in Finset.range n, a (k+1)) = n ^ 2 + 3 * n :=
by
  sorry

theorem target_sum (n : ℕ) (hn : n ≠ 0) :
  (∑ k in Finset.range n, a (k+1)^2 / (k+2)) = 2 * n^2 + 6 * n :=
by
  sorry

end sequence_sum_target_sum_l370_370690


namespace find_a_sum_binom_coeff_l370_370264

-- Definitions for part (1)
def binomial_coeff_6_x2 (a : ℝ) := 60 * a ^ 4
def coef_given := 960
def eq_a_part1 := binomial_coeff_6_x2 a = coef_given

-- Theorem for part (1)
theorem find_a (a : ℝ) (h : eq_a_part1) : a = 2 := 
sorry

-- Definitions for part (2)
def sum_coeffs_expansion (a n : ℝ) := (a + 2)^n
def sum_given := 3 ^ 10
def n_given := 5
def a_given := 7
def eq_n_a := a + n = 12 
def eq_sum := sum_coeffs_expansion a n = sum_given

-- Theorem for part (2)
theorem sum_binom_coeff (a n : ℝ) (h1 : eq_sum) (h2 : eq_n_a) :
  2 ^ n = 32 := 
sorry

end find_a_sum_binom_coeff_l370_370264


namespace ratio_of_triangle_to_square_l370_370730

theorem ratio_of_triangle_to_square (x : ℝ) (h1 : 0 < x) :
  let a := 4 * x in
  let abce_area := a * a in
  let bfd_area := 7 * x * x in
  (bfd_area / abce_area) = 7 / 16 := by
  sorry

end ratio_of_triangle_to_square_l370_370730


namespace impossible_option_l370_370654

theorem impossible_option (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
(h4 : log x / log 2 = log y / log 3) (h5 : log y / log 3 = log z / log 5) : 
¬ (y / 3 < z / 5 ∧ z / 5 < x / 2) :=
by
  sorry

end impossible_option_l370_370654


namespace AR_eq_AS_l370_370635

noncomputable def CircleCenter (O : Point) (A : Point) (l : Line) : Prop :=
  Perpendicular O A l

def EquidistantPoints (A B C : Point) (l : Line) : Prop :=
  OnLine B l ∧ OnLine C l ∧ Distance A B = Distance A C

def SecantsIntersect (P Q M N : Point) (circle : Circle) (B C : Point) : Prop :=
  OnSecant P Q B circle ∧ OnSecant M N C circle

def IntersectWithLine (PM QN : Line) (l : Line) (R S : Point) : Prop :=
  Intersect PM l R ∧ Intersect QN l S

def EqualSegments (A R S : Point) : Prop :=
  Distance A R = Distance A S

theorem AR_eq_AS
  (O A B C P Q M N R S : Point)
  (l PM QN : Line)
  (circle : Circle) :
  CircleCenter O A l →
  EquidistantPoints A B C l →
  SecantsIntersect P Q M N circle B C →
  IntersectWithLine PM QN l R S →
  EqualSegments A R S := by
  sorry

end AR_eq_AS_l370_370635


namespace ones_digit_of_six_pow_45_l370_370464

theorem ones_digit_of_six_pow_45 : 
  ∃ n, n ≥ 1 ∧ 6^45 % 10 = 6 :=
by
  use 45
  split
  · exact Nat.le_refl 45
  · sorry

end ones_digit_of_six_pow_45_l370_370464


namespace bridge_length_l370_370888

/-- The length of the bridge, given the conditions about the train’s length, speed, and crossing time, is 666.85 meters. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 250 → train_speed_kmph = 60 → crossing_time = 55 → 
  ∃ bridge_length : ℝ, bridge_length = 666.85 :=
begin
  intros h_train_length h_train_speed h_crossing_time,
  sorry
end

end bridge_length_l370_370888


namespace simplify_expr1_simplify_expr2_l370_370815

theorem simplify_expr1 (a : ℝ) : 2 * (a - 1) - (2 * a - 3) + 3 = 4 :=
by
  sorry

theorem simplify_expr2 (x : ℝ) : 3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = 5 * x^2 - 3 * x - 3 :=
by
  sorry

end simplify_expr1_simplify_expr2_l370_370815


namespace domain_of_f_range_of_f_a_greater_than_1_range_of_f_0_less_than_a_less_than_1_l370_370640

variable {a : ℝ} (h₁ : a > 0) (h₂ : a ≠ 1)

def f (x : ℝ) : ℝ := log a (1 - x)

theorem domain_of_f : ∀ x : ℝ, x < 1 ↔ ∃ y : ℝ, y = f x := 
by
  sorry

theorem range_of_f_a_greater_than_1 :
  ∀ x : ℝ, a > 1 → (f x > 0 ↔ x < 0) := 
by
  sorry

theorem range_of_f_0_less_than_a_less_than_1 :
  ∀ x : ℝ, 0 < a ∧ a < 1 → (f x > 0 ↔ 0 < x ∧ x < 1) := 
by
  sorry

end domain_of_f_range_of_f_a_greater_than_1_range_of_f_0_less_than_a_less_than_1_l370_370640


namespace sufficient_but_not_necessary_l370_370708

def p (x : ℝ) : Prop := x > 0
def q (x : ℝ) : Prop := |x| > 0

theorem sufficient_but_not_necessary (x : ℝ) : 
  (p x → q x) ∧ (¬(q x → p x)) :=
by
  sorry

end sufficient_but_not_necessary_l370_370708


namespace range_of_m_l370_370380

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (3 * x^2 + 2 * x + 2) / (x^2 + x + 1) ≥ m) ↔ (m ≤ 2) :=
by
  sorry

end range_of_m_l370_370380


namespace sum_of_integer_values_n_l370_370088

theorem sum_of_integer_values_n (h : ∀ n : ℤ, (30 / (2 * n - 1)) ∈ ℤ → n ∈ {1, 2, 3, 8}) : 
∑ n in {1, 2, 3, 8}, n = 14 := 
by 
     sorry

end sum_of_integer_values_n_l370_370088


namespace number_of_numbers_l370_370045

theorem number_of_numbers (N : ℕ) (h_avg : (18 * N + 40) / N = 22) : N = 10 :=
by
  sorry

end number_of_numbers_l370_370045


namespace base_k_132_eq_30_l370_370926

theorem base_k_132_eq_30 (k : ℕ) (h : 1 * k^2 + 3 * k + 2 = 30) : k = 4 :=
sorry

end base_k_132_eq_30_l370_370926


namespace number_of_valid_numbers_l370_370283

-- Define a predicate for a three-digit number
def is_three_digit (n : Nat) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define a predicate for containing the digit 3
def contains_digit_three (n : Nat) : Prop :=
  (n / 100) = 3 ∨ (n / 10 % 10) = 3 ∨ (n % 10) = 3

-- Define a predicate for being divisible by 3
def is_divisible_by_three (n : Nat) : Prop :=
  n % 3 = 0

-- Define a predicate that combines all conditions
def valid_number (n : Nat) : Prop :=
  is_three_digit n ∧ contains_digit_three n ∧ is_divisible_by_three n

-- Statement of the theorem
theorem number_of_valid_numbers : Finset.card (Finset.filter valid_number (Finset.range 900)) = 80 :=
by
  sorry

end number_of_valid_numbers_l370_370283


namespace minimum_length_MN_l370_370731

noncomputable def min_length_MN : ℝ :=
  let C := (1, 1) : ℝ × ℝ
  let A := (2, 0) : ℝ × ℝ
  let B := (0, 2) : ℝ × ℝ
  let r := 1 : ℝ
  let OC_perp_MN_at_tangent := true in
  let M_on_OA := true in
  let N_on_OB := true in
  2 * sqrt 2 - 2

theorem minimum_length_MN :
  let C := (1, 1) : ℝ × ℝ
  let A := (2, 0) : ℝ × ℝ
  let B := (0, 2) : ℝ × ℝ
  let r := 1 : ℝ
  let OC_perp_MN_at_tangent := true in
  let M_on_OA := true in
  let N_on_OB := true in
  is_tangent_segment MN C → |MN| = 2 * sqrt 2 - 2 := by
  sorry

end minimum_length_MN_l370_370731


namespace complement_intersection_l370_370344

open Set

-- Definitions of sets U, M, and N
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- The theorem stating the complement of M ∩ N in U is {1, 4}
theorem complement_intersection (U M N : Set ℕ) (hU : U = {1, 2, 3, 4}) (hM : M = {1, 2, 3}) (hN : N = {2, 3, 4}) :
  compl (M ∩ N) ∈ U = {1, 4} :=
by
  sorry

#check @complement_intersection

end complement_intersection_l370_370344


namespace quadrilateral_is_parallelogram_if_diagonals_equal_l370_370009

variables {A B C D : Type}
variables [AffineSpace A] [AffineSpace B] [AffineSpace C] [AffineSpace D]

def diags_equal_and_parallelogram (ABCD : Quadrilateral A) : Prop :=
  let (a, b, c, d) := ABCD in
  (diagonal AC = diagonal BD) → (is_parallelogram ABCD)

theorem quadrilateral_is_parallelogram_if_diagonals_equal (ABCD : Quadrilateral A) :
  diags_equal_and_parallelogram ABCD :=
by
  sorry

end quadrilateral_is_parallelogram_if_diagonals_equal_l370_370009


namespace lcm_12_15_18_l370_370882

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end lcm_12_15_18_l370_370882


namespace integer_diff_of_two_squares_l370_370810

theorem integer_diff_of_two_squares (m : ℤ) : 
  (∃ x y : ℤ, m = x^2 - y^2) ↔ (∃ k : ℤ, m ≠ 4 * k + 2) := by
  sorry

end integer_diff_of_two_squares_l370_370810


namespace triangle_PR_eq_8_l370_370311

open Real

theorem triangle_PR_eq_8 (P Q R M : ℝ) 
  (PQ QR PM : ℝ) 
  (hPQ : PQ = 6) (hQR : QR = 10) (hPM : PM = 5) 
  (M_midpoint : M = (Q + R) / 2) :
  dist P R = 8 :=
by
  sorry

end triangle_PR_eq_8_l370_370311


namespace price_of_adult_ticket_l370_370496

theorem price_of_adult_ticket
  (price_child : ℤ)
  (price_adult : ℤ)
  (num_adults : ℤ)
  (num_children : ℤ)
  (total_amount : ℤ)
  (h1 : price_adult = 2 * price_child)
  (h2 : num_adults = 400)
  (h3 : num_children = 200)
  (h4 : total_amount = 16000) :
  num_adults * price_adult + num_children * price_child = total_amount → price_adult = 32 := by
    sorry

end price_of_adult_ticket_l370_370496


namespace shaded_area_k_l370_370307

-- Definitions based on problem conditions
def triangle_ABC : Prop :=
  let AC := 3
  let AB := 4
  let BC := 5
  right_triangle ABC AC AB BC

def circle_O_P : Prop :=
  ∃ rO rP : ℝ, (rO = 2 * rP) ∧ area_circle_O = 4 * area_circle_P

def radii_conditions : Prop :=
  ∃ y x : ℝ, (2 * y + 2 * x = 3) ∧ (2 * y + 4 * x = 4)

-- Proving the value of k
theorem shaded_area_k : triangle_ABC → circle_O_P → radii_conditions → k = 9 / 8 :=
by
  sorry

end shaded_area_k_l370_370307


namespace max_square_plots_l370_370534

theorem max_square_plots (length width available_fencing : ℕ) 
(h : length = 30 ∧ width = 60 ∧ available_fencing = 2500) : 
  ∃ n : ℕ, n = 72 ∧ ∀ s : ℕ, ((30 * (60 / s - 1)) + (60 * (30 / s - 1)) ≤ 2500) → ((30 / s) * (60 / s) = n) := by
  sorry

end max_square_plots_l370_370534


namespace touching_line_eq_l370_370479

theorem touching_line_eq (f : ℝ → ℝ) (f_def : ∀ x, f x = 3 * x^4 - 4 * x^3) : 
  ∃ l : ℝ → ℝ, (∀ x, l x = - (8 / 9) * x - (4 / 27)) ∧ 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (f x₁ = l x₁ ∧ f x₂ = l x₂) :=
by sorry

end touching_line_eq_l370_370479


namespace sugar_amount_l370_370747

theorem sugar_amount (S F B : ℕ) (h1 : S = 5 * F / 4) (h2 : F = 10 * B) (h3 : F = 8 * (B + 60)) : S = 3000 := by
  sorry

end sugar_amount_l370_370747


namespace find_cos_theta_l370_370671

theorem find_cos_theta :
  let coeff1 := Nat.choose 5 3 * (cos θ)^2 -- Coefficient of x^2
  let coeff2 := Nat.choose 4 1 * (5 / 4 : ℝ) -- Coefficient of x^3
  coeff1 = coeff2 →
  cos θ = sqrt 2 / 2 ∨ cos θ = -sqrt 2 / 2 :=
by
  intros coeff_eq
  sorry

end find_cos_theta_l370_370671


namespace max_a_value_l370_370183

noncomputable def max_slope_no_lattice_points (a : ℚ) : Prop :=
  ∀ (m : ℚ), (1 / 3 < m ∧ m < a) → 
  ∀ (x : ℤ), (1 ≤ x ∧ x ≤ 50) → 
  ∀ (y : ℤ), y ≠ m * x + 3

theorem max_a_value : max_slope_no_lattice_points (17/51) := sorry

end max_a_value_l370_370183


namespace total_arrangements_l370_370139

theorem total_arrangements (volunteers elderly : ℕ) (total_positions : ℕ)
  (elderly_unit_positions valid_positions : ℕ)
  (arrangements_within_unit arranegements_of_units : ℕ)
  (valid_arrangements : ℕ):
  volunteers = 5 → elderly = 2 → total_positions = 7 →
  elderly_unit_positions = 1 → valid_positions = 4 →
  arrangements_within_unit = 2 → arranegements_of_units = 6! →
  valid_arrangements = 960 →
  (valid_arrangements = (arranegements_of_units * arrangements_within_unit)) → 
  True :=
by sorry

end total_arrangements_l370_370139


namespace A_alone_completion_l370_370900

noncomputable def work_completion_days (W A B: ℝ) : ℝ :=
  if h : A + B = W / 30 ∧ 20 * (A + B) + 20 * A = W then 60 else 0

theorem A_alone_completion (A B : ℝ) (W : ℝ) (h1 : A + B = W / 30)
  (h2 : 20 * (A + B) + 20 * A = W) : 
  work_completion_days W A B = 60 :=
by
  unfold work_completion_days
  rw [if_pos]
  apply rfl
  exact ⟨h1, h2⟩

end A_alone_completion_l370_370900


namespace average_speed_of_bus_trip_l370_370505

theorem average_speed_of_bus_trip
  (v : ℝ)
  (distance : ℝ)
  (time_difference : ℝ)
  (speed_increment : ℝ)
  (original_time : ℝ := distance / v)
  (faster_time : ℝ := distance / (v + speed_increment))
  (h1 : distance = 360)
  (h2 : time_difference = 1)
  (h3 : speed_increment = 5)
  (h4 : original_time - time_difference = faster_time) :
  v = 40 :=
by
  sorry

end average_speed_of_bus_trip_l370_370505


namespace find_arithmetic_sequence_l370_370069

theorem find_arithmetic_sequence (a d : ℝ) : 
(a - d) + a + (a + d) = 6 ∧ (a - d) * a * (a + d) = -10 → 
  (a = 2 ∧ d = 3 ∨ a = 2 ∧ d = -3) :=
by
  sorry

end find_arithmetic_sequence_l370_370069


namespace balls_still_in_baskets_l370_370513

theorem balls_still_in_baskets 
  (tennis_balls_per_basket : ℕ)
  (soccer_balls_per_basket : ℕ)
  (baskets : ℕ)
  (students_3_removed : ℕ)
  (students_2_removed : ℕ)
  (students_3 : ℕ := 3)
  (students_2 : ℕ := 2) :
  students_3_removed = 8 → 
  students_2_removed = 10 → 
  tennis_balls_per_basket = 15 → 
  soccer_balls_per_basket = 5 → 
  baskets = 5 → 
  15 * 5 + 5 * 5 - (students_3 * 8 + students_2 * 10) = 56 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end balls_still_in_baskets_l370_370513


namespace first_group_people_count_l370_370130

def group_ice_cream (P : ℕ) : Prop :=
  let total_days_per_person1 := P * 10
  let total_days_per_person2 := 5 * 16
  total_days_per_person1 = total_days_per_person2

theorem first_group_people_count 
  (P : ℕ) 
  (H1 : group_ice_cream P) : 
  P = 8 := 
sorry

end first_group_people_count_l370_370130


namespace square_plot_area_l370_370517

-- initially declare condition parameters
variable (cost_per_foot : ℝ) (total_cost : ℝ)

-- declare proposition to be proven
theorem square_plot_area (h1: cost_per_foot = 56) (h2: total_cost = 3808) : 
  let P := total_cost / cost_per_foot in
  let s := P / 4 in
  let A := s^2 in
  A = 289 := by
  -- sorry used to skip full proof
  sorry

end square_plot_area_l370_370517


namespace expected_hit_targets_correct_expected_hit_targets_at_least_half_l370_370799

noncomputable def expected_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - (1 : ℝ) / n)^n)

theorem expected_hit_targets_correct (n : ℕ) (h_pos : n > 0) :
  expected_hit_targets n = n * (1 - (1 - (1 : ℝ) / n)^n) :=
by
  unfold expected_hit_targets
  sorry

theorem expected_hit_targets_at_least_half (n : ℕ) (h_pos : n > 0) :
  expected_hit_targets n >= n / 2 :=
by
  unfold expected_hit_targets
  sorry

end expected_hit_targets_correct_expected_hit_targets_at_least_half_l370_370799


namespace correct_exponentiation_l370_370098

theorem correct_exponentiation (a : ℝ) : (-2 * a^3) ^ 4 = 16 * a ^ 12 :=
by sorry

end correct_exponentiation_l370_370098


namespace B_can_do_work_in_8_days_l370_370126

theorem B_can_do_work_in_8_days (A_work_6_days : ℕ)
                                (total_payment : ℕ)
                                (C_payment : ℕ)
                                (work_days_with_C : ℕ)
                                (C_work_share : ℕ) :
  A_work_6_days = 6 →
  work_days_with_C = 3 →
  total_payment = 5000 →
  C_payment = 625 →
  C_work_share = 24 →
  ∃ (B : ℕ), B = 8 := by
  intros h1 h2 h3 h4 h5
  use 8
  sorry

end B_can_do_work_in_8_days_l370_370126


namespace kolya_walking_speed_is_correct_l370_370768

-- Conditions
def distance_traveled := 3 * x -- Total distance
def initial_speed := 10 -- Initial speed in km/h
def doubled_speed := 20 -- Doubled speed in km/h
def total_time_to_store_closing := distance_traveled / initial_speed -- Time to store's closing

-- Times for each segment
def time_first_segment := x / initial_speed
def time_second_segment := x / doubled_speed
def time_first_two_thirds := time_first_segment + time_second_segment
def remaining_time := total_time_to_store_closing - time_first_two_thirds

-- Prove Kolya's walking speed is 20/3 km/h
theorem kolya_walking_speed_is_correct :
  (x / remaining_time) = (20 / 3) :=
by
  sorry

end kolya_walking_speed_is_correct_l370_370768


namespace intersecting_points_value_l370_370740

open Real

noncomputable def point_P : ℝ × ℝ := (3, sqrt 3)
noncomputable def polar_curve_eq (ρ θ : ℝ) : Prop := ρ^2 * cos (2 * θ) = 9

theorem intersecting_points_value :
  ∀ A B : ℝ × ℝ, 
  (exists t : ℝ, A = (3 + (sqrt 3)/2 * t, sqrt 3 + (1/2) * t) ∧ 
   polar_curve_eq (sqrt ((3 + (sqrt 3)/2 * t)^2 + (sqrt 3 + (1/2) * t)^2))
                 (atan2 (sqrt 3 + (1/2) * t) (3 + (sqrt 3)/2 * t))) ∧
  (exists t : ℝ, B = (3 + (sqrt 3)/2 * t, sqrt 3 + (1/2) * t) ∧ 
   polar_curve_eq (sqrt ((3 + (sqrt 3)/2 * t)^2 + (sqrt 3 + (1/2) * t)^2))
                 (atan2 (sqrt 3 + (1/2) * t) (3 + (sqrt 3)/2 * t))) →
  let PA := sqrt ((A.1 - 3) ^ 2 + (A.2 - (sqrt 3)) ^ 2),
      PB := sqrt ((B.1 - 3) ^ 2 + (B.2 - (sqrt 3)) ^ 2) in
  (1 / |PA|) + (1 / |PB|) = sqrt 2 := 
sorry

end intersecting_points_value_l370_370740


namespace Clare_has_more_pencils_than_Jeanine_l370_370316

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

end Clare_has_more_pencils_than_Jeanine_l370_370316


namespace Mary_investment_l370_370486

def simple_interest (P r t : ℝ) : ℝ := P * (1 + r * t)

theorem Mary_investment : ∃ P r : ℝ, simple_interest P r 2 = 260 ∧ simple_interest P r 5 = 350 ∧ P = 200 :=
by
  use [200, (30 / 400)] -- these values are derived from solving the equations
  sorry

end Mary_investment_l370_370486


namespace max_height_of_structure_l370_370244

noncomputable def triangle_PQR (PQ QR RP : ℝ) (h_R h_Q : ℝ) : ℝ :=
  let A : ℝ := 210 -- calculated area
  let h    : ℝ := (h_R * h_Q) / (h_R + h_Q)
  h

theorem max_height_of_structure :
  ∃ h, ∀ PQ QR RP h_R h_Q, PQ = 20 → QR = 21 → RP = 29 →
    h_R = (2 * 210) / 20 → h_Q = (2 * 210) / 29 →
    h = triangle_PQR PQ QR RP h_R h_Q ∧ abs (h - 8.57) < 0.01 :=
begin
  sorry
end

end max_height_of_structure_l370_370244


namespace treasure_coins_l370_370445

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l370_370445


namespace coffee_cost_correct_l370_370473

noncomputable def total_cost : ℝ := 
  let mom_donuts := 8
  let dad_donuts := 12
  let sister_donuts := 16
  let brother_donuts := 10
  let grandma_donuts := 6
  
  let french_roast_ounces_per_donut := 2
  let columbian_roast_ounces_per_donut := 3
  let ethiopian_roast_ounces_per_donut := 4
  let sumatran_roast_ounces_per_donut := 1.5

  let french_roast_ounces_needed := mom_donuts * french_roast_ounces_per_donut
  let columbian_roast_ounces_needed := dad_donuts * columbian_roast_ounces_per_donut + sister_donuts * columbian_roast_ounces_per_donut
  let ethiopian_roast_ounces_needed := brother_donuts * ethiopian_roast_ounces_per_donut
  let sumatran_roast_ounces_needed := grandma_donuts * sumatran_roast_ounces_per_donut

  let french_roast_pot_ounces := 12
  let columbian_roast_pot_ounces := 15
  let ethiopian_roast_pot_ounces := 20
  let sumatran_roast_pot_ounces := 10
  
  let french_roast_pot_cost := 3
  let columbian_roast_pot_cost := 4
  let ethiopian_roast_pot_cost := 5
  let sumatran_roast_pot_cost := 3.5

  let french_roast_pots := Real.ceil (french_roast_ounces_needed / french_roast_pot_ounces)
  let columbian_roast_pots := Real.ceil (columbian_roast_ounces_needed / columbian_roast_pot_ounces)
  let ethiopian_roast_pots := Real.ceil (ethiopian_roast_ounces_needed / ethiopian_roast_pot_ounces)
  let sumatran_roast_pots := Real.ceil (sumatran_roast_ounces_needed / sumatran_roast_pot_ounces)

  (french_roast_pots * french_roast_pot_cost) + (columbian_roast_pots * columbian_roast_pot_cost) + 
  (ethiopian_roast_pots * ethiopian_roast_pot_cost) + (sumatran_roast_pots * sumatran_roast_pot_cost)

theorem coffee_cost_correct : total_cost = 43.5 := 
  by
    sorry

end coffee_cost_correct_l370_370473


namespace store_A_more_advantageous_l370_370937

theorem store_A_more_advantageous (x : ℕ) (h : x > 5) : 
  6000 + 4500 * (x - 1) < 4800 * x := 
by 
  sorry

end store_A_more_advantageous_l370_370937


namespace area_difference_l370_370383

-- Define the original and new rectangle dimensions
def original_rect_area (length width : ℕ) : ℕ := length * width
def new_rect_area (length width : ℕ) : ℕ := (length - 2) * (width + 2)

-- Define the problem statement
theorem area_difference (a : ℕ) : new_rect_area a 5 - original_rect_area a 5 = 2 * a - 14 :=
by
  -- Insert proof here
  sorry

end area_difference_l370_370383


namespace simplify_cube_root_of_5488000_l370_370019

theorem simplify_cube_root_of_5488000 :
  ∛(5488000) = 140 * 2^(1/3 : ℝ) :=
by
  have h1: 5488000 = 10^3 * 5488, by norm_num
  have h2: 5488 = 2^4 * 7^3, by norm_num
  sorry

end simplify_cube_root_of_5488000_l370_370019


namespace pirate_treasure_l370_370447

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l370_370447


namespace treasure_coins_l370_370442

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l370_370442


namespace resistor_parallel_l370_370726

theorem resistor_parallel (x y r : ℝ)
  (h1 : x = 5)
  (h2 : r = 2.9166666666666665)
  (h3 : 1 / r = 1 / x + 1 / y) : y = 7 :=
by
  -- proof omitted
  sorry

end resistor_parallel_l370_370726


namespace towel_bleaching_area_decrease_l370_370905

theorem towel_bleaching_area_decrease 
  (L B : ℝ) 
  (hL : 0 < L) 
  (hB : 0 < B) : 
  let original_area := L * B
  let new_length := 0.80 * L
  let new_breadth := 0.90 * B
  let new_area := new_length * new_breadth
  let percentage_decrease := ((original_area - new_area) / original_area) * 100
  in percentage_decrease = 28 :=
by
  sorry

end towel_bleaching_area_decrease_l370_370905


namespace conjugate_quadrant_l370_370047

open Complex

theorem conjugate_quadrant (z : ℂ) (h : z * I = 2 - I) : -1 ≤ (conj z).re ∧ (conj z).im > 0 :=
by
let z := (2 - I) / I
have hz : z = -1 - 2 * I := by calc
  z = (2 - I) / I        : by rw [Complex.div_eq_self_iff.2] -- Simplifies division by i
    ... = -I * (2 - I) / (I * I) : by rw [Complex.div_div_eq_mul_div, div_self I_zero_ne]
    ... = -I * (2 - I) / (-1)   : by rw [Complex.I_mul_I]
    ... = I - 2 * I / 1       : by rw [I_mul, one_div_eq]
    ... = -1 - I * 2           : by simp
have hconj : conj z = -1 + 2 * I := by rw [hz, conj_neg_im]
exact ⟨by linarith, by linarith⟩

end conjugate_quadrant_l370_370047


namespace complex_polygon_area_l370_370224

theorem complex_polygon_area :
  let side_length := 8
  ∧ let θ2 := real.of_int 20
  ∧ let θ3 := real.of_int 45
  ∧ let θ4 := real.of_int 70
  ∧ let R := 4 * real.sqrt 2
  ∧ let n := 24
  ∧ let α := 15
  ∧ let side := 1.65
  ∧ let triangle_area := 8.2336
  (4 * triangle_area * n / 2 = 198) :=
  sorry

end complex_polygon_area_l370_370224


namespace Kolya_walking_speed_l370_370766

theorem Kolya_walking_speed
  (x : ℝ) 
  (h1 : x > 0) 
  (t_closing : ℝ := (3 * x) / 10) 
  (t_travel : ℝ := ((x / 10) + (x / 20))) 
  (remaining_time : ℝ := t_closing - t_travel)
  (walking_speed : ℝ := x / remaining_time)
  (correct_speed : ℝ := 20 / 3) :
  walking_speed = correct_speed := 
by 
  sorry

end Kolya_walking_speed_l370_370766


namespace solution_set_arcsin_inequality_l370_370026

noncomputable def f (x : ℝ) : ℝ := Real.arcsin (x^2) + Real.arcsin x + x^6 + x^3

theorem solution_set_arcsin_inequality :
  (∀ x, x ∈ set.Icc (-1 : ℝ) 1 → monotone f) →
  (∀ x, f x > f 0) →
  {x : ℝ | f x > 0} = set.Ioc 0 1 :=
by 
  intros h_mono h_gt
  sorry

end solution_set_arcsin_inequality_l370_370026


namespace european_postcards_cost_before_70s_l370_370920

-- Define the cost per postcard in cents
def postCardCostEuropean : ℕ := 7

-- Define the number of postcards from Germany and Italy before the 70's
def germanyPostcardsBefore70s : ℕ := 11
def italyPostcardsBefore70s : ℕ := 17

-- Define the conversion from cents to dollars
def centsToDollars (cents : ℕ) : ℝ := cents / 100

-- Define the total cost in cents for Germany and Italy
def totalCostEuropeanBefore70sInCents : ℕ :=
  germanyPostcardsBefore70s * postCardCostEuropean
  + italyPostcardsBefore70s * postCardCostEuropean

-- Define the total cost in dollars
def totalCostEuropeanBefore70s : ℝ :=
  centsToDollars totalCostEuropeanBefore70sInCents

-- The proof goal
theorem european_postcards_cost_before_70s :
  totalCostEuropeanBefore70s = 1.96 := by
  sorry

end european_postcards_cost_before_70s_l370_370920


namespace min_area_of_circle_tangent_to_line_l370_370733

open Real

noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

theorem min_area_of_circle_tangent_to_line (a b : ℝ) (h1 : a = 3) (h2 : b = 1) (h3 : c = -4) :
  ∃ (r : ℝ), Circle.circumference_radius C = AB / 2) ∧ Circle.tangent_to_line C (3, 1, -4) ∧ AB is the diameter of the circle C
  minimum_area : (∃ (r : ℝ), r = distance_point_to_line 0 0 3 1 (-4) / 2),
  S = π * r^2 
  S = (2/5) * π :=
begin
  -- proof goes here
  sorry,
end

end min_area_of_circle_tangent_to_line_l370_370733


namespace compare_abc_l370_370229

theorem compare_abc 
  (a : ℝ := 1 / 11) 
  (b : ℝ := Real.sqrt (1 / 10)) 
  (c : ℝ := Real.log (11 / 10)) : 
  b > c ∧ c > a := 
by
  sorry

end compare_abc_l370_370229


namespace weight_of_empty_jar_l370_370472

variable (W : ℝ) -- Weight of the empty jar
variable (w : ℝ) -- Weight of water for one-fifth of the jar

-- Conditions
variable (h1 : W + w = 560)
variable (h2 : W + 4 * w = 740)

-- Theorem statement
theorem weight_of_empty_jar (W w : ℝ) (h1 : W + w = 560) (h2 : W + 4 * w = 740) : W = 500 := 
by
  sorry

end weight_of_empty_jar_l370_370472


namespace total_sales_first_three_days_correct_highest_lowest_sales_difference_correct_actual_total_weekly_sales_correct_total_weekly_wage_correct_l370_370508
noncomputable theory

def sales_for_day_of_week :=
  [4, -3, 14, -7, -9, 21, -6]

def planned_sales_per_day := 100

def total_sales_first_three_days := 3 * planned_sales_per_day + sales_for_day_of_week.take 3 |>.sum

def highest_lowest_sales_difference := sales_for_day_of_week.maximum'.get_or_else 0 - sales_for_day_of_week.minimum'.get_or_else 0

def actual_total_weekly_sales := 7 * planned_sales_per_day + sales_for_day_of_week.sum

def total_weekly_wage :=
  (7 * planned_sales_per_day + sales_for_day_of_week.sum) * 40
    + (sales_for_day_of_week 0 + sales_for_day_of_week 2 + sales_for_day_of_week 5) * 15
    - (sales_for_day_of_week 1 + sales_for_day_of_week 3 + sales_for_day_of_week 4 + sales_for_day_of_week 6) * 20

theorem total_sales_first_three_days_correct:
  total_sales_first_three_days = 315 := sorry

theorem highest_lowest_sales_difference_correct:
  highest_lowest_sales_difference = 30 := sorry

theorem actual_total_weekly_sales_correct:
  actual_total_weekly_sales >= 7 * planned_sales_per_day := sorry

theorem total_weekly_wage_correct:
  total_weekly_wage = 28645 := sorry

end total_sales_first_three_days_correct_highest_lowest_sales_difference_correct_actual_total_weekly_sales_correct_total_weekly_wage_correct_l370_370508


namespace solve_inequality_system_l370_370027

theorem solve_inequality_system (x : ℝ) : 
  (2 + x > 7 - 4x ∧ x < (4 + x) / 2) → (1 < x ∧ x < 4) :=
by
  intro h
  cases h with h1 h2
  sorry

end solve_inequality_system_l370_370027


namespace arithmetic_sequence_sum_l370_370549

theorem arithmetic_sequence_sum :
  2 * (∑ k in Finset.range 25, (51 + 2 * k)) = 3750 := by
  sorry

end arithmetic_sequence_sum_l370_370549


namespace bisect_segment_l370_370072

theorem bisect_segment (A B C D M E F K L : Point)
  (h1 : diagonal_intersection A B C D M)
  (h2 : secant_through_point E M F)
  (h3 : secant_segment_intersects E F A D B C)
  (h4 : extensions_intersect A B C D K L)
  (h5 : bisected_by_point E F M) :
  bisected_by_point K L M := 
sorry

end bisect_segment_l370_370072


namespace range_of_k_l370_370745

theorem range_of_k (k : ℤ) (a : ℤ → ℤ) (h_a : ∀ n : ℕ, a n = |n - k| + |n + 2 * k|)
  (h_a3_equal_a4 : a 3 = a 4) : k ≤ -2 ∨ k ≥ 4 :=
sorry

end range_of_k_l370_370745


namespace total_pencils_in_drawer_l370_370060

-- Definitions based on conditions from the problem
def initial_pencils : ℕ := 138
def pencils_by_Nancy : ℕ := 256
def pencils_by_Steven : ℕ := 97

-- The theorem proving the total number of pencils in the drawer
theorem total_pencils_in_drawer : initial_pencils + pencils_by_Nancy + pencils_by_Steven = 491 :=
by
  -- This statement is equivalent to the mathematical problem given
  sorry

end total_pencils_in_drawer_l370_370060


namespace train_lateness_l370_370156

/-- A train moves at 6/7 of its usual speed. 
    The usual time for the train to complete the journey is 0.9999999999999997 hours. 
    To prove that the train is approximately 10 minutes late in this journey --/
theorem train_lateness :
  let usual_time_hrs := 0.9999999999999997
  let usual_time_mins := 60 * usual_time_hrs
  let actual_time_mins := usual_time_mins * (7 / 6) 
  actual_time_mins - usual_time_mins ≈ 10 := 
by
  sorry

end train_lateness_l370_370156


namespace team_X_games_l370_370460

variable (x : ℕ)

-- Definitions for team X
def win_rate_X := (3 : ℚ) / 4
def lose_rate_X := (1 : ℚ) / 4

-- Definitions for team Y
def win_rate_Y := (2 : ℚ) / 3
def lose_rate_Y := (1 : ℚ) / 3

-- Total games played by team Y
def total_games_Y : ℕ := x + 4

-- Winning games conditions for both teams
def win_games_X : ℚ := win_rate_X * x
def win_games_Y : ℚ := win_rate_Y * total_games_Y

-- Winning difference condition
axiom win_condition : win_games_Y = win_games_X + 9

def number_of_games_played_by_team_X : Prop :=
  x = 76

theorem team_X_games (h1 : win_condition) :
  number_of_games_played_by_team_X :=
sorry

end team_X_games_l370_370460


namespace pirate_treasure_l370_370448

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l370_370448


namespace count_odd_three_digit_decreasing_order_l370_370703

theorem count_odd_three_digit_decreasing_order : 
  let valid_digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} \ {0}
  let odd_digits := {1, 3, 5, 7, 9}
  let count_triples := λ (a b c : ℕ), 
    (a ∈ valid_digits ∧ b ∈ valid_digits ∧ c ∈ odd_digits ∧ a > b ∧ b > c)
  (finset.sum (finset.filter count_triples {abc | ∃ (a b c : ℕ), a ∈ valid_digits ∧ b ∈ valid_digits ∧ c ∈ odd_digits})) = 22 :=
by
  sorry

end count_odd_three_digit_decreasing_order_l370_370703


namespace angles_a1_b1_c1_form_ap_l370_370844

-- Definition for the problem context

def angles_form_ap (α : ℝ) (A B C : ℝ) : Prop :=
  B = α + π / 7 ∧ C = α - π / 7

def angle_bisectors_intersect_at (D : Point) (A B C : Point) : Prop :=
  -- Placeholder for the definition of angle bisectors intersecting at a point
  sorry 

def points_on_extensions (D A B C A1 B1 C1 : Point) : Prop :=
  -- Placeholder for the definition that points A1, B1, C1 are on the extensions at equal distance
  sorry

noncomputable def common_difference (α : ℝ) : ℝ :=
  π / 28

theorem angles_a1_b1_c1_form_ap (α : ℝ) (A B C D A1 B1 C1 : Point) 
  (H1 : angles_form_ap α A B C)
  (H2 : angle_bisectors_intersect_at D A B C)
  (H3 : points_on_extensions D A B C A1 B1 C1) : 
  ∃ d : ℝ, d = common_difference α ∧ 
  angles_form_ap (π / 2 - α / 2) (angle_at A1) (angle_at B1) (angle_at C1) :=
sorry

end angles_a1_b1_c1_form_ap_l370_370844


namespace find_s_l370_370328

theorem find_s (c d m r s : ℝ) (hc : c = (m + real.sqrt (m^2 - 12)) / 2) 
  (hd : d = (m - real.sqrt (m^2 - 12)) / 2) 
  (hcd : c * d = 3)
  (roots_in_second_eq : ∀ x, x^2 - r * x + s = 0 ↔ (c + 1/d) = x ∨ (d + 1/c) = x) :
  s = 16 / 3 :=
by
  sorry

end find_s_l370_370328


namespace original_price_of_book_original_price_of_pen_l370_370076

theorem original_price_of_book (sale_price_book : ℝ) (discount_rate_book : ℝ) (original_price_book : ℝ) :
  sale_price_book = 8 → discount_rate_book = (1 / 8) → original_price_book = (sale_price_book / discount_rate_book) → original_price_book = 64 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

theorem original_price_of_pen (sale_price_pen : ℝ) (discount_rate_pen : ℝ) (original_price_pen : ℝ) :
  sale_price_pen = 4 → discount_rate_pen = (1 / 5) → original_price_pen = (sale_price_pen / discount_rate_pen) → original_price_pen = 20 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end original_price_of_book_original_price_of_pen_l370_370076


namespace pure_imaginary_number_solution_l370_370332

noncomputable def z (a : ℝ) : ℂ := a * complex.I

theorem pure_imaginary_number_solution :
  ∀ (a : ℝ), (|z(a) - 1| = | -1 + complex.I |) → (z(a) = complex.I ∨ z(a) = -complex.I) :=
by
  assume a : ℝ
  have h₁ := congr_arg complex.abs (z(a) - 1) 
  have h₂ := |-1 + complex.I|
  simp at h₁ h₂
  sorry

end pure_imaginary_number_solution_l370_370332


namespace minimum_spent_on_boxes_l370_370892

theorem minimum_spent_on_boxes :
  let box_volume := 20 * 20 * 15,
      total_volume := 3060000,
      price_per_box := 0.5,
      number_of_boxes := Int.ceil (total_volume / box_volume)
  in number_of_boxes * price_per_box = 255 := sorry

end minimum_spent_on_boxes_l370_370892


namespace domain_P_l370_370691

def M : Set ℝ := { y | ∃ x, y = x⁻² }
def P : Set ℝ := { x | ∃ y, y = Real.sqrt (x-1) }

theorem domain_P : P = { x | 1 ≤ x } :=
by
  sorry

end domain_P_l370_370691


namespace real_roots_condition_l370_370223

theorem real_roots_condition (a : ℝ) (h : a ≠ -1) : 
    (∃ x : ℝ, x^2 + a * x + (a + 1)^2 = 0) ↔ a ∈ Set.Icc (-2 : ℝ) (-2 / 3) :=
sorry

end real_roots_condition_l370_370223


namespace polygon_interior_angle_l370_370981

theorem polygon_interior_angle (n : ℕ) (hn : 3 * (180 - 180 * (n - 2) / n) + 180 = 180 * (n - 2) / n + 180) : n = 9 :=
by {
  sorry
}

end polygon_interior_angle_l370_370981


namespace sin_210_l370_370565

theorem sin_210 : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end sin_210_l370_370565


namespace probability_two_red_or_blue_correct_l370_370127

noncomputable def probability_two_red_or_blue_sequential : ℚ := 1 / 5

theorem probability_two_red_or_blue_correct :
  let total_marbles := 15
  let red_blue_marbles := 7
  let first_draw_prob := (7 : ℚ) / 15
  let second_draw_prob := (6 : ℚ) / 14
  first_draw_prob * second_draw_prob = probability_two_red_or_blue_sequential :=
by
  sorry

end probability_two_red_or_blue_correct_l370_370127


namespace negation_example_l370_370389

theorem negation_example : ¬(∀ x : ℝ, x > 1 → x^2 > 1) ↔ ∃ x : ℝ, x > 1 ∧ x^2 ≤ 1 := by
  sorry

end negation_example_l370_370389


namespace min_value_f_in_interval_l370_370845

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 + sqrt 3 * sin x * cos x

theorem min_value_f_in_interval : 
  is_min_on f (Icc (real.pi / 4) (real.pi / 2)) 1 :=
sorry

end min_value_f_in_interval_l370_370845


namespace pirates_treasure_l370_370416

theorem pirates_treasure :
  ∃ m : ℕ,
    (m / 3 + 1) +
    (m / 4 + 5) +
    (m / 5 + 20) = m ∧
    m = 120 :=
by {
  sorry,
}

end pirates_treasure_l370_370416


namespace intersecting_line_circle_no_parallel_OD_PQ_l370_370305

-- Definitions according to given problem conditions
def circle (x y : ℝ) := x^2 + y^2 - 12 * x + 32 = 0
def line (k x y : ℝ) := y = k * x + 2
def P := (0, 2 : ℝ)
def Q := (6, 0 : ℝ)

-- Proof of the first part: the range of k
theorem intersecting_line_circle (k : ℝ) : 
  (∃ x y, circle x y ∧ line k x y) ↔ k ∈ Ioo (-3/4) 0 := sorry

-- Proof of the second part: impossibility of the required parallelogram
theorem no_parallel_OD_PQ (k : ℝ) : 
  ¬ (∃ k, k ∈ Ioo (-3/4) 0 ∧ ∃ D, 
    let ⟨x1, y1⟩ := D, ⟨x2, y2⟩ := D in
    (x1 + x2, y1 + y2) = (6, -2) ∧ 
    (0 - x1) / (x1 - 6) = (2 - y1) / (y1 + 2)) := sorry

end intersecting_line_circle_no_parallel_OD_PQ_l370_370305


namespace jump_rope_cost_l370_370190

-- Define conditions
def board_game_cost : ℕ := 12
def playground_ball_cost : ℕ := 4
def allowance : ℕ := 6
def uncle_gift : ℕ := 13
def additional_needed : ℕ := 4

-- Define total cost calculated
def total_cost (jump_rope_cost board_game_cost playground_ball_cost : ℕ) : ℕ :=
\add board_game_cost playground_ball_cost jump_rope_cost

-- Define current and needed money
def current_money (allowance uncle_gift : ℕ) : ℕ :=
\add allowance uncle_gift

-- Combine all conditions into a proof statement
theorem jump_rope_cost :
  total_cost jump_rope board_game_cost playground_ball_cost = current_money allowance uncle_gift + additional_needed →
  jump_rope = 7 :=
begin
  sorry
end

end jump_rope_cost_l370_370190


namespace at_least_33_rhombuses_l370_370145

-- Define the main properties and conditions of the problem
def plane (n : ℕ) : Prop := 
  ∃ (triangles : Finset (ℕ × ℕ)), 
    triangles.card = n ∧ 
    (∀ (t1 t2 : ℕ × ℕ), t1 ∈ triangles → t2 ∈ triangles → 
     (t1.1 = t2.1 ∧ (t1.2 = t2.2 + 1 ∨ t1.2 = t2.2 - 1)) ∨ 
     (t1.2 = t2.2 ∧ (t1.1 = t2.1 + 1 ∨ t1.1 = t2.1 - 1))) -- adjancency of triangles
    
def connected (triangles : Finset (ℕ × ℕ)) : Prop :=
  ∀ (t1 t2 : ℕ × ℕ), t1 ∈ triangles → t2 ∈ triangles →
  ∃ (p : List (ℕ × ℕ)), List.chain (λ t1 t2, 
    ((t1.1 = t2.1 ∧ (t1.2 = t2.2 + 1 ∨ t1.2 = t2.2 - 1)) ∨ 
     (t1.2 = t2.2 ∧ (t1.1 = t2.1 + 1 ∨ t1.1 = t2.1 - 1)))) t1 p ∧ List.last p t1 = some t2

-- Define the condition of 100 triangles forming a connected shape
def connected_shape_100 (triangles : Finset (ℕ × ℕ)) : Prop :=
  plane 100 triangles ∧ connected triangles

theorem at_least_33_rhombuses : 
  ∃ (set_of_rhombuses : Finset (Finset (ℕ × ℕ))),
    (∀ (r : Finset (ℕ × ℕ)), r ∈ set_of_rhombuses → r.card = 2 ∧ plane 2 r) ∧ 
    set_of_rhombuses.card ≥ 33 :=
by
  sorry

end at_least_33_rhombuses_l370_370145


namespace conference_registration_analysis_l370_370169

open Real

noncomputable def percent_of_initial_registration (total : ℝ) (registered_one_month_advance : ℝ) : ℝ :=
  100 * registered_one_month_advance / total

noncomputable def percent_of_two_weeks_registration (total : ℝ) (registered_one_month_advance : ℝ)
    (registered_one_month_to_two_weeks_advance : ℝ) : ℝ :=
  100 * (registered_one_month_advance + registered_one_month_to_two_weeks_advance) / total

noncomputable def percent_of_payment_methods (total : ℝ) (discounted_attendees : ℝ) 
    (credit_card : ℝ) (debit_card : ℝ) (other_methods : ℝ) : ℝ :=
  100 * ((credit_card * discounted_attendees / total) + 
  (debit_card * discounted_attendees / total) + 
  (other_methods * discounted_attendees / total))

theorem conference_registration_analysis : 
  ∀ (total attendees registered_one_month_advance registered_one_month_to_two_weeks_advance : ℝ)
    (credit_card debit_card other_methods : ℝ),

  registered_one_month_advance = 0.80 * total →

  registered_one_month_to_two_weeks_advance = 0.12 * total →

  credit_card = 0.20 →

  debit_card = 0.60 →

  other_methods = 0.20 →

  percent_of_initial_registration total registered_one_month_advance = 80 ∧

  percent_of_two_weeks_registration total registered_one_month_advance registered_one_month_to_two_weeks_advance = 92 ∧

  let discounted_attendees := (registered_one_month_advance + registered_one_month_to_two_weeks_advance) in

  percent_of_payment_methods total discounted_attendees credit_card debit_card other_methods = 92 :=
by
  intros total attendees registered_one_month_advance registered_one_month_to_two_weeks_advance
  credit_card debit_card other_methods
  intros h1 h2 h3 h4 h5
  simp [percent_of_initial_registration, percent_of_two_weeks_registration, percent_of_payment_methods]
  sorry

end conference_registration_analysis_l370_370169


namespace no_common_solution_general_case_l370_370753

-- Define the context: three linear equations in two variables
variables {a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℝ}

-- Statement of the theorem
theorem no_common_solution_general_case :
  (∃ (x y : ℝ), a1 * x + b1 * y = c1 ∧ a2 * x + b2 * y = c2 ∧ a3 * x + b3 * y = c3) →
  (a1 * b2 ≠ a2 * b1 ∧ a1 * b3 ≠ a3 * b1 ∧ a2 * b3 ≠ a3 * b2) →
  false := 
sorry

end no_common_solution_general_case_l370_370753


namespace greatest_possible_individual_award_l370_370106

theorem greatest_possible_individual_award
  (total_prize : ℝ := 400)
  (num_winners : ℝ := 20)
  (min_award : ℝ := 20)
  (fraction_prize : ℝ := 2 / 5)
  (fraction_winners : ℝ := 3 / 5)
  (remaining_prize := total_prize - fraction_prize * total_prize)
  (remaining_winners := num_winners - fraction_winners * num_winners) :
  greatest_possible_award : ℝ :=
  let required_award := min_award * (remaining_winners - 1) in
  greatest_possible_award = remaining_prize - required_award ∧ 
  greatest_possible_award = 100 :=
by
  sorry

end greatest_possible_individual_award_l370_370106


namespace rectangle_area_l370_370533

theorem rectangle_area (P : ℕ) (w : ℕ) (h : ℕ) (A : ℕ) 
  (hP : P = 28) 
  (hw : w = 6)
  (hW : P = 2 * (h + w)) 
  (hA : A = h * w) : 
  A = 48 :=
by
  sorry

end rectangle_area_l370_370533


namespace probability_waiting_time_tan_double_angle_min_value_of_log_graph_function_properties_l370_370493

-- Probability Problem
theorem probability_waiting_time (total_time : ℕ) (waiting_time : ℕ) (h1 : total_time = 60) (h2 : waiting_time = 12) : 
  (waiting_time : ℝ) / total_time = 1 / 5 := 
by sorry

-- Trigonometric Problem
theorem tan_double_angle (α : ℝ) (h1 : π < α ∧ α < 3 * π / 2) (h2 : cos (π + α) = 4 / 5) :
  tan (2 * α) = 24 / 7 := 
by sorry

-- Logarithmic Graph Problem
theorem min_value_of_log_graph (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (m n : ℝ) (h3 : m > 0) (h4 : n > 0) 
  (h5 : m * -4 + n * -2 + 8 = 0) : 
  2 / m + 1 / n = 9 / 4 := 
by sorry

-- Function Properties Problem
theorem function_properties (f : ℝ → ℝ) (h1 : ∀ x, f(x) + f(x-1) = 1) (h2 : ∀ x ∈ [0,1], f x = x^2) 
  (h3 : (f (-2005.5) = 3 / 4) = false) (h4 : (∀ x ∈ [1,2], f x = 2 * x - x^2)) : 
  (∀ x, f(x+2) = f x) ∧ (∀ x ∈ [1,2], f x = 2 * x - x^2) ∧ ¬(∀ x, f(-x) = f x) := 
by sorry

end probability_waiting_time_tan_double_angle_min_value_of_log_graph_function_properties_l370_370493


namespace hyperbola_triangle_area_l370_370260

theorem hyperbola_triangle_area :
  ∀ (A : ℝ × ℝ),
    (3 * A.fst + 4 * A.snd = 0 ∨ 3 * A.fst - 4 * A.snd = 0) →
    (sqrt ((A.fst + 5) ^ 2 + A.snd ^ 2) - sqrt ((A.fst - 5) ^ 2 + A.snd ^ 2) = 8) →
    (dist (A, (-5, 0)) + dist (A, (5, 0)) = 10) →
    real.angle.rad ((-5, 0), A, (5, 0)) = real.pi / 3 →
    1/2 * sqrt ((A.fst + 5) ^ 2 + A.snd ^ 2) * sqrt ((A.fst - 5) ^ 2 + A.snd ^ 2) * sqrt 3 / 2 = 9 * sqrt 3 :=
begin
    intros A Hasymptotes Hdist1 Hdist2 Hangle,
    sorry
end

end hyperbola_triangle_area_l370_370260


namespace louise_winning_strategy_2023x2023_l370_370906

theorem louise_winning_strategy_2023x2023 :
  ∀ (n : ℕ), (n % 2 = 1) → (n = 2023) →
  ∃ (strategy : ℕ × ℕ → Prop),
    (∀ turn : ℕ, ∃ (i j : ℕ), i < n ∧ j < n ∧ strategy (i, j)) ∧
    (∃ i j : ℕ, strategy (i, j) ∧ (i = 0 ∧ j = 0)) :=
by
  sorry

end louise_winning_strategy_2023x2023_l370_370906


namespace general_term_formula_l370_370241

theorem general_term_formula (S : ℕ → ℤ) (a : ℕ → ℤ) : 
  (∀ n, S n = 3 * n ^ 2 - 2 * n) → 
  (∀ n ≥ 2, a n = S n - S (n - 1)) ∧ a 1 = S 1 → 
  ∀ n, a n = 6 * n - 5 := 
by
  sorry

end general_term_formula_l370_370241


namespace correct_probability_l370_370376

/-
Define the faces of the two dice.
-/
def die1_faces : set ℕ := {1, 2, 3, 3, 4, 5}
def die2_faces : set ℕ := {2, 3, 4, 6, 7, 9}

/-
Define the condition for a sum to be 6, 8, or 10.
-/
def is_valid_sum (sum : ℕ) : Prop := sum = 6 ∨ sum = 8 ∨ sum = 10

/-
Define the probability of obtaining a valid sum from the dice.
-/
noncomputable def probability_of_valid_sum : ℚ :=
  (die1_faces.product die2_faces).count (λ ⟨x, y⟩, is_valid_sum (x + y)) / 
  (die1_faces.card * die2_faces.card : ℚ)

/-
Statement asserting the computed probability is equal to the correct answer.
-/
theorem correct_probability : probability_of_valid_sum = 11 / 36 :=
sorry

end correct_probability_l370_370376


namespace triangle_right_angled_l370_370006

theorem triangle_right_angled
  (a b c R r r_1 : ℝ)
  (abc_eq : a * b * c = 4 * R * r * r_1)
  (is_triangle : ∃ (A B C : Type) [ordered_ring A] (a b c : A), a + b > c ∧ a + c > b ∧ b + c > a) :
  ∃ (abc_right : Type) [ordered_ring abc_right] (A B C : abc_right) (right_angle : Prop), right_angle = 90 :=
sorry

end triangle_right_angled_l370_370006


namespace median_of_remaining_rooms_l370_370170

def room_assignment := List.range' 1 30

def unoccupied_rooms := [16, 17, 18]

def remaining_rooms (rooms : List ℕ) (unoccupied : List ℕ) : List ℕ :=
  rooms.filter (λ r => ¬(r ∈ unoccupied))

def median_room (rooms : List ℕ) : ℕ :=
  if rooms.length % 2 = 0 then (rooms.nth! (rooms.length / 2) + rooms.nth! (rooms.length / 2 - 1)) / 2
  else rooms.nth! (rooms.length / 2)

theorem median_of_remaining_rooms : 
  median_room (remaining_rooms room_assignment unoccupied_rooms) = 14 :=
by
  sorry

end median_of_remaining_rooms_l370_370170


namespace a_seq_general_formula_T_sum_first_n_terms_l370_370327

open Nat

-- Definition of the sequence a_n
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => 2 * a n

-- Sum of the first n terms of the sequence a_n
def S (n : ℕ) : ℕ :=
  (range n).sum (λ i => a i)

-- Verification of the given condition
theorem a_seq_general_formula (n : ℕ) : (S (n + 1) + 1) * a n = (S n + 1) * a (n + 1) :=
  sorry

-- Definition of c_k
def c_k (k : ℕ) : ℕ :=
  ((k + 2) * (a k + a (k + 1)) / 2) - (a k + a (k + 1))

-- Sum of the first n terms of the sequence c_n
def T (n : ℕ) : ℕ :=
  (range n).sum (λ k => c_k k)

-- The theorem to find the sum of the first n terms of c_n
theorem T_sum_first_n_terms (n : ℕ) : T (n) = (3 / 2 * (1 + (n - 1) * 2 ^ n)) :=
  sorry

end a_seq_general_formula_T_sum_first_n_terms_l370_370327


namespace sum_of_values_of_x_satisfying_g_eq_1_l370_370790

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ 0 then 2 * x - 5 else x / 3 + 2

theorem sum_of_values_of_x_satisfying_g_eq_1 : ∑ x in {x : ℝ | g x = 1}, x = 0 := by
  sorry

end sum_of_values_of_x_satisfying_g_eq_1_l370_370790


namespace complex_division_l370_370219

theorem complex_division (i : ℂ) (h : i * i = -1) : 3 / (1 - i) ^ 2 = (3 / 2) * i :=
by
  sorry

end complex_division_l370_370219


namespace permutations_satisfy_inequality_l370_370622

theorem permutations_satisfy_inequality :
  ∀ (a : Fin 7 → ℕ) (h_perm : Multiset.ofFn a = {1, 2, 3, 4, 5, 6, 7}),
  (∏ i, (a i + (i : ℕ) + 1) / 2) ^ 2 > 2 * (nat.factorial 7) :=
by sorry

end permutations_satisfy_inequality_l370_370622


namespace find_x_l370_370277

def vector := (ℝ × ℝ × ℝ)

def dot_product (u v : vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def add_vectors (u v : vector) : vector :=
  (u.1 + v.1, u.2 + v.2, u.3 + v.3)

def a : vector := (2, 1, 2)
def b (x : ℝ) : vector := (-2, x, 1)
def c : vector := (4, 3, 2)
def d : vector := add_vectors a c

theorem find_x (x : ℝ) (h : dot_product (b x) d = 0) : x = 2 :=
  by
  sorry

end find_x_l370_370277


namespace simplify_cube_root_of_5488000_l370_370018

theorem simplify_cube_root_of_5488000 :
  ∛(5488000) = 140 * 2^(1/3 : ℝ) :=
by
  have h1: 5488000 = 10^3 * 5488, by norm_num
  have h2: 5488 = 2^4 * 7^3, by norm_num
  sorry

end simplify_cube_root_of_5488000_l370_370018


namespace solve_inequality_system_l370_370029

theorem solve_inequality_system (x : ℝ) : 
  (2 + x > 7 - 4x ∧ x < (4 + x) / 2) → (1 < x ∧ x < 4) :=
by
  intro h
  cases h with h1 h2
  sorry

end solve_inequality_system_l370_370029


namespace complex_zero_real_part_l370_370829

theorem complex_zero_real_part (a b : ℝ) (z : ℂ)
  (h_eq : z * (z + Complex.i) * (z + 3 * Complex.i) = 2002 * Complex.i):
  z = a + b * Complex.i →
  a > 0 →
  b > 0 →
  a = Real.sqrt 118 :=
by
  sorry

end complex_zero_real_part_l370_370829


namespace flowchart_output_correct_l370_370848

-- Define the conditions of the problem
def program_flowchart (initial : ℕ) : ℕ :=
  let step1 := initial * 2
  let step2 := step1 * 2
  let step3 := step2 * 2
  step3

-- State the proof problem
theorem flowchart_output_correct : program_flowchart 1 = 8 :=
by
  -- Sorry to skip the proof
  sorry

end flowchart_output_correct_l370_370848


namespace corn_plants_per_row_l370_370551

variable (g pm pd pdp m r pc pdu wpd wpdw wr)

-- Conditions
def gallons_per_minute := 3
def minutes_pumping := 25
def pigs := 10
def ducks := 20
def water_per_pig := 4
def water_per_duck := 0.25
def rows_of_corn := 4
def water_per_corn_plant := 0.5

-- Correct Answer: 15
theorem corn_plants_per_row (g : ℕ) (pm : ℕ) (pd : ℕ) (pdp : ℕ) (m : ℕ) (r : ℕ) (pc : ℕ) (pdu : ℕ) (wpd : ℕ) (wpdw : ℕ) :
  g = gallons_per_minute →
  pm = minutes_pumping →
  pd = pigs →
  pdp = water_per_pig →
  pdu = ducks →
  wpd = water_per_duck →
  m = (g * pm - (pd * pdp + pdu * wpd)) / water_per_corn_plant →
  r = rows_of_corn →
  pc = m / r →
  pc = 15 :=
by {
  sorry
}

end corn_plants_per_row_l370_370551


namespace first_woman_hours_l370_370949

-- Definitions and conditions
variables (W k y t η : ℝ)
variables (work_rate : k * y * 45 = W)
variables (total_work : W = k * (t * ((y-1) * y) / 2 + y * η))
variables (first_vs_last : (y-1) * t + η = 5 * η)

-- The goal to prove
theorem first_woman_hours :
  (y - 1) * t + η = 75 := 
by
  sorry

end first_woman_hours_l370_370949


namespace coefficient_of_x_in_binomial_expansion_l370_370207

theorem coefficient_of_x_in_binomial_expansion :
  let T_r := (nat.choose 5 r) * x ^ (10 - 3 * r)
  (T_r / x) = 10 :=
by
  sorry

end coefficient_of_x_in_binomial_expansion_l370_370207


namespace isosceles_triangle_base_length_l370_370986

theorem isosceles_triangle_base_length :
  ∃ (x y : ℝ), 
    ((x + x / 2 = 15 ∧ y + x / 2 = 6) ∨ (x + x / 2 = 6 ∧ y + x / 2 = 15)) ∧ y = 1 :=
by
  sorry

end isosceles_triangle_base_length_l370_370986


namespace number_of_customers_l370_370506

theorem number_of_customers 
  (offices sandwiches_per_office total_sandwiches group_sandwiches_per_customer half_group_sandwiches : ℕ)
  (h1 : offices = 3)
  (h2 : sandwiches_per_office = 10)
  (h3 : total_sandwiches = 54)
  (h4 : group_sandwiches_per_customer = 4)
  (h5 : half_group_sandwiches = 54 - (3 * 10))
  : half_group_sandwiches = 24 → 2 * 12 = 24 :=
by
  sorry

end number_of_customers_l370_370506


namespace burger_meal_cost_l370_370175

theorem burger_meal_cost 
  (x : ℝ) 
  (h : 5 * (x + 1) = 35) : 
  x = 6 := 
sorry

end burger_meal_cost_l370_370175


namespace age_difference_l370_370138

theorem age_difference
  (A B : ℕ)
  (hB : B = 48)
  (h_condition : A + 10 = 2 * (B - 10)) :
  A - B = 18 :=
by
  sorry

end age_difference_l370_370138


namespace min_value_of_quadratic_l370_370891

theorem min_value_of_quadratic :
  ∃ x : ℝ, (x^2 + 6*x - 8) ≤ y ∀ y ∈ (λ x, x^2 + 6*x - 8) :=
begin
  use -3,
  apply min_at,
  sorry
end

end min_value_of_quadratic_l370_370891


namespace total_hours_played_is_1_5_l370_370763

-- Define the time Jose played football in minutes
def football_minutes : ℕ := 30

-- Define the time Jose played basketball in minutes
def basketball_minutes : ℕ := 60

-- Define the total time played in minutes as the sum of football and basketball minutes
def total_minutes (f b : ℕ) := f + b

-- Define the conversion from minutes to hours
def minutes_to_hours (m : ℕ) : ℝ := m / 60.0

-- Prove that the total hours played is 1.5 hours
theorem total_hours_played_is_1_5 :
  minutes_to_hours (total_minutes football_minutes basketball_minutes) = 1.5 :=
  sorry

end total_hours_played_is_1_5_l370_370763


namespace remaining_budget_is_correct_l370_370950

def budget := 750
def flasks_cost := 200
def test_tubes_cost := (2 / 3) * flasks_cost
def safety_gear_cost := (1 / 2) * test_tubes_cost
def chemicals_cost := (3 / 4) * flasks_cost
def instruments_min_cost := 50

def total_spent := flasks_cost + test_tubes_cost + safety_gear_cost + chemicals_cost
def remaining_budget_before_instruments := budget - total_spent
def remaining_budget_after_instruments := remaining_budget_before_instruments - instruments_min_cost

theorem remaining_budget_is_correct :
  remaining_budget_after_instruments = 150 := by
  unfold remaining_budget_after_instruments remaining_budget_before_instruments total_spent flasks_cost test_tubes_cost safety_gear_cost chemicals_cost budget
  sorry

end remaining_budget_is_correct_l370_370950


namespace triangle_constructibility_l370_370596

-- Given definitions
variables (a R α ϱ : ℝ)
variables {A B C O K : ℝ}

-- Constructing a triangle with side 'a', opposite angle α, and inradius ϱ
def constructible_triangle (R : ℝ) (alpha : ℝ) (varrho : ℝ) : Prop :=
  varrho ≤ 2 * R * real.sin (alpha / 2) * (1 - real.sin (alpha / 2))

-- Constructibility condition
theorem triangle_constructibility (a R α ϱ : ℝ) : 
  constructible_triangle R α ϱ :=
sorry

end triangle_constructibility_l370_370596


namespace collatz_7_steps_16_l370_370552

def collatz (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def hailstone_sequence (n : ℕ) : ℕ :=
  if n = 1 then 0 else 1 + hailstone_sequence (collatz n)

theorem collatz_7_steps_16 : hailstone_sequence 7 = 16 := 
sorry

end collatz_7_steps_16_l370_370552


namespace min_fraction_sum_l370_370644

theorem min_fraction_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) : 
  (∀ x, x = 1 / m + 2 / n → x ≥ 8) :=
  sorry

end min_fraction_sum_l370_370644


namespace treasure_coins_l370_370440

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l370_370440


namespace smallest_four_digit_divisible_by_8_with_3_even_1_odd_l370_370890

theorem smallest_four_digit_divisible_by_8_with_3_even_1_odd : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 0 ∧ 
  (∃ d1 d2 d3 d4, n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
    (d1 % 2 = 0) ∧ (d2 % 2 = 0 ∨ d2 % 2 ≠ 0) ∧ 
    (d3 % 2 = 0) ∧ (d4 % 2 = 0 ∨ d4 % 2 ≠ 0) ∧ 
    (d2 % 2 ≠ 0 ∨ d4 % 2 ≠ 0) ) ∧ n = 1248 :=
by
  sorry

end smallest_four_digit_divisible_by_8_with_3_even_1_odd_l370_370890


namespace platform_length_l370_370925

theorem platform_length 
  (train_length : ℝ) 
  (time_cross_signal_pole : ℝ) 
  (time_cross_platform : ℝ) 
  (speed_train : ℝ := train_length / time_cross_signal_pole) 
  (distance_cross_platform : ℝ := speed_train * time_cross_platform)
  (platform_length : ℝ := distance_cross_platform - train_length) :
  train_length = 300 ∧ time_cross_signal_pole = 16 ∧ time_cross_platform = 39 → 
  platform_length = 431.25 :=
by {
  intros h,
  sorry  -- proof not required
}

end platform_length_l370_370925


namespace general_formula_find_k_l370_370119

noncomputable def arithmetic_sequence (a₁ a₃ : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * ((a₃ - a₁) / 2)

theorem general_formula {a₁ a₃ : ℤ} (h₁ : a₁ = 1) (h₃ : a₃ = -3) (n : ℕ) :
  arithmetic_sequence a₁ a₃ n = 3 - 2 * n :=
by
  sorry

theorem find_k {a₁ a₃ : ℤ} {Sₖ : ℤ} (h₁ : a₁ = 1) (h₃ : a₃ = -3) (hₛ : Sₖ = -35) :
  let d := (a₃ - a₁) / 2
      Sn (k : ℕ) := k * (2 * a₁ + (k - 1) * d) / 2
  in Sn 7 = Sₖ →
     7 = 7 :=
by
  sorry

end general_formula_find_k_l370_370119


namespace treasures_coins_count_l370_370405

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l370_370405


namespace shoe_size_15_is_9point25_l370_370954

noncomputable def smallest_shoe_length (L : ℝ) := L
noncomputable def largest_shoe_length (L : ℝ) := L + 9 * (1/4 : ℝ)
noncomputable def length_ratio_condition (L : ℝ) := largest_shoe_length L = 1.30 * smallest_shoe_length L
noncomputable def shoe_length_size_15 (L : ℝ) := L + 7 * (1/4 : ℝ)

theorem shoe_size_15_is_9point25 : ∃ L : ℝ, length_ratio_condition L → shoe_length_size_15 L = 9.25 :=
by
  sorry

end shoe_size_15_is_9point25_l370_370954


namespace correct_statements_l370_370256

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
noncomputable def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem correct_statements (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_even_shift : is_even_function (λ x, f (x + (π / 2)))) :
  (∀ x, f (x - π) = f x) ∧ (∃ x, f (-π, 0) = (x, f x)) :=
by
  sorry

end correct_statements_l370_370256


namespace sequence_initial_term_l370_370240

theorem sequence_initial_term (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n + 1) = a n + n)
  (h2 : a 61 = 2010) : a 1 = 180 :=
by
  sorry

end sequence_initial_term_l370_370240


namespace geometric_sequence_common_ratio_l370_370718

-- Define the geometric sequence and conditions
variable {a : ℕ → ℝ}

def is_geometric_sequence (q : ℝ) : Prop :=
  ∀ n, a (n+1) = a n * q

def all_terms_positive : Prop :=
  ∀ n, a n > 0

def forms_arithmetic_sequence (a1 a2 a3 : ℝ) : Prop :=
  a1 + a3 = 2 * a2

noncomputable def common_ratio (q : ℝ) : Prop :=
  ∀ (a : ℕ → ℝ) (h_geom : is_geometric_sequence q) (h_pos : all_terms_positive), forms_arithmetic_sequence (3 * a 0) (2 * a 1) (1/2 * a 2) → q = 3

-- Statement of the theorem to prove
theorem geometric_sequence_common_ratio (q : ℝ) : common_ratio q := by
  sorry

end geometric_sequence_common_ratio_l370_370718


namespace ab_squared_value_l370_370835

theorem ab_squared_value (a b : ℝ) 
  (h1 : ∀ x : ℝ, (a * cos x + b * sin x) * cos x ≤ 2) 
  (h2 : ∀ x : ℝ, (a * cos x + b * sin x) * cos x ≥ -1) : 
  (a * b) ^ 2 = 8 :=
sorry

end ab_squared_value_l370_370835


namespace cost_of_gas_used_l370_370808

def initial_odometer : ℕ := 85340
def final_odometer : ℕ := 85368
def fuel_efficiency : ℚ := 32 -- miles per gallon
def price_per_gallon : ℚ := 3.95

theorem cost_of_gas_used :
  let distance_traveled := final_odometer - initial_odometer in
  let gas_used := (distance_traveled : ℚ) / fuel_efficiency in
  let cost := gas_used * price_per_gallon in
  cost = 3.46 :=
by
  sorry

end cost_of_gas_used_l370_370808


namespace pirates_treasure_l370_370412

theorem pirates_treasure :
  ∃ m : ℕ,
    (m / 3 + 1) +
    (m / 4 + 5) +
    (m / 5 + 20) = m ∧
    m = 120 :=
by {
  sorry,
}

end pirates_treasure_l370_370412


namespace harvest_apples_l370_370754

def sacks_per_section : ℕ := 45
def sections : ℕ := 8
def total_sacks_per_day : ℕ := 360

theorem harvest_apples : sacks_per_section * sections = total_sacks_per_day := by
  sorry

end harvest_apples_l370_370754


namespace sin_210_l370_370563

theorem sin_210 : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end sin_210_l370_370563


namespace deriv_f_at_1_eq_neg6_l370_370707

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 2 * (deriv f 2) * x + 3

theorem deriv_f_at_1_eq_neg6 (h : differentiable ℝ f) (h_deriv_at_2 : deriv f 2 = -4) :
  deriv f 1 = -6 :=
by
  sorry

end deriv_f_at_1_eq_neg6_l370_370707


namespace sequence_count_correct_l370_370651

noncomputable def count_sequences : ℕ :=
  let possible_diffs := [-1, 1/3, 1]
  let total_diff := 1 -- b_8 - b_1
  let terms := 8

  -- Define the conditions
  let is_valid (seq : Fin terms → ℚ) :=
    (seq 0 = 2014) ∧
    (seq (7) = 2015) ∧
    (∀ n : Fin 7, seq (n + 1).val = seq n.val + (possible_diffs))

  -- Counting valid sequences
  let count := (finset.univ.filter is_valid).card

  count

theorem sequence_count_correct :
  count_sequences = 252 :=
  sorry

end sequence_count_correct_l370_370651


namespace smallest_integer_sum_of_squares_and_cubes_infinite_integers_sum_of_squares_and_cubes_l370_370215

theorem smallest_integer_sum_of_squares_and_cubes :
  ∃ (n : ℕ) (a b c d : ℕ), n > 2 ∧ n = a^2 + b^2 ∧ n = c^3 + d^3 ∧
  ∀ (m : ℕ) (x y u v : ℕ), (m > 2 ∧ m = x^2 + y^2 ∧ m = u^3 + v^3) → n ≤ m := 
sorry

theorem infinite_integers_sum_of_squares_and_cubes :
  ∀ (k : ℕ), ∃ (n : ℕ) (a b c d : ℕ), n = 1 + 2^(6*k) ∧ n = a^2 + b^2 ∧ n = c^3 + d^3 :=
sorry

end smallest_integer_sum_of_squares_and_cubes_infinite_integers_sum_of_squares_and_cubes_l370_370215


namespace hypotenuse_square_l370_370725

theorem hypotenuse_square (a : ℕ) : (a + 1)^2 + a^2 = 2 * a^2 + 2 * a + 1 := 
by sorry

end hypotenuse_square_l370_370725


namespace shaded_area_percentage_l370_370470

theorem shaded_area_percentage (total_area shaded_area : ℕ) (h_total : total_area = 49) (h_shaded : shaded_area = 33) : 
  (shaded_area : ℚ) / total_area = 33 / 49 := 
by
  sorry

end shaded_area_percentage_l370_370470


namespace integer_multiples_l370_370700

theorem integer_multiples (a b m : ℕ) (h1 : 200 = a) (h2 : 400 = b) (h3 : 117 = m) :
  ∃ (n : ℕ), n = 2 ∧ (∃ x, 200 ≤ x ∧ x ≤ 400 ∧ x % m = 0) :=
by {
  have h4 : a = 200 := h1,
  have h5 : b = 400 := h2,
  have h6 : m = 117 := h3,
  sorry
}

end integer_multiples_l370_370700


namespace ratio_EG_ES_l370_370302

-- conditions
variable {E F G H Q R S : Type}
variable [AddCommGroup E] [Module ℝ E]
variable [AffinoidBasis E F G H Q R S : AddCommGroup E]

def parallelogram (EFGH : Type) [AddCommGroup EFGH] [Module ℝ EFGH] := sorry

def point_on_EF (Q : E) (EF : ℝ) (EQ : ℝ) :=
    EQ / EF = 1 / 8

def point_on_EH (R : E) (EH : ℝ) (ER : ℝ) :=
    ER / EH = 1 / 9

def intersection (EG QR : E) :=
    ∃ S : E, S ∈ EG ∧ S ∈ QR

-- problem statement
theorem ratio_EG_ES (P : parallelogram E) (Q R S : E)
  (hQ : point_on_EF Q 8 1) (hR : point_on_EH R 9 1) (hS : intersection E QR) :
  EG / ES = 2 :=
sorry

end ratio_EG_ES_l370_370302


namespace medians_divide_segment_into_three_equal_parts_l370_370114

theorem medians_divide_segment_into_three_equal_parts
    {A B C P E F : Point}
    (K : Point)
    (L : Point)
    (AK : Line) (CL : Line) (PE : Line) (PF : Line)
    (AC : Line) (BC : Line) (AB : Line)
    (EF : Segment) (M N : Point) (O : Point)
    (hP_on_AC : P ∈ AC)
    (hPE_parallel_AK : parallel PE AK)
    (hPF_parallel_CL : parallel PF CL)
    (hE_on_BC : E ∈ BC)
    (hF_on_AB : F ∈ AB)
    (hM_intersect_AK : M ∈ AK ∧ M ∈ EF)
    (hN_intersect_CL : N ∈ CL ∧ N ∈ EF)
    (hO_is_centroid : centroid O A B C)
    : divides_ef_in_three_equal_parts AK CL EF :=
sorry

end medians_divide_segment_into_three_equal_parts_l370_370114


namespace pirates_treasure_l370_370427

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l370_370427


namespace program_output_is_20_l370_370465

-- Definitions based on the conditions
def PRINT (n : ℕ) : ℕ := n

theorem program_output_is_20
  (h1 : PRINT(3 + 2) * 4 = 20) : (3 + 2) * 4 = 20 :=
by
  sorry

end program_output_is_20_l370_370465


namespace max_A_k_value_l370_370779

noncomputable def A_k (k : ℕ) : ℝ := (19^k + 66^k) / k.factorial

theorem max_A_k_value : 
  ∃ k : ℕ, (∀ m : ℕ, (A_k m ≤ A_k k)) ∧ k = 65 :=
by
  sorry

end max_A_k_value_l370_370779


namespace conjugate_axis_length_l370_370656

variable (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
variable (e : ℝ) (h3 : e = Real.sqrt 7 / 2)
variable (c : ℝ) (h4 : c = a * e)
variable (P : ℝ × ℝ) (h5 : P = (c, b^2 / a))
variable (F1 F2 : ℝ × ℝ) (h6 : F1 = (-c, 0)) (h7 : F2 = (c, 0))
variable (h8 : dist P F2 = 9 / 2)
variable (h9 : P.1 = c) (h10 : P.2 = b^2 / a)
variable (h11 : PF_2 ⊥ F_1F_2)

theorem conjugate_axis_length : 2 * b = 6 * Real.sqrt 3 := by
  sorry

end conjugate_axis_length_l370_370656


namespace clock_angle_degrees_l370_370633

theorem clock_angle_degrees :
  let hour_rotation_per_min := 360 / (12 * 60) in
  let minute_rotation_per_min := 360 / 60 in
  let hour_rotation := 20 * hour_rotation_per_min in
  let minute_rotation := 20 * minute_rotation_per_min in
  let angle_at_8_20 := abs (minute_rotation - (8 * 30 + hour_rotation)) in
  hour_rotation = 10 ∧ minute_rotation = 120 ∧ angle_at_8_20 = 130 := 
by 
  sorry

end clock_angle_degrees_l370_370633


namespace pirate_treasure_l370_370451

/-- Given: 
  - The first pirate received (m / 3) + 1 coins.
  - The second pirate received (m / 4) + 5 coins.
  - The third pirate received (m / 5) + 20 coins.
  - All coins were distributed, i.e., (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m.
  Prove: m = 120
-/
theorem pirate_treasure (m : ℕ) 
  (h₁ : m / 3 + 1 = first_pirate_share)
  (h₂ : m / 4 + 5 = second_pirate_share)
  (h₃ : m / 5 + 20 = third_pirate_share)
  (h₄ : first_pirate_share + second_pirate_share + third_pirate_share = m)
  : m = 120 :=
sorry

end pirate_treasure_l370_370451


namespace inequality_system_solution_l370_370032

theorem inequality_system_solution (x : ℝ) (h1 : 2 + x > 7 - 4x) (h2 : x < (4 + x) / 2) : 1 < x ∧ x < 4 :=
by
  sorry -- Proof goes here

end inequality_system_solution_l370_370032


namespace rod_division_l370_370963

noncomputable def gcd (a b : ℕ) : ℕ := sorry
noncomputable def lcm (a b : ℕ) : ℕ := sorry

theorem rod_division (L m n x : ℕ) (hmn: m > n)
  (h1 : gcd m n = x + 1) 
  (h2 : 1 / m - 1 / n = 1 / 13) 
  (h3 : m * n - 13 * n + 13 * m = 169)
  (h4 : 170 segments (using m and n) with exactly 100 longest segments')
  : (m, n) = (26, 14) := sorry

end rod_division_l370_370963


namespace total_surveyed_l370_370299

theorem total_surveyed (m_dangerous_percent mistaken_percent total_mistaken : ℝ) 
                      (total_believe_danger : ℕ) (total_people : ℕ) 
                      (h1 : m_dangerous_percent = 0.825) 
                      (h2 : mistaken_percent = 0.524) 
                      (h3 : total_mistaken = 27) 
                      (h4 : total_believe_danger = Int.round (total_mistaken / mistaken_percent)) 
                      (h5 : total_people = Int.round (total_believe_danger / m_dangerous_percent)): 
                      total_people = 63 :=
by
  sorry

end total_surveyed_l370_370299


namespace compare_f_1_2_l370_370778

variable (f : ℝ → ℝ)

theorem compare_f_1_2
  (h_diff : Differentiable ℝ f)
  (h_pos : ∀ x : ℝ, 0 < f x)
  (h_ineq : ∀ x : ℝ, f x < x * (f' x)) :
  2 * (f 1) < (f 2) := by
  sorry

end compare_f_1_2_l370_370778


namespace no_students_unable_to_partner_l370_370402

def students_males_females :=
  let males_6th_class1 : Nat := 17
  let females_6th_class1 : Nat := 13
  let males_6th_class2 : Nat := 14
  let females_6th_class2 : Nat := 18
  let males_6th_class3 : Nat := 15
  let females_6th_class3 : Nat := 17
  let males_7th_class : Nat := 22
  let females_7th_class : Nat := 20

  let total_males := males_6th_class1 + males_6th_class2 + males_6th_class3 + males_7th_class
  let total_females := females_6th_class1 + females_6th_class2 + females_6th_class3 + females_7th_class

  total_males == total_females

theorem no_students_unable_to_partner : students_males_females = true := by
  -- Skipping the proof
  sorry

end no_students_unable_to_partner_l370_370402


namespace stratified_sampling_num_of_female_employees_l370_370939

theorem stratified_sampling_num_of_female_employees :
  ∃ (total_employees male_employees sample_size female_employees_to_draw : ℕ),
    total_employees = 750 ∧
    male_employees = 300 ∧
    sample_size = 45 ∧
    female_employees_to_draw = (total_employees - male_employees) * sample_size / total_employees ∧
    female_employees_to_draw = 27 :=
by
  sorry

end stratified_sampling_num_of_female_employees_l370_370939


namespace zero_point_interval_l370_370044

def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem zero_point_interval :
  (0 < 2) → (2 < Real.e) → f 2 < 0 → f Real.e > 0 →
  ∃ c ∈ (2, Real.e), f c = 0 :=
by
  intros h1 h2 h3 h4
  have hf : ∀ x y, x < y → f x < f y := sorry
  obtain ⟨c, hc⟩ : ∃ c, 2 < c ∧ c < Real.e ∧ f c = 0 := sorry
  exact ⟨c, hc⟩

end zero_point_interval_l370_370044


namespace find_K_l370_370319

def surface_area_cube (side_length : ℝ) : ℝ :=
  6 * side_length^2

def surface_area_sphere (radius : ℝ) : ℝ :=
  4 * Real.pi * radius^2

def volume_sphere (radius : ℝ) : ℝ :=
  (4 / 3) * Real.pi * radius^3

theorem find_K :
  let side_length := 3
  let surface_area := surface_area_cube side_length
  let radius := Real.sqrt (27 / (2 * Real.pi))
  let volume := volume_sphere radius
  volume = (K : ℝ) * Real.sqrt 6 / (Real.sqrt Real.pi)
  → K = 27 * Real.sqrt 6 / Real.sqrt 2 := 
by
  intros side_length surface_area radius volume h
  sorry

end find_K_l370_370319


namespace joe_initial_money_l370_370321

theorem joe_initial_money (cost_notebook cost_book money_left : ℕ) 
                          (num_notebooks num_books : ℕ)
                          (h1 : cost_notebook = 4) 
                          (h2 : cost_book = 7)
                          (h3 : num_notebooks = 7) 
                          (h4 : num_books = 2) 
                          (h5 : money_left = 14) :
  (num_notebooks * cost_notebook + num_books * cost_book + money_left) = 56 := by
  sorry

end joe_initial_money_l370_370321


namespace anna_coaching_days_l370_370167

/-- The total number of days from January 1 to September 4 in a non-leap year -/
def total_days_in_non_leap_year_up_to_sept4 : ℕ :=
  let days_in_january := 31
  let days_in_february := 28
  let days_in_march := 31
  let days_in_april := 30
  let days_in_may := 31
  let days_in_june := 30
  let days_in_july := 31
  let days_in_august := 31
  let days_up_to_sept4 := 4
  days_in_january + days_in_february + days_in_march + days_in_april +
  days_in_may + days_in_june + days_in_july + days_in_august + days_up_to_sept4

theorem anna_coaching_days : total_days_in_non_leap_year_up_to_sept4 = 247 :=
by
  -- Proof omitted
  sorry

end anna_coaching_days_l370_370167


namespace expectation_of_X_variance_of_X_l370_370960

noncomputable def pdf (n : ℕ) : ℝ → ℝ 
| x if x ≥ 0 := real.exp (-x) * x^n / n.factorial
| _ := 0

def M_X (n : ℕ) : ℝ :=
  ∫ x in set.Ioi 0, x * pdf n x

def M_X2 (n : ℕ) : ℝ :=
  ∫ x in set.Ioi 0, x^2 * pdf n x

def D_X (n : ℕ) : ℝ :=
  M_X2 n - (M_X n)^2

theorem expectation_of_X (n : ℕ) : 
  M_X n = n + 1 := 
sorry

theorem variance_of_X (n : ℕ) : 
  D_X n = n + 1 := 
sorry

end expectation_of_X_variance_of_X_l370_370960


namespace secondMatchLoser_l370_370866

-- Definitions for the problem statement
def players : List String := ["Nikanor", "Philemon", "Agathon"]

def knockOutTableTennisGame (totalMatches : Nat) : Prop :=
  let NikanorMatches := 10
  let PhilemonMatches := 15
  let AgathonMatches := 17
  let totalGames := (NikanorMatches + PhilemonMatches + AgathonMatches) / 2
  totalGames = totalMatches ∧
  NoConsecutiveMissGames

-- Main theorem to prove who lost the second match
theorem secondMatchLoser : ∀ (totalMatches : Nat),
  knockOutTableTennisGame totalMatches →
  "Nikanor" = "Nikanor" := sorry -- here we assert that Nikanor lost the second match

-- where "NoConsecutiveMissGames" and other auxiliary constructs might be further defined to align with game rules

end secondMatchLoser_l370_370866


namespace min_period_sin_cos_fn_l370_370387

theorem min_period_sin_cos_fn (n m : ℕ) (hn : 0 < n) (hm : 0 < m) :
    ∃ L, (∀ x, (sin x ^ (2 * n) - cos x ^ (2 * m - 1) = sin (x + L) ^ (2 * n) - cos (x + L) ^ (2 * m - 1))) ∧ (L > 0) ∧ ∀ L', (L' > 0) → (∀ x, sin x ^ (2 * n) - cos x ^ (2 * m - 1) = sin (x + L') ^ (2 * n) - cos (x + L') ^ (2 * m - 1)) → (L ≤ L') 
:= begin
    use 2 * π,
    split,
    { intro x,
      simp,
      sorry },
    split,
    { norm_num, },
    { intros L' hL' h,
      sorry }
end

end min_period_sin_cos_fn_l370_370387


namespace trigonometric_identity1_l370_370117

theorem trigonometric_identity1 (α : ℝ) :
  (cos (2 * α)) / (2 * tan (π/4 - α) * (sin (π/4 + α))^2) = 1 :=
by sorry

end trigonometric_identity1_l370_370117


namespace probability_XOX_OXO_l370_370225

open Nat

/-- Setting up the math problem to be proved -/
def X : Finset ℕ := {1, 2, 3, 4}
def O : Finset ℕ := {5, 6, 7}

def totalArrangements : ℕ := choose 7 4

def favorableArrangements : ℕ := 1

theorem probability_XOX_OXO : (favorableArrangements : ℚ) / (totalArrangements : ℚ) = 1 / 35 := by
  have h_total : totalArrangements = 35 := by sorry
  have h_favorable : favorableArrangements = 1 := by sorry
  rw [h_total, h_favorable]
  norm_num

end probability_XOX_OXO_l370_370225


namespace correct_statement_II_l370_370144

def digit_from_0_to_9 := {n : ℕ // n ≤ 9}

def statement_I (d : digit_from_0_to_9) : Prop := d.val = 0
def statement_II (d : digit_from_0_to_9) : Prop := d.val ≠ 1
def statement_III (d : digit_from_0_to_9) : Prop := d.val = 2
def statement_IV (d : digit_from_0_to_9) : Prop := d.val ≠ 3

theorem correct_statement_II (d : digit_from_0_to_9) :
  (∃ S : set (Prop), S = {statement_I d, statement_II d, statement_III d, statement_IV d} ∧ card S = 3 ∧
    ¬(∀ (s ∈ S), s)) → statement_II d :=
by
  -- Proof would go here
  sorry

end correct_statement_II_l370_370144


namespace sum_external_angles_inscribed_quadrilateral_l370_370532

theorem sum_external_angles_inscribed_quadrilateral (A B C D: Point)
  (circ : Circle): quadrilateral_inscribed A B C D circ → sum_external_angles A B C D = 360 :=
begin 
  sorry 
end

end sum_external_angles_inscribed_quadrilateral_l370_370532


namespace pirates_treasure_l370_370415

theorem pirates_treasure :
  ∃ m : ℕ,
    (m / 3 + 1) +
    (m / 4 + 5) +
    (m / 5 + 20) = m ∧
    m = 120 :=
by {
  sorry,
}

end pirates_treasure_l370_370415


namespace find_p_plus_q_l370_370039

theorem find_p_plus_q (x : ℝ) (p q : ℤ) (h₁ : Real.sec x + Real.tan x = 17 / 4)
  (h₂ : Real.csc x + Real.cot x = p / q) (h₃ : Int.gcd p q = 1) : p + q = 51 := by
  sorry

end find_p_plus_q_l370_370039


namespace bag_of_potatoes_weight_l370_370128

variable (W : ℝ)

-- Define the condition given in the problem.
def condition : Prop := W = 12 / (W / 2)

-- Define the statement we want to prove.
theorem bag_of_potatoes_weight : condition W → W = 24 := by
  intro h
  sorry

end bag_of_potatoes_weight_l370_370128


namespace minimum_value_f_b_eq_neg_a_maximum_value_ab_inequality_for_f_and_f_l370_370268

noncomputable def f (a b x : ℝ) := Real.exp x - a * x - b

theorem minimum_value_f_b_eq_neg_a (a : ℝ) (h : 0 < a) :
  ∃ m, m = 2 * a - a * Real.log a ∧ ∀ x : ℝ, f a (-a) x ≥ m :=
sorry

theorem maximum_value_ab (a b : ℝ) (h : ∀ x : ℝ, f a b x + a ≥ 0) :
  ab ≤ (1 / 2) * Real.exp 3 :=
sorry

theorem inequality_for_f_and_f' (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : b = -a) (h3 : f a b x1 = 0) (h4 : f a b x2 = 0) (h5 : x1 < x2)
  : f a (-a) (3 * Real.log a) > (Real.exp ((2 * x1 * x2) / (x1 + x2)) - a) :=
sorry

end minimum_value_f_b_eq_neg_a_maximum_value_ab_inequality_for_f_and_f_l370_370268


namespace solution_set_l370_370679

open BigOperators

noncomputable def f (x : ℝ) := 2016^x + Real.log (Real.sqrt (x^2 + 1) + x) / Real.log 2016 - 2016^(-x)

theorem solution_set (x : ℝ) (h1 : ∀ x, f (-x) = -f (x)) (h2 : ∀ x1 x2, x1 < x2 → f (x1) < f (x2)) :
  x > -1 / 4 ↔ f (3 * x + 1) + f (x) > 0 := 
by
  sorry

end solution_set_l370_370679


namespace total_amount_spent_correct_l370_370136

noncomputable def total_amount_spent (mango_cost pineapple_cost cost_pineapple total_people : ℕ) : ℕ :=
  let pineapple_people := cost_pineapple / pineapple_cost
  let mango_people := total_people - pineapple_people
  let mango_cost_total := mango_people * mango_cost
  cost_pineapple + mango_cost_total

theorem total_amount_spent_correct :
  total_amount_spent 5 6 54 17 = 94 := by
  -- This is where the proof would go, but it's omitted per instructions
  sorry

end total_amount_spent_correct_l370_370136


namespace pirates_treasure_l370_370425

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l370_370425


namespace integer_multiples_l370_370701

theorem integer_multiples (a b m : ℕ) (h1 : 200 = a) (h2 : 400 = b) (h3 : 117 = m) :
  ∃ (n : ℕ), n = 2 ∧ (∃ x, 200 ≤ x ∧ x ≤ 400 ∧ x % m = 0) :=
by {
  have h4 : a = 200 := h1,
  have h5 : b = 400 := h2,
  have h6 : m = 117 := h3,
  sorry
}

end integer_multiples_l370_370701


namespace angle_A_is_135_l370_370294

theorem angle_A_is_135 (a b c : ℝ) (A : ℝ) (h : (a + c) * (a - c) = b * (b + real.sqrt 2 * c)) : A = 135 :=
by sorry

end angle_A_is_135_l370_370294


namespace real_part_of_z_l370_370646

variable (z : ℂ) (a : ℝ)

noncomputable def condition1 : Prop := z / (2 + (a : ℂ) * Complex.I) = 2 / (1 + Complex.I)
noncomputable def condition2 : Prop := z.im = -3

theorem real_part_of_z (h1 : condition1 z a) (h2 : condition2 z) : z.re = 1 := sorry

end real_part_of_z_l370_370646


namespace distance_apart_after_6_hours_l370_370543

def distance (v1 v2 t : ℝ) : ℝ :=
  Real.sqrt ((v1 * t) ^ 2 + (v2 * t) ^ 2)

theorem distance_apart_after_6_hours
  (v1 v2 : ℝ) (hv1 : v1 = 12) (hv2 : v2 = 9) (d : ℝ) (hd : d = 90) :
  ∃ t : ℝ, distance v1 v2 t = d ∧ t = 6 :=
by
  sorry

end distance_apart_after_6_hours_l370_370543


namespace triangle_base_calculation_l370_370737

/--
Let the perimeter of a square be 64 and the height of a triangle be 32.
If the areas of the square and triangle are equal, then the base \( x \) of the triangle is equal to 16.
-/
theorem triangle_base_calculation {p h x : ℕ} (hp : p = 64) (hh : h = 32) (A_square = 256)
    (A_triangle = 16 * x) (h_eq : A_square = A_triangle) : x = 16 :=
by
  sorry

end triangle_base_calculation_l370_370737


namespace min_AB_distance_line_eq_tangents_through_P_l370_370688

noncomputable theory

variables {k : ℝ}
def line_l (k : ℝ) : ℝ × ℝ → Prop := 
  λ p, (2 * k + 1) * p.1 + (k - 1) * p.2 - (4 * k - 1) = 0

def circle_C : ℝ × ℝ → Prop := 
  λ p, p.1^2 + p.2^2 - 4 * p.1 - 2 * p.2 + 1 = 0

def point_P : ℝ × ℝ := (4, 4)

theorem min_AB_distance_line_eq :
  ∃ k, ∀ A B, line_l k A → line_l k B → circle_C A → circle_C B → -- line intersects the circle at points A and B
    let AB := (A.1 - B.1)^2 + (A.2 - B.2)^2 √ in
    AB = 2 * √2 ∧ ∀ k', (2 * k' + 1) * x + (k' - 1) * y - (4 * k' - 1) = 0 -> 
        (x - y + 1 = 0) :=
sorry

theorem tangents_through_P :
  ∃ k, ∀ x y, circle_C (x, y) → 
    let dist := abs((2 * k - 1) * √((4 - 4 * k) * (4 - 4 * k) + 1)) in
    dist = 2 →
    (5 * x - 12 * y + 28 = 0 ∨ x = 4) :=
sorry

end min_AB_distance_line_eq_tangents_through_P_l370_370688


namespace no_integer_solution_l370_370877

theorem no_integer_solution :
  ∀ x : ℤ, ¬ prime (|4 * x^2 - 39 * x + 35|) :=
by sorry

end no_integer_solution_l370_370877


namespace probability_C_D_l370_370929

variable (P : String → ℚ)

axiom h₁ : P "A" = 1/4
axiom h₂ : P "B" = 1/3
axiom h₃ : P "A" + P "B" + P "C" + P "D" = 1

theorem probability_C_D : P "C" + P "D" = 5/12 := by
  sorry

end probability_C_D_l370_370929


namespace score_correct_l370_370802

def numbers : List (String × ℝ) := [
  ("A", Real.pi),
  ("B", Real.sqrt 2 + Real.sqrt 3),
  ("C", Real.sqrt 10),
  ("D", 355 / 113),
  ("E", 16 * Real.arctan (1/5) - 4 * Real.arctan (1/240)),
  ("F", Real.log 23),
  ("G", 2 ^ Real.sqrt Real.exp 1)
]

def correct_ordered_subset : List String := ["F", "G", "A", "D", "E"]

def score_formula (N : ℕ) : ℕ := (N - 2) * (N - 3)

theorem score_correct :
  correct_ordered_subset.length = 5 →
  List.sort (λ x y, (numbers.lookup x).getOrElse 0 < (numbers.lookup y).getOrElse 0) correct_ordered_subset = correct_ordered_subset →
  score_formula 5 = 6 :=
by
  intros length_correct order_correct
  -- Proof here (to be provided)
  sorry

end score_correct_l370_370802


namespace total_boxes_l370_370722

theorem total_boxes (n : ℕ) (h : n = 10) : ∑ i in finset.range (n + 1), i = 55 :=
by
  rw h
  norm_num
  sorry

end total_boxes_l370_370722


namespace treasures_coins_count_l370_370408

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l370_370408


namespace simplify_and_evaluate_l370_370016

noncomputable def x : ℕ := 2023
noncomputable def y : ℕ := 2

theorem simplify_and_evaluate :
  (x + 2 * y)^2 - ((x^3 + 4 * x^2 * y) / x) = 16 :=
by
  sorry

end simplify_and_evaluate_l370_370016


namespace probability_reach_correct_l370_370955

noncomputable def probability_reach (n : ℕ) : ℚ :=
  (2/3) + (1/12) * (1 - (-1/3)^(n-1))

theorem probability_reach_correct (n : ℕ) (P_n : ℚ) :
  P_n = probability_reach n :=
by
  sorry

end probability_reach_correct_l370_370955


namespace minimum_value_of_f_symmetry_of_f_monotonic_decreasing_f_l370_370675

noncomputable def f (x : Real) : Real := Real.cos (2*x) - 2*Real.sin x + 1

theorem minimum_value_of_f : ∃ x : Real, f x = -2 := sorry

theorem symmetry_of_f : ∀ x : Real, f x = f (π - x) := sorry

theorem monotonic_decreasing_f : ∀ x y : Real, 0 < x ∧ x < y ∧ y < π / 2 → f y < f x := sorry

end minimum_value_of_f_symmetry_of_f_monotonic_decreasing_f_l370_370675


namespace polar_equivalence_l370_370303

def equivalent_polar_point : Prop :=
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
               (r, θ) = (4, 7 * Real.pi / 6) ∧
               (r, θ) is equivalent to (-4, Real.pi / 6)

theorem polar_equivalence : equivalent_polar_point :=
by
  -- Proof goes here
  sorry

end polar_equivalence_l370_370303


namespace determine_T_4544_l370_370627

-- Define the conditions
variables {b c : ℝ} (n : ℕ) (S : ℕ → ℝ)

-- Define the arithmetic sequence sum
def S_n (n : ℕ) : ℝ := (2 * b + (n - 1) * c) / 2 * n

-- Define the cumulative sum T_n
def T_n := ∑ k in finset.range(n + 1), S_n k

-- Given the value of S_3030
axiom S_3030_unique : S_n 3030 = 3030 * (b + 1514.5 * c)

-- The proof statement
theorem determine_T_4544 : ∃! T_4544, (S_n 3030 = 3030 * (b + 1514.5 * c)) → T_n 4544 = sorry :=
sorry

end determine_T_4544_l370_370627


namespace lines_perpendicular_l370_370391

theorem lines_perpendicular
    (ρ : ℝ) (θ : ℝ)
    (h1 : ρ * sin (θ + π / 4) = 2011)
    (h2 : ρ * sin (θ - π / 4) = 2012) :
    (are_perpendicular : ℝ) := by sorry

end lines_perpendicular_l370_370391


namespace pirates_treasure_l370_370419

theorem pirates_treasure (m : ℝ) :
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirates_treasure_l370_370419


namespace persimmons_count_l370_370062

variables {P T : ℕ}

-- Conditions from the problem
axiom total_eq : P + T = 129
axiom diff_eq : P = T - 43

-- Theorem to prove that there are 43 persimmons
theorem persimmons_count : P = 43 :=
by
  -- Putting the proof placeholder
  sorry

end persimmons_count_l370_370062


namespace lcm_of_12_15_18_is_180_l370_370885

theorem lcm_of_12_15_18_is_180 :
  Nat.lcm 12 (Nat.lcm 15 18) = 180 := by
  sorry

end lcm_of_12_15_18_is_180_l370_370885


namespace cost_of_items_l370_370008

variable (e t d : ℝ)

noncomputable def ques :=
  5 * e + 5 * t + 2 * d

axiom cond1 : 3 * e + 4 * t = 3.40
axiom cond2 : 4 * e + 3 * t = 4.00
axiom cond3 : 5 * e + 4 * t + 3 * d = 7.50

theorem cost_of_items : ques e t d = 6.93 :=
by
  sorry

end cost_of_items_l370_370008


namespace fraction_of_quarters_in_1790s_l370_370362

theorem fraction_of_quarters_in_1790s (total_coins : ℕ) (coins_in_1790s : ℕ) :
  total_coins = 30 ∧ coins_in_1790s = 7 → 
  (coins_in_1790s : ℚ) / total_coins = 7 / 30 :=
by
  sorry

end fraction_of_quarters_in_1790s_l370_370362


namespace projection_a_plus_e_on_a_eq_three_halves_l370_370658

variables (a e : euclidean_space ℝ (fin 3))
variables (h_a_norm : ∥a∥ = 2) (h_e_unit : ∥e∥ = 1)
variables (angle_a_e : real.angle a e = (2 * real.pi / 3))

theorem projection_a_plus_e_on_a_eq_three_halves
  (ha : ∥a∥ = 2) (he : ∥e∥ = 1) 
  (h_angle : real.angle a e = (2 * real.pi / 3)) :
  (∥a + e∥ * real.cos (real.angle (a + e) a)) = 3 / 2 :=
sorry

end projection_a_plus_e_on_a_eq_three_halves_l370_370658


namespace shortest_path_length_l370_370750

theorem shortest_path_length 
  (A B: ℝ × ℝ) 
  (C: ℝ × ℝ) 
  (r: ℝ) 
  [hA: A = (0,0)] 
  [hB: B = (15,20)] 
  [hC: C = (7,9)] 
  [hr: r = 6] 
  [h_circle: ∀ x y, (x - C.1)^2 + (y - C.2)^2 = r*r → false] : 
  (dist A C)  
  (arc_length r ((2:ℝ) * real.pi / 4)) 
  (dist C A)
  = 2 * real.sqrt 94 + 3 * real.pi := 
sorry

end shortest_path_length_l370_370750


namespace distinct_factors_of_product_l370_370789

theorem distinct_factors_of_product (m a b d : ℕ) (hm : m ≥ 1) (ha : m^2 < a ∧ a < m^2 + m)
  (hb : m^2 < b ∧ b < m^2 + m) (hab : a ≠ b) (hd : d ∣ (a * b)) (hd_range: m^2 < d ∧ d < m^2 + m) :
  d = a ∨ d = b :=
sorry

end distinct_factors_of_product_l370_370789


namespace number_1000_in_column_B_l370_370979

def column_sequence : List Char := ['B', 'C', 'D', 'E', 'F', 'E', 'D', 'C', 'B', 'A']

-- We want to prove that the 1000th integer falls in column 'B' given the pattern.
theorem number_1000_in_column_B :
  let pos := 999 % 10
  in column_sequence.nth pos = some 'B' :=
by
  sorry

end number_1000_in_column_B_l370_370979


namespace flux_through_plane_first_octant_l370_370209

noncomputable def vector_field (x y z : ℝ) : EuclideanSpace ℝ (Fin 3) := ![-x, 2*y, z]

noncomputable def plane_eq (x y z : ℝ) : Prop := x + 2*y + 3*z = 1

noncomputable def in_first_octant (x y z : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

noncomputable def normal_acute_with_oz (normal : EuclideanSpace ℝ (Fin 3)) : Prop := 
  (inner normal (EuclideanSpace.single 2 1) > 0)

theorem flux_through_plane_first_octant : 
  ∃ flux : ℝ, 
  flux = (1 / 18) ∧ 
  (∀ (x y z : ℝ), plane_eq x y z → in_first_octant x y z → normal_acute_with_oz (vector_field x y z)) :=
sorry

end flux_through_plane_first_octant_l370_370209


namespace tully_twice_kate_in_three_years_l370_370715

-- Definitions for the conditions
def tully_was := 60
def kate_is := 29

-- Number of years from now when Tully will be twice as old as Kate
theorem tully_twice_kate_in_three_years : 
  ∃ (x : ℕ), (tully_was + 1 + x = 2 * (kate_is + x)) ∧ x = 3 :=
by
  sorry

end tully_twice_kate_in_three_years_l370_370715


namespace kite_area_eq_twenty_find_a_plus_b_l370_370849

theorem kite_area_eq_twenty_find_a_plus_b (a b : ℝ)
  (h₁ : ∃ x, ax^2 + 3 = 0 ∧ x ∈ { -√(-3 / a), √(-3 / a) })
  (h₂ : ∃ x, 5 - bx^2 = 0 ∧ x ∈ { -√(5 / b), √(5 / b) })
  (h₃ : ∃ y₁ y₂, y₁ = 3 ∧ y₂ = 5)
  (h_area : 20 = 2 * √(5 / b)) :
  a + b = 1 / 50 :=
sorry

end kite_area_eq_twenty_find_a_plus_b_l370_370849


namespace sally_buttons_l370_370363

theorem sally_buttons :
  let shirts_monday := 4
  let shirts_tuesday := 3
  let shirts_wednesday := 2
  let buttons_per_shirt := 5
  (shirts_monday + shirts_tuesday + shirts_wednesday) * buttons_per_shirt = 45 :=
by
  let shirts_monday := 4
  let shirts_tuesday := 3
  let shirts_wednesday := 2
  let buttons_per_shirt := 5
  have h_shirts : shirts_monday + shirts_tuesday + shirts_wednesday = 9 := by rfl
  rw [h_shirts]
  have h_buttons : 9 * buttons_per_shirt = 45 := by norm_num
  exact h_buttons

end sally_buttons_l370_370363


namespace all_letters_identical_if_all_swaps_repetitive_l370_370923

def is_word {α : Type} (w : List α) : Prop := True

def is_repetitive {α : Type} (w : List α) : Prop :=
  ∃ u : List α, u ≠ [] ∧ w = u ++ u

def swap_adjacent {α : Type} (w : List α) (i : ℕ) : List α :=
  if i + 1 < w.length then
    (w.take i) ++ [w.get! (i + 1)] ++ [w.get! i] ++ (w.drop (i + 2))
  else w

def all_swaps_repetitive {α : Type} [DecidableEq α] (w : List α) : Prop :=
  ∀ i, i < w.length - 1 → is_repetitive (swap_adjacent w i)

theorem all_letters_identical_if_all_swaps_repetitive {α : Type} [DecidableEq α] (w : List α) :
  all_swaps_repetitive w → ∀ i j, i < w.length → j < w.length → w.get! i = w.get! j := 
begin
  sorry
end

end all_letters_identical_if_all_swaps_repetitive_l370_370923


namespace parabola_point_M_coordinates_triangle_area_MAB_l370_370956

theorem parabola_point_M_coordinates (p : ℝ) (hp : 0 < p) :
  ∃ (x y : ℝ), y^2 = 2 * p * x ∧ x = 4 - p / 2 ∧ y = 4 :=
sorry

theorem triangle_area_MAB (x y k x1 y1 x2 y2 : ℝ) (h_par : y = (8 * x) ^ (1 / 2))
(h_intersect : (x1, y1) ≠ (x2, y2))
(h_AB : (y1 + y2) = -8 ∧ (0, -1) ∈ set_of (λ (x y : ℝ), y = -x - 1))
(h_M : x = 2 ∧ y = 4)
: 0 < ((1/2) * 8 * (7 * real.sqrt 2 / 2)) :=
sorry

end parabola_point_M_coordinates_triangle_area_MAB_l370_370956


namespace minimum_value_expression_l370_370786

theorem minimum_value_expression (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  4 ≤ (5 * r) / (3 * p + q) + (5 * p) / (q + 3 * r) + (2 * q) / (p + r) :=
by sorry

end minimum_value_expression_l370_370786


namespace greatest_integer_b_for_no_real_roots_l370_370210

theorem greatest_integer_b_for_no_real_roots :
  ∃ (b : ℤ), (b * b < 20) ∧ (∀ (c : ℤ), (c * c < 20) → c ≤ 4) :=
by
  sorry

end greatest_integer_b_for_no_real_roots_l370_370210


namespace prime_power_m_l370_370616

theorem prime_power_m (m : ℕ) (hm : m > 1)
  (f : ℤ → ℤ)
  (hf0 : ∃ x : ℤ, f(x) % m = 0)
  (hf1 : ∃ x : ℤ, f(x) % m = 1) :
  ∃ p k : ℕ, Nat.Prime p ∧ k > 0 ∧ m = p^k :=
sorry

end prime_power_m_l370_370616


namespace min_f_x1x2_l370_370639

def f (x : ℝ) : ℝ := (Real.log x / Real.log 2 - 1) / (2 * (Real.log x / Real.log 2) + 1)

theorem min_f_x1x2 (x1 x2 : ℝ) (h1 : x1 > 2) (h2 : x2 > 2) 
(h3 : f x1 + f (2 * x2) = 1 / 2) :
  f (x1 * x2) ≥ 1 / 3 :=
by sorry

end min_f_x1x2_l370_370639


namespace find_total_coins_l370_370433

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l370_370433


namespace radius_of_circle_tangent_to_xaxis_l370_370739

theorem radius_of_circle_tangent_to_xaxis
  (Ω : Set (ℝ × ℝ)) (Γ : Set (ℝ × ℝ))
  (hΓ : ∀ x y : ℝ, (x, y) ∈ Γ ↔ y^2 = 4 * x)
  (F : ℝ × ℝ) (hF : F = (1, 0))
  (hΩ_tangent : ∃ r : ℝ, ∀ x y : ℝ, (x - 1)^2 + (y - r)^2 = r^2 ∧ (1, 0) ∈ Ω)
  (hΩ_intersect : ∀ x y : ℝ, (x, y) ∈ Ω → (x, y) ∈ Γ → (x, y) = (1, 0)) :
  ∃ r : ℝ, r = 4 * Real.sqrt 3 / 9 :=
sorry

end radius_of_circle_tangent_to_xaxis_l370_370739


namespace haley_seeds_in_big_garden_l370_370278

def seeds_in_big_garden (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  total_seeds - (small_gardens * seeds_per_small_garden)

theorem haley_seeds_in_big_garden :
  let total_seeds := 56
  let small_gardens := 7
  let seeds_per_small_garden := 3
  seeds_in_big_garden total_seeds small_gardens seeds_per_small_garden = 35 :=
by
  sorry

end haley_seeds_in_big_garden_l370_370278


namespace price_of_adult_ticket_l370_370497

theorem price_of_adult_ticket
  (price_child : ℤ)
  (price_adult : ℤ)
  (num_adults : ℤ)
  (num_children : ℤ)
  (total_amount : ℤ)
  (h1 : price_adult = 2 * price_child)
  (h2 : num_adults = 400)
  (h3 : num_children = 200)
  (h4 : total_amount = 16000) :
  num_adults * price_adult + num_children * price_child = total_amount → price_adult = 32 := by
    sorry

end price_of_adult_ticket_l370_370497


namespace num_students_third_section_l370_370964

-- Define the conditions
def num_students_first_section : ℕ := 65
def num_students_second_section : ℕ := 35
def num_students_fourth_section : ℕ := 42
def mean_marks_first_section : ℝ := 50
def mean_marks_second_section : ℝ := 60
def mean_marks_third_section : ℝ := 55
def mean_marks_fourth_section : ℝ := 45
def overall_average_marks : ℝ := 51.95

-- Theorem stating the number of students in the third section
theorem num_students_third_section
  (x : ℝ)
  (h : (num_students_first_section * mean_marks_first_section
       + num_students_second_section * mean_marks_second_section
       + x * mean_marks_third_section
       + num_students_fourth_section * mean_marks_fourth_section)
       = overall_average_marks * (num_students_first_section + num_students_second_section + x + num_students_fourth_section)) :
  x = 45 :=
by
  -- Proof will go here
  sorry

end num_students_third_section_l370_370964


namespace number_of_distinct_collections_l370_370798

noncomputable def distinct_letter_collections (letters : Multiset Char) : Nat := 
  let vowels := {'A', 'E', 'I'}.toMultiset
  let consonants := {'M', 'T', 'H', 'C', 'S'}.toMultiset
  (Multiset.count 'A' letters).choose 1 * (vowels - {'A'}.toMultiset).choose 2
  + (Multiset.count 'A' letters).choose 2 * (vowels - {'A', 'A'}.toMultiset).choose 1
  + 0

theorem number_of_distinct_collections : distinct_letter_collections "MATHEMATICS".toList.toMultiset = 33 := by
  sorry

end number_of_distinct_collections_l370_370798


namespace distinct_integer_solutions_l370_370780

theorem distinct_integer_solutions (n : ℕ) (h_pos : n > 0) (a : Fin n → ℕ) 
  (h_distinct : Function.Injective a) 
  (h_eq : ∑ j, a j ^ 3 = (∑ j, a j) ^ 2) : 
  ∃ b : Fin n → ℕ, (∀ j, b j = j + 1) ∧ (∀ j, a j = b j) := 
by
  sorry

end distinct_integer_solutions_l370_370780


namespace football_goals_in_fifth_match_l370_370518

theorem football_goals_in_fifth_match (G : ℕ) (h1 : (4 / 5 : ℝ) = (4 - G) / 4 + 0.3) : G = 2 :=
by
  sorry

end football_goals_in_fifth_match_l370_370518


namespace maximize_profit_l370_370129

def problem_1 (x : ℕ) : ℕ := 10 - (2 * x) / 3

def profit_function (x : ℕ) (y : ℕ) : ℕ := 4800 * x + 9000 * y

theorem maximize_profit : 
  ∃ (x y : ℕ), 
  (4 * x + 6 * y = 60) ∧ 
  (x ≥ y) ∧ 
  y = problem_1 x ∧ 
  profit_function x y = -1200 * x + 90000 ∧ 
  ∀ z : ℕ, z ≥ 6 → profit_function 6 6 ≥ profit_function z (problem_1 z) := 
sorry

end maximize_profit_l370_370129


namespace unique_two_digit_t_l370_370625

theorem unique_two_digit_t : 
  ∃! t : ℕ, (t > 9 ∧ t < 100) ∧ (13 * t ≡ 47 [MOD 100]) := 
begin
  use 19,
  split,
  { split,
    { linarith, },
    { norm_num, }, },
  { intros t ht,
    cases ht with ht1 ht2,
    have h : (77 : ℤ) * 47 ≡ 1 [ZMOD 100],
    { norm_num, },
    have h' : t * 13 ≡ 47 [ZMOD 100],
    { exact_mod_cast ht2, },
    rw [← h', ← mul_assoc, mul_comm 13, mul_assoc, ← zmod.eq_iff_modeq_nat] at h,
    norm_num [mul_inv_cancel_left] at h,
    have h2 : t ≡ 19 [MOD 100],
    { exact_mod_cast h, },
    rw zmod.eq_iff_modeq_nat at h2,
    norm_num at h2, }
end

end unique_two_digit_t_l370_370625


namespace stratified_sampling_activity_l370_370510

theorem stratified_sampling_activity 
  (total_students : ℕ)
  (third_year_students : ℕ)
  (selected_students : ℕ)
  (first_and_second_year_students : ℕ := total_students - third_year_students)
  (selection_rate : ℚ := selected_students / total_students) :
  total_students = 1000 → 
  third_year_students = 300 → 
  selected_students = 20 → 
  first_and_second_year_students * selection_rate = 14 :=
by
  intros h1 h2 h3
  have h4: first_and_second_year_students = 700, from by
    simp [first_and_second_year_students, h1, h2]
  have h5: selection_rate = 0.02, from by
    simp [selection_rate, h1, h3]
  simp [h4, h5]
  norm_num


end stratified_sampling_activity_l370_370510


namespace correct_choices_l370_370095

theorem correct_choices :
  (∃ u : ℝ × ℝ, (2 * u.1 + u.2 + 3 = 0) → u = (1, -2)) ∧
  ¬ (∀ a : ℝ, (a = -1 ↔ a^2 * x - y + 1 = 0 ∧ x - a * y - 2 = 0) → a = -1) ∧
  ((∃ (l : ℝ) (P : ℝ × ℝ), l = x + y - 6 → P = (2, 4) → 2 + 4 = l) → x + y - 6 = 0) ∧
  ((∃ (m b : ℝ), y = m * x + b → b = -2) → y = 3 * x - 2) :=
sorry

end correct_choices_l370_370095


namespace modified_triangle_array_sum_100_l370_370591

def triangle_array_sum (n : ℕ) : ℕ :=
  2^n - 2

theorem modified_triangle_array_sum_100 :
  triangle_array_sum 100 = 2^100 - 2 :=
sorry

end modified_triangle_array_sum_100_l370_370591


namespace find_certain_age_l370_370226

theorem find_certain_age 
(Kody_age : ℕ) 
(Mohamed_age : ℕ) 
(certain_age : ℕ) 
(h1 : Kody_age = 32) 
(h2 : Mohamed_age = 2 * certain_age) 
(h3 : ∀ four_years_ago, four_years_ago = Kody_age - 4 → four_years_ago * 2 = Mohamed_age - 4) :
  certain_age = 30 := sorry

end find_certain_age_l370_370226


namespace total_cupcakes_l370_370063

theorem total_cupcakes (children : ℕ) (cupcakes_per_child : ℕ) (h1 : children = 8) (h2 : cupcakes_per_child = 12) : children * cupcakes_per_child = 96 :=
by
  sorry

end total_cupcakes_l370_370063


namespace smallest_n_integer_val_l370_370592

def y1 := Real.root 4 4
def y2 := y1^y1
def sqrt4 := Real.root 4 4

-- Recursive definition of the sequence
noncomputable def y : ℕ → ℝ
| 1 => y1
| n+1 => (y n)^sqrt4

theorem smallest_n_integer_val : ∃ n : ℕ, y n ∈ Int ∧ (∀ m < n, y m ∉ Int) :=
  exists.intro 6 (and.intro (by
    sorry) (by
      intros m hm
      sorry))

end smallest_n_integer_val_l370_370592


namespace equation_is_linear_in_one_variable_l370_370670

theorem equation_is_linear_in_one_variable (n : ℤ) :
  (∀ x : ℝ, (n - 2) * x ^ |n - 1| + 5 = 0 → False) → n = 0 := by
  sorry

end equation_is_linear_in_one_variable_l370_370670


namespace four_digit_number_divisible_by_9_l370_370359

theorem four_digit_number_divisible_by_9
    (a b c d e f g h i j : ℕ)
    (h₀ : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
               f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
               g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
               h ≠ i ∧ h ≠ j ∧
               i ≠ j )
    (h₁ : a + b + c + d + e + f + g + h + i + j = 45)
    (h₂ : 100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 * g + 100 * h + 10 * i + j) :
  ((1000 * g + 100 * h + 10 * i + j) % 9 = 0) := sorry

end four_digit_number_divisible_by_9_l370_370359


namespace cube_edge_length_l370_370862

theorem cube_edge_length (sum_edges length_edge : ℝ) (cube_has_12_edges : 12 * length_edge = sum_edges) (sum_edges_eq_144 : sum_edges = 144) : length_edge = 12 :=
by
  sorry

end cube_edge_length_l370_370862


namespace next_in_sequence_l370_370243

-- Given sequence definition
def seq : ℕ → ℤ
| 0       := 1
| 1       := -5
| 2       := 9
| 3       := -13
| 4       := 17
| (n + 5) := (if n % 2 = 0 then seq (n + 4) - 4 * (2 * (n // 2) + 3) else seq (n + 4) + 4 * (2 * (n // 2) + 3))

theorem next_in_sequence : seq 5 = -21 := by
  sorry

end next_in_sequence_l370_370243


namespace exists_point_in_at_least_15_circles_l370_370298

noncomputable section

/- Define the conditions -/
def circles : Set (Set Point) := sorry -- Assume circles are sets of points.

def circles_have_common_point (c1 c2 : Set Point) (h1 : c1 ∈ circles) (h2 : c2 ∈ circles) : ∃ p : Point, p ∈ c1 ∧ p ∈ c2 := sorry

/- Main theorem statement -/
theorem exists_point_in_at_least_15_circles (h : circles.finite ∧ circles.card = 100) : ∃ p : Point, ∣{c ∈ circles | p ∈ c}∣ ≥ 15 :=
sorry

end exists_point_in_at_least_15_circles_l370_370298


namespace gcd_of_three_numbers_l370_370620

-- Define the given numbers
def a := 72
def b := 120
def c := 168

-- Define the GCD function and prove the required statement
theorem gcd_of_three_numbers : Nat.gcd (Nat.gcd a b) c = 24 := by
  -- Intermediate steps and their justifications would go here in the proof, but we are putting sorry
  sorry

end gcd_of_three_numbers_l370_370620


namespace distinct_equilateral_triangles_count_l370_370272

def ten_sided_polygon : Type := {P : Fin 10 → (ℝ × ℝ) // is_regular_polygon P 10}

noncomputable def number_of_distinct_equilateral_triangles (P : ten_sided_polygon) : Nat :=
  by
    sorry

theorem distinct_equilateral_triangles_count (P : ten_sided_polygon) :
  number_of_distinct_equilateral_triangles P = 82 :=
  by
    sorry

end distinct_equilateral_triangles_count_l370_370272


namespace max_value_of_f_l370_370386

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x - 5) + real.sqrt (24 - 3 * x)

theorem max_value_of_f :
  ∃ x ∈ set.Icc 5 8, f x = 2 * real.sqrt 3 ∧
  ∀ y ∈ set.Icc 5 8, f y ≤ f x :=
begin
  sorry
end

end max_value_of_f_l370_370386


namespace range_y_l370_370605

noncomputable theory

def y (x : ℝ) : ℝ := x + 2 / x

theorem range_y : set.range (λ x, y x) = set.Ici 3 :=
by 
  let f : ℝ → ℝ := λ x, y x 
  have h_domain : ∀ x, x ≥ 2 → x + 2 / x ≥ 3, sorry
  have h_increasing : ∀ x₁ x₂, x₁ < x₂ → (x₁ ≥ 2) → (x₂ ≥ 2) → f x₁ < f x₂, sorry
  sorry

end range_y_l370_370605


namespace num_three_digit_integers_with_zero_in_units_place_divisible_by_30_l370_370282

noncomputable def countThreeDigitMultiplesOf30WithZeroInUnitsPlace : ℕ :=
  let a := 120
  let d := 30
  let l := 990
  (l - a) / d + 1

theorem num_three_digit_integers_with_zero_in_units_place_divisible_by_30 :
  countThreeDigitMultiplesOf30WithZeroInUnitsPlace = 30 := by
  sorry

end num_three_digit_integers_with_zero_in_units_place_divisible_by_30_l370_370282


namespace john_return_time_is_five_hours_l370_370761

-- Definitions for the conditions
def distance_to_city (speed : ℕ) (time : ℕ) : ℕ :=
  speed * time

def speed_return_trip (initial_speed : ℕ) (increase : ℕ) : ℕ :=
  initial_speed + increase

def time_to_return (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

-- Theorem stating the problem with provided conditions
theorem john_return_time_is_five_hours :
  let distance := distance_to_city 60 6 in
  let return_speed := speed_return_trip 60 12 in
  time_to_return distance return_speed = 5 :=
by
  -- Main proof skipped
  sorry

end john_return_time_is_five_hours_l370_370761


namespace wrongly_noted_mark_l370_370821

theorem wrongly_noted_mark (x : ℕ) (h_wrong_avg : (30 : ℕ) * 100 = 3000)
    (h_correct_avg : (30 : ℕ) * 98 = 2940) (h_correct_sum : 3000 - x + 10 = 2940) : 
    x = 70 := by
  sorry

end wrongly_noted_mark_l370_370821


namespace value_of_a_l370_370261

theorem value_of_a (a : ℝ) :
    (∃ A B : ℝ × ℝ, (A.1 + A.2 = a ∧ A.1^2 + A.2^2 = 4 ∧
    B.1 + B.2 = a ∧ B.1^2 + B.2^2 = 4 ∧
    (∥(A.1, A.2) + (B.1, B.2)∥ = ∥(A.1, A.2) - (B.1, B.2)∥)) →
    (a = 2 ∨ a = -2)) :=
by
    sorry

end value_of_a_l370_370261


namespace sum_of_integer_values_l370_370085

theorem sum_of_integer_values (n : ℤ) (h : ∃ k : ℤ, 30 = k * (2 * n - 1)) : 
  ∑ n in {n | ∃ k : ℤ, 30 = k * (2 * n - 1)}, n = 14 :=
by
  sorry

end sum_of_integer_values_l370_370085


namespace max_parrots_l370_370857

theorem max_parrots (x y z : ℕ) (h1 : y + z ≤ 9) (h2 : x + z ≤ 11) : x + y + z ≤ 19 :=
sorry

end max_parrots_l370_370857


namespace solve_log_eq_l370_370817

theorem solve_log_eq (x : ℝ) (h1 : log 2 x + log 8 x = 9) : x = 2^(27/4) :=
sorry

end solve_log_eq_l370_370817


namespace probability_top_face_odd_l370_370593

/-- An 8-sided die has faces numbered 1 through 8, such that the face number indicates its dots. 
    Two dots are removed at random. Prove that the probability that the top face has an odd number 
    of dots is 1/2. -/
theorem probability_top_face_odd (remove_dot_probability : ℚ) (prob_top_odd : ℚ) 
  (prob_odd_face_remain_odd : ℚ) (prob_even_face_remain_even : ℚ)
  (face_count : ℕ) 
  (total_dots : ℕ) 
  (dot_removal_count : ℕ) 
  (die_faces : list ℕ)
  (h1 : face_count = 8)
  (h2 : total_dots = list.sum die_faces)
  (h3 : dot_removal_count = 2)
  (h4 : ∀ f ∈ die_faces, 1 ≤ f ∧ f ≤ 8)
  (h5 : (∀ f ∈ die_faces, f % 2 = 1 → prob_odd_face_remain_odd = 1 / 18) ∧ 
        (∀ f ∈ die_faces, f % 2 = 0 → prob_even_face_remain_even = 17 / 18))
  (h6 : remove_dot_probability = dot_removal_count / total_dots)
  (h7 : prob_top_odd = 1 / 2) 
  : prob_top_odd = 1 / 2 := 
sorry

end probability_top_face_odd_l370_370593


namespace number_of_valid_sets_l370_370710

-- Define the set and properties in Lean
def SatisfiesCondition (S : set ℕ) : Prop :=
  (∀ a ∈ S, 6 - a ∈ S) ∧ S ⊆ {1, 2, 3, 4, 5} ∧ S ≠ ∅

-- Theorem stating the number of such sets is 7
theorem number_of_valid_sets : {S : set ℕ | SatisfiesCondition S}.toFinset.card = 7 :=
sorry

end number_of_valid_sets_l370_370710


namespace length_PQ_eq_b_l370_370742

open Real

variables {a b : ℝ} (h : a > b) (p : ℝ × ℝ) (h₁ : (p.fst / a) ^ 2 + (p.snd / b) ^ 2 = 1)
variables (F₁ F₂ : ℝ × ℝ) (P Q : ℝ × ℝ)
variable (Q_on_segment : Q.1 = (F₁.1 + F₂.1) / 2)
variable (equal_inradii : inradius (triangle P Q F₁) = inradius (triangle P Q F₂))

theorem length_PQ_eq_b : dist P Q = b :=
by
  sorry

end length_PQ_eq_b_l370_370742


namespace Kamal_marks_in_Mathematics_l370_370764

theorem Kamal_marks_in_Mathematics (E P C B M : ℕ) (hE : E = 76) (hP : P = 82) (hC : C = 67) (hB : B = 85)
    (h_avg : (E + P + C + B + M) / 5 = 75) : M = 65 := by
  have h_total_marks_known : E + P + C + B = 76 + 82 + 67 + 85 := by
    rw [hE, hP, hC, hB]
  have h_total_marks_known_sum : E + P + C + B = 310 := by
    norm_num [h_total_marks_known]
  have h_total_marks : E + P + C + B + M = 375 := by
    rw ← nat.mul_eq_mul_right (zero_lt_succ 4)
    rw ← h_avg
    norm_cast
  have h_M : M = 375 - (E + P + C + B) := by
    rw h_total_marks
    rw h_total_marks_known_sum
    linarith
  exact h_M.symm


end Kamal_marks_in_Mathematics_l370_370764


namespace smallest_mult_to_cube_l370_370331

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_mult_to_cube (n : ℕ) (h : ∃ n, ∃ k, n * y = k^3) : n = 4500 := 
  sorry

end smallest_mult_to_cube_l370_370331


namespace digit_sum_divisible_by_5_l370_370280

def digitSum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem digit_sum_divisible_by_5 :
  { n | 1 ≤ n ∧ n ≤ 2001 ∧ digitSum n % 5 = 0 }.card = 399 := sorry

end digit_sum_divisible_by_5_l370_370280


namespace minimal_quotient_value_three_digit_number_l370_370154

theorem minimal_quotient_value_three_digit_number :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (∀ (d1 d2 d3 : ℕ), d1 ≠ d2 ∧ d2 ≠ d3 ∧ d3 ≠ d1 ∧ 
  d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ n = 100 * d1 + 10 * d2 + d3 ∧ (n % (d1 + d2 + d3) = 0) → 
  (↑n / (d1 + d2 + d3) = 10.5)) :=
sorry

end minimal_quotient_value_three_digit_number_l370_370154


namespace number_of_isosceles_triangles_l370_370704

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let dx := p2.1 - p1.1
  let dy := p2.2 - p1.2
  real.sqrt (dx^2 + dy^2)

def is_isosceles (v1 v2 v3 : ℝ × ℝ) : Prop :=
  let a := distance v1 v2
  let b := distance v2 v3
  let c := distance v1 v3
  a = b ∨ b = c ∨ a = c

def triangle_A_is_isosceles : Prop := is_isosceles (1, 5) (3, 5) (2, 3)
def triangle_B_is_isosceles : Prop := is_isosceles (4, 3) (4, 5) (6, 3)
def triangle_C_is_isosceles : Prop := is_isosceles (1, 2) (3, 1) (5, 2)
def triangle_D_is_isosceles : Prop := is_isosceles (7, 3) (6, 5) (9, 3)
def triangle_E_is_isosceles : Prop := is_isosceles (8, 2) (9, 4) (10, 1)

theorem number_of_isosceles_triangles : 
  (if triangle_A_is_isosceles then 1 else 0) +
  (if triangle_B_is_isosceles then 1 else 0) +
  (if triangle_C_is_isosceles then 1 else 0) +
  (if triangle_D_is_isosceles then 1 else 0) +
  (if triangle_E_is_isosceles then 1 else 0) = 4 :=
by sorry

end number_of_isosceles_triangles_l370_370704


namespace polynomial_exists_l370_370235

noncomputable def condition (f : ℤ → ℤ) : Prop :=
  ∀ p : ℕ, Prime p → ∃ Q_p : Polynomial ℤ, Q_p.degree ≤ 2013 ∧ ∀ n : ℤ, (f n - Q_p.eval n) % p = 0

theorem polynomial_exists (f : ℤ → ℤ) (hf : condition f) :
  ∃ g : Polynomial ℝ, ∀ n : ℤ, g.eval n = f n :=
sorry

end polynomial_exists_l370_370235


namespace m_range_l370_370375

noncomputable def f (m x : ℝ) : ℝ := -|m * x - 3|

theorem m_range (f : ℝ → ℝ → ℝ) (C : Set ℝ) (h1 : ∀ x, x ∈ C → f f (x + 6) ≤ f f x)
  (h2 : ∀ x, x ∈ [0, +∞) → f f x = -|birth_party0 _Party.mem.zero_]
  (h3 : ∀ x ∈ [0, +∞), f f (x + 6) ≤ f f x) :
  ∀ m, f f = (λ (m x : ℝ), -|m * x - 3|) → m ∈ (-∞, 0] ∪ [1, +∞) :=
begin
  sorry
end

end m_range_l370_370375


namespace garden_area_l370_370519

noncomputable def radius_of_semicircle := 8 / 2
noncomputable def area_of_semicircle := (1 / 2 : ℝ) * π * radius_of_semicircle^2
def area_of_square := radius_of_semicircle^2
def total_area := area_of_semicircle + area_of_square

theorem garden_area :
  total_area = 8 * π + 16 :=
by
  -- proofs here
  sorry

end garden_area_l370_370519


namespace integral_improper_rational_function_eq_l370_370211

open Real

noncomputable def improper_integral_rational_function : ℝ → ℝ := λ x,
  (x^3 / 3) + x^2 - x 
  + (1 / (4 * sqrt 2)) * log (abs (x^2 - (sqrt 2) * x + 1) / abs (x^2 + (sqrt 2) * x + 1))
  + (1 / (2 * sqrt 2)) * (arctan ((sqrt 2) * x + 1) + arctan ((sqrt 2) * x - 1))

theorem integral_improper_rational_function_eq :
  ∫ (x : ℝ) in 0..x, (x^6 + 2 * x^5 - x^4 + x^2 + 2 * x) / (x^4 + 1) = 
  (improper_integral_rational_function x) + C := by
  sorry

end integral_improper_rational_function_eq_l370_370211


namespace remainder_3_pow_17_mod_5_l370_370083

theorem remainder_3_pow_17_mod_5 :
  (3^17) % 5 = 3 :=
by
  have h : 3^4 % 5 = 1 := by norm_num
  sorry

end remainder_3_pow_17_mod_5_l370_370083


namespace treasures_coins_count_l370_370404

theorem treasures_coins_count : ∃ m : ℕ, 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m ∧ m = 120 :=
by
  sorry

end treasures_coins_count_l370_370404


namespace arithmetic_sequence_properties_l370_370661

theorem arithmetic_sequence_properties (a_1 d : ℤ)
    (h_d_neg : d < 0) 
    (h2 : (a_1 + d) * (a_1 + 3 * d) = 12)
    (h5 : a_1 + (a_1 + 4 * d) = 8) : 
    a_1 = 4 ∧ d = -1 ∧ (∑ k in Finset.range(10), a_1 + k * d) = -5 := by
  sorry

end arithmetic_sequence_properties_l370_370661


namespace range_subset_of_codomain_l370_370631

variables {A B : Type*} (f : A → B)

theorem range_subset_of_codomain : set.range f ⊆ set.univ : B :=
by
  sorry

end range_subset_of_codomain_l370_370631


namespace river_current_speed_l370_370953

theorem river_current_speed (v s_man t_total d_total : ℝ) 
  (h_man_speed : s_man = 7.5) 
  (h_total_time : t_total = 5 / 6) 
  (h_total_distance : d_total = 6) :
  (v = 1.5) :=
by
  -- declare the effective speed in still water
  let u_up := s_man - v
  let u_down := s_man + v
  -- the distance each way is constant
  let d_each_way := d_total / 2
  -- total time as sum of upstream and downstream times
  let t_each_way := d_each_way / u_up + d_each_way / u_down
  have t_each_way_eq : t_each_way = t_total
    := by sorry
  have : eq t_each_way (5/6)
    := by sorry
  -- thus, conclude the proof
  have proof_result : v = 1.5
    := by sorry
  exact ((eq_of_sub_eq_zero_iff v 1.5) proof_result)
  sorry

end river_current_speed_l370_370953


namespace max_value_of_y_tan_alpha_value_l370_370118

-- For problem (1)
theorem max_value_of_y (x : ℝ) (h₁ : x ∈ set.Icc (Real.pi / 6) (7 * Real.pi / 6)) : 
  ∃ y, y = 2 ∧ ∀ z, z = 3 - Real.sin x - 2 * (Real.cos x) ^ 2 → z ≤ y := 
begin
  sorry
end

-- For problem (2)
theorem tan_alpha_value (α β : ℝ) (h₁ : 5 * Real.sin β = Real.sin (2 * α + β)) (h₂ : Real.tan (α + β) = 9 / 4) : 
  Real.tan α = 3 / 2 := 
begin
  sorry
end

end max_value_of_y_tan_alpha_value_l370_370118


namespace arithmetic_sequence_common_difference_l370_370918

variable (a₁ d : ℝ)

def sum_odd := 5 * a₁ + 20 * d
def sum_even := 5 * a₁ + 25 * d

theorem arithmetic_sequence_common_difference 
  (h₁ : sum_odd a₁ d = 15) 
  (h₂ : sum_even a₁ d = 30) :
  d = 3 := 
by
  sorry

end arithmetic_sequence_common_difference_l370_370918


namespace simplify_cubic_root_l370_370022

theorem simplify_cubic_root (a b : ℕ) (h1 : 2744000 = b) (h2 : ∛ b = 140) (h3 : 5488000 = 2 * b) :
  ∛ 5488000 = 140 * ∛ 2 :=
by
  sorry

end simplify_cubic_root_l370_370022


namespace total_plates_l370_370756

-- Define the initial conditions
def flower_plates_initial : ℕ := 4
def checked_plates : ℕ := 8
def polka_dotted_plates := 2 * checked_plates
def flower_plates_remaining := flower_plates_initial - 1

-- Prove the total number of plates Jack has left
theorem total_plates : flower_plates_remaining + polka_dotted_plates + checked_plates = 27 :=
by
  sorry

end total_plates_l370_370756


namespace correct_statement_l370_370894

theorem correct_statement :
  let A := ∀ x : ℝ, x^2 = -4 → x = -2
  let B := ∀ x : ℝ, x^2 = 25 → x = 5
  let C := ∀ x : ℝ, x^2 = 9 → x = 3
  let D := ∀ x : ℝ, x^3 = 8 → x = 2 ∨ x = -2
  B := (∃! x : ℝ, x^2 = 25 ∧ x ≥ 0) :=
  B sorry

end correct_statement_l370_370894


namespace longer_diagonal_rhombus_l370_370961

theorem longer_diagonal_rhombus (a b d1 d2 : ℝ) 
  (h1 : a = 35) 
  (h2 : d1 = 42) : 
  d2 = 56 := 
by 
  sorry

end longer_diagonal_rhombus_l370_370961


namespace categorize_numbers_l370_370610

noncomputable def num_list : List ℝ := [
  0 + 0.12121212, 0, -9, -6.8, 2 - Real.pi, 13/9, -Real.sqrt 2, 0.8, Real.sqrt3 729, 0.7373373337, Real.sqrt ((-3)^2)
]

def is_irrational (x : ℝ) : Prop := ¬(∃ a b : ℤ, b ≠ 0 ∧ x = a / b)

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

def is_fraction (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

def is_real (x : ℝ) : Prop := True

def irrationals : List ℝ := [2 - Real.pi, -Real.sqrt 2, 0.7373373337]

def integers : List ℝ := [0, -9, Real.sqrt3 729, Real.sqrt ((-3)^2)]

def fractions : List ℝ := [0 + 0.12121212, -6.8, 13/9, 0.8]

def real_numbers : List ℝ := [
  0 + 0.12121212, 0, -9, -6.8, 2 - Real.pi, 13/9, -Real.sqrt 2, 0.8, Real.sqrt3 729, 0.7373373337, Real.sqrt ((-3)^2)
]

theorem categorize_numbers :
  (∀ x ∈ irrationals, is_irrational x) ∧
  (∀ x ∈ integers, is_integer x) ∧
  (∀ x ∈ fractions, is_fraction x) ∧
  (∀ x ∈ real_numbers, is_real x) := by
  sorry

end categorize_numbers_l370_370610


namespace probability_of_not_adjacent_to_edge_is_16_over_25_l370_370140

def total_squares : ℕ := 100
def perimeter_squares : ℕ := 36
def non_perimeter_squares : ℕ := total_squares - perimeter_squares
def probability_not_adjacent_to_edge : ℚ := non_perimeter_squares / total_squares

theorem probability_of_not_adjacent_to_edge_is_16_over_25 :
  probability_not_adjacent_to_edge = 16 / 25 := by
  sorry

end probability_of_not_adjacent_to_edge_is_16_over_25_l370_370140


namespace parametric_equation_valid_l370_370189

noncomputable def parametric_forms (t : ℝ) : Prop :=
(x y : ℝ) => 
(x = t^(1/2) ∧ y = t^(-1/2) → x * y ≠ 1) ∧
(∃ t, x = sin t ∧ y = (1 / sin t) → x ≠ 0 ∧ x * y ≠ 1) ∧
(∃ t, x = cos t ∧ y = (1 / cos t) → x ≠ 0 ∧ x * y ≠ 1) ∧
(∃ t, x = tan t ∧ y = (1 / tan t) → x * y = 1)

theorem parametric_equation_valid :
∃ t : ℝ, parametric_forms t := sorry

end parametric_equation_valid_l370_370189


namespace total_distance_from_A_through_B_to_C_l370_370065

noncomputable def distance_A_B_map : ℝ := 120
noncomputable def distance_B_C_map : ℝ := 70
noncomputable def map_scale : ℝ := 10 -- km per cm

noncomputable def distance_A_B := distance_A_B_map * map_scale -- Distance from City A to City B in km
noncomputable def distance_B_C := distance_B_C_map * map_scale -- Distance from City B to City C in km
noncomputable def total_distance := distance_A_B + distance_B_C -- Total distance in km

theorem total_distance_from_A_through_B_to_C :
  total_distance = 1900 := by
  sorry

end total_distance_from_A_through_B_to_C_l370_370065


namespace part1_part2_probability_part2_expected_value_l370_370043

-- Definitions based on conditions from the problem
def male_students (n : ℕ) := 10 * n
def female_students (n : ℕ) := 10 * n
def K_squared (n : ℕ) := 4.040

-- Given the total male and female students surveyed is 10n
-- Given K^2 ≈ 4.040 from the table
-- Proof of n = 20 assuming K_squared is accurate
theorem part1 (n : ℕ) (h1 : K_squared n = 4.040)
(h2 : 20 * n / 99 = 4.040) : n = 20 := sorry

-- Definitions based on conditions in problem 2.1
def total_students_not_understand := 9
def total_females_not_understand := 5
def total_males_not_understand := 4
noncomputable def prob_at_least_one_female_selected : ℚ :=
1 - (Nat.choose 4 3 : ℚ) / (Nat.choose 9 3)

-- Proof that the probability of selecting at least one female is 20/21
theorem part2_probability (h3 : prob_at_least_one_female_selected = 20/21) :
prob_at_least_one_female_selected = 20 / 21 := sorry

-- Definitions based on conditions in problem 2.2
noncomputable def expected_value (n : ℚ) := 10 * (11/20 : ℚ)

-- Proof that the expected value of X is 11/2
theorem part2_expected_value : expected_value 10 = 11 / 2 := sorry

end part1_part2_probability_part2_expected_value_l370_370043


namespace inequality_system_solution_l370_370030

theorem inequality_system_solution (x : ℝ) (h1 : 2 + x > 7 - 4x) (h2 : x < (4 + x) / 2) : 1 < x ∧ x < 4 :=
by
  sorry -- Proof goes here

end inequality_system_solution_l370_370030


namespace probability_C_D_l370_370930

variable (P : String → ℚ)

axiom h₁ : P "A" = 1/4
axiom h₂ : P "B" = 1/3
axiom h₃ : P "A" + P "B" + P "C" + P "D" = 1

theorem probability_C_D : P "C" + P "D" = 5/12 := by
  sorry

end probability_C_D_l370_370930


namespace range_of_values_l370_370636

theorem range_of_values (x y : ℝ) (h : (x + 2)^2 + y^2 / 4 = 1) :
  ∃ (a b : ℝ), a = 1 ∧ b = 28 / 3 ∧ a ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ b := by
  sorry

end range_of_values_l370_370636


namespace brian_oranges_is_12_l370_370992

-- Define the number of oranges the person has
def person_oranges : Nat := 12

-- Define the number of oranges Brian has, which is zero fewer than the person's oranges
def brian_oranges : Nat := person_oranges - 0

-- The theorem stating that Brian has 12 oranges
theorem brian_oranges_is_12 : brian_oranges = 12 :=
by
  -- Proof is omitted
  sorry

end brian_oranges_is_12_l370_370992


namespace pirates_treasure_l370_370429

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l370_370429


namespace sufficient_but_not_necessary_condition_l370_370048

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (m = -2) → (∀ x y, ((m + 2) * x + m * y + 1 = 0) ∧ ((m - 2) * x + (m + 2) * y - 3 = 0) → (m = 1) ∨ (m = -2)) → (m = -2) → (∀ x y, ((m + 2) * x + m * y + 1 = 0) ∧ ((m - 2) * x + (m + 2) * y - 3 = 0) → false) :=
by
  intros hm h_perp h
  sorry

end sufficient_but_not_necessary_condition_l370_370048


namespace variance_transformed_seq_l370_370040

variable (k : ℕ → ℝ) (σ² : ℝ)
-- Condition: The variance of k_i, k_2, ..., k_8 is 3
axiom var_k : ∑ i in finset.range 8, (k i - (1/8) * ∑ i in finset.range 8, k i)^2 / 8 = 3

-- Define the transformed sequence
def transformed_seq (i : ℕ) : ℝ := 2 * (k i - 3)

-- The target is to prove the variance of the transformed sequence is 12
theorem variance_transformed_seq : 
  ∑ i in finset.range 8, (transformed_seq k i - (1/8) * ∑ i in finset.range 8, transformed_seq k i)^2 / 8 = 12 := 
sorry

end variance_transformed_seq_l370_370040


namespace ratio_of_areas_l370_370148

theorem ratio_of_areas (l b : ℕ) (hl : l > 0) (hb : b > 0) :
  let A_original := l * b,
      A_new := (2 * l) * (3 * b) in A_original / A_new = (1 : ℚ) / 6 :=
by
  sorry

end ratio_of_areas_l370_370148


namespace first_term_of_geo_series_l370_370166

-- Define the conditions
def common_ratio : ℚ := 1 / 4
def sum_S : ℚ := 40

-- Define the question to be proven
theorem first_term_of_geo_series (a : ℚ) (h : sum_S = a / (1 - common_ratio)) : a = 30 := 
by
  sorry

end first_term_of_geo_series_l370_370166


namespace tim_original_vocab_l370_370073

theorem tim_original_vocab (days_in_year : ℕ) (years : ℕ) (learned_per_day : ℕ) (vocab_increase : ℝ) :
  let days := days_in_year * years
  let learned_words := learned_per_day * days
  let original_vocab := learned_words / vocab_increase
  original_vocab = 14600 :=
by
  let days := days_in_year * years
  let learned_words := learned_per_day * days
  let original_vocab := learned_words / vocab_increase
  show original_vocab = 14600
  sorry

end tim_original_vocab_l370_370073


namespace average_speed_round_trip_l370_370482

def distance : ℝ := sorry
def speed_to_sf : ℝ := 48
def time_to_sf (D : ℝ) (S : ℝ) : ℝ := D / S
def time_back_sf (T : ℝ) : ℝ := 2 * T
def total_distance (D : ℝ) : ℝ := 2 * D
def total_time (T1 T2 : ℝ) : ℝ := T1 + T2

theorem average_speed_round_trip 
  (D : ℝ) 
  (H1 : speed_to_sf = 48) 
  (H2 : time_back_sf (time_to_sf D speed_to_sf) = 2 * time_to_sf D speed_to_sf) 
  (H3 : total_distance D = 2 * D)
  (H4 : total_time (time_to_sf D speed_to_sf) (time_back_sf (time_to_sf D speed_to_sf)) = (3 * D) / 48) :
  2 * D / ((3 * D) / 48) = 32 :=
by
  sorry

end average_speed_round_trip_l370_370482


namespace decimal_to_base9_l370_370186

theorem decimal_to_base9 (n : ℕ) (h : n = 1729) : 
  (2 * 9^3 + 3 * 9^2 + 3 * 9^1 + 1 * 9^0) = n :=
by sorry

end decimal_to_base9_l370_370186


namespace transformedGraphEquation_l370_370868

variable (x : ℝ)

def originalFunction : ℝ → ℝ := λ x, Real.sin x

def translatedFunction : ℝ → ℝ := λ x, originalFunction (x + Real.pi / 3)

def stretchedFunction : ℝ → ℝ := λ x, translatedFunction (x / 2)

/-- The graph transformation results in the function y = sin((x / 2) + pi / 3) -/
theorem transformedGraphEquation :
  stretchedFunction x = Real.sin (x / 2 + Real.pi / 3) :=
sorry

end transformedGraphEquation_l370_370868


namespace tan_cot_eq_unique_solution_l370_370195

theorem tan_cot_eq_unique_solution :
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), tan (Real.pi / 3 + Real.pi / 4 * Real.cos x) = Real.cot (Real.pi / 4 * Real.sin x)) ∧ 
  (∀ x y ∈ Set.Icc 0 (Real.pi / 2), tan (Real.pi / 3 + Real.pi / 4 * Real.cos x) = Real.cot (Real.pi / 4 * Real.sin x) ∧
    tan (Real.pi / 3 + Real.pi / 4 * Real.cos y) = Real.cot (Real.pi / 4 * Real.sin y) → x = y) :=
sorry

end tan_cot_eq_unique_solution_l370_370195


namespace average_speed_without_stoppages_l370_370539

variables (d : ℝ) (t : ℝ) (v_no_stop : ℝ)

-- The train stops for 12 minutes per hour
def stoppage_per_hour := 12 / 60
def moving_fraction := 1 - stoppage_per_hour

-- Given speed with stoppages is 160 km/h
def speed_with_stoppage := 160

-- Average speed of the train without stoppages
def speed_without_stoppage := speed_with_stoppage / moving_fraction

-- The average speed without stoppages should equal 200 km/h
theorem average_speed_without_stoppages : speed_without_stoppage = 200 :=
by
  unfold speed_without_stoppage
  unfold moving_fraction
  unfold stoppage_per_hour
  norm_num
  sorry

end average_speed_without_stoppages_l370_370539


namespace sin_210_eq_neg_one_half_l370_370578

theorem sin_210_eq_neg_one_half :
  sin (Real.pi * (210 / 180)) = -1 / 2 :=
by
  have angle_eq : 210 = 180 + 30 := by norm_num
  have sin_30 : sin (Real.pi / 6) = 1 / 2 := by norm_num
  have cos_30 : cos (Real.pi / 6) = sqrt 3 / 2 := by norm_num
  sorry

end sin_210_eq_neg_one_half_l370_370578


namespace greatest_real_part_of_z6_is_neg3_l370_370354

def options : List Complex :=
  [(-3 : ℂ), (-2 * Real.sqrt 2 + 2 * Real.sqrt 2 * Complex.I), (-Real.sqrt 3 + Real.sqrt 3 * Complex.I),
   (-1 + 2 * Real.sqrt 3 * Complex.I), (3 * Complex.I)]

theorem greatest_real_part_of_z6_is_neg3 :
  ∀ z ∈ options, (Re (z ^ 6)) ≤ 729 :=
by
  sorry

end greatest_real_part_of_z6_is_neg3_l370_370354


namespace average_weight_increase_l370_370823

-- Define the initial conditions
def initial_weight : ℕ := 65
def new_weight : ℕ := 85
def num_people : ℕ := 8

-- Define the hypothesis and statement
theorem average_weight_increase :
  (new_weight - initial_weight) / num_people = 2.5 :=
by
  sorry

end average_weight_increase_l370_370823


namespace sin_210_eq_neg_one_half_l370_370584

theorem sin_210_eq_neg_one_half :
  ∀ (θ : ℝ), 
  θ = 210 * (π / 180) → -- angle 210 degrees
  ∃ (refθ : ℝ), 
  refθ = 30 * (π / 180) ∧ -- reference angle 30 degrees
  sin refθ = 1 / 2 → -- sin of reference angle
  sin θ = -1 / 2 := 
by
  intros θ hθ refθ hrefθ hrefθ_sin -- introduce variables and hypotheses
  sorry

end sin_210_eq_neg_one_half_l370_370584


namespace solve_log_eq_l370_370816

theorem solve_log_eq (x : ℝ) (h1 : log 2 x + log 8 x = 9) : x = 2^(27/4) :=
sorry

end solve_log_eq_l370_370816


namespace crayons_selection_l370_370932

theorem crayons_selection :
  ∃ (number_of_ways : ℕ), number_of_ways = Nat.choose 15 6 ∧ number_of_ways = 5005 :=
by
  use Nat.choose 15 6
  split
  · rfl
  · sorry

end crayons_selection_l370_370932


namespace multiply_fractions_l370_370993

theorem multiply_fractions :
  (2 / 3) * (5 / 7) * (8 / 9) = 80 / 189 :=
by sorry

end multiply_fractions_l370_370993


namespace bitangent_to_curve_l370_370826

def curve (x : ℝ) : ℝ := x^4 + 2*x^3 - 11*x^2 - 13*x + 35

def bitangent_line (x: ℝ) : ℝ := -x - 1

theorem bitangent_to_curve {x1 x2 : ℝ} 
  (h1 : curve x1 = bitangent_line x1) 
  (h2 : curve x2 = bitangent_line x2) 
  (hx1x2 : x1 ≠ x2) 
  (hderiv1 : deriv curve x1 = deriv bitangent_line x1) 
  (hderiv2 : deriv curve x2 = deriv bitangent_line x2) : 
  bitangent_line = - (λ x, x) - 1 := 
  by 
  sorry

end bitangent_to_curve_l370_370826


namespace sqrt_problem_l370_370173

theorem sqrt_problem : sqrt 12 - 3 * sqrt (1 / 3) = sqrt 3 :=
by
  -- Given conditions to be assumed implicitly within the proof context
  have h1: sqrt 12 = 2 * sqrt 3 := by sorry
  have h2: 3 * sqrt (1 / 3) = sqrt 3 := by sorry
  -- Main statement to be proven
  sorry -- Skip the actual proof as required

end sqrt_problem_l370_370173


namespace parabola_equation_l370_370853

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

-- Define the standard equation form of the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

-- Define the right vertex of the hyperbola
def right_vertex (a : ℝ) : ℝ × ℝ :=
  (a, 0)

-- State the final proof problem
theorem parabola_equation :
  hyperbola 4 0 →
  parabola 8 x y →
  y^2 = 16 * x :=
by
  -- Skip the proof for now
  sorry

end parabola_equation_l370_370853


namespace chris_birthday_days_l370_370996

theorem chris_birthday_days (mod : ℕ → ℕ → ℕ) (day_of_week : ℕ → ℕ) :
  (mod 75 7 = 5) ∧ (mod 30 7 = 2) →
  (day_of_week 0 = 1) →
  (day_of_week 75 = 6) ∧ (day_of_week 30 = 3) := 
sorry

end chris_birthday_days_l370_370996


namespace price_of_adult_ticket_l370_370498

theorem price_of_adult_ticket
  (price_child : ℤ)
  (price_adult : ℤ)
  (num_adults : ℤ)
  (num_children : ℤ)
  (total_amount : ℤ)
  (h1 : price_adult = 2 * price_child)
  (h2 : num_adults = 400)
  (h3 : num_children = 200)
  (h4 : total_amount = 16000) :
  num_adults * price_adult + num_children * price_child = total_amount → price_adult = 32 := by
    sorry

end price_of_adult_ticket_l370_370498


namespace train_crossing_time_l370_370540

variable (speed_kmph : ℕ)
variable (length_meters : ℝ)

noncomputable def time_to_cross_pole (speed_kmph : ℕ) (length_meters : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600 in
  length_meters / speed_mps

theorem train_crossing_time :
  time_to_cross_pole 180 250.02 ≈ 5 := 
by
  sorry

end train_crossing_time_l370_370540


namespace sin_210_l370_370566

theorem sin_210 : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end sin_210_l370_370566


namespace max_value_4287_5_l370_370330

noncomputable def maximum_value_of_expression (x y : ℝ) := x * y * (105 - 2 * x - 5 * y)

theorem max_value_4287_5 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y < 105) :
  maximum_value_of_expression x y ≤ 4287.5 :=
sorry

end max_value_4287_5_l370_370330


namespace four_points_cyclic_l370_370357

-- Define the structure of the problem
structure Triangle (α : Type*) :=
(A B C : α)

structure Point (α : Type*) :=
(x y : α)

-- Specify that there exist points that divide each side into three equal segments
variables {α : Type*} [Field α] {T : Triangle α}
(A1 A2 B1 B2 C1 C2 : Point α)

-- Hypotheses: Each side is divided into three equal segments
axiom div_eq_A1A2 : dist T.A A1 = dist A1 A2 ∧ dist A2 T.B = dist T.A A / 3
axiom div_eq_B1B2 : dist T.B B1 = dist B1 B2 ∧ dist B2 T.C = dist T.B C / 3
axiom div_eq_C1C2 : dist T.C C1 = dist C1 C2 ∧ dist C2 T.A = dist T.C A / 3

-- Proof goal: Prove that at least 4 of these 6 points lie on the same circle
theorem four_points_cyclic : 
  ∃ (O : Point α) (r : α), ∃ (P Q R S : Point α),
    (P = A1 ∨ P = A2 ∨ P = B1 ∨ P = B2 ∨ P = C1 ∨ P = C2) ∧ 
    (Q = A1 ∨ Q = A2 ∨ Q = B1 ∨ Q = B2 ∨ Q = C1 ∨ Q = C2) ∧ 
    (R = A1 ∨ R = A2 ∨ R = B1 ∨ R = B2 ∨ R = C1 ∨ R = C2) ∧
    (S = A1 ∨ S = A2 ∨ S = B1 ∨ S = B2 ∨ S = C1 ∨ S = C2) ∧ 
    P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ S ∧ 
    dist O P = r ∧ dist O Q = r ∧ dist O R = r ∧ dist O S = r :=
sorry

end four_points_cyclic_l370_370357


namespace twelve_points_on_sphere_l370_370748

noncomputable def semi_perimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

-- Defining the points and distances
theorem twelve_points_on_sphere 
  (A B C D A1 A2 A3 B1 B2 B3 C1 C2 C3 D1 D2 D3 : ℝ × ℝ × ℝ)
  (s_AB s_AC s_AD s_BC s_BD s_CD : ℝ)
  (hA1 : dist A A1 = s_CD)
  (hA2 : dist A A2 = s_CD)
  (hA3 : dist A A3 = s_CD)
  (hB1 : dist B B1 = s_AD)
  (hB2 : dist B B2 = s_AD)
  (hB3 : dist B B3 = s_AD)
  (hC1 : dist C C1 = s_BD)
  (hC2 : dist C C2 = s_BD)
  (hC3 : dist C C3 = s_BD)
  (hD1 : dist D D1 = s_BC)
  (hD2 : dist D D2 = s_BC)
  (hD3 : dist D D3 = s_BC)
  (h_sphere : ∃ S : set (ℝ × ℝ × ℝ), is_sphere S ∧ ∀ (P Q : ℝ × ℝ × ℝ), (P ∈ S ∧ Q ∈ S → dist P Q = s_AB ∨ dist P Q = s_AC ∨ dist P Q = s_AD ∨ dist P Q = s_BC ∨ dist P Q = s_BD ∨ dist P Q = s_CD)) :
  ∃ S' : set (ℝ × ℝ × ℝ), is_sphere S' ∧ ∀ p, p ∈ ({A1, A2, A3, B1, B2, B3, C1, C2, C3, D1, D2, D3} : set (ℝ × ℝ × ℝ)) → p ∈ S' :=
sorry

end twelve_points_on_sphere_l370_370748


namespace fraction_equiv_l370_370286

theorem fraction_equiv (m n : ℚ) (h : m / n = 3 / 4) : (m + n) / n = 7 / 4 :=
sorry

end fraction_equiv_l370_370286


namespace rectangle_area_is_72_l370_370068

noncomputable def congruent_circles_area (P Q R : Point) (A B C D : Point) 
    (radius : ℝ) (h1 : distance P Q = 2 * radius)
    (h2 : distance Q R = 2 * radius)
    (h3 : distance P R = 4 * radius)
    (hPQ: circle_centre_radius P radius)
    (hQR: circle_centre_radius Q radius)
    (hRP: circle_centre_radius R radius)
    (hA : A = (0, 0))
    (hB : B = (12, 0))
    (hC : C = (12, 6))
    (hD : D = (0, 6))
    (hTouch: ∀ {x : Point}, touches_two_sides x (rect A B C D)) : ℝ := 
  let height : ℝ := distance A D in
  let width : ℝ := distance A B in
  height * width

theorem rectangle_area_is_72 : 
  ∃ (P Q R A B C D : Point) (radius : ℝ),
  radius = 3 ∧ 
  distance P Q = 2 * radius ∧
  distance Q R = 2 * radius ∧
  distance P R = 4 * radius ∧
  circle_centre_radius P radius ∧
  circle_centre_radius Q radius ∧
  circle_centre_radius R radius ∧
  touches_two_sides P (rect A B C D) ∧
  touches_two_sides Q (rect A B C D) ∧
  touches_two_sides R (rect A B C D) ∧
  A = (0, 0) ∧
  B = (12, 0) ∧
  C = (12, 6) ∧
  D = (0, 6) ∧
  congruent_circles_area P Q R A B C D radius = 72 :=
sorry

end rectangle_area_is_72_l370_370068


namespace no_such_integers_l370_370990

theorem no_such_integers (a b : ℤ) : 
  ¬ (∃ a b : ℤ, ∃ k₁ k₂ : ℤ, a^5 * b + 3 = k₁^3 ∧ a * b^5 + 3 = k₂^3) :=
by 
  sorry

end no_such_integers_l370_370990


namespace max_omega_increasing_on_interval_l370_370680

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 4)

noncomputable def g (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem max_omega_increasing_on_interval :
  ∀ (ω : ℝ), (ω > 0) →
  (∀ x : ℝ, x ∈ set.Icc (-Real.pi / 6) (Real.pi / 4) → 
    ∀ x' : ℝ, x' ∈ set.Icc (-Real.pi / 6) (Real.pi / 4) → 
      (x < x' → g x ω < g x' ω)) → ω ≤ 2 :=
by
  sorry

end max_omega_increasing_on_interval_l370_370680


namespace dave_pickup_trays_l370_370600

theorem dave_pickup_trays :
  ∀ (trips trays_per_trip trays_from_one_table : ℕ),
    trays_per_trip = 9 →
    trays_from_one_table = 17 →
    trips = 8 →
    let total_trays := trips * trays_per_trip in
    let trays_from_second_table := total_trays - trays_from_one_table in
    trays_from_second_table = 55 :=
by
  intros trips trays_per_trip trays_from_one_table h1 h2 h3
  let total_trays := trips * trays_per_trip
  let trays_from_second_table := total_trays - trays_from_one_table
  sorry

end dave_pickup_trays_l370_370600


namespace geometric_sequence_common_ratio_l370_370720

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : 3 * a 0 + 2 * a 1 = a 2 / 0.5) :
  q = 3 :=
  sorry

end geometric_sequence_common_ratio_l370_370720


namespace roots_sum_l370_370659

open Function

-- Definitions for even and odd functions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

-- The main theorem stating the result
theorem roots_sum {f g : ℝ → ℝ} (h_even : is_even_function f) (h_odd : is_odd_function g)
  (a b c d : ℕ) :
  (∃ x : ℝ, f (f x) = 0) = a ∧
  (∃ x : ℝ, f (g x) = 0) = b ∧
  (∃ x : ℝ, g (g x) = 0) = c ∧
  (∃ x : ℝ, g (f x) = 0) = d →
  a + b + c + d = 8 :=
sorry

end roots_sum_l370_370659


namespace find_a_minus_b_l370_370641

theorem find_a_minus_b
  (f : ℝ → ℝ)
  (a b : ℝ)
  (hf : ∀ x, f x = x^2 + 3 * a * x + 4)
  (h_even : ∀ x, f (-x) = f x)
  (hb_condition : b - 3 = -2 * b) :
  a - b = -1 :=
sorry

end find_a_minus_b_l370_370641


namespace triangle_other_side_length_l370_370239

theorem triangle_other_side_length (a b : ℝ) (c : ℝ) (h_a : a = 3) (h_b : b = 4) (h_right_angle : c * c = a * a + b * b ∨ a * a = c * c + b * b):
  c = Real.sqrt 7 ∨ c = 5 :=
by
  sorry

end triangle_other_side_length_l370_370239


namespace maximum_g_sum_l370_370181

theorem maximum_g_sum :
  ∃(f g : ℕ → ℕ), (∀ n, n ≤ 300 → f n ≥ f (n + 1) ∧ f 300 ≥ 0) ∧
                (∑ i in Finset.range 301, f i ≤ 300) ∧
                (∀ (n_vector : Fin 20 → ℕ), g (∑ i, n_vector i) ≤ ∑ i, f (n_vector i)) ∧
                (∑ j in Finset.range 6001, g j = 115440) := sorry

end maximum_g_sum_l370_370181


namespace intersection_eq_l370_370273

def U : Set ℝ := {x : ℝ | True}
def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x ≥ 1}
def CU_N : Set ℝ := {x : ℝ | x < 1}

theorem intersection_eq : M ∩ CU_N = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_eq_l370_370273


namespace union_A_B_l370_370251

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def C : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem union_A_B : A ∪ B = C := 
by sorry

end union_A_B_l370_370251


namespace angle_between_planes_l370_370395

-- Define the necessary geometrical entities and conditions
variables (a : ℝ) (P Q : Plane) (B C : ℝ)
variables (AB AC BC : Triangle) (ϕ α β : ℝ)
variables (H K M : Point)
variables (perp_AC : IsPerpendicular AB AC)
variables (perp_AM : IsPerpendicular AB AM)
variables (triangle_equilateral : IsEquilateral AC)

-- Define the problem statement in Lean 4
theorem angle_between_planes 
  (h_phi : (ϕ ∈ acute_angles)) 
  (h_alpha : (α ∈ acute_angles)) 
  (h_AB_on_P : AB.on_face P) 
  (h_AC_on_Q : AC.on_face Q) 
  (h_angle_AB_edge : AB.forms_angle_with_edge α) :
  angle_plane_plane ABC Q = β := sorry

end angle_between_planes_l370_370395


namespace rational_root_count_l370_370179

-- Given conditions:
def polynomial (b3 b2 b1 : ℤ) : polynomial ℚ := 12 * X^4 + b3 * X^3 + b2 * X^2 + b1 * X + 18

-- The Rational Root Theorem allows us to determine possible rational roots.
theorem rational_root_count (b3 b2 b1 : ℤ) (h : ∃ (x : ℤ), polynomial b3 b2 b1 = C 0) :
  (finset.univ : finset ℚ).card = 13 := sorry

end rational_root_count_l370_370179


namespace ab_eq_zero_l370_370050

theorem ab_eq_zero (a b : ℤ) (h : ∀ m n : ℕ, ∃ k : ℤ, a * (m^2 : ℤ) + b * (n^2 : ℤ) = k^2) : a * b = 0 :=
by
  sorry

end ab_eq_zero_l370_370050


namespace log_equivalence_l370_370285

theorem log_equivalence (m : ℝ) (h : (2/(1 + Complex.I)) = 1 + m * Complex.I) :
  Real.logBase 4 (0.5 ^ m) = 1 / 2 := by
  sorry

end log_equivalence_l370_370285


namespace first_term_geometric_series_l370_370165

variable (a : ℝ)
variable (r : ℝ := 1/4)
variable (S : ℝ := 80)

theorem first_term_geometric_series 
  (h1 : r = 1/4) 
  (h2 : S = 80)
  : a = 60 :=
by 
  sorry

end first_term_geometric_series_l370_370165


namespace find_alpha_l370_370254

noncomputable def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * (a 2 / a 1)

-- Given that {a_n} is a geometric sequence,
-- a_1 and a_8 are roots of the equation
-- x^2 - 2x * sin(alpha) - √3 * sin(alpha) = 0,
-- and (a_1 + a_8)^2 = 2 * a_3 * a_6 + 6,
-- prove that alpha = π / 3.
theorem find_alpha :
  ∃ α : ℝ,
  (∀ (a : ℕ → ℝ), isGeometricSequence a ∧ 
  (∃ (a1 a8 : ℝ), 
    (a1 + a8)^2 = 2 * a 3 * a 6 + 6 ∧
    a1 + a8 = 2 * Real.sin α ∧
    a1 * a8 = - Real.sqrt 3 * Real.sin α)) →
  α = Real.pi / 3 :=
by 
  sorry

end find_alpha_l370_370254


namespace cds_probability_l370_370143

def probability (total favorable : ℕ) : ℚ := favorable / total

theorem cds_probability :
  probability 120 24 = 1 / 5 :=
by
  sorry

end cds_probability_l370_370143


namespace sin_210_eq_neg_half_l370_370558

theorem sin_210_eq_neg_half : Real.sin (210 * Real.pi / 180) = -1 / 2 := by
  -- We use the given angles and their known sine values.
  have angle_30 := Real.pi / 6
  have sin_30 := Real.sin angle_30
  -- Expression for the sine of 210 degrees in radians.
  have angle_210 := 210 * Real.pi / 180
  -- Proving the sine of 210 degrees using angle addition formula and unit circle properties.
  calc
    Real.sin angle_210 
    -- 210 degrees is 180 + 30 degrees, translating to pi and pi/6 in radians.
    = Real.sin (Real.pi + Real.pi / 6) : by rw [←Real.ofReal_nat_cast, ←Real.ofReal_nat_cast, Real.ofReal_add, Real.ofReal_div, Real.ofReal_nat_cast]
    -- Using the sine addition formula: sin(pi + x) = -sin(x).
    ... = - Real.sin (Real.pi / 6) : by exact Real.sin_add_pi_div_two angle_30
    -- Substituting the value of sin(30 degrees).
    ... = - 1 / 2 : by rw sin_30

end sin_210_eq_neg_half_l370_370558


namespace expected_total_rain_l370_370542

-- Define conditions for weekdays and weekends
def weekday_rain_expectation : ℝ := (0.30 * 0) + (0.20 * 5) + (0.50 * 8)
def weekend_rain_expectation : ℝ := (0.50 * 0) + (0.25 * 2) + (0.25 * 6)

-- The expected number of inches of rain from Monday to Sunday
theorem expected_total_rain : weekday_rain_expectation * 5 + weekend_rain_expectation * 2 = 29 := by
  calc
    weekday_rain_expectation * 5 + weekend_rain_expectation * 2
      = (5 * ((0.30 * 0) + (0.20 * 5) + (0.50 * 8))) + (2 * ((0.50 * 0) + (0.25 * 2) + (0.25 * 6))) : by rfl
  ... = 25 + 4 : by norm_num
  ... = 29 : by norm_num

end expected_total_rain_l370_370542


namespace tara_additional_stamps_l370_370820

def stamps_needed (current_stamps total_stamps : Nat) : Nat :=
  if total_stamps % 9 == 0 then 0 else 9 - (total_stamps % 9)

theorem tara_additional_stamps :
  stamps_needed 38 45 = 7 := by
  sorry

end tara_additional_stamps_l370_370820


namespace sin_210_l370_370562

theorem sin_210 : Real.sin (210 * Real.pi / 180) = -1/2 := by
  sorry

end sin_210_l370_370562


namespace correct_option_is_D_l370_370097

def is_linear_in_two_variables (eq : String) : Prop :=
  match eq with
  | "2x + 3y = z" => False
  | "4 / x + y = 5" => False
  | "1 / 2 * x ^ 2 + y = 0" => False
  | "y = 1 / 2 * (x + 8)" => True
  | _ => False

theorem correct_option_is_D : is_linear_in_two_variables "2x + 3y = z" = False ∧
                              is_linear_in_two_variables "4 / x + y = 5" = False ∧
                              is_linear_in_two_variables "1 / 2 * x ^ 2 + y = 0" = False ∧
                              is_linear_in_two_variables "y = 1 / 2 * (x + 8)" = True :=
by 
  repeat { split }; 
  simp [is_linear_in_two_variables]; 
  sorry

end correct_option_is_D_l370_370097


namespace all_stars_seating_l370_370723

theorem all_stars_seating (C R Y D : ℕ)
  (cubs : C = 4) (red_sox : R = 3) (yankees : Y = 2) (dodger : D = 1) :
  (4! * 4! * 3! * 2! * 1!) = 6912 :=
by sorry

end all_stars_seating_l370_370723


namespace determine_m_l370_370198

theorem determine_m (m : ℝ) 
  (f : ℝ → ℝ := λ x, (1 / 2) ^ (-x^2 + 2*m*x - m^2 - 1))
  (h_increase : ∀ x, x ≥ m → f x ≥ 2) :
  m = 2 := sorry

end determine_m_l370_370198


namespace min_n_minus_m_l370_370685

def f (x : ℝ) : ℝ := 1 - real.sqrt (1 - 2 * x)
def g (x : ℝ) : ℝ := real.log x

theorem min_n_minus_m : 
  ∀ (m n : ℝ), 
  m ≤ 1 / 2 → 
  0 < n → 
  f m = g n → 
  n - m = 1 :=
sorry

end min_n_minus_m_l370_370685


namespace distance_traveled_l370_370488

-- Definitions for the given problem conditions
def speed_boat_still_water : ℝ := 20 -- speed of boat in still water in km/hr
def rate_current : ℝ := 5           -- rate of current in km/hr
def time_minutes : ℝ := 27          -- time in minutes

-- Convert time from minutes to hours
def time_hours : ℝ := time_minutes / 60

-- Effective speed of the boat downstream
def effective_speed_downstream : ℝ := speed_boat_still_water + rate_current

-- Distance traveled downstream calculation
def distance_downstream : ℝ := effective_speed_downstream * time_hours

-- Proof statement
theorem distance_traveled (h1 : speed_boat_still_water = 20) (h2 : rate_current = 5) (h3 : time_minutes = 27) : distance_downstream = 11.25 :=
by 
  unfold distance_downstream effective_speed_downstream time_hours
  rw [h1, h2, h3]
  norm_num
  sorry

end distance_traveled_l370_370488


namespace max_interval_length_l370_370648

theorem max_interval_length (a b : ℝ) (h : ∀ x ∈ set.Icc a b, (1/2) * Real.sin x - (Real.sqrt 3 / 2) * Real.cos x ∈ set.Icc (-1/2) 1) : 
  b - a ≤ (4 * Real.pi / 3) :=
sorry

end max_interval_length_l370_370648


namespace triplet_D_sum_not_one_l370_370100

def triplet_sum_not_equal_to_one : Prop :=
  (1.2 + -0.2 + 0.0 ≠ 1)

theorem triplet_D_sum_not_one : triplet_sum_not_equal_to_one := 
  by
    sorry

end triplet_D_sum_not_one_l370_370100


namespace find_other_number_l370_370838

def gcd (x y : Nat) : Nat := Nat.gcd x y
def lcm (x y : Nat) : Nat := Nat.lcm x y

theorem find_other_number (b : Nat) :
  gcd 360 b = 36 ∧ lcm 360 b = 8820 → b = 882 := by
  sorry

end find_other_number_l370_370838


namespace find_angle_A_find_range_of_perimeter_l370_370714

variables {A B C a b c : ℝ}

-- Problem 1
theorem find_angle_A (h1: a ≠ 0) (h2: cos C ≠ 0) (h3: sin A ≠ 0)
                    (m : ℝ × ℝ := (a, 1 / 2))
                    (n : ℝ × ℝ := (cos C, c - 2 * b))
                    (h₁ : m.1 * n.1 + m.2 * n.2 = 0) :
                    A = π / 3 :=
sorry

-- Problem 2
theorem find_range_of_perimeter (h: a = 1) (h₁: π / 6 < B + π / 6) (h₂: B + π / 6 ≤ 5 * π / 6) :
       ∃ l, 2 < l ∧ l ≤ 3 :=
sorry

end find_angle_A_find_range_of_perimeter_l370_370714


namespace treasure_coins_l370_370444

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l370_370444


namespace sin_210_eq_neg_one_half_l370_370587

theorem sin_210_eq_neg_one_half :
  ∀ (θ : ℝ), 
  θ = 210 * (π / 180) → -- angle 210 degrees
  ∃ (refθ : ℝ), 
  refθ = 30 * (π / 180) ∧ -- reference angle 30 degrees
  sin refθ = 1 / 2 → -- sin of reference angle
  sin θ = -1 / 2 := 
by
  intros θ hθ refθ hrefθ hrefθ_sin -- introduce variables and hypotheses
  sorry

end sin_210_eq_neg_one_half_l370_370587


namespace line_circle_intersection_proof_l370_370220

variable {m : ℝ} {r : ℝ}

theorem line_circle_intersection_proof (h1 : ∀ m : ℝ, ∃ A B : ℝ × ℝ, (A ≠ B) ∧ (mx - y + 1 = 0) ∧ (x^2 + y^2 = r^2)) 
  (h2 : 1 < r ∧ r ≤ Real.sqrt 2) : ∃ m : ℝ, | ⟨0, 0⟩ - A | + | ⟨0, 0⟩ - B | ≥ | A - B | :=
sorry

end line_circle_intersection_proof_l370_370220


namespace decimal100_to_binary_l370_370598

theorem decimal100_to_binary :
  nat.to_digits 2 100 = [1, 1, 0, 0, 1, 0, 0] :=
by sorry

end decimal100_to_binary_l370_370598


namespace total_cost_after_rebate_l370_370796

theorem total_cost_after_rebate :
  let polo_shirts_cost := 3 * 26,
      necklaces_cost := 2 * 83,
      computer_game_cost := 1 * 90,
      rebate := 12,
      total_before_rebate := polo_shirts_cost + necklaces_cost + computer_game_cost,
      total_after_rebate := total_before_rebate - rebate in
  total_after_rebate = 322 :=
by
  sorry

end total_cost_after_rebate_l370_370796


namespace final_price_of_coat_is_correct_l370_370133

-- Define the conditions as constants
def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

-- Define the discounted amount calculation
def discount_amount : ℝ := original_price * discount_rate

-- Define the sale price after the discount
def sale_price : ℝ := original_price - discount_amount

-- Define the tax amount calculation on the sale price
def tax_amount : ℝ := sale_price * tax_rate

-- Define the total selling price
def total_selling_price : ℝ := sale_price + tax_amount

-- The theorem that needs to be proven
theorem final_price_of_coat_is_correct : total_selling_price = 96.6 :=
by
  sorry

end final_price_of_coat_is_correct_l370_370133


namespace increasing_interval_f_range_g_on_interval_l370_370695

open Real
open Set

def a (x : ℝ) : ℝ × ℝ := (sin x, cos x)
def b : ℝ × ℝ := (sqrt 3, -1)
def f (x : ℝ) : ℝ := (sqrt 3) * sin x - cos x
def g (x : ℝ) : ℝ := 2 * sin (x - π / 3)

theorem increasing_interval_f (k : ℤ) : 
  ∀ x, 2 * k * π - π / 3 ≤ x ∧ x ≤ 2 * k * π + 2 * π / 3 → strict_mono_on (f) [2 * k * π - π / 3, 2 * k * π + 2 * π / 3] :=
sorry

theorem range_g_on_interval : 
  ∀ x, 0 ≤ x ∧ x ≤ π → g x ∈ [√3, 2] :=
sorry

end increasing_interval_f_range_g_on_interval_l370_370695


namespace maximize_passengers_l370_370397

-- Define the functional relationship t
def t (n : ℕ) : ℕ := -2 * n + 24

-- Define the function for the number of passengers carried per day
def passengers_per_day (n : ℕ) : ℕ := t n * n * 110 * 2

-- Theorem to prove
theorem maximize_passengers : 
  ∃ n_max : ℕ, n_max = 6 ∧ passengers_per_day n_max = 15840 := 
by
  use 6
  split
  . refl
  . calc
    passengers_per_day 6
    = (-2 * 6 + 24) * 6 * 110 * 2 : rfl
    ... = 12 * 6 * 110 * 2 : by norm_num
    ... = 15840 : by norm_num

end maximize_passengers_l370_370397


namespace number_of_even_functions_is_one_l370_370604

def f1 (x : ℝ) : ℝ := x^2 - 2*x
def f2 (x : ℝ) : ℝ := (x + 1) * Real.sqrt ((1 - x) / (1 + x))
def f3 (x : ℝ) : ℝ := (x - 1)^2
def f4 (x : ℝ) : ℝ := Real.log10 (Real.sqrt (x^2 - 2))

theorem number_of_even_functions_is_one : 
  (∃ f ∈ set_of_even_functions, ∀ f ∈ {f1, f2, f3, f4}, f ∈ set_of_even_functions ∧ set_of_even_functions.card = 1) :=
by
  sorry

end number_of_even_functions_is_one_l370_370604


namespace q_form_l370_370336

theorem q_form (p q k : ℕ) (hp : nat.prime p) (hp_odd : p % 2 = 1)
               (hq : nat.prime q) (hq_fact : q ∣ 2^p - 1) :
               ∃ (k : ℕ), q = 2 * k * p + 1 :=
by
  sorry

end q_form_l370_370336


namespace tan_945_equals_1_l370_370177

noncomputable def tan_circular (x : ℝ) : ℝ := Real.tan x

theorem tan_945_equals_1 :
  tan_circular 945 = 1 := 
by
  sorry

end tan_945_equals_1_l370_370177


namespace minimum_percentage_increase_in_mean_replacing_with_primes_l370_370080

def mean (S : List ℤ) : ℚ :=
  (S.sum : ℚ) / S.length

noncomputable def percentage_increase (original new : ℚ) : ℚ :=
  ((new - original) / original) * 100

theorem minimum_percentage_increase_in_mean_replacing_with_primes :
  let F := [-4, -1, 0, 6, 9] 
  let G := [2, 3, 0, 6, 9] 
  percentage_increase (mean F) (mean G) = 100 :=
by {
  let F := [-4, -1, 0, 6, 9] 
  let G := [2, 3, 0, 6, 9] 
  sorry 
}

end minimum_percentage_increase_in_mean_replacing_with_primes_l370_370080


namespace time_to_eat_5_pounds_approx_48_minutes_l370_370351

noncomputable def mr_fat_rate : ℚ := 1 / 25
noncomputable def mr_thin_rate : ℚ := 1 / 35
noncomputable def mr_medium_rate : ℚ := 1 / 28
noncomputable def combined_rate : ℚ := mr_fat_rate + mr_thin_rate + mr_medium_rate
noncomputable def time_to_eat (rate : ℚ) (amount : ℚ) : ℚ := amount / rate

theorem time_to_eat_5_pounds_approx_48_minutes :
  (time_to_eat combined_rate 5).round = 48 := 
sorry

end time_to_eat_5_pounds_approx_48_minutes_l370_370351


namespace concyclic_ACOHI_l370_370337

theorem concyclic_ACOHI 
  (A B C O H I : Type) 
  [triangle_ABC : ∃ (A B C : Type), angle ABC = 60°]
  (circumcenter_O : is_circumcenter A B C O)
  (orthocenter_H : is_orthocenter A B C H)
  (incenter_I : is_incenter A B C I) :
  are_concyclic A C O H I :=
sorry

end concyclic_ACOHI_l370_370337


namespace sum_of_factors_32_l370_370217

theorem sum_of_factors_32 : ∑ i in (finset.filter (λ d, 32 % d = 0) (finset.range 33)), i = 63 :=
by
  sorry

end sum_of_factors_32_l370_370217


namespace sin_210_eq_neg_one_half_l370_370588

theorem sin_210_eq_neg_one_half :
  ∀ (θ : ℝ), 
  θ = 210 * (π / 180) → -- angle 210 degrees
  ∃ (refθ : ℝ), 
  refθ = 30 * (π / 180) ∧ -- reference angle 30 degrees
  sin refθ = 1 / 2 → -- sin of reference angle
  sin θ = -1 / 2 := 
by
  intros θ hθ refθ hrefθ hrefθ_sin -- introduce variables and hypotheses
  sorry

end sin_210_eq_neg_one_half_l370_370588


namespace area_of_region_l370_370193

theorem area_of_region : 
  (∃ x y : ℝ, |5 * x - 10| + |4 * y + 20| ≤ 10) →
  ∃ area : ℝ, 
  area = 10 :=
sorry

end area_of_region_l370_370193


namespace sum_of_answers_is_sixteen_l370_370863

def answers := {1, 2, 3, 4}

variables (q1 q2 q3 q4 q5 q6 q7 : ℕ)

def question_condition_1 := q1 ∈ answers ∧ q2 ∈ answers ∧ q3 ∈ answers ∧ q4 ∈ answers ∧ q5 ∈ answers ∧ q6 ∈ answers ∧ q7 ∈ answers

def question_condition_2 := (Finset.card (Finset.filter (λ q => q = 4) (Finset.mk [q1, q2, q3, q4, q5, q6, q7])) = 1)

def question_condition_3 := (Finset.card (Finset.filter (λ q => q ≠ 2 ∧ q ≠ 3) (Finset.mk [q1, q2, q3, q4, q5, q6, q7])) = 3)

def question_condition_4 := ((q5 + q6) / 2 ∈ ℤ) -- Assumed equivalent to getting an integer answer for the average.

def question_condition_5 := (q1 - q2 = -3) 

def question_condition_6 := (q1 + q7 = 2)

def question_condition_7 := (Finset.filter (λ q => q = 2) (Finset.mk [q1, q2, q3, q4, q5, q6, q7])).min' sorry = q5

def question_condition_8 := (Finset.card (Finset.filter (λ q => Finset.card (Finset.filter (λ r => r = q) (Finset.mk [q1, q2, q3, q4, q5, q6, q7])) = 1) (Finset.mk [q1, q2, q3, q4, q5, q6, q7])) = 4)

theorem sum_of_answers_is_sixteen :
  question_condition_1 q1 q2 q3 q4 q5 q6 q7 ∧
  question_condition_2 q1 q2 q3 q4 q5 q6 q7 ∧
  question_condition_3 q1 q2 q3 q4 q5 q6 q7 ∧
  question_condition_4 q1 q2 q3 q4 q5 q6 q7 ∧
  question_condition_5 q1 q2 q3 q4 q5 q6 q7 ∧
  question_condition_6 q1 q2 q3 q4 q5 q6 q7 ∧
  question_condition_7 q1 q2 q3 q4 q5 q6 q7 ∧
  question_condition_8 q1 q2 q3 q4 q5 q6 q7 →
  (q1 + q2 + q3 + q4 + q5 + q6 + q7 = 16) :=
by
  sorry

end sum_of_answers_is_sixteen_l370_370863


namespace problem_l370_370343

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log (1 + x) / log (a) + log (3 - x) / log (a)

theorem problem
  (x : ℝ) (a : ℝ)
  (h_cond1 : x > 0)
  (h_cond2 : a ≠ 1)
  (h_f1 : f a 1 = 2)
  (h_dom1 : x ∈ Ioc 0 3) :
  (a = 2)
  ∧ (∀ x, f 2 x = log ((1 + x) * (3 - x)) / log 2)
  ∧ (f 2 '' (Icc 0 (3 / 2)) = set.Icc (log 3 / log 2) 2) := sorry

end problem_l370_370343


namespace total_milk_consumed_l370_370293

theorem total_milk_consumed (regular_milk : ℝ) (soy_milk : ℝ) (H1 : regular_milk = 0.5) (H2: soy_milk = 0.1) :
    regular_milk + soy_milk = 0.6 :=
  by
  sorry

end total_milk_consumed_l370_370293


namespace diameter_of_field_l370_370618

noncomputable def rate : ℝ := 3
noncomputable def total_cost : ℝ := 395.84067435231395
noncomputable def pi : ℝ := Real.pi

theorem diameter_of_field :
  let circumference := total_cost / rate in
  let diameter := circumference / pi in
  diameter = 42 :=
by
  sorry

end diameter_of_field_l370_370618


namespace quadratic_equation_correct_l370_370476

theorem quadratic_equation_correct :
    (∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^2 = 5)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x y : ℝ, x + 2 * y = 0)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^2 + 1/x = 0)) ∧
    ¬(∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0) ↔ (∃ x : ℝ, x^3 + x^2 = 0)) :=
by
  sorry

end quadratic_equation_correct_l370_370476


namespace valid_sequences_length_15_l370_370602

-- Definitions based on conditions
def x : ℕ → ℕ
| 1       := 1
| n@(nat.succ (nat.succ _)) := y (n - 1)

def y : ℕ → ℕ
| 1       := 0
| 2       := 1
| (n@(nat.succ (nat.succ _))) := x (n - 2) + y (n - 2)

-- The main statement to verify
theorem valid_sequences_length_15 : (x 15) + (y 15) = 377 := by
sorry

end valid_sequences_length_15_l370_370602


namespace triangles_congruent_medians_l370_370168

-- Define a triangle in terms of its vertices
structure Triangle :=
(A B C : Point)

-- Define the midpoint function
def midpoint (P Q : Point) : Point := sorry

-- Define the function to draw medians
def median (T : Triangle) (M : Point) (V : Point) : Line := sorry

-- Define the congruence of triangles
def congruent (T1 T2 : Triangle) : Prop := sorry

-- Define the main theorem 
theorem triangles_congruent_medians (T : Triangle) :
  let D := midpoint T.A T.B
      E := midpoint T.B T.C
      F := midpoint T.C T.A
      M1 := median T T.A D
      M2 := median T T.B E
      M3 := median T T.C F
      T1 := Triangle.mk T.A D F
      T2 := Triangle.mk D T.B E
      T3 := Triangle.mk E T.C F
      T4 := Triangle.mk D E F
  in congruent T1 T2 ∧ congruent T2 T3 ∧ congruent T3 T4 ∧ congruent T1 T4 := sorry

end triangles_congruent_medians_l370_370168


namespace company_partition_l370_370346

-- Given:
-- 1. G is a graph representing the company, which is 3-indecomposable
-- 2. There are no four pairwise acquainted individuals in G

theorem company_partition (G : SimpleGraph α) 
  (hG3 : ¬(coloring G 3)) -- 3-indecomposable
  (no_four_clique : ∀ {s : Finset α}, s.card = 4 → ¬s.complete_subgraph G) -- No four pairwise acquainted individuals
: ∃ (G1 G2 : SimpleGraph α), 
    (G1 ∪ G2 = G) ∧
    (¬(coloring G1 2)) ∧ -- G1 is 2-indecomposable
    (∃ v ∈ V(G2), ∀ w ∈ V(G2), G2.adj v w). -- G2 is 1-indecomposable (complete subgraph)
Proof 
s := "sorry"

end company_partition_l370_370346


namespace find_sphere_radius_l370_370002

-- Definitions of conditions
def cone_radius_1 : ℝ := 72
def cone_radius_2 : ℝ := 28
def cone_radius_3 : ℝ := 28

def apex_angle_1 : ℝ := - (Real.pi / 3)
def apex_angle_2 : ℝ := 2 * (Real.pi / 3)
def apex_angle_3 : ℝ := 2 * (Real.pi / 3)

-- Definition of the radius of the sphere
noncomputable def sphere_radius : ℝ := (√3 + 1) / 2

-- Main theorem statement
theorem find_sphere_radius
  (r1 r2 r3 : ℝ)
  (a1 a2 a3 : ℝ)
  (h1 : r1 = cone_radius_1)
  (h2 : r2 = cone_radius_2)
  (h3 : r3 = cone_radius_3)
  (h4 : a1 = apex_angle_1)
  (h5 : a2 = apex_angle_2)
  (h6 : a3 = apex_angle_3) :
  ∃ R : ℝ, R = sphere_radius :=
sorry

end find_sphere_radius_l370_370002


namespace sin_210_eq_neg_one_half_l370_370585

theorem sin_210_eq_neg_one_half :
  ∀ (θ : ℝ), 
  θ = 210 * (π / 180) → -- angle 210 degrees
  ∃ (refθ : ℝ), 
  refθ = 30 * (π / 180) ∧ -- reference angle 30 degrees
  sin refθ = 1 / 2 → -- sin of reference angle
  sin θ = -1 / 2 := 
by
  intros θ hθ refθ hrefθ hrefθ_sin -- introduce variables and hypotheses
  sorry

end sin_210_eq_neg_one_half_l370_370585


namespace tasty_tuples_count_ge_2019_l370_370222

open BigOperators

def is_tasty {k n : ℕ} (a : ℕ → ℕ) :=
  ∃ S : Finset ℕ, 
    (S.card = k) ∧ 
    (∀ x : ℕ, x ∈ Finset.range k → a x ≤ x + 1) ∧
    (∃ x y : ℕ, x < y ∧ x ∈ Finset.range k ∧ y ∈ Finset.range k ∧ a x = a y) ∧
    (∀ i j : ℕ, i < j → a i ≤ a j)

def tasty_tuples_count (n : ℕ) : ℕ :=
  let k_range := Finset.range (n + 1) \ {0, 1}
  k_range.sum fun k => (Finset.univ.filter (λ a : ℕ → ℕ, is_tasty a)).card

theorem tasty_tuples_count_ge_2019 : tasty_tuples_count 8 > 2018 :=
  sorry

end tasty_tuples_count_ge_2019_l370_370222


namespace third_smallest_four_digit_in_pascals_triangle_l370_370467

theorem third_smallest_four_digit_in_pascals_triangle : 
  ∃ n k, (nat.choose n k) = 1002 ∧ (∀ k' < k, (nat.choose n k') < 1002) ∧ 
         (∃ n1 k1 n2 k2, (nat.choose n1 k1) = 1000 ∧ (nat.choose n2 k2) = 1001 ∧ 
                         (∀ k1' < k1, (nat.choose n1 k1') < 1000) ∧ 
                         (∀ k2' < k2, (nat.choose n2 k2') < 1001)) := 
by
  sorry

end third_smallest_four_digit_in_pascals_triangle_l370_370467


namespace animal_count_in_hollow_l370_370001

theorem animal_count_in_hollow (heads legs : ℕ) (animals_with_odd_legs animals_with_even_legs : ℕ) :
  heads = 18 →
  legs = 24 →
  (∀ n, n % 2 = 1 → animals_with_odd_legs * 2 = heads - 2 * n) →
  (∀ m, m % 2 = 0 → animals_with_even_legs * 1 = heads - m) →
  (animals_with_odd_legs + animals_with_even_legs = 10 ∨
   animals_with_odd_legs + animals_with_even_legs = 12 ∨
   animals_with_odd_legs + animals_with_even_legs = 14) :=
sorry

end animal_count_in_hollow_l370_370001


namespace convex_polygon_nonagon_l370_370983

theorem convex_polygon_nonagon (n : ℕ) (E I : ℝ) 
  (h1 : I = 3 * E + 180)
  (h2 : I + E = 180)
  (h3 : n * E = 360) : n = 9 :=
begin
  sorry
end

end convex_polygon_nonagon_l370_370983


namespace calculate_x_l370_370834

theorem calculate_x (x : ℝ) :
  let area_square := 9 * x^2,
      area_rectangle := 12 * x^2,
      area_triangle := 3 * x^2,
      total_area := area_square + area_rectangle + area_triangle
  in total_area = 1000 → x = (5 * Real.sqrt 15) / 3 :=
by
  intro h1
  simp [total_area, area_square, area_rectangle, area_triangle] at h1
  sorry

end calculate_x_l370_370834


namespace angle_RBC_10_degrees_l370_370734

noncomputable def compute_angle_RBC (angle_BRA angle_BAC angle_ABC : ℝ) : ℝ :=
  let angle_RBA := 180 - angle_BRA - angle_BAC
  angle_RBA - angle_ABC

theorem angle_RBC_10_degrees :
  ∀ (angle_BRA angle_BAC angle_ABC : ℝ), 
    angle_BRA = 72 → angle_BAC = 43 → angle_ABC = 55 → 
    compute_angle_RBC angle_BRA angle_BAC angle_ABC = 10 :=
by
  intros
  unfold compute_angle_RBC
  sorry

end angle_RBC_10_degrees_l370_370734


namespace distance_M_to_NF_l370_370952

theorem distance_M_to_NF (p : ℝ) (p_pos : 0 < p) :
  let focus := (1, 0)
  let directrix := -p / 2
  let slope := real.sqrt 3
  let parabola (x y : ℝ) := (y^2 = 2 * p * x)
  let line_through_focus (x y : ℝ) := (y = slope * (x - 1))
  let intersection_M (x y : ℝ) := parabola x y ∧ line_through_focus x y
  ∃ Mx My, intersection_M Mx My ∧ My > 0 →
  let directrix_N := λ x y : ℝ, (y = directrix)
  ∃ Nx Ny, directrix_N Nx Ny ∧ abs (Nx - 1) = 4 →
  let line_NF (x y : ℝ) := (real.sqrt 3 * x + y - real.sqrt 3 = 0)
  let distance (Mx My : ℝ) := abs (3 * real.sqrt 3 + 2 * real.sqrt 3 - real.sqrt 3) / real.sqrt 4 →
  distance Mx My = 2 * real.sqrt 3 :=
sorry

end distance_M_to_NF_l370_370952


namespace binomial_theorem_coeff_x_l370_370309

-- Define the binomial theorem
theorem binomial_theorem {a b : ℤ} (n : ℕ) :
  (a + b) ^ n = ∑ k in finset.range (n + 1), nat.choose n k * a ^ (n - k) * b ^ k := sorry

-- Define the problem coefficients and expression
def expansion_expr (x : ℤ) : ℤ :=
  (1 + x) * (x - 2 / x) ^ 3

-- State the problem of finding the coefficient of x term
theorem coeff_x (x : ℤ) : (1 + x) * (x - 2 / x) ^ 3 = -6 * x + other_terms :=
begin
  sorry
end

end binomial_theorem_coeff_x_l370_370309


namespace conversion_base_10_to_5_l370_370188

theorem conversion_base_10_to_5 : 
  (425 : ℕ) = 3 * 5^3 + 2 * 5^2 + 0 * 5^1 + 0 * 5^0 :=
by sorry

end conversion_base_10_to_5_l370_370188


namespace additional_tobacco_acres_l370_370516

def original_land : ℕ := 1350
def original_ratio_units : ℕ := 9
def new_ratio_units : ℕ := 9

def acres_per_unit := original_land / original_ratio_units

def tobacco_old := 2 * acres_per_unit
def tobacco_new := 5 * acres_per_unit

theorem additional_tobacco_acres :
  tobacco_new - tobacco_old = 450 := by
  sorry

end additional_tobacco_acres_l370_370516


namespace largest_value_l370_370594

theorem largest_value : 
  let x := 10^(-2024)
  in max (5 + x) (max (5 - x) (max (5 * x) (max (5 / x) (x / 5)))) = 5 / x :=
by 
  let x := 10^(-2024)
  sorry

end largest_value_l370_370594


namespace weight_of_new_person_l370_370911

noncomputable def new_person_weight (old_weight : ℝ) (avg_increase : ℝ) : ℝ :=
  let n := 8 -- number of persons
  let old_avg_weight := old_weight / n -- old average weight
  let new_avg_weight := old_avg_weight + avg_increase -- new average weight
  let total_increase := n * avg_increase -- total weight increase
  old_weight + total_increase

theorem weight_of_new_person
  (old_weight : ℝ)
  (avg_increase : ℝ)
  (weight_replaced : ℝ) :
  new_person_weight old_weight avg_increase = 93 :=
by
  have h := new_person_weight old_weight avg_increase
  rw [new_person_weight, weight_replaced, avg_increase]
  sorry

end weight_of_new_person_l370_370911


namespace max_squares_circle_radius_100_l370_370512

theorem max_squares_circle_radius_100 :
  let r := 100
  ∀ (C : ℝ × ℝ → Prop),
  (∃ (center : ℝ × ℝ),
    C = (λ (p : ℝ × ℝ), (p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2 = r ^ 2) ∧
    ¬ (∃ (x y : ℤ), (x, y) ∈ C) ∧
    ¬ (∃ (a b : ℤ), (a, b + 1) ∈ C ∨ (a + 1, b) ∈ C)) →
  (∃ (n : ℤ), n = 800) :=
by
  sorry

end max_squares_circle_radius_100_l370_370512


namespace average_distinct_s_l370_370665

theorem average_distinct_s (s : ℕ) (h: Polynomial) :
  (∃ p q : ℕ, p ≠ q ∧ Prime p ∧ Prime q ∧ p + q = 7 ∧ s = p * q) →
  s = 10 :=
by
  sorry

end average_distinct_s_l370_370665


namespace certain_number_any_number_l370_370290

theorem certain_number_any_number (k : ℕ) (n : ℕ) (h1 : 5^k - k^5 = 1) (h2 : 15^k ∣ n) : true :=
by
  sorry

end certain_number_any_number_l370_370290


namespace compute_a_sq_sub_b_sq_l370_370694

variables {a b : (ℝ × ℝ)}

-- Conditions
axiom a_nonzero : a ≠ (0, 0)
axiom b_nonzero : b ≠ (0, 0)
axiom a_add_b_eq_neg3_6 : a + b = (-3, 6)
axiom a_sub_b_eq_neg3_2 : a - b = (-3, 2)

-- Question and the correct answer
theorem compute_a_sq_sub_b_sq : (a.1^2 + a.2^2) - (b.1^2 + b.2^2) = 21 :=
by sorry

end compute_a_sq_sub_b_sq_l370_370694


namespace smallest_n_term_divisible_by_5_million_l370_370340

variable (a₁ a₂ : ℕ) (n : ℕ) 

def geometric_sequence (first : ℝ) (second : ℝ) (n : ℕ) := 
  let r := second / first
  in r^(n-1) * first 

theorem smallest_n_term_divisible_by_5_million
    (h1 : geometric_sequence (2 / 3) 10 n = 5 * 10^6) :
    n = 8 :=
sorry

end smallest_n_term_divisible_by_5_million_l370_370340
