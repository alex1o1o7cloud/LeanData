import Mathlib

namespace nine_distinct_numbers_product_l2293_229316

variable (a b c d e f g h i : ℕ)

theorem nine_distinct_numbers_product (ha : a = 12) (hb : b = 9) (hc : c = 2)
                                      (hd : d = 1) (he : e = 6) (hf : f = 36)
                                      (hg : g = 18) (hh : h = 4) (hi : i = 3) :
  (a * b * c = 216) ∧ (d * e * f = 216) ∧ (g * h * i = 216) ∧
  (a * d * g = 216) ∧ (b * e * h = 216) ∧ (c * f * i = 216) ∧
  (a * e * i = 216) ∧ (c * e * g = 216) :=
by
  sorry

end nine_distinct_numbers_product_l2293_229316


namespace M_intersect_N_eq_l2293_229323

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

-- Define what we need to prove
theorem M_intersect_N_eq : M ∩ N = {y | y ≥ 1} :=
by
  sorry

end M_intersect_N_eq_l2293_229323


namespace largest_angle_in_ratio_triangle_l2293_229310

theorem largest_angle_in_ratio_triangle (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 
  5 * (180 / (3 + 4 + 5)) = 75 := by
  sorry

end largest_angle_in_ratio_triangle_l2293_229310


namespace no_intersection_curves_l2293_229307

theorem no_intersection_curves (k : ℕ) (hn : k > 0) 
  (h_intersection : ∀ x y : ℝ, ¬(x^2 + y^2 = k^2 ∧ x * y = k)) : 
  k = 1 := 
sorry

end no_intersection_curves_l2293_229307


namespace area_of_triangle_MAB_l2293_229304

noncomputable def triangle_area (A B M : ℝ × ℝ) : ℝ :=
  0.5 * ((B.1 - A.1) * (M.2 - A.2) - (M.1 - A.1) * (B.2 - A.2))

theorem area_of_triangle_MAB :
  let C1 (p : ℝ × ℝ) := p.1^2 - p.2^2 = 2
  let C2 (p : ℝ × ℝ) := ∃ θ, p.1 = 2 + 2 * Real.cos θ ∧ p.2 = 2 * Real.sin θ
  let M := (3.0, 0.0)
  let A := (2, 2 * Real.sin (Real.pi / 6))
  let B := (2 * Real.sqrt 3, 2 * Real.sin (Real.pi / 6))
  triangle_area A B M = (3 * Real.sqrt 3 - 3) / 2 :=
by
  sorry

end area_of_triangle_MAB_l2293_229304


namespace probability_x_gt_3y_l2293_229349

noncomputable def rect_region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3020 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3010}

theorem probability_x_gt_3y : 
  (∫ p in rect_region, if p.1 > 3 * p.2 then 1 else (0:ℝ)) / 
  (∫ p in rect_region, (1:ℝ)) = 1007 / 6020 := sorry

end probability_x_gt_3y_l2293_229349


namespace raj_house_area_l2293_229377

theorem raj_house_area :
  let bedroom_area := 11 * 11
  let bedrooms_total := bedroom_area * 4
  let bathroom_area := 6 * 8
  let bathrooms_total := bathroom_area * 2
  let kitchen_area := 265
  let living_area := kitchen_area
  bedrooms_total + bathrooms_total + kitchen_area + living_area = 1110 :=
by
  -- Proof to be filled in
  sorry

end raj_house_area_l2293_229377


namespace max_area_dog_roam_l2293_229384

theorem max_area_dog_roam (r : ℝ) (s : ℝ) (half_s : ℝ) (midpoint : Prop) :
  r = 10 → s = 20 → half_s = s / 2 → midpoint → 
  r > half_s → 
  π * r^2 = 100 * π :=
by 
  intros hr hs h_half_s h_midpoint h_rope_length
  sorry

end max_area_dog_roam_l2293_229384


namespace find_k_l2293_229339

theorem find_k 
    (x y k : ℝ)
    (h1 : 1.5 * x + y = 20)
    (h2 : -4 * x + y = k)
    (hx : x = -6) :
    k = 53 :=
by
  sorry

end find_k_l2293_229339


namespace general_term_of_arithmetic_seq_l2293_229367

variable {a : ℕ → ℤ}

def arithmetic_seq (a : ℕ → ℤ) := ∃ d, ∀ n, a n = a 0 + n * d

theorem general_term_of_arithmetic_seq :
  arithmetic_seq a →
  a 2 = 9 →
  (∃ x y, (x ^ 2 - 16 * x + 60 = 0) ∧ (a 0 = x) ∧ (a 4 = y)) →
  ∀ n, a n = -n + 11 :=
by
  intros h_arith h_a2 h_root
  sorry

end general_term_of_arithmetic_seq_l2293_229367


namespace part_I_part_II_l2293_229302

open Set

-- Define the sets A and B
def A : Set ℝ := { x | 1 < x ∧ x < 2 }
def B (a : ℝ) : Set ℝ := { x | 2 * a - 1 < x ∧ x < 2 * a + 1 }

-- Part (Ⅰ): Given A ⊆ B, prove that 1/2 ≤ a ≤ 1
theorem part_I (a : ℝ) : A ⊆ B a → (1 / 2 ≤ a ∧ a ≤ 1) :=
by sorry

-- Part (Ⅱ): Given A ∩ B = ∅, prove that a ≥ 3/2 or a ≤ 0
theorem part_II (a : ℝ) : A ∩ B a = ∅ → (a ≥ 3 / 2 ∨ a ≤ 0) :=
by sorry

end part_I_part_II_l2293_229302


namespace sarah_initial_trucks_l2293_229398

theorem sarah_initial_trucks (trucks_given : ℕ) (trucks_left : ℕ) (initial_trucks : ℕ) :
  trucks_given = 13 → trucks_left = 38 → initial_trucks = trucks_left + trucks_given → initial_trucks = 51 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sarah_initial_trucks_l2293_229398


namespace interest_rate_C_l2293_229332

theorem interest_rate_C (P A G : ℝ) (R : ℝ) (t : ℝ := 3) (rate_A : ℝ := 0.10) :
  P = 4000 ∧ rate_A = 0.10 ∧ G = 180 →
  (P * rate_A * t + G) = P * (R / 100) * t →
  R = 11.5 :=
by
  intros h_cond h_eq
  -- proof to be filled, use the given conditions and equations
  sorry

end interest_rate_C_l2293_229332


namespace music_class_uncool_parents_l2293_229389

theorem music_class_uncool_parents:
  ∀ (total students coolDads coolMoms bothCool : ℕ),
  total = 40 →
  coolDads = 25 →
  coolMoms = 19 →
  bothCool = 8 →
  (total - (bothCool + (coolDads - bothCool) + (coolMoms - bothCool))) = 4 :=
by
  intros total coolDads coolMoms bothCool h_total h_dads h_moms h_both
  sorry

end music_class_uncool_parents_l2293_229389


namespace darren_total_tshirts_l2293_229313

def num_white_packs := 5
def num_white_tshirts_per_pack := 6
def num_blue_packs := 3
def num_blue_tshirts_per_pack := 9

def total_tshirts (wpacks : ℕ) (wtshirts_per_pack : ℕ) (bpacks : ℕ) (btshirts_per_pack : ℕ) : ℕ :=
  (wpacks * wtshirts_per_pack) + (bpacks * btshirts_per_pack)

theorem darren_total_tshirts : total_tshirts num_white_packs num_white_tshirts_per_pack num_blue_packs num_blue_tshirts_per_pack = 57 :=
by
  -- proof needed
  sorry

end darren_total_tshirts_l2293_229313


namespace growth_rate_l2293_229387

variable (x : ℝ)

def initial_investment : ℝ := 500
def expected_investment : ℝ := 720

theorem growth_rate (x : ℝ) (h : 500 * (1 + x)^2 = 720) : x = 0.2 :=
by
  sorry

end growth_rate_l2293_229387


namespace OHaraTriple_example_l2293_229361

def OHaraTriple (a b x : ℕ) : Prop :=
  (Nat.sqrt a + Nat.sqrt b = x)

theorem OHaraTriple_example : OHaraTriple 49 64 15 :=
by
  sorry

end OHaraTriple_example_l2293_229361


namespace min_value_geq_9div2_l2293_229324

noncomputable def min_value (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 12) : ℝ := 
  (x + y + z : ℝ) * ((1 : ℝ) / (x + y) + (1 : ℝ) / (x + z) + (1 : ℝ) / (y + z))

theorem min_value_geq_9div2 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 12) :
  min_value x y z hx hy hz h_sum ≥ 9 / 2 := 
sorry

end min_value_geq_9div2_l2293_229324


namespace intersection_A_B_l2293_229359

-- Definitions of sets A and B based on the given conditions
def A : Set ℕ := {4, 5, 6, 7}
def B : Set ℕ := {x | 3 ≤ x ∧ x < 6}

-- The theorem stating the proof problem
theorem intersection_A_B : A ∩ B = {4, 5} :=
by
  sorry

end intersection_A_B_l2293_229359


namespace cost_of_pumpkin_seeds_l2293_229353

theorem cost_of_pumpkin_seeds (P : ℝ)
    (h1 : ∃(P_tomato P_chili : ℝ), P_tomato = 1.5 ∧ P_chili = 0.9) 
    (h2 : 3 * P + 4 * 1.5 + 5 * 0.9 = 18) 
    : P = 2.5 :=
by sorry

end cost_of_pumpkin_seeds_l2293_229353


namespace add_decimals_l2293_229375

theorem add_decimals :
  5.623 + 4.76 = 10.383 :=
by sorry

end add_decimals_l2293_229375


namespace smallest_k_l2293_229393

theorem smallest_k (a b c d e k : ℕ) (h1 : a + 2 * b + 3 * c + 4 * d + 5 * e = k)
  (h2 : 5 * a = 4 * b) (h3 : 4 * b = 3 * c) (h4 : 3 * c = 2 * d) (h5 : 2 * d = e) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) : k = 522 :=
sorry

end smallest_k_l2293_229393


namespace set_A_is_2_3_l2293_229308

noncomputable def A : Set ℤ := { x : ℤ | 3 / (x - 1) > 1 }

theorem set_A_is_2_3 : A = {2, 3} :=
by
  sorry

end set_A_is_2_3_l2293_229308


namespace parametric_eqn_and_max_sum_l2293_229357

noncomputable def polar_eq (ρ θ : ℝ) := ρ^2 = 4 * ρ * (Real.cos θ + Real.sin θ) - 6

theorem parametric_eqn_and_max_sum (θ : ℝ):
  (∃ (x y : ℝ), (2 + Real.sqrt 2 * Real.cos θ, 2 + Real.sqrt 2 * Real.sin θ) = (x, y)) ∧
  (∃ (θ : ℝ), θ = Real.pi / 4 → (3, 3) = (3, 3) ∧ 6 = 6) :=
by {
  sorry
}

end parametric_eqn_and_max_sum_l2293_229357


namespace volume_rotation_l2293_229311

theorem volume_rotation
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (a b : ℝ)
  (h₁ : ∀ (x : ℝ), f x = x^3)
  (h₂ : ∀ (x : ℝ), g x = x^(1/2))
  (h₃ : a = 0)
  (h₄ : b = 1):
  ∫ x in a..b, π * ((g x)^2 - (f x)^2) = 5 * π / 14 :=
by
  sorry

end volume_rotation_l2293_229311


namespace find_principal_amount_l2293_229314

noncomputable def principal_amount (difference : ℝ) (rate : ℝ) : ℝ :=
  let ci := rate / 2
  let si := rate
  difference / (ci ^ 2 - 1 - si)

theorem find_principal_amount :
  principal_amount 4.25 0.10 = 1700 :=
by 
  sorry

end find_principal_amount_l2293_229314


namespace percentage_increase_in_consumption_l2293_229379

theorem percentage_increase_in_consumption 
  (T C : ℝ) 
  (h1 : 0.8 * T * C * (1 + P / 100) = 0.88 * T * C)
  : P = 10 := 
by 
  sorry

end percentage_increase_in_consumption_l2293_229379


namespace find_percentage_l2293_229372

theorem find_percentage (x p : ℝ) (h₀ : x = 780) (h₁ : 0.25 * x = (p / 100) * 1500 - 30) : p = 15 :=
by
  sorry

end find_percentage_l2293_229372


namespace amy_hours_per_week_school_year_l2293_229376

variable (hours_per_week_summer : ℕ)
variable (weeks_summer : ℕ)
variable (earnings_summer : ℕ)
variable (additional_earnings_needed : ℕ)
variable (weeks_school_year : ℕ)
variable (hourly_wage : ℝ := earnings_summer / (hours_per_week_summer * weeks_summer))

theorem amy_hours_per_week_school_year :
  hours_per_week_school_year = (additional_earnings_needed / hourly_wage) / weeks_school_year :=
by 
  -- Using the hourly wage and total income needed, calculate the hours.
  let total_hours_needed := additional_earnings_needed / hourly_wage
  have h1 : hours_per_week_school_year = total_hours_needed / weeks_school_year := sorry
  exact h1

end amy_hours_per_week_school_year_l2293_229376


namespace not_inequality_neg_l2293_229370

theorem not_inequality_neg (x y : ℝ) (h : x > y) : ¬ (-x > -y) :=
by {
  sorry
}

end not_inequality_neg_l2293_229370


namespace minimum_f_l2293_229347

def f (x y : ℤ) : ℤ := |5 * x^2 + 11 * x * y - 5 * y^2|

theorem minimum_f (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) : ∃ (m : ℤ), m = 5 ∧ ∀ (x y : ℤ), (x ≠ 0 ∨ y ≠ 0) → f x y ≥ m :=
by sorry

end minimum_f_l2293_229347


namespace general_term_formula_sum_of_sequence_l2293_229305

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℤ := n - 1

-- Conditions: a_5 = 4, a_3 + a_8 = 9
def cond1 : Prop := a 5 = 4
def cond2 : Prop := a 3 + a 8 = 9

theorem general_term_formula (h1 : cond1) (h2 : cond2) : ∀ n : ℕ, a n = n - 1 :=
by
  -- Place holder for proof
  sorry

-- Define the sequence {b_n}
def b (n : ℕ) : ℤ := 2 * a n - 1

-- Sum of the first n terms of b_n
def S (n : ℕ) : ℤ := n * (n - 2)

theorem sum_of_sequence (h1 : cond1) (h2 : cond2) : ∀ n : ℕ, (Finset.range (n + 1)).sum b = S n :=
by
  -- Place holder for proof
  sorry

end general_term_formula_sum_of_sequence_l2293_229305


namespace find_height_l2293_229322

namespace RightTriangleProblem

variables {x h : ℝ}

-- Given the conditions described in the problem
def right_triangle_proportional (a b c : ℝ) : Prop :=
  ∃ (x : ℝ), a = 3 * x ∧ b = 4 * x ∧ c = 5 * x

def hypotenuse (c : ℝ) : Prop := 
  c = 25

def leg (b : ℝ) : Prop :=
  b = 20

-- The theorem stating that the height h of the triangle is 12
theorem find_height (a b c : ℝ) (h : ℝ)
  (H1 : right_triangle_proportional a b c)
  (H2 : hypotenuse c)
  (H3 : leg b) :
  h = 12 :=
by
  sorry

end RightTriangleProblem

end find_height_l2293_229322


namespace calculate_a_over_b_l2293_229321

noncomputable def system_solution (x y a b : ℝ) : Prop :=
  (8 * x - 5 * y = a) ∧ (10 * y - 15 * x = b) ∧ (x ≠ 0) ∧ (y ≠ 0) ∧ (b ≠ 0)

theorem calculate_a_over_b (x y a b : ℝ) (h : system_solution x y a b) : a / b = 8 / 15 :=
by
  sorry

end calculate_a_over_b_l2293_229321


namespace sales_on_same_days_l2293_229312

-- Definitions representing the conditions
def bookstore_sales_days : List ℕ := [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
def toy_store_sales_days : List ℕ := [2, 9, 16, 23, 30]

-- Lean statement to prove the number of common sale days
theorem sales_on_same_days : (bookstore_sales_days ∩ toy_store_sales_days).length = 2 :=
by sorry

end sales_on_same_days_l2293_229312


namespace longest_leg_of_smallest_triangle_l2293_229360

-- Definitions based on conditions
def is306090Triangle (h : ℝ) (s : ℝ) (l : ℝ) : Prop :=
  s = h / 2 ∧ l = s * (Real.sqrt 3)

def chain_of_306090Triangles (H : ℝ) : Prop :=
  ∃ h1 s1 l1 h2 s2 l2 h3 s3 l3 h4 s4 l4,
    is306090Triangle h1 s1 l1 ∧
    is306090Triangle h2 s2 l2 ∧
    is306090Triangle h3 s3 l3 ∧
    is306090Triangle h4 s4 l4 ∧
    h1 = H ∧ l1 = h2 ∧ l2 = h3 ∧ l3 = h4

-- Main theorem
theorem longest_leg_of_smallest_triangle (H : ℝ) (h : ℝ) (l : ℝ) (H_cond : H = 16) 
  (h_cond : h = 9) :
  chain_of_306090Triangles H →
  ∃ h4 s4 l4, is306090Triangle h4 s4 l4 ∧ l = h4 →
  l = 9 := 
by
  sorry

end longest_leg_of_smallest_triangle_l2293_229360


namespace perpendicular_vectors_vector_sum_norm_min_value_f_l2293_229330

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (Real.cos (3*x/2), Real.sin (3*x/2))

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (Real.cos (x/2), -Real.sin (x/2))

noncomputable def f (x m : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 - 2 * m * Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2

theorem perpendicular_vectors (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  (a x).1 * (b x).1 + (a x).2 * (b x).2 = 0 ↔ x = Real.pi / 4 := sorry

theorem vector_sum_norm (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  Real.sqrt ((a x).1 + (b x).1)^2 + ((a x).2 + (b x).2)^2 ≥ 1 ↔ 0 ≤ x ∧ x ≤ Real.pi / 3 := sorry

theorem min_value_f (m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x m ≥ -2) ↔ m = Real.sqrt 2 / 2 := sorry

end perpendicular_vectors_vector_sum_norm_min_value_f_l2293_229330


namespace intersection_complement_eq_l2293_229391

noncomputable def U : Set Int := {-3, -2, -1, 0, 1, 2, 3}
noncomputable def A : Set Int := {-1, 0, 1, 2}
noncomputable def B : Set Int := {-3, 0, 2, 3}

-- Complement of B with respect to U
noncomputable def U_complement_B : Set Int := U \ B

-- The statement we need to prove
theorem intersection_complement_eq :
  A ∩ U_complement_B = {-1, 1} :=
by
  sorry

end intersection_complement_eq_l2293_229391


namespace solve_system_of_equations_l2293_229369

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x + 2 * y = 5 ∧ 3 * x - y = 1 ∧ x = 1 ∧ y = 2 := 
by
  sorry

end solve_system_of_equations_l2293_229369


namespace oranges_thrown_away_l2293_229328

theorem oranges_thrown_away (initial_oranges new_oranges current_oranges : ℕ) (x : ℕ) 
  (h1 : initial_oranges = 50)
  (h2 : new_oranges = 24)
  (h3 : current_oranges = 34) : 
  initial_oranges - x + new_oranges = current_oranges → x = 40 :=
by
  intros h
  rw [h1, h2, h3] at h
  sorry

end oranges_thrown_away_l2293_229328


namespace trent_walks_to_bus_stop_l2293_229306

theorem trent_walks_to_bus_stop (x : ℕ) (h1 : 2 * (x + 7) = 22) : x = 4 :=
sorry

end trent_walks_to_bus_stop_l2293_229306


namespace no_six_consecutive_nat_num_sum_eq_2015_l2293_229383

theorem no_six_consecutive_nat_num_sum_eq_2015 :
  ∀ (a b c d e f : ℕ),
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ e + 1 = f →
  a * b * c + d * e * f ≠ 2015 :=
by
  intros a b c d e f h
  sorry

end no_six_consecutive_nat_num_sum_eq_2015_l2293_229383


namespace min_value_of_f_l2293_229341

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3| + Real.exp x

theorem min_value_of_f :
  ∃ x ∈ Set.Icc (Real.exp 0) (Real.exp 3), f x = 6 - 2 * Real.log 2 :=
sorry

end min_value_of_f_l2293_229341


namespace jogger_distance_l2293_229363

theorem jogger_distance 
(speed_jogger : ℝ := 9)
(speed_train : ℝ := 45)
(train_length : ℕ := 120)
(time_to_pass : ℕ := 38)
(relative_speed_mps : ℝ := (speed_train - speed_jogger) * (1 / 3.6))
(distance_covered : ℝ := (relative_speed_mps * time_to_pass))
(d : ℝ := distance_covered - train_length) :
d = 260 := sorry

end jogger_distance_l2293_229363


namespace polynomial_expansion_l2293_229380

theorem polynomial_expansion (x : ℝ) : 
  (1 + x^2) * (1 - x^3) = 1 + x^2 - x^3 - x^5 :=
by sorry

end polynomial_expansion_l2293_229380


namespace partial_fraction_product_zero_l2293_229396

theorem partial_fraction_product_zero
  (A B C : ℚ)
  (partial_fraction_eq : ∀ x : ℚ,
    x^2 - 25 = A * (x + 3) * (x - 5) + B * (x - 3) * (x - 5) + C * (x - 3) * (x + 3))
  (fact_3 : C = 0)
  (fact_neg3 : B = 1/3)
  (fact_5 : A = 0) :
  A * B * C = 0 := 
sorry

end partial_fraction_product_zero_l2293_229396


namespace line_perpendicular_exists_k_line_intersects_circle_l2293_229326

theorem line_perpendicular_exists_k (k : ℝ) :
  ∃ k, (k * (1 / 2)) = -1 :=
sorry

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (k * x - y + 2 * k = 0) ∧ (x^2 + y^2 = 8) :=
sorry

end line_perpendicular_exists_k_line_intersects_circle_l2293_229326


namespace necessary_condition_l2293_229335

theorem necessary_condition (a b : ℝ) (h : b ≠ 0) (h2 : a > b) (h3 : b > 0) : (1 / a < 1 / b) :=
sorry

end necessary_condition_l2293_229335


namespace sellingPrice_is_459_l2293_229309

-- Definitions based on conditions
def costPrice : ℝ := 540
def markupPercentage : ℝ := 0.15
def discountPercentage : ℝ := 0.2608695652173913

-- Calculating the marked price based on the given conditions
def markedPrice (cp : ℝ) (markup : ℝ) : ℝ := cp + (markup * cp)

-- Calculating the discount amount based on the marked price and the discount percentage
def discount (mp : ℝ) (discountPct : ℝ) : ℝ := discountPct * mp

-- Calculating the selling price
def sellingPrice (mp : ℝ) (discountAmt : ℝ) : ℝ := mp - discountAmt

-- Stating the final proof problem
theorem sellingPrice_is_459 :
  sellingPrice (markedPrice costPrice markupPercentage) (discount (markedPrice costPrice markupPercentage) discountPercentage) = 459 :=
by
  sorry

end sellingPrice_is_459_l2293_229309


namespace find_some_number_l2293_229388

theorem find_some_number :
  ∃ (x : ℝ), abs (x - 0.004) < 0.0001 ∧ 9.237333333333334 = (69.28 * x) / 0.03 := by
  sorry

end find_some_number_l2293_229388


namespace trapezoid_fraction_l2293_229355

theorem trapezoid_fraction 
  (shorter_base longer_base side_length : ℝ)
  (angle_adjacent : ℝ)
  (h1 : shorter_base = 120)
  (h2 : longer_base = 180)
  (h3 : side_length = 130)
  (h4 : angle_adjacent = 60) :
  ∃ fraction : ℝ, fraction = 1 / 2 :=
by
  sorry

end trapezoid_fraction_l2293_229355


namespace sawyer_joined_coaching_l2293_229343

variable (daily_fees total_fees : ℕ)
variable (year_not_leap : Prop)
variable (discontinue_day : ℕ)

theorem sawyer_joined_coaching :
  daily_fees = 39 → 
  total_fees = 11895 → 
  year_not_leap → 
  discontinue_day = 307 → 
  ∃ start_day, start_day = 30 := 
by
  intros h_daily_fees h_total_fees h_year_not_leap h_discontinue_day
  sorry

end sawyer_joined_coaching_l2293_229343


namespace solve_logarithmic_equation_l2293_229325

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem solve_logarithmic_equation (x : ℝ) (h_pos : x > 0) :
  log_base 8 x + log_base 4 (x^2) + log_base 2 (x^3) = 15 ↔ x = 2 ^ (45 / 13) :=
by
  have h1 : log_base 8 x = (1 / 3) * log_base 2 x :=
    by { sorry }
  have h2 : log_base 4 (x^2) = log_base 2 x :=
    by { sorry }
  have h3 : log_base 2 (x^3) = 3 * log_base 2 x :=
    by { sorry }
  have h4 : (1 / 3) * log_base 2 x + log_base 2 x + 3 * log_base 2 x = 15 ↔ log_base 2 x = 45 / 13 :=
    by { sorry }
  exact sorry

end solve_logarithmic_equation_l2293_229325


namespace find_c_l2293_229300

noncomputable def cubic_function (x : ℝ) (c : ℝ) : ℝ :=
  x^3 - 3 * x + c

theorem find_c (c : ℝ) :
  (∃ x₁ x₂ : ℝ, cubic_function x₁ c = 0 ∧ cubic_function x₂ c = 0 ∧ x₁ ≠ x₂) →
  (c = -2 ∨ c = 2) :=
by
  sorry

end find_c_l2293_229300


namespace problem_statement_l2293_229366

-- Define the statement for positive integers m and n
def div_equiv (m n : ℕ) : Prop :=
  19 ∣ (11 * m + 2 * n) ↔ 19 ∣ (18 * m + 5 * n)

-- The final theorem statement
theorem problem_statement (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : div_equiv m n :=
by
  sorry

end problem_statement_l2293_229366


namespace solve_system_of_equations_l2293_229344

theorem solve_system_of_equations 
  (a1 a2 a3 a4 : ℝ) (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4)
  (x1 x2 x3 x4 : ℝ)
  (h1 : |a1 - a1| * x1 + |a1 - a2| * x2 + |a1 - a3| * x3 + |a1 - a4| * x4 = 1)
  (h2 : |a2 - a1| * x1 + |a2 - a2| * x2 + |a2 - a3| * x3 + |a2 - a4| * x4 = 1)
  (h3 : |a3 - a1| * x1 + |a3 - a2| * x2 + |a3 - a3| * x3 + |a3 - a4| * x4 = 1)
  (h4 : |a4 - a1| * x1 + |a4 - a2| * x2 + |a4 - a3| * x3 + |a4 - a4| * x4 = 1) :
  x1 = 1 / (a1 - a4) ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 1 / (a1 - a4) :=
sorry

end solve_system_of_equations_l2293_229344


namespace inequality_solution_sum_of_m_and_2n_l2293_229315

-- Define the function f(x) = |x - a|
def f (x a : ℝ) : ℝ := abs (x - a)

-- Part (1): The inequality problem for a = 2
theorem inequality_solution (x : ℝ) :
  f x 2 ≥ 4 - abs (x - 1) → x ≤ 2 / 3 := sorry

-- Part (2): Given conditions with solution set [0, 2] and condition on m and n
theorem sum_of_m_and_2n (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : ∀ x, f x 1 ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) (h₄ : 1 / m + 1 / (2 * n) = 1) :
  m + 2 * n ≥ 4 := sorry

end inequality_solution_sum_of_m_and_2n_l2293_229315


namespace conference_session_time_l2293_229392

def conference_duration_hours : ℕ := 8
def conference_duration_minutes : ℕ := 45
def break_time : ℕ := 30

theorem conference_session_time :
  (conference_duration_hours * 60 + conference_duration_minutes) - break_time = 495 :=
by sorry

end conference_session_time_l2293_229392


namespace inequality_subtract_l2293_229301

-- Definitions of the main variables and conditions
variables {a b : ℝ}
-- Condition that should hold
axiom h : a > b

-- Expected conclusion
theorem inequality_subtract : a - 1 > b - 2 :=
by
  sorry

end inequality_subtract_l2293_229301


namespace cuboid_height_l2293_229386

-- Define the necessary constants
def width : ℕ := 30
def length : ℕ := 22
def sum_edges : ℕ := 224

-- Theorem stating the height of the cuboid
theorem cuboid_height (h : ℕ) : 4 * length + 4 * width + 4 * h = sum_edges → h = 4 := by
  sorry

end cuboid_height_l2293_229386


namespace sum_coordinates_D_is_13_l2293_229364

theorem sum_coordinates_D_is_13 
  (A B C D : ℝ × ℝ) 
  (hA : A = (4, 8))
  (hB : B = (2, 2))
  (hC : C = (6, 4))
  (hD : D = (8, 5))
  (h_mid1 : (A.1 + B.1) / 2 = 3 ∧ (A.2 + B.2) / 2 = 5)
  (h_mid2 : (B.1 + C.1) / 2 = 4 ∧ (B.2 + C.2) / 2 = 3)
  (h_mid3 : (C.1 + D.1) / 2 = 7 ∧ (C.2 + D.2) / 2 = 4.5)
  (h_mid4 : (D.1 + A.1) / 2 = 6 ∧ (D.2 + A.2) / 2 = 6.5)
  (h_square : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (3, 5) ∧
               ((B.1 + C.1) / 2, (B.2 + C.2) / 2) = (4, 3) ∧
               ((C.1 + D.1) / 2, (C.2 + D.2) / 2) = (7, 4.5) ∧
               ((D.1 + A.1) / 2, (D.2 + A.2) / 2) = (6, 6.5))
  : (8 + 5) = 13 :=
by
  sorry

end sum_coordinates_D_is_13_l2293_229364


namespace solve_abs_eq_l2293_229350

theorem solve_abs_eq : ∀ x : ℚ, (|2 * x + 6| = 3 * x + 9) ↔ (x = -3) := by
  intros x
  sorry

end solve_abs_eq_l2293_229350


namespace R_depends_on_d_and_n_l2293_229342

variable (n a d : ℕ)

noncomputable def s1 : ℕ := (n * (2 * a + (n - 1) * d)) / 2
noncomputable def s2 : ℕ := (2 * n * (2 * a + (2 * n - 1) * d)) / 2
noncomputable def s3 : ℕ := (3 * n * (2 * a + (3 * n - 1) * d)) / 2
noncomputable def R : ℕ := s3 n a d - s2 n a d - s1 n a d

theorem R_depends_on_d_and_n : R n a d = 2 * d * n^2 :=
by
  sorry

end R_depends_on_d_and_n_l2293_229342


namespace max_parts_three_planes_divide_space_l2293_229365

-- Define the conditions given in the problem.
-- Condition 1: A plane divides the space into two parts.
def plane_divides_space (n : ℕ) : ℕ := 2

-- Condition 2: Two planes can divide the space into either three or four parts.
def two_planes_divide_space (n : ℕ) : ℕ := if n = 2 then 3 else 4

-- Condition 3: Three planes can divide the space into four, six, seven, or eight parts.
def three_planes_divide_space (n : ℕ) : ℕ := if n = 4 then 8 else sorry

-- The statement to be proved.
theorem max_parts_three_planes_divide_space : 
  ∃ n, three_planes_divide_space n = 8 := by
  use 4
  sorry

end max_parts_three_planes_divide_space_l2293_229365


namespace seating_arrangements_l2293_229327

theorem seating_arrangements (n_seats : ℕ) (n_people : ℕ) (n_adj_empty : ℕ) (h1 : n_seats = 6) 
    (h2 : n_people = 3) (h3 : n_adj_empty = 2) : 
    ∃ arrangements : ℕ, arrangements = 48 := 
by
  sorry

end seating_arrangements_l2293_229327


namespace paul_homework_average_l2293_229336

def hoursOnWeeknights : ℕ := 2 * 5
def hoursOnWeekend : ℕ := 5
def totalHomework : ℕ := hoursOnWeeknights + hoursOnWeekend
def practiceNights : ℕ := 2
def daysAvailable : ℕ := 7 - practiceNights
def averageHomeworkPerNight : ℕ := totalHomework / daysAvailable

theorem paul_homework_average :
  averageHomeworkPerNight = 3 := 
by
  -- sorry because we skip the proof
  sorry

end paul_homework_average_l2293_229336


namespace Tameka_sold_40_boxes_on_Friday_l2293_229337

noncomputable def TamekaSalesOnFriday (F : ℕ) : Prop :=
  let SaturdaySales := 2 * F - 10
  let SundaySales := (2 * F - 10) / 2
  F + SaturdaySales + SundaySales = 145

theorem Tameka_sold_40_boxes_on_Friday : ∃ F : ℕ, TamekaSalesOnFriday F ∧ F = 40 := 
by 
  sorry

end Tameka_sold_40_boxes_on_Friday_l2293_229337


namespace geometric_series_sum_l2293_229382

theorem geometric_series_sum (a r : ℝ) (h : |r| < 1) (h_a : a = 2 / 3) (h_r : r = 2 / 3) :
  ∑' i : ℕ, (a * r^i) = 2 :=
by
  sorry

end geometric_series_sum_l2293_229382


namespace sum_of_xyz_l2293_229352

theorem sum_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : (x + y + z)^3 - x^3 - y^3 - z^3 = 504) : x + y + z = 9 :=
by {
  sorry
}

end sum_of_xyz_l2293_229352


namespace find_number_l2293_229345

variable (a n : ℝ)

theorem find_number (h1: 2 * a = 3 * n) (h2: a * n ≠ 0) (h3: (a / 3) / (n / 2) = 1) : 
  n = 2 * a / 3 :=
sorry

end find_number_l2293_229345


namespace general_formula_an_l2293_229331

theorem general_formula_an {a : ℕ → ℝ} (S : ℕ → ℝ) (d : ℝ) (hS : ∀ n, S n = (n / 2) * (a 1 + a n)) (hd : d = a 2 - a 1) : 
  ∀ n, a n = a 1 + (n - 1) * d :=
sorry

end general_formula_an_l2293_229331


namespace mutually_exclusive_not_opposed_l2293_229374

-- Define the types for cards and people
inductive Card
| red : Card
| white : Card
| black : Card

inductive Person
| A : Person
| B : Person
| C : Person

-- Define the event that a person receives a specific card
def receives (p : Person) (c : Card) : Prop := sorry

-- Conditions
axiom A_receives_red : receives Person.A Card.red → ¬ receives Person.B Card.red
axiom B_receives_red : receives Person.B Card.red → ¬ receives Person.A Card.red

-- The proof problem statement
theorem mutually_exclusive_not_opposed :
  (receives Person.A Card.red → ¬ receives Person.B Card.red) ∧
  (¬(receives Person.A Card.red ∧ receives Person.B Card.red)) ∧
  (¬∀ p : Person, receives p Card.red) :=
sorry

end mutually_exclusive_not_opposed_l2293_229374


namespace find_b_l2293_229346

open Real

noncomputable def triangle_b (a b c : ℝ) (A B C : ℝ) (sin_A sin_B : ℝ) (area : ℝ) : Prop :=
  B < π / 2 ∧
  sin_B = sqrt 7 / 4 ∧
  area = 5 * sqrt 7 / 4 ∧
  sin_A / sin_B = 5 * c / (2 * b) ∧
  a = 5 / 2 * c ∧
  area = 1 / 2 * a * c * sin_B

theorem find_b (a b c : ℝ) (A B C : ℝ) (sin_A sin_B : ℝ) (area : ℝ) :
  triangle_b a b c A B C sin_A sin_B area → b = sqrt 14 := by
  sorry

end find_b_l2293_229346


namespace total_pieces_of_clothing_l2293_229318

-- Define Kaleb's conditions
def pieces_in_one_load : ℕ := 19
def num_equal_loads : ℕ := 5
def pieces_per_load : ℕ := 4

-- The total pieces of clothing Kaleb has
theorem total_pieces_of_clothing : pieces_in_one_load + num_equal_loads * pieces_per_load = 39 :=
by
  sorry

end total_pieces_of_clothing_l2293_229318


namespace ratio_A_to_B_investment_l2293_229378

variable (A B C : Type) [Field A] [Field B] [Field C]
variable (investA investB investC profit total_profit : A) 

-- Conditions
axiom A_invests_some_times_as_B : ∃ n : A, investA = n * investB
axiom B_invests_two_thirds_of_C : investB = (2/3) * investC
axiom total_profit_statement : total_profit = 3300
axiom B_share_statement : profit = 600

-- Theorem: Ratio of A's investment to B's investment is 3:1
theorem ratio_A_to_B_investment : ∃ n : A, investA = 3 * investB :=
sorry

end ratio_A_to_B_investment_l2293_229378


namespace count_p_values_l2293_229334

theorem count_p_values (p : ℤ) (n : ℝ) :
  (n = 16 * 10^(-p)) →
  (-4 < p ∧ p < 4) →
  ∃ m, p ∈ m ∧ (m.count = 3 ∧ m = [-2, 0, 2]) :=
by 
  sorry

end count_p_values_l2293_229334


namespace simplify_expression_l2293_229348

theorem simplify_expression :
  (360 / 24) * (10 / 240) * (6 / 3) * (9 / 18) = 5 / 8 := by
  sorry

end simplify_expression_l2293_229348


namespace roots_calculation_l2293_229362

theorem roots_calculation (c d : ℝ) (h : c^2 - 5*c + 6 = 0) (h' : d^2 - 5*d + 6 = 0) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 := by
  sorry

end roots_calculation_l2293_229362


namespace abs_diff_l2293_229397

theorem abs_diff (a b : ℝ) (h_ab : a < b) (h_a : abs a = 6) (h_b : abs b = 3) :
  a - b = -9 ∨ a - b = 9 :=
by
  sorry

end abs_diff_l2293_229397


namespace contrapositive_x_squared_l2293_229371

theorem contrapositive_x_squared :
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1) := 
sorry

end contrapositive_x_squared_l2293_229371


namespace function_satisfies_conditions_l2293_229329

-- Define the functional equation condition
def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + x * y) = f x * f (y + 1)

-- Lean statement for the proof problem
theorem function_satisfies_conditions (f : ℝ → ℝ) (h : functional_eq f) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) ∨ (∀ x : ℝ, f x = x) :=
sorry

end function_satisfies_conditions_l2293_229329


namespace find_f_neg1_l2293_229333

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def f : ℝ → ℝ
| x => if 0 < x then x^2 + 2 else if x = 0 then 2 else -(x^2 + 2)

axiom odd_f : is_odd_function f

theorem find_f_neg1 : f (-1) = -3 := by
  sorry

end find_f_neg1_l2293_229333


namespace am_gm_inequality_l2293_229303

theorem am_gm_inequality (a1 a2 a3 : ℝ) (h₀ : 0 < a1) (h₁ : 0 < a2) (h₂ : 0 < a3) (h₃ : a1 + a2 + a3 = 1) : 
  1 / a1 + 1 / a2 + 1 / a3 ≥ 9 :=
by
  sorry

end am_gm_inequality_l2293_229303


namespace fixed_points_l2293_229340

noncomputable def f (x : ℝ) : ℝ := x^2 - x - 3

theorem fixed_points : { x : ℝ | f x = x } = { -1, 3 } :=
by
  sorry

end fixed_points_l2293_229340


namespace speed_of_jogger_l2293_229358

noncomputable def jogger_speed_problem (jogger_distance_ahead train_length train_speed_kmh time_to_pass : ℕ) :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := jogger_distance_ahead + train_length
  let relative_speed := total_distance / time_to_pass
  let jogger_speed_ms := train_speed_ms - relative_speed
  let jogger_speed_kmh := jogger_speed_ms * 3600 / 1000
  jogger_speed_kmh

theorem speed_of_jogger :
  jogger_speed_problem 240 210 45 45 = 9 :=
by
  sorry

end speed_of_jogger_l2293_229358


namespace arithmetic_sequence_a12_l2293_229351

theorem arithmetic_sequence_a12 (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 7 + a 9 = 16) (h2 : a 4 = 1) 
  (h3 : ∀ n, a (n + 1) = a n + d) : a 12 = 15 := 
by {
  -- Proof steps would go here
  sorry
}

end arithmetic_sequence_a12_l2293_229351


namespace elyse_passing_threshold_l2293_229317

def total_questions : ℕ := 90
def programming_questions : ℕ := 20
def database_questions : ℕ := 35
def networking_questions : ℕ := 35
def programming_correct_rate : ℝ := 0.8
def database_correct_rate : ℝ := 0.5
def networking_correct_rate : ℝ := 0.7
def passing_percentage : ℝ := 0.65

theorem elyse_passing_threshold :
  let programming_correct := programming_correct_rate * programming_questions
  let database_correct := database_correct_rate * database_questions
  let networking_correct := networking_correct_rate * networking_questions
  let total_correct := programming_correct + database_correct + networking_correct
  let required_to_pass := passing_percentage * total_questions
  total_correct = required_to_pass → 0 = 0 :=
by
  intro _h
  sorry

end elyse_passing_threshold_l2293_229317


namespace totalWeightAlF3_is_correct_l2293_229395

-- Define the atomic weights of Aluminum and Fluorine
def atomicWeightAl : ℝ := 26.98
def atomicWeightF : ℝ := 19.00

-- Define the number of atoms of Fluorine in Aluminum Fluoride (AlF3)
def numFluorineAtoms : ℕ := 3

-- Define the number of moles of Aluminum Fluoride
def numMolesAlF3 : ℕ := 7

-- Calculate the molecular weight of Aluminum Fluoride (AlF3)
noncomputable def molecularWeightAlF3 : ℝ :=
  atomicWeightAl + (numFluorineAtoms * atomicWeightF)

-- Calculate the total weight of the given moles of AlF3
noncomputable def totalWeight : ℝ :=
  molecularWeightAlF3 * numMolesAlF3

-- Theorem stating the total weight of 7 moles of AlF3
theorem totalWeightAlF3_is_correct : totalWeight = 587.86 := sorry

end totalWeightAlF3_is_correct_l2293_229395


namespace kendra_and_tony_keep_two_each_l2293_229368

-- Define the conditions
def kendra_packs : Nat := 4
def tony_packs : Nat := 2
def pens_per_pack : Nat := 3
def pens_given_to_friends : Nat := 14

-- Define the total pens each has
def kendra_pens : Nat := kendra_packs * pens_per_pack
def tony_pens : Nat := tony_packs * pens_per_pack

-- Define the total pens
def total_pens : Nat := kendra_pens + tony_pens

-- Define the pens left after distribution
def pens_left : Nat := total_pens - pens_given_to_friends

-- Define the number of pens each keeps
def pens_each_kept : Nat := pens_left / 2

-- Prove the final statement
theorem kendra_and_tony_keep_two_each :
  pens_each_kept = 2 :=
by
  sorry

end kendra_and_tony_keep_two_each_l2293_229368


namespace percentage_increase_of_bill_l2293_229385

theorem percentage_increase_of_bill 
  (original_bill : ℝ) 
  (increased_bill : ℝ)
  (h1 : original_bill = 60)
  (h2 : increased_bill = 78) : 
  ((increased_bill - original_bill) / original_bill * 100) = 30 := 
by 
  rw [h1, h2]
  -- The following steps show the intended logic:
  -- calc 
  --   [(78 - 60) / 60 * 100]
  --   = [(18) / 60 * 100]
  --   = [0.3 * 100]
  --   = 30
  sorry

end percentage_increase_of_bill_l2293_229385


namespace x_cubed_lt_one_of_x_lt_one_abs_x_lt_one_of_x_lt_one_l2293_229319

variable {x : ℝ}

theorem x_cubed_lt_one_of_x_lt_one (hx : x < 1) : x^3 < 1 :=
sorry

theorem abs_x_lt_one_of_x_lt_one (hx : x < 1) : |x| < 1 :=
sorry

end x_cubed_lt_one_of_x_lt_one_abs_x_lt_one_of_x_lt_one_l2293_229319


namespace planting_trees_equation_l2293_229390

theorem planting_trees_equation (x : ℝ) (h1 : x > 0) : 
  20 / x - 20 / ((1 + 0.1) * x) = 4 :=
sorry

end planting_trees_equation_l2293_229390


namespace correct_system_of_equations_l2293_229320

theorem correct_system_of_equations
  (x y : ℝ)
  (h1 : x + (1 / 2) * y = 50)
  (h2 : y + (2 / 3) * x = 50) :
  (x + (1 / 2) * y = 50) ∧ (y + (2 / 3) * x = 50) :=
by
  exact ⟨h1, h2⟩

end correct_system_of_equations_l2293_229320


namespace forgotten_code_possibilities_l2293_229338

theorem forgotten_code_possibilities:
  let digits_set := {d | ∀ n:ℕ, 0≤n ∧ n≤9 → n≠0 → 
                     (n + 4 + 4 + last_digit ≡ 0 [MOD 3]) ∨ 
                     (n + 7 + 7 + last_digit ≡ 0 [MOD 3]) ∨
                     (n + 4 + 7 + last_digit ≡ 0 [MOD 3]) ∨
                     (n + 7 + 4 + last_digit ≡ 0 [MOD 3])
                    }
  let valid_first_digits := {1, 2, 4, 5, 7, 8}
  let total_combinations := 4 * 3 + 4 * 3 -- middle combinations * valid first digit combinations
  total_combinations = 24 ∧ digits_set = valid_first_digits := by
  sorry

end forgotten_code_possibilities_l2293_229338


namespace metropolis_hospital_babies_l2293_229373

theorem metropolis_hospital_babies 
    (a b d : ℕ) 
    (h1 : a = 3 * b) 
    (h2 : b = 2 * d) 
    (h3 : 2 * a + 3 * b + 5 * d = 1200) : 
    5 * d = 260 := 
sorry

end metropolis_hospital_babies_l2293_229373


namespace total_revenue_calculation_l2293_229394

-- Define the total number of etchings sold
def total_etchings : ℕ := 16

-- Define the number of etchings sold at $35 each
def etchings_sold_35 : ℕ := 9

-- Define the price per etching sold at $35
def price_per_etching_35 : ℕ := 35

-- Define the price per etching sold at $45
def price_per_etching_45 : ℕ := 45

-- Define the total revenue calculation
def total_revenue : ℕ :=
  let revenue_35 := etchings_sold_35 * price_per_etching_35
  let etchings_sold_45 := total_etchings - etchings_sold_35
  let revenue_45 := etchings_sold_45 * price_per_etching_45
  revenue_35 + revenue_45

-- Theorem stating the total revenue is $630
theorem total_revenue_calculation : total_revenue = 630 := by
  sorry

end total_revenue_calculation_l2293_229394


namespace initial_bacteria_count_l2293_229381

theorem initial_bacteria_count :
  ∀ (n : ℕ), (n * 5^8 = 1953125) → n = 5 :=
by
  intro n
  intro h
  sorry

end initial_bacteria_count_l2293_229381


namespace solutions_eq_l2293_229399

theorem solutions_eq :
  { (a, b, c) : ℕ × ℕ × ℕ | a * b + b * c + c * a = 2 * (a + b + c) } =
  { (2, 2, 2),
    (1, 2, 4), (1, 4, 2), 
    (2, 1, 4), (2, 4, 1),
    (4, 1, 2), (4, 2, 1) } :=
by sorry

end solutions_eq_l2293_229399


namespace no_n_satisfies_l2293_229354

def sum_first_n_terms_arith_seq (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem no_n_satisfies (n : ℕ) (h_n : n ≠ 0) :
  let s1 := sum_first_n_terms_arith_seq 5 6 n
  let s2 := sum_first_n_terms_arith_seq 12 4 n
  (s1 * s2 = 24 * n^2) → False :=
by
  sorry

end no_n_satisfies_l2293_229354


namespace Martha_cards_l2293_229356

theorem Martha_cards :
  let initial_cards := 76.0
  let given_away_cards := 3.0
  initial_cards - given_away_cards = 73.0 :=
by 
  let initial_cards := 76.0
  let given_away_cards := 3.0
  have h : initial_cards - given_away_cards = 73.0 := by sorry
  exact h

end Martha_cards_l2293_229356
