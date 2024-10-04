import Mathlib

namespace sqrt_factorial_product_l248_248114

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248114


namespace _l248_248719

-- Definitions for the geometric entities
variables (A B C D M P : Type)
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited M] [inhabited P]

-- Condition definitions
variable (triangle_ABC : A → B → C → Prop)
variable (angle_bisector : A → B → C → D → Prop)
variable (midpoint : D → M → Prop)
variable (intersection : A → B → C → M → P → Prop)
variable (AB_length : A → B → ℝ := 15)
variable (AC_length : A → C → ℝ := 8)

noncomputable def problem_statement : Prop :=
  ∀ A B C D M P : Type,
    triangle_ABC A B C →
    angle_bisector A B C D →
    midpoint D M →
    intersection A B C M P →
    (∃ (m n : ℕ), (nat.coprime m n) ∧ ((BP_length B P) / (PA_length P A) = 23/8) ∧ (m + n = 31))

noncomputable def theorem : problem_statement A B C D M P := sorry

end _l248_248719


namespace sqrt_factorial_equality_l248_248159

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l248_248159


namespace xiao_ning_min_cooking_time_l248_248847

def min_cooking_time (washing_pot : ℕ) (washing_vegetables : ℕ) (preparing_noodles : ℕ) 
  (boiling_water : ℕ) (cooking_noodles : ℕ) : ℕ :=
  washing_pot + boiling_water + cooking_noodles

theorem xiao_ning_min_cooking_time : 
  min_cooking_time 2 6 2 10 3 = 15 := 
by 
  rw [min_cooking_time, add_assoc, add_comm 10 3, add_assoc 2 10 3]
  sorry

end xiao_ning_min_cooking_time_l248_248847


namespace prince_spending_l248_248946

theorem prince_spending (CDs_total : ℕ) (CDs_10_percent : ℕ) (CDs_10_cost : ℕ) (CDs_5_cost : ℕ) 
  (Prince_10_fraction : ℚ) (Prince_5_fraction : ℚ) 
  (total_10_CDs : ℕ) (total_5_CDs : ℕ) (Prince_10_CDs : ℕ) (Prince_5_CDs : ℕ) (total_cost : ℕ) :
  CDs_total = 200 →
  CDs_10_percent = 40 →
  CDs_10_cost = 10 →
  CDs_5_cost = 5 →
  Prince_10_fraction = 1/2 →
  Prince_5_fraction = 1 →
  total_10_CDs = CDs_total * CDs_10_percent / 100 →
  total_5_CDs = CDs_total - total_10_CDs →
  Prince_10_CDs = total_10_CDs * Prince_10_fraction →
  Prince_5_CDs = total_5_CDs * Prince_5_fraction →
  total_cost = (Prince_10_CDs * CDs_10_cost) + (Prince_5_CDs * CDs_5_cost) →
  total_cost = 1000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end prince_spending_l248_248946


namespace sqrt_factorial_product_l248_248163

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248163


namespace sqrt_factorial_eq_l248_248135

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l248_248135


namespace battery_current_at_given_resistance_l248_248476

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248476


namespace battery_current_l248_248330

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248330


namespace identify_smart_person_l248_248942

theorem identify_smart_person (F S : ℕ) (h_total : F + S = 30) (h_max_fools : F ≤ 8) : S ≥ 1 :=
by {
  sorry
}

end identify_smart_person_l248_248942


namespace simplify_expression_l248_248957

theorem simplify_expression (a : ℝ) : 2 * (a + 2) - 2 * a = 4 :=
by
  sorry

end simplify_expression_l248_248957


namespace modulus_of_power_l248_248613

theorem modulus_of_power :
  complex.abs ((4 : ℂ) + 2 * complex.I) ^ 5 = 160 * real.sqrt 5 := 
sorry

end modulus_of_power_l248_248613


namespace odd_numbers_satisfy_polygon_property_l248_248969

theorem odd_numbers_satisfy_polygon_property (n : ℕ) (h : n ≥ 3) : 
  (∀ (P : polygon), is_convex P → (∀ d, side_length P d = 1) → 
    (∃ T : triangle, T ⊆ P ∧ is_equilateral T ∧ side_length T = 1)) ↔ odd n := sorry

end odd_numbers_satisfy_polygon_property_l248_248969


namespace volume_of_cylindrical_water_tank_l248_248836

-- Given the conditions
variables (diameter depth : ℝ)
-- Condition that diameter = 20
axiom diameter_is_20 : diameter = 20
-- Condition that depth = 6
axiom depth_is_6 : depth = 6

-- Define the radius and volume
def radius := diameter / 2
def volume := π * (radius ^ 2) * depth

-- State the theorem
theorem volume_of_cylindrical_water_tank : volume = 600 * π :=
by
  have r_def : radius = 10 := by 
    rw [radius, diameter_is_20]
    norm_num
  rw [volume, r_def]
  have vol_expr : π * (10 ^ 2) * 6 = 600 * π := by 
    norm_num
  rw vol_expr
  sorry

end volume_of_cylindrical_water_tank_l248_248836


namespace total_revenue_correct_l248_248779

-- Define the costs of different types of returns
def cost_federal : ℕ := 50
def cost_state : ℕ := 30
def cost_quarterly : ℕ := 80

-- Define the quantities sold for different types of returns
def qty_federal : ℕ := 60
def qty_state : ℕ := 20
def qty_quarterly : ℕ := 10

-- Calculate the total revenue for the day
def total_revenue : ℕ := (cost_federal * qty_federal) + (cost_state * qty_state) + (cost_quarterly * qty_quarterly)

-- The theorem stating the total revenue calculation
theorem total_revenue_correct : total_revenue = 4400 := by
  sorry

end total_revenue_correct_l248_248779


namespace battery_current_when_resistance_12_l248_248514

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248514


namespace current_at_resistance_12_l248_248299

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248299


namespace sum_of_possible_values_a_l248_248932

theorem sum_of_possible_values_a :
  ∀ (a b c d : ℤ), a > b → b > c → c > d →
    a + b + c + d = 50 →
    (∀ x y : ℤ, (x - y) ∈ {2, 3, 5, 7, 8, 10} ∨ (y - x) ∈ {2, 3, 5, 7, 8, 10}) →
    a = 17 :=
begin
  intros a b c d hab hbc hcd habcd sum_diff,
  sorry
end

end sum_of_possible_values_a_l248_248932


namespace current_value_l248_248425

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248425


namespace sqrt_factorial_product_l248_248073

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l248_248073


namespace current_value_l248_248265

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248265


namespace find_current_l248_248452

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248452


namespace current_value_l248_248422

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248422


namespace sqrt_factorial_mul_factorial_l248_248210

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l248_248210


namespace change_after_buying_tickets_l248_248824

def cost_per_ticket_3d : ℕ := 12
def student_discount_rate : ℚ := 0.25
def total_amount_brought : ℕ := 25

def discounted_ticket_price := cost_per_ticket_3d - (student_discount_rate * cost_per_ticket_3d).toNat
def total_cost := cost_per_ticket_3d + discounted_ticket_price

theorem change_after_buying_tickets :
  total_amount_brought - total_cost = 4 :=
by
  sorry

end change_after_buying_tickets_l248_248824


namespace simplify_parentheses_l248_248213

theorem simplify_parentheses (a b c x y : ℝ) : (3 * a - (2 * a - c) = 3 * a - 2 * a + c) := 
by 
  sorry

end simplify_parentheses_l248_248213


namespace current_when_resistance_12_l248_248463

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248463


namespace escalator_speed_l248_248940

-- Definitions for the given conditions
def length_of_escalator : ℝ := 180
def walking_speed_of_person : ℝ := 3
def time_taken : ℝ := 10

-- The statement using the given conditions to prove the rate of escalator
theorem escalator_speed :
  ∃ v : ℝ, (walking_speed_of_person + v) * time_taken = length_of_escalator ∧ v = 15 :=
by {
  let v := (length_of_escalator / time_taken) - walking_speed_of_person,
  use v,
  split,
  { 
    calc (walking_speed_of_person + v) * time_taken = length_of_escalator : sorry
  },
  {
    calc v = 15 : sorry
  }
}

end escalator_speed_l248_248940


namespace simple_interest_duration_l248_248027

noncomputable def si (P r t : ℝ) : ℝ := P * r * t
noncomputable def ci (P r n t : ℝ) : ℝ := P * (1 + r / n)^(n * t) - P

theorem simple_interest_duration :
  let Pₛᵢ := 2625.0000000000027
  let rₛᵢ := 0.08
  let P*cᴵ := 4000
  let r*cᴵ := 0.10
  let t*cᴵ := 2
  let si_half_c₀ := 0.5 * ci P*cᴵ r*cᴵ 1 t*cᴵ
  t = t*cᴵ := by
  let SI := si Pₛᵢ rₛᵢ t
  have h : SI = si_half_c₀ := by sorry
  exact sorry

end simple_interest_duration_l248_248027


namespace sqrt_factorial_multiplication_l248_248175

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l248_248175


namespace tangent_line_correct_and_value_of_a_l248_248752

-- Define the function f(x) = (2-x)e^x
def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

-- Define the tangent line at x = 0
def tangent_line_at_0 (x : ℝ) : ℝ := x + 2

-- Prove that the function satisfies the conditions mentioned
theorem tangent_line_correct_and_value_of_a (a : ℝ) :
  (∀ x, f x ≤ a * x + 2) → a ≥ 1 :=
begin
  -- Mathematical proof can be filled in here
  sorry
end

end tangent_line_correct_and_value_of_a_l248_248752


namespace current_value_l248_248494

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248494


namespace current_value_l248_248400

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248400


namespace new_shoes_cost_l248_248880

-- We define the given conditions
def cost_of_repairing_used_shoes : ℝ := 11.50
def lifespan_of_used_shoes : ℝ := 1
def lifespan_of_new_shoes : ℝ := 2
def percentage_increase : ℝ := 0.2173913043478261

-- We now state the theorem to be proven
theorem new_shoes_cost :
  ∃ P : ℝ, (P / 2) = cost_of_repairing_used_shoes * (1 + percentage_increase) ∧ P = 28.00 :=
begin
  existsi 28.00,
  split,
  { 
    calc 28.00 / 2 
        = 14.00    : by norm_num
    ... = 11.50 * (1 + 0.2173913043478261) : by norm_num,
  },
  { 
    refl,
  },
end

end new_shoes_cost_l248_248880


namespace quadratic_complete_square_l248_248804

theorem quadratic_complete_square :
  ∃ a b c : ℤ, (8 * x^2 - 48 * x - 320 = a * (x + b)^2 + c) ∧ (a + b + c = -387) :=
sorry

end quadratic_complete_square_l248_248804


namespace points_lie_on_parabola_l248_248625

theorem points_lie_on_parabola (u : ℝ) :
  ∃ (x y : ℝ), x = 3^u - 4 ∧ y = 9^u - 7 * 3^u - 2 ∧ y = x^2 + x - 14 :=
by
  sorry

end points_lie_on_parabola_l248_248625


namespace current_value_l248_248506

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248506


namespace smallest_perimeter_of_triangle_l248_248066

noncomputable def smallest_perimeter : ℕ :=
  let a := 4; let b := 6; let c := 8 in a + b + c

theorem smallest_perimeter_of_triangle :
  (∀ a b c : ℕ, even a ∧ even b ∧ even c ∧ b = a + 2 ∧ c = b + 2 →
  (a + b > c ∧ a + c > b ∧ b + c > a) → (a + b + c ≥ 18)) :=
by
  intros a b c h_ev h_bc h_ineq
  simp only [even] at h_ev
  cases h_ev with ha hb
  cases hb with hb hc
  simp only at h_bc
  cases h_bc with h_b h_c
  cases h_ineq with h_ineq1 h_ineq2
  cases h_ineq2 with h_ineq2 h_ineq3
  have h := calc
    a + b + c = 4 + 6 + 8 := by { sorry }
  show 4 + 6 + 8 ≥ 18 from by { sorry }
  sorry

end smallest_perimeter_of_triangle_l248_248066


namespace a_n_bound_l248_248025

noncomputable def a_seq : ℕ → ℝ
| 0     := 15
| (n+1) := (Real.sqrt (a_seq n ^ 2 + 1) - 1) / a_seq n

theorem a_n_bound (n : ℕ) (h : n ≥ 1) : a_seq n > 3 / 2 ^ n :=
sorry

end a_n_bound_l248_248025


namespace willow_catkin_diameter_scientific_notation_l248_248865

theorem willow_catkin_diameter_scientific_notation :
    ∃ d : ℝ, d = 1.05 * (10 : ℝ) ^ (-5) ∧ d = 0.0000105 := 
by
    use 0.0000105
    split
    · -- 0.0000105 = 1.05 * 10 ^ (-5)
    sorry
    · -- d = 0.0000105
    rfl

end willow_catkin_diameter_scientific_notation_l248_248865


namespace cone_base_circumference_l248_248896

theorem cone_base_circumference (V : ℝ) (h : ℝ) (r : ℝ) (C : ℝ) 
  (hV : V = 18 * Real.pi)
  (hh : h = 6) 
  (hV_cone : V = (1/3) * Real.pi * r^2 * h) :
  C = 2 * Real.pi * r → C = 6 * Real.pi :=
by 
  -- We assume as conditions are only mentioned
  sorry

end cone_base_circumference_l248_248896


namespace problem_solution_l248_248214

/-- Definitions of various vectors and their properties. --/
structure Vector3 :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def is_parallel (v1 v2 : Vector3) : Prop :=
  ∃ λ : ℝ, λ ≠ 0 ∧ v1 = { x := λ * v2.x, y := λ * v2.y, z := λ * v2.z }

def normal_vectors_parallel_to_planes (n1 n2 : Vector3) : Prop :=
  is_parallel n1 n2

def vector_perpendicular (v1 v2 : Vector3) : Prop :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z = 0

def vectors_opposite_directions (v1 v2 : Vector3) : Prop :=
  v1.x = -v2.x ∧ v1.y = -v2.y ∧ v1.z = -v2.z

noncomputable def misinterpretation1 (a : Vector3) (alpha : Type*) : Prop := sorry
noncomputable def misinterpretation2 (n1 n2 : Vector3) (alpha beta : Type*) : Prop := sorry
noncomputable def misinterpretation3 (v : Vector3) (l alpha : Type*) (n1 : Vector3) : Prop := sorry
noncomputable def correct_interpretation (ab cd : Vector3) : Prop := vectors_opposite_directions ab cd

def proposition_A (a : Vector3) (alpha : Type*) : Prop :=
  misinterpretation1 a alpha

def proposition_B (n1 n2 : Vector3) (alpha beta : Type*) : Prop :=
  misinterpretation2 n1 n2 alpha beta

def proposition_C (v : Vector3) (l alpha : Type*) (n1 : Vector3) : Prop :=
  misinterpretation3 v l alpha n1

def proposition_D (ab cd : Vector3) : Prop :=
  correct_interpretation ab cd

theorem problem_solution :
  ∀ (a : Vector3) (alpha beta : Type*) (n1 n2 : Vector3) (v : Vector3) (l : Type*) (ab cd : Vector3),
    ¬ proposition_A a alpha ∧
    ¬ proposition_B n1 n2 alpha beta ∧
    ¬ proposition_C v l alpha n1 ∧
    proposition_D ab cd := by
  -- The proof is long and complex, so it's skipped here.
  sorry

end problem_solution_l248_248214


namespace current_value_l248_248385

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248385


namespace line_tangent_ellipse_l248_248887

-- Define the conditions of the problem
def line (m : ℝ) (x y : ℝ) : Prop := y = m * x + 2
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9

-- Prove the statement about the intersection of the line and ellipse
theorem line_tangent_ellipse (m : ℝ) :
  (∀ x y, line m x y → ellipse x y → x = 0.0 ∧ y = 2.0)
  ↔ m^2 = 1 / 3 :=
sorry

end line_tangent_ellipse_l248_248887


namespace current_value_l248_248388

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248388


namespace current_value_l248_248499

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248499


namespace symmetric_point_coordinate_l248_248785

theorem symmetric_point_coordinate (P Q : ℝ × ℝ) (hP : P = (2, 5)) (hQ_symmetric : is_symmetric P Q (λ x y, x + y = 1)) : 
  Q = (-4, -1) := 
sorry

end symmetric_point_coordinate_l248_248785


namespace current_value_l248_248428

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248428


namespace current_value_l248_248509

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248509


namespace angle_BDC_is_21_5_degrees_l248_248820

noncomputable def angle_ABC := 65
noncomputable def angle_BAC := 72
noncomputable def angle_A := 180 - angle_ABC - angle_BAC
noncomputable def angle_A_div2 := angle_A / 2

theorem angle_BDC_is_21_5_degrees
  (ABC_tangent_to_circle : Triangle ABC is tangent to circle with center D)
  (angle_BAC_eq : angle BAC = 72) 
  (angle_ABC_eq : angle ABC = 65) 
  : angle BDC = 21.5 := 
by 
  sorry

end angle_BDC_is_21_5_degrees_l248_248820


namespace cannot_lie_in_second_quadrant_l248_248962

theorem cannot_lie_in_second_quadrant (a : ℝ) : 
  ¬ (let x_vertex := (a + 1) / 2,
         y_vertex := -a^2 - a - 1 in
         x_vertex < 0 ∧ y_vertex > 0) := 
by sorry

end cannot_lie_in_second_quadrant_l248_248962


namespace garden_area_percentage_increase_l248_248006

theorem garden_area_percentage_increase :
  let R1 := 5
  let R2 := 2
  let A1 := Float.pi * R1^2
  let A2 := Float.pi * R2^2
  let delta_A := A1 - A2
  let percentage_increase := (delta_A / A2) * 100
  percentage_increase = 525 := sorry

end garden_area_percentage_increase_l248_248006


namespace bamboo_break_height_l248_248873

theorem bamboo_break_height (x : ℝ) (h₁ : 0 < x) (h₂ : x < 9) (h₃ : x^2 + 3^2 = (9 - x)^2) : x = 4 :=
by
  sorry

end bamboo_break_height_l248_248873


namespace range_of_m_l248_248666

theorem range_of_m (m : ℝ) : 
  let a := (-1, 1)
  let b := (1, m)
  (a.1 * b.1 + a.2 * b.2 < 0) → 
  (m ∈ set.Iio (-1) ∨ (m ∈ set.Ioo (-1, 1))) := 
by
  let a : ℝ × ℝ := (-1, 1)
  let b : ℝ × ℝ := (1, m)
  sorry

end range_of_m_l248_248666


namespace sqrt_factorial_product_l248_248069

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l248_248069


namespace find_current_l248_248440

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248440


namespace sqrt_factorial_mul_factorial_eq_l248_248142

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l248_248142


namespace maximal_possible_r_squared_l248_248821

/-- Two congruent right circular cones each with base radius 4 and height 10 have axes of symmetry 
that intersect at right angles at a point in the interior of the cones, a distance 2 from the base 
of each cone. A sphere with radius r lies within both cones. The maximal possible value of r^2 is 144/29. -/
theorem maximal_possible_r_squared : 
  ∃ (r : ℝ), (congruent_cones 4 10) ∧ (axes_intersect_right_angle 2) ∧ (sphere_within_cones r) ∧ 
  (∀ x, cone_cross_section_tangency_points x → (r^2 ≤ (144/29))) :=
sorry

end maximal_possible_r_squared_l248_248821


namespace parallelepiped_length_l248_248907

theorem parallelepiped_length (n : ℕ) :
  (n - 2) * (n - 4) * (n - 6) = 2 * n * (n - 2) * (n - 4) / 3 →
  n = 18 :=
by
  intros h
  sorry

end parallelepiped_length_l248_248907


namespace min_k_2017_l248_248032

noncomputable def find_polynomial_min_k (P : ℤ[X]) : ℕ := sorry

theorem min_k_2017 (P : ℤ[X]) (hdeg : degree P = 2017) (hleading : P.leadingCoeff = 1) :
  find_polynomial_min_k P = 2017 :=
sorry

end min_k_2017_l248_248032


namespace current_when_resistance_12_l248_248467

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248467


namespace optimal_play_winner_l248_248862

theorem optimal_play_winner (A B : Type) (turns : ℕ → A)
  (initial_piles : ℕ × ℕ)
  (take_coins : ℕ → ℕ → Prop)
  (last_coin_win : ℕ → ℕ → Prop) :
  initial_piles = (2010, 2010) →
  (∀ n m, (take_coins n m) → (take_coins (n - 1) m ∨ take_coins n (m - 1) ∨ take_coins (n - 1) (m - 1))) →
  (∀ n, last_coin_win n 0 ∨ last_coin_win 0 n →
    (turns n = A → turns (n + 1) = B) ∧ (turns n = B → turns (n + 1) = A)) →
  (∀ n m, (last_coin_win n m) ↔ (n = 0 ∧ m = 0)) →
  ∃ k, turns k = B :=
by
  sorry

end optimal_play_winner_l248_248862


namespace additional_trams_proof_l248_248047

-- Definitions for the conditions
def initial_tram_count : Nat := 12
def total_distance : Nat := 60
def initial_interval : Nat := total_distance / initial_tram_count
def reduced_interval : Nat := initial_interval - (initial_interval / 5)
def final_tram_count : Nat := total_distance / reduced_interval
def additional_trams_needed : Nat := final_tram_count - initial_tram_count

-- The theorem we need to prove
theorem additional_trams_proof : additional_trams_needed = 3 :=
by
  sorry

end additional_trams_proof_l248_248047


namespace sqrt_factorial_equality_l248_248155

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l248_248155


namespace current_when_resistance_12_l248_248462

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248462


namespace sqrt_factorial_product_eq_24_l248_248195

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l248_248195


namespace battery_current_l248_248317

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248317


namespace probability_top_card_is_5_or_king_l248_248899

theorem probability_top_card_is_5_or_king :
  let total_cards := 52
  let count_fives := 4
  let count_kings := 4
  let total_desirable := count_fives + count_kings
  let probability := total_desirable / total_cards
  probability = 2 / 13 :=
by
  let total_cards := 52
  let count_fives := 4
  let count_kings := 4
  let total_desirable := count_fives + count_kings
  let probability := total_desirable / total_cards
  show probability = 2 / 13
  sorry

end probability_top_card_is_5_or_king_l248_248899


namespace MQ_parallel_NP_l248_248943

noncomputable def rhombus_parallel (A B C D O E F G H M N P Q : ℝ) : Prop := sorry

theorem MQ_parallel_NP {A B C D O E F G H M N P Q : ℝ}
  (h1 : rhombus_parallel A B C D O E F G H M N P Q)
  (inscribed : InscribedCircle A B C D O)
  (touch_points : TouchPoints O A B C D E F G H)
  (tangent_lines : TangentLines O E F G H M N P Q)
  (intersection_points : IntersectionPoints A B C D M N P Q) :
  Parallel MQ NP := sorry

end MQ_parallel_NP_l248_248943


namespace sqrt_factorial_mul_factorial_l248_248203

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l248_248203


namespace total_pints_l248_248611

-- Define the given conditions as constants
def annie_picked : Int := 8
def kathryn_picked : Int := annie_picked + 2
def ben_picked : Int := kathryn_picked - 3

-- State the main theorem to prove
theorem total_pints : annie_picked + kathryn_picked + ben_picked = 25 := by
  sorry

end total_pints_l248_248611


namespace ratio_second_number_is_approx_10_l248_248805

def proportion_second_number : ℝ := 215
def proportion_first_ratio : ℝ := 537
def proportion_second_ratio : ℝ := 26

theorem ratio_second_number_is_approx_10 :
  ∃ x : ℝ, (proportion_second_number / x = proportion_first_ratio / proportion_second_ratio) ∧ (x ≈ 10) :=
by
  sorry

end ratio_second_number_is_approx_10_l248_248805


namespace current_value_l248_248500

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248500


namespace correct_articles_l248_248234

-- Define the given conditions
def specific_experience : Prop := true
def countable_noun : Prop := true

-- Problem statement: given the conditions, choose the correct articles to fill in the blanks
theorem correct_articles (h1 : specific_experience) (h2 : countable_noun) : 
  "the; a" = "the; a" :=
by
  sorry

end correct_articles_l248_248234


namespace battery_current_l248_248374

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248374


namespace max_value_sum_cubes_l248_248750

theorem max_value_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) : 
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * real.sqrt 5 :=
sorry

end max_value_sum_cubes_l248_248750


namespace sequence_3001_values_l248_248590

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then x else
  if n = 2 then 3000 else
  (sequence x (n - 1) + 1) / sequence x (n - 2)

theorem sequence_3001_values : 
  ∃ n : ℕ, ∀ x : ℝ, (∃ n, sequence x n = 3001) → 
    x = 3001 ∨ x = 1 ∨ x = 3001 / 9002999 ∨ x = 9002999 :=
sorry

end sequence_3001_values_l248_248590


namespace kite_perimeter_l248_248886

/-- 
Prove that the perimeter of a kite with diagonals measuring 20 feet and 16 feet
and intersecting at right angles is 8√41 feet.
-/
theorem kite_perimeter (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 16) (h3 : true): 
  4 * (real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 8 * real.sqrt 41 := 
by
  obtain ⟨h3⟩ := h3;
  sorry

end kite_perimeter_l248_248886


namespace sqrt_factorial_product_l248_248094

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l248_248094


namespace find_current_when_resistance_is_12_l248_248353

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248353


namespace sqrt_factorial_product_l248_248098

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l248_248098


namespace add_trams_l248_248051

theorem add_trams (total_trams : ℕ) (total_distance : ℝ) (initial_intervals : ℝ) (new_intervals : ℝ) (additional_trams : ℕ) :
  total_trams = 12 → total_distance = 60 → initial_intervals = total_distance / total_trams →
  new_intervals = initial_intervals - (initial_intervals / 5) →
  additional_trams = (total_distance / new_intervals) - total_trams →
  additional_trams = 3 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end add_trams_l248_248051


namespace probability_sum_odd_correct_l248_248771

noncomputable def probability_sum_odd : ℚ :=
  let total_ways := 10
  let ways_sum_odd := 6
  ways_sum_odd / total_ways

theorem probability_sum_odd_correct :
  probability_sum_odd = 3 / 5 :=
by
  unfold probability_sum_odd
  rfl

end probability_sum_odd_correct_l248_248771


namespace tony_age_l248_248818

theorem tony_age (combined_age : ℕ) (certain_age : ℕ) (belinda_age : ℕ) (belinda_equation: belinda_age = 2 * certain_age + 8)
  (belinda_is_40 : belinda_age = 40) (combined_is_56 : combined_age = 56) : 
  (combined_age - belinda_age = 16) := 
by 
  have belinda_eq_40 := belinda_is_40
  rw [belinda_eq_40] at combined_is_56
  suffices belinda_age = 40 by sorry
  sorry

end tony_age_l248_248818


namespace sqrt_factorial_eq_l248_248132

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l248_248132


namespace sqrt_factorial_eq_l248_248136

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l248_248136


namespace battery_current_when_resistance_12_l248_248517

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248517


namespace storybook_pages_l248_248851

def reading_start_date := 10
def reading_end_date := 20
def pages_per_day := 11
def number_of_days := reading_end_date - reading_start_date + 1
def total_pages := pages_per_day * number_of_days

theorem storybook_pages : total_pages = 121 := by
  sorry

end storybook_pages_l248_248851


namespace current_at_R_12_l248_248240

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248240


namespace sqrt_factorial_product_l248_248174

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248174


namespace sum_digits_odds_proof_l248_248992

open Nat

-- Definition: sum_digits computes the sum of digits of a given number n.
noncomputable def sum_digits (n : ℕ) : ℕ :=
  if n == 0 then 0 else n % 10 + sum_digits (n / 10)

-- Condition: Sum of digits of all odd numbers from 1 to 10000.
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Sum the digits of all odd numbers from 1 to 10000.
noncomputable def sum_of_digits_of_odds (m n : ℕ) : ℕ :=
  if m > n then 0 else 
  if is_odd m then sum_digits m + sum_of_digits_of_odds (m + 2) n
  else sum_of_digits_of_odds (m + 1) n 

-- Proof problem: Demonstrate that the calculated sum is equal to 97500.
theorem sum_digits_odds_proof (m n : ℕ) (h₁ : m = 1) (h₂ : n = 10000) :
  sum_of_digits_of_odds m n = 97500 := 
sorry

end sum_digits_odds_proof_l248_248992


namespace find_current_l248_248448

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248448


namespace greatest_power_of_two_factor_l248_248065

theorem greatest_power_of_two_factor (n : ℕ) (h : n = 1000) :
  ∃ k, 2^k ∣ 10^n + 4^(n/2) ∧ k = 1003 :=
by {
  sorry
}

end greatest_power_of_two_factor_l248_248065


namespace current_at_resistance_12_l248_248545

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248545


namespace sqrt_factorial_product_l248_248121

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l248_248121


namespace heaps_never_empty_l248_248815

-- Define initial conditions
def initial_heaps := (1993, 199, 19)

-- Allowed operations
def add_stones (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
if a = 1993 then (a + b + c, b, c)
else if b = 199 then (a, b + a + c, c)
else (a, b, c + a + b)

def remove_stones (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
if a = 1993 then (a - (b + c), b, c)
else if b = 199 then (a, b - (a + c), c)
else (a, b, c - (a + b))

-- The proof statement
theorem heaps_never_empty :
  ∀ a b c : ℕ, a = 1993 ∧ b = 199 ∧ c = 19 ∧ (∀ n : ℕ, (a + b + c) % 2 = 1) ∧ (a - (b + c) % 2 = 1) → ¬(a = 0 ∨ b = 0 ∨ c = 0) := 
by {
  sorry
}

end heaps_never_empty_l248_248815


namespace battery_current_at_given_resistance_l248_248489

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248489


namespace shifted_tangent_increasing_on_the_intervals_l248_248601

-- Define the shifted tangent function
def shifted_tangent (x : ℝ) : ℝ := Real.tan (x - Real.pi / 4)

-- Define what it means for the shifted tangent function to be increasing
def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x1 x2 ∈ I, x1 < x2 → f x1 < f x2

-- State the intervals in which the function is proposed to be increasing
def increasing_intervals (k : ℤ) : Set ℝ :=
  Set.Ioo (k * Real.pi - Real.pi / 4) (k * Real.pi + 3 * Real.pi / 4)

-- Main theorem statement
theorem shifted_tangent_increasing_on_the_intervals :
  ∀ (k : ℤ), is_increasing_on shifted_tangent (increasing_intervals k) :=
sorry

end shifted_tangent_increasing_on_the_intervals_l248_248601


namespace current_at_resistance_12_l248_248533

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248533


namespace sqrt_factorial_product_l248_248126

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l248_248126


namespace sqrt_factorial_product_l248_248097

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l248_248097


namespace number_of_triangles_l248_248593

theorem number_of_triangles : 
  ∀ (rectangle : Type) (lines : ℕ),
    (lines = 3) → 
    (∀ (rectangles : ℕ), rectangles = 6 → 
      (∀ (triangles1 : ℕ), triangles1 = rectangles * 2 →
        (∀ (additional_lines : ℕ), additional_lines = 2 →
          (∀ (sub_rectangles : ℕ), sub_rectangles = rectangles * (additional_lines + 1) →
            (∀ (triangles2 : ℕ), triangles2 = sub_rectangles * 2 →
              (triangles1 + triangles2 = 48)))))
:sorry

end number_of_triangles_l248_248593


namespace max_possible_n_l248_248223

theorem max_possible_n (n : ℤ) (h : 101 * n ^ 2 ≤ 6400) : n ≤ 7 :=
by {
  sorry
}

end max_possible_n_l248_248223


namespace prime_power_implies_one_l248_248732

theorem prime_power_implies_one (p : ℕ) (a : ℤ) (n : ℕ) (h_prime : Nat.Prime p) (h_eq : 2^p + 3^p = a^n) :
  n = 1 :=
sorry

end prime_power_implies_one_l248_248732


namespace sum_of_five_consecutive_odd_numbers_l248_248806

theorem sum_of_five_consecutive_odd_numbers (x : ℤ) : 
  (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 5 * x :=
by
  sorry

end sum_of_five_consecutive_odd_numbers_l248_248806


namespace battery_current_l248_248320

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248320


namespace battery_current_l248_248285

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248285


namespace sqrt_factorial_product_l248_248077

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l248_248077


namespace current_when_resistance_is_12_l248_248406

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248406


namespace oliver_dishes_count_l248_248895

def total_dishes : ℕ := 42
def mango_salsa_dishes : ℕ := 5
def fresh_mango_dishes : ℕ := total_dishes / 6
def mango_jelly_dishes : ℕ := 2
def strawberry_dishes : ℕ := 3
def pineapple_dishes : ℕ := 5
def kiwi_dishes : ℕ := 4
def mango_dishes_oliver_picks_out : ℕ := 3

def total_mango_dishes : ℕ := mango_salsa_dishes + fresh_mango_dishes + mango_jelly_dishes
def mango_dishes_oliver_wont_eat : ℕ := total_mango_dishes - mango_dishes_oliver_picks_out
def max_strawberry_pineapple_dishes : ℕ := strawberry_dishes

def dishes_left_for_oliver : ℕ := total_dishes - mango_dishes_oliver_wont_eat - max_strawberry_pineapple_dishes

theorem oliver_dishes_count : dishes_left_for_oliver = 28 := 
by 
  sorry

end oliver_dishes_count_l248_248895


namespace sqrt_factorial_product_l248_248078

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l248_248078


namespace algebraic_expression_value_l248_248680

theorem algebraic_expression_value (m : ℝ) (h : m^2 - 6 * m - 5 = 0) : 11 + 6 * m - m^2 = 6 :=
by {
  sorry,
}

end algebraic_expression_value_l248_248680


namespace adjacent_side_length_l248_248557

-- Given the conditions
variables (a b : ℝ)
-- Area of the rectangular flower bed
def area := 6 * a * b - 2 * b
-- One side of the rectangular flower bed
def side1 := 2 * b

-- Prove the length of the adjacent side
theorem adjacent_side_length : 
  (6 * a * b - 2 * b) / (2 * b) = 3 * a - 1 :=
by sorry

end adjacent_side_length_l248_248557


namespace cylinder_volume_increase_l248_248841

theorem cylinder_volume_increase (r h : ℝ) (V : ℝ) : 
  V = π * r^2 * h → 
  (3 * h) * (2 * r)^2 * π = 12 * V := by
    sorry

end cylinder_volume_increase_l248_248841


namespace percentage_above_wholesale_correct_l248_248558

variable (wholesale_cost retail_cost employee_payment : ℝ)
variable (employee_discount percentage_above_wholesale : ℝ)

theorem percentage_above_wholesale_correct :
  wholesale_cost = 200 → 
  employee_discount = 0.25 → 
  employee_payment = 180 → 
  retail_cost = wholesale_cost + (percentage_above_wholesale / 100) * wholesale_cost →
  employee_payment = (1 - employee_discount) * retail_cost →
  percentage_above_wholesale = 20 :=
by
  intros
  sorry

end percentage_above_wholesale_correct_l248_248558


namespace find_sum_pqrsts_l248_248737

def Q (x : ℝ) : ℝ := x^2 - 5*x - 9

def condition_x (x : ℝ) : Prop := 3 ≤ x ∧ x ≤ 18

noncomputable def PQRST : ℝ × ℝ × ℝ × ℝ × ℝ := (2, 5, 0, 0, 5)

theorem find_sum_pqrsts (x : ℝ) (p q r s t : ℝ) (h : PQRST = (p, q, r, s, t)) (h_x : condition_x x):
  \lfloor sqrt (abs (Q x)) \rfloor = sqrt (abs (Q (floor x))) → (p + q + r + s + t = 7) :=
  sorry

end find_sum_pqrsts_l248_248737


namespace battery_current_l248_248344

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248344


namespace battery_current_l248_248347

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248347


namespace calculate_expression_l248_248581

theorem calculate_expression : (sqrt 3 - 2)^2 + sqrt 12 + 6 * sqrt (1 / 3) = 7 :=
by
  sorry

end calculate_expression_l248_248581


namespace total_revenue_l248_248782

def price_federal := 50
def price_state := 30
def price_quarterly := 80
def num_federal := 60
def num_state := 20
def num_quarterly := 10

theorem total_revenue : (num_federal * price_federal + num_state * price_state + num_quarterly * price_quarterly) = 4400 := by
  sorry

end total_revenue_l248_248782


namespace current_value_l248_248389

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248389


namespace current_value_l248_248427

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248427


namespace battery_current_at_given_resistance_l248_248488

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248488


namespace sqrt_factorial_product_l248_248115

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l248_248115


namespace find_current_when_resistance_is_12_l248_248360

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248360


namespace integer_triplets_prime_l248_248749

theorem integer_triplets_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ sol : ℕ, ((∃ (x y z : ℤ), (3 * x + y + z) * (x + 2 * y + z) * (x + y + z) = p) ∧
  if p = 2 then sol = 4 else sol = 12) :=
by
  sorry

end integer_triplets_prime_l248_248749


namespace distance_CF_eq_1_l248_248016

-- Define the points and lengths
variables {A C E F : Type} [MetricSpace A] [MetricSpace C] [MetricSpace E] [MetricSpace F]
variable (AC : ℝ) (CD : ℝ) (DE : ℝ)
variable (areaABC : ℝ)
variable (distanceCF : ℝ)

-- Given conditions
axiom AC_eq_20m : AC = 20
axiom CD_eq_10m : CD = 10
axiom DE_eq_10m : DE = 10
axiom area_ABC_eq_120 : areaABC = 120
axiom total_area_eq_270 : (areaABC + (CD * DE + (DE * AC / 2))) = 270

-- Total area for each part when divided
def half_total_area (total_area : ℝ) : ℝ := total_area / 2

-- State the theorem to prove distance CF for equal land division
theorem distance_CF_eq_1.5m
  (h : half_total_area 270 = 135) :
  distanceCF = 1.5 := sorry

end distance_CF_eq_1_l248_248016


namespace overtime_hours_l248_248890

theorem overtime_hours (regular_rate: ℝ) (regular_hours: ℝ) (total_payment: ℝ) (overtime_rate_multiplier: ℝ) (overtime_hours: ℝ):
  regular_rate = 3 → regular_hours = 40 → total_payment = 198 → overtime_rate_multiplier = 2 → 
  overtime_hours = (total_payment - (regular_rate * regular_hours)) / (regular_rate * overtime_rate_multiplier) →
  overtime_hours = 13 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end overtime_hours_l248_248890


namespace battery_current_l248_248291

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248291


namespace current_value_l248_248395

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248395


namespace current_value_l248_248398

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248398


namespace battery_current_l248_248370

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248370


namespace find_current_when_resistance_is_12_l248_248351

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248351


namespace smallest_n_prob_lt_half_l248_248875

noncomputable def probability_of_red_apple_drawn_each_time (n : ℕ) : ℚ :=
  (List.prod (List.range n).map (λ k, (9 - k) / (10 - k)))

theorem smallest_n_prob_lt_half : ∃ n : ℕ, probability_of_red_apple_drawn_each_time n < (1 / 2) ∧ n = 8 := by
  sorry

end smallest_n_prob_lt_half_l248_248875


namespace greatest_sum_of_integers_squared_eq_twenty_l248_248813

theorem greatest_sum_of_integers_squared_eq_twenty : 
  ∃ (x y : ℤ), x^2 + y^2 = 20 ∧ ∀ (x' y' : ℤ), x'^2 + y'^2 = 20 → x + y ≥ x' + y' :=
begin
  use [4, 2],
  split,
  { have eq_test : 4^2 + 2^2 = 20, by norm_num,
    exact eq_test },
  { intros x' y' h,
    have possible_pairs := λ x' y', x'^2 + y'^2 = 20,
    have valid_pairs := {(2, 4), (2, -4), (-2, 4), (-2, -4), (4, 2), (4, -2), (-4, 2), (-4, -2)} : set (ℤ × ℤ),
    have all_pairs := finite valid_pairs,
    have max_sum_verified, sorry }
end

end greatest_sum_of_integers_squared_eq_twenty_l248_248813


namespace parallelepiped_length_l248_248914

theorem parallelepiped_length (n : ℕ)
  (h1 : ∃ n : ℕ, n = 18) 
  (h2 : one_third_of_cubes_have_red_faces : (∃ k : ℕ, k = ((n * (n - 2) * (n - 4)) / 3)) 
        ∧ (remaining_unpainted_cubes : (∃ m : ℕ , m = (2 * (n * (n - 2) * (n - 4)) / 3))))
  (h3 : painted_and_cut_into_cubes : (∃ a b c : ℕ, a = n ∧ b = (n - 2) ∧ c = (n - 4)))
  (h4 : all_sides_whole_cm : (∃ d : ℕ , d = n ∧ d = (n - 2) ∧ d = (n - 4))) :
  n = 18 :=
begin
  sorry
end

end parallelepiped_length_l248_248914


namespace monotonicity_f_gt_1_monotonicity_f_lt_1_l248_248609

noncomputable def f (a x : ℝ) : ℝ :=
  real.log (a) ((x + 1) / (x - 1))

theorem monotonicity_f_gt_1
  (a : ℝ)
  (h1 : a > 1)
  (x1 x2 : ℝ)
  (hx1 : x1 > 1)
  (hx2 : x2 > 1)
  (hx12 : x1 < x2) :
  f a x2 < f a x1 :=
begin
  sorry
end

theorem monotonicity_f_lt_1
  (a : ℝ)
  (h1 : 0 < a)
  (h2 : a < 1)
  (x1 x2 : ℝ)
  (hx1 : x1 > 1)
  (hx2 : x2 > 1)
  (hx12 : x1 < x2) :
  f a x1 < f a x2 :=
begin
  sorry
end

end monotonicity_f_gt_1_monotonicity_f_lt_1_l248_248609


namespace find_current_l248_248449

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248449


namespace current_value_l248_248437

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248437


namespace time_to_eat_3_pounds_l248_248758

-- Define the eating rates.
def mrFatRate : ℝ := 1 / 20  -- pounds per minute
def mrThinRate : ℝ := 1 / 30  -- pounds per minute
def combinedRate : ℝ := mrFatRate + mrThinRate  -- combined pounds per minute

-- State the goal.
theorem time_to_eat_3_pounds : (3 : ℝ) / combinedRate = 36 :=
  sorry

end time_to_eat_3_pounds_l248_248758


namespace no_square_reassembly_l248_248638

def area (shape : ℝ) : ℝ := shape -- placeholder for shape area

noncomputable def cut_and_reassemble_to_square (circular_paper : ℝ) : Prop :=
  ∃ pieces : List ℝ, area circular_paper = area (join pieces) ∧ 
  ∀ piece ∈ pieces, is_straight_or_arc piece ∧ shape_aspects piece

theorem no_square_reassembly (circle : ℝ) :
  ¬cut_and_reassemble_to_square circle :=
  sorry

end no_square_reassembly_l248_248638


namespace polyhedron_value_l248_248979

/-- Given Euler's formula for a convex polyhedron V - E + F = 2,
    where F = 24 (faces), each face is either a quadrilateral (q) or hexagon (h).
    Each vertex is incident with one quadrilateral and one hexagon (Q = 1, H = 1).
    Prove that the value of 100H + 10Q + V is 136. -/
theorem polyhedron_value :
  ∃ V E q h : ℕ, q + h = 24 ∧ E = 2*q + 3*h ∧ V - E + 24 = 2 ∧ 100*1 + 10*1 + V = 136 :=
begin
  sorry
end

end polyhedron_value_l248_248979


namespace can_display_total_l248_248550

theorem can_display_total (n : ℕ) :
  (∀ k : ℕ, k < n → 1 + 2 * k = 0 + sum (range n)) →
  (∑ k in range n, (2 * k + 1)) = 100 → n = 10 :=
by
  sorry

end can_display_total_l248_248550


namespace sqrt_factorial_mul_factorial_l248_248082

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l248_248082


namespace no_six_digit_perfect_square_with_conditions_l248_248616

theorem no_six_digit_perfect_square_with_conditions :
  ∀ (N : ℕ), 
  (100000 ≤ N ∧ N < 1000000) ∧ 
  (∃ x : ℕ, N = x^2 ∧ 1000 ≤ x ∧ x < 10000) ∧ 
  (∀ digit : ℕ, digit ∈ digits N → digit ≠ 0) ∧ 
  (∀ i j k l m n : ℕ, 
    digits N = [i, j, k, l, m, n] →
    [i, j, k, l, m, n].nodup) ∧ 
  (∀ a b c : ℕ, 
    digits N = [a, b, c] →
    (a, b, c ∈ { x | x = 16 ∨ x = 25 ∨ x = 36 ∨ x = 49 ∨ x = 64 ∨ x = 81 }) ∧ 
      a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  false := 
by sorry

end no_six_digit_perfect_square_with_conditions_l248_248616


namespace parallelepiped_length_l248_248913

theorem parallelepiped_length (n : ℕ)
  (h1 : ∃ n : ℕ, n = 18) 
  (h2 : one_third_of_cubes_have_red_faces : (∃ k : ℕ, k = ((n * (n - 2) * (n - 4)) / 3)) 
        ∧ (remaining_unpainted_cubes : (∃ m : ℕ , m = (2 * (n * (n - 2) * (n - 4)) / 3))))
  (h3 : painted_and_cut_into_cubes : (∃ a b c : ℕ, a = n ∧ b = (n - 2) ∧ c = (n - 4)))
  (h4 : all_sides_whole_cm : (∃ d : ℕ , d = n ∧ d = (n - 2) ∧ d = (n - 4))) :
  n = 18 :=
begin
  sorry
end

end parallelepiped_length_l248_248913


namespace find_x_log_eq_l248_248982

theorem find_x_log_eq (x : ℝ) (h : log x 243 = 5 / 3) : x = 27 :=
sorry

end find_x_log_eq_l248_248982


namespace perp_vectors_eq_l248_248665

theorem perp_vectors_eq {k : ℝ} 
  (ha : (-1 : ℝ), 3)
  (hb : (1 : ℝ), k) 
  (hperp : (-1) * 1 + 3 * k = 0) : k = 1 / 3 := 
  sorry

end perp_vectors_eq_l248_248665


namespace bug_paths_l248_248876

def point := ℕ

structure HexLattice :=
  (A B : point)
  (midpoints : point)
  (endpoints : point)
  (arrow_from_A_to_midpoints : \u03ASigma) -- List of arrows from A to midpoints
  (arrows_from_midpoints_to_endpoints : \u03ASigma) -- List of arrows from midpoints to endpoints
  (arrow_from_endpoints_to_B : \u03ASigma) -- List of arrows from endpoints to B
  (all_arrows_distinct : arrows_distinct)  -- Condition: not traveling on the same segment more than once

axiom arrow_count : ∀ (l : HexLattice), 
  (list.length l.arrow_from_A_to_midpoints = 3) ∧
  (list.length l.arrows_from_midpoints_to_endpoints = 6) ∧
  (list.length l.arrow_from_endpoints_to_B = 1)

theorem bug_paths (l : HexLattice) :
  3 * 6 * 1 = 18 :=
by sorry

end bug_paths_l248_248876


namespace sqrt_factorial_multiplication_l248_248186

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l248_248186


namespace y_is_multiple_of_12_y_is_multiple_of_3_y_is_multiple_of_4_y_is_multiple_of_6_l248_248734

def y : ℕ := 36 + 48 + 72 + 144 + 216 + 432 + 1296

theorem y_is_multiple_of_12 : y % 12 = 0 := by
  sorry

theorem y_is_multiple_of_3 : y % 3 = 0 := by
  have h := y_is_multiple_of_12
  sorry

theorem y_is_multiple_of_4 : y % 4 = 0 := by
  have h := y_is_multiple_of_12
  sorry

theorem y_is_multiple_of_6 : y % 6 = 0 := by
  have h := y_is_multiple_of_12
  sorry

end y_is_multiple_of_12_y_is_multiple_of_3_y_is_multiple_of_4_y_is_multiple_of_6_l248_248734


namespace sqrt_factorial_mul_factorial_l248_248087

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l248_248087


namespace proof_problem_l248_248704

variable (a b c : ℝ) (A B C : ℝ)
variable (acuteTri : ∀ (x : ℝ), x > 0)
variable (eq1 : (a^2 + b^2 - c^2) / (-a * c) = cos (A + C) / (sin A * cos A))
variable (A_eq_pi4 : A = π / 4)
variable (a_eq_sqrt3 : a = sqrt 3)
variable (maxEq : sin B + cos (C - (7 * π) / 12) = sqrt 3)

theorem proof_problem :
  (A = π / 4) ∧ 
  (a = sqrt 3) → 
  ((sin B + cos (C - (7 * π) / 12) = sqrt 3) ∧ (B = π / 3) ∧ (b = 3 * sqrt 6 / 2)) :=
sorry

end proof_problem_l248_248704


namespace current_value_l248_248271

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248271


namespace current_value_l248_248391

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248391


namespace find_higher_selling_price_l248_248936

-- Define the constants and initial conditions
def cost_price : ℕ := 200
def selling_price_1 : ℕ := 340
def gain_1 : ℕ := selling_price_1 - cost_price
def new_gain : ℕ := gain_1 + gain_1 * 5 / 100

-- Define the problem statement
theorem find_higher_selling_price : 
  ∀ P : ℕ, P = cost_price + new_gain → P = 347 :=
by
  intro P
  intro h
  sorry

end find_higher_selling_price_l248_248936


namespace current_value_l248_248266

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248266


namespace potatoes_already_cooked_l248_248546

-- Definitions of conditions
def total_potatoes : Nat := 13
def time_per_potato : Nat := 6
def remaining_cooking_time : Nat := 48

-- Statement of the theorem
theorem potatoes_already_cooked :
  let potatoes_left := remaining_cooking_time / time_per_potato
  in (total_potatoes - potatoes_left) = 5 := by
  sorry

end potatoes_already_cooked_l248_248546


namespace current_at_resistance_12_l248_248538

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248538


namespace maximum_sum_of_distances_l248_248766

open Set
open Function
open EuclideanGeometry

variable {P : Type*} [MetricSpace P] [NormedGroup P] [NormedSpace ℝ P] [EuclideanSpace ℝ P]

theorem maximum_sum_of_distances (A B C M : P) (l : AffineSubspace ℝ P) 
  (hM : midpoint ℝ B C M) (hAM : convex ℝ ({A, M} : Set P)) 
  (h_perp : ⊥ (l.direction ⊓ (lineThrough ℝ A M).direction)) :
  ∀ l', (l'.direction ⊓ (lineThrough ℝ A M).direction = ⊥) → (sum_distances_to_line A B C l ≤ sum_distances_to_line A B C l') :=
sorry

end maximum_sum_of_distances_l248_248766


namespace battery_current_when_resistance_12_l248_248526

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248526


namespace battery_current_l248_248368

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248368


namespace solve_for_C_days_l248_248869

noncomputable def A_work_rate : ℚ := 1 / 20
noncomputable def B_work_rate : ℚ := 1 / 15
noncomputable def C_work_rate : ℚ := 1 / 50
noncomputable def total_work_done_by_A_B : ℚ := 6 * (A_work_rate + B_work_rate)
noncomputable def remaining_work : ℚ := 1 - total_work_done_by_A_B

theorem solve_for_C_days : ∃ d : ℚ, d * C_work_rate = remaining_work ∧ d = 15 :=
by
  use 15
  simp [C_work_rate, remaining_work, total_work_done_by_A_B, A_work_rate, B_work_rate]
  sorry

end solve_for_C_days_l248_248869


namespace current_value_l248_248431

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248431


namespace current_value_l248_248426

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248426


namespace ammonium_chloride_reacts_with_potassium_hydroxide_l248_248617

/-- Prove that 1 mole of ammonium chloride is required to react with 
    1 mole of potassium hydroxide to form 1 mole of ammonia, 
    1 mole of water, and 1 mole of potassium chloride, 
    given the balanced chemical equation:
    NH₄Cl + KOH → NH₃ + H₂O + KCl
-/
theorem ammonium_chloride_reacts_with_potassium_hydroxide :
    ∀ (NH₄Cl KOH NH₃ H₂O KCl : ℕ), 
    (NH₄Cl + KOH = NH₃ + H₂O + KCl) → 
    (NH₄Cl = 1) → 
    (KOH = 1) → 
    (NH₃ = 1) → 
    (H₂O = 1) → 
    (KCl = 1) → 
    NH₄Cl = 1 :=
by
  intros
  sorry

end ammonium_chloride_reacts_with_potassium_hydroxide_l248_248617


namespace length_of_parallelepiped_l248_248924

def number_of_cubes_with_painted_faces (n : ℕ) := (n - 2) * (n - 4) * (n - 6) 
def total_number_of_cubes (n : ℕ) := n * (n - 2) * (n - 4)

theorem length_of_parallelepiped (n : ℕ) (h1 : total_number_of_cubes n = 3 * number_of_cubes_with_painted_faces n) : 
  n = 18 :=
by 
  sorry

end length_of_parallelepiped_l248_248924


namespace sqrt_factorial_product_l248_248116

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l248_248116


namespace sqrt_factorial_multiplication_l248_248177

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l248_248177


namespace complex_simplification_l248_248686

theorem complex_simplification :
  let z := (2 + complex.I) / complex.I in
  let a := z.re in
  let b := z.im in
  (b / a) = -2 :=
by
  sorry

end complex_simplification_l248_248686


namespace train_length_is_correct_l248_248902

noncomputable def speed_conversion (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def relative_speed (train_speed : ℝ) (bike_speed : ℝ) : ℝ :=
  speed_conversion(train_speed) - speed_conversion(bike_speed)

noncomputable def train_length (train_speed : ℝ) (bike_speed : ℝ) (time_seconds : ℕ) : ℝ :=
  relative_speed(train_speed, bike_speed) * time_seconds

theorem train_length_is_correct :
  train_length 150 90 12 = 200.04 := by
  sorry

end train_length_is_correct_l248_248902


namespace coefficient_of_x_squared_in_expansion_l248_248010

theorem coefficient_of_x_squared_in_expansion :
  let f := (1 + x) * (2 - x)^4 in
  coeff (expand f) x 2 = -8 :=
by 
  let f := (1 + x) * (2 - x)^4
  have h : coeff (expand f) x 2 = -8
  sorry

end coefficient_of_x_squared_in_expansion_l248_248010


namespace sequence_inequality_l248_248748

theorem sequence_inequality 
  (n : ℕ) 
  (x y : Fin n → ℝ) 
  (z : Fin (2 * n) → ℝ) 
  (hn : 0 < n)
  (h_pos_x : ∀ i, 0 < x i) 
  (h_pos_y : ∀ j, 0 < y j) 
  (h_pos_z : ∀ k, 2 ≤ k → k < 2 * n + 2 → 0 < z ⟨k - 2, by sorry⟩)
  (h_ineq : ∀ i j : Fin n, z ⟨i + j + 2, by sorry⟩ ^ 2 ≥ x i * y j) :
  ( (max (Fin.range (2 * n) \ {0, 1}) fun x => z x + (∑ i, z ⟨i + 2, by sorry⟩)) / (2 * n) ) ^ 2  ≥ 
  ( ( ∑ i, x i) / n) * ( ( ∑ i, y i) / n) := 
sorry

end sequence_inequality_l248_248748


namespace battery_current_when_resistance_12_l248_248523

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248523


namespace tan_alpha_minus_beta_over_2_cos_alpha_plus_beta_over_2_l248_248643

variables (α β : ℝ)

noncomputable def tan_value := 12 / 5
noncomputable def cos_value := 56 / 65

-- Conditions
axiom condition1 : 0 < β ∧ β < π/2 ∧ π/2 < α ∧ α < π
axiom condition2 : cos (α - β/2) = 5 / 13
axiom condition3 : sin (α/2 - β) = 3 / 5

-- Proof goals
theorem tan_alpha_minus_beta_over_2 : tan (α - β/2) = tan_value :=
begin
  sorry -- Proof omitted
end

theorem cos_alpha_plus_beta_over_2 : cos ((α + β)/2) = cos_value :=
begin
  sorry -- Proof omitted
end

end tan_alpha_minus_beta_over_2_cos_alpha_plus_beta_over_2_l248_248643


namespace current_at_resistance_12_l248_248307

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248307


namespace sqrt_factorial_product_l248_248068

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l248_248068


namespace sqrt_factorial_eq_l248_248134

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l248_248134


namespace total_revenue_l248_248781

def price_federal := 50
def price_state := 30
def price_quarterly := 80
def num_federal := 60
def num_state := 20
def num_quarterly := 10

theorem total_revenue : (num_federal * price_federal + num_state * price_state + num_quarterly * price_quarterly) = 4400 := by
  sorry

end total_revenue_l248_248781


namespace initial_number_of_people_l248_248956

theorem initial_number_of_people (X : ℕ) (h : ((X - 10) + 15 = 17)) : X = 12 :=
by
  sorry

end initial_number_of_people_l248_248956


namespace brian_final_cards_l248_248577

-- Definitions of initial conditions
def initial_cards : ℕ := 76
def cards_taken : ℕ := 59
def packs_bought : ℕ := 3
def cards_per_pack : ℕ := 15

-- The proof problem: Prove that the final number of cards is 62
theorem brian_final_cards : initial_cards - cards_taken + packs_bought * cards_per_pack = 62 :=
by
  -- Proof goes here, 'sorry' used to skip actual proof
  sorry

end brian_final_cards_l248_248577


namespace current_value_l248_248492

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248492


namespace farmer_children_l248_248885

theorem farmer_children (n : ℕ) 
  (h1 : 15 * n - 8 - 7 = 60) : n = 5 := 
by
  sorry

end farmer_children_l248_248885


namespace current_value_l248_248269

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248269


namespace current_when_resistance_12_l248_248457

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248457


namespace determine_a_l248_248598

theorem determine_a (a : ℕ) (p1 p2 : ℕ) (h1 : Prime p1) (h2 : Prime p2) (h3 : 2 * p1 * p2 = a) (h4 : p1 + p2 = 15) : 
  a = 52 :=
by
  sorry

end determine_a_l248_248598


namespace add_trams_l248_248049

theorem add_trams (total_trams : ℕ) (total_distance : ℝ) (initial_intervals : ℝ) (new_intervals : ℝ) (additional_trams : ℕ) :
  total_trams = 12 → total_distance = 60 → initial_intervals = total_distance / total_trams →
  new_intervals = initial_intervals - (initial_intervals / 5) →
  additional_trams = (total_distance / new_intervals) - total_trams →
  additional_trams = 3 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end add_trams_l248_248049


namespace battery_current_at_given_resistance_l248_248477

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248477


namespace battery_current_when_resistance_12_l248_248524

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248524


namespace battery_current_at_given_resistance_l248_248479

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248479


namespace change_occurs_in_3_years_l248_248769

theorem change_occurs_in_3_years (P A1 A2 : ℝ) (R T : ℝ) (h1 : P = 825) (h2 : A1 = 956) (h3 : A2 = 1055)
    (h4 : A1 = P + (P * R * T) / 100)
    (h5 : A2 = P + (P * (R + 4) * T) / 100) : T = 3 :=
by
  sorry

end change_occurs_in_3_years_l248_248769


namespace max_roads_possible_l248_248868
-- importing all necessary libraries

-- defining the problem contiditions
variables (N : ℕ) (d : ℕ)

-- defining the maximum possible value of d
def max_possible_d {N : ℕ} : ℕ := N * (N - 1) * (N - 2) / 6

-- stating the theorem to be proved
theorem max_roads_possible (h: N > 2) :
  d = max_possible_d N :=
sorry

end max_roads_possible_l248_248868


namespace current_value_l248_248421

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248421


namespace battery_current_at_given_resistance_l248_248475

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248475


namespace triangle_area_l248_248569

theorem triangle_area (x : ℝ) 
  (h1 : 13 * x = 10) -- hypotenuse equal to the diameter of the circle
  (h2 : 0 < x) : -- x should be positive to maintain triangle inequality and positivity
  (1 / 2 * 5 * x * 12 * x = 3000 / 169) :=
begin
  have h3 : x = 10 / 13,
  { exact (eq_div_iff (by norm_num)).mpr h1 },
  rw h3,
  norm_num,
end

end triangle_area_l248_248569


namespace remaining_volume_of_cube_l248_248884

/-- A cube with a side length of 6 feet from which a cylindrical section with radius 3 feet 
and height 6 feet is removed has the remaining volume 216 - 54 * π cubic feet. -/
theorem remaining_volume_of_cube (π : ℝ) : 
  let side_length := 6
      radius := 3
      height := side_length
      volume_cube := side_length ^ 3
      volume_cylinder := π * radius ^ 2 * height
  in volume_cube - volume_cylinder = 216 - 54 * π := 
by
  -- Definitions to use in the remaining proof.
  let side_length := 6
  let radius := 3
  let height := side_length
  let volume_cube := side_length ^ 3
  let volume_cylinder := π * radius ^ 2 * height

  -- Proof (ommitted)
  sorry

end remaining_volume_of_cube_l248_248884


namespace sqrt_factorial_product_l248_248071

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l248_248071


namespace isosceles_triangle_angle_bisector_tangent_l248_248588

theorem isosceles_triangle_angle_bisector_tangent
  (A B C : Point)
  (circumcircle : Circle)
  (angle_bisector : Line) -- Angle bisector of \(\angle ACB\)
  (tangent : Line) -- Tangent line at point \(C\) to the circumcircle
  (h1 : circumcircle ∋ A)
  (h2 : circumcircle ∋ B)
  (h3 : circumcircle ∋ C)
  (h4 : angle_bisector.bisects_angle A C B)
  (h5 : tangent.is_tangent_at C circumcircle)
  (intersection_D : Point := angle_bisector ∩ Line(A, B))
  (intersection_E : Point := tangent ∩ Line(A, B)) :
  is_triangle isosceles (triangle intersection_D C intersection_E) := 
sorry

end isosceles_triangle_angle_bisector_tangent_l248_248588


namespace battery_current_at_given_resistance_l248_248490

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248490


namespace find_angle_D_l248_248000

theorem find_angle_D (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 50) (h4 : ∃ B_adj, B_adj = 60 ∧ A + B_adj + B = 180) : D = 25 :=
sorry

end find_angle_D_l248_248000


namespace sqrt_factorial_product_eq_24_l248_248189

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l248_248189


namespace current_value_l248_248424

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248424


namespace sqrt_factorial_multiplication_l248_248182

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l248_248182


namespace arithmetic_sequence_a1_a9_l248_248626

variable (a : ℕ → ℝ)

-- This statement captures if given condition holds, prove a_1 + a_9 = 18.
theorem arithmetic_sequence_a1_a9 (h : a 4 + a 5 + a 6 = 27)
    (h_seq : ∀ (n : ℕ), a (n + 1) = a n + (a 2 - a 1)) :
    a 1 + a 9 = 18 :=
sorry

end arithmetic_sequence_a1_a9_l248_248626


namespace current_when_resistance_is_12_l248_248402

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248402


namespace no_three_parabolas_l248_248961

theorem no_three_parabolas (a b c : ℝ) : ¬ (b^2 > 4*a*c ∧ a^2 > 4*b*c ∧ c^2 > 4*a*b) := by
  sorry

end no_three_parabolas_l248_248961


namespace range_of_f_l248_248995

theorem range_of_f (a x : ℝ) (h : a ∈ set.Icc (-1) 1) :
  (f x = x^2 + (a-4) * x + 4 - 2 * a) → 
  (∀ x, f x > 0 ↔ x < 1 ∨ x > 2) :=
by 
  sorry

end range_of_f_l248_248995


namespace sqrt_factorial_mul_factorial_eq_l248_248149

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l248_248149


namespace log_expression_is_zero_l248_248606

noncomputable def log_expr : ℝ := (Real.logb 2 3 + Real.logb 2 27) * (Real.logb 4 4 + Real.logb 4 (1/4))

theorem log_expression_is_zero : log_expr = 0 :=
by
  sorry

end log_expression_is_zero_l248_248606


namespace marsha_remainder_l248_248757

-- Definitions based on problem conditions
def a (n : ℤ) : ℤ := 90 * n + 84
def b (m : ℤ) : ℤ := 120 * m + 114
def c (p : ℤ) : ℤ := 150 * p + 144

-- Proof statement
theorem marsha_remainder (n m p : ℤ) : ((a n + b m + c p) % 30) = 12 :=
by 
  -- Notice we need to add the proof steps here
  sorry 

end marsha_remainder_l248_248757


namespace find_principal_continuous_compounding_l248_248900

theorem find_principal_continuous_compounding (A : ℝ) (P : ℝ) (r t : ℝ) (e : ℝ) : 
  A = P * Real.exp(r * t) → A = 3087 → r = 0.05 → t = 2 → e = Real.exp 1 → 
  P ≈ 2793.57 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_principal_continuous_compounding_l248_248900


namespace current_at_resistance_12_l248_248542

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248542


namespace sqrt_factorial_multiplication_l248_248181

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l248_248181


namespace current_when_resistance_is_12_l248_248415

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248415


namespace additional_trams_proof_l248_248044

-- Definitions for the conditions
def initial_tram_count : Nat := 12
def total_distance : Nat := 60
def initial_interval : Nat := total_distance / initial_tram_count
def reduced_interval : Nat := initial_interval - (initial_interval / 5)
def final_tram_count : Nat := total_distance / reduced_interval
def additional_trams_needed : Nat := final_tram_count - initial_tram_count

-- The theorem we need to prove
theorem additional_trams_proof : additional_trams_needed = 3 :=
by
  sorry

end additional_trams_proof_l248_248044


namespace current_at_R_12_l248_248256

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248256


namespace extreme_values_of_function_l248_248988

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 5

theorem extreme_values_of_function :
  ∃ (x_max x_min : ℝ), f x_max = 10 ∧ f x_min = -22 ∧ (∀ x, f x_max ≥ f x) ∧ (∀ x, f x_min ≤ f x) :=
by
  have h_deriv : (deriv f) x = 3 * x^2 - 6 * x - 9,
  { 
    sorry 
  },
  have h_critical_points : ∃ (x₁ x₂ : ℝ), (deriv f) x₁ = 0 ∧ (deriv f) x₂ = 0 ∧ x₁ = -1 ∧ x₂ = 3,
  { 
    sorry 
  },
  use [-1, 3],
  split,
  {
    show f (-1) = 10,
    { 
      sorry 
    }
  },
  split,
  {
    show f 3 = -22,
    { 
      sorry 
    }
  },
  split,
  {
    show ∀ x, f (-1) ≥ f x,
    {
      sorry 
    }
  },
  {
    show ∀ x, f 3 ≤ f x,
    {
      sorry 
    }
  }

end extreme_values_of_function_l248_248988


namespace sqrt_factorial_mul_factorial_l248_248208

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l248_248208


namespace current_value_l248_248401

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248401


namespace battery_current_l248_248324

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248324


namespace triangle_area_l248_248721

-- Definitions for sides and angles of a triangle
variables {A B C : ℝ} {a b c : ℝ}
-- Given conditions
variable (h1 : b = a * Real.cos C + c * Real.cos B)
variable (h2 : ⟪vector.ofReal (C - A), vector.ofReal (C - B)⟫ = 1)
variable (h3 : c = 2)

-- The theorem statement
theorem triangle_area (A B C a b c : ℝ)
  (h1 : b = a * Real.cos C + c * Real.cos B)
  (h2 : ⟪vector.ofReal (C - A), vector.ofReal (C - B)⟫ = 1)
  (h3 : c = 2) :
  (1 / 2) * a * b * Real.sin C = Real.sqrt 2 := 
by
  -- Proof placeholder
  sorry

end triangle_area_l248_248721


namespace find_current_l248_248438

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248438


namespace current_when_resistance_12_l248_248469

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248469


namespace sqrt_factorial_product_l248_248092

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l248_248092


namespace current_value_l248_248507

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248507


namespace parallelepiped_length_l248_248916

theorem parallelepiped_length (n : ℕ)
  (h1 : ∃ n : ℕ, n = 18) 
  (h2 : one_third_of_cubes_have_red_faces : (∃ k : ℕ, k = ((n * (n - 2) * (n - 4)) / 3)) 
        ∧ (remaining_unpainted_cubes : (∃ m : ℕ , m = (2 * (n * (n - 2) * (n - 4)) / 3))))
  (h3 : painted_and_cut_into_cubes : (∃ a b c : ℕ, a = n ∧ b = (n - 2) ∧ c = (n - 4)))
  (h4 : all_sides_whole_cm : (∃ d : ℕ , d = n ∧ d = (n - 2) ∧ d = (n - 4))) :
  n = 18 :=
begin
  sorry
end

end parallelepiped_length_l248_248916


namespace cricket_team_matches_l248_248695

theorem cricket_team_matches 
  (M : ℕ) (W : ℕ) 
  (h1 : W = 20 * M / 100) 
  (h2 : (W + 80) * 100 = 52 * M) : 
  M = 250 :=
by
  sorry

end cricket_team_matches_l248_248695


namespace correct_line_equation_l248_248571

/-
  Let P₂(n, 0) be a point through which a line passes, and let the slope of the line exist and be non-zero.
  We want to prove that the line can be represented by the equation x = ny + n.
-/
theorem correct_line_equation (n : ℝ) (k : ℝ) (h : k ≠ 0) :
  ∃ (f : ℝ → ℝ), f(0) = n ∧ ∀ y : ℝ, f(y) = ny + n :=
sorry

end correct_line_equation_l248_248571


namespace battery_current_l248_248322

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248322


namespace simplify_expression_l248_248775

theorem simplify_expression : 
  (3.875 * (1 / 5) + (38 + 3 / 4) * 0.09 - 0.155 / 0.4) / 
  (2 + 1 / 6 + (((4.32 - 1.68 - (1 + 8 / 25)) * (5 / 11) - 2 / 7) / (1 + 9 / 35)) + (1 + 11 / 24))
  = 1 := sorry

end simplify_expression_l248_248775


namespace some_number_value_l248_248849

theorem some_number_value (some_number : ℝ) (h : (some_number * 14) / 100 = 0.045388) :
  some_number = 0.3242 :=
sorry

end some_number_value_l248_248849


namespace x_y_difference_is_perfect_square_l248_248733

theorem x_y_difference_is_perfect_square (x y : ℕ) (h : 3 * x^2 + x = 4 * y^2 + y) : ∃ k : ℕ, k^2 = x - y :=
by {sorry}

end x_y_difference_is_perfect_square_l248_248733


namespace decrease_percent_is_23_l248_248809

-- Defining the initial conditions
variables {T C : ℝ}

-- Defining the initial revenue
def original_revenue : ℝ := T * C

-- Defining new tax and new consumption based on the given conditions
def new_tax : ℝ := 0.70 * T
def new_consumption : ℝ := 1.10 * C

-- Calculating the new revenue based on the new tax and consumption
def new_revenue : ℝ := new_tax * new_consumption

-- Calculating the decrease in revenue
def decrease_in_revenue : ℝ := original_revenue - new_revenue

-- Calculating the percentage decrease in revenue
def decrease_percent : ℝ := (decrease_in_revenue / original_revenue) * 100

-- Theorem statement: The decrease percent in the revenue is 23%
theorem decrease_percent_is_23 : decrease_percent = 23 := 
by sorry

end decrease_percent_is_23_l248_248809


namespace prob1_converse_prob1_inverse_prob1_contrapositive_prob2_converse_prob2_inverse_prob2_contrapositive_l248_248216

-- Problem 1: Original proposition converse, inverse, contrapositive
theorem prob1_converse (x y : ℝ) (h : x = 0 ∨ y = 0) : x * y = 0 :=
sorry

theorem prob1_inverse (x y : ℝ) (h : x * y ≠ 0) : x ≠ 0 ∧ y ≠ 0 :=
sorry

theorem prob1_contrapositive (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) : x * y ≠ 0 :=
sorry

-- Problem 2: Original proposition converse, inverse, contrapositive
theorem prob2_converse (x y : ℝ) (h : x * y > 0) : x > 0 ∧ y > 0 :=
sorry

theorem prob2_inverse (x y : ℝ) (h : x ≤ 0 ∨ y ≤ 0) : x * y ≤ 0 :=
sorry

theorem prob2_contrapositive (x y : ℝ) (h : x * y ≤ 0) : x ≤ 0 ∨ y ≤ 0 :=
sorry

end prob1_converse_prob1_inverse_prob1_contrapositive_prob2_converse_prob2_inverse_prob2_contrapositive_l248_248216


namespace parallelepiped_length_l248_248915

theorem parallelepiped_length (n : ℕ)
  (h1 : ∃ n : ℕ, n = 18) 
  (h2 : one_third_of_cubes_have_red_faces : (∃ k : ℕ, k = ((n * (n - 2) * (n - 4)) / 3)) 
        ∧ (remaining_unpainted_cubes : (∃ m : ℕ , m = (2 * (n * (n - 2) * (n - 4)) / 3))))
  (h3 : painted_and_cut_into_cubes : (∃ a b c : ℕ, a = n ∧ b = (n - 2) ∧ c = (n - 4)))
  (h4 : all_sides_whole_cm : (∃ d : ℕ , d = n ∧ d = (n - 2) ∧ d = (n - 4))) :
  n = 18 :=
begin
  sorry
end

end parallelepiped_length_l248_248915


namespace sum_first_2019_terms_l248_248030

noncomputable def S : ℕ → ℝ 
| 0       := 0
| (n + 1) := S n + (2 * (n + 1) - 1) * Real.sin ((n + 1) * Real.pi / 2 + 2019 * Real.pi)

theorem sum_first_2019_terms : S 2019 = 2020 :=
sorry

end sum_first_2019_terms_l248_248030


namespace parallelepiped_inequality_equality_condition_l248_248714

-- Definitions for vectors and their properties
variables {V : Type} [InnerProductSpace ℝ V]

-- Define the vectors corresponding to the problem's parallelepiped
variables (b d e : V)

-- The main statement to be proved in Lean
theorem parallelepiped_inequality (hb : b ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) : 
  ∥e + b∥ + ∥d + e∥ + ∥b + d + e∥ ≤ ∥b∥ + ∥d∥ + ∥e∥ + ∥b + d∥ :=
sorry

-- The condition for equality
theorem equality_condition (hb : b ≠ 0) (hd : d ≠ 0) (he : e ≠ 0) (h_eq : ∥e + b∥ + ∥d + e∥ + ∥b + d + e∥ = ∥b∥ + ∥d∥ + ∥e∥ + ∥b + d∥) :
  ∃ k : ℝ, b = k • d ∧ b = k • e :=
sorry

end parallelepiped_inequality_equality_condition_l248_248714


namespace battery_current_l248_248290

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248290


namespace constant_term_polynomial_expansion_l248_248064

def polynomial1 := (x^6 + 3*x^2 + 7)
def polynomial2 := (2*x^4 + x^3 + 11)

theorem constant_term_polynomial_expansion : 
  let poly1 := 7 in 
  let poly2 := 11 in 
  poly1 * poly2 = 77 := 
by
  sorry

end constant_term_polynomial_expansion_l248_248064


namespace total_revenue_correct_l248_248780

-- Define the costs of different types of returns
def cost_federal : ℕ := 50
def cost_state : ℕ := 30
def cost_quarterly : ℕ := 80

-- Define the quantities sold for different types of returns
def qty_federal : ℕ := 60
def qty_state : ℕ := 20
def qty_quarterly : ℕ := 10

-- Calculate the total revenue for the day
def total_revenue : ℕ := (cost_federal * qty_federal) + (cost_state * qty_state) + (cost_quarterly * qty_quarterly)

-- The theorem stating the total revenue calculation
theorem total_revenue_correct : total_revenue = 4400 := by
  sorry

end total_revenue_correct_l248_248780


namespace percentage_reduction_of_faculty_l248_248561

noncomputable def percentage_reduction (original reduced : ℝ) : ℝ :=
  ((original - reduced) / original) * 100

theorem percentage_reduction_of_faculty :
  percentage_reduction 226.74 195 = 13.99 :=
by sorry

end percentage_reduction_of_faculty_l248_248561


namespace prove_alpha1_prove_alpha2_prove_alpha3_prove_alpha4_l248_248237

noncomputable def alpha1 (k : ℤ) : ℝ := real.arctan (24/7) + 2 * real.pi * k
noncomputable def alpha2 (k : ℤ) : ℝ := real.pi + 2 * real.pi * k
noncomputable def alpha3 (k : ℤ) : ℝ := real.arctan ((4 + 3 * real.sqrt 24) / (4 * real.sqrt 24 - 3)) + 2 * real.pi * k
noncomputable def alpha4 (k : ℤ) : ℝ := real.arctan ((3 * real.sqrt 24 - 4) / (4 * real.sqrt 24 + 3)) + 2 * real.pi * k

theorem prove_alpha1 (k : ℤ) : alpha1 k ≡ real.arctan (24/7) [ZMOD 2 * real.pi] := sorry
theorem prove_alpha2 (k : ℤ) : alpha2 k ≡ real.pi [ZMOD 2 * real.pi] := sorry
theorem prove_alpha3 (k : ℤ) : alpha3 k ≡ real.arctan ((4 + 3 * real.sqrt 24) / (4 * real.sqrt 24 - 3)) [ZMOD 2 * real.pi] := sorry
theorem prove_alpha4 (k : ℤ) : alpha4 k ≡ real.arctan ((3 * real.sqrt 24 - 4) / (4 * real.sqrt 24 + 3)) [ZMOD 2 * real.pi] := sorry

end prove_alpha1_prove_alpha2_prove_alpha3_prove_alpha4_l248_248237


namespace triangle_proof_l248_248718

noncomputable def Triangle (α β γ : ℕ) : Prop :=
AB > AC

noncomputable def Circumcircle (ABC : Prop) : Prop :=
∃ E, ExternalAngleBisector(∠A) ∧ MeetsCircumcircle(E) ∧
  (∃ F, Perpendicular(E, AB, F))

/- Conditions -/
variables {α β γ : ℕ}
variables (AB AC AF : ℝ)
variables (ABC : Triangle α β γ)
variables (Circum : Circumcircle ABC)

/- Theorem Statement -/
theorem triangle_proof (h : Circum) : 2 * AF = AB - AC := sorry

end triangle_proof_l248_248718


namespace trig_identity_l248_248691

noncomputable def triangle_lengths (DE DF EF : ℝ) := DE = 7 ∧ DF = 8 ∧ EF = 5

theorem trig_identity :
  ∀ (D E F : ℝ), triangle_lengths 7 8 5 →
    ( (cos ((D - E) / 2)) / (sin (F / 2)) - (sin ((D - E) / 2)) / (cos (F / 2)) ) = 16 / 7 :=
by
  intros D E F h
  sorry

end trig_identity_l248_248691


namespace probability_tina_changes_twenty_dollar_bill_l248_248057

theorem probability_tina_changes_twenty_dollar_bill :
  let toys := list.range 10 |>.map (λ n, (n + 1) * 50)
  let favorite_toy_price := 400
  let tina_quarters := 12
  let total_permutations := (10.fact : ℚ)
  let favorable_permutations := (4.fact * 6.fact : ℚ)
  let probability_direct_purchase := favorable_permutations / total_permutations
  let required_change_probability := 1 - probability_direct_purchase
  required_change_probability = (999802 / 1000000 : ℚ) :=
by
  let toys := list.range 10 |>.map (λ n, (n + 1) * 50)
  let favorite_toy_price := 400
  let tina_quarters := 12
  let total_permutations := (10.fact : ℚ)
  let favorable_permutations := (4.fact * 6.fact : ℚ)
  let probability_direct_purchase := favorable_permutations / total_permutations
  let required_change_probability := 1 - probability_direct_purchase
  sorry

end probability_tina_changes_twenty_dollar_bill_l248_248057


namespace current_value_l248_248387

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248387


namespace battery_current_l248_248313

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248313


namespace current_value_l248_248433

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248433


namespace slope_of_line_l248_248029

theorem slope_of_line (x y : ℝ) : 
  (∃ (m : ℝ), (∀ (x y : ℝ), x - sqrt 3 * y - sqrt 3 = 0 → y = m * x + b ∧ m = sqrt 3 / 3 ∧ (tan 30) = (sqrt 3 / 3)) → ∀ m (h : m = sqrt 3 / 3), tan 30 = m := 
begin 
  sorry 
end

end slope_of_line_l248_248029


namespace current_at_R_12_l248_248251

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248251


namespace sqrt_factorial_mul_factorial_l248_248202

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l248_248202


namespace grasshopper_can_reach_any_point_l248_248228

variables {n : ℕ}
variables (A : Fin n → ℝ × ℝ) (B : Fin n → ℝ × ℝ)

-- Define the predicate that checks if two segments intersect
def segments_do_not_intersect (S1 S2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

-- Define a predicate for the condition that segments A_iB_i do not intersect any A_kB_k where k ≠ i, j
def segments_not_intersecting (A B : Fin n → ℝ × ℝ) : Prop :=
  ∀ i j k : Fin n, i ≠ j → i ≠ k → j ≠ k →
  segments_do_not_intersect (A i, B i) (A k, B k)

-- Define predicate for the grasshopper's ability to jump from A_i to A_j
def can_jump (A : Fin n → ℝ × ℝ) (B : Fin n → ℝ × ℝ) (i j : Fin n) : Prop :=
  i ≠ j ∧ ∀ k : Fin n, k ≠ i ∧ k ≠ j → segments_do_not_intersect (A i, A j) (A k, B k)

-- Main theorem statement
theorem grasshopper_can_reach_any_point (A B : Fin n → ℝ × ℝ) (h : segments_not_intersecting A B) :
  ∀ (p q : Fin n), ∃ (k : Fin n → Fin n), k 0 = p ∧ k (n - 1) = q ∧
    ∀ i : Fin (n-1), can_jump A B (k i) (k (i + 1)) :=
sorry

end grasshopper_can_reach_any_point_l248_248228


namespace wooden_parallelepiped_length_l248_248929

theorem wooden_parallelepiped_length (n : ℕ) (h1 : n ≥ 7)
    (h2 : ∀ total_cubes unpainted_cubes : ℕ,
      total_cubes = n * (n - 2) * (n - 4) ∧
      unpainted_cubes = (n - 2) * (n - 4) * (n - 6) ∧
      unpainted_cubes = 2 / 3 * total_cubes) :
  n = 18 := 
sorry

end wooden_parallelepiped_length_l248_248929


namespace sqrt_factorial_mul_factorial_eq_l248_248150

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l248_248150


namespace current_when_resistance_is_12_l248_248413

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248413


namespace cost_of_iphone_l248_248582

def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80
def weeks_worked : ℕ := 7
def total_earnings := weekly_earnings * weeks_worked
def total_money := total_earnings + trade_in_value
def new_iphone_cost : ℕ := 800

theorem cost_of_iphone :
  total_money = new_iphone_cost := by
  sorry

end cost_of_iphone_l248_248582


namespace battery_current_l248_248277

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248277


namespace sum_of_areas_l248_248867

-- Define the initial conditions of the right-angled quadrilateral
def right_angled_quadrilateral {α : Type*} [linear_ordered_field α] (AB AD : α) :=
AB < AD ∧ ∃ M, M = AD / 2 ∧ ∃ N, N = AB / 2

-- Define the similarity condition for folding
def similar_on_folding {α : Type*} [linear_ordered_field α] (AB AD : α) :=
(AB / AD) = (AD / (2 * AB))

-- Define the similarity condition for cutting off a square
def similar_on_cutting_square {α : Type*} [linear_ordered_field α] (AB AD : α) :=
(AB / AD) = ((AD - AB) / AB)

-- The final theorem statement
theorem sum_of_areas {α : Type*} [linear_ordered_field α] (AB AD : α) :
  right_angled_quadrilateral AB AD 
  ∧ similar_on_folding AB AD 
  ∧ similar_on_cutting_square AB AD 
  → ∃ T, T = AD^2 :=
by
  intros h,
  sorry

end sum_of_areas_l248_248867


namespace battery_current_l248_248340

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248340


namespace current_value_l248_248270

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248270


namespace battery_current_l248_248281

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248281


namespace F_of_3153_max_value_of_N_l248_248994

-- Define friendly number predicate
def is_friendly (M : ℕ) : Prop :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  a - b = c - d

-- Define F(M)
def F (M : ℕ) : ℕ :=
  let a := M / 1000
  let b := (M / 100) % 10
  let s := M / 10
  let t := M % 1000
  s - t - 10 * b

-- Prove F(3153) = 152
theorem F_of_3153 : F 3153 = 152 :=
by sorry

-- Define the given predicate for N
def is_k_special (N : ℕ) : Prop :=
  let x := N / 1000
  let y := (N / 100) % 10
  let m := (N / 30) % 10
  let n := N % 10
  (N % 5 = 1) ∧ (1000 * x + 100 * y + 30 * m + n + 1001 = N) ∧
  (0 ≤ y ∧ y < x ∧ x ≤ 8) ∧ (0 ≤ m ∧ m ≤ 3) ∧ (0 ≤ n ∧ n ≤ 8) ∧ 
  is_friendly N

-- Prove the maximum value satisfying the given constraints
theorem max_value_of_N : ∀ N, is_k_special N → N ≤ 9696 :=
by sorry

end F_of_3153_max_value_of_N_l248_248994


namespace impossible_to_arrange_circle_l248_248960

theorem impossible_to_arrange_circle : 
  ¬∃ (f : Fin 10 → Fin 10), 
    (∀ i : Fin 10, (abs ((f i).val - (f (i + 1)).val : Int) = 3 
                ∨ abs ((f i).val - (f (i + 1)).val : Int) = 4 
                ∨ abs ((f i).val - (f (i + 1)).val : Int) = 5)) :=
sorry

end impossible_to_arrange_circle_l248_248960


namespace current_value_l248_248505

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248505


namespace area_difference_l248_248787

noncomputable def π := 3.14159

theorem area_difference 
  (d_square d_circle : ℝ) 
  (h₁ : d_square = 10) 
  (h₂ : d_circle = 10) : 
  abs ((π * ((d_circle / 2) ^ 2)) - ((d_square / (sqrt 2)) ^ 2)) = 28.5 :=
by
  sorry

end area_difference_l248_248787


namespace sqrt_factorial_product_l248_248165

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248165


namespace current_value_l248_248429

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248429


namespace length_of_parallelepiped_l248_248923

def number_of_cubes_with_painted_faces (n : ℕ) := (n - 2) * (n - 4) * (n - 6) 
def total_number_of_cubes (n : ℕ) := n * (n - 2) * (n - 4)

theorem length_of_parallelepiped (n : ℕ) (h1 : total_number_of_cubes n = 3 * number_of_cubes_with_painted_faces n) : 
  n = 18 :=
by 
  sorry

end length_of_parallelepiped_l248_248923


namespace find_current_when_resistance_is_12_l248_248349

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248349


namespace current_when_resistance_12_l248_248458

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248458


namespace length_of_second_train_l248_248826

variables (length_first_train speed_first_train_km_hr speed_second_train_km_hr time_cross_seconds : Real)

def speed_first_train_m_s := speed_first_train_km_hr * (1000/3600)
def speed_second_train_m_s := speed_second_train_km_hr * (1000/3600)
def relative_speed := speed_first_train_m_s + speed_second_train_m_s

/-- Given the length of the first train, the speeds of the two trains, and the time to cross each other,
    the length of the second train is 160 meters. -/
theorem length_of_second_train : 
  length_first_train = 140 → 
  speed_first_train_km_hr = 60 → 
  speed_second_train_km_hr = 40 → 
  time_cross_seconds = 10.799136069114471 → 
  let L := (relative_speed * time_cross_seconds) - length_first_train in 
  L = 160 :=
by
  sorry

end length_of_second_train_l248_248826


namespace townX_employed_females_percentage_of_employed_l248_248858

def townX_employed_percentage : ℝ := 0.96
def townX_employed_males_percentage : ℝ := 0.24

theorem townX_employed_females_percentage_of_employed :
  let employed_females_percentage := townX_employed_percentage - townX_employed_males_percentage in
  employed_females_percentage / townX_employed_percentage * 100 = 75 :=
by
  let employed_females_percentage := townX_employed_percentage - townX_employed_males_percentage
  have h : employed_females_percentage = 0.72, by linarith
  have h_pct : employed_females_percentage / townX_employed_percentage * 100 = 75, by
    rw h
    field_simp [employed_females_percentage, townX_employed_percentage]
    norm_num
  exact h_pct
  

end townX_employed_females_percentage_of_employed_l248_248858


namespace sqrt_factorial_eq_l248_248130

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l248_248130


namespace sqrt_factorial_mul_factorial_l248_248085

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l248_248085


namespace probability_AB_together_l248_248802

theorem probability_AB_together : 
  let total_events := 6
  let ab_together_events := 4
  let probability := ab_together_events / total_events
  probability = 2 / 3 :=
by
  sorry

end probability_AB_together_l248_248802


namespace pyramid_cross_section_area_eq_sum_l248_248717

def cross_section_area (p q : ℝ) (alpha : ℝ) (inscribed_center : Prop) : ℝ :=
  p + q

theorem pyramid_cross_section_area_eq_sum (ABCD : Type)
  (ABC ABD : ABCD → Prop)
  (p q : ℝ)
  (alpha : ℝ)
  (inscribed_center : Prop)
  (area_ABC : ABC abc → area = p)
  (area_ABD : ABD abd → area = q)
  (angle_ABC_ABD : angle ABC ABD = alpha) :
  cross_section_area p q alpha inscribed_center = p + q :=
sorry

end pyramid_cross_section_area_eq_sum_l248_248717


namespace solve_integral_l248_248620

noncomputable def integral_problem : Prop :=
  ∫ x, (x + 2) / (x^2 + 2*x + 5) = 
  (1 / 2) * Real.log (x^2 + 2*x + 5) + 
  (1 / 2) * Real.arctan ((x + 1) / 2) + C

theorem solve_integral : integral_problem :=
  sorry

end solve_integral_l248_248620


namespace garden_area_l248_248019

theorem garden_area (w l : ℕ) (h1 : l = 3 * w + 30) (h2 : 2 * (w + l) = 780) : 
  w * l = 27000 := 
by 
  sorry

end garden_area_l248_248019


namespace joel_donated_22_toys_l248_248727

-- Given conditions
variables (T : ℕ)
variables (stuffed_animals action_figures board_games puzzles : ℕ)
variables (total_donated : ℕ)

-- Define the conditions
def conditions := 
  stuffed_animals = 18 ∧ 
  action_figures = 42 ∧ 
  board_games = 2 ∧ 
  puzzles = 13 ∧ 
  total_donated = 108

-- Calculating the total number of toys from friends
def friends_total := 
  stuffed_animals + action_figures + board_games + puzzles

-- The total number of toys Joel and his sister donated
def total_joel_sister := 
  T + 2 * T

-- The proof problem
theorem joel_donated_22_toys 
  (h : conditions) : 
  3 * T + friends_total = total_donated → 2 * T = 22 :=
by
  intros
  sorry

end joel_donated_22_toys_l248_248727


namespace sqrt_factorial_product_l248_248100

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l248_248100


namespace TrianglePyramid_volume_l248_248703

-- Define the necessary conditions
variables (a : Real) (A B C D : Point)
variable (TrianglePyramid : ℝ → Point → Point → Point → Prop)

-- Condition specifications
def EdgeEqual (A B : Point) (a : ℝ) : Prop :=
  dist A B = a

def RightAngle (D B : Point) (a : ℝ) : Prop :=
  dist D B = a

def PlaneAngles :
  ∀ {A B C D : Point}, TrianglePyramid a A B C D → 
  ∠ B D C = π / 2 ∧ ∠ A D B = π / 3 ∧ ∠ A D C = π / 3

-- Volume of the pyramid proof statement
theorem TrianglePyramid_volume :
  ∀ (a : ℝ) (A B C D : Point),
  EdgeEqual A D a ∧ EdgeEqual B D a ∧ EdgeEqual C D a ∧
  PlaneAngles a A B C D →
  volume A B C D = a^3 * Real.sqrt 6 / 12 :=
sorry

end TrianglePyramid_volume_l248_248703


namespace current_value_l248_248504

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248504


namespace current_value_l248_248503

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248503


namespace current_when_resistance_12_l248_248460

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248460


namespace current_at_resistance_12_l248_248544

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248544


namespace shared_bill_per_person_l248_248810

noncomputable def totalBill : ℝ := 139.00
noncomputable def tipPercentage : ℝ := 0.10
noncomputable def totalPeople : ℕ := 5

theorem shared_bill_per_person :
  let tipAmount := totalBill * tipPercentage
  let totalBillWithTip := totalBill + tipAmount
  let amountPerPerson := totalBillWithTip / totalPeople
  amountPerPerson = 30.58 :=
by
  let tipAmount := totalBill * tipPercentage
  let totalBillWithTip := totalBill + tipAmount
  let amountPerPerson := totalBillWithTip / totalPeople
  have h1 : tipAmount = 13.90 := by sorry
  have h2 : totalBillWithTip = 152.90 := by sorry
  have h3 : amountPerPerson = 30.58 := by sorry
  exact h3

end shared_bill_per_person_l248_248810


namespace current_at_resistance_12_l248_248539

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248539


namespace arithmetic_sequence_terms_l248_248972

theorem arithmetic_sequence_terms (a d n : ℤ) (last_term : ℤ)
  (h_a : a = 5)
  (h_d : d = 3)
  (h_last_term : last_term = 149)
  (h_n_eq : last_term = a + (n - 1) * d) :
  n = 49 :=
by sorry

end arithmetic_sequence_terms_l248_248972


namespace num_selections_avoid_arithmetic_seq_l248_248770

open Finset

def select_avoid_arithmetic_seq : ℕ :=
  let total_selections := (10.choose 3) in
  let total_arithmetic_seqs := 8 + 6 + 4 + 2 in
  total_selections - total_arithmetic_seqs

theorem num_selections_avoid_arithmetic_seq : select_avoid_arithmetic_seq = 100 :=
by
  sorry

end num_selections_avoid_arithmetic_seq_l248_248770


namespace min_value_of_cos2_plus_sin_in_interval_l248_248021

noncomputable
def minValueFunction : ℝ :=
  let f : ℝ → ℝ := λ x, (Real.cos x) ^ 2 + Real.sin x
  let interval : Set ℝ := Set.interval (-Real.pi / 4) (Real.pi / 4)
  let minimum_value : ℝ := (1 - Real.sqrt 2) / 2
  if f ∈ continuous_on interval then minimum_value else sorry

theorem min_value_of_cos2_plus_sin_in_interval :
  let f : ℝ → ℝ := λ x, (Real.cos x) ^ 2 + Real.sin x
  let interval : Set ℝ := Set.interval (-Real.pi / 4) (Real.pi / 4)
  has_minimum_on f interval ((1 - Real.sqrt 2) / 2) :=
by
  sorry

end min_value_of_cos2_plus_sin_in_interval_l248_248021


namespace circle_line_no_intersection_l248_248656

theorem circle_line_no_intersection (b : ℝ) :
  (∀ x y : ℝ, ¬ (x^2 + y^2 = 2 ∧ y = x + b)) ↔ (b > 2 ∨ b < -2) :=
by sorry

end circle_line_no_intersection_l248_248656


namespace weight_of_bowling_ball_l248_248993

variable (b c : ℝ)

axiom h1 : 5 * b = 2 * c
axiom h2 : 3 * c = 84

theorem weight_of_bowling_ball : b = 11.2 :=
by
  sorry

end weight_of_bowling_ball_l248_248993


namespace probability_different_colors_l248_248872

theorem probability_different_colors :
  let total_ways := Nat.choose 4 2 
  let different_color_ways := 2 * 2 
  let probability := different_color_ways / total_ways
  probability = 2 / 3 := by
  -- Definitions from conditions
  have h1 : 4 = 4 := rfl -- 4 balls in total
  have h2 : 2 = 2 := rfl -- 2 red balls
  have h3 : 2 = 2 := rfl -- 2 white balls
  have h4 : Nat.choose 4 2 = 6 := by -- Number of ways to draw 2 balls from 4 balls
    exact Nat.choose_eq 4 2
  have h5 : 2 * 2 = 4 := by -- Number of ways to draw 2 balls of different colors
    norm_num
  have h6 : 4 / 6 = 2 / 3 := by -- Simplifying the fraction
    norm_num
  sorry -- The proof itself

end probability_different_colors_l248_248872


namespace sum_k_sqrt_k_plus_one_lt_one_sixth_n_n_plus_two_sq_l248_248636

theorem sum_k_sqrt_k_plus_one_lt_one_sixth_n_n_plus_two_sq (n : ℕ) (h : 1 < n) :
  (∑ k in Finset.range n, k * Real.sqrt (k + 1)) < (1 / 6) * n * (n + 2)^2 := 
by sorry

end sum_k_sqrt_k_plus_one_lt_one_sixth_n_n_plus_two_sq_l248_248636


namespace find_cone_radius_l248_248651

noncomputable def cone_radius : ℝ :=
  let θ : ℝ := Real.pi / 3
  let V : ℝ := 3 * Real.pi
  let h := λ r : ℝ, r * Real.tan θ
  let volume_formula := λ r : ℝ, (1 / 3) * Real.pi * r^2 * h r
  let r_sol := Real.sqrt 3
  r_sol

theorem find_cone_radius :
  let θ : ℝ := Real.pi / 3
  let V : ℝ := 3 * Real.pi
  let h := λ r : ℝ, r * Real.tan θ
  volume_formula r_sol = V :=
sorry

end find_cone_radius_l248_248651


namespace line_angle_l248_248829

noncomputable def inclination_angle (m : ℝ) : ℝ :=
real.arctan m

theorem line_angle : inclination_angle 1 = π / 4 := by
  sorry

end line_angle_l248_248829


namespace inverse_proportion_symmetry_l248_248630

theorem inverse_proportion_symmetry (a b : ℝ) :
  (b = - 6 / (-a)) → (-b = - 6 / a) :=
by
  intro h
  sorry

end inverse_proportion_symmetry_l248_248630


namespace company_employees_count_l248_248814

theorem company_employees_count :
  ∃ E : ℕ, E = 80 + 100 - 30 + 20 := 
sorry

end company_employees_count_l248_248814


namespace current_value_l248_248262

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248262


namespace find_higher_selling_price_l248_248938

def cost_price := 200
def selling_price_low := 340
def gain_low := selling_price_low - cost_price
def gain_high := gain_low + (5 / 100) * gain_low
def higher_selling_price := cost_price + gain_high

theorem find_higher_selling_price : higher_selling_price = 347 := 
by 
  sorry

end find_higher_selling_price_l248_248938


namespace halt_duration_l248_248009

theorem halt_duration (avg_speed : ℝ) (distance : ℝ) (start_time end_time : ℝ) (halt_duration : ℝ) :
  avg_speed = 87 ∧ distance = 348 ∧ start_time = 9 ∧ end_time = 13.75 →
  halt_duration = (end_time - start_time) - (distance / avg_speed) → 
  halt_duration = 0.75 :=
by
  sorry

end halt_duration_l248_248009


namespace sin_cos_solution_count_l248_248675

-- Statement of the problem
theorem sin_cos_solution_count : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin (3 * x) = Real.cos (x / 2)) ∧ s.card = 6 := by
  sorry

end sin_cos_solution_count_l248_248675


namespace current_when_resistance_is_12_l248_248417

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248417


namespace A_leaves_after_one_day_l248_248877

-- Define and state all the conditions
def A_work_rate := 1 / 21
def B_work_rate := 1 / 28
def C_work_rate := 1 / 35
def total_work := 1
def B_time_after_A_leave := 21
def C_intermittent_working_cycle := 3 / 1 -- C works 1 out of every 3 days

-- The statement that needs to be proved
theorem A_leaves_after_one_day :
  ∃ x : ℕ, x = 1 ∧
  (A_work_rate * x + B_work_rate * x + (C_work_rate * (x / C_intermittent_working_cycle)) + (B_work_rate * B_time_after_A_leave) + (C_work_rate * (B_time_after_A_leave / C_intermittent_working_cycle)) = total_work) :=
sorry

end A_leaves_after_one_day_l248_248877


namespace remainder_of_second_largest_division_l248_248848

theorem remainder_of_second_largest_division :
  ∃ (numbers : List Nat) (second_largest smallest : Nat), 
    numbers = [10, 11, 12, 13] ∧ 
    second_largest = numbers.max' none (≠ 13) ∧ 
    smallest = numbers.min' none ∧ 
    second_largest % smallest = 2 :=
by
  sorry

end remainder_of_second_largest_division_l248_248848


namespace same_function_l248_248935

noncomputable def f (x : ℝ) : ℝ := abs x
noncomputable def g (x : ℝ) : ℝ := real.sqrt (x ^ 2)

theorem same_function : ∀ x : ℝ, f x = g x :=
by {
  intro x,
  unfold f g,
  sorry
}

end same_function_l248_248935


namespace prince_spent_1000_l248_248952

noncomputable def total_CDs := 200
noncomputable def percent_CDs_10 := 0.40
noncomputable def price_per_CD_10 := 10
noncomputable def price_per_CD_5 := 5

-- Number of CDs sold at $10 each
noncomputable def num_CDs_10 := percent_CDs_10 * total_CDs

-- Number of CDs sold at $5 each
noncomputable def num_CDs_5 := total_CDs - num_CDs_10

-- Number of $10 CDs bought by Prince
noncomputable def prince_CDs_10 := num_CDs_10 / 2

-- Total cost of $10 CDs bought by Prince
noncomputable def cost_CDs_10 := prince_CDs_10 * price_per_CD_10

-- Total cost of $5 CDs bought by Prince
noncomputable def cost_CDs_5 := num_CDs_5 * price_per_CD_5

-- Total amount of money Prince spent
noncomputable def total_spent := cost_CDs_10 + cost_CDs_5

theorem prince_spent_1000 : total_spent = 1000 := by
  -- Definitions from conditions
  have h1 : total_CDs = 200 := rfl
  have h2 : percent_CDs_10 = 0.40 := rfl
  have h3 : price_per_CD_10 = 10 := rfl
  have h4 : price_per_CD_5 = 5 := rfl

  -- Calculations from solution steps (insert sorry to skip actual proofs)
  have h5 : num_CDs_10 = 80 := sorry
  have h6 : num_CDs_5 = 120 := sorry
  have h7 : prince_CDs_10 = 40 := sorry
  have h8 : cost_CDs_10 = 400 := sorry
  have h9 : cost_CDs_5 = 600 := sorry

  show total_spent = 1000
  sorry

end prince_spent_1000_l248_248952


namespace battery_current_at_given_resistance_l248_248474

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248474


namespace sqrt_factorial_product_l248_248095

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l248_248095


namespace battery_current_l248_248326

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248326


namespace parking_probability_l248_248563

noncomputable def total_ways_to_park (n m : ℕ) : ℕ :=
  Nat.choose n m

noncomputable def configurations_blocking_two_adjacent_spaces (remaining empty_spaces : ℕ) : ℕ :=
  Nat.choose (remaining - 2) (empty_spaces - 1)

def probability_unable_to_park (remaining empty_spaces : ℕ) : ℚ :=
  (configurations_blocking_two_adjacent_spaces remaining empty_spaces : ℚ) / 
  (total_ways_to_park 18 15 : ℚ)

theorem parking_probability : 
  let p_unable_to_park := probability_unable_to_park 15 3
  let p_able_to_park := 1 - p_unable_to_park
  p_able_to_park = 16 / 51 := by
  sorry

end parking_probability_l248_248563


namespace number_of_shirts_washed_l248_248596

variable (short_sleeve_shirts long_sleeve_shirts did_not_wash_shirts : ℕ)

theorem number_of_shirts_washed 
  (h1 : short_sleeve_shirts = 9)
  (h2 : long_sleeve_shirts = 27)
  (h3 : did_not_wash_shirts = 16) :
  (short_sleeve_shirts + long_sleeve_shirts - did_not_wash_shirts) = 20 :=
by
  rw [h1, h2, h3]
  norm_num

end number_of_shirts_washed_l248_248596


namespace sqrt_factorial_product_l248_248113

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248113


namespace cole_drive_time_to_work_is_72_minutes_l248_248965

-- Definitions based on the conditions
def avg_speed_home_to_work := 70  -- km/h
def avg_speed_work_to_home := 105 -- km/h
def total_round_trip_time := 2    -- hours

-- Define the distance from home to work as D
variable (D : ℝ)

-- Define the times based on average speeds
def time_home_to_work := D / avg_speed_home_to_work
def time_work_to_home := D / avg_speed_work_to_home

-- Total round trip time condition
def round_trip_time_eq : Prop := time_home_to_work + time_work_to_home = total_round_trip_time

-- Prove that the time it took Cole to drive to work is 72 minutes (1.2 hours)
theorem cole_drive_time_to_work_is_72_minutes : round_trip_time_eq → time_home_to_work * 60 = 72 :=
by
  sorry

end cole_drive_time_to_work_is_72_minutes_l248_248965


namespace sqrt_factorial_product_l248_248096

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l248_248096


namespace current_value_l248_248267

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248267


namespace sqrt_factorial_product_l248_248173

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248173


namespace current_at_R_12_l248_248255

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248255


namespace problem_statement_l248_248763

-- Definitions for the problem
def is_ellipse (M A B : Point) (a : ℝ) : Prop :=
  a > 0 ∧ dist M A + dist M B = 2 * a ∧ 2 * a > dist A B

def is_necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬ (P → Q)

-- Main statement of the problem
theorem problem_statement (M A B : Point) (a : ℝ) (h : is_ellipse M A B a) :
  is_necessary_not_sufficient (dist M A + dist M B = 2 * a) (∃ M, is_ellipse M A B a) :=
sorry

end problem_statement_l248_248763


namespace integer_root_counts_l248_248594

theorem integer_root_counts (a b c d e : ℤ) :
  let f : ℤ → ℤ := λ x, x^5 + a * x^4 + b * x^3 + c * x^2 + d * x + e
  let m := (finset.univ.filter (λ x : ℤ, f x = 0)).card
  m = 0 ∨ m = 1 ∨ m = 2 ∨ m = 5 :=
sorry

end integer_root_counts_l248_248594


namespace train_length_l248_248568

theorem train_length (time_to_cross : ℝ) (platform_length : ℝ) (train_speed_kmph : ℕ) (length_of_train : ℝ) :
  time_to_cross = 7.499400047996161 →
  platform_length = 165 →
  train_speed_kmph = 132 →
  (let train_speed_mps := (train_speed_kmph : ℝ) * (5 / 18) in
   let total_distance := platform_length + length_of_train in
   total_distance = train_speed_mps * time_to_cross →
   length_of_train = 110) := sorry

end train_length_l248_248568


namespace satisfies_functional_equation_l248_248983

noncomputable def f : ℝ → ℝ := sorry

theorem satisfies_functional_equation (f : ℝ → ℝ)
(h : ∀ x y : ℝ, f(x) * f(y) - f(x - 1) - f(y + 1) = f(x * y) + 2 * x - 2 * y - 4) :
∀ x : ℝ, f(x) = x^2 + 1 := sorry

end satisfies_functional_equation_l248_248983


namespace dot_product_range_l248_248678

variables {a b : EuclideanSpace ℝ}

theorem dot_product_range (ha : ‖a‖ = 7) (hb : ‖b‖ = 11) :
  a ⬝ b ∈ Set.Icc (-77 : ℝ) 77 :=
sorry

end dot_product_range_l248_248678


namespace sqrt_factorial_product_l248_248117

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l248_248117


namespace current_when_resistance_12_l248_248472

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248472


namespace current_value_l248_248273

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248273


namespace current_at_R_12_l248_248257

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248257


namespace who_finished_10th_l248_248696

noncomputable def position_of_diana (ben : ℕ) : ℕ := ben - 3
noncomputable def position_of_emma (diana : ℕ) : ℕ := diana + 4
noncomputable def position_of_fiona (emma : ℕ) : ℕ := emma - 2
noncomputable def position_of_carlos (fiona : ℕ) : ℕ := fiona - 3
noncomputable def position_of_alice (carlos : ℕ) : ℕ := carlos + 2

theorem who_finished_10th (ben : ℕ) (h_ben : ben = 7)
  (h_diana : position_of_diana ben = 4)
  (h_emma : position_of_emma (position_of_diana ben) = 8)
  (h_fiona : position_of_fiona (position_of_emma (position_of_diana ben)) = 10)
  (h_carlos : position_of_carlos (position_of_fiona (position_of_emma (position_of_diana ben))) = 6)
  (h_alice : position_of_alice (position_of_carlos (position_of_fiona (position_of_emma (position_of_diana ben)))) < 15) :
  ∃ fiona, fiona = 10 :=
by
  use 10
  exact h_fiona

end who_finished_10th_l248_248696


namespace inclination_angle_of_line_through_origin_and_neg_one_one_l248_248683

noncomputable def inclination_angle (p1 p2 : ℝ × ℝ) : ℝ :=
if p1.1 = p2.1 then π / 2 else Real.atan2 (p2.2 - p1.2) (p2.1 - p1.1)

theorem inclination_angle_of_line_through_origin_and_neg_one_one :
  inclination_angle (0, 0) (-1, 1) = 3 * π / 4 :=
by
  sorry

end inclination_angle_of_line_through_origin_and_neg_one_one_l248_248683


namespace diagonals_of_rectangle_are_equal_l248_248001

theorem diagonals_of_rectangle_are_equal (ABCD : Type) [IsRectangle ABCD] :
  (∀ (r : ABCD), (isRectangle r → diagonalsEqual r)) → (diagonalsEqual ABCD) :=
by
  intro H
  apply H
  exact IsRectangleClass ABCD
  sorry

-- Definitions needed for the theorem (Not complete proofs, just structuring the Lean statement context):
class IsRectangle (ABCD : Type) := (isRectangle : Prop)
def isRectangle (ABCD : Type) [IsRectangle ABCD] : Prop := IsRectangle.isRectangle
class diagonalsEqual (ABCD : Type) := (diagonalsEqual : Prop)
noncomputable def diagonalsEqual (ABCD : Type) [diagonalsEqual ABCD] : Prop := 
  diagonalsEqual.diagonalsEqual ABCD

end diagonals_of_rectangle_are_equal_l248_248001


namespace monotone_increasing_interval_l248_248022

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

theorem monotone_increasing_interval :
  ∀ x : ℝ, (1 < x) → (0 < x) → (f x > f 1) :=
by
  assume x hx1 hx0
  -- insert the proof here
  sorry

end monotone_increasing_interval_l248_248022


namespace current_value_l248_248432

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248432


namespace sum_of_powers_of_neg_one_l248_248958

theorem sum_of_powers_of_neg_one : ∑ k in Finset.range(39), (-1)^(k-19) = -1 :=
begin
  sorry
end

end sum_of_powers_of_neg_one_l248_248958


namespace current_value_l248_248436

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248436


namespace sqrt_factorial_mul_factorial_l248_248206

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l248_248206


namespace sqrt_factorial_mul_factorial_l248_248204

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l248_248204


namespace current_at_R_12_l248_248246

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248246


namespace Joel_contributed_22_toys_l248_248725

/-
Define the given conditions as separate variables and statements in Lean:
1. Toys collected from friends.
2. Total toys donated.
3. Relationship between Joel's and his sister's toys.
4. Prove that Joel donated 22 toys.
-/

theorem Joel_contributed_22_toys (S : ℕ) (toys_from_friends : ℕ) (total_toys : ℕ) (sisters_toys : ℕ) 
  (h1 : toys_from_friends = 18 + 42 + 2 + 13)
  (h2 : total_toys = 108)
  (h3 : S + 2 * S = total_toys - toys_from_friends)
  (h4 : sisters_toys = S) :
  2 * S = 22 :=
  sorry

end Joel_contributed_22_toys_l248_248725


namespace find_current_when_resistance_is_12_l248_248362

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248362


namespace susan_typing_time_l248_248730

def typing_rate (minutes : ℕ) (pages : ℕ) : ℚ :=
  pages / minutes

theorem susan_typing_time :
  ∀ (S : ℚ),
  (typing_rate 40 10 + typing_rate S 10 + typing_rate 24 10 = 1) →
  (S = 3) :=
begin
  intros S h,
  sorry
end

end susan_typing_time_l248_248730


namespace battery_current_l248_248289

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248289


namespace find_current_l248_248439

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248439


namespace height_of_taller_pot_l248_248823

-- Conditions in the problem
def h : ℝ := 20 -- height of the shorter pot
def s : ℝ := 10 -- shadow length of the shorter pot
def S : ℝ := 20 -- shadow length of the taller pot

-- Proving the height H of the taller pot is 40 inches
theorem height_of_taller_pot : (∃ H : ℝ, H / S = h / s) → (∃ H : ℝ, H = 40) := 
by 
  intro h_div_s_eq
  cases h_div_s_eq with H h_div_s_eq
  use H
  have ratio := h_div_s_eq
  have H_eq_40 : H = 2 * S := 
    by rw [h_div_s_eq, div_eq_iff, mul_comm]; norm_num; assumption

  rw H_eq_40
  norm_num
  sorry

end height_of_taller_pot_l248_248823


namespace length_of_parallelepiped_l248_248925

def number_of_cubes_with_painted_faces (n : ℕ) := (n - 2) * (n - 4) * (n - 6) 
def total_number_of_cubes (n : ℕ) := n * (n - 2) * (n - 4)

theorem length_of_parallelepiped (n : ℕ) (h1 : total_number_of_cubes n = 3 * number_of_cubes_with_painted_faces n) : 
  n = 18 :=
by 
  sorry

end length_of_parallelepiped_l248_248925


namespace problem_3_l248_248235

theorem problem_3 (x : Fin 2023 → ℝ) (h_nonneg : ∀ k : Fin 2023, 0 ≤ x k)
  (h_cond : ∀ k : Fin 2020, x k + x (k + 1) + x (k + 2) ≤ 2) :
  ∑ k in Finset.range 2020, x k * x (k + 2) ≤ 1010 := 
sorry

end problem_3_l248_248235


namespace battery_current_when_resistance_12_l248_248522

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248522


namespace sector_angle_l248_248684

theorem sector_angle
  (area : ℝ)
  (radius : ℝ)
  (h_area : area = 3 * Real.pi / 8)
  (h_radius : radius = 1) :
  ∃ α : ℝ, 
  α = 3 * Real.pi / 4 :=
by
  -- S_sector = (1 / 2) * α * R^2
  let S_sector := (1/2) * α * radius^2
  -- α = 3*pi/4
  sorry

end sector_angle_l248_248684


namespace current_when_resistance_12_l248_248466

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248466


namespace current_at_resistance_12_l248_248535

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248535


namespace total_pints_l248_248612

-- Define the given conditions as constants
def annie_picked : Int := 8
def kathryn_picked : Int := annie_picked + 2
def ben_picked : Int := kathryn_picked - 3

-- State the main theorem to prove
theorem total_pints : annie_picked + kathryn_picked + ben_picked = 25 := by
  sorry

end total_pints_l248_248612


namespace battery_current_l248_248339

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248339


namespace current_value_l248_248508

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248508


namespace sqrt_factorial_product_l248_248123

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l248_248123


namespace correct_statements_in_triangle_l248_248023

theorem correct_statements_in_triangle (a b c : ℝ) (A B C : ℝ) (S : ℝ) : 
  ((a > b) → (cos A < cos B)) ∧
  ((0 < A ∧ A < π / 2) → (b^2 + c^2 - a^2 > 0)) ∧
  ((C = π / 4) ∧ (a^2 - c^2 = b * c) → 
   (A = π / 2 ∧ B + C = π / 2 ∧ a = b * sqrt 2)) ∧
  ((b = 3) ∧ (A = π / 3) ∧ (S = 3 * sqrt 3) →
   ((sqrt 13) / (2 * sqrt 3 / 2) = (sqrt 39) / 3)) :=
begin
  sorry
end

end correct_statements_in_triangle_l248_248023


namespace distinct_values_count_l248_248591

-- Declare the sequence
noncomputable def sequence (x : ℝ) : ℕ → ℝ
| 0       := x
| 1       := 3000
| (n + 2) := (sequence n * sequence (n + 1) - 1)

def appears_3001 (x : ℝ) : Prop := ∃ n : ℕ, sequence x n = 3001

-- Main statement
theorem distinct_values_count :
  {x : ℝ | appears_3001 x}.card  = 4 :=
sorry

end distinct_values_count_l248_248591


namespace g_at_pi_over_4_l248_248657

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2) / 2 * Real.sin (2 * x) + (Real.sqrt 6) / 2 * Real.cos (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 4)

theorem g_at_pi_over_4 : g (Real.pi / 4) = (Real.sqrt 6) / 2 := by
  sorry

end g_at_pi_over_4_l248_248657


namespace current_at_R_12_l248_248252

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248252


namespace speed_of_marcos_l248_248753

theorem speed_of_marcos (distance_miles : ℝ) (time_minutes : ℝ) (speed_required : ℝ) :
  distance_miles = 5 → time_minutes = 10 → speed_required = 30 :=
by
  intros h_distance h_time
  have h_speed : speed_required = (distance_miles / time_minutes) * 60 := sorry
  show speed_required = 30 from sorry

end speed_of_marcos_l248_248753


namespace current_when_resistance_12_l248_248468

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248468


namespace find_y_given_conditions_l248_248586

variable (x z : ℝ)

-- Define the relationship between y, x, and z
def y_varies_as_x_inv_z_sq (k : ℝ) (x z : ℝ) : ℝ := k * x / (z ^ 2)

-- Given conditions
theorem find_y_given_conditions :
  (∀ (x z : ℝ), y_varies_as_x_inv_z_sq 2 x z = 10 ↔ (x = 5 ∧ z = 1)) →
  y_varies_as_x_inv_z_sq 2 (-10) 2 = -5 :=
by
  sorry

end find_y_given_conditions_l248_248586


namespace sqrt_factorial_product_eq_24_l248_248188

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l248_248188


namespace quadrilateral_AD_length_l248_248026

theorem quadrilateral_AD_length
  (A B C D : ℝ)
  (AB BC CD : ℝ)
  (B_obtuse C_obtuse : Prop)
  (sin_C eq : ℝ) :
  simple_quadrilateral A B C D →
  AB = 4 →
  BC = 5 →
  CD = 20 →
  B_obtuse →
  C_obtuse →
  sin_C = 3 / 5 →
  -cos B = 3 / 5 →
  length_AD A B C D = 25
:=
by
  sorry

end quadrilateral_AD_length_l248_248026


namespace sqrt_factorial_product_l248_248072

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l248_248072


namespace num_routes_A_to_B_in_3x3_grid_l248_248670

theorem num_routes_A_to_B_in_3x3_grid : 
  let total_moves := 10
  let right_moves := 5
  let down_moves := 5
  let routes_count := Nat.choose total_moves right_moves
  routes_count = 252 := 
by
  let total_moves := 10
  let right_moves := 5
  let down_moves := 5
  let routes_count := Nat.choose total_moves right_moves
  have fact_10 : 10! = 3628800 := by sorry
  have fact_5  : 5!  = 120 := by sorry
  have comb_10_5: Nat.choose 10 5 = 252 := by sorry
  exact comb_10_5

end num_routes_A_to_B_in_3x3_grid_l248_248670


namespace current_when_resistance_12_l248_248464

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248464


namespace current_at_resistance_12_l248_248529

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248529


namespace battery_current_when_resistance_12_l248_248519

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248519


namespace hyperbola_asymptotes_l248_248690

theorem hyperbola_asymptotes (x y : ℝ) (E : x^2 / 4 - y^2 = 1) :
  y = (1 / 2) * x ∨ y = -(1 / 2) * x :=
sorry

end hyperbola_asymptotes_l248_248690


namespace sqrt_factorial_product_l248_248104

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248104


namespace battery_current_l248_248276

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248276


namespace percentage_increase_undetermined_l248_248756

theorem percentage_increase_undetermined :
  ∀ (Mario_last_year Bob_current : ℝ), 
  Mario_last_year + 0.4 * Mario_last_year = 4000 →
  Bob_current = 3 * 4000 + (p : ℝ) * 3 * 4000 →
  ∃ y, y = Bob_current → false :=
by {
  intros Mario_last_year Bob_current,
  intros hMario hBob,
  have Mario_last : Mario_last_year = 4000 / 1.4, {
    sorry -- omitted proof for simplicity
  },
  have Bob_last : ¬ ∃ y, y = Bob_current, {
    sorry -- omitted proof for simplicity
  },
  exact ⟨Bob_current, rfl⟩
}

end percentage_increase_undetermined_l248_248756


namespace current_at_R_12_l248_248242

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248242


namespace parallelepiped_length_l248_248911

theorem parallelepiped_length (n : ℕ) :
  (n - 2) * (n - 4) * (n - 6) = 2 * n * (n - 2) * (n - 4) / 3 →
  n = 18 :=
by
  intros h
  sorry

end parallelepiped_length_l248_248911


namespace functional_expression_y_range_of_t_l248_248574

-- Define the coordinates of point A
def A : ℝ × ℝ := (0, 1)

-- Define the coordinates of point B
def B (y : ℝ) (h : y > 0) : ℝ × ℝ := (0, -y)

-- Define the condition that rhombus ABCD is formed
structure Rhombus :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (AB_eq_AD : dist A B = dist A D)
  (BC_eq_CD : dist B C = dist C D)
  (CD_eq_DA : dist C D = dist D A)
  (M_on_x_axis : ∃ M : ℝ × ℝ, M.2 = 0)

-- Define function y = 1/4 * x^2 for x ≠ 0
def y (x : ℝ) (h : x ≠ 0) : ℝ := (1 / 4) * x^2

-- Define the functional approximation problem for (1)
theorem functional_expression_y (x : ℝ) (h : x ≠ 0) :
  y x h = (1 / 4) * x^2 :=
sorry

-- Define a point P on the graph
def P (x y : ℝ) (h : x ≠ 0) : ℝ × ℝ := (x, y)

-- Define a point Q fixed on the y-axis
def Q (t : ℝ) : ℝ × ℝ := (0, t)

-- Define the condition for the fixed chord length
def fixed_chord_length (x y t : ℝ) (h₁ : x ≠ 0) (h₂ : y = 1 / 4 * x^2)
  (l : ℝ) (h₃ : l = t - 1) : Prop :=
  l > 0

-- Define the statement for range of t for (2)
theorem range_of_t {x t : ℝ} (h₁ : x ≠ 0) (h₂ : y x h₁ = (1/4) * x^2) :
  t > 1 :=
sorry

end functional_expression_y_range_of_t_l248_248574


namespace sqrt_factorial_product_l248_248070

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l248_248070


namespace cannot_achieve_80_cents_with_six_coins_l248_248776

theorem cannot_achieve_80_cents_with_six_coins:
  ¬ (∃ (p n d : ℕ), p + n + d = 6 ∧ p + 5 * n + 10 * d = 80) :=
by
  sorry

end cannot_achieve_80_cents_with_six_coins_l248_248776


namespace distinct_values_count_l248_248592

-- Declare the sequence
noncomputable def sequence (x : ℝ) : ℕ → ℝ
| 0       := x
| 1       := 3000
| (n + 2) := (sequence n * sequence (n + 1) - 1)

def appears_3001 (x : ℝ) : Prop := ∃ n : ℕ, sequence x n = 3001

-- Main statement
theorem distinct_values_count :
  {x : ℝ | appears_3001 x}.card  = 4 :=
sorry

end distinct_values_count_l248_248592


namespace current_at_R_12_l248_248250

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248250


namespace additional_trams_proof_l248_248048

-- Definitions for the conditions
def initial_tram_count : Nat := 12
def total_distance : Nat := 60
def initial_interval : Nat := total_distance / initial_tram_count
def reduced_interval : Nat := initial_interval - (initial_interval / 5)
def final_tram_count : Nat := total_distance / reduced_interval
def additional_trams_needed : Nat := final_tram_count - initial_tram_count

-- The theorem we need to prove
theorem additional_trams_proof : additional_trams_needed = 3 :=
by
  sorry

end additional_trams_proof_l248_248048


namespace total_animals_count_l248_248054

theorem total_animals_count (a m : ℕ) (h1 : a = 35) (h2 : a + 7 = m) : a + m = 77 :=
by
  sorry

end total_animals_count_l248_248054


namespace sqrt_factorial_mul_factorial_l248_248090

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l248_248090


namespace angle_ECD_l248_248692

noncomputable theory

open_locale big_operators

variables {A B C D E : Type*}
variables [ordered_ring A] [linear_order A] [ordered_ring B] [linear_order B]
variables [ordered_add_comm_group C] [linear_ordered_comm_group C] [add_torsor C D] [linear_ordered_field E]

-- Given conditions
variable (h1 : AC = BC)
variable (h2 : m ∠ DCB = 50°)
variable (h3 : parallel CD AB)
variable (h4 : AE = EC)

-- Question to be proved
theorem angle_ECD :
  m ∠ ECD = 50° :=
sorry

end angle_ECD_l248_248692


namespace current_when_resistance_is_12_l248_248408

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248408


namespace add_trams_l248_248053

theorem add_trams (total_trams : ℕ) (total_distance : ℝ) (initial_intervals : ℝ) (new_intervals : ℝ) (additional_trams : ℕ) :
  total_trams = 12 → total_distance = 60 → initial_intervals = total_distance / total_trams →
  new_intervals = initial_intervals - (initial_intervals / 5) →
  additional_trams = (total_distance / new_intervals) - total_trams →
  additional_trams = 3 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end add_trams_l248_248053


namespace battery_current_l248_248283

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248283


namespace current_value_l248_248386

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248386


namespace parallelogram_intersection_bisectors_rhombus_l248_248702

-- Definitions and conditions as per the problem statement
variables {P : Type*} [Parallelogram P]
variables {A B C D : P}

-- The overarching statement to be proved
theorem parallelogram_intersection_bisectors_rhombus 
  (parallelogram_conditions : parallelogram P)
  (diagonals_drawn : diagonals P)
  (angle_bisectors : angle_bisectors P)
  (intersection_points : intersection_points P A B C D) :
  rhombus A B C D := 
sorry

end parallelogram_intersection_bisectors_rhombus_l248_248702


namespace permutation_divisibility_l248_248853

theorem permutation_divisibility :
  ∃ (a : Fin 101 → ℕ), 
    (∀ k : Fin 101, 2 ≤ a k ∧ a k ≤ 102) ∧ 
    (∀ k₁ k₂ : Fin 101, k₁ ≠ k₂ → a k₁ ≠ a k₂) ∧ 
    (∀ k : Fin 101, a k % (k + 1) = 0) ∧ 
    (a = [ (1, 102), (1, 2, 102), (1, 3, 102), (1, 6, 102), (1, 17, 102), (1, 34, 102), 
           (1, 51, 102), (1, 2, 6, 102), (1, 2, 34, 102), (1, 3, 6, 102), (1, 3, 51, 102), 
           (1, 17, 34, 102), (1, 17, 51, 102)].toFinset) :=
sorry

end permutation_divisibility_l248_248853


namespace train_speed_l248_248567

noncomputable def speedOfMan : ℝ := 10 * 1000 / 3600 -- Speed of the man in m/s
noncomputable def trainLength : ℝ := 250 -- Length of the train in meters
noncomputable def timeToCrossMan : ℝ := 6 -- Time to cross the man in seconds
noncomputable def relativeSpeed : ℝ := trainLength / timeToCrossMan -- Relative speed in m/s

theorem train_speed (v : ℝ) (h_v : v = (relativeSpeed - speedOfMan)) : 
  v * 3.6 = 140 :=
begin
  sorry
end

end train_speed_l248_248567


namespace battery_current_l248_248379

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248379


namespace find_expression_xn_sub_invxn_l248_248644

theorem find_expression_xn_sub_invxn
  (phi : ℂ) (x : ℂ) (n : ℕ) (hn : n % 2 = 1) -- n is positive and odd
  (hphi : 0 < phi.im ∧ 0 < phi.re ∧ phi.re < π) -- φ is a complex number with 0 < φ.im < π and 0 < φ.re < π
  (hx : x - x⁻¹ = 2 * complex.I * complex.sin (φ)) :
  x^n - (x^n)⁻¹ = 2 * complex.I^n * (complex.sin φ)^n :=
sorry

end find_expression_xn_sub_invxn_l248_248644


namespace battery_current_l248_248333

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248333


namespace current_at_resistance_12_l248_248311

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248311


namespace cartesian_curve_C_eqn_range_of_x_minus_y_l248_248706

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

theorem cartesian_curve_C_eqn :
  ∀ (ρ θ : ℝ), ρ = 2 * Real.sin θ → 
  let (x, y) := polar_to_cartesian ρ θ in
  x^2 + (y - 1)^2 = 1 :=
by
  intro ρ θ h
  let ⟨x, y⟩ := polar_to_cartesian ρ θ
  sorry

theorem range_of_x_minus_y :
  ∀ (t : ℝ), -1 < t ∧ t < 0 →
  let x := (3/5) * t in
  let y := 1 + (4/5) * t in
  - (6/5) < x - y ∧ x - y < - (4/5) :=
by
  intro t ht
  let x := (3/5) * t
  let y := 1 + (4/5) * t
  sorry

end cartesian_curve_C_eqn_range_of_x_minus_y_l248_248706


namespace maximum_real_part_of_root_of_polynomial_l248_248933

def max_real_part_root (p : Polynomial ℂ) : ℝ :=
  p.roots.map (λ z, z.re).sup

theorem maximum_real_part_of_root_of_polynomial :
  let p := (X^6 - X^4 + X^2 - 1 : Polynomial ℂ) in max_real_part_root p = 1 :=
by
  sorry

end maximum_real_part_of_root_of_polynomial_l248_248933


namespace trams_to_add_l248_248043

theorem trams_to_add (initial_trams : ℕ) (initial_interval new_interval : ℤ)
  (reduce_by_fraction : ℤ) (total_distance : ℤ)
  (h1 : initial_trams = 12)
  (h2 : initial_interval = total_distance / initial_trams)
  (h3 : reduce_by_fraction = 5)
  (h4 : new_interval = initial_interval - initial_interval / reduce_by_fraction) :
  initial_trams + (total_distance / new_interval - initial_trams) = 15 :=
by
  sorry

end trams_to_add_l248_248043


namespace routes_A_to_B_in_grid_l248_248672

theorem routes_A_to_B_in_grid : 
  let m := 3 in 
  let n := 3 in 
  let total_moves := m + n in 
  let move_to_right := m in 
  let move_down := n in 
  Nat.choose total_moves move_to_right = 20 := 
by
  let m := 3
  let n := 3
  let total_moves := m + n
  let move_to_right := m
  let move_down := n
  show Nat.choose total_moves move_to_right = 20
  sorry

end routes_A_to_B_in_grid_l248_248672


namespace pizzeria_provolone_shred_l248_248892

theorem pizzeria_provolone_shred 
    (cost_blend : ℝ) 
    (cost_mozzarella : ℝ) 
    (cost_romano : ℝ) 
    (cost_provolone : ℝ) 
    (prop_mozzarella : ℝ) 
    (prop_romano : ℝ) 
    (prop_provolone : ℝ) 
    (shredded_mozzarella : ℕ) 
    (shredded_romano : ℕ) 
    (shredded_provolone_needed : ℕ) :
  cost_blend = 696.05 ∧ 
  cost_mozzarella = 504.35 ∧ 
  cost_romano = 887.75 ∧ 
  cost_provolone = 735.25 ∧ 
  prop_mozzarella = 2 ∧ 
  prop_romano = 1 ∧ 
  prop_provolone = 2 ∧ 
  shredded_mozzarella = 20 ∧ 
  shredded_romano = 10 → 
  shredded_provolone_needed = 20 :=
by {
  sorry -- proof to be provided
}

end pizzeria_provolone_shred_l248_248892


namespace max_PQ_over_MN_l248_248585

-- Define the problem conditions
def parabola (p : ℝ) (hp : p > 0) :=
  {x : ℝ × ℝ | ∃ y, y^2 = 2 * p * x ∧ x ≥ 0}

def F (p : ℝ) : ℝ × ℝ := (p, 0)

-- Problem statement in Lean
theorem max_PQ_over_MN (p : ℝ) (hp : p > 0)
  (M N : ℝ × ℝ) (hM : parabola p hp M) (hN : parabola p hp N) 
  (h_perp : (M.1 - F p.1) * (N.1 - F p.1) + (M.2 - F p.2) * (N.2 - F p.2) = 0) : 
  (let P := ((M.1 + N.1) / 2, (M.2 + N.2) / 2), 
        PQ := abs (P.2),
        MN := dist M N in 
    PQ / MN ≤ sqrt 2 / 2) :=
by
  sorry

end max_PQ_over_MN_l248_248585


namespace current_when_resistance_is_12_l248_248403

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248403


namespace largest_visits_is_4_l248_248055

noncomputable def largest_number_of_stores_visited (total_visits stores shoppers visiting_two_stores : ℕ) 
  (H1 : total_visits = 23) 
  (H2 : stores = 8) 
  (H3 : shoppers = 12) 
  (H4 : visiting_two_stores = 8) : ℕ :=
  max_visited_stores total_visits stores shoppers visiting_two_stores

theorem largest_visits_is_4 : 
  largest_number_of_stores_visited 23 8 12 8 = 4 := 
  by 
    sorry

end largest_visits_is_4_l248_248055


namespace prince_spending_l248_248947

theorem prince_spending (CDs_total : ℕ) (CDs_10_percent : ℕ) (CDs_10_cost : ℕ) (CDs_5_cost : ℕ) 
  (Prince_10_fraction : ℚ) (Prince_5_fraction : ℚ) 
  (total_10_CDs : ℕ) (total_5_CDs : ℕ) (Prince_10_CDs : ℕ) (Prince_5_CDs : ℕ) (total_cost : ℕ) :
  CDs_total = 200 →
  CDs_10_percent = 40 →
  CDs_10_cost = 10 →
  CDs_5_cost = 5 →
  Prince_10_fraction = 1/2 →
  Prince_5_fraction = 1 →
  total_10_CDs = CDs_total * CDs_10_percent / 100 →
  total_5_CDs = CDs_total - total_10_CDs →
  Prince_10_CDs = total_10_CDs * Prince_10_fraction →
  Prince_5_CDs = total_5_CDs * Prince_5_fraction →
  total_cost = (Prince_10_CDs * CDs_10_cost) + (Prince_5_CDs * CDs_5_cost) →
  total_cost = 1000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end prince_spending_l248_248947


namespace symmetric_trig_graphs_l248_248791

theorem symmetric_trig_graphs :
  (∀ x, sin (2 * (2 * (π / 24) - x) - π / 3) = sin (2 * x - π / 3)) ∧
  (∀ x, cos (2 * x + 2 * π / 3) = cos (2 * x + 5 * π / 6 - 4 * (π / 24))) :=
  sorry

end symmetric_trig_graphs_l248_248791


namespace toads_no_meeting_paths_l248_248825

theorem toads_no_meeting_paths :
  let binom := Nat.choose in
  let total_paths := binom 10 5 * binom 10 5 in
  let meet_paths := binom 12 5 * binom 8 5 in
  let no_meeting_paths := total_paths - meet_paths in
  no_meeting_paths = 19152 :=
by
  let binom := Nat.choose
  let total_paths := binom 10 5 * binom 10 5
  let meet_paths := binom 12 5 * binom 8 5
  let no_meeting_paths := total_paths - meet_paths
  have h : no_meeting_paths = 19152 := sorry
  exact h

end toads_no_meeting_paths_l248_248825


namespace sqrt_factorial_mul_factorial_l248_248079

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l248_248079


namespace probability_of_50_in_middle_l248_248999

theorem probability_of_50_in_middle :
  let S := (Finset.range 100).map (λ n, n + 1)
  let ways_to_choose_3 := Nat.choose 100 3
  let ways_to_choose_given_50 := (Nat.choose 49 1) * (Nat.choose 50 1)
  (ways_to_choose_given_50 / ways_to_choose_3 : ℚ) = 1 / 66 :=
by
  let S := (Finset.range 100).map (λ n, n + 1)
  let ways_to_choose_3 := Nat.choose 100 3
  let ways_to_choose_given_50 := (Nat.choose 49 1) * (Nat.choose 50 1)
  have h : (ways_to_choose_given_50 / ways_to_choose_3 : ℚ) = 1 / 66
  sorry

end probability_of_50_in_middle_l248_248999


namespace sum_lent_out_l248_248564

theorem sum_lent_out (P R : ℝ) (h1 : 720 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 7) / 100) : P = 600 := by
  sorry

end sum_lent_out_l248_248564


namespace find_unknown_l248_248852

theorem find_unknown :
  let x := 36 in
  let perc (p : ℝ) (n : ℝ) := (p / 100) * n in
  ( (45 * x / (3 / 4)) * (2 / 3) + 200 = perc 37.5 3000 - perc 62.5 800 + perc 27.5 2400 ) :=
by
  sorry

end find_unknown_l248_248852


namespace two_digit_number_tens_place_l248_248903

theorem two_digit_number_tens_place (x y : Nat) (hx1 : 0 ≤ x) (hx2 : x ≤ 9) (hy1 : 0 ≤ y) (hy2 : y ≤ 9)
    (h : (x + y) * 3 = 10 * x + y - 2) : x = 2 := 
sorry

end two_digit_number_tens_place_l248_248903


namespace selling_price_40_percent_profit_l248_248803

variable (C L : ℝ)

-- Condition: the profit earned by selling at $832 is equal to the loss incurred when selling at some price "L".
axiom eq_profit_loss : 832 - C = C - L

-- Condition: the desired profit price for a 40% profit on the cost price is $896.
axiom forty_percent_profit : 1.40 * C = 896

-- Theorem: the selling price for making a 40% profit is $896.
theorem selling_price_40_percent_profit : 1.40 * C = 896 :=
by
  sorry

end selling_price_40_percent_profit_l248_248803


namespace battery_current_l248_248366

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248366


namespace parallelepiped_length_l248_248920

theorem parallelepiped_length :
  ∃ n : ℕ, (n ≥ 7) ∧ (n * (n - 2) * (n - 4) = 3 * ((n - 2) * (n - 4) * (n - 6))) ∧ n = 18 :=
by
  sorry

end parallelepiped_length_l248_248920


namespace integer_solutions_count_number_of_solutions_l248_248624

theorem integer_solutions_count (d : ℤ) (h : 0 ≤ d ∧ d ≤ 2000) :
  (∃ x : ℝ, 8 * ⌊x⌋ + 3 * ⌈x⌉ = d) ↔ d % 11 = 0 ∨ d % 11 = 3 :=
by sorry

theorem number_of_solutions : Finset.card (Finset.filter (λ (d : ℤ), (0 ≤ d ∧ d ≤ 2000) ∧ (d % 11 = 0 ∨ d % 11 = 3)) (Finset.range 2001)) = 364 :=
by sorry

end integer_solutions_count_number_of_solutions_l248_248624


namespace sqrt_factorial_product_l248_248103

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248103


namespace number_of_shelves_l248_248863

def initial_bears : ℕ := 17
def shipment_bears : ℕ := 10
def bears_per_shelf : ℕ := 9

theorem number_of_shelves : (initial_bears + shipment_bears) / bears_per_shelf = 3 :=
by
  sorry

end number_of_shelves_l248_248863


namespace maximize_sector_area_radius_correct_l248_248655

noncomputable def maximize_sector_area_radius (C : ℝ) : ℝ := 
  let R := C / 4 in
  R

theorem maximize_sector_area_radius_correct (C : ℝ) (hC : C = 20) : 
  maximize_sector_area_radius C = 5 :=
by
  rw [maximize_sector_area_radius, hC] 
  norm_num
  sorry

end maximize_sector_area_radius_correct_l248_248655


namespace current_at_resistance_12_l248_248294

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248294


namespace sqrt_factorial_equality_l248_248160

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l248_248160


namespace sqrt_factorial_mul_factorial_l248_248207

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l248_248207


namespace sqrt_factorial_multiplication_l248_248184

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l248_248184


namespace battery_current_at_given_resistance_l248_248480

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248480


namespace wooden_parallelepiped_length_l248_248931

theorem wooden_parallelepiped_length (n : ℕ) (h1 : n ≥ 7)
    (h2 : ∀ total_cubes unpainted_cubes : ℕ,
      total_cubes = n * (n - 2) * (n - 4) ∧
      unpainted_cubes = (n - 2) * (n - 4) * (n - 6) ∧
      unpainted_cubes = 2 / 3 * total_cubes) :
  n = 18 := 
sorry

end wooden_parallelepiped_length_l248_248931


namespace find_current_when_resistance_is_12_l248_248365

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248365


namespace sqrt_factorial_equality_l248_248152

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l248_248152


namespace trains_clear_in_correct_time_l248_248827

noncomputable def time_to_clear (length1 length2 : ℝ) (speed1_kmph speed2_kmph : ℝ) : ℝ :=
  let speed1_mps := speed1_kmph * 1000 / 3600
  let speed2_mps := speed2_kmph * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := length1 + length2
  total_distance / relative_speed

-- The lengths of the trains
def length1 : ℝ := 151
def length2 : ℝ := 165

-- The speeds of the trains in km/h
def speed1_kmph : ℝ := 80
def speed2_kmph : ℝ := 65

-- The correct answer
def correct_time : ℝ := 7.844

theorem trains_clear_in_correct_time :
  time_to_clear length1 length2 speed1_kmph speed2_kmph = correct_time :=
by
  -- Skipping proof
  sorry

end trains_clear_in_correct_time_l248_248827


namespace current_when_resistance_12_l248_248461

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248461


namespace prince_spent_1000_l248_248948

def total_cds : ℕ := 200
def percentage_ten_dollars : ℚ := 0.40
def percentage_five_dollars : ℚ := 0.60
def price_ten_dollars : ℚ := 10
def price_five_dollars : ℚ := 5
def prince_share_ten_dollars : ℚ := 0.50

def count_ten_dollar_cds : ℕ := (percentage_ten_dollars * total_cds).to_nat
def count_five_dollar_cds : ℕ := (percentage_five_dollars * total_cds).to_nat
def count_prince_ten_dollar_cds : ℕ := (prince_share_ten_dollars * count_ten_dollar_cds).to_nat
def count_prince_five_dollar_cds : ℕ := count_five_dollar_cds

def total_money_spent (count_ten count_five : ℕ) : ℚ := (count_ten * price_ten_dollars) + (count_five * price_five_dollars)

theorem prince_spent_1000 :
  total_money_spent count_prince_ten_dollar_cds count_prince_five_dollar_cds = 1000 := by
  sorry

end prince_spent_1000_l248_248948


namespace collinear_EFN_l248_248700

variables {α : Type*}
variables (A B C D E F M N : α)
variables [AffineGeometry α]

-- Defining cyclic quadrilateral condition
def cyclic_quadrilateral (A B C D : α) : Prop :=
  ∃ (O : α), AffineGeometry.circle O r A ∧ AffineGeometry.circle O r B ∧ AffineGeometry.circle O r C ∧ AffineGeometry.circle O r D

-- Intersection and midpoint conditions
def is_intersection (E X Y Z : α) : Prop :=
  AffineGeometry.collinear X Y E ∧ AffineGeometry.collinear Y Z E

def is_midpoint (M X Y : α) : Prop :=
  AffineGeometry.between X M Y ∧ AffineGeometry.dist X M = AffineGeometry.dist M Y

-- Harmonic division condition
def harmonic_division (A B M N : α) : Prop :=
  AffineGeometry.cross_ratio A B M N = -1

-- Main theorem statement
theorem collinear_EFN :
  cyclic_quadrilateral A B C D →
  is_intersection E A D B C →
  is_intersection F B D A C →
  is_midpoint M C D →
  N ≠ M →
  harmonic_division A B M N →
  AffineGeometry.collinear E F N :=
by
  sorry

end collinear_EFN_l248_248700


namespace average_of_original_set_l248_248007

theorem average_of_original_set (A : ℝ) (h1 : 7 * A = 125 * 7 / 5) : A = 25 := 
sorry

end average_of_original_set_l248_248007


namespace problem_solution_l248_248603

noncomputable def problem_conditions : Prop :=
  (0 ∈ ({0} : Set Nat)) ∧
  ({0} ⊇ ∅) ∧
  (∃ r s : Int, r ≠ 0 ∧ 0.3 = r / s) ∧
  (0 ∈ {0}) ∧
  ({ x : Int | x^2 - 2 = 0 } = (∅ : Set Int))

theorem problem_solution : (counts_incorrect_conditions 1) :=
  sorry

end problem_solution_l248_248603


namespace solve_for_x_l248_248222

def f (x : ℝ) : ℝ := 2 * x - 3

theorem solve_for_x : ∃ (x : ℝ), 2 * (f x) - 11 = f (x - 2) :=
by
  use 5
  have h1 : f 5 = 2 * 5 - 3 := rfl
  have h2 : f (5 - 2) = 2 * (5 - 2) - 3 := rfl
  simp [f] at *
  exact sorry

end solve_for_x_l248_248222


namespace battery_current_l248_248369

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248369


namespace current_at_R_12_l248_248247

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248247


namespace ratio_of_areas_l248_248797

noncomputable def side_length_S : ℝ := sorry
noncomputable def side_length_longer_R : ℝ := 1.2 * side_length_S
noncomputable def side_length_shorter_R : ℝ := 0.8 * side_length_S
noncomputable def area_S : ℝ := side_length_S ^ 2
noncomputable def area_R : ℝ := side_length_longer_R * side_length_shorter_R

theorem ratio_of_areas :
  (area_R / area_S) = (24 / 25) :=
by
  sorry

end ratio_of_areas_l248_248797


namespace no_solutions_eq_l248_248056

theorem no_solutions_eq {t s k :ℝ} :
  ¬ ∃ t s : ℝ, (⟨3, 5⟩ : ℝ × ℝ) + t • (⟨4, -7⟩ : ℝ × ℝ) = (⟨2, -2⟩ : ℝ × ℝ) + s • (⟨-1, k⟩ : ℝ × ℝ) ↔ k = 7 / 4 := 
begin
  sorry
end

end no_solutions_eq_l248_248056


namespace simplify_expression_l248_248774

theorem simplify_expression (x : ℝ) : 
  (4 * x + 6 * x^3 + 8 - (3 - 6 * x^3 - 4 * x)) = 12 * x^3 + 8 * x + 5 := 
by
  sorry

end simplify_expression_l248_248774


namespace current_value_l248_248430

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248430


namespace passengers_at_station_in_an_hour_l248_248955

-- Define the conditions
def train_interval_minutes := 5
def passengers_off_per_train := 200
def passengers_on_per_train := 320

-- Define the time period we're considering
def time_period_minutes := 60

-- Calculate the expected values based on conditions
def expected_trains_per_hour := time_period_minutes / train_interval_minutes
def expected_passengers_off_per_hour := passengers_off_per_train * expected_trains_per_hour
def expected_passengers_on_per_hour := passengers_on_per_train * expected_trains_per_hour
def expected_total_passengers := expected_passengers_off_per_hour + expected_passengers_on_per_hour

theorem passengers_at_station_in_an_hour :
  expected_total_passengers = 6240 :=
by
  -- Structure of the proof omitted. Just ensuring conditions and expected value defined.
  sorry

end passengers_at_station_in_an_hour_l248_248955


namespace horses_legs_problem_l248_248879

theorem horses_legs_problem 
    (m h a b : ℕ) 
    (h_eq : h = m) 
    (men_to_A : m = 3 * a) 
    (men_to_B : m = 4 * b) 
    (total_legs : 2 * m + 4 * (h / 2) + 3 * a + 4 * b = 200) : 
    h = 25 :=
  sorry

end horses_legs_problem_l248_248879


namespace triangle_area_zero_vertex_l248_248221

theorem triangle_area_zero_vertex (x1 y1 x2 y2 : ℝ) :
  (1 / 2) * |x1 * y2 - x2 * y1| = 
    abs (1 / 2 * (x1 * y2 - x2 * y1)) := 
sorry

end triangle_area_zero_vertex_l248_248221


namespace battery_current_l248_248329

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248329


namespace percent_calc_l248_248578

theorem percent_calc (p n : ℝ) (h : p = 0.375) : p * n = 271.875 :=
by 
  have h1 : p * 725 = 0.375 * 725,
  rw h,
  simp,
  sorry

end percent_calc_l248_248578


namespace smallest_b_l248_248604

theorem smallest_b (b : ℕ) : 
  (b % 4 = 3) → 
  (b % 6 = 5) → 
  (b = 11) := 
by 
  intros h1 h2
  sorry

end smallest_b_l248_248604


namespace artist_paint_usage_l248_248573

def ounces_of_paint_used (extra_large: ℕ) (large: ℕ) (medium: ℕ) (small: ℕ) : ℕ :=
  4 * extra_large + 3 * large + 2 * medium + 1 * small

theorem artist_paint_usage : ounces_of_paint_used 3 5 6 8 = 47 := by
  sorry

end artist_paint_usage_l248_248573


namespace battery_current_l248_248321

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248321


namespace trams_required_l248_248034

theorem trams_required (initial_trams : ℕ) (initial_interval : ℚ) (reduction_fraction : ℚ) :
  initial_trams = 12 ∧ initial_interval = 5 ∧ reduction_fraction = 1/5 →
  (initial_trams + initial_trams * reduction_fraction - initial_trams) = 3 :=
by
  sorry

end trams_required_l248_248034


namespace triangle_angle_sum_90_l248_248584

theorem triangle_angle_sum_90
    (A B C D E : Type)
    [noncomputable α : ℝ]
    (angle : A → B → C → D → α)
    (angle_sum : ∀ {a b c : Type}, angle a b c = 180)
    (BAC : angle A B C = angle A B D + angle A D C)
    (AD_bisect : angle A B D = angle A D C / 2)
    (EBC : angle B C E = 90)
    (triangle_ABC : (angle A B C = α) ∨ (angle B C A = α/2) ∨ (angle C A B = α/2) := or.inl rfl) :
  angle A B C = 90 :=
by
  sorry

end triangle_angle_sum_90_l248_248584


namespace find_circle_M_common_chord_length_l248_248963

/-- Define points B and C --/
def B : ℝ × ℝ := (0, -2)
def C : ℝ × ℝ := (4, 0)

/-- Define the equation of line on which the center of circle M lies --/
def line : ℝ × ℝ → Prop := λ P, P.1 - P.2 = 0

/-- Given the general equation of circle M --/
def circleM (x y : ℝ) (D E F : ℝ) : Prop := x^2 + y^2 + D*x + E*y + F = 0

/-- Circle N is given by the equation (x-3)² + y² = 25 --/
def circleN (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

/-- Equation for the common chord length between circles M and N --/
def commonChordLength (M N : ℝ × ℝ → Prop) (len : ℝ) : Prop :=
  ∃ P Q : ℝ × ℝ, M P.1 P.2 ∧ N P.1 P.2 ∧ M Q.1 Q.2 ∧ N Q.1 Q.2 ∧ 
    ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 = len^2)

/-- First task: finding the equation of circle M --/
theorem find_circle_M : ∃ D E F : ℝ, 
  D = -2 ∧ E = -2 ∧ F = -8 ∧ 
  ( ∃ P : ℝ × ℝ, circleM P.1 P.2 D E F ∧ P = B ) ∧
  ( ∃ Q : ℝ × ℝ, circleM Q.1 Q.2 D E F ∧ Q = C ) ∧
  line (-D/2, -E/2) := 
sorry

/-- Second task: finding the length of the common chord between circle M and N --/
theorem common_chord_length : ∀ (D E F : ℝ), 
  D = -2 ∧ E = -2 ∧ F = -8 → 
  ∃ len : ℝ, len = 2 * Real.sqrt 5 ∧ 
  commonChordLength (circleM D E F) circleN len :=
sorry

end find_circle_M_common_chord_length_l248_248963


namespace max_absolute_difference_l248_248229

theorem max_absolute_difference (a b c d e : ℤ) (p : ℤ) :
  0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e ∧ e ≤ 100 ∧ p = (a + b + c + d + e) / 5 →
  (|p - c| ≤ 40) :=
by
  sorry

end max_absolute_difference_l248_248229


namespace sqrt_factorial_mul_factorial_l248_248083

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l248_248083


namespace battery_current_l248_248315

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248315


namespace equation1_solution_equation2_solution_l248_248777

theorem equation1_solution (x : ℚ) : 2 * (x - 3) = 1 - 3 * (x + 1) → x = 4 / 5 :=
by sorry

theorem equation2_solution (x : ℚ) : 3 * x + (x - 1) / 2 = 3 - (x - 1) / 3 → x = 1 :=
by sorry

end equation1_solution_equation2_solution_l248_248777


namespace child_ticket_cost_l248_248816

noncomputable def cost_of_child_ticket : ℝ := 3.50

theorem child_ticket_cost
  (adult_ticket_price : ℝ)
  (total_tickets : ℕ)
  (total_cost : ℝ)
  (adult_tickets_bought : ℕ)
  (adult_ticket_price_eq : adult_ticket_price = 5.50)
  (total_tickets_bought_eq : total_tickets = 21)
  (total_cost_eq : total_cost = 83.50)
  (adult_tickets_count : adult_tickets_bought = 5) :
  cost_of_child_ticket = 3.50 :=
by
  sorry

end child_ticket_cost_l248_248816


namespace current_when_resistance_12_l248_248459

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248459


namespace slope_angle_of_line_l248_248028

theorem slope_angle_of_line (a : ℝ) : 
  (∃ (a : ℝ), ∀ (x y : ℝ), (ax + y + 2 = 0) ∧ (atan 1 = pi / 4)) →
  a = -1 :=
by
  sorry

end slope_angle_of_line_l248_248028


namespace hotel_room_assignment_l248_248551

/-- In a hotel with 6 rooms, 6 friends decide to spend the night.
There are no other guests that night. The friends can room in any combination they wish,
but with no more than 2 friends per room and using no more than 5 rooms in total.
Prove that the number of ways the hotel manager can assign guests to rooms is 10440. -/
theorem hotel_room_assignment :
  let rooms := 6
  let guests := 6
  let max_friends_per_room := 2
  let max_rooms_used := 5
  (number_of_ways rooms guests max_friends_per_room max_rooms_used) = 10440 := 
sorry

end hotel_room_assignment_l248_248551


namespace part1_part2_l248_248663

noncomputable def f (x : ℝ) : ℝ := (√3 / 2) * sin (2 * x) - (1 / 2) * (cos x ^ 2 - sin x ^ 2) - 1
noncomputable def g (x : ℝ) : ℝ := f (x + π / 6)

variables {a b c A B C : ℝ} (h1 : c = √7) (h2 : f C = 0) (h3 : sin B = 3 * sin A) (h4 : g B = 0)
variables vec_m : ℝ × ℝ := (cos A, cos B)
variables vec_n : ℝ × ℝ := (1, sin A - cos A * tan B)

theorem part1 : a = 1 ∧ b = 3 :=
sorry

theorem part2 : vec_m.1 * vec_n.1 + vec_m.2 * vec_n.2 ∈ set.Ioo 0 1 :=
sorry

end part1_part2_l248_248663


namespace monic_quadratic_with_root_eq_l248_248621

noncomputable def monic_quadratic_with_root (a b : ℝ) : Polynomial ℂ :=
  Polynomial.C (complex.of_real b) * (Polynomial.X - Polynomial.C (complex.of_real a))^2

theorem monic_quadratic_with_root_eq :
  (∃ p : Polynomial ℝ, Polynomial.monic p ∧ p.coeff 1 = -2 ∧ p.coeff 0 = 2 ∧
    p.coeff 2 = 1 ∧ Polynomial.eval (1 - complex.i) p = 0) :=
begin
  use Polynomial.X^2 - 2 * Polynomial.X + 2,
  split,
  { apply Polynomial.monic_of_leading_coeff_eq_one,
    rw [Polynomial.leading_coeff, Polynomial.nat_degree_X_pow_sub_C],
    simp },
  split,
  { rw [Polynomial.coeff_X_pow_sub_C],
    simp },
  split,
  { rw [Polynomial.coeff_X_pow_sub_C],
    simp },
  split,
  { rw [Polynomial.to_complex_X_pow_sub_C_eq_of_real],
    simp },
  { rw [Polynomial.eval_X_pow_sub_C_eq_of_real],
    simp,
    norm_cast,
    field_simp }
end

end monic_quadratic_with_root_eq_l248_248621


namespace distance_between_points_l248_248580

-- Define the points as constants
def p1 : ℝ × ℝ := (2, 3)
def p2 : ℝ × ℝ := (5, 10)

-- Define a function to compute the distance between two points in 2D
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- State the theorem we want to prove
theorem distance_between_points : distance p1 p2 = real.sqrt 58 := by
  sorry

end distance_between_points_l248_248580


namespace current_at_resistance_12_l248_248309

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248309


namespace battery_current_l248_248378

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248378


namespace find_positive_real_solution_l248_248622

theorem find_positive_real_solution :
∃ x : ℝ, 0 < x ∧ (1/3 * (7 * x^2 - 3) = (x^2 - 70 * x - 20) * (x^2 + 35 * x + 7)) :=
sorry

end find_positive_real_solution_l248_248622


namespace battery_current_when_resistance_12_l248_248527

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248527


namespace current_at_resistance_12_l248_248537

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248537


namespace focus_of_parabola_l248_248784

theorem focus_of_parabola (x y : ℝ) (h : x^2 = -y) : (0, -1/4) = (0, -1/4) :=
by sorry

end focus_of_parabola_l248_248784


namespace sixty_cubes_exposed_faces_l248_248059

/-- Prove that when sixty 1x1x1 cubes are joined face to face in a single row and placed on a table, the number of exposed 1x1 faces is 182. --/
theorem sixty_cubes_exposed_faces : 
    let number_of_cubes := 60 in 
    let exposed_faces_per_cube := 3 in 
    let additional_exposed_faces := 2 in
    number_of_cubes * exposed_faces_per_cube + additional_exposed_faces = 182 :=
by
  let number_of_cubes := 60
  let exposed_faces_per_cube := 3
  let additional_exposed_faces := 2
  sorry

end sixty_cubes_exposed_faces_l248_248059


namespace num_sum_g7_product_l248_248747

noncomputable def realFunction : (ℝ → ℝ) → Prop :=
λ g, ∀ x y z : ℝ, g (x^2 + y * g z + z) = x * g x + z * g y + y

theorem num_sum_g7_product (g : ℝ → ℝ) (hg : realFunction g) :
  let n := 2 in
  let s := 7 in
  n * s = 14 :=
by
  sorry

end num_sum_g7_product_l248_248747


namespace find_current_l248_248442

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248442


namespace thermochemical_equation_correct_l248_248236

noncomputable def stable_state (water: String) : Prop :=
  water = "liquid"

noncomputable def heat_of_combustion_eq (eq: String) : Prop :=
  eq = "B₂H₆(g) + 3O₂(g) == B₂O₃(s) + 3H₂O(l) △H = -2165 kJ·mol⁻¹"

theorem thermochemical_equation_correct : 
  ∀ (initial_mol: ℝ) (heat_released: ℝ) (state_water: String) (equation: String),
  initial_mol = 0.3 ∧ 
  heat_released = 609.9 ∧ 
  stable_state state_water ∧ 
  heat_of_combustion_eq equation →
  equation = "B₂H₆(g) + 3O₂(g) == B₂O₃(s) + 3H₂O(l) △H = -2165 kJ·mol⁻¹" :=
begin
  intros,
  sorry,
end

end thermochemical_equation_correct_l248_248236


namespace trams_required_l248_248035

theorem trams_required (initial_trams : ℕ) (initial_interval : ℚ) (reduction_fraction : ℚ) :
  initial_trams = 12 ∧ initial_interval = 5 ∧ reduction_fraction = 1/5 →
  (initial_trams + initial_trams * reduction_fraction - initial_trams) = 3 :=
by
  sorry

end trams_required_l248_248035


namespace current_at_resistance_12_l248_248530

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248530


namespace sqrt_factorial_eq_l248_248127

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l248_248127


namespace num_of_integer_solutions_sqrt_5x_l248_248808

noncomputable def num_of_integer_solutions (a b : ℝ) (f : ℝ → ℝ) : ℕ :=
  Set.toFinset {x : ℤ | a < f (x.toReal) ∧ f (x.toReal) < b}.card

theorem num_of_integer_solutions_sqrt_5x : num_of_integer_solutions 2 5 (λ x, Real.sqrt (5 * x)) = 4 := by
  sorry

end num_of_integer_solutions_sqrt_5x_l248_248808


namespace yasna_reading_schedule_l248_248217

def pages_per_day (total_pages : ℕ) (total_days : ℕ) : ℕ := total_pages / total_days

theorem yasna_reading_schedule :
  let book1 := 180
  let book2 := 100
  let total_days := 2 * 7
  pages_per_day (book1 + book2) total_days = 20 :=
by
  let book1 := 180
  let book2 := 100
  let total_days := 2 * 7
  have total_pages : ℕ := book1 + book2
  show pages_per_day total_pages total_days = 20
  sorry

end yasna_reading_schedule_l248_248217


namespace train_crossing_time_l248_248668

noncomputable def train_length : ℝ := 385
noncomputable def train_speed_kmph : ℝ := 90
noncomputable def bridge_length : ℝ := 1250

noncomputable def convert_speed_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

noncomputable def time_to_cross_bridge (train_length bridge_length train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_mps := convert_speed_to_mps train_speed_kmph
  total_distance / speed_mps

theorem train_crossing_time :
  time_to_cross_bridge train_length bridge_length train_speed_kmph = 65.4 :=
by
  sorry

end train_crossing_time_l248_248668


namespace remaining_games_win_percent_l248_248565

variable (totalGames : ℕ) (firstGames : ℕ) (firstWinPercent : ℕ) (seasonWinPercent : ℕ)

-- Given conditions expressed as assumptions:
-- The total number of games played in a season is 40
axiom total_games_condition : totalGames = 40
-- The number of first games played is 30
axiom first_games_condition : firstGames = 30
-- The team won 40% of the first 30 games
axiom first_win_percent_condition : firstWinPercent = 40
-- The team won 50% of all its games in the season
axiom season_win_percent_condition : seasonWinPercent = 50

-- We need to prove that the percentage of the remaining games that the team won is 80%
theorem remaining_games_win_percent {remainingWinPercent : ℕ} :
  totalGames = 40 →
  firstGames = 30 →
  firstWinPercent = 40 →
  seasonWinPercent = 50 →
  remainingWinPercent = 80 :=
by
  intros
  sorry

end remaining_games_win_percent_l248_248565


namespace polynomial_exists_2024_roots_l248_248996

noncomputable def polynomial_sequence (P : ℝ[X]) (n : ℕ) : ℝ[X] :=
if n = 1 then P else polynomial_sequence (P.comp (polynomial_sequence P (n - 1))) 1

theorem polynomial_exists_2024_roots (a : ℝ) (P : ℝ[X])
  (C : ℝ) (h1 : a > 2) (h2 : C ≥ 4 * a) (h3 : C + 2 > 2 * sqrt (1 + 4 * a / C)) :
  (∀ t : ℝ, t ∈ Ioo (-a) a →  ∃ P : ℝ[X], (polynomial_sequence P 2024).roots.length = 2^2024) := by
sorry

end polynomial_exists_2024_roots_l248_248996


namespace find_roots_of_poly_l248_248990

-- Define the polynomial
def poly := (Polynomial.X^2 - 5 * Polynomial.X + 6) * (Polynomial.X - 3) * (Polynomial.X + 2)

-- State the theorem
theorem find_roots_of_poly : (Polynomial.roots poly).to_finset = {2, 3, -2} :=
by
  sorry

end find_roots_of_poly_l248_248990


namespace sqrt_factorial_product_l248_248093

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l248_248093


namespace remainder_when_divided_by_6_l248_248838

theorem remainder_when_divided_by_6 (n : ℕ) (h1 : n % 12 = 8) : n % 6 = 2 :=
sorry

end remainder_when_divided_by_6_l248_248838


namespace cistern_fill_time_l248_248220

theorem cistern_fill_time (F E : ℝ) (hF : F = 1 / 7) (hE : E = 1 / 9) : (1 / (F - E)) = 31.5 :=
by
  sorry

end cistern_fill_time_l248_248220


namespace length_of_AB_l248_248710

variables (A B C D : Type) [Real.Number]
variables (AB AC BC BD CD: ℝ)

axioms
  (isosceles_ABC : AB = AC)
  (isosceles_BCD : BC = CD)
  (perimeter_CBD : BC + CD + BD = 22)
  (perimeter_ABC : AB + AC + BC = 24)
  (length_BD : BD = 8)

theorem length_of_AB : AB = 8.5 :=
by sorry

end length_of_AB_l248_248710


namespace current_when_resistance_12_l248_248471

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248471


namespace maria_total_cost_l248_248754

-- Define the conditions
def pencil_cost : ℝ := 8
def pen_cost : ℝ := pencil_cost / 2
def eraser_cost : ℝ := 2 * pen_cost

def pencil_cost_with_tax : ℝ := pencil_cost * 1.08
def pen_cost_with_tax : ℝ := pen_cost * 1.05
def eraser_cost_with_tax : ℝ := eraser_cost * 1.10

def total_cost_before_discount : ℝ :=
  pencil_cost_with_tax + pen_cost_with_tax + eraser_cost_with_tax

def discount_amount : ℝ := total_cost_before_discount * 0.10
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

-- Statement to prove
theorem maria_total_cost : total_cost_after_discount = 19.48 := by
  sorry

end maria_total_cost_l248_248754


namespace Trevor_brother_age_proof_l248_248819

-- Given conditions
variables (Trevor_age_10_years_ago : ℕ)
          (Trevor_age_20_years_ago : ℕ)
          (Trevor_current_age : ℕ)
          (Trevor_brother_age_20_years_ago : ℕ)
          (Trevor_brother_current_age : ℕ)
          (h1 : Trevor_age_10_years_ago = 16)
          (h2 : Trevor_brother_age_20_years_ago = 2 * Trevor_age_20_years_ago)
          (h3 : Trevor_age_20_years_ago = Trevor_current_age - 20)
          (h4 : Trevor_current_age = Trevor_age_10_years_ago + 10)

-- We need to prove that Trevor's brother is currently 32 years old
theorem Trevor_brother_age_proof : Trevor_brother_current_age = 32 :=
by 
  -- Set up variables and hypotheses
  have h5 : Trevor_current_age = 26, by {
    rw [h4, h1],
  },
  have h6 : Trevor_age_20_years_ago = 6, by {
    rw [h5, sub_eq_add_neg, ← add_assoc, ← add_assoc, add_right_neg, zero_add],
  },
  have h7 : Trevor_brother_age_20_years_ago = 12, by {
    rw [h2, h6, nat.mul_succ, nat.mul_succ],
  },
  have h8 : Trevor_brother_current_age = 32, by {
    rw [← add_assoc, ← add_assoc],

  -- Use hypotheses to compute the result
  exact h7.trans (eq.symm h3),
  exact h8,
  sorry

end Trevor_brother_age_proof_l248_248819


namespace max_value_of_f_f_is_increasing_on_intervals_l248_248661

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x)

theorem max_value_of_f :
  ∃ (k : ℤ), ∀ (x : ℝ), x = k * Real.pi + Real.pi / 6 → f x = 3 :=
sorry

theorem f_is_increasing_on_intervals :
  ∀ (k : ℤ), ∀ (x y : ℝ), k * Real.pi - Real.pi / 3 ≤ x →
                x ≤ y → y ≤ k * Real.pi + Real.pi / 6 →
                f x ≤ f y :=
sorry

end max_value_of_f_f_is_increasing_on_intervals_l248_248661


namespace update_equipment_after_n_years_l248_248882

theorem update_equipment_after_n_years :
  ∃ n : ℕ+,  n + (100 / n) + 1.5 = 21.5 :=
by
  sorry

end update_equipment_after_n_years_l248_248882


namespace battery_current_l248_248343

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248343


namespace battery_current_at_given_resistance_l248_248484

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248484


namespace trams_required_l248_248037

theorem trams_required (initial_trams : ℕ) (initial_interval : ℚ) (reduction_fraction : ℚ) :
  initial_trams = 12 ∧ initial_interval = 5 ∧ reduction_fraction = 1/5 →
  (initial_trams + initial_trams * reduction_fraction - initial_trams) = 3 :=
by
  sorry

end trams_required_l248_248037


namespace correct_sum_of_t_values_l248_248973

noncomputable def sum_of_t_values : Real :=
  (let A := (Real.cos (30 * Real.pi / 180), Real.sin (30 * Real.pi / 180)) in
   let B := (Real.cos (90 * Real.pi / 180), Real.sin (90 * Real.pi / 180)) in
   let is_isosceles (t: Real) : Prop :=
     let C := (Real.cos (t * Real.pi / 180), Real.sin (t * Real.pi / 180)) in
     (dist A B = dist A C ∨ dist B A = dist B C ∨ dist C A = dist C B) in
   let t_values := [150, 210, 330] in
   ∑ t in t_values, t)
  
theorem correct_sum_of_t_values : sum_of_t_values = 690 := by
  sorry

end correct_sum_of_t_values_l248_248973


namespace smaller_angle_at_10_oclock_l248_248772

def degreeMeasureSmallerAngleAt10 := 
  let totalDegrees := 360
  let numHours := 12
  let degreesPerHour := totalDegrees / numHours
  let hourHandPosition := 10
  let minuteHandPosition := 12
  let divisionsBetween := if hourHandPosition < minuteHandPosition then minuteHandPosition - hourHandPosition else hourHandPosition - minuteHandPosition
  degreesPerHour * divisionsBetween

theorem smaller_angle_at_10_oclock : degreeMeasureSmallerAngleAt10 = 60 :=
  by 
    let totalDegrees := 360
    let numHours := 12
    let degreesPerHour := totalDegrees / numHours
    have h1 : degreesPerHour = 30 := by norm_num
    let hourHandPosition := 10
    let minuteHandPosition := 12
    let divisionsBetween := minuteHandPosition - hourHandPosition
    have h2 : divisionsBetween = 2 := by norm_num
    show 30 * divisionsBetween = 60
    calc 
      30 * 2 = 60 := by norm_num

end smaller_angle_at_10_oclock_l248_248772


namespace current_when_resistance_is_12_l248_248405

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248405


namespace equal_donations_amount_l248_248553

def raffle_tickets_sold := 25
def cost_per_ticket := 2
def total_raised := 100
def single_donation := 20
def amount_equal_donations (D : ℕ) : Prop := 2 * D + single_donation = total_raised - (raffle_tickets_sold * cost_per_ticket)

theorem equal_donations_amount (D : ℕ) (h : amount_equal_donations D) : D = 15 :=
  sorry

end equal_donations_amount_l248_248553


namespace sqrt_factorial_mul_factorial_eq_l248_248144

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l248_248144


namespace battery_current_when_resistance_12_l248_248515

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248515


namespace battery_current_at_given_resistance_l248_248481

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248481


namespace battery_current_l248_248332

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248332


namespace tangent_chord_length_l248_248964

-- Define the radii of the circles
def r_large : ℝ := 13
def r_small1 : ℝ := 4
def r_small2 : ℝ := 7

-- Define the distance between the centers of tangent circles
def dist_tangent_circles (r1 r2 : ℝ) : ℝ := r1 + r2

-- Define the distance between the centers of internally tangent circles
def dist_internal_tangent_circle_center (r_large r_small1 r_small2 : ℝ) : ℝ :=
  r_large - ((r_small1 * r_small1 + r_small2 * r_small2) / (r_small1 + r_small2))

-- Define the length squared of the tangent chord
def tangent_chord_length_squared (r_large r_small1 r_small2 : ℝ) : ℝ :=
  let dist_inter := dist_internal_tangent_circle_center r_large r_small1 r_small2
  in 4 * ((r_large * r_large) - (dist_inter * dist_inter))

-- The theorem we aim to prove
theorem tangent_chord_length (r_large r_small1 r_small2 : ℝ) :
  r_large = 13 → r_small1 = 4 → r_small2 = 7 → tangent_chord_length_squared r_large r_small1 r_small2 = 576 :=
by
  intros h_large h_small1 h_small2
  rw [h_large, h_small1, h_small2]
  unfold tangent_chord_length_squared dist_internal_tangent_circle_center
  norm_num
  sorry

end tangent_chord_length_l248_248964


namespace negation_of_exists_is_forall_l248_248801

open Classical

variable (P : ℕ → Prop) (Q : ℕ → Prop)

-- Proposition P
def P (x : ℕ) : Prop := x^2 + 2 * x ≥ 3

-- Negation of P is Q
def Q (x : ℕ) : Prop := x^2 + 2 * x < 3

theorem negation_of_exists_is_forall :
  (¬ ∃ x : ℕ, P x) ↔ ∀ x : ℕ, Q x := by sorry

end negation_of_exists_is_forall_l248_248801


namespace find_current_when_resistance_is_12_l248_248361

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248361


namespace prince_spent_1000_l248_248949

def total_cds : ℕ := 200
def percentage_ten_dollars : ℚ := 0.40
def percentage_five_dollars : ℚ := 0.60
def price_ten_dollars : ℚ := 10
def price_five_dollars : ℚ := 5
def prince_share_ten_dollars : ℚ := 0.50

def count_ten_dollar_cds : ℕ := (percentage_ten_dollars * total_cds).to_nat
def count_five_dollar_cds : ℕ := (percentage_five_dollars * total_cds).to_nat
def count_prince_ten_dollar_cds : ℕ := (prince_share_ten_dollars * count_ten_dollar_cds).to_nat
def count_prince_five_dollar_cds : ℕ := count_five_dollar_cds

def total_money_spent (count_ten count_five : ℕ) : ℚ := (count_ten * price_ten_dollars) + (count_five * price_five_dollars)

theorem prince_spent_1000 :
  total_money_spent count_prince_ten_dollar_cds count_prince_five_dollar_cds = 1000 := by
  sorry

end prince_spent_1000_l248_248949


namespace sqrt_factorial_multiplication_l248_248178

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l248_248178


namespace problem1_l248_248230

theorem problem1 :
  abs (-1 - 5) - (Real.pi - 2023)^0 + Real.sqrt 25 - (1/2)^(-1) = 8 :=
by
  sorry

end problem1_l248_248230


namespace current_when_resistance_is_12_l248_248409

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248409


namespace current_when_resistance_is_12_l248_248412

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248412


namespace probability_first_two_heads_l248_248837

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

end probability_first_two_heads_l248_248837


namespace battery_current_l248_248279

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248279


namespace battery_current_l248_248292

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248292


namespace cos_x1_minus_x2_l248_248645

theorem cos_x1_minus_x2 (x₁ x₂ : ℝ) (h₁ : 0 ≤ x₁ ∧ x₁ ≤ π) (h₂ : 0 ≤ x₂ ∧ x₂ ≤ π) 
  (h₃ : x₁ ≠ x₂) (h₄ : (-1 / 3) + sin x₁ = 0) (h₅ : (-1 / 3) + sin x₂ = 0) :
  cos (x₁ - x₂) = -7 / 9 :=
sorry

end cos_x1_minus_x2_l248_248645


namespace sqrt_factorial_product_l248_248101

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l248_248101


namespace find_current_l248_248454

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248454


namespace inequality_2_pow_ge_n_sq_l248_248615

theorem inequality_2_pow_ge_n_sq (n : ℕ) (hn : n ≠ 3) : 2^n ≥ n^2 :=
sorry

end inequality_2_pow_ge_n_sq_l248_248615


namespace Jack_emails_afternoon_l248_248722

-- Define the conditions given in the problem
variable (emails_morning : ℕ) (emails_afternoon : ℕ) (emails_evening : ℕ)
variable (total_afternoon_evening : ℕ)

-- Assume the conditions given
def Jack_conditions : Prop :=
  emails_morning = 4 ∧
  emails_evening = 8 ∧
  total_afternoon_evening = 13

-- State the theorem to prove the number of emails in the afternoon
theorem Jack_emails_afternoon (h : Jack_conditions) : emails_afternoon = 5 := by
  sorry

end Jack_emails_afternoon_l248_248722


namespace sqrt_factorial_mul_factorial_l248_248200

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l248_248200


namespace harmonic_series_exceeds_five_l248_248997

theorem harmonic_series_exceeds_five : ∃ k : ℕ, (∑ n in Finset.range k, 1 / (n + 1 : ℝ)) > 5 :=
by
  let k := 256
  have h : (∑ n in Finset.range k, 1 / (n + 1 : ℝ)) > 5
  {
    sorry -- detailed proof here
  }
  use k
  exact h

end harmonic_series_exceeds_five_l248_248997


namespace current_at_R_12_l248_248254

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248254


namespace distance_between_circle_centers_l248_248741

-- Define the given side lengths of the triangle
def DE : ℝ := 12
def DF : ℝ := 15
def EF : ℝ := 9

-- Define the problem and assertion
theorem distance_between_circle_centers :
  ∃ d : ℝ, d = 12 * Real.sqrt 13 :=
sorry

end distance_between_circle_centers_l248_248741


namespace sqrt_factorial_product_l248_248164

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248164


namespace sqrt_factorial_product_eq_24_l248_248191

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l248_248191


namespace average_temperature_Robertson_l248_248579

def temperatures : List ℝ := [18, 21, 19, 22, 20]

noncomputable def average (temps : List ℝ) : ℝ :=
  (temps.sum) / (temps.length)

theorem average_temperature_Robertson :
  average temperatures = 20.0 :=
by
  sorry

end average_temperature_Robertson_l248_248579


namespace domain_transformation_l248_248652

theorem domain_transformation (f : ℝ → ℝ) : (∀ y, -3 ≤ y ∧ y ≤ 3 → ∃ x, y = 2*x + 1) → ∀ x, -2 ≤ x ∧ x ≤ 1 → ∃ y, f(y) = f(2*x + 1) :=
by
  sorry

end domain_transformation_l248_248652


namespace primitive_root_mod_pow_alpha_l248_248765

variable {α : ℕ}
variable {p x : ℕ}

def odd_prime (p : ℕ) : Prop :=
  p > 2 ∧ ∀ d : ℕ, (d ∣ p → d = 1 ∨ d = p)

def primitive_root_mod (x p: ℕ) :=
  ∀ n : ℕ, (n > 0 → x^n ≡ 1 [MOD p] → n = totient p)

theorem primitive_root_mod_pow_alpha (h_prime : odd_prime p) (h_prim_root : primitive_root_mod x (p^2)) (h_alpha : α ≥ 2) :
  primitive_root_mod x (p^α) :=
sorry

end primitive_root_mod_pow_alpha_l248_248765


namespace find_blotted_digits_l248_248633

theorem find_blotted_digits :
  ∃ x y : ℕ, (∃ n : ℕ, n * 3600 = 234 * 1000 + x * 100 + y * 10) ∧ x = 7 ∧ y = 0 :=
begin
  sorry
end

end find_blotted_digits_l248_248633


namespace probability_of_selecting_double_l248_248547

-- Define the conditions and the question
def total_integers : ℕ := 13

def number_of_doubles : ℕ := total_integers

def total_pairings : ℕ := 
  (total_integers * (total_integers + 1)) / 2

def probability_double : ℚ := 
  number_of_doubles / total_pairings

-- Statement to be proved 
theorem probability_of_selecting_double : 
  probability_double = 1/7 := 
sorry

end probability_of_selecting_double_l248_248547


namespace sqrt_factorial_product_l248_248067

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l248_248067


namespace current_when_resistance_12_l248_248470

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248470


namespace current_when_resistance_12_l248_248473

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248473


namespace inverse_proportion_symmetric_l248_248628

theorem inverse_proportion_symmetric (a b : ℝ) (h : a ≠ 0) (h_ab : b = -6 / -a) : (-b) = -6 / a :=
by
  -- the proof goes here
  sorry

end inverse_proportion_symmetric_l248_248628


namespace length_of_AB_is_10_l248_248711

noncomputable def length_AB (triangleABC : Triangle) (triangleCBD : Triangle) 
  (isoscelesABC : triangleABC.isIsosceles) 
  (isoscelesCBD : triangleCBD.isIsosceles) 
  (perimeterCBD : triangleCBD.perimeter = 22) 
  (perimeterABC : triangleABC.perimeter = 24) 
  (BD_length : triangleCBD.sideBD.length = 8) : Real :=
10

theorem length_of_AB_is_10 :
  ∀ (triangleABC : Triangle) (triangleCBD : Triangle), 
    triangleABC.isIsosceles →
    triangleCBD.isIsosceles →
    triangleCBD.perimeter = 22 →
    triangleABC.perimeter = 24 →
    triangleCBD.sideBD.length = 8 →
    triangleABC.sideAB.length = 10 :=
by
  intro triangleABC triangleCBD isIsoscelesABC isIsoscelesCBD perimeterCBD perimeterABC BD_length
  rw [length_AB triangleABC triangleCBD isIsoscelesABC isIsoscelesCBD perimeterCBD perimeterABC BD_length]
  sorry

end length_of_AB_is_10_l248_248711


namespace product_of_two_numbers_l248_248792

theorem product_of_two_numbers (a b : ℤ) (h1 : lcm a b = 72) (h2 : gcd a b = 8) :
  a * b = 576 :=
sorry

end product_of_two_numbers_l248_248792


namespace number_of_ways_to_distribute_students_l248_248975

-- Define the students and the classes
inductive Student : Type
| A | B | C | D

inductive Class : Type
| Class1 | Class2 | Class3

-- Define the condition function for assignment
def valid_assignment (assign : Student → Class) : Prop :=
  (Student.rec_on assign A ≠ Student.rec_on assign B) ∧
  (Student.rec_on assign C ≠ Student.rec_on assign A ∧ Student.rec_on assign C ≠ Student.rec_on assign B ∨
   Student.rec_on assign D ≠ Student.rec_on assign A ∧ Student.rec_on assign D ≠ Student.rec_on assign B)

-- The main statement of the problem, asserting the number of valid distributions
theorem number_of_ways_to_distribute_students : 
  (finset.univ.filter valid_assignment).card = 30 :=
sorry

end number_of_ways_to_distribute_students_l248_248975


namespace sqrt_factorial_product_l248_248168

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248168


namespace sqrt_factorial_product_l248_248105

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248105


namespace no_quad_poly_solutions_l248_248959

theorem no_quad_poly_solutions :
  ¬ (∃ (f g h : ℝ → ℝ), 
        (∀ x, f(x) = a*x^2 + b*x + c) ∧ 
        (∀ x, g(x) = d*x^2 + e*x + f) ∧ 
        (∀ x, h(x) = g*x^2 + h*x + i) ∧ 
        ∀ x, x ∈ ({1, 2, 3, 4, 5, 6, 7, 8} : set ℝ) → f(g(h(x))) = 0) :=
sorry

end no_quad_poly_solutions_l248_248959


namespace prince_spent_1000_l248_248951

noncomputable def total_CDs := 200
noncomputable def percent_CDs_10 := 0.40
noncomputable def price_per_CD_10 := 10
noncomputable def price_per_CD_5 := 5

-- Number of CDs sold at $10 each
noncomputable def num_CDs_10 := percent_CDs_10 * total_CDs

-- Number of CDs sold at $5 each
noncomputable def num_CDs_5 := total_CDs - num_CDs_10

-- Number of $10 CDs bought by Prince
noncomputable def prince_CDs_10 := num_CDs_10 / 2

-- Total cost of $10 CDs bought by Prince
noncomputable def cost_CDs_10 := prince_CDs_10 * price_per_CD_10

-- Total cost of $5 CDs bought by Prince
noncomputable def cost_CDs_5 := num_CDs_5 * price_per_CD_5

-- Total amount of money Prince spent
noncomputable def total_spent := cost_CDs_10 + cost_CDs_5

theorem prince_spent_1000 : total_spent = 1000 := by
  -- Definitions from conditions
  have h1 : total_CDs = 200 := rfl
  have h2 : percent_CDs_10 = 0.40 := rfl
  have h3 : price_per_CD_10 = 10 := rfl
  have h4 : price_per_CD_5 = 5 := rfl

  -- Calculations from solution steps (insert sorry to skip actual proofs)
  have h5 : num_CDs_10 = 80 := sorry
  have h6 : num_CDs_5 = 120 := sorry
  have h7 : prince_CDs_10 = 40 := sorry
  have h8 : cost_CDs_10 = 400 := sorry
  have h9 : cost_CDs_5 = 600 := sorry

  show total_spent = 1000
  sorry

end prince_spent_1000_l248_248951


namespace sqrt_factorial_product_l248_248119

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l248_248119


namespace additional_trams_proof_l248_248046

-- Definitions for the conditions
def initial_tram_count : Nat := 12
def total_distance : Nat := 60
def initial_interval : Nat := total_distance / initial_tram_count
def reduced_interval : Nat := initial_interval - (initial_interval / 5)
def final_tram_count : Nat := total_distance / reduced_interval
def additional_trams_needed : Nat := final_tram_count - initial_tram_count

-- The theorem we need to prove
theorem additional_trams_proof : additional_trams_needed = 3 :=
by
  sorry

end additional_trams_proof_l248_248046


namespace current_at_resistance_12_l248_248297

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248297


namespace current_when_resistance_is_12_l248_248414

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248414


namespace find_magnitude_l248_248635

noncomputable def magnitude_sum (x y : ℝ) (a b c : ℝ × ℝ) : ℝ :=
let (ax, ay) := a in
let (bx, by) := b in
let (cx, cy) := c in
if ax * cx + ay * cy = 0 ∧ (cy*y = cx * by)
then real.sqrt ((ax + bx) ^ 2 + (ay + by) ^ 2)
else 0

theorem find_magnitude :
  ∀ (x y : ℝ),
  let a := (x, 2) in
  let b := (1, y) in
  let c := (2, -6) in
  (x * 2 - 12 = 0) ∧ (-3 * y = -6) → magnitude_sum x y a b c = 5 * real.sqrt 2 :=
by
  intros x y a b c
  simp only [a, b, c, magnitude_sum]
  sorry

end find_magnitude_l248_248635


namespace find_current_l248_248445

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248445


namespace find_current_l248_248451

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248451


namespace joel_donated_22_toys_l248_248726

-- Given conditions
variables (T : ℕ)
variables (stuffed_animals action_figures board_games puzzles : ℕ)
variables (total_donated : ℕ)

-- Define the conditions
def conditions := 
  stuffed_animals = 18 ∧ 
  action_figures = 42 ∧ 
  board_games = 2 ∧ 
  puzzles = 13 ∧ 
  total_donated = 108

-- Calculating the total number of toys from friends
def friends_total := 
  stuffed_animals + action_figures + board_games + puzzles

-- The total number of toys Joel and his sister donated
def total_joel_sister := 
  T + 2 * T

-- The proof problem
theorem joel_donated_22_toys 
  (h : conditions) : 
  3 * T + friends_total = total_donated → 2 * T = 22 :=
by
  intros
  sorry

end joel_donated_22_toys_l248_248726


namespace current_at_resistance_12_l248_248532

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248532


namespace channel_depth_l248_248786

theorem channel_depth :
  ∀ (a b A h : ℝ),
  a = 12 ∧ b = 8 ∧ A = 700 → (A = 0.5 * (a + b) * h) → h = 70 :=
by 
  intros a b A h h_eq_area conds,
  sorry

end channel_depth_l248_248786


namespace find_current_when_resistance_is_12_l248_248355

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248355


namespace sqrt_factorial_mul_factorial_l248_248086

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l248_248086


namespace households_used_both_brands_l248_248239

theorem households_used_both_brands (X : ℕ) : 
  (80 + 60 + X + 3 * X = 260) → X = 30 :=
by
  sorry

end households_used_both_brands_l248_248239


namespace current_at_resistance_12_l248_248534

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248534


namespace triangle_classification_l248_248687

namespace TriangleProblem

-- Definitions to express the conditions and statement in Lean
variables {α : Type*} [linear_ordered_field α]

structure Triangle (α : Type*) :=
  (A B C : α) 
  (angle_non_neg : 0 < A ∧ 0 < B ∧ 0 < C)
  (angle_sum : A + B + C = π)

variables (T1 T2 : Triangle α)

-- Hypotheses based on the given conditions
def condition (T1 T2 : Triangle α) : Prop :=
  (cos T1.A = sin T2.A) ∧ (cos T1.B = sin T2.B) ∧ (cos T1.C = sin T2.C)

-- Main statement to be proved in Lean
theorem triangle_classification (h : condition T1 T2) : 
  (0 < T1.A ∧ T1.A < π/2 ∧ 0 < T1.B ∧ T1.B < π/2 ∧ 0 < T1.C ∧ T1.C < π/2) ∧
  (T2.A > π/2 ∨ T2.B > π/2 ∨ T2.C > π/2) := 
sorry

end TriangleProblem

end triangle_classification_l248_248687


namespace distance_between_l1_l2_is_correct_l248_248688

-- Definitions
def l1 (x y : ℝ) : Prop := x - 2 * y + 1 = 0
def l2 (x y : ℝ) (a : ℝ) : Prop := 2 * x + a * y - 2 = 0

-- Condition for lines to be parallel
def parallel_lines (a : ℝ) : Prop := -2/a = 1/2

-- Distance between two parallel lines l1 and l2 when a = -4
noncomputable def distance_between_parallel_lines (a : ℝ) : ℝ :=
if parallel_lines a then
  (|1 - (-1)| : ℝ) / real.sqrt (1^2 + (-2)^2)
else
  0

-- Theorem
theorem distance_between_l1_l2_is_correct (a : ℝ) (h : a = -4) : distance_between_parallel_lines a = (2 * real.sqrt 5) / 5 :=
by
  sorry

end distance_between_l1_l2_is_correct_l248_248688


namespace sqrt_factorial_mul_factorial_eq_l248_248147

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l248_248147


namespace sqrt_factorial_product_l248_248122

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l248_248122


namespace prince_spent_1000_l248_248950

def total_cds : ℕ := 200
def percentage_ten_dollars : ℚ := 0.40
def percentage_five_dollars : ℚ := 0.60
def price_ten_dollars : ℚ := 10
def price_five_dollars : ℚ := 5
def prince_share_ten_dollars : ℚ := 0.50

def count_ten_dollar_cds : ℕ := (percentage_ten_dollars * total_cds).to_nat
def count_five_dollar_cds : ℕ := (percentage_five_dollars * total_cds).to_nat
def count_prince_ten_dollar_cds : ℕ := (prince_share_ten_dollars * count_ten_dollar_cds).to_nat
def count_prince_five_dollar_cds : ℕ := count_five_dollar_cds

def total_money_spent (count_ten count_five : ℕ) : ℚ := (count_ten * price_ten_dollars) + (count_five * price_five_dollars)

theorem prince_spent_1000 :
  total_money_spent count_prince_ten_dollar_cds count_prince_five_dollar_cds = 1000 := by
  sorry

end prince_spent_1000_l248_248950


namespace sqrt_factorial_eq_l248_248129

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l248_248129


namespace probability_of_intersection_in_nonagon_l248_248967

-- Define the problem and the required proofs
noncomputable def diagonals_in_nonagon : ℕ := 27
noncomputable def pairs_of_diagonals : ℕ := 351
noncomputable def intersecting_diagonals : ℕ := 126
noncomputable def probability_of_intersection : ℚ := 14 / 39

theorem probability_of_intersection_in_nonagon :
  let N := diagonals_in_nonagon,
  let P := pairs_of_diagonals,
  let I := intersecting_diagonals,
  ∃ k: ℚ, k = I / P ∧ k = probability_of_intersection :=
by {
  sorry -- Proof steps would go here
}

end probability_of_intersection_in_nonagon_l248_248967


namespace area_of_square_with_diagonal_40_l248_248015

-- Define the necessary variables and properties
variables (d : ℝ) (s : ℝ) (A : ℝ)

-- Given conditions
def diagonal_eq_40 := d = 40
def relation_diagonal_side := d^2 = 2 * s^2
def area_square := A = s^2

-- The statement to be proven
theorem area_of_square_with_diagonal_40 (h1 : diagonal_eq_40) (h2 : relation_diagonal_side) : A = 800 :=
by
  sorry

end area_of_square_with_diagonal_40_l248_248015


namespace sqrt_factorial_eq_l248_248128

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l248_248128


namespace robbin_bobbin_anti_matter_l248_248701

theorem robbin_bobbin_anti_matter : 
  ∃ (x y z : ℕ), 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ 11 * ↑x + 1.1 * ↑y + 0.11 * ↑z = 20.13 :=
by
  sorry

end robbin_bobbin_anti_matter_l248_248701


namespace sqrt_factorial_equality_l248_248154

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l248_248154


namespace current_value_l248_248390

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248390


namespace expand_product_l248_248981

theorem expand_product (x : ℝ) : (x^2 - 2*x + 2) * (x^2 + 2*x + 2) = x^4 + 4 :=
by
  sorry

end expand_product_l248_248981


namespace students_with_uncool_parents_l248_248699

theorem students_with_uncool_parents 
  (total_students : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (cool_both : ℕ) 
  (h1 : total_students = 35)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 20)
  (h4 : cool_both = 11) : 
  total_students - (cool_dads + cool_moms - cool_both) = 8 :=
by 
  rw [h1, h2, h3, h4]
  have h5: 18 + 20 - 11 = 27 := by norm_num
  rw h5
  norm_num

end students_with_uncool_parents_l248_248699


namespace current_at_R_12_l248_248248

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248248


namespace bus_fare_with_train_change_in_total_passengers_l248_248860

variables (p : ℝ) (q : ℝ) (TC : ℝ → ℝ)
variables (p_train : ℝ) (train_capacity : ℝ)

-- Demand function
def demand_function (p : ℝ) : ℝ := 4200 - 100 * p

-- Train fare is fixed
def train_fare : ℝ := 4

-- Train capacity
def train_cap : ℝ := 800

-- Bus total cost function
def total_cost (y : ℝ) : ℝ := 10 * y + 225

-- Case when there is competition (train available)
def optimal_bus_fare_with_train : ℝ := 22

-- Case when there is no competition (train service is closed)
def optimal_bus_fare_without_train : ℝ := 26

-- Change in the number of passengers when the train service closes
def change_in_passengers : ℝ := 400

-- Theorems to prove
theorem bus_fare_with_train : optimal_bus_fare_with_train = 22 := sorry
theorem change_in_total_passengers : change_in_passengers = 400 := sorry

end bus_fare_with_train_change_in_total_passengers_l248_248860


namespace find_current_when_resistance_is_12_l248_248356

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248356


namespace current_when_resistance_is_12_l248_248407

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248407


namespace sqrt_factorial_mul_factorial_l248_248084

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l248_248084


namespace volume_increase_factor_l248_248839

variable (π : ℝ) (r h : ℝ)

def original_volume : ℝ := π * r^2 * h

def new_volume : ℝ := π * (2 * r)^2 * (3 * h)

theorem volume_increase_factor : new_volume π r h = 12 * original_volume π r h :=
by
  -- Here we would include the proof that new_volume = 12 * original_volume
  sorry

end volume_increase_factor_l248_248839


namespace orange_slices_needed_l248_248883

theorem orange_slices_needed (total_slices containers_capacity leftover_slices: ℕ) 
(h1 : containers_capacity = 4) 
(h2 : total_slices = 329) 
(h3 : leftover_slices = 1) :
    containers_capacity - leftover_slices = 3 :=
by
  sorry

end orange_slices_needed_l248_248883


namespace current_value_l248_248275

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248275


namespace natural_numbers_pq_equal_l248_248800

theorem natural_numbers_pq_equal (p q : ℕ) (h : p^p + q^q = p^q + q^p) : p = q :=
sorry

end natural_numbers_pq_equal_l248_248800


namespace problem_statement_l248_248679

theorem problem_statement (a b c : ℕ) (h1 : a < 12) (h2 : b < 12) (h3 : c < 12) (h4 : b + c = 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b + c :=
by
  sorry

end problem_statement_l248_248679


namespace current_when_resistance_is_12_l248_248416

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248416


namespace brad_must_make_5_trips_l248_248576

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r ^ 2 * h

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r ^ 2 * h

theorem brad_must_make_5_trips (r_barrel h_barrel r_bucket h_bucket : ℝ)
    (h1 : r_barrel = 10) (h2 : h_barrel = 15) (h3 : r_bucket = 10) (h4 : h_bucket = 10) :
    let trips := volume_of_cylinder r_barrel h_barrel / volume_of_cone r_bucket h_bucket
    let trips_needed := Int.ceil trips
    trips_needed = 5 := 
by
  sorry

end brad_must_make_5_trips_l248_248576


namespace problem_I_problem_II_l248_248646

variable (t a : ℝ)

-- Problem (I)
theorem problem_I (h1 : a = 1) (h2 : t^2 - 5 * a * t + 4 * a^2 < 0) (h3 : (t - 2) * (t - 6) < 0) : 2 < t ∧ t < 4 := 
by 
  sorry   -- Proof omitted as per instructions

-- Problem (II)
theorem problem_II (h1 : (t - 2) * (t - 6) < 0 → t^2 - 5 * a * t + 4 * a^2 < 0) : 3 / 2 ≤ a ∧ a ≤ 2 :=
by 
  sorry   -- Proof omitted as per instructions

end problem_I_problem_II_l248_248646


namespace parallelepiped_length_l248_248921

theorem parallelepiped_length :
  ∃ n : ℕ, (n ≥ 7) ∧ (n * (n - 2) * (n - 4) = 3 * ((n - 2) * (n - 4) * (n - 6))) ∧ n = 18 :=
by
  sorry

end parallelepiped_length_l248_248921


namespace battery_current_l248_248338

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248338


namespace current_value_l248_248397

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248397


namespace boat_stream_speed_l248_248874

theorem boat_stream_speed (v : ℝ) (h : (60 / (15 - v)) - (60 / (15 + v)) = 2) : v = 3.5 := 
by 
  sorry
 
end boat_stream_speed_l248_248874


namespace sqrt_factorial_product_l248_248171

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248171


namespace three_squares_not_divisible_by_three_l248_248767

theorem three_squares_not_divisible_by_three 
  (N : ℕ) (a b c : ℤ) 
  (h₁ : N = 9 * (a^2 + b^2 + c^2)) :
  ∃ x y z : ℤ, N = x^2 + y^2 + z^2 ∧ ¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z) := 
sorry

end three_squares_not_divisible_by_three_l248_248767


namespace same_graph_iff_same_function_D_l248_248844

theorem same_graph_iff_same_function_D :
  ∀ x : ℝ, (|x| = if x ≥ 0 then x else -x) :=
by
  intro x
  sorry

end same_graph_iff_same_function_D_l248_248844


namespace find_k_l248_248549

def f : ℝ → ℝ

/-- Given conditions -/
axiom condition1 : f 1 = 4
axiom condition2 : ∀ x y : ℝ, f (x + y) = f x + f y + k * x * y + 4
axiom condition3 : f 2 + f 5 = 125

-- Defining k as a real number
variable (k : ℝ)

/-- Statement to be proved -/
theorem find_k : k = 7 :=
by sorry

end find_k_l248_248549


namespace battery_current_l248_248327

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248327


namespace battery_current_l248_248325

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248325


namespace current_at_resistance_12_l248_248296

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248296


namespace current_value_l248_248501

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248501


namespace number_of_arrangements_l248_248669

def arrangement_count (A B C : ℕ) (prefix middle suffix : List Char) : ℕ :=
  -- Assume this function counts valid arrangements of A's, B's, and C's 
  -- according to the provided constraints in prefix, middle, and suffix
  sorry

theorem number_of_arrangements :
  arrangement_count 4 6 5
    ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']  -- 4 A's and no C's
    ['B', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C', 'C']  -- 6 B's and no A's
    ['A', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'C']  -- 5 C's and no B's
  = 25 :=
sorry

end number_of_arrangements_l248_248669


namespace current_value_l248_248264

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248264


namespace battery_current_l248_248371

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248371


namespace battery_current_l248_248373

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248373


namespace sequence_3001_values_l248_248589

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then x else
  if n = 2 then 3000 else
  (sequence x (n - 1) + 1) / sequence x (n - 2)

theorem sequence_3001_values : 
  ∃ n : ℕ, ∀ x : ℝ, (∃ n, sequence x n = 3001) → 
    x = 3001 ∨ x = 1 ∨ x = 3001 / 9002999 ∨ x = 9002999 :=
sorry

end sequence_3001_values_l248_248589


namespace triangle_angle_A_triangle_angle_C_l248_248720

theorem triangle_angle_A (a b c S : ℝ) (h1 : a^2 + 4 * S = b^2 + c^2) (h2 : S = (1 / 2) * b * c * real.sin (real.arccos (c/b))) :
  real.arccos (c/b) = real.pi / 4 :=
sorry

theorem triangle_angle_C (a b c : ℝ) (ha : a = real.sqrt 2) (hb : b = real.sqrt 3) (h1 : a^2 + 4 * (1/2 * b * c * real.sin (real.arccos (c/b))) = b^2 + c^2) (h2 : real.arccos (c/b) = real.pi / 4) :
  real.pi - real.pi / 4 - real.arccos (real.sqrt 3 / 2) = 5 * real.pi / 12 :=
sorry

end triangle_angle_A_triangle_angle_C_l248_248720


namespace current_at_resistance_12_l248_248306

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248306


namespace determine_ab_l248_248607

variables {m n : ℝ} (hmn : m ≠ n) (hm : m ≠ 0) (hn : n ≠ 0)

noncomputable def x (a b : ℝ) : ℝ := a * m + b * n

theorem determine_ab (a b : ℝ) (hmn : m ≠ n) (hm : m ≠ 0) (hn : n ≠ 0) :
  ((x a b + m)^3 - (x a b + n)^3 = (m - n)^3) ↔ (a = -1 ∧ b = 1) :=
sorry

end determine_ab_l248_248607


namespace length_of_parallelepiped_l248_248926

def number_of_cubes_with_painted_faces (n : ℕ) := (n - 2) * (n - 4) * (n - 6) 
def total_number_of_cubes (n : ℕ) := n * (n - 2) * (n - 4)

theorem length_of_parallelepiped (n : ℕ) (h1 : total_number_of_cubes n = 3 * number_of_cubes_with_painted_faces n) : 
  n = 18 :=
by 
  sorry

end length_of_parallelepiped_l248_248926


namespace probability_log3_integer_l248_248556

open nat real

def is_power_of_three (N : ℕ) : Prop := ∃ (k : ℕ), N = 3^k

theorem probability_log3_integer :
  let total := 900 in
  let favourable := card (finset.filter is_power_of_three (finset.range 1000 \ finset.range 100)) in
  (favourable : ℝ) / total = 1 / 450 :=
by {
  sorry
}

end probability_log3_integer_l248_248556


namespace current_value_l248_248498

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248498


namespace battery_current_l248_248345

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248345


namespace battery_current_l248_248323

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248323


namespace current_at_resistance_12_l248_248298

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248298


namespace battery_current_l248_248337

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248337


namespace current_value_l248_248493

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248493


namespace current_value_l248_248392

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248392


namespace wooden_parallelepiped_length_l248_248930

theorem wooden_parallelepiped_length (n : ℕ) (h1 : n ≥ 7)
    (h2 : ∀ total_cubes unpainted_cubes : ℕ,
      total_cubes = n * (n - 2) * (n - 4) ∧
      unpainted_cubes = (n - 2) * (n - 4) * (n - 6) ∧
      unpainted_cubes = 2 / 3 * total_cubes) :
  n = 18 := 
sorry

end wooden_parallelepiped_length_l248_248930


namespace current_at_resistance_12_l248_248302

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248302


namespace hyperbola_eccentricity_l248_248685

theorem hyperbola_eccentricity :
  ∀ (a b c e : ℝ), 
  (a > 0) ∧ (b > 0) ∧ (a = sqrt 3 * b) ∧ (c^2 = a^2 + b^2) ∧ (e = c / a) 
  → e = 2*sqrt 3 / 3 :=
by sorry

end hyperbola_eccentricity_l248_248685


namespace find_current_when_resistance_is_12_l248_248363

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248363


namespace prince_spent_1000_l248_248953

noncomputable def total_CDs := 200
noncomputable def percent_CDs_10 := 0.40
noncomputable def price_per_CD_10 := 10
noncomputable def price_per_CD_5 := 5

-- Number of CDs sold at $10 each
noncomputable def num_CDs_10 := percent_CDs_10 * total_CDs

-- Number of CDs sold at $5 each
noncomputable def num_CDs_5 := total_CDs - num_CDs_10

-- Number of $10 CDs bought by Prince
noncomputable def prince_CDs_10 := num_CDs_10 / 2

-- Total cost of $10 CDs bought by Prince
noncomputable def cost_CDs_10 := prince_CDs_10 * price_per_CD_10

-- Total cost of $5 CDs bought by Prince
noncomputable def cost_CDs_5 := num_CDs_5 * price_per_CD_5

-- Total amount of money Prince spent
noncomputable def total_spent := cost_CDs_10 + cost_CDs_5

theorem prince_spent_1000 : total_spent = 1000 := by
  -- Definitions from conditions
  have h1 : total_CDs = 200 := rfl
  have h2 : percent_CDs_10 = 0.40 := rfl
  have h3 : price_per_CD_10 = 10 := rfl
  have h4 : price_per_CD_5 = 5 := rfl

  -- Calculations from solution steps (insert sorry to skip actual proofs)
  have h5 : num_CDs_10 = 80 := sorry
  have h6 : num_CDs_5 = 120 := sorry
  have h7 : prince_CDs_10 = 40 := sorry
  have h8 : cost_CDs_10 = 400 := sorry
  have h9 : cost_CDs_5 = 600 := sorry

  show total_spent = 1000
  sorry

end prince_spent_1000_l248_248953


namespace find_current_when_resistance_is_12_l248_248350

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248350


namespace inverse_proportion_symmetric_l248_248627

theorem inverse_proportion_symmetric (a b : ℝ) (h : a ≠ 0) (h_ab : b = -6 / -a) : (-b) = -6 / a :=
by
  -- the proof goes here
  sorry

end inverse_proportion_symmetric_l248_248627


namespace sqrt_factorial_mul_factorial_l248_248209

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l248_248209


namespace brick_height_l248_248878

theorem brick_height
  (wall_l : Real) (wall_w : Real) (wall_h : Real)
  (brick_l : Real) (brick_w : Real) (n_bricks : Nat) :
  wall_l = 750 → wall_w = 600 → wall_h = 22.5 → 
  brick_l = 25 → brick_w = 11.25 → n_bricks = 6000 → 
  ∃ h : Real, brick_l * brick_w * h * n_bricks = wall_l * wall_w * wall_h :=
begin
  intros,
  use 6,
  sorry
end

end brick_height_l248_248878


namespace sqrt_factorial_product_eq_24_l248_248198

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l248_248198


namespace correct_statement_is_D_l248_248974

-- Define each of the initial conditions as hypotheses
def StatementA (T : Triangle) : Prop :=
  ∀ triangle_type, (triangle_type = Acute T → all_altitudes_inside T) ∧ (triangle_type = Obtuse T → ¬ all_altitudes_inside T)

def StatementB : Prop :=
  ∃ ℓ : Line, ∀ P : Point, ∃! m : Line, (m // ℓ) ∧ (P ∈ m)

def StatementC : Prop :=
  ∀ l1 l2 t : Line, (t ∩ l1).nonempty ∧ (t ∩ l2).nonempty → are_parallel l1 l2 → interior_angles_on_same_side_supplementary t l1 l2

def StatementD : Prop :=
  ∀ F : Figure, ∀ T : Translation, shape_and_size_unchanged F T

-- State the proof problem
theorem correct_statement_is_D :
  (StatementA ∨ StatementB ∨ StatementC ∨ StatementD) → StatementD :=
by
  intros h
  cases h
  exact sorry

  cases h
  exact sorry

  cases h
  exact sorry

  exact h

end correct_statement_is_D_l248_248974


namespace sqrt_factorial_mul_factorial_eq_l248_248143

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l248_248143


namespace current_value_l248_248393

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248393


namespace max_students_l248_248798

def num_pens : Nat := 1204
def num_pencils : Nat := 840

theorem max_students (n_pens n_pencils : Nat) (h_pens : n_pens = num_pens) (h_pencils : n_pencils = num_pencils) :
  Nat.gcd n_pens n_pencils = 16 := by
  sorry

end max_students_l248_248798


namespace wooden_parallelepiped_length_l248_248928

theorem wooden_parallelepiped_length (n : ℕ) (h1 : n ≥ 7)
    (h2 : ∀ total_cubes unpainted_cubes : ℕ,
      total_cubes = n * (n - 2) * (n - 4) ∧
      unpainted_cubes = (n - 2) * (n - 4) * (n - 6) ∧
      unpainted_cubes = 2 / 3 * total_cubes) :
  n = 18 := 
sorry

end wooden_parallelepiped_length_l248_248928


namespace sqrt_factorial_mul_factorial_l248_248088

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l248_248088


namespace sum_solutions_abs_eq_twenty_l248_248225

theorem sum_solutions_abs_eq_twenty : 
  (∑ x in {x : ℝ | abs (x - 6)^2 + abs (x - 6) = 20}, x) = 12 := 
by
  sorry

end sum_solutions_abs_eq_twenty_l248_248225


namespace triangles_similar_l248_248745

-- Define vertices A, B, C as vectors in the complex plane
variables (a b c : ℂ) (k : ℝ)

-- Conditions
def P : ℂ := a + k * (b - a)
def Q : ℂ := b + k * (c - b)
def R : ℂ := c + k * (a - c)

def A' : ℂ := a + k * complex.I * (c - b)
def B' : ℂ := b + k * complex.I * (a - c)
def C' : ℂ := c + k * complex.I * (b - a)

-- Proof statement (theorem)
theorem triangles_similar (h : k > 1):
  (|P - Q| = |A' - B'|) ∧ (|P - R| = |A' - C'|) ∧ (|Q - R| = |B' - C'|) ∧ 
  ((arg (P - Q) = arg (A' - B') % (2 * real.pi)) ∧
   (arg (P - R) = arg (A' - C') % (2 * real.pi)) ∧
   (arg (Q - R) = arg (B' - C') % (2 * real.pi))) :=
sorry

end triangles_similar_l248_248745


namespace find_current_when_resistance_is_12_l248_248348

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248348


namespace trams_to_add_l248_248041

theorem trams_to_add (initial_trams : ℕ) (initial_interval new_interval : ℤ)
  (reduce_by_fraction : ℤ) (total_distance : ℤ)
  (h1 : initial_trams = 12)
  (h2 : initial_interval = total_distance / initial_trams)
  (h3 : reduce_by_fraction = 5)
  (h4 : new_interval = initial_interval - initial_interval / reduce_by_fraction) :
  initial_trams + (total_distance / new_interval - initial_trams) = 15 :=
by
  sorry

end trams_to_add_l248_248041


namespace sqrt_factorial_product_l248_248110

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248110


namespace current_value_l248_248434

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248434


namespace current_at_R_12_l248_248245

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248245


namespace ratio_of_areas_l248_248796

noncomputable def side_length_S : ℝ := sorry
noncomputable def side_length_longer_R : ℝ := 1.2 * side_length_S
noncomputable def side_length_shorter_R : ℝ := 0.8 * side_length_S
noncomputable def area_S : ℝ := side_length_S ^ 2
noncomputable def area_R : ℝ := side_length_longer_R * side_length_shorter_R

theorem ratio_of_areas :
  (area_R / area_S) = (24 / 25) :=
by
  sorry

end ratio_of_areas_l248_248796


namespace current_when_resistance_is_12_l248_248411

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248411


namespace subspace_basis_and_dimension_l248_248743

-- Define the subspace F
def F : Set (ℝ × ℝ × ℝ) := { p | p.1 + p.2 + p.3 = 0 }

-- Define the set of basis vectors
def basis_vectors : Set (ℝ × ℝ × ℝ) := {(-1, 1, 0), (-1, 0, 1)}

-- Theorem statement
theorem subspace_basis_and_dimension :
  ∃ (basis : Set (ℝ × ℝ × ℝ)), basis = basis_vectors ∧ Submodule.span ℝ (basis : Set ℝ^3) = (F : Set ℝ^3) ∧ 
  FiniteDimensional.findim ℝ (Submodule.span ℝ (basis : Set ℝ^3)) = 2 := 
by {
  sorry
}

end subspace_basis_and_dimension_l248_248743


namespace sqrt_factorial_product_l248_248169

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248169


namespace sqrt_factorial_equality_l248_248151

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l248_248151


namespace knights_probability_sum_l248_248058

theorem knights_probability_sum (P : ℚ) (P_num P_denom : ℕ) (hP_num : 14_1505) (hP_denom : 142_506) :
  let P := ((141505 : ℕ) / (142506 : ℕ) : ℚ) in
  (P_num / P_denom = P) →
  (P_num + P_denom = 284011) :=
by
  intro P hP
  let P := (141505 / 142506 : ℚ)
  have h1 : P = (141505 / 142506 : ℚ) := by rfl
  have h2 : 141505 + 142506 = 284011 := by norm_num
  rw [h1] at hP
  rw [hP] at h1
  exact h2

end knights_probability_sum_l248_248058


namespace parallelepiped_length_l248_248917

theorem parallelepiped_length :
  ∃ n : ℕ, (n ≥ 7) ∧ (n * (n - 2) * (n - 4) = 3 * ((n - 2) * (n - 4) * (n - 6))) ∧ n = 18 :=
by
  sorry

end parallelepiped_length_l248_248917


namespace total_time_is_134_minutes_l248_248761

-- Definitions for the conditions
def swimming_distance := 1.5 -- km
def cycling_distance := 40  -- km
def running_distance := 10  -- km

def swimming_speed (x : ℝ) := x -- km/min
def running_speed (x : ℝ) := 5 * x -- km/min
def cycling_speed (x : ℝ) := 2.5 * running_speed x -- km/min

def swimming_time (x : ℝ) := swimming_distance / swimming_speed x -- minutes
def running_time (x : ℝ) := running_distance / running_speed x -- minutes
def cycling_time (x : ℝ) := cycling_distance / cycling_speed x -- minutes

-- The given condition in problem
def combined_time_condition (x : ℝ) := (swimming_time x + running_time x = cycling_time x + 6) 

-- Prove the total time for the race is 134 minutes
theorem total_time_is_134_minutes (x : ℝ) (h : combined_time_condition x) : 
  swimming_time x + running_time x + cycling_time x = 134 :=
sorry

end total_time_is_134_minutes_l248_248761


namespace current_when_resistance_is_12_l248_248410

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248410


namespace sqrt_factorial_mul_factorial_l248_248080

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l248_248080


namespace option_C_correct_l248_248215

theorem option_C_correct (a : ℤ) : (a = 3 → a = a + 1 → a = 4) :=
by {
  sorry
}

end option_C_correct_l248_248215


namespace wooden_parallelepiped_length_l248_248927

theorem wooden_parallelepiped_length (n : ℕ) (h1 : n ≥ 7)
    (h2 : ∀ total_cubes unpainted_cubes : ℕ,
      total_cubes = n * (n - 2) * (n - 4) ∧
      unpainted_cubes = (n - 2) * (n - 4) * (n - 6) ∧
      unpainted_cubes = 2 / 3 * total_cubes) :
  n = 18 := 
sorry

end wooden_parallelepiped_length_l248_248927


namespace rectangle_square_area_ratio_l248_248794

theorem rectangle_square_area_ratio (s : ℝ) (hs : s > 0) :
  let area_square := s^2 in
  let area_rectangle := (1.2 * s) * (0.8 * s) in
  area_rectangle / area_square = 24 / 25 :=
by
  sorry

end rectangle_square_area_ratio_l248_248794


namespace complex_apartment_exchange_two_days_l248_248697

-- Define the properties and conditions of the apartment exchange
variables (Family : Type) (Apartment : Type)
variable exchange : Family → Apartment → Apartment → Prop

-- Condition 1: Only paired exchanges are allowed
def paired_exchanges (f1 f2 : Family) (a1 a2 : Apartment) : Prop :=
  exchange f1 a2 a1 ∧ exchange f2 a1 a2

-- Condition 2: Any family can participate in at most one exchange per day
def one_exchange_per_day (f : Family) (a1 a2 a3 a4 : Apartment) : Prop :=
  ¬ (exchange f a1 a2 ∧ exchange f a2 a3 ∧ exchange f a3 a4)

-- Condition 3: Each family occupies one apartment both before and after the exchange
def occupies_one_apartment (f : Family) (a : Apartment) : Prop :=
  ¬ ∃ b : Apartment, b ≠ a ∧ (exchange f a b ∨ exchange f b a)

-- Condition 4: The families remain intact
def families_intact (f : Family) : Prop := true  -- Simplified as this is essentially a tautology

-- Main theorem statement: Any complex apartment exchange can be completed in two days
theorem complex_apartment_exchange_two_days
  (families : List Family) (apartments : List Apartment) 
  (exchanges : (Family → Apartment → Apartment → Prop)) :
  (∀ (f1 f2 : Family), paired_exchanges f1 f2 (apartments.nth! 0) (apartments.nth! 1)) →
  (∀ f, one_exchange_per_day f (apartments.nth! 0) (apartments.nth! 1) (apartments.nth! 2) (apartments.nth! 3)) →
  (∀ f, occupies_one_apartment f (apartments.nth! 0)) →
  (∀ f, families_intact f) →
  ∃ days : List (List (Family → Apartment → Apartment → Prop)),
  days.length = 2 ∧
  ∀ d : (Family → Apartment → Apartment → Prop), d ∈ days → ∀ (f : Family), (exists a1 a2 : Apartment, exchange f a1 a2) :=
sorry

end complex_apartment_exchange_two_days_l248_248697


namespace battery_current_l248_248293

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248293


namespace complement_A_in_U_is_interval_l248_248811

noncomputable theory

namespace ComplementProof

-- Define the universal set U as ℝ
def U := ℝ

-- Define the set A := {x | x^2 + 2x >= 0}
def A : set ℝ := {x | x^2 + 2x ≥ 0}

-- Define the complement of A in U
def complement_U_A : set ℝ := {x | x ∈ U ∧ x ∉ A}

-- The goal is to show that the complement of A in U is (-2, 0)
theorem complement_A_in_U_is_interval :
  complement_U_A = {x | -2 < x ∧ x < 0} :=
by
  sorry

end ComplementProof

end complement_A_in_U_is_interval_l248_248811


namespace battery_current_l248_248381

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248381


namespace routes_A_to_B_in_grid_l248_248673

theorem routes_A_to_B_in_grid : 
  let m := 3 in 
  let n := 3 in 
  let total_moves := m + n in 
  let move_to_right := m in 
  let move_down := n in 
  Nat.choose total_moves move_to_right = 20 := 
by
  let m := 3
  let n := 3
  let total_moves := m + n
  let move_to_right := m
  let move_down := n
  show Nat.choose total_moves move_to_right = 20
  sorry

end routes_A_to_B_in_grid_l248_248673


namespace expression_positive_l248_248971

theorem expression_positive (x : ℝ) : 
  ((x < -3) ∨ (x > 2)) ↔ ((x - 2) * (x + 3) > 0) :=
by 
  split
  sorry

end expression_positive_l248_248971


namespace heartsuit_fraction_l248_248681

-- Define the operation heartsuit
def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

-- Define the proof statement
theorem heartsuit_fraction :
  (heartsuit 2 4) / (heartsuit 4 2) = 2 :=
by
  -- We use 'sorry' to skip the actual proof steps
  sorry

end heartsuit_fraction_l248_248681


namespace find_diameter_l248_248870

noncomputable def path_width : ℝ := 0.25
noncomputable def path_area : ℝ := 3.3379421944391545

def total_area (D : ℝ) : ℝ := Real.pi * ((D / 2 + path_width) ^ 2)
def garden_area (D : ℝ) : ℝ := Real.pi * ((D / 2) ^ 2)
def actual_path_area (D : ℝ) : ℝ := total_area D - garden_area D

theorem find_diameter : 
  ∃ D, abs (4 - D) < 0.001 ∧ abs (actual_path_area D - path_area) < 0.001 := 
sorry

end find_diameter_l248_248870


namespace max_possible_matches_l248_248698

-- Define the conditions and parameters for the problem
variable (n : ℕ) (d : ℕ)
hypothesis h_n : n = 2017
hypothesis h_d : d = 22
hypothesis match_condition : ∀ (u v : ℕ), u < n → v < n → adjacency_matrix u v = 1 → (degree u ≤ d ∨ degree v ≤ d)

-- Define the function to calculate the maximum number of matches
noncomputable def max_matches (n d : ℕ) : ℕ :=
  43890

-- Statement of the theorem
theorem max_possible_matches : 
  max_matches 2017 22 = 43890 := 
by {
  exact rfl
}

end max_possible_matches_l248_248698


namespace current_value_l248_248259

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248259


namespace sqrt_factorial_eq_l248_248131

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l248_248131


namespace circles_C_D_intersect_prob_l248_248061

open MeasureTheory

-- Define the distribution of C_X and D_X
noncomputable def uniform_dist (a b : ℝ) : Measure ℝ := 
  MeasureTheory.measure (Set.Icc a b)

-- Integration bounds and Probability calculation
noncomputable def probability_intersect_C_D : ℝ :=
  ∫ x in 0..3, 
    (min (4 : ℝ) (x + sqrt 5) - max (1 : ℝ) (x - sqrt 5)) / 3

-- Main theorem
theorem circles_C_D_intersect_prob :
  probability_intersect_C_D = (* insert the correct answer here *) :=
begin
  sorry
end

end circles_C_D_intersect_prob_l248_248061


namespace battery_current_l248_248280

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248280


namespace sqrt_factorial_multiplication_l248_248185

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l248_248185


namespace min_value_m_l248_248641

open Finset

def Ai_sets (A : Finset ℕ) : Finset (Finset ℕ) :=
  { B : Finset ℕ | B.card = 5 }

theorem min_value_m 
    (A : Finset ℕ)
    (Ai : Fin n → Finset ℕ)
    (hAi : ∀ i, Ai i ∈ Ai_sets A)
    (hInter : ∀ i j, i ≠ j → (Ai i ∩ Ai j).card ≥ 2)
    (hUnion : A = Finset.bUnion Finset.univ Ai)
    (ki : A → ℕ)
    (hki : ∀ x, ki x = (Finset.filter (λ i, x ∈ Ai i) Finset.univ).card)
    (sum_ki : (A.sum ki) = 50)
    (sum_Cki2 : A.sum (λ x, (ki x).choose 2) ≥ 90) : 
  ∃ m, m = (A.image ki).max' sorry ∧ m ≥ 5 :=
by {
  sorry
}

end min_value_m_l248_248641


namespace current_when_resistance_is_12_l248_248418

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248418


namespace chord_lengths_sum_l248_248881

theorem chord_lengths_sum
  (x y : ℝ)
  (h_circle_eq : x^2 + y^2 - 2*x - 4*y - 20 = 0)
  (h_passes_through : ∃ (x y : ℝ), (x = 1) ∧ (y = -1) ∧ (x^2 + y^2 - 2*x - 4*y - 20 = 0)) :
  let m := 10 in  -- Longest chord passing through (1, -1)
  let n := 8 in   -- Shortest chord passing through (1, -1)
  m + n = 18 :=
by
  sorry

end chord_lengths_sum_l248_248881


namespace current_at_R_12_l248_248249

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248249


namespace tens_digit_1998_pow_2003_minus_1995_l248_248832

theorem tens_digit_1998_pow_2003_minus_1995 :
  (1998 ^ 2003 - 1995) % 100 / 10 % 10 = 0 :=
by
  have h1 : 1998 % 100 = 98 := by norm_num
  have h2 : 1995 % 100 = 95 := by norm_num
  have h3 : 98 ^ 2003 % 100 = 98 := by sorry
  have h4 : (98 - 95) % 100 = 3 := by norm_num
  calc
    (1998 ^ 2003 - 1995) % 100 = ((98 ^ 2003 % 100) - (1995 % 100)) % 100 := by rw [h1, h2]
    ... = (98 - 95) % 100 := by rw [h3]
    ... = 3 := by norm_num
    ... / 10 % 10 = 0 := by norm_num

end tens_digit_1998_pow_2003_minus_1995_l248_248832


namespace problem1_problem2_problem3_l248_248600

-- Problem 1
theorem problem1 (f : ℝ → ℝ) (hf_quad : ∃ a b c, f = λ x, a * x^2 + b * x + c) 
  (hf0 : f 0 = 0) (hf_rec : ∀ x, f (x + 1) = f x + x + 1) : 
  f = λ x, (1 / 2) * x^2 + (1 / 2) * x :=
by sorry

-- Problem 2
theorem problem2 (f : ℝ → ℝ) 
  (hf_def : ∀ x, f (real.sqrt x + 1) = x + 2 * real.sqrt x) : 
  f = λ x, x^2 - 1 :=
by sorry

-- Problem 3
theorem problem3 (f : ℝ → ℝ) (a : ℝ) 
  (hf_def : ∀ x, f x + 2 * f (1 / x) = a * x) : 
  f = λ x, (2 * a) / (3 * x) - (a * x) / 3 :=
by sorry

end problem1_problem2_problem3_l248_248600


namespace cows_milk_problem_l248_248682

variable (x : ℕ)

theorem cows_milk_problem :
  (∃ y : ℕ, y = (x + 4) * x * (x + 1) / ((x + 2) * (x + 3))) :=
begin
  use (x + 4) * x * (x + 1) / ((x + 2) * (x + 3)),
  sorry
end

end cows_milk_problem_l248_248682


namespace current_value_l248_248420

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248420


namespace num_routes_A_to_B_in_3x3_grid_l248_248671

theorem num_routes_A_to_B_in_3x3_grid : 
  let total_moves := 10
  let right_moves := 5
  let down_moves := 5
  let routes_count := Nat.choose total_moves right_moves
  routes_count = 252 := 
by
  let total_moves := 10
  let right_moves := 5
  let down_moves := 5
  let routes_count := Nat.choose total_moves right_moves
  have fact_10 : 10! = 3628800 := by sorry
  have fact_5  : 5!  = 120 := by sorry
  have comb_10_5: Nat.choose 10 5 = 252 := by sorry
  exact comb_10_5

end num_routes_A_to_B_in_3x3_grid_l248_248671


namespace current_value_l248_248272

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248272


namespace sqrt_factorial_multiplication_l248_248179

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l248_248179


namespace convert_1589_base10_to_base3_l248_248595

theorem convert_1589_base10_to_base3 : 
  -- Definition of conversion, base 10 number 1589, and its base 3 equivalent
  (nat.to_digits 3 1589 = [2, 2, 1, 1, 0, 0, 2]) :=
by sorry

end convert_1589_base10_to_base3_l248_248595


namespace current_at_R_12_l248_248253

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248253


namespace solution_set_of_inequality_l248_248233

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_deriv_neg : ∀ x : ℝ, 0 < x → (x^2 + 1) * deriv f x + 2 * x * f x < 0)
  (h_f_neg1_zero : f (-1) = 0) :
  { x : ℝ | f x > 0 } = { x | x < -1 } ∪ { x | 0 < x ∧ x < 1 } := by
  sorry

end solution_set_of_inequality_l248_248233


namespace battery_current_when_resistance_12_l248_248520

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248520


namespace sqrt_factorial_product_l248_248074

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l248_248074


namespace fraction_value_l248_248548

theorem fraction_value (a b : ℚ) (h₁ : b / (a - 2) = 3 / 4) (h₂ : b / (a + 9) = 5 / 7) : b / a = 165 / 222 := 
by sorry

end fraction_value_l248_248548


namespace parallelepiped_length_l248_248909

theorem parallelepiped_length (n : ℕ) :
  (n - 2) * (n - 4) * (n - 6) = 2 * n * (n - 2) * (n - 4) / 3 →
  n = 18 :=
by
  intros h
  sorry

end parallelepiped_length_l248_248909


namespace sqrt_factorial_product_l248_248076

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l248_248076


namespace add_trams_l248_248052

theorem add_trams (total_trams : ℕ) (total_distance : ℝ) (initial_intervals : ℝ) (new_intervals : ℝ) (additional_trams : ℕ) :
  total_trams = 12 → total_distance = 60 → initial_intervals = total_distance / total_trams →
  new_intervals = initial_intervals - (initial_intervals / 5) →
  additional_trams = (total_distance / new_intervals) - total_trams →
  additional_trams = 3 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end add_trams_l248_248052


namespace max_det_of_matrix_is_sqrt_341_l248_248740

noncomputable def v : ℝ × ℝ × ℝ := (3, 2, -2)
noncomputable def w : ℝ × ℝ × ℝ := (2, -1, 4)
noncomputable def u : ℝ × ℝ × ℝ := sorry  -- unit vector will be specified in proof

theorem max_det_of_matrix_is_sqrt_341 :
  ∃ (u : ℝ × ℝ × ℝ), ∥u∥ = 1 → abs (u.1 * ((v.2 * w.3 - v.3 * w.2)) + 
                              u.2 * ((v.3 * w.1 - v.1 * w.3)) + 
                              u.3 * ((v.1 * w.2 - v.2 * w.1))) = sqrt 341 :=
sorry

end max_det_of_matrix_is_sqrt_341_l248_248740


namespace find_lambda_l248_248667

variable (λ : ℝ)

def a : ℝ × ℝ := (3, 2)
def b (λ : ℝ) : ℝ × ℝ := (λ, -1)

-- Definition of parallel vectors
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = k • w

-- The mathematical problem as a Lean theorem
theorem find_lambda (h : parallel (a - 2 • (b λ)) a) : λ = -3/2 := 
sorry

end find_lambda_l248_248667


namespace sqrt_factorial_product_l248_248170

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248170


namespace fraction_of_votes_counted_l248_248212

theorem fraction_of_votes_counted 
  (V C : ℝ) 
  (x : ℝ)
  (h1 : (3/4) * C / (V - C) = 1 / 2)
  (h2 : x = 0.7857142857142856) :
  C / V ≈ 0.3857 :=
by {
  sorry
}

end fraction_of_votes_counted_l248_248212


namespace sum_of_three_numbers_eq_zero_l248_248020

theorem sum_of_three_numbers_eq_zero 
  (a b c : ℝ) 
  (h_sorted : a ≤ b ∧ b ≤ c) 
  (h_median : b = 10) 
  (h_mean_least : (a + b + c) / 3 = a + 20) 
  (h_mean_greatest : (a + b + c) / 3 = c - 10) 
  : a + b + c = 0 := 
by 
  sorry

end sum_of_three_numbers_eq_zero_l248_248020


namespace series_sum_eq_one_fourth_l248_248599

noncomputable def series : ℕ → ℝ
| 0       := 1 / (5 + 1)
| (n + 1) := (2 ^ (n + 1)) / (5 ^ (2 ^ (n + 1)) + 1)

theorem series_sum_eq_one_fourth :
  tsum (λ n, series n) = 1 / 4 :=
begin
  sorry
end

end series_sum_eq_one_fourth_l248_248599


namespace prob_of_selecting_A_or_B_l248_248904

open_locale classical

/-- 
A unit needs to randomly select 2 out of 4 employees (including two people A and B) 
to go on a business trip. 
Prove that the probability that at least one of A and B is selected is 5/6.
-/
theorem prob_of_selecting_A_or_B :
  let employees := {A, B, C, D} in
  let selections := finset.powerset_len 2 (employees.to_finset) in
  let favorable_selections := selections.filter (λ s, ¬ (s ∩ {A, B}).empty) in
  (favorable_selections.card : ℝ) / (selections.card : ℝ) = 5 / 6 :=
by sorry

end prob_of_selecting_A_or_B_l248_248904


namespace sqrt_factorial_equality_l248_248157

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l248_248157


namespace current_value_l248_248263

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248263


namespace terrell_lifting_l248_248778

theorem terrell_lifting :
  (3 * 25 * 10 = 3 * 20 * 12.5) :=
by
  sorry

end terrell_lifting_l248_248778


namespace inverse_proportion_symmetry_l248_248629

theorem inverse_proportion_symmetry (a b : ℝ) :
  (b = - 6 / (-a)) → (-b = - 6 / a) :=
by
  intro h
  sorry

end inverse_proportion_symmetry_l248_248629


namespace battery_current_when_resistance_12_l248_248521

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248521


namespace cost_of_rusted_side_l248_248894

-- Define the conditions
def perimeter (s : ℕ) (l : ℕ) : ℕ :=
  2 * s + 2 * l

def long_side (s : ℕ) : ℕ :=
  3 * s

def cost_per_foot : ℕ :=
  5

-- Given these conditions, we prove the cost of replacing one short side.
theorem cost_of_rusted_side (s l : ℕ) (h1 : perimeter s l = 640) (h2 : l = long_side s) : 
  5 * s = 400 :=
by 
  sorry

end cost_of_rusted_side_l248_248894


namespace mean_score_of_seniors_l248_248822

variables (students total_score : ℕ)
variables (seniors non_seniors : ℕ)
variables (mean_score all_mean_score seniors_mean_score non_seniors_mean_score : ℝ)

-- Conditions
def condition1 : students = 200 := by sorry
def condition2 : all_mean_score = 120 := by sorry
def condition3 : non_seniors = seniors + 0.75 * seniors := by sorry
def condition4 : seniors_mean_score = 1.25 * non_seniors_mean_score := by sorry

-- Total score
def total_score_eq : total_score = students * all_mean_score := by sorry
def total_students_eq : students = seniors + non_seniors := by sorry

-- Given the conditions, prove the mean score of the seniors.
theorem mean_score_of_seniors :
  seniors_mean_score = 136.84 :=
by sorry

end mean_score_of_seniors_l248_248822


namespace length_of_AB_is_10_l248_248712

noncomputable def length_AB (triangleABC : Triangle) (triangleCBD : Triangle) 
  (isoscelesABC : triangleABC.isIsosceles) 
  (isoscelesCBD : triangleCBD.isIsosceles) 
  (perimeterCBD : triangleCBD.perimeter = 22) 
  (perimeterABC : triangleABC.perimeter = 24) 
  (BD_length : triangleCBD.sideBD.length = 8) : Real :=
10

theorem length_of_AB_is_10 :
  ∀ (triangleABC : Triangle) (triangleCBD : Triangle), 
    triangleABC.isIsosceles →
    triangleCBD.isIsosceles →
    triangleCBD.perimeter = 22 →
    triangleABC.perimeter = 24 →
    triangleCBD.sideBD.length = 8 →
    triangleABC.sideAB.length = 10 :=
by
  intro triangleABC triangleCBD isIsoscelesABC isIsoscelesCBD perimeterCBD perimeterABC BD_length
  rw [length_AB triangleABC triangleCBD isIsoscelesABC isIsoscelesCBD perimeterCBD perimeterABC BD_length]
  sorry

end length_of_AB_is_10_l248_248712


namespace current_at_resistance_12_l248_248304

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248304


namespace sqrt_factorial_product_eq_24_l248_248187

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l248_248187


namespace battery_current_at_given_resistance_l248_248491

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248491


namespace AKPM_is_cyclic_l248_248227

open EuclideanGeometry

variables {A B C K L M P : Point}
variable [Plane ℝ A B C]

-- Points K, L, and M lie on sides AB, BC, and AC respectively
axiom K_on_AB : Collinear A B K
axiom L_on_BC : Collinear B C L
axiom M_on_AC : Collinear A C M

-- Given angles
axiom angle_BLK_eq_angle_BAC : ∠ B L K = ∠ B A C
axiom angle_CLM_eq_angle_BAC : ∠ C L M = ∠ B A C

-- Interesection of segments
axiom BM_intersects_CK_at_P : Line.Through B M ∩ Line.Through C K = {P}

-- Prove that quadrilateral AKPM is cyclic
theorem AKPM_is_cyclic :
  CyclicQuadrilateral A K P M :=
sorry

end AKPM_is_cyclic_l248_248227


namespace domain_of_g_is_all_real_l248_248966

def function_g (x : ℝ) : ℝ := 1 / (⌊x^2 - 8 * x + 18⌋)

theorem domain_of_g_is_all_real : ∀ x : ℝ, function_g x = 1 / (⌊x^2 - 8 * x + 18⌋) := 
by
  sorry

end domain_of_g_is_all_real_l248_248966


namespace smaller_circle_arrangement_l248_248897

theorem smaller_circle_arrangement :
  ∀ (side_length : ℝ) (circle_radius : ℝ), side_length = 2 → circle_radius = 1 →
    (∃ n : ℕ, (∀ (k : ℕ), k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 → n = 4) ∧
      (∀ (i j : ℕ), i ≠ j → touches_square_at_one_point side_length circle_radius i ∧
        touches_adjacent_circle circle_radius i j)) := by
  intros side_length circle_radius h_side h_radius
  use 4
  split
  · intros k h
    -- We define the finite set of indices to {1, 2, 3, 4}
    -- showing that n must be 4
    sorry
  · intros i j h_ij
    -- Here we need to demonstrate the touching conditions as per the geometric layout specified
    sorry

end smaller_circle_arrangement_l248_248897


namespace current_value_l248_248261

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248261


namespace volume_of_cube_with_triple_surface_area_l248_248835

theorem volume_of_cube_with_triple_surface_area (V₁ : ℝ) (s₁ : ℝ) (s₂ : ℝ) (V₂ : ℝ) (A₁ : ℝ) (A₂ : ℝ) :
  V₁ = 8 → 
  V₁ = s₁^3 →
  A₁ = 6 * s₁^2 →
  A₂ = 3 * A₁ →
  A₂ = 6 * s₂^2 →
  V₂ = s₂^3 →
  V₂ = 24 * (3.sqrt) :=
by
  sorry

end volume_of_cube_with_triple_surface_area_l248_248835


namespace find_natural_triples_l248_248984

open Nat

noncomputable def satisfies_conditions (a b c : ℕ) : Prop :=
  (a + b) % c = 0 ∧ (b + c) % a = 0 ∧ (c + a) % b = 0

theorem find_natural_triples :
  ∀ (a b c : ℕ), satisfies_conditions a b c ↔
    (∃ a, (a = b ∧ b = c) ∨ 
          (a = b ∧ c = 2 * a) ∨ 
          (b = 2 * a ∧ c = 3 * a) ∨ 
          (b = 3 * a ∧ c = 2 * a) ∨ 
          (a = 2 * b ∧ c = 3 * b) ∨ 
          (a = 3 * b ∧ c = 2 * b)) :=
sorry

end find_natural_triples_l248_248984


namespace sqrt_factorial_mul_factorial_eq_l248_248148

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l248_248148


namespace volume_of_pool_l248_248004

variable (P T V C : ℝ)

/-- 
The volume of the pool is given as P * T divided by percentage C.
The question is to prove that the volume V of the pool equals 90000 cubic feet given:
  P: The hose can remove 60 cubic feet per minute.
  T: It takes 1200 minutes to drain the pool.
  C: The pool was at 80% capacity when draining started.
-/
theorem volume_of_pool (h1 : P = 60) 
                       (h2 : T = 1200) 
                       (h3 : C = 0.80) 
                       (h4 : P * T / C = V) :
  V = 90000 := 
sorry

end volume_of_pool_l248_248004


namespace number_of_nonzero_terms_is_4_l248_248674

-- Define the polynomials
def poly1 (x : ℝ) : ℝ := 2 * x + 5
def poly2 (x : ℝ) : ℝ := 3 * x^2 - x + 4
def poly3 (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + x - 1

-- Define the given polynomial expression
def expression (x : ℝ) : ℝ := (poly1 x) * (poly2 x) - 4 * (poly3 x)

-- The goal is to prove that the number of nonzero terms in the expansion of the expression is 4
theorem number_of_nonzero_terms_is_4 : ∀ x : ℝ, (expression x).terms.nonzero_card = 4 :=
by
  sorry

end number_of_nonzero_terms_is_4_l248_248674


namespace total_pints_l248_248610

-- Define the given conditions as constants
def annie_picked : Int := 8
def kathryn_picked : Int := annie_picked + 2
def ben_picked : Int := kathryn_picked - 3

-- State the main theorem to prove
theorem total_pints : annie_picked + kathryn_picked + ben_picked = 25 := by
  sorry

end total_pints_l248_248610


namespace sqrt_factorial_product_l248_248120

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l248_248120


namespace additional_trams_proof_l248_248045

-- Definitions for the conditions
def initial_tram_count : Nat := 12
def total_distance : Nat := 60
def initial_interval : Nat := total_distance / initial_tram_count
def reduced_interval : Nat := initial_interval - (initial_interval / 5)
def final_tram_count : Nat := total_distance / reduced_interval
def additional_trams_needed : Nat := final_tram_count - initial_tram_count

-- The theorem we need to prove
theorem additional_trams_proof : additional_trams_needed = 3 :=
by
  sorry

end additional_trams_proof_l248_248045


namespace parallelepiped_length_l248_248918

theorem parallelepiped_length :
  ∃ n : ℕ, (n ≥ 7) ∧ (n * (n - 2) * (n - 4) = 3 * ((n - 2) * (n - 4) * (n - 6))) ∧ n = 18 :=
by
  sorry

end parallelepiped_length_l248_248918


namespace battery_current_when_resistance_12_l248_248518

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248518


namespace find_current_when_resistance_is_12_l248_248359

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248359


namespace parallelepiped_length_l248_248912

theorem parallelepiped_length (n : ℕ)
  (h1 : ∃ n : ℕ, n = 18) 
  (h2 : one_third_of_cubes_have_red_faces : (∃ k : ℕ, k = ((n * (n - 2) * (n - 4)) / 3)) 
        ∧ (remaining_unpainted_cubes : (∃ m : ℕ , m = (2 * (n * (n - 2) * (n - 4)) / 3))))
  (h3 : painted_and_cut_into_cubes : (∃ a b c : ℕ, a = n ∧ b = (n - 2) ∧ c = (n - 4)))
  (h4 : all_sides_whole_cm : (∃ d : ℕ , d = n ∧ d = (n - 2) ∧ d = (n - 4))) :
  n = 18 :=
begin
  sorry
end

end parallelepiped_length_l248_248912


namespace oleg_can_find_adjacent_cells_divisible_by_4_l248_248731

theorem oleg_can_find_adjacent_cells_divisible_by_4 :
  ∀ (grid : Fin 22 → Fin 22 → ℕ),
  (∀ i j, 1 ≤ grid i j ∧ grid i j ≤ 22 * 22) →
  ∃ i j k l, ((i = k ∧ (j = l + 1 ∨ j = l - 1)) ∨ ((i = k + 1 ∨ i = k - 1) ∧ j = l)) ∧ ((grid i j + grid k l) % 4 = 0) :=
by
  sorry

end oleg_can_find_adjacent_cells_divisible_by_4_l248_248731


namespace last_student_position_l248_248861

theorem last_student_position (n : ℕ) (h : n = 1991) : 
  let initial_position := 1 in
  let step := 3 in
  let positions (round : ℕ) : List ℕ :=
    (List.range n).map (λ i => initial_position + step * (round * i)) in
  List.length (positions 0) = n - (n / 3) ->
  List.length (positions 1) = (n - (n / 3)) - ((n - (n / 3)) / 3) ->
  -- Continue the same pattern for subsequent rounds
  positions 6 = [1093] :=
sorry

end last_student_position_l248_248861


namespace probability_slope_condition_l248_248736

theorem probability_slope_condition :
  let p := 63
  let q := 128
  let point_of_interest := (3/4, 1/4)
  let unit_square := { (x, y) | 0 <= x ∧ x <= 1 ∧ 0 <= y ∧ y <= 1 }
  let condition_met := { (x, y) ∈ unit_square | y ≥ (3/4) * x - 5/16 }
  let prob := (condition_met.measure / unit_square.measure)
  p + q = 191 := sorry

end probability_slope_condition_l248_248736


namespace weather_on_july_15_l248_248954

theorem weather_on_july_15 
  (T: ℝ) (sunny: Prop) (W: ℝ) (crowded: Prop) 
  (h1: (T ≥ 85 ∧ sunny ∧ W < 15) → crowded) 
  (h2: ¬ crowded) : (T < 85 ∨ ¬ sunny ∨ W ≥ 15) :=
sorry

end weather_on_july_15_l248_248954


namespace sqrt_factorial_mul_factorial_l248_248201

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l248_248201


namespace battery_current_l248_248377

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248377


namespace sqrt_factorial_product_l248_248108

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248108


namespace sqrt_factorial_product_eq_24_l248_248196

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l248_248196


namespace sqrt_factorial_mul_factorial_eq_l248_248146

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l248_248146


namespace LemonadeCalories_l248_248755

noncomputable def total_calories (lemon_juice sugar water honey : ℕ) (cal_per_100g_lemon_juice cal_per_100g_sugar cal_per_100g_honey : ℕ) : ℝ :=
  (lemon_juice / 100) * cal_per_100g_lemon_juice +
  (sugar / 100) * cal_per_100g_sugar +
  (honey / 100) * cal_per_100g_honey

noncomputable def calories_in_250g (total_calories : ℝ) (total_weight : ℕ) : ℝ :=
  (total_calories / total_weight) * 250

theorem LemonadeCalories :
  let lemon_juice := 150
  let sugar := 200
  let water := 300
  let honey := 50
  let cal_per_100g_lemon_juice := 25
  let cal_per_100g_sugar := 386
  let cal_per_100g_honey := 64
  let total_weight := lemon_juice + sugar + water + honey
  let total_cal := total_calories lemon_juice sugar water honey cal_per_100g_lemon_juice cal_per_100g_sugar cal_per_100g_honey
  calories_in_250g total_cal total_weight = 301 :=
by
  sorry

end LemonadeCalories_l248_248755


namespace current_at_resistance_12_l248_248543

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248543


namespace manager_salary_l248_248859

open_locale arithmetic

theorem manager_salary 
  (avg_salary_20 : ℝ)
  (num_employees : ℕ)
  (salary_increase : ℝ)
  (new_avg_salary : ℝ)
  (new_total_salary : ℝ)
  (M : ℝ) :
  avg_salary_20 = 1500 ->
  num_employees = 20 ->
  salary_increase = 600 ->
  new_avg_salary = avg_salary_20 + salary_increase ->
  new_total_salary = (num_employees + 1) * new_avg_salary ->
  M = new_total_salary - (num_employees * avg_salary_20) ->
  M = 14100 := 
  by intro h1 h2 h3 h4 h5 h6; sorry

end manager_salary_l248_248859


namespace battery_current_l248_248278

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248278


namespace marathon_end_time_l248_248891

open Nat

def marathonStart := 15 * 60  -- 3:00 p.m. in minutes (15 hours * 60 minutes)
def marathonDuration := 780    -- Duration in minutes

theorem marathon_end_time : marathonStart + marathonDuration = 28 * 60 := -- 4:00 a.m. in minutes (28 hours * 60 minutes)
  sorry

end marathon_end_time_l248_248891


namespace isosceles_right_triangle_area_invariant_l248_248998

theorem isosceles_right_triangle_area_invariant (ABC : Triangle) (is_isosceles_right : ABC.isIsoscelesRightTriangle) (P : Point)
    (hP_on_hypotenuse : P ∈ ABC.hypotenuseAB)
    (S_ABC : ℝ) (hS_ABC : S_ABC = 1)
    (AP : ℝ) (hAP : AP = p * ABC.hypotenuseAB)
    (area_AA1P : ℝ) (h_area_AA1P : area_AA1P = p^2)
    (area_BB1P : ℝ) (h_area_BB1P : area_BB1P = (1 - p)^2)
    (area_rectangle_CA1PB1 : ℝ) (h_area_rectangle : area_rectangle_CA1PB1 = 2 * p * (1 - p)) :
  ¬ (area_AA1P < 4 / 9 ∧ area_BB1P < 4 / 9 ∧ area_rectangle_CA1PB1 < 4 / 9) :=
sorry

end isosceles_right_triangle_area_invariant_l248_248998


namespace battery_current_l248_248314

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248314


namespace final_percentage_weight_loss_l248_248218

variable (W : ℝ) -- Assume initial weight is W

-- Conditions
def weight_after_loss (W : ℝ) := 0.85 * W
def final_weight_with_clothes (W : ℝ) := weight_after_loss W * 1.02

-- Theorem to prove the final percentage of weight loss
theorem final_percentage_weight_loss : 
  ((W - final_weight_with_clothes W) / W) * 100 = 13.3 := 
by
  unfold weight_after_loss final_weight_with_clothes
  sorry

end final_percentage_weight_loss_l248_248218


namespace ship_graph_correct_l248_248560

-- Define points and conditions of the problem
def Point := ℝ × ℝ -- represent points as tuples

-- Define the distance function
def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the points A, B, C, and Island X
variables (A B C X : Point)
-- Assume the radius of the circular path from A to B centered at X
variables (r : ℝ)

-- Conditions
axiom travel_AB : distance A X = r ∧ distance B X = r
axiom travel_BC : ∀ t ∈ set.Icc 0 1, distance (B.1 + t * (C.1 - B.1), B.2 + t*(C.2 - B.2)) X > r

-- The theorem we need to prove
theorem ship_graph_correct :
  (∀ t ∈ set.Icc 0 1, distance (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2)) X = r)
  ∧
  (∀ t ∈ set.Icc 0 1, distance (B.1 + t * (C.1 - B.1), B.2 + t * (C.2 - B.2)) X > r) :=
sorry

end ship_graph_correct_l248_248560


namespace unique_parallel_plane_thm_infinite_parallel_lines_thm_l248_248060

-- Definition of the problem conditions
def point_outside_plane (P : Type) [AffineSpace ℝ P] (point : P) (plane : AffineSubspace ℝ P) : Prop :=
  ∀ p ∈ plane, point ≠ p

noncomputable def unique_parallel_plane (P : Type) [AffineSpace ℝ P] (point : P) (plane : AffineSubspace ℝ P) 
[point : point_outside_plane P point plane] : AffineSubspace ℝ P :=
  sorry

noncomputable def infinite_parallel_lines (P : Type) [AffineSpace ℝ P] (point : P) (plane : AffineSubspace ℝ P) 
[point : point_outside_plane P point plane] : set (Line ℝ P) :=
  sorry

-- Theorem statements
theorem unique_parallel_plane_thm (P : Type) [AffineSpace ℝ P] (point : P) (plane : AffineSubspace ℝ P) 
[point : point_outside_plane P point plane] : ∃! p', p' = unique_parallel_plane P point plane :=
begin
  sorry
end

theorem infinite_parallel_lines_thm (P : Type) [AffineSpace ℝ P] (point : P) (plane : AffineSubspace ℝ P) 
[point : point_outside_plane P point plane] : ∃ l_set, l_set = infinite_parallel_lines P point plane ∧ infinite l_set :=
begin
  sorry
end

end unique_parallel_plane_thm_infinite_parallel_lines_thm_l248_248060


namespace fraction_written_form_reading_of_100_45_l248_248232

-- Definitions based on conditions
def decimal_reading (n : ℕ) (d : ℕ) : String :=
  if n = 0 then
    "zero point " ++ (d.toString)
  else
    n.toString ++ " point " ++ (d.toString)

-- Questions and proofs
theorem fraction_written_form : (7 / 16) = (7 / 16) :=
by
  sorry

theorem reading_of_100_45 : decimal_reading 100 45 = "100 point 45" :=
by
  sorry

end fraction_written_form_reading_of_100_45_l248_248232


namespace total_distance_from_top_to_bottom_l248_248898

-- Define the conditions of the problem
def ring_thickness := 1.5
def top_ring_outer_diameter := 25
def bottom_ring_outer_diameter := 4

-- The problem statement
theorem total_distance_from_top_to_bottom : 
  ring_thickness > 0 → 
  top_ring_outer_diameter > bottom_ring_outer_diameter → 
  (∀ n : ℕ, bottom_ring_outer_diameter ≤ top_ring_outer_diameter - n) → 
  ∑ i in finset.range 22, 1.5 = 253 :=
by 
  sorry

end total_distance_from_top_to_bottom_l248_248898


namespace locus_is_semicircle_l248_248812

noncomputable def locus_of_point_C (O A B C : Point) (k : ℝ) 
  (h_right_angle : is_right_angle O A B)
  (h_const_sum : (dist O A) + (dist O B) = k)
  (h_diameter : is_diameter O A B C) 
  (OC_parallel_AB : parallel (line_through O C) (line_through A B)) : Set Point :=
  { C : Point | exists A B, (dist O A + dist O B = k) ∧ (is_on_circle_with_diameter A B C) ∧ parallel (line_through O C) (line_through A B)}

theorem locus_is_semicircle {O A B C : Point} (k : ℝ)
  (h_right_angle : is_right_angle O A B)
  (h_const_sum : (dist O A) + (dist O B) = k)
  (h_diameter : is_diameter O A B C) 
  (OC_parallel_AB : parallel (line_through O C) (line_through A B)) :
  locus_of_point_C O A B C k h_right_angle h_const_sum h_diameter OC_parallel_AB = 
  { C : Point | is_on_semicircle_above_line {
                      center := midpoint A B,
                      radius := dist (midpoint A B) A,
                      endpoints := (A, B),
                      passes_through := O 
                                                    } } :=
sorry

end locus_is_semicircle_l248_248812


namespace binomial_expansion_eq_one_l248_248751

-- Definitions and conditions
def binomial_expansion (n : ℕ) : ℤ :=
  ∑ k in Finset.range (n + 1), (-1)^k * (Nat.choose n k) * (2^(n - k))

theorem binomial_expansion_eq_one (n : ℕ) : binomial_expansion n = 1 := by
  sorry

end binomial_expansion_eq_one_l248_248751


namespace only_prime_triplet_is_2_5_7_l248_248985

open Nat

def isPrimeTriplet (p q r : ℕ) : Prop :=
  p < q ∧ q < r ∧ Prime p ∧ Prime q ∧ Prime r ∧ Prime (abs (q - p)) ∧ Prime (abs (r - q)) ∧ Prime (abs (r - p))

theorem only_prime_triplet_is_2_5_7 :
  ∀ p q r : ℕ, isPrimeTriplet p q r → (p = 2 ∧ q = 5 ∧ r = 7) :=
by
  intros p q r h
  sorry

end only_prime_triplet_is_2_5_7_l248_248985


namespace tangent_line_equation_l248_248619

noncomputable def tangent_line (x : ℝ) : ℝ := 2 * x + 1

def curve (x : ℝ) : ℝ := (x + 1) * Real.exp x

def point_of_tangency : (ℝ × ℝ) := (0, 1)

theorem tangent_line_equation :
  tangent_line = λ x, 2 * x + 1 :=
by sorry

end tangent_line_equation_l248_248619


namespace sqrt_factorial_product_l248_248102

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l248_248102


namespace sqrt_factorial_product_l248_248112

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248112


namespace problem_l248_248715

-- Definitions of the parametric curve C
def curve_C_parametric (α : ℝ) : ℝ × ℝ := ⟨2 * Real.cos α, Real.sin α⟩

-- Polar equation of line l
def line_l_polar (ρ θ : ℝ) : Prop := 2 * ρ * Real.cos θ - ρ * Real.sin θ + 2 = 0

-- Cartesian equation of curve C
def curve_C_cartesian (x y : ℝ) : Prop := (x^2 / 4) + (y^2) = 1

-- Rectangular coordinate equation of line l
def line_l_rectangular (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Main statement to be proved.
theorem problem (A B P: ℝ × ℝ) (α : ℝ) (θ ρ: ℝ) :
  curve_C_cartesian (2 * Real.cos α) (Real.sin α) →
  line_l_polar ρ θ →
  line_l_rectangular A.1 A.2 →
  line_l_rectangular B.1 B.2 →
  P = (0, 2) →
  let PA := Real.sqrt ((0 - A.1)^2 + (2 - A.2)^2),
      PB := Real.sqrt ((0 - B.1)^2 + (2 - B.2)^2) in
  (1 / PA + 1 / PB = (8 * Real.sqrt 5) / 15)
:= sorry

end problem_l248_248715


namespace valid_sequences_count_l248_248705

-- Definitions based on given conditions:
def num_contestants : Nat := 6
def num_females : Nat := 3
def num_males : Nat := 3
def contestant_A_cannot_be_first (seq : List Nat) : Prop := seq.head ≠ 1
def males_cannot_be_consecutive (seq : List Nat) : Prop := ∀ i, i < seq.length - 1 → (seq[i] > 3 → seq[i + 1] ≤ 3)

-- The theorem to prove that the number of valid sequences is 132:
theorem valid_sequences_count : 
  ∃ (seqs : List (List Nat)), 
    (∀ seq ∈ seqs, length seq = num_contestants ∧ contestant_A_cannot_be_first seq ∧ males_cannot_be_consecutive seq) ∧
    seqs.length = 132 :=
by
  sorry

end valid_sequences_count_l248_248705


namespace snatch_percentage_increase_l248_248728

-- Define the problem conditions
def initial_clean_jerk : ℝ := 80
def initial_snatch : ℝ := 50
def new_combined_total : ℝ := 250
def new_clean_jerk : ℝ := 2 * initial_clean_jerk
def new_snatch : ℝ := new_combined_total - new_clean_jerk
def percentage_increase := ((new_snatch - initial_snatch) / initial_snatch) * 100

-- Prove the percentage increase in Snatch weight is 80%
theorem snatch_percentage_increase : percentage_increase = 80 := by
  sorry

end snatch_percentage_increase_l248_248728


namespace sqrt_factorial_product_l248_248118

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l248_248118


namespace square_triangle_ratios_l248_248587

theorem square_triangle_ratios (s t : ℝ) 
  (P_s := 4 * s) 
  (R_s := s * Real.sqrt 2 / 2)
  (P_t := 3 * t) 
  (R_t := t * Real.sqrt 3 / 3) 
  (h : s = t) : 
  (P_s / P_t = 4 / 3) ∧ (R_s / R_t = Real.sqrt 6 / 2) := 
by
  sorry

end square_triangle_ratios_l248_248587


namespace battery_current_l248_248286

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248286


namespace quadratic_trinomial_inequality_l248_248639

variables (a b c : ℝ) (n : ℕ)
variables (x : ℕ → ℝ)

def f (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem quadratic_trinomial_inequality
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a + b + c = 1)
  (h_prod_eq_one : ∏ i in finset.range n, x i = 1)
  (h_x_pos : ∀ i, 0 < x i) :
  ∏ i in finset.range n, f (x i) ≥ 1 :=
by {
  sorry
}

end quadratic_trinomial_inequality_l248_248639


namespace probability_decreasing_order_l248_248871

noncomputable def probability_strictly_decreasing_order : ℚ :=
let total_permutations := 20^5 in
let favorable_outcome := 1 in
favorable_outcome / total_permutations

theorem probability_decreasing_order :
  probability_strictly_decreasing_order = 1 / 1397760 :=
sorry

end probability_decreasing_order_l248_248871


namespace chocolates_distribution_l248_248901

theorem chocolates_distribution :
  ∀ (total_chocolates : ℕ) (num_children num_boys num_girls chocolates_per_boy : ℕ),
  total_chocolates = 3000 →
  num_children = 120 →
  num_boys = 60 →
  num_girls = 60 →
  chocolates_per_boy = 2 →
  num_boys + num_girls = num_children →
  ∃ chocolates_per_girl,
    num_girls * chocolates_per_girl =
    total_chocolates - (num_boys * chocolates_per_boy) ∧
    chocolates_per_girl = 48 :=
by
  intros total_chocolates num_children num_boys num_girls chocolates_per_boy
  intros total_chocolates_eq num_children_eq num_boys_eq num_girls_eq chocolates_per_boy_eq sum_eq
  use 48
  split
  {
    calc
      num_girls * 48 = 60 * 48 : by rw [num_girls_eq]
      ... = 2880 : by norm_num
      ... = 3000 - 120 : by rw [←total_chocolates_eq]; norm_num
      ... = total_chocolates - (num_boys * chocolates_per_boy) : by rw [chocolates_per_boy_eq, num_boys_eq]
  }
  {
    refl
  }

end chocolates_distribution_l248_248901


namespace sqrt_factorial_product_l248_248091

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l248_248091


namespace find_first_number_l248_248555

theorem find_first_number (x : ℝ) : (x + 16 + 8 + 22) / 4 = 13 ↔ x = 6 :=
by 
  sorry

end find_first_number_l248_248555


namespace battery_current_l248_248342

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248342


namespace find_current_when_resistance_is_12_l248_248357

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248357


namespace current_value_l248_248394

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248394


namespace difference_between_largest_and_smallest_l248_248850

def largest_number := 9765310
def smallest_number := 1035679
def expected_difference := 8729631
def digits := [3, 9, 6, 0, 5, 1, 7]

theorem difference_between_largest_and_smallest :
  (largest_number - smallest_number) = expected_difference :=
sorry

end difference_between_largest_and_smallest_l248_248850


namespace length_of_parallelepiped_l248_248922

def number_of_cubes_with_painted_faces (n : ℕ) := (n - 2) * (n - 4) * (n - 6) 
def total_number_of_cubes (n : ℕ) := n * (n - 2) * (n - 4)

theorem length_of_parallelepiped (n : ℕ) (h1 : total_number_of_cubes n = 3 * number_of_cubes_with_painted_faces n) : 
  n = 18 :=
by 
  sorry

end length_of_parallelepiped_l248_248922


namespace find_current_when_resistance_is_12_l248_248364

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248364


namespace current_at_resistance_12_l248_248305

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248305


namespace battery_current_when_resistance_12_l248_248516

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248516


namespace current_at_resistance_12_l248_248301

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248301


namespace find_current_when_resistance_is_12_l248_248354

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248354


namespace whitewashing_cost_l248_248788

noncomputable def cost_of_whitewashing (l w h : ℝ) (c : ℝ) (door_area window_area : ℝ) (num_windows : ℝ) : ℝ :=
  let perimeter := 2 * (l + w)
  let total_wall_area := perimeter * h
  let total_window_area := num_windows * window_area
  let total_paintable_area := total_wall_area - (door_area + total_window_area)
  total_paintable_area * c

theorem whitewashing_cost:
  cost_of_whitewashing 25 15 12 6 (6 * 3) (4 * 3) 3 = 5436 := by
  sorry

end whitewashing_cost_l248_248788


namespace range_of_k_l248_248660

open Real

noncomputable def h (x : ℝ) : ℝ := -2 * log x / x

theorem range_of_k :
  let f : ℝ → ℝ := λ x, k * x,
      g : ℝ → ℝ := λ x, (1 / exp 1) ^ (x / 2) in
  (∃ x₁ x₂, (1 / exp 1) ≤ x₁ ∧ x₁ ≤ exp 2 ∧ (1 / exp 1) ≤ x₂ ∧ x₂ ≤ exp 2
   ∧ f x₁ = y ∧ g x₂ = y ∧ y = y ∧ x₁ + x₂ = 1) -> 
   λ k, -2 / (exp 1 : ℝ) ≤ k ∧ k ≤ 2 * exp 1 :=
begin
  sorry
end

end range_of_k_l248_248660


namespace current_value_l248_248496

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248496


namespace composition_evaluation_l248_248742

noncomputable def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 10) / (x^2 - x + 5)
def g (x : ℚ) : ℚ := x^2 - 1

theorem composition_evaluation (x : ℚ) :
  f(g(x)) + g(f(x)) = (3136 : ℚ) / (539 : ℚ) :=
  by
  sorry

end composition_evaluation_l248_248742


namespace find_x_l248_248708

-- Given angles and parallel lines
def angle_ACE : ℝ := 180
def angle_ECD : ℝ := 105
def angle_ADC : ℝ := 115
def AB_parallel_DC : Prop := ∀ (P Q : ℝ), P = Q  -- placeholder for parallel lines condition

-- Define the angles in the problem
def angle_ACB : ℝ := angle_ACE - angle_ECD 
def angle_BAC : ℝ := 180 - angle_ACB - angle_ACB
def angle_ACD : ℝ := angle_BAC               -- by parallel condition
def angle_DAC : ℝ := 180 - angle_ADC - angle_ACD

-- The value of x we need to prove
def x := angle_DAC

theorem find_x (AB_parallel_DC : AB_parallel_DC) (angle_ACE : angle_ACE = 180)
  (angle_ECD : angle_ECD = 105) (angle_ADC : angle_ADC = 115):
  x = 35 :=
by
  unfold x angle_DAC angle_ADC  angle_ACD angle_BAC angle_ACB
  -- Calculation for each angle based on the given conditions and basic arithmetic
  rw [angle_ACE, angle_ECD]
  have h1 : angle_ACB = 180 - 105 := rfl
  rw [h1]
  have h2 : angle_ACB = 75 := by norm_num
  rw [h2]
  have h3 : angle_BAC = 180 - 75 - 75 := rfl
  rw [h3]
  have h4 : angle_BAC = 30 := by norm_num
  rw [h4]
  have h5 : angle_ACD = 30 := rfl
  rw [h5]
  have h6 : angle_DAC = 180 - 115 - 30 := rfl
  rw [h6]
  have h7 : angle_DAC = 35 := by norm_num
  rw [h7]
  exact rfl

-- Sorry to skip the proof of the theorem.

end find_x_l248_248708


namespace current_value_l248_248495

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248495


namespace min_f_value_l248_248637

noncomputable def f (x y : ℝ) := abs (2 * x - 3 * y - 12)

theorem min_f_value :
  ∃ x y : ℝ, 
  let z1 := x + Real.sqrt 5 - y * Complex.I,
      z2 := x - Real.sqrt 5 + y * Complex.I in
  Complex.abs z1 + Complex.abs z2 = 6 ∧
  f x y = 12 - 6 * Real.sqrt 2 :=
sorry

end min_f_value_l248_248637


namespace sqrt_factorial_product_l248_248124

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l248_248124


namespace sqrt_factorial_mul_factorial_eq_l248_248141

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l248_248141


namespace total_bill_cost_l248_248003

theorem total_bill_cost
  (half_off : ℝ → ℝ := λ x, x / 2)
  (salisbury_steak_cost : ℝ := 16)
  (chicken_fried_steak_cost : ℝ := 18)
  (ate_between_2_and_4_pm : true) :
  half_off salisbury_steak_cost + half_off chicken_fried_steak_cost = 17 :=
by
  sorry

end total_bill_cost_l248_248003


namespace battery_current_at_given_resistance_l248_248478

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248478


namespace find_current_l248_248444

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248444


namespace percentage_increase_l248_248650

variable (x y p : ℝ)

theorem percentage_increase (h : x = y + (p / 100) * y) : p = 100 * ((x - y) / y) := 
by 
  sorry

end percentage_increase_l248_248650


namespace heat_production_example_l248_248970

noncomputable def heat_produced_by_current (R : ℝ) (I : ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
∫ (t : ℝ) in t1..t2, (I t)^2 * R

theorem heat_production_example :
  heat_produced_by_current 40 (λ t => 5 + 4 * t) 0 10 = 303750 :=
by
  sorry

end heat_production_example_l248_248970


namespace battery_current_l248_248312

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248312


namespace skt_lineups_l248_248033

theorem skt_lineups :
  let total_progamer_count : ℕ := 111,
      initial_team_size : ℕ := 11,
      lineup_size : ℕ := 5,
      new_progamers_count : ℕ := total_progamer_count - initial_team_size,
      case1_count := Nat.choose initial_team_size lineup_size,
      case2_count := Nat.choose initial_team_size (lineup_size - 1) * new_progamers_count,
      total_unordered_lineups := case1_count + case2_count,
      ordered_lineups := total_unordered_lineups * lineup_size.factorial
  in ordered_lineups = 4015440 :=
by
  sorry

end skt_lineups_l248_248033


namespace trams_to_add_l248_248040

theorem trams_to_add (initial_trams : ℕ) (initial_interval new_interval : ℤ)
  (reduce_by_fraction : ℤ) (total_distance : ℤ)
  (h1 : initial_trams = 12)
  (h2 : initial_interval = total_distance / initial_trams)
  (h3 : reduce_by_fraction = 5)
  (h4 : new_interval = initial_interval - initial_interval / reduce_by_fraction) :
  initial_trams + (total_distance / new_interval - initial_trams) = 15 :=
by
  sorry

end trams_to_add_l248_248040


namespace battery_current_l248_248328

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248328


namespace x_squared_y_cubed_eq_200_l248_248648

theorem x_squared_y_cubed_eq_200 (x y : ℕ) (h : 2^x * 9^y = 200) : x^2 * y^3 = 200 := by
  sorry

end x_squared_y_cubed_eq_200_l248_248648


namespace trams_to_add_l248_248042

theorem trams_to_add (initial_trams : ℕ) (initial_interval new_interval : ℤ)
  (reduce_by_fraction : ℤ) (total_distance : ℤ)
  (h1 : initial_trams = 12)
  (h2 : initial_interval = total_distance / initial_trams)
  (h3 : reduce_by_fraction = 5)
  (h4 : new_interval = initial_interval - initial_interval / reduce_by_fraction) :
  initial_trams + (total_distance / new_interval - initial_trams) = 15 :=
by
  sorry

end trams_to_add_l248_248042


namespace current_at_resistance_12_l248_248300

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248300


namespace find_current_when_resistance_is_12_l248_248358

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248358


namespace problem_1_1_eval_l248_248238

noncomputable def E (a b c : ℝ) : ℝ :=
  let A := (1/a - 1/(b+c))/(1/a + 1/(b+c))
  let B := 1 + (b^2 + c^2 - a^2)/(2*b*c)
  let C := (a - b - c)/(a * b * c)
  (A * B) / C

theorem problem_1_1_eval :
  E 0.02 (-11.05) 1.07 = 0.1 :=
by
  -- Proof goes here
  sorry

end problem_1_1_eval_l248_248238


namespace find_current_l248_248446

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248446


namespace find_current_l248_248450

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248450


namespace sqrt_factorial_eq_l248_248138

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l248_248138


namespace wire_length_before_cutting_l248_248906

theorem wire_length_before_cutting (L S : ℝ) (h1 : S = 40) (h2 : S = 2 / 5 * L) : L + S = 140 :=
by
  sorry

end wire_length_before_cutting_l248_248906


namespace least_positive_n_l248_248746

def sequence (n : ℕ) : ℕ :=
  if n = 5 then 5
  else if n > 5 then 50 * sequence (n-1) + n^2
  else 0 -- handling case n < 5 just to make it total, though it won't be used

theorem least_positive_n :
  ∃ n > 5, sequence n % 101 = 0 ∧ ∀ m, m > 5 ∧ m < n → sequence m % 101 ≠ 0 := sorry

end least_positive_n_l248_248746


namespace volume_of_pyramid_SPQR_is_264_l248_248762

noncomputable def volume_of_pyramid_SPQR
  (P Q R S : Type)
  [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S]
  (SP SQ SR : ℝ)
  (h_perpendicular1 : SP ⊥ SQ)
  (h_perpendicular2 : SQ ⊥ SR)
  (h_perpendicular3 : SR ⊥ SP)
  (h_SP : SP = 12)
  (h_SQ : SQ = 12)
  (h_SR : SR = 11) : ℝ :=
264

theorem volume_of_pyramid_SPQR_is_264
  (P Q R S : Type)
  [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S]
  (SP SQ SR : ℝ)
  (h_perpendicular1 : SP ⊥ SQ)
  (h_perpendicular2 : SQ ⊥ SR)
  (h_perpendicular3 : SR ⊥ SP)
  (h_SP : SP = 12)
  (h_SQ : SQ = 12)
  (h_SR : SR = 11) :
  volume_of_pyramid_SPQR P Q R S SP SQ SR h_perpendicular1 h_perpendicular2 h_perpendicular3 h_SP h_SQ h_SR = 264 := 
sorry

end volume_of_pyramid_SPQR_is_264_l248_248762


namespace count_irrationals_equal_two_l248_248572

-- Definitions of irrational numbers based on conditions
def is_irrational (x : ℝ) : Prop :=
  ∀ (r : ℚ), x ≠ r ∧ ¬(∃ (a b : ℤ), b ≠ 0 ∧ x = a / b)

-- Given set of numbers
def numbers : set ℝ := {0, 3, -real.sqrt 6, 35, real.pi, 23 / 7, 3.14}

-- Function to count irrational numbers in the set
noncomputable def count_irrationals (s : set ℝ) : ℕ :=
  s.to_finset.countp is_irrational

-- Main theorem statement
theorem count_irrationals_equal_two : count_irrationals numbers = 2 :=
by 
  sorry

end count_irrationals_equal_two_l248_248572


namespace smallest_number_of_fruits_l248_248978

theorem smallest_number_of_fruits 
  (n_apple_slices : ℕ) (n_grapes : ℕ) (n_orange_wedges : ℕ) (n_cherries : ℕ)
  (h_apple : n_apple_slices = 18)
  (h_grape : n_grapes = 9)
  (h_orange : n_orange_wedges = 12)
  (h_cherry : n_cherries = 6)
  : ∃ (n : ℕ), n = 36 ∧ (n % n_apple_slices = 0) ∧ (n % n_grapes = 0) ∧ (n % n_orange_wedges = 0) ∧ (n % n_cherries = 0) :=
sorry

end smallest_number_of_fruits_l248_248978


namespace battery_current_at_given_resistance_l248_248487

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248487


namespace paper_folding_ratio_l248_248562

theorem paper_folding_ratio :
  ∃ (side length small_perim large_perim : ℕ), 
    side_length = 6 ∧ 
    small_perim = 2 * (3 + 3) ∧ 
    large_perim = 2 * (6 + 3) ∧ 
    small_perim / large_perim = 2 / 3 :=
by sorry

end paper_folding_ratio_l248_248562


namespace average_first_n_odd_numbers_l248_248008

theorem average_first_n_odd_numbers (n : ℕ) (h1 : ∀ i, i < n → 2*i + 1 = 2*Nat.succ i - 1):
  (∑ i in Finset.range n, (2 * i + 1)) / n = 2 * n - 1 := by
sorry

end average_first_n_odd_numbers_l248_248008


namespace battery_current_l248_248282

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248282


namespace problem_proof_l248_248654

noncomputable def a (n : ℕ) : ℕ := 3 * n - 2

def b (n : ℕ) : ℚ := 3 / ((n + 1) * (a n + 2))

def S (n : ℕ) : ℚ := ∑ k in finset.range n + 1, b k

theorem problem_proof (n : ℕ) (h : 0 < n) :
  (∀ n : ℕ, 0 < n → a n = 3 * n - 2) ∧
  S n = n / (n + 1) :=
by
  sorry

end problem_proof_l248_248654


namespace battery_current_when_resistance_12_l248_248510

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248510


namespace sqrt_factorial_product_l248_248166

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248166


namespace battery_current_l248_248383

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248383


namespace inequality_solution_l248_248614

theorem inequality_solution (x : ℝ) : 
    (2 / (x + 1) + 10 / (x + 4) ≥ 3 / (x + 2)) ↔ (x ∈ set.Ioo (-4) (-1) ∪ set.Ioi (-4 / 3)) :=
sorry

end inequality_solution_l248_248614


namespace parallelepiped_length_l248_248908

theorem parallelepiped_length (n : ℕ) :
  (n - 2) * (n - 4) * (n - 6) = 2 * n * (n - 2) * (n - 4) / 3 →
  n = 18 :=
by
  intros h
  sorry

end parallelepiped_length_l248_248908


namespace inequality_solution_l248_248623

theorem inequality_solution :
  ∃ a b : ℕ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → sqrt (1 - x) + sqrt (1 + x) ≤ 2 - b * x ^ a) ∧ a = 2 ∧ b = 1 / 4 :=
by
  sorry

end inequality_solution_l248_248623


namespace battery_current_l248_248318

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248318


namespace volume_increase_factor_l248_248840

variable (π : ℝ) (r h : ℝ)

def original_volume : ℝ := π * r^2 * h

def new_volume : ℝ := π * (2 * r)^2 * (3 * h)

theorem volume_increase_factor : new_volume π r h = 12 * original_volume π r h :=
by
  -- Here we would include the proof that new_volume = 12 * original_volume
  sorry

end volume_increase_factor_l248_248840


namespace current_at_resistance_12_l248_248528

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248528


namespace current_value_l248_248502

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248502


namespace evaluate_trig_expression_l248_248980

theorem evaluate_trig_expression :
  (sin 15 - cos 15) / (sin 15 + cos 15) = -real.sqrt 3 / 3 :=
by sorry

end evaluate_trig_expression_l248_248980


namespace arrangement_count_is_36_l248_248934

def people : List ℕ := [1, 2, 3, 4, 5]

-- Key idea: Represent A, B, C, D, E as 1, 2, 3, 4, 5 respectively
noncomputable def validArrangements: Nat :=
  (people.permutations.toList.filter (λ l =>
    ¬((l.indexOf 1 + 1 == l.indexOf 3) ∨ (l.indexOf 1 - 1 == l.indexOf 3) ∨
      (l.indexOf 2 + 1 == l.indexOf 3) ∨ (l.indexOf 2 - 1 == l.indexOf 3)))).length

theorem arrangement_count_is_36 : validArrangements = 36 :=
  sorry

end arrangement_count_is_36_l248_248934


namespace cubic_ineq_l248_248764

theorem cubic_ineq (x p q : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 := 
  sorry

end cubic_ineq_l248_248764


namespace constant_term_of_binomial_expansion_l248_248011

noncomputable def constant_term_binomial (x : ℝ) : ℝ :=
  ∑ r in finset.range 10.succ, (nat.choose 10 r) * (x^2)^(10-r) * (1 / (2 * real.sqrt x))^r

theorem constant_term_of_binomial_expansion :
  constant_term_binomial 1 = 45 / 256 := sorry

end constant_term_of_binomial_expansion_l248_248011


namespace domain_of_function_domain_is_real_l248_248987

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 4*x + 5)

theorem domain_of_function : ∀ x : ℝ, (x^2 + 4*x + 5) ≠ 0 :=
by
  intro x
  have h : x^2 + 4*x + 5 = (x + 2) ^ 2 + 1 := by ring
  rw [h]
  apply ne_of_gt
  exact add_pos (sq_pos_of_ne_zero (x + 2) (by linarith)) zero_lt_one

theorem domain_is_real : ∀ x : ℝ, f x ∈ ℝ := by
  intro x
  exact (domain_of_function x).elim

end domain_of_function_domain_is_real_l248_248987


namespace generalized_inequality_l248_248647

theorem generalized_inequality (x : ℝ) (n : ℕ) (h1 : x > 0) : x^n + (n : ℝ) / x > n + 1 := 
sorry

end generalized_inequality_l248_248647


namespace current_at_R_12_l248_248243

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248243


namespace current_at_R_12_l248_248241

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248241


namespace current_value_l248_248258

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248258


namespace coeff_x_squared_mul_l248_248830

def poly1 : ℤ[X] := -3 * X^3 - 4 * X^2 - 8 * X + 2
def poly2 : ℤ[X] := -2 * X^2 - 7 * X + 3

theorem coeff_x_squared_mul (p1 p2 : ℤ[X]) (h1 : p1 = poly1) (h2 : p2 = poly2) :
  coeff (p1 * p2) 2 = 40 :=
by sorry

end coeff_x_squared_mul_l248_248830


namespace cylinder_volume_increase_l248_248842

theorem cylinder_volume_increase (r h : ℝ) (V : ℝ) : 
  V = π * r^2 * h → 
  (3 * h) * (2 * r)^2 * π = 12 * V := by
    sorry

end cylinder_volume_increase_l248_248842


namespace total_heads_l248_248889

def number_of_heads := 1
def number_of_feet_hen := 2
def number_of_feet_cow := 4
def total_feet := 144

theorem total_heads (H C : ℕ) (h_hens : H = 24) (h_feet : number_of_feet_hen * H + number_of_feet_cow * C = total_feet) :
  H + C = 48 :=
sorry

end total_heads_l248_248889


namespace battery_current_at_given_resistance_l248_248485

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248485


namespace add_trams_l248_248050

theorem add_trams (total_trams : ℕ) (total_distance : ℝ) (initial_intervals : ℝ) (new_intervals : ℝ) (additional_trams : ℕ) :
  total_trams = 12 → total_distance = 60 → initial_intervals = total_distance / total_trams →
  new_intervals = initial_intervals - (initial_intervals / 5) →
  additional_trams = (total_distance / new_intervals) - total_trams →
  additional_trams = 3 :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end add_trams_l248_248050


namespace determine_p_and_q_l248_248608

noncomputable def find_p_and_q (a : ℝ) (p q : ℝ) : Prop :=
  (∀ x : ℝ, x = 1 ∨ x = -1 → (x^4 + p * x^2 + q * x + a^2 = 0))

theorem determine_p_and_q (a p q : ℝ) (h : find_p_and_q a p q) : p = -(a^2 + 1) ∧ q = 0 :=
by
  -- The proof would go here.
  sorry

end determine_p_and_q_l248_248608


namespace kids_stay_home_correct_l248_248977

def total_number_of_kids : ℕ := 1363293
def kids_who_go_to_camp : ℕ := 455682
def kids_staying_home : ℕ := total_number_of_kids - kids_who_go_to_camp

theorem kids_stay_home_correct :
  kids_staying_home = 907611 := by 
  sorry

end kids_stay_home_correct_l248_248977


namespace expectation_linear_transformation_correct_variance_linear_transformation_incorrect_binomial_distribution_probability_correct_normal_distribution_probability_correct_l248_248845

-- Definitions for the conditions
noncomputable def linearity_of_expectation {X : Type} (a b : ℝ) (E : X → ℝ) (X : X) : Prop :=
  E(a * X + b) = a * E(X) + b

noncomputable def variance_scaling {X : Type} (a : ℝ) (D : X → ℝ) (X : X) : Prop :=
  D(a * X) = a^2 * D(X)

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

noncomputable def normal_probability (μ σ : ℝ) (X : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, X x = (1 / (σ * (Real.pi * 2)^(1/2))) * Real.exp(-(x - μ)^2 / (2 * σ^2))

-- The proof problems (questions)
theorem expectation_linear_transformation_correct {X : Type} (E : X → ℝ) (X : X) :
  linearity_of_expectation 2 3 E X := sorry

theorem variance_linear_transformation_incorrect {X : Type} (D : X → ℝ) (X : X) :
  ¬variance_scaling 2 D X := sorry

theorem binomial_distribution_probability_correct :
  binomial_probability 6 3 (1 / 2) = 5 / 16 := sorry

theorem normal_distribution_probability_correct {σ : ℝ} (hσ : σ > 0) :
  (P : Set ℝ) → (P (λ x, x < 4) = 0.9) → (P (λ x, 0 < x ∧ x < 2) = 0.4) := sorry

end expectation_linear_transformation_correct_variance_linear_transformation_incorrect_binomial_distribution_probability_correct_normal_distribution_probability_correct_l248_248845


namespace sqrt_factorial_mul_factorial_l248_248081

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l248_248081


namespace current_at_resistance_12_l248_248295

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248295


namespace smallest_prime_perimeter_l248_248559

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_scalene (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_triple_prime (a b c : ℕ) : Prop := is_prime a ∧ is_prime b ∧ is_prime c

def is_triangle (a b c : ℕ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

theorem smallest_prime_perimeter :
  ∃ a b c : ℕ, is_scalene a b c ∧ is_triple_prime a b c ∧ is_prime (a + b + c) ∧ a + b + c = 23 :=
sorry

end smallest_prime_perimeter_l248_248559


namespace current_value_l248_248396

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248396


namespace least_common_multiple_prime_numbers_l248_248793

theorem least_common_multiple_prime_numbers (x y : ℕ) (hx_prime : Prime x) (hy_prime : Prime y)
  (hxy : y < x) (h_eq : 2 * x + y = 12) : Nat.lcm x y = 10 :=
by
  sorry

end least_common_multiple_prime_numbers_l248_248793


namespace sqrt_sum_odds_l248_248833

theorem sqrt_sum_odds : 
  (Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + Real.sqrt (1+3+5+7+9) + Real.sqrt (1+3+5+7+9+11)) = 21 := 
by
  sorry

end sqrt_sum_odds_l248_248833


namespace sqrt_factorial_mul_factorial_eq_l248_248140

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l248_248140


namespace max_objective_function_l248_248031

variable {x y a z : ℝ}

noncomputable def Omega := { p : ℝ × ℝ | p.2 - 1 ≥ 0 ∧ p.1 - p.2 + 2 ≥ 0 ∧ p.1 + 4 * p.2 - 8 ≤ 0 }

theorem max_objective_function :
  (∃ a : ℝ, a > 1 ∧ set.area_ratio (Omega ∩ {p : ℝ × ℝ | p.1 < a}) (Omega ∩ {p : ℝ × ℝ | p.1 > a}) = 1 / 4) →
  ∃ a : ℝ, ∃ x y : ℝ, z = a * x + y ∧ z = 9 := 
sorry

end max_objective_function_l248_248031


namespace battery_current_l248_248376

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248376


namespace train_speed_l248_248566

theorem train_speed (len_train len_bridge time : ℝ)
  (h1 : len_train = 100)
  (h2 : len_bridge = 180)
  (h3 : time = 27.997760179185665) :
  (len_train + len_bridge) / time * 3.6 = 36 :=
by
  sorry

end train_speed_l248_248566


namespace total_amount_saved_l248_248856

def priceX : ℝ := 575
def surcharge_rateX : ℝ := 0.04
def installation_chargeX : ℝ := 82.50
def total_chargeX : ℝ := priceX + surcharge_rateX * priceX + installation_chargeX

def priceY : ℝ := 530
def surcharge_rateY : ℝ := 0.03
def installation_chargeY : ℝ := 93.00
def total_chargeY : ℝ := priceY + surcharge_rateY * priceY + installation_chargeY

def savings : ℝ := total_chargeX - total_chargeY

theorem total_amount_saved : savings = 41.60 :=
by
  sorry

end total_amount_saved_l248_248856


namespace current_value_l248_248435

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248435


namespace area_of_triangle_BDK_l248_248226

variables (a b d : ℝ)
variables (AD AK DK BP S_BDK : ℝ)

def is_isosceles_trapezoid (a b d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ d > 0 ∧ a ≠ b

def calculate_BP (a b d : ℝ) : ℝ := 
  ( √(4 * d^2 - (a - b)^2) ) / 2

def calculate_DK (a b : ℝ) : ℝ := 
  a - b

def area_triangle_BDK (DK BP : ℝ) : ℝ := 
  (1 / 4) * (a - b) * √(4 * d^2 - (a - b)^2)

theorem area_of_triangle_BDK
  (ha : a > 0) (hb : b > 0) (hd : d > 0) (hab : a ≠ b) :
  S_BDK = (1 / 4) * |a - b| * √(4 * d^2 - (a - b)^2) :=
by 
  let DK := calculate_DK a b
  let BP := calculate_BP a b d
  let S_BDK := area_triangle_BDK DK BP
  sorry

end area_of_triangle_BDK_l248_248226


namespace battery_current_l248_248346

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248346


namespace alternating_sum_series_l248_248605

theorem alternating_sum_series :
  ∑ n in (Finset.range 14400.succ).filter (λn, n ≠ 0),
  (-1)^(if (∃ k : ℕ, n = (5 * k)^2) then 1 else 0) * n
  = ∑ k in Finset.range 25, (-1)^k * (250 * k^3 - 1500 * k^2 + 4380 * k - 313) :=
by sorry

end alternating_sum_series_l248_248605


namespace max_value_a_l248_248649

theorem max_value_a (a b c d : ℝ) 
  (h1 : a ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h2 : b ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h3 : c ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h4 : d ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h5 : Real.sin a + Real.sin b + Real.sin c + Real.sin d = 1)
  (h6 : Real.cos (2 * a) + Real.cos (2 * b) + Real.cos (2 * c) + Real.cos (2 * d) ≥ 10 / 3) : 
  a ≤ Real.arcsin (1 / 2) := 
sorry

end max_value_a_l248_248649


namespace Joel_contributed_22_toys_l248_248724

/-
Define the given conditions as separate variables and statements in Lean:
1. Toys collected from friends.
2. Total toys donated.
3. Relationship between Joel's and his sister's toys.
4. Prove that Joel donated 22 toys.
-/

theorem Joel_contributed_22_toys (S : ℕ) (toys_from_friends : ℕ) (total_toys : ℕ) (sisters_toys : ℕ) 
  (h1 : toys_from_friends = 18 + 42 + 2 + 13)
  (h2 : total_toys = 108)
  (h3 : S + 2 * S = total_toys - toys_from_friends)
  (h4 : sisters_toys = S) :
  2 * S = 22 :=
  sorry

end Joel_contributed_22_toys_l248_248724


namespace battery_current_l248_248380

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248380


namespace age_of_other_girl_l248_248017

-- Definitions of the conditions
variables (age1 age2 : ℕ)

-- Conditions
def ageDifference : Prop := |age1 - age2| = 1
def ageSum : Prop := age1 + age2 = 27
def oneGirlAge : Prop := age1 = 14 ∨ age2 = 14

-- Statement to prove
theorem age_of_other_girl : ageDifference age1 age2 → ageSum age1 age2 → oneGirlAge age1 age2 → (age1 = 13 ∨ age2 = 13) :=
by
  intros hDiff hSum hOneAge
  sorry

end age_of_other_girl_l248_248017


namespace joan_video_game_spending_l248_248723

theorem joan_video_game_spending:
  let basketball_game := 5.20
  let racing_game := 4.23
  basketball_game + racing_game = 9.43 := 
by
  sorry

end joan_video_game_spending_l248_248723


namespace prove_is_necessary_not_sufficient_l248_248864

def is_necessary_not_sufficient (a : ℝ) : Prop :=
  ∀ f : ℝ → ℝ, f = (λ x, -x^3 + (1 / 2) * (a + 3) * x^2 - a * x - 1) → 
    (a ≥ 3 → (∃ b : ℝ, b > a ∧ ∃ x, x = 1 ∧ ∀ y, f y ≥ f x) ∧ ¬(a = 3 ∧ ∃ x, x = 1 ∧ ∀ y, f y > f x))
/- The term _∃ b : ℝ, b > a ∧ ∃ x, x = 1 ∧ ∀ y, f y ≥ f x_ ensures that a necessary condition is realized,
   and the term _¬(a = 3 ∧ ∃ x, x = 1 ∧ ∀ y, f y > f x)_ ensures it is not sufficient when a = 3. -/

theorem prove_is_necessary_not_sufficient (a : ℝ) : is_necessary_not_sufficient a := sorry

end prove_is_necessary_not_sufficient_l248_248864


namespace complex_conjugate_of_fraction_l248_248986

/-- Given a complex number z = 5 / (1 + 2i), prove that its complex conjugate is 1 + 2i. -/
theorem complex_conjugate_of_fraction :
  let z : ℂ := (5 : ℂ) / (1 + 2 * Complex.I) in Complex.conj z = 1 + 2 * Complex.I :=
by
  sorry

end complex_conjugate_of_fraction_l248_248986


namespace battery_current_l248_248287

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248287


namespace battery_current_l248_248319

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248319


namespace battery_current_at_given_resistance_l248_248486

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248486


namespace bisectAltitude_l248_248783

noncomputable def touchCircleParallelogram (A B C D K L M : Point) (P : Circle) : Prop :=
  (P.touches A B K) ∧ (P.touches B C L) ∧ (P.touches C D M) ∧ (Parallelogram A B C D)

theorem bisectAltitude (A B C D K L M : Point) (P : Circle) 
  (cond : touchCircleParallelogram A B C D K L M P) : 
  bisects (Line K L) (Altitude C (Line A B)) :=
sorry

end bisectAltitude_l248_248783


namespace current_at_resistance_12_l248_248308

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248308


namespace find_higher_selling_price_l248_248937

-- Define the constants and initial conditions
def cost_price : ℕ := 200
def selling_price_1 : ℕ := 340
def gain_1 : ℕ := selling_price_1 - cost_price
def new_gain : ℕ := gain_1 + gain_1 * 5 / 100

-- Define the problem statement
theorem find_higher_selling_price : 
  ∀ P : ℕ, P = cost_price + new_gain → P = 347 :=
by
  intro P
  intro h
  sorry

end find_higher_selling_price_l248_248937


namespace fraction_exp_simplifies_l248_248834

theorem fraction_exp_simplifies (x : ℝ) (h : x = 3) :
  (∏ i in finset.range(15), x^(3 * (i + 1))) / 
  (∏ i in finset.range(10), x^(5 * (i + 1))) = 3^(-295) :=
by
  -- Placeholder for the proof
  sorry

end fraction_exp_simplifies_l248_248834


namespace battery_current_l248_248331

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248331


namespace current_value_l248_248423

theorem current_value (R : ℝ) (hR : R = 12) : 48 / R = 4 :=
by
  sorry

end current_value_l248_248423


namespace denomination_of_second_note_l248_248554

theorem denomination_of_second_note
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)
  (h1 : x = y)
  (h2 : y = z)
  (h3 : x + y + z = 75)
  (h4 : 1 * x + y * x + 10 * x = 400):
  y = 5 := by
  sorry

end denomination_of_second_note_l248_248554


namespace sqrt_factorial_product_l248_248125

/-- Define the factorial of 4 -/
def factorial_four : ℕ := 4!

/-- Define the product of the factorial of 4 with itself -/
def product_of_factorials : ℕ := factorial_four * factorial_four

/-- Prove the value of the square root of product_of_factorials is 24 -/
theorem sqrt_factorial_product : Real.sqrt (product_of_factorials) = 24 := by
  have fact_4_eq_24 : factorial_four = 24 := by norm_num
  rw [product_of_factorials, fact_4_eq_24, Nat.mul_self_eq, Real.sqrt_sq]
  norm_num
  exact Nat.zero_le 24

end sqrt_factorial_product_l248_248125


namespace dart_score_ratio_l248_248976

theorem dart_score_ratio (score_third_dart bullseye_points total_score : ℕ) (h_bullseye : bullseye_points = 50) (h_missed : 0 = 0) (h_total_score : total_score = 75) :
  let x := total_score - bullseye_points in
  score_third_dart = x ∧ (x : ℚ) / bullseye_points = 1 / 2 :=
by
  sorry

end dart_score_ratio_l248_248976


namespace team_a_played_4_l248_248002

/-- Define the winning and losing ratios for Teams A and B,
    as well as the additional win/loss conditions for Team B compared to Team A. -/
variables (a : ℕ)  -- Number of games played by Team A
variables (b : ℕ)  -- Number of games played by Team B
variables (wa : ℕ) (la : ℕ) -- Number of wins and losses by Team A
variables (wb : ℕ) (lb : ℕ) -- Number of wins and losses by Team B

/-- Define the ratios and additional win/loss conditions given in the problem. -/
def conditions : Prop :=
  wa = 3 / 4 * a ∧ la = 1 / 4 * a ∧
  wa + la = a ∧  -- Team A's total games played
  wb = 2 / 3 * (a + 8) ∧ lb = 1 / 3 * (a + 8) ∧
  wb + lb = (a + 8) ∧  -- Team B's total games played
  wb = wa + 5 ∧ lb = la + 3

/-- Prove that Team A has played 4 games given the conditions. -/
theorem team_a_played_4 (h : conditions a wa la wb lb) : a = 4 :=
sorry

end team_a_played_4_l248_248002


namespace pseudocode_final_S_value_l248_248570

/-- Given the initial values of S and I, and a loop that updates these values,
    prove that the final value of S is 7 when the loop condition I < 8 no longer holds. -/
theorem pseudocode_final_S_value : 
  let S := 1
  let I := 1
  let iter (S I : ℕ) := (S + 2, I + 3)
  let (S, I) := iter (S, I)
  let (S, I) := iter (S, I)
  let (S, I) := iter (S, I)
  S = 7 :=
by
  sorry

end pseudocode_final_S_value_l248_248570


namespace minimum_value_of_m_minus_n_l248_248653

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

end minimum_value_of_m_minus_n_l248_248653


namespace sqrt_factorial_equality_l248_248161

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l248_248161


namespace current_at_resistance_12_l248_248536

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248536


namespace battery_current_l248_248335

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248335


namespace simplify_and_evaluate_expression_l248_248773

theorem simplify_and_evaluate_expression : 
  let a := 1
  let b := -1 in
  3 * a^2 * b + 2 * (a * b - (3 / 2) * a^2 * b) - (2 * a * b^2 - (3 * a * b^2 - a * b)) = 0 :=
by
  sorry

end simplify_and_evaluate_expression_l248_248773


namespace battery_current_l248_248288

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248288


namespace number_of_divisors_l248_248989

theorem number_of_divisors (a b c : ℕ) :
  (a = 10^10) ∧ (b = 15^7) ∧ (c = 18^11) →
  let max_power_2 := max 10 11,
      max_power_3 := max 7 22,
      max_power_5 := max 10 7,
      count_divisors_2 := max_power_2 + 1,
      count_divisors_3 := max_power_3 + 1,
      count_divisors_5 := max_power_5 + 1,
      total_divisors := count_divisors_2 * count_divisors_3 * count_divisors_5 - (count_divisors_2 + count_divisors_3 + count_divisors_5 - 3) in
  total_divisors = 3697 := by
  sorry

end number_of_divisors_l248_248989


namespace all_pos_of_alpha_ge_3_l248_248968

noncomputable def rec_seq : ℕ → ℝ → (ℕ → ℝ)
| 0, α => λ n, (α : ℝ) -- Initialize with α at n = 0
| (n+1), α => λ n, 2 * (rec_seq n α) n - (n : ℝ) ^ 2

theorem all_pos_of_alpha_ge_3 (α : ℝ) (h : α ≥ 3) : ∀ n : ℕ, rec_seq n α n > 0 :=
  by
    sorry -- Proof omitted.

end all_pos_of_alpha_ge_3_l248_248968


namespace solution_set_inequality_l248_248807

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) / (x + 2) < 0 ↔ -2 < x ∧ x < 3 :=
sorry

end solution_set_inequality_l248_248807


namespace measure_C_is_pi_div_six_value_c2_over_a2_is_correct_l248_248694

-- Definitions based on given conditions
def triangle := Type
constants (A B C : triangle) (a b c : ℝ)
constant (acute_C : Prop)
constant (sin_A sin_B sin_C cos_C : ℝ)
noncomputable def measure_C := Real.pi / 6
noncomputable def value_c2_over_a2 := 3 - Real.sqrt 6

-- Problem statement as Lean theorems
theorem measure_C_is_pi_div_six (h : acute_C) (eq1 : a * sin_A = b * sin_B * sin_C) (eq2 : b = Real.sqrt 2 * a) : 
  measure_C = Real.pi / 6 := 
begin
  sorry
end

theorem value_c2_over_a2_is_correct (h : acute_C) (eq1 : a * sin_A = b * sin_B * sin_C) (eq2 : b = Real.sqrt 2 * a) : 
  value_c2_over_a2 = 3 - Real.sqrt 6 := 
begin
  sorry
end

end measure_C_is_pi_div_six_value_c2_over_a2_is_correct_l248_248694


namespace trams_to_add_l248_248039

theorem trams_to_add (initial_trams : ℕ) (initial_interval new_interval : ℤ)
  (reduce_by_fraction : ℤ) (total_distance : ℤ)
  (h1 : initial_trams = 12)
  (h2 : initial_interval = total_distance / initial_trams)
  (h3 : reduce_by_fraction = 5)
  (h4 : new_interval = initial_interval - initial_interval / reduce_by_fraction) :
  initial_trams + (total_distance / new_interval - initial_trams) = 15 :=
by
  sorry

end trams_to_add_l248_248039


namespace quadratic_inequality_l248_248689

theorem quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * a * x + 4 > 0) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end quadratic_inequality_l248_248689


namespace intersection_condition_sufficient_but_not_necessary_l248_248642

theorem intersection_condition_sufficient_but_not_necessary (k : ℝ) :
  (-Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3) →
  ((∃ x : ℝ, (k^2 + 1) * x^2 + (2 * k^2 - 2) * x + k^2 = 0) ∧ 
   ¬ (∃ k, (∃ x : ℝ, (k^2 + 1) * x^2 + (2 * k^2 - 2) * x + k^2 = 0) → 
   (¬ (-Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3)))) :=
sorry

end intersection_condition_sufficient_but_not_necessary_l248_248642


namespace current_when_resistance_12_l248_248456

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248456


namespace sqrt_factorial_mul_factorial_l248_248205

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l248_248205


namespace sqrt_factorial_equality_l248_248156

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l248_248156


namespace cyclic_permutations_sum_41234_l248_248632

theorem cyclic_permutations_sum_41234 :
  let n1 := 41234
  let n2 := 34124
  let n3 := 23414
  let n4 := 12434
  3 * (n1 + n2 + n3 + n4) = 396618 :=
by
  let n1 := 41234
  let n2 := 34124
  let n3 := 23414
  let n4 := 12434
  show 3 * (n1 + n2 + n3 + n4) = 396618
  sorry

end cyclic_permutations_sum_41234_l248_248632


namespace find_higher_selling_price_l248_248939

def cost_price := 200
def selling_price_low := 340
def gain_low := selling_price_low - cost_price
def gain_high := gain_low + (5 / 100) * gain_low
def higher_selling_price := cost_price + gain_high

theorem find_higher_selling_price : higher_selling_price = 347 := 
by 
  sorry

end find_higher_selling_price_l248_248939


namespace battery_current_l248_248284

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l248_248284


namespace question1_solution_question2_solution_l248_248662

noncomputable def f (x m : ℝ) : ℝ := x^2 - m * x + m - 1

theorem question1_solution (x : ℝ) :
  ∀ x, f x 3 ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2 :=
sorry

theorem question2_solution (m : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f x m ≥ -1) ↔ m ≤ 4 :=
sorry

end question1_solution_question2_solution_l248_248662


namespace log_comparison_l248_248866

variable {R : Type*} [LinearOrderedField R]

theorem log_comparison (a b c : R) (h₁ : a = log 6 / log 3) (h₂ : b = log 10 / log 5) (h₃ : c = log 14 / log 7) : a > b ∧ b > c := 
by
  sorry

end log_comparison_l248_248866


namespace distance_between_points_l248_248618

-- Define the given points
def point1 : ℝ × ℝ := (2, 3)
def point2 : ℝ × ℝ := (7, 11)

-- Define the horizontal and vertical distances
def horizontal_separation := point2.1 - point1.1
def vertical_separation := point2.2 - point1.2

-- Define the theorem to be proved
theorem distance_between_points : real.sqrt (horizontal_separation ^ 2 + vertical_separation ^ 2) = real.sqrt 89 :=
by
  sorry

end distance_between_points_l248_248618


namespace asymptotes_of_hyperbola_l248_248888

theorem asymptotes_of_hyperbola (h : ∀ {a : ℝ}, (a > 0) → 
  ∃ p : ℝ × ℝ, p.1 ^ 2 + p.2 ^ 2 = 3 ∧ 
  ((2 * p.1 + p.2 = 3 ∧ 2 * p.1 + p.2 = 3) ∧
  ∃ (focus : ℝ), focus = Real.sqrt (2 + a^2) ∧ (focus = 3 / 2)) → a = 1 / 2 ∧ 
  (∀ x y, 4 * x ^ 2 - y ^ 2 / 2 = 1) → 
  y = 2 * Real.sqrt 2 * x ∨ y = -2 * Real.sqrt 2 * x) 
: ∀ {a : ℝ} (a_gt_0 : a > 0),
  ∃ eq_asymptotes : String, eq_asymptotes = "y = ±2√2x" :=
begin
  sorry
end

end asymptotes_of_hyperbola_l248_248888


namespace sum_ratio_l248_248739

-- Given: Sum of arithmetic sequence \( S_n \)
def S_n (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

-- Given: \( a₁ \), \( a₂ \), \( a₄ \) form a geometric sequence
def is_geometric (a₁ a₂ a₄ : ℝ) : Prop :=
  a₂^2 = a₁ * a₄

-- Prove: \( \frac{S₄}{S₂} = \frac{10}{3} \) given the conditions
theorem sum_ratio (a₁ d : ℝ) (h₁ : a₁ ≠ 0) 
  (h₂ : d = a₁) 
  (h₃ : is_geometric a₁ (a₁ + d) (a₁ + 3 * d)) : S_n a₁ d 4 / S_n a₁ d 2 = 10 / 3 := by
  sorry

end sum_ratio_l248_248739


namespace parallelepiped_length_l248_248910

theorem parallelepiped_length (n : ℕ) :
  (n - 2) * (n - 4) * (n - 6) = 2 * n * (n - 2) * (n - 4) / 3 →
  n = 18 :=
by
  intros h
  sorry

end parallelepiped_length_l248_248910


namespace current_value_l248_248399

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248399


namespace sqrt_factorial_equality_l248_248162

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l248_248162


namespace current_at_resistance_12_l248_248540

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248540


namespace num_digits_3_pow_10_mul_5_pow_10_l248_248602

def num_digits (n : ℕ) : ℕ := (Int.log10 n).natAbs + 1

theorem num_digits_3_pow_10_mul_5_pow_10 :
  num_digits (3^10 * 5^10) = 12 :=
by
  sorry

end num_digits_3_pow_10_mul_5_pow_10_l248_248602


namespace choose_president_and_secretary_same_gender_l248_248759

theorem choose_president_and_secretary_same_gender :
  let total_members := 25
  let boys := 15
  let girls := 10
  ∃ (total_ways : ℕ), total_ways = (boys * (boys - 1)) + (girls * (girls - 1)) := sorry

end choose_president_and_secretary_same_gender_l248_248759


namespace largest_4_digit_divisible_by_l248_248831

theorem largest_4_digit_divisible_by (d : ℕ) (h : d = 8) : ∃ k : ℕ, 9984 = k * d :=
by
  use 1248
  rw h
  exact Nat.mul_comm 1248 8

end largest_4_digit_divisible_by_l248_248831


namespace watch_correction_l248_248905

def watch_loss_per_day : ℚ := 13 / 4

def hours_from_march_15_noon_to_march_22_9am : ℚ := 7 * 24 + 21

def per_hour_loss : ℚ := watch_loss_per_day / 24

def total_loss_in_minutes : ℚ := hours_from_march_15_noon_to_march_22_9am * per_hour_loss

theorem watch_correction :
  total_loss_in_minutes = 2457 / 96 :=
by
  sorry

end watch_correction_l248_248905


namespace sqrt_factorial_product_l248_248107

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248107


namespace police_emergency_number_has_prime_gt_7_l248_248893

def is_police_emergency_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 1000 * k + 133

theorem police_emergency_number_has_prime_gt_7 {n : ℕ} (h : is_police_emergency_number n) : 
  ∃ p : ℕ, prime p ∧ p > 7 ∧ p ∣ n :=
by
  sorry

end police_emergency_number_has_prime_gt_7_l248_248893


namespace find_c_work_rate_l248_248219

variables (A B C : ℚ)   -- Using rational numbers for the work rates

theorem find_c_work_rate (h1 : A + B = 1/3) (h2 : B + C = 1/4) (h3 : C + A = 1/6) : 
  C = 1/24 := 
sorry 

end find_c_work_rate_l248_248219


namespace trailing_zeros_of_9_pow_999_plus_1_l248_248676

theorem trailing_zeros_of_9_pow_999_plus_1 :
  ∃ n : ℕ, n = 999 ∧ (9^n + 1) % 10 = 0 ∧ (9^n + 1) % 100 ≠ 0 :=
by
  sorry

end trailing_zeros_of_9_pow_999_plus_1_l248_248676


namespace find_y_angle_l248_248707

-- Define the conditions given in the problem.
def is_isosceles_right_triangle (Q R S : Type) [inner_product_space ℝ (Q × R × S)] 
  (QR SR : ℝ) (angle_QRS angle_RQS angle_RSQ : ℝ) : Prop :=
QR = SR ∧ angle_QRS = 90 ∧ angle_RQS = 45 ∧ angle_RSQ = 45

-- Define the problem statement with the conclusion to be proven.
theorem find_y_angle {Q R S P T U V : Type} [inner_product_space ℝ (Q × R × S)] 
  (QR SR : ℝ) (angle_QRS : ℝ) (h1 : is_isosceles_right_triangle Q R S QR SR angle_QRS 45 45) 
  (PT_intersects_U : intersects P T U S Q) (PT_intersects_V : intersects P T V S R) 
  (angle_PUQ angle_RVT : ℝ)
  (h2 : angle_PUQ = angle_RVT) : angle_PUQ = 67.5 :=
sorry

end find_y_angle_l248_248707


namespace find_current_l248_248453

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248453


namespace current_value_l248_248260

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248260


namespace parallelepiped_length_l248_248919

theorem parallelepiped_length :
  ∃ n : ℕ, (n ≥ 7) ∧ (n * (n - 2) * (n - 4) = 3 * ((n - 2) * (n - 4) * (n - 6))) ∧ n = 18 :=
by
  sorry

end parallelepiped_length_l248_248919


namespace find_collinear_vector_l248_248664

def vec3 := (ℝ × ℝ × ℝ)

def collinear (a x : vec3) : Prop :=
  ∃ m : ℝ, x = (2 * m, -1 * m, 2 * m)

def dot_product (a x : vec3) : ℝ :=
  a.1 * x.1 + a.2 * x.2 + a.3 * x.3

theorem find_collinear_vector :
  let a := (2, -1, 2)
  let x := (-4, 2, -4)
  collinear a x ∧ dot_product a x = -18 :=
by
  -- Definitions to establish the conditions
  let a := (2, -1, 2)
  let x := (-4, 2, -4)
  -- Prove that x is collinear with a and dot_product a x = -18
  sorry

end find_collinear_vector_l248_248664


namespace no_adjacent_same_roll_probability_l248_248631

noncomputable def probability_no_adjacent_same_roll : ℚ :=
  (1331 / 1728)

theorem no_adjacent_same_roll_probability :
  (probability_no_adjacent_same_roll = (1331 / 1728)) :=
by
  sorry

end no_adjacent_same_roll_probability_l248_248631


namespace current_value_l248_248384

theorem current_value (V R : ℝ) (hV : V = 48) (hR : R = 12) : (I : ℝ) (hI : I = V / R) → I = 4 :=
by
  intros
  rw [hI, hV, hR]
  norm_num
  sorry

end current_value_l248_248384


namespace current_when_resistance_is_12_l248_248404

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248404


namespace polynomial_divisibility_l248_248597

-- Define the set of numbers {2^0, 2^1, ..., 2^n}
def number_set (n : ℕ) := finset.image (λ (i : fin n.succ), 2^i.val) (finset.univ)

-- Define the set of all polynomials with coefficients as permutations of {2^0, 2^1, ..., 2^n}
def P (n : ℕ) := { p : polynomial ℕ // ∀ (coeff : ℕ), coeff ∈ p.coeff.support -> coeff ∈ number_set n }

-- Define what it means for a polynomial to be divisible by a number d at k
def divisible_at (k d : ℕ) (p : polynomial ℕ) := d ∣ p.eval k

-- Since we require pairs (k, d) such that divisible_at holds for some n and all p ∈ P(n)
def exists_divisible_pair := 
  ∃ (a b : ℕ), ∀ (n : ℕ), ∀ (p : P n),
    divisible_at (b * (2 * a + 1) + 1) (2 * a + 1) p.val

-- The theorem to prove the equivalence
theorem polynomial_divisibility:
  exists_divisible_pair :=
sorry

end polynomial_divisibility_l248_248597


namespace sqrt_factorial_equality_l248_248158

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l248_248158


namespace battery_current_l248_248341

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248341


namespace battery_current_l248_248367

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248367


namespace battery_current_at_given_resistance_l248_248482

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248482


namespace prince_spending_l248_248945

theorem prince_spending (CDs_total : ℕ) (CDs_10_percent : ℕ) (CDs_10_cost : ℕ) (CDs_5_cost : ℕ) 
  (Prince_10_fraction : ℚ) (Prince_5_fraction : ℚ) 
  (total_10_CDs : ℕ) (total_5_CDs : ℕ) (Prince_10_CDs : ℕ) (Prince_5_CDs : ℕ) (total_cost : ℕ) :
  CDs_total = 200 →
  CDs_10_percent = 40 →
  CDs_10_cost = 10 →
  CDs_5_cost = 5 →
  Prince_10_fraction = 1/2 →
  Prince_5_fraction = 1 →
  total_10_CDs = CDs_total * CDs_10_percent / 100 →
  total_5_CDs = CDs_total - total_10_CDs →
  Prince_10_CDs = total_10_CDs * Prince_10_fraction →
  Prince_5_CDs = total_5_CDs * Prince_5_fraction →
  total_cost = (Prince_10_CDs * CDs_10_cost) + (Prince_5_CDs * CDs_5_cost) →
  total_cost = 1000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end prince_spending_l248_248945


namespace find_current_l248_248441

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248441


namespace land_sufficient_for_house_l248_248941

theorem land_sufficient_for_house :
  let land_area := 120 * 100
  let house_area := 25 * 35
  let flower_pot_radius := 5 / 2
  let flower_pot_area := Real.pi * (flower_pot_radius^2)
  let total_flower_pots_area := 9 * flower_pot_area
  land_area - house_area - total_flower_pots_area > 0 := by
  sorry

end land_sufficient_for_house_l248_248941


namespace sqrt_factorial_multiplication_l248_248180

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l248_248180


namespace angle_at_3_20_l248_248575

def hour_hand_position_at_three_twenty (hour_hand_per_hour : ℝ) : ℝ :=
  let initial_position := 3 * hour_hand_per_hour in
  let additional_position := (hour_hand_per_hour / 60) * 20 in
  initial_position + additional_position

def minute_hand_position_at_three_twenty (minute_hand_per_minute : ℝ) : ℝ :=
  20 * minute_hand_per_minute

def angle_between_hands (hour_position minute_position : ℝ) : ℝ :=
  |minute_position - hour_position|

theorem angle_at_3_20 :
  let hour_hand_per_hour := 30
  let minute_hand_per_minute := 6
  let hour_hand := hour_hand_position_at_three_twenty hour_hand_per_hour
  let minute_hand := minute_hand_position_at_three_twenty minute_hand_per_minute
  angle_between_hands hour_hand minute_hand = 20 :=
by
  sorry

end angle_at_3_20_l248_248575


namespace find_current_l248_248455

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248455


namespace find_c_l248_248018

theorem find_c (x c : ℚ) (h1 : 3 * x + 5 = 1) (h2 : c * x + 15 = 3) : c = 9 :=
by sorry

end find_c_l248_248018


namespace coconut_trees_per_sqm_l248_248768

def farm_area : ℕ := 20
def harvests : ℕ := 2
def total_earnings : ℝ := 240
def coconut_price : ℝ := 0.50
def coconuts_per_tree : ℕ := 6

theorem coconut_trees_per_sqm : 
  let total_coconuts := total_earnings / coconut_price / harvests
  let total_trees := total_coconuts / coconuts_per_tree 
  let trees_per_sqm := total_trees / farm_area 
  trees_per_sqm = 2 :=
by
  sorry

end coconut_trees_per_sqm_l248_248768


namespace current_when_resistance_12_l248_248465

-- Definitions based on the conditions
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- Theorem proof statement
theorem current_when_resistance_12 : current resistance = 4 :=
  by sorry

end current_when_resistance_12_l248_248465


namespace sqrt_factorial_equality_l248_248153

theorem sqrt_factorial_equality : Real.sqrt (4! * 4!) = 24 := 
by
  sorry

end sqrt_factorial_equality_l248_248153


namespace rectangle_square_area_ratio_l248_248795

theorem rectangle_square_area_ratio (s : ℝ) (hs : s > 0) :
  let area_square := s^2 in
  let area_rectangle := (1.2 * s) * (0.8 * s) in
  area_rectangle / area_square = 24 / 25 :=
by
  sorry

end rectangle_square_area_ratio_l248_248795


namespace correct_polynomial_multiplication_l248_248062

theorem correct_polynomial_multiplication (a b : ℤ) (x : ℝ)
  (h1 : 2 * b - 3 * a = 11)
  (h2 : 2 * b + a = -9) :
  (2 * x + a) * (3 * x + b) = 6 * x^2 - 19 * x + 10 := by
  sorry

end correct_polynomial_multiplication_l248_248062


namespace battery_current_l248_248372

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248372


namespace max_at_zero_l248_248790

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem max_at_zero : ∀ x : ℝ, f x ≤ f 0 :=
by
  sorry

end max_at_zero_l248_248790


namespace find_current_l248_248443

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248443


namespace sqrt_factorial_product_eq_24_l248_248194

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l248_248194


namespace remainder_31_31_plus_31_mod_32_l248_248211

theorem remainder_31_31_plus_31_mod_32 : (31 ^ 31 + 31) % 32 = 30 := 
by sorry

end remainder_31_31_plus_31_mod_32_l248_248211


namespace sqrt_factorial_product_l248_248099

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l248_248099


namespace battery_current_when_resistance_12_l248_248513

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248513


namespace sqrt_factorial_product_eq_24_l248_248193

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l248_248193


namespace current_at_resistance_12_l248_248310

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248310


namespace current_value_l248_248274

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248274


namespace current_value_l248_248497

/-- Define the given voltage V as 48 --/
def V : ℝ := 48

/-- Define the current I as a function of resistance R --/
def I (R : ℝ) : ℝ := V / R

/-- Define a particular resistance R of 12 ohms --/
def R : ℝ := 12

/-- Formulating the proof problem as a Lean theorem statement --/
theorem current_value : I R = 4 := by
  sorry

end current_value_l248_248497


namespace battery_current_l248_248334

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248334


namespace secretaries_total_hours_l248_248005

theorem secretaries_total_hours :
  ∃ (x : ℕ), 
  let s1 := 2 * x in 
  let s2 := 3 * x in 
  let s3 := 5 * x in 
  s3 = 40 ∧ s1 + s2 + s3 = 80 :=
by
  use 8
  split
  . exact rfl
  . sorry

end secretaries_total_hours_l248_248005


namespace sqrt_factorial_product_eq_24_l248_248192

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l248_248192


namespace current_at_resistance_12_l248_248531

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248531


namespace least_xy_value_l248_248640

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : (1/x : ℚ) + 1/(2*y) = 1/8) :
  xy ≥ 128 :=
sorry

end least_xy_value_l248_248640


namespace scientific_notation_28400_is_correct_l248_248944

theorem scientific_notation_28400_is_correct : (28400 : ℝ) = 2.84 * 10^4 := 
by 
  sorry

end scientific_notation_28400_is_correct_l248_248944


namespace sqrt_factorial_product_l248_248172

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248172


namespace sqrt_factorial_product_l248_248111

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248111


namespace sqrt_factorial_mul_factorial_l248_248199

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l248_248199


namespace sqrt_factorial_multiplication_l248_248176

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l248_248176


namespace length_of_AB_l248_248709

variables (A B C D : Type) [Real.Number]
variables (AB AC BC BD CD: ℝ)

axioms
  (isosceles_ABC : AB = AC)
  (isosceles_BCD : BC = CD)
  (perimeter_CBD : BC + CD + BD = 22)
  (perimeter_ABC : AB + AC + BC = 24)
  (length_BD : BD = 8)

theorem length_of_AB : AB = 8.5 :=
by sorry

end length_of_AB_l248_248709


namespace find_R_when_S_7_l248_248738

-- Define the variables and equations in Lean
variables (R S g : ℕ)

-- The theorem statement based on the given conditions and desired conclusion
theorem find_R_when_S_7 (h1 : R = 2 * g * S + 3) (h2: R = 23) (h3 : S = 5) : (∃ g : ℕ, R = 2 * g * 7 + 3) :=
by {
  -- This part enforces the proof will be handled later
  sorry
}

end find_R_when_S_7_l248_248738


namespace battery_current_l248_248375

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248375


namespace max_value_l248_248713

theorem max_value (a b c d : ℕ) (h : {a, b, c, d} = {0, 1, 4, 5}) : c * a ^ b - d ≤ 625 := by
  sorry

end max_value_l248_248713


namespace delta_y_proof_l248_248658

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 1

theorem delta_y_proof
  (x : ℝ) (Δx : ℝ) (h₁ : x = 1) (h₂ : Δx = 0.1) :
  f(x + Δx) - f(x) = 0.63 :=
by 
  sorry

end delta_y_proof_l248_248658


namespace flight_cost_l248_248014

theorem flight_cost (ground_school_cost flight_portion_addition total_cost flight_portion_cost: ℕ) 
  (h₁ : ground_school_cost = 325)
  (h₂ : flight_portion_addition = 625)
  (h₃ : flight_portion_cost = ground_school_cost + flight_portion_addition):
  flight_portion_cost = 950 :=
by
  -- placeholder for proofs
  sorry

end flight_cost_l248_248014


namespace sqrt_factorial_product_l248_248167

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_product :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248167


namespace compare_abc_l248_248634

variable (α : ℝ) (hα : α ∈ Ioo (Real.pi / 4) (Real.pi / 2))

def a : ℝ := Real.log (Real.sin α) / Real.log 3
def b : ℝ := 2 ^ (Real.sin α)
def c : ℝ := 2 ^ (Real.cos α)

theorem compare_abc : b > c ∧ c > a := by
  have hsin : 0 < Real.sin α := sorry
  have hcos : 0 < Real.cos α := sorry
  have hb : b = 2 ^ Real.sin α := by rfl
  have hc : c = 2 ^ Real.cos α := by rfl
  have ha : a = Real.log (Real.sin α) / Real.log 3 := by rfl
  sorry

end compare_abc_l248_248634


namespace find_current_when_resistance_is_12_l248_248352

-- Defining the conditions as given in the problem
def voltage : ℝ := 48
def resistance : ℝ := 12
def current (R : ℝ) : ℝ := voltage / R

-- The theorem we need to prove
theorem find_current_when_resistance_is_12 :
  current resistance = 4 :=
by
  -- skipping the formal proof steps
  sorry

end find_current_when_resistance_is_12_l248_248352


namespace battery_current_when_resistance_12_l248_248525

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248525


namespace sqrt_factorial_product_l248_248075

theorem sqrt_factorial_product:
  sqrt ((fact 4) * (fact 4)) = 24 :=
by sorry

end sqrt_factorial_product_l248_248075


namespace parabola_circle_intersection_l248_248024

theorem parabola_circle_intersection :
  (∃ x y : ℝ, y = (x - 2)^2 ∧ x + 1 = (y + 2)^2) →
  (∃ r : ℝ, ∀ x y : ℝ, (y = (x - 2)^2 ∧ x + 1 = (y + 2)^2) →
    (x - 5/2)^2 + (y + 3/2)^2 = r^2 ∧ r^2 = 3/2) :=
by
  intros
  sorry

end parabola_circle_intersection_l248_248024


namespace trams_required_l248_248038

theorem trams_required (initial_trams : ℕ) (initial_interval : ℚ) (reduction_fraction : ℚ) :
  initial_trams = 12 ∧ initial_interval = 5 ∧ reduction_fraction = 1/5 →
  (initial_trams + initial_trams * reduction_fraction - initial_trams) = 3 :=
by
  sorry

end trams_required_l248_248038


namespace extremum_point_at_1_range_of_t_l248_248659

-- Problem (I)
def f (x : ℝ) (m : ℝ) := (1/2) * m * x^2 - 2 * x + Real.log (x + 1)

theorem extremum_point_at_1 (m : ℝ) : 
  m = 3 / 2 →
  ∃ x : ℝ, x = 1 ∧ ∀ ε > 0, f x m ≤ f 1 m + ε :=
by
  sorry

-- Problem (II)
def g (x : ℝ) (m : ℝ) := f x m - Real.log (x + 1) + x^3

theorem range_of_t (m : ℝ) (t : ℝ) :
  -4 ≤ m ∧ m < -1 →
  (∀ x ∈ set.Icc 1 t, g x m ≤ g 1 m) →
  1 < t ∧ t ≤ (1 + Real.sqrt 13) / 2 :=
by
  sorry

end extremum_point_at_1_range_of_t_l248_248659


namespace train_cross_bridge_time_l248_248857

def length_of_train := 130 -- in meters
def length_of_bridge := 150 -- in meters
def speed_of_train_kmh := 65 -- in km/hr
def speed_conversion_factor := 1000 / 3600 -- from km/hr to m/s

noncomputable def total_distance := length_of_train + length_of_bridge -- in meters
noncomputable def speed_of_train_ms := speed_of_train_kmh * speed_conversion_factor -- converted speed in m/s

noncomputable def time_to_cross_bridge := total_distance / speed_of_train_ms -- time in seconds

theorem train_cross_bridge_time : time_to_cross_bridge ≈ 15.51 := by
  sorry 

end train_cross_bridge_time_l248_248857


namespace sqrt_factorial_product_eq_24_l248_248190

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l248_248190


namespace vertex_of_parabola_is_correct_l248_248013

theorem vertex_of_parabola_is_correct :
  ∀ x y : ℝ, y = -5 * (x + 2) ^ 2 - 6 → (x = -2 ∧ y = -6) :=
by
  sorry

end vertex_of_parabola_is_correct_l248_248013


namespace horror_movie_more_than_triple_romance_l248_248799

-- Definitions and Conditions
def tickets_sold_romance : ℕ := 25
def tickets_sold_horror : ℕ := 93
def triple_tickets_romance := 3 * tickets_sold_romance

-- Theorem Statement
theorem horror_movie_more_than_triple_romance :
  (tickets_sold_horror - triple_tickets_romance) = 18 :=
by
  sorry

end horror_movie_more_than_triple_romance_l248_248799


namespace battery_current_l248_248382

theorem battery_current (R : ℝ) (h₁ : R = 12) (h₂ : I = 48 / R) : I = 4 := by
  sorry

end battery_current_l248_248382


namespace range_of_y_l248_248231

theorem range_of_y (x : ℝ) : 
  - (Real.sqrt 3) / 3 ≤ (Real.sin x) / (2 - Real.cos x) ∧ (Real.sin x) / (2 - Real.cos x) ≤ (Real.sqrt 3) / 3 :=
sorry

end range_of_y_l248_248231


namespace sqrt_factorial_multiplication_l248_248183

theorem sqrt_factorial_multiplication : (Real.sqrt ((4! : ℝ) * (4! : ℝ)) = 24) := 
by sorry

end sqrt_factorial_multiplication_l248_248183


namespace sequence_general_formula_l248_248716

theorem sequence_general_formula {a : ℕ → ℕ} 
  (h₁ : a 1 = 2) 
  (h₂ : ∀ n : ℕ, a (n + 1) = 2 * a n + 3 * 5 ^ n) 
  : ∀ n : ℕ, a n = 5 ^ n - 3 * 2 ^ (n - 1) :=
sorry

end sequence_general_formula_l248_248716


namespace sqrt_factorial_eq_l248_248137

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l248_248137


namespace regular_star_n_points_l248_248760

theorem regular_star_n_points (θ : ℝ) (n : ℕ)
  (h1 : ∀ i j : ℕ, i < n → j < n → angle_at_A_i i = angle_at_A_j j)
  (h2 : ∀ i j : ℕ, i < n → j < n → angle_at_B_i i = angle_at_B_j j)
  (h3 : ∀ i : ℕ, i < n → angle_at_A_i i = θ - 15)
  (h4 : ∀ i : ℕ, i < n → angle_at_B_i i = θ)
  (h5 : ∀ i : ℕ, i < n → S (angle_at_B_i i) - S (angle_at_A_i i) = 180) :
  n = 12 :=
by
  sorry

end regular_star_n_points_l248_248760


namespace area_of_T_l248_248744

open Complex

-- Define the complex number ω
def ω : ℂ := -1/2 + Complex.i * sqrt 3 / 2

-- Define the set T
def T : Set ℂ := { z | ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ z = a + b * ω + c * ω^2 }

-- The proof problem: show that the area of T is 6 * sqrt 3
theorem area_of_T : sorry :=
  sorry

end area_of_T_l248_248744


namespace current_value_l248_248268

theorem current_value (R : ℝ) (h : R = 12) : (48 / R) = 4 :=
by
  rw [h]
  norm_num
  sorry

end current_value_l248_248268


namespace which_is_right_triangle_l248_248843

-- Definitions for each group of numbers
def sides_A := (1, 2, 3)
def sides_B := (3, 4, 5)
def sides_C := (4, 5, 6)
def sides_D := (7, 8, 9)

-- Definition of a condition for right triangle using the converse of the Pythagorean theorem
def is_right_triangle (a b c: ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem which_is_right_triangle :
    ¬is_right_triangle 1 2 3 ∧
    ¬is_right_triangle 4 5 6 ∧
    ¬is_right_triangle 7 8 9 ∧
    is_right_triangle 3 4 5 :=
by
  sorry

end which_is_right_triangle_l248_248843


namespace sqrt_factorial_product_l248_248109

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248109


namespace convex_polyhedron_equal_faces_constant_sum_of_distances_tetrahedron_ratio_sum_one_l248_248854

-- Define the problem for part (a)
theorem convex_polyhedron_equal_faces_constant_sum_of_distances
    (P : Polyhedron) 
    (convex : P.convex) 
    (equal_face_areas : ∀ {f1 f2 : Face}, f1 ∈ P.faces → f2 ∈ P.faces → area f1 = area f2) 
    (interior_point : Point) :
    ∑ face_distance_intersect planes interior_point = (3 * volume P) / (area (arbitrary_face P)) := 
sorry

-- Define the problem for part (b)
theorem tetrahedron_ratio_sum_one 
    (T : Tetrahedron) 
    (interior_point : Point)
    (heights : Vector ℝ 4)
    (distances : Vector ℝ 4) 
    (h_eq_h : heights = λ i, height (face_of_index T i))
    (d_eq_d : distances = λ i, distance_to_face interior_point (face_of_index T i))
    :
    ∑ i in fin 4, distances[i] / heights[i] = 1 := 
sorry

end convex_polyhedron_equal_faces_constant_sum_of_distances_tetrahedron_ratio_sum_one_l248_248854


namespace rest_area_milepost_is_110_l248_248789

-- Define the mileposts for the fifth and fifteenth exits.
def milepost_fifth_exit : ℕ := 50
def milepost_fifteenth_exit : ℕ := 230

-- Define the fraction representing one-third.
def one_third : ℝ := 1 / 3

-- Define the function to calculate the rest area milepost.
def rest_area_milepost : ℕ :=
  let distance_between_exits := milepost_fifteenth_exit - milepost_fifth_exit in
  milepost_fifth_exit + (one_third * distance_between_exits).toNat

theorem rest_area_milepost_is_110 : rest_area_milepost = 110 := by
  sorry

end rest_area_milepost_is_110_l248_248789


namespace sqrt_factorial_eq_l248_248133

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_factorial_eq :
  sqrt (factorial 4 * factorial 4) = 24 :=
by
  have h : factorial 4 = 24 := by
    unfold factorial
    simpa using [factorial, factorial, factorial]
  rw [h, h]
  sorry

end sqrt_factorial_eq_l248_248133


namespace current_at_resistance_12_l248_248303

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (V : ℝ) (h1 : V = 48) (h2 : I = V / R) (h3 : R = 12) : I = 4 := 
by
  subst h3
  subst h1
  rw [h2]
  simp
  norm_num
  rfl
  sorry

end current_at_resistance_12_l248_248303


namespace battery_current_l248_248336

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end battery_current_l248_248336


namespace intersection_bisectors_l248_248735

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def bisector (P Q R : Point) : Line := sorry
noncomputable def perp_bisector (P Q : Point) : Line := sorry
noncomputable def intersection (l₁ l₂ : Line) : Point := sorry
noncomputable def distance (P Q : Point) : Real := sorry
noncomputable def on_circumcircle (P A B C : Point) : Prop := sorry

theorem intersection_bisectors {A B C : Point} :
  let Γ := circumcenter A B C,
      I := incenter A B C,
      D := bisector A B C,
      Delta := perp_bisector B C,
      K := intersection D Γ in
  on_circumcircle K A B C ∧ distance K B = distance K C ∧ distance K I = distance K B :=
by
  sorry

end intersection_bisectors_l248_248735


namespace find_current_l248_248447

-- Define the conditions and the question
def V : ℝ := 48  -- Voltage is 48V
def R : ℝ := 12  -- Resistance is 12Ω

-- Define the function I = 48 / R
def I (R : ℝ) : ℝ := V / R

-- Prove that I = 4 when R = 12
theorem find_current : I R = 4 := by
  -- skipping the proof
  sorry

end find_current_l248_248447


namespace sqrt_factorial_mul_factorial_l248_248089

theorem sqrt_factorial_mul_factorial (n : ℕ) : 
  n = 4 → sqrt ((nat.factorial n) * (nat.factorial n)) = nat.factorial n :=
by
  intro h
  rw [h, nat.factorial, mul_self_sqrt (nat.factorial_nonneg 4)]

-- Note: While the final "mul_self_sqrt (nat.factorial_nonneg 4)" line is a sketch of the idea,
-- the proof is not complete as requested.

end sqrt_factorial_mul_factorial_l248_248089


namespace sum_of_solutions_y_plus_inverse_eq_twelve_l248_248991

theorem sum_of_solutions_y_plus_inverse_eq_twelve :
  (∑ y in {y : ℝ | y + 36 / y = 12}, y) = 6 :=
by
  sorry

end sum_of_solutions_y_plus_inverse_eq_twelve_l248_248991


namespace trams_required_l248_248036

theorem trams_required (initial_trams : ℕ) (initial_interval : ℚ) (reduction_fraction : ℚ) :
  initial_trams = 12 ∧ initial_interval = 5 ∧ reduction_fraction = 1/5 →
  (initial_trams + initial_trams * reduction_fraction - initial_trams) = 3 :=
by
  sorry

end trams_required_l248_248036


namespace battery_current_when_resistance_12_l248_248511

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248511


namespace battery_current_at_given_resistance_l248_248483

theorem battery_current_at_given_resistance :
  ∀ (R : ℝ), R = 12 → (I = 48 / R) → I = 4 := by
  intro R hR hf
  rw hR at hf
  sorry

end battery_current_at_given_resistance_l248_248483


namespace current_when_resistance_is_12_l248_248419

/--
  Suppose we have a battery with voltage 48V. The current \(I\) (in amperes) is defined as follows:
  If the resistance \(R\) (in ohms) is given by the function \(I = \frac{48}{R}\),
  then when the resistance \(R\) is 12 ohms, prove that the current \(I\) is 4 amperes.
-/
theorem current_when_resistance_is_12 (R : ℝ) (I : ℝ) (h : I = 48 / R) (hR : R = 12) : I = 4 := 
by 
  have : I = 48 / 12 := by rw [h, hR]
  exact this

end current_when_resistance_is_12_l248_248419


namespace battery_current_when_resistance_12_l248_248512

theorem battery_current_when_resistance_12 :
  ∀ (V : ℝ) (I R : ℝ), V = 48 ∧ I = 48 / R ∧ R = 12 → I = 4 :=
by
  intros V I R h
  cases h with hV hIR
  cases hIR with hI hR
  rw [hV, hR] at hI
  have : I = 48 / 12, from hI
  have : I = 4, by simp [this]
  exact this

end battery_current_when_resistance_12_l248_248512


namespace sqrt_factorial_product_eq_24_l248_248197

theorem sqrt_factorial_product_eq_24 : (sqrt (fact 4 * fact 4) = 24) :=
by sorry

end sqrt_factorial_product_eq_24_l248_248197


namespace sqrt_factorial_mul_factorial_eq_l248_248139

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l248_248139


namespace sum_of_inradii_l248_248693

-- Definitions of the given conditions
def Triangle (α : Type) := (α × α × α) -- A triangle defined by three points.
def Point := ℝ × ℝ

-- Declare the points A, B, C as real tuples (coordinates are not strictly necessary in this problem)
variables (A B C E : Point)
variables (AB AC BC AE EC BE : ℝ)

-- In triangle ABC, AB = 7, AC = 9, BC = 12
axiom h_AB : dist A B = 7
axiom h_AC : dist A C = 9
axiom h_BC : dist B C = 12

-- E is the midpoint of AC, hence AE = EC = 4.5
axiom h_E : E = (A.1 + C.1) / 2, (A.2 + C.2) / 2
axiom h_AE : AE = 9 / 2
axiom h_EC : EC = 9 / 2

-- Sum of the radii of the inscribed circles in triangles ABE and BCE
def inradius (a b c : ℝ) : ℝ :=
  sqrt ((a + b - c) * (b + c - a) * (c + a - b) / (4 * (a + b + c)))

def r_abe := inradius (dist A B) (dist A E) (dist B E)
def r_bce := inradius (dist B C) (dist B E) (dist C E)

-- The proof we need
theorem sum_of_inradii :
  r_abe + r_bce = 15 * sqrt 30 / 9 :=
  sorry

end sum_of_inradii_l248_248693


namespace time_to_cover_escalator_l248_248855

theorem time_to_cover_escalator (escalator_speed person_speed length : ℕ) (h1 : escalator_speed = 11) (h2 : person_speed = 3) (h3 : length = 126) : 
  length / (escalator_speed + person_speed) = 9 := by
  sorry

end time_to_cover_escalator_l248_248855


namespace part_a_part_b_l248_248846

-- Define the tower of exponents function for convenience
def tower (base : ℕ) (height : ℕ) : ℕ :=
  if height = 0 then 1 else base^(tower base (height - 1))

-- Part a: Tower of 3s with height 99 is greater than Tower of 2s with height 100
theorem part_a : tower 3 99 > tower 2 100 := sorry

-- Part b: Tower of 3s with height 100 is greater than Tower of 3s with height 99
theorem part_b : tower 3 100 > tower 3 99 := sorry

end part_a_part_b_l248_248846


namespace overall_loss_rs_3_46_l248_248729

variables (rsDollars rsPounds rsEuros : ℝ) (grinderCost mobileCost refrigeratorCost televisionCost : ℝ)
variables (grinderLoss mobileProfit refrigeratorProfit televisionLoss : ℝ)
variables (indiaTax usTax ukTax germanyTax : ℝ)

def johnsOverallProfitLoss (grinderCost rsDollars mobileCost rsPounds refrigeratorCost rsEuros televisionCost : ℝ)
(grinderLoss mobileProfit refrigeratorProfit televisionLoss indiaTax usTax ukTax germanyTax : ℝ) : ℝ :=
  let grinderSelling := grinderCost * (1 - grinderLoss / 100)
  let mobileSelling := (mobileCost * (1 + mobileProfit / 100)) * rsDollars
  let refrigeratorSelling := (refrigeratorCost * (1 + refrigeratorProfit / 100)) * rsPounds
  let televisionSelling := (televisionCost * (1 - televisionLoss / 100)) * rsEuros
  let grinderProfit := grinderSelling - grinderCost
  let mobileProfitNet := (mobileSelling - mobileCost * rsDollars) * (1 - usTax / 100)
  let refrigeratorProfitNet := (refrigeratorSelling - refrigeratorCost * rsPounds) * (1 - ukTax / 100)
  let televisionProfit := televisionSelling - televisionCost * rsEuros
  (mobileProfitNet + refrigeratorProfitNet) - (grinderCost - grinderSelling + televisionCost * rsEuros - televisionSelling)

theorem overall_loss_rs_3_46 : 
  johnsOverallProfitLoss 15000 75 100 101 200 90 300 4 10 8 6 5 7 6 9 = -3.46 := 
by sorry

end overall_loss_rs_3_46_l248_248729


namespace cyclic_quadrilaterals_count_l248_248063

theorem cyclic_quadrilaterals_count :
  ∃ n : ℕ, n = 568 ∧
  ∀ (a b c d : ℕ), 
    a + b + c + d = 32 ∧
    a ≤ b ∧ b ≤ c ∧ c ≤ d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (a + b + c > d) ∧ (b + c + d > a) ∧ (c + d + a > b) ∧ (d + a + b > c) ∧
    (c - a)^2 + (d - b)^2 = (c + a)^2 + (d + b)^2
      → n = 568 := 
sorry

end cyclic_quadrilaterals_count_l248_248063


namespace non_defective_probability_l248_248224

/-- In a box of 10 pens, a total of 3 are defective. If a customer buys 2 pens selected 
at random from the box, the probability that neither pen will be defective is 7/15. -/
theorem non_defective_probability :
  let total_pens := 10
  let defective_pens := 3
  let non_defective_pens := total_pens - defective_pens
  let first_draw_non_defective_prob := non_defective_pens / total_pens
  let second_draw_non_defective_prob := (non_defective_pens - 1) / (total_pens - 1)
  let both_non_defective_prob := first_draw_non_defective_prob * second_draw_non_defective_prob
  both_non_defective_prob = 7 / 15 :=
by
  let total_pens := 10
  let defective_pens := 3
  let non_defective_pens := total_pens - defective_pens
  let first_draw_non_defective_prob := non_defective_pens / total_pens
  let second_draw_non_defective_prob := (non_defective_pens - 1) / (total_pens - 1)
  let both_non_defective_prob := first_draw_non_defective_prob * second_draw_non_defective_prob
  have h1 : first_draw_non_defective_prob = 7 / 10 := by simp [first_draw_non_defective_prob, non_defective_pens]
  have h2 : second_draw_non_defective_prob = 6 / 9 := by simp [second_draw_non_defective_prob, non_defective_pens]
  have h3 : both_non_defective_prob = (7 / 10) * (6 / 9) := by simp [both_non_defective_prob, h1, h2]
  have h4 : (7 / 10) * (6 / 9) = 42 / 90 := by simp
  have h5 : 42 / 90 = 7 / 15 := by norm_num
  rw [h3, h4, h5]
  sorry

end non_defective_probability_l248_248224


namespace transform_sin_to_cos_l248_248817

theorem transform_sin_to_cos :
  ∀ x : ℝ, y = sin (2 * x + π / 4) ↔ y = cos (2 * x - π / 4) := sorry

end transform_sin_to_cos_l248_248817


namespace current_at_resistance_12_l248_248541

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l248_248541


namespace constant_speed_l248_248552

-- Definitions based on conditions stated in part a)
def poly (t x : ℝ) : ℝ := sorry -- polynomial of t and x

-- position function p(t,x) satisfies given conditions
def p (t x : ℝ) : ℝ := poly t x

-- condition given in the problem
lemma condition (t k x : ℝ) : 
  p(t, x) - p(k, x) = p(t - k, p(k, x)) := sorry

-- now we state the main proposition to prove the speed v is constant
theorem constant_speed (t x : ℝ) (ht : t ≠ 0) (hx : x ≠ 0) :
  ∃ (a : ℝ), ∀ (t : ℝ), speed (t, x) = a := 
sorry

end constant_speed_l248_248552


namespace sqrt_factorial_product_l248_248106

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l248_248106


namespace current_at_R_12_l248_248244

-- Define the function I related to R
def I (R : ℝ) : ℝ := 48 / R

-- Define the condition
def R_val : ℝ := 12

-- State the theorem to prove
theorem current_at_R_12 : I R_val = 4 := by
  -- substitute directly the condition and require to prove equality
  sorry

end current_at_R_12_l248_248244


namespace wade_final_profit_l248_248828

theorem wade_final_profit :
  let tips_per_customer_friday := 2.00
  let customers_friday := 28
  let tips_per_customer_saturday := 2.50
  let customers_saturday := 3 * customers_friday
  let tips_per_customer_sunday := 1.50
  let customers_sunday := 36
  let cost_ingredients_per_hotdog := 1.25
  let price_per_hotdog := 4.00
  let truck_maintenance_daily_cost := 50.00
  let total_taxes := 150.00
  let revenue_tips_friday := tips_per_customer_friday * customers_friday
  let revenue_hotdogs_friday := customers_friday * price_per_hotdog
  let cost_ingredients_friday := customers_friday * cost_ingredients_per_hotdog
  let revenue_friday := revenue_tips_friday + revenue_hotdogs_friday
  let total_costs_friday := cost_ingredients_friday + truck_maintenance_daily_cost
  let profit_friday := revenue_friday - total_costs_friday
  let revenue_tips_saturday := tips_per_customer_saturday * customers_saturday
  let revenue_hotdogs_saturday := customers_saturday * price_per_hotdog
  let cost_ingredients_saturday := customers_saturday * cost_ingredients_per_hotdog
  let revenue_saturday := revenue_tips_saturday + revenue_hotdogs_saturday
  let total_costs_saturday := cost_ingredients_saturday + truck_maintenance_daily_cost
  let profit_saturday := revenue_saturday - total_costs_saturday
  let revenue_tips_sunday := tips_per_customer_sunday * customers_sunday
  let revenue_hotdogs_sunday := customers_sunday * price_per_hotdog
  let cost_ingredients_sunday := customers_sunday * cost_ingredients_per_hotdog
  let revenue_sunday := revenue_tips_sunday + revenue_hotdogs_sunday
  let total_costs_sunday := cost_ingredients_sunday + truck_maintenance_daily_cost
  let profit_sunday := revenue_sunday - total_costs_sunday
  let total_profit := profit_friday + profit_saturday + profit_sunday
  let final_profit := total_profit - total_taxes
  final_profit = 427.00 :=
by
  sorry

end wade_final_profit_l248_248828


namespace battery_current_l248_248316

theorem battery_current (R : ℝ) (I : ℝ) (h₁ : I = 48 / R) (h₂ : R = 12) : I = 4 := 
by
  sorry

end battery_current_l248_248316


namespace equal_sides_implies_equal_angles_l248_248012

theorem equal_sides_implies_equal_angles 
  (h1: ∀ (T: Type) [triangle T], (∀ a b c: T, side_eq T a b c → angle_eq T a b c))
  :
  (∀ (T: Type) [triangle T], (∀ a b c: T, ¬ angle_eq T a b c → ¬ side_eq T a b c)) ∧
  (∀ (T: Type) [triangle T], (∀ a b c: T, angle_eq T a b c → side_eq T a b c)) ∧
  (∀ (T: Type) [triangle T], (∀ a b c: T, side_eq T a b c → angle_eq T a b c)):
  true :=
by
  sorry  -- the proof is omitted as requested

end equal_sides_implies_equal_angles_l248_248012


namespace sqrt_factorial_mul_factorial_eq_l248_248145

theorem sqrt_factorial_mul_factorial_eq :
  Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := by
sorry

end sqrt_factorial_mul_factorial_eq_l248_248145


namespace binomial_7_4_l248_248583

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_7_4 : binomial 7 4 = 35 := by
  sorry

end binomial_7_4_l248_248583


namespace max_value_of_x_minus_y_l248_248677

noncomputable def max_value_x_minus_y (x y : ℝ) : ℝ := x - y

theorem max_value_of_x_minus_y (x y : ℝ) (h1 : 0 < y) (h2 : y ≤ x) (h3 : x < π / 2) (h4 : tan x = 3 * tan y) : 
  max_value_x_minus_y x y = π / 6 := 
by
  sorry

end max_value_of_x_minus_y_l248_248677
