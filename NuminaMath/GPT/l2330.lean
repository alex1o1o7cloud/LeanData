import Mathlib

namespace find_general_formula_l2330_233064

section sequence

variables {R : Type*} [LinearOrderedField R]
variable (c : R)
variable (h_c : c ≠ 0)

def seq (a : Nat → R) : Prop :=
  a 1 = 1 ∧ ∀ n : Nat, n > 0 → a (n + 1) = c * a n + c^(n + 1) * (2 * n + 1)

def general_formula (a : Nat → R) : Prop :=
  ∀ n : Nat, n > 0 → a n = (n^2 - 1) * c^n + c^(n - 1)

theorem find_general_formula :
  ∃ a : Nat → R, seq c a ∧ general_formula c a :=
by
  sorry

end sequence

end find_general_formula_l2330_233064


namespace line_intersect_yaxis_at_l2330_233058

theorem line_intersect_yaxis_at
  (x1 y1 x2 y2 : ℝ) : (x1 = 3) → (y1 = 19) → (x2 = -7) → (y2 = -1) →
  ∃ y : ℝ, (0, y) = (0, 13) :=
by
  intros h1 h2 h3 h4
  sorry

end line_intersect_yaxis_at_l2330_233058


namespace scientific_notation_of_360_billion_l2330_233079

def number_in_scientific_notation (n : ℕ) : String :=
  match n with
  | 360000000000 => "3.6 × 10^11"
  | _ => "Unknown"

theorem scientific_notation_of_360_billion : 
  number_in_scientific_notation 360000000000 = "3.6 × 10^11" :=
by
  -- insert proof steps here
  sorry

end scientific_notation_of_360_billion_l2330_233079


namespace f_sqrt_2_l2330_233050

noncomputable def f : ℝ → ℝ :=
sorry

axiom domain_f : ∀ x, 0 < x → 0 < f x
axiom add_property : ∀ x y, f (x * y) = f x + f y
axiom f_at_8 : f 8 = 6

theorem f_sqrt_2 : f (Real.sqrt 2) = 1 :=
by
  have sqrt2pos : 0 < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  sorry

end f_sqrt_2_l2330_233050


namespace find_base_k_l2330_233097

theorem find_base_k (k : ℕ) (hk : 0 < k) (h : 7/51 = (2 * k + 3) / (k^2 - 1)) : k = 16 :=
sorry

end find_base_k_l2330_233097


namespace hyperbola_vertex_distance_l2330_233008

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  16 * x^2 + 64 * x - 4 * y^2 + 8 * y + 36 = 0

-- Statement: The distance between the vertices of the hyperbola is 1
theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_eq x y → 2 * (1 / 2) = 1 :=
by
  intros x y H
  sorry

end hyperbola_vertex_distance_l2330_233008


namespace product_of_numbers_l2330_233005

theorem product_of_numbers (x y z : ℤ) 
  (h1 : x + y + z = 30) 
  (h2 : x = 3 * ((y + z) - 2))
  (h3 : y = 4 * z - 1) : 
  x * y * z = 294 := 
  sorry

end product_of_numbers_l2330_233005


namespace quadratic_equation_reciprocal_integer_roots_l2330_233046

noncomputable def quadratic_equation_conditions (a b c : ℝ) : Prop :=
  (∃ r : ℝ, (r * (1/r) = 1) ∧ (r + (1/r) = 4)) ∧ 
  (c = a) ∧ 
  (b = -4 * a)

theorem quadratic_equation_reciprocal_integer_roots (a b c : ℝ) (h1 : quadratic_equation_conditions a b c) : 
  c = a ∧ b = -4 * a :=
by
  obtain ⟨r, hr₁, hr₂⟩ := h1.1
  sorry

end quadratic_equation_reciprocal_integer_roots_l2330_233046


namespace find_number_of_girls_l2330_233051

variable (B G : ℕ)

theorem find_number_of_girls
  (h1 : B = G / 2)
  (h2 : B + G = 90)
  : G = 60 :=
sorry

end find_number_of_girls_l2330_233051


namespace angelina_speed_l2330_233039

theorem angelina_speed (v : ℝ) (h1 : 200 / v - 50 = 300 / (2 * v)) : 2 * v = 2 := 
by
  sorry

end angelina_speed_l2330_233039


namespace minvalue_expression_l2330_233026

theorem minvalue_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
    9 * z / (3 * x + y) + 9 * x / (y + 3 * z) + 4 * y / (x + z) ≥ 3 := 
by
  sorry

end minvalue_expression_l2330_233026


namespace no_right_obtuse_triangle_l2330_233028

theorem no_right_obtuse_triangle :
  ∀ (α β γ : ℝ),
  (α + β + γ = 180) →
  (α = 90 ∨ β = 90 ∨ γ = 90) →
  (α > 90 ∨ β > 90 ∨ γ > 90) →
  false :=
by
  sorry

end no_right_obtuse_triangle_l2330_233028


namespace poly_a_roots_poly_b_roots_l2330_233031

-- Define the polynomials
def poly_a (x : ℤ) : ℤ := 2 * x ^ 3 - 3 * x ^ 2 - 11 * x + 6
def poly_b (x : ℤ) : ℤ := x ^ 4 + 4 * x ^ 3 - 9 * x ^ 2 - 16 * x + 20

-- Assert the integer roots for poly_a
theorem poly_a_roots : {x : ℤ | poly_a x = 0} = {-2, 3} := sorry

-- Assert the integer roots for poly_b
theorem poly_b_roots : {x : ℤ | poly_b x = 0} = {1, 2, -2, -5} := sorry

end poly_a_roots_poly_b_roots_l2330_233031


namespace find_B_l2330_233020

variable {A B C a b c : Real}

noncomputable def B_value (A B C a b c : Real) : Prop :=
  B = 2 * Real.pi / 3

theorem find_B 
  (h_triangle: a^2 + b^2 + c^2 = 2*a*b*Real.cos C)
  (h_cos_eq: (2 * a + c) * Real.cos B + b * Real.cos C = 0) : 
  B_value A B C a b c :=
by
  sorry

end find_B_l2330_233020


namespace final_result_is_8_l2330_233086

theorem final_result_is_8 (n : ℕ) (h1 : n = 2976) (h2 : (n / 12) - 240 = 8) : (n / 12) - 240 = 8 :=
by {
  -- Proof steps would go here
  sorry
}

end final_result_is_8_l2330_233086


namespace matrix_mult_3I_l2330_233045

variable (N : Matrix (Fin 3) (Fin 3) ℝ)

theorem matrix_mult_3I (w : Fin 3 → ℝ):
  (∀ (w : Fin 3 → ℝ), N.mulVec w = 3 * w) ↔ (N = 3 • (1 : Matrix (Fin 3) (Fin 3) ℝ)) :=
by
  sorry

end matrix_mult_3I_l2330_233045


namespace line_eq_l2330_233055

theorem line_eq (x y : ℝ) (point eq_direction_vector) (h₀ : point = (3, -2))
    (h₁ : eq_direction_vector = (-5, 3)) :
    3 * x + 5 * y + 1 = 0 := by sorry

end line_eq_l2330_233055


namespace arithmetic_mean_probability_l2330_233025

theorem arithmetic_mean_probability
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : b = (a + c) / 2) :
  b = 1 / 3 :=
by
  sorry

end arithmetic_mean_probability_l2330_233025


namespace malcolm_joshua_time_difference_l2330_233016

-- Define the constants
def malcolm_speed : ℕ := 5 -- minutes per mile
def joshua_speed : ℕ := 8 -- minutes per mile
def race_distance : ℕ := 12 -- miles

-- Define the times it takes each runner to finish
def malcolm_time : ℕ := malcolm_speed * race_distance
def joshua_time : ℕ := joshua_speed * race_distance

-- Define the time difference and the proof statement
def time_difference : ℕ := joshua_time - malcolm_time

theorem malcolm_joshua_time_difference : time_difference = 36 := by
  sorry

end malcolm_joshua_time_difference_l2330_233016


namespace area_between_chords_is_correct_l2330_233059

noncomputable def circle_radius : ℝ := 10
noncomputable def chord_distance_apart : ℝ := 12
noncomputable def area_between_chords : ℝ := 44.73

theorem area_between_chords_is_correct 
    (r : ℝ) (d : ℝ) (A : ℝ) 
    (hr : r = circle_radius) 
    (hd : d = chord_distance_apart) 
    (hA : A = area_between_chords) : 
    ∃ area : ℝ, area = A := by 
  sorry

end area_between_chords_is_correct_l2330_233059


namespace employee_saves_86_25_l2330_233094

def initial_purchase_price : ℝ := 500
def markup_rate : ℝ := 0.15
def employee_discount_rate : ℝ := 0.15

def retail_price : ℝ := initial_purchase_price * (1 + markup_rate)
def employee_discount_amount : ℝ := retail_price * employee_discount_rate
def employee_savings : ℝ := retail_price - (retail_price - employee_discount_amount)

theorem employee_saves_86_25 :
  employee_savings = 86.25 := 
sorry

end employee_saves_86_25_l2330_233094


namespace find_p_l2330_233074

theorem find_p (p : ℕ) : 64^5 = 8^p → p = 10 :=
by
  intro h
  sorry

end find_p_l2330_233074


namespace triangle_side_difference_l2330_233024

theorem triangle_side_difference (y : ℝ) (h : y > 6) :
  max (y + 6) (y + 3) - min (y + 6) (y + 3) = 3 :=
by
  sorry

end triangle_side_difference_l2330_233024


namespace sum_of_two_primes_l2330_233090

theorem sum_of_two_primes (k : ℕ) (n : ℕ) (h : n = 1 + 10 * k) :
  (n = 1 ∨ ∃ p1 p2 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ n = p1 + p2) :=
by
  sorry

end sum_of_two_primes_l2330_233090


namespace wire_length_approx_is_correct_l2330_233085

noncomputable def S : ℝ := 5.999999999999998
noncomputable def L : ℝ := (5 / 2) * S
noncomputable def W : ℝ := S + L

theorem wire_length_approx_is_correct : abs (W - 21) < 1e-16 := by
  sorry

end wire_length_approx_is_correct_l2330_233085


namespace polygon_sides_from_diagonals_l2330_233013

/-- A theorem to prove that a regular polygon with 740 diagonals has 40 sides. -/
theorem polygon_sides_from_diagonals (n : ℕ) (h : (n * (n - 3)) / 2 = 740) : n = 40 := sorry

end polygon_sides_from_diagonals_l2330_233013


namespace initial_amount_of_milk_l2330_233000

theorem initial_amount_of_milk (M : ℝ) (h : 0 < M) (h2 : 0.10 * M = 0.05 * (M + 20)) : M = 20 := 
sorry

end initial_amount_of_milk_l2330_233000


namespace sum_of_two_numbers_l2330_233032

theorem sum_of_two_numbers (x : ℤ) (sum certain value : ℤ) (h₁ : 25 - x = 5) : 25 + x = 45 := by
  sorry

end sum_of_two_numbers_l2330_233032


namespace midpoint_condition_l2330_233027

theorem midpoint_condition (c : ℝ) :
  (∃ A B : ℝ × ℝ,
    A ∈ { p : ℝ × ℝ | p.2 = p.1^2 - 2 * p.1 - 3 } ∧
    B ∈ { p : ℝ × ℝ | p.2 = p.1^2 - 2 * p.1 - 3 } ∧
    A ∈ { p : ℝ × ℝ | p.2 = -p.1^2 + 4 * p.1 + c } ∧
    B ∈ { p : ℝ × ℝ | p.2 = -p.1^2 + 4 * p.1 + c } ∧
    ((A.1 + B.1) / 2 + (A.2 + B.2) / 2) = 2017
  ) ↔
  c = 4031 := sorry

end midpoint_condition_l2330_233027


namespace diagonal_length_l2330_233077

noncomputable def convertHectaresToSquareMeters (hectares : ℝ) : ℝ :=
  hectares * 10000

noncomputable def sideLength (areaSqMeters : ℝ) : ℝ :=
  Real.sqrt areaSqMeters

noncomputable def diagonal (side : ℝ) : ℝ :=
  side * Real.sqrt 2

theorem diagonal_length (area : ℝ) (h : area = 1 / 2) :
  let areaSqMeters := convertHectaresToSquareMeters area
  let side := sideLength areaSqMeters
  let diag := diagonal side
  abs (diag - 100) < 1 :=
by
  sorry

end diagonal_length_l2330_233077


namespace fraction_replaced_l2330_233012

theorem fraction_replaced (x : ℝ) (h₁ : 0.15 * (1 - x) + 0.19000000000000007 * x = 0.16) : x = 0.25 :=
by
  sorry

end fraction_replaced_l2330_233012


namespace eliana_refill_l2330_233076

theorem eliana_refill (total_spent cost_per_refill : ℕ) (h1 : total_spent = 63) (h2 : cost_per_refill = 21) : (total_spent / cost_per_refill) = 3 :=
sorry

end eliana_refill_l2330_233076


namespace James_present_age_l2330_233080

variable (D J : ℕ)

theorem James_present_age 
  (h1 : D / J = 6 / 5)
  (h2 : D + 4 = 28) :
  J = 20 := 
by
  sorry

end James_present_age_l2330_233080


namespace solve_x_l2330_233099

-- Define the structure of the pyramid
def pyramid (x : ℕ) : Prop :=
  let level1 := [x + 4, 12, 15, 18]
  let level2 := [x + 16, 27, 33]
  let level3 := [x + 43, 60]
  let top := x + 103
  top = 120

theorem solve_x : ∃ x : ℕ, pyramid x → x = 17 :=
by
  -- Proof omitted
  sorry

end solve_x_l2330_233099


namespace age_difference_l2330_233023

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 16) : A - C = 16 :=
sorry

end age_difference_l2330_233023


namespace power_function_solution_l2330_233067

theorem power_function_solution (f : ℝ → ℝ) (alpha : ℝ)
  (h₀ : ∀ x, f x = x ^ alpha)
  (h₁ : f (1 / 8) = 2) :
  f (-1 / 8) = -2 :=
sorry

end power_function_solution_l2330_233067


namespace triangle_solution_condition_l2330_233075

-- Definitions of segments
variables {A B D E : Type}
variables (c f g : Real)

-- Allow noncomputable definitions for geometric constraints
noncomputable def triangle_construction (c f g : Real) : String :=
  if c > f then "more than one solution"
  else if c = f then "exactly one solution"
  else "no solution"

-- The proof problem statement
theorem triangle_solution_condition (c f g : Real) :
  (c > f → triangle_construction c f g = "more than one solution") ∧
  (c = f → triangle_construction c f g = "exactly one solution") ∧
  (c < f → triangle_construction c f g = "no solution") :=
by
  sorry

end triangle_solution_condition_l2330_233075


namespace range_of_k_decreasing_l2330_233010

theorem range_of_k_decreasing (k b : ℝ) (h : ∀ x₁ x₂, x₁ < x₂ → (k^2 - 3*k + 2) * x₁ + b > (k^2 - 3*k + 2) * x₂ + b) : 1 < k ∧ k < 2 :=
by
  -- Proof 
  sorry

end range_of_k_decreasing_l2330_233010


namespace selling_price_l2330_233082

/-- 
Prove that the selling price (S) of an article with a cost price (C) of 180 sold at a 15% profit (P) is 207.
-/
theorem selling_price (C P S : ℝ) (hC : C = 180) (hP : P = 15) (hS : S = 207) :
  S = C + (P / 100 * C) :=
by
  -- here we rely on sorry to skip the proof details
  sorry

end selling_price_l2330_233082


namespace mary_needs_10_charges_to_vacuum_house_l2330_233035

theorem mary_needs_10_charges_to_vacuum_house :
  (let bedroom_time := 10
   let kitchen_time := 12
   let living_room_time := 8
   let dining_room_time := 6
   let office_time := 9
   let bathroom_time := 5
   let battery_duration := 8
   3 * bedroom_time + kitchen_time + living_room_time + dining_room_time + office_time + 2 * bathroom_time) / battery_duration = 10 :=
by sorry

end mary_needs_10_charges_to_vacuum_house_l2330_233035


namespace car_catch_truck_l2330_233065

theorem car_catch_truck (truck_speed car_speed : ℕ) (time_head_start : ℕ) (t : ℕ)
  (h1 : truck_speed = 45) (h2 : car_speed = 60) (h3 : time_head_start = 1) :
  45 * t + 45 = 60 * t → t = 3 := by
  intro h
  sorry

end car_catch_truck_l2330_233065


namespace problem_l2330_233049

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≥ f y
def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def max_value_in (f : ℝ → ℝ) (a b : ℝ) (v : ℝ) : Prop := ∀ x, a ≤ x → x ≤ b → f x ≤ v ∧ (∃ z, a ≤ z ∧ z ≤ b ∧ f z = v)

theorem problem
  (h_even : even_function f)
  (h_decreasing : decreasing_on f (-5) (-2))
  (h_max : max_value_in f (-5) (-2) 7) :
  increasing_on f 2 5 ∧ max_value_in f 2 5 7 :=
by
  sorry

end problem_l2330_233049


namespace circle_cartesian_line_circle_intersect_l2330_233087

noncomputable def L_parametric (t : ℝ) : ℝ × ℝ :=
  (t, 1 + 2 * t)

noncomputable def C_polar (θ : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

def L_cartesian (x y : ℝ) : Prop :=
  y = 2 * x + 1

def C_cartesian (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem circle_cartesian :
  ∀ x y : ℝ, C_polar x = y ↔ C_cartesian x y :=
sorry

theorem line_circle_intersect (x y : ℝ) :
  L_cartesian x y → C_cartesian x y → True :=
sorry

end circle_cartesian_line_circle_intersect_l2330_233087


namespace problem_equivalence_l2330_233001

theorem problem_equivalence :
  (∃ a a1 a2 a3 a4 a5 : ℝ, ((1 - x)^5 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5)) → 
  ∀ (a a1 a2 a3 a4 a5 : ℝ), (1 - 1)^5 = a + a1 + a2 + a3 + a4 + a5 →
  (1 + 1)^5 = a - a1 + a2 - a3 + a4 - a5 →
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := 
by
  intros h a a1 a2 a3 a4 a5 e1 e2
  sorry

end problem_equivalence_l2330_233001


namespace Holly_throws_5_times_l2330_233034

def Bess.throw_distance := 20
def Bess.throw_times := 4
def Holly.throw_distance := 8
def total_distance := 200

theorem Holly_throws_5_times : 
  (total_distance - Bess.throw_times * 2 * Bess.throw_distance) / Holly.throw_distance = 5 :=
by 
  sorry

end Holly_throws_5_times_l2330_233034


namespace smallest_angle_in_convex_20_gon_seq_l2330_233053

theorem smallest_angle_in_convex_20_gon_seq :
  ∃ (α : ℕ), (α + 19 * (1:ℕ) = 180 ∧ α < 180 ∧ ∀ n, 1 ≤ n ∧ n ≤ 20 → α + (n - 1) * 1 < 180) ∧ α = 161 := 
by
  sorry

end smallest_angle_in_convex_20_gon_seq_l2330_233053


namespace no_primes_divisible_by_60_l2330_233095

theorem no_primes_divisible_by_60 (p : ℕ) (prime_p : Nat.Prime p) : ¬ (60 ∣ p) :=
by
  sorry

end no_primes_divisible_by_60_l2330_233095


namespace trig_inequality_l2330_233017

theorem trig_inequality (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.cos β)^2 * (Real.sin β)^2) ≥ 9) := by
  sorry

end trig_inequality_l2330_233017


namespace carol_name_tag_l2330_233071

theorem carol_name_tag (a b c : ℕ) (ha : Prime a ∧ a ≥ 10 ∧ a < 100) (hb : Prime b ∧ b ≥ 10 ∧ b < 100) (hc : Prime c ∧ c ≥ 10 ∧ c < 100) 
  (h1 : b + c = 14) (h2 : a + c = 20) (h3 : a + b = 18) : c = 11 := 
by 
  sorry

end carol_name_tag_l2330_233071


namespace roots_abs_lt_one_l2330_233030

theorem roots_abs_lt_one
  (a b : ℝ)
  (h1 : |a| + |b| < 1)
  (h2 : a^2 - 4 * b ≥ 0) :
  ∀ (x : ℝ), x^2 + a * x + b = 0 → |x| < 1 :=
sorry

end roots_abs_lt_one_l2330_233030


namespace monotonicity_f_a_eq_1_domain_condition_inequality_condition_l2330_233041

noncomputable def f (x a : ℝ) := (Real.log (x^2 - 2 * x + a)) / (x - 1)

theorem monotonicity_f_a_eq_1 :
  ∀ x : ℝ, 1 < x → 
  (f x 1 < f (e + 1) 1 → 
   ∀ y, 1 < y ∧ y < e + 1 → f y 1 < f (e + 1) 1) ∧ 
  (f (e + 1) 1 < f x 1 → 
   ∀ z, e + 1 < z → f (e + 1) 1 < f z 1) :=
sorry

theorem domain_condition (a : ℝ) :
  (∀ x : ℝ, (x < 1 ∨ x > 1) → x^2 - 2 * x + a > 0) ↔ a ≥ 1 :=
sorry

theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f x a < (x - 1) * Real.exp x)) ↔ (1 + 1 / Real.exp 1 ≤ a ∧ a ≤ 2) :=
sorry

end monotonicity_f_a_eq_1_domain_condition_inequality_condition_l2330_233041


namespace points_on_parabola_l2330_233033

theorem points_on_parabola (a : ℝ) (y1 y2 y3 : ℝ) 
  (h_a : a < -1) 
  (h1 : y1 = (a - 1)^2) 
  (h2 : y2 = a^2) 
  (h3 : y3 = (a + 1)^2) : 
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end points_on_parabola_l2330_233033


namespace solve_number_puzzle_l2330_233054

def number_puzzle (N : ℕ) : Prop :=
  (1/4) * (1/3) * (2/5) * N = 14 → (40/100) * N = 168

theorem solve_number_puzzle : ∃ N, number_puzzle N := by
  sorry

end solve_number_puzzle_l2330_233054


namespace fraction_of_n_is_80_l2330_233003

-- Definitions from conditions
def n := (5 / 6) * 240

-- The theorem we want to prove
theorem fraction_of_n_is_80 : (2 / 5) * n = 80 :=
by
  -- This is just a placeholder to complete the statement, 
  -- actual proof logic is not included based on the prompt instructions
  sorry

end fraction_of_n_is_80_l2330_233003


namespace range_of_k_l2330_233036

def P (x k : ℝ) : Prop := x^2 + k*x + 1 > 0
def Q (x k : ℝ) : Prop := k*x^2 + x + 2 < 0

theorem range_of_k (k : ℝ) : (¬ (P 2 k ∧ Q 2 k)) ↔ k ∈ (Set.Iic (-5/2) ∪ Set.Ici (-1)) := 
by
  sorry

end range_of_k_l2330_233036


namespace fraction_a_over_b_l2330_233092

theorem fraction_a_over_b (x y a b : ℝ) (hb : b ≠ 0) (h1 : 4 * x - 2 * y = a) (h2 : 9 * y - 18 * x = b) :
  a / b = -2 / 9 :=
by
  sorry

end fraction_a_over_b_l2330_233092


namespace intersection_cardinality_l2330_233083

variable {a b : ℝ}
variable {f : ℝ → ℝ}

theorem intersection_cardinality {a b : ℝ} {f : ℝ → ℝ} :
  (∃! y, (0, y) ∈ ({ (x, y) | y = f x ∧ a ≤ x ∧ x ≤ b } ∩ { (x, y) | x = 0 })) ∨
  ¬ (∃ y, (0, y) ∈ { (x, y) | y = f x ∧ a ≤ x ∧ x ≤ b }) :=
by
  sorry

end intersection_cardinality_l2330_233083


namespace wholesale_price_l2330_233057

theorem wholesale_price (R : ℝ) (W : ℝ)
  (hR : R = 120)
  (h_discount : ∀ SP : ℝ, SP = R - (0.10 * R))
  (h_profit : ∀ P : ℝ, P = 0.20 * W)
  (h_SP_eq_W_P : ∀ SP P : ℝ, SP = W + P) :
  W = 90 := by
  sorry

end wholesale_price_l2330_233057


namespace Jerry_needs_72_dollars_l2330_233014

def action_figures_current : ℕ := 7
def action_figures_total : ℕ := 16
def cost_per_figure : ℕ := 8
def money_needed : ℕ := 72

theorem Jerry_needs_72_dollars : 
  (action_figures_total - action_figures_current) * cost_per_figure = money_needed :=
by
  sorry

end Jerry_needs_72_dollars_l2330_233014


namespace problem1_problem2_problem3_l2330_233073

-- Problem 1
theorem problem1 (A B : Set α) (x : α) : x ∈ A ∪ B → x ∈ A ∨ x ∈ B :=
by sorry

-- Problem 2
theorem problem2 (A B : Set α) (x : α) : x ∈ A ∩ B → x ∈ A ∧ x ∈ B :=
by sorry

-- Problem 3
theorem problem3 (a b : ℝ) : a > 0 ∧ b > 0 → a * b > 0 :=
by sorry

end problem1_problem2_problem3_l2330_233073


namespace parabola_directrix_l2330_233069

theorem parabola_directrix (vertex_origin : ∀ (x y : ℝ), x = 0 ∧ y = 0)
    (directrix : ∀ (y : ℝ), y = 4) : ∃ p, x^2 = -2 * p * y ∧ p = 8 ∧ x^2 = -16 * y := 
sorry

end parabola_directrix_l2330_233069


namespace probability_diff_topics_l2330_233093

theorem probability_diff_topics
  (num_topics : ℕ)
  (num_combinations : ℕ)
  (num_different_combinations : ℕ)
  (h1 : num_topics = 6)
  (h2 : num_combinations = num_topics * num_topics)
  (h3 : num_combinations = 36)
  (h4 : num_different_combinations = num_topics * (num_topics - 1))
  (h5 : num_different_combinations = 30) :
  (num_different_combinations / num_combinations) = 5 / 6 := 
by 
  sorry

end probability_diff_topics_l2330_233093


namespace f_neg_1_l2330_233002

-- Define the functions
variable (f : ℝ → ℝ) -- f is a real-valued function
variable (g : ℝ → ℝ) -- g is a real-valued function

-- Given conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_def : ∀ x, g x = f x + 4
axiom g_at_1 : g 1 = 2

-- Define the theorem to prove
theorem f_neg_1 : f (-1) = 2 :=
by
  -- Proof goes here
  sorry

end f_neg_1_l2330_233002


namespace age_of_b_l2330_233006

variable {a b c d Y : ℝ}

-- Conditions
def condition1 (a b : ℝ) := a = b + 2
def condition2 (b c : ℝ) := b = 2 * c
def condition3 (a d : ℝ) := d = a / 2
def condition4 (a b c d Y : ℝ) := a + b + c + d = Y

-- Theorem to prove
theorem age_of_b (a b c d Y : ℝ) 
  (h1 : condition1 a b) 
  (h2 : condition2 b c) 
  (h3 : condition3 a d) 
  (h4 : condition4 a b c d Y) : 
  b = Y / 3 - 1 := 
sorry

end age_of_b_l2330_233006


namespace absolute_value_simplification_l2330_233098

theorem absolute_value_simplification (x : ℝ) (h : x > 3) : 
  |x - Real.sqrt ((x - 3)^2)| = 3 := 
by 
  sorry

end absolute_value_simplification_l2330_233098


namespace gain_percent_l2330_233047

theorem gain_percent (CP SP : ℝ) (hCP : CP = 110) (hSP : SP = 125) : 
  (SP - CP) / CP * 100 = 13.64 := by
  sorry

end gain_percent_l2330_233047


namespace monthly_earnings_l2330_233072

-- Defining the initial conditions and known information
def current_worth : ℝ := 90
def months : ℕ := 5

-- Let I be the initial investment, and E be the earnings per month.

noncomputable def initial_investment (I : ℝ) := I * 3 = current_worth
noncomputable def earned_twice_initial (E : ℝ) (I : ℝ) := E * months = 2 * I

-- Proving the monthly earnings
theorem monthly_earnings (I E : ℝ) (h1 : initial_investment I) (h2 : earned_twice_initial E I) : E = 12 :=
sorry

end monthly_earnings_l2330_233072


namespace nails_needed_for_house_wall_l2330_233060

theorem nails_needed_for_house_wall
    (large_planks : ℕ)
    (small_planks : ℕ)
    (nails_for_large_planks : ℕ)
    (nails_for_small_planks : ℕ)
    (H1 : large_planks = 12)
    (H2 : small_planks = 10)
    (H3 : nails_for_large_planks = 15)
    (H4 : nails_for_small_planks = 5) :
    (nails_for_large_planks + nails_for_small_planks) = 20 := by
  sorry

end nails_needed_for_house_wall_l2330_233060


namespace correct_answer_l2330_233056

theorem correct_answer (m n : ℤ) (h : 3 * m * n + 3 * m = n + 2) : 3 * m + n = -2 := 
by
  sorry

end correct_answer_l2330_233056


namespace range_of_c_l2330_233084

def P (c : ℝ) : Prop := ∀ x1 x2 : ℝ, x1 < x2 → (c ^ x1) > (c ^ x2)
def q (c : ℝ) : Prop := ∀ x : ℝ, x > (1 / 2) → (2 * c * x - c) > 0

theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1)
  (h3 : ¬ (P c ∧ q c)) (h4 : (P c ∨ q c)) :
  (1 / 2) < c ∧ c < 1 :=
by
  sorry

end range_of_c_l2330_233084


namespace largest_multiple_of_7_negation_greater_than_neg_150_l2330_233021

theorem largest_multiple_of_7_negation_greater_than_neg_150 : 
  ∃ k : ℤ, k * 7 = 147 ∧ ∀ n : ℤ, (k < n → n * 7 ≤ 150) :=
by
  use 21
  sorry

end largest_multiple_of_7_negation_greater_than_neg_150_l2330_233021


namespace distinct_shell_arrangements_l2330_233042

/--
John draws a regular five pointed star and places one of ten different sea shells at each of the 5 outward-pointing points and 5 inward-pointing points. 
Considering rotations and reflections of an arrangement as equivalent, prove that the number of ways he can place the shells is 362880.
-/
theorem distinct_shell_arrangements : 
  let total_arrangements := Nat.factorial 10
  let symmetries := 10
  total_arrangements / symmetries = 362880 :=
by
  sorry

end distinct_shell_arrangements_l2330_233042


namespace intersection_A_B_l2330_233043

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

def setA : Set ℝ := { x | Real.log x > 0 }
def setB : Set ℝ := { x | Real.exp x * Real.exp x < 3 }

theorem intersection_A_B : setA ∩ setB = { x | 1 < x ∧ x < log2 3 } :=
by
  sorry

end intersection_A_B_l2330_233043


namespace max_third_side_l2330_233061

open Real

variables {A B C : ℝ} {a b c : ℝ} 

theorem max_third_side (h : cos (4 * A) + cos (4 * B) + cos (4 * C) = 1) 
                       (ha : a = 8) (hb : b = 15) : c = 17 :=
 by
  sorry 

end max_third_side_l2330_233061


namespace sum_of_numbers_l2330_233063

-- Define the given conditions.
def S : ℕ := 30
def F : ℕ := 2 * S
def T : ℕ := F / 3

-- State the proof problem.
theorem sum_of_numbers : F + S + T = 110 :=
by
  -- Assume the proof here.
  sorry

end sum_of_numbers_l2330_233063


namespace find_abc_l2330_233081

theorem find_abc (a b c : ℤ) 
  (h₁ : a^4 - 2 * b^2 = a)
  (h₂ : b^4 - 2 * c^2 = b)
  (h₃ : c^4 - 2 * a^2 = c)
  (h₄ : a + b + c = -3) : 
  a = -1 ∧ b = -1 ∧ c = -1 := 
sorry

end find_abc_l2330_233081


namespace train_stops_one_minute_per_hour_l2330_233022

theorem train_stops_one_minute_per_hour (D : ℝ) (h1 : D / 400 = T₁) (h2 : D / 360 = T₂) : 
  (T₂ - T₁) * 60 = 1 :=
by
  sorry

end train_stops_one_minute_per_hour_l2330_233022


namespace find_x_l2330_233066

theorem find_x (n : ℕ) (h1 : x = 8^n - 1) (h2 : Nat.Prime 31) 
  (h3 : ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 = 31 ∧ 
  (∀ p : ℕ, Nat.Prime p → p ∣ x → (p = p1 ∨ p = p2 ∨ p = p3))) : 
  x = 32767 :=
by
  sorry

end find_x_l2330_233066


namespace ellipse_major_minor_axes_product_l2330_233040

-- Definitions based on conditions
def OF : ℝ := 8
def inradius_triangle_OCF : ℝ := 2  -- diameter / 2

-- Define a and b based on the ellipse properties and conditions
def a : ℝ := 10  -- Solved from the given conditions and steps
def b : ℝ := 6   -- Solved from the given conditions and steps

-- Defining the axes of the ellipse in terms of a and b
def AB : ℝ := 2 * a
def CD : ℝ := 2 * b

-- The product (AB)(CD) we are interested in
def product_AB_CD := AB * CD

-- The main proof statement
theorem ellipse_major_minor_axes_product : product_AB_CD = 240 :=
by
  sorry

end ellipse_major_minor_axes_product_l2330_233040


namespace ratio_wx_l2330_233078

theorem ratio_wx (w x y : ℚ) (h1 : w / y = 3 / 4) (h2 : (x + y) / y = 13 / 4) : w / x = 1 / 3 :=
  sorry

end ratio_wx_l2330_233078


namespace negation_false_l2330_233096

theorem negation_false (a b : ℝ) : ¬ ((a ≤ 1 ∨ b ≤ 1) → a + b ≤ 2) :=
sorry

end negation_false_l2330_233096


namespace value_of_x_l2330_233029

theorem value_of_x (x : ℕ) : (8^4 + 8^4 + 8^4 = 2^x) → x = 13 :=
by
  sorry

end value_of_x_l2330_233029


namespace trigonometric_expression_l2330_233062

variable (α : Real)
open Real

theorem trigonometric_expression (h : tan α = 3) : 
  (2 * sin α - cos α) / (sin α + 3 * cos α) = 5 / 6 := 
by
  sorry

end trigonometric_expression_l2330_233062


namespace find_b_l2330_233044

noncomputable def circle1 (x y a : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 5 - a^2 = 0
noncomputable def circle2 (x y b : ℝ) : Prop := x^2 + y^2 - (2*b - 10)*x - 2*b*y + 2*b^2 - 10*b + 16 = 0
def is_intersection (x1 y1 x2 y2 : ℝ) : Prop := x1^2 + y1^2 = x2^2 + y2^2

theorem find_b (a x1 y1 x2 y2 : ℝ) (b : ℝ) :
  (circle1 x1 y1 a) ∧ (circle1 x2 y2 a) ∧ 
  (circle2 x1 y1 b) ∧ (circle2 x2 y2 b) ∧ 
  is_intersection x1 y1 x2 y2 →
  b = 5 / 3 :=
sorry

end find_b_l2330_233044


namespace min_value_of_c_l2330_233048

noncomputable def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k^2 = n

theorem min_value_of_c (c : ℕ) (n m : ℕ) (h1 : 5 * c = n^3) (h2 : 3 * c = m^2) : c = 675 := by
  sorry

end min_value_of_c_l2330_233048


namespace find_f_f_neg1_l2330_233052

def f (x : Int) : Int :=
  if x >= 0 then x + 2 else 1

theorem find_f_f_neg1 : f (f (-1)) = 3 :=
by
  sorry

end find_f_f_neg1_l2330_233052


namespace mingi_math_test_total_pages_l2330_233070

theorem mingi_math_test_total_pages (first_page last_page : Nat) (h_first_page : first_page = 8) (h_last_page : last_page = 21) : first_page <= last_page -> ((last_page - first_page + 1) = 14) :=
by
  sorry

end mingi_math_test_total_pages_l2330_233070


namespace original_number_of_boys_l2330_233011

theorem original_number_of_boys (n : ℕ) (W : ℕ) 
  (h1 : W = n * 35) 
  (h2 : W + 135 = (n + 3) * 36) : 
  n = 27 := 
by 
  sorry

end original_number_of_boys_l2330_233011


namespace cone_cube_volume_ratio_l2330_233004

noncomputable def volumeRatio (s : ℝ) : ℝ :=
  let r := s / 2
  let h := s
  let volume_cone := (1 / 3) * Real.pi * r^2 * h
  let volume_cube := s^3
  volume_cone / volume_cube

theorem cone_cube_volume_ratio (s : ℝ) (h_cube_eq_s : s > 0) :
  volumeRatio s = Real.pi / 12 :=
by
  sorry

end cone_cube_volume_ratio_l2330_233004


namespace m_value_for_positive_root_eq_l2330_233037

-- We start by defining the problem:
-- Given the condition that the equation (3x - 1)/(x + 1) - m/(x + 1) = 1 has a positive root,
-- we need to prove that m = -4.

theorem m_value_for_positive_root_eq (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 * x - 1) / (x + 1) - m / (x + 1) = 1) → m = -4 :=
by
  sorry

end m_value_for_positive_root_eq_l2330_233037


namespace points_earned_l2330_233088

def each_enemy_points : ℕ := 3
def total_enemies : ℕ := 6
def defeated_enemies : ℕ := total_enemies - 2

theorem points_earned : defeated_enemies * each_enemy_points = 12 :=
by
  -- proof goes here
  sorry

end points_earned_l2330_233088


namespace local_value_of_7_in_diff_l2330_233015

-- Definitions based on conditions
def local_value (n : ℕ) (d : ℕ) : ℕ :=
  if h : d < 10 ∧ (n / Nat.pow 10 (Nat.log 10 n - Nat.log 10 d)) % 10 = d then
    d * Nat.pow 10 (Nat.log 10 n - Nat.log 10 d)
  else
    0

def diff (a b : ℕ) : ℕ := a - b

-- Question translated to Lean 4 statement
theorem local_value_of_7_in_diff :
  local_value (diff 100889 (local_value 28943712 3)) 7 = 70000 :=
by sorry

end local_value_of_7_in_diff_l2330_233015


namespace joey_hourly_wage_l2330_233089

def sneakers_cost : ℕ := 92
def mowing_earnings (lawns : ℕ) (rate : ℕ) : ℕ := lawns * rate
def selling_earnings (figures : ℕ) (rate : ℕ) : ℕ := figures * rate
def total_additional_earnings (mowing : ℕ) (selling : ℕ) : ℕ := mowing + selling
def remaining_amount (total_cost : ℕ) (earned : ℕ) : ℕ := total_cost - earned
def hourly_wage (remaining : ℕ) (hours : ℕ) : ℕ := remaining / hours

theorem joey_hourly_wage :
  let total_mowing := mowing_earnings 3 8
  let total_selling := selling_earnings 2 9
  let total_earned := total_additional_earnings total_mowing total_selling
  let remaining := remaining_amount sneakers_cost total_earned
  hourly_wage remaining 10 = 5 :=
by
  sorry

end joey_hourly_wage_l2330_233089


namespace final_game_deficit_l2330_233019

-- Define the points for each scoring action
def free_throw_points := 1
def three_pointer_points := 3
def jump_shot_points := 2
def layup_points := 2
def and_one_points := layup_points + free_throw_points

-- Define the points scored by Liz
def liz_free_throws := 5 * free_throw_points
def liz_three_pointers := 4 * three_pointer_points
def liz_jump_shots := 5 * jump_shot_points
def liz_and_one := and_one_points

def liz_points := liz_free_throws + liz_three_pointers + liz_jump_shots + liz_and_one

-- Define the points scored by Taylor
def taylor_three_pointers := 2 * three_pointer_points
def taylor_jump_shots := 3 * jump_shot_points

def taylor_points := taylor_three_pointers + taylor_jump_shots

-- Define the points for Liz's team
def team_points := liz_points + taylor_points

-- Define the points scored by the opposing team players
def opponent_player1_points := 4 * three_pointer_points

def opponent_player2_jump_shots := 4 * jump_shot_points
def opponent_player2_free_throws := 2 * free_throw_points
def opponent_player2_points := opponent_player2_jump_shots + opponent_player2_free_throws

def opponent_player3_jump_shots := 2 * jump_shot_points
def opponent_player3_three_pointer := 1 * three_pointer_points
def opponent_player3_points := opponent_player3_jump_shots + opponent_player3_three_pointer

-- Define the points for the opposing team
def opponent_team_points := opponent_player1_points + opponent_player2_points + opponent_player3_points

-- Initial deficit
def initial_deficit := 25

-- Final net scoring in the final quarter
def net_quarter_scoring := team_points - opponent_team_points

-- Final deficit
def final_deficit := initial_deficit - net_quarter_scoring

theorem final_game_deficit : final_deficit = 12 := by
  sorry

end final_game_deficit_l2330_233019


namespace triangle_altitude_l2330_233018

theorem triangle_altitude
  (base : ℝ) (height : ℝ) (side : ℝ)
  (h_base : base = 6)
  (h_side : side = 6)
  (area_triangle : ℝ) (area_square : ℝ)
  (h_area_square : area_square = side ^ 2)
  (h_area_equal : area_triangle = area_square)
  (h_area_triangle : area_triangle = (base * height) / 2) :
  height = 12 := 
by
  sorry

end triangle_altitude_l2330_233018


namespace bus_ride_difference_l2330_233068

theorem bus_ride_difference (vince_bus_length zachary_bus_length : Real)
    (h_vince : vince_bus_length = 0.62)
    (h_zachary : zachary_bus_length = 0.5) :
    vince_bus_length - zachary_bus_length = 0.12 :=
by
  sorry

end bus_ride_difference_l2330_233068


namespace amount_of_salmon_sold_first_week_l2330_233009

-- Define the conditions
def fish_sold_in_two_weeks (x : ℝ) := x + 3 * x = 200

-- Define the theorem we want to prove
theorem amount_of_salmon_sold_first_week (x : ℝ) (h : fish_sold_in_two_weeks x) : x = 50 :=
by
  sorry

end amount_of_salmon_sold_first_week_l2330_233009


namespace slope_of_line_l2330_233007

noncomputable def line_equation (x y : ℝ) : Prop := 4 * y + 2 * x = 10

theorem slope_of_line (x y : ℝ) (h : line_equation x y) : -1 / 2 = -1 / 2 :=
by
  sorry

end slope_of_line_l2330_233007


namespace arrangement_count_l2330_233091

def numArrangements : Nat := 15000

theorem arrangement_count (students events : ℕ) (nA nB : ℕ) 
  (A_ne_B : nA ≠ nB) 
  (all_students : students = 7) 
  (all_events : events = 5) 
  (one_event_per_student : ∀ (e : ℕ), e < events → ∃ s, s < students ∧ (∀ (s' : ℕ), s' < students → s' ≠ s → e ≠ s')) :
  numArrangements = 15000 := 
sorry

end arrangement_count_l2330_233091


namespace find_x_l2330_233038

variable (x : ℤ)
def A : Set ℤ := {x^2, x + 1, -3}
def B : Set ℤ := {x - 5, 2 * x - 1, x^2 + 1}

theorem find_x (h : A x ∩ B x = {-3}) : x = -1 :=
sorry

end find_x_l2330_233038
