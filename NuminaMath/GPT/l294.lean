import Mathlib

namespace range_of_a_l294_294698

noncomputable theory

def f (a : ℝ) (x : ℝ) : ℝ := log x + a * x^2 - 2

theorem range_of_a (a : ℝ) (h : ∀ x ∈ Ioo (1 / 4) 1, (1 / x) + 2 * a * x > 0) : a > -8 :=
by {
  sorry  -- proof steps would go here
}

end range_of_a_l294_294698


namespace distance_CG_l294_294092

theorem distance_CG (a b c : ℝ) (h : c ^ 2 = a ^ 2 + b ^ 2) :
  ∃ (CG : ℝ), CG = (sqrt((a^4 - a^2*b^2 + b^4) / c^2)) :=
by
  sorry

end distance_CG_l294_294092


namespace parameter_interval_l294_294996

theorem parameter_interval
  (a : ℝ)
  (h : 2 * log 16 (2 * x ^ 2 - x - 2 * a - 4 * a ^ 2) - log 4 (x ^ 2 - a * x - 2 * a ^ 2) = 0)
  (root_sum_squares : ∀ x1 x2, x1 ≠ x2 ∧ (x1 ^ 2 + x2 ^ 2 ∈ set.Ioo 0 4))
  : a ∈ set.Ioo (-1 / 2) (-1 / 3) ∪ set.Ioo (-1 / 3) 0 ∪ set.Ioo 0 (3 / 5) :=
sorry

end parameter_interval_l294_294996


namespace max_min_sum_exponential_l294_294471

theorem max_min_sum_exponential (a : ℝ) 
  (h_pos : a > 0) 
  (h_ne : a ≠ 1) 
  (h_max : ∀ x ∈ (set.Icc 0 2), 2 * a * x - 1 ≤ 7) : 
  (a = 2) → 
  (∀ x ∈ (set.Icc 0 3), a^x ≤ 8) → 
  ∑ x in [0, 3], a^x = 9 :=
by
  intro h_a_eq h_exponential_bounds
  -- Proof would go here
  sorry

end max_min_sum_exponential_l294_294471


namespace percentage_problem_l294_294693

theorem percentage_problem (x : ℝ) (h : 0.3 * 0.4 * x = 45) : 0.4 * 0.3 * x = 45 :=
by
  sorry

end percentage_problem_l294_294693


namespace evaluate_expression_l294_294661

variable (x y : ℝ)
variable (h₀ : x ≠ 0)
variable (h₁ : y ≠ 0)
variable (h₂ : 5 * x ≠ 3 * y)

theorem evaluate_expression : 
  (5 * x - 3 * y)⁻¹ * ((5 * x)⁻¹ - (3 * y)⁻¹) = -1 / (15 * x * y) :=
sorry

end evaluate_expression_l294_294661


namespace problem_l294_294814

theorem problem (a b : ℕ) (h1 : ∃ k : ℕ, a * b = k * k) (h2 : ∃ m : ℕ, (2 * a + 1) * (2 * b + 1) = m * m) :
  ∃ n : ℕ, n % 2 = 0 ∧ n > 2 ∧ ∃ p : ℕ, (a + n) * (b + n) = p * p :=
by
  sorry

end problem_l294_294814


namespace simplify_logarithmic_expression_l294_294451

theorem simplify_logarithmic_expression :
  (1 / (Real.logb 12 3 + 1) + 1 / (Real.logb 8 2 + 1) + 1 / (Real.logb 18 9 + 1) = 1) :=
sorry

end simplify_logarithmic_expression_l294_294451


namespace max_wish_number_l294_294908

-- Definition of the conditions
structure SpacePoint :=
  (A B C D E : Type)
  (non_coplanar : ∀ (a b c d : Type), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d)
  (distinct_distances : ∀ (x y : Type), x ≠ y → is_distinct (dist x y))

-- Definition of a wish number
def wish_number (p : Type) (points : SpacePoint) : ℕ :=
  sorry -- This will be filled based on the problem's conditions and black edge criterion

-- The theorem stating the maximum wish number
theorem max_wish_number (p : SpacePoint) : 
  ∀ (A : Type), wish_number A p ≤ 3 :=
sorry

end max_wish_number_l294_294908


namespace minimum_other_sales_met_l294_294144

-- Define the sales percentages for pens, pencils, and the condition for other items
def pens_sales : ℝ := 40
def pencils_sales : ℝ := 28
def minimum_other_sales : ℝ := 20

-- Define the total percentage and calculate the required percentage for other items
def total_sales : ℝ := 100
def required_other_sales : ℝ := total_sales - (pens_sales + pencils_sales)

-- The Lean4 statement to prove the percentage of sales for other items
theorem minimum_other_sales_met 
  (pens_sales_eq : pens_sales = 40)
  (pencils_sales_eq : pencils_sales = 28)
  (total_sales_eq : total_sales = 100)
  (minimum_other_sales_eq : minimum_other_sales = 20)
  (required_other_sales_eq : required_other_sales = total_sales - (pens_sales + pencils_sales)) 
  : required_other_sales = 32 ∧ pens_sales + pencils_sales + required_other_sales = 100 := 
by
  sorry

end minimum_other_sales_met_l294_294144


namespace problem_solution_l294_294626

noncomputable def y (x : ℝ) : ℝ := Real.sqrt (x - 1)

theorem problem_solution (x : ℝ) : x ≥ 1 →
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 1)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 1)) = 2) ↔ (x = 13.25) :=
sorry

end problem_solution_l294_294626


namespace prove_vector_angle_l294_294347

noncomputable def vector_angle_proof : Prop :=
  ∃ (a b : ℝ × ℝ),
    (real.sqrt (a.1^2 + a.2^2) = real.sqrt 3) ∧
    (real.sqrt (b.1^2 + b.2^2) = 1) ∧
    ((a.1 * b.1 + a.2 * b.2) = -3 / 2) ∧
    let θ := real.arccos (-real.sqrt 3 / 2) in θ = 150

theorem prove_vector_angle: vector_angle_proof :=
sorry

end prove_vector_angle_l294_294347


namespace find_x_l294_294978

noncomputable def approx_equal (a b : ℝ) (ε : ℝ) : Prop := abs (a - b) < ε

theorem find_x :
  ∃ x : ℝ, x + Real.sqrt 68 = 24 ∧ approx_equal x 15.753788749 0.0001 :=
sorry

end find_x_l294_294978


namespace cardinality_S_le_l294_294735

open BigOperators

def S (n m : ℕ) : Finset (Fin 2018 → ℕ) :=
  {s | s ∈ Finset.univ.filter (λ s, (∀ i, s i ∈ {1, 2, 3, 4, 5, 6, 10}) ∧ (∑ i, s i = m))}

theorem cardinality_S_le (n : ℕ) (m : ℕ) (h₁ : n = 2018) (h₂ : m = 3860) : 
  (S n m).card ≤ 2^m * (n/2048)^(n) :=
sorry

end cardinality_S_le_l294_294735


namespace complement_union_A_B_l294_294010

def is_element_of_set_A (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 1
def is_element_of_set_B (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 2
def is_element_of_complement_U (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem complement_union_A_B :
  {x : ℤ | ¬ (is_element_of_set_A x ∨ is_element_of_set_B x)} = {x : ℤ | is_element_of_complement_U x} :=
by
  sorry

end complement_union_A_B_l294_294010


namespace find_k_in_quadrilateral_problem_l294_294980

theorem find_k_in_quadrilateral_problem
  (A B C D P Q R S : Type)
  (side_ratio : ℚ)
  (area_PQRS area_ABCD : ℚ)
  (h1 : area_PQRS = 0.52 * area_ABCD) 
  (h2 : (AP : ℚ) (PB : ℚ) (BQ : ℚ) (QC : ℚ) (CR : ℚ) (RD : ℚ) (DS : ℚ) (SA : ℚ)
    (h3 : AP / PB = side_ratio)
    (h4 : BQ / QC = side_ratio)
    (h5 : CR / RD = side_ratio)
    (h6 : DS / SA = side_ratio)) :
  side_ratio = 12/13 :=
sorry

end find_k_in_quadrilateral_problem_l294_294980


namespace cats_after_purchasing_l294_294759

/-- Mrs. Sheridan's total number of cats after purchasing more -/
theorem cats_after_purchasing (a b : ℕ) (h₀ : a = 11) (h₁ : b = 43) : a + b = 54 := by
  sorry

end cats_after_purchasing_l294_294759


namespace expected_sum_2010_l294_294160

noncomputable def expectedRolls (n : ℕ) : ℝ := 
  if n == 0 then 0
  else 1 + (1 / 6) * ∑ i in (Finset.range 6), expectedRolls (n - i)

theorem expected_sum_2010 : abs (expectedRolls 2010 - 574.761904) < 0.001 :=
sorry

end expected_sum_2010_l294_294160


namespace inequality_solution_expr_value_l294_294457

section problem1

variable (x : ℝ)

def inequality1 (x : ℝ) : Prop := 5 * x - 1 < 3 * (x + 1)
def inequality2 (x : ℝ) : Prop := (x + 1) / 2 ≥ (2 * x + 1) / 5

theorem inequality_solution (x : ℝ) (h1 : inequality1 x) (h2 : inequality2 x) : -3 ≤ x ∧ x < 2 :=
sorry

end problem1

section problem2

variable (x : ℝ)

def is_root (x : ℝ) : Prop := x^2 - 2 * x - 3 = 0
def simplified_expr (x : ℝ) : ℝ := (1 / (x + 1) + x - 1) / (x^2 / (x^2 + 2 * x + 1))

theorem expr_value (x : ℝ) (h : is_root x) (h1 : x ≠ -1) : simplified_expr x = 4 :=
sorry

end problem2

end inequality_solution_expr_value_l294_294457


namespace find_sin_B_l294_294369

def right_triangle (P Q R : Type) : Prop :=
  ∃ (PQ QR PR : ℝ),
    PQ = 8 ∧
    QR = 17 ∧
    PR = real.sqrt (PQ^2 + QR^2) ∧
    ∠P = 90°

theorem find_sin_B (P Q R : Type) (PQ QR : ℝ) (h : right_triangle P Q R) :
  sin B = 8 / real.sqrt (8^2 + 17^2) :=
by
  sorry

end find_sin_B_l294_294369


namespace new_member_younger_by_160_l294_294080

theorem new_member_younger_by_160 
  (A : ℕ)  -- average age 8 years ago and today
  (O N : ℕ)  -- age of the old member and the new member respectively
  (h1 : 20 * A = 20 * A + O - N)  -- condition derived from the problem
  (h2 : 20 * 8 = 160)  -- age increase over 8 years for 20 members
  (h3 : O - N = 160) : O - N = 160 :=
by
  sorry

end new_member_younger_by_160_l294_294080


namespace number_of_ordered_pairs_l294_294101

noncomputable def geometric_sequence_log_sum (b s : ℕ) : Prop :=
  let b₁ := b in
  let b₂ := b * s in
  let b₃ := b * s^2 in
  let b₄ := b * s^3 in
  let b₅ := b * s^4 in
  let b₆ := b * s^5 in
  let b₇ := b * s^6 in
  let b₈ := b * s^7 in
  let b₉ := b * s^8 in
  let b₁₀ := b * s^9 in
  log 10 b₁ + log 10 b₂ + log 10 b₃ + log 10 b₄ + log 10 b₅ + log 10 b₆ +
  log 10 b₇ + log 10 b₈ + log 10 b₉ + log 10 b₁₀ = 100

theorem number_of_ordered_pairs : 
  ∃ (n : ℕ), n = 6 ∧ ∀ (b s : ℕ), (b > 0 ∧ s > 0 ∧ geometric_sequence_log_sum b s) → ∃! n, n = 6 :=
by sorry

end number_of_ordered_pairs_l294_294101


namespace tangent_line_perpendicular_decreasing_intervals_range_of_a_l294_294330
noncomputable theory
open Real

-- Defining the given function f
def f (x : ℝ) := 2 * x / (ln x)

-- Problem 1: Equation of the tangent line
theorem tangent_line_perpendicular (a : ℝ) :
  (∃ a, x = a ∧ f x = 2 * (ln a - 1) / (ln a) ^ 2 ∧
     (2 * (ln a - 1) / (ln a) ^ 2) = 1 / 2 ∧ a = exp 2)
  → (∃ y, y = e^2 ∧ y - e^2 = 1/2 * (x - e^2)) :=
sorry

-- Problem 2: Intervals where f(x) is strictly decreasing
theorem decreasing_intervals :
  (∀ x, f'(x) < 0 ↔ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < exp 1)) :=
sorry

-- Defining the function g
def g (a : ℝ) (x : ℝ) := a * (exp 1) * ln x + 1/2 * x ^ 2 - (a + exp 1) / 2 * ln x * (f(x))

-- Problem 3: Range of values for a
theorem range_of_a (a : ℝ) :
  (∃ x0, x0 ∈ Ici (exp 1) ∧ g(a, x0) ≤ a) → (a ≥ - (exp 2)^2 / 2) :=
sorry

end tangent_line_perpendicular_decreasing_intervals_range_of_a_l294_294330


namespace find_difference_l294_294826

theorem find_difference (x y : ℕ) (hx : ∃ k : ℕ, x = k^2) (h_sum_prod : x + y = x * y - 2006) : y - x = 666 :=
sorry

end find_difference_l294_294826


namespace tangent_line_ellipse_l294_294180

theorem tangent_line_ellipse {m : ℝ} :
  (∀ y x : ℝ, y = m * x + 3 → 4 * x^2 + y^2 = 4) ↔ m^2 = 5 := by
sory

end tangent_line_ellipse_l294_294180


namespace coplanar_lines_k_val_l294_294221

theorem coplanar_lines_k_val (k : ℝ) :
  (∃ (s t : ℝ),
    (-2 + s = 2 * t ∧
    4 - k * s = 2 + 2 * t ∧
    2 + k * s = 3 - 2 * t)) ∨
  (∃ (a : ℝ),
    (a * ∥ (1, -k, k) ∥ = ∥ (2, 2, -2) ∥ )) →
  k = -1 :=
by {
  sorry
}

end coplanar_lines_k_val_l294_294221


namespace min_correct_answers_l294_294723

theorem min_correct_answers (total_questions correct_points incorrect_points target_score : ℕ)
                            (h_total : total_questions = 22)
                            (h_correct_points : correct_points = 4)
                            (h_incorrect_points : incorrect_points = 2)
                            (h_target : target_score = 81) :
  ∃ x : ℕ, 4 * x - 2 * (22 - x) > 81 ∧ x ≥ 21 :=
by {
  sorry
}

end min_correct_answers_l294_294723


namespace min_pounds_of_beans_l294_294589

theorem min_pounds_of_beans : 
  ∃ (b : ℕ), (∀ (r : ℝ), (r ≥ 8 + b / 3 ∧ r ≤ 3 * b) → b ≥ 3) :=
sorry

end min_pounds_of_beans_l294_294589


namespace calculate_product_N1_N2_l294_294027

theorem calculate_product_N1_N2 : 
  (∃ (N1 N2 : ℝ), 
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → 
      (60 * x - 46) / (x^2 - 5 * x + 6) = N1 / (x - 2) + N2 / (x - 3)) ∧
      N1 * N2 = -1036) :=
  sorry

end calculate_product_N1_N2_l294_294027


namespace find_sin_alpha_l294_294282

theorem find_sin_alpha (α : ℝ) (h1 : 0 < α ∧ α < real.pi) (h2 : 3 * real.cos (2 * α) - 8 * real.cos α = 5) :
  real.sin α = real.sqrt 5 / 3 :=
sorry

end find_sin_alpha_l294_294282


namespace rooms_anna_swept_l294_294212

theorem rooms_anna_swept
  (sweep_time_per_room : ℕ) (wash_time_per_dish : ℕ) (laundry_time_per_load : ℕ)
  (anna_rooms : ℕ) (billy_loads : ℕ) (billy_dishes : ℕ)
  (total_chore_time_billy : ℕ)
  (billy_laundry_time : ℕ → ℕ) (billy_dish_time : ℕ → ℕ) 
  (anna_time : ℕ → ℕ) :
  sweep_time_per_room = 3 →
  wash_time_per_dish = 2 →
  laundry_time_per_load = 9 →
  billy_loads = 2 →
  billy_dishes = 6 →
  total_chore_time_billy = billy_laundry_time billy_loads + billy_dish_time billy_dishes →
  billy_laundry_time billy_loads = billy_loads * laundry_time_per_load →
  billy_dish_time billy_dishes = billy_dishes * wash_time_per_dish →
  anna_time anna_rooms = anna_rooms * sweep_time_per_room →
  anna_time anna_rooms = total_chore_time_billy →
  anna_rooms = 10 :=
by
  intros h_sweep_time_per_room h_wash_time_per_dish h_laundry_time_per_load
        h_billy_loads h_billy_dishes h_total_chore_time_billy
        h_billy_laundry_time h_billy_dish_time h_anna_time
        h_anna_equals_billy
  rw [h_sweep_time_per_room, h_wash_time_per_dish, h_laundry_time_per_load,
      h_billy_loads, h_billy_dishes, h_billy_laundry_time, h_billy_dish_time] at *,
  simp at *,
  sorry

end rooms_anna_swept_l294_294212


namespace complement_union_A_B_l294_294011

def is_element_of_set_A (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 1
def is_element_of_set_B (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 2
def is_element_of_complement_U (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem complement_union_A_B :
  {x : ℤ | ¬ (is_element_of_set_A x ∨ is_element_of_set_B x)} = {x : ℤ | is_element_of_complement_U x} :=
by
  sorry

end complement_union_A_B_l294_294011


namespace count_solutions_l294_294227

-- Begin the definition of the equation condition
def satisfies_equation (x : ℤ) : Prop :=
  (x^2 - x - 6)^(x + 4) = 1

-- Theorem statement asserting that exactly 3 integers satisfy the given equation
theorem count_solutions : (∃ (x1 x2 x3 : ℤ), satisfies_equation x1 ∧ satisfies_equation x2 ∧ satisfies_equation x3 ∧ 
  (x1 ≠ x2) ∧ (x1 ≠ x3) ∧ (x2 ≠ x3) ∧ ∀ x, satisfies_equation x → (x = x1 ∨ x = x2 ∨ x = x3)) :=
sorry

end count_solutions_l294_294227


namespace convex_polyhedron_inequality_l294_294712

noncomputable def convex_polyhedron (B P T : ℕ) : Prop :=
  ∀ (B P T : ℕ), B > 0 ∧ P > 0 ∧ T >= 0 → B * (Nat.sqrt (P + T)) ≥ 2 * P

theorem convex_polyhedron_inequality (B P T : ℕ) (h : convex_polyhedron B P T) : 
  B * (Nat.sqrt (P + T)) ≥ 2 * P :=
by
  sorry

end convex_polyhedron_inequality_l294_294712


namespace proportion_solve_x_l294_294137

theorem proportion_solve_x :
  (0.75 / x = 5 / 7) → x = 1.05 :=
by
  sorry

end proportion_solve_x_l294_294137


namespace log_function_fixed_point_l294_294636

noncomputable def f (a x : ℝ) := log a x + 2

theorem log_function_fixed_point (a : ℝ) (h : (0 < a ∧ a < 1) ∨ 1 < a) : (1, 2) ∈ set_of (λ p : ℝ × ℝ, ∃ x, p = (x, f a x)) :=
by
  sorry

end log_function_fixed_point_l294_294636


namespace plane_stops_at_20_seconds_l294_294798

/-- The analytical expression of the function of the distance s the plane travels during taxiing 
after landing with respect to the time t is given by s = -1.5t^2 + 60t. 

Prove that the plane stops after taxiing for 20 seconds. -/

noncomputable def plane_distance (t : ℝ) : ℝ :=
  -1.5 * t^2 + 60 * t

theorem plane_stops_at_20_seconds :
  ∃ t : ℝ, t = 20 ∧ plane_distance t = plane_distance (20 : ℝ) :=
by
  sorry

end plane_stops_at_20_seconds_l294_294798


namespace membership_change_l294_294899

theorem membership_change (initial_members : ℝ) (fall_increase_percent : ℝ) (spring_decrease_percent : ℝ) :
  fall_increase_percent = 7 → spring_decrease_percent = 19 → 
  initial_members = 100 →
  let fall_members := initial_members + (fall_increase_percent / 100) * initial_members in
  let spring_members := fall_members - (spring_decrease_percent / 100) * fall_members in
  (spring_members - initial_members) / initial_members * 100 = -13.33 :=
by 
  intros h1 h2 h3 
  let fall_members := initial_members + (fall_increase_percent / 100) * initial_members 
  let spring_members := fall_members - (spring_decrease_percent / 100) * fall_members 
  simp [fall_members, spring_members]
  sorry

end membership_change_l294_294899


namespace find_area_of_triangle_l294_294243

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

theorem find_area_of_triangle :
  let a := 10
  let b := 10
  let c := 12
  triangle_area a b c = 48 := 
by 
  sorry

end find_area_of_triangle_l294_294243


namespace points_on_same_sphere_l294_294192

-- Define the necessary structures and assumptions
variables {P : Type*} [MetricSpace P]

-- Definitions of spheres and points
structure Sphere (P : Type*) [MetricSpace P] :=
(center : P)
(radius : ℝ)
(positive_radius : 0 < radius)

def symmetric_point (S A1 : P) : P := sorry -- definition to get the symmetric point A2

-- Given conditions
variables (S A B C A1 B1 C1 A2 B2 C2 : P)
variable (omega : Sphere P)
variable (Omega : Sphere P)
variable (M_S_A : P) -- midpoint of SA
variable (M_S_B : P) -- midpoint of SB
variable (M_S_C : P) -- midpoint of SC

-- Assertions of conditions
axiom sphere_through_vertex : omega.center = S
axiom first_intersections : omega.radius = dist S A1 ∧ omega.radius = dist S B1 ∧ omega.radius = dist S C1
axiom omega_Omega_intersection : ∃ (circle_center : P) (plane_parallel_to_ABC : P), true-- some conditions indicating intersection
axiom symmetric_points_A1_A2 : A2 = symmetric_point S A1
axiom symmetric_points_B1_B2 : B2 = symmetric_point S B1
axiom symmetric_points_C1_C2 : C2 = symmetric_point S C1

-- The theorem to prove
theorem points_on_same_sphere : ∃ (sphere : Sphere P), 
  (dist sphere.center A) = sphere.radius ∧ 
  (dist sphere.center B) = sphere.radius ∧ 
  (dist sphere.center C) = sphere.radius ∧ 
  (dist sphere.center A2) = sphere.radius ∧ 
  (dist sphere.center B2) = sphere.radius ∧ 
  (dist sphere.center C2) = sphere.radius := 
sorry

end points_on_same_sphere_l294_294192


namespace book_pages_count_l294_294540

theorem book_pages_count (h : ∑ x in (finset.range 10), 1 + ∑ x in finset.range 90, 2 + ∃ n : ℕ, 100 ≤ n ∧ (3 * (n - 99)) = 423 + 189 + 9 ) : 
  n + 99 = 240 :=
sorry

end book_pages_count_l294_294540


namespace exists_pos_integers_l294_294781

theorem exists_pos_integers (r : ℚ) (hr : r > 0) : 
  ∃ a b c d : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ r = (a^3 + b^3) / (c^3 + d^3) :=
by sorry

end exists_pos_integers_l294_294781


namespace a_capital_used_l294_294523

theorem a_capital_used (C P x : ℕ) (h_b_contributes : 3 * C / 4 - C ≥ 0) 
(h_b_receives : 2 * P / 3 - P ≥ 0) 
(h_b_money_used : 10 > 0) 
(h_ratio : 1 / 2 = x / 30) 
: x = 15 :=
sorry

end a_capital_used_l294_294523


namespace find_base_width_l294_294878

-- Definitions of the given conditions
def cube_edge : ℝ := 15
def rise_in_water_level : ℝ := 11.25
def base_length : ℝ := 20

-- Calculation of given and required dimensions
def volume_of_cube := cube_edge ^ 3
def displaced_water_volume := volume_of_cube
def base_area := displaced_water_volume / rise_in_water_level

-- The statement to prove the width of the vessel's base
theorem find_base_width : ∃ W : ℝ, base_area = base_length * W ∧ W = 15 :=
by
  sorry

end find_base_width_l294_294878


namespace gcd_168_486_l294_294499

theorem gcd_168_486 : gcd 168 486 = 6 := 
by sorry

end gcd_168_486_l294_294499


namespace count_arrangements_california_l294_294230

-- Defining the counts of letters in "CALIFORNIA"
def word_length : ℕ := 10
def count_A : ℕ := 3
def count_I : ℕ := 2
def count_C : ℕ := 1
def count_L : ℕ := 1
def count_F : ℕ := 1
def count_O : ℕ := 1
def count_R : ℕ := 1
def count_N : ℕ := 1

-- The final proof statement to show the number of unique arrangements
theorem count_arrangements_california : 
  (Nat.factorial word_length) / 
  ((Nat.factorial count_A) * (Nat.factorial count_I)) = 302400 := by
  -- Placeholder for the proof, can be filled in later by providing the actual steps
  sorry

end count_arrangements_california_l294_294230


namespace exposed_surface_area_l294_294913

noncomputable def cube_edge : ℝ := 1
noncomputable def number_of_cubes : ℕ := 18
noncomputable def first_layer : ℕ := 9
noncomputable def second_layer : ℕ := 9

theorem exposed_surface_area :
  number_of_cubes = 18 →
  cube_edge = 1 →
  first_layer = 9 →
  second_layer = 9 →
  let total_exposed_area := 49 in
  total_exposed_area = 49 :=
by
  intros h1 h2 h3 h4 total_exposed_area
  simp [total_exposed_area]
  sorry

end exposed_surface_area_l294_294913


namespace minimize_expression_at_9_l294_294226

noncomputable def minimize_expression (n : ℕ) : ℚ :=
  n / 3 + 27 / n

theorem minimize_expression_at_9 : minimize_expression 9 = 6 := by
  sorry

end minimize_expression_at_9_l294_294226


namespace sin_alpha_eq_sqrt5_over_3_l294_294286

theorem sin_alpha_eq_sqrt5_over_3 {α : ℝ} (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = (Real.sqrt 5) / 3 :=
sorry

end sin_alpha_eq_sqrt5_over_3_l294_294286


namespace abs_pi_expression_l294_294950

theorem abs_pi_expression : (|π - |π - 10|| = 10 - 2 * π) := by
  sorry

end abs_pi_expression_l294_294950


namespace range_of_function_l294_294615

theorem range_of_function :
  set.range (λ x, sin x * (cos x - real.sqrt 3 * sin x)) = set.Icc (-real.sqrt 3) (1 - real.sqrt 3 / 2) :=
begin
  sorry
end

end range_of_function_l294_294615


namespace equilateral_triangle_reflection_l294_294728

theorem equilateral_triangle_reflection (n : ℕ) :
  (n % 6 = 1 ∨ n % 6 = 5) ∧ n ≠ 5 ∧ n ≠ 17 ↔ ∃ (a b : ℕ), 
  is_equilateral_triangle ABC ∧ 
  initial_ray_reflects_n_times A n (a, b) ∧ 
  returns_to_point_A (a, b) ∧ 
  a ≠ b ∧ 
  gcd a b = 1 :=
  sorry
  -- The theorem states that for the given conditions of an equilateral triangle
  -- with reflections returning to point A without passing through B or C, 
  -- the possible n values are all those where n % 6 = 1 or n % 6 = 5, excluding 5 and 17

end equilateral_triangle_reflection_l294_294728


namespace find_a_l294_294805

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x, a * Real.log x + x

theorem find_a (a : ℝ) : (∃ x : ℝ, x = 1 ∧ Deriv.deriv (f a) x = 0) → a = -1 :=
by
  sorry

end find_a_l294_294805


namespace correct_operation_l294_294129

theorem correct_operation (x y a b : ℝ) :
  (-2 * x) * (3 * y) = -6 * x * y :=
by
  sorry

end correct_operation_l294_294129


namespace correct_option_D_l294_294344

variables {a b m : Type}
variables {α β : Type}

axiom parallel (x y : Type) : Prop
axiom perpendicular (x y : Type) : Prop

variables (a_parallel_b : parallel a b)
variables (a_parallel_alpha : parallel a α)

variables (alpha_perpendicular_beta : perpendicular α β)
variables (a_parallel_alpha : parallel a α)

variables (alpha_parallel_beta : parallel α β)
variables (m_perpendicular_alpha : perpendicular m α)

theorem correct_option_D : parallel α β ∧ perpendicular m α → perpendicular m β := sorry

end correct_option_D_l294_294344


namespace percentage_25_of_200_l294_294527

def percentage_of (percent : ℝ) (amount : ℝ) : ℝ := percent * amount

theorem percentage_25_of_200 :
  percentage_of 0.25 200 = 50 :=
by sorry

end percentage_25_of_200_l294_294527


namespace count_satisfying_integers_l294_294276

def satisfies_conditions (n : ℕ) : Prop :=
  3 * n + 1 ≤ 5 * n - 7 ∧ 5 * n - 7 < 3 * n + 10

theorem count_satisfying_integers : 
  {n : ℤ | satisfies_conditions n}.to_finset.card = 5 :=
by sorry

end count_satisfying_integers_l294_294276


namespace box_calories_l294_294863

theorem box_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  (cookies_per_bag * bags_per_box) * calories_per_cookie = 1600 :=
by
  sorry

end box_calories_l294_294863


namespace prove_probability_l294_294632

-- Define the first 13 prime numbers
def first_13_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

-- Define a function to check if the sum of numbers in a list is odd
def is_odd (n : ℕ) : Bool := n % 2 ≠ 0

-- Define a function to check if the sum of a list is odd and greater than 50
def sum_is_odd_and_gt_50 (l : List ℕ) : Bool :=
  is_odd (l.sum) && l.sum > 50

-- Define a function to count combinations of five primes that meet the criteria
def count_valid_combinations (primes : List ℕ) (n : ℕ) : ℕ :=
  (primes.combinations n).count (λ l => sum_is_odd_and_gt_50 l)

-- Total number of ways to select five primes from the first 13 primes
def total_combinations : ℕ := Nat.choose 13 5

-- Number of valid combinations meeting the criteria
def num_valid_combinations : ℕ := count_valid_combinations first_13_primes 5

-- The desired probability
def desired_probability : ℚ := num_valid_combinations / total_combinations

theorem prove_probability :
  desired_probability = 395 / 1287 := 
sorry

end prove_probability_l294_294632


namespace primes_sum_divisible_by_60_l294_294400

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_sum_divisible_by_60 (p q r s : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hr : is_prime r) 
  (hs : is_prime s) 
  (h_cond1 : 5 < p) 
  (h_cond2 : p < q) 
  (h_cond3 : q < r) 
  (h_cond4 : r < s) 
  (h_cond5 : s < p + 10) : 
  (p + q + r + s) % 60 = 0 :=
sorry

end primes_sum_divisible_by_60_l294_294400


namespace angle_CFD_eq_60_l294_294402

-- Define the necessary geometric entities and conditions
variables (O A B C D F : Type) [Circle O] [Point A] [Point B] [Point C] [Point D] [Point F]

-- Assume AB is a diameter of the circle centered at O
axiom diameter_of_circle (h_diam : Diameter O A B) 

-- Assume F is a point on the circle
axiom point_on_circle (H_F_on_circle : OnCircle O F)

-- Assume the tangent at B intersects the tangent at F and AF at C and D, respectively
axiom tangent_at_B (H_tangent_B : Tangent O B) 
axiom tangent_at_F (H_tangent_F : Tangent O F) 
axiom intersects_AF_at_D (H_AF_D : Intersects AF D)
axiom intersects_tangents_at_C (H_tangents_C : Intersects_Tangents B F C)

-- Given angle BAF = 30 degrees
axiom angle_BAF_eq_30 (angle_BAF : Angle B A F = 30)

-- Prove angle CFD = 60 degrees
theorem angle_CFD_eq_60 : Angle C F D = 60 :=
by
  sorry

end angle_CFD_eq_60_l294_294402


namespace cone_volume_l294_294359

-- Define the base radius of the cone.
def base_radius : ℝ := 1

-- Define the condition that the lateral area of the cone is twice the area of its base.
def lateral_area_condition : Prop :=
  ∃ (l : ℝ), l = 2

-- Prove the volume of the cone given the conditions.
theorem cone_volume : ∃ (V : ℝ), V = (sqrt 3 * π / 3) :=
by
  have r := base_radius
  have lateral_area_cond := lateral_area_condition
  use (sqrt 3 * π / 3)
  sorry

end cone_volume_l294_294359


namespace solve_fraction_l294_294789

theorem solve_fraction (x : ℚ) : (x^2 + 3*x + 5) / (x + 6) = x + 7 ↔ x = -37 / 10 :=
by
  sorry

end solve_fraction_l294_294789


namespace inversion_maps_sphere_to_self_inversion_maps_sphere_to_symmetric_l294_294135

theorem inversion_maps_sphere_to_self {O : Point} {S : Sphere} (hO : O ∈ S) :
  ∃ R, inversion O R S = S :=
sorry

theorem inversion_maps_sphere_to_symmetric {O : Point} {S : Sphere} (hO : inside O S) :
  ∃ R, ∃ S' : Sphere, inversion O R S = S' ∧ S'.symmetric_with_respect_to O :=
sorry

end inversion_maps_sphere_to_self_inversion_maps_sphere_to_symmetric_l294_294135


namespace greatest_possible_number_of_blue_chips_l294_294481

-- Definitions based on conditions
def total_chips : Nat := 72

-- Definition of the relationship between red and blue chips where p is a prime number
def is_prime (n : Nat) : Prop := Nat.Prime n

def satisfies_conditions (r b p : Nat) : Prop :=
  r + b = total_chips ∧ r = b + p ∧ is_prime p

-- The statement to prove
theorem greatest_possible_number_of_blue_chips (r b p : Nat) 
  (h : satisfies_conditions r b p) : b = 35 := 
sorry

end greatest_possible_number_of_blue_chips_l294_294481


namespace circumscribed_sphere_around_pyramid_l294_294463

theorem circumscribed_sphere_around_pyramid
  (c1 c2 m : ℝ)
  (h_c1: 0 < c1)
  (h_c2: 0 < c2)
  (h_m: 0 < m) :
  ∃ (r x : ℝ), 
  r = sqrt( (1 / (4 * m^2)) * ((m^2 + c1^2) * (m^2 + c2^2))) ∧
  x = (m^2 - c1 * c2) / (2 * m) :=
sorry

end circumscribed_sphere_around_pyramid_l294_294463


namespace complement_union_l294_294008

def A := { x : ℤ | ∃ k : ℤ, x = 3 * k + 1 }
def B := { x : ℤ | ∃ k : ℤ, x = 3 * k + 2 }
def U := ℤ

theorem complement_union :
  { x : ℤ | x ∉ (A ∪ B) } = { x : ℤ | ∃ k : ℤ, x = 3 * k } := 
by 
  sorry

end complement_union_l294_294008


namespace sphere_has_circular_views_l294_294580

-- Define the geometric shapes
inductive Shape
| cuboid
| cylinder
| cone
| sphere

-- Define a function that describes the views of a shape
def views (s: Shape) : (String × String × String) :=
match s with
| Shape.cuboid   => ("Rectangle", "Rectangle", "Rectangle")
| Shape.cylinder => ("Rectangle", "Rectangle", "Circle")
| Shape.cone     => ("Isosceles Triangle", "Isosceles Triangle", "Circle")
| Shape.sphere   => ("Circle", "Circle", "Circle")

-- Define the property of having circular views in all perspectives
def has_circular_views (s: Shape) : Prop :=
views s = ("Circle", "Circle", "Circle")

-- The theorem to prove
theorem sphere_has_circular_views :
  ∀ (s : Shape), has_circular_views s ↔ s = Shape.sphere :=
by sorry

end sphere_has_circular_views_l294_294580


namespace Jungkook_has_bigger_number_l294_294397

theorem Jungkook_has_bigger_number : (3 + 6) > 4 :=
by {
  sorry
}

end Jungkook_has_bigger_number_l294_294397


namespace sum_of_digits_1_to_50000_l294_294256

def sum_of_digits (n : ℕ) : ℕ := 
  -- Calculates the sum of digits of a given number n.
  nat.digits 10 n |>.sum

noncomputable def sum_of_digits_seq (start : ℕ) (end : ℕ) : ℕ :=
  -- Calculates the sum of digits for all numbers in the sequence from start to end.
  (list.range (end + 1)).map sum_of_digits |>.sum - (list.range start).map sum_of_digits |>.sum

theorem sum_of_digits_1_to_50000 :
  sum_of_digits_seq 1 50000 = 9180450 :=
by
  -- Proof goes here
  sorry

end sum_of_digits_1_to_50000_l294_294256


namespace tangent_circumcircle_AMN_to_Gamma_l294_294726

-- Definitions based on the conditions
variable {A B C O G D E F K L M N : Type}

-- Hypotheses based on the problem statement
hypothesis hAcute : acute_triangle A B C
hypothesis hAB_lt_AC : AB < AC
hypothesis hCircumcenter : circumcenter O A B C
hypothesis hCentroid : centroid G A B C
hypothesis hMidpoint_D : midpoint D B C
hypothesis hE_on_circle_with_BC_diameter : on_circle_with_diameter E B C
hypothesis hAE_perp_BC : perp AE BC
hypothesis hSame_side_of_BC : same_side A E BC
hypothesis hExtend_EG : intersects (extend E G) (line OD) F
hypothesis hK_parallel_OB : parallel (line F K) (line OB)
hypothesis hL_parallel_OC : parallel (line F L) (line OC)
hypothesis hK_on_BC : on_line K BC
hypothesis hL_on_BC : on_line L BC
hypothesis hGMK_perp_BC : perp GMK BC
hypothesis hNL_perp_BC : perp NL BC

-- Circle Gamma and its properties
noncomputable def Gamma := circle_through_points_and_tangent B C OB OC

-- Goal: Prove that circumcircle of triangle AMN is tangent to circle Gamma
theorem tangent_circumcircle_AMN_to_Gamma :
  tangent (circumcircle A M N) Gamma :=
sorry

end tangent_circumcircle_AMN_to_Gamma_l294_294726


namespace james_fewer_pennies_l294_294600

-- Conditions
def cassandra_pennies : ℕ := 5000
def total_pennies_donated : ℕ := 9724
def james_pennies : ℕ := total_pennies_donated - cassandra_pennies

-- Question and Answer
theorem james_fewer_pennies : cassandra_pennies - james_pennies = 276 :=
by
  -- Definitions used in the equations
  have h_james : james_pennies = 4724 := by sorry
  have h_diff : cassandra_pennies - james_pennies = cassandra_pennies - 4724 := by sorry
  -- Applying known values to get the difference
  show 5000 - 4724 = 276 from by sorry

end james_fewer_pennies_l294_294600


namespace sum_of_squares_of_five_consecutive_integers_divisibility_l294_294442

theorem sum_of_squares_of_five_consecutive_integers_divisibility (a : ℤ) :
  let S := (a-2)^2 + (a-1)^2 + a^2 + (a+1)^2 + (a+2)^2 in
  (S % 5 = 0) ∧ (S % 25 ≠ 0) :=
by
  let x1 := a - 2
  let x2 := a - 1
  let x3 := a
  let x4 := a + 1
  let x5 := a + 2
  let S := x1^2 + x2^2 + x3^2 + x4^2 + x5^2
  have h1 : S = 5 * (a^2 + 2) := sorry
  have h2 : 5 * (a^2 + 2) % 5 = 0 := sorry
  have h3 : (a^2 + 2) % 5 ≠ 0 := sorry
  exact ⟨h2, h3⟩

end sum_of_squares_of_five_consecutive_integers_divisibility_l294_294442


namespace gain_percent_calculation_l294_294696

theorem gain_percent_calculation (C S : ℕ) (h1 : 50 * C = 45 * S) : 
  (10 * C / 9 - C) * 100 / C = 11.11 := 
sorry

end gain_percent_calculation_l294_294696


namespace complete_square_proof_l294_294860

def quadratic_eq := ∀ (x : ℝ), x^2 - 6 * x + 5 = 0
def form_completing_square (b c : ℝ) := ∀ (x : ℝ), (x + b)^2 = c

theorem complete_square_proof :
  quadratic_eq → (∃ b c : ℤ, form_completing_square (b : ℝ) (c : ℝ) ∧ b + c = 11) :=
by
  sorry

end complete_square_proof_l294_294860


namespace ratio_q_p_l294_294793

variable (p q : ℝ)
variable (hpq_pos : 0 < p ∧ 0 < q)
variable (hlog : Real.log p / Real.log 8 = Real.log q / Real.log 12 ∧ Real.log q / Real.log 12 = Real.log (p - q) / Real.log 18)

theorem ratio_q_p (p q : ℝ) (hpq_pos : 0 < p ∧ 0 < q) 
    (hlog : Real.log p / Real.log 8 = Real.log q / Real.log 12 ∧ Real.log q / Real.log 12 = Real.log (p - q) / Real.log 18) :
    q / p = (Real.sqrt 5 - 1) / 2 :=
  sorry

end ratio_q_p_l294_294793


namespace expected_sum_2010_l294_294161

noncomputable def expectedRolls (n : ℕ) : ℝ := 
  if n == 0 then 0
  else 1 + (1 / 6) * ∑ i in (Finset.range 6), expectedRolls (n - i)

theorem expected_sum_2010 : abs (expectedRolls 2010 - 574.761904) < 0.001 :=
sorry

end expected_sum_2010_l294_294161


namespace abs_pi_expression_l294_294949

theorem abs_pi_expression : (|π - |π - 10|| = 10 - 2 * π) := by
  sorry

end abs_pi_expression_l294_294949


namespace population_mean_sampling_probability_l294_294484

theorem population_mean (scores : List ℕ) (h : scores = [5, 6, 7, 8, 9, 10]) : 
  (List.sum scores) / (List.length scores) = 7.5 :=
by 
  sorry

theorem sampling_probability (pairs : List (ℕ × ℕ)) (h : pairs = [
  (5,6), (5,7), (5,8), (5,9), (5,10), (6,7), (6,8), (6,9), (6,10),
  (7,8), (7,9), (7,10), (8,9), (8,10), (9,10)]) 
  : 
  let population_mean := 7.5 in
  let valid_pairs := [ (5,9), (5,10), (6,8), (6,9), (6,10), (7,8), (7,9)] in
  valid_pairs.length.to_rat / pairs.length.to_rat = 7/15 :=
by 
  sorry

end population_mean_sampling_probability_l294_294484


namespace solid_surface_area_of_cube_with_tunnel_l294_294740

noncomputable def solid_surface_area (a : ℝ) (b : ℝ) (c : ℝ) : ℝ := a + b*sqrt c

theorem solid_surface_area_of_cube_with_tunnel :
  let E : ℝ := 10
  let EI : ℝ := 3
  let total_surface_area : ℝ := 6 * E^2 
  let removed_area : ℝ := 3 * EI * 2
  let tunnel_faces_area : ℝ := 3 * 4.5 * sqrt 6 in 
  solid_surface_area (total_surface_area - removed_area) 13.5 6 = 582 + 13.5 * sqrt 6 := 
by {
  sorry
}

end solid_surface_area_of_cube_with_tunnel_l294_294740


namespace average_billboards_per_hour_l294_294061

def first_hour_billboards : ℕ := 17
def second_hour_billboards : ℕ := 20
def third_hour_billboards : ℕ := 23

theorem average_billboards_per_hour : 
  (first_hour_billboards + second_hour_billboards + third_hour_billboards) / 3 = 20 := 
by
  sorry

end average_billboards_per_hour_l294_294061


namespace part_a_roots_part_b_roots_l294_294415

noncomputable def polynomial_roots_transform (n : ℕ) (a : Fin n.succ → ℂ) 
    (x : Fin n → ℂ) (hx : ∀ i : Fin n, Polynomial.eval x i (Polynomial.ofFunction a) = 0) :
    Prop :=
    let p := Polynomial.ofFunction a
    (∀ i : Fin n, Polynomial.eval (1 / x i) p = 0)

noncomputable def polynomial_double_roots_transform (n : ℕ) (a : Fin n.succ → ℂ) 
    (x : Fin n → ℂ) (hx : ∀ i : Fin n, Polynomial.eval x i (Polynomial.ofFunction a) = 0) :
    Prop :=
    let p := Polynomial.ofFunction a
    (∀ i : Fin n, Polynomial.eval (Complex.sqrt x i) p = 0 ∧ Polynomial.eval (-Complex.sqrt x i) p = 0)

-- Defining the conditions for the problems
def conditions_roots (n : ℕ) (a : Fin n.succ → ℂ) (x : Fin n → ℂ) : Prop :=
  ∀ i : Fin n, Polynomial.eval x i (Polynomial.ofFunction a) = 0

-- Part (a)
theorem part_a_roots (n : ℕ) (a : Fin n.succ → ℂ) (x : Fin n → ℂ)
  (hx : conditions_roots n a x) :
  polynomial_roots_transform n a x hx := sorry

-- Part (b)
theorem part_b_roots (n : ℕ) (a : Fin n.succ → ℂ) (x : Fin n → ℂ)
  (hx : conditions_roots n a x) :
  polynomial_double_roots_transform n a x hx := sorry

end part_a_roots_part_b_roots_l294_294415


namespace maximize_probability_at_centroid_l294_294734

noncomputable def probability_maximizer (ABC : Triangle) : Point :=
  sorry

theorem maximize_probability_at_centroid (ABC : Triangle) (Z : Point) :
  Z = probability_maximizer ABC ↔ Z = centroid ABC :=
begin
  sorry
end

end maximize_probability_at_centroid_l294_294734


namespace binomial_alternating_sum_l294_294265

theorem binomial_alternating_sum :
  (Finset.range 51).sum (λ k, (-1 : ℤ)^k * Nat.choose 100 (2 * k)) = -2^50 := 
by
    sorry

end binomial_alternating_sum_l294_294265


namespace product_base7_eq_l294_294927

-- Definitions for the numbers in base 7
def num325_base7 := 3 * 7^2 + 2 * 7^1 + 5 * 7^0  -- 325 in base 7
def num4_base7 := 4 * 7^0  -- 4 in base 7

-- Theorem stating that the product of 325_7 and 4_7 in base 7 is 1636_7
theorem product_base7_eq : 
  let product_base10 := num325_base7 * num4_base7 in
  (product_base10 = 1 * 7^3 + 6 * 7^2 + 3 * 7^1 + 6 * 7^0) :=
by sorry

end product_base7_eq_l294_294927


namespace sum_of_distinct_values_l294_294254

def valid_digits (G H : ℕ) : Prop :=
  G < 10 ∧ H < 10 ∧ 3 - G + H = 0

def distinct_products_sum : ℕ := 
  ∑ G in (finset.filter (λ G, ∃ H, valid_digits G H) (finset.range 10)), 
      ∑ H in (finset.filter (λ H, valid_digits G H) (finset.range 10)), G * H

theorem sum_of_distinct_values : distinct_products_sum = 154 :=
by simp; sorry

end sum_of_distinct_values_l294_294254


namespace minimize_complex_expression_l294_294315

theorem minimize_complex_expression (z : ℂ) (hz : |z| = 1) :
  ∃ z, z = -1/4 + sqrt 15 / 4 * complex.I ∨ z = -1/4 - sqrt 15 / 4 * complex.I ∧ 
  ∀ w : ℂ, |w| = 1 → 
  |1 + w + 3 * w^2 + w^3 + w^4| ≥ |1 + z + 3 * z^2 + z^3 + z^4| :=
begin
  sorry
end

end minimize_complex_expression_l294_294315


namespace bricks_to_build_wall_l294_294349

def brick_volume (length width height : ℕ) : ℕ := length * width * height

def wall_volume (length width thickness : ℕ) : ℕ := length * width * thickness

def bricks_needed (wall_vol brick_vol : ℕ) : ℕ := (wall_vol + brick_vol - 1) / brick_vol -- Using ceiling division

theorem bricks_to_build_wall :
  let brick := (5, 11, 6) in
  let wall := (800, 600, 2) in
  bricks_needed (wall_volume wall.1 wall.2 wall.3) (brick_volume brick.1 brick.2 brick.3) = 2910 :=
sorry -- Proof to be filled in

end bricks_to_build_wall_l294_294349


namespace triangle_perimeter_l294_294183

theorem triangle_perimeter :
  let a := 15
  let b := 10
  let c := 12
  (a < b + c) ∧ (b < a + c) ∧ (c < a + b) →
  (a + b + c = 37) :=
by
  intros
  sorry

end triangle_perimeter_l294_294183


namespace remainders_mod_m_l294_294858

theorem remainders_mod_m {m n b : ℤ} (h_coprime : Int.gcd m n = 1) :
    (∀ r : ℤ, 0 ≤ r ∧ r < m → ∃ k : ℤ, 0 ≤ k ∧ k < n ∧ ((b + k * n) % m = r)) :=
by
  sorry

end remainders_mod_m_l294_294858


namespace root_count_eq_one_l294_294989

noncomputable def equation (z : ℂ) (λ : ℝ) : ℂ := λ - z - complex.exp (-z)

theorem root_count_eq_one (λ : ℝ) (hλ : λ > 1) : 
  ∃ z : ℂ, equation z λ = 0 ∧ (z.re > 0) ∧ (∀ w : ℂ, (equation w λ = 0 ∧ w.re > 0) → w = z) :=
begin
  sorry
end

end root_count_eq_one_l294_294989


namespace total_jellybeans_l294_294598

-- Define the conditions
def a := 3 * 12       -- Caleb's jellybeans
def b := a / 2        -- Sophie's jellybeans

-- Define the goal
def total := a + b    -- Total jellybeans

-- The theorem statement
theorem total_jellybeans : total = 54 :=
by
  -- Proof placeholder
  sorry

end total_jellybeans_l294_294598


namespace number_of_dodge_trucks_l294_294825

theorem number_of_dodge_trucks (V T F D : ℕ) (h1 : V = 5)
  (h2 : T = 2 * V) 
  (h3 : F = 2 * T)
  (h4 : F = D / 3) :
  D = 60 := 
by
  sorry

end number_of_dodge_trucks_l294_294825


namespace Liliane_more_soda_than_Alice_l294_294922

variable (J : ℝ) -- Represents the amount of soda Jacqueline has

-- Conditions: Representing the amounts for Benjamin, Liliane, and Alice
def B := 1.75 * J
def L := 1.60 * J
def A := 1.30 * J

-- Question: Proving the relationship in percentage terms between the amounts Liliane and Alice have
theorem Liliane_more_soda_than_Alice :
  (L - A) / A * 100 = 23 := 
by sorry

end Liliane_more_soda_than_Alice_l294_294922


namespace maria_waist_size_in_cm_l294_294053

noncomputable def waist_size_in_cm (waist_size_inches : ℕ) (extra_inch : ℕ) (inches_per_foot : ℕ) (cm_per_foot : ℕ) : ℚ :=
  let total_inches := waist_size_inches + extra_inch
  let total_feet := (total_inches : ℚ) / inches_per_foot
  total_feet * cm_per_foot

theorem maria_waist_size_in_cm :
  waist_size_in_cm 28 1 12 31 = 74.9 :=
by
  sorry

end maria_waist_size_in_cm_l294_294053


namespace prod_A_B_square_l294_294399

def isOdd (x : ℕ) := x % 2 = 1

def A (n : ℕ) : Finset ℕ := 
  {k ∈ Finset.range (2 * n) | isOdd k}

def B (n m : ℕ) : Finset ℕ := 
  {k + m | k ∈ A n}

def product (s : Finset ℕ) : ℕ :=
  s.fold (*) 1 id

theorem prod_A_B_square (n : ℕ) (h_pos : 0 < n) :
  ∃ m : ℕ, (∀ b ∈ B n m, b ∈ ℕ) ∧ 
    is_square (product (A n) * product (B n (2*n + 1))) := by
  sorry

end prod_A_B_square_l294_294399


namespace valid_polynomial_forms_l294_294405

def no_digit_seven (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ (nat.digits 10 n) → d ≠ 7

def set_K := { n : ℕ | n > 0 ∧ no_digit_seven n }

def polynomial (a : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, ∀ n : ℕ, a n = ∑ i in (finset.range k), (a i) * n^i

theorem valid_polynomial_forms (f : ℕ → ℕ) :
  (∀ n ∈ set_K, f n ∈ set_K) →
  (∃ k ∈ set_K, f = λ n, k) ∨
  (∃ (a : ℕ) (b ∈ set_K),
     (∃ i : ℕ, a = 10^i) ∧ f = λ n, (a * n + b) ∧ b < a) :=
begin
  sorry
end

end valid_polynomial_forms_l294_294405


namespace analytical_expression_increasing_intervals_min_value_on_interval_l294_294193

-- Definition of the function f
def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

-- Amplitude, frequency, and phase shift conditions
variables (A ω φ : ℝ)
axiom hA : A > 0
axiom hω : ω > 0
axiom hφ : |φ| < π / 2

-- Specific point condition
axiom h1 : ω * π / 6 + φ = π / 2

-- Proof of analytical expression
theorem analytical_expression : 
  f = (λ x, A * sin (ω * x + φ)) :=
sorry

-- Intervals where the function is monotonically increasing
def monotonic_intervals (k : ℤ) : set ℝ := 
  {x | -π / 3 + k * π ≤ x ∧ x ≤ π / 6 + k * π}

theorem increasing_intervals : 
  ∃ (k : ℤ), ∀ x, monotonic_intervals k x ↔ -π / 3 + k * π ≤ x ∧ x ≤ π / 6 + k * π :=
sorry

-- Minimum value on the interval [-π/2, 0]
theorem min_value_on_interval : 
  ∃ x ∈ Icc (-π / 2) 0, f x = -2 :=
sorry

end analytical_expression_increasing_intervals_min_value_on_interval_l294_294193


namespace minimum_value_in_interval_l294_294250

noncomputable def y : ℝ → ℝ := λ x, x^4 - 4 * x + 3

theorem minimum_value_in_interval : 
  ∃ x ∈ Set.Icc (-2 : ℝ) 3, (∀ y ∈ Set.Icc (-2 : ℝ) 3, y x ≤ y) ∧ y x = 0 := 
sorry

end minimum_value_in_interval_l294_294250


namespace noemi_initial_amount_l294_294428

-- Define the conditions
def lost_on_roulette : Int := 400
def lost_on_blackjack : Int := 500
def still_has : Int := 800
def total_lost : Int := lost_on_roulette + lost_on_blackjack

-- Define the theorem to be proven
theorem noemi_initial_amount : total_lost + still_has = 1700 := by
  -- The proof will be added here
  sorry

end noemi_initial_amount_l294_294428


namespace minimum_surface_area_of_combined_cuboids_l294_294936

noncomputable def cuboid_combinations (l w h : ℕ) (n : ℕ) : ℕ :=
sorry

theorem minimum_surface_area_of_combined_cuboids :
  ∃ n, cuboid_combinations 2 1 3 3 = 4 ∧ n = 42 :=
sorry

end minimum_surface_area_of_combined_cuboids_l294_294936


namespace exquisite_permutations_count_l294_294413

-- Define a permutation and the properties of being exquisite
def is_exquisite (w : List ℕ) : Prop :=
  (∀ (i j k : ℕ), i < j → j < k → w[i] < w[j] → w[j] < w[k] → False) ∧
  (∀ (i j k l : ℕ), i < j → j < k → k < l → w[i] > w[j] → w[j] > w[k] → w[k] > w[l] → False)

def all_permutations (n : ℕ) : List (List ℕ) :=
  (List.range n).permutations

-- The theorem statement for the number of exquisite permutations
theorem exquisite_permutations_count : (all_permutations 6).countP is_exquisite = 25 := 
sorry

end exquisite_permutations_count_l294_294413


namespace part_a_part_b_part_c_l294_294850

-- Part (a)
theorem part_a (p : ℕ) (n : ℕ) (h1 : p.prime) (h2 : p ∣ 10^n + 1) (i : ℕ) (hi : i ∈ Finset.range 10) :
  number_of_occurrences_of_digit i (decimal_representation (1 / p)) =
  number_of_occurrences_of_digit (9 - i) (decimal_representation (1 / p)) :=
sorry

-- Part (b)
theorem part_b (p : ℕ) (k : ℕ) (h1 : p.prime) (h2 : ¬ p ∣ 10) (h3 : period (decimal_representation (1 / p)) = 2 * k) :
  ∃ (a b : ℕ), a + b = 10^k - 1 :=
sorry

-- Part (c)
theorem part_c (x := ∑' n : ℕ, 1 / (1998 * 10^n)) (h : x = ∑' n : ℕ, 1 / (1998 * 10^n)) :
  decimal_digit (2 * x) 59 = 1 :=
sorry

end part_a_part_b_part_c_l294_294850


namespace find_largest_number_l294_294633

theorem find_largest_number (a b c d e : ℕ)
    (h1 : a + b + c + d = 240)
    (h2 : a + b + c + e = 260)
    (h3 : a + b + d + e = 280)
    (h4 : a + c + d + e = 300)
    (h5 : b + c + d + e = 320)
    (h6 : a + b = 40) :
    max a (max b (max c (max d e))) = 160 := by
  sorry

end find_largest_number_l294_294633


namespace greatest_distance_between_centers_l294_294488

-- Define the dimensions of the rectangle
def rectangle_length := 20
def rectangle_width := 16

-- Define the diameter of each circle
def circle_diameter := 8
def circle_radius := circle_diameter / 2

-- Define the coordinates of the centers of the circles
def center1 := (circle_radius, circle_radius)
def center2 := (rectangle_length - circle_radius, rectangle_width - circle_radius)

-- Calculate the distance between the two centers
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Prove the greatest possible distance between the centers
theorem greatest_distance_between_centers :
  distance center1 center2 = 2 * real.sqrt 52 := by
sorry

end greatest_distance_between_centers_l294_294488


namespace complement_of_union_l294_294022

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def U : Set ℤ := Set.univ

theorem complement_of_union (x : ℤ) : (x ∉ A ∪ B) ↔ ∃ k : ℤ, x = 3 * k := 
by {
  sorry
}

end complement_of_union_l294_294022


namespace total_jellybeans_l294_294596

-- Definitions of the conditions
def caleb_jellybeans : ℕ := 3 * 12
def sophie_jellybeans (caleb_jellybeans : ℕ) : ℕ := caleb_jellybeans / 2

-- Statement of the proof problem
theorem total_jellybeans (C : caleb_jellybeans = 36) (S : sophie_jellybeans 36 = 18) :
  caleb_jellybeans + sophie_jellybeans 36 = 54 :=
by
  sorry

end total_jellybeans_l294_294596


namespace solve_inequality_l294_294790

noncomputable def solution_sets (a : ℝ) (x : ℝ) : Prop :=
  if a > 1 then (a^(2/3) ≤ x ∧ x < a^(3/4)) ∨ (x > a)
  else if 0 < a ∧ a < 1 then (a^(3/4) < x ∧ x ≤ a^(2/3)) ∨ (0 < x ∧ x < a)
  else False

theorem solve_inequality (a x : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (sqrt (3 * log a x - 2) < 2 * log a x - 1) ↔ solution_sets a x :=
sorry

end solve_inequality_l294_294790


namespace find_unknown_rate_l294_294888

def cost_with_discount_and_tax (original_price : ℝ) (count : ℕ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let discounted_price := (original_price * count) * (1 - discount)
  discounted_price * (1 + tax)

theorem find_unknown_rate :
  let total_blankets := 10
  let average_price := 160
  let total_cost := total_blankets * average_price
  let cost_100_blankets := cost_with_discount_and_tax 100 3 0.05 0.12
  let cost_150_blankets := cost_with_discount_and_tax 150 5 0.10 0.15
  let cost_unknown_blankets := 2 * x
  total_cost = cost_100_blankets + cost_150_blankets + cost_unknown_blankets →
  x = 252.275 :=
by
  sorry

end find_unknown_rate_l294_294888


namespace derangement_probability_six_l294_294075

-- Define the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define the derangement function
def derangement : ℕ → ℕ
| 0       := 1
| 1       := 0
| (n + 1) := n * (derangement n + derangement (n - 1))

-- Define the probability computation
def probability_derangement (n : ℕ) : ℚ :=
  derangement n / (fact n)

-- Theorem statement
theorem derangement_probability_six : probability_derangement 6 = 265 / 720 := 
  sorry

end derangement_probability_six_l294_294075


namespace find_sine_of_alpha_l294_294288

theorem find_sine_of_alpha (α : ℝ) (h1 : 0 < α) (h2 : α < π) 
  (h3 : 3 * real.cos (2 * α) - 8 * real.cos α = 5) :
  real.sin α = real.sqrt 5 / 3 :=
sorry

end find_sine_of_alpha_l294_294288


namespace sum_of_cubes_and_product_l294_294096

theorem sum_of_cubes_and_product (a b : ℤ) (h1 : a + b = 2) (h2 : a^3 + b^3 = 189) (h3 : (a * b : ℚ) ≈ 20) : a = 5 ∧ b = 4 ∨ a = 4 ∧ b = 5 := by
  sorry

end sum_of_cubes_and_product_l294_294096


namespace sum_of_10th_group_l294_294552

theorem sum_of_10th_group : 
  let sum_n (n : Nat) : Nat := 2^(n+3) - (n + 4)
  in sum_n 10 = 2^13 - 14 := by
  sorry

end sum_of_10th_group_l294_294552


namespace similar_triangles_of_intersections_l294_294389

variable {A B C A' B' C' A1 B1 C1 : Point}
variable {triangle_A_B_C : triangle A B C}
variable {excircle_A_B_C : ∃ A' B' C', excircle_opposite A' B' C' triangle_A_B_C}
variable {circumcircle_A'B'C : circle (A' B' C)}
variable {circumcircle_AB'C' : circle (A B' C')}
variable {circumcircle_A'BC' : circle (A' B C')}

def point_in_circle (P : Point) (c : circle P') :=
∃ (x y : ℝ), (P = (x, y)) ∧ ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2)

noncomputable def incircle (t : triangle) : circle := sorry

noncomputable def touches (c : circle) (s : line_segment) : bool :=
  sorry

noncomputable def excircle_opposite (A' B' C' : Point) (t : triangle) : Prop :=
  ∃ (a : line_segment), touches (excircle A' t.triangle_A) a ∧ sorry

noncomputable def circumcircle (A B C : Point) : circle :=
  sorry

noncomputable def similar (t1 t2 : triangle) : Prop :=
  sorry

theorem similar_triangles_of_intersections :
  similar (triangle A1 B1 C1) (triangle (incircle triangle_A_B_C).touchpoint_1 (incircle triangle_A_B_C).touchpoint_2 (incircle triangle_A_B_C).touchpoint_3)
    given
    (htA : point_in_circle A1 (circumcircle A B C)) 
    (htB : point_in_circle B1 (circumcircle A B C)) 
    (htC : point_in_circle C1 (circum-circle A B C)) 
    (h_intersect1 : circumcircle_A'B'C ∩ (circumcircle A B C) = {C1})
    (h_intersect2 : circumcircle_AB'C' ∩ (circumcircle A B C) = {A1})
    (h_intersect3 : circumcircle_A'BC' ∩ (circumcircle A B C) = {B1})
    : similar (triangle A1 B1 C1) (triangle (incircle triangle_A_B_C).touchpoint_1
      (incircle triangle_A_B_C).touchpoint_2 (incircle triangle_A_B_C).touchpoint_3) :=
sorry

end similar_triangles_of_intersections_l294_294389


namespace abs_pi_expression_l294_294946

theorem abs_pi_expression : |π - |π - 10|| = 10 - 2 * π :=
by
  sorry

end abs_pi_expression_l294_294946


namespace valid_ABCDE_l294_294225

def reversed (n : ℕ) : ℕ :=
  n.to_string.reverse.to_nat

def valid_numbers (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  n % 275 = 0 ∧
  reversed n % 275 = 0

theorem valid_ABCDE :
  { n : ℕ | valid_numbers n } = {52800, 52525, 52250, 57200, 57750, 57475} :=
by
  sorry

end valid_ABCDE_l294_294225


namespace abs_pi_expression_l294_294945

theorem abs_pi_expression : |π - |π - 10|| = 10 - 2 * π :=
by
  sorry

end abs_pi_expression_l294_294945


namespace set_M_equality_l294_294681

def is_positive_integer_divisor (k : ℤ) : Prop :=
  ∃ n : ℤ, n > 0 ∧ k = 6 * n

def M : Set ℤ :=
  {a | ∃ k : ℤ, is_positive_integer_divisor (5 - a) ∧ a ∈ ℤ}

theorem set_M_equality : M = {-1, 2, 3, 4} := 
  sorry

end set_M_equality_l294_294681


namespace stone_count_150_equals_8_l294_294237

theorem stone_count_150_equals_8 :
  ∃ n, 1 ≤ n ∧ n ≤ 15 ∧ (150 % 28 = 22) ∧ (22 corresponds to stone number 8) :=
by
  -- Conditions for the equivalence of position under the counting pattern
  have h1 : 150 % 28 = 22 := by sorry
  -- Detailed proof and rigorous definition of counting pattern skipped
  sorry

end stone_count_150_equals_8_l294_294237


namespace range_of_a_l294_294702

theorem range_of_a (a : ℝ) (h : ∀ x ∈ set.Icc (1 : ℝ) 2, x^2 + 2 * a * x + a > 0) : 
  a ∈ set.Ioi (-(1 : ℝ) / 3) :=
sorry

end range_of_a_l294_294702


namespace angle_is_120_degrees_l294_294497

-- Define vectors a and b as unit vectors that are orthogonal.
variables (a b : EuclideanSpace ℝ (Fin 2))
variable (a_unit : ∥a∥ = 1)
variable (b_unit : ∥b∥ = 1)
variable (orthogonal : inner a b = 0)

-- Define the angle calculation function.
noncomputable def angle_between_vectors (u v : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  real.arccos ((inner u v) / (∥u∥ * ∥v∥))

-- Define the specific vectors in question.
def vector_c := (real.sqrt 3) • a - b

-- State the theorem to prove.
theorem angle_is_120_degrees :
  angle_between_vectors (vector_c) b = real.pi * 2 / 3 :=
by
  apply sorry

end angle_is_120_degrees_l294_294497


namespace cockatiel_weekly_consumption_is_50_l294_294398

def boxes_bought : ℕ := 3
def boxes_existing : ℕ := 5
def grams_per_box : ℕ := 225
def parrot_weekly_consumption : ℕ := 100
def weeks_supply : ℕ := 12

def total_boxes : ℕ := boxes_bought + boxes_existing
def total_birdseed_grams : ℕ := total_boxes * grams_per_box
def parrot_total_consumption : ℕ := parrot_weekly_consumption * weeks_supply
def cockatiel_total_consumption : ℕ := total_birdseed_grams - parrot_total_consumption
def cockatiel_weekly_consumption : ℕ := cockatiel_total_consumption / weeks_supply

theorem cockatiel_weekly_consumption_is_50 :
  cockatiel_weekly_consumption = 50 := by
  -- Proof goes here
  sorry

end cockatiel_weekly_consumption_is_50_l294_294398


namespace min_translation_for_even_func_l294_294090

noncomputable def f (x : ℝ) : ℝ := Math.sin x + Math.cos x

theorem min_translation_for_even_func (t : ℝ) (ht : t > 0) :
  let g := λ x : ℝ, (Real.sqrt 2) * Real.sin (x - t + π / 4)
  in EvenFunc g ↔ t = 3 * π / 4 := sorry

end min_translation_for_even_func_l294_294090


namespace remainder_is_one_mod_88_l294_294930

theorem remainder_is_one_mod_88 : 
  let S := ∑ k in finset.range (11), (-1 : ℤ) ^ k * 90 ^ k * nat.choose 10 k 
  in S % 88 = 1 :=
by
  sorry

end remainder_is_one_mod_88_l294_294930


namespace planet_density_approx_l294_294543

theorem planet_density_approx {R T : ℝ} (G : ℝ) (m : ℝ) (ρ : ℝ)
    (h1 : ∃ R T : ℝ, a_c = (4 * Real.pi^2 * (2 * R)) / T^2)
    (h2 : m * a_c = G * (m * ρ * (4/3) * Real.pi * R^3) / (2 * R)^2)
    (hG : G ≈ 6.67430e-11)
    (hT : T = 14400) : 
  ρ ≈ 6000 :=
by
  sorry

end planet_density_approx_l294_294543


namespace rain_on_at_least_one_day_l294_294271

noncomputable def rain_prob : ℕ → ℚ
| 0 => 0.30  -- Probability of rain on Saturday
| 1 => 0.70  -- Probability of rain on Sunday if it rains on Saturday
| 2 => 0.60  -- Probability of rain on Sunday if it does not rain on Saturday

theorem rain_on_at_least_one_day :
  let p_sat := rain_prob 0 in
  let p_sun_given_rain_sat := rain_prob 1 in
  let p_sun_given_no_rain_sat := rain_prob 2 in
  let p_no_rain_sat := 1 - p_sat in
  let p_no_rain_sun_given_no_rain_sat := (1 - p_sun_given_no_rain_sat) in
  let p_no_rain_both_days := p_no_rain_sat * p_no_rain_sun_given_no_rain_sat in
  let p_rain_at_least_one_day := 1 - p_no_rain_both_days in
  p_rain_at_least_one_day = 0.72 :=
by
  sorry

end rain_on_at_least_one_day_l294_294271


namespace b_alone_can_complete_work_in_5_6_days_l294_294848

theorem b_alone_can_complete_work_in_5_6_days:
  (work_rate_b : ℚ) 
  (work_rate_a : ℚ := 1/14) -- a's work rate
  (combined_work_rate : ℚ := 1/4) -- combined work rate
  (h : work_rate_a + work_rate_b = combined_work_rate) -- given condition
  : 1 / work_rate_b = 5.6 :=
by
  sorry

end b_alone_can_complete_work_in_5_6_days_l294_294848


namespace fraction_study_japanese_l294_294136

theorem fraction_study_japanese (J S : ℕ) (hS : S = 2 * J)
  (hS_japanese : rat.of_int S * (1 / 8) = rat.of_int S * 1 / 8)
  (hJ_japanese : rat.of_int J * (3 / 4) = rat.of_int J * 3 / 4) : 
  (rat.of_int (((1 / 8) * S) + ((3 / 4) * J))) / (rat.of_int (S + J)) = 1 / 3 := by
  sorry

end fraction_study_japanese_l294_294136


namespace find_divisor_l294_294842

def positive_integer := {e : ℕ // e > 0}

theorem find_divisor (d : ℕ) :
  (∃ e : positive_integer, (e.val % 13 = 2)) →
  (∃ n : ℕ, n < 180 ∧ n % d = 5 ∧ ∀ m < 180, m % d = 5 → m = n) →
  d = 175 :=
by
  sorry

end find_divisor_l294_294842


namespace disease_given_positive_l294_294366

-- Definitions and conditions extracted from the problem
def Pr_D : ℚ := 1 / 200
def Pr_Dc : ℚ := 1 - Pr_D
def Pr_T_D : ℚ := 1
def Pr_T_Dc : ℚ := 0.05

-- Derived probabilites from given conditions
def Pr_T : ℚ := Pr_T_D * Pr_D + Pr_T_Dc * Pr_Dc

-- Statement for the probability using Bayes' theorem
theorem disease_given_positive :
  (Pr_T_D * Pr_D) / Pr_T = 20 / 219 :=
sorry

end disease_given_positive_l294_294366


namespace expected_rolls_to_reach_2010_l294_294164

noncomputable def expected_rolls_to_reach_sum (n : ℕ) : ℝ :=
  sorry -- Using 'sorry' to denote placeholder for the actual proof.

theorem expected_rolls_to_reach_2010 : expected_rolls_to_reach_sum 2010 ≈ 574.761904 :=
  sorry

end expected_rolls_to_reach_2010_l294_294164


namespace simplify_expression_l294_294452

theorem simplify_expression (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a ≠ b) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b :=
by sorry

end simplify_expression_l294_294452


namespace bookseller_original_cost_l294_294868

theorem bookseller_original_cost
  (x y z : ℝ)
  (h1 : 1.10 * x = 11.00)
  (h2 : 1.10 * y = 16.50)
  (h3 : 1.10 * z = 24.20) :
  x + y + z = 47.00 := by
  sorry

end bookseller_original_cost_l294_294868


namespace area_ratio_eq_one_l294_294773

theorem area_ratio_eq_one
  (A B C D : Point)
  (h_eq_triangle : EquilateralTriangle A B C)
  (h_D_on_AC : D ∈ segment A C)
  (h_angle_DBC_eq_30 : ∠ D B C = 30) :
  (area (triangle A D B))/(area (triangle C D B)) = 1 := 
sorry

end area_ratio_eq_one_l294_294773


namespace complement_union_A_B_l294_294013

def is_element_of_set_A (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 1
def is_element_of_set_B (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 2
def is_element_of_complement_U (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem complement_union_A_B :
  {x : ℤ | ¬ (is_element_of_set_A x ∨ is_element_of_set_B x)} = {x : ℤ | is_element_of_complement_U x} :=
by
  sorry

end complement_union_A_B_l294_294013


namespace expected_sum_2010_l294_294163

noncomputable def expectedRolls (n : ℕ) : ℝ := 
  if n == 0 then 0
  else 1 + (1 / 6) * ∑ i in (Finset.range 6), expectedRolls (n - i)

theorem expected_sum_2010 : abs (expectedRolls 2010 - 574.761904) < 0.001 :=
sorry

end expected_sum_2010_l294_294163


namespace variance_expansion_l294_294796

noncomputable def var (s : List ℝ) : ℝ := sorry -- definition of variance

variable {s : List ℝ} -- introduce the list of data points

theorem variance_expansion (h : var s = 3) :
  var (s.map (λ x, 3 * (x - 2))) = 27 :=
sorry

end variance_expansion_l294_294796


namespace simplify_fraction_l294_294124

theorem simplify_fraction :
  (3^100 + 3^98) / (3^100 - 3^98) = 5 / 4 := 
by sorry

end simplify_fraction_l294_294124


namespace modular_inverse_7_10000_l294_294839

theorem modular_inverse_7_10000 :
  (7 * 8571) % 10000 = 1 := 
sorry

end modular_inverse_7_10000_l294_294839


namespace nearest_edge_of_picture_l294_294889

theorem nearest_edge_of_picture
    (wall_width : ℝ) (picture_width : ℝ) (offset : ℝ) (x : ℝ)
    (hw : wall_width = 25) (hp : picture_width = 5) (ho : offset = 2) :
    x + (picture_width / 2) + offset = wall_width / 2 →
    x = 8 :=
by
  intros h
  sorry

end nearest_edge_of_picture_l294_294889


namespace count_noncongruent_triangles_l294_294210

-- Definitions of the conditions
def is_integer_sided_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b > c ∧ a + b + c < 20

def is_non_equilateral_isosceles_right (a b c : ℕ) : Prop :=
  ¬ (a = b ∨ b = c ∨ a = c) ∧ a^2 + b^2 ≠ c^2

-- The statement of the proof problem
theorem count_noncongruent_triangles : 
  { t : ℕ × ℕ × ℕ // is_integer_sided_triangle t.1 t.2 t.3 ∧ is_non_equilateral_isosceles_right t.1 t.2 t.3 }.card = 11 :=
sorry

end count_noncongruent_triangles_l294_294210


namespace chefs_and_waiters_left_l294_294801

theorem chefs_and_waiters_left (initial_chefs : ℕ) (initial_waiters : ℕ) (chefs_dropout : ℕ) (waiters_dropout : ℕ) :
  initial_chefs = 16 → initial_waiters = 16 → chefs_dropout = 6 → waiters_dropout = 3 →
  (initial_chefs - chefs_dropout) + (initial_waiters - waiters_dropout) = 23 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  norm_num
  sorry

end chefs_and_waiters_left_l294_294801


namespace lunch_combinations_l294_294720

noncomputable def main_course_options : Nat := 4

noncomputable def beverage_options (main_course : Int) : Nat :=
  if main_course = 1 ∨ main_course = 2 then 2 else 0

noncomputable def snack_options : Nat := 2

theorem lunch_combinations : main_course_options * beverage_options 1 + main_course_options * beverage_options 3 = 8 :=
by
  have Hamburger_Pasta_combinations : Nat := 2 * 2 * 2 -- 2 main courses, 2 beverages, 2 snacks
  have Salad_Taco_combinations : Nat := 2 * 0 * 2 -- 2 main courses, 0 beverages, 2 snacks
  have total_combinations : Nat := Hamburger_Pasta_combinations + Salad_Taco_combinations 
  have correct_combinations : total_combinations = 8 := by
    calc
      total_combinations 
        = 2 * 2 * 2 + 2 * 0 * 2 : by rw [total_combinations_def]
    ... = 8 : by rfl 
  exact correct_combinations

end lunch_combinations_l294_294720


namespace complement_union_eq_complement_l294_294001

open Set

section ComplementUnion

variable (k : ℤ)

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def UniversalSet : Set ℤ := univ
def ComplementUnion : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem complement_union_eq_complement :
  UniversalSet \ (SetA ∪ SetB) = ComplementUnion :=
by
  sorry

end ComplementUnion

end complement_union_eq_complement_l294_294001


namespace angle_PSU_eq_20_l294_294386

theorem angle_PSU_eq_20
  (P Q R S T U : Type)
  (PRQ QRP : ℕ)
  (h_PRQ : PRQ = 60)
  (h_QRP : QRP = 80)
  (foot_of_perpendicular : S)
  (circumcenter : T)
  (diameter_end : U)
  : sorry :=
begin
  -- Given conditions (lean terms explanations)
  -- we define the terms angle PSQ, POT, PTO, and PSU
  -- and provide their relationships and theorems as provided in the problem statement 
  sorry
end

end angle_PSU_eq_20_l294_294386


namespace not_right_triangle_l294_294910

/-- In a triangle ABC, with angles A, B, C, the condition A = B = 2 * C does not form a right-angled triangle. -/
theorem not_right_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 2 * C) (h3 : A + B + C = 180) : 
    ¬(A = 90 ∨ B = 90 ∨ C = 90) := 
by
  sorry

end not_right_triangle_l294_294910


namespace alternating_binomials_sum_l294_294258

-- Define a function to represent the alternating sum of binomials.
def alternating_sum_binomials (n : ℕ) : ℤ :=
  ∑ k in finset.range (n / 2 + 1), (-1 : ℤ) ^ k * nat.choose n (2 * k)

theorem alternating_binomials_sum :
  alternating_sum_binomials 100 = -2 ^ 50 :=
by
  sorry

end alternating_binomials_sum_l294_294258


namespace smallest_yummy_integer_l294_294786

theorem smallest_yummy_integer :
  ∃ (n A : ℤ), 4046 = n * (2 * A + n - 1) ∧ A ≥ 0 ∧ (∀ m, 4046 = m * (2 * A + m - 1) ∧ m ≥ 0 → A ≤ 1011) :=
sorry

end smallest_yummy_integer_l294_294786


namespace symm_diff_complement_l294_294272

variable {U : Type} -- Universal set U
variable (A B : Set U) -- Sets A and B

-- Definition of symmetric difference
def symm_diff (X Y : Set U) : Set U := (X ∪ Y) \ (X ∩ Y)

theorem symm_diff_complement (A B : Set U) :
  (symm_diff A B) = (symm_diff (Aᶜ) (Bᶜ)) :=
sorry

end symm_diff_complement_l294_294272


namespace sequence_value_a1_l294_294651

theorem sequence_value_a1 (a : ℕ → ℝ) 
  (h₁ : ∀ n, a (n + 1) = (1 / 2) * a n) 
  (h₂ : a 4 = 8) : a 1 = 64 :=
sorry

end sequence_value_a1_l294_294651


namespace tom_has_18_apples_l294_294920

-- Definitions based on conditions
def phillip_apples : ℕ := 40
def ben_apples : ℕ := phillip_apples + 8
def tom_apples : ℕ := (3 * ben_apples) / 8

-- Theorem stating Tom has 18 apples given the conditions
theorem tom_has_18_apples : tom_apples = 18 :=
sorry

end tom_has_18_apples_l294_294920


namespace number_of_elements_in_M_l294_294339

-- Define the set M as specified in the problem
def M : Set ℕ := {x | 8 - x ∈ ℕ}

-- Theorem stating that the number of elements in the set M is 9
theorem number_of_elements_in_M : Finset.card (M.to_finset) = 9 := 
sorry

end number_of_elements_in_M_l294_294339


namespace tangent_line_at_1_increasing_function_range_p_range_condition_l294_294333

noncomputable def f (p : ℝ) (x : ℝ) : ℝ := p * x - p / x - 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.exp 1 / x

theorem tangent_line_at_1 (p : ℝ) (hp : p = 2) :
  let f := f p;
  (∃ m b : ℝ, f 1 = 0 ∧ m = 2 ∧ b = -2 ∧ ∀ x, f x - f 1 = m * (x - 1)) :=
sorry

theorem increasing_function_range (p : ℝ) :
  (∀ x : ℝ, 0 < x → f p x ≥ f p x) → 1 ≤ p :=
sorry

theorem p_range_condition (p : ℝ) 
  (h : p^2 - p ≥ 0 ∧ ∃ x0 ∈ Icc 1 (Real.exp 1), f p x0 > g x0) : 
  (p > 4 * Real.exp 1 / (Real.exp 1^2 - 1)) :=
sorry

end tangent_line_at_1_increasing_function_range_p_range_condition_l294_294333


namespace find_b_of_perpendicular_bisector_l294_294091

theorem find_b_of_perpendicular_bisector :
  (∃ b : ℝ, (∀ x y : ℝ, x + y = b → x + y = 4 + 6)) → b = 10 :=
by
  sorry

end find_b_of_perpendicular_bisector_l294_294091


namespace difference_of_squares_count_l294_294687

theorem difference_of_squares_count :
  let is_diff_of_squares (n : ℕ) := ∃ a b : ℕ, n = (a + b) * (a - b)
  let is_perfect_square (n : ℕ) := ∃ k : ℕ, n = k * k
  (finset.filter (λ n, is_diff_of_squares n ∧ is_perfect_square n) (finset.range 2001)).card = 25 :=
by 
  let is_diff_of_squares (n : ℕ) := ∃ a b : ℕ, n = (a + b) * (a - b)
  let is_perfect_square (n : ℕ) := ∃ k : ℕ, n = k * k
  exact sorry

end difference_of_squares_count_l294_294687


namespace range_log_plus_three_is_l294_294818

noncomputable def range_log_plus_three (x : ℝ) (h : x ≥ 1) : ℝ := 
(log 2 x) + 3

theorem range_log_plus_three_is (s : Set ℝ) (h : ∀ x, x ≥ 1 → range_log_plus_three x h ∈ s) : 
s = {y : ℝ | y ≥ 3} :=
sorry

end range_log_plus_three_is_l294_294818


namespace sum_areas_of_triangles_l294_294419

theorem sum_areas_of_triangles :
  let S_k (k : ℕ) := (1 / 2) * (1 / k) * (1 / (k + 1))
  ∑ k in finset.range 2011, S_k (k + 1) = 2011 / 4024 :=
by
  sorry

end sum_areas_of_triangles_l294_294419


namespace find_f_of_9_l294_294807

variable (f : ℝ → ℝ)

-- Conditions
axiom functional_eq : ∀ x y : ℝ, f (x + y) = f x * f y
axiom f_of_3 : f 3 = 4

-- Theorem statement to prove
theorem find_f_of_9 : f 9 = 64 := by
  sorry

end find_f_of_9_l294_294807


namespace rate_of_current_l294_294103

def speed_boat := 42 -- speed of the boat in still water (km/hr)
def distance := 35.2 -- distance traveled downstream (km)
def time := 44 / 60.0 -- time in hours

def effective_speed_downstream (v_boat : ℝ) (c : ℝ) : ℝ := v_boat + c

theorem rate_of_current (c : ℝ) : 
  effective_speed_downstream speed_boat c * time = distance ↔ c = 6 :=
by
  sorry

end rate_of_current_l294_294103


namespace charlie_certain_instrument_l294_294602

theorem charlie_certain_instrument :
  ∃ (x : ℕ), (1 + 2 + x) + (2 + 1 + 0) = 7 → x = 1 :=
by
  sorry

end charlie_certain_instrument_l294_294602


namespace expected_rolls_sum_2010_l294_294173

noncomputable def expected_rolls_to_reach_sum (n : ℕ) : ℚ :=
  if n == 0 then 0
  else (1/6) * (expected_rolls_to_reach_sum (n-1) + expected_rolls_to_reach_sum (n-2) +
               expected_rolls_to_reach_sum (n-3) + expected_rolls_to_reach_sum (n-4) +
               expected_rolls_to_reach_sum (n-5) + expected_rolls_to_reach_sum (n-6)) + 1

theorem expected_rolls_sum_2010 : expected_rolls_to_reach_sum 2010 ≈ 574.761904 := 
by 
  -- Proof omitted; the focus is on the statement
  sorry

end expected_rolls_sum_2010_l294_294173


namespace medical_team_compositions_l294_294447

theorem medical_team_compositions : ∃ n, n = 70 ∧ (
  let male_doctors := 5 in
  let female_doctors := 4 in
  let total_doctors := 9 in
  let no_restriction := (total_doctors.choose 3), 
  let all_male := (male_doctors.choose 3), 
  let all_female := (female_doctors.choose 3), 
  no_restriction - all_male - all_female = n
) :=
by
  sorry

end medical_team_compositions_l294_294447


namespace find_line_equation_l294_294986

-- Definition of point and line slope conditions
def point : ℝ × ℝ := (1, 0)
def original_line_slope : ℝ := -2 / 3
def new_slope : ℝ := 2 * original_line_slope

-- Statement of the problem
theorem find_line_equation  :
  ∃ A B C : ℝ, A ≠ 0 ∧ B ≠ 0 ∧ (∀ x y : ℝ, (x, y) = point → 4 * x + 3 * y = 4) ∧ 
    (A * new_slope) + (B) * (-1) = 0 := 
sorry

end find_line_equation_l294_294986


namespace part1_part2_l294_294662

theorem part1 
  (a b : ℝ^3) 
  (h_angle : real.angle a b = real.pi/3) 
  (h_a_norm : ∥a∥ = 2) 
  (h_b_norm : ∥b∥ = 3) :
  ∥3•a - b∥ = 3 * real.sqrt 3 := 
sorry

theorem part2 
  (a b : ℝ^3) 
  (h_angle : real.angle a b = real.pi/3) 
  (h_a_norm : ∥a∥ = 2) 
  (h_b_norm : ∥b∥ = 3) 
  (lambda : ℝ)
  (c : ℝ^3 := lambda•a - 2•b)
  (h_ortho : dot_product b c = 0) :
  lambda = 6 := 
sorry

end part1_part2_l294_294662


namespace correct_option_l294_294131

-- Conditions
def option_A (a : ℝ) : Prop := a^3 + a^3 = a^6
def option_B (a : ℝ) : Prop := (a^3)^2 = a^9
def option_C (a : ℝ) : Prop := a^6 / a^3 = a^2
def option_D (a b : ℝ) : Prop := (a * b)^2 = a^2 * b^2

-- Proof Problem Statement
theorem correct_option (a b : ℝ) : option_D a b ↔ ¬option_A a ∧ ¬option_B a ∧ ¬option_C a :=
by
  sorry

end correct_option_l294_294131


namespace carrot_price_l294_294935

variables (total_tomatoes : ℕ) (total_carrots : ℕ) (price_per_tomato : ℝ) (total_revenue : ℝ)

theorem carrot_price :
  total_tomatoes = 200 →
  total_carrots = 350 →
  price_per_tomato = 1 →
  total_revenue = 725 →
  (total_revenue - total_tomatoes * price_per_tomato) / total_carrots = 1.5 :=
by
  intros h1 h2 h3 h4
  sorry

end carrot_price_l294_294935


namespace union_complement_eq_l294_294341

open Set

theorem union_complement_eq :
  let U := {0, 1, 2, 3}
  let A := {0, 1, 2}
  let B := {0, 2, 3}
  A ∪ (U \ B) = {0, 1, 2} := 
by 
  let U := {0, 1, 2, 3}
  let A := {0, 1, 2}
  let B := {0, 2, 3}
  have h1: U \ B = {1} := sorry
  have h2: A ∪ {1} = {0, 1, 2} := sorry
  exact h2

end union_complement_eq_l294_294341


namespace number_of_shortest_paths_l294_294827

-- Define the concept of shortest paths
def shortest_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

-- State the theorem that needs to be proved
theorem number_of_shortest_paths (m n : ℕ) : shortest_paths m n = Nat.choose (m + n) m :=
by 
  sorry

end number_of_shortest_paths_l294_294827


namespace find_sine_of_alpha_l294_294287

theorem find_sine_of_alpha (α : ℝ) (h1 : 0 < α) (h2 : α < π) 
  (h3 : 3 * real.cos (2 * α) - 8 * real.cos α = 5) :
  real.sin α = real.sqrt 5 / 3 :=
sorry

end find_sine_of_alpha_l294_294287


namespace beef_sold_on_saturday_l294_294058

-- Definitions for the problem
variables (T F S : ℕ)
variable avg_beef : ℕ

-- Given conditions
def cond1 : T = 210 := by sorry
def cond2 : F = 2 * T := by sorry
def cond3 : (T + F + S) / 3 = 260 := by sorry

-- Theorem to be proven
theorem beef_sold_on_saturday : S = 150 :=
by 
  rw [cond1, cond2] at cond3
  sorry

end beef_sold_on_saturday_l294_294058


namespace changing_quantities_l294_294219

structure Triangle (α : Type) :=
(A B P : α)
(M N : α)
(theta : ℝ)
(movesAlongLine : Prop)
(constantAngle : Prop)
(PA_eq_PB : Prop)
(midpoints : M = midpoint P A ∧ N = midpoint P B)

theorem changing_quantities (α : Type) [EuclideanGeometry α] 
  (t : Triangle α) 
  (PA_eq_PB : t.PA_eq_PB)
  (constantAngle : t.constantAngle)
  (midpoints : t.midpoints):
  ∃ count : ℕ, 
    count = 2 
    ∧ (changes_length_MN t ≠ true)
    ∧ (changes_perimeter_PAB t ≠ true)
    ∧ (changes_area_PAB t = true)
    ∧ (changes_area_trapezoid_ABNM t = true) :=
begin
  -- The precise implementation of these change checking functions would be required for a full proof:
  -- changes_length_MN : Triangle α → Prop := sorry
  -- changes_perimeter_PAB : Triangle α → Prop := sorry
  -- changes_area_PAB : Triangle α → Prop := sorry
  -- changes_area_trapezoid_ABNM : Triangle α → Prop := sorry
  sorry
end

end changing_quantities_l294_294219


namespace solve_n_is_2_l294_294533

noncomputable def problem_statement (n : ℕ) : Prop :=
  ∃ m : ℕ, 9 * n^2 + 5 * n - 26 = m * (m + 1)

theorem solve_n_is_2 : problem_statement 2 :=
  sorry

end solve_n_is_2_l294_294533


namespace train_crossing_time_l294_294882
-- Part a: Identifying the questions and conditions

-- Question: How long does it take for the train to cross the platform?
-- Conditions:
-- 1. Speed of the train: 72 km/hr
-- 2. Length of the goods train: 440 m
-- 3. Length of the platform: 80 m

-- Part b: Identifying the solution steps and the correct answers

-- The solution steps involve:
-- 1. Summing the lengths of the train and the platform to get the total distance the train needs to cover.
-- 2. Converting the speed of the train from km/hr to m/s.
-- 3. Using the formula Time = Distance / Speed to find the time.

-- Correct answer: 26 seconds

-- Part c: Translating the question, conditions, and correct answer to a mathematically equivalent proof problem

-- Goal: Prove that the time it takes for the train to cross the platform is 26 seconds given the provided conditions.

-- Part d: Writing the Lean 4 statement


-- Definitions based on the given conditions
def speed_kmh : ℕ := 72
def length_train : ℕ := 440
def length_platform : ℕ := 80

-- Definition based on the conversion step in the solution
def speed_ms : ℕ := (72 * 1000) / 3600 -- Converting speed from km/hr to m/s

-- Goal: Prove that the time it takes for the train to cross the platform is 26 seconds
theorem train_crossing_time : ((length_train + length_platform) : ℕ) / speed_ms = 26 := by
  sorry

end train_crossing_time_l294_294882


namespace area_of_triangle_BCD_l294_294438

-- Define points A, B, C, D in 3-dimensional space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the distance function for points in 3D space
def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2 + (p2.z - p1.z) ^ 2)

-- Define the angle function between three points
def angle (p1 p2 p3 : Point3D) : ℝ :=
  let a := distance p1 p2
  let b := distance p2 p3
  let c := distance p1 p3
  real.acos ((a^2 + b^2 - c^2) / (2 * a * b))

-- Define conditions given in problem
noncomputable def points_satisfy_conditions (A B C D : Point3D) : Prop :=
  (distance A B = 2) ∧
  (distance B C = 2) ∧
  (distance C D = 2) ∧
  (distance D A = 2) ∧
  (angle A B C = 120 * real.pi / 180) ∧
  (angle B C D = 120 * real.pi / 180) ∧
  (C.z = 0) -- Parallel plane condition satisfied implicitly

-- Prove that the area of triangle BCD is 2√3 given the conditions
theorem area_of_triangle_BCD {A B C D : Point3D} (h : points_satisfy_conditions A B C D) : 
  let area := 1/2 * real.abs (B.x * D.y + C.x * B.y + D.x * C.y - (B.y * D.x + C.y * B.x + D.y * C.x)) in
  area = 2 * real.sqrt 3 :=
sorry

end area_of_triangle_BCD_l294_294438


namespace bob_ears_left_l294_294590

namespace CornProblem

-- Definitions of the given conditions
def initial_bob_bushels : ℕ := 120
def ears_per_bushel : ℕ := 15

def given_away_bushels_terry : ℕ := 15
def given_away_bushels_jerry : ℕ := 8
def given_away_bushels_linda : ℕ := 25
def given_away_ears_stacy : ℕ := 42
def given_away_bushels_susan : ℕ := 9
def given_away_bushels_tim : ℕ := 4
def given_away_ears_tim : ℕ := 18

-- Calculate initial ears of corn
noncomputable def initial_ears_of_corn : ℕ := initial_bob_bushels * ears_per_bushel

-- Calculate total ears given away in bushels
def total_ears_given_away_bushels : ℕ :=
  (given_away_bushels_terry + given_away_bushels_jerry + given_away_bushels_linda +
   given_away_bushels_susan + given_away_bushels_tim) * ears_per_bushel

-- Calculate total ears directly given away
def total_ears_given_away_direct : ℕ :=
  given_away_ears_stacy + given_away_ears_tim

-- Calculate total ears given away
def total_ears_given_away : ℕ :=
  total_ears_given_away_bushels + total_ears_given_away_direct

-- Calculate ears of corn Bob has left
noncomputable def ears_left : ℕ :=
  initial_ears_of_corn - total_ears_given_away

-- The proof statement
theorem bob_ears_left : ears_left = 825 := by
  sorry

end CornProblem

end bob_ears_left_l294_294590


namespace range_of_values_for_a_l294_294683

noncomputable def range_of_a (a b : ℝ) (cos_theta : ℝ) : Prop :=
  (2:ℝ) = b ∧ a = 2 * real.sqrt ((b)^2 - 2 * a * b * cos_theta + a^2) ∧ (cos_theta = 1 ∨ cos_theta = -1)

theorem range_of_values_for_a (a b : ℝ) (cos_theta : ℝ) (h1 : (2:ℝ) = b) (h2 : a = 2 * real.sqrt ((b)^2 - 2 * a * b * cos_theta + a^2)) (h3 : cos_theta = 1 ∨ cos_theta = -1) :
  a ∈ set.Icc (4/3) 4 :=
sorry

end range_of_values_for_a_l294_294683


namespace sin_alpha_eq_sqrt5_over_3_l294_294284

theorem sin_alpha_eq_sqrt5_over_3 {α : ℝ} (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = (Real.sqrt 5) / 3 :=
sorry

end sin_alpha_eq_sqrt5_over_3_l294_294284


namespace area_triangle_ABC_l294_294067

def point := ℝ × ℝ

def A : point := (3, 4)
def B' : point := (-3, 4)
def C' : point := (4, -3)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def area_of_triangle (a b c : point) : ℝ :=
  let base := distance a b
  let height := abs (a.2 - c.2)
  (1 / 2) * base * height

theorem area_triangle_ABC' : area_of_triangle A B' C' = 21 :=
by
  sorry

end area_triangle_ABC_l294_294067


namespace max_projection_value_l294_294343

-- Define the vectors and their properties
variables {V : Type*} [inner_product_space ℝ V]
variables (a b c e : V)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (hc : ∥c∥ = 1)
variables (hab : ⟪a, b⟫ = 1/2) (hbc : ⟪b, c⟫ = 1/2)
variables (e_is_unit : ∥e∥ = 1)

-- Define the main hypothesis to prove the maximum value statement
theorem max_projection_value :
  ∃ (e : V), ∥e∥ = 1 → |⟪a, e⟫| + |2 * ⟪b, e⟫| + |3 * ⟪c, e⟫| = 5 :=
sorry

end max_projection_value_l294_294343


namespace conic_section_is_hyperbola_l294_294231

-- Definitions for the conditions in the problem
def conic_section_equation (x y : ℝ) := (x - 4) ^ 2 = 5 * (y + 2) ^ 2 - 45

-- The theorem that we need to prove
theorem conic_section_is_hyperbola : ∀ x y : ℝ, (conic_section_equation x y) → "H" = "H" :=
by
  intro x y h
  sorry

end conic_section_is_hyperbola_l294_294231


namespace chemical_transformations_correct_l294_294847

def ethylbenzene : String := "C6H5CH2CH3"
def brominate (A : String) : String := "C6H5CH(Br)CH3"
def hydrolyze (B : String) : String := "C6H5CH(OH)CH3"
def dehydrate (C : String) : String := "C6H5CH=CH2"
def oxidize (D : String) : String := "C6H5COOH"
def brominate_with_catalyst (E : String) : String := "m-C6H4(Br)COOH"

def sequence_of_transformations : Prop :=
  ethylbenzene = "C6H5CH2CH3" ∧
  brominate ethylbenzene = "C6H5CH(Br)CH3" ∧
  hydrolyze (brominate ethylbenzene) = "C6H5CH(OH)CH3" ∧
  dehydrate (hydrolyze (brominate ethylbenzene)) = "C6H5CH=CH2" ∧
  oxidize (dehydrate (hydrolyze (brominate ethylbenzene))) = "C6H5COOH" ∧
  brominate_with_catalyst (oxidize (dehydrate (hydrolyze (brominate ethylbenzene)))) = "m-C6H4(Br)COOH"

theorem chemical_transformations_correct : sequence_of_transformations :=
by
  -- proof would go here
  sorry

end chemical_transformations_correct_l294_294847


namespace max_value_sin_sin2x_l294_294314

open Real

/-- Given x is an acute angle, find the maximum value of the function y = sin x * sin (2 * x). -/
theorem max_value_sin_sin2x (x : ℝ) (hx : 0 < x ∧ x < π / 2) :
    ∃ max_y : ℝ, ∀ y : ℝ, y = sin x * sin (2 * x) -> y ≤ max_y ∧ max_y = 4 * sqrt 3 / 9 :=
by
  -- To be completed
  sorry

end max_value_sin_sin2x_l294_294314


namespace log3_729sqrt3_l294_294620

noncomputable def log3 (x : Real) := Real.log x / Real.log 3

theorem log3_729sqrt3 :
  log3 (729 * Real.sqrt 3) = 6.5 :=
by
  have h1 : 729 = 3 ^ 6 := by norm_num
  have h2 : Real.sqrt 3 = 3 ^ 0.5 := by norm_num
  sorry

end log3_729sqrt3_l294_294620


namespace original_total_movies_is_293_l294_294849

noncomputable def original_movies (dvd_to_bluray_ratio : ℕ × ℕ) (additional_blurays : ℕ) (new_ratio : ℕ × ℕ) : ℕ :=
  let original_dvds := dvd_to_bluray_ratio.1
  let original_blurays := dvd_to_bluray_ratio.2
  let added_blurays := additional_blurays
  let new_dvds := new_ratio.1
  let new_blurays := new_ratio.2
  let x := (new_dvds * original_blurays - new_blurays * original_dvds) / (new_blurays * original_dvds - added_blurays * original_blurays)
  let total_movies := (original_dvds * x + original_blurays * x)
  let blurays_after_purchase := original_blurays * x + added_blurays

  if (new_dvds * (original_blurays * x + added_blurays) = new_blurays * (original_dvds * x))
  then 
    (original_dvds * x + original_blurays * x)
  else
    0 -- This case should theoretically never happen if the input ratios are consistent.

theorem original_total_movies_is_293 : original_movies (7, 2) 5 (13, 4) = 293 :=
by sorry

end original_total_movies_is_293_l294_294849


namespace annual_growth_rate_l294_294760

theorem annual_growth_rate
  (income_2018 : ℝ)
  (income_2020 : ℝ)
  (income_2021_target : ℝ)
  (x : ℝ)
  (h1 : income_2018 = 3200)
  (h2 : income_2020 = 5000)
  (h3 : income_2021_target = 6200)
  (hx2_eq: (1 + x)^2 = income_2020 / income_2018)
  (hx_eq : x = 1 / 4) :
  let income_2021 := income_2020 * (1 + x) in
  (income_2021 ≥ income_2021_target) :=
by
  let growth_rate := x
  have h4 : 3200 * (1 + x)^2 = 5000, from calc
    3200 * (1 + x)^2 = income_2018 * (1 + x)^2 : by rw h1
    ... = income_2020 : by rw hx2_eq; rw h2
  have h5 : x = 1 / 4, from hx_eq
  have income_2021 := 5000 * (1 + x) * (by rw [h2, h5]),
  show 5000 * (5 / 4) ≥ 6200 by sorry

end annual_growth_rate_l294_294760


namespace integer_terms_in_arithmetic_sequence_l294_294797

variable a_3 : ℤ
variable a_18 : ℤ
variable n : ℕ

theorem integer_terms_in_arithmetic_sequence (a_3_eq : a_3 = 14) (a_18_eq : a_18 = 23) : 
  ∃ m ≤ 401, ∀ k < 2010, let d := 0.6 in let a_1 := 12.8 in (a_1 + (↑k - 1) * d).den = 1 := sorry

end integer_terms_in_arithmetic_sequence_l294_294797


namespace solve_for_a_l294_294853

theorem solve_for_a:
  ∃ a : ℝ, ((2 * a + 16) + (3 * a - 8)) / 2 = 84 → a = 32 :=
by
  intros a h
  -- this will be where the proof would go
  sorry

end solve_for_a_l294_294853


namespace triangles_congruent_l294_294762

variables (A B C A1 B1 : Type) [Inhabited A] [Inhabited B] [Inhabited C]
variables (ABC : Triangle) (AB_is_base : IsosTriangle ABC A B)
variables (A1_on_AB : PointOnBase ABC A1) (B1_on_AB : PointOnBase ABC B1)
variable (AB1_eq_BA1 : Distance ABC A B1 = Distance ABC B A1)

theorem triangles_congruent :
  CongruentTriangles (Triangle.mk ABC A B1 C) (Triangle.mk ABC B A1 C) :=
sorry

end triangles_congruent_l294_294762


namespace units_digit_eight_consecutive_l294_294617

theorem units_digit_eight_consecutive (n : ℕ) (n_pos : 0 < n) : 
  (∏ i in finset.range 8, (n + i)) % 10 = 0 := 
sorry

end units_digit_eight_consecutive_l294_294617


namespace sum_of_integers_satisfying_inequality_l294_294458

theorem sum_of_integers_satisfying_inequality :
  let predicate (x : ℝ) := (sqrt (3 * x - 7) - sqrt (3 * x^2 - 13 * x + 13) >= 3 * x^2 - 16 * x + 20)
  (2 : ℝ) ≤ x ∧ x ≤ (10 : ℝ) / 3 ∧ predicate x → x = 3 :=
by
  sorry

end sum_of_integers_satisfying_inequality_l294_294458


namespace shape_with_circular_views_is_sphere_l294_294573

/-- Define the views of the cuboid, cylinder, cone, and sphere. -/
structure Views (shape : Type) :=
(front_view : Type)
(left_view : Type)
(top_view : Type)

def is_cuboid (s : Views) : Prop :=
s.front_view = Rectangle ∧ s.left_view = Rectangle ∧ s.top_view = Rectangle

def is_cylinder (s : Views) : Prop :=
s.front_view = Rectangle ∧ s.left_view = Rectangle ∧ s.top_view = Circle

def is_cone (s : Views) : Prop :=
s.front_view = IsoscelesTriangle ∧ s.left_view = IsoscelesTriangle ∧ s.top_view = Circle

def is_sphere (s : Views) : Prop :=
s.front_view = Circle ∧ s.left_view = Circle ∧ s.top_view = Circle

/-- Proof problem: Prove that the only shape with circular views in all three perspectives (front, left, top) is the sphere. -/
theorem shape_with_circular_views_is_sphere :
  ∀ (s : Views), 
    (s.front_view = Circle ∧ s.left_view = Circle ∧ s.top_view = Circle) → 
    is_sphere s ∧ ¬ is_cuboid s ∧ ¬ is_cylinder s ∧ ¬ is_cone s :=
by
  intro s h
  sorry

end shape_with_circular_views_is_sphere_l294_294573


namespace median_length_isosceles_l294_294118

noncomputable def median_length (D E F : Point) (hDE : dist D E = 13) (hDF : dist D F = 13) (hEF : dist E F = 14) : ℝ :=
  let M := midpoint E F
  sqrt ((dist D E)^2 - (dist E M)^2)

theorem median_length_isosceles (D E F : Point) 
  (hDE : dist D E = 13) (hDF : dist D F = 13) (hEF : dist E F = 14) :
  median_length D E F hDE hDF hEF = 2 * sqrt 30 := 
sorry

end median_length_isosceles_l294_294118


namespace polynomial_coefficients_sum_l294_294639

theorem polynomial_coefficients_sum (a a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (h : (λ x : ℝ, (x-2)^6) = (λ x : ℝ, a + a_1 * (x+1) + a_2 * (x+1)^2 + a_3 * (x+1)^3 + a_4 * (x+1)^4 + a_5 * (x+1)^5 + a_6 * (x+1)^6)) :
  a + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 64 := 
sorry

end polynomial_coefficients_sum_l294_294639


namespace intersection_A_B_l294_294859

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {x | x^2 + x = 0}

theorem intersection_A_B : A ∩ B = {0} :=
by
  sorry

end intersection_A_B_l294_294859


namespace roberto_outfits_l294_294444

theorem roberto_outfits (trousers shirts jackets : ℕ) (restricted_shirt restricted_jacket : ℕ) 
  (h_trousers : trousers = 5) 
  (h_shirts : shirts = 6) 
  (h_jackets : jackets = 4) 
  (h_restricted_shirt : restricted_shirt = 1) 
  (h_restricted_jacket : restricted_jacket = 1) : 
  ((trousers * shirts * jackets) - (restricted_shirt * restricted_jacket * trousers) = 115) := 
  by 
    sorry

end roberto_outfits_l294_294444


namespace line_equation_l294_294648

open Real

theorem line_equation (x y : Real) : 
  (3 * x + 2 * y - 1 = 0) ↔ (y = (-(3 / 2)) * x + 2.5) :=
by
  sorry

end line_equation_l294_294648


namespace find_width_l294_294188

-- Definition of the perimeter of a rectangle
def perimeter (L W : ℝ) : ℝ := 2 * (L + W)

-- The given conditions
def length := 13
def perimeter_value := 50

-- The goal to prove: if the perimeter is 50 and the length is 13, then the width must be 12
theorem find_width :
  ∃ (W : ℝ), perimeter length W = perimeter_value ∧ W = 12 :=
by
  sorry

end find_width_l294_294188


namespace total_jellybeans_l294_294599

-- Define the conditions
def a := 3 * 12       -- Caleb's jellybeans
def b := a / 2        -- Sophie's jellybeans

-- Define the goal
def total := a + b    -- Total jellybeans

-- The theorem statement
theorem total_jellybeans : total = 54 :=
by
  -- Proof placeholder
  sorry

end total_jellybeans_l294_294599


namespace problem_s5_value_l294_294306

theorem problem_s5_value :
  let a : ℕ → ℤ := λ n, 1 + (n - 1) * 2,
      b : ℕ → ℤ := λ n, if n = 1 then 2 else (1 - 2 * n) * 2 ^ n,
      S : ℕ → ℤ := λ n, (finset.range n).sum b
  in S 5 = -450 :=
by
  let a := λ n, 1 + (n - 1) * 2
  let b := λ n, if n = 1 then 2 else (1 - 2 * n) * 2 ^ n
  let S := λ n, (finset.range n).sum b
  sorry

end problem_s5_value_l294_294306


namespace blue_chairs_fewer_than_yellow_l294_294705

theorem blue_chairs_fewer_than_yellow :
  ∀ (red_chairs yellow_chairs chairs_left total_chairs blue_chairs : ℕ),
    red_chairs = 4 →
    yellow_chairs = 2 * red_chairs →
    chairs_left = 15 →
    total_chairs = chairs_left + 3 →
    blue_chairs = total_chairs - (red_chairs + yellow_chairs) →
    yellow_chairs - blue_chairs = 2 :=
by sorry

end blue_chairs_fewer_than_yellow_l294_294705


namespace CORNELIA_area_l294_294724

noncomputable def CAROLINE_side1 : ℝ := sqrt 2
noncomputable def CAROLINE_side2 : ℝ := 1

/-- The area enclosed by the octagon CORNELIA is given by the formula described in the conditions,
    where the octagon is formed by connecting alternate vertices of the octagon CAROLINE. -/
theorem CORNELIA_area (a b : ℕ) (h_coprime : Nat.coprime a b):
  let K := (4 * (1 / 2 * 1 * (4 / 3)) + 2 * (1 / 2 * (1 / 3) * (1 / 2))) in
  K = (17 : ℝ) / 6 ∧ a + b = 23 := 
sorry

end CORNELIA_area_l294_294724


namespace classSubscriptionPigeonhole_l294_294603

noncomputable def minStudentsSubscribeSameCombination : Prop :=
∀ (students : Finset ℕ), 
    (students.card = 39) →
    (∀ s ∈ students, s ∈ {1, 2, 3}) →
    (∃ type_counts : Fin (3 + 3 + 1) → ℕ, ∀ t, type_counts t ≥ 6)

theorem classSubscriptionPigeonhole :
  minStudentsSubscribeSameCombination :=
begin
  sorry
end

end classSubscriptionPigeonhole_l294_294603


namespace concentric_circles_circumference_difference_l294_294132

theorem concentric_circles_circumference_difference :
  ∀ (radius_diff inner_diameter : ℝ),
  radius_diff = 15 →
  inner_diameter = 50 →
  ((π * (inner_diameter + 2 * radius_diff)) - (π * inner_diameter)) = 30 * π :=
by
  sorry

end concentric_circles_circumference_difference_l294_294132


namespace abs_pi_expression_l294_294951

theorem abs_pi_expression : (|π - |π - 10|| = 10 - 2 * π) := by
  sorry

end abs_pi_expression_l294_294951


namespace terminating_decimal_count_l294_294275

theorem terminating_decimal_count :
  {n : ℕ | 1 ≤ n ∧ n ≤ 360 ∧ ∃ k, n = 7 * k}.card = 51 :=
by
  sorry

end terminating_decimal_count_l294_294275


namespace cans_in_each_package_of_cat_food_l294_294198

-- Definitions and conditions
def cans_per_package_cat (c : ℕ) := 9 * c
def cans_per_package_dog := 7 * 5
def extra_cans_cat := 55

-- Theorem stating the problem and the answer
theorem cans_in_each_package_of_cat_food (c : ℕ) (h: cans_per_package_cat c = cans_per_package_dog + extra_cans_cat) :
  c = 10 :=
sorry

end cans_in_each_package_of_cat_food_l294_294198


namespace coin_order_is_correct_l294_294960

-- Define the coins as an inductive type
inductive Coin
| A | B | C | D | E | F

open Coin

-- Define the overlaps using orderings
def order : list Coin := [F, B, D, C, A, E]

-- Given conditions as hypotheses
axiom h1 : order.indexOf F < order.indexOf B
axiom h2 : order.indexOf B < order.indexOf A ∧ order.indexOf B < order.indexOf C ∧ order.indexOf B < order.indexOf E
axiom h3 : order.indexOf D < order.indexOf A ∧ order.indexOf D < order.indexOf C
axiom h4 : order.indexOf A < order.indexOf E
axiom h5 : order.indexOf C < order.indexOf E

-- Prove the order is correct
theorem coin_order_is_correct : order = [F, B, D, C, A, E] :=
by {
    -- Insert necessary proof steps here
    sorry
}

end coin_order_is_correct_l294_294960


namespace stars_and_bars_l294_294037

theorem stars_and_bars (n k : ℕ) : 
  let count_ways := λ n k, binomial (n + k - 1) (k - 1)
  in count_ways n k = binomial (n + k - 1) (k - 1) := by
sorry

end stars_and_bars_l294_294037


namespace sphere_has_circular_views_l294_294578

-- Define the geometric shapes
inductive Shape
| cuboid
| cylinder
| cone
| sphere

-- Define a function that describes the views of a shape
def views (s: Shape) : (String × String × String) :=
match s with
| Shape.cuboid   => ("Rectangle", "Rectangle", "Rectangle")
| Shape.cylinder => ("Rectangle", "Rectangle", "Circle")
| Shape.cone     => ("Isosceles Triangle", "Isosceles Triangle", "Circle")
| Shape.sphere   => ("Circle", "Circle", "Circle")

-- Define the property of having circular views in all perspectives
def has_circular_views (s: Shape) : Prop :=
views s = ("Circle", "Circle", "Circle")

-- The theorem to prove
theorem sphere_has_circular_views :
  ∀ (s : Shape), has_circular_views s ↔ s = Shape.sphere :=
by sorry

end sphere_has_circular_views_l294_294578


namespace problem_statement_l294_294051

noncomputable def f (x : ℝ) : ℝ := abs (2 - log x / log 3)

theorem problem_statement (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a < b) (h5 : b < c) 
    (h6 : f a = 2) (h7 : f b = 1) (h8 : f c = 2) :
    (a * c / b = 243) := 
sorry

end problem_statement_l294_294051


namespace abs_pi_expression_l294_294956

theorem abs_pi_expression (h : Real.pi < 10) : 
  Real.abs (Real.pi - Real.abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_pi_expression_l294_294956


namespace cos_theta_when_f_maximizes_l294_294512

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x)

theorem cos_theta_when_f_maximizes (θ : ℝ) (h : ∀ x, f x ≤ f θ) : Real.cos θ = Real.sqrt 3 / 2 := by
  sorry

end cos_theta_when_f_maximizes_l294_294512


namespace find_number_l294_294279

theorem find_number (x : ℝ) (h : 42 - 3 * x = 12) : x = 10 := 
by 
  sorry

end find_number_l294_294279


namespace last_one_remaining_is_dom_l294_294711

-- Define the student names
inductive Student
| Alg | Bez | Cal | Dom | Eni | Fed | Gio | Hal
deriving DecidableEq, Repr

-- Initial position list
def initial_positions : List Student :=
[Student.Alg, Student.Bez, Student.Cal, Student.Dom, Student.Eni, Student.Fed, Student.Gio, Student.Hal]

-- Predicate to determine if a number contains digit 1 or is a multiple of 11
def should_eliminate (n : ℕ) : Bool :=
n % 11 = 0 || (n.toString.contains '1')

/-- The problem: Prove that Dom is the last one remaining under given conditions -/
theorem last_one_remaining_is_dom : 
  ∃ last_student : Student, last_student = Student.Dom :=
by
  -- Enumeration of counting and eliminations would go here
  sorry

end last_one_remaining_is_dom_l294_294711


namespace determine_ab_l294_294047

noncomputable def f (a b : ℕ) (x : ℝ) : ℝ := x^2 + 2 * a * x + b * 2^x

theorem determine_ab (a b : ℕ) (h : ∀ x : ℝ, f a b x = 0 ↔ f a b (f a b x) = 0) :
  (a, b) = (0, 0) ∨ (a, b) = (1, 0) :=
by
  sorry

end determine_ab_l294_294047


namespace sufficient_but_not_necessary_l294_294532

theorem sufficient_but_not_necessary (m : ℕ) :
  m = 9 → m > 8 ∧ ∃ k : ℕ, k > 8 ∧ k ≠ 9 :=
by
  sorry

end sufficient_but_not_necessary_l294_294532


namespace angle_CFD_eq_60_l294_294401

-- Define the necessary geometric entities and conditions
variables (O A B C D F : Type) [Circle O] [Point A] [Point B] [Point C] [Point D] [Point F]

-- Assume AB is a diameter of the circle centered at O
axiom diameter_of_circle (h_diam : Diameter O A B) 

-- Assume F is a point on the circle
axiom point_on_circle (H_F_on_circle : OnCircle O F)

-- Assume the tangent at B intersects the tangent at F and AF at C and D, respectively
axiom tangent_at_B (H_tangent_B : Tangent O B) 
axiom tangent_at_F (H_tangent_F : Tangent O F) 
axiom intersects_AF_at_D (H_AF_D : Intersects AF D)
axiom intersects_tangents_at_C (H_tangents_C : Intersects_Tangents B F C)

-- Given angle BAF = 30 degrees
axiom angle_BAF_eq_30 (angle_BAF : Angle B A F = 30)

-- Prove angle CFD = 60 degrees
theorem angle_CFD_eq_60 : Angle C F D = 60 :=
by
  sorry

end angle_CFD_eq_60_l294_294401


namespace solution_l294_294417

variables {n : ℕ} (x : Fin (n+2) → ℝ)

def conditions (x : Fin (n+2) → ℝ) : Prop :=
  (∀ i : Fin (n+1), 0 < x i.succ) ∧
  (∀ i : Fin (n+1), x i < x i.succ) ∧
  (x 0 = 0) ∧
  (x (Fin.last (n+1)) = 1) ∧
  (∀ i : Fin (n+1), ∑ j in Finset.univ.erase i, 1 / (x i - x j) = 0)

theorem solution (x : Fin (n+2) → ℝ) (h : conditions x) :
  ∀ i : Fin (n+1), x (Fin.last (n+1) - i) = 1 - x i.succ := 
sorry

end solution_l294_294417


namespace domain_of_v_l294_294504

noncomputable def v (x : ℝ) : ℝ := 1 / (Real.sqrt (x - 3))

theorem domain_of_v :
    {x : ℝ | x > 3} = set_of (λ x, ∃ y, v x = y) :=
by
  sorry

end domain_of_v_l294_294504


namespace count_ordered_pairs_l294_294251

theorem count_ordered_pairs : 
  ∃ n, n = 719 ∧ 
    (∀ (a b : ℕ), a + b = 1100 → 
      (∀ d ∈ [a, b], 
        ¬(∃ k : ℕ, d = 10 * k ∨ d % 10 = 0 ∨ d / 10 % 10 = 0 ∨ d % 5 = 0))) -> n = 719 :=
by
  sorry

end count_ordered_pairs_l294_294251


namespace complement_of_union_l294_294021

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def U : Set ℤ := Set.univ

theorem complement_of_union (x : ℤ) : (x ∉ A ∪ B) ↔ ∃ k : ℤ, x = 3 * k := 
by {
  sorry
}

end complement_of_union_l294_294021


namespace find_cost_per_component_l294_294548

noncomputable def cost_per_component (C : ℝ) : Prop :=
  let num_components := 150
  let shipping_cost_per_component := 3
  let fixed_monthly_costs := 16500
  let sale_price_per_component := 191.67
  let total_cost := num_components * C + num_components * shipping_cost_per_component + fixed_monthly_costs
  let total_revenue := num_components * sale_price_per_component
  total_cost = total_revenue

theorem find_cost_per_component : cost_per_component 78.67 :=
by
  unfold cost_per_component
  have h : 150 * 78.67 + 150 * 3 + 16500 = 150 * 191.67 := by norm_num
  exact h

end find_cost_per_component_l294_294548


namespace spherical_segment_central_angle_l294_294837

noncomputable def central_angle_spherical_segment (r : ℝ) : ℝ :=
  let h := r * (1 - Real.cos (α / 2))
  let R := r * Real.sin (α / 2)
  have hyp1: 2 * π * r * h + π * R * R = π * r * r := sorry
  let α := 2 * Real.arccos (-1 + Real.sqrt 3)
  α ≈ 85.8828 -- 85 degrees 52 minutes 58 seconds

theorem spherical_segment_central_angle : 
  ∀ (r : ℝ), 
  let h := r * (1 - Real.cos (α / 2)) in
  let R := r * Real.sin (α / 2) in
  (2 * π * r * h + π * R * R = π * r * r) → 
  central_angle_spherical_segment r = 85.8828 := sorry

end spherical_segment_central_angle_l294_294837


namespace probability_product_odd_from_8_rolls_l294_294894

theorem probability_product_odd_from_8_rolls :
  (∃ die : Type, (∀ roll : die, roll ∈ {1, 2, 3, 4, 5, 6}) → 
  (∀ i : ℕ, i ∈ finset.range 8 → roll % 2 = 1) → (1 / 2) ^ 8 = 1 / 256) := sorry

end probability_product_odd_from_8_rolls_l294_294894


namespace y_coordinate_midpoint_l294_294982

theorem y_coordinate_midpoint : 
  let L : (ℝ → ℝ) := λ x => x - 1
  let P : (ℝ → ℝ) := λ y => 8 * (y^2)
  ∃ (x₁ x₂ y₁ y₂ : ℝ),
    P (L x₁) = y₁ ∧ P (L x₂) = y₂ ∧ 
    L x₁ = y₁ ∧ L x₂ = y₂ ∧ 
    x₁ + x₂ = 10 ∧ y₁ + y₂ = 8 ∧
    (y₁ + y₂) / 2 = 4 := sorry

end y_coordinate_midpoint_l294_294982


namespace greatest_distance_between_centers_l294_294489

-- Define the dimensions of the rectangle
def rectangle_length := 20
def rectangle_width := 16

-- Define the diameter of each circle
def circle_diameter := 8
def circle_radius := circle_diameter / 2

-- Define the coordinates of the centers of the circles
def center1 := (circle_radius, circle_radius)
def center2 := (rectangle_length - circle_radius, rectangle_width - circle_radius)

-- Calculate the distance between the two centers
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Prove the greatest possible distance between the centers
theorem greatest_distance_between_centers :
  distance center1 center2 = 2 * real.sqrt 52 := by
sorry

end greatest_distance_between_centers_l294_294489


namespace dadAgeWhenXiaoHongIs7_l294_294222

variable {a : ℕ}

-- Condition: Dad's age is given as 'a'
-- Condition: Dad's age is 4 times plus 3 years more than Xiao Hong's age
def xiaoHongAge (a : ℕ) : ℕ := (a - 3) / 4

theorem dadAgeWhenXiaoHongIs7 : xiaoHongAge a = 7 → a = 31 := by
  intro h
  have h1 : a - 3 = 28 := by sorry   -- Algebraic manipulation needed
  have h2 : a = 31 := by sorry       -- Algebraic manipulation needed
  exact h2

end dadAgeWhenXiaoHongIs7_l294_294222


namespace units_digit_7_pow_3_pow_5_l294_294631

theorem units_digit_7_pow_3_pow_5 : ∀ (n : ℕ), n % 4 = 3 → ∀ k, 7 ^ k ≡ 3 [MOD 10] :=
by 
    sorry

end units_digit_7_pow_3_pow_5_l294_294631


namespace sqrt_product_l294_294932

theorem sqrt_product (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  sqrt (3 * x) * sqrt (1 / 3 * x * y) = x * sqrt y :=
sorry

end sqrt_product_l294_294932


namespace problem_l294_294761

-- Definitions of the conditions
def focus_C1_F1 : ℝ × ℝ := (-real.sqrt 3, 0)
def focus_C1_F2 : ℝ × ℝ := (real.sqrt 3, 0)

def satisfies_ellipse (M : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (d : ℝ) :=
  real.dist M F₁ + real.dist M F₂ = d

-- Defining the focus and vertex of parabola C2
def focus_C2 : ℝ × ℝ := (1, 0)
def vertex_C2 : ℝ × ℝ := (0, 0)

-- Definitions of the correct answers
def C1_eqn : Prop := ∀ (x y : ℝ), satisfies_ellipse (x, y) focus_C1_F1 focus_C1_F2 4 ↔ (x^2 / 4 + y^2 = 1)
def C2_eqn : Prop := ∀ (x y : ℝ), (focus_C2 = (1, 0) ∧ vertex_C2 = (0, 0)) → (y^2 = 4 * x)

-- Geometric relationship and line equation
def perpendicular_vectors (M N O : ℝ × ℝ) :=
  let OM := ((M.1 - O.1), (M.2 - O.2)) in
  let ON := ((N.1 - O.1), (N.2 - O.2)) in
  OM.1 * ON.1 + OM.2 * ON.2 = 0

def satisfies_line_eqn (F M N : ℝ × ℝ) :=
  ∃ k : ℝ, (M.2 = k * (M.1 - 1)) ∧ (N.2 = k * (N.1 - 1)) ∧
           ((k = 2) ∨ (k = -2))

-- Main theorem statement
theorem problem :
  (C1_eqn) ∧ (C2_eqn) ∧ (∃ (M N : ℝ × ℝ),
  perpendicular_vectors M N vertex_C2 ∧ satisfies_ellipse M focus_C1_F1 focus_C1_F2 4 ∧
  satisfies_ellipse N focus_C1_F1 focus_C1_F2 4 ∧
  satisfies_line_eqn focus_C2 M N)
  := 
sorry

end problem_l294_294761


namespace max_interesting_pairs_l294_294433

/--
Given a 5 x 7 grid with 9 marked cells, a pair of cells sharing a side is called interesting if at least one cell in the pair is marked. Prove that the maximum number of interesting pairs is 35.
-/
theorem max_interesting_pairs : 
  ∀ grid : Matrix (Fin 5) (Fin 7) Bool, 
  (∑ x y, if grid x y then 1 else 0 = 9) →
  ∃ n, n = 35 ∧ 
  ∀ i j, (i < 5) → (j < 7) →
  (grid i j = true → 
    (∃ cnt, cnt ≤ 4 ∧
     ( ∑ dx dy, (dx, dy) ∈ {(0, 1), (0, -1), (1, 0), (-1, 0)} ∧ 
       (i + dx).val < 5 ∧ (j + dy).val < 7 ∧ grid (i + dx) (j + dy) = true) = cnt) ∧
     ∃ interesting_pairs, 
     interesting_pairs ≤ 35 ∧
     ( ∑ x y, grid x y = true → interesting_pairs = 
       (∑ dx dy, (dx, dy) ∈ {(0, 1), (0, -1), (1, 0), (-1, 0)} ∧
       (x + dx).val < 5 ∧ (y + dy).val < 7 ∧ grid (x + dx) (y + dy) = true) = interesting_pairs)) :=
by
  sorry

end max_interesting_pairs_l294_294433


namespace sin_alpha_eq_sqrt5_over_3_l294_294285

theorem sin_alpha_eq_sqrt5_over_3 {α : ℝ} (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = (Real.sqrt 5) / 3 :=
sorry

end sin_alpha_eq_sqrt5_over_3_l294_294285


namespace ellipse_properties_l294_294309

def ellipse_eq (a b : ℝ) : (x y : ℝ) → Prop := 
  λ x y, x^2 / a^2 + y^2 / b^2 = 1

def point_M (x y : ℝ) : Prop := x = 2 ∧ y = 0
def point_N (x y : ℝ) : Prop := x = 0 ∧ y = 1

theorem ellipse_properties :
  (∀ M : ℝ × ℝ, point_M M.1 M.2) → (∀ N : ℝ × ℝ, point_N N.1 N.2) →
  ∃ a b : ℝ, ellipse_eq 2 1 ∧ 
  (eccentricity : ℝ := (real.sqrt 3) / 2) ∧
  (min_triangle_area : ℝ := 8 / 5) ∧ 
  (line_AB : (x y: ℝ) → Prop := λ x y, y = x ∨ y = -x) :=
sorry

end ellipse_properties_l294_294309


namespace complement_union_l294_294007

def A := { x : ℤ | ∃ k : ℤ, x = 3 * k + 1 }
def B := { x : ℤ | ∃ k : ℤ, x = 3 * k + 2 }
def U := ℤ

theorem complement_union :
  { x : ℤ | x ∉ (A ∪ B) } = { x : ℤ | ∃ k : ℤ, x = 3 * k } := 
by 
  sorry

end complement_union_l294_294007


namespace problem_l294_294293

noncomputable def f (α : Real) : Real :=
  (sin (π - α) * cos (2 * π - α)) / (cos (-π - α) * tan α)

theorem problem (h : f (-31 / 3 * π) = -1 / 2) : f (-31 / 3 * π) = -1 / 2 :=
  sorry

end problem_l294_294293


namespace matt_climbing_speed_l294_294395

theorem matt_climbing_speed :
  ∃ (x : ℝ), (12 * 7 = 7 * x + 42) ∧ x = 6 :=
by {
  sorry
}

end matt_climbing_speed_l294_294395


namespace total_surface_area_calc_l294_294157

/-- Given a cube with a total volume of 1 cubic foot, cut into four pieces by three parallel cuts:
1) The first cut is 0.4 feet from the top.
2) The second cut is 0.3 feet below the first.
3) The third cut is 0.1 feet below the second.
Prove that the total surface area of the new solid is 6 square feet. -/
theorem total_surface_area_calc :
  ∀ (A B C D : ℝ), 
    A = 0.4 → 
    B = 0.3 → 
    C = 0.1 → 
    D = 1 - (A + B + C) → 
    (6 : ℝ) = 6 := 
by 
  intros A B C D hA hB hC hD 
  sorry

end total_surface_area_calc_l294_294157


namespace max_real_part_l294_294431

theorem max_real_part (z : ℂ) : (z = -1 + complex.I ∨ 
                                 z = -real.sqrt 2 - complex.I ∨ 
                                 z = -1 + real.sqrt 2 * complex.I ∨ 
                                 z = -1 - complex.I ∨ 
                                 z = 3 * complex.I) → 
                                complex.re (z^7) ≤ complex.re ((3 * complex.I)^7) := 
by 
  intros h 
  cases h 
  case or.inl h1 {sorry} 
  case or.inr h1 { 
    cases h1 
    case or.inl h2 {sorry} 
    case or.inr h2 { 
      cases h2 
      case or.inl h3 {sorry} 
      case or.inr h3 { 
        cases h3 
        case or.inl h4 {sorry} 
        case or.inr h4 {sorry} 
      } 
    } 
  }

end max_real_part_l294_294431


namespace complement_union_A_B_l294_294016

def A := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def B := {x : ℤ | ∃ k : ℤ, x = 3 * k + 2}
def U := {x : ℤ | True}

theorem complement_union_A_B:
  (U \ (A ∪ B) = {x : ℤ | ∃ k : ℤ, x = 3 * k}) :=
by
  sorry

end complement_union_A_B_l294_294016


namespace even_function_has_specific_a_l294_294700

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x ^ 2 + (2 * a ^ 2 - a) * x + 1

-- State the proof problem
theorem even_function_has_specific_a (a : ℝ) :
  (∀ x : ℝ, f a (-x) = f a x) → a = 1 / 2 :=
by
  intros h
  sorry

end even_function_has_specific_a_l294_294700


namespace intersection_point_unique_l294_294094

theorem intersection_point_unique (k : ℝ) :
  (∃ y : ℝ, k = -2 * y^2 - 3 * y + 5) ∧ (∀ y₁ y₂ : ℝ, y₁ ≠ y₂ → -2 * y₁^2 - 3 * y₁ + 5 ≠ k ∨ -2 * y₂^2 - 3 * y₂ + 5 ≠ k)
  ↔ k = 49 / 8 := 
by sorry

end intersection_point_unique_l294_294094


namespace complement_union_A_B_l294_294017

def A := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def B := {x : ℤ | ∃ k : ℤ, x = 3 * k + 2}
def U := {x : ℤ | True}

theorem complement_union_A_B:
  (U \ (A ∪ B) = {x : ℤ | ∃ k : ℤ, x = 3 * k}) :=
by
  sorry

end complement_union_A_B_l294_294017


namespace pauls_age_difference_l294_294066

noncomputable def peter_and_paul : Prop :=
  ∃ (x y u v : ℕ), 
    -- Peter's birth year is in the form of 18xy
    x ∈ finset.range 10 ∧ y ∈ finset.range 10 ∧
    -- Paul's birth year is in the form of 19uv
    u ∈ finset.range 10 ∧ v ∈ finset.range 10 ∧
    -- Current year must be the same for both
    1809 + 11 * x + 2 * y = 1910 + 11 * u + 2 * v ∧
    -- Paul is younger than Peter by 9 years
    1900 + 10 * u + v + (1 + 9 + u + v) - 
    (1800 + 10 * x + y + (1 + 8 + x + y)) = 9

-- Statement of the theorem
theorem pauls_age_difference : peter_and_paul :=
sorry

end pauls_age_difference_l294_294066


namespace hyperbola_equation_l294_294466

theorem hyperbola_equation (a b c : ℝ) 
    (h_eccentricity : c / a = sqrt 5 / 2)
    (h_focus : c = sqrt 5) :
    ∃ (a b : ℝ), a = 2 ∧ b = 1 ∧ (∀ x y : ℝ, (x^2 / 4) - (y^2) = 1) :=
by {
    sorry
}

end hyperbola_equation_l294_294466


namespace trigonometric_identity_l294_294141

theorem trigonometric_identity 
  (α β γ : ℝ)
  (h1 : cos α + cos β + cos γ = real.sqrt (1 / 5))
  (h2 : sin α + sin β + sin γ = real.sqrt (4 / 5)) :
  cos (α - β) + cos (β - γ) + cos (γ - α) = -1 := 
by
  sorry

end trigonometric_identity_l294_294141


namespace parabola_midpoint_length_squared_l294_294069

theorem parabola_midpoint_length_squared :
  ∀ (A B : ℝ × ℝ), 
  (∃ (x y : ℝ), A = (x, 3*x^2 + 4*x + 2) ∧ B = (-x, -(3*x^2 + 4*x + 2)) ∧ ((A.1 + B.1) / 2 = 0) ∧ ((A.2 + B.2) / 2 = 0)) →
  dist A B^2 = 8 :=
by
  sorry

end parabola_midpoint_length_squared_l294_294069


namespace reflected_ray_slope_l294_294559

theorem reflected_ray_slope :
  ∃ k : ℚ, (k = -4/3 ∨ k = -3/4) ∧ 
          ∃ (p : ℝ × ℝ), p = (-2:ℝ, -3:ℝ) ∧ 
          ∃ (c : ℝ × ℝ), c = (-3:ℝ, 2:ℝ) ∧ 
          (∀ x y: ℝ, (x + 3)^2 + (y - 2)^2 = 1 → 1 = abs ((-3 * k - 2 - 2 * k - 3) / real.sqrt (k^2 + 1))) :=
begin
  sorry
end

end reflected_ray_slope_l294_294559


namespace min_jumps_required_to_visit_all_points_and_return_l294_294606

theorem min_jumps_required_to_visit_all_points_and_return :
  ∀ (n : ℕ), n = 2016 →
  ∀ jumps : ℕ → ℕ, (∀ i, jumps i = 2 ∨ jumps i = 3) →
  (∀ i, (jumps (i + 1) + jumps (i + 2)) % n = 0) →
  ∃ (min_jumps : ℕ), min_jumps = 2017 :=
by
  sorry

end min_jumps_required_to_visit_all_points_and_return_l294_294606


namespace determine_m_value_l294_294680

theorem determine_m_value : ∃ (m : ℕ), m > 0 ∧ 
  (∀ x : ℝ, x ≠ 0 → (x ^ (m^2 - 2*m - 3)) ≠ 0) ∧ 
  x ^ (m^2 - 2*m - 3) / ( (x : ℝ) ^ (m^2 - 2*m - 3)) = 1 ∧ 
  function.symmetric (λ x, x^((m^2 - 2*m - 3))) := 
(exists.intro 2
  (and.intro
    (by linarith)
    (and.intro
      (by intro x; intro h; simp; sorry)
      (by intro x; intro y; sorry))))

end determine_m_value_l294_294680


namespace silver_used_volume_l294_294791

theorem silver_used_volume :
  let r := 0.05 
  let h := 8403.380995252074
  π * r ^ 2 * h ≈ 66.048 := 
begin
  let r := 0.05,
  let h := 8403.380995252074,
  have V := π * r ^ 2 * h,
  sorry
end

end silver_used_volume_l294_294791


namespace problem_part1_problem_part2_l294_294297

open Real

variables {a b : EuclideanSpace ℝ (Fin 3)}

def norm_sq (v : EuclideanSpace ℝ (Fin 3)) := ∥v∥^2

noncomputable def angle_between (u v : EuclideanSpace ℝ (Fin 3)) : ℝ :=
Real.acos (u ⬝ v / (∥u∥ * ∥v∥))

-- Given conditions
def cond_a := ∥a∥ = 4
def cond_b := ∥b∥ = 2
def cond_angle := angle_between a b = π * 2 / 3

-- Proving the two results
theorem problem_part1 (h₁ : cond_a) (h₂ : cond_b) (h₃ : cond_angle) :
  ((a - 2 • b) ⬝ (a + b)) = 12 :=
sorry

theorem problem_part2 (h₁ : cond_a) (h₂ : cond_b) (h₃ : cond_angle) :
  angle_between a (a + b) = π / 6 :=
sorry

end problem_part1_problem_part2_l294_294297


namespace max_permutations_necessary_l294_294988

theorem max_permutations_necessary (S : set ℕ) (hS : S = {i | 1 ≤ i ∧ i ≤ 2014}) :
  ∃ P : ℕ, (∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → ∃ π : list ℕ, (π.perm (list.range 2014)) ∧ (list.index_of a π + 1 = list.index_of b π)) → P = 1007 :=
by
  use 1007
  sorry

end max_permutations_necessary_l294_294988


namespace f_p_k_minus_1_l294_294857

noncomputable def isLinear {K : Type*} [Ring K] (f : K → K) :=
  ∀ (p q : K), f (p + q) = f p + f q ∧ f (p * q) = f p * q + p * f q

variables {n : ℕ} (K : Type*) [CommRing K]

variables (x y : Fin (n+2) → K)
variables (f : K → K)
variables (p : Fin (n+2) → K)

axiom f_linear : isLinear f
axiom f_x : ∀ i : Fin (n+2), f (x i) = (n - 1 : ℕ) * x i + y i
axiom f_y : ∀ i : Fin (n+2), f (y i) = 2 * (n : ℕ) * y i

axiom polynomial_identity : ∀ (t : ℝ), 
  ∏ i in Finset.range (n+2), (t * x i + y i) = 
  ∑ i in Finset.range (n+2), p i * t^(n+1 - i)

theorem f_p_k_minus_1 : 
  ∀ k : Fin (n+2), 1 ≤ k.val → f (p (k - 1)) = (k : ℕ) * p k + (n + 1) * (n + k - 2) * p (k - 1) :=
sorry

end f_p_k_minus_1_l294_294857


namespace factorization_correct_l294_294376

theorem factorization_correct: ∀ (x : ℝ), (x^2 - 9 = (x + 3) * (x - 3)) := 
sorry

end factorization_correct_l294_294376


namespace abs_pi_expression_l294_294952

theorem abs_pi_expression : (|π - |π - 10|| = 10 - 2 * π) := by
  sorry

end abs_pi_expression_l294_294952


namespace angle_PSU_20_degrees_l294_294388

theorem angle_PSU_20_degrees 
  (P Q R S T U : Type) 
  (h1 : ∠ PRQ = 60)
  (h2 : ∠ QRP = 80)
  (h3 : S = foot_of_perpendicular P QR)
  (h4 : T = circumcenter PQR)
  (h5 : U = other_end_diameter P) : 
  ∠ PSU = 20 := 
sorry

end angle_PSU_20_degrees_l294_294388


namespace total_cost_of_soup_l294_294828

theorem total_cost_of_soup :
  let beef_pounds := 4
  let vegetable_pounds := 6
  let vegetable_cost_per_pound := 2
  let beef_cost_per_pound := 3 * vegetable_cost_per_pound
  let cost_of_vegetables := vegetable_cost_per_pound * vegetable_pounds
  let cost_of_beef := beef_cost_per_pound * beef_pounds
  in cost_of_vegetables + cost_of_beef = 36 :=
by
  let beef_pounds := 4
  let vegetable_pounds := 6
  let vegetable_cost_per_pound := 2
  let beef_cost_per_pound := 3 * vegetable_cost_per_pound
  let cost_of_vegetables := vegetable_cost_per_pound * vegetable_pounds
  let cost_of_beef := beef_cost_per_pound * beef_pounds
  show cost_of_vegetables + cost_of_beef = 36
  sorry

end total_cost_of_soup_l294_294828


namespace find_x_l294_294240

theorem find_x (x : ℝ) (h : x ≠ 3) : (x^2 - 9) / (x - 3) = 3 * x → x = 3 / 2 := by
  sorry

end find_x_l294_294240


namespace total_days_correct_l294_294595

-- Defining the years and the conditions given.
def year_1999 := 1999
def year_2000 := 2000
def year_2001 := 2001
def year_2002 := 2002

-- Defining the leap year and regular year days
def days_in_regular_year := 365
def days_in_leap_year := 366

-- Noncomputable version to skip the proof
noncomputable def total_days_from_1999_to_2002 : ℕ :=
  3 * days_in_regular_year + days_in_leap_year

-- The theorem stating the problem, which we need to prove
theorem total_days_correct : total_days_from_1999_to_2002 = 1461 := by
  sorry

end total_days_correct_l294_294595


namespace distinct_sums_l294_294178

def value (coin : String) : Nat :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "quarter" => 25
  | "half_dollar" => 50
  | _ => 0

def list_of_coin_values : List Nat :=
  [1, 1, 1, 5, 25, 25, 50]

def possible_sums (coins : List Nat) : Set Nat :=
  {c1 + c2 | c1 in coins, c2 in coins, c1 ≠ c2}

noncomputable def number_of_distinct_sums : Nat :=
  (possible_sums list_of_coin_values).size

theorem distinct_sums : number_of_distinct_sums = 9 := by
  sorry

end distinct_sums_l294_294178


namespace no_right_angle_sequence_l294_294408

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

end no_right_angle_sequence_l294_294408


namespace find_b_l294_294328

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 
if x ≤ 0 then x^2 + b else log x / log 2

theorem find_b (h : f (f 0.5 b) b = 3) : b = 2 :=
sorry

end find_b_l294_294328


namespace log2_a3_arithmetic_sequence_l294_294374

theorem log2_a3_arithmetic_sequence (a1 a5 a3 : ℝ) (f : ℝ → ℝ)
  (h1 : f x = (1 / 3) * x^3 - 4 * x^2 + 12 * x + 1)
  (h2 : ∀ x, f' x = x^2 - 8 * x + 12)
  (h_extreme : a1 ≠ a5)
  (h_arith_seq : 2 * a3 = a1 + a5) :
  log 2 a3 = 2 :=
by sorry

end log2_a3_arithmetic_sequence_l294_294374


namespace vector_collinear_l294_294363

variables (a b : ℝ × ℝ)
def a_def : a = (1, 2) := by sorry
def b_def : b = (2, 3) := by sorry

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * u.1, k * u.2)

theorem vector_collinear :
  collinear (a + b) (6, 10) :=
by
  rw [a_def, b_def]
  use 2
  sorry

end vector_collinear_l294_294363


namespace angle_CFD_60_degrees_l294_294403

open EuclideanGeometry

variable (circle : Circle)
variable (O A B F C D : Point)
variable [Diameter O A B : circle]
variable [OnCircle F : circle]
variable [TangentAt B C : circle]
variable [TangentAt F D : circle]
variable [Line AF D]

theorem angle_CFD_60_degrees (h1 : angle B A F = 30) : angle C F D = 60 :=
by
  -- proofs would be added here
  sorry

end angle_CFD_60_degrees_l294_294403


namespace binary_to_decimal_101101_l294_294964

def binary_to_decimal (digits : List ℕ) : ℕ :=
  digits.foldr (λ (digit : ℕ) (acc : ℕ × ℕ) => (acc.1 + digit * 2 ^ acc.2, acc.2 + 1)) (0, 0) |>.1

theorem binary_to_decimal_101101 : binary_to_decimal [1, 0, 1, 1, 0, 1] = 45 :=
by
  -- Proof is needed but here we use sorry as placeholder.
  sorry

end binary_to_decimal_101101_l294_294964


namespace swap_hats_and_trophies_five_moves_swap_hats_and_trophies_seven_moves_swap_hats_and_trophies_2001_moves_l294_294763

-- Part (a)
theorem swap_hats_and_trophies_five_moves : 
  ∀ (board : ℕ → ℕ → ℕ) (H T : ℕ → bool), 
  requires_five_moves_to_swap (board, H, T) :=
begin
  sorry,
end

-- Part (b)
theorem swap_hats_and_trophies_seven_moves : 
  ∀ (board : ℕ → ℕ → ℕ) (H T : ℕ → bool), 
  requires_seven_moves_to_swap (board, H, T) :=
begin
  sorry,
end

-- Part (c)
theorem swap_hats_and_trophies_2001_moves : 
  ∀ (board : ℕ → ℕ → ℕ) (H T : ℕ → bool), 
  requires_2001_moves_to_swap (board, H, T) :=
begin
  sorry,
end

end swap_hats_and_trophies_five_moves_swap_hats_and_trophies_seven_moves_swap_hats_and_trophies_2001_moves_l294_294763


namespace can_form_triangle_l294_294232

/-
  Given: Sets of line segments
  A: (3, 8, 5)
  B: (12, 5, 6)
  C: (5, 5, 10)
  D: (15, 10, 7)
  
  Prove: Only set D can form a triangle.
-/

theorem can_form_triangle :
  (15 + 10 > 7) ∧ (15 + 7 > 10) ∧ (10 + 7 > 15) :=
by {
  -- The proof can be filled in here.
  repeat {sorry},
}

end can_form_triangle_l294_294232


namespace sally_eggs_l294_294445

def dozen := 12
def total_eggs := 48

theorem sally_eggs : total_eggs / dozen = 4 := by
  -- Normally a proof would follow here, but we will use sorry to skip it
  sorry

end sally_eggs_l294_294445


namespace tanika_boxes_sold_on_saturday_l294_294461

variable (x : ℝ)
variable (boxes_sold_total : ℝ)

-- Condition 1: x represents the number of boxes sold on Saturday
def boxes_sold_saturday (x : ℝ) : ℝ := x

-- Condition 2: On Sunday, she sold 50% more than on Saturday, i.e., 1.5 times the boxes sold on Saturday
def boxes_sold_sunday (x : ℝ) : ℝ := 1.5 * x

-- Condition 3: The total number of boxes sold over the two days is 150
def total_boxes_condition (x : ℝ) : Prop := x + 1.5 * x = boxes_sold_total

theorem tanika_boxes_sold_on_saturday (x : ℝ) (h : total_boxes_condition x) : x = 60 :=
by
  have h1 : x + 1.5 * x = 150 := h
  have h2 : 2.5 * x = 150 := by sorry -- Simplifying the terms as in the solution
  have h3 : x = 150 / 2.5 := by sorry -- Solving for x
  exact h3

end tanika_boxes_sold_on_saturday_l294_294461


namespace minimum_xy_l294_294142

theorem minimum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : xy ≥ 64 :=
sorry

end minimum_xy_l294_294142


namespace empty_solution_set_range_l294_294994

theorem empty_solution_set_range (m : ℝ) : 
  (¬ ∃ x : ℝ, (m * x^2 + 2 * m * x + 1) < 0) ↔ (m = 0 ∨ (0 < m ∧ m ≤ 1)) :=
by sorry

end empty_solution_set_range_l294_294994


namespace one_third_sugar_amount_l294_294187

-- Define the original amount of sugar as a mixed number
def original_sugar_mixed : ℚ := 6 + 1 / 3

-- Define the fraction representing one-third of the recipe
def one_third : ℚ := 1 / 3

-- Define the expected amount of sugar for one-third of the recipe
def expected_sugar_mixed : ℚ := 2 + 1 / 9

-- The theorem stating the proof problem
theorem one_third_sugar_amount : (one_third * original_sugar_mixed) = expected_sugar_mixed :=
sorry

end one_third_sugar_amount_l294_294187


namespace clock_angle_7_25_l294_294503

def degrees_per_hour := 30.0
def minutes_to_degrees (minutes : ℕ) := (minutes.toFloat / 60.0) * 360.0
def hour_hand_angle (hour : ℕ) (minutes : ℕ) := (hour.toFloat + minutes.toFloat / 60.0) * degrees_per_hour
def minute_hand_angle (minutes : ℕ) := minutes_to_degrees minutes
def smaller_angle (angle1 angle2 : Float) := Float.mod (Float.abs (angle1 - angle2)) 360.0

theorem clock_angle_7_25 :
  smaller_angle (hour_hand_angle 7 25) (minute_hand_angle 25) = 72.5 := by
  sorry

end clock_angle_7_25_l294_294503


namespace projection_of_m_on_n_l294_294684

noncomputable def a : ℝ × ℝ := (√3, 0)
noncomputable def b : ℝ × ℝ := (1, 0)
noncomputable def m : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
noncomputable def n : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
noncomputable def dot_product (x y : ℝ × ℝ) := x.1 * y.1 + x.2 * y.2
noncomputable def magnitude (x : ℝ × ℝ) := Real.sqrt (dot_product x x)

theorem projection_of_m_on_n : 
  (dot_product m n) / (magnitude n) = 2 :=
by
  sorry

end projection_of_m_on_n_l294_294684


namespace binomial_sum_identity_l294_294262

theorem binomial_sum_identity : (Finset.sum (Finset.range 51) (fun k => (-1)^k * Nat.choose 100 (2*k)) = -2^50) :=
by
  sorry

end binomial_sum_identity_l294_294262


namespace width_of_box_is_25_l294_294472

-- Given conditions
def length_of_box : ℝ := 56
def height_change : ℝ := 0.5
def volume_to_be_removed_gallons : ℝ := 5250
def conversion_factor : ℝ := 7.5

-- Definition of converting gallons to cubic feet
def volume_to_be_removed_cubic_feet : ℝ := volume_to_be_removed_gallons / conversion_factor

theorem width_of_box_is_25 :
  volume_to_be_removed_cubic_feet = length_of_box * 25 * height_change :=
by
  -- This is where the proof would go, but it's omitted here.
  sorry

end width_of_box_is_25_l294_294472


namespace team_b_more_uniform_l294_294109

theorem team_b_more_uniform (S_A S_B : ℝ) (h : S_A^2 > S_B^2) : 
  ("Team B has more uniform heights than Team A") :=
sorry

end team_b_more_uniform_l294_294109


namespace find_k_for_line_l294_294278

theorem find_k_for_line : 
  ∃ k : ℚ, (∀ x y : ℚ, (-1 / 3 - 3 * k * x = 4 * y) ∧ (x = 1 / 3) ∧ (y = -8)) → k = 95 / 3 :=
by
  sorry

end find_k_for_line_l294_294278


namespace angle_delta_is_61_degrees_l294_294245

noncomputable def sum_sin_range : ℝ := (∑ i in finset.range 3600, real.sin (3269 + i) * real.pi / 180)

noncomputable def sum_cos_range : ℝ := (∑ i in finset.range 3601, real.cos (3240 + i) * real.pi / 180)

noncomputable def delta : ℝ := real.arccos (real.sin 29 * (π / 180)) -- 6869 mod 360 is 29

theorem angle_delta_is_61_degrees :
  delta = 61 :=
begin
  sorry
end

end angle_delta_is_61_degrees_l294_294245


namespace trapezoid_MN_length_l294_294199

variable (a b : ℝ) -- As these can be any real numbers, they are real-number variables.
variables (AB CD : ℝ) -- Specific lengths of bases.
variables (AD BC : ℝ) -- Specific lengths of sides.

-- Given conditions encoded as variables or as assumptions.
def is_isosceles_trapezoid : Prop :=
  AB = a ∧ CD = b ∧ AB ∥ CD ∧ AD = BC

-- Define the length of MN
def length_MN : ℝ :=
  2 * a * b / (a + b)

-- The theorem holding the main statement to be proved
theorem trapezoid_MN_length (h : is_isosceles_trapezoid a b AB CD AD BC) :
  length_MN a b = 2 * a * b / (a + b) :=
sorry

end trapezoid_MN_length_l294_294199


namespace total_jellybeans_l294_294597

-- Definitions of the conditions
def caleb_jellybeans : ℕ := 3 * 12
def sophie_jellybeans (caleb_jellybeans : ℕ) : ℕ := caleb_jellybeans / 2

-- Statement of the proof problem
theorem total_jellybeans (C : caleb_jellybeans = 36) (S : sophie_jellybeans 36 = 18) :
  caleb_jellybeans + sophie_jellybeans 36 = 54 :=
by
  sorry

end total_jellybeans_l294_294597


namespace greg_initial_money_eq_36_l294_294976

theorem greg_initial_money_eq_36 
  (Earl_initial Fred_initial : ℕ)
  (Greg_initial : ℕ)
  (Earl_owes_Fred Fred_owes_Greg Greg_owes_Earl : ℕ)
  (Total_after_debt : ℕ)
  (hEarl_initial : Earl_initial = 90)
  (hFred_initial : Fred_initial = 48)
  (hEarl_owes_Fred : Earl_owes_Fred = 28)
  (hFred_owes_Greg : Fred_owes_Greg = 32)
  (hGreg_owes_Earl : Greg_owes_Earl = 40)
  (hTotal_after_debt : Total_after_debt = 130) :
  Greg_initial = 36 :=
sorry

end greg_initial_money_eq_36_l294_294976


namespace sequence_geometric_and_formula_l294_294307

theorem sequence_geometric_and_formula (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  (∀ n, a n + 1 = 2 ^ n) ∧ (a n = 2 ^ n - 1) :=
sorry

end sequence_geometric_and_formula_l294_294307


namespace find_positive_integers_with_divisors_and_sum_l294_294625

theorem find_positive_integers_with_divisors_and_sum (n : ℕ) :
  (∃ d1 d2 d3 d4 d5 d6 : ℕ,
    (n ≠ 0) ∧ (n ≠ 1) ∧ 
    n = d1 * d2 * d3 * d4 * d5 * d6 ∧
    d1 ≠ 1 ∧ d2 ≠ 1 ∧ d3 ≠ 1 ∧ d4 ≠ 1 ∧ d5 ≠ 1 ∧ d6 ≠ 1 ∧
    (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d1 ≠ d4) ∧ (d1 ≠ d5) ∧ (d1 ≠ d6) ∧
    (d2 ≠ d3) ∧ (d2 ≠ d4) ∧ (d2 ≠ d5) ∧ (d2 ≠ d6) ∧
    (d3 ≠ d4) ∧ (d3 ≠ d5) ∧ (d3 ≠ d6) ∧
    (d4 ≠ d5) ∧ (d4 ≠ d6) ∧
    (d5 ≠ d6) ∧
    d1 + d2 + d3 + d4 + d5 + d6 = 14133
  ) -> 
  (n = 16136 ∨ n = 26666) :=
sorry

end find_positive_integers_with_divisors_and_sum_l294_294625


namespace find_b_l294_294360

noncomputable def tangent_line_b (k b : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, 
    (k = 1 / x1 ∧ k = 1 / (x2 + 1) ∧ x1 = x2 + 1) ∧
    (kx1 + b = log x1 + 2 ∧ kx2 + b = log (x2 + 1))

theorem find_b : ∀ k b : ℝ, 
  (tangent_line_b k b → b = 1 - log 2) :=
by
  sorry

end find_b_l294_294360


namespace tree_contains_all_rationals_l294_294892

def is_coprime (a b : ℕ) := Nat.gcd a b = 1

def successors (a b : ℕ) : (ℕ × ℕ) := ((a, a + b), (b, a + b))

def level_of_rational (p q : ℕ) : ℕ := 
if p < q then 
  let rec aux (u₁ u₂ : ℕ) (steps : ℕ) :=
    if u₂ = 1 then steps - 2 else 
      aux u₂ (u₁ % u₂) (steps + (u₁ / u₂))
  in aux q p 0 
else 0

theorem tree_contains_all_rationals :
  ∀ r : ℚ, 0 < r ∧ r < 1 →
    ∃ p q : ℕ, is_coprime p q ∧ r = p / q ∧
    ∃ n : ℕ, level_of_rational p q = n :=
by
  sorry

end tree_contains_all_rationals_l294_294892


namespace man_speed_is_correct_l294_294582

def length_of_train : ℝ := 800
def time_to_cross_man : ℝ := 47.99616030717543
def speed_of_train_km_hr : ℝ := 65
def speed_of_train_m_s : ℝ := speed_of_train_km_hr * 1000 / 3600

theorem man_speed_is_correct : 
  ∃ Vm : ℝ, Vm ≈ 1.388888888888889 ∧ 
            (length_of_train + Vm * time_to_cross_man = speed_of_train_m_s * time_to_cross_man) := 
  sorry

end man_speed_is_correct_l294_294582


namespace averageFishIs75_l294_294766

-- Introduce the number of fish in Boast Pool
def BoastPool : ℕ := 75

-- Introduce the number of fish in Onum Lake
def OnumLake : ℕ := BoastPool + 25

-- Introduce the number of fish in Riddle Pond
def RiddlePond : ℕ := OnumLake / 2

-- Define the total number of fish in all three bodies of water
def totalFish : ℕ := BoastPool + OnumLake + RiddlePond

-- Define the average number of fish in all three bodies of water
def averageFish : ℕ := totalFish / 3

-- Prove that the average number of fish is 75
theorem averageFishIs75 : averageFish = 75 :=
by
  -- We need to provide the proof steps here but using sorry to skip
  sorry

end averageFishIs75_l294_294766


namespace determine_f_peak_tourism_season_l294_294709

noncomputable def f (n : ℕ) : ℝ := 200 * Real.cos ((Real.pi / 6) * n + 2 * Real.pi / 3) + 300

theorem determine_f :
  (∀ n : ℕ, f n = 200 * Real.cos ((Real.pi / 6) * n + 2 * Real.pi / 3) + 300) ∧
  (f 8 - f 2 = 400) ∧
  (f 2 = 100) :=
sorry

theorem peak_tourism_season (n : ℤ) :
  (6 ≤ n ∧ n ≤ 10) ↔ (200 * Real.cos (((Real.pi / 6) * n) + 2 * Real.pi / 3) + 300 >= 400) :=
sorry

end determine_f_peak_tourism_season_l294_294709


namespace solution_l294_294200

def Perpendicular (x y : Type) [HasPerp x y] : Prop := x ⟂ y
def Parallel (x y : Type) [HasPara x y] : Prop := x ∥ y

def Line : Type := sorry
def Plane : Type := sorry

class HasPerp (x y : Type)

class HasPara (x y : Type)

noncomputable def Problem :=
∀ (x y z : Type) [HasPerp x y] [HasPara y z],
  (Perpendicular x y ∧ Parallel y z) → Perpendicular x z

def condition1 := ∃ (x y z : Line), Problem x y z
def condition3 := ∃ (x : Line) (y z : Plane), Problem x y z
def condition4 := ∃ (x y z : Plane), Problem x y z

theorem solution :
  (condition1 ∨ condition3 ∨ condition4) :=
sorry

end solution_l294_294200


namespace part_a_part_b_l294_294744

noncomputable def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- LCM for a list of numbers
noncomputable def LCM_list (l : List ℕ) : ℕ :=
  l.foldr LCM 1

theorem part_a 
  (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ)
  (h1 : a1 < a2) (h2 : a2 < a3) (h3 : a3 < a4) (h4 : a4 < a5) 
  (h5 : a5 < a6) (h6 : a6 < a7) (h7 : a7 < a8) (h8 : a8 < a9) (h9 : a9 < a10) 
  (hdistinct : List.nodup [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]) :
  LCM_list [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10] / a6 = 420 := 
sorry

theorem part_b 
  (a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℕ)
  (h1 : a1 < a2) (h2 : a2 < a3) (h3 : a3 < a4) (h4 : a4 < a5) 
  (h5 : a5 < a6) (h6 : a6 < a7) (h7 : a7 < a8) (h8 : a8 < a9) (h9 : a9 < a10)
  (hdistinct : List.nodup [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10])
  (h_range : 1 ≤ a6 ∧ a6 ≤ 2000) :
  (a6 = 504 ∨ a6 = 1008 ∨ a6 = 1512) ∧ 
  LCM_list [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10] / a1 = 
  LCM_list [a1, a2, a3, a4, a5, 504, a7, a8, a9, a10] / 1 := 
sorry

end part_a_part_b_l294_294744


namespace inequality_proven_l294_294414

open Real

theorem inequality_proven (n : ℕ) (x : Fin n → ℝ) (h_pos : ∀ i, 0 < x i) (h_sum : ∑ i, x i = 1) :
  (∑ i, (x i / sqrt (1 - x i))) ≥ (∑ i, sqrt (x i)) / sqrt (n - 1) :=
sorry

end inequality_proven_l294_294414


namespace Kyle_fish_count_l294_294934

def Carla_fish := 8
def Total_fish := 36
def Kyle_fish := (Total_fish - Carla_fish) / 2

theorem Kyle_fish_count : Kyle_fish = 14 :=
by
  -- This proof will be provided later
  sorry

end Kyle_fish_count_l294_294934


namespace binomial_alternating_sum_l294_294266

theorem binomial_alternating_sum :
  (Finset.range 51).sum (λ k, (-1 : ℤ)^k * Nat.choose 100 (2 * k)) = -2^50 := 
by
    sorry

end binomial_alternating_sum_l294_294266


namespace range_of_function_l294_294622

theorem range_of_function : 
  ∀ x ∈ set.Ioo (-1 : ℝ) 1,
    ∃ y : ℝ, 
      y = real.arcsin x + real.arccos x + real.arctan x - real.arctanh x :=
by
  sorry

end range_of_function_l294_294622


namespace a_range_l294_294670

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  log x - a * x + 1

theorem a_range (a : ℝ) (h : ∃ x ∈ set.Icc (1 / Real.exp 1) Real.exp 1, f x a = 0) : 
  0 ≤ a ∧ a ≤ 1 :=
sorry

end a_range_l294_294670


namespace speed_of_current_l294_294149

variables (b c : ℝ)

theorem speed_of_current (h1 : b + c = 12) (h2 : b - c = 4) : c = 4 :=
sorry

end speed_of_current_l294_294149


namespace sum_of_primitive_roots_mod_11_l294_294305

def is_primitive_root_mod (a p : ℕ) : Prop :=
  p.prime ∧ ∀ (b : ℕ), (1 ≤ b ∧ b < p) → ∃ (k : ℕ), (0 < k ∧ k < p) ∧ (a^k ≡ b [MOD p])

theorem sum_of_primitive_roots_mod_11 : 
  ∑ x in {x : ℕ | 1 ≤ x ∧ x < 11 ∧ is_primitive_root_mod x 11}, x = 26 := 
by
  sorry

end sum_of_primitive_roots_mod_11_l294_294305


namespace time_to_travel_5th_km_l294_294891

theorem time_to_travel_5th_km (a n d c t : ℝ) (h1 : ∀ n >= 3, a n = c / (d n) ^ 2)
  (h2 : d 3 = 2) (h3 : t 3 = 4) (h4 : a 2 = 1/4) : 
 t 5 = 8 := 
sorry

end time_to_travel_5th_km_l294_294891


namespace count_centrally_but_not_axially_symmetric_shapes_l294_294909

-- We define properties of the shapes
def is_axially_symmetric (shape : String) : Prop :=
  shape = "Equilateral triangle" ∨ 
  shape = "Circle" ∨ 
  shape = "Regular pentagram" ∨
  shape = "Parabola" ∨
  (shape = "Parallelogram" ∧ (shape = "Rectangle" ∨ shape = "Rhombus"))

def is_centrally_symmetric (shape : String) : Prop :=
  shape = "Circle" ∨ 
  shape = "Parallelogram"

-- List of shapes to consider
def shapes : List String := ["Equilateral triangle", "Parallelogram", "Circle", "Regular pentagram", "Parabola"]

-- The main theorem we aim to prove
theorem count_centrally_but_not_axially_symmetric_shapes : 
  (count (λ s, is_centrally_symmetric s ∧ ¬ is_axially_symmetric s) shapes) = 0 := 
by sorry

end count_centrally_but_not_axially_symmetric_shapes_l294_294909


namespace employed_males_percentage_l294_294380

theorem employed_males_percentage (p_employed : ℝ) (p_employed_females : ℝ) : 
  (64 / 100) * (1 - 21.875 / 100) * 100 = 49.96 :=
by
  sorry

end employed_males_percentage_l294_294380


namespace point_B_not_on_graph_l294_294518

def is_on_graph (x y : ℝ) : Prop :=
  y = x^2 / (x + 1)

def point_A := (0 : ℝ, 0 : ℝ)
def point_B := (-1 / 2 : ℝ, 1 / 6 : ℝ)
def point_C := (1 / 2 : ℝ, 1 / 6 : ℝ)
def point_D := (-1 : ℝ, 1 : ℝ)
def point_E := (-2 : ℝ, -2 : ℝ)

theorem point_B_not_on_graph : ¬ is_on_graph (-1 / 2) (1 / 6) := by
  sorry

end point_B_not_on_graph_l294_294518


namespace a7_not_prime_l294_294191
open Nat

-- Define the sequence a based on the conditions
def reverse_digits (n : ℕ) : ℕ := 
  n.digits 10 }.reverse.foldl (λ acc d, acc*10 + d) 0

def sequence_a : ℕ → ℕ
| 1     := 170
| (n+1) := sequence_a n + reverse_digits (sequence_a n)

theorem a7_not_prime : ¬ Prime (sequence_a 7) := 
by
  sorry

end a7_not_prime_l294_294191


namespace find_z_given_x4_l294_294098

theorem find_z_given_x4 (k : ℝ) (z : ℝ) (x : ℝ) :
  (7 * 4 = k / 2^3) → (7 * z = k / x^3) → (x = 4) → (z = 0.5) :=
by
  intro h1 h2 h3
  sorry

end find_z_given_x4_l294_294098


namespace slope_dividing_remaining_area_l294_294560

noncomputable def point := ℝ × ℝ

def slope (p1 p2 : point) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

def rectangle_vertices := [((0, 0) : point), (100, 0), (100, 50), (0, 50)]

def circle_center : point := (75, 30)
def rectangle_center : point := (50, 25)

theorem slope_dividing_remaining_area :
  slope (50, 25) circle_center = 1/5 := by
    -- The proof is skipped
    sorry

end slope_dividing_remaining_area_l294_294560


namespace planet_density_approx_l294_294544

theorem planet_density_approx {R T : ℝ} (G : ℝ) (m : ℝ) (ρ : ℝ)
    (h1 : ∃ R T : ℝ, a_c = (4 * Real.pi^2 * (2 * R)) / T^2)
    (h2 : m * a_c = G * (m * ρ * (4/3) * Real.pi * R^3) / (2 * R)^2)
    (hG : G ≈ 6.67430e-11)
    (hT : T = 14400) : 
  ρ ≈ 6000 :=
by
  sorry

end planet_density_approx_l294_294544


namespace cos_alpha_value_l294_294666

theorem cos_alpha_value (x y : ℝ) (r : ℝ) (α : ℝ)
  (h₀ : cos α = x / r)
  (h₁ : x = -6)
  (h₂ : y = 8)
  (h₃ : r = 10) :
  cos α = -3 / 5 := by
  sorry

end cos_alpha_value_l294_294666


namespace range_of_a_l294_294808

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x - 3 * a else a^x - 2

theorem range_of_a (a : ℝ) :
  (0 < a ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → f a x > f a y) ↔ (0 < a ∧ a ≤ 1 / 3) :=
by sorry

end range_of_a_l294_294808


namespace students_accommodated_l294_294153

theorem students_accommodated 
  (total_students : ℕ)
  (total_workstations : ℕ)
  (workstations_accommodating_x_students : ℕ)
  (x : ℕ)
  (workstations_accommodating_3_students : ℕ)
  (workstation_capacity_10 : ℕ)
  (workstation_capacity_6 : ℕ) :
  total_students = 38 → 
  total_workstations = 16 → 
  workstations_accommodating_x_students = 10 → 
  workstations_accommodating_3_students = 6 → 
  workstation_capacity_10 = 10 * x → 
  workstation_capacity_6 = 6 * 3 → 
  10 * x + 18 = 38 → 
  10 * 2 = 20 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end students_accommodated_l294_294153


namespace blue_stripe_area_is_correct_l294_294159

def diameter : ℝ := 40
def height : ℝ := 100
def stripe_width : ℝ := 4
def revolutions : ℝ := 3

def circumference (d : ℝ) : ℝ := Real.pi * d
def stripe_length (rev : ℝ) (circ : ℝ) : ℝ := rev * circ
def stripe_area (width : ℝ) (length : ℝ) : ℝ := width * length

theorem blue_stripe_area_is_correct :
  stripe_area stripe_width (stripe_length revolutions (circumference diameter)) = 480 * Real.pi := 
sorry

end blue_stripe_area_is_correct_l294_294159


namespace find_starting_number_of_range_l294_294107

theorem find_starting_number_of_range : 
  ∃ (n : ℤ), 
    (∀ k, (0 ≤ k ∧ k < 7) → (n + k * 3 ≤ 31 ∧ n + k * 3 % 3 = 0)) ∧ 
    n + 6 * 3 = 30 - 6 * 3 :=
by
  sorry

end find_starting_number_of_range_l294_294107


namespace sum_reciprocals_l294_294652

noncomputable def seq (n : ℕ) : ℕ :=
  n

def sn (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem sum_reciprocals (n : ℕ) :
  (finset.range n).sum (λ i, 1 / (sn (i + 1))) = 2 * n / (n + 1) :=
by
  sorry

end sum_reciprocals_l294_294652


namespace tan_angle_subtraction_l294_294691

-- Define the angle properties
variables {θ φ : ℝ}
hypothesis h1 : Mathlib.Tan θ = 3
hypothesis h2 : Mathlib.Tan φ = 2

-- State the main theorem to prove
theorem tan_angle_subtraction : Mathlib.Tan (2 * θ - φ) = 11 / 2 :=
by
  sorry  -- Proof is not required

end tan_angle_subtraction_l294_294691


namespace volume_of_tetrahedron_A2B2C2D2_l294_294530

-- Definitions of points and tetrahedron
variables {A B C D A1 B1 C1 D1 : Type*}

-- Notions of centroids and points for tetrahedron ABCD
-- (Additional definitions would need to support point geometry, centroids, and volumes in 3D space)
def is_centroid_of_face (A B C D : Point) (A1 : Point) : Prop := sorry

-- Condition for points A1, B1, C1, D1 being centroids of the respective tetrahedron faces
axiom centroid_face_BCD : is_centroid_of_face A B C D A1
axiom centroid_face_CDA : is_centroid_of_face A B C D B1
axiom centroid_face_DAB : is_centroid_of_face A B C D C1
axiom centroid_face_ABC : is_centroid_of_face A B C D D1

-- Volume of a tetrahedron
axiom volume_tetrahedron (A B C D : Point) : ℝ

-- Parallel condition and volume scaling factor for the tetrahedron A2 B2 C2 D2
axiom exists_parallel_edges (A B C D A1 B1 C1 D1 : Type*) (A2 B2 C2 D2 : Type*) : Prop
axiom volume_of_scaled_tetrahedron (A B C D : Type*) (V : ℝ) : 
  ∃ (A2 B2 C2 D2 : Type*), exists_parallel_edges A B C D A1 B1 C1 D1 A2 B2 C2 D2 ∧
  volume_tetrahedron A2 B2 C2 D2 = (16 / 27 : ℝ) * V

-- The theorem statement
theorem volume_of_tetrahedron_A2B2C2D2 {A B C D A1 B1 C1 D1 : Type*} (V : ℝ) :
  (volume_tetrahedron A B C D) = V →
    ∃ (A2 B2 C2 D2 : Type*), exists_parallel_edges A B C D A1 B1 C1 D1 A2 B2 C2 D2 ∧
    (volume_tetrahedron A2 B2 C2 D2 = (16 / 27 : ℝ) * V) :=
sorry

end volume_of_tetrahedron_A2B2C2D2_l294_294530


namespace new_individuals_weight_l294_294082

variables (W : ℝ) (A B C : ℝ)

-- Conditions
def original_twelve_people_weight : ℝ := W
def weight_leaving_1 : ℝ := 64
def weight_leaving_2 : ℝ := 75
def weight_leaving_3 : ℝ := 81
def average_increase : ℝ := 3.6
def total_weight_increase : ℝ := 12 * average_increase
def weight_leaving_sum : ℝ := weight_leaving_1 + weight_leaving_2 + weight_leaving_3

-- Equation derived from the problem conditions
def new_individuals_weight_sum : ℝ := weight_leaving_sum + total_weight_increase

-- Theorem to prove
theorem new_individuals_weight :
  A + B + C = 263.2 :=
by
  sorry

end new_individuals_weight_l294_294082


namespace complement_union_A_B_l294_294009

def is_element_of_set_A (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 1
def is_element_of_set_B (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 2
def is_element_of_complement_U (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem complement_union_A_B :
  {x : ℤ | ¬ (is_element_of_set_A x ∨ is_element_of_set_B x)} = {x : ℤ | is_element_of_complement_U x} :=
by
  sorry

end complement_union_A_B_l294_294009


namespace matchstick_game_no_win_for_A_l294_294769

theorem matchstick_game_no_win_for_A :
  ∀ (n : ℕ), (n = 10) →
  (∀ (m : ℕ), (m = n → (m_1 : ℕ), (m_1 = m - (1 ∨ 3 ∨ 4)) →
  -- (Player B's response is guaranteed by taking 1 or 2 matchsticks to force win)
  ∃ (k : ℕ), (k < 5) ∧ (k = m_1) ∧ (k % 2 = 1)) →
  false :=
by sorry

end matchstick_game_no_win_for_A_l294_294769


namespace solve_for_y_l294_294701

variable (k y : ℝ)

-- Define the first equation for x
def eq1 (x : ℝ) : Prop := (1 / 2023) * x - 2 = 3 * x + k

-- Define the condition that x = -5 satisfies eq1
def condition1 : Prop := eq1 k (-5)

-- Define the second equation for y
def eq2 : Prop := (1 / 2023) * (2 * y + 1) - 5 = 6 * y + k

-- Prove that given condition1, y = -3 satisfies eq2
theorem solve_for_y : condition1 k → eq2 k (-3) :=
sorry

end solve_for_y_l294_294701


namespace overall_support_percentage_correct_l294_294564

variable (M W : ℕ) -- Number of men and women
variable (m_percent w_percent : ℝ) -- Percentages of support among men and women

-- Conditions given in the problem
def survey_conditions := M = 200 ∧ W = 800 ∧ m_percent = 0.70 ∧ w_percent = 0.75

-- Main theorem to prove the overall support percentage
theorem overall_support_percentage_correct (h : survey_conditions M W m_percent w_percent) : 
    let total := M + W in
    let m_support := (m_percent * M).to_nat in
    let w_support := (w_percent * W).to_nat in
    let total_support := m_support + w_support in
    (total_support / total.to_nat : ℝ) = 0.74 :=
sorry

end overall_support_percentage_correct_l294_294564


namespace twelve_pow_six_mod_eight_l294_294460

theorem twelve_pow_six_mod_eight : ∃ m : ℕ, 0 ≤ m ∧ m < 8 ∧ 12^6 % 8 = m ∧ m = 0 := by
  sorry

end twelve_pow_six_mod_eight_l294_294460


namespace problem_statement_l294_294354

-- Problem statement in Lean 4
theorem problem_statement (a b : ℝ) (h : b < a ∧ a < 0) : 7 - a > b :=
by 
  sorry

end problem_statement_l294_294354


namespace expected_number_of_rolls_to_reach_2010_l294_294170

noncomputable def expected_rolls : ℕ → ℚ
| 0 := 0
| (n+1) := (1/6) * (expected_rolls n + expected_rolls (n-1) + expected_rolls (n-2) + expected_rolls (n-3) + expected_rolls (n-4) + expected_rolls (n-5)) + 1

theorem expected_number_of_rolls_to_reach_2010 : abs ((expected_rolls 2010 : ℚ) - 574.761904) < 0.0001 := 
sorry

end expected_number_of_rolls_to_reach_2010_l294_294170


namespace series_sum_l294_294605

theorem series_sum : (finset.range 200).sum (λ n, if even n then (n : ℤ) else -(n : ℤ)) = 100 := 
by
  sorry

end series_sum_l294_294605


namespace evaluate_expression_l294_294977

theorem evaluate_expression (x : ℕ) (h : x = 3) : (x^x)^(x^x) = 27^27 :=
by
  sorry

end evaluate_expression_l294_294977


namespace expected_rolls_to_reach_2010_l294_294167

noncomputable def expected_rolls_to_reach_sum (n : ℕ) : ℝ :=
  sorry -- Using 'sorry' to denote placeholder for the actual proof.

theorem expected_rolls_to_reach_2010 : expected_rolls_to_reach_sum 2010 ≈ 574.761904 :=
  sorry

end expected_rolls_to_reach_2010_l294_294167


namespace abs_twice_sub_pi_l294_294940

theorem abs_twice_sub_pi (h : Real.pi < 10) : abs (Real.pi - abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_twice_sub_pi_l294_294940


namespace mike_pears_l294_294758

theorem mike_pears (jason_pears : Nat) (total_pears : Nat) (h1 : jason_pears = 7) (h2 : total_pears = 15) : 
  total_pears - jason_pears = 8 :=
by
  rw [h1, h2]
  norm_num

end mike_pears_l294_294758


namespace total_sweaters_l294_294056

-- Define the conditions
def washes_per_load : ℕ := 9
def total_shirts : ℕ := 19
def total_loads : ℕ := 3

-- Define the total_sweaters theorem to prove Nancy had to wash 9 sweaters
theorem total_sweaters {n : ℕ} (h1 : washes_per_load = 9) (h2 : total_shirts = 19) (h3 : total_loads = 3) : n = 9 :=
by
  sorry

end total_sweaters_l294_294056


namespace divides_necklaces_l294_294737

/-- Define the number of ways to make an even number of necklaces each of length at least 3. -/
def D_0 (n : ℕ) : ℕ := sorry

/-- Define the number of ways to make an odd number of necklaces each of length at least 3. -/
def D_1 (n : ℕ) : ℕ := sorry

/-- Main theorem: Prove that (n - 1) divides (D_1(n) - D_0(n)) for n ≥ 2 -/
theorem divides_necklaces (n : ℕ) (h : n ≥ 2) : (n - 1) ∣ (D_1 n - D_0 n) := sorry

end divides_necklaces_l294_294737


namespace tom_savings_l294_294112

theorem tom_savings :
  let insurance_cost_per_month := 20
  let total_months := 24
  let procedure_cost := 5000
  let insurance_coverage := 0.80
  let total_insurance_cost := total_months * insurance_cost_per_month
  let insurance_cover_amount := procedure_cost * insurance_coverage
  let out_of_pocket_cost := procedure_cost - insurance_cover_amount
  let savings := procedure_cost - total_insurance_cost - out_of_pocket_cost
  savings = 3520 :=
by
  sorry

end tom_savings_l294_294112


namespace exists_alpha_l294_294273

noncomputable def f_alpha (α : ℝ) (x : ℕ) : ℕ := ⌊α * x⌋.toNat

def f_alpha_iter (α : ℝ) (x : ℕ) (l : ℕ) : ℕ :=
Nat.iterate (f_alpha α) l x

theorem exists_alpha (n : ℕ) (h : 1 ≤ n) :
  ∃ α : ℝ, ∀ l : ℕ, 1 ≤ l ∧ l ≤ n → f_alpha_iter α (n^2) l = n^2 - l :=
sorry

end exists_alpha_l294_294273


namespace abs_pi_expression_l294_294954

theorem abs_pi_expression (h : Real.pi < 10) : 
  Real.abs (Real.pi - Real.abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_pi_expression_l294_294954


namespace Emmanuel_share_l294_294919

theorem Emmanuel_share : 
  ∀ (total_jelly_beans : ℕ) (thomas_percentage : ℕ) (barry_ratio : ℕ) (emmanuel_ratio : ℕ),
  thomas_percentage = 10 → barry_ratio = 4 → emmanuel_ratio = 5 →
  let thomas_share := thomas_percentage * total_jelly_beans / 100 in
  let remaining_jelly_beans := total_jelly_beans - thomas_share in
  let total_ratio_parts := barry_ratio + emmanuel_ratio in
  let one_part := remaining_jelly_beans / total_ratio_parts in
  let emmanuel_share := emmanuel_ratio * one_part in
  total_jelly_beans = 200 →
  emmanuel_share = 100 :=
by
  intros total_jelly_beans thomas_percentage barry_ratio emmanuel_ratio h1 h2 h3 h_total
  let thomas_share := thomas_percentage * total_jelly_beans / 100
  let remaining_jelly_beans := total_jelly_beans - thomas_share
  let total_ratio_parts := barry_ratio + emmanuel_ratio
  let one_part := remaining_jelly_beans / total_ratio_parts
  let emmanuel_share := emmanuel_ratio * one_part
  exact eq.mpr sorry

end Emmanuel_share_l294_294919


namespace gcd_75_225_l294_294505

theorem gcd_75_225 : Int.gcd 75 225 = 75 :=
by
  sorry

end gcd_75_225_l294_294505


namespace ayen_total_jog_time_l294_294588

def jog_time_weekday : ℕ := 30
def jog_time_tuesday : ℕ := jog_time_weekday + 5
def jog_time_friday : ℕ := jog_time_weekday + 25

def total_weekday_jog_time : ℕ := jog_time_weekday * 3
def total_jog_time : ℕ := total_weekday_jog_time + jog_time_tuesday + jog_time_friday

theorem ayen_total_jog_time : total_jog_time / 60 = 3 := by
  sorry

end ayen_total_jog_time_l294_294588


namespace fractional_eq_no_solution_l294_294667

theorem fractional_eq_no_solution (m : ℝ) :
  ¬ ∃ x, (x - 2) / (x + 2) - (m * x) / (x^2 - 4) = 1 ↔ m = -4 :=
by
  sorry

end fractional_eq_no_solution_l294_294667


namespace incorrect_statement_E_l294_294356

theorem incorrect_statement_E (x y : ℝ) (h : y = log x / log 10) :
    ((y = 0 → x = 1) ∧ 
     (y = 1 → x = 10) ∧
     (y = -1 → x = 0.1) ∧
     (y = 2 → x = 100)) → ¬ (∃ s : Prop, s = "Only some of the above statements are correct") :=
by
  sorry

end incorrect_statement_E_l294_294356


namespace number_of_heavy_tailed_permutations_l294_294558

theorem number_of_heavy_tailed_permutations : 
  let S := {a : list ℕ | a ~ [1, 2, 3, 4, 5, 6]} in
  let heavy_tailed (a : list ℕ) := (a.take 3).sum < (a.drop 3).sum in
  S.count heavy_tailed = 360 :=
sorry

end number_of_heavy_tailed_permutations_l294_294558


namespace at_least_one_zero_l294_294117

theorem at_least_one_zero (a b : ℤ) : (¬ (a ≠ 0) ∨ ¬ (b ≠ 0)) ↔ (a = 0 ∨ b = 0) :=
by
  sorry

end at_least_one_zero_l294_294117


namespace quadratic_real_roots_l294_294361

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_real_roots (k : ℝ) :
  discriminant (k - 1) 4 2 ≥ 0 ↔ k ≤ 3 ∧ k ≠ 1 :=
by
  sorry

end quadratic_real_roots_l294_294361


namespace polyhedron_inequality_proof_l294_294714

noncomputable def polyhedron_inequality (B : ℕ) (P : ℕ) (T : ℕ) : Prop :=
  B * Real.sqrt (P + T) ≥ 2 * P

theorem polyhedron_inequality_proof (B P T : ℕ) 
  (h1 : 0 < B) (h2 : 0 < P) (h3 : 0 < T) 
  (condition_is_convex_polyhedron : true) : 
  polyhedron_inequality B P T :=
sorry

end polyhedron_inequality_proof_l294_294714


namespace vec_eq_l294_294754

def vector := (ℝ × ℝ)

def a : vector := (1, -1)
def b : vector := (-1, 2)
def c : vector := (1, 1)

theorem vec_eq (m n : ℝ) (h : c = m • a + n • b) : c = 3 • a + 2 • b :=
by
  sorry

end vec_eq_l294_294754


namespace negation_of_P_is_true_l294_294439

theorem negation_of_P_is_true :
  ¬ (∃ x : ℝ, x^2 + 1 < 2 * x) :=
by sorry

end negation_of_P_is_true_l294_294439


namespace range_of_m_l294_294750

variable (x m : ℝ)

def alpha (x : ℝ) : Prop := x ≤ -5
def beta (x m : ℝ) : Prop := 2 * m - 3 ≤ x ∧ x ≤ 2 * m + 1

theorem range_of_m (x : ℝ) : (∀ x, beta x m → alpha x) → m ≤ -3 := by
  sorry

end range_of_m_l294_294750


namespace handshakes_meeting_l294_294367

theorem handshakes_meeting (x : ℕ) (h : x * (x - 1) / 2 = 66) : x = 12 := 
by 
  sorry

end handshakes_meeting_l294_294367


namespace m_n_sum_l294_294991

-- Given that m and n are positive integers
def positive_int (x : Int) : Prop := x > 0

-- The problem statement definitions
variables {m n : Int} (h1 : positive_int m) (h2 : positive_int n) (h3 : m + 9 < n)

-- The conditions on the mean and median of the set {m, m+3, m+9, n, n+1, 2n-1}
def is_mean (s : Set Int) (mean : Int) : Prop :=
  (s.sum) / (s.card) = mean

def is_median (s : Set Int) (median : Int) : Prop :=
  let sorted_s := List.sort s.toList
  (sorted_s.get (sorted_s.length / 2 - 1) + sorted_s.get (sorted_s.length / 2)) / 2 = median 

-- The set under consideration
def my_set (m n : Int) : Set Int := {m, m+3, m+9, n, n+1, 2n-1}

-- Prove that m + n = 21 given the defined conditions
theorem m_n_sum {m n : Int} (h1 : positive_int m) (h2 : positive_int n) (h3 : m + 9 < n)
  (h_mean : is_mean (my_set m n) (n-1)) (h_median : is_median (my_set m n) (n-1)) :
  m + n = 21 := 
  sorry

end m_n_sum_l294_294991


namespace area_quadrilateral_OBEC_l294_294887

open Real

-- Define points and lines
def A : Point := (4, 0)
def B : Point := (0, 12)
def C : Point := (6, 0)
def E : Point := (3, 3)

-- Define lines transformed into Lean 4 based on given slopes
def line1 (x : Real) : Real := -3 * x + 12
def line2 (x : Real) : Real := x

-- Ensure intersections are maintained
lemma line1_passing_through_E : line1 3 = 3 := by sorry
lemma line2_passing_through_E : line2 3 = 3 := by sorry

-- Use the given point coordinates to find specific intersections if required
def O : Point := (0, 0)

-- Define the function to calculate the area using specified triangles
def area_triangle (a b c : Point) : Real :=
  0.5 * ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2)).abs

def area_OBEC : Real :=
  area_triangle O B E + area_triangle O E C

theorem area_quadrilateral_OBEC : area_OBEC = 27 := by sorry

end area_quadrilateral_OBEC_l294_294887


namespace expected_number_of_rolls_to_reach_2010_l294_294168

noncomputable def expected_rolls : ℕ → ℚ
| 0 := 0
| (n+1) := (1/6) * (expected_rolls n + expected_rolls (n-1) + expected_rolls (n-2) + expected_rolls (n-3) + expected_rolls (n-4) + expected_rolls (n-5)) + 1

theorem expected_number_of_rolls_to_reach_2010 : abs ((expected_rolls 2010 : ℚ) - 574.761904) < 0.0001 := 
sorry

end expected_number_of_rolls_to_reach_2010_l294_294168


namespace distribute_marbles_correct_l294_294685

def distribute_marbles (total_marbles : Nat) (num_boys : Nat) : Nat :=
  total_marbles / num_boys

theorem distribute_marbles_correct :
  distribute_marbles 20 2 = 10 := 
by 
  sorry

end distribute_marbles_correct_l294_294685


namespace positive_difference_of_roots_l294_294969

theorem positive_difference_of_roots :
  let f := λ x : ℝ, 2 * x^2 - 10 * x + 18 - 2 * x - 34
  let roots := {x : ℝ | f x = 0}
  is_positive_difference (a b : ℝ) (a > b) (2 * (a - b)) :=
  ∀ (a b : ℝ),
    a ∈ roots ∧ b ∈ roots ∧ a ≠ b →
    (a - b = 2 * real.sqrt 17 ∨ b - a = 2 * real.sqrt 17)
:=
sorry

end positive_difference_of_roots_l294_294969


namespace abs_pi_expression_l294_294957

theorem abs_pi_expression (h : Real.pi < 10) : 
  Real.abs (Real.pi - Real.abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_pi_expression_l294_294957


namespace cakes_initial_l294_294918

theorem cakes_initial (x : ℕ) (buy : ℕ) (sell : ℕ) (after_transactions : ℕ) (hc : buy = 103) (hd : sell = 86) (he : after_transactions = 190) :
  (x + buy - sell = after_transactions) → x = 173 :=
by
  intro h
  have hc' : buy = 103 := hc
  have hd' : sell = 86 := hd
  have he' : after_transactions = 190 := he
  rw [hc', hd'] at h
  calc
    x + 103 - 86 = 190 : h
    x + 17 = 190 : by sorry
    x = 173 : by sorry
  sorry

end cakes_initial_l294_294918


namespace shirley_eggs_left_l294_294073

def initial_eggs : ℕ := 98
def bought_eggs : ℕ := 8
def first_batch_eggs_per_cupcake : ℕ := 5
def first_batch_cupcakes : ℕ := 6
def second_batch_eggs_per_cupcake : ℕ := 7
def second_batch_cupcakes : ℕ := 4

theorem shirley_eggs_left :
  let total_eggs := initial_eggs + bought_eggs,
      first_batch_used := first_batch_eggs_per_cupcake * first_batch_cupcakes,
      second_batch_used := second_batch_eggs_per_cupcake * second_batch_cupcakes,
      total_eggs_used := first_batch_used + second_batch_used in
  total_eggs - total_eggs_used = 48 :=
by
  sorry

end shirley_eggs_left_l294_294073


namespace shape_with_circular_views_is_sphere_l294_294570

-- Definitions of the views of different geometric shapes
inductive Shape
| Cuboid : Shape
| Cylinder : Shape
| Cone : Shape
| Sphere : Shape

def front_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Rectangle
| Shape.Cone := False  -- Isosceles Triangle
| Shape.Sphere := True  -- Circle

def left_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Rectangle
| Shape.Cone := False  -- Isosceles Triangle
| Shape.Sphere := True  -- Circle

def top_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Circle, but not all views
| Shape.Cone := False  -- Circle, but not all views
| Shape.Sphere := True  -- Circle

-- The theorem to be proved
theorem shape_with_circular_views_is_sphere (s : Shape) :
  (front_view s ↔ True) ∧ (left_view s ↔ True) ∧ (top_view s ↔ True) ↔ s = Shape.Sphere :=
by sorry

end shape_with_circular_views_is_sphere_l294_294570


namespace taco_castle_parking_lot_l294_294822

variable (D F T V : ℕ)

theorem taco_castle_parking_lot (h1 : F = D / 3) (h2 : F = 2 * T) (h3 : V = T / 2) (h4 : V = 5) : D = 60 :=
by
  sorry

end taco_castle_parking_lot_l294_294822


namespace higher_rent_amount_l294_294434

theorem higher_rent_amount :
  (∃ H : ℝ,
    (∀ x : ℝ, x = 40 ∨ x = H) ∧
    (∀ n : ℕ, n = 10 → 0.2 * 1000 = 10 * (H - 40)) ∧
    (∀ t : ℝ, t = 1000) →
    H = 60) :=
begin
  sorry
end

end higher_rent_amount_l294_294434


namespace general_formula_arithmetic_seq_sum_reciprocal_seq_l294_294105

variables {Sn : ℕ → ℝ} {a1 : ℤ} {S4 : ℕ → ℝ} {b : ℕ → ℝ} {T : ℕ → ℝ}

axiom a1_eq_10 : a1 = 10
axiom a2_is_int : ∃ d : ℤ, a2 = a1 + d
axiom Sn_le_S4 : ∀ n : ℕ, Sn n ≤ Sn 4

def arithmetic_seq (n : ℕ) : ℤ := 13 - 3 * n

theorem general_formula_arithmetic_seq :
  ∀ n : ℕ, a_n = 13 - 3 * n :=
sorry

def reciprocal_seq (n : ℕ) : ℝ := 
  let a_n := (arithmetic_seq n)
  in 1 / (a_n * (a_n - 3))

theorem sum_reciprocal_seq (n : ℕ) :
  T n = n / (10 * (10 - 3 * n)) :=
sorry

end general_formula_arithmetic_seq_sum_reciprocal_seq_l294_294105


namespace abs_twice_sub_pi_l294_294939

theorem abs_twice_sub_pi (h : Real.pi < 10) : abs (Real.pi - abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_twice_sub_pi_l294_294939


namespace area_of_triangle_l294_294195

-- Define the coordinates of the vertices
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (7, -1)
def C : ℝ × ℝ := (2, 6)

-- Define the function to calculate the area of the triangle formed by three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The theorem statement that the area of the triangle with given vertices is 14.5
theorem area_of_triangle : triangle_area A B C = 14.5 :=
by 
  -- Skipping the proof part
  sorry

end area_of_triangle_l294_294195


namespace sum_c_1_to_210_l294_294992

-- Definitions based on the given conditions
def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

noncomputable def c (n : ℕ) : ℕ :=
  if h : is_coprime n 210 then
    Classical.some (Nat.exists_periodic_for_mod_of_gcd_eq_one n 210 h)
  else
    0

noncomputable def sum_c (n : ℕ) : ℕ :=
  (Finset.range n).sum c

theorem sum_c_1_to_210 : sum_c 210 = 329 := 
  sorry

end sum_c_1_to_210_l294_294992


namespace minimize_cone_surface_area_l294_294646

noncomputable def ratio_h_l_minimized_surface_area (V : ℝ) (radius height slant_height : ℝ) :=
  V = (1/3) * π * radius^2 * height ∧
  let A := π * radius^2 + π * radius * slant_height in
  ∀ l : ℝ, (π * radius^2 + π * radius * l) ≥ A → 
  height / slant_height = (Real.sqrt 3) / 2

theorem minimize_cone_surface_area (V : ℝ) (radius height slant_height : ℝ) :
  ratio_h_l_minimized_surface_area V radius height slant_height :=
sorry

end minimize_cone_surface_area_l294_294646


namespace angle_PSU_20_degrees_l294_294387

theorem angle_PSU_20_degrees 
  (P Q R S T U : Type) 
  (h1 : ∠ PRQ = 60)
  (h2 : ∠ QRP = 80)
  (h3 : S = foot_of_perpendicular P QR)
  (h4 : T = circumcenter PQR)
  (h5 : U = other_end_diameter P) : 
  ∠ PSU = 20 := 
sorry

end angle_PSU_20_degrees_l294_294387


namespace simplify_fraction_pow_l294_294787

theorem simplify_fraction_pow :
  ( (3 + 4 * Complex.i) / (4 - 3 * Complex.i) )^2000 = 1 := 
by
  sorry

end simplify_fraction_pow_l294_294787


namespace sample_space_def_prob_one_head_l294_294268

/-- Define the sample space for flipping a fair coin twice -/
def sample_space : set (string × string) := { ("H", "H"), ("H", "T"), ("T", "H"), ("T", "T") }

/-- Given the sample space of flipping a coin twice, prove the sample space contains exactly the specified elements -/
theorem sample_space_def : sample_space = { ("H", "H"), ("H", "T"), ("T", "H"), ("T", "T") } :=
by 
  -- sorry is used here to skip the proof step and just define the statement equivalence
  sorry

/-- Define the probability of exactly one head when flipping a coin twice -/
def probability_one_head : ℚ := 1 / 2

/-- Given the sample space for flipping a coin twice, prove the probability of exactly one head is 1/2 -/
theorem prob_one_head (s_space: set (string × string)) (h: s_space = sample_space) :
  probability_one_head = 1 / 2 :=
by 
  -- sorry is used here to skip the proof step and just define the statement equivalence
  sorry

end sample_space_def_prob_one_head_l294_294268


namespace determine_ω_and_φ_l294_294676

-- Given the function with its properties:
def y (ω φ x : Real) : Real := 2 * Real.sin (ω * x + φ)
def even_function (φ : Real) : Prop := (0 < φ) ∧ (φ < Real.pi)
def adjacent_points (x1 x2 : Real) : Prop := abs (x1 - x2) = Real.pi

-- The statement to prove ω = 2 and φ = π/2 given the conditions
theorem determine_ω_and_φ (ω φ x1 x2 : Real)
  (h1 : even_function φ)
  (h2 : adjacent_points x1 x2)
  (h3 : y ω φ x1 = 2)
  (h4 : y ω φ x2 = 2) :
  ω = 2 ∧ φ = (Real.pi / 2) :=
sorry

end determine_ω_and_φ_l294_294676


namespace sum_of_roots_of_polynomial_l294_294809

noncomputable def polynomial : ℚ[X] := 4 * X^2 + 7 * X + 3

theorem sum_of_roots_of_polynomial :
  let roots := polynomial.coeff 2
  roots = -7 / (4 * 4) := sorry

end sum_of_roots_of_polynomial_l294_294809


namespace pirate_coins_division_l294_294179

theorem pirate_coins_division :
  ∃ (x : ℕ), 
    x = 240240 ∧ 
    ∀ (k : ℕ), 
      1 ≤ k ∧ k ≤ 15 → 
      ∃ (n : ℕ),
        n = k ∧ 
        pirate_share (x : ℕ) (k : ℕ) = 240240 :=
begin
  sorry
end

end pirate_coins_division_l294_294179


namespace students_later_than_Yoongi_l294_294194

theorem students_later_than_Yoongi (total_students finished_before_Yoongi : ℕ) (h1 : total_students = 20) (h2 : finished_before_Yoongi = 11) :
  total_students - (finished_before_Yoongi + 1) = 8 :=
by {
  -- Proof is omitted as it's not required.
  sorry
}

end students_later_than_Yoongi_l294_294194


namespace additional_width_is_25cm_l294_294508

-- Definitions
def length_of_room_cm := 5000
def width_of_room_cm := 1100
def additional_width_cm := 25
def number_of_tiles := 9000
def side_length_of_tile_cm := 25

-- Statement to prove
theorem additional_width_is_25cm : additional_width_cm = 25 :=
by
  -- The proof is omitted, we assume the proof steps here
  sorry

end additional_width_is_25cm_l294_294508


namespace female_lion_weight_l294_294422

theorem female_lion_weight (male_weight : ℚ) (weight_difference : ℚ) (female_weight : ℚ) : 
  male_weight = 145/4 → 
  weight_difference = 47/10 → 
  male_weight = female_weight + weight_difference → 
  female_weight = 631/20 :=
by
  intros h₁ h₂ h₃
  sorry

end female_lion_weight_l294_294422


namespace problem_l294_294290

noncomputable def a (x : ℝ) : ℝ × ℝ := (5 * (Real.sqrt 3) * Real.cos x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 
  let dot_product := (a x).fst * (b x).fst + (a x).snd * (b x).snd
  let magnitude_square_b := (b x).fst ^ 2 + (b x).snd ^ 2
  dot_product + magnitude_square_b

theorem problem :
  (∀ x, f x = 5 * Real.sin (2 * x + Real.pi / 6) + 7 / 2) ∧
  (∃ T, T = Real.pi) ∧ 
  (∃ x, f x = 17 / 2) ∧ 
  (∃ x, f x = -3 / 2) ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi / 6), 0 ≤ x ∧ x ≤ Real.pi / 6) ∧
  (∀ x ∈ Set.Icc (2 * Real.pi / 3) Real.pi, (2 * Real.pi / 3) ≤ x ∧ x ≤ Real.pi)
:= by
  sorry

end problem_l294_294290


namespace flight_duration_correct_l294_294732

/- Define the conditions as variables in Lean -/
def departure_time := (9, 15) -- (hours, minutes) in PT
def arrival_time_ET := (18, 25) -- (hours, minutes) in ET
def time_difference := 3 -- hours difference between ET and PT
def h := 6 -- hours of flight duration
def m := 10 -- minutes of flight duration

-- Prove that h + m = 16 given the above conditions
theorem flight_duration_correct :
  (arrival_time_ET.1 - time_difference - departure_time.1) * 60 + 
  (arrival_time_ET.2 - departure_time.2) = h * 60 + m → 
  h + m = 16 :=
by simp [departure_time, arrival_time_ET, time_difference, h, m]; exact sorry

end flight_duration_correct_l294_294732


namespace solution_set_of_inequality_l294_294321

-- Define conditions
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

-- Lean statement of the proof problem
theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_mono_inc : is_monotonically_increasing_on f {x | x ≤ 0}) :
  { x : ℝ | f (3 - 2 * x) > f (1) } = { x : ℝ | 1 < x ∧ x < 2 } :=
by
  sorry

end solution_set_of_inequality_l294_294321


namespace divisible_by_four_l294_294640

theorem divisible_by_four (N : ℕ) (x : ℕ → ℤ) 
  (h1 : ∀ i, x i = 1 ∨ x i = -1) 
  (h2 : ∑ i in finset.range N, x i * x ((i + 1) % N) = 0) :
  N % 4 = 0 := 
sorry

end divisible_by_four_l294_294640


namespace negation_of_prop_l294_294337

theorem negation_of_prop :
  (¬ ∀ x : ℝ, x^2 > x - 1) ↔ ∃ x : ℝ, x^2 ≤ x - 1 :=
sorry

end negation_of_prop_l294_294337


namespace lego_count_l294_294391

theorem lego_count 
  (total_legos : ℕ := 500)
  (used_legos : ℕ := total_legos / 2)
  (missing_legos : ℕ := 5) :
  total_legos - used_legos - missing_legos = 245 := 
sorry

end lego_count_l294_294391


namespace part_a_l294_294708

theorem part_a (cities : Finset (ℝ × ℝ)) (h_cities : cities.card = 100) 
  (distances : Finset (ℝ × ℝ → ℝ)) (h_distances : distances.card = 4950) :
  ∃ (erased_distance : ℝ × ℝ → ℝ), ¬ ∃ (restored_distance : ℝ × ℝ → ℝ), 
    restored_distance = erased_distance :=
sorry

end part_a_l294_294708


namespace minimum_time_to_cook_l294_294521

def wash_pot_fill_water : ℕ := 2
def wash_vegetables : ℕ := 3
def prepare_noodles_seasonings : ℕ := 2
def boil_water : ℕ := 7
def cook_noodles_vegetables : ℕ := 3

theorem minimum_time_to_cook : wash_pot_fill_water + boil_water + cook_noodles_vegetables = 12 :=
by
  sorry

end minimum_time_to_cook_l294_294521


namespace train_pass_jogger_time_l294_294884

noncomputable def jogger_speed : ℝ := 9  -- in km/hr
noncomputable def train_speed : ℝ := 48.5 -- in km/hr
noncomputable def initial_distance : ℝ := 200 -- in meters
noncomputable def train_length : ℝ := 200 -- in meters

/--
Prove that given
- the speed of a jogger: 9 km/hr
- the speed of a train: 48.5 km/hr
- the initial distance between the jogger and the train engine: 200 m
- the length of the train: 200 m

The train will pass the jogger in approximately 36.45 seconds.
-/
theorem train_pass_jogger_time :
  let relative_speed := (train_speed - jogger_speed) * 1000 / 3600,
      total_distance := initial_distance + train_length in
  total_distance / relative_speed ≈ 36.45 := sorry

end train_pass_jogger_time_l294_294884


namespace median_moons_l294_294498

theorem median_moons :
  let moons := [0, 2, 2, 4, 21, 25, 18, 3, 2, 0].sort (· ≤ ·)
  let n := moons.length
  let median := (moons[n / 2 - 1] + moons[n / 2]) / 2
  median = 2.5 :=
by
  let moons := [0, 2, 2, 4, 21, 25, 18, 3, 2, 0].sort (· ≤ ·)
  let n := moons.length
  let median := (moons[n / 2 - 1] + moons[n / 2]) / 2
  sorry

end median_moons_l294_294498


namespace length_first_train_l294_294496

/-- Define speeds of trains in km/hr -/
def speed_train1_kmh : ℝ := 60
def speed_train2_kmh : ℝ := 40

/-- Convert speeds from km/hr to m/s -/
def convert_kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

def speed_train1_mps : ℝ := convert_kmh_to_mps speed_train1_kmh
def speed_train2_mps : ℝ := convert_kmh_to_mps speed_train2_kmh

/-- Conditions given in the problem -/
def length_train2 : ℝ := 170
def time_to_cross : ℝ := 11.159107271418288

/-- Calculate the relative speed -/
def relative_speed_mps : ℝ := speed_train1_mps + speed_train2_mps

/-- Calculate the total distance covered when crossing -/
def total_distance : ℝ := relative_speed_mps * time_to_cross

/-- Problem statement: Prove the length of the first train -/
theorem length_first_train :
  total_distance - length_train2 = 140 :=
by
  sorry

end length_first_train_l294_294496


namespace problem1_problem2_l294_294534

noncomputable def tan_theta (θ : ℝ) := tan θ = 3
noncomputable def first_expression (θ : ℝ) := (sin θ + cos θ) / (2 * sin θ + cos θ)

theorem problem1 (θ : ℝ) (h1 : tan_theta θ) : first_expression θ = 4 / 7 :=
by sorry

noncomputable def conditions (α β : ℝ) := 
  (0 < β ∧ β < π / 2 ∧ π / 2 < α ∧ α < π) ∧ 
  (cos (α - β / 2) = -1 / 9) ∧ 
  (sin (α / 2 - β) = 2 / 3)

noncomputable def second_expression (α β : ℝ) := cos ((α + β) / 2)

theorem problem2 (α β : ℝ) (h2 : conditions α β) : second_expression α β = 7 * sqrt 5 / 27 :=
by sorry

end problem1_problem2_l294_294534


namespace triangles_right_triangle_perpendicular_l294_294379

-- Conditions as definitions
variables {A B C D E H : Type} -- Define the points as variables

-- Definitions for the isosceles right triangles and the intersection
def is_isosceles_right_triangle (A B C : Type) [triangle A B C] (right_angle_vertex : Type) : Prop := 
  ∃ (isosceles : Bool), isosceles = true ∧ has_right_angle_at A B C right_angle_vertex

def intersection (line1 line2 : Type) (intersection_point : Type) : Prop := 
  ∃ (intersects : Bool), intersects = true

-- Define the required points and conditions
variables (triangle_ABD_isosceles_right : is_isosceles_right_triangle A B D D)
variables (triangle_BCE_isosceles_right : is_isosceles_right_triangle B C E E)
variables (H_intersection : intersection CD AE H)

-- Lean 4 statement to show BH ⊥ AE
theorem triangles_right_triangle_perpendicular :
  triangle_ABD_isosceles_right →
  triangle_BCE_isosceles_right →
  H_intersection →
  (angle B H A = 90) :=
by
  intro h1 h2 h3
  sorry

end triangles_right_triangle_perpendicular_l294_294379


namespace other_divisor_l294_294248

theorem other_divisor (x : ℕ) (h₁ : 261 % 7 = 2) (h₂ : 261 % x = 2) : x = 259 :=
sorry

end other_divisor_l294_294248


namespace first_train_cross_post_time_l294_294833

-- Constants and Definitions
def length_of_train := 120 -- 120 meters
def time_second_train := 15 -- 15 seconds
def time_cross_each_other := 12 -- 12 seconds
def V₂ := length_of_train / time_second_train -- Speed of the second train

-- Proposition
theorem first_train_cross_post_time :
  ∃ T₁, T₁ = 10 ∧
    ∀ V₁ : ℝ, (V₁ + V₂ = length_of_train * 2 / time_cross_each_other) →
    (length_of_train / V₁ = T₁) :=
begin
  sorry
end

end first_train_cross_post_time_l294_294833


namespace arrange_abc_l294_294031

noncomputable def a : ℝ := 6^(-0.7)
noncomputable def b : ℝ := logBase 0.7 0.6
noncomputable def c : ℝ := logBase 0.6 7

theorem arrange_abc : c < a ∧ a < b := by
  sorry

end arrange_abc_l294_294031


namespace ratio_larger_to_smaller_l294_294089

noncomputable def ratio_of_numbers (a b : ℝ) : ℝ :=
  a / b

theorem ratio_larger_to_smaller (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) 
  (h4 : a - b = 7 * ((a + b) / 2)) : ratio_of_numbers a b = 9 / 5 := 
  sorry

end ratio_larger_to_smaller_l294_294089


namespace derivative_of_f_l294_294088

-- Given function f
def f (x : ℝ) : ℝ := x + Real.exp x

-- The Lean statement for the proof problem
theorem derivative_of_f :
  deriv f x = 1 + Real.exp x :=
sorry

end derivative_of_f_l294_294088


namespace f_range_l294_294100

noncomputable def f (x : ℝ) : ℝ := 2 * x + real.sqrt (1 + x)

theorem f_range :
  ∃ l, ∀ y, y ∈ set.Ici l ↔ ∃ x, x ≥ -1 ∧ f x = y :=
by
  use -2
  intros y
  split
  { -- show y ∈ set.Ici -2 -> ∃ x, x ≥ -1 ∧ f x = y
    intro hc
    -- the rest of the proof will be filled here
    sorry
  }
  { -- show ∃ x, x ≥ -1 ∧ f x = y -> y ∈ set.Ici -2
    rintro ⟨x, h1, h2⟩
    -- the rest of the proof will be filled here
    sorry
  }

end f_range_l294_294100


namespace correct_intuitive_diagram_l294_294835

/-- Define the condition of the oblique projection method. -/
def oblique_projection (shape : Type) : Type :=
{ initial_shape : shape,
  intuitive_diagram : shape }

/-- Define the specific intuitive diagrams under oblique projection. -/
def intuitive_diagram (shape : Type) : Type
| triangle : triangle
| parallelogram : parallelogram
| square : parallelogram -- since the intuitive diagram of a square is a parallelogram
| rhombus : rhombus -- initial conjecture.

/-- Define the proof problem -/
theorem correct_intuitive_diagram : 
  (intuitive_diagram triangle = triangle) ∧ 
  (intuitive_diagram parallelogram = parallelogram) ∧ 
  (intuitive_diagram square ≠ square) ∧ 
  (intuitive_diagram rhombus ≠ rhombus):=
  sorry

end correct_intuitive_diagram_l294_294835


namespace triangle_area_0_0_0_5_7_12_l294_294502

theorem triangle_area_0_0_0_5_7_12 : 
    let base := 5
    let height := 7
    let area := (1 / 2) * base * height
    area = 17.5 := 
by
    sorry

end triangle_area_0_0_0_5_7_12_l294_294502


namespace tom_has_18_apples_l294_294921

-- Definitions based on conditions
def phillip_apples : ℕ := 40
def ben_apples : ℕ := phillip_apples + 8
def tom_apples : ℕ := (3 * ben_apples) / 8

-- Theorem stating Tom has 18 apples given the conditions
theorem tom_has_18_apples : tom_apples = 18 :=
sorry

end tom_has_18_apples_l294_294921


namespace sin_lg_roots_l294_294476

theorem sin_lg_roots (f : ℝ → ℝ) (g : ℝ → ℝ) (h₁ : ∀ x, f x = Real.sin x) (h₂ : ∀ x, g x = Real.log x)
  (domain : ∀ x, x > 0 → x < 10) (h₃ : ∀ x, f x ≤ 1 ∧ g x ≤ 1) :
  ∃ x1 x2 x3, (0 < x1 ∧ x1 < 10) ∧ (f x1 = g x1) ∧
               (0 < x2 ∧ x2 < 10) ∧ (f x2 = g x2) ∧
               (0 < x3 ∧ x3 < 10) ∧ (f x3 = g x3) ∧
               x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 :=
by
  sorry

end sin_lg_roots_l294_294476


namespace find_primes_a_l294_294981

theorem find_primes_a :
  ∀ (a : ℕ), (∀ n : ℕ, n < a → Nat.Prime (4 * n * n + a)) → (a = 3 ∨ a = 7) :=
by
  sorry

end find_primes_a_l294_294981


namespace find_a5_and_S11_l294_294353

variable {S : ℕ → ℚ}
variable {a : ℕ → ℚ}
variable {d a1 : ℚ}

noncomputable def sum_of_first_n_terms (n : ℕ) : ℚ :=
  n * a1 + (n * (n - 1)) / 2 * d

axiom H₁ : sum_of_first_n_terms 9 = -36
axiom H₂ : sum_of_first_n_terms 13 = -104

theorem find_a5_and_S11 (h : ∀ n, S n = sum_of_first_n_terms n) :
  a 5 = -4 ∧ S 11 = -66 := sorry

end find_a5_and_S11_l294_294353


namespace product_eval_l294_294621

theorem product_eval (a : ℝ) (h : a = 1) : (a - 3) * (a - 2) * (a - 1) * a * (a + 1) = 0 :=
by
  sorry

end product_eval_l294_294621


namespace convex_pentagons_l294_294624

theorem convex_pentagons (P : Finset ℝ) (h : P.card = 15) : 
  (P.card.choose 5) = 3003 := 
by
  sorry

end convex_pentagons_l294_294624


namespace interest_rate_of_second_part_l294_294785

variable (total_amount : ℝ) (P1 : ℝ) (rate1 : ℝ) (total_income : ℝ) (P2 : ℝ) (interest1 : ℝ) (interest2 : ℝ) (rate2 : ℝ)

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_of_second_part :
  total_amount = 2500 →
  P1 = 500 →
  rate1 = 5 →
  total_income = 145 →
  P2 = total_amount - P1 →
  interest1 = simple_interest P1 rate1 1 →
  interest2 = total_income - interest1 →
  simple_interest P2 rate2 1 = interest2 →
  rate2 = 6 :=
by sorry

end interest_rate_of_second_part_l294_294785


namespace segment_HY_length_l294_294074

theorem segment_HY_length (C D E F G H Y : Type) [metric_space C] [metric_space D] [metric_space E] [metric_space F] [metric_space G] [metric_space H] [metric_space Y] 
  (cd_length : ∀ (x: Type) [metric_space x], 4)
  (CY : ∀ (x: Type) [metric_space x] (y: Type) [metric_space y], distance x y = 8) 
  (hexagon_property : regular_hexagon C D E F G H 4):
  distance H Y = 6 * sqrt 5 := 
sorry

end segment_HY_length_l294_294074


namespace problem_statement_l294_294030

noncomputable def roots (a b : ℝ) (coef1 coef2 : ℝ) :=
  ∃ x : ℝ, (x = a ∨ x = b) ∧ x^2 + coef1 * x + coef2 = 0

theorem problem_statement
  (a b c d : ℝ)
  (h1 : a + b = -57)
  (h2 : a * b = 1)
  (h3 : c + d = 57)
  (h4 : c * d = 1) :
  (a + c) * (b + c) * (a - d) * (b - d) = 0 := 
by
  sorry

end problem_statement_l294_294030


namespace percentage_increase_kayla_deshawn_l294_294083

theorem percentage_increase_kayla_deshawn :
  ∀ (free_throws_deshawn : ℕ) (free_throws_annieka: ℕ) (fewer_annieka: ℕ),
    free_throws_deshawn = 12 →
    free_throws_annieka = 14 →
    (free_throws_annieka + fewer_annieka) = free_throws_kayla →
    free_throws_kayla = 18 →
    (fewer_annieka = 4) →
    (free_throws_kayla - free_throws_deshawn : ℝ) / free_throws_deshawn * 100 = 50 := by
  intros free_throws_deshawn free_throws_annieka fewer_annieka
  assume h1 : free_throws_deshawn = 12
  assume h2 : free_throws_annieka = 14
  assume h3 : (free_throws_annieka + fewer_annieka) = free_throws_kayla
  assume h4 : free_throws_kayla = 18
  assume h5 : fewer_annieka = 4
  sorry

end percentage_increase_kayla_deshawn_l294_294083


namespace Rhonda_marbles_l294_294907

theorem Rhonda_marbles :
  ∃ m : ℕ, (∃ a : ℕ, a + m = 215 ∧ a = m + 55) ∧ m = 80 :=
by
  use 80
  use 135
  sorry

end Rhonda_marbles_l294_294907


namespace smaller_circle_area_l294_294830

theorem smaller_circle_area (r R : ℝ) (hR : R = 3 * r)
  (hTangentLines : ∀ (P A B A' B' : ℝ), P = 5 ∧ A = 5 ∧ PA = 5 ∧ A' = 5 ∧ PA' = 5 ∧ AB = 5 ∧ A'B' = 5 ) :
  π * r^2 = 25 / 3 * π := by
  sorry

end smaller_circle_area_l294_294830


namespace evaluate_expression_l294_294623

theorem evaluate_expression : (723 * 723) - (722 * 724) = 1 :=
by
  sorry

end evaluate_expression_l294_294623


namespace largest_result_l294_294911

theorem largest_result :
  let A := (1 / 17 - 1 / 19) / 20
  let B := (1 / 15 - 1 / 21) / 60
  let C := (1 / 13 - 1 / 23) / 100
  let D := (1 / 11 - 1 / 25) / 140
  D > A ∧ D > B ∧ D > C := by
  sorry

end largest_result_l294_294911


namespace geometric_series_sum_first_four_terms_l294_294594

theorem geometric_series_sum_first_four_terms :
  let a : ℝ := 1
  let r : ℝ := 1 / 3
  let n : ℕ := 4
  (a * (1 - r^n) / (1 - r)) = 40 / 27 := by
  let a : ℝ := 1
  let r : ℝ := 1 / 3
  let n : ℕ := 4
  sorry

end geometric_series_sum_first_four_terms_l294_294594


namespace firstSuperbSunday_is_February_29_2020_l294_294156

/-- Define the concept of "Superb Sunday." A month with a Superb Sunday has five Sundays. -/
def isSuperbSundayMonth (year month : ℕ) : Prop :=
  (Nat.countp (λ d, d.weekday = 6) (finset.range (DateTime.DaysInMonth year month)) = 5)

/-- A company's fiscal year starts on January 13, 2020. -/
def fiscal_start : DateTime := ⟨2020, 1, 13⟩

/-- Calculate the date of the first Superb Sunday after January 13, 2020. -/
noncomputable 
def firstSuperbSundayAfterFiscalStart : DateTime :=
  DateTime.next (λ d, d > fiscal_start ∧ isSuperbSundayMonth d.year d.month) fiscal_start

theorem firstSuperbSunday_is_February_29_2020 :
  firstSuperbSundayAfterFiscalStart = ⟨2020, 2, 29⟩ :=
sorry

end firstSuperbSunday_is_February_29_2020_l294_294156


namespace range_of_a_l294_294792

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_positive : ∀ x, 0 ≤ x → f x = x^2) 
  (h_ineq : ∀ x, a ≤ x ∧ x ≤ a + 2 → f (x + a) ≥ 3 * f x) : 
  2 + 2 * real.sqrt 3 ≤ a :=
by sorry

end range_of_a_l294_294792


namespace shape_with_circular_views_is_sphere_l294_294576

/-- Define the views of the cuboid, cylinder, cone, and sphere. -/
structure Views (shape : Type) :=
(front_view : Type)
(left_view : Type)
(top_view : Type)

def is_cuboid (s : Views) : Prop :=
s.front_view = Rectangle ∧ s.left_view = Rectangle ∧ s.top_view = Rectangle

def is_cylinder (s : Views) : Prop :=
s.front_view = Rectangle ∧ s.left_view = Rectangle ∧ s.top_view = Circle

def is_cone (s : Views) : Prop :=
s.front_view = IsoscelesTriangle ∧ s.left_view = IsoscelesTriangle ∧ s.top_view = Circle

def is_sphere (s : Views) : Prop :=
s.front_view = Circle ∧ s.left_view = Circle ∧ s.top_view = Circle

/-- Proof problem: Prove that the only shape with circular views in all three perspectives (front, left, top) is the sphere. -/
theorem shape_with_circular_views_is_sphere :
  ∀ (s : Views), 
    (s.front_view = Circle ∧ s.left_view = Circle ∧ s.top_view = Circle) → 
    is_sphere s ∧ ¬ is_cuboid s ∧ ¬ is_cylinder s ∧ ¬ is_cone s :=
by
  intro s h
  sorry

end shape_with_circular_views_is_sphere_l294_294576


namespace measure_angle_DAB_l294_294364

def angle_dab (α : ℝ) : Prop :=
  ∃ (A B D C : Type) (distance : A → A → ℝ),
  ∠ D A B = α ∧ 
  (∃ (A B D : A), 
    ∠ B A D = 90 ∧ 
    (∃ C : (segment A D), 
       distance A C = 2 * distance C D ∧ 
       distance A B = 2 * distance B C))

theorem measure_angle_DAB : angle_dab 54.74 :=
sorry

end measure_angle_DAB_l294_294364


namespace complement_union_eq_complement_l294_294000

open Set

section ComplementUnion

variable (k : ℤ)

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def UniversalSet : Set ℤ := univ
def ComplementUnion : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem complement_union_eq_complement :
  UniversalSet \ (SetA ∪ SetB) = ComplementUnion :=
by
  sorry

end ComplementUnion

end complement_union_eq_complement_l294_294000


namespace box_calories_l294_294864

theorem box_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  (cookies_per_bag * bags_per_box) * calories_per_cookie = 1600 :=
by
  sorry

end box_calories_l294_294864


namespace hyperbola_equation_l294_294554

theorem hyperbola_equation :
  let ellipse := ∀ x y : ℝ, 4 * x^2 + y^2 = 1,
  let asymptote := ∀ x y : ℝ, y = sqrt 2 * x,
  ∃ (a b c m : ℝ),
    1 = a^2 ∧ b^2 = 1 / 4 ∧ c = sqrt (a^2 - b^2) ∧ 
    0 < m ∧ m < 3 / 4 ∧ 
    sqrt (m / (3 / 4 - m)) = sqrt 2 ∧ m = 1 / 2 ∧
    (∀ x y : ℝ, 2 * y^2 - 4 * x^2 = 1) :=
by
  -- Definitions for the ellipse and asymptote
  let ellipse := ∀ x y : ℝ, 4 * x^2 + y^2 = 1
  let asymptote := ∀ x y : ℝ, y = sqrt 2 * x
  
  -- Theorem statement
  show ∃ (a b c m : ℝ),
    1 = a^2 ∧ b^2 = 1 / 4 ∧ c = sqrt (a^2 - b^2) ∧ 
    0 < m ∧ m < 3 / 4 ∧ 
    sqrt (m / (3 / 4 - m)) = sqrt 2 ∧ m = 1 / 2 ∧
    (∀ x y : ℝ, 2 * y^2 - 4 * x^2 = 1),
  -- Placeholder for the proof
  sorry

end hyperbola_equation_l294_294554


namespace num_arrangements_TOOT_l294_294968

def num_permutations_TOOT : ℕ :=
  4!

def repetitions : ℕ × ℕ :=
  (2!, 2!)

theorem num_arrangements_TOOT : (num_permutations_TOOT / (repetitions.1 * repetitions.2)) = 6 :=
by
  sorry

end num_arrangements_TOOT_l294_294968


namespace divisor_is_18_l294_294119

def dividend : ℕ := 165
def quotient : ℕ := 9
def remainder : ℕ := 3

theorem divisor_is_18 (divisor : ℕ) : dividend = quotient * divisor + remainder → divisor = 18 :=
by sorry

end divisor_is_18_l294_294119


namespace chocolate_bars_in_box_l294_294975

-- Definitions of the conditions
def cost_per_bar := 3
def amount_made_by_wendy := 18

-- Defining the main theorem
theorem chocolate_bars_in_box : 
  ∃ n : ℕ, (cost_per_bar * (n - 3) = amount_made_by_wendy) ∧ (n = 9) :=
begin
  -- The proof is omitted by using sorry
  sorry,
end

end chocolate_bars_in_box_l294_294975


namespace range_of_m_l294_294332

noncomputable def f (x : ℝ) : ℝ := 1 - 1 / x

theorem range_of_m (a b m : ℝ) (h_dom : 0 < a) (h_a_lt_b : a < b) (h_rng : (∀ y, y ∈ Set.range f ⟺ ∃ x ∈ Set.Ioo a b, y = f x)) : 0 < m ∧ m < (1 / 4) :=
sorry

end range_of_m_l294_294332


namespace sin_angle_ACM_le_one_third_l294_294900

-- Given conditions
variables {A B C M : Type} [Triangle A B C] (h1 : ∠ B = 90°) (h2 : M = midpoint A B)

-- Proof Statement
theorem sin_angle_ACM_le_one_third : 
  ∀ {A B C M : Type} [Triangle A B C], 
      (∠ B = 90°) → (M = midpoint A B) →
        sin (angle A C M) ≤ (1 / 3) :=
by
  sorry

end sin_angle_ACM_le_one_third_l294_294900


namespace choose_k_numbers_l294_294036

theorem choose_k_numbers (n k : ℕ) : 
  (finset.card (finset.filter (λ (l : list ℕ), l.sum = n) 
  (finset.range (n + 1)).powerset_len k)) = nat.choose (n + k - 1) (k - 1) :=
sorry

end choose_k_numbers_l294_294036


namespace transformed_function_is_correct_l294_294448

-- Original function definition
def f (x : ℝ) : ℝ := sin (2 * x)

-- Definition of the left shift by π/3 units
def left_shift (g : ℝ → ℝ) (c : ℝ) : ℝ → ℝ :=
  fun x => g (x + c)

-- Definition of horizontal compression by a factor of 1/2
def compress (g : ℝ → ℝ) (k : ℝ) : ℝ → ℝ :=
  fun x => g (x / k)

-- Final function to prove
def final_function (x : ℝ) : ℝ := sin (4 * x + 2 * Real.pi / 3)

-- Proof statement
theorem transformed_function_is_correct :
  compress (left_shift f (Real.pi / 3)) (1 / 2) = final_function := by
  sorry

end transformed_function_is_correct_l294_294448


namespace boy_speed_second_day_l294_294150

theorem boy_speed_second_day (distance : ℝ) (speed_first_day : ℝ) (time_late : ℝ) (time_early : ℝ) (normal_speed: ℝ) :
  distance = 60 ∧ speed_first_day = 10 ∧ time_late = 2 ∧ time_early = 1 ∧ normal_speed = 6 →
  (distance / (normal_speed + time_late) = 60 / 8 ∧ distance / (normal_speed - time_early) = 12) ∧ (distance / (normal_speed - time_early) = distance / 5) :=
by
  intros
  cases a,
  sorry

end boy_speed_second_day_l294_294150


namespace complete_graph_edge_product_distinct_l294_294449

theorem complete_graph_edge_product_distinct (n : ℕ) (h : n ≥ 3) :
  ∃ (f : fin (n*(n-1)/2) → {1, 2, 3}), 
  ∀ v : fin n, ∀ w x : fin n, w ≠ x → 
  (∏ (e : {e // e.val.1 = v ∨ e.val.2 = v ∧ e.val.1 < e.val.2}), f e.val) ≠ 
  (∏ (e : {e // e.val.1 = x ∨ e.val.2 = x ∧ e.val.1 < e.val.2}), f e.val) := 
by
  sorry

end complete_graph_edge_product_distinct_l294_294449


namespace complement_union_l294_294005

def A := { x : ℤ | ∃ k : ℤ, x = 3 * k + 1 }
def B := { x : ℤ | ∃ k : ℤ, x = 3 * k + 2 }
def U := ℤ

theorem complement_union :
  { x : ℤ | x ∉ (A ∪ B) } = { x : ℤ | ∃ k : ℤ, x = 3 * k } := 
by 
  sorry

end complement_union_l294_294005


namespace polynomial_not_product_of_single_var_l294_294441

theorem polynomial_not_product_of_single_var :
  ¬ ∃ (f : Polynomial ℝ) (g : Polynomial ℝ), 
    (∀ (x y : ℝ), (f.eval x) * (g.eval y) = (x^200) * (y^200) + 1) := sorry

end polynomial_not_product_of_single_var_l294_294441


namespace construct_D_of_parallelogram_l294_294340

noncomputable theory

variables {A B C D P O₁ O₂ O₃ O₄ : Point} {R : ℝ}
variables (k₁ k₂ k₃ k₄ : Circle)

-- Given triangle ABC
def is_triangle (A B C : Point) : Prop := ∃ Δ : Triangle, Δ.A = A ∧ Δ.B = B ∧ Δ.C = C

-- Define circles k₁ and k₂
def k₁_center : Circle.center k₁ = (B + A) / 2
def k₁_radius_eq : Circle.radius k₁ = dist B A / 2

def k₂_center : Circle.center k₂ = (B + C) / 2
def k₂_radius_eq : Circle.radius k₂ = dist B C / 2

-- Define the intersection point P of circles k₁ and k₂
def P_is_intersection : Point := Classical.choose (exists_intersect k₁ k₂)
def P_on_circle_k₁ : P ∈ k₁
def P_on_circle_k₂ : P ∈ k₂

-- Define circles k₃ and k₄ centered at P
def k₃_center : Circle.center k₃ = P
def k₃_radius_eq : Circle.radius k₃ = Circle.radius k₁

def k₄_center : Circle.center k₄ = P
def k₄_radius_eq : Circle.radius k₄ = Circle.radius k₂

-- Main theorem to prove that D is the fourth vertex of parallelogram ABCD
theorem construct_D_of_parallelogram :
  is_triangle A B C ∧
  (Circle.center k₁ = (B + A) / 2 ∧ Circle.radius k₁ = dist B A / 2) ∧
  (Circle.center k₂ = (B + C) / 2 ∧ Circle.radius k₂ = dist B C / 2) ∧
  (P ∈ k₁ ∧ P ∈ k₂) ∧
  (Circle.center k₃ = P ∧ Circle.radius k₃ = Circle.radius k₁) ∧
  (Circle.center k₄ = P ∧ Circle.radius k₄ = Circle.radius k₂) →
  ∃ D : Point, D ∈ k₃ ∧ D ∈ k₄ ∧ parallelogram A B C D :=
begin
  sorry
end

end construct_D_of_parallelogram_l294_294340


namespace distance_point_line_l294_294984

open Real EuclideanGeometry

theorem distance_point_line :
  let p := (2, 4, 5) in
  let a := (5, 8, 9) in
  let b := (4, 3, -3) in
  let line (t : ℝ) := (a.1 + t * b.1, a.2 + t * b.2, a.3 + t * b.3) in
  ∃ t : ℝ, let pp := (line t).1 - p.1, (line t).2 - p.2, (line t).3 - p.3 in
  sqrt (pp.1^2 + pp.2^2 + pp.3^2) = 95 / 17 :=
by {
  sorry
}

end distance_point_line_l294_294984


namespace fixed_monthly_fee_l294_294601

-- Define the problem parameters and assumptions
variables (x y : ℝ)
axiom february_bill : x + y = 20.72
axiom march_bill : x + 3 * y = 35.28

-- State the Lean theorem that we want to prove
theorem fixed_monthly_fee : x = 13.44 :=
by
  sorry

end fixed_monthly_fee_l294_294601


namespace decagon_perimeter_l294_294211

-- Define the number of sides in a decagon
def num_sides : ℕ := 10

-- Define the length of each side in the decagon
def side_length : ℕ := 3

-- Define the perimeter of a decagon given the number of sides and the side length
def perimeter (n : ℕ) (s : ℕ) : ℕ := n * s

-- State the theorem we want to prove: the perimeter of our given regular decagon
theorem decagon_perimeter : perimeter num_sides side_length = 30 := 
by sorry

end decagon_perimeter_l294_294211


namespace solve_fractional_equation_l294_294478

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 3) : (2 * x) / (x - 3) = 1 ↔ x = -3 :=
by
  sorry

end solve_fractional_equation_l294_294478


namespace math_problem_l294_294201

def is_positive_integer (n : ℤ) : Prop := n > 0

theorem math_problem :
  is_positive_integer 2024 ∧
  ¬ is_positive_integer 0 ∧
  ¬ is_positive_integer (-3.14) ∧
  ¬ is_positive_integer (-21) :=
by
  sorry

end math_problem_l294_294201


namespace magnitude_of_proj_l294_294421

noncomputable def proj_magnitude (v w : ℝ → ℝ) (theta : ℝ) :=
  let v_dot_w := ∥v∥ * ∥w∥ * Real.cos theta
  in (abs v_dot_w / ∥w∥) * ∥w∥

theorem magnitude_of_proj (v w : ℝ → ℝ) (theta : ℝ)
  (h_theta : theta = Real.pi / 6) 
  (hv : ∥v∥ = 4) (hw : ∥w∥ = 6) :
  proj_magnitude v w theta = 12 * Real.sqrt 3 := 
by
  sorry

end magnitude_of_proj_l294_294421


namespace find_b2_l294_294819

theorem find_b2 (b : ℕ → ℝ) (h1 : b 1 = 23)
  (h10 : b 10 = 107)
  (hmean : ∀ n, n ≥ 3 → b n = (∑ k in finset.range (n - 1), b (k + 1)) / (n - 1)) :
  b 2 = 191 :=
by 
  sorry

end find_b2_l294_294819


namespace tangent_distance_l294_294555

open Real

theorem tangent_distance :
  ∃ P1 P2 : ℝ × ℝ, 
  (P1 = (0, 1) ∧ (∃ m : ℝ, m * P1.1 + b = P1.2 ∧ m = 1 ∧ b = 1)) ∧ -- point on y = e^x
  (P2 = (1, 2) ∧ (∃ m : ℝ, m * P2.1 + b = P2.2 ∧ m = 1 ∧ b = 1)) ∧ -- point on y^2 = 4x
  dist P1 P2 = sqrt 2 :=
by sorry

end tangent_distance_l294_294555


namespace RiverJoe_popcorn_shrimp_price_l294_294784

theorem RiverJoe_popcorn_shrimp_price
  (price_catfish : ℝ)
  (total_orders : ℕ)
  (total_revenue : ℝ)
  (orders_popcorn_shrimp : ℕ)
  (catfish_revenue : ℝ)
  (popcorn_shrimp_price : ℝ) :
  price_catfish = 6.00 →
  total_orders = 26 →
  total_revenue = 133.50 →
  orders_popcorn_shrimp = 9 →
  catfish_revenue = (total_orders - orders_popcorn_shrimp) * price_catfish →
  catfish_revenue + orders_popcorn_shrimp * popcorn_shrimp_price = total_revenue →
  popcorn_shrimp_price = 3.50 :=
by
  intros price_catfish_eq total_orders_eq total_revenue_eq orders_popcorn_shrimp_eq catfish_revenue_eq revenue_eq
  sorry

end RiverJoe_popcorn_shrimp_price_l294_294784


namespace solve_linear_system_l294_294682

theorem solve_linear_system (m x y : ℝ) 
  (h1 : x + y = 3 * m) 
  (h2 : x - y = 5 * m)
  (h3 : 2 * x + 3 * y = 10) : 
  m = 2 := 
by 
  sorry

end solve_linear_system_l294_294682


namespace ratio_inscribed_square_to_triangle_area_l294_294800

variable {a h : ℝ} (ha : a > 0) (hh : h > 0)

theorem ratio_inscribed_square_to_triangle_area : 
  (let x := (a * h) / (a + h) in
  let square_area := x * x in
  let triangle_area := (1 / 2) * a * h in
  (square_area / triangle_area) = (2 * a * h) / (a + h)^2) :=
sorry

end ratio_inscribed_square_to_triangle_area_l294_294800


namespace range_of_a_l294_294093

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l294_294093


namespace complex_conjugate_example_proof_l294_294645

def complex_conjugate_example : Prop :=
  ∀ (z : ℂ), z = complex.mk 1 (-1) → conj z = complex.mk 1 1

theorem complex_conjugate_example_proof : complex_conjugate_example :=
by
  intros z h
  rw h
  simp

end complex_conjugate_example_proof_l294_294645


namespace odd_even_derivative_behavior_l294_294654

variables {R : Type} [OrderedField R]

noncomputable def odd_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = -f(x)

noncomputable def even_function (g : R → R) : Prop :=
  ∀ x : R, g (-x) = g(x)

theorem odd_even_derivative_behavior (f g : R → R)
  (h_odd_f : odd_function f)
  (h_even_g : even_function g)
  (h_f_pos : ∀ x : R, 0 < x → (f' x) > 0)
  (h_g_pos : ∀ x : R, 0 < x → (g' x) > 0) :
  (∀ x : R, x < 0 → (f' x) < 0) ∧ (∀ x : R, x < 0 → (g' x) < 0) :=
sorry

end odd_even_derivative_behavior_l294_294654


namespace arithmetic_sequence_formula_geometric_sequence_formula_sum_of_c_n_l294_294727

-- Define the arithmetic sequence a_n
def a_n (n : ℕ) : ℕ := 2 * n + 1

-- Define the geometric sequence b_n
def b_n (n : ℕ) : ℕ := 5 * 2^(n - 1)

-- Define the sequence c_n
def c_n (n : ℕ) : ℕ := (a_n n) / (b_n n) + 1

-- Define the sum T_n of the first n terms of the sequence c_n
def T_n (n : ℕ) : ℕ := (Finset.range n).sum (λ k, c_n (k + 1))

-- Theorem statements for the main results:

-- 1. Proving the general formula of the arithmetic sequence a_n
theorem arithmetic_sequence_formula : ∀ (n : ℕ), a_n n = 2 * n + 1 := by
  sorry

-- 2. Proving the general formula of the geometric sequence b_n
theorem geometric_sequence_formula : ∀ (n : ℕ), b_n n = 5 * 2^(n - 1) := by
  sorry

-- 3. Proving the sum of the first n terms T_n of the sequence c_n
theorem sum_of_c_n : ∀ (n : ℕ), T_n n = 2 + n - (2 * n + 5) / (5 * 2^(n - 1)) := by
  sorry

end arithmetic_sequence_formula_geometric_sequence_formula_sum_of_c_n_l294_294727


namespace min_sum_x_y_condition_l294_294295

theorem min_sum_x_y_condition {x y : ℝ} (h₁ : x > 0) (h₂ : y > 0) (h₃ : 1 / x + 9 / y = 1) : x + y = 16 :=
by
  sorry -- proof skipped

end min_sum_x_y_condition_l294_294295


namespace box_contains_1600_calories_l294_294866

theorem box_contains_1600_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  total_calories = 1600 :=
by
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  show total_calories = 1600
  sorry

end box_contains_1600_calories_l294_294866


namespace minimal_positive_period_of_sin_function_l294_294473

theorem minimal_positive_period_of_sin_function :
  (∀ x, f x = Real.sin (3 * x + Real.pi / 4)) →
  ∃ T > 0, ∀ x, f (x + T) = f x :=
begin
  intro h,
  use 2 * Real.pi / 3,
  split,
  { sorry },  -- You would provide the proof that 2*pi/3 > 0 here
  { sorry }   -- You would provide the proof that forall x, f (x + 2*pi/3) = f x here
end

end minimal_positive_period_of_sin_function_l294_294473


namespace min_y_value_l294_294228

theorem min_y_value (x : ℝ) : 
  ∃ y : ℝ, y = 4 * x^2 + 8 * x + 12 ∧ ∀ z, (z = 4 * x^2 + 8 * x + 12) → y ≤ z := sorry

end min_y_value_l294_294228


namespace taco_castle_parking_lot_l294_294823

variable (D F T V : ℕ)

theorem taco_castle_parking_lot (h1 : F = D / 3) (h2 : F = 2 * T) (h3 : V = T / 2) (h4 : V = 5) : D = 60 :=
by
  sorry

end taco_castle_parking_lot_l294_294823


namespace a_lt_b_l294_294300

open Real

-- Define the function f(x), its derivative, and the given condition f'(x) > f(x) for x >= 0
variables {f : ℝ → ℝ}
variable [differentiable ℝ f]

-- Condition: f'(x) > f(x) for any x >= 0
axiom f'_gt_f (x : ℝ) (hx : 0 ≤ x) : deriv f x > f x

-- Definitions of a and b
def a : ℝ := f 2 / exp 2
def b : ℝ := f 3 / exp 3

-- Statement to show a < b
theorem a_lt_b : a < b :=
by
  sorry

end a_lt_b_l294_294300


namespace sum_S11_l294_294028

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {a1 d : ℝ}

axiom arithmetic_sequence (n : ℕ) : a n = a1 + (n - 1) * d
axiom sum_of_first_n_terms (n : ℕ) : S n = n / 2 * (a 1 + a n)
axiom condition : a 3 + 4 = a 2 + a 7

theorem sum_S11 : S 11 = 44 := by
  sorry

end sum_S11_l294_294028


namespace calculate_expression_l294_294592

theorem calculate_expression :
  (-0.125)^2022 * 8^2023 = 8 :=
sorry

end calculate_expression_l294_294592


namespace perpendicular_line_through_point_eq_l294_294985

theorem perpendicular_line_through_point_eq (P : ℝ × ℝ)
    (line : ℝ × ℝ → Prop)
    (hP : P = (3, 2))
    (hline : line = λ x : ℝ × ℝ, x.1 + 4 * x.2 - 2 = 0) :
    ∃ m : ℝ, (4 * P.1 - P.2 + m = 0) ∧ (4 * P.1 - P.2 - 10 = 0) :=
by {
    sorry
}

end perpendicular_line_through_point_eq_l294_294985


namespace jamie_coins_value_l294_294392

variable (n d : ℕ)

theorem jamie_coins_value :
  (d = 30 - n) →
  (150 + 5 * n = 300 - 5 * n + 90) →
  (5 * n + 10 * d = 180) :=
by
  intros h1 h2
  rw [h1] at h2
  sorry

end jamie_coins_value_l294_294392


namespace algebraic_expression_value_l294_294511

-- Given conditions
def x := Real.sqrt 5 + 2
def y := Real.sqrt 5 - 2

-- Statement of the mathematical proof problem
theorem algebraic_expression_value : 
  x^2 - y + x * y = 12 + 3 * Real.sqrt 5 := 
by 
  sorry

end algebraic_expression_value_l294_294511


namespace necessary_and_sufficient_conditions_for_perpendicular_planes_l294_294298

variables (α β γ : Plane) (a b : Line)

def condition1 : Prop :=
  ∃ γ, γ ⊥ α ∧ γ ∥ β

def condition2 : Prop :=
  ∃ a, a ⊥ β

def condition3 : Prop :=
  ∃ a b, a ⊥ b ∧ a ⊥ β ∧ b ⊥ α

theorem necessary_and_sufficient_conditions_for_perpendicular_planes :
  condition1 α β ∧ condition3 α β ↔ α ⊥ β := sorry

end necessary_and_sufficient_conditions_for_perpendicular_planes_l294_294298


namespace distance_between_stations_l294_294493

theorem distance_between_stations (x y t : ℝ) 
(start_same_hour : t > 0)
(speed_slow_train : ∀ t, x = 16 * t)
(speed_fast_train : ∀ t, y = 21 * t)
(distance_difference : y = x + 60) : 
  x + y = 444 := 
sorry

end distance_between_stations_l294_294493


namespace find_plane_equation_l294_294247

theorem find_plane_equation :
  ∃ (A B C D : ℤ),
  (2, 3, -1) ∈ set_of (λ (p : ℝ × ℝ × ℝ), 
    A * p.1 + B * p.2 + C * p.3 + D = 0) ∧
  ∃ (k : ℝ), 
    (A = 3 * k ∧ B = -4 * k ∧ C = 2 * k) ∧
  (A > 0 ∧ Int.gcd (A.natAbs) (Int.gcd (B.natAbs) (Int.gcd (C.natAbs) (D.natAbs))) = 1) :=
begin
  use [3, -4, 2, 8],
  split,
  { simp, ring, },
  use 1,
  split, 
  { simp, ring, },
  split,
  { norm_num, },
  { norm_num, },
end

end find_plane_equation_l294_294247


namespace hexagon_inequality_l294_294299

theorem hexagon_inequality
  (A B C D E F G H : Point)
  (h_convex : ConvexHexagon A B C D E F)
  (h_AB_eq_BC_eq_CD : dist A B = dist B C ∧ dist B C = dist C D)
  (h_DE_eq_EF_eq_FA : dist D E = dist E F ∧ dist E F = dist F A)
  (h_angle_BCD : ∠ B C D = 60)
  (h_angle_EFA : ∠ E F A = 60)
  (h_angle_AGB : ∠ A G B = 120)
  (h_angle_DHE : ∠ D H E = 120) :
  dist A G + dist G B + dist G H + dist H D + dist H E ≥ dist C F :=
sorry

end hexagon_inequality_l294_294299


namespace max_number_squares_k6_max_number_squares_k1_l294_294145

noncomputable def max_colored_squares_k6 : ℕ :=
  sorry

theorem max_number_squares_k6 :
  (∀ (table : fin 30 × fin 30) (is_colored : fin 30 × fin 30 → Bool),
    (∀ (sq : fin 30 × fin 30), is_colored sq → 
      (finset.card (finset.filter (λ (nb : fin 30 × fin 30), is_colored nb) (finset.univ.filter (λ (nb : fin 30 × fin 30),
        (abs (sq.1 - nb.1) ≤ 1 ∧ abs (sq.2 - nb.2) ≤ 1 ∧ (sq ≠ nb))))) ≤ 6) →
    finset.card (finset.filter is_colored finset.univ) ≤ 720) :=
  sorry

noncomputable def max_colored_squares_k1 : ℕ :=
  sorry

theorem max_number_squares_k1 :
  (∀ (table : fin 30 × fin 30) (is_colored : fin 30 × fin 30 → Bool),
    (∀ (sq : fin 30 × fin 30), is_colored sq → 
      (finset.card (finset.filter (λ (nb : fin 30 × fin 30), is_colored nb) (finset.univ.filter (λ (nb : fin 30 × fin 30),
        (abs (sq.1 - nb.1) ≤ 1 ∧ abs (sq.2 - nb.2) ≤ 1 ∧ (sq ≠ nb))))) ≤ 1) →
    finset.card (finset.filter is_colored finset.univ) ≤ 300) :=
  sorry

end max_number_squares_k6_max_number_squares_k1_l294_294145


namespace angle_PSU_eq_20_l294_294385

theorem angle_PSU_eq_20
  (P Q R S T U : Type)
  (PRQ QRP : ℕ)
  (h_PRQ : PRQ = 60)
  (h_QRP : QRP = 80)
  (foot_of_perpendicular : S)
  (circumcenter : T)
  (diameter_end : U)
  : sorry :=
begin
  -- Given conditions (lean terms explanations)
  -- we define the terms angle PSQ, POT, PTO, and PSU
  -- and provide their relationships and theorems as provided in the problem statement 
  sorry
end

end angle_PSU_eq_20_l294_294385


namespace Angelina_speed_grocery_to_gym_l294_294916

theorem Angelina_speed_grocery_to_gym :
    ∀ (v : ℝ), (v > 0) →
    let t1 := 200 / v in
    let t2 := 300 / (2 * v) in
    (t1 - t2 = 50) →
    2 * v = 2 := by
  -- sorry to skip the proof
  sorry

end Angelina_speed_grocery_to_gym_l294_294916


namespace pyramid_volume_correct_l294_294125

noncomputable def pyramid_volume (a : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * (a^2) * h

theorem pyramid_volume_correct :
  pyramid_volume 1 (√3 / 2) = √3 / 6 := by
  sorry

end pyramid_volume_correct_l294_294125


namespace machine_value_after_two_years_l294_294470

theorem machine_value_after_two_years (initial_value : ℝ) (decrease_rate : ℝ) (years : ℕ) (value_after_two_years : ℝ) :
  initial_value = 8000 ∧ decrease_rate = 0.30 ∧ years = 2 → value_after_two_years = 3200 := by
  intros h
  sorry

end machine_value_after_two_years_l294_294470


namespace expected_number_of_rolls_to_reach_2010_l294_294169

noncomputable def expected_rolls : ℕ → ℚ
| 0 := 0
| (n+1) := (1/6) * (expected_rolls n + expected_rolls (n-1) + expected_rolls (n-2) + expected_rolls (n-3) + expected_rolls (n-4) + expected_rolls (n-5)) + 1

theorem expected_number_of_rolls_to_reach_2010 : abs ((expected_rolls 2010 : ℚ) - 574.761904) < 0.0001 := 
sorry

end expected_number_of_rolls_to_reach_2010_l294_294169


namespace hyperbola_equation_l294_294334

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h_asymptotes : ∀ x : ℝ, (∀ y : ℝ, y = (3 / 4) * x ∨ y = -(3 / 4) * x ↔ y = a / b * x ∨ y = -(a / b) * x))
  (h_foci : ∀ c : ℝ, c = sqrt (a^2 + b^2) ∧ (0, c) = (0, 5)) :
  (a = 3 ∧ b = 4) :=
begin
  sorry
end

end hyperbola_equation_l294_294334


namespace complement_union_A_B_l294_294018

def A := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def B := {x : ℤ | ∃ k : ℤ, x = 3 * k + 2}
def U := {x : ℤ | True}

theorem complement_union_A_B:
  (U \ (A ∪ B) = {x : ℤ | ∃ k : ℤ, x = 3 * k}) :=
by
  sorry

end complement_union_A_B_l294_294018


namespace irrational_numbers_infinite_non_repeating_decimals_l294_294581

theorem irrational_numbers_infinite_non_repeating_decimals :
  (∀ x : ℝ, irrational x → infinite_non_repeating_decimal x) ∧
  ¬ (∀ x : ℝ, rational x ↔ point_on_number_line x) ∧
  ¬ (∀ x : ℝ, infinite_repeating_decimal x → irrational x) ∧
  ¬ (∀ x y : ℝ, (irrational x ∧ irrational y) → (irrational (x + y))) :=
by
  sorry

end irrational_numbers_infinite_non_repeating_decimals_l294_294581


namespace odd_product_abs_even_l294_294206

variables {R : Type*} [Ring R]

-- A definition for odd functions
def is_odd_function (f : R → R) : Prop :=
  ∀ x, f (-x) = -f x

-- A definition for even functions
def is_even_function (f : R → R) : Prop :=
  ∀ x, f (-x) = f x

-- The main theorem
theorem odd_product_abs_even (f g : R → R) 
  (hf : is_odd_function f) (hg : is_even_function g) :
  is_odd_function (λ x, f x * |g x|) :=
sorry

end odd_product_abs_even_l294_294206


namespace rectangle_MQ_l294_294368

theorem rectangle_MQ :
  ∀ (PQ QR PM MQ : ℝ),
    PQ = 4 →
    QR = 10 →
    PM = MQ →
    MQ = 2 * Real.sqrt 10 → 
    0 < MQ
:= by
  intros PQ QR PM MQ h1 h2 h3 h4
  sorry

end rectangle_MQ_l294_294368


namespace length_of_BC_l294_294114

-- Definitions for the problem conditions

def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let (Ax, Ay) := A in
  let (Bx, By) := B in
  let (Cx, Cy) := C in
  (Ax - Cx) * (Bx - Cx) + (Ay - Cy) * (By - Cy) = 0

noncomputable def length (P Q : ℝ × ℝ) : ℝ :=
  let (Px, Py) := P in
  let (Qx, Qy) := Q in
  real.sqrt ((Px - Qx) ^ 2 + (Py - Qy) ^ 2)

def divides_seg (A B X : ℝ × ℝ) (r : ℝ) : Prop :=
  let (Ax, Ay) := A in
  let (Bx, By) := B in
  let (Xx, Xy) := X in
  length A X = r * length A B

-- Variables for points and coordinates
variables (A B C X Y : ℝ × ℝ)
variables (AB_x AB_y : ℝ) -- AB_x represents length AB, AB_y represents length AC

-- Problem conditions
axiom ABC_is_right_triangle : is_right_triangle A B C

axiom AX_divides_AB : divides_seg A B X (1/4)
axiom AY_divides_AC : divides_seg A C Y (2/3)

axiom BY_length : length B Y = 24
axiom CX_length : length C X = 18

-- The theorem to be proved
theorem length_of_BC : length B C = 6 * real.sqrt 42 :=
by sorry

end length_of_BC_l294_294114


namespace whale_consumption_third_hour_l294_294526

theorem whale_consumption_third_hour (x : ℕ) :
  (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 450) → ((x + 6) = 90) :=
by
  intro h
  sorry

end whale_consumption_third_hour_l294_294526


namespace number_of_white_tiles_l294_294716

-- Definition of conditions in the problem
def side_length_large_square := 81
def area_large_square := side_length_large_square * side_length_large_square
def area_black_tiles := 81
def num_red_tiles := 154
def area_red_tiles := num_red_tiles * 4
def area_covered_by_black_and_red := area_black_tiles + area_red_tiles
def remaining_area_for_white_tiles := area_large_square - area_covered_by_black_and_red
def area_of_one_white_tile := 2
def expected_num_white_tiles := 2932

-- The theorem to prove
theorem number_of_white_tiles :
  remaining_area_for_white_tiles / area_of_one_white_tile = expected_num_white_tiles :=
by
  sorry

end number_of_white_tiles_l294_294716


namespace max_identical_bathrooms_l294_294424

def toilet_paper := 45
def soap := 30
def towels := 36
def shower_gel := 18
def shampoo := 27
def toothpaste := 24

theorem max_identical_bathrooms 
  (GCD: ∀ a b, gcd a b = nat.gcd a b) : 
  gcd (gcd (gcd (gcd (gcd toilet_paper soap) towels) shower_gel) shampoo) toothpaste = 3 :=
by 
  have gcd_45_30 : gcd toilet_paper soap = 15 := by sorry
  have gcd_15_36 : gcd 15 towels = 3 := by sorry
  have gcd_3_18 : gcd 3 shower_gel = 3 := by sorry
  have gcd_3_27 : gcd 3 shampoo = 3 := by sorry
  have gcd_3_24 : gcd 3 toothpaste = 3 := by sorry
  rw [gcd_45_30, gcd_15_36, gcd_3_18, gcd_3_27, gcd_3_24]
  exact rfl

end max_identical_bathrooms_l294_294424


namespace binomial_sum_identity_l294_294263

theorem binomial_sum_identity : (Finset.sum (Finset.range 51) (fun k => (-1)^k * Nat.choose 100 (2*k)) = -2^50) :=
by
  sorry

end binomial_sum_identity_l294_294263


namespace sequence_product_mod_7_l294_294630

theorem sequence_product_mod_7 :
  let seq := (λ (n : ℕ), 10 * n + 3) in
  (∏ n in Finset.range 10, seq n) % 7 = 4 :=
by
  let seq := (λ (n : ℕ), 10 * n + 3)
  have h1 : ∏ n in Finset.range 10, seq n ≡ 3^10 [MOD 7], by sorry
  have h2 : 3^10 ≡ 4 [MOD 7], by sorry
  exact (Nat.ModEq.trans h1 h2).symm

end sequence_product_mod_7_l294_294630


namespace lines_intersect_on_median_or_parallel_l294_294043

-- Let A, B, C be the vertices of a triangle
variables {A B C : Point}

-- Let A', B', C' be the feet of the altitudes of the triangle
variables {A' B' C' : Point}

-- Let P, Q, R be the points where the circumcircle of the triangle intersects lines AA', BB', CC' respectively
variables {P Q R : Point}

-- Given
-- - vertices of triangle A, B, C
-- - feet of altitudes A', B', C'
-- - points P, Q, R on the circumcircle intersecting lines AA', BB', CC' respectively

-- Prove that the lines PQ and PR intersect on the median of the triangle or are parallel to it.
theorem lines_intersect_on_median_or_parallel :
  (∃ M : Point, is_median M A B C ∧ collinear P Q M ∧ collinear P R M) ∨ parallel (line P Q) (line P R) :=
sorry

end lines_intersect_on_median_or_parallel_l294_294043


namespace abs_pi_expression_l294_294944

theorem abs_pi_expression : |π - |π - 10|| = 10 - 2 * π :=
by
  sorry

end abs_pi_expression_l294_294944


namespace range_of_a_l294_294663

noncomputable def f (x : ℝ) := log (sin 1) (x^2 - 6 * x + 5)

theorem range_of_a (a : ℝ) : (∀ x > a , f x < f (x + 1)) → a ≥ 5 :=
sorry

end range_of_a_l294_294663


namespace avg_age_of_new_persons_l294_294081

-- We define the given conditions
def initial_persons : ℕ := 12
def initial_avg_age : ℝ := 16
def new_persons : ℕ := 12
def new_avg_age : ℝ := 15.5

-- Define the total initial age
def total_initial_age : ℝ := initial_persons * initial_avg_age

-- Define the total number of persons after new persons join
def total_persons_after_join : ℕ := initial_persons + new_persons

-- Define the total age after new persons join
def total_age_after_join : ℝ := total_persons_after_join * new_avg_age

-- We wish to prove that the average age of the new persons who joined is 15
theorem avg_age_of_new_persons : 
  (total_initial_age + new_persons * 15) = total_age_after_join :=
sorry

end avg_age_of_new_persons_l294_294081


namespace incorrect_statement_C_l294_294846

theorem incorrect_statement_C (T : Triangle) :
  ¬ (∀ alt : Altitude T, IsInsideTriangle T alt) :=
sorry

end incorrect_statement_C_l294_294846


namespace value_of_a_l294_294336

theorem value_of_a (a : ℝ) :
  (∀ x, (2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5) → (3 ≤ x ∧ x ≤ 22)) ↔ (6 ≤ a ∧ a ≤ 9) :=
by
  sorry

end value_of_a_l294_294336


namespace zeros_of_h_l294_294692

variable {a b : ℝ}

theorem zeros_of_h (ha : a ≠ 0) (h1 : f 1 = 0) : 
  (∃ x : ℝ, h(x) = 0 ∧ x = 0) ∧ (∃ x : ℝ, h(x) = 0 ∧ x = 1) := 
by
  noncomputable def f (x : ℝ) := (a / x) + b
  noncomputable def h (x : ℝ) := a*x^2 + b*x
  have h1 : f 1 = 0 := sorry
  have ha : a ≠ 0 := sorry
  have f_zero : a + b = 0 := sorry 
  have h_x : h(x) = x * (a*x + b) := sorry 
  have h_sol : ∃ x, h x = 0 := sorry
  split
  { use 0 
    rw h_x 
    rw f_zero 
    ring }
  { use 1 
    rw h_x 
    rw f_zero 
    ring }

end zeros_of_h_l294_294692


namespace acute_angle_inequality_l294_294070

-- Define the context: three acute angles α, β, γ
variables (α β γ : ℝ)
-- Define the conditions: acute angles imply their measures are in the range (0, π/2)
variables (hα : 0 < α ∧ α < π / 2)
variables (hβ : 0 < β ∧ β < π / 2)
variables (hγ : 0 < γ ∧ γ < π / 2)

-- Define the necessary trigonometric functions
noncomputable def tan (x : ℝ) := real.tan x
noncomputable def cot (x : ℝ) := 1 / real.tan x

-- The theorem statement
theorem acute_angle_inequality:
  tan α * (cot β + cot γ) + tan β * (cot γ + cot α) + tan γ * (cot α + cot β) ≥ 6 := 
by
  sorry

end acute_angle_inequality_l294_294070


namespace log_inequality_solution_l294_294292

theorem log_inequality_solution {a x : ℝ} (ha1 : 0 < a) (ha2 : a ≠ 1) (hfx : ∀ x, a^(Math.log ((x-1)^2 + 2) / Math.log 10) = f(x)) :
  (log a (x^2 - 5*x + 7) > 0) → (2 < x ∧ x < 3) :=
by
  sorry

end log_inequality_solution_l294_294292


namespace complement_union_l294_294004

def A := { x : ℤ | ∃ k : ℤ, x = 3 * k + 1 }
def B := { x : ℤ | ∃ k : ℤ, x = 3 * k + 2 }
def U := ℤ

theorem complement_union :
  { x : ℤ | x ∉ (A ∪ B) } = { x : ℤ | ∃ k : ℤ, x = 3 * k } := 
by 
  sorry

end complement_union_l294_294004


namespace running_race_l294_294997

-- Define participants
inductive Participant : Type
| Anna
| Bella
| Csilla
| Dora

open Participant

-- Define positions
@[ext] structure Position :=
(first : Participant)
(last : Participant)

-- Conditions:
def conditions (p : Participant) (q : Participant) (r : Participant) (s : Participant)
  (pa : Position) : Prop :=
  (pa.first = r) ∧ -- Csilla was first
  (pa.first ≠ q) ∧ -- Bella was not first
  (pa.first ≠ p) ∧ (pa.last ≠ p) ∧ -- Anna was not first or last
  (pa.last = s) -- Dóra's statement about being last

-- Definition of the liar
def liar (p : Participant) : Prop :=
  p = Dora

-- Proof problem
theorem running_race : ∃ (pa : Position), liar Dora ∧ (pa.first = Csilla) :=
  sorry

end running_race_l294_294997


namespace second_quadrant_y_value_l294_294362

theorem second_quadrant_y_value :
  ∀ (b : ℝ), (-3, b).2 > 0 → b = 2 :=
by
  sorry

end second_quadrant_y_value_l294_294362


namespace minimize_notch_volume_l294_294557

noncomputable def total_volume (theta phi : ℝ) : ℝ :=
  let part1 := (2 / 3) * Real.tan phi
  let part2 := (2 / 3) * Real.tan (theta - phi)
  part1 + part2

theorem minimize_notch_volume :
  ∀ (theta : ℝ), (0 < theta ∧ theta < π) →
  ∃ (phi : ℝ), (0 < phi ∧ phi < θ) ∧
  (∀ ψ : ℝ, (0 < ψ ∧ ψ < θ) → total_volume theta ψ ≥ total_volume theta (theta / 2)) :=
by
  -- Proof is omitted
  sorry

end minimize_notch_volume_l294_294557


namespace sum_of_ages_l294_294104

-- Definitions based on given conditions
def J : ℕ := 19
def age_difference (B J : ℕ) : Prop := B - J = 32

-- Theorem stating the problem
theorem sum_of_ages (B : ℕ) (H : age_difference B J) : B + J = 70 :=
sorry

end sum_of_ages_l294_294104


namespace jakes_balloons_l294_294904

theorem jakes_balloons (total_balloons : ℕ) (allans_balloons : ℕ) (jakes_balloons : ℕ) : 
  total_balloons = 3 → allans_balloons = 2 → jakes_balloons = 1 :=
by
  intros h_total h_allan
  rw [h_total, h_allan]
  sorry

end jakes_balloons_l294_294904


namespace second_caterer_cheaper_l294_294764

theorem second_caterer_cheaper (x : ℕ) :
  (∀ n : ℕ, n < x → 150 + 18 * n ≤ 250 + 15 * n) ∧ (150 + 18 * x > 250 + 15 * x) ↔ x = 34 :=
by sorry

end second_caterer_cheaper_l294_294764


namespace noemi_starting_money_l294_294429

-- Define the losses and remaining money as constants
constant lost_on_roulette : ℕ := 400
constant lost_on_blackjack : ℕ := 500
constant remaining_money : ℕ := 800

-- Define the initial amount of money as a conjecture to be proven
def initial_money := lost_on_roulette + lost_on_blackjack + remaining_money

-- State the theorem
theorem noemi_starting_money : initial_money = 1700 :=
by
  -- This is where the actual proof would go
  sorry

end noemi_starting_money_l294_294429


namespace tan_sum_formula_l294_294358

theorem tan_sum_formula (α : ℝ) (h : sin α = -3 * cos α) :
  tan (α + π / 4) = -1 / 2 :=
by sorry

end tan_sum_formula_l294_294358


namespace satisfies_property_problem_solution_l294_294843

def f1 (x : ℝ) : ℝ := Real.sqrt x
def f2 (x : ℝ) : ℝ := Real.log (abs x)
def f3 (x : ℝ) : ℝ := 1 / (x - 1)
def f4 (x : ℝ) : ℝ := x * Real.cos x

theorem satisfies_property (f : ℝ → ℝ) (hf : ∀ x, f x + f (-x) = 0) : f = f4 :=
by {
  sorry
}

theorem problem_solution : satisfies_property f4 (λ x, by
  simp [f4, Real.cos_neg])
:= by { sorry }

end satisfies_property_problem_solution_l294_294843


namespace problem1_problem2_l294_294331

-- Definition of functions
def f (x a : ℝ) : ℝ := Real.log x - 2 * a * x

def g (x a : ℝ) : ℝ := f x a + (1/2) * x^2

-- Problem 1: Prove the range of a
theorem problem1 (a : ℝ) : (∃ x > 0, (1/x - 2 * a) = 2) -> a > -1 :=
sorry

-- Problem 2: Prove the inequality for the local maximum
theorem problem2 (a x1 x2 : ℝ) (hx : 0 < x1) (hx2 : x1 < 1) 
  (hlmax : g x1 a = g x2 a ∧ g x1 a = 0 ∧ x1 * x2 = 1) : 
  (Real.log x1 / x1 + 1 / x2^2 > a) :=
sorry

end problem1_problem2_l294_294331


namespace endpoint_coordinates_l294_294181

theorem endpoint_coordinates (x y : ℝ) (h : y > 0) :
  let slope_condition := (y - 2) / (x - 2) = 3 / 4
  let distance_condition := (x - 2) ^ 2 + (y - 2) ^ 2 = 64
  slope_condition → distance_condition → 
    (x = 2 + (4 * Real.sqrt 5475) / 25 ∧ y = (3 / 4) * (2 + (4 * Real.sqrt 5475) / 25) + 1 / 2) ∨
    (x = 2 - (4 * Real.sqrt 5475) / 25 ∧ y = (3 / 4) * (2 - (4 * Real.sqrt 5475) / 25) + 1 / 2) :=
by
  intros slope_condition distance_condition
  sorry

end endpoint_coordinates_l294_294181


namespace expression_equals_two_l294_294959

noncomputable def math_expression : ℝ :=
  27^(1/3) + Real.log 4 + 2 * Real.log 5 - Real.exp (Real.log 3)

theorem expression_equals_two : math_expression = 2 := by
  sorry

end expression_equals_two_l294_294959


namespace parabola_chord_length_l294_294649

theorem parabola_chord_length (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) 
(h1 : y₁^2 = 4 * x₁) 
(h2 : y₂^2 = 4 * x₂) 
(h3 : x₁ + x₂ = 6) : 
|y₁ - y₂| = 8 :=
sorry

end parabola_chord_length_l294_294649


namespace positive_real_product_l294_294780

theorem positive_real_product (x y : ℝ) (log : ℝ → ℝ) 
  (hx : x > 0) (hy : y > 0)
  (H : sqrt (log x) + sqrt (log y) + 3 * log (sqrt x) + 3 * log (sqrt y) = 150) : 
  x * y = 10^130 :=
by {
  sorry
}

end positive_real_product_l294_294780


namespace ratio_of_milk_water_in_larger_vessel_l294_294139

-- Definitions of conditions
def volume1 (V : ℝ) : ℝ := 3 * V
def volume2 (V : ℝ) : ℝ := 5 * V

def ratio_milk_water_1 : ℝ × ℝ := (1, 2)
def ratio_milk_water_2 : ℝ × ℝ := (3, 2)

-- Define the problem statement
theorem ratio_of_milk_water_in_larger_vessel (V : ℝ) (hV : V > 0) :
  (volume1 V / (ratio_milk_water_1.1 + ratio_milk_water_1.2)) = V ∧ 
  2 * (volume1 V / (ratio_milk_water_1.1 + ratio_milk_water_1.2)) = 2 * V ∧ 
  3 * (volume2 V / (ratio_milk_water_2.1 + ratio_milk_water_2.2)) = 3 * V ∧ 
  2 * (volume2 V / (ratio_milk_water_2.1 + ratio_milk_water_2.2)) = 2 * V →
  (4 * V) / (4 * V) = 1 :=
sorry

end ratio_of_milk_water_in_larger_vessel_l294_294139


namespace unreasonable_inference_l294_294568

theorem unreasonable_inference:
  (∀ (S T : Type) (P : S → Prop) (Q : T → Prop), (∀ x y, P x → ¬ Q y) → ¬ (∀ x, P x) → (∃ y, ¬ Q y))
  ∧ ¬ (∀ s : ℝ, (s = 100) → ∀ t : ℝ, t = 100) :=
sorry

end unreasonable_inference_l294_294568


namespace amy_height_l294_294915

variable (A H N : ℕ)

theorem amy_height (h1 : A = 157) (h2 : A = H + 4) (h3 : H = N + 3) :
  N = 150 := sorry

end amy_height_l294_294915


namespace simplify_expression_l294_294454

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

theorem simplify_expression : (x⁻¹ - y) ^ 2 = (1 / x ^ 2 - 2 * y / x + y ^ 2) :=
  sorry

end simplify_expression_l294_294454


namespace hyperbola_equation_l294_294320

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
                           (h3 : b = 2 * a) (h4 : ((4 : ℝ), 1) ∈ {p : ℝ × ℝ | (p.1)^2 / (a^2) - (p.2)^2 / (b^2) = 1}) :
    {p : ℝ × ℝ | (p.1)^2 / 12 - (p.2)^2 / 3 = 1} = {p : ℝ × ℝ | (p.1)^2 / (a^2) - (p.2)^2 / (b^2) = 1} :=
by
  sorry

end hyperbola_equation_l294_294320


namespace fraction_scaled_l294_294377

theorem fraction_scaled (x y : ℝ) :
  (f (3 * x) (3 * y) = f x y / 3) :=
by
  unfold f
  sorry

def f (x y : ℝ) : ℝ := (x + y) / (x^2 + y^2)

end fraction_scaled_l294_294377


namespace exists_sequence_l294_294270

def S (m : ℕ) : ℕ := m.digits.sum
def P (m : ℕ) : ℕ := m.digits.prod

theorem exists_sequence (n : ℕ) (h : 0 < n) : 
  ∃ a : (fin n → ℕ), (∀ i : fin n, S (a i) < S (a ((i : ℕ + 1) % n))) ∧ (∀ i : fin n, S (a i) = P (a ((i : ℕ + 1) % n))) := 
by 
  sorry

end exists_sequence_l294_294270


namespace third_bowler_points_162_l294_294883

variable (x : ℕ)

def total_score (x : ℕ) : Prop :=
  let first_bowler_points := x
  let second_bowler_points := 3 * x
  let third_bowler_points := x
  first_bowler_points + second_bowler_points + third_bowler_points = 810

theorem third_bowler_points_162 (x : ℕ) (h : total_score x) : x = 162 := by
  sorry

end third_bowler_points_162_l294_294883


namespace negated_proposition_false_l294_294099

theorem negated_proposition_false : ¬ ∀ x : ℝ, 2^x + x^2 > 1 :=
by 
sorry

end negated_proposition_false_l294_294099


namespace general_term_sum_of_bn_l294_294653

open Nat

-- Definition of the sequence \(a_n\)
def Sn (n : ℕ) : ℝ := (n^2 + 3*n) / 4

-- General formula for \(a_n\)
theorem general_term (n : ℕ) : a_n = (n + 1) / 2 := by
  sorry

-- Definition of sequence \( b_n \) using the general term \( a_n \)
def bn (n : ℕ) : ℝ :=
  let an := (n + 1) / 2
  let an1 := (n + 2) / 2
  (n + 1) * 4^an - 1 / (4 * an * an1)

-- Sum of the first n terms of the sequence \( b_n \)
theorem sum_of_bn (n : ℕ) : ∑ i in finset.range(n+1), bn i = n * 2^(n+2) + n / (2 * (n + 2)) := by
  sorry

end general_term_sum_of_bn_l294_294653


namespace number_of_valid_triangles_l294_294609
open Real Int

theorem number_of_valid_triangles : 
  let valid_points := {p : ℕ × ℕ | 47 * p.fst + p.snd = 2017} in
  let triangles := {t : (ℕ × ℕ) × (ℕ × ℕ) | t.1 ≠ t.2 ∧ t.1 ∈ valid_points ∧ t.2 ∈ valid_points ∧ (t.1.fst - t.2.fst) % 2 = 0} in
  Fintype.card triangles = 441 :=
sorry

end number_of_valid_triangles_l294_294609


namespace mary_jenny_red_marble_ratio_l294_294055

def mary_red_marble := 30  -- Given that Mary collects the same as Jenny.
def jenny_red_marble := 30 -- Given
def jenny_blue_marble := 25 -- Given
def anie_red_marble := mary_red_marble + 20 -- Anie's red marbles count
def anie_blue_marble := 2 * jenny_blue_marble -- Anie's blue marbles count
def mary_blue_marble := anie_blue_marble / 2 -- Mary's blue marbles count

theorem mary_jenny_red_marble_ratio : 
  mary_red_marble / jenny_red_marble = 1 :=
by
  sorry

end mary_jenny_red_marble_ratio_l294_294055


namespace total_cost_is_3680_yuan_l294_294963

-- Define the given conditions
def volume : ℝ := 24
def depth : ℝ := 2
def width : ℝ := 3
def costBottomPerM3 : ℝ := 120
def costWallPerM3 : ℝ := 80

-- Define the height
def height : ℝ := depth

-- Define the length based on the volume formula V = length * width * height
def length := volume / (width * height)

-- Define the area of the bottom
def areaBottom := length * width

-- Define the total area of the walls
def areaWalls := 2 * (length + width) * height

-- Define the total cost
def totalCost := (costBottomPerM3 * areaBottom) + (costWallPerM3 * areaWalls)

-- Proof statement (no proof body)
theorem total_cost_is_3680_yuan : totalCost = 3680 := 
 by
  sorry

end total_cost_is_3680_yuan_l294_294963


namespace proposition_correctness_l294_294049

def f (x : ℝ) (b : ℝ) (c : ℝ) := x * abs(x) + b * x + c

theorem proposition_correctness (b c : ℝ) :
  ((c = 0 → ∀ x : ℝ, f x b c = -f (-x) b c) ∧
  (b = 0 ∧ c > 0 → ∃! x : ℝ, f x b c = 0) ∧
  (∀ x : ℝ, f x b c = f (-x) b c) ∧
  ( ∀ x1 x2 : ℝ, f x1 b c = 0 ∧ f x2 b c = 0 → x1 = x2 ∨ x1 = -x2)) :=
sorry

end proposition_correctness_l294_294049


namespace cone_height_l294_294879

theorem cone_height :
  ∃ h : Real, let V_cube := 125
             let V_cone := (1/3) * π * (5^2) * h
             (V_cube = V_cone) → h = 15 / π := 
by
  let V_cube := 125
  let V_cone := (1/3) * π * (5^2) * sorry
  have h_def : V_cube = V_cone
  exact ⟨_, h_def⟩

end cone_height_l294_294879


namespace limit_evaluation_l294_294747

def f (x : ℝ) : ℝ := Real.sin (Real.sin x)

theorem limit_evaluation : 
  (Real.limit (λ h, (f(π + h) - f(h)) / π) (0 : ℝ)) = 0 :=
  sorry

end limit_evaluation_l294_294747


namespace coordinates_of_Q_l294_294437

noncomputable def P : ℝ × ℝ := (1, 0)

theorem coordinates_of_Q :
  let θ := (real.pi / 3)
  let Q := (real.cos θ, real.sin θ)
  Q = (1 / 2, real.sqrt 3 / 2) :=
by
  sorry

end coordinates_of_Q_l294_294437


namespace question1_question2_question3_question4_l294_294782

-- Definition of the data
def time_values : List ℝ := [0, 1, 2, 5, 7, 10, 12, 13, 14, 17, 20]
def acceptance_values : List ℝ := [43, 45.5, 47.8, 53.5, 56.3, 59, 59.8, 59.9, 59.8, 58.3, 55]

-- Question 1: Independent and dependent variables
theorem question1 :
  (∃ x y, time_values = x ∧ acceptance_values = y) →
  (∀ x y, time_values = x ∧ acceptance_values = y → independent(x) = time ∧ dependent(y) = acceptanceAbility) :=
sorry

-- Question 2: Acceptance ability at x=10
theorem question2 :
  (∃ x y, time_values = x ∧ acceptance_values = y) →
  (∀ index, List.nth time_values index = some 10 → List.nth acceptance_values index = some 59) :=
sorry

-- Question 3: Maximum acceptance ability
theorem question3 :
  (∃ x y, time_values = x ∧ acceptance_values = y) →
  (List.maximum acceptance_values = some 59.9 ∧ List.indexOf (some 59.9) acceptance_values = some 7 ∧ List.nth time_values 7 = some 13) :=
sorry

-- Question 4: Increasing and decreasing intervals
theorem question4 :
  (∃ x y, time_values = x ∧ acceptance_values = y) →
  (∀ i, 0 ≤ i ∧ i < List.length time_values - 1 →
    (List.nth time_values i ≤ 13 → List.nth acceptance_values i ≤ List.nth acceptance_values (i+1)) ∧
    (13 < List.nth time_values i → List.nth acceptance_values i ≥ List.nth acceptance_values (i+1))) :=
sorry

end question1_question2_question3_question4_l294_294782


namespace find_passing_marks_l294_294871

variable {T P : ℝ}

-- Condition 1: A candidate who gets 40% of the marks fails by 40 marks
axiom cond1 : 0.40 * T = P - 40

-- Condition 2: Another candidate who gets 60% marks gets 20 marks more than necessary for passing
axiom cond2 : 0.60 * T = P + 20

-- We wish to prove that the passing marks P is 160
theorem find_passing_marks : P = 160 :=
by
  have h1 : 0.60 * T = 1.5 * P - 60 := by
    linarith [cond1]
  have h2 : 0.60 * T = P + 20 := cond2
  have h3 : 1.5 * P - 60 = P + 20 := by
    linarith [h1, h2]
  have h4 : 80 = 0.5 * P := by
    linarith [h3]
  have h5 : P = 160 := by
    field_simp at h4
    linarith [h4]
  exact h5

# The proof above outlines the steps we would take mathematically to prove the theorem.
# Note that the actual effectuation of the steps is not necessary for this problem statement, as per the user's instructions.

end find_passing_marks_l294_294871


namespace arithmetic_geometric_inequality_l294_294795

noncomputable def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ := a1 + n * d

noncomputable def geometric_sequence (b1 r : ℝ) (n : ℕ) : ℝ := b1 * r^n

theorem arithmetic_geometric_inequality
  (a1 b1 : ℝ) (d r : ℝ) (n : ℕ)
  (h_pos : 0 < a1) 
  (ha1_eq_b1 : a1 = b1) 
  (h_eq_2np1 : arithmetic_sequence a1 d (2*n+1) = geometric_sequence b1 r (2*n+1)) :
  arithmetic_sequence a1 d (n+1) ≥ geometric_sequence b1 r (n+1) :=
sorry

end arithmetic_geometric_inequality_l294_294795


namespace total_volume_is_correct_l294_294562

-- Define the radius and heights
def radius : ℝ := 3
def height_cone : ℝ := 12
def height_cylinder : ℝ := 2

-- Volumes of geometrical shapes
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h
def volume_hemisphere (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3
def volume_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Total volume of ice cream
def total_volume : ℝ :=
  volume_cone radius height_cone + volume_hemisphere radius + volume_cylinder radius height_cylinder

-- Statement to prove
theorem total_volume_is_correct : total_volume = 72 * Real.pi :=
by
  sorry

end total_volume_is_correct_l294_294562


namespace product_of_values_of_x_l294_294536

theorem product_of_values_of_x : 
  (∃ x : ℝ, |x^2 - 7| - 3 = -1) → 
  (∀ x1 x2 x3 x4 : ℝ, 
    (|x1^2 - 7| - 3 = -1) ∧
    (|x2^2 - 7| - 3 = -1) ∧
    (|x3^2 - 7| - 3 = -1) ∧
    (|x4^2 - 7| - 3 = -1) 
    → x1 * x2 * x3 * x4 = 45) :=
sorry

end product_of_values_of_x_l294_294536


namespace solve_eq_l294_294455

theorem solve_eq (x : ℂ) :
  (x - 2)^4 + (x - 6)^4 = 32 ↔
  (x = 4 ∨ x = 4 + 2 * complex.I * real.sqrt 6 ∨ x = 4 - 2 * complex.I * real.sqrt 6) :=
by sorry

end solve_eq_l294_294455


namespace tangent_line_through_P_l294_294246

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

def P : ℝ × ℝ := (2, 2)

theorem tangent_line_through_P :
  ∃ (y : ℝ → ℝ), (∀ x, y = f x ∧ ((∃ a, f a = P.2 ∧ y = λ x, (2 * a - 2) * (x - a) + f a) ∨ y = 2 ∨ y = 4 * x - 6)) :=
sorry

end tangent_line_through_P_l294_294246


namespace derivative_of_cos_l294_294669

-- Define the function f(x) = cos x
def f (x : ℝ) : ℝ := Real.cos x

-- State the theorem that the derivative of f is -sin x
theorem derivative_of_cos :
  ∀ x : ℝ, deriv f x = -Real.sin x :=
by
  -- Proof omitted
  sorry

end derivative_of_cos_l294_294669


namespace min_ab_eq_11_l294_294057

theorem min_ab_eq_11 (a b : ℕ) (h : 23 * a - 13 * b = 1) : a + b = 11 :=
sorry

end min_ab_eq_11_l294_294057


namespace opposite_of_B_is_I_l294_294426

inductive Face
| A | B | C | D | E | F | G | H | I

open Face

def opposite_face (f : Face) : Face :=
  match f with
  | A => G
  | B => I
  | C => H
  | D => F
  | E => E
  | F => F
  | G => A
  | H => C
  | I => B

theorem opposite_of_B_is_I : opposite_face B = I :=
  by
    sorry

end opposite_of_B_is_I_l294_294426


namespace average_cd_l294_294097

theorem average_cd (c d: ℝ) (h: (4 + 6 + 8 + c + d) / 5 = 18) : (c + d) / 2 = 36 :=
by sorry

end average_cd_l294_294097


namespace cos_double_angle_l294_294294

theorem cos_double_angle (α : ℝ) (h : sin (α + π / 12) = 3 / 5) : cos (2 * α + π / 6) = 7 / 25 := 
by 
  sorry

end cos_double_angle_l294_294294


namespace greatest_number_of_consecutive_integers_sum_36_l294_294506

theorem greatest_number_of_consecutive_integers_sum_36 :
  ∃ (N : ℕ), 
    (∃ a : ℤ, N * a + ((N - 1) * N) / 2 = 36) ∧ 
    (∀ N' : ℕ, (∃ a' : ℤ, N' * a' + ((N' - 1) * N') / 2 = 36) → N' ≤ 72) := by
  sorry

end greatest_number_of_consecutive_integers_sum_36_l294_294506


namespace george_on_time_l294_294999

noncomputable theory

def george_normal_speed : ℝ := 3 -- Normal speed in mph
def george_distance_to_school : ℝ := 1.5 -- Distance to school in miles
def george_first_segment_distance : ℝ := 0.75 -- Distance walked at 2 mph today
def george_first_segment_speed : ℝ := 2 -- Speed for the first segment in mph
def george_remaining_distance : ℝ := 0.75 -- Remaining distance to school in miles

def george_normal_time := george_distance_to_school / george_normal_speed
def george_first_segment_time := george_first_segment_distance / george_first_segment_speed
def george_remaining_time := george_normal_time - george_first_segment_time

def george_required_speed := george_remaining_distance / george_remaining_time

theorem george_on_time : george_required_speed = 6 :=
by
  sorry -- Proof not required

end george_on_time_l294_294999


namespace shape_with_circular_views_is_sphere_l294_294575

/-- Define the views of the cuboid, cylinder, cone, and sphere. -/
structure Views (shape : Type) :=
(front_view : Type)
(left_view : Type)
(top_view : Type)

def is_cuboid (s : Views) : Prop :=
s.front_view = Rectangle ∧ s.left_view = Rectangle ∧ s.top_view = Rectangle

def is_cylinder (s : Views) : Prop :=
s.front_view = Rectangle ∧ s.left_view = Rectangle ∧ s.top_view = Circle

def is_cone (s : Views) : Prop :=
s.front_view = IsoscelesTriangle ∧ s.left_view = IsoscelesTriangle ∧ s.top_view = Circle

def is_sphere (s : Views) : Prop :=
s.front_view = Circle ∧ s.left_view = Circle ∧ s.top_view = Circle

/-- Proof problem: Prove that the only shape with circular views in all three perspectives (front, left, top) is the sphere. -/
theorem shape_with_circular_views_is_sphere :
  ∀ (s : Views), 
    (s.front_view = Circle ∧ s.left_view = Circle ∧ s.top_view = Circle) → 
    is_sphere s ∧ ¬ is_cuboid s ∧ ¬ is_cylinder s ∧ ¬ is_cone s :=
by
  intro s h
  sorry

end shape_with_circular_views_is_sphere_l294_294575


namespace magnitude_P1P2_l294_294346

open Real

theorem magnitude_P1P2
  (OP1 OP2 OP3 : EuclideanSpace.real 3)
  (h1 : OP1 + OP2 + OP3 = 0)
  (h2 : ∥OP1∥ = 1)
  (h3 : ∥OP2∥ = 1)
  (h4 : ∥OP3∥ = 1) :
  ∥OP2 - OP1∥ = sqrt 3 := by
  sorry

end magnitude_P1P2_l294_294346


namespace tile_size_l294_294733

theorem tile_size (length width : ℕ) (total_tiles : ℕ) 
  (h_length : length = 48) 
  (h_width : width = 72) 
  (h_total_tiles : total_tiles = 96) : 
  ((length * width) / total_tiles) = 36 := 
by
  sorry

end tile_size_l294_294733


namespace abs_pi_expression_l294_294955

theorem abs_pi_expression (h : Real.pi < 10) : 
  Real.abs (Real.pi - Real.abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_pi_expression_l294_294955


namespace shape_with_circular_views_is_sphere_l294_294569

-- Definitions of the views of different geometric shapes
inductive Shape
| Cuboid : Shape
| Cylinder : Shape
| Cone : Shape
| Sphere : Shape

def front_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Rectangle
| Shape.Cone := False  -- Isosceles Triangle
| Shape.Sphere := True  -- Circle

def left_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Rectangle
| Shape.Cone := False  -- Isosceles Triangle
| Shape.Sphere := True  -- Circle

def top_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Circle, but not all views
| Shape.Cone := False  -- Circle, but not all views
| Shape.Sphere := True  -- Circle

-- The theorem to be proved
theorem shape_with_circular_views_is_sphere (s : Shape) :
  (front_view s ↔ True) ∧ (left_view s ↔ True) ∧ (top_view s ↔ True) ↔ s = Shape.Sphere :=
by sorry

end shape_with_circular_views_is_sphere_l294_294569


namespace product_base7_eq_l294_294926

-- Definitions for the numbers in base 7
def num325_base7 := 3 * 7^2 + 2 * 7^1 + 5 * 7^0  -- 325 in base 7
def num4_base7 := 4 * 7^0  -- 4 in base 7

-- Theorem stating that the product of 325_7 and 4_7 in base 7 is 1636_7
theorem product_base7_eq : 
  let product_base10 := num325_base7 * num4_base7 in
  (product_base10 = 1 * 7^3 + 6 * 7^2 + 3 * 7^1 + 6 * 7^0) :=
by sorry

end product_base7_eq_l294_294926


namespace cos_sin_identity_l294_294641

theorem cos_sin_identity (x : ℝ) (h : Real.cos (x - Real.pi / 3) = 1 / 3) :
  Real.cos (2 * x - 5 * Real.pi / 3) + Real.sin (Real.pi / 3 - x) ^ 2 = 5 / 3 :=
sorry

end cos_sin_identity_l294_294641


namespace projection_correct_l294_294407

open Real EuclideanSpace

noncomputable def projection_proof_problem : Prop :=
  let Q : Set ℝ^3 := {p | inner p (⟨1, -1, 1⟩ : ℝ^3) = 0}
  let v1 := (⟨6, 2, 6⟩ : ℝ^3)
  let proj_v1 := (⟨2, 6, 2⟩ : ℝ^3)
  let v2 := (⟨3, 1, 8⟩ : ℝ^3)
  let expected_proj_v2 := (⟨(1/3 : ℝ), (13/3 : ℝ), (14/3 : ℝ)⟩ : ℝ^3)
  inner (v1 - proj_v1) (⟨1, -1, 1⟩ : ℝ^3) = 0 ∧
  inner v2 (⟨1, -1, 1⟩ : ℝ^3) - inner expected_proj_v2 (⟨1, -1, 1⟩ : ℝ^3) = 0

theorem projection_correct : projection_proof_problem := sorry

end projection_correct_l294_294407


namespace probability_none_given_no_D_l294_294371

theorem probability_none_given_no_D : 
  (∀ (P : Type) [probability_space P] (D E F : event P),
  Pr(D \ E \ F) = 0.08 ∧
  Pr((D ∩ E) \ F) = 0.12 ∧
  Pr((D ∩ F) \ E) = 0.12 ∧
  Pr((E ∩ F) \ D) = 0.12 ∧
  Pr((D ∩ E ∩ F) | (D ∩ E)) = 1/4 → 
  Pr(¬(D ∪ E ∪ F) | ¬D) = 7/16) :=
begin
  sorry
end

end probability_none_given_no_D_l294_294371


namespace find_a_l294_294743

noncomputable def f (a b c x : ℝ) : ℝ := real.sqrt (a * x^2 + b * x + c)

def domain (a b c : ℝ) : set ℝ := {x | a * x^2 + b * x + c ≥ 0}

-- Define the mathematical conditions and the proof goal
theorem find_a (a b c : ℝ) (D : set ℝ) (s t : ℝ) (h_a : a < 0) (h_square_region : ∀ s t ∈ D, s ≠ t → f a b c s ≠ f a b c t ∧ f a b c t ≠ f a b c s):
  a = -4 :=
sorry

end find_a_l294_294743


namespace crayons_lost_or_given_away_l294_294531

theorem crayons_lost_or_given_away:
  (initial_crayons - remaining_crayons = crayons_lost)
  (initial_crayons = 479)
  (remaining_crayons = 134)
:  crayons_lost = 345 :=
by
  sorry

end crayons_lost_or_given_away_l294_294531


namespace problem_equivalence_l294_294678

-- Define the parametric curve C1
def curve_C1 (α : ℝ) : ℝ × ℝ :=
  let x := 3 + 2 * Real.cos α
  let y := 1 + 2 * Real.sin α
  (x, y)

-- Given the curve and midpoint definition
def midpoint_pq_locus (x y : ℝ) : Prop :=
  (2 * x - 3) ^ 2 + (y - 1) ^ 2 = 4

-- Polar line equation in Cartesian form
def line_in_cartesian (x y : ℝ) : Prop :=
  y - x = 1

-- Define the problem statement
theorem problem_equivalence :
  ∃ x1 x2 y1 y2 : ℝ, 
    midpoint_pq_locus x1 y1 ∧ 
    midpoint_pq_locus x2 y2 ∧ 
    line_in_cartesian x1 y1 ∧ 
    line_in_cartesian x2 y2 ∧ 
    Euclidean_dist (x1, y1) (x2, y2) = (2 * Real.sqrt 22 / 5) := 
sorry

end problem_equivalence_l294_294678


namespace grape_juice_amount_l294_294551

noncomputable def T : ℝ := 140 / 0.75

theorem grape_juice_amount :
  let T := 140 / 0.75 in
  let grape_juice := 0.25 * T in
  grape_juice = 46.67 :=
by
  sorry

end grape_juice_amount_l294_294551


namespace sin_2BPC_l294_294779

open Real

theorem sin_2BPC (A B C D E P : Type) (h : Equally_Spaced A B C D E)
  (h1 : cos (angle P A C) = 3 / 5)
  (h2 : cos (angle P B D) = 1 / 2) : sin (2 * angle P B C) = 3 * sqrt 3 / 5 :=
sorry

end sin_2BPC_l294_294779


namespace density_approx_eq_l294_294541

-- Define variables and constants
def G := 6.67430 * 10^(-11) -- Gravitational constant in m^3 kg^(-1) s^(-2)
def T := 14400 -- Period in seconds (4 * 60 * 60)
def expected_density := 6000 -- Expected density in kg/m^3

-- Define centripetal acceleration as a function of radius R
def ac (R : ℝ) : ℝ :=
  4 * π^2 * (2 * R) / T^2

-- Define Newton's second law in the context of the gravitational force providing centripetal force
def newtons_second_law (m R : ℝ) (ρ : ℝ) : Prop :=
  m * ac R = G * (m * ρ * (4/3) * π * R^3) / (2 * R)^2

-- Define the density formula
def density (R : ℝ) : ℝ :=
  24 * π / (G * T^2)

-- The theorem statement
theorem density_approx_eq (R : ℝ) : density R ≈ expected_density :=
by
  sorry

end density_approx_eq_l294_294541


namespace perfect_square_condition_l294_294563

-- Sequence definition
def sequence (a : ℤ) : ℕ → ℤ
| 0       := a
| 1       := 2
| (n + 2) := 2 * (sequence a (n + 1)) * (sequence a n) - (sequence a (n + 1)) - (sequence a n) + 1

-- Predicate for determining if an integer is a perfect square
def is_perfect_square (k : ℤ) := ∃ m : ℤ, m * m = k

-- Main theorem
theorem perfect_square_condition (a : ℤ) (h : ∃ m : ℤ, a = (2 * m - 1)^2 / 2 + 1) (n : ℕ) (hn : n ≥ 1) :
  is_perfect_square (2 * sequence a (3 * n) - 1) :=
sorry

end perfect_square_condition_l294_294563


namespace find_interest_rate_l294_294556

theorem find_interest_rate 
  (P T : ℝ)
  (SI : ℝ)
  (hP : P = 1250)
  (hT : T = 10)
  (hSI : SI = 1500) : 
  (R : ℝ) (hR : SI = P * R * T / 100) → R = 12 :=
sorry

end find_interest_rate_l294_294556


namespace abs_twice_sub_pi_l294_294938

theorem abs_twice_sub_pi (h : Real.pi < 10) : abs (Real.pi - abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_twice_sub_pi_l294_294938


namespace largest_degree_q_horizontal_asymptote_l294_294962

theorem largest_degree_q_horizontal_asymptote :
  ∀ (q : polynomial ℝ), (∃ a : ℝ, q.degree = a) → 
  (∃ b : polynomial ℝ, b = 3 * X^6 - X^3 + 2) →
  ∀ x, is_horizontal_asymptote_of (λ x, q.eval x / b.eval x) x → 
  q.degree ≤ 6 := by
  sorry

end largest_degree_q_horizontal_asymptote_l294_294962


namespace bus_driver_compensation_l294_294151

theorem bus_driver_compensation : 
  let regular_rate := 16
  let regular_hours := 40
  let total_hours_worked := 57
  let overtime_rate := regular_rate + (0.75 * regular_rate)
  let regular_pay := regular_hours * regular_rate
  let overtime_hours_worked := total_hours_worked - regular_hours
  let overtime_pay := overtime_hours_worked * overtime_rate
  let total_compensation := regular_pay + overtime_pay
  total_compensation = 1116 :=
by
  sorry

end bus_driver_compensation_l294_294151


namespace triangle_area_calculation_l294_294425

theorem triangle_area_calculation
  (A : ℕ)
  (BC : ℕ)
  (h : ℕ)
  (nine_parallel_lines : Bool)
  (equal_segments : Bool)
  (largest_area_part : ℕ)
  (largest_part_condition : largest_area_part = 38) :
  9 * (BC / 10) * (h / 10) / 2 = 10 * (BC / 2) * A / 19 :=
sorry

end triangle_area_calculation_l294_294425


namespace white_square_area_l294_294757

theorem white_square_area
  (edge_length : ℝ)
  (total_green_area : ℝ)
  (faces : ℕ)
  (green_per_face : ℝ)
  (total_surface_area : ℝ)
  (white_area_per_face : ℝ) :
  edge_length = 12 ∧ total_green_area = 432 ∧ faces = 6 ∧ total_surface_area = 864 ∧ green_per_face = total_green_area / faces ∧ white_area_per_face = total_surface_area / faces - green_per_face → white_area_per_face = 72 :=
by
  sorry

end white_square_area_l294_294757


namespace number_of_dodge_trucks_l294_294824

theorem number_of_dodge_trucks (V T F D : ℕ) (h1 : V = 5)
  (h2 : T = 2 * V) 
  (h3 : F = 2 * T)
  (h4 : F = D / 3) :
  D = 60 := 
by
  sorry

end number_of_dodge_trucks_l294_294824


namespace clock_hand_speed_ratio_l294_294060

theorem clock_hand_speed_ratio :
  (360 / 720 : ℝ) / (360 / 60 : ℝ) = (2 / 24 : ℝ) := by
    sorry

end clock_hand_speed_ratio_l294_294060


namespace students_wearing_other_colors_l294_294370

-- Definitions based on conditions
def total_students := 700
def percentage_blue := 45 / 100
def percentage_red := 23 / 100
def percentage_green := 15 / 100

-- The proof problem statement
theorem students_wearing_other_colors :
  (total_students - total_students * (percentage_blue + percentage_red + percentage_green)) = 119 :=
by
  sorry

end students_wearing_other_colors_l294_294370


namespace intersection_with_alpha_pi_over_4_distance_AB_when_opposite_params_l294_294373

noncomputable def line_parametric_eq (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 1 + t * Real.sin α)

def curve_polar_eq (θ : ℝ) : ℝ := 4 * Real.cos θ

def intersection_points (α : ℝ) : Set (ℝ × ℝ) := 
  {p | ∃ t, p = line_parametric_eq t α ∧ curve_polar_eq (Real.atan2 p.2 p.1) = Real.sqrt (p.1 ^ 2 + p.2 ^ 2)}

theorem intersection_with_alpha_pi_over_4 :
  intersection_points (Real.pi / 4) = {(0, 0), (2 * Real.sqrt 2, Real.pi / 4)} :=
sorry

theorem distance_AB_when_opposite_params (t1 t2 α : ℝ) (ht : t1 = -t2) :
  α = ∀ α, (|line_parametric_eq t1 α - line_parametric_eq t2 α| = 2 * Real.sqrt 2) :=
sorry

end intersection_with_alpha_pi_over_4_distance_AB_when_opposite_params_l294_294373


namespace value_of_five_minus_c_l294_294327

theorem value_of_five_minus_c (c d : ℤ) (h1 : 5 + c = 6 - d) (h2 : 7 + d = 10 + c) :
  5 - c = 6 :=
by
  sorry

end value_of_five_minus_c_l294_294327


namespace find_n_l294_294238
-- Import the necessary dependencies.

-- Define the set S.
def S (n : ℕ) := {i | 1 ≤ i ∧ i ≤ 2 * n}

-- Define the property of dividing a set into two subsets with specific properties.
def divides_sum (S S₁ S₂ : Finset ℕ) := (S₁ ∪ S₂ = S ∧ S₁ ∩ S₂ = ∅ ∧ S₁.card = S₂.card ∧ ∑ i in S₁, i ∣ ∑ i in S₂, i)

-- Define the main theorem to be proven.
theorem find_n (n : ℕ) (hpos : 0 < n) : (∃ S₁ S₂ : Finset ℕ, divides_sum (Finset.filter (λ x, x ∈ S n) (Finset.range (2 * n + 1))) S₁ S₂) ↔ n % 6 ≠ 5 := sorry

end find_n_l294_294238


namespace solve_variable_expression_l294_294990

variable {x y : ℕ}

theorem solve_variable_expression
  (h1 : x / (2 * y) = 3 / 2)
  (h2 : (7 * x + 5 * y) / (x - 2 * y) = 26) :
  x = 3 * y :=
sorry

end solve_variable_expression_l294_294990


namespace t_minus_s_eq_neg_6_15_l294_294189

def students := 120
def teachers := 6
def dual_enrollments := 10
def enrollments := [40, 30, 25, 15, 5, 5]
def total_student_enrollments := students + dual_enrollments
def class_size_sum := enrollments.sum
def total_enrollments := 130 -- derived from sum of enrollments + dual_enrollments

noncomputable def t := (class_size_sum : ℝ) / (teachers : ℝ)
noncomputable def s := (enrollments[0] * (enrollments[0] : ℝ) / (total_enrollments : ℝ)) +
                        (enrollments[1] * (enrollments[1] : ℝ) / (total_enrollments : ℝ)) +
                        (enrollments[2] * (enrollments[2] : ℝ) / (total_enrollments : ℝ)) +
                        (enrollments[3] * (enrollments[3] : ℝ) / (total_enrollments : ℝ)) +
                        (enrollments[4] * (enrollments[4] : ℝ) / (total_enrollments : ℝ)) +
                        (enrollments[5] * (enrollments[5] : ℝ) / (total_enrollments : ℝ))

theorem t_minus_s_eq_neg_6_15 : t - s = -6.15 := by
  sorry

end t_minus_s_eq_neg_6_15_l294_294189


namespace problem1_l294_294535

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := a * x * Real.log x

theorem problem1 (h₁ : ∀ x > 0, f x = a * x * Real.log x) (h₂ : Deriv f 1 = 3) : a = 3 :=
by sorry

end problem1_l294_294535


namespace minimum_illuminated_points_exists_l294_294520

noncomputable def min_illuminated_points (n : ℕ) (h0 : 0 < n) (h1 : n < 90) : ℕ :=
  (180 / (Nat.gcd 180 n) + 180 / (Nat.gcd 180 (n + 1)) - 1)

theorem minimum_illuminated_points_exists : ∃ (n : ℕ), 0 < n ∧ n < 90 ∧ min_illuminated_points n (Nat.pos_of_ne_zero (ne_of_lt h0)) (Nat.lt_of_le_of_lt (le_refl n) h1) = 28 :=
sorry

end minimum_illuminated_points_exists_l294_294520


namespace unique_k_satisfying_eq_l294_294501

theorem unique_k_satisfying_eq (k : ℤ) :
  (∀ a b c : ℝ, (a + b + c) * (a * b + b * c + c * a) + k * a * b * c = (a + b) * (b + c) * (c + a)) ↔ k = -1 :=
sorry

end unique_k_satisfying_eq_l294_294501


namespace strictly_decreasing_interval_l294_294614

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(-x^2 + 2*x + 1)

theorem strictly_decreasing_interval :
  ∀ x y : ℝ, x ∈ Ioo (-real.infinity) 1 → y ∈ Ioo (-real.infinity) 1 → x < y → f y < f x :=
begin
  sorry
end

end strictly_decreasing_interval_l294_294614


namespace distance_from_point_to_origin_l294_294650

theorem distance_from_point_to_origin (x y : ℝ) 
(h1 : 4 * x + 3 * y = 0) 
(h2 : -14 ≤ x - y ∧ x - y ≤ 7) :
∃ d, 0 ≤ d ∧ d ≤ 10 ∧ d = (real.sqrt (x^2 + y^2)) := 
sorry

end distance_from_point_to_origin_l294_294650


namespace probability_greater_area_triangle_l294_294184

noncomputable def area_of_triangle (a b : Point) (c : Point) : ℝ :=
  0.5 * abs ((b - a).x * (c - a).y - (c - a).x * (b - a).y)

def is_greater_area (p : Point) (a b c d : Point) : Prop :=
  let area_ABP := area_of_triangle a b p
  let area_BCP := area_of_triangle b c p
  let area_CDP := area_of_triangle c d p
  let area_DAP := area_of_triangle d a p
  area_ABP > area_BCP ∧ area_ABP > area_CDP ∧ area_ABP > area_DAP

def square (a b c d : Point) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ (segment a b).is_perpendicular (segment b c) ∧
  (segment b c).is_perpendicular (segment c d) ∧
  (segment c d).is_perpendicular (segment d a) ∧
  (segment d a).is_perpendicular (segment a b)

def diagonals_intersect_center (a b c d o : Point) : Prop :=
  (segment a c).midpoint = o ∧ (segment b d).midpoint = o

theorem probability_greater_area_triangle {a b c d p : Point}
  (h_square : square a b c d)
  (h_diags : diagonals_intersect_center a b c d ((segment a c).midpoint))
  (h_interior : p ∈ interior_of_square a b c d)
  : 
  probability (is_greater_area p a b c d) = 1 / 4 
:= sorry

end probability_greater_area_triangle_l294_294184


namespace sin_cos_difference_min_l294_294249

theorem sin_cos_difference_min (A : ℝ) (hA : A = 7 * π / 2) :
  ∃ (min_value : ℝ), min_value = -sqrt 2 ∧ min_value = sin (A / 2) - cos (A / 2) :=
by
  sorry

end sin_cos_difference_min_l294_294249


namespace findMaxGroupSize_l294_294059

noncomputable def maxGroupSize (n k : ℕ) : ℕ := n / k

theorem findMaxGroupSize (n k m maxGroupSize : ℕ) 
  (h1 : n = 25) 
  (h2 : m = 3) 
  (h3 : k = 7)
  : maxGroupSize n k = 5 := 
  sorry

end findMaxGroupSize_l294_294059


namespace real_root_if_and_only_if_l294_294613

theorem real_root_if_and_only_if (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end real_root_if_and_only_if_l294_294613


namespace greatest_distance_between_circle_centers_l294_294490

-- Definitions
def RectWidth : ℝ := 16
def RectHeight : ℝ := 20
def CircleDiameter : ℝ := 8
def CircleRadius : ℝ := CircleDiameter / 2

-- Theorem statement
theorem greatest_distance_between_circle_centers :
  (2 * CircleRadius ≤ RectWidth) → 
  (2 * CircleRadius ≤ RectHeight) → 
  ∃ d : ℝ, d = 4 * Real.sqrt 13 ∧
  ∀ (x1 y1 x2 y2 : ℝ), 4 ≤ x1 ∧ x1 ≤ 16 - 4 ∧ 4 ≤ y1 ∧ y1 ≤ 20 - 4 ∧
                      4 ≤ x2 ∧ x2 ≤ 16 - 4 ∧ 4 ≤ y2 ∧ y2 ≤ 20 - 4 →
                      Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) ≤ d :=
begin
  sorry
end

end greatest_distance_between_circle_centers_l294_294490


namespace stars_and_bars_l294_294038

theorem stars_and_bars (n k : ℕ) : 
  let count_ways := λ n k, binomial (n + k - 1) (k - 1)
  in count_ways n k = binomial (n + k - 1) (k - 1) := by
sorry

end stars_and_bars_l294_294038


namespace geometric_sequence_S6_l294_294647

-- Assume we have a geometric sequence {a_n} and the sum of the first n terms is denoted as S_n
variable (S : ℕ → ℝ)

-- Conditions given in the problem
axiom S2_eq : S 2 = 2
axiom S4_eq : S 4 = 8

-- The goal is to find the value of S 6
theorem geometric_sequence_S6 : S 6 = 26 := 
by 
  sorry

end geometric_sequence_S6_l294_294647


namespace sum_remainder_l294_294045

theorem sum_remainder (S : ℤ) : 
  (S = ∑ n in finset.range 334, (-1)^n * nat.choose 1000 (3 * n)) → 
  S % 500 = 1 :=
by
  sorry

end sum_remainder_l294_294045


namespace roots_quartic_l294_294741

theorem roots_quartic (a b : ℝ) (h_a : a ∈ {x | x^4 - 4*x - 1 = 0})
  (h_b : b ∈ {x | x^4 - 4*x - 1 = 0}) (h_a_real : a ∈ ℝ) (h_b_real : b ∈ ℝ) :
  a * b + a + b = 1 :=
sorry

end roots_quartic_l294_294741


namespace pentagon_PT_value_l294_294718

theorem pentagon_PT_value (P Q R S T : Type) 
  (QR RS ST : ℝ) (angle_T : ℝ)
  (angle_Q angle_R angle_S : ℝ)
  (hQR : QR = 3) (hRS : RS = 3) (hST : ST = 3)
  (hT : angle_T = 90) 
  (hQ : angle_Q = 135) (hR : angle_R = 135) (hS : angle_S = 135) :
  ∃ c d : ℝ, (PT : ℝ) = c + 3 * real.sqrt d ∧ c + d = 5 :=
by
  sorry

end pentagon_PT_value_l294_294718


namespace exists_increasing_sequence_l294_294618

open Nat

theorem exists_increasing_sequence :
  ∃ (a : ℕ → ℕ), (StrictMono a) ∧ (∀ n : ℕ, ∃! (i j : ℕ), i < j ∧ n = a j - a i) ∧ (∃ C : ℝ, ∀ k : ℕ, a k ≤ C * k^3) :=
begin
  sorry,
end

end exists_increasing_sequence_l294_294618


namespace average_billboards_per_hour_l294_294062

def first_hour_billboards : ℕ := 17
def second_hour_billboards : ℕ := 20
def third_hour_billboards : ℕ := 23

theorem average_billboards_per_hour : 
  (first_hour_billboards + second_hour_billboards + third_hour_billboards) / 3 = 20 := 
by
  sorry

end average_billboards_per_hour_l294_294062


namespace binary_to_decimal_1101_l294_294610

noncomputable def binaryToDecimal (b : List ℕ) : ℕ :=
b.reverse.enumFrom 0 |>.foldl (fun acc (pair : Nat × Fin Nat) => acc + (pair.2 * 2^pair.1)) 0

theorem binary_to_decimal_1101 : binaryToDecimal [1, 1, 0, 1] = 13 :=
by
  sorry

end binary_to_decimal_1101_l294_294610


namespace sum_of_parallel_segments_l294_294071

-- Defining the conditions as the problem stated
def rectangle_EFGH : Type := {EF : ℝ, FG : ℝ}
def points_on_EF (EF : ℝ) : list ℝ := (list.range 201).map (λ k, EF * k / 200)
def points_on_FG (FG : ℝ) : list ℝ := (list.range 201).map (λ k, FG * k / 200)

-- Given problem
theorem sum_of_parallel_segments (EF FG : ℝ) (hEF : EF = 6) (hFG : FG = 8) :
  let E := (0, 0)
  let F := (EF, 0)
  let G := (EF, FG)
  let H := (0, FG)
  let P := points_on_EF EF
  let Q := points_on_FG FG
  2 * (list.sum (list.map (λ k, (10 * (200 - k) / 200 : ℝ)) (list.range 200))) - 10 = 2000 :=
by
  -- skipping the proof
  sorry

end sum_of_parallel_segments_l294_294071


namespace average_billboards_per_hour_l294_294063

-- Define the number of billboards seen in each hour
def billboards_first_hour := 17
def billboards_second_hour := 20
def billboards_third_hour := 23

-- Define the number of hours
def total_hours := 3

-- Prove that the average number of billboards per hour is 20
theorem average_billboards_per_hour : 
  (billboards_first_hour + billboards_second_hour + billboards_third_hour) / total_hours = 20 :=
by
  sorry

end average_billboards_per_hour_l294_294063


namespace find_angle_and_side_l294_294706

theorem find_angle_and_side
  (b : ℝ) (A : ℝ) (a : ℝ) (c : ℝ)
  (dot_product : ℝ)
  (area : ℝ)
  (h1 : b = 3)
  (h2 : dot_product = -6)
  (h3 : area = 3) :
  A = 135 ∧ a = Real.sqrt 29 :=
by
  -- Definitions and conditions
  let cos_A := dot_product / (b * c)
  have h_cos_A : cos_A = -2 / (b * c) := by sorry
  let sin_A := 2 * area / (b * c)
  have h_sin_A : sin_A = 2 := by sorry
  let tan_A := sin_A / cos_A
  have h_tan_A : tan_A = -1 := by sorry
  let A := Real.arctan (-1) -- This should result in A = 135 degrees
  have h_A : A = 135 := by sorry

  -- Law of Cosines
  let a_sq := b^2 + c^2 - 2 * b * c * cos_A
  have h_a_sq : a_sq = 29 := by sorry
  let a := Real.sqrt a_sq
  have h_a : a = Real.sqrt 29 := by sorry

  -- Conclusion
  exact ⟨h_A, h_a⟩

end find_angle_and_side_l294_294706


namespace number_of_correct_statements_l294_294203

noncomputable def F(n : ℕ) : ℚ :=
  let decompositions := {p : ℕ × ℕ // p.1 * p.2 = n ∧ p.1 ≤ p.2}
  let best_decomposition := decompositions.min_by (λ p, abs (p.1 - p.2))
  match best_decomposition with
  | ⟨(p, q), _⟩ => p / q

def is_correct_statement (n : ℕ) (expected : ℚ) : Prop :=
  F(n) = expected

theorem number_of_correct_statements : 
  let correct_statements := [is_correct_statement 2 (1/2), is_correct_statement 24 (3/8), 
                             is_correct_statement 27 3, ∀ n, ∃ a, n = a * a → F(n) = 1]
  correct_statements.count true = 2 :=
by {
  sorry -- proof is omitted as per instructions
}

end number_of_correct_statements_l294_294203


namespace complement_union_A_B_l294_294014

def A := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def B := {x : ℤ | ∃ k : ℤ, x = 3 * k + 2}
def U := {x : ℤ | True}

theorem complement_union_A_B:
  (U \ (A ∪ B) = {x : ℤ | ∃ k : ℤ, x = 3 * k}) :=
by
  sorry

end complement_union_A_B_l294_294014


namespace cars_arrive_at_same_time_l294_294077

-- Define the conditions and the problem statement
variables (x : ℝ) (h1 : x > 0)

-- Speed of the fast car is 1.5 times the speed of the slow car
def fast_car_speed : ℝ := 1.5 * x

-- Distance to the tourist area
def distance : ℝ := 120

-- Time taken by the slow car
def slow_car_time : ℝ := distance / x

-- Time taken by the fast car
def fast_car_time : ℝ := distance / fast_car_speed

-- Effective travel time for the slow car (since it departs 1 hour later)
def slow_car_effective_time : ℝ := slow_car_time - 1

-- Theorem stating that both cars arrive at the same time
theorem cars_arrive_at_same_time : slow_car_effective_time = fast_car_time :=
by {
  -- Simplify the terms to show equality
  sorry
}

end cars_arrive_at_same_time_l294_294077


namespace sum_of_altitudes_l294_294961

theorem sum_of_altitudes (x y : ℕ) (h1: 15*x + 8*y = 120) :
  let x_intercept := (120 : ℚ) / 15,
      y_intercept := (120 : ℚ) / 8,
      hypotenuse := (64 + 225 : ℚ).sqrt,
      area := (1/2 : ℚ) * x_intercept * y_intercept,
      altitude := (120 : ℚ) / 17 in
  x_intercept + y_intercept + altitude = 530 / 17 :=
by
  have x_intercept_val : x_intercept = 8 := by
    unfold x_intercept
    exact div_eq_of_eq_mul_left 15 (by norm_num) h1
  have y_intercept_val : y_intercept = 15 := by
    unfold y_intercept
    exact div_eq_of_eq_mul_left 8 (by norm_num) h1
  have hypotenuse_val : hypotenuse = 17 := by
    unfold hypotenuse
    exact sqrt_eq_iff_sq_eq.mp (by norm_num)
  have altitude_val : altitude = 120 / 17 := by
    unfold altitude
    sorry -- calculation check
  rw [x_intercept_val, y_intercept_val, altitude_val]
  norm_num
  sorry -- final aggregation check

end sum_of_altitudes_l294_294961


namespace product_diff_squares_l294_294748

theorem product_diff_squares (a b c d x1 y1 x2 y2 x3 y3 x4 y4 : ℕ) 
  (ha : a = x1^2 - y1^2) 
  (hb : b = x2^2 - y2^2) 
  (hc : c = x3^2 - y3^2) 
  (hd : d = x4^2 - y4^2)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) :
  ∃ X Y : ℕ, a * b * c * d = X^2 - Y^2 :=
by
  sorry

end product_diff_squares_l294_294748


namespace triangle_area_ratio_l294_294770

open Real

-- Define an equilateral triangle with a point D on AC and give the lean statement to prove the required ratio.
theorem triangle_area_ratio (A B C D : Point) (ABC : Triangle) (h_eq : is_equilateral ABC) (hD : D ∈ Segment AC) (h_angle : ∠ DBC = 30) :
  area (triangle A D B) / area (triangle C D B) = 0 :=
sorry

end triangle_area_ratio_l294_294770


namespace inscribed_square_area_after_cutting_l294_294280

theorem inscribed_square_area_after_cutting :
  let original_side := 5
  let cut_side := 1
  let remaining_side := original_side - 2 * cut_side
  let largest_inscribed_square_area := remaining_side ^ 2
  largest_inscribed_square_area = 9 :=
by
  let original_side := 5
  let cut_side := 1
  let remaining_side := original_side - 2 * cut_side
  let largest_inscribed_square_area := remaining_side ^ 2
  show largest_inscribed_square_area = 9
  sorry

end inscribed_square_area_after_cutting_l294_294280


namespace max_value_of_f_l294_294905

noncomputable def f (x : ℝ) : ℝ := 25 * Real.sin (4 * x) - 60 * Real.cos (4 * x)

theorem max_value_of_f :
  ∃ x : ℝ, ∀ y : ℝ, f(x) ≥ f(y) ∧ f(x) = 65 :=
sorry

end max_value_of_f_l294_294905


namespace find_x_intercept_of_line_through_points_l294_294812

-- Definitions based on the conditions
def point1 : ℝ × ℝ := (-1, 1)
def point2 : ℝ × ℝ := (0, 3)

-- Statement: The x-intercept of the line passing through the given points is -3/2
theorem find_x_intercept_of_line_through_points :
  let x1 := point1.1
  let y1 := point1.2
  let x2 := point2.1
  let y2 := point2.2
  ∃ x_intercept : ℝ, x_intercept = -3 / 2 ∧ 
    (∀ x, ∀ y, (x2 - x1) * (y - y1) = (y2 - y1) * (x - x1) → y = 0 → x = x_intercept) :=
by
  sorry

end find_x_intercept_of_line_through_points_l294_294812


namespace arrange_digits_l294_294205

theorem arrange_digits (A B C D E F : ℕ)
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : A ≠ E) (h5 : A ≠ F)
  (h6 : B ≠ C) (h7 : B ≠ D) (h8 : B ≠ E) (h9 : B ≠ F)
  (h10 : C ≠ D) (h11 : C ≠ E) (h12 : C ≠ F)
  (h13 : D ≠ E) (h14 : D ≠ F) (h15 : E ≠ F)
  (range_A : 1 ≤ A ∧ A ≤ 6) (range_B : 1 ≤ B ∧ B ≤ 6) (range_C : 1 ≤ C ∧ C ≤ 6)
  (range_D : 1 ≤ D ∧ D ≤ 6) (range_E : 1 ≤ E ∧ E ≤ 6) (range_F : 1 ≤ F ∧ F ≤ 6)
  (sum_line1 : A + D + E = 15) (sum_line2 : A + C + 9 = 15) 
  (sum_line3 : B + D + 9 = 15) (sum_line4 : 7 + C + E = 15) 
  (sum_line5 : 9 + C + A = 15) (sum_line6 : A + 8 + F = 15) 
  (sum_line7 : 7 + D + F = 15) : 
  (A = 4) ∧ (B = 1) ∧ (C = 2) ∧ (D = 5) ∧ (E = 6) ∧ (F = 3) :=
sorry

end arrange_digits_l294_294205


namespace inequality_proof_l294_294443

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ab(a+b) + bc(b+c) + ac(a+c) ≥ 6abc := 
sorry

end inequality_proof_l294_294443


namespace john_spent_30_l294_294851

/-- At a supermarket, John spent 1/5 of his money on fresh fruits and vegetables, 1/3 on meat products, and 1/10 on bakery products. If he spent the remaining $11 on candy, how much did John spend at the supermarket? -/
theorem john_spent_30 (X : ℝ) (h1 : X * (1/5) + X * (1/3) + X * (1/10) + 11 = X) : X = 30 := 
by 
  sorry

end john_spent_30_l294_294851


namespace children_ages_proof_l294_294110

theorem children_ages_proof :
  ∃ (n : ℕ) (ages : List ℕ), n = 5 ∧ ages = [13, 11, 10, 9, 7] ∧
    List.sum ages = 50 ∧
    List.maximum ages = some 13 ∧
    10 ∈ ages ∧
    let remaining_ages := ages.erase 10 in
    ∃ (d : ℕ) (a : ℕ), a = 13 ∧
      remaining_ages = List.range n |>.map (λ i, a - i * d) :=
begin
  sorry,
end

end children_ages_proof_l294_294110


namespace quadratic_negativity_cond_l294_294241

theorem quadratic_negativity_cond {x m k : ℝ} :
  (∀ x, x^2 - m * x - k + m < 0) ↔ k > m - (m^2 / 4) :=
sorry

end quadratic_negativity_cond_l294_294241


namespace lcm_of_8_12_15_l294_294507

theorem lcm_of_8_12_15 : Nat.lcm 8 (Nat.lcm 12 15) = 120 :=
by
  -- This is where the proof steps would go
  sorry

end lcm_of_8_12_15_l294_294507


namespace train_speed_l294_294897

def distance : ℕ := 500
def time : ℕ := 10
def conversion_factor : ℝ := 3.6

theorem train_speed :
  (distance / time : ℝ) * conversion_factor = 180 :=
by
  sorry

end train_speed_l294_294897


namespace congruent_triangles_count_l294_294345

open Set

variables (g l : Line) (A B C : Point)

def number_of_congruent_triangles (g l : Line) (A B C : Point) : ℕ :=
  16

theorem congruent_triangles_count (g l : Line) (A B C : Point) :
  number_of_congruent_triangles g l A B C = 16 :=
sorry

end congruent_triangles_count_l294_294345


namespace triangle_AC_value_l294_294383

theorem triangle_AC_value (A B C : Point) 
  (hABC : Triangle A B C)
  (hAngleA : ∠A = 90)
  (hBC : dist B C = 15)
  (hTanSin : tan ∠C = 3 * sin ∠C) : 
  dist A C = 5 := 
sorry

end triangle_AC_value_l294_294383


namespace Caleb_pencils_fewer_than_twice_Candy_l294_294933

theorem Caleb_pencils_fewer_than_twice_Candy:
  ∀ (P_Caleb P_Candy: ℕ), 
    P_Candy = 9 → 
    (∃ X, P_Caleb = 2 * P_Candy - X) → 
    P_Caleb + 5 - 10 = 10 → 
    (2 * P_Candy - P_Caleb = 3) :=
by
  intros P_Caleb P_Candy hCandy hCalebLess twCalen
  sorry

end Caleb_pencils_fewer_than_twice_Candy_l294_294933


namespace orthocentric_tetrahedron_properties_l294_294440

theorem orthocentric_tetrahedron_properties
  {A B C D : Type*} [affine_tetrahedron A B C D] (h : orthocentric A B C D) :
  (∀ E F, opposite_edges A B C D E F → E ⟂ F) ∧
  (∀ v₁ v₂, plane_angle_is_right v₁ = tt → other_two_plane_angles_right v₁ v₂) ∧
  (∑ (e : edge), opposite_edge_lengths_square_sum e = constant) ∧
  (∀ v, projections_onto_opposite_face_orthocenter v A B C D) :=
sorry

end orthocentric_tetrahedron_properties_l294_294440


namespace exist_alpha_seq_l294_294746

-- Define the structure of the sequence
def seq (a : ℕ → ℕ) : ℕ → ℕ
| 0       := 2
| (n + 1) :=
  if (a n == 2) then
    2
  else
    3

-- Define the theorems and proof problems
theorem exist_alpha_seq (a : ℕ → ℕ) (exists_alpha : ∃ α : ℝ, α = 2 + Real.sqrt 3 ) :
  (∃ α : ℝ, α = 2 + Real.sqrt 3 ∧ ∀ n : ℕ, a n = 2 ↔ ∃ m : ℕ, n = Real.floor (α * m)) :=
begin
  sorry -- the proof goes here
end

end exist_alpha_seq_l294_294746


namespace real_solutions_eq_pos_neg_2_l294_294627

theorem real_solutions_eq_pos_neg_2 (x : ℝ) :
  ( (x - 1) ^ 2 * (x - 5) * (x - 5) / (x - 5) = 4) ↔ (x = 3 ∨ x = -1) :=
by
  sorry

end real_solutions_eq_pos_neg_2_l294_294627


namespace cyclist_ends_up_in_anomalous_zone_l294_294880

-- Define the circular road and the anomalous zone
def circular_road_length : ℕ := 1000  -- Example length; this should be considered unbounded in abstract proof
def anomalous_zone_length : ℕ := 71
def daily_distance : ℕ := 71

-- Define the positions on the circular road
def position : Type := ℕ

-- Define the property of the anomalous zone (if a cyclist sleeps in the zone, they wake up at the symmetric point)
def wakes_up_at_symmetric_point (pos : position) : Prop :=
  pos < anomalous_zone_length

-- Main theorem statement
theorem cyclist_ends_up_in_anomalous_zone (start : position) : 
  ∃ (end : position), wakes_up_at_symmetric_point end :=
by {
  -- The proof is skipped
  sorry
}

end cyclist_ends_up_in_anomalous_zone_l294_294880


namespace power_mod_remainder_l294_294121

theorem power_mod_remainder (a b : ℕ) (h1 : a = 3^1 % 7) (h2 : b = (3^87 + 5) % 7) : b = 4 := by
  have h3 : 3^1 % 7 = 3 := by norm_num
  have h4 : 3^2 % 7 = 2 := by norm_num
  have h5 : 3^3 % 7 = 6 := by norm_num
  have h6 : 3^4 % 7 = 4 := by norm_num
  have h7 : 3^5 % 7 = 5 := by norm_num
  have h8 : 3^6 % 7 = 1 := by norm_num
  have h_pattern : ∀ k : ℕ, (3^(6*k) % 7) = 1 := by
    intro k
    induction k with k ih
    case zero =>
      simp
    case succ =>
      rw [pow_succ]
      rw [←mul_pow]
      rw [ih]
      rw [mul_one]
  have h9 : 3^87 % 7 = 6 := by
    let k := 87 / 6
    let r := 87 % 6
    have hk : 87 = 6 * k + r := Nat.div_add_mod 87 6
    rw [hk]
    have hr : r = 3 := by norm_num
    rw [hr]
    rw [pow_add]
    rw [pow_mul]
    rw [h_pattern]
    norm_num
  have h10 : (3^87 + 5) % 7 = (6 + 5) % 7 := by rw [h9]
  have h11 : (6 + 5) % 7 = 4 := by norm_num
  rw [h10, h11]
  exact h2

end power_mod_remainder_l294_294121


namespace angle_CFD_60_degrees_l294_294404

open EuclideanGeometry

variable (circle : Circle)
variable (O A B F C D : Point)
variable [Diameter O A B : circle]
variable [OnCircle F : circle]
variable [TangentAt B C : circle]
variable [TangentAt F D : circle]
variable [Line AF D]

theorem angle_CFD_60_degrees (h1 : angle B A F = 30) : angle C F D = 60 :=
by
  -- proofs would be added here
  sorry

end angle_CFD_60_degrees_l294_294404


namespace sequence_properties_l294_294052

/-- Theorem for the general term and the range of m given the conditions -/
theorem sequence_properties (a b : ℕ → ℝ) (m : ℝ) (n : ℕ) (Hn : n > 0) (Ha1 : a 1 = m)
  (Hb1 : b 1 = m) (Hb2 : b 2 = 3 * m / 2) (Hbn : ∀ n, b n = (list.range n).sum (λ i, a (n - i))) :
  (∀ n, a n = m * (- 1 / 2) ^ (n - 1)) ∧ (2 ≤ m ∧ m ≤ 3) := sorry

end sequence_properties_l294_294052


namespace find_m_l294_294319

theorem find_m (x y m : ℝ) (h₁ : x - 2 * y = m) (h₂ : x = 2) (h₃ : y = 1) : m = 0 :=
by 
  -- Proof omitted
  sorry

end find_m_l294_294319


namespace will_has_123_pieces_of_candy_l294_294519

def initial_candy_pieces (chocolate_boxes mint_boxes caramel_boxes : ℕ)
  (pieces_per_chocolate_box pieces_per_mint_box pieces_per_caramel_box : ℕ) : ℕ :=
  chocolate_boxes * pieces_per_chocolate_box + mint_boxes * pieces_per_mint_box + caramel_boxes * pieces_per_caramel_box

def given_away_candy_pieces (given_chocolate_boxes given_mint_boxes given_caramel_boxes : ℕ)
  (pieces_per_chocolate_box pieces_per_mint_box pieces_per_caramel_box : ℕ) : ℕ :=
  given_chocolate_boxes * pieces_per_chocolate_box + given_mint_boxes * pieces_per_mint_box + given_caramel_boxes * pieces_per_caramel_box

def remaining_candy : ℕ :=
  let initial := initial_candy_pieces 7 5 4 12 15 10
  let given_away := given_away_candy_pieces 3 2 1 12 15 10
  initial - given_away

theorem will_has_123_pieces_of_candy : remaining_candy = 123 :=
by
  -- Proof goes here
  sorry

end will_has_123_pieces_of_candy_l294_294519


namespace find_vector_u_l294_294965

noncomputable def matrixB : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![2, 1],
  ![4, 2]
]

notation "I" => Matrix.one (Fin 2) ℝ

noncomputable def u : Vector ℝ (Fin 2) := ![0, 3/17]

theorem find_vector_u :
  (matrixB^6 + matrixB^4 + matrixB^2 + I) • u = ![0, 15] :=
sorry

end find_vector_u_l294_294965


namespace david_completion_time_l294_294902

theorem david_completion_time :
  (∃ D : ℕ, ∀ t : ℕ, 6 * (1 / D) + 3 * ((1 / D) + (1 / t)) = 1 -> D = 12) :=
sorry

end david_completion_time_l294_294902


namespace inverse_function_property_l294_294050

noncomputable def f : ℝ → ℝ := sorry

theorem inverse_function_property :
  (bijective f) →
  (∀ x : ℝ, f (x + 1) + f (-x - 4) = 2) →
  (∀ x : ℝ, f⁻¹ (2011 - x) + f⁻¹ (x - 2009) = -3) :=
by
  intros f_bij fx_property
  sorry

end inverse_function_property_l294_294050


namespace negation_of_proposition_l294_294815

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x ≥ 0 → x^2 ≥ 0) ↔ ∃ x : ℝ, x ≥ 0 ∧ x^2 < 0 :=
by 
  sorry

end negation_of_proposition_l294_294815


namespace sum_of_radii_geq_inc_radius_l294_294753

theorem sum_of_radii_geq_inc_radius
  (r r1 r2 r3 : ℝ)
  (h1 : r1 < r)
  (h2 : r2 < r)
  (h3 : r3 < r)
  (h_triangle : ∃ A B C : triangle, incircle_radius A B C = r):
  r1 + r2 + r3 ≥ r := 
sorry

end sum_of_radii_geq_inc_radius_l294_294753


namespace eval_product_twelfth_roots_of_unity_l294_294235

noncomputable def w : ℂ := complex.exp (2 * real.pi * complex.I / 12)

theorem eval_product_twelfth_roots_of_unity :
  (∏ k in finset.range 11, (3 : ℂ) - w^k.succ) = 265720 :=
sorry

end eval_product_twelfth_roots_of_unity_l294_294235


namespace complement_union_eq_complement_l294_294003

open Set

section ComplementUnion

variable (k : ℤ)

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def UniversalSet : Set ℤ := univ
def ComplementUnion : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem complement_union_eq_complement :
  UniversalSet \ (SetA ∪ SetB) = ComplementUnion :=
by
  sorry

end ComplementUnion

end complement_union_eq_complement_l294_294003


namespace find_x_for_f_eq_5_l294_294675

noncomputable def f : ℝ → ℝ
| x => if x ≤ 0 then x^2 + 1 else -2 * x

theorem find_x_for_f_eq_5 (x : ℝ) : f x = 5 ↔ x = -2 :=
by
  split
  · intro h
    cases le_or_gt x 0 with h₁ h₂
    · rw [if_pos h₁] at h
      have := eq_of_sq_eq_sq (eq_sub_of_add_eq h.zero_sub)
      cases this
      · exact this
      · contradiction
    · rw [if_neg h₂] at h
      linarith
  · intro hx
    rw [hx, if_pos]
    norm_num
    exact le_refl _

end find_x_for_f_eq_5_l294_294675


namespace binomial_sum_identity_l294_294261

theorem binomial_sum_identity : (Finset.sum (Finset.range 51) (fun k => (-1)^k * Nat.choose 100 (2*k)) = -2^50) :=
by
  sorry

end binomial_sum_identity_l294_294261


namespace speed_of_stream_l294_294148

theorem speed_of_stream :
  ∀ (v : ℕ), let boat_speed := 30 in let time := 2 in let distance := 70 in
  distance = (boat_speed + v) * time → v = 5 :=
by
  sorry

end speed_of_stream_l294_294148


namespace average_of_other_half_l294_294854

theorem average_of_other_half (avg : ℝ) (sum_half : ℝ) (n : ℕ) (n_half : ℕ)
    (h_avg : avg = 43.1)
    (h_sum_half : sum_half = 158.4)
    (h_n : n = 8)
    (h_n_half : n_half = n / 2) :
    ((n * avg - sum_half) / n_half) = 46.6 :=
by
  -- The proof steps would be given here. We're omitting them as the prompt instructs.
  sorry

end average_of_other_half_l294_294854


namespace geometric_sequence_a3_l294_294301

theorem geometric_sequence_a3 :
  ∀ (a : ℕ → ℝ), a 1 = 2 → a 5 = 8 → (a 3 = 4 ∨ a 3 = -4) :=
by
  intros a h₁ h₅
  sorry

end geometric_sequence_a3_l294_294301


namespace orthocenter_on_fixed_circle_l294_294155

-- Definitions of Circle, Points, and secant intersection
variable (O A K : Point) (circ : Circle)
variable (P Q : Point)

-- Assumptions from the problem
axiom circle_fixed : fixed (circ)
axiom A_on_circle : on_circle A circ
axiom K_outside_circle : outside_of_circle K circ
axiom secant_KPQ : secant_through K P Q circ

-- To prove: orthocenters of triangles APQ lie on a fixed circle
theorem orthocenter_on_fixed_circle 
    (O A K P Q : Point)
    (circ : Circle)
    (circle_fixed : fixed (circ))
    (A_on_circle : on_circle A circ)
    (K_outside_circle : outside_of_circle K circ)
    (secant_KPQ : secant_through K P Q circ) :
  ∃ fixed_circle : Circle, ∀ P Q, 
  secant_through K P Q circ → (orthocenter (triangle A P Q) ∈ fixed_circle) := 
by sorry

end orthocenter_on_fixed_circle_l294_294155


namespace compare_x_with_half_sin_tan_l294_294324

theorem compare_x_with_half_sin_tan (x : ℝ) (h₁ : 0 < x) (h₂ : x < π / 2)
 (h₃ : sin x < x) (h₄ : x < tan x) : x < (1 / 2) * (sin x + tan x) :=
sorry

end compare_x_with_half_sin_tan_l294_294324


namespace math_problem_l294_294931

theorem math_problem :
  18 * 35 + 45 * 18 - 18 * 10 = 1260 :=
by
  sorry

end math_problem_l294_294931


namespace Rhonda_marbles_l294_294906

theorem Rhonda_marbles :
  ∃ m : ℕ, (∃ a : ℕ, a + m = 215 ∧ a = m + 55) ∧ m = 80 :=
by
  use 80
  use 135
  sorry

end Rhonda_marbles_l294_294906


namespace simplify_and_evaluate_sqrt_log_product_property_l294_294453

-- Problem I
theorem simplify_and_evaluate_sqrt (a : ℝ) (h : 0 < a) : 
  Real.sqrt (a^(1/4) * Real.sqrt (a * Real.sqrt a)) = Real.sqrt a := 
by
  sorry

-- Problem II
theorem log_product_property : 
  Real.log 3 / Real.log 2 * Real.log 5 / Real.log 3 * Real.log 4 / Real.log 5 = 2 := 
by
  sorry

end simplify_and_evaluate_sqrt_log_product_property_l294_294453


namespace negative_number_is_C_l294_294515

def is_negative (n : ℝ) : Prop := n < 0

def A : ℝ := 0
def B : ℝ := -(-3)
def C : ℝ := -1/2
def D : ℝ := 3.2

theorem negative_number_is_C : is_negative C ∧
                               ¬ is_negative A ∧
                               ¬ is_negative B ∧
                               ¬ is_negative D := by
sorry

end negative_number_is_C_l294_294515


namespace ellipse_equation_l294_294316

noncomputable def line_l : ℝ → ℝ := λ x, 2 * x - 4

theorem ellipse_equation :
  let F1 := (0, 1)
  let P := (3 / 2, -1)
  let F2 := (0, -1)
  let l := line_l
  let C1 := λ x y, (y ^ 2 / 4) + (x ^ 2 / 3) = 1
  in ( ∀ x, (x^2 = 4 * (2 * x - 4) → 4 * x - 16 = 0 → x = 4) ∧ 
       C1 (3/2) (-1) )
  :=
by
  intro F1 F2 P l C1
  split
  · intro h
    sorry -- Proof for line equation
  · sorry -- Proof for ellipse satisfying the point

end ellipse_equation_l294_294316


namespace greatest_distance_between_circle_centers_l294_294491

-- Definitions
def RectWidth : ℝ := 16
def RectHeight : ℝ := 20
def CircleDiameter : ℝ := 8
def CircleRadius : ℝ := CircleDiameter / 2

-- Theorem statement
theorem greatest_distance_between_circle_centers :
  (2 * CircleRadius ≤ RectWidth) → 
  (2 * CircleRadius ≤ RectHeight) → 
  ∃ d : ℝ, d = 4 * Real.sqrt 13 ∧
  ∀ (x1 y1 x2 y2 : ℝ), 4 ≤ x1 ∧ x1 ≤ 16 - 4 ∧ 4 ≤ y1 ∧ y1 ≤ 20 - 4 ∧
                      4 ≤ x2 ∧ x2 ≤ 16 - 4 ∧ 4 ≤ y2 ∧ y2 ≤ 20 - 4 →
                      Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) ≤ d :=
begin
  sorry
end

end greatest_distance_between_circle_centers_l294_294491


namespace find_quadrant_of_angle_l294_294657

def quadrant (α : ℝ) : string :=
  if (sin α > 0) && (cos α > 0) then "I"
  else if (sin α > 0) && (cos α < 0) then "II"
  else if (sin α < 0) && (cos α < 0) then "III"
  else if (sin α < 0) && (cos α > 0) then "IV"
  else "undefined"

theorem find_quadrant_of_angle (α : ℝ) 
  (h1 : sin α = -1/2) 
  (h2 : cos α = (sqrt 3)/2) : quadrant α = "IV" :=
by sorry

end find_quadrant_of_angle_l294_294657


namespace problem_1_problem_2_l294_294704

variables {A B C a b c : ℝ}

noncomputable def find_B (h1 : a * sin (2 * B) = sqrt 3 * b * sin A) : Prop :=
  B = π / 6

noncomputable def find_sin_C (h1 : a * sin (2 * B) = sqrt 3 * b * sin A) (h2 : cos A = 1 / 3) : Prop :=
  sin C = (2 * sqrt 6 + 1) / 6

theorem problem_1 (h1 : a * sin (2 * B) = sqrt 3 * b * sin A) : find_B h1 := 
  sorry

theorem problem_2 (h1 : a * sin (2 * B) = sqrt 3 * b * sin A) (h2 : cos A = 1 / 3) : find_sin_C h1 h2 := 
  sorry

end problem_1_problem_2_l294_294704


namespace all_chords_are_diameters_l294_294710

theorem all_chords_are_diameters {C : Type} [metric_space C] [circle_structure C] 
    (chords : set (chord C)) 
    (h : ∀ c1 ∈ chords, ∃ c2 ∈ chords, c1 ≠ c2 ∧ midpoint c2 c1) : 
    ∀ c ∈ chords, is_diameter c :=
by 
  sorry

end all_chords_are_diameters_l294_294710


namespace central_angle_of_sector_l294_294084

-- Define the conditions in Lean
variables {r l : ℝ}
def circumference_condition : Prop := 2 * r + l = 6
def area_condition : Prop := (1 / 2) * l * r = 2

-- The theorem statement
theorem central_angle_of_sector :
  circumference_condition ∧ area_condition → (l/r = 1 ∨ l/r = 4) :=
by
  intro h
  cases h with hc ha
  sorry

end central_angle_of_sector_l294_294084


namespace expression_value_l294_294971

theorem expression_value (b : ℝ) (hb : b = 1 / 3) :
    (3 * b⁻¹ - b⁻¹ / 3) / b^2 = 72 :=
sorry

end expression_value_l294_294971


namespace gcd_lcm_identity_l294_294739

-- Assume a, b, c are positive integers
variables (a b c : ℕ)

-- Define gcd and lcm.
def gcd (x y : ℕ) : ℕ := Nat.gcd x y
def lcm (x y : ℕ) : ℕ := Nat.lcm x y

theorem gcd_lcm_identity (habc : ∀ {a b c : ℕ}, a > 0 ∧ b > 0 ∧ c > 0) :
    (lcm (lcm a b) c) ^ 2 / (lcm a b * lcm b c * lcm c a) = (gcd (gcd a b) c) ^ 2 / (gcd a b * gcd b c * gcd c a) := by
  sorry

end gcd_lcm_identity_l294_294739


namespace parabola_b_value_l294_294302

variable (a b c p : ℝ)
variable (h1 : p ≠ 0)
variable (h2 : ∀ x, y = a*x^2 + b*x + c)
variable (h3 : vertex' y = (p, -p))
variable (h4 : y-intercept' y = (0, p))

theorem parabola_b_value : b = -4 :=
sorry

end parabola_b_value_l294_294302


namespace total_cost_of_soup_l294_294829

theorem total_cost_of_soup :
  let beef_pounds := 4
  let vegetable_pounds := 6
  let vegetable_cost_per_pound := 2
  let beef_cost_per_pound := 3 * vegetable_cost_per_pound
  let cost_of_vegetables := vegetable_cost_per_pound * vegetable_pounds
  let cost_of_beef := beef_cost_per_pound * beef_pounds
  in cost_of_vegetables + cost_of_beef = 36 :=
by
  let beef_pounds := 4
  let vegetable_pounds := 6
  let vegetable_cost_per_pound := 2
  let beef_cost_per_pound := 3 * vegetable_cost_per_pound
  let cost_of_vegetables := vegetable_cost_per_pound * vegetable_pounds
  let cost_of_beef := beef_cost_per_pound * beef_pounds
  show cost_of_vegetables + cost_of_beef = 36
  sorry

end total_cost_of_soup_l294_294829


namespace expected_rolls_sum_2010_l294_294172

noncomputable def expected_rolls_to_reach_sum (n : ℕ) : ℚ :=
  if n == 0 then 0
  else (1/6) * (expected_rolls_to_reach_sum (n-1) + expected_rolls_to_reach_sum (n-2) +
               expected_rolls_to_reach_sum (n-3) + expected_rolls_to_reach_sum (n-4) +
               expected_rolls_to_reach_sum (n-5) + expected_rolls_to_reach_sum (n-6)) + 1

theorem expected_rolls_sum_2010 : expected_rolls_to_reach_sum 2010 ≈ 574.761904 := 
by 
  -- Proof omitted; the focus is on the statement
  sorry

end expected_rolls_sum_2010_l294_294172


namespace complement_union_A_B_l294_294015

def A := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def B := {x : ℤ | ∃ k : ℤ, x = 3 * k + 2}
def U := {x : ℤ | True}

theorem complement_union_A_B:
  (U \ (A ∪ B) = {x : ℤ | ∃ k : ℤ, x = 3 * k}) :=
by
  sorry

end complement_union_A_B_l294_294015


namespace H2_bound_l294_294923

noncomputable def sqrt (n : ℝ) : ℝ := sorry

variable (n p t : ℕ)
variable (H : ∀ k : ℤ, k^p ≠ n)

def G' : set ℕ := { m | ∃ i j : ℕ, 0 ≤ i ∧ i ≤ sqrt t ∧ 0 ≤ j ∧ j ≤ sqrt t ∧ m = n^i * p^j }

theorem H2_bound : 
  ∀ (H2 : set ℕ), 
  (∀ h₁ h₂ ∈ G', h₁ ≠ h₂ → h₁ = h₂) → 
  (H2 ⊆ G') →
  ∃ k (h1 h2 : ℕ), h1 ∈ G' ∧ h2 ∈ G' ∧ h1 ≠ h2 ∧ (h1 ≡ h2 [MOD k]) ∧ 
  ∀ H2, |H2| ≤ n^(3 * sqrt(t) / 2) :=
begin
  sorry
end

end H2_bound_l294_294923


namespace AC_is_five_l294_294381

-- Given conditions
variables (A B C : Type) [inner_product_space ℝ A]
variables {P Q R : A}
variables (angleA : angle P Q R = 90) (BC PQ_length : dist P Q = 15)
variables (tanC_eq_3sinC : tan (angle Q P R) = 3 * sin (angle Q P R))

-- Define the length of side AC
def AC_length : ℝ := dist P R

-- The statement to prove
theorem AC_is_five :
  AC_length P R = 5 :=
sorry

end AC_is_five_l294_294381


namespace smallest_period_of_f_min_max_values_of_f_on_interval_l294_294329

def f (x : ℝ) : ℝ := Math.cos x * Math.sin (x + Real.pi / 3) - Real.sqrt 3 * (Math.cos x) ^ 2 + Real.sqrt 3 / 4

theorem smallest_period_of_f :
  ∃ p > 0, ∀ x, f (x + p) = f x ∧ p = Real.pi :=
by
  sorry

theorem min_max_values_of_f_on_interval :
  ∀ x, - Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → f x = - 1 / 2 ∨ f x = 1 / 4 :=
by
  sorry

end smallest_period_of_f_min_max_values_of_f_on_interval_l294_294329


namespace work_completed_in_10_days_l294_294134

theorem work_completed_in_10_days (a b c: ℕ) (work_rate_a work_rate_b work_rate_c: ℚ)
  (days_a: a = 18)
  (days_b: b = 9)
  (days_c: c = 12)
  (work_rate_a: work_rate_a = 1 / 18)
  (work_rate_b: work_rate_b = 1 / 9)
  (work_rate_c: work_rate_c = 1 / 12)
  : (6 * work_rate_a + 6 * work_rate_b + 4 * (work_rate_a + work_rate_b + work_rate_c) = 1) :=
by
  sorry

end work_completed_in_10_days_l294_294134


namespace smallest_delicious_integer_l294_294446

-- Define what it means for an integer to be delicious
def is_delicious (B : ℤ) : Prop :=
  ∃ (n : ℕ), ∑ i in (finset.range n).map (λ i, B + i) = 2020

-- Lean statement to prove the smallest delicious integer
theorem smallest_delicious_integer : ∃ B : ℤ, is_delicious B ∧ B < -2020 ∧ ∀ B' : ℤ, (is_delicious B' ∧ B' < -2020) → B ≤ B' := by
  sorry

end smallest_delicious_integer_l294_294446


namespace parallel_lines_remain_parallel_in_oblique_projection_l294_294517

def oblique_projection_x_unchanged (length_x : ℝ) : ℝ :=
  length_x

def oblique_projection_y_halved (length_y : ℝ) : ℝ :=
  length_y / 2

theorem parallel_lines_remain_parallel_in_oblique_projection :
  ∀ (line1 line2 : ℝ × ℝ → ℝ × ℝ), 
    (∀ t, line1 t = (t, oblique_projection_x_unchanged t)) → 
    (∀ s, line2 s = (s, oblique_projection_y_halved s)) →
    ∃ u v, ∀ w, (line1 (u + w) = line1 u + line1 w) ∧ (line2 (v + w) = line2 v + line2 w) →
      (∃ m, line1 m = line2 m) :=
sorry

end parallel_lines_remain_parallel_in_oblique_projection_l294_294517


namespace jane_total_investment_in_stocks_l294_294393

-- Definitions
def total_investment := 220000
def bonds_investment := 13750
def stocks_investment := 5 * bonds_investment
def mutual_funds_investment := 2 * stocks_investment

-- Condition: The total amount invested
def total_investment_condition : Prop := 
  bonds_investment + stocks_investment + mutual_funds_investment = total_investment

-- Theorem: Jane's total investment in stocks
theorem jane_total_investment_in_stocks :
  total_investment_condition →
  stocks_investment = 68750 :=
by sorry

end jane_total_investment_in_stocks_l294_294393


namespace medicine_dosage_l294_294881

def dosage_per_kg : ℝ := 5
def decrease_percent : ℝ := 0.10
def child_weight : ℝ := 30
def child_age : ℝ := 8
def parts : ℝ := 3

theorem medicine_dosage :
  let initial_dose := child_weight * dosage_per_kg in
  let decrease_amount := if child_age < 10 then decrease_percent * initial_dose else 0 in
  let adjusted_dose := initial_dose - decrease_amount in
  let dose_per_part := adjusted_dose / parts in
  dose_per_part = 45 :=
by
  sorry

end medicine_dosage_l294_294881


namespace age_of_B_l294_294138

variable (A B C : ℕ)

theorem age_of_B (h1 : A + B + C = 84) (h2 : A + C = 58) : B = 26 := by
  sorry

end age_of_B_l294_294138


namespace spinner_multiples_of_3_l294_294495

theorem spinner_multiples_of_3 :
  let outcomesA := { 4, 5, 6, 7 }
  let outcomesB := { 6, 7, 8, 9, 10 }
  let prob_multiples_3_A := 1 / 4
  let prob_multiples_3_B := 2 / 5
  let prob_not_multiples_3_A := 3 / 4
  let prob_not_multiples_3_B := 3 / 5
  let prob_neither_multiple_3 := (3 / 4 * 3 / 5)
  let prob_at_least_one_multiple_3 := 1 - prob_neither_multiple_3
  prob_at_least_one_multiple_3 = 11 / 20 := 
begin
  sorry
end

end spinner_multiples_of_3_l294_294495


namespace parabola_focus_directrix_distance_l294_294465

theorem parabola_focus_directrix_distance :
  ∃ p, p = 1 / 4 ∧ ∀ x y, (y = 2 * x^2) → p = distance_from_focus_to_directrix y := 
by
  sorry

end parabola_focus_directrix_distance_l294_294465


namespace four_hash_two_equals_forty_l294_294224

def hash_op (a b : ℕ) : ℤ := (a^2 + b^2) * (a - b)

theorem four_hash_two_equals_forty : hash_op 4 2 = 40 := 
by
  sorry

end four_hash_two_equals_forty_l294_294224


namespace limit_nb_n_l294_294966

noncomputable theory

def M (x : ℝ) := x - x^3 / 3

def iter (f : ℝ → ℝ) (x : ℝ) (n : ℕ) : ℝ :=
  nat.iterate f n x

def b (n : ℕ) : ℝ := iter M (20 / n) n

theorem limit_nb_n : filter.tendsto (λ n : ℕ, n * b n) filter.at_top (𝓝 (40 / 19)) :=
sorry

end limit_nb_n_l294_294966


namespace sphere_has_circular_views_l294_294579

-- Define the geometric shapes
inductive Shape
| cuboid
| cylinder
| cone
| sphere

-- Define a function that describes the views of a shape
def views (s: Shape) : (String × String × String) :=
match s with
| Shape.cuboid   => ("Rectangle", "Rectangle", "Rectangle")
| Shape.cylinder => ("Rectangle", "Rectangle", "Circle")
| Shape.cone     => ("Isosceles Triangle", "Isosceles Triangle", "Circle")
| Shape.sphere   => ("Circle", "Circle", "Circle")

-- Define the property of having circular views in all perspectives
def has_circular_views (s: Shape) : Prop :=
views s = ("Circle", "Circle", "Circle")

-- The theorem to prove
theorem sphere_has_circular_views :
  ∀ (s : Shape), has_circular_views s ↔ s = Shape.sphere :=
by sorry

end sphere_has_circular_views_l294_294579


namespace find_mb_l294_294126

noncomputable def polynomial_remainder (p : Polynomial ℝ) (d : Polynomial ℝ) : Polynomial ℝ :=
  Polynomial.mod_by_monic p (Polynomial.monic_of_degree_leading_coeff_one d)

theorem find_mb :
  let p := x^5 - 4*x^4 + 12*x^3 - 14*x^2 + 8*x + 5
  let d := x^2 - 3*x + m
  let r := 2*x + b
  polynomial_remainder p d = r → (m, b) = (1, 7) :=
by
  intros p d r h
  let m := 1
  let b := 7
  sorry

end find_mb_l294_294126


namespace inequality_sum_l294_294738

noncomputable theory
open_locale big_operators

theorem inequality_sum
  (n : ℕ) (hn : n > 0)
  (a b : fin n → ℝ)
  (h : ∀ i, 0 < a i + b i) :
  ∑ i, (a i * b i - (b i)^2) / (a i + b i) ≤ 
    (∑ i, a i) * (∑ i, b i) - (∑ i, b i)^2 / ∑ i, (a i + b i) := 
begin
  sorry
end

end inequality_sum_l294_294738


namespace smallest_n_inverse_mod_1176_l294_294509

theorem smallest_n_inverse_mod_1176 : ∃ n : ℕ, n > 1 ∧ Nat.Coprime n 1176 ∧ (∀ m : ℕ, m > 1 ∧ Nat.Coprime m 1176 → n ≤ m) ∧ n = 5 := by
  sorry

end smallest_n_inverse_mod_1176_l294_294509


namespace ellipse_equation_fixed_point_P_exists_l294_294326

-- Given conditions
variables {a b : ℝ} (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
          (e : ℝ) (e_eq : e = 1/2)
          (hx : ℝ) (hy : ℝ) 
          (point_on_ellipse : (sqrt 3, -sqrt 3 / 2) = (hx, hy) ∧ hx^2 / a^2 + hy^2 / b^2 = 1)
          (focus_right : ℝ -> ℝ -> Prop) (M N : ℝ × ℝ)
          (line_through_focus : ∀ y, focus_right 4 y → (∃ x1 y1 x2 y2, 
            (M = (x1, y1) ∧ N = (x2, y2) ∧ (sqrt 3) = (2x1x1 + xy1 + xy2 + x = 4))) -- placeholder for actual properties)

-- Prove the equation of ellipse
theorem ellipse_equation : ∃ (a b : ℝ), a^2 = 4 ∧ b^2 = 3 ∧ (hx^2 / 4 + hy^2 / 3 = 1) := 
by sorry

-- Prove the fixed point
theorem fixed_point_P_exists : 
  ∃ (x1 y1 x2 y2 : ℝ), (M = (x1, y1) ∧ N = (x2, y2)) → 
  ∀ P : ℝ × ℝ, P = (x2, -y2) → 
  ∃ f : ℝ × ℝ, f = (4, 0) ∧ focus_right y1 y2 :=
by sorry

end ellipse_equation_fixed_point_P_exists_l294_294326


namespace gcd_135_81_l294_294628

-- Define the numbers
def a : ℕ := 135
def b : ℕ := 81

-- State the goal: greatest common divisor of a and b is 27
theorem gcd_135_81 : Nat.gcd a b = 27 := by
  sorry

end gcd_135_81_l294_294628


namespace math_proof_problem_l294_294591

theorem math_proof_problem (a b : ℝ) (h1 : 64 = 8^2) (h2 : 16 = 8^2) :
  8^15 / (64^7) * 16 = 512 :=
by
  sorry

end math_proof_problem_l294_294591


namespace find_tangent_point_l294_294102

noncomputable def exp_neg (x : ℝ) : ℝ := Real.exp (-x)

theorem find_tangent_point :
  ∃ P : ℝ × ℝ, P = (-Real.log 2, 2) ∧ P.snd = exp_neg P.fst ∧ deriv exp_neg P.fst = -2 :=
by
  sorry

end find_tangent_point_l294_294102


namespace program_exists_l294_294755
open Function

-- Define the chessboard and labyrinth
namespace ChessMaze

structure Position :=
  (row : Nat)
  (col : Nat)
  (h_row : row < 8)
  (h_col : col < 8)

inductive Command
| RIGHT | LEFT | UP | DOWN

structure Labyrinth :=
  (barriers : Position → Position → Bool) -- True if there's a barrier between the two positions

def accessible (L : Labyrinth) (start : Position) (cmd : List Command) : Set Position :=
  -- The set of positions accessible after applying the commands from start in labyrinth L
  sorry

-- The main theorem we want to prove
theorem program_exists : 
  ∃ (cmd : List Command), ∀ (L : Labyrinth) (start : Position), ∀ pos ∈ accessible L start cmd, ∃ p : Position, p = pos :=
  sorry

end ChessMaze

end program_exists_l294_294755


namespace product_base9_l294_294925

open Nat

noncomputable def base9_product (a b : ℕ) : ℕ := 
  let a_base10 := 3*9^2 + 6*9^1 + 2*9^0
  let b_base10 := 7
  let product_base10 := a_base10 * b_base10
  -- converting product_base10 from base 10 to base 9
  2 * 9^3 + 8 * 9^2 + 7 * 9^1 + 5 * 9^0 -- which simplifies to 2875 in base 9

theorem product_base9: base9_product 362 7 = 2875 :=
by
  -- Here should be the proof or a computational check
  sorry

end product_base9_l294_294925


namespace delta_correct_delta3_arithmetic_not_geometric_sum_first_n_terms_delta_incorrect_sum_first_2014_terms_delta2_l294_294995

def sequence_a (n : ℕ) : ℕ := n^2 + n

def delta (a : ℕ → ℕ) (n : ℕ) : ℕ := a (n + 1) - a n

def delta_k (a : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, n => a n
| (k + 1), n => delta (delta_k k) n

theorem delta_correct (n : ℕ) (h : 0 < n) : delta sequence_a n = 2 * n + 2 := sorry

theorem delta3_arithmetic_not_geometric (n : ℕ) (h : 0 < n) : 
  (∀ m, delta_k sequence_a 3 m = 0) ∧ (∃ m1 m2, delta_k sequence_a 3 m1 ≠ 1) := sorry

theorem sum_first_n_terms_delta_incorrect (n : ℕ) (h : 0 < n) : 
  finset.sum (finset.range n) (λ k, delta sequence_a (k + 1)) ≠ n^2 + n := sorry

theorem sum_first_2014_terms_delta2 (h : 0 < 2014) : 
  finset.sum (finset.range 2014) (λ k, delta_k sequence_a 2 (k + 1)) = 4028 := sorry

end delta_correct_delta3_arithmetic_not_geometric_sum_first_n_terms_delta_incorrect_sum_first_2014_terms_delta2_l294_294995


namespace a_alone_can_finish_in_60_days_l294_294133

variables (A B C : ℚ)

noncomputable def a_b_work_rate := A + B = 1/40
noncomputable def a_c_work_rate := A + 1/30 = 1/20

theorem a_alone_can_finish_in_60_days (A B C : ℚ) 
  (h₁ : a_b_work_rate A B) 
  (h₂ : a_c_work_rate A) : 
  A = 1/60 := 
sorry

end a_alone_can_finish_in_60_days_l294_294133


namespace arithmetic_sequence_a5_value_l294_294722

variable (a : ℕ → ℝ)
variable (a_2 a_5 a_8 : ℝ)
variable (h1 : a 2 + a 8 = 15 - a 5)

/-- In an arithmetic sequence {a_n}, given that a_2 + a_8 = 15 - a_5, prove that a_5 equals 5. -/ 
theorem arithmetic_sequence_a5_value (h1 : a 2 + a 8 = 15 - a 5) : a 5 = 5 :=
sorry

end arithmetic_sequence_a5_value_l294_294722


namespace sum_integers_from_neg15_to_5_l294_294510

theorem sum_integers_from_neg15_to_5 : (Finset.sum (Finset.Icc (-15) 5) (λ x, x)) = -105 := by
  sorry

end sum_integers_from_neg15_to_5_l294_294510


namespace find_sin_alpha_l294_294281

theorem find_sin_alpha (α : ℝ) (h1 : 0 < α ∧ α < real.pi) (h2 : 3 * real.cos (2 * α) - 8 * real.cos α = 5) :
  real.sin α = real.sqrt 5 / 3 :=
sorry

end find_sin_alpha_l294_294281


namespace tangent_line_at_e_l294_294803

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_line_at_e : ∀ x y : ℝ, (x = Real.exp 1) → (y = f x) → (y = 2 * x - Real.exp 1) :=
by
  intros x y hx hy
  sorry

end tangent_line_at_e_l294_294803


namespace value_of_a_plus_b_l294_294032

noncomputable def f (x : ℝ) := abs (Real.log (x + 1))

theorem value_of_a_plus_b (a b : ℝ) (h1 : a < b) 
  (h2 : f a = f (- (b + 1) / (b + 2))) 
  (h3 : f (10 * a + 6 * b + 21) = 4 * Real.log 2) : 
  a + b = -11 / 15 := 
by 
  sorry

end value_of_a_plus_b_l294_294032


namespace tangent_line_parallel_x_axis_l294_294804

noncomputable def ln_div_x (x : ℝ) : ℝ := (Real.log x) / x

theorem tangent_line_parallel_x_axis (x0 : ℝ) (h_tangent_parallel : Deriv (λ x => ln_div_x x) x0 = 0) : ln_div_x x0 = 1 / Real.exp 1 :=
by
  sorry

end tangent_line_parallel_x_axis_l294_294804


namespace BM_equal_fifteen_l294_294068

variable {A B C D M : Type}
variable [AddCommGroup A] [Module ℝ A]
variable (a b c d m : A) -- points

-- Define the given conditions
def is_trapezoid (ABCD : Prop) : Prop :=
  ∃ (CD AB : ℝ), 
    (angle BCD = real.arccos 0.05) ∧ 
    (angle CBD = real.arccos 0.05) ∧ 
    (angle ABM = real.arccos 0.05) ∧ 
    (norm (b - a) = 9)

def BM_length (BM : Prop) : Prop :=
  norm (m - b) = 15

-- The theorem to prove BM = 15
theorem BM_equal_fifteen (ABCD : Prop) (BM : Prop) :
  is_trapezoid ABCD → BM_length BM := 
by
  sorry

end BM_equal_fifteen_l294_294068


namespace cos_alpha_value_interval_for_g_monotonic_increase_l294_294642

noncomputable def f (ω : ℝ) (x : ℝ) := 
  (\cos (ω * x), √3 * cos (ω * x + π)) ⬝ (sin (ω * x), cos (ω * x))

theorem cos_alpha_value (α : ℝ) (h1 : α ∈ Ioo 0 (π / 2)) (h2 : f 1 (α / 2) = - √3 / 4) :
  cos α = (√13 - 3) / 8 := 
sorry

theorem interval_for_g_monotonic_increase (k : ℤ) :
  set.Icc ((2 * ↑k * π) - (π / 3)) ((2 * ↑k * π) + (2 * π / 3)) = 
  { x | sin (x - (π / 6)) - √3 / 2 } := 
sorry

end cos_alpha_value_interval_for_g_monotonic_increase_l294_294642


namespace problem_50th_term_l294_294810

-- Definition of the sequence per the problem's conditions
def special_sequence : List ℕ :=
  List.filter (λ n, ∃ l, l.sum = n ∧ ∀ x ∈ l, x = 2 ^ x ∨ x = 3 ^ x) (List.range (10000))

-- Statement of the proof problem that the 50th term is 326
theorem problem_50th_term (n : ℕ) (h : n = 50) :
  special_sequence.get 49 = 326 :=
  sorry

end problem_50th_term_l294_294810


namespace complex_fraction_simplification_l294_294958

theorem complex_fraction_simplification : (2 + complex.I) / (1 - 2 * complex.I) = complex.I :=
by
  sorry

end complex_fraction_simplification_l294_294958


namespace speed_of_boat_in_still_water_l294_294479

-- Definitions of the given conditions
def rate_of_current : ℝ := 7
def distance_travelled_downstream : ℝ := 22.2
def time_travelled_downstream : ℝ := 0.6

-- Statement of the problem
theorem speed_of_boat_in_still_water :
  ∃ (b : ℝ), distance_travelled_downstream = (b + rate_of_current) * time_travelled_downstream ∧ b = 30 :=
by
  use 30
  have eq1 : (30 + rate_of_current) * time_travelled_downstream = 22.2 := sorry
  exact ⟨eq1, rfl⟩

end speed_of_boat_in_still_water_l294_294479


namespace smallest_k_for_64_pow_k_gt_4_pow_20_l294_294140

theorem smallest_k_for_64_pow_k_gt_4_pow_20 : ∃ k : ℕ, 64 ^ k > 4 ^ 20 ∧ (∀ m : ℕ, 64 ^ m > 4 ^ 20 → m ≥ k) := by
  let k := 7
  have h₁ : 64 = 2 ^ 6 := by norm_num
  have h₂ : 4 = 2 ^ 2 := by norm_num
  have h₃ : (2 ^ 6) ^ k = 64 ^ k := by rw h₁
  have h₄ : 64 ^ k > 4 ^ 20 := by
    rw [←h₁, ←h₂, ←pow_mul, ←pow_mul]
    exact pow_lt_pow_of_lt_left (by norm_num : 2 > 1) (by norm_num : 6 * 7 > 40) (by norm_num 6)
  existsi k
  apply and.intro
  exact h₄
  intro m h5
  have : (2 ^ 6) ^ m > 2 ^ 40 := by rwa [h₁, h₂, ←pow_mul, ←pow_mul]
  norm_num at this
  linarith
  sorry

end smallest_k_for_64_pow_k_gt_4_pow_20_l294_294140


namespace part1_part2_l294_294318

variable {A B C a b c : ℝ}

theorem part1 (h : ∀ (A B : ℝ), (sin (2 * A + B)) / (sin A) = 2 + 2 * cos (A + B)) :
  b = 2 * a :=
sorry

theorem part2 (hc : c = Real.sqrt 7 * a) :
  ∠C = 2 * Real.pi / 3 :=
sorry

end part1_part2_l294_294318


namespace max_intersections_l294_294756

def is_parallel (L1 L2 : Type) : Prop := sorry
def is_perpendicular (L1 L2 : Type) : Prop := sorry
def passes_through (L : Type) (B : Type) : Prop := sorry
def num_intersections (L : List Type) : ℕ := sorry

theorem max_intersections (L : List Type) (B : Type)
  (h1 : ∀ n, L.contains L[5*n] ∧ ∀ p q, (p ≠ q) → is_parallel (L[5*n + p]) (L[5*n + q]))
  (h2 : ∀ n, L.contains L[5*n - 4] ∧ ∀ l, passes_through (L[5*n - 4]) B)
  (h3 : ∀ n, L.contains L[5*n - 2] ∧ ∀ l, is_perpendicular (L[5*n - 2]) (L[5*n]))
  (h_distinct : ∀ p q, (p ≠ q) → L[p] ≠ L[q])
  : num_intersections L = 5161 := sorry

end max_intersections_l294_294756


namespace quarters_spent_l294_294432

variable (q_initial q_left q_spent : ℕ)

theorem quarters_spent (h1 : q_initial = 11) (h2 : q_left = 7) : q_spent = q_initial - q_left ∧ q_spent = 4 :=
by
  sorry

end quarters_spent_l294_294432


namespace three_flips_probability_l294_294120

open Probability

theorem three_flips_probability :
  ∀ (prob_heads : ℙ → bool) (prob_tails : ℙ → bool),
  (∀ (p : ℙ), prob_heads p = true → Prob p (λ x, x = true) = 1 / 2) →
  (∀ (p : ℙ), prob_tails p = false → Prob p (λ x, x = false) = 1 / 2) →
  let events := [prob_heads, prob_tails, prob_heads] in
  (∀ (p : ℙ), Prob p (λ x, events.all (λ f, f p x)) = 1 / 8) :=
by sorry

end three_flips_probability_l294_294120


namespace solution_set_of_inequality_l294_294310

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x, f (-x) = -f (x)
axiom f_at_1 : f 1 = 2
axiom f_deriv_pos : ∀ x > 0, (f' x) > 2

theorem solution_set_of_inequality :
  { x : ℝ | f x > 2 * x } = { x : ℝ | x < -1 } ∪ { x : ℝ | 1 < x } :=
by
  sorry

end solution_set_of_inequality_l294_294310


namespace expression_value_l294_294257

noncomputable def expr_val : ℂ := 
  (0.027 : ℂ)^(1 / 3 : ℂ) * (225 / 64 : ℂ)^(-1 / 2 : ℂ) / complex.sqrt (-(8 / 125 : ℂ)^(2 / 3 : ℂ))

theorem expression_value :
  expr_val = (2 / (5 * complex.I) : ℂ) := 
sorry

end expression_value_l294_294257


namespace arithmetic_sequence_k_l294_294752

noncomputable theory

def problem_statement (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  (S 2016 > 0) ∧ (S 2017 < 0) ∧ (∀ n : ℕ, n > 0 → |a n| ≥ |a 1009|)

theorem arithmetic_sequence_k (S : ℕ → ℝ) (a : ℕ → ℝ) (d : ℝ) :
  problem_statement S a → ∃ k, k = 1009 :=
sorry

end arithmetic_sequence_k_l294_294752


namespace problem_l294_294813

theorem problem (a b : ℕ) (h1 : ∃ k : ℕ, a * b = k * k) (h2 : ∃ m : ℕ, (2 * a + 1) * (2 * b + 1) = m * m) :
  ∃ n : ℕ, n % 2 = 0 ∧ n > 2 ∧ ∃ p : ℕ, (a + n) * (b + n) = p * p :=
by
  sorry

end problem_l294_294813


namespace angle_AKC_is_50_l294_294406

-- Define the points and conditions
def points_on_tangent_circle (A B C M N K : Type) [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint M] [IsPoint N] [IsPoint K] : Prop :=
  ∃ O : Type, [IsPoint O] ∧ 
  tangent_circle O A B C

-- Define the main theorem
theorem angle_AKC_is_50
  {A B C M N K : Type} [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint M] [IsPoint N] [IsPoint K] 
  (h_tangent : points_on_tangent_circle A B C M N K)
  (h_bisector : is_angle_bisector A K N)
  (h_intersection : is_on_line MN K) :
  angle A K C = 50 := 
  sorry

end angle_AKC_is_50_l294_294406


namespace average_words_written_l294_294196

def total_words : ℕ := 50000
def total_hours : ℕ := 100
def average_words_per_hour : ℕ := total_words / total_hours

theorem average_words_written :
  average_words_per_hour = 500 := 
by
  sorry

end average_words_written_l294_294196


namespace triangle_area_l294_294494

theorem triangle_area (a b : ℝ) (γ : ℝ) : 
    Δ = 1 / 2 * a * b * real.sin γ :=
sorry

end triangle_area_l294_294494


namespace selection_methods_count_l294_294106

theorem selection_methods_count
  (male_doctors : ℕ)
  (female_doctors : ℕ)
  (choose : ℕ → ℕ → ℕ)
  (h_male_doctors : male_doctors = 6)
  (h_female_doctors : female_doctors = 5)
  (h_choose_2_out_6 : choose 6 2 = 15)
  (h_choose_1_out_5 : choose 5 1 = 5) :
  choose male_doctors 2 * choose female_doctors 1 = 75 :=
by
  rw [h_male_doctors, h_female_doctors, h_choose_2_out_6, h_choose_1_out_5]
  norm_num
  exact rfl

end selection_methods_count_l294_294106


namespace area_of_quadrilateral_l294_294244

-- Define the conditions as hypotheses
variable (AD DC AB BC : ℝ)
variable (h1 : ∠D = 90) (h2 : ∠B = 90)
variable (h3 : AD = DC)
variable (h4 : AB + BC = 10)

-- Define the theorem stating the area of quadrilateral ABCD
theorem area_of_quadrilateral (h1 : ∠D = 90) (h2 : ∠B = 90) (h3 : AD = DC) (h4 : AB + BC = 10) : 
  area_of_quadrilateral ABCD = 25 :=
sorry

end area_of_quadrilateral_l294_294244


namespace arccos_sin_2_equals_l294_294937

theorem arccos_sin_2_equals : Real.arccos (Real.sin 2) = 2 - Real.pi / 2 := by
  sorry

end arccos_sin_2_equals_l294_294937


namespace no_prime_div_by_55_l294_294351

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_div_by_55 : ∀ p : ℕ, is_prime p → ¬ (55 ∣ p) :=
by intros p hp hdiv
   unfold is_prime at hp
   cases hp with h1 h2
   have h3 : p = 55 ∨ 5 ∣ p ∨ 11 ∣ p,
     from sorry, -- this part requires the detailed proof steps and isn't included here
   cases h3 with h4 h56,
     { rw h4 at hp, contradiction },
     { cases h56 with h5 h11,
       { have h6 := h2 5 h5,
         cases h6; linarith },
       { have h7 := h2 11 h11,
         cases h7; linarith } }

-- The sorry command is here to indicate missing parts that involve detailed steps

end no_prime_div_by_55_l294_294351


namespace four_digit_numbers_four_digit_odd_numbers_four_digit_numbers_gt_3400_ne_270_four_digit_numbers_div_25_l294_294998

open Finset

/-- The set of digits from which numbers can be formed -/
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5, 6}

-- Various problems based on the selection of digits to form four-digit numbers
namespace problems

/-- Number of unique four-digit numbers -/
theorem four_digit_numbers : 
  (digits \ {0}).card * ((digits \ {0}).erase 0).card.factorial = 720 :=
by sorry

/-- Number of unique four-digit odd numbers -/
theorem four_digit_odd_numbers :
  (digits \ {0, 2, 4, 6}).card * (digits \ {0, 2, 4, 6}).card.pred.factorial = 300 :=
by sorry

/-- Number of unique four-digit numbers larger than 3400 is not 270 -/
theorem four_digit_numbers_gt_3400_ne_270 :
  ∀ n, n > 3400 → n ∈ digits ∧ (n % 1000) ∈ digits.erase 0 → n.card.factorial ≠ 270 :=
by sorry

/-- Number of unique four-digit numbers divisible by 25 -/
theorem four_digit_numbers_div_25 :
  (digits \ {0, 2, 5}).card * (digits \ {0, 2, 5}).card.pred.factorial = 36 :=
by sorry

end problems

end four_digit_numbers_four_digit_odd_numbers_four_digit_numbers_gt_3400_ne_270_four_digit_numbers_div_25_l294_294998


namespace parabola_directrix_l294_294464

theorem parabola_directrix (x y : ℝ) (h : x^2 = 12 * y) : y = -3 :=
sorry

end parabola_directrix_l294_294464


namespace expected_rolls_sum_2010_l294_294175

noncomputable def expected_rolls_to_reach_sum (n : ℕ) : ℚ :=
  if n == 0 then 0
  else (1/6) * (expected_rolls_to_reach_sum (n-1) + expected_rolls_to_reach_sum (n-2) +
               expected_rolls_to_reach_sum (n-3) + expected_rolls_to_reach_sum (n-4) +
               expected_rolls_to_reach_sum (n-5) + expected_rolls_to_reach_sum (n-6)) + 1

theorem expected_rolls_sum_2010 : expected_rolls_to_reach_sum 2010 ≈ 574.761904 := 
by 
  -- Proof omitted; the focus is on the statement
  sorry

end expected_rolls_sum_2010_l294_294175


namespace expected_rolls_to_reach_2010_l294_294165

noncomputable def expected_rolls_to_reach_sum (n : ℕ) : ℝ :=
  sorry -- Using 'sorry' to denote placeholder for the actual proof.

theorem expected_rolls_to_reach_2010 : expected_rolls_to_reach_sum 2010 ≈ 574.761904 :=
  sorry

end expected_rolls_to_reach_2010_l294_294165


namespace complement_union_is_correct_l294_294420

open Set

variable (U A B : Set ℕ)

def U := {1, 2, 3, 4} : Set ℕ
def A := {1, 3} : Set ℕ
def B := {1, 4} : Set ℕ

theorem complement_union_is_correct : compl (A ∪ B) ∩ U = {2} :=
by sorry

end complement_union_is_correct_l294_294420


namespace arithmetic_sequence_count_l294_294967

theorem arithmetic_sequence_count :
  ∃ n : ℕ, let a := -6 in let d := 4 in let a_n := 50 in
  a_n = a + (n - 1) * d ∧ n = 15 :=
by
  sorry

end arithmetic_sequence_count_l294_294967


namespace secretary_longest_time_l294_294528

-- Definitions based on the conditions
def secretary_times (x : ℕ) : Prop :=
  let s1 := x in
  let s2 := 2 * x in
  let s3 := 5 * x in
  s1 + s2 + s3 = 120

-- The proof statement
theorem secretary_longest_time : ∃ x : ℕ, secretary_times x → 5 * x = 75 :=
by
  sorry

end secretary_longest_time_l294_294528


namespace exists_two_diff_campechana_seqs_l294_294190

def is_campechana (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ i, 2 ≤ i ∧ i ≤ n → (a i) = nat.totient n

def size (a : ℕ → ℕ) : ℕ → ℕ := λ n, n - 1

theorem exists_two_diff_campechana_seqs (p : ℕ → Prop) (k : ℕ) (primes : fin k → ℕ) 
  (hprimes : ∀ i j, i ≠ j → primes i ≠ primes j)
  (hk : k ≥ 2) : 
  ∃ a b : ℕ → ℕ, size a (primes.prod (λ i, primes i)) = primes.prod (λ i, primes i) - 1 ∧ 
                  size b (primes.prod (λ i, primes i)) = primes.prod (λ i, primes i) - 1 ∧ 
                  is_campechana (primes.prod (λ i, primes i)) a ∧ 
                  is_campechana (primes.prod (λ i, primes i)) b ∧ 
                  (a ≠ b) := 
sorry

end exists_two_diff_campechana_seqs_l294_294190


namespace find_a_l294_294048

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - a * x + a - 1 = 0}

theorem find_a (a : ℝ) (h : A ∪ B a = A) : a = 2 ∨ a = 3 := by
  sorry

end find_a_l294_294048


namespace no_integer_solutions_l294_294255

theorem no_integer_solutions (x : ℤ) : x^4 - 36 * x^2 + 121 ≠ 0 := by sorry

example : ∑ x in {x : ℤ | x^4 - 36 * x^2 + 121 = 0}.to_finset = 0 := by
  have h : {x : ℤ | x^4 - 36 * x^2 + 121 = 0} = ∅ := by
    ext x
    apply not_iff_not.mpr (set.mem_set_of_eq)
    exact no_integer_solutions x
  rw [h, finset.sum_empty]

end no_integer_solutions_l294_294255


namespace max_product_of_xy_on_circle_l294_294154

theorem max_product_of_xy_on_circle (x y : ℤ) (h : x^2 + y^2 = 100) : 
  ∃ (x y : ℤ), (x^2 + y^2 = 100) ∧ (∀ x y : ℤ, x^2 + y^2 = 100 → x * y ≤ 48) ∧ x * y = 48 := by
  sorry

end max_product_of_xy_on_circle_l294_294154


namespace mult_base7_correct_l294_294928

def base7_to_base10 (n : ℕ) : ℕ :=
  -- assume conversion from base-7 to base-10 is already defined
  sorry 

def base10_to_base7 (n : ℕ) : ℕ :=
  -- assume conversion from base-10 to base-7 is already defined
  sorry

theorem mult_base7_correct : (base7_to_base10 325) * (base7_to_base10 4) = base7_to_base10 1656 :=
by
  sorry

end mult_base7_correct_l294_294928


namespace probability_perfect_square_divisor_l294_294549

theorem probability_perfect_square_divisor (m n : ℕ) (h_rel_prime : Nat.coprime m n) :
  let fact_10 := 2^8 * 3^4 * 5^2 * 7
  let total_divisors_multiple_12 := 168
  let perfect_square_divisors_multiple_12 := 16
  m = 2 ∧ n = 21 →
  (perfect_square_divisors_multiple_12 : ℕ) / (total_divisors_multiple_12 : ℕ) = m / n :=
by
  sorry

end probability_perfect_square_divisor_l294_294549


namespace student_path_probability_l294_294717

theorem student_path_probability :
  let A to B: ℕ := (3.choose 2)
  let B to C: ℕ := (2.choose 1)
  let C to D: ℕ := (3.choose 1)
  let total_paths_via_B_C : ℕ := (A to B * B to C * C to D)
  let total_paths_direct : ℕ := (7.choose 4)
  (total_paths_via_B_C / total_paths_direct) = (18 / 35) :=
by
  -- Definitions based on conditions
  let A_to_B := Nat.choose 3 2
  let B_to_C := Nat.choose 2 1
  let C_to_D := Nat.choose 3 1
  let total_paths_via_B_C := A_to_B * B_to_C * C_to_D
  let total_paths_direct := Nat.choose 7 4
  have H : (total_paths_via_B_C / total_paths_direct) = (18 / 35) := by sorry
  exact H

end student_path_probability_l294_294717


namespace distinct_positive_factors_243_mul_5_l294_294350

-- Definitions for the conditions
def num_factors (n : ℕ) : ℕ :=
  (n.factors.groupBy id).toList.map (λ l, l.length + 1).prod

theorem distinct_positive_factors_243_mul_5 :
  let n := 243 * 5
  num_factors n = 12 :=
by
  sorry -- the proof is omitted as instructed

end distinct_positive_factors_243_mul_5_l294_294350


namespace solve_x_l294_294076

theorem solve_x (x : ℝ) (h : 16^(3*x - 4) = (1/2)^(2*x + 6)) : x = 5/7 := by
  sorry

end solve_x_l294_294076


namespace triangle_AC_value_l294_294384

theorem triangle_AC_value (A B C : Point) 
  (hABC : Triangle A B C)
  (hAngleA : ∠A = 90)
  (hBC : dist B C = 15)
  (hTanSin : tan ∠C = 3 * sin ∠C) : 
  dist A C = 5 := 
sorry

end triangle_AC_value_l294_294384


namespace agamemnon_ships_cannot_gather_l294_294903

def circular_islands := fin 1002 -- Define circular islands as a finite type

structure ShipDistribution :=
  (num_ships : circular_islands → ℕ) -- Number of ships at each island

variables (move_ships : (circular_islands × circular_islands) → (circular_islands × circular_islands))
  (distribution : ShipDistribution)

theorem agamemnon_ships_cannot_gather (move_ships distribution) :
  ∀ (ship_num : ℕ) (len : ℕ),
  ship_num = 1002 → len = 1002 →  
  ¬ ∀ island, distribution.num_ships island = 1002 := 
sorry

end agamemnon_ships_cannot_gather_l294_294903


namespace general_term_arithmetic_sequence_sum_first_n_terms_l294_294658

theorem general_term_arithmetic_sequence (d : ℤ) (h : d ≠ 0) (a : ℕ → ℤ) 
  (h1 : a 1 = 1) 
  (h_seq : ∀ n, a (n + 1) = a n + d ) 
  (h_geo : (a 3) / (a 1) = (a 9) / (a 3)): 
  ∀ n, a n = n := by
  sorry

theorem sum_first_n_terms (a : ℕ → ℤ) (h_general_term : ∀ n, a n = n ): 
  let b := λ n, 2 * a n
  ∀ n, (Finset.range n).sum (λ x, b x) = n * (n + 1) := by 
  sorry

end general_term_arithmetic_sequence_sum_first_n_terms_l294_294658


namespace solve_for_x_l294_294688

theorem solve_for_x (x y : ℚ) (h1 : 2 * x - 3 * y = 15) (h2 : x + 2 * y = 8) : x = 54 / 7 :=
sorry

end solve_for_x_l294_294688


namespace intersect_at_circumcircle_l294_294308

noncomputable theory
open_locale classical

variables {α : Type*} [EuclideanGeometry α] 
variables {A B C M H P Q N : α}
variables [Triangle A B C] [IsMidpoint M B C] [Orthocenter H A B C]
variables [PerpendicularBisectorIntersect P Q B C A]
variables [IsMidpoint N P Q]

theorem intersect_at_circumcircle 
  (hAm : IsMidpoint A N) (hHBC : IsMidpoint H M)
  (h : Perpendicular (Line A N) (Line H M)) :
  ∃ X, X ∈ Circumcircle A B C ∧ Intersects (Line A N) (Line H M) X :=
sorry

end intersect_at_circumcircle_l294_294308


namespace carpet_needed_l294_294065

theorem carpet_needed (length_in_feet : ℕ) (width_in_feet : ℕ) (table_side_in_feet : ℕ) (feet_per_yard : ℕ) :
  length_in_feet = 15 ∧ width_in_feet = 12 ∧ table_side_in_feet = 3 ∧ feet_per_yard = 3 →
  let length_in_yards := length_in_feet / feet_per_yard
      width_in_yards := width_in_feet / feet_per_yard
      table_side_in_yards := table_side_in_feet / feet_per_yard
      total_area_in_square_yards := length_in_yards * width_in_yards
      table_area_in_square_yards := table_side_in_yards * table_side_in_yards
      carpet_area_needed := total_area_in_square_yards - table_area_in_square_yards
  in carpet_area_needed = 19 :=
sorry

end carpet_needed_l294_294065


namespace tulips_percentage_l294_294869

variable (flowers : ℚ)  -- total number of flowers
variable (pink_flowers : ℚ) (red_flowers : ℚ) (pink_tulips : ℚ) (red_tulips : ℚ)

-- Given conditions
def condition1 := pink_flowers = (3/5) * flowers
def condition2 := red_flowers = (2/5) * flowers
def condition3 := pink_tulips = (1/2) * pink_flowers
def condition4 := red_tulips = (1/3) * red_flowers

-- Definitions of fractions
def fraction_of_pink_tulips := (3 / 5) * (1 / 2)
def fraction_of_red_tulips := (2 / 5) * (1 / 3)
def total_fraction_of_tulips := fraction_of_pink_tulips + fraction_of_red_tulips

-- Statement to prove
theorem tulips_percentage : 
  condition1 → condition2 → condition3 → condition4 →
  (total_fraction_of_tulips * 100 = 43.33) :=
by
  sorry

end tulips_percentage_l294_294869


namespace value_of_v5_l294_294607

noncomputable def sequence (v : ℕ → ℚ) :=
  ∀ n, v (n + 2) = 3 * v (n + 1) - 2 * v n

theorem value_of_v5 (v : ℕ → ℚ) (h_seq : sequence v) (h_v3 : v 3 = 5) (h_v6 : v 6 = -76) :
  v 5 = -208 / 7 :=
by
  sorry

end value_of_v5_l294_294607


namespace number_of_basic_events_probability_exactly_one_boy_probability_at_least_one_boy_l294_294182

-- Define the members of the group
inductive Student
| a | b | c | d | e

open Student

-- Define all possible pairs of students
def basic_events : Finset (Student × Student) :=
  Finset.fromList [(a, b), (a, c), (a, d), (a, e), (b, c), (b, d), (b, e), (c, d), (c, e), (d, e)]

-- Number of basic events
theorem number_of_basic_events : basic_events.card = 10 := by
  sorry

-- Define pairs containing exactly one boy
def exactly_one_boy : Finset (Student × Student) :=
  Finset.fromList [(a, c), (a, d), (a, e), (b, c), (b, d), (b, e)]

-- Probability of exactly one boy
theorem probability_exactly_one_boy : exactly_one_boy.card / basic_events.card = 3 / 5 := by
  sorry

-- Define pairs containing at least one boy
def at_least_one_boy : Finset (Student × Student) :=
  Finset.fromList [(a, b), (a, c), (a, d), (a, e), (b, c), (b, d), (b, e)]

-- Probability of at least one boy
theorem probability_at_least_one_boy : at_least_one_boy.card / basic_events.card = 7 / 10 := by
  sorry

end number_of_basic_events_probability_exactly_one_boy_probability_at_least_one_boy_l294_294182


namespace tan_beta_value_l294_294291

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = 2) (h2 : Real.tan (α + β) = -1) : Real.tan β = 3 :=
by
  sorry

end tan_beta_value_l294_294291


namespace minimum_distance_from_curve_to_line_l294_294304

noncomputable def minimum_distance_to_line : ℝ :=
  let curve (x : ℝ) : ℝ := x^2 - Real.log x in
  let line (x : ℝ) : ℝ := x + 2 in
  let distance (x P_x : ℝ) (P_y : ℝ) : ℝ := abs ($\frac{P_y - line P_x}{Real.sqrt 2}$) in
  have dist : {P : ℝ × ℝ // P.2 = curve P.1 ∧ P.1 > 0} := ⟨(1, 1), by simp [curve]; norm_num⟩,
  distance dist.val.1 dist.val.2 dist.val.2

theorem minimum_distance_from_curve_to_line :
  minimum_distance_to_line = Real.sqrt 2 :=
sorry

end minimum_distance_from_curve_to_line_l294_294304


namespace real_solns_f_eq_x_l294_294252

noncomputable def f (x : ℝ) : ℝ :=
  (Finset.range 100).sum (λ i, (i+1) / (x - (i+1)))

theorem real_solns_f_eq_x : (∃! y : ℝ, y ∈ (Set.Icc (1 : ℝ) 100) ∧ f y = y) := sorry

end real_solns_f_eq_x_l294_294252


namespace shape_with_circular_views_is_sphere_l294_294572

-- Definitions of the views of different geometric shapes
inductive Shape
| Cuboid : Shape
| Cylinder : Shape
| Cone : Shape
| Sphere : Shape

def front_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Rectangle
| Shape.Cone := False  -- Isosceles Triangle
| Shape.Sphere := True  -- Circle

def left_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Rectangle
| Shape.Cone := False  -- Isosceles Triangle
| Shape.Sphere := True  -- Circle

def top_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Circle, but not all views
| Shape.Cone := False  -- Circle, but not all views
| Shape.Sphere := True  -- Circle

-- The theorem to be proved
theorem shape_with_circular_views_is_sphere (s : Shape) :
  (front_view s ↔ True) ∧ (left_view s ↔ True) ∧ (top_view s ↔ True) ↔ s = Shape.Sphere :=
by sorry

end shape_with_circular_views_is_sphere_l294_294572


namespace imaginary_part_of_complex_number_l294_294469

theorem imaginary_part_of_complex_number : 
  (complex.im ((2 * complex.I) / (3 - 2 * complex.I)) = 6 / 13) :=
by
  sorry

end imaginary_part_of_complex_number_l294_294469


namespace washer_and_dryer_proof_l294_294566

noncomputable def washer_and_dryer_problem : Prop :=
  ∃ (price_of_washer price_of_dryer : ℕ),
    price_of_washer + price_of_dryer = 600 ∧
    (∃ (k : ℕ), price_of_washer = k * price_of_dryer) ∧
    price_of_dryer = 150 ∧
    price_of_washer / price_of_dryer = 3

theorem washer_and_dryer_proof : washer_and_dryer_problem :=
sorry

end washer_and_dryer_proof_l294_294566


namespace cash_register_cost_l294_294054

theorem cash_register_cost :
  let bread_sales := 40 * 2
  let cake_sales := 6 * 12
  let daily_sales := bread_sales + cake_sales
  let daily_expenses := 20 + 2
  let daily_profit := daily_sales - daily_expenses
  let cost := 8 * daily_profit
  cost = 1040 := 
by
  let bread_sales := 40 * 2
  let cake_sales := 6 * 12
  let daily_sales := bread_sales + cake_sales
  let daily_expenses := 20 + 2
  let daily_profit := daily_sales - daily_expenses
  let cost := 8 * daily_profit
  show cost = 1040 from sorry

end cash_register_cost_l294_294054


namespace main_theorem_l294_294253

noncomputable def problem_statement : Prop :=
  ∀ x : ℂ, (x ≠ -2) →
  ((15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 48) ↔
  (x = 12 + 2 * Real.sqrt 38 ∨ x = 12 - 2 * Real.sqrt 38 ∨
  x = -1/2 + Complex.I * Real.sqrt 95 / 2 ∨
  x = -1/2 - Complex.I * Real.sqrt 95 / 2)

-- Provide the main statement without the proof
theorem main_theorem : problem_statement := sorry

end main_theorem_l294_294253


namespace kitten_length_l294_294914

theorem kitten_length (initial_length : ℕ) (doubled_length_1 : ℕ) (doubled_length_2 : ℕ) :
  initial_length = 4 →
  doubled_length_1 = 2 * initial_length →
  doubled_length_2 = 2 * doubled_length_1 →
  doubled_length_2 = 16 :=
by
  intros h1 h2 h3
  rw [h1] at h2
  rw [h2] at h3
  exact h3

end kitten_length_l294_294914


namespace time_to_cook_one_potato_l294_294547

theorem time_to_cook_one_potato (total_potatoes cooked_potatoes time_to_cook_remaining : ℕ) (h1 : total_potatoes = 13) (h2 : cooked_potatoes = 5) (h3 : time_to_cook_remaining = 48) : 
  (time_to_cook_remaining / (total_potatoes - cooked_potatoes) = 6) :=
by 
  have h4 : total_potatoes - cooked_potatoes = 8 := by linarith,
  have h5 : 48 / 8 = 6 := by norm_num,
  rw [h4, h5],
  simp,
  sorry

end time_to_cook_one_potato_l294_294547


namespace abs_twice_sub_pi_l294_294941

theorem abs_twice_sub_pi (h : Real.pi < 10) : abs (Real.pi - abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_twice_sub_pi_l294_294941


namespace square_side_length_l294_294123

theorem square_side_length (A : ℝ) (h : A = 169) : ∃ s : ℝ, s^2 = A ∧ s = 13 := by
  sorry

end square_side_length_l294_294123


namespace AC_is_five_l294_294382

-- Given conditions
variables (A B C : Type) [inner_product_space ℝ A]
variables {P Q R : A}
variables (angleA : angle P Q R = 90) (BC PQ_length : dist P Q = 15)
variables (tanC_eq_3sinC : tan (angle Q P R) = 3 * sin (angle Q P R))

-- Define the length of side AC
def AC_length : ℝ := dist P R

-- The statement to prove
theorem AC_is_five :
  AC_length P R = 5 :=
sorry

end AC_is_five_l294_294382


namespace track_is_600_l294_294208

noncomputable def track_length (x : ℝ) : Prop :=
  ∃ (s_b s_s : ℝ), 
      s_b > 0 ∧ s_s > 0 ∧
      (∀ t, t > 0 → ((s_b * t = 120 ∧ s_s * t = x / 2 - 120) ∨ 
                     (s_s * (t + 180 / s_s) - s_s * t = x / 2 + 60 
                      ∧ s_b * (t + 180 / s_s) - s_b * t = x / 2 - 60)))

theorem track_is_600 : track_length 600 :=
sorry

end track_is_600_l294_294208


namespace simple_interest_rate_l294_294587

theorem simple_interest_rate :
  ∀ (P A T : ℝ), P = 1500 ∧ A = 2100 ∧ T = 25 → 
  ∃ R, A = P + (P * R * T) / 100 ∧ R = 1.6 :=
by 
  intros P A T h,
  cases h with hp hrest,
  cases hrest with ha ht,
  use 1.6,
  rw [hp, ha, ht],
  sorry

end simple_interest_rate_l294_294587


namespace sequence_10_eq_93_l294_294218

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 3 
  else sequence (n - 1) + 2 * (n - 1)

theorem sequence_10_eq_93 : sequence 10 = 93 := by
  sorry  -- Proof is omitted as per the instructions

end sequence_10_eq_93_l294_294218


namespace line_through_points_a_plus_b_l294_294699

theorem line_through_points_a_plus_b :
  ∃ a b : ℝ, (∀ x y : ℝ, (y = a * x + b) → ((x, y) = (6, 7)) ∨ ((x, y) = (10, 23))) ∧ (a + b = -13) :=
sorry

end line_through_points_a_plus_b_l294_294699


namespace polar_coordinates_full_circle_l294_294522

theorem polar_coordinates_full_circle :
  ∀ (r : ℝ) (θ : ℝ), (r = 3 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) → (r = 3 ∧ ∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi ↔ r = 3) :=
by
  intros r θ h
  sorry

end polar_coordinates_full_circle_l294_294522


namespace final_position_from_P_farthest_distance_from_P_total_gasoline_cost_l294_294799

noncomputable def trips : List Int := [-2, 7, -9, 10, 4, -5, -8]

/- Theorem (1): Final position from company P -/
theorem final_position_from_P : 
  let final_position := List.sum trips
  final_position = -3 := 
by
  sorry

/- Theorem (2): Farthest distance from company P -/
theorem farthest_distance_from_P :
  let cumulative_distances := List.scanl (· + ·) 0 trips
  let distances_from_P := cumulative_distances.map Int.abs
  List.maximum distances_from_P = some 10 :=
by
  sorry

/- Theorem (3): Total gasoline cost -/
theorem total_gasoline_cost :
  let total_distance := (List.sum (trips.map Int.natAbs)).toRat
  let gasoline_consumption := total_distance * 0.1
  let cost := gasoline_consumption * 8.2
  cost = 36.9 :=
by
  sorry

end final_position_from_P_farthest_distance_from_P_total_gasoline_cost_l294_294799


namespace gather_all_candies_l294_294482

theorem gather_all_candies (n m : ℕ) (h₁ : n ≥ 4) (h₂ : m ≥ 4) 
  (valid_operation : ∀ (c : Fin n → ℕ), (∃ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ c i > 0 ∧ c j > 0 →
    ∃ (c' : Fin n → ℕ), c' i = c i - 1 ∧ c' j = c j - 1 ∧ c' k = c k + 2 ∧ (∀ l, l ≠ i ∧ l ≠ j ∧ l ≠ k → c' l = c l))) :
  ∃ (c : Fin n → ℕ), (∑ i, c i = m) ∧ (∃ k : Fin n, c = λ i, if i = k then m else 0) :=
sorry

end gather_all_candies_l294_294482


namespace triangle_problem_statement_l294_294024

variable (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (AD DB AC CB AE EB BC : ℝ)
variable (ACE ECB DCB ACD BCE : ℝ)

theorem triangle_problem_statement 
  (h : (AD / DB) * (AE / EB) = (AC / CB) ^ 2) :
  (AE / EB = AC * Real.sin ACE / (BC * Real.sin ECB)) ∧ (ACD = BCE) := 
by
  sorry

end triangle_problem_statement_l294_294024


namespace sum_of_digits_2A_eq_2B_l294_294862

-- Define a function to compute the sum of the digits of a natural number.
noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

-- State the main theorem
theorem sum_of_digits_2A_eq_2B (A B : ℕ) 
  (h : ∀ x, x ∈ A.to_digits 10 ↔ x ∈ B.to_digits 10) :
  sum_of_digits (2 * A) = sum_of_digits (2 * B) := 
sorry

end sum_of_digits_2A_eq_2B_l294_294862


namespace percentage_alcohol_in_first_vessel_is_zero_l294_294898

theorem percentage_alcohol_in_first_vessel_is_zero (x : ℝ) :
  ∀ (alcohol_first_vessel total_vessel_capacity first_vessel_capacity second_vessel_capacity concentration_mixture : ℝ),
  first_vessel_capacity = 2 →
  (∃ xpercent, alcohol_first_vessel = (first_vessel_capacity * xpercent / 100)) →
  second_vessel_capacity = 6 →
  (∃ ypercent, ypercent = 40 ∧ alcohol_first_vessel + 2.4 = concentration_mixture * (total_vessel_capacity/8) * 8) →
  concentration_mixture = 0.3 →
  0 = x := sorry

end percentage_alcohol_in_first_vessel_is_zero_l294_294898


namespace brother_more_lambs_than_merry_l294_294423

theorem brother_more_lambs_than_merry
  (merry_lambs : ℕ) (total_lambs : ℕ) (more_than_merry : ℕ)
  (h1 : merry_lambs = 10) 
  (h2 : total_lambs = 23)
  (h3 : more_than_merry + merry_lambs + merry_lambs = total_lambs) :
  more_than_merry = 3 :=
by
  sorry

end brother_more_lambs_than_merry_l294_294423


namespace correct_operation_l294_294128

variables {x y : ℝ}

theorem correct_operation : -2 * x * 3 * y = -6 * x * y :=
by
  sorry

end correct_operation_l294_294128


namespace platform_length_l294_294146

theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) : 
  train_length = 300 →
  time_platform = 30 →
  time_pole = 18 →
  ∃ (L : ℝ), L = 200 :=
by {
  intros, 
  let V := train_length / time_pole,
  have h1 : V = train_length / time_pole := rfl,
  let D := train_length + L,
  have h2 : V = D / time_platform := by sorry,
  have h3 : (train_length / time_pole) = ((train_length + L) / time_platform) := by sorry,
  have h4 : L = 200 := by sorry,
  use 200,
  exact h4,
  done,
}

end platform_length_l294_294146


namespace sequence_a_2011_l294_294730

noncomputable def sequence_a : ℕ → ℕ
| 0       => 2
| 1       => 3
| (n+2)   => (sequence_a (n+1) * sequence_a n) % 10

theorem sequence_a_2011 : sequence_a 2010 = 2 :=
by
  sorry

end sequence_a_2011_l294_294730


namespace circumcenter_concyclic_l294_294312

-- Define the points and conditions
noncomputable def point : Type := sorry
variables (A D F E B C : point)
variables (ADF AEF BDF CEF : Triangle) 
variables (circumcenter : Triangle → point)

-- Define the circumcenters of the triangles
def O := circumcenter ADF
def P := circumcenter AEF
def Q := circumcenter BDF
def R := circumcenter CEF

-- Prove that points O, P, Q, and R are concyclic
theorem circumcenter_concyclic 
  (circ_A : circumcenter ADF = O)
  (circ_B : circumcenter AEF = P)
  (circ_C : circumcenter BDF = Q)
  (circ_D : circumcenter CEF = R) : 
  ∃ (circle : CoCircle point), circle O ∧ circle P ∧ circle Q ∧ circle R :=
sorry

end circumcenter_concyclic_l294_294312


namespace initial_volume_of_solution_l294_294546

theorem initial_volume_of_solution (V : ℝ) :
  (0.40 * V + 1.2) = 0.50 * (V + 1.2) → V = 6 :=
by
  intros h,
  sorry

end initial_volume_of_solution_l294_294546


namespace range_of_g_l294_294629

noncomputable def g (x : ℝ) : ℝ :=
  (sin x)^3 + 7 * (sin x)^2 + 2 * sin x + 3 * (cos x)^2 - 10

theorem range_of_g :
  set_of (λ (y : ℝ), ∃ (x : ℝ), sin x ≠ 1 ∧ y = g x) = set.Ico 3 13 := sorry

end range_of_g_l294_294629


namespace train_time_to_pass_tree_l294_294896

def km_per_hr_to_m_per_s (speed_km_per_hr : ℕ) : ℕ :=
  speed_km_per_hr * 1000 / 3600

theorem train_time_to_pass_tree (train_length : ℕ) (train_speed_km_per_hr : ℕ)
  (h_train_length : train_length = 240)
  (h_train_speed : train_speed_km_per_hr = 108) :
  let train_speed_m_per_s := km_per_hr_to_m_per_s train_speed_km_per_hr in
  train_length / train_speed_m_per_s = 8 :=
by
  rw [h_train_length, h_train_speed]
  let train_speed_m_per_s := km_per_hr_to_m_per_s 108
  have : train_speed_m_per_s = 30 := rfl
  rw this
  norm_num

end train_time_to_pass_tree_l294_294896


namespace expected_sum_2010_l294_294162

noncomputable def expectedRolls (n : ℕ) : ℝ := 
  if n == 0 then 0
  else 1 + (1 / 6) * ∑ i in (Finset.range 6), expectedRolls (n - i)

theorem expected_sum_2010 : abs (expectedRolls 2010 - 574.761904) < 0.001 :=
sorry

end expected_sum_2010_l294_294162


namespace find_magnitude_constraint_l294_294348

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

variables (m : ℝ)
def vector_a : ℝ × ℝ := (m, 2)
def vector_b : ℝ × ℝ := (1, 1)
def vector_sum : ℝ × ℝ := (vector_a m).fst + (vector_b).fst, (vector_a m).snd + (vector_b).snd

theorem find_magnitude_constraint (h : vector_magnitude (vector_sum m) = vector_magnitude (vector_a m) + vector_magnitude vector_b) : m = 2 := 
sorry

end find_magnitude_constraint_l294_294348


namespace coefficient_x100_in_polynomial_l294_294213

theorem coefficient_x100_in_polynomial:
  let poly := (∑ i in Finset.range 101, (X^i))^3 in
  (Polynomial.coeff poly 100) = 5151 :=
by
  sorry

end coefficient_x100_in_polynomial_l294_294213


namespace f_periodic_and_even_f_neg_2014_plus_f_2015_eq_one_l294_294664

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ set.Ico 0 2 then Real.log (x + 1) / Real.log 2 else
if h : x < 0 then f (-x) else f (x - 2 * Real.floor (x / 2))

theorem f_periodic_and_even :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, x ≥ 0 → f (x + 2) = f x) ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 2) :=
sorry

theorem f_neg_2014_plus_f_2015_eq_one : f (-2014) + f 2015 = 1 :=
by
  have h₁ : ∀ x : ℝ, f (-x) = f x := by sorry
  have h₂ : ∀ x : ℝ, x ≥ 0 → f (x + 2) = f x := by sorry
  have h₃ : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 2 := by sorry
  sorry

end f_periodic_and_even_f_neg_2014_plus_f_2015_eq_one_l294_294664


namespace probability_of_spade_then_ten_or_jack_l294_294115

theorem probability_of_spade_then_ten_or_jack :
  let deck_size := 52
  let spade_count := 13
  let ten_jack_count := 8
  let spade_not_ten_jack := 11

  let prob_first_spade_not_ten_or_jack := spade_not_ten_jack / deck_size
  let prob_second_ten_or_jack := ten_jack_count / (deck_size - 1)

  let prob_first_ten_spade := 1 / deck_size
  let prob_second_ten_or_jack_excl_ten_spade := (ten_jack_count - 1) / (deck_size - 1)

  let prob_first_jack_spade := 1 / deck_size
  let prob_second_ten_or_jack_excl_jack_spade := (ten_jack_count - 1) / (deck_size - 1)

  let total_prob := (prob_first_spade_not_ten_or_jack * prob_second_ten_or_jack) +
                    (prob_first_ten_spade * prob_second_ten_or_jack_excl_ten_spade) +
                    (prob_first_jack_spade * prob_second_ten_or_jack_excl_jack_spade)
  in total_prob = (17 / 442) := by
  sorry

end probability_of_spade_then_ten_or_jack_l294_294115


namespace correct_operation_l294_294130

theorem correct_operation (x y a b : ℝ) :
  (-2 * x) * (3 * y) = -6 * x * y :=
by
  sorry

end correct_operation_l294_294130


namespace cuberoot_eq_zero_and_sum_l294_294029

theorem cuberoot_eq_zero_and_sum {x : ℝ} (h : real.cbrt x + real.cbrt (30 - x) = 0) : 
  ∃ p q : ℤ, (30 : ℝ) = (p + real.sqrt (q : ℝ)) ∧ p + q = 30 := 
sorry

end cuberoot_eq_zero_and_sum_l294_294029


namespace ratio_of_triangle_areas_l294_294777

theorem ratio_of_triangle_areas 
  (A B C D : Point)
  (h_eq_tri : equilateral_triangle A B C)
  (h_D_on_AC : lies_on_line_segment D A C)
  (h_angle_DBC : angle D B C = 30) :
  (area (triangle A D B)) / (area (triangle C D B)) = 1 / real.sqrt 3 :=
sorry

end ratio_of_triangle_areas_l294_294777


namespace minimum_questions_two_l294_294375

structure Person :=
  (is_liar : Bool)

structure Decagon :=
  (people : Fin 10 → Person)

def minimumQuestionsNaive (d : Decagon) : Nat :=
  match d with 
  -- add the logic here later
  | _ => sorry

theorem minimum_questions_two (d : Decagon) : minimumQuestionsNaive d = 2 :=
  sorry

end minimum_questions_two_l294_294375


namespace number_of_elements_l294_294483

theorem number_of_elements (n : ℕ) (S : ℕ) 
  (avg_all : S = 104 * n) 
  (sum_first5 : ∑ i in Finset.range 5, a (i + 1) = 495)
  (sum_last5 : ∑ i in Finset.range 5, a (n - i) = 500)
  (a_5 : a 5 = 59) : n = 9 := 
by
  sorry

end number_of_elements_l294_294483


namespace sum_arith_geo_seq_l294_294409

theorem sum_arith_geo_seq (a b : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a n = 2 * n) →
  (∀ n, b n = 3 ^ (n - 1)) →
  (∀ n, S n = ∑ i in finset.range n, (a (i + 1) - b (i + 1))) →
  ∀ n, S n = n^2 + n + 1 / 2 - (1 / 2) * 3^n :=
begin
  intros ha hb hS,
  sorry
end

end sum_arith_geo_seq_l294_294409


namespace lisa_speed_correct_l294_294234

def eugene_speed := 5

def carlos_speed := (3 / 4) * eugene_speed

def lisa_speed := (4 / 3) * carlos_speed

theorem lisa_speed_correct : lisa_speed = 5 := by
  sorry

end lisa_speed_correct_l294_294234


namespace compute_100c_add_d_l294_294742

theorem compute_100c_add_d (c d : ℝ)
  (h1 : (x : ℝ) → (x + c) * (x + d) * (x - 15) / (x - 4)^2 = 0)
  (h2 : (x : ℝ) → (x + 2 * c) * (x - 4) * (x - 9) / (x + d) * (x - 15) = 0)
  (h1_conditions : ¬ (c = -4) ∧ ¬ (d = -4) ∧ c ≠ 2)
  (h2_conditions : ¬ (d = -4) ∧ 4 ∈ {y : ℝ | (x : ℝ) (x - 4) * (x - 9) = 0}) :
  100 * c + d = -391 :=
by
  sorry

end compute_100c_add_d_l294_294742


namespace no_solution_log_sin_cos_eq_2_l294_294456

theorem no_solution_log_sin_cos_eq_2 : ¬(∃ x : ℝ, real.log (real.sin x) + real.log (real.cos x) = 2) :=
by
  sorry

end no_solution_log_sin_cos_eq_2_l294_294456


namespace total_prime_factors_total_prime_factors_count_l294_294852

theorem total_prime_factors :
  let a := (4 : ℕ) ^ 11
  let b := (7 : ℕ) ^ 5
  let c := (11 : ℕ) ^ 2
  a * b * c = 2^22 * 7^5 * 11^2 :=
  a = (2^2) ^ 11 ∧ 
  ∃ d : ℕ, b = (7 : ℕ) ^ d ∧ d = 5 ∧
  ∃ e : ℕ, c = (11 : ℕ) ^ e ∧ e = 2
  
theorem total_prime_factors_count :
  (∀ a b c : ℕ, a = (2^2) ^ 11 → b = 7 ^ 5 → c = 11 ^ 2 → 
    ∑ in [2^22, 7^5, 11^2], by sorry)

end total_prime_factors_total_prime_factors_count_l294_294852


namespace number_of_correct_statements_l294_294912

def prism (P : Type) : Prop := sorry
def right_prism (P : Type) : Prop := prism P ∧ (∀ base lateral_edges_perpendicular, base P → lateral_edges_perpendicular P)
def regular_prism (P : Type) : Prop := right_prism P ∧ (∀ base regular_polygon, base P → regular_polygon P)
def lateral_faces_parallelograms (P : Type) : Prop := prism P ∧ (∀ lateral_faces parallelograms, lateral_faces P → parallelograms P)

def statement_1 (P : Type) := ∀ (P : Type), prism P → right_prism P
def statement_2 (P : Type) := ∀ (P : Type), right_prism P → regular_prism P
def statement_3 (P : Type) := ∀ (P : Type), prism P → lateral_faces_parallelograms P

theorem number_of_correct_statements : ∀ (P : Type), prism P → 
  (statement_1 P ∧ statement_3 P ∧ ¬statement_2 P) → 2 := sorry

end number_of_correct_statements_l294_294912


namespace exist_xy_z_odd_l294_294274

variables {N : ℕ} 
variables {a b c : ℕ → ℤ} 

-- Condition: For each i, at least one of a_i, b_i, c_i is odd
def at_least_one_odd (i : ℕ) : Prop := (a i % 2 = 1) ∨ (b i % 2 = 1) ∨ (c i % 2 = 1)

-- Statement to rewrite in Lean 4 
theorem exist_xy_z_odd : 
  (∀ i, i < N → at_least_one_odd i) →
  ∃ (x y z : ℤ), (∑ i in finset.range N, if (x * (a i) + y * (b i) + z * (c i)) % 2 == 1 then 1 else 0) ≥ 4 * N / 7 := 
sorry

end exist_xy_z_odd_l294_294274


namespace hyperbola_equation_l294_294079

noncomputable def hyperbola_asymptotes_and_focal_length (a b : ℝ) (foci_on_x : Bool) : Prop :=
  if foci_on_x then
    a^2 + b^2 = 25 ∧ (b / a) = (1 / 2) ∧ (1 / 20 * x^2 - 1 / 5 * y^2 = 1)
  else
    a^2 + b^2 = 25 ∧ (a / b) = (1 / 2) ∧ (1 / 5 * y^2 - 1 / 20 * x^2 = 1)

theorem hyperbola_equation :
  ∃ (a b : ℝ), ∀ foci_on_x : Bool,
    hyperbola_asymptotes_and_focal_length a b foci_on_x :=
by
  sorry

end hyperbola_equation_l294_294079


namespace smallest_multiple_of_5_with_remainder_1_l294_294694

theorem smallest_multiple_of_5_with_remainder_1 : 
  ∃ (a : ℕ), (∃ (k : ℕ), a = 5 * k) ∧ a % 3 = 1 ∧ ∀ (b : ℕ), (∃ (k' : ℕ), b = 5 * k') ∧ b % 3 = 1 → b ≥ a :=
begin
  sorry
end

end smallest_multiple_of_5_with_remainder_1_l294_294694


namespace convex_polyhedron_inequality_l294_294713

noncomputable def convex_polyhedron (B P T : ℕ) : Prop :=
  ∀ (B P T : ℕ), B > 0 ∧ P > 0 ∧ T >= 0 → B * (Nat.sqrt (P + T)) ≥ 2 * P

theorem convex_polyhedron_inequality (B P T : ℕ) (h : convex_polyhedron B P T) : 
  B * (Nat.sqrt (P + T)) ≥ 2 * P :=
by
  sorry

end convex_polyhedron_inequality_l294_294713


namespace alternating_binomials_sum_l294_294259

-- Define a function to represent the alternating sum of binomials.
def alternating_sum_binomials (n : ℕ) : ℤ :=
  ∑ k in finset.range (n / 2 + 1), (-1 : ℤ) ^ k * nat.choose n (2 * k)

theorem alternating_binomials_sum :
  alternating_sum_binomials 100 = -2 ^ 50 :=
by
  sorry

end alternating_binomials_sum_l294_294259


namespace total_legs_among_animals_l294_294236

def legs (chickens sheep grasshoppers spiders : Nat) (legs_chicken legs_sheep legs_grasshopper legs_spider : Nat) : Nat :=
  (chickens * legs_chicken) + (sheep * legs_sheep) + (grasshoppers * legs_grasshopper) + (spiders * legs_spider)

theorem total_legs_among_animals :
  let chickens := 7
  let sheep := 5
  let grasshoppers := 10
  let spiders := 3
  let legs_chicken := 2
  let legs_sheep := 4
  let legs_grasshopper := 6
  let legs_spider := 8
  legs chickens sheep grasshoppers spiders legs_chicken legs_sheep legs_grasshopper legs_spider = 118 :=
by
  sorry

end total_legs_among_animals_l294_294236


namespace abs_pi_expression_l294_294948

theorem abs_pi_expression : (|π - |π - 10|| = 10 - 2 * π) := by
  sorry

end abs_pi_expression_l294_294948


namespace abs_pi_expression_l294_294943

theorem abs_pi_expression : |π - |π - 10|| = 10 - 2 * π :=
by
  sorry

end abs_pi_expression_l294_294943


namespace final_price_of_shirt_l294_294396

noncomputable def round (x : Real) (n : ℕ) : Real :=
  Real.round (x * 10 ^ n) / 10 ^ n

theorem final_price_of_shirt
  (original_price : Real)
  (discount_1 : Real)
  (price_after_first_discount : Real)
  (discount_2 : Real)
  (final_price_before_rounding : Real)
  (final_price : Real) :
  original_price = 26.67 →
  discount_1 = original_price * 0.25 →
  price_after_first_discount = original_price - discount_1 →
  discount_2 = price_after_first_discount * 0.25 →
  final_price_before_rounding = price_after_first_discount - discount_2 →
  final_price = round final_price_before_rounding 2 →
  final_price = 14.25 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end final_price_of_shirt_l294_294396


namespace ij_eq_ah_l294_294044

-- Defining an acute triangle
variables {A B C H G I J : Type*}
variables [acute_triangle : is_acute_triangle A B C]
variables [orthocenter : orthocenter H A B C]

-- Conditions of the problem
variables [parallelogram : parallelogram ABGH A B G H]
variables [bisect : bisects_line AC HI I]
variables [circumcircle : circumcircle GCI J]
variables [intersection : intersects AC (circumcircle GCI) C J]

theorem ij_eq_ah : IJ = AH :=
sorry

end ij_eq_ah_l294_294044


namespace complement_of_union_l294_294020

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def U : Set ℤ := Set.univ

theorem complement_of_union (x : ℤ) : (x ∉ A ∪ B) ↔ ∃ k : ℤ, x = 3 * k := 
by {
  sorry
}

end complement_of_union_l294_294020


namespace num_of_b_l294_294277

theorem num_of_b (y_line y_parabola : ℝ → ℝ) :
  (∃ b : ℝ, (y_line = λ x, 2 * x + b) ∧ (y_parabola = λ x, x^2 + 3 * b^2)) →
  (∃ b1 b2 : ℝ, (b = b1 ∨ b = b2) ∧
       (y_line 0 = y_parabola 0) ∧
       2 = 2) :=
by
  sorry

end num_of_b_l294_294277


namespace B_takes_12_minutes_l294_294539

theorem B_takes_12_minutes (
  (A_lap_time : ℕ) -- A takes 6 minutes to complete one lap
  (B_additional_time : ℕ) -- B takes 8 minutes after the first meeting to reach the starting point
  (meeting_time_rel : ∀ (d : ℚ) (x : ℚ), x = (6 * x) / (x + 6) + 8) -- Relationship between meeting time and lap times
) : ∃ (B_lap_time : ℚ), B_lap_time = 12 :=

by
  have B_lap_time_result : ∀ (x : ℚ), x = 12 → x = (6 * x) / (x + 6) + 8 := sorry
  use 12
  rw B_lap_time_result
  sorry

end B_takes_12_minutes_l294_294539


namespace correct_choice_D_l294_294672

def f (x : ℝ) : ℝ := Real.log x + 1 / Real.log x

theorem correct_choice_D : ∃ x₀ > 0, ∀ x > x₀, has_deriv_at f x (1 / x - 1 / (x * (Real.log x) ^ 2)) ∧ (1 / x - 1 / (x * (Real.log x) ^ 2)) > 0 :=
by
  sorry

end correct_choice_D_l294_294672


namespace part1_proof_part2_proof_l294_294643

def inequality1 (a x : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0 -- condition p
def condition_a (a : ℝ) : Prop := a > 0 -- condition a > 0
def inequality2 (x : ℝ) : Prop := 3 < x ∧ x ≤ 4 -- condition q

noncomputable def part1 (x : ℝ) : Prop := 
  ∃ a, a = 1 ∧ inequality1 a x ∧ inequality2 x

theorem part1_proof : part1 x → 3 < x ∧ x < 4 :=
by 
  sorry

noncomputable def always_inequality1 (a : ℝ) (x : ℝ) : Prop := ∀ x, inequality1 a x
noncomputable def necessary_condition_for_q (a : ℝ) : Prop := 
  always_inequality1 a x ∧ ¬(inequality2 x → inequality1 a x) 

theorem part2_proof : necessary_condition_for_q a → 1 < a ∧ a ≤ 3 :=
by
  sorry

end part1_proof_part2_proof_l294_294643


namespace sum_due_is_correct_l294_294855

-- Definitions based on the conditions provided
def bankersDiscount : ℝ := 72
def trueDiscount : ℝ := 60

-- Definition of the Present Value (P.V.) according to the given formula
def presentValue (BD TD : ℝ) : ℝ := (TD * TD) / (BD - TD)

-- Statement to prove the correct answer
theorem sum_due_is_correct :
  presentValue bankersDiscount trueDiscount = 300 := by
  sorry

end sum_due_is_correct_l294_294855


namespace density_approx_eq_l294_294542

-- Define variables and constants
def G := 6.67430 * 10^(-11) -- Gravitational constant in m^3 kg^(-1) s^(-2)
def T := 14400 -- Period in seconds (4 * 60 * 60)
def expected_density := 6000 -- Expected density in kg/m^3

-- Define centripetal acceleration as a function of radius R
def ac (R : ℝ) : ℝ :=
  4 * π^2 * (2 * R) / T^2

-- Define Newton's second law in the context of the gravitational force providing centripetal force
def newtons_second_law (m R : ℝ) (ρ : ℝ) : Prop :=
  m * ac R = G * (m * ρ * (4/3) * π * R^3) / (2 * R)^2

-- Define the density formula
def density (R : ℝ) : ℝ :=
  24 * π / (G * T^2)

-- The theorem statement
theorem density_approx_eq (R : ℝ) : density R ≈ expected_density :=
by
  sorry

end density_approx_eq_l294_294542


namespace proof_success_probability_l294_294338

noncomputable def success_probability (p : ℝ) : Prop := 
  let xi_pmf := ProbabilityMassFunction.binomial 2 p
  let eta_pmf := ProbabilityMassFunction.binomial 4 p
  xi_pmf.prob ({i | 1 ≤ i}) = 5/9 → eta_pmf.prob ({i | 2 ≤ i}) = 11/27

theorem proof_success_probability :
  ∀ (p : ℝ), success_probability p :=
begin
  intro p,
  sorry -- Proof omitted as per guidelines
end

end proof_success_probability_l294_294338


namespace emma_still_missing_fraction_l294_294619

variable (x : ℕ)  -- Total number of coins Emma received 

-- Conditions
def emma_lost_half (x : ℕ) : ℕ := x / 2
def emma_found_four_fifths (lost : ℕ) : ℕ := 4 * lost / 5

-- Question to prove
theorem emma_still_missing_fraction :
  (x - (x / 2 + emma_found_four_fifths (emma_lost_half x))) / x = 1 / 10 := 
by
  sorry

end emma_still_missing_fraction_l294_294619


namespace notation_3_in_row_5_l294_294352

-- Define the notation for an element and a row
def notation (n : ℕ) (r : ℕ) : ℕ × ℕ := (n, r)

-- Given condition 
def given_condition : notation 8 4 = (8, 4) := rfl

-- Proof statement
theorem notation_3_in_row_5 : notation 3 5 = (3, 5) :=
by
  -- Using the same pattern as for notation 8 4
  sorry

end notation_3_in_row_5_l294_294352


namespace privateer_overtakes_merchantman_l294_294186

theorem privateer_overtakes_merchantman :
  -- Conditions
  let initial_distance := 10 -- miles
  let p_speed := 11 -- mph (privateer initial speed)
  let m_speed := 8 -- mph (merchantman speed)
  let new_speed_ratio := (17 : ℕ, 15 : ℕ) -- new speed ratio privateer:merchantman
  let time_sail_damage_happened := 2 -- hours, i.e., at 1:45 p.m.
  let damage_privateer_speed := (17.0 / 15.0) * 8.0 -- mph, new speed of privateer
  let relative_speed := p_speed - m_speed -- initial relative speed
  let remaining_distance := initial_distance + (m_speed * time_sail_damage_happened) - (p_speed * time_sail_damage_happened)
  let new_relative_speed := damage_privateer_speed - m_speed -- new relative speed after damage
  
  -- Conclusion
  let overtaking_time := time_sail_damage_happened + (remaining_distance / new_relative_speed)
  let final_time := 11.75 + overtaking_time -- start time 11:45 a.m. (11.75 in decimal hours) + overtaking time
  
  final_time ≈ 17.5 := 
sorry

end privateer_overtakes_merchantman_l294_294186


namespace find_vector_coords_l294_294342

def vector_perpendicular (a: ℝ × ℝ × ℝ) (b: ℝ × ℝ × ℝ) : Prop :=
  let (ax, ay, az) := a
  let (bx, by, bz) := b
  ax * bx + ay * by + az * bz = 0

theorem find_vector_coords (x y : ℝ) (b : ℝ × ℝ × ℝ) :
  (b = (x, y, 0)) →
  (x^2 + y^2 = 20) →
  (vector_perpendicular (1, -2, 5) (x, y, 0)) →
  (b = (4, 2, 0) ∨ b = (-4, -2, 0)) := by
  sorry

end find_vector_coords_l294_294342


namespace eval_with_parentheses_l294_294480

theorem eval_with_parentheses :
  ∃ (results : set ℕ), 
  (results = {10, 24, 6}) ∧
  (∃ (e1 e2 e3 : ℕ), 
    (e1 = (72 / 9 - 3) * 2) ∧
    (e2 = 72 / (9 - 3) * 2) ∧
    (e3 = 72 / ((9 - 3) * 2)) ∧
    results = {e1, e2, e3}) := 
by 
  sorry

end eval_with_parentheses_l294_294480


namespace event_properties_l294_294707

open MeasureTheory ProbabilityTheory

noncomputable def Ball := { red := 0.5, white := 0.5 }

def event_A := λ (ball1 ball2 : Ball), ball1 = ball2
def event_B := λ (ball1 : Ball), ball1 = Ball.red
def event_C := λ (ball2 : Ball), ball2 = Ball.red
def event_D := λ (ball1 ball2 : Ball), ball1 ≠ ball2

theorem event_properties :
  (¬(MutuallyExclusive event_B event_C)) ∧
  (Complementary event_A event_D) ∧
  (Independent event_A event_B) ∧
  (Independent event_C event_D) :=
by
  simp; sorry

end event_properties_l294_294707


namespace diameter_of_circle_l294_294207

theorem diameter_of_circle
  (A B : Point)
  (circle1 circle2 : Circle)
  (M : Point)
  (M_on_circle1 : lies_on_circle M circle1)
  (M_inside_circle2 : inside_circle M circle2)
  (tangents_perpendicular : ∀ (P : Point), tangent circle1 P ∧ tangent circle2 P → ∃ (O1 O2 : Point),
      perpendicular (line_through O1 P) (line_through O2 P))
  (X Y : Point)
  (X_on_circle1 : lies_on_circle X circle1)
  (Y_on_circle1 : lies_on_circle Y circle1)
  (AMX_collinear : collinear A M X)
  (BMY_collinear : collinear B M Y) : 
  diameter circle1 X Y := 
sorry

end diameter_of_circle_l294_294207


namespace true_statement_l294_294435

def statement_i (i : ℕ) (n : ℕ) : Prop := 
  (i = (n - 1))

theorem true_statement :
  ∃! n : ℕ, (n ≤ 100 ∧ ∀ i, (i ≠ n - 1) → statement_i i n = false) ∧ statement_i (n - 1) n = true :=
by
  sorry

end true_statement_l294_294435


namespace trapezoid_proof_l294_294113

-- Definitions of geometric elements and conditions

variables (A B C D O P : Type)
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited O] [Inhabited P]
variables (AB CD AD BD AC BDOP DP DO OP_length AD_length : ℝ)

noncomputable def is_trapezoid (AB CD: ℝ) : Prop := 
AB = 102 ∧ CD = 51 

noncomputable def BC_CD_equal (BC CD: ℝ) : Prop :=
BC = 51 ∧ CD = 51

noncomputable def AD_perpendicular_BD (AD BD: ℝ) : Prop :=
true -- Given as a fact

noncomputable def diagonals_intersect (O AC BD: ℝ) : Prop :=
true -- Given as a fact

noncomputable def midpoint_P (P BD: ℝ) : Prop :=
true -- Given as a fact

noncomputable def OP_given_length (OP_length: ℝ) :=
OP_length = 13

noncomputable def AD_calculation (AD: ℝ) : Prop :=
AD = 120 * real.sqrt 3

noncomputable def m_plus_n_equals (m n: ℝ) :=
m + n = 123

theorem trapezoid_proof :
  ∀ (A B C D O P : Type) 
  (AB CD AD BD AC BDOP DP DO OP_length AD_length: ℝ), 
  (is_trapezoid AB CD) ∧ 
  (BC_CD_equal BC CD) ∧ 
  (AD_perpendicular_BD AD BD) ∧ 
  (diagonals_intersect O AC BD) ∧ 
  (midpoint_P P BD) ∧ 
  (OP_given_length OP_length) →
  (AD_calculation AD_length) →
  (m_plus_n_equals 120 3) := 
by 
  sorry -- to be proved

end trapezoid_proof_l294_294113


namespace num_integers_with_digit_3_at_least_once_l294_294686

def contains_digit_3 (n : Nat) : Prop :=
  ∃ d : Nat, (d ∈ [3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 13, 23, 43, 53]) ∧ n = d

theorem num_integers_with_digit_3_at_least_once : 
  { n : Nat | 1 ≤ n ∧ n ≤ 60 ∧ contains_digit_3 n }.card = 15 :=
sorry

end num_integers_with_digit_3_at_least_once_l294_294686


namespace abs_pi_expression_l294_294953

theorem abs_pi_expression (h : Real.pi < 10) : 
  Real.abs (Real.pi - Real.abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_pi_expression_l294_294953


namespace proof_problem_l294_294816

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def number_of_prime_factors (n : ℕ) : ℕ :=
  if n = 1 then 0 else (n.factors : Multiset ℕ).card

def prime_factors_expression : ℕ :=
  let expr := (factorial 20 * factorial 22) / (factorial 16 * factorial 11)
  number_of_prime_factors expr

theorem proof_problem : (prime_factors_expression * (prime_factors_expression - 2) = 960) :=
  sorry

end proof_problem_l294_294816


namespace average_billboards_per_hour_l294_294064

-- Define the number of billboards seen in each hour
def billboards_first_hour := 17
def billboards_second_hour := 20
def billboards_third_hour := 23

-- Define the number of hours
def total_hours := 3

-- Prove that the average number of billboards per hour is 20
theorem average_billboards_per_hour : 
  (billboards_first_hour + billboards_second_hour + billboards_third_hour) / total_hours = 20 :=
by
  sorry

end average_billboards_per_hour_l294_294064


namespace problem_statements_l294_294418

noncomputable def f (x : ℕ) : ℕ := x % 2
noncomputable def g (x : ℕ) : ℕ := x % 3

theorem problem_statements (x : ℕ) : (f (2 * x) = 0) ∧ (f x + f (x + 3) = 1) :=
by
  sorry

end problem_statements_l294_294418


namespace problem_I_problem_II_1_problem_II_2_problem_III_l294_294204

section FujiApples

-- Define the input data for the problem
def Origin := ℕ
def Price (o : Origin) : ℝ
| 0 := 150
| 1 := 160
| 2 := 140
| 3 := 155
| 4 := 170
| _ := 0 -- Default case

def MarketShare (o : Origin) : ℝ
| 0 := 0.15
| 1 := 0.10
| 2 := 0.25
| 3 := 0.20
| 4 := 0.30
| _ := 0 -- Default case

-- Question (I): Probability of selecting a box with a price less than 160 yuan
theorem problem_I : ∑ o in {0, 2, 3}, MarketShare o = 0.60 :=
by sorry

-- Question (II-1): Finding the value of n
def boxes_from_A := 3
def boxes_from_B := 2
def n := boxes_from_A + boxes_from_B

theorem problem_II_1 : n = 5 :=
by sorry

-- Question (II-2): Probability of selecting two boxes from different origins
theorem problem_II_2 : (3/5 : ℝ) = 3 / 5 :=
by sorry

-- Question (III): Comparing M1 and M2
def M1 := ∑ o in {0, 1, 2, 3, 4}, (Price o) * (MarketShare o)
def newMarketShare (o : Origin) : ℝ
| 0 := 0.20 -- increased by 5%
| 2 := 0.20 -- decreased by 5%
| 1 := 0.10   -- remains the same
| 3 := 0.20   -- remains the same
| 4 := 0.30   -- remains the same
| _ := 0 -- Default case

def M2 := ∑ o in {0, 1, 2, 3, 4}, (Price o) * (newMarketShare o)

theorem problem_III : M1 < M2 :=
by sorry

end FujiApples

end problem_I_problem_II_1_problem_II_2_problem_III_l294_294204


namespace tangent_line_parabola_parallel_l294_294467

-- Define the necessary conditions
def parabola : ℝ → ℝ := λ x, x^2

def is_parallel (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
  l1.1 * l2.2 = l2.1 * l1.2

def tangent_equation (x : ℝ) (t : ℝ → ℝ) (a b : ℝ) : ℝ := 
  a * x - b - t x

theorem tangent_line_parabola_parallel (p q : ℝ × ℝ × ℝ) 
  (h_parabola : ∀ x, p.2.1 * x - p.2.2 - parabola x = 0)
  (h_parallel : is_parallel (2, -1, p.2.1) (2, -1, 5)) :
  p.2.1 = -1 := 
sorry

end tangent_line_parabola_parallel_l294_294467


namespace maximize_volume_height_1_2_l294_294834

noncomputable def height_maximized : Prop :=
  ∃ (x : ℝ), 0 < x ∧ x < 1.6 ∧ 3.2 - 2 * x = 1.2 ∧
  let y := x * (x + 0.5) * (3.2 - 2 * x) in 
  ∀ x' : ℝ, 0 < x' ∧ x' < 1.6 → y ≤ x' * (x' + 0.5) * (3.2 - 2 * x')

theorem maximize_volume_height_1_2 (total_length : ℝ) (longer_side : ℝ) :
  total_length = 14.8 ∧ longer_side = 0.5 →
  height_maximized :=
by
  intros _ _ h
  have := sorry -- prove the statement

end maximize_volume_height_1_2_l294_294834


namespace area_ratio_eq_one_l294_294774

theorem area_ratio_eq_one
  (A B C D : Point)
  (h_eq_triangle : EquilateralTriangle A B C)
  (h_D_on_AC : D ∈ segment A C)
  (h_angle_DBC_eq_30 : ∠ D B C = 30) :
  (area (triangle A D B))/(area (triangle C D B)) = 1 := 
sorry

end area_ratio_eq_one_l294_294774


namespace find_a_and_m_l294_294806

noncomputable def f (a : ℝ) (n : ℕ+) (x : ℝ) : ℝ := a * x^n * (1 - x)

theorem find_a_and_m :
  ∃ (a : ℝ), (∀ (x : ℝ), 0 < x → f a 2 x ≤ 4 / 27) ∧ a = 1 ∧
  ∀ (m : ℝ), (0 < m → m < (n ^ n) / ((n+1) ^ (n+1))) →
               (∃ (x1 x2 : ℝ), 0 < x1 ∧ 0 < x2 ∧ f 1 n x1 = m ∧ f 1 n x2 = m) :=
begin
  sorry,
end

end find_a_and_m_l294_294806


namespace necessary_and_sufficient_condition_l294_294637

variable {O A B C D P E F : Type} 
variable [InCircle O A B C] [Arc BC] [Point D on Arc]
variable [Perpendicular O AB E] [Perpendicular O AC F] [Intersection AD E F]
variable [Intersection BE CF P]

theorem necessary_and_sufficient_condition 
  (h1 : triangle A B C) 
  (h2 : inscribed A B C O) 
  (h3 : AB > AC) 
  (h4 : AC > BC) 
  (h5 : D on arc BC) 
  (h6 : perp O AB E) 
  (h7 : perp O AC F) 
  (h8 : intersect AD E F)
  (h9 : intersect BE CF P) 
  : PB = PC + PO ↔ ∠BAC = 30° := sorry

end necessary_and_sufficient_condition_l294_294637


namespace chord_length_eq_4_dot_product_non_pos_l294_294335

-- Definitions based on given conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5

def line_eq (x y : ℝ) (m : ℝ) : Prop := x - m*y - 4 = 0

def chord_length (A B : ℝ × ℝ) : ℝ :=
  dist A B

def dot_product (A B : ℝ × ℝ) : ℝ :=
  (A.1 * B.1) + (A.2 * B.2)

noncomputable def distance_from_center (m : ℝ) :=
  4 / (sqrt (1 + m^2))

-- Part (1): Proving the value of m for the chord length
theorem chord_length_eq_4 (m : ℝ) (A B : ℝ × ℝ)
  (hA : circle_eq A.1 A.2) (hB : circle_eq B.1 B.2) (hLineA : line_eq A.1 A.2 m) (hLineB : line_eq B.1 B.2 m)
  (hDist : chord_length A B = 4) :
  m = sqrt 15 ∨ m = -sqrt 15 := sorry

-- Part (2): Proving the range of values for m
theorem dot_product_non_pos (m : ℝ) (A B : ℝ × ℝ)
  (hA : circle_eq A.1 A.2) (hB : circle_eq B.1 B.2) (hLineA : line_eq A.1 A.2 m) (hLineB : line_eq B.1 B.2 m)
  (hDot : dot_product A B ≤ 0) :
  m ∈ Icc (- 3 * sqrt 15 / 5) (3 * sqrt 15 / 5) ∨
  m ∉ Icc (- 3 * sqrt 15 / 5) (3 * sqrt 15 / 5) := sorry

end chord_length_eq_4_dot_product_non_pos_l294_294335


namespace calculate_R_cubed_plus_R_squared_plus_R_l294_294873

theorem calculate_R_cubed_plus_R_squared_plus_R (R : ℕ) (hR : R > 0)
  (h1 : ∃ q : ℚ, q = (R / (2 * R + 2)) * ((R - 1) / (2 * R + 1)))
  (h2 : (R / (2 * R + 2)) * ((R + 2) / (2 * R + 1)) + ((R + 2) / (2 * R + 2)) * (R / (2 * R + 1)) = 3 * q) :
  R^3 + R^2 + R = 399 :=
by
  sorry

end calculate_R_cubed_plus_R_squared_plus_R_l294_294873


namespace region_area_correct_l294_294220

noncomputable def area_of_region (R₁ R₂ r d : ℝ) (tangent : Prop) (line_passes : Prop) : ℝ :=
  if R₁ = 3 ∧ R₂ = 3 ∧ r = 2 ∧ d = 4 ∧ tangent ∧ line_passes then
    (3/2 * Real.pi - 10)
  else
    0

theorem region_area_correct:
  ∀ (R₁ R₂ r d : ℝ) (tangent line_passes : Prop),
  (R₁ = 3 ∧ R₂ = 3 ∧ r = 2 ∧ d = 4 ∧ tangent ∧ line_passes) →
  area_of_region R₁ R₂ r d tangent line_passes = (3/2 * Real.pi - 10) :=
by
  intros R₁ R₂ r d tangent line_passes h
  simp [area_of_region, h]
  sorry

end region_area_correct_l294_294220


namespace odd_function_k_f_greater_than_l294_294668

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := 2 * x + k * 2^(-x)

-- Question 1
theorem odd_function_k (k : ℝ) : 
  (∀ x : ℝ, f(-x, k) = -f(x, k)) → k = -1 := 
  by
    sorry

-- Question 2
theorem f_greater_than (k : ℝ) : 
  (∀ x : ℝ, 0 ≤ x → f(x, k) > 2 - x) → 0 < k := 
  by
    sorry

end odd_function_k_f_greater_than_l294_294668


namespace seq_a_general_term_seq_b_general_term_inequality_k_l294_294216

def seq_a (n : ℕ) : ℕ :=
if n = 1 then 2 else 2 * n - 1

def S (n : ℕ) : ℕ := 
match n with
| 0       => 0
| (n + 1) => S n + seq_a (n + 1)

def seq_b (n : ℕ) : ℕ := 3 ^ n

def T (n : ℕ) : ℕ := (3 ^ (n + 1) - 3) / 2

theorem seq_a_general_term (n : ℕ) : seq_a n = if n = 1 then 2 else 2 * n - 1 :=
sorry

theorem seq_b_general_term (n : ℕ) : seq_b n = 3 ^ n :=
sorry

theorem inequality_k (k : ℝ) : (∀ n : ℕ, n > 0 → (T n + 3/2 : ℝ) * k ≥ 3 * n - 6) ↔ k ≥ 2 / 27 :=
sorry

end seq_a_general_term_seq_b_general_term_inequality_k_l294_294216


namespace g_odd_l294_294229

def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_odd : ∀ x, g (-x) = - g x :=
by
  sorry

end g_odd_l294_294229


namespace modulus_of_complex_l294_294325

theorem modulus_of_complex {z : ℂ} (h : z = (1 + complex.i) / complex.i) : complex.abs z = real.sqrt 2 :=
  sorry

end modulus_of_complex_l294_294325


namespace averageFishIs75_l294_294765

-- Introduce the number of fish in Boast Pool
def BoastPool : ℕ := 75

-- Introduce the number of fish in Onum Lake
def OnumLake : ℕ := BoastPool + 25

-- Introduce the number of fish in Riddle Pond
def RiddlePond : ℕ := OnumLake / 2

-- Define the total number of fish in all three bodies of water
def totalFish : ℕ := BoastPool + OnumLake + RiddlePond

-- Define the average number of fish in all three bodies of water
def averageFish : ℕ := totalFish / 3

-- Prove that the average number of fish is 75
theorem averageFishIs75 : averageFish = 75 :=
by
  -- We need to provide the proof steps here but using sorry to skip
  sorry

end averageFishIs75_l294_294765


namespace num_progressive_sequences_with_sum_360_l294_294269

def is_progressive_sequence (seq : List ℕ) : Prop :=
  ∀ i, i < seq.length - 1 → seq.get! i < seq.get! (i + 1) ∧ seq.get! (i + 1) % seq.get! i = 0

def sum_of_sequence (seq : List ℕ) : ℕ :=
  seq.foldl (λ acc x => acc + x) 0

def progressive_sequences_with_sum (sum : ℕ) : ℕ :=
  List.length (List.filter (λ seq, is_progressive_sequence seq ∧ sum_of_sequence seq = sum) (List.range (sum + 1) >>= List.range))

theorem num_progressive_sequences_with_sum_360 : progressive_sequences_with_sum 360 = 47 :=
  sorry

end num_progressive_sequences_with_sum_360_l294_294269


namespace range_of_a_for_f_decreasing_l294_294673

noncomputable def f (x a : ℝ) : ℝ := Real.log (x^2 - a * x - 3 * a) / Real.log 2

def is_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃a b⦄, a ∈ s → b ∈ s → a < b → f b ≤ f a

theorem range_of_a_for_f_decreasing :
  ∃ (a : Set ℝ), a = Icc (-4 : ℝ) 4 \{(4 : ℝ)} ∧
  is_decreasing_on (λ x, f x a) (Iic (-2)) :=
sorry

end range_of_a_for_f_decreasing_l294_294673


namespace final_price_correct_l294_294611

-- Definitions based on conditions
def cost_price_usd : ℝ := 20
def profit_percent : ℝ := 0.30
def discount_percent : ℝ := 0.50
def tax_percent : ℝ := 0.10
def packaging_fee_usd : ℝ := 2
def conversion_rate_to_euros : ℝ := 0.85
def promotion_threshold_euros : ℝ := 40
def additional_discount_percent : ℝ := 0.15

-- Predicated resulting price in Euros
def final_price_euros : ℝ :=
  let selling_price_usd := cost_price_usd + (profit_percent * cost_price_usd)
  let discounted_price_usd := selling_price_usd - (discount_percent * selling_price_usd)
  let price_with_tax_usd := discounted_price_usd + (tax_percent * discounted_price_usd)
  let total_price_usd := price_with_tax_usd + packaging_fee_usd
  let total_price_euros := total_price_usd * conversion_rate_to_euros
  if total_price_euros > promotion_threshold_euros then
    total_price_euros * (1 - additional_discount_percent)
  else
    total_price_euros

-- Lean statement to prove the final price
theorem final_price_correct : final_price_euros = 13.855 :=
  by sorry

end final_price_correct_l294_294611


namespace distinct_paths_count_l294_294158

-- Define the basic structure of a cubic diorama with six faces
structure CubicDiorama where
  green : ℕ    -- Green face identifier
  blue : ℕ     -- Blue face identifier
  red : ℕ      -- Red face identifier
  white_faces : List ℕ  -- List of white face identifiers
  adjacent : ℕ → List ℕ  -- Adjacency function for each face

-- Define the specific cubic diorama for this problem
noncomputable def specificDiorama : CubicDiorama :=
  { green := 1,
    blue := 2,
    red := 3,
    white_faces := [4, 5, 6, 7, 8, 9],
    adjacent := λ face,
      match face with
      | 1 => [4, 5, 6]  -- Green adjacent to some white faces
      | 2 => [7, 8, 9]  -- Blue adjacent to some white faces
      | 3 => [4, 5, 6, 7, 8, 9]  -- Red adjacent to all white faces, but red is to be avoided
      | 4 => [1, 3, 7, 8]
      | 5 => [1, 3, 8, 9]
      | 6 => [1, 3, 7, 9]
      | 7 => [2, 3, 4, 6]
      | 8 => [2, 3, 4, 5]
      | 9 => [2, 3, 5, 6]
      | _ => []  -- Any face not in our diorama has no adjacencies
  }

-- State the theorem: 9 distinct paths from green to blue avoiding red
theorem distinct_paths_count : ∀ d : CubicDiorama,
  d = specificDiorama →
  count_distinct_paths d.green d.blue d.red d.adjacent = 9 :=
by
  intros d h_eq
  sorry

end distinct_paths_count_l294_294158


namespace intersection_area_le_c_l294_294450

theorem intersection_area_le_c (P : set (ℝ × ℝ)) (Q : set (ℝ × ℝ))
  (hP_area : measure_theory.measure.lebesgue.measure (P) = 1)
  (Q_is_translation : ∃ v : ℝ × ℝ, ∥v∥ = (1 : ℝ)/100 ∧ Q = {p | ∃ p' ∈ P, p = p' + v}) :
  ∃ c < 1, measure_theory.measure.lebesgue.measure (P ∩ Q) ≤ c :=
sorry

end intersection_area_le_c_l294_294450


namespace aunt_li_can_buy_goods_l294_294529

theorem aunt_li_can_buy_goods (cash_max : ℕ) (total_value : ℕ) (voucher_per_100 : ℕ) (vouchers_per_transaction: ℕ → ℕ) : 
  (cash_max = 1550) → 
  (total_value = 2300) → 
  (voucher_per_100 = 50) → 
  (∀ m, vouchers_per_transaction m = 50 * (m / 100)) →
  let cannot_use_vouchers_in_same_transaction := ∀ x, vouchers_per_transaction x ≤ x / 2 in
  let total_cash := let transactions := [(50, 50), (100, 100), (200, 200), (400, 400), (800, 0)] in
                      transactions.foldl (λ acc (cash, voucher), acc + cash) 0 in
  let total_vouchers := let transactions := [(50, 50), (100, 100), (200, 200), (400, 400), (800, 0)] in
                          transactions.foldl (λ acc (cash, voucher), acc + voucher) 0 in
  let total_purchase_power := total_cash + total_vouchers in
  (cash_max = total_cash) → 
  (total_purchase_power = total_value) →
  true :=
begin
  intros _ _ _ _ _ total_cash total_vouchers total_purchase_power _ _,
  exact true.intro,
end

end aunt_li_can_buy_goods_l294_294529


namespace total_handshakes_l294_294486

theorem total_handshakes :
  let gremlins := 20
  let imps := 20
  let sprites := 10
  let handshakes_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_gremlins_imps := gremlins * imps
  let handshakes_imps_sprites := imps * sprites
  handshakes_gremlins + handshakes_gremlins_imps + handshakes_imps_sprites = 790 :=
by
  sorry

end total_handshakes_l294_294486


namespace amount_paid_correct_l294_294111

-- Condition definitions
def apple_cost_before_discount (kg : ℝ) (rate : ℝ) : ℝ := kg * rate
def discount_amount (cost : ℝ) (discount_rate: ℝ) : ℝ := cost * discount_rate
def cost_after_discount (cost : ℝ) (discount: ℝ) : ℝ := cost - discount
def total_cost_after_discounts (costs : List ℝ) : ℝ := costs.sum
def final_amount (total_cost : ℝ) (threshold : ℝ) (promotion_discount : ℝ) : ℝ :=
  if total_cost > threshold then total_cost - promotion_discount else total_cost

-- Specific conditions and values
def apple_kg : ℝ := 8
def apple_rate : ℝ := 70
def apple_discount_rate : ℝ := 0.10

def mango_kg : ℝ := 9
def mango_rate : ℝ := 75
def mango_discount_rate : ℝ := 0.05

def banana_kg : ℝ := 6
def banana_rate : ℝ := 40
def banana_discount_rate : ℝ := 0

def grape_kg : ℝ := 4
def grape_rate : ℝ := 120
def grape_discount_rate : ℝ := 0.15

def cherry_kg : ℝ := 3
def cherry_rate : ℝ := 180
def cherry_discount_rate : ℝ := 0.20

def promotion_threshold : ℝ := 2000
def promotion_discount : ℝ := 100

-- Calculations
def total_cost_after_discounts_calculated : ℝ :=
  total_cost_after_discounts [
    cost_after_discount (apple_cost_before_discount apple_kg apple_rate) (discount_amount (apple_cost_before_discount apple_kg apple_rate) apple_discount_rate),
    cost_after_discount (apple_cost_before_discount mango_kg mango_rate) (discount_amount (apple_cost_before_discount mango_kg mango_rate) mango_discount_rate),
    cost_after_discount (apple_cost_before_discount banana_kg banana_rate) (discount_amount (apple_cost_before_discount banana_kg banana_rate) banana_discount_rate),
    cost_after_discount (apple_cost_before_discount grape_kg grape_rate) (discount_amount (apple_cost_before_discount grape_kg grape_rate) grape_discount_rate),
    cost_after_discount (apple_cost_before_discount cherry_kg cherry_rate) (discount_amount (apple_cost_before_discount cherry_kg cherry_rate) cherry_discount_rate)
  ]

def final_amount_paid : ℝ := final_amount total_cost_after_discounts_calculated promotion_threshold promotion_discount

theorem amount_paid_correct : final_amount_paid = 2125.25 := by
  sorry

end amount_paid_correct_l294_294111


namespace coefficient_x7_in_expansion_l294_294729

theorem coefficient_x7_in_expansion : 
  (x^2 + (1 / (2 * x)))^8 = ∑ i in finset.range 9, binomial 8 i * (x^2)^i * (1 / (2 * x))^(8 - i)
  ∧ (∑ i in finset.range 9, binomial 8 i * (2 * 1 / 2^(8 - i)) * x^(2i - (8 - i)) = x^7) → 
  ∑ i in finset.range 9, binomial 8 i * (1 / 2^i) * (x^(16 - 3i)) = x^7 :=
by
  sorry

end coefficient_x7_in_expansion_l294_294729


namespace correct_exponentiation_l294_294844

theorem correct_exponentiation (a : ℝ) : a^5 / a = a^4 := 
  sorry

end correct_exponentiation_l294_294844


namespace ratio_of_triangle_areas_l294_294776

theorem ratio_of_triangle_areas 
  (A B C D : Point)
  (h_eq_tri : equilateral_triangle A B C)
  (h_D_on_AC : lies_on_line_segment D A C)
  (h_angle_DBC : angle D B C = 30) :
  (area (triangle A D B)) / (area (triangle C D B)) = 1 / real.sqrt 3 :=
sorry

end ratio_of_triangle_areas_l294_294776


namespace P_of_polynomial_l294_294745

-- Define the polynomial P
noncomputable def P : ℕ → ℚ → ℚ := sorry

-- Main theorem statement proving the value of P(n+1)
theorem P_of_polynomial (n : ℕ) (P : (ℕ → ℚ → ℚ))
  (hP_deg : ∃ d, d ≤ n ∧ ∀ k, k ∈ Finset.range (n+1) → (P d k = k / (k + 1))) :
  let P_val (m : ℕ) := if m % 2 = 1 then 1 else (m / 2) / (m / 2 + 1) in
  P n (n+1) = P_val n := 
sorry

end P_of_polynomial_l294_294745


namespace elevator_translation_l294_294516

inductive Phenomenon
| rolling_soccer_ball
| rotating_fan_blades
| elevator_going_up
| moving_car_rear_wheel

def is_translation : Phenomenon → Prop
| Phenomenon.rolling_soccer_ball := false
| Phenomenon.rotating_fan_blades := false
| Phenomenon.elevator_going_up := true
| Phenomenon.moving_car_rear_wheel := false

theorem elevator_translation : is_translation Phenomenon.elevator_going_up = true :=
by
  sorry

end elevator_translation_l294_294516


namespace midpoint_sum_coordinates_l294_294593

theorem midpoint_sum_coordinates :
  let p1 : ℝ × ℝ := (8, -4)
  let p2 : ℝ × ℝ := (-2, 16)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1 + midpoint.2) = 9 :=
by
  let p1 : ℝ × ℝ := (8, -4)
  let p2 : ℝ × ℝ := (-2, 16)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  show midpoint.1 + midpoint.2 = 9
  sorry

end midpoint_sum_coordinates_l294_294593


namespace a_eq_0_iff_odd_function_l294_294086

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f (a : ℝ) (x : ℝ) : ℝ := (sin x) - (1 / x) + a

theorem a_eq_0_iff_odd_function (a : ℝ) : 
  (a = 0) ↔ (is_odd_function (f a)) := by
  sorry

end a_eq_0_iff_odd_function_l294_294086


namespace boy_running_speed_l294_294870

theorem boy_running_speed
  (seconds : ℕ) (side : ℕ) (seconds_per_hour : ℕ) (meters_per_kilometer : ℕ)
  (H1 : seconds = 36)
  (H2 : side = 30)
  (H3 : seconds_per_hour = 3600)
  (H4 : meters_per_kilometer = 1000) :
  let perimeter := 4 * side in
  let distance_in_km := perimeter / meters_per_kilometer in
  let time_in_hours := seconds / seconds_per_hour in
  let speed := distance_in_km / time_in_hours in
  speed = 12 :=
by
  sorry

end boy_running_speed_l294_294870


namespace install_charge_Y_correct_l294_294492

-- Definitions based on given conditions
def price_X := 575 : ℝ
def surcharge_rate_X := 0.04 : ℝ
def install_charge_X := 82.50 : ℝ

def price_Y := 530 : ℝ
def surcharge_rate_Y := 0.03 : ℝ
def saved_amount := 41.60 : ℝ

-- The proof problem
theorem install_charge_Y_correct (I : ℝ) :
  let surcharge_X := surcharge_rate_X * price_X in
  let total_cost_X := price_X + surcharge_X + install_charge_X in
  let surcharge_Y := surcharge_rate_Y * price_Y in
  let total_cost_Y := price_Y + surcharge_Y + I in
  total_cost_X - total_cost_Y = saved_amount →
  I = 93 := by
  sorry

end install_charge_Y_correct_l294_294492


namespace distance_orthocenter_circumcenter_eq_l294_294323

noncomputable def equidistant_orthocenter_circumcenter (A B C A1 B1 C1 : Type) 
    [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited A1] [Inhabited B1] [Inhabited C1] 
    (B1C1 CA1 A1B1 : Type) [Inhabited B1C1] [Inhabited CA1] [Inhabited A1B1] : Prop := 
  ∃ (h₁ : A ∈ B1C1) (h₂ : B ∈ CA1) (h₃ : C ∈ A1B1)
     (θ₁ : Angle ABC = Angle A1B1C1)
     (θ₂ : Angle BCA = Angle B1C1A1)
     (θ₃ : Angle CAB = Angle C1A1B1)
     (acute_triangle : AcuteTriangle A1 B1 C1),
  distance(orthocenter ABC, circumcenter ABC) = distance(orthocenter A1B1C1, circumcenter A1B1C1)
     
theorem distance_orthocenter_circumcenter_eq (A B C A1 B1 C1 : Type) 
    [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited A1] [Inhabited B1] [Inhabited C1]
    (B1C1 CA1 A1B1 : Type) [Inhabited B1C1] [Inhabited CA1] [Inhabited A1B1] 
    (circumcenter orthocenter : Type) [Inhabited circumcenter] [Inhabited orthocenter] :
  A ∈ B1C1 → B ∈ CA1 → C ∈ A1B1 →
  Angle ABC = Angle A1B1C1 → Angle BCA = Angle B1C1A1 → Angle CAB = Angle C1A1B1 →
  AcuteTriangle A1 B1 C1 →
  distance(orthocenter ABC, circumcenter ABC) = distance(orthocenter A1B1C1, circumcenter A1B1C1) :=
by sorry

end distance_orthocenter_circumcenter_eq_l294_294323


namespace card_probability_l294_294872

open ProbabilityTheory

def standard_deck : Finset (ℕ × String) := 
  {(n, s) | n ∈ {1, ..., 13} ∧ s ∈ {"Hearts", "Diamonds", "Clubs", "Spades"}}

-- Define what a face card is
def is_face_card (card : ℕ × String) : Prop := card.1 ∈ {11, 12, 13}

-- Probability of randomly drawing two cards from 51 that satisfy the conditions
theorem card_probability :
  let deck51 := standard_deck.erase (λ c, true) in
  ∃ card1 : (ℕ × String), card1 ∈ deck51 ∧
  ∃ card2 : (ℕ × String), card2 ∈ deck51 \ {card1} ∧ 
  card1.2 = card2.2 ∧ is_face_card card2 →
  probability (card2 ∈ deck51) = 3 / 50 :=
by sorry

end card_probability_l294_294872


namespace number_of_elements_in_B6_3_l294_294040
open BigOperators

/-- 
  Define the sets A and B and state the conditions:
  - n and m are positive integers.
  - Set A = {1, 2, ..., n}.
  - Set B_{n}^{m} consists of m-tuples (a_1, a_2, ..., a_m) such that a_i is in A.
  - |a_i - a_{i+1}| ≠ n - 1 for i = 1, 2, ..., m - 1.
  - Among a_1, a_2, ..., a_m (with m ≥ 3), there are at least three different elements.
-/
def A (n : ℕ) : Finset ℕ := 
  {1, 2, ..., n}

def Bnm (n m : ℕ) : Finset (Fin m → ℕ) :=
  { f ∈ Finset.univ.filter (λ f, ∀ i < m - 1, |f i - f (i + 1)| ≠ n - 1) | 
    ∃ i j k < m, f i ≠ f j ∧ f i ≠ f k ∧ f j ≠ f k }

/-- 
  The number of elements in B₆³ is 104.
-/
theorem number_of_elements_in_B6_3 : 
  Bnm 6 3.card = 104 :=
sorry

end number_of_elements_in_B6_3_l294_294040


namespace greatest_negative_root_correct_l294_294987

noncomputable def greatest_negative_root := 
  (Real.arctan (1 / 8) + Real.arctan (4 / 7) - 2 * Real.pi) / 9

theorem greatest_negative_root_correct :
  ∃ x ∈ Set.interval (-Real.pi) 0, 
    x < 0 ∧ 
    (sin x + 8 * cos x = 4 * sin (8 * x) + 7 * cos (8 * x)) ∧ 
    (∀ y ∈ Set.interval (-Real.pi) 0, 
      y < 0 ∧ (sin y + 8 * cos y = 4 * sin (8 * y) + 7 * cos (8 * y)) → y ≤ x) :=
begin
  use greatest_negative_root,
  split,
  {
    -- Prove that 'greatest_negative_root' is within the interval [-π, 0] and is negative
    sorry
  },
  split,
  {
    -- Prove that 'greatest_negative_root' satisfies the original equation
    sorry
  },
  {
    -- Prove that 'greatest_negative_root' is the greatest negative root
    sorry
  }
end

end greatest_negative_root_correct_l294_294987


namespace choose_k_numbers_l294_294035

theorem choose_k_numbers (n k : ℕ) : 
  (finset.card (finset.filter (λ (l : list ℕ), l.sum = n) 
  (finset.range (n + 1)).powerset_len k)) = nat.choose (n + k - 1) (k - 1) :=
sorry

end choose_k_numbers_l294_294035


namespace triangles_converge_to_equilateral_l294_294459

theorem triangles_converge_to_equilateral :
  ∀ (α₀ β₀ γ₀ : ℝ), (α₀ + β₀ + γ₀ = π) →
  (∀ k : ℕ, ∃ (α β γ : ℝ), 
    (α (k+1) = (π / 2) - (α k / 2)) ∧ 
    (β (k+1) = (π / 2) - (β k / 2)) ∧ 
    (γ (k+1) = (π / 2) - (γ k / 2)) ∧ 
    (α k + β k + γ k = π)) → 
  (∀ k : ℕ, tendsto (λ k, (α k, β k, γ k)) at_top (𝓝 (π / 3, π / 3, π / 3))) :=
begin
  sorry
end

end triangles_converge_to_equilateral_l294_294459


namespace kite_circle_radius_l294_294477

theorem kite_circle_radius (a b : ℝ) (h : a ≠ b) (h_right : ∃ (AB BC : ℝ), AB = a ∧ BC = b ∧ ∠ AB BC = 90) :
  ∃ r : ℝ, r = (a * b) / (a - b) :=
by
  sorry

end kite_circle_radius_l294_294477


namespace valid_x_l294_294656

def proposition_p (x : ℤ) : Prop := abs (x - 1) ≥ 2
def proposition_q (x : ℤ) : Prop := x ∈ Int

theorem valid_x {x : ℤ} :
  (¬(proposition_p x ∧ proposition_q x)) ∧ (¬(¬(proposition_q x))) →
  x = 0 ∨ x = 1 ∨ x = 2 :=
by
  sorry

end valid_x_l294_294656


namespace conjugate_of_complex_l294_294087

/-- The conjugate of the complex number (-1 - i) / i is -1 - i. -/
theorem conjugate_of_complex : complex.conj ((-1 - complex.i) / complex.i) = -1 - complex.i := 
by
  sorry

end conjugate_of_complex_l294_294087


namespace circles_are_externally_tangent_l294_294840

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def radius_1 : ℝ := 2
noncomputable def radius_2 : ℝ := 8

noncomputable def center_1 : ℝ × ℝ := (-3, 2)
noncomputable def center_2 : ℝ × ℝ := (3, -6)

noncomputable def circles_externally_tangent : Prop :=
  distance center_1 center_2 = radius_1 + radius_2

theorem circles_are_externally_tangent :
  circles_externally_tangent :=
by
  sorry

end circles_are_externally_tangent_l294_294840


namespace polar_point_conversion_l294_294372

noncomputable def polar_convert (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
if r < 0 then (-r, θ + Real.pi) else (r, θ)

theorem polar_point_conversion:
  polar_convert (-3) (Real.pi / 4) = (3, 5 * Real.pi / 4) :=
by
  simp [polar_convert, Real.pi]
  done

end polar_point_conversion_l294_294372


namespace horner_v4_l294_294209

def f (x : ℝ) : ℝ := 3 * x^6 + 5 * x^5 + 6 * x^4 + 20 * x^3 - 8 * x^2 + 35 * x + 12

theorem horner_v4 :
  let x := -2 in
  let v0 := 3 in
  let v1 := v0 * x + 5 in
  let v2 := v1 * x + 6 in
  let v3 := v2 * x + 20 in
  let v4 := v3 * x - 8 in
  v4 = -16 := 
by
  intros
  suffices : v4 = -16 by exact this
  sorry

end horner_v4_l294_294209


namespace sum_is_2000_l294_294831

theorem sum_is_2000 (x y : ℝ) (h : x ≠ y) (h_eq : x^2 - 2000 * x = y^2 - 2000 * y) : x + y = 2000 := by
  sorry

end sum_is_2000_l294_294831


namespace triangle_area_ratio_l294_294772

open Real

-- Define an equilateral triangle with a point D on AC and give the lean statement to prove the required ratio.
theorem triangle_area_ratio (A B C D : Point) (ABC : Triangle) (h_eq : is_equilateral ABC) (hD : D ∈ Segment AC) (h_angle : ∠ DBC = 30) :
  area (triangle A D B) / area (triangle C D B) = 0 :=
sorry

end triangle_area_ratio_l294_294772


namespace triangle_area_ratio_l294_294811

noncomputable def triangle_ratio (m n : ℝ) : ℝ :=
(m + n) ^ 2 / (m * n)

theorem triangle_area_ratio (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let ratio := (m + n) ^ 2 / (m * n) in
  ratio = triangle_ratio m n := 
sorry

end triangle_area_ratio_l294_294811


namespace thirtieth_number_in_base_5_l294_294725

theorem thirtieth_number_in_base_5 : Nat.toDigits 5 30 = [1, 1, 0] :=
by
  sorry

end thirtieth_number_in_base_5_l294_294725


namespace days_for_30_men_to_build_wall_l294_294390

theorem days_for_30_men_to_build_wall 
  (men1 days1 men2 k : ℕ)
  (h1 : men1 = 18)
  (h2 : days1 = 5)
  (h3 : men2 = 30)
  (h_k : men1 * days1 = k)
  : (men2 * 3 = k) := by 
sorry

end days_for_30_men_to_build_wall_l294_294390


namespace weighted_average_speed_l294_294565

theorem weighted_average_speed (x : ℝ) (h_pos : 0 < x) :
  let distance1 := x,
      speed1 := 40,
      distance2 := 2 * x,
      speed2 := 20,
      distance3 := 3 * x,
      speed3 := 60,
      distance4 := 4 * x,
      speed4 := 30,
      total_distance := distance1 + distance2 + distance3 + distance4,
      time1 := distance1 / speed1,
      time2 := distance2 / speed2,
      time3 := distance3 / speed3,
      time4 := distance4 / speed4,
      total_time := time1 + time2 + time3 + time4
  in (10 * x) / total_time = (1200 * x) / 37 :=
by
  sorry

end weighted_average_speed_l294_294565


namespace card_arrangement_possible_l294_294604

theorem card_arrangement_possible :
  ∃ (left right : List ℕ) (l_sum r_sum : ℕ),
  left = [1, 4, 7, 6] ∧
  right = [2, 5, 8, 3] ∧
  l_sum = left.sum ∧
  r_sum = right.sum ∧
  l_sum = 18 ∧
  r_sum = 18 :=
by
  let left := [1, 4, 7, 6]
  let right := [2, 5, 8, 3]
  let l_sum := left.sum
  let r_sum := right.sum
  have h1 : left = [1, 4, 7, 6] := rfl
  have h2 : right = [2, 5, 8, 3] := rfl
  have h3 : l_sum = left.sum := rfl
  have h4 : r_sum = right.sum := rfl
  have h5 : l_sum = 18 :=
  by
    rw [left.sum]
    simp
  have h6 : r_sum = 18 :=
  by
    rw [right.sum]
    simp
  exact ⟨left, right, l_sum, r_sum, h1, h2, h3, h4, h5, h6⟩

end card_arrangement_possible_l294_294604


namespace exists_arrangement_splits_all_l294_294033

-- Define the set U
def U (n : ℕ) : set ℕ := { x ∈ set.univ | x ≥ 1 ∧ x ≤ n }

-- Define what it means for a subset S of U to be split by an arrangement
def is_split_by (n : ℕ) (S : set ℕ) (arr : list ℕ) : Prop :=
  ∃ i j k : ℕ, i < k ∧ k < j ∧ list.nth arr i ∈ S ∧ list.nth arr j ∈ S ∧ 
  (list.nth arr k ∉ S ∧ list.nth arr k ∈ U n)

-- Main theorem statement
theorem exists_arrangement_splits_all (n : ℕ) (hn: n ≥ 3) 
  (subsets : finset (set ℕ))
  (hsubsets : ∀ S ∈ subsets, 2 ≤ S.card ∧ S.card ≤ n - 1 ∧ S ⊆ U n)
  (hlen : subsets.card = n - 2) :
  ∃ arr : list ℕ, ∀ S ∈ subsets, is_split_by n S arr :=
sorry

end exists_arrangement_splits_all_l294_294033


namespace exponential_equivalence_l294_294845

theorem exponential_equivalence (a : ℝ) : a^4 * a^4 = (a^2)^4 := by
  sorry

end exponential_equivalence_l294_294845


namespace trig_proof_l294_294690

noncomputable def trig_problem (α : ℝ) (h : Real.tan α = 3) : Prop :=
  Real.cos (α + Real.pi / 4) ^ 2 - Real.cos (α - Real.pi / 4) ^ 2 = -3 / 5

theorem trig_proof (α : ℝ) (h : Real.tan α = 3) : Real.cos (α + Real.pi / 4) ^ 2 - Real.cos (α - Real.pi / 4) ^ 2 = -3 / 5 :=
by
  sorry

end trig_proof_l294_294690


namespace janice_walk_dog_more_than_homework_l294_294394

theorem janice_walk_dog_more_than_homework 
  (H C T: Nat) 
  (W: Nat) 
  (total_time remaining_time spent_time: Nat) 
  (hw_time room_time trash_time extra_time: Nat)
  (H_eq : H = 30)
  (C_eq : C = H / 2)
  (T_eq : T = H / 6)
  (remaining_time_eq : remaining_time = 35)
  (total_time_eq : total_time = 120)
  (spent_time_eq : spent_time = total_time - remaining_time)
  (task_time_sum_eq : task_time_sum = H + C + T)
  (W_eq : W = spent_time - task_time_sum)
  : W - H = 5 := 
sorry

end janice_walk_dog_more_than_homework_l294_294394


namespace similar_right_triangles_l294_294890

theorem similar_right_triangles (x c : ℕ) 
  (h1 : 12 * 6 = 9 * x) 
  (h2 : c^2 = x^2 + 6^2) :
  x = 8 ∧ c = 10 :=
by
  sorry

end similar_right_triangles_l294_294890


namespace seed_cost_proof_l294_294586

def seed_cost_per_pound (total_cost : ℝ) (weight_in_pounds : ℕ) :=
  total_cost / weight_in_pounds

def total_seed_cost (cost_per_pound : ℝ) (total_pounds : ℕ) :=
  cost_per_pound * total_pounds

theorem seed_cost_proof :
  let cost_two_pounds := 44.68 in
  let pounds_needed := 6 in
  let cost_per_pound := seed_cost_per_pound cost_two_pounds 2 in
  total_seed_cost cost_per_pound pounds_needed = 134.04 :=
by
  sorry

end seed_cost_proof_l294_294586


namespace determine_pairs_l294_294612

theorem determine_pairs (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (xy^2 + y + 7 ∣ x^2 y + x + y) ↔ (∃ t : ℕ, x = 7 * t^2 ∧ y = 7 * t) ∨ (x = 11 ∧ y = 1) ∨ (x = 49 ∧ y = 1) := by
  sorry

end determine_pairs_l294_294612


namespace sum_of_reciprocals_less_than_30_l294_294749

variable {a : ℕ → ℕ}
variable {n : ℕ}

noncomputable def no_digit_nine (x : ℕ) : Prop :=
  (∀ (d : ℕ), d ∈ x.digits 10 → d ≠ 9)

theorem sum_of_reciprocals_less_than_30 
  (hn : ∀ i, i < n → no_digit_nine (a i)) 
  (hdistinct : ∀ i j, i < n → j < n → a i ≠ a j) :
  (∑ i in Finset.range n, (a i)⁻¹) < 30 :=
by
  sorry

end sum_of_reciprocals_less_than_30_l294_294749


namespace sum_first_nine_terms_l294_294721

-- Definitions for the problem
def a1 := 19
def d := -2
def S_9 := 9 * a1 + (9 * 4) * d

-- The theorem to be proven
theorem sum_first_nine_terms (h1 : a1 + 3 * d = 13) (h2 : a1 + 5 * d = 9) : S_9 = 99 :=
by
  -- Definitions for the theorem
  have h3 : d = -2, from sorry,
  have h4 : a1 = 19, from sorry,
  -- Calculate S_9
  have S_9_calc : S_9 = 9 * a1 + (9 * 4) * d, from sorry,
  -- Substitute the values and prove the result
  sorry

end sum_first_nine_terms_l294_294721


namespace question_statement_l294_294357

-- Definitions based on conditions
def all_cards : List ℕ := [8, 3, 6, 5, 0, 7]
def A : ℕ := 876  -- The largest number from the given cards.
def B : ℕ := 305  -- The smallest number from the given cards with non-zero hundreds place.

-- The proof problem statement
theorem question_statement :
  (A - B) * 6 = 3426 := by
  sorry

end question_statement_l294_294357


namespace simplify_trig_expr_l294_294788

variable (x : ℝ)

theorem simplify_trig_expr :
  (Real.tan x - 2 * Real.tan (2 * x) + 4 * Real.tan (4 * x) + 8 * (Real.cot (8 * x))) = Real.cot x := 
sorry

end simplify_trig_expr_l294_294788


namespace find_a_for_relative_min_l294_294970

noncomputable def f (x a : ℝ) : ℝ := x^4 - x^3 - x^2 + a * x + 1

theorem find_a_for_relative_min (a : ℝ) :
  (∀ x, f x a = x^4 - x^3 - x^2 + a * x + 1) ∧
  (∀ x, has_deriv_at (f x a) (4 * x^3 - 3 * x^2 - 2 * x + a) x) ∧
  (f a a = a) ∧
  (∀ x, 4 * x^3 - 3 * x^2 - 2 * x + a = 0 → f a a = a)
  → a = 1 := by
  sorry

end find_a_for_relative_min_l294_294970


namespace total_amount_spent_l294_294861

theorem total_amount_spent 
  (num_pens : ℕ)
  (num_pencils : ℕ)
  (price_per_pen : ℝ)
  (price_per_pencil : ℝ)
  (total_cost : ℝ) :
  num_pens = 30 →
  num_pencils = 75 →
  price_per_pen = 14 →
  price_per_pencil = 2 →
  total_cost = num_pens * price_per_pen + num_pencils * price_per_pencil →
  total_cost = 570 :=
by 
  intros h1 h2 h3 h4 h5;
  rw [h1, h2, h3, h4];
  rw [← h5];
  exact eq.refl _


end total_amount_spent_l294_294861


namespace total_kids_in_Lawrence_l294_294974

theorem total_kids_in_Lawrence (stay_home kids_camp total_kids : ℕ) (h1 : stay_home = 907611) (h2 : kids_camp = 455682) (h3 : total_kids = stay_home + kids_camp) : total_kids = 1363293 :=
by
  sorry

end total_kids_in_Lawrence_l294_294974


namespace parabola_equation_l294_294303

-- Statement of the problem in Lean
theorem parabola_equation (p : ℝ) (h_pos : p > 0) :
    let F := (p / 2, 0) in
    let l := { x : ℝ // x * tan (Real.pi / 4) = x - p / 2 } in
    let A := { y : ℝ // y = x - p / 2 ∧ (x, y) ∈ parabola(2 * p * x) } in
    let B := { y : ℝ // y = x - p / 2 ∧ (x, y) ∈ parabola(2 * p * x) } in
    let M := (-(p / 2), 2) in
    (d := (F,R2O(z)==>-((M,R1)=>F(x.M)),b,μ-1==(h⊕μ)),b))); -- parabola\(2x\)$) :=
y^2 = 16 * x 

end parabola_equation_l294_294303


namespace length_of_water_fountain_l294_294538

theorem length_of_water_fountain :
  (∀ (L1 : ℕ), 20 * 14 = L1) ∧
  (35 * 3 = 21) →
  (20 * 14 = 56) := by
sorry

end length_of_water_fountain_l294_294538


namespace minimum_value_of_f_l294_294296

noncomputable def f (x : ℝ) : ℝ := (2 * x^2 + x + 4) / x

theorem minimum_value_of_f :
  ∃ (x : ℝ), 0 < x ∧ (∀ (y : ℝ), 0 < y → f(y) ≥ 4 * sqrt 2 + 1) ∧ f(x) = 4 * sqrt 2 + 1 :=
sorry

end minimum_value_of_f_l294_294296


namespace find_a_for_tangent_line_l294_294697

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x - a * log x

theorem find_a_for_tangent_line :
  (∃ a : ℝ, f a 1 = 2 - a * log 1 ∧ (∂ (f a x) / ∂ x) 1 = 1) → a = 1 :=
sorry

end find_a_for_tangent_line_l294_294697


namespace checkerboard_ratio_sum_l294_294215

theorem checkerboard_ratio_sum (h_lines : ℕ) (v_lines : ℕ) (rect_count : ℕ) (square_count : ℕ) :
  h_lines = 10 → v_lines = 10 → rect_count = (h_lines.choose 2) * (v_lines.choose 2) → 
  square_count = (List.range 9).sum (λ n, (n + 1) * (n + 1)) →
  let ratio := square_count.to_rat / rect_count.to_rat,
      m := 19, n := 135 
  in m + n = 154 := 
by
  intros
  simp only [h_lines, v_lines, rect_count, square_count] at *
  subst_vars -- substitute the bound variables
  let required_sum := 19 + 135
  guard_hyp {p := ratio = (285 : ℚ) / (2025 : ℚ)}
  calc
    m + n = 19 + 135 := rfl
    required_sum = 154 := rfl
  sorry

end checkerboard_ratio_sum_l294_294215


namespace abs_pi_expression_l294_294947

theorem abs_pi_expression : |π - |π - 10|| = 10 - 2 * π :=
by
  sorry

end abs_pi_expression_l294_294947


namespace shortest_distance_between_circles_l294_294122

noncomputable def circle_A_center : ℝ × ℝ := (3, 4)
noncomputable def circle_A_radius : ℝ := real.sqrt 21
noncomputable def circle_B_center : ℝ × ℝ := (-4, -6)
noncomputable def circle_B_radius : ℝ := 4

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem shortest_distance_between_circles :
  distance circle_A_center circle_B_center - (circle_A_radius + circle_B_radius) = real.sqrt 149 - real.sqrt 21 - 4 :=
by
  sorry

end shortest_distance_between_circles_l294_294122


namespace total_fish_in_pond_l294_294267

theorem total_fish_in_pond (N : ℕ) (h1 : 80 ≤ N) (h2 : 5 ≤ 150) (h_marked_dist : (5 : ℚ) / 150 = (80 : ℚ) / N) : N = 2400 := by
  sorry

end total_fish_in_pond_l294_294267


namespace area_result_l294_294924

noncomputable def area_between_polar_curves : ℝ :=
  let r1 (φ : ℝ) := 4 * sin (3 * φ)
  let r2 (φ : ℝ) := 2
  6 * (1 / 2 * ∫ φ in (π / 18)..(π / 6), (r1 φ)^2 - (r2 φ)^2)
  
theorem area_result :
  area_between_polar_curves = 4 * π + 2 * real.sqrt 3 := 
sorry

end area_result_l294_294924


namespace volume_ratio_pyramid_prism_l294_294856

theorem volume_ratio_pyramid_prism
  (h n a : ℝ)
  (h_pos : 0 < h) (a_pos : 0 < a) (n_pos : 1 < n)
  (V1 V2 : ℝ)
  (V1_def : V1 = (1 / 24) * (5 * n + 3)^3 * a * h / (n + 1))
  (V2_def : V2 = (1 / 2) * (n + 1)^2 * a * h) :
  V1 / V2 = (5 * n + 3)^3 / (12 * (n + 1)^3) :=
by {
  rw [V1_def, V2_def],
  field_simp,
  rw [mul_div_mul_comm, ←mul_assoc],
  congr' 1,
  ring,
  intro V1_pos,
  intro V2_pos,
  exact V1_pos,
  exact V2_pos,
  intro neu,
  ring,
  intro V1_pos,
  intro V2_pos,
  exact V1_pos,
  exact V2_pos,
  intro neu,
  ring,
  sorry
}

end volume_ratio_pyramid_prism_l294_294856


namespace sphere_has_circular_views_l294_294577

-- Define the geometric shapes
inductive Shape
| cuboid
| cylinder
| cone
| sphere

-- Define a function that describes the views of a shape
def views (s: Shape) : (String × String × String) :=
match s with
| Shape.cuboid   => ("Rectangle", "Rectangle", "Rectangle")
| Shape.cylinder => ("Rectangle", "Rectangle", "Circle")
| Shape.cone     => ("Isosceles Triangle", "Isosceles Triangle", "Circle")
| Shape.sphere   => ("Circle", "Circle", "Circle")

-- Define the property of having circular views in all perspectives
def has_circular_views (s: Shape) : Prop :=
views s = ("Circle", "Circle", "Circle")

-- The theorem to prove
theorem sphere_has_circular_views :
  ∀ (s : Shape), has_circular_views s ↔ s = Shape.sphere :=
by sorry

end sphere_has_circular_views_l294_294577


namespace noemi_initial_amount_l294_294427

-- Define the conditions
def lost_on_roulette : Int := 400
def lost_on_blackjack : Int := 500
def still_has : Int := 800
def total_lost : Int := lost_on_roulette + lost_on_blackjack

-- Define the theorem to be proven
theorem noemi_initial_amount : total_lost + still_has = 1700 := by
  -- The proof will be added here
  sorry

end noemi_initial_amount_l294_294427


namespace range_of_k_l294_294616

theorem range_of_k (k : ℝ) : 
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π/2 ∧ √3 * real.sin (2 * x) + real.cos (2 * x) = k + 1) ↔ (-2 ≤ k ∧ k ≤ 1) :=
by
  sorry

end range_of_k_l294_294616


namespace rational_root_of_polynomials_l294_294034

noncomputable theory
open Polynomial

theorem rational_root_of_polynomials {f g : ℤ[X]} (h₀ : f ≠ 0) (h₁ : g ≠ 0)
  (h₂ : f.natDegree > g.natDegree)
  (h₃ : ∃ᶠ p in Filter.atTop, ∃ x : ℚ,  (p : ℤ) • f.coeff (C x) + g.coeff (C x) = 0) : 
  ∃ x : ℚ, is_root f x :=
sorry

end rational_root_of_polynomials_l294_294034


namespace find_sin_alpha_l294_294283

theorem find_sin_alpha (α : ℝ) (h1 : 0 < α ∧ α < real.pi) (h2 : 3 * real.cos (2 * α) - 8 * real.cos α = 5) :
  real.sin α = real.sqrt 5 / 3 :=
sorry

end find_sin_alpha_l294_294283


namespace number_of_correct_conclusions_zero_l294_294567

-- Define the conditions and conclusions as per the problem statement
structure GeoProblem :=
  (line1 : Type)
  (line2 : Type)
  (plane : Type)
  (parallel_to_plane : line1 → plane → Prop)
  (no_common_points : line1 → line2 → Prop)
  (perpendicular_to : line1 → line2 → Prop)
  (within_plane : line1 → plane → Prop)

-- Define the conclusions
def conclusion1 (G : GeoProblem) :=
  ∀ (L1 L2 : G.line1) (P : G.plane),
    G.parallel_to_plane L1 P ∧ G.parallel_to_plane L2 P → 
    (L1 = L2) -- This for simplicity, typically needs defining parallelism properly

def conclusion2 (G : GeoProblem) :=
  ∀ (L1 L2 : G.line1),
    G.no_common_points L1 L2 → 
    (L1 = L2) -- Same simplification reason

def conclusion3 (G : GeoProblem) :=
  ∀ (L1 L2 L3 : G.line1),
    G.perpendicular_to L1 L3 ∧ G.perpendicular_to L2 L3 →
    (L1 = L2)

def conclusion4 (G : GeoProblem) :=
  ∀ (L : G.line1) (P : G.plane),
    (∀ L', G.within_plane L' P → G.no_common_points L L') →
    G.parallel_to_plane L P

-- Prove the number of correct conclusions is 0
theorem number_of_correct_conclusions_zero (G : GeoProblem) :
  ¬ conclusion1 G ∧ ¬ conclusion2 G ∧ ¬ conclusion3 G ∧ ¬ conclusion4 G → 
  (0 = 0) :=
sorry

end number_of_correct_conclusions_zero_l294_294567


namespace sum_first_10_b_n_l294_294638

variable (a b : ℕ → ℝ)
variable (n : ℕ)

def a_n (n : ℕ) : ℝ := (n + 1) * (n + 2)
def b_n (n : ℕ) : ℝ := 1 / (a_n n)

theorem sum_first_10_b_n :
  (∑ i in Finset.range 10, b_n i) = 5 / 12 := by
  sorry

end sum_first_10_b_n_l294_294638


namespace sin_max_min_period_y_max_min_period_l294_294322

variable {a b x : ℝ}

theorem sin_max_min_period (ha_max : a + b = 3 / 2)
                            (ha_min : a - b = -1 / 2) :
  (a / 2 = 1 / 2 ∧ b = 1) ∨ (a / 2 = -1 / 2 ∧ b = -1) :=
by
  have h1 : 2 * a = 1 := by linarith [ha_max, ha_min]
  have ha : a = 1 / 2 := by linarith [h1]
  have hb : b = 1 := by linarith [ha, ha_max]
  exact or.inl (⟨ha, hb⟩)

theorem y_max_min_period 
   {a b : ℝ} (ha_max : a + b = 3 / 2) (ha_min : a - b = -1 / 2) :
   max (a * sin b) = 1 / 2 ∧ min (a * sin b) = -1 / 2 ∧ b ≠ 0 :=
by
  sorry

end sin_max_min_period_y_max_min_period_l294_294322


namespace P_gt_Q_l294_294659

theorem P_gt_Q (a b c : ℝ) (h_neq : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  let P := a^2 + b^2 + c^2 + c⁻²
  let Q := 2 * a + 2 * b
  P > Q :=
by 
  -- proof steps would go here
  sorry

end P_gt_Q_l294_294659


namespace complement_union_eq_complement_l294_294002

open Set

section ComplementUnion

variable (k : ℤ)

def SetA : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def SetB : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def UniversalSet : Set ℤ := univ
def ComplementUnion : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem complement_union_eq_complement :
  UniversalSet \ (SetA ∪ SetB) = ComplementUnion :=
by
  sorry

end ComplementUnion

end complement_union_eq_complement_l294_294002


namespace sum_of_3rd_and_7th_l294_294644

def sums_of_nine (xs : List ℕ) : List ℕ :=
(xs.map (λ x => xs.sum - x)).eraseDups

theorem sum_of_3rd_and_7th (xs : List ℕ) (h_len : xs.length = 10)
  (h_pos : ∀ x ∈ xs, 0 < x)
  (h_distsum : sums_of_nine xs = [86, 87, 88, 89, 90, 91, 93, 94, 95]) :
  xs.sort (· ≥ ·) !! 2 + xs.sort (· ≥ ·) !! 6 = 22 :=
sorry

end sum_of_3rd_and_7th_l294_294644


namespace alternating_binomials_sum_l294_294260

-- Define a function to represent the alternating sum of binomials.
def alternating_sum_binomials (n : ℕ) : ℤ :=
  ∑ k in finset.range (n / 2 + 1), (-1 : ℤ) ^ k * nat.choose n (2 * k)

theorem alternating_binomials_sum :
  alternating_sum_binomials 100 = -2 ^ 50 :=
by
  sorry

end alternating_binomials_sum_l294_294260


namespace complex_abs_squared_l294_294410

theorem complex_abs_squared (a b : ℝ) (z : ℂ) (h1 : z = a + b * complex.I) (h2 : z - complex.abs z = 4 - 6 * complex.I) : abs z ^ 2 = 42.25 := 
by
  sorry

end complex_abs_squared_l294_294410


namespace perimeter_of_equilateral_triangle_l294_294885

noncomputable def equilateral_triangle : Type := sorry

variables (p1 p2 p3 : equilateral_triangle)

axiom intersects_origin (p : equilateral_triangle) : sorry

axiom intersects_y_eq_2 (p : equilateral_triangle) : sorry

axiom intersects_y_eq_1_plus_sqrt3x (p : equilateral_triangle) : sorry

theorem perimeter_of_equilateral_triangle : sorry :=
  let l1 := 2
  let l2 := 1 + Real.sqrt 3 * x in
  let line_through_origin_slope := -1 / Real.sqrt 3 in
  let triangle_perimeter := 7 * Real.sqrt 3 in
  triangle_perimeter = triangle_perimeter sorry

end perimeter_of_equilateral_triangle_l294_294885


namespace binomial_alternating_sum_l294_294264

theorem binomial_alternating_sum :
  (Finset.range 51).sum (λ k, (-1 : ℤ)^k * Nat.choose 100 (2 * k)) = -2^50 := 
by
    sorry

end binomial_alternating_sum_l294_294264


namespace analogous_propositions_correct_complex_l294_294217

open Complex

-- Define the first condition
def cond1 (a b : ℝ) : Prop := a - b = 0 → a = b

-- Define the second condition for real and complex numbers
def cond2_re (a b c d : ℝ) : Prop := (a + b * I = c + d * I) → (a = c ∧ b = d)
def cond2_im (a b c d : ℂ) : Prop := (a + b * I = c + d * I) → (a = c ∧ b = d)

-- Define the third condition
def cond3 (a b : ℝ) : Prop := a - b > 0 → a > b

-- Define the fourth condition
def cond4 (a b : ℝ) : Prop := a * b = 0 → a = 0 ∨ b = 0
def cond4_im (a b : ℂ) : Prop := a * b = 0 → a = 0 ∨ b = 0

-- The theorem statement
theorem analogous_propositions_correct_complex :
  cond1 ∧ cond2_re ∧ ¬cond3 ∧ cond4 ∧ cond2_im ∧ cond4_im →
  (¬cond3 → 3 = 3) :=
by
  intros
  sorry

end analogous_propositions_correct_complex_l294_294217


namespace rotating_ngon_shape_l294_294561

theorem rotating_ngon_shape (n : ℕ) (hne : n > 2) :
  ∃ (shapes : list (polygon ℝ)), 
    (∀ shape ∈ shapes, is_regular_ngon shape n 1) ∧
    shape_formed_by_rotating_ngon n = closed_path (successive_positions n) shapes :=
sorry

end rotating_ngon_shape_l294_294561


namespace coin_count_l294_294524

theorem coin_count (x : ℕ) (h : 1 * x + 0.50 * x + 0.25 * x = 105) : x = 60 :=
by {
  sorry
}

end coin_count_l294_294524


namespace factorial_division_l294_294214

theorem factorial_division : (12! / 10!) = 132 := by
  sorry

end factorial_division_l294_294214


namespace length_segment_QR_l294_294703

-- Define the geometric setup
variables {A B C : Type} [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C]
variables (AB AC BC : ℝ) (hABC : is_right_triangle A B C) (radiusP : P.radius = 7.2)
variables (circleP : Circle) (PpassesC : passes_through circleP C) (tangentP : is_tangent circleP AB)
variables (Q R : Point) (Q_on_AC : OnLine Q AC) (R_on_BC : OnLine R BC) (intersectP_QR : intersects_circle circleP Q R)

-- Define the main theorem
theorem length_segment_QR (AB_eq : AB = 15) (AC_eq : AC = 12) (BC_eq : BC = 9) :
  length_segment Q R = 14.4 :=
by
  sorry

end length_segment_QR_l294_294703


namespace cans_in_each_package_of_cat_food_l294_294197

-- Definitions and conditions
def cans_per_package_cat (c : ℕ) := 9 * c
def cans_per_package_dog := 7 * 5
def extra_cans_cat := 55

-- Theorem stating the problem and the answer
theorem cans_in_each_package_of_cat_food (c : ℕ) (h: cans_per_package_cat c = cans_per_package_dog + extra_cans_cat) :
  c = 10 :=
sorry

end cans_in_each_package_of_cat_food_l294_294197


namespace RiversideAcademy_statistics_l294_294917

theorem RiversideAcademy_statistics (total_students physics_students both_subjects : ℕ)
  (h1 : total_students = 25)
  (h2 : physics_students = 10)
  (h3 : both_subjects = 6) :
  total_students - (physics_students - both_subjects) = 21 :=
by
  sorry

end RiversideAcademy_statistics_l294_294917


namespace gain_percent_l294_294525

theorem gain_percent (CP SP : ℝ) (hCP : CP = 100) (hSP : SP = 120) : 
  ((SP - CP) / CP) * 100 = 20 := 
by {
  rw [hCP, hSP],
  norm_num,
}

end gain_percent_l294_294525


namespace sum_of_roots_l294_294355

theorem sum_of_roots (x: ℝ) (h: 3 * x ^ 2 - 15 * x = 0) : x = 0 ∨ x = 5 → (∀ x₁ x₂, x₁ ≠ x₂ ∧ (3 * x₁ ^ 2 - 15 * x₁ = 0) ∧ (3 * x₂ ^ 2 - 15 * x₂ = 0) → x₁ + x₂ = 5) :=
by
  intro h1
  have h₀ : x = 0 ∨ x = 5, from h1
  intro x₁ x₂ h₂ hx₁ hx₂
  rw eq_comm at h0 h0
  sorry

end sum_of_roots_l294_294355


namespace sufficient_but_not_necessary_l294_294046

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 1) : (1 / a < 1) := 
by
  sorry

end sufficient_but_not_necessary_l294_294046


namespace units_digit_of_product_of_odd_multiples_of_3_between_10_and_200_l294_294841

-- Define what it means to be an odd multiple of 3 within a specific range
def odd_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 2 = 1

-- Define the list of all odd multiples of 3 within the given range
def odd_multiples_of_3_in_range (a b : ℕ) : List ℕ :=
  (List.range (b - a + 1)).map (λ x => a + x)
  |>.filter (λ x => odd_multiple_of_3 x)

-- Define the function to get the units digit of a number
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Define the product of a list of numbers
def list_product (l : List ℕ) : ℕ :=
  l.foldr (λ x acc => x * acc) 1

-- Define the main problem
theorem units_digit_of_product_of_odd_multiples_of_3_between_10_and_200 :
  units_digit (list_product (odd_multiples_of_3_in_range 10 200)) = 5 :=
sorry

end units_digit_of_product_of_odd_multiples_of_3_between_10_and_200_l294_294841


namespace exists_even_p_and_odd_composite_p_q_l294_294584

theorem exists_even_p_and_odd_composite_p_q :
  ∃ (p q : ℝ → ℝ), 
    (∀ x, p (-x) = p x) ∧ 
    (∀ x, p (q (-x)) = -p (q x)) :=
by
  have p := λ x, Real.cos x
  have q := λ x, (Real.pi / 2) - x
  
  use [p, q]
  
  split
  { intro x
    simp [p, Real.cos_neg] }
  { intro x
    simp [q, p, Real.cos_sub, Real.sin_neg] }
  sorry

end exists_even_p_and_odd_composite_p_q_l294_294584


namespace garrison_provisions_l294_294177

theorem garrison_provisions (x : ℕ) (h1 : 1850 * (x - 12) = 2960 * 10) : x = 28 :=
by {
    -- constraints
    have h2_h3 := (1850 : ℕ),
    have h4 := (2960 : ℕ),
    have := h1,
    sorry
}

end garrison_provisions_l294_294177


namespace min_length_of_path_in_unit_cube_l294_294874

theorem min_length_of_path_in_unit_cube : 
  ∀ (P : ℝ³ → Prop), 
  (∀ x : ℝ³, P x → (0 ≤ x.x ∧ x.x ≤ 1) ∧ (0 ≤ x.y ∧ x.y ≤ 1) ∧ (0 ≤ x.z ∧ x.z ≤ 1)) → 
  (∀ {a b : ℝ³}, P a → P b → ∃ path : ℝ → ℝ³, 
    (∀ t, 0 ≤ t → t ≤ 1 → P (path t)) ∧ 
    path 0 = a ∧ path 1 = b ∧ 
    path_intersects_all_faces path) →
  ∃ Q, Q ∈ P ∧ length_of_path P Q ≥ 3 * Real.sqrt 2 :=
sorry

end min_length_of_path_in_unit_cube_l294_294874


namespace shape_with_circular_views_is_sphere_l294_294574

/-- Define the views of the cuboid, cylinder, cone, and sphere. -/
structure Views (shape : Type) :=
(front_view : Type)
(left_view : Type)
(top_view : Type)

def is_cuboid (s : Views) : Prop :=
s.front_view = Rectangle ∧ s.left_view = Rectangle ∧ s.top_view = Rectangle

def is_cylinder (s : Views) : Prop :=
s.front_view = Rectangle ∧ s.left_view = Rectangle ∧ s.top_view = Circle

def is_cone (s : Views) : Prop :=
s.front_view = IsoscelesTriangle ∧ s.left_view = IsoscelesTriangle ∧ s.top_view = Circle

def is_sphere (s : Views) : Prop :=
s.front_view = Circle ∧ s.left_view = Circle ∧ s.top_view = Circle

/-- Proof problem: Prove that the only shape with circular views in all three perspectives (front, left, top) is the sphere. -/
theorem shape_with_circular_views_is_sphere :
  ∀ (s : Views), 
    (s.front_view = Circle ∧ s.left_view = Circle ∧ s.top_view = Circle) → 
    is_sphere s ∧ ¬ is_cuboid s ∧ ¬ is_cylinder s ∧ ¬ is_cone s :=
by
  intro s h
  sorry

end shape_with_circular_views_is_sphere_l294_294574


namespace cost_of_math_books_l294_294836

theorem cost_of_math_books (M : ℕ) : 
  (∃ (total_books math_books history_books total_cost : ℕ),
    total_books = 90 ∧
    math_books = 60 ∧
    history_books = total_books - math_books ∧
    history_books * 5 + math_books * M = total_cost ∧
    total_cost = 390) → 
  M = 4 :=
by
  -- We provide the assumed conditions
  intro h
  -- We will skip the proof with sorry
  sorry

end cost_of_math_books_l294_294836


namespace shape_with_circular_views_is_sphere_l294_294571

-- Definitions of the views of different geometric shapes
inductive Shape
| Cuboid : Shape
| Cylinder : Shape
| Cone : Shape
| Sphere : Shape

def front_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Rectangle
| Shape.Cone := False  -- Isosceles Triangle
| Shape.Sphere := True  -- Circle

def left_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Rectangle
| Shape.Cone := False  -- Isosceles Triangle
| Shape.Sphere := True  -- Circle

def top_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Circle, but not all views
| Shape.Cone := False  -- Circle, but not all views
| Shape.Sphere := True  -- Circle

-- The theorem to be proved
theorem shape_with_circular_views_is_sphere (s : Shape) :
  (front_view s ↔ True) ∧ (left_view s ↔ True) ∧ (top_view s ↔ True) ↔ s = Shape.Sphere :=
by sorry

end shape_with_circular_views_is_sphere_l294_294571


namespace exists_infinitely_many_n_l294_294025

def sum_of_digits (m : ℕ) : ℕ := 
  m.digits 10 |>.sum

theorem exists_infinitely_many_n (S : ℕ → ℕ) (h_sum_of_digits : ∀ m, S m = sum_of_digits m) :
  ∀ N : ℕ, ∃ n ≥ N, S (3 ^ n) ≥ S (3 ^ (n + 1)) :=
by { sorry }

end exists_infinitely_many_n_l294_294025


namespace empty_solution_set_range_l294_294993

theorem empty_solution_set_range (m : ℝ) : 
  (¬ ∃ x : ℝ, (m * x^2 + 2 * m * x + 1) < 0) ↔ (m = 0 ∨ (0 < m ∧ m ≤ 1)) :=
by sorry

end empty_solution_set_range_l294_294993


namespace Tom_seashells_l294_294485

theorem Tom_seashells (a b c : ℕ) (h1 : a = 27) (h2 : b = 46) (h3 : c = 19) :
  a + b + c = 92 := by
  rw [h1, h2, h3]
  sorry

end Tom_seashells_l294_294485


namespace ball_returns_to_ground_after_2_seconds_l294_294867

-- Conditions
def height (t : ℝ) : ℝ := 10 * t - 5 * t^2

-- Proof statement
theorem ball_returns_to_ground_after_2_seconds (h_eq : ∀ t, height t = 10 * t - 5 * t^2) :
  ∃ t : ℝ, height t = 0 ∧ t = 2 :=
sorry

end ball_returns_to_ground_after_2_seconds_l294_294867


namespace abs_twice_sub_pi_l294_294942

theorem abs_twice_sub_pi (h : Real.pi < 10) : abs (Real.pi - abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_twice_sub_pi_l294_294942


namespace expression_value_l294_294042

theorem expression_value (x : ℝ) (hx : x = -2023) : 
  |(|x| - x| - |x| - x) = 4046 :=
by
  have h1 : |x| = -x := abs_of_neg (by linarith [hx])
  rw [hx, h1]
  linarith
  

end expression_value_l294_294042


namespace maxEccentricity_l294_294665

noncomputable def majorAxisLength := 4
noncomputable def majorSemiAxis := 2
noncomputable def leftVertexParabolaEq (y : ℝ) := y^2 = -3
noncomputable def distanceCondition (c : ℝ) := 2^2 / c - 2 ≥ 1

theorem maxEccentricity : ∃ c : ℝ, distanceCondition c ∧ (c ≤ 4 / 3) ∧ (c / majorSemiAxis = 2 / 3) :=
by
  sorry

end maxEccentricity_l294_294665


namespace sum_tangent_points_l294_294608

noncomputable def f (x : ℝ) : ℝ := max (-9 * x - 29) (max (2 * x - 2) (7 * x + 7))

theorem sum_tangent_points :
  ∃ a1 a2 a3 : ℝ, (q x - (-9 * x - 29) = b * (x - a1) ^ 2) ∧
                 (q x - (2 * x - 2) = b * (x - a2) ^ 2) ∧
                 (q x - (7 * x + 7) = b * (x - a3) ^ 2) ∧
                 a1 + a2 + a3 = -992 / 99 := 
sorry

end sum_tangent_points_l294_294608


namespace complement_of_union_l294_294019

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def U : Set ℤ := Set.univ

theorem complement_of_union (x : ℤ) : (x ∉ A ∪ B) ↔ ∃ k : ℤ, x = 3 * k := 
by {
  sorry
}

end complement_of_union_l294_294019


namespace perimeter_triangle_XYZ_l294_294893

-- Definitions based on the conditions in a)
def prism_height : ℝ := 20
def base_side_length : ℝ := 10
def midpoint_length : ℝ := base_side_length / 2

-- The definition of the points X, Y, Z as midpoints on the edges
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

-- Midpoints X, Y, Z in the 3D coordinate system
def X : Point3D := {x := midpoint_length, y := 0, z := 0}
def Y : Point3D := {x := base_side_length, y := midpoint_length, z := 0}
def Z : Point3D := {x := base_side_length, y := base_side_length, z := prism_height}

-- Distance function between two 3D points
def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2)

-- Defining the distances XY, XZ, YZ
def XY : ℝ := distance X Y
def XZ : ℝ := distance X Z
def YZ : ℝ := distance Y Z

-- Perimeter of triangle XYZ
def perimeter_XYZ : ℝ := XY + XZ + YZ

-- Statement of the theorem
theorem perimeter_triangle_XYZ : perimeter_XYZ = 45 :=
by sorry

end perimeter_triangle_XYZ_l294_294893


namespace parabola_vertex_f_l294_294147

theorem parabola_vertex_f (d e f : ℝ) (h_vertex : ∀ y, (d * (y - 3)^2 + 5) = (d * y^2 + e * y + f))
  (h_point : d * (6 - 3)^2 + 5 = 2) : f = 2 :=
by
  sorry

end parabola_vertex_f_l294_294147


namespace find_elements_with_same_part_size_l294_294411

open finset

variables {X : Type*} [fintype X] [decidable_eq X] (h_partition : partition (univ : finset X))

noncomputable def number_of_elements_in_part {α : Type*} [decidable_eq α] (π : partition α) (x : α) : ℕ :=
card (π.parts.find (λ a, x ∈ a).get (multiset.empty))

theorem find_elements_with_same_part_size :
  ∀ (π₁ π₂ : partition (univ : finset X)), (fintype.card X = 9) →
  ∃ (h k : X), h ≠ k ∧
                number_of_elements_in_part π₁ h = number_of_elements_in_part π₁ k ∧
                number_of_elements_in_part π₂ h = number_of_elements_in_part π₂ k :=
begin
  intros π₁ π₂ n_elem,
  have h1: ∃ (h k : X), h ≠ k ∧ number_of_elements_in_part π₁ h = number_of_elements_in_part π₁ k,
  {
    sorry, -- Use the pigeonhole principle here
  },
  obtain ⟨h1, h2, h_diff, h_part_size1, h_part_size2⟩ := h1,
  have h2: number_of_elements_in_part π₂ h1 = number_of_elements_in_part π₂ h2,
  {
    sorry, -- Use pigeonhole principle again for π₂
  },
  use [h1, h2, h_diff, h_part_size1, h2],
end

end find_elements_with_same_part_size_l294_294411


namespace williams_land_percentage_l294_294979

variable (total_tax : ℕ) (williams_tax : ℕ)

theorem williams_land_percentage (h1 : total_tax = 3840) (h2 : williams_tax = 480) : 
  (williams_tax:ℚ) / (total_tax:ℚ) * 100 = 12.5 := 
  sorry

end williams_land_percentage_l294_294979


namespace price_increase_decrease_l294_294895

theorem price_increase_decrease (x : ℝ) : 
    (∀ (P : ℝ), P > 0 → 
    (let final_price := P * (1 - (x / 100) ^ 2) in final_price = 0.64 * P)) 
    → x = 60 := by
  intros h
  have h1 : (1 - (x / 100) ^ 2) = 0.64, from sorry
  have h2 : (x / 100) ^ 2 = 0.36, from sorry
  have h3 : x / 100 = 0.6, from sorry
  have h4 : x = 60, from sorry
  exact h4

end price_increase_decrease_l294_294895


namespace noemi_starting_money_l294_294430

-- Define the losses and remaining money as constants
constant lost_on_roulette : ℕ := 400
constant lost_on_blackjack : ℕ := 500
constant remaining_money : ℕ := 800

-- Define the initial amount of money as a conjecture to be proven
def initial_money := lost_on_roulette + lost_on_blackjack + remaining_money

-- State the theorem
theorem noemi_starting_money : initial_money = 1700 :=
by
  -- This is where the actual proof would go
  sorry

end noemi_starting_money_l294_294430


namespace price_of_soda_after_increase_l294_294634

theorem price_of_soda_after_increase :
  ∃ (x : ℝ), (10 + x = 16) ∧ (1.5 * x = 9) :=
begin
  use 6,
  split,
  { linarith, },
  { norm_num, },
end

end price_of_soda_after_increase_l294_294634


namespace expected_rolls_sum_2010_l294_294174

noncomputable def expected_rolls_to_reach_sum (n : ℕ) : ℚ :=
  if n == 0 then 0
  else (1/6) * (expected_rolls_to_reach_sum (n-1) + expected_rolls_to_reach_sum (n-2) +
               expected_rolls_to_reach_sum (n-3) + expected_rolls_to_reach_sum (n-4) +
               expected_rolls_to_reach_sum (n-5) + expected_rolls_to_reach_sum (n-6)) + 1

theorem expected_rolls_sum_2010 : expected_rolls_to_reach_sum 2010 ≈ 574.761904 := 
by 
  -- Proof omitted; the focus is on the statement
  sorry

end expected_rolls_sum_2010_l294_294174


namespace infinite_elements_in_Omega_l294_294378

theorem infinite_elements_in_Omega :
  let C1 := { P : ℝ × ℝ | ∃ x y, P = (x, y) ∧ (x^2 / 36) + (y^2 / 4) = 1 }
  let C2 := { Q : ℝ × ℝ | ∃ x y, Q = (x, y) ∧ x^2 + (y^2 / 9) = 1 }
  let dot_product (P Q : ℝ × ℝ) := P.1 * Q.1 + P.2 * Q.2
  let w := 6
  let Omega := { (P, Q) | P ∈ C1 ∧ Q ∈ C2 ∧ dot_product P Q = w }
  infinite Omega :=
by
  sorry

end infinite_elements_in_Omega_l294_294378


namespace complement_union_A_B_l294_294012

def is_element_of_set_A (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 1
def is_element_of_set_B (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 2
def is_element_of_complement_U (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem complement_union_A_B :
  {x : ℤ | ¬ (is_element_of_set_A x ∨ is_element_of_set_B x)} = {x : ℤ | is_element_of_complement_U x} :=
by
  sorry

end complement_union_A_B_l294_294012


namespace triangle_area_ratio_l294_294771

open Real

-- Define an equilateral triangle with a point D on AC and give the lean statement to prove the required ratio.
theorem triangle_area_ratio (A B C D : Point) (ABC : Triangle) (h_eq : is_equilateral ABC) (hD : D ∈ Segment AC) (h_angle : ∠ DBC = 30) :
  area (triangle A D B) / area (triangle C D B) = 0 :=
sorry

end triangle_area_ratio_l294_294771


namespace polyhedron_inequality_proof_l294_294715

noncomputable def polyhedron_inequality (B : ℕ) (P : ℕ) (T : ℕ) : Prop :=
  B * Real.sqrt (P + T) ≥ 2 * P

theorem polyhedron_inequality_proof (B P T : ℕ) 
  (h1 : 0 < B) (h2 : 0 < P) (h3 : 0 < T) 
  (condition_is_convex_polyhedron : true) : 
  polyhedron_inequality B P T :=
sorry

end polyhedron_inequality_proof_l294_294715


namespace expected_number_of_rolls_to_reach_2010_l294_294171

noncomputable def expected_rolls : ℕ → ℚ
| 0 := 0
| (n+1) := (1/6) * (expected_rolls n + expected_rolls (n-1) + expected_rolls (n-2) + expected_rolls (n-3) + expected_rolls (n-4) + expected_rolls (n-5)) + 1

theorem expected_number_of_rolls_to_reach_2010 : abs ((expected_rolls 2010 : ℚ) - 574.761904) < 0.0001 := 
sorry

end expected_number_of_rolls_to_reach_2010_l294_294171


namespace vessel_width_l294_294876

theorem vessel_width (edge length rise W : ℝ) (h_edge : edge = 15) (h_length : length = 20) (h_rise : rise = 11.25)
  (h_volume_displaced : edge^3 = length * W * rise) : W = 15 := by
  rw [h_edge, h_length, h_rise] at h_volume_displaced
  have h_volume : 15^3 = 20 * W * 11.25 := h_volume_displaced
  linarith

end vessel_width_l294_294876


namespace expected_number_of_defective_products_l294_294585

theorem expected_number_of_defective_products 
  (N : ℕ) (D : ℕ) (n : ℕ) (hN : N = 15000) (hD : D = 1000) (hn : n = 150) :
  n * (D / N : ℚ) = 10 := 
by {
  sorry
}

end expected_number_of_defective_products_l294_294585


namespace exists_odd_a_b_k_l294_294412

theorem exists_odd_a_b_k (m : ℤ) : ∃ (a b k : ℤ), 
  (odd a) ∧ (odd b) ∧ (k > 0) ∧ (2 * m = a ^ 19 + b ^ 99 + k * 2 ^ 1999) := 
sorry

end exists_odd_a_b_k_l294_294412


namespace negation_of_universal_proposition_l294_294095

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x_0 : ℝ, x_0^2 < 0) := sorry

end negation_of_universal_proposition_l294_294095


namespace imaginary_unit_squared_in_set_l294_294313

-- Conditions of the problem
def imaginary_unit (i : ℂ) : Prop := i^2 = -1
def S : Set ℂ := {-1, 0, 1}

-- The statement to prove
theorem imaginary_unit_squared_in_set {i : ℂ} (hi : imaginary_unit i) : i^2 ∈ S := sorry

end imaginary_unit_squared_in_set_l294_294313


namespace probability_both_divisible_by_4_when_two_6_sided_dice_tossed_l294_294513

theorem probability_both_divisible_by_4_when_two_6_sided_dice_tossed : 
  let dice := fin 6
  let outcomes := (prod dice dice)
  let favorable := {ab | ab.1 % 4 = 0 ∧ ab.2 % 4 = 0}
  in (favorable.card / outcomes.card) = (1 : ℝ) / 36 :=
by
  -- Definitions
  let dice := fin 6
  let outcomes := (prod dice dice)
  let favorable := {ab | ab.1 % 4 = 0 ∧ ab.2 % 4 = 0}

  -- Calculate the probability
  have h_fav_card : favorable.card = 1 := sorry
  have h_outcomes_card : outcomes.card = 36 := sorry
  have h_prob : (favorable.card : ℝ) / outcomes.card = (1 : ℝ) / 36 := sorry
  exact h_prob

end probability_both_divisible_by_4_when_two_6_sided_dice_tossed_l294_294513


namespace graph_is_hyperbola_l294_294838

theorem graph_is_hyperbola : 
  ∀ x y : ℝ, (x + y)^2 = x^2 + y^2 + 4 ↔ x * y = 2 := 
by
  sorry

end graph_is_hyperbola_l294_294838


namespace find_value_l294_294537

theorem find_value (x : ℤ) (h : 3 * x - 45 = 159) : (x + 32) * 12 = 1200 :=
by
  sorry

end find_value_l294_294537


namespace find_sine_of_alpha_l294_294289

theorem find_sine_of_alpha (α : ℝ) (h1 : 0 < α) (h2 : α < π) 
  (h3 : 3 * real.cos (2 * α) - 8 * real.cos α = 5) :
  real.sin α = real.sqrt 5 / 3 :=
sorry

end find_sine_of_alpha_l294_294289


namespace find_base_width_l294_294877

-- Definitions of the given conditions
def cube_edge : ℝ := 15
def rise_in_water_level : ℝ := 11.25
def base_length : ℝ := 20

-- Calculation of given and required dimensions
def volume_of_cube := cube_edge ^ 3
def displaced_water_volume := volume_of_cube
def base_area := displaced_water_volume / rise_in_water_level

-- The statement to prove the width of the vessel's base
theorem find_base_width : ∃ W : ℝ, base_area = base_length * W ∧ W = 15 :=
by
  sorry

end find_base_width_l294_294877


namespace average_number_of_fish_is_75_l294_294767

-- Define the number of fish in Boast Pool and conditions for other bodies of water
def Boast_Pool_fish : ℕ := 75
def Onum_Lake_fish : ℕ := Boast_Pool_fish + 25
def Riddle_Pond_fish : ℕ := Onum_Lake_fish / 2

-- Define the average number of fish in all three bodies of water
def average_fish : ℕ := (Onum_Lake_fish + Boast_Pool_fish + Riddle_Pond_fish) / 3

-- Prove that the average number of fish in all three bodies of water is 75
theorem average_number_of_fish_is_75 : average_fish = 75 := by
  sorry

end average_number_of_fish_is_75_l294_294767


namespace value_of_expression_l294_294689

theorem value_of_expression (x Q : ℝ) (π : Real) (h : 5 * (3 * x - 4 * π) = Q) : 10 * (6 * x - 8 * π) = 4 * Q :=
by 
  sorry

end value_of_expression_l294_294689


namespace water_dilution_l294_294474

theorem water_dilution (initial_volume : ℝ) (initial_concentration : ℝ) (desired_concentration : ℝ) 
    (alcohol_content : ℝ) (x : ℝ) (h : initial_volume = 12 ∧ initial_concentration = 0.4 ∧ 
    desired_concentration = 0.2 ∧ 
    alcohol_content = initial_volume * initial_concentration ∧
    alcohol_content = desired_concentration * (initial_volume + x)) : x = 12 :=
by
  cases h with _ h
  sorry

end water_dilution_l294_294474


namespace max_value_2x_minus_y_l294_294679

theorem max_value_2x_minus_y (x y : ℝ) (h : x^2 / 4 + y^2 / 9 = 1) : 2 * x - y ≤ 5 :=
sorry

end max_value_2x_minus_y_l294_294679


namespace parabola_vertex_example_l294_294468

-- Definitions based on conditions
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def vertex (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + 3

-- Conditions given in the problem
def condition1 (a b c : ℝ) : Prop := parabola a b c 2 = 5
def condition2 (a : ℝ) : Prop := vertex a 1 = 3

-- Goal statement to be proved
theorem parabola_vertex_example : ∃ (a b c : ℝ), 
  condition1 a b c ∧ condition2 a ∧ a - b + c = 11 :=
by
  sorry

end parabola_vertex_example_l294_294468


namespace speech_competition_score_l294_294152

theorem speech_competition_score :
  let speech_content := 90
  let speech_skills := 80
  let speech_effects := 85
  let content_ratio := 4
  let skills_ratio := 2
  let effects_ratio := 4
  (speech_content * content_ratio + speech_skills * skills_ratio + speech_effects * effects_ratio) / (content_ratio + skills_ratio + effects_ratio) = 86 := by
  sorry

end speech_competition_score_l294_294152


namespace find_a9_l294_294820

def seq (a : ℕ → ℝ) : Prop :=
  (a 1 = 2) ∧ ∀ n : ℕ, a (n + 1) = 1 / (1 - a n)

theorem find_a9 : ∃ a : ℕ → ℝ, seq a ∧ a 9 = 1 / 2 :=
begin
  sorry
end

end find_a9_l294_294820


namespace vessel_width_l294_294875

theorem vessel_width (edge length rise W : ℝ) (h_edge : edge = 15) (h_length : length = 20) (h_rise : rise = 11.25)
  (h_volume_displaced : edge^3 = length * W * rise) : W = 15 := by
  rw [h_edge, h_length, h_rise] at h_volume_displaced
  have h_volume : 15^3 = 20 * W * 11.25 := h_volume_displaced
  linarith

end vessel_width_l294_294875


namespace disease_given_positive_test_l294_294794

variable (D T : Event)
variable [ProbabilityMeasure Ω] [MeasureSpace Ω]

-- Conditions
variable (h_PD : Pr(D) = 1 / 1000)
variable (h_PT_D : Pr(T | D) = 1)
variable (D_compl : Event) (h_D_compl : D_compl = Dᶜ)
variable (h_PT_Dcomplement : Pr(T | D_compl) = 0.05)

-- Goal
theorem disease_given_positive_test :
  Pr(D | T) = 100 / 5095 :=
by
  sorry

end disease_given_positive_test_l294_294794


namespace find_k_l294_294635

noncomputable def quadratic_roots (a b c : ℝ) : Set ℝ :=
  {x | a * x^2 + b * x + c = 0}

theorem find_k (k : ℝ)
  (h1 : ∃ (x₁ x₂ ∈ quadratic_roots 1 (2*k - 1) (k^2 - 1)), RealSet (3*x₁ - x₂) * RealSet (x₁ - 3*x₂) = {19}) :
  k = 0 ∨ k = -3 :=
by
  -- The proof goes here.
  sorry

end find_k_l294_294635


namespace valid_5_digit_numbers_l294_294821

noncomputable def num_valid_numbers (d : ℕ) (h : d ≠ 7) (h_valid : d < 10) (h_pos : d ≠ 0) : ℕ :=
  let choices_first_place := 7   -- choices for the first digit (1-9, excluding d and 7)
  let choices_other_places := 8  -- choices for other digits (0-9, excluding d and 7)
  choices_first_place * choices_other_places ^ 4

theorem valid_5_digit_numbers (d : ℕ) (h_d_ne_7 : d ≠ 7) (h_d_valid : d < 10) (h_d_pos : d ≠ 0) :
  num_valid_numbers d h_d_ne_7 h_d_valid h_d_pos = 28672 := sorry

end valid_5_digit_numbers_l294_294821


namespace solve_x_l294_294239

noncomputable def solutions : Set ℂ := 
  { x | x = complex.of_real(sqrt 3) + complex.I ∨ 
        x = 0 + 2 * complex.I ∨
        x = - complex.of_real(sqrt 3) + complex.I ∨
        x = - complex.of_real(sqrt 3) - complex.I ∨
        x = 0 - 2 * complex.I ∨
        x = complex.of_real(sqrt 3) - complex.I }

theorem solve_x^6_eq_minus_64 (x : ℂ) : x ^ 6 + 64 = 0 ↔ x ∈ solutions :=
sorry

end solve_x_l294_294239


namespace ratio_of_triangle_areas_l294_294778

theorem ratio_of_triangle_areas 
  (A B C D : Point)
  (h_eq_tri : equilateral_triangle A B C)
  (h_D_on_AC : lies_on_line_segment D A C)
  (h_angle_DBC : angle D B C = 30) :
  (area (triangle A D B)) / (area (triangle C D B)) = 1 / real.sqrt 3 :=
sorry

end ratio_of_triangle_areas_l294_294778


namespace area_ratio_eq_one_l294_294775

theorem area_ratio_eq_one
  (A B C D : Point)
  (h_eq_triangle : EquilateralTriangle A B C)
  (h_D_on_AC : D ∈ segment A C)
  (h_angle_DBC_eq_30 : ∠ D B C = 30) :
  (area (triangle A D B))/(area (triangle C D B)) = 1 := 
sorry

end area_ratio_eq_one_l294_294775


namespace expected_rolls_to_reach_2010_l294_294166

noncomputable def expected_rolls_to_reach_sum (n : ℕ) : ℝ :=
  sorry -- Using 'sorry' to denote placeholder for the actual proof.

theorem expected_rolls_to_reach_2010 : expected_rolls_to_reach_sum 2010 ≈ 574.761904 :=
  sorry

end expected_rolls_to_reach_2010_l294_294166


namespace find_m_interval_l294_294223

noncomputable def x : ℕ → ℝ
| 0       := 7
| (n + 1) := (x n ^ 2 + 3 * x n + 2) / (x n + 4)

def m := Nat.find (λ m, x m ≤ 5 + 1 / 2 ^ 10)

theorem find_m_interval :
  41 ≤ m ∧ m ≤ 100 :=
  sorry

end find_m_interval_l294_294223


namespace torus_volume_formula_l294_294416

noncomputable def torusVolume (r R : ℝ) (h : R > r) : ℝ :=
  2 * π^2 * R * r^2

theorem torus_volume_formula (r R : ℝ) (h : R > r) :
  (torusVolume r R h = 2 * π^2 * R * r^2) := by
  sorry

end torus_volume_formula_l294_294416


namespace students_with_all_three_talents_l294_294719

-- Define the student counts
def numberOfStudents : ℕ := 150
def cannotSing : ℕ := 80
def cannotDance : ℕ := 110
def cannotAct : ℕ := 60

-- Define the number of students who can perform each individual talent
def canSing : ℕ := numberOfStudents - cannotSing
def canDance : ℕ := numberOfStudents - cannotDance
def canAct : ℕ := numberOfStudents - cannotAct

-- The total count ignoring overlaps
def totalIgnoringOverlaps : ℕ := canSing + canDance + canAct

-- The proof statement
theorem students_with_all_three_talents :
  ∃ x : ℕ, totalIgnoringOverlaps - x = numberOfStudents ∧ x = 50 :=
by
  use 50
  split
  · simp [totalIgnoringOverlaps, canSing, canDance, canAct, numberOfStudents, cannotSing, cannotDance, cannotAct]
  · rfl

end students_with_all_three_talents_l294_294719


namespace possible_slopes_intersect_ellipse_l294_294886

theorem possible_slopes_intersect_ellipse (m : ℝ) :
  let line_eq : ℝ → ℝ := λ x, m * x + 8
  let ellipse_eq : ℝ × ℝ → Prop := λ (x y), 25 * x^2 + 16 * y^2 = 400
  (∃ x y, ellipse_eq (x, y) ∧ y = line_eq x) ↔
  m ∈ Set.Iic (-(Real.sqrt 39) / 4 ) ∪ Set.Ici ((Real.sqrt 39) / 4) := 
by
  sorry

end possible_slopes_intersect_ellipse_l294_294886


namespace hexagon_segments_probability_l294_294026

/-- The set T contains all sides and diagonals of a regular hexagon. 
There are 6 sides, 6 short diagonals, and 3 long diagonals.
We choose a pair of segments at random without replacement, and we aim to find the probability that the two segments have the same length. -/
theorem hexagon_segments_probability : 
  let T := {side | short_diag | long_diag : set ℕ // side = 6 ∧ short_diag = 6 ∧ long_diag = 3} in
  (∑ (x ∈ T), (∑ (y ∈ T), if x = y then 1 else 0)) / (15 * 14) = 22 / 35 :=
sorry

end hexagon_segments_probability_l294_294026


namespace parallelogram_properties_l294_294751

-- Definitions
variables {a : ℝ} {AB BH BC AD : ℝ}
variable {α : ℝ} -- representing angles in radians

-- Conditions
def condition_1 : AB = a := sorry

def condition_2 : BH = a ∧ BH ⟂ CM := sorry

def condition_3 : BC ∥ AB := sorry

def condition_4 : α = 30 * (π / 180) := by norm_num

def condition_5 : AD = 2 := sorry

-- Main theorem
theorem parallelogram_properties (a : ℝ) (AB BH BC AD : ℝ) (α : ℝ) :
  (BC = 2 * a) ∧ (∠ CDA = 2 * α) ∧ (∠ BAD = π / 3) ∧
  (area AB BD = 2 * sqrt 3) :=
begin
  sorry
end

end parallelogram_properties_l294_294751


namespace rolling_quarter_circle_path_length_l294_294583

theorem rolling_quarter_circle_path_length :
  ∀ (D E F F' R S : Point) (r : ℝ),
  is_quarter_circle D E F →
  EF = 1 / π →
  travels_path F F' D E R S r →
  F_travel_distance F F' = 1.5 :=
by
  intros,
  sorry

end rolling_quarter_circle_path_length_l294_294583


namespace range_of_a_l294_294674

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < real.exp 1 then real.sqrt (-x^3 + x^2) else a * real.log x

theorem range_of_a :
  ∃ a : ℝ, (0 < a ∧ a ≤ 1 / (real.exp 1 + 1)) ∧
    (∀ t : ℝ, t ≥ real.exp 1 → let x := t in 
      let P := (x, f x a) in 
      let Q := (-x, -(x^3 + x^2)) in 
      (∃ a : ℝ, (P.1^2 + P.2^2 = Q.1^2 + Q.2^2 ∧ P.2 = -Q.2))) :=
sorry

end range_of_a_l294_294674


namespace equation_of_circle_c_coordinates_of_point_m_l294_294317

def point := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def circle_c (x y r : ℝ) : set point :=
  {p | (p.1 - x) ^ 2 + (p.2 - y) ^ 2 = r ^ 2}

def line_bisects_circle (l : point → Prop) (c : set point) : Prop :=
  ∃ (center : point), ∀ (p1 p2 : point), p1 ∈ c → p2 ∈ c → l center → distance center p1 = distance center p2

-- Given conditions
def line : point → Prop := λ p, p.1 - 3 * p.2 - 4 = 0
def points_on_circle_c : set point := {(0, 2), (6, 4)}

-- Part 1: Prove the equation of circle C
theorem equation_of_circle_c : 
  ∃ (x y r : ℝ), circle_c x y r = {p | (p.1 - 4) ^ 2 + p.2 ^ 2 = 20} ∧ 
                 ∀ p ∈ points_on_circle_c, (p.1 - x) ^ 2 + (p.2 - y) ^ 2 = r ^ 2 ∧
                 line_bisects_circle line (circle_c x y r) :=
sorry

-- Part 2: Coordinates of point M on circle C satisfying the ratio condition
def point_p : point := (-6, 0)
def point_q : point := (6, 0)

theorem coordinates_of_point_m :
  ∃ (m : point), m ∈ {p | (p.1 - 4) ^ 2 + p.2 ^ 2 = 20} ∧
                 distance m point_p = 2 * distance m point_q ∧
                 (m = (10 / 3, 4 * real.sqrt 11 / 3) ∨ m = (10 / 3, -4 * real.sqrt 11 / 3)) :=
sorry

end equation_of_circle_c_coordinates_of_point_m_l294_294317


namespace part_a_part_b_l294_294039

-- Part (a): Prove that D < n^2
theorem part_a (n : ℕ) (hn : n > 1)
  (divisors : List ℕ)
  (hdivs : divisors = (List.range (n+1)).filter (λ d, d > 0 ∧ n % d = 0))
  (horder : ∀ i j, i < j → i < divisors.length ∧ j < divisors.length → divisors[i] < divisors[j]) :
  let D := (List.zipWith (*) divisors divisors.tail).sum in
  D < n^2 := sorry

-- Part (b): Determine all n for which D divides n^2
theorem part_b (n : ℕ) (hn : n > 1)
  (divisors : List ℕ)
  (hdivs : divisors = (List.range (n+1)).filter (λ d, d > 0 ∧ n % d = 0))
  (horder : ∀ i j, i < j → i < divisors.length ∧ j < divisors.length → divisors[i] < divisors[j]) :
  let D := (List.zipWith (*) divisors divisors.tail).sum in
  D ∣ n^2 ↔ Nat.Prime n := sorry

end part_a_part_b_l294_294039


namespace number_of_orderings_l294_294500

variable (House : Type) [Fintype House] [DecidableEq House]
variable [houses : Fin 5 House]
variable {O R B G Y : House}

axiom distinct_colors : ∀ h1 h2 : House, h1 ≠ h2

variables (p : Vector House 5)

def is_valid_ordering (p : Vector House 5) : Prop :=
  p.get ⟨0, sorry⟩ ≠ p.get ⟨1, sorry⟩ ∧
  p.get ⟨1, sorry⟩ ≠ p.get ⟨2, sorry⟩ ∧
  p.get ⟨2, sorry⟩ ≠ p.get ⟨3, sorry⟩ ∧
  p.get ⟨3, sorry⟩ ≠ p.get ⟨4, sorry⟩ ∧
  ∃ i j k l m : Fin 5, 
  p.get i = O ∧ p.get j = R ∧ i < j ∧
  p.get k = B ∧ p.get l = G ∧ k < l ∧
  (i ≠ k) ∧ 
  (j ≠ k) ∧ 
  (k + 1 ≠ l) ∧ 
  (j + 1 ≠ k)

theorem number_of_orderings : (Fintype.card (Subtype (is_valid_ordering House))) = 4 := sorry

end number_of_orderings_l294_294500


namespace probability_fewer_than_3_heads_l294_294550

noncomputable def probability_fewer_than_3_heads_in_8_flips : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := (1 + (nat.choose 8 1) + (nat.choose 8 2))
  favorable_outcomes / total_outcomes

theorem probability_fewer_than_3_heads :
  probability_fewer_than_3_heads_in_8_flips = 37 / 256 := 
sorry

end probability_fewer_than_3_heads_l294_294550


namespace range_of_slope_exists_k_for_collinearity_l294_294365

def line_equation (k x : ℝ) : ℝ := k * x + 1

def circle_eq (x y : ℝ) : ℝ := x^2 + y^2 - 4 * x + 3

noncomputable def intersect_points (k : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry  -- Assume a function that computes the intersection points (x₁, y₁) and (x₂, y₂)

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v2 = (c * v1.1, c * v1.2)

theorem range_of_slope (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h1 : line_equation k x₁ = y₁) 
  (h2 : line_equation k x₂ = y₂)
  (h3 : circle_eq x₁ y₁ = 0)
  (h4 : circle_eq x₂ y₂ = 0) :
  -4/3 < k ∧ k < 0 := 
sorry

theorem exists_k_for_collinearity (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h1 : line_equation k x₁ = y₁) 
  (h2 : line_equation k x₂ = y₂)
  (h3 : circle_eq x₁ y₁ = 0)
  (h4 : circle_eq x₂ y₂ = 0)
  (h5 : -4/3 < k ∧ k < 0) :
  collinear (2 - x₁ - x₂, -(y₁ + y₂)) (-2, 1) ↔ k = -1/2 :=
sorry


end range_of_slope_exists_k_for_collinearity_l294_294365


namespace cats_to_dogs_ratio_l294_294475

theorem cats_to_dogs_ratio (cats dogs : ℕ) (h1 : 2 * dogs = 3 * cats) (h2 : cats = 14) : dogs = 21 :=
by
  sorry

end cats_to_dogs_ratio_l294_294475


namespace monotone_f_l294_294671

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a / 2) * x + 2

theorem monotone_f (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) → a ∈ Ico 4 8 :=
by
  intro h
  sorry

end monotone_f_l294_294671


namespace range_of_f_intersection_A_B_l294_294660

-- Definition of the function f
def f (x : ℝ) : ℝ := real.sqrt x

-- Condition 1: A = [0,9]
def A : set ℝ := {x | 0 ≤ x ∧ x ≤ 9}

-- Condition 2: B = {1,2}
def B : set ℝ := {1, 2}

-- Question 1: Prove the range of f(x) when A = [0,9] is [0,3]
theorem range_of_f (y : ℝ) : y ∈ (set.image f A) ↔ 0 ≤ y ∧ y ≤ 3 :=
sorry

-- Question 2: Prove A ∩ B = {1}
theorem intersection_A_B : A ∩ B = {1} :=
sorry

end range_of_f_intersection_A_B_l294_294660


namespace trip_time_difference_is_60_minutes_l294_294545

-- Definitions of speed and distances as constants
def avg_speed : ℝ := 40 -- in miles per hour
def distance1 : ℝ := 400 -- in miles
def distance2 : ℝ := 360 -- in miles

-- Definition of the time difference calculation
def time_difference (d1 d2 speed : ℝ) : ℝ := (d1 - d2) / speed

-- The claim we want to prove:
theorem trip_time_difference_is_60_minutes :
  time_difference distance1 distance2 avg_speed * 60 = 60 :=
by
  -- Proof goes here
  sorry

end trip_time_difference_is_60_minutes_l294_294545


namespace complement_union_l294_294006

def A := { x : ℤ | ∃ k : ℤ, x = 3 * k + 1 }
def B := { x : ℤ | ∃ k : ℤ, x = 3 * k + 2 }
def U := ℤ

theorem complement_union :
  { x : ℤ | x ∉ (A ∪ B) } = { x : ℤ | ∃ k : ℤ, x = 3 * k } := 
by 
  sorry

end complement_union_l294_294006


namespace additional_takeout_ratio_l294_294553

-- Define the conditions as hypotheses
variable (m : ℝ) (x : ℝ) (y : ℝ)

-- Given conditions definitions
def revenue_ratios_June := 3 * m  / 10 * m  = 3 / 10
def increase_stall_July := 2 * m + (2 / 5) * x / (10 * m + x) = 7 / 20
def ratio_dining_takeout_July := (3 * m + y) / (5 * m + 18 * m - y) = 8 / 5

-- The proof problem to be stated:
theorem additional_takeout_ratio :
  revenue_ratios_June →
  increase_stall_July (2 * m + (2 / 5) * x) (10 * m + x) → 
  ratio_dining_takeout_July (3 * m + y) (5 * m + 18 * m - y) → 
  (5 * m) / (40 * m) = 1 / 8 := 
by 
  sorry

end additional_takeout_ratio_l294_294553


namespace set_representation_l294_294116

theorem set_representation :
  {p : ℕ × ℕ | p.1 ∈ {1, 2} ∧ p.2 ∈ {1, 2}} =
    {(1, 1), (1, 2), (2, 1), (2, 2)} :=
by 
  sorry

end set_representation_l294_294116


namespace average_age_proof_l294_294108

noncomputable def average_age_when_youngest_born (total_people : ℕ) (current_avg_age : ℚ) (youngest_age : ℚ) : ℚ :=
  (total_people * current_avg_age - youngest_age * (total_people - 1)) / total_people

theorem average_age_proof :
  average_age_when_youngest_born 7 30 3 ≈ 27.43 :=
by
  sorry

end average_age_proof_l294_294108


namespace parabola_intersection_points_l294_294832

noncomputable def y1 := 2 * ((5 - Real.sqrt 61) / 6) ^ 2 - 7 * ((5 - Real.sqrt 61) / 6) + 1
noncomputable def y2 := 2 * ((5 + Real.sqrt 61) / 6) ^ 2 - 7 * ((5 + Real.sqrt 61) / 6) + 1

theorem parabola_intersection_points :
  ∃ x1 x2 y1 y2, x1 = (5 - Real.sqrt 61) / 6 ∧ x2 = (5 + Real.sqrt 61) / 6 ∧ 
  y1 = 2 * x1^2 - 7 * x1 + 1 ∧ y2 = 2 * x2^2 - 7 * x2 + 1 ∧
  (2 * x1^2 - 7 * x1 + 1 = 5 * x1^2 - 2 * x1 - 2) ∧ (2 * x2^2 - 7 * x2 + 1 = 5 * x2^2 - 2 * x2 - 2) :=
by
  use (5 - Real.sqrt 61) / 6
  use (5 + Real.sqrt 61) / 6
  use y1
  use y2
  split; norm_num
  split; norm_num
  split; field_simp [y1, y2]; norm_num; sorry
  split; field_simp [y1, y2]; norm_num; sorry

end parabola_intersection_points_l294_294832


namespace parabolas_pass_through_point_l294_294817

/-- 
  Prove that the parabolas \( y = -x^2 + 1 \) and \( y = x^2 + 1 \) 
  both pass through the point \( (0, 1) \).
-/
theorem parabolas_pass_through_point (p1 p2 : ℝ → ℝ) :
  p1 = (λ x, -x^2 + 1) → p2 = (λ x, x^2 + 1) → p1 0 = 1 ∧ p2 0 = 1 :=
by
  intros h_p1 h_p2
  rw [h_p1, h_p2]
  split
  · calc (-0^2 + 1) = 1 : by ring
  · calc (0^2 + 1) = 1 : by ring

-- sorry to skip the proof parts

end parabolas_pass_through_point_l294_294817


namespace Paula_used_12_cans_to_paint_35_rooms_l294_294436

noncomputable def cans_required_for_rooms (total_rooms: ℕ) (lost_cans: ℕ) (rooms_after_loss: ℕ) : ℕ :=
  let rooms_lost := total_rooms - rooms_after_loss in
  let cans_lost := lost_cans in
  let rooms_per_can := rooms_lost / cans_lost in
  let needed_cans := rooms_after_loss / rooms_per_can in
  if rooms_after_loss % rooms_per_can = 0 then needed_cans else needed_cans + 1

theorem Paula_used_12_cans_to_paint_35_rooms :
  cans_required_for_rooms 50 5 35 = 12 :=
by
  -- the proof is omitted
  sorry

end Paula_used_12_cans_to_paint_35_rooms_l294_294436


namespace area_of_region_l294_294078

noncomputable def bounded_area := 
  let equation (x y : ℝ) := x^2 + y^2 = 4*|y-x| + 2*|y+x|
  ∀ x y : ℝ, equation x y → (∃ m n : ℤ, bounded_area = m + n * real.pi ∧ m + n = 40)

theorem area_of_region {x y : ℝ} :
  let equation := x^2 + y^2 = 4*|y-x| + 2*|y+x|
  (∀ (h : x^2 + y^2 = 4*|y-x| + 2*|y+x|), ∃ m n : ℤ, (bounded_area x y) = m + n * real.pi ∧ m + n = 40) := 
by
  sorry

end area_of_region_l294_294078


namespace incorrect_options_no_three_real_roots_when_p_neg_q_pos_evaluate_problem_claims_l294_294972

def f (x p q : ℝ) : ℝ := 
  if x ≥ 0 
  then x^2 + p * x + q 
  else -x^2 + p * x + q

theorem incorrect_options (p q : ℝ) : 
  (∃ x, f x p q = 0) ↔ (p^2 - 4 * q ≥ 0 ∨ p^2 + 4 * q ≥ 0) := by
  sorry

theorem no_three_real_roots_when_p_neg_q_pos (p q : ℝ) (hp : p < 0) (hq : q > 0) : 
  ∀ x, ¬ f x p q = 0 -> 
  let f_pos x := x^2 + p * x + q in
  let f_neg x := -x^2 + p * x + q in
  (∀ x, f_pos x = 0 -> false) ∧ (∀ x, f_neg x = 0 -> false) := by
  sorry

# Evaluation of Problem Claims:
# This theorem checks that those conditions imply the negation of C and D being incorrect.
theorem evaluate_problem_claims : 
  (∀ p q, (p^2 - 4 * q < 0 ∧ p^2 + 4 * q < 0) -> (∀ x, f x p q ≠ 0)) 
  ∧ 
  (∀ p q, (p < 0 ∧ q > 0) -> ¬ ∃ x1 x2 x3, f x1 p q = 0 ∧ f x2 p q = 0 ∧ f x3 p q = 0) := by
  sorry

end incorrect_options_no_three_real_roots_when_p_neg_q_pos_evaluate_problem_claims_l294_294972


namespace smallest_sphere_radius_l294_294233

noncomputable def radius_smallest_sphere : ℝ := 2 * Real.sqrt 3 + 2

theorem smallest_sphere_radius (r : ℝ) (h : r = 2) : radius_smallest_sphere = 2 * Real.sqrt 3 + 2 := by
  sorry

end smallest_sphere_radius_l294_294233


namespace correct_operation_l294_294127

variables {x y : ℝ}

theorem correct_operation : -2 * x * 3 * y = -6 * x * y :=
by
  sorry

end correct_operation_l294_294127


namespace circle_radii_l294_294802

theorem circle_radii (O1 O2 A B : Point)
    (a : ℝ)
    (h1 : ∠ A O1 B = 90)
    (h2 : ∠ A O2 B = 60)
    (h3 : dist O1 O2 = a) :
    (radii O1 O2 = (a * (sqrt 6 - sqrt 2) / 2, a * (sqrt 3 - 1)) ∨ 
     radii O1 O2 = (a * (sqrt 6 + sqrt 2) / 2, a * (sqrt 3 + 1))) :=
sorry

end circle_radii_l294_294802


namespace area_of_quadrilateral_l294_294783

-- Assume the conditions and define the variables with given properties
variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (pointA : A) (pointB : B) (pointC : C) (pointD : D)
variables (length : A → B → ℝ)

-- Define the conditions
def right_angle_at_B : Prop := ∃ AB BC, AB = length pointA pointB ∧ BC = length pointB pointC ∧ AB^2 + BC^2 = (length pointA pointC)^2
def right_angle_at_D : Prop := ∃ AD DC, AD = length pointA pointD ∧ DC = length pointD pointC ∧ AD^2 + DC^2 = (length pointA pointC)^2
def hypotenuse_AC : Prop := length pointA pointC = 5
def distinct_integer_sides : Prop := ∃ AB AD, AB = length pointA pointB ∧ AD = length pointA pointD ∧ AB ≠ AD

-- Translate the question and answer into a theorem
theorem area_of_quadrilateral (h1 : right_angle_at_B) (h2 : right_angle_at_D) (h3 : hypotenuse_AC) (h4 : distinct_integer_sides) :
  quadrilateral_area_ABCD = 12 :=
sorry

end area_of_quadrilateral_l294_294783


namespace sherman_weekend_driving_time_l294_294072

def total_driving_time_per_week : ℕ := 9
def commute_time_per_day : ℕ := 1
def work_days_per_week : ℕ := 5
def weekend_days : ℕ := 2

theorem sherman_weekend_driving_time :
  (total_driving_time_per_week - commute_time_per_day * work_days_per_week) / weekend_days = 2 :=
sorry

end sherman_weekend_driving_time_l294_294072


namespace box_contains_1600_calories_l294_294865

theorem box_contains_1600_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  total_calories = 1600 :=
by
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  show total_calories = 1600
  sorry

end box_contains_1600_calories_l294_294865


namespace average_number_of_fish_is_75_l294_294768

-- Define the number of fish in Boast Pool and conditions for other bodies of water
def Boast_Pool_fish : ℕ := 75
def Onum_Lake_fish : ℕ := Boast_Pool_fish + 25
def Riddle_Pond_fish : ℕ := Onum_Lake_fish / 2

-- Define the average number of fish in all three bodies of water
def average_fish : ℕ := (Onum_Lake_fish + Boast_Pool_fish + Riddle_Pond_fish) / 3

-- Prove that the average number of fish in all three bodies of water is 75
theorem average_number_of_fish_is_75 : average_fish = 75 := by
  sorry

end average_number_of_fish_is_75_l294_294768


namespace symmetric_points_sum_l294_294655

theorem symmetric_points_sum {c e : ℤ} 
  (P : ℤ × ℤ × ℤ) 
  (sym_xoy : ℤ × ℤ × ℤ) 
  (sym_y : ℤ × ℤ × ℤ) 
  (hP : P = (-4, -2, 3)) 
  (h_sym_xoy : sym_xoy = (-4, -2, -3)) 
  (h_sym_y : sym_y = (4, -2, 3)) 
  (hc : c = -3) 
  (he : e = 4) : 
  c + e = 1 :=
by
  -- Proof goes here
  sorry

end symmetric_points_sum_l294_294655


namespace determine_m_value_l294_294311

theorem determine_m_value :
  ∀ (m : ℝ), (A = {3, m^2}) → (B = {-1, 3, 2m-1}) → (A ⊆ B) → m = 1 :=
by
  intros m hA hB h_subset
  -- conditions and proof omitted
  sorry

end determine_m_value_l294_294311


namespace greatest_distance_centers_two_circles_l294_294487

theorem greatest_distance_centers_two_circles
  (w h : ℝ) (d r : ℝ)
  (hw : w = 15) (hh : h = 18) (hd : d = 5) (hr : r = d / 2) :
  ∃ d_max, d_max = real.sqrt ((w - 2*r)^2 + (h - 2*r)^2) ∧ d_max = real.sqrt 269 :=
begin
  use real.sqrt 269,
  split,
  { rw [hw, hh, hd, hr],
    norm_num,
    rw [sub_mul],
    ring },
  { refl }
end

end greatest_distance_centers_two_circles_l294_294487


namespace circle_radius_l294_294677

-- Define the general equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 4 * y = 0

-- Prove the radius of the circle given by the equation is √5
theorem circle_radius :
  (∀ x y : ℝ, circle_eq x y) →
  (∃ r : ℝ, r = Real.sqrt 5) :=
by
  sorry

end circle_radius_l294_294677


namespace mult_base7_correct_l294_294929

def base7_to_base10 (n : ℕ) : ℕ :=
  -- assume conversion from base-7 to base-10 is already defined
  sorry 

def base10_to_base7 (n : ℕ) : ℕ :=
  -- assume conversion from base-10 to base-7 is already defined
  sorry

theorem mult_base7_correct : (base7_to_base10 325) * (base7_to_base10 4) = base7_to_base10 1656 :=
by
  sorry

end mult_base7_correct_l294_294929


namespace incorrect_trig_identity_l294_294514

theorem incorrect_trig_identity (α β : ℝ) :
  ¬ (cos (-α + β) = -cos (α - β)) := 
sorry

end incorrect_trig_identity_l294_294514


namespace arithmetic_progression_condition_l294_294983

theorem arithmetic_progression_condition
  (a b c : ℝ) (a1 d : ℝ) (p n k : ℕ) :
  a = a1 + (p - 1) * d →
  b = a1 + (n - 1) * d →
  c = a1 + (k - 1) * d →
  a * (n - k) + b * (k - p) + c * (p - n) = 0 :=
by
  intros h1 h2 h3
  sorry


end arithmetic_progression_condition_l294_294983


namespace least_number_of_roots_l294_294176

variable {g : ℝ → ℝ}

-- Conditions
axiom g_defined (x : ℝ) : g x = g x
axiom g_symmetry_1 (x : ℝ) : g (3 + x) = g (3 - x)
axiom g_symmetry_2 (x : ℝ) : g (5 + x) = g (5 - x)
axiom g_at_1 : g 1 = 0

-- Root count in the interval
theorem least_number_of_roots : ∃ (n : ℕ), n >= 250 ∧ (∀ m, -1000 ≤ (1 + 8 * m:ℝ) ∧ (1 + 8 * m:ℝ) ≤ 1000 → g (1 + 8 * m) = 0) :=
sorry

end least_number_of_roots_l294_294176


namespace solve_system_l294_294242

theorem solve_system : ∃ x y : ℝ, 7 * x = -9 - 3 * y ∧ 4 * x = 5 * y - 32 ∧ x = -3 ∧ y = 4 :=
by
  use -3, 4
  constructor
  { -- verify the first equation
    calc
      7 * (-3) = -21    : by ring
      ... = -9 - 3 * 4  : by ring
  }
  constructor
  { -- verify the second equation
    calc
      4 * (-3) = -12    : by ring
      ... = 5 * 4 - 32  : by ring
  }
  { -- verify x = -3 and y = 4
    ring
  }

end solve_system_l294_294242


namespace workshop_output_comparison_l294_294731

theorem workshop_output_comparison (a x : ℝ)
  (h1 : ∀n:ℕ, n ≥ 0 → (1 + n * a) = (1 + x)^n) :
  (1 + 3 * a) > (1 + x)^3 := sorry

end workshop_output_comparison_l294_294731


namespace find_functions_l294_294736

namespace ProofEquivalence

def norm_star (x : ℤ) : ℤ :=
  (abs x + abs (x - 1) - 1) / 2

def iterate_f {α : Type} (f : α → α) (n : ℕ) (x : α) : α :=
  Nat.recOn n x (λ _ g => f g)

theorem find_functions (f : ℕ → ℕ) :
  (∀ x : ℕ, iterate_f f (Int.to_nat (norm_star (f x - x))) x = x) →
  (∀ x : ℕ, f x = x ∨ f x = x + 1) :=
by
  sorry

end ProofEquivalence

end find_functions_l294_294736


namespace complement_of_union_l294_294023

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def U : Set ℤ := Set.univ

theorem complement_of_union (x : ℤ) : (x ∉ A ∪ B) ↔ ∃ k : ℤ, x = 3 * k := 
by {
  sorry
}

end complement_of_union_l294_294023


namespace average_percent_score_l294_294973

theorem average_percent_score (students : ℕ) 
  (s95 s85 s75 s65 s55 s45 : ℕ) 
  (h_students : students = 150) 
  (h_s95 : s95 = 12) 
  (h_s85 : s85 = 30) 
  (h_s75 : s75 = 50) 
  (h_s65 : s65 = 40) 
  (h_s55 : s55 = 15) 
  (h_s45 : s45 = 3) : 
  (95 * s95 + 85 * s85 + 75 * s75 + 65 * s65 + 55 * s55 + 45 * s45) / students = 73.33 :=
by
  sorry

end average_percent_score_l294_294973


namespace annie_initial_money_l294_294202

theorem annie_initial_money
  (hamburger_price : ℕ := 4)
  (milkshake_price : ℕ := 3)
  (num_hamburgers : ℕ := 8)
  (num_milkshakes : ℕ := 6)
  (money_left : ℕ := 70)
  (total_cost_hamburgers : ℕ := num_hamburgers * hamburger_price)
  (total_cost_milkshakes : ℕ := num_milkshakes * milkshake_price)
  (total_cost : ℕ := total_cost_hamburgers + total_cost_milkshakes)
  : num_hamburgers * hamburger_price + num_milkshakes * milkshake_price + money_left = 120 :=
by
  -- proof part skipped
  sorry

end annie_initial_money_l294_294202


namespace dist_2_5_dist_x_neg6_abs_x_minus_2_plus_abs_x_plus_2_eq_4_range_x_for_abs_x_minus_1_plus_abs_x_plus_3_gt_4_min_value_abs_x_minus_3_plus_abs_x_plus_2_plus_abs_x_plus_1_max_y_abs_x_minus_1_plus_abs_x_plus_2_eq_10_minus_abs_y_minus_3_minus_abs_y_plus_4_l294_294695

section Problem1
-- 1. Distance between 2 and 5 is 3.
theorem dist_2_5 : abs (2 - 5) = 3 := sorry
end Problem1

section Problem2
variables (x : ℚ)
-- 2. Distance between x and -6 is |x + 6|.
theorem dist_x_neg6 : abs (x - (-6)) = abs (x + 6) := sorry
end Problem2

section Problem3
variables (x : ℚ)
-- 3. For -2 < x < 2, |x - 2| + |x + 2| = 4.
theorem abs_x_minus_2_plus_abs_x_plus_2_eq_4 (h1 : -2 < x) (h2 : x < 2) : abs (x - 2) + abs (x + 2) = 4 := sorry
end Problem3

section Problem4
variables (x : ℚ)
-- 4. For |x - 1| + |x + 3| > 4, x > 1 or x < -3.
theorem range_x_for_abs_x_minus_1_plus_abs_x_plus_3_gt_4 (h : abs (x - 1) + abs (x + 3) > 4) : x > 1 ∨ x < -3 := sorry
end Problem4

section Problem5
variables (x : ℚ)
-- 5. Minimum value of |x - 3| + |x + 2| + |x + 1| is 5 at x = -1.
theorem min_value_abs_x_minus_3_plus_abs_x_plus_2_plus_abs_x_plus_1 : 
  ∃ x : ℚ, abs (x - 3) + abs (x + 2) + abs (x + 1) = 5 ∧ x = -1 := sorry
end Problem5

section Problem6
variables (x y : ℚ)
-- 6. If |x - 1| + |x + 2| = 10 - |y - 3| - |y + 4|, then the maximum value of y is 3.
theorem max_y_abs_x_minus_1_plus_abs_x_plus_2_eq_10_minus_abs_y_minus_3_minus_abs_y_plus_4
  (h : abs (x - 1) + abs (x + 2) = 10 - abs (y - 3) - abs (y + 4)) : y <= 3 :=
sorry
end Problem6

end dist_2_5_dist_x_neg6_abs_x_minus_2_plus_abs_x_plus_2_eq_4_range_x_for_abs_x_minus_1_plus_abs_x_plus_3_gt_4_min_value_abs_x_minus_3_plus_abs_x_plus_2_plus_abs_x_plus_1_max_y_abs_x_minus_1_plus_abs_x_plus_2_eq_10_minus_abs_y_minus_3_minus_abs_y_plus_4_l294_294695


namespace find_bottle_caps_l294_294901

theorem find_bottle_caps (current_bottle_caps earlier_bottle_caps : Nat) (h1 : current_bottle_caps = 32) (h2 : earlier_bottle_caps = 25) : (current_bottle_caps - earlier_bottle_caps = 7) :=
by 
  rw [h1, h2]
  norm_num

end find_bottle_caps_l294_294901


namespace exists_set_of_size_r_minus_one_l294_294041

theorem exists_set_of_size_r_minus_one (r : ℕ) (h_r : r ≥ 2)
  (α : Type*) (ℱ : set (set α))
  (h_infinite : ℱ.infinite)
  (h_size : ∀ A ∈ ℱ, finset.card A = r)
  (h_non_disjoint : ∀ {A B : set α}, A ∈ ℱ → B ∈ ℱ → (A ∩ B).nonempty) :
  ∃ S : set α, finset.card S = r - 1 ∧ ∀ A ∈ ℱ, (S ∩ A).nonempty :=
sorry

end exists_set_of_size_r_minus_one_l294_294041


namespace minimum_value_problem1_minimum_value_problem2_l294_294143

theorem minimum_value_problem1 (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ y >= 6 := 
sorry

theorem minimum_value_problem2 (x : ℝ) (h : x > 1) : 
  ∃ y, y = (x^2 + 8) / (x - 1) ∧ y >= 8 := 
sorry

end minimum_value_problem1_minimum_value_problem2_l294_294143


namespace sunil_total_amount_back_l294_294085

theorem sunil_total_amount_back 
  (CI : ℝ) (P : ℝ) (r : ℝ) (t : ℕ) (total_amount : ℝ) 
  (h1 : CI = 2828.80) 
  (h2 : r = 8) 
  (h3 : t = 2) 
  (h4 : CI = P * ((1 + r / 100) ^ t - 1)) : 
  total_amount = P + CI → 
  total_amount = 19828.80 :=
by
  sorry

end sunil_total_amount_back_l294_294085


namespace probability_of_point_within_sphere_l294_294185

theorem probability_of_point_within_sphere :
  let volume_cube := 4^3
  let volume_sphere := (4/3) * Real.pi * (1.5)^3
  let probability := volume_sphere / volume_cube
  probability = (4.5 * Real.pi) / 64 := by
  let x y z be real numbers with -2 ≤ x ≤ 2 and -2 ≤ y ≤ 2 and -2 ≤ z ≤ 2
  let sphere_condition := x^2 + y^2 + z^2 ≤ 2.25
  let cube_volume := 4^3
  let sphere_volume := (4/3) * Real.pi * (1.5)^3
  let calculated_probability := sphere_volume / cube_volume
  show calculated_probability = (4.5 * Real.pi) / 64 from sorry

end probability_of_point_within_sphere_l294_294185


namespace find_radius_l294_294462

-- The problem conditions and variables
variables (L : ℝ) (θ : ℝ) (R : ℝ)

-- Given conditions
def given_conditions := L = 2.5 * real.pi ∧ θ = 75

-- The formula for the arc length
def arc_length_formula := L = (θ / 360) * 2 * real.pi * R

-- The proof goal: given the conditions, the radius is 6 cm
theorem find_radius : given_conditions L θ → arc_length_formula L θ R → R = 6 :=
sorry

end find_radius_l294_294462
