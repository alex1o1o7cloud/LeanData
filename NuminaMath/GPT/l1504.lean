import Mathlib

namespace contractor_absent_days_l1504_150475

noncomputable def solve_contractor_problem : Prop :=
  ∃ (x y : ℕ), 
    x + y = 30 ∧ 
    25 * x - 750 / 100 * y = 555 ∧
    y = 6

theorem contractor_absent_days : solve_contractor_problem :=
  sorry

end contractor_absent_days_l1504_150475


namespace find_num_alligators_l1504_150471

-- We define the conditions as given in the problem
def journey_to_delta_hours : ℕ := 4
def extra_hours : ℕ := 2
def combined_time_alligators_walked : ℕ := 46

-- We define the hypothesis in terms of Lean variables
def num_alligators_traveled_with_Paul (A : ℕ) : Prop :=
  (journey_to_delta_hours + (journey_to_delta_hours + extra_hours) * A) = combined_time_alligators_walked

-- Now the theorem statement where we prove that the number of alligators (A) is 7
theorem find_num_alligators :
  ∃ A : ℕ, num_alligators_traveled_with_Paul A ∧ A = 7 :=
by
  existsi 7
  unfold num_alligators_traveled_with_Paul
  simp
  sorry -- this is where the actual proof would go

end find_num_alligators_l1504_150471


namespace solve_for_m_l1504_150492

theorem solve_for_m {m : ℝ} (h : ∀ x : ℝ, (m - 5) * x = 0) : m = 5 :=
sorry

end solve_for_m_l1504_150492


namespace range_of_k_in_first_quadrant_l1504_150487

theorem range_of_k_in_first_quadrant (k : ℝ) (h₁ : k ≠ -1) :
  (∃ x y : ℝ, y = k * x - 1 ∧ x + y - 1 = 0 ∧ x > 0 ∧ y > 0) ↔ 1 < k := by sorry

end range_of_k_in_first_quadrant_l1504_150487


namespace nth_equation_l1504_150435

theorem nth_equation (n : ℕ) (hn: n ≥ 1) : 
  (n+1) / ((n+1)^2 - 1) - 1 / (n * (n+1) * (n+2)) = 1 / (n+1) :=
by
  sorry

end nth_equation_l1504_150435


namespace sum_of_square_roots_l1504_150452

theorem sum_of_square_roots : 
  (Real.sqrt 1) + (Real.sqrt (1 + 3)) + (Real.sqrt (1 + 3 + 5)) + (Real.sqrt (1 + 3 + 5 + 7)) = 10 := 
by 
  sorry

end sum_of_square_roots_l1504_150452


namespace domain_of_function_l1504_150404

theorem domain_of_function : {x : ℝ | 3 - 2 * x - x ^ 2 ≥ 0 } = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end domain_of_function_l1504_150404


namespace volume_of_convex_solid_l1504_150429

variables {m V t6 T t3 : ℝ} 

-- Definition of the distance between the two parallel planes
def distance_between_planes (m : ℝ) : Prop := m > 0

-- Areas of the two parallel faces
def area_hexagon_face (t6 : ℝ) : Prop := t6 > 0
def area_triangle_face (t3 : ℝ) : Prop := t3 > 0

-- Area of the cross-section of the solid with a plane perpendicular to the height at its midpoint
def area_cross_section (T : ℝ) : Prop := T > 0

-- Volume of the convex solid
def volume_formula_holds (V m t6 T t3 : ℝ) : Prop :=
  V = (m / 6) * (t6 + 4 * T + t3)

-- Formal statement of the problem
theorem volume_of_convex_solid
  (m t6 t3 T V : ℝ)
  (h₁ : distance_between_planes m)
  (h₂ : area_hexagon_face t6)
  (h₃ : area_triangle_face t3)
  (h₄ : area_cross_section T) :
  volume_formula_holds V m t6 T t3 :=
by
  sorry

end volume_of_convex_solid_l1504_150429


namespace odd_function_h_l1504_150421

noncomputable def f (x h k : ℝ) : ℝ := Real.log (abs ((1 / (x + 1)) + k)) + h

theorem odd_function_h (k : ℝ) (h : ℝ) (H : ∀ x : ℝ, x ≠ -1 → f x h k = -f (-x) h k) :
  h = Real.log 2 :=
sorry

end odd_function_h_l1504_150421


namespace boys_belong_to_other_communities_l1504_150409

-- Definitions for the given problem
def total_boys : ℕ := 850
def percent_muslims : ℝ := 0.34
def percent_hindus : ℝ := 0.28
def percent_sikhs : ℝ := 0.10
def percent_other : ℝ := 1 - (percent_muslims + percent_hindus + percent_sikhs)

-- Statement to prove that the number of boys belonging to other communities is 238
theorem boys_belong_to_other_communities : 
  (percent_other * total_boys) = 238 := by 
  sorry

end boys_belong_to_other_communities_l1504_150409


namespace find_number_l1504_150459

theorem find_number (n : ℝ) (h : (1 / 3) * n = 6) : n = 18 :=
sorry

end find_number_l1504_150459


namespace minute_hand_angle_45min_l1504_150446

theorem minute_hand_angle_45min
  (duration : ℝ)
  (h1 : duration = 45) :
  (-(3 / 4) * 2 * Real.pi = - (3 * Real.pi / 2)) :=
by
  sorry

end minute_hand_angle_45min_l1504_150446


namespace select_defective_products_l1504_150418

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem select_defective_products :
  let total_products := 200
  let defective_products := 3
  let selected_products := 5
  let ways_2_defective := choose defective_products 2 * choose (total_products - defective_products) 3
  let ways_3_defective := choose defective_products 3 * choose (total_products - defective_products) 2
  ways_2_defective + ways_3_defective = choose defective_products 2 * choose (total_products - defective_products) 3 + choose defective_products 3 * choose (total_products - defective_products) 2 :=
by
  sorry

end select_defective_products_l1504_150418


namespace total_dogs_l1504_150414

theorem total_dogs (D : ℕ) 
(h1 : 12 = 12)
(h2 : D / 2 = D / 2)
(h3 : D / 4 = D / 4)
(h4 : 10 = 10)
(h_eq : 12 + D / 2 + D / 4 + 10 = D) : 
D = 88 := by
sorry

end total_dogs_l1504_150414


namespace product_of_solutions_l1504_150458

theorem product_of_solutions :
  (∃ x y : ℝ, (|x^2 - 6 * x| + 5 = 41) ∧ (|y^2 - 6 * y| + 5 = 41) ∧ x ≠ y ∧ x * y = -36) :=
by
  sorry

end product_of_solutions_l1504_150458


namespace three_digit_division_l1504_150447

theorem three_digit_division (abc : ℕ) (a b c : ℕ) (h1 : 100 ≤ abc ∧ abc < 1000) (h2 : abc = 100 * a + 10 * b + c) (h3 : a ≠ 0) :
  (1001 * abc) / 7 / 11 / 13 = abc :=
by
  sorry

end three_digit_division_l1504_150447


namespace map_distance_l1504_150419

/--
On a map, 8 cm represents 40 km. Prove that 20 cm represents 100 km.
-/
theorem map_distance (scale_factor : ℕ) (distance_cm : ℕ) (distance_km : ℕ) 
  (h_scale : scale_factor = 5) (h_distance_cm : distance_cm = 20) : 
  distance_km = 20 * scale_factor := 
by {
  sorry
}

end map_distance_l1504_150419


namespace algebraic_expression_solution_l1504_150483

theorem algebraic_expression_solution
  (a b : ℝ)
  (h : -2 * a + 3 * b = 10) :
  9 * b - 6 * a + 2 = 32 :=
by 
  -- We would normally provide the proof here
  sorry

end algebraic_expression_solution_l1504_150483


namespace rectangle_enclosed_by_lines_l1504_150476

theorem rectangle_enclosed_by_lines : 
  ∃ (ways : ℕ), 
  (ways = (Nat.choose 5 2) * (Nat.choose 4 2)) ∧ 
  ways = 60 := 
by
  sorry

end rectangle_enclosed_by_lines_l1504_150476


namespace Liu_Wei_parts_per_day_l1504_150482

theorem Liu_Wei_parts_per_day :
  ∀ (total_parts days_needed parts_per_day_worked initial_days days_remaining : ℕ), 
  total_parts = 190 →
  parts_per_day_worked = 15 →
  initial_days = 2 →
  days_needed = 10 →
  days_remaining = days_needed - initial_days →
  (total_parts - (initial_days * parts_per_day_worked)) / days_remaining = 20 :=
by
  intros total_parts days_needed parts_per_day_worked initial_days days_remaining h1 h2 h3 h4 h5
  sorry

end Liu_Wei_parts_per_day_l1504_150482


namespace polynomial_root_sum_nonnegative_l1504_150460

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c
noncomputable def g (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem polynomial_root_sum_nonnegative 
  (m1 m2 k1 k2 b c p q : ℝ)
  (h1 : f m1 b c = 0) (h2 : f m2 b c = 0)
  (h3 : g k1 p q = 0) (h4 : g k2 p q = 0) :
  f k1 b c + f k2 b c + g m1 p q + g m2 p q ≥ 0 := 
by
  sorry  -- Proof placeholders

end polynomial_root_sum_nonnegative_l1504_150460


namespace find_m_from_equation_l1504_150431

theorem find_m_from_equation :
  ∀ (x m : ℝ), (x^2 + 2 * x - 1 = 0) → ((x + m)^2 = 2) → m = 1 :=
by
  intros x m h1 h2
  sorry

end find_m_from_equation_l1504_150431


namespace complex_quadrant_l1504_150495

-- Declare the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Declare the complex number z as per the condition
noncomputable def z : ℂ := (2 * i) / (i - 1)

-- State and prove that the complex number z lies in the fourth quadrant
theorem complex_quadrant : (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end complex_quadrant_l1504_150495


namespace determine_x_value_l1504_150439

theorem determine_x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y ^ 3) (h2 : x / 9 = 9 * y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 := by 
  sorry

end determine_x_value_l1504_150439


namespace polynomial_factors_l1504_150407

theorem polynomial_factors (x : ℝ) : 
  (x^4 - 4*x^2 + 4) = (x^2 - 2*x + 2) * (x^2 + 2*x + 2) :=
by
  sorry

end polynomial_factors_l1504_150407


namespace triangle_is_isosceles_l1504_150464

variable (a b m_a m_b : ℝ)

-- Conditions: 
-- A circle touches two sides of a triangle (denoted as a and b).
-- The circle also touches the medians m_a and m_b drawn to these sides.
-- Given equations:
axiom Eq1 : (1/2) * a + (1/3) * m_b = (1/2) * b + (1/3) * m_a
axiom Eq3 : (1/2) * a + m_b = (1/2) * b + m_a

-- Question: Prove that the triangle is isosceles, i.e., a = b
theorem triangle_is_isosceles : a = b :=
by
  sorry

end triangle_is_isosceles_l1504_150464


namespace range_of_s_triangle_l1504_150470

theorem range_of_s_triangle (inequalities_form_triangle : Prop) : 
  (0 < s ∧ s ≤ 2) ∨ (s ≥ 4) ↔ inequalities_form_triangle := 
sorry

end range_of_s_triangle_l1504_150470


namespace capacitor_capacitance_l1504_150434

theorem capacitor_capacitance 
  (U ε Q : ℝ) 
  (hQ : Q = (U^2 * (ε - 1)^2 * C) /  (2 * ε * (ε + 1)))
  : C = (2 * ε * (ε + 1) * Q) / (U^2 * (ε - 1)^2) :=
by
  sorry

end capacitor_capacitance_l1504_150434


namespace simplification_problem_l1504_150426

theorem simplification_problem (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h_sum : p + q + r = 1) :
  (1 / (q^2 + r^2 - p^2) + 1 / (p^2 + r^2 - q^2) + 1 / (p^2 + q^2 - r^2) = 3 / (1 - 2 * q * r)) :=
by
  sorry

end simplification_problem_l1504_150426


namespace area_of_annulus_l1504_150400

theorem area_of_annulus (R r t : ℝ) (h : R > r) (h_tangent : R^2 = r^2 + t^2) : 
  π * (R^2 - r^2) = π * t^2 :=
by 
  sorry

end area_of_annulus_l1504_150400


namespace total_carriages_l1504_150436

theorem total_carriages (Euston Norfolk Norwich FlyingScotsman : ℕ) 
  (h1 : Euston = 130)
  (h2 : Norfolk = Euston - 20)
  (h3 : Norwich = 100)
  (h4 : FlyingScotsman = Norwich + 20) :
  Euston + Norfolk + Norwich + FlyingScotsman = 460 :=
by 
  sorry

end total_carriages_l1504_150436


namespace graph_passes_through_fixed_point_l1504_150453

theorem graph_passes_through_fixed_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
    ∃ (x y : ℝ), (x = -3) ∧ (y = -1) ∧ (y = a^(x + 3) - 2) :=
by
  sorry

end graph_passes_through_fixed_point_l1504_150453


namespace total_weight_of_peppers_l1504_150451

def green_peppers := 0.3333333333333333
def red_peppers := 0.4444444444444444
def yellow_peppers := 0.2222222222222222
def orange_peppers := 0.7777777777777778

theorem total_weight_of_peppers :
  green_peppers + red_peppers + yellow_peppers + orange_peppers = 1.7777777777777777 :=
by
  sorry

end total_weight_of_peppers_l1504_150451


namespace complex_power_identity_l1504_150485

theorem complex_power_identity (w : ℂ) (h : w + w⁻¹ = 2) : w^(2022 : ℕ) + (w⁻¹)^(2022 : ℕ) = 2 := by
  sorry

end complex_power_identity_l1504_150485


namespace solve_for_A_l1504_150416

theorem solve_for_A (A : ℕ) (h1 : 3 + 68 * A = 691) (h2 : 68 * A < 1000) (h3 : 68 * A ≥ 100) : A = 8 :=
by
  sorry

end solve_for_A_l1504_150416


namespace range_of_a_if_p_and_not_q_l1504_150415

open Real

def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem range_of_a_if_p_and_not_q : 
  (∃ a : ℝ, (p a ∧ ¬q a)) → 
  (∀ a : ℝ, (p a ∧ ¬q a) → (-1 ≤ a ∧ a < 0)) :=
sorry

end range_of_a_if_p_and_not_q_l1504_150415


namespace profit_in_2004_correct_l1504_150420

-- We define the conditions as given in the problem
def annual_profit_2002 : ℝ := 10
def annual_growth_rate (p : ℝ) : ℝ := p

-- The expression for the annual profit in 2004 given the above conditions
def annual_profit_2004 (p : ℝ) : ℝ := annual_profit_2002 * (1 + p) * (1 + p)

-- The theorem to prove that the computed annual profit in 2004 matches the expected answer
theorem profit_in_2004_correct (p : ℝ) :
  annual_profit_2004 p = 10 * (1 + p)^2 := 
by 
  sorry

end profit_in_2004_correct_l1504_150420


namespace evaluate_expression_l1504_150486

theorem evaluate_expression : 2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := by
  sorry

end evaluate_expression_l1504_150486


namespace worker_wage_before_promotion_l1504_150403

variable (W_new : ℝ)
variable (W : ℝ)

theorem worker_wage_before_promotion (h1 : W_new = 45) (h2 : W_new = 1.60 * W) :
  W = 28.125 := by
  sorry

end worker_wage_before_promotion_l1504_150403


namespace math_problem_l1504_150480

variables {a b c d e : ℤ}

theorem math_problem 
(h1 : a - b + c - e = 7)
(h2 : b - c + d + e = 9)
(h3 : c - d + a - e = 5)
(h4 : d - a + b + e = 1)
: a + b + c + d + e = 11 := 
by 
  sorry

end math_problem_l1504_150480


namespace remainder_of_2_pow_33_mod_9_l1504_150491

theorem remainder_of_2_pow_33_mod_9 : (2 ^ 33) % 9 = 8 :=
by
  sorry

end remainder_of_2_pow_33_mod_9_l1504_150491


namespace condition_necessary_but_not_sufficient_l1504_150494

theorem condition_necessary_but_not_sufficient (a : ℝ) :
  ((1 / a > 1) → (a < 1)) ∧ (∃ (a : ℝ), a < 1 ∧ 1 / a < 1) :=
by
  sorry

end condition_necessary_but_not_sufficient_l1504_150494


namespace average_of_all_5_numbers_is_20_l1504_150457

def average_of_all_5_numbers
  (sum_3_numbers : ℕ)
  (avg_2_numbers : ℕ) : ℕ :=
(sum_3_numbers + 2 * avg_2_numbers) / 5

theorem average_of_all_5_numbers_is_20 :
  average_of_all_5_numbers 48 26 = 20 :=
by
  unfold average_of_all_5_numbers -- Expand the definition
  -- Sum of 5 numbers is 48 (sum of 3) + (2 * 26) (sum of other 2)
  -- Total sum is 48 + 52 = 100
  -- Average is 100 / 5 = 20
  norm_num -- Check the numeric calculation
  -- sorry

end average_of_all_5_numbers_is_20_l1504_150457


namespace maximum_value_l1504_150441

noncomputable def conditions (m n t : ℝ) : Prop :=
  -- m, n, t are positive real numbers
  (0 < m) ∧ (0 < n) ∧ (0 < t) ∧
  -- Equation condition
  (m^2 - 3 * m * n + 4 * n^2 - t = 0)

noncomputable def minimum_u (m n t : ℝ) : Prop :=
  -- Minimum value condition for t / mn
  (t / (m * n) = 1)

theorem maximum_value (m n t : ℝ) (h1 : conditions m n t) (h2 : minimum_u m n t) :
  -- Proving the maximum value of m + 2n - t
  (m + 2 * n - t) = 2 :=
sorry

end maximum_value_l1504_150441


namespace isosceles_vertex_angle_l1504_150413

-- Let T be a type representing triangles, with a function base_angle returning the degree of a base angle,
-- and vertex_angle representing the degree of the vertex angle.
axiom Triangle : Type
axiom is_isosceles (t : Triangle) : Prop
axiom base_angle_deg (t : Triangle) : ℝ
axiom vertex_angle_deg (t : Triangle) : ℝ

theorem isosceles_vertex_angle (t : Triangle) (h_isosceles : is_isosceles t)
  (h_base_angle : base_angle_deg t = 50) : vertex_angle_deg t = 80 := by
  sorry

end isosceles_vertex_angle_l1504_150413


namespace larger_integer_l1504_150417

theorem larger_integer (a b : ℕ) (h_diff : a - b = 8) (h_prod : a * b = 224) : a = 16 :=
by
  sorry

end larger_integer_l1504_150417


namespace fraction_relationships_l1504_150474

variables (a b c d : ℚ)

theorem fraction_relationships (h1 : a / b = 3) (h2 : b / c = 2 / 3) (h3 : c / d = 5) :
  d / a = 1 / 10 :=
by
  sorry

end fraction_relationships_l1504_150474


namespace fresh_grape_weight_l1504_150465

variable (D : ℝ) (F : ℝ)

axiom dry_grape_weight : D = 66.67
axiom fresh_grape_water_content : F * 0.25 = D * 0.75

theorem fresh_grape_weight : F = 200.01 :=
by sorry

end fresh_grape_weight_l1504_150465


namespace equation_of_plane_passing_through_points_l1504_150481

/-
Let M1, M2, and M3 be points in three-dimensional space.
M1 = (1, 2, 0)
M2 = (1, -1, 2)
M3 = (0, 1, -1)
We need to prove that the plane passing through these points has the equation 5x - 2y - 3z - 1 = 0.
-/

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def M1 : Point3D := ⟨1, 2, 0⟩
def M2 : Point3D := ⟨1, -1, 2⟩
def M3 : Point3D := ⟨0, 1, -1⟩

theorem equation_of_plane_passing_through_points :
  ∃ (a b c d : ℝ), (∀ (P : Point3D), 
  P = M1 ∨ P = M2 ∨ P = M3 → a * P.x + b * P.y + c * P.z + d = 0)
  ∧ a = 5 ∧ b = -2 ∧ c = -3 ∧ d = -1 :=
by
  sorry

end equation_of_plane_passing_through_points_l1504_150481


namespace coloring_count_l1504_150445

theorem coloring_count (n : ℕ) (h : 0 < n) :
  ∃ (num_colorings : ℕ), num_colorings = 2 :=
sorry

end coloring_count_l1504_150445


namespace does_not_balance_l1504_150401

variables (square odot circ triangle O : ℝ)

-- Conditions represented as hypothesis
def condition1 : Prop := 4 * square = odot + circ
def condition2 : Prop := 2 * circ + odot = 2 * triangle

-- Statement to be proved
theorem does_not_balance (h1 : condition1 square odot circ) (h2 : condition2 circ odot triangle)
 : ¬(2 * triangle + square = triangle + odot + square) := 
sorry

end does_not_balance_l1504_150401


namespace intersection_point_divides_chord_l1504_150448

theorem intersection_point_divides_chord (R AB PO : ℝ)
    (hR: R = 11) (hAB: AB = 18) (hPO: PO = 7) :
    ∃ (AP PB : ℝ), (AP / PB = 2 ∨ AP / PB = 1 / 2) ∧ (AP + PB = AB) := by
  sorry

end intersection_point_divides_chord_l1504_150448


namespace math_problem_l1504_150405

open Real

theorem math_problem
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z)
  (hxyz : x + y + z = 1) :
  ( (1 / x^2 + x) * (1 / y^2 + y) * (1 / z^2 + z) ≥ (28 / 3)^3 ) :=
by {
  sorry
}

end math_problem_l1504_150405


namespace movie_ticket_notation_l1504_150461

-- Definition of movie ticket notation
def ticket_notation (row : ℕ) (seat : ℕ) : (ℕ × ℕ) :=
  (row, seat)

-- Given condition: "row 10, seat 3" is denoted as (10, 3)
def given := ticket_notation 10 3 = (10, 3)

-- Proof statement: "row 6, seat 16" is denoted as (6, 16)
theorem movie_ticket_notation : ticket_notation 6 16 = (6, 16) :=
by
  -- Proof omitted, since the theorem statement is the focus
  sorry

end movie_ticket_notation_l1504_150461


namespace option_C_represents_same_function_l1504_150440

-- Definitions of the functions from option C
def f (x : ℝ) := x^2 - 1
def g (t : ℝ) := t^2 - 1

-- The proof statement that needs to be proven
theorem option_C_represents_same_function :
  f = g :=
sorry

end option_C_represents_same_function_l1504_150440


namespace trig_identity_l1504_150443

theorem trig_identity (a : ℝ) (h : (1 + Real.sin a) / Real.cos a = -1 / 2) : 
  (Real.cos a / (Real.sin a - 1)) = 1 / 2 := by
  -- Proof goes here
  sorry

end trig_identity_l1504_150443


namespace area_of_circle_l1504_150433

theorem area_of_circle (x y : ℝ) :
  x^2 + y^2 + 8 * x + 10 * y = -9 → 
  ∃ a : ℝ, a = 32 * Real.pi :=
by
  sorry

end area_of_circle_l1504_150433


namespace John_pays_2400_per_year_l1504_150450

theorem John_pays_2400_per_year
  (hours_per_month : ℕ)
  (average_length : ℕ)
  (cost_per_song : ℕ)
  (h1 : hours_per_month = 20)
  (h2 : average_length = 3)
  (h3 : cost_per_song = 50) :
  (hours_per_month * 60 / average_length * cost_per_song * 12 = 2400) :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end John_pays_2400_per_year_l1504_150450


namespace express_a_in_terms_of_b_l1504_150477

noncomputable def a : ℝ := Real.log 1250 / Real.log 6
noncomputable def b : ℝ := Real.log 50 / Real.log 3

theorem express_a_in_terms_of_b : a = (b + 0.6826) / 1.2619 :=
by
  sorry

end express_a_in_terms_of_b_l1504_150477


namespace original_cost_of_dolls_l1504_150462

theorem original_cost_of_dolls 
  (x : ℝ) -- original cost of each Russian doll
  (savings : ℝ) -- total savings of Daniel
  (h1 : savings = 15 * x) -- Daniel saves enough to buy 15 dolls at original price
  (h2 : savings = 20 * 3) -- with discounted price, he can buy 20 dolls
  : x = 4 :=
by
  sorry

end original_cost_of_dolls_l1504_150462


namespace sequence_a_correct_l1504_150411

open Nat -- Opening the natural numbers namespace

noncomputable def a : ℕ → ℝ
| 0       => 1
| (n + 1) => (1 / 2 : ℝ) * a n

theorem sequence_a_correct : 
  (∀ n, 0 < a n) ∧ 
  a 1 = 1 ∧ 
  (∀ n, a (n + 1) = a n / 2) ∧
  a 2 = 1 / 2 ∧
  a 3 = 1 / 4 ∧
  ∀ n, a n = 1 / 2^(n - 1) :=
by
  sorry

end sequence_a_correct_l1504_150411


namespace find_x_squared_minus_y_squared_l1504_150454

variable (x y : ℝ)

theorem find_x_squared_minus_y_squared 
(h1 : y + 6 = (x - 3)^2)
(h2 : x + 6 = (y - 3)^2)
(h3 : x ≠ y) :
x^2 - y^2 = 27 := by
  sorry

end find_x_squared_minus_y_squared_l1504_150454


namespace find_initial_population_l1504_150442

-- Define the conditions that the population increases annually by 20%
-- and that the population after 2 years is 14400.
def initial_population (P : ℝ) : Prop :=
  1.44 * P = 14400

-- The theorem states that given the conditions, the initial population is 10000.
theorem find_initial_population (P : ℝ) (h : initial_population P) : P = 10000 :=
  sorry

end find_initial_population_l1504_150442


namespace equal_semi_circles_radius_l1504_150402

-- Define the segments and semicircles given in the problem as conditions.
def segment1 : ℝ := 12
def segment2 : ℝ := 22
def segment3 : ℝ := 22
def segment4 : ℝ := 16
def segment5 : ℝ := 22

def total_horizontal_path1 (r : ℝ) : ℝ := 2*r + segment1 + 2*r + segment1 + 2*r
def total_horizontal_path2 (r : ℝ) : ℝ := segment2 + 2*r + segment4 + 2*r + segment5

-- The theorem that proves the radius is 18.
theorem equal_semi_circles_radius : ∃ r : ℝ, total_horizontal_path1 r = total_horizontal_path2 r ∧ r = 18 := by
  use 18
  simp [total_horizontal_path1, total_horizontal_path2, segment1, segment2, segment3, segment4, segment5]
  sorry

end equal_semi_circles_radius_l1504_150402


namespace rahul_matches_played_l1504_150456

theorem rahul_matches_played
  (current_avg : ℕ)
  (runs_today : ℕ)
  (new_avg : ℕ)
  (m: ℕ)
  (h1 : current_avg = 51)
  (h2 : runs_today = 78)
  (h3 : new_avg = 54)
  (h4 : (51 * m + runs_today) / (m + 1) = new_avg) :
  m = 8 :=
by
  sorry

end rahul_matches_played_l1504_150456


namespace choose_socks_l1504_150466

open Nat

theorem choose_socks :
  (Nat.choose 8 4) = 70 :=
by 
  sorry

end choose_socks_l1504_150466


namespace exists_negative_number_satisfying_inequality_l1504_150473

theorem exists_negative_number_satisfying_inequality :
  ∃ x : ℝ, x < 0 ∧ (1 + x) * (1 - 9 * x) > 0 :=
sorry

end exists_negative_number_satisfying_inequality_l1504_150473


namespace expression_value_l1504_150498

theorem expression_value (x y : ℝ) (h : y = 2 - x) : 4 * x + 4 * y - 3 = 5 :=
by
  sorry

end expression_value_l1504_150498


namespace cos_half_diff_proof_l1504_150469

noncomputable def cos_half_diff (A B C : ℝ) (h_triangle : A + B + C = 180)
  (h_relation : A + C = 2 * B)
  (h_equation : (1 / Real.cos A) + (1 / Real.cos C) = - (Real.sqrt 2 / Real.cos B)) : Real :=
  Real.cos ((A - C) / 2)

theorem cos_half_diff_proof (A B C : ℝ)
  (h_triangle : A + B + C = 180)
  (h_relation : A + C = 2 * B)
  (h_equation : (1 / Real.cos A) + (1 / Real.cos C) = - (Real.sqrt 2 / Real.cos B)) :
  cos_half_diff A B C h_triangle h_relation h_equation = -Real.sqrt 2 / 2 :=
sorry

end cos_half_diff_proof_l1504_150469


namespace isosceles_triangle_perimeter_l1504_150484

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 4)
  (h3 : a = b ∨ 2 * a > b) :
  (a ≠ b ∨ b = 2 * a) → 
  ∃ p : ℝ, p = a + b + b ∧ p = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l1504_150484


namespace meaningful_fraction_range_l1504_150490

theorem meaningful_fraction_range (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) := sorry

end meaningful_fraction_range_l1504_150490


namespace sin_double_angle_l1504_150432

theorem sin_double_angle {θ : ℝ} (h : Real.tan θ = 1 / 3) : 
  Real.sin (2 * θ) = 3 / 5 := 
  sorry

end sin_double_angle_l1504_150432


namespace sum_of_integers_square_greater_272_l1504_150472

theorem sum_of_integers_square_greater_272 (x : ℤ) (h : x^2 = x + 272) :
  ∃ (roots : List ℤ), (roots = [17, -16]) ∧ (roots.sum = 1) :=
sorry

end sum_of_integers_square_greater_272_l1504_150472


namespace upper_limit_of_range_l1504_150449

theorem upper_limit_of_range (n : ℕ) (h : (10 + 10 * n) / 2 = 255) : 10 * n = 500 :=
by 
  sorry

end upper_limit_of_range_l1504_150449


namespace reciprocal_sum_hcf_lcm_l1504_150427

variables (m n : ℕ)

def HCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem reciprocal_sum_hcf_lcm (h₁ : HCF m n = 6) (h₂ : LCM m n = 210) (h₃ : m + n = 60) :
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 21 :=
by
  -- The proof will be inserted here.
  sorry

end reciprocal_sum_hcf_lcm_l1504_150427


namespace largest_divisor_of_composite_sum_and_square_l1504_150493

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ n = a * b

theorem largest_divisor_of_composite_sum_and_square (n : ℕ) (h : is_composite n) : ( ∃ (k : ℕ), ∀ n : ℕ, is_composite n → ∃ m : ℕ, n + n^2 = m * k) → k = 2 :=
by
  sorry

end largest_divisor_of_composite_sum_and_square_l1504_150493


namespace cyrus_pages_proof_l1504_150444

def pages_remaining (total_pages: ℝ) (day1: ℝ) (day2: ℝ) (day3: ℝ) (day4: ℝ) (day5: ℝ) : ℝ :=
  total_pages - (day1 + day2 + day3 + day4 + day5)

theorem cyrus_pages_proof :
  let total_pages := 750
  let day1 := 30
  let day2 := 1.5 * day1
  let day3 := day2 / 2
  let day4 := 2.5 * day3
  let day5 := 15
  pages_remaining total_pages day1 day2 day3 day4 day5 = 581.25 :=
by 
  sorry

end cyrus_pages_proof_l1504_150444


namespace daily_earnings_c_l1504_150489

theorem daily_earnings_c (A B C : ℕ) (h1 : A + B + C = 600) (h2 : A + C = 400) (h3 : B + C = 300) : C = 100 :=
sorry

end daily_earnings_c_l1504_150489


namespace smallest_base10_integer_l1504_150408

-- Definitions of the integers a and b as bases larger than 3.
variables {a b : ℕ}

-- Definitions of the base-10 representation of the given numbers.
def thirteen_in_a (a : ℕ) : ℕ := 1 * a + 3
def thirty_one_in_b (b : ℕ) : ℕ := 3 * b + 1

-- The proof statement.
theorem smallest_base10_integer (h₁ : a > 3) (h₂ : b > 3) :
  (∃ (n : ℕ), thirteen_in_a a = n ∧ thirty_one_in_b b = n) → ∃ n, n = 13 :=
by
  sorry

end smallest_base10_integer_l1504_150408


namespace degree_poly_product_l1504_150410

open Polynomial

-- Given conditions: p and q are polynomials with specified degrees
variables {R : Type*} [CommRing R]
variable (p q : R[X])
variable (hp : degree p = 3)
variable (hq : degree q = 6)

-- Proposition: The degree of p(x^2) * q(x^4) is 30
theorem degree_poly_product : degree (p.comp ((X : R[X])^2) * (q.comp ((X : R[X])^4))) = 30 :=
by sorry

end degree_poly_product_l1504_150410


namespace largest_integer_less_100_leaves_remainder_4_l1504_150467

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end largest_integer_less_100_leaves_remainder_4_l1504_150467


namespace cone_volume_proof_l1504_150438

noncomputable def cone_volume (l h : ℕ) : ℝ :=
  let r := Real.sqrt (l^2 - h^2)
  1 / 3 * Real.pi * r^2 * h

theorem cone_volume_proof :
  cone_volume 13 12 = 100 * Real.pi :=
by
  sorry

end cone_volume_proof_l1504_150438


namespace blue_balls_in_JarB_l1504_150479

-- Defining the conditions
def ratio_white_blue (white blue : ℕ) : Prop := white / gcd white blue = 5 ∧ blue / gcd white blue = 3

def white_balls_in_B := 15

-- Proof statement
theorem blue_balls_in_JarB :
  ∃ (blue : ℕ), ratio_white_blue 15 blue ∧ blue = 9 :=
by {
  -- Proof outline (not required, thus just using sorry)
  sorry
}


end blue_balls_in_JarB_l1504_150479


namespace largest_4_digit_div_by_5_smallest_primes_l1504_150478

noncomputable def LCM_5_smallest_primes : ℕ := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))

theorem largest_4_digit_div_by_5_smallest_primes :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 9240 := by
  sorry

end largest_4_digit_div_by_5_smallest_primes_l1504_150478


namespace sales_tax_is_8_percent_l1504_150468

-- Define the conditions
def total_before_tax : ℝ := 150
def total_with_tax : ℝ := 162

-- Define the relationship to find the sales tax percentage
noncomputable def sales_tax_percent (before_tax after_tax : ℝ) : ℝ :=
  ((after_tax - before_tax) / before_tax) * 100

-- State the theorem to prove the sales tax percentage is 8%
theorem sales_tax_is_8_percent :
  sales_tax_percent total_before_tax total_with_tax = 8 :=
by
  -- skipping the proof
  sorry

end sales_tax_is_8_percent_l1504_150468


namespace even_odd_function_value_l1504_150496

theorem even_odd_function_value 
  (f g : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_odd : ∀ x, g (-x) = - g x)
  (h_eqn : ∀ x, f x + g x = x^3 + x^2 + 1) :
  f 1 - g 1 = 1 := 
by {
  sorry
}

end even_odd_function_value_l1504_150496


namespace difference_between_local_and_face_value_l1504_150423

def numeral := 657903

def local_value (n : ℕ) : ℕ :=
  if n = 7 then 70000 else 0

def face_value (n : ℕ) : ℕ :=
  n

theorem difference_between_local_and_face_value :
  local_value 7 - face_value 7 = 69993 :=
by
  sorry

end difference_between_local_and_face_value_l1504_150423


namespace range_of_a_l1504_150497

theorem range_of_a (a : ℝ) :
  (∃ M : ℝ × ℝ, (M.1 - a)^2 + (M.2 - a + 2)^2 = 1 ∧ (M.1)^2 + (M.2 + 3)^2 = 4 * ((M.1)^2 + (M.2)^2))
  → 0 ≤ a ∧ a ≤ 3 :=
sorry

end range_of_a_l1504_150497


namespace tina_savings_l1504_150422

theorem tina_savings :
  let june_savings : ℕ := 27
  let july_savings : ℕ := 14
  let august_savings : ℕ := 21
  let books_spending : ℕ := 5
  let shoes_spending : ℕ := 17
  let total_savings := june_savings + july_savings + august_savings
  let total_spending := books_spending + shoes_spending
  let remaining_money := total_savings - total_spending
  remaining_money = 40 :=
by
  sorry

end tina_savings_l1504_150422


namespace g_five_l1504_150463

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_multiplicative : ∀ x y : ℝ, g (x * y) = g x * g y
axiom g_zero : g 0 = 0
axiom g_one : g 1 = 1

theorem g_five : g 5 = 1 := by
  sorry

end g_five_l1504_150463


namespace seedling_prices_l1504_150499

theorem seedling_prices (x y : ℝ) (a b : ℝ) 
  (h1 : 3 * x + 2 * y = 12)
  (h2 : x + 3 * y = 11) 
  (h3 : a + b = 200) 
  (h4 : 2 * 100 * a + 3 * 100 * b ≥ 50000) :
  x = 2 ∧ y = 3 ∧ b ≥ 100 := 
sorry

end seedling_prices_l1504_150499


namespace prime_number_identity_l1504_150412

theorem prime_number_identity (p m : ℕ) (h1 : Nat.Prime p) (h2 : m > 0) (h3 : 2 * p^2 + p + 9 = m^2) :
  p = 5 ∧ m = 8 :=
sorry

end prime_number_identity_l1504_150412


namespace length_of_road_l1504_150428

-- Definitions based on conditions
def trees : Nat := 10
def interval : Nat := 10

-- Statement of the theorem
theorem length_of_road 
  (trees : Nat) (interval : Nat) (beginning_planting : Bool) (h_trees : trees = 10) (h_interval : interval = 10) (h_beginning : beginning_planting = true) 
  : (trees - 1) * interval = 90 := 
by 
  sorry

end length_of_road_l1504_150428


namespace eq_cont_fracs_l1504_150488

noncomputable def cont_frac : Nat -> Rat
| 0       => 0
| (n + 1) => (n : Rat) + 1 / (cont_frac n)

theorem eq_cont_fracs (n : Nat) : 
  1 - cont_frac n = cont_frac n - 1 :=
sorry

end eq_cont_fracs_l1504_150488


namespace geometric_sequence_a3_eq_sqrt_5_l1504_150424

theorem geometric_sequence_a3_eq_sqrt_5 (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * r)
  (h_a1 : a 1 = 1) (h_a5 : a 5 = 5) :
  a 3 = Real.sqrt 5 :=
sorry

end geometric_sequence_a3_eq_sqrt_5_l1504_150424


namespace arithmetic_seq_8th_term_l1504_150430

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l1504_150430


namespace range_of_a_for_circle_l1504_150425

theorem range_of_a_for_circle (a : ℝ) : 
  -2 < a ∧ a < 2/3 ↔ 
  ∃ (x y : ℝ), (x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1) = 0 :=
sorry

end range_of_a_for_circle_l1504_150425


namespace king_paid_after_tip_l1504_150437

-- Define the cost of the crown and the tip percentage
def cost_of_crown : ℝ := 20000
def tip_percentage : ℝ := 0.10

-- Define the total amount paid after the tip
def total_amount_paid (C : ℝ) (tip_pct : ℝ) : ℝ :=
  C + (tip_pct * C)

-- Theorem statement: The total amount paid after the tip is $22,000
theorem king_paid_after_tip : total_amount_paid cost_of_crown tip_percentage = 22000 := by
  sorry

end king_paid_after_tip_l1504_150437


namespace completing_the_square_l1504_150406

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end completing_the_square_l1504_150406


namespace union_of_S_and_T_l1504_150455

def S : Set ℕ := {1, 3, 5}
def T : Set ℕ := {3, 6}

theorem union_of_S_and_T : S ∪ T = {1, 3, 5, 6} := 
by
  sorry

end union_of_S_and_T_l1504_150455
