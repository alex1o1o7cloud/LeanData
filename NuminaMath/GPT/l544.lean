import Mathlib

namespace NUMINAMATH_GPT_simultaneous_eq_solvable_l544_54451

theorem simultaneous_eq_solvable (m : ℝ) : 
  (∃ x y : ℝ, y = m * x + 4 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_simultaneous_eq_solvable_l544_54451


namespace NUMINAMATH_GPT_sum_C_D_equals_seven_l544_54465

def initial_grid : Matrix (Fin 4) (Fin 4) (Option Nat) :=
  ![ ![ some 1, none, none, none ],
     ![ none, some 2, none, none ],
     ![ none, none, none, none ],
     ![ none, none, none, some 4 ] ]

def valid_grid (grid : Matrix (Fin 4) (Fin 4) (Option Nat)) : Prop :=
  ∀ i j, grid i j ≠ none →
    (∀ k, k ≠ j → grid i k ≠ grid i j) ∧ 
    (∀ k, k ≠ i → grid k j ≠ grid i j)

theorem sum_C_D_equals_seven :
  ∃ (C D : Nat), C + D = 7 ∧ valid_grid initial_grid :=
sorry

end NUMINAMATH_GPT_sum_C_D_equals_seven_l544_54465


namespace NUMINAMATH_GPT_total_clients_l544_54435

theorem total_clients (V K B N : Nat) (hV : V = 7) (hK : K = 8) (hB : B = 3) (hN : N = 18) :
    V + K - B + N = 30 := by
  sorry

end NUMINAMATH_GPT_total_clients_l544_54435


namespace NUMINAMATH_GPT_required_range_of_a_l544_54467

variable (a : ℝ) (f : ℝ → ℝ)
def function_increasing_on (f : ℝ → ℝ) (a : ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, DifferentiableAt ℝ f x ∧ (deriv f x) ≥ 0

theorem required_range_of_a (h : function_increasing_on (fun x => a * Real.log x + x) a (Set.Icc 2 3)) :
  a ≥ -2 :=
sorry

end NUMINAMATH_GPT_required_range_of_a_l544_54467


namespace NUMINAMATH_GPT_rods_in_one_mile_l544_54431

-- Define the given conditions
def mile_to_chains : ℕ := 10
def chain_to_rods : ℕ := 4

-- Prove the number of rods in one mile
theorem rods_in_one_mile : (1 * mile_to_chains * chain_to_rods) = 40 := by
  sorry

end NUMINAMATH_GPT_rods_in_one_mile_l544_54431


namespace NUMINAMATH_GPT_trigonometric_expression_value_l544_54487

theorem trigonometric_expression_value :
  4 * Real.cos (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) -
  Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 3 / 4 := sorry

end NUMINAMATH_GPT_trigonometric_expression_value_l544_54487


namespace NUMINAMATH_GPT_cos_pi_over_3_plus_2alpha_correct_l544_54442

noncomputable def cos_pi_over_3_plus_2alpha (α : Real) (h : Real.sin (Real.pi / 3 - α) = 1 / 4) : Real :=
  Real.cos (Real.pi / 3 + 2 * α)

theorem cos_pi_over_3_plus_2alpha_correct (α : Real) (h : Real.sin (Real.pi / 3 - α) = 1 / 4) :
  cos_pi_over_3_plus_2alpha α h = -7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_cos_pi_over_3_plus_2alpha_correct_l544_54442


namespace NUMINAMATH_GPT_triangle_properties_l544_54494

open Real

noncomputable def vec_m (a : ℝ) : ℝ × ℝ := (2 * sin (a / 2), sqrt 3)
noncomputable def vec_n (a : ℝ) : ℝ × ℝ := (cos a, 2 * cos (a / 4)^2 - 1)
noncomputable def area_triangle := 3 * sqrt 3 / 2

theorem triangle_properties (a b c : ℝ) (A : ℝ)
  (ha : a = sqrt 7)
  (hA : (1 / 2) * b * c * sin A = area_triangle)
  (hparallel : vec_m A = vec_n A) :
  A = π / 3 ∧ b + c = 5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_properties_l544_54494


namespace NUMINAMATH_GPT_shaded_triangle_probability_l544_54452

noncomputable def total_triangles : ℕ := 5
noncomputable def shaded_triangles : ℕ := 2
noncomputable def probability_shaded : ℚ := shaded_triangles / total_triangles

theorem shaded_triangle_probability : probability_shaded = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_shaded_triangle_probability_l544_54452


namespace NUMINAMATH_GPT_paint_containers_left_l544_54415

theorem paint_containers_left (initial_containers : ℕ)
  (tiled_wall_containers : ℕ)
  (ceiling_containers : ℕ)
  (gradient_walls : ℕ)
  (additional_gradient_containers_per_wall : ℕ)
  (remaining_containers : ℕ) :
  initial_containers = 16 →
  tiled_wall_containers = 1 →
  ceiling_containers = 1 →
  gradient_walls = 3 →
  additional_gradient_containers_per_wall = 1 →
  remaining_containers = initial_containers - tiled_wall_containers - (ceiling_containers + gradient_walls * additional_gradient_containers_per_wall) →
  remaining_containers = 11 :=
by
  intros h_initial h_tiled h_ceiling h_gradient_walls h_additional_gradient h_remaining_calc
  rw [h_initial, h_tiled, h_ceiling, h_gradient_walls, h_additional_gradient] at h_remaining_calc
  exact h_remaining_calc

end NUMINAMATH_GPT_paint_containers_left_l544_54415


namespace NUMINAMATH_GPT_remainder_of_division_l544_54499

-- Define the dividend and divisor
def dividend : ℕ := 3^303 + 303
def divisor : ℕ := 3^101 + 3^51 + 1

-- State the theorem to be proven
theorem remainder_of_division:
  (dividend % divisor) = 303 := by
  sorry

end NUMINAMATH_GPT_remainder_of_division_l544_54499


namespace NUMINAMATH_GPT_closest_approx_w_l544_54438

noncomputable def w : ℝ := ((69.28 * 123.57 * 0.004) - (42.67 * 3.12)) / (0.03 * 8.94 * 1.25)

theorem closest_approx_w : |w + 296.073| < 0.001 :=
by
  sorry

end NUMINAMATH_GPT_closest_approx_w_l544_54438


namespace NUMINAMATH_GPT_problem_statement_l544_54463

theorem problem_statement (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : (a - c) ^ 3 > (b - c) ^ 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l544_54463


namespace NUMINAMATH_GPT_father_current_age_is_85_l544_54492

theorem father_current_age_is_85 (sebastian_age : ℕ) (sister_diff : ℕ) (age_sum_fraction : ℕ → ℕ → ℕ → Prop) :
  sebastian_age = 40 →
  sister_diff = 10 →
  (∀ (s s' f : ℕ), age_sum_fraction s s' f → f = 4 * (s + s') / 3) →
  age_sum_fraction (sebastian_age - 5) (sebastian_age - sister_diff - 5) (40 + 5) →
  ∃ father_age : ℕ, father_age = 85 :=
by
  intros
  sorry

end NUMINAMATH_GPT_father_current_age_is_85_l544_54492


namespace NUMINAMATH_GPT_increased_cost_per_person_l544_54436

-- Declaration of constants
def initial_cost : ℕ := 30000000000 -- 30 billion dollars in dollars
def people_sharing : ℕ := 300000000 -- 300 million people
def inflation_rate : ℝ := 0.10 -- 10% inflation rate

-- Calculation of increased cost per person
theorem increased_cost_per_person : (initial_cost * (1 + inflation_rate) / people_sharing) = 110 :=
by sorry

end NUMINAMATH_GPT_increased_cost_per_person_l544_54436


namespace NUMINAMATH_GPT_trapezoid_base_ratio_l544_54446

theorem trapezoid_base_ratio 
  (a b h : ℝ) 
  (a_gt_b : a > b) 
  (quad_area_cond : (h * (a - b)) / 4 = (h * (a + b)) / 8) : 
  a = 3 * b := 
sorry

end NUMINAMATH_GPT_trapezoid_base_ratio_l544_54446


namespace NUMINAMATH_GPT_avg_first_six_results_l544_54445

theorem avg_first_six_results (average_11 : ℕ := 52) (average_last_6 : ℕ := 52) (sixth_result : ℕ := 34) :
  ∃ A : ℕ, (6 * A + 6 * average_last_6 - sixth_result = 11 * average_11) ∧ A = 49 :=
by
  sorry

end NUMINAMATH_GPT_avg_first_six_results_l544_54445


namespace NUMINAMATH_GPT_area_of_rectangular_garden_l544_54416

theorem area_of_rectangular_garden (length width : ℝ) (h_length : length = 2.5) (h_width : width = 0.48) :
  length * width = 1.2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangular_garden_l544_54416


namespace NUMINAMATH_GPT_solve_for_C_and_D_l544_54448

theorem solve_for_C_and_D (C D : ℚ) (h1 : 2 * C + 3 * D + 4 = 31) (h2 : D = C + 2) :
  C = 21 / 5 ∧ D = 31 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_C_and_D_l544_54448


namespace NUMINAMATH_GPT_correct_product_l544_54419

theorem correct_product (a b c : ℕ) (ha : 10 * c + 1 = a) (hb : 10 * c + 7 = a) 
(hl : (10 * c + 1) * b = 255) (hw : (10 * c + 7 + 6) * b = 335) : 
  a * b = 285 := 
  sorry

end NUMINAMATH_GPT_correct_product_l544_54419


namespace NUMINAMATH_GPT_find_y_l544_54430

variable (A B C : Point)

def carla_rotate (θ : ℕ) (A B C : Point) : Prop := 
  -- Definition to indicate point A rotated by θ degrees clockwise about point B lands at point C
  sorry

def devon_rotate (θ : ℕ) (A B C : Point) : Prop := 
  -- Definition to indicate point A rotated by θ degrees counterclockwise about point B lands at point C
  sorry

theorem find_y
  (h1 : carla_rotate 690 A B C)
  (h2 : ∀ y, devon_rotate y A B C)
  (h3 : y < 360) :
  ∃ y, y = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l544_54430


namespace NUMINAMATH_GPT_find_younger_age_l544_54402

def younger_age (y e : ℕ) : Prop :=
  (e = y + 20) ∧ (e - 5 = 5 * (y - 5))

theorem find_younger_age (y e : ℕ) (h : younger_age y e) : y = 10 :=
by sorry

end NUMINAMATH_GPT_find_younger_age_l544_54402


namespace NUMINAMATH_GPT_car_clock_time_correct_l544_54424

noncomputable def car_clock (t : ℝ) : ℝ := t * (4 / 3)

theorem car_clock_time_correct :
  ∀ t_real t_car,
  (car_clock 0 = 0) ∧
  (car_clock 0.5 = 2 / 3) ∧
  (car_clock t_real = t_car) ∧
  (t_car = (8 : ℝ)) → (t_real = 6) → (t_real + 1 = 7) :=
by
  intro t_real t_car h
  sorry

end NUMINAMATH_GPT_car_clock_time_correct_l544_54424


namespace NUMINAMATH_GPT_Walter_age_in_2003_l544_54455

-- Defining the conditions
def Walter_age_1998 (walter_age_1998 grandmother_age_1998 : ℝ) : Prop :=
  walter_age_1998 = grandmother_age_1998 / 3

def birth_years_sum (walter_age_1998 grandmother_age_1998 : ℝ) : Prop :=
  (1998 - walter_age_1998) + (1998 - grandmother_age_1998) = 3858

-- Defining the theorem to be proved
theorem Walter_age_in_2003 (walter_age_1998 grandmother_age_1998 : ℝ) 
  (h1 : Walter_age_1998 walter_age_1998 grandmother_age_1998) 
  (h2 : birth_years_sum walter_age_1998 grandmother_age_1998) : 
  walter_age_1998 + 5 = 39.5 :=
  sorry

end NUMINAMATH_GPT_Walter_age_in_2003_l544_54455


namespace NUMINAMATH_GPT_max_value_of_sum_l544_54480

open Real

theorem max_value_of_sum (x y z : ℝ)
    (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
    (h2 : (1 / x) + (1 / y) + (1 / z) + x + y + z = 0)
    (h3 : (x ≤ -1 ∨ x ≥ 1) ∧ (y ≤ -1 ∨ y ≥ 1) ∧ (z ≤ -1 ∨ z ≥ 1)) :
    x + y + z ≤ 0 := 
sorry

end NUMINAMATH_GPT_max_value_of_sum_l544_54480


namespace NUMINAMATH_GPT_sum_of_roots_is_zero_l544_54408

variable {R : Type*} [LinearOrderedField R]

-- Define the function f : R -> R and its properties
variable (f : R → R)
variable (even_f : ∀ x, f x = f (-x))
variable (roots_f : Finset R)
variable (roots_f_four : roots_f.card = 4)
variable (roots_f_set : ∀ x, x ∈ roots_f → f x = 0)

theorem sum_of_roots_is_zero : (roots_f.sum id) = 0 := 
sorry

end NUMINAMATH_GPT_sum_of_roots_is_zero_l544_54408


namespace NUMINAMATH_GPT_circle_area_ratio_l544_54417

/-- If the diameter of circle R is 60% of the diameter of circle S, 
the area of circle R is 36% of the area of circle S. -/
theorem circle_area_ratio (D_S D_R A_S A_R : ℝ) (h : D_R = 0.60 * D_S) 
  (hS : A_S = Real.pi * (D_S / 2) ^ 2) (hR : A_R = Real.pi * (D_R / 2) ^ 2): 
  A_R = 0.36 * A_S := 
sorry

end NUMINAMATH_GPT_circle_area_ratio_l544_54417


namespace NUMINAMATH_GPT_odd_function_behavior_l544_54473

-- Define that f is odd
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x

-- Define f for x > 0
def f_pos (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → (f x = (Real.log x / Real.log 2) - 2 * x)

-- Prove that for x < 0, f(x) == -log₂(-x) - 2x
theorem odd_function_behavior (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_pos : f_pos f) :
  ∀ x, x < 0 → f x = -((Real.log (-x)) / (Real.log 2)) - 2 * x := 
by
  sorry -- proof goes here

end NUMINAMATH_GPT_odd_function_behavior_l544_54473


namespace NUMINAMATH_GPT_farmer_initial_days_l544_54490

theorem farmer_initial_days 
  (x : ℕ) 
  (plan_daily : ℕ) 
  (actual_daily : ℕ) 
  (extra_days : ℕ) 
  (left_area : ℕ) 
  (total_area : ℕ)
  (h1 : plan_daily = 120) 
  (h2 : actual_daily = 85) 
  (h3 : extra_days = 2) 
  (h4 : left_area = 40) 
  (h5 : total_area = 720): 
  85 * (x + extra_days) + left_area = total_area → x = 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_farmer_initial_days_l544_54490


namespace NUMINAMATH_GPT_find_m_l544_54459

-- Define the given equations of the lines
def line1 (m : ℝ) : ℝ × ℝ → Prop := fun p => (3 + m) * p.1 - 4 * p.2 = 5 - 3 * m
def line2 : ℝ × ℝ → Prop := fun p => 2 * p.1 - p.2 = 8

-- Define the condition for parallel lines based on the given equations
def are_parallel (m : ℝ) : Prop := (3 + m) / 4 = 2

-- The main theorem stating the value of m
theorem find_m (m : ℝ) (h1 : ∀ p : ℝ × ℝ, line1 m p) (h2 : ∀ p : ℝ × ℝ, line2 p) (h_parallel : are_parallel m) : m = 5 :=
sorry

end NUMINAMATH_GPT_find_m_l544_54459


namespace NUMINAMATH_GPT_problem_equivalence_l544_54401

theorem problem_equivalence (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (⌊(a^2 : ℚ) / b⌋ + ⌊(b^2 : ℚ) / a⌋ = ⌊(a^2 + b^2 : ℚ) / (a * b)⌋ + a * b) ↔
  (∃ k : ℕ, k > 0 ∧ ((a = k ∧ b = k^2 + 1) ∨ (a = k^2 + 1 ∧ b = k))) :=
sorry

end NUMINAMATH_GPT_problem_equivalence_l544_54401


namespace NUMINAMATH_GPT_sqrt_eq_l544_54468

noncomputable def sqrt_22500 := 150

theorem sqrt_eq (h : sqrt_22500 = 150) : Real.sqrt 0.0225 = 0.15 :=
sorry

end NUMINAMATH_GPT_sqrt_eq_l544_54468


namespace NUMINAMATH_GPT_complement_of_A_in_reals_l544_54474

open Set

theorem complement_of_A_in_reals :
  (compl {x : ℝ | (x - 1) / (x - 2) ≥ 0}) = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_reals_l544_54474


namespace NUMINAMATH_GPT_contractor_daily_wage_l544_54484

theorem contractor_daily_wage 
  (total_days : ℕ)
  (daily_wage : ℝ)
  (fine_per_absence : ℝ)
  (total_earned : ℝ)
  (absent_days : ℕ)
  (H1 : total_days = 30)
  (H2 : fine_per_absence = 7.5)
  (H3 : total_earned = 555.0)
  (H4 : absent_days = 6)
  (H5 : total_earned = daily_wage * (total_days - absent_days) - fine_per_absence * absent_days) :
  daily_wage = 25 := by
  sorry

end NUMINAMATH_GPT_contractor_daily_wage_l544_54484


namespace NUMINAMATH_GPT_ratio_tin_copper_in_b_l544_54437

variable (L_a T_a T_b C_b : ℝ)

-- Conditions
axiom h1 : 170 + 250 = 420
axiom h2 : L_a / T_a = 1 / 3
axiom h3 : T_a + T_b = 221.25
axiom h4 : T_a + L_a = 170
axiom h5 : T_b + C_b = 250

-- Target
theorem ratio_tin_copper_in_b (h1 : 170 + 250 = 420) (h2 : L_a / T_a = 1 / 3)
  (h3 : T_a + T_b = 221.25) (h4 : T_a + L_a = 170) (h5 : T_b + C_b = 250) :
  T_b / C_b = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_tin_copper_in_b_l544_54437


namespace NUMINAMATH_GPT_quadrilateral_area_is_correct_l544_54423

-- Let's define the situation
structure TriangleDivisions where
  T1_area : ℝ
  T2_area : ℝ
  T3_area : ℝ
  Q_area : ℝ

def triangleDivisionExample : TriangleDivisions :=
  { T1_area := 4,
    T2_area := 9,
    T3_area := 9,
    Q_area := 36 }

-- The statement to prove
theorem quadrilateral_area_is_correct (T : TriangleDivisions) (h1 : T.T1_area = 4) 
  (h2 : T.T2_area = 9) (h3 : T.T3_area = 9) : T.Q_area = 36 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_is_correct_l544_54423


namespace NUMINAMATH_GPT_calculate_A_minus_B_l544_54461

variable (A B : ℝ)
variable (h1 : A + B + B = 814.8)
variable (h2 : 10 * B = A)

theorem calculate_A_minus_B : A - B = 611.1 :=
by
  sorry

end NUMINAMATH_GPT_calculate_A_minus_B_l544_54461


namespace NUMINAMATH_GPT_shape_is_cylinder_l544_54400

def positive_constant (c : ℝ) := c > 0

def is_cylinder (r θ z : ℝ) (c : ℝ) : Prop :=
  r = c

theorem shape_is_cylinder (c : ℝ) (r θ z : ℝ) 
  (h_pos : positive_constant c) (h_eq : r = c) :
  is_cylinder r θ z c := by
  sorry

end NUMINAMATH_GPT_shape_is_cylinder_l544_54400


namespace NUMINAMATH_GPT_length_of_BC_l544_54483

def triangle_perimeter (a b c : ℝ) : Prop :=
  a + b + c = 20

def triangle_area (a b : ℝ) : Prop :=
  (1/2) * a * b * (Real.sqrt 3 / 2) = 10

theorem length_of_BC (a b c : ℝ) (h1 : triangle_perimeter a b c) (h2 : triangle_area a b) : c = 7 :=
  sorry

end NUMINAMATH_GPT_length_of_BC_l544_54483


namespace NUMINAMATH_GPT_modem_B_download_time_l544_54404

theorem modem_B_download_time
    (time_A : ℝ) (speed_ratio : ℝ) 
    (h1 : time_A = 25.5) 
    (h2 : speed_ratio = 0.17) : 
    ∃ t : ℝ, t = 110.5425 := 
by
  sorry

end NUMINAMATH_GPT_modem_B_download_time_l544_54404


namespace NUMINAMATH_GPT_quadratic_roots_relation_l544_54454

theorem quadratic_roots_relation (a b s p : ℝ) (h : a^2 + b^2 = 15) (h1 : s = a + b) (h2 : p = a * b) : s^2 - 2 * p = 15 :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_relation_l544_54454


namespace NUMINAMATH_GPT_remainder_when_dividing_25197631_by_17_l544_54444

theorem remainder_when_dividing_25197631_by_17 :
  25197631 % 17 = 10 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_dividing_25197631_by_17_l544_54444


namespace NUMINAMATH_GPT_boyden_family_tickets_l544_54410

theorem boyden_family_tickets (child_ticket_cost : ℕ) (adult_ticket_cost : ℕ) (total_cost : ℕ) (num_adults : ℕ) (num_children : ℕ) :
  adult_ticket_cost = child_ticket_cost + 6 →
  total_cost = 77 →
  adult_ticket_cost = 19 →
  num_adults = 2 →
  num_children = 3 →
  num_adults * adult_ticket_cost + num_children * child_ticket_cost = total_cost →
  num_adults + num_children = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_boyden_family_tickets_l544_54410


namespace NUMINAMATH_GPT_sara_has_total_quarters_l544_54476

-- Define the number of quarters Sara originally had
def original_quarters : ℕ := 21

-- Define the number of quarters Sara's dad gave her
def added_quarters : ℕ := 49

-- Define the total number of quarters Sara has now
def total_quarters : ℕ := original_quarters + added_quarters

-- Prove that the total number of quarters is 70
theorem sara_has_total_quarters : total_quarters = 70 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_sara_has_total_quarters_l544_54476


namespace NUMINAMATH_GPT_rational_pair_exists_l544_54450

theorem rational_pair_exists (a b : ℚ) (h1 : a = 3/2) (h2 : b = 3) : a ≠ b ∧ a + b = a * b :=
by {
  sorry
}

end NUMINAMATH_GPT_rational_pair_exists_l544_54450


namespace NUMINAMATH_GPT_find_missing_number_l544_54493

theorem find_missing_number (x : ℕ) (h : (1 + x + 23 + 24 + 25 + 26 + 27 + 2) / 8 = 20) : x = 32 := 
by sorry

end NUMINAMATH_GPT_find_missing_number_l544_54493


namespace NUMINAMATH_GPT_jack_walked_distance_l544_54472

theorem jack_walked_distance (time_in_hours : ℝ) (rate : ℝ) (expected_distance : ℝ) : 
  time_in_hours = 1 + 15 / 60 ∧ 
  rate = 6.4 →
  expected_distance = 8 → 
  rate * time_in_hours = expected_distance :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_jack_walked_distance_l544_54472


namespace NUMINAMATH_GPT_smallest_integer_y_solution_l544_54426

theorem smallest_integer_y_solution :
  ∃ y : ℤ, (∀ z : ℤ, (z / 4 + 3 / 7 > 9 / 4) → (z ≥ y)) ∧ (y = 8) := 
by
  sorry

end NUMINAMATH_GPT_smallest_integer_y_solution_l544_54426


namespace NUMINAMATH_GPT_isosceles_triangle_angles_l544_54456

noncomputable 
def is_triangle_ABC_isosceles (A B C : ℝ) (alpha beta : ℝ) (AB AC : ℝ) 
  (h1 : AB = AC) (h2 : alpha = 2 * beta) : Prop :=
  180 - 3 * beta = C ∧ C / 2 = 90 - 1.5 * beta

theorem isosceles_triangle_angles (A B C C1 C2 : ℝ) (alpha beta : ℝ) (AB AC : ℝ)
  (h1 : AB = AC) (h2 : alpha = 2 * beta) :
  (180 - 3 * beta) / 2 = 90 - 1.5 * beta :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_angles_l544_54456


namespace NUMINAMATH_GPT_total_jelly_beans_l544_54460

-- Given conditions:
def vanilla_jelly_beans : ℕ := 120
def grape_jelly_beans := 5 * vanilla_jelly_beans + 50

-- The proof problem:
theorem total_jelly_beans :
  vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_total_jelly_beans_l544_54460


namespace NUMINAMATH_GPT_percentage_respondents_liked_B_l544_54496

variables (X Y : ℝ)
variables (likedA likedB likedBoth likedNeither : ℝ)
variables (totalRespondents : ℕ)

-- Conditions from the problem
def liked_conditions : Prop :=
    totalRespondents ≥ 100 ∧ 
    likedA = X ∧ 
    likedB = Y ∧ 
    likedBoth = 23 ∧ 
    likedNeither = 23

-- Proof statement
theorem percentage_respondents_liked_B (h : liked_conditions X Y likedA likedB likedBoth likedNeither totalRespondents) :
  Y = 100 - X :=
sorry

end NUMINAMATH_GPT_percentage_respondents_liked_B_l544_54496


namespace NUMINAMATH_GPT_complex_problem_l544_54486

open Complex

theorem complex_problem (a b : ℝ) (h : (2 + b * Complex.I) / (1 - Complex.I) = a * Complex.I) : a + b = 4 := by
  sorry

end NUMINAMATH_GPT_complex_problem_l544_54486


namespace NUMINAMATH_GPT_concatenated_number_divisible_by_37_l544_54485

theorem concatenated_number_divisible_by_37
  (a b : ℕ) (ha : 100 ≤ a ∧ a ≤ 999) (hb : 100 ≤ b ∧ b ≤ 999)
  (h₁ : a % 37 ≠ 0) (h₂ : b % 37 ≠ 0) (h₃ : (a + b) % 37 = 0) :
  (1000 * a + b) % 37 = 0 :=
sorry

end NUMINAMATH_GPT_concatenated_number_divisible_by_37_l544_54485


namespace NUMINAMATH_GPT_volume_of_prism_l544_54471

-- Given dimensions a, b, and c, with the following conditions:
variables (a b c : ℝ)
axiom ab_eq_30 : a * b = 30
axiom ac_eq_40 : a * c = 40
axiom bc_eq_60 : b * c = 60

-- The volume of the prism is given by:
theorem volume_of_prism : a * b * c = 120 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l544_54471


namespace NUMINAMATH_GPT_right_triangle_sides_l544_54498

theorem right_triangle_sides 
  (a b c : ℝ) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_area : (1 / 2) * a * b = 150) 
  (h_perimeter : a + b + c = 60) 
  : (a = 15 ∧ b = 20 ∧ c = 25) ∨ (a = 20 ∧ b = 15 ∧ c = 25) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_sides_l544_54498


namespace NUMINAMATH_GPT_sqrt_25_eq_pm_5_l544_54409

theorem sqrt_25_eq_pm_5 : {x : ℝ | x^2 = 25} = {5, -5} :=
by
  sorry

end NUMINAMATH_GPT_sqrt_25_eq_pm_5_l544_54409


namespace NUMINAMATH_GPT_sqrt_D_irrational_l544_54478

theorem sqrt_D_irrational (a b c : ℤ) (h : a + 1 = b) (h_c : c = a + b) : 
  Irrational (Real.sqrt ((a^2 : ℤ) + (b^2 : ℤ) + (c^2 : ℤ))) :=
  sorry

end NUMINAMATH_GPT_sqrt_D_irrational_l544_54478


namespace NUMINAMATH_GPT_flea_reach_B_with_7_jumps_flea_reach_C_with_9_jumps_flea_cannot_reach_D_with_2028_jumps_l544_54422

-- Problem (a)
theorem flea_reach_B_with_7_jumps (A B : ℤ) (jumps : ℤ) (distance : ℤ) (ways : ℕ) :
  B = A + 5 → jumps = 7 → distance = 5 → 
  ways = Nat.choose (7) (1) := 
sorry

-- Problem (b)
theorem flea_reach_C_with_9_jumps (A C : ℤ) (jumps : ℤ) (distance : ℤ) (ways : ℕ) :
  C = A + 5 → jumps = 9 → distance = 5 → 
  ways = Nat.choose (9) (2) :=
sorry

-- Problem (c)
theorem flea_cannot_reach_D_with_2028_jumps (A D : ℤ) (jumps : ℤ) (distance : ℤ) :
  D = A + 2013 → jumps = 2028 → distance = 2013 → 
  ∃ x y : ℤ, x + y = 2028 ∧ x - y = 2013 → false :=
sorry

end NUMINAMATH_GPT_flea_reach_B_with_7_jumps_flea_reach_C_with_9_jumps_flea_cannot_reach_D_with_2028_jumps_l544_54422


namespace NUMINAMATH_GPT_number_of_ways_to_select_one_person_number_of_ways_to_select_one_person_each_type_l544_54418

-- Definitions based on the conditions
def peopleO : ℕ := 28
def peopleA : ℕ := 7
def peopleB : ℕ := 9
def peopleAB : ℕ := 3

-- Proof for Question 1
theorem number_of_ways_to_select_one_person : peopleO + peopleA + peopleB + peopleAB = 47 := by
  sorry

-- Proof for Question 2
theorem number_of_ways_to_select_one_person_each_type : peopleO * peopleA * peopleB * peopleAB = 5292 := by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_select_one_person_number_of_ways_to_select_one_person_each_type_l544_54418


namespace NUMINAMATH_GPT_find_n_given_sum_l544_54475

noncomputable def geometric_sequence_general_term (n : ℕ) : ℝ :=
  if n ≥ 2 then 2^(2 * n - 3) else 0

def b_n (n : ℕ) : ℝ :=
  2 * n - 3

def sum_b_n (n : ℕ) : ℝ :=
  n^2 - 2 * n

theorem find_n_given_sum : ∃ n : ℕ, sum_b_n n = 360 :=
  by { use 20, sorry }

end NUMINAMATH_GPT_find_n_given_sum_l544_54475


namespace NUMINAMATH_GPT_geometric_seq_common_ratio_l544_54495

theorem geometric_seq_common_ratio 
  (a : ℝ) (q : ℝ)
  (h1 : a * q^2 = 4)
  (h2 : a * q^5 = 1 / 2) : 
  q = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_geometric_seq_common_ratio_l544_54495


namespace NUMINAMATH_GPT_equidistant_point_x_axis_l544_54405

theorem equidistant_point_x_axis (x : ℝ) (C D : ℝ × ℝ)
  (hC : C = (-3, 0))
  (hD : D = (0, 5))
  (heqdist : ∀ p : ℝ × ℝ, p.2 = 0 → 
    dist p C = dist p D) :
  x = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_equidistant_point_x_axis_l544_54405


namespace NUMINAMATH_GPT_landscape_breadth_l544_54489

theorem landscape_breadth (L B : ℝ) 
  (h1 : B = 6 * L) 
  (h2 : L * B = 29400) : 
  B = 420 :=
by
  sorry

end NUMINAMATH_GPT_landscape_breadth_l544_54489


namespace NUMINAMATH_GPT_max_value_y_l544_54429

theorem max_value_y (x : ℝ) (h : x < 5 / 4) : 
  (4 * x - 2 + 1 / (4 * x - 5)) ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_y_l544_54429


namespace NUMINAMATH_GPT_number_of_quartets_l544_54457

theorem number_of_quartets :
  let n := 5
  let factorial (x : Nat) := Nat.factorial x
  factorial n ^ 3 = 120 ^ 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_quartets_l544_54457


namespace NUMINAMATH_GPT_minimize_slope_at_one_l544_54443

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  2 * a * x^2 - (1 / (a * x))

noncomputable def f_deriv (a x : ℝ) : ℝ :=
  4 * a * x - (1 / (a * x^2))

noncomputable def slope_at_one (a : ℝ) : ℝ :=
  f_deriv a 1

theorem minimize_slope_at_one : ∀ a : ℝ, a > 0 → slope_at_one a ≥ 4 ∧ (slope_at_one a = 4 ↔ a = 1 / 2) :=
by 
  sorry

end NUMINAMATH_GPT_minimize_slope_at_one_l544_54443


namespace NUMINAMATH_GPT_min_f_x_eq_one_implies_a_eq_zero_or_two_l544_54403

theorem min_f_x_eq_one_implies_a_eq_zero_or_two (a : ℝ) :
  (∃ x : ℝ, |x + 1| + |x + a| = 1) → (a = 0 ∨ a = 2) := by
  sorry

end NUMINAMATH_GPT_min_f_x_eq_one_implies_a_eq_zero_or_two_l544_54403


namespace NUMINAMATH_GPT_min_b1_b2_sum_l544_54464

def sequence_relation (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (3 * b n + 4073) / (2 + b (n + 1))

theorem min_b1_b2_sum (b : ℕ → ℕ) (h_seq : sequence_relation b) 
  (h_b1_pos : b 1 > 0) (h_b2_pos : b 2 > 0) :
  b 1 + b 2 = 158 :=
sorry

end NUMINAMATH_GPT_min_b1_b2_sum_l544_54464


namespace NUMINAMATH_GPT_eight_row_triangle_pieces_l544_54491

def unit_rods (n : ℕ) : ℕ := 3 * (n * (n + 1)) / 2

def connectors (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem eight_row_triangle_pieces : unit_rods 8 + connectors 9 = 153 :=
by
  sorry

end NUMINAMATH_GPT_eight_row_triangle_pieces_l544_54491


namespace NUMINAMATH_GPT_integer_solution_of_inequality_l544_54481

theorem integer_solution_of_inequality :
  ∀ (x : ℤ), 0 < (x - 1 : ℚ) * (x - 1) / (x + 1) ∧ (x - 1) * (x - 1) / (x + 1) < 1 →
  x > -1 ∧ x ≠ 1 ∧ x < 3 → 
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_of_inequality_l544_54481


namespace NUMINAMATH_GPT_log_sum_l544_54440

theorem log_sum : (Real.log 0.01 / Real.log 10) + (Real.log 16 / Real.log 2) = 2 := by
  sorry

end NUMINAMATH_GPT_log_sum_l544_54440


namespace NUMINAMATH_GPT_max_distinct_sums_l544_54428

/-- Given 3 boys and 20 girls standing in a row, each child counts the number of girls to their 
left and the number of boys to their right and adds these two counts together. Prove that 
the maximum number of different sums that the children could have obtained is 20. -/
theorem max_distinct_sums (boys girls : ℕ) (total_children : ℕ) 
  (h_boys : boys = 3) (h_girls : girls = 20) (h_total : total_children = boys + girls) : 
  ∃ (max_sums : ℕ), max_sums = 20 := 
by 
  sorry

end NUMINAMATH_GPT_max_distinct_sums_l544_54428


namespace NUMINAMATH_GPT_envelope_weight_l544_54488

theorem envelope_weight :
  (7.225 * 1000) / 850 = 8.5 :=
by
  sorry

end NUMINAMATH_GPT_envelope_weight_l544_54488


namespace NUMINAMATH_GPT_field_trip_vans_l544_54497

-- Define the number of students and adults
def students := 12
def adults := 3

-- Define the capacity of each van
def van_capacity := 5

-- Total number of people
def total_people := students + adults

-- Calculate the number of vans needed
def vans_needed := (total_people + van_capacity - 1) / van_capacity  -- For rounding up division

theorem field_trip_vans : vans_needed = 3 :=
by
  -- Calculation and proof would go here
  sorry

end NUMINAMATH_GPT_field_trip_vans_l544_54497


namespace NUMINAMATH_GPT_moles_of_CO2_required_l544_54462

theorem moles_of_CO2_required (n_H2O n_H2CO3 : ℕ) (h1 : n_H2O = n_H2CO3) (h2 : n_H2O = 2): 
  (n_H2O = 2) → (∃ n_CO2 : ℕ, n_CO2 = n_H2O) :=
by
  sorry

end NUMINAMATH_GPT_moles_of_CO2_required_l544_54462


namespace NUMINAMATH_GPT_smallest_x_l544_54466

theorem smallest_x {
    x : ℤ
} : (x % 11 = 9) ∧ (x % 13 = 11) ∧ (x % 15 = 13) → x = 2143 := by
sorry

end NUMINAMATH_GPT_smallest_x_l544_54466


namespace NUMINAMATH_GPT_count_routes_from_A_to_B_l544_54411

-- Define cities as an inductive type
inductive City
| A
| B
| C
| D
| E

-- Define roads as a list of pairs of cities
def roads : List (City × City) := [
  (City.A, City.B),
  (City.A, City.D),
  (City.B, City.D),
  (City.C, City.D),
  (City.D, City.E),
  (City.B, City.E)
]

-- Define the problem statement
noncomputable def route_count : ℕ :=
  3  -- This should be proven

theorem count_routes_from_A_to_B : route_count = 3 :=
  by
    sorry  -- Proof goes here

end NUMINAMATH_GPT_count_routes_from_A_to_B_l544_54411


namespace NUMINAMATH_GPT_exterior_angle_of_octagon_is_45_degrees_l544_54427

noncomputable def exterior_angle_of_regular_octagon : ℝ :=
  let n : ℝ := 8
  let interior_angle_sum := 180 * (n - 2) -- This is the sum of interior angles of any n-gon
  let each_interior_angle := interior_angle_sum / n -- Each interior angle in a regular polygon
  let each_exterior_angle := 180 - each_interior_angle -- Exterior angle is supplement of interior angle
  each_exterior_angle

theorem exterior_angle_of_octagon_is_45_degrees :
  exterior_angle_of_regular_octagon = 45 := by
  sorry

end NUMINAMATH_GPT_exterior_angle_of_octagon_is_45_degrees_l544_54427


namespace NUMINAMATH_GPT_complex_quadrant_l544_54433

theorem complex_quadrant (i : ℂ) (h_imag : i = Complex.I) :
  let z := (1 + i)⁻¹
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end NUMINAMATH_GPT_complex_quadrant_l544_54433


namespace NUMINAMATH_GPT_euler_distance_formula_l544_54479

theorem euler_distance_formula 
  (d R r : ℝ) 
  (h₁ : d = distance_between_centers_of_inscribed_and_circumscribed_circles_of_triangle)
  (h₂ : R = circumradius_of_triangle)
  (h₃ : r = inradius_of_triangle) : 
  d^2 = R^2 - 2 * R * r := 
sorry

end NUMINAMATH_GPT_euler_distance_formula_l544_54479


namespace NUMINAMATH_GPT_cost_buses_minimize_cost_buses_l544_54414

theorem cost_buses
  (x y : ℕ) 
  (h₁ : x + y = 500)
  (h₂ : 2 * x + 3 * y = 1300) :
  x = 200 ∧ y = 300 :=
by 
  sorry

theorem minimize_cost_buses
  (m : ℕ) 
  (h₃: 15 * m + 25 * (8 - m) ≥ 180) :
  m = 2 ∧ (200 * m + 300 * (8 - m) = 2200) :=
by 
  sorry

end NUMINAMATH_GPT_cost_buses_minimize_cost_buses_l544_54414


namespace NUMINAMATH_GPT_tank_capacity_l544_54434

theorem tank_capacity :
  (∃ (C : ℕ), ∀ (leak_rate inlet_rate net_rate : ℕ),
    leak_rate = C / 6 ∧
    inlet_rate = 6 * 60 ∧
    net_rate = C / 12 ∧
    inlet_rate - leak_rate = net_rate → C = 1440) :=
sorry

end NUMINAMATH_GPT_tank_capacity_l544_54434


namespace NUMINAMATH_GPT_total_cost_of_products_l544_54407

-- Conditions
def smartphone_price := 300
def personal_computer_price := smartphone_price + 500
def advanced_tablet_price := smartphone_price + personal_computer_price

-- Theorem statement for the total cost of one of each product
theorem total_cost_of_products :
  smartphone_price + personal_computer_price + advanced_tablet_price = 2200 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_products_l544_54407


namespace NUMINAMATH_GPT_geese_flock_size_l544_54425

theorem geese_flock_size : 
  ∃ x : ℕ, x + x + (x / 2) + (x / 4) + 1 = 100 ∧ x = 36 := 
by
  sorry

end NUMINAMATH_GPT_geese_flock_size_l544_54425


namespace NUMINAMATH_GPT_votes_for_winner_is_744_l544_54449

variable (V : ℝ) -- Total number of votes cast

-- Conditions
axiom two_candidates : True
axiom winner_received_62_percent : True
axiom winner_won_by_288_votes : 0.62 * V - 0.38 * V = 288

-- Theorem to prove
theorem votes_for_winner_is_744 :
  0.62 * V = 744 :=
by
  sorry

end NUMINAMATH_GPT_votes_for_winner_is_744_l544_54449


namespace NUMINAMATH_GPT_student_score_l544_54447

theorem student_score
    (total_questions : ℕ)
    (correct_responses : ℕ)
    (grading_method : ℕ → ℕ → ℕ)
    (h1 : total_questions = 100)
    (h2 : correct_responses = 92)
    (h3 : grading_method = λ correct incorrect => correct - 2 * incorrect) :
  grading_method correct_responses (total_questions - correct_responses) = 76 :=
by
  -- proof would be here, but is skipped
  sorry

end NUMINAMATH_GPT_student_score_l544_54447


namespace NUMINAMATH_GPT_find_a5_a6_l544_54469

namespace ArithmeticSequence

variable {a : ℕ → ℝ}

-- Given conditions
axiom h1 : a 1 + a 2 = 5
axiom h2 : a 3 + a 4 = 7

-- Arithmetic sequence property
axiom arith_seq : ∀ n : ℕ, a (n + 1) = a n + (a 2 - a 1)

-- The statement we want to prove
theorem find_a5_a6 : a 5 + a 6 = 9 :=
sorry

end ArithmeticSequence

end NUMINAMATH_GPT_find_a5_a6_l544_54469


namespace NUMINAMATH_GPT_minimum_omega_l544_54421

theorem minimum_omega 
  (ω : ℝ)
  (hω : ω > 0)
  (h_shift : ∃ T > 0, T = 2 * π / ω ∧ T = 2 * π / 3) : 
  ω = 3 := 
sorry

end NUMINAMATH_GPT_minimum_omega_l544_54421


namespace NUMINAMATH_GPT_propositions_imply_implication_l544_54439

theorem propositions_imply_implication (p q r : Prop) :
  ( ((p ∧ q ∧ ¬r) → ((p ∧ q) → r) = False) ∧ 
    ((¬p ∧ q ∧ r) → ((p ∧ q) → r) = True) ∧ 
    ((p ∧ ¬q ∧ r) → ((p ∧ q) → r) = True) ∧ 
    ((¬p ∧ ¬q ∧ ¬r) → ((p ∧ q) → r) = True) ) → 
  ( (∀ (x : ℕ), x = 3) ) :=
by
  sorry

end NUMINAMATH_GPT_propositions_imply_implication_l544_54439


namespace NUMINAMATH_GPT_quadratic_value_at_sum_of_roots_is_five_l544_54453

noncomputable def quadratic_func (a b x : ℝ) : ℝ := a * x^2 + b * x + 5

theorem quadratic_value_at_sum_of_roots_is_five
  (a b x₁ x₂ : ℝ)
  (hA : quadratic_func a b x₁ = 2023)
  (hB : quadratic_func a b x₂ = 2023)
  (ha : a ≠ 0) :
  quadratic_func a b (x₁ + x₂) = 5 :=
sorry

end NUMINAMATH_GPT_quadratic_value_at_sum_of_roots_is_five_l544_54453


namespace NUMINAMATH_GPT_number_of_days_b_worked_l544_54420

variables (d_a : ℕ) (d_c : ℕ) (total_earnings : ℝ)
variables (wage_ratio : ℝ) (wage_c : ℝ) (d_b : ℕ) (wages : ℝ)
variables (total_wage_a : ℝ) (total_wage_c : ℝ) (total_wage_b : ℝ)

-- Given conditions
def given_conditions :=
  d_a = 6 ∧
  d_c = 4 ∧
  wage_c = 95 ∧
  wage_ratio = wage_c / 5 ∧
  wages = 3 * wage_ratio ∧
  total_earnings = 1406 ∧
  total_wage_a = d_a * wages ∧
  total_wage_c = d_c * wage_c ∧
  total_wage_b = d_b * (4 * wage_ratio) ∧
  total_wage_a + total_wage_b + total_wage_c = total_earnings

-- Theorem to prove
theorem number_of_days_b_worked :
  given_conditions d_a d_c total_earnings wage_ratio wage_c d_b wages total_wage_a total_wage_c total_wage_b →
  d_b = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_days_b_worked_l544_54420


namespace NUMINAMATH_GPT_tangent_line_ellipse_l544_54413

variables {x y x0 y0 r a b : ℝ}

/-- Given the tangent line to the circle x^2 + y^2 = r^2 at the point (x0, y0) is x0 * x + y0 * y = r^2,
we prove the tangent line to the ellipse x^2 / a^2 + y^2 / b^2 = 1 at the point (x0, y0) is x0 * x / a^2 + y0 * y / b^2 = 1. -/
theorem tangent_line_ellipse :
  (x0 * x + y0 * y = r^2) →
  (x0^2 / a^2 + y0^2 / b^2 = 1) →
  (x0 * x / a^2 + y0 * y / b^2 = 1) :=
by
  intros hc he
  sorry

end NUMINAMATH_GPT_tangent_line_ellipse_l544_54413


namespace NUMINAMATH_GPT_find_denominator_l544_54406

theorem find_denominator (y x : ℝ) (hy : y > 0) (h : (1 * y) / x + (3 * y) / 10 = 0.35 * y) : x = 20 := by
  sorry

end NUMINAMATH_GPT_find_denominator_l544_54406


namespace NUMINAMATH_GPT_sinA_mul_sinC_eq_three_fourths_l544_54458
open Real

-- Definitions based on conditions
def angles_form_arithmetic_sequence (A B C : ℝ) : Prop :=
  2 * B = A + C

def sides_form_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

-- The theorem to prove
theorem sinA_mul_sinC_eq_three_fourths
  (A B C a b c : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_angles : A + B + C = π)
  (h_angles_arithmetic : angles_form_arithmetic_sequence A B C)
  (h_sides_geometric : sides_form_geometric_sequence a b c) :
  sin A * sin C = 3 / 4 :=
sorry

end NUMINAMATH_GPT_sinA_mul_sinC_eq_three_fourths_l544_54458


namespace NUMINAMATH_GPT_sqrt_cos_sin_relation_l544_54482

variable {a b c θ : ℝ}

theorem sqrt_cos_sin_relation 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * (Real.cos θ) ^ 2 + b * (Real.sin θ) ^ 2 < c) :
  Real.sqrt a * (Real.cos θ) ^ 2 + Real.sqrt b * (Real.sin θ) ^ 2 < Real.sqrt c :=
sorry

end NUMINAMATH_GPT_sqrt_cos_sin_relation_l544_54482


namespace NUMINAMATH_GPT_even_function_implies_a_is_2_l544_54412

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end NUMINAMATH_GPT_even_function_implies_a_is_2_l544_54412


namespace NUMINAMATH_GPT_correct_propositions_l544_54441

noncomputable def proposition1 : Prop :=
  (∀ x : ℝ, x^2 - 3 * x + 2 = 0 -> x = 1) ->
  (∀ x : ℝ, x ≠ 1 -> x^2 - 3 * x + 2 ≠ 0)

noncomputable def proposition2 : Prop :=
  (∀ p q : Prop, p ∨ q -> p ∧ q) ->
  (∀ p q : Prop, p ∧ q -> p ∨ q)

noncomputable def proposition3 : Prop :=
  (∀ p q : Prop, ¬(p ∧ q) -> ¬p ∧ ¬q)

noncomputable def proposition4 : Prop :=
  (∃ x : ℝ, x^2 + x + 1 < 0) ->
  (∀ x : ℝ, x^2 + x + 1 ≥ 0)

theorem correct_propositions :
  proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4 :=
by sorry

end NUMINAMATH_GPT_correct_propositions_l544_54441


namespace NUMINAMATH_GPT_olivia_spent_38_l544_54432

def initial_amount : ℕ := 128
def amount_left : ℕ := 90
def money_spent (initial amount_left : ℕ) : ℕ := initial - amount_left

theorem olivia_spent_38 :
  money_spent initial_amount amount_left = 38 :=
by 
  sorry

end NUMINAMATH_GPT_olivia_spent_38_l544_54432


namespace NUMINAMATH_GPT_number_of_correct_conclusions_is_two_l544_54470

section AnalogicalReasoning
  variable (a b c : ℝ) (x y : ℂ)

  -- Condition 1: The analogy for distributive property over addition in ℝ and division
  def analogy1 : (c ≠ 0) → ((a + b) * c = a * c + b * c) → (a + b) / c = a / c + b / c := by
    sorry

  -- Condition 2: The analogy for equality of real and imaginary parts in ℂ
  def analogy2 : (x - y = 0) → x = y := by
    sorry

  -- Theorem stating that the number of correct conclusions is 2
  theorem number_of_correct_conclusions_is_two : 2 = 2 := by
    -- which implies that analogy1 and analogy2 are valid, and the other two analogies are not
    sorry

end AnalogicalReasoning

end NUMINAMATH_GPT_number_of_correct_conclusions_is_two_l544_54470


namespace NUMINAMATH_GPT_sasha_fractions_l544_54477

theorem sasha_fractions (x y z t : ℕ) 
  (hx : x ≠ y) (hxy : x ≠ z) (hxz : x ≠ t)
  (hyz : y ≠ z) (hyt : y ≠ t) (hzt : z ≠ t) :
  ∃ (q1 q2 : ℚ), (q1 ≠ q2) ∧ 
    (q1 = x / y ∨ q1 = x / z ∨ q1 = x / t ∨ q1 = y / x ∨ q1 = y / z ∨ q1 = y / t ∨ q1 = z / x ∨ q1 = z / y ∨ q1 = z / t ∨ q1 = t / x ∨ q1 = t / y ∨ q1 = t / z) ∧ 
    (q2 = x / y ∨ q2 = x / z ∨ q2 = x / t ∨ q2 = y / x ∨ q2 = y / z ∨ q2 = y / t ∨ q2 = z / x ∨ q2 = z / y ∨ q2 = z / t ∨ q2 = t / x ∨ q2 = t / y ∨ q2 = t / z) ∧ 
    |q1 - q2| ≤ 11 / 60 := by 
  sorry

end NUMINAMATH_GPT_sasha_fractions_l544_54477
