import Mathlib

namespace NUMINAMATH_GPT_solve_inequality_l177_17740

theorem solve_inequality (a b : ℝ) (h : ∀ x, (x > 1 ∧ x < 2) ↔ (x - a) * (x - b) < 0) : a + b = 3 :=
sorry

end NUMINAMATH_GPT_solve_inequality_l177_17740


namespace NUMINAMATH_GPT_smallest_factor_l177_17791

theorem smallest_factor (x : ℕ) (h1 : 936 = 2^3 * 3^1 * 13^1)
  (h2 : ∃ (x : ℕ), (936 * x) % 2^5 = 0 ∧ (936 * x) % 3^3 = 0 ∧ (936 * x) % 13^2 = 0) : x = 468 := 
sorry

end NUMINAMATH_GPT_smallest_factor_l177_17791


namespace NUMINAMATH_GPT_color_opposite_gold_is_yellow_l177_17750

-- Define the colors as a datatype for clarity
inductive Color
| B | Y | O | K | S | G

-- Define the type for each face's color
structure CubeFaces :=
(top front right back left bottom : Color)

-- Given conditions
def first_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.Y ∧ c.right = Color.O

def second_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.K ∧ c.right = Color.O

def third_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.S ∧ c.right = Color.O

-- Problem statement
theorem color_opposite_gold_is_yellow (c : CubeFaces) :
  first_view c → second_view c → third_view c → (c.back = Color.G) → (c.front = Color.Y) :=
by
  sorry

end NUMINAMATH_GPT_color_opposite_gold_is_yellow_l177_17750


namespace NUMINAMATH_GPT_geometric_series_sum_l177_17739

-- Define the geometric series
def geometricSeries (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

-- Define the conditions
def a : ℚ := 1 / 4
def r : ℚ := 1 / 4
def n : ℕ := 5

-- Define the sum of the first n terms using the provided formula
def S_n := geometricSeries a r n

-- State the theorem: the sum S_5 equals the given answer
theorem geometric_series_sum :
  S_n = 1023 / 3072 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l177_17739


namespace NUMINAMATH_GPT_engagement_ring_savings_l177_17741

theorem engagement_ring_savings 
  (yearly_salary : ℝ) 
  (monthly_savings : ℝ) 
  (monthly_salary := yearly_salary / 12) 
  (ring_cost := 2 * monthly_salary) 
  (saving_months := ring_cost / monthly_savings) 
  (h_salary : yearly_salary = 60000) 
  (h_savings : monthly_savings = 1000) :
  saving_months = 10 := 
sorry

end NUMINAMATH_GPT_engagement_ring_savings_l177_17741


namespace NUMINAMATH_GPT_solve_system_equations_l177_17796

theorem solve_system_equations :
  ∃ x y : ℚ, (5 * x * (y + 6) = 0 ∧ 2 * x + 3 * y = 1) ∧
  (x = 0 ∧ y = 1 / 3 ∨ x = 19 / 2 ∧ y = -6) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_equations_l177_17796


namespace NUMINAMATH_GPT_ratio_of_elements_l177_17738

theorem ratio_of_elements (total_weight : ℕ) (element_B_weight : ℕ) 
  (h_total : total_weight = 324) (h_B : element_B_weight = 270) :
  (total_weight - element_B_weight) / element_B_weight = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_elements_l177_17738


namespace NUMINAMATH_GPT_rewrite_equation_l177_17761

theorem rewrite_equation (x y : ℝ) (h : 2 * x - y = 4) : y = 2 * x - 4 :=
by
  sorry

end NUMINAMATH_GPT_rewrite_equation_l177_17761


namespace NUMINAMATH_GPT_Haman_initial_trays_l177_17776

theorem Haman_initial_trays 
  (eggs_in_tray : ℕ)
  (total_eggs_sold : ℕ)
  (trays_dropped : ℕ)
  (additional_trays : ℕ)
  (trays_finally_sold : ℕ)
  (std_trays_sold : total_eggs_sold / eggs_in_tray = trays_finally_sold) 
  (eggs_in_tray_def : eggs_in_tray = 30) 
  (total_eggs_sold_def : total_eggs_sold = 540)
  (trays_dropped_def : trays_dropped = 2)
  (additional_trays_def : additional_trays = 7) :
  trays_finally_sold - additional_trays + trays_dropped = 13 := 
by 
  sorry

end NUMINAMATH_GPT_Haman_initial_trays_l177_17776


namespace NUMINAMATH_GPT_min_value_ax_over_rR_l177_17737

theorem min_value_ax_over_rR (a b c r R : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_le_b : a ≤ b) (h_le_c : a ≤ c) (h_inradius : ∀ (a b c : ℝ), r = 2 * area / (a + b + c))
  (h_circumradius : ∀ (a b c : ℝ), R = (a * b * c) / (4 * area))
  (x : ℝ) (h_x : x = (b + c - a) / 2) (area : ℝ) :
  (a * x / (r * R)) ≥ 3 :=
sorry

end NUMINAMATH_GPT_min_value_ax_over_rR_l177_17737


namespace NUMINAMATH_GPT_min_value_frac_l177_17716

theorem min_value_frac (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 1) : 
  (1 / m + 2 / n) = 8 :=
sorry

end NUMINAMATH_GPT_min_value_frac_l177_17716


namespace NUMINAMATH_GPT_negative_solution_condition_l177_17717

variable {a b c x y : ℝ}

theorem negative_solution_condition (h1 : a * x + b * y = c)
    (h2 : b * x + c * y = a)
    (h3 : c * x + a * y = b)
    (hx : x < 0)
    (hy : y < 0) :
    a + b + c = 0 :=
sorry

end NUMINAMATH_GPT_negative_solution_condition_l177_17717


namespace NUMINAMATH_GPT_solve_for_x_l177_17735

theorem solve_for_x (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 + 8 * x - 16 = 0) : x = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l177_17735


namespace NUMINAMATH_GPT_solve_r_l177_17799

variable (r : ℝ)

theorem solve_r : (r + 3) / (r - 2) = (r - 1) / (r + 1) → r = -1/7 := by
  sorry

end NUMINAMATH_GPT_solve_r_l177_17799


namespace NUMINAMATH_GPT_mark_reading_time_l177_17708

-- Definitions based on conditions
def daily_reading_hours : ℕ := 3
def days_in_week : ℕ := 7
def weekly_increase : ℕ := 6

-- Proof statement
theorem mark_reading_time : daily_reading_hours * days_in_week + weekly_increase = 27 := by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_mark_reading_time_l177_17708


namespace NUMINAMATH_GPT_system_of_equations_l177_17759

theorem system_of_equations (x y k : ℝ) 
  (h1 : x + 2 * y = k + 2) 
  (h2 : 2 * x - 3 * y = 3 * k - 1) : 
  x + 9 * y = 7 :=
  sorry

end NUMINAMATH_GPT_system_of_equations_l177_17759


namespace NUMINAMATH_GPT_ratio_of_square_sides_sum_l177_17702

theorem ratio_of_square_sides_sum 
  (h : (75:ℚ) / 128 = (75:ℚ) / 128) :
  let a := 5
  let b := 6
  let c := 16
  a + b + c = 27 :=
by
  -- Our goal is to show that the sum of a + b + c equals 27
  let a := 5
  let b := 6
  let c := 16
  have h1 : a + b + c = 27 := sorry
  exact h1

end NUMINAMATH_GPT_ratio_of_square_sides_sum_l177_17702


namespace NUMINAMATH_GPT_find_compound_interest_principal_l177_17784

noncomputable def SI (P R T: ℝ) := (P * R * T) / 100
noncomputable def CI (P R T: ℝ) := P * (1 + R / 100)^T - P

theorem find_compound_interest_principal :
  let SI_amount := 3500.000000000004
  let SI_years := 2
  let SI_rate := 6
  let CI_years := 2
  let CI_rate := 10
  let SI_value := SI SI_amount SI_rate SI_years
  let P := 4000
  (SI_value = (CI P CI_rate CI_years) / 2) →
  P = 4000 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_compound_interest_principal_l177_17784


namespace NUMINAMATH_GPT_gear_rotations_l177_17785

-- Definitions from the conditions
def gearA_teeth : ℕ := 12
def gearB_teeth : ℕ := 54

-- The main problem: prove that gear A needs 9 rotations and gear B needs 2 rotations
theorem gear_rotations :
  ∃ x y : ℕ, 12 * x = 54 * y ∧ x = 9 ∧ y = 2 := by
  sorry

end NUMINAMATH_GPT_gear_rotations_l177_17785


namespace NUMINAMATH_GPT_one_fourth_in_one_eighth_l177_17749

theorem one_fourth_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_one_fourth_in_one_eighth_l177_17749


namespace NUMINAMATH_GPT_difference_q_r_l177_17764

-- Conditions
variables (p q r : ℕ) (x : ℕ)
variables (h_ratio : 3 * x = p) (h_ratio2 : 7 * x = q) (h_ratio3 : 12 * x = r)
variables (h_diff_pq : q - p = 3200)

-- Proof problem to solve
theorem difference_q_r : q - p = 3200 → 12 * x - 7 * x = 4000 :=
by 
  intro h_diff_pq
  rw [h_ratio, h_ratio2, h_ratio3] at *
  sorry

end NUMINAMATH_GPT_difference_q_r_l177_17764


namespace NUMINAMATH_GPT_mark_total_spending_l177_17767

variable (p_tomato_cost : ℕ) (p_apple_cost : ℕ) 
variable (pounds_tomato : ℕ) (pounds_apple : ℕ)

def total_cost (p_tomato_cost : ℕ) (pounds_tomato : ℕ) (p_apple_cost : ℕ) (pounds_apple : ℕ) : ℕ :=
  (p_tomato_cost * pounds_tomato) + (p_apple_cost * pounds_apple)

theorem mark_total_spending :
  total_cost 5 2 6 5 = 40 :=
by
  sorry

end NUMINAMATH_GPT_mark_total_spending_l177_17767


namespace NUMINAMATH_GPT_find_certain_number_l177_17780

def certain_number (x : ℤ) : Prop := x - 9 = 5

theorem find_certain_number (x : ℤ) (h : certain_number x) : x = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l177_17780


namespace NUMINAMATH_GPT_trees_probability_l177_17782

theorem trees_probability (num_maple num_oak num_birch total_slots total_trees : ℕ) 
                         (maple_count oak_count birch_count : Prop)
                         (prob_correct : Prop) :
  num_maple = 4 →
  num_oak = 5 →
  num_birch = 6 →
  total_trees = 15 →
  total_slots = 10 →
  maple_count → oak_count → birch_count →
  prob_correct →
  (m + n = 57) :=
by
  intros
  sorry

end NUMINAMATH_GPT_trees_probability_l177_17782


namespace NUMINAMATH_GPT_circle_area_l177_17756

theorem circle_area (r : ℝ) (h : 3 * (1 / (2 * π * r)) = r) : 
  π * r^2 = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_area_l177_17756


namespace NUMINAMATH_GPT_find_f_neg2003_l177_17725

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_neg2003 (f_defined : ∀ x : ℝ, ∃ y : ℝ, f y = x → f y ≠ 0)
  (cond1 : ∀ ⦃x y w : ℝ⦄, x > y → (f x + x ≥ w → w ≥ f y + y → ∃ z, y ≤ z ∧ z ≤ x ∧ f z = w - z))
  (cond2 : ∃ u : ℝ, f u = 0 ∧ ∀ v : ℝ, f v = 0 → u ≤ v)
  (cond3 : f 0 = 1)
  (cond4 : f (-2003) ≤ 2004)
  (cond5 : ∀ x y : ℝ, f x * f y = f (x * f y + y * f x + x * y)) :
  f (-2003) = 2004 :=
sorry

end NUMINAMATH_GPT_find_f_neg2003_l177_17725


namespace NUMINAMATH_GPT_cost_price_computer_table_l177_17779

theorem cost_price_computer_table (S : ℝ) (C : ℝ) (h1 : S = C * 1.15) (h2 : S = 5750) : C = 5000 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_computer_table_l177_17779


namespace NUMINAMATH_GPT_problem_CorrectOption_l177_17713

def setA : Set ℝ := {y | ∃ x : ℝ, y = |x| - 1}
def setB : Set ℝ := {x | x ≥ 2}

theorem problem_CorrectOption : setA ∩ setB = setB := 
  sorry

end NUMINAMATH_GPT_problem_CorrectOption_l177_17713


namespace NUMINAMATH_GPT_infinitely_many_arithmetic_sequences_l177_17707

theorem infinitely_many_arithmetic_sequences (x : ℕ) (hx : 0 < x) :
  ∃ y z : ℕ, y = 5 * x + 2 ∧ z = 7 * x + 3 ∧ x * (x + 1) < y * (y + 1) ∧ y * (y + 1) < z * (z + 1) ∧
  y * (y + 1) - x * (x + 1) = z * (z + 1) - y * (y + 1) :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_arithmetic_sequences_l177_17707


namespace NUMINAMATH_GPT_segment_order_l177_17757

def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

def order_segments (angles_ABC angles_XYZ angles_ZWX : ℝ → ℝ → ℝ) : Prop :=
  let A := angles_ABC 55 60
  let B := angles_XYZ 95 70
  ∀ (XY YZ ZX WX WZ: ℝ), 
    YZ < ZX ∧ ZX < XY ∧ ZX < WZ ∧ WZ < WX

theorem segment_order:
  ∀ (A B C X Y Z W : Type)
  (XYZ_ang ZWX_ang : ℝ), 
  angle_sum_triangle 55 60 65 →
  angle_sum_triangle 95 70 15 →
  order_segments (angles_ABC) (angles_XYZ) (angles_ZWX)
:= sorry

end NUMINAMATH_GPT_segment_order_l177_17757


namespace NUMINAMATH_GPT_range_of_k_for_domain_real_l177_17727

theorem range_of_k_for_domain_real (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 6 * k * x + (k + 8) ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
sorry

end NUMINAMATH_GPT_range_of_k_for_domain_real_l177_17727


namespace NUMINAMATH_GPT_residue_system_mod_3n_l177_17798

theorem residue_system_mod_3n (n : ℕ) (h_odd : n % 2 = 1) :
  ∃ (a b : ℕ → ℕ) (k : ℕ), 
  (∀ i, a i = 3 * i - 2) ∧ 
  (∀ i, b i = 3 * i - 3) ∧
  (∀ i (k : ℕ), 0 < k ∧ k < n → 
    (a i + a (i + 1)) % (3 * n) ≠ (a i + b i) % (3 * n) ∧ 
    (a i + b i) % (3 * n) ≠ (b i + b (i + k)) % (3 * n) ∧ 
    (a i + a (i + 1)) % (3 * n) ≠ (b i + b (i + k)) % (3 * n)) :=
sorry

end NUMINAMATH_GPT_residue_system_mod_3n_l177_17798


namespace NUMINAMATH_GPT_sum_of_first_5n_l177_17747

theorem sum_of_first_5n (n : ℕ) : 
  (n * (n + 1) / 2) + 210 = ((4 * n) * (4 * n + 1) / 2) → 
  (5 * n) * (5 * n + 1) / 2 = 465 :=
by sorry

end NUMINAMATH_GPT_sum_of_first_5n_l177_17747


namespace NUMINAMATH_GPT_andy_solves_16_problems_l177_17719

theorem andy_solves_16_problems :
  ∃ N : ℕ, 
    N = (125 - 78)/3 + 1 ∧
    (78 + (N - 1) * 3 <= 125) ∧
    N = 16 := 
by 
  sorry

end NUMINAMATH_GPT_andy_solves_16_problems_l177_17719


namespace NUMINAMATH_GPT_plane_passing_through_A_perpendicular_to_BC_l177_17794

-- Define the points A, B, and C
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def A : Point3D := { x := -3, y := 7, z := 2 }
def B : Point3D := { x := 3, y := 5, z := 1 }
def C : Point3D := { x := 4, y := 5, z := 3 }

-- Define the vector BC as the difference between points C and B
def vectorBC (B C : Point3D) : Point3D :=
{ x := C.x - B.x,
  y := C.y - B.y,
  z := C.z - B.z }

-- Define the equation of the plane passing through point A and 
-- perpendicular to vector BC
def plane_eq (A : Point3D) (n : Point3D) (x y z : ℝ) : Prop :=
n.x * (x - A.x) + n.y * (y - A.y) + n.z * (z - A.z) = 0

-- Define the proof problem
theorem plane_passing_through_A_perpendicular_to_BC :
  ∀ (x y z : ℝ), plane_eq A (vectorBC B C) x y z ↔ x + 2 * z - 1 = 0 :=
by
  -- the proof part
  sorry

end NUMINAMATH_GPT_plane_passing_through_A_perpendicular_to_BC_l177_17794


namespace NUMINAMATH_GPT_garden_length_l177_17775

theorem garden_length :
  ∀ (w : ℝ) (l : ℝ),
  (l = 2 * w) →
  (2 * l + 2 * w = 150) →
  l = 50 :=
by
  intros w l h1 h2
  sorry

end NUMINAMATH_GPT_garden_length_l177_17775


namespace NUMINAMATH_GPT_root_properties_of_cubic_l177_17715

theorem root_properties_of_cubic (z1 z2 : ℂ) (h1 : z1^2 + z1 + 1 = 0) (h2 : z2^2 + z2 + 1 = 0) :
  z1 * z2 = 1 ∧ z1^3 = 1 ∧ z2^3 = 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_root_properties_of_cubic_l177_17715


namespace NUMINAMATH_GPT_sum_of_digits_is_base_6_l177_17704

def is_valid_digit (x : ℕ) : Prop := x > 0 ∧ x < 6 
def distinct_3 (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a  

theorem sum_of_digits_is_base_6 :
  ∃ (S H E : ℕ), is_valid_digit S ∧ is_valid_digit H ∧ is_valid_digit E
  ∧ distinct_3 S H E 
  ∧ (E + E) % 6 = S 
  ∧ (S + H) % 6 = E 
  ∧ (S + H + E) % 6 = 11 % 6 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_digits_is_base_6_l177_17704


namespace NUMINAMATH_GPT_sum_of_digits_l177_17769

theorem sum_of_digits :
  ∃ (E M V Y : ℕ), 
    (E ≠ M ∧ E ≠ V ∧ E ≠ Y ∧ M ≠ V ∧ M ≠ Y ∧ V ≠ Y) ∧
    (10 * Y + E) * (10 * M + E) = 111 * V ∧ 
    1 ≤ V ∧ V ≤ 9 ∧ 
    E + M + V + Y = 21 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_digits_l177_17769


namespace NUMINAMATH_GPT_graph_avoid_third_quadrant_l177_17710

theorem graph_avoid_third_quadrant (k : ℝ) : 
  (∀ x y : ℝ, y = (2 * k - 1) * x + k → ¬ (x < 0 ∧ y < 0)) ↔ 0 ≤ k ∧ k < (1 / 2) :=
by sorry

end NUMINAMATH_GPT_graph_avoid_third_quadrant_l177_17710


namespace NUMINAMATH_GPT_min_value_frac_l177_17774

theorem min_value_frac (x y a b c d : ℝ) (hx : 0 < x) (hy : 0 < y)
  (harith : x + y = a + b) (hgeo : x * y = c * d) : (a + b) ^ 2 / (c * d) ≥ 4 := 
by sorry

end NUMINAMATH_GPT_min_value_frac_l177_17774


namespace NUMINAMATH_GPT_minimum_rooms_to_accommodate_fans_l177_17754

/-
Each hotel room can accommodate no more than 3 people. The hotel manager knows 
that a group of 100 football fans, who support three different teams, will soon 
arrive. A room can only house either men or women; and fans of different teams 
cannot be housed together. Prove that at least 37 rooms are needed to accommodate 
all the fans.
-/

noncomputable def minimum_rooms_needed (total_fans : ℕ) (fans_per_room : ℕ) : ℕ :=
  if h : fans_per_room > 0 then (total_fans + fans_per_room - 1) / fans_per_room else 0

theorem minimum_rooms_to_accommodate_fans :
  ∀ (total_fans : ℕ) (fans_per_room : ℕ)
    (num_teams : ℕ) (num_genders : ℕ),
  total_fans = 100 →
  fans_per_room = 3 →
  num_teams = 3 →
  num_genders = 2 →
  (minimum_rooms_needed total_fans fans_per_room) ≥ 37 :=
by
  intros total_fans fans_per_room num_teams num_genders h_total h_per_room h_teams h_genders
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_minimum_rooms_to_accommodate_fans_l177_17754


namespace NUMINAMATH_GPT_trig_identity_evaluation_l177_17765

theorem trig_identity_evaluation :
  let θ1 := 70 * Real.pi / 180 -- angle 70 degrees in radians
  let θ2 := 10 * Real.pi / 180 -- angle 10 degrees in radians
  let θ3 := 20 * Real.pi / 180 -- angle 20 degrees in radians
  (Real.tan θ1 * Real.cos θ2 * (Real.sqrt 3 * Real.tan θ3 - 1) = -1) := 
by 
  sorry

end NUMINAMATH_GPT_trig_identity_evaluation_l177_17765


namespace NUMINAMATH_GPT_minimum_value_of_m_plus_n_l177_17743

-- Define the conditions and goals as a Lean 4 statement with a proof goal.
theorem minimum_value_of_m_plus_n (m n : ℝ) (h : m * n > 0) (hA : m + n = 3 * m * n) : m + n = 4 / 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_m_plus_n_l177_17743


namespace NUMINAMATH_GPT_ways_to_distribute_items_l177_17731

/-- The number of ways to distribute 5 different items into 4 identical bags, with some bags possibly empty, is 36. -/
theorem ways_to_distribute_items : ∃ (n : ℕ), n = 36 := by
  sorry

end NUMINAMATH_GPT_ways_to_distribute_items_l177_17731


namespace NUMINAMATH_GPT_number_of_white_balls_l177_17766

theorem number_of_white_balls (x : ℕ) (h : (x : ℚ) / (x + 12) = 2 / 3) : x = 24 :=
sorry

end NUMINAMATH_GPT_number_of_white_balls_l177_17766


namespace NUMINAMATH_GPT_quadratic_roots_l177_17795

theorem quadratic_roots : ∀ x : ℝ, x * (x - 2) = 2 - x ↔ (x = 2 ∨ x = -1) := by
  intros
  sorry

end NUMINAMATH_GPT_quadratic_roots_l177_17795


namespace NUMINAMATH_GPT_complex_number_quadrant_l177_17720

def inSecondQuadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_quadrant : inSecondQuadrant (i / (1 - i)) :=
by
  sorry

end NUMINAMATH_GPT_complex_number_quadrant_l177_17720


namespace NUMINAMATH_GPT_inequality_proof_l177_17721

theorem inequality_proof (a b : ℝ) (h₁ : a ≥ b) (h₂ : b > 0) : 
  2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l177_17721


namespace NUMINAMATH_GPT_num_points_P_on_ellipse_l177_17772

noncomputable def ellipse : Set (ℝ × ℝ) := {p | (p.1)^2 / 16 + (p.2)^2 / 9 = 1}
noncomputable def line : Set (ℝ × ℝ) := {p | p.1 / 4 + p.2 / 3 = 1}
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  (1 / 2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem num_points_P_on_ellipse (A B : ℝ × ℝ) 
  (hA_on_line : A ∈ line) (hA_on_ellipse : A ∈ ellipse) 
  (hB_on_line : B ∈ line) (hB_on_ellipse : B ∈ ellipse)
  : ∃ P1 P2 : ℝ × ℝ, P1 ∈ ellipse ∧ P2 ∈ ellipse ∧ 
    area_triangle A B P1 = 3 ∧ area_triangle A B P2 = 3 ∧ 
    P1 ≠ P2 ∧ 
    (∀ P : ℝ × ℝ, P ∈ ellipse ∧ area_triangle A B P = 3 → P = P1 ∨ P = P2) := 
sorry

end NUMINAMATH_GPT_num_points_P_on_ellipse_l177_17772


namespace NUMINAMATH_GPT_find_c_of_perpendicular_lines_l177_17701

theorem find_c_of_perpendicular_lines (c : ℤ) :
  (∀ x y : ℤ, y = -3 * x + 4 → ∃ y' : ℤ, y' = (c * x + 18) / 9) →
  c = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_c_of_perpendicular_lines_l177_17701


namespace NUMINAMATH_GPT_not_all_polynomials_sum_of_cubes_l177_17770

theorem not_all_polynomials_sum_of_cubes :
  ¬ ∀ P : Polynomial ℤ, ∃ Q : Polynomial ℤ, P = Q^3 + Q^3 + Q^3 :=
by
  sorry

end NUMINAMATH_GPT_not_all_polynomials_sum_of_cubes_l177_17770


namespace NUMINAMATH_GPT_neg_neg_eq_pos_l177_17781

theorem neg_neg_eq_pos : -(-2023) = 2023 :=
by
  sorry

end NUMINAMATH_GPT_neg_neg_eq_pos_l177_17781


namespace NUMINAMATH_GPT_order_of_a_b_c_l177_17773

noncomputable def a : ℝ := Real.log 2 / Real.log 3 -- a = log_3 2
noncomputable def b : ℝ := Real.log 2 -- b = ln 2
noncomputable def c : ℝ := Real.sqrt 5 -- c = 5^(1/2)

theorem order_of_a_b_c : a < b ∧ b < c := by
  sorry

end NUMINAMATH_GPT_order_of_a_b_c_l177_17773


namespace NUMINAMATH_GPT_find_f_value_l177_17744

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x + 1

theorem find_f_value : f 2019 + f (-2019) = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_value_l177_17744


namespace NUMINAMATH_GPT_monotonicity_of_f_l177_17763

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + 1

theorem monotonicity_of_f (a x : ℝ) :
  (a > 0 → ((∀ x, (x < -2 * a / 3 → f a x' > f a x) ∧ (x > 0 → f a x' > f a x)) ∧ (∀ x, (-2 * a / 3 < x ∧ x < 0 → f a x' < f a x)))) ∧
  (a = 0 → ∀ x, f a x' > f a x) ∧
  (a < 0 → ((∀ x, (x < 0 → f a x' > f a x) ∧ (x > -2 * a / 3 → f a x' > f a x)) ∧ (∀ x, (0 < x ∧ x < -2 * a / 3 → f a x' < f a x)))) :=
sorry

end NUMINAMATH_GPT_monotonicity_of_f_l177_17763


namespace NUMINAMATH_GPT_not_true_n_gt_24_l177_17755

theorem not_true_n_gt_24 (n : ℕ) (h : 1/3 + 1/4 + 1/6 + 1/n = 1) : n ≤ 24 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_not_true_n_gt_24_l177_17755


namespace NUMINAMATH_GPT_sequence_general_term_l177_17736

/-- Given the sequence {a_n} defined by a_n = 2^n * a_{n-1} for n > 1 and a_1 = 1,
    prove that the general term a_n = 2^((n^2 + n - 2) / 2) -/
theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n > 1, a n = 2^n * a (n-1)) :
  ∀ n, a n = 2^((n^2 + n - 2) / 2) :=
sorry

end NUMINAMATH_GPT_sequence_general_term_l177_17736


namespace NUMINAMATH_GPT_contradiction_method_assumption_l177_17722

theorem contradiction_method_assumption (a b c : ℝ) :
  (¬(a > 0 ∨ b > 0 ∨ c > 0) → false) :=
sorry

end NUMINAMATH_GPT_contradiction_method_assumption_l177_17722


namespace NUMINAMATH_GPT_opposite_of_neg_two_thirds_l177_17742

theorem opposite_of_neg_two_thirds : -(- (2 / 3)) = (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_two_thirds_l177_17742


namespace NUMINAMATH_GPT_average_weight_of_abc_l177_17768

theorem average_weight_of_abc (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 43) (h3 : B = 40) : 
  (A + B + C) / 3 = 42 := 
sorry

end NUMINAMATH_GPT_average_weight_of_abc_l177_17768


namespace NUMINAMATH_GPT_raghu_investment_l177_17753

theorem raghu_investment (R T V : ℝ) (h1 : T = 0.9 * R) (h2 : V = 1.1 * T) (h3 : R + T + V = 5780) : R = 2000 :=
by
  sorry

end NUMINAMATH_GPT_raghu_investment_l177_17753


namespace NUMINAMATH_GPT_coin_flip_probability_l177_17709

theorem coin_flip_probability (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)
  (h_win : ∑' n, (1 - p) ^ n * p ^ (n + 1) = 1 / 2) :
  p = (3 - Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_coin_flip_probability_l177_17709


namespace NUMINAMATH_GPT_total_balloons_l177_17732

-- Define the number of balloons each person has
def joan_balloons : ℕ := 40
def melanie_balloons : ℕ := 41

-- State the theorem about the total number of balloons
theorem total_balloons : joan_balloons + melanie_balloons = 81 :=
by
  sorry

end NUMINAMATH_GPT_total_balloons_l177_17732


namespace NUMINAMATH_GPT_kelly_held_longest_l177_17733

variable (K : ℕ)

-- Conditions
def Brittany_held (K : ℕ) : ℕ := K - 20
def Buffy_held : ℕ := 120

-- Theorem to prove
theorem kelly_held_longest (h : K > Buffy_held) : K > 120 :=
by sorry

end NUMINAMATH_GPT_kelly_held_longest_l177_17733


namespace NUMINAMATH_GPT_solve_x_l177_17718

theorem solve_x (x : ℚ) : (∀ z : ℚ, 10 * x * z - 15 * z + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_solve_x_l177_17718


namespace NUMINAMATH_GPT_least_b_not_in_range_l177_17712

theorem least_b_not_in_range : ∃ b : ℤ, -10 = b ∧ ∀ x : ℝ, x^2 + b * x + 20 ≠ -10 :=
sorry

end NUMINAMATH_GPT_least_b_not_in_range_l177_17712


namespace NUMINAMATH_GPT_half_angle_in_first_quadrant_l177_17734

theorem half_angle_in_first_quadrant (α : ℝ) (h : 0 < α ∧ α < π / 2) : 0 < α / 2 ∧ α / 2 < π / 4 :=
by
  sorry

end NUMINAMATH_GPT_half_angle_in_first_quadrant_l177_17734


namespace NUMINAMATH_GPT_area_of_region_B_l177_17711

-- Given conditions
def region_B (z : ℂ) : Prop :=
  (0 ≤ (z.re / 50) ∧ (z.re / 50) ≤ 1 ∧ 0 ≤ (z.im / 50) ∧ (z.im / 50) ≤ 1)
  ∧
  (0 ≤ (50 * z.re / (z.re^2 + z.im^2)) ∧ (50 * z.re / (z.re^2 + z.im^2)) ≤ 1 ∧ 
  0 ≤ (50 * z.im / (z.re^2 + z.im^2)) ∧ (50 * z.im / (z.re^2 + z.im^2)) ≤ 1)

-- Theorem to be proved
theorem area_of_region_B : 
  (∫ z in {z : ℂ | region_B z}, 1) = 1875 - 312.5 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_region_B_l177_17711


namespace NUMINAMATH_GPT_tank_empty_time_l177_17797

theorem tank_empty_time 
  (time_to_empty_leak : ℝ) 
  (inlet_rate_per_minute : ℝ) 
  (tank_volume : ℝ) 
  (net_time_to_empty : ℝ) : 
  time_to_empty_leak = 7 → 
  inlet_rate_per_minute = 6 → 
  tank_volume = 6048.000000000001 → 
  net_time_to_empty = 12 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_tank_empty_time_l177_17797


namespace NUMINAMATH_GPT_semicircle_perimeter_l177_17752

theorem semicircle_perimeter (r : ℝ) (π : ℝ) (h : 0 < π) (r_eq : r = 14):
  (14 * π + 28) = 14 * π + 28 :=
by
  sorry

end NUMINAMATH_GPT_semicircle_perimeter_l177_17752


namespace NUMINAMATH_GPT_square_root_of_4_is_pm2_l177_17793

theorem square_root_of_4_is_pm2 : ∃ (x : ℤ), x * x = 4 ∧ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_GPT_square_root_of_4_is_pm2_l177_17793


namespace NUMINAMATH_GPT_fresh_grapes_water_content_l177_17760

theorem fresh_grapes_water_content:
  ∀ (P : ℝ), 
  (∀ (x y : ℝ), P = x) → 
  (∃ (fresh_grapes dry_grapes : ℝ), fresh_grapes = 25 ∧ dry_grapes = 3.125 ∧ 
  (100 - P) / 100 * fresh_grapes = 0.8 * dry_grapes ) → 
  P = 90 :=
by 
  sorry

end NUMINAMATH_GPT_fresh_grapes_water_content_l177_17760


namespace NUMINAMATH_GPT_unique_solution_iff_d_ne_4_l177_17746

theorem unique_solution_iff_d_ne_4 (c d : ℝ) : 
  (∃! (x : ℝ), 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 := 
by 
  sorry

end NUMINAMATH_GPT_unique_solution_iff_d_ne_4_l177_17746


namespace NUMINAMATH_GPT_exponential_order_l177_17706

theorem exponential_order (x y : ℝ) (a : ℝ) (hx : x > y) (hy : y > 1) (ha1 : 0 < a) (ha2 : a < 1) : a^x < a^y :=
sorry

end NUMINAMATH_GPT_exponential_order_l177_17706


namespace NUMINAMATH_GPT_trig_identity_on_line_l177_17783

theorem trig_identity_on_line (α : ℝ) (h : Real.tan α = 2) :
  Real.sin α ^ 2 - Real.cos α ^ 2 + Real.sin α * Real.cos α = 1 :=
sorry

end NUMINAMATH_GPT_trig_identity_on_line_l177_17783


namespace NUMINAMATH_GPT_Jenny_original_number_l177_17703

theorem Jenny_original_number (y : ℝ) (h : 10 * (y / 2 - 6) = 70) : y = 26 :=
by
  sorry

end NUMINAMATH_GPT_Jenny_original_number_l177_17703


namespace NUMINAMATH_GPT_no_such_function_exists_l177_17724

theorem no_such_function_exists 
  (f : ℝ → ℝ) 
  (h_f_pos : ∀ x, 0 < x → 0 < f x) 
  (h_eq : ∀ x y, 0 < x → 0 < y → f (x + y) = f x + f y + (1 / 2012)) : 
  false :=
sorry

end NUMINAMATH_GPT_no_such_function_exists_l177_17724


namespace NUMINAMATH_GPT_evaluate_complex_fraction_l177_17723

def complex_fraction : Prop :=
  let expr : ℚ := 2 + (3 / (4 + (5 / 6)))
  expr = 76 / 29

theorem evaluate_complex_fraction : complex_fraction :=
by
  let expr : ℚ := 2 + (3 / (4 + (5 / 6)))
  show expr = 76 / 29
  sorry

end NUMINAMATH_GPT_evaluate_complex_fraction_l177_17723


namespace NUMINAMATH_GPT_calc_power_expression_l177_17778

theorem calc_power_expression (a b c : ℕ) (h₁ : b = 2) (h₂ : c = 3) :
  3^15 * (3^b)^5 / (3^c)^6 = 2187 := 
sorry

end NUMINAMATH_GPT_calc_power_expression_l177_17778


namespace NUMINAMATH_GPT_avg_weight_increase_l177_17790

theorem avg_weight_increase
  (A : ℝ) -- Initial average weight
  (n : ℕ) -- Initial number of people
  (w_old : ℝ) -- Weight of the person being replaced
  (w_new : ℝ) -- Weight of the new person
  (h_n : n = 8) -- Initial number of people is 8
  (h_w_old : w_old = 85) -- Weight of the replaced person is 85
  (h_w_new : w_new = 105) -- Weight of the new person is 105
  : ((8 * A + (w_new - w_old)) / 8) - A = 2.5 := 
sorry

end NUMINAMATH_GPT_avg_weight_increase_l177_17790


namespace NUMINAMATH_GPT_find_number_l177_17714

theorem find_number (n : ℕ) (h : 582964 * n = 58293485180) : n = 100000 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l177_17714


namespace NUMINAMATH_GPT_sharing_watermelons_l177_17758

theorem sharing_watermelons (h : 8 = people_per_watermelon) : people_for_4_watermelons = 32 :=
by
  let people_per_watermelon := 8
  let watermelons := 4
  let people_for_4_watermelons := people_per_watermelon * watermelons
  sorry

end NUMINAMATH_GPT_sharing_watermelons_l177_17758


namespace NUMINAMATH_GPT_ticket_difference_l177_17771

/-- 
  Define the initial number of tickets Billy had,
  the number of tickets after buying a yoyo,
  and state the proof that the difference is 16.
--/

theorem ticket_difference (initial_tickets : ℕ) (remaining_tickets : ℕ) 
  (h₁ : initial_tickets = 48) (h₂ : remaining_tickets = 32) : 
  initial_tickets - remaining_tickets = 16 :=
by
  /- This is where the prover would go, 
     no need to implement it as we know the expected result -/
  sorry

end NUMINAMATH_GPT_ticket_difference_l177_17771


namespace NUMINAMATH_GPT_average_age_of_three_l177_17787

theorem average_age_of_three (Tonya_age John_age Mary_age : ℕ)
  (h1 : John_age = 2 * Mary_age)
  (h2 : Tonya_age = 2 * John_age)
  (h3 : Tonya_age = 60) :
  (Tonya_age + John_age + Mary_age) / 3 = 35 := by
  sorry

end NUMINAMATH_GPT_average_age_of_three_l177_17787


namespace NUMINAMATH_GPT_sum_gcd_lcm_eq_180195_l177_17705

def gcd_60_45045 := Nat.gcd 60 45045
def lcm_60_45045 := Nat.lcm 60 45045

theorem sum_gcd_lcm_eq_180195 : gcd_60_45045 + lcm_60_45045 = 180195 := by
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_eq_180195_l177_17705


namespace NUMINAMATH_GPT_number_of_ordered_triplets_l177_17786

theorem number_of_ordered_triplets :
  ∃ count : ℕ, (∀ (a b c : ℕ), lcm a b = 1000 ∧ lcm b c = 2000 ∧ lcm c a = 2000 →
  count = 70) :=
sorry

end NUMINAMATH_GPT_number_of_ordered_triplets_l177_17786


namespace NUMINAMATH_GPT_problem_equiv_proof_l177_17748

variable (a b : ℝ)
variable (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1)

theorem problem_equiv_proof :
  (2 ^ a + 2 ^ b ≥ 2 * Real.sqrt 2) ∧
  (Real.log a / Real.log 2 + Real.log b / Real.log 2 ≤ -2) ∧
  (a ^ 2 + b ^ 2 ≥ 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_equiv_proof_l177_17748


namespace NUMINAMATH_GPT_camp_cedar_counselors_l177_17729

theorem camp_cedar_counselors (boys : ℕ) (girls : ℕ) (total_children : ℕ) (counselors : ℕ)
  (h1 : boys = 40)
  (h2 : girls = 3 * boys)
  (h3 : total_children = boys + girls)
  (h4 : counselors = total_children / 8) : 
  counselors = 20 :=
by sorry

end NUMINAMATH_GPT_camp_cedar_counselors_l177_17729


namespace NUMINAMATH_GPT_solve_equation_error_step_l177_17788

theorem solve_equation_error_step 
  (equation : ∀ x : ℝ, (x - 1) / 2 + 1 = (2 * x + 1) / 3) :
  ∃ (step : ℕ), step = 1 ∧
  let s1 := ((x - 1) / 2 + 1) * 6;
  ∀ (x : ℝ), s1 ≠ (((2 * x + 1) / 3) * 6) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_error_step_l177_17788


namespace NUMINAMATH_GPT_determine_c_square_of_binomial_l177_17762

theorem determine_c_square_of_binomial (c : ℝ) : (∀ x : ℝ, 16 * x^2 + 40 * x + c = (4 * x + 5)^2) → c = 25 :=
by
  intro h
  have key := h 0
  -- By substitution, we skip the expansion steps and immediately conclude the value of c
  sorry

end NUMINAMATH_GPT_determine_c_square_of_binomial_l177_17762


namespace NUMINAMATH_GPT_sonny_cookie_problem_l177_17751

theorem sonny_cookie_problem 
  (total_boxes : ℕ) (boxes_sister : ℕ) (boxes_cousin : ℕ) (boxes_left : ℕ) (boxes_brother : ℕ) : 
  total_boxes = 45 → boxes_sister = 9 → boxes_cousin = 7 → boxes_left = 17 → 
  boxes_brother = total_boxes - boxes_left - boxes_sister - boxes_cousin → 
  boxes_brother = 12 :=
by
  intros h_total h_sister h_cousin h_left h_brother
  rw [h_total, h_sister, h_cousin, h_left] at h_brother
  exact h_brother

end NUMINAMATH_GPT_sonny_cookie_problem_l177_17751


namespace NUMINAMATH_GPT_total_rounds_played_l177_17745

/-- William and Harry played some rounds of tic-tac-toe.
    William won 5 more rounds than Harry.
    William won 10 rounds.
    Prove that the total number of rounds they played is 15. -/
theorem total_rounds_played (williams_wins : ℕ) (harrys_wins : ℕ)
  (h1 : williams_wins = 10)
  (h2 : williams_wins = harrys_wins + 5) :
  williams_wins + harrys_wins = 15 := 
by
  sorry

end NUMINAMATH_GPT_total_rounds_played_l177_17745


namespace NUMINAMATH_GPT_chloe_and_friends_points_l177_17726

-- Define the conditions as Lean definitions and then state the theorem to be proven.

def total_pounds_recycled : ℕ := 28 + 2

def pounds_per_point : ℕ := 6

def points_earned (total_pounds : ℕ) (pounds_per_point : ℕ) : ℕ :=
  total_pounds / pounds_per_point

theorem chloe_and_friends_points :
  points_earned total_pounds_recycled pounds_per_point = 5 :=
by
  sorry

end NUMINAMATH_GPT_chloe_and_friends_points_l177_17726


namespace NUMINAMATH_GPT_total_silver_dollars_l177_17792

-- Definitions based on conditions
def chiu_silver_dollars : ℕ := 56
def phung_silver_dollars : ℕ := chiu_silver_dollars + 16
def ha_silver_dollars : ℕ := phung_silver_dollars + 5

-- Theorem statement
theorem total_silver_dollars : chiu_silver_dollars + phung_silver_dollars + ha_silver_dollars = 205 :=
by
  -- We use "sorry" to fill in the proof part as instructed
  sorry

end NUMINAMATH_GPT_total_silver_dollars_l177_17792


namespace NUMINAMATH_GPT_fly_distance_from_ceiling_l177_17730

/-- 
Assume a room where two walls and the ceiling meet at right angles at point P.
Let point P be the origin (0, 0, 0). 
Let the fly's position be (2, 7, z), where z is the distance from the ceiling.
Given the fly is 2 meters from one wall, 7 meters from the other wall, 
and 10 meters from point P, prove that the fly is at a distance sqrt(47) from the ceiling.
-/
theorem fly_distance_from_ceiling : 
  ∀ (z : ℝ), 
  (2^2 + 7^2 + z^2 = 10^2) → 
  z = Real.sqrt 47 :=
by 
  intro z h
  sorry

end NUMINAMATH_GPT_fly_distance_from_ceiling_l177_17730


namespace NUMINAMATH_GPT_find_b_plus_m_l177_17728

section MatrixPower

open Matrix

-- Define our matrices
def A (b m : ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![1, 3, b], 
    ![0, 1, 5], 
    ![0, 0, 1]]

def B : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![1, 27, 3008], 
    ![0, 1, 45], 
    ![0, 0, 1]]

-- The problem statement
noncomputable def power_eq_matrix (b m : ℕ) : Prop :=
  (A b m) ^ m = B

-- The final goal
theorem find_b_plus_m (b m : ℕ) (h : power_eq_matrix b m) : b + m = 283 := sorry

end MatrixPower

end NUMINAMATH_GPT_find_b_plus_m_l177_17728


namespace NUMINAMATH_GPT_derek_alice_pair_l177_17789

-- Variables and expressions involved
variable (x b c : ℝ)

-- Definitions of the conditions
def derek_eq := |x + 3| = 5 
def alice_eq := ∀ a, (a - 2) * (a + 8) = a^2 + b * a + c

-- The theorem to prove
theorem derek_alice_pair : derek_eq x → alice_eq b c → (b, c) = (6, -16) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_derek_alice_pair_l177_17789


namespace NUMINAMATH_GPT_cos_triple_angle_l177_17700

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := by
  sorry

end NUMINAMATH_GPT_cos_triple_angle_l177_17700


namespace NUMINAMATH_GPT_sum_of_interior_angles_increases_l177_17777

theorem sum_of_interior_angles_increases (n : ℕ) (h : n ≥ 3) : (n-2) * 180 > (n-3) * 180 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_increases_l177_17777
