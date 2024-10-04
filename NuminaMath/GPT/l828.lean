import Mathlib

namespace rows_per_shelf_l828_828011

theorem rows_per_shelf (cans_per_row : ℕ) (shelves_per_closet : ℕ) (cans_per_closet : ℕ)
    (h_cans_per_row : cans_per_row = 12) (h_shelves_per_closet : shelves_per_closet = 10)
    (h_cans_per_closet : cans_per_closet = 480) :
    (cans_per_closet / cans_per_row) / shelves_per_closet = 4 :=
by
  rw [h_cans_per_row, h_shelves_per_closet, h_cans_per_closet]
  dsimp
  norm_num
  sorry

end rows_per_shelf_l828_828011


namespace calculate_rent_l828_828022

def monthly_income : ℝ := 3200
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries_eating_out : ℝ := 300
def insurance : ℝ := 200
def miscellaneous : ℝ := 200
def car_payment : ℝ := 350
def gas_maintenance : ℝ := 350

def total_expenses : ℝ := utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous + car_payment + gas_maintenance
def rent : ℝ := monthly_income - total_expenses

theorem calculate_rent : rent = 1250 := by
  -- condition proof here
  sorry

end calculate_rent_l828_828022


namespace gcd_lcm_mul_l828_828058

theorem gcd_lcm_mul (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := 
by
  sorry

end gcd_lcm_mul_l828_828058


namespace angle_at_3_15_is_7_point_5_degrees_l828_828152

-- Definitions for the positions of the hour and minute hands
def minute_hand_position (minutes: ℕ) : ℝ := (minutes / 60.0) * 360.0
def hour_hand_position (hours: ℕ) (minutes: ℕ) : ℝ := (hours * 30) + (minutes * 0.5)

-- The time 3:15
def time_3_15 := (3, 15)

-- The acute angle calculation
def acute_angle_between_hands (hour: ℕ) (minute: ℕ) : ℝ :=
  let minute_angle := minute_hand_position minute
  let hour_angle := hour_hand_position hour minute
  abs (minute_angle - hour_angle)

-- The theorem statement
theorem angle_at_3_15_is_7_point_5_degrees : 
  acute_angle_between_hands 3 15 = 7.5 := 
  sorry

end angle_at_3_15_is_7_point_5_degrees_l828_828152


namespace smallest_result_l828_828159

theorem smallest_result :
  let a := (-2)^3
  let b := (-2) + 3
  let c := (-2) * 3
  let d := (-2) - 3
  a < b ∧ a < c ∧ a < d :=
by
  -- Lean proof steps would go here
  sorry

end smallest_result_l828_828159


namespace service_station_location_l828_828081

/-- The first exit is at milepost 35. -/
def first_exit_milepost : ℕ := 35

/-- The eighth exit is at milepost 275. -/
def eighth_exit_milepost : ℕ := 275

/-- The expected milepost of the service station built halfway between the first exit and the eighth exit is 155. -/
theorem service_station_location : (first_exit_milepost + (eighth_exit_milepost - first_exit_milepost) / 2) = 155 := by
  sorry

end service_station_location_l828_828081


namespace divisor_of_635_l828_828055

theorem divisor_of_635 (p : ℕ) (h1 : Nat.Prime p) (k : ℕ) (h2 : 635 = 7 * k * p + 11) : p = 89 :=
sorry

end divisor_of_635_l828_828055


namespace trajectory_of_P_l828_828100

theorem trajectory_of_P (M P : ℝ × ℝ) (OM OP : ℝ) (x y : ℝ) :
  (M = (4, y)) →
  (P = (x, y)) →
  (OM = Real.sqrt (4^2 + y^2)) →
  (OP = Real.sqrt ((x - 4)^2 + y^2)) →
  (OM * OP = 16) →
  (x - 2)^2 + y^2 = 4 :=
by sorry

end trajectory_of_P_l828_828100


namespace domain_of_g_l828_828139

noncomputable def g : ℝ → ℝ := λ x, (x - 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_g :
  {x : ℝ | g x ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_g_l828_828139


namespace value_of_expression_at_x_eq_2_l828_828157

theorem value_of_expression_at_x_eq_2 :
  (2 * (2: ℕ)^2 - 3 * 2 + 4 = 6) := 
by sorry

end value_of_expression_at_x_eq_2_l828_828157


namespace acute_angle_at_3_15_l828_828147

/-- The hour and minute hands' angles and movements are defined as follows. -/
def hour_hand_angle (h m : Nat) : Real := (h % 12) * 30 + m * 0.5
def minute_hand_angle (m : Nat) : Real := (m % 60) * 6

/-- The condition that an acute angle is the smaller angle between hands. -/
def acute_angle (angle1 angle2 : Real) : Real := abs (angle1 - angle2)

/-- At 3:15, the acute angle between the hour and minute hands should be 7.5 degrees. -/
theorem acute_angle_at_3_15
    : acute_angle (hour_hand_angle 3 15) (minute_hand_angle 15) = 7.5 :=
by
    sorry

end acute_angle_at_3_15_l828_828147


namespace find_a_for_polynomial_identity_l828_828117

theorem find_a_for_polynomial_identity : 
  ∃ (a : ℤ), ∀ (b c : ℤ), 
  (∀ x : ℤ, (x - a) * (x - 5) + 4 = (x + b) * (x + c)) → a = 5 :=
begin
  sorry
end

end find_a_for_polynomial_identity_l828_828117


namespace distance_between_points_l828_828133

theorem distance_between_points :
  let (x1, y1) := (1, 2)
  let (x2, y2) := (6, 5)
  let d := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
  d = Real.sqrt 34 :=
by
  sorry

end distance_between_points_l828_828133


namespace frog_arrangements_l828_828114

theorem frog_arrangements :
  let total_frogs := 7
  let green_frogs := 2
  let red_frogs := 3
  let blue_frogs := 2
  let valid_sequences := 4
  let green_permutations := Nat.factorial green_frogs
  let red_permutations := Nat.factorial red_frogs
  let blue_permutations := Nat.factorial blue_frogs
  let total_permutations := valid_sequences * (green_permutations * red_permutations * blue_permutations)
  total_frogs = green_frogs + red_frogs + blue_frogs → 
  green_frogs = 2 ∧ red_frogs = 3 ∧ blue_frogs = 2 →
  valid_sequences = 4 →
  total_permutations = 96 := 
by
  -- Given conditions lead to the calculation of total permutations 
  sorry

end frog_arrangements_l828_828114


namespace annual_interest_correct_l828_828064

-- Define the conditions
def Rs_total : ℝ := 3400
def P1 : ℝ := 1300
def P2 : ℝ := Rs_total - P1
def Rate1 : ℝ := 0.03
def Rate2 : ℝ := 0.05

-- Define the interests
def Interest1 : ℝ := P1 * Rate1
def Interest2 : ℝ := P2 * Rate2

-- The total interest
def Total_Interest : ℝ := Interest1 + Interest2

-- The theorem to prove
theorem annual_interest_correct :
  Total_Interest = 144 :=
by
  sorry

end annual_interest_correct_l828_828064


namespace rabbit_clearing_10_square_yards_per_day_l828_828130

noncomputable def area_cleared_by_one_rabbit_per_day (length width : ℕ) (rabbits : ℕ) (days : ℕ) : ℕ :=
  (length * width) / (3 * 3 * rabbits * days)

theorem rabbit_clearing_10_square_yards_per_day :
  area_cleared_by_one_rabbit_per_day 200 900 100 20 = 10 :=
by sorry

end rabbit_clearing_10_square_yards_per_day_l828_828130


namespace triangle_min_perimeter_l828_828088

theorem triangle_min_perimeter (x : ℕ) (h1 : 51 + 67 > x) (h2 : 51 + x > 67) (h3 : 67 + x > 51) : 51 + 67 + x = 135 := by
  have hx : x = 17 := by
    -- derive from the inequalities
    sorry
  rw [hx]
  exact rfl

end triangle_min_perimeter_l828_828088


namespace focal_distance_of_ellipse_l828_828082

theorem focal_distance_of_ellipse : 
  ∀ (θ : ℝ), (∃ (c : ℝ), (x = 5 * Real.cos θ ∧ y = 4 * Real.sin θ) → 2 * c = 6) :=
by
  sorry

end focal_distance_of_ellipse_l828_828082


namespace integer_d_iff_exists_ef_l828_828025

noncomputable def question (a b c d e f : ℝ) (x : ℝ) : ℝ :=
  (⟦ (⟦(x + a) / b⟧ + c) / d ⟧)

noncomputable def answer (e f : ℝ) (x : ℝ) : ℝ :=
  ⟦ (x + e) / f ⟧

theorem integer_d_iff_exists_ef (a b c d : ℝ) :
  d ∈ ℤ ↔ ∃ e f : ℝ, e > 0 ∧ f > 0 ∧ ∀ x : ℝ, question a b c d e f x = answer e f x :=
sorry

end integer_d_iff_exists_ef_l828_828025


namespace sum_row_col_products_nonzero_l828_828068

theorem sum_row_col_products_nonzero (n : ℕ) 
  (hodd : n % 2 = 1)
  (grid : Fin n → Fin n → ℤ)
  (h_values : ∀ i j, grid i j = 1 ∨ grid i j = -1)
  (r : Fin n → ℤ)
  (hr : ∀ j, r j = ∏ i, grid i j)
  (c : Fin n → ℤ)
  (hc : ∀ k, c k = ∏ j, grid j k) :
  (Finset.univ.sum r + Finset.univ.sum c) ≠ 0 := 
by sorry

end sum_row_col_products_nonzero_l828_828068


namespace length_of_segment_AP_l828_828009

theorem length_of_segment_AP 
  (ABC : Type)
  [euclidean_geometry ABC]
  {A B C K M P : ABC}
  {a b : ℝ}
  (h1 : segment_length C B = a)
  (h2 : segment_length C A = b)
  (h3 : is_angle_bisector_of (∠ A C B) K)
  (h4 : K ∈ segment A B)
  (h5 : M ∈ circumcircle A B C)
  (h6 : M ∈ circumcircle A M K)
  (h7 : P ∈ segment A C)
  (h8 : P ∈ circumcircle A M K) :
  segment_length A P = real.abs (a - b) :=
begin
  sorry
end

end length_of_segment_AP_l828_828009


namespace range_of_x_l828_828084

theorem range_of_x (f : ℝ → ℝ) (h_inc : ∀ (x y : ℝ), x < y → f x < f y)
  (h_dom : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → true) :
  {x : ℝ | f (x - 1) < f (x^2 - 1)} = {x : ℝ | 1 < x ∧ x ≤ real.sqrt 2} :=
begin
  sorry
end

end range_of_x_l828_828084


namespace trajectory_is_parabola_l828_828034

-- defining the operation \otimes
def tensor_op (x1 x2 : ℝ) : ℝ :=
  (x1 + x2)^2 - (x1 - x2)^2

-- defining the point P(x, sqrt(x ⊗ 2))
def point_P (x : ℝ) : ℝ × ℝ :=
  (x, real.sqrt (tensor_op x 2))

-- stating the problem
theorem trajectory_is_parabola (x : ℝ) (hx : x ≥ 0) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ point_P x = (x, real.sqrt (x * 8)) :=
sorry

end trajectory_is_parabola_l828_828034


namespace speeds_and_time_l828_828165

theorem speeds_and_time (x s : ℕ) (t : ℝ)
  (h1 : ∀ {t : ℝ}, t = 2 → x * t > s * t + 24)
  (h2 : ∀ {t : ℝ}, t = 0.5 → x * t = 8) :
  x = 16 ∧ s = 4 ∧ t = 8 :=
by {
  sorry
}

end speeds_and_time_l828_828165


namespace exists_path_length_a_b_succ_l828_828115

variable {Player : Type}
variable (wins : Player → Player → Prop)
variable (a b : ℕ) (ha : 1 ≤ a) (hb : 1 ≤ b)

-- Condition 1: Each player wins against at least a players and loses to at least b players.
axiom win_loss_conditions (p : Player) : (∃ W, ∃ L, (∀ w ∈ W, wins p w) ∧ (∀ l ∈ L, wins l p) ∧ W.card ≥ a ∧ L.card ≥ b)

-- Condition 2: For any two players A, B, there exist some players P_1, ..., P_k ...
axiom transitive_wins (A B : Player) : ∃ (k : ℕ) (P : Fin k → Player), k ≥ 2 ∧ P 0 = A ∧ P (k-1) = B ∧ (∀ i : Fin (k-1), wins (P i) (P (i+1)))

-- Theorem: There exist a+b+1 distinct players Q_1, ..., Q_{a+b+1} such that Q_i wins against Q_{i+1}.
theorem exists_path_length_a_b_succ (a b : ℕ) (ha : 1 ≤ a) (hb : 1 ≤ b) : 
  ∃ (Q : Fin (a + b + 1) → Player), ∀ i : Fin (a + b), wins (Q i) (Q (i+1)) := 
sorry

end exists_path_length_a_b_succ_l828_828115


namespace eq_distance_AB_AC_l828_828089

noncomputable theory
open_locale classical

-- Define the conditions of the problem
def line1 (x y : ℝ) : Prop := x + y = 3
def line2 (x y : ℝ) : Prop := 2 * x - y = 0
def line3 (x y : ℝ) (t : ℝ) : Prop := 3 * x - t * y = 4

-- Define points based on line intersections
def pointA := (1, 2)  -- Intersection of line1 and line2
def pointB (t : ℝ) : (ℝ × ℝ) := (4 / (3 - t) / 3 - t), 5 / (3 - t)
def pointC (t : ℝ) : (ℝ × ℝ) := (4 / (3 - 2 * t), 8 / (3 - 2 * t))

-- Define distance formula
def distance (p1 p2: ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem eq_distance_AB_AC (t : ℝ) :
  t = -1/2 ∨ t = 9 + 3 * real.sqrt 10 ∨ t = 9 - 3 * real.sqrt 10 →
  (distance (pointA) (pointB t)) = (distance (pointA) (pointC t)) :=
by {
  sorry
}

end eq_distance_AB_AC_l828_828089


namespace simplify_fraction_l828_828161

theorem simplify_fraction (c : ℝ) : (5 - 4 * c) / 9 - 3 = (-22 - 4 * c) / 9 := 
sorry

end simplify_fraction_l828_828161


namespace sum_first_10_terms_arithmetic_seq_l828_828002

theorem sum_first_10_terms_arithmetic_seq :
  (∃ a : ℕ → ℚ, a 1 = -2 
    ∧ (∀ n : ℕ, n > 0 → 2 * a (n + 1) = 1 + 2 * a n)
    ∧ (∑ i in finset.range 10, a (i + 1)) = 5 / 2) :=
begin
  sorry -- Proof not required
end

end sum_first_10_terms_arithmetic_seq_l828_828002


namespace volume_tetrahedron_ABCD_l828_828003

noncomputable def volume_of_tetrahedron (AB CD distance angle : ℝ) : ℝ :=
  (1 / 3) * ((1 / 2) * AB * CD * Real.sin angle) * distance

theorem volume_tetrahedron_ABCD :
  volume_of_tetrahedron 1 (Real.sqrt 3) 2 (Real.pi / 3) = 1 / 2 :=
by
  unfold volume_of_tetrahedron
  sorry

end volume_tetrahedron_ABCD_l828_828003


namespace roses_in_vase_l828_828119

theorem roses_in_vase (r_initial r_added : ℕ) (h1 : r_initial = 10) (h2 : r_added = 8) : r_initial + r_added = 18 :=
by
  rw [h1, h2]
  norm_num

end roses_in_vase_l828_828119


namespace total_surface_area_1221_l828_828048

def cubes : List ℕ := [1, 8, 27, 64, 125, 216, 343, 512, 729]

def side_length (v : ℕ) : ℕ := Nat.cbrt v

def surface_area (n : ℕ) : ℕ :=
  match n with
  | 9 => 5 * (side_length 729)^2
  | 8 => 4 * (side_length 512)^2
  | 7 => 4 * (side_length 343)^2
  | 6 => 4 * (side_length 216)^2
  | 5 => 4 * (side_length 125)^2
  | 4 => 4 * (side_length 64)^2
  | 3 => 4 * (side_length 27)^2
  | 2 => 4 * (side_length 8)^2
  | 1 => 4 * (side_length 1)^2
  | _ => 0

def total_surface_area : ℕ :=
  surface_area 9 + surface_area 8 +
  surface_area 7 + surface_area 6 +
  surface_area 5 + surface_area 4 +
  surface_area 3 + surface_area 2 +
  surface_area 1

theorem total_surface_area_1221 : total_surface_area = 1221 := by
  sorry

end total_surface_area_1221_l828_828048


namespace transistors_in_2010_l828_828047

theorem transistors_in_2010 (initial_transistors: ℕ) 
    (doubling_period_years: ℕ) (start_year: ℕ) (end_year: ℕ) 
    (h_initial: initial_transistors = 500000)
    (h_period: doubling_period_years = 2) 
    (h_start: start_year = 1992) 
    (h_end: end_year = 2010) :
  let years_passed := end_year - start_year
  let number_of_doublings := years_passed / doubling_period_years
  let transistors_in_end_year := initial_transistors * 2^number_of_doublings
  transistors_in_end_year = 256000000 := by
    sorry

end transistors_in_2010_l828_828047


namespace work_done_by_force_l828_828074

def point := ℝ × ℝ × ℝ
def vector := ℝ × ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def displacement (A B : point) : vector :=
  (B.1 - A.1, B.2 - A.2, B.3 - A.3)

theorem work_done_by_force :
  let A : point := (1, 1, -2)
  let B : point := (3, 4, -2 + Real.sqrt 2)
  let F : vector := (2, 2, 2 * Real.sqrt 2)
  let S := displacement A B
  in dot_product F S = 14 :=
by
  sorry

end work_done_by_force_l828_828074


namespace unique_solution_quadratic_l828_828071

theorem unique_solution_quadratic {a : ℚ} (h : ∃ x : ℚ, 2 * a * x^2 + 15 * x + 9 = 0) : 
  a = 25 / 8 ∧ (∃ x : ℚ, 2 * (25 / 8) * x^2 + 15 * x + 9 = 0 ∧ x = -12 / 5) := 
by
  sorry

end unique_solution_quadratic_l828_828071


namespace max_value_on_interval_l828_828090

def f (x : ℝ) : ℝ := x^2 + 3 * x + 2

theorem max_value_on_interval : ∃ x ∈ (set.Icc (-5) 5), ∀ y ∈ (set.Icc (-5) 5), f y ≤ f x := 
by
  use 5
  have h5 : 5 ∈ set.Icc (-5 : ℝ) (5 : ℝ) := ⟨by linarith, by linarith⟩
  use h5
  intros y hy
  have h1: -5 ≤ y := hy.left
  have h2: y ≤ 5 := hy.right
  calc
    f y = y^2 + 3 * y + 2 : by sorry
    ... ≤ 5^2 + 3 * 5 + 2 : by sorry
  sorry

end max_value_on_interval_l828_828090


namespace minimum_edges_with_triangle_l828_828056

theorem minimum_edges_with_triangle (G : SimpleGraph (Fin 21)) (h1 : ∀ (u v : Fin 21), u ≠ v → G.adj u v → ¬ G.adj v u)
  (h2 : ∃ (m : ℕ) (hc : Odd m), ∃ (cycle : Fin m → Fin 21), (∀ i, G.adj (cycle i) (cycle (i + 1) % m))) :
  ∀ (n ≤ 101), ∃ u v x : Fin 21, G.adj u v ∧ G.adj v x ∧ G.adj x u :=
begin
  sorry
end

end minimum_edges_with_triangle_l828_828056


namespace correct_calculation_l828_828158

variable (a : ℝ)

theorem correct_calculation : (a^2)^3 = a^6 := 
by sorry

end correct_calculation_l828_828158


namespace cost_per_page_l828_828012

theorem cost_per_page
  (num_notebooks : ℕ)
  (pages_per_notebook : ℕ)
  (total_dollars_paid : ℕ)
  (h1 : num_notebooks = 2)
  (h2 : pages_per_notebook = 50)
  (h3 : total_dollars_paid = 5) :
  (total_dollars_paid * 100) / (num_notebooks * pages_per_notebook) = 5 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end cost_per_page_l828_828012


namespace clock_angle_at_315_l828_828150

theorem clock_angle_at_315 : 
  (angle_between_hour_and_minute_at (hours := 3) (minutes := 15)) = 7.5 :=
sorry

end clock_angle_at_315_l828_828150


namespace necessary_condition_for_x_squared_l828_828102

theorem necessary_condition_for_x_squared (a : ℝ) :
  (∃ x ∈ set.Icc 1 2, x^2 - a > 0) -> a ≤ 4 :=
by
  sorry

end necessary_condition_for_x_squared_l828_828102


namespace side_length_square_l828_828106

theorem side_length_square (rectangle_length rectangle_width : ℕ) (square_perimeter : ℕ) (h1 : rectangle_length = 8) (h2 : rectangle_width = 10) (h3 : square_perimeter = 2 * (rectangle_length + rectangle_width)) : 
  square_perimeter / 4 = 9 := 
by {
  -- Using given conditions to derive needed side length of square
  rw [h1, h2] at h3,
  norm_num at h3,
  rw h3,
  norm_num,
}

end side_length_square_l828_828106


namespace avg_of_first_n_multiples_of_7_l828_828131

theorem avg_of_first_n_multiples_of_7 (n : ℕ) (h : (SumOfFirstNMultiplesOf7 n) / n = 77) : n = 21 :=
by
  sorry

def SumOfFirstNMultiplesOf7 (n : ℕ) : ℕ :=
  (n * (7 + 7 * n)) / 2

end avg_of_first_n_multiples_of_7_l828_828131


namespace problem_statement_final_answer_l828_828073

theorem problem_statement {x y z : ℝ} (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (xyz_eq : x * y * z = 1) 
  (x_m_z : x + 1/z = 5) 
  (y_m_x : y + 1/x = 29) : 
  z + 1/y = 1/4 := 
sorry

theorem final_answer : 
  ∃ (m n : ℕ), Nat.coprime m n ∧ (m + n = 5) ∧ do
  h : \( m / n = 1 / 4 \) :=
sorry

end problem_statement_final_answer_l828_828073


namespace acute_angle_at_315_equals_7_5_l828_828141

/-- The degrees in a full circle -/
def fullCircle := 360

/-- The number of hours on a clock -/
def hoursOnClock := 12

/-- The measure in degrees of the acute angle formed by the minute hand and the hour hand at 3:15 -/
def acuteAngleAt315 : ℝ :=
  let degreesPerHour := fullCircle / hoursOnClock
  let hourHandAt3 := degreesPerHour * 3
  let additionalDegrees := (15 / 60) * degreesPerHour
  let hourHandPosition := hourHandAt3 + additionalDegrees
  let minuteHandPosition := (15 / 60) * fullCircle
  abs (hourHandPosition - minuteHandPosition)

theorem acute_angle_at_315_equals_7_5 : acuteAngleAt315 = 7.5 := by
  sorry

end acute_angle_at_315_equals_7_5_l828_828141


namespace boats_solution_l828_828052

theorem boats_solution (x y : ℕ) (h1 : x + y = 42) (h2 : 6 * x = 8 * y) : x = 24 ∧ y = 18 :=
by
  sorry

end boats_solution_l828_828052


namespace number_of_distinct_remainders_l828_828027

def p : ℕ := 2017

def n_determinant_condition (f : Matrix (Fin n) (Fin n) (ZMod p) → ZMod p) := 
  ∀ (A : Matrix (Fin n) (Fin n) (ZMod p)), ∀ i j, (i ≠ j) → 
  let A' := A.update_row i (A i + A j) in f A = f A'

def a_n (n : ℕ) : ℕ :=
  (card { f : Matrix (Fin n) (Fin n) (ZMod p) → ZMod p // n_determinant_condition f })

def q : ℕ := (2017 ^ 2017 - 1) * (2017 ^ 2016 - 1) / 2016

theorem number_of_distinct_remainders :
  (∃ (r : ℕ), a_n r % q = 2017 * 6 + 4) := sorry

end number_of_distinct_remainders_l828_828027


namespace coprime_f_divisor_prime_even_f_l828_828038

-- Definition of f(n): sum of all divisors of n
def f (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).sum id

-- 1. If m and n are coprime, then f(mn) = f(m) * f(n)
theorem coprime_f (m n : ℕ) (cop : Nat.coprime m n) : 
  f (m * n) = f m * f n := 
  sorry

-- 2. If a is a divisor of n and a < n, also f(n) = n + a, then n is a prime number
theorem divisor_prime (a n : ℕ) (h₁ : a ∣ n) (h₂ : a < n) (h₃ : f n = n + a) : 
  Nat.Prime n := 
  sorry

-- 3. If n is an even number and f(n) = 2n, then there exists a prime number p such that n = 2^(p - 1) * (2^p - 1)
theorem even_f (n : ℕ) (h₁ : Even n) (h₂ : f n = 2 * n) : 
  ∃ p : ℕ, Nat.Prime p ∧ n = 2^(p - 1) * (2^p - 1) := 
  sorry

end coprime_f_divisor_prime_even_f_l828_828038


namespace compute_P_19_l828_828037

noncomputable def T (n : ℕ) := n * (n + 1) * (2 * n + 1) / 6

noncomputable def P (n : ℕ) := ∏ i in (range (n/2)).map (λ k, 2*k+1).filter odd, T i / (T i + 1)

theorem compute_P_19 : P 19 = 0.9 :=
by
  sorry

end compute_P_19_l828_828037


namespace karen_bonus_problem_l828_828020

theorem karen_bonus_problem (n already_graded last_two target : ℕ) (h_already_graded : already_graded = 8)
  (h_last_two : last_two = 290) (h_target : target = 600) (max_score : ℕ)
  (h_max_score : max_score = 150) (required_avg : ℕ) (h_required_avg : required_avg = 75) :
  ∃ A : ℕ, (A = 70) ∧ (target = 600) ∧ (last_two = 290) ∧ (already_graded = 8) ∧
  (required_avg = 75) := by
  sorry

end karen_bonus_problem_l828_828020


namespace domain_of_g_l828_828138

noncomputable def g : ℝ → ℝ := λ x, (x - 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_g :
  {x : ℝ | g x ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_g_l828_828138


namespace tan_A_in_right_triangle_l828_828006

theorem tan_A_in_right_triangle (AC : ℝ) (AB : ℝ) (BC : ℝ) (hAC : AC = Real.sqrt 20) (hAB : AB = 4) (h_right_triangle : AC^2 = AB^2 + BC^2) :
  Real.tan (Real.arcsin (AB / AC)) = 1 / 2 :=
by
  sorry

end tan_A_in_right_triangle_l828_828006


namespace domain_of_g_l828_828137

noncomputable def g : ℝ → ℝ := λ x, (x - 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_g :
  {x : ℝ | g x ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_g_l828_828137


namespace clock_angle_at_315_l828_828148

theorem clock_angle_at_315 : 
  (angle_between_hour_and_minute_at (hours := 3) (minutes := 15)) = 7.5 :=
sorry

end clock_angle_at_315_l828_828148


namespace scientific_notation_of_one_point_six_million_l828_828051

-- Define the given number
def one_point_six_million : ℝ := 1.6 * 10^6

-- State the theorem to prove the equivalence
theorem scientific_notation_of_one_point_six_million :
  one_point_six_million = 1.6 * 10^6 :=
by
  sorry

end scientific_notation_of_one_point_six_million_l828_828051


namespace jason_spent_on_shorts_l828_828016

def total_spent : ℝ := 14.28
def jacket_spent : ℝ := 4.74
def shorts_spent : ℝ := total_spent - jacket_spent

theorem jason_spent_on_shorts :
  shorts_spent = 9.54 :=
by
  -- Placeholder for the proof. The statement is correct as it matches the given problem data.
  sorry

end jason_spent_on_shorts_l828_828016


namespace larger_sample_more_accurate_l828_828001

theorem larger_sample_more_accurate
  (population : Type)
  (sample_size accuracy : ℕ)
  (estimate_accuracy_not_related_to_population_size :
    ∀ (n m : ℕ), accuracy n = accuracy m)
  (estimate_accuracy_related_sample_size :
    ∀ (n : ℕ), ∃ k : ℕ, accuracy k > accuracy n) :
  ∀ n m : ℕ, (n < m) → (accuracy n < accuracy m) :=
by
  sorry

end larger_sample_more_accurate_l828_828001


namespace paint_liters_needed_l828_828046

theorem paint_liters_needed :
  let cost_brushes : ℕ := 20
  let cost_canvas : ℕ := 3 * cost_brushes
  let cost_paint_per_liter : ℕ := 8
  let total_costs : ℕ := 120
  ∃ (liters_of_paint : ℕ), cost_brushes + cost_canvas + cost_paint_per_liter * liters_of_paint = total_costs ∧ liters_of_paint = 5 :=
by
  sorry

end paint_liters_needed_l828_828046


namespace area_of_triangle_l828_828044

-- Definitions for the triangle and its properties
def Triangle (A B C : Type) : Prop := sorry

-- Define the properties of the triangle
def right_triangle (T : Triangle) (R : (ℝ × ℝ)) : Prop := sorry
def hypotenuse (T : Triangle) (PQ_length : ℝ) : Prop := PQ_length = 50
def median_lines (P Q : (ℝ × ℝ)) : Prop := 
  line_through (P : P.2 = P.1 + 5) ∧ 
  line_through (Q : Q.2 = 3 * Q.1 + 6)

-- The main theorem stating the area of the given triangle
theorem area_of_triangle (P Q R : (ℝ × ℝ)) (T : Triangle P Q R) 
  (right_angle_at_R : right_triangle T R)
  (PQ_length_50 : hypotenuse T 50)
  (medians_condition : median_lines P Q) : 
  (Area : ℝ) := 
  Area = 11250 / 31 :=
sorry

end area_of_triangle_l828_828044


namespace degree_polynomial_example_l828_828132

noncomputable def degree_of_monomial (c : ℝ) (n : ℕ) : ℕ := n

noncomputable def poly_degree (P : Type) (deg_P : ℕ) (n : ℕ) : ℕ :=
  n * deg_P

theorem degree_polynomial_example :
  let P := (2 : ℝ) * (λ x : ℝ, x ^ 3) + 5
  let deg_P := degree_of_monomial 2 3
  let deg_P8 := poly_degree P deg_P 8
  let deg_P8_7 := poly_degree P deg_P8 7
  deg_P8_7 = 168 :=
by
  sorry

end degree_polynomial_example_l828_828132


namespace acute_angle_at_3_15_l828_828144

/-- The hour and minute hands' angles and movements are defined as follows. -/
def hour_hand_angle (h m : Nat) : Real := (h % 12) * 30 + m * 0.5
def minute_hand_angle (m : Nat) : Real := (m % 60) * 6

/-- The condition that an acute angle is the smaller angle between hands. -/
def acute_angle (angle1 angle2 : Real) : Real := abs (angle1 - angle2)

/-- At 3:15, the acute angle between the hour and minute hands should be 7.5 degrees. -/
theorem acute_angle_at_3_15
    : acute_angle (hour_hand_angle 3 15) (minute_hand_angle 15) = 7.5 :=
by
    sorry

end acute_angle_at_3_15_l828_828144


namespace tangent_parabola_ellipse_l828_828098

-- Definitions of the parabolic and elliptic equations
def parabola (y x : ℝ) : Prop := y = x^2 + 2
def ellipse (y x m : ℝ) : Prop := 2 * m * x^2 + y^2 = 4

-- The tangency condition implies a specific value for m
theorem tangent_parabola_ellipse (m : ℝ) :
  (∀ (x y : ℝ), parabola y x → ellipse y x m) ↔ m = 2 + sqrt 3 := 
sorry

end tangent_parabola_ellipse_l828_828098


namespace bird_wings_l828_828017

theorem bird_wings (P Pi C : ℕ) (h_total_money : 4 * 50 = 200)
  (h_total_cost : 30 * P + 20 * Pi + 15 * C = 200)
  (h_P_ge : P ≥ 1) (h_Pi_ge : Pi ≥ 1) (h_C_ge : C ≥ 1) :
  2 * (P + Pi + C) = 24 :=
sorry

end bird_wings_l828_828017


namespace solve_y_positive_root_l828_828067

def quadratic_solution (a b c : ℝ) (y : ℝ) : Prop :=
  a * y^2 + b * y + c = 0

theorem solve_y_positive_root :
  ∃ y > 0, quadratic_solution 6 5 (-12) y ∧ y = (-5 + real.sqrt 313) / 12 :=
by
  sorry

end solve_y_positive_root_l828_828067


namespace team_a_builds_30m_per_day_l828_828075

noncomputable def road_building (x : ℝ) : Prop :=
  120 * (x + 10) = 160 * x

theorem team_a_builds_30m_per_day :
  ∃ (x : ℝ), road_building x ∧ x = 30 :=
by
  use 30
  unfold road_building
  have h : 120 * (30 + 10) = 160 * 30 := by
    calc
      120 * (30 + 10) = 120 * 40 : by sorry
      _ = 4800 : by sorry
      160 * 30 = 4800 : by sorry
  exact ⟨h, rfl⟩

end team_a_builds_30m_per_day_l828_828075


namespace nephroid_envelope_l828_828059

noncomputable theory
open Complex

-- Define the conditions
def A (ψ : ℝ) : ℂ := exp (I * ψ)
def A' (ψ : ℝ) : ℂ := exp (-I * ψ)
def A1 (ψ : ℝ) : ℂ := exp (I * (3 * ψ + Real.pi))

-- State the theorem to be proved
theorem nephroid_envelope :
  ∀ ψ ∈ Icc 0 (2 * Real.pi), is_nephroid (A1 ψ) := sorry

end nephroid_envelope_l828_828059


namespace geometric_sequence_ratio_l828_828043

-- Define the geometric sequence and sums
variables {α : Type*} [linear_ordered_field α]
variables (a r : α) (S : ℕ → α)

-- Define the term and sum of the geometric sequence
def a_n (n : ℕ) : α := a * r ^ n
def S_n (n : ℕ) : α := ∑ i in range n, a_n a r i

-- Assume conditions from part (a)
variables (h1 : S_n a r 6 / S_n a r 3 = 3)

-- State the goal to prove
theorem geometric_sequence_ratio :
  S_n a r 12 / S_n a r 9 = 15 / 7 := 
sorry

end geometric_sequence_ratio_l828_828043


namespace ratio_PE_ED_l828_828057

variables {s : ℝ}

-- Define the coordinates of the square ABCD with side length s.
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (s, 0)
def C : ℝ × ℝ := (s, s)
def D : ℝ × ℝ := (0, s)

-- Define the coordinates of points P and Q based on given ratios.
def P : ℝ × ℝ := (s / 3, 0)
def Q : ℝ × ℝ := (s, 2 * s / 3)

-- Define the coordinates of the intersection point E of lines DP and AQ.
def E : ℝ × ℝ := (3 * s / 11, 2 * s / 11)

-- Define the distances PE and ED.
def PE : ℝ := real.sqrt ((3 * s / 11 - s / 3)^2 + (2 * s / 11 - 0)^2)
def ED : ℝ := real.sqrt ((3 * s / 11 - 0)^2 + (2 * s / 11 - s)^2)

-- Prove the ratio of PE to ED.
theorem ratio_PE_ED : PE / ED = 2 / 9 := by
  sorry

end ratio_PE_ED_l828_828057


namespace shorter_tree_height_l828_828107

theorem shorter_tree_height
  (s : ℝ)
  (h₁ : ∀ s, s > 0 )
  (h₂ : s + (s + 20) = 240)
  (h₃ : s / (s + 20) = 5 / 7) :
  s = 110 :=
by
sorry

end shorter_tree_height_l828_828107


namespace find_common_ratio_l828_828000

-- Define a geometric sequence
def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∃ a₁, ∀ n, a (n + 1) = a₁ * q^n

-- Sum of the first n terms of the sequence
def sum_seq (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n, S n = (n + 1) * a 1 * (1 - q^(n + 1)) / (1 - q)

noncomputable def a₅ (a : ℕ → ℝ) (S : ℕ → ℝ) := 2 * S 4 + 3
noncomputable def a₆ (a : ℕ → ℝ) (S : ℕ → ℝ) := 2 * S 5 + 3

-- Main theorem statement
theorem find_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h_geom : geom_seq a q) 
  (h_sum : sum_seq a S) 
  (h₁ : a 5 = a₅ a S) 
  (h₂ : a 6 = a₆ a S) : 
  q = 3 := 
  sorry

end find_common_ratio_l828_828000


namespace decreasing_interval_of_log_function_l828_828092

noncomputable def f (x : ℝ) := Real.log (4 + 3 * x - x^2)

theorem decreasing_interval_of_log_function :
  (∀ x : ℝ, 4 + 3 * x - x^2 > 0 → (∃ interval : set ℝ, interval = [3 / 2, 4) ∧ ∀ y ∈ interval, ∀ z ∈ interval, y ≤ z → f y ≥ f z)) :=
by 
  sorry

end decreasing_interval_of_log_function_l828_828092


namespace angle_at_3_15_is_7_point_5_degrees_l828_828155

-- Definitions for the positions of the hour and minute hands
def minute_hand_position (minutes: ℕ) : ℝ := (minutes / 60.0) * 360.0
def hour_hand_position (hours: ℕ) (minutes: ℕ) : ℝ := (hours * 30) + (minutes * 0.5)

-- The time 3:15
def time_3_15 := (3, 15)

-- The acute angle calculation
def acute_angle_between_hands (hour: ℕ) (minute: ℕ) : ℝ :=
  let minute_angle := minute_hand_position minute
  let hour_angle := hour_hand_position hour minute
  abs (minute_angle - hour_angle)

-- The theorem statement
theorem angle_at_3_15_is_7_point_5_degrees : 
  acute_angle_between_hands 3 15 = 7.5 := 
  sorry

end angle_at_3_15_is_7_point_5_degrees_l828_828155


namespace projection_calculation_l828_828101

noncomputable def vector_projection (u w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let scalar_multiply (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (c * v.1, c * v.2, c * v.3)
  let magnitude_sq (v : ℝ × ℝ × ℝ) : ℝ := dot_product v v
  scalar_multiply (dot_product u w / magnitude_sq w) w

theorem projection_calculation :
  let w := (-1, 0.5, -1.5)
  let u1 := (2, -1, 3)
  let proj_u1_on_w := vector_projection u1 w
  let u2 := (-3, 2, 4)
  proj_u1_on_w = w →
  vector_projection u2 w = (-16/7, 8/7, -24/7) :=
by
  sorry

end projection_calculation_l828_828101


namespace range_of_arctan_f_l828_828103

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 1)

theorem range_of_arctan_f :
  (set.Ioo (-π / 2) (-π / 4) ∪ set.Ioc (-π / 4) 0 ∪ set.Ioo 0 (π / 2)) =
    {y | ∃ x : ℝ, y = arctan (f x)} :=
begin
  sorry
end

end range_of_arctan_f_l828_828103


namespace quadrilaterals_not_congruent_l828_828010

theorem quadrilaterals_not_congruent (A B C D A' B' C' D' : Type*) 
  (h1 : AB = A'B') (h2 : BC = B'C') (h3 : CD = C'D') (h4 : DA = D'A') :
  ¬(ABCD ≅ A'B'C'D') :=
sorry

end quadrilaterals_not_congruent_l828_828010


namespace maximum_value_of_expression_l828_828033

noncomputable def max_value (x y z w : ℝ) : ℝ := 2 * x + 3 * y + 5 * z - 4 * w

theorem maximum_value_of_expression 
  (x y z w : ℝ)
  (h : 9 * x^2 + 4 * y^2 + 25 * z^2 + 16 * w^2 = 4) : 
  max_value x y z w ≤ 6 * Real.sqrt 6 :=
sorry

end maximum_value_of_expression_l828_828033


namespace angle_at_3_15_is_7_point_5_degrees_l828_828154

-- Definitions for the positions of the hour and minute hands
def minute_hand_position (minutes: ℕ) : ℝ := (minutes / 60.0) * 360.0
def hour_hand_position (hours: ℕ) (minutes: ℕ) : ℝ := (hours * 30) + (minutes * 0.5)

-- The time 3:15
def time_3_15 := (3, 15)

-- The acute angle calculation
def acute_angle_between_hands (hour: ℕ) (minute: ℕ) : ℝ :=
  let minute_angle := minute_hand_position minute
  let hour_angle := hour_hand_position hour minute
  abs (minute_angle - hour_angle)

-- The theorem statement
theorem angle_at_3_15_is_7_point_5_degrees : 
  acute_angle_between_hands 3 15 = 7.5 := 
  sorry

end angle_at_3_15_is_7_point_5_degrees_l828_828154


namespace M_not_integer_N_not_integer_K_not_integer_l828_828061

noncomputable def M (n : ℕ) : ℝ := (∑ i in finset.range(n-1) + 2, 1 / (i : ℝ))

noncomputable def N (n m : ℕ) : ℝ := (∑ i in finset.range(m + 1), 1 / (n + i : ℝ))

noncomputable def K (n : ℕ) : ℝ := (∑ i in finset.range(n), 1 / (2 * i + 3 : ℝ))

theorem M_not_integer (n : ℕ) (h : 0 < n) : ¬(M n ∈ ℤ) := sorry

theorem N_not_integer (n m : ℕ) (hn : 0 < n) (hm : 0 < m) : ¬(N n m ∈ ℤ) := sorry

theorem K_not_integer (n : ℕ) (h : 0 < n) : ¬(K n ∈ ℤ) := sorry

end M_not_integer_N_not_integer_K_not_integer_l828_828061


namespace acute_angle_at_315_equals_7_5_l828_828143

/-- The degrees in a full circle -/
def fullCircle := 360

/-- The number of hours on a clock -/
def hoursOnClock := 12

/-- The measure in degrees of the acute angle formed by the minute hand and the hour hand at 3:15 -/
def acuteAngleAt315 : ℝ :=
  let degreesPerHour := fullCircle / hoursOnClock
  let hourHandAt3 := degreesPerHour * 3
  let additionalDegrees := (15 / 60) * degreesPerHour
  let hourHandPosition := hourHandAt3 + additionalDegrees
  let minuteHandPosition := (15 / 60) * fullCircle
  abs (hourHandPosition - minuteHandPosition)

theorem acute_angle_at_315_equals_7_5 : acuteAngleAt315 = 7.5 := by
  sorry

end acute_angle_at_315_equals_7_5_l828_828143


namespace triangle_inradius_point_of_contact_l828_828024

theorem triangle_inradius_point_of_contact
  (A B C I T : Point)
  (r : ℝ)
  (h_triangle : Triangle ABC)
  (h_angle_A : Angle A = 60)
  (h_contact : IncircleNinePointContact I T)
  (h_inradius : Inradius ABC = r)
  : Distance A T = r := by
  sorry

end triangle_inradius_point_of_contact_l828_828024


namespace pyramid_volume_l828_828076

noncomputable def volume_of_inscribed_pyramid 
  (a b : ℝ) 
  (α : ℝ) 
  (α_pos : α > 0)
  (α_lt_pi_div_2 : α < π / 2) : ℝ := 
  (Real.cot α * (b^2 + a * b)^(3/2)) / 24

theorem pyramid_volume 
  (a b α : ℝ) 
  (α_pos : α > 0)
  (α_lt_pi_div_2 : α < π / 2)
  (h_cone: α ∈ (set.Ioo 0 (π / 2))) :
  let V := volume_of_inscribed_pyramid a b α in
  V = (Real.cot α * (b^2 + a * b)^(3 / 2)) / 24 := 
begin
  sorry
end

end pyramid_volume_l828_828076


namespace find_arithmetic_fifth_term_l828_828036

open Real

theorem find_arithmetic_fifth_term (S : ℕ → ℝ) (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) :
  S 6 = -33 ∧ a 1 = 2 ∧ 
  (∀ n, S n = n * a1 + (n * (n - 1)) / 2 * d) ∧ 
  (∀ n, a n = a1 + (n - 1) * d) → 
  a 5 = -10 :=
by
  assume h : S 6 = -33 ∧ a 1 = 2 ∧ 
              (∀ n, S n = n * a1 + (n * (n - 1)) / 2 * d) ∧ 
              (∀ n, a n = a1 + (n - 1) * d)
  sorry

end find_arithmetic_fifth_term_l828_828036


namespace foci_of_hyperbola_l828_828078

theorem foci_of_hyperbola (x y : ℝ) : x^2 - 4 * y^2 = 4 → ∃ c, c = √5 ∧ ((x, y) = (c, 0) ∨ (x, y) = (-c, 0)) :=
by
  intro h
  have h : (x^2 / 4) - (y^2 / 1) = 1 := sorry
  have a := 2
  have b := 1
  have c : ℝ := Real.sqrt (a^2 + b^2)
  sorry

end foci_of_hyperbola_l828_828078


namespace irrational_division_l828_828160

theorem irrational_division (x : ℝ) (h : irrational x) : (0 < x ∨ x < 0) :=
sorry

end irrational_division_l828_828160


namespace find_sequence_l828_828105

/-- The given sequence sum condition -/
def S : ℕ → ℕ
| n := 2^n

/-- Required sequence a_n solution -/
def a : ℕ → ℕ
| 0 := 0  -- Typically, sequences start from n = 1, and n = 0 can be treated separately 
| 1 := 2
| (n + 2) := 2^(n + 1)

theorem find_sequence (n : ℕ) : 
    S := λ n, 2^n → 
    a n = 
    if n = 1 then 2 
    else if n ≥ 2 then 2^(n-1)
    else 0 :=
begin
  intros,
  sorry,
end

end find_sequence_l828_828105


namespace negation_of_existence_l828_828093

theorem negation_of_existence : 
  (¬ ∃ x : ℝ, Real.exp x = x - 1) = (∀ x : ℝ, Real.exp x ≠ x - 1) :=
by
  sorry

end negation_of_existence_l828_828093


namespace solution_set_of_inequality_l828_828085

def f (x : ℝ) : ℝ :=
if x < 2 then 2 * Real.exp(x - 1) else Real.log (x^2 - 1) / Real.log 3

theorem solution_set_of_inequality :
  {x : ℝ | f x > 2} = {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | Real.sqrt 10 < x} :=
by
  sorry

end solution_set_of_inequality_l828_828085


namespace max_sum_frac_eq_half_l828_828035

theorem max_sum_frac_eq_half :
  ∀ (x : Fin 50 → ℝ), (∀ i, 0 < x i) → (∑ i, (x i) ^ 2) = 1 → 
    (∑ i, x i / (1 + (x i) ^ 2)) ≤ 1/2 :=
by
  intros x hx hsum
  sorry

end max_sum_frac_eq_half_l828_828035


namespace minimum_rental_cost_l828_828121

open Nat

theorem minimum_rental_cost :
  ∃ (x : ℕ), 3 ≤ x ∧ x ≤ 5 ∧
  (∃ y : ℕ, y = 8 - x ∧ (35 * x + 30 * y ≥ 255) ∧ (400 * x + 320 * y ≤ 3000) ∧
  (∀ z : ℕ, 3 ≤ z ∧ z ≤ 5 →
    let w := 8 - z in
    (35 * z + 30 * w ≥ 255) →
    (400 * z + 320 * w ≤ 3000) →
    400 * x + 320 * y ≤ 400 * z + 320 * w)) ∧
  400 * x + 320 * (8 - x) = 2800 := 
by
  exists 3
  apply and.intro
  show 3 ≤ 3 from le_refl 3
  apply and.intro
  show 3 ≤ 5 from le_of_lt (by norm_num)
  exists (8 - 3)
  apply and.intro
  show 8 - 3 = 5 from rfl
  sorry

end minimum_rental_cost_l828_828121


namespace goods_train_speed_l828_828167

theorem goods_train_speed (man_train_speed_kmh : Float) 
    (goods_train_length_m : Float) 
    (passing_time_s : Float) 
    (kmh_to_ms : Float := 1000 / 3600) : 
    man_train_speed_kmh = 50 → 
    goods_train_length_m = 280 → 
    passing_time_s = 9 → 
    Float.round ((goods_train_length_m / passing_time_s + man_train_speed_kmh * kmh_to_ms) * 3600 / 1000) = 61.99
:= by
  sorry

end goods_train_speed_l828_828167


namespace set_contains_negatives_l828_828042

theorem set_contains_negatives (A : Set ℝ) (n : ℕ) (h1 : 2 ≤ n) (h2 : Set.card A < n) 
  (h3 : ∀ k ∈ Finset.range n, ∃ (B : Finset ℝ), ↑B ⊆ A ∧ B.sum id = 2^k) : 
  ∃ a ∈ A, a < 0 := 
sorry

end set_contains_negatives_l828_828042


namespace area_ratio_of_triangles_l828_828007

def Triangle :=
{ A B C : Type }
  (side : B → B → ℝ)
  (angle_bisector : C → C → B → Prop)

noncomputable def triangle_example : Triangle :=
{ A := ℝ,
  B := ℝ,
  C := ℝ,
  side := λ x y, if x = y then 0 else if x < y then y - x else x - y,
  angle_bisector := λ h k p, p * p = h * h + k * k - h * k }

axiom DE : ℝ := 36
axiom DF : ℝ := 45
axiom splits_angle : ∀ {EF G}, triangle_example.angle_bisector DE DF EF → triangle_example.angle_bisector DE DF G → G ∈ EF 

noncomputable def area_ratio (D E F G : ℝ) : ℝ :=
if G ∈ (DE, DF) then (DF / DE) else 0

theorem area_ratio_of_triangles (D E F G : ℝ) (h : triangle_example.angle_bisector D E F) (split : triangle_example.angle_bisector D E G ∧ splits_angle) :
  area_ratio D E F G = 5 / 4 :=
by
  sorry

end area_ratio_of_triangles_l828_828007


namespace acute_angle_at_315_equals_7_5_l828_828140

/-- The degrees in a full circle -/
def fullCircle := 360

/-- The number of hours on a clock -/
def hoursOnClock := 12

/-- The measure in degrees of the acute angle formed by the minute hand and the hour hand at 3:15 -/
def acuteAngleAt315 : ℝ :=
  let degreesPerHour := fullCircle / hoursOnClock
  let hourHandAt3 := degreesPerHour * 3
  let additionalDegrees := (15 / 60) * degreesPerHour
  let hourHandPosition := hourHandAt3 + additionalDegrees
  let minuteHandPosition := (15 / 60) * fullCircle
  abs (hourHandPosition - minuteHandPosition)

theorem acute_angle_at_315_equals_7_5 : acuteAngleAt315 = 7.5 := by
  sorry

end acute_angle_at_315_equals_7_5_l828_828140


namespace sum_x_y_m_l828_828039

theorem sum_x_y_m (x y m : ℕ) (h1 : x >= 10 ∧ x < 100) (h2 : y >= 10 ∧ y < 100) 
  (h3 : ∃ a b : ℕ, x = 10 * a + b ∧ y = 10 * b + a ∧ a < 10 ∧ b < 10) 
  (h4 : x^2 - y^2 = 4 * m^2) : 
  x + y + m = 105 := 
sorry

end sum_x_y_m_l828_828039


namespace miles_to_burger_restaurant_l828_828021

-- Definitions and conditions
def miles_per_gallon : ℕ := 19
def gallons_of_gas : ℕ := 2
def miles_to_school : ℕ := 15
def miles_to_softball_park : ℕ := 6
def miles_to_friend_house : ℕ := 4
def miles_to_home : ℕ := 11
def total_gas_distance := miles_per_gallon * gallons_of_gas
def total_known_distances := miles_to_school + miles_to_softball_park + miles_to_friend_house + miles_to_home

-- Problem statement to prove
theorem miles_to_burger_restaurant :
  ∃ (miles_to_burger_restaurant : ℕ), 
  total_gas_distance = total_known_distances + miles_to_burger_restaurant ∧ miles_to_burger_restaurant = 2 := 
by
  sorry

end miles_to_burger_restaurant_l828_828021


namespace acute_angle_at_3_15_l828_828145

/-- The hour and minute hands' angles and movements are defined as follows. -/
def hour_hand_angle (h m : Nat) : Real := (h % 12) * 30 + m * 0.5
def minute_hand_angle (m : Nat) : Real := (m % 60) * 6

/-- The condition that an acute angle is the smaller angle between hands. -/
def acute_angle (angle1 angle2 : Real) : Real := abs (angle1 - angle2)

/-- At 3:15, the acute angle between the hour and minute hands should be 7.5 degrees. -/
theorem acute_angle_at_3_15
    : acute_angle (hour_hand_angle 3 15) (minute_hand_angle 15) = 7.5 :=
by
    sorry

end acute_angle_at_3_15_l828_828145


namespace find_e_value_l828_828045

noncomputable def e_value (a b : ℝ) := ((a + b*complex.I)^2 + (complex.inv (a + b*complex.I))^2)

theorem find_e_value (a b : ℝ) (h : (a + b*complex.I) + complex.inv (a + b*complex.I) = 5) : 
  e_value a b = 3.5 :=
begin
  sorry
end

end find_e_value_l828_828045


namespace angle_at_3_15_is_7_point_5_degrees_l828_828153

-- Definitions for the positions of the hour and minute hands
def minute_hand_position (minutes: ℕ) : ℝ := (minutes / 60.0) * 360.0
def hour_hand_position (hours: ℕ) (minutes: ℕ) : ℝ := (hours * 30) + (minutes * 0.5)

-- The time 3:15
def time_3_15 := (3, 15)

-- The acute angle calculation
def acute_angle_between_hands (hour: ℕ) (minute: ℕ) : ℝ :=
  let minute_angle := minute_hand_position minute
  let hour_angle := hour_hand_position hour minute
  abs (minute_angle - hour_angle)

-- The theorem statement
theorem angle_at_3_15_is_7_point_5_degrees : 
  acute_angle_between_hands 3 15 = 7.5 := 
  sorry

end angle_at_3_15_is_7_point_5_degrees_l828_828153


namespace blue_balls_removed_l828_828109

theorem blue_balls_removed :
  ∀ (total_balls percent_red percent_red_target: ℕ),
  total_balls = 100 →
  percent_red = 36 →
  percent_red_target = 72 →
  ∃ (x: ℕ), (x = 50) ∧ (percent_red * total_balls = percent_red_target * (total_balls - x)) :=
by
  intros total_balls percent_red percent_red_target h_total h_red h_red_target
  use 50
  split
  · refl
  · sorry

end blue_balls_removed_l828_828109


namespace tangent_line_passing_through_origin_l828_828080

open Real

/-- The curve is defined as y = e^(x - 1) + x. We are to prove that the tangent line at the point of tangency passing through the origin is y = 2x. -/
theorem tangent_line_passing_through_origin :
  (∀ x : ℝ, let y := exp (x - 1) + x in
  let y' := exp (x - 1) + 1 in
  ∃ x0 : ℝ, (let k := exp (x0 - 1) + 1 in
  let tangent := k * (x - x0) + (exp (x0 - 1) + x0) in
  tangent = 2 * x)) :=
sorry

end tangent_line_passing_through_origin_l828_828080


namespace largest_divisor_of_m_squared_minus_n_squared_l828_828032

theorem largest_divisor_of_m_squared_minus_n_squared (a b : ℤ) (h : a > b) : 
  ∃ d, (∀ m n, (m = 2 * a + 3) → (n = 2 * b + 1) → d ∣ (m^2 - n^2)) ∧ 
       (∀ d', (∀ m n, (m = 2 * a + 3) → (n = 2 * b + 1) → d' ∣ (m^2 - n^2)) → d' ≤ d) :=
begin
  use 4,
  split,
  { intros m n hm hn,
    rw [hm, hn],
    ring,
    apply dvd.intro (4 * (a - b + 1) * (a + b + 1)) rfl, },
  { intros d' hd',
    by_contradiction,
    sorry, } -- Proof omitted
end

end largest_divisor_of_m_squared_minus_n_squared_l828_828032


namespace pond_volume_l828_828168

theorem pond_volume (L W H : ℝ) (hL : L = 20) (hW : W = 10) (hH : H = 5) : 
  L * W * H = 1000 :=
by
  rw [hL, hW, hH]
  norm_num

end pond_volume_l828_828168


namespace propositional_truths_l828_828113

-- Define the propositions as terms
def P1 := ∃ x : ℝ, sin x + cos x = 2
def P2 := ∃ x : ℝ, sin 2x = sin x
def P3 := ∀ x ∈ set.Icc (-Real.pi / 2) (Real.pi / 2), Real.sqrt ((1 + cos (2 * x)) / 2) = cos x
def P4 := ∀ x ∈ set.Ioo 0 Real.pi, sin x > cos x

-- Statement that encapsulates the problem and answers directly
theorem propositional_truths :
  ¬ P1 ∧ P2 ∧ P3 ∧ ¬ P4 :=
by
  sorry

end propositional_truths_l828_828113


namespace students_stand_together_arrangements_students_not_next_each_other_arrangements_teachers_students_alternate_arrangements_l828_828111

theorem students_stand_together_arrangements (teachers students : ℕ) 
    (h1 : teachers = 4) (h2 : students = 4) :
    let total_arrangements := 2880 in
    total_arrangements = 2880 := by
  sorry

theorem students_not_next_each_other_arrangements (teachers students : ℕ) 
    (h1 : teachers = 4) (h2 : students = 4) :
    let total_arrangements := 2880 in
    total_arrangements = 2880 := by
  sorry

theorem teachers_students_alternate_arrangements (teachers students : ℕ) 
    (h1 : teachers = 4) (h2 : students = 4) :
    let total_arrangements := 1152 in
    total_arrangements = 1152 := by
  sorry

end students_stand_together_arrangements_students_not_next_each_other_arrangements_teachers_students_alternate_arrangements_l828_828111


namespace domain_g_l828_828135

noncomputable def g (x : ℝ) : ℝ := (x - 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_g : {x : ℝ | g x = (x - 3) / Real.sqrt (x^2 - 5 * x + 6)} = 
  {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_g_l828_828135


namespace milk_lowering_height_l828_828091

-- Define the conditions
def box_length : ℝ := 64
def box_width : ℝ := 25
def gallons_to_remove : ℝ := 6000
def gallons_per_cubic_foot : ℝ := 7.48052

-- The theorem to prove
theorem milk_lowering_height : 
  let volume_to_remove := gallons_to_remove / gallons_per_cubic_foot in
  let area := box_length * box_width in
  let height_in_feet := volume_to_remove / area in
  let height_in_inches := height_in_feet * 12 in
  height_in_inches ≈ 6.015 :=
by
  sorry

end milk_lowering_height_l828_828091


namespace fourth_term_geometric_sequence_l828_828083

theorem fourth_term_geometric_sequence (x : ℝ) :
  ∃ r : ℝ, (r > 0) ∧ 
  x ≠ 0 ∧
  (3 * x + 3)^2 = x * (6 * x + 6) →
  x = -3 →
  6 * x + 6 ≠ 0 →
  4 * (6 * x + 6) * (3 * x + 3) = -24 :=
by
  -- Placeholder for the proof steps
  sorry

end fourth_term_geometric_sequence_l828_828083


namespace pulsar_stand_time_l828_828062

theorem pulsar_stand_time : ∃ P : ℝ, 
  P + 3 * P + P / 2 = 45 ∧ P = 10 :=
by 
  use 10
  split
  { norm_num }
  { refl }

end pulsar_stand_time_l828_828062


namespace rectangle_toothpicks_l828_828127

/-- A rectangle formed with toothpicks, where the length is 20 toothpicks and the width is 10 toothpicks,
    will use a total of 430 toothpicks --/
theorem rectangle_toothpicks (length width : ℕ) (h_length : length = 20) (h_width : width = 10) : 
    (length + 1) * width + (width + 1) * length = 430 := 
by {
  rw [h_length, h_width],
  sorry
}

end rectangle_toothpicks_l828_828127


namespace xiaogang_xiaoqiang_speeds_and_time_l828_828164

theorem xiaogang_xiaoqiang_speeds_and_time
  (x y : ℕ)
  (distance_meeting : 2 * x = 2 * y + 24)
  (xiaogang_time_after_meeting : 0.5 * x = d_x)
  (total_distance : 2 * x + (2 * x - 24) = D)
  (xiaogang_time_total : D / x = meeting_time + 0.5)
  (xiaoqiang_time_total : D / y = meeting_time + time_xiaoqiang_to_A) :
  x = 16 ∧ y = 4 ∧ time_xiaoqiang_to_A = 8 := by
sorry

end xiaogang_xiaoqiang_speeds_and_time_l828_828164


namespace cubic_sum_identity_l828_828070

theorem cubic_sum_identity
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : ab + ac + bc = -3)
  (h3 : abc = 9) :
  a^3 + b^3 + c^3 = 22 :=
by
  sorry

end cubic_sum_identity_l828_828070


namespace amount_of_tin_in_new_alloy_l828_828116

noncomputable def percentage_of_zinc_in_alloys := 30
noncomputable def percentage_of_tin_in_first_alloy := 40
noncomputable def percentage_of_copper_in_second_alloy := 26
noncomputable def weight_of_first_alloy := 150 : ℝ -- kg
noncomputable def weight_of_second_alloy := 250 : ℝ -- kg
noncomputable def percentage_of_tin_in_second_alloy := 100 - (percentage_of_copper_in_second_alloy + percentage_of_zinc_in_alloys) : ℝ
noncomputable def total_weight := weight_of_first_alloy + weight_of_second_alloy

noncomputable def amount_of_tin_in_first_alloy : ℝ := (percentage_of_tin_in_first_alloy / 100) * weight_of_first_alloy
noncomputable def amount_of_tin_in_second_alloy : ℝ := (percentage_of_tin_in_second_alloy / 100) * weight_of_second_alloy

theorem amount_of_tin_in_new_alloy : 
  amount_of_tin_in_first_alloy + amount_of_tin_in_second_alloy = 170 :=
by
  sorry

end amount_of_tin_in_new_alloy_l828_828116


namespace g_decreasing_on_interval_l828_828123

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * cos (2 * x) - sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := f (x - π / 12)

theorem g_decreasing_on_interval : ∀ x ∈ Set.Icc (0 : ℝ) (π / 2), g x' < g x :=
sorry

end g_decreasing_on_interval_l828_828123


namespace parabola_intersections_l828_828125

theorem parabola_intersections :
  (∀ x y, (y = 4 * x^2 + 4 * x - 7) ↔ (y = x^2 + 5)) →
  (∃ (points : List (ℝ × ℝ)),
    (points = [(-2, 9), (2, 9)]) ∧
    (∀ p ∈ points, ∃ x, p = (x, x^2 + 5) ∧ y = 4 * x^2 + 4 * x - 7)) :=
by sorry

end parabola_intersections_l828_828125


namespace area_of_triangle_CDJE_l828_828063

variables {C D E F G H J : Type} [linear_ordered_field F] [measure_space F] [add_comm_group F]
          {CDEF : F} [parallelogram CDEF] (G H : F) [midpoint G C D] [midpoint H E F] (CDJ : F) 

theorem area_of_triangle_CDJE (h₁ : area CDEF = 36) : area CDJ = 36 := 
sorry

end area_of_triangle_CDJE_l828_828063


namespace product_inequality_l828_828040

noncomputable theory
open_locale big_operators

theorem product_inequality (n : ℕ) (x : (Fin n → ℝ)) (hx : ∀ i, 0 < x i) :
  (∏ i in Finset.range n, ∑ j in Finset.range (i + 1), x j + 1) ≥ Real.sqrt ((n.succ)^(n.succ) * ∏ i in Finset.range n, x i) :=
sorry

end product_inequality_l828_828040


namespace sum_nine_terms_l828_828029

variable {a : ℕ → ℝ} (d : ℝ)

/-- The definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), a (n + 1) - a n = d

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n : ℝ) * (a 1 + a n) / 2

noncomputable def a5 (a : ℕ → ℝ) := a 1 + 4 * d

theorem sum_nine_terms (a : ℕ → ℝ) (d : ℝ) (h1 : is_arithmetic_sequence a) (h2 : a 3 + a 5 + a 7 = 27) :
  sum_arithmetic_sequence a 9 = 81 :=
  sorry

end sum_nine_terms_l828_828029


namespace remainder_b_91_mod_49_l828_828031

def b (n : ℕ) := 12^n + 14^n

theorem remainder_b_91_mod_49 : (b 91) % 49 = 38 := by
  sorry

end remainder_b_91_mod_49_l828_828031


namespace clock_angle_at_315_l828_828151

theorem clock_angle_at_315 : 
  (angle_between_hour_and_minute_at (hours := 3) (minutes := 15)) = 7.5 :=
sorry

end clock_angle_at_315_l828_828151


namespace John_bought_new_socks_l828_828018

theorem John_bought_new_socks (initial_socks : ℕ) (thrown_away_socks : ℕ) (current_socks : ℕ) :
    initial_socks = 33 → thrown_away_socks = 19 → current_socks = 27 → 
    current_socks = (initial_socks - thrown_away_socks) + 13 :=
by
  sorry

end John_bought_new_socks_l828_828018


namespace clock_angle_at_315_l828_828149

theorem clock_angle_at_315 : 
  (angle_between_hour_and_minute_at (hours := 3) (minutes := 15)) = 7.5 :=
sorry

end clock_angle_at_315_l828_828149


namespace misha_lying_l828_828118

-- Definitions based on the problem statement
def players := Fin 10
def goals := {n // n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 5}

-- Defining the goals scored if Misha is assumed to tell the truth
def misha_goals_correct : goals := ⟨2, or.inr $ or.inl rfl⟩

-- Conditions and the goal to check if 2n3 + 4n5 = 9 leads to contradiction
theorem misha_lying (n1 n3 n5 : ℕ) :
  n1 + n3 + n5 = 9 → 1 * n1 + 3 * n3 + 5 * n5 + 2 = 20 → ¬ (2 * n3 + 4 * n5 = 9) :=
by
  intro h1 h2
  have h : (1 * n1) + 3 * n3 + 5 * n5 = 18 := by linarith
  have h_eq : 2 * n3 + 4 * n5 = 9 := by linarith
  -- Since 2n3 + 4n5 = 9 has no integer solutions, hence Misha is lying.
  sorry

end misha_lying_l828_828118


namespace evaluate_expression_l828_828156

theorem evaluate_expression (a b : ℤ) (h_a : a = 4) (h_b : b = -3) : -a - b^3 + a * b = 11 :=
by
  rw [h_a, h_b]
  sorry

end evaluate_expression_l828_828156


namespace height_of_fifth_sphere_l828_828054

theorem height_of_fifth_sphere (r : ℝ) (h_r : r = 22 - 11 * Real.sqrt 2) : 
  let d := 3 * r in -- distance from the topmost point of the fifth sphere to the plane
  d = 22 := 
by
  -- proof by construction
  sorry

end height_of_fifth_sphere_l828_828054


namespace rate_of_mangoes_l828_828122

def rate_of_apples := 70
def quantity_of_apples := 8
def total_amount_paid := 1055
def quantity_of_mangoes := 9

theorem rate_of_mangoes : 
    let cost_of_apples := rate_of_apples * quantity_of_apples in
    let total_cost_of_mangoes := total_amount_paid - cost_of_apples in
    total_cost_of_mangoes / quantity_of_mangoes = 55 :=
by
    sorry

end rate_of_mangoes_l828_828122


namespace certain_number_divisibility_l828_828110

-- Define a number certain_number such that there are exactly 3 integers between 1 and certain_number
-- that are divisible by the least common multiple of 10, 25, and 35.
def lcm_10_25_35 : ℕ := Nat.lcm 10 (Nat.lcm 25 35)

theorem certain_number_divisibility : ∃ certain_number : ℕ, (∀ n : ℕ, 1 ≤ n ∧ n ≤ certain_number → n % lcm_10_25_35 = 0 → n ∈ {350, 700, 1050}) ∧ certain_number = 1399 :=
by
  -- Skip the proof with sorry
  sorry

end certain_number_divisibility_l828_828110


namespace sqrt_expression_eq_2_sqrt_expression_eq_2_sqrt_l828_828060

theorem sqrt_expression_eq_2 (a : ℝ) (h : 1 ≤ a ∧ a ≤ 2) :
  real.sqrt (a + 2 * real.sqrt (a - 1)) + real.sqrt (a - 2 * real.sqrt (a - 1)) = 2 :=
sorry

theorem sqrt_expression_eq_2_sqrt (a : ℝ) (h : 2 < a) :
  real.sqrt (a + 2 * real.sqrt (a - 1)) + real.sqrt (a - 2 * real.sqrt (a - 1)) = 2 * real.sqrt (a - 1) :=
sorry

end sqrt_expression_eq_2_sqrt_expression_eq_2_sqrt_l828_828060


namespace xiaogang_xiaoqiang_speeds_and_time_l828_828163

theorem xiaogang_xiaoqiang_speeds_and_time
  (x y : ℕ)
  (distance_meeting : 2 * x = 2 * y + 24)
  (xiaogang_time_after_meeting : 0.5 * x = d_x)
  (total_distance : 2 * x + (2 * x - 24) = D)
  (xiaogang_time_total : D / x = meeting_time + 0.5)
  (xiaoqiang_time_total : D / y = meeting_time + time_xiaoqiang_to_A) :
  x = 16 ∧ y = 4 ∧ time_xiaoqiang_to_A = 8 := by
sorry

end xiaogang_xiaoqiang_speeds_and_time_l828_828163


namespace length_of_first_train_l828_828126

theorem length_of_first_train
    (speed_first_train_kph : ℝ) (speed_second_train_kph : ℝ)
    (length_second_train_m : ℝ) (time_to_cross_sec : ℝ)
    (speed_first_train_kph = 60) (speed_second_train_kph = 40)
    (length_second_train_m = 150) (time_to_cross_sec = 10.439164866810657) :
    let speed_first_train_mps := speed_first_train_kph * (5 / 18),
        speed_second_train_mps := speed_second_train_kph * (5 / 18),
        relative_speed_mps := speed_first_train_mps + speed_second_train_mps,
        total_distance_m := time_to_cross_sec * relative_speed_mps,
        length_first_train_m := total_distance_m - length_second_train_m in
    length_first_train_m = 140 :=
by
  -- Proof can be filled in here
  sorry

end length_of_first_train_l828_828126


namespace cost_per_page_of_notebooks_l828_828015

-- Define the conditions
def notebooks : Nat := 2
def pages_per_notebook : Nat := 50
def cost_in_dollars : Nat := 5

-- Define the conversion constants
def dollars_to_cents : Nat := 100

-- Define the correct answer
def expected_cost_per_page := 5

-- State the theorem to prove the cost per page
theorem cost_per_page_of_notebooks :
  let total_pages := notebooks * pages_per_notebook
  let total_cost_in_cents := cost_in_dollars * dollars_to_cents
  let cost_per_page := total_cost_in_cents / total_pages
  cost_per_page = expected_cost_per_page :=
by
  -- Skip the proof with sorry
  sorry

end cost_per_page_of_notebooks_l828_828015


namespace XiaoMingAgeWhenFathersAgeIsFiveTimes_l828_828162

-- Define the conditions
def XiaoMingAgeCurrent : ℕ := 12
def FatherAgeCurrent : ℕ := 40

-- Prove the question given the conditions
theorem XiaoMingAgeWhenFathersAgeIsFiveTimes : 
  ∃ (x : ℕ), (FatherAgeCurrent - x) = 5 * x - XiaoMingAgeCurrent ∧ x = 7 := 
by
  use 7
  sorry

end XiaoMingAgeWhenFathersAgeIsFiveTimes_l828_828162


namespace circulation_ratio_l828_828077

variable (A : ℕ) -- Assuming A to be a natural number for simplicity

theorem circulation_ratio (h : ∀ t : ℕ, t = 1971 → t = 4 * A) : 4 / 13 = 4 / 13 := 
by
  sorry

end circulation_ratio_l828_828077


namespace angle_DHO_is_30_l828_828008

/-- Given triangle DOG with angle DGO equal to angle DOG, and angle DOG equal to 60 degrees,
if OH bisects angle DOG, then the angle DHO is 30 degrees. -/
theorem angle_DHO_is_30 
  (DOG : Type) [triangle DOG]
  (D G O H : DOG)
  (angle_DGO_eq_angle_DOG : ∠DGO = ∠DOG)
  (angle_DOG_is_60 : ∠DOG = 60)
  (OH_bisects_angle_DOG : bisects OH ∠DOG) :
  ∠DHO = 30 := 
sorry

end angle_DHO_is_30_l828_828008


namespace solve_for_x_l828_828066

theorem solve_for_x (x : ℝ) (h : log 3 ((4 * x + 12) / (6 * x - 4)) + log 3 ((6 * x - 4) / (2 * x - 3)) = 2) : x = 39 / 14 :=
sorry

end solve_for_x_l828_828066


namespace domain_g_l828_828136

noncomputable def g (x : ℝ) : ℝ := (x - 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_g : {x : ℝ | g x = (x - 3) / Real.sqrt (x^2 - 5 * x + 6)} = 
  {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_g_l828_828136


namespace acute_angle_at_3_15_l828_828146

/-- The hour and minute hands' angles and movements are defined as follows. -/
def hour_hand_angle (h m : Nat) : Real := (h % 12) * 30 + m * 0.5
def minute_hand_angle (m : Nat) : Real := (m % 60) * 6

/-- The condition that an acute angle is the smaller angle between hands. -/
def acute_angle (angle1 angle2 : Real) : Real := abs (angle1 - angle2)

/-- At 3:15, the acute angle between the hour and minute hands should be 7.5 degrees. -/
theorem acute_angle_at_3_15
    : acute_angle (hour_hand_angle 3 15) (minute_hand_angle 15) = 7.5 :=
by
    sorry

end acute_angle_at_3_15_l828_828146


namespace sum_perpendiculars_constant_l828_828041

-- Define the isosceles triangle and the moving point P
variables {A B C P X Y : Type} [metric_space P] [add_comm_monoid P] [has_mul ℝ P] [complete_space P]

-- Conditions for the isosceles triangle ABC with AB = AC
def is_isosceles_triangle (A B C : P) : Prop :=
  dist A B = dist A C

-- Positions of perpendiculars from P to AB and AC
def is_perpendicular (P X A B C : P) : Prop :=
  ∃ (PX : P), is_perpendicular_to_line P X A B

def is_perpendicular (P Y A B C : P) : Prop :=
  ∃ (PY : P), is_perpendicular_to_line P Y A C

-- Statement of the theorem
theorem sum_perpendiculars_constant (A B C P X Y : P) (h_isosceles : is_isosceles_triangle A B C) 
  (h1 : is_perpendicular P X A B C) (h2 : is_perpendicular P Y A B C):
  ∃ k : ℝ, ∀ (P : P), PX + PY = k :=
sorry

end sum_perpendiculars_constant_l828_828041


namespace oliver_seashells_l828_828049

noncomputable def seashells_monday : ℕ := 2
noncomputable def seashells_total : ℕ := 4
def seashells_tuesday : ℕ := seashells_total - seashells_monday

theorem oliver_seashells :
  seashells_tuesday = 2 :=
by
  unfold seashells_tuesday
  simp
  sorry

end oliver_seashells_l828_828049


namespace balance_balls_l828_828053

variable (R O B P : ℝ)

-- Conditions based on the problem statement
axiom h1 : 4 * R = 8 * B
axiom h2 : 3 * O = 7.5 * B
axiom h3 : 8 * B = 6 * P

-- The theorem we need to prove
theorem balance_balls : 5 * R + 3 * O + 3 * P = 21.5 * B :=
by 
  sorry

end balance_balls_l828_828053


namespace odd_square_minus_one_multiple_of_eight_l828_828030

theorem odd_square_minus_one_multiple_of_eight (a : ℤ) 
  (h₁ : a > 0) 
  (h₂ : a % 2 = 1) : 
  ∃ k : ℤ, a^2 - 1 = 8 * k :=
by
  sorry

end odd_square_minus_one_multiple_of_eight_l828_828030


namespace consecutive_cards_probability_l828_828112

theorem consecutive_cards_probability : 
  let S := {1, 2, 3, 4, 5}
  let total_pairs := Finset.card (Finset.pairs S)
  let consecutive_pairs := Finset.card (Finset.filter (λ p, abs (p.1 - p.2) = 1) (Finset.pairs S))
  total_pairs = 10 ∧ consecutive_pairs = 4 → consecutive_pairs / total_pairs = 0.4 :=
by
  intro S total_pairs consecutive_pairs h
  sorry

end consecutive_cards_probability_l828_828112


namespace parabola_hyperbola_tangent_l828_828099

-- Definitions of the parabola and hyperbola
def parabola (x : ℝ) : ℝ := x^2 + 4
def hyperbola (x y : ℝ) (m : ℝ) : Prop := y^2 - m*x^2 = 1

-- Tangency condition stating that the parabola and hyperbola are tangent implies m = 8 + 2*sqrt(15)
theorem parabola_hyperbola_tangent (m : ℝ) :
  (∀ x y : ℝ, parabola x = y → hyperbola x y m) → m = 8 + 2 * Real.sqrt 15 :=
by
  sorry

end parabola_hyperbola_tangent_l828_828099


namespace minimize_transportation_cost_l828_828079

theorem minimize_transportation_cost : ∀ (v : ℝ), (0 < v ∧ v ≤ 50) →
  (∀ v1 v2 : ℝ, 0 < v1 ∧ v1 < v2 ∧ v2 ≤ 50 →
    (let f := λ v, (10000 / v) + (4 * v) in f v1 < f v2)) ∧
  (let f := λ v, (10000 / v) + (4 * v) in (∀ v, (0 < v ∧ v ≤ 50) → f v ≥ f 50)) :=
by
  intro v hv
  split
  intro v1 v2 h
  sorry
  intro f hf
  sorry

end minimize_transportation_cost_l828_828079


namespace employed_females_percentage_l828_828004

theorem employed_females_percentage (total_population_percent employed_population_percent employed_males_percent : ℝ) :
  employed_population_percent = 70 → employed_males_percent = 21 →
  (employed_population_percent - employed_males_percent) / employed_population_percent * 100 = 70 :=
by
  -- Assume the total population percentage is 100%, which allows us to work directly with percentages.
  let employed_population_percent := 70
  let employed_males_percent := 21
  sorry

end employed_females_percentage_l828_828004


namespace acute_angle_at_315_equals_7_5_l828_828142

/-- The degrees in a full circle -/
def fullCircle := 360

/-- The number of hours on a clock -/
def hoursOnClock := 12

/-- The measure in degrees of the acute angle formed by the minute hand and the hour hand at 3:15 -/
def acuteAngleAt315 : ℝ :=
  let degreesPerHour := fullCircle / hoursOnClock
  let hourHandAt3 := degreesPerHour * 3
  let additionalDegrees := (15 / 60) * degreesPerHour
  let hourHandPosition := hourHandAt3 + additionalDegrees
  let minuteHandPosition := (15 / 60) * fullCircle
  abs (hourHandPosition - minuteHandPosition)

theorem acute_angle_at_315_equals_7_5 : acuteAngleAt315 = 7.5 := by
  sorry

end acute_angle_at_315_equals_7_5_l828_828142


namespace mart_income_percentage_l828_828169

variables (T J M : ℝ)

theorem mart_income_percentage (h1 : M = 1.60 * T) (h2 : T = 0.50 * J) :
  M = 0.80 * J :=
by
  sorry

end mart_income_percentage_l828_828169


namespace F_eq_arithmetic_mean_l828_828028

open Finset

variable {n r : ℕ}

-- Definition of F(n, r)
noncomputable def F (n r : ℕ) : ℚ :=
  let subsets := (powerset (range n.succ)).filter (λ s => card s = r ∧ s.nonempty) in
  (subsets.sum (λ s, (s.min' (by simp) : ℚ))) / (subsets.card : ℚ)

-- Theorem we want to prove
theorem F_eq_arithmetic_mean {n r : ℕ} (h₁ : 1 ≤ r) (h₂ : r ≤ n) : 
    F n r = (n + 1 : ℚ) / (r + 1 : ℚ) := 
  sorry

end F_eq_arithmetic_mean_l828_828028


namespace triangle_tangent_limit_l828_828005

theorem triangle_tangent_limit
  (A B C D : Point)
  (BC_len : BC.length = 6)
  (angle_C : ∠ A B C = Real.pi / 4)
  (midpoint_D : Midpoint D B C) :
  ∃ (f : ℝ → ℝ), 
  (∀ x, f x = (x - 3 * Real.sqrt 2) / (x + 3 * Real.sqrt 2)) ∧ 
  Tendsto f atTop (𝓝 1) :=
by
  sorry

end triangle_tangent_limit_l828_828005


namespace sqrt_x_minus_2_defined_l828_828104

theorem sqrt_x_minus_2_defined (x : ℝ) : (∃ y : ℝ, y = sqrt (x-2)) ↔ x ≥ 2 :=
by sorry

end sqrt_x_minus_2_defined_l828_828104


namespace intersecting_points_l828_828086

theorem intersecting_points:
  let f1 (x : ℝ) := x^3 - 4 * x + 3
  let f2 (x : ℝ) := -(x / 3) + 1
  let points : List (ℝ × ℝ) := List.map (λ x, (x, f2 x)) (Real.roots (λ x, 3 * x^3 - 11 * x + 6))
  ∃ (x1 x2 x3 : ℝ) (h1 : (x1, f2 x1) ∈ points) (h2 : (x2, f2 x2) ∈ points) (h3 : (x3, f2 x3) ∈ points), 
  (x1 + x2 + x3 = 0) ∧ (f2 x1 + f2 x2 + f2 x3 = 3) :=
by
  sorry

end intersecting_points_l828_828086


namespace eleven_power_five_mod_nine_l828_828069

theorem eleven_power_five_mod_nine : ∃ n : ℕ, (11^5 ≡ n [MOD 9]) ∧ (0 ≤ n ∧ n < 9) ∧ (n = 5) := 
  by 
    sorry

end eleven_power_five_mod_nine_l828_828069


namespace speeds_and_time_l828_828166

theorem speeds_and_time (x s : ℕ) (t : ℝ)
  (h1 : ∀ {t : ℝ}, t = 2 → x * t > s * t + 24)
  (h2 : ∀ {t : ℝ}, t = 0.5 → x * t = 8) :
  x = 16 ∧ s = 4 ∧ t = 8 :=
by {
  sorry
}

end speeds_and_time_l828_828166


namespace dice_sum_not_18_l828_828128

theorem dice_sum_not_18 (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6) (h2 : 1 ≤ d2 ∧ d2 ≤ 6) 
    (h3 : 1 ≤ d3 ∧ d3 ≤ 6) (h4 : 1 ≤ d4 ∧ d4 ≤ 6) (h_prod : d1 * d2 * d3 * d4 = 144) : 
    d1 + d2 + d3 + d4 ≠ 18 := 
sorry

end dice_sum_not_18_l828_828128


namespace contest_participants_l828_828096

/-- The organizing committee of a local contest wishes to arrange examination rooms 
    for the contestants. If each room is assigned 30 contestants, one room will have 26 contestants.
    If each room is assigned 26 contestants, one room will have 20 contestants,
    and this will require 9 more rooms than the previous arrangement. 
    Prove that the number of contestants participating in the exam from this region is 1736. -/
def exam_number_of_contestants : Prop :=
  ∃ (x y : ℕ), (30 * x - 4 = 26 * y - 6) ∧ (y = x + 9) ∧ (30 * x - 4 = 1736)

theorem contest_participants : exam_number_of_contestants :=
by
  use 58, 67
  repeat { split }
  · exact rfl
  · exact Nat.add_sub_cancel 58 9
  · exact rfl

end contest_participants_l828_828096


namespace integer_between_squares_l828_828026

theorem integer_between_squares (a b c d: ℝ) (h₀: 0 < a) (h₁: 0 < b) (h₂: 0 < c) (h₃: 0 < d) (h₄: c * d = 1) : 
  ∃ n : ℤ, ab ≤ n^2 ∧ n^2 ≤ (a + c) * (b + d) := 
by 
  sorry

end integer_between_squares_l828_828026


namespace domain_g_l828_828134

noncomputable def g (x : ℝ) : ℝ := (x - 3) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_g : {x : ℝ | g x = (x - 3) / Real.sqrt (x^2 - 5 * x + 6)} = 
  {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_g_l828_828134


namespace people_who_own_neither_l828_828108

theorem people_who_own_neither (total_people cat_owners cat_and_dog_owners dog_owners non_cat_dog_owners: ℕ)
        (h1: total_people = 522)
        (h2: 20 * cat_and_dog_owners = cat_owners)
        (h3: 7 * dog_owners = 10 * (dog_owners + cat_and_dog_owners))
        (h4: 2 * non_cat_dog_owners = (non_cat_dog_owners + dog_owners)):
    non_cat_dog_owners = 126 := 
by
  sorry

end people_who_own_neither_l828_828108


namespace triangle_side_sum_l828_828124

/-- In a triangle with angles 60 and 45 degrees, and opposite side of 60-degree angle = 12 units, 
the sum of the lengths of the other two sides is 41.0 (nearest tenth). -/
theorem triangle_side_sum (A B C : Type)
  {angleA : ℝ}
  {angleC : ℝ}
  {side_opposite_C : ℝ}
  (h_A : angleA = 45)
  (h_C : angleC = 60)
  (h_opposite_C : side_opposite_C = 12) :
  (side_opposite_C * sqrt 2 + 24) = 41.0 := 
sorry

end triangle_side_sum_l828_828124


namespace point_opposite_sides_line_l828_828097

theorem point_opposite_sides_line (a : ℝ) :
  (0 < a ∧ a < 2) ↔ (-(a : ℝ) * (1 + 1 - a) < 0) := 
begin
  sorry
end

end point_opposite_sides_line_l828_828097


namespace hyperbola_eccentricity_l828_828087

noncomputable def hyperbola_eccentricity_range (a b: ℝ) (h_a : a > 0) (h_b : b > 0) : Prop :=
  let e := (λ (c a : ℝ), c / a)
  ∃ c : ℝ, ∃ P : ℝ × ℝ, (c^2 = a^2 + b^2) ∧
  (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
  (| √(P.1 - c)^2 + P.2^2 | = 2 * | √(P.1 + c)^2 + P.2^2 |) ∧
  1 < e c a ∧ e c a ≤ 3

theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  hyperbola_eccentricity_range a b h_a h_b := by
  sorry

end hyperbola_eccentricity_l828_828087


namespace total_people_in_church_l828_828023

def c : ℕ := 80
def m : ℕ := 60
def f : ℕ := 60

theorem total_people_in_church : c + m + f = 200 :=
by
  sorry

end total_people_in_church_l828_828023


namespace negate_exists_statement_l828_828094

theorem negate_exists_statement : 
  (∃ x : ℝ, x^2 + x - 2 < 0) ↔ ¬ (∀ x : ℝ, x^2 + x - 2 ≥ 0) :=
by sorry

end negate_exists_statement_l828_828094


namespace smallest_n_and_k_l828_828072

variables {a₁ b₁ a₂ b₂ k n : ℕ}
variables (h_reduced₁ : Nat.coprime a₁ b₁) (h_reduced₂ : Nat.coprime a₂ b₂) (h_diff : a₂ * b₁ - a₁ * b₂ = 1)

theorem smallest_n_and_k 
(h_a1_b1_lt_k_n : ∀ k n, a₁ * n < b₁ * k)
(h_k_n_lt_a2_b2 : ∀ k n, b₂ * k < a₂ * n) :
  (n = b₁ + b₂ ∧ k = a₁ + a₂) :=
sorry

end smallest_n_and_k_l828_828072


namespace johns_initial_bench_press_weight_l828_828019

noncomputable def initialBenchPressWeight (currentWeight: ℝ) (injuryPercentage: ℝ) (trainingFactor: ℝ) :=
  (currentWeight / (injuryPercentage / 100 * trainingFactor))

theorem johns_initial_bench_press_weight:
  (initialBenchPressWeight 300 80 3) = 500 :=
by
  sorry

end johns_initial_bench_press_weight_l828_828019


namespace cost_per_page_l828_828013

theorem cost_per_page
  (num_notebooks : ℕ)
  (pages_per_notebook : ℕ)
  (total_dollars_paid : ℕ)
  (h1 : num_notebooks = 2)
  (h2 : pages_per_notebook = 50)
  (h3 : total_dollars_paid = 5) :
  (total_dollars_paid * 100) / (num_notebooks * pages_per_notebook) = 5 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end cost_per_page_l828_828013


namespace cost_per_page_of_notebooks_l828_828014

-- Define the conditions
def notebooks : Nat := 2
def pages_per_notebook : Nat := 50
def cost_in_dollars : Nat := 5

-- Define the conversion constants
def dollars_to_cents : Nat := 100

-- Define the correct answer
def expected_cost_per_page := 5

-- State the theorem to prove the cost per page
theorem cost_per_page_of_notebooks :
  let total_pages := notebooks * pages_per_notebook
  let total_cost_in_cents := cost_in_dollars * dollars_to_cents
  let cost_per_page := total_cost_in_cents / total_pages
  cost_per_page = expected_cost_per_page :=
by
  -- Skip the proof with sorry
  sorry

end cost_per_page_of_notebooks_l828_828014


namespace sum_of_diffs_is_10_l828_828065

-- Define the number of fruits each person has
def Sharon_plums : ℕ := 7
def Allan_plums : ℕ := 10
def Dave_oranges : ℕ := 12

-- Define the differences in the number of fruits
def diff_Sharon_Allan : ℕ := Allan_plums - Sharon_plums
def diff_Sharon_Dave : ℕ := Dave_oranges - Sharon_plums
def diff_Allan_Dave : ℕ := Dave_oranges - Allan_plums

-- Define the sum of these differences
def sum_of_diffs : ℕ := diff_Sharon_Allan + diff_Sharon_Dave + diff_Allan_Dave

-- State the theorem to be proved
theorem sum_of_diffs_is_10 : sum_of_diffs = 10 := by
  sorry

end sum_of_diffs_is_10_l828_828065


namespace tim_pencils_l828_828120

-- Problem statement: If x = 2 and z = 5, then y = z - x where y is the number of pencils Tim placed.
def pencils_problem (x y z : Nat) : Prop :=
  x = 2 ∧ z = 5 → y = z - x

theorem tim_pencils : pencils_problem 2 3 5 :=
by
  sorry

end tim_pencils_l828_828120


namespace ai_eq_i_l828_828129

namespace Problem

def gcd (m n : ℕ) : ℕ := Nat.gcd m n

def sequence_satisfies (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → gcd (a i) (a j) = gcd i j

theorem ai_eq_i (a : ℕ → ℕ) (h : sequence_satisfies a) : ∀ i : ℕ, a i = i :=
by
  sorry

end Problem

end ai_eq_i_l828_828129


namespace scientific_notation_of_one_point_six_million_l828_828050

-- Define the given number
def one_point_six_million : ℝ := 1.6 * 10^6

-- State the theorem to prove the equivalence
theorem scientific_notation_of_one_point_six_million :
  one_point_six_million = 1.6 * 10^6 :=
by
  sorry

end scientific_notation_of_one_point_six_million_l828_828050


namespace num_sets_l828_828095

theorem num_sets : 
  ∃ (A : set (set ℕ)), 
  {B | {1,3} ⊆ B ∧ B ⊆ {1,3,5,7,9}} = A ∧ 
  A.card = 7 := 
sorry

end num_sets_l828_828095
