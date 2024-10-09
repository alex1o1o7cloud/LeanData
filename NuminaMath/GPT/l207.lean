import Mathlib

namespace range_of_m_l207_20771

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - 4 * x ≥ m) → m ≤ -3 := 
sorry

end range_of_m_l207_20771


namespace arrangement_correct_l207_20730

def A := 4
def B := 1
def C := 2
def D := 5
def E := 6
def F := 3

def sum1 := A + B + C
def sum2 := A + D + F
def sum3 := B + E + D
def sum4 := C + F + E
def sum5 := A + E + F
def sum6 := B + D + C
def sum7 := B + C + F

theorem arrangement_correct :
  sum1 = 15 ∧ sum2 = 15 ∧ sum3 = 15 ∧ sum4 = 15 ∧ sum5 = 15 ∧ sum6 = 15 ∧ sum7 = 15 := 
by
  unfold sum1 sum2 sum3 sum4 sum5 sum6 sum7 
  unfold A B C D E F
  sorry

end arrangement_correct_l207_20730


namespace min_z_value_l207_20777

variable (x y z : ℝ)

theorem min_z_value (h1 : x - 1 ≥ 0) (h2 : 2 * x - y - 1 ≤ 0) (h3 : x + y - 3 ≤ 0) :
  z = x - y → z = -1 :=
by sorry

end min_z_value_l207_20777


namespace smallest_total_students_l207_20798

theorem smallest_total_students :
  (∃ (n : ℕ), 4 * n + (n + 2) > 50 ∧ ∀ m, 4 * m + (m + 2) > 50 → m ≥ n) → 4 * 10 + (10 + 2) = 52 :=
by
  sorry

end smallest_total_students_l207_20798


namespace min_blocks_to_remove_l207_20705

theorem min_blocks_to_remove (n : ℕ) (h₁ : n = 59) : ∃ k, ∃ m, (m*m*m ≤ n ∧ n < (m+1)*(m+1)*(m+1)) ∧ k = n - m*m*m ∧ k = 32 :=
by {
  sorry
}

end min_blocks_to_remove_l207_20705


namespace interest_rate_second_part_l207_20710

noncomputable def P1 : ℝ := 2799.9999999999995
noncomputable def P2 : ℝ := 4000 - P1
noncomputable def Interest1 : ℝ := P1 * (3 / 100)
noncomputable def TotalInterest : ℝ := 144
noncomputable def Interest2 : ℝ := TotalInterest - Interest1

theorem interest_rate_second_part :
  ∃ r : ℝ, Interest2 = P2 * (r / 100) ∧ r = 5 :=
by
  sorry

end interest_rate_second_part_l207_20710


namespace quadratic_has_one_real_root_l207_20747

theorem quadratic_has_one_real_root (m : ℝ) (h : (6 * m)^2 - 4 * 1 * 4 * m = 0) : m = 4 / 9 :=
by sorry

end quadratic_has_one_real_root_l207_20747


namespace distance_from_point_to_focus_l207_20783

noncomputable def point_on_parabola (P : ℝ × ℝ) (y : ℝ) : Prop :=
  y^2 = 16 * P.1 ∧ (P.2 = y ∨ P.2 = -y)

noncomputable def parabola_focus : ℝ × ℝ :=
  (4, 0)

theorem distance_from_point_to_focus
  (P : ℝ × ℝ) (y : ℝ)
  (h1 : point_on_parabola P y)
  (h2 : dist P (0, P.2) = 12) :
  dist P parabola_focus = 13 :=
sorry

end distance_from_point_to_focus_l207_20783


namespace ticket_cost_l207_20702

theorem ticket_cost 
  (V G : ℕ)
  (h1 : V + G = 320)
  (h2 : V = G - 212) :
  40 * V + 15 * G = 6150 := 
by
  sorry

end ticket_cost_l207_20702


namespace maximize_profit_l207_20729

noncomputable def I (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 2 then 2 * (x - 1) * Real.exp (x - 2) + 2
  else if h' : 2 < x ∧ x ≤ 50 then 440 + 3050 / x - 9000 / x^2
  else 0 -- default case for Lean to satisfy definition

noncomputable def P (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 2 then 2 * x * (x - 1) * Real.exp (x - 2) - 448 * x - 180
  else if h' : 2 < x ∧ x ≤ 50 then -10 * x - 9000 / x + 2870
  else 0 -- default case for Lean to satisfy definition

theorem maximize_profit :
  (∀ x : ℝ, 0 < x ∧ x ≤ 50 → P x ≤ 2270) ∧ P 30 = 2270 :=
by
  sorry

end maximize_profit_l207_20729


namespace a_9_value_l207_20760

-- Define the sequence and its sum of the first n terms
def S (n : ℕ) : ℕ := n^2

-- Define the terms of the sequence
def a (n : ℕ) : ℕ := if n = 1 then S 1 else S n - S (n - 1)

-- The main statement to be proved
theorem a_9_value : a 9 = 17 :=
by
  sorry

end a_9_value_l207_20760


namespace exists_mutual_shooters_l207_20708

theorem exists_mutual_shooters (n : ℕ) (h : 0 ≤ n) (d : Fin (2 * n + 1) → Fin (2 * n + 1) → ℝ)
  (hdistinct : ∀ i j k l : Fin (2 * n + 1), i ≠ j → k ≠ l → d i j ≠ d k l)
  (hc : ∀ i : Fin (2 * n + 1), ∃ j : Fin (2 * n + 1), i ≠ j ∧ (∀ k : Fin (2 * n + 1), k ≠ j → d i j < d i k)) :
  ∃ i j : Fin (2 * n + 1), i ≠ j ∧
  (∀ k : Fin (2 * n + 1), k ≠ j → d i j < d i k) ∧
  (∀ k : Fin (2 * n + 1), k ≠ i → d j i < d j k) :=
by
  sorry

end exists_mutual_shooters_l207_20708


namespace distinct_int_divisible_by_12_l207_20732

variable {a b c d : ℤ}

theorem distinct_int_divisible_by_12 (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) :=
by
  sorry

end distinct_int_divisible_by_12_l207_20732


namespace roberto_outfit_combinations_l207_20785

-- Define the components of the problem
def trousers_count : ℕ := 5
def shirts_count : ℕ := 7
def jackets_count : ℕ := 4
def disallowed_combinations : ℕ := 7

-- Define the requirements
theorem roberto_outfit_combinations :
  (trousers_count * shirts_count * jackets_count) - disallowed_combinations = 133 := by
  sorry

end roberto_outfit_combinations_l207_20785


namespace ellipse_slope_product_l207_20793

theorem ellipse_slope_product (x₀ y₀ : ℝ) (hp : x₀^2 / 4 + y₀^2 / 3 = 1) :
  (y₀ / (x₀ + 2)) * (y₀ / (x₀ - 2)) = -3 / 4 :=
by
  -- The proof is omitted.
  sorry

end ellipse_slope_product_l207_20793


namespace number_B_expression_l207_20770

theorem number_B_expression (A B : ℝ) (h : A = B - (4/5) * B) : B = (A + B) / (4 / 5) :=
sorry

end number_B_expression_l207_20770


namespace coin_problem_exists_l207_20789

theorem coin_problem_exists (n : ℕ) : 
  (∃ n, n % 8 = 6 ∧ n % 7 = 5 ∧ (∀ m, (m % 8 = 6 ∧ m % 7 = 5) → n ≤ m)) →
  (∃ n, (n % 8 = 6) ∧ (n % 7 = 5) ∧ (n % 9 = 0)) :=
by
  sorry

end coin_problem_exists_l207_20789


namespace edith_novel_count_l207_20791

-- Definitions based on conditions
variables (N W : ℕ)

-- Conditions from the problem
def condition1 : Prop := N = W / 2
def condition2 : Prop := N + W = 240

-- Target statement
theorem edith_novel_count (N W : ℕ) (h1 : N = W / 2) (h2 : N + W = 240) : N = 80 :=
by
  sorry

end edith_novel_count_l207_20791


namespace shared_property_l207_20780

-- Definitions of the shapes
structure Parallelogram where
  sides_equal    : Bool -- Parallelograms have opposite sides equal but not necessarily all four.

structure Rectangle where
  sides_equal    : Bool -- Rectangles have opposite sides equal.
  diagonals_equal: Bool

structure Rhombus where
  sides_equal: Bool -- Rhombuses have all sides equal.
  diagonals_perpendicular: Bool -- Diagonals of a rhombus are perpendicular.

structure Square where
  sides_equal: Bool -- Squares have all sides equal.
  diagonals_perpendicular: Bool -- Diagonals of a square are perpendicular.
  diagonals_equal: Bool -- Diagonals of a square are equal in length.

-- Definitions of properties
def all_sides_equal (p1 p2 p3 p4 : Parallelogram) := p1.sides_equal ∧ p2.sides_equal ∧ p3.sides_equal ∧ p4.sides_equal
def diagonals_equal (r1 r2 r3 : Rectangle) (s1 s2 : Square) := r1.diagonals_equal ∧ r2.diagonals_equal ∧ s1.diagonals_equal ∧ s2.diagonals_equal
def diagonals_perpendicular (r1 : Rhombus) (s1 s2 : Square) := r1.diagonals_perpendicular ∧ s1.diagonals_perpendicular ∧ s2.diagonals_perpendicular
def diagonals_bisect_each_other (p1 p2 p3 p4 : Parallelogram) (r1 : Rectangle) (r2 : Rhombus) (s1 s2 : Square) := True -- All these shapes have diagonals that bisect each other.

-- The statement we need to prove
theorem shared_property (p1 p2 p3 p4 : Parallelogram) (r1 r2 : Rectangle) (r3 : Rhombus) (s1 s2 : Square) : 
  (diagonals_bisect_each_other p1 p2 p3 p4 r1 r3 s1 s2) :=
by
  sorry

end shared_property_l207_20780


namespace minimum_area_triangle_ABC_l207_20754

-- Define the vertices of the triangle
def A : ℤ × ℤ := (0,0)
def B : ℤ × ℤ := (30,18)

-- Define a function to calculate the area of the triangle using the Shoelace formula
def area_of_triangle (A B C : ℤ × ℤ) : ℤ := 15 * (C.2).natAbs

-- State the theorem
theorem minimum_area_triangle_ABC : 
  ∀ C : ℤ × ℤ, C ≠ (0,0) → area_of_triangle A B C ≥ 15 :=
by
  sorry -- Skip the proof

end minimum_area_triangle_ABC_l207_20754


namespace max_rectangle_area_with_prime_dimension_l207_20757

theorem max_rectangle_area_with_prime_dimension :
  ∃ (l w : ℕ), 2 * (l + w) = 120 ∧ (Prime l ∨ Prime w) ∧ l * w = 899 :=
by
  sorry

end max_rectangle_area_with_prime_dimension_l207_20757


namespace greatest_x_value_l207_20727

noncomputable def greatest_possible_value (x : ℕ) : ℕ :=
  if (x % 5 = 0) ∧ (x^3 < 3375) then x else 0

theorem greatest_x_value :
  ∃ x, greatest_possible_value x = 10 ∧ (∀ y, ((y % 5 = 0) ∧ (y^3 < 3375)) → y ≤ x) :=
by
  sorry

end greatest_x_value_l207_20727


namespace delta_x_not_zero_l207_20763

noncomputable def average_rate_of_change (f : ℝ → ℝ) (x : ℝ) (delta_x : ℝ) : ℝ :=
  (f (x + delta_x) - f x) / delta_x

theorem delta_x_not_zero (f : ℝ → ℝ) (x delta_x : ℝ) (h_neq : delta_x ≠ 0):
  average_rate_of_change f x delta_x ≠ 0 := 
by
  sorry

end delta_x_not_zero_l207_20763


namespace proof_problem_l207_20733

-- Conditions
def op1 := (15 + 3) / (8 - 2) = 3
def op2 := (9 + 4) / (14 - 7)

-- Statement
theorem proof_problem : op1 → op2 = 13 / 7 :=
by 
  intro h
  unfold op2
  sorry

end proof_problem_l207_20733


namespace no_integer_solutions_l207_20762

theorem no_integer_solutions :
  ∀ (m n : ℤ), (m^3 + 4 * m^2 + 3 * m ≠ 8 * n^3 + 12 * n^2 + 6 * n + 1) := by
  sorry

end no_integer_solutions_l207_20762


namespace solution_set_of_inequality_l207_20758

theorem solution_set_of_inequality (x : ℝ) (h : 2 * x + 3 ≤ 1) : x ≤ -1 :=
sorry

end solution_set_of_inequality_l207_20758


namespace number_of_ways_to_win_championships_l207_20792

-- Definitions for the problem
def num_athletes := 5
def num_events := 3

-- Proof statement
theorem number_of_ways_to_win_championships : 
  (num_athletes ^ num_events) = 125 := 
by 
  sorry

end number_of_ways_to_win_championships_l207_20792


namespace minimum_omega_l207_20749

/-- Given function f and its properties, determine the minimum valid ω. -/
theorem minimum_omega {f : ℝ → ℝ} 
  (Hf : ∀ x : ℝ, f x = (1 / 2) * Real.cos (ω * x + φ) + 1)
  (Hsymmetry : ∃ k : ℤ, ω * (π / 3) + φ = k * π)
  (Hvalue : ∃ n : ℤ, f (π / 12) = 1 ∧ ω * (π / 12) + φ = n * π + π / 2)
  (Hpos : ω > 0) : ω = 2 := 
sorry

end minimum_omega_l207_20749


namespace tank_full_after_50_minutes_l207_20796

-- Define the conditions as constants
def tank_capacity : ℕ := 850
def pipe_a_rate : ℕ := 40
def pipe_b_rate : ℕ := 30
def pipe_c_rate : ℕ := 20
def cycle_duration : ℕ := 3  -- duration of each cycle in minutes
def net_water_per_cycle : ℕ := pipe_a_rate + pipe_b_rate - pipe_c_rate  -- net liters added per cycle

-- Define the statement to be proved: the tank will be full at exactly 50 minutes
theorem tank_full_after_50_minutes :
  ∀ minutes_elapsed : ℕ, (minutes_elapsed = 50) →
  ((minutes_elapsed / cycle_duration) * net_water_per_cycle = tank_capacity - pipe_c_rate) :=
sorry

end tank_full_after_50_minutes_l207_20796


namespace goods_train_speed_l207_20704

theorem goods_train_speed (train_length platform_length : ℝ) (time_sec : ℝ) : 
  train_length = 270.0416 ∧ platform_length = 250 ∧ time_sec = 26 → 
  (train_length + platform_length) / time_sec * 3.6 = 72.00576 :=
by
  sorry

end goods_train_speed_l207_20704


namespace angies_age_l207_20756

variable (A : ℝ)

theorem angies_age (h : 2 * A + 4 = 20) : A = 8 :=
sorry

end angies_age_l207_20756


namespace calc_a_squared_plus_b_squared_and_ab_l207_20736

theorem calc_a_squared_plus_b_squared_and_ab (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 3) :
  a^2 + b^2 = 5 ∧ a * b = 1 :=
by
  sorry

end calc_a_squared_plus_b_squared_and_ab_l207_20736


namespace purchase_price_is_60_l207_20714

variable (P S D : ℝ)
variable (GP : ℝ := 4)

theorem purchase_price_is_60
  (h1 : S = P + 0.25 * S)
  (h2 : D = 0.80 * S)
  (h3 : GP = D - P) :
  P = 60 :=
by
  sorry

end purchase_price_is_60_l207_20714


namespace parabola_line_intersection_l207_20716

theorem parabola_line_intersection :
  let a := (3 + Real.sqrt 11) / 2
  let b := (3 - Real.sqrt 11) / 2
  let p1 := (a, (9 + Real.sqrt 11) / 2)
  let p2 := (b, (9 - Real.sqrt 11) / 2)
  (3 * a^2 - 9 * a + 4 = (9 + Real.sqrt 11) / 2) ∧
  (-a^2 + 3 * a + 6 = (9 + Real.sqrt 11) / 2) ∧
  ((9 + Real.sqrt 11) / 2 = a + 3) ∧
  (3 * b^2 - 9 * b + 4 = (9 - Real.sqrt 11) / 2) ∧
  (-b^2 + 3 * b + 6 = (9 - Real.sqrt 11) / 2) ∧
  ((9 - Real.sqrt 11) / 2 = b + 3) :=
by
  sorry

end parabola_line_intersection_l207_20716


namespace sides_of_triangle_l207_20738

variable (a b c : ℝ)

theorem sides_of_triangle (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ineq : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) :=
  sorry

end sides_of_triangle_l207_20738


namespace triangle_area_ratio_l207_20768

theorem triangle_area_ratio :
  let base_jihye := 3
  let height_jihye := 2
  let base_donggeon := 3
  let height_donggeon := 6.02
  let area_jihye := (base_jihye * height_jihye) / 2
  let area_donggeon := (base_donggeon * height_donggeon) / 2
  (area_donggeon / area_jihye) = 3.01 :=
by
  sorry

end triangle_area_ratio_l207_20768


namespace right_triangle_third_side_product_l207_20724

theorem right_triangle_third_side_product :
  ∀ (a b : ℝ), (a = 6 ∧ b = 8 ∧ (a^2 + b^2 = c^2 ∨ a^2 = b^2 - c^2)) →
  (a * b = 53.0) :=
by
  intros a b h
  sorry

end right_triangle_third_side_product_l207_20724


namespace kate_spent_on_mouse_l207_20787

theorem kate_spent_on_mouse :
  let march := 27
  let april := 13
  let may := 28
  let saved := march + april + may
  let keyboard := 49
  let left := 14
  saved - left - keyboard = 5 :=
by
  let march := 27
  let april := 13
  let may := 28
  let saved := march + april + may
  let keyboard := 49
  let left := 14
  show saved - left - keyboard = 5
  sorry

end kate_spent_on_mouse_l207_20787


namespace find_x_l207_20734

theorem find_x (x y : ℤ) (h₁ : x + 3 * y = 10) (h₂ : y = 3) : x = 1 := 
by
  sorry

end find_x_l207_20734


namespace smallest_class_size_l207_20781

variable (x : ℕ) 

theorem smallest_class_size
  (h1 : 5 * x + 2 > 40)
  (h2 : x ≥ 0) : 
  5 * 8 + 2 = 42 :=
by sorry

end smallest_class_size_l207_20781


namespace find_u_plus_v_l207_20719

theorem find_u_plus_v (u v : ℚ) (h1 : 3 * u - 7 * v = 17) (h2 : 5 * u + 3 * v = 1) : 
  u + v = - 6 / 11 :=
  sorry

end find_u_plus_v_l207_20719


namespace tan_range_l207_20722

theorem tan_range :
  ∀ (x : ℝ), -Real.pi / 4 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ Real.pi / 4 → -1 ≤ Real.tan x ∧ Real.tan x < 0 ∨ 0 < Real.tan x ∧ Real.tan x ≤ 1 :=
by
  sorry

end tan_range_l207_20722


namespace geometric_sequence_sum_l207_20721

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h1 : a 1 + a 3 = 20)
  (h2 : a 2 + a 4 = 40)
  :
  a 3 + a 5 = 80 :=
sorry

end geometric_sequence_sum_l207_20721


namespace find_s_of_2_l207_20776

-- Define t and s as per the given conditions
def t (x : ℚ) : ℚ := 4 * x - 9
def s (x : ℚ) : ℚ := x^2 + 4 * x - 5

-- The theorem that we need to prove
theorem find_s_of_2 : s 2 = 217 / 16 := by
  sorry

end find_s_of_2_l207_20776


namespace problem1_problem2_problem3_problem4_l207_20703

theorem problem1 : (-8) + 10 - 2 + (-1) = -1 := 
by
  sorry

theorem problem2 : 12 - 7 * (-4) + 8 / (-2) = 36 := 
by 
  sorry

theorem problem3 : ( (1/2) + (1/3) - (1/6) ) / (-1/18) = -12 := 
by 
  sorry

theorem problem4 : - 1 ^ 4 - (1 + 0.5) * (1/3) * (-4) ^ 2 = -33 / 32 := 
by 
  sorry


end problem1_problem2_problem3_problem4_l207_20703


namespace smaller_cone_volume_ratio_l207_20742

theorem smaller_cone_volume_ratio :
  let r := 12
  let theta1 := 120
  let theta2 := 240
  let arc_length_small := (theta1 / 360) * (2 * Real.pi * r)
  let arc_length_large := (theta2 / 360) * (2 * Real.pi * r)
  let r1 := arc_length_small / (2 * Real.pi)
  let r2 := arc_length_large / (2 * Real.pi)
  let l := r
  let h1 := Real.sqrt (l^2 - r1^2)
  let h2 := Real.sqrt (l^2 - r2^2)
  let V1 := (1 / 3) * Real.pi * r1^2 * h1
  let V2 := (1 / 3) * Real.pi * r2^2 * h2
  V1 / V2 = Real.sqrt 10 / 10 := sorry

end smaller_cone_volume_ratio_l207_20742


namespace present_age_ratio_l207_20753

-- Define the conditions as functions in Lean.
def age_difference (M R : ℝ) : Prop := M - R = 7.5
def future_age_ratio (M R : ℝ) : Prop := (R + 10) / (M + 10) = 2 / 3

-- Define the goal as a proof problem in Lean.
theorem present_age_ratio (M R : ℝ) 
  (h1 : age_difference M R) 
  (h2 : future_age_ratio M R) : 
  R / M = 2 / 5 := 
by 
  sorry  -- Proof to be completed

end present_age_ratio_l207_20753


namespace problem1_l207_20779

theorem problem1
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)
  (h₁ : (3*x - 2)^(6) = a₀ + a₁ * (2*x - 1) + a₂ * (2*x - 1)^2 + a₃ * (2*x - 1)^3 + a₄ * (2*x - 1)^4 + a₅ * (2*x - 1)^5 + a₆ * (2*x - 1)^6)
  (h₂ : a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 1)
  (h₃ : a₀ - a₁ + a₂ - a₃ + a₄ - a₅ + a₆ = 64) :
  (a₁ + a₃ + a₅) / (a₀ + a₂ + a₄ + a₆) = -63 / 65 := by
  sorry

end problem1_l207_20779


namespace order_of_arrival_l207_20774

noncomputable def position_order (P S O E R : ℕ) : Prop :=
  S = O - 10 ∧ S = R + 25 ∧ R = E - 5 ∧ E = P - 25

theorem order_of_arrival (P S O E R : ℕ) (h : position_order P S O E R) :
  P > (S + 10) ∧ S > (O - 10) ∧ O > (E + 5) ∧ E > R :=
sorry

end order_of_arrival_l207_20774


namespace range_of_k_l207_20746

theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, k * x ^ 2 + 2 * k * x + 3 ≠ 0) ↔ (0 ≤ k ∧ k < 3) :=
by sorry

end range_of_k_l207_20746


namespace minimum_m_value_l207_20712

theorem minimum_m_value (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_eq : 24 * m = n^4) : m = 54 := sorry

end minimum_m_value_l207_20712


namespace series_inequality_l207_20711

open BigOperators

theorem series_inequality :
  (∑ k in Finset.range 2012, (1 / (((k + 1) * Real.sqrt k) + (k * Real.sqrt (k + 1))))) > 0.97 :=
sorry

end series_inequality_l207_20711


namespace repeating_decimal_division_l207_20700

-- Define x and y as the repeating decimals.
noncomputable def x : ℚ := 84 / 99
noncomputable def y : ℚ := 21 / 99

-- Proof statement of the equivalence.
theorem repeating_decimal_division : (x / y) = 4 := by
  sorry

end repeating_decimal_division_l207_20700


namespace min_value_of_quadratic_l207_20772

theorem min_value_of_quadratic :
  (∀ x : ℝ, 3 * x^2 - 12 * x + 908 ≥ 896) ∧ (∃ x : ℝ, 3 * x^2 - 12 * x + 908 = 896) :=
by
  sorry

end min_value_of_quadratic_l207_20772


namespace equation_is_correct_l207_20794

-- Define the numbers
def n1 : ℕ := 2
def n2 : ℕ := 2
def n3 : ℕ := 11
def n4 : ℕ := 11

-- Define the mathematical expression and the target result
def expression : ℚ := (n1 + n2 / n3) * n4
def target_result : ℚ := 24

-- The proof statement
theorem equation_is_correct : expression = target_result := by
  sorry

end equation_is_correct_l207_20794


namespace inequality_holds_l207_20739

theorem inequality_holds (x : ℝ) (n : ℕ) (hn : 0 < n) : 
  Real.sin (2 * x)^n + (Real.sin x^n - Real.cos x^n)^2 ≤ 1 := 
sorry

end inequality_holds_l207_20739


namespace find_a_plus_b_l207_20741

theorem find_a_plus_b (x a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hx : x = a + Real.sqrt b)
  (hxeq : x^2 + 5*x + 5/x + 1/(x^2) = 42) : a + b = 5 :=
sorry

end find_a_plus_b_l207_20741


namespace quadratic_root_shift_l207_20767

theorem quadratic_root_shift (A B p : ℤ) (α β : ℤ) 
  (h1 : ∀ x, x^2 + p * x + 19 = 0 → x = α + 1 ∨ x = β + 1)
  (h2 : ∀ x, x^2 - A * x + B = 0 → x = α ∨ x = β)
  (h3 : α + β = A)
  (h4 : α * β = B) :
  A + B = 18 := 
sorry

end quadratic_root_shift_l207_20767


namespace fraction_of_raisins_in_mixture_l207_20797

def cost_of_raisins (R : ℝ) := 3 * R
def cost_of_nuts (R : ℝ) := 3 * (3 * R)
def total_cost (R : ℝ) := cost_of_raisins R + cost_of_nuts R

theorem fraction_of_raisins_in_mixture (R : ℝ) (hR_pos : R > 0) : 
  cost_of_raisins R / total_cost R = 1 / 4 :=
by
  sorry

end fraction_of_raisins_in_mixture_l207_20797


namespace count_perfect_square_factors_l207_20761

theorem count_perfect_square_factors : 
  let n := (2^10) * (3^12) * (5^15) * (7^7)
  ∃ (count : ℕ), count = 1344 ∧
    (∀ (a b c d : ℕ), 0 ≤ a ∧ a ≤ 10 ∧ 0 ≤ b ∧ b ≤ 12 ∧ 0 ≤ c ∧ c ≤ 15 ∧ 0 ≤ d ∧ d ≤ 7 →
      ((a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ (d % 2 = 0) →
        ∃ (k : ℕ), (2^a * 3^b * 5^c * 7^d) = k ∧ k ∣ n)) :=
by
  sorry

end count_perfect_square_factors_l207_20761


namespace boy_running_time_l207_20744

theorem boy_running_time (s : ℝ) (v : ℝ) (h1 : s = 35) (h2 : v = 9) : 
  (4 * s) / (v * 1000 / 3600) = 56 := by
  sorry

end boy_running_time_l207_20744


namespace exists_x_eq_1_l207_20735

theorem exists_x_eq_1 (x y z t : ℕ) (h : x + y + z + t = 10) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  ∃ x, x = 1 :=
sorry

end exists_x_eq_1_l207_20735


namespace g_neg_2_eq_3_l207_20726

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem g_neg_2_eq_3 : g (-2) = 3 :=
by
  sorry

end g_neg_2_eq_3_l207_20726


namespace find_deepaks_age_l207_20701

variable (R D : ℕ)

theorem find_deepaks_age
  (h1 : R / D = 4 / 3)
  (h2 : R + 2 = 26) :
  D = 18 := by
  sorry

end find_deepaks_age_l207_20701


namespace bag_of_chips_weight_l207_20782

theorem bag_of_chips_weight (c : ℕ) : 
  (∀ (t : ℕ), t = 9) → 
  (∀ (b : ℕ), b = 6) → 
  (∀ (x : ℕ), x = 4 * 6) → 
  (21 * 16 = 336) →
  (336 - 24 * 9 = 6 * c) → 
  c = 20 :=
by
  intros ht hb hx h_weight_total h_weight_chips
  sorry

end bag_of_chips_weight_l207_20782


namespace box_weight_difference_l207_20755

theorem box_weight_difference:
  let w1 := 2
  let w2 := 3
  let w3 := 13
  let w4 := 7
  let w5 := 10
  (max (max (max (max w1 w2) w3) w4) w5) - (min (min (min (min w1 w2) w3) w4) w5) = 11 :=
by
  sorry

end box_weight_difference_l207_20755


namespace sufficient_but_not_necessary_condition_l207_20766

theorem sufficient_but_not_necessary_condition (k : ℝ) : 
  (k = 1 → ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x - y + k = 0) ∧ 
  ¬(∀ k : ℝ, ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x - y + k = 0 → k = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l207_20766


namespace area_of_region_l207_20718

theorem area_of_region : ∃ A, (∀ x y : ℝ, x^2 + y^2 + 6*x - 4*y = 12 → A = 25 * Real.pi) :=
by
  -- Completing the square and identifying the circle
  -- We verify that the given equation represents a circle
  existsi (25 * Real.pi)
  intros x y h
  sorry

end area_of_region_l207_20718


namespace min_value_PQ_l207_20786

variable (t : ℝ) (x y : ℝ)

-- Parametric equations of line l
def line_l : Prop := (x = 4 * t - 1) ∧ (y = 3 * t - 3 / 2)

-- Polar equation of circle C
def polar_eq_circle_c (ρ θ : ℝ) : Prop :=
  ρ^2 = 2 * Real.sqrt 2 * ρ * Real.sin (θ - Real.pi / 4)

-- General equation of line l
def general_eq_line_l (x y : ℝ) : Prop := 3 * x - 4 * y = 3

-- Rectangular equation of circle C
def rectangular_eq_circle_c (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 1)^2 = 2

-- Definition of the condition where P is on line l
def p_on_line_l (x y : ℝ) : Prop := ∃ t : ℝ, line_l t x y

-- Minimum value of |PQ|
theorem min_value_PQ :
  p_on_line_l x y →
  general_eq_line_l x y →
  rectangular_eq_circle_c x y →
  ∃ d : ℝ, d = Real.sqrt 2 :=
by intros; sorry

end min_value_PQ_l207_20786


namespace nonneg_solution_iff_m_range_l207_20725

theorem nonneg_solution_iff_m_range (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 1) + 3 / (1 - x) = 1)) ↔ (m ≥ 2 ∧ m ≠ 3) :=
sorry

end nonneg_solution_iff_m_range_l207_20725


namespace eddie_rate_l207_20759

variables (hours_sam hours_eddie rate_sam total_crates rate_eddie : ℕ)

def sam_conditions :=
  hours_sam = 6 ∧ rate_sam = 60

def eddie_conditions :=
  hours_eddie = 4 ∧ total_crates = hours_sam * rate_sam

theorem eddie_rate (hs : sam_conditions hours_sam rate_sam)
                   (he : eddie_conditions hours_sam hours_eddie rate_sam total_crates) :
  rate_eddie = 90 :=
by sorry

end eddie_rate_l207_20759


namespace seq_nat_eq_n_l207_20750

theorem seq_nat_eq_n (a : ℕ → ℕ) (h_inc : ∀ n, a n < a (n + 1))
  (h_le : ∀ n, a n ≤ n + 2020)
  (h_div : ∀ n, a (n + 1) ∣ (n^3 * a n - 1)) :
  ∀ n, a n = n :=
by
  sorry

end seq_nat_eq_n_l207_20750


namespace rectangle_area_theorem_l207_20743

def rectangle_area (d : ℝ) (area : ℝ) : Prop :=
  ∃ w : ℝ, 0 < w ∧ 9 * w^2 + w^2 = d^2 ∧ area = 3 * w^2

theorem rectangle_area_theorem (d : ℝ) : rectangle_area d (3 * d^2 / 10) :=
sorry

end rectangle_area_theorem_l207_20743


namespace green_fish_count_l207_20752

theorem green_fish_count (B O G : ℕ) (h1 : B = (2 / 5) * 200)
  (h2 : O = 2 * B - 30) (h3 : G = (3 / 2) * O) (h4 : B + O + G = 200) : 
  G = 195 :=
by
  sorry

end green_fish_count_l207_20752


namespace complement_union_M_N_eq_16_l207_20799

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subsets M and N
def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {4, 5}

-- Define the union of M and N
def unionMN : Set ℕ := M ∪ N

-- Define the complement of M ∪ N in U
def complementUnionMN : Set ℕ := U \ unionMN

-- State the theorem that the complement is {1, 6}
theorem complement_union_M_N_eq_16 : complementUnionMN = {1, 6} := by
  sorry

end complement_union_M_N_eq_16_l207_20799


namespace cos_triple_angle_l207_20748

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3 * θ) = -117/125 := 
by
  sorry

end cos_triple_angle_l207_20748


namespace problem_sum_of_pairwise_prime_product_l207_20784

theorem problem_sum_of_pairwise_prime_product:
  ∃ a b c d: ℕ, a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧
  a * b * c * d = 288000 ∧
  gcd a b = 1 ∧ gcd a c = 1 ∧ gcd a d = 1 ∧
  gcd b c = 1 ∧ gcd b d = 1 ∧ gcd c d = 1 ∧
  a + b + c + d = 390 :=
sorry

end problem_sum_of_pairwise_prime_product_l207_20784


namespace compressor_station_distances_compressor_station_distances_when_a_is_30_l207_20707

theorem compressor_station_distances (a : ℝ) (h : 0 < a ∧ a < 60) :
  ∃ x y z : ℝ, x + y = 3 * z ∧ z + y = x + a ∧ x + z = 60 :=
sorry

theorem compressor_station_distances_when_a_is_30 :
  ∃ x y z : ℝ, 
  (x + y = 3 * z) ∧ (z + y = x + 30) ∧ (x + z = 60) ∧ 
  (x = 35) ∧ (y = 40) ∧ (z = 25) :=
sorry

end compressor_station_distances_compressor_station_distances_when_a_is_30_l207_20707


namespace number_of_team_members_l207_20790

theorem number_of_team_members (x x1 x2 : ℕ) (h₀ : x = x1 + x2) (h₁ : 3 * x1 + 4 * x2 = 33) : x = 6 :=
sorry

end number_of_team_members_l207_20790


namespace scientific_notation_of_8200000_l207_20728

theorem scientific_notation_of_8200000 : 
  (8200000 : ℝ) = 8.2 * 10^6 := 
sorry

end scientific_notation_of_8200000_l207_20728


namespace initial_shipment_robot_rascals_l207_20731

theorem initial_shipment_robot_rascals 
(T : ℝ) 
(h1 : (0.7 * T = 168)) : 
  T = 240 :=
sorry

end initial_shipment_robot_rascals_l207_20731


namespace length_of_CD_l207_20740

theorem length_of_CD (x y: ℝ) (h1: 5 * x = 3 * y) (u v: ℝ) (h2: u = x + 3) (h3: v = y - 3) (h4: 7 * u = 4 * v) : x + y = 264 :=
by
  sorry

end length_of_CD_l207_20740


namespace final_number_lt_one_l207_20715

theorem final_number_lt_one :
  ∀ (numbers : Finset ℕ),
    (numbers = Finset.range 3000 \ Finset.range 1000) →
    (∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → a ≤ b →
    ∃ (numbers' : Finset ℕ), numbers' = (numbers \ {a, b}) ∪ {a / 2}) →
    ∃ (x : ℕ), x ∈ numbers ∧ x < 1 :=
by
  sorry

end final_number_lt_one_l207_20715


namespace office_distance_l207_20765

theorem office_distance (d t : ℝ) 
    (h1 : d = 40 * (t + 1.5)) 
    (h2 : d - 40 = 60 * (t - 2)) : 
    d = 340 :=
by
  -- The detailed proof omitted
  sorry

end office_distance_l207_20765


namespace problem_statement_l207_20795

variable {R : Type*} [LinearOrderedField R]

theorem problem_statement
  (x1 x2 x3 y1 y2 y3 : R)
  (h1 : x1 + x2 + x3 = 0)
  (h2 : y1 + y2 + y3 = 0)
  (h3 : x1 * y1 + x2 * y2 + x3 * y3 = 0)
  (h4 : (x1^2 + x2^2 + x3^2) * (y1^2 + y2^2 + y3^2) > 0) :
  (x1^2 / (x1^2 + x2^2 + x3^2) + y1^2 / (y1^2 + y2^2 + y3^2) = 2 / 3) := 
sorry

end problem_statement_l207_20795


namespace problem_divisibility_l207_20706

theorem problem_divisibility 
  (a b c : ℕ) 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : b ∣ a^3)
  (h2 : c ∣ b^3)
  (h3 : a ∣ c^3) : 
  (a + b + c) ^ 13 ∣ a * b * c := 
sorry

end problem_divisibility_l207_20706


namespace remainder_3_pow_2n_plus_8_l207_20764

theorem remainder_3_pow_2n_plus_8 (n : Nat) : (3 ^ (2 * n) + 8) % 8 = 1 := by
  sorry

end remainder_3_pow_2n_plus_8_l207_20764


namespace inequality_problem_l207_20720

theorem inequality_problem 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a + b + c + 2 = a * b * c) : 
  (a + 1) * (b + 1) * (c + 1) ≥ 27 := 
by sorry

end inequality_problem_l207_20720


namespace geometric_sequence_tenth_fifth_terms_l207_20737

variable (a r : ℚ) (n : ℕ)

def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

theorem geometric_sequence_tenth_fifth_terms :
  (geometric_sequence 4 (4/3) 10 = 1048576 / 19683) ∧ (geometric_sequence 4 (4/3) 5 = 1024 / 81) :=
by
  sorry

end geometric_sequence_tenth_fifth_terms_l207_20737


namespace meaningful_sqrt_range_l207_20775

theorem meaningful_sqrt_range (x : ℝ) (h : 0 ≤ x + 3) : -3 ≤ x :=
by sorry

end meaningful_sqrt_range_l207_20775


namespace triangle_inequality_l207_20751
open Real

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_abc : a + b > c) (h_acb : a + c > b) (h_bca : b + c > a) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_l207_20751


namespace computation_problems_count_l207_20788

theorem computation_problems_count : 
  ∃ (x y : ℕ), 3 * x + 5 * y = 110 ∧ x + y = 30 ∧ x = 20 :=
by
  sorry

end computation_problems_count_l207_20788


namespace f_le_one_l207_20773

open Real

theorem f_le_one (x : ℝ) (hx : 0 < x) : (1 + log x) / x ≤ 1 := 
sorry

end f_le_one_l207_20773


namespace compare_2_roses_3_carnations_l207_20709

variable (x y : ℝ)

def condition1 : Prop := 6 * x + 3 * y > 24
def condition2 : Prop := 4 * x + 5 * y < 22

theorem compare_2_roses_3_carnations (h1 : condition1 x y) (h2 : condition2 x y) : 2 * x > 3 * y := sorry

end compare_2_roses_3_carnations_l207_20709


namespace go_stones_perimeter_l207_20713

-- Define the conditions for the problem
def stones_wide : ℕ := 4
def stones_tall : ℕ := 8

-- Define what we want to prove based on the conditions
theorem go_stones_perimeter : 2 * stones_wide + 2 * stones_tall - 4 = 20 :=
by
  -- Proof would normally go here
  sorry

end go_stones_perimeter_l207_20713


namespace find_value_of_y_l207_20717

theorem find_value_of_y (x y : ℕ) 
    (h1 : 2^x - 2^y = 3 * 2^12) 
    (h2 : x = 14) : 
    y = 13 := 
by
  sorry

end find_value_of_y_l207_20717


namespace probability_sequence_rw_10_l207_20769

noncomputable def probability_red_white_red : ℚ :=
  (4 / 10) * (6 / 9) * (3 / 8)

theorem probability_sequence_rw_10 :
    probability_red_white_red = 1 / 10 := by
  sorry

end probability_sequence_rw_10_l207_20769


namespace strike_time_10_times_l207_20745

def time_to_strike (n : ℕ) : ℝ :=
  if n = 0 then 0 else (n - 1) * 6

theorem strike_time_10_times : time_to_strike 10 = 60 :=
  by {
    -- Proof outline
    -- time_to_strike 10 = (10 - 1) * 6 = 9 * 6 = 54. Thanks to provided solution -> we shall consider that time take 10 seconds for the clock to start striking.
    sorry
  }

end strike_time_10_times_l207_20745


namespace smallest_sum_of_squares_value_l207_20778

noncomputable def collinear_points_min_value (A B C D E P : ℝ): Prop :=
  let AB := 3
  let BC := 2
  let CD := 5
  let DE := 4
  let pos_A := 0
  let pos_B := pos_A + AB
  let pos_C := pos_B + BC
  let pos_D := pos_C + CD
  let pos_E := pos_D + DE
  let P := P
  let AP := (P - pos_A)
  let BP := (P - pos_B)
  let CP := (P - pos_C)
  let DP := (P - pos_D)
  let EP := (P - pos_E)
  let sum_squares := AP^2 + BP^2 + CP^2 + DP^2 + EP^2
  (sum_squares = 85.2)

theorem smallest_sum_of_squares_value : ∃ (A B C D E P : ℝ), collinear_points_min_value A B C D E P :=
sorry

end smallest_sum_of_squares_value_l207_20778


namespace no_solutions_ordered_triples_l207_20723

theorem no_solutions_ordered_triples :
  ¬ ∃ (x y z : ℤ), 
    x^2 - 4 * x * y + 3 * y^2 - z^2 = 25 ∧
    -x^2 + 5 * y * z + 3 * z^2 = 55 ∧
    x^2 + 2 * x * y + 9 * z^2 = 150 :=
by
  sorry

end no_solutions_ordered_triples_l207_20723
