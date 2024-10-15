import Mathlib

namespace NUMINAMATH_GPT_simplify_expression_l1829_182976

theorem simplify_expression (x : ℝ) : 
  3 - 5 * x - 7 * x^2 + 9 - 11 * x + 13 * x^2 - 15 + 17 * x + 19 * x^2 = 25 * x^2 + x - 3 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1829_182976


namespace NUMINAMATH_GPT_percentage_music_students_l1829_182949

variables (total_students : ℕ) (dance_students : ℕ) (art_students : ℕ)
  (music_students : ℕ) (music_percentage : ℚ)

def students_music : ℕ := total_students - (dance_students + art_students)
def percentage_students_music : ℚ := (students_music total_students dance_students art_students : ℚ) / (total_students : ℚ) * 100

theorem percentage_music_students (h1 : total_students = 400)
                                  (h2 : dance_students = 120)
                                  (h3 : art_students = 200) :
  percentage_students_music total_students dance_students art_students = 20 := by {
  sorry
}

end NUMINAMATH_GPT_percentage_music_students_l1829_182949


namespace NUMINAMATH_GPT_problem_proof_l1829_182921

-- Define the geometric sequence and vectors conditions
variables (a : ℕ → ℝ) (q : ℝ)
variables (h1 : ∀ n, a (n + 1) = q * a n)
variables (h2 : a 2 = a 2)
variables (h3 : a 3 = q * a 2)
variables (h4 : 3 * a 2 = 2 * a 3)

-- Statement to prove
theorem problem_proof:
  (a 2 + a 4) / (a 3 + a 5) = 2 / 3 :=
  sorry

end NUMINAMATH_GPT_problem_proof_l1829_182921


namespace NUMINAMATH_GPT_find_second_equation_value_l1829_182935

theorem find_second_equation_value:
  (∃ x y : ℝ, 2 * x + y = 26 ∧ (x + y) / 3 = 4) →
  (∃ x y : ℝ, 2 * x + y = 26 ∧ x + 2 * y = 10) :=
by
  sorry

end NUMINAMATH_GPT_find_second_equation_value_l1829_182935


namespace NUMINAMATH_GPT_trigonometric_identity_simplification_l1829_182911

open Real

theorem trigonometric_identity_simplification (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 4) :
  (sqrt (1 - 2 * sin (3 * π - θ) * sin (π / 2 + θ)) = cos θ - sin θ) :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_simplification_l1829_182911


namespace NUMINAMATH_GPT_Ellipse_area_constant_l1829_182989

-- Definitions of given conditions and problem setup
def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0

def point_on_ellipse (a b : ℝ) : Prop :=
  ellipse_equation 1 (Real.sqrt 3 / 2) a b

def eccentricity (c a : ℝ) : Prop :=
  c / a = Real.sqrt 3 / 2

def moving_points_on_ellipse (a b x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse_equation x₁ y₁ a b ∧ ellipse_equation x₂ y₂ a b

def slopes_condition (k₁ k₂ : ℝ) : Prop :=
  k₁ * k₂ = -1/4

def area_OMN := 1

-- Main theorem statement
theorem Ellipse_area_constant
(a b : ℝ) 
(h_ellipse : point_on_ellipse a b)
(h_eccentricity : eccentricity (Real.sqrt 3 / 2 * a) a)
(M N : ℝ × ℝ) 
(h_points : moving_points_on_ellipse a b M.1 M.2 N.1 N.2)
(k₁ k₂ : ℝ) 
(h_slopes : slopes_condition k₁ k₂) : 
a^2 = 4 ∧ b^2 = 1 ∧ area_OMN = 1 := 
sorry

end NUMINAMATH_GPT_Ellipse_area_constant_l1829_182989


namespace NUMINAMATH_GPT_Victor_bought_6_decks_l1829_182929

theorem Victor_bought_6_decks (V : ℕ) (h1 : 2 * 8 + 8 * V = 64) : V = 6 := by
  sorry

end NUMINAMATH_GPT_Victor_bought_6_decks_l1829_182929


namespace NUMINAMATH_GPT_remainder_is_cx_plus_d_l1829_182986

-- Given a polynomial Q, assume the following conditions
variables {Q : ℕ → ℚ}

-- Conditions
axiom condition1 : Q 15 = 12
axiom condition2 : Q 10 = 4

theorem remainder_is_cx_plus_d : 
  ∃ c d, (c = 8 / 5) ∧ (d = -12) ∧ 
          ∀ x, Q x % ((x - 10) * (x - 15)) = c * x + d :=
by
  sorry

end NUMINAMATH_GPT_remainder_is_cx_plus_d_l1829_182986


namespace NUMINAMATH_GPT_percent_alcohol_new_solution_l1829_182999

theorem percent_alcohol_new_solution :
  let original_volume := 40
  let original_percent_alcohol := 5
  let added_alcohol := 2.5
  let added_water := 7.5
  let original_alcohol := original_volume * (original_percent_alcohol / 100)
  let total_alcohol := original_alcohol + added_alcohol
  let new_total_volume := original_volume + added_alcohol + added_water
  (total_alcohol / new_total_volume) * 100 = 9 :=
by
  sorry

end NUMINAMATH_GPT_percent_alcohol_new_solution_l1829_182999


namespace NUMINAMATH_GPT_min_value_of_2a_b_c_l1829_182992

-- Given conditions
variables (a b c : ℝ)
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)
variables (h : a * (a + b + c) + b * c = 4 + 2 * Real.sqrt 3)

-- Question to prove
theorem min_value_of_2a_b_c : 2 * a + b + c = 2 * Real.sqrt 3 + 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_2a_b_c_l1829_182992


namespace NUMINAMATH_GPT_apple_multiple_l1829_182956

theorem apple_multiple (K Ka : ℕ) (M : ℕ) 
  (h1 : K + Ka = 340)
  (h2 : Ka = M * K + 10)
  (h3 : Ka = 274) : 
  M = 4 := 
by
  sorry

end NUMINAMATH_GPT_apple_multiple_l1829_182956


namespace NUMINAMATH_GPT_correct_arrangements_count_l1829_182907

def valid_arrangements_count : Nat :=
  let houses := ['O', 'R', 'B', 'Y', 'G']
  let arrangements := houses.permutations
  let valid_arr := arrangements.filter (fun a =>
    let o_idx := a.indexOf 'O'
    let r_idx := a.indexOf 'R'
    let b_idx := a.indexOf 'B'
    let y_idx := a.indexOf 'Y'
    let constraints_met :=
      o_idx < r_idx ∧       -- O before R
      b_idx < y_idx ∧       -- B before Y
      (b_idx + 1 != y_idx) ∧ -- B not next to Y
      (r_idx + 1 != b_idx) ∧ -- R not next to B
      (b_idx + 1 != r_idx)   -- symmetrical R not next to B

    constraints_met)
  valid_arr.length

theorem correct_arrangements_count : valid_arrangements_count = 5 :=
  by
    -- To be filled with proof steps.
    sorry

end NUMINAMATH_GPT_correct_arrangements_count_l1829_182907


namespace NUMINAMATH_GPT_xiao_ming_valid_paths_final_valid_paths_l1829_182951

-- Definitions from conditions
def paths_segments := ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
def initial_paths := 256
def invalid_paths := 64

-- Theorem statement
theorem xiao_ming_valid_paths : initial_paths - invalid_paths = 192 :=
by sorry

theorem final_valid_paths : 192 * 2 = 384 :=
by sorry

end NUMINAMATH_GPT_xiao_ming_valid_paths_final_valid_paths_l1829_182951


namespace NUMINAMATH_GPT_bonnie_roark_wire_ratio_l1829_182913

theorem bonnie_roark_wire_ratio :
  let bonnie_wire_length := 12 * 8
  let bonnie_cube_volume := 8 ^ 3
  let roark_cube_volume := 2
  let roark_edge_length := 1.5
  let roark_cube_edge_count := 12
  let num_roark_cubes := bonnie_cube_volume / roark_cube_volume
  let roark_wire_per_cube := roark_cube_edge_count * roark_edge_length
  let roark_total_wire := num_roark_cubes * roark_wire_per_cube
  bonnie_wire_length / roark_total_wire = 1 / 48 :=
  by
  sorry

end NUMINAMATH_GPT_bonnie_roark_wire_ratio_l1829_182913


namespace NUMINAMATH_GPT_bridget_block_collection_l1829_182912

-- Defining the number of groups and blocks per group.
def num_groups : ℕ := 82
def blocks_per_group : ℕ := 10

-- Defining the total number of blocks calculation.
def total_blocks : ℕ := num_groups * blocks_per_group

-- Theorem stating the total number of blocks is 820.
theorem bridget_block_collection : total_blocks = 820 :=
  by
  sorry

end NUMINAMATH_GPT_bridget_block_collection_l1829_182912


namespace NUMINAMATH_GPT_range_of_a_l1829_182973

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1829_182973


namespace NUMINAMATH_GPT_tan_alpha_neg_seven_l1829_182968

noncomputable def tan_alpha (α : ℝ) := Real.tan α

theorem tan_alpha_neg_seven {α : ℝ} 
  (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h2 : Real.cos α ^ 2 + Real.sin (Real.pi + 2 * α) = 3 / 10) : 
  tan_alpha α = -7 := 
sorry

end NUMINAMATH_GPT_tan_alpha_neg_seven_l1829_182968


namespace NUMINAMATH_GPT_frac_sum_eq_one_l1829_182947

variable {x y : ℝ}

theorem frac_sum_eq_one (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : (1 / x) + (1 / y) = 1 :=
by sorry

end NUMINAMATH_GPT_frac_sum_eq_one_l1829_182947


namespace NUMINAMATH_GPT_math_problem_l1829_182987

theorem math_problem : 2 + 5 * 4 - 6 + 3 = 19 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1829_182987


namespace NUMINAMATH_GPT_find_n_solution_l1829_182965

theorem find_n_solution (n : ℚ) (h : (2 / (n+2)) + (4 / (n+2)) + (n / (n+2)) = 4) : n = -2 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_n_solution_l1829_182965


namespace NUMINAMATH_GPT_resort_total_cost_l1829_182944

noncomputable def first_cabin_cost (P : ℝ) := P
noncomputable def second_cabin_cost (P : ℝ) := (1/2) * P
noncomputable def third_cabin_cost (P : ℝ) := (1/6) * P
noncomputable def land_cost (P : ℝ) := 4 * P
noncomputable def pool_cost (P : ℝ) := P

theorem resort_total_cost (P : ℝ) (h : P = 22500) :
  first_cabin_cost P + pool_cost P + second_cabin_cost P + third_cabin_cost P + land_cost P = 150000 :=
by
  sorry

end NUMINAMATH_GPT_resort_total_cost_l1829_182944


namespace NUMINAMATH_GPT_solve_inequality_l1829_182928

theorem solve_inequality (x : ℝ) :
  (2 * x - 1) / (3 * x + 1) > 0 ↔ x < -1/3 ∨ x > 1/2 :=
  sorry

end NUMINAMATH_GPT_solve_inequality_l1829_182928


namespace NUMINAMATH_GPT_P_subsetneq_Q_l1829_182983

def P : Set ℝ := { x : ℝ | x > 1 }
def Q : Set ℝ := { x : ℝ | x^2 - x > 0 }

theorem P_subsetneq_Q : P ⊂ Q :=
by
  sorry

end NUMINAMATH_GPT_P_subsetneq_Q_l1829_182983


namespace NUMINAMATH_GPT_largest_subset_size_with_property_l1829_182950

def no_four_times_property (S : Finset ℕ) : Prop := 
  ∀ {x y}, x ∈ S → y ∈ S → x = 4 * y → False

noncomputable def max_subset_size : ℕ := 145

theorem largest_subset_size_with_property :
  ∃ (S : Finset ℕ), (∀ x ∈ S, x ≤ 150) ∧ no_four_times_property S ∧ S.card = max_subset_size :=
sorry

end NUMINAMATH_GPT_largest_subset_size_with_property_l1829_182950


namespace NUMINAMATH_GPT_circle_diameter_l1829_182962

theorem circle_diameter (A : ℝ) (hA : A = 25 * π) (r : ℝ) (h : A = π * r^2) : 2 * r = 10 := by
  sorry

end NUMINAMATH_GPT_circle_diameter_l1829_182962


namespace NUMINAMATH_GPT_base_three_to_base_ten_l1829_182922

theorem base_three_to_base_ten : 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0 = 178) :=
by
  sorry

end NUMINAMATH_GPT_base_three_to_base_ten_l1829_182922


namespace NUMINAMATH_GPT_different_signs_abs_value_larger_l1829_182923

variable {a b : ℝ}

theorem different_signs_abs_value_larger (h1 : a + b < 0) (h2 : ab < 0) : 
  (a > 0 ∧ b < 0 ∧ |a| < |b|) ∨ (a < 0 ∧ b > 0 ∧ |b| < |a|) :=
sorry

end NUMINAMATH_GPT_different_signs_abs_value_larger_l1829_182923


namespace NUMINAMATH_GPT_total_area_of_sheet_l1829_182903

theorem total_area_of_sheet (x : ℕ) (h1 : 4 * x - x = 2208) : x + 4 * x = 3680 := 
sorry

end NUMINAMATH_GPT_total_area_of_sheet_l1829_182903


namespace NUMINAMATH_GPT_initial_saltwater_amount_l1829_182966

variable (x y : ℝ)
variable (h1 : 0.04 * x = (x - y) * 0.1)
variable (h2 : ((x - y) * 0.1 + 300 * 0.04) / (x - y + 300) = 0.064)

theorem initial_saltwater_amount : x = 500 :=
by
  sorry

end NUMINAMATH_GPT_initial_saltwater_amount_l1829_182966


namespace NUMINAMATH_GPT_union_eq_interval_l1829_182964

def A := { x : ℝ | 1 < x ∧ x < 4 }
def B := { x : ℝ | (x - 3) * (x + 1) ≤ 0 }

theorem union_eq_interval : (A ∪ B) = { x : ℝ | -1 ≤ x ∧ x < 4 } :=
by
  sorry

end NUMINAMATH_GPT_union_eq_interval_l1829_182964


namespace NUMINAMATH_GPT_speed_of_current_l1829_182967

theorem speed_of_current (upstream_time : ℝ) (downstream_time : ℝ) :
    upstream_time = 25 / 60 ∧ downstream_time = 12 / 60 →
    ( (60 / downstream_time - 60 / upstream_time) / 2 ) = 1.3 :=
by
  -- Introduce the conditions
  intro h
  -- Simplify using given facts
  have h1 := h.1
  have h2 := h.2
  -- Calcuation of the speed of current
  sorry

end NUMINAMATH_GPT_speed_of_current_l1829_182967


namespace NUMINAMATH_GPT_profit_percent_approx_l1829_182918

noncomputable def purchase_price : ℝ := 225
noncomputable def overhead_expenses : ℝ := 30
noncomputable def selling_price : ℝ := 300

noncomputable def cost_price : ℝ := purchase_price + overhead_expenses
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percent : ℝ := (profit / cost_price) * 100

theorem profit_percent_approx :
  purchase_price = 225 ∧ 
  overhead_expenses = 30 ∧ 
  selling_price = 300 → 
  abs (profit_percent - 17.65) < 0.01 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_profit_percent_approx_l1829_182918


namespace NUMINAMATH_GPT_dot_product_of_a_and_c_is_4_l1829_182982

def vector := (ℝ × ℝ)

def a : vector := (1, -2)
def b : vector := (-3, 2)

def three_a : vector := (3 * 1, 3 * -2)
def two_b_minus_a : vector := (2 * -3 - 1, 2 * 2 - -2)

def c : vector := (-(-three_a.fst + two_b_minus_a.fst), -(-three_a.snd + two_b_minus_a.snd))

def dot_product (u v : vector) : ℝ := u.fst * v.fst + u.snd * v.snd

theorem dot_product_of_a_and_c_is_4 : dot_product a c = 4 := 
by
  sorry

end NUMINAMATH_GPT_dot_product_of_a_and_c_is_4_l1829_182982


namespace NUMINAMATH_GPT_job_completion_l1829_182974

theorem job_completion (A_rate D_rate : ℝ) (h₁ : A_rate = 1 / 12) (h₂ : A_rate + D_rate = 1 / 4) : D_rate = 1 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_job_completion_l1829_182974


namespace NUMINAMATH_GPT_max_distance_between_P_and_Q_l1829_182960

-- Definitions of the circle and ellipse
def is_on_circle (P : ℝ × ℝ) : Prop := P.1^2 + (P.2 - 6)^2 = 2
def is_on_ellipse (Q : ℝ × ℝ) : Prop := (Q.1^2) / 10 + Q.2^2 = 1

-- The maximum distance between any point on the circle and any point on the ellipse
theorem max_distance_between_P_and_Q :
  ∃ P Q : ℝ × ℝ, is_on_circle P ∧ is_on_ellipse Q ∧ dist P Q = 6 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_distance_between_P_and_Q_l1829_182960


namespace NUMINAMATH_GPT_zoe_calories_l1829_182902

theorem zoe_calories 
  (s : ℕ) (y : ℕ) (c_s : ℕ) (c_y : ℕ)
  (s_eq : s = 12) (y_eq : y = 6) (cs_eq : c_s = 4) (cy_eq : c_y = 17) :
  s * c_s + y * c_y = 150 :=
by
  sorry

end NUMINAMATH_GPT_zoe_calories_l1829_182902


namespace NUMINAMATH_GPT_find_first_number_l1829_182924

variable {A B C D : ℕ}

theorem find_first_number (h1 : A + B + C = 60) (h2 : B + C + D = 45) (h3 : D = 18) : A = 33 := 
  sorry

end NUMINAMATH_GPT_find_first_number_l1829_182924


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l1829_182940

variable {a b : ℝ}

theorem line_passes_through_fixed_point : 
  (∀ (x y : ℝ), a + 2 * b = 1 ∧ ax + 3 * y + b = 0 → (x, y) = (1/2, -1/6)) :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l1829_182940


namespace NUMINAMATH_GPT_ordered_pair_solution_l1829_182939

theorem ordered_pair_solution :
  ∃ x y : ℤ, (x + y = (3 - x) + (3 - y)) ∧ (x - y = (x - 2) + (y - 2)) ∧ (x = 2) ∧ (y = 1) :=
by
  use 2, 1
  repeat { sorry }

end NUMINAMATH_GPT_ordered_pair_solution_l1829_182939


namespace NUMINAMATH_GPT_area_of_circle_with_given_circumference_l1829_182938

-- Defining the given problem's conditions as variables
variables (C : ℝ) (r : ℝ) (A : ℝ)
  
-- The condition that circumference is 12π meters
def circumference_condition : Prop := C = 12 * Real.pi
  
-- The relationship between circumference and radius
def radius_relationship : Prop := C = 2 * Real.pi * r
  
-- The formula to calculate the area of the circle
def area_formula : Prop := A = Real.pi * r^2
  
-- The proof goal that we need to establish
theorem area_of_circle_with_given_circumference :
  circumference_condition C ∧ radius_relationship C r ∧ area_formula A r → A = 36 * Real.pi :=
by
  intros
  sorry -- Skipping the proof, to be done later

end NUMINAMATH_GPT_area_of_circle_with_given_circumference_l1829_182938


namespace NUMINAMATH_GPT_mark_asphalt_total_cost_l1829_182908

noncomputable def total_cost (road_length : ℕ) (road_width : ℕ) (area_per_truckload : ℕ) (cost_per_truckload : ℕ) (sales_tax_rate : ℚ) : ℚ :=
  let total_area := road_length * road_width
  let num_truckloads := total_area / area_per_truckload
  let cost_before_tax := num_truckloads * cost_per_truckload
  let sales_tax := cost_before_tax * sales_tax_rate
  let total_cost := cost_before_tax + sales_tax
  total_cost

theorem mark_asphalt_total_cost :
  total_cost 2000 20 800 75 0.2 = 4500 := 
by sorry

end NUMINAMATH_GPT_mark_asphalt_total_cost_l1829_182908


namespace NUMINAMATH_GPT_inequality_proof_l1829_182917

variable (m n : ℝ)

theorem inequality_proof (hm : m < 0) (hn : n > 0) (h_sum : m + n < 0) : m < -n ∧ -n < n ∧ n < -m :=
by
  -- introduction and proof commands would go here, but we use sorry to indicate the proof is omitted
  sorry

end NUMINAMATH_GPT_inequality_proof_l1829_182917


namespace NUMINAMATH_GPT_watch_current_price_l1829_182909

-- Definitions based on conditions
def original_price : ℝ := 15
def first_reduction_rate : ℝ := 0.25
def second_reduction_rate : ℝ := 0.40

-- The price after the first reduction
def first_reduced_price : ℝ := original_price * (1 - first_reduction_rate)

-- The price after the second reduction
def final_price : ℝ := first_reduced_price * (1 - second_reduction_rate)

-- The theorem that needs to be proved
theorem watch_current_price : final_price = 6.75 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_watch_current_price_l1829_182909


namespace NUMINAMATH_GPT_trapezoid_PQRS_PQ_squared_l1829_182931

theorem trapezoid_PQRS_PQ_squared
  (PR PS PQ : ℝ)
  (cond1 : PR = 13)
  (cond2 : PS = 17)
  (h : PQ^2 + PR^2 = PS^2) :
  PQ^2 = 120 :=
by
  rw [cond1, cond2] at h
  sorry

end NUMINAMATH_GPT_trapezoid_PQRS_PQ_squared_l1829_182931


namespace NUMINAMATH_GPT_rectangle_perimeter_is_70_l1829_182991

-- Define the length and width of the rectangle
def length : ℕ := 19
def width : ℕ := 16

-- Define the perimeter function for a rectangle
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- The theorem statement asserting that the perimeter of the given rectangle is 70 cm
theorem rectangle_perimeter_is_70 :
  perimeter length width = 70 := 
sorry

end NUMINAMATH_GPT_rectangle_perimeter_is_70_l1829_182991


namespace NUMINAMATH_GPT_intersecting_graphs_value_l1829_182955

theorem intersecting_graphs_value (a b c d : ℝ) 
  (h1 : 5 = -|2 - a| + b) 
  (h2 : 3 = -|8 - a| + b) 
  (h3 : 5 = |2 - c| + d) 
  (h4 : 3 = |8 - c| + d) : 
  a + c = 10 :=
sorry

end NUMINAMATH_GPT_intersecting_graphs_value_l1829_182955


namespace NUMINAMATH_GPT_trigonometric_identity_l1829_182933

theorem trigonometric_identity (θ : ℝ) (h : Real.tan (π / 4 + θ) = 3) :
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1829_182933


namespace NUMINAMATH_GPT_savings_if_together_l1829_182985

def price_per_window : ℕ := 150

def discount_offer (n : ℕ) : ℕ := n - n / 7

def cost (n : ℕ) : ℕ := price_per_window * discount_offer n

def alice_windows : ℕ := 9
def bob_windows : ℕ := 10

def separate_cost : ℕ := cost alice_windows + cost bob_windows

def total_windows : ℕ := alice_windows + bob_windows

def together_cost : ℕ := cost total_windows

def savings : ℕ := separate_cost - together_cost

theorem savings_if_together : savings = 150 := by
  sorry

end NUMINAMATH_GPT_savings_if_together_l1829_182985


namespace NUMINAMATH_GPT_solve_abs_inequality_l1829_182995

theorem solve_abs_inequality (x : ℝ) : 
  (3 ≤ abs (x + 2) ∧ abs (x + 2) ≤ 6) ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) := 
by sorry

end NUMINAMATH_GPT_solve_abs_inequality_l1829_182995


namespace NUMINAMATH_GPT_abs_diff_of_m_and_n_l1829_182988

theorem abs_diff_of_m_and_n (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 7) : |m - n| = 5 :=
sorry

end NUMINAMATH_GPT_abs_diff_of_m_and_n_l1829_182988


namespace NUMINAMATH_GPT_series_satisfies_l1829_182948

noncomputable def series (x : ℝ) : ℝ :=
  let S₁ := 1 / (1 + x^2)
  let S₂ := x / (1 + x^2)
  (S₁ - S₂)

theorem series_satisfies (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  x = series x ↔ x^3 + 2 * x - 1 = 0 :=
by 
  -- Proof outline:
  -- 1. Calculate the series S as a function of x
  -- 2. Equate series x to x and simplify to derive the polynomial equation
  sorry

end NUMINAMATH_GPT_series_satisfies_l1829_182948


namespace NUMINAMATH_GPT_single_elimination_matches_l1829_182957

theorem single_elimination_matches (n : ℕ) (h : n = 512) :
  ∃ (m : ℕ), m = n - 1 ∧ m = 511 :=
by
  sorry

end NUMINAMATH_GPT_single_elimination_matches_l1829_182957


namespace NUMINAMATH_GPT_solve_problem1_solve_problem2_l1829_182996

-- Problem 1
theorem solve_problem1 (x : ℚ) : (3 * x - 1) ^ 2 = 9 ↔ x = 4 / 3 ∨ x = -2 / 3 := 
by sorry

-- Problem 2
theorem solve_problem2 (x : ℚ) : x * (2 * x - 4) = (2 - x) ^ 2 ↔ x = 2 ∨ x = -2 :=
by sorry

end NUMINAMATH_GPT_solve_problem1_solve_problem2_l1829_182996


namespace NUMINAMATH_GPT_no_valid_base_l1829_182952

theorem no_valid_base (b : ℤ) (n : ℤ) : b^2 + 2*b + 2 ≠ n^2 := by
  sorry

end NUMINAMATH_GPT_no_valid_base_l1829_182952


namespace NUMINAMATH_GPT_cost_of_rope_l1829_182984

theorem cost_of_rope : 
  ∀ (total_money sheet_cost propane_burner_cost helium_cost_per_ounce helium_per_foot max_height rope_cost : ℝ),
  total_money = 200 ∧
  sheet_cost = 42 ∧
  propane_burner_cost = 14 ∧
  helium_cost_per_ounce = 1.50 ∧
  helium_per_foot = 113 ∧
  max_height = 9492 ∧
  rope_cost = total_money - (sheet_cost + propane_burner_cost + (max_height / helium_per_foot) * helium_cost_per_ounce) →
  rope_cost = 18 :=
by
  intros total_money sheet_cost propane_burner_cost helium_cost_per_ounce helium_per_foot max_height rope_cost
  rintro ⟨h_total, h_sheet, h_propane, h_helium, h_perfoot, h_max, h_rope⟩
  rw [h_total, h_sheet, h_propane, h_helium, h_perfoot, h_max] at h_rope
  simp only [inv_mul_eq_iff_eq_mul, div_eq_mul_inv] at h_rope
  norm_num at h_rope
  sorry

end NUMINAMATH_GPT_cost_of_rope_l1829_182984


namespace NUMINAMATH_GPT_part1_part2_l1829_182927

theorem part1 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
sorry

theorem part2 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) ≥ (2 / (1 + a)) + (2 / (1 + b)) + (2 / (1 + c)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1829_182927


namespace NUMINAMATH_GPT_second_group_students_l1829_182981

theorem second_group_students (S : ℕ) : 
    (1200 / 40) = 9 + S + 11 → S = 10 :=
by sorry

end NUMINAMATH_GPT_second_group_students_l1829_182981


namespace NUMINAMATH_GPT_negation_of_proposition_l1829_182993

theorem negation_of_proposition :
  ¬ (∃ x_0 : ℝ, 2^x_0 < x_0^2) ↔ (∀ x : ℝ, 2^x ≥ x^2) :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l1829_182993


namespace NUMINAMATH_GPT_brick_laying_days_l1829_182920

theorem brick_laying_days (a m n d : ℕ) (hm : 0 < m) (hn : 0 < n) (hd : 0 < d) :
  let rate_M := m / (a * d)
  let rate_N := n / (a * (2 * d))
  let total_days := 3 * a^2 / (m + n)
  (a * rate_M * (d * total_days) + 2 * a * rate_N * (d * total_days)) = (a + 2 * a) :=
by
  -- Definitions from the problem conditions
  let rate_M := m / (a * d)
  let rate_N := n / (a * (2 * d))
  let total_days := 3 * a^2 / (m + n)
  have h0 : a * rate_M * (d * total_days) = a := sorry
  have h1 : 2 * a * rate_N * (d * total_days) = 2 * a := sorry
  exact sorry

end NUMINAMATH_GPT_brick_laying_days_l1829_182920


namespace NUMINAMATH_GPT_intervals_of_monotonic_increase_max_area_acute_triangle_l1829_182975

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ :=
  (sin x, (sqrt 3 / 2) * (sin x - cos x))

noncomputable def vector_b (x : ℝ) : ℝ × ℝ :=
  (cos x, sin x + cos x)

noncomputable def f (x : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b x
  a.1 * b.1 + a.2 * b.2

-- Problem 1: Proving the intervals of monotonic increase for the function f(x)
theorem intervals_of_monotonic_increase :
  ∀ k : ℤ, ∀ x : ℝ, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) →
  ∀ x₁ x₂ : ℝ, (k * π - π / 12 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ k * π + 5 * π / 12) → f x₁ ≤ f x₂ :=
sorry

-- Problem 2: Proving the maximum area of triangle ABC
theorem max_area_acute_triangle (A : ℝ) (a b c : ℝ) :
  (f A = 1 / 2) → (a = sqrt 2) →
  ∀ S : ℝ, S ≤ (1 + sqrt 2) / 2 :=
sorry

end NUMINAMATH_GPT_intervals_of_monotonic_increase_max_area_acute_triangle_l1829_182975


namespace NUMINAMATH_GPT_simplify_fraction_l1829_182910

theorem simplify_fraction (h1 : 90 = 2 * 3^2 * 5) (h2 : 150 = 2 * 3 * 5^2) : (90 / 150 : ℚ) = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1829_182910


namespace NUMINAMATH_GPT_factorize_polynomial_l1829_182969

theorem factorize_polynomial (x y : ℝ) :
  3 * x ^ 2 + 6 * x * y + 3 * y ^ 2 = 3 * (x + y) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l1829_182969


namespace NUMINAMATH_GPT_token_exchange_l1829_182916

def booth1 (r : ℕ) (x : ℕ) : ℕ × ℕ × ℕ := (r - 3 * x, 2 * x, x)
def booth2 (b : ℕ) (y : ℕ) : ℕ × ℕ × ℕ := (y, b - 4 * y, y)

theorem token_exchange (x y : ℕ) (h1 : 100 - 3 * x + y = 2) (h2 : 50 + x - 4 * y = 3) :
  x + y = 58 :=
sorry

end NUMINAMATH_GPT_token_exchange_l1829_182916


namespace NUMINAMATH_GPT_implication_equivalence_l1829_182963

variable (P Q : Prop)

theorem implication_equivalence :
  (¬Q → ¬P) ∧ (¬P ∨ Q) ↔ (P → Q) :=
by sorry

end NUMINAMATH_GPT_implication_equivalence_l1829_182963


namespace NUMINAMATH_GPT_bob_distance_when_meet_l1829_182936

def distance_xy : ℝ := 10
def yolanda_rate : ℝ := 3
def bob_rate : ℝ := 4
def time_start_diff : ℝ := 1

theorem bob_distance_when_meet : ∃ t : ℝ, yolanda_rate * t + bob_rate * (t - time_start_diff) = distance_xy ∧ bob_rate * (t - time_start_diff) = 4 :=
by
  sorry

end NUMINAMATH_GPT_bob_distance_when_meet_l1829_182936


namespace NUMINAMATH_GPT_slope_of_line_n_l1829_182979

noncomputable def tan_double_angle (t : ℝ) : ℝ := (2 * t) / (1 - t^2)

theorem slope_of_line_n :
  let slope_m := 6
  let alpha := Real.arctan slope_m
  let slope_n := tan_double_angle slope_m
  slope_n = -12 / 35 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_n_l1829_182979


namespace NUMINAMATH_GPT_most_lines_of_symmetry_l1829_182942

def regular_pentagon_lines_of_symmetry : ℕ := 5
def kite_lines_of_symmetry : ℕ := 1
def regular_hexagon_lines_of_symmetry : ℕ := 6
def isosceles_triangle_lines_of_symmetry : ℕ := 1
def scalene_triangle_lines_of_symmetry : ℕ := 0

theorem most_lines_of_symmetry :
  regular_hexagon_lines_of_symmetry = max
    (max (max (max regular_pentagon_lines_of_symmetry kite_lines_of_symmetry)
              regular_hexagon_lines_of_symmetry)
        isosceles_triangle_lines_of_symmetry)
    scalene_triangle_lines_of_symmetry :=
sorry

end NUMINAMATH_GPT_most_lines_of_symmetry_l1829_182942


namespace NUMINAMATH_GPT_work_rate_problem_l1829_182994

theorem work_rate_problem
  (W : ℕ) -- total work
  (A_rate : ℕ) -- A's work rate in days
  (B_rate : ℕ) -- B's work rate in days
  (x : ℕ) -- days A worked alone
  (total_days : ℕ) -- days A and B worked together
  (hA : A_rate = 12) -- A can do the work in 12 days
  (hB : B_rate = 6) -- B can do the work in 6 days
  (hx : total_days = 3) -- remaining days they together work
  : x = 3 := 
by
  sorry

end NUMINAMATH_GPT_work_rate_problem_l1829_182994


namespace NUMINAMATH_GPT_square_side_length_l1829_182937

theorem square_side_length (x : ℝ) (h : x ^ 2 = 4 * 3) : x = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_square_side_length_l1829_182937


namespace NUMINAMATH_GPT_percentage_differences_equal_l1829_182906

noncomputable def calculation1 : ℝ := 0.60 * 50
noncomputable def calculation2 : ℝ := 0.30 * 30
noncomputable def calculation3 : ℝ := 0.45 * 90
noncomputable def calculation4 : ℝ := 0.20 * 40

noncomputable def diff1 : ℝ := abs (calculation1 - calculation2)
noncomputable def diff2 : ℝ := abs (calculation2 - calculation3)
noncomputable def diff3 : ℝ := abs (calculation3 - calculation4)
noncomputable def largest_diff1 : ℝ := max diff1 (max diff2 diff3)

noncomputable def calculation5 : ℝ := 0.40 * 120
noncomputable def calculation6 : ℝ := 0.25 * 80
noncomputable def calculation7 : ℝ := 0.35 * 150
noncomputable def calculation8 : ℝ := 0.55 * 60

noncomputable def diff4 : ℝ := abs (calculation5 - calculation6)
noncomputable def diff5 : ℝ := abs (calculation6 - calculation7)
noncomputable def diff6 : ℝ := abs (calculation7 - calculation8)
noncomputable def largest_diff2 : ℝ := max diff4 (max diff5 diff6)

theorem percentage_differences_equal :
  largest_diff1 = largest_diff2 :=
sorry

end NUMINAMATH_GPT_percentage_differences_equal_l1829_182906


namespace NUMINAMATH_GPT_unique_sequence_l1829_182971

theorem unique_sequence (a : ℕ → ℕ) (h_distinct: ∀ m n, a m = a n → m = n)
    (h_divisible: ∀ n, a n % a (a n) = 0) : ∀ n, a n = n :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_unique_sequence_l1829_182971


namespace NUMINAMATH_GPT_solve_for_x_l1829_182977

theorem solve_for_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 7 * x^2 + 14 * x * y = x^3 + 3 * x^2 * y) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1829_182977


namespace NUMINAMATH_GPT_spinner_final_direction_l1829_182978

theorem spinner_final_direction 
  (initial_direction : ℕ) -- 0 for north, 1 for east, 2 for south, 3 for west
  (clockwise_revolutions : ℚ)
  (counterclockwise_revolutions : ℚ)
  (net_revolutions : ℚ) -- derived via net movement calculation
  (final_position : ℕ) -- correct position after net movement
  : initial_direction = 3 → clockwise_revolutions = 9/4 → counterclockwise_revolutions = 15/4 → final_position = 1 :=
by
  sorry

end NUMINAMATH_GPT_spinner_final_direction_l1829_182978


namespace NUMINAMATH_GPT_degree_to_radian_60_eq_pi_div_3_l1829_182997

theorem degree_to_radian_60_eq_pi_div_3 (pi : ℝ) (deg : ℝ) 
  (h : 180 * deg = pi) : 60 * deg = pi / 3 := 
by
  sorry

end NUMINAMATH_GPT_degree_to_radian_60_eq_pi_div_3_l1829_182997


namespace NUMINAMATH_GPT_factorization_correct_l1829_182990

theorem factorization_correct :
    (∀ (x y : ℝ), x * (2 * x - y) + 2 * y * (2 * x - y) = (x + 2 * y) * (2 * x - y)) :=
by
  intro x y
  sorry

end NUMINAMATH_GPT_factorization_correct_l1829_182990


namespace NUMINAMATH_GPT_correct_exponentiation_incorrect_division_incorrect_multiplication_incorrect_addition_l1829_182972

theorem correct_exponentiation (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

-- Incorrect options for clarity
theorem incorrect_division (a : ℝ) : a^6 / a^2 ≠ a^3 :=
by sorry

theorem incorrect_multiplication (a : ℝ) : a^2 * a^3 ≠ a^6 :=
by sorry

theorem incorrect_addition (a : ℝ) : (a^2 + a^3) ≠ a^5 :=
by sorry

end NUMINAMATH_GPT_correct_exponentiation_incorrect_division_incorrect_multiplication_incorrect_addition_l1829_182972


namespace NUMINAMATH_GPT_scientific_notation_coronavirus_diameter_l1829_182915

theorem scientific_notation_coronavirus_diameter : 0.00000011 = 1.1 * 10^(-7) :=
by {
  sorry
}

end NUMINAMATH_GPT_scientific_notation_coronavirus_diameter_l1829_182915


namespace NUMINAMATH_GPT_multiplier_for_difference_l1829_182934

variable (x y k : ℕ)
variable (h1 : x + y = 81)
variable (h2 : x^2 - y^2 = k * (x - y))
variable (h3 : x ≠ y)

theorem multiplier_for_difference : k = 81 := 
by
  sorry

end NUMINAMATH_GPT_multiplier_for_difference_l1829_182934


namespace NUMINAMATH_GPT_find_num_yoYos_l1829_182954

variables (x y z w : ℕ)

def stuffed_animals_frisbees_puzzles := x + y + w = 80
def total_prizes := x + y + z + w + 180 + 60
def cars_and_robots := 180 + 60 = x + y + z + w + 15

theorem find_num_yoYos 
(h1 : stuffed_animals_frisbees_puzzles x y w)
(h2 : total_prizes = 300)
(h3 : cars_and_robots x y z w) : z = 145 :=
sorry

end NUMINAMATH_GPT_find_num_yoYos_l1829_182954


namespace NUMINAMATH_GPT_find_prime_triplet_l1829_182941

theorem find_prime_triplet (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  3 * p^4 - 5 * q^4 - 4 * r^2 = 26 ↔ (p, q, r) = (5, 3, 19) :=
by
  sorry

end NUMINAMATH_GPT_find_prime_triplet_l1829_182941


namespace NUMINAMATH_GPT_problem_difference_l1829_182925

-- Define the sum of first n natural numbers
def sumFirstN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Define the rounding rule to the nearest multiple of 5
def roundToNearest5 (x : ℕ) : ℕ :=
  match x % 5 with
  | 0 => x
  | 1 => x - 1
  | 2 => x - 2
  | 3 => x + 2
  | 4 => x + 1
  | _ => x  -- This case is theoretically unreachable

-- Define the sum of the first n natural numbers after rounding to nearest 5
def sumRoundedFirstN (n : ℕ) : ℕ :=
  (List.range (n + 1)).map roundToNearest5 |>.sum

theorem problem_difference : sumFirstN 120 - sumRoundedFirstN 120 = 6900 := by
  sorry

end NUMINAMATH_GPT_problem_difference_l1829_182925


namespace NUMINAMATH_GPT_point_A_inside_circle_O_l1829_182914

-- Definitions based on conditions in the problem
def radius := 5 -- in cm
def distance_to_center := 4 -- in cm

-- The theorem to be proven
theorem point_A_inside_circle_O (r d : ℝ) (hr : r = 5) (hd : d = 4) (h : r > d) : true :=
by {
  sorry
}

end NUMINAMATH_GPT_point_A_inside_circle_O_l1829_182914


namespace NUMINAMATH_GPT_yoongi_has_fewer_apples_l1829_182998

-- Define the number of apples Jungkook originally has and receives more.
def jungkook_original_apples := 6
def jungkook_received_apples := 3

-- Calculate the total number of apples Jungkook has.
def jungkook_total_apples := jungkook_original_apples + jungkook_received_apples

-- Define the number of apples Yoongi has.
def yoongi_apples := 4

-- State that Yoongi has fewer apples than Jungkook.
theorem yoongi_has_fewer_apples : yoongi_apples < jungkook_total_apples := by
  sorry

end NUMINAMATH_GPT_yoongi_has_fewer_apples_l1829_182998


namespace NUMINAMATH_GPT_best_model_is_model1_l1829_182900

noncomputable def model_best_fitting (R1 R2 R3 R4 : ℝ) :=
  R1 = 0.975 ∧ R2 = 0.79 ∧ R3 = 0.55 ∧ R4 = 0.25

theorem best_model_is_model1 (R1 R2 R3 R4 : ℝ) (h : model_best_fitting R1 R2 R3 R4) :
  R1 = max R1 (max R2 (max R3 R4)) :=
by
  cases h with
  | intro h1 h_rest =>
    cases h_rest with
    | intro h2 h_rest2 =>
      cases h_rest2 with
      | intro h3 h4 =>
        sorry

end NUMINAMATH_GPT_best_model_is_model1_l1829_182900


namespace NUMINAMATH_GPT_new_game_cost_l1829_182980

theorem new_game_cost (G : ℕ) (h_initial_money : 83 = G + 9 * 4) : G = 47 := by
  sorry

end NUMINAMATH_GPT_new_game_cost_l1829_182980


namespace NUMINAMATH_GPT_min_value_of_quadratic_l1829_182901

theorem min_value_of_quadratic : ∀ x : ℝ, (x^2 + 6*x + 5) ≥ -4 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l1829_182901


namespace NUMINAMATH_GPT_infinite_sorted_subsequence_l1829_182943

theorem infinite_sorted_subsequence : 
  ∀ (warriors : ℕ → ℕ), (∀ n, ∃ m, m > n ∧ warriors m < warriors n) 
  ∨ (∃ k, warriors k = 0) → 
  ∃ (remaining : ℕ → ℕ), (∀ i j, i < j → remaining i > remaining j) :=
by
  intros warriors h
  sorry

end NUMINAMATH_GPT_infinite_sorted_subsequence_l1829_182943


namespace NUMINAMATH_GPT_elvins_first_month_bill_l1829_182961

theorem elvins_first_month_bill (F C : ℝ) 
  (h1 : F + C = 52)
  (h2 : F + 2 * C = 76) : 
  F + C = 52 :=
by
  sorry

end NUMINAMATH_GPT_elvins_first_month_bill_l1829_182961


namespace NUMINAMATH_GPT_find_common_ratio_sum_arithmetic_sequence_l1829_182905

-- Conditions
variable {a : ℕ → ℝ}   -- a_n is a numeric sequence
variable (S : ℕ → ℝ)   -- S_n is the sum of the first n terms
variable {q : ℝ}       -- q is the common ratio
variable (k : ℕ)

-- Given: a_n is a geometric sequence with common ratio q, q ≠ 1, q ≠ 0
variable (h_geometric : ∀ n, a (n + 1) = a n * q)
variable (h_q_ne_one : q ≠ 1)
variable (h_q_ne_zero : q ≠ 0)

-- Given: S_n = a_1 * (1 - q^n) / (1 - q) when q ≠ 1 and q ≠ 0
variable (h_S : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q))

-- Given: a_5, a_3, a_4 form an arithmetic sequence, so 2a_3 = a_5 + a_4
variable (h_arithmetic : 2 * a 3 = a 5 + a 4)

-- Prove part 1: common ratio q is -2
theorem find_common_ratio (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_arithmetic : 2 * a 3 = a 5 + a 4) 
  (h_q_ne_one : q ≠ 1) (h_q_ne_zero : q ≠ 0) : q = -2 :=
sorry

-- Prove part 2: S_(k+2), S_k, S_(k+1) form an arithmetic sequence
theorem sum_arithmetic_sequence (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_q_ne_one : q ≠ 1) (h_q_ne_zero : q ≠ 0)
  (h_S : ∀ n, S n = a 1 * (1 - q ^ (n + 1)) / (1 - q))
  (k : ℕ) : S (k + 2) + S k = 2 * S (k + 1) :=
sorry

end NUMINAMATH_GPT_find_common_ratio_sum_arithmetic_sequence_l1829_182905


namespace NUMINAMATH_GPT_cos_of_acute_angle_l1829_182946

theorem cos_of_acute_angle (θ : ℝ) (hθ1 : 0 < θ ∧ θ < π / 2) (hθ2 : Real.sin θ = 1 / 3) :
  Real.cos θ = 2 * Real.sqrt 2 / 3 :=
by
  -- The proof steps will be filled here
  sorry

end NUMINAMATH_GPT_cos_of_acute_angle_l1829_182946


namespace NUMINAMATH_GPT_domain_of_f_parity_of_f_range_of_f_l1829_182959

noncomputable def f (a x : ℝ) := Real.log (1 + x) / Real.log a - Real.log (1 - x) / Real.log a

variables {a x : ℝ}

-- The properties derived:
theorem domain_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  (-1 < x ∧ x < 1) ↔ ∃ y, f a x = y :=
sorry

theorem parity_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  f a (-x) = -f a x :=
sorry

theorem range_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  (f a x > 0 ↔ (a > 1 ∧ 0 < x ∧ x < 1) ∨ (0 < a ∧ a < 1 ∧ -1 < x ∧ x < 0)) :=
sorry

end NUMINAMATH_GPT_domain_of_f_parity_of_f_range_of_f_l1829_182959


namespace NUMINAMATH_GPT_negation_of_implication_iff_l1829_182953

variable (a : ℝ)

theorem negation_of_implication_iff (p : a > 1 → a^2 > 1) :
  ¬(a > 1 → a^2 > 1) ↔ (a ≤ 1 → a^2 ≤ 1) :=
by sorry

end NUMINAMATH_GPT_negation_of_implication_iff_l1829_182953


namespace NUMINAMATH_GPT_sum_gt_two_l1829_182945

noncomputable def f (x : ℝ) : ℝ := ((x - 1) * Real.log x) / x

theorem sum_gt_two (x₁ x₂ : ℝ) (h₁ : f x₁ = f x₂) (h₂ : x₁ ≠ x₂) : x₁ + x₂ > 2 := 
sorry

end NUMINAMATH_GPT_sum_gt_two_l1829_182945


namespace NUMINAMATH_GPT_inequality_solution_l1829_182970

theorem inequality_solution (x : ℝ) :
  (x < -2 ∨ (-1 < x ∧ x < 1) ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 6) ∨ 7 < x) →
  (1 / (x - 1)) - (4 / (x - 2)) + (4 / (x - 3)) - (1 / (x - 4)) < 1 / 30 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1829_182970


namespace NUMINAMATH_GPT_correct_calculation_l1829_182926

theorem correct_calculation (a b : ℝ) : (3 * a * b) ^ 2 = 9 * a ^ 2 * b ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1829_182926


namespace NUMINAMATH_GPT_div_ad_bc_by_k_l1829_182904

theorem div_ad_bc_by_k 
  (a b c d l k m n : ℤ)
  (h1 : a * l + b = k * m)
  (h2 : c * l + d = k * n) : 
  k ∣ (a * d - b * c) :=
sorry

end NUMINAMATH_GPT_div_ad_bc_by_k_l1829_182904


namespace NUMINAMATH_GPT_magnitude_vec_sum_l1829_182930

open Real

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
a.1 * b.1 + a.2 * b.2

theorem magnitude_vec_sum
    (a b : ℝ × ℝ)
    (h_angle : ∃ θ, θ = 150 * (π / 180) ∧ cos θ = cos (5 * π / 6))
    (h_norm_a : ‖a‖ = sqrt 3)
    (h_norm_b : ‖b‖ = 2) :
  ‖(2 * a.1 + b.1, 2 * a.2 + b.2)‖ = 2 :=
  by
  sorry

end NUMINAMATH_GPT_magnitude_vec_sum_l1829_182930


namespace NUMINAMATH_GPT_expected_defective_chips_in_60000_l1829_182958

def shipmentS1 := (2, 5000)
def shipmentS2 := (4, 12000)
def shipmentS3 := (2, 15000)
def shipmentS4 := (4, 16000)

def total_defective_chips := shipmentS1.1 + shipmentS2.1 + shipmentS3.1 + shipmentS4.1
def total_chips := shipmentS1.2 + shipmentS2.2 + shipmentS3.2 + shipmentS4.2

def defective_ratio := total_defective_chips / total_chips
def shipment60000 := 60000

def expected_defectives (ratio : ℝ) (total_chips : ℝ) := ratio * total_chips

theorem expected_defective_chips_in_60000 :
  expected_defectives defective_ratio shipment60000 = 15 :=
by
  sorry

end NUMINAMATH_GPT_expected_defective_chips_in_60000_l1829_182958


namespace NUMINAMATH_GPT_range_of_m_l1829_182919

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - m * x + m > 0) ↔ 0 < m ∧ m < 4 :=
by sorry

end NUMINAMATH_GPT_range_of_m_l1829_182919


namespace NUMINAMATH_GPT_intersection_sets_l1829_182932

theorem intersection_sets :
  let M := {x : ℝ | 0 < x} 
  let N := {y : ℝ | 1 ≤ y}
  M ∩ N = {z : ℝ | 1 ≤ z} :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_intersection_sets_l1829_182932
