import Mathlib

namespace isosceles_triangle_problem_l141_141628

theorem isosceles_triangle_problem 
  (a h b : ℝ) 
  (area_relation : (1/2) * a * h = (1/3) * a ^ 2) 
  (leg_relation : b = a - 1)
  (height_relation : h = (2/3) * a) 
  (pythagorean_theorem : h ^ 2 + (a / 2) ^ 2 = b ^ 2) : 
  a = 6 ∧ b = 5 ∧ h = 4 :=
sorry

end isosceles_triangle_problem_l141_141628


namespace cubic_has_one_real_root_iff_l141_141268

theorem cubic_has_one_real_root_iff (a : ℝ) :
  (∃! x : ℝ, x^3 + (1 - a) * x^2 - 2 * a * x + a^2 = 0) ↔ a < -1/4 := by
  sorry

end cubic_has_one_real_root_iff_l141_141268


namespace a2_equals_3_l141_141476

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem a2_equals_3 (a : ℕ → ℕ) (S3 : ℕ) (h1 : a 1 = 1) (h2 : a 1 + a 2 + a 3 = 9) : a 2 = 3 :=
by
  sorry

end a2_equals_3_l141_141476


namespace michael_num_dogs_l141_141976

variable (total_cost : ℕ)
variable (cost_per_animal : ℕ)
variable (num_cats : ℕ)
variable (num_dogs : ℕ)

-- Conditions
def michael_total_cost := total_cost = 65
def michael_num_cats := num_cats = 2
def michael_cost_per_animal := cost_per_animal = 13

-- Theorem to prove
theorem michael_num_dogs (h_total_cost : michael_total_cost total_cost)
                         (h_num_cats : michael_num_cats num_cats)
                         (h_cost_per_animal : michael_cost_per_animal cost_per_animal) :
  num_dogs = 3 :=
by
  sorry

end michael_num_dogs_l141_141976


namespace single_elimination_games_l141_141414

theorem single_elimination_games (n : ℕ) (h : n = 128) : (n - 1) = 127 :=
by
  sorry

end single_elimination_games_l141_141414


namespace remainder_of_sum_l141_141550

theorem remainder_of_sum (k j : ℤ) (a b : ℤ) (h₁ : a = 60 * k + 53) (h₂ : b = 45 * j + 17) : ((a + b) % 15) = 5 :=
by
  sorry

end remainder_of_sum_l141_141550


namespace sine_ratio_triangle_area_l141_141993

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {area : ℝ}

-- Main statement for part 1
theorem sine_ratio 
  (h1 : a * c * Real.cos B - b * c * Real.cos A = 3 * b^2) :
  (Real.sin A / Real.sin B) = Real.sqrt 7 := 
sorry

-- Main statement for part 2
theorem triangle_area 
  (h1 : a * c * Real.cos B - b * c * Real.cos A = 3 * b^2)
  (h2 : c = Real.sqrt 11)
  (h3 : Real.sin C = (2 * Real.sqrt 2)/3)
  (h4 : C < π / 2) :
  area = Real.sqrt 14 :=
sorry

end sine_ratio_triangle_area_l141_141993


namespace triangle_area_and_angle_l141_141172

theorem triangle_area_and_angle (a b c A B C : ℝ) 
  (habc: A + B + C = Real.pi)
  (h1: (2*a + b)*Real.cos C + c*Real.cos B = 0)
  (h2: c = 2*Real.sqrt 6 / 3)
  (h3: Real.sin A * Real.cos B = (Real.sqrt 3 - 1)/4) :
  (C = 2*Real.pi / 3) ∧ (1/2 * b * c * Real.sin A = (6 - 2 * Real.sqrt 3)/9) :=
by
  sorry

end triangle_area_and_angle_l141_141172


namespace enter_exit_ways_eq_sixteen_l141_141098

theorem enter_exit_ways_eq_sixteen (n : ℕ) (h : n = 4) : n * n = 16 :=
by sorry

end enter_exit_ways_eq_sixteen_l141_141098


namespace find_m_plus_n_l141_141000

variable (U : Set ℝ) (A : Set ℝ) (CUA : Set ℝ) (m n : ℝ)
  -- Condition 1: The universal set U is the set of all real numbers
  (hU : U = Set.univ)
  -- Condition 2: A is defined as the set of all x such that (x - 1)(x - m) > 0
  (hA : A = { x : ℝ | (x - 1) * (x - m) > 0 })
  -- Condition 3: The complement of A in U is [-1, -n]
  (hCUA : CUA = { x : ℝ | x ∈ U ∧ x ∉ A } ∧ CUA = Icc (-1) (-n))

theorem find_m_plus_n : m + n = -2 :=
  sorry 

end find_m_plus_n_l141_141000


namespace sin_B_value_triangle_area_l141_141837

-- Problem 1: sine value of angle B given the conditions
theorem sin_B_value (a b c : ℝ) (A B C : ℝ)
  (h1 : 3 * b = 4 * c)
  (h2 : B = 2 * C) :
  Real.sin B = (4 * Real.sqrt 5) / 9 :=
sorry

-- Problem 2: Area of triangle ABC given the conditions and b = 4
theorem triangle_area (a b c : ℝ) (A B C : ℝ)
  (h1 : 3 * b = 4 * c)
  (h2 : B = 2 * C)
  (h3 : b = 4) :
  (1 / 2) * b * c * Real.sin A = (14 * Real.sqrt 5) / 9 :=
sorry

end sin_B_value_triangle_area_l141_141837


namespace reflection_proof_l141_141831

def original_center : (ℝ × ℝ) := (8, -3)
def reflection_line (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)
def reflected_center : (ℝ × ℝ) := reflection_line original_center

theorem reflection_proof : reflected_center = (-3, -8) := by
  sorry

end reflection_proof_l141_141831


namespace find_price_of_each_part_l141_141283

def original_price (total_cost : ℝ) (num_parts : ℕ) (price_per_part : ℝ) :=
  num_parts * price_per_part = total_cost

theorem find_price_of_each_part :
  original_price 439 7 62.71 :=
by
  sorry

end find_price_of_each_part_l141_141283


namespace correct_operation_l141_141043

theorem correct_operation (a : ℝ) : (a^3)^3 = a^9 := 
sorry

end correct_operation_l141_141043


namespace sum_of_relatively_prime_integers_l141_141050

theorem sum_of_relatively_prime_integers (n : ℕ) (h : n ≥ 7) :
  ∃ a b : ℕ, n = a + b ∧ a > 1 ∧ b > 1 ∧ Nat.gcd a b = 1 :=
by
  sorry

end sum_of_relatively_prime_integers_l141_141050


namespace compare_logarithmic_values_l141_141310

theorem compare_logarithmic_values :
  let a := Real.log 3.4 / Real.log 2
  let b := Real.log 3.6 / Real.log 4
  let c := Real.log 0.3 / Real.log 3
  c < b ∧ b < a :=
by
  sorry

end compare_logarithmic_values_l141_141310


namespace volume_of_cuboid_l141_141318

theorem volume_of_cuboid (a b c : ℕ) (h_a : a = 2) (h_b : b = 5) (h_c : c = 8) : 
  a * b * c = 80 := 
by 
  sorry

end volume_of_cuboid_l141_141318


namespace cos_of_theta_cos_double_of_theta_l141_141416

noncomputable def theta : ℝ := sorry -- Placeholder for theta within the interval (0, π/2)
axiom theta_in_range : 0 < theta ∧ theta < Real.pi / 2
axiom sin_theta_eq : Real.sin theta = 1/3

theorem cos_of_theta : Real.cos theta = 2 * Real.sqrt 2 / 3 := by
  sorry

theorem cos_double_of_theta : Real.cos (2 * theta) = 7 / 9 := by
  sorry

end cos_of_theta_cos_double_of_theta_l141_141416


namespace sachin_rahul_age_ratio_l141_141818

theorem sachin_rahul_age_ratio 
(S_age : ℕ) 
(R_age : ℕ) 
(h1 : R_age = S_age + 4) 
(h2 : S_age = 14) : 
S_age / Int.gcd S_age R_age = 7 ∧ R_age / Int.gcd S_age R_age = 9 := 
by 
sorry

end sachin_rahul_age_ratio_l141_141818


namespace cucumber_kinds_l141_141885

theorem cucumber_kinds (x : ℕ) :
  (3 * 5) + (4 * x) + 30 + 85 = 150 → x = 5 :=
by
  intros h
  -- h : 15 + 4 * x + 30 + 85 = 150 

  -- Proof would go here
  sorry

end cucumber_kinds_l141_141885


namespace well_rate_correct_l141_141467

noncomputable def well_rate (depth : ℝ) (diameter : ℝ) (total_cost : ℝ) : ℝ :=
  let r := diameter / 2
  let volume := Real.pi * r^2 * depth
  total_cost / volume

theorem well_rate_correct :
  well_rate 14 3 1583.3626974092558 = 15.993 :=
by
  sorry

end well_rate_correct_l141_141467


namespace ad_minus_bc_divisible_by_2017_l141_141427

theorem ad_minus_bc_divisible_by_2017 
  (a b c d n : ℕ) 
  (h1 : (a * n + b) % 2017 = 0) 
  (h2 : (c * n + d) % 2017 = 0) : 
  (a * d - b * c) % 2017 = 0 :=
sorry

end ad_minus_bc_divisible_by_2017_l141_141427


namespace linear_function_does_not_pass_through_quadrant_3_l141_141390

theorem linear_function_does_not_pass_through_quadrant_3
  (f : ℝ → ℝ) (h : ∀ x, f x = -3 * x + 5) :
  ¬ (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ f x = y) :=
by
  sorry

end linear_function_does_not_pass_through_quadrant_3_l141_141390


namespace percentage_decrease_l141_141576

-- Define the initial conditions
def total_cans : ℕ := 600
def initial_people : ℕ := 40
def new_total_cans : ℕ := 420

-- Use the conditions to define the resulting quantities
def cans_per_person : ℕ := total_cans / initial_people
def new_people : ℕ := new_total_cans / cans_per_person

-- Prove the percentage decrease in the number of people
theorem percentage_decrease :
  let original_people := initial_people
  let new_people := new_people
  let decrease := original_people - new_people
  let percentage_decrease := (decrease * 100) / original_people
  percentage_decrease = 30 :=
by
  sorry

end percentage_decrease_l141_141576


namespace problem_bound_l141_141056

theorem problem_bound (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by 
  sorry

end problem_bound_l141_141056


namespace range_of_z_minus_x_z_minus_y_l141_141257

theorem range_of_z_minus_x_z_minus_y (x y z : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z) (h_sum : x + y + z = 1) :
  -1 / 8 ≤ (z - x) * (z - y) ∧ (z - x) * (z - y) ≤ 1 := by
  sorry

end range_of_z_minus_x_z_minus_y_l141_141257


namespace find_a_l141_141645

theorem find_a (a x : ℝ) (h1 : 3 * a - x = x / 2 + 3) (h2 : x = 2) : a = 2 := 
by
  sorry

end find_a_l141_141645


namespace sample_size_is_59_l141_141533

def totalStudents : Nat := 295
def samplingRatio : Nat := 5

theorem sample_size_is_59 : totalStudents / samplingRatio = 59 := 
by
  sorry

end sample_size_is_59_l141_141533


namespace quadratic_solution_l141_141036

theorem quadratic_solution :
  ∀ x : ℝ, (3 * x - 1) * (2 * x + 4) = 1 ↔ x = (-5 + Real.sqrt 55) / 6 ∨ x = (-5 - Real.sqrt 55) / 6 :=
by
  sorry

end quadratic_solution_l141_141036


namespace base_for_784_as_CDEC_l141_141862

theorem base_for_784_as_CDEC : 
  ∃ (b : ℕ), 
  (b^3 ≤ 784 ∧ 784 < b^4) ∧ 
  (∃ C D : ℕ, C ≠ D ∧ 784 = (C * b^3 + D * b^2 + C * b + C) ∧ 
  b = 6) :=
sorry

end base_for_784_as_CDEC_l141_141862


namespace parabola_solution_l141_141170

theorem parabola_solution (a b : ℝ) : 
  (∃ y : ℝ, y = 2^2 + 2 * a + b ∧ y = 20) ∧ 
  (∃ y : ℝ, y = (-2)^2 + (-2) * a + b ∧ y = 0) ∧ 
  b = (0^2 + 0 * a + b) → 
  a = 5 ∧ b = 6 := 
by {
  sorry
}

end parabola_solution_l141_141170


namespace planes_formed_through_three_lines_l141_141106

theorem planes_formed_through_three_lines (L1 L2 L3 : ℝ × ℝ × ℝ → Prop) (P : ℝ × ℝ × ℝ) :
  (∀ (x : ℝ × ℝ × ℝ), L1 x → L2 x → L3 x → x = P) →
  (∃ n : ℕ, n = 1 ∨ n = 3) :=
sorry

end planes_formed_through_three_lines_l141_141106


namespace problem_l141_141598

open Set

-- Definitions for set A and set B
def setA : Set ℝ := { x | x^2 + 2 * x - 3 < 0 }
def setB : Set ℤ := { k : ℤ | true }
def evenIntegers : Set ℝ := { x : ℝ | ∃ k : ℤ, x = 2 * k }

-- The intersection of set A and even integers over ℝ
def A_cap_B : Set ℝ := setA ∩ evenIntegers

-- The Proposition that A_cap_B equals {-2, 0}
theorem problem : A_cap_B = ({-2, 0} : Set ℝ) :=
by 
  sorry

end problem_l141_141598


namespace range_of_a_l141_141833

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), (x^2 - 2*a*x + a) > 0) → (a ≤ 0 ∨ a ≥ 1) :=
by
  -- Proof goes here
  sorry

end range_of_a_l141_141833


namespace range_of_m_l141_141080

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 2^|x| + m = 0) → m ≤ -1 :=
by
  sorry

end range_of_m_l141_141080


namespace length_segment_l141_141304

/--
Given a cylinder with a radius of 5 units capped with hemispheres at each end and having a total volume of 900π,
prove that the length of the line segment AB is 88/3 units.
-/
theorem length_segment (r : ℝ) (V : ℝ) (h : ℝ) : r = 5 ∧ V = 900 * Real.pi → h = 88 / 3 := by
  sorry

end length_segment_l141_141304


namespace maximum_food_per_guest_l141_141235

theorem maximum_food_per_guest (total_food : ℕ) (min_guests : ℕ) (total_food_eq : total_food = 337) (min_guests_eq : min_guests = 169) :
  ∃ max_food_per_guest, max_food_per_guest = total_food / min_guests ∧ max_food_per_guest = 2 := 
by
  sorry

end maximum_food_per_guest_l141_141235


namespace mike_bricks_l141_141368

theorem mike_bricks (total_bricks bricks_A bricks_B bricks_other: ℕ) 
  (h1 : bricks_A = 40) 
  (h2 : bricks_B = bricks_A / 2)
  (h3 : total_bricks = 150) 
  (h4 : total_bricks = bricks_A + bricks_B + bricks_other) : bricks_other = 90 := 
by 
  sorry

end mike_bricks_l141_141368


namespace fraction_identity_l141_141020

theorem fraction_identity
  (x w y z : ℝ)
  (hxw_pos : x * w > 0)
  (hyz_pos : y * z > 0)
  (hxw_inv_sum : 1 / x + 1 / w = 20)
  (hyz_inv_sum : 1 / y + 1 / z = 25)
  (hxw_inv : 1 / (x * w) = 6)
  (hyz_inv : 1 / (y * z) = 8) :
  (x + y) / (z + w) = 155 / 7 :=
by
  -- proof omitted
  sorry

end fraction_identity_l141_141020


namespace unique_integer_in_ranges_l141_141689

theorem unique_integer_in_ranges {x : ℤ} :
  1 < x ∧ x < 9 → 
  2 < x ∧ x < 15 → 
  -1 < x ∧ x < 7 → 
  0 < x ∧ x < 4 → 
  x + 1 < 5 → 
  x = 3 := by
  intros _ _ _ _ _
  sorry

end unique_integer_in_ranges_l141_141689


namespace find_percentage_of_alcohol_l141_141148

theorem find_percentage_of_alcohol 
  (Vx : ℝ) (Px : ℝ) (Vy : ℝ) (Py : ℝ) (Vp : ℝ) (Pp : ℝ)
  (hx : Px = 10) (hvx : Vx = 300) (hvy : Vy = 100) (hvxy : Vx + Vy = 400) (hpxy : Pp = 15) :
  (Vy * Py / 100) = 30 :=
by
  sorry

end find_percentage_of_alcohol_l141_141148


namespace molecular_weight_l141_141232

theorem molecular_weight (w8 : ℝ) (n : ℝ) (w1 : ℝ) (h1 : w8 = 2376) (h2 : n = 8) : w1 = 297 :=
by
  sorry

end molecular_weight_l141_141232


namespace largest_value_l141_141273

def value (word : List Char) : Nat :=
  word.foldr (fun c acc =>
    acc + match c with
      | 'A' => 1
      | 'B' => 2
      | 'C' => 3
      | 'D' => 4
      | 'E' => 5
      | _ => 0
    ) 0

theorem largest_value :
  value ['B', 'E', 'E'] > value ['D', 'A', 'D'] ∧
  value ['B', 'E', 'E'] > value ['B', 'A', 'D'] ∧
  value ['B', 'E', 'E'] > value ['C', 'A', 'B'] ∧
  value ['B', 'E', 'E'] > value ['B', 'E', 'D'] :=
by sorry

end largest_value_l141_141273


namespace jungkook_red_balls_l141_141937

-- Definitions from conditions
def num_boxes : ℕ := 2
def red_balls_per_box : ℕ := 3

-- Theorem stating the problem
theorem jungkook_red_balls : (num_boxes * red_balls_per_box) = 6 :=
by sorry

end jungkook_red_balls_l141_141937


namespace parametric_line_eq_l141_141567

theorem parametric_line_eq (t : ℝ) : 
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x = 3 * t + 6 → y = 5 * t - 8 → y = m * x + b)) ∧ m = 5 / 3 ∧ b = -18 :=
sorry

end parametric_line_eq_l141_141567


namespace led_message_count_l141_141344

theorem led_message_count : 
  let n := 7
  let colors := 2
  let lit_leds := 3
  let non_adjacent_combinations := 10
  (non_adjacent_combinations * (colors ^ lit_leds)) = 80 :=
by
  sorry

end led_message_count_l141_141344


namespace arithmetic_progression_implies_equality_l141_141372

theorem arithmetic_progression_implies_equality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ((a + b) / 2) = ((Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2) → a = b :=
by
  sorry

end arithmetic_progression_implies_equality_l141_141372


namespace inequality_solution_l141_141867

theorem inequality_solution {x : ℝ} : 5 * x^2 + 7 * x > 3 ↔ x < -1 ∨ x > 3/5 := by
  sorry

end inequality_solution_l141_141867


namespace area_of_triangle_formed_by_tangent_line_l141_141113

noncomputable def curve (x : ℝ) : ℝ := Real.log x - 2 * x

noncomputable def slope_of_tangent_at (x : ℝ) : ℝ := (1 / x) - 2

def point_of_tangency : ℝ × ℝ := (1, -2)

-- Define the tangent line equation at the point (1, -2)
noncomputable def tangent_line (x : ℝ) : ℝ := -x - 1

-- Define x and y intercepts of the tangent line
def x_intercept_of_tangent : ℝ := -1
def y_intercept_of_tangent : ℝ := -1

-- Define the area of the triangle formed by the tangent line and the coordinate axes
def triangle_area : ℝ := 0.5 * (-1) * (-1)

-- State the theorem to prove the area of the triangle
theorem area_of_triangle_formed_by_tangent_line : 
  triangle_area = 0.5 := by 
sorry

end area_of_triangle_formed_by_tangent_line_l141_141113


namespace sixth_grader_count_l141_141771

theorem sixth_grader_count : 
  ∃ x y : ℕ, (3 / 7) * x = (1 / 3) * y ∧ x + y = 140 ∧ x = 61 :=
by {
  sorry  -- Proof not required
}

end sixth_grader_count_l141_141771


namespace atomic_number_cannot_be_x_plus_4_l141_141182

-- Definitions for atomic numbers and elements in the same main group
def in_same_main_group (A B : Type) (atomic_num_A atomic_num_B : ℕ) : Prop :=
  atomic_num_B ≠ atomic_num_A + 4

-- Noncomputable definition is likely needed as the problem involves non-algorithmic aspects.
noncomputable def periodic_table_condition (A B : Type) (x : ℕ) : Prop :=
  in_same_main_group A B x (x + 4)

-- Main theorem stating the mathematical proof problem
theorem atomic_number_cannot_be_x_plus_4
  (A B : Type)
  (x : ℕ)
  (h : periodic_table_condition A B x) : false :=
  by
    sorry

end atomic_number_cannot_be_x_plus_4_l141_141182


namespace complex_fraction_simplification_l141_141493

open Complex

theorem complex_fraction_simplification : (1 + 2 * I) / (1 - I)^2 = 1 - 1 / 2 * I :=
by
  -- Proof omitted
  sorry

end complex_fraction_simplification_l141_141493


namespace a2009_equals_7_l141_141653

def sequence_element (n k : ℕ) : ℚ :=
  if k = 0 then 0 else (n - k + 1) / k

def cumulative_count (n : ℕ) : ℕ := n * (n + 1) / 2

theorem a2009_equals_7 : 
  let n := 63
  let m := 2009
  let subset_cumulative_count := cumulative_count n
  (2 * m = n * (n + 1) - 14 ∧
   m = subset_cumulative_count - 7 ∧ 
   sequence_element n 8 = 7) →
  sequence_element n (subset_cumulative_count - m + 1) = 7 :=
by
  -- proof steps to be filled here
  sorry

end a2009_equals_7_l141_141653


namespace correct_mean_after_correction_l141_141812

theorem correct_mean_after_correction
  (n : ℕ) (incorrect_mean : ℝ) (incorrect_value : ℝ) (correct_value : ℝ)
  (h : n = 30) (h_mean : incorrect_mean = 150) (h_incorrect_value : incorrect_value = 135) (h_correct_value : correct_value = 165) :
  (incorrect_mean * n - incorrect_value + correct_value) / n = 151 :=
  by
  sorry

end correct_mean_after_correction_l141_141812


namespace martha_flower_cost_l141_141033

theorem martha_flower_cost :
  let roses_per_centerpiece := 8
  let orchids_per_centerpiece := 2 * roses_per_centerpiece
  let lilies_per_centerpiece := 6
  let centerpieces := 6
  let cost_per_flower := 15
  let total_roses := roses_per_centerpiece * centerpieces
  let total_orchids := orchids_per_centerpiece * centerpieces
  let total_lilies := lilies_per_centerpiece * centerpieces
  let total_flowers := total_roses + total_orchids + total_lilies
  let total_cost := total_flowers * cost_per_flower
  total_cost = 2700 :=
by
  let roses_per_centerpiece := 8
  let orchids_per_centerpiece := 2 * roses_per_centerpiece
  let lilies_per_centerpiece := 6
  let centerpieces := 6
  let cost_per_flower := 15
  let total_roses := roses_per_centerpiece * centerpieces
  let total_orchids := orchids_per_centerpiece * centerpieces
  let total_lilies := lilies_per_centerpiece * centerpieces
  let total_flowers := total_roses + total_orchids + total_lilies
  let total_cost := total_flowers * cost_per_flower
  -- Proof to be added here
  sorry

end martha_flower_cost_l141_141033


namespace river_flow_rate_l141_141734

noncomputable def volume_per_minute : ℝ := 3200
noncomputable def depth_of_river : ℝ := 3
noncomputable def width_of_river : ℝ := 32
noncomputable def cross_sectional_area : ℝ := depth_of_river * width_of_river

noncomputable def flow_rate_m_per_minute : ℝ := volume_per_minute / cross_sectional_area
-- Conversion factors
noncomputable def minutes_per_hour : ℝ := 60
noncomputable def meters_per_km : ℝ := 1000

noncomputable def flow_rate_kmph : ℝ := (flow_rate_m_per_minute * minutes_per_hour) / meters_per_km

theorem river_flow_rate :
  flow_rate_kmph = 2 :=
by
  -- We skip the proof and use sorry to focus on the statement structure.
  sorry

end river_flow_rate_l141_141734


namespace cost_per_serving_l141_141362

-- Define the costs
def pasta_cost : ℝ := 1.00
def sauce_cost : ℝ := 2.00
def meatball_cost : ℝ := 5.00

-- Define the number of servings
def servings : ℝ := 8.0

-- State the theorem
theorem cost_per_serving : (pasta_cost + sauce_cost + meatball_cost) / servings = 1.00 :=
by
  sorry

end cost_per_serving_l141_141362


namespace factor_expression_l141_141809

variable (a : ℝ)

theorem factor_expression : 45 * a^2 + 135 * a + 90 = 45 * a * (a + 5) :=
by
  sorry

end factor_expression_l141_141809


namespace squares_centers_equal_perpendicular_l141_141967

def Square (center : (ℝ × ℝ)) (side : ℝ) := {p : ℝ × ℝ // abs (p.1 - center.1) ≤ side / 2 ∧ abs (p.2 - center.2) ≤ side / 2}

theorem squares_centers_equal_perpendicular 
  (a b : ℝ)
  (O A B C : ℝ × ℝ)
  (hA : A = (a, a))
  (hB : B = (b, 2 * a + b))
  (hC : C = (- (a + b), a + b))
  (hO_vertex : O = (0, 0)) :
  dist O B = dist A C ∧ ∃ m₁ m₂ : ℝ, (B.2 - O.2) / (B.1 - O.1) = m₁ ∧ (C.2 - A.2) / (C.1 - A.1) = m₂ ∧ m₁ * m₂ = -1 := sorry

end squares_centers_equal_perpendicular_l141_141967


namespace days_C_alone_l141_141930

theorem days_C_alone (r_A r_B r_C : ℝ) (h1 : r_A + r_B = 1 / 3) (h2 : r_B + r_C = 1 / 6) (h3 : r_A + r_C = 5 / 18) : 
  1 / r_C = 18 := 
  sorry

end days_C_alone_l141_141930


namespace four_x_sq_plus_nine_y_sq_l141_141765

theorem four_x_sq_plus_nine_y_sq (x y : ℝ) 
  (h1 : 2 * x + 3 * y = 9)
  (h2 : x * y = -12) : 
  4 * x^2 + 9 * y^2 = 225 := 
by
  sorry

end four_x_sq_plus_nine_y_sq_l141_141765


namespace track_length_l141_141090

theorem track_length
  (meet1_dist : ℝ)
  (meet2_sally_additional_dist : ℝ)
  (constant_speed : ∀ (b_speed s_speed : ℝ), b_speed = s_speed)
  (opposite_start : true)
  (brenda_first_meet : meet1_dist = 100)
  (sally_second_meet : meet2_sally_additional_dist = 200) :
  ∃ L : ℝ, L = 200 :=
by
  sorry

end track_length_l141_141090


namespace tan_sum_identity_sin_2alpha_l141_141311

theorem tan_sum_identity_sin_2alpha (α : ℝ) (h : Real.tan (π/4 + α) = 2) : Real.sin (2*α) = 3/5 :=
by
  sorry

end tan_sum_identity_sin_2alpha_l141_141311


namespace simplify_expression_l141_141086

theorem simplify_expression (w : ℝ) : 2 * w + 3 - 4 * w - 5 + 6 * w + 7 - 8 * w - 9 = -4 * w - 4 :=
by
  -- Proof steps would go here
  sorry

end simplify_expression_l141_141086


namespace max_knights_seated_l141_141956

theorem max_knights_seated (total_islanders : ℕ) (half_islanders : ℕ) 
  (knight_statement_half : ℕ) (liar_statement_half : ℕ) :
  total_islanders = 100 ∧ knight_statement_half = 50 
    ∧ liar_statement_half = 50 
    ∧ (∀ (k : ℕ), (knight_statement_half = k ∧ liar_statement_half = k)
    → (k ≤ 67)) →
  ∃ K : ℕ, K ≤ 67 :=
by
  -- the proof goes here
  sorry

end max_knights_seated_l141_141956


namespace conic_section_is_hyperbola_l141_141638

noncomputable def is_hyperbola (x y : ℝ) : Prop :=
  (x - 4) ^ 2 = 9 * (y + 3) ^ 2 + 27

theorem conic_section_is_hyperbola : ∀ x y : ℝ, is_hyperbola x y → "H" = "H" := sorry

end conic_section_is_hyperbola_l141_141638


namespace greatest_possible_value_x_l141_141377

theorem greatest_possible_value_x :
  ∀ x : ℚ, (∃ y : ℚ, y = (5 * x - 25) / (4 * x - 5) ∧ y^2 + y = 18) →
  x ≤ 55 / 29 :=
by sorry

end greatest_possible_value_x_l141_141377


namespace arithmetic_sequence_sum_l141_141932

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic property of the sequence
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (h1 : is_arithmetic_sequence a d)
  (h2 : a 2 + a 4 + a 7 + a 11 = 44) :
  a 3 + a 5 + a 10 = 33 := 
sorry

end arithmetic_sequence_sum_l141_141932


namespace find_n_l141_141375

theorem find_n :
  ∃ (n : ℤ), (4 ≤ n ∧ n ≤ 8) ∧ (n % 5 = 2) ∧ (n = 7) :=
by
  sorry

end find_n_l141_141375


namespace height_at_2_years_l141_141105

variable (height : ℕ → ℕ) -- height function representing the height of the tree at the end of n years
variable (triples_height : ∀ n, height (n + 1) = 3 * height n) -- tree triples its height every year
variable (height_4 : height 4 = 81) -- height at the end of 4 years is 81 feet

-- We need the height at the end of 2 years
theorem height_at_2_years : height 2 = 9 :=
by {
  sorry
}

end height_at_2_years_l141_141105


namespace roots_in_interval_l141_141528

theorem roots_in_interval (P : Polynomial ℝ) (h : ∀ i, P.coeff i = 1 ∨ P.coeff i = 0 ∨ P.coeff i = -1) : 
  ∀ x : ℝ, P.eval x = 0 → -2 ≤ x ∧ x ≤ 2 :=
by {
  -- Proof omitted
  sorry
}

end roots_in_interval_l141_141528


namespace complex_number_quadrant_l141_141285

def i := Complex.I
def z := i * (1 + i)

theorem complex_number_quadrant 
  : z.re < 0 ∧ z.im > 0 := 
by
  sorry

end complex_number_quadrant_l141_141285


namespace solve_quadratic_1_solve_quadratic_2_l141_141655

theorem solve_quadratic_1 (x : ℝ) : 3 * x^2 - 8 * x + 4 = 0 ↔ x = 2/3 ∨ x = 2 := by
  sorry

theorem solve_quadratic_2 (x : ℝ) : (2 * x - 1)^2 = (x - 3)^2 ↔ x = 4/3 ∨ x = -2 := by
  sorry

end solve_quadratic_1_solve_quadratic_2_l141_141655


namespace solve_for_a_l141_141307

noncomputable def a_value (a x : ℝ) : Prop :=
  (3 / 10) * a + (2 * x + 4) / 2 = 4 * (x - 1)

theorem solve_for_a (a : ℝ) : a_value a 3 → a = 10 :=
by
  sorry

end solve_for_a_l141_141307


namespace inequality_solution_l141_141185

theorem inequality_solution (x : ℝ) : x ^ 2 < |x| + 2 ↔ -2 < x ∧ x < 2 :=
by
  sorry

end inequality_solution_l141_141185


namespace option_D_correct_l141_141861

theorem option_D_correct (f : ℕ+ → ℕ) (h : ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2) 
  (hf : f 4 ≥ 25) : ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 :=
by
  sorry

end option_D_correct_l141_141861


namespace farm_horse_food_needed_l141_141481

-- Definitions given in the problem
def sheep_count : ℕ := 16
def sheep_to_horse_ratio : ℕ × ℕ := (2, 7)
def food_per_horse_per_day : ℕ := 230

-- The statement we want to prove
theorem farm_horse_food_needed : 
  ∃ H : ℕ, (sheep_count * sheep_to_horse_ratio.2 = sheep_to_horse_ratio.1 * H) ∧ 
           (H * food_per_horse_per_day = 12880) :=
sorry

end farm_horse_food_needed_l141_141481


namespace b_investment_l141_141219

theorem b_investment (x : ℝ) (total_profit A_investment B_investment C_investment A_profit: ℝ)
  (h1 : A_investment = 6300)
  (h2 : B_investment = x)
  (h3 : C_investment = 10500)
  (h4 : total_profit = 12600)
  (h5 : A_profit = 3780)
  (ratio_eq : (A_investment / (A_investment + B_investment + C_investment)) = (A_profit / total_profit)) :
  B_investment = 13700 :=
  sorry

end b_investment_l141_141219


namespace union_A_B_l141_141898

def setA : Set ℝ := { x | Real.log x / Real.log (1/2) > -1 }
def setB : Set ℝ := { x | 2^x > Real.sqrt 2 }

theorem union_A_B : setA ∪ setB = { x | 0 < x } := by
  sorry

end union_A_B_l141_141898


namespace maisy_earnings_increase_l141_141783

-- Define the conditions from the problem
def current_job_hours_per_week : ℕ := 8
def current_job_wage_per_hour : ℕ := 10

def new_job_hours_per_week : ℕ := 4
def new_job_wage_per_hour : ℕ := 15
def new_job_bonus_per_week : ℕ := 35

-- Define the weekly earnings calculations
def current_job_earnings : ℕ := current_job_hours_per_week * current_job_wage_per_hour
def new_job_earnings_without_bonus : ℕ := new_job_hours_per_week * new_job_wage_per_hour
def new_job_earnings_with_bonus : ℕ := new_job_earnings_without_bonus + new_job_bonus_per_week

-- Define the difference in earnings
def earnings_difference : ℕ := new_job_earnings_with_bonus - current_job_earnings

-- The theorem to prove: Maisy will earn $15 more per week at her new job
theorem maisy_earnings_increase : earnings_difference = 15 := by
  sorry

end maisy_earnings_increase_l141_141783


namespace intersection_A_B_l141_141434

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | x^2 - 2 * x < 0 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 2 } :=
by
  -- We are going to skip the proof for now
  sorry

end intersection_A_B_l141_141434


namespace train_length_correct_l141_141600

open Real

-- Define the conditions
def bridge_length : ℝ := 150
def time_to_cross_bridge : ℝ := 7.5
def time_to_cross_lamp_post : ℝ := 2.5

-- Define the length of the train
def train_length : ℝ := 75

theorem train_length_correct :
  ∃ L : ℝ, (L / time_to_cross_lamp_post = (L + bridge_length) / time_to_cross_bridge) ∧ L = train_length :=
by
  sorry

end train_length_correct_l141_141600


namespace bridge_length_increase_l141_141475

open Real

def elevation_change : ℝ := 800
def original_gradient : ℝ := 0.02
def new_gradient : ℝ := 0.015

theorem bridge_length_increase :
  let original_length := elevation_change / original_gradient
  let new_length := elevation_change / new_gradient
  new_length - original_length = 13333 := by
  sorry

end bridge_length_increase_l141_141475


namespace g_f_neg3_eq_1741_l141_141270

def f (x : ℤ) : ℤ := x^3 - 3
def g (x : ℤ) : ℤ := 2*x^2 + 2*x + 1

theorem g_f_neg3_eq_1741 : g (f (-3)) = 1741 := 
by 
  sorry

end g_f_neg3_eq_1741_l141_141270


namespace pascal_triangle_45th_number_l141_141585

theorem pascal_triangle_45th_number (n k : ℕ) (h1 : n = 47) (h2 : k = 44) : 
  Nat.choose (n - 1) k = 1035 :=
by
  sorry

end pascal_triangle_45th_number_l141_141585


namespace altitude_correct_l141_141365

-- Define the given sides and area of the triangle
def AB : ℝ := 30
def BC : ℝ := 17
def AC : ℝ := 25
def area_ABC : ℝ := 120

-- The length of the altitude from the vertex C to the base AB
def height_C_to_AB : ℝ := 8

-- Problem statement to be proven
theorem altitude_correct : (1 / 2) * AB * height_C_to_AB = area_ABC :=
by
  sorry

end altitude_correct_l141_141365


namespace geometric_sequence_a_5_l141_141959

noncomputable def a_n : ℕ → ℝ := sorry

theorem geometric_sequence_a_5 :
  (∀ n : ℕ, ∃ r : ℝ, a_n (n + 1) = r * a_n n) →  -- geometric sequence property
  (∃ x₁ x₂ : ℝ, x₁ + x₂ = -7 ∧ x₁ * x₂ = 9 ∧ a_n 3 = x₁ ∧ a_n 7 = x₂) →  -- roots of the quadratic equation and their assignments
  a_n 5 = -3 := sorry

end geometric_sequence_a_5_l141_141959


namespace hotel_charge_comparison_l141_141605

def charge_R (R G : ℝ) (P : ℝ) : Prop :=
  P = 0.8 * R ∧ P = 0.9 * G

def discounted_charge_R (R2 : ℝ) (R : ℝ) : Prop :=
  R2 = 0.85 * R

theorem hotel_charge_comparison (R G P R2 : ℝ)
  (h1 : charge_R R G P)
  (h2 : discounted_charge_R R2 R)
  (h3 : R = 1.125 * G) :
  R2 = 0.95625 * G := by
  sorry

end hotel_charge_comparison_l141_141605


namespace gcd_of_lcm_and_ratio_l141_141970

theorem gcd_of_lcm_and_ratio (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 2) : Nat.gcd X Y = 18 :=
sorry

end gcd_of_lcm_and_ratio_l141_141970


namespace number_of_crowns_l141_141828

-- Define the conditions
def feathers_per_crown : ℕ := 7
def total_feathers : ℕ := 6538

-- Theorem statement
theorem number_of_crowns : total_feathers / feathers_per_crown = 934 :=
by {
  sorry  -- proof omitted
}

end number_of_crowns_l141_141828


namespace national_flag_length_l141_141742

-- Definitions from the conditions specified in the problem
def width : ℕ := 128
def ratio_length_to_width (L W : ℕ) : Prop := L / W = 3 / 2

-- The main theorem to prove
theorem national_flag_length (L : ℕ) (H : ratio_length_to_width L width) : L = 192 :=
by
  sorry

end national_flag_length_l141_141742


namespace condition_for_ellipse_l141_141712

theorem condition_for_ellipse (m : ℝ) : 
  (3 < m ∧ m < 7) ↔ (7 - m > 0 ∧ m - 3 > 0 ∧ (7 - m) ≠ (m - 3)) :=
by sorry

end condition_for_ellipse_l141_141712


namespace enthalpy_change_l141_141189

def DeltaH_prods : Float := -286.0 - 297.0
def DeltaH_reacts : Float := -20.17
def HessLaw (DeltaH_prods DeltaH_reacts : Float) : Float := DeltaH_prods - DeltaH_reacts

theorem enthalpy_change : HessLaw DeltaH_prods DeltaH_reacts = -1125.66 := by
  -- Lean needs a proof, which is not needed per instructions
  sorry

end enthalpy_change_l141_141189


namespace no_solution_inequality_l141_141572

theorem no_solution_inequality (a : ℝ) : (∀ x : ℝ, ¬(|x - 3| + |x - a| < 1)) ↔ (a ≤ 2 ∨ a ≥ 4) := 
sorry

end no_solution_inequality_l141_141572


namespace roots_of_transformed_quadratic_l141_141385

theorem roots_of_transformed_quadratic (a b p q s1 s2 : ℝ)
    (h_quad_eq : s1 ^ 2 + a * s1 + b = 0 ∧ s2 ^ 2 + a * s2 + b = 0)
    (h_sum_roots : s1 + s2 = -a)
    (h_prod_roots : s1 * s2 = b) :
        p = -(a ^ 4 - 4 * a ^ 2 * b + 2 * b ^ 2) ∧ 
        q = b ^ 4 :=
by
  sorry

end roots_of_transformed_quadratic_l141_141385


namespace linear_eq_k_l141_141195

theorem linear_eq_k (k : ℝ) : (k - 3) * x ^ (|k| - 2) + 5 = k - 4 → |k| = 3 → k ≠ 3 → k = -3 :=
by
  intros h1 h2 h3
  sorry

end linear_eq_k_l141_141195


namespace arithmetic_sequence_a3_l141_141644

theorem arithmetic_sequence_a3 (a1 a2 a3 a4 a5 : ℝ) 
  (h1 : a2 = a1 + (a1 + a5 - a1) / 4)
  (h2 : a3 = a1 + 2 * (a1 + a5 - a1) / 4) 
  (h3 : a4 = a1 + 3 * (a1 + a5 - a1) / 4) 
  (h4 : a5 = a1 + 4 * (a1 + a5 - a1) / 4)
  (h_sum : 5 * a3 = 15) : 
  a3 = 3 :=
sorry

end arithmetic_sequence_a3_l141_141644


namespace largest_angle_in_ratio_triangle_l141_141627

theorem largest_angle_in_ratio_triangle (a b c : ℕ) (h_ratios : 2 * c = 3 * b ∧ 3 * b = 4 * a)
  (h_sum : a + b + c = 180) : max a (max b c) = 80 :=
by
  sorry

end largest_angle_in_ratio_triangle_l141_141627


namespace girls_in_blue_dresses_answered_affirmatively_l141_141962

theorem girls_in_blue_dresses_answered_affirmatively :
  ∃ (n : ℕ), n = 17 ∧
  ∀ (total_girls red_dresses blue_dresses answer_girls : ℕ),
  total_girls = 30 →
  red_dresses = 13 →
  blue_dresses = 17 →
  answer_girls = n →
  answer_girls = blue_dresses :=
sorry

end girls_in_blue_dresses_answered_affirmatively_l141_141962


namespace arith_seq_seventh_term_l141_141824

theorem arith_seq_seventh_term (a1 a25 : ℝ) (n : ℕ) (d : ℝ) (a7 : ℝ) :
  a1 = 5 → a25 = 80 → n = 25 → d = (a25 - a1) / (n - 1) → a7 = a1 + (7 - 1) * d → a7 = 23.75 :=
by
  intros h1 h2 h3 hd ha7
  sorry

end arith_seq_seventh_term_l141_141824


namespace tax_difference_is_250000_l141_141142

noncomputable def old_tax_rate : ℝ := 0.20
noncomputable def new_tax_rate : ℝ := 0.30
noncomputable def old_income : ℝ := 1000000
noncomputable def new_income : ℝ := 1500000
noncomputable def old_taxes_paid := old_tax_rate * old_income
noncomputable def new_taxes_paid := new_tax_rate * new_income
noncomputable def tax_difference := new_taxes_paid - old_taxes_paid

theorem tax_difference_is_250000 : tax_difference = 250000 := by
  sorry

end tax_difference_is_250000_l141_141142


namespace correct_multiplication_result_l141_141699

theorem correct_multiplication_result (x : ℕ) (h : 9 * x = 153) : 6 * x = 102 :=
by {
  -- We would normally provide a detailed proof here, but as per instruction, we add sorry.
  sorry
}

end correct_multiplication_result_l141_141699


namespace abel_arrival_earlier_l141_141413

variable (distance : ℕ) (speed_abel : ℕ) (speed_alice : ℕ) (start_delay_alice : ℕ)

theorem abel_arrival_earlier (h_dist : distance = 1000) 
                             (h_speed_abel : speed_abel = 50) 
                             (h_speed_alice : speed_alice = 40) 
                             (h_start_delay : start_delay_alice = 1) : 
                             (start_delay_alice + distance / speed_alice) * 60 - (distance / speed_abel) * 60 = 360 :=
by
  sorry

end abel_arrival_earlier_l141_141413


namespace cherries_initially_l141_141137

theorem cherries_initially (x : ℕ) (h₁ : x - 6 = 10) : x = 16 :=
by
  sorry

end cherries_initially_l141_141137


namespace min_length_GH_l141_141067

theorem min_length_GH :
  let ellipse (x y : ℝ) := (x^2 / 4) + y^2 = 1
  let A := (-2, 0)
  let B := (2, 0)
  ∀ P G H : ℝ × ℝ,
    (P.1^2 / 4 + P.2^2 = 1) →
    P.2 > 0 →
    (G.2 = 3) →
    (H.2 = 3) →
    ∃ k : ℝ, k > 0 ∧ G.1 = 3 / k - 2 ∧ H.1 = -12 * k + 2 →
    |G.1 - H.1| = 8 :=
sorry

end min_length_GH_l141_141067


namespace no_solution_for_x_l141_141753

noncomputable def proof_problem : Prop :=
  ∀ x : ℝ, ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ ≠ 12345

theorem no_solution_for_x : proof_problem :=
  by
    intro x
    sorry

end no_solution_for_x_l141_141753


namespace num_green_hats_l141_141741

-- Definitions
def total_hats : ℕ := 85
def blue_hat_cost : ℕ := 6
def green_hat_cost : ℕ := 7
def total_cost : ℕ := 548

-- Prove the number of green hats (g) is 38 given the conditions
theorem num_green_hats (b g : ℕ) 
  (h₁ : b + g = total_hats)
  (h₂ : blue_hat_cost * b + green_hat_cost * g = total_cost) : 
  g = 38 := by
  sorry

end num_green_hats_l141_141741


namespace average_distance_scientific_notation_l141_141608

theorem average_distance_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ a * 10 ^ n = 384000000 ∧ a = 3.84 ∧ n = 8 :=
sorry

end average_distance_scientific_notation_l141_141608


namespace complement_P_eq_Ioo_l141_141294

def U : Set ℝ := Set.univ
def P : Set ℝ := { x | x^2 - 5 * x - 6 ≥ 0 }
def complement_of_P_in_U : Set ℝ := Set.Ioo (-1) 6

theorem complement_P_eq_Ioo :
  (U \ P) = complement_of_P_in_U :=
by sorry

end complement_P_eq_Ioo_l141_141294


namespace games_did_not_work_l141_141683

theorem games_did_not_work 
  (games_from_friend : ℕ) 
  (games_from_garage_sale : ℕ) 
  (good_games : ℕ) 
  (total_games : ℕ := games_from_friend + games_from_garage_sale) 
  (did_not_work : ℕ := total_games - good_games) :
  games_from_friend = 41 ∧ 
  games_from_garage_sale = 14 ∧ 
  good_games = 24 → 
  did_not_work = 31 := 
by
  intro h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end games_did_not_work_l141_141683


namespace jason_borrowed_amount_l141_141176

theorem jason_borrowed_amount :
  let cycle := [1, 3, 5, 7, 9, 11]
  let total_chores := 48
  let chores_per_cycle := cycle.length
  let earnings_one_cycle := cycle.sum
  let complete_cycles := total_chores / chores_per_cycle
  let total_earnings := complete_cycles * earnings_one_cycle
  total_earnings = 288 :=
by
  sorry

end jason_borrowed_amount_l141_141176


namespace scientific_notation_of_taichulight_performance_l141_141435

noncomputable def trillion := 10^12

def convert_to_scientific_notation (x : ℝ) (n : ℤ) : Prop :=
  1 ≤ x ∧ x < 10 ∧ x * 10^n = 12.5 * trillion

theorem scientific_notation_of_taichulight_performance :
  ∃ (x : ℝ) (n : ℤ), convert_to_scientific_notation x n ∧ x = 1.25 ∧ n = 13 :=
by
  unfold convert_to_scientific_notation
  use 1.25
  use 13
  sorry

end scientific_notation_of_taichulight_performance_l141_141435


namespace initial_marbles_l141_141676

theorem initial_marbles (total_marbles now found: ℕ) (h_found: found = 7) (h_now: now = 28) : 
  total_marbles = now - found → total_marbles = 21 := by
  -- Proof goes here.
  sorry

end initial_marbles_l141_141676


namespace problem1_problem2_l141_141721

-- Assume x and y are positive numbers
variables (x y : ℝ) (hx : 0 < x) (hy : 0 < y)

-- Prove that x^3 + y^3 >= x^2*y + y^2*x
theorem problem1 : x^3 + y^3 ≥ x^2 * y + y^2 * x :=
by sorry

-- Prove that m ≤ 2 given the additional condition
variables (m : ℝ)
theorem problem2 (cond : (x/y^2 + y/x^2) ≥ m/2 * (1/x + 1/y)) : m ≤ 2 :=
by sorry

end problem1_problem2_l141_141721


namespace num_digits_sum_l141_141516

theorem num_digits_sum (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) :
  let num1 := 9643
  let num2 := A * 10 ^ 2 + 7 * 10 + 5
  let num3 := 5 * 10 ^ 2 + B * 10 + 2
  let sum := num1 + num2 + num3
  10^4 ≤ sum ∧ sum < 10^5 :=
by {
  sorry
}

end num_digits_sum_l141_141516


namespace remainder_14_div_5_l141_141793

theorem remainder_14_div_5 : 14 % 5 = 4 := by
  sorry

end remainder_14_div_5_l141_141793


namespace quadratic_inequality_solution_l141_141801

theorem quadratic_inequality_solution :
  { x : ℝ | x^2 + 3 * x - 4 < 0 } = { x : ℝ | -4 < x ∧ x < 1 } :=
by
  sorry

end quadratic_inequality_solution_l141_141801


namespace diego_apples_weight_l141_141779

-- Definitions based on conditions
def bookbag_capacity : ℕ := 20
def weight_watermelon : ℕ := 1
def weight_grapes : ℕ := 1
def weight_oranges : ℕ := 1

-- Lean statement to check
theorem diego_apples_weight : 
  bookbag_capacity - (weight_watermelon + weight_grapes + weight_oranges) = 17 :=
by
  sorry

end diego_apples_weight_l141_141779


namespace Charles_has_13_whistles_l141_141289

-- Conditions
def Sean_whistles : ℕ := 45
def more_whistles_than_Charles : ℕ := 32

-- Let C be the number of whistles Charles has
def C : ℕ := Sean_whistles - more_whistles_than_Charles

-- Theorem to be proven
theorem Charles_has_13_whistles : C = 13 := by
  -- skipping proof
  sorry

end Charles_has_13_whistles_l141_141289


namespace mean_of_four_numbers_l141_141694

theorem mean_of_four_numbers (a b c d : ℚ) (h : a + b + c + d = 1/2) : (a + b + c + d) / 4 = 1 / 8 :=
by
  -- proof skipped
  sorry

end mean_of_four_numbers_l141_141694


namespace ratio_of_discretionary_income_l141_141554

theorem ratio_of_discretionary_income 
  (salary : ℝ) (D : ℝ)
  (h_salary : salary = 3500)
  (h_discretionary : 0.15 * D = 105) :
  D / salary = 1 / 5 :=
by
  sorry

end ratio_of_discretionary_income_l141_141554


namespace find_x_value_l141_141458

theorem find_x_value
  (y₁ y₂ z₁ z₂ x₁ x w k : ℝ)
  (h₁ : y₁ = 3) (h₂ : z₁ = 2) (h₃ : x₁ = 1)
  (h₄ : y₂ = 6) (h₅ : z₂ = 5)
  (inv_rel : ∀ y z k, x = k * (z / y^2))
  (const_prod : ∀ x w, x * w = 1) :
  x = 5 / 8 :=
by
  -- omitted proof steps
  sorry

end find_x_value_l141_141458


namespace number_of_perfect_squares_between_50_and_200_l141_141072

theorem number_of_perfect_squares_between_50_and_200 :
  ∃ n: ℕ, 50 < n^2 ∧ n^2 < 200 ∧ (14 - 8 + 1 = 7) := sorry

end number_of_perfect_squares_between_50_and_200_l141_141072


namespace printer_time_equation_l141_141586

theorem printer_time_equation (x : ℝ) (rate1 rate2 : ℝ) (flyers1 flyers2 : ℝ)
  (h1 : rate1 = 100) (h2 : flyers1 = 1000) (h3 : flyers2 = 1000) 
  (h4 : flyers1 / rate1 = 10) (h5 : flyers1 / (rate1 + rate2) = 4) : 
  1 / 10 + 1 / x = 1 / 4 :=
by 
  sorry

end printer_time_equation_l141_141586


namespace vector_dot_product_l141_141399

-- Definitions based on the given conditions
variables (A B C M : ℝ)  -- points in 2D or 3D space can be generalized as real numbers for simplicity
variables (BA BC BM : ℝ) -- vector magnitudes
variables (AC : ℝ) -- magnitude of AC

-- Hypotheses from the problem conditions
variable (hM : 2 * BM = BA + BC)  -- M is the midpoint of AC
variable (hAC : AC = 4)
variable (hBM : BM = 3)

-- Theorem statement asserting the desired result
theorem vector_dot_product :
  BA * BC = 5 :=
by {
  sorry
}

end vector_dot_product_l141_141399


namespace stock_reaches_N_fourth_time_l141_141980

noncomputable def stock_at_k (c0 a b : ℝ) (k : ℕ) : ℝ :=
  if k % 2 = 0 then c0 + (k / 2) * (a - b)
  else c0 + (k / 2 + 1) * a - (k / 2) * b

theorem stock_reaches_N_fourth_time (c0 a b N : ℝ) (hN3 : ∃ k1 k2 k3 : ℕ, k1 ≠ k2 ∧ k2 ≠ k3 ∧ k1 ≠ k3 ∧ stock_at_k c0 a b k1 = N ∧ stock_at_k c0 a b k2 = N ∧ stock_at_k c0 a b k3 = N) :
  ∃ k4 : ℕ, k4 ≠ k1 ∧ k4 ≠ k2 ∧ k4 ≠ k3 ∧ stock_at_k c0 a b k4 = N := 
sorry

end stock_reaches_N_fourth_time_l141_141980


namespace simplify_expression_l141_141614

variable (x : ℝ)

theorem simplify_expression : 
  (3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3 - 3 * x^3) 
  = (-x^3 - x^2 + 23 * x - 3) :=
by
  sorry

end simplify_expression_l141_141614


namespace highest_numbered_street_l141_141138

theorem highest_numbered_street (L : ℕ) (d : ℕ) (H : L = 15000 ∧ d = 500) : 
    (L / d) - 2 = 28 :=
by
  sorry

end highest_numbered_street_l141_141138


namespace intersection_M_N_l141_141161

-- Define set M
def M : Set ℝ := {x : ℝ | ∃ t : ℝ, x = 2^(-t) }

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sin x }

-- Theorem stating the intersection of M and N
theorem intersection_M_N :
  (M ∩ N) = {y : ℝ | 0 < y ∧ y ≤ 1} :=
by
  sorry

end intersection_M_N_l141_141161


namespace eval_exponents_l141_141222

theorem eval_exponents : (2^3)^2 - 4^3 = 0 := by
  sorry

end eval_exponents_l141_141222


namespace distance_from_origin_l141_141774

theorem distance_from_origin :
  ∃ (m : ℝ), m = Real.sqrt (108 + 8 * Real.sqrt 10) ∧
              (∃ (x y : ℝ), y = 8 ∧ 
                            (x - 2)^2 + (y - 5)^2 = 49 ∧ 
                            x = 2 + 2 * Real.sqrt 10 ∧ 
                            m = Real.sqrt ((x^2) + (y^2))) :=
by
  sorry

end distance_from_origin_l141_141774


namespace trig_identity_l141_141433

theorem trig_identity (α : ℝ) 
  (h : Real.tan (α - Real.pi / 4) = 1 / 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 := 
sorry

end trig_identity_l141_141433


namespace radius_of_tangent_circle_l141_141059

theorem radius_of_tangent_circle (a b : ℕ) (r1 r2 r3 : ℚ) (R : ℚ)
  (h1 : a = 6) (h2 : b = 8)
  (h3 : r1 = a / 2) (h4 : r2 = b / 2) (h5 : r3 = (Real.sqrt (a^2 + b^2)) / 2) :
  R = 144 / 23 := sorry

end radius_of_tangent_circle_l141_141059


namespace find_third_side_l141_141100

def vol_of_cube (side : ℝ) : ℝ := side ^ 3

def vol_of_box (length width height : ℝ) : ℝ := length * width * height

theorem find_third_side (n : ℝ) (vol_cube : ℝ) (num_cubes : ℝ) (l w : ℝ) (vol_box : ℝ) :
  num_cubes = 24 →
  vol_cube = 27 →
  l = 8 →
  w = 12 →
  vol_box = num_cubes * vol_cube →
  vol_box = vol_of_box l w n →
  n = 6.75 :=
by
  intros hcubes hc_vol hl hw hvbox1 hvbox2
  -- The proof goes here
  sorry

end find_third_side_l141_141100


namespace third_speed_correct_l141_141552

variable (total_time : ℝ := 11)
variable (total_distance : ℝ := 900)
variable (speed1_km_hr : ℝ := 3)
variable (speed2_km_hr : ℝ := 9)

noncomputable def convert_speed_km_hr_to_m_min (speed: ℝ) : ℝ := speed * 1000 / 60

noncomputable def equal_distance : ℝ := total_distance / 3

noncomputable def third_speed_m_min : ℝ :=
  let speed1_m_min := convert_speed_km_hr_to_m_min speed1_km_hr
  let speed2_m_min := convert_speed_km_hr_to_m_min speed2_km_hr
  let d := equal_distance
  300 / (total_time - (d / speed1_m_min + d / speed2_m_min))

noncomputable def third_speed_km_hr : ℝ := third_speed_m_min * 60 / 1000

theorem third_speed_correct : third_speed_km_hr = 6 := by
  sorry

end third_speed_correct_l141_141552


namespace tangent_line_at_1_extreme_points_range_of_a_l141_141840

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * (x ^ 2 - 3 * x + 2)

theorem tangent_line_at_1 (a : ℝ) (h : a = 0) :
  ∃ m b, ∀ x, f x a = m * x + b ∧ m = 1 ∧ b = -1 := sorry

theorem extreme_points (a : ℝ) :
  (0 < a ∧ a <= 8 / 9 → ∀ x, 0 < x → f x a = 0) ∧
  (a > 8 / 9 → ∃ x1 x2, x1 < x2 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧
   (∀ x, 0 < x ∧ x < x1 → f x a = 0) ∧
   (∀ x, x1 < x ∧ x < x2 → f x a = 0) ∧
   (∀ x, x2 < x → f x a = 0)) ∧
  (a < 0 → ∃ x1 x2, x1 < 0 ∧ 0 < x2 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧
   (∀ x, 0 < x ∧ x < x2 → f x a = 0) ∧
   (∀ x, x2 < x → f x a = 0)) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x → f x a >= 0) ↔ 0 ≤ a ∧ a ≤ 1 := sorry

end tangent_line_at_1_extreme_points_range_of_a_l141_141840


namespace smallest_pos_multiple_6_15_is_30_l141_141141

theorem smallest_pos_multiple_6_15_is_30 :
  ∃ b > 0, 6 ∣ b ∧ 15 ∣ b ∧ (∀ b', b' > 0 ∧ b' < b → ¬ (6 ∣ b' ∧ 15 ∣ b')) :=
by
  -- Implementation to be done
  sorry

end smallest_pos_multiple_6_15_is_30_l141_141141


namespace estimate_pi_l141_141684

theorem estimate_pi (m : ℝ) (n : ℝ) (a : ℝ) (b : ℝ) (h1 : m = 56) (h2 : n = 200) (h3 : a = 1/2) (h4 : b = 1/4) :
  (m / n) = (π / 4 - 1 / 2) ↔ π = 78 / 25 :=
by
  sorry

end estimate_pi_l141_141684


namespace polynomial_identity_l141_141804

theorem polynomial_identity (x : ℝ) :
  (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * (x - 1) + 1 = x^4 :=
sorry

end polynomial_identity_l141_141804


namespace remainder_19008_div_31_l141_141972

theorem remainder_19008_div_31 :
  ∀ (n : ℕ), (n = 432 * 44) → n % 31 = 5 :=
by
  intro n h
  sorry

end remainder_19008_div_31_l141_141972


namespace binary_to_decimal_l141_141264

theorem binary_to_decimal : (11010 : ℕ) = 26 := by
  sorry

end binary_to_decimal_l141_141264


namespace salary_of_A_l141_141282

theorem salary_of_A (A B : ℝ) (h1 : A + B = 7000) (h2 : 0.05 * A = 0.15 * B) : A = 5250 := 
by 
  sorry

end salary_of_A_l141_141282


namespace coloring_problem_l141_141371

theorem coloring_problem (a : ℕ → ℕ) (n t : ℕ) 
  (h1 : ∀ i j, i < j → a i < a j) 
  (h2 : ∀ x : ℤ, ∃ i, 0 < i ∧ i ≤ n ∧ ((x + a (i - 1)) % t) = 0) : 
  n ∣ t :=
by
  sorry

end coloring_problem_l141_141371


namespace Chloe_pairs_shoes_l141_141162

theorem Chloe_pairs_shoes (cost_per_shoe total_cost : ℤ) (h_cost: cost_per_shoe = 37) (h_total: total_cost = 1036) :
  (total_cost / cost_per_shoe) / 2 = 14 :=
by
  -- proof goes here
  sorry

end Chloe_pairs_shoes_l141_141162


namespace triangle_largest_angle_and_type_l141_141027

theorem triangle_largest_angle_and_type
  (a b c : ℝ) 
  (h1 : a + b + c = 180)
  (h2 : a = 4 * k) 
  (h3 : b = 3 * k) 
  (h4 : c = 2 * k) 
  (h5 : a ≥ b) 
  (h6 : a ≥ c) : 
  a = 80 ∧ a < 90 ∧ b < 90 ∧ c < 90 := 
by
  -- Replace 'by' with 'sorry' to denote that the proof should go here
  sorry

end triangle_largest_angle_and_type_l141_141027


namespace colleen_paid_more_l141_141806

def pencils_joy : ℕ := 30
def pencils_colleen : ℕ := 50
def cost_per_pencil : ℕ := 4

theorem colleen_paid_more : 
  (pencils_colleen - pencils_joy) * cost_per_pencil = 80 :=
by
  sorry

end colleen_paid_more_l141_141806


namespace cos_angle_equiv_370_l141_141014

open Real

noncomputable def find_correct_n : ℕ :=
  sorry

theorem cos_angle_equiv_370 (n : ℕ) (h : 0 ≤ n ∧ n ≤ 180) : cos (n * π / 180) = cos (370 * π / 180) → n = 10 :=
by
  sorry

end cos_angle_equiv_370_l141_141014


namespace root_quadratic_sum_product_l141_141514

theorem root_quadratic_sum_product (x1 x2 : ℝ) (h1 : x1^2 - 2 * x1 - 5 = 0) (h2 : x2^2 - 2 * x2 - 5 = 0) 
  (h3 : x1 ≠ x2) : (x1 + x2 + 3 * (x1 * x2)) = -13 := 
by 
  sorry

end root_quadratic_sum_product_l141_141514


namespace no_solution_l141_141335

theorem no_solution (x : ℝ) : ¬ (x / -4 ≥ 3 + x ∧ |2*x - 1| < 4 + 2*x) := 
by sorry

end no_solution_l141_141335


namespace consumption_reduction_l141_141159

variable (P C : ℝ)

theorem consumption_reduction (h : P > 0 ∧ C > 0) : 
  (1.25 * P * (0.8 * C) = P * C) :=
by
  -- Conditions: original price P, original consumption C
  -- New price 1.25 * P, New consumption 0.8 * C
  sorry

end consumption_reduction_l141_141159


namespace sum_of_first_3n_terms_l141_141312

variable {S : ℕ → ℝ}
variable {n : ℕ}
variable {a b : ℝ}

def arithmetic_sum (S : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ m : ℕ, S (m + 1) = S m + (d * (m + 1))

theorem sum_of_first_3n_terms (h1 : S n = a) (h2 : S (2 * n) = b) 
  (h3 : arithmetic_sum S) : S (3 * n) = 3 * b - 2 * a :=
by
  sorry

end sum_of_first_3n_terms_l141_141312


namespace diameter_twice_radius_l141_141747

theorem diameter_twice_radius (r d : ℝ) (h : d = 2 * r) : d = 2 * r :=
by
  exact h

end diameter_twice_radius_l141_141747


namespace range_of_a_l141_141016

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 3| + |x + 5| > a) → a < 8 := by
  sorry

end range_of_a_l141_141016


namespace ribbon_cost_l141_141538

variable (c_g c_m s : ℝ)

theorem ribbon_cost (h1 : 5 * c_g + s = 295) (h2 : 7 * c_m + s = 295) (h3 : 2 * c_m + c_g = 102) : s = 85 :=
sorry

end ribbon_cost_l141_141538


namespace probability_both_truth_l141_141756

variable (P_A : ℝ) (P_B : ℝ)

theorem probability_both_truth (hA : P_A = 0.55) (hB : P_B = 0.60) :
  P_A * P_B = 0.33 :=
by
  sorry

end probability_both_truth_l141_141756


namespace union_of_A_B_complement_intersection_l141_141649

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | -x^2 + 2*x + 15 ≤ 0 }

def B : Set ℝ := { x | |x - 5| < 1 }

theorem union_of_A_B :
  A ∪ B = { x | x ≤ -3 ∨ x > 4 } :=
by
  sorry

theorem complement_intersection :
  (U \ A) ∩ B = { x | 4 < x ∧ x < 5 } :=
by
  sorry

end union_of_A_B_complement_intersection_l141_141649


namespace A_left_after_3_days_l141_141569

def work_done_by_A_and_B_together (x : ℕ) : ℚ :=
  (1 / 21) * x + (1 / 28) * x

def work_done_by_B_alone (days : ℕ) : ℚ :=
  (1 / 28) * days

def total_work_done (x days_b_alone : ℕ) : ℚ :=
  work_done_by_A_and_B_together x + work_done_by_B_alone days_b_alone

theorem A_left_after_3_days :
  ∀ (x : ℕ), total_work_done x 21 = 1 ↔ x = 3 := by
  sorry

end A_left_after_3_days_l141_141569


namespace range_of_a_l141_141920

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := if x ≤ 1 then (a - 3) * x - 3 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ 3 < a ∧ a ≤ 6 :=
by
  sorry

end range_of_a_l141_141920


namespace determine_signs_l141_141253

theorem determine_signs (a b c : ℝ) (h1 : a != 0 ∧ b != 0 ∧ c == 0)
  (h2 : a > 0 ∨ (b + c) > 0) : a > 0 ∧ b < 0 ∧ c = 0 :=
by
  sorry

end determine_signs_l141_141253


namespace total_birds_on_fence_l141_141827

theorem total_birds_on_fence (initial_pairs : ℕ) (birds_per_pair : ℕ) 
                             (new_pairs : ℕ) (new_birds_per_pair : ℕ)
                             (initial_birds : initial_pairs * birds_per_pair = 24)
                             (new_birds : new_pairs * new_birds_per_pair = 8) : 
                             ((initial_pairs * birds_per_pair) + (new_pairs * new_birds_per_pair) = 32) :=
sorry

end total_birds_on_fence_l141_141827


namespace twice_shorter_vs_longer_l141_141506

-- Definitions and conditions
def total_length : ℝ := 20
def shorter_length : ℝ := 8
def longer_length : ℝ := total_length - shorter_length

-- Statement to prove
theorem twice_shorter_vs_longer :
  2 * shorter_length - longer_length = 4 :=
by
  sorry

end twice_shorter_vs_longer_l141_141506


namespace parallel_vectors_l141_141686

open Real

theorem parallel_vectors (k : ℝ) 
  (a : ℝ × ℝ := (k-1, 1)) 
  (b : ℝ × ℝ := (k+3, k)) 
  (h : a.1 * b.2 = a.2 * b.1) : 
  k = 3 ∨ k = -1 :=
by
  sorry

end parallel_vectors_l141_141686


namespace arc_length_l141_141708

theorem arc_length (circumference : ℝ) (angle_degrees : ℝ) (h : circumference = 90) (θ : angle_degrees = 45) :
  (angle_degrees / 360) * circumference = 11.25 := 
  by 
    sorry

end arc_length_l141_141708


namespace initial_number_of_men_l141_141223

theorem initial_number_of_men (M : ℕ) 
  (h1 : M * 8 * 40 = (M + 30) * 6 * 50) 
  : M = 450 :=
by 
  sorry

end initial_number_of_men_l141_141223


namespace common_root_equations_l141_141905

theorem common_root_equations (a b : ℝ) 
  (h : ∃ x₀ : ℝ, (x₀ ^ 2 + a * x₀ + b = 0) ∧ (x₀ ^ 2 + b * x₀ + a = 0)) 
  (hc : ∀ x₁ x₂ : ℝ, (x₁ ^ 2 + a * x₁ + b = 0 ∧ x₂ ^ 2 + bx₀ + a = 0) → x₁ = x₂) :
  a + b = -1 :=
sorry

end common_root_equations_l141_141905


namespace total_balloons_l141_141092

theorem total_balloons (F S M : ℕ) (hF : F = 5) (hS : S = 6) (hM : M = 7) : F + S + M = 18 :=
by 
  sorry

end total_balloons_l141_141092


namespace rectangle_perimeter_inequality_l141_141491

-- Define rectilinear perimeters
def perimeter (length : ℝ) (width : ℝ) : ℝ := 2 * (length + width)

-- Definitions for rectangles contained within each other
def rectangle_contained (len1 wid1 len2 wid2 : ℝ) : Prop :=
  len1 ≤ len2 ∧ wid1 ≤ wid2

-- Statement of the problem
theorem rectangle_perimeter_inequality (l1 w1 l2 w2 : ℝ) (h : rectangle_contained l1 w1 l2 w2) :
  perimeter l1 w1 ≤ perimeter l2 w2 :=
sorry

end rectangle_perimeter_inequality_l141_141491


namespace cone_volume_l141_141343

theorem cone_volume (r l h V: ℝ) (h1: 15 * Real.pi = Real.pi * r^2 + Real.pi * r * l)
  (h2: 2 * Real.pi * r = (1 / 3) * Real.pi * l) :
  (V = (1 / 3) * Real.pi * r^2 * h) → h = Real.sqrt (l^2 - r^2) → l = 6 * r → r = Real.sqrt (15 / 7) → 
  V = (25 * Real.sqrt 3 / 7) * Real.pi :=
sorry

end cone_volume_l141_141343


namespace fraction_to_terminating_decimal_l141_141352

theorem fraction_to_terminating_decimal :
  (53 : ℚ)/160 = 0.33125 :=
by sorry

end fraction_to_terminating_decimal_l141_141352


namespace sufficient_condition_for_P_l141_141492

noncomputable def increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

theorem sufficient_condition_for_P (f : ℝ → ℝ) (t : ℝ) 
  (h_inc : increasing f) (h_val1 : f (-1) = -4) (h_val2 : f 2 = 2) :
  (∀ x, (x ∈ {x | -1 - t < x ∧ x < 2 - t}) → x < -1) → t ≥ 3 :=
by
  sorry

end sufficient_condition_for_P_l141_141492


namespace total_birds_is_1300_l141_141323

def initial_birds : ℕ := 300
def birds_doubled (b : ℕ) : ℕ := 2 * b
def birds_reduced (b : ℕ) : ℕ := b - 200
def total_birds_three_days : ℕ := initial_birds + birds_doubled initial_birds + birds_reduced (birds_doubled initial_birds)

theorem total_birds_is_1300 : total_birds_three_days = 1300 :=
by
  unfold total_birds_three_days initial_birds birds_doubled birds_reduced
  simp
  done

end total_birds_is_1300_l141_141323


namespace sales_volume_conditions_l141_141532

noncomputable def sales_volume (x : ℝ) (a k : ℝ) : ℝ :=
if 1 < x ∧ x ≤ 3 then a * (x - 4)^2 + 6 / (x - 1)
else if 3 < x ∧ x ≤ 5 then k * x + 7
else 0

theorem sales_volume_conditions (a k : ℝ) :
  (sales_volume 3 a k = 4) ∧ (sales_volume 5 a k = 2) ∧
  ((∃ x, 1 < x ∧ x ≤ 3 ∧ sales_volume x a k = 10) ∨ 
   (∃ x, 3 < x ∧ x ≤ 5 ∧ sales_volume x a k = 9)) :=
sorry

end sales_volume_conditions_l141_141532


namespace fraction_cost_of_raisins_l141_141802

variable (cost_raisins cost_nuts total_cost_raisins total_cost_nuts total_cost : ℝ)

theorem fraction_cost_of_raisins (h1 : cost_nuts = 3 * cost_raisins)
                                 (h2 : total_cost_raisins = 4 * cost_raisins)
                                 (h3 : total_cost_nuts = 4 * cost_nuts)
                                 (h4 : total_cost = total_cost_raisins + total_cost_nuts) :
                                 (total_cost_raisins / total_cost) = (1 / 4) :=
by
  sorry

end fraction_cost_of_raisins_l141_141802


namespace books_read_in_eight_hours_l141_141303

noncomputable def pages_per_hour : ℕ := 120
noncomputable def pages_per_book : ℕ := 360
noncomputable def total_reading_time : ℕ := 8

theorem books_read_in_eight_hours (h1 : pages_per_hour = 120) 
                                  (h2 : pages_per_book = 360) 
                                  (h3 : total_reading_time = 8) : 
                                  total_reading_time * pages_per_hour / pages_per_book = 2 := 
by sorry

end books_read_in_eight_hours_l141_141303


namespace sinks_per_house_l141_141897

theorem sinks_per_house (total_sinks : ℕ) (houses : ℕ) (h_total_sinks : total_sinks = 266) (h_houses : houses = 44) :
  total_sinks / houses = 6 :=
by {
  sorry
}

end sinks_per_house_l141_141897


namespace complement_A_in_U_l141_141943

def U : Set ℝ := {x : ℝ | x > 0}
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 2}
def AC : Set ℝ := {x : ℝ | (0 < x ∧ x ≤ 1) ∨ (2 ≤ x)}

theorem complement_A_in_U : U \ A = AC := 
by 
  sorry

end complement_A_in_U_l141_141943


namespace find_m_n_difference_l141_141726

theorem find_m_n_difference (x y m n : ℤ)
  (hx : x = 2)
  (hy : y = -3)
  (hm : x + y = m)
  (hn : 2 * x - y = n) :
  m - n = -8 :=
by {
  sorry
}

end find_m_n_difference_l141_141726


namespace divisible_iff_exists_t_l141_141821

theorem divisible_iff_exists_t (a b m α : ℤ) (h_coprime : Int.gcd a m = 1) (h_divisible : a * α + b ≡ 0 [ZMOD m]):
  ∀ x : ℤ, (a * x + b ≡ 0 [ZMOD m]) ↔ ∃ t : ℤ, x = α + m * t :=
sorry

end divisible_iff_exists_t_l141_141821


namespace incorrect_statement_B_l141_141946

open Set

-- Define the relevant events as described in the problem
def event_subscribe_at_least_one (ω : Type) (A B : Set ω) : Set ω := A ∪ B
def event_subscribe_at_most_one (ω : Type) (A B : Set ω) : Set ω := (A ∩ B)ᶜ

-- Define the problem statement
theorem incorrect_statement_B (ω : Type) (A B : Set ω) :
  ¬ (event_subscribe_at_least_one ω A B) = (event_subscribe_at_most_one ω A B)ᶜ :=
sorry

end incorrect_statement_B_l141_141946


namespace sqrt_product_simplification_l141_141147

variable (q : ℝ)

theorem sqrt_product_simplification (hq : q ≥ 0) : 
  Real.sqrt (15 * q) * Real.sqrt (8 * q^3) * Real.sqrt (12 * q^5) = 12 * q^4 * Real.sqrt (10 * q) :=
by
  sorry

end sqrt_product_simplification_l141_141147


namespace solve_fraction_equation_l141_141209

theorem solve_fraction_equation : ∀ (x : ℝ), (x + 2) / (2 * x - 1) = 1 → x = 3 :=
by
  intros x h
  sorry

end solve_fraction_equation_l141_141209


namespace sum_of_first_11_terms_is_minus_66_l141_141790

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d 

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a n + a 1)) / 2

theorem sum_of_first_11_terms_is_minus_66 
  (a : ℕ → ℤ) 
  (h_seq : arithmetic_sequence a)
  (h_roots : ∃ a2 a10, (a2 = a 2 ∧ a10 = a 10) ∧ (a2 + a10 = -12) ∧ (a2 * a10 = -8)) 
  : sum_of_first_n_terms a 11 = -66 :=
by
  sorry

end sum_of_first_11_terms_is_minus_66_l141_141790


namespace shapes_identification_l141_141053

theorem shapes_identification :
  (∃ x y: ℝ, (x - 1/2)^2 + y^2 = 1/4) ∧ (∃ t: ℝ, x = -t ∧ y = 2 + t → x + y + 1 = 0) :=
by
  sorry

end shapes_identification_l141_141053


namespace greatest_remainder_when_dividing_by_10_l141_141922

theorem greatest_remainder_when_dividing_by_10 (x : ℕ) : 
  ∃ r : ℕ, r < 10 ∧ r = x % 10 ∧ r = 9 :=
by
  sorry

end greatest_remainder_when_dividing_by_10_l141_141922


namespace math_problem_equivalence_l141_141882

section

variable (x y z : ℝ) (w : String)

theorem math_problem_equivalence (h₀ : x / 15 = 4 / 5) (h₁ : y = 80) (h₂ : z = 0.8) (h₃ : w = "八折"):
  x = 12 ∧ y = 80 ∧ z = 0.8 ∧ w = "八折" :=
by
  sorry

end

end math_problem_equivalence_l141_141882


namespace parallel_vectors_condition_l141_141299

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b = (k * a.1, k * a.2)

theorem parallel_vectors_condition (m : ℝ) :
  vectors_parallel (1, m + 1) (m, 2) ↔ m = -2 ∨ m = 1 := by
  sorry

end parallel_vectors_condition_l141_141299


namespace wire_length_l141_141650

variables (L M S W : ℕ)

def ratio_condition (L M S : ℕ) : Prop :=
  L * 2 = 7 * S ∧ M * 2 = 3 * S

def total_length (L M S : ℕ) : ℕ :=
  L + M + S

theorem wire_length (h : ratio_condition L M 16) : total_length L M 16 = 96 :=
by sorry

end wire_length_l141_141650


namespace complement_A_union_B_l141_141819

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the quadratic equation condition
def B : Set ℤ := {x | x^2 - 4*x + 3 = 0}

-- State the theorem we want to prove
theorem complement_A_union_B : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_A_union_B_l141_141819


namespace sum_of_possible_values_CDF_l141_141984

theorem sum_of_possible_values_CDF 
  (C D F : ℕ) 
  (hC: 0 ≤ C ∧ C ≤ 9)
  (hD: 0 ≤ D ∧ D ≤ 9)
  (hF: 0 ≤ F ∧ F ≤ 9)
  (hdiv: (C + 4 + 9 + 8 + D + F + 4) % 9 = 0) :
  C + D + F = 2 ∨ C + D + F = 11 → (2 + 11 = 13) :=
by sorry

end sum_of_possible_values_CDF_l141_141984


namespace roberto_raise_percentage_l141_141045

theorem roberto_raise_percentage
    (starting_salary : ℝ)
    (previous_salary : ℝ)
    (current_salary : ℝ)
    (h1 : starting_salary = 80000)
    (h2 : previous_salary = starting_salary * 1.40)
    (h3 : current_salary = 134400) :
    ((current_salary - previous_salary) / previous_salary) * 100 = 20 :=
by sorry

end roberto_raise_percentage_l141_141045


namespace expression_value_l141_141216

theorem expression_value (x y z : ℤ) (hx : x = 26) (hy : y = 3 * x / 2) (hz : z = 11) :
  x - (y - z) - ((x - y) - z) = 22 := 
by
  -- problem statement here
  -- simplified proof goes here
  sorry

end expression_value_l141_141216


namespace polynomial_roots_distinct_and_expression_is_integer_l141_141331

-- Defining the conditions and the main theorem
theorem polynomial_roots_distinct_and_expression_is_integer (a b c : ℂ) :
  (a^3 - a^2 - a - 1 = 0) → (b^3 - b^2 - b - 1 = 0) → (c^3 - c^2 - c - 1 = 0) → 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  ∃ k : ℤ, ((a^(1982) - b^(1982)) / (a - b) + (b^(1982) - c^(1982)) / (b - c) + (c^(1982) - a^(1982)) / (c - a) = k) :=
by
  intros h1 h2 h3
  -- Proof omitted
  sorry

end polynomial_roots_distinct_and_expression_is_integer_l141_141331


namespace direct_proportion_inequality_l141_141336

theorem direct_proportion_inequality (k x1 x2 y1 y2 : ℝ) (h_k : k < 0) (h_y1 : y1 = k * x1) (h_y2 : y2 = k * x2) (h_x : x1 < x2) : y1 > y2 :=
by
  -- The proof will be written here, currently leaving it as sorry
  sorry

end direct_proportion_inequality_l141_141336


namespace converse_of_propositions_is_true_l141_141396

theorem converse_of_propositions_is_true :
  (∀ x : ℝ, (x = 1 ∨ x = 2) ↔ (x^2 - 3 * x + 2 = 0)) ∧
  (∀ x y : ℝ, (x^2 + y^2 = 0) ↔ (x = 0 ∧ y = 0)) := 
by {
  sorry
}

end converse_of_propositions_is_true_l141_141396


namespace probability_of_s_in_statistics_l141_141381

theorem probability_of_s_in_statistics :
  let totalLetters := 10
  let count_s := 3
  (count_s / totalLetters : ℚ) = 3 / 10 := by
  sorry

end probability_of_s_in_statistics_l141_141381


namespace find_y_l141_141169

theorem find_y (y : ℝ) (h : (15 + 25 + y) / 3 = 23) : y = 29 :=
sorry

end find_y_l141_141169


namespace find_function_ex_l141_141109

theorem find_function_ex (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  (∀ x : ℝ, f x = x - a) :=
by
  intros h x
  sorry

end find_function_ex_l141_141109


namespace minimum_y_value_y_at_4_eq_6_l141_141810

noncomputable def y (x : ℝ) : ℝ := x + 4 / (x - 2)

theorem minimum_y_value (x : ℝ) (h : x > 2) : y x ≥ 6 :=
sorry

theorem y_at_4_eq_6 : y 4 = 6 :=
sorry

end minimum_y_value_y_at_4_eq_6_l141_141810


namespace union_A_B_complement_A_l141_141711

-- Definition of Universe U
def U : Set ℝ := Set.univ

-- Definition of set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Definition of set B
def B : Set ℝ := {x | -2 < x ∧ x < 2}

-- Theorem 1: Proving the union A ∪ B
theorem union_A_B : A ∪ B = {x | -2 < x ∧ x ≤ 3} := 
sorry

-- Theorem 2: Proving the complement of A with respect to U
theorem complement_A : (U \ A) = {x | x < -1 ∨ x > 3} := 
sorry

end union_A_B_complement_A_l141_141711


namespace geometric_sequence_S5_eq_11_l141_141547

theorem geometric_sequence_S5_eq_11 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (q : ℤ)
  (h1 : a 1 = 1)
  (h4 : a 4 = -8)
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_S : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  S 5 = 11 := 
by
  -- Proof omitted
  sorry

end geometric_sequence_S5_eq_11_l141_141547


namespace division_result_l141_141551

theorem division_result :
  3486 / 189 = 18.444444444444443 := by
  sorry

end division_result_l141_141551


namespace grey_pairs_coincide_l141_141007

theorem grey_pairs_coincide (h₁ : 4 = orange_count / 2) 
                                (h₂ : 6 = green_count / 2)
                                (h₃ : 9 = grey_count / 2)
                                (h₄ : 3 = orange_pairs)
                                (h₅ : 4 = green_pairs)
                                (h₆ : 1 = orange_grey_pairs) :
    grey_pairs = 6 := by
  sorry

noncomputable def half_triangle_counts : (ℕ × ℕ × ℕ) := (4, 6, 9)

noncomputable def triangle_pairs : (ℕ × ℕ × ℕ) := (3, 4, 1)

noncomputable def prove_grey_pairs (orange_count green_count grey_count : ℕ)
                                   (orange_pairs green_pairs orange_grey_pairs : ℕ) : ℕ :=
  sorry

end grey_pairs_coincide_l141_141007


namespace store_second_reduction_percentage_l141_141255

theorem store_second_reduction_percentage (P : ℝ) :
  let first_reduction := 0.88 * P
  let second_reduction := 0.792 * P
  ∃ R : ℝ, (1 - R) * first_reduction = second_reduction ∧ R = 0.1 :=
by
  let first_reduction := 0.88 * P
  let second_reduction := 0.792 * P
  use 0.1
  sorry

end store_second_reduction_percentage_l141_141255


namespace find_first_number_l141_141063

/-- The lcm of two numbers is 2310 and hcf (gcd) is 26. One of the numbers is 286. What is the other number? --/
theorem find_first_number (A : ℕ) 
  (h_lcm : Nat.lcm A 286 = 2310) 
  (h_gcd : Nat.gcd A 286 = 26) : 
  A = 210 := 
by
  sorry

end find_first_number_l141_141063


namespace eval_ceil_floor_sum_l141_141661

def ceil_floor_sum : ℤ :=
  ⌈(7:ℚ) / (3:ℚ)⌉ + ⌊-((7:ℚ) / (3:ℚ))⌋

theorem eval_ceil_floor_sum : ceil_floor_sum = 0 :=
sorry

end eval_ceil_floor_sum_l141_141661


namespace parabola_min_value_sum_abc_zero_l141_141465

theorem parabola_min_value_sum_abc_zero
  (a b c : ℝ)
  (h1 : ∀ x y : ℝ, y = ax^2 + bx + c → y = 0 ∧ ((x = 1) ∨ (x = 5)))
  (h2 : ∃ x : ℝ, (∀ t : ℝ, ax^2 + bx + c ≤ t^2) ∧ (ax^2 + bx + c = 36)) :
  a + b + c = 0 :=
sorry

end parabola_min_value_sum_abc_zero_l141_141465


namespace sum_divisible_by_4_l141_141559

theorem sum_divisible_by_4 (a b c d x : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_eq : (x^2 - (a+b)*x + a*b) * (x^2 - (c+d)*x + c*d) = 9) : 4 ∣ (a + b + c + d) :=
by
  sorry

end sum_divisible_by_4_l141_141559


namespace integer_roots_of_polynomial_l141_141915

theorem integer_roots_of_polynomial : 
  {x : ℤ | x^3 - 6 * x^2 - 4 * x + 24 = 0} = {2, -2} :=
by
  sorry

end integer_roots_of_polynomial_l141_141915


namespace positive_root_exists_iff_p_range_l141_141088

theorem positive_root_exists_iff_p_range (p : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x^4 + 4 * p * x^3 + x^2 + 4 * p * x + 4 = 0) ↔ 
  p ∈ Set.Iio (-Real.sqrt 2 / 2) ∪ Set.Ioi (Real.sqrt 2 / 2) :=
by
  sorry

end positive_root_exists_iff_p_range_l141_141088


namespace sum_divides_exp_sum_l141_141184

theorem sum_divides_exp_sum (p a b c d : ℕ) [Fact (Nat.Prime p)] 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < p)
  (h6 : a^4 % p = b^4 % p) (h7 : b^4 % p = c^4 % p) (h8 : c^4 % p = d^4 % p) :
  (a + b + c + d) ∣ (a^2013 + b^2013 + c^2013 + d^2013) :=
sorry

end sum_divides_exp_sum_l141_141184


namespace line_equation_l141_141463

theorem line_equation (p : ℝ × ℝ) (a : ℝ × ℝ) :
  p = (4, -4) →
  a = (1, 2 / 7) →
  ∃ (m b : ℝ), m = 2 / 7 ∧ b = -36 / 7 ∧ ∀ x y : ℝ, y = m * x + b :=
by
  intros hp ha
  sorry

end line_equation_l141_141463


namespace total_cost_is_eight_times_l141_141625

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l141_141625


namespace shaded_area_is_10_l141_141843

-- Definitions based on conditions:
def rectangle_area : ℕ := 12
def unshaded_triangle_area : ℕ := 2

-- Proof statement without the actual proof.
theorem shaded_area_is_10 : rectangle_area - unshaded_triangle_area = 10 := by
  sorry

end shaded_area_is_10_l141_141843


namespace power_is_seventeen_l141_141553

theorem power_is_seventeen (x : ℕ) : (1000^7 : ℝ) / (10^x) = (10000 : ℝ) ↔ x = 17 := by
  sorry

end power_is_seventeen_l141_141553


namespace negate_universal_to_existential_l141_141386

variable {f : ℝ → ℝ}

theorem negate_universal_to_existential :
  (¬ (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0)) ↔
  (∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0) :=
  sorry

end negate_universal_to_existential_l141_141386


namespace find_min_value_l141_141924

noncomputable def expression (x : ℝ) : ℝ :=
  (Real.sin x ^ 8 + Real.cos x ^ 8 + 2) / (Real.sin x ^ 6 + Real.cos x ^ 6 + 2)

theorem find_min_value : ∃ x : ℝ, expression x = 5 / 4 :=
sorry

end find_min_value_l141_141924


namespace smallest_N_divisors_of_8_l141_141705

theorem smallest_N_divisors_of_8 (N : ℕ) (h0 : N % 10 = 0) (h8 : ∃ (divisors : ℕ), divisors = 8 ∧ (∀ k, k ∣ N → k ≤ divisors)) : N = 30 := 
sorry

end smallest_N_divisors_of_8_l141_141705


namespace product_of_integers_abs_val_not_less_than_1_and_less_than_3_l141_141330

theorem product_of_integers_abs_val_not_less_than_1_and_less_than_3 :
  (-2) * (-1) * 1 * 2 = 4 :=
by
  sorry

end product_of_integers_abs_val_not_less_than_1_and_less_than_3_l141_141330


namespace find_y_l141_141767

theorem find_y (x y : ℕ) (h_pos_y : 0 < y) (h_rem : x % y = 7) (h_div : x = 86 * y + (1 / 10) * y) :
  y = 70 :=
sorry

end find_y_l141_141767


namespace noemi_initial_money_l141_141079

variable (money_lost_roulette : ℕ := 400)
variable (money_lost_blackjack : ℕ := 500)
variable (money_left : ℕ)
variable (money_started : ℕ)

axiom money_left_condition : money_left > 0
axiom total_loss_condition : money_lost_roulette + money_lost_blackjack = 900

theorem noemi_initial_money (h1 : money_lost_roulette = 400) (h2 : money_lost_blackjack = 500)
    (h3 : money_started - 900 = money_left) (h4 : money_left > 0) :
    money_started > 900 := by
  sorry

end noemi_initial_money_l141_141079


namespace star_comm_star_assoc_star_id_exists_star_not_dist_add_l141_141057

def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

-- Statement 1: Commutativity
theorem star_comm : ∀ x y : ℝ, star x y = star y x := 
by sorry

-- Statement 2: Associativity
theorem star_assoc : ∀ x y z : ℝ, star (star x y) z = star x (star y z) := 
by sorry

-- Statement 3: Identity Element
theorem star_id_exists : ∃ e : ℝ, ∀ x : ℝ, star x e = x := 
by sorry

-- Statement 4: Distributivity Over Addition
theorem star_not_dist_add : ∃ x y z : ℝ, star x (y + z) ≠ star x y + star x z := 
by sorry

end star_comm_star_assoc_star_id_exists_star_not_dist_add_l141_141057


namespace launch_country_is_soviet_union_l141_141357

-- Definitions of conditions
def launch_date : String := "October 4, 1957"
def satellite_launched_on (date : String) : Prop := date = "October 4, 1957"
def choices : List String := ["A. United States", "B. Soviet Union", "C. European Union", "D. Germany"]

-- Problem statement
theorem launch_country_is_soviet_union : 
  satellite_launched_on launch_date → 
  "B. Soviet Union" ∈ choices := 
by
  sorry

end launch_country_is_soviet_union_l141_141357


namespace face_value_shares_l141_141733

theorem face_value_shares (market_value : ℝ) (dividend_rate desired_rate : ℝ) (FV : ℝ) 
  (h1 : dividend_rate = 0.09)
  (h2 : desired_rate = 0.12)
  (h3 : market_value = 36.00000000000001)
  (h4 : (dividend_rate * FV) = (desired_rate * market_value)) :
  FV = 48.00000000000001 :=
by
  sorry

end face_value_shares_l141_141733


namespace coefficient_of_q_is_correct_l141_141693

theorem coefficient_of_q_is_correct (q' : ℕ → ℕ) : 
  (∀ q : ℕ, q' q = 3 * q - 3) ∧  q' (q' 7) = 306 → ∃ a : ℕ, (∀ q : ℕ, q' q = a * q - 3) ∧ a = 17 :=
by
  sorry

end coefficient_of_q_is_correct_l141_141693


namespace parabola_equation_l141_141866

-- Definitions for the given conditions
def parabola_vertex_origin (y x : ℝ) : Prop := y = 0 ↔ x = 0
def axis_of_symmetry_x (y x : ℝ) : Prop := (x = -y) ↔ (x = y)
def focus_on_line (y x : ℝ) : Prop := 3 * x - 4 * y - 12 = 0

-- The statement to be proved
theorem parabola_equation :
  ∀ (y x : ℝ),
  (parabola_vertex_origin y x) ∧ (axis_of_symmetry_x y x) ∧ (focus_on_line y x) →
  y^2 = 16 * x :=
by
  intros y x h
  sorry

end parabola_equation_l141_141866


namespace find_f_neg2_l141_141588

noncomputable def f : ℝ → ℝ := sorry

axiom f_add (a b : ℝ) : f (a + b) = f a * f b
axiom f_pos (x : ℝ) : f x > 0
axiom f_one : f 1 = 1 / 3

theorem find_f_neg2 : f (-2) = 9 :=
by {
  sorry
}

end find_f_neg2_l141_141588


namespace smallest_term_abs_l141_141513

noncomputable def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem smallest_term_abs {a : ℕ → ℝ}
  (h_arith : arithmetic_sequence a)
  (h1 : a 1 > 0)
  (hS12 : (12 / 2) * (2 * a 1 + 11 * (a 2 - a 1)) > 0)
  (hS13 : (13 / 2) * (2 * a 1 + 12 * (a 2 - a 1)) < 0) :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 13 → n ≠ 7 → abs (a 6) > abs (a 1 + 6 * (a 2 - a 1)) :=
sorry

end smallest_term_abs_l141_141513


namespace parabola_tangent_xaxis_at_p2_parabola_vertex_yaxis_at_p0_parabolas_symmetric_m_point_parabola_familiy_point_through_l141_141906

noncomputable def parabola (p x : ℝ) : ℝ := (p-1) * x^2 + 2 * p * x + 4

-- 1. Prove that if \( p = 2 \), the parabola \( g_p \) is tangent to the \( x \)-axis.
theorem parabola_tangent_xaxis_at_p2 : ∀ x, parabola 2 x = (x + 2)^2 := 
by 
  intro x
  sorry

-- 2. Prove that if \( p = 0 \), the vertex of the parabola \( g_p \) lies on the \( y \)-axis.
theorem parabola_vertex_yaxis_at_p0 : ∃ x, parabola 0 x = 4 := 
by 
  sorry

-- 3. Prove the parabolas for \( p = 2 \) and \( p = 0 \) are symmetric with respect to \( M(-1, 2) \).
theorem parabolas_symmetric_m_point : ∀ x, 
  (parabola 2 x = (x + 2)^2) → 
  (parabola 0 x = -x^2 + 4) → 
  (-1, 2) = (-1, 2) := 
by 
  sorry

-- 4. Prove that the points \( (0, 4) \) and \( (-2, 0) \) lie on the curve for all \( p \).
theorem parabola_familiy_point_through : ∀ p, 
  parabola p 0 = 4 ∧ 
  parabola p (-2) = 0 :=
by 
  sorry

end parabola_tangent_xaxis_at_p2_parabola_vertex_yaxis_at_p0_parabolas_symmetric_m_point_parabola_familiy_point_through_l141_141906


namespace find_g7_l141_141290

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x ^ 7 + b * x ^ 3 + d * x ^ 2 + c * x - 8

theorem find_g7 (a b c d : ℝ) (h : g (-7) a b c d = 3) (h_d : d = 0) : g 7 a b c d = -19 :=
by
  simp [g, h, h_d]
  sorry

end find_g7_l141_141290


namespace return_trip_time_l141_141408

theorem return_trip_time (d p w : ℝ) (h1 : d = 84 * (p - w)) (h2 : d / (p + w) = d / p - 9) :
  (d / (p + w) = 63) ∨ (d / (p + w) = 12) :=
by
  sorry

end return_trip_time_l141_141408


namespace students_voted_both_l141_141144

def total_students : Nat := 300
def students_voted_first : Nat := 230
def students_voted_second : Nat := 190
def students_voted_none : Nat := 40

theorem students_voted_both :
  students_voted_first + students_voted_second - (total_students - students_voted_none) = 160 :=
by
  sorry

end students_voted_both_l141_141144


namespace coordinates_of_P_l141_141145

theorem coordinates_of_P (A B : ℝ × ℝ × ℝ) (m : ℝ) :
  A = (1, 0, 2) ∧ B = (1, -3, 1) ∧ (0, 0, m) = (0, 0, -3) :=
by 
  sorry

end coordinates_of_P_l141_141145


namespace mr_smiths_sixth_child_not_represented_l141_141974

def car_plate_number := { n : ℕ // ∃ a b : ℕ, n = 1001 * a + 110 * b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 }
def mr_smith_is_45 (n : ℕ) := (n % 100) = 45
def divisible_by_children_ages (n : ℕ) : Prop := ∀ i : ℕ, 1 ≤ i ∧ i ≤ 9 → n % i = 0

theorem mr_smiths_sixth_child_not_represented :
    ∃ n : car_plate_number, mr_smith_is_45 n.val ∧ divisible_by_children_ages n.val → ¬ (6 ∣ n.val) :=
by
  sorry

end mr_smiths_sixth_child_not_represented_l141_141974


namespace robotics_club_neither_l141_141028

theorem robotics_club_neither (total students programming electronics both: ℕ) 
  (h1: total = 120)
  (h2: programming = 80)
  (h3: electronics = 50)
  (h4: both = 15) : 
  total - ((programming - both) + (electronics - both) + both) = 5 :=
by
  sorry

end robotics_club_neither_l141_141028


namespace not_divisible_by_2006_l141_141121

theorem not_divisible_by_2006 (k : ℤ) : ¬ ∃ m : ℤ, k^2 + k + 1 = 2006 * m :=
sorry

end not_divisible_by_2006_l141_141121


namespace roden_gold_fish_count_l141_141678

theorem roden_gold_fish_count
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (gold_fish : ℕ)
  (h1 : total_fish = 22)
  (h2 : blue_fish = 7)
  (h3 : total_fish = blue_fish + gold_fish) : gold_fish = 15 :=
by
  sorry

end roden_gold_fish_count_l141_141678


namespace min_value_frac_sum_l141_141210

theorem min_value_frac_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 1) :
  ∃ (a b : ℝ), (a + 3 * b = 1) ∧ (a > 0) ∧ (b > 0) ∧ (∀ (a b : ℝ), (a + 3 * b = 1) → 0 < a → 0 < b → (1 / a + 3 / b) ≥ 16) :=
sorry

end min_value_frac_sum_l141_141210


namespace distance_first_to_last_tree_l141_141740

theorem distance_first_to_last_tree 
    (n_trees : ℕ) 
    (distance_first_to_fifth : ℕ)
    (h1 : n_trees = 8)
    (h2 : distance_first_to_fifth = 80) 
    : ∃ distance_first_to_last, distance_first_to_last = 140 := by
  sorry

end distance_first_to_last_tree_l141_141740


namespace trapezoid_base_solutions_l141_141081

theorem trapezoid_base_solutions (A h : ℕ) (d : ℕ) (bd : ℕ → Prop)
  (hA : A = 1800) (hH : h = 60) (hD : d = 10) (hBd : ∀ (x : ℕ), bd x ↔ ∃ (k : ℕ), x = d * k) :
  ∃ m n : ℕ, bd (10 * m) ∧ bd (10 * n) ∧ 10 * (m + n) = 60 ∧ m + n = 6 :=
by
  simp [hA, hH, hD, hBd]
  sorry

end trapezoid_base_solutions_l141_141081


namespace max_value_of_y_l141_141064

open Real

theorem max_value_of_y (x : ℝ) (h1 : 0 < x) (h2 : x < sqrt 3) : x * sqrt (3 - x^2) ≤ 9 / 4 :=
sorry

end max_value_of_y_l141_141064


namespace zero_of_function_l141_141696

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 4

theorem zero_of_function (x : ℝ) (h : f x = 0) (x1 x2 : ℝ)
  (h1 : -1 < x1 ∧ x1 < x)
  (h2 : x < x2 ∧ x2 < 2) :
  f x1 < 0 ∧ f x2 > 0 :=
by
  sorry

end zero_of_function_l141_141696


namespace bert_fraction_spent_l141_141524

theorem bert_fraction_spent (f : ℝ) :
  let initial := 52
  let after_hardware := initial - initial * f
  let after_cleaners := after_hardware - 9
  let after_grocery := after_cleaners / 2
  let final := 15
  after_grocery = final → f = 1/4 :=
by
  intros h
  sorry

end bert_fraction_spent_l141_141524


namespace find_a₁_l141_141153

noncomputable def S_3 (a₁ q : ℝ) : ℝ := a₁ + a₁ * q + a₁ * q^2

theorem find_a₁ (S₃_eq : S_3 a₁ q = a₁ + 3 * (a₁ * q)) (a₄_eq : a₁ * q^3 = 8) : a₁ = 1 :=
by
  -- proof skipped
  sorry

end find_a₁_l141_141153


namespace X_Y_Z_sum_eq_17_l141_141503

variable {X Y Z : ℤ}

def base_ten_representation_15_fac (X Y Z : ℤ) : Prop :=
  Z = 0 ∧ (28 + X + Y) % 9 = 8 ∧ (X - Y) % 11 = 11

theorem X_Y_Z_sum_eq_17 (X Y Z : ℤ) (h : base_ten_representation_15_fac X Y Z) : X + Y + Z = 17 :=
by
  sorry

end X_Y_Z_sum_eq_17_l141_141503


namespace gcd_g50_g52_l141_141420

def g (x : ℕ) : ℕ := x^2 - 2 * x + 2021

theorem gcd_g50_g52 : Nat.gcd (g 50) (g 52) = 1 := by
  sorry

end gcd_g50_g52_l141_141420


namespace english_textbook_cost_l141_141083

variable (cost_english_book : ℝ)

theorem english_textbook_cost :
  let geography_book_cost := 10.50
  let num_books := 35
  let total_order_cost := 630
  (num_books * cost_english_book + num_books * geography_book_cost = total_order_cost) →
  cost_english_book = 7.50 :=
by {
sorry
}

end english_textbook_cost_l141_141083


namespace solve_for_x_l141_141735

-- Define the custom operation
def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

-- Main statement to prove
theorem solve_for_x : (∃ x : ℝ, custom_mul 3 (custom_mul 4 x) = 10) ↔ (x = 7.5) :=
by
  sorry

end solve_for_x_l141_141735


namespace find_a_b_a_b_values_l141_141346

/-
Define the matrix M as given in the problem.
Define the constants a and b, and state the condition that proves their correct values such that M_inv = a * M + b * I.
-/

open Matrix

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2, 0;
     1, -3]

noncomputable def M_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![1/2, 0;
     1/6, -1/3]

theorem find_a_b :
  ∃ (a b : ℚ), (M⁻¹) = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

theorem a_b_values :
  (∃ (a b : ℚ), (M⁻¹) = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ)) ∧
  (∃ a b : ℚ, a = 1/6 ∧ b = 1/6) :=
sorry

end find_a_b_a_b_values_l141_141346


namespace quadratic_inequality_solution_l141_141997

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * x + a ≥ 0) → a ≥ 1 :=
by
  sorry

end quadratic_inequality_solution_l141_141997


namespace fill_in_square_l141_141918

variable {α : Type*} [CommRing α]

theorem fill_in_square (a b : α) (square : α) (h : square * 3 * a * b = 3 * a^2 * b) : square = a :=
sorry

end fill_in_square_l141_141918


namespace smallest_six_digit_number_exists_l141_141677

def three_digit_number (n : ℕ) := n % 4 = 2 ∧ n % 5 = 2 ∧ n % 6 = 2 ∧ 100 ≤ n ∧ n < 1000

def valid_six_digit_number (m n : ℕ) := 
  (m * 1000 + n) % 4 = 0 ∧ (m * 1000 + n) % 5 = 0 ∧ (m * 1000 + n) % 6 = 0 ∧ 
  three_digit_number n ∧ 0 ≤ m ∧ m < 1000

theorem smallest_six_digit_number_exists : 
  ∃ m n, valid_six_digit_number m n ∧ (∀ m' n', valid_six_digit_number m' n' → m * 1000 + n ≤ m' * 1000 + n') :=
sorry

end smallest_six_digit_number_exists_l141_141677


namespace problem1_problem2_l141_141931

-- Sub-problem 1
theorem problem1 (x y : ℝ) (h1 : 9 * x + 10 * y = 1810) (h2 : 11 * x + 8 * y = 1790) : 
  x - y = -10 := 
sorry

-- Sub-problem 2
theorem problem2 (x y : ℝ) (h1 : 2 * x + 2.5 * y = 1200) (h2 : 1000 * x + 900 * y = 530000) :
  x = 350 ∧ y = 200 := 
sorry

end problem1_problem2_l141_141931


namespace initial_hair_length_l141_141165

-- Definitions based on the conditions
def hair_cut_off : ℕ := 13
def current_hair_length : ℕ := 1

-- The problem statement to be proved
theorem initial_hair_length : (current_hair_length + hair_cut_off = 14) :=
by
  sorry

end initial_hair_length_l141_141165


namespace polynomial_integer_values_l141_141380

theorem polynomial_integer_values (a b c d : ℤ) (h1 : ∃ (n : ℤ), n = (a * (-1)^3 + b * (-1)^2 - c * (-1) - d))
  (h2 : ∃ (n : ℤ), n = (a * 0^3 + b * 0^2 - c * 0 - d))
  (h3 : ∃ (n : ℤ), n = (a * 1^3 + b * 1^2 - c * 1 - d))
  (h4 : ∃ (n : ℤ), n = (a * 2^3 + b * 2^2 - c * 2 - d)) :
  ∀ x : ℤ, ∃ m : ℤ, m = a * x^3 + b * x^2 - c * x - d :=
by {
  -- proof goes here
  sorry
}

end polynomial_integer_values_l141_141380


namespace range_of_m_l141_141324

theorem range_of_m (m : ℝ) : 0 < m ∧ m < 2 ↔ (2 - m > 0 ∧ - (1 / 2) * m < 0) := by
  sorry

end range_of_m_l141_141324


namespace Corey_goal_reachable_l141_141716

theorem Corey_goal_reachable :
  ∀ (goal balls_found_saturday balls_found_sunday additional_balls : ℕ),
    goal = 48 →
    balls_found_saturday = 16 →
    balls_found_sunday = 18 →
    additional_balls = goal - (balls_found_saturday + balls_found_sunday) →
    additional_balls = 14 :=
by
  intros goal balls_found_saturday balls_found_sunday additional_balls
  intro goal_eq
  intro saturday_eq
  intro sunday_eq
  intro additional_eq
  sorry

end Corey_goal_reachable_l141_141716


namespace find_z_l141_141865

open Complex

noncomputable def sqrt_five : ℝ := Real.sqrt 5

theorem find_z (z : ℂ) 
  (hz1 : z.re < 0) 
  (hz2 : z.im > 0) 
  (h_modulus : abs z = 3) 
  (h_real_part : z.re = -sqrt_five) : 
  z = -sqrt_five + 2 * I :=
by
  sorry

end find_z_l141_141865


namespace solution_set_of_inequality_l141_141975

theorem solution_set_of_inequality : 
  {x : ℝ | (x - 2) * (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by sorry

end solution_set_of_inequality_l141_141975


namespace total_flour_needed_l141_141604

noncomputable def katie_flour : ℝ := 3

noncomputable def sheila_flour : ℝ := katie_flour + 2

noncomputable def john_flour : ℝ := 1.5 * sheila_flour

theorem total_flour_needed :
  katie_flour + sheila_flour + john_flour = 15.5 :=
by
  sorry

end total_flour_needed_l141_141604


namespace intersection_of_M_and_N_l141_141472

def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 6 > 0}
def intersection := {x : ℝ | -4 ≤ x ∧ x ≤ -2 ∨ 3 < x ∧ x ≤ 7}

theorem intersection_of_M_and_N : M ∩ N = intersection := by
  sorry

end intersection_of_M_and_N_l141_141472


namespace min_students_orchestra_l141_141633

theorem min_students_orchestra (n : ℕ) 
  (h1 : n % 9 = 0)
  (h2 : n % 10 = 0)
  (h3 : n % 11 = 0) : 
  n ≥ 990 ∧ ∃ k, n = 990 * k :=
by
  sorry

end min_students_orchestra_l141_141633


namespace quilt_cost_proof_l141_141764

-- Definitions for conditions
def length := 7
def width := 8
def cost_per_sq_foot := 40

-- Definition for the calculation of the area
def area := length * width

-- Definition for the calculation of the cost
def total_cost := area * cost_per_sq_foot

-- Theorem stating the final proof
theorem quilt_cost_proof : total_cost = 2240 := by
  sorry

end quilt_cost_proof_l141_141764


namespace city_tax_problem_l141_141851

theorem city_tax_problem :
  ∃ (x y : ℕ), 
    ((x + 3000) * (y - 10) = x * y) ∧
    ((x - 1000) * (y + 10) = x * y) ∧
    (x = 3000) ∧
    (y = 20) ∧
    (x * y = 60000) :=
by
  sorry

end city_tax_problem_l141_141851


namespace relatively_prime_bound_l141_141224

theorem relatively_prime_bound {m n : ℕ} {a : ℕ → ℕ} (h1 : 1 < m) (h2 : 1 < n) (h3 : m ≥ n)
  (h4 : ∀ i j, i ≠ j → a i = a j → False) (h5 : ∀ i, a i ≤ m) (h6 : ∀ i j, i ≠ j → a i ∣ a j → a i = 1) 
  (x : ℝ) : ∃ i, dist (a i * x) (round (a i * x)) ≥ 2 / (m * (m + 1)) * dist x (round x) :=
sorry

end relatively_prime_bound_l141_141224


namespace pete_nickels_spent_l141_141639

-- Definitions based on conditions
def initial_amount_per_person : ℕ := 250 -- 250 cents for $2.50
def total_initial_amount : ℕ := 2 * initial_amount_per_person
def total_expense : ℕ := 200 -- they spent 200 cents in total
def raymond_dimes_left : ℕ := 7
def value_of_dime : ℕ := 10
def raymond_remaining_amount : ℕ := raymond_dimes_left * value_of_dime
def raymond_spent_amount : ℕ := total_expense - raymond_remaining_amount
def value_of_nickel : ℕ := 5

-- Theorem to prove Pete spent 14 nickels
theorem pete_nickels_spent : 
  (total_expense - raymond_spent_amount) / value_of_nickel = 14 :=
by
  sorry

end pete_nickels_spent_l141_141639


namespace beavers_fraction_l141_141669

theorem beavers_fraction (total_beavers : ℕ) (swim_percentage : ℕ) (work_percentage : ℕ) (fraction_working : ℕ) : 
total_beavers = 4 → 
swim_percentage = 75 → 
work_percentage = 100 - swim_percentage → 
fraction_working = 1 →
(work_percentage * total_beavers) / 100 = fraction_working → 
fraction_working / total_beavers = 1 / 4 :=
by 
  intros h1 h2 h3 h4 h5 
  sorry

end beavers_fraction_l141_141669


namespace g_g_g_g_of_2_eq_242_l141_141692

def g (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 3 * x + 2

theorem g_g_g_g_of_2_eq_242 : g (g (g (g 2))) = 242 :=
by
  sorry

end g_g_g_g_of_2_eq_242_l141_141692


namespace intersection_points_of_graphs_l141_141632

open Real

theorem intersection_points_of_graphs (f : ℝ → ℝ) (hf : Function.Injective f) :
  ∃! x : ℝ, (f (x^3) = f (x^6)) ∧ (x = -1 ∨ x = 0 ∨ x = 1) :=
by
  -- Provide the structure of the proof
  sorry

end intersection_points_of_graphs_l141_141632


namespace max_non_intersecting_diagonals_l141_141795

theorem max_non_intersecting_diagonals (n : ℕ) (h : n ≥ 3) :
  ∃ k, k ≤ n - 3 ∧ (∀ m, m > k → ¬(m ≤ n - 3)) :=
by
  sorry

end max_non_intersecting_diagonals_l141_141795


namespace find_cubic_polynomial_l141_141464

theorem find_cubic_polynomial (a b c d : ℚ) :
  (a + b + c + d = -5) →
  (8 * a + 4 * b + 2 * c + d = -8) →
  (27 * a + 9 * b + 3 * c + d = -17) →
  (64 * a + 16 * b + 4 * c + d = -34) →
  a = -1/3 ∧ b = -1 ∧ c = -2/3 ∧ d = -3 :=
by
  intros h1 h2 h3 h4
  sorry

end find_cubic_polynomial_l141_141464


namespace total_apples_picked_l141_141578

def benny_apples : Nat := 2
def dan_apples : Nat := 9

theorem total_apples_picked : benny_apples + dan_apples = 11 := 
by
  sorry

end total_apples_picked_l141_141578


namespace third_angle_is_90_triangle_is_right_l141_141872

-- Define the given angles
def angle1 : ℝ := 56
def angle2 : ℝ := 34

-- Define the sum of angles in a triangle
def angle_sum : ℝ := 180

-- Define the third angle
def third_angle : ℝ := angle_sum - angle1 - angle2

-- Prove that the third angle is 90 degrees
theorem third_angle_is_90 : third_angle = 90 := by
  sorry

-- Define the type of the triangle based on the largest angle
def is_right_triangle : Prop := third_angle = 90

-- Prove that the triangle is a right triangle
theorem triangle_is_right : is_right_triangle := by
  sorry

end third_angle_is_90_triangle_is_right_l141_141872


namespace product_of_fractions_is_eight_l141_141326

theorem product_of_fractions_is_eight :
  (8 / 4) * (14 / 7) * (20 / 10) * (25 / 50) * (9 / 18) * (12 / 6) * (21 / 42) * (16 / 8) = 8 :=
by
  sorry

end product_of_fractions_is_eight_l141_141326


namespace freken_bok_weight_l141_141306

variables (K F M : ℕ)

theorem freken_bok_weight 
  (h1 : K + F = M + 75) 
  (h2 : F + M = K + 45) : 
  F = 60 :=
sorry

end freken_bok_weight_l141_141306


namespace side_increase_percentage_l141_141637

theorem side_increase_percentage (s : ℝ) (p : ℝ) 
  (h : (s^2) * (1.5625) = (s * (1 + p / 100))^2) : p = 25 := 
sorry

end side_increase_percentage_l141_141637


namespace paving_stone_size_l141_141077

theorem paving_stone_size (length_courtyard width_courtyard : ℕ) (num_paving_stones : ℕ) (area_courtyard : ℕ) (s : ℕ)
  (h₁ : length_courtyard = 30) 
  (h₂ : width_courtyard = 18)
  (h₃ : num_paving_stones = 135)
  (h₄ : area_courtyard = length_courtyard * width_courtyard)
  (h₅ : area_courtyard = num_paving_stones * s * s) :
  s = 2 := 
by
  sorry

end paving_stone_size_l141_141077


namespace rectangle_area_l141_141442

open Real

theorem rectangle_area (A : ℝ) (s l w : ℝ) (h1 : A = 9 * sqrt 3) (h2 : A = (sqrt 3 / 4) * s^2)
  (h3 : w = s) (h4 : l = 3 * w) : w * l = 108 :=
by
  sorry

end rectangle_area_l141_141442


namespace map_distance_l141_141558

theorem map_distance
  (s d_m : ℝ) (d_r : ℝ)
  (h1 : s = 0.4)
  (h2 : d_r = 5.3)
  (h3 : d_m = 64) :
  (d_m * d_r / s) = 848 := by
  sorry

end map_distance_l141_141558


namespace solution_set_of_inequality_l141_141615

theorem solution_set_of_inequality (x : ℝ) :
  (abs x * (x - 1) ≥ 0) ↔ (x ≥ 1 ∨ x = 0) := 
by
  sorry

end solution_set_of_inequality_l141_141615


namespace find_a_l141_141403

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), (x - a)^2 + y^2 = (x^2 + (y-1)^2)) ∧ (¬ ∃ x y : ℝ, y = x + 1) → a = 1 :=
by
  sorry

end find_a_l141_141403


namespace smallest_number_satisfies_conditions_l141_141849

-- Define the number we are looking for
def number : ℕ := 391410

theorem smallest_number_satisfies_conditions :
  (number % 7 = 2) ∧
  (number % 11 = 2) ∧
  (number % 13 = 2) ∧
  (number % 17 = 3) ∧
  (number % 23 = 0) ∧
  (number % 5 = 0) :=
by
  -- We need to prove that 391410 satisfies all the given conditions.
  -- This proof will include detailed steps to verify each condition
  sorry

end smallest_number_satisfies_conditions_l141_141849


namespace northern_village_population_l141_141876

theorem northern_village_population
    (x : ℕ) -- Northern village population
    (western_village_population : ℕ := 400)
    (southern_village_population : ℕ := 200)
    (total_conscripted : ℕ := 60)
    (northern_village_conscripted : ℕ := 10)
    (h : (northern_village_conscripted : ℚ) / total_conscripted = (x : ℚ) / (x + western_village_population + southern_village_population)) : 
    x = 120 :=
    sorry

end northern_village_population_l141_141876


namespace turtle_marathon_time_l141_141841

/-- Given a marathon distance of 42 kilometers and 195 meters and a turtle's speed of 15 meters per minute,
prove that the turtle will reach the finish line in 1 day, 22 hours, and 53 minutes. -/
theorem turtle_marathon_time :
  let speed := 15 -- meters per minute
  let distance_km := 42 -- kilometers
  let distance_m := 195 -- meters
  let total_distance := distance_km * 1000 + distance_m -- total distance in meters
  let time_min := total_distance / speed -- time to complete the marathon in minutes
  let hours := time_min / 60 -- time to complete the marathon in hours (division and modulus)
  let minutes := time_min % 60 -- remaining minutes after converting total minutes to hours
  let days := hours / 24 -- time to complete the marathon in days (division and modulus)
  let remaining_hours := hours % 24 -- remaining hours after converting total hours to days
  (days, remaining_hours, minutes) = (1, 22, 53) -- expected result
:= 
sorry

end turtle_marathon_time_l141_141841


namespace ferry_P_travel_time_l141_141237

-- Define the conditions based on the problem statement
variables (t : ℝ) -- travel time of ferry P
def speed_P := 6 -- speed of ferry P in km/h
def speed_Q := speed_P + 3 -- speed of ferry Q in km/h
def distance_P := speed_P * t -- distance traveled by ferry P in km
def distance_Q := 3 * distance_P -- distance traveled by ferry Q in km
def time_Q := t + 3 -- travel time of ferry Q

-- Theorem to prove that travel time t for ferry P is 3 hours
theorem ferry_P_travel_time : time_Q * speed_Q = distance_Q → t = 3 :=
by {
  -- Since you've mentioned to include the statement only and not the proof,
  -- Therefore, the proof body is left as an exercise or represented by sorry.
  sorry
}

end ferry_P_travel_time_l141_141237


namespace segment_combination_l141_141664

theorem segment_combination (x y : ℕ) :
  7 * x + 12 * y = 100 ↔ (x, y) = (4, 6) :=
by
  sorry

end segment_combination_l141_141664


namespace prove_P_plus_V_eq_zero_l141_141021

variable (P Q R S T U V : ℤ)

-- Conditions in Lean
def sequence_conditions (P Q R S T U V : ℤ) :=
  S = 7 ∧
  P + Q + R = 27 ∧
  Q + R + S = 27 ∧
  R + S + T = 27 ∧
  S + T + U = 27 ∧
  T + U + V = 27 ∧
  U + V + P = 27

-- Assertion that needs to be proved
theorem prove_P_plus_V_eq_zero (P Q R S T U V : ℤ) (h : sequence_conditions P Q R S T U V) : 
  P + V = 0 := by
  sorry

end prove_P_plus_V_eq_zero_l141_141021


namespace halfway_between_ratios_l141_141750

theorem halfway_between_ratios :
  let a := (1 : ℚ) / 8
  let b := (1 : ℚ) / 3
  (a + b) / 2 = 11 / 48 := by
  sorry

end halfway_between_ratios_l141_141750


namespace optimal_purchasing_plan_l141_141120

def price_carnation := 5
def price_lily := 10
def total_flowers := 300
def max_carnations (x : ℕ) : Prop := x ≤ 2 * (total_flowers - x)

theorem optimal_purchasing_plan :
  ∃ (x y : ℕ), (x + y = total_flowers) ∧ (x = 200) ∧ (y = 100) ∧ (max_carnations x) ∧ 
  ∀ (x' y' : ℕ), (x' + y' = total_flowers) → max_carnations x' →
    (price_carnation * x + price_lily * y ≤ price_carnation * x' + price_lily * y') :=
by
  sorry

end optimal_purchasing_plan_l141_141120


namespace radius_of_circle_l141_141691

theorem radius_of_circle (r : ℝ) : 
  (∀ x : ℝ, (4 * x^2 + r = x) → (1 - 16 * r = 0)) → r = 1 / 16 :=
by
  intro H
  have h := H 0
  simp at h
  sorry

end radius_of_circle_l141_141691


namespace problem_statement_l141_141717

   def f (a : ℤ) : ℤ := a - 2
   def F (a b : ℤ) : ℤ := b^2 + a

   theorem problem_statement : F 3 (f 4) = 7 := by
     sorry
   
end problem_statement_l141_141717


namespace sum_of_w_l141_141038

def g (y : ℝ) : ℝ := (2 * y)^3 - 2 * (2 * y) + 5

theorem sum_of_w (w1 w2 w3 : ℝ)
  (hw1 : g (2 * w1) = 13)
  (hw2 : g (2 * w2) = 13)
  (hw3 : g (2 * w3) = 13) :
  w1 + w2 + w3 = -1 / 4 :=
sorry

end sum_of_w_l141_141038


namespace kurt_less_marbles_than_dennis_l141_141525

theorem kurt_less_marbles_than_dennis
  (Laurie_marbles : ℕ)
  (Kurt_marbles : ℕ)
  (Dennis_marbles : ℕ)
  (h1 : Laurie_marbles = 37)
  (h2 : Laurie_marbles = Kurt_marbles + 12)
  (h3 : Dennis_marbles = 70) :
  Dennis_marbles - Kurt_marbles = 45 := by
  sorry

end kurt_less_marbles_than_dennis_l141_141525


namespace books_in_library_final_l141_141428

variable (initial_books : ℕ) (books_taken_out_tuesday : ℕ) 
          (books_returned_wednesday : ℕ) (books_taken_out_thursday : ℕ)

def books_left_in_library (initial_books books_taken_out_tuesday 
                          books_returned_wednesday books_taken_out_thursday : ℕ) : ℕ :=
  initial_books - books_taken_out_tuesday + books_returned_wednesday - books_taken_out_thursday

theorem books_in_library_final 
  (initial_books := 250) 
  (books_taken_out_tuesday := 120) 
  (books_returned_wednesday := 35) 
  (books_taken_out_thursday := 15) :
  books_left_in_library initial_books books_taken_out_tuesday 
                        books_returned_wednesday books_taken_out_thursday = 150 :=
by 
  sorry

end books_in_library_final_l141_141428


namespace minimum_value_x_plus_2y_l141_141991

theorem minimum_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = x * y) : x + 2 * y ≥ 8 :=
sorry

end minimum_value_x_plus_2y_l141_141991


namespace john_lift_total_weight_l141_141555

-- Define the conditions as constants
def initial_weight : ℝ := 135
def weight_increase : ℝ := 265
def bracer_factor : ℝ := 6

-- Define a theorem to prove the total weight John can lift
theorem john_lift_total_weight : initial_weight + weight_increase + (initial_weight + weight_increase) * bracer_factor = 2800 := by
  -- proof here
  sorry

end john_lift_total_weight_l141_141555


namespace A_plus_B_zero_l141_141732

def f (A B x : ℝ) : ℝ := 3 * A * x + 2 * B
def g (A B x : ℝ) : ℝ := 2 * B * x + 3 * A

theorem A_plus_B_zero (A B : ℝ) (h1 : A ≠ B) (h2 : ∀ x : ℝ, f A B (g A B x) - g A B (f A B x) = 3 * (B - A)) :
  A + B = 0 :=
sorry

end A_plus_B_zero_l141_141732


namespace original_price_l141_141855

variable (p q : ℝ)

theorem original_price (x : ℝ)
  (hp : x * (1 + p / 100) * (1 - q / 100) = 1) :
  x = 10000 / (10000 + 100 * (p - q) - p * q) :=
sorry

end original_price_l141_141855


namespace square_simplify_l141_141292

   variable (y : ℝ)

   theorem square_simplify :
     (7 - Real.sqrt (y^2 - 49)) ^ 2 = y^2 - 14 * Real.sqrt (y^2 - 49) :=
   sorry
   
end square_simplify_l141_141292


namespace neg_two_squared_result_l141_141657

theorem neg_two_squared_result : -2^2 = -4 :=
by
  sorry

end neg_two_squared_result_l141_141657


namespace pentagon_area_sol_l141_141073

theorem pentagon_area_sol (a b : ℤ) (h1 : 0 < b) (h2 : b < a) (h3 : a * (3 * b + a) = 792) : a + b = 45 :=
sorry

end pentagon_area_sol_l141_141073


namespace boys_without_calculators_l141_141215

/-- In Mrs. Robinson's math class, there are 20 boys, and 30 of her students bring their calculators to class. 
    If 18 of the students who brought calculators are girls, then the number of boys who didn't bring their calculators is 8. -/
theorem boys_without_calculators (num_boys : ℕ) (num_students_with_calculators : ℕ) (num_girls_with_calculators : ℕ)
  (h1 : num_boys = 20)
  (h2 : num_students_with_calculators = 30)
  (h3 : num_girls_with_calculators = 18) :
  num_boys - (num_students_with_calculators - num_girls_with_calculators) = 8 :=
by 
  -- proof goes here
  sorry

end boys_without_calculators_l141_141215


namespace seven_digit_divisible_by_11_l141_141720

def is_digit (d : ℕ) : Prop := d ≤ 9

def valid7DigitNumber (b n : ℕ) : Prop :=
  let sum_odd := 3 + 5 + 6
  let sum_even := b + n + 7 + 8
  let diff := sum_odd - sum_even
  diff % 11 = 0

theorem seven_digit_divisible_by_11 (b n : ℕ) (hb : is_digit b) (hn : is_digit n)
  (h_valid : valid7DigitNumber b n) : b + n = 10 := 
sorry

end seven_digit_divisible_by_11_l141_141720


namespace correct_statements_l141_141845

def f (x : ℝ) (b : ℝ) (c : ℝ) := x * (abs x) + b * x + c

theorem correct_statements (b c : ℝ) :
  (∀ x, c = 0 → f (-x) b 0 = - f x b 0) ∧
  (∀ x, b = 0 → c > 0 → (f x 0 c = 0 → x = 0) ∧ ∀ y, f y 0 c ≤ 0) ∧
  (∀ x, ∃ k : ℝ, f (k + x) b c = f (k - x) b c) ∧
  ¬(∀ x, x > 0 → f x b c = c - b^2 / 2) :=
by
  sorry

end correct_statements_l141_141845


namespace range_of_m_for_one_real_root_l141_141952

def f (x : ℝ) (m : ℝ) : ℝ := x^3 - 3*x + m

theorem range_of_m_for_one_real_root :
  (∃! x : ℝ, f x m = 0) ↔ (m < -2 ∨ m > 2) := by
  sorry

end range_of_m_for_one_real_root_l141_141952


namespace monica_problem_l141_141412

open Real

noncomputable def completingSquare : Prop :=
  ∃ (b c : ℤ), (∀ x : ℝ, (x - 4) ^ 2 = x^2 - 8 * x + 16) ∧ b = -4 ∧ c = 8 ∧ (b + c = 4)

theorem monica_problem : completingSquare := by
  sorry

end monica_problem_l141_141412


namespace initial_gasohol_amount_l141_141181

variable (x : ℝ)

def gasohol_ethanol_percentage (initial_gasohol : ℝ) := 0.05 * initial_gasohol
def mixture_ethanol_percentage (initial_gasohol : ℝ) := gasohol_ethanol_percentage initial_gasohol + 3

def optimal_mixture (total_volume : ℝ) := 0.10 * total_volume

theorem initial_gasohol_amount :
  ∀ (initial_gasohol : ℝ), 
  mixture_ethanol_percentage initial_gasohol = optimal_mixture (initial_gasohol + 3) →
  initial_gasohol = 54 :=
by
  intros
  sorry

end initial_gasohol_amount_l141_141181


namespace evaluate_expression_l141_141527

theorem evaluate_expression (a b c : ℚ) (ha : a = 1/2) (hb : b = 1/4) (hc : c = 5) :
  a^2 * b^3 * c = 5 / 256 :=
by
  rw [ha, hb, hc]
  norm_num

end evaluate_expression_l141_141527


namespace remainder_of_N_mod_103_l141_141926

noncomputable def N : ℕ :=
  sorry -- This will capture the mathematical calculation of N using the conditions stated.

theorem remainder_of_N_mod_103 : (N % 103) = 43 :=
  sorry

end remainder_of_N_mod_103_l141_141926


namespace julia_tag_kids_monday_l141_141448

-- Definitions based on conditions
def total_tag_kids (M T : ℕ) : Prop := M + T = 20
def tag_kids_Tuesday := 13

-- Problem statement
theorem julia_tag_kids_monday (M : ℕ) : total_tag_kids M tag_kids_Tuesday → M = 7 := 
by
  intro h
  sorry

end julia_tag_kids_monday_l141_141448


namespace probability_increase_l141_141061

theorem probability_increase:
  let P_win1 := 0.30
  let P_lose1 := 0.70
  let P_win2 := 0.50
  let P_lose2 := 0.50
  let P_win3 := 0.40
  let P_lose3 := 0.60
  let P_win4 := 0.25
  let P_lose4 := 0.75
  let P_win_all := P_win1 * P_win2 * P_win3 * P_win4
  let P_lose_all := P_lose1 * P_lose2 * P_lose3 * P_lose4
  (P_lose_all - P_win_all) / P_win_all = 9.5 :=
by
  sorry

end probability_increase_l141_141061


namespace Deepak_age_l141_141124

theorem Deepak_age (A D : ℕ) (h1 : A / D = 4 / 3) (h2 : A + 6 = 26) : D = 15 :=
by
  sorry

end Deepak_age_l141_141124


namespace distance_ratio_l141_141769

variable (d_RB d_BC : ℝ)

theorem distance_ratio
    (h1 : d_RB / 60 + d_BC / 20 ≠ 0)
    (h2 : 36 * (d_RB / 60 + d_BC / 20) = d_RB + d_BC) : 
    d_RB / d_BC = 2 := 
sorry

end distance_ratio_l141_141769


namespace sqrt_identity_l141_141996

theorem sqrt_identity (x : ℝ) (hx : x = Real.sqrt 5 - 3) : Real.sqrt (x^2 + 6*x + 9) = Real.sqrt 5 :=
by
  sorry

end sqrt_identity_l141_141996


namespace interest_rate_correct_l141_141084

noncomputable def annual_interest_rate : ℝ :=
  4^(1/10) - 1

theorem interest_rate_correct (P A₁₀ A₁₅ : ℝ) (h₁ : P = 6000) (h₂ : A₁₀ = 24000) (h₃ : A₁₅ = 48000) :
  (P * (1 + annual_interest_rate)^10 = A₁₀) ∧ (P * (1 + annual_interest_rate)^15 = A₁₅) :=
by
  sorry

end interest_rate_correct_l141_141084


namespace fraction_comparison_l141_141167

theorem fraction_comparison : 
  (15 / 11 : ℝ) > (17 / 13 : ℝ) ∧ (17 / 13 : ℝ) > (19 / 15 : ℝ) :=
by
  sorry

end fraction_comparison_l141_141167


namespace dacid_average_marks_is_75_l141_141356

/-- Defining the marks obtained in each subject as constants -/
def english_marks : ℕ := 76
def mathematics_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85

/-- Total marks calculation -/
def total_marks : ℕ :=
  english_marks + mathematics_marks + physics_marks + chemistry_marks + biology_marks

/-- Number of subjects -/
def number_of_subjects : ℕ := 5

/-- Average marks calculation -/
def average_marks : ℕ :=
  total_marks / number_of_subjects

/-- Theorem proving that Dacid's average marks is 75 -/
theorem dacid_average_marks_is_75 : average_marks = 75 :=
  sorry

end dacid_average_marks_is_75_l141_141356


namespace Randy_trip_distance_l141_141556

noncomputable def total_distance (x : ℝ) :=
  (x / 4) + 40 + 10 + (x / 6)

theorem Randy_trip_distance (x : ℝ) (h : total_distance x = x) : x = 600 / 7 :=
by
  sorry

end Randy_trip_distance_l141_141556


namespace longest_side_triangle_l141_141500

theorem longest_side_triangle (x : ℝ) 
  (h1 : 7 + (x + 4) + (2 * x + 1) = 36) : 
  max 7 (max (x + 4) (2 * x + 1)) = 17 :=
by sorry

end longest_side_triangle_l141_141500


namespace number_of_perfect_squares_and_cubes_l141_141460

theorem number_of_perfect_squares_and_cubes (n m k : ℕ) (hn : n^2 < 1000) (hn' : (n + 1)^2 ≥ 1000) (hm : m^3 < 1000) (hm' : (m + 1)^3 ≥ 1000) (hk : k^6 < 1000) (hk' : (k + 1)^6 ≥ 1000) :
  (n + m - k) = 38 :=
sorry

end number_of_perfect_squares_and_cubes_l141_141460


namespace product_expression_evaluates_to_32_l141_141728

theorem product_expression_evaluates_to_32 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by
  -- The proof itself is not required, hence we can put sorry here
  sorry

end product_expression_evaluates_to_32_l141_141728


namespace theodoreEarningsCorrect_l141_141642

noncomputable def theodoreEarnings : ℝ := 
  let s := 10
  let ps := 20
  let w := 20
  let pw := 5
  let b := 15
  let pb := 15
  let m := 150
  let l := 200
  let t := 0.10
  let totalEarnings := (s * ps) + (w * pw) + (b * pb)
  let expenses := m + l
  let earningsBeforeTaxes := totalEarnings - expenses
  let taxes := t * earningsBeforeTaxes
  earningsBeforeTaxes - taxes

theorem theodoreEarningsCorrect :
  theodoreEarnings = 157.50 :=
by sorry

end theodoreEarningsCorrect_l141_141642


namespace each_person_paid_l141_141785

-- Define the conditions: total bill and number of people
def totalBill : ℕ := 135
def numPeople : ℕ := 3

-- Define the question as a theorem to prove the correct answer
theorem each_person_paid : totalBill / numPeople = 45 :=
by
  -- Here, we can skip the proof since the statement is required only.
  sorry

end each_person_paid_l141_141785


namespace taeyeon_height_proof_l141_141536

noncomputable def seonghee_height : ℝ := 134.5
noncomputable def taeyeon_height : ℝ := seonghee_height * 1.06

theorem taeyeon_height_proof : taeyeon_height = 142.57 := 
by
  sorry

end taeyeon_height_proof_l141_141536


namespace find_f_inv_value_l141_141353

noncomputable def f (x : ℝ) : ℝ := 8^x
noncomputable def f_inv (y : ℝ) : ℝ := Real.logb 8 y

theorem find_f_inv_value (a : ℝ) (h : a = 8^(1/3)) : f_inv (a + 2) = Real.logb 8 (8^(1/3) + 2) := by
  sorry

end find_f_inv_value_l141_141353


namespace find_b_l141_141925

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  2 * x / (x^2 + b * x + 1)

noncomputable def f_inverse (y : ℝ) : ℝ :=
  (1 - y) / y

theorem find_b (b : ℝ) (h : ∀ x, f_inverse (f x b) = x) : b = 4 :=
sorry

end find_b_l141_141925


namespace time_train_passes_jogger_l141_141940

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600

noncomputable def train_speed_kmph : ℝ := 45
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps

noncomputable def initial_lead_m : ℝ := 150
noncomputable def train_length_m : ℝ := 100

noncomputable def total_distance_to_cover_m : ℝ := initial_lead_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ := total_distance_to_cover_m / relative_speed_mps

theorem time_train_passes_jogger : time_to_pass_jogger_s = 25 := by
  sorry

end time_train_passes_jogger_l141_141940


namespace golf_money_l141_141794

-- Definitions based on conditions
def cost_per_round : ℤ := 80
def number_of_rounds : ℤ := 5

-- The theorem/problem statement
theorem golf_money : cost_per_round * number_of_rounds = 400 := 
by {
  -- Proof steps would go here, but to skip the proof, we use sorry
  sorry
}

end golf_money_l141_141794


namespace inequality_x_y_z_squares_l141_141405

theorem inequality_x_y_z_squares (x y z m : ℝ) (h : x + y + z = m) : x^2 + y^2 + z^2 ≥ (m^2) / 3 := by
  sorry

end inequality_x_y_z_squares_l141_141405


namespace equal_share_of_candles_l141_141329

-- Define conditions
def ambika_candles : ℕ := 4
def aniyah_candles : ℕ := 6 * ambika_candles
def bree_candles : ℕ := 2 * aniyah_candles
def caleb_candles : ℕ := bree_candles + (bree_candles / 2)

-- Define the total candles and the equal share
def total_candles : ℕ := ambika_candles + aniyah_candles + bree_candles + caleb_candles
def each_share : ℕ := total_candles / 4

-- State the problem
theorem equal_share_of_candles : each_share = 37 := by
  sorry

end equal_share_of_candles_l141_141329


namespace initial_tomatoes_l141_141772

def t_picked : ℕ := 83
def t_left : ℕ := 14
def t_total : ℕ := t_picked + t_left

theorem initial_tomatoes : t_total = 97 := by
  rw [t_total]
  rfl

end initial_tomatoes_l141_141772


namespace correct_option_is_A_l141_141296

-- Define the conditions
def chromosome_counts (phase : String) (is_meiosis : Bool) : Nat :=
  if phase = "prophase" then 2
  else if phase = "metaphase" then 2
  else if phase = "anaphase" then if is_meiosis then 2 else 4
  else if phase = "telophase" then if is_meiosis then 1 else 2
  else 0

def dna_counts (phase : String) (is_meiosis : Bool) : Nat :=
  if phase = "prophase" then 4
  else if phase = "metaphase" then 4
  else if phase = "anaphase" then 4
  else if phase = "telophase" then 2
  else 0

def chromosome_behavior (phase : String) (is_meiosis : Bool) : String :=
  if is_meiosis && phase = "prophase" then "synapsis"
  else if is_meiosis && phase = "metaphase" then "tetrad formation"
  else if is_meiosis && phase = "anaphase" then "separation"
  else if is_meiosis && phase = "telophase" then "recombination"
  else "no special behavior"

-- Problem statement in terms of a Lean theorem
theorem correct_option_is_A :
  ∀ (phase : String),
  (chromosome_counts phase false = chromosome_counts phase true ∧
   chromosome_behavior phase false ≠ chromosome_behavior phase true ∧
   dna_counts phase false ≠ dna_counts phase true) →
  "A" = "A" :=
by 
  intro phase 
  simp only [imp_self]
  sorry

end correct_option_is_A_l141_141296


namespace continuous_stripe_probability_l141_141539

-- Definitions based on conditions from a)
def total_possible_combinations : ℕ := 4^6

def favorable_outcomes : ℕ := 12

def probability_of_continuous_stripe : ℚ := favorable_outcomes / total_possible_combinations

-- The theorem equivalent to prove the given problem
theorem continuous_stripe_probability :
  probability_of_continuous_stripe = 3 / 1024 :=
by
  sorry

end continuous_stripe_probability_l141_141539


namespace cards_not_in_box_correct_l141_141256

-- Total number of cards Robie had at the beginning.
def total_cards : ℕ := 75

-- Number of cards in each box.
def cards_per_box : ℕ := 10

-- Number of boxes Robie gave away.
def boxes_given_away : ℕ := 2

-- Number of boxes Robie has with him.
def boxes_with_rob : ℕ := 5

-- The number of cards not placed in a box.
def cards_not_in_box : ℕ :=
  total_cards - (boxes_given_away * cards_per_box + boxes_with_rob * cards_per_box)

theorem cards_not_in_box_correct : cards_not_in_box = 5 :=
by
  unfold cards_not_in_box
  unfold total_cards
  unfold boxes_given_away
  unfold cards_per_box
  unfold boxes_with_rob
  sorry

end cards_not_in_box_correct_l141_141256


namespace find_pq_l141_141654

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_pq (p q : ℕ) 
(hp : is_prime p) 
(hq : is_prime q) 
(h : is_prime (q^2 - p^2)) : 
  p * q = 6 :=
by sorry

end find_pq_l141_141654


namespace exists_infinitely_many_n_l141_141130

def sum_of_digits (m : ℕ) : ℕ := 
  m.digits 10 |>.sum

theorem exists_infinitely_many_n (S : ℕ → ℕ) (h_sum_of_digits : ∀ m, S m = sum_of_digits m) :
  ∀ N : ℕ, ∃ n ≥ N, S (3 ^ n) ≥ S (3 ^ (n + 1)) :=
by { sorry }

end exists_infinitely_many_n_l141_141130


namespace range_of_a_l141_141319

def is_monotonically_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, (0 < x) → (x < y) → (f x ≤ f y)

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

theorem range_of_a (a : ℝ) : 
  is_monotonically_increasing (f a) a → a ≤ 2 :=
sorry

end range_of_a_l141_141319


namespace number_of_blue_balls_l141_141359

theorem number_of_blue_balls (b : ℕ) 
  (h1 : 0 < b ∧ b ≤ 15)
  (prob : (b / 15) * ((b - 1) / 14) = 1 / 21) :
  b = 5 := sorry

end number_of_blue_balls_l141_141359


namespace determine_c_l141_141471

theorem determine_c (c : ℚ) : (∀ x : ℝ, (x + 7) * (x^2 * c * x + 19 * x^2 - c * x - 49) = 0) → c = 21 / 8 :=
by
  sorry

end determine_c_l141_141471


namespace parabola_distance_x_coord_l141_141573

theorem parabola_distance_x_coord
  (M : ℝ × ℝ) 
  (hM : M.2^2 = 4 * M.1)
  (hMF : (M.1 - 1)^2 + M.2^2 = 4^2)
  : M.1 = 3 :=
sorry

end parabola_distance_x_coord_l141_141573


namespace part1_part2_l141_141566

def f (x a : ℝ) : ℝ := |x + 1| + 2 * |x - a|

theorem part1 (x : ℝ) : (∀ x, f x 2 ≤ x + 4 → (1 / 2 ≤ x ∧ x ≤ 7 / 2)) :=
by sorry

theorem part2 (x : ℝ) : (∀ x, f x a ≥ 4) ↔ (a ≤ -5 ∨ a ≥ 3) :=
by sorry

end part1_part2_l141_141566


namespace contrapositive_roots_l141_141668

theorem contrapositive_roots {a b c : ℝ} (h : a ≠ 0) (hac : a * c ≤ 0) :
  ¬ (∀ x : ℝ, (a * x^2 - b * x + c = 0) → x > 0) :=
sorry

end contrapositive_roots_l141_141668


namespace profit_per_meter_is_25_l141_141126

def sell_price : ℕ := 8925
def cost_price_per_meter : ℕ := 80
def meters_sold : ℕ := 85
def total_cost_price : ℕ := cost_price_per_meter * meters_sold
def total_profit : ℕ := sell_price - total_cost_price
def profit_per_meter : ℕ := total_profit / meters_sold

theorem profit_per_meter_is_25 : profit_per_meter = 25 := by
  sorry

end profit_per_meter_is_25_l141_141126


namespace dodecagon_area_l141_141789

theorem dodecagon_area (s : ℝ) (n : ℕ) (angles : ℕ → ℝ)
  (h_s : s = 10) (h_n : n = 12) 
  (h_angles : ∀ i, angles i = if i % 3 == 2 then 270 else 90) :
  ∃ area : ℝ, area = 500 := 
sorry

end dodecagon_area_l141_141789


namespace common_remainder_zero_l141_141621

theorem common_remainder_zero (n r : ℕ) (h1: n > 1) 
(h2 : n % 25 = r) (h3 : n % 7 = r) (h4 : n = 175) : r = 0 :=
by
  sorry

end common_remainder_zero_l141_141621


namespace find_k_l141_141542

noncomputable def a : ℚ := sorry -- Represents positive rational number a
noncomputable def b : ℚ := sorry -- Represents positive rational number b

def minimal_period (x : ℚ) : ℕ := sorry -- Function to determine minimal period of a rational number

-- Conditions as definitions
axiom h1 : minimal_period a = 30
axiom h2 : minimal_period b = 30
axiom h3 : minimal_period (a - b) = 15

-- Statement to prove smallest natural number k such that minimal period of (a + k * b) is 15
theorem find_k : ∃ k : ℕ, minimal_period (a + k * b) = 15 ∧ ∀ n < k, minimal_period (a + n * b) ≠ 15 :=
sorry

end find_k_l141_141542


namespace jame_initial_gold_bars_l141_141078

theorem jame_initial_gold_bars (X : ℝ) (h1 : X * 0.1 + 0.5 * (X * 0.9) = 0.5 * (X * 0.9) - 27) :
  X = 60 :=
by
-- Placeholder for proof
sorry

end jame_initial_gold_bars_l141_141078


namespace find_pqr_eq_1680_l141_141122

theorem find_pqr_eq_1680
  {p q r : ℤ} (hpqz : p ≠ 0) (hqqz : q ≠ 0) (hrqz : r ≠ 0)
  (h_sum : p + q + r = 30)
  (h_cond : (1:ℚ) / p + (1:ℚ) / q + (1:ℚ) / r + 390 / (p * q * r) = 1) :
  p * q * r = 1680 :=
sorry

end find_pqr_eq_1680_l141_141122


namespace problem_statement_l141_141485

theorem problem_statement (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end problem_statement_l141_141485


namespace ratio_of_girls_to_boys_l141_141990

theorem ratio_of_girls_to_boys (total_students girls boys : ℕ) 
  (h1 : total_students = 26) 
  (h2 : girls = boys + 6) 
  (h3 : girls + boys = total_students) : 
  (girls : ℚ) / boys = 8 / 5 :=
by
  sorry

end ratio_of_girls_to_boys_l141_141990


namespace part1_part2_l141_141026

-- Part (1)
theorem part1 (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) 
  (h1 : ∀ n, S_n n = n * a_n 1 + (n-1) * a_n 2 + 2 * a_n (n-1) + a_n n)
  (arithmetic_seq : ∀ n, a_n (n+1) = a_n n + d)
  (S1_eq : S_n 1 = 5)
  (S2_eq : S_n 2 = 18) :
  ∀ n, a_n n = 3 * n + 2 := by
  sorry

-- Part (2)
theorem part2 (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) 
  (h1 : ∀ n, S_n n = n * a_n 1 + (n-1) * a_n 2 + 2 * a_n (n-1) + a_n n)
  (geometric_seq : ∃ q, ∀ n, a_n (n+1) = q * a_n n)
  (S1_eq : S_n 1 = 3)
  (S2_eq : S_n 2 = 15) :
  ∀ n, S_n n = (3^(n+2) - 6 * n - 9) / 4 := by
  sorry

end part1_part2_l141_141026


namespace royalty_amount_l141_141479

-- Define the conditions and the question proof.
theorem royalty_amount (x : ℝ) :
  (800 ≤ x ∧ x ≤ 4000 → (x - 800) * 0.14 = 420) ∧
  (x > 4000 → x * 0.11 = 420) ∧
  420 = 420 →
  x = 3800 :=
by
  sorry

end royalty_amount_l141_141479


namespace simplify_expression_l141_141896

theorem simplify_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 :=
by
  sorry

end simplify_expression_l141_141896


namespace students_more_than_guinea_pigs_l141_141135

-- Definitions based on the problem's conditions
def students_per_classroom : Nat := 22
def guinea_pigs_per_classroom : Nat := 3
def classrooms : Nat := 5

-- The proof statement
theorem students_more_than_guinea_pigs :
  (students_per_classroom * classrooms) - (guinea_pigs_per_classroom * classrooms) = 95 :=
by
  sorry

end students_more_than_guinea_pigs_l141_141135


namespace factorize_expression_l141_141443

theorem factorize_expression (m n : ℤ) : m^2 * n - 9 * n = n * (m + 3) * (m - 3) := by
  sorry

end factorize_expression_l141_141443


namespace college_application_distributions_l141_141776

theorem college_application_distributions : 
  let total_students := 6
  let colleges := 3
  ∃ n : ℕ, n = 540 ∧ 
    (n = (colleges^total_students - colleges * (2^total_students) + 
      (colleges.choose 2) * 1)) := sorry

end college_application_distributions_l141_141776


namespace find_f2_l141_141203

-- A condition of the problem is the specific form of the function
def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

-- Given condition
theorem find_f2 (a b : ℝ) (h : f (-2) a b = 3) : f 2 a b = -19 :=
by
  sorry

end find_f2_l141_141203


namespace yoongi_has_5_carrots_l141_141877

def yoongis_carrots (initial_carrots sister_gave: ℕ) : ℕ :=
  initial_carrots + sister_gave

theorem yoongi_has_5_carrots : yoongis_carrots 3 2 = 5 := by 
  sorry

end yoongi_has_5_carrots_l141_141877


namespace min_distance_origin_to_intersections_l141_141913

theorem min_distance_origin_to_intersections (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hline : (1 : ℝ)/a + 4/b = 1) :
  |(0 : ℝ) - a| + |(0 : ℝ) - b| = 9 :=
sorry

end min_distance_origin_to_intersections_l141_141913


namespace task1_on_time_task2_not_on_time_l141_141499

/-- Define the probabilities for task 1 and task 2 -/
def P_A : ℚ := 3 / 8
def P_B : ℚ := 3 / 5

/-- The probability that task 1 will be completed on time but task 2 will not is 3 / 20. -/
theorem task1_on_time_task2_not_on_time (P_A : ℚ) (P_B : ℚ) : P_A = 3 / 8 → P_B = 3 / 5 → P_A * (1 - P_B) = 3 / 20 :=
by
  intros hPA hPB
  rw [hPA, hPB]
  norm_num

end task1_on_time_task2_not_on_time_l141_141499


namespace point_A_equidistant_l141_141878

/-
This statement defines the problem of finding the coordinates of point A that is equidistant from points B and C.
-/
theorem point_A_equidistant (x : ℝ) :
  (dist (x, 0, 0) (3, 5, 6)) = (dist (x, 0, 0) (1, 2, 3)) ↔ x = 14 :=
by {
  sorry
}

end point_A_equidistant_l141_141878


namespace ratio_c_b_l141_141700

theorem ratio_c_b (x y a b c : ℝ) (h1 : x ≥ 1) (h2 : x + y ≤ 4) (h3 : a * x + b * y + c ≤ 0) 
    (h_max : ∀ x y, (x,y) = (2, 2) → 2 * x + y = 6) (h_min : ∀ x y, (x,y) = (1, -1) → 2 * x + y = 1) (h_b : b ≠ 0) :
    c / b = 4 := sorry

end ratio_c_b_l141_141700


namespace Tobias_monthly_allowance_l141_141800

noncomputable def monthly_allowance (shoes_cost monthly_saving_period lawn_charge driveway_charge change num_lawns num_driveways : ℕ) : ℕ :=
  (shoes_cost + change - (num_lawns * lawn_charge + num_driveways * driveway_charge)) / monthly_saving_period

theorem Tobias_monthly_allowance :
  let shoes_cost := 95
  let monthly_saving_period := 3
  let lawn_charge := 15
  let driveway_charge := 7
  let change := 15
  let num_lawns := 4
  let num_driveways := 5
  monthly_allowance shoes_cost monthly_saving_period lawn_charge driveway_charge change num_lawns num_driveways = 5 :=
by
  sorry

end Tobias_monthly_allowance_l141_141800


namespace find_sum_of_xyz_l141_141383

theorem find_sum_of_xyz (x y z : ℕ) (h1 : 0 < x ∧ 0 < y ∧ 0 < z)
  (h2 : (x + y + z)^3 - x^3 - y^3 - z^3 = 300) : x + y + z = 7 :=
by
  sorry

end find_sum_of_xyz_l141_141383


namespace grazing_months_l141_141822

theorem grazing_months :
  ∀ (m : ℕ),
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let b_months := 5
  let c_oxen := 15
  let rent := 210
  let c_share := 54
  let a_ox_months := a_oxen * a_months
  let b_ox_months := b_oxen * b_months
  let c_ox_months := c_oxen * m
  let total_ox_months := a_ox_months + b_ox_months + c_ox_months
  let c_part := (c_ox_months : ℝ) / (total_ox_months : ℝ) * rent
  (c_part = c_share) → m = 3 :=
by { sorry }

end grazing_months_l141_141822


namespace period_ending_time_l141_141848

theorem period_ending_time (start_time : ℕ) (rain_duration : ℕ) (no_rain_duration : ℕ) (end_time : ℕ) :
  start_time = 8 ∧ rain_duration = 4 ∧ no_rain_duration = 5 ∧ end_time = 8 + rain_duration + no_rain_duration
  → end_time = 17 :=
by
  sorry

end period_ending_time_l141_141848


namespace chord_length_ne_l141_141107

-- Define the ellipse
def ellipse (x y : ℝ) := (x^2 / 8) + (y^2 / 4) = 1

-- Define the first line
def line_l (k x : ℝ) := (k * x + 1)

-- Define the second line
def line_l_option_D (k x y : ℝ) := (k * x + y - 2)

-- Prove the chord length inequality for line_l_option_D
theorem chord_length_ne (k : ℝ) :
  ∀ x y : ℝ, ellipse x y →
  ∃ x1 x2 y1 y2 : ℝ, ellipse x1 y1 ∧ line_l k x1 = y1 ∧ ellipse x2 y2 ∧ line_l k x2 = y2 ∧
  ∀ x3 x4 y3 y4 : ℝ, ellipse x3 y3 ∧ line_l_option_D k x3 y3 = 0 ∧ ellipse x4 y4 ∧ line_l_option_D k x4 y4 = 0 →
  dist (x1, y1) (x2, y2) ≠ dist (x3, y3) (x4, y4) :=
sorry

end chord_length_ne_l141_141107


namespace part_a_part_b_part_c_l141_141328

-- Given conditions and questions
variable (x y : ℝ)
variable (h : (x - y)^2 - 2 * (x + y) + 1 = 0)

-- Part (a): Prove neither x nor y can be negative
theorem part_a (h : (x - y)^2 - 2 * (x + y) + 1 = 0) : x ≥ 0 ∧ y ≥ 0 := 
sorry

-- Part (b): Prove if x > 1 and y < x, then sqrt{x} - sqrt{y} = 1
theorem part_b (h : (x - y)^2 - 2 * (x + y) + 1 = 0) (hx : x > 1) (hy : y < x) : 
  Real.sqrt x - Real.sqrt y = 1 := 
sorry

-- Part (c): Prove if x < 1 and y < 1, then sqrt{x} + sqrt{y} = 1
theorem part_c (h : (x - y)^2 - 2 * (x + y) + 1 = 0) (hx : x < 1) (hy : y < 1) : 
  Real.sqrt x + Real.sqrt y = 1 := 
sorry

end part_a_part_b_part_c_l141_141328


namespace system_solution_l141_141192

theorem system_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : 0 < x₃) (h₄ : 0 < x₄) (h₅ : 0 < x₅)
  (h₆ : x₁ + x₂ = x₃^2) (h₇ : x₃ + x₄ = x₅^2) (h₈ : x₄ + x₅ = x₁^2) (h₉ : x₅ + x₁ = x₂^2) :
  x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 := 
by 
  sorry

end system_solution_l141_141192


namespace h_h_3_eq_2915_l141_141510

def h (x : ℕ) : ℕ := 3 * x^2 + x + 1

theorem h_h_3_eq_2915 : h (h 3) = 2915 := by
  sorry

end h_h_3_eq_2915_l141_141510


namespace find_integer_l141_141041

theorem find_integer (n : ℤ) (h1 : n ≥ 50) (h2 : n ≤ 100) (h3 : n % 7 = 0) (h4 : n % 9 = 3) (h5 : n % 6 = 3) : n = 84 := 
by 
  sorry

end find_integer_l141_141041


namespace no_four_distinct_integers_with_product_plus_2006_perfect_square_l141_141729

theorem no_four_distinct_integers_with_product_plus_2006_perfect_square : 
  ¬ ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (∃ k1 k2 k3 k4 k5 k6 : ℕ, a * b + 2006 = k1^2 ∧ 
                          a * c + 2006 = k2^2 ∧ 
                          a * d + 2006 = k3^2 ∧ 
                          b * c + 2006 = k4^2 ∧ 
                          b * d + 2006 = k5^2 ∧ 
                          c * d + 2006 = k6^2) := 
sorry

end no_four_distinct_integers_with_product_plus_2006_perfect_square_l141_141729


namespace reported_length_correct_l141_141065

def length_in_yards := 80
def conversion_factor := 3 -- 1 yard is 3 feet
def length_in_feet := 240

theorem reported_length_correct :
  length_in_feet = length_in_yards * conversion_factor :=
by rfl

end reported_length_correct_l141_141065


namespace f_odd_function_no_parallel_lines_l141_141239

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - (1 / a^x))

theorem f_odd_function {a : ℝ} (h_pos : a > 0) (h_ne : a ≠ 1) : 
  ∀ x : ℝ, f a (-x) = -f a x := 
by
  sorry

theorem no_parallel_lines {a : ℝ} (h_pos : a > 0) (h_ne : a ≠ 1) : 
  ∀ x1 x2 : ℝ, x1 ≠ x2 → f a x1 ≠ f a x2 :=
by
  sorry

end f_odd_function_no_parallel_lines_l141_141239


namespace ages_total_l141_141308

theorem ages_total (P Q : ℕ) (h1 : P - 8 = (1 / 2) * (Q - 8)) (h2 : P / Q = 3 / 4) : P + Q = 28 :=
by
  sorry

end ages_total_l141_141308


namespace largest_angle_sine_of_C_l141_141713

-- Given conditions
def side_a : ℝ := 7
def side_b : ℝ := 3
def side_c : ℝ := 5

-- 1. Prove the largest angle
theorem largest_angle (a b c : ℝ) (h₁ : a = 7) (h₂ : b = 3) (h₃ : c = 5) : 
  ∃ A : ℝ, A = 120 :=
by
  sorry

-- 2. Prove the sine value of angle C
theorem sine_of_C (a b c A : ℝ) (h₁ : a = 7) (h₂ : b = 3) (h₃ : c = 5) (h₄ : A = 120) : 
  ∃ sinC : ℝ, sinC = 5 * (Real.sqrt 3) / 14 :=
by
  sorry

end largest_angle_sine_of_C_l141_141713


namespace jerry_cut_maple_trees_l141_141376

theorem jerry_cut_maple_trees :
  (∀ pine maple walnut : ℕ, 
    pine = 8 * 80 ∧ 
    walnut = 4 * 100 ∧ 
    1220 = pine + walnut + maple * 60) → 
  maple = 3 := 
by 
  sorry

end jerry_cut_maple_trees_l141_141376


namespace total_number_of_rats_l141_141575

theorem total_number_of_rats (Kenia Hunter Elodie Teagan : ℕ) 
  (h1 : Elodie = 30)
  (h2 : Elodie = Hunter + 10)
  (h3 : Kenia = 3 * (Hunter + Elodie))
  (h4 : Teagan = 2 * Elodie)
  (h5 : Teagan = Kenia - 5) : 
  Kenia + Hunter + Elodie + Teagan = 260 :=
by 
  sorry

end total_number_of_rats_l141_141575


namespace find_p_q_of_divisibility_l141_141341

theorem find_p_q_of_divisibility 
  (p q : ℤ) 
  (h1 : (x + 3) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 6)) 
  (h2 : (x - 2) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 6)) 
  : p = -31 ∧ q = -71 :=
by
  sorry

end find_p_q_of_divisibility_l141_141341


namespace divisibility_of_special_number_l141_141817

theorem divisibility_of_special_number (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
    ∃ d : ℕ, 100100 * a + 10010 * b + 1001 * c = 11 * d := 
sorry

end divisibility_of_special_number_l141_141817


namespace sum_first_5n_l141_141441

theorem sum_first_5n (n : ℕ) (h : (3 * n * (3 * n + 1)) / 2 = (n * (n + 1)) / 2 + 210) : 
  (5 * n * (5 * n + 1)) / 2 = 630 :=
sorry

end sum_first_5n_l141_141441


namespace count_integers_with_sum_of_digits_18_l141_141571

def sum_of_digits (n : ℕ) : ℕ := (n / 100) + (n / 10 % 10) + (n % 10)

def valid_integer_count : ℕ :=
  let range := List.range' 700 (900 - 700 + 1)
  List.length $ List.filter (λ n => sum_of_digits n = 18) range

theorem count_integers_with_sum_of_digits_18 :
  valid_integer_count = 17 :=
sorry

end count_integers_with_sum_of_digits_18_l141_141571


namespace stepa_multiplied_numbers_l141_141582

theorem stepa_multiplied_numbers (x : ℤ) (hx : (81 * x) % 16 = 0) :
  ∃ (a b : ℕ), a * b = 54 ∧ a < 10 ∧ b < 10 :=
by {
  sorry
}

end stepa_multiplied_numbers_l141_141582


namespace seq_10_is_4_l141_141276

-- Define the sequence with given properties
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | 1 => 4
  | (n + 2) => if n % 2 = 0 then 4 else 3

-- Theorem statement: The 10th term of the sequence is 4
theorem seq_10_is_4 : seq 9 = 4 :=
by sorry

end seq_10_is_4_l141_141276


namespace distance_P_to_y_axis_l141_141610

-- Definition: Given point P in Cartesian coordinates
def P : ℝ × ℝ := (-3, -4)

-- Definition: Function to calculate distance to y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ := abs p.1

-- Theorem: The distance from point P to the y-axis is 3
theorem distance_P_to_y_axis : distance_to_y_axis P = 3 :=
by
  sorry

end distance_P_to_y_axis_l141_141610


namespace sqrt_expression_is_869_l141_141082

theorem sqrt_expression_is_869 :
  (31 * 30 * 29 * 28 + 1) = 869 := 
sorry

end sqrt_expression_is_869_l141_141082


namespace minimize_y_at_x_l141_141003

noncomputable def minimize_y (a b x : ℝ) : ℝ :=
  (x - a)^2 + (x - b)^2 + 2 * (a - b) * x

theorem minimize_y_at_x (a b : ℝ) :
  ∃ x : ℝ, minimize_y a b x = minimize_y a b (b / 2) := by
  sorry

end minimize_y_at_x_l141_141003


namespace total_food_per_day_l141_141565

def num_dogs : ℝ := 2
def food_per_dog_per_day : ℝ := 0.12

theorem total_food_per_day : (num_dogs * food_per_dog_per_day) = 0.24 :=
by sorry

end total_food_per_day_l141_141565


namespace original_amount_spent_l141_141099

noncomputable def price_per_mango : ℝ := 383.33 / 115
noncomputable def new_price_per_mango : ℝ := 0.9 * price_per_mango

theorem original_amount_spent (N : ℝ) (H1 : (N + 12) * new_price_per_mango = N * price_per_mango) : 
  N * price_per_mango = 359.64 :=
by 
  sorry

end original_amount_spent_l141_141099


namespace calculate_length_of_floor_l141_141838

-- Define the conditions and the objective to prove
variable (breadth length : ℝ)
variable (cost rate : ℝ)
variable (area : ℝ)

-- Given conditions
def length_more_by_percentage : Prop := length = 2 * breadth
def painting_cost : Prop := cost = 529 ∧ rate = 3

-- Objective
def length_of_floor : ℝ := 2 * breadth

theorem calculate_length_of_floor : 
  (length_more_by_percentage breadth length) →
  (painting_cost cost rate) →
  length_of_floor breadth = 18.78 :=
by
  sorry

end calculate_length_of_floor_l141_141838


namespace regular_tetrahedron_properties_l141_141850

-- Definitions
def equilateral (T : Type) : Prop := sorry -- equilateral triangle property
def equal_sides (T : Type) : Prop := sorry -- all sides equal property
def equal_angles (T : Type) : Prop := sorry -- all angles equal property

def regular (H : Type) : Prop := sorry -- regular tetrahedron property
def equal_edges (H : Type) : Prop := sorry -- all edges are equal
def equal_edge_angles (H : Type) : Prop := sorry -- angles between two edges at the same vertex are equal
def congruent_equilateral_faces (H : Type) : Prop := sorry -- faces are congruent equilateral triangles
def equal_dihedral_angles (H : Type) : Prop := sorry -- dihedral angles between adjacent faces are equal

-- Theorem statement
theorem regular_tetrahedron_properties :
  ∀ (T H : Type), 
    (equilateral T → equal_sides T ∧ equal_angles T) →
    (regular H → 
      (equal_edges H ∧ equal_edge_angles H) ∧
      (congruent_equilateral_faces H ∧ equal_dihedral_angles H) ∧
      (congruent_equilateral_faces H ∧ equal_edge_angles H)) :=
by
  intros T H hT hH
  sorry

end regular_tetrahedron_properties_l141_141850


namespace poly_div_simplification_l141_141360

-- Assume a and b are real numbers.
variables (a b : ℝ)

-- Theorem to prove the equivalence
theorem poly_div_simplification (a b : ℝ) : (4 * a^2 - b^2) / (b - 2 * a) = -2 * a - b :=
by
  -- The proof will go here
  sorry

end poly_div_simplification_l141_141360


namespace proof_expression_value_l141_141695

noncomputable def a : ℝ := 0.15
noncomputable def b : ℝ := 0.06
noncomputable def x : ℝ := a^3
noncomputable def y : ℝ := b^3
noncomputable def z : ℝ := a^2
noncomputable def w : ℝ := b^2

theorem proof_expression_value :
  ( (x - y) / (z + w) ) + 0.009 + w^4 = 0.1300341679616 := sorry

end proof_expression_value_l141_141695


namespace contributions_before_john_l141_141102

theorem contributions_before_john (n : ℕ) (A : ℚ) 
  (h1 : 1.5 * A = 75)
  (h2 : (n * A + 225) / (n + 1) = 75) : n = 6 :=
by {
  sorry
}

end contributions_before_john_l141_141102


namespace julian_comic_pages_l141_141097

-- Definitions from conditions
def frames_per_page : ℝ := 143.0
def total_frames : ℝ := 1573.0

-- The theorem stating the proof problem
theorem julian_comic_pages : total_frames / frames_per_page = 11 :=
by
  sorry

end julian_comic_pages_l141_141097


namespace diamondsuit_result_l141_141710

def diam (a b : ℕ) : ℕ := a

theorem diamondsuit_result : (diam 7 (diam 4 8)) = 7 :=
by sorry

end diamondsuit_result_l141_141710


namespace circle_trajectory_l141_141384

theorem circle_trajectory (a b : ℝ) :
  ∃ x y : ℝ, (b - 3)^2 + a^2 = (b + 3)^2 → x^2 = 12 * y := 
sorry

end circle_trajectory_l141_141384


namespace percent_problem_l141_141070

variable (x : ℝ)

theorem percent_problem (h : 0.30 * 0.15 * x = 27) : 0.15 * 0.30 * x = 27 :=
by sorry

end percent_problem_l141_141070


namespace kittens_per_bunny_l141_141873

-- Conditions
def total_initial_bunnies : ℕ := 30
def fraction_given_to_friend : ℚ := 2 / 5
def total_bunnies_after_birth : ℕ := 54

-- Determine the number of kittens each bunny gave birth to
theorem kittens_per_bunny (initial_bunnies given_fraction total_bunnies_after : ℕ) 
  (h1 : initial_bunnies = total_initial_bunnies)
  (h2 : given_fraction = fraction_given_to_friend)
  (h3 : total_bunnies_after = total_bunnies_after_birth) :
  (total_bunnies_after - (total_initial_bunnies - (total_initial_bunnies * fraction_given_to_friend))) / 
    (total_initial_bunnies * (1 - fraction_given_to_friend)) = 2 :=
by
  sorry

end kittens_per_bunny_l141_141873


namespace compound_interest_amount_l141_141759

/-
Given:
- Principal amount P = 5000
- Annual interest rate r = 0.07
- Time period t = 15 years

We aim to prove:
A = 5000 * (1 + 0.07) ^ 15 = 13795.15
-/
theorem compound_interest_amount :
  let P : ℝ := 5000
  let r : ℝ := 0.07
  let t : ℝ := 15
  let A : ℝ := P * (1 + r) ^ t
  A = 13795.15 :=
by
  sorry

end compound_interest_amount_l141_141759


namespace candy_bars_to_buy_l141_141288

variable (x : ℕ)

theorem candy_bars_to_buy (h1 : 25 * x + 2 * 75 + 50 = 11 * 25) : x = 3 :=
by
  sorry

end candy_bars_to_buy_l141_141288


namespace harmonic_mean_closest_integer_l141_141766

theorem harmonic_mean_closest_integer (a b : ℝ) (ha : a = 1) (hb : b = 2016) :
  abs ((2 * a * b) / (a + b) - 2) < 1 :=
by
  sorry

end harmonic_mean_closest_integer_l141_141766


namespace park_bench_problem_l141_141634

/-- A single bench section at a park can hold either 8 adults or 12 children.
When N bench sections are connected end to end, an equal number of adults and 
children seated together will occupy all the bench space.
This theorem states that the smallest positive integer N such that this condition 
is satisfied is 3. -/
theorem park_bench_problem : ∃ N : ℕ, N > 0 ∧ (8 * N = 12 * N) ∧ N = 3 :=
by
  sorry

end park_bench_problem_l141_141634


namespace second_part_lent_years_l141_141367

theorem second_part_lent_years 
  (P1 P2 T : ℝ)
  (h1 : P1 + P2 = 2743)
  (h2 : P2 = 1688)
  (h3 : P1 * 0.03 * 8 = P2 * 0.05 * T) 
  : T = 3 :=
sorry

end second_part_lent_years_l141_141367


namespace basket_ratio_l141_141557

variable (S A H : ℕ)

theorem basket_ratio 
  (alex_baskets : A = 8) 
  (hector_baskets : H = 2 * S) 
  (total_baskets : A + S + H = 80) : 
  (S : ℚ) / (A : ℚ) = 3 := 
by 
  sorry

end basket_ratio_l141_141557


namespace perpendicular_lines_m_value_l141_141410

def is_perpendicular (m : ℝ) : Prop :=
    let slope1 := 1 / 2
    let slope2 := -2 / m
    slope1 * slope2 = -1

theorem perpendicular_lines_m_value (m : ℝ) (h : is_perpendicular m) : m = 1 := by
    sorry

end perpendicular_lines_m_value_l141_141410


namespace not_possible_155_cents_five_coins_l141_141746

/-- It is not possible to achieve a total value of 155 cents using exactly five coins 
    from a piggy bank containing only pennies (1 cent), nickels (5 cents), 
    quarters (25 cents), and half-dollars (50 cents). -/
theorem not_possible_155_cents_five_coins (n_pennies n_nickels n_quarters n_half_dollars : ℕ) 
    (h : n_pennies + n_nickels + n_quarters + n_half_dollars = 5) : 
    n_pennies * 1 + n_nickels * 5 + n_quarters * 25 + n_half_dollars * 50 ≠ 155 := 
sorry

end not_possible_155_cents_five_coins_l141_141746


namespace slope_of_line_l141_141760

theorem slope_of_line {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (h : 5 / x + 4 / y = 0) :
  ∃ x₁ x₂ y₁ y₂, (5 / x₁ + 4 / y₁ = 0) ∧ (5 / x₂ + 4 / y₂ = 0) ∧ 
  (y₂ - y₁) / (x₂ - x₁) = -4 / 5 :=
sorry

end slope_of_line_l141_141760


namespace opposite_of_negative_six_is_six_l141_141662

-- Define what it means for one number to be the opposite of another.
def is_opposite (a b : Int) : Prop :=
  a = -b

-- The statement to be proved: the opposite number of -6 is 6.
theorem opposite_of_negative_six_is_six : is_opposite (-6) 6 :=
  by sorry

end opposite_of_negative_six_is_six_l141_141662


namespace fraction_eval_l141_141449

theorem fraction_eval : 
  ((12^4 + 324) * (24^4 + 324) * (36^4 + 324)) / 
  ((6^4 + 324) * (18^4 + 324) * (30^4 + 324)) = (84 / 35) :=
by
  sorry

end fraction_eval_l141_141449


namespace kristy_initial_cookies_l141_141522

-- Define the initial conditions
def initial_cookies (total_cookies_left : Nat) (c1 c2 c3 : Nat) (c4 c5 c6 : Nat) : Nat :=
  total_cookies_left + c1 + c2 + c3 + c4 + c5 + c6

-- Now we can state the theorem
theorem kristy_initial_cookies :
  initial_cookies 6 5 5 3 1 2 = 22 :=
by
  -- Proof is omitted
  sorry

end kristy_initial_cookies_l141_141522


namespace isosceles_triangle_perimeter_l141_141957

theorem isosceles_triangle_perimeter 
  (a b : ℕ) 
  (h_iso : a = b ∨ a = 3 ∨ b = 3) 
  (h_sides : a = 6 ∨ b = 6) 
  : a + b + 3 = 15 := by
  sorry

end isosceles_triangle_perimeter_l141_141957


namespace solve_for_xy_l141_141775

theorem solve_for_xy (x y : ℝ) (h1 : 3 * x ^ 2 - 9 * y ^ 2 = 0) (h2 : x + y = 5) :
    (x = (15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 * Real.sqrt 3 - 5) / 2) ∨
    (x = (-15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 + 5 * Real.sqrt 3) / 2) :=
by
  sorry

end solve_for_xy_l141_141775


namespace batsman_average_after_12th_innings_l141_141881

noncomputable def batsman_average (runs_in_12th_innings : ℕ) (average_increase : ℕ) (initial_average_after_11_innings : ℕ) : ℕ :=
initial_average_after_11_innings + average_increase

theorem batsman_average_after_12th_innings
(score_in_12th_innings : ℕ)
(average_increase : ℕ)
(initial_average_after_11_innings : ℕ)
(total_runs_after_11_innings := 11 * initial_average_after_11_innings)
(total_runs_after_12_innings := total_runs_after_11_innings + score_in_12th_innings)
(new_average_after_12_innings := total_runs_after_12_innings / 12)
:
score_in_12th_innings = 80 ∧ average_increase = 3 ∧ initial_average_after_11_innings = 44 → 
batsman_average score_in_12th_innings average_increase initial_average_after_11_innings = 47 := 
by
  -- skipping the actual proof for now
  sorry

end batsman_average_after_12th_innings_l141_141881


namespace coin_selection_probability_l141_141202

noncomputable def probability_at_least_50_cents : ℚ := 
  let total_ways := Nat.choose 12 6 -- total ways to choose 6 coins out of 12
  let case1 := 1 -- 6 dimes
  let case2 := (Nat.choose 6 5) * (Nat.choose 4 1) -- 5 dimes and 1 nickel
  let case3 := (Nat.choose 6 4) * (Nat.choose 4 2) -- 4 dimes and 2 nickels
  let successful_ways := case1 + case2 + case3 -- total successful outcomes
  successful_ways / total_ways

theorem coin_selection_probability : 
  probability_at_least_50_cents = 127 / 924 := by 
  sorry

end coin_selection_probability_l141_141202


namespace cad_to_jpy_l141_141987

theorem cad_to_jpy (h : 2000 / 18 =  y / 5) : y = 556 := 
by 
  sorry

end cad_to_jpy_l141_141987


namespace total_wages_l141_141596

-- Definitions and conditions
def A_one_day_work : ℚ := 1 / 10
def B_one_day_work : ℚ := 1 / 15
def A_share_wages : ℚ := 2040

-- Stating the problem in Lean
theorem total_wages (X : ℚ) : (3 / 5) * X = A_share_wages → X = 3400 := 
  by 
  sorry

end total_wages_l141_141596


namespace aria_cookies_per_day_l141_141234

theorem aria_cookies_per_day 
  (cost_per_cookie : ℕ)
  (total_amount_spent : ℕ)
  (days_in_march : ℕ)
  (h_cost : cost_per_cookie = 19)
  (h_spent : total_amount_spent = 2356)
  (h_days : days_in_march = 31) : 
  (total_amount_spent / cost_per_cookie) / days_in_march = 4 :=
by
  sorry

end aria_cookies_per_day_l141_141234


namespace min_ratio_number_l141_141630

theorem min_ratio_number (H T U : ℕ) (h1 : H - T = 8 ∨ T - H = 8) (hH : 1 ≤ H ∧ H ≤ 9) (hT : 0 ≤ T ∧ T ≤ 9) (hU : 0 ≤ U ∧ U ≤ 9) :
  100 * H + 10 * T + U = 190 :=
by sorry

end min_ratio_number_l141_141630


namespace total_cost_is_eight_times_l141_141119

theorem total_cost_is_eight_times (cost_shorts cost_shirt cost_boots cost_shin_guards : ℕ) :
  (cost_shorts + cost_shirt = 2 * cost_shorts) →
  (cost_shorts + cost_boots = 5 * cost_shorts) →
  (cost_shorts + cost_shin_guards = 3 * cost_shorts) →
  cost_shorts + cost_shirt + cost_boots + cost_shin_guards = 8 * cost_shorts := 
by 
  intros h_shirt h_boots h_shin_guards 
  sorry

end total_cost_is_eight_times_l141_141119


namespace volume_of_regular_quadrilateral_pyramid_l141_141541

noncomputable def volume_of_pyramid (a : ℝ) : ℝ :=
  let x := 1 -- A placeholder to outline the structure
  let PM := (6 * a) / 5
  let V := (2 * a^3) / 5
  V

theorem volume_of_regular_quadrilateral_pyramid
  (a PM : ℝ)
  (h1 : PM = (6 * a) / 5)
  [InstReal : Nonempty (Real)] :
  volume_of_pyramid a = (2 * a^3) / 5 :=
by
  sorry

end volume_of_regular_quadrilateral_pyramid_l141_141541


namespace focus_of_parabola_l141_141914

theorem focus_of_parabola (x y : ℝ) : (y^2 + 4 * x = 0) → (x = -1 ∧ y = 0) :=
by sorry

end focus_of_parabola_l141_141914


namespace pure_imaginary_sol_l141_141260

theorem pure_imaginary_sol (m : ℝ) (h : (m^2 - m - 2) = 0 ∧ (m + 1) ≠ 0) : m = 2 :=
sorry

end pure_imaginary_sol_l141_141260


namespace find_a4_l141_141397

variable (a : ℕ → ℤ)

def S (n : ℕ) : ℤ := (n * (a 1 + a n)) / 2

theorem find_a4 (h₁ : S 5 = 25) (h₂ : a 2 = 3) : a 4 = 7 := by
  sorry

end find_a4_l141_141397


namespace parameterized_line_solution_l141_141469

theorem parameterized_line_solution :
  ∃ (s l : ℚ), 
  (∀ t : ℚ, 
    ∃ x y : ℚ, 
      x = -3 + t * l ∧ 
      y = s + t * (-7) ∧ 
      y = 3 * x + 2
  ) ∧
  s = -7 ∧ l = -7 / 3 := 
sorry

end parameterized_line_solution_l141_141469


namespace doll_cost_l141_141409

theorem doll_cost (D : ℝ) (h : 4 * D = 60) : D = 15 :=
by {
  sorry
}

end doll_cost_l141_141409


namespace diophantine_eq_solutions_l141_141619

theorem diophantine_eq_solutions (p q r k : ℕ) (hp : p > 1) (hq : q > 1) (hr : r > 1) 
  (hp_prime : Prime p) (hq_prime : Prime q) (hr_prime : Prime r) (hk : k > 0) :
  p^2 + q^2 + 49*r^2 = 9*k^2 - 101 ↔ 
  (p = 3 ∧ q = 5 ∧ r = 3 ∧ k = 8) ∨ (p = 5 ∧ q = 3 ∧ r = 3 ∧ k = 8) :=
by sorry

end diophantine_eq_solutions_l141_141619


namespace donna_pizza_slices_l141_141611

theorem donna_pizza_slices :
  ∀ (total_slices : ℕ) (half_eaten_for_lunch : ℕ) (one_third_eaten_for_dinner : ℕ),
  total_slices = 12 →
  half_eaten_for_lunch = total_slices / 2 →
  one_third_eaten_for_dinner = half_eaten_for_lunch / 3 →
  (half_eaten_for_lunch - one_third_eaten_for_dinner) = 4 :=
by
  intros total_slices half_eaten_for_lunch one_third_eaten_for_dinner
  intros h1 h2 h3
  sorry

end donna_pizza_slices_l141_141611


namespace tan_alpha_value_l141_141451

variable (α : Real)
variable (h1 : Real.sin α = 4/5)
variable (h2 : π/2 < α ∧ α < π)

theorem tan_alpha_value : Real.tan α = -4/3 := by
  sorry

end tan_alpha_value_l141_141451


namespace marbles_count_l141_141590

theorem marbles_count (red green blue total : ℕ) (h_red : red = 38)
  (h_green : green = red / 2) (h_total : total = 63) 
  (h_sum : total = red + green + blue) : blue = 6 :=
by
  sorry

end marbles_count_l141_141590


namespace paint_price_and_max_boxes_l141_141116

theorem paint_price_and_max_boxes (x y m : ℕ) 
  (h1 : x + 2 * y = 56) 
  (h2 : 2 * x + y = 64) 
  (h3 : 24 * m + 16 * (200 - m) ≤ 3920) : 
  x = 24 ∧ y = 16 ∧ m ≤ 90 := 
  by 
    sorry

end paint_price_and_max_boxes_l141_141116


namespace find_length_PB_l141_141839

noncomputable def radius (O : Type*) : ℝ := sorry

structure Circle (α : Type*) :=
(center : α)
(radius : ℝ)

variables {α : Type*}

def Point (α : Type*) := α

variables (P T A B : Point ℝ) (O : Circle ℝ) (r : ℝ)

def PA := (4 : ℝ)
def PT (AB : ℝ) := AB - 2
def PB (AB : ℝ) := 4 + AB

def power_of_a_point (PA PB PT : ℝ) := PA * PB = PT^2

theorem find_length_PB (AB : ℝ) 
  (h1 : power_of_a_point PA (PB AB) (PT AB)) 
  (h2 : PA < PB AB) : 
  PB AB = 18 := 
by 
  sorry

end find_length_PB_l141_141839


namespace equal_distribution_l141_141891

variables (Emani Howard : ℕ)

-- Emani has $30 more than Howard
axiom emani_condition : Emani = Howard + 30

-- Emani has $150
axiom emani_has_money : Emani = 150

theorem equal_distribution : (Emani + Howard) / 2 = 135 :=
by
  sorry

end equal_distribution_l141_141891


namespace problem_statement_l141_141342

-- Define the arithmetic sequence conditions
variables (a : ℕ → ℕ) (d : ℕ)
axiom h1 : a 1 = 2
axiom h2 : a 2018 = 2019
axiom arithmetic_seq : ∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ := (n * a 1) + (n * (n-1) * d / 2)

theorem problem_statement : sum_seq a 5 + a 2014 = 2035 :=
by sorry

end problem_statement_l141_141342


namespace solve_for_x_l141_141110

theorem solve_for_x (x : ℤ) (h : 3 * x = 2 * x + 6) : x = 6 := by
  sorry

end solve_for_x_l141_141110


namespace processing_plant_growth_eq_l141_141540

-- Definition of the conditions given in the problem
def initial_amount : ℝ := 10
def november_amount : ℝ := 13
def growth_rate (x : ℝ) : ℝ := initial_amount * (1 + x)^2

-- Lean theorem statement to prove the equation
theorem processing_plant_growth_eq (x : ℝ) : 
  growth_rate x = november_amount ↔ initial_amount * (1 + x)^2 = 13 := 
by
  sorry

end processing_plant_growth_eq_l141_141540


namespace boat_speed_upstream_l141_141457

noncomputable def V_b : ℝ := 11
noncomputable def V_down : ℝ := 15
noncomputable def V_s : ℝ := V_down - V_b
noncomputable def V_up : ℝ := V_b - V_s

theorem boat_speed_upstream :
  V_up = 7 := by
  sorry

end boat_speed_upstream_l141_141457


namespace trajectory_eq_ellipse_range_sum_inv_dist_l141_141989

-- Conditions for circle M
def CircleM := { center : ℝ × ℝ // center = (-3, 0) }
def radiusM := 1

-- Conditions for circle N
def CircleN := { center : ℝ × ℝ // center = (3, 0) }
def radiusN := 9

-- Conditions for circle P
def CircleP (x y : ℝ) (r : ℝ) := 
  (dist (x, y) (-3, 0) = r + radiusM) ∧
  (dist (x, y) (3, 0) = radiusN - r)

-- Proof for the equation of the trajectory
theorem trajectory_eq_ellipse :
  ∃ (x y : ℝ), CircleP x y r → x^2 / 25 + y^2 / 16 = 1 :=
sorry

-- Proof for the range of 1/PM + 1/PN
theorem range_sum_inv_dist :
  ∃ (r PM PN : ℝ), 
    PM ∈ [2, 8] ∧ 
    PN = 10 - PM ∧ 
    CircleP (PM - radiusM) (PN - radiusN) r → 
    (2/5 ≤ (1/PM + 1/PN) ∧ (1/PM + 1/PN) ≤ 5/8) :=
sorry

end trajectory_eq_ellipse_range_sum_inv_dist_l141_141989


namespace range_of_smallest_nonprime_with_condition_l141_141480

def smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 : ℕ :=
121

theorem range_of_smallest_nonprime_with_condition :
  120 < smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 ∧ 
  smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 ≤ 130 :=
by
  unfold smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10
  exact ⟨by norm_num, by norm_num⟩

end range_of_smallest_nonprime_with_condition_l141_141480


namespace P_iff_nonQ_l141_141666

-- Given conditions
def P (x y : ℝ) : Prop := x^2 + y^2 = 0
def Q (x y : ℝ) : Prop := x ≠ 0 ∨ y ≠ 0
def nonQ (x y : ℝ) : Prop := x = 0 ∧ y = 0

-- Main statement
theorem P_iff_nonQ (x y : ℝ) : P x y ↔ nonQ x y :=
sorry

end P_iff_nonQ_l141_141666


namespace value_of_expression_l141_141401

theorem value_of_expression :
  (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2000 :=
by {
  sorry
}

end value_of_expression_l141_141401


namespace students_no_A_l141_141626

theorem students_no_A (T AH AM AHAM : ℕ) (h1 : T = 35) (h2 : AH = 10) (h3 : AM = 15) (h4 : AHAM = 5) :
  T - (AH + AM - AHAM) = 15 :=
by
  sorry

end students_no_A_l141_141626


namespace min_days_to_find_poisoned_apple_l141_141054

theorem min_days_to_find_poisoned_apple (n : ℕ) (n_pos : 0 < n) : 
  ∀ k : ℕ, 2^k ≥ 2021 → k ≥ 11 :=
  sorry

end min_days_to_find_poisoned_apple_l141_141054


namespace number_of_people_l141_141242

theorem number_of_people (x : ℕ) (h1 : 175 = 175) (h2: 2 = 2) (h3 : ∀ (p : ℕ), p * x = 175 + p * 10) : x = 7 :=
sorry

end number_of_people_l141_141242


namespace sphere_surface_area_l141_141186

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : ∀ r, V = (4/3) * π * r^3)
  (h2 : V = 72 * π) : A = 36 * π * 2^(2/3) :=
by 
  sorry

end sphere_surface_area_l141_141186


namespace relation_y1_y2_y3_l141_141988

-- Definition of being on the parabola
def on_parabola (x : ℝ) (y m : ℝ) : Prop := y = -3*x^2 - 12*x + m

-- The conditions given in the problem
variables {y1 y2 y3 m : ℝ}

-- The points (-3, y1), (-2, y2), (1, y3) are on the parabola given by the equation
axiom h1 : on_parabola (-3) y1 m
axiom h2 : on_parabola (-2) y2 m
axiom h3 : on_parabola (1) y3 m

-- We need to prove the relationship between y1, y2, and y3
theorem relation_y1_y2_y3 : y2 > y1 ∧ y1 > y3 :=
by { sorry }

end relation_y1_y2_y3_l141_141988


namespace one_plus_i_squared_eq_two_i_l141_141246

theorem one_plus_i_squared_eq_two_i (i : ℂ) (h : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end one_plus_i_squared_eq_two_i_l141_141246


namespace sum_of_g_35_l141_141039

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 - 3
noncomputable def g (y : ℝ) : ℝ := y^2 + y + 1

theorem sum_of_g_35 : g 35 = 21 := 
by
  sorry

end sum_of_g_35_l141_141039


namespace B_and_C_have_together_l141_141675

theorem B_and_C_have_together
  (A B C : ℕ)
  (h1 : A + B + C = 700)
  (h2 : A + C = 300)
  (h3 : C = 200) :
  B + C = 600 := by
  sorry

end B_and_C_have_together_l141_141675


namespace line_circle_no_intersection_l141_141378

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → (x - 1)^2 + (y + 1)^2 ≠ 1) :=
by
  sorry

end line_circle_no_intersection_l141_141378


namespace ladybugs_total_total_ladybugs_is_5_l141_141087

def num_ladybugs (x y : ℕ) : ℕ :=
  x + y

theorem ladybugs_total (x y n : ℕ) 
    (h_spot_calc_1: 6 * x + 4 * y = 30 ∨ 6 * x + 4 * y = 26)
    (h_total_spots_30: (6 * x + 4 * y = 30) ↔ 3 * x + 2 * y = 15)
    (h_total_spots_26: (6 * x + 4 * y = 26) ↔ 3 * x + 2 * y = 13)
    (h_truth_only_one: 
       (6 * x + 4 * y = 30 ∧ ¬(6 * x + 4 * y = 26)) ∨
       (¬(6 * x + 4 * y = 30) ∧ 6 * x + 4 * y = 26))
    : n = x + y :=
by 
  sorry

theorem total_ladybugs_is_5 : ∃ x y : ℕ, num_ladybugs x y = 5 :=
  ⟨3, 2, rfl⟩

end ladybugs_total_total_ladybugs_is_5_l141_141087


namespace scientific_notation_of_4600000000_l141_141963

theorem scientific_notation_of_4600000000 :
  4.6 * 10^9 = 4600000000 := 
by
  sorry

end scientific_notation_of_4600000000_l141_141963


namespace smallest_k_for_abk_l141_141646

theorem smallest_k_for_abk : ∃ (k : ℝ), (∀ (a b : ℝ), a + b = k ∧ ab = k → k = 4) :=
sorry

end smallest_k_for_abk_l141_141646


namespace midpoint_coordinates_l141_141960

theorem midpoint_coordinates (xM yM xN yN : ℝ) (hM : xM = 3) (hM' : yM = -2) (hN : xN = -1) (hN' : yN = 0) :
  (xM + xN) / 2 = 1 ∧ (yM + yN) / 2 = -1 :=
by
  simp [hM, hM', hN, hN']
  sorry

end midpoint_coordinates_l141_141960


namespace number_of_round_table_arrangements_l141_141019

theorem number_of_round_table_arrangements : (Nat.factorial 5) / 5 = 24 := 
by
  sorry

end number_of_round_table_arrangements_l141_141019


namespace bisection_min_calculations_l141_141315

theorem bisection_min_calculations 
  (a b : ℝ)
  (h_interval : a = 1.4 ∧ b = 1.5)
  (delta : ℝ)
  (h_delta : delta = 0.001) :
  ∃ n : ℕ, 0.1 / (2 ^ n) ≤ delta ∧ n = 7 :=
sorry

end bisection_min_calculations_l141_141315


namespace solve_for_x_l141_141757

theorem solve_for_x (x : ℝ) (h : 5 + 7 / x = 6 - 5 / x) : x = 12 := 
sorry

end solve_for_x_l141_141757


namespace probability_of_two_red_balls_l141_141797

-- Define the total number of balls, number of red balls, and number of white balls
def total_balls := 6
def red_balls := 4
def white_balls := 2
def drawn_balls := 2

-- Define the combination formula
def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The number of ways to choose 2 red balls from 4
def ways_to_choose_red := choose 4 2

-- The number of ways to choose any 2 balls from the total of 6
def ways_to_choose_any := choose 6 2

-- The corresponding probability
def probability := ways_to_choose_red / ways_to_choose_any

-- The theorem we want to prove
theorem probability_of_two_red_balls :
  probability = 2 / 5 :=
by
  sorry

end probability_of_two_red_balls_l141_141797


namespace calc_angle_CAB_l141_141501

theorem calc_angle_CAB (α β γ ε : ℝ) (hα : α = 79) (hβ : β = 63) (hγ : γ = 131) (hε : ε = 123.5) : 
  ∃ φ : ℝ, φ = 24 + 52 / 60 :=
by
  sorry

end calc_angle_CAB_l141_141501


namespace triangle_inequality_l141_141363

theorem triangle_inequality
  (a b c : ℝ)
  (habc : ¬(a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a)) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 + 4 * a * b * c > a^3 + b^3 + c^3 := 
by {
  sorry
}

end triangle_inequality_l141_141363


namespace determine_m_l141_141129

-- Define a complex number structure in Lean
structure ComplexNumber where
  re : ℝ  -- real part
  im : ℝ  -- imaginary part

-- Define the condition where the complex number is purely imaginary
def is_purely_imaginary (z : ComplexNumber) : Prop :=
  z.re = 0

-- State the Lean theorem
theorem determine_m (m : ℝ) (h : is_purely_imaginary (ComplexNumber.mk (m^2 - m) m)) : m = 1 :=
by
  sorry

end determine_m_l141_141129


namespace sequence_m_l141_141707

noncomputable def a (n : ℕ) : ℕ :=
  if n = 0 then 0  -- We usually start sequences from n = 1; hence, a_0 is irrelevant
  else (n * n) - n + 1

theorem sequence_m (m : ℕ) (h_positive : m > 0) (h_bound : 43 < a m ∧ a m < 73) : m = 8 :=
by {
  sorry
}

end sequence_m_l141_141707


namespace frustum_midsection_area_relation_l141_141814

theorem frustum_midsection_area_relation 
  (S₁ S₂ S₀ : ℝ) 
  (h₁: 0 ≤ S₁ ∧ 0 ≤ S₂ ∧ 0 ≤ S₀)
  (h₂: ∃ a h, (a / (a + 2 * h))^2 = S₂ / S₁ ∧ (a / (a + h))^2 = S₂ / S₀) :
  2 * Real.sqrt S₀ = Real.sqrt S₁ + Real.sqrt S₂ := 
sorry

end frustum_midsection_area_relation_l141_141814


namespace optimal_selling_price_maximizes_profit_l141_141060

/-- The purchase price of a certain product is 40 yuan. -/
def cost_price : ℝ := 40

/-- At a selling price of 50 yuan, 50 units can be sold. -/
def initial_selling_price : ℝ := 50
def initial_quantity_sold : ℝ := 50

/-- If the selling price increases by 1 yuan, the sales volume decreases by 1 unit. -/
def price_increase_effect (x : ℝ) : ℝ := initial_selling_price + x
def quantity_decrease_effect (x : ℝ) : ℝ := initial_quantity_sold - x

/-- The revenue function. -/
def revenue (x : ℝ) : ℝ := (price_increase_effect x) * (quantity_decrease_effect x)

/-- The cost function. -/
def cost (x : ℝ) : ℝ := cost_price * (quantity_decrease_effect x)

/-- The profit function. -/
def profit (x : ℝ) : ℝ := revenue x - cost x

/-- The proof that the optimal selling price to maximize profit is 70 yuan. -/
theorem optimal_selling_price_maximizes_profit : price_increase_effect 20 = 70 :=
by
  sorry

end optimal_selling_price_maximizes_profit_l141_141060


namespace triangle_cannot_have_two_right_angles_l141_141214

theorem triangle_cannot_have_two_right_angles (A B C : ℝ) (h : A + B + C = 180) : 
  ¬ (A = 90 ∧ B = 90) :=
by {
  sorry
}

end triangle_cannot_have_two_right_angles_l141_141214


namespace multiple_of_larger_number_l141_141229

variables (S L M : ℝ)

-- Conditions
def small_num := S = 10.0
def sum_eq := S + L = 24
def multiplication_relation := 7 * S = M * L

-- Theorem statement
theorem multiple_of_larger_number (S L M : ℝ) 
  (h1 : small_num S) 
  (h2 : sum_eq S L) 
  (h3 : multiplication_relation S L M) : 
  M = 5 := by
  sorry

end multiple_of_larger_number_l141_141229


namespace find_fraction_l141_141305

variable (x : ℝ) (f : ℝ)
axiom thirty_percent_of_x : 0.30 * x = 63.0000000000001
axiom fraction_condition : f = 0.40 * x + 12

theorem find_fraction : f = 96 := by
  sorry

end find_fraction_l141_141305


namespace sequence_finite_l141_141042

def sequence_terminates (a_0 : ℕ) : Prop :=
  ∀ (a : ℕ → ℕ), (a 0 = a_0) ∧ 
                  (∀ n, ((a n > 5) ∧ (a n % 10 ≤ 5) → a (n + 1) = a n / 10)) ∧
                  (∀ n, ((a n > 5) ∧ (a n % 10 > 5) → a (n + 1) = 9 * a n)) → 
                  ∃ n, a n ≤ 5 

theorem sequence_finite (a_0 : ℕ) : sequence_terminates a_0 :=
sorry

end sequence_finite_l141_141042


namespace find_distinct_numbers_l141_141826

theorem find_distinct_numbers (k l : ℕ) (h : 64 / k = 4 * (64 / l)) : k = 1 ∧ l = 4 :=
by
  sorry

end find_distinct_numbers_l141_141826


namespace boxes_in_attic_l141_141309

theorem boxes_in_attic (B : ℕ)
  (h1 : 6 ≤ B)
  (h2 : ∀ T : ℕ, T = (B - 6) / 2 ∧ T = 10)
  (h3 : ∀ O : ℕ, O = 180 + 2 * T ∧ O = 20 * T) :
  B = 26 :=
by
  sorry

end boxes_in_attic_l141_141309


namespace solution_set_of_quadratic_inequality_l141_141973

theorem solution_set_of_quadratic_inequality (a b : ℝ)
  (h : ∀ x : ℝ, ax^2 + bx - 2 > 0 ↔ x ∈ Set.Ioo (-4 : ℝ) 1) :
  a + b = 2 := 
sorry

end solution_set_of_quadratic_inequality_l141_141973


namespace contradiction_problem_l141_141164

theorem contradiction_problem (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) → False := 
by
  sorry

end contradiction_problem_l141_141164


namespace geometric_series_sum_l141_141935

theorem geometric_series_sum :
  let a := 3
  let r := 3
  let n := 9
  let last_term := a * r^(n - 1)
  last_term = 19683 →
  let S := a * (r^n - 1) / (r - 1)
  S = 29523 :=
by
  intros
  sorry

end geometric_series_sum_l141_141935


namespace solve_equation_l141_141338

theorem solve_equation (x : ℝ) : 
  (3 * x + 2) * (x + 3) = x + 3 ↔ (x = -3 ∨ x = -1/3) :=
by sorry

end solve_equation_l141_141338


namespace helen_chocolate_chip_cookies_l141_141374

def number_of_raisin_cookies := 231
def difference := 25

theorem helen_chocolate_chip_cookies :
  ∃ C, C = number_of_raisin_cookies + difference ∧ C = 256 :=
by
  sorry -- Skipping the proof

end helen_chocolate_chip_cookies_l141_141374


namespace number_of_kg_of_mangoes_l141_141899

variable {m : ℕ}
def cost_apples := 8 * 70
def cost_mangoes (m : ℕ) := 75 * m
def total_cost := 1235

theorem number_of_kg_of_mangoes (h : cost_apples + cost_mangoes m = total_cost) : m = 9 :=
by
  sorry

end number_of_kg_of_mangoes_l141_141899


namespace top_face_not_rotated_by_90_l141_141780

-- Define the cube and the conditions of rolling and returning
structure Cube :=
  (initial_top_face_orientation : ℕ) -- an integer representation of the orientation of the top face
  (position : ℤ × ℤ) -- (x, y) coordinates on a 2D plane

def rolls_over_edges (c : Cube) : Cube :=
  sorry -- placeholder for the actual rolling operation

def returns_to_original_position (c : Cube) (original : Cube) : Prop :=
  c.position = original.position ∧ c.initial_top_face_orientation = original.initial_top_face_orientation

-- The main theorem to prove
theorem top_face_not_rotated_by_90 {c : Cube} (original : Cube) :
  returns_to_original_position c original → c.initial_top_face_orientation ≠ (original.initial_top_face_orientation + 1) % 4 :=
sorry

end top_face_not_rotated_by_90_l141_141780


namespace max_sides_three_obtuse_l141_141211

theorem max_sides_three_obtuse (n : ℕ) (convex : Prop) (obtuse_angles : ℕ) :
  (convex = true ∧ obtuse_angles = 3) → n ≤ 6 :=
by
  sorry

end max_sides_three_obtuse_l141_141211


namespace dave_winfield_home_runs_l141_141830

theorem dave_winfield_home_runs : 
  ∃ x : ℕ, 755 = 2 * x - 175 ∧ x = 465 :=
by
  sorry

end dave_winfield_home_runs_l141_141830


namespace north_pond_ducks_l141_141421

-- Definitions based on the conditions
def ducks_lake_michigan : ℕ := 100
def twice_ducks_lake_michigan : ℕ := 2 * ducks_lake_michigan
def additional_ducks : ℕ := 6
def ducks_north_pond : ℕ := twice_ducks_lake_michigan + additional_ducks

-- Theorem to prove the answer
theorem north_pond_ducks : ducks_north_pond = 206 :=
by
  sorry

end north_pond_ducks_l141_141421


namespace correct_number_of_statements_l141_141912

-- Definitions based on the problem's conditions
def condition_1 : Prop :=
  ∀ (n : ℕ) (a b c d e : ℚ), n = 5 ∧ ∃ x y z, (x = a ∧ y = b ∧ z = c) ∧ (x < 0 ∧ y < 0 ∧ z < 0 ∧ d ≥ 0 ∧ e ≥ 0) →
  (a * b * c * d * e < 0 ∨ a * b * c * d * e = 0)

def condition_2 : Prop := 
  ∀ m : ℝ, |m| + m = 0 → m ≤ 0

def condition_3 : Prop := 
  ∀ a b : ℝ, (1 / a < 1 / b) → ¬ (a < b ∨ b < a)

def condition_4 : Prop := 
  ∀ a : ℝ, ∃ max_val, max_val = 5 ∧ 5 - |a - 5| ≤ max_val

-- Main theorem to state the correct number of true statements
theorem correct_number_of_statements : 
  (condition_2 ∧ condition_4) ∧
  ¬condition_1 ∧ 
  ¬condition_3 :=
by
  sorry

end correct_number_of_statements_l141_141912


namespace calculate_allocations_l141_141663

variable (new_revenue : ℝ)
variable (ratio_employee_salaries ratio_stock_purchases ratio_rent ratio_marketing_costs : ℕ)

theorem calculate_allocations :
  let total_ratio := ratio_employee_salaries + ratio_stock_purchases + ratio_rent + ratio_marketing_costs
  let part_value := new_revenue / total_ratio
  let employee_salary_alloc := ratio_employee_salaries * part_value
  let rent_alloc := ratio_rent * part_value
  let marketing_costs_alloc := ratio_marketing_costs * part_value
  employee_salary_alloc + rent_alloc + marketing_costs_alloc = 7800 :=
by
  sorry

end calculate_allocations_l141_141663


namespace solve_equation_l141_141200

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (x / (x - 2) - 3 / x = 1) → x = 6 :=
by
  sorry

end solve_equation_l141_141200


namespace freeze_alcohol_time_l141_141561

theorem freeze_alcohol_time :
  ∀ (init_temp freeze_temp : ℝ)
    (cooling_rate : ℝ), 
    init_temp = 12 → 
    freeze_temp = -117 → 
    cooling_rate = 1.5 →
    (freeze_temp - init_temp) / cooling_rate = -129 / cooling_rate :=
by
  intros init_temp freeze_temp cooling_rate h1 h2 h3
  rw [h2, h1, h3]
  exact sorry

end freeze_alcohol_time_l141_141561


namespace unique_solution_l141_141240
-- Import necessary mathematical library

-- Define mathematical statement
theorem unique_solution (N : ℕ) (hN: N > 0) :
  ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ (m + (1 / 2 : ℝ) * (m + n - 1) * (m + n - 2) = N) :=
by {
  sorry
}

end unique_solution_l141_141240


namespace contractor_fired_two_people_l141_141426

theorem contractor_fired_two_people
  (total_days : ℕ) (initial_people : ℕ) (days_worked : ℕ) (fraction_completed : ℚ)
  (remaining_days : ℕ) (people_fired : ℕ)
  (h1 : total_days = 100)
  (h2 : initial_people = 10)
  (h3 : days_worked = 20)
  (h4 : fraction_completed = 1/4)
  (h5 : remaining_days = 75)
  (h6 : remaining_days + days_worked = total_days)
  (h7 : people_fired = initial_people - 8) :
  people_fired = 2 :=
  sorry

end contractor_fired_two_people_l141_141426


namespace rectangle_area_solution_l141_141351

theorem rectangle_area_solution (x : ℝ) (h1 : (x + 3) * (2*x - 1) = 12*x + 5) : 
  x = (7 + Real.sqrt 113) / 4 :=
by 
  sorry

end rectangle_area_solution_l141_141351


namespace rhombus_perimeter_l141_141150

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  ∃ p : ℝ, p = 16 * Real.sqrt 13 := by
sorry

end rhombus_perimeter_l141_141150


namespace unique_roots_of_system_l141_141968

theorem unique_roots_of_system {x y z : ℂ} 
  (h1 : x + y + z = 3) 
  (h2 : x^2 + y^2 + z^2 = 3) 
  (h3 : x^3 + y^3 + z^3 = 3) : 
  (x = 1 ∧ y = 1 ∧ z = 1) :=
sorry

end unique_roots_of_system_l141_141968


namespace rectangular_prism_inequalities_l141_141177

variable {a b c : ℝ}

noncomputable def p (a b c : ℝ) := 4 * (a + b + c)
noncomputable def S (a b c : ℝ) := 2 * (a * b + b * c + c * a)
noncomputable def d (a b c : ℝ) := Real.sqrt (a^2 + b^2 + c^2)

theorem rectangular_prism_inequalities (h : a > b) (h1 : b > c) :
  a > (1 / 3) * (p a b c / 4 + Real.sqrt (d a b c ^ 2 - (1 / 2) * S a b c)) ∧
  c < (1 / 3) * (p a b c / 4 - Real.sqrt (d a b c ^ 2 - (1 / 2) * S a b c)) :=
by
  sorry

end rectangular_prism_inequalities_l141_141177


namespace application_methods_count_l141_141660

theorem application_methods_count (total_universities: ℕ) (universities_with_coinciding_exams: ℕ) (chosen_universities: ℕ) 
  (remaining_universities: ℕ) (remaining_combinations: ℕ) : 
  total_universities = 6 → universities_with_coinciding_exams = 2 → chosen_universities = 3 → 
  remaining_universities = 4 → remaining_combinations = 16 := 
by
  intros
  sorry

end application_methods_count_l141_141660


namespace train_speed_l141_141613

theorem train_speed 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h_train_length : train_length = 400) 
  (h_bridge_length : bridge_length = 300) 
  (h_crossing_time : crossing_time = 45) : 
  (train_length + bridge_length) / crossing_time = 700 / 45 := 
  by
    rw [h_train_length, h_bridge_length, h_crossing_time]
    sorry

end train_speed_l141_141613


namespace max_last_place_score_l141_141736

theorem max_last_place_score (n : ℕ) (h : n ≥ 4) :
  ∃ k, (∀ m, m < n -> (k + m) < (n * 3)) ∧ 
     (∀ i, ∃ j, j < n ∧ i = k + j) ∧
     (n * 2 - 2) = (k + n - 1) ∧ 
     k = n - 2 := 
sorry

end max_last_place_score_l141_141736


namespace youngest_child_age_l141_141075

theorem youngest_child_age (x : ℕ) 
  (h : x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) + (x + 24) = 112) : 
  x = 4 := by
  sorry

end youngest_child_age_l141_141075


namespace bacon_vs_tomatoes_l141_141526

theorem bacon_vs_tomatoes :
  let (n_b : ℕ) := 337
  let (n_t : ℕ) := 23
  n_b - n_t = 314 := by
  let n_b := 337
  let n_t := 23
  have h1 : n_b = 337 := rfl
  have h2 : n_t = 23 := rfl
  sorry

end bacon_vs_tomatoes_l141_141526


namespace triangle_side_c_l141_141636

theorem triangle_side_c
  (a b c : ℝ)
  (A B C : ℝ)
  (h_bc : b = 3)
  (h_sinC : Real.sin C = 56 / 65)
  (h_sinB : Real.sin B = 12 / 13)
  (h_Angles : A + B + C = π)
  (h_valid_triangle : ∀ {x y z : ℝ}, x + y > z ∧ x + z > y ∧ y + z > x):
  c = 14 / 5 :=
sorry

end triangle_side_c_l141_141636


namespace slope_of_tangent_at_0_l141_141659

theorem slope_of_tangent_at_0 (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp (2 * x)) : 
  (deriv f 0) = 2 :=
sorry

end slope_of_tangent_at_0_l141_141659


namespace potato_cost_l141_141517

variables (x : ℝ)
variables (b a : ℝ)

def andrey_earnings (x : ℝ) : ℝ := 120 * x
def boris_earnings (x : ℝ) : ℝ := 124.8 * x

theorem potato_cost :
  (boris_earnings x) - (andrey_earnings x) = 1200 → x = 250 :=
  by
    unfold andrey_earnings
    unfold boris_earnings
    sorry

end potato_cost_l141_141517


namespace original_price_of_tshirt_l141_141046

theorem original_price_of_tshirt :
  ∀ (P : ℝ), 
    (∀ discount quantity_sold revenue : ℝ, discount = 8 ∧ quantity_sold = 130 ∧ revenue = 5590 ∧
      revenue = quantity_sold * (P - discount)) → P = 51 := 
by
  intros P
  intro h
  sorry

end original_price_of_tshirt_l141_141046


namespace quotient_of_division_l141_141231

theorem quotient_of_division (dividend divisor remainder quotient : ℕ) 
  (h1 : dividend = 52) 
  (h2 : divisor = 3) 
  (h3 : remainder = 4) 
  (h4 : dividend = divisor * quotient + remainder) : 
  quotient = 16 :=
by
  sorry

end quotient_of_division_l141_141231


namespace abs_x_lt_2_sufficient_but_not_necessary_l141_141490

theorem abs_x_lt_2_sufficient_but_not_necessary (x : ℝ) :
  (|x| < 2) → (x ^ 2 - x - 6 < 0) ∧ ¬ ((x ^ 2 - x - 6 < 0) → (|x| < 2)) := by
  sorry

end abs_x_lt_2_sufficient_but_not_necessary_l141_141490


namespace trigonometric_identity_tan_two_l141_141607

theorem trigonometric_identity_tan_two (α : ℝ) (h : Real.tan α = 2) :
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 11 / 5 :=
by
  sorry

end trigonometric_identity_tan_two_l141_141607


namespace largest_n_sum_pos_l141_141155

section
variables {a : ℕ → ℤ}
variables {d : ℤ}
variables {n : ℕ}

axiom a_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom a1_pos : a 1 > 0
axiom a2013_2014_pos : a 2013 + a 2014 > 0
axiom a2013_2014_neg : a 2013 * a 2014 < 0

theorem largest_n_sum_pos :
  ∃ n : ℕ, (∀ k ≤ n, (k * (2 * a 1 + (k - 1) * d) / 2) > 0) → n = 4026 := sorry

end

end largest_n_sum_pos_l141_141155


namespace largest_triangle_perimeter_l141_141175

theorem largest_triangle_perimeter (x : ℤ) (hx1 : 7 + 11 > x) (hx2 : 7 + x > 11) (hx3 : 11 + x > 7) (hx4 : 5 ≤ x) (hx5 : x < 18) : 
  7 + 11 + x = 35 :=
sorry

end largest_triangle_perimeter_l141_141175


namespace bamboo_tube_middle_capacity_l141_141544

-- Definitions and conditions
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem bamboo_tube_middle_capacity:
  ∃ a d, (arithmetic_sequence a d 0 + arithmetic_sequence a d 1 + arithmetic_sequence a d 2 = 3.9) ∧
         (arithmetic_sequence a d 5 + arithmetic_sequence a d 6 + arithmetic_sequence a d 7 + arithmetic_sequence a d 8 = 3) ∧
         (arithmetic_sequence a d 4 = 1) :=
sorry

end bamboo_tube_middle_capacity_l141_141544


namespace greatest_possible_n_l141_141459

theorem greatest_possible_n (n : ℤ) (h : 93 * n^3 ≤ 145800) : n ≤ 11 :=
by {
  sorry
}

end greatest_possible_n_l141_141459


namespace find_tangent_c_l141_141393

theorem find_tangent_c (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x → (-12)^2 - 4 * (1) * (12 * c) = 0) → c = 3 :=
sorry

end find_tangent_c_l141_141393


namespace calc1_calc2_calc3_calc4_l141_141191

theorem calc1 : 327 + 46 - 135 = 238 := by sorry
theorem calc2 : 1000 - 582 - 128 = 290 := by sorry
theorem calc3 : (124 - 62) * 6 = 372 := by sorry
theorem calc4 : 500 - 400 / 5 = 420 := by sorry

end calc1_calc2_calc3_calc4_l141_141191


namespace mark_paired_with_mike_prob_l141_141724

def total_students := 16
def other_students := 15
def prob_pairing (mark: Nat) (mike: Nat) : ℚ := 1 / other_students

theorem mark_paired_with_mike_prob : prob_pairing 1 2 = 1 / 15 := 
sorry

end mark_paired_with_mike_prob_l141_141724


namespace no_equal_differences_between_products_l141_141927

theorem no_equal_differences_between_products (a b c d : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
    (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h_order : a < b ∧ b < c ∧ c < d) :
    ¬ (∃ k : ℕ, ac - ab = k ∧ ad - ac = k ∧ bc - ad = k ∧ bd - bc = k ∧ cd - bd = k) :=
by
  sorry

end no_equal_differences_between_products_l141_141927


namespace perimeter_square_l141_141025

-- Definition of the side length
def side_length : ℝ := 9

-- Definition of the perimeter calculation
def perimeter (s : ℝ) : ℝ := 4 * s

-- Theorem stating that the perimeter of a square with side length 9 cm is 36 cm
theorem perimeter_square : perimeter side_length = 36 := 
by sorry

end perimeter_square_l141_141025


namespace remainder_1125_1127_1129_div_12_l141_141111

theorem remainder_1125_1127_1129_div_12 :
  (1125 * 1127 * 1129) % 12 = 3 :=
by
  -- Proof can be written here
  sorry

end remainder_1125_1127_1129_div_12_l141_141111


namespace Goat_guilty_l141_141166

-- Condition definitions
def Goat_lied : Prop := sorry
def Beetle_testimony_true : Prop := sorry
def Mosquito_testimony_true : Prop := sorry
def Goat_accused_Beetle_or_Mosquito : Prop := sorry
def Beetle_accused_Goat_or_Mosquito : Prop := sorry
def Mosquito_accused_Beetle_or_Goat : Prop := sorry

-- Theorem: The Goat is guilty
theorem Goat_guilty (G_lied : Goat_lied) 
    (B_true : Beetle_testimony_true) 
    (M_true : Mosquito_testimony_true)
    (G_accuse : Goat_accused_Beetle_or_Mosquito)
    (B_accuse : Beetle_accused_Goat_or_Mosquito)
    (M_accuse : Mosquito_accused_Beetle_or_Goat) : 
  Prop :=
  sorry

end Goat_guilty_l141_141166


namespace radius_of_inscribed_circle_l141_141136

noncomputable def inscribed_circle_radius (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem radius_of_inscribed_circle :
  inscribed_circle_radius 6 8 10 = 2 :=
by
  sorry

end radius_of_inscribed_circle_l141_141136


namespace odd_square_minus_one_multiple_of_eight_l141_141616

theorem odd_square_minus_one_multiple_of_eight (a : ℤ) 
  (h₁ : a > 0) 
  (h₂ : a % 2 = 1) : 
  ∃ k : ℤ, a^2 - 1 = 8 * k :=
by
  sorry

end odd_square_minus_one_multiple_of_eight_l141_141616


namespace ship_speed_l141_141340

theorem ship_speed 
  (D : ℝ)
  (h1 : (D/2) - 200 = D/3)
  (S := (D / 2) / 20):
  S = 30 :=
by
  -- proof here
  sorry

end ship_speed_l141_141340


namespace find_x_l141_141248

def vector := (ℝ × ℝ)

-- Define the vectors a and b
def a (x : ℝ) : vector := (x, 3)
def b : vector := (3, 1)

-- Define the perpendicular condition
def perpendicular (v1 v2 : vector) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Prove that under the given conditions, x = -1
theorem find_x (x : ℝ) (h : perpendicular (a x) b) : x = -1 :=
  sorry

end find_x_l141_141248


namespace min_pieces_for_net_l141_141685

theorem min_pieces_for_net (n : ℕ) : ∃ (m : ℕ), m = n * (n + 1) := by
  sorry

end min_pieces_for_net_l141_141685


namespace product_of_g_at_roots_l141_141339

noncomputable def f (x : ℝ) : ℝ := x^5 + x^2 + 1
noncomputable def g (x : ℝ) : ℝ := x^2 - 2
noncomputable def roots : List ℝ := sorry -- To indicate the list of roots x_1, x_2, x_3, x_4, x_5 of the polynomial f(x)

theorem product_of_g_at_roots :
  (roots.map g).prod = -23 := sorry

end product_of_g_at_roots_l141_141339


namespace solution_k_values_l141_141951

theorem solution_k_values (k : ℕ) : 
  (∃ m n : ℕ, 0 < m ∧ 0 < n ∧ m * (m + k) = n * (n + 1)) 
  → k = 1 ∨ 4 ≤ k := 
by
  sorry

end solution_k_values_l141_141951


namespace max_items_for_2019_students_l141_141999

noncomputable def max_items (students : ℕ) : ℕ :=
  students / 2

theorem max_items_for_2019_students : max_items 2019 = 1009 := by
  sorry

end max_items_for_2019_students_l141_141999


namespace baseball_attendance_difference_l141_141293

theorem baseball_attendance_difference:
  ∃ C D: ℝ, 
    (59500 ≤ C ∧ C ≤ 80500 ∧ 69565 ≤ D ∧ D ≤ 94118) ∧ 
    (max (D - C) (C - D) = 35000 ∧ min (D - C) (C - D) = 11000) := by
  sorry

end baseball_attendance_difference_l141_141293


namespace ratio_of_pants_to_shirts_l141_141076

noncomputable def cost_shirt : ℝ := 6
noncomputable def cost_pants : ℝ := 8
noncomputable def num_shirts : ℝ := 10
noncomputable def total_cost : ℝ := 100

noncomputable def num_pants : ℝ :=
  (total_cost - (num_shirts * cost_shirt)) / cost_pants

theorem ratio_of_pants_to_shirts : num_pants / num_shirts = 1 / 2 := by
  sorry

end ratio_of_pants_to_shirts_l141_141076


namespace students_ages_average_l141_141261

variables (a b c : ℕ)

theorem students_ages_average (h1 : (14 * a + 13 * b + 12 * c) = 13 * (a + b + c)) : a = c :=
by
  sorry

end students_ages_average_l141_141261


namespace central_angle_of_sector_l141_141348

theorem central_angle_of_sector 
  (r : ℝ) (s : ℝ) (c : ℝ)
  (h1 : r = 5)
  (h2 : s = 15)
  (h3 : c = 2 * π * r) :
  ∃ n : ℝ, (n * s * π / 180 = c) ∧ n = 120 :=
by
  use 120
  sorry

end central_angle_of_sector_l141_141348


namespace product_of_variables_l141_141502

variables (a b c d : ℚ)

theorem product_of_variables :
  4 * a + 5 * b + 7 * c + 9 * d = 82 →
  d + c = 2 * b →
  2 * b + 2 * c = 3 * a →
  c - 2 = d →
  a * b * c * d = 276264960 / 14747943 := by
  sorry

end product_of_variables_l141_141502


namespace f_x_plus_1_l141_141034

-- Given function definition
def f (x : ℝ) := x^2

-- Statement to prove
theorem f_x_plus_1 (x : ℝ) : f (x + 1) = x^2 + 2 * x + 1 := 
by
  rw [f]
  -- This simplifies to:
  -- (x + 1)^2 = x^2 + 2 * x + 1
  sorry

end f_x_plus_1_l141_141034


namespace giant_slide_wait_is_15_l141_141062

noncomputable def wait_time_for_giant_slide
  (hours_at_carnival : ℕ) 
  (roller_coaster_wait : ℕ)
  (tilt_a_whirl_wait : ℕ)
  (rides_roller_coaster : ℕ)
  (rides_tilt_a_whirl : ℕ)
  (rides_giant_slide : ℕ) : ℕ :=
  
  (hours_at_carnival * 60 - (roller_coaster_wait * rides_roller_coaster + tilt_a_whirl_wait * rides_tilt_a_whirl)) / rides_giant_slide

theorem giant_slide_wait_is_15 :
  wait_time_for_giant_slide 4 30 60 4 1 4 = 15 := 
sorry

end giant_slide_wait_is_15_l141_141062


namespace eval_expression_l141_141751

theorem eval_expression (a x : ℤ) (h : x = a + 9) : (x - a + 5) = 14 :=
by
  sorry

end eval_expression_l141_141751


namespace even_of_form_4a_plus_2_not_diff_of_squares_l141_141208

theorem even_of_form_4a_plus_2_not_diff_of_squares (a x y : ℤ) : ¬ (4 * a + 2 = x^2 - y^2) :=
by sorry

end even_of_form_4a_plus_2_not_diff_of_squares_l141_141208


namespace diff_quotient_remainder_n_75_l141_141349

theorem diff_quotient_remainder_n_75 :
  ∃ n q r p : ℕ,  n = 75 ∧ n = 5 * q ∧ n = 34 * p + r ∧ q > r ∧ (q - r = 8) :=
by
  sorry

end diff_quotient_remainder_n_75_l141_141349


namespace product_of_odd_and_even_is_odd_l141_141706

theorem product_of_odd_and_even_is_odd {f g : ℝ → ℝ} 
  (hf : ∀ x : ℝ, f (-x) = -f x)
  (hg : ∀ x : ℝ, g (-x) = g x) :
  ∀ x : ℝ, (f x) * (g x) = -(f (-x) * g (-x)) :=
by
  sorry

end product_of_odd_and_even_is_odd_l141_141706


namespace abs_diff_of_two_numbers_l141_141983

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : |x - y| = 10 :=
by
  sorry

end abs_diff_of_two_numbers_l141_141983


namespace first_bag_brown_mms_l141_141058

theorem first_bag_brown_mms :
  ∀ (x : ℕ),
  (12 + 8 + 8 + 3 + x) / 5 = 8 → x = 9 :=
by
  intros x h
  sorry

end first_bag_brown_mms_l141_141058


namespace double_root_possible_values_l141_141921

theorem double_root_possible_values (b_3 b_2 b_1 : ℤ) (s : ℤ)
  (h : (Polynomial.X - Polynomial.C s) ^ 2 ∣
    Polynomial.C 24 + Polynomial.C b_1 * Polynomial.X + Polynomial.C b_2 * Polynomial.X ^ 2 + Polynomial.C b_3 * Polynomial.X ^ 3 + Polynomial.X ^ 4) :
  s = -2 ∨ s = -1 ∨ s = 1 ∨ s = 2 :=
sorry

end double_root_possible_values_l141_141921


namespace smallest_positive_integer_problem_l141_141052

theorem smallest_positive_integer_problem
  (n : ℕ) 
  (h1 : 50 ∣ n) 
  (h2 : (∃ e1 e2 e3 : ℕ, n = 2^e1 * 5^e2 * 3^e3 ∧ (e1 + 1) * (e2 + 1) * (e3 + 1) = 100)) 
  (h3 : ∀ m : ℕ, (50 ∣ m) → ((∃ e1 e2 e3 : ℕ, m = 2^e1 * 5^e2 * 3^e3 ∧ (e1 + 1) * (e2 + 1) * (e3 + 1) = 100) → (n ≤ m))) :
  n / 50 = 8100 := 
sorry

end smallest_positive_integer_problem_l141_141052


namespace journey_total_distance_l141_141594

theorem journey_total_distance (D : ℝ) 
  (train_fraction : ℝ := 3/5) 
  (bus_fraction : ℝ := 7/20) 
  (walk_distance : ℝ := 6.5) 
  (total_fraction : ℝ := 1) : 
  (1 - (train_fraction + bus_fraction)) * D = walk_distance → D = 130 := 
by
  sorry

end journey_total_distance_l141_141594


namespace inequality_always_holds_l141_141251

theorem inequality_always_holds (a b c : ℝ) (h1 : a > b) (h2 : a * b ≠ 0) : a + c > b + c :=
sorry

end inequality_always_holds_l141_141251


namespace total_height_of_sculpture_and_base_l141_141853

def height_of_sculpture_m : Float := 0.88
def height_of_base_cm : Float := 20
def meter_to_cm : Float := 100

theorem total_height_of_sculpture_and_base :
  (height_of_sculpture_m * meter_to_cm + height_of_base_cm) = 108 :=
by
  sorry

end total_height_of_sculpture_and_base_l141_141853


namespace find_triplets_geometric_and_arithmetic_prog_l141_141722

theorem find_triplets_geometric_and_arithmetic_prog :
  ∃ a1 a2 b1 b2,
    (a2 = a1 * ((12:ℚ) / a1) ∧ 12 = a1 * ((12:ℚ) / a1)^2) ∧
    (b2 = b1 + ((9:ℚ) - b1) / 2 ∧ 9 = b1 + 2 * (((9:ℚ) - b1) / 2)) ∧
    ((a1 = b1) ∧ (a2 = b2)) ∧ 
    (∀ (a1 a2 : ℚ), ((a1 = -9) ∧ (a2 = -6)) ∨ ((a1 = 15) ∧ (a2 = 12))) :=
by sorry

end find_triplets_geometric_and_arithmetic_prog_l141_141722


namespace problem_A_problem_B_problem_C_problem_D_l141_141948

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (Real.exp x)

theorem problem_A : ∀ x: ℝ, 0 < x ∧ x < 1 → f x < 0 := 
by sorry

theorem problem_B : ∃! (x : ℝ), ∃ c : ℝ, deriv f x = 0 := 
by sorry

theorem problem_C : ∀ (x : ℝ), ∃ c : ℝ, deriv f x = 0 → ¬∃ d : ℝ, d ≠ c ∧ deriv f d = 0 := 
by sorry

theorem problem_D : ¬ ∃ x₀ : ℝ, f x₀ = 1 / Real.exp 1 := 
by sorry

end problem_A_problem_B_problem_C_problem_D_l141_141948


namespace martha_total_clothes_l141_141419

-- Define the conditions
def jackets_bought : ℕ := 4
def t_shirts_bought : ℕ := 9
def free_jacket_condition : ℕ := 2
def free_t_shirt_condition : ℕ := 3

-- Define calculations based on conditions
def free_jackets : ℕ := jackets_bought / free_jacket_condition
def free_t_shirts : ℕ := t_shirts_bought / free_t_shirt_condition
def total_jackets := jackets_bought + free_jackets
def total_t_shirts := t_shirts_bought + free_t_shirts
def total_clothes := total_jackets + total_t_shirts

-- Prove the total number of clothes
theorem martha_total_clothes : total_clothes = 18 :=
by
    sorry

end martha_total_clothes_l141_141419


namespace pure_imaginary_m_eq_zero_l141_141269

noncomputable def z (m : ℝ) : ℂ := (m * (m - 1) : ℂ) + (m - 1) * Complex.I

theorem pure_imaginary_m_eq_zero (m : ℝ) (h : z m = (m - 1) * Complex.I) : m = 0 :=
by
  sorry

end pure_imaginary_m_eq_zero_l141_141269


namespace cost_of_used_cd_l141_141040

theorem cost_of_used_cd (N U : ℝ) 
    (h1 : 6 * N + 2 * U = 127.92) 
    (h2 : 3 * N + 8 * U = 133.89) :
    U = 9.99 :=
by 
  sorry

end cost_of_used_cd_l141_141040


namespace triangle_angle_A_eq_60_l141_141788

theorem triangle_angle_A_eq_60 (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π)
  (h_tan : (Real.tan A) / (Real.tan B) = (2 * c - b) / b) : 
  A = π / 3 :=
by
  sorry

end triangle_angle_A_eq_60_l141_141788


namespace late_fisherman_arrival_l141_141731

-- Definitions of conditions
variables (n d : ℕ) -- n is the number of fishermen on Monday, d is the number of days the late fisherman fished
variable (total_fish : ℕ := 370)
variable (fish_per_day_per_fisherman : ℕ := 10)
variable (days_fished : ℕ := 5) -- From Monday to Friday

-- Condition in Lean: total fish caught from Monday to Friday
def total_fish_caught (n d : ℕ) := 50 * n + 10 * d

theorem late_fisherman_arrival (n d : ℕ) (h : total_fish_caught n d = 370) : 
  d = 2 :=
by
  sorry

end late_fisherman_arrival_l141_141731


namespace isosceles_triangle_roots_l141_141048

theorem isosceles_triangle_roots (k : ℝ) (a b : ℝ) 
  (h1 : a = 2 ∨ b = 2)
  (h2 : a^2 - 6 * a + k = 0)
  (h3 : b^2 - 6 * b + k = 0) :
  k = 9 :=
by
  sorry

end isosceles_triangle_roots_l141_141048


namespace virginia_more_than_adrienne_l141_141112

def teaching_years (V A D : ℕ) : Prop :=
  V + A + D = 102 ∧ D = 43 ∧ V = D - 9

theorem virginia_more_than_adrienne (V A : ℕ) (h : teaching_years V A 43) : V - A = 9 :=
by
  sorry

end virginia_more_than_adrienne_l141_141112


namespace geometric_series_has_value_a_l141_141577

theorem geometric_series_has_value_a (a : ℝ) (S : ℕ → ℝ)
  (h : ∀ n, S (n + 1) = a * (1 / 4) ^ n + 6) :
  a = -3 / 2 :=
sorry

end geometric_series_has_value_a_l141_141577


namespace compare_powers_l141_141244

theorem compare_powers (a b c : ℝ) (h1 : a = 2^555) (h2 : b = 3^444) (h3 : c = 6^222) : a < c ∧ c < b :=
by
  sorry

end compare_powers_l141_141244


namespace distance_walked_is_18_miles_l141_141044

-- Defining the variables for speed, time, and distance
variables (x t d : ℕ)

-- Declaring the conditions given in the problem
def walked_distance_at_usual_rate : Prop :=
  d = x * t

def walked_distance_at_increased_rate : Prop :=
  d = (x + 1) * (3 * t / 4)

def walked_distance_at_decreased_rate : Prop :=
  d = (x - 1) * (t + 3)

-- The proof problem statement to show the distance walked is 18 miles
theorem distance_walked_is_18_miles
  (hx : walked_distance_at_usual_rate x t d)
  (hz : walked_distance_at_increased_rate x t d)
  (hy : walked_distance_at_decreased_rate x t d) :
  d = 18 := by
  sorry

end distance_walked_is_18_miles_l141_141044


namespace johns_shell_arrangements_l141_141635

-- Define the total number of arrangements without considering symmetries
def totalArrangements := Nat.factorial 12

-- Define the number of equivalent arrangements due to symmetries
def symmetries := 6 * 2

-- Define the number of distinct arrangements
def distinctArrangements : Nat := totalArrangements / symmetries

-- State the theorem
theorem johns_shell_arrangements : distinctArrangements = 479001600 :=
by
  sorry

end johns_shell_arrangements_l141_141635


namespace tangents_collinear_F_minimum_area_triangle_l141_141173

noncomputable def ellipse_condition : Prop :=
  ∀ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1

noncomputable def point_P_on_line (P : ℝ × ℝ) : Prop :=
  P.1 = 4

noncomputable def tangent_condition (P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop) : Prop :=
  -- Tangent lines meet the ellipse equation at points A and B
  ellipse A ∧ ellipse B

noncomputable def collinear (A F B : ℝ × ℝ) : Prop :=
  (A.2 - F.2) * (B.1 - F.1) = (B.2 - F.2) * (A.1 - F.1)

noncomputable def minimum_area (P A B : ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((A.1 * B.2 + B.1 * P.2 + P.1 * A.2) - (A.2 * B.1 + B.2 * P.1 + P.2 * A.1))

theorem tangents_collinear_F (F : ℝ × ℝ) (P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop)
  (h_ellipse_F : F = (1, 0))
  (h_point_P_on_line : point_P_on_line P)
  (h_tangent_condition : tangent_condition P A B ellipse)
  (h_ellipse_def : ellipse_condition) :
  collinear A F B :=
sorry

theorem minimum_area_triangle (F P A B : ℝ × ℝ) (ellipse : ℝ × ℝ → Prop)
  (h_ellipse_F : F = (1, 0))
  (h_point_P_on_line : point_P_on_line P)
  (h_tangent_condition : tangent_condition P A B ellipse)
  (h_ellipse_def : ellipse_condition) :
  minimum_area P A B = 9 / 2 :=
sorry

end tangents_collinear_F_minimum_area_triangle_l141_141173


namespace food_bank_remaining_after_four_weeks_l141_141778

def week1_donated : ℝ := 40
def week1_given_out : ℝ := 0.6 * week1_donated
def week1_remaining : ℝ := week1_donated - week1_given_out

def week2_donated : ℝ := 1.5 * week1_donated
def week2_given_out : ℝ := 0.7 * week2_donated
def week2_remaining : ℝ := week2_donated - week2_given_out
def total_remaining_after_week2 : ℝ := week1_remaining + week2_remaining

def week3_donated : ℝ := 1.25 * week2_donated
def week3_given_out : ℝ := 0.8 * week3_donated
def week3_remaining : ℝ := week3_donated - week3_given_out
def total_remaining_after_week3 : ℝ := total_remaining_after_week2 + week3_remaining

def week4_donated : ℝ := 0.9 * week3_donated
def week4_given_out : ℝ := 0.5 * week4_donated
def week4_remaining : ℝ := week4_donated - week4_given_out
def total_remaining_after_week4 : ℝ := total_remaining_after_week3 + week4_remaining

theorem food_bank_remaining_after_four_weeks : total_remaining_after_week4 = 82.75 := by
  sorry

end food_bank_remaining_after_four_weeks_l141_141778


namespace oranges_needed_l141_141574

theorem oranges_needed 
  (total_fruit_needed : ℕ := 12) 
  (apples : ℕ := 3) 
  (bananas : ℕ := 4) : 
  total_fruit_needed - (apples + bananas) = 5 :=
by 
  sorry

end oranges_needed_l141_141574


namespace parallel_lines_slope_l141_141603

-- Define the equations of the lines in Lean
def line1 (x : ℝ) : ℝ := 7 * x + 3
def line2 (c : ℝ) (x : ℝ) : ℝ := (3 * c) * x + 5

-- State the theorem: if the lines are parallel, then c = 7/3
theorem parallel_lines_slope (c : ℝ) :
  (∀ x : ℝ, (7 * x + 3 = (3 * c) * x + 5)) → c = (7/3) :=
by
  sorry

end parallel_lines_slope_l141_141603


namespace circle_line_intersection_l141_141108

theorem circle_line_intersection (a : ℝ) : 
  (∃ x y : ℝ, (x - y + 1 = 0) ∧ ((x - a)^2 + y^2 = 2)) ↔ -3 ≤ a ∧ a ≤ 1 := 
by
  sorry

end circle_line_intersection_l141_141108


namespace bob_should_give_l141_141535

theorem bob_should_give (alice_paid bob_paid charlie_paid : ℕ)
  (h_alice : alice_paid = 120)
  (h_bob : bob_paid = 150)
  (h_charlie : charlie_paid = 180) :
  bob_paid - (120 + 150 + 180) / 3 = 0 := 
by
  sorry

end bob_should_give_l141_141535


namespace ratio_length_breadth_l141_141392

-- Define the conditions
def length := 135
def area := 6075

-- Define the breadth in terms of the area and length
def breadth := area / length

-- The problem statement as a Lean 4 theorem to prove the ratio
theorem ratio_length_breadth : length / breadth = 3 := 
by
  -- Proof goes here
  sorry

end ratio_length_breadth_l141_141392


namespace probability_of_black_ball_l141_141860

/-- Let the probability of drawing a red ball be 0.42, and the probability of drawing a white ball be 0.28. Prove that the probability of drawing a black ball is 0.3. -/
theorem probability_of_black_ball (p_red p_white p_black : ℝ) (h1 : p_red = 0.42) (h2 : p_white = 0.28) (h3 : p_red + p_white + p_black = 1) : p_black = 0.3 :=
by
  sorry

end probability_of_black_ball_l141_141860


namespace geometric_sequence_ratio_l141_141562

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 + a 8 = 15) 
  (h2 : a 3 * a 7 = 36) 
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  (a 19 / a 13 = 4) ∨ (a 19 / a 13 = 1 / 4) :=
by
  sorry

end geometric_sequence_ratio_l141_141562


namespace first_part_results_count_l141_141133

theorem first_part_results_count : 
    ∃ n, n * 10 + 90 + (25 - n) * 20 = 25 * 18 ∧ n = 14 :=
by
  sorry

end first_part_results_count_l141_141133


namespace quadratic_distinct_zeros_l141_141012

theorem quadratic_distinct_zeros (m : ℝ) : 
  (x^2 + m * x + (m + 3)) = 0 → 
  (0 < m^2 - 4 * (m + 3)) ↔ (m < -2) ∨ (m > 6) :=
sorry

end quadratic_distinct_zeros_l141_141012


namespace proof_allison_brian_noah_l141_141919

-- Definitions based on the problem conditions

-- Definition for the cubes
def allison_cube := [6, 6, 6, 6, 6, 6]
def brian_cube := [1, 2, 2, 3, 3, 4]
def noah_cube := [3, 3, 3, 3, 5, 5]

-- Helper function to calculate the probability of succeeding conditions
def probability_succeeding (A B C : List ℕ) : ℚ :=
  if (A.all (λ x => x = 6)) ∧ (B.all (λ x => x ≤ 5)) ∧ (C.all (λ x => x ≤ 5)) then 1 else 0

-- Define the proof statement for the given problem
theorem proof_allison_brian_noah :
  probability_succeeding allison_cube brian_cube noah_cube = 1 :=
by
  -- Since all conditions fulfill the requirement, we'll use sorry to skip the proof for now
  sorry

end proof_allison_brian_noah_l141_141919


namespace ring_area_l141_141436

theorem ring_area (r1 r2 : ℝ) (h1 : r1 = 12) (h2 : r2 = 5) : 
  (π * r1^2) - (π * r2^2) = 119 * π := 
by simp [h1, h2]; sorry

end ring_area_l141_141436


namespace faye_pencils_l141_141317

theorem faye_pencils (rows : ℕ) (pencils_per_row : ℕ) (h_rows : rows = 30) (h_pencils_per_row : pencils_per_row = 24) :
  rows * pencils_per_row = 720 :=
by
  sorry

end faye_pencils_l141_141317


namespace remaining_amount_to_be_paid_l141_141894

theorem remaining_amount_to_be_paid (part_payment : ℝ) (percentage : ℝ) (h : part_payment = 650 ∧ percentage = 0.15) :
    (part_payment / percentage - part_payment) = 3683.33 := by
  cases h with
  | intro h1 h2 =>
    sorry

end remaining_amount_to_be_paid_l141_141894


namespace absolute_sum_value_l141_141205

theorem absolute_sum_value (x1 x2 x3 x4 x5 : ℝ) 
(h : x1 + 1 = x2 + 2 ∧ x2 + 2 = x3 + 3 ∧ x3 + 3 = x4 + 4 ∧ x4 + 4 = x5 + 5 ∧ x5 + 5 = x1 + x2 + x3 + x4 + x5 + 6) :
  |(x1 + x2 + x3 + x4 + x5)| = 3.75 := 
by
  sorry

end absolute_sum_value_l141_141205


namespace correct_option_l141_141908

theorem correct_option :
  (3 * a^2 - a^2 = 2 * a^2) ∧
  (¬ (a^2 * a^3 = a^6)) ∧
  (¬ ((3 * a)^2 = 6 * a^2)) ∧
  (¬ (a^6 / a^3 = a^2)) :=
by
  -- We only need to state the theorem; the proof details are omitted per the instructions.
  sorry

end correct_option_l141_141908


namespace induction_base_case_l141_141579

theorem induction_base_case : (-1 : ℤ) + 3 - 5 + (-1)^2 * 1 = (-1 : ℤ) := sorry

end induction_base_case_l141_141579


namespace radius_decrease_l141_141680

theorem radius_decrease (r r' : ℝ) (A A' : ℝ) (h_original_area : A = π * r^2)
  (h_area_decrease : A' = 0.25 * A) (h_new_area : A' = π * r'^2) : r' = 0.5 * r :=
by
  sorry

end radius_decrease_l141_141680


namespace box_one_contains_at_least_one_ball_l141_141431

-- Define the conditions
def boxes : List ℕ := [1, 2, 3, 4]
def balls : List ℕ := [1, 2, 3]

-- Define the problem
def count_ways_box_one_contains_ball :=
  let total_ways := (boxes.length)^(balls.length)
  let ways_box_one_empty := (boxes.length - 1)^(balls.length)
  total_ways - ways_box_one_empty

-- The proof problem statement
theorem box_one_contains_at_least_one_ball : count_ways_box_one_contains_ball = 37 := by
  sorry

end box_one_contains_at_least_one_ball_l141_141431


namespace sum_interior_numbers_eight_l141_141140

noncomputable def sum_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2 -- This is a general formula derived from the pattern

theorem sum_interior_numbers_eight :
  sum_interior_numbers 8 = 126 :=
by
  -- No proof required, so we use sorry.
  sorry

end sum_interior_numbers_eight_l141_141140


namespace boar_sausages_left_l141_141321

def boar_sausages_final_count(sausages_initial : ℕ) : ℕ :=
  let after_monday := sausages_initial - (2 / 5 * sausages_initial)
  let after_tuesday := after_monday - (1 / 2 * after_monday)
  let after_wednesday := after_tuesday - (1 / 4 * after_tuesday)
  let after_thursday := after_wednesday - (1 / 3 * after_wednesday)
  let after_sharing := after_thursday - (1 / 5 * after_thursday)
  let after_eating := after_sharing - (3 / 5 * after_sharing)
  after_eating

theorem boar_sausages_left : boar_sausages_final_count 1200 = 58 := 
  sorry

end boar_sausages_left_l141_141321


namespace teams_have_equal_people_l141_141886

-- Definitions capturing the conditions
def managers : Nat := 3
def employees : Nat := 3
def teams : Nat := 3

-- The total number of people
def total_people : Nat := managers + employees

-- The proof statement
theorem teams_have_equal_people : total_people / teams = 2 := by
  sorry

end teams_have_equal_people_l141_141886


namespace dividend_correct_l141_141406

-- Given constants for the problem
def divisor := 19
def quotient := 7
def remainder := 6

-- Dividend formula
def dividend := (divisor * quotient) + remainder

-- The proof problem statement
theorem dividend_correct : dividend = 139 := by
  sorry

end dividend_correct_l141_141406


namespace sum_of_final_two_numbers_l141_141022

theorem sum_of_final_two_numbers (x y T : ℕ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 :=
by
  sorry

end sum_of_final_two_numbers_l141_141022


namespace present_age_of_son_is_22_l141_141249

theorem present_age_of_son_is_22 (S F : ℕ) (h1 : F = S + 24) (h2 : F + 2 = 2 * (S + 2)) : S = 22 :=
by
  sorry

end present_age_of_son_is_22_l141_141249


namespace remainder_of_2_pow_2017_mod_11_l141_141287

theorem remainder_of_2_pow_2017_mod_11 : (2 ^ 2017) % 11 = 7 := by
  sorry

end remainder_of_2_pow_2017_mod_11_l141_141287


namespace carls_garden_area_is_correct_l141_141777

-- Define the conditions
def isRectangle (length width : ℕ) : Prop :=
∃ l w, l * w = length * width

def validFencePosts (shortSidePosts longSidePosts totalPosts : ℕ) : Prop :=
∃ x, totalPosts = 2 * x + 2 * (2 * x) - 4 ∧ x = shortSidePosts

def validSpacing (shortSideSpaces longSideSpaces : ℕ) : Prop :=
shortSideSpaces = 4 * (shortSideSpaces - 1) ∧ longSideSpaces = 4 * (longSideSpaces - 1)

def correctArea (shortSide longSide expectedArea : ℕ) : Prop :=
shortSide * longSide = expectedArea

-- Prove the conditions lead to the expected area
theorem carls_garden_area_is_correct :
  ∃ shortSide longSide,
  isRectangle shortSide longSide ∧
  validFencePosts 5 10 24 ∧
  validSpacing 5 10 ∧
  correctArea (4 * (5-1)) (4 * (10-1)) 576 :=
by
  sorry

end carls_garden_area_is_correct_l141_141777


namespace original_weight_of_beef_l141_141593

variable (W : ℝ)

def first_stage_weight := 0.80 * W
def second_stage_weight := 0.70 * (first_stage_weight W)
def third_stage_weight := 0.75 * (second_stage_weight W)

theorem original_weight_of_beef :
  third_stage_weight W = 392 → W = 933.33 :=
by
  intro h
  sorry

end original_weight_of_beef_l141_141593


namespace product_of_equal_numbers_l141_141226

theorem product_of_equal_numbers (a b : ℕ) (mean : ℕ) (sum : ℕ)
  (h1 : mean = 20)
  (h2 : a = 22)
  (h3 : b = 34)
  (h4 : sum = 4 * mean)
  (h5 : sum - a - b = 2 * x)
  (h6 : sum = 80)
  (h7 : x = 12) 
  : x * x = 144 :=
by
  sorry

end product_of_equal_numbers_l141_141226


namespace find_present_ratio_l141_141199

noncomputable def present_ratio_of_teachers_to_students : Prop :=
  ∃ (S T S' T' : ℕ),
    (T = 3) ∧
    (S = 50 * T) ∧
    (S' = S + 50) ∧
    (T' = T + 5) ∧
    (S' / T' = 25 / 1) ∧ 
    (T / S = 1 / 50)

theorem find_present_ratio : present_ratio_of_teachers_to_students :=
by
  sorry

end find_present_ratio_l141_141199


namespace train_length_is_135_l141_141286

noncomputable def speed_km_per_hr : ℝ := 54
noncomputable def time_seconds : ℝ := 9
noncomputable def speed_m_per_s : ℝ := speed_km_per_hr * (1000 / 3600)
noncomputable def length_of_train : ℝ := speed_m_per_s * time_seconds

theorem train_length_is_135 : length_of_train = 135 := by
  sorry

end train_length_is_135_l141_141286


namespace simplify_vector_eq_l141_141011

-- Define points A, B, C
variables {A B C O : Type} [AddGroup A]

-- Define vector operations corresponding to overrightarrow.
variables (AB OC OB AC AO BO : A)

-- Conditions in Lean definitions
-- Assuming properties like vector addition and subtraction, and associative properties
def vector_eq : Prop := AB + OC - OB = AC

theorem simplify_vector_eq :
  AB + OC - OB = AC :=
by
  -- Proof steps go here
  sorry

end simplify_vector_eq_l141_141011


namespace zero_points_in_intervals_l141_141761

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * x - Real.log x

theorem zero_points_in_intervals :
  (∀ x : ℝ, x ∈ Set.Ioo (1 / Real.exp 1) 1 → f x ≠ 0) ∧
  (∃ x : ℝ, x ∈ Set.Ioo 1 (Real.exp 1) ∧ f x = 0) :=
by
  sorry

end zero_points_in_intervals_l141_141761


namespace cheerleader_total_l141_141782

theorem cheerleader_total 
  (size2 : ℕ)
  (size6 : ℕ)
  (size12 : ℕ)
  (h1 : size2 = 4)
  (h2 : size6 = 10)
  (h3 : size12 = size6 / 2) :
  size2 + size6 + size12 = 19 :=
by
  sorry

end cheerleader_total_l141_141782


namespace trajectory_of_center_of_moving_circle_l141_141091

noncomputable def center_trajectory (x y : ℝ) : Prop :=
  0 < y ∧ y ≤ 1 ∧ x^2 = 4 * (y - 1)

theorem trajectory_of_center_of_moving_circle (x y : ℝ) :
  0 ≤ y ∧ y ≤ 2 ∧ x^2 + y^2 = 4 ∧ 0 < y → center_trajectory x y :=
by
  sorry

end trajectory_of_center_of_moving_circle_l141_141091


namespace escalator_steps_l141_141453

theorem escalator_steps
  (x : ℕ)
  (time_me : ℕ := 60)
  (steps_me : ℕ := 20)
  (time_wife : ℕ := 72)
  (steps_wife : ℕ := 16)
  (escalator_speed_me : x - steps_me = 60 * (x - 20) / 72)
  (escalator_speed_wife : x - steps_wife = 72 * (x - 16) / 60) :
  x = 40 := by
  sorry

end escalator_steps_l141_141453


namespace max_plus_min_value_of_f_l141_141652

noncomputable def f (x : ℝ) : ℝ := (2 * (x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_plus_min_value_of_f :
  let M := ⨆ x, f x
  let m := ⨅ x, f x
  M + m = 4 :=
by 
  sorry

end max_plus_min_value_of_f_l141_141652


namespace average_minutes_correct_l141_141032

variable (s : ℕ)
def sixth_graders := 3 * s
def seventh_graders := s
def eighth_graders := s / 2

def minutes_sixth_graders := 18 * sixth_graders s
def minutes_seventh_graders := 20 * seventh_graders s
def minutes_eighth_graders := 22 * eighth_graders s

def total_minutes := minutes_sixth_graders s + minutes_seventh_graders s + minutes_eighth_graders s
def total_students := sixth_graders s + seventh_graders s + eighth_graders s

def average_minutes := total_minutes s / total_students s

theorem average_minutes_correct : average_minutes s = 170 / 9 := sorry

end average_minutes_correct_l141_141032


namespace find_number_l141_141869

theorem find_number (x : ℝ) (h : x / 100 = 31.76 + 0.28) : x = 3204 := 
  sorry

end find_number_l141_141869


namespace largest_number_not_sum_of_two_composites_l141_141892

-- Define what it means to be a composite number
def isComposite (n : ℕ) : Prop :=
∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the problem statement
theorem largest_number_not_sum_of_two_composites :
  ∃ n : ℕ, (¬∃ a b : ℕ, isComposite a ∧ isComposite b ∧ n = a + b) ∧
           ∀ m : ℕ, (¬∃ x y : ℕ, isComposite x ∧ isComposite y ∧ m = x + y) → m ≥ n :=
  sorry

end largest_number_not_sum_of_two_composites_l141_141892


namespace fraction_numerator_l141_141679

theorem fraction_numerator (x : ℕ) (h1 : 4 * x - 4 > 0) (h2 : (x : ℚ) / (4 * x - 4) = 3 / 8) : x = 3 :=
by {
  sorry
}

end fraction_numerator_l141_141679


namespace candy_initial_amount_l141_141388

namespace CandyProblem

variable (initial_candy given_candy left_candy : ℕ)

theorem candy_initial_amount (h1 : given_candy = 10) (h2 : left_candy = 68) (h3 : left_candy = initial_candy - given_candy) : initial_candy = 78 := 
  sorry
end CandyProblem

end candy_initial_amount_l141_141388


namespace original_treadmill_price_l141_141534

-- Given conditions in Lean definitions
def discount_rate : ℝ := 0.30
def plate_cost : ℝ := 50
def num_plates : ℕ := 2
def total_paid : ℝ := 1045

noncomputable def treadmill_price :=
  let plate_total := num_plates * plate_cost
  let treadmill_discount := (1 - discount_rate)
  (total_paid - plate_total) / treadmill_discount

theorem original_treadmill_price :
  treadmill_price = 1350 := by
  sorry

end original_treadmill_price_l141_141534


namespace probability_of_popped_white_is_12_over_17_l141_141601

noncomputable def probability_white_given_popped (white_kernels yellow_kernels : ℚ) (pop_white pop_yellow : ℚ) : ℚ :=
  let p_white_popped := white_kernels * pop_white
  let p_yellow_popped := yellow_kernels * pop_yellow
  let p_popped := p_white_popped + p_yellow_popped
  p_white_popped / p_popped

theorem probability_of_popped_white_is_12_over_17 :
  probability_white_given_popped (3/4) (1/4) (3/5) (3/4) = 12/17 :=
by
  sorry

end probability_of_popped_white_is_12_over_17_l141_141601


namespace gift_equation_l141_141901

theorem gift_equation (x : ℝ) : 15 * (x + 40) = 900 := 
by
  sorry

end gift_equation_l141_141901


namespace original_speed_of_person_B_l141_141709

-- Let v_A and v_B be the speeds of person A and B respectively
variable (v_A v_B : ℝ)

-- Conditions for problem
axiom initial_ratio : v_A / v_B = (5 / 4 * v_A) / (v_B + 10)

-- The goal: Prove that v_B = 40
theorem original_speed_of_person_B : v_B = 40 := 
  sorry

end original_speed_of_person_B_l141_141709


namespace shifted_parabola_eq_l141_141228

-- Definitions
def original_parabola (x y : ℝ) : Prop := y = 3 * x^2

def shifted_origin (x' y' x y : ℝ) : Prop :=
  (x' = x + 1) ∧ (y' = y + 1)

-- Target statement
theorem shifted_parabola_eq : ∀ (x y x' y' : ℝ),
  original_parabola x y →
  shifted_origin x' y' x y →
  y' = 3*(x' - 1)*(x' - 1) + 1 → 
  y = 3*(x + 1)*(x + 1) - 1 :=
by
  intros x y x' y' h_orig h_shifted h_new_eq
  sorry

end shifted_parabola_eq_l141_141228


namespace arithmetic_sequence_8th_term_l141_141373

theorem arithmetic_sequence_8th_term 
    (a₁ : ℝ) (a₅ : ℝ) (n : ℕ) (a₈ : ℝ) 
    (h₁ : a₁ = 3) 
    (h₂ : a₅ = 78) 
    (h₃ : n = 25) : 
    a₈ = 24.875 := by
  sorry

end arithmetic_sequence_8th_term_l141_141373


namespace find_g_at_6_l141_141903

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 20 * x ^ 3 + 37 * x ^ 2 - 18 * x - 80

theorem find_g_at_6 : g 6 = 712 := by
  -- We apply the remainder theorem to determine the value of g(6).
  sorry

end find_g_at_6_l141_141903


namespace starting_number_divisible_by_3_l141_141549

theorem starting_number_divisible_by_3 (x : ℕ) (h₁ : ∀ n, 1 ≤ n → n < 14 → ∃ k, x + (n - 1) * 3 = 3 * k ∧ x + (n - 1) * 3 ≤ 50) :
  x = 12 :=
by
  sorry

end starting_number_divisible_by_3_l141_141549


namespace largest_class_students_l141_141787

theorem largest_class_students (x : ℕ)
  (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 95) :
  x = 23 :=
by
  sorry

end largest_class_students_l141_141787


namespace hyperbola_asymptotes_l141_141478

theorem hyperbola_asymptotes:
  (∀ x y : Real, (x^2 / 16 - y^2 / 9 = 1) → (y = 3 / 4 * x ∨ y = -3 / 4 * x)) :=
by {
  sorry
}

end hyperbola_asymptotes_l141_141478


namespace shopkeeper_gain_first_pulse_shopkeeper_gain_second_pulse_shopkeeper_gain_third_pulse_l141_141123

def false_weight_kgs (false_weight_g : ℕ) : ℚ := false_weight_g / 1000

def shopkeeper_gain_percentage (false_weight_g price_per_kg : ℕ) : ℚ :=
  let actual_price := false_weight_kgs false_weight_g * price_per_kg
  let gain := price_per_kg - actual_price
  (gain / actual_price) * 100

theorem shopkeeper_gain_first_pulse :
  shopkeeper_gain_percentage 950 10 = 5.26 := 
sorry

theorem shopkeeper_gain_second_pulse :
  shopkeeper_gain_percentage 960 15 = 4.17 := 
sorry

theorem shopkeeper_gain_third_pulse :
  shopkeeper_gain_percentage 970 20 = 3.09 := 
sorry

end shopkeeper_gain_first_pulse_shopkeeper_gain_second_pulse_shopkeeper_gain_third_pulse_l141_141123


namespace part1_solution_set_part2_inequality_l141_141954

noncomputable def f (x : ℝ) : ℝ := 
  x * Real.exp (x + 1)

theorem part1_solution_set (h : 0 < x) : 
  f x < 3 * Real.log 3 - 3 ↔ 0 < x ∧ x < Real.log 3 - 1 :=
sorry

theorem part2_inequality (h1 : f x1 = 3 * Real.exp x1 + 3 * Real.exp (Real.log x1)) 
    (h2 : f x2 = 3 * Real.exp x2 + 3 * Real.exp (Real.log x2)) (h_distinct : x1 ≠ x2) :
  x1 + x2 + Real.log (x1 * x2) > 2 :=
sorry

end part1_solution_set_part2_inequality_l141_141954


namespace not_all_positive_l141_141982

theorem not_all_positive (a b c : ℝ) (h1 : a + b + c = 4) (h2 : a^2 + b^2 + c^2 = 12) (h3 : a * b * c = 1) : a ≤ 0 ∨ b ≤ 0 ∨ c ≤ 0 :=
sorry

end not_all_positive_l141_141982


namespace base_length_of_isosceles_triangle_l141_141139

theorem base_length_of_isosceles_triangle 
  (a b : ℕ) 
  (h1 : a = 6) 
  (h2 : b = 6) 
  (perimeter : ℕ) 
  (h3 : 2*a + b = perimeter)
  (h4 : perimeter = 20) 
  : b = 8 := 
by
  sorry

end base_length_of_isosceles_triangle_l141_141139


namespace unique_solution_l141_141682

noncomputable def system_of_equations (x y z : ℝ) : Prop :=
  (2 * x^3 = 2 * y * (x^2 + 1) - (z^2 + 1)) ∧
  (2 * y^4 = 3 * z * (y^2 + 1) - 2 * (x^2 + 1)) ∧
  (2 * z^5 = 4 * x * (z^2 + 1) - 3 * (y^2 + 1))

theorem unique_solution : ∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) → system_of_equations x y z → (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  intro x y z hx hy hz h
  sorry

end unique_solution_l141_141682


namespace find_a_b_c_l141_141955

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 37) / 2 + 3 / 2)

theorem find_a_b_c :
  ∃ a b c : ℕ, (x^80 = 2 * x^78 + 8 * x^76 + 9 * x^74 - x^40 + a * x^36 + b * x^34 + c * x^30) ∧ (a + b + c = 151) :=
by
  sorry

end find_a_b_c_l141_141955


namespace a_b_condition_l141_141013

theorem a_b_condition (a b : ℂ) (h : (a + b) / a = b / (a + b)) :
  (∃ x y : ℂ, x = a ∧ y = b ∧ ((¬ x.im = 0 ∧ y.im = 0) ∨ (x.im = 0 ∧ ¬ y.im = 0) ∨ (¬ x.im = 0 ∧ ¬ y.im = 0))) :=
by
  sorry

end a_b_condition_l141_141013


namespace sector_area_l141_141333

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = 2 * Real.pi / 5) (hr : r = 20) :
  1 / 2 * r^2 * θ = 80 * Real.pi := by
  sorry

end sector_area_l141_141333


namespace cost_of_two_books_and_one_magazine_l141_141703

-- Definitions of the conditions
def condition1 (x y : ℝ) : Prop := 3 * x + 2 * y = 18.40
def condition2 (x y : ℝ) : Prop := 2 * x + 3 * y = 17.60

-- Proof problem
theorem cost_of_two_books_and_one_magazine (x y : ℝ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  2 * x + y = 11.20 :=
sorry

end cost_of_two_books_and_one_magazine_l141_141703


namespace break_even_point_l141_141825

def cost_of_commodity (a : ℝ) : ℝ := a

def profit_beginning_of_month (a : ℝ) : ℝ := 100 + (a + 100) * 0.024

def profit_end_of_month : ℝ := 115

theorem break_even_point (a : ℝ) : profit_end_of_month - profit_beginning_of_month a = 0 → a = 525 := 
by sorry

end break_even_point_l141_141825


namespace forgotten_angles_sum_l141_141813

theorem forgotten_angles_sum (n : ℕ) (h : (n-2) * 180 = 3240 + x) : x = 180 :=
by {
  sorry
}

end forgotten_angles_sum_l141_141813


namespace HCF_of_two_numbers_l141_141267

theorem HCF_of_two_numbers (H L : ℕ) (product : ℕ) (h1 : product = 2560) (h2 : L = 128)
  (h3 : H * L = product) : H = 20 := by {
  -- The proof goes here.
  sorry
}

end HCF_of_two_numbers_l141_141267


namespace question_l141_141005

variable (a : ℝ)

def condition_p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2 * x ≤ a^2 - a - 3

def condition_q (a : ℝ) : Prop := ∀ (x y : ℝ) , x > y → (5 - 2 * a)^x < (5 - 2 * a)^y

theorem question (h1 : condition_p a ∨ condition_q a)
                (h2 : ¬ (condition_p a ∧ condition_q a)) : a = 2 ∨ a ≥ 5 / 2 :=
sorry

end question_l141_141005


namespace maximum_value_40_l141_141498

theorem maximum_value_40 (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 10) :
  (a-b)^2 + (a-c)^2 + (a-d)^2 + (b-c)^2 + (b-d)^2 + (c-d)^2 ≤ 40 :=
sorry

end maximum_value_40_l141_141498


namespace mean_age_of_oldest_three_l141_141823

theorem mean_age_of_oldest_three (x : ℕ) (h : (x + (x + 1) + (x + 2)) / 3 = 6) : 
  (((x + 4) + (x + 5) + (x + 6)) / 3 = 10) := 
by
  sorry

end mean_age_of_oldest_three_l141_141823


namespace value_of_S_2016_l141_141816

variable (a d : ℤ)
variable (S : ℕ → ℤ)

-- Definitions of conditions
def a_1 := -2014
def sum_2012 := S 2012
def sum_10 := S 10
def S_n (n : ℕ) : ℤ := n * a_1 + (n * (n - 1) / 2) * d

-- Given conditions
axiom S_condition : (sum_2012 / 2012) - (sum_10 / 10) = 2002
axiom S_def : ∀ n : ℕ, S n = S_n n

-- The theorem to be proved
theorem value_of_S_2016 : S 2016 = 2016 := by
  sorry

end value_of_S_2016_l141_141816


namespace proposition_correctness_l141_141910

theorem proposition_correctness :
  (∀ x : ℝ, (|x-1| < 2) → (x < 3)) ∧
  (∀ (P Q : Prop), (Q → ¬ P) → (P → ¬ Q)) :=
by 
sorry

end proposition_correctness_l141_141910


namespace work_days_l141_141745

theorem work_days (p_can : ℕ → ℝ) (q_can : ℕ → ℝ) (together_can: ℕ → ℝ) :
  (together_can 6 = 1) ∧ (q_can 10 = 1) → (1 / (p_can x) + 1 / (q_can 10) = 1 / (together_can 6)) → (x = 15) :=
by
  sorry

end work_days_l141_141745


namespace mike_books_before_yard_sale_l141_141225

-- Problem definitions based on conditions
def books_bought_at_yard_sale : ℕ := 21
def books_now_in_library : ℕ := 56
def books_before_yard_sale := books_now_in_library - books_bought_at_yard_sale

-- Theorem to prove the equivalent proof problem
theorem mike_books_before_yard_sale : books_before_yard_sale = 35 := by
  sorry

end mike_books_before_yard_sale_l141_141225


namespace number_of_occupied_cars_l141_141563

theorem number_of_occupied_cars (k : ℕ) (x y : ℕ) :
  18 * k / 9 = 2 * k → 
  3 * x + 2 * y = 12 → 
  x + y ≤ 18 → 
  18 - x - y = 13 :=
by sorry

end number_of_occupied_cars_l141_141563


namespace diameter_of_circumscribed_circle_l141_141612

noncomputable def right_triangle_circumcircle_diameter (a b : ℕ) : ℕ :=
  let hypotenuse := (a * a + b * b).sqrt
  if hypotenuse = max a b then hypotenuse else 2 * max a b

theorem diameter_of_circumscribed_circle
  (a b : ℕ)
  (h : a = 16 ∨ b = 16)
  (h1 : a = 12 ∨ b = 12) :
  right_triangle_circumcircle_diameter a b = 16 ∨ right_triangle_circumcircle_diameter a b = 20 :=
by
  -- The proof goes here.
  sorry

end diameter_of_circumscribed_circle_l141_141612


namespace algebraic_expression_value_l141_141128

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x - 3 = 0) :
  (x - 1)^2 - x * (x - 3) + (x + 1) * (x - 1) = 3 :=
by
  sorry

end algebraic_expression_value_l141_141128


namespace proof_smallest_integer_proof_sum_of_integers_l141_141096

def smallest_integer (n : Int) : Prop :=
  ∃ (a b c d e : Int), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧ e = n + 8 ∧ a + e = 204 ∧ n = 98

def sum_of_integers (n : Int) : Prop :=
  ∃ (a b c d e : Int), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧ e = n + 8 ∧ a + e = 204 ∧ a + b + c + d + e = 510

theorem proof_smallest_integer : ∃ n : Int, smallest_integer n := by
  sorry

theorem proof_sum_of_integers : ∃ n : Int, sum_of_integers n := by
  sorry

end proof_smallest_integer_proof_sum_of_integers_l141_141096


namespace remainder_when_divided_by_6_l141_141929

theorem remainder_when_divided_by_6 (a : ℕ) (h1 : a % 2 = 1) (h2 : a % 3 = 2) : a % 6 = 5 :=
sorry

end remainder_when_divided_by_6_l141_141929


namespace calc_x6_plus_inv_x6_l141_141001

theorem calc_x6_plus_inv_x6 (x : ℝ) (hx : x + (1 / x) = 7) : x^6 + (1 / x^6) = 103682 := by
  sorry

end calc_x6_plus_inv_x6_l141_141001


namespace Tony_can_add_4_pairs_of_underwear_l141_141796

-- Define relevant variables and conditions
def max_weight : ℕ := 50
def w_socks : ℕ := 2
def w_underwear : ℕ := 4
def w_shirt : ℕ := 5
def w_shorts : ℕ := 8
def w_pants : ℕ := 10

def pants : ℕ := 1
def shirts : ℕ := 2
def shorts : ℕ := 1
def socks : ℕ := 3

def total_weight (pants shirts shorts socks : ℕ) : ℕ :=
  pants * w_pants + shirts * w_shirt + shorts * w_shorts + socks * w_socks

def remaining_weight : ℕ :=
  max_weight - total_weight pants shirts shorts socks

def additional_pairs_of_underwear_cannot_exceed : ℕ :=
  remaining_weight / w_underwear

-- Problem statement in Lean
theorem Tony_can_add_4_pairs_of_underwear :
  additional_pairs_of_underwear_cannot_exceed = 4 :=
  sorry

end Tony_can_add_4_pairs_of_underwear_l141_141796


namespace Kelly_baking_powder_difference_l141_141887

theorem Kelly_baking_powder_difference : 0.4 - 0.3 = 0.1 :=
by 
  -- sorry is a placeholder for a proof
  sorry

end Kelly_baking_powder_difference_l141_141887


namespace determine_placemat_length_l141_141429

theorem determine_placemat_length :
  ∃ (y : ℝ), ∀ (r : ℝ), r = 5 →
  (∀ (n : ℕ), n = 8 →
  (∀ (w : ℝ), w = 1 →
  y = 10 * Real.sin (5 * Real.pi / 16))) :=
by
  sorry

end determine_placemat_length_l141_141429


namespace john_has_leftover_bulbs_l141_141402

-- Definitions of the problem statements
def initial_bulbs : ℕ := 40
def used_bulbs : ℕ := 16
def remaining_bulbs_after_use : ℕ := initial_bulbs - used_bulbs
def given_to_friend : ℕ := remaining_bulbs_after_use / 2

-- Statement to prove
theorem john_has_leftover_bulbs :
  remaining_bulbs_after_use - given_to_friend = 12 :=
by
  sorry

end john_has_leftover_bulbs_l141_141402


namespace returned_books_percentage_is_correct_l141_141188

-- This function takes initial_books, end_books, and loaned_books and computes the percentage of books returned.
noncomputable def percent_books_returned (initial_books : ℕ) (end_books : ℕ) (loaned_books : ℕ) : ℚ :=
  let books_out_on_loan := initial_books - end_books
  let books_returned := loaned_books - books_out_on_loan
  (books_returned : ℚ) / (loaned_books : ℚ) * 100

-- The main theorem that states the percentage of books returned is 70%
theorem returned_books_percentage_is_correct :
  percent_books_returned 75 57 60 = 70 := by
  sorry

end returned_books_percentage_is_correct_l141_141188


namespace smallest_integer_proof_l141_141509

def smallest_integer_condition (n : ℤ) : Prop := n^2 - 15 * n + 56 ≤ 0

theorem smallest_integer_proof :
  ∃ n : ℤ, smallest_integer_condition n ∧ ∀ m : ℤ, smallest_integer_condition m → n ≤ m :=
sorry

end smallest_integer_proof_l141_141509


namespace geometric_ratio_l141_141670

theorem geometric_ratio (a₁ q : ℝ) (h₀ : a₁ ≠ 0) (h₁ : a₁ + a₁ * q + a₁ * q^2 = 3 * a₁) : q = -2 ∨ q = 1 :=
by
  sorry

end geometric_ratio_l141_141670


namespace aarti_work_multiple_l141_141291

-- Aarti can do a piece of work in 5 days
def days_per_unit_work := 5

-- It takes her 15 days to complete the certain multiple of work
def days_for_multiple_work := 15

-- Prove the ratio of the days for multiple work to the days per unit work equals 3
theorem aarti_work_multiple :
  days_for_multiple_work / days_per_unit_work = 3 :=
sorry

end aarti_work_multiple_l141_141291


namespace number_of_true_propositions_is_one_l141_141592

-- Define propositions
def prop1 (a b c : ℝ) : Prop := a > b ∧ c ≠ 0 → a * c > b * c
def prop2 (a b c : ℝ) : Prop := a > b → a * c^2 > b * c^2
def prop3 (a b c : ℝ) : Prop := a * c^2 > b * c^2 → a > b
def prop4 (a b : ℝ) : Prop := a > b → (1 / a) < (1 / b)
def prop5 (a b c d : ℝ) : Prop := a > b ∧ b > 0 ∧ c > d → a * c > b * d

-- The main theorem stating the number of true propositions
theorem number_of_true_propositions_is_one (a b c d : ℝ) :
  (prop3 a b c) ∧ (¬ prop1 a b c) ∧ (¬ prop2 a b c) ∧ (¬ prop4 a b) ∧ (¬ prop5 a b c d) :=
by
  sorry

end number_of_true_propositions_is_one_l141_141592


namespace problem_l141_141355

-- Definitions for angles A, B, C and sides a, b, c of a triangle.
variables {A B C : ℝ} {a b c : ℝ}
-- Given condition
variables (h : a = b * Real.cos C + c * Real.sin B)

-- Triangle inequality and angle conditions
variables (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
variables (suma : A + B + C = Real.pi)

-- Goal: to prove that under the given condition, angle B is π/4
theorem problem : B = Real.pi / 4 :=
by {
  sorry
}

end problem_l141_141355


namespace circle_equation_l141_141047

theorem circle_equation (x y : ℝ) (h_eq : x = 0) (k_eq : y = -2) (r_eq : y = 4) :
  (x - 0)^2 + (y - (-2))^2 = 16 := 
by
  sorry

end circle_equation_l141_141047


namespace arithmetic_sequence_nth_term_l141_141995

theorem arithmetic_sequence_nth_term (S : ℕ → ℕ) (h : ∀ n, S n = 5 * n + 4 * n^2) (r : ℕ) : 
  S r - S (r - 1) = 8 * r + 1 := 
by
  sorry

end arithmetic_sequence_nth_term_l141_141995


namespace log_identity_l141_141874

theorem log_identity
  (x : ℝ)
  (h1 : x < 1)
  (h2 : (Real.log x / Real.log 10)^2 - Real.log (x^4) / Real.log 10 = 100) :
  (Real.log x / Real.log 10)^3 - Real.log (x^5) / Real.log 10 = -114 + Real.sqrt 104 := 
by
  sorry

end log_identity_l141_141874


namespace find_b_minus_a_l141_141262

theorem find_b_minus_a (a b : ℤ) (h1 : a * b = 2 * (a + b) + 11) (h2 : b = 7) : b - a = 2 :=
by sorry

end find_b_minus_a_l141_141262


namespace metals_inductive_reasoning_l141_141730

def conducts_electricity (metal : String) : Prop :=
  metal = "Gold" ∨ metal = "Silver" ∨ metal = "Copper" ∨ metal = "Iron"

def all_metals_conduct_electricity (metals : List String) : Prop :=
  ∀ metal, metal ∈ metals → conducts_electricity metal

theorem metals_inductive_reasoning 
  (h1 : conducts_electricity "Gold")
  (h2 : conducts_electricity "Silver")
  (h3 : conducts_electricity "Copper")
  (h4 : conducts_electricity "Iron") :
  (all_metals_conduct_electricity ["Gold", "Silver", "Copper", "Iron"] → 
  all_metals_conduct_electricity ["All metals"]) :=
  sorry -- Proof skipped, as per instructions.

end metals_inductive_reasoning_l141_141730


namespace rational_quotient_of_arith_geo_subseq_l141_141051

theorem rational_quotient_of_arith_geo_subseq (A d : ℝ) (h_d_nonzero : d ≠ 0)
    (h_contains_geo : ∃ (q : ℝ) (k m n : ℕ), q ≠ 1 ∧ q ≠ 0 ∧ 
        A + k * d = (A + m * d) * q ∧ A + m * d = (A + n * d) * q)
    : ∃ (r : ℚ), A / d = r :=
  sorry

end rational_quotient_of_arith_geo_subseq_l141_141051


namespace ice_cream_flavors_l141_141015

theorem ice_cream_flavors : (Nat.choose 8 3) = 56 := 
by {
    sorry
}

end ice_cream_flavors_l141_141015


namespace system_solve_l141_141651

theorem system_solve (x y : ℚ) (h1 : 2 * x + y = 3) (h2 : 3 * x - 2 * y = 12) : x + y = 3 / 7 :=
by
  -- The proof will go here, but we skip it for now.
  sorry

end system_solve_l141_141651


namespace min_a_for_decreasing_f_l141_141568

theorem min_a_for_decreasing_f {a : ℝ} :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 1 - a / (2 * Real.sqrt x) ≤ 0) →
  a ≥ 4 :=
sorry

end min_a_for_decreasing_f_l141_141568


namespace arithmetic_sum_l141_141953

variable {a : ℕ → ℝ}

def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sum :
  is_arithmetic_seq a →
  a 5 + a 6 + a 7 = 15 →
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by
  intros
  sorry

end arithmetic_sum_l141_141953


namespace no_nat_n_exists_l141_141671

theorem no_nat_n_exists (n : ℕ) : ¬ ∃ n, ∃ k, n ^ 2012 - 1 = 2 ^ k := by
  sorry

end no_nat_n_exists_l141_141671


namespace v3_value_at_2_l141_141762

def f (x : ℝ) : ℝ :=
  x^6 - 12 * x^5 + 60 * x^4 - 160 * x^3 + 240 * x^2 - 192 * x + 64

def v3 (x : ℝ) : ℝ :=
  ((x - 12) * x + 60) * x - 160

theorem v3_value_at_2 :
  v3 2 = -80 :=
by
  sorry

end v3_value_at_2_l141_141762


namespace least_grapes_in_heap_l141_141201

theorem least_grapes_in_heap :
  ∃ n : ℕ, (n % 19 = 1) ∧ (n % 23 = 1) ∧ (n % 29 = 1) ∧ n = 12209 :=
by
  sorry

end least_grapes_in_heap_l141_141201


namespace hamburgers_leftover_l141_141095

-- Define the number of hamburgers made and served
def hamburgers_made : ℕ := 9
def hamburgers_served : ℕ := 3

-- Prove the number of leftover hamburgers
theorem hamburgers_leftover : hamburgers_made - hamburgers_served = 6 := 
by
  sorry

end hamburgers_leftover_l141_141095


namespace compare_magnitudes_proof_l141_141438

noncomputable def compare_magnitudes (a b c : ℝ) (ha : 0 < a) (hbc : b * c > a^2) (heq : a^2 - 2 * a * b + c^2 = 0) : Prop :=
  b > c ∧ c > a ∧ b > a

theorem compare_magnitudes_proof (a b c : ℝ) (ha : 0 < a) (hbc : b * c > a^2) (heq : a^2 - 2 * a * b + c^2 = 0) :
  compare_magnitudes a b c ha hbc heq :=
sorry

end compare_magnitudes_proof_l141_141438


namespace maximum_small_circles_l141_141690

-- Definitions for small circle radius, large circle radius, and the maximum number n.
def smallCircleRadius : ℝ := 1
def largeCircleRadius : ℝ := 11

-- Function to check if small circles can be placed without overlapping
def canPlaceCircles (n : ℕ) : Prop := n * 2 < 2 * Real.pi * (largeCircleRadius - smallCircleRadius)

theorem maximum_small_circles : ∀ n : ℕ, canPlaceCircles n → n ≤ 31 := by
  sorry

end maximum_small_circles_l141_141690


namespace trapezoid_two_heights_l141_141786

-- Define trivially what a trapezoid is, in terms of having two parallel sides.
structure Trapezoid :=
(base1 base2 : ℝ)
(height1 height2 : ℝ)
(has_two_heights : height1 = height2)

theorem trapezoid_two_heights (T : Trapezoid) : ∃ h1 h2 : ℝ, h1 = h2 :=
by
  use T.height1
  use T.height2
  exact T.has_two_heights

end trapezoid_two_heights_l141_141786


namespace sum_of_reciprocals_of_squares_l141_141174

theorem sum_of_reciprocals_of_squares (x y : ℕ) (hxy : x * y = 17) : 
  1 / (x:ℚ)^2 + 1 / (y:ℚ)^2 = 290 / 289 := 
by
  sorry

end sum_of_reciprocals_of_squares_l141_141174


namespace tan_alpha_solution_l141_141297

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l141_141297


namespace max_possible_value_l141_141928

theorem max_possible_value (P Q : ℤ) (hP : P * P ≤ 729 ∧ 729 ≤ -P * P * P)
  (hQ : Q * Q ≤ 729 ∧ 729 ≤ -Q * Q * Q) :
  10 * (P - Q) = 180 :=
by
  sorry

end max_possible_value_l141_141928


namespace polynomial_coefficients_correct_l141_141998

-- Define the polynomial equation
def polynomial_equation (x a b c d : ℝ) : Prop :=
  x^4 + 4*x^3 + 3*x^2 + 2*x + 1 = (x+1)^4 + a*(x+1)^3 + b*(x+1)^2 + c*(x+1) + d

-- The problem to prove
theorem polynomial_coefficients_correct :
  ∀ x : ℝ, polynomial_equation x 0 (-3) 4 (-1) :=
by
  intro x
  unfold polynomial_equation
  sorry

end polynomial_coefficients_correct_l141_141998


namespace stacy_grew_more_l141_141842

variable (initial_height_stacy current_height_stacy brother_growth stacy_growth_more : ℕ)

-- Conditions
def stacy_initial_height : initial_height_stacy = 50 := by sorry
def stacy_current_height : current_height_stacy = 57 := by sorry
def brother_growth_last_year : brother_growth = 1 := by sorry

-- Compute Stacy's growth
def stacy_growth : ℕ := current_height_stacy - initial_height_stacy

-- Prove the difference in growth
theorem stacy_grew_more :
  stacy_growth - brother_growth = stacy_growth_more → stacy_growth_more = 6 := 
by sorry

end stacy_grew_more_l141_141842


namespace road_length_kopatych_to_losyash_l141_141430

variable (T Krosh_dist Yozhik_dist : ℕ)
variable (d_k d_y r_k r_y : ℕ)

theorem road_length_kopatych_to_losyash : 
    (d_k = 20) → (d_y = 16) → (r_k = 30) → (r_y = 60) → 
    (Krosh_dist = 5 * T / 9) → (Yozhik_dist = 4 * T / 9) → 
    (T = Krosh_dist + r_k) →
    (T = Yozhik_dist + r_y) → 
    (T = 180) :=
by
  intros
  sorry

end road_length_kopatych_to_losyash_l141_141430


namespace complement_of_M_in_U_l141_141564

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}

theorem complement_of_M_in_U : U \ M = {2, 3, 5} := by
  sorry

end complement_of_M_in_U_l141_141564


namespace acute_angle_of_parallel_vectors_l141_141212
open Real

theorem acute_angle_of_parallel_vectors (α : ℝ) (h₁ : abs (α * π / 180) < π / 2) :
  let a := (3 / 2, sin (α * π / 180))
  let b := (sin (α * π / 180), 1 / 6) 
  a.1 * b.2 = a.2 * b.1 → α = 30 :=
by
  sorry

end acute_angle_of_parallel_vectors_l141_141212


namespace fair_split_adjustment_l141_141196

theorem fair_split_adjustment
    (A B : ℝ)
    (h : A < B)
    (d1 d2 d3 : ℝ)
    (h1 : d1 = 120)
    (h2 : d2 = 150)
    (h3 : d3 = 180)
    (bernardo_pays_twice : ∀ D, (2 : ℝ) * D = d1 + d2 + d3) :
    (B - A) / 2 - 75 = ((d1 + d2 + d3) - 450) / 2 - (A - (d1 + d2 + d3) / 3) :=
by
  sorry

end fair_split_adjustment_l141_141196


namespace find_y_coordinate_l141_141245

noncomputable def y_coordinate_of_point_on_line : ℝ :=
  let x1 := 10
  let y1 := 3
  let x2 := 4
  let y2 := 0
  let x := -2
  let m := (y1 - y2) / (x1 - x2)
  let b := y1 - m * x1
  m * x + b

theorem find_y_coordinate :
  (y_coordinate_of_point_on_line = -3) :=
by
  sorry

end find_y_coordinate_l141_141245


namespace kate_needs_more_money_for_trip_l141_141497

theorem kate_needs_more_money_for_trip:
  let kate_money_base6 := 3 * 6^3 + 2 * 6^2 + 4 * 6^1 + 2 * 6^0
  let ticket_cost := 1000
  kate_money_base6 - ticket_cost = -254 :=
by
  -- Proving the theorem, steps will go here.
  sorry

end kate_needs_more_money_for_trip_l141_141497


namespace max_value_of_product_l141_141857

theorem max_value_of_product (x y z w : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w) (h_sum : x + y + z + w = 1) : 
  x^2 * y^2 * z^2 * w ≤ 64 / 823543 :=
by
  sorry

end max_value_of_product_l141_141857


namespace find_second_number_l141_141958

theorem find_second_number (x : ℝ) 
    (h : (14 + x + 53) / 3 = (21 + 47 + 22) / 3 + 3) : 
    x = 32 := 
by 
    sorry

end find_second_number_l141_141958


namespace ratio_albert_betty_l141_141618

theorem ratio_albert_betty (A M B : ℕ) (h1 : A = 2 * M) (h2 : M = A - 10) (h3 : B = 5) :
  A / B = 4 :=
by
  -- the proof goes here
  sorry

end ratio_albert_betty_l141_141618


namespace sqrt_21_between_4_and_5_l141_141131

theorem sqrt_21_between_4_and_5 : 4 < Real.sqrt 21 ∧ Real.sqrt 21 < 5 := 
by 
  sorry

end sqrt_21_between_4_and_5_l141_141131


namespace prime_sol_is_7_l141_141418

theorem prime_sol_is_7 (p : ℕ) (x y : ℕ) (hp : Nat.Prime p) 
  (hx : p + 1 = 2 * x^2) (hy : p^2 + 1 = 2 * y^2) : 
  p = 7 := 
  sorry

end prime_sol_is_7_l141_141418


namespace determine_jug_capacity_l141_141985

variable (jug_capacity : Nat)
variable (small_jug : Nat)

theorem determine_jug_capacity (h1 : jug_capacity = 5) (h2 : small_jug = 3 ∨ small_jug = 4):
  (∃ overflow_remains : Nat, 
    (overflow_remains = jug_capacity ∧ small_jug = 4) ∨ 
    (¬(overflow_remains = jug_capacity) ∧ small_jug = 3)) :=
by
  sorry

end determine_jug_capacity_l141_141985


namespace f_fe_eq_neg1_f_x_gt_neg1_solution_l141_141452

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (1 / x)
  else if x < 0 then 1 / x
  else 0  -- handle the case for x = 0 explicitly if needed

theorem f_fe_eq_neg1 : 
  f (f (Real.exp 1)) = -1 := 
by
  -- proof to be filled in
  sorry

theorem f_x_gt_neg1_solution :
  {x : ℝ | f x > -1} = {x : ℝ | (x < -1) ∨ (0 < x ∧ x < Real.exp 1)} :=
by
  -- proof to be filled in
  sorry

end f_fe_eq_neg1_f_x_gt_neg1_solution_l141_141452


namespace fraction_division_l141_141548

theorem fraction_division :
  (3 / 4) / (5 / 8) = 6 / 5 :=
by
  sorry

end fraction_division_l141_141548


namespace min_value_abs_sum_exists_min_value_abs_sum_l141_141250

theorem min_value_abs_sum (x : ℝ) : |x - 1| + |x - 4| ≥ 3 :=
by sorry

theorem exists_min_value_abs_sum : ∃ x : ℝ, |x - 1| + |x - 4| = 3 :=
by sorry

end min_value_abs_sum_exists_min_value_abs_sum_l141_141250


namespace solve_inequality_system_l141_141880

theorem solve_inequality_system (x : ℝ) :
  (x - 1 < 2 * x + 1) ∧ ((2 * x - 5) / 3 ≤ 1) → (-2 < x ∧ x ≤ 4) :=
by
  intro cond
  sorry

end solve_inequality_system_l141_141880


namespace num_diamonds_in_G6_l141_141617

noncomputable def triangular_number (k : ℕ) : ℕ :=
  (k * (k + 1)) / 2

noncomputable def total_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 4 * (Finset.sum (Finset.range (n - 1)) (λ k => triangular_number (k + 1)))

theorem num_diamonds_in_G6 :
  total_diamonds 6 = 141 := by
  -- This will be proven
  sorry

end num_diamonds_in_G6_l141_141617


namespace square_side_length_l141_141868

theorem square_side_length (s : ℝ) (h : s^2 = 3 * 4 * s) : s = 12 :=
by
  sorry

end square_side_length_l141_141868


namespace lionsAfterOneYear_l141_141031

-- Definitions based on problem conditions
def initialLions : Nat := 100
def birthRate : Nat := 5
def deathRate : Nat := 1
def monthsInYear : Nat := 12

-- Theorem statement
theorem lionsAfterOneYear :
  initialLions + birthRate * monthsInYear - deathRate * monthsInYear = 148 :=
by
  sorry

end lionsAfterOneYear_l141_141031


namespace charlene_initial_necklaces_l141_141466

-- Definitions for the conditions.
def necklaces_sold : ℕ := 16
def necklaces_giveaway : ℕ := 18
def necklaces_left : ℕ := 26

-- Statement to prove that the initial number of necklaces is 60.
theorem charlene_initial_necklaces : necklaces_sold + necklaces_giveaway + necklaces_left = 60 := by
  sorry

end charlene_initial_necklaces_l141_141466


namespace ab_value_l141_141391

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := 
by 
  sorry

end ab_value_l141_141391


namespace mass_of_man_l141_141183

theorem mass_of_man (L B : ℝ) (h : ℝ) (ρ : ℝ) (V : ℝ) : L = 8 ∧ B = 3 ∧ h = 0.01 ∧ ρ = 1 ∧ V = L * 100 * B * 100 * h → V / 1000 = 240 :=
by
  sorry

end mass_of_man_l141_141183


namespace hyperbola_equation_through_point_l141_141074

theorem hyperbola_equation_through_point
  (hyp_passes_through : ∀ (x y : ℝ), (x, y) = (1, 1) → ∃ (a b t : ℝ), (y^2 / a^2 - x^2 / b^2 = t))
  (asymptotes : ∀ (x y : ℝ), (y / x = Real.sqrt 2 ∨ y / x = -Real.sqrt 2) → ∃ (a b t : ℝ), (a = b * Real.sqrt 2)) :
  ∃ (a b t : ℝ), (2 * (1:ℝ)^2 - (1:ℝ)^2 = 1) :=
by
  sorry

end hyperbola_equation_through_point_l141_141074


namespace find_fourth_score_l141_141103

theorem find_fourth_score
  (a b c : ℕ) (d : ℕ)
  (ha : a = 70) (hb : b = 80) (hc : c = 90)
  (average_eq : (a + b + c + d) / 4 = 70) :
  d = 40 := 
sorry

end find_fourth_score_l141_141103


namespace flour_needed_for_one_loaf_l141_141834

-- Define the conditions
def flour_needed_for_two_loaves : ℚ := 5 -- cups of flour needed for two loaves

-- Define the theorem to prove
theorem flour_needed_for_one_loaf : flour_needed_for_two_loaves / 2 = 2.5 :=
by 
  -- Skip the proof.
  sorry

end flour_needed_for_one_loaf_l141_141834


namespace sequence_a5_l141_141504

theorem sequence_a5 : 
    ∃ (a : ℕ → ℚ), 
    a 1 = 1 / 3 ∧ 
    (∀ (n : ℕ), n ≥ 2 → a n = (-1 : ℚ)^n * 2 * a (n - 1)) ∧ 
    a 5 = -16 / 3 := 
sorry

end sequence_a5_l141_141504


namespace total_amount_paid_l141_141702

theorem total_amount_paid (grapes_kg mangoes_kg rate_grapes rate_mangoes : ℕ) 
    (h1 : grapes_kg = 8) (h2 : mangoes_kg = 8) 
    (h3 : rate_grapes = 70) (h4 : rate_mangoes = 55) : 
    (grapes_kg * rate_grapes + mangoes_kg * rate_mangoes) = 1000 :=
by
  sorry

end total_amount_paid_l141_141702


namespace dot_product_property_l141_141437

-- Definitions based on conditions
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)
def scalar_mult (c : ℝ) (v : ℝ × ℝ) := (c * v.1, c * v.2)
def vec_add (v1 v2 : ℝ × ℝ) := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

-- Required property
theorem dot_product_property : dot_product (vec_add (scalar_mult 2 vec_a) vec_b) vec_a = 6 :=
by sorry

end dot_product_property_l141_141437


namespace factor_81_minus_36x4_l141_141280

theorem factor_81_minus_36x4 (x : ℝ) : 
    81 - 36 * x^4 = 9 * (Real.sqrt 3 - Real.sqrt 2 * x) * (Real.sqrt 3 + Real.sqrt 2 * x) * (3 + 2 * x^2) :=
sorry

end factor_81_minus_36x4_l141_141280


namespace sam_has_two_nickels_l141_141969

def average_value_initial (total_value : ℕ) (total_coins : ℕ) := total_value / total_coins = 15
def average_value_with_extra_dime (total_value : ℕ) (total_coins : ℕ) := (total_value + 10) / (total_coins + 1) = 16

theorem sam_has_two_nickels (total_value total_coins : ℕ) (h1 : average_value_initial total_value total_coins) (h2 : average_value_with_extra_dime total_value total_coins) : 
∃ (nickels : ℕ), nickels = 2 := 
by 
  sorry

end sam_has_two_nickels_l141_141969


namespace total_wages_of_12_men_l141_141519

variable {M W B x y : Nat}
variable {total_wages : Nat}

-- Condition 1: 12 men do the work equivalent to W women
axiom work_equivalent_1 : 12 * M = W

-- Condition 2: 12 men do the work equivalent to 20 boys
axiom work_equivalent_2 : 12 * M = 20 * B

-- Condition 3: All together earn Rs. 450
axiom total_earnings : (12 * M) + (x * (12 * M / W)) + (y * (12 * M / (20 * B))) = 450

-- The theorem to prove
theorem total_wages_of_12_men : total_wages = 12 * M → false :=
by sorry

end total_wages_of_12_men_l141_141519


namespace johan_painted_green_fraction_l141_141068

theorem johan_painted_green_fraction :
  let total_rooms := 10
  let walls_per_room := 8
  let purple_walls := 32
  let purple_rooms := purple_walls / walls_per_room
  let green_rooms := total_rooms - purple_rooms
  (green_rooms : ℚ) / total_rooms = 3 / 5 := by
  sorry

end johan_painted_green_fraction_l141_141068


namespace min_a_plus_b_l141_141744

theorem min_a_plus_b (a b : ℤ) (h : a * b = 144) : a + b ≥ -145 := sorry

end min_a_plus_b_l141_141744


namespace length_of_ribbon_l141_141889

theorem length_of_ribbon (perimeter : ℝ) (sides : ℕ) (h1 : perimeter = 42) (h2 : sides = 6) : (perimeter / sides) = 7 :=
by {
  sorry
}

end length_of_ribbon_l141_141889


namespace fractional_pizza_eaten_after_six_trips_l141_141037

def pizza_eaten : ℚ := (1/3) * (1 - (2/3)^6) / (1 - 2/3)

theorem fractional_pizza_eaten_after_six_trips : pizza_eaten = 665 / 729 :=
by
  -- proof will go here
  sorry

end fractional_pizza_eaten_after_six_trips_l141_141037


namespace increasing_interval_l141_141944

def my_function (x : ℝ) : ℝ := -(x - 3) * |x|

theorem increasing_interval : ∀ x y : ℝ, 0 ≤ x → x ≤ y → my_function x ≤ my_function y :=
by
  sorry

end increasing_interval_l141_141944


namespace tweets_when_hungry_l141_141884

theorem tweets_when_hungry (H : ℕ) : 
  (18 * 20) + (H * 20) + (45 * 20) = 1340 → H = 4 := by
  sorry

end tweets_when_hungry_l141_141884


namespace carter_drum_sticks_l141_141799

def sets_per_show (used : ℕ) (tossed : ℕ) : ℕ := used + tossed

def total_sets (sets_per_show : ℕ) (num_shows : ℕ) : ℕ := sets_per_show * num_shows

theorem carter_drum_sticks :
  sets_per_show 8 10 * 45 = 810 :=
by
  sorry

end carter_drum_sticks_l141_141799


namespace cost_of_remaining_shirt_l141_141832

theorem cost_of_remaining_shirt :
  ∀ (shirts total_cost cost_per_shirt remaining_shirt_cost : ℕ),
  shirts = 5 →
  total_cost = 85 →
  cost_per_shirt = 15 →
  (3 * cost_per_shirt) + (2 * remaining_shirt_cost) = total_cost →
  remaining_shirt_cost = 20 :=
by
  intros shirts total_cost cost_per_shirt remaining_shirt_cost
  intros h_shirts h_total h_cost_per_shirt h_equation
  sorry

end cost_of_remaining_shirt_l141_141832


namespace unique_not_in_range_of_g_l141_141265

noncomputable def g (m n p q : ℝ) (x : ℝ) : ℝ := (m * x + n) / (p * x + q)

theorem unique_not_in_range_of_g (m n p q : ℝ) (hne1 : m ≠ 0) (hne2 : n ≠ 0) (hne3 : p ≠ 0) (hne4 : q ≠ 0)
  (h₁ : g m n p q 23 = 23) (h₂ : g m n p q 53 = 53) (h₃ : ∀ (x : ℝ), x ≠ -q / p → g m n p q (g m n p q x) = x) :
  ∃! x : ℝ, ¬ ∃ y : ℝ, g m n p q y = x ∧ x = -38 :=
sorry

end unique_not_in_range_of_g_l141_141265


namespace range_of_a_l141_141440

theorem range_of_a {
  a : ℝ
} :
  (∀ x ∈ Set.Ici (2 : ℝ), (x^2 + (2 - a) * x + 4 - 2 * a) > 0) ↔ a < 3 :=
by
  sorry

end range_of_a_l141_141440


namespace sum_of_coefficients_l141_141754

theorem sum_of_coefficients (b_0 b_1 b_2 b_3 b_4 b_5 b_6 : ℝ) :
  (5 * 1 - 2)^6 = b_6 * 1^6 + b_5 * 1^5 + b_4 * 1^4 + b_3 * 1^3 + b_2 * 1^2 + b_1 * 1 + b_0
  → b_0 + b_1 + b_2 + b_3 + b_4 + b_5 + b_6 = 729 := by
  sorry

end sum_of_coefficients_l141_141754


namespace subproblem1_l141_141807

theorem subproblem1 (a : ℝ) : a^3 * a + (2 * a^2)^2 = 5 * a^4 := 
by sorry

end subproblem1_l141_141807


namespace f_of_3_eq_11_l141_141496

theorem f_of_3_eq_11 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1 / x) = x^2 + 1 / x^2) : f 3 = 11 :=
by
  sorry

end f_of_3_eq_11_l141_141496


namespace evaluate_expression_l141_141055

/-- Given conditions: -/
def a : ℕ := 3998
def b : ℕ := 3999

theorem evaluate_expression :
  b^3 - 2 * a * b^2 - 2 * a^2 * b + (b - 2)^3 = 95806315 :=
  sorry

end evaluate_expression_l141_141055


namespace find_d_l141_141389

-- Define AP terms as S_n = a + (n-1)d, sum of first 10 terms, and difference expression
def arithmetic_progression (S : ℕ → ℕ) (a d : ℕ) : Prop :=
  ∀ n, S n = a + (n - 1) * d

def sum_first_ten (S : ℕ → ℕ) : Prop :=
  (S 1) + (S 2) + (S 3) + (S 4) + (S 5) + (S 6) + (S 7) + (S 8) + (S 9) + (S 10) = 55

def difference_expression (S : ℕ → ℕ) (d : ℕ) : Prop :=
  (S 10 - S 8) + (S 9 - S 7) + (S 8 - S 6) + (S 7 - S 5) + (S 6 - S 4) +
  (S 5 - S 3) + (S 4 - S 2) + (S 3 - S 1) = d

theorem find_d : ∃ (d : ℕ) (S : ℕ → ℕ) (a : ℕ), 
  (∀ n, S n = a + (n - 1) * d) ∧ 
  (S 1) + (S 2) + (S 3) + (S 4) + (S 5) + (S 6) + (S 7) + (S 8) + (S 9) + (S 10) = 55 ∧
  (S 10 - S 8) + (S 9 - S 7) + (S 8 - S 6) + (S 7 - S 5) + (S 6 - S 4) +
  (S 5 - S 3) + (S 4 - S 2) + (S 3 - S 1) = 16 :=
by
  sorry  -- proof is not required

end find_d_l141_141389


namespace train_speed_l141_141945

theorem train_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 400) (h_time : time = 40) : distance / time = 10 := by
  rw [h_distance, h_time]
  norm_num

end train_speed_l141_141945


namespace inequality_comparison_l141_141738

theorem inequality_comparison 
  (a b : ℝ) (x y : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : x^2 / a^2 + y^2 / b^2 ≤ 1) :
  a^2 + b^2 ≥ (x + y)^2 :=
sorry

end inequality_comparison_l141_141738


namespace part1_zero_of_f_part2_a_range_l141_141295

-- Define the given function f
def f (x a b : ℝ) : ℝ := (x - a) * |x| + b

-- Define the problem statement for Part 1
theorem part1_zero_of_f :
  ∀ (x : ℝ),
    f x 2 3 = 0 ↔ x = -1 := 
by
  sorry

-- Define the problem statement for Part 2
theorem part2_a_range :
  ∀ (a : ℝ),
    (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → f x a (-2) < 0) ↔ a > -1 :=
by
  sorry

end part1_zero_of_f_part2_a_range_l141_141295


namespace ronald_profit_fraction_l141_141681

theorem ronald_profit_fraction:
  let initial_units : ℕ := 200
  let total_investment : ℕ := 3000
  let selling_price_per_unit : ℕ := 20
  let total_selling_price := initial_units * selling_price_per_unit
  let total_profit := total_selling_price - total_investment
  (total_profit : ℚ) / total_investment = (1 : ℚ) / 3 :=
by
  -- here we will put the steps needed to prove the theorem.
  sorry

end ronald_profit_fraction_l141_141681


namespace classroom_students_count_l141_141404

theorem classroom_students_count (b g : ℕ) (hb : 3 * g = 5 * b) (hg : g = b + 4) : b + g = 16 :=
by
  sorry

end classroom_students_count_l141_141404


namespace real_root_uncertainty_l141_141687

noncomputable def f (x m : ℝ) : ℝ := m * x^2 - 2 * (m + 2) * x + m + 5
noncomputable def g (x m : ℝ) : ℝ := (m - 5) * x^2 - 2 * (m + 2) * x + m

theorem real_root_uncertainty (m : ℝ) :
  (∀ x : ℝ, f x m ≠ 0) → 
  (m ≤ 5 → ∃ x : ℝ, g x m = 0 ∧ ∀ y : ℝ, y ≠ x → g y m = 0) ∧
  (m > 5 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0) :=
sorry

end real_root_uncertainty_l141_141687


namespace smallest_number_is_1111_in_binary_l141_141947

theorem smallest_number_is_1111_in_binary :
  let a := 15   -- Decimal equivalent of 1111 in binary
  let b := 78   -- Decimal equivalent of 210 in base 6
  let c := 64   -- Decimal equivalent of 1000 in base 4
  let d := 65   -- Decimal equivalent of 101 in base 8
  a < b ∧ a < c ∧ a < d := 
by
  let a := 15
  let b := 78
  let c := 64
  let d := 65
  show a < b ∧ a < c ∧ a < d
  sorry

end smallest_number_is_1111_in_binary_l141_141947


namespace union_A_B_l141_141909

-- Define them as sets
def A : Set ℝ := {x | -3 < x ∧ x ≤ 2}
def B : Set ℝ := {x | -2 < x ∧ x ≤ 3}

-- Statement of the theorem
theorem union_A_B : A ∪ B = {x | -3 < x ∧ x ≤ 3} :=
by
  sorry

end union_A_B_l141_141909


namespace sakshi_days_l141_141781

theorem sakshi_days (Sakshi_efficiency Tanya_efficiency : ℝ) (Sakshi_days Tanya_days : ℝ) (h_efficiency : Tanya_efficiency = 1.25 * Sakshi_efficiency) (h_days : Tanya_days = 8) : Sakshi_days = 10 :=
by
  sorry

end sakshi_days_l141_141781


namespace competition_sequences_l141_141023

-- Define the problem conditions
def team_size : Nat := 7

-- Define the statement to prove
theorem competition_sequences :
  (Nat.choose (2 * team_size) team_size) = 3432 :=
by
  -- Proof will go here
  sorry

end competition_sequences_l141_141023


namespace red_bowling_balls_count_l141_141758

theorem red_bowling_balls_count (G R : ℕ) (h1 : G = R + 6) (h2 : R + G = 66) : R = 30 :=
by
  sorry

end red_bowling_balls_count_l141_141758


namespace designated_time_to_B_l141_141907

theorem designated_time_to_B (s v : ℝ) (x : ℝ) (V' : ℝ)
  (h1 : s / 2 = (x + 2) * V')
  (h2 : s / (2 * V') + 1 + s / (2 * (V' + v)) = x) :
  x = (v + Real.sqrt (9 * v ^ 2 + 6 * v * s)) / v :=
by
  sorry

end designated_time_to_B_l141_141907


namespace cos_diff_l141_141152

theorem cos_diff (x y : ℝ) (hx : x = Real.cos (20 * Real.pi / 180)) (hy : y = Real.cos (40 * Real.pi / 180)) :
  x - y = 1 / 2 :=
by sorry

end cos_diff_l141_141152


namespace four_digit_integer_l141_141890

theorem four_digit_integer (a b c d : ℕ) 
    (h1 : a + b + c + d = 16) 
    (h2 : b + c = 10) 
    (h3 : a - d = 2) 
    (h4 : (a - b + c - d) % 11 = 0) : 
    a = 4 ∧ b = 4 ∧ c = 6 ∧ d = 2 :=
sorry

end four_digit_integer_l141_141890


namespace distinct_integer_pairs_l141_141723

theorem distinct_integer_pairs :
  ∃ pairs : (Nat × Nat) → Prop,
  (∀ x y : Nat, pairs (x, y) → 0 < x ∧ x < y ∧ (8 * Real.sqrt 31 = Real.sqrt x + Real.sqrt y))
  ∧ (∃! p, pairs p) → (∃! q, pairs q) → (∃! r, pairs r) → true := sorry

end distinct_integer_pairs_l141_141723


namespace probability_all_same_color_is_correct_l141_141473

-- Definitions of quantities
def yellow_marbles := 3
def green_marbles := 7
def purple_marbles := 5
def total_marbles := yellow_marbles + green_marbles + purple_marbles

-- Calculation of drawing 4 marbles all the same color
def probability_all_yellow : ℚ := (yellow_marbles / total_marbles) * ((yellow_marbles - 1) / (total_marbles - 1)) * ((yellow_marbles - 2) / (total_marbles - 2)) * ((yellow_marbles - 3) / (total_marbles - 3))
def probability_all_green : ℚ := (green_marbles / total_marbles) * ((green_marbles - 1) / (total_marbles - 1)) * ((green_marbles - 2) / (total_marbles - 2)) * ((green_marbles - 3) / (total_marbles - 3))
def probability_all_purple : ℚ := (purple_marbles / total_marbles) * ((purple_marbles - 1) / (total_marbles - 1)) * ((purple_marbles - 2) / (total_marbles - 2)) * ((purple_marbles - 3) / (total_marbles - 3))

-- Total probability of drawing 4 marbles all the same color
def total_probability_same_color : ℚ := probability_all_yellow + probability_all_green + probability_all_purple

-- Theorem statement
theorem probability_all_same_color_is_correct : total_probability_same_color = 532 / 4095 :=
by
  sorry

end probability_all_same_color_is_correct_l141_141473


namespace solution_set_of_cx_sq_minus_bx_plus_a_l141_141835

theorem solution_set_of_cx_sq_minus_bx_plus_a (a b c : ℝ) (h1 : a < 0)
(h2 : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ 2 < x ∧ x < 3) :
  ∀ x : ℝ, cx^2 - bx + a > 0 ↔ -1/2 < x ∧ x < -1/3 :=
by
  sorry

end solution_set_of_cx_sq_minus_bx_plus_a_l141_141835


namespace intersection_complement_A_B_subset_A_C_l141_141394

-- Definition of sets A, B, and complements in terms of conditions
def setA : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def setB : Set ℝ := { x | 2 < x ∧ x < 10 }
def complement_A : Set ℝ := { x | x < 3 ∨ x ≥ 7 }

-- Proof Problem (1)
theorem intersection_complement_A_B :
  ((complement_A) ∩ setB) = { x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } := 
  sorry

-- Definition of set C 
def setC (a : ℝ) : Set ℝ := { x | x < a }
-- Proof Problem (2)
theorem subset_A_C {a : ℝ} (h : setA ⊆ setC a) : a ≥ 7 :=
  sorry

end intersection_complement_A_B_subset_A_C_l141_141394


namespace selling_price_correct_l141_141247

def initial_cost : ℕ := 800
def repair_cost : ℕ := 200
def gain_percent : ℕ := 40
def total_cost := initial_cost + repair_cost
def gain := (gain_percent * total_cost) / 100
def selling_price := total_cost + gain

theorem selling_price_correct : selling_price = 1400 := 
by
  sorry

end selling_price_correct_l141_141247


namespace wickets_before_last_match_l141_141462

theorem wickets_before_last_match (W : ℕ) (avg_before : ℝ) (wickets_taken : ℕ) (runs_conceded : ℝ) (avg_drop : ℝ) :
  avg_before = 12.4 → wickets_taken = 4 → runs_conceded = 26 → avg_drop = 0.4 →
  (avg_before - avg_drop) * (W + wickets_taken) = avg_before * W + runs_conceded →
  W = 55 :=
by
  intros
  sorry

end wickets_before_last_match_l141_141462


namespace scientific_notation_1742000_l141_141320

theorem scientific_notation_1742000 : 1742000 = 1.742 * 10^6 := 
by
  sorry

end scientific_notation_1742000_l141_141320


namespace inequality_sum_l141_141489

theorem inequality_sum (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a ^ 2 + b ^ 2 + c ^ 2 = 1) :
  (a / (a ^ 3 + b * c) + b / (b ^ 3 + c * a) + c / (c ^ 3 + a * b)) > 3 :=
by
  sorry

end inequality_sum_l141_141489


namespace range_of_a_for_decreasing_function_l141_141923

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := -x^2 - 2 * (a - 1) * x + 5

noncomputable def f' (x : ℝ) : ℝ := -2 * x - 2 * (a - 1)

theorem range_of_a_for_decreasing_function :
  (∀ x : ℝ, -1 ≤ x → f' a x ≤ 0) → 2 ≤ a := sorry

end range_of_a_for_decreasing_function_l141_141923


namespace YoongiHasSevenPets_l141_141347

def YoongiPets (dogs cats : ℕ) : ℕ := dogs + cats

theorem YoongiHasSevenPets : YoongiPets 5 2 = 7 :=
by
  sorry

end YoongiHasSevenPets_l141_141347


namespace least_number_subtracted_l141_141236

theorem least_number_subtracted (n m1 m2 m3 r : ℕ) (h_n : n = 642) (h_m1 : m1 = 11) (h_m2 : m2 = 13) (h_m3 : m3 = 17) (h_r : r = 4) :
  ∃ x : ℕ, (n - x) % m1 = r ∧ (n - x) % m2 = r ∧ (n - x) % m3 = r ∧ n - x = 638 :=
sorry

end least_number_subtracted_l141_141236


namespace option_d_not_necessarily_true_l141_141609

theorem option_d_not_necessarily_true (a b c : ℝ) (h: a > b) : ¬(a * c^2 > b * c^2) ↔ c = 0 :=
by sorry

end option_d_not_necessarily_true_l141_141609


namespace sin_cos_identity_l141_141277

variable (α : Real)

theorem sin_cos_identity (h : Real.sin α - Real.cos α = -5/4) : Real.sin α * Real.cos α = -9/32 :=
by
  sorry

end sin_cos_identity_l141_141277


namespace range_of_a_l141_141748

def A (x : ℝ) : Prop := abs (x - 4) < 2 * x

def B (x a : ℝ) : Prop := x * (x - a) ≥ (a + 6) * (x - a)

theorem range_of_a (a : ℝ) :
  (∀ x, A x → B x a) → a ≤ -14 / 3 :=
  sorry

end range_of_a_l141_141748


namespace find_range_of_t_l141_141811

variable {f : ℝ → ℝ}

-- Definitions for the conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ ⦃x y⦄, a < x ∧ x < b ∧ a < y ∧ y < b ∧ x < y → f x > f y

-- Given the conditions, we need to prove the statement
theorem find_range_of_t (h_odd : is_odd_function f)
    (h_decreasing : is_decreasing_on f (-1) 1)
    (h_inequality : ∀ t : ℝ, -1 < t ∧ t < 1 → f (1 - t) + f (1 - t^2) < 0) :
  ∀ t, -1 < t ∧ t < 1 → 0 < t ∧ t < 1 :=
  by
  sorry

end find_range_of_t_l141_141811


namespace order_of_y1_y2_y3_l141_141118

/-
Given three points A(-3, y1), B(3, y2), and C(4, y3) all lie on the parabola y = 2*(x - 2)^2 + 1,
prove that y2 < y3 < y1.
-/
theorem order_of_y1_y2_y3 :
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  sorry

end order_of_y1_y2_y3_l141_141118


namespace investment_ratio_l141_141602

-- Definitions of all the conditions
variables (A B C profit b_share: ℝ)

-- Conditions based on the provided problem
def condition1 (n : ℝ) : Prop := A = n * B
def condition2 : Prop := B = (2 / 3) * C
def condition3 : Prop := profit = 4400
def condition4 : Prop := b_share = 800

-- The theorem we want to prove
theorem investment_ratio (n : ℝ) :
  (condition1 A B n) ∧ (condition2 B C) ∧ (condition3 profit) ∧ (condition4 b_share) → A / B = 3 :=
by
  sorry

end investment_ratio_l141_141602


namespace sin2theta_plus_cos2theta_l141_141805

theorem sin2theta_plus_cos2theta (θ : ℝ) (h : Real.tan θ = 2) : Real.sin (2 * θ) + Real.cos (2 * θ) = 1 / 5 :=
by
  sorry

end sin2theta_plus_cos2theta_l141_141805


namespace angle_OA_plane_ABC_l141_141151

noncomputable def sphere_radius (A B C : Type*) (O : Type*) : ℝ :=
  let surface_area : ℝ := 48 * Real.pi
  let AB : ℝ := 2
  let BC : ℝ := 4
  let angle_ABC : ℝ := Real.pi / 3
  let radius := Real.sqrt (surface_area / (4 * Real.pi))
  radius

noncomputable def length_AC (A B C : Type*) : ℝ :=
  let AB : ℝ := 2
  let BC : ℝ := 4
  let angle_ABC : ℝ := Real.pi / 3
  let AC := Real.sqrt (AB ^ 2 + BC ^ 2 - 2 * AB * BC * Real.cos angle_ABC)
  AC

theorem angle_OA_plane_ABC 
(A B C O : Type*)
(radius : ℝ)
(AC : ℝ) :
radius = 2 * Real.sqrt 3 ∧
AC = 2 * Real.sqrt 3 ∧ 
(AB : ℝ) = 2 ∧ 
(BC : ℝ) = 4 ∧ 
(angle_ABC : ℝ) = Real.pi / 3
→ ∃ (angle_OA_plane_ABC : ℝ), angle_OA_plane_ABC = Real.arccos (Real.sqrt 3 / 3) :=
by
  intro h
  sorry

end angle_OA_plane_ABC_l141_141151


namespace square_difference_l141_141008

theorem square_difference (a b : ℝ) (h : a > b) :
  ∃ c : ℝ, c^2 = a^2 - b^2 :=
by
  sorry

end square_difference_l141_141008


namespace min_value_geometric_sequence_l141_141035

-- Definition for conditions and problem setup
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n

-- Given data
variable (a : ℕ → ℝ)
variable (h_geom : is_geometric_sequence a)
variable (h_sum : a 2015 + a 2017 = Real.pi)

-- Goal statement
theorem min_value_geometric_sequence :
  ∃ (min_val : ℝ), min_val = (Real.pi^2) / 2 ∧ (
    ∀ a : ℕ → ℝ, 
    is_geometric_sequence a → 
    a 2015 + a 2017 = Real.pi → 
    a 2016 * (a 2014 + a 2018) ≥ (Real.pi^2) / 2
  ) :=
sorry

end min_value_geometric_sequence_l141_141035


namespace min_value_of_function_l141_141146

theorem min_value_of_function (x : ℝ) (h : x > -1) : 
  (∀ x₀ : ℝ, x₀ > -1 → (x₀ + 1 + 1 / (x₀ + 1) - 1) ≥ 1) ∧ (x = 0) :=
sorry

end min_value_of_function_l141_141146


namespace root_exists_l141_141623

variable {R : Type} [LinearOrderedField R]
variables (a b c : R)

def f (x : R) : R := a * x^2 + b * x + c

theorem root_exists (h : f a b c ((a - b - c) / (2 * a)) = 0) : f a b c (-1) = 0 ∨ f a b c 1 = 0 := by
  sorry

end root_exists_l141_141623


namespace total_value_of_item_l141_141531

theorem total_value_of_item (V : ℝ) 
  (h1 : ∃ V > 1000, 
              0.07 * (V - 1000) + 
              (if 55 > 50 then (55 - 50) * 0.15 else 0) + 
              0.05 * V = 112.70) :
  V = 1524.58 :=
by 
  sorry

end total_value_of_item_l141_141531


namespace actual_number_of_children_l141_141704

theorem actual_number_of_children (N : ℕ) (B : ℕ) 
  (h1 : B = 2 * N)
  (h2 : ∀ k : ℕ, k = N - 330)
  (h3 : B = 4 * (N - 330)) : 
  N = 660 :=
by 
  sorry

end actual_number_of_children_l141_141704


namespace max_second_smallest_l141_141387

noncomputable def f (M : ℕ) : ℕ :=
  (M - 1) * (90 - M) * (89 - M) * (88 - M)

theorem max_second_smallest (M : ℕ) (cond : 1 ≤ M ∧ M ≤ 89) : M = 23 ↔ (∀ N : ℕ, f M ≥ f N) :=
by
  sorry

end max_second_smallest_l141_141387


namespace R_and_D_expense_corresponding_to_productivity_increase_l141_141631

/-- Given values for R&D expenses and increase in average labor productivity -/
def R_and_D_t : ℝ := 2640.92
def Delta_APL_t_plus_2 : ℝ := 0.81

/-- Statement to be proved: the R&D expense in million rubles corresponding 
    to an increase in average labor productivity by 1 million rubles per person -/
theorem R_and_D_expense_corresponding_to_productivity_increase : 
  R_and_D_t / Delta_APL_t_plus_2 = 3260 := 
by
  sorry

end R_and_D_expense_corresponding_to_productivity_increase_l141_141631


namespace arithmetic_sequence_sum_l141_141029

-- Definitions from conditions
def arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a6 : a 6 = -2) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3 :=
sorry

end arithmetic_sequence_sum_l141_141029


namespace AE_length_l141_141358

theorem AE_length :
  ∀ (A B C D E : Type) 
    (AB CD AC BD AE EC : ℕ),
  AB = 12 → CD = 15 → AC = 18 → BD = 27 → 
  (AE + EC = AC) → 
  (AE * (18 - AE)) = (4 / 9 * 18 * 8) → 
  9 * AE = 72 → 
  AE = 8 := 
by
  intros A B C D E AB CD AC BD AE EC hAB hCD hAC hBD hSum hEqual hSolve
  sorry

end AE_length_l141_141358


namespace marker_cost_l141_141160

theorem marker_cost (s n c : ℕ) (h_majority : s > 20) (h_markers : n > 1) (h_cost : c > n) (h_total_cost : s * n * c = 3388) : c = 11 :=
by {
  sorry
}

end marker_cost_l141_141160


namespace problem1_problem2_l141_141545

variable {A B C : ℝ} {AC BC : ℝ}

-- Condition: BC = 2AC
def condition1 (AC BC : ℝ) : Prop := BC = 2 * AC

-- Problem 1: Prove 4cos^2(B) - cos^2(A) = 3
theorem problem1 (h : condition1 AC BC) :
  4 * Real.cos B ^ 2 - Real.cos A ^ 2 = 3 :=
sorry

-- Problem 2: Prove the maximum value of (sin(A) / (2cos(B) + cos(A))) is 2/3 for A ∈ (0, π)
theorem problem2 (h : condition1 AC BC) (hA : 0 < A ∧ A < Real.pi) :
  ∃ t : ℝ, (t = Real.sin A / (2 * Real.cos B + Real.cos A) ∧ t ≤ 2/3) :=
sorry

end problem1_problem2_l141_141545


namespace base_conversion_problem_l141_141125

theorem base_conversion_problem :
  ∃ A B : ℕ, 0 ≤ A ∧ A < 8 ∧ 0 ≤ B ∧ B < 6 ∧
           8 * A + B = 6 * B + A ∧
           8 * A + B = 45 :=
by
  sorry

end base_conversion_problem_l141_141125


namespace mr_brown_net_result_l141_141739

noncomputable def C1 := 1.50 / 1.3
noncomputable def C2 := 1.50 / 0.9
noncomputable def profit_from_first_pen := 1.50 - C1
noncomputable def tax := 0.05 * profit_from_first_pen
noncomputable def total_cost := C1 + C2
noncomputable def total_revenue := 3.00
noncomputable def net_result := total_revenue - total_cost - tax

theorem mr_brown_net_result : net_result = 0.16 :=
by
  sorry

end mr_brown_net_result_l141_141739


namespace sum_of_coordinates_of_B_l141_141483

def point := (ℝ × ℝ)

noncomputable def point_A : point := (0, 0)

def line_y_equals_6 (B : point) : Prop := B.snd = 6

def slope_AB (A B : point) (m : ℝ) : Prop := (B.snd - A.snd) / (B.fst - A.fst) = m

theorem sum_of_coordinates_of_B (B : point) 
  (h1 : B.snd = 6)
  (h2 : slope_AB point_A B (3/5)) :
  B.fst + B.snd = 16 :=
sorry

end sum_of_coordinates_of_B_l141_141483


namespace range_of_x_l141_141971

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x - 1

theorem range_of_x (x : ℝ) (h : f (x^2 - 4) < 2) : 
  (-Real.sqrt 5 < x ∧ x < -2) ∨ (2 < x ∧ x < Real.sqrt 5) :=
sorry

end range_of_x_l141_141971


namespace gcd_1995_228_eval_f_at_2_l141_141879

-- Euclidean Algorithm Problem
theorem gcd_1995_228 : Nat.gcd 1995 228 = 57 :=
by
  sorry

-- Horner's Method Problem
def f (x : ℝ) : ℝ := 3 * x ^ 5 + 2 * x ^ 3 - 8 * x + 5

theorem eval_f_at_2 : f 2 = 101 :=
by
  sorry

end gcd_1995_228_eval_f_at_2_l141_141879


namespace point_on_circle_x_value_l141_141870

/-
In the xy-plane, the segment with endpoints (-3,0) and (21,0) is the diameter of a circle.
If the point (x,12) is on the circle, then x = 9.
-/
theorem point_on_circle_x_value :
  let c := (9, 0) -- center of the circle
  let r := 12 -- radius of the circle
  let circle := {p | (p.1 - 9)^2 + p.2^2 = 144} -- equation of the circle
  ∀ x : Real, (x, 12) ∈ circle → x = 9 :=
by
  intros
  sorry

end point_on_circle_x_value_l141_141870


namespace walter_equal_share_l141_141622

-- Conditions
def jefferson_bananas : ℕ := 56
def walter_fewer_fraction : ℚ := 1 / 4
def walter_fewer_bananas := walter_fewer_fraction * jefferson_bananas
def walter_bananas := jefferson_bananas - walter_fewer_bananas
def total_bananas := walter_bananas + jefferson_bananas

-- Proof problem: Prove that Walter gets 49 bananas when they share the total number of bananas equally.
theorem walter_equal_share : total_bananas / 2 = 49 := 
by sorry

end walter_equal_share_l141_141622


namespace chord_bisected_by_point_l141_141752

theorem chord_bisected_by_point (x y : ℝ) (h : (x - 2)^2 / 16 + (y - 1)^2 / 8 = 1) :
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -3 ∧ (∀ x y : ℝ, (a * x + b * y + c = 0 ↔ (x - 2)^2 / 16 + (y - 1)^2 / 8 = 1)) := by
  sorry

end chord_bisected_by_point_l141_141752


namespace books_about_sports_l141_141966

theorem books_about_sports (total_books school_books sports_books : ℕ) 
  (h1 : total_books = 58)
  (h2 : school_books = 19) 
  (h3 : sports_books = total_books - school_books) :
  sports_books = 39 :=
by 
  rw [h1, h2] at h3 
  exact h3

end books_about_sports_l141_141966


namespace draw_white_ball_is_impossible_l141_141220

-- Definitions based on the conditions
def redBalls : Nat := 2
def blackBalls : Nat := 6
def totalBalls : Nat := redBalls + blackBalls

-- Definition for the white ball drawing event
def whiteBallDraw (redBalls blackBalls : Nat) : Prop :=
  ∀ (n : Nat), n ≠ 0 → n ≤ redBalls + blackBalls → false

-- Theorem to prove the event is impossible
theorem draw_white_ball_is_impossible : whiteBallDraw redBalls blackBalls :=
  by
  sorry

end draw_white_ball_is_impossible_l141_141220


namespace tan_alpha_minus_beta_l141_141487

theorem tan_alpha_minus_beta (α β : ℝ) (hα : Real.tan α = 8) (hβ : Real.tan β = 7) :
  Real.tan (α - β) = 1 / 57 := 
sorry

end tan_alpha_minus_beta_l141_141487


namespace IncorrectOption_l141_141895

namespace Experiment

def OptionA : Prop := 
  ∃ method : String, method = "sampling detection"

def OptionB : Prop := 
  ¬(∃ experiment : String, experiment = "does not need a control group, nor repeated experiments")

def OptionC : Prop := 
  ∃ action : String, action = "test tube should be gently shaken"

def OptionD : Prop := 
  ∃ condition : String, condition = "field of view should not be too bright"

theorem IncorrectOption : OptionB :=
  sorry

end Experiment

end IncorrectOption_l141_141895


namespace paint_floor_cost_l141_141791

theorem paint_floor_cost :
  ∀ (L : ℝ) (rate : ℝ)
  (condition1 : L = 3 * (L / 3))
  (condition2 : L = 19.595917942265423)
  (condition3 : rate = 5),
  rate * (L * (L / 3)) = 640 :=
by
  intros L rate condition1 condition2 condition3
  sorry

end paint_floor_cost_l141_141791


namespace problem_I_problem_II_l141_141423

-- Definitions
def p (x : ℝ) : Prop := (x + 2) * (x - 3) ≤ 0
def q (m : ℝ) (x : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Problem (I)
theorem problem_I (m : ℝ) : m > 0 → (∀ x : ℝ, q m x → p x) → 0 < m ∧ m ≤ 2 := by
  sorry

-- Problem (II)
theorem problem_II (x : ℝ) : 7 > 0 → 
  (p x ∨ q 7 x) ∧ ¬(p x ∧ q 7 x) → 
  (-6 ≤ x ∧ x < -2) ∨ (3 < x ∧ x ≤ 8) := by
  sorry

end problem_I_problem_II_l141_141423


namespace common_ratio_is_2_l141_141546

noncomputable def common_ratio (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 2 * d) ^ 2 = a1 * (a1 + 6 * d)) : ℝ :=
(a1 + 2 * d) / a1

theorem common_ratio_is_2 (a1 d : ℝ) (h1 : d ≠ 0) (h2 : (a1 + 2 * d) ^ 2 = a1 * (a1 + 6 * d)) : 
    common_ratio a1 d h1 h2 = 2 :=
by
  -- Proof would go here
  sorry

end common_ratio_is_2_l141_141546


namespace travel_distance_l141_141089

-- Define the average speed of the car
def speed : ℕ := 68

-- Define the duration of the trip in hours
def time : ℕ := 12

-- Define the distance formula for constant speed
def distance (speed time : ℕ) : ℕ := speed * time

-- Proof statement
theorem travel_distance : distance speed time = 756 := by
  -- Provide a placeholder for the proof
  sorry

end travel_distance_l141_141089


namespace evaluate_expression_l141_141482

theorem evaluate_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := by 
  sorry

end evaluate_expression_l141_141482


namespace find_g_seven_l141_141844

variable {g : ℝ → ℝ}

theorem find_g_seven (h : ∀ x : ℝ, g (3 * x - 2) = 5 * x + 4) : g 7 = 19 :=
by
  sorry

end find_g_seven_l141_141844


namespace compare_final_values_l141_141484

noncomputable def final_value_Almond (initial: ℝ): ℝ := (initial * 1.15) * 0.85
noncomputable def final_value_Bean (initial: ℝ): ℝ := (initial * 0.80) * 1.20
noncomputable def final_value_Carrot (initial: ℝ): ℝ := (initial * 1.10) * 0.90

theorem compare_final_values (initial: ℝ) (h_positive: 0 < initial):
  final_value_Almond initial < final_value_Bean initial ∧ 
  final_value_Bean initial < final_value_Carrot initial := by
  sorry

end compare_final_values_l141_141484


namespace only_solution_to_n_mul_a_n_mul_b_eq_n_mul_c_l141_141640

theorem only_solution_to_n_mul_a_n_mul_b_eq_n_mul_c
  (n a b c : ℕ) (hn : n > 1) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hca : c > a) (hcb : c > b) (hab : a ≤ b) :
  n * a + n * b = n * c ↔ (n = 2 ∧ b = a ∧ c = a + 1) := by
  sorry

end only_solution_to_n_mul_a_n_mul_b_eq_n_mul_c_l141_141640


namespace nathan_blankets_l141_141629

theorem nathan_blankets (b : ℕ) (hb : 21 = (b / 2) * 3) : b = 14 :=
by sorry

end nathan_blankets_l141_141629


namespace cafeteria_apples_count_l141_141284

def initial_apples : ℕ := 17
def used_monday : ℕ := 2
def bought_monday : ℕ := 23
def used_tuesday : ℕ := 4
def bought_tuesday : ℕ := 15
def used_wednesday : ℕ := 3

def final_apples (initial_apples used_monday bought_monday used_tuesday bought_tuesday used_wednesday : ℕ) : ℕ :=
  initial_apples - used_monday + bought_monday - used_tuesday + bought_tuesday - used_wednesday

theorem cafeteria_apples_count :
  final_apples initial_apples used_monday bought_monday used_tuesday bought_tuesday used_wednesday = 46 :=
by
  sorry

end cafeteria_apples_count_l141_141284


namespace right_triangle_hypotenuse_length_l141_141667

theorem right_triangle_hypotenuse_length (a b : ℝ) (c : ℝ) (h₁ : a = 10) (h₂ : b = 24) (h₃ : c^2 = a^2 + b^2) : c = 26 :=
by
  -- sorry is used to skip the actual proof
  sorry

end right_triangle_hypotenuse_length_l141_141667


namespace exists_consecutive_numbers_with_prime_divisors_l141_141259

theorem exists_consecutive_numbers_with_prime_divisors (p q : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (h : p < q ∧ q < 2 * p) :
  ∃ n m : ℕ, (m = n + 1) ∧ 
             (Nat.gcd n p = p) ∧ (Nat.gcd m p = 1) ∧ 
             (Nat.gcd m q = q) ∧ (Nat.gcd n q = 1) :=
by
  sorry

end exists_consecutive_numbers_with_prime_divisors_l141_141259


namespace total_whales_observed_l141_141521

-- Define the conditions
def trip1_male_whales : ℕ := 28
def trip1_female_whales : ℕ := 2 * trip1_male_whales
def trip1_total_whales : ℕ := trip1_male_whales + trip1_female_whales

def baby_whales_trip2 : ℕ := 8
def adult_whales_trip2 : ℕ := 2 * baby_whales_trip2
def trip2_total_whales : ℕ := baby_whales_trip2 + adult_whales_trip2

def trip3_male_whales : ℕ := trip1_male_whales / 2
def trip3_female_whales : ℕ := trip1_female_whales
def trip3_total_whales : ℕ := trip3_male_whales + trip3_female_whales

-- Prove the total number of whales observed
theorem total_whales_observed : trip1_total_whales + trip2_total_whales + trip3_total_whales = 178 := by
  -- Assuming all intermediate steps are correct
  sorry

end total_whales_observed_l141_141521


namespace ratio_of_areas_l141_141964

-- Define the squares and their side lengths
def Square (side_length : ℝ) := side_length * side_length

-- Define the side lengths of Square C and Square D
def side_C (x : ℝ) : ℝ := x
def side_D (x : ℝ) : ℝ := 3 * x

-- Define their areas
def area_C (x : ℝ) : ℝ := Square (side_C x)
def area_D (x : ℝ) : ℝ := Square (side_D x)

-- The statement to prove
theorem ratio_of_areas (x : ℝ) (hx : x ≠ 0) : area_C x / area_D x = 1 / 9 := by
  sorry

end ratio_of_areas_l141_141964


namespace joy_sees_grandma_in_48_hours_l141_141961

def days_until_joy_sees_grandma : ℕ := 2
def hours_per_day : ℕ := 24

theorem joy_sees_grandma_in_48_hours :
  days_until_joy_sees_grandma * hours_per_day = 48 := 
by
  sorry

end joy_sees_grandma_in_48_hours_l141_141961


namespace car_speed_l141_141302

theorem car_speed (v : ℝ) (h : (1 / v) = (1 / 100 + 2 / 3600)) : v = 3600 / 38 := 
by
  sorry

end car_speed_l141_141302


namespace sum_of_angles_l141_141537

theorem sum_of_angles (x u v : ℝ) (h1 : u = Real.sin x) (h2 : v = Real.cos x)
  (h3 : 0 ≤ x ∧ x ≤ 2 * Real.pi) 
  (h4 : Real.sin x ^ 4 - Real.cos x ^ 4 = (u - v) / (u * v)) 
  : x = Real.pi / 4 ∨ x = 5 * Real.pi / 4 → (Real.pi / 4 + 5 * Real.pi / 4) = 3 * Real.pi / 2 := 
by
  intro h
  sorry

end sum_of_angles_l141_141537


namespace sqrt_of_4_l141_141327

theorem sqrt_of_4 : {x : ℤ | x^2 = 4} = {2, -2} :=
by
  sorry

end sqrt_of_4_l141_141327


namespace polygon_with_equal_angle_sums_is_quadrilateral_l141_141979

theorem polygon_with_equal_angle_sums_is_quadrilateral 
    (n : ℕ)
    (h1 : (n - 2) * 180 = 360)
    (h2 : 360 = 360) :
  n = 4 := 
sorry

end polygon_with_equal_angle_sums_is_quadrilateral_l141_141979


namespace ken_climbing_pace_l141_141154

noncomputable def sari_pace : ℝ := 350 -- Sari's pace in meters per hour, derived from 700 meters in 2 hours.

def ken_pace : ℝ := 500 -- We will need to prove this.

theorem ken_climbing_pace :
  let start_time_sari := 5
  let start_time_ken := 7
  let end_time_ken := 12
  let time_ken_climbs := end_time_ken - start_time_ken
  let sari_initial_headstart := 700 -- meters
  let sari_behind_ken := 50 -- meters
  let sari_total_climb := sari_pace * time_ken_climbs
  let total_distance_ken := sari_total_climb + sari_initial_headstart + sari_behind_ken
  ken_pace = total_distance_ken / time_ken_climbs :=
by
  sorry

end ken_climbing_pace_l141_141154


namespace pool_capacity_percentage_l141_141337

theorem pool_capacity_percentage :
  let width := 60 
  let length := 150 
  let depth := 10 
  let drain_rate := 60 
  let time := 1200 
  let total_volume := width * length * depth
  let water_removed := drain_rate * time
  let capacity_percentage := (water_removed / total_volume : ℚ) * 100
  capacity_percentage = 80 := by
  sorry

end pool_capacity_percentage_l141_141337


namespace total_cost_is_correct_l141_141587

-- Define the conditions
def piano_cost : ℝ := 500
def lesson_cost_per_lesson : ℝ := 40
def number_of_lessons : ℝ := 20
def discount_rate : ℝ := 0.25

-- Define the total cost of lessons before discount
def total_lesson_cost_before_discount : ℝ := lesson_cost_per_lesson * number_of_lessons

-- Define the discount amount
def discount_amount : ℝ := discount_rate * total_lesson_cost_before_discount

-- Define the total cost of lessons after discount
def total_lesson_cost_after_discount : ℝ := total_lesson_cost_before_discount - discount_amount

-- Define the total cost of everything
def total_cost : ℝ := piano_cost + total_lesson_cost_after_discount

-- The statement to be proven
theorem total_cost_is_correct : total_cost = 1100 := by
  sorry

end total_cost_is_correct_l141_141587


namespace percent_same_grades_l141_141784

theorem percent_same_grades 
    (total_students same_A same_B same_C same_D same_E : ℕ)
    (h_total_students : total_students = 40)
    (h_same_A : same_A = 3)
    (h_same_B : same_B = 5)
    (h_same_C : same_C = 6)
    (h_same_D : same_D = 2)
    (h_same_E : same_E = 1):
    ((same_A + same_B + same_C + same_D + same_E : ℚ) / total_students * 100) = 42.5 :=
by
  sorry

end percent_same_grades_l141_141784


namespace parabola_intersections_l141_141117

open Real

-- Definition of the two parabolas
def parabola1 (x : ℝ) : ℝ := 3*x^2 - 6*x + 2
def parabola2 (x : ℝ) : ℝ := 9*x^2 - 4*x - 5

-- Theorem stating the intersections are (-7/3, 9) and (0.5, -0.25)
theorem parabola_intersections : 
  {p : ℝ × ℝ | parabola1 p.1 = p.2 ∧ parabola2 p.1 = p.2} =
  {(-7/3, 9), (0.5, -0.25)} :=
by 
  sorry

end parabola_intersections_l141_141117


namespace prime_square_plus_two_is_prime_iff_l141_141263

theorem prime_square_plus_two_is_prime_iff (p : ℕ) (hp : Prime p) : Prime (p^2 + 2) ↔ p = 3 :=
sorry

end prime_square_plus_two_is_prime_iff_l141_141263


namespace evaluate_expression_l141_141470

theorem evaluate_expression : ((2^2 - 2) - (3^2 - 3) + (4^2 - 4) = 8) :=
sorry

end evaluate_expression_l141_141470


namespace domino_tile_count_l141_141977

theorem domino_tile_count (low high : ℕ) (tiles_standard_set : ℕ) (range_standard_set : ℕ) (range_new_set : ℕ) :
  range_standard_set = 6 → tiles_standard_set = 28 →
  low = 0 → high = 12 →
  range_new_set = 13 → 
  (∀ n, 0 ≤ n ∧ n ≤ range_standard_set → ∀ m, n ≤ m ∧ m ≤ range_standard_set → n ≤ m → true) →
  (∀ n, 0 ≤ n ∧ n ≤ range_new_set → ∀ m, n ≤ m ∧ m <= range_new_set → n <= m → true) →
  tiles_new_set = 91 :=
by
  intros h_range_standard h_tiles_standard h_low h_high h_range_new h_standard_pairs h_new_pairs
  --skipping the proof
  sorry

end domino_tile_count_l141_141977


namespace deposit_is_500_l141_141158

-- Definitions corresponding to the conditions
def janet_saved : ℕ := 2225
def rent_per_month : ℕ := 1250
def advance_months : ℕ := 2
def extra_needed : ℕ := 775

-- Definition that encapsulates the deposit calculation
def deposit_required (saved rent_monthly months_advance extra : ℕ) : ℕ :=
  let total_rent := months_advance * rent_monthly
  let total_needed := saved + extra
  total_needed - total_rent

-- Theorem statement for the proof problem
theorem deposit_is_500 : deposit_required janet_saved rent_per_month advance_months extra_needed = 500 :=
by
  sorry

end deposit_is_500_l141_141158


namespace area_of_parallelogram_l141_141316

-- Define the vectors
def v : ℝ × ℝ := (7, -5)
def w : ℝ × ℝ := (14, -4)

-- Prove the area of the parallelogram
theorem area_of_parallelogram : 
  abs (v.1 * w.2 - v.2 * w.1) = 42 :=
by
  sorry

end area_of_parallelogram_l141_141316


namespace find_f_of_2_l141_141507

theorem find_f_of_2 (f g : ℝ → ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x) 
                    (h₂ : ∀ x : ℝ, g x = f x + 9) (h₃ : g (-2) = 3) :
                    f 2 = 6 :=
by
  sorry

end find_f_of_2_l141_141507


namespace smallest_value_x_squared_plus_six_x_plus_nine_l141_141163

theorem smallest_value_x_squared_plus_six_x_plus_nine : ∀ x : ℝ, x^2 + 6 * x + 9 ≥ 0 :=
by sorry

end smallest_value_x_squared_plus_six_x_plus_nine_l141_141163


namespace cost_price_per_meter_l141_141718

-- Given conditions
def total_selling_price : ℕ := 18000
def total_meters_sold : ℕ := 400
def loss_per_meter : ℕ := 5

-- Statement to be proven
theorem cost_price_per_meter : 
    ((total_selling_price + (loss_per_meter * total_meters_sold)) / total_meters_sold) = 50 := 
by
    sorry

end cost_price_per_meter_l141_141718


namespace gcd_228_2008_l141_141494

theorem gcd_228_2008 : Int.gcd 228 2008 = 4 := by
  sorry

end gcd_228_2008_l141_141494


namespace complex_power_difference_l141_141697

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 40 - (1 - i) ^ 40 = 0 := by 
  sorry

end complex_power_difference_l141_141697


namespace cupcakes_frosted_l141_141715

def Cagney_rate := 1 / 25
def Lacey_rate := 1 / 35
def time_duration := 600
def combined_rate := Cagney_rate + Lacey_rate
def total_cupcakes := combined_rate * time_duration

theorem cupcakes_frosted (Cagney_rate Lacey_rate time_duration combined_rate total_cupcakes : ℝ) 
  (hC: Cagney_rate = 1 / 25)
  (hL: Lacey_rate = 1 / 35)
  (hT: time_duration = 600)
  (hCR: combined_rate = Cagney_rate + Lacey_rate)
  (hTC: total_cupcakes = combined_rate * time_duration) :
  total_cupcakes = 41 :=
sorry

end cupcakes_frosted_l141_141715


namespace decreasing_hyperbola_l141_141178

theorem decreasing_hyperbola (m : ℝ) (x : ℝ) (hx : x > 0) (y : ℝ) (h_eq : y = (1 - m) / x) :
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > x₁ → (1 - m) / x₂ < (1 - m) / x₁) ↔ m < 1 :=
by
  sorry

end decreasing_hyperbola_l141_141178


namespace max_tiles_on_floor_l141_141366

theorem max_tiles_on_floor
  (tile_w tile_h floor_w floor_h : ℕ)
  (h_tile_w : tile_w = 25)
  (h_tile_h : tile_h = 65)
  (h_floor_w : floor_w = 150)
  (h_floor_h : floor_h = 390) :
  max ((floor_h / tile_h) * (floor_w / tile_w))
      ((floor_h / tile_w) * (floor_w / tile_h)) = 36 :=
by
  -- Given conditions and calculations will be proved in the proof.
  sorry

end max_tiles_on_floor_l141_141366


namespace twenty_percent_of_x_l141_141085

noncomputable def x := 1800 / 1.2

theorem twenty_percent_of_x (h : 1.2 * x = 1800) : 0.2 * x = 300 :=
by
  -- The proof would go here, but we'll replace it with sorry.
  sorry

end twenty_percent_of_x_l141_141085


namespace union_of_A_and_B_l141_141495

namespace SetUnionProof

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x | x ≤ 2 }
def C : Set ℝ := { x | x ≤ 2 }

theorem union_of_A_and_B : A ∪ B = C := by
  -- proof goes here
  sorry

end SetUnionProof

end union_of_A_and_B_l141_141495


namespace roots_quadratic_identity_l141_141446

theorem roots_quadratic_identity :
  ∀ (r s : ℝ), (r^2 - 5 * r + 3 = 0) ∧ (s^2 - 5 * s + 3 = 0) → r^2 + s^2 = 19 :=
by
  intros r s h
  sorry

end roots_quadratic_identity_l141_141446


namespace compare_expressions_l141_141213

theorem compare_expressions (x : ℝ) : (x - 2) * (x + 3) > x^2 + x - 7 :=
by {
  -- below proof is left as an exercise
  sorry
}

end compare_expressions_l141_141213


namespace jill_average_number_of_stickers_l141_141515

def average_stickers (packs : List ℕ) : ℚ :=
  (packs.sum : ℚ) / packs.length

theorem jill_average_number_of_stickers :
  average_stickers [5, 7, 9, 9, 11, 15, 15, 17, 19, 21] = 12.8 :=
by
  sorry

end jill_average_number_of_stickers_l141_141515


namespace smallest_x_l141_141486

theorem smallest_x 
  (x : ℝ)
  (h : ( ( (5 * x - 20) / (4 * x - 5) ) ^ 2 + ( (5 * x - 20) / (4 * x - 5) ) ) = 6 ) :
  x = -10 / 3 := sorry

end smallest_x_l141_141486


namespace number_of_common_tangents_of_two_circles_l141_141858

theorem number_of_common_tangents_of_two_circles 
  (x y : ℝ)
  (circle1 : x^2 + y^2 = 1)
  (circle2 : x^2 + y^2 - 6 * x - 8 * y + 9 = 0) :
  ∃ n : ℕ, n = 3 :=
by
  sorry

end number_of_common_tangents_of_two_circles_l141_141858


namespace ratio_of_percentages_l141_141938

theorem ratio_of_percentages (x y : ℝ) (h : 0.10 * x = 0.20 * y) : x / y = 2 :=
by
  sorry

end ratio_of_percentages_l141_141938


namespace part_a_part_b_l141_141808

theorem part_a (a : Fin 10 → ℤ) : ∃ i j : Fin 10, i ≠ j ∧ 27 ∣ (a i)^3 - (a j)^3 := sorry
theorem part_b (b : Fin 8 → ℤ) : ∃ i j : Fin 8, i ≠ j ∧ 27 ∣ (b i)^3 - (b j)^3 := sorry

end part_a_part_b_l141_141808


namespace amount_after_a_year_l141_141204

def initial_amount : ℝ := 90
def interest_rate : ℝ := 0.10

theorem amount_after_a_year : initial_amount * (1 + interest_rate) = 99 := 
by
  -- Here 'sorry' indicates that the proof is not provided.
  sorry

end amount_after_a_year_l141_141204


namespace inequality_solution_l141_141643

theorem inequality_solution (x : ℝ) : (1 - x > 0) ∧ ((x + 2) / 3 - 1 ≤ x) ↔ (-1/2 ≤ x ∧ x < 1) :=
by
  sorry

end inequality_solution_l141_141643


namespace anna_and_bob_play_together_l141_141254

-- Definitions based on the conditions
def total_players := 12
def matches_per_week := 2
def players_per_match := 6
def anna_and_bob := 2
def other_players := total_players - anna_and_bob
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Lean statement based on the equivalent proof problem
theorem anna_and_bob_play_together :
  combination other_players (players_per_match - anna_and_bob) = 210 := by
  -- To use Binomial Theorem in Lean
  -- The mathematical equivalent is C(10, 4) = 210
  sorry

end anna_and_bob_play_together_l141_141254


namespace find_mean_l141_141570

noncomputable def mean_of_normal_distribution (σ : ℝ) (value : ℝ) (std_devs : ℝ) : ℝ :=
value + std_devs * σ

theorem find_mean
  (σ : ℝ := 1.5)
  (value : ℝ := 12)
  (std_devs : ℝ := 2)
  (h : value = mean_of_normal_distribution σ (value - std_devs * σ) std_devs) :
  mean_of_normal_distribution σ value std_devs = 15 :=
sorry

end find_mean_l141_141570


namespace seeder_path_length_l141_141004

theorem seeder_path_length (initial_grain : ℤ) (decrease_percent : ℝ) (seeding_rate : ℝ) (width : ℝ) 
  (H_initial_grain : initial_grain = 250) 
  (H_decrease_percent : decrease_percent = 14 / 100) 
  (H_seeding_rate : seeding_rate = 175) 
  (H_width : width = 4) :
  (initial_grain * decrease_percent / seeding_rate) * 10000 / width = 500 := 
by 
  sorry

end seeder_path_length_l141_141004


namespace H_function_is_f_x_abs_x_l141_141846

-- Definition: A function f is odd if ∀ x ∈ ℝ, f(-x) = -f(x)
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Condition: A function f is strictly increasing if ∀ x1, x2 ∈ ℝ, x1 < x2 implies f(x1) < f(x2)
def is_strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

-- Define the function f(x) = x * |x|
def f (x : ℝ) : ℝ := x * abs x

-- The main theorem which states that f(x) = x * |x| is an "H function"
theorem H_function_is_f_x_abs_x : is_odd f ∧ is_strictly_increasing f :=
  sorry

end H_function_is_f_x_abs_x_l141_141846


namespace geometric_sequence_8th_term_l141_141171

theorem geometric_sequence_8th_term (a : ℚ) (r : ℚ) (n : ℕ) (h_a : a = 27) (h_r : r = 2/3) (h_n : n = 8) :
  a * r^(n-1) = 128 / 81 :=
by
  rw [h_a, h_r, h_n]
  sorry

end geometric_sequence_8th_term_l141_141171


namespace square_diagonal_y_coordinate_l141_141770

theorem square_diagonal_y_coordinate 
(point_vertex : ℝ × ℝ) 
(x_int : ℝ) 
(area_square : ℝ) 
(y_int : ℝ) :
(point_vertex = (-6, -4)) →
(x_int = 3) →
(area_square = 324) →
(y_int = 5) → 
y_int = 5 := 
by
  intros h1 h2 h3 h4
  exact h4

end square_diagonal_y_coordinate_l141_141770


namespace compute_sqrt_fraction_l141_141523

theorem compute_sqrt_fraction :
  (Real.sqrt ((16^10 + 2^30) / (16^6 + 2^35))) = (256 / Real.sqrt 2049) :=
sorry

end compute_sqrt_fraction_l141_141523


namespace vaishali_total_stripes_l141_141773

def total_stripes (hats_with_3_stripes hats_with_4_stripes hats_with_no_stripes : ℕ) 
  (hats_with_5_stripes hats_with_7_stripes hats_with_1_stripe : ℕ) 
  (hats_with_10_stripes hats_with_2_stripes : ℕ)
  (stripes_per_hat_with_3 stripes_per_hat_with_4 stripes_per_hat_with_no : ℕ)
  (stripes_per_hat_with_5 stripes_per_hat_with_7 stripes_per_hat_with_1 : ℕ)
  (stripes_per_hat_with_10 stripes_per_hat_with_2 : ℕ) : ℕ :=
  hats_with_3_stripes * stripes_per_hat_with_3 +
  hats_with_4_stripes * stripes_per_hat_with_4 +
  hats_with_no_stripes * stripes_per_hat_with_no +
  hats_with_5_stripes * stripes_per_hat_with_5 +
  hats_with_7_stripes * stripes_per_hat_with_7 +
  hats_with_1_stripe * stripes_per_hat_with_1 +
  hats_with_10_stripes * stripes_per_hat_with_10 +
  hats_with_2_stripes * stripes_per_hat_with_2

#eval total_stripes 4 3 6 2 1 4 2 3 3 4 0 5 7 1 10 2 -- 71

theorem vaishali_total_stripes : (total_stripes 4 3 6 2 1 4 2 3 3 4 0 5 7 1 10 2) = 71 :=
by
  sorry

end vaishali_total_stripes_l141_141773


namespace compute_3X4_l141_141936

def operation_X (a b : ℤ) : ℤ := b + 12 * a - a^2

theorem compute_3X4 : operation_X 3 4 = 31 := 
by
  sorry

end compute_3X4_l141_141936


namespace percentage_caught_customers_l141_141206

noncomputable def total_sampling_percentage : ℝ := 0.25
noncomputable def caught_percentage : ℝ := 0.88

theorem percentage_caught_customers :
  total_sampling_percentage * caught_percentage = 0.22 :=
by
  sorry

end percentage_caught_customers_l141_141206


namespace mrs_doe_inheritance_l141_141238

noncomputable def calculateInheritance (totalTaxes : ℝ) : ℝ :=
  totalTaxes / 0.3625

theorem mrs_doe_inheritance (h : 0.3625 * calculateInheritance 15000 = 15000) :
  calculateInheritance 15000 = 41379 :=
by
  unfold calculateInheritance
  field_simp
  norm_cast
  sorry

end mrs_doe_inheritance_l141_141238


namespace tangent_line_ellipse_l141_141725

variable (a b x0 y0 : ℝ)
variable (x y : ℝ)

def ellipse (x y a b : ℝ) := (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

theorem tangent_line_ellipse :
  ellipse x y a b ∧ a > b ∧ (x0 ≠ 0 ∨ y0 ≠ 0) ∧ (x0 ^ 2) / (a ^ 2) + (y0 ^ 2) / (b ^ 2) > 1 →
  (x0 * x) / (a ^ 2) + (y0 * y) / (b ^ 2) = 1 :=
  sorry

end tangent_line_ellipse_l141_141725


namespace func_identity_equiv_l141_141719

theorem func_identity_equiv (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f (x) + f (y)) ↔ (∀ x y : ℝ, f (xy + x + y) = f (xy) + f (x) + f (y)) :=
by
  sorry

end func_identity_equiv_l141_141719


namespace exam_questions_count_l141_141411

theorem exam_questions_count (Q S : ℕ) 
    (hS : S = (4 * Q) / 5)
    (sergio_correct : Q - 4 = S + 6) : 
    Q = 50 :=
by 
  sorry

end exam_questions_count_l141_141411


namespace martha_total_points_l141_141580

-- Define the costs and points
def cost_beef := 11 * 3
def cost_fruits_vegetables := 4 * 8
def cost_spices := 6 * 3
def cost_other := 37

def total_spending := cost_beef + cost_fruits_vegetables + cost_spices + cost_other

def points_per_dollar := 50 / 10
def base_points := total_spending * points_per_dollar
def bonus_points := if total_spending > 100 then 250 else 0

def total_points := base_points + bonus_points

-- The theorem to prove the question == answer given the conditions
theorem martha_total_points :
  total_points = 850 :=
by
  sorry

end martha_total_points_l141_141580


namespace ways_to_climb_four_steps_l141_141395

theorem ways_to_climb_four_steps (ways_to_climb : ℕ → ℕ) 
  (h1 : ways_to_climb 1 = 1) 
  (h2 : ways_to_climb 2 = 2) 
  (h3 : ways_to_climb 3 = 3) 
  (h_step : ∀ n, ways_to_climb n = ways_to_climb (n - 1) + ways_to_climb (n - 2)) : 
  ways_to_climb 4 = 5 := 
sorry

end ways_to_climb_four_steps_l141_141395


namespace product_fraction_l141_141792

open Int

def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def first_three_primes : List ℕ := [2, 3, 5]
def next_three_composites : List ℕ := [14, 15, 16]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem product_fraction :
  (product first_six_composites : ℚ) / (product (first_three_primes ++ next_three_composites) : ℚ) = 24 / 7 :=
by 
  sorry

end product_fraction_l141_141792


namespace max_area_right_triangle_l141_141049

def right_triangle_max_area (l : ℝ) (p : ℝ) (h : ℝ) : ℝ :=
  l + p + h

noncomputable def maximal_area (x y : ℝ) : ℝ :=
  (1/2) * x * y

theorem max_area_right_triangle (x y : ℝ) (h : ℝ) (hp : h = Real.sqrt (x^2 + y^2)) (hp2: x + y + h = 60) :
  maximal_area 30 30 = 450 :=
by
  sorry

end max_area_right_triangle_l141_141049


namespace ribbon_leftover_correct_l141_141743

def initial_ribbon : ℕ := 84
def used_ribbon : ℕ := 46
def leftover_ribbon : ℕ := 38

theorem ribbon_leftover_correct : initial_ribbon - used_ribbon = leftover_ribbon :=
by
  sorry

end ribbon_leftover_correct_l141_141743


namespace find_f_15_l141_141856

theorem find_f_15
  (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f x + f (2 * x + y) + 7 * x * y = f (3 * x - 2 * y) + 3 * x ^ 2 + 2) :
  f 15 = 1202 := 
sorry

end find_f_15_l141_141856


namespace weight_distribution_l141_141859

theorem weight_distribution (x y z : ℕ) 
  (h1 : x + y + z = 100) 
  (h2 : x + 10 * y + 50 * z = 500) : 
  x = 60 ∧ y = 39 ∧ z = 1 :=
by {
  sorry
}

end weight_distribution_l141_141859


namespace sum_positive_132_l141_141620

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem sum_positive_132 {a: ℕ → ℝ}
  (h1: a 66 < 0)
  (h2: a 67 > 0)
  (h3: a 67 > |a 66|):
  ∃ n, ∀ k < n, S k > 0 :=
by
  have h4 : (a 67 - a 66) > 0 := sorry
  have h5 : a 67 + a 66 > 0 := sorry
  have h6 : 66 * (a 67 + a 66) > 0 := sorry
  have h7 : S 132 = 66 * (a 67 + a 66) := sorry
  existsi 132
  intro k hk
  sorry

end sum_positive_132_l141_141620


namespace max_area_guaranteed_l141_141911

noncomputable def max_rectangle_area (board_size : ℕ) (removed_cells : ℕ) : ℕ :=
  if board_size = 8 ∧ removed_cells = 8 then 8 else 0

theorem max_area_guaranteed :
  max_rectangle_area 8 8 = 8 :=
by
  -- Proof logic goes here
  sorry

end max_area_guaranteed_l141_141911


namespace route_down_distance_l141_141180

noncomputable def rate_up : ℝ := 3
noncomputable def time_up : ℝ := 2
noncomputable def time_down : ℝ := 2
noncomputable def rate_down := 1.5 * rate_up

theorem route_down_distance : rate_down * time_down = 9 := by
  sorry

end route_down_distance_l141_141180


namespace circles_intersect_line_l141_141069

theorem circles_intersect_line (m c : ℝ)
  (hA : (1 : ℝ) - 3 + c = 0)
  (hB : 1 = -(m - 1) / (-4)) :
  m + c = -1 :=
by
  sorry

end circles_intersect_line_l141_141069


namespace problem_solution_l141_141271

theorem problem_solution (a b : ℕ) (x : ℝ) (h1 : x^2 + 14 * x = 24) (h2 : x = Real.sqrt a - b) (h3 : a > 0) (h4 : b > 0) :
  a + b = 80 := 
sorry

end problem_solution_l141_141271


namespace buildingC_floors_if_five_times_l141_141820

-- Defining the number of floors in Building B
def floorsBuildingB : ℕ := 13

-- Theorem to prove the number of floors in Building C if it had five times as many floors as Building B
theorem buildingC_floors_if_five_times (FB : ℕ) (h : FB = floorsBuildingB) : (5 * FB) = 65 :=
by
  rw [h]
  exact rfl

end buildingC_floors_if_five_times_l141_141820


namespace large_box_chocolate_bars_l141_141379

theorem large_box_chocolate_bars (num_small_boxes : ℕ) (chocolates_per_box : ℕ) 
  (h1 : num_small_boxes = 18) (h2 : chocolates_per_box = 28) : 
  num_small_boxes * chocolates_per_box = 504 := by
  sorry

end large_box_chocolate_bars_l141_141379


namespace value_of_x_l141_141461

/-
Given the following conditions:
  x = a + 7,
  a = b + 9,
  b = c + 15,
  c = d + 25,
  d = 60,
Prove that x = 116.
-/

theorem value_of_x (a b c d x : ℤ) 
    (h1 : x = a + 7)
    (h2 : a = b + 9)
    (h3 : b = c + 15)
    (h4 : c = d + 25)
    (h5 : d = 60) : x = 116 := 
  sorry

end value_of_x_l141_141461


namespace markup_rate_l141_141508

variable (S : ℝ) (C : ℝ)
variable (profit_percent : ℝ := 0.12) (expense_percent : ℝ := 0.18)
variable (selling_price : ℝ := 8.00)

theorem markup_rate (h1 : C + profit_percent * S + expense_percent * S = S)
                    (h2 : S = selling_price) :
  ((S - C) / C) * 100 = 42.86 := by
  sorry

end markup_rate_l141_141508


namespace gcd_p4_minus_1_eq_240_l141_141093

theorem gcd_p4_minus_1_eq_240 (p : ℕ) (hp : Prime p) (h_gt_5 : p > 5) :
  gcd (p^4 - 1) 240 = 240 :=
by sorry

end gcd_p4_minus_1_eq_240_l141_141093


namespace jane_babysitting_start_l141_141714

-- Definitions based on the problem conditions
def jane_current_age := 32
def years_since_babysitting := 10
def oldest_current_child_age := 24

-- Definition for the starting babysitting age
def starting_babysitting_age : ℕ := 8

-- Theorem statement to prove
theorem jane_babysitting_start (h1 : jane_current_age - years_since_babysitting = 22)
  (h2 : oldest_current_child_age - years_since_babysitting = 14)
  (h3 : ∀ (age_jane age_child : ℕ), age_child ≤ age_jane / 2) :
  starting_babysitting_age = 8 :=
by
  sorry

end jane_babysitting_start_l141_141714


namespace length_AC_correct_l141_141361

noncomputable def length_AC (A B C D : Type) : ℝ := 105 / 17

variable {A B C D : Type}
variables (angle_BAC angle_ADB length_AD length_BC : ℝ)

theorem length_AC_correct
  (h1 : angle_BAC = 60)
  (h2 : angle_ADB = 30)
  (h3 : length_AD = 3)
  (h4 : length_BC = 9) :
  length_AC A B C D = 105 / 17 :=
sorry

end length_AC_correct_l141_141361


namespace simplify_expression_l141_141143

theorem simplify_expression (z : ℝ) : (5 - 2*z^2) - (4*z^2 - 7) = 12 - 6*z^2 :=
by
  sorry

end simplify_expression_l141_141143


namespace tangent_parallel_line_coordinates_l141_141656

theorem tangent_parallel_line_coordinates :
  ∃ (m n : ℝ), 
    (∀ x : ℝ, (deriv (λ x => x^4 + x) x = 4 * x^3 + 1)) ∧ 
    (deriv (λ x => x^4 + x) m = -3) ∧ 
    (n = m^4 + m) ∧ 
    (m, n) = (-1, 0) :=
by
  sorry

end tangent_parallel_line_coordinates_l141_141656


namespace certain_number_unique_l141_141512

-- Define the necessary conditions and statement
def is_certain_number (n : ℕ) : Prop :=
  (∃ k : ℕ, 25 * k = n) ∧ (∃ k : ℕ, 35 * k = n) ∧ 
  (n > 0) ∧ (∃ a b c : ℕ, 1 ≤ a * n ∧ a * n ≤ 1050 ∧ 1 ≤ b * n ∧ b * n ≤ 1050 ∧ 1 ≤ c * n ∧ c * n ≤ 1050 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c)

theorem certain_number_unique :
  ∃ n : ℕ, is_certain_number n ∧ n = 350 :=
by 
  sorry

end certain_number_unique_l141_141512


namespace remainder_is_nine_l141_141024

-- Define the dividend and divisor
def n : ℕ := 4039
def d : ℕ := 31

-- Prove that n mod d equals 9
theorem remainder_is_nine : n % d = 9 := by
  sorry

end remainder_is_nine_l141_141024


namespace initial_kids_count_l141_141763

-- Define the initial number of kids as a variable
def init_kids (current_kids join_kids : Nat) : Nat :=
  current_kids - join_kids

-- Define the total current kids and kids joined
def current_kids : Nat := 36
def join_kids : Nat := 22

-- Prove that the initial number of kids was 14
theorem initial_kids_count : init_kids current_kids join_kids = 14 :=
by
  -- Proof skipped
  sorry

end initial_kids_count_l141_141763


namespace number_of_ensembles_sold_l141_141217

-- Define the prices
def necklace_price : ℕ := 25
def bracelet_price : ℕ := 15
def earring_price : ℕ := 10
def ensemble_price : ℕ := 45

-- Define the quantities sold
def necklaces_sold : ℕ := 5
def bracelets_sold : ℕ := 10
def earrings_sold : ℕ := 20

-- Define the total income
def total_income : ℕ := 565

-- Define the function or theorem that determines the number of ensembles sold
theorem number_of_ensembles_sold : 
  (total_income = (necklaces_sold * necklace_price) + (bracelets_sold * bracelet_price) + (earrings_sold * earring_price) + (2 * ensemble_price)) :=
sorry

end number_of_ensembles_sold_l141_141217


namespace roots_bounds_if_and_only_if_conditions_l141_141017

theorem roots_bounds_if_and_only_if_conditions (a b c : ℝ) (h : a > 0) (x1 x2 : ℝ) (hr : ∀ {x : ℝ}, a * x^2 + b * x + c = 0 → x = x1 ∨ x = x2) :
  (|x1| ≤ 1 ∧ |x2| ≤ 1) ↔ (a + b + c ≥ 0 ∧ a - b + c ≥ 0 ∧ a - c ≥ 0) :=
sorry

end roots_bounds_if_and_only_if_conditions_l141_141017


namespace car_travel_distance_l141_141529

noncomputable def distance_traveled (diameter : ℝ) (revolutions : ℝ) : ℝ :=
  let pi := Real.pi
  let circumference := pi * diameter
  circumference * revolutions / 12 / 5280

theorem car_travel_distance
  (diameter : ℝ)
  (revolutions : ℝ)
  (h_diameter : diameter = 13)
  (h_revolutions : revolutions = 775.5724667489372) :
  distance_traveled diameter revolutions = 0.5 :=
by
  simp [distance_traveled, h_diameter, h_revolutions, Real.pi]
  sorry

end car_travel_distance_l141_141529


namespace surfers_ratio_l141_141755

theorem surfers_ratio (S1 : ℕ) (S3 : ℕ) : S1 = 1500 → 
  (∀ S2 : ℕ, S2 = S1 + 600 → (1400 * 3 = S1 + S2 + S3) → 
  S3 = 600) → (S3 / S1 = 2 / 5) :=
sorry

end surfers_ratio_l141_141755


namespace find_m_l141_141281

noncomputable def g (n : ℤ) : ℤ :=
if n % 2 ≠ 0 then 2 * n + 3
else if n % 3 = 0 then n / 3
else n - 1

theorem find_m :
  ∃ m : ℤ, m % 2 ≠ 0 ∧ g (g (g m)) = 36 ∧ m = 54 :=
by
  sorry

end find_m_l141_141281


namespace veranda_width_l141_141455

def area_of_veranda (w : ℝ) : ℝ :=
  let room_area := 19 * 12
  let total_area := room_area + 140
  let total_length := 19 + 2 * w
  let total_width := 12 + 2 * w
  total_length * total_width - room_area

theorem veranda_width:
  ∃ w : ℝ, area_of_veranda w = 140 := by
  sorry

end veranda_width_l141_141455


namespace problem_statement_l141_141934

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b < a) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = |Real.log x|) (h_eq : f a = f b) :
  a * b = 1 ∧ Real.exp a + Real.exp b > 2 * Real.exp 1 ∧ (1 / a)^2 - b + 5 / 4 ≥ 1 :=
by
  sorry

end problem_statement_l141_141934


namespace cos_double_angle_l141_141888

open Real

theorem cos_double_angle (α : ℝ) (h : tan α = 3) : cos (2 * α) = -4 / 5 :=
sorry

end cos_double_angle_l141_141888


namespace total_sum_of_grid_is_745_l141_141447

theorem total_sum_of_grid_is_745 :
  let top_row := [12, 13, 15, 17, 19]
  let left_column := [12, 14, 16, 18]
  let total_sum := 360 + 375 + 10
  total_sum = 745 :=
by
  -- The theorem establishes the total sum calculation.
  sorry

end total_sum_of_grid_is_745_l141_141447


namespace binary_to_decimal_110_eq_6_l141_141279

theorem binary_to_decimal_110_eq_6 : (1 * 2^2 + 1 * 2^1 + 0 * 2^0 = 6) :=
by
  sorry

end binary_to_decimal_110_eq_6_l141_141279


namespace mul_99_105_l141_141369

theorem mul_99_105 : 99 * 105 = 10395 := 
by
  -- Annotations and imports are handled; only the final Lean statement provided as requested.
  sorry

end mul_99_105_l141_141369


namespace ratio_of_sector_CPD_l141_141300

-- Define the given angles
def angle_AOC : ℝ := 40
def angle_DOB : ℝ := 60
def angle_COP : ℝ := 110

-- Calculate the angle CPD
def angle_CPD : ℝ := angle_COP - angle_AOC - angle_DOB

-- State the theorem to prove the ratio
theorem ratio_of_sector_CPD (hAOC : angle_AOC = 40) (hDOB : angle_DOB = 60)
(hCOP : angle_COP = 110) : 
  angle_CPD / 360 = 1 / 36 := by
  -- Proof will go here
  sorry

end ratio_of_sector_CPD_l141_141300


namespace letters_with_line_not_dot_l141_141584

-- Defining the conditions
def num_letters_with_dot_and_line : ℕ := 9
def num_letters_with_dot_only : ℕ := 7
def total_letters : ℕ := 40

-- Proving the number of letters with a straight line but not a dot
theorem letters_with_line_not_dot :
  (num_letters_with_dot_and_line + num_letters_with_dot_only + x = total_letters) → x = 24 :=
by
  intros h
  sorry

end letters_with_line_not_dot_l141_141584


namespace lucy_found_shells_l141_141354

theorem lucy_found_shells (original current : ℕ) (h1 : original = 68) (h2 : current = 89) : current - original = 21 :=
by {
    sorry
}

end lucy_found_shells_l141_141354


namespace red_flower_ratio_l141_141243

theorem red_flower_ratio
  (total : ℕ)
  (O : ℕ)
  (P Pu : ℕ)
  (R Y : ℕ)
  (h_total : total = 105)
  (h_orange : O = 10)
  (h_pink_purple : P + Pu = 30)
  (h_equal_pink_purple : P = Pu)
  (h_yellow : Y = R - 5)
  (h_sum : R + Y + O + P + Pu = total) :
  (R / O) = 7 / 2 :=
by
  sorry

end red_flower_ratio_l141_141243


namespace find_g_2022_l141_141829

def g : ℝ → ℝ := sorry -- This is pre-defined to say there exists such a function

theorem find_g_2022 (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g (x - y) = g x + g y - 2021 * (x + y)) :
  g 2022 = 4086462 :=
sorry

end find_g_2022_l141_141829


namespace boys_count_l141_141518

-- Define the number of girls
def girls : ℕ := 635

-- Define the number of boys as being 510 more than the number of girls
def boys : ℕ := girls + 510

-- Prove that the number of boys in the school is 1145
theorem boys_count : boys = 1145 := by
  sorry

end boys_count_l141_141518


namespace national_park_sightings_l141_141010

def january_sightings : ℕ := 26

def february_sightings : ℕ := 3 * january_sightings

def march_sightings : ℕ := february_sightings / 2

def total_sightings : ℕ := january_sightings + february_sightings + march_sightings

theorem national_park_sightings : total_sightings = 143 := by
  sorry

end national_park_sightings_l141_141010


namespace women_with_fair_hair_percentage_l141_141066

theorem women_with_fair_hair_percentage
  (A : ℝ) (B : ℝ)
  (hA : A = 0.40)
  (hB : B = 0.25) :
  A * B = 0.10 := 
by
  rw [hA, hB]
  norm_num

end women_with_fair_hair_percentage_l141_141066


namespace inequality_solution_set_l141_141424

   theorem inequality_solution_set : 
     {x : ℝ | (4 * x - 5)^2 + (3 * x - 2)^2 < (x - 3)^2} = {x : ℝ | (2 / 3 : ℝ) < x ∧ x < (5 / 4 : ℝ)} :=
   by
     sorry
   
end inequality_solution_set_l141_141424


namespace player5_points_combination_l141_141798

theorem player5_points_combination :
  ∃ (two_point_shots three_pointers free_throws : ℕ), 
  (two_point_shots * 2 + three_pointers * 3 + free_throws * 1 = 14) :=
sorry

end player5_points_combination_l141_141798


namespace train_speed_l141_141002

theorem train_speed (length_train length_bridge time : ℝ) (h_train : length_train = 125) (h_bridge : length_bridge = 250) (h_time : time = 30) :
    (length_train + length_bridge) / time * 3.6 = 45 := by
  sorry

end train_speed_l141_141002


namespace total_students_correct_l141_141313

def num_boys : ℕ := 272
def num_girls : ℕ := num_boys + 106
def total_students : ℕ := num_boys + num_girls

theorem total_students_correct : total_students = 650 :=
by
  sorry

end total_students_correct_l141_141313


namespace smallest_integer_x_l141_141272

theorem smallest_integer_x (x : ℤ) : (x^2 - 11 * x + 24 < 0) → x ≥ 4 ∧ x < 8 :=
by
sorry

end smallest_integer_x_l141_141272


namespace quadrilateral_possible_with_2_2_2_l141_141187

theorem quadrilateral_possible_with_2_2_2 :
  ∀ (s1 s2 s3 s4 : ℕ), (s1 = 2) → (s2 = 2) → (s3 = 2) → (s4 = 5) →
  s1 + s2 + s3 > s4 :=
by
  intros s1 s2 s3 s4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Proof omitted
  sorry

end quadrilateral_possible_with_2_2_2_l141_141187


namespace volleyball_team_selection_l141_141854

open Nat

def binom (n k : ℕ) : ℕ :=
  if k > n then 0
  else (n.choose k)

theorem volleyball_team_selection : 
  let quadruplets := ["Bella", "Bianca", "Becca", "Brooke"];
  let total_players := 16;
  let starters := 7;
  let num_quadruplets := quadruplets.length;
  ∃ ways : ℕ, 
    ways = binom num_quadruplets 3 * binom (total_players - num_quadruplets) (starters - 3) 
    ∧ ways = 1980 :=
by
  sorry

end volleyball_team_selection_l141_141854


namespace apartment_building_count_l141_141345

theorem apartment_building_count 
  (floors_per_building : ℕ) 
  (apartments_per_floor : ℕ) 
  (doors_per_apartment : ℕ) 
  (total_doors_needed : ℕ) 
  (doors_per_building : ℕ) 
  (number_of_buildings : ℕ)
  (h1 : floors_per_building = 12)
  (h2 : apartments_per_floor = 6) 
  (h3 : doors_per_apartment = 7) 
  (h4 : total_doors_needed = 1008) 
  (h5 : doors_per_building = apartments_per_floor * doors_per_apartment * floors_per_building)
  (h6 : number_of_buildings = total_doors_needed / doors_per_building) : 
  number_of_buildings = 2 := 
by 
  rw [h1, h2, h3] at h5 
  rw [h5, h4] at h6 
  exact h6

end apartment_building_count_l141_141345


namespace ratio_pentagon_rectangle_l141_141207

-- Definitions of conditions.
def pentagon_side_length (p : ℕ) : Prop := 5 * p = 30
def rectangle_width (w : ℕ) : Prop := 6 * w = 30

-- The theorem to prove.
theorem ratio_pentagon_rectangle (p w : ℕ) (h1 : pentagon_side_length p) (h2 : rectangle_width w) :
  p / w = 6 / 5 :=
by sorry

end ratio_pentagon_rectangle_l141_141207


namespace division_of_fractions_l141_141847

theorem division_of_fractions : (2 / 3) / (1 / 4) = (8 / 3) := by
  sorry

end division_of_fractions_l141_141847


namespace total_cost_l141_141456

-- Definition: Cost of first 100 notebooks
def cost_first_100_notebooks : ℕ := 230

-- Definition: Cost per notebook beyond the first 100 notebooks
def cost_additional_notebooks (n : ℕ) : ℕ := n * 2

-- Theorem: Total cost given a > 100 notebooks
theorem total_cost (a : ℕ) (h : a > 100) : (cost_first_100_notebooks + cost_additional_notebooks (a - 100) = 2 * a + 30) := by
  sorry

end total_cost_l141_141456


namespace prove_value_l141_141902

variable (m n : ℤ)

-- Conditions from the problem
def condition1 : Prop := m^2 + 2 * m * n = 384
def condition2 : Prop := 3 * m * n + 2 * n^2 = 560

-- Proposition to be proved
theorem prove_value (h1 : condition1 m n) (h2 : condition2 m n) : 2 * m^2 + 13 * m * n + 6 * n^2 - 444 = 2004 := by
  sorry

end prove_value_l141_141902


namespace students_with_same_grade_l141_141370

theorem students_with_same_grade :
  let total_students := 40
  let students_with_same_A := 3
  let students_with_same_B := 2
  let students_with_same_C := 6
  let students_with_same_D := 1
  let total_same_grade_students := students_with_same_A + students_with_same_B + students_with_same_C + students_with_same_D
  total_same_grade_students = 12 →
  (total_same_grade_students / total_students) * 100 = 30 :=
by
  sorry

end students_with_same_grade_l141_141370


namespace flea_jump_no_lava_l141_141233

theorem flea_jump_no_lava
  (A B F : ℕ)
  (n : ℕ) 
  (h_posA : 0 < A)
  (h_posB : 0 < B)
  (h_AB : A < B)
  (h_2A : B < 2 * A)
  (h_ineq1 : A * (n + 1) ≤ B - A * n)
  (h_ineq2 : B - A < A * n) :
  ∃ (F : ℕ), F = (n - 1) * A + B := sorry

end flea_jump_no_lava_l141_141233


namespace students_per_class_l141_141149

variable (c : ℕ) (s : ℕ)

def books_per_month := 6
def months_per_year := 12
def books_per_year := books_per_month * months_per_year
def total_books_read := 72

theorem students_per_class : (s * c = 1 ∧ s * books_per_year = total_books_read) → s = 1 := by
  intros h
  have h1: books_per_year = total_books_read := by
    calc
      books_per_year = books_per_month * months_per_year := rfl
      _ = 6 * 12 := rfl
      _ = 72 := rfl
  sorry

end students_per_class_l141_141149


namespace isosceles_trapezoid_area_l141_141425

theorem isosceles_trapezoid_area (m h : ℝ) (hg : h = 3) (mg : m = 15) : 
  (m * h = 45) :=
by
  simp [hg, mg]
  sorry

end isosceles_trapezoid_area_l141_141425


namespace tree_planting_problem_l141_141444

noncomputable def total_trees_needed (length width tree_distance : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let intervals := perimeter / tree_distance
  intervals

theorem tree_planting_problem : total_trees_needed 150 60 10 = 42 :=
by
  sorry

end tree_planting_problem_l141_141444


namespace race_runners_l141_141325

theorem race_runners (k : ℕ) (h1 : 2*(k - 1) = k - 1) (h2 : 2*(2*(k + 9) - 12) = k + 9) : 3*k - 2 = 31 :=
by
  sorry

end race_runners_l141_141325


namespace range_of_a_l141_141883

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 2 * x - a

theorem range_of_a (a : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ a > 2 - 2 * Real.log 2 :=
by
  sorry

end range_of_a_l141_141883


namespace maximize_triangle_areas_l141_141179

theorem maximize_triangle_areas (L W : ℝ) (h1 : 2 * L + 2 * W = 80) (h2 : L ≤ 25) : W = 15 :=
by 
  sorry

end maximize_triangle_areas_l141_141179


namespace find_xy_l141_141863

variable (x y : ℝ)

theorem find_xy (hx : x ≠ 0) (hy : y ≠ 0) (h : (3 / x) * (2 / y) = 1 / 3) : x * y = 18 := by
  sorry

end find_xy_l141_141863


namespace proof_problem_l141_141168

-- Given definitions
def A := { y : ℝ | ∃ x : ℝ, y = x^2 + 1 }
def B := { p : ℝ × ℝ | ∃ x : ℝ, p.snd = x^2 + 1 }

-- Theorem to prove 1 ∉ B and 2 ∈ A
theorem proof_problem : 1 ∉ B ∧ 2 ∈ A :=
by
  sorry

end proof_problem_l141_141168


namespace evaluate_expression_l141_141445

theorem evaluate_expression : 4 * (9 - 3)^2 - 8 = 136 := by
  sorry

end evaluate_expression_l141_141445


namespace solve_problem_l141_141274

def problem_statement : Prop := (245245 % 35 = 0)

theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l141_141274


namespace lemon_pie_degrees_l141_141648

def total_students : ℕ := 45
def chocolate_pie_students : ℕ := 15
def apple_pie_students : ℕ := 10
def blueberry_pie_students : ℕ := 7
def cherry_and_lemon_students := total_students - (chocolate_pie_students + apple_pie_students + blueberry_pie_students)
def lemon_pie_students := cherry_and_lemon_students / 2

theorem lemon_pie_degrees (students_nonnegative : lemon_pie_students ≥ 0) (students_rounding : lemon_pie_students = 7) :
  (lemon_pie_students * 360 / total_students) = 56 := 
by
  -- Proof to be provided
  sorry

end lemon_pie_degrees_l141_141648


namespace intersection_is_solution_l141_141334

theorem intersection_is_solution (a b : ℝ) :
  (b = 3 * a + 6 ∧ b = 2 * a - 4) ↔ (3 * a - b = -6 ∧ 2 * a - b = 4) := 
by sorry

end intersection_is_solution_l141_141334


namespace smallest_fraction_divides_exactly_l141_141673

theorem smallest_fraction_divides_exactly (a b c p q r m n : ℕ)
    (h1: a = 6) (h2: b = 5) (h3: c = 10) (h4: p = 7) (h5: q = 14) (h6: r = 21)
    (h1_frac: 6/7 = a/p) (h2_frac: 5/14 = b/q) (h3_frac: 10/21 = c/r)
    (h_lcm: m = Nat.lcm p (Nat.lcm q r)) (h_gcd: n = Nat.gcd a (Nat.gcd b c)) :
  (n/m) = 1/42 :=
by 
  sorry

end smallest_fraction_divides_exactly_l141_141673


namespace h_inverse_left_h_inverse_right_l141_141104

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x - 1
noncomputable def h (x : ℝ) : ℝ := f (g x)
noncomputable def h_inv (y : ℝ) : ℝ := 1 + (Real.sqrt (3 * y + 12)) / 4 -- Correct answer

-- Theorem statements to prove the inverse relationship
theorem h_inverse_left (x : ℝ) : h (h_inv x) = x :=
by
  sorry -- Proof of the left inverse

theorem h_inverse_right (y : ℝ) : h_inv (h y) = y :=
by
  sorry -- Proof of the right inverse

end h_inverse_left_h_inverse_right_l141_141104


namespace medial_triangle_AB_AC_BC_l141_141400

theorem medial_triangle_AB_AC_BC
  (l m n : ℝ)
  (A B C : Type)
  (midpoint_BC := (l, 0, 0))
  (midpoint_AC := (0, m, 0))
  (midpoint_AB := (0, 0, n)) :
  (AB^2 + AC^2 + BC^2) / (l^2 + m^2 + n^2) = 8 :=
by
  sorry

end medial_triangle_AB_AC_BC_l141_141400


namespace mean_eq_value_of_z_l141_141875

theorem mean_eq_value_of_z (z : ℤ) : 
  ((6 + 15 + 9 + 20) / 4 : ℚ) = ((13 + z) / 2 : ℚ) → (z = 12) := by
  sorry

end mean_eq_value_of_z_l141_141875


namespace derivative_of_y_l141_141422

variable (a b c x : ℝ)

def y : ℝ := (x - a) * (x - b) * (x - c)

theorem derivative_of_y :
  deriv (fun x:ℝ => (x - a) * (x - b) * (x - c)) x = 3 * x^2 - 2 * (a + b + c) * x + (a * b + a * c + b * c) :=
by
  sorry

end derivative_of_y_l141_141422


namespace values_of_k_for_exactly_one_real_solution_l141_141672

variable {k : ℝ}

def quadratic_eq (k : ℝ) : Prop := 3 * k^2 + 42 * k - 573 = 0

theorem values_of_k_for_exactly_one_real_solution :
  quadratic_eq k ↔ k = 8 ∨ k = -22 := by
  sorry

end values_of_k_for_exactly_one_real_solution_l141_141672


namespace kate_money_left_l141_141916

def kate_savings_march := 27
def kate_savings_april := 13
def kate_savings_may := 28
def kate_expenditure_keyboard := 49
def kate_expenditure_mouse := 5

def total_savings := kate_savings_march + kate_savings_april + kate_savings_may
def total_expenditure := kate_expenditure_keyboard + kate_expenditure_mouse
def money_left := total_savings - total_expenditure

-- Prove that Kate has $14 left
theorem kate_money_left : money_left = 14 := 
by 
  sorry

end kate_money_left_l141_141916


namespace minimum_effort_to_qualify_l141_141194

def minimum_effort_to_qualify_for_mop (AMC_points_per_effort : ℕ := 6 * 1/3)
                                       (AIME_points_per_effort : ℕ := 10 * 1/7)
                                       (USAMO_points_per_effort : ℕ := 1 * 1/10)
                                       (required_amc_aime_points : ℕ := 200)
                                       (required_usamo_points : ℕ := 21) : ℕ :=
  let max_amc_points : ℕ := 150
  let effort_amc : ℕ := (max_amc_points / AMC_points_per_effort) * 3
  let remaining_aime_points : ℕ := 200 - max_amc_points
  let effort_aime : ℕ := (remaining_aime_points / AIME_points_per_effort) * 7
  let effort_usamo : ℕ := required_usamo_points * 10
  let total_effort : ℕ := effort_amc + effort_aime + effort_usamo
  total_effort

theorem minimum_effort_to_qualify : minimum_effort_to_qualify_for_mop 6 (10 * 1/7) (1 * 1/10) 200 21 = 320 := by
  sorry

end minimum_effort_to_qualify_l141_141194


namespace range_of_a_increasing_f_on_interval_l141_141836

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- Define the condition that f(x) is increasing on [4, +∞)
def isIncreasingOnInterval (a : ℝ) : Prop :=
  ∀ x y : ℝ, 4 ≤ x → x ≤ y → f a x ≤ f a y

theorem range_of_a_increasing_f_on_interval :
  (∀ a : ℝ, isIncreasingOnInterval a → a ≥ -3) := 
by
  sorry

end range_of_a_increasing_f_on_interval_l141_141836


namespace no_solution_eq_l141_141950

theorem no_solution_eq (k : ℝ) :
  (¬ ∃ x : ℝ, x ≠ 3 ∧ x ≠ 7 ∧ (x + 2) / (x - 3) = (x - k) / (x - 7)) ↔ k = 2 :=
by
  sorry

end no_solution_eq_l141_141950


namespace side_length_of_S2_l141_141595

theorem side_length_of_S2 :
  ∀ (r s : ℕ), 
    (2 * r + s = 2000) → 
    (2 * r + 5 * s = 3030) → 
    s = 258 :=
by
  intros r s h1 h2
  sorry

end side_length_of_S2_l141_141595


namespace base_five_to_decimal_l141_141698

def base5_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 2 => 2 * 5^0
  | 3 => 3 * 5^1
  | 1 => 1 * 5^2
  | _ => 0

theorem base_five_to_decimal : base5_to_base10 2 + base5_to_base10 3 + base5_to_base10 1 = 42 :=
by sorry

end base_five_to_decimal_l141_141698


namespace division_remainder_l141_141933

theorem division_remainder (dividend quotient divisor remainder : ℕ) 
  (h_dividend : dividend = 12401) 
  (h_quotient : quotient = 76) 
  (h_divisor : divisor = 163) 
  (h_remainder : dividend = quotient * divisor + remainder) : 
  remainder = 13 := 
by
  sorry

end division_remainder_l141_141933


namespace number_of_observations_is_14_l141_141190

theorem number_of_observations_is_14
  (mean_original : ℚ) (mean_new : ℚ) (original_sum : ℚ) 
  (corrected_sum : ℚ) (n : ℚ)
  (h1 : mean_original = 36)
  (h2 : mean_new = 36.5)
  (h3 : corrected_sum = original_sum + 7)
  (h4 : mean_new = corrected_sum / n)
  (h5 : original_sum = mean_original * n) :
  n = 14 :=
by
  -- Here goes the proof
  sorry

end number_of_observations_is_14_l141_141190


namespace fraction_solution_l141_141893

theorem fraction_solution (a : ℕ) (h : a > 0) (h_eq : (a : ℚ) / (a + 45) = 0.75) : a = 135 :=
sorry

end fraction_solution_l141_141893


namespace loan_interest_rate_l141_141520

theorem loan_interest_rate (P SI T R : ℕ) (h1 : P = 900) (h2 : SI = 729) (h3 : T = R) :
  (SI = (P * R * T) / 100) -> R = 9 :=
by
  sorry

end loan_interest_rate_l141_141520


namespace solution_set_of_inequality_l141_141227

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

end solution_set_of_inequality_l141_141227


namespace add_coefficients_l141_141275

theorem add_coefficients (a : ℕ) : 2 * a + a = 3 * a :=
by 
  sorry

end add_coefficients_l141_141275


namespace horizontal_force_magnitude_l141_141688

-- We state our assumptions and goal
theorem horizontal_force_magnitude (W : ℝ) : 
  (∀ μ : ℝ, μ = (Real.sin (Real.pi / 6)) / (Real.cos (Real.pi / 6)) ∧ 
    (∀ P : ℝ, 
      (P * (Real.sin (Real.pi / 3))) = 
      ((μ * (W * (Real.cos (Real.pi / 6)) + P * (Real.cos (Real.pi / 3)))) + W * (Real.sin (Real.pi / 6))) →
      P = W * Real.sqrt 3)) :=
sorry

end horizontal_force_magnitude_l141_141688


namespace heartsuit_xx_false_l141_141364

def heartsuit (x y : ℝ) : ℝ := |x - y|

theorem heartsuit_xx_false (x : ℝ) : heartsuit x x ≠ x :=
by sorry

end heartsuit_xx_false_l141_141364


namespace lateral_surface_area_of_cylinder_l141_141942

variable (m n : ℝ) (S : ℝ)

theorem lateral_surface_area_of_cylinder (h1 : S > 0) (h2 : m > 0) (h3 : n > 0) :
  ∃ (lateral_surface_area : ℝ),
    lateral_surface_area = (π * S) / (Real.sin (π * n / (m + n))) :=
sorry

end lateral_surface_area_of_cylinder_l141_141942


namespace unattainable_y_l141_141350

theorem unattainable_y (x : ℝ) (h : 4 * x + 5 ≠ 0) : 
  (y = (3 - x) / (4 * x + 5)) → (y ≠ -1/4) :=
sorry

end unattainable_y_l141_141350


namespace find_angle_C_find_sum_a_b_l141_141727

noncomputable def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  c = 7 / 2 ∧
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 ∧
  (Real.tan A + Real.tan B = Real.sqrt 3 * (Real.tan A * Real.tan B - 1))

theorem find_angle_C (a b c A B C : ℝ) (h : triangle_condition a b c A B C) : C = Real.pi / 3 :=
  sorry

theorem find_sum_a_b (a b c A B C : ℝ) (h : triangle_condition a b c A B C) (hC : C = Real.pi / 3) : a + b = 11 / 2 :=
  sorry

end find_angle_C_find_sum_a_b_l141_141727


namespace multiples_of_4_between_50_and_300_l141_141278

theorem multiples_of_4_between_50_and_300 : 
  (∃ n : ℕ, 50 < n ∧ n < 300 ∧ n % 4 = 0) ∧ 
  (∃ k : ℕ, k = 62) :=
by
  sorry

end multiples_of_4_between_50_and_300_l141_141278


namespace heracles_age_l141_141591

theorem heracles_age
  (H : ℕ)
  (audrey_current_age : ℕ)
  (audrey_in_3_years : ℕ)
  (h1 : audrey_current_age = H + 7)
  (h2 : audrey_in_3_years = audrey_current_age + 3)
  (h3 : audrey_in_3_years = 2 * H)
  : H = 10 :=
by
  sorry

end heracles_age_l141_141591


namespace intersection_is_correct_l141_141301

noncomputable def A := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
noncomputable def B := { x : ℝ | 0 < x ∧ x ≤ 3 }

theorem intersection_is_correct : 
  (A ∩ B) = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_is_correct_l141_141301


namespace ratio_sub_div_a_l141_141415

theorem ratio_sub_div_a (a b : ℝ) (h : a / b = 5 / 8) : (b - a) / a = 3 / 5 :=
sorry

end ratio_sub_div_a_l141_141415


namespace necessary_and_sufficient_condition_l141_141917

theorem necessary_and_sufficient_condition (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 2 ∧ x2 < 2 ∧ x1^2 - m * x1 - 1 = 0 ∧ x2^2 - m * x2 - 1 = 0) ↔ m > 1.5 :=
by
  sorry

end necessary_and_sufficient_condition_l141_141917


namespace tram_speed_l141_141241

variables (V : ℝ)

theorem tram_speed (h : (V + 5) / (V - 5) = 600 / 225) : V = 11 :=
sorry

end tram_speed_l141_141241


namespace no_real_solution_for_inequality_l141_141432

theorem no_real_solution_for_inequality :
  ∀ x : ℝ, ¬(3 * x^2 - x + 2 < 0) :=
by
  sorry

end no_real_solution_for_inequality_l141_141432


namespace temp_interpretation_l141_141904

theorem temp_interpretation (below_zero : ℤ) (above_zero : ℤ) (h : below_zero = -2):
  above_zero = 3 → 3 = 0 := by
  intro h2
  have : above_zero = 3 := h2
  sorry

end temp_interpretation_l141_141904


namespace pentagon_area_l141_141658

-- Definitions of the side lengths of the pentagon
def side1 : ℕ := 12
def side2 : ℕ := 17
def side3 : ℕ := 25
def side4 : ℕ := 18
def side5 : ℕ := 17

-- Definitions for the rectangle and triangle dimensions
def rectangle_width : ℕ := side4
def rectangle_height : ℕ := side1
def triangle_base : ℕ := side4
def triangle_height : ℕ := side3 - side1

-- The area of the pentagon proof statement
theorem pentagon_area : rectangle_width * rectangle_height +
    (triangle_base * triangle_height) / 2 = 333 := by
  sorry

end pentagon_area_l141_141658


namespace find_m_l141_141606

theorem find_m (m : ℝ) (h1 : |m - 3| = 4) (h2 : m - 7 ≠ 0) : m = -1 :=
sorry

end find_m_l141_141606


namespace ordered_pairs_count_l141_141543

theorem ordered_pairs_count : ∃ (count : ℕ), count = 4 ∧
  ∀ (m n : ℕ), m > 0 → n > 0 → m ≥ n → m^2 - n^2 = 144 → (∃ (i : ℕ), i < count) := by
  sorry

end ordered_pairs_count_l141_141543


namespace james_total_distance_l141_141477

structure Segment where
  speed : ℝ -- speed in mph
  time : ℝ -- time in hours

def totalDistance (segments : List Segment) : ℝ :=
  segments.foldr (λ seg acc => seg.speed * seg.time + acc) 0

theorem james_total_distance :
  let segments := [
    Segment.mk 30 0.5,
    Segment.mk 60 0.75,
    Segment.mk 75 1.5,
    Segment.mk 60 2
  ]
  totalDistance segments = 292.5 :=
by
  sorry

end james_total_distance_l141_141477


namespace age_difference_l141_141581

variable (S M : ℕ)

theorem age_difference (hS : S = 28) (hM : M + 2 = 2 * (S + 2)) : M - S = 30 :=
by
  sorry

end age_difference_l141_141581


namespace inequality_proof_l141_141597

theorem inequality_proof
  (p q a b c d e : Real)
  (hpq : 0 < p ∧ p ≤ a ∧ p ≤ b ∧ p ≤ c ∧ p ≤ d ∧ p ≤ e)
  (hq : a ≤ q ∧ b ≤ q ∧ c ≤ q ∧ d ≤ q ∧ e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e)
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p)) ^ 2 :=
sorry

end inequality_proof_l141_141597


namespace total_money_in_wallet_l141_141530

-- Definitions of conditions
def initial_five_dollar_bills := 7
def initial_ten_dollar_bills := 1
def initial_twenty_dollar_bills := 3
def initial_fifty_dollar_bills := 1
def initial_one_dollar_coins := 8

def spent_groceries := 65
def paid_fifty_dollar_bill := 1
def paid_twenty_dollar_bill := 1
def received_five_dollar_bill_change := 1
def received_one_dollar_coin_change := 5

def received_twenty_dollar_bills_from_friend := 2
def received_one_dollar_bills_from_friend := 2

-- Proving total amount of money
theorem total_money_in_wallet : 
  initial_five_dollar_bills * 5 + 
  initial_ten_dollar_bills * 10 + 
  initial_twenty_dollar_bills * 20 + 
  initial_fifty_dollar_bills * 50 + 
  initial_one_dollar_coins * 1 - 
  spent_groceries + 
  received_five_dollar_bill_change * 5 + 
  received_one_dollar_coin_change * 1 + 
  received_twenty_dollar_bills_from_friend * 20 + 
  received_one_dollar_bills_from_friend * 1 
  = 150 := 
by
  -- This is where the proof would be located
  sorry

end total_money_in_wallet_l141_141530


namespace evaluate_g_at_8_l141_141332

def g (n : ℕ) : ℕ := n^2 - 3 * n + 29

theorem evaluate_g_at_8 : g 8 = 69 := by
  unfold g
  calc
    8^2 - 3 * 8 + 29 = 64 - 24 + 29 := by simp
                      _ = 69 := by norm_num

end evaluate_g_at_8_l141_141332


namespace range_of_a_l141_141560

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, |x - a| + |x - 1| ≤ 4) : -3 ≤ a ∧ a ≤ 5 := 
sorry

end range_of_a_l141_141560


namespace usual_time_to_office_l141_141218

theorem usual_time_to_office
  (S T : ℝ) 
  (h1 : ∀ D : ℝ, D = S * T)
  (h2 : ∀ D : ℝ, D = (4 / 5) * S * (T + 10)):
  T = 40 := 
by
  sorry

end usual_time_to_office_l141_141218


namespace euler_totient_inequality_l141_141749

variable {n : ℕ}
def even (n : ℕ) := ∃ k : ℕ, n = 2 * k
def positive (n : ℕ) := n > 0

theorem euler_totient_inequality (h_even : even n) (h_positive : positive n) : 
  Nat.totient n ≤ n / 2 :=
sorry

end euler_totient_inequality_l141_141749


namespace sum_of_ages_l141_141589

def Tyler_age : ℕ := 5

def Clay_age (T C : ℕ) : Prop :=
  T = 3 * C + 1

theorem sum_of_ages (C : ℕ) (h : Clay_age Tyler_age C) :
  Tyler_age + C = 6 :=
sorry

end sum_of_ages_l141_141589


namespace cost_of_items_l141_141674

variable (p q r : ℝ)

theorem cost_of_items :
  8 * p + 2 * q + r = 4.60 → 
  2 * p + 5 * q + r = 3.90 → 
  p + q + 3 * r = 2.75 → 
  4 * p + 3 * q + 2 * r = 7.4135 :=
by
  intros h1 h2 h3
  sorry

end cost_of_items_l141_141674


namespace observed_wheels_l141_141468

theorem observed_wheels (num_cars wheels_per_car : ℕ) (h1 : num_cars = 12) (h2 : wheels_per_car = 4) : num_cars * wheels_per_car = 48 := by
  sorry

end observed_wheels_l141_141468


namespace inequality_solution_l141_141647

theorem inequality_solution (x : ℝ) :
  (abs ((x^2 - 5 * x + 4) / 3) < 1) ↔ 
  ((5 - Real.sqrt 21) / 2 < x) ∧ (x < (5 + Real.sqrt 21) / 2) := 
sorry

end inequality_solution_l141_141647


namespace average_of_class_is_49_5_l141_141949

noncomputable def average_score_of_class : ℝ :=
  let total_students := 50
  let students_95 := 5
  let students_0 := 5
  let students_85 := 5
  let remaining_students := total_students - (students_95 + students_0 + students_85)
  let total_marks := (students_95 * 95) + (students_0 * 0) + (students_85 * 85) + (remaining_students * 45)
  total_marks / total_students

theorem average_of_class_is_49_5 : average_score_of_class = 49.5 := 
by sorry

end average_of_class_is_49_5_l141_141949


namespace contractor_fired_people_l141_141221

theorem contractor_fired_people :
  ∀ (total_days : ℕ) (initial_people : ℕ) (partial_days : ℕ) 
    (partial_work_fraction : ℚ) (remaining_days : ℕ) 
    (fired_people : ℕ),
  total_days = 100 →
  initial_people = 10 →
  partial_days = 20 →
  partial_work_fraction = 1 / 4 →
  remaining_days = 75 →
  (initial_people - fired_people) * remaining_days * (1 - partial_work_fraction) / partial_days = initial_people * total_days →
  fired_people = 2 :=
by
  intros total_days initial_people partial_days partial_work_fraction remaining_days fired_people
  intro h1 h2 h3 h4 h5 h6
  sorry

end contractor_fired_people_l141_141221


namespace ineq_one_of_two_sqrt_amgm_l141_141197

-- Lean 4 statement for Question 1
theorem ineq_one_of_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
sorry

-- Lean 4 statement for Question 2
theorem sqrt_amgm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 :=
sorry

end ineq_one_of_two_sqrt_amgm_l141_141197


namespace change_combinations_12_dollars_l141_141992

theorem change_combinations_12_dollars :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), 
  (∀ (n d q : ℕ), (n, d, q) ∈ solutions ↔ 5 * n + 10 * d + 25 * q = 1200 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1) ∧ solutions.card = 61 :=
sorry

end change_combinations_12_dollars_l141_141992


namespace books_inequality_system_l141_141737

theorem books_inequality_system (x : ℕ) (n : ℕ) (h1 : x = 5 * n + 6) (h2 : (1 ≤ x - 7 * (x - 6) / 5 + 7)) :
  1 ≤ x - 7 * (x - 6) / 5 + 7 ∧ x - 7 * (x - 6) / 5 + 7 < 7 := 
by
  sorry

end books_inequality_system_l141_141737


namespace two_digit_number_solution_l141_141978

theorem two_digit_number_solution (N : ℕ) (x y : ℕ) :
  (10 * x + y = N) ∧ (4 * x + 2 * y = (10 * x + y) / 2) →
  N = 32 ∨ N = 64 ∨ N = 96 := 
sorry

end two_digit_number_solution_l141_141978


namespace a_sufficient_not_necessary_for_a_squared_eq_b_squared_l141_141157

theorem a_sufficient_not_necessary_for_a_squared_eq_b_squared
  (a b : ℝ) :
  (a = b) → (a^2 = b^2) ∧ ¬ ((a^2 = b^2) → (a = b)) :=
  sorry

end a_sufficient_not_necessary_for_a_squared_eq_b_squared_l141_141157


namespace find_value_of_expression_l141_141599

theorem find_value_of_expression (m : ℝ) (h_m : m^2 - 3 * m + 1 = 0) : 2 * m^2 - 6 * m - 2024 = -2026 := by
  sorry

end find_value_of_expression_l141_141599


namespace fifth_derivative_l141_141134

noncomputable def f (x : ℝ) : ℝ := (2 * x ^ 2 - 7) * Real.log (x - 1)

theorem fifth_derivative :
  ∀ x, (deriv^[5] f) x = 8 * (x ^ 2 - 5 * x - 11) / ((x - 1) ^ 5) :=
by
  sorry

end fifth_derivative_l141_141134


namespace range_of_a_l141_141864

noncomputable def f (a x : ℝ) :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x - a) ∧
  (∀ x y : ℝ, x < y → f a x ≥ f a y) →
  1 / 7 ≤ a ∧ a < 1 / 3 := 
sorry

end range_of_a_l141_141864


namespace find_AX_bisect_ACB_l141_141101

theorem find_AX_bisect_ACB (AC BX BC : ℝ) (h₁ : AC = 21) (h₂ : BX = 28) (h₃ : BC = 30) :
  ∃ (AX : ℝ), AX = 98 / 5 :=
by
  existsi 98 / 5
  sorry

end find_AX_bisect_ACB_l141_141101


namespace find_triples_tan_l141_141986

open Real

theorem find_triples_tan (x y z : ℝ) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z → 
  ∃ (A B C : ℝ), x = tan A ∧ y = tan B ∧ z = tan C :=
by
  sorry

end find_triples_tan_l141_141986


namespace polygon_sides_eq_eleven_l141_141439

theorem polygon_sides_eq_eleven (n : ℕ) (D : ℕ)
(h1 : D = n + 33)
(h2 : D = n * (n - 3) / 2) :
  n = 11 :=
by {
  sorry
}

end polygon_sides_eq_eleven_l141_141439


namespace max_individual_score_l141_141115

open Nat

theorem max_individual_score (n : ℕ) (total_points : ℕ) (minimum_points : ℕ) (H1 : n = 12) (H2 : total_points = 100) (H3 : ∀ i : Fin n, 7 ≤ minimum_points) :
  ∃ max_points : ℕ, max_points = 23 :=
by 
  sorry

end max_individual_score_l141_141115


namespace samantha_sleep_hours_l141_141127

def time_in_hours (hours minutes : ℕ) : ℕ :=
  hours + (minutes / 60)

def hours_slept (bed_time wake_up_time : ℕ) : ℕ :=
  if bed_time < wake_up_time then wake_up_time - bed_time + 12 else 24 - bed_time + wake_up_time

theorem samantha_sleep_hours : hours_slept 7 11 = 16 := by
  sorry

end samantha_sleep_hours_l141_141127


namespace paul_diner_total_cost_l141_141009

/-- At Paul's Diner, sandwiches cost $5 each and sodas cost $3 each. If a customer buys
more than 4 sandwiches, they receive a $10 discount on the total bill. Calculate the total
cost if a customer purchases 6 sandwiches and 3 sodas. -/
def totalCost (num_sandwiches num_sodas : ℕ) : ℕ :=
  let sandwich_cost := 5
  let soda_cost := 3
  let discount := if num_sandwiches > 4 then 10 else 0
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) - discount

theorem paul_diner_total_cost : totalCost 6 3 = 29 :=
by
  sorry

end paul_diner_total_cost_l141_141009


namespace josh_and_fred_age_l141_141006

theorem josh_and_fred_age
    (a b k : ℕ)
    (h1 : 10 * a + b > 10 * b + a)
    (h2 : 99 * (a^2 - b^2) = k^2)
    (ha : a ≥ 0 ∧ a ≤ 9)
    (hb : b ≥ 0 ∧ b ≤ 9) : 
    10 * a + b = 65 ∧ 
    10 * b + a = 56 := 
sorry

end josh_and_fred_age_l141_141006


namespace given_cond_then_geq_eight_l141_141382

theorem given_cond_then_geq_eight (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 1) : 
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 := 
  sorry

end given_cond_then_geq_eight_l141_141382


namespace arithmetic_sum_expression_zero_l141_141965

theorem arithmetic_sum_expression_zero (a d : ℤ) (i j k : ℕ) (S_i S_j S_k : ℤ) :
  S_i = i * (a + (i - 1) * d / 2) →
  S_j = j * (a + (j - 1) * d / 2) →
  S_k = k * (a + (k - 1) * d / 2) →
  (S_i / i * (j - k) + S_j / j * (k - i) + S_k / k * (i - j) = 0) :=
by
  intros hS_i hS_j hS_k
  -- Proof omitted
  sorry

end arithmetic_sum_expression_zero_l141_141965


namespace water_content_in_boxes_l141_141398

noncomputable def totalWaterInBoxes (num_boxes : ℕ) (bottles_per_box : ℕ) (capacity_per_bottle : ℚ) (fill_fraction : ℚ) : ℚ :=
  num_boxes * bottles_per_box * capacity_per_bottle * fill_fraction

theorem water_content_in_boxes :
  totalWaterInBoxes 10 50 12 (3 / 4) = 4500 := 
by
  sorry

end water_content_in_boxes_l141_141398


namespace custom_op_12_7_l141_141815

def custom_op (a b : ℤ) := (a + b) * (a - b)

theorem custom_op_12_7 : custom_op 12 7 = 95 := by
  sorry

end custom_op_12_7_l141_141815


namespace parameterization_of_line_l141_141701

theorem parameterization_of_line (t : ℝ) (g : ℝ → ℝ) 
  (h : ∀ t, (g t - 10) / 2 = t ) :
  g t = 5 * t + 10 := by
  sorry

end parameterization_of_line_l141_141701


namespace terminating_decimal_expansion_of_17_div_200_l141_141252

theorem terminating_decimal_expansion_of_17_div_200 :
  (17 / 200 : ℚ) = 34 / 10000 := sorry

end terminating_decimal_expansion_of_17_div_200_l141_141252


namespace min_area_rectangle_l141_141314

theorem min_area_rectangle (P : ℕ) (hP : P = 60) :
  ∃ (l w : ℕ), 2 * l + 2 * w = P ∧ l * w = 29 :=
by
  sorry

end min_area_rectangle_l141_141314


namespace total_number_of_seats_l141_141298

def number_of_trains : ℕ := 3
def cars_per_train : ℕ := 12
def seats_per_car : ℕ := 24

theorem total_number_of_seats :
  number_of_trains * cars_per_train * seats_per_car = 864 := by
  sorry

end total_number_of_seats_l141_141298


namespace part1_part2_l141_141322

def first_order_ratio_increasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → x < y → (f x) / x < (f y) / y

def second_order_ratio_increasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → x < y → (f x) / x^2 < (f y) / y^2

noncomputable def f (h : ℝ) (x : ℝ) : ℝ :=
  x^3 - 2 * h * x^2 - h * x

theorem part1 (h : ℝ) (h1 : first_order_ratio_increasing (f h)) (h2 : ¬ second_order_ratio_increasing (f h)) :
  h < 0 :=
sorry

theorem part2 (f : ℝ → ℝ) (h : second_order_ratio_increasing f) (h2 : ∃ k > 0, ∀ x > 0, f x < k) :
  ∃ k, k = 0 ∧ ∀ x > 0, f x < k :=
sorry

end part1_part2_l141_141322


namespace find_smaller_number_l141_141474

theorem find_smaller_number (a b : ℕ) (h1 : b = 2 * a - 3) (h2 : a + b = 39) : a = 14 :=
by
  -- Sorry to skip the proof
  sorry

end find_smaller_number_l141_141474


namespace smallest_possible_n_l141_141193

theorem smallest_possible_n (n : ℕ) (h1 : n ≥ 100) (h2 : n < 1000)
  (h3 : n % 9 = 2) (h4 : n % 7 = 2) : n = 128 :=
by
  sorry

end smallest_possible_n_l141_141193


namespace Michael_rides_six_miles_l141_141230

theorem Michael_rides_six_miles
  (rate : ℝ)
  (time : ℝ)
  (interval_time : ℝ)
  (interval_distance : ℝ)
  (intervals : ℝ)
  (total_distance : ℝ) :
  rate = 1.5 ∧ time = 40 ∧ interval_time = 10 ∧ interval_distance = 1.5 ∧ intervals = time / interval_time ∧ total_distance = intervals * interval_distance →
  total_distance = 6 :=
by
  intros h
  -- Placeholder for the proof
  sorry

end Michael_rides_six_miles_l141_141230


namespace sin_tan_correct_value_l141_141114

noncomputable def sin_tan_value (x y : ℝ) (h : x^2 + y^2 = 1) : ℝ :=
  let sin_alpha := y
  let tan_alpha := y / x
  sin_alpha * tan_alpha

theorem sin_tan_correct_value :
  sin_tan_value (3/5) (-4/5) (by norm_num) = 16/15 := 
by
  sorry

end sin_tan_correct_value_l141_141114


namespace quadratic_trinomials_unique_root_value_l141_141132

theorem quadratic_trinomials_unique_root_value (p q : ℝ) :
  ∀ x, (x^2 + p * x + q) + (x^2 + q * x + p) = (2 * x^2 + (p + q) * x + (p + q)) →
  (((p + q = 0 ∨ p + q = 8) → (2 * (2:ℝ)^2 + (p + q) * 2 + (p + q) = 8 ∨ 2 * (2:ℝ)^2 + (p + q) * 2 + (p + q) = 32))) :=
by
  sorry

end quadratic_trinomials_unique_root_value_l141_141132


namespace fraction_equality_l141_141768
-- Import the necessary library

-- The proof statement
theorem fraction_equality : (16 + 8) / (4 - 2) = 12 := 
by {
  -- Inserting 'sorry' to indicate that the proof is omitted
  sorry
}

end fraction_equality_l141_141768


namespace find_x_given_conditions_l141_141198

variables {x y z : ℝ}

theorem find_x_given_conditions (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = (576 : ℝ)^(1/7) := 
sorry

end find_x_given_conditions_l141_141198


namespace verify_optionD_is_correct_l141_141803

-- Define the equations as options
def optionA : Prop := -abs (-6) = 6
def optionB : Prop := -(-6) = -6
def optionC : Prop := abs (-6) = -6
def optionD : Prop := -(-6) = 6

-- The proof problem to verify option D is correct
theorem verify_optionD_is_correct : optionD :=
by
  sorry

end verify_optionD_is_correct_l141_141803


namespace sample_avg_std_dev_xy_l141_141258

theorem sample_avg_std_dev_xy {x y : ℝ} (h1 : (4 + 5 + 6 + x + y) / 5 = 5)
  (h2 : (( (4 - 5)^2 + (5 - 5)^2 + (6 - 5)^2 + (x - 5)^2 + (y - 5)^2 ) / 5) = 2) : x * y = 21 :=
by
  sorry

end sample_avg_std_dev_xy_l141_141258


namespace rectangle_percentage_increase_l141_141994

theorem rectangle_percentage_increase (L W : ℝ) (P : ℝ) (h : (1 + P / 100) ^ 2 = 1.44) : P = 20 :=
by {
  -- skipped proof
  sorry
}

end rectangle_percentage_increase_l141_141994


namespace king_middle_school_teachers_l141_141900

theorem king_middle_school_teachers 
    (students : ℕ)
    (classes_per_student : ℕ)
    (normal_class_size : ℕ)
    (special_classes : ℕ)
    (special_class_size : ℕ)
    (classes_per_teacher : ℕ)
    (H1 : students = 1500)
    (H2 : classes_per_student = 5)
    (H3 : normal_class_size = 30)
    (H4 : special_classes = 10)
    (H5 : special_class_size = 15)
    (H6 : classes_per_teacher = 3) : 
    ∃ teachers : ℕ, teachers = 85 :=
by
  sorry

end king_middle_school_teachers_l141_141900


namespace square_side_factor_l141_141624

theorem square_side_factor (k : ℝ) (h : k^2 = 1) : k = 1 :=
sorry

end square_side_factor_l141_141624


namespace S13_is_52_l141_141156

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {n : ℕ}

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ n, a n = a1 + (n - 1) * d

-- Define the sum of the first n terms
def sum_of_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem S13_is_52 (h1 : is_arithmetic_sequence a)
                  (h2 : a 3 + a 7 + a 11 = 12)
                  (h3 : sum_of_first_n_terms S a) :
  S 13 = 52 :=
by sorry

end S13_is_52_l141_141156


namespace find_a_l141_141505

variable (a : ℝ)

def augmented_matrix (a : ℝ) :=
  ([1, -1, -3], [a, 3, 4])

def solution := (-1, 2)

theorem find_a (hx : -1 - 2 = -3)
               (hy : a * (-1) + 3 * 2 = 4) :
               a = 2 :=
by
  sorry

end find_a_l141_141505


namespace cost_of_pants_is_250_l141_141871

variable (costTotal : ℕ) (costTShirt : ℕ) (numTShirts : ℕ) (numPants : ℕ)

def costPants (costTotal costTShirt numTShirts numPants : ℕ) : ℕ :=
  let costTShirts := numTShirts * costTShirt
  let costPantsTotal := costTotal - costTShirts
  costPantsTotal / numPants

-- Given conditions
axiom h1 : costTotal = 1500
axiom h2 : costTShirt = 100
axiom h3 : numTShirts = 5
axiom h4 : numPants = 4

-- Prove each pair of pants costs $250
theorem cost_of_pants_is_250 : costPants costTotal costTShirt numTShirts numPants = 250 :=
by
  -- Place proof here
  sorry

end cost_of_pants_is_250_l141_141871


namespace find_a_l141_141583

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 + x

-- Define the derivative of the function f(x)
def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - 2 * a * x + 1

-- The main theorem: if the tangent at x = 1 is parallel to the line y = 2x, then a = 1
theorem find_a (a : ℝ) : f' 1 a = 2 → a = 1 :=
by
  intro h
  -- The proof is skipped
  sorry

end find_a_l141_141583


namespace sum_first_20_terms_arithmetic_seq_l141_141018

theorem sum_first_20_terms_arithmetic_seq :
  ∃ (a d : ℤ) (S_20 : ℤ), d > 0 ∧
  (a + 2 * d) * (a + 6 * d) = -12 ∧
  (a + 3 * d) + (a + 5 * d) = -4 ∧
  S_20 = 20 * a + (20 * 19 / 2) * d ∧
  S_20 = 180 :=
by
  sorry

end sum_first_20_terms_arithmetic_seq_l141_141018


namespace nancy_picked_l141_141511

variable (total_picked : ℕ) (alyssa_picked : ℕ)

-- Assuming the conditions given in the problem
def conditions := total_picked = 59 ∧ alyssa_picked = 42

-- Proving that Nancy picked 17 pears
theorem nancy_picked : conditions total_picked alyssa_picked → total_picked - alyssa_picked = 17 := by
  sorry

end nancy_picked_l141_141511


namespace ellipse_eq_find_k_l141_141488

open Real

-- Part I: Prove the equation of the ellipse
theorem ellipse_eq (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1 :=
by
  sorry

-- Part II: Prove the value of k
theorem find_k (P : ℝ × ℝ) (hP : P = (-2, 1)) (MN_length : ℝ) (hMN : MN_length = 2) 
  (k : ℝ) (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∃ k : ℝ, k = -4 :=
by
  sorry

end ellipse_eq_find_k_l141_141488


namespace sum_of_coefficients_l141_141450

def polynomial (x : ℤ) : ℤ := 3 * (x^8 - 2 * x^5 + 4 * x^3 - 7) - 5 * (2 * x^4 - 3 * x^2 + 8) + 6 * (x^6 - 3)

theorem sum_of_coefficients : polynomial 1 = -59 := 
by
  sorry

end sum_of_coefficients_l141_141450


namespace max_m_sq_plus_n_sq_l141_141417

theorem max_m_sq_plus_n_sq (m n : ℤ) (h1 : 1 ≤ m ∧ m ≤ 1981) (h2 : 1 ≤ n ∧ n ≤ 1981) (h3 : (n^2 - m*n - m^2)^2 = 1) : m^2 + n^2 ≤ 3524578 :=
sorry

end max_m_sq_plus_n_sq_l141_141417


namespace parabola_focus_coordinates_l141_141407

theorem parabola_focus_coordinates :
  ∃ h k : ℝ, (y = -1/8 * x^2 + 2 * x - 1) ∧ (h = 8 ∧ k = 5) :=
sorry

end parabola_focus_coordinates_l141_141407


namespace triangle_side_sum_l141_141094

def sum_of_remaining_sides_of_triangle (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) : Prop :=
  α = 40 ∧ β = 50 ∧ γ = 180 - α - β ∧ c = 8 * Real.sqrt 3 →
  (a + b) = 34.3

theorem triangle_side_sum (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) :
  sum_of_remaining_sides_of_triangle A B C a b c α β γ :=
sorry

end triangle_side_sum_l141_141094


namespace max_rabbits_l141_141454

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l141_141454


namespace cats_in_shelter_l141_141941

theorem cats_in_shelter (C D: ℕ) (h1 : 15 * D = 7 * C) 
                        (h2 : 15 * (D + 12) = 11 * C) :
    C = 45 := by
  sorry

end cats_in_shelter_l141_141941


namespace proof_problem_l141_141641

-- Variables representing the numbers a, b, and c
variables {a b c : ℝ}

-- Given condition
def given_condition (a b c : ℝ) : Prop :=
  (a^2 + b^2) / (b^2 + c^2) = a / c

-- Required to prove
def to_prove (a b c : ℝ) : Prop :=
  (a / b = b / c) → False

-- Theorem stating that the given condition does not imply the required assertion
theorem proof_problem (a b c : ℝ) (h : given_condition a b c) : to_prove a b c :=
sorry

end proof_problem_l141_141641


namespace penultimate_digit_even_l141_141852

theorem penultimate_digit_even (n : ℕ) (h : n > 2) : ∃ k : ℕ, ∃ d : ℕ, d % 2 = 0 ∧ 10 * d + k = (3 ^ n) % 100 :=
sorry

end penultimate_digit_even_l141_141852


namespace value_of_fraction_l141_141030

theorem value_of_fraction : (121^2 - 112^2) / 9 = 233 := by
  -- use the difference of squares property
  sorry

end value_of_fraction_l141_141030


namespace interest_rate_l141_141665

/-- 
Given a principal amount that doubles itself in 10 years at simple interest,
prove that the rate of interest per annum is 10%.
-/
theorem interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (h1 : SI = P) (h2 : T = 10) (h3 : SI = P * R * T / 100) : 
  R = 10 := by
  sorry

end interest_rate_l141_141665


namespace estimate_students_correct_l141_141266

noncomputable def estimate_students_below_85 
  (total_students : ℕ)
  (mean_score : ℝ)
  (variance : ℝ)
  (prob_90_to_95 : ℝ) : ℕ :=
if total_students = 50 ∧ mean_score = 90 ∧ prob_90_to_95 = 0.3 then 10 else 0

theorem estimate_students_correct 
  (total_students : ℕ)
  (mean_score : ℝ)
  (variance : ℝ)
  (prob_90_to_95 : ℝ)
  (h1 : total_students = 50) 
  (h2 : mean_score = 90)
  (h3 : prob_90_to_95 = 0.3) : 
  estimate_students_below_85 total_students mean_score variance prob_90_to_95 = 10 :=
by
  sorry

end estimate_students_correct_l141_141266


namespace pyramid_rhombus_side_length_l141_141981

theorem pyramid_rhombus_side_length
  (α β S: ℝ) (hα : 0 < α) (hβ : 0 < β) (hS : 0 < S) :
  ∃ a : ℝ, a = 2 * Real.sqrt (2 * S * Real.cos β / Real.sin α) :=
by
  sorry

end pyramid_rhombus_side_length_l141_141981


namespace express_a_b_find_a_b_m_n_find_a_l141_141939

-- 1. Prove that a = m^2 + 5n^2 and b = 2mn given a + b√5 = (m + n√5)^2
theorem express_a_b (a b m n : ℤ) (h : a + b * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) :
  a = m ^ 2 + 5 * n ^ 2 ∧ b = 2 * m * n := sorry

-- 2. Prove there exists positive integers a = 6, b = 2, m = 1, and n = 1 such that 
-- a + b√5 = (m + n√5)^2.
theorem find_a_b_m_n : ∃ (a b m n : ℕ), a = 6 ∧ b = 2 ∧ m = 1 ∧ n = 1 ∧ 
  (a + b * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) := sorry

-- 3. Prove a = 46 or a = 14 given a + 6√5 = (m + n√5)^2 and a, m, n are positive integers.
theorem find_a (a m n : ℕ) (h : a + 6 * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) :
  a = 46 ∨ a = 14 := sorry

end express_a_b_find_a_b_m_n_find_a_l141_141939


namespace negative_root_m_positive_l141_141071

noncomputable def is_negative_root (m : ℝ) : Prop :=
  ∃ x : ℝ, x < 0 ∧ x^2 + m * x - 4 = 0

theorem negative_root_m_positive : ∀ m : ℝ, is_negative_root m → m > 0 :=
by
  intro m
  intro h
  sorry

end negative_root_m_positive_l141_141071
