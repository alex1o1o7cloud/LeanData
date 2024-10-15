import Mathlib

namespace NUMINAMATH_GPT_angle_rotation_l2043_204386

theorem angle_rotation (initial_angle : ℝ) (rotation : ℝ) :
  initial_angle = 30 → rotation = 450 → 
  ∃ (new_angle : ℝ), new_angle = 60 :=
by
  sorry

end NUMINAMATH_GPT_angle_rotation_l2043_204386


namespace NUMINAMATH_GPT_function_monotonically_increasing_iff_range_of_a_l2043_204390

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem function_monotonically_increasing_iff_range_of_a (a : ℝ) :
  (∀ x, (deriv (f a) x) ≥ 0) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_function_monotonically_increasing_iff_range_of_a_l2043_204390


namespace NUMINAMATH_GPT_lees_friend_initial_money_l2043_204362

theorem lees_friend_initial_money (lee_initial_money friend_initial_money total_cost change : ℕ) 
  (h1 : lee_initial_money = 10) 
  (h2 : total_cost = 15) 
  (h3 : change = 3) 
  (h4 : (lee_initial_money + friend_initial_money) - total_cost = change) : 
  friend_initial_money = 8 := by
  sorry

end NUMINAMATH_GPT_lees_friend_initial_money_l2043_204362


namespace NUMINAMATH_GPT_train_speed_kmph_l2043_204356

theorem train_speed_kmph (len_train : ℝ) (len_platform : ℝ) (time_cross : ℝ) (total_distance : ℝ) (speed_mps : ℝ) (speed_kmph : ℝ) 
  (h1 : len_train = 250) 
  (h2 : len_platform = 150.03) 
  (h3 : time_cross = 20) 
  (h4 : total_distance = len_train + len_platform) 
  (h5 : speed_mps = total_distance / time_cross) 
  (h6 : speed_kmph = speed_mps * 3.6) : 
  speed_kmph = 72.0054 := 
by 
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_train_speed_kmph_l2043_204356


namespace NUMINAMATH_GPT_problem_proof_l2043_204303

-- Define the given conditions and the target statement
theorem problem_proof (a b : ℝ) (h1 : a - b = 2) (h2 : a * b = 10.5) : a^2 + b^2 = 25 := 
by sorry

end NUMINAMATH_GPT_problem_proof_l2043_204303


namespace NUMINAMATH_GPT_find_retail_price_l2043_204383

-- Define the wholesale price
def wholesale_price : ℝ := 90

-- Define the profit as 20% of the wholesale price
def profit (w : ℝ) : ℝ := 0.2 * w

-- Define the selling price as the wholesale price plus the profit
def selling_price (w p : ℝ) : ℝ := w + p

-- Define the selling price as 90% of the retail price t
def discount_selling_price (t : ℝ) : ℝ := 0.9 * t

-- Prove that the retail price t is 120 given the conditions
theorem find_retail_price :
  ∃ t : ℝ, wholesale_price + (profit wholesale_price) = discount_selling_price t → t = 120 :=
by
  sorry

end NUMINAMATH_GPT_find_retail_price_l2043_204383


namespace NUMINAMATH_GPT_sqrt_of_four_is_pm_two_l2043_204389

theorem sqrt_of_four_is_pm_two (y : ℤ) : y * y = 4 → y = 2 ∨ y = -2 := by
  sorry

end NUMINAMATH_GPT_sqrt_of_four_is_pm_two_l2043_204389


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2043_204397

theorem arithmetic_sequence_sum (c d : ℕ) (h₁ : 3 + 5 = 8) (h₂ : 8 + 5 = 13) (h₃ : c = 13 + 5) (h₄ : d = 18 + 5) (h₅ : d + 5 = 28) : c + d = 41 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2043_204397


namespace NUMINAMATH_GPT_david_cups_consumed_l2043_204363

noncomputable def cups_of_water (time_in_minutes : ℕ) : ℝ :=
  time_in_minutes / 20

theorem david_cups_consumed : cups_of_water 225 = 11.25 := by
  sorry

end NUMINAMATH_GPT_david_cups_consumed_l2043_204363


namespace NUMINAMATH_GPT_curve_crosses_itself_l2043_204314

theorem curve_crosses_itself :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ (t1^2 - 3 = t2^2 - 3) ∧ (t1^3 - 6*t1 + 2 = t2^3 - 6*t2 + 2) ∧
  ((t1^2 - 3 = 3) ∧ (t1^3 - 6*t1 + 2 = 2)) :=
by
  sorry

end NUMINAMATH_GPT_curve_crosses_itself_l2043_204314


namespace NUMINAMATH_GPT_sequence_1005th_term_l2043_204311

-- Definitions based on conditions
def first_term : ℚ := sorry
def second_term : ℚ := 10
def third_term : ℚ := 4 * first_term - (1:ℚ)
def fourth_term : ℚ := 4 * first_term + (1:ℚ)

-- Common difference
def common_difference : ℚ := (fourth_term - third_term)

-- Arithmetic sequence term calculation
def nth_term (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n-1) * d

-- Theorem statement
theorem sequence_1005th_term : nth_term first_term common_difference 1005 = 5480 := sorry

end NUMINAMATH_GPT_sequence_1005th_term_l2043_204311


namespace NUMINAMATH_GPT_total_people_l2043_204331

theorem total_people (N B : ℕ) (h1 : N = 4 * B + 10) (h2 : N = 5 * B + 1) : N = 46 := by
  -- The proof will follow from the conditions, but it is not required in this script.
  sorry

end NUMINAMATH_GPT_total_people_l2043_204331


namespace NUMINAMATH_GPT_part_a_part_b_l2043_204320

-- Define sum conditions for consecutive odd integers
def consecutive_odd_sum (N : ℕ) : Prop :=
  ∃ (n k : ℕ), n ≥ 2 ∧ N = n * (2 * k + n)

-- Part (a): Prove 2005 can be written as sum of consecutive odd positive integers
theorem part_a : consecutive_odd_sum 2005 :=
by
  sorry

-- Part (b): Prove 2006 cannot be written as sum of consecutive odd positive integers
theorem part_b : ¬consecutive_odd_sum 2006 :=
by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l2043_204320


namespace NUMINAMATH_GPT_num_assignments_l2043_204307

/-- 
Mr. Wang originally planned to grade at a rate of 6 assignments per hour.
After grading for 2 hours, he increased his rate to 8 assignments per hour,
finishing 3 hours earlier than initially planned. 
Prove that the total number of assignments is 84. 
-/
theorem num_assignments (x : ℕ) (h : ℕ) (H1 : 6 * h = x) (H2 : 8 * (h - 5) = x - 12) : x = 84 :=
by
  sorry

end NUMINAMATH_GPT_num_assignments_l2043_204307


namespace NUMINAMATH_GPT_inscribed_circle_radius_l2043_204375

noncomputable def radius_of_inscribed_circle (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem inscribed_circle_radius :
  radius_of_inscribed_circle 8 8 5 = 38 / 21 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l2043_204375


namespace NUMINAMATH_GPT_amount_diana_owes_l2043_204348

-- Problem definitions
def principal : ℝ := 75
def rate : ℝ := 0.07
def time : ℝ := 1
def interest := principal * rate * time
def total_owed := principal + interest

-- Theorem to prove that the total amount owed is $80.25
theorem amount_diana_owes : total_owed = 80.25 := by
  sorry

end NUMINAMATH_GPT_amount_diana_owes_l2043_204348


namespace NUMINAMATH_GPT_complement_N_star_in_N_l2043_204358

-- The set of natural numbers
def N : Set ℕ := { n | true }

-- The set of positive integers
def N_star : Set ℕ := { n | n > 0 }

-- The complement of N_star in N is the set {0}
theorem complement_N_star_in_N : { n | n ∈ N ∧ n ∉ N_star } = {0} := by
  sorry

end NUMINAMATH_GPT_complement_N_star_in_N_l2043_204358


namespace NUMINAMATH_GPT_find_solutions_l2043_204364

theorem find_solutions (x y : ℝ) :
    (x * y^2 = 15 * x^2 + 17 * x * y + 15 * y^2 ∧ x^2 * y = 20 * x^2 + 3 * y^2) ↔ 
    (x = 0 ∧ y = 0) ∨ (x = -19 ∧ y = -2) :=
by sorry

end NUMINAMATH_GPT_find_solutions_l2043_204364


namespace NUMINAMATH_GPT_part_I_part_II_part_III_l2043_204318

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 1
noncomputable def g (x : ℝ) : ℝ := Real.exp x

-- Part (I)
theorem part_I (a : ℝ) (h_a : a = 1) : 
  ∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 0 ∧ f x a * g x = 1 := sorry

-- Part (II)
theorem part_II (a : ℝ) (h_a : a = -1) (k : ℝ) :
  (∃ x : ℝ, f x a = k * g x ∧ ∀ y : ℝ, y ≠ x → f y a ≠ k * g y) ↔ 
  (k > 3 * Real.exp (-2) ∨ (0 < k ∧ k < 1 * Real.exp (-1))) := sorry

-- Part (III)
theorem part_III (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), (x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ x₁ ≠ x₂) →
  abs (f x₁ a - f x₂ a) < abs (g x₁ - g x₂)) ↔
  (-1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2) := sorry

end NUMINAMATH_GPT_part_I_part_II_part_III_l2043_204318


namespace NUMINAMATH_GPT_infinite_points_with_sum_of_squares_condition_l2043_204376

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a circle centered at origin with given radius
def isWithinCircle (P : Point2D) (r : ℝ) :=
  P.x^2 + P.y^2 ≤ r^2

-- Define the distance squared from a point to another point
def dist2 (P Q : Point2D) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the problem
theorem infinite_points_with_sum_of_squares_condition :
  ∃ P : Point2D, isWithinCircle P 1 → (dist2 P ⟨-1, 0⟩ + dist2 P ⟨1, 0⟩ = 3) :=
by  
  sorry

end NUMINAMATH_GPT_infinite_points_with_sum_of_squares_condition_l2043_204376


namespace NUMINAMATH_GPT_pine_saplings_in_sample_l2043_204313

-- Definitions based on conditions
def total_saplings : ℕ := 30000
def pine_saplings : ℕ := 4000
def sample_size : ℕ := 150

-- Main theorem to prove
theorem pine_saplings_in_sample : (pine_saplings * sample_size) / total_saplings = 20 :=
by sorry

end NUMINAMATH_GPT_pine_saplings_in_sample_l2043_204313


namespace NUMINAMATH_GPT_max_xy_l2043_204368

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 7 * x + 8 * y = 112) : xy ≤ 56 :=
sorry

end NUMINAMATH_GPT_max_xy_l2043_204368


namespace NUMINAMATH_GPT_smallest_positive_integer_mod_l2043_204319

theorem smallest_positive_integer_mod (a : ℕ) (h1 : a ≡ 4 [MOD 5]) (h2 : a ≡ 6 [MOD 7]) : a = 34 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_mod_l2043_204319


namespace NUMINAMATH_GPT_triangle_QR_length_l2043_204347

noncomputable def length_PM : ℝ := 6 -- PM = 6 cm
noncomputable def length_MA : ℝ := 12 -- MA = 12 cm
noncomputable def length_NB : ℝ := 9 -- NB = 9 cm
def MN_parallel_PQ : Prop := true -- MN ∥ PQ

theorem triangle_QR_length 
  (h1 : MN_parallel_PQ)
  (h2 : length_PM = 6)
  (h3 : length_MA = 12)
  (h4 : length_NB = 9) : 
  length_QR = 27 :=
sorry

end NUMINAMATH_GPT_triangle_QR_length_l2043_204347


namespace NUMINAMATH_GPT_find_y_when_x_is_1_l2043_204329

theorem find_y_when_x_is_1 
  (k : ℝ) 
  (h1 : ∀ y, x = k / y^2) 
  (h2 : x = 1) 
  (h3 : x = 0.1111111111111111) 
  (y : ℝ) 
  (hy : y = 6) 
  (hx_k : k = 0.1111111111111111 * 36) :
  y = 2 := sorry

end NUMINAMATH_GPT_find_y_when_x_is_1_l2043_204329


namespace NUMINAMATH_GPT_arithmetic_sequence_a3_is_8_l2043_204306

-- Define the arithmetic sequence
def arithmetic_sequence (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

-- Theorem to prove a3 = 8 given a1 = 4 and d = 2
theorem arithmetic_sequence_a3_is_8 (a1 d : ℕ) (h1 : a1 = 4) (h2 : d = 2) : arithmetic_sequence a1 d 3 = 8 :=
by
  sorry -- Proof not required as per instruction

end NUMINAMATH_GPT_arithmetic_sequence_a3_is_8_l2043_204306


namespace NUMINAMATH_GPT_range_of_x_l2043_204361

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - (1 / x) + 2 * Real.sin x

theorem range_of_x (x : ℝ) (h₀ : x > 0) (h₁ : f (1 - x) > f x) : x < (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l2043_204361


namespace NUMINAMATH_GPT_sum_of_p_and_q_l2043_204315

-- Definitions for points and collinearity condition
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := {x := 1, y := 3, z := -2}
def B : Point3D := {x := 2, y := 5, z := 1}
def C (p q : ℝ) : Point3D := {x := p, y := 7, z := q - 2}

def collinear (A B C : Point3D) : Prop :=
  ∃ (k : ℝ), B.x - A.x = k * (C.x - A.x) ∧ B.y - A.y = k * (C.y - A.y) ∧ B.z - A.z = k * (C.z - A.z)

theorem sum_of_p_and_q (p q : ℝ) (h : collinear A B (C p q)) : p + q = 9 := by
  sorry

end NUMINAMATH_GPT_sum_of_p_and_q_l2043_204315


namespace NUMINAMATH_GPT_triangle_formation_conditions_l2043_204384

theorem triangle_formation_conditions (a b c : ℝ) :
  (a + b > c ∧ |a - b| < c) ↔ (a + b > c ∧ b + c > a ∧ c + a > b ∧ |a - b| < c ∧ |b - c| < a ∧ |c - a| < b) :=
sorry

end NUMINAMATH_GPT_triangle_formation_conditions_l2043_204384


namespace NUMINAMATH_GPT_mixed_gender_appointment_schemes_l2043_204396

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 
  else n * factorial (n - 1)

noncomputable def P (n r : ℕ) : ℕ :=
  factorial n / factorial (n - r)

theorem mixed_gender_appointment_schemes : 
  let total_students := 9
  let total_permutations := P total_students 3
  let male_students := 5
  let female_students := 4
  let male_permutations := P male_students 3
  let female_permutations := P female_students 3
  total_permutations - (male_permutations + female_permutations) = 420 :=
by 
  sorry

end NUMINAMATH_GPT_mixed_gender_appointment_schemes_l2043_204396


namespace NUMINAMATH_GPT_evaluate_expression_l2043_204344

theorem evaluate_expression : 
  |-2| + (1 / 4) - 1 - 4 * Real.cos (Real.pi / 4) + Real.sqrt 8 = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2043_204344


namespace NUMINAMATH_GPT_tan_sin_equality_l2043_204354

theorem tan_sin_equality :
  (Real.tan (30 * Real.pi / 180))^2 + (Real.sin (45 * Real.pi / 180))^2 = 5 / 6 :=
by sorry

end NUMINAMATH_GPT_tan_sin_equality_l2043_204354


namespace NUMINAMATH_GPT_TileD_in_AreaZ_l2043_204335

namespace Tiles

structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

def TileA : Tile := {top := 5, right := 3, bottom := 2, left := 4}
def TileB : Tile := {top := 2, right := 4, bottom := 5, left := 3}
def TileC : Tile := {top := 3, right := 6, bottom := 1, left := 5}
def TileD : Tile := {top := 5, right := 2, bottom := 3, left := 6}

variables (X Y Z W : Tile)
variable (tiles : List Tile := [TileA, TileB, TileC, TileD])

noncomputable def areaZContains : Tile := sorry

theorem TileD_in_AreaZ  : areaZContains = TileD := sorry

end Tiles

end NUMINAMATH_GPT_TileD_in_AreaZ_l2043_204335


namespace NUMINAMATH_GPT_min_A_div_B_l2043_204323

theorem min_A_div_B (x A B : ℝ) (hx_pos : 0 < x) (hA_pos : 0 < A) (hB_pos : 0 < B) 
  (h1 : x^2 + 1 / x^2 = A) (h2 : x - 1 / x = B + 3) : 
  (A / B) = 6 + 2 * Real.sqrt 11 :=
sorry

end NUMINAMATH_GPT_min_A_div_B_l2043_204323


namespace NUMINAMATH_GPT_off_road_vehicle_cost_l2043_204322

theorem off_road_vehicle_cost
  (dirt_bike_count : ℕ) (dirt_bike_cost : ℕ)
  (off_road_vehicle_count : ℕ) (register_cost : ℕ)
  (total_cost : ℕ) (off_road_vehicle_cost : ℕ) :
  dirt_bike_count = 3 → dirt_bike_cost = 150 →
  off_road_vehicle_count = 4 → register_cost = 25 →
  total_cost = 1825 →
  3 * dirt_bike_cost + 4 * off_road_vehicle_cost + 7 * register_cost = total_cost →
  off_road_vehicle_cost = 300 := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_off_road_vehicle_cost_l2043_204322


namespace NUMINAMATH_GPT_train_length_l2043_204333

theorem train_length (t_post t_platform l_platform : ℕ) (L : ℚ) : 
  t_post = 15 → t_platform = 25 → l_platform = 100 →
  (L / t_post) = (L + l_platform) / t_platform → 
  L = 150 :=
by 
  intros h1 h2 h3 h4
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_train_length_l2043_204333


namespace NUMINAMATH_GPT_height_difference_l2043_204350

-- Define the heights of Eiffel Tower and Burj Khalifa as constants
def eiffelTowerHeight : ℕ := 324
def burjKhalifaHeight : ℕ := 830

-- Define the statement that needs to be proven
theorem height_difference : burjKhalifaHeight - eiffelTowerHeight = 506 := by
  sorry

end NUMINAMATH_GPT_height_difference_l2043_204350


namespace NUMINAMATH_GPT_line_passes_point_l2043_204379

theorem line_passes_point (k : ℝ) :
  ((1 + 4 * k) * 2 - (2 - 3 * k) * 2 + (2 - 14 * k)) = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_passes_point_l2043_204379


namespace NUMINAMATH_GPT_square_chord_length_eq_l2043_204398

def radius1 := 10
def radius2 := 7
def centers_distance := 15
def chord_length (x : ℝ) := 2 * x

theorem square_chord_length_eq :
    ∀ (x : ℝ), chord_length x = 15 →
    (10 + x)^2 - 200 * (Real.sqrt ((1 + 19.0 / 35.0) / 2)) = 200 - 200 * Real.sqrt (27.0 / 35.0) :=
sorry

end NUMINAMATH_GPT_square_chord_length_eq_l2043_204398


namespace NUMINAMATH_GPT_minimum_sum_l2043_204346

open Matrix

noncomputable def a := 54
noncomputable def b := 40
noncomputable def c := 5
noncomputable def d := 4

theorem minimum_sum 
  (a b c d : ℕ) 
  (ha : 4 * a = 24 * a - 27 * b) 
  (hb : 4 * b = 15 * a - 17 * b) 
  (hc : 3 * c = 24 * c - 27 * d) 
  (hd : 3 * d = 15 * c - 17 * d) 
  (Hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : 
  a + b + c + d = 103 :=
by
  sorry

end NUMINAMATH_GPT_minimum_sum_l2043_204346


namespace NUMINAMATH_GPT_total_time_equals_l2043_204382

-- Define the distances and speeds
def first_segment_distance : ℝ := 50
def first_segment_speed : ℝ := 30
def second_segment_distance (b : ℝ) : ℝ := b
def second_segment_speed : ℝ := 80

-- Prove that the total time is equal to (400 + 3b) / 240 hours
theorem total_time_equals (b : ℝ) : 
  (first_segment_distance / first_segment_speed) + (second_segment_distance b / second_segment_speed) 
  = (400 + 3 * b) / 240 := 
by
  sorry

end NUMINAMATH_GPT_total_time_equals_l2043_204382


namespace NUMINAMATH_GPT_john_cakes_bought_l2043_204340

-- Conditions
def cake_price : ℕ := 12
def john_paid : ℕ := 18

-- Definition of the total cost
def total_cost : ℕ := 2 * john_paid

-- Calculate number of cakes
def num_cakes (total_cost cake_price : ℕ) : ℕ := total_cost / cake_price

-- Theorem to prove that the number of cakes John Smith bought is 3
theorem john_cakes_bought : num_cakes total_cost cake_price = 3 := by
  sorry

end NUMINAMATH_GPT_john_cakes_bought_l2043_204340


namespace NUMINAMATH_GPT_find_BC_distance_l2043_204378

-- Definitions of constants as per problem conditions
def ACB_angle : ℝ := 120
def AC_distance : ℝ := 2
def AB_distance : ℝ := 3

-- The theorem to prove the distance BC
theorem find_BC_distance (BC : ℝ) (h : AC_distance * AC_distance + (BC * BC) - 2 * AC_distance * BC * Real.cos (ACB_angle * Real.pi / 180) = AB_distance * AB_distance) : BC = Real.sqrt 6 - 1 :=
by
  sorry

end NUMINAMATH_GPT_find_BC_distance_l2043_204378


namespace NUMINAMATH_GPT_no_quad_term_l2043_204393

theorem no_quad_term (x m : ℝ) : 
  (2 * x^2 - 2 * (7 + 3 * x - 2 * x^2) + m * x^2) = -6 * x - 14 → m = -6 := 
by 
  sorry

end NUMINAMATH_GPT_no_quad_term_l2043_204393


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l2043_204317

theorem infinite_geometric_series_sum :
  let a := (5 : ℚ) / 3
  let r := -(3 : ℚ) / 4
  |r| < 1 →
  (∀ S, S = a / (1 - r) → S = 20 / 21) :=
by
  intros a r h_abs_r S h_S
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l2043_204317


namespace NUMINAMATH_GPT_contrapositive_statement_l2043_204381

theorem contrapositive_statement :
  (∀ n : ℕ, (n % 2 = 0 ∨ n % 5 = 0) → n % 10 = 0) →
  (∀ n : ℕ, n % 10 ≠ 0 → ¬(n % 2 = 0 ∧ n % 5 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_statement_l2043_204381


namespace NUMINAMATH_GPT_sin_value_l2043_204399

theorem sin_value (alpha : ℝ) (h1 : -π / 6 < alpha ∧ alpha < π / 6)
  (h2 : Real.cos (alpha + π / 6) = 4 / 5) :
  Real.sin (2 * alpha + π / 12) = 17 * Real.sqrt 2 / 50 :=
by
    sorry

end NUMINAMATH_GPT_sin_value_l2043_204399


namespace NUMINAMATH_GPT_eggs_per_basket_l2043_204301

theorem eggs_per_basket
  (kids : ℕ)
  (friends : ℕ)
  (adults : ℕ)
  (baskets : ℕ)
  (eggs_per_person : ℕ)
  (htotal : kids + friends + adults + 1 = 20)
  (eggs_total : (kids + friends + adults + 1) * eggs_per_person = 180)
  (baskets_count : baskets = 15)
  : (180 / 15) = 12 :=
by
  sorry

end NUMINAMATH_GPT_eggs_per_basket_l2043_204301


namespace NUMINAMATH_GPT_geometric_mean_of_1_and_4_l2043_204373

theorem geometric_mean_of_1_and_4 :
  ∃ a : ℝ, a^2 = 4 ∧ (a = 2 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_geometric_mean_of_1_and_4_l2043_204373


namespace NUMINAMATH_GPT_probability_of_qualified_product_l2043_204325

theorem probability_of_qualified_product :
  let p1 := 0.30   -- Proportion of the first batch
  let d1 := 0.05   -- Defect rate of the first batch
  let p2 := 0.70   -- Proportion of the second batch
  let d2 := 0.04   -- Defect rate of the second batch
  -- Probability of selecting a qualified product
  p1 * (1 - d1) + p2 * (1 - d2) = 0.957 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_qualified_product_l2043_204325


namespace NUMINAMATH_GPT_total_journey_distance_l2043_204337

/-- 
A woman completes a journey in 5 hours. She travels the first half of the journey 
at 21 km/hr and the second half at 24 km/hr. Find the total journey in km.
-/
theorem total_journey_distance :
  ∃ D : ℝ, (D / 2) / 21 + (D / 2) / 24 = 5 ∧ D = 112 :=
by
  use 112
  -- Please prove the following statements
  sorry

end NUMINAMATH_GPT_total_journey_distance_l2043_204337


namespace NUMINAMATH_GPT_find_c_l2043_204391

theorem find_c (y c : ℝ) (h : y > 0) (h₂ : (8*y)/20 + (c*y)/10 = 0.7*y) : c = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l2043_204391


namespace NUMINAMATH_GPT_prize_difference_l2043_204336

def mateo_hourly_rate : ℕ := 20
def sydney_daily_rate : ℕ := 400
def hours_in_a_week : ℕ := 24 * 7
def days_in_a_week : ℕ := 7

def mateo_total : ℕ := mateo_hourly_rate * hours_in_a_week
def sydney_total : ℕ := sydney_daily_rate * days_in_a_week

def difference_amount : ℕ := 560

theorem prize_difference : mateo_total - sydney_total = difference_amount := sorry

end NUMINAMATH_GPT_prize_difference_l2043_204336


namespace NUMINAMATH_GPT_find_first_number_l2043_204302

theorem find_first_number (a b : ℕ) (k : ℕ) (h1 : a = 3 * k) (h2 : b = 4 * k) (h3 : Nat.lcm a b = 84) : a = 21 := 
sorry

end NUMINAMATH_GPT_find_first_number_l2043_204302


namespace NUMINAMATH_GPT_cuboidal_box_area_l2043_204305

/-- Given conditions about a cuboidal box:
    - The area of one face is 72 cm²
    - The area of an adjacent face is 60 cm²
    - The volume of the cuboidal box is 720 cm³,
    Prove that the area of the third adjacent face is 120 cm². -/
theorem cuboidal_box_area (l w h : ℝ) (h1 : l * w = 72) (h2 : w * h = 60) (h3 : l * w * h = 720) :
  l * h = 120 :=
sorry

end NUMINAMATH_GPT_cuboidal_box_area_l2043_204305


namespace NUMINAMATH_GPT_polygon_with_45_deg_exterior_angle_is_eight_gon_l2043_204308

theorem polygon_with_45_deg_exterior_angle_is_eight_gon
  (each_exterior_angle : ℝ) (h1 : each_exterior_angle = 45) 
  (sum_exterior_angles : ℝ) (h2 : sum_exterior_angles = 360) :
  ∃ (n : ℕ), n = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_with_45_deg_exterior_angle_is_eight_gon_l2043_204308


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_not_sufficient_condition_l2043_204374

theorem necessary_but_not_sufficient (a b : ℝ) : (a > 2 ∧ b > 2) → (a + b > 4) :=
sorry

theorem not_sufficient_condition (a b : ℝ) : (a + b > 4) → ¬(a > 2 ∧ b > 2) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_not_sufficient_condition_l2043_204374


namespace NUMINAMATH_GPT_largest_value_l2043_204365

theorem largest_value (A B C D E : ℕ)
  (hA : A = (3 + 5 + 2 + 8))
  (hB : B = (3 * 5 + 2 + 8))
  (hC : C = (3 + 5 * 2 + 8))
  (hD : D = (3 + 5 + 2 * 8))
  (hE : E = (3 * 5 * 2 * 8)) :
  max (max (max (max A B) C) D) E = E := 
sorry

end NUMINAMATH_GPT_largest_value_l2043_204365


namespace NUMINAMATH_GPT_sum_first_4_terms_of_arithmetic_sequence_eq_8_l2043_204360

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + (a 1 - a 0)

def S4 (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_first_4_terms_of_arithmetic_sequence_eq_8
  (a : ℕ → ℤ) 
  (h_seq : arithmetic_seq a) 
  (h_a2 : a 1 = 1) 
  (h_a3 : a 2 = 3) :
  S4 a = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_4_terms_of_arithmetic_sequence_eq_8_l2043_204360


namespace NUMINAMATH_GPT_minimum_value_l2043_204357

open Real

-- Given the conditions
variables (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k)

-- The theorem
theorem minimum_value (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < k) : 
  ∃ x, x = (3 : ℝ) / k ∧ ∀ y, y = (a / (k * b) + b / (k * c) + c / (k * a)) → y ≥ x :=
sorry

end NUMINAMATH_GPT_minimum_value_l2043_204357


namespace NUMINAMATH_GPT_fraction_of_emilys_coins_l2043_204353

theorem fraction_of_emilys_coins {total_states : ℕ} (h1 : total_states = 30)
    {states_from_1790_to_1799 : ℕ} (h2 : states_from_1790_to_1799 = 9) :
    (states_from_1790_to_1799 / total_states : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_fraction_of_emilys_coins_l2043_204353


namespace NUMINAMATH_GPT_Michelangelo_ceiling_painting_l2043_204309

theorem Michelangelo_ceiling_painting (C : ℕ) : 
  ∃ C, (C + (1/4) * C = 15) ∧ (28 - (C + (1/4) * C) = 13) :=
sorry

end NUMINAMATH_GPT_Michelangelo_ceiling_painting_l2043_204309


namespace NUMINAMATH_GPT_minimum_path_proof_l2043_204349

noncomputable def minimum_path (r : ℝ) (h : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ :=
  let R := Real.sqrt (r^2 + h^2)
  let theta := 2 * Real.pi * (R / (2 * Real.pi * r))
  let A := (d1, 0)
  let B := (-d2 * Real.cos (theta / 2), -d2 * Real.sin (theta / 2))
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem minimum_path_proof :
  minimum_path 800 (300 * Real.sqrt 3) 150 (450 * Real.sqrt 2) = 562.158 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_path_proof_l2043_204349


namespace NUMINAMATH_GPT_max_value_proof_l2043_204371

noncomputable def max_value_b_minus_a (a b : ℝ) : ℝ :=
  b - a

theorem max_value_proof (a b : ℝ) (h1 : a < 0) (h2 : ∀ x, (x^2 + 2017 * a) * (x + 2016 * b) ≥ 0) : max_value_b_minus_a a b ≤ 2017 :=
sorry

end NUMINAMATH_GPT_max_value_proof_l2043_204371


namespace NUMINAMATH_GPT_integer_solutions_inequality_system_l2043_204372

noncomputable def check_inequality_system (x : ℤ) : Prop :=
  (3 * x + 1 < x - 3) ∧ ((1 + x) / 2 ≤ (1 + 2 * x) / 3 + 1)

theorem integer_solutions_inequality_system :
  {x : ℤ | check_inequality_system x} = {-5, -4, -3} :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_inequality_system_l2043_204372


namespace NUMINAMATH_GPT_find_y_l2043_204341

variable (x y z : ℕ)

-- Conditions
def condition1 : Prop := 100 + 200 + 300 + x = 1000
def condition2 : Prop := 300 + z + 100 + x + y = 1000

-- Theorem to be proven
theorem find_y (h1 : condition1 x) (h2 : condition2 x y z) : z + y = 200 :=
sorry

end NUMINAMATH_GPT_find_y_l2043_204341


namespace NUMINAMATH_GPT_closest_point_exists_l2043_204385

def closest_point_on_line_to_point (x : ℝ) (y : ℝ) : Prop :=
  ∃(p : ℝ × ℝ), p = (3, 1) ∧ ∀(q : ℝ × ℝ), q.2 = (q.1 + 3) / 3 → dist p (3, 2) ≤ dist q (3, 2)

theorem closest_point_exists :
  closest_point_on_line_to_point 3 2 :=
sorry

end NUMINAMATH_GPT_closest_point_exists_l2043_204385


namespace NUMINAMATH_GPT_desktops_to_sell_l2043_204326

theorem desktops_to_sell (laptops desktops : ℕ) (ratio_laptops desktops_sold laptops_expected : ℕ) :
  ratio_laptops = 5 → desktops_sold = 3 → laptops_expected = 40 → 
  desktops = (desktops_sold * laptops_expected) / ratio_laptops :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry -- This is where the proof would go, but it's not needed for this task

end NUMINAMATH_GPT_desktops_to_sell_l2043_204326


namespace NUMINAMATH_GPT_find_a_l2043_204377

theorem find_a (a : ℝ) : (∀ x : ℝ, -1 < x ∧ x < 2 ↔ |a * x + 2| < 6) → a = -4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l2043_204377


namespace NUMINAMATH_GPT_distinct_arrangements_balloon_l2043_204352

theorem distinct_arrangements_balloon : 
  let n := 7 
  let freq_l := 2 
  let freq_o := 2 
  let freq_b := 1 
  let freq_a := 1 
  let freq_n := 1 
  Nat.factorial n / (Nat.factorial freq_l * Nat.factorial freq_o * Nat.factorial freq_b * Nat.factorial freq_a * Nat.factorial freq_n) = 1260 :=
by
  sorry

end NUMINAMATH_GPT_distinct_arrangements_balloon_l2043_204352


namespace NUMINAMATH_GPT_construction_company_total_weight_l2043_204328

noncomputable def total_weight_of_materials_in_pounds : ℝ :=
  let weight_of_concrete := 12568.3
  let weight_of_bricks := 2108 * 2.20462
  let weight_of_stone := 7099.5
  let weight_of_wood := 3778 * 2.20462
  let weight_of_steel := 5879 * (1 / 16)
  let weight_of_glass := 12.5 * 2000
  let weight_of_sand := 2114.8
  weight_of_concrete + weight_of_bricks + weight_of_stone + weight_of_wood + weight_of_steel + weight_of_glass + weight_of_sand

theorem construction_company_total_weight : total_weight_of_materials_in_pounds = 60129.72 :=
by
  sorry

end NUMINAMATH_GPT_construction_company_total_weight_l2043_204328


namespace NUMINAMATH_GPT_total_questions_attempted_l2043_204316

theorem total_questions_attempted (C W T : ℕ) (hC : C = 42) (h_score : 4 * C - W = 150) : T = C + W → T = 60 :=
by
  sorry

end NUMINAMATH_GPT_total_questions_attempted_l2043_204316


namespace NUMINAMATH_GPT_sin_cos_identity_l2043_204387

theorem sin_cos_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) + Real.cos (20 * Real.pi / 180) * Real.sin (140 * Real.pi / 180)) =
  (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l2043_204387


namespace NUMINAMATH_GPT_find_n_from_binomial_term_l2043_204327

noncomputable def binomial_coefficient (n r : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

theorem find_n_from_binomial_term :
  (∃ n : ℕ, 3^2 * binomial_coefficient n 2 = 54) ↔ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_n_from_binomial_term_l2043_204327


namespace NUMINAMATH_GPT_which_is_lying_l2043_204310

-- Ben's statement
def ben_says (dan_truth cam_truth : Bool) : Bool :=
  (dan_truth ∧ ¬ cam_truth) ∨ (¬ dan_truth ∧ cam_truth)

-- Dan's statement
def dan_says (ben_truth cam_truth : Bool) : Bool :=
  (ben_truth ∧ ¬ cam_truth) ∨ (¬ ben_truth ∧ cam_truth)

-- Cam's statement
def cam_says (ben_truth dan_truth : Bool) : Bool :=
  ¬ ben_truth ∧ ¬ dan_truth

-- Lean statement to be proven
theorem which_is_lying :
  (∃ (ben_truth dan_truth cam_truth : Bool), 
    ben_says dan_truth cam_truth ∧ 
    dan_says ben_truth cam_truth ∧ 
    cam_says ben_truth dan_truth ∧
    ¬ ben_truth ∧ ¬ dan_truth ∧ cam_truth) ↔ (¬ ben_truth ∧ ¬ dan_truth ∧ cam_truth) :=
sorry

end NUMINAMATH_GPT_which_is_lying_l2043_204310


namespace NUMINAMATH_GPT_age_problem_l2043_204321

theorem age_problem (A N : ℕ) (h₁: A = 18) (h₂: N * (A + 3) - N * (A - 3) = A) : N = 3 := by
  sorry

end NUMINAMATH_GPT_age_problem_l2043_204321


namespace NUMINAMATH_GPT_option_A_is_linear_equation_l2043_204312

-- Definitions for considering an equation being linear in two variables
def is_linear_equation (e : Prop) : Prop :=
  ∃ (a b c : ℝ), e = (a = b + c) ∧ a ≠ 0 ∧ b ≠ 0

-- The given equation in option A
def Eq_A : Prop := ∀ (x y : ℝ), (2 * y - 1) / 5 = 2 - (3 * x - 2) / 4

-- Proof problem statement
theorem option_A_is_linear_equation : is_linear_equation Eq_A :=
sorry

end NUMINAMATH_GPT_option_A_is_linear_equation_l2043_204312


namespace NUMINAMATH_GPT_price_of_chips_l2043_204359

theorem price_of_chips (P : ℝ) (h1 : 1.5 = 1.5) (h2 : 45 = 45) (h3 : 15 = 15) (h4 : 10 = 10) :
  15 * P + 10 * 1.5 = 45 → P = 2 :=
by
  sorry

end NUMINAMATH_GPT_price_of_chips_l2043_204359


namespace NUMINAMATH_GPT_min_cost_theater_tickets_l2043_204304

open Real

variable (x y : ℝ)

theorem min_cost_theater_tickets :
  (x + y = 140) →
  (y ≥ 2 * x) →
  ∀ x y, 60 * x + 100 * y ≥ 12160 :=
by
  sorry

end NUMINAMATH_GPT_min_cost_theater_tickets_l2043_204304


namespace NUMINAMATH_GPT_no_x_satisfies_inequality_l2043_204355

def f (x : ℝ) : ℝ := x^2 + x

theorem no_x_satisfies_inequality : ¬ ∃ x : ℝ, f (x - 2) + f x < 0 :=
by 
  unfold f 
  sorry

end NUMINAMATH_GPT_no_x_satisfies_inequality_l2043_204355


namespace NUMINAMATH_GPT_time_comparison_l2043_204342

-- Definitions from the conditions
def speed_first_trip (v : ℝ) : ℝ := v
def distance_first_trip : ℝ := 80
def distance_second_trip : ℝ := 240
def speed_second_trip (v : ℝ) : ℝ := 4 * v

-- Theorem to prove
theorem time_comparison (v : ℝ) (hv : v > 0) :
  (distance_second_trip / speed_second_trip v) = (3 / 4) * (distance_first_trip / speed_first_trip v) :=
by
  -- Outline of the proof, we skip the actual steps
  sorry

end NUMINAMATH_GPT_time_comparison_l2043_204342


namespace NUMINAMATH_GPT_solve_four_tuple_l2043_204395

-- Define the problem conditions
theorem solve_four_tuple (a b c d : ℝ) : 
    (ab + c + d = 3) → 
    (bc + d + a = 5) → 
    (cd + a + b = 2) → 
    (da + b + c = 6) → 
    (a = 2) ∧ (b = 0) ∧ (c = 0) ∧ (d = 3) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_solve_four_tuple_l2043_204395


namespace NUMINAMATH_GPT_circumscribed_circle_radius_l2043_204343

noncomputable def circumradius_of_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) : ℚ :=
  c / 2

theorem circumscribed_circle_radius :
  circumradius_of_right_triangle 30 40 50 (by norm_num : 30^2 + 40^2 = 50^2) = 25 := by
norm_num /- correct answer confirmed -/
sorry

end NUMINAMATH_GPT_circumscribed_circle_radius_l2043_204343


namespace NUMINAMATH_GPT_problem1_l2043_204351

noncomputable def log6_7 : ℝ := Real.logb 6 7
noncomputable def log7_6 : ℝ := Real.logb 7 6

theorem problem1 : log6_7 > log7_6 := 
by
  sorry

end NUMINAMATH_GPT_problem1_l2043_204351


namespace NUMINAMATH_GPT_calc_expr_value_l2043_204300

theorem calc_expr_value : (0.5 ^ 4) / (0.05 ^ 2.5) = 559.06 := 
by 
  sorry

end NUMINAMATH_GPT_calc_expr_value_l2043_204300


namespace NUMINAMATH_GPT_avg_of_first_21_multiples_l2043_204334

theorem avg_of_first_21_multiples (n : ℕ) (h : (21 * 11 * n / 21) = 88) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_avg_of_first_21_multiples_l2043_204334


namespace NUMINAMATH_GPT_balance_balls_l2043_204367

variable (R O B P : ℝ)

-- Conditions based on the problem statement
axiom h1 : 4 * R = 8 * B
axiom h2 : 3 * O = 7.5 * B
axiom h3 : 8 * B = 6 * P

-- The theorem we need to prove
theorem balance_balls : 5 * R + 3 * O + 3 * P = 21.5 * B :=
by 
  sorry

end NUMINAMATH_GPT_balance_balls_l2043_204367


namespace NUMINAMATH_GPT_solve_diamond_l2043_204332

theorem solve_diamond : 
  (∃ (Diamond : ℤ), Diamond * 5 + 3 = Diamond * 6 + 2) →
  (∃ (Diamond : ℤ), Diamond = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_diamond_l2043_204332


namespace NUMINAMATH_GPT_problem_statement_l2043_204380

open Real

theorem problem_statement (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * π)
  (h₁ : 2 * cos x ≤ sqrt (1 + sin (2 * x)) - sqrt (1 - sin (2 * x))
  ∧ sqrt (1 + sin (2 * x)) - sqrt (1 - sin (2 * x)) ≤ sqrt 2) :
  π / 4 ≤ x ∧ x ≤ 7 * π / 4 := sorry

end NUMINAMATH_GPT_problem_statement_l2043_204380


namespace NUMINAMATH_GPT_total_balloons_l2043_204369

-- Define the number of yellow balloons each person has
def tom_balloons : Nat := 18
def sara_balloons : Nat := 12
def alex_balloons : Nat := 7

-- Prove that the total number of balloons is 37
theorem total_balloons : tom_balloons + sara_balloons + alex_balloons = 37 := 
by 
  sorry

end NUMINAMATH_GPT_total_balloons_l2043_204369


namespace NUMINAMATH_GPT_Chris_buys_48_golf_balls_l2043_204392

theorem Chris_buys_48_golf_balls (total_golf_balls : ℕ) (dozen_to_balls : ℕ → ℕ)
  (dan_buys : ℕ) (gus_buys : ℕ) (chris_buys : ℕ) :
  dozen_to_balls 1 = 12 →
  dan_buys = 5 →
  gus_buys = 2 →
  total_golf_balls = 132 →
  (chris_buys * 12) + (dan_buys * 12) + (gus_buys * 12) = total_golf_balls →
  chris_buys * 12 = 48 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_Chris_buys_48_golf_balls_l2043_204392


namespace NUMINAMATH_GPT_cheapest_shipping_option_l2043_204370

/-- Defines the cost options for shipping, given a weight of 5 pounds. -/
def cost_A (weight : ℕ) : ℝ := 5.00 + 0.80 * weight
def cost_B (weight : ℕ) : ℝ := 4.50 + 0.85 * weight
def cost_C (weight : ℕ) : ℝ := 3.00 + 0.95 * weight

/-- Proves that for a package weighing 5 pounds, the cheapest shipping option is Option C costing $7.75. -/
theorem cheapest_shipping_option : cost_C 5 < cost_A 5 ∧ cost_C 5 < cost_B 5 ∧ cost_C 5 = 7.75 :=
by
  -- Calculation is omitted
  sorry

end NUMINAMATH_GPT_cheapest_shipping_option_l2043_204370


namespace NUMINAMATH_GPT_balls_remaining_l2043_204324

-- Define the initial number of balls in the box
def initial_balls := 10

-- Define the number of balls taken by Yoongi
def balls_taken := 3

-- Define the number of balls left after Yoongi took some balls
def balls_left := initial_balls - balls_taken

-- The theorem statement to be proven
theorem balls_remaining : balls_left = 7 :=
by
    -- Skipping the proof
    sorry

end NUMINAMATH_GPT_balls_remaining_l2043_204324


namespace NUMINAMATH_GPT_sector_area_l2043_204345

-- Define the properties and conditions
def perimeter_of_sector (r l : ℝ) : Prop :=
  l + 2 * r = 8

def central_angle_arc_length (r : ℝ) : ℝ :=
  2 * r

-- Theorem to prove the area of the sector
theorem sector_area (r : ℝ) (l : ℝ) 
  (h_perimeter : perimeter_of_sector r l) 
  (h_arc_length : l = central_angle_arc_length r) : 
  1 / 2 * l * r = 4 := 
by
  -- This is the place where the proof would go; we use sorry to indicate it's incomplete
  sorry

end NUMINAMATH_GPT_sector_area_l2043_204345


namespace NUMINAMATH_GPT_fencing_problem_l2043_204330

theorem fencing_problem (W L : ℝ) (hW : W = 40) (hArea : W * L = 320) : 
  2 * L + W = 56 :=
by
  sorry

end NUMINAMATH_GPT_fencing_problem_l2043_204330


namespace NUMINAMATH_GPT_area_of_given_triangle_l2043_204339

def point := (ℝ × ℝ)

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

theorem area_of_given_triangle :
  area_of_triangle (0, 0) (4, 0) (4, 6) = 12.0 :=
by 
  sorry

end NUMINAMATH_GPT_area_of_given_triangle_l2043_204339


namespace NUMINAMATH_GPT_parabola_equation_l2043_204394

variables (a b c p : ℝ)
variables (h1 : a > 0) (h2 : b > 0) (h3 : p > 0)
variables (h_eccentricity : c / a = 2)
variables (h_b : b = Real.sqrt (3) * a)
variables (h_c : c = Real.sqrt (a^2 + b^2))
variables (d : ℝ) (h_distance : d = 2) (h_d_formula : d = (a * p) / (2 * c))

theorem parabola_equation (h : (a > 0) ∧ (b > 0) ∧ (p > 0) ∧ (c / a = 2) ∧ (b = (Real.sqrt 3) * a) ∧ (c = Real.sqrt (a^2 + b^2)) ∧ (d = 2) ∧ (d = (a * p) / (2 * c))) : x^2 = 16 * y :=
by {
  -- Lean does not require an actual proof here, so we use sorry.
  sorry
}

end NUMINAMATH_GPT_parabola_equation_l2043_204394


namespace NUMINAMATH_GPT_bus_stops_for_4_minutes_per_hour_l2043_204338

theorem bus_stops_for_4_minutes_per_hour
  (V_excluding_stoppages V_including_stoppages : ℝ)
  (h1 : V_excluding_stoppages = 90)
  (h2 : V_including_stoppages = 84) :
  (60 * (V_excluding_stoppages - V_including_stoppages)) / V_excluding_stoppages = 4 :=
by
  sorry

end NUMINAMATH_GPT_bus_stops_for_4_minutes_per_hour_l2043_204338


namespace NUMINAMATH_GPT_triangle_area_change_l2043_204366

theorem triangle_area_change (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let A_original := (B * H) / 2
  let H_new := H * 0.60
  let B_new := B * 1.40
  let A_new := (B_new * H_new) / 2
  (A_new = A_original * 0.84) :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_change_l2043_204366


namespace NUMINAMATH_GPT_largest_value_is_B_l2043_204388

def exprA := 1 + 2 * 3 + 4
def exprB := 1 + 2 + 3 * 4
def exprC := 1 + 2 + 3 + 4
def exprD := 1 * 2 + 3 + 4
def exprE := 1 * 2 + 3 * 4

theorem largest_value_is_B : exprB = 15 ∧ exprB > exprA ∧ exprB > exprC ∧ exprB > exprD ∧ exprB > exprE := 
by
  sorry

end NUMINAMATH_GPT_largest_value_is_B_l2043_204388
