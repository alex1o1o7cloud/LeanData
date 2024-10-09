import Mathlib

namespace volume_is_85_l2283_228381

/-!
# Proof Problem
Prove that the total volume of Carl's and Kate's cubes is 85, given the conditions,
Carl has 3 cubes each with a side length of 3, and Kate has 4 cubes each with a side length of 1.
-/

-- Definitions for the problem conditions:
def volume_of_cube (s : ℕ) : ℕ := s^3

def total_volume (n : ℕ) (s : ℕ) : ℕ := n * volume_of_cube s

-- Given conditions
def carls_cubes_volume : ℕ := total_volume 3 3
def kates_cubes_volume : ℕ := total_volume 4 1

-- The total volume of Carl's and Kate's cubes:
def total_combined_volume : ℕ := carls_cubes_volume + kates_cubes_volume

-- Prove the total volume is 85
theorem volume_is_85 : total_combined_volume = 85 :=
by sorry

end volume_is_85_l2283_228381


namespace triangle_side_length_l2283_228312

-- Defining basic properties and known lengths of the similar triangles
def GH : ℝ := 8
def HI : ℝ := 16
def YZ : ℝ := 24
def XY : ℝ := 12

-- Defining the similarity condition for triangles GHI and XYZ
def triangles_similar : Prop := 
  -- The similarity of the triangles implies proportionality of the sides
  (XY / GH = YZ / HI)

-- The theorem statement to prove
theorem triangle_side_length (h_sim : triangles_similar) : XY = 12 :=
by
  -- assuming the similarity condition and known lengths
  sorry -- This will be the detailed proof

end triangle_side_length_l2283_228312


namespace integers_even_condition_l2283_228311

-- Definitions based on conditions
def is_even (n : ℤ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℤ) : Prop :=
(is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
(¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
(¬ is_even a ∧ ¬ is_even b ∧ is_even c)

-- Proof statement
theorem integers_even_condition (a b c : ℤ) (h : ¬ exactly_one_even a b c) :
  (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (is_even a ∧ is_even b) ∨
  (is_even a ∧ is_even c) ∨
  (is_even b ∧ is_even c) :=
sorry

end integers_even_condition_l2283_228311


namespace roots_of_quadratic_l2283_228395

theorem roots_of_quadratic (α β : ℝ) (h1 : α^2 - 4*α - 5 = 0) (h2 : β^2 - 4*β - 5 = 0) :
  3*α^4 + 10*β^3 = 2593 := 
by
  sorry

end roots_of_quadratic_l2283_228395


namespace bus_cost_proof_l2283_228310

-- Define conditions
def train_cost (bus_cost : ℚ) : ℚ := bus_cost + 6.85
def discount_rate : ℚ := 0.15
def service_fee : ℚ := 1.25
def combined_cost : ℚ := 10.50

-- Formula for the total cost after discount
def discounted_train_cost (bus_cost : ℚ) : ℚ := (train_cost bus_cost) * (1 - discount_rate)
def total_cost (bus_cost : ℚ) : ℚ := discounted_train_cost bus_cost + bus_cost + service_fee

-- Lean 4 statement asserting the cost of the bus ride before service fee
theorem bus_cost_proof : ∃ (B : ℚ), total_cost B = combined_cost ∧ B = 1.85 :=
sorry

end bus_cost_proof_l2283_228310


namespace one_more_square_possible_l2283_228352

def grid_size : ℕ := 29
def total_cells : ℕ := grid_size * grid_size
def number_of_squares_removed : ℕ := 99
def cells_per_square : ℕ := 4
def total_removed_cells : ℕ := number_of_squares_removed * cells_per_square
def remaining_cells : ℕ := total_cells - total_removed_cells

theorem one_more_square_possible :
  remaining_cells ≥ cells_per_square :=
sorry

end one_more_square_possible_l2283_228352


namespace inequality_ln_l2283_228347

theorem inequality_ln (x : ℝ) (h₁ : x > -1) (h₂ : x ≠ 0) :
    (2 * abs x) / (2 + x) < abs (Real.log (1 + x)) ∧ abs (Real.log (1 + x)) < (abs x) / Real.sqrt (1 + x) :=
by
  sorry

end inequality_ln_l2283_228347


namespace function_no_real_zeros_l2283_228330

variable (a b c : ℝ)

-- Conditions: a, b, c form a geometric sequence and ac > 0
def geometric_sequence (a b c : ℝ) : Prop := b^2 = a * c
def positive_product (a c : ℝ) : Prop := a * c > 0

theorem function_no_real_zeros (h_geom : geometric_sequence a b c) (h_pos : positive_product a c) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 := 
by
  sorry

end function_no_real_zeros_l2283_228330


namespace sugar_content_of_mixture_l2283_228396

theorem sugar_content_of_mixture 
  (volume_juice1 : ℝ) (conc_juice1 : ℝ)
  (volume_juice2 : ℝ) (conc_juice2 : ℝ) 
  (total_volume : ℝ) (total_sugar : ℝ) 
  (resulting_sugar_content : ℝ) :
  volume_juice1 = 2 →
  conc_juice1 = 0.1 →
  volume_juice2 = 3 →
  conc_juice2 = 0.15 →
  total_volume = volume_juice1 + volume_juice2 →
  total_sugar = (conc_juice1 * volume_juice1) + (conc_juice2 * volume_juice2) →
  resulting_sugar_content = (total_sugar / total_volume) * 100 →
  resulting_sugar_content = 13 :=
by
  intros
  sorry

end sugar_content_of_mixture_l2283_228396


namespace swimming_speed_in_still_water_l2283_228319

theorem swimming_speed_in_still_water (v : ℝ) (current_speed : ℝ) (time : ℝ) (distance : ℝ) (effective_speed : current_speed = 10) (time_to_return : time = 6) (distance_to_return : distance = 12) (speed_eq : v - current_speed = distance / time) : v = 12 :=
by
  sorry

end swimming_speed_in_still_water_l2283_228319


namespace total_distance_walked_l2283_228321

theorem total_distance_walked (t1 t2 : ℝ) (r : ℝ) (total_distance : ℝ)
  (h1 : t1 = 15 / 60)  -- Convert 15 minutes to hours
  (h2 : t2 = 25 / 60)  -- Convert 25 minutes to hours
  (h3 : r = 3)         -- Average speed in miles per hour
  (h4 : total_distance = r * (t1 + t2))
  : total_distance = 2 :=
by
  -- here is where the proof would go
  sorry

end total_distance_walked_l2283_228321


namespace car_A_faster_than_car_B_l2283_228338

noncomputable def car_A_speed := 
  let t_A1 := 50 / 60 -- time for the first 50 miles at 60 mph
  let t_A2 := 50 / 40 -- time for the next 50 miles at 40 mph
  let t_A := t_A1 + t_A2 -- total time for Car A
  100 / t_A -- average speed of Car A

noncomputable def car_B_speed := 
  let t_B := 1 + (1 / 4) + 1 -- total time for Car B, including a 15-minute stop
  100 / t_B -- average speed of Car B

theorem car_A_faster_than_car_B : car_A_speed > car_B_speed := 
by sorry

end car_A_faster_than_car_B_l2283_228338


namespace problem_statement_l2283_228334

theorem problem_statement : 
  10 - 1.05 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.93 :=
by sorry

end problem_statement_l2283_228334


namespace probability_of_two_eights_l2283_228387

-- Define a function that calculates the factorial of a number
noncomputable def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Definition of binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ :=
  fact n / (fact k * fact (n - k))

-- Probability of exactly two dice showing 8 out of eight 8-sided dice
noncomputable def prob_exactly_two_eights : ℚ :=
  binom 8 2 * ((1 / 8 : ℚ) ^ 2) * ((7 / 8 : ℚ) ^ 6)

-- Main theorem statement
theorem probability_of_two_eights :
  prob_exactly_two_eights = 0.196 := by
  sorry

end probability_of_two_eights_l2283_228387


namespace evalExpression_at_3_2_l2283_228390

def evalExpression (x y : ℕ) : ℕ := 3 * x^y + 4 * y^x

theorem evalExpression_at_3_2 : evalExpression 3 2 = 59 := by
  sorry

end evalExpression_at_3_2_l2283_228390


namespace solve_for_a_l2283_228355

-- Define the sets M and N as given in the problem
def M : Set ℝ := {x : ℝ | x^2 + 6 * x - 16 = 0}
def N (a : ℝ) : Set ℝ := {x : ℝ | x * a - 3 = 0}

-- Define the proof statement
theorem solve_for_a (a : ℝ) : (N a ⊆ M) ↔ (a = 0 ∨ a = 3/2 ∨ a = -3/8) :=
by
  -- The proof would go here
  sorry

end solve_for_a_l2283_228355


namespace factor_expression_l2283_228317

-- Problem Statement
theorem factor_expression (x y : ℝ) : 60 * x ^ 2 + 40 * y = 20 * (3 * x ^ 2 + 2 * y) :=
by
  -- Proof to be provided
  sorry

end factor_expression_l2283_228317


namespace parallel_vectors_implies_x_value_l2283_228362

variable (x : ℝ)

def vec_a : ℝ × ℝ := (1, 2)
def vec_b (x : ℝ) : ℝ × ℝ := (x, 1)
def scalar_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem parallel_vectors_implies_x_value :
  (∃ k : ℝ, vec_add vec_a (scalar_mul 2 (vec_b x)) = scalar_mul k (vec_sub (scalar_mul 2 vec_a) (scalar_mul 2 (vec_b x)))) →
  x = 1 / 2 :=
by
  sorry

end parallel_vectors_implies_x_value_l2283_228362


namespace diagonal_of_rectangular_prism_l2283_228371

noncomputable def diagonal_length (l w h : ℝ) : ℝ :=
  Real.sqrt (l^2 + w^2 + h^2)

theorem diagonal_of_rectangular_prism :
  diagonal_length 15 25 20 = 25 * Real.sqrt 2 :=
by
  sorry

end diagonal_of_rectangular_prism_l2283_228371


namespace largest_of_w_l2283_228309

variable {x y z w : ℝ}

namespace MathProof

theorem largest_of_w
  (h1 : x + 3 = y - 1)
  (h2 : x + 3 = z + 5)
  (h3 : x + 3 = w - 2) :  
  w > y ∧ w > x ∧ w > z :=
by
  sorry

end MathProof

end largest_of_w_l2283_228309


namespace gumball_water_wednesday_l2283_228336

theorem gumball_water_wednesday :
  ∀ (total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water : ℕ),
  total_weekly_water = 60 →
  monday_thursday_saturday_water = 9 →
  tuesday_friday_sunday_water = 8 →
  total_weekly_water - (monday_thursday_saturday_water * 3 + tuesday_friday_sunday_water * 3) = 9 :=
by
  intros total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water
  sorry

end gumball_water_wednesday_l2283_228336


namespace max_value_pq_qr_rs_sp_l2283_228348

def max_pq_qr_rs_sp (p q r s : ℕ) : ℕ :=
  p * q + q * r + r * s + s * p

theorem max_value_pq_qr_rs_sp :
  ∀ (p q r s : ℕ), (p = 1 ∨ p = 5 ∨ p = 3 ∨ p = 6) → 
                    (q = 1 ∨ q = 5 ∨ q = 3 ∨ q = 6) →
                    (r = 1 ∨ r = 5 ∨ r = 3 ∨ r = 6) → 
                    (s = 1 ∨ s = 5 ∨ s = 3 ∨ s = 6) →
                    p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s → 
                    max_pq_qr_rs_sp p q r s ≤ 56 := by
  sorry

end max_value_pq_qr_rs_sp_l2283_228348


namespace ratio_of_area_of_shaded_square_l2283_228335

theorem ratio_of_area_of_shaded_square 
  (large_square : Type) 
  (smaller_squares : Finset large_square) 
  (area_large_square : ℝ) 
  (area_smaller_square : ℝ) 
  (h_division : smaller_squares.card = 25)
  (h_equal_area : ∀ s ∈ smaller_squares, area_smaller_square = (area_large_square / 25))
  (shaded_square : Finset large_square)
  (h_shaded_sub : shaded_square ⊆ smaller_squares)
  (h_shaded_card : shaded_square.card = 5) :
  (5 * area_smaller_square) / area_large_square = 1 / 5 := 
by
  sorry

end ratio_of_area_of_shaded_square_l2283_228335


namespace ashton_sheets_l2283_228353
-- Import the entire Mathlib to bring in the necessary library

-- Defining the conditions and proving the statement
theorem ashton_sheets (t j a : ℕ) (h1 : t = j + 10) (h2 : j = 32) (h3 : j + a = t + 30) : a = 40 := by
  -- Sorry placeholder for the proof
  sorry

end ashton_sheets_l2283_228353


namespace smallest_t_for_given_roots_l2283_228354

-- Define the polynomial with integer coefficients and specific roots
def poly (x : ℝ) : ℝ := (x + 3) * (x - 4) * (x - 6) * (2 * x - 1)

-- Define the main theorem statement
theorem smallest_t_for_given_roots :
  ∃ (t : ℤ), 0 < t ∧ t = 72 := by
  -- polynomial expansion skipped, proof will come here
  sorry

end smallest_t_for_given_roots_l2283_228354


namespace cars_left_in_parking_lot_l2283_228356

-- Define constants representing the initial number of cars and cars that went out.
def initial_cars : ℕ := 24
def first_out : ℕ := 8
def second_out : ℕ := 6

-- State the theorem to prove the remaining cars in the parking lot.
theorem cars_left_in_parking_lot : 
  initial_cars - first_out - second_out = 10 := 
by {
  -- Here, 'sorry' is used to indicate the proof is omitted.
  sorry
}

end cars_left_in_parking_lot_l2283_228356


namespace smallest_difference_l2283_228389

theorem smallest_difference (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 362880) (h_order : a < b ∧ b < c) : c - a = 92 := 
sorry

end smallest_difference_l2283_228389


namespace parabola_coefficient_c_l2283_228349

def parabola (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem parabola_coefficient_c (b c : ℝ) (h1 : parabola b c 1 = -1) (h2 : parabola b c 3 = 9) : 
  c = -3 := 
by
  sorry

end parabola_coefficient_c_l2283_228349


namespace percentage_of_students_in_60_to_69_range_is_20_l2283_228343

theorem percentage_of_students_in_60_to_69_range_is_20 :
  let scores := [4, 8, 6, 5, 2]
  let total_students := scores.sum
  let students_in_60_to_69 := 5
  (students_in_60_to_69 * 100 / total_students) = 20 := by
  sorry

end percentage_of_students_in_60_to_69_range_is_20_l2283_228343


namespace find_sum_fusion_number_l2283_228346

def sum_fusion_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * (2 * k + 1)

theorem find_sum_fusion_number (n : ℕ) :
  n = 2020 ↔ sum_fusion_number n :=
sorry

end find_sum_fusion_number_l2283_228346


namespace xy_value_l2283_228307

theorem xy_value : 
  ∀ (x y : ℝ),
  (∀ (A B C : ℝ × ℝ), A = (1, 8) ∧ B = (x, y) ∧ C = (6, 3) → 
  (C.1 = (A.1 + B.1) / 2) ∧ (C.2 = (A.2 + B.2) / 2)) → 
  x * y = -22 :=
sorry

end xy_value_l2283_228307


namespace increasing_interval_of_g_l2283_228369

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (Real.pi / 3 - 2 * x)) -
  2 * (Real.sin (Real.pi / 4 + x) * Real.sin (Real.pi / 4 - x))

noncomputable def g (x : ℝ) : ℝ :=
  f (x + Real.pi / 12)

theorem increasing_interval_of_g :
  ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2),
  ∃ a b, a = -Real.pi / 12 ∧ b = Real.pi / 4 ∧
      (∀ x y, (a ≤ x ∧ x ≤ y ∧ y ≤ b) → g x ≤ g y) :=
sorry

end increasing_interval_of_g_l2283_228369


namespace sum_of_coordinates_inv_graph_l2283_228384

variable {f : ℝ → ℝ}
variable (hf : f 2 = 12)

theorem sum_of_coordinates_inv_graph :
  ∃ (x y : ℝ), y = f⁻¹ x / 3 ∧ x = 12 ∧ y = 2 / 3 ∧ x + y = 38 / 3 := by
  sorry

end sum_of_coordinates_inv_graph_l2283_228384


namespace bella_age_l2283_228383

theorem bella_age (B : ℕ) 
  (h1 : (B + 9) + B + B / 2 = 27) 
  : B = 6 :=
by sorry

end bella_age_l2283_228383


namespace Seokjin_tangerines_per_day_l2283_228300

theorem Seokjin_tangerines_per_day 
  (T_initial : ℕ) (D : ℕ) (T_remaining : ℕ) 
  (h1 : T_initial = 29) 
  (h2 : D = 8) 
  (h3 : T_remaining = 5) : 
  (T_initial - T_remaining) / D = 3 := 
by
  sorry

end Seokjin_tangerines_per_day_l2283_228300


namespace find_odd_natural_numbers_l2283_228379

-- Definition of a friendly number
def is_friendly (n : ℕ) : Prop :=
  ∀ i, (n / 10^i) % 10 = (n / 10^(i + 1)) % 10 + 1 ∨ (n / 10^i) % 10 = (n / 10^(i + 1)) % 10 - 1

-- Given condition: n is divisible by 64m
def is_divisible_by_64m (n m : ℕ) : Prop :=
  64 * m ∣ n

-- Proof problem statement
theorem find_odd_natural_numbers (m : ℕ) (hm1 : m % 2 = 1) :
  (5 ∣ m → ¬ ∃ n, is_friendly n ∧ is_divisible_by_64m n m) ∧ 
  (¬ 5 ∣ m → ∃ n, is_friendly n ∧ is_divisible_by_64m n m) :=
by
  sorry

end find_odd_natural_numbers_l2283_228379


namespace red_tulips_l2283_228345

theorem red_tulips (white_tulips : ℕ) (bouquets : ℕ)
  (hw : white_tulips = 21)
  (hb : bouquets = 7)
  (div_prop : ∀ n, white_tulips % n = 0 ↔ bouquets % n = 0) : 
  ∃ red_tulips : ℕ, red_tulips = 7 :=
by
  sorry

end red_tulips_l2283_228345


namespace chocolate_bars_per_box_l2283_228322

-- Definitions for the given conditions
def total_chocolate_bars : ℕ := 849
def total_boxes : ℕ := 170

-- The statement to prove
theorem chocolate_bars_per_box : total_chocolate_bars / total_boxes = 5 :=
by 
  -- Proof is omitted here
  sorry

end chocolate_bars_per_box_l2283_228322


namespace solution_set_of_inequality_l2283_228331

theorem solution_set_of_inequality :
  { x : ℝ | |x + 1| + |x - 4| ≥ 7 } = { x : ℝ | x ≤ -2 ∨ x ≥ 5 } := sorry

end solution_set_of_inequality_l2283_228331


namespace prob_A_prob_B_l2283_228376

variable (a b : ℝ) -- Declare variables a and b as real numbers
variable (h_ab : a + b = 1) -- Declare the condition a + b = 1
variable (h_pos_a : 0 < a) -- Declare a is a positive real number
variable (h_pos_b : 0 < b) -- Declare b is a positive real number

-- Prove that 1/a + 1/b ≥ 4 under the given conditions
theorem prob_A (h_ab : a + b = 1) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  (1 / a) + (1 / b) ≥ 4 :=
by
  sorry

-- Prove that a^2 + b^2 ≥ 1/2 under the given conditions
theorem prob_B (h_ab : a + b = 1) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  a^2 + b^2 ≥ 1 / 2 :=
by
  sorry

end prob_A_prob_B_l2283_228376


namespace walkway_area_correct_l2283_228367

-- Define the dimensions and conditions
def bed_width : ℝ := 4
def bed_height : ℝ := 3
def walkway_width : ℝ := 2
def num_rows : ℕ := 4
def num_columns : ℕ := 3
def num_beds : ℕ := num_rows * num_columns

-- Total dimensions of garden including walkways
def total_width : ℝ := (num_columns * bed_width) + ((num_columns + 1) * walkway_width)
def total_height : ℝ := (num_rows * bed_height) + ((num_rows + 1) * walkway_width)

-- Areas
def total_garden_area : ℝ := total_width * total_height
def total_bed_area : ℝ := (bed_width * bed_height) * num_beds

-- Correct answer we want to prove
def walkway_area : ℝ := total_garden_area - total_bed_area

theorem walkway_area_correct : walkway_area = 296 := by
  sorry

end walkway_area_correct_l2283_228367


namespace gcd_60_75_l2283_228333

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l2283_228333


namespace village_distance_l2283_228358

theorem village_distance
  (d : ℝ)
  (uphill_speed : ℝ) (downhill_speed : ℝ)
  (total_time : ℝ)
  (h1 : uphill_speed = 15)
  (h2 : downhill_speed = 30)
  (h3 : total_time = 4) :
  d = 40 :=
by
  sorry

end village_distance_l2283_228358


namespace monotonic_increasing_interval_l2283_228360

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem monotonic_increasing_interval :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → ∀ (x1 x2 : ℝ), 0 ≤ x1 ∧ x1 ≤ 1 → 0 ≤ x2 ∧ x2 ≤ 1 → x1 ≤ x2 → sqrt (- x1 ^ 2 + 2 * x1) ≤ sqrt (- x2 ^ 2 + 2 * x2) :=
sorry

end monotonic_increasing_interval_l2283_228360


namespace aftershave_lotion_volume_l2283_228380

theorem aftershave_lotion_volume (V : ℝ) (h1 : 0.30 * V = 0.1875 * (V + 30)) : V = 50 := 
by 
-- sorry is added to indicate proof is omitted.
sorry

end aftershave_lotion_volume_l2283_228380


namespace at_least_one_casket_made_by_Cellini_son_l2283_228301

-- Definitions for casket inscriptions
def golden_box := "The silver casket was made by Cellini"
def silver_box := "The golden casket was made by someone other than Cellini"

-- Predicate indicating whether a box was made by Cellini
def made_by_Cellini (box : String) : Prop :=
  box = "The golden casket was made by someone other than Cellini" ∨ box = "The silver casket was made by Cellini"

-- Our goal is to prove that at least one of the boxes was made by Cellini's son
theorem at_least_one_casket_made_by_Cellini_son :
  (¬ made_by_Cellini golden_box ∧ made_by_Cellini silver_box) ∨ (made_by_Cellini golden_box ∧ ¬ made_by_Cellini silver_box) → (¬ made_by_Cellini golden_box ∨ ¬ made_by_Cellini silver_box) :=
sorry

end at_least_one_casket_made_by_Cellini_son_l2283_228301


namespace B_pow_2024_l2283_228359

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![Real.cos (Real.pi / 4), 0, -Real.sin (Real.pi / 4)],
    ![0, 1, 0],
    ![Real.sin (Real.pi / 4), 0, Real.cos (Real.pi / 4)]
  ]

theorem B_pow_2024 :
  B ^ 2024 = ![
    ![-1, 0, 0],
    ![0, 1, 0],
    ![0, 0, -1]
  ] :=
by
  sorry

end B_pow_2024_l2283_228359


namespace inner_hexagon_area_l2283_228365

-- Define necessary conditions in Lean 4
variable (a b c d e f : ℕ)
variable (a1 a2 a3 a4 a5 a6 : ℕ)

-- Congruent equilateral triangles conditions forming a hexagon
axiom congruent_equilateral_triangles_overlap : 
  a1 = 1 ∧ a2 = 1 ∧ a3 = 9 ∧ a4 = 9 ∧ a5 = 16 ∧ a6 = 16

-- We want to show that the area of the inner hexagon is 38
theorem inner_hexagon_area : 
  a1 = 1 ∧ a2 = 1 ∧ a3 = 9 ∧ a4 = 9 ∧ a5 = 16 ∧ a6 = 16 → a = 38 :=
by
  intro h
  sorry

end inner_hexagon_area_l2283_228365


namespace find_pairs_l2283_228394

-- Definitions for the conditions in the problem
def is_positive (x : ℝ) : Prop := x > 0

def equations (x y : ℝ) : Prop :=
  (Real.log (x^2 + y^2) / Real.log 10 = 2) ∧ 
  (Real.log x / Real.log 2 - 4 = Real.log 3 / Real.log 2 - Real.log y / Real.log 2)

-- Lean 4 Statement
theorem find_pairs (x y : ℝ) : 
  is_positive x ∧ is_positive y ∧ equations x y → (x, y) = (8, 6) ∨ (x, y) = (6, 8) :=
by
  sorry

end find_pairs_l2283_228394


namespace john_saves_money_l2283_228361

def original_spending (coffees_per_day: ℕ) (price_per_coffee: ℕ) : ℕ :=
  coffees_per_day * price_per_coffee

def new_price (original_price: ℕ) (increase_percentage: ℕ) : ℕ :=
  original_price + (original_price * increase_percentage / 100)

def new_coffees_per_day (original_coffees_per_day: ℕ) (reduction_fraction: ℕ) : ℕ :=
  original_coffees_per_day / reduction_fraction

def current_spending (new_coffees_per_day: ℕ) (new_price_per_coffee: ℕ) : ℕ :=
  new_coffees_per_day * new_price_per_coffee

theorem john_saves_money
  (coffees_per_day : ℕ := 4)
  (price_per_coffee : ℕ := 2)
  (increase_percentage : ℕ := 50)
  (reduction_fraction : ℕ := 2) :
  original_spending coffees_per_day price_per_coffee
  - current_spending (new_coffees_per_day coffees_per_day reduction_fraction)
                     (new_price price_per_coffee increase_percentage)
  = 2 := by
{
  sorry
}

end john_saves_money_l2283_228361


namespace death_rate_is_three_l2283_228351

-- Let birth_rate be the average birth rate in people per two seconds
def birth_rate : ℕ := 6
-- Let net_population_increase be the net population increase per day
def net_population_increase : ℕ := 129600
-- Let seconds_per_day be the total number of seconds in a day
def seconds_per_day : ℕ := 86400

noncomputable def death_rate_per_two_seconds : ℕ :=
  let net_increase_per_second := net_population_increase / seconds_per_day
  let birth_rate_per_second := birth_rate / 2
  2 * (birth_rate_per_second - net_increase_per_second)

theorem death_rate_is_three :
  death_rate_per_two_seconds = 3 := by
  sorry

end death_rate_is_three_l2283_228351


namespace intersection_of_A_and_B_l2283_228316

open Set

noncomputable def A : Set ℝ := { x | (x - 2) / (x + 5) < 0 }
noncomputable def B : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

theorem intersection_of_A_and_B : A ∩ B = { x : ℝ | -5 < x ∧ x ≤ -1 } :=
sorry

end intersection_of_A_and_B_l2283_228316


namespace range_of_a_l2283_228304

theorem range_of_a (a : ℝ) : (∃ x > 0, (2 * x - a) / (x + 1) = 1) ↔ a > -1 :=
by {
    sorry
}

end range_of_a_l2283_228304


namespace total_peaches_l2283_228344

-- Definitions of conditions
def initial_peaches : ℕ := 13
def picked_peaches : ℕ := 55

-- Proof problem statement
theorem total_peaches : initial_peaches + picked_peaches = 68 :=
by
  -- Including sorry to skip the actual proof
  sorry

end total_peaches_l2283_228344


namespace overlapping_squares_proof_l2283_228324

noncomputable def overlapping_squares_area (s : ℝ) : ℝ :=
  let AB := s
  let MN := s
  let areaMN := s^2
  let intersection_area := areaMN / 4
  intersection_area

theorem overlapping_squares_proof (s : ℝ) :
  overlapping_squares_area s = s^2 / 4 := by
    -- proof would go here
    sorry

end overlapping_squares_proof_l2283_228324


namespace option_a_is_correct_l2283_228329

theorem option_a_is_correct (a b : ℝ) :
  (a - b) * (-a - b) = b^2 - a^2 :=
sorry

end option_a_is_correct_l2283_228329


namespace polyhedron_value_l2283_228332

theorem polyhedron_value (T H V E : ℕ) (h t : ℕ) 
  (F : ℕ) (h_eq : h = 10) (t_eq : t = 10)
  (F_eq : F = 20)
  (edges_eq : E = (3 * t + 6 * h) / 2)
  (vertices_eq : V = E - F + 2)
  (T_value : T = 2) (H_value : H = 2) :
  100 * H + 10 * T + V = 227 := by
  sorry

end polyhedron_value_l2283_228332


namespace compare_abc_l2283_228366

noncomputable def a : ℝ := 4 ^ (Real.log 2 / Real.log 3)
noncomputable def b : ℝ := 4 ^ (Real.log 6 / (2 * Real.log 3))
noncomputable def c : ℝ := 2 ^ (Real.sqrt 5)

theorem compare_abc : c > b ∧ b > a := 
by
  sorry

end compare_abc_l2283_228366


namespace find_total_price_l2283_228386

noncomputable def total_price (p : ℝ) : Prop := 0.20 * p = 240

theorem find_total_price (p : ℝ) (h : total_price p) : p = 1200 :=
by sorry

end find_total_price_l2283_228386


namespace convert_base_8_to_base_10_l2283_228391

def to_base_10 (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ digit acc => acc * b + digit) 0

theorem convert_base_8_to_base_10 : 
  to_base_10 10 8 [6, 4, 2] = 166 := by
  sorry

end convert_base_8_to_base_10_l2283_228391


namespace triple_root_possible_values_l2283_228325

-- Definitions and conditions
def polynomial (x : ℤ) (b3 b2 b1 : ℤ) := x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 24

-- The proof problem
theorem triple_root_possible_values 
  (r b3 b2 b1 : ℤ)
  (h_triple_root : polynomial r b3 b2 b1 = (x * (x - 1) * (x - 2)) * (x - r) ) :
  r = -2 ∨ r = -1 ∨ r = 1 ∨ r = 2 :=
by
  sorry

end triple_root_possible_values_l2283_228325


namespace necessary_and_sufficient_condition_l2283_228363

theorem necessary_and_sufficient_condition (x : ℝ) (h : x > 0) : (x + 1/x ≥ 2) ↔ (x > 0) :=
sorry

end necessary_and_sufficient_condition_l2283_228363


namespace ironing_pants_each_day_l2283_228303

-- Given conditions:
def minutes_ironing_shirt := 5 -- minutes per day
def days_per_week := 5 -- days per week
def total_minutes_ironing_4_weeks := 160 -- minutes over 4 weeks

-- Target statement to prove:
theorem ironing_pants_each_day : 
  (total_minutes_ironing_4_weeks / 4 - minutes_ironing_shirt * days_per_week) /
  days_per_week = 3 :=
by 
sorry

end ironing_pants_each_day_l2283_228303


namespace simplify_expression_l2283_228377

theorem simplify_expression :
  10 / (2 / 0.3) / (0.3 / 0.04) / (0.04 / 0.05) = 0.25 :=
by
  sorry

end simplify_expression_l2283_228377


namespace athul_downstream_distance_l2283_228357

-- Define the conditions
def upstream_distance : ℝ := 16
def upstream_time : ℝ := 4
def speed_of_stream : ℝ := 1
def downstream_time : ℝ := 4

-- Translate the conditions into properties and prove the downstream distance
theorem athul_downstream_distance (V : ℝ) 
  (h1 : upstream_distance = (V - speed_of_stream) * upstream_time) :
  (V + speed_of_stream) * downstream_time = 24 := 
by
  -- Given the conditions, the proof would be filled here
  sorry

end athul_downstream_distance_l2283_228357


namespace line_does_not_pass_through_fourth_quadrant_l2283_228375

theorem line_does_not_pass_through_fourth_quadrant
  (A B C : ℝ) (hAB : A * B < 0) (hBC : B * C < 0) :
  ¬ ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ A * x + B * y + C = 0 :=
by
  sorry

end line_does_not_pass_through_fourth_quadrant_l2283_228375


namespace average_speed_second_bus_l2283_228399

theorem average_speed_second_bus (x : ℝ) (h1 : x > 0) :
  (12 / x) - (12 / (1.2 * x)) = 3 / 60 :=
by
  sorry

end average_speed_second_bus_l2283_228399


namespace number_of_students_l2283_228388

theorem number_of_students (n : ℕ) (h1 : n < 50) (h2 : n % 8 = 5) (h3 : n % 6 = 4) : n = 13 :=
by
  sorry

end number_of_students_l2283_228388


namespace angle_between_hands_at_3_27_l2283_228370

noncomputable def minute_hand_angle (m : ℕ) : ℝ :=
  (m / 60.0) * 360.0

noncomputable def hour_hand_angle (h : ℕ) (m : ℕ) : ℝ :=
  ((h + m / 60.0) / 12.0) * 360.0

theorem angle_between_hands_at_3_27 : 
  minute_hand_angle 27 - hour_hand_angle 3 27 = 58.5 :=
by
  rw [minute_hand_angle, hour_hand_angle]
  simp
  sorry

end angle_between_hands_at_3_27_l2283_228370


namespace total_fencing_costs_l2283_228350

theorem total_fencing_costs (c1 c2 c3 c4 l1 l2 l3 : ℕ) 
    (h_c1 : c1 = 79) (h_c2 : c2 = 92) (h_c3 : c3 = 85) (h_c4 : c4 = 96)
    (h_l1 : l1 = 5) (h_l2 : l2 = 7) (h_l3 : l3 = 9) :
    (c1 + c2 + c3 + c4) * l1 = 1760 ∧ 
    (c1 + c2 + c3 + c4) * l2 = 2464 ∧ 
    (c1 + c2 + c3 + c4) * l3 = 3168 := 
by {
    sorry -- Proof to be constructed
}

end total_fencing_costs_l2283_228350


namespace total_spent_on_date_l2283_228328

-- Constants representing costs
def ticket_cost : ℝ := 10.00
def combo_meal_cost : ℝ := 11.00
def candy_cost : ℝ := 2.50

-- Numbers of items to buy
def num_tickets : ℝ := 2
def num_candies : ℝ := 2

-- Total cost calculation
def total_cost : ℝ := (ticket_cost * num_tickets) + (candy_cost * num_candies) + combo_meal_cost

-- Prove that the total cost is $36.00
theorem total_spent_on_date : total_cost = 36.00 := by
  sorry

end total_spent_on_date_l2283_228328


namespace parabola_constant_term_l2283_228368

theorem parabola_constant_term 
  (b c : ℝ)
  (h1 : 2 = 2 * (1 : ℝ)^2 + b * (1 : ℝ) + c)
  (h2 : 2 = 2 * (3 : ℝ)^2 + b * (3 : ℝ) + c) : 
  c = 8 :=
by
  sorry

end parabola_constant_term_l2283_228368


namespace edge_length_of_inscribed_cube_in_sphere_l2283_228397

noncomputable def edge_length_of_cube_in_sphere (surface_area_sphere : ℝ) (π_cond : surface_area_sphere = 36 * Real.pi) : ℝ :=
  let x := 2 * Real.sqrt 3
  x

theorem edge_length_of_inscribed_cube_in_sphere (surface_area_sphere : ℝ) (π_cond : surface_area_sphere = 36 * Real.pi) :
  edge_length_of_cube_in_sphere surface_area_sphere π_cond = 2 * Real.sqrt 3 :=
by
  sorry

end edge_length_of_inscribed_cube_in_sphere_l2283_228397


namespace shaded_region_area_is_15_l2283_228382

noncomputable def area_of_shaded_region : ℝ :=
  let radius := 1
  let area_of_one_circle := Real.pi * (radius ^ 2)
  4 * area_of_one_circle + 3 * (4 - area_of_one_circle)

theorem shaded_region_area_is_15 : 
  abs (area_of_shaded_region - 15) < 1 :=
by
  exact sorry

end shaded_region_area_is_15_l2283_228382


namespace polygon_sides_l2283_228385

theorem polygon_sides {R : ℝ} (hR : R > 0) : 
  (∃ n : ℕ, n > 2 ∧ (1/2) * n * R^2 * Real.sin (2 * Real.pi / n) = 4 * R^2) → 
  ∃ n : ℕ, n = 15 :=
by
  sorry

end polygon_sides_l2283_228385


namespace total_apples_eq_l2283_228342

-- Define the conditions for the problem
def baskets : ℕ := 37
def apples_per_basket : ℕ := 17

-- Define the theorem to prove the total number of apples
theorem total_apples_eq : baskets * apples_per_basket = 629 :=
by
  sorry

end total_apples_eq_l2283_228342


namespace value_of_a_l2283_228339

theorem value_of_a (x a : ℝ) (h1 : 0 < x) (h2 : x < 1 / a) (h3 : ∀ x, x * (1 - a * x) ≤ 1 / 12) : a = 3 :=
sorry

end value_of_a_l2283_228339


namespace drummer_difference_l2283_228373

def flute_players : Nat := 5
def trumpet_players : Nat := 3 * flute_players
def trombone_players : Nat := trumpet_players - 8
def clarinet_players : Nat := 2 * flute_players
def french_horn_players : Nat := trombone_players + 3
def total_seats_needed : Nat := 65
def total_seats_taken : Nat := flute_players + trumpet_players + trombone_players + clarinet_players + french_horn_players
def drummers : Nat := total_seats_needed - total_seats_taken

theorem drummer_difference : drummers - trombone_players = 11 := by
  sorry

end drummer_difference_l2283_228373


namespace find_f_2024_l2283_228372

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_f_2024 (a b c : ℝ)
  (h1 : f 2021 a b c = 2021)
  (h2 : f 2022 a b c = 2022)
  (h3 : f 2023 a b c = 2023) :
  f 2024 a b c = 2030 := sorry

end find_f_2024_l2283_228372


namespace proof_problem_l2283_228315

def setA : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def setB : Set ℝ := {x | -1 < x ∧ x < 3}

def complementB : Set ℝ := {x | x ≥ 3 ∨ x ≤ -1}
def intersection : Set ℝ := {x | 3 ≤ x ∧ x ≤ 5}

theorem proof_problem :
  (setA ∩ complementB) = intersection := 
by
  sorry

end proof_problem_l2283_228315


namespace parallel_lines_condition_l2283_228337

theorem parallel_lines_condition (a : ℝ) : 
    (∀ x y : ℝ, 2 * x + a * y + 2 ≠ (a - 1) * x + y - 2) ↔ a = 2 := 
sorry

end parallel_lines_condition_l2283_228337


namespace right_triangle_hypotenuse_l2283_228318

theorem right_triangle_hypotenuse (a b : ℕ) (h₁ : a = 3) (h₂ : b = 5) : 
  ∃ h : ℝ, h = Real.sqrt (a^2 + b^2) ∧ h = Real.sqrt 34 := 
by
  sorry

end right_triangle_hypotenuse_l2283_228318


namespace problem_solution_l2283_228398

def diamond (x y k : ℝ) : ℝ := x^2 - k * y

theorem problem_solution (h : ℝ) (k : ℝ) (hk : k = 3) : 
  diamond h (diamond h h k) k = -2 * h^2 + 9 * h :=
by
  rw [hk, diamond, diamond]
  sorry

end problem_solution_l2283_228398


namespace expected_value_of_biased_die_l2283_228341

-- Define the probabilities
def P1 : ℚ := 1/10
def P2 : ℚ := 1/10
def P3 : ℚ := 2/10
def P4 : ℚ := 2/10
def P5 : ℚ := 2/10
def P6 : ℚ := 2/10

-- Define the outcomes
def X1 : ℚ := 1
def X2 : ℚ := 2
def X3 : ℚ := 3
def X4 : ℚ := 4
def X5 : ℚ := 5
def X6 : ℚ := 6

-- Define the expected value calculation according to the probabilities and outcomes
def expected_value : ℚ := P1 * X1 + P2 * X2 + P3 * X3 + P4 * X4 + P5 * X5 + P6 * X6

-- The theorem we want to prove
theorem expected_value_of_biased_die : expected_value = 3.9 := by
  -- We skip the proof here with sorry for now
  sorry

end expected_value_of_biased_die_l2283_228341


namespace union_A_B_eq_A_union_B_l2283_228392

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def B : Set ℝ := { x | x > 3 / 2 }

theorem union_A_B_eq_A_union_B :
  (A ∪ B) = { x | -1 ≤ x } :=
by
  sorry

end union_A_B_eq_A_union_B_l2283_228392


namespace cone_lateral_surface_area_l2283_228306

theorem cone_lateral_surface_area (r l : ℝ) (h_r : r = 2) (h_l : l = 3) : 
  (r * l * Real.pi = 6 * Real.pi) := by
  sorry

end cone_lateral_surface_area_l2283_228306


namespace angle_with_same_terminal_side_315_l2283_228313

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + β

theorem angle_with_same_terminal_side_315:
  same_terminal_side (-45) 315 :=
by
  sorry

end angle_with_same_terminal_side_315_l2283_228313


namespace solve_system_l2283_228323

theorem solve_system :
  (∃ x y : ℝ, 4 * x + y = 5 ∧ 2 * x - 3 * y = 13) ↔ (x = 2 ∧ y = -3) :=
by
  sorry

end solve_system_l2283_228323


namespace angle_skew_lines_range_l2283_228326

theorem angle_skew_lines_range (θ : ℝ) (h1 : 0 < θ) (h2 : θ ≤ 90) : 0 < θ ∧ θ ≤ 90 :=
by sorry

end angle_skew_lines_range_l2283_228326


namespace smallest_D_l2283_228374

theorem smallest_D {A B C D : ℕ} (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h2 : (A * 100 + B * 10 + C) * B = D * 1000 + C * 100 + B * 10 + D) : 
  D = 1 :=
sorry

end smallest_D_l2283_228374


namespace exists_prime_and_cube_root_l2283_228364

theorem exists_prime_and_cube_root (n : ℕ) (hn : 0 < n) :
  ∃ (p m : ℕ), p.Prime ∧ p % 6 = 5 ∧ ¬p ∣ n ∧ n ≡ m^3 [MOD p] :=
sorry

end exists_prime_and_cube_root_l2283_228364


namespace coloring_two_corners_removed_l2283_228378

noncomputable def coloring_count (total_ways : Nat) (ways_without_corner_a : Nat) : Nat :=
  total_ways - 2 * (total_ways - ways_without_corner_a) / 2 + 
  (ways_without_corner_a - (total_ways - ways_without_corner_a) / 2)

theorem coloring_two_corners_removed : coloring_count 120 96 = 78 := by
  sorry

end coloring_two_corners_removed_l2283_228378


namespace find_largest_N_l2283_228308

noncomputable def largest_N : ℕ :=
  by
    -- This proof needs to demonstrate the solution based on constraints.
    -- Proof will be filled here.
    sorry

theorem find_largest_N :
  largest_N = 44 := 
  by
    -- Proof to establish the largest N will be completed here.
    sorry

end find_largest_N_l2283_228308


namespace num_two_wheelers_wheels_eq_two_l2283_228393

variable (num_two_wheelers num_four_wheelers total_wheels : ℕ)

def total_wheels_eq : Prop :=
  2 * num_two_wheelers + 4 * num_four_wheelers = total_wheels

theorem num_two_wheelers_wheels_eq_two (h1 : num_four_wheelers = 13)
                                        (h2 : total_wheels = 54)
                                        (h_total_eq : total_wheels_eq num_two_wheelers num_four_wheelers total_wheels) :
  2 * num_two_wheelers = 2 :=
by
  unfold total_wheels_eq at h_total_eq
  sorry

end num_two_wheelers_wheels_eq_two_l2283_228393


namespace candy_totals_l2283_228314

-- Definitions of the conditions
def sandra_bags := 2
def sandra_pieces_per_bag := 6

def roger_bags1 := 11
def roger_bags2 := 3

def emily_bags1 := 4
def emily_bags2 := 7
def emily_bags3 := 5

-- Definitions of total pieces of candy
def sandra_total_candy := sandra_bags * sandra_pieces_per_bag
def roger_total_candy := roger_bags1 + roger_bags2
def emily_total_candy := emily_bags1 + emily_bags2 + emily_bags3

-- The proof statement
theorem candy_totals :
  sandra_total_candy = 12 ∧ roger_total_candy = 14 ∧ emily_total_candy = 16 :=
by
  -- Here we would provide the proof but we'll use sorry to skip it
  sorry

end candy_totals_l2283_228314


namespace carriages_people_equation_l2283_228340

theorem carriages_people_equation (x : ℕ) :
  3 * (x - 2) = 2 * x + 9 :=
sorry

end carriages_people_equation_l2283_228340


namespace ratio_a_b_l2283_228320

variables {x y a b : ℝ}

theorem ratio_a_b (h1 : 8 * x - 6 * y = a)
                  (h2 : 12 * y - 18 * x = b)
                  (hx : x ≠ 0)
                  (hy : y ≠ 0)
                  (hb : b ≠ 0) :
  a / b = -4 / 9 :=
sorry

end ratio_a_b_l2283_228320


namespace marble_problem_l2283_228302

theorem marble_problem
  (M : ℕ)
  (X : ℕ)
  (h1 : M = 18 * X)
  (h2 : M = 20 * (X - 1)) :
  M = 180 :=
by
  sorry

end marble_problem_l2283_228302


namespace find_a_b_l2283_228327

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

theorem find_a_b (a b c : ℝ) (h1 : (12 * a + b = 0)) (h2 : (4 * a + b = -3)) :
  a = 3 / 8 ∧ b = -9 / 2 := by
  sorry

end find_a_b_l2283_228327


namespace cos_arithmetic_sequence_l2283_228305

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem cos_arithmetic_sequence (a : ℕ → ℝ) (h_seq : arithmetic_sequence a) (h_sum : a 1 + a 5 + a 9 = 8 * Real.pi) :
  Real.cos (a 3 + a 7) = -1 / 2 :=
sorry

end cos_arithmetic_sequence_l2283_228305
