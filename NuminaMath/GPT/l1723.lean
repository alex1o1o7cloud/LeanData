import Mathlib

namespace NUMINAMATH_GPT_black_queen_awake_at_10_l1723_172346

-- Define the logical context
def king_awake_at_10 (king_asleep : Prop) : Prop :=
  king_asleep -> false

def king_asleep_at_10 (king_asleep : Prop) : Prop :=
  king_asleep

def queen_awake_at_10 (queen_asleep : Prop) : Prop :=
  queen_asleep -> false

-- Define the main theorem
theorem black_queen_awake_at_10 
  (king_asleep : Prop)
  (queen_asleep : Prop)
  (king_belief : king_asleep ↔ (king_asleep ∧ queen_asleep)) :
  queen_awake_at_10 queen_asleep :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_black_queen_awake_at_10_l1723_172346


namespace NUMINAMATH_GPT_tan_double_angle_solution_l1723_172392

theorem tan_double_angle_solution (x : ℝ) (h : Real.tan (x + Real.pi / 4) = 2) :
  (Real.tan x) / (Real.tan (2 * x)) = 4 / 9 :=
sorry

end NUMINAMATH_GPT_tan_double_angle_solution_l1723_172392


namespace NUMINAMATH_GPT_number_of_solutions_l1723_172372

theorem number_of_solutions (x y : ℕ) : (3 * x + 2 * y = 1001) → ∃! (n : ℕ), n = 167 := by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l1723_172372


namespace NUMINAMATH_GPT_find_p_l1723_172337

variable (A B C D p q u v w : ℝ)
variable (hu : u + v + w = -B / A)
variable (huv : u * v + v * w + w * u = C / A)
variable (huvw : u * v * w = -D / A)
variable (hpq : u^2 + v^2 = -p)
variable (hq : u^2 * v^2 = q)

theorem find_p (A B C D : ℝ) (u v w : ℝ) 
  (H1 : u + v + w = -B / A)
  (H2 : u * v + v * w + w * u = C / A)
  (H3 : u * v * w = -D / A)
  (H4 : v = -u - w)
  : p = (B^2 - 2 * C) / A^2 :=
by sorry

end NUMINAMATH_GPT_find_p_l1723_172337


namespace NUMINAMATH_GPT_A_inter_complement_B_eq_set_minus_one_to_zero_l1723_172344

open Set

theorem A_inter_complement_B_eq_set_minus_one_to_zero :
  let U := @univ ℝ
  let A := {x : ℝ | x < 0}
  let B := {x : ℝ | x ≤ -1}
  A ∩ (U \ B) = {x : ℝ | -1 < x ∧ x < 0} := 
by
  sorry

end NUMINAMATH_GPT_A_inter_complement_B_eq_set_minus_one_to_zero_l1723_172344


namespace NUMINAMATH_GPT_units_digit_of_3_pow_1987_l1723_172381

theorem units_digit_of_3_pow_1987 : 3 ^ 1987 % 10 = 7 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_3_pow_1987_l1723_172381


namespace NUMINAMATH_GPT_hockey_championship_max_k_volleyball_championship_max_k_l1723_172365

theorem hockey_championship_max_k : ∃ (k : ℕ), 0 < k ∧ k ≤ 20 ∧ k = 18 :=
by
  -- proof goes here
  sorry

theorem volleyball_championship_max_k : ∃ (k : ℕ), 0 < k ∧ k ≤ 20 ∧ k = 15 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_hockey_championship_max_k_volleyball_championship_max_k_l1723_172365


namespace NUMINAMATH_GPT_max_rectangle_area_l1723_172396

theorem max_rectangle_area (P : ℝ) (hP : 0 < P) : 
  ∃ (x y : ℝ), (2*x + 2*y = P) ∧ (x * y = P ^ 2 / 16) :=
by
  sorry

end NUMINAMATH_GPT_max_rectangle_area_l1723_172396


namespace NUMINAMATH_GPT_find_negative_a_l1723_172323

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ 0 then -x else 3 * x - 22

theorem find_negative_a (a : ℝ) (ha : a < 0) :
  g (g (g 7)) = g (g (g a)) ↔ a = -23 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_negative_a_l1723_172323


namespace NUMINAMATH_GPT_quadratic_solution_l1723_172347

theorem quadratic_solution (m n : ℝ) (h1 : m ≠ 0) (h2 : m * 1^2 + n * 1 - 1 = 0) : m + n = 1 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_l1723_172347


namespace NUMINAMATH_GPT_allocate_plots_l1723_172369

theorem allocate_plots (x y : ℕ) (h : x > y) : 
  ∃ u v : ℕ, (u^2 + v^2 = 2 * (x^2 + y^2)) :=
by
  sorry

end NUMINAMATH_GPT_allocate_plots_l1723_172369


namespace NUMINAMATH_GPT_proof_problem_l1723_172325

theorem proof_problem (a1 a2 a3 : ℕ) (h1 : a1 = a2 - 1) (h2 : a3 = a2 + 1) : 
  a2^3 ∣ (a1 * a2 * a3 + a2) :=
by sorry

end NUMINAMATH_GPT_proof_problem_l1723_172325


namespace NUMINAMATH_GPT_solution_set_correct_l1723_172336

def inequality_solution (x : ℝ) : Prop :=
  (x - 1) * (x - 2) * (x - 3)^2 > 0

theorem solution_set_correct : 
  ∀ x : ℝ, inequality_solution x ↔ (x < 1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 3) ∨ x > 3) := 
by sorry

end NUMINAMATH_GPT_solution_set_correct_l1723_172336


namespace NUMINAMATH_GPT_product_of_w_and_z_l1723_172384

variable (EF FG GH HE : ℕ)
variable (w z : ℕ)

-- Conditions from the problem
def parallelogram_conditions : Prop :=
  EF = 42 ∧ FG = 4 * z^3 ∧ GH = 3 * w + 6 ∧ HE = 32 ∧ EF = GH ∧ FG = HE

-- The proof problem proving the requested product given the conditions
theorem product_of_w_and_z (h : parallelogram_conditions EF FG GH HE w z) : (w * z) = 24 :=
by
  sorry

end NUMINAMATH_GPT_product_of_w_and_z_l1723_172384


namespace NUMINAMATH_GPT_geometric_sequence_min_value_l1723_172348

theorem geometric_sequence_min_value
  (s : ℝ) (b1 b2 b3 : ℝ)
  (h1 : b1 = 2)
  (h2 : b2 = 2 * s)
  (h3 : b3 = 2 * s ^ 2) :
  ∃ (s : ℝ), 3 * b2 + 4 * b3 = -9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_min_value_l1723_172348


namespace NUMINAMATH_GPT_train_or_plane_not_ship_possible_modes_l1723_172378

-- Define the probabilities of different modes of transportation
def P_train : ℝ := 0.3
def P_ship : ℝ := 0.2
def P_car : ℝ := 0.1
def P_plane : ℝ := 0.4

-- 1. Proof that probability of train or plane is 0.7
theorem train_or_plane : P_train + P_plane = 0.7 :=
by sorry

-- 2. Proof that probability of not taking a ship is 0.8
theorem not_ship : 1 - P_ship = 0.8 :=
by sorry

-- 3. Proof that if probability is 0.5, the modes are either (ship, train) or (car, plane)
theorem possible_modes (P_value : ℝ) (h1 : P_value = 0.5) :
  (P_ship + P_train = P_value) ∨ (P_car + P_plane = P_value) :=
by sorry

end NUMINAMATH_GPT_train_or_plane_not_ship_possible_modes_l1723_172378


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_l1723_172373

noncomputable def is_arithmetic (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∃ a1 : ℚ, ∀ n : ℕ, a n = a1 + n * d

noncomputable def sum_of_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  n * (a 0 + a (n - 1)) / 2

theorem sum_arithmetic_sequence (a : ℕ → ℚ) (h_arith : is_arithmetic a)
  (h1 : 2 * a 3 = 5) (h2 : a 4 + a 12 = 9) : sum_of_first_n_terms a 10 = 35 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_l1723_172373


namespace NUMINAMATH_GPT_painted_cube_l1723_172350

noncomputable def cube_side_length : ℕ :=
  7

theorem painted_cube (painted_faces: ℕ) (one_side_painted_cubes: ℕ) (orig_side_length: ℕ) :
    painted_faces = 6 ∧ one_side_painted_cubes = 54 ∧ (orig_side_length + 2) ^ 2 / 6 = 9 →
    orig_side_length = cube_side_length :=
by
  sorry

end NUMINAMATH_GPT_painted_cube_l1723_172350


namespace NUMINAMATH_GPT_temperature_difference_l1723_172329

/-- The average temperature at the top of Mount Tai. -/
def T_top : ℝ := -9

/-- The average temperature at the foot of Mount Tai. -/
def T_foot : ℝ := -1

/-- The temperature difference between the average temperature at the foot and the top of Mount Tai is 8 degrees Celsius. -/
theorem temperature_difference : T_foot - T_top = 8 := by
  sorry

end NUMINAMATH_GPT_temperature_difference_l1723_172329


namespace NUMINAMATH_GPT_perfect_squares_divide_l1723_172310

-- Define the problem and the conditions as Lean definitions
def numFactors (base exponent : ℕ) := (exponent / 2) + 1

def countPerfectSquareFactors : ℕ := 
  let choices2 := numFactors 2 3
  let choices3 := numFactors 3 5
  let choices5 := numFactors 5 7
  let choices7 := numFactors 7 9
  choices2 * choices3 * choices5 * choices7

theorem perfect_squares_divide (numFactors : (ℕ → ℕ → ℕ)) 
(countPerfectSquareFactors : ℕ) : countPerfectSquareFactors = 120 :=
by
  -- We skip the proof here
  -- Proof steps would go here if needed
  sorry

end NUMINAMATH_GPT_perfect_squares_divide_l1723_172310


namespace NUMINAMATH_GPT_percentage_yield_l1723_172389

theorem percentage_yield (market_price annual_dividend : ℝ) (yield : ℝ) 
  (H1 : yield = 0.12)
  (H2 : market_price = 125)
  (H3 : annual_dividend = yield * market_price) :
  (annual_dividend / market_price) * 100 = 12 := 
sorry

end NUMINAMATH_GPT_percentage_yield_l1723_172389


namespace NUMINAMATH_GPT_yz_zx_xy_minus_2xyz_leq_7_27_l1723_172360

theorem yz_zx_xy_minus_2xyz_leq_7_27 (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  (y * z + z * x + x * y - 2 * x * y * z) ≤ 7 / 27 := 
by 
  sorry

end NUMINAMATH_GPT_yz_zx_xy_minus_2xyz_leq_7_27_l1723_172360


namespace NUMINAMATH_GPT_net_effect_on_sale_value_l1723_172367

theorem net_effect_on_sale_value 
  (P Original_Sales_Volume : ℝ) 
  (reduced_by : ℝ := 0.18) 
  (sales_increase : ℝ := 0.88) 
  (additional_tax : ℝ := 0.12) :
  P * Original_Sales_Volume * ((1 - reduced_by) * (1 + additional_tax) * (1 + sales_increase) - 1) = P * Original_Sales_Volume * 0.7184 :=
  by
  sorry

end NUMINAMATH_GPT_net_effect_on_sale_value_l1723_172367


namespace NUMINAMATH_GPT_smallest_N_l1723_172358

-- Definitions corresponding to the conditions
def circular_table (chairs : ℕ) : Prop := chairs = 72

def proper_seating (N chairs : ℕ) : Prop :=
  ∀ (new_person : ℕ), new_person < chairs →
    (∃ seated, seated < N ∧ (seated - new_person).gcd chairs = 1)

-- Problem statement
theorem smallest_N (chairs : ℕ) :
  circular_table chairs →
  ∃ N, proper_seating N chairs ∧ (∀ M < N, ¬ proper_seating M chairs) ∧ N = 18 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_smallest_N_l1723_172358


namespace NUMINAMATH_GPT_gcd_100_450_l1723_172371

theorem gcd_100_450 : Int.gcd 100 450 = 50 := 
by sorry

end NUMINAMATH_GPT_gcd_100_450_l1723_172371


namespace NUMINAMATH_GPT_greatest_four_digit_number_divisible_by_3_and_4_l1723_172377

theorem greatest_four_digit_number_divisible_by_3_and_4 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (n % 12 = 0) ∧ (∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ (m % 12 = 0) → m ≤ 9996) :=
by sorry

end NUMINAMATH_GPT_greatest_four_digit_number_divisible_by_3_and_4_l1723_172377


namespace NUMINAMATH_GPT_valid_quadratic_polynomials_l1723_172376

theorem valid_quadratic_polynomials (b c : ℤ)
  (h₁ : ∃ x₁ x₂ : ℤ, b = -(x₁ + x₂) ∧ c = x₁ * x₂)
  (h₂ : 1 + b + c = 10) :
  (b = -13 ∧ c = 22) ∨ (b = -9 ∧ c = 18) ∨ (b = 9 ∧ c = 0) ∨ (b = 5 ∧ c = 4) := sorry

end NUMINAMATH_GPT_valid_quadratic_polynomials_l1723_172376


namespace NUMINAMATH_GPT_max_omega_l1723_172338

noncomputable def f (ω φ x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem max_omega (ω φ : ℝ) (k k' : ℤ) (hω_pos : ω > 0) (hφ1 : 0 < φ)
  (hφ2 : φ < Real.pi / 2) (h1 : f ω φ (-Real.pi / 4) = 0)
  (h2 : ∀ x, f ω φ (Real.pi / 4 - x) = f ω φ (Real.pi / 4 + x))
  (h3 : ∀ x, x ∈ Set.Ioo (Real.pi / 18) (2 * Real.pi / 9) →
    Monotone (f ω φ)) :
  ω = 5 :=
sorry

end NUMINAMATH_GPT_max_omega_l1723_172338


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1723_172341

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m: ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Sum of the first n terms of a sequence
def sum_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Specific statement we want to prove
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a)
  (h_S9 : sum_seq a 9 = 72) :
  a 2 + a 4 + a 9 = 24 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1723_172341


namespace NUMINAMATH_GPT_xy_product_l1723_172351

theorem xy_product (x y : ℝ) (h : x^2 + y^2 - 22*x - 20*y + 221 = 0) : x * y = 110 := 
sorry

end NUMINAMATH_GPT_xy_product_l1723_172351


namespace NUMINAMATH_GPT_ratio_new_circumference_to_original_diameter_l1723_172327

-- Define the problem conditions
variables (r k : ℝ) (hk : k > 0)

-- Define the Lean theorem to express the proof problem
theorem ratio_new_circumference_to_original_diameter (r k : ℝ) (hk : k > 0) :
  (π * (1 + k / r)) = (2 * π * (r + k)) / (2 * r) :=
by {
  -- Placeholder proof, to be filled in
  sorry
}

end NUMINAMATH_GPT_ratio_new_circumference_to_original_diameter_l1723_172327


namespace NUMINAMATH_GPT_max_x_plus_2y_l1723_172398

theorem max_x_plus_2y {x y : ℝ} (h : x^2 - x * y + y^2 = 1) :
  x + 2 * y ≤ (2 * Real.sqrt 21) / 3 :=
sorry

end NUMINAMATH_GPT_max_x_plus_2y_l1723_172398


namespace NUMINAMATH_GPT_heptagon_isosceles_same_color_l1723_172366

theorem heptagon_isosceles_same_color 
  (color : Fin 7 → Prop) (red blue : Prop)
  (h_heptagon : ∀ i : Fin 7, color i = red ∨ color i = blue) :
  ∃ (i j k : Fin 7), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ color i = color j ∧ color j = color k ∧ ((i + j) % 7 = k ∨ (j + k) % 7 = i ∨ (k + i) % 7 = j) :=
sorry

end NUMINAMATH_GPT_heptagon_isosceles_same_color_l1723_172366


namespace NUMINAMATH_GPT_line_through_points_C_D_has_undefined_slope_and_angle_90_l1723_172353

theorem line_through_points_C_D_has_undefined_slope_and_angle_90 (m : ℝ) (n : ℝ) (hn : n ≠ 0) :
  ∃ θ : ℝ, (∀ (slope : ℝ), false) ∧ θ = 90 :=
by { sorry }

end NUMINAMATH_GPT_line_through_points_C_D_has_undefined_slope_and_angle_90_l1723_172353


namespace NUMINAMATH_GPT_find_xyz_value_l1723_172305

noncomputable def xyz_satisfying_conditions (x y z : ℝ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧
  (x + 1/y = 5) ∧
  (y + 1/z = 2) ∧
  (z + 1/x = 3)

theorem find_xyz_value (x y z : ℝ) (h : xyz_satisfying_conditions x y z) : x * y * z = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_xyz_value_l1723_172305


namespace NUMINAMATH_GPT_foreign_objects_total_sum_l1723_172386

-- define the conditions
def dog_burrs : Nat := 12
def dog_ticks := 6 * dog_burrs
def dog_fleas := 3 * dog_ticks

def cat_burrs := 2 * dog_burrs
def cat_ticks := dog_ticks / 3
def cat_fleas := 4 * cat_ticks

-- calculate the total foreign objects
def total_dog := dog_burrs + dog_ticks + dog_fleas
def total_cat := cat_burrs + cat_ticks + cat_fleas

def total_objects := total_dog + total_cat

-- state the theorem
theorem foreign_objects_total_sum : total_objects = 444 := by
  sorry

end NUMINAMATH_GPT_foreign_objects_total_sum_l1723_172386


namespace NUMINAMATH_GPT_circle_equation_l1723_172319

-- Definitions for the given conditions
def line1 (x y : ℝ) : Prop := x + y + 2 = 0
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def line2 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def is_solution (x y : ℝ) : Prop := x^2 + y^2 - 6 * x - 6 * y - 16 = 0

-- Problem statement in Lean
theorem circle_equation : ∃ x y : ℝ, 
  (line1 x y ∧ circle1 x y ∧ line2 (x / 2) (x / 2)) → is_solution x y :=
sorry

end NUMINAMATH_GPT_circle_equation_l1723_172319


namespace NUMINAMATH_GPT_chess_tournament_boys_l1723_172356

noncomputable def num_boys_in_tournament (n k : ℕ) : Prop :=
  (6 + k * n = (n + 2) * (n + 1) / 2) ∧ (n > 2)

theorem chess_tournament_boys :
  ∃ (n : ℕ), num_boys_in_tournament n (if n = 5 then 3 else if n = 10 then 6 else 0) ∧ (n = 5 ∨ n = 10) :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_boys_l1723_172356


namespace NUMINAMATH_GPT_calculate_fraction_l1723_172304

theorem calculate_fraction :
  (5 * 6 - 4) / 8 = 13 / 4 := 
by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l1723_172304


namespace NUMINAMATH_GPT_perpendicular_lines_slope_l1723_172397

theorem perpendicular_lines_slope (a : ℝ) :
  (∀ x1 y1 x2 y2: ℝ, y1 = a * x1 - 2 ∧ y2 = x2 + 1 → (a * 1) = -1) → a = -1 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_slope_l1723_172397


namespace NUMINAMATH_GPT_prove_intersection_points_l1723_172385

noncomputable def sqrt5 := Real.sqrt 5

def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 5 / 2
def curve2 (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1
def curve3 (x y : ℝ) : Prop := x^2 + y^2 / 4 = 1
def curve4 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def line (x y : ℝ) : Prop := x + y = sqrt5

theorem prove_intersection_points :
  (∃! (x y : ℝ), curve1 x y ∧ line x y) ∧
  (∃! (x y : ℝ), curve3 x y ∧ line x y) ∧
  (∃! (x y : ℝ), curve4 x y ∧ line x y) :=
by
  sorry

end NUMINAMATH_GPT_prove_intersection_points_l1723_172385


namespace NUMINAMATH_GPT_complement_U_M_inter_N_eq_l1723_172370

def U : Set ℝ := Set.univ

def M : Set ℝ := { y | ∃ x, y = 2 * x + 1 ∧ -1/2 ≤ x ∧ x ≤ 1/2 }

def N : Set ℝ := { x | ∃ y, y = Real.log (x^2 + 3 * x) ∧ (x < -3 ∨ x > 0) }

def complement_U_M : Set ℝ := U \ M

theorem complement_U_M_inter_N_eq :
  (complement_U_M ∩ N) = ((Set.Iio (-3 : ℝ)) ∪ (Set.Ioi (2 : ℝ))) :=
sorry

end NUMINAMATH_GPT_complement_U_M_inter_N_eq_l1723_172370


namespace NUMINAMATH_GPT_mary_fruits_l1723_172391

noncomputable def totalFruitsLeft 
    (initial_apples: ℕ) (initial_oranges: ℕ) (initial_blueberries: ℕ) (initial_grapes: ℕ) (initial_kiwis: ℕ)
    (salad_apples: ℕ) (salad_oranges: ℕ) (salad_blueberries: ℕ)
    (snack_apples: ℕ) (snack_oranges: ℕ) (snack_kiwis: ℕ)
    (given_apples: ℕ) (given_oranges: ℕ) (given_blueberries: ℕ) (given_grapes: ℕ) (given_kiwis: ℕ) : ℕ :=
  let remaining_apples := initial_apples - salad_apples - snack_apples - given_apples
  let remaining_oranges := initial_oranges - salad_oranges - snack_oranges - given_oranges
  let remaining_blueberries := initial_blueberries - salad_blueberries - given_blueberries
  let remaining_grapes := initial_grapes - given_grapes
  let remaining_kiwis := initial_kiwis - snack_kiwis - given_kiwis
  remaining_apples + remaining_oranges + remaining_blueberries + remaining_grapes + remaining_kiwis

theorem mary_fruits :
    totalFruitsLeft 26 35 18 12 22 6 10 8 2 3 1 5 7 4 3 3 = 61 := by
  sorry

end NUMINAMATH_GPT_mary_fruits_l1723_172391


namespace NUMINAMATH_GPT_sphere_tangency_relation_l1723_172395

noncomputable def sphere_tangents (r R : ℝ) (h : R > r) :=
  (R >= (2 / (Real.sqrt 3) - 1) * r) ∧
  (∃ x, x = (R * (R + r - Real.sqrt (R^2 + 2 * R * r - r^2 / 3))) /
            (r + Real.sqrt (R^2 + 2 * R * r - r^2 / 3) - R)) 

theorem sphere_tangency_relation (r R: ℝ) (h : R > r) :
  sphere_tangents r R h :=
by
  sorry

end NUMINAMATH_GPT_sphere_tangency_relation_l1723_172395


namespace NUMINAMATH_GPT_cheaper_store_price_difference_in_cents_l1723_172300

theorem cheaper_store_price_difference_in_cents :
  let list_price : ℝ := 59.99
  let discount_budget_buys := list_price * 0.15
  let discount_frugal_finds : ℝ := 20
  let sale_price_budget_buys := list_price - discount_budget_buys
  let sale_price_frugal_finds := list_price - discount_frugal_finds
  let difference_in_price := sale_price_budget_buys - sale_price_frugal_finds
  let difference_in_cents := difference_in_price * 100
  difference_in_cents = 1099.15 :=
by
  sorry

end NUMINAMATH_GPT_cheaper_store_price_difference_in_cents_l1723_172300


namespace NUMINAMATH_GPT_find_z_l1723_172359

theorem find_z (x y : ℤ) (h1 : x * y + x + y = 106) (h2 : x^2 * y + x * y^2 = 1320) :
  x^2 + y^2 = 748 ∨ x^2 + y^2 = 5716 :=
sorry

end NUMINAMATH_GPT_find_z_l1723_172359


namespace NUMINAMATH_GPT_find_radius_of_sphere_l1723_172301

def radius_of_sphere (width : ℝ) (depth : ℝ) (r : ℝ) : Prop :=
  (width / 2) ^ 2 + (r - depth) ^ 2 = r ^ 2

theorem find_radius_of_sphere (r : ℝ) : radius_of_sphere 30 10 r → r = 16.25 :=
by
  intros h1
  -- sorry is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_find_radius_of_sphere_l1723_172301


namespace NUMINAMATH_GPT_bicycle_saves_time_l1723_172311

-- Define the conditions
def time_to_walk : ℕ := 98
def time_saved_by_bicycle : ℕ := 34

-- Prove the question equals the answer
theorem bicycle_saves_time :
  time_saved_by_bicycle = 34 := 
by
  sorry

end NUMINAMATH_GPT_bicycle_saves_time_l1723_172311


namespace NUMINAMATH_GPT_roots_of_quadratic_l1723_172383

theorem roots_of_quadratic (p q x1 x2 : ℕ) (hp : p + q = 28) (hroots : ∀ x, x^2 + p * x + q = 0 → (x = x1 ∨ x = x2)) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
  (x1 = 30 ∧ x2 = 2) ∨ (x1 = 2 ∧ x2 = 30) :=
sorry

end NUMINAMATH_GPT_roots_of_quadratic_l1723_172383


namespace NUMINAMATH_GPT_quadratic_expression_l1723_172315

theorem quadratic_expression (x1 x2 : ℝ) (h1 : x1^2 - 3 * x1 + 1 = 0) (h2 : x2^2 - 3 * x2 + 1 = 0) : 
  x1^2 - 2 * x1 + x2 = 2 :=
sorry

end NUMINAMATH_GPT_quadratic_expression_l1723_172315


namespace NUMINAMATH_GPT_perimeter_of_field_l1723_172328

theorem perimeter_of_field (b l : ℕ) (h1 : l = b + 30) (h2 : b * l = 18000) : 2 * (l + b) = 540 := 
by 
  -- Proof goes here
sorry

end NUMINAMATH_GPT_perimeter_of_field_l1723_172328


namespace NUMINAMATH_GPT_inequality_solution_set_l1723_172316

theorem inequality_solution_set (x : ℝ) : (x-1)/(x+2) > 1 → x < -2 := sorry

end NUMINAMATH_GPT_inequality_solution_set_l1723_172316


namespace NUMINAMATH_GPT_evaluate_expression_l1723_172339

theorem evaluate_expression : (532 * 532) - (531 * 533) = 1 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1723_172339


namespace NUMINAMATH_GPT_trees_per_square_meter_l1723_172340

-- Definitions of the given conditions
def side_length : ℕ := 100
def total_trees : ℕ := 120000

def area_of_street : ℤ := side_length * side_length
def area_of_forest : ℤ := 3 * area_of_street

-- The question translated to Lean theorem statement
theorem trees_per_square_meter (h1: area_of_street = side_length * side_length)
    (h2: area_of_forest = 3 * area_of_street) 
    (h3: total_trees = 120000) : 
    (total_trees / area_of_forest) = 4 :=
sorry

end NUMINAMATH_GPT_trees_per_square_meter_l1723_172340


namespace NUMINAMATH_GPT_carlotta_tantrum_time_l1723_172388

theorem carlotta_tantrum_time :
  (∀ (T P S : ℕ), 
   S = 6 ∧ T + P + S = 54 ∧ P = 3 * S → T = 5 * S) :=
by
  intro T P S
  rintro ⟨hS, hTotal, hPractice⟩
  sorry

end NUMINAMATH_GPT_carlotta_tantrum_time_l1723_172388


namespace NUMINAMATH_GPT_smallest_n_divisible_l1723_172390

theorem smallest_n_divisible (n : ℕ) : (15 * n - 3) % 11 = 0 ↔ n = 9 := by
  sorry

end NUMINAMATH_GPT_smallest_n_divisible_l1723_172390


namespace NUMINAMATH_GPT_diane_initial_amount_l1723_172324

theorem diane_initial_amount
  (X : ℝ)        -- the amount Diane started with
  (won_amount : ℝ := 65)
  (total_loss : ℝ := 215)
  (owing_friends : ℝ := 50)
  (final_amount := X + won_amount - total_loss - owing_friends) :
  X = 100 := 
by 
  sorry

end NUMINAMATH_GPT_diane_initial_amount_l1723_172324


namespace NUMINAMATH_GPT_team_total_score_l1723_172312

theorem team_total_score (Connor_score Amy_score Jason_score : ℕ)
  (h1 : Connor_score = 2)
  (h2 : Amy_score = Connor_score + 4)
  (h3 : Jason_score = 2 * Amy_score) :
  Connor_score + Amy_score + Jason_score = 20 :=
by
  sorry

end NUMINAMATH_GPT_team_total_score_l1723_172312


namespace NUMINAMATH_GPT_veranda_area_correct_l1723_172331

-- Define the dimensions of the room.
def room_length : ℕ := 20
def room_width : ℕ := 12

-- Define the width of the veranda.
def veranda_width : ℕ := 2

-- Calculate the total dimensions with the veranda.
def total_length : ℕ := room_length + 2 * veranda_width
def total_width : ℕ := room_width + 2 * veranda_width

-- Calculate the area of the room and the total area including the veranda.
def room_area : ℕ := room_length * room_width
def total_area : ℕ := total_length * total_width

-- Prove that the area of the veranda is 144 m².
theorem veranda_area_correct : total_area - room_area = 144 := by
  sorry

end NUMINAMATH_GPT_veranda_area_correct_l1723_172331


namespace NUMINAMATH_GPT_sqrt_sum_fractions_eq_l1723_172357

theorem sqrt_sum_fractions_eq :
  (Real.sqrt ((1 / 25) + (1 / 36)) = (Real.sqrt 61) / 30) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sum_fractions_eq_l1723_172357


namespace NUMINAMATH_GPT_sum_term_addition_l1723_172399

theorem sum_term_addition (k : ℕ) (hk : k ≥ 2) :
  (2^(k+1) - 1) - (2^k - 1) = 2^k := by
  sorry

end NUMINAMATH_GPT_sum_term_addition_l1723_172399


namespace NUMINAMATH_GPT_train_speed_l1723_172375

noncomputable def train_speed_kmph (L_t L_b : ℝ) (T : ℝ) : ℝ :=
  (L_t + L_b) / T * 3.6

theorem train_speed (L_t L_b : ℝ) (T : ℝ) :
  L_t = 110 ∧ L_b = 190 ∧ T = 17.998560115190784 → train_speed_kmph L_t L_b T = 60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_train_speed_l1723_172375


namespace NUMINAMATH_GPT_max_months_to_build_l1723_172334

theorem max_months_to_build (a b c x : ℝ) (h1 : 1/a + 1/b = 1/6)
                            (h2 : 1/a + 1/c = 1/5)
                            (h3 : 1/c + 1/b = 1/4)
                            (h4 : (1/a + 1/b + 1/c) * x = 1) :
                            x = 4 :=
sorry

end NUMINAMATH_GPT_max_months_to_build_l1723_172334


namespace NUMINAMATH_GPT_find_n_solution_l1723_172352

theorem find_n_solution : ∃ n : ℤ, (1 / (n + 1 : ℝ) + 2 / (n + 1 : ℝ) + (n : ℝ) / (n + 1 : ℝ) = 3) :=
by
  use 0
  sorry

end NUMINAMATH_GPT_find_n_solution_l1723_172352


namespace NUMINAMATH_GPT_number_of_solutions_l1723_172306

theorem number_of_solutions (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 5) :
  (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 4 → x = -2 :=
sorry

end NUMINAMATH_GPT_number_of_solutions_l1723_172306


namespace NUMINAMATH_GPT_number_of_truthful_dwarfs_l1723_172332

/-- 
Each of the 10 dwarfs either always tells the truth or always lies. 
Each dwarf likes exactly one type of ice cream: vanilla, chocolate, or fruit.
When asked, every dwarf raised their hand for liking vanilla ice cream.
When asked, 5 dwarfs raised their hand for liking chocolate ice cream.
When asked, only 1 dwarf raised their hand for liking fruit ice cream.
Prove that the number of truthful dwarfs is 4.
-/
theorem number_of_truthful_dwarfs (T L : ℕ) 
  (h1 : T + L = 10) 
  (h2 : T + 2 * L = 16) : 
  T = 4 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_number_of_truthful_dwarfs_l1723_172332


namespace NUMINAMATH_GPT_find_t_l1723_172320

theorem find_t :
  ∃ t : ℕ, 10 ≤ t ∧ t < 100 ∧ 13 * t % 100 = 52 ∧ t = 44 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l1723_172320


namespace NUMINAMATH_GPT_tshirt_more_expensive_l1723_172349

-- Definitions based on given conditions
def jeans_price : ℕ := 30
def socks_price : ℕ := 5
def tshirt_price : ℕ := jeans_price / 2

-- Statement to prove (The t-shirt is $10 more expensive than the socks)
theorem tshirt_more_expensive : (tshirt_price - socks_price) = 10 :=
by
  rw [tshirt_price, socks_price]
  sorry  -- proof steps are omitted

end NUMINAMATH_GPT_tshirt_more_expensive_l1723_172349


namespace NUMINAMATH_GPT_find_m_l1723_172368

def g (x : ℤ) (A : ℤ) (B : ℤ) (C : ℤ) : ℤ := A * x^2 + B * x + C

theorem find_m (A B C m : ℤ) 
  (h1 : g 2 A B C = 0)
  (h2 : 100 < g 9 A B C ∧ g 9 A B C < 110)
  (h3 : 150 < g 10 A B C ∧ g 10 A B C < 160)
  (h4 : 10000 * m < g 200 A B C ∧ g 200 A B C < 10000 * (m + 1)) : 
  m = 16 :=
sorry

end NUMINAMATH_GPT_find_m_l1723_172368


namespace NUMINAMATH_GPT_sequence_geometric_and_general_term_sum_of_sequence_l1723_172321

theorem sequence_geometric_and_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ)
  (h1 : ∀ k : ℕ, S k = 2 * a k - k) : 
  (a 0 = 1) ∧ 
  (∀ k : ℕ, a (k + 1) = 2 * a k + 1) ∧ 
  (∀ k : ℕ, a k = 2^k - 1) :=
sorry

theorem sum_of_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ)
  (h1 : ∀ k : ℕ, a k = 2^k - 1)
  (h2 : ∀ k : ℕ, b k = 1 / a (k+1) + 1 / (a k * a (k+1))) :
  T n = 1 - 1 / (2^(n+1) - 1) :=
sorry

end NUMINAMATH_GPT_sequence_geometric_and_general_term_sum_of_sequence_l1723_172321


namespace NUMINAMATH_GPT_cube_volume_l1723_172394

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end NUMINAMATH_GPT_cube_volume_l1723_172394


namespace NUMINAMATH_GPT_calculate_total_money_l1723_172330

theorem calculate_total_money (n100 n50 n10 : ℕ) 
  (h1 : n100 = 2) (h2 : n50 = 5) (h3 : n10 = 10) : 
  (n100 * 100 + n50 * 50 + n10 * 10 = 550) :=
by
  sorry

end NUMINAMATH_GPT_calculate_total_money_l1723_172330


namespace NUMINAMATH_GPT_average_height_of_students_l1723_172379

theorem average_height_of_students (x : ℕ) (female_height male_height : ℕ) 
  (female_height_eq : female_height = 170) (male_height_eq : male_height = 185) 
  (ratio : 2 * x = x * 2) : 
  ((2 * x * male_height + x * female_height) / (2 * x + x) = 180) := 
by
  sorry

end NUMINAMATH_GPT_average_height_of_students_l1723_172379


namespace NUMINAMATH_GPT_initial_girls_is_11_l1723_172308

-- Definitions of initial parameters and transformations
def initially_girls_percent : ℝ := 0.35
def final_girls_percent : ℝ := 0.25
def three : ℝ := 3

-- 35% of the initial total is girls
def initially_girls (p : ℝ) : ℝ := initially_girls_percent * p
-- After three girls leave and three boys join, the count of girls
def final_girls (p : ℝ) : ℝ := initially_girls p - three

-- Using the condition that after the change, 25% are girls
def proof_problem : Prop := ∀ (p : ℝ), 
  (final_girls p) / p = final_girls_percent →
  (0.1 * p) = 3 → 
  initially_girls p = 11

-- The statement of the theorem to be proved in Lean 4
theorem initial_girls_is_11 : proof_problem := sorry

end NUMINAMATH_GPT_initial_girls_is_11_l1723_172308


namespace NUMINAMATH_GPT_cosine_sine_inequality_theorem_l1723_172393

theorem cosine_sine_inequality_theorem (θ : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → 
    x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔
    (π / 12 < θ ∧ θ < 5 * π / 12) :=
by
  sorry

end NUMINAMATH_GPT_cosine_sine_inequality_theorem_l1723_172393


namespace NUMINAMATH_GPT_apogee_reach_second_stage_model_engine_off_time_l1723_172355

-- Given conditions
def altitudes := [(0, 0), (1, 24), (2, 96), (4, 386), (5, 514), (6, 616), (9, 850), (13, 994), (14, 1000), (16, 976), (19, 850), (24, 400)]
def second_stage_curve (x : ℝ) : ℝ := -6 * x^2 + 168 * x - 176

-- Proof problems
theorem apogee_reach : (14, 1000) ∈ altitudes :=
sorry  -- Need to prove the inclusion of the apogee point in the table

theorem second_stage_model : 
    second_stage_curve 14 = 1000 ∧ 
    second_stage_curve 16 = 976 ∧ 
    second_stage_curve 19 = 850 ∧ 
    ∃ n, n = 4 :=
sorry  -- Need to prove the analytical expression is correct and n = 4

theorem engine_off_time : 
    ∃ t : ℝ, t = 14 + 5 * Real.sqrt 6 ∧ second_stage_curve t = 100 :=
sorry  -- Need to prove the engine off time calculation

end NUMINAMATH_GPT_apogee_reach_second_stage_model_engine_off_time_l1723_172355


namespace NUMINAMATH_GPT_total_word_count_is_5000_l1723_172343

def introduction : ℕ := 450
def conclusion : ℕ := 3 * introduction
def body_sections : ℕ := 4 * 800

def total_word_count : ℕ := introduction + conclusion + body_sections

theorem total_word_count_is_5000 : total_word_count = 5000 := 
by
  -- Lean proof code will go here.
  sorry

end NUMINAMATH_GPT_total_word_count_is_5000_l1723_172343


namespace NUMINAMATH_GPT_statue_original_cost_l1723_172322

theorem statue_original_cost (selling_price : ℝ) (profit_percent : ℝ) (original_cost : ℝ) 
  (h1 : selling_price = 620) (h2 : profit_percent = 25) : 
  original_cost = 496 :=
by
  have h3 : profit_percent / 100 + 1 = 1.25 := by sorry
  have h4 : 1.25 * original_cost = selling_price := by sorry
  have h5 : original_cost = 620 / 1.25 := by sorry
  have h6 : 620 / 1.25 = 496 := by sorry
  exact sorry

end NUMINAMATH_GPT_statue_original_cost_l1723_172322


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1723_172335

theorem isosceles_triangle_perimeter (a b : ℕ) (c : ℕ) 
  (h1 : a = 5) (h2 : b = 5) (h3 : c = 2) 
  (isosceles : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a) : 
  a + b + c = 12 :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1723_172335


namespace NUMINAMATH_GPT_part_one_part_two_l1723_172314

noncomputable def f (a x : ℝ) : ℝ :=
  |x + (1 / a)| + |x - a + 1|

theorem part_one (a : ℝ) (h : a > 0) (x : ℝ) : f a x ≥ 1 :=
sorry

theorem part_two (a : ℝ) (h : a > 0) : f a 3 < 11 / 2 → 2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4 :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l1723_172314


namespace NUMINAMATH_GPT_cube_lateral_surface_area_l1723_172362

theorem cube_lateral_surface_area (V : ℝ) (h_V : V = 125) : 
  ∃ A : ℝ, A = 100 :=
by
  sorry

end NUMINAMATH_GPT_cube_lateral_surface_area_l1723_172362


namespace NUMINAMATH_GPT_angle_tuvels_equiv_l1723_172380

-- Defining the conditions
def full_circle_tuvels : ℕ := 400
def degree_angle_in_circle : ℕ := 360
def specific_angle_degrees : ℕ := 45

-- Proof statement showing the equivalence
theorem angle_tuvels_equiv :
  (specific_angle_degrees * full_circle_tuvels) / degree_angle_in_circle = 50 :=
by
  sorry

end NUMINAMATH_GPT_angle_tuvels_equiv_l1723_172380


namespace NUMINAMATH_GPT_gcd_calculation_l1723_172333

theorem gcd_calculation :
  let a := 97^7 + 1
  let b := 97^7 + 97^3 + 1
  gcd a b = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_calculation_l1723_172333


namespace NUMINAMATH_GPT_ab_cd_value_l1723_172313

theorem ab_cd_value (a b c d : ℝ) 
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = 0) : 
  a * b + c * d = -31 :=
by
  sorry

end NUMINAMATH_GPT_ab_cd_value_l1723_172313


namespace NUMINAMATH_GPT_total_books_equals_45_l1723_172307

-- Define the number of books bought in each category
def adventure_books : ℝ := 13.0
def mystery_books : ℝ := 17.0
def crime_books : ℝ := 15.0

-- Total number of books bought
def total_books := adventure_books + mystery_books + crime_books

-- The theorem we need to prove
theorem total_books_equals_45 : total_books = 45.0 := by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_books_equals_45_l1723_172307


namespace NUMINAMATH_GPT_rotate_A_180_about_B_l1723_172317

-- Define the points A, B, and C
def A : ℝ × ℝ := (-4, 1)
def B : ℝ × ℝ := (-1, 4)
def C : ℝ × ℝ := (-1, 1)

-- Define the 180 degrees rotation about B
def rotate_180_about (p q : ℝ × ℝ) : ℝ × ℝ :=
  let translated_p := (p.1 - q.1, p.2 - q.2) 
  let rotated_p := (-translated_p.1, -translated_p.2)
  (rotated_p.1 + q.1, rotated_p.2 + q.2)

-- Prove the image of point A after a 180 degrees rotation about point B
theorem rotate_A_180_about_B : rotate_180_about A B = (2, 7) :=
by
  sorry

end NUMINAMATH_GPT_rotate_A_180_about_B_l1723_172317


namespace NUMINAMATH_GPT_negation_of_p_l1723_172345

-- Define the proposition p
def p : Prop := ∃ x : ℝ, x + 2 ≤ 0

-- Define the negation of p
def not_p : Prop := ∀ x : ℝ, x + 2 > 0

-- State the theorem that the negation of p is not_p
theorem negation_of_p : ¬ p = not_p := by 
  sorry -- Proof not provided

end NUMINAMATH_GPT_negation_of_p_l1723_172345


namespace NUMINAMATH_GPT_valid_cone_from_sector_l1723_172363

-- Given conditions
def sector_angle : ℝ := 300
def circle_radius : ℝ := 15

-- Definition of correct option E
def base_radius_E : ℝ := 12
def slant_height_E : ℝ := 15

theorem valid_cone_from_sector :
  ( (sector_angle / 360) * (2 * Real.pi * circle_radius) = 25 * Real.pi ) ∧
  (slant_height_E = circle_radius) ∧
  (base_radius_E = 12) ∧
  (15^2 = 12^2 + 9^2) :=
by
  -- This theorem states that given sector angle and circle radius, the valid option is E
  sorry

end NUMINAMATH_GPT_valid_cone_from_sector_l1723_172363


namespace NUMINAMATH_GPT_polygon_sides_l1723_172387

-- Given conditions
def is_interior_angle (angle : ℝ) : Prop :=
  angle = 150

-- The theorem to prove the number of sides
theorem polygon_sides (h : is_interior_angle 150) : ∃ n : ℕ, n = 12 :=
  sorry

end NUMINAMATH_GPT_polygon_sides_l1723_172387


namespace NUMINAMATH_GPT_prime_divisors_6270_l1723_172303

theorem prime_divisors_6270 : 
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 11 ∧ p5 = 19 ∧ 
  (p1 * p2 * p3 * p4 * p5 = 6270) ∧ 
  (Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5) ∧ 
  (∀ q, Nat.Prime q ∧ q ∣ 6270 → (q = p1 ∨ q = p2 ∨ q = p3 ∨ q = p4 ∨ q = p5)) := 
by 
  sorry

end NUMINAMATH_GPT_prime_divisors_6270_l1723_172303


namespace NUMINAMATH_GPT_profit_inequality_solution_l1723_172326

theorem profit_inequality_solution (x : ℝ) (h₁ : 1 ≤ x) (h₂ : x ≤ 10) :
  100 * 2 * (5 * x + 1 - 3 / x) ≥ 3000 ↔ 3 ≤ x ∧ x ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_profit_inequality_solution_l1723_172326


namespace NUMINAMATH_GPT_zookeeper_feeding_ways_l1723_172309

/-- We define the total number of ways the zookeeper can feed all the animals following the rules. -/
def feed_animal_ways : ℕ :=
  6 * 5^2 * 4^2 * 3^2 * 2^2 * 1^2

/-- Theorem statement: The number of ways to feed all the animals is 86400. -/
theorem zookeeper_feeding_ways : feed_animal_ways = 86400 :=
by
  sorry

end NUMINAMATH_GPT_zookeeper_feeding_ways_l1723_172309


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1723_172302

theorem solve_quadratic_eq (x y z w d X Y Z W : ℤ) 
    (h1 : w % 2 = z % 2) 
    (h2 : x = 2 * d * (X * Z - Y * W))
    (h3 : y = 2 * d * (X * W + Y * Z))
    (h4 : z = d * (X^2 + Y^2 - Z^2 - W^2))
    (h5 : w = d * (X^2 + Y^2 + Z^2 + W^2)) :
    x^2 + y^2 + z^2 = w^2 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1723_172302


namespace NUMINAMATH_GPT_find_smallest_x_l1723_172354

theorem find_smallest_x :
  ∃ (x : ℕ), x > 1 ∧ (x^2 % 1000 = x % 1000) ∧ x = 376 := by
  sorry

end NUMINAMATH_GPT_find_smallest_x_l1723_172354


namespace NUMINAMATH_GPT_egg_production_l1723_172364

theorem egg_production (n_chickens1 n_chickens2 n_eggs1 n_eggs2 n_days1 n_days2 : ℕ)
  (h1 : n_chickens1 = 6) (h2 : n_eggs1 = 30) (h3 : n_days1 = 5) (h4 : n_chickens2 = 10) (h5 : n_days2 = 8) :
  n_eggs2 = 80 :=
sorry

end NUMINAMATH_GPT_egg_production_l1723_172364


namespace NUMINAMATH_GPT_expected_profit_may_is_3456_l1723_172382

-- Given conditions as definitions
def february_profit : ℝ := 2000
def april_profit : ℝ := 2880
def growth_rate (x : ℝ) : Prop := (2000 * (1 + x)^2 = 2880)

-- The expected profit in May
def expected_may_profit (x : ℝ) : ℝ := april_profit * (1 + x)

-- The theorem to be proved based on the given conditions
theorem expected_profit_may_is_3456 (x : ℝ) (h : growth_rate x) (h_pos : x = (1:ℝ)/5) : 
    expected_may_profit x = 3456 :=
by sorry

end NUMINAMATH_GPT_expected_profit_may_is_3456_l1723_172382


namespace NUMINAMATH_GPT_min_dist_AB_l1723_172318

-- Definitions of the conditions
structure Point3D where
  x : Float
  y : Float
  z : Float

def O := Point3D.mk 0 0 0
def B := Point3D.mk (Float.sqrt 3) (Float.sqrt 2) 2

def dist (P Q : Point3D) : Float :=
  Float.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

-- Given points
variables (A : Point3D)
axiom AO_eq_1 : dist A O = 1

-- Minimum value of |AB|
theorem min_dist_AB : dist A B ≥ 2 := 
sorry

end NUMINAMATH_GPT_min_dist_AB_l1723_172318


namespace NUMINAMATH_GPT_haleys_car_distance_l1723_172374

theorem haleys_car_distance (fuel_ratio : ℕ) (distance_ratio : ℕ) (fuel_used : ℕ) (distance_covered : ℕ) 
   (h_ratio : fuel_ratio = 4) (h_distance_ratio : distance_ratio = 7) (h_fuel_used : fuel_used = 44) :
   distance_covered = 77 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_haleys_car_distance_l1723_172374


namespace NUMINAMATH_GPT_three_right_angled_triangles_l1723_172342

theorem three_right_angled_triangles 
  (a b c : ℕ)
  (h_area : 1/2 * (a * b) = 2 * (a + b + c))
  (h_pythagorean : a^2 + b^2 = c^2)
  (h_int_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  (a = 9 ∧ b = 40 ∧ c = 41) ∨ 
  (a = 10 ∧ b = 24 ∧ c = 26) ∨ 
  (a = 12 ∧ b = 16 ∧ c = 20) := 
sorry

end NUMINAMATH_GPT_three_right_angled_triangles_l1723_172342


namespace NUMINAMATH_GPT_problem_statement_l1723_172361

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f (x)

theorem problem_statement (f : ℝ → ℝ) :
  is_odd_function f →
  (∀ x : ℝ, f (x + 6) = f (x) + 3) →
  f 1 = 1 →
  f 2015 + f 2016 = 2015 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1723_172361
