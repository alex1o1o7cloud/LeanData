import Mathlib

namespace NUMINAMATH_GPT_equal_volume_rect_parallelepipeds_decomposable_equal_volume_prisms_decomposable_l354_35474

-- Definition of volumes for rectangular parallelepipeds
def volume_rect_parallelepiped (a b c: ℝ) : ℝ := a * b * c

-- Definition of volumes for prisms
def volume_prism (base_area height: ℝ) : ℝ := base_area * height

-- Definition of decomposability of rectangular parallelepipeds
def decomposable_rect_parallelepipeds (a1 b1 c1 a2 b2 c2: ℝ) : Prop :=
  (volume_rect_parallelepiped a1 b1 c1) = (volume_rect_parallelepiped a2 b2 c2)

-- Lean statement for part (a)
theorem equal_volume_rect_parallelepipeds_decomposable (a1 b1 c1 a2 b2 c2: ℝ) (h: decomposable_rect_parallelepipeds a1 b1 c1 a2 b2 c2) :
  True := sorry

-- Definition of decomposability of prisms
def decomposable_prisms (base_area1 height1 base_area2 height2: ℝ) : Prop :=
  (volume_prism base_area1 height1) = (volume_prism base_area2 height2)

-- Lean statement for part (b)
theorem equal_volume_prisms_decomposable (base_area1 height1 base_area2 height2: ℝ) (h: decomposable_prisms base_area1 height1 base_area2 height2) :
  True := sorry

end NUMINAMATH_GPT_equal_volume_rect_parallelepipeds_decomposable_equal_volume_prisms_decomposable_l354_35474


namespace NUMINAMATH_GPT_find_solution_l354_35457

theorem find_solution (x y z : ℝ) :
  (x * (y^2 + z) = z * (z + x * y)) ∧ 
  (y * (z^2 + x) = x * (x + y * z)) ∧ 
  (z * (x^2 + y) = y * (y + x * z)) → 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_solution_l354_35457


namespace NUMINAMATH_GPT_find_y_l354_35486

variable {R : Type} [Field R] (y : R)

-- The condition: y = (1/y) * (-y) + 3
def condition (y : R) : Prop :=
  y = (1 / y) * (-y) + 3

-- The theorem to prove: under the condition, y = 2
theorem find_y (y : R) (h : condition y) : y = 2 := 
sorry

end NUMINAMATH_GPT_find_y_l354_35486


namespace NUMINAMATH_GPT_nicky_pace_l354_35468

theorem nicky_pace :
  ∃ v : ℝ, v = 3 ∧ (
    ∀ (head_start : ℝ) (cristina_pace : ℝ) (time : ℝ) (distance_encounter : ℝ), 
      head_start = 36 ∧ cristina_pace = 4 ∧ time = 36 ∧ distance_encounter = cristina_pace * time - head_start →
      distance_encounter / time = v
  ) :=
sorry

end NUMINAMATH_GPT_nicky_pace_l354_35468


namespace NUMINAMATH_GPT_lindas_nickels_l354_35497

theorem lindas_nickels
  (N : ℕ)
  (initial_dimes : ℕ := 2)
  (initial_quarters : ℕ := 6)
  (initial_nickels : ℕ := N)
  (additional_dimes : ℕ := 2)
  (additional_quarters : ℕ := 10)
  (additional_nickels : ℕ := 2 * N)
  (total_coins : ℕ := 35)
  (h : initial_dimes + initial_quarters + initial_nickels + additional_dimes + additional_quarters + additional_nickels = total_coins) :
  N = 5 := by
  sorry

end NUMINAMATH_GPT_lindas_nickels_l354_35497


namespace NUMINAMATH_GPT_phoenix_hike_distance_l354_35467

variable (a b c d : ℕ)

theorem phoenix_hike_distance
  (h1 : a + b = 24)
  (h2 : b + c = 30)
  (h3 : c + d = 32)
  (h4 : a + c = 28) :
  a + b + c + d = 56 :=
by
  sorry

end NUMINAMATH_GPT_phoenix_hike_distance_l354_35467


namespace NUMINAMATH_GPT_exists_root_in_interval_l354_35400

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x - 3

theorem exists_root_in_interval : ∃ c ∈ Set.Ioo (2 : ℝ) (3 : ℝ), f c = 0 :=
by
  sorry

end NUMINAMATH_GPT_exists_root_in_interval_l354_35400


namespace NUMINAMATH_GPT_smallest_integer_b_l354_35472

theorem smallest_integer_b (b : ℕ) : 27 ^ b > 3 ^ 9 ↔ b = 4 := by
  sorry

end NUMINAMATH_GPT_smallest_integer_b_l354_35472


namespace NUMINAMATH_GPT_totalTilesUsed_l354_35412

-- Define the dining room dimensions
def diningRoomLength : ℕ := 18
def diningRoomWidth : ℕ := 15

-- Define the border width
def borderWidth : ℕ := 2

-- Define tile dimensions
def tile1x1 : ℕ := 1
def tile2x2 : ℕ := 2

-- Calculate the number of tiles used along the length and width for the border
def borderTileCountLength : ℕ := 2 * 2 * (diningRoomLength - 2 * borderWidth)
def borderTileCountWidth : ℕ := 2 * 2 * (diningRoomWidth - 2 * borderWidth)

-- Total number of one-foot by one-foot tiles for the border
def totalBorderTileCount : ℕ := borderTileCountLength + borderTileCountWidth

-- Calculate the inner area dimensions
def innerLength : ℕ := diningRoomLength - 2 * borderWidth
def innerWidth : ℕ := diningRoomWidth - 2 * borderWidth
def innerArea : ℕ := innerLength * innerWidth

-- Number of two-foot by two-foot tiles needed
def tile2x2Count : ℕ := (innerArea + tile2x2 * tile2x2 - 1) / (tile2x2 * tile2x2) -- Ensures rounding up without floating point arithmetic

-- Prove that the total number of tiles used is 139
theorem totalTilesUsed : totalBorderTileCount + tile2x2Count = 139 := by
  sorry

end NUMINAMATH_GPT_totalTilesUsed_l354_35412


namespace NUMINAMATH_GPT_number_of_pencils_bought_l354_35456

-- Define the conditions
def cost_of_glue : ℕ := 270
def cost_per_pencil : ℕ := 210
def amount_paid : ℕ := 1000
def change_received : ℕ := 100

-- Define the statement to prove
theorem number_of_pencils_bought : 
  ∃ (n : ℕ), cost_of_glue + (cost_per_pencil * n) = amount_paid - change_received :=
by {
  sorry 
}

end NUMINAMATH_GPT_number_of_pencils_bought_l354_35456


namespace NUMINAMATH_GPT_find_x_from_arithmetic_mean_l354_35415

theorem find_x_from_arithmetic_mean (x : ℝ) 
  (h : (x + 10 + 18 + 3 * x + 16 + (x + 5) + (3 * x + 6)) / 6 = 25) : 
  x = 95 / 8 := by
  sorry

end NUMINAMATH_GPT_find_x_from_arithmetic_mean_l354_35415


namespace NUMINAMATH_GPT_fraction_multiplication_division_l354_35420

theorem fraction_multiplication_division :
  ((3 / 4) * (5 / 6)) / (7 / 8) = 5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_division_l354_35420


namespace NUMINAMATH_GPT_paper_cost_l354_35466
noncomputable section

variables (P C : ℝ)

theorem paper_cost (h : 100 * P + 200 * C = 6.00) : 
  20 * P + 40 * C = 1.20 :=
sorry

end NUMINAMATH_GPT_paper_cost_l354_35466


namespace NUMINAMATH_GPT_coffee_consumption_l354_35479

-- Defining the necessary variables and conditions
variable (Ivory_cons Brayan_cons : ℕ)
variable (hr : ℕ := 1)
variable (hrs : ℕ := 5)

-- Condition: Brayan drinks twice as much coffee as Ivory
def condition1 := Brayan_cons = 2 * Ivory_cons

-- Condition: Brayan drinks 4 cups of coffee in an hour
def condition2 := Brayan_cons = 4

-- The proof problem
theorem coffee_consumption : ∀ (Ivory_cons Brayan_cons : ℕ), (Brayan_cons = 2 * Ivory_cons) → 
  (Brayan_cons = 4) → 
  ((Brayan_cons * hrs) + (Ivory_cons * hrs) = 30) :=
by
  intro hBrayan hIvory hr
  sorry

end NUMINAMATH_GPT_coffee_consumption_l354_35479


namespace NUMINAMATH_GPT_perimeter_reduction_percentage_l354_35417

-- Given initial dimensions x and y
-- Initial Perimeter
def initial_perimeter (x y : ℝ) : ℝ := 2 * (x + y)

-- First reduction
def first_reduction_length (x : ℝ) : ℝ := 0.9 * x
def first_reduction_width (y : ℝ) : ℝ := 0.8 * y

-- New perimeter after first reduction
def new_perimeter_first (x y : ℝ) : ℝ := 2 * (first_reduction_length x + first_reduction_width y)

-- Condition: new perimeter is 88% of the initial perimeter
def perimeter_condition (x y : ℝ) : Prop := new_perimeter_first x y = 0.88 * initial_perimeter x y

-- Solve for x in terms of y
def solve_for_x (y : ℝ) : ℝ := 4 * y

-- Second reduction
def second_reduction_length (x : ℝ) : ℝ := 0.8 * x
def second_reduction_width (y : ℝ) : ℝ := 0.9 * y

-- New perimeter after second reduction
def new_perimeter_second (x y : ℝ) : ℝ := 2 * (second_reduction_length x + second_reduction_width y)

-- Proof statement
theorem perimeter_reduction_percentage (x y : ℝ) (h : perimeter_condition x y) : 
  new_perimeter_second x y = 0.82 * initial_perimeter x y :=
by
  sorry

end NUMINAMATH_GPT_perimeter_reduction_percentage_l354_35417


namespace NUMINAMATH_GPT_construct_length_one_l354_35489

theorem construct_length_one
    (a : ℝ) 
    (h_a : a = Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) : 
    ∃ (b : ℝ), b = 1 :=
by
    sorry

end NUMINAMATH_GPT_construct_length_one_l354_35489


namespace NUMINAMATH_GPT_running_time_l354_35460

variable (t : ℝ)
variable (v_j v_p d : ℝ)

-- Given conditions
variable (v_j : ℝ := 0.133333333333)  -- Joe's speed
variable (v_p : ℝ := 0.0666666666665) -- Pete's speed
variable (d : ℝ := 16)                -- Distance between them after t minutes

theorem running_time (h : v_j + v_p = 0.2 * t) : t = 80 :=
by
  -- Distance covered by Joe and Pete running in opposite directions
  have h1 : v_j * t + v_p * t = d := by sorry
  -- Given combined speeds
  have h2 : v_j + v_p = 0.2 := by sorry
  -- Using the equation to solve for time t
  exact sorry

end NUMINAMATH_GPT_running_time_l354_35460


namespace NUMINAMATH_GPT_inequality_example_l354_35473

theorem inequality_example (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
sorry

end NUMINAMATH_GPT_inequality_example_l354_35473


namespace NUMINAMATH_GPT_employed_females_percentage_l354_35416

variable (P : ℝ) -- Total population of town X
variable (E_P : ℝ) -- Percentage of the population that is employed
variable (M_E_P : ℝ) -- Percentage of the population that are employed males

-- Conditions
axiom h1 : E_P = 0.64
axiom h2 : M_E_P = 0.55

-- Target: Prove the percentage of employed people in town X that are females
theorem employed_females_percentage (h : P > 0) : 
  (E_P * P - M_E_P * P) / (E_P * P) * 100 = 14.06 := by
sorry

end NUMINAMATH_GPT_employed_females_percentage_l354_35416


namespace NUMINAMATH_GPT_price_of_brand_Y_pen_l354_35422

theorem price_of_brand_Y_pen (cost_X : ℝ) (num_X : ℕ) (total_pens : ℕ) (total_cost : ℝ) :
  cost_X = 4 ∧ num_X = 6 ∧ total_pens = 12 ∧ total_cost = 42 →
  (∃ (price_Y : ℝ), price_Y = 3) :=
by
  sorry

end NUMINAMATH_GPT_price_of_brand_Y_pen_l354_35422


namespace NUMINAMATH_GPT_initial_crayons_count_l354_35450

variable (x : ℕ) -- x represents the initial number of crayons

theorem initial_crayons_count (h1 : x + 3 = 12) : x = 9 := 
by sorry

end NUMINAMATH_GPT_initial_crayons_count_l354_35450


namespace NUMINAMATH_GPT_seven_a_plus_seven_b_l354_35409

noncomputable def g (x : ℝ) : ℝ := 7 * x - 6
noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 4
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem seven_a_plus_seven_b (a b : ℝ) (h₁ : ∀ x, g x = f_inv x - 2) (h₂ : ∀ x, f_inv (f x a b) = x) :
  7 * a + 7 * b = 5 :=
by
  sorry

end NUMINAMATH_GPT_seven_a_plus_seven_b_l354_35409


namespace NUMINAMATH_GPT_amount_lent_to_B_l354_35449

theorem amount_lent_to_B
  (rate_of_interest_per_annum : ℝ)
  (P_C : ℝ)
  (years_C : ℝ)
  (total_interest : ℝ)
  (years_B : ℝ)
  (IB : ℝ)
  (IC : ℝ)
  (P_B : ℝ):
  (rate_of_interest_per_annum = 10) →
  (P_C = 3000) →
  (years_C = 4) →
  (total_interest = 2200) →
  (years_B = 2) →
  (IC = (P_C * rate_of_interest_per_annum * years_C) / 100) →
  (IB = (P_B * rate_of_interest_per_annum * years_B) / 100) →
  (total_interest = IB + IC) →
  P_B = 5000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_amount_lent_to_B_l354_35449


namespace NUMINAMATH_GPT_domain_of_f_equals_l354_35425

noncomputable def domain_of_function := {x : ℝ | x > -1 ∧ -(x+4) * (x-1) > 0}

theorem domain_of_f_equals : domain_of_function = { x : ℝ | -1 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_equals_l354_35425


namespace NUMINAMATH_GPT_Yuna_boarding_place_l354_35413

-- Conditions
def Eunji_place : ℕ := 10
def people_after_Eunji : ℕ := 11

-- Proof Problem: Yuna's boarding place calculation
theorem Yuna_boarding_place :
  Eunji_place + people_after_Eunji + 1 = 22 :=
by
  sorry

end NUMINAMATH_GPT_Yuna_boarding_place_l354_35413


namespace NUMINAMATH_GPT_augmented_matrix_determinant_l354_35488

theorem augmented_matrix_determinant (m : ℝ) 
  (h : (1 - 2 * m) / (3 - 2) = 5) : 
  m = -2 :=
  sorry

end NUMINAMATH_GPT_augmented_matrix_determinant_l354_35488


namespace NUMINAMATH_GPT_problem_1_2_a_problem_1_2_b_l354_35426

theorem problem_1_2_a (x : ℝ) : x * (1 - x) ≤ 1 / 4 := sorry

theorem problem_1_2_b (x a : ℝ) : x * (a - x) ≤ a^2 / 4 := sorry

end NUMINAMATH_GPT_problem_1_2_a_problem_1_2_b_l354_35426


namespace NUMINAMATH_GPT_commensurable_iff_rat_l354_35403

def commensurable (A B : ℝ) : Prop :=
  ∃ d : ℝ, ∃ m n : ℤ, A = m * d ∧ B = n * d

theorem commensurable_iff_rat (A B : ℝ) :
  commensurable A B ↔ ∃ (m n : ℤ) (h : n ≠ 0), A / B = m / n :=
by
  sorry

end NUMINAMATH_GPT_commensurable_iff_rat_l354_35403


namespace NUMINAMATH_GPT_angie_pretzels_dave_pretzels_l354_35469

theorem angie_pretzels (B S A : ℕ) (hB : B = 12) (hS : S = B / 2) (hA : A = 3 * S) : A = 18 := by
  -- We state the problem using variables B, S, and A for Barry, Shelly, and Angie respectively
  sorry

theorem dave_pretzels (A S D : ℕ) (hA : A = 18) (hS : S = 12 / 2) (hD : D = 25 * (A + S) / 100) : D = 6 := by
  -- We use variables A and S from the first theorem, and introduce D for Dave
  sorry

end NUMINAMATH_GPT_angie_pretzels_dave_pretzels_l354_35469


namespace NUMINAMATH_GPT_previous_year_profit_percentage_l354_35430

theorem previous_year_profit_percentage (R : ℝ) (P : ℝ) :
  (0.16 * 0.70 * R = 1.1200000000000001 * (P / 100 * R)) → P = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_previous_year_profit_percentage_l354_35430


namespace NUMINAMATH_GPT_correct_total_score_l354_35465

theorem correct_total_score (total_score1 total_score2 : ℤ) : 
  (total_score1 = 5734 ∨ total_score2 = 5734) → (total_score1 = 5735 ∨ total_score2 = 5735) → 
  (total_score1 % 2 = 0 ∨ total_score2 % 2 = 0) → 
  (total_score1 ≠ total_score2) → 
  5734 % 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_correct_total_score_l354_35465


namespace NUMINAMATH_GPT_find_second_number_l354_35481

theorem find_second_number (n : ℕ) 
  (h1 : Nat.lcm 24 (Nat.lcm n 42) = 504)
  (h2 : 504 = 2^3 * 3^2 * 7) 
  (h3 : Nat.lcm 24 42 = 168) : n = 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_second_number_l354_35481


namespace NUMINAMATH_GPT_prob_correct_l354_35452

noncomputable def r : ℝ := (4.5 : ℝ)  -- derived from solving area and line equations
noncomputable def s : ℝ := (7.5 : ℝ)  -- derived from solving area and line equations

theorem prob_correct (P Q T : ℝ × ℝ)
  (hP : P = (9, 0))
  (hQ : Q = (0, 15))
  (hT : T = (r, s))
  (hline : s = -5/3 * r + 15)
  (harea : 2 * (1/2 * 9 * 15) = (1/2 * 9 * s) * 4) :
  r + s = 12 := by
  sorry

end NUMINAMATH_GPT_prob_correct_l354_35452


namespace NUMINAMATH_GPT_custom_op_evaluation_l354_35482

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x

theorem custom_op_evaluation : custom_op 6 4 - custom_op 4 6 = -6 :=
by
  sorry

end NUMINAMATH_GPT_custom_op_evaluation_l354_35482


namespace NUMINAMATH_GPT_prob_score_5_points_is_three_over_eight_l354_35493

noncomputable def probability_of_scoring_5_points : ℚ :=
  let total_events := 2^3
  let favorable_events := 3 -- Calculated from combinatorial logic.
  favorable_events / total_events

theorem prob_score_5_points_is_three_over_eight :
  probability_of_scoring_5_points = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_prob_score_5_points_is_three_over_eight_l354_35493


namespace NUMINAMATH_GPT_maximize_profit_l354_35496

theorem maximize_profit 
  (cost_per_product : ℝ)
  (initial_price : ℝ)
  (initial_sales : ℝ)
  (price_increase_effect : ℝ)
  (daily_sales_decrease : ℝ)
  (max_profit_price : ℝ)
  (max_profit : ℝ)
  :
  cost_per_product = 8 ∧ initial_price = 10 ∧ initial_sales = 100 ∧ price_increase_effect = 1 ∧ daily_sales_decrease = 10 → 
  max_profit_price = 14 ∧
  max_profit = 360 :=
by 
  intro h
  have h_cost := h.1
  have h_initial_price := h.2.1
  have h_initial_sales := h.2.2.1
  have h_price_increase_effect := h.2.2.2.1
  have h_daily_sales_decrease := h.2.2.2.2
  sorry

end NUMINAMATH_GPT_maximize_profit_l354_35496


namespace NUMINAMATH_GPT_purchase_price_l354_35410

noncomputable def cost_price_after_discount (P : ℝ) : ℝ :=
  0.8 * P + 375

theorem purchase_price {P : ℝ} (h : 1.15 * P = 18400) : cost_price_after_discount P = 13175 := by
  sorry

end NUMINAMATH_GPT_purchase_price_l354_35410


namespace NUMINAMATH_GPT_polygon_sides_in_arithmetic_progression_l354_35470

theorem polygon_sides_in_arithmetic_progression 
  (a : ℕ → ℝ) (n : ℕ) (h1: ∀ i, 1 ≤ i ∧ i ≤ n → a i = a 1 + (i - 1) * 10) 
  (h2 : a n = 150) : n = 12 :=
sorry

end NUMINAMATH_GPT_polygon_sides_in_arithmetic_progression_l354_35470


namespace NUMINAMATH_GPT_dot_product_of_vectors_l354_35408

theorem dot_product_of_vectors :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (-1, 2)
  a.1 * b.1 + a.2 * b.2 = -4 :=
by
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (-1, 2)
  sorry

end NUMINAMATH_GPT_dot_product_of_vectors_l354_35408


namespace NUMINAMATH_GPT_updated_mean_of_decremented_observations_l354_35431

theorem updated_mean_of_decremented_observations (mean : ℝ) (n : ℕ) (decrement : ℝ) 
  (h_mean : mean = 200) (h_n : n = 50) (h_decrement : decrement = 47) : 
  (mean * n - decrement * n) / n = 153 := 
by 
  sorry

end NUMINAMATH_GPT_updated_mean_of_decremented_observations_l354_35431


namespace NUMINAMATH_GPT_value_of_expression_l354_35436

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 2005 = -2004 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l354_35436


namespace NUMINAMATH_GPT_car_overtakes_truck_l354_35461

theorem car_overtakes_truck 
  (car_speed : ℝ)
  (truck_speed : ℝ)
  (car_arrival_time : ℝ)
  (truck_arrival_time : ℝ)
  (route_same : Prop)
  (time_difference : ℝ)
  (car_speed_km_min : car_speed = 66 / 60)
  (truck_speed_km_min : truck_speed = 42 / 60)
  (arrival_time_difference : truck_arrival_time - car_arrival_time = 18 / 60) :
  ∃ d : ℝ, d = 34.65 := 
by {
  sorry
}

end NUMINAMATH_GPT_car_overtakes_truck_l354_35461


namespace NUMINAMATH_GPT_y_share_per_rupee_of_x_l354_35433

theorem y_share_per_rupee_of_x (share_y : ℝ) (total_amount : ℝ) (z_per_x : ℝ) (y_per_x : ℝ) 
  (h1 : share_y = 54) 
  (h2 : total_amount = 210) 
  (h3 : z_per_x = 0.30) 
  (h4 : share_y = y_per_x * (total_amount / (1 + y_per_x + z_per_x))) : 
  y_per_x = 0.45 :=
sorry

end NUMINAMATH_GPT_y_share_per_rupee_of_x_l354_35433


namespace NUMINAMATH_GPT_simplify_expression_l354_35445

theorem simplify_expression : (2468 * 2468) / (2468 + 2468) = 1234 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l354_35445


namespace NUMINAMATH_GPT_johns_age_l354_35444

variable (J : ℕ)

theorem johns_age :
  J - 5 = (1 / 2) * (J + 8) → J = 18 := by
    sorry

end NUMINAMATH_GPT_johns_age_l354_35444


namespace NUMINAMATH_GPT_total_time_last_two_videos_l354_35411

theorem total_time_last_two_videos
  (first_video_length : ℕ := 2 * 60)
  (second_video_length : ℕ := 4 * 60 + 30)
  (total_time : ℕ := 510) :
  ∃ t1 t2 : ℕ, t1 ≠ t2 ∧ t1 + t2 = total_time - first_video_length - second_video_length := by
  sorry

end NUMINAMATH_GPT_total_time_last_two_videos_l354_35411


namespace NUMINAMATH_GPT_new_student_weight_l354_35440

theorem new_student_weight 
  (w_avg : ℝ)
  (w_new : ℝ)
  (condition : (5 * w_avg - 72 = 5 * (w_avg - 12) + w_new)) 
  : w_new = 12 := 
  by 
  sorry

end NUMINAMATH_GPT_new_student_weight_l354_35440


namespace NUMINAMATH_GPT_willy_episodes_per_day_l354_35438

def total_episodes (seasons : ℕ) (episodes_per_season : ℕ) : ℕ :=
  seasons * episodes_per_season

def episodes_per_day (total_episodes : ℕ) (days : ℕ) : ℕ :=
  total_episodes / days

theorem willy_episodes_per_day :
  episodes_per_day (total_episodes 3 20) 30 = 2 :=
by
  sorry

end NUMINAMATH_GPT_willy_episodes_per_day_l354_35438


namespace NUMINAMATH_GPT_number_represented_by_B_l354_35495

theorem number_represented_by_B (b : ℤ) : 
  (abs (b - 3) = 5) -> (b = 8 ∨ b = -2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_represented_by_B_l354_35495


namespace NUMINAMATH_GPT_circle_radius_l354_35447

-- Given conditions
def central_angle : ℝ := 225
def perimeter : ℝ := 83
noncomputable def pi_val : ℝ := Real.pi

-- Formula for the radius
noncomputable def radius : ℝ := 332 / (5 * pi_val + 8)

-- Prove that the radius is correct given the conditions
theorem circle_radius (theta : ℝ) (P : ℝ) (r : ℝ) (h_theta : theta = central_angle) (h_P : P = perimeter) :
  r = radius :=
sorry

end NUMINAMATH_GPT_circle_radius_l354_35447


namespace NUMINAMATH_GPT_total_sections_l354_35418

theorem total_sections (boys girls max_students per_section boys_ratio girls_ratio : ℕ)
  (hb : boys = 408) (hg : girls = 240) (hm : max_students = 24) 
  (br : boys_ratio = 3) (gr : girls_ratio = 2)
  (hboy_sec : (boys + max_students - 1) / max_students = 17)
  (hgirl_sec : (girls + max_students - 1) / max_students = 10) 
  : (3 * (((boys + max_students - 1) / max_students) + 2 * ((girls + max_students - 1) / max_students))) / 5 = 30 :=
by
  sorry

end NUMINAMATH_GPT_total_sections_l354_35418


namespace NUMINAMATH_GPT_translation_line_segment_l354_35476

theorem translation_line_segment (a b : ℝ) :
  (∃ A B A1 B1: ℝ × ℝ,
    A = (1,0) ∧ B = (3,2) ∧ A1 = (a, 1) ∧ B1 = (4,b) ∧
    ∃ t : ℝ × ℝ, A + t = A1 ∧ B + t = B1) →
  a = 2 ∧ b = 3 :=
by
  sorry

end NUMINAMATH_GPT_translation_line_segment_l354_35476


namespace NUMINAMATH_GPT_initial_savings_amount_l354_35423

theorem initial_savings_amount (A : ℝ) (P : ℝ) (r1 r2 t1 t2 : ℝ) (hA : A = 2247.50) (hr1 : r1 = 0.08) (hr2 : r2 = 0.04) (ht1 : t1 = 0.25) (ht2 : t2 = 0.25) :
  P = 2181 :=
by
  sorry

end NUMINAMATH_GPT_initial_savings_amount_l354_35423


namespace NUMINAMATH_GPT_five_student_committee_l354_35405

theorem five_student_committee : ∀ (students : Finset ℕ) (alice bob : ℕ), 
  alice ∈ students → bob ∈ students → students.card = 8 → ∃ (committees : Finset (Finset ℕ)),
  (∀ committee ∈ committees, alice ∈ committee ∧ bob ∈ committee) ∧
  ∀ committee ∈ committees, committee.card = 5 ∧ committees.card = 20 :=
by
  sorry

end NUMINAMATH_GPT_five_student_committee_l354_35405


namespace NUMINAMATH_GPT_A_share_value_l354_35424

-- Define the shares using the common multiplier x
variable (x : ℝ)

-- Define the shares in terms of x
def A_share := 5 * x
def B_share := 2 * x
def C_share := 4 * x
def D_share := 3 * x

-- Given condition that C gets Rs. 500 more than D
def condition := C_share - D_share = 500

-- State the theorem to determine A's share given the conditions
theorem A_share_value (h : condition) : A_share = 2500 := by 
  sorry

end NUMINAMATH_GPT_A_share_value_l354_35424


namespace NUMINAMATH_GPT_smallest_fraction_l354_35494

theorem smallest_fraction {a b c d e : ℚ}
  (ha : a = 7/15)
  (hb : b = 5/11)
  (hc : c = 16/33)
  (hd : d = 49/101)
  (he : e = 89/183) :
  (b < a) ∧ (b < c) ∧ (b < d) ∧ (b < e) := 
sorry

end NUMINAMATH_GPT_smallest_fraction_l354_35494


namespace NUMINAMATH_GPT_part1_part2_part3_l354_35454

-- Define the sequences a_n and b_n as described in the problem
def X_sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ (∀ n : ℕ, n > 0 → (a n = 0 ∨ a n = 1))

def accompanying_sequence (a b : ℕ → ℝ) : Prop :=
  (b 1 = 1) ∧ (∀ n : ℕ, n > 0 → b (n + 1) = abs (a n - (a (n + 1) / 2)) * b n)

-- 1. Prove the values of b_2, b_3, and b_4
theorem part1 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  a 2 = 1 → a 3 = 0 → a 4 = 1 →
  b 2 = 1 / 2 ∧ b 3 = 1 / 2 ∧ b 4 = 1 / 4 := 
sorry

-- 2. Prove the equivalence for geometric sequence and constant sequence
theorem part2 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  (∀ n : ℕ, n > 0 → a n = 1) ↔ (∃ r : ℝ, ∀ n : ℕ, n > 0 → b (n + 1) = r * b n) := 
sorry

-- 3. Prove the maximum value of b_2019
theorem part3 (a b : ℕ → ℝ) (h_a : X_sequence a) (h_b : accompanying_sequence a b) :
  b 2019 ≤ 1 / 2^1009 := 
sorry

end NUMINAMATH_GPT_part1_part2_part3_l354_35454


namespace NUMINAMATH_GPT_max_S_at_n_four_l354_35427

-- Define the sequence sum S_n
def S (n : ℕ) : ℤ := -(n^2 : ℤ) + (8 * n : ℤ)

-- Prove that S_n attains its maximum value at n = 4
theorem max_S_at_n_four : ∀ n : ℕ, S n ≤ S 4 :=
by
  sorry

end NUMINAMATH_GPT_max_S_at_n_four_l354_35427


namespace NUMINAMATH_GPT_product_of_solutions_of_quadratic_l354_35428

theorem product_of_solutions_of_quadratic :
  ∀ (x p q : ℝ), 36 - 9 * x - x^2 = 0 ∧ (x = p ∨ x = q) → p * q = -36 :=
by sorry

end NUMINAMATH_GPT_product_of_solutions_of_quadratic_l354_35428


namespace NUMINAMATH_GPT_seven_balls_expected_positions_l354_35499

theorem seven_balls_expected_positions :
  let n := 7
  let swaps := 4
  let p_stay := (1 - 2/7)^4 + 6 * (2/7)^2 * (5/7)^2 + (2/7)^4
  let expected_positions := n * p_stay
  expected_positions = 3.61 :=
by
  let n := 7
  let swaps := 4
  let p_stay := (1 - 2/7)^4 + 6 * (2/7)^2 * (5/7)^2 + (2/7)^4
  let expected_positions := n * p_stay
  exact sorry

end NUMINAMATH_GPT_seven_balls_expected_positions_l354_35499


namespace NUMINAMATH_GPT_arithmetic_sequence_a7_value_l354_35401

variable (a : ℕ → ℝ) (a1 a13 a7 : ℝ)

theorem arithmetic_sequence_a7_value
  (h1 : a 1 = a1)
  (h13 : a 13 = a13)
  (h_sum : a1 + a13 = 12)
  (h_arith : 2 * a7 = a1 + a13) :
  a7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a7_value_l354_35401


namespace NUMINAMATH_GPT_express_repeating_decimal_as_fraction_l354_35437

noncomputable def repeating_decimal_to_fraction : ℚ :=
  3 + 7 / 9  -- Representation of 3.\overline{7} as a Rational number representation

theorem express_repeating_decimal_as_fraction :
  (3 + 7 / 9 : ℚ) = 34 / 9 :=
by
  -- Placeholder for proof steps
  sorry

end NUMINAMATH_GPT_express_repeating_decimal_as_fraction_l354_35437


namespace NUMINAMATH_GPT_base10_equivalent_of_43210_7_l354_35442

def base7ToDecimal (num : Nat) : Nat :=
  let digits := [4, 3, 2, 1, 0]
  digits[0] * 7^4 + digits[1] * 7^3 + digits[2] * 7^2 + digits[3] * 7^1 + digits[4] * 7^0

theorem base10_equivalent_of_43210_7 :
  base7ToDecimal 43210 = 10738 :=
by
  sorry

end NUMINAMATH_GPT_base10_equivalent_of_43210_7_l354_35442


namespace NUMINAMATH_GPT_johns_initial_money_l354_35471

/-- John's initial money given that he gives 3/8 to his mother and 3/10 to his father,
and he has $65 left after giving away the money. Prove that he initially had $200. -/
theorem johns_initial_money 
  (M : ℕ)
  (h_left : (M : ℚ) - (3 / 8) * M - (3 / 10) * M = 65) :
  M = 200 :=
sorry

end NUMINAMATH_GPT_johns_initial_money_l354_35471


namespace NUMINAMATH_GPT_adoption_cost_l354_35443

theorem adoption_cost :
  let cost_cat := 50
  let cost_adult_dog := 100
  let cost_puppy := 150
  let num_cats := 2
  let num_adult_dogs := 3
  let num_puppies := 2
  (num_cats * cost_cat + num_adult_dogs * cost_adult_dog + num_puppies * cost_puppy) = 700 :=
by
  sorry

end NUMINAMATH_GPT_adoption_cost_l354_35443


namespace NUMINAMATH_GPT_arithmetic_sequence_solution_l354_35458

theorem arithmetic_sequence_solution (x : ℝ) (h : 2 * (x + 1) = 2 * x + (x + 2)) : x = 0 :=
by {
  -- To avoid actual proof steps, we add sorry.
  sorry 
}

end NUMINAMATH_GPT_arithmetic_sequence_solution_l354_35458


namespace NUMINAMATH_GPT_factorize_expr1_factorize_expr2_l354_35483

-- Problem (1) Statement
theorem factorize_expr1 (x y : ℝ) : 
  -x^5 * y^3 + x^3 * y^5 = -x^3 * y^3 * (x + y) * (x - y) :=
sorry

-- Problem (2) Statement
theorem factorize_expr2 (a : ℝ) : 
  (a^2 + 1)^2 - 4 * a^2 = (a + 1)^2 * (a - 1)^2 :=
sorry

end NUMINAMATH_GPT_factorize_expr1_factorize_expr2_l354_35483


namespace NUMINAMATH_GPT_find_a_l354_35485

theorem find_a (a : ℝ) (b : ℝ) :
  (9 * x^2 - 27 * x + a = (3 * x + b)^2) → b = -4.5 → a = 20.25 := 
by sorry

end NUMINAMATH_GPT_find_a_l354_35485


namespace NUMINAMATH_GPT_sum_cubes_of_roots_l354_35462

noncomputable def cube_root_sum_cubes (α β γ : ℝ) : ℝ :=
  α^3 + β^3 + γ^3
  
theorem sum_cubes_of_roots : 
  (cube_root_sum_cubes (Real.rpow 27 (1/3)) (Real.rpow 64 (1/3)) (Real.rpow 125 (1/3))) - 3 * ((Real.rpow 27 (1/3)) * (Real.rpow 64 (1/3)) * (Real.rpow 125 (1/3)) + 4/3) = 36 
  ∧
  ((Real.rpow 27 (1/3) + Real.rpow 64 (1/3) + Real.rpow 125 (1/3)) * ((Real.rpow 27 (1/3) + Real.rpow 64 (1/3) + Real.rpow 125 (1/3))^2 - 3 * ((Real.rpow 27 (1/3)) * (Real.rpow 64 (1/3)) + (Real.rpow 64 (1/3)) * (Real.rpow 125 (1/3)) + (Real.rpow 125 (1/3)) * (Real.rpow 27 (1/3)))) = 36) 
  → 
  cube_root_sum_cubes (Real.rpow 27 (1/3)) (Real.rpow 64 (1/3)) (Real.rpow 125 (1/3)) = 220 := 
sorry

end NUMINAMATH_GPT_sum_cubes_of_roots_l354_35462


namespace NUMINAMATH_GPT_value_of_expression_l354_35490

theorem value_of_expression (x y z : ℤ) (h1 : x = 2) (h2 : y = 3) (h3 : z = 4) :
  (4 * x^2 - 6 * y^3 + z^2) / (5 * x + 7 * z - 3 * y^2) = -130 / 11 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l354_35490


namespace NUMINAMATH_GPT_sum_of_x_and_y_l354_35435

theorem sum_of_x_and_y (x y : ℕ) (h_pos_x: 0 < x) (h_pos_y: 0 < y) (h_gt: x > y) (h_eq: x + x * y = 391) : x + y = 39 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l354_35435


namespace NUMINAMATH_GPT_cities_with_highest_increase_l354_35480

-- Define population changes for each city
def cityF_initial := 30000
def cityF_final := 45000
def cityG_initial := 55000
def cityG_final := 77000
def cityH_initial := 40000
def cityH_final := 60000
def cityI_initial := 70000
def cityI_final := 98000
def cityJ_initial := 25000
def cityJ_final := 37500

-- Function to calculate percentage increase
def percentage_increase (initial final : ℕ) : ℚ :=
  ((final - initial) : ℚ) / (initial : ℚ) * 100

-- Theorem stating cities F, H, and J had the highest percentage increase
theorem cities_with_highest_increase :
  percentage_increase cityF_initial cityF_final = 50 ∧
  percentage_increase cityH_initial cityH_final = 50 ∧
  percentage_increase cityJ_initial cityJ_final = 50 ∧
  percentage_increase cityG_initial cityG_final < 50 ∧
  percentage_increase cityI_initial cityI_final < 50 :=
by
-- Proof omitted
sorry

end NUMINAMATH_GPT_cities_with_highest_increase_l354_35480


namespace NUMINAMATH_GPT_correct_option_l354_35478

def condition_A : Prop := abs ((-5 : ℤ)^2) = -5
def condition_B : Prop := abs (9 : ℤ) = 3 ∨ abs (9 : ℤ) = -3
def condition_C : Prop := abs (3 : ℤ) / abs (((-2)^3 : ℤ)) = -2
def condition_D : Prop := (2 * abs (3 : ℤ))^2 = 6 

theorem correct_option : ¬condition_A ∧ ¬condition_B ∧ condition_C ∧ ¬condition_D :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l354_35478


namespace NUMINAMATH_GPT_max_value_of_quadratic_l354_35492

theorem max_value_of_quadratic:
  ∀ (x : ℝ), (∃ y : ℝ, y = -3 * x ^ 2 + 9) → (∃ max_y : ℝ, max_y = 9 ∧ ∀ x : ℝ, -3 * x ^ 2 + 9 ≤ max_y) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l354_35492


namespace NUMINAMATH_GPT_hyperbola_foci_coordinates_l354_35434

theorem hyperbola_foci_coordinates :
  let a : ℝ := Real.sqrt 7
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 + b^2)
  (c = Real.sqrt 10 ∧
  ∀ x y, (x^2 / 7 - y^2 / 3 = 1) → ((x, y) = (c, 0) ∨ (x, y) = (-c, 0))) :=
by
  let a := Real.sqrt 7
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 + b^2)
  have hc : c = Real.sqrt 10 := sorry
  have h_foci : ∀ x y, (x^2 / 7 - y^2 / 3 = 1) → ((x, y) = (c, 0) ∨ (x, y) = (-c, 0)) := sorry
  exact ⟨hc, h_foci⟩

end NUMINAMATH_GPT_hyperbola_foci_coordinates_l354_35434


namespace NUMINAMATH_GPT_real_y_values_for_given_x_l354_35406

theorem real_y_values_for_given_x (x : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 6 * x * y + 2 * x + 4 = 0) ↔ (x ≤ -2 / 3 ∨ x ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_real_y_values_for_given_x_l354_35406


namespace NUMINAMATH_GPT_perimeter_of_smaller_polygon_l354_35441

/-- The ratio of the areas of two similar polygons is 1:16, and the difference in their perimeters is 9.
Find the perimeter of the smaller polygon. -/
theorem perimeter_of_smaller_polygon (a b : ℝ) (h1 : a / b = 1 / 16) (h2 : b - a = 9) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_smaller_polygon_l354_35441


namespace NUMINAMATH_GPT_factorize_polynomial_l354_35498

theorem factorize_polynomial (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := 
by sorry

end NUMINAMATH_GPT_factorize_polynomial_l354_35498


namespace NUMINAMATH_GPT_height_of_cylinder_l354_35419

theorem height_of_cylinder (r_hemisphere : ℝ) (r_cylinder : ℝ) (h_cylinder : ℝ) :
  r_hemisphere = 7 → r_cylinder = 3 → h_cylinder = 2 * Real.sqrt 10 :=
by
  intro r_hemisphere_eq r_cylinder_eq
  sorry

end NUMINAMATH_GPT_height_of_cylinder_l354_35419


namespace NUMINAMATH_GPT_four_played_games_l354_35439

theorem four_played_games
  (A B C D E : Prop)
  (A_answer : ¬A)
  (B_answer : A ∧ ¬B)
  (C_answer : B ∧ ¬C)
  (D_answer : C ∧ ¬D)
  (E_answer : D ∧ ¬E)
  (truth_condition : (¬A ∧ ¬B) ∨ (¬B ∧ ¬C) ∨ (¬C ∧ ¬D) ∨ (¬D ∧ ¬E)) :
  A ∨ B ∨ C ∨ D ∧ E := sorry

end NUMINAMATH_GPT_four_played_games_l354_35439


namespace NUMINAMATH_GPT_product_of_base8_digits_of_5432_l354_35407

open Nat

def base8_digits (n : ℕ) : List ℕ :=
  let rec digits_helper (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc
    else digits_helper (n / 8) ((n % 8) :: acc)
  digits_helper n []

def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (· * ·) 1

theorem product_of_base8_digits_of_5432 : 
    product_of_digits (base8_digits 5432) = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_of_base8_digits_of_5432_l354_35407


namespace NUMINAMATH_GPT_find_side_DF_in_triangle_DEF_l354_35446

theorem find_side_DF_in_triangle_DEF
  (DE EF DM : ℝ)
  (h_DE : DE = 7)
  (h_EF : EF = 10)
  (h_DM : DM = 5) :
  ∃ DF : ℝ, DF = Real.sqrt 51 :=
by
  sorry

end NUMINAMATH_GPT_find_side_DF_in_triangle_DEF_l354_35446


namespace NUMINAMATH_GPT_find_numbers_in_progressions_l354_35484

theorem find_numbers_in_progressions (a b c d : ℝ) :
    (a + b + c = 114) ∧ -- Sum condition
    (b^2 = a * c) ∧ -- Geometric progression condition
    (b = a + 3 * d) ∧ -- Arithmetic progression first condition
    (c = a + 24 * d) -- Arithmetic progression second condition
    ↔ (a = 38 ∧ b = 38 ∧ c = 38) ∨ (a = 2 ∧ b = 14 ∧ c = 98) := by
  sorry

end NUMINAMATH_GPT_find_numbers_in_progressions_l354_35484


namespace NUMINAMATH_GPT_trajectory_of_M_l354_35455

noncomputable def P : ℝ × ℝ := (2, 2)
noncomputable def circleC (x y : ℝ) : Prop := x^2 + y^2 - 8 * y = 0
noncomputable def isMidpoint (A B M : ℝ × ℝ) : Prop := M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def isIntersectionPoint (l : ℝ × ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, circleC x y ∧ l (x, y) ∧ ((A = (x, y)) ∨ (B = (x, y))) 

theorem trajectory_of_M (M : ℝ × ℝ) : 
  (∃ A B : ℝ × ℝ, isIntersectionPoint (fun p => ∃ k : ℝ, p = (k, k)) A B ∧ isMidpoint A B M) →
  (M.1 - 1)^2 + (M.2 - 3)^2 = 2 := 
sorry

end NUMINAMATH_GPT_trajectory_of_M_l354_35455


namespace NUMINAMATH_GPT_math_problem_l354_35453

theorem math_problem (a b : ℕ) (x y : ℚ) (h1 : a = 10) (h2 : b = 11) (h3 : x = 1.11) (h4 : y = 1.01) :
  ∃ k : ℕ, k * y = 2.02 ∧ (a * x + b * y - k * y = 20.19) :=
by {
  sorry
}

end NUMINAMATH_GPT_math_problem_l354_35453


namespace NUMINAMATH_GPT_calculate_brick_quantity_l354_35429

noncomputable def brick_quantity (brick_length brick_width brick_height wall_length wall_height wall_width : ℝ) : ℝ :=
  let brick_volume := brick_length * brick_width * brick_height
  let wall_volume := wall_length * wall_height * wall_width
  wall_volume / brick_volume

theorem calculate_brick_quantity :
  brick_quantity 20 10 8 1000 800 2450 = 1225000 := 
by 
  -- Volume calculations are shown but proof is omitted
  sorry

end NUMINAMATH_GPT_calculate_brick_quantity_l354_35429


namespace NUMINAMATH_GPT_evaluate_expression_l354_35404

theorem evaluate_expression : 
  let a := 2
  let b := 1 / 2
  2 * (a^2 - 2 * a * b) - 3 * (a^2 - a * b - 4 * b^2) = -2 :=
by
  let a := 2
  let b := 1 / 2
  sorry

end NUMINAMATH_GPT_evaluate_expression_l354_35404


namespace NUMINAMATH_GPT_units_digit_of_quotient_l354_35459

theorem units_digit_of_quotient (n : ℕ) (h1 : n = 1987) : 
  (((4^n + 6^n) / 5) % 10) = 0 :=
by
  have pattern_4 : ∀ (k : ℕ), (4^k) % 10 = if k % 2 = 0 then 6 else 4 := sorry
  have pattern_6 : ∀ (k : ℕ), (6^k) % 10 = 6 := sorry
  have units_sum : (4^1987 % 10 + 6^1987 % 10) % 10 = 0 := sorry
  have multiple_of_5 : (4^1987 + 6^1987) % 5 = 0 := sorry
  sorry

end NUMINAMATH_GPT_units_digit_of_quotient_l354_35459


namespace NUMINAMATH_GPT_solution_set_of_inequality_l354_35463

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h₁ : ∀ x > 0, deriv f x + 2 * f x > 0) :
  {x : ℝ | x + 2018 > 0 ∧ x + 2018 < 5} = {x : ℝ | -2018 < x ∧ x < -2013} := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l354_35463


namespace NUMINAMATH_GPT_prop_p_iff_prop_q_iff_not_or_p_q_l354_35487

theorem prop_p_iff (m : ℝ) :
  (∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + (2 + m) = 0) ↔ (m ≤ -1 ∨ m ≥ 2) :=
sorry

theorem prop_q_iff (m : ℝ) :
  (∃ x y : ℝ, (x^2)/(1 - 2*m) + (y^2)/(m + 2) = 1) ↔ (m < -2 ∨ m > 1/2) :=
sorry

theorem not_or_p_q (m : ℝ) :
  ¬(∃ x₀ : ℝ, x₀^2 + 2 * m * x₀ + (2 + m) = 0) ∧
  ¬(∃ x y : ℝ, (x^2)/(1 - 2*m) + (y^2)/(m + 2) = 1) ↔
  (-1 < m ∧ m ≤ 1/2) :=
sorry

end NUMINAMATH_GPT_prop_p_iff_prop_q_iff_not_or_p_q_l354_35487


namespace NUMINAMATH_GPT_solve_triangle_l354_35477

theorem solve_triangle :
  (a = 6 ∧ b = 6 * Real.sqrt 3 ∧ A = 30) →
  ((B = 60 ∧ C = 90 ∧ c = 12) ∨ (B = 120 ∧ C = 30 ∧ c = 6)) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solve_triangle_l354_35477


namespace NUMINAMATH_GPT_investment_inequality_l354_35402

-- Defining the initial investment
def initial_investment : ℝ := 200

-- Year 1 changes
def alpha_year1 := initial_investment * 1.30
def beta_year1 := initial_investment * 0.80
def gamma_year1 := initial_investment * 1.10
def delta_year1 := initial_investment * 0.90

-- Year 2 changes
def alpha_final := alpha_year1 * 0.85
def beta_final := beta_year1 * 1.30
def gamma_final := gamma_year1 * 0.95
def delta_final := delta_year1 * 1.20

-- Prove the final inequality
theorem investment_inequality : beta_final < gamma_final ∧ gamma_final < delta_final ∧ delta_final < alpha_final :=
by {
  sorry
}

end NUMINAMATH_GPT_investment_inequality_l354_35402


namespace NUMINAMATH_GPT_perfect_square_trinomial_m_l354_35464

theorem perfect_square_trinomial_m (m : ℤ) : (∃ (a : ℤ), (x : ℝ) → x^2 + m * x + 9 = (x + a)^2) → (m = 6 ∨ m = -6) :=
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_m_l354_35464


namespace NUMINAMATH_GPT_rain_forest_animals_l354_35414

theorem rain_forest_animals (R : ℕ) 
  (h1 : 16 = 3 * R - 5) : R = 7 := 
  by sorry

end NUMINAMATH_GPT_rain_forest_animals_l354_35414


namespace NUMINAMATH_GPT_sum_of_234_and_142_in_base_4_l354_35432

theorem sum_of_234_and_142_in_base_4 :
  (234 + 142) = 376 ∧ (376 + 0) = 256 * 1 + 64 * 1 + 16 * 3 + 4 * 2 + 1 * 0 :=
by sorry

end NUMINAMATH_GPT_sum_of_234_and_142_in_base_4_l354_35432


namespace NUMINAMATH_GPT_sum_of_distinct_prime_factors_315_l354_35475

theorem sum_of_distinct_prime_factors_315 : 
  ∃ factors : List ℕ, factors = [3, 5, 7] ∧ 315 = 3 * 3 * 5 * 7 ∧ factors.sum = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_distinct_prime_factors_315_l354_35475


namespace NUMINAMATH_GPT_average_speed_palindrome_trip_l354_35451

theorem average_speed_palindrome_trip :
  ∀ (initial final : ℕ) (time : ℝ),
    initial = 13431 → final = 13531 → time = 3 →
    (final - initial) / time = 33 :=
by
  intros initial final time h_initial h_final h_time
  rw [h_initial, h_final, h_time]
  norm_num
  sorry

end NUMINAMATH_GPT_average_speed_palindrome_trip_l354_35451


namespace NUMINAMATH_GPT_puppies_sold_l354_35448

theorem puppies_sold (total_puppies sold_puppies puppies_per_cage total_cages : ℕ)
  (h1 : total_puppies = 102)
  (h2 : puppies_per_cage = 9)
  (h3 : total_cages = 9)
  (h4 : total_puppies - sold_puppies = puppies_per_cage * total_cages) :
  sold_puppies = 21 :=
by {
  -- Proof details would go here
  sorry
}

end NUMINAMATH_GPT_puppies_sold_l354_35448


namespace NUMINAMATH_GPT_league_games_and_weeks_l354_35491

/--
There are 15 teams in a league, and each team plays each of the other teams exactly once.
Due to scheduling limitations, each team can only play one game per week.
Prove that the total number of games played is 105 and the minimum number of weeks needed to complete all the games is 15.
-/
theorem league_games_and_weeks :
  let teams := 15
  let total_games := teams * (teams - 1) / 2
  let games_per_week := Nat.div teams 2
  total_games = 105 ∧ total_games / games_per_week = 15 :=
by
  sorry

end NUMINAMATH_GPT_league_games_and_weeks_l354_35491


namespace NUMINAMATH_GPT_probability_third_white_no_replacement_probability_red_no_more_than_4_in_6_draws_with_replacement_l354_35421

noncomputable section

-- Problem 1: Probability of drawing a white ball on the third draw without replacement is 1/3.
theorem probability_third_white_no_replacement :
  let red := 4
  let white := 2
  let totalBalls := red + white
  let totalWaysToDraw3 := Nat.choose totalBalls 3
  let favorableWays := Nat.choose (totalBalls - 1) 2 * Nat.choose white 1
  let probability := favorableWays / totalWaysToDraw3
  probability = 1 / 3 :=
by
  sorry

-- Problem 2: Probability of drawing red balls no more than 4 times in 6 draws with replacement is 441/729.
theorem probability_red_no_more_than_4_in_6_draws_with_replacement :
  let red := 4
  let white := 2
  let totalBalls := red + white
  let p_red := red / totalBalls
  let p_X5 := Nat.choose 6 5 * p_red^5 * (1 - p_red)
  let p_X6 := Nat.choose 6 6 * p_red^6
  let probability := 1 - p_X5 - p_X6
  probability = 441 / 729 :=
by
  sorry

end NUMINAMATH_GPT_probability_third_white_no_replacement_probability_red_no_more_than_4_in_6_draws_with_replacement_l354_35421
