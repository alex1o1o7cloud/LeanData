import Mathlib

namespace NUMINAMATH_GPT_geometric_prog_common_ratio_one_l952_95230

variable {x y z : ℝ}
variable {r : ℝ}

theorem geometric_prog_common_ratio_one
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x)
  (hgeom : ∃ a : ℝ, a = x * (y - z) ∧ a * r = y * (z - x) ∧ a * r^2 = z * (x - y))
  (hprod : (x * (y - z)) * (y * (z - x)) * (z * (x - y)) * r^3 = (y * (z - x))^2) : 
  r = 1 := sorry

end NUMINAMATH_GPT_geometric_prog_common_ratio_one_l952_95230


namespace NUMINAMATH_GPT_parallel_vectors_implies_m_eq_neg1_l952_95297

theorem parallel_vectors_implies_m_eq_neg1 (m : ℝ) :
  let a := (m, -1)
  let b := (1, m + 2)
  a.1 * b.2 = a.2 * b.1 → m = -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parallel_vectors_implies_m_eq_neg1_l952_95297


namespace NUMINAMATH_GPT_blue_face_probability_l952_95270

def sides : ℕ := 12
def green_faces : ℕ := 5
def blue_faces : ℕ := 4
def red_faces : ℕ := 3

theorem blue_face_probability : 
  (blue_faces : ℚ) / sides = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_blue_face_probability_l952_95270


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l952_95266

def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 10*x^3 + 20*x^2 - 5*x - 21

theorem remainder_when_divided_by_x_minus_2 :
  f 2 = 33 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l952_95266


namespace NUMINAMATH_GPT_total_white_balls_l952_95229

theorem total_white_balls : ∃ W R B : ℕ,
  W + R = 300 ∧ B = 100 ∧
  ∃ (bw1 bw2 rw3 rw W3 : ℕ),
  bw1 = 27 ∧
  rw3 + rw = 42 ∧
  W3 = rw ∧
  B = bw1 + W3 + rw3 + bw2 ∧
  W = bw1 + 2 * bw2 + 3 * W3 ∧
  R = 3 * rw3 + rw ∧
  W = 158 :=
by
  sorry

end NUMINAMATH_GPT_total_white_balls_l952_95229


namespace NUMINAMATH_GPT_value_of_nested_expression_l952_95207

def nested_expression : ℕ :=
  3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2

theorem value_of_nested_expression : nested_expression = 1457 := by
  sorry

end NUMINAMATH_GPT_value_of_nested_expression_l952_95207


namespace NUMINAMATH_GPT_find_value_of_expression_l952_95274

theorem find_value_of_expression (x : ℝ) (h : x^2 + (1 / x^2) = 5) : x^4 + (1 / x^4) = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l952_95274


namespace NUMINAMATH_GPT_meaningful_expression_l952_95265

theorem meaningful_expression (x : ℝ) : (1 / (x - 2) ≠ 0) ↔ (x ≠ 2) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_expression_l952_95265


namespace NUMINAMATH_GPT_games_needed_to_declare_winner_l952_95277

def single_elimination_games (T : ℕ) : ℕ :=
  T - 1

theorem games_needed_to_declare_winner (T : ℕ) :
  (single_elimination_games 23 = 22) :=
by
  sorry

end NUMINAMATH_GPT_games_needed_to_declare_winner_l952_95277


namespace NUMINAMATH_GPT_symmetricPointCorrectCount_l952_95210

-- Define a structure for a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the four symmetry conditions
def isSymmetricXaxis (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := P.z }
def isSymmetricYOZplane (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := -P.z }
def isSymmetricYaxis (P Q : Point3D) : Prop := Q = { x := P.x, y := -P.y, z := P.z }
def isSymmetricOrigin (P Q : Point3D) : Prop := Q = { x := -P.x, y := -P.y, z := -P.z }

-- Define a theorem to count the valid symmetric conditions
theorem symmetricPointCorrectCount (P : Point3D) :
  (isSymmetricXaxis P { x := P.x, y := -P.y, z := P.z } = true → false) ∧
  (isSymmetricYOZplane P { x := P.x, y := -P.y, z := -P.z } = true → false) ∧
  (isSymmetricYaxis P { x := P.x, y := -P.y, z := P.z } = true → false) ∧
  (isSymmetricOrigin P { x := -P.x, y := -P.y, z := -P.z } = true → true) :=
by
  sorry

end NUMINAMATH_GPT_symmetricPointCorrectCount_l952_95210


namespace NUMINAMATH_GPT_women_at_each_table_l952_95298

theorem women_at_each_table (W : ℕ) (h1 : ∃ W, ∀ i : ℕ, (i < 7) → W + 2 = 7 * W + 14) (h2 : 7 * W + 14 = 63) : W = 7 :=
by
  sorry

end NUMINAMATH_GPT_women_at_each_table_l952_95298


namespace NUMINAMATH_GPT_solve_students_in_fifth_grade_class_l952_95242

noncomputable def number_of_students_in_each_fifth_grade_class 
    (third_grade_classes : ℕ) 
    (third_grade_students_per_class : ℕ)
    (fourth_grade_classes : ℕ) 
    (fourth_grade_students_per_class : ℕ) 
    (fifth_grade_classes : ℕ)
    (total_lunch_cost : ℝ)
    (hamburger_cost : ℝ)
    (carrot_cost : ℝ)
    (cookie_cost : ℝ) : ℝ :=
  
  let total_students_third := third_grade_classes * third_grade_students_per_class
  let total_students_fourth := fourth_grade_classes * fourth_grade_students_per_class
  let lunch_cost_per_student := hamburger_cost + carrot_cost + cookie_cost
  let total_students := total_students_third + total_students_fourth
  let total_cost_third_fourth := total_students * lunch_cost_per_student
  let total_cost_fifth := total_lunch_cost - total_cost_third_fourth
  let fifth_grade_students := total_cost_fifth / lunch_cost_per_student
  let students_per_fifth_class := fifth_grade_students / fifth_grade_classes
  students_per_fifth_class

theorem solve_students_in_fifth_grade_class : 
    number_of_students_in_each_fifth_grade_class 5 30 4 28 4 1036 2.10 0.50 0.20 = 27 := 
by 
  sorry

end NUMINAMATH_GPT_solve_students_in_fifth_grade_class_l952_95242


namespace NUMINAMATH_GPT_incorrect_statement_trajectory_of_P_l952_95204

noncomputable def midpoint_of_points (x1 x2 y1 y2 : ℝ) : ℝ × ℝ :=
((x1 + x2) / 2, (y1 + y2) / 2)

theorem incorrect_statement_trajectory_of_P (p k x0 y0 : ℝ) (hp : p > 0)
    (A B : ℝ × ℝ)
    (hA : A.1 * A.1 + 2 * p * A.2 = 0)
    (hB : B.1 * B.1 + 2 * p * B.2 = 0)
    (hMid : (x0, y0) = midpoint_of_points A.1 B.1 A.2 B.2)
    (hLine : A.2 = k * (A.1 - p / 2))
    (hLineIntersection : B.2 = k * (B.1 - p / 2)) : y0 ^ 2 ≠ 4 * p * (x0 - p / 2) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_trajectory_of_P_l952_95204


namespace NUMINAMATH_GPT_necessary_condition_for_A_l952_95289

variable {x a : ℝ}

def A : Set ℝ := { x | (x - 2) / (x + 1) ≤ 0 }

theorem necessary_condition_for_A (x : ℝ) (h : x ∈ A) (ha : x ≥ a) : a ≤ -1 :=
sorry

end NUMINAMATH_GPT_necessary_condition_for_A_l952_95289


namespace NUMINAMATH_GPT_maximum_volume_l952_95245

noncomputable def volume (x : ℝ) : ℝ :=
  (48 - 2*x)^2 * x

theorem maximum_volume :
  (∀ x : ℝ, (0 < x) ∧ (x < 24) → volume x ≤ volume 8) ∧ (volume 8 = 8192) :=
by
  sorry

end NUMINAMATH_GPT_maximum_volume_l952_95245


namespace NUMINAMATH_GPT_range_of_f_l952_95252

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x ^ 2 else Real.cos x

theorem range_of_f : Set.range f = Set.Ici (-1) := 
by
  sorry

end NUMINAMATH_GPT_range_of_f_l952_95252


namespace NUMINAMATH_GPT_maximum_x_value_l952_95293

theorem maximum_x_value (x y z : ℝ) (h1 : x + y + z = 10) (h2 : x * y + x * z + y * z = 20) : 
  x ≤ 10 / 3 := sorry

end NUMINAMATH_GPT_maximum_x_value_l952_95293


namespace NUMINAMATH_GPT_mens_wages_l952_95250

-- Definitions from the conditions.
variables (men women boys total_earnings : ℕ) (wage : ℚ)
variable (equivalence : 5 * men = 8 * boys)
variable (totalEarnings : total_earnings = 120)

-- The final statement to prove the men's wages.
theorem mens_wages (h_eq : 5 = 5) : wage = 46.15 :=
by
  sorry

end NUMINAMATH_GPT_mens_wages_l952_95250


namespace NUMINAMATH_GPT_initial_pages_l952_95275

/-
Given:
1. Sammy uses 25% of the pages for his science project.
2. Sammy uses another 10 pages for his math homework.
3. There are 80 pages remaining in the pad.

Prove that the initial number of pages in the pad (P) is 120.
-/

theorem initial_pages (P : ℝ) (h1 : P * 0.25 + 10 + 80 = P) : 
  P = 120 :=
by 
  sorry

end NUMINAMATH_GPT_initial_pages_l952_95275


namespace NUMINAMATH_GPT_solution_set_of_inequality_l952_95286

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 3 * x - 2 > 0 ↔ 1 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l952_95286


namespace NUMINAMATH_GPT_tan_alpha_tan_beta_l952_95281

/-- Given the cosine values of the sum and difference of two angles, 
    find the value of the product of their tangents. -/
theorem tan_alpha_tan_beta (α β : ℝ) 
  (h1 : Real.cos (α + β) = 1/3) 
  (h2 : Real.cos (α - β) = 1/5) : 
  Real.tan α * Real.tan β = -1/4 := sorry

end NUMINAMATH_GPT_tan_alpha_tan_beta_l952_95281


namespace NUMINAMATH_GPT_yellow_candles_count_l952_95200

def CalebCandles (grandfather_age : ℕ) (red_candles : ℕ) (blue_candles : ℕ) : ℕ :=
    grandfather_age - (red_candles + blue_candles)

theorem yellow_candles_count :
    CalebCandles 79 14 38 = 27 := by
    sorry

end NUMINAMATH_GPT_yellow_candles_count_l952_95200


namespace NUMINAMATH_GPT_simplify_expression_l952_95208

-- Define the variables and the polynomials
variables (y : ℤ)

-- Define the expressions
def expr1 := (2 * y - 1) * (5 * y^12 - 3 * y^11 + y^9 - 4 * y^8)
def expr2 := 10 * y^13 - 11 * y^12 + 3 * y^11 + y^10 - 9 * y^9 + 4 * y^8

-- State the theorem
theorem simplify_expression : expr1 = expr2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l952_95208


namespace NUMINAMATH_GPT_probability_at_least_one_five_or_six_l952_95278

theorem probability_at_least_one_five_or_six
  (P_neither_five_nor_six: ℚ)
  (h: P_neither_five_nor_six = 4 / 9) :
  (1 - P_neither_five_nor_six) = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_five_or_six_l952_95278


namespace NUMINAMATH_GPT_value_of_expression_l952_95234

theorem value_of_expression (a : ℝ) (h : a^2 + a = 0) : 4*a^2 + 4*a + 2011 = 2011 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l952_95234


namespace NUMINAMATH_GPT_double_burger_cost_l952_95217

theorem double_burger_cost (D : ℝ) : 
  let single_burger_cost := 1.00
  let total_burgers := 50
  let double_burgers := 37
  let total_cost := 68.50
  let single_burgers := total_burgers - double_burgers
  let singles_cost := single_burgers * single_burger_cost
  let doubles_cost := total_cost - singles_cost
  let burger_cost := doubles_cost / double_burgers
  burger_cost = D := 
by 
  sorry

end NUMINAMATH_GPT_double_burger_cost_l952_95217


namespace NUMINAMATH_GPT_sin_three_pi_div_two_l952_95224

theorem sin_three_pi_div_two : Real.sin (3 * Real.pi / 2) = -1 := 
by
  sorry

end NUMINAMATH_GPT_sin_three_pi_div_two_l952_95224


namespace NUMINAMATH_GPT_average_seven_numbers_l952_95285

theorem average_seven_numbers (A B C D E F G : ℝ) 
  (h1 : (A + B + C + D) / 4 = 4)
  (h2 : (D + E + F + G) / 4 = 4)
  (hD : D = 11) : 
  (A + B + C + D + E + F + G) / 7 = 3 :=
by
  sorry

end NUMINAMATH_GPT_average_seven_numbers_l952_95285


namespace NUMINAMATH_GPT_min_sum_squares_roots_l952_95246

theorem min_sum_squares_roots (m : ℝ) :
  (∃ (α β : ℝ), 2 * α^2 - 3 * α + m = 0 ∧ 2 * β^2 - 3 * β + m = 0 ∧ α ≠ β) → 
  (9 - 8 * m ≥ 0) →
  (α^2 + β^2 = (3/2)^2 - 2 * (m/2)) →
  (α^2 + β^2 = 9/8) ↔ m = 9/8 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_squares_roots_l952_95246


namespace NUMINAMATH_GPT_toluene_production_l952_95292

def molar_mass_benzene : ℝ := 78.11 -- The molar mass of benzene in g/mol
def benzene_mass : ℝ := 156 -- The mass of benzene in grams
def methane_moles : ℝ := 2 -- The moles of methane

-- Define the balanced chemical reaction
def balanced_reaction (benzene methanol toluene hydrogen : ℝ) : Prop :=
  benzene + methanol = toluene + hydrogen

-- The main theorem statement
theorem toluene_production (h1 : balanced_reaction benzene_mass methane_moles 1 1)
  (h2 : benzene_mass / molar_mass_benzene = 2) :
  ∃ toluene_moles : ℝ, toluene_moles = 2 :=
by
  sorry

end NUMINAMATH_GPT_toluene_production_l952_95292


namespace NUMINAMATH_GPT_no_infinite_sequence_of_positive_integers_l952_95235

theorem no_infinite_sequence_of_positive_integers (a : ℕ → ℕ) (H : ∀ n, a n > 0) :
  ¬(∀ n, (a (n+1))^2 ≥ 2 * (a n) * (a (n+2))) :=
sorry

end NUMINAMATH_GPT_no_infinite_sequence_of_positive_integers_l952_95235


namespace NUMINAMATH_GPT_set_aside_bars_each_day_l952_95262

-- Definitions for the conditions
def total_bars : Int := 20
def bars_traded : Int := 3
def bars_per_sister : Int := 5
def number_of_sisters : Int := 2
def days_in_week : Int := 7

-- Our goal is to prove that Greg set aside 1 bar per day
theorem set_aside_bars_each_day
  (h1 : 20 - 3 = 17)
  (h2 : 5 * 2 = 10)
  (h3 : 17 - 10 = 7)
  (h4 : 7 / 7 = 1) :
  (total_bars - bars_traded - (bars_per_sister * number_of_sisters)) / days_in_week = 1 := by
  sorry

end NUMINAMATH_GPT_set_aside_bars_each_day_l952_95262


namespace NUMINAMATH_GPT_find_x_squared_l952_95233

theorem find_x_squared :
  ∃ x : ℕ, (x^2 >= 2525 * 10^8) ∧ (x^2 < 2526 * 10^8) ∧ (x % 100 = 17 ∨ x % 100 = 33 ∨ x % 100 = 67 ∨ x % 100 = 83) ∧
    (x = 502517 ∨ x = 502533 ∨ x = 502567 ∨ x = 502583) :=
sorry

end NUMINAMATH_GPT_find_x_squared_l952_95233


namespace NUMINAMATH_GPT_smallest_b_for_perfect_square_l952_95215

theorem smallest_b_for_perfect_square : ∃ (b : ℤ), b > 4 ∧ ∃ (n : ℤ), 3 * b + 4 = n * n ∧ b = 7 := by
  sorry

end NUMINAMATH_GPT_smallest_b_for_perfect_square_l952_95215


namespace NUMINAMATH_GPT_cross_product_correct_l952_95243

def v : ℝ × ℝ × ℝ := (-3, 4, 5)
def w : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(a.2.1 * b.2.2 - a.2.2 * b.2.1,
 a.2.2 * b.1 - a.1 * b.2.2,
 a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_correct : cross_product v w = (21, 22, -5) :=
by
  sorry

end NUMINAMATH_GPT_cross_product_correct_l952_95243


namespace NUMINAMATH_GPT_tank_fill_time_l952_95288

-- Define the conditions
def start_time : ℕ := 1 -- 1 pm
def first_hour_rainfall : ℕ := 2 -- 2 inches rainfall in the first hour from 1 pm to 2 pm
def next_four_hours_rate : ℕ := 1 -- 1 inch/hour rainfall rate from 2 pm to 6 pm
def following_rate : ℕ := 3 -- 3 inches/hour rainfall rate from 6 pm onwards
def tank_height : ℕ := 18 -- 18 inches tall fish tank

-- Define what needs to be proved
theorem tank_fill_time : 
  ∃ t : ℕ, t = 22 ∧ (tank_height ≤ (first_hour_rainfall + 4 * next_four_hours_rate + (t - 6)) + (t - 6 - 4) * following_rate) := 
by 
  sorry

end NUMINAMATH_GPT_tank_fill_time_l952_95288


namespace NUMINAMATH_GPT_all_statements_true_l952_95225

theorem all_statements_true (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) :
  (a^2 + b^2 < (a + b)^2) ∧ 
  (ab > 0) ∧ 
  (a > b) ∧ 
  (a > 0) ∧
  (b > 0) :=
by
  sorry

end NUMINAMATH_GPT_all_statements_true_l952_95225


namespace NUMINAMATH_GPT_a_value_l952_95222

-- Definition of the operation
def star (x y : ℝ) : ℝ := x + y - x * y

-- Main theorem to prove
theorem a_value :
  let a := star 1 (star 0 1)
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_a_value_l952_95222


namespace NUMINAMATH_GPT_cubic_roots_real_parts_neg_l952_95241

variable {a0 a1 a2 a3 : ℝ}

theorem cubic_roots_real_parts_neg (h_same_signs : (a0 > 0 ∧ a1 > 0 ∧ a2 > 0 ∧ a3 > 0) ∨ (a0 < 0 ∧ a1 < 0 ∧ a2 < 0 ∧ a3 < 0)) 
  (h_root_condition : a1 * a2 - a0 * a3 > 0) : 
    ∀ (x : ℝ), (a0 * x^3 + a1 * x^2 + a2 * x + a3 = 0 → x < 0 ∨ (∃ (z : ℂ), z.re < 0 ∧ z.im ≠ 0 ∧ z^2 = x)) :=
sorry

end NUMINAMATH_GPT_cubic_roots_real_parts_neg_l952_95241


namespace NUMINAMATH_GPT_bigger_wheel_roll_distance_l952_95244

/-- The circumference of the bigger wheel is 12 meters -/
def bigger_wheel_circumference : ℕ := 12

/-- The circumference of the smaller wheel is 8 meters -/
def smaller_wheel_circumference : ℕ := 8

/-- The distance the bigger wheel must roll for the points P1 and P2 to coincide again -/
theorem bigger_wheel_roll_distance : Nat.lcm bigger_wheel_circumference smaller_wheel_circumference = 24 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_bigger_wheel_roll_distance_l952_95244


namespace NUMINAMATH_GPT_find_g_inv_f_3_l952_95205

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

axiom f_inv_g_eq : ∀ x : ℝ, f_inv (g x) = x^4 - x + 2
axiom g_has_inverse : ∀ y : ℝ, g (g_inv y) = y 

theorem find_g_inv_f_3 :
  ∃ α : ℝ, (α^4 - α - 1 = 0) ∧ g_inv (f 3) = α :=
sorry

end NUMINAMATH_GPT_find_g_inv_f_3_l952_95205


namespace NUMINAMATH_GPT_problem_l952_95214

variable (x y z : ℚ)

-- Conditions as definitions
def cond1 : Prop := x / y = 3
def cond2 : Prop := y / z = 5 / 2

-- Theorem statement with the final proof goal
theorem problem (h1 : cond1 x y) (h2 : cond2 y z) : z / x = 2 / 15 := 
by 
  sorry

end NUMINAMATH_GPT_problem_l952_95214


namespace NUMINAMATH_GPT_journey_total_time_l952_95291

def journey_time (d1 d2 : ℕ) (total_distance : ℕ) (car_speed walk_speed : ℕ) : ℕ :=
  d1 / car_speed + (total_distance - d1) / walk_speed

theorem journey_total_time :
  let total_distance := 150
  let car_speed := 30
  let walk_speed := 3
  let d1 := 50
  let d2 := 15
  
  journey_time d1 d2 total_distance car_speed walk_speed =
  max (journey_time d1 0 total_distance car_speed walk_speed / car_speed + 
       (total_distance - d1) / walk_speed)
      ((d1 / car_speed + (d1 - d2) / car_speed + (total_distance - d1 + d2) / car_speed)) :=
by
  sorry

end NUMINAMATH_GPT_journey_total_time_l952_95291


namespace NUMINAMATH_GPT_odd_function_iff_l952_95209

def f (x a b : ℝ) : ℝ := x * abs (x + a) + b

theorem odd_function_iff (a b : ℝ) : 
  (∀ x, f x a b = -f (-x) a b) ↔ (a ^ 2 + b ^ 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_odd_function_iff_l952_95209


namespace NUMINAMATH_GPT_vasya_wins_game_l952_95213

/- Define the conditions of the problem -/

def grid_size : Nat := 9
def total_matchsticks : Nat := 2 * grid_size * (grid_size + 1)

/-- Given a game on a 9x9 matchstick grid with Petya going first, 
    Prove that Vasya can always win by ensuring that no whole 1x1 
    squares remain in the end. -/
theorem vasya_wins_game : 
  ∃ strategy_for_vasya : Nat → Nat → Prop, -- Define a strategy for Vasya
  ∀ (matchsticks_left : Nat),
  matchsticks_left % 2 = 1 →     -- Petya makes a move and the remaining matchsticks are odd
  strategy_for_vasya matchsticks_left total_matchsticks :=
sorry

end NUMINAMATH_GPT_vasya_wins_game_l952_95213


namespace NUMINAMATH_GPT_domain_of_f_symmetry_of_f_l952_95221

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (4 * x^2 - x^4)) / (abs (x - 2) - 2)

theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | (-2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

theorem symmetry_of_f :
  ∀ x : ℝ, f (x + 1) + 1 = f (-(x + 1)) + 1 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_symmetry_of_f_l952_95221


namespace NUMINAMATH_GPT_evaluate_expression_l952_95216

theorem evaluate_expression : 1273 + 120 / 60 - 173 = 1102 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l952_95216


namespace NUMINAMATH_GPT_find_tangent_equal_l952_95253

theorem find_tangent_equal (n : ℤ) (hn : -90 < n ∧ n < 90) (htan : Real.tan (n * Real.pi / 180) = Real.tan (75 * Real.pi / 180)) : n = 75 :=
sorry

end NUMINAMATH_GPT_find_tangent_equal_l952_95253


namespace NUMINAMATH_GPT_value_of_a_minus_b_l952_95264

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the invertible function f

theorem value_of_a_minus_b (a b : ℝ) (hf_inv : Function.Injective f)
  (hfa : f a = b) (hfb : f b = 6) (ha1 : f 3 = 1) (hb1 : f 1 = 6) : a - b = 2 :=
sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l952_95264


namespace NUMINAMATH_GPT_function_neither_even_nor_odd_l952_95269

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := x^2 - x

theorem function_neither_even_nor_odd : ¬is_even_function f ∧ ¬is_odd_function f := by
  sorry

end NUMINAMATH_GPT_function_neither_even_nor_odd_l952_95269


namespace NUMINAMATH_GPT_gcd_5280_12155_l952_95223

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 5 :=
by
  sorry

end NUMINAMATH_GPT_gcd_5280_12155_l952_95223


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l952_95203

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = -6) : 
  (1 - a / (a - 3)) / ((a^2 + 3 * a) / (a^2 - 9)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l952_95203


namespace NUMINAMATH_GPT_lois_final_books_l952_95232

-- Definitions for the conditions given in the problem.
def initial_books : ℕ := 40
def books_given_to_nephew (b : ℕ) : ℕ := b / 4
def books_remaining_after_giving (b_given : ℕ) (b : ℕ) : ℕ := b - b_given
def books_donated_to_library (b_remaining : ℕ) : ℕ := b_remaining / 3
def books_remaining_after_donating (b_donated : ℕ) (b_remaining : ℕ) : ℕ := b_remaining - b_donated
def books_purchased : ℕ := 3
def total_books (b_final_remaining : ℕ) (b_purchased : ℕ) : ℕ := b_final_remaining + b_purchased

-- Theorem stating: Given the initial conditions, Lois should have 23 books in the end.
theorem lois_final_books : 
  total_books 
    (books_remaining_after_donating (books_donated_to_library (books_remaining_after_giving (books_given_to_nephew initial_books) initial_books)) 
    (books_remaining_after_giving (books_given_to_nephew initial_books) initial_books))
    books_purchased = 23 :=
  by
    sorry  -- Proof omitted as per instructions.

end NUMINAMATH_GPT_lois_final_books_l952_95232


namespace NUMINAMATH_GPT_Linda_outfits_l952_95290

theorem Linda_outfits (skirts blouses shoes : ℕ) 
  (hskirts : skirts = 5) 
  (hblouses : blouses = 8) 
  (hshoes : shoes = 2) :
  skirts * blouses * shoes = 80 := by
  -- We provide the proof here
  sorry

end NUMINAMATH_GPT_Linda_outfits_l952_95290


namespace NUMINAMATH_GPT_arithmetic_geom_seq_l952_95254

variable {a_n : ℕ → ℝ}
variable {d a_1 : ℝ}
variable (h_seq : ∀ n, a_n n = a_1 + (n-1) * d)
variable (d_ne_zero : d ≠ 0)
variable (a_1_ne_zero : a_1 ≠ 0)
variable (geo_seq : (a_1 + d)^2 = a_1 * (a_1 + 3 * d))

theorem arithmetic_geom_seq :
  (a_1 + a_n 14) / a_n 3 = 5 := by
  sorry

end NUMINAMATH_GPT_arithmetic_geom_seq_l952_95254


namespace NUMINAMATH_GPT_rational_root_uniqueness_l952_95276

theorem rational_root_uniqueness (c : ℚ) :
  ∀ x1 x2 : ℚ, (x1 ≠ x2) →
  (x1^3 - 3 * c * x1^2 - 3 * x1 + c = 0) →
  (x2^3 - 3 * c * x2^2 - 3 * x2 + c = 0) →
  false := 
by
  intros x1 x2 h1 h2 h3
  sorry

end NUMINAMATH_GPT_rational_root_uniqueness_l952_95276


namespace NUMINAMATH_GPT_mabel_visits_helen_l952_95201

-- Define the number of steps Mabel lives from Lake High school
def MabelSteps : ℕ := 4500

-- Define the number of steps Helen lives from the school
def HelenSteps : ℕ := (3 * MabelSteps) / 4

-- Define the total number of steps Mabel will walk to visit Helen
def TotalSteps : ℕ := MabelSteps + HelenSteps

-- Prove that the total number of steps Mabel walks to visit Helen is 7875
theorem mabel_visits_helen :
  TotalSteps = 7875 :=
sorry

end NUMINAMATH_GPT_mabel_visits_helen_l952_95201


namespace NUMINAMATH_GPT_find_divisible_xy9z_l952_95255

-- Define a predicate for numbers divisible by 132
def divisible_by_132 (n : ℕ) : Prop :=
  n % 132 = 0

-- Define the given number form \(\overline{xy9z}\) as a number maker
def form_xy9z (x y z : ℕ) : ℕ :=
  1000 * x + 100 * y + 90 + z

-- Stating the theorem for finding all numbers of form \(\overline{xy9z}\) that are divisible by 132
theorem find_divisible_xy9z (x y z : ℕ) :
  (divisible_by_132 (form_xy9z x y z)) ↔
  form_xy9z x y z = 3696 ∨
  form_xy9z x y z = 4092 ∨
  form_xy9z x y z = 6996 ∨
  form_xy9z x y z = 7392 :=
by sorry

end NUMINAMATH_GPT_find_divisible_xy9z_l952_95255


namespace NUMINAMATH_GPT_lance_more_pebbles_l952_95284

-- Given conditions
def candy_pebbles : ℕ := 4
def lance_pebbles : ℕ := 3 * candy_pebbles

-- Proof statement
theorem lance_more_pebbles : lance_pebbles - candy_pebbles = 8 :=
by
  sorry

end NUMINAMATH_GPT_lance_more_pebbles_l952_95284


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l952_95249

theorem quadratic_inequality_solution:
  ∃ P q : ℝ,
  (1 / P < 0) ∧
  (-P * q = 6) ∧
  (P^2 = 8) ∧
  (P = -2 * Real.sqrt 2) ∧
  (q = 3 / 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l952_95249


namespace NUMINAMATH_GPT_hyperbola_focal_length_l952_95237

theorem hyperbola_focal_length (m : ℝ) 
  (h0 : (∀ x y, x^2 / 16 - y^2 / m = 1)) 
  (h1 : (2 * Real.sqrt (16 + m) = 4 * Real.sqrt 5)) : 
  m = 4 := 
by sorry

end NUMINAMATH_GPT_hyperbola_focal_length_l952_95237


namespace NUMINAMATH_GPT_farm_own_more_horses_than_cows_after_transaction_l952_95247

theorem farm_own_more_horses_than_cows_after_transaction :
  ∀ (x : Nat), 
    3 * (3 * x - 15) = 5 * (x + 15) →
    75 - 45 = 30 :=
by
  intro x h
  -- This is a placeholder for the proof steps which we skip.
  sorry

end NUMINAMATH_GPT_farm_own_more_horses_than_cows_after_transaction_l952_95247


namespace NUMINAMATH_GPT_equilateral_triangle_l952_95268

theorem equilateral_triangle {a b c : ℝ} (h1 : a + b - c = 2) (h2 : 2 * a * b - c^2 = 4) : a = b ∧ b = c :=
by {
  sorry
}

end NUMINAMATH_GPT_equilateral_triangle_l952_95268


namespace NUMINAMATH_GPT_sum_of_reciprocals_negative_l952_95239

theorem sum_of_reciprocals_negative {a b c : ℝ} (h₁ : a + b + c = 0) (h₂ : a * b * c > 0) :
  1/a + 1/b + 1/c < 0 :=
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_negative_l952_95239


namespace NUMINAMATH_GPT_trailing_zeros_in_square_l952_95212

-- Define x as given in the conditions
def x : ℕ := 10^12 - 4

-- State the theorem which asserts that the number of trailing zeros in x^2 is 11
theorem trailing_zeros_in_square : 
  ∃ n : ℕ, n = 11 ∧ x^2 % 10^12 = 0 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_trailing_zeros_in_square_l952_95212


namespace NUMINAMATH_GPT_graph_of_equation_l952_95238

theorem graph_of_equation (x y : ℝ) : (x + y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := 
by
  sorry

end NUMINAMATH_GPT_graph_of_equation_l952_95238


namespace NUMINAMATH_GPT_find_valid_pairs_l952_95231

-- Decalred the main definition for the problem.
def valid_pairs (x y : ℕ) : Prop :=
  (10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99) ∧ ((x + y)^2 = 100 * x + y)

-- Stating the theorem without the proof.
theorem find_valid_pairs :
  valid_pairs 20 25 ∧ valid_pairs 30 25 :=
sorry

end NUMINAMATH_GPT_find_valid_pairs_l952_95231


namespace NUMINAMATH_GPT_find_x_value_l952_95218

open Real

theorem find_x_value (a b c : ℤ) (x : ℝ) (h : 5 / (a^2 + b * log x) = c) : 
  x = 10^((5 / c - a^2) / b) := 
by 
  sorry

end NUMINAMATH_GPT_find_x_value_l952_95218


namespace NUMINAMATH_GPT_loss_percentage_l952_95258

variable (CP SP : ℕ) -- declare the variables for cost price and selling price

theorem loss_percentage (hCP : CP = 1400) (hSP : SP = 1190) : 
  ((CP - SP) / CP * 100) = 15 := by
sorry

end NUMINAMATH_GPT_loss_percentage_l952_95258


namespace NUMINAMATH_GPT_minimum_width_l952_95299

theorem minimum_width (A l w : ℝ) (hA : A >= 150) (hl : l = 2 * w) (hA_def : A = w * l) : 
  w >= 5 * Real.sqrt 3 := 
  by
    -- Using the given conditions, we can prove that w >= 5 * sqrt(3)
    sorry

end NUMINAMATH_GPT_minimum_width_l952_95299


namespace NUMINAMATH_GPT_angle_sum_solution_l952_95206

theorem angle_sum_solution
  (x : ℝ)
  (h : 3 * x + 140 = 360) :
  x = 220 / 3 :=
by
  sorry

end NUMINAMATH_GPT_angle_sum_solution_l952_95206


namespace NUMINAMATH_GPT_minimum_f_value_l952_95272

noncomputable def f (x y : ℝ) : ℝ :=
  y / x + 16 * x / (2 * x + y)

theorem minimum_f_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ t, (∀ x y, f x y ≥ t) ∧ t = 6 := sorry

end NUMINAMATH_GPT_minimum_f_value_l952_95272


namespace NUMINAMATH_GPT_find_FC_l952_95228

theorem find_FC 
  (DC CB AD: ℝ)
  (h1 : DC = 9)
  (h2 : CB = 6)
  (h3 : AB = (1 / 3) * AD)
  (h4 : ED = (2 / 3) * AD) :
  FC = 9 :=
sorry

end NUMINAMATH_GPT_find_FC_l952_95228


namespace NUMINAMATH_GPT_pencil_length_total_l952_95260

theorem pencil_length_total :
  (1.5 + 0.5 + 2 + 1.25 + 0.75 + 1.8 + 2.5 = 10.3) :=
by
  sorry

end NUMINAMATH_GPT_pencil_length_total_l952_95260


namespace NUMINAMATH_GPT_solution_set_l952_95219

-- Define the conditions
variable (f : ℝ → ℝ)
variable (odd_func : ∀ x : ℝ, f (-x) = -f x)
variable (increasing_pos : ∀ a b : ℝ, 0 < a → 0 < b → a < b → f a < f b)
variable (f_neg3_zero : f (-3) = 0)

-- State the theorem
theorem solution_set (x : ℝ) : x * f x < 0 ↔ (-3 < x ∧ x < 0 ∨ 0 < x ∧ x < 3) :=
sorry

end NUMINAMATH_GPT_solution_set_l952_95219


namespace NUMINAMATH_GPT_train_speed_l952_95259

def distance := 11.67 -- distance in km
def time := 10.0 / 60.0 -- time in hours (10 minutes is 10/60 hours)

theorem train_speed : (distance / time) = 70.02 := by
  sorry

end NUMINAMATH_GPT_train_speed_l952_95259


namespace NUMINAMATH_GPT_ball_hits_ground_in_3_seconds_l952_95280

noncomputable def ball_height (t : ℝ) : ℝ := -16 * t^2 - 32 * t + 240

theorem ball_hits_ground_in_3_seconds :
  ∃ t : ℝ, ball_height t = 0 ∧ t = 3 :=
sorry

end NUMINAMATH_GPT_ball_hits_ground_in_3_seconds_l952_95280


namespace NUMINAMATH_GPT_length_of_AB_l952_95240

theorem length_of_AB (x1 y1 x2 y2 : ℝ) 
  (h_parabola_A : y1^2 = 8 * x1) 
  (h_focus_line_A : y1 = 2 * (x1 - 2)) 
  (h_parabola_B : y2^2 = 8 * x2) 
  (h_focus_line_B : y2 = 2 * (x2 - 2)) 
  (h_sum_x : x1 + x2 = 6) : 
  |x1 - x2| = 10 :=
sorry

end NUMINAMATH_GPT_length_of_AB_l952_95240


namespace NUMINAMATH_GPT_letters_identity_l952_95296

theorem letters_identity (l1 l2 l3 : Prop) 
  (h1 : l1 → l2 → false)
  (h2 : ¬(l1 ∧ l3))
  (h3 : ¬(l2 ∧ l3))
  (h4 : l3 → ¬l1 ∧ l2 ∧ ¬(l1 ∧ ¬l2)) :
  (¬l1 ∧ l2 ∧ ¬l3) :=
by 
  sorry

end NUMINAMATH_GPT_letters_identity_l952_95296


namespace NUMINAMATH_GPT_sum_is_correct_l952_95251

def number : ℕ := 81
def added_number : ℕ := 15
def sum_value (x : ℕ) (y : ℕ) : ℕ := x + y

theorem sum_is_correct : sum_value number added_number = 96 := 
by 
  sorry

end NUMINAMATH_GPT_sum_is_correct_l952_95251


namespace NUMINAMATH_GPT_range_m_if_B_subset_A_range_m_if_A_inter_B_empty_l952_95236

variable (m : ℝ)

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def set_B : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Problem 1: Prove the range of m if B ⊆ A is (-∞, 3]
theorem range_m_if_B_subset_A : (set_B m ⊆ set_A) ↔ m ≤ 3 := sorry

-- Problem 2: Prove the range of m if A ∩ B = ∅ is m < 2 or m > 4
theorem range_m_if_A_inter_B_empty : (set_A ∩ set_B m = ∅) ↔ m < 2 ∨ m > 4 := sorry

end NUMINAMATH_GPT_range_m_if_B_subset_A_range_m_if_A_inter_B_empty_l952_95236


namespace NUMINAMATH_GPT_john_tax_rate_l952_95256

theorem john_tax_rate { P: Real → Real → Real → Real → Prop }:
  ∀ (cNikes cBoots totalPaid taxRate: ℝ), 
  cNikes = 150 →
  cBoots = 120 →
  totalPaid = 297 →
  taxRate = ((totalPaid - (cNikes + cBoots)) / (cNikes + cBoots)) * 100 →
  taxRate = 10 :=
by
  intros cNikes cBoots totalPaid taxRate HcNikes HcBoots HtotalPaid HtaxRate
  sorry

end NUMINAMATH_GPT_john_tax_rate_l952_95256


namespace NUMINAMATH_GPT_remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1_l952_95282

-- Definitions from the conditions
def a : ℕ := 3^302
def b : ℕ := 3^151 + 3^101 + 1

-- Theorem: Prove that the remainder when a + 302 is divided by b is 302.
theorem remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1 :
  (a + 302) % b = 302 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1_l952_95282


namespace NUMINAMATH_GPT_split_fraction_l952_95220

theorem split_fraction (n d a b x y : ℤ) (h_d : d = a * b) (h_ad : a.gcd b = 1) (h_frac : (n:ℚ) / (d:ℚ) = 58 / 77) (h_eq : 11 * x + 7 * y = 58) : 
  (58:ℚ) / 77 = (4:ℚ) / 7 + (2:ℚ) / 11 :=
by
  sorry

end NUMINAMATH_GPT_split_fraction_l952_95220


namespace NUMINAMATH_GPT_count_perfect_squares_diff_l952_95283

theorem count_perfect_squares_diff (a b : ℕ) : 
  ∃ (count : ℕ), 
  count = 25 ∧ 
  (∀ (a : ℕ), (∃ (b : ℕ), a^2 = 2 * b + 1 ∧ a^2 < 2500) ↔ (∃ (k : ℕ), 1 ≤ k ∧ k ≤ 25 ∧ 2 * k - 1 = a)) :=
by
  sorry

end NUMINAMATH_GPT_count_perfect_squares_diff_l952_95283


namespace NUMINAMATH_GPT_solve_inequality_l952_95267

theorem solve_inequality (a b x : ℝ) (h : a ≠ b) :
  a^2 * x + b^2 * (1 - x) ≥ (a * x + b * (1 - x))^2 ↔ 0 ≤ x ∧ x ≤ 1 :=
sorry

end NUMINAMATH_GPT_solve_inequality_l952_95267


namespace NUMINAMATH_GPT_solve_inequality_l952_95263

theorem solve_inequality : 
  {x : ℝ | (x^3 - x^2 - 6 * x) / (x^2 - 3 * x + 2) > 0} = 
  {x : ℝ | (-2 < x ∧ x < 0) ∨ (1 < x ∧ x < 2) ∨ (3 < x)} :=
sorry

end NUMINAMATH_GPT_solve_inequality_l952_95263


namespace NUMINAMATH_GPT_proportion_a_value_l952_95202

theorem proportion_a_value (a b c d : ℝ) (h1 : b = 3) (h2 : c = 4) (h3 : d = 6) (h4 : a / b = c / d) : a = 2 :=
by sorry

end NUMINAMATH_GPT_proportion_a_value_l952_95202


namespace NUMINAMATH_GPT_total_students_l952_95261

theorem total_students (S F G B N : ℕ) 
  (hF : F = 41) 
  (hG : G = 22) 
  (hB : B = 9) 
  (hN : N = 24) 
  (h_total : S = (F + G - B) + N) : 
  S = 78 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l952_95261


namespace NUMINAMATH_GPT_valid_divisors_of_196_l952_95271

theorem valid_divisors_of_196 : 
  ∃ d : Finset Nat, (∀ x ∈ d, 1 < x ∧ x < 196 ∧ 196 % x = 0) ∧ d.card = 7 := by
  sorry

end NUMINAMATH_GPT_valid_divisors_of_196_l952_95271


namespace NUMINAMATH_GPT_solve_fractional_equation_l952_95295

theorem solve_fractional_equation :
  {x : ℝ | 1 / (x^2 + 8 * x - 6) + 1 / (x^2 + 5 * x - 6) + 1 / (x^2 - 14 * x - 6) = 0}
  = {3, -2, -6, 1} :=
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l952_95295


namespace NUMINAMATH_GPT_cups_remaining_l952_95287

-- Definitions based on problem conditions
def initial_cups : ℕ := 12
def mary_morning_cups : ℕ := 1
def mary_evening_cups : ℕ := 1
def frank_afternoon_cups : ℕ := 1
def frank_late_evening_cups : ℕ := 2 * frank_afternoon_cups

-- Hypothesis combining all conditions:
def total_given_cups : ℕ :=
  mary_morning_cups + mary_evening_cups + frank_afternoon_cups + frank_late_evening_cups

-- Theorem to prove
theorem cups_remaining : initial_cups - total_given_cups = 7 :=
  sorry

end NUMINAMATH_GPT_cups_remaining_l952_95287


namespace NUMINAMATH_GPT_chess_games_won_l952_95294

theorem chess_games_won (W L : ℕ) (h1 : W + L = 44) (h2 : 4 * L = 7 * W) : W = 16 :=
by
  sorry

end NUMINAMATH_GPT_chess_games_won_l952_95294


namespace NUMINAMATH_GPT_sally_seashells_l952_95257

theorem sally_seashells (T S: ℕ) (hT : T = 37) (h_total : T + S = 50) : S = 13 := by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_sally_seashells_l952_95257


namespace NUMINAMATH_GPT_remainder_n_plus_1008_l952_95227

variable (n : ℕ)

theorem remainder_n_plus_1008 (h1 : n % 4 = 1) (h2 : n % 5 = 3) : (n + 1008) % 4 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_n_plus_1008_l952_95227


namespace NUMINAMATH_GPT_solve_for_a_l952_95248

noncomputable def line_slope_parallels (a : ℝ) : Prop :=
  (a^2 - a) = 6

theorem solve_for_a : { a : ℝ // line_slope_parallels a } → (a = -2 ∨ a = 3) := by
  sorry

end NUMINAMATH_GPT_solve_for_a_l952_95248


namespace NUMINAMATH_GPT_mike_travel_miles_l952_95211

theorem mike_travel_miles
  (toll_fees_mike : ℝ) (toll_fees_annie : ℝ) (mike_start_fee : ℝ) 
  (annie_start_fee : ℝ) (mike_per_mile : ℝ) (annie_per_mile : ℝ) 
  (annie_travel_time : ℝ) (annie_speed : ℝ) (mike_cost : ℝ) 
  (annie_cost : ℝ) 
  (h_mike_cost_eq : mike_cost = mike_start_fee + toll_fees_mike + mike_per_mile * 36)
  (h_annie_cost_eq : annie_cost = annie_start_fee + toll_fees_annie + annie_per_mile * annie_speed * annie_travel_time)
  (h_equal_costs : mike_cost = annie_cost)
  : 36 = 36 :=
by 
  sorry

end NUMINAMATH_GPT_mike_travel_miles_l952_95211


namespace NUMINAMATH_GPT_admission_price_for_adults_l952_95273

-- Constants and assumptions
def children_ticket_price : ℕ := 25
def total_persons : ℕ := 280
def total_collected_dollars : ℕ := 140
def total_collected_cents : ℕ := total_collected_dollars * 100
def children_attended : ℕ := 80

-- Definitions based on the conditions
def adults_attended : ℕ := total_persons - children_attended
def total_amount_from_children : ℕ := children_attended * children_ticket_price
def total_amount_from_adults (A : ℕ) : ℕ := total_collected_cents - total_amount_from_children
def adult_ticket_price := (total_collected_cents - total_amount_from_children) / adults_attended

-- Theorem statement to be proved
theorem admission_price_for_adults : adult_ticket_price = 60 := by
  sorry

end NUMINAMATH_GPT_admission_price_for_adults_l952_95273


namespace NUMINAMATH_GPT_eight_n_plus_nine_is_perfect_square_l952_95279

theorem eight_n_plus_nine_is_perfect_square 
  (n : ℕ) (N : ℤ) 
  (hN : N = 2 ^ (4 * n + 1) - 4 ^ n - 1)
  (hdiv : 9 ∣ N) :
  ∃ k : ℤ, 8 * N + 9 = k ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_eight_n_plus_nine_is_perfect_square_l952_95279


namespace NUMINAMATH_GPT_sum_first_n_terms_geometric_sequence_l952_95226

def geometric_sequence_sum (n : ℕ) (k : ℝ) : ℝ :=
  if n = 0 then 0 else (3 * 2^n + k)

theorem sum_first_n_terms_geometric_sequence (k : ℝ) :
  (geometric_sequence_sum 1 k = 6 + k) ∧ 
  (∀ n > 1, geometric_sequence_sum n k - geometric_sequence_sum (n - 1) k = 3 * 2^(n-1))
  → k = -3 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_n_terms_geometric_sequence_l952_95226
