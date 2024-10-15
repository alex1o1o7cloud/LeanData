import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_sum_19_l2099_209998

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_19 (h1 : is_arithmetic_sequence a)
  (h2 : a 9 = 11) (h3 : a 11 = 9) (h4 : ∀ n, S n = n / 2 * (a 1 + a n)) :
  S 19 = 190 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_19_l2099_209998


namespace NUMINAMATH_GPT_length_of_living_room_l2099_209932

theorem length_of_living_room
  (l : ℝ) -- length of the living room
  (w : ℝ) -- width of the living room
  (boxes_coverage : ℝ) -- area covered by one box
  (initial_area : ℝ) -- area already covered
  (additional_boxes : ℕ) -- additional boxes required
  (total_area : ℝ) -- total area required
  (w_condition : w = 20)
  (boxes_coverage_condition : boxes_coverage = 10)
  (initial_area_condition : initial_area = 250)
  (additional_boxes_condition : additional_boxes = 7)
  (total_area_condition : total_area = l * w)
  (full_coverage_condition : additional_boxes * boxes_coverage + initial_area = total_area) :
  l = 16 := by
  sorry

end NUMINAMATH_GPT_length_of_living_room_l2099_209932


namespace NUMINAMATH_GPT_ivanov_entitled_to_12_million_rubles_l2099_209999

def equal_contributions (x : ℝ) : Prop :=
  let ivanov_contribution := 70 * x
  let petrov_contribution := 40 * x
  let sidorov_contribution := 44
  ivanov_contribution = 44 ∧ petrov_contribution = 44 ∧ (ivanov_contribution + petrov_contribution + sidorov_contribution) / 3 = 44

def money_ivanov_receives (x : ℝ) : ℝ :=
  let ivanov_contribution := 70 * x
  ivanov_contribution - 44

theorem ivanov_entitled_to_12_million_rubles :
  ∃ x : ℝ, equal_contributions x → money_ivanov_receives x = 12 :=
sorry

end NUMINAMATH_GPT_ivanov_entitled_to_12_million_rubles_l2099_209999


namespace NUMINAMATH_GPT_find_a_3_l2099_209996

noncomputable def a_n (n : ℕ) : ℤ := 2 + (n - 1)  -- Definition of the arithmetic sequence

theorem find_a_3 (d : ℤ) (a : ℕ → ℤ) 
  (h1 : a 1 = 2)
  (h2 : a 5 + a 7 = 2 * a 4 + 4) : a 3 = 4 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_3_l2099_209996


namespace NUMINAMATH_GPT_bottles_not_placed_in_crate_l2099_209954

-- Defining the constants based on the conditions
def bottles_per_crate : Nat := 12
def total_bottles : Nat := 130
def crates : Nat := 10

-- Theorem statement based on the question and the correct answer
theorem bottles_not_placed_in_crate :
  total_bottles - (bottles_per_crate * crates) = 10 :=
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_bottles_not_placed_in_crate_l2099_209954


namespace NUMINAMATH_GPT_anna_final_stamp_count_l2099_209979

theorem anna_final_stamp_count (anna_initial : ℕ) (alison_initial : ℕ) (jeff_initial : ℕ)
  (anna_receive_from_alison : ℕ) (anna_give_jeff : ℕ) (anna_receive_jeff : ℕ) :
  anna_initial = 37 →
  alison_initial = 28 →
  jeff_initial = 31 →
  anna_receive_from_alison = alison_initial / 2 →
  anna_give_jeff = 2 →
  anna_receive_jeff = 1 →
  ∃ result : ℕ, result = 50 :=
by
  intros
  sorry

end NUMINAMATH_GPT_anna_final_stamp_count_l2099_209979


namespace NUMINAMATH_GPT_ajay_saves_each_month_l2099_209918

def monthly_income : ℝ := 90000
def spend_household : ℝ := 0.50 * monthly_income
def spend_clothes : ℝ := 0.25 * monthly_income
def spend_medicines : ℝ := 0.15 * monthly_income
def total_spent : ℝ := spend_household + spend_clothes + spend_medicines
def amount_saved : ℝ := monthly_income - total_spent

theorem ajay_saves_each_month : amount_saved = 9000 :=
by sorry

end NUMINAMATH_GPT_ajay_saves_each_month_l2099_209918


namespace NUMINAMATH_GPT_cost_of_fixing_clothes_l2099_209904

def num_shirts : ℕ := 10
def num_pants : ℕ := 12
def time_per_shirt : ℝ := 1.5
def time_per_pant : ℝ := 3.0
def rate_per_hour : ℝ := 30.0

theorem cost_of_fixing_clothes : 
  let total_time := (num_shirts * time_per_shirt) + (num_pants * time_per_pant)
  let total_cost := total_time * rate_per_hour
  total_cost = 1530 :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_fixing_clothes_l2099_209904


namespace NUMINAMATH_GPT_VivianMailApril_l2099_209946

variable (piecesMailApril piecesMailMay piecesMailJune piecesMailJuly piecesMailAugust : ℕ)

-- Conditions
def condition_double_monthly (a b : ℕ) : Prop := b = 2 * a

axiom May : piecesMailMay = 10
axiom June : piecesMailJune = 20
axiom July : piecesMailJuly = 40
axiom August : piecesMailAugust = 80

axiom patternMay : condition_double_monthly piecesMailApril piecesMailMay
axiom patternJune : condition_double_monthly piecesMailMay piecesMailJune
axiom patternJuly : condition_double_monthly piecesMailJune piecesMailJuly
axiom patternAugust : condition_double_monthly piecesMailJuly piecesMailAugust

-- Statement to prove
theorem VivianMailApril :
  piecesMailApril = 5 :=
by
  sorry

end NUMINAMATH_GPT_VivianMailApril_l2099_209946


namespace NUMINAMATH_GPT_total_distance_walked_l2099_209953

variables
  (distance1 : ℝ := 1.2)
  (distance2 : ℝ := 0.8)
  (distance3 : ℝ := 1.5)
  (distance4 : ℝ := 0.6)
  (distance5 : ℝ := 2)

theorem total_distance_walked :
  distance1 + distance2 + distance3 + distance4 + distance5 = 6.1 :=
sorry

end NUMINAMATH_GPT_total_distance_walked_l2099_209953


namespace NUMINAMATH_GPT_smallest_value_expression_geq_three_l2099_209952

theorem smallest_value_expression_geq_three :
  ∀ (x y : ℝ), 4 + x^2 * y^4 + x^4 * y^2 - 3 * x^2 * y^2 ≥ 3 := 
by
  sorry

end NUMINAMATH_GPT_smallest_value_expression_geq_three_l2099_209952


namespace NUMINAMATH_GPT_convex_polygon_diagonals_25_convex_polygon_triangles_25_l2099_209978

-- Define a convex polygon with 25 sides
def convex_polygon_sides : ℕ := 25

-- Define the number of diagonals in a convex polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Define the number of triangles that can be formed by choosing any three vertices from n vertices
def number_of_triangles (n : ℕ) : ℕ := n.choose 3

-- Theorem to prove the number of diagonals is 275 for a convex polygon with 25 sides
theorem convex_polygon_diagonals_25 : number_of_diagonals convex_polygon_sides = 275 :=
by sorry

-- Theorem to prove the number of triangles is 2300 for a convex polygon with 25 sides
theorem convex_polygon_triangles_25 : number_of_triangles convex_polygon_sides = 2300 :=
by sorry

end NUMINAMATH_GPT_convex_polygon_diagonals_25_convex_polygon_triangles_25_l2099_209978


namespace NUMINAMATH_GPT_problem_solution_l2099_209962

theorem problem_solution
  (k : ℝ)
  (y : ℝ → ℝ)
  (quadratic_fn : ∀ x, y x = (k + 2) * x^(k^2 + k - 4))
  (increase_for_neg_x : ∀ x : ℝ, x < 0 → y (x + 1) > y x) :
  k = -3 ∧ (∀ m n : ℝ, -2 ≤ m ∧ m ≤ 1 → y m = n → -4 ≤ n ∧ n ≤ 0) := 
sorry

end NUMINAMATH_GPT_problem_solution_l2099_209962


namespace NUMINAMATH_GPT_value_of_f_at_minus_point_two_l2099_209989

noncomputable def f (x : ℝ) : ℝ := 1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem value_of_f_at_minus_point_two : f (-0.2) = 0.81873 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_f_at_minus_point_two_l2099_209989


namespace NUMINAMATH_GPT_find_x_l2099_209906

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 182) : x = 13 :=
sorry

end NUMINAMATH_GPT_find_x_l2099_209906


namespace NUMINAMATH_GPT_exponent_calculation_l2099_209971

theorem exponent_calculation (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : 
  a^(2 * m - 3 * n) = 9 / 8 := 
by
  sorry

end NUMINAMATH_GPT_exponent_calculation_l2099_209971


namespace NUMINAMATH_GPT_diameter_percentage_l2099_209984

theorem diameter_percentage (d_R d_S : ℝ) (h : π * (d_R / 2)^2 = 0.16 * π * (d_S / 2)^2) :
  (d_R / d_S) * 100 = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_diameter_percentage_l2099_209984


namespace NUMINAMATH_GPT_x_add_y_add_one_is_composite_l2099_209985

theorem x_add_y_add_one_is_composite (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (k : ℕ) (h : x^2 + x * y - y = k^2) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (x + y + 1 = a * b) :=
by
  sorry

end NUMINAMATH_GPT_x_add_y_add_one_is_composite_l2099_209985


namespace NUMINAMATH_GPT_c_share_l2099_209938

theorem c_share (x y z a b c : ℝ) 
  (H1 : b = (65/100) * a)
  (H2 : c = (40/100) * a)
  (H3 : a + b + c = 328) : 
  c = 64 := 
sorry

end NUMINAMATH_GPT_c_share_l2099_209938


namespace NUMINAMATH_GPT_least_value_of_b_l2099_209970

variable {x y b : ℝ}

noncomputable def condition_inequality (x y b : ℝ) : Prop :=
  (x^2 + y^2)^2 ≤ b * (x^4 + y^4)

theorem least_value_of_b (h : ∀ x y : ℝ, condition_inequality x y b) : b ≥ 2 := 
sorry

end NUMINAMATH_GPT_least_value_of_b_l2099_209970


namespace NUMINAMATH_GPT_compare_expressions_l2099_209925

theorem compare_expressions (a b : ℝ) : (a + 3) * (a - 5) < (a + 2) * (a - 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_compare_expressions_l2099_209925


namespace NUMINAMATH_GPT_relationship_xy_qz_l2099_209913

theorem relationship_xy_qz
  (a c b d : ℝ)
  (x y q z : ℝ)
  (h1 : a^(2 * x) = c^(2 * q) ∧ c^(2 * q) = b^2)
  (h2 : c^(3 * y) = a^(3 * z) ∧ a^(3 * z) = d^2) :
  x * y = q * z :=
by
  sorry

end NUMINAMATH_GPT_relationship_xy_qz_l2099_209913


namespace NUMINAMATH_GPT_max_sum_first_n_terms_l2099_209944

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem max_sum_first_n_terms (a1 : ℝ) (h1 : a1 > 0)
  (h2 : 5 * a_n a1 d 8 = 8 * a_n a1 d 13) :
  ∃ n : ℕ, n = 21 ∧ ∀ m : ℕ, S_n a1 d m ≤ S_n a1 d n :=
by
  sorry

end NUMINAMATH_GPT_max_sum_first_n_terms_l2099_209944


namespace NUMINAMATH_GPT_b_a_range_l2099_209966
open Real

-- Definitions of angles A, B, and sides a, b in an acute triangle ABC we assume that these are given.
variables {A B C a b c : ℝ}
variable {ABC_acute : A + B + C = π}
variable {angle_condition : B = 2 * A}
variable {sides : a = b * (sin A / sin B)}

theorem b_a_range (h₁ : 0 < A) (h₂ : A < π/2) (h₃ : 0 < C) (h₄ : C < π/2) :
  (∃ A, 30 * (π/180) < A ∧ A < 45 * (π/180)) → 
  (∃ b a, b / a = 2 * cos A) → 
  (∃ x : ℝ, x = b / a ∧ sqrt 2 < x ∧ x < sqrt 3) :=
sorry

end NUMINAMATH_GPT_b_a_range_l2099_209966


namespace NUMINAMATH_GPT_angle_SRT_l2099_209922

-- Define angles in degrees
def angle_P : ℝ := 50
def angle_Q : ℝ := 60
def angle_R : ℝ := 40

-- Define the problem: Prove that angle SRT is 30 degrees given the above conditions
theorem angle_SRT : 
  (angle_P = 50 ∧ angle_Q = 60 ∧ angle_R = 40) → (∃ angle_SRT : ℝ, angle_SRT = 30) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_angle_SRT_l2099_209922


namespace NUMINAMATH_GPT_smallest_number_increased_by_nine_divisible_by_8_11_24_l2099_209917

theorem smallest_number_increased_by_nine_divisible_by_8_11_24 :
  ∃ x : ℕ, (x + 9) % 8 = 0 ∧ (x + 9) % 11 = 0 ∧ (x + 9) % 24 = 0 ∧ x = 255 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_increased_by_nine_divisible_by_8_11_24_l2099_209917


namespace NUMINAMATH_GPT_circle_diameter_mn_origin_l2099_209942

-- Definitions based on conditions in (a)
def circle_equation (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 4 * y + m = 0
def line_equation (x y : ℝ) : Prop := x + 2 * y - 4 = 0
def orthogonal (x1 x2 y1 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem to prove (based on conditions and correct answer in (b))
theorem circle_diameter_mn_origin 
  (m : ℝ) 
  (x1 y1 x2 y2 : ℝ)
  (h1: circle_equation m x1 y1) 
  (h2: circle_equation m x2 y2)
  (h3: line_equation x1 y1)
  (h4: line_equation x2 y2)
  (h5: orthogonal x1 x2 y1 y2) :
  m = 8 / 5 := 
sorry

end NUMINAMATH_GPT_circle_diameter_mn_origin_l2099_209942


namespace NUMINAMATH_GPT_sum_fraction_nonnegative_le_one_l2099_209933

theorem sum_fraction_nonnegative_le_one 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc : a + b + c = 2) :
  a * b / (c^2 + 1) + b * c / (a^2 + 1) + c * a / (b^2 + 1) ≤ 1 :=
sorry

end NUMINAMATH_GPT_sum_fraction_nonnegative_le_one_l2099_209933


namespace NUMINAMATH_GPT_angle_D_measure_l2099_209982

theorem angle_D_measure (E D F : ℝ) (h1 : E + D + F = 180) (h2 : E = 30) (h3 : D = 2 * F) : D = 100 :=
by
  -- The proof is not required, only the statement
  sorry

end NUMINAMATH_GPT_angle_D_measure_l2099_209982


namespace NUMINAMATH_GPT_fraction_sum_identity_l2099_209976

variable (a b c : ℝ)

theorem fraction_sum_identity (h1 : a + b + c = 0) (h2 : a / b + b / c + c / a = 100) : 
  b / a + c / b + a / c = -103 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_fraction_sum_identity_l2099_209976


namespace NUMINAMATH_GPT_right_triangle_third_side_l2099_209968

/-- In a right triangle, given the lengths of two sides are 4 and 5, prove that the length of the
third side is either sqrt 41 or 3. -/
theorem right_triangle_third_side (a b : ℕ) (h1 : a = 4 ∨ a = 5) (h2 : b = 4 ∨ b = 5) (h3 : a ≠ b) :
  ∃ c, c = Real.sqrt 41 ∨ c = 3 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_third_side_l2099_209968


namespace NUMINAMATH_GPT_pineapple_rings_per_pineapple_l2099_209986

def pineapples_purchased : Nat := 6
def cost_per_pineapple : Nat := 3
def rings_sold_per_set : Nat := 4
def price_per_set_of_4_rings : Nat := 5
def profit_made : Nat := 72

theorem pineapple_rings_per_pineapple : (90 / 5 * 4 / 6) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_pineapple_rings_per_pineapple_l2099_209986


namespace NUMINAMATH_GPT_sequence_a_10_l2099_209916

theorem sequence_a_10 : ∀ {a : ℕ → ℕ}, (a 1 = 1) → (∀ n, a (n+1) = a n + 2^n) → (a 10 = 1023) :=
by
  intros a h1 h_rec
  sorry

end NUMINAMATH_GPT_sequence_a_10_l2099_209916


namespace NUMINAMATH_GPT_find_number_of_white_balls_l2099_209987

theorem find_number_of_white_balls (n : ℕ) (h : 6 / (6 + n) = 2 / 5) : n = 9 :=
sorry

end NUMINAMATH_GPT_find_number_of_white_balls_l2099_209987


namespace NUMINAMATH_GPT_find_x_l2099_209990

theorem find_x 
  (x : ℕ)
  (h : 3^x = 3^(20) * 3^(20) * 3^(18) + 3^(19) * 3^(20) * 3^(19) + 3^(18) * 3^(21) * 3^(19)) :
  x = 59 :=
sorry

end NUMINAMATH_GPT_find_x_l2099_209990


namespace NUMINAMATH_GPT_record_cost_calculation_l2099_209965

theorem record_cost_calculation :
  ∀ (books_owned book_price records_bought money_left total_selling_price money_spent_per_record record_cost : ℕ),
  books_owned = 200 →
  book_price = 3 / 2 →
  records_bought = 75 →
  money_left = 75 →
  total_selling_price = books_owned * book_price →
  money_spent_per_record = total_selling_price - money_left →
  record_cost = money_spent_per_record / records_bought →
  record_cost = 3 :=
by
  intros books_owned book_price records_bought money_left total_selling_price money_spent_per_record record_cost
  sorry

end NUMINAMATH_GPT_record_cost_calculation_l2099_209965


namespace NUMINAMATH_GPT_number_of_true_propositions_l2099_209928

variable {a b c : ℝ}

theorem number_of_true_propositions :
  (2 = (if (a > b → a * c ^ 2 > b * c ^ 2) then 1 else 0) +
       (if (a * c ^ 2 > b * c ^ 2 → a > b) then 1 else 0) +
       (if (¬(a * c ^ 2 > b * c ^ 2) → ¬(a > b)) then 1 else 0) +
       (if (¬(a > b) → ¬(a * c ^ 2 > b * c ^ 2)) then 1 else 0)) :=
sorry

end NUMINAMATH_GPT_number_of_true_propositions_l2099_209928


namespace NUMINAMATH_GPT_trips_to_collect_all_trays_l2099_209957

-- Definition of conditions
def trays_at_once : ℕ := 7
def trays_one_table : ℕ := 23
def trays_other_table : ℕ := 5

-- Theorem statement
theorem trips_to_collect_all_trays : 
  (trays_one_table / trays_at_once) + (if trays_one_table % trays_at_once = 0 then 0 else 1) + 
  (trays_other_table / trays_at_once) + (if trays_other_table % trays_at_once = 0 then 0 else 1) = 5 := 
by
  sorry

end NUMINAMATH_GPT_trips_to_collect_all_trays_l2099_209957


namespace NUMINAMATH_GPT_height_of_pyramid_l2099_209977

theorem height_of_pyramid :
  let edge_cube := 6
  let edge_base_square_pyramid := 10
  let cube_volume := edge_cube ^ 3
  let sphere_volume := cube_volume
  let pyramid_volume := 2 * sphere_volume
  let base_area_square_pyramid := edge_base_square_pyramid ^ 2
  let height_pyramid := 12.96
  pyramid_volume = (1 / 3) * base_area_square_pyramid * height_pyramid :=
by
  sorry

end NUMINAMATH_GPT_height_of_pyramid_l2099_209977


namespace NUMINAMATH_GPT_add_base6_l2099_209975

def base6_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  6 * d1 + d0

theorem add_base6 (a b : Nat) (ha : base6_to_base10 a = 23) (hb : base6_to_base10 b = 10) : 
  base6_to_base10 (53 : Nat) = 33 :=
by
  sorry

end NUMINAMATH_GPT_add_base6_l2099_209975


namespace NUMINAMATH_GPT_buckets_needed_to_fill_tank_l2099_209924

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem buckets_needed_to_fill_tank :
  let radius_tank := 8
  let height_tank := 32
  let radius_bucket := 8
  let volume_bucket := volume_of_sphere radius_bucket
  let volume_tank := volume_of_cylinder radius_tank height_tank
  volume_tank / volume_bucket = 3 :=
by sorry

end NUMINAMATH_GPT_buckets_needed_to_fill_tank_l2099_209924


namespace NUMINAMATH_GPT_solve_for_x_l2099_209941

theorem solve_for_x (x : ℚ) (h : (x + 10) / (x - 4) = (x + 3) / (x - 6)) : x = 48 / 5 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l2099_209941


namespace NUMINAMATH_GPT_arithmetic_sequence_a10_l2099_209927

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 = 1) (h3 : a 3 = 5) 
  (h_diff : d = (a 3 - a 1) / (3 - 1)) :
  a 10 = 19 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a10_l2099_209927


namespace NUMINAMATH_GPT_minimum_value_y_l2099_209959

variable {x y : ℝ}

theorem minimum_value_y (h : y * Real.log y = Real.exp (2 * x) - y * Real.log (2 * x)) : y ≥ Real.exp 1 :=
sorry

end NUMINAMATH_GPT_minimum_value_y_l2099_209959


namespace NUMINAMATH_GPT_marbles_before_purchase_l2099_209903

-- Lean 4 statement for the problem
theorem marbles_before_purchase (bought : ℝ) (total_now : ℝ) (initial : ℝ) 
    (h1 : bought = 134.0) 
    (h2 : total_now = 321) 
    (h3 : total_now = initial + bought) : 
    initial = 187 :=
by 
    sorry

end NUMINAMATH_GPT_marbles_before_purchase_l2099_209903


namespace NUMINAMATH_GPT_remainder_of_2519_div_8_l2099_209919

theorem remainder_of_2519_div_8 : 2519 % 8 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_of_2519_div_8_l2099_209919


namespace NUMINAMATH_GPT_equation_one_equation_two_l2099_209907

-- Equation (1): Show that for the equation ⟦ ∀ x, (x / (2 * x - 1) + 2 / (1 - 2 * x) = 3 ↔ x = 1 / 5) ⟧
theorem equation_one (x : ℝ) : (x / (2 * x - 1) + 2 / (1 - 2 * x) = 3) ↔ (x = 1 / 5) :=
sorry

-- Equation (2): Show that for the equation ⟦ ∀ x, ((4 / (x^2 - 4) - 1 / (x - 2) = 0) ↔ false) ⟧
theorem equation_two (x : ℝ) : (4 / (x^2 - 4) - 1 / (x - 2) = 0) ↔ false :=
sorry

end NUMINAMATH_GPT_equation_one_equation_two_l2099_209907


namespace NUMINAMATH_GPT_not_perfect_square_infinitely_many_l2099_209991

theorem not_perfect_square_infinitely_many (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_gt : b > a) (h_prime : Prime (b - a)) :
  ∃ᶠ n in at_top, ¬ IsSquare ((a ^ n + a + 1) * (b ^ n + b + 1)) :=
sorry

end NUMINAMATH_GPT_not_perfect_square_infinitely_many_l2099_209991


namespace NUMINAMATH_GPT_zionsDadX_l2099_209967

section ZionProblem

-- Define the conditions
variables (Z : ℕ) (D : ℕ) (X : ℕ)

-- Zion's current age
def ZionAge : Prop := Z = 8

-- Zion's dad's age in terms of Zion's age and X
def DadsAge : Prop := D = 4 * Z + X

-- Zion's dad's age in 10 years compared to Zion's age in 10 years
def AgeInTenYears : Prop := D + 10 = (Z + 10) + 27

-- The theorem statement to be proved
theorem zionsDadX :
  ZionAge Z →  
  DadsAge Z D X →  
  AgeInTenYears Z D →  
  X = 3 := 
sorry

end ZionProblem

end NUMINAMATH_GPT_zionsDadX_l2099_209967


namespace NUMINAMATH_GPT_each_child_play_time_l2099_209908

theorem each_child_play_time (n_children : ℕ) (game_time : ℕ) (children_per_game : ℕ)
  (h1 : n_children = 8) (h2 : game_time = 120) (h3 : children_per_game = 2) :
  ((children_per_game * game_time) / n_children) = 30 :=
  sorry

end NUMINAMATH_GPT_each_child_play_time_l2099_209908


namespace NUMINAMATH_GPT_cos_double_angle_l2099_209909

theorem cos_double_angle (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 3 / 5) : Real.cos (2 * α) = -7 / 25 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l2099_209909


namespace NUMINAMATH_GPT_roots_cubic_inv_sum_l2099_209936

theorem roots_cubic_inv_sum (a b c r s : ℝ) (h_eq : ∃ (r s : ℝ), r^2 * a + b * r - c = 0 ∧ s^2 * a + b * s - c = 0) :
  (1 / r^3) + (1 / s^3) = (b^3 + 3 * a * b * c) / c^3 :=
by
  sorry

end NUMINAMATH_GPT_roots_cubic_inv_sum_l2099_209936


namespace NUMINAMATH_GPT_polyhedron_faces_l2099_209981

theorem polyhedron_faces (V E : ℕ) (F T P : ℕ) (h1 : F = 40) (h2 : V - E + F = 2) (h3 : T + P = 40) 
  (h4 : E = (3 * T + 4 * P) / 2) (h5 : V = (160 - T) / 2 - 38) (h6 : P = 3) (h7 : T = 1) :
  100 * P + 10 * T + V = 351 :=
by
  sorry

end NUMINAMATH_GPT_polyhedron_faces_l2099_209981


namespace NUMINAMATH_GPT_total_students_l2099_209945

-- Given conditions
variable (A B : ℕ)
noncomputable def M_A := 80 * A
noncomputable def M_B := 70 * B

axiom classA_condition1 : M_A - 160 = 90 * (A - 8)
axiom classB_condition1 : M_B - 180 = 85 * (B - 6)

-- Required proof in Lean 4 statement
theorem total_students : A + B = 78 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l2099_209945


namespace NUMINAMATH_GPT_journeymen_percentage_after_layoff_l2099_209947

noncomputable def total_employees : ℝ := 20210
noncomputable def fraction_journeymen : ℝ := 2 / 7
noncomputable def total_journeymen : ℝ := total_employees * fraction_journeymen
noncomputable def laid_off_journeymen : ℝ := total_journeymen / 2
noncomputable def remaining_journeymen : ℝ := total_journeymen / 2
noncomputable def remaining_employees : ℝ := total_employees - laid_off_journeymen
noncomputable def journeymen_percentage : ℝ := (remaining_journeymen / remaining_employees) * 100

theorem journeymen_percentage_after_layoff : journeymen_percentage = 16.62 := by
  sorry

end NUMINAMATH_GPT_journeymen_percentage_after_layoff_l2099_209947


namespace NUMINAMATH_GPT_find_a_value_l2099_209972

noncomputable def A (a : ℝ) : Set ℝ := {x | x = a}
noncomputable def B (a : ℝ) : Set ℝ := if a = 0 then ∅ else {x | a * x = 1}

theorem find_a_value (a : ℝ) :
  (A a ∩ B a = B a) → (a = 1 ∨ a = -1 ∨ a = 0) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_value_l2099_209972


namespace NUMINAMATH_GPT_fraction_of_25_l2099_209902

theorem fraction_of_25 (x : ℝ) (h1 : 0.65 * 40 = 26) (h2 : 26 = x * 25 + 6) : x = 4 / 5 :=
sorry

end NUMINAMATH_GPT_fraction_of_25_l2099_209902


namespace NUMINAMATH_GPT_find_angle_y_l2099_209997

open Real

theorem find_angle_y 
    (angle_ABC angle_BAC : ℝ)
    (h1 : angle_ABC = 70)
    (h2 : angle_BAC = 50)
    (triangle_sum : ∀ {A B C : ℝ}, A + B + C = 180)
    (right_triangle_sum : ∀ D E : ℝ, D + E = 90) :
    30 = 30 :=
by
    -- Given, conditions, and intermediate results (skipped)
    sorry

end NUMINAMATH_GPT_find_angle_y_l2099_209997


namespace NUMINAMATH_GPT_last_four_digits_5_pow_2015_l2099_209963

theorem last_four_digits_5_pow_2015 :
  (5^2015) % 10000 = 8125 :=
by
  sorry

end NUMINAMATH_GPT_last_four_digits_5_pow_2015_l2099_209963


namespace NUMINAMATH_GPT_speed_in_still_water_l2099_209900

theorem speed_in_still_water (upstream downstream : ℝ) (h_upstream : upstream = 37) (h_downstream : downstream = 53) : 
  (upstream + downstream) / 2 = 45 := 
by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l2099_209900


namespace NUMINAMATH_GPT_minimum_value_of_f_l2099_209956

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x + 1)

theorem minimum_value_of_f (x : ℝ) (h : x > -1) : f x = 1 ↔ x = 0 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l2099_209956


namespace NUMINAMATH_GPT_winnie_keeps_balloons_l2099_209958

theorem winnie_keeps_balloons :
  let blueBalloons := 15
  let yellowBalloons := 40
  let purpleBalloons := 70
  let orangeBalloons := 90
  let friends := 9
  let totalBalloons := blueBalloons + yellowBalloons + purpleBalloons + orangeBalloons
  (totalBalloons % friends) = 8 := 
by 
  -- Definitions
  let blueBalloons := 15
  let yellowBalloons := 40
  let purpleBalloons := 70
  let orangeBalloons := 90
  let friends := 9
  let totalBalloons := blueBalloons + yellowBalloons + purpleBalloons + orangeBalloons
  -- Conclusion
  show totalBalloons % friends = 8
  sorry

end NUMINAMATH_GPT_winnie_keeps_balloons_l2099_209958


namespace NUMINAMATH_GPT_least_gumballs_to_get_four_same_color_l2099_209943

theorem least_gumballs_to_get_four_same_color
  (R W B : ℕ)
  (hR : R = 9)
  (hW : W = 7)
  (hB : B = 8) : 
  ∃ n, n = 10 ∧ (∀ m < n, ∀ r w b : ℕ, r + w + b = m → r < 4 ∧ w < 4 ∧ b < 4) ∧ 
  (∀ r w b : ℕ, r + w + b = n → r = 4 ∨ w = 4 ∨ b = 4) :=
sorry

end NUMINAMATH_GPT_least_gumballs_to_get_four_same_color_l2099_209943


namespace NUMINAMATH_GPT_abcd_solution_l2099_209994

-- Define the problem statement
theorem abcd_solution (a b c d : ℤ) (h1 : a + c = -2) (h2 : a * c + b + d = 3) (h3 : a * d + b * c = 4) (h4 : b * d = -10) : 
  a + b + c + d = 1 := by 
  sorry

end NUMINAMATH_GPT_abcd_solution_l2099_209994


namespace NUMINAMATH_GPT_average_transformation_l2099_209960

theorem average_transformation (a b c : ℝ) (h : (a + b + c) / 3 = 12) : ((2 * a + 1) + (2 * b + 2) + (2 * c + 3) + 2) / 4 = 20 :=
by
  sorry

end NUMINAMATH_GPT_average_transformation_l2099_209960


namespace NUMINAMATH_GPT_average_is_700_l2099_209950

-- Define the list of known numbers
def numbers_without_x : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755]

-- Define the value of x
def x : ℕ := 755

-- Define the list of all numbers including x
def all_numbers : List ℕ := numbers_without_x.append [x]

-- Define the total length of the list containing x
def n : ℕ := all_numbers.length

-- Define the sum of the numbers in the list including x
noncomputable def sum_all_numbers : ℕ := all_numbers.sum

-- Define the average formula
noncomputable def average : ℕ := sum_all_numbers / n

-- State the theorem
theorem average_is_700 : average = 700 := by
  sorry

end NUMINAMATH_GPT_average_is_700_l2099_209950


namespace NUMINAMATH_GPT_largest_possible_value_l2099_209923

theorem largest_possible_value (X Y Z m: ℕ) 
  (hX_range: 0 ≤ X ∧ X ≤ 4) 
  (hY_range: 0 ≤ Y ∧ Y ≤ 4) 
  (hZ_range: 0 ≤ Z ∧ Z ≤ 4) 
  (h1: m = 25 * X + 5 * Y + Z)
  (h2: m = 81 * Z + 9 * Y + X):
  m = 121 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_largest_possible_value_l2099_209923


namespace NUMINAMATH_GPT_q_minus_p_897_l2099_209995

def smallest_three_digit_integer_congruent_7_mod_13 := ∃ p : ℕ, p ≥ 100 ∧ p < 1000 ∧ p % 13 = 7
def smallest_four_digit_integer_congruent_7_mod_13 := ∃ q : ℕ, q ≥ 1000 ∧ q < 10000 ∧ q % 13 = 7

theorem q_minus_p_897 : 
  (∃ p : ℕ, p ≥ 100 ∧ p < 1000 ∧ p % 13 = 7) → 
  (∃ q : ℕ, q ≥ 1000 ∧ q < 10000 ∧ q % 13 = 7) → 
  ∀ p q : ℕ, 
    (p = 8*13+7) → 
    (q = 77*13+7) → 
    q - p = 897 :=
by
  intros h1 h2 p q hp hq
  sorry

end NUMINAMATH_GPT_q_minus_p_897_l2099_209995


namespace NUMINAMATH_GPT_hilda_loan_compounding_difference_l2099_209973

noncomputable def difference_due_to_compounding (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  let A_monthly := P * (1 + r / 12)^(12 * t)
  let A_annually := P * (1 + r)^t
  A_monthly - A_annually

theorem hilda_loan_compounding_difference :
  difference_due_to_compounding 8000 0.10 5 = 376.04 :=
sorry

end NUMINAMATH_GPT_hilda_loan_compounding_difference_l2099_209973


namespace NUMINAMATH_GPT_largest_four_digit_number_with_property_l2099_209974

theorem largest_four_digit_number_with_property :
  ∃ (a b c d : ℕ), (a = 9 ∧ b = 0 ∧ c = 9 ∧ d = 9 ∧ c = a + b ∧ d = b + c ∧ 1000 * a + 100 * b + 10 * c + d = 9099) :=
sorry

end NUMINAMATH_GPT_largest_four_digit_number_with_property_l2099_209974


namespace NUMINAMATH_GPT_diana_can_paint_statues_l2099_209930

theorem diana_can_paint_statues : (3 / 6) / (1 / 6) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_diana_can_paint_statues_l2099_209930


namespace NUMINAMATH_GPT_ones_digit_7_pow_35_l2099_209983

theorem ones_digit_7_pow_35 : (7^35) % 10 = 3 := 
by
  sorry

end NUMINAMATH_GPT_ones_digit_7_pow_35_l2099_209983


namespace NUMINAMATH_GPT_smallest_pos_int_for_congruence_l2099_209920

theorem smallest_pos_int_for_congruence :
  ∃ (n : ℕ), 5 * n % 33 = 980 % 33 ∧ n > 0 ∧ n = 19 := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_pos_int_for_congruence_l2099_209920


namespace NUMINAMATH_GPT_complement_union_l2099_209921

-- Definition of the universal set U
def U : Set ℤ := {x | x^2 - 5 * x - 6 ≤ 0}

-- Definition of set A
def A : Set ℤ := {x | x * (2 - x) ≥ 0}

-- Definition of set B
def B : Set ℤ := {1, 2, 3}

-- The proof statement
theorem complement_union (h : U = {x | x^2 - 5 * x - 6 ≤ 0} ∧ 
                           A = {x | x * (2 - x) ≥ 0} ∧ 
                           B = {1, 2, 3}) : 
  U \ (A ∪ B) = {-1, 4, 5, 6} :=
by {
  sorry
}

end NUMINAMATH_GPT_complement_union_l2099_209921


namespace NUMINAMATH_GPT_intersection_is_integer_for_m_l2099_209961

noncomputable def intersects_at_integer_point (m : ℤ) : Prop :=
∃ x y : ℤ, y = x - 4 ∧ y = m * x + 2 * m

theorem intersection_is_integer_for_m :
  intersects_at_integer_point 8 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_intersection_is_integer_for_m_l2099_209961


namespace NUMINAMATH_GPT_matrix_pow_C_50_l2099_209934

def C : Matrix (Fin 2) (Fin 2) ℤ := 
  !![3, 1; -4, -1]

theorem matrix_pow_C_50 : C^50 = !![101, 50; -200, -99] := 
  sorry

end NUMINAMATH_GPT_matrix_pow_C_50_l2099_209934


namespace NUMINAMATH_GPT_num_perfect_cubes_between_bounds_l2099_209905

   noncomputable def lower_bound := 2^8 + 1
   noncomputable def upper_bound := 2^18 + 1

   theorem num_perfect_cubes_between_bounds : 
     ∃ (k : ℕ), k = 58 ∧ (∀ (n : ℕ), (lower_bound ≤ n^3 ∧ n^3 ≤ upper_bound) ↔ (7 ≤ n ∧ n ≤ 64)) :=
   sorry
   
end NUMINAMATH_GPT_num_perfect_cubes_between_bounds_l2099_209905


namespace NUMINAMATH_GPT_fraction_meaningful_l2099_209964

theorem fraction_meaningful (x : ℝ) : (∃ y, y = 1 / (x - 2)) → x ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l2099_209964


namespace NUMINAMATH_GPT_calculator_change_problem_l2099_209911

theorem calculator_change_problem :
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  change_received = 28 := by
{
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  have h1 : scientific_cost = 16 := sorry
  have h2 : graphing_cost = 48 := sorry
  have h3 : total_cost = 72 := sorry
  have h4 : change_received = 28 := sorry
  exact h4
}

end NUMINAMATH_GPT_calculator_change_problem_l2099_209911


namespace NUMINAMATH_GPT_value_of_each_baseball_card_l2099_209955

theorem value_of_each_baseball_card (x : ℝ) (h : 2 * x + 3 = 15) : x = 6 := by
  sorry

end NUMINAMATH_GPT_value_of_each_baseball_card_l2099_209955


namespace NUMINAMATH_GPT_problem_inequality_solution_l2099_209992

theorem problem_inequality_solution (x : ℝ) :
  5 ≤ (x - 1) / (3 * x - 7) ∧ (x - 1) / (3 * x - 7) < 10 ↔ (69 / 29) < x ∧ x ≤ (17 / 7) :=
by sorry

end NUMINAMATH_GPT_problem_inequality_solution_l2099_209992


namespace NUMINAMATH_GPT_simplify_expression_l2099_209915

theorem simplify_expression : 
  (20 * (9 / 14) * (1 / 18) : ℚ) = (5 / 7) := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l2099_209915


namespace NUMINAMATH_GPT_units_digit_of_2_to_the_10_l2099_209948

theorem units_digit_of_2_to_the_10 : ∃ d : ℕ, (d < 10) ∧ (2^10 % 10 = d) ∧ (d == 4) :=
by {
  -- sorry to skip the proof
  sorry
}

end NUMINAMATH_GPT_units_digit_of_2_to_the_10_l2099_209948


namespace NUMINAMATH_GPT_coprime_integers_lt_15_l2099_209901

theorem coprime_integers_lt_15 : ∃ (S : Finset ℕ), S.card = 8 ∧ (∀ a ∈ S, a < 15 ∧ Nat.gcd a 15 = 1) :=
by
  sorry

end NUMINAMATH_GPT_coprime_integers_lt_15_l2099_209901


namespace NUMINAMATH_GPT_functional_equation_solutions_l2099_209980

theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x * f y) + f (x + y) = f (x * y)) : 
  (∀ x : ℝ, f x = 0) ∨
  (∀ x : ℝ, f x = x - 1) ∨
  (∀ x : ℝ, f x = 1 - x) :=
sorry

end NUMINAMATH_GPT_functional_equation_solutions_l2099_209980


namespace NUMINAMATH_GPT_basketball_free_throws_l2099_209929

theorem basketball_free_throws:
  ∀ (a b x : ℕ),
    3 * b = 4 * a →
    x = 2 * a →
    2 * a + 3 * b + x = 65 →
    x = 18 := 
by
  intros a b x h1 h2 h3
  sorry

end NUMINAMATH_GPT_basketball_free_throws_l2099_209929


namespace NUMINAMATH_GPT_negation_proposition_l2099_209937

theorem negation_proposition (x : ℝ) :
  ¬(∀ x : ℝ, x^2 - x + 3 > 0) ↔ ∃ x : ℝ, x^2 - x + 3 ≤ 0 := 
by { sorry }

end NUMINAMATH_GPT_negation_proposition_l2099_209937


namespace NUMINAMATH_GPT_arccos_sin_three_l2099_209940

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_arccos_sin_three_l2099_209940


namespace NUMINAMATH_GPT_sum_expression_l2099_209935

theorem sum_expression (x k : ℝ) (h1 : y = 3 * x) (h2 : z = k * y) : x + y + z = (4 + 3 * k) * x :=
by
  sorry

end NUMINAMATH_GPT_sum_expression_l2099_209935


namespace NUMINAMATH_GPT_sum_of_cubes_l2099_209931

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : a^3 + b^3 = 26 :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_l2099_209931


namespace NUMINAMATH_GPT_find_ages_l2099_209939

-- Definitions of the conditions
def cond1 (D S : ℕ) : Prop := D = 3 * S
def cond2 (D S : ℕ) : Prop := D + 5 = 2 * (S + 5)

-- Theorem statement
theorem find_ages (D S : ℕ) 
  (h1 : cond1 D S) 
  (h2 : cond2 D S) : 
  D = 15 ∧ S = 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_ages_l2099_209939


namespace NUMINAMATH_GPT_perpendicular_vectors_l2099_209951

open scoped BigOperators

noncomputable def i : ℝ × ℝ := (1, 0)
noncomputable def j : ℝ × ℝ := (0, 1)
noncomputable def u : ℝ × ℝ := (1, 3)
noncomputable def v : ℝ × ℝ := (3, -1)

theorem perpendicular_vectors :
  (u.1 * v.1 + u.2 * v.2) = 0 :=
by
  have hi : i = (1, 0) := rfl
  have hj : j = (0, 1) := rfl
  have hu : u = (1, 3) := rfl
  have hv : v = (3, -1) := rfl
  -- using the dot product definition for perpendicularity
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_l2099_209951


namespace NUMINAMATH_GPT_original_price_l2099_209949

theorem original_price (P : ℝ) (h1 : P + 0.10 * P = 330) : P = 300 := 
by
  sorry

end NUMINAMATH_GPT_original_price_l2099_209949


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2099_209988

theorem solution_set_of_inequality : { x : ℝ | x^2 - 2 * x + 1 ≤ 0 } = {1} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2099_209988


namespace NUMINAMATH_GPT_maximum_M_value_l2099_209910

theorem maximum_M_value (x y z u M : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < u)
  (h5 : x - 2 * y = z - 2 * u) (h6 : 2 * y * z = u * x) (h7 : z ≥ y) 
  : ∃ M, M ≤ z / y ∧ M ≤ 6 + 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_maximum_M_value_l2099_209910


namespace NUMINAMATH_GPT_length_of_rectangle_l2099_209914

theorem length_of_rectangle (L : ℝ) (W : ℝ) (A_triangle : ℝ) (hW : W = 4) (hA_triangle : A_triangle = 60)
  (hRatio : (L * W) / A_triangle = 2 / 5) : L = 6 :=
by
  sorry

end NUMINAMATH_GPT_length_of_rectangle_l2099_209914


namespace NUMINAMATH_GPT_no_rational_x_y_m_n_with_conditions_l2099_209926

noncomputable def f (t : ℚ) : ℚ := t^3 + t

theorem no_rational_x_y_m_n_with_conditions :
  ¬ ∃ (x y : ℚ) (m n : ℕ), xy = 3 ∧ m > 0 ∧ n > 0 ∧
    (f^[m] x = f^[n] y) := 
sorry

end NUMINAMATH_GPT_no_rational_x_y_m_n_with_conditions_l2099_209926


namespace NUMINAMATH_GPT_eq_value_of_2a_plus_b_l2099_209912

theorem eq_value_of_2a_plus_b (a b : ℝ) (h : abs (a + 2) + (b - 5)^2 = 0) : 2 * a + b = 1 := by
  sorry

end NUMINAMATH_GPT_eq_value_of_2a_plus_b_l2099_209912


namespace NUMINAMATH_GPT_black_cars_count_l2099_209993

theorem black_cars_count
    (r b : ℕ)
    (r_ratio : r = 33)
    (ratio_condition : r / b = 3 / 8) :
    b = 88 :=
by 
  sorry

end NUMINAMATH_GPT_black_cars_count_l2099_209993


namespace NUMINAMATH_GPT_Kath_payment_l2099_209969

noncomputable def reducedPrice (standardPrice discount : ℝ) : ℝ :=
  standardPrice - discount

noncomputable def totalCost (numPeople price : ℝ) : ℝ :=
  numPeople * price

theorem Kath_payment :
  let standardPrice := 8
  let discount := 3
  let numPeople := 6
  let movieTime := 16 -- 4 P.M. in 24-hour format
  let reduced := reducedPrice standardPrice discount
  totalCost numPeople reduced = 30 :=
by
  sorry

end NUMINAMATH_GPT_Kath_payment_l2099_209969
