import Mathlib

namespace factorize_difference_of_squares_l2379_237916

variable (x : ℝ)

theorem factorize_difference_of_squares :
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorize_difference_of_squares_l2379_237916


namespace gcd_115_161_l2379_237904

theorem gcd_115_161 : Nat.gcd 115 161 = 23 := by
  sorry

end gcd_115_161_l2379_237904


namespace tan_of_angle_in_fourth_quadrant_l2379_237975

-- Define the angle α in the fourth quadrant in terms of its cosine value
variable (α : Real)
variable (h1 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) -- fourth quadrant condition
variable (h2 : Real.cos α = 4/5) -- given condition

-- Define the proof problem that tan α equals -3/4 given the conditions
theorem tan_of_angle_in_fourth_quadrant (α : Real) (h1 : α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) (h2 : Real.cos α = 4/5) : 
  Real.tan α = -3/4 :=
sorry

end tan_of_angle_in_fourth_quadrant_l2379_237975


namespace total_weight_of_beef_l2379_237929

-- Define the conditions
def packages_weight := 4
def first_butcher_packages := 10
def second_butcher_packages := 7
def third_butcher_packages := 8

-- Define the total weight calculation
def total_weight := (first_butcher_packages * packages_weight) +
                    (second_butcher_packages * packages_weight) +
                    (third_butcher_packages * packages_weight)

-- The statement to prove
theorem total_weight_of_beef : total_weight = 100 := by
  -- proof goes here
  sorry

end total_weight_of_beef_l2379_237929


namespace average_is_20_l2379_237946

-- Define the numbers and the variable n
def a := 3
def b := 16
def c := 33
def n := 27
def d := n + 1

-- Define the sum of the numbers
def sum := a + b + c + d

-- Define the average as sum divided by 4
def average := sum / 4

-- Prove that the average is 20
theorem average_is_20 : average = 20 := by
  sorry

end average_is_20_l2379_237946


namespace santa_chocolate_candies_l2379_237981

theorem santa_chocolate_candies (C M : ℕ) (h₁ : C + M = 2023) (h₂ : C = 3 * M / 4) : C = 867 :=
sorry

end santa_chocolate_candies_l2379_237981


namespace percent_within_one_std_dev_l2379_237963

theorem percent_within_one_std_dev (m d : ℝ) (dist : ℝ → ℝ)
  (symm : ∀ x, dist (m + x) = dist (m - x))
  (less_than_upper_bound : ∀ x, (x < (m + d)) → dist x < 0.92) :
  ∃ p : ℝ, p = 0.84 :=
by
  sorry

end percent_within_one_std_dev_l2379_237963


namespace ned_pieces_left_l2379_237961

def boxes_bought : ℝ := 14.0
def boxes_given : ℝ := 7.0
def pieces_per_box : ℝ := 6.0
def boxes_left (bought : ℝ) (given : ℝ) : ℝ := bought - given
def total_pieces (boxes : ℝ) (pieces_per_box : ℝ) : ℝ := boxes * pieces_per_box

theorem ned_pieces_left : total_pieces (boxes_left boxes_bought boxes_given) pieces_per_box = 42.0 := by
  sorry

end ned_pieces_left_l2379_237961


namespace find_a_l2379_237927

theorem find_a (a b c : ℤ) (h : (∀ x : ℝ, (x - a) * (x - 5) + 4 = (x + b) * (x + c))) :
  a = 0 ∨ a = 1 :=
sorry

end find_a_l2379_237927


namespace simplify_evaluate_l2379_237970

def f (x y : ℝ) : ℝ := 4 * x^2 * y - (6 * x * y - 3 * (4 * x - 2) - x^2 * y) + 1

theorem simplify_evaluate : f (-2) (1/2) = -13 := by
  sorry

end simplify_evaluate_l2379_237970


namespace coffee_vacation_days_l2379_237910

theorem coffee_vacation_days 
  (pods_per_day : ℕ := 3)
  (pods_per_box : ℕ := 30)
  (box_cost : ℝ := 8.00)
  (total_spent : ℝ := 32) :
  (total_spent / box_cost) * pods_per_box / pods_per_day = 40 := 
by 
  sorry

end coffee_vacation_days_l2379_237910


namespace solve_for_n_l2379_237943

def number_of_balls : ℕ := sorry

axiom A : number_of_balls = 2

theorem solve_for_n (n : ℕ) (h : (1 + 1 + n = number_of_balls) ∧ ((n : ℝ) / (1 + 1 + n) = 1 / 2)) : n = 2 :=
sorry

end solve_for_n_l2379_237943


namespace expression_equals_k_times_10_pow_1007_l2379_237908

theorem expression_equals_k_times_10_pow_1007 :
  (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 588 * 10^1007 := by
  sorry

end expression_equals_k_times_10_pow_1007_l2379_237908


namespace trig_expression_value_l2379_237952

theorem trig_expression_value (x : ℝ) (h : Real.tan x = 1/2) :
  (2 * Real.sin x + 3 * Real.cos x) / (Real.cos x - Real.sin x) = 8 :=
by
  sorry

end trig_expression_value_l2379_237952


namespace rice_field_sacks_l2379_237997

theorem rice_field_sacks (x : ℝ)
  (h1 : ∀ x, x + 1.20 * x = 44) : x = 20 :=
sorry

end rice_field_sacks_l2379_237997


namespace greatest_possible_bent_strips_l2379_237924

theorem greatest_possible_bent_strips (strip_count : ℕ) (cube_length cube_faces flat_strip_cover : ℕ) 
  (unit_squares_per_face total_squares flat_strips unit_squares_covered_by_flats : ℕ):
  strip_count = 18 →
  cube_length = 3 →
  cube_faces = 6 →
  flat_strip_cover = 3 →
  unit_squares_per_face = cube_length * cube_length →
  total_squares = cube_faces * unit_squares_per_face →
  flat_strips = 4 →
  unit_squares_covered_by_flats = flat_strips * flat_strip_cover →
  ∃ bent_strips,
  flat_strips * flat_strip_cover + bent_strips * flat_strip_cover = total_squares 
  ∧ bent_strips = 14 := by
  intros
  -- skipped proof
  sorry

end greatest_possible_bent_strips_l2379_237924


namespace quadratic_roots_sum_l2379_237900

theorem quadratic_roots_sum (x₁ x₂ m : ℝ) 
  (eq1 : x₁^2 - (2 * m - 2) * x₁ + (m^2 - 2 * m) = 0) 
  (eq2 : x₂^2 - (2 * m - 2) * x₂ + (m^2 - 2 * m) = 0)
  (h : x₁ + x₂ = 10) : m = 6 :=
sorry

end quadratic_roots_sum_l2379_237900


namespace distance_between_Q_and_R_l2379_237985

noncomputable def distance_QR : Real :=
  let YZ := 9
  let XZ := 12
  let XY := 15
  
  -- assume QY = QX and tangent to YZ at Y, and RX = RY and tangent to XZ at X
  let QY := 12.5
  let QX := 12.5
  let RY := 12.5
  let RX := 12.5

  -- calculate and return the distance QR based on these assumptions
  (QX^2 + RY^2 - 2 * QX * RX * Real.cos 90)^(1/2)

theorem distance_between_Q_and_R (YZ XZ XY : ℝ) (QY QX RY RX : ℝ) (h1 : YZ = 9) (h2 : XZ = 12) (h3 : XY = 15)
  (h4 : QY = 12.5) (h5 : QX = 12.5) (h6 : RY = 12.5) (h7 : RX = 12.5) :
  distance_QR = 15 :=
by
  sorry

end distance_between_Q_and_R_l2379_237985


namespace intersection_vertices_of_regular_octagon_l2379_237999

noncomputable def set_A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1| + |p.2| = a ∧ a > 0}

def set_B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.1 * p.2| + 1 = |p.1| + |p.2|}

theorem intersection_vertices_of_regular_octagon (a : ℝ) :
  (∃ (p : ℝ × ℝ), p ∈ set_A a ∧ p ∈ set_B) ↔ (a = Real.sqrt 2 ∨ a = 2 + Real.sqrt 2) :=
  sorry

end intersection_vertices_of_regular_octagon_l2379_237999


namespace annulus_area_l2379_237957

theorem annulus_area (B C RW : ℝ) (h1 : B > C)
  (h2 : B^2 - (C + 5)^2 = RW^2) : 
  π * RW^2 = π * (B^2 - (C + 5)^2) :=
by
  sorry

end annulus_area_l2379_237957


namespace misha_scored_48_in_second_attempt_l2379_237912

theorem misha_scored_48_in_second_attempt (P1 P2 P3 : ℕ)
  (h1 : P2 = 2 * P1)
  (h2 : P3 = (3 / 2) * P2)
  (h3 : 24 ≤ P1)
  (h4 : (3 / 2) * 2 * P1 = 72) : P2 = 48 :=
by sorry

end misha_scored_48_in_second_attempt_l2379_237912


namespace AM_GM_inequality_l2379_237994

theorem AM_GM_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : 
  (a / b + b / c + c / d + d / a) ≥ 4 := 
sorry

end AM_GM_inequality_l2379_237994


namespace tiffany_ate_pies_l2379_237945

theorem tiffany_ate_pies (baking_days : ℕ) (pies_per_day : ℕ) (wc_per_pie : ℕ) 
                         (remaining_wc : ℕ) (total_pies : ℕ) (total_wc : ℕ) :
  baking_days = 11 → pies_per_day = 3 → wc_per_pie = 2 → remaining_wc = 58 →
  total_pies = pies_per_day * baking_days → total_wc = total_pies * wc_per_pie →
  (total_wc - remaining_wc) / wc_per_pie = 4 :=
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end tiffany_ate_pies_l2379_237945


namespace problem_l2379_237944

def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}
def C := (Aᶜ) ∩ B

theorem problem : C = {x | 0 < x ∧ x ≤ 2} :=
by
  sorry

end problem_l2379_237944


namespace cars_on_river_road_l2379_237953

theorem cars_on_river_road (B C : ℕ) (h1 : B = C - 40) (h2 : B * 3 = C) : C = 60 := 
sorry

end cars_on_river_road_l2379_237953


namespace ratio_problem_l2379_237941

theorem ratio_problem
  (a b c d e : ℚ)
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 2) :
  e / a = 2 / 35 := 
sorry

end ratio_problem_l2379_237941


namespace number_of_keepers_l2379_237919

theorem number_of_keepers (hens goats camels : ℕ) (keepers feet heads : ℕ)
  (h_hens : hens = 50)
  (h_goats : goats = 45)
  (h_camels : camels = 8)
  (h_equation : (2 * hens + 4 * goats + 4 * camels + 2 * keepers) = (hens + goats + camels + keepers + 224))
  : keepers = 15 :=
by
sorry

end number_of_keepers_l2379_237919


namespace shortest_distance_from_circle_to_line_l2379_237962

theorem shortest_distance_from_circle_to_line :
  let circle := { p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 - 3)^2 = 9 }
  let line := { p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 2 = 0 }
  ∀ (M : ℝ × ℝ), M ∈ circle → ∃ d : ℝ, d = 2 ∧ ∀ q ∈ line, dist M q = d := 
sorry

end shortest_distance_from_circle_to_line_l2379_237962


namespace regular_polygon_exterior_angle_l2379_237959

theorem regular_polygon_exterior_angle (n : ℕ) (h : 1 ≤ n) :
  (360 : ℝ) / (n : ℝ) = 60 → n = 6 :=
by
  intro h1
  sorry

end regular_polygon_exterior_angle_l2379_237959


namespace abs_b_lt_abs_a_lt_2abs_b_l2379_237980

variable {a b : ℝ}

theorem abs_b_lt_abs_a_lt_2abs_b (h : (6 * a + 9 * b) / (a + b) < (4 * a - b) / (a - b)) :
  |b| < |a| ∧ |a| < 2 * |b| :=
sorry

end abs_b_lt_abs_a_lt_2abs_b_l2379_237980


namespace decimal_equivalent_of_fraction_l2379_237914

theorem decimal_equivalent_of_fraction :
  (16 : ℚ) / 50 = 32 / 100 :=
by sorry

end decimal_equivalent_of_fraction_l2379_237914


namespace calc_length_RS_l2379_237932

-- Define the trapezoid properties
def trapezoid (PQRS : Type) (PR QS : ℝ) (h A : ℝ) : Prop :=
  PR = 12 ∧ QS = 20 ∧ h = 10 ∧ A = 180

-- Define the length of the side RS
noncomputable def length_RS (PQRS : Type) (PR QS h A : ℝ) : ℝ :=
  18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3

-- Define the theorem statement
theorem calc_length_RS {PQRS : Type} (PR QS h A : ℝ) :
  trapezoid PQRS PR QS h A → length_RS PQRS PR QS h A = 18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3 :=
by
  intros
  exact Eq.refl (18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3)

end calc_length_RS_l2379_237932


namespace rows_per_floor_l2379_237922

theorem rows_per_floor
  (right_pos : ℕ) (left_pos : ℕ)
  (floors : ℕ) (total_cars : ℕ)
  (h_right : right_pos = 5) (h_left : left_pos = 4)
  (h_floors : floors = 10) (h_total : total_cars = 1600) :
  ∃ rows_per_floor : ℕ, rows_per_floor = 20 :=
by {
  sorry
}

end rows_per_floor_l2379_237922


namespace correlation_height_weight_l2379_237958

def is_functional_relationship (pair: String) : Prop :=
  pair = "The area of a square and its side length" ∨
  pair = "The distance traveled by a vehicle moving at a constant speed and time"

def has_no_correlation (pair: String) : Prop :=
  pair = "A person's height and eyesight"

def is_correlation (pair: String) : Prop :=
  ¬ is_functional_relationship pair ∧ ¬ has_no_correlation pair

theorem correlation_height_weight :
  is_correlation "A person's height and weight" :=
by sorry

end correlation_height_weight_l2379_237958


namespace length_of_water_fountain_l2379_237974

theorem length_of_water_fountain :
  (∀ (L1 : ℕ), 20 * 14 = L1) ∧
  (35 * 3 = 21) →
  (20 * 14 = 56) := by
sorry

end length_of_water_fountain_l2379_237974


namespace petya_correct_square_l2379_237984

theorem petya_correct_square :
  ∃ x a b : ℕ, (1 ≤ x ∧ x ≤ 9) ∧
              (x^2 = 10 * a + b) ∧ 
              (2 * x = 10 * b + a) ∧
              (x^2 = 81) :=
by
  sorry

end petya_correct_square_l2379_237984


namespace total_profit_is_50_l2379_237978

-- Define the initial conditions
def initial_milk : ℕ := 80
def initial_water : ℕ := 20
def milk_cost_per_liter : ℕ := 22
def first_mixture_milk : ℕ := 40
def first_mixture_water : ℕ := 5
def first_mixture_price : ℕ := 19
def second_mixture_milk : ℕ := 25
def second_mixture_water : ℕ := 10
def second_mixture_price : ℕ := 18
def third_mixture_milk : ℕ := initial_milk - (first_mixture_milk + second_mixture_milk)
def third_mixture_water : ℕ := 5
def third_mixture_price : ℕ := 21

-- Define variables for revenue calculations
def first_mixture_revenue : ℕ := (first_mixture_milk + first_mixture_water) * first_mixture_price
def second_mixture_revenue : ℕ := (second_mixture_milk + second_mixture_water) * second_mixture_price
def third_mixture_revenue : ℕ := (third_mixture_milk + third_mixture_water) * third_mixture_price
def total_revenue : ℕ := first_mixture_revenue + second_mixture_revenue + third_mixture_revenue

-- Define the total milk cost
def total_milk_used : ℕ := first_mixture_milk + second_mixture_milk + third_mixture_milk
def total_cost : ℕ := total_milk_used * milk_cost_per_liter

-- Define the profit as the difference between total revenue and total cost
def profit : ℕ := total_revenue - total_cost

-- Prove that the total profit is Rs. 50
theorem total_profit_is_50 : profit = 50 := by
  sorry

end total_profit_is_50_l2379_237978


namespace find_number_l2379_237987

variable (a : ℕ) (n : ℕ)

theorem find_number (h₁ : a = 105) (h₂ : a ^ 3 = 21 * n * 45 * 49) : n = 25 :=
by
  sorry

end find_number_l2379_237987


namespace total_money_divided_l2379_237921

theorem total_money_divided (A B C : ℝ) (hA : A = 280) (h1 : A = (2 / 3) * (B + C)) (h2 : B = (2 / 3) * (A + C)) :
  A + B + C = 700 := by
  sorry

end total_money_divided_l2379_237921


namespace equilateral_triangle_of_roots_of_unity_l2379_237971

open Complex

/-- Given three distinct non-zero complex numbers z1, z2, z3 such that z1 * z2 = z3 ^ 2 and z2 * z3 = z1 ^ 2.
Prove that if z2 = z1 * alpha, then alpha is a cube root of unity and the points corresponding to z1, z2, z3
form an equilateral triangle in the complex plane -/
theorem equilateral_triangle_of_roots_of_unity {z1 z2 z3 : ℂ} (h1 : z1 ≠ 0) (h2 : z2 ≠ 0) (h3 : z3 ≠ 0)
  (h_distinct : z1 ≠ z2 ∧ z2 ≠ z3 ∧ z1 ≠ z3)
  (h1_2 : z1 * z2 = z3 ^ 2) (h2_3 : z2 * z3 = z1 ^ 2) (alpha : ℂ) (hz2 : z2 = z1 * alpha) :
  alpha^3 = 1 ∧ ∃ (w1 w2 w3 : ℂ), (w1 = z1) ∧ (w2 = z2) ∧ (w3 = z3) ∧ ((w1, w2, w3) = (z1, z1 * α, z3) 
  ∨ (w1, w2, w3) = (z3, z1, z1 * α) ∨ (w1, w2, w3) = (z1 * α, z3, z1)) 
  ∧ dist w1 w2 = dist w2 w3 ∧ dist w2 w3 = dist w3 w1 := sorry

end equilateral_triangle_of_roots_of_unity_l2379_237971


namespace ratio_of_albums_l2379_237998

variable (M K B A : ℕ)
variable (s : ℕ)

-- Conditions
def adele_albums := (A = 30)
def bridget_albums := (B = A - 15)
def katrina_albums := (K = 6 * B)
def miriam_albums := (M = s * K)
def total_albums := (M + K + B + A = 585)

-- Proof statement
theorem ratio_of_albums (h1 : adele_albums A) (h2 : bridget_albums B A) (h3 : katrina_albums K B) 
(h4 : miriam_albums M s K) (h5 : total_albums M K B A) :
  s = 5 :=
by
  sorry

end ratio_of_albums_l2379_237998


namespace isosceles_triangle_perimeter_l2379_237942

-- Define the conditions
def equilateral_triangle_side : ℕ := 15
def isosceles_triangle_side : ℕ := 15
def isosceles_triangle_base : ℕ := 10

-- Define the theorem to prove the perimeter of the isosceles triangle
theorem isosceles_triangle_perimeter : 
  (2 * isosceles_triangle_side + isosceles_triangle_base = 40) :=
by
  -- Placeholder for the actual proof
  sorry

end isosceles_triangle_perimeter_l2379_237942


namespace Vasya_distance_fraction_l2379_237954

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end Vasya_distance_fraction_l2379_237954


namespace surface_area_of_given_cube_l2379_237972

-- Define the cube with its volume
def volume_of_cube : ℝ := 4913

-- Define the side length of the cube
def side_of_cube : ℝ := volume_of_cube^(1/3)

-- Define the surface area of the cube
def surface_area_of_cube (side : ℝ) : ℝ := 6 * (side^2)

-- Statement of the theorem
theorem surface_area_of_given_cube : 
  surface_area_of_cube side_of_cube = 1734 := 
by
  -- Proof goes here
  sorry

end surface_area_of_given_cube_l2379_237972


namespace spending_spring_months_l2379_237955

theorem spending_spring_months (spend_end_March spend_end_June : ℝ)
  (h1 : spend_end_March = 1) (h2 : spend_end_June = 4) :
  (spend_end_June - spend_end_March) = 3 :=
by
  rw [h1, h2]
  norm_num

end spending_spring_months_l2379_237955


namespace poly_sum_of_squares_iff_nonneg_l2379_237992

open Polynomial

variable {R : Type*} [Ring R] [OrderedRing R]

theorem poly_sum_of_squares_iff_nonneg (A : Polynomial ℝ) :
  (∃ P Q : Polynomial ℝ, A = P^2 + Q^2) ↔ ∀ x : ℝ, 0 ≤ A.eval x := sorry

end poly_sum_of_squares_iff_nonneg_l2379_237992


namespace range_of_a_l2379_237948

noncomputable def f (x : ℝ) : ℝ := 6 / x - x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → x^2 + a * x - 6 > 0) ↔ 5 ≤ a :=
by
  sorry

end range_of_a_l2379_237948


namespace calculate_fg1_l2379_237979

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 + 2

theorem calculate_fg1 : f (g 1) = -1 :=
by {
  sorry
}

end calculate_fg1_l2379_237979


namespace packs_of_yellow_balls_l2379_237949

theorem packs_of_yellow_balls (Y : ℕ) : 
  3 * 19 + Y * 19 + 8 * 19 = 399 → Y = 10 :=
by sorry

end packs_of_yellow_balls_l2379_237949


namespace incorrect_conclusion_l2379_237902

theorem incorrect_conclusion (a b c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b > c) (h4 : c > 0) : ¬ (a / b > a / c) :=
sorry

end incorrect_conclusion_l2379_237902


namespace friend_reading_time_l2379_237947

theorem friend_reading_time (S : ℝ) (H1 : S > 0) (H2 : 3 = 2 * (3 / 2)) : 
  (1.5 / (5 * S)) = 0.3 :=
by 
  sorry

end friend_reading_time_l2379_237947


namespace determine_h_l2379_237937

variable {R : Type*} [CommRing R]

def h_poly (x : R) : R := -8*x^4 + 2*x^3 + 4*x^2 - 6*x + 2

theorem determine_h (x : R) :
  (8*x^4 - 4*x^2 + 2 + h_poly x = 2*x^3 - 6*x + 4) ->
  h_poly x = -8*x^4 + 2*x^3 + 4*x^2 - 6*x + 2 :=
by
  intro h
  sorry

end determine_h_l2379_237937


namespace part_a_answer_part_b_answer_l2379_237909

noncomputable def part_a_problem : Prop :=
  ∃! (x k : ℕ), x > 0 ∧ k > 0 ∧ 3^k - 1 = x^3

noncomputable def part_b_problem (n : ℕ) : Prop :=
  n > 1 ∧ n ≠ 3 → ∀ (x k : ℕ), ¬ (x > 0 ∧ k > 0 ∧ 3^k - 1 = x^n)

theorem part_a_answer : part_a_problem :=
  sorry

theorem part_b_answer (n : ℕ) : part_b_problem n :=
  sorry

end part_a_answer_part_b_answer_l2379_237909


namespace minimum_value_is_14_div_27_l2379_237996

noncomputable def minimum_value_expression (x : ℝ) : ℝ :=
  (Real.sin x)^8 + (Real.cos x)^8 + 1 / (Real.sin x)^6 + (Real.cos x)^6 + 1

theorem minimum_value_is_14_div_27 :
  ∃ x : ℝ, minimum_value_expression x = (14 / 27) :=
by
  sorry

end minimum_value_is_14_div_27_l2379_237996


namespace total_pens_count_l2379_237930

def total_pens (red black blue : ℕ) : ℕ :=
  red + black + blue

theorem total_pens_count :
  let red := 8
  let black := red + 10
  let blue := red + 7
  total_pens red black blue = 41 :=
by
  sorry

end total_pens_count_l2379_237930


namespace evaluate_polynomial_at_neg_one_l2379_237925

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

-- Define the value x at which we want to evaluate f
def x_val : ℝ := -1

-- State the theorem with the result using Horner's method
theorem evaluate_polynomial_at_neg_one : f x_val = 6 :=
by
  -- Approach to solution is in solution steps, skipped here
  sorry

end evaluate_polynomial_at_neg_one_l2379_237925


namespace marble_prism_weight_l2379_237915

theorem marble_prism_weight :
  let height := 8
  let base_side := 2
  let density := 2700
  let volume := base_side * base_side * height
  volume * density = 86400 :=
by
  let height := 8
  let base_side := 2
  let density := 2700
  let volume := base_side * base_side * height
  sorry

end marble_prism_weight_l2379_237915


namespace yujin_wire_length_is_correct_l2379_237931

def junhoe_wire_length : ℝ := 134.5
def multiplicative_factor : ℝ := 1.06
def yujin_wire_length (junhoe_length : ℝ) (factor : ℝ) : ℝ := junhoe_length * factor

theorem yujin_wire_length_is_correct : 
  yujin_wire_length junhoe_wire_length multiplicative_factor = 142.57 := 
by 
  sorry

end yujin_wire_length_is_correct_l2379_237931


namespace quadratic_ineq_solution_range_l2379_237907

theorem quadratic_ineq_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2*x^2 - 8*x - 4 - a > 0) ↔ a < -4 :=
by
  sorry

end quadratic_ineq_solution_range_l2379_237907


namespace albert_number_solution_l2379_237993

theorem albert_number_solution (A B C : ℝ) 
  (h1 : A = 2 * B + 1) 
  (h2 : B = 2 * C + 1) 
  (h3 : C = 2 * A + 2) : 
  A = -11 / 7 := 
by 
  sorry

end albert_number_solution_l2379_237993


namespace shaded_area_of_larger_circle_l2379_237913

theorem shaded_area_of_larger_circle (R r : ℝ) (A_larger A_smaller : ℝ)
  (hR : R = 9)
  (hr : r = 4.5)
  (hA_larger : A_larger = Real.pi * R^2)
  (hA_smaller : A_smaller = 3 * Real.pi * r^2) :
  A_larger - A_smaller = 20.25 * Real.pi := by
  sorry

end shaded_area_of_larger_circle_l2379_237913


namespace remainder_of_171_divided_by_21_l2379_237995

theorem remainder_of_171_divided_by_21 : 
  ∃ r, 171 = (21 * 8) + r ∧ r = 3 := 
by
  sorry

end remainder_of_171_divided_by_21_l2379_237995


namespace no_six_odd_numbers_sum_to_one_l2379_237956

theorem no_six_odd_numbers_sum_to_one (a b c d e f : ℕ)
  (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) (hd : d % 2 = 1) (he : e % 2 = 1) (hf : f % 2 = 1)
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) :
  (1 / a : ℝ) + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f ≠ 1 :=
by
  sorry

end no_six_odd_numbers_sum_to_one_l2379_237956


namespace distance_to_right_focus_l2379_237926

open Real

-- Define the elements of the problem
variable (a c : ℝ)
variable (P : ℝ × ℝ) -- Point P on the hyperbola
variable (F1 F2 : ℝ × ℝ) -- Left and right foci
variable (D : ℝ) -- The left directrix

-- Define conditions as Lean statements
def hyperbola_eq : Prop := (a ≠ 0) ∧ (c ≠ 0) ∧ (P.1^2 / a^2 - P.2^2 / 16 = 1)
def point_on_right_branch : Prop := P.1 > 0
def distance_diff : Prop := abs (dist P F1 - dist P F2) = 6
def distance_to_left_directrix : Prop := abs (P.1 - D) = 34 / 5

-- Define theorem to prove the distance from P to the right focus
theorem distance_to_right_focus
  (hp : hyperbola_eq a c P)
  (hbranch : point_on_right_branch P)
  (hdiff : distance_diff P F1 F2)
  (hdirectrix : distance_to_left_directrix P D) :
  dist P F2 = 16 / 3 :=
sorry

end distance_to_right_focus_l2379_237926


namespace fraction_product_l2379_237906

theorem fraction_product : (2 / 9) * (5 / 11) = 10 / 99 := 
by
  sorry

end fraction_product_l2379_237906


namespace circular_garden_area_l2379_237968

theorem circular_garden_area (r : ℝ) (A C : ℝ) (h_radius : r = 6) (h_relationship : C = (1 / 3) * A) 
  (h_circumference : C = 2 * Real.pi * r) (h_area : A = Real.pi * r ^ 2) : 
  A = 36 * Real.pi :=
by
  sorry

end circular_garden_area_l2379_237968


namespace elsa_final_marbles_l2379_237991

def initial_marbles : ℕ := 40
def marbles_lost_at_breakfast : ℕ := 3
def marbles_given_to_susie : ℕ := 5
def marbles_bought_by_mom : ℕ := 12
def twice_marbles_given_back : ℕ := 2 * marbles_given_to_susie

theorem elsa_final_marbles :
    initial_marbles
    - marbles_lost_at_breakfast
    - marbles_given_to_susie
    + marbles_bought_by_mom
    + twice_marbles_given_back = 54 := 
by
    sorry

end elsa_final_marbles_l2379_237991


namespace geometric_figure_area_l2379_237901

theorem geometric_figure_area :
  (∀ (z : ℂ),
     (0 < (z.re / 20)) ∧ ((z.re / 20) < 1) ∧ 
     (0 < (z.im / 20)) ∧ ((z.im / 20) < 1) ∧ 
     (0 < (20 / z.re)) ∧ ((20 / z.re) < 1) ∧ 
     (0 < (20 / z.im)) ∧ ((20 / z.im) < 1)) →
     (∃ (area : ℝ), area = 400 - 50 * Real.pi) :=
by
  sorry

end geometric_figure_area_l2379_237901


namespace difference_of_squares_l2379_237928

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 19) : x^2 - y^2 = 190 :=
by
  sorry

end difference_of_squares_l2379_237928


namespace velocity_at_2_l2379_237969

variable (t : ℝ) (s : ℝ)

noncomputable def displacement (t : ℝ) : ℝ := t^2 + 3 / t

noncomputable def velocity (t : ℝ) : ℝ := (deriv displacement) t

theorem velocity_at_2 : velocity t = 2 * 2 - (3 / 4) := by
  sorry

end velocity_at_2_l2379_237969


namespace raccoon_carrots_hid_l2379_237905

theorem raccoon_carrots_hid 
  (r : ℕ)
  (b : ℕ)
  (h1 : 5 * r = 8 * b)
  (h2 : b = r - 3) 
  : 5 * r = 40 :=
by
  sorry

end raccoon_carrots_hid_l2379_237905


namespace base7_addition_problem_l2379_237938

theorem base7_addition_problem
  (X Y : ℕ) :
  (5 * 7^1 + X * 7^0 + Y * 7^0 + 0 * 7^2 + 6 * 7^1 + 2 * 7^0) = (6 * 7^1 + 4 * 7^0 + X * 7^0 + 0 * 7^2) →
  X + 6 = 1 * 7 + 4 →
  Y + 2 = X →
  X + Y = 8 :=
by
  intro h1 h2 h3
  sorry

end base7_addition_problem_l2379_237938


namespace find_A_and_B_l2379_237990

theorem find_A_and_B : 
  ∃ A B : ℝ, 
    (A = 6.5 ∧ B = 0.5) ∧
    (∀ x : ℝ, (8 * x - 17) / ((3 * x + 5) * (x - 3)) = A / (3 * x + 5) + B / (x - 3)) :=
by
  sorry

end find_A_and_B_l2379_237990


namespace integer_solutions_for_xyz_l2379_237966

theorem integer_solutions_for_xyz (x y z : ℤ) : 
  (x - y - 1)^3 + (y - z - 2)^3 + (z - x + 3)^3 = 18 ↔
  (x = y ∧ y = z) ∨
  (x = y - 1 ∧ y = z) ∨
  (x = y ∧ y = z + 5) ∨
  (x = y + 4 ∧ y = z + 5) ∨
  (x = y + 4 ∧ z = y) ∨
  (x = y - 1 ∧ z = y + 4) :=
by {
  sorry
}

end integer_solutions_for_xyz_l2379_237966


namespace square_side_length_difference_l2379_237976

theorem square_side_length_difference : 
  let side_A := Real.sqrt 25
  let side_B := Real.sqrt 81
  side_B - side_A = 4 :=
by
  sorry

end square_side_length_difference_l2379_237976


namespace smallest_prime_sum_of_three_different_primes_is_19_l2379_237936

theorem smallest_prime_sum_of_three_different_primes_is_19 :
  ∃ (p : ℕ), Prime p ∧ p = 19 ∧ (∀ a b c : ℕ, a ≠ b → b ≠ c → a ≠ c → Prime a → Prime b → Prime c → a + b + c = p → p ≥ 19) :=
by
  sorry

end smallest_prime_sum_of_three_different_primes_is_19_l2379_237936


namespace geometric_sequence_a4_l2379_237986

theorem geometric_sequence_a4 :
    ∀ (a : ℕ → ℝ) (n : ℕ), 
    a 1 = 2 → 
    (∀ n : ℕ, a (n + 1) = 3 * a n) → 
    a 4 = 54 :=
by
  sorry

end geometric_sequence_a4_l2379_237986


namespace Mo_tea_cups_l2379_237911

theorem Mo_tea_cups (n t : ℕ) 
  (h1 : 2 * n + 5 * t = 36)
  (h2 : 5 * t = 2 * n + 14) : 
  t = 5 :=
by
  sorry

end Mo_tea_cups_l2379_237911


namespace xy_exists_5n_l2379_237939

theorem xy_exists_5n (n : ℕ) (hpos : 0 < n) :
  ∃ x y : ℤ, x^2 + y^2 = 5^n ∧ Int.gcd x 5 = 1 ∧ Int.gcd y 5 = 1 :=
sorry

end xy_exists_5n_l2379_237939


namespace derivative_at_pi_div_2_l2379_237988

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.cos x

theorem derivative_at_pi_div_2 : (deriv f (Real.pi / 2)) = 4 := 
by
  sorry

end derivative_at_pi_div_2_l2379_237988


namespace neg_cos_ge_a_l2379_237951

theorem neg_cos_ge_a (a : ℝ) : (¬ ∃ x : ℝ, Real.cos x ≥ a) ↔ a = 2 := 
sorry

end neg_cos_ge_a_l2379_237951


namespace triangle_square_ratio_l2379_237950

theorem triangle_square_ratio (t s : ℝ) 
  (h1 : 3 * t = 15) 
  (h2 : 4 * s = 12) : 
  t / s = 5 / 3 :=
by 
  -- skipping the proof
  sorry

end triangle_square_ratio_l2379_237950


namespace students_count_l2379_237920

theorem students_count :
  ∀ (sets marbles_per_set marbles_per_student total_students : ℕ),
    sets = 3 →
    marbles_per_set = 32 →
    marbles_per_student = 4 →
    total_students = (sets * marbles_per_set) / marbles_per_student →
    total_students = 24 :=
by
  intros sets marbles_per_set marbles_per_student total_students
  intros h_sets h_marbles_per_set h_marbles_per_student h_total_students
  rw [h_sets, h_marbles_per_set, h_marbles_per_student] at h_total_students
  exact h_total_students

end students_count_l2379_237920


namespace inequality_solution_set_empty_range_l2379_237989

theorem inequality_solution_set_empty_range (m : ℝ) :
  (∀ x : ℝ, mx^2 - mx - 1 < 0) ↔ -4 < m ∧ m ≤ 0 :=
by
  sorry

end inequality_solution_set_empty_range_l2379_237989


namespace num_divisible_by_10_l2379_237934

theorem num_divisible_by_10 (a b d : ℕ) (h1 : 100 ≤ a) (h2 : a ≤ 500) (h3 : 100 ≤ b) (h4 : b ≤ 500) (h5 : Nat.gcd d 10 = 10) :
  (b - a) / d + 1 = 41 := by
  sorry

end num_divisible_by_10_l2379_237934


namespace fraction_irreducible_l2379_237923

theorem fraction_irreducible (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by 
  sorry

end fraction_irreducible_l2379_237923


namespace find_f_prime_one_l2379_237965

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)
def f_condition (x : ℝ) : Prop := f (1 / x) = x / (1 + x)

theorem find_f_prime_one : f_condition 1 → deriv f 1 = -1 / 4 := by
  intro h
  sorry

end find_f_prime_one_l2379_237965


namespace find_number_l2379_237933

theorem find_number : ∃ x : ℝ, 3 * x - 1 = 2 * x ∧ x = 1 := sorry

end find_number_l2379_237933


namespace allen_change_l2379_237935

-- Define the cost per box and the number of boxes
def cost_per_box : ℕ := 7
def num_boxes : ℕ := 5

-- Define the total cost including the tip
def total_cost := num_boxes * cost_per_box
def tip := total_cost / 7
def total_paid := total_cost + tip

-- Define the amount given to the delivery person
def amount_given : ℕ := 100

-- Define the change received
def change := amount_given - total_paid

-- The statement to prove
theorem allen_change : change = 60 :=
by
  -- sorry is used here to skip the proof, as per the instruction
  sorry

end allen_change_l2379_237935


namespace curlers_total_l2379_237983

theorem curlers_total (P B G : ℕ) (h1 : 4 * P = P + B + G) (h2 : B = 2 * P) (h3 : G = 4) : 
  4 * P = 16 := 
by sorry

end curlers_total_l2379_237983


namespace polynomial_term_count_l2379_237977

open Nat

theorem polynomial_term_count (N : ℕ) (h : (N.choose 5) = 2002) : N = 17 :=
by
  sorry

end polynomial_term_count_l2379_237977


namespace find_number_l2379_237982

theorem find_number :
  (∃ m : ℝ, 56 = (3 / 2) * m) ∧ (56 = 0.7 * 80) → m = 37 := by
  sorry

end find_number_l2379_237982


namespace weight_of_b_l2379_237964

theorem weight_of_b (a b c : ℝ) (h1 : a + b + c = 135) (h2 : a + b = 80) (h3 : b + c = 82) : b = 27 :=
by
  sorry

end weight_of_b_l2379_237964


namespace roots_of_polynomial_l2379_237960

theorem roots_of_polynomial :
  (3 * (2 + Real.sqrt 3)^4 - 19 * (2 + Real.sqrt 3)^3 + 34 * (2 + Real.sqrt 3)^2 - 19 * (2 + Real.sqrt 3) + 3 = 0) ∧ 
  (3 * (2 - Real.sqrt 3)^4 - 19 * (2 - Real.sqrt 3)^3 + 34 * (2 - Real.sqrt 3)^2 - 19 * (2 - Real.sqrt 3) + 3 = 0) ∧
  (3 * ((7 + Real.sqrt 13) / 6)^4 - 19 * ((7 + Real.sqrt 13) / 6)^3 + 34 * ((7 + Real.sqrt 13) / 6)^2 - 19 * ((7 + Real.sqrt 13) / 6) + 3 = 0) ∧
  (3 * ((7 - Real.sqrt 13) / 6)^4 - 19 * ((7 - Real.sqrt 13) / 6)^3 + 34 * ((7 - Real.sqrt 13) / 6)^2 - 19 * ((7 - Real.sqrt 13) / 6) + 3 = 0) :=
by sorry

end roots_of_polynomial_l2379_237960


namespace born_in_1890_l2379_237973

theorem born_in_1890 (x : ℕ) (h1 : x^2 - x - 2 = 1890) (h2 : x^2 < 1950) : x = 44 :=
by {
    sorry
}

end born_in_1890_l2379_237973


namespace problem_statement_l2379_237918

noncomputable def find_sum (x y : ℝ) : ℝ := x + y

theorem problem_statement (x y : ℝ)
  (hx : |x| + x + y = 12)
  (hy : x + |y| - y = 14) :
  find_sum x y = 22 / 5 :=
sorry

end problem_statement_l2379_237918


namespace smallest_n_for_terminating_decimal_l2379_237903

theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ (∀ p: ℕ, (p ∣ (m + 150) → (p = 2 ∨ p = 5)) → m ≥ n)) ∧
  (∃ two_five_factors : ℕ, (two_five_factors = 5 ^ 3 * 2 ^ 3) ∧ (n + 150 = two_five_factors)) := 
by 
   exact ⟨50, by sorry⟩

end smallest_n_for_terminating_decimal_l2379_237903


namespace no_solution_eq1_l2379_237917

   theorem no_solution_eq1 : ¬ ∃ x, (3 - x) / (x - 4) - 1 / (4 - x) = 1 :=
   by
     sorry
   
end no_solution_eq1_l2379_237917


namespace abs_b_leq_one_l2379_237967

theorem abs_b_leq_one (a b : ℝ) (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) ≤ 1) : |b| ≤ 1 := 
sorry

end abs_b_leq_one_l2379_237967


namespace root_fraction_power_l2379_237940

theorem root_fraction_power (a : ℝ) (ha : a = 5) : 
  (a^(1/3)) / (a^(1/5)) = a^(2/15) := by
  sorry

end root_fraction_power_l2379_237940
