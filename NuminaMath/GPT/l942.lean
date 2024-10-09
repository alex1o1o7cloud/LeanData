import Mathlib

namespace product_of_two_numbers_l942_94207

theorem product_of_two_numbers (a b : ℤ) (h1 : Int.gcd a b = 10) (h2 : Int.lcm a b = 90) : a * b = 900 := 
sorry

end product_of_two_numbers_l942_94207


namespace real_roots_in_intervals_l942_94260

theorem real_roots_in_intervals (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  ∃ x1 x2 : ℝ, (x1 = a / 3 ∨ x1 = -2 * b / 3) ∧ (x2 = a / 3 ∨ x2 = -2 * b / 3) ∧ x1 ≠ x2 ∧
  (a / 3 ≤ x1 ∧ x1 ≤ 2 * a / 3) ∧ (-2 * b / 3 ≤ x2 ∧ x2 ≤ -b / 3) ∧
  (x1 > 0 ∧ x2 < 0) ∧ (1 / x1 + 1 / (x1 - a) + 1 / (x1 + b) = 0) ∧
  (1 / x2 + 1 / (x2 - a) + 1 / (x2 + b) = 0) :=
sorry

end real_roots_in_intervals_l942_94260


namespace find_number_l942_94282

theorem find_number (x : ℤ) 
  (h1 : 3 * (2 * x + 9) = 51) : x = 4 := 
by 
  sorry

end find_number_l942_94282


namespace f_positive_for_all_x_f_min_value_negative_two_f_triangle_sides_l942_94219

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (4^x + k * 2^x + 1) / (4^x + 2^x + 1)

-- Part (1)
theorem f_positive_for_all_x (k : ℝ) : (∀ x : ℝ, f x k > 0) ↔ k > -2 := sorry

-- Part (2)
theorem f_min_value_negative_two (k : ℝ) : (∀ x : ℝ, f x k ≥ -2) → k = -8 := sorry

-- Part (3)
theorem f_triangle_sides (k : ℝ) :
  (∀ x1 x2 x3 : ℝ, (f x1 k + f x2 k > f x3 k) ∧ (f x2 k + f x3 k > f x1 k) ∧ (f x3 k + f x1 k > f x2 k)) ↔ (-1/2 ≤ k ∧ k ≤ 4) := sorry

end f_positive_for_all_x_f_min_value_negative_two_f_triangle_sides_l942_94219


namespace ratio_A_B_l942_94253

theorem ratio_A_B (A B C : ℕ) (h1 : A + B + C = 98) (h2 : B = 30) (h3 : 5 * C = 8 * B) : A / B = 2 / 3 := 
by sorry

end ratio_A_B_l942_94253


namespace trigonometric_identity_l942_94261

-- Define variables
variables (α : ℝ) (hα : α ∈ Ioc 0 π) (h_tan : Real.tan α = 2)

-- The Lean statement
theorem trigonometric_identity :
  Real.cos (5 * Real.pi / 2 + 2 * α) = -4 / 5 :=
sorry

end trigonometric_identity_l942_94261


namespace measure_of_arc_BD_l942_94221

-- Definitions for conditions
def diameter (A B M : Type) : Prop := sorry -- Placeholder definition for diameter
def chord (C D M : Type) : Prop := sorry -- Placeholder definition for chord intersecting at point M
def angle_measure (A B C : Type) (angle_deg: ℝ) : Prop := sorry -- Placeholder for angle measure
def arc_measure (C B : Type) (arc_deg: ℝ) : Prop := sorry -- Placeholder for arc measure

-- Main theorem to prove
theorem measure_of_arc_BD
  (A B C D M : Type)
  (h_diameter : diameter A B M)
  (h_chord : chord C D M)
  (h_angle_CMB : angle_measure C M B 73)
  (h_arc_BC : arc_measure B C 110) :
  ∃ (arc_BD : ℝ), arc_BD = 144 :=
by
  sorry

end measure_of_arc_BD_l942_94221


namespace fifty_percent_of_2002_is_1001_l942_94246

theorem fifty_percent_of_2002_is_1001 :
  (1 / 2) * 2002 = 1001 :=
sorry

end fifty_percent_of_2002_is_1001_l942_94246


namespace sum_of_altitudes_of_triangle_l942_94209

theorem sum_of_altitudes_of_triangle : 
  let x_intercept := 6
  let y_intercept := 16
  let area := 48
  let altitude1 := x_intercept
  let altitude2 := y_intercept
  let altitude3 := 48 / Real.sqrt (64 + 9)
  altitude1 + altitude2 + altitude3 = (22 * Real.sqrt 73 + 48) / Real.sqrt 73 :=
by
  let x_intercept := 6
  let y_intercept := 16
  let area := 48
  let altitude1 := x_intercept
  let altitude2 := y_intercept
  let altitude3 := 48 / Real.sqrt (64 + 9)
  sorry

end sum_of_altitudes_of_triangle_l942_94209


namespace models_kirsty_can_buy_l942_94278

def original_price : ℝ := 0.45
def saved_for_models : ℝ := 30 * original_price
def new_price : ℝ := 0.50

theorem models_kirsty_can_buy :
  saved_for_models / new_price = 27 :=
sorry

end models_kirsty_can_buy_l942_94278


namespace units_digit_of_expression_l942_94283

noncomputable def C : ℝ := 7 + Real.sqrt 50
noncomputable def D : ℝ := 7 - Real.sqrt 50

theorem units_digit_of_expression (C D : ℝ) (hC : C = 7 + Real.sqrt 50) (hD : D = 7 - Real.sqrt 50) : 
  ((C ^ 21 + D ^ 21) % 10) = 4 :=
  sorry

end units_digit_of_expression_l942_94283


namespace range_of_m_l942_94242

variable (p q : Prop)
variable (m : ℝ)
variable (hp : (∀ x y : ℝ, (x^2 / (2 * m) + y^2 / (1 - m) = 1) → (0 < m ∧ m < 1/3)))
variable (hq : (m^2 - 15 * m < 0))

theorem range_of_m (h_not_p_and_q : ¬ (p ∧ q)) (h_p_or_q : p ∨ q) :
  (1/3 ≤ m ∧ m < 15) :=
sorry

end range_of_m_l942_94242


namespace football_players_count_l942_94241

def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def softball_players : ℕ := 13
def total_players : ℕ := 59

theorem football_players_count :
  total_players - (cricket_players + hockey_players + softball_players) = 18 :=
by 
  sorry

end football_players_count_l942_94241


namespace cost_per_person_l942_94217

-- Definitions based on conditions
def totalCost : ℕ := 13500
def numberOfFriends : ℕ := 15

-- Main statement
theorem cost_per_person : totalCost / numberOfFriends = 900 :=
by sorry

end cost_per_person_l942_94217


namespace division_theorem_l942_94228

theorem division_theorem (k : ℕ) (h : k = 6) : 24 / k = 4 := by
  sorry

end division_theorem_l942_94228


namespace lines_parallel_l942_94288

theorem lines_parallel :
  ∀ (x y : ℝ), (x - y + 2 = 0) ∧ (x - y + 1 = 0) → False :=
by
  intros x y h
  sorry

end lines_parallel_l942_94288


namespace calculate_fraction_l942_94284

theorem calculate_fraction: (1 / (2 + 1 / (3 + 1 / 4))) = 13 / 30 := by
  sorry

end calculate_fraction_l942_94284


namespace total_amount_spent_l942_94216

theorem total_amount_spent (num_pigs num_hens avg_price_hen avg_price_pig : ℕ)
                          (h_num_pigs : num_pigs = 3)
                          (h_num_hens : num_hens = 10)
                          (h_avg_price_hen : avg_price_hen = 30)
                          (h_avg_price_pig : avg_price_pig = 300) :
                          num_hens * avg_price_hen + num_pigs * avg_price_pig = 1200 :=
by
  sorry

end total_amount_spent_l942_94216


namespace distance_from_minus_one_is_four_or_minus_six_l942_94210

theorem distance_from_minus_one_is_four_or_minus_six :
  {x : ℝ | abs (x + 1) = 5} = {-6, 4} :=
sorry

end distance_from_minus_one_is_four_or_minus_six_l942_94210


namespace billy_gaming_percentage_l942_94257

-- Define the conditions
def free_time_per_day := 8
def days_in_weekend := 2
def total_free_time := free_time_per_day * days_in_weekend
def books_read := 3
def pages_per_book := 80
def reading_rate := 60 -- pages per hour
def total_pages_read := books_read * pages_per_book
def reading_time := total_pages_read / reading_rate
def gaming_time := total_free_time - reading_time
def gaming_percentage := (gaming_time / total_free_time) * 100

-- State the theorem
theorem billy_gaming_percentage : gaming_percentage = 75 := by
  sorry

end billy_gaming_percentage_l942_94257


namespace simple_interest_rate_l942_94292

theorem simple_interest_rate (P A : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ) :
  P = 800 → A = 950 → T = 5 → SI = A - P → SI = (P * R * T) / 100 → R = 3.75 :=
  by
  intros hP hA hT hSI h_formula
  sorry

end simple_interest_rate_l942_94292


namespace D_coin_count_l942_94271

def A_coin_count : ℕ := 21
def B_coin_count := A_coin_count - 9
def C_coin_count := B_coin_count + 17
def sum_A_B := A_coin_count + B_coin_count
def sum_C_D := sum_A_B + 5

theorem D_coin_count :
  ∃ D : ℕ, sum_C_D - C_coin_count = D :=
sorry

end D_coin_count_l942_94271


namespace abs_ac_bd_leq_one_l942_94294

theorem abs_ac_bd_leq_one {a b c d : ℝ} (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) : |a * c + b * d| ≤ 1 :=
by
  sorry

end abs_ac_bd_leq_one_l942_94294


namespace intervals_of_monotonicity_minimum_value_l942_94212

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x

theorem intervals_of_monotonicity (a : ℝ) (h : a > 0) :
  (∀ x, 0 < x ∧ x ≤ 1 / a → f a x ≤ f a (1 / a)) ∧
  (∀ x, x ≥ 1 / a → f a x ≥ f a (1 / a)) :=
sorry

theorem minimum_value (a : ℝ) (h : a > 0) :
  (a < Real.log 2 → ∃ x ∈ Set.Icc (1:ℝ) (2:ℝ), f a x = -a) ∧
  (a ≥ Real.log 2 → ∃ x ∈ Set.Icc (1:ℝ) (2:ℝ), f a x = Real.log 2 - 2 * a) :=
sorry

end intervals_of_monotonicity_minimum_value_l942_94212


namespace sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l942_94231

theorem sum_of_consecutive_integers_a_lt_sqrt3_lt_b 
  (a b : ℤ) (h1 : a < b) (h2 : ∀ x : ℤ, x ≤ a → x < b) (h3 : a < Real.sqrt 3) (h4 : Real.sqrt 3 < b) : 
  a + b = 3 :=
by
  sorry

end sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l942_94231


namespace y_intercept_line_l942_94248

theorem y_intercept_line : ∀ y : ℝ, (∃ x : ℝ, x = 0 ∧ x - 3 * y - 1 = 0) → y = -1/3 :=
by
  intro y
  intro h
  sorry

end y_intercept_line_l942_94248


namespace equal_side_length_is_4_or_10_l942_94251

-- Define the conditions
def isosceles_triangle (base_length equal_side_length : ℝ) :=
  base_length = 7 ∧
  (equal_side_length > base_length ∧ equal_side_length - base_length = 3) ∨
  (equal_side_length < base_length ∧ base_length - equal_side_length = 3)

-- Lean 4 statement to prove
theorem equal_side_length_is_4_or_10 (base_length equal_side_length : ℝ) 
  (h : isosceles_triangle base_length equal_side_length) : 
  equal_side_length = 4 ∨ equal_side_length = 10 :=
by 
  sorry

end equal_side_length_is_4_or_10_l942_94251


namespace Yoongi_has_smaller_number_l942_94208

def Jungkook_number : ℕ := 6 + 3
def Yoongi_number : ℕ := 4

theorem Yoongi_has_smaller_number : Yoongi_number < Jungkook_number :=
by
  exact sorry

end Yoongi_has_smaller_number_l942_94208


namespace infinitely_many_divisors_l942_94272

theorem infinitely_many_divisors (a : ℕ) : ∃ᶠ n in at_top, n ∣ a ^ (n - a + 1) - 1 :=
sorry

end infinitely_many_divisors_l942_94272


namespace units_digit_37_pow_37_l942_94276

theorem units_digit_37_pow_37: (37^37) % 10 = 7 :=
by sorry

end units_digit_37_pow_37_l942_94276


namespace train_speed_on_time_l942_94222

theorem train_speed_on_time (v : ℕ) (t : ℕ) :
  (15 / v + 1 / 4 = 15 / 50) ∧ (t = 15) → v = 300 := by
  sorry

end train_speed_on_time_l942_94222


namespace area_of_region_l942_94215

theorem area_of_region : 
  (∃ x y : ℝ, |5 * x - 10| + |4 * y + 20| ≤ 10) →
  ∃ area : ℝ, 
  area = 10 :=
sorry

end area_of_region_l942_94215


namespace hexagon_side_count_l942_94252

noncomputable def convex_hexagon_sides (a b perimeter : ℕ) : ℕ := 
  if a ≠ b then 6 - (perimeter - (6 * b)) else 0

theorem hexagon_side_count (G H I J K L : ℕ)
  (a b : ℕ)
  (p : ℕ)
  (dist_a : a = 7)
  (dist_b : b = 8)
  (perimeter : p = 46)
  (cond : GHIJKL = [a, b, X, Y, Z, W] ∧ ∀ x ∈ [X, Y, Z, W], x = a ∨ x = b)
  : convex_hexagon_sides a b p = 4 :=
by 
  sorry

end hexagon_side_count_l942_94252


namespace simplify_fraction_l942_94254

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l942_94254


namespace parabola_equation_through_origin_point_l942_94205

-- Define the conditions
def vertex_origin := (0, 0)
def point_on_parabola := (-2, 4)

-- Define what it means to be a standard equation of a parabola passing through a point
def standard_equation_passing_through (p : ℝ) (x y : ℝ) : Prop :=
  (y^2 = -2 * p * x ∨ x^2 = 2 * p * y)

-- The theorem stating the conclusion
theorem parabola_equation_through_origin_point :
  ∃ p > 0, standard_equation_passing_through p (-2) 4 ∧
  (4^2 = -8 * (-2) ∨ (-2)^2 = 4) := 
sorry

end parabola_equation_through_origin_point_l942_94205


namespace movement_down_l942_94299

def point := (ℤ × ℤ)

theorem movement_down (C D : point) (hC : C = (1, 2)) (hD : D = (1, -1)) :
  D = (C.1, C.2 - 3) :=
by
  sorry

end movement_down_l942_94299


namespace find_common_ratio_l942_94281

-- We need to state that q is the common ratio of the geometric sequence

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

-- Define the sum of the first three terms for the geometric sequence
def S_3 (a : ℕ → ℝ) := a 0 + a 1 + a 2

-- State the Lean 4 declaration of the proof problem
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : (S_3 a) / (a 2) = 3) :
  q = 1 := 
sorry

end find_common_ratio_l942_94281


namespace mario_meet_speed_l942_94289

noncomputable def Mario_average_speed (x : ℝ) : ℝ :=
  let t1 := x / 5
  let t2 := x / 3
  let t3 := x / 4
  let t4 := x / 10
  let T := t1 + t2 + t3 + t4
  let d_mario := 1.5 * x
  d_mario / T

theorem mario_meet_speed : ∀ (x : ℝ), x > 0 → Mario_average_speed x = 90 / 53 :=
by
  intros
  rw [Mario_average_speed]
  -- You can insert calculations similar to those in the provided solution
  sorry

end mario_meet_speed_l942_94289


namespace length_of_AP_l942_94259

variables {x : ℝ} (M B C P A : Point) (circle : Circle)
  (BC AB MP : Line)

-- Definitions of conditions
def is_midpoint_of_arc (M B C : Point) (circle : Circle) : Prop := sorry
def is_perpendicular (MP AB : Line) (P : Point) : Prop := sorry
def chord_length (BC : Line) (length : ℝ) : Prop := sorry
def segment_length (BP : Line) (length : ℝ) : Prop := sorry

-- Prove statement
theorem length_of_AP
  (h1 : is_midpoint_of_arc M B C circle)
  (h2 : is_perpendicular MP AB P)
  (h3 : chord_length BC (2 * x))
  (h4 : segment_length BP (3 * x)) :
  ∃AP : Line, segment_length AP (2 * x) :=
sorry

end length_of_AP_l942_94259


namespace min_chord_length_m_l942_94266

-- Definition of the circle and the line
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 6 * y + 4 = 0
def line_eq (m x y : ℝ) : Prop := m * x - y + 1 = 0

-- Theorem statement: value of m that minimizes the length of the chord
theorem min_chord_length_m (m : ℝ) : m = 1 ↔
  ∃ x y : ℝ, circle_eq x y ∧ line_eq m x y := sorry

end min_chord_length_m_l942_94266


namespace roots_of_quadratic_eq_l942_94258

theorem roots_of_quadratic_eq {x y : ℝ} (h1 : x + y = 10) (h2 : (x - y) * (x + y) = 48) : 
    ∃ a b c : ℝ, (a ≠ 0) ∧ (x^2 - a*x + b = 0) ∧ (y^2 - a*y + b = 0) ∧ b = 19.24 := 
by
  sorry

end roots_of_quadratic_eq_l942_94258


namespace who_received_q_first_round_l942_94277

-- Define the variables and conditions
variables (p q r : ℕ) (A B C : ℕ → ℕ) (n : ℕ)

-- Conditions
axiom h1 : 0 < p
axiom h2 : p < q
axiom h3 : q < r
axiom h4 : n ≥ 3
axiom h5 : A n = 20
axiom h6 : B n = 10
axiom h7 : C n = 9
axiom h8 : ∀ k, k > 0 → (B k = r → B (k-1) ≠ r)
axiom h9 : p + q + r = 13

-- Theorem to prove
theorem who_received_q_first_round : C 1 = q :=
sorry

end who_received_q_first_round_l942_94277


namespace rhomboid_toothpicks_l942_94267

/-- 
Given:
- The rhomboid consists of two sections, each similar to half of a large equilateral triangle split along its height.
- The longest diagonal of the rhomboid contains 987 small equilateral triangles.
- The effective fact that each small equilateral triangle contributes on average 1.5 toothpicks due to shared sides.

Prove:
- The number of toothpicks required to construct the rhomboid is 1463598.
-/

-- Defining the number of small triangles along the base of the rhomboid
def base_triangles : ℕ := 987

-- Calculating the number of triangles in one section of the rhomboid
def triangles_in_section : ℕ := (base_triangles * (base_triangles + 1)) / 2

-- Calculating the total number of triangles in the rhomboid
def total_triangles : ℕ := 2 * triangles_in_section

-- Given the effective sides per triangle contributing to toothpicks is on average 1.5
def avg_sides_per_triangle : ℚ := 1.5

-- Calculating the total number of toothpicks required
def total_toothpicks : ℚ := avg_sides_per_triangle * total_triangles

theorem rhomboid_toothpicks (h : base_triangles = 987) : total_toothpicks = 1463598 := by
  sorry

end rhomboid_toothpicks_l942_94267


namespace paintings_total_l942_94244

def june_paintings : ℕ := 2
def july_paintings : ℕ := 2 * june_paintings
def august_paintings : ℕ := 3 * july_paintings
def total_paintings : ℕ := june_paintings + july_paintings + august_paintings

theorem paintings_total : total_paintings = 18 :=
by {
  sorry
}

end paintings_total_l942_94244


namespace fraction_equality_x_eq_neg1_l942_94247

theorem fraction_equality_x_eq_neg1 (x : ℝ) (h : (5 + x) / (7 + x) = (3 + x) / (4 + x)) : x = -1 := by
  sorry

end fraction_equality_x_eq_neg1_l942_94247


namespace ratio_of_intercepts_l942_94296

theorem ratio_of_intercepts
  (u v : ℚ)
  (h1 : 2 = 5 * u)
  (h2 : 3 = -7 * v) :
  u / v = -14 / 15 :=
by
  sorry

end ratio_of_intercepts_l942_94296


namespace train_cross_time_l942_94286

noncomputable def train_length : ℝ := 317.5
noncomputable def train_speed_kph : ℝ := 153.3
noncomputable def convert_speed_to_mps (speed_kph : ℝ) : ℝ :=
  (speed_kph * 1000) / 3600

noncomputable def train_speed_mps : ℝ := convert_speed_to_mps train_speed_kph
noncomputable def time_to_cross_pole (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_cross_time :
  time_to_cross_pole train_length train_speed_mps = 7.456 :=
by 
  -- This is where the proof would go
  sorry

end train_cross_time_l942_94286


namespace function_pass_through_point_l942_94202

theorem function_pass_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), y = a^(x-2) - 1 ∧ (x, y) = (2, 0) := 
by
  use 2
  use 0
  sorry

end function_pass_through_point_l942_94202


namespace find_triplet_x_y_z_l942_94214

theorem find_triplet_x_y_z :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + 1 / (y + 1 / z : ℝ) = (10 : ℝ) / 7) ∧ (x = 1 ∧ y = 2 ∧ z = 3) :=
by
  sorry

end find_triplet_x_y_z_l942_94214


namespace seventieth_even_integer_l942_94224

theorem seventieth_even_integer : 2 * 70 = 140 :=
by
  sorry

end seventieth_even_integer_l942_94224


namespace percentage_of_carnations_is_44_percent_l942_94237

noncomputable def total_flowers : ℕ := sorry
def pink_percentage : ℚ := 2 / 5
def red_percentage : ℚ := 2 / 5
def yellow_percentage : ℚ := 1 / 5
def pink_roses_fraction : ℚ := 2 / 5
def red_carnations_fraction : ℚ := 1 / 2

theorem percentage_of_carnations_is_44_percent
  (F : ℕ)
  (h_pink : pink_percentage * F = 2 / 5 * F)
  (h_red : red_percentage * F = 2 / 5 * F)
  (h_yellow : yellow_percentage * F = 1 / 5 * F)
  (h_pink_roses : pink_roses_fraction * (pink_percentage * F) = 2 / 25 * F)
  (h_red_carnations : red_carnations_fraction * (red_percentage * F) = 1 / 5 * F) :
  ((6 / 25 * F + 5 / 25 * F) / F) * 100 = 44 := sorry

end percentage_of_carnations_is_44_percent_l942_94237


namespace rectangle_area_l942_94279

theorem rectangle_area (b l : ℕ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 112) : l * b = 588 := by
  sorry

end rectangle_area_l942_94279


namespace sum_first_six_terms_l942_94275

-- Define the conditions given in the problem
def a3 := 7
def a4 := 11
def a5 := 15

-- Define the common difference
def d := a4 - a3 -- 4

-- Define the first term
def a1 := a3 - 2 * d -- -1

-- Define the sum of the first six terms of the arithmetic sequence
def S6 := (6 / 2) * (2 * a1 + (6 - 1) * d) -- 54

-- The theorem we want to prove
theorem sum_first_six_terms : S6 = 54 := by
  sorry

end sum_first_six_terms_l942_94275


namespace find_y_l942_94263

theorem find_y : (12 : ℝ)^3 * (2 : ℝ)^4 / 432 = 5184 → (2 : ℝ) = 2 :=
by
  intro h
  sorry

end find_y_l942_94263


namespace amanda_days_needed_to_meet_goal_l942_94226

def total_tickets : ℕ := 80
def first_day_friends : ℕ := 5
def first_day_per_friend : ℕ := 4
def first_day_tickets : ℕ := first_day_friends * first_day_per_friend
def second_day_tickets : ℕ := 32
def third_day_tickets : ℕ := 28

theorem amanda_days_needed_to_meet_goal : 
  first_day_tickets + second_day_tickets + third_day_tickets = total_tickets → 
  3 = 3 :=
by
  intro h
  sorry

end amanda_days_needed_to_meet_goal_l942_94226


namespace acrobats_count_l942_94268

theorem acrobats_count (a g : ℕ) 
  (h1 : 2 * a + 4 * g = 32) 
  (h2 : a + g = 10) : 
  a = 4 := by
  -- Proof omitted
  sorry

end acrobats_count_l942_94268


namespace find_certain_number_l942_94204

theorem find_certain_number : 
  ∃ (certain_number : ℕ), 1038 * certain_number = 173 * 240 ∧ certain_number = 40 :=
by
  sorry

end find_certain_number_l942_94204


namespace sample_size_of_survey_l942_94298

theorem sample_size_of_survey (total_students : ℕ) (analyzed_students : ℕ)
  (h1 : total_students = 4000) (h2 : analyzed_students = 500) :
  analyzed_students = 500 :=
by
  sorry

end sample_size_of_survey_l942_94298


namespace xiaoli_estimate_larger_l942_94223

theorem xiaoli_estimate_larger (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) : 
  (1.1 * x) / (0.9 * y) > x / y :=
by
  sorry

end xiaoli_estimate_larger_l942_94223


namespace largest_digit_M_divisible_by_six_l942_94269

theorem largest_digit_M_divisible_by_six :
  (∃ M : ℕ, M ≤ 9 ∧ (45670 + M) % 6 = 0 ∧ ∀ m : ℕ, m ≤ M → (45670 + m) % 6 ≠ 0) :=
sorry

end largest_digit_M_divisible_by_six_l942_94269


namespace each_niece_gets_fifty_ice_cream_sandwiches_l942_94218

theorem each_niece_gets_fifty_ice_cream_sandwiches
  (total_sandwiches : ℕ)
  (total_nieces : ℕ)
  (h1 : total_sandwiches = 1857)
  (h2 : total_nieces = 37) :
  (total_sandwiches / total_nieces) = 50 :=
by
  sorry

end each_niece_gets_fifty_ice_cream_sandwiches_l942_94218


namespace annual_interest_rate_l942_94229

-- Definitions based on conditions
def initial_amount : ℝ := 1000
def spent_amount : ℝ := 440
def final_amount : ℝ := 624

-- The main theorem
theorem annual_interest_rate (x : ℝ) : 
  (initial_amount * (1 + x) - spent_amount) * (1 + x) = final_amount →
  x = 0.04 :=
by
  intro h
  sorry

end annual_interest_rate_l942_94229


namespace find_S30_l942_94201

variable {S : ℕ → ℝ} -- Assuming S is a function from natural numbers to real numbers

-- Arithmetic sequence is defined such that the sum of first n terms follows a specific format
def is_arithmetic_sequence (S : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, S (n + 1) - S n = d

-- Given conditions
axiom S10 : S 10 = 4
axiom S20 : S 20 = 20
axiom S_arithmetic : is_arithmetic_sequence S

-- The equivalent proof problem
theorem find_S30 : S 30 = 48 :=
by
  sorry

end find_S30_l942_94201


namespace solve_y_percentage_l942_94234

noncomputable def y_percentage (x y : ℝ) : ℝ :=
  100 * y / x

theorem solve_y_percentage (x y : ℝ) (h : 0.20 * (x - y) = 0.14 * (x + y)) :
  y_percentage x y = 300 / 17 :=
by
  sorry

end solve_y_percentage_l942_94234


namespace solve_for_x_l942_94265

-- Definitions of the conditions
def condition (x : ℚ) : Prop :=
  (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 3 * x - 18) / (x^2 - 2 * x - 24)

-- Statement of the theorem
theorem solve_for_x (x : ℚ) (h : condition x) : x = -5 / 4 :=
by 
  sorry

end solve_for_x_l942_94265


namespace set_B_listing_method_l942_94233

variable (A : Set ℕ) (B : Set ℕ)

theorem set_B_listing_method (hA : A = {1, 2, 3}) (hB : B = {x | x ∈ A}) :
  B = {1, 2, 3} :=
  by
    sorry

end set_B_listing_method_l942_94233


namespace dima_and_serezha_meet_time_l942_94255

-- Define the conditions and the main theorem to be proven.
theorem dima_and_serezha_meet_time :
  let dima_run_time := 15 / 60.0 -- Dima runs for 15 minutes
  let dima_run_speed := 6.0 -- Dima's running speed is 6 km/h
  let serezha_boat_speed := 20.0 -- Serezha's boat speed is 20 km/h
  let serezha_boat_time := 30 / 60.0 -- Serezha's boat time is 30 minutes
  let common_run_speed := 6.0 -- Both run at 6 km/h towards each other
  let distance_to_meet := dima_run_speed * dima_run_time -- Distance Dima runs along the shore
  let total_time := distance_to_meet / (common_run_speed + common_run_speed) -- Time until they meet after parting
  total_time = 7.5 / 60.0 := -- 7.5 minutes converted to hours
sorry

end dima_and_serezha_meet_time_l942_94255


namespace triangle_angle_not_greater_than_60_l942_94240

theorem triangle_angle_not_greater_than_60 (A B C : ℝ) (h1 : A + B + C = 180) :
  A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 :=
sorry -- proof by contradiction to be implemented here

end triangle_angle_not_greater_than_60_l942_94240


namespace coalsBurnedEveryTwentyMinutes_l942_94239

-- Definitions based on the conditions
def totalGrillingTime : Int := 240
def coalsPerBag : Int := 60
def numberOfBags : Int := 3
def grillingInterval : Int := 20

-- Derived definitions based on conditions
def totalCoals : Int := numberOfBags * coalsPerBag
def numberOfIntervals : Int := totalGrillingTime / grillingInterval

-- The Lean theorem we want to prove
theorem coalsBurnedEveryTwentyMinutes : (totalCoals / numberOfIntervals) = 15 := by
  sorry

end coalsBurnedEveryTwentyMinutes_l942_94239


namespace calculate_expression_l942_94249

theorem calculate_expression : 4 * 6 * 8 + 24 / 4 - 10 = 188 := by
  sorry

end calculate_expression_l942_94249


namespace ratio_of_pens_to_notebooks_is_5_to_4_l942_94227

theorem ratio_of_pens_to_notebooks_is_5_to_4 (P N : ℕ) (hP : P = 50) (hN : N = 40) :
  (P / Nat.gcd P N) = 5 ∧ (N / Nat.gcd P N) = 4 :=
by
  -- Proof goes here
  sorry

end ratio_of_pens_to_notebooks_is_5_to_4_l942_94227


namespace arccos_one_eq_zero_l942_94232

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l942_94232


namespace weight_of_new_man_l942_94256

theorem weight_of_new_man (avg_increase : ℝ) (num_oarsmen : ℕ) (old_weight : ℝ) (weight_increase : ℝ) 
  (h1 : avg_increase = 1.8) (h2 : num_oarsmen = 10) (h3 : old_weight = 53) (h4 : weight_increase = num_oarsmen * avg_increase) :
  ∃ W : ℝ, W = old_weight + weight_increase :=
by
  sorry

end weight_of_new_man_l942_94256


namespace octal_subtraction_correct_l942_94264

-- Define the octal numbers
def octal752 : ℕ := 7 * 8^2 + 5 * 8^1 + 2 * 8^0
def octal364 : ℕ := 3 * 8^2 + 6 * 8^1 + 4 * 8^0
def octal376 : ℕ := 3 * 8^2 + 7 * 8^1 + 6 * 8^0

-- Prove the octal number subtraction
theorem octal_subtraction_correct : octal752 - octal364 = octal376 := by
  sorry

end octal_subtraction_correct_l942_94264


namespace f_5_eq_25sqrt5_l942_94290

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom continuous_f : Continuous f
axiom functional_eq : ∀ x y : ℝ, f (x + y) = f x * f y
axiom f_2 : f 2 = 5

theorem f_5_eq_25sqrt5 : f 5 = 25 * Real.sqrt 5 := by
  sorry

end f_5_eq_25sqrt5_l942_94290


namespace man_l942_94235

-- Lean 4 statement
theorem man's_speed_against_stream (speed_with_stream : ℝ) (speed_still_water : ℝ) 
(h1 : speed_with_stream = 16) (h2 : speed_still_water = 4) : 
  |speed_still_water - (speed_with_stream - speed_still_water)| = 8 :=
by
  -- Dummy proof since only statement is required
  sorry

end man_l942_94235


namespace last_digit_of_2_pow_2018_l942_94273

-- Definition of the cyclic pattern
def last_digit_cycle : List ℕ := [2, 4, 8, 6]

-- Function to find the last digit of 2^n using the cycle
def last_digit_of_power_of_two (n : ℕ) : ℕ :=
  last_digit_cycle.get! ((n % 4) - 1)

-- Main theorem statement
theorem last_digit_of_2_pow_2018 : last_digit_of_power_of_two 2018 = 4 :=
by
  -- The proof part is omitted
  sorry

end last_digit_of_2_pow_2018_l942_94273


namespace banker_discount_calculation_l942_94245

-- Define the future value function with given interest rates and periods.
def face_value (PV : ℝ) : ℝ :=
  (PV * (1 + 0.10) ^ 4) * (1 + 0.12) ^ 4

-- Define the true discount as the difference between the future value and the present value.
def true_discount (PV : ℝ) : ℝ :=
  face_value PV - PV

-- Given conditions
def banker_gain : ℝ := 900

-- Define the banker's discount.
def banker_discount (PV : ℝ) : ℝ :=
  banker_gain + true_discount PV

-- The proof statement to prove the relationship.
theorem banker_discount_calculation (PV : ℝ) :
  banker_discount PV = banker_gain + (face_value PV - PV) := by
  sorry

end banker_discount_calculation_l942_94245


namespace students_neither_play_l942_94287

theorem students_neither_play (total_students football_players tennis_players both_players neither_players : ℕ)
  (h1 : total_students = 40)
  (h2 : football_players = 26)
  (h3 : tennis_players = 20)
  (h4 : both_players = 17)
  (h5 : neither_players = total_students - (football_players + tennis_players - both_players)) :
  neither_players = 11 :=
by
  sorry

end students_neither_play_l942_94287


namespace find_integer_pairs_l942_94297

theorem find_integer_pairs :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (m n : ℤ), (m, n) ∈ S ↔ mn ≤ 0 ∧ m^3 + n^3 - 37 * m * n = 343) ∧ S.card = 9 :=
sorry

end find_integer_pairs_l942_94297


namespace expand_polynomial_product_l942_94295

variable (x : ℝ)

def P (x : ℝ) : ℝ := 5 * x ^ 2 + 3 * x - 4
def Q (x : ℝ) : ℝ := 6 * x ^ 3 + 2 * x ^ 2 - x + 7

theorem expand_polynomial_product :
  (P x) * (Q x) = 30 * x ^ 5 + 28 * x ^ 4 - 23 * x ^ 3 + 24 * x ^ 2 + 25 * x - 28 :=
by
  sorry

end expand_polynomial_product_l942_94295


namespace f_2202_minus_f_2022_l942_94203

-- Definitions and conditions
def f : ℕ+ → ℕ+ := sorry -- The exact function is provided through conditions and will be proven property-wise.

axiom f_increasing {a b : ℕ+} : a < b → f a < f b
axiom f_range (n : ℕ+) : ∃ m : ℕ+, f n = ⟨m, sorry⟩ -- ensuring f maps to ℕ+
axiom f_property (n : ℕ+) : f (f n) = 3 * n

-- Prove the statement
theorem f_2202_minus_f_2022 : f 2202 - f 2022 = 1638 :=
by sorry

end f_2202_minus_f_2022_l942_94203


namespace remainder_17_pow_45_div_5_l942_94213

theorem remainder_17_pow_45_div_5 : (17 ^ 45) % 5 = 2 :=
by
  -- proof goes here
  sorry

end remainder_17_pow_45_div_5_l942_94213


namespace find_b2_a2_a1_l942_94285

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def geometric_sequence (b : ℕ → ℝ) : Prop :=
∀ n : ℕ, b (n + 1) / b n = b 1 / b 0

theorem find_b2_a2_a1 (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a1 : a 0 = a₁) (h_a2 : a 2 = a₂)
  (h_b2 : b 2 = b₂) :
  b₂ * (a₂ - a₁) = 6 ∨ b₂ * (a₂ - a₁) = -6 :=
by
  sorry

end find_b2_a2_a1_l942_94285


namespace sufficient_but_not_necessary_l942_94262

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a > 2 ∧ b > 2) → (a + b > 4 ∧ a * b > 4) ∧ ¬((a + b > 4 ∧ a * b > 4) → (a > 2 ∧ b > 2)) :=
by
  sorry

end sufficient_but_not_necessary_l942_94262


namespace calculate_uphill_distance_l942_94211

noncomputable def uphill_speed : ℝ := 30
noncomputable def downhill_speed : ℝ := 40
noncomputable def downhill_distance : ℝ := 50
noncomputable def average_speed : ℝ := 32.73

theorem calculate_uphill_distance : ∃ d : ℝ, d = 99.86 ∧ 
  32.73 = (d + downhill_distance) / (d / uphill_speed + downhill_distance / downhill_speed) :=
by
  sorry

end calculate_uphill_distance_l942_94211


namespace circle_equation_l942_94280

-- Define the circle's equation as a predicate
def is_circle (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Given conditions, defining the known center and passing point
def center_x : ℝ := 2
def center_y : ℝ := -3
def point_M_x : ℝ := -1
def point_M_y : ℝ := 1

-- Prove that the circle with the given conditions has the correct equation
theorem circle_equation :
  is_circle x y center_x center_y 5 ↔ 
  ∀ x y : ℝ, (x - center_x)^2 + (y + center_y)^2 = 25 := sorry

end circle_equation_l942_94280


namespace triangle_area_30_l942_94274

theorem triangle_area_30 (h : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 5 ∧ c = 13 ∧ b > 0) : 
  ∃ area : ℝ, area = 1 / 2 * 5 * (b : ℝ) ∧ area = 30 :=
by
  sorry

end triangle_area_30_l942_94274


namespace sum_of_squares_of_consecutive_even_integers_l942_94200

theorem sum_of_squares_of_consecutive_even_integers (n : ℤ) (h : (2 * n - 2) * (2 * n) * (2 * n + 2) = 12 * ((2 * n - 2) + (2 * n) + (2 * n + 2))) :
  (2 * n - 2) ^ 2 + (2 * n) ^ 2 + (2 * n + 2) ^ 2 = 440 :=
by
  sorry

end sum_of_squares_of_consecutive_even_integers_l942_94200


namespace lcm_gcf_ratio_280_450_l942_94206

open Nat

theorem lcm_gcf_ratio_280_450 :
  let a := 280
  let b := 450
  lcm a b / gcd a b = 1260 :=
by
  let a := 280
  let b := 450
  sorry

end lcm_gcf_ratio_280_450_l942_94206


namespace trapezoid_sides_and_height_l942_94243

def trapezoid_base_height (a h A: ℝ) :=
  (h = (2 * a + 3) / 2) ∧
  (A = a^2 + 3 * a + 9 / 4) ∧
  (A = 2 * a^2 - 7.75)

theorem trapezoid_sides_and_height :
  ∃ (a b h : ℝ), (b = a + 3) ∧
  trapezoid_base_height a h 7.75 ∧
  a = 5 ∧ b = 8 ∧ h = 6.5 :=
by
  sorry

end trapezoid_sides_and_height_l942_94243


namespace find_s_l942_94270

theorem find_s (s t : ℚ) (h1 : 8 * s + 6 * t = 120) (h2 : s = t - 3) : s = 51 / 7 := by
  sorry

end find_s_l942_94270


namespace regression_passes_through_none_l942_94293

theorem regression_passes_through_none (b a x y : ℝ) (h₀ : (0, 0) ≠ (0*b + a, 0))
                                     (h₁ : (x, 0) ≠ (x*b + a, 0))
                                     (h₂ : (x, y) ≠ (x*b + a, y)) : 
                                     ¬ ((0, 0) = (0*b + a, 0) ∨ (x, 0) = (x*b + a, 0) ∨ (x, y) = (x*b + a, y)) :=
by sorry

end regression_passes_through_none_l942_94293


namespace correct_expression_for_representatives_l942_94238

/-- Definition for the number of representatives y given the class size x
    and the conditions that follow. -/
def elect_representatives (x : ℕ) : ℕ :=
  if 6 < x % 10 then (x + 3) / 10 else x / 10

theorem correct_expression_for_representatives (x : ℕ) :
  elect_representatives x = (x + 3) / 10 :=
by
  sorry

end correct_expression_for_representatives_l942_94238


namespace smallest_whole_number_above_perimeter_triangle_l942_94220

theorem smallest_whole_number_above_perimeter_triangle (s : ℕ) (h1 : 12 < s) (h2 : s < 26) :
  53 = Nat.ceil ((7 + 19 + s : ℕ) / 1) := by
  sorry

end smallest_whole_number_above_perimeter_triangle_l942_94220


namespace problems_per_hour_l942_94236

def num_math_problems : ℝ := 17.0
def num_spelling_problems : ℝ := 15.0
def total_hours : ℝ := 4.0

theorem problems_per_hour :
  (num_math_problems + num_spelling_problems) / total_hours = 8.0 := by
  sorry

end problems_per_hour_l942_94236


namespace expression_for_x_expression_for_y_l942_94250

variables {A B C : ℝ}

-- Conditions: A, B, and C are positive numbers with A > B > C > 0
axiom h1 : A > 0
axiom h2 : B > 0
axiom h3 : C > 0
axiom h4 : A > B
axiom h5 : B > C

-- A is x% greater than B
variables {x : ℝ}
axiom h6 : A = (1 + x / 100) * B

-- A is y% greater than C
variables {y : ℝ}
axiom h7 : A = (1 + y / 100) * C

-- Proving the expressions for x and y
theorem expression_for_x : x = 100 * ((A - B) / B) :=
sorry

theorem expression_for_y : y = 100 * ((A - C) / C) :=
sorry

end expression_for_x_expression_for_y_l942_94250


namespace sqrt_of_square_eq_seven_l942_94291

theorem sqrt_of_square_eq_seven (x : ℝ) (h : x^2 = 7) : x = Real.sqrt 7 ∨ x = -Real.sqrt 7 :=
sorry

end sqrt_of_square_eq_seven_l942_94291


namespace kayak_rental_cost_l942_94225

theorem kayak_rental_cost
    (canoe_cost_per_day : ℕ := 14)
    (total_revenue : ℕ := 288)
    (canoe_kayak_ratio : ℕ × ℕ := (3, 2))
    (canoe_kayak_difference : ℕ := 4)
    (number_of_kayaks : ℕ := 8)
    (number_of_canoes : ℕ := number_of_kayaks + canoe_kayak_difference)
    (canoe_revenue : ℕ := number_of_canoes * canoe_cost_per_day) :
    number_of_kayaks * kayak_cost_per_day = total_revenue - canoe_revenue →
    kayak_cost_per_day = 15 := 
by
  sorry

end kayak_rental_cost_l942_94225


namespace mersenne_primes_less_than_1000_l942_94230

open Nat

-- Definitions and Conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_mersenne_prime (p : ℕ) : Prop := ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

-- Theorem Statement
theorem mersenne_primes_less_than_1000 : {p : ℕ | is_mersenne_prime p ∧ p < 1000} = {3, 7, 31, 127} :=
by
  sorry

end mersenne_primes_less_than_1000_l942_94230
