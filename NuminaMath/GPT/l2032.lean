import Mathlib

namespace intersection_point_of_lines_l2032_203298

theorem intersection_point_of_lines :
  ∃ (x y : ℝ), x + 2 * y - 4 = 0 ∧ 2 * x - y + 2 = 0 ∧ (x, y) = (0, 2) :=
by
  sorry

end intersection_point_of_lines_l2032_203298


namespace f_1993_of_3_l2032_203252

def f (x : ℚ) := (1 + x) / (1 - 3 * x)

def f_n (x : ℚ) : ℕ → ℚ
| 0 => x
| (n + 1) => f (f_n x n)

theorem f_1993_of_3 :
  f_n 3 1993 = 1 / 5 :=
sorry

end f_1993_of_3_l2032_203252


namespace sin_330_eq_neg_half_l2032_203260

-- Define conditions as hypotheses in Lean
def angle_330 (θ : ℝ) : Prop := θ = 330
def angle_transform (θ : ℝ) : Prop := θ = 360 - 30
def sin_pos (θ : ℝ) : Prop := Real.sin θ = 1 / 2
def sin_neg_in_4th_quadrant (θ : ℝ) : Prop := θ = 330 -> Real.sin θ < 0

-- The main theorem statement
theorem sin_330_eq_neg_half : ∀ θ : ℝ, angle_330 θ → angle_transform θ → sin_pos 30 → sin_neg_in_4th_quadrant θ → Real.sin θ = -1 / 2 := by
  intro θ h1 h2 h3 h4
  sorry

end sin_330_eq_neg_half_l2032_203260


namespace floor_ineq_l2032_203277

theorem floor_ineq (x y : ℝ) : 
  Int.floor (2 * x) + Int.floor (2 * y) ≥ Int.floor x + Int.floor y + Int.floor (x + y) := 
sorry

end floor_ineq_l2032_203277


namespace glass_volume_l2032_203274

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l2032_203274


namespace xyz_final_stock_price_l2032_203234

def initial_stock_price : ℝ := 120
def first_year_increase_rate : ℝ := 0.80
def second_year_decrease_rate : ℝ := 0.30

def final_stock_price_after_two_years : ℝ :=
  (initial_stock_price * (1 + first_year_increase_rate)) * (1 - second_year_decrease_rate)

theorem xyz_final_stock_price :
  final_stock_price_after_two_years = 151.2 := by
  sorry

end xyz_final_stock_price_l2032_203234


namespace wall_length_proof_l2032_203241

-- Define the conditions from the problem
def wall_height : ℝ := 100 -- Height in cm
def wall_thickness : ℝ := 5 -- Thickness in cm
def brick_length : ℝ := 25 -- Brick length in cm
def brick_width : ℝ := 11 -- Brick width in cm
def brick_height : ℝ := 6 -- Brick height in cm
def number_of_bricks : ℝ := 242.42424242424244

-- Calculate the volume of one brick
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Calculate the total volume of the bricks
def total_brick_volume : ℝ := brick_volume * number_of_bricks

-- Define the proof problem
theorem wall_length_proof : total_brick_volume = wall_height * wall_thickness * 800 :=
sorry

end wall_length_proof_l2032_203241


namespace distinct_parenthesizations_of_3_3_3_3_l2032_203206

theorem distinct_parenthesizations_of_3_3_3_3 : 
  ∃ (v1 v2 v3 v4 v5 : ℕ), 
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v1 ≠ v5 ∧ 
    v2 ≠ v3 ∧ v2 ≠ v4 ∧ v2 ≠ v5 ∧ 
    v3 ≠ v4 ∧ v3 ≠ v5 ∧ 
    v4 ≠ v5 ∧ 
    v1 = 3 ^ (3 ^ (3 ^ 3)) ∧ 
    v2 = 3 ^ ((3 ^ 3) ^ 3) ∧ 
    v3 = (3 ^ 3) ^ (3 ^ 3) ∧ 
    v4 = ((3 ^ 3) ^ 3) ^ 3 ∧ 
    v5 = 3 ^ (27 ^ 27) :=
  sorry

end distinct_parenthesizations_of_3_3_3_3_l2032_203206


namespace function_pair_solution_l2032_203254

-- Define the conditions for f and g
variables (f g : ℝ → ℝ)

-- Define the main hypothesis
def main_hypothesis : Prop := 
∀ (x y : ℝ), 
  x ≠ 0 → y ≠ 0 → 
  f (x + y) = g (1/x + 1/y) * (x * y) ^ 2008

-- The theorem that proves f and g are of the given form
theorem function_pair_solution (c : ℝ) (h : main_hypothesis f g) : 
  (∀ x, f x = c * x ^ 2008) ∧ 
  (∀ x, g x = c * x ^ 2008) :=
sorry

end function_pair_solution_l2032_203254


namespace frame_painting_ratio_l2032_203246

theorem frame_painting_ratio :
  ∃ (x : ℝ), (20 + 2 * x) * (30 + 6 * x) = 1800 → 1 = 2 * (20 + 2 * x) / (30 + 6 * x) :=
by
  sorry

end frame_painting_ratio_l2032_203246


namespace value_of_expression_l2032_203242

theorem value_of_expression (a : ℚ) (h : a = 1/3) : (3 * a⁻¹ + a⁻¹ / 3) / (2 * a) = 15 := by
  sorry

end value_of_expression_l2032_203242


namespace least_common_multiple_l2032_203285

theorem least_common_multiple (x : ℕ) (hx : x > 0) : 
  lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (9 * x))) = 1 / (36 * x) :=
by
  sorry

end least_common_multiple_l2032_203285


namespace length_of_platform_is_280_l2032_203250

-- Add conditions for speed, times and conversions
def speed_kmph : ℕ := 72
def time_platform : ℕ := 30
def time_man : ℕ := 16

-- Conversion from km/h to m/s
def speed_mps : ℤ := speed_kmph * 1000 / 3600

-- The length of the train when it crosses the man
def length_of_train : ℤ := speed_mps * time_man

-- The length of the platform
def length_of_platform : ℤ := (speed_mps * time_platform) - length_of_train

theorem length_of_platform_is_280 :
  length_of_platform = 280 := by
  sorry

end length_of_platform_is_280_l2032_203250


namespace trigonometric_identity_l2032_203245

variable {θ u : ℝ} {n : ℤ}

-- Given condition
def cos_condition (θ u : ℝ) : Prop := 2 * Real.cos θ = u + (1 / u)

-- Theorem to prove
theorem trigonometric_identity (h : cos_condition θ u) : 2 * Real.cos (n * θ) = u^n + (1 / u^n) :=
sorry

end trigonometric_identity_l2032_203245


namespace total_flowers_l2032_203284

def tulips : ℕ := 3
def carnations : ℕ := 4

theorem total_flowers : tulips + carnations = 7 := by
  sorry

end total_flowers_l2032_203284


namespace find_y_l2032_203228

theorem find_y (x y : ℕ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 119) : y = 1 :=
sorry

end find_y_l2032_203228


namespace sum_of_series_l2032_203272

def series_sum : ℕ := 2 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))))

theorem sum_of_series : series_sum = 2730 := by
  -- Expansion: 2 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4))))) = 2 + 2 * 4 + 2 * 4^2 + 2 * 4^3 + 2 * 4^4 + 2 * 4^5 
  -- Geometric series sum formula application: S = 2 + 2*4 + 2*4^2 + 2*4^3 + 2*4^4 + 2*4^5 = 2730
  sorry

end sum_of_series_l2032_203272


namespace total_books_arithmetic_sequence_l2032_203290

theorem total_books_arithmetic_sequence :
  ∃ (n : ℕ) (a₁ a₂ aₙ d S : ℤ), 
    n = 11 ∧
    a₁ = 32 ∧
    a₂ = 29 ∧
    aₙ = 2 ∧
    d = -3 ∧
    S = (n * (a₁ + aₙ)) / 2 ∧
    S = 187 :=
by sorry

end total_books_arithmetic_sequence_l2032_203290


namespace martian_angle_conversion_l2032_203273

-- Defines the full circle measurements
def full_circle_clerts : ℕ := 600
def full_circle_degrees : ℕ := 360
def angle_degrees : ℕ := 60

-- The main statement to prove
theorem martian_angle_conversion : 
    (full_circle_clerts * angle_degrees) / full_circle_degrees = 100 :=
by
  sorry  

end martian_angle_conversion_l2032_203273


namespace sin_four_thirds_pi_l2032_203255

theorem sin_four_thirds_pi : Real.sin (4 / 3 * Real.pi) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_four_thirds_pi_l2032_203255


namespace angle_ZAX_pentagon_triangle_common_vertex_l2032_203295

theorem angle_ZAX_pentagon_triangle_common_vertex :
  let n_pentagon := 5
  let n_triangle := 3
  let internal_angle_pentagon := (n_pentagon - 2) * 180 / n_pentagon
  let internal_angle_triangle := 60
  let common_angle_A := 360 - (internal_angle_pentagon + internal_angle_pentagon + internal_angle_triangle + internal_angle_triangle) / 2
  common_angle_A = 192 := by
  let n_pentagon := 5
  let n_triangle := 3
  let internal_angle_pentagon := (n_pentagon - 2) * 180 / n_pentagon
  let internal_angle_triangle := 60
  let common_angle_A := 360 - (internal_angle_pentagon + internal_angle_pentagon + internal_angle_triangle + internal_angle_triangle) / 2
  sorry

end angle_ZAX_pentagon_triangle_common_vertex_l2032_203295


namespace seeds_sum_l2032_203268

def Bom_seeds : ℕ := 300

def Gwi_seeds : ℕ := Bom_seeds + 40

def Yeon_seeds : ℕ := 3 * Gwi_seeds

def total_seeds : ℕ := Bom_seeds + Gwi_seeds + Yeon_seeds

theorem seeds_sum : total_seeds = 1660 := by
  sorry

end seeds_sum_l2032_203268


namespace eggs_broken_l2032_203280

theorem eggs_broken (brown_eggs white_eggs total_pre total_post broken_eggs : ℕ) 
  (h1 : brown_eggs = 10)
  (h2 : white_eggs = 3 * brown_eggs)
  (h3 : total_pre = brown_eggs + white_eggs)
  (h4 : total_post = 20)
  (h5 : broken_eggs = total_pre - total_post) : broken_eggs = 20 :=
by
  sorry

end eggs_broken_l2032_203280


namespace cherries_in_mix_l2032_203265

theorem cherries_in_mix (total_fruit : ℕ) (blueberries : ℕ) (raspberries : ℕ) (cherries : ℕ) 
  (H1 : total_fruit = 300)
  (H2: raspberries = 3 * blueberries)
  (H3: cherries = 5 * blueberries)
  (H4: total_fruit = blueberries + raspberries + cherries) : cherries = 167 :=
by
  sorry

end cherries_in_mix_l2032_203265


namespace quadratic_discriminant_l2032_203207

variable {a b c : ℝ}
variable (h₁ : a ≠ 0)
variable (h₂ : (b - 1)^2 - 4 * a * (c + 2) = 0)
variable (h₃ : (b + 1/2)^2 - 4 * a * (c - 1) = 0)

theorem quadratic_discriminant : b^2 - 4 * a * c = -1 / 2 := 
by
  have h₁' : (b - 1)^2 - 4 * a * (c + 2) = 0 := h₂
  have h₂' : (b + 1/2)^2 - 4 * a * (c - 1) = 0 := h₃
  sorry

end quadratic_discriminant_l2032_203207


namespace poly_coeff_sum_l2032_203244

theorem poly_coeff_sum :
  (∃ a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℤ,
  (∀ x : ℤ, ((x^2 + 1) * (x - 2)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_10 * x^10 + a_11 * x^11))
  ∧ a_0 = -512) →
  (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 510) :=
by
  sorry

end poly_coeff_sum_l2032_203244


namespace smallest_n_proof_l2032_203218

-- Given conditions and the problem statement in Lean 4
noncomputable def smallest_n : ℕ := 11

theorem smallest_n_proof :
  ∃ (x y z : ℕ), (x > 0 ∧ y > 0 ∧ z > 0) ∧ (smallest_n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 11) :=
sorry

end smallest_n_proof_l2032_203218


namespace greatest_possible_length_l2032_203278

theorem greatest_possible_length :
  ∃ (g : ℕ), g = Nat.gcd 700 (Nat.gcd 385 1295) ∧ g = 35 :=
by
  sorry

end greatest_possible_length_l2032_203278


namespace k_is_perfect_square_l2032_203216

theorem k_is_perfect_square (m n : ℤ) (hm : m > 0) (hn : n > 0) (k := ((m + n) ^ 2) / (4 * m * (m - n) ^ 2 + 4)) :
  ∃ (a : ℤ), k = a ^ 2 := by
  sorry

end k_is_perfect_square_l2032_203216


namespace range_of_a_l2032_203230

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 1 → |x| < a) → 1 ≤ a :=
by 
  sorry

end range_of_a_l2032_203230


namespace max_value_x2y_l2032_203225

theorem max_value_x2y : 
  ∃ (x y : ℕ), 
    7 * x + 4 * y = 140 ∧
    (∀ (x' y' : ℕ),
       7 * x' + 4 * y' = 140 → 
       x' ^ 2 * y' ≤ x ^ 2 * y) ∧
    x ^ 2 * y = 2016 :=
by {
  sorry
}

end max_value_x2y_l2032_203225


namespace problem1_problem2_l2032_203257

-- problem (1): Prove that if a = 1 and (p ∨ q) is true, then the range of x is 1 < x < 3
def p (a x : ℝ) : Prop := x ^ 2 - 4 * a * x + 3 * a ^ 2 < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem problem1 (x : ℝ) (a : ℝ) (h₁ : a = 1) (h₂ : p a x ∨ q x) : 
    1 < x ∧ x < 3 :=
sorry

-- problem (2): Prove that if p is a necessary but not sufficient condition for q,
-- then the range of a is 1 ≤ a ≤ 2
theorem problem2 (a : ℝ) :
  (∀ x : ℝ, q x → p a x) ∧ (∃ x : ℝ, p a x ∧ ¬q x) → 
  1 ≤ a ∧ a ≤ 2 := 
sorry

end problem1_problem2_l2032_203257


namespace arithmetic_sequence_fifth_term_l2032_203201

theorem arithmetic_sequence_fifth_term :
  ∀ (a : ℕ → ℕ), (∀ n, a n.succ = a n + 2) → a 1 = 2 → a 5 = 10 :=
by
  intros a h1 h2
  sorry

end arithmetic_sequence_fifth_term_l2032_203201


namespace problem_l2032_203221

open Real 

noncomputable def sqrt_log_a (a : ℝ) : ℝ := sqrt (log a / log 10)
noncomputable def sqrt_log_b (b : ℝ) : ℝ := sqrt (log b / log 10)

theorem problem (a b : ℝ) 
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (condition1 : sqrt_log_a a + 2 * sqrt_log_b b + 2 * log (sqrt a) / log 10 + log (sqrt b) / log 10 = 150)
  (int_sqrt_log_a : ∃ (m : ℕ), sqrt_log_a a = m)
  (int_sqrt_log_b : ∃ (n : ℕ), sqrt_log_b b = n)
  (condition2 : a^2 * b = 10^81) :
  a * b = 10^85 :=
sorry

end problem_l2032_203221


namespace x_cubed_plus_y_cubed_l2032_203275

theorem x_cubed_plus_y_cubed:
  ∀ (x y : ℝ), (x * (x ^ 4 + y ^ 4) = y ^ 5) → (x ^ 2 * (x + y) ≠ y ^ 3) → (x ^ 3 + y ^ 3 = 1) :=
by
  intros x y h1 h2
  sorry

end x_cubed_plus_y_cubed_l2032_203275


namespace sean_whistles_l2032_203240

def charles_whistles : ℕ := 13
def extra_whistles : ℕ := 32

theorem sean_whistles : charles_whistles + extra_whistles = 45 := by
  sorry

end sean_whistles_l2032_203240


namespace average_speed_of_car_l2032_203220

noncomputable def average_speed (total_distance total_time : ℕ) : ℕ :=
  total_distance / total_time

theorem average_speed_of_car :
  let d1 := 80
  let d2 := 40
  let d3 := 60
  let d4 := 50
  let d5 := 90
  let d6 := 100
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  average_speed total_distance total_time = 70 :=
by
  let d1 := 80
  let d2 := 40
  let d3 := 60
  let d4 := 50
  let d5 := 90
  let d6 := 100
  let total_distance := d1 + d2 + d3 + d4 + d5 + d6
  let total_time := 6
  exact sorry

end average_speed_of_car_l2032_203220


namespace circle_radius_value_l2032_203214

theorem circle_radius_value (k : ℝ) :
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + k = 0 ↔ (x - 4)^2 + (y + 5)^2 = 25) → k = 16 :=
by
  sorry

end circle_radius_value_l2032_203214


namespace square_of_1023_l2032_203232

theorem square_of_1023 : 1023^2 = 1045529 := by
  sorry

end square_of_1023_l2032_203232


namespace parabola_fixed_point_l2032_203261

theorem parabola_fixed_point (t : ℝ) : ∃ y, y = 4 * 3^2 + 2 * t * 3 - 3 * t ∧ y = 36 :=
by
  exists 36
  sorry

end parabola_fixed_point_l2032_203261


namespace factorize_expression_l2032_203262

variable (x y : ℝ)

theorem factorize_expression :
  4 * (x - y + 1) + y * (y - 2 * x) = (y - 2) * (y - 2 - 2 * x) :=
by 
  sorry

end factorize_expression_l2032_203262


namespace number_of_routes_A_to_B_l2032_203239

theorem number_of_routes_A_to_B :
  (∃ f : ℕ × ℕ → ℕ,
  (∀ n m, f (n + 1, m) = f (n, m) + f (n + 1, m - 1)) ∧
  f (0, 0) = 1 ∧ 
  (∀ i, f (i, 0) = 1) ∧ 
  (∀ j, f (0, j) = 1) ∧ 
  f (3, 5) = 23) :=
sorry

end number_of_routes_A_to_B_l2032_203239


namespace max_volume_range_of_a_x1_x2_inequality_l2032_203223

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

noncomputable def g (a x : ℝ) : ℝ := (Real.exp (a * x^2) - Real.exp 1 * x + a * x^2 - 1) / x

theorem max_volume (x : ℝ) (hx : 1 < x) :
  ∃ V : ℝ, V = (Real.pi / 3) * ((Real.log x)^2 / x) ∧ V = (4 * Real.pi / (3 * (Real.exp 2)^2)) :=
sorry

theorem range_of_a (x1 x2 a : ℝ) (hx1 : 1 < x1) (hx2 : 1 < x2) (hx12 : x1 < x2)
  (h_eq : ∀ x > 1, f x = g a x) :
  0 < a ∧ a < (1/2) * (Real.exp 1) :=
sorry

theorem x1_x2_inequality (x1 x2 a : ℝ) (hx1 : 1 < x1) (hx2 : 1 < x2) (hx12 : x1 < x2)
  (h_eq : ∀ x > 1, f x = g a x) :
  x1^2 + x2^2 > 2 / Real.exp 1 :=
sorry

end max_volume_range_of_a_x1_x2_inequality_l2032_203223


namespace find_m_value_l2032_203217

variable (m : ℝ)

theorem find_m_value (h1 : m^2 - 3 * m = 4)
                     (h2 : m^2 = 5 * m + 6) : m = -1 :=
sorry

end find_m_value_l2032_203217


namespace muirheadable_decreasing_columns_iff_l2032_203235

def isMuirheadable (n : ℕ) (grid : List (List ℕ)) : Prop :=
  -- Placeholder definition; the actual definition should specify the conditions
  sorry

theorem muirheadable_decreasing_columns_iff (n : ℕ) (h : n > 0) :
  (∃ grid : List (List ℕ), isMuirheadable n grid) ↔ n ≠ 3 :=
by 
  sorry

end muirheadable_decreasing_columns_iff_l2032_203235


namespace Nell_initial_cards_l2032_203203

theorem Nell_initial_cards 
  (cards_given : ℕ)
  (cards_left : ℕ)
  (cards_given_eq : cards_given = 301)
  (cards_left_eq : cards_left = 154) :
  cards_given + cards_left = 455 := by
sorry

end Nell_initial_cards_l2032_203203


namespace find_m_l2032_203227

theorem find_m (m : ℝ) (A B : Set ℝ) (hA : A = {-1, 3, 2*m - 1}) (hB: B = {3, m^2}) (h_subset: B ⊆ A) : m = 1 :=
by
  sorry

end find_m_l2032_203227


namespace zoe_distance_more_than_leo_l2032_203294

theorem zoe_distance_more_than_leo (d t s : ℝ)
  (maria_driving_time : ℝ := t + 2)
  (maria_speed : ℝ := s + 15)
  (zoe_driving_time : ℝ := t + 3)
  (zoe_speed : ℝ := s + 20)
  (leo_distance : ℝ := s * t)
  (maria_distance : ℝ := (s + 15) * (t + 2))
  (zoe_distance : ℝ := (s + 20) * (t + 3))
  (maria_leo_distance_diff : ℝ := 110)
  (h1 : maria_distance = leo_distance + maria_leo_distance_diff)
  : zoe_distance - leo_distance = 180 :=
by
  sorry

end zoe_distance_more_than_leo_l2032_203294


namespace parameterize_circle_l2032_203259

noncomputable def parametrization (t : ℝ) : ℝ × ℝ :=
  ( (t^2 - 1) / (t^2 + 1), (-2 * t) / (t^2 + 1) )

theorem parameterize_circle (t : ℝ) : 
  let x := (t^2 - 1) / (t^2 + 1) 
  let y := (-2 * t) / (t^2 + 1) 
  (x^2 + y^2) = 1 :=
by 
  let x := (t^2 - 1) / (t^2 + 1) 
  let y := (-2 * t) / (t^2 + 1) 
  sorry

end parameterize_circle_l2032_203259


namespace mobot_coloring_six_colorings_l2032_203292

theorem mobot_coloring_six_colorings (n m : ℕ) (h : n ≥ 3 ∧ m ≥ 3) :
  (∃ mobot, mobot = (1, 1)) ↔ (∃ colorings : ℕ, colorings = 6) :=
sorry

end mobot_coloring_six_colorings_l2032_203292


namespace pyramid_volume_is_1_12_l2032_203271

def base_rectangle_length_1 := 1
def base_rectangle_width_1_4 := 1 / 4
def pyramid_height_1 := 1

noncomputable def pyramid_volume : ℝ :=
  (1 / 3) * (base_rectangle_length_1 * base_rectangle_width_1_4) * pyramid_height_1

theorem pyramid_volume_is_1_12 : pyramid_volume = 1 / 12 :=
sorry

end pyramid_volume_is_1_12_l2032_203271


namespace problem_l2032_203208

theorem problem (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : f 1 = f 3) 
  (h2 : f 1 > f 4) 
  (hf : ∀ x, f x = a * x ^ 2 + b * x + c) :
  a < 0 ∧ 4 * a + b = 0 :=
by
  sorry

end problem_l2032_203208


namespace longest_interval_green_l2032_203213

-- Definitions for the conditions
def light_cycle_duration : ℕ := 180 -- total cycle duration in seconds
def green_duration : ℕ := 90 -- green light duration in seconds
def red_delay : ℕ := 10 -- red light delay between consecutive lights in seconds
def num_lights : ℕ := 8 -- number of lights

-- Theorem statement to be proved
theorem longest_interval_green (h1 : ∀ i : ℕ, i < num_lights → 
  ∃ t : ℕ, t < light_cycle_duration ∧ (∀ k : ℕ, i + k < num_lights → t + k * red_delay < light_cycle_duration ∧ t + k * red_delay + green_duration <= light_cycle_duration)):
  ∃ interval : ℕ, interval = 20 :=
sorry

end longest_interval_green_l2032_203213


namespace line_through_nodes_l2032_203267

def Point := (ℤ × ℤ)

structure Triangle :=
  (A B C : Point)

def is_node (p : Point) : Prop := 
  ∃ (x y : ℤ), p = (x, y)

def strictly_inside (p : Point) (t : Triangle) : Prop := 
  -- Assume we have a function that defines if a point is strictly inside a triangle
  sorry

def nodes_inside (t : Triangle) (nodes : List Point) : Prop := 
  nodes.length = 2 ∧ ∀ p, p ∈ nodes → strictly_inside p t

theorem line_through_nodes (t : Triangle) (node1 node2 : Point) (h_inside : nodes_inside t [node1, node2]) :
   ∃ (v : Point), v ∈ [t.A, t.B, t.C] ∨
   (∃ (s : Triangle -> Point -> Point -> Prop), s t node1 node2) := 
sorry

end line_through_nodes_l2032_203267


namespace find_value_expression_l2032_203264

theorem find_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 75)
  (h2 : y^2 + y * z + z^2 = 4)
  (h3 : z^2 + x * z + x^2 = 79) :
  x * y + y * z + x * z = 20 := 
sorry

end find_value_expression_l2032_203264


namespace lewis_earnings_during_harvest_l2032_203237

-- Define the conditions
def regular_earnings_per_week : ℕ := 28
def overtime_earnings_per_week : ℕ := 939
def number_of_weeks : ℕ := 1091

-- Define the total earnings per week
def total_earnings_per_week := regular_earnings_per_week + overtime_earnings_per_week

-- Define the total earnings during the harvest season
def total_earnings_during_harvest := total_earnings_per_week * number_of_weeks

-- Theorem statement
theorem lewis_earnings_during_harvest : total_earnings_during_harvest = 1055497 := by
  sorry

end lewis_earnings_during_harvest_l2032_203237


namespace shopkeeper_loss_percentage_l2032_203297

theorem shopkeeper_loss_percentage
  (total_stock_value : ℝ)
  (overall_loss : ℝ)
  (first_part_percentage : ℝ)
  (first_part_profit_percentage : ℝ)
  (remaining_part_loss : ℝ)
  (total_worth_first_part : ℝ)
  (first_part_profit : ℝ)
  (remaining_stock_value : ℝ)
  (remaining_stock_loss : ℝ)
  (loss_percentage : ℝ) :
  total_stock_value = 16000 →
  overall_loss = 400 →
  first_part_percentage = 0.10 →
  first_part_profit_percentage = 0.20 →
  total_worth_first_part = total_stock_value * first_part_percentage →
  first_part_profit = total_worth_first_part * first_part_profit_percentage →
  remaining_stock_value = total_stock_value * (1 - first_part_percentage) →
  remaining_stock_loss = overall_loss + first_part_profit →
  loss_percentage = (remaining_stock_loss / remaining_stock_value) * 100 →
  loss_percentage = 5 :=
by intros; sorry

end shopkeeper_loss_percentage_l2032_203297


namespace probability_interval_l2032_203248

-- Define the probability distribution and conditions
def P (xi : ℕ) (c : ℚ) : ℚ := c / (xi * (xi + 1))

-- Given conditions
variables (c : ℚ)
axiom condition : P 1 c + P 2 c + P 3 c + P 4 c = 1

-- Define the interval probability
def interval_prob (c : ℚ) : ℚ := P 1 c + P 2 c

-- Prove that the computed probability matches the expected value
theorem probability_interval : interval_prob (5 / 4) = 5 / 6 :=
by
  -- skip proof
  sorry

end probability_interval_l2032_203248


namespace total_population_of_Springfield_and_Greenville_l2032_203296

theorem total_population_of_Springfield_and_Greenville :
  let Springfield := 482653
  let diff := 119666
  let Greenville := Springfield - diff
  Springfield + Greenville = 845640 := by
  sorry

end total_population_of_Springfield_and_Greenville_l2032_203296


namespace geometric_sequence_general_term_arithmetic_sequence_sum_l2032_203276

variable {n : ℕ}

-- Defining sequences and sums
def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℕ := sorry
def T (n : ℕ) : ℕ := sorry
def b (n : ℕ) : ℕ := sorry

-- Given conditions
axiom h1 : 2 * S n = 3 * a n - 3
axiom h2 : b 1 = a 1
axiom h3 : b 7 = b 1 * b 2
axiom a1_value : a 1 = 3
axiom d_value : ∃ d : ℕ, b 2 = b 1 + d ∧ b 7 = b 1 + 6 * d

theorem geometric_sequence_general_term : a n = 3 ^ n :=
by sorry

theorem arithmetic_sequence_sum : T n = n^2 + 2*n :=
by sorry

end geometric_sequence_general_term_arithmetic_sequence_sum_l2032_203276


namespace jacob_walked_8_miles_l2032_203270

theorem jacob_walked_8_miles (rate time : ℝ) (h_rate : rate = 4) (h_time : time = 2) :
  rate * time = 8 := by
  -- conditions
  have hr : rate = 4 := h_rate
  have ht : time = 2 := h_time
  -- problem
  sorry

end jacob_walked_8_miles_l2032_203270


namespace inequality_a_b_c_l2032_203288

noncomputable def a := Real.log (Real.pi / 3)
noncomputable def b := Real.log (Real.exp 1 / 3)
noncomputable def c := Real.exp (0.5)

theorem inequality_a_b_c : c > a ∧ a > b := by
  sorry

end inequality_a_b_c_l2032_203288


namespace divisible_by_6_l2032_203282

theorem divisible_by_6 (n : ℤ) : 6 ∣ (n * (n + 1) * (n + 2)) :=
sorry

end divisible_by_6_l2032_203282


namespace grade_point_average_one_third_classroom_l2032_203283

theorem grade_point_average_one_third_classroom
  (gpa1 : ℝ) -- grade point average of one third of the classroom
  (gpa_rest : ℝ) -- grade point average of the rest of the classroom
  (gpa_whole : ℝ) -- grade point average of the whole classroom
  (h_rest : gpa_rest = 45)
  (h_whole : gpa_whole = 48) :
  gpa1 = 54 :=
by
  sorry

end grade_point_average_one_third_classroom_l2032_203283


namespace average_marbles_of_other_colors_l2032_203279

theorem average_marbles_of_other_colors :
  let total_percentage := 100
  let clear_percentage := 40
  let black_percentage := 20
  let other_percentage := total_percentage - clear_percentage - black_percentage
  let marbles_taken := 5
  (other_percentage / 100) * marbles_taken = 2 :=
by
  sorry

end average_marbles_of_other_colors_l2032_203279


namespace euler_phi_divisibility_l2032_203281

def euler_phi (n : ℕ) : ℕ := sorry -- Placeholder for the Euler phi-function

theorem euler_phi_divisibility (n : ℕ) (hn : n > 0) :
    2^(n * (n + 1)) ∣ 32 * euler_phi (2^(2^n) - 1) :=
sorry

end euler_phi_divisibility_l2032_203281


namespace find_f_l2032_203204

-- Define the function space and conditions
def func (f : ℕ+ → ℝ) :=
  (∀ m n : ℕ+, f (m * n) = f m + f n) ∧
  (∀ n : ℕ+, f (n + 1) ≥ f n)

-- Define the theorem statement
theorem find_f (f : ℕ+ → ℝ) (hf : func f) : ∀ n : ℕ+, f n = 0 :=
sorry

end find_f_l2032_203204


namespace books_from_first_shop_l2032_203219

theorem books_from_first_shop (x : ℕ) (h : (2080 : ℚ) / (x + 50) = 18.08695652173913) : x = 65 :=
by
  -- proof steps
  sorry

end books_from_first_shop_l2032_203219


namespace glorias_ratio_l2032_203289

variable (Q : ℕ) -- total number of quarters
variable (dimes : ℕ) -- total number of dimes, given as 350
variable (quarters_left : ℕ) -- number of quarters left

-- Given conditions
def conditions (Q dimes quarters_left : ℕ) : Prop :=
  dimes = 350 ∧
  quarters_left = (3 * Q) / 5 ∧
  (dimes + quarters_left = 392)

-- The ratio of dimes to quarters left
def ratio_of_dimes_to_quarters_left (dimes quarters_left : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd dimes quarters_left
  (dimes / gcd, quarters_left / gcd)

theorem glorias_ratio (Q : ℕ) (quarters_left : ℕ) : conditions Q 350 quarters_left → ratio_of_dimes_to_quarters_left 350 quarters_left = (25, 3) := by 
  sorry

end glorias_ratio_l2032_203289


namespace initial_books_l2032_203233

variable (B : ℤ)

theorem initial_books (h1 : 4 / 6 * B = B - 3300) (h2 : 3300 = 2 / 6 * B) : B = 9900 :=
by
  sorry

end initial_books_l2032_203233


namespace greatest_multiple_of_30_less_than_800_l2032_203243

theorem greatest_multiple_of_30_less_than_800 : 
    ∃ n : ℤ, (n % 30 = 0) ∧ (n < 800) ∧ (∀ m : ℤ, (m % 30 = 0) ∧ (m < 800) → m ≤ n) ∧ n = 780 :=
by
  sorry

end greatest_multiple_of_30_less_than_800_l2032_203243


namespace problem_statement_l2032_203224

def scientific_notation_correct (x : ℝ) : Prop :=
  x = 5.642 * 10 ^ 5

theorem problem_statement : scientific_notation_correct 564200 :=
by
  sorry

end problem_statement_l2032_203224


namespace cat_food_insufficient_l2032_203215

variable (B S : ℝ)

theorem cat_food_insufficient (h1 : B > S) (h2 : B < 2 * S) (h3 : B + 2 * S = 2 * D) : 
  4 * B + 4 * S < 3 * (B + 2 * S) :=
by
  sorry

end cat_food_insufficient_l2032_203215


namespace find_sum_2017_l2032_203202

-- Define the arithmetic sequence and its properties
def is_arithmetic_sequence (a : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + a n) / 2

-- Given conditions
variables (a : ℕ → ℤ)
axiom h1 : is_arithmetic_sequence a
axiom h2 : sum_first_n_terms a 2011 = -2011
axiom h3 : a 1012 = 3

-- Theorem to be proven
theorem find_sum_2017 : sum_first_n_terms a 2017 = 2017 :=
by sorry

end find_sum_2017_l2032_203202


namespace smallest_points_to_guarantee_victory_l2032_203256

noncomputable def pointsForWinning : ℕ := 5
noncomputable def pointsForSecond : ℕ := 3
noncomputable def pointsForThird : ℕ := 1

theorem smallest_points_to_guarantee_victory :
  ∀ (student_points : ℕ),
  (exists (x y z : ℕ), (x = pointsForWinning ∨ x = pointsForSecond ∨ x = pointsForThird) ∧
                         (y = pointsForWinning ∨ y = pointsForSecond ∨ y = pointsForThird) ∧
                         (z = pointsForWinning ∨ z = pointsForSecond ∨ z = pointsForThird) ∧
                         student_points = x + y + z) →
  (∃ (victory_points : ℕ), victory_points = 13) →
  (∀ other_points : ℕ, other_points < victory_points) :=
sorry

end smallest_points_to_guarantee_victory_l2032_203256


namespace margie_change_l2032_203291

theorem margie_change :
  let num_apples := 5
  let cost_per_apple := 0.30
  let discount := 0.10
  let amount_paid := 10.00
  let total_cost := num_apples * cost_per_apple
  let discounted_cost := total_cost * (1 - discount)
  let change_received := amount_paid - discounted_cost
  change_received = 8.65 := sorry

end margie_change_l2032_203291


namespace highest_y_coordinate_l2032_203209

-- Define the conditions
def ellipse_condition (x y : ℝ) : Prop :=
  (x^2 / 25) + ((y - 3)^2 / 9) = 1

-- The theorem to prove
theorem highest_y_coordinate : ∃ x : ℝ, ∀ y : ℝ, ellipse_condition x y → y ≤ 6 :=
sorry

end highest_y_coordinate_l2032_203209


namespace number_of_3digit_even_numbers_divisible_by_9_l2032_203266

theorem number_of_3digit_even_numbers_divisible_by_9 : 
    ∃ n : ℕ, (n = 50) ∧
    (∀ k, (108 + (k - 1) * 18 = 990) ↔ (108 ≤ 108 + (k - 1) * 18 ∧ 108 + (k - 1) * 18 ≤ 999)) :=
by {
  sorry
}

end number_of_3digit_even_numbers_divisible_by_9_l2032_203266


namespace machine_part_masses_l2032_203247

theorem machine_part_masses :
  ∃ (x y : ℝ), (y - 2 * x = 100) ∧ (875 / x - 900 / y = 3) ∧ (x = 175) ∧ (y = 450) :=
by {
  sorry
}

end machine_part_masses_l2032_203247


namespace division_result_l2032_203211

theorem division_result:
    35 / 0.07 = 500 := by
  sorry

end division_result_l2032_203211


namespace beta_angle_relationship_l2032_203251

theorem beta_angle_relationship (α β γ : ℝ) (h1 : β - α = 3 * γ) (h2 : α + β + γ = 180) : β = 90 + γ :=
sorry

end beta_angle_relationship_l2032_203251


namespace fraction_of_calls_processed_by_team_B_l2032_203287

theorem fraction_of_calls_processed_by_team_B
  (C_B : ℕ) -- the number of calls processed by each member of team B
  (B : ℕ)  -- the number of call center agents in team B
  (C_A : ℕ := C_B / 5) -- each member of team A processes 1/5 the number of calls as each member of team B
  (A : ℕ := 5 * B / 8) -- team A has 5/8 as many agents as team B
: 
  (B * C_B) / ((A * C_A) + (B * C_B)) = (8 / 9 : ℚ) :=
sorry

end fraction_of_calls_processed_by_team_B_l2032_203287


namespace sum_of_two_numbers_l2032_203222

theorem sum_of_two_numbers (a b : ℝ) (h1 : a + b = 25) (h2 : a * b = 144) (h3 : |a - b| = 7) : a + b = 25 := 
  by
  sorry

end sum_of_two_numbers_l2032_203222


namespace problem_part1_problem_part2_problem_part3_l2032_203299

open Set

-- Define the universal set U
def U : Set ℝ := Set.univ 

-- Define sets A and B within the universal set U
def A : Set ℝ := { x | 0 < x ∧ x ≤ 2 }
def B : Set ℝ := { x | x < -3 ∨ x > 1 }

-- Define the complements of A and B within U
def complement_A : Set ℝ := U \ A
def complement_B : Set ℝ := U \ B

-- Define the results as goals to be proved
theorem problem_part1 : A ∩ B = { x | 1 < x ∧ x ≤ 2 } := 
by
  sorry

theorem problem_part2 : complement_A ∩ complement_B = { x | -3 ≤ x ∧ x ≤ 0 } :=
by
  sorry

theorem problem_part3 : U \ (A ∪ B) = { x | -3 ≤ x ∧ x ≤ 0 } :=
by
  sorry

end problem_part1_problem_part2_problem_part3_l2032_203299


namespace nat_numbers_square_minus_one_power_of_prime_l2032_203212

def is_power_of_prime (x : ℕ) : Prop :=
  ∃ (p : ℕ), Nat.Prime p ∧ ∃ (k : ℕ), x = p ^ k

theorem nat_numbers_square_minus_one_power_of_prime (n : ℕ) (hn : 1 ≤ n) :
  is_power_of_prime (n ^ 2 - 1) ↔ (n = 2 ∨ n = 3) := by
  sorry

end nat_numbers_square_minus_one_power_of_prime_l2032_203212


namespace class_7th_grade_1_has_higher_average_score_class_7th_grade_2_has_higher_weighted_score_l2032_203286

noncomputable def average_score (costume pitch innovation : ℕ) : ℚ :=
  (costume + pitch + innovation) / 3

noncomputable def weighted_average_score (costume pitch innovation : ℕ) : ℚ :=
  (costume + 7 * pitch + 2 * innovation) / 10

theorem class_7th_grade_1_has_higher_average_score :
  average_score 90 77 85 > average_score 74 95 80 :=
by sorry

theorem class_7th_grade_2_has_higher_weighted_score :
  weighted_average_score 74 95 80 > weighted_average_score 90 77 85 :=
by sorry

end class_7th_grade_1_has_higher_average_score_class_7th_grade_2_has_higher_weighted_score_l2032_203286


namespace units_digit_of_3_pow_7_pow_6_l2032_203231

theorem units_digit_of_3_pow_7_pow_6 :
  (3 ^ (7 ^ 6) % 10) = 3 := 
sorry

end units_digit_of_3_pow_7_pow_6_l2032_203231


namespace b_catches_a_distance_l2032_203258

-- Define the initial conditions
def a_speed : ℝ := 10  -- A's speed in km/h
def b_speed : ℝ := 20  -- B's speed in km/h
def start_delay : ℝ := 3  -- B starts cycling 3 hours after A in hours

-- Define the target distance to prove
theorem b_catches_a_distance : ∃ (d : ℝ), d = 60 := 
by 
  sorry

end b_catches_a_distance_l2032_203258


namespace color_triplet_exists_l2032_203263

theorem color_triplet_exists (color : ℕ → Prop) :
  (∀ n, color n ∨ ¬ color n) → ∃ x y z : ℕ, (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ color x = color y ∧ color y = color z ∧ x * y = z ^ 2 :=
by
  sorry

end color_triplet_exists_l2032_203263


namespace number_of_solutions_l2032_203205

theorem number_of_solutions :
  (∃(x y : ℤ), x^4 + y^2 = 6 * y - 8) ∧ ∃!(x y : ℤ), x^4 + y^2 = 6 * y - 8 := 
sorry

end number_of_solutions_l2032_203205


namespace find_point_on_line_and_distance_l2032_203249

def distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem find_point_on_line_and_distance :
  ∃ P : ℝ × ℝ, (2 * P.1 - 3 * P.2 + 5 = 0) ∧ (distance P (2, 3) = 13) →
  (P = (5, 5) ∨ P = (-1, 1)) :=
by
  sorry

end find_point_on_line_and_distance_l2032_203249


namespace common_chord_circle_eq_l2032_203253

theorem common_chord_circle_eq {a b : ℝ} (hb : b ≠ 0) :
  ∃ x y : ℝ, 
    (x^2 + y^2 - 2 * a * x = 0) ∧ 
    (x^2 + y^2 - 2 * b * y = 0) ∧ 
    (a^2 + b^2) * (x^2 + y^2) - 2 * a * b * (b * x + a * y) = 0 :=
by sorry

end common_chord_circle_eq_l2032_203253


namespace lindsey_savings_l2032_203236

theorem lindsey_savings
  (september_savings : Nat := 50)
  (october_savings : Nat := 37)
  (november_savings : Nat := 11)
  (additional_savings : Nat := 25)
  (video_game_cost : Nat := 87)
  (total_savings := september_savings + october_savings + november_savings)
  (mom_bonus : Nat := if total_savings > 75 then additional_savings else 0)
  (final_amount := total_savings + mom_bonus - video_game_cost) :
  final_amount = 36 := by
  sorry

end lindsey_savings_l2032_203236


namespace find_divisor_value_l2032_203226

theorem find_divisor_value
  (D : ℕ) 
  (h1 : ∃ k : ℕ, 242 = k * D + 6)
  (h2 : ∃ l : ℕ, 698 = l * D + 13)
  (h3 : ∃ m : ℕ, 940 = m * D + 5) : 
  D = 14 :=
by
  sorry

end find_divisor_value_l2032_203226


namespace triangle_perimeter_l2032_203200

theorem triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 6)
  (c1 c2 : ℝ) (h3 : (c1 - 2) * (c1 - 4) = 0) (h4 : (c2 - 2) * (c2 - 4) = 0) :
  c1 = 2 ∨ c1 = 4 → c2 = 2 ∨ c2 = 4 → 
  (c1 ≠ 2 ∧ c1 = 4 ∨ c2 ≠ 2 ∧ c2 = 4) → 
  (a + b + c1 = 13 ∨ a + b + c2 = 13) :=
by
  sorry

end triangle_perimeter_l2032_203200


namespace smallest_divisible_1_to_10_l2032_203293

open Nat

def is_divisible_by_all (n : ℕ) (s : List ℕ) : Prop :=
  ∀ x ∈ s, x ∣ n

theorem smallest_divisible_1_to_10 : ∃ n, is_divisible_by_all n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ∧ n = 2520 := by
  sorry

end smallest_divisible_1_to_10_l2032_203293


namespace expression_evaluation_l2032_203238

theorem expression_evaluation (a : ℝ) (h : a = 9) : ( (a ^ (1 / 3)) / (a ^ (1 / 5)) ) = a^(2 / 15) :=
by
  sorry

end expression_evaluation_l2032_203238


namespace intersection_of_M_and_N_l2032_203269

-- Definitions of the sets M and N
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Statement of the theorem proving the intersection of M and N
theorem intersection_of_M_and_N :
  M ∩ N = {2, 3} :=
by sorry

end intersection_of_M_and_N_l2032_203269


namespace binomial_expansion_max_coefficient_l2032_203229

theorem binomial_expansion_max_coefficient (n : ℕ) (h : n > 0) 
  (h_max_coefficient: ∀ m : ℕ, m ≠ 5 → (Nat.choose n m ≤ Nat.choose n 5)) : 
  n = 10 :=
sorry

end binomial_expansion_max_coefficient_l2032_203229


namespace range_of_a_l2032_203210

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (|x - 1| + |x - 2| ≤ a^2 + a + 1)) → -1 < a ∧ a < 0 :=
by
  sorry

end range_of_a_l2032_203210
