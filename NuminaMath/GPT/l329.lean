import Mathlib

namespace NUMINAMATH_GPT_uncle_wang_withdraw_amount_l329_32982

noncomputable def total_amount (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal + principal * rate * time

theorem uncle_wang_withdraw_amount :
  total_amount 100000 (315/10000) 2 = 106300 := by
  sorry

end NUMINAMATH_GPT_uncle_wang_withdraw_amount_l329_32982


namespace NUMINAMATH_GPT_greatest_value_exprD_l329_32969

-- Conditions
def a : ℚ := 2
def b : ℚ := 5

-- Expressions
def exprA := a / b
def exprB := b / a
def exprC := a - b
def exprD := b - a
def exprE := (1/2 : ℚ) * a

-- Proof problem statement
theorem greatest_value_exprD : exprD = 3 ∧ exprD > exprA ∧ exprD > exprB ∧ exprD > exprC ∧ exprD > exprE := sorry

end NUMINAMATH_GPT_greatest_value_exprD_l329_32969


namespace NUMINAMATH_GPT_stock_exchange_total_l329_32967

theorem stock_exchange_total (L H : ℕ) 
  (h1 : H = 1080) 
  (h2 : H = 6 * L / 5) : 
  (L + H = 1980) :=
by {
  -- L and H are given as natural numbers
  -- h1: H = 1080
  -- h2: H = 1.20L -> H = 6L/5 as Lean does not handle floating point well directly in integers.
  sorry
}

end NUMINAMATH_GPT_stock_exchange_total_l329_32967


namespace NUMINAMATH_GPT_max_value_m_l329_32936

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem max_value_m (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, quadratic_function a b c (x-4) = quadratic_function a b c (2-x))
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → quadratic_function a b c x ≤ ( (x+1)/2 )^2)
  (h4 : ∀ x : ℝ, quadratic_function a b c x ≥ 0)
  (h_min : ∃ x : ℝ, quadratic_function a b c x = 0) :
  ∃ (m : ℝ), m > 1 ∧ (∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → quadratic_function a b c (x+t) ≤ x) ∧ m = 9 := 
sorry

end NUMINAMATH_GPT_max_value_m_l329_32936


namespace NUMINAMATH_GPT_sector_area_l329_32925

theorem sector_area (n : ℝ) (r : ℝ) (h₁ : n = 120) (h₂ : r = 4) : 
  (n * Real.pi * r^2 / 360) = (16 * Real.pi / 3) :=
by 
  sorry

end NUMINAMATH_GPT_sector_area_l329_32925


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l329_32980

theorem common_ratio_of_geometric_series :
  let a := (8:ℚ) / 10
  let second_term := (-6:ℚ) / 15 
  let r := second_term / a
  r = -1 / 2 :=
by
  let a := (8:ℚ) / 10
  let second_term := (-6:ℚ) / 15 
  let r := second_term / a
  have : r = -1 / 2 := sorry
  exact this

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l329_32980


namespace NUMINAMATH_GPT_wall_width_is_4_l329_32913

structure Wall where
  width : ℝ
  height : ℝ
  length : ℝ
  volume : ℝ

theorem wall_width_is_4 (h_eq_6w : ∀ (wall : Wall), wall.height = 6 * wall.width)
                        (l_eq_7h : ∀ (wall : Wall), wall.length = 7 * wall.height)
                        (volume_16128 : ∀ (wall : Wall), wall.volume = 16128) :
  ∃ (wall : Wall), wall.width = 4 :=
by
  sorry

end NUMINAMATH_GPT_wall_width_is_4_l329_32913


namespace NUMINAMATH_GPT_function_increasing_l329_32960

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) :=
  ∀ (x1 x2 : ℝ), x1 < x2 → f x1 < f x2

theorem function_increasing {f : ℝ → ℝ}
  (H : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 < f x2) :
  is_monotonically_increasing f :=
by
  sorry

end NUMINAMATH_GPT_function_increasing_l329_32960


namespace NUMINAMATH_GPT_x_varies_as_sin_squared_l329_32914

variable {k j z : ℝ}
variable (x y : ℝ)

-- condition: x is proportional to y^2
def proportional_xy_square (x y : ℝ) (k : ℝ) : Prop :=
  x = k * y ^ 2

-- condition: y is proportional to sin(z)
def proportional_y_sin (y : ℝ) (j z : ℝ) : Prop :=
  y = j * Real.sin z

-- statement to prove: x is proportional to (sin(z))^2
theorem x_varies_as_sin_squared (k j z : ℝ) (x y : ℝ)
  (h1 : proportional_xy_square x y k)
  (h2 : proportional_y_sin y j z) :
  ∃ m, x = m * (Real.sin z) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_x_varies_as_sin_squared_l329_32914


namespace NUMINAMATH_GPT_find_middle_and_oldest_sons_l329_32942

-- Defining the conditions
def youngest_age : ℕ := 2
def father_age : ℕ := 33
def father_age_in_12_years : ℕ := father_age + 12
def youngest_age_in_12_years : ℕ := youngest_age + 12

-- Lean theorem statement to find the ages of the middle and oldest sons
theorem find_middle_and_oldest_sons (y z : ℕ) (h1 : father_age_in_12_years = (youngest_age_in_12_years + 12 + y + 12 + z + 12)) :
  y = 3 ∧ z = 4 :=
sorry

end NUMINAMATH_GPT_find_middle_and_oldest_sons_l329_32942


namespace NUMINAMATH_GPT_tank_capacity_l329_32961

theorem tank_capacity :
  ∃ T : ℝ, (5/8) * T + 12 = (11/16) * T ∧ T = 192 :=
sorry

end NUMINAMATH_GPT_tank_capacity_l329_32961


namespace NUMINAMATH_GPT_Jayden_less_Coraline_l329_32976

variables (M J : ℕ)
def Coraline_number := 80
def total_sum := 180

theorem Jayden_less_Coraline
  (h1 : M = J + 20)
  (h2 : J < Coraline_number)
  (h3 : M + J + Coraline_number = total_sum) :
  Coraline_number - J = 40 := by
  sorry

end NUMINAMATH_GPT_Jayden_less_Coraline_l329_32976


namespace NUMINAMATH_GPT_equal_sundays_tuesdays_l329_32937

theorem equal_sundays_tuesdays (days_in_month : ℕ) (week_days : ℕ) (extra_days : ℕ) :
  days_in_month = 30 → week_days = 7 → extra_days = 2 → 
  ∃ n, n = 3 ∧ ∀ start_day : ℕ, start_day = 3 ∨ start_day = 4 ∨ start_day = 5 :=
by sorry

end NUMINAMATH_GPT_equal_sundays_tuesdays_l329_32937


namespace NUMINAMATH_GPT_max_value_expression_l329_32988

theorem max_value_expression : ∃ s_max : ℝ, 
  (∀ s : ℝ, -3 * s^2 + 24 * s - 7 ≤ -3 * s_max^2 + 24 * s_max - 7) ∧
  (-3 * s_max^2 + 24 * s_max - 7 = 41) :=
sorry

end NUMINAMATH_GPT_max_value_expression_l329_32988


namespace NUMINAMATH_GPT_sales_tax_difference_l329_32979

theorem sales_tax_difference :
  let price : ℝ := 50
  let tax_rate1 : ℝ := 0.075
  let tax_rate2 : ℝ := 0.07
  (price * tax_rate1) - (price * tax_rate2) = 0.25 := by
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l329_32979


namespace NUMINAMATH_GPT_area_difference_l329_32950

theorem area_difference (radius1 radius2 : ℝ) (pi : ℝ) (h1 : radius1 = 15) (h2 : radius2 = 14 / 2) :
  pi * radius1 ^ 2 - pi * radius2 ^ 2 = 176 * pi :=
by 
  sorry

end NUMINAMATH_GPT_area_difference_l329_32950


namespace NUMINAMATH_GPT_find_original_price_l329_32985

variable (original_price : ℝ)
variable (final_price : ℝ) (first_reduction_rate : ℝ) (second_reduction_rate : ℝ)

theorem find_original_price :
  final_price = 15000 →
  first_reduction_rate = 0.30 →
  second_reduction_rate = 0.40 →
  0.42 * original_price = final_price →
  original_price = 35714 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_find_original_price_l329_32985


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l329_32935

-- Problem 1
theorem problem1 : (-10 + (-5) - (-18)) = 3 := 
by
  sorry

-- Problem 2
theorem problem2 : (-80 * (-(4 / 5)) / (abs 16)) = -4 := 
by 
  sorry

-- Problem 3
theorem problem3 : ((1/2 - 5/9 + 5/6 - 7/12) * (-36)) = -7 := 
by 
  sorry

-- Problem 4
theorem problem4 : (- 3^2 * (-1/3)^2 +(-2)^2 / (- (2/3))^3) = -29 / 27 :=
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l329_32935


namespace NUMINAMATH_GPT_hyperbola_equation_l329_32949

theorem hyperbola_equation (a b k : ℝ) (p : ℝ × ℝ) (h_asymptotes : b = 3 * a)
  (h_hyperbola_passes_point : p = (2, -3 * Real.sqrt 3)) (h_hyperbola : ∀ x y, x^2 - (y^2 / (3 * a)^2) = k) :
  ∃ k, k = 1 :=
by
  -- Given the point p and asymptotes, we should prove k = 1.
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l329_32949


namespace NUMINAMATH_GPT_student_avg_greater_actual_avg_l329_32903

theorem student_avg_greater_actual_avg
  (x y z : ℝ)
  (hxy : x < y)
  (hyz : y < z) :
  (x + y + 2 * z) / 4 > (x + y + z) / 3 := by
  sorry

end NUMINAMATH_GPT_student_avg_greater_actual_avg_l329_32903


namespace NUMINAMATH_GPT_carpet_interior_length_l329_32978

/--
A carpet is designed using three different colors, forming three nested rectangles with different areas in an arithmetic progression. 
The innermost rectangle has a width of two feet. Each of the two colored borders is 2 feet wide on all sides.
Determine the length in feet of the innermost rectangle. 
-/
theorem carpet_interior_length 
  (x : ℕ) -- length of the innermost rectangle
  (hp : ∀ (a b c : ℕ), a = 2 * x ∧ b = (4 * x + 24) ∧ c = (4 * x + 56) → (b - a) = (c - b)) 
  : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_carpet_interior_length_l329_32978


namespace NUMINAMATH_GPT_inequality_proof_l329_32954

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / b + b^2 / c + c^2 / a) ≥ 3 * (a^3 + b^3 + c^3) / (a^2 + b^2 + c^2) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l329_32954


namespace NUMINAMATH_GPT_sequence_property_l329_32908

theorem sequence_property (k : ℝ) (h_k : 0 < k) (x : ℕ → ℝ)
  (h₀ : x 0 = 1)
  (h₁ : x 1 = 1 + k)
  (rec1 : ∀ n, x (2*n + 1) - x (2*n) = x (2*n) - x (2*n - 1))
  (rec2 : ∀ n, x (2*n) / x (2*n - 1) = x (2*n - 1) / x (2*n - 2)) :
  ∃ N, ∀ n ≥ N, x n > 1994 :=
by
  sorry

end NUMINAMATH_GPT_sequence_property_l329_32908


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l329_32927

variable (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℕ)

def S₁₀ (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℕ) : ℕ :=
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀

theorem arithmetic_sequence_sum (h : S₁₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ = 120) :
  a₁ + a₁₀ = 24 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l329_32927


namespace NUMINAMATH_GPT_polynomial_remainder_is_zero_l329_32998

theorem polynomial_remainder_is_zero :
  ∀ (x : ℤ), ((x^5 - 1) * (x^3 - 1)) % (x^2 + x + 1) = 0 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_is_zero_l329_32998


namespace NUMINAMATH_GPT_cookies_in_fridge_l329_32943

theorem cookies_in_fridge (total_baked : ℕ) (cookies_Tim : ℕ) (cookies_Mike : ℕ) (cookies_Sarah : ℕ) (cookies_Anna : ℕ)
  (h_total_baked : total_baked = 1024)
  (h_cookies_Tim : cookies_Tim = 48)
  (h_cookies_Mike : cookies_Mike = 58)
  (h_cookies_Sarah : cookies_Sarah = 78)
  (h_cookies_Anna : cookies_Anna = (2 * (cookies_Tim + cookies_Mike)) - (cookies_Sarah / 2)) :
  total_baked - (cookies_Tim + cookies_Mike + cookies_Sarah + cookies_Anna) = 667 := by
sorry

end NUMINAMATH_GPT_cookies_in_fridge_l329_32943


namespace NUMINAMATH_GPT_range_of_m_l329_32953

variable (a b c m y1 y2 y3 : Real)

-- Given points and the parabola equation
def on_parabola (x y a b c : Real) : Prop := y = a * x^2 + b * x + c

-- Conditions
variable (hP : on_parabola (-2) y1 a b c)
variable (hQ : on_parabola 4 y2 a b c)
variable (hM : on_parabola m y3 a b c)
variable (h_vertex : 2 * a * m + b = 0)
variable (h_y_order : y3 ≥ y2 ∧ y2 > y1)

-- Theorem to prove m > 1
theorem range_of_m : m > 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l329_32953


namespace NUMINAMATH_GPT_minimum_value_l329_32965

theorem minimum_value (a_n : ℕ → ℤ) (h : ∀ n, a_n n = n^2 - 8 * n + 5) : ∃ n, a_n n = -11 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l329_32965


namespace NUMINAMATH_GPT_compare_values_l329_32924

variable (f : ℝ → ℝ)
variable (hf_even : ∀ x, f x = f (-x))
variable (hf_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)

noncomputable def a : ℝ := f 1
noncomputable def b : ℝ := f (Real.log 3 / Real.log 0.5)
noncomputable def c : ℝ := f ((Real.log 3 / Real.log 2) - 1)

theorem compare_values (h_log1 : Real.log 3 / Real.log 0.5 < -1) 
                       (h_log2 : 0 < (Real.log 3 / Real.log 2) - 1 ∧ (Real.log 3 / Real.log 2) - 1 < 1) : 
  b < a ∧ a < c :=
by
  sorry

end NUMINAMATH_GPT_compare_values_l329_32924


namespace NUMINAMATH_GPT_range_of_a_l329_32938

noncomputable def f (x : ℝ) := (Real.log x) / x
noncomputable def g (x a : ℝ) := -Real.exp 1 * x^2 + a * x

theorem range_of_a (a : ℝ) : (∀ x1 : ℝ, ∃ x2 ∈ Set.Icc (1/3) 2, f x1 ≤ g x2 a) → 2 ≤ a :=
sorry

end NUMINAMATH_GPT_range_of_a_l329_32938


namespace NUMINAMATH_GPT_hypotenuse_of_45_45_90_triangle_l329_32945

theorem hypotenuse_of_45_45_90_triangle (a : ℝ) (h : ℝ) 
  (ha : a = 15) 
  (angle_opposite_leg : ℝ) 
  (h_angle : angle_opposite_leg = 45) 
  (right_triangle : ∃ θ : ℝ, θ = 90) : 
  h = 15 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_hypotenuse_of_45_45_90_triangle_l329_32945


namespace NUMINAMATH_GPT_single_discount_equivalence_l329_32939

variable (p : ℝ) (d1 d2 d3 : ℝ)

def apply_discount (price discount : ℝ) : ℝ :=
  price * (1 - discount)

def apply_multiple_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem single_discount_equivalence :
  p = 1200 →
  d1 = 0.15 →
  d2 = 0.10 →
  d3 = 0.05 →
  let final_price_multiple := apply_multiple_discounts p [d1, d2, d3]
  let single_discount := (p - final_price_multiple) / p
  single_discount = 0.27325 :=
by
  intros h1 h2 h3 h4
  let final_price_multiple := apply_multiple_discounts p [d1, d2, d3]
  let single_discount := (p - final_price_multiple) / p
  sorry

end NUMINAMATH_GPT_single_discount_equivalence_l329_32939


namespace NUMINAMATH_GPT_scientific_notation_32000000_l329_32922

def scientific_notation (n : ℕ) : String := sorry

theorem scientific_notation_32000000 :
  scientific_notation 32000000 = "3.2 × 10^7" :=
sorry

end NUMINAMATH_GPT_scientific_notation_32000000_l329_32922


namespace NUMINAMATH_GPT_problem_1_problem_2_l329_32946

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |3 * x - 2|

theorem problem_1 {a b : ℝ} (h : ∀ x, f x ≤ 5 → -4 * a / 5 ≤ x ∧ x ≤ 3 * b / 5) : 
  a = 1 ∧ b = 2 :=
sorry

theorem problem_2 {a b m : ℝ} (h1 : a = 1) (h2 : b = 2) (h3 : ∀ x, |x - a| + |x + b| ≥ m^2 - 3 * m + 5) :
  ∃ m, m = 2 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l329_32946


namespace NUMINAMATH_GPT_calculate_mirror_area_l329_32907

def outer_frame_width : ℝ := 65
def outer_frame_height : ℝ := 85
def frame_width : ℝ := 15

def mirror_width : ℝ := outer_frame_width - 2 * frame_width
def mirror_height : ℝ := outer_frame_height - 2 * frame_width
def mirror_area : ℝ := mirror_width * mirror_height

theorem calculate_mirror_area : mirror_area = 1925 := by
  sorry

end NUMINAMATH_GPT_calculate_mirror_area_l329_32907


namespace NUMINAMATH_GPT_poly_sequence_correct_l329_32992

-- Sequence of polynomials defined recursively
def f : ℕ → ℕ → ℕ 
| 0, x => 1
| 1, x => 1 + x 
| (k + 1), x => ((x + 1) * f (k) (x) - (x - k) * f (k - 1) (x)) / (k + 1)

-- Prove f(k, k) = 2^k for all k ≥ 0
theorem poly_sequence_correct (k : ℕ) : f k k = 2 ^ k := by
  sorry

end NUMINAMATH_GPT_poly_sequence_correct_l329_32992


namespace NUMINAMATH_GPT_total_fish_l329_32926

def LillyFish : ℕ := 10
def RosyFish : ℕ := 8
def MaxFish : ℕ := 15

theorem total_fish : LillyFish + RosyFish + MaxFish = 33 := by
  sorry

end NUMINAMATH_GPT_total_fish_l329_32926


namespace NUMINAMATH_GPT_bicyclist_speed_remainder_l329_32986

noncomputable def speed_of_bicyclist (total_distance first_distance remaining_distance time_for_first_distance total_time : ℝ) : ℝ :=
  remaining_distance / (total_time - time_for_first_distance)

theorem bicyclist_speed_remainder 
  (total_distance : ℝ)
  (first_distance : ℝ)
  (remaining_distance : ℝ)
  (first_speed : ℝ)
  (average_speed : ℝ)
  (correct_speed : ℝ) :
  total_distance = 250 → 
  first_distance = 100 →
  remaining_distance = total_distance - first_distance →
  first_speed = 20 →
  average_speed = 16.67 →
  correct_speed = 15 →
  speed_of_bicyclist total_distance first_distance remaining_distance (first_distance / first_speed) (total_distance / average_speed) = correct_speed :=
by
  sorry

end NUMINAMATH_GPT_bicyclist_speed_remainder_l329_32986


namespace NUMINAMATH_GPT_find_a_l329_32955

theorem find_a (a x1 x2 : ℝ)
  (h1: 4 * x1 ^ 2 - 4 * (a + 2) * x1 + a ^ 2 + 11 = 0)
  (h2: 4 * x2 ^ 2 - 4 * (a + 2) * x2 + a ^ 2 + 11 = 0)
  (h3: x1 - x2 = 3) : a = 4 := sorry

end NUMINAMATH_GPT_find_a_l329_32955


namespace NUMINAMATH_GPT_simplify_fraction_l329_32911

variable {a b c : ℝ}

theorem simplify_fraction (h : a + b + c ≠ 0) :
  (a^2 + 3*a*b + b^2 - c^2) / (a^2 + 3*a*c + c^2 - b^2) = (a + b - c) / (a - b + c) := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l329_32911


namespace NUMINAMATH_GPT_probability_no_shaded_rectangle_l329_32963

theorem probability_no_shaded_rectangle :
  let n := (1002 * 1001) / 2
  let m := 501 * 501
  (1 - (m / n) = 500 / 1001) := sorry

end NUMINAMATH_GPT_probability_no_shaded_rectangle_l329_32963


namespace NUMINAMATH_GPT_transformed_graph_area_l329_32902

theorem transformed_graph_area (g : ℝ → ℝ) (a b : ℝ)
  (h_area_g : ∫ x in a..b, g x = 15) :
  ∫ x in a..b, 2 * g (x + 3) = 30 := 
sorry

end NUMINAMATH_GPT_transformed_graph_area_l329_32902


namespace NUMINAMATH_GPT_circumscribed_sphere_radius_l329_32918

theorem circumscribed_sphere_radius (a b R : ℝ) (ha : a > 0) (hb : b > 0) :
  R = b^2 / (2 * (Real.sqrt (b^2 - a^2))) :=
sorry

end NUMINAMATH_GPT_circumscribed_sphere_radius_l329_32918


namespace NUMINAMATH_GPT_candidate_B_valid_votes_l329_32947

theorem candidate_B_valid_votes:
  let eligible_voters := 12000
  let abstained_percent := 0.1
  let invalid_votes_percent := 0.2
  let votes_for_C_percent := 0.05
  let A_less_B_percent := 0.2
  let total_voted := (1 - abstained_percent) * eligible_voters
  let valid_votes := (1 - invalid_votes_percent) * total_voted
  let votes_for_C := votes_for_C_percent * valid_votes
  (∃ Vb, valid_votes = (1 - A_less_B_percent) * Vb + Vb + votes_for_C 
         ∧ Vb = 4560) :=
sorry

end NUMINAMATH_GPT_candidate_B_valid_votes_l329_32947


namespace NUMINAMATH_GPT_simplify_expression_l329_32974

variable (a b : ℤ)

theorem simplify_expression : 
  (50 * a + 130 * b) + (21 * a + 64 * b) - (30 * a + 115 * b) - 2 * (10 * a - 25 * b) = 21 * a + 129 * b := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l329_32974


namespace NUMINAMATH_GPT_table_area_l329_32940

theorem table_area (A : ℝ) (runner_total : ℝ) (cover_percentage : ℝ) (double_layer : ℝ) (triple_layer : ℝ) :
  runner_total = 208 ∧
  cover_percentage = 0.80 ∧
  double_layer = 24 ∧
  triple_layer = 22 →
  A = 260 :=
by
  sorry

end NUMINAMATH_GPT_table_area_l329_32940


namespace NUMINAMATH_GPT_three_times_first_number_minus_second_value_l329_32973

theorem three_times_first_number_minus_second_value (x y : ℕ) 
  (h1 : x + y = 48) 
  (h2 : y = 17) : 
  3 * x - y = 76 := 
by 
  sorry

end NUMINAMATH_GPT_three_times_first_number_minus_second_value_l329_32973


namespace NUMINAMATH_GPT_geometric_sequence_x_value_l329_32981

theorem geometric_sequence_x_value (x : ℝ) (r : ℝ) 
  (h1 : 12 * r = x) 
  (h2 : x * r = 2 / 3) 
  (h3 : 0 < x) :
  x = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_x_value_l329_32981


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l329_32948

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 3) :
  (1 / (a : ℚ)^2) + (1 / (b : ℚ)^2) = 10 / 9 :=
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l329_32948


namespace NUMINAMATH_GPT_compare_fractions_l329_32958

theorem compare_fractions : (6/29 : ℚ) < (8/25 : ℚ) ∧ (8/25 : ℚ) < (11/31 : ℚ):=
by
  have h1 : (6/29 : ℚ) < (8/25 : ℚ) := sorry
  have h2 : (8/25 : ℚ) < (11/31 : ℚ) := sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_compare_fractions_l329_32958


namespace NUMINAMATH_GPT_find_y_l329_32910

-- Definitions for the given conditions
def angle_sum_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

def right_triangle (A B : ℝ) : Prop :=
  A + B = 90

-- The main theorem to prove
theorem find_y 
  (angle_ABC : ℝ)
  (angle_BAC : ℝ)
  (angle_DCE : ℝ)
  (h1 : angle_ABC = 70)
  (h2 : angle_BAC = 50)
  (h3 : right_triangle angle_DCE 30)
  : 30 = 30 :=
sorry

end NUMINAMATH_GPT_find_y_l329_32910


namespace NUMINAMATH_GPT_tim_more_points_than_joe_l329_32995

variable (J K T : ℕ)

theorem tim_more_points_than_joe (h1 : T = 30) (h2 : T = K / 2) (h3 : J + T + K = 100) : T - J = 20 :=
by
  sorry

end NUMINAMATH_GPT_tim_more_points_than_joe_l329_32995


namespace NUMINAMATH_GPT_sum_of_areas_of_triangles_l329_32987

theorem sum_of_areas_of_triangles 
  (AB BG GE DE : ℕ) 
  (A₁ A₂ : ℕ)
  (H1 : AB = 2) 
  (H2 : BG = 3) 
  (H3 : GE = 4) 
  (H4 : DE = 5) 
  (H5 : 3 * A₁ + 4 * A₂ = 48)
  (H6 : 9 * A₁ + 5 * A₂ = 102) : 
  1 * AB * A₁ / 2 + 1 * DE * A₂ / 2 = 23 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_triangles_l329_32987


namespace NUMINAMATH_GPT_arithmetic_sequence_diff_l329_32921

theorem arithmetic_sequence_diff (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 7 = a 3 + 4 * d) :
  a 2008 - a 2000 = 8 * d :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_diff_l329_32921


namespace NUMINAMATH_GPT_exercise_l329_32996

open Set

theorem exercise (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5, 6}) (hA : A = {1, 3, 5}) (hB : B = {2, 4, 5}) :
  A ∩ (U \ B) = {1, 3} := by
  sorry

end NUMINAMATH_GPT_exercise_l329_32996


namespace NUMINAMATH_GPT_find_number_l329_32991

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 11) : x = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l329_32991


namespace NUMINAMATH_GPT_sphere_surface_area_of_circumscribing_cuboid_l329_32975

theorem sphere_surface_area_of_circumscribing_cuboid :
  ∀ (a b c : ℝ), a = 5 ∧ b = 4 ∧ c = 3 → 4 * Real.pi * ((Real.sqrt ((a^2 + b^2 + c^2)) / 2) ^ 2) = 50 * Real.pi :=
by
  -- introduction of variables and conditions
  intros a b c h
  obtain ⟨_, _, _⟩ := h -- decomposing the conditions
  -- the proof is skipped
  sorry

end NUMINAMATH_GPT_sphere_surface_area_of_circumscribing_cuboid_l329_32975


namespace NUMINAMATH_GPT_polar_to_rectangular_coordinates_l329_32930

noncomputable def rectangular_coordinates_from_polar (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_coordinates :
  rectangular_coordinates_from_polar 12 (5 * Real.pi / 4) = (-6 * Real.sqrt 2, -6 * Real.sqrt 2) :=
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_coordinates_l329_32930


namespace NUMINAMATH_GPT_math_problem_proof_l329_32977

-- Define the problem statement
def problem_expr : ℕ :=
  28 * 7 * 25 + 12 * 7 * 25 + 7 * 11 * 3 + 44

-- Prove the problem statement equals to the correct answer
theorem math_problem_proof : problem_expr = 7275 := by
  sorry

end NUMINAMATH_GPT_math_problem_proof_l329_32977


namespace NUMINAMATH_GPT_parallel_vectors_m_eq_neg3_l329_32904

theorem parallel_vectors_m_eq_neg3
  (m : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h1 : a = (m + 1, -3))
  (h2 : b = (2, 3))
  (h3 : ∃ k : ℝ, a = (k * b.1, k * b.2)) :
  m = -3 := 
sorry

end NUMINAMATH_GPT_parallel_vectors_m_eq_neg3_l329_32904


namespace NUMINAMATH_GPT_pause_point_l329_32919

-- Definitions
def total_movie_length := 60 -- In minutes
def remaining_time := 30 -- In minutes

-- Theorem stating the pause point in the movie
theorem pause_point : total_movie_length - remaining_time = 30 := by
  -- This is the original solution in mathematical terms, omitted in lean statement.
  -- total_movie_length - remaining_time = 60 - 30 = 30
  sorry

end NUMINAMATH_GPT_pause_point_l329_32919


namespace NUMINAMATH_GPT_calculate_expr_l329_32957

theorem calculate_expr : (125 : ℝ)^(2/3) * 2 = 50 := sorry

end NUMINAMATH_GPT_calculate_expr_l329_32957


namespace NUMINAMATH_GPT_samuel_faster_than_sarah_l329_32970

-- Definitions based on the conditions
def time_samuel : ℝ := 30
def time_sarah : ℝ := 1.3 * 60

-- The theorem to prove that Samuel finished his homework 48 minutes faster than Sarah
theorem samuel_faster_than_sarah : (time_sarah - time_samuel) = 48 := by
  sorry

end NUMINAMATH_GPT_samuel_faster_than_sarah_l329_32970


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l329_32990

theorem quadratic_inequality_solution (x : ℝ) : (-x^2 + 5 * x - 4 < 0) ↔ (1 < x ∧ x < 4) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l329_32990


namespace NUMINAMATH_GPT_movie_theater_attendance_l329_32909

theorem movie_theater_attendance : 
  let total_seats := 750
  let empty_seats := 218
  let people := total_seats - empty_seats
  people = 532 :=
by
  sorry

end NUMINAMATH_GPT_movie_theater_attendance_l329_32909


namespace NUMINAMATH_GPT_min_value_of_fractions_l329_32989

theorem min_value_of_fractions (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
    (a+b)/(c+d) + (a+c)/(b+d) + (a+d)/(b+c) + (b+c)/(a+d) + (b+d)/(a+c) + (c+d)/(a+b) ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_fractions_l329_32989


namespace NUMINAMATH_GPT_minimize_prod_time_l329_32971

noncomputable def shortest_production_time
  (items : ℕ) 
  (workers : ℕ) 
  (shaping_time : ℕ) 
  (firing_time : ℕ) : ℕ := by
  sorry

-- The main theorem statement
theorem minimize_prod_time
  (items : ℕ := 75)
  (workers : ℕ := 13)
  (shaping_time : ℕ := 15)
  (drying_time : ℕ := 10)
  (firing_time : ℕ := 30)
  (optimal_time : ℕ := 325) :
  shortest_production_time items workers shaping_time firing_time = optimal_time := by
  sorry

end NUMINAMATH_GPT_minimize_prod_time_l329_32971


namespace NUMINAMATH_GPT_readers_of_science_fiction_l329_32952

variable (Total S L B : Nat)

theorem readers_of_science_fiction 
  (h1 : Total = 400) 
  (h2 : L = 230) 
  (h3 : B = 80) 
  (h4 : Total = S + L - B) : 
  S = 250 := 
by
  sorry

end NUMINAMATH_GPT_readers_of_science_fiction_l329_32952


namespace NUMINAMATH_GPT_kim_average_increase_l329_32915

noncomputable def avg (scores : List ℚ) : ℚ :=
  (scores.sum) / (scores.length)

theorem kim_average_increase :
  let scores_initial := [85, 89, 90, 92]  -- Initial scores
  let score_fifth := 95  -- Fifth score
  let original_average := avg scores_initial
  let new_average := avg (scores_initial ++ [score_fifth])
  new_average - original_average = 1.2 := by
  let scores_initial : List ℚ := [85, 89, 90, 92]
  let score_fifth : ℚ := 95
  let original_average : ℚ := avg scores_initial
  let new_average : ℚ := avg (scores_initial ++ [score_fifth])
  have : new_average - original_average = 1.2 := sorry
  exact this

end NUMINAMATH_GPT_kim_average_increase_l329_32915


namespace NUMINAMATH_GPT_total_winter_clothing_l329_32983

def first_box_items : Nat := 3 + 5 + 2
def second_box_items : Nat := 4 + 3 + 1
def third_box_items : Nat := 2 + 6 + 3
def fourth_box_items : Nat := 1 + 7 + 2

theorem total_winter_clothing : first_box_items + second_box_items + third_box_items + fourth_box_items = 39 := by
  sorry

end NUMINAMATH_GPT_total_winter_clothing_l329_32983


namespace NUMINAMATH_GPT_integer_not_natural_l329_32934

theorem integer_not_natural (n : ℕ) (a : ℝ) (b : ℝ) (x y z : ℝ) 
  (h₁ : x = (1 + a) ^ n) 
  (h₂ : y = (1 - a) ^ n) 
  (h₃ : z = a): 
  ∃ k : ℤ, (x - y) / z = ↑k ∧ (k < 0 ∨ k ≠ 0) :=
by 
  sorry

end NUMINAMATH_GPT_integer_not_natural_l329_32934


namespace NUMINAMATH_GPT_roots_of_quadratic_l329_32932

theorem roots_of_quadratic (x : ℝ) : 3 * (x - 3) = (x - 3) ^ 2 → x = 3 ∨ x = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l329_32932


namespace NUMINAMATH_GPT_convert_quadratic_l329_32920

theorem convert_quadratic (x : ℝ) :
  (1 + 3 * x) * (x - 3) = 2 * x ^ 2 + 1 ↔ x ^ 2 - 8 * x - 4 = 0 := 
by sorry

end NUMINAMATH_GPT_convert_quadratic_l329_32920


namespace NUMINAMATH_GPT_center_of_circle_in_second_quadrant_l329_32933

theorem center_of_circle_in_second_quadrant (a : ℝ) (h : a > 12) :
  ∃ x y : ℝ, x^2 + y^2 + a * x - 2 * a * y + a^2 + 3 * a = 0 ∧ (-a / 2, a).2 > 0 ∧ (-a / 2, a).1 < 0 :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_in_second_quadrant_l329_32933


namespace NUMINAMATH_GPT_arithmetic_sequence_inequality_l329_32972

variable {α : Type*} [OrderedRing α]

theorem arithmetic_sequence_inequality 
  (a : ℕ → α) (d : α) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_pos : ∀ n, a n > 0)
  (h_d_ne_zero : d ≠ 0) : 
  a 0 * a 7 < a 3 * a 4 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_inequality_l329_32972


namespace NUMINAMATH_GPT_calculation1_calculation2_calculation3_calculation4_l329_32951

theorem calculation1 : 72 * 54 + 28 * 54 = 5400 := 
by sorry

theorem calculation2 : 60 * 25 * 8 = 12000 := 
by sorry

theorem calculation3 : 2790 / (250 * 12 - 2910) = 31 := 
by sorry

theorem calculation4 : (100 - 1456 / 26) * 78 = 3432 := 
by sorry

end NUMINAMATH_GPT_calculation1_calculation2_calculation3_calculation4_l329_32951


namespace NUMINAMATH_GPT_servings_per_day_l329_32912

-- Definitions based on the given problem conditions
def serving_size : ℚ := 0.5
def container_size : ℚ := 32 - 2 -- 1 quart is 32 ounces and the jar is 2 ounces less
def days_last : ℕ := 20

-- The theorem statement to prove
theorem servings_per_day (h1 : serving_size = 0.5) (h2 : container_size = 30) (h3 : days_last = 20) :
  (container_size / days_last) / serving_size = 3 :=
by
  sorry

end NUMINAMATH_GPT_servings_per_day_l329_32912


namespace NUMINAMATH_GPT_find_t_l329_32964

theorem find_t (t : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - t| + |5 - x|) (h2 : ∃ x, f x = 3) : t = 2 ∨ t = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l329_32964


namespace NUMINAMATH_GPT_non_increasing_condition_l329_32900

variable {a b : ℝ} (f : ℝ → ℝ)

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem non_increasing_condition (h₀ : ∀ x y, a ≤ x → x < y → y ≤ b → ¬ (f x > f y)) :
  ¬ increasing_on_interval f a b :=
by
  intro h1
  have : ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y := h1
  exact sorry

end NUMINAMATH_GPT_non_increasing_condition_l329_32900


namespace NUMINAMATH_GPT_polygon_sides_l329_32993

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) (h2 : n > 2) : n = 8 := by
  -- Conditions given:
  -- h1: (n - 2) * 180 = 3 * 360
  -- h2: n > 2
  sorry

end NUMINAMATH_GPT_polygon_sides_l329_32993


namespace NUMINAMATH_GPT_ratio_of_ages_l329_32968

theorem ratio_of_ages (x m : ℕ) 
  (mother_current_age : ℕ := 41) 
  (daughter_current_age : ℕ := 23) 
  (age_diff : ℕ := mother_current_age - daughter_current_age) 
  (eq : (mother_current_age - x) = m * (daughter_current_age - x)) : 
  (41 - x) / (23 - x) = m :=
by
  -- Proof not required
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l329_32968


namespace NUMINAMATH_GPT_hoursWorkedPerDay_l329_32962

-- Define the conditions
def widgetsPerHour := 20
def daysPerWeek := 5
def totalWidgetsPerWeek := 800

-- Theorem statement
theorem hoursWorkedPerDay : (totalWidgetsPerWeek / widgetsPerHour) / daysPerWeek = 8 := 
  sorry

end NUMINAMATH_GPT_hoursWorkedPerDay_l329_32962


namespace NUMINAMATH_GPT_gopi_servant_salary_l329_32984

theorem gopi_servant_salary (S : ℕ) (turban_price : ℕ) (cash_received : ℕ) (months_worked : ℕ) (total_months : ℕ) :
  turban_price = 70 →
  cash_received = 50 →
  months_worked = 9 →
  total_months = 12 →
  S = 160 :=
by
  sorry

end NUMINAMATH_GPT_gopi_servant_salary_l329_32984


namespace NUMINAMATH_GPT_intersection_eq_two_l329_32956

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-2, 2}

theorem intersection_eq_two : A ∩ B = {2} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_two_l329_32956


namespace NUMINAMATH_GPT_choose_team_captains_l329_32966

open Nat

def binom (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem choose_team_captains :
  let total_members := 15
  let shortlisted := 5
  let regular := total_members - shortlisted
  binom total_members 4 - binom regular 4 = 1155 :=
by
  sorry

end NUMINAMATH_GPT_choose_team_captains_l329_32966


namespace NUMINAMATH_GPT_range_of_abs_function_l329_32941

theorem range_of_abs_function : ∀ (y : ℝ), (∃ (x : ℝ), y = |x + 5| - |x - 3|) ↔ y ∈ Set.Icc (-8) 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_abs_function_l329_32941


namespace NUMINAMATH_GPT_total_dolls_combined_l329_32944

-- Define the number of dolls for Vera
def vera_dolls : ℕ := 20

-- Define the relationship that Sophie has twice as many dolls as Vera
def sophie_dolls : ℕ := 2 * vera_dolls

-- Define the relationship that Aida has twice as many dolls as Sophie
def aida_dolls : ℕ := 2 * sophie_dolls

-- The statement to prove that the total number of dolls is 140
theorem total_dolls_combined : aida_dolls + sophie_dolls + vera_dolls = 140 :=
by
  sorry

end NUMINAMATH_GPT_total_dolls_combined_l329_32944


namespace NUMINAMATH_GPT_joshua_final_bottle_caps_l329_32916

def initial_bottle_caps : ℕ := 150
def bought_bottle_caps : ℕ := 23
def given_away_bottle_caps : ℕ := 37

theorem joshua_final_bottle_caps : (initial_bottle_caps + bought_bottle_caps - given_away_bottle_caps) = 136 := by
  sorry

end NUMINAMATH_GPT_joshua_final_bottle_caps_l329_32916


namespace NUMINAMATH_GPT_cannot_form_equilateral_triangle_from_spliced_isosceles_right_triangles_l329_32923

/- Definitions -/
def is_isosceles_right_triangle (triangle : Type) (a b c : ℝ) (angleA angleB angleC : ℝ) : Prop :=
  -- A triangle is isosceles right triangle if it has two equal angles of 45 degrees and a right angle of 90 degrees
  a = b ∧ angleA = 45 ∧ angleB = 45 ∧ angleC = 90

/- Main Problem Statement -/
theorem cannot_form_equilateral_triangle_from_spliced_isosceles_right_triangles
  (T1 T2 : Type) (a1 b1 c1 a2 b2 c2 : ℝ) 
  (angleA1 angleB1 angleC1 angleA2 angleB2 angleC2 : ℝ) :
  is_isosceles_right_triangle T1 a1 b1 c1 angleA1 angleB1 angleC1 →
  is_isosceles_right_triangle T2 a2 b2 c2 angleA2 angleB2 angleC2 →
  ¬ (∃ (a b c : ℝ), a = b ∧ b = c ∧ a = c ∧ (a + b + c = 180)) :=
by
  intros hT1 hT2
  intro h
  sorry

end NUMINAMATH_GPT_cannot_form_equilateral_triangle_from_spliced_isosceles_right_triangles_l329_32923


namespace NUMINAMATH_GPT_quad_inequality_necessary_but_not_sufficient_l329_32929

def quad_inequality (x : ℝ) : Prop := x^2 - x - 6 > 0
def less_than_negative_five (x : ℝ) : Prop := x < -5

theorem quad_inequality_necessary_but_not_sufficient :
  (∀ x : ℝ, less_than_negative_five x → quad_inequality x) ∧ 
  (∃ x : ℝ, quad_inequality x ∧ ¬ less_than_negative_five x) :=
by
  sorry

end NUMINAMATH_GPT_quad_inequality_necessary_but_not_sufficient_l329_32929


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l329_32905

noncomputable def b (n : ℕ) : ℤ := sorry -- define the arithmetic sequence

theorem arithmetic_sequence_product (d : ℤ) 
  (h_seq : ∀ n, b (n + 1) = b n + d)
  (h_inc : ∀ m n, m < n → b m < b n)
  (h_prod : b 4 * b 5 = 30) :
  b 3 * b 6 = -1652 ∨ b 3 * b 6 = -308 ∨ b 3 * b 6 = -68 ∨ b 3 * b 6 = 28 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l329_32905


namespace NUMINAMATH_GPT_polynomial_expansion_proof_l329_32917

variable (z : ℤ)

-- Define the polynomials p and q
noncomputable def p (z : ℤ) : ℤ := 3 * z^2 - 4 * z + 1
noncomputable def q (z : ℤ) : ℤ := 2 * z^3 + 3 * z^2 - 5 * z + 2

-- Define the expanded polynomial
noncomputable def expanded (z : ℤ) : ℤ :=
  6 * z^5 + z^4 - 25 * z^3 + 29 * z^2 - 13 * z + 2

-- The goal is to prove the equivalence of (p * q) == expanded 
theorem polynomial_expansion_proof :
  (p z) * (q z) = expanded z :=
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_proof_l329_32917


namespace NUMINAMATH_GPT_correct_match_results_l329_32959

-- Define the teams in the league
inductive Team
| Scotland : Team
| England  : Team
| Wales    : Team
| Ireland  : Team

-- Define a match result for a pair of teams
structure MatchResult where
  team1 : Team
  team2 : Team
  goals1 : ℕ
  goals2 : ℕ

def scotland_vs_england : MatchResult := {
  team1 := Team.Scotland,
  team2 := Team.England,
  goals1 := 3,
  goals2 := 0
}

-- All possible match results
def england_vs_ireland : MatchResult := {
  team1 := Team.England,
  team2 := Team.Ireland,
  goals1 := 1,
  goals2 := 0
}

def wales_vs_england : MatchResult := {
  team1 := Team.Wales,
  team2 := Team.England,
  goals1 := 1,
  goals2 := 1
}

def wales_vs_ireland : MatchResult := {
  team1 := Team.Wales,
  team2 := Team.Ireland,
  goals1 := 2,
  goals2 := 1
}

def scotland_vs_ireland : MatchResult := {
  team1 := Team.Scotland,
  team2 := Team.Ireland,
  goals1 := 2,
  goals2 := 0
}

theorem correct_match_results : 
  (england_vs_ireland.goals1 = 1 ∧ england_vs_ireland.goals2 = 0) ∧
  (wales_vs_england.goals1 = 1 ∧ wales_vs_england.goals2 = 1) ∧
  (scotland_vs_england.goals1 = 3 ∧ scotland_vs_england.goals2 = 0) ∧
  (wales_vs_ireland.goals1 = 2 ∧ wales_vs_ireland.goals2 = 1) ∧
  (scotland_vs_ireland.goals1 = 2 ∧ scotland_vs_ireland.goals2 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_correct_match_results_l329_32959


namespace NUMINAMATH_GPT_miniature_model_to_actual_statue_scale_l329_32994

theorem miniature_model_to_actual_statue_scale (height_actual : ℝ) (height_model : ℝ) : 
  height_actual = 90 → height_model = 6 → 
  (height_actual / height_model = 15) := 
by
  intros h_actual h_model
  rw [h_actual, h_model]
  sorry

end NUMINAMATH_GPT_miniature_model_to_actual_statue_scale_l329_32994


namespace NUMINAMATH_GPT_Ian_hours_worked_l329_32997

theorem Ian_hours_worked (money_left: ℝ) (hourly_rate: ℝ) (spent: ℝ) (earned: ℝ) (hours: ℝ) :
  money_left = 72 → hourly_rate = 18 → spent = earned / 2 → earned = money_left * 2 → 
  earned = hourly_rate * hours → hours = 8 :=
by
  intros h1 h2 h3 h4 h5
  -- Begin mathematical validation process here
  sorry

end NUMINAMATH_GPT_Ian_hours_worked_l329_32997


namespace NUMINAMATH_GPT_find_c_l329_32931

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_c (c : ℝ) (h1 : f 1 = 1) (h2 : ∀ x y : ℝ, f (x + y) = f x + f y + 8 * x * y - c) (h3 : f 7 = 163) :
  c = 2 / 3 :=
sorry

end NUMINAMATH_GPT_find_c_l329_32931


namespace NUMINAMATH_GPT_skylar_current_age_l329_32901

noncomputable def skylar_age_now (donation_start_age : ℕ) (annual_donation total_donation : ℕ) : ℕ :=
  donation_start_age + total_donation / annual_donation

theorem skylar_current_age : skylar_age_now 13 5000 105000 = 34 := by
  -- Proof follows from the conditions
  sorry

end NUMINAMATH_GPT_skylar_current_age_l329_32901


namespace NUMINAMATH_GPT_tailor_trim_length_l329_32928

theorem tailor_trim_length (x : ℕ) : 
  (18 - x) * 15 = 120 → x = 10 := 
by
  sorry

end NUMINAMATH_GPT_tailor_trim_length_l329_32928


namespace NUMINAMATH_GPT_polygon_perimeter_l329_32999

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end NUMINAMATH_GPT_polygon_perimeter_l329_32999


namespace NUMINAMATH_GPT_xiaoliang_prob_correct_l329_32906

def initial_box_setup : List (Nat × Nat) := [(1, 2), (2, 2), (3, 2), (4, 2)]

def xiaoming_draw : List Nat := [1, 1, 3]

def remaining_balls_after_xiaoming : List (Nat × Nat) := [(1, 0), (2, 2), (3, 1), (4, 2)]

def remaining_ball_count (balls : List (Nat × Nat)) : Nat :=
  balls.foldl (λ acc ⟨_, count⟩ => acc + count) 0

theorem xiaoliang_prob_correct :
  (1 : ℚ) / (remaining_ball_count remaining_balls_after_xiaoming) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_xiaoliang_prob_correct_l329_32906
