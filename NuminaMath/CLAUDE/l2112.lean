import Mathlib

namespace job_completion_time_l2112_211208

/-- Given that A can complete a job in 10 hours alone and A and D together can complete a job in 5 hours, prove that D can complete the job in 10 hours alone. -/
theorem job_completion_time (a_time : ℝ) (ad_time : ℝ) (d_time : ℝ) 
    (ha : a_time = 10) 
    (had : ad_time = 5) : 
  d_time = 10 := by
  sorry

end job_completion_time_l2112_211208


namespace minimal_area_circle_circle_center_on_line_l2112_211209

-- Define the points A and B
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (-2, -5)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define a circle passing through two points
def circle_through_points (center : ℝ × ℝ) (r : ℝ) : Prop :=
  (center.1 - A.1)^2 + (center.2 - A.2)^2 = r^2 ∧
  (center.1 - B.1)^2 + (center.2 - B.2)^2 = r^2

-- Theorem for minimal area circle
theorem minimal_area_circle :
  ∀ (center : ℝ × ℝ) (r : ℝ),
  circle_through_points center r →
  (∀ (center' : ℝ × ℝ) (r' : ℝ), circle_through_points center' r' → r ≤ r') →
  center = (0, -4) ∧ r^2 = 5 :=
sorry

-- Theorem for circle with center on the line
theorem circle_center_on_line :
  ∀ (center : ℝ × ℝ) (r : ℝ),
  circle_through_points center r →
  line_eq center.1 center.2 →
  center = (-1, -2) ∧ r^2 = 10 :=
sorry

end minimal_area_circle_circle_center_on_line_l2112_211209


namespace max_product_distances_l2112_211270

/-- Two perpendicular lines passing through points A and B, intersecting at P -/
structure PerpendicularLines where
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ
  perpendicular : (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

/-- The maximum value of |PA| * |PB| for perpendicular lines through A(0, 0) and B(1, 3) -/
theorem max_product_distances (l : PerpendicularLines) 
  (h_A : l.A = (0, 0)) (h_B : l.B = (1, 3)) : 
  ∃ (max : ℝ), ∀ (P : ℝ × ℝ), 
    ((P.1 - l.A.1)^2 + (P.2 - l.A.2)^2) * ((P.1 - l.B.1)^2 + (P.2 - l.B.2)^2) ≤ max^2 ∧ 
    max = 5 :=
sorry

end max_product_distances_l2112_211270


namespace not_good_pair_3_3_l2112_211276

/-- A pair of natural numbers is good if there exists a polynomial with integer coefficients and distinct integers satisfying certain conditions. -/
def is_good_pair (r s : ℕ) : Prop :=
  ∃ (P : ℤ → ℤ) (a b : Fin r → ℤ) (c d : Fin s → ℤ),
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    (∀ i j, i ≠ j → c i ≠ c j) ∧
    (∀ i j, a i ≠ c j) ∧
    (∀ i, P (a i) = 2) ∧
    (∀ i, P (c i) = 5) ∧
    (∀ x y : ℤ, (x - y) ∣ (P x - P y))

/-- Theorem stating that (3, 3) is not a good pair. -/
theorem not_good_pair_3_3 : ¬ is_good_pair 3 3 := by
  sorry

end not_good_pair_3_3_l2112_211276


namespace hyperbola_intersection_theorem_l2112_211284

/-- The hyperbola C with real semi-axis length √3 and the same foci as the ellipse x²/8 + y²/4 = 1 -/
structure Hyperbola where
  /-- The real semi-axis length of the hyperbola -/
  a : ℝ
  /-- The imaginary semi-axis length of the hyperbola -/
  b : ℝ
  /-- The focal distance of the hyperbola -/
  c : ℝ
  /-- The real semi-axis length is √3 -/
  ha : a = Real.sqrt 3
  /-- The focal distance is the same as the ellipse x²/8 + y²/4 = 1 -/
  hc : c = 2
  /-- Relation between a, b, and c in a hyperbola -/
  hab : c^2 = a^2 + b^2

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The line intersecting the hyperbola -/
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + Real.sqrt 2

/-- The dot product of two points with the origin -/
def dot_product (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  x₁ * x₂ + y₁ * y₂

/-- The main theorem -/
theorem hyperbola_intersection_theorem (h : Hyperbola) :
  (∀ x y, hyperbola_equation h x y ↔ x^2 / 3 - y^2 = 1) ∧
  (∀ k, (∃ x₁ y₁ x₂ y₂, 
    x₁ ≠ x₂ ∧
    hyperbola_equation h x₁ y₁ ∧
    hyperbola_equation h x₂ y₂ ∧
    line_equation k x₁ y₁ ∧
    line_equation k x₂ y₂ ∧
    dot_product x₁ y₁ x₂ y₂ > 2) ↔
   (k > -1 ∧ k < -Real.sqrt 3 / 3) ∨ (k > Real.sqrt 3 / 3 ∧ k < 1)) :=
sorry

end hyperbola_intersection_theorem_l2112_211284


namespace pyramid_frustum_volume_l2112_211283

/-- Given a square pyramid with base edge s and altitude h, 
    if a smaller similar pyramid with altitude h/3 is removed from the apex, 
    the volume of the remaining frustum is 26/27 of the original pyramid's volume. -/
theorem pyramid_frustum_volume 
  (s h : ℝ) 
  (h_pos : 0 < h) 
  (s_pos : 0 < s) : 
  let v_original := (1 / 3) * s^2 * h
  let v_smaller := (1 / 3) * (s / 3)^2 * (h / 3)
  let v_frustum := v_original - v_smaller
  v_frustum = (26 / 27) * v_original := by
  sorry

end pyramid_frustum_volume_l2112_211283


namespace sum_of_roots_is_zero_l2112_211254

theorem sum_of_roots_is_zero (x : ℝ) :
  (x^2 - 7*|x| + 6 = 0) →
  (∃ x₁ x₂ x₃ x₄ : ℝ,
    ((x₁ > 0 ∧ x₂ > 0 ∧ x₁^2 - 7*x₁ + 6 = 0 ∧ x₂^2 - 7*x₂ + 6 = 0) ∨
     (x₃ < 0 ∧ x₄ < 0 ∧ x₃^2 + 7*x₃ + 6 = 0 ∧ x₄^2 + 7*x₄ + 6 = 0)) ∧
    x₁ + x₂ + x₃ + x₄ = 0) :=
by sorry

end sum_of_roots_is_zero_l2112_211254


namespace normal_distribution_symmetry_l2112_211235

-- Define the random variable ξ with normal distribution N(2,9)
def ξ : Real → Real := sorry

-- Define the probability density function for ξ
def pdf_ξ (x : Real) : Real := sorry

-- Define the cumulative distribution function for ξ
def cdf_ξ (x : Real) : Real := sorry

-- State the theorem
theorem normal_distribution_symmetry (c : Real) :
  (∀ x, pdf_ξ x = 1 / (3 * Real.sqrt (2 * Real.pi)) * Real.exp (-(x - 2)^2 / (2 * 9))) →
  (cdf_ξ (c - 1) = 1 - cdf_ξ (c + 3)) →
  c = 1 := by sorry

end normal_distribution_symmetry_l2112_211235


namespace andrew_mango_purchase_l2112_211287

/-- Calculates the quantity of mangoes purchased given the total amount paid,
    grape quantity, grape price, and mango price. -/
def mango_quantity (total_paid : ℕ) (grape_quantity : ℕ) (grape_price : ℕ) (mango_price : ℕ) : ℕ :=
  (total_paid - grape_quantity * grape_price) / mango_price

theorem andrew_mango_purchase :
  mango_quantity 908 7 68 48 = 9 := by
  sorry

end andrew_mango_purchase_l2112_211287


namespace partners_shares_correct_l2112_211273

/-- Represents the investment ratio of partners A, B, and C -/
def investment_ratio : Fin 3 → ℕ
| 0 => 2  -- Partner A
| 1 => 3  -- Partner B
| 2 => 5  -- Partner C

/-- The total profit in rupees -/
def total_profit : ℕ := 22400

/-- Calculates a partner's share of the profit based on their investment ratio -/
def partner_share (i : Fin 3) : ℕ :=
  (investment_ratio i * total_profit) / (investment_ratio 0 + investment_ratio 1 + investment_ratio 2)

/-- Theorem stating that the partners' shares are correct -/
theorem partners_shares_correct :
  partner_share 0 = 4480 ∧
  partner_share 1 = 6720 ∧
  partner_share 2 = 11200 := by
  sorry


end partners_shares_correct_l2112_211273


namespace bingley_has_six_bracelets_l2112_211246

/-- The number of bracelets Bingley has remaining after exchanges with Kelly and his sister -/
def bingleysRemainingBracelets (bingleyInitial : ℕ) (kellyInitial : ℕ) : ℕ :=
  let bingleyAfterKelly := bingleyInitial + kellyInitial / 4
  bingleyAfterKelly - bingleyAfterKelly / 3

/-- Theorem stating that Bingley has 6 bracelets remaining -/
theorem bingley_has_six_bracelets :
  bingleysRemainingBracelets 5 16 = 6 := by
  sorry

end bingley_has_six_bracelets_l2112_211246


namespace carlas_daily_collection_l2112_211257

/-- Represents the number of items Carla needs to collect each day for her project -/
def daily_collection_amount (leaves bugs days : ℕ) : ℕ :=
  (leaves + bugs) / days

/-- Proves that Carla needs to collect 5 items per day given the project conditions -/
theorem carlas_daily_collection :
  daily_collection_amount 30 20 10 = 5 := by
  sorry

#eval daily_collection_amount 30 20 10

end carlas_daily_collection_l2112_211257


namespace repeating_decimal_length_1_221_l2112_211280

theorem repeating_decimal_length_1_221 : ∃ n : ℕ, n > 0 ∧ n = 48 ∧ ∀ k : ℕ, (10^k - 1) % 221 = 0 ↔ n ∣ k := by sorry

end repeating_decimal_length_1_221_l2112_211280


namespace integer_solution_theorem_l2112_211215

theorem integer_solution_theorem (x y z w : ℤ) :
  (x * y * z / w : ℚ) + (y * z * w / x : ℚ) + (z * w * x / y : ℚ) + (w * x * y / z : ℚ) = 4 →
  ((x = 1 ∧ y = 1 ∧ z = 1 ∧ w = 1) ∨
   (x = -1 ∧ y = -1 ∧ z = -1 ∧ w = -1) ∨
   (x = -1 ∧ y = -1 ∧ z = 1 ∧ w = 1) ∨
   (x = -1 ∧ y = 1 ∧ z = -1 ∧ w = 1) ∨
   (x = -1 ∧ y = 1 ∧ z = 1 ∧ w = -1) ∨
   (x = 1 ∧ y = -1 ∧ z = -1 ∧ w = 1) ∨
   (x = 1 ∧ y = -1 ∧ z = 1 ∧ w = -1) ∨
   (x = 1 ∧ y = 1 ∧ z = -1 ∧ w = -1)) :=
by sorry

end integer_solution_theorem_l2112_211215


namespace quadratic_derivative_condition_l2112_211212

/-- Given a quadratic function f(x) = 3x² + bx + c, prove that if the derivative at x = b is 14, then b = 2 -/
theorem quadratic_derivative_condition (b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 + b * x + c
  (∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f (b + Δx) - f b) / Δx) - 14| < ε) → 
  b = 2 := by
sorry

end quadratic_derivative_condition_l2112_211212


namespace profit_per_meter_of_cloth_l2112_211275

/-- Profit per meter of cloth calculation -/
theorem profit_per_meter_of_cloth
  (total_meters : ℕ)
  (selling_price : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : total_meters = 45)
  (h2 : selling_price = 4500)
  (h3 : cost_price_per_meter = 86) :
  (selling_price - total_meters * cost_price_per_meter) / total_meters = 14 := by
sorry

end profit_per_meter_of_cloth_l2112_211275


namespace distance_and_midpoint_l2112_211223

/-- Given two points in a 2D plane, calculate their distance and midpoint -/
theorem distance_and_midpoint (p1 p2 : ℝ × ℝ) : 
  p1 = (2, 3) → p2 = (5, 9) → 
  (∃ (d : ℝ), d = 3 * Real.sqrt 5 ∧ d = Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)) ∧ 
  (∃ (m : ℝ × ℝ), m = (3.5, 6) ∧ m = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)) := by
  sorry

end distance_and_midpoint_l2112_211223


namespace aitana_jayda_spending_l2112_211218

theorem aitana_jayda_spending (jayda_spent : ℚ) (total_spent : ℚ) 
  (h1 : jayda_spent = 400)
  (h2 : total_spent = 960) : 
  (total_spent - jayda_spent) / jayda_spent = 2/5 := by
  sorry

end aitana_jayda_spending_l2112_211218


namespace fair_distribution_exists_l2112_211267

/-- Represents a piece of ham with its value according to the store scale -/
structure HamPiece where
  value : ℕ

/-- Represents a woman and her belief about the ham's value -/
inductive Woman
  | TrustsHomeScales
  | TrustsStoreScales
  | BelievesEqual

/-- Represents the distribution of ham pieces to women -/
def Distribution := Woman → HamPiece

/-- Checks if a distribution is fair according to each woman's belief -/
def is_fair_distribution (d : Distribution) : Prop :=
  (d Woman.TrustsHomeScales).value ≥ 15 ∧
  (d Woman.TrustsStoreScales).value ≥ 15 ∧
  (d Woman.BelievesEqual).value > 0

/-- The main theorem stating that a fair distribution exists -/
theorem fair_distribution_exists : ∃ (d : Distribution), is_fair_distribution d := by
  sorry

end fair_distribution_exists_l2112_211267


namespace square_area_change_l2112_211295

theorem square_area_change (original_area : ℝ) (decrease_percent : ℝ) (increase_percent : ℝ) : 
  original_area = 625 →
  decrease_percent = 0.2 →
  increase_percent = 0.2 →
  let original_side : ℝ := Real.sqrt original_area
  let new_side1 : ℝ := original_side * (1 - decrease_percent)
  let new_side2 : ℝ := original_side * (1 + increase_percent)
  new_side1 * new_side2 = 600 := by
sorry

end square_area_change_l2112_211295


namespace power_simplification_l2112_211236

theorem power_simplification :
  ((5^13 / 5^11)^2 * 5^2) / 2^5 = 15625 / 32 := by
  sorry

end power_simplification_l2112_211236


namespace rectangular_solid_width_l2112_211201

/-- The surface area of a rectangular solid given its length, width, and depth. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The width of a rectangular solid with length 9 meters, depth 5 meters, 
    and surface area 314 square meters is 8 meters. -/
theorem rectangular_solid_width : 
  ∃ (w : ℝ), w = 8 ∧ surface_area 9 w 5 = 314 := by
  sorry

end rectangular_solid_width_l2112_211201


namespace fraction_subtraction_l2112_211217

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 8 = 3 / 56 := by
  sorry

end fraction_subtraction_l2112_211217


namespace max_value_at_negative_one_l2112_211216

-- Define a monic cubic polynomial
def monic_cubic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = x^3 + a*x^2 + b*x + c

-- Define the condition that all roots are non-negative
def non_negative_roots (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = 0 → x ≥ 0

-- Main theorem
theorem max_value_at_negative_one (f : ℝ → ℝ) :
  monic_cubic f →
  f 0 = -64 →
  non_negative_roots f →
  ∀ g : ℝ → ℝ, monic_cubic g → g 0 = -64 → non_negative_roots g →
  f (-1) ≤ -125 ∧ (∃ h : ℝ → ℝ, monic_cubic h ∧ h 0 = -64 ∧ non_negative_roots h ∧ h (-1) = -125) :=
by sorry

end max_value_at_negative_one_l2112_211216


namespace original_price_after_discount_l2112_211248

theorem original_price_after_discount (P : ℝ) : 
  P * (1 - 0.2) = P - 50 → P = 250 := by
  sorry

end original_price_after_discount_l2112_211248


namespace prime_value_of_polynomial_l2112_211290

theorem prime_value_of_polynomial (a : ℕ) :
  Nat.Prime (a^4 - 4*a^3 + 15*a^2 - 30*a + 27) →
  a^4 - 4*a^3 + 15*a^2 - 30*a + 27 = 11 :=
by sorry

end prime_value_of_polynomial_l2112_211290


namespace mp3_song_count_l2112_211245

theorem mp3_song_count (initial_songs : ℕ) (deleted_songs : ℕ) (added_songs : ℕ) 
  (h1 : initial_songs = 15)
  (h2 : deleted_songs = 8)
  (h3 : added_songs = 50) :
  initial_songs - deleted_songs + added_songs = 57 := by
  sorry

end mp3_song_count_l2112_211245


namespace odd_product_probability_l2112_211233

theorem odd_product_probability (n : ℕ) (h : n = 2020) :
  let total := n
  let odds := n / 2
  let p := (odds / total) * ((odds - 1) / (total - 1)) * ((odds - 2) / (total - 2))
  p < 1 / 8 := by
  sorry

end odd_product_probability_l2112_211233


namespace bin_game_expectation_l2112_211210

theorem bin_game_expectation (k : ℕ+) : 
  let total_balls : ℕ := 8 + k
  let green_prob : ℚ := 8 / total_balls
  let purple_prob : ℚ := k / total_balls
  let expected_value : ℚ := green_prob * 3 + purple_prob * (-1)
  expected_value = 60 / 100 → k = 12 := by
sorry

end bin_game_expectation_l2112_211210


namespace painting_time_theorem_l2112_211238

def painter_a_rate : ℝ := 50
def painter_b_rate : ℝ := 40
def painter_c_rate : ℝ := 30

def room_7_area : ℝ := 220
def room_8_area : ℝ := 320
def room_9_area : ℝ := 420
def room_10_area : ℝ := 270

def total_area : ℝ := room_7_area + room_8_area + room_9_area + room_10_area
def combined_rate : ℝ := painter_a_rate + painter_b_rate + painter_c_rate

theorem painting_time_theorem : 
  total_area / combined_rate = 10.25 := by sorry

end painting_time_theorem_l2112_211238


namespace hyperbola_eccentricity_l2112_211262

/-- The eccentricity of a hyperbola with specific intersection properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let Γ := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}
  let c := Real.sqrt (a^2 + b^2)
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  ∀ A B : ℝ × ℝ,
    A ∈ Γ → B ∈ Γ →
    (∃ t : ℝ, A = F₂ + t • (B - F₂)) →
    ‖A - F₁‖ = ‖F₁ - F₂‖ →
    ‖B - F₂‖ = 2 * ‖A - F₂‖ →
    c / a = 5 / 3 :=
by sorry

end hyperbola_eccentricity_l2112_211262


namespace integral_of_power_function_l2112_211285

theorem integral_of_power_function : 
  ∫ x in (0:ℝ)..2, (1 + 3*x)^4 = 1120.4 := by sorry

end integral_of_power_function_l2112_211285


namespace two_color_rectangle_exists_l2112_211291

/-- A color type with two possible values -/
inductive Color
| Red
| Blue

/-- A point in a 2D grid -/
structure Point where
  x : Nat
  y : Nat

/-- A coloring function that assigns a color to each point in a grid -/
def Coloring := Point → Color

/-- A rectangle defined by its four vertices -/
structure Rectangle where
  topLeft : Point
  topRight : Point
  bottomLeft : Point
  bottomRight : Point

/-- Predicate to check if all vertices of a rectangle have the same color -/
def sameColorRectangle (c : Coloring) (r : Rectangle) : Prop :=
  c r.topLeft = c r.topRight ∧
  c r.topLeft = c r.bottomLeft ∧
  c r.topLeft = c r.bottomRight

/-- Theorem stating that in any 7x3 grid colored with two colors,
    there exists a rectangle with vertices of the same color -/
theorem two_color_rectangle_exists :
  ∀ (c : Coloring),
  (∀ (p : Point), p.x < 7 ∧ p.y < 3 → (c p = Color.Red ∨ c p = Color.Blue)) →
  ∃ (r : Rectangle),
    r.topLeft.x < 7 ∧ r.topLeft.y < 3 ∧
    r.topRight.x < 7 ∧ r.topRight.y < 3 ∧
    r.bottomLeft.x < 7 ∧ r.bottomLeft.y < 3 ∧
    r.bottomRight.x < 7 ∧ r.bottomRight.y < 3 ∧
    sameColorRectangle c r :=
by
  sorry


end two_color_rectangle_exists_l2112_211291


namespace tailor_buttons_l2112_211294

theorem tailor_buttons (green : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : yellow = green + 10)
  (h2 : blue = green - 5)
  (h3 : green + yellow + blue = 275) :
  green = 90 := by
sorry

end tailor_buttons_l2112_211294


namespace jack_multiple_is_ten_l2112_211213

/-- The multiple of Michael's current trophies that Jack will have in three years -/
def jack_multiple (michael_current : ℕ) (michael_increase : ℕ) (total_after : ℕ) : ℕ :=
  (total_after - (michael_current + michael_increase)) / michael_current

theorem jack_multiple_is_ten :
  jack_multiple 30 100 430 = 10 := by sorry

end jack_multiple_is_ten_l2112_211213


namespace solution_ordered_pair_l2112_211230

theorem solution_ordered_pair : ∃ x y : ℝ, 
  (x + y = (7 - x) + (7 - y)) ∧ 
  (x - y = (x + 1) + (y + 1)) ∧
  x = 8 ∧ y = -1 := by
sorry

end solution_ordered_pair_l2112_211230


namespace zuzka_structure_bounds_l2112_211252

/-- A structure made of cubes -/
structure CubeStructure where
  base : Nat
  layers : Nat
  third_layer : Nat
  total : Nat

/-- The conditions of Zuzka's cube structure -/
def zuzka_structure (s : CubeStructure) : Prop :=
  s.base = 16 ∧ 
  s.layers ≥ 3 ∧ 
  s.third_layer = 2 ∧
  s.total = s.base + (s.layers - 1) * s.third_layer + (s.total - s.base - s.third_layer)

/-- The theorem stating the range of possible total cubes -/
theorem zuzka_structure_bounds (s : CubeStructure) :
  zuzka_structure s → 22 ≤ s.total ∧ s.total ≤ 27 :=
by
  sorry


end zuzka_structure_bounds_l2112_211252


namespace power_zero_equivalence_l2112_211263

theorem power_zero_equivalence (x : ℝ) (h : x ≠ 0) : x^0 = 1/(x^0) := by
  sorry

end power_zero_equivalence_l2112_211263


namespace train_crossing_time_l2112_211253

/-- Proves that a train 360 m long traveling at 43.2 km/h takes 30 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 360 →
  train_speed_kmh = 43.2 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 30 := by
  sorry

#check train_crossing_time

end train_crossing_time_l2112_211253


namespace sum_of_squared_medians_l2112_211214

/-- The sum of squares of medians in a triangle with sides 13, 14, and 15 --/
theorem sum_of_squared_medians (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let m_a := (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2)
  let m_b := (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2)
  let m_c := (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)
  m_a^2 + m_b^2 + m_c^2 = 442.5 := by
  sorry

#check sum_of_squared_medians

end sum_of_squared_medians_l2112_211214


namespace isosceles_triangle_side_length_l2112_211224

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the properties of the triangle
def isIsosceles (t : Triangle) : Prop :=
  dist t.A t.B = dist t.A t.C

def pointInside (t : Triangle) : Prop :=
  -- This is a simplified condition; in reality, we'd need a more complex definition
  true

-- Define the given distances
def givenDistances (t : Triangle) : Prop :=
  dist t.A t.P = 2 ∧
  dist t.B t.P = 2 * Real.sqrt 2 ∧
  dist t.C t.P = 3

-- Theorem statement
theorem isosceles_triangle_side_length (t : Triangle) :
  isIsosceles t → pointInside t → givenDistances t →
  dist t.B t.C = 2 * Real.sqrt 6 :=
sorry

end isosceles_triangle_side_length_l2112_211224


namespace mans_rowing_speed_l2112_211296

/-- The man's rowing speed in still water -/
def rowing_speed : ℝ := 3.9

/-- The speed of the current -/
def current_speed : ℝ := 1.3

/-- The ratio of time taken to row upstream compared to downstream -/
def time_ratio : ℝ := 2

theorem mans_rowing_speed :
  (rowing_speed + current_speed) * time_ratio = (rowing_speed - current_speed) * (time_ratio * 2) ∧
  rowing_speed = 3 * current_speed := by
  sorry

#check mans_rowing_speed

end mans_rowing_speed_l2112_211296


namespace parallel_vectors_cos_2theta_l2112_211298

/-- Given two parallel vectors a and b, prove that cos(2θ) = -1/3 -/
theorem parallel_vectors_cos_2theta (θ : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (Real.cos θ, 1)) 
  (hb : b = (1, 3 * Real.cos θ)) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a = k • b) : 
  Real.cos (2 * θ) = -1/3 := by
  sorry

end parallel_vectors_cos_2theta_l2112_211298


namespace right_triangle_complex_roots_l2112_211207

theorem right_triangle_complex_roots : 
  ∃! (S : Finset ℂ), 
    (∀ z ∈ S, z ≠ 0 ∧ 
      (z.re * (z^6 - z).re + z.im * (z^6 - z).im = 0)) ∧ 
    S.card = 5 := by sorry

end right_triangle_complex_roots_l2112_211207


namespace movie_ticket_cost_l2112_211271

theorem movie_ticket_cost 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (concession_cost : ℚ) 
  (total_cost : ℚ) 
  (child_ticket_cost : ℚ) : 
  num_adults = 5 → 
  num_children = 2 → 
  concession_cost = 12 → 
  total_cost = 76 → 
  child_ticket_cost = 7 → 
  (total_cost - concession_cost - num_children * child_ticket_cost) / num_adults = 10 :=
by sorry

end movie_ticket_cost_l2112_211271


namespace circus_cages_l2112_211250

theorem circus_cages (n : ℕ) (ways : ℕ) (h1 : n = 6) (h2 : ways = 240) :
  ∃ x : ℕ, x = 3 ∧ (n! / x! = ways) :=
by sorry

end circus_cages_l2112_211250


namespace sum_interior_angles_octagon_l2112_211247

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The sum of the interior angles of an octagon is 1080° -/
theorem sum_interior_angles_octagon :
  sum_interior_angles octagon_sides = 1080 := by
  sorry


end sum_interior_angles_octagon_l2112_211247


namespace qinJiushaoResult_l2112_211258

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qinJiushaoAlgorithm (n : ℕ) (x : ℕ) : ℕ :=
  let rec loop : ℕ → ℕ → ℕ
    | 0, v => v
    | i+1, v => loop i (x * v + 1)
  loop n 1

/-- Theorem stating the result of Qin Jiushao's algorithm for n=5 and x=2 -/
theorem qinJiushaoResult : qinJiushaoAlgorithm 5 2 = 2^5 + 2^4 + 2^3 + 2^2 + 2 + 1 := by
  sorry

#eval qinJiushaoAlgorithm 5 2

end qinJiushaoResult_l2112_211258


namespace quarter_circle_roll_path_length_l2112_211272

/-- The length of the path traveled by the center of a quarter-circle when rolled along a straight line -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 1 / Real.pi) : 
  let path_length := 2 * (r * Real.pi / 4) + r * Real.pi / 2
  path_length = 3 / 2 := by
  sorry

end quarter_circle_roll_path_length_l2112_211272


namespace muffin_mix_buyers_l2112_211232

/-- Given a set of buyers with specific purchasing patterns for cake and muffin mixes,
    prove that the number of buyers who purchase muffin mix is 40. -/
theorem muffin_mix_buyers (total : ℕ) (cake : ℕ) (both : ℕ) (neither_prob : ℚ) :
  total = 100 →
  cake = 50 →
  both = 16 →
  neither_prob = 26 / 100 →
  ∃ (muffin : ℕ),
    muffin = 40 ∧
    (cake + muffin - both : ℚ) = total - (neither_prob * total) :=
by sorry

end muffin_mix_buyers_l2112_211232


namespace divisibility_of_f_minus_p_l2112_211260

/-- 
For a number n in base 10, let f(n) be the sum of all numbers possible by removing some digits of n 
(including none and all). This theorem proves that for any 2011-digit integer p, f(p) - p is divisible by 9.
-/
theorem divisibility_of_f_minus_p (p : ℕ) (h : 10^2010 ≤ p ∧ p < 10^2011) : 
  ∃ k : ℤ, (2^2010 - 1) * p = 9 * k := by
  sorry

end divisibility_of_f_minus_p_l2112_211260


namespace polynomial_simplification_l2112_211286

theorem polynomial_simplification (x : ℝ) :
  (2 * x^5 - 3 * x^4 + x^3 + 5 * x^2 - 8 * x + 15) + (-5 * x^4 - 2 * x^3 + 3 * x^2 + 8 * x + 9) =
  2 * x^5 - 8 * x^4 - x^3 + 8 * x^2 + 24 := by
    sorry

end polynomial_simplification_l2112_211286


namespace equal_diagonal_polygon_l2112_211249

/-- A convex polygon -/
structure ConvexPolygon where
  vertices : ℕ
  is_convex : Bool
  diagonals_equal : Bool

/-- Definition of a quadrilateral -/
def is_quadrilateral (p : ConvexPolygon) : Prop :=
  p.vertices = 4

/-- Definition of a pentagon -/
def is_pentagon (p : ConvexPolygon) : Prop :=
  p.vertices = 5

/-- Main theorem -/
theorem equal_diagonal_polygon (F : ConvexPolygon) 
  (h1 : F.vertices ≥ 4) 
  (h2 : F.is_convex = true) 
  (h3 : F.diagonals_equal = true) : 
  is_quadrilateral F ∨ is_pentagon F :=
sorry

end equal_diagonal_polygon_l2112_211249


namespace angle_D_value_l2112_211292

-- Define the angles as real numbers
variable (A B C D E : ℝ)

-- State the given conditions
axiom angle_sum : A + B = 180
axiom angle_C_eq_D : C = D
axiom angle_A_value : A = 50
axiom angle_E_value : E = 60
axiom triangle1_sum : A + B + E = 180
axiom triangle2_sum : B + C + D = 180

-- State the theorem to be proved
theorem angle_D_value : D = 55 := by
  sorry

end angle_D_value_l2112_211292


namespace divisibility_of_n_squared_plus_n_plus_two_l2112_211282

theorem divisibility_of_n_squared_plus_n_plus_two :
  (∀ n : ℕ, 2 ∣ (n^2 + n + 2)) ∧
  (∃ n : ℕ, ¬(5 ∣ (n^2 + n + 2))) := by
  sorry

end divisibility_of_n_squared_plus_n_plus_two_l2112_211282


namespace expression_simplification_l2112_211261

theorem expression_simplification (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  (1 + 1/x) * (1 - 2/(x+1)) * (1 + 2/(x-1)) = (x+1)/x :=
by sorry

end expression_simplification_l2112_211261


namespace sufficient_condition_range_l2112_211264

theorem sufficient_condition_range (m : ℝ) : 
  (∀ x : ℝ, |x - 4| ≤ 6 → x ≤ 1 + m) ∧ 
  (∃ x : ℝ, x ≤ 1 + m ∧ |x - 4| > 6) → 
  m ≥ 9 := by sorry

end sufficient_condition_range_l2112_211264


namespace problem_statement_l2112_211219

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2023)^2 = 0) : a^b = -1 := by
  sorry

end problem_statement_l2112_211219


namespace condition_sufficient_not_necessary_l2112_211288

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, x > 4 → x ≥ 4) ∧
  (∃ x : ℝ, x ≥ 4 ∧ ¬(x > 4)) := by
  sorry

end condition_sufficient_not_necessary_l2112_211288


namespace yeri_change_correct_l2112_211274

def calculate_change (num_candies : ℕ) (candy_cost : ℕ) (num_chocolates : ℕ) (chocolate_cost : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_candies * candy_cost + num_chocolates * chocolate_cost)

theorem yeri_change_correct : 
  calculate_change 5 120 3 350 2500 = 850 := by
  sorry

end yeri_change_correct_l2112_211274


namespace product_of_sum_and_sum_of_squares_l2112_211203

theorem product_of_sum_and_sum_of_squares (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^2 + b^2 = 6) : 
  a * b = 5 := by
  sorry

end product_of_sum_and_sum_of_squares_l2112_211203


namespace angle_A1C1_B1C_is_60_degrees_l2112_211220

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Calculates the angle between two lines in 3D space -/
def angle_between_lines (p1 p2 p3 p4 : Point3D) : ℝ :=
  sorry

/-- Theorem: In a cube, the angle between A1C1 and B1C is 60 degrees -/
theorem angle_A1C1_B1C_is_60_degrees (cube : Cube) :
  angle_between_lines cube.A1 cube.C1 cube.B1 cube.C = 60 * π / 180 :=
sorry

end angle_A1C1_B1C_is_60_degrees_l2112_211220


namespace area_BCD_l2112_211268

-- Define the points A, B, C, D
variable (A B C D : ℝ × ℝ)

-- Define the conditions
variable (area_ABC : Real)
variable (length_AC : Real)
variable (length_CD : Real)

-- Axioms
axiom area_ABC_value : area_ABC = 45
axiom AC_length : length_AC = 10
axiom CD_length : length_CD = 30
axiom B_perpendicular_AD : (B.2 - A.2) * (D.1 - A.1) = (B.1 - A.1) * (D.2 - A.2)

-- Theorem to prove
theorem area_BCD (h : ℝ) : 
  area_ABC = 1/2 * length_AC * h → 
  1/2 * length_CD * h = 135 :=
sorry

end area_BCD_l2112_211268


namespace intersection_of_M_and_N_l2112_211200

-- Define the sets M and N
def M : Set ℝ := {x | x + 1 ≥ 0}
def N : Set ℝ := {x | x - 2 < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

end intersection_of_M_and_N_l2112_211200


namespace complex_magnitude_l2112_211259

theorem complex_magnitude (z : ℂ) : z = (2 - I) / (1 + I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_magnitude_l2112_211259


namespace infinite_divisors_of_power_plus_one_l2112_211234

theorem infinite_divisors_of_power_plus_one (a : ℕ) (h1 : a > 1) (h2 : Even a) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, n ∣ a^n + 1 := by
  sorry

end infinite_divisors_of_power_plus_one_l2112_211234


namespace scientific_notation_of_8500_billion_l2112_211269

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coefficient : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The value in yuan -/
def value : ℝ := 8500000000000

/-- The scientific notation representation of the value -/
def scientificForm : ScientificNotation := toScientificNotation value

/-- Theorem stating that the scientific notation of 8500 billion yuan is 8.5 × 10^11 -/
theorem scientific_notation_of_8500_billion :
  scientificForm.coefficient = 8.5 ∧ scientificForm.exponent = 11 := by
  sorry

end scientific_notation_of_8500_billion_l2112_211269


namespace increasing_f_implies_a_nonnegative_l2112_211299

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x

theorem increasing_f_implies_a_nonnegative (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x < f a y) → a ≥ 0 := by sorry

end increasing_f_implies_a_nonnegative_l2112_211299


namespace cos_75_deg_l2112_211211

/-- Prove that cos 75° = (√6 - √2) / 4 using the angle sum identity for cosine with angles 60° and 15° -/
theorem cos_75_deg : 
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_deg_l2112_211211


namespace bicycle_race_fraction_l2112_211227

theorem bicycle_race_fraction (total_racers : ℕ) (total_wheels : ℕ) 
  (bicycle_wheels : ℕ) (tricycle_wheels : ℕ) :
  total_racers = 40 →
  total_wheels = 96 →
  bicycle_wheels = 2 →
  tricycle_wheels = 3 →
  ∃ (bicycles : ℕ) (tricycles : ℕ),
    bicycles + tricycles = total_racers ∧
    bicycles * bicycle_wheels + tricycles * tricycle_wheels = total_wheels ∧
    (bicycles : ℚ) / total_racers = 3 / 5 :=
by sorry

end bicycle_race_fraction_l2112_211227


namespace greatest_integer_less_than_negative_seventeen_thirds_l2112_211255

theorem greatest_integer_less_than_negative_seventeen_thirds :
  ⌊-17/3⌋ = -6 :=
sorry

end greatest_integer_less_than_negative_seventeen_thirds_l2112_211255


namespace mn_length_in_isosceles_triangle_l2112_211225

-- Define the triangle XYZ
structure Triangle :=
  (area : ℝ)
  (altitude : ℝ)
  (isIsosceles : Bool)

-- Define the line MN
structure ParallelLine :=
  (length : ℝ)

-- Define the trapezoid formed by MN
structure Trapezoid :=
  (area : ℝ)

-- Main theorem
theorem mn_length_in_isosceles_triangle 
  (XYZ : Triangle) 
  (MN : ParallelLine) 
  (trap : Trapezoid) : 
  XYZ.area = 144 ∧ 
  XYZ.altitude = 24 ∧ 
  XYZ.isIsosceles = true ∧
  trap.area = 108 →
  MN.length = 6 :=
sorry

end mn_length_in_isosceles_triangle_l2112_211225


namespace inequality_proof_l2112_211281

theorem inequality_proof (x : ℝ) :
  (x > 0 → x + 1/x ≥ 2) ∧
  (x > 0 → (x + 1/x = 2 ↔ x = 1)) ∧
  (x < 0 → x + 1/x ≤ -2) := by
  sorry

end inequality_proof_l2112_211281


namespace park_area_l2112_211242

/-- Given a rectangular park with length l and width w, where:
    1) l = 3w + 20
    2) The perimeter is 800 feet
    Prove that the area of the park is 28,975 square feet -/
theorem park_area (w l : ℝ) (h1 : l = 3 * w + 20) (h2 : 2 * l + 2 * w = 800) :
  w * l = 28975 := by
  sorry

end park_area_l2112_211242


namespace rhombus_area_example_l2112_211222

/-- Given a rhombus with height h and diagonal d, calculates its area -/
def rhombusArea (h d : ℝ) : ℝ := sorry

/-- Theorem: A rhombus with height 12 cm and diagonal 15 cm has an area of 150 cm² -/
theorem rhombus_area_example : rhombusArea 12 15 = 150 := by sorry

end rhombus_area_example_l2112_211222


namespace jordan_fourth_period_shots_l2112_211289

/-- The number of shots blocked by Jordan in each period of a hockey game --/
structure ShotsBlocked where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Conditions for Jordan's shot-blocking performance --/
def jordan_performance (shots : ShotsBlocked) : Prop :=
  shots.first = 4 ∧
  shots.second = 2 * shots.first ∧
  shots.third = shots.second - 3 ∧
  shots.first + shots.second + shots.third + shots.fourth = 21

/-- Theorem stating that Jordan blocked 4 shots in the fourth period --/
theorem jordan_fourth_period_shots (shots : ShotsBlocked) 
  (h : jordan_performance shots) : shots.fourth = 4 := by
  sorry

#check jordan_fourth_period_shots

end jordan_fourth_period_shots_l2112_211289


namespace additional_track_length_l2112_211240

/-- Calculates the additional track length required to reduce the grade of a railroad line --/
theorem additional_track_length (rise : ℝ) (initial_grade : ℝ) (final_grade : ℝ) :
  rise = 800 →
  initial_grade = 0.04 →
  final_grade = 0.015 →
  ∃ (additional_length : ℝ), 
    33333 ≤ additional_length ∧ 
    additional_length < 33334 ∧
    additional_length = (rise / final_grade) - (rise / initial_grade) :=
by sorry

end additional_track_length_l2112_211240


namespace cubes_not_touching_foil_l2112_211221

/-- Represents a rectangular prism made of 1-inch cubes -/
structure CubePrism where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the volume of a CubePrism -/
def volume (p : CubePrism) : ℕ := p.width * p.length * p.height

/-- Represents the prism of cubes not touching any tin foil -/
def innerPrism (outer : CubePrism) : CubePrism where
  width := outer.width - 2
  length := (outer.width - 2) / 2
  height := (outer.width - 2) / 2

theorem cubes_not_touching_foil (outer : CubePrism) 
  (h1 : outer.width = 10) 
  (h2 : innerPrism outer = { width := 8, length := 4, height := 4 }) : 
  volume (innerPrism outer) = 128 := by
  sorry

end cubes_not_touching_foil_l2112_211221


namespace inequality_proof_l2112_211243

theorem inequality_proof (a b c d : ℝ) : 
  (a + b + c + d)^2 ≤ 3 * (a^2 + b^2 + c^2 + d^2) + 6 * a * b := by
  sorry

end inequality_proof_l2112_211243


namespace arithmetic_geometric_sequence_common_difference_l2112_211237

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (start finish : ℕ) : Prop :=
  ∃ r : ℝ, ∀ n ∈ Finset.range (finish - start), a (start + n + 1) = r * a (start + n)

theorem arithmetic_geometric_sequence_common_difference :
  ∀ a : ℕ → ℝ,
  is_arithmetic_sequence a →
  is_geometric_sequence a 1 3 →
  a 1 = 1 →
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 0 :=
sorry

end arithmetic_geometric_sequence_common_difference_l2112_211237


namespace token_count_after_removal_l2112_211206

/-- Represents a token on the board -/
inductive Token
| White
| Black
| Empty

/-- Represents the board state -/
def Board (n : ℕ) := Fin (2*n) → Fin (2*n) → Token

/-- Counts the number of tokens of a specific type on the board -/
def countTokens (b : Board n) (t : Token) : ℕ := sorry

/-- Performs the token removal process -/
def removeTokens (b : Board n) : Board n := sorry

theorem token_count_after_removal (n : ℕ) (initial_board : Board n) :
  let final_board := removeTokens initial_board
  (countTokens final_board Token.Black ≤ n^2) ∧ 
  (countTokens final_board Token.White ≤ n^2) := by
  sorry

end token_count_after_removal_l2112_211206


namespace cos_pi_fourth_plus_alpha_l2112_211241

theorem cos_pi_fourth_plus_alpha (α : Real) 
  (h : Real.sin (π / 4 - α) = 1 / 3) : 
  Real.cos (π / 4 + α) = 1 / 3 := by
  sorry

end cos_pi_fourth_plus_alpha_l2112_211241


namespace professor_seating_theorem_l2112_211226

/-- The number of chairs in a row -/
def num_chairs : ℕ := 10

/-- The number of professors -/
def num_professors : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 7

/-- The minimum number of students required between each professor -/
def min_students_between : ℕ := 2

/-- A function that calculates the number of ways professors can choose their chairs -/
def professor_seating_arrangements (n_chairs : ℕ) (n_profs : ℕ) (n_students : ℕ) (min_between : ℕ) : ℕ :=
  sorry -- The actual implementation is not provided here

/-- Theorem stating that the number of seating arrangements for professors is 6 -/
theorem professor_seating_theorem :
  professor_seating_arrangements num_chairs num_professors num_students min_students_between = 6 :=
by sorry

end professor_seating_theorem_l2112_211226


namespace no_equation_fits_l2112_211297

def points : List (ℝ × ℝ) := [(0, 200), (1, 140), (2, 80), (3, 20), (4, 0)]

def equation1 (x : ℝ) : ℝ := 200 - 15 * x
def equation2 (x : ℝ) : ℝ := 200 - 20 * x + 5 * x^2
def equation3 (x : ℝ) : ℝ := 200 - 30 * x + 10 * x^2
def equation4 (x : ℝ) : ℝ := 150 - 50 * x

theorem no_equation_fits : 
  ∀ (x y : ℝ), (x, y) ∈ points → 
    (y ≠ equation1 x) ∨ 
    (y ≠ equation2 x) ∨ 
    (y ≠ equation3 x) ∨ 
    (y ≠ equation4 x) := by
  sorry

end no_equation_fits_l2112_211297


namespace min_value_function_l2112_211265

theorem min_value_function (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 + y) / (y^2 - 1) + (y^2 + x) / (x^2 - 1) ≥ 8/3 := by
  sorry

end min_value_function_l2112_211265


namespace fraction_transformation_l2112_211228

theorem fraction_transformation (a b : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (a + 2 : ℚ) / (b^3 : ℚ) = a / (3 * b : ℚ) → a = 1 ∧ b = 3 := by
sorry

end fraction_transformation_l2112_211228


namespace retail_price_calculation_l2112_211204

/-- The retail price of a machine given wholesale price, discount, and profit percentage -/
theorem retail_price_calculation (wholesale_price discount_percent profit_percent : ℝ) 
  (h_wholesale : wholesale_price = 90)
  (h_discount : discount_percent = 10)
  (h_profit : profit_percent = 20) :
  ∃ (retail_price : ℝ), 
    retail_price = 120 ∧ 
    (1 - discount_percent / 100) * retail_price = wholesale_price + (profit_percent / 100 * wholesale_price) := by
  sorry


end retail_price_calculation_l2112_211204


namespace farm_food_calculation_l2112_211231

/-- Given a farm with sheep and horses, calculate the daily food requirement per horse -/
theorem farm_food_calculation (sheep_count horse_count total_food : ℕ) 
  (h1 : sheep_count = 56)
  (h2 : sheep_count = horse_count)
  (h3 : total_food = 12880) :
  total_food / horse_count = 230 := by
sorry

end farm_food_calculation_l2112_211231


namespace geometric_sequence_sum_l2112_211256

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 := by
  sorry

end geometric_sequence_sum_l2112_211256


namespace smallest_solution_of_equation_l2112_211244

theorem smallest_solution_of_equation (y : ℝ) :
  (3 * y^2 + 33 * y - 90 = y * (y + 16)) →
  y ≥ -10 :=
by sorry

end smallest_solution_of_equation_l2112_211244


namespace square_sum_from_means_l2112_211251

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 18) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 92) : 
  x^2 + y^2 = 1112 := by sorry

end square_sum_from_means_l2112_211251


namespace complement_intersection_equals_set_l2112_211202

-- Define the universal set U
def U : Set Nat := {1, 3, 5, 6, 8}

-- Define set A
def A : Set Nat := {1, 6}

-- Define set B
def B : Set Nat := {5, 6, 8}

-- Theorem to prove
theorem complement_intersection_equals_set :
  (U \ A) ∩ B = {5, 8} := by sorry

end complement_intersection_equals_set_l2112_211202


namespace wall_bricks_proof_l2112_211277

/-- The time (in hours) it takes Alice to build the wall alone -/
def alice_time : ℝ := 8

/-- The time (in hours) it takes Bob to build the wall alone -/
def bob_time : ℝ := 12

/-- The decrease in productivity (in bricks per hour) when Alice and Bob work together -/
def productivity_decrease : ℝ := 15

/-- The time (in hours) it takes Alice and Bob to build the wall together -/
def combined_time : ℝ := 6

/-- The number of bricks in the wall -/
def wall_bricks : ℝ := 360

theorem wall_bricks_proof :
  let alice_rate := wall_bricks / alice_time
  let bob_rate := wall_bricks / bob_time
  let combined_rate := alice_rate + bob_rate - productivity_decrease
  combined_rate * combined_time = wall_bricks := by
  sorry

#check wall_bricks_proof

end wall_bricks_proof_l2112_211277


namespace three_digit_multiples_of_three_l2112_211278

theorem three_digit_multiples_of_three : 
  (Finset.filter (fun c => (100 + 10 * c + 7) % 3 = 0) (Finset.range 10)).card = 3 := by
  sorry

end three_digit_multiples_of_three_l2112_211278


namespace functional_equation_solution_l2112_211205

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y) = f x * f y - 2 * x * y) →
  (∀ x : ℝ, f x = 2 * x) ∨ (∀ x : ℝ, f x = -x) :=
by sorry

end functional_equation_solution_l2112_211205


namespace not_54_after_60_operations_l2112_211239

def Operation := Nat → Nat

def is_valid_operation (op : Operation) : Prop :=
  ∀ n, (op n = 2 * n) ∨ (op n = n / 2) ∨ (op n = 3 * n) ∨ (op n = n / 3)

def apply_operations (initial : Nat) (ops : List Operation) : Nat :=
  ops.foldl (λ acc op => op acc) initial

theorem not_54_after_60_operations (ops : List Operation) 
  (h_length : ops.length = 60) 
  (h_valid : ∀ op ∈ ops, is_valid_operation op) : 
  apply_operations 12 ops ≠ 54 := by
  sorry

end not_54_after_60_operations_l2112_211239


namespace binomial_coefficient_identity_a_l2112_211229

theorem binomial_coefficient_identity_a (r m k : ℕ) (h1 : k ≤ m) (h2 : m ≤ r) :
  Nat.choose r m * Nat.choose m k = Nat.choose r k * Nat.choose (r - k) (m - k) :=
sorry

end binomial_coefficient_identity_a_l2112_211229


namespace orange_division_l2112_211293

theorem orange_division (oranges : ℕ) (friends : ℕ) (pieces_per_friend : ℕ) 
  (h1 : oranges = 80) 
  (h2 : friends = 200) 
  (h3 : pieces_per_friend = 4) : 
  (friends * pieces_per_friend) / oranges = 10 := by
  sorry

end orange_division_l2112_211293


namespace sqrt_equation_solution_l2112_211279

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 2 * x) = 5 → x = 21 / 2 := by
  sorry

end sqrt_equation_solution_l2112_211279


namespace tropical_storm_sally_rainfall_l2112_211266

theorem tropical_storm_sally_rainfall (day1 day2 day3 : ℝ) : 
  day2 = 5 * day1 →
  day3 = day1 + day2 - 6 →
  day3 = 18 →
  day1 = 4 := by
sorry

end tropical_storm_sally_rainfall_l2112_211266
