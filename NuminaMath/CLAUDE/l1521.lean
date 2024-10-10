import Mathlib

namespace smallest_sum_of_squares_l1521_152168

theorem smallest_sum_of_squares (x y z : ℝ) : 
  (x + 4) * (y - 4) = 0 → 
  3 * z - 2 * y = 5 → 
  x^2 + y^2 + z^2 ≥ 457/9 :=
by sorry

end smallest_sum_of_squares_l1521_152168


namespace second_year_interest_l1521_152193

/-- Compound interest calculation -/
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n - P

/-- Theorem: Given compound interest for third year and interest rate, calculate second year interest -/
theorem second_year_interest (P : ℝ) (r : ℝ) (CI_3 : ℝ) :
  r = 0.06 → CI_3 = 1272 → compound_interest P r 2 = 1200 := by
  sorry

end second_year_interest_l1521_152193


namespace light_ray_reflection_and_tangent_l1521_152146

/-- A light ray problem with reflection and tangent to a circle -/
theorem light_ray_reflection_and_tangent 
  (A : ℝ × ℝ) 
  (h_A : A = (-3, 3))
  (C : Set (ℝ × ℝ))
  (h_C : C = {(x, y) | x^2 + y^2 - 4*x - 4*y + 7 = 0}) :
  ∃ (incident_ray reflected_ray : Set (ℝ × ℝ)) (distance : ℝ),
    -- Incident ray equation
    incident_ray = {(x, y) | 4*x + 3*y + 3 = 0} ∧
    -- Reflected ray equation
    reflected_ray = {(x, y) | 3*x + 4*y - 3 = 0} ∧
    -- Reflected ray is tangent to circle C
    ∃ (p : ℝ × ℝ), p ∈ C ∧ p ∈ reflected_ray ∧
      ∀ (q : ℝ × ℝ), q ∈ C ∩ reflected_ray → q = p ∧
    -- Distance traveled
    distance = 7 ∧
    -- Distance is from A to tangent point
    ∃ (tangent_point : ℝ × ℝ), 
      tangent_point ∈ C ∧ 
      tangent_point ∈ reflected_ray ∧
      Real.sqrt ((A.1 - tangent_point.1)^2 + (A.2 - tangent_point.2)^2) +
      Real.sqrt ((0 - tangent_point.1)^2 + (0 - tangent_point.2)^2) = distance :=
by sorry

end light_ray_reflection_and_tangent_l1521_152146


namespace jeds_stamp_cards_l1521_152122

/-- Jed's stamp card collection problem -/
theorem jeds_stamp_cards (X : ℕ) : 
  (X + 6 * 4 - 2 * 2 = 40) → X = 20 := by
  sorry

end jeds_stamp_cards_l1521_152122


namespace apartment_cost_difference_l1521_152199

-- Define the parameters for each apartment
def rent1 : ℕ := 800
def utilities1 : ℕ := 260
def miles1 : ℕ := 31

def rent2 : ℕ := 900
def utilities2 : ℕ := 200
def miles2 : ℕ := 21

-- Define common parameters
def workdays : ℕ := 20
def cost_per_mile : ℚ := 58 / 100

-- Function to calculate total monthly cost
def total_cost (rent : ℕ) (utilities : ℕ) (miles : ℕ) : ℚ :=
  rent + utilities + (miles * workdays * cost_per_mile)

-- Theorem statement
theorem apartment_cost_difference :
  ⌊total_cost rent1 utilities1 miles1 - total_cost rent2 utilities2 miles2⌋ = 76 := by
  sorry


end apartment_cost_difference_l1521_152199


namespace exactly_one_hit_probability_l1521_152188

def hit_probability : ℝ := 0.5

theorem exactly_one_hit_probability :
  let p := hit_probability
  let q := 1 - p
  p * q + q * p = 0.5 := by sorry

end exactly_one_hit_probability_l1521_152188


namespace evaluate_expression_l1521_152167

theorem evaluate_expression : 5^3 * 5^4 * 2 = 156250 := by
  sorry

end evaluate_expression_l1521_152167


namespace polygon_perimeter_equals_rectangle_perimeter_l1521_152121

/-- A polygon that forms part of a rectangle -/
structure PartialRectanglePolygon where
  -- The length of the rectangle
  length : ℝ
  -- The width of the rectangle
  width : ℝ

/-- The perimeter of a rectangle -/
def rectanglePerimeter (rect : PartialRectanglePolygon) : ℝ :=
  2 * (rect.length + rect.width)

/-- The perimeter of the polygon that forms part of the rectangle -/
def polygonPerimeter (poly : PartialRectanglePolygon) : ℝ :=
  rectanglePerimeter poly

theorem polygon_perimeter_equals_rectangle_perimeter (poly : PartialRectanglePolygon) :
  polygonPerimeter poly = rectanglePerimeter poly := by
  sorry

#check polygon_perimeter_equals_rectangle_perimeter

end polygon_perimeter_equals_rectangle_perimeter_l1521_152121


namespace solution_set_min_value_l1521_152151

-- Part I
def f (x : ℝ) : ℝ := |3 * x - 1| + |x + 3|

theorem solution_set (x : ℝ) : f x ≥ 4 ↔ x ≤ 0 ∨ x ≥ 1/2 := by sorry

-- Part II
def g (b c x : ℝ) : ℝ := |x - b| + |x + c|

theorem min_value (b c : ℝ) (hb : b > 0) (hc : c > 0) 
  (h_min : ∃ (x : ℝ), ∀ (y : ℝ), g b c x ≤ g b c y) 
  (h_eq : ∃ (x : ℝ), g b c x = 1) :
  (1 / b + 1 / c) ≥ 4 ∧ ∃ (b₀ c₀ : ℝ), 1 / b₀ + 1 / c₀ = 4 := by sorry

end solution_set_min_value_l1521_152151


namespace inverse_proportion_relationship_l1521_152113

theorem inverse_proportion_relationship (k x₁ x₂ y₁ y₂ : ℝ) :
  k ≠ 0 →
  x₁ < 0 →
  0 < x₂ →
  y₁ = k / x₁ →
  y₂ = k / x₂ →
  k < 0 →
  y₂ < 0 ∧ 0 < y₁ :=
by sorry

end inverse_proportion_relationship_l1521_152113


namespace distance_calculation_l1521_152174

theorem distance_calculation (D : ℝ) : 
  (1/4 : ℝ) * D + (1/2 : ℝ) * D + 10 = D → D = 40 := by
sorry

end distance_calculation_l1521_152174


namespace ellipse_eccentricity_l1521_152192

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0) : 
  let e := Real.sqrt (1 - b^2 / a^2)
  let c := e * a
  (∃ (P : ℝ × ℝ), 
    P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧ 
    P.2 = 0 ∧ 
    Real.sqrt ((P.1 + c)^2 + P.2^2) = 3/4 * (a + c)) →
  e = 1/4 := by
sorry

end ellipse_eccentricity_l1521_152192


namespace clark_number_is_23_l1521_152163

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digits_form_unique_prime (n : ℕ) : Prop :=
  is_prime n ∧
  n < 100 ∧
  ∀ m : ℕ, m ≠ n → (m = n % 10 * 10 + n / 10 ∨ m = n) → ¬(is_prime m)

def digits_are_ambiguous (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≠ n ∧ is_prime m ∧ 
    ((m % 10 = n % 10 ∧ m / 10 = n / 10) ∨ 
     (m % 10 = n / 10 ∧ m / 10 = n % 10))

theorem clark_number_is_23 :
  ∃! n : ℕ, digits_form_unique_prime n ∧ digits_are_ambiguous n ∧ n = 23 :=
sorry

end clark_number_is_23_l1521_152163


namespace dividing_chord_length_l1521_152158

/-- A hexagon inscribed in a circle with alternating side lengths -/
structure InscribedHexagon where
  side1 : ℝ
  side2 : ℝ

/-- A chord dividing the hexagon into two trapezoids -/
def dividingChord (h : InscribedHexagon) : ℝ := sorry

/-- Theorem stating the length of the dividing chord -/
theorem dividing_chord_length (h : InscribedHexagon) 
  (h_sides : h.side1 = 4 ∧ h.side2 = 7) : 
  dividingChord h = 560 / 81 := by sorry

end dividing_chord_length_l1521_152158


namespace union_equals_N_l1521_152130

def M : Set ℝ := {x | x - x < 0}
def N : Set ℝ := {x | -3 < x ∧ x < 3}

theorem union_equals_N : M ∪ N = N := by sorry

end union_equals_N_l1521_152130


namespace owl_wings_area_l1521_152107

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  bottomLeft : Point
  topRight : Point

/-- Calculates the area of a triangle given three points using the shoelace formula -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs ((p1.x * p2.y + p2.x * p3.y + p3.x * p1.y) - (p1.y * p2.x + p2.y * p3.x + p3.y * p1.x))

/-- Theorem: The area of the shaded region in the specified rectangle is 4 -/
theorem owl_wings_area (rect : Rectangle) 
    (h1 : rect.topRight.x - rect.bottomLeft.x = 4) 
    (h2 : rect.topRight.y - rect.bottomLeft.y = 5) 
    (h3 : rect.topRight.x - rect.bottomLeft.x = rect.topRight.y - rect.bottomLeft.y - 1) :
    ∃ (p1 p2 p3 : Point), triangleArea p1 p2 p3 = 4 := by
  sorry

end owl_wings_area_l1521_152107


namespace f_always_positive_iff_l1521_152150

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

-- State the theorem
theorem f_always_positive_iff (x : ℝ) :
  (∀ a ∈ Set.Icc (-1 : ℝ) 1, f x a > 0) ↔ (x < 1 ∨ x > 3) := by
  sorry

end f_always_positive_iff_l1521_152150


namespace chord_length_is_four_l1521_152160

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a line in polar form ρ(sin θ - cos θ) = k -/
structure PolarLine where
  k : ℝ

/-- Represents a circle in polar form ρ = a sin θ -/
structure PolarCircle where
  a : ℝ

/-- The length of the chord cut by a polar line from a polar circle -/
noncomputable def chordLength (l : PolarLine) (c : PolarCircle) : ℝ := sorry

/-- Theorem: The chord length is 4 for the given line and circle -/
theorem chord_length_is_four :
  let l : PolarLine := { k := 2 }
  let c : PolarCircle := { a := 4 }
  chordLength l c = 4 := by sorry

end chord_length_is_four_l1521_152160


namespace equation_solution_l1521_152108

theorem equation_solution : 
  ∃ x : ℝ, (Real.sqrt (x^2 + 6*x + 10) + Real.sqrt (x^2 - 6*x + 10) = 8) ↔ 
  (x = (4 * Real.sqrt 42) / 7 ∨ x = -(4 * Real.sqrt 42) / 7) :=
sorry

end equation_solution_l1521_152108


namespace exists_multiple_of_ones_l1521_152125

theorem exists_multiple_of_ones (n : ℕ) (h_pos : 0 < n) (h_coprime : Nat.Coprime n 10) :
  ∃ k : ℕ, (10^k - 1) % (9 * n) = 0 := by
sorry

end exists_multiple_of_ones_l1521_152125


namespace jason_total_money_l1521_152197

/-- Represents the value of different coin types in dollars -/
def coin_value : Fin 3 → ℚ
  | 0 => 0.25  -- Quarter
  | 1 => 0.10  -- Dime
  | 2 => 0.05  -- Nickel
  | _ => 0     -- Unreachable case

/-- Calculates the total value of coins given their quantities -/
def total_value (quarters dimes nickels : ℕ) : ℚ :=
  quarters * coin_value 0 + dimes * coin_value 1 + nickels * coin_value 2

/-- Jason's initial coin quantities -/
def initial_coins : Fin 3 → ℕ
  | 0 => 49  -- Quarters
  | 1 => 32  -- Dimes
  | 2 => 18  -- Nickels
  | _ => 0   -- Unreachable case

/-- Additional coins given by Jason's dad -/
def additional_coins : Fin 3 → ℕ
  | 0 => 25  -- Quarters
  | 1 => 15  -- Dimes
  | 2 => 10  -- Nickels
  | _ => 0   -- Unreachable case

/-- Theorem stating that Jason's total money is $24.60 -/
theorem jason_total_money :
  total_value (initial_coins 0 + additional_coins 0)
              (initial_coins 1 + additional_coins 1)
              (initial_coins 2 + additional_coins 2) = 24.60 := by
  sorry

end jason_total_money_l1521_152197


namespace dollar_cube_difference_l1521_152182

/-- The dollar operation: a $ b = (a + b)² + ab -/
def dollar (a b : ℝ) : ℝ := (a + b)^2 + a * b

/-- Theorem: For any real numbers x and y, (x - y)³ $ (y - x)³ = -(x - y)⁶ -/
theorem dollar_cube_difference (x y : ℝ) : 
  dollar ((x - y)^3) ((y - x)^3) = -((x - y)^6) := by
  sorry

end dollar_cube_difference_l1521_152182


namespace B_pow_15_l1521_152157

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, 1]]

theorem B_pow_15 : B ^ 15 = ![![0,  1, 0],
                              ![-1, 0, 0],
                              ![0,  0, 1]] := by
  sorry

end B_pow_15_l1521_152157


namespace ellipse_equation_l1521_152173

/-- Given an ellipse with equation (x²/a²) + (y²/b²) = 1, where a > b > 0,
    if the minimum value of |k₁| + |k₂| is 1 (where k₁ and k₂ are slopes of lines 
    from any point P on the ellipse to the left and right vertices respectively)
    and the ellipse passes through the point (√3, 1/2), 
    then the equation of the ellipse is x²/4 + y² = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → 
    ∃ k₁ k₂ : ℝ, k₁ * k₂ ≠ 0 ∧ 
    (∀ k₁' k₂' : ℝ, |k₁'| + |k₂'| ≥ |k₁| + |k₂|) ∧
    |k₁| + |k₂| = 1) →
  3 / a^2 + (1/4) / b^2 = 1 →
  ∀ x y : ℝ, x^2 / 4 + y^2 = 1 := by
sorry

end ellipse_equation_l1521_152173


namespace area_triangle_DBC_l1521_152138

/-- Given a triangle ABC with vertices A(0,10), B(0,0), and C(12,0),
    and midpoints D of AB, E of BC, and F of AC,
    prove that the area of triangle DBC is 30. -/
theorem area_triangle_DBC (A B C D E F : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 0) →
  C = (12, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  (1/2 : ℝ) * (C.1 - B.1) * D.2 = 30 := by
  sorry


end area_triangle_DBC_l1521_152138


namespace benjamin_has_45_presents_l1521_152148

/-- The number of presents Benjamin has -/
def benjamins_presents (ethans_presents : ℝ) : ℝ :=
  ethans_presents + 22 - 8.5

/-- Theorem stating that Benjamin has 45 presents given the conditions -/
theorem benjamin_has_45_presents :
  benjamins_presents 31.5 = 45 := by
  sorry

end benjamin_has_45_presents_l1521_152148


namespace factor_expression_l1521_152179

theorem factor_expression (x : ℝ) : x^2*(x+3) + 2*x*(x+3) + (x+3) = (x+1)^2*(x+3) := by
  sorry

end factor_expression_l1521_152179


namespace power_inequalities_l1521_152117

theorem power_inequalities :
  (∀ (x : ℝ), x > 1 → ∀ (a b : ℝ), 0 < a → a < b → x^a < x^b) ∧
  (∀ (x y z : ℝ), 1 < x → x < y → 0 < z → z < 1 → x^z > y^z) :=
sorry

end power_inequalities_l1521_152117


namespace intersection_M_N_l1521_152185

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l1521_152185


namespace conditional_probability_same_color_given_first_red_l1521_152140

def total_balls : ℕ := 5
def red_balls : ℕ := 2
def black_balls : ℕ := 3

def P_A : ℚ := red_balls / total_balls
def P_AB : ℚ := (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1))

theorem conditional_probability_same_color_given_first_red :
  P_AB / P_A = 1 / 4 := by sorry

end conditional_probability_same_color_given_first_red_l1521_152140


namespace expression_evaluation_l1521_152162

theorem expression_evaluation : 
  ((-1) ^ 2022) + |1 - Real.sqrt 2| + ((-27) ^ (1/3 : ℝ)) - Real.sqrt (((-2) ^ 2)) = Real.sqrt 2 - 5 := by
  sorry

end expression_evaluation_l1521_152162


namespace smallest_visible_sum_l1521_152142

/-- Represents a die with 6 faces -/
structure Die :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7)

/-- Represents a 4x4x4 cube made of dice -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → Die

/-- The sum of visible face values on the large cube -/
def visible_sum (cube : LargeCube) : ℕ :=
  sorry

theorem smallest_visible_sum (cube : LargeCube) :
  visible_sum cube ≥ 144 :=
sorry

end smallest_visible_sum_l1521_152142


namespace book_count_proof_l1521_152134

/-- Given the number of books each person has, calculate the total number of books. -/
def total_books (darryl lamont loris : ℕ) : ℕ :=
  darryl + lamont + loris

theorem book_count_proof (darryl lamont loris : ℕ) 
  (h1 : darryl = 20)
  (h2 : lamont = 2 * darryl)
  (h3 : loris + 3 = lamont) :
  total_books darryl lamont loris = 97 := by
  sorry

#check book_count_proof

end book_count_proof_l1521_152134


namespace unique_prime_pair_divisibility_l1521_152177

theorem unique_prime_pair_divisibility : 
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (p^p + q^q + 1) % (p * q) = 0 → 
    (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2) := by
  sorry

end unique_prime_pair_divisibility_l1521_152177


namespace goods_train_speed_l1521_152147

theorem goods_train_speed
  (express_speed : ℝ)
  (head_start : ℝ)
  (catch_up_time : ℝ)
  (h1 : express_speed = 90)
  (h2 : head_start = 6)
  (h3 : catch_up_time = 4) :
  ∃ (goods_speed : ℝ),
    goods_speed * (head_start + catch_up_time) = express_speed * catch_up_time ∧
    goods_speed = 36 := by
  sorry

end goods_train_speed_l1521_152147


namespace sum_of_cubes_divisible_l1521_152196

theorem sum_of_cubes_divisible (a : ℤ) : 
  ∃ k : ℤ, (a - 1)^3 + a^3 + (a + 1)^3 = 3 * a * k := by
sorry

end sum_of_cubes_divisible_l1521_152196


namespace min_volume_ratio_l1521_152119

theorem min_volume_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  8 * (x + y) * (y + z) * (z + x) / (x * y * z) ≥ 64 := by
  sorry

end min_volume_ratio_l1521_152119


namespace sophomore_sample_size_l1521_152118

/-- Represents the number of students to be selected from a stratum in stratified sampling. -/
def stratified_sample (total_population : ℕ) (stratum_size : ℕ) (sample_size : ℕ) : ℕ :=
  (stratum_size * sample_size) / total_population

/-- Theorem stating that in the given stratified sampling scenario, 
    32 sophomores should be selected. -/
theorem sophomore_sample_size : 
  stratified_sample 2000 640 100 = 32 := by
  sorry

end sophomore_sample_size_l1521_152118


namespace parallelogram_product_l1521_152111

-- Define the parallelogram EFGH
def EFGH (EF FG GH HE : ℝ) : Prop :=
  EF = GH ∧ FG = HE

-- Theorem statement
theorem parallelogram_product (x y : ℝ) :
  EFGH 47 (6 * y^2) (3 * x + 7) 27 →
  x * y = 20 * Real.sqrt 2 := by
  sorry

end parallelogram_product_l1521_152111


namespace alley_width_l1521_152165

theorem alley_width (l : ℝ) (h₁ h₂ : ℝ) (θ₁ θ₂ : ℝ) (w : ℝ) 
  (hl : l = 10)
  (hh₁ : h₁ = 4)
  (hh₂ : h₂ = 3)
  (hθ₁ : θ₁ = 30 * π / 180)
  (hθ₂ : θ₂ = 120 * π / 180) :
  w = 5 * (Real.sqrt 3 + 1) :=
sorry

end alley_width_l1521_152165


namespace no_prime_roots_for_quadratic_l1521_152191

theorem no_prime_roots_for_quadratic : ¬∃ k : ℤ, ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (p : ℤ) + q = 90 ∧ 
  (p : ℤ) * q = k ∧
  ∀ x : ℤ, x^2 - 90*x + k = 0 ↔ x = p ∨ x = q :=
by sorry

end no_prime_roots_for_quadratic_l1521_152191


namespace prime_square_plus_twelve_mod_twelve_l1521_152161

theorem prime_square_plus_twelve_mod_twelve (p : ℕ) (h_prime : Nat.Prime p) (h_gt_three : p > 3) :
  (p^2 + 12) % 12 = 1 := by
  sorry

end prime_square_plus_twelve_mod_twelve_l1521_152161


namespace lcm_54_198_l1521_152101

theorem lcm_54_198 : Nat.lcm 54 198 = 594 := by
  sorry

end lcm_54_198_l1521_152101


namespace not_necessarily_p_or_q_l1521_152155

theorem not_necessarily_p_or_q (P Q : Prop) 
  (h1 : ¬P) 
  (h2 : ¬(P ∧ Q)) : 
  ¬∀ (P Q : Prop), (¬P ∧ ¬(P ∧ Q)) → (P ∨ Q) :=
by sorry

end not_necessarily_p_or_q_l1521_152155


namespace isabels_candy_l1521_152183

/-- Given that Isabel initially had 68 pieces of candy and ended up with 93 pieces,
    prove that her friend gave her 25 pieces. -/
theorem isabels_candy (initial : ℕ) (final : ℕ) (h1 : initial = 68) (h2 : final = 93) :
  final - initial = 25 := by
  sorry

end isabels_candy_l1521_152183


namespace friends_meet_time_l1521_152100

def carl_lap : ℕ := 5
def jenna_lap : ℕ := 8
def marco_lap : ℕ := 9
def leah_lap : ℕ := 10

def start_time : ℕ := 9 * 60  -- 9:00 AM in minutes since midnight

theorem friends_meet_time :
  let meeting_time := start_time + Nat.lcm carl_lap (Nat.lcm jenna_lap (Nat.lcm marco_lap leah_lap))
  meeting_time = 15 * 60  -- 3:00 PM in minutes since midnight
  := by sorry

end friends_meet_time_l1521_152100


namespace sequence_property_l1521_152156

/-- Two sequences satisfying the given conditions -/
def sequences (a b : ℕ+ → ℚ) : Prop :=
  a 1 = 1/2 ∧
  (∀ n : ℕ+, a n + b n = 1) ∧
  (∀ n : ℕ+, b (n + 1) = b n / (1 - (a n)^2))

/-- The theorem to be proved -/
theorem sequence_property (a b : ℕ+ → ℚ) (h : sequences a b) :
  ∀ n : ℕ+, b n = n / (n + 1) :=
sorry

end sequence_property_l1521_152156


namespace stratified_sampling_l1521_152169

theorem stratified_sampling (total_sample : ℕ) (ratio_first : ℕ) (ratio_second : ℕ) (ratio_third : ℕ) : 
  total_sample = 50 → ratio_first = 3 → ratio_second = 4 → ratio_third = 3 →
  (ratio_second : ℚ) / (ratio_first + ratio_second + ratio_third : ℚ) * total_sample = 20 := by
  sorry

end stratified_sampling_l1521_152169


namespace train_travel_time_l1521_152164

/-- Given a train that travels 360 miles in 3 hours, prove that it takes 2 hours to travel an additional 240 miles at the same rate. -/
theorem train_travel_time (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ) :
  initial_distance = 360 →
  initial_time = 3 →
  additional_distance = 240 →
  (additional_distance / (initial_distance / initial_time)) = 2 := by
  sorry

end train_travel_time_l1521_152164


namespace parabola_equation_l1521_152166

/-- A parabola with the same shape as y = -5x^2 + 2 and vertex at (4, -2) -/
structure Parabola where
  /-- The coefficient of x^2 in the parabola equation -/
  a : ℝ
  /-- The x-coordinate of the vertex -/
  h : ℝ
  /-- The y-coordinate of the vertex -/
  k : ℝ
  /-- The parabola has the same shape as y = -5x^2 + 2 -/
  shape_cond : a = -5
  /-- The vertex is at (4, -2) -/
  vertex_cond : h = 4 ∧ k = -2

/-- The analytical expression of the parabola -/
def parabola_expression (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2 + p.k

theorem parabola_equation (p : Parabola) :
  ∀ x, parabola_expression p x = -5 * (x - 4)^2 - 2 := by
  sorry

end parabola_equation_l1521_152166


namespace unique_m_exists_l1521_152149

/-- A right triangle in the coordinate plane with legs parallel to x and y axes -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The median to the midpoint of the leg parallel to the x-axis -/
def medianX (t : RightTriangle) : ℝ → ℝ := fun x => 3 * x + 1

/-- The median to the midpoint of the leg parallel to the y-axis -/
def medianY (t : RightTriangle) (m : ℝ) : ℝ → ℝ := fun x => (2 * m + 1) * x + 3

/-- The existence and uniqueness of m for a valid right triangle -/
theorem unique_m_exists : ∃! m : ℝ, ∃ t : RightTriangle, 
  (∀ x : ℝ, medianX t x = 3 * x + 1) ∧ 
  (∀ x : ℝ, medianY t m x = (2 * m + 1) * x + 3) :=
sorry

end unique_m_exists_l1521_152149


namespace functional_equation_solution_l1521_152105

/-- A continuous function satisfying the given functional equation -/
structure FunctionalEquation where
  f : ℝ → ℝ
  continuous : Continuous f
  equation : ∀ x y, f (x + y) = f x + f y + f x * f y

/-- The theorem stating the form of the function satisfying the equation -/
theorem functional_equation_solution (fe : FunctionalEquation) :
  ∃ a : ℝ, a ≥ 1 ∧ ∀ x, fe.f x = a^x - 1 := by
  sorry

end functional_equation_solution_l1521_152105


namespace friends_games_count_l1521_152135

/-- The number of games Katie's new friends have -/
def new_friends_games : ℕ := 88

/-- The number of games Katie's old friends have -/
def old_friends_games : ℕ := 53

/-- The total number of games Katie's friends have -/
def total_friends_games : ℕ := new_friends_games + old_friends_games

theorem friends_games_count : total_friends_games = 141 := by
  sorry

end friends_games_count_l1521_152135


namespace network_engineers_from_university_a_l1521_152132

theorem network_engineers_from_university_a 
  (total_original : ℕ) 
  (new_hires : ℕ) 
  (fraction_from_a : ℚ) :
  total_original = 20 →
  new_hires = 8 →
  fraction_from_a = 3/4 →
  (fraction_from_a * (total_original + new_hires : ℚ) - new_hires) / total_original = 13/20 :=
by sorry

end network_engineers_from_university_a_l1521_152132


namespace center_is_seven_l1521_152102

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if two positions in the grid are adjacent or diagonal -/
def adjacent_or_diagonal (i j i' j' : Fin 3) : Prop :=
  (i = i' ∧ |j - j'| = 1) ∨ (j = j' ∧ |i - i'| = 1) ∨ (|i - i'| = 1 ∧ |j - j'| = 1)

/-- The main theorem -/
theorem center_is_seven (g : Grid) : 
  (∀ n : ℕ, n ∈ Finset.range 9 → ∃ i j : Fin 3, g i j = n + 1) →
  (g 0 0 + g 0 2 + g 2 0 + g 2 2 = 20) →
  (∀ n : ℕ, n ∈ Finset.range 8 → 
    ∃ i j i' j' : Fin 3, g i j = n + 1 ∧ g i' j' = n + 2 ∧ adjacent_or_diagonal i j i' j') →
  g 1 1 = 7 := by
  sorry

end center_is_seven_l1521_152102


namespace min_value_of_expression_l1521_152175

theorem min_value_of_expression (x : ℝ) (h : x > 0) : 4 * x + 1 / x ≥ 4 ∧ 
  (4 * x + 1 / x = 4 ↔ x = 1 / 2) := by
  sorry

end min_value_of_expression_l1521_152175


namespace ryan_final_tokens_l1521_152181

def token_calculation (initial_tokens : ℕ) : ℕ :=
  let after_pacman := initial_tokens - (2 * initial_tokens / 3)
  let after_candy_crush := after_pacman - (after_pacman / 2)
  let after_skiball := after_candy_crush - 7
  let after_friend_borrowed := after_skiball - 5
  let after_friend_returned := after_friend_borrowed + 8
  let after_parents_bought := after_friend_returned + (10 * 7)
  after_parents_bought - 3

theorem ryan_final_tokens : 
  token_calculation 36 = 75 := by sorry

end ryan_final_tokens_l1521_152181


namespace simplify_radical_sum_l1521_152184

theorem simplify_radical_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end simplify_radical_sum_l1521_152184


namespace cricket_average_increase_l1521_152194

theorem cricket_average_increase 
  (score_19th_inning : ℕ) 
  (average_after_19 : ℚ) 
  (h1 : score_19th_inning = 97) 
  (h2 : average_after_19 = 25) : 
  average_after_19 - (((19 * average_after_19) - score_19th_inning) / 18) = 4 := by
sorry

end cricket_average_increase_l1521_152194


namespace prove_a_value_l1521_152176

/-- Custom operation @ for positive integers -/
def custom_op (k : ℕ+) (j : ℕ+) : ℕ+ :=
  sorry

/-- Given b and t, prove a = 1060 -/
theorem prove_a_value (b t : ℚ) (h1 : b = 2120) (h2 : t = 1/2) :
  ∃ a : ℚ, t = a / b ∧ a = 1060 := by
  sorry

end prove_a_value_l1521_152176


namespace store_discount_proof_l1521_152126

/-- Calculates the actual discount percentage given the initial discount and VIP discount -/
def actual_discount (initial_discount : ℝ) (vip_discount : ℝ) : ℝ :=
  1 - (1 - initial_discount) * (1 - vip_discount)

/-- Proves that the actual discount is 28% given a 20% initial discount and 10% VIP discount -/
theorem store_discount_proof :
  actual_discount 0.2 0.1 = 0.28 := by
  sorry

#eval actual_discount 0.2 0.1

end store_discount_proof_l1521_152126


namespace union_equals_interval_l1521_152144

def A : Set ℝ := {1, 2, 3, 4}

def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

theorem union_equals_interval (a : ℝ) :
  A ∪ B a = Set.Iic 5 → a = 5 := by sorry

end union_equals_interval_l1521_152144


namespace complement_union_equality_l1521_152115

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3, 4}

-- Define set A
def A : Set ℕ := {0, 1, 2}

-- Define set B
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_union_equality :
  (Set.compl A ∩ U) ∪ B = {2, 3, 4} := by sorry

end complement_union_equality_l1521_152115


namespace min_value_xy_expression_l1521_152172

theorem min_value_xy_expression (x y : ℝ) : (x*y - 2)^2 + (x^2 + y^2) ≥ 4 := by
  sorry

end min_value_xy_expression_l1521_152172


namespace cubic_equation_solution_mean_l1521_152110

theorem cubic_equation_solution_mean :
  let f : ℝ → ℝ := λ x => x^3 + 5*x^2 - 14*x
  let solutions := {x : ℝ | f x = 0}
  ∃ (s : Finset ℝ), s.card = 3 ∧ (∀ x ∈ s, f x = 0) ∧ 
    (s.sum id) / s.card = -5/3 :=
sorry

end cubic_equation_solution_mean_l1521_152110


namespace explorer_findings_l1521_152171

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- The total value of the explorer's findings -/
def totalValue : ℕ :=
  base6ToBase10 1524 + base6ToBase10 305 + base6ToBase10 1432

theorem explorer_findings :
  totalValue = 905 := by sorry

end explorer_findings_l1521_152171


namespace square_area_from_perspective_l1521_152154

-- Define a square
structure Square where
  side : ℝ
  area : ℝ
  area_eq : area = side * side

-- Define a parallelogram
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ

-- Define the perspective drawing relation
def perspective_drawing (s : Square) (p : Parallelogram) : Prop :=
  (p.side1 = s.side ∨ p.side1 = s.side / 2) ∧ 
  (p.side2 = s.side ∨ p.side2 = s.side / 2)

-- Theorem statement
theorem square_area_from_perspective (s : Square) (p : Parallelogram) :
  perspective_drawing s p → (p.side1 = 4 ∨ p.side2 = 4) → (s.area = 16 ∨ s.area = 64) :=
by sorry

end square_area_from_perspective_l1521_152154


namespace mole_cannot_survive_winter_l1521_152189

/-- Represents the amount of grain in bags -/
structure GrainReserves where
  largeBags : ℕ
  smallBags : ℕ

/-- Represents the exchange rate between large and small bags -/
structure ExchangeRate where
  largeBags : ℕ
  smallBags : ℕ

/-- Represents the grain consumption per month -/
structure MonthlyConsumption where
  largeBags : ℕ

def canSurviveWinter (reserves : GrainReserves) (consumption : MonthlyConsumption) 
                     (exchangeRate : ExchangeRate) (months : ℕ) : Prop :=
  reserves.largeBags ≥ consumption.largeBags * months

theorem mole_cannot_survive_winter : 
  let reserves := GrainReserves.mk 20 32
  let consumption := MonthlyConsumption.mk 7
  let exchangeRate := ExchangeRate.mk 2 3
  let winterMonths := 3
  ¬(canSurviveWinter reserves consumption exchangeRate winterMonths) := by
  sorry

#check mole_cannot_survive_winter

end mole_cannot_survive_winter_l1521_152189


namespace misha_initial_dollars_l1521_152170

/-- The amount of dollars Misha needs to earn -/
def dollars_to_earn : ℕ := 13

/-- The total amount of dollars Misha will have after earning -/
def total_dollars : ℕ := 47

/-- Misha's initial amount of dollars -/
def initial_dollars : ℕ := total_dollars - dollars_to_earn

theorem misha_initial_dollars : initial_dollars = 34 := by
  sorry

end misha_initial_dollars_l1521_152170


namespace quadratic_sum_of_p_q_l1521_152129

/-- Given a quadratic equation 9x^2 - 54x + 63 = 0, when transformed
    into the form (x + p)^2 = q, the sum of p and q is equal to -1 -/
theorem quadratic_sum_of_p_q : ∃ (p q : ℝ),
  (∀ x, 9 * x^2 - 54 * x + 63 = 0 ↔ (x + p)^2 = q) ∧
  p + q = -1 := by
  sorry

end quadratic_sum_of_p_q_l1521_152129


namespace inequality_proof_l1521_152141

theorem inequality_proof (x : ℝ) (h1 : x ≥ 5) (h2 : x ≠ 2) :
  (x - 5) / (x^2 + x + 3) ≥ 0 := by
  sorry

end inequality_proof_l1521_152141


namespace coin_arrangements_l1521_152136

/-- Represents the number of gold coins -/
def gold_coins : ℕ := 6

/-- Represents the number of silver coins -/
def silver_coins : ℕ := 4

/-- Represents the total number of coins -/
def total_coins : ℕ := gold_coins + silver_coins

/-- Represents the number of possible color arrangements -/
def color_arrangements : ℕ := Nat.choose total_coins silver_coins

/-- Represents the number of possible face orientations -/
def face_orientations : ℕ := total_coins + 1

/-- The main theorem stating the number of distinguishable arrangements -/
theorem coin_arrangements :
  color_arrangements * face_orientations = 2310 := by sorry

end coin_arrangements_l1521_152136


namespace max_value_of_f_on_interval_l1521_152123

def f (x : ℝ) := -x + 1

theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (1/2 : ℝ) 2 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (1/2 : ℝ) 2 → f x ≤ f c ∧
  f c = 1/2 :=
sorry

end max_value_of_f_on_interval_l1521_152123


namespace arithmetic_sequence_problem_l1521_152139

theorem arithmetic_sequence_problem (a₁ d : ℝ) : 
  let a := fun n => a₁ + (n - 1) * d
  (a 9) / (a 2) = 5 ∧ (a 13) = 2 * (a 6) + 5 → a₁ = 3 ∧ d = 4 := by
  sorry

end arithmetic_sequence_problem_l1521_152139


namespace largest_factorable_n_l1521_152178

/-- The largest value of n for which 5x^2 + nx + 110 can be factored with integer coefficients -/
def largest_n : ℕ := 551

/-- Predicate to check if a polynomial can be factored with integer coefficients -/
def can_be_factored (n : ℤ) : Prop :=
  ∃ (A B : ℤ), 5 * B + A = n ∧ A * B = 110

theorem largest_factorable_n :
  (∀ m : ℕ, m > largest_n → ¬(can_be_factored m)) ∧
  (can_be_factored largest_n) :=
sorry

end largest_factorable_n_l1521_152178


namespace base_conversion_count_l1521_152195

theorem base_conversion_count : 
  ∃! n : ℕ, n = (Finset.filter (fun c : ℕ => c ≥ 2 ∧ c^2 ≤ 256 ∧ 256 < c^3) (Finset.range 257)).card ∧ n = 10 := by
  sorry

end base_conversion_count_l1521_152195


namespace fraction_simplification_l1521_152127

theorem fraction_simplification : (1998 - 998) / 1000 = 1 := by
  sorry

end fraction_simplification_l1521_152127


namespace polynomial_bound_l1521_152187

theorem polynomial_bound (z : ℂ) (h : Complex.abs z = 1) :
  ∃ p : Polynomial ℂ, (∀ i : Fin 1996, p.coeff i = 1 ∨ p.coeff i = -1) ∧
    p.degree = 1995 ∧ Complex.abs (p.eval z) ≤ 4 := by
  sorry

end polynomial_bound_l1521_152187


namespace min_games_to_dominate_leaderboard_l1521_152180

/-- Represents the game with a leaderboard of 30 scores -/
structure Game where
  leaderboard_size : Nat
  leaderboard_size_eq : leaderboard_size = 30

/-- Calculates the number of games needed to achieve all scores -/
def games_needed (game : Game) : Nat :=
  game.leaderboard_size + (game.leaderboard_size * (game.leaderboard_size - 1)) / 2

/-- Theorem stating the minimum number of games required -/
theorem min_games_to_dominate_leaderboard (game : Game) :
  games_needed game = 465 := by
  sorry

#check min_games_to_dominate_leaderboard

end min_games_to_dominate_leaderboard_l1521_152180


namespace horse_food_per_day_l1521_152114

/-- Given the ratio of sheep to horses, the number of sheep, and the total amount of horse food,
    calculate the amount of food per horse. -/
theorem horse_food_per_day (sheep_ratio : ℕ) (horse_ratio : ℕ) (num_sheep : ℕ) (total_food : ℕ) :
  sheep_ratio = 5 →
  horse_ratio = 7 →
  num_sheep = 40 →
  total_food = 12880 →
  (total_food / (horse_ratio * num_sheep / sheep_ratio) : ℚ) = 230 := by
  sorry

end horse_food_per_day_l1521_152114


namespace fabric_delivery_problem_l1521_152106

/-- Represents the fabric delivery problem for Daniel's textile company -/
theorem fabric_delivery_problem (monday_delivery : ℝ) : 
  monday_delivery * 2 * 3.5 = 140 → monday_delivery = 20 := by
  sorry

#check fabric_delivery_problem

end fabric_delivery_problem_l1521_152106


namespace unique_positive_solution_l1521_152153

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (1/2 * (4*x^2 - 1) = (x^2 - 50*x - 20) * (x^2 + 25*x + 10)) ∧ x = 26 + Real.sqrt 677 := by
  sorry

end unique_positive_solution_l1521_152153


namespace evaluate_expression_l1521_152143

theorem evaluate_expression : 
  (125 : ℝ) ^ (1/3) / (64 : ℝ) ^ (1/2) * (81 : ℝ) ^ (1/4) = 15/8 := by
  sorry

end evaluate_expression_l1521_152143


namespace plane_speed_l1521_152104

/-- The speed of a plane in still air, given its performance with and against wind. -/
theorem plane_speed (distance_with_wind : ℝ) (distance_against_wind : ℝ) (wind_speed : ℝ) :
  distance_with_wind = 400 →
  distance_against_wind = 320 →
  wind_speed = 20 →
  ∃ (plane_speed : ℝ),
    distance_with_wind / (plane_speed + wind_speed) = distance_against_wind / (plane_speed - wind_speed) ∧
    plane_speed = 180 :=
by sorry

end plane_speed_l1521_152104


namespace marias_score_l1521_152152

/-- Given that Maria's score is 50 points more than Tom's and their average score is 105,
    prove that Maria's score is 130. -/
theorem marias_score (tom_score : ℕ) : 
  let maria_score := tom_score + 50
  let average := (maria_score + tom_score) / 2
  average = 105 → maria_score = 130 := by
sorry

end marias_score_l1521_152152


namespace problem_solution_l1521_152133

theorem problem_solution :
  ∃ (x y : ℝ),
    (0.3 * x = 0.4 * 150 + 90) ∧
    (0.2 * x = 0.5 * 180 - 60) ∧
    (y = 0.75 * x) ∧
    (y^2 = x + 100) ∧
    (x = 150) ∧
    (y = 112.5) :=
by sorry

end problem_solution_l1521_152133


namespace units_digit_sum_factorials_2023_l1521_152198

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_2023 :
  (sum_factorials 2023) % 10 = 3 := by sorry

end units_digit_sum_factorials_2023_l1521_152198


namespace gcd_factorial_eight_ten_l1521_152190

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem gcd_factorial_eight_ten : 
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end gcd_factorial_eight_ten_l1521_152190


namespace no_valid_sequences_for_420_l1521_152112

/-- Represents a sequence of consecutive positive integers -/
structure ConsecutiveSequence where
  start : ℕ
  length : ℕ
  h_length : length ≥ 2

/-- The sum of a consecutive sequence -/
def sum_consecutive_sequence (seq : ConsecutiveSequence) : ℕ :=
  seq.length * seq.start + seq.length * (seq.length - 1) / 2

/-- Predicate for a natural number being a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- Theorem stating that there are no valid sequences summing to 420 -/
theorem no_valid_sequences_for_420 :
  ¬∃ (seq : ConsecutiveSequence), 
    sum_consecutive_sequence seq = 420 ∧ 
    is_perfect_square seq.start :=
sorry

end no_valid_sequences_for_420_l1521_152112


namespace class_composition_l1521_152186

theorem class_composition (total : ℕ) (girls : ℕ) (boys : ℕ) 
  (h1 : total = 70) 
  (h2 : 4 * boys = 3 * girls) 
  (h3 : girls + boys = total) : 
  girls = 40 ∧ boys = 30 := by
  sorry

end class_composition_l1521_152186


namespace point_side_line_range_l1521_152109

/-- Given that points (3,1) and (-4,6) are on the same side of the line 3x-2y+a=0,
    the range of values for a is a < -7 or a > 24 -/
theorem point_side_line_range (a : ℝ) : 
  ((3 * 3 - 2 * 1 + a) * (-4 * 3 - 2 * 6 + a) > 0) ↔ (a < -7 ∨ a > 24) := by
  sorry

end point_side_line_range_l1521_152109


namespace flour_needed_for_90_muffins_l1521_152159

-- Define the given ratio of flour to muffins
def flour_per_muffin : ℚ := 1.5 / 15

-- Define the number of muffins Maria wants to bake
def muffins_to_bake : ℕ := 90

-- Theorem to prove
theorem flour_needed_for_90_muffins :
  (flour_per_muffin * muffins_to_bake : ℚ) = 9 := by
  sorry

end flour_needed_for_90_muffins_l1521_152159


namespace family_change_is_74_l1521_152103

/-- Represents the cost of tickets for a family visit to an amusement park --/
def amusement_park_change (regular_price : ℕ) (child_discount : ℕ) (amount_given : ℕ) : ℕ :=
  let adult_cost := regular_price
  let child_cost := regular_price - child_discount
  let total_cost := 2 * adult_cost + 2 * child_cost
  amount_given - total_cost

/-- Theorem stating that the change received by the family is $74 --/
theorem family_change_is_74 :
  amusement_park_change 109 5 500 = 74 := by
  sorry

end family_change_is_74_l1521_152103


namespace matt_flour_bags_matt_flour_bags_correct_l1521_152131

theorem matt_flour_bags (cookies_per_batch : ℕ) (flour_per_batch : ℕ) 
  (flour_per_bag : ℕ) (cookies_eaten : ℕ) (cookies_left : ℕ) : ℕ :=
  let total_cookies := cookies_eaten + cookies_left
  let total_dozens := total_cookies / cookies_per_batch
  let total_flour := total_dozens * flour_per_batch
  total_flour / flour_per_bag

#check matt_flour_bags 12 2 5 15 105 = 4

theorem matt_flour_bags_correct : matt_flour_bags 12 2 5 15 105 = 4 := by
  sorry

end matt_flour_bags_matt_flour_bags_correct_l1521_152131


namespace verify_coin_weights_l1521_152120

/-- Represents a coin with a denomination and weight -/
structure Coin where
  denomination : ℕ
  weight : ℕ

/-- Represents a balance scale measurement -/
def BalanceMeasurement := List Coin → List Coin → Bool

/-- Checks if the total weight of coins on both sides of the scale is equal -/
def isBalanced (coins1 coins2 : List Coin) : Bool :=
  (coins1.map (λ c => c.weight)).sum = (coins2.map (λ c => c.weight)).sum

/-- Represents the available weight for measurements -/
def WeightValue : ℕ := 9

/-- Theorem stating that it's possible to verify the weights of the coins -/
theorem verify_coin_weights (coins : List Coin) 
  (h1 : coins.length = 4)
  (h2 : coins.map (λ c => c.denomination) = [1, 2, 3, 5])
  (h3 : ∀ c ∈ coins, c.weight = c.denomination)
  (balance : BalanceMeasurement) 
  (h4 : ∀ c1 c2, balance c1 c2 = isBalanced c1 c2) :
  ∃ (measurements : List (List Coin × List Coin)),
    measurements.length ≤ 4 ∧ 
    (∀ m ∈ measurements, balance m.1 m.2 = true) ∧
    (∀ c ∈ coins, c.weight = c.denomination) :=
  sorry

end verify_coin_weights_l1521_152120


namespace total_get_well_cards_l1521_152137

/-- Represents the number of cards Mariela received in different categories -/
structure CardCounts where
  handwritten : ℕ
  multilingual : ℕ
  multiplePages : ℕ

/-- Calculates the total number of cards given the counts for each category -/
def totalCards (counts : CardCounts) : ℕ :=
  counts.handwritten + counts.multilingual + counts.multiplePages

/-- Theorem stating the total number of get well cards Mariela received -/
theorem total_get_well_cards 
  (hospital : CardCounts) 
  (home : CardCounts) 
  (h1 : hospital.handwritten = 152)
  (h2 : hospital.multilingual = 98)
  (h3 : hospital.multiplePages = 153)
  (h4 : totalCards hospital = 403)
  (h5 : home.handwritten = 121)
  (h6 : home.multilingual = 66)
  (h7 : home.multiplePages = 100)
  (h8 : totalCards home = 287) :
  totalCards hospital + totalCards home = 690 := by
  sorry

#check total_get_well_cards

end total_get_well_cards_l1521_152137


namespace zoo_animals_l1521_152124

/-- The number of sea horses at the zoo -/
def num_sea_horses : ℕ := 70

/-- The number of penguins at the zoo -/
def num_penguins : ℕ := num_sea_horses + 85

/-- The ratio of sea horses to penguins is 5:11 -/
axiom ratio_constraint : (num_sea_horses : ℚ) / num_penguins = 5 / 11

theorem zoo_animals : num_sea_horses = 70 := by
  sorry

end zoo_animals_l1521_152124


namespace theo_cookie_consumption_l1521_152128

/-- The number of cookies Theo eats at a time -/
def cookies_per_time : ℕ := 35

/-- The number of times Theo eats cookies per day -/
def times_per_day : ℕ := 7

/-- The number of days in a month -/
def days_per_month : ℕ := 30

/-- The number of months we are considering -/
def total_months : ℕ := 12

/-- The total number of cookies Theo can eat in the given period -/
def total_cookies : ℕ := cookies_per_time * times_per_day * days_per_month * total_months

theorem theo_cookie_consumption : total_cookies = 88200 := by
  sorry

end theo_cookie_consumption_l1521_152128


namespace parallel_tangents_f_1_equals_1_l1521_152116

def f (a b x : ℝ) : ℝ := x^3 + a*x + b

theorem parallel_tangents_f_1_equals_1 (a b : ℝ) (h1 : a ≠ b) 
  (h2 : 3*a^2 + a = 3*b^2 + a) : f a b 1 = 1 := by
  sorry

end parallel_tangents_f_1_equals_1_l1521_152116


namespace product_equals_sum_solution_l1521_152145

theorem product_equals_sum_solution (x y : ℝ) (h1 : x * y = x + y) (h2 : y ≠ 1) :
  x = y / (y - 1) := by
  sorry

end product_equals_sum_solution_l1521_152145
