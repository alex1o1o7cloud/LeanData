import Mathlib

namespace tan_pi_over_a_equals_sqrt_three_l1892_189216

theorem tan_pi_over_a_equals_sqrt_three (a : ℝ) (h : a ^ 3 = 27) : 
  Real.tan (π / a) = Real.sqrt 3 := by sorry

end tan_pi_over_a_equals_sqrt_three_l1892_189216


namespace base_b_is_7_l1892_189229

/-- Given a base b, this function represents the number 15 in that base -/
def number_15 (b : ℕ) : ℕ := b + 5

/-- Given a base b, this function represents the number 433 in that base -/
def number_433 (b : ℕ) : ℕ := 4*b^2 + 3*b + 3

/-- The theorem states that if the square of the number represented by 15 in base b
    equals the number represented by 433 in base b, then b must be 7 in base 10 -/
theorem base_b_is_7 : ∃ (b : ℕ), (number_15 b)^2 = number_433 b ∧ b = 7 := by
  sorry

end base_b_is_7_l1892_189229


namespace power_inequality_l1892_189200

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^5 + b^5 + c^5 ≥ a^3*b*c + a*b^3*c + a*b*c^3 := by
  sorry

end power_inequality_l1892_189200


namespace point_on_circle_l1892_189265

theorem point_on_circle (t : ℝ) : 
  let x := (3 - t^3) / (3 + t^3)
  let y := 3*t / (3 + t^3)
  x^2 + y^2 = 1 := by
sorry

end point_on_circle_l1892_189265


namespace work_completion_time_l1892_189225

/-- Given two workers a and b, where a does half as much work as b in 3/4 of the time,
    and b takes 30 days to complete the work alone, prove that they take 18 days
    to complete the work together. -/
theorem work_completion_time (a b : ℝ) : 
  (a * (3/4 * 30) = (1/2) * b * 30) →  -- a does half as much work as b in 3/4 of the time
  (b * 30 = 1) →  -- b completes the work in 30 days
  (a + b) * 18 = 1  -- they complete the work together in 18 days
:= by sorry

end work_completion_time_l1892_189225


namespace expression_simplification_l1892_189282

theorem expression_simplification (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4) / (a^2 + 2*a + 1)) = -a - 1 := by
  sorry

end expression_simplification_l1892_189282


namespace parabola_focus_theorem_l1892_189242

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = 4 * x^2 + 8 * x - 5

/-- The focus of a parabola y = a(x - h)^2 + k is at (h, k + 1/(4a)) -/
def parabola_focus (a h k x y : ℝ) : Prop :=
  x = h ∧ y = k + 1 / (4 * a)

/-- Theorem: The focus of the parabola y = 4x^2 + 8x - 5 is at (-1, -8.9375) -/
theorem parabola_focus_theorem :
  ∃ (x y : ℝ), parabola_equation x y ∧ parabola_focus 4 (-1) (-9) x y ∧ x = -1 ∧ y = -8.9375 := by
  sorry

end parabola_focus_theorem_l1892_189242


namespace problem_solution_l1892_189253

theorem problem_solution (a b : ℕ+) (q r : ℕ) :
  a^2 + b^2 = q * (a + b) + r ∧ q^2 + r = 1977 →
  ((a = 50 ∧ b = 7) ∨ (a = 50 ∧ b = 37) ∨ (a = 7 ∧ b = 50) ∨ (a = 37 ∧ b = 50)) :=
by sorry

end problem_solution_l1892_189253


namespace dice_roll_probability_l1892_189295

theorem dice_roll_probability (m : ℝ) : 
  (∀ x y : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 → (x^2 : ℝ) + y^2 ≤ m) ↔ 
  72 ≤ m :=
by sorry

end dice_roll_probability_l1892_189295


namespace set_intersection_proof_l1892_189208

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem set_intersection_proof : M ∩ N = {0, 1} := by
  sorry

end set_intersection_proof_l1892_189208


namespace shirt_tie_combinations_l1892_189280

/-- The number of shirts -/
def num_shirts : ℕ := 8

/-- The number of ties -/
def num_ties : ℕ := 6

/-- The number of shirts that can be paired with the specific tie -/
def specific_shirts : ℕ := 2

/-- The number of different shirt-and-tie combinations -/
def total_combinations : ℕ := (num_shirts - specific_shirts) * (num_ties - 1) + specific_shirts

theorem shirt_tie_combinations : total_combinations = 32 := by
  sorry

end shirt_tie_combinations_l1892_189280


namespace probability_one_girl_no_growth_pie_l1892_189239

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def shrink_pies : ℕ := 4
def pies_given : ℕ := 3

def probability_no_growth_pie : ℚ :=
  1 - (Nat.choose shrink_pies (pies_given - 1) : ℚ) / (Nat.choose total_pies pies_given)

theorem probability_one_girl_no_growth_pie :
  probability_no_growth_pie = 7/10 :=
sorry

end probability_one_girl_no_growth_pie_l1892_189239


namespace min_team_size_for_handshake_probability_l1892_189220

theorem min_team_size_for_handshake_probability (n : ℕ) : n ≥ 20 ↔ 
  (2 : ℚ) / (n + 1 : ℚ) < (1 : ℚ) / 10 ∧ 
  ∀ m : ℕ, m < n → (2 : ℚ) / (m + 1 : ℚ) ≥ (1 : ℚ) / 10 :=
by sorry

end min_team_size_for_handshake_probability_l1892_189220


namespace max_boxes_A_l1892_189269

def price_A : ℝ := 24
def price_B : ℝ := 16
def total_boxes : ℕ := 200
def max_cost : ℝ := 3920

theorem max_boxes_A : 
  price_A + 2 * price_B = 56 →
  2 * price_A + price_B = 64 →
  (∀ m : ℕ, m ≤ total_boxes → 
    price_A * m + price_B * (total_boxes - m) ≤ max_cost →
    m ≤ 90) ∧
  (∃ m : ℕ, m = 90 ∧ 
    price_A * m + price_B * (total_boxes - m) ≤ max_cost) :=
by sorry

end max_boxes_A_l1892_189269


namespace reciprocal_of_2016_l1892_189223

theorem reciprocal_of_2016 : (2016⁻¹ : ℚ) = 1 / 2016 := by sorry

end reciprocal_of_2016_l1892_189223


namespace centers_connection_line_l1892_189234

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the centers of the circles
def center1 : ℝ × ℝ := (2, -3)
def center2 : ℝ × ℝ := (3, 0)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem centers_connection_line : 
  line_equation (center1.1) (center1.2) ∧ 
  line_equation (center2.1) (center2.2) ∧
  ∀ (x y : ℝ), line_equation x y ↔ ∃ (t : ℝ), 
    x = center1.1 + t * (center2.1 - center1.1) ∧
    y = center1.2 + t * (center2.2 - center1.2) :=
sorry

end centers_connection_line_l1892_189234


namespace container_capacity_is_20_l1892_189273

-- Define the capacity of the container
def container_capacity : ℝ := 20

-- Define the initial fill percentage
def initial_fill_percentage : ℝ := 0.30

-- Define the final fill percentage
def final_fill_percentage : ℝ := 0.75

-- Define the amount of water added
def water_added : ℝ := 9

-- Theorem stating the container capacity is 20 liters
theorem container_capacity_is_20 :
  (final_fill_percentage * container_capacity - initial_fill_percentage * container_capacity = water_added) ∧
  (container_capacity = 20) :=
sorry

end container_capacity_is_20_l1892_189273


namespace inscribed_sphere_volume_l1892_189222

theorem inscribed_sphere_volume (r h l : ℝ) (V : ℝ) :
  r = 2 →
  2 * π * r * l = 8 * π →
  h^2 + r^2 = l^2 →
  (h - V^(1/3) * ((3 * r) / (4 * π))^(1/3)) / l = V^(1/3) * ((3 * r) / (4 * π))^(1/3) / r →
  V = (32 * Real.sqrt 3) / 27 * π :=
by sorry

end inscribed_sphere_volume_l1892_189222


namespace some_number_approximation_l1892_189247

/-- Given that (3.241 * 14) / x = 0.045374000000000005, prove that x ≈ 1000 -/
theorem some_number_approximation (x : ℝ) 
  (h : (3.241 * 14) / x = 0.045374000000000005) : 
  ∃ ε > 0, |x - 1000| < ε :=
sorry

end some_number_approximation_l1892_189247


namespace min_value_theorem_l1892_189272

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : m * 1 - n * (-1) - 1 = 0) : 
  (1 / m + 2 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

#check min_value_theorem

end min_value_theorem_l1892_189272


namespace right_triangle_area_l1892_189291

/-- Given a right-angled triangle with height 5 cm and median to hypotenuse 6 cm, its area is 30 cm². -/
theorem right_triangle_area (h : ℝ) (m : ℝ) (area : ℝ) : 
  h = 5 → m = 6 → area = (1/2) * (2*m) * h → area = 30 := by
  sorry

end right_triangle_area_l1892_189291


namespace intersection_complement_equality_l1892_189204

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 1}

-- Define set N
def N : Set ℝ := {x : ℝ | x ≤ 0}

-- State the theorem
theorem intersection_complement_equality :
  M ∩ (U \ N) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end intersection_complement_equality_l1892_189204


namespace value_of_y_l1892_189256

theorem value_of_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := by
  sorry

end value_of_y_l1892_189256


namespace largest_angle_obtuse_l1892_189233

-- Define a triangle with altitudes
structure TriangleWithAltitudes where
  h₁ : ℝ
  h₂ : ℝ
  h₃ : ℝ
  h₁_pos : h₁ > 0
  h₂_pos : h₂ > 0
  h₃_pos : h₃ > 0

-- Define the property of having a specific set of altitudes
def hasAltitudes (t : TriangleWithAltitudes) : Prop :=
  t.h₁ = 8 ∧ t.h₂ = 10 ∧ t.h₃ = 25

-- Define an obtuse angle
def isObtuse (θ : ℝ) : Prop :=
  θ > Real.pi / 2 ∧ θ < Real.pi

-- Theorem statement
theorem largest_angle_obtuse (t : TriangleWithAltitudes) (h : hasAltitudes t) :
  ∃ θ, isObtuse θ ∧ (∀ φ, φ ≤ θ) :=
sorry

end largest_angle_obtuse_l1892_189233


namespace cafeteria_green_apples_l1892_189212

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 42

/-- The number of students who wanted fruit -/
def students_wanting_fruit : ℕ := 9

/-- The number of extra fruit the cafeteria ended up with -/
def extra_fruit : ℕ := 40

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 7

theorem cafeteria_green_apples :
  red_apples + green_apples - students_wanting_fruit = extra_fruit :=
by sorry

end cafeteria_green_apples_l1892_189212


namespace prime_square_difference_l1892_189231

theorem prime_square_difference (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (hp_form : ∃ k, p = 4*k + 3) (hq_form : ∃ k, q = 4*k + 3)
  (h_exists : ∃ (x y : ℤ), x^2 - p*q*y^2 = 1) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (p*a^2 - q*b^2 = 1 ∨ q*b^2 - p*a^2 = 1) :=
by sorry

end prime_square_difference_l1892_189231


namespace cos_165_degrees_l1892_189288

theorem cos_165_degrees : 
  Real.cos (165 * π / 180) = -(Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end cos_165_degrees_l1892_189288


namespace sum_of_reciprocals_bound_l1892_189237

theorem sum_of_reciprocals_bound {α β k : ℝ} (hα : α > 0) (hβ : β > 0) (hk : k > 0)
  (hαβ : α ≠ β) (hfα : |Real.log α| = k) (hfβ : |Real.log β| = k) :
  1 / α + 1 / β > 2 := by
  sorry

end sum_of_reciprocals_bound_l1892_189237


namespace area_NPQ_approx_l1892_189258

/-- Triangle XYZ with given side lengths -/
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ
  xy_length : dist X Y = 15
  xz_length : dist X Z = 20
  yz_length : dist Y Z = 13

/-- P is the circumcenter of triangle XYZ -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Q is the incenter of triangle XYZ -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- N is the center of a circle tangent to sides XZ, YZ, and the circumcircle of XYZ -/
def excircle_center (t : Triangle) : ℝ × ℝ := sorry

/-- The area of triangle NPQ -/
def area_NPQ (t : Triangle) : ℝ := sorry

/-- Theorem stating the area of triangle NPQ is approximately 49.21 -/
theorem area_NPQ_approx (t : Triangle) : 
  abs (area_NPQ t - 49.21) < 0.01 := by sorry

end area_NPQ_approx_l1892_189258


namespace empty_solution_set_implies_a_range_l1892_189292

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 4 ≥ 0) → a ∈ Set.Icc (-4 : ℝ) 4 := by
sorry

end empty_solution_set_implies_a_range_l1892_189292


namespace largest_square_with_three_lattice_points_l1892_189226

/-- A lattice point in a 2D plane. -/
def LatticePoint (p : ℝ × ℝ) : Prop := Int.floor p.1 = p.1 ∧ Int.floor p.2 = p.2

/-- A square in a 2D plane. -/
structure Square where
  center : ℝ × ℝ
  sideLength : ℝ
  rotation : ℝ  -- Angle of rotation in radians

/-- Predicate to check if a point is in the interior of a square. -/
def IsInteriorPoint (s : Square) (p : ℝ × ℝ) : Prop := sorry

/-- The number of lattice points in the interior of a square. -/
def InteriorLatticePointCount (s : Square) : ℕ := sorry

/-- Theorem stating that the area of the largest square containing exactly three lattice points in its interior is 5. -/
theorem largest_square_with_three_lattice_points :
  ∃ (s : Square), InteriorLatticePointCount s = 3 ∧
    ∀ (s' : Square), InteriorLatticePointCount s' = 3 → s'.sideLength^2 ≤ s.sideLength^2 ∧
    s.sideLength^2 = 5 := by sorry

end largest_square_with_three_lattice_points_l1892_189226


namespace income_difference_l1892_189268

-- Define the incomes of A and B
def A (B : ℝ) : ℝ := 0.75 * B

-- Theorem statement
theorem income_difference (B : ℝ) (h : B > 0) : 
  (B - A B) / (A B) = 1/3 := by sorry

end income_difference_l1892_189268


namespace polynomial_division_remainder_l1892_189298

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^4 + X^2 + 1 : Polynomial ℝ) = q * (X^2 - 4*X + 7) + (12*X - 69) :=
by sorry

end polynomial_division_remainder_l1892_189298


namespace smallest_prime_divisor_of_sum_l1892_189293

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), 
    Nat.Prime p ∧ 
    p ∣ (3^11 + 5^13) ∧ 
    ∀ (q : ℕ), Nat.Prime q → q ∣ (3^11 + 5^13) → p ≤ q ∧
    p = 2 :=
by sorry

end smallest_prime_divisor_of_sum_l1892_189293


namespace library_initial_books_l1892_189254

/-- The number of books purchased last year -/
def books_last_year : ℕ := 50

/-- The number of books purchased this year -/
def books_this_year : ℕ := 3 * books_last_year

/-- The total number of books in the library now -/
def total_books_now : ℕ := 300

/-- The number of books in the library before the new purchases last year -/
def initial_books : ℕ := total_books_now - books_last_year - books_this_year

theorem library_initial_books :
  initial_books = 100 := by sorry

end library_initial_books_l1892_189254


namespace exists_zero_term_l1892_189289

def recursion (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 
    (a n ≥ b n → a (n + 1) = a n - b n ∧ b (n + 1) = 2 * b n) ∧
    (a n < b n → a (n + 1) = 2 * a n ∧ b (n + 1) = b n - a n)

theorem exists_zero_term (a b : ℕ → ℕ) :
  recursion a b →
  (∃ k : ℕ, a k = 0) ↔
  (∃ m : ℕ, m > 0 ∧ (a 1 + b 1) / Nat.gcd (a 1) (b 1) = 2^m) :=
sorry

end exists_zero_term_l1892_189289


namespace base12Addition_l1892_189209

/-- Converts a base 12 number represented as a list of digits to its decimal equivalent -/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 12 + d) 0

/-- Converts a decimal number to its base 12 representation -/
def decimalToBase12 (n : Nat) : List Nat :=
  if n < 12 then [n]
  else (n % 12) :: decimalToBase12 (n / 12)

/-- Represents the base 12 number 857₁₂ -/
def num1 : List Nat := [7, 5, 8]

/-- Represents the base 12 number 296₁₂ -/
def num2 : List Nat := [6, 9, 2]

/-- Represents the base 12 number B31₁₂ -/
def result : List Nat := [1, 3, 11]

theorem base12Addition :
  decimalToBase12 (base12ToDecimal num1 + base12ToDecimal num2) = result := by
  sorry

#eval base12ToDecimal num1
#eval base12ToDecimal num2
#eval base12ToDecimal result
#eval decimalToBase12 (base12ToDecimal num1 + base12ToDecimal num2)

end base12Addition_l1892_189209


namespace set_equality_l1892_189243

theorem set_equality : Set ℝ := by
  have h1 : Set ℝ := {x | x = -2 ∨ x = 1}
  have h2 : Set ℝ := {x | (x - 1) * (x + 2) = 0}
  sorry

#check set_equality

end set_equality_l1892_189243


namespace pizza_slices_left_l1892_189290

theorem pizza_slices_left (total_slices : ℕ) (eaten_fraction : ℚ) (h1 : total_slices = 16) (h2 : eaten_fraction = 3/4) : 
  total_slices * (1 - eaten_fraction) = 4 := by
  sorry

end pizza_slices_left_l1892_189290


namespace increasing_function_inequality_l1892_189294

theorem increasing_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_sum_positive : a + b > 0) : 
  f a + f b > f (-a) + f (-b) := by
  sorry

end increasing_function_inequality_l1892_189294


namespace interest_rate_calculation_l1892_189245

-- Define the compound interest function
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n

-- State the theorem
theorem interest_rate_calculation (P : ℝ) (r : ℝ) 
  (h1 : compound_interest P r 6 = 6000)
  (h2 : compound_interest P r 7 = 7500) :
  r = 0.25 := by sorry

end interest_rate_calculation_l1892_189245


namespace equation_has_three_solutions_l1892_189206

-- Define the complex polynomial in the numerator
def numerator (z : ℂ) : ℂ := z^4 - 1

-- Define the complex polynomial in the denominator
def denominator (z : ℂ) : ℂ := z^3 - 3*z + 2

-- Define the equation
def equation (z : ℂ) : Prop := numerator z = 0 ∧ denominator z ≠ 0

-- Theorem statement
theorem equation_has_three_solutions :
  ∃ (s : Finset ℂ), s.card = 3 ∧ (∀ z ∈ s, equation z) ∧ (∀ z, equation z → z ∈ s) :=
sorry

end equation_has_three_solutions_l1892_189206


namespace two_valid_configurations_l1892_189228

/-- Represents a quadrant in the yard --/
inductive Quadrant
| I
| II
| III
| IV

/-- Represents a configuration of apple trees in the yard --/
def Configuration := Quadrant → Nat

/-- Checks if a configuration is valid (total of 4 trees) --/
def is_valid_configuration (c : Configuration) : Prop :=
  c Quadrant.I + c Quadrant.II + c Quadrant.III + c Quadrant.IV = 4

/-- Checks if a configuration has equal trees on both sides of each path --/
def is_balanced_configuration (c : Configuration) : Prop :=
  c Quadrant.I + c Quadrant.II = c Quadrant.III + c Quadrant.IV ∧
  c Quadrant.I + c Quadrant.IV = c Quadrant.II + c Quadrant.III ∧
  c Quadrant.I + c Quadrant.III = c Quadrant.II + c Quadrant.IV

/-- Theorem: There exist at least two different valid and balanced configurations --/
theorem two_valid_configurations : ∃ (c1 c2 : Configuration),
  c1 ≠ c2 ∧
  is_valid_configuration c1 ∧
  is_valid_configuration c2 ∧
  is_balanced_configuration c1 ∧
  is_balanced_configuration c2 :=
sorry

end two_valid_configurations_l1892_189228


namespace fraction_evaluation_l1892_189299

theorem fraction_evaluation (x : ℝ) (h : x = 8) :
  (x^10 - 32*x^5 + 1024) / (x^5 - 32) = 32768 := by
  sorry

end fraction_evaluation_l1892_189299


namespace number_of_divisors_30030_l1892_189275

theorem number_of_divisors_30030 : Nat.card {d : ℕ | d > 0 ∧ 30030 % d = 0} = 64 := by
  sorry

end number_of_divisors_30030_l1892_189275


namespace function_property_l1892_189241

open Set Function Real

theorem function_property (f : ℝ → ℝ) (a : ℝ) 
  (h : ∀ x, x ≠ a → f x ≠ f a) :
  (∀ x, x ≠ a → f x ≠ f a) ∧ 
  ¬(∀ x, f x ≠ f a → x ≠ a) := by
  sorry

end function_property_l1892_189241


namespace exactly_fourteen_numbers_l1892_189207

/-- A function that reverses a two-digit number -/
def reverse_two_digit (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- The property that a number satisfies the given condition -/
def satisfies_condition (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ ∃ k : ℕ, (reverse_two_digit n - n) = k^2

/-- The theorem stating that there are exactly 14 numbers satisfying the condition -/
theorem exactly_fourteen_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_condition n) ∧ s.card = 14 :=
sorry

end exactly_fourteen_numbers_l1892_189207


namespace parabola_properties_l1892_189202

/-- Represents a parabola of the form y = a(x-3)^2 + 2 -/
structure Parabola where
  a : ℝ

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem stating properties of a specific parabola -/
theorem parabola_properties (p : Parabola) (A B : Point) :
  (p.a * (1 - 3)^2 + 2 = -2) →  -- parabola passes through (1, -2)
  (A.y = p.a * (A.x - 3)^2 + 2) →  -- point A is on the parabola
  (B.y = p.a * (B.x - 3)^2 + 2) →  -- point B is on the parabola
  (A.x < B.x) →  -- m < n
  (B.x < 3) →  -- n < 3
  (p.a = -1 ∧ A.y < B.y) := by
  sorry

end parabola_properties_l1892_189202


namespace double_average_l1892_189250

theorem double_average (n : ℕ) (original_avg : ℚ) (new_avg : ℚ) : 
  n = 11 → original_avg = 36 → new_avg = 2 * original_avg → new_avg = 72 := by
  sorry

end double_average_l1892_189250


namespace cup_arrangement_theorem_l1892_189246

/-- Represents the number of ways to arrange cups in a circular pattern -/
def circularArrangements (yellow blue red : ℕ) : ℕ := sorry

/-- Represents the number of ways to arrange cups in a circular pattern with adjacent red cups -/
def circularArrangementsAdjacentRed (yellow blue red : ℕ) : ℕ := sorry

/-- The main theorem stating the number of valid arrangements -/
theorem cup_arrangement_theorem :
  circularArrangements 4 3 2 - circularArrangementsAdjacentRed 4 3 2 = 105 := by
  sorry

end cup_arrangement_theorem_l1892_189246


namespace rectangular_parallelepiped_edge_sum_l1892_189277

theorem rectangular_parallelepiped_edge_sum (a b c : ℕ) (V : ℕ) : 
  V = a * b * c → 
  V.Prime → 
  V > 2 → 
  Odd (a + b + c) := by
sorry

end rectangular_parallelepiped_edge_sum_l1892_189277


namespace increasing_cubic_function_condition_l1892_189252

/-- The function f(x) = 2x^3 - 3mx^2 + 6x is increasing on (1, +∞) if and only if m ≤ 2 -/
theorem increasing_cubic_function_condition (m : ℝ) :
  (∀ x > 1, Monotone (fun x => 2*x^3 - 3*m*x^2 + 6*x)) ↔ m ≤ 2 := by
  sorry

end increasing_cubic_function_condition_l1892_189252


namespace gcd_282_470_l1892_189210

theorem gcd_282_470 : Nat.gcd 282 470 = 94 := by
  sorry

end gcd_282_470_l1892_189210


namespace ticket_sales_result_l1892_189214

/-- Represents a section in the stadium -/
structure Section where
  name : String
  seats : Nat
  price : Nat

/-- Represents the stadium configuration -/
def Stadium : List Section := [
  ⟨"A", 40, 10⟩,
  ⟨"B", 30, 15⟩,
  ⟨"C", 25, 20⟩
]

/-- Theorem stating the result of the ticket sales -/
theorem ticket_sales_result 
  (children : Nat) (adults : Nat) (seniors : Nat)
  (h1 : children = 52)
  (h2 : adults = 29)
  (h3 : seniors = 15)
  (h4 : children + adults + seniors = Stadium.foldr (fun s acc => s.seats + acc) 0 + 1) :
  (∀ s : Section, s ∈ Stadium → 
    (if s.name = "A" then adults + seniors else children) ≥ s.seats) ∧
  (Stadium.foldr (fun s acc => s.seats * s.price + acc) 0 = 1350) := by
  sorry

#check ticket_sales_result

end ticket_sales_result_l1892_189214


namespace y_coordinate_range_of_C_l1892_189296

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = x + 4

-- Define point A
def A : ℝ × ℝ := (0, 2)

-- Define perpendicularity of line segments
def perpendicular (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0

-- Theorem statement
theorem y_coordinate_range_of_C 
  (B C : ℝ × ℝ) 
  (hB : parabola B.1 B.2)
  (hC : parabola C.1 C.2)
  (h_perp : perpendicular A B C) :
  C.2 ≤ 0 ∨ C.2 ≥ 4 := by
  sorry


end y_coordinate_range_of_C_l1892_189296


namespace simplify_expression_l1892_189263

theorem simplify_expression : (324 : ℝ)^(1/4) * (98 : ℝ)^(1/2) = 42 := by
  sorry

end simplify_expression_l1892_189263


namespace arithmetic_harmonic_geometric_proportion_l1892_189255

theorem arithmetic_harmonic_geometric_proportion (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / ((a + b) / 2) = (2 * a * b / (a + b)) / b := by
  sorry

end arithmetic_harmonic_geometric_proportion_l1892_189255


namespace square_sum_of_system_l1892_189257

theorem square_sum_of_system (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 10344 / 169 := by
sorry

end square_sum_of_system_l1892_189257


namespace magic_shop_change_theorem_final_change_theorem_l1892_189224

/-- Represents the currency system in the magic shop -/
structure MagicShopCurrency where
  silver_to_gold_rate : ℚ
  cloak_price_gold : ℚ

/-- Calculate the change in silver coins when buying a cloak with gold coins -/
def change_in_silver (c : MagicShopCurrency) (gold_paid : ℚ) : ℚ :=
  (gold_paid - c.cloak_price_gold) * (1 / c.silver_to_gold_rate)

/-- Theorem: Buying a cloak with 14 gold coins results in 10 silver coins as change -/
theorem magic_shop_change_theorem (c : MagicShopCurrency) 
  (h1 : 20 = c.cloak_price_gold * c.silver_to_gold_rate + 4 * c.silver_to_gold_rate)
  (h2 : 15 = c.cloak_price_gold * c.silver_to_gold_rate + 1 * c.silver_to_gold_rate) :
  change_in_silver c 14 = 10 := by
  sorry

/-- The correct change is 10 silver coins -/
def correct_change : ℚ := 10

/-- The final theorem stating the correct change -/
theorem final_change_theorem (c : MagicShopCurrency) 
  (h1 : 20 = c.cloak_price_gold * c.silver_to_gold_rate + 4 * c.silver_to_gold_rate)
  (h2 : 15 = c.cloak_price_gold * c.silver_to_gold_rate + 1 * c.silver_to_gold_rate) :
  change_in_silver c 14 = correct_change := by
  sorry

end magic_shop_change_theorem_final_change_theorem_l1892_189224


namespace perpendicular_to_vertical_line_l1892_189213

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A vertical line represented by its x-coordinate -/
structure VerticalLine where
  x : ℝ

/-- Two lines are perpendicular if one is vertical and the other is horizontal -/
def isPerpendicular (l : Line) (v : VerticalLine) : Prop :=
  l.slope = 0

theorem perpendicular_to_vertical_line (k : ℝ) :
  isPerpendicular (Line.mk k 1) (VerticalLine.mk 1) → k = 0 := by
  sorry

end perpendicular_to_vertical_line_l1892_189213


namespace bubble_sort_probability_bubble_sort_probability_proof_l1892_189287

/-- The probability that the 10th element in a random sequence of 50 distinct elements 
    will end up in the 25th position after one bubble pass -/
theorem bubble_sort_probability (n : ℕ) (h : n = 50) : ℝ :=
  24 / 25

/-- Proof of the bubble_sort_probability theorem -/
theorem bubble_sort_probability_proof (n : ℕ) (h : n = 50) : 
  bubble_sort_probability n h = 24 / 25 := by
  sorry

end bubble_sort_probability_bubble_sort_probability_proof_l1892_189287


namespace vector_dot_product_problem_l1892_189284

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem vector_dot_product_problem (x : ℝ) :
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (5, -3)
  dot_product a b = 7 → x = 2 := by
sorry

end vector_dot_product_problem_l1892_189284


namespace prime_sqrt_sum_integer_l1892_189259

theorem prime_sqrt_sum_integer (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ n : ℕ, ∃ m : ℕ, (Nat.sqrt (p + n) + Nat.sqrt n : ℕ) = m :=
sorry

end prime_sqrt_sum_integer_l1892_189259


namespace range_of_a_l1892_189279

-- Define the proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, Real.exp x - a ≥ 0

-- State the theorem
theorem range_of_a (a : ℝ) : p a ↔ a ∈ Set.Iic (Real.exp 1) :=
sorry

end range_of_a_l1892_189279


namespace equilateral_roots_l1892_189271

/-- Given complex numbers p and q, and z₁ and z₂ being the roots of z² + pz + q = 0
    such that 0, z₁, and z₂ form an equilateral triangle in the complex plane,
    prove that p²/q = 1 -/
theorem equilateral_roots (p q z₁ z₂ : ℂ) : 
  z₁^2 + p*z₁ + q = 0 ∧ 
  z₂^2 + p*z₂ + q = 0 ∧ 
  ∃ ω : ℂ, ω^3 = 1 ∧ ω ≠ 1 ∧ z₂ = ω * z₁ →
  p^2 / q = 1 := by
  sorry

end equilateral_roots_l1892_189271


namespace simple_interest_rate_is_five_percent_l1892_189249

/-- Calculates the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

/-- Theorem: The simple interest rate is 5% given the problem conditions -/
theorem simple_interest_rate_is_five_percent :
  simple_interest_rate 750 900 4 = 5 := by
  sorry

end simple_interest_rate_is_five_percent_l1892_189249


namespace binomial_n_value_l1892_189235

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_n_value (X : BinomialRV) 
  (h_exp : expectation X = 2)
  (h_var : variance X = 3/2) :
  X.n = 8 := by sorry

end binomial_n_value_l1892_189235


namespace trade_value_trade_value_correct_l1892_189203

theorem trade_value (matt_cards : ℕ) (matt_card_value : ℕ) 
  (traded_cards : ℕ) (received_cheap_cards : ℕ) (cheap_card_value : ℕ) 
  (profit : ℕ) : ℕ :=
  let total_traded_value := traded_cards * matt_card_value
  let received_cheap_value := received_cheap_cards * cheap_card_value
  let total_received_value := total_traded_value + profit
  total_received_value - received_cheap_value

#check trade_value 8 6 2 3 2 3 = 9

theorem trade_value_correct : trade_value 8 6 2 3 2 3 = 9 := by
  sorry

end trade_value_trade_value_correct_l1892_189203


namespace A_more_likely_to_win_prob_at_least_one_wins_l1892_189211

-- Define the probabilities for A and B in each round
def prob_A_first : ℚ := 3/5
def prob_A_second : ℚ := 2/3
def prob_B_first : ℚ := 3/4
def prob_B_second : ℚ := 2/5

-- Define the probability of winning for each participant
def prob_A_win : ℚ := prob_A_first * prob_A_second
def prob_B_win : ℚ := prob_B_first * prob_B_second

-- Theorem 1: A has a greater probability of winning than B
theorem A_more_likely_to_win : prob_A_win > prob_B_win := by sorry

-- Theorem 2: The probability that at least one of A and B wins is 29/50
theorem prob_at_least_one_wins : 1 - (1 - prob_A_win) * (1 - prob_B_win) = 29/50 := by sorry

end A_more_likely_to_win_prob_at_least_one_wins_l1892_189211


namespace blown_away_leaves_calculation_mikeys_leaves_calculation_l1892_189261

/-- Given an initial number of leaves and the number of leaves remaining,
    calculate the number of leaves that blew away. -/
def leaves_blown_away (initial_leaves remaining_leaves : ℕ) : ℕ :=
  initial_leaves - remaining_leaves

/-- Theorem: The number of leaves that blew away is equal to the difference
    between the initial number of leaves and the remaining number of leaves. -/
theorem blown_away_leaves_calculation 
  (initial_leaves remaining_leaves : ℕ) 
  (h : initial_leaves ≥ remaining_leaves) :
  leaves_blown_away initial_leaves remaining_leaves = initial_leaves - remaining_leaves :=
by
  sorry

/-- In Mikey's specific case -/
theorem mikeys_leaves_calculation :
  leaves_blown_away 356 112 = 244 :=
by
  sorry

end blown_away_leaves_calculation_mikeys_leaves_calculation_l1892_189261


namespace arithmetic_sequence_difference_l1892_189240

/-- An arithmetic sequence with sum Sn for the first n terms -/
structure ArithmeticSequence where
  Sn : ℕ → ℚ
  a : ℕ → ℚ
  d : ℚ
  sum_formula : ∀ n, Sn n = n * (2 * a 1 + (n - 1) * d) / 2
  term_formula : ∀ n, a n = a 1 + (n - 1) * d

/-- The common difference of the arithmetic sequence is 1/5 -/
theorem arithmetic_sequence_difference (seq : ArithmeticSequence) 
    (h1 : seq.Sn 5 = 6)
    (h2 : seq.a 2 = 1) : 
  seq.d = 1/5 := by
  sorry

end arithmetic_sequence_difference_l1892_189240


namespace auction_sale_total_l1892_189267

/-- Calculate the total amount received from selling a TV and a phone at an auction -/
theorem auction_sale_total (tv_initial_cost phone_initial_cost : ℚ) 
  (tv_price_increase phone_price_increase : ℚ) : ℚ :=
  by
  -- Define the initial costs and price increases
  have h1 : tv_initial_cost = 500 := by sorry
  have h2 : tv_price_increase = 2 / 5 := by sorry
  have h3 : phone_initial_cost = 400 := by sorry
  have h4 : phone_price_increase = 40 / 100 := by sorry

  -- Calculate the final prices
  let tv_final_price := tv_initial_cost + tv_initial_cost * tv_price_increase
  let phone_final_price := phone_initial_cost + phone_initial_cost * phone_price_increase

  -- Calculate the total amount received
  let total_amount := tv_final_price + phone_final_price

  -- Prove that the total amount is equal to 1260
  sorry

end auction_sale_total_l1892_189267


namespace quadrilateral_numbers_multiple_of_14_l1892_189274

def quadrilateral_number (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

def is_multiple_of_14 (n : ℕ) : Prop := ∃ k : ℕ, n = 14 * k

theorem quadrilateral_numbers_multiple_of_14 (t : ℤ) :
  (∀ n : ℤ, (n = 28 * t ∨ n = 28 * t + 6 ∨ n = 28 * t + 7 ∨ n = 28 * t + 12 ∨ 
             n = 28 * t + 14 ∨ n = 28 * t - 9 ∨ n = 28 * t - 8 ∨ n = 28 * t - 2 ∨ 
             n = 28 * t - 1) → 
    is_multiple_of_14 (quadrilateral_number n.toNat)) ∧
  (∀ n : ℕ, is_multiple_of_14 (quadrilateral_number n) → 
    ∃ t : ℤ, n = (28 * t).toNat ∨ n = (28 * t + 6).toNat ∨ n = (28 * t + 7).toNat ∨ 
              n = (28 * t + 12).toNat ∨ n = (28 * t + 14).toNat ∨ n = (28 * t - 9).toNat ∨ 
              n = (28 * t - 8).toNat ∨ n = (28 * t - 2).toNat ∨ n = (28 * t - 1).toNat) :=
by
  sorry


end quadrilateral_numbers_multiple_of_14_l1892_189274


namespace tangent_product_l1892_189230

theorem tangent_product (α β : Real) 
  (h1 : Real.cos (α + β) = 1/3)
  (h2 : Real.cos (α - β) = 1/5) :
  Real.tan α * Real.tan β = -1/4 := by
  sorry

end tangent_product_l1892_189230


namespace namjoon_marbles_l1892_189205

def marble_problem (sets : ℕ) (marbles_per_set : ℕ) (boxes : ℕ) (marbles_per_box : ℕ) : ℕ :=
  boxes * marbles_per_box - sets * marbles_per_set

theorem namjoon_marbles : marble_problem 3 7 6 6 = 15 := by
  sorry

end namjoon_marbles_l1892_189205


namespace computer_table_markup_l1892_189283

/-- The percentage markup on a product's cost price, given its selling price and cost price. -/
def percentageMarkup (sellingPrice costPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

/-- Theorem stating that the percentage markup on a computer table with a selling price of 8215 
    and a cost price of 6625 is 24%. -/
theorem computer_table_markup :
  percentageMarkup 8215 6625 = 24 := by
  sorry

end computer_table_markup_l1892_189283


namespace congruence_problem_l1892_189260

theorem congruence_problem (c d : ℤ) (h_c : c ≡ 25 [ZMOD 53]) (h_d : d ≡ 88 [ZMOD 53]) :
  ∃ m : ℤ, m = 149 ∧ 150 ≤ m ∧ m ≤ 200 ∧ c - d ≡ m [ZMOD 53] ∧
  ∀ k : ℤ, 150 ≤ k ∧ k ≤ 200 ∧ c - d ≡ k [ZMOD 53] → k ≤ m :=
by sorry

end congruence_problem_l1892_189260


namespace smallest_number_with_property_l1892_189285

theorem smallest_number_with_property : ∃! n : ℕ, 
  n > 0 ∧
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  (∀ m : ℕ, m > 0 ∧
    m % 2 = 1 ∧
    m % 3 = 2 ∧
    m % 4 = 3 ∧
    m % 5 = 4 ∧
    m % 6 = 5 ∧
    m % 7 = 6 ∧
    m % 8 = 7 ∧
    m % 9 = 8 ∧
    m % 10 = 9 → m ≥ n) ∧
  n = 2519 :=
by sorry

end smallest_number_with_property_l1892_189285


namespace polynomial_characterization_l1892_189286

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that must be satisfied by a, b, and c -/
def SatisfiesCondition (a b c : ℝ) : Prop :=
  a * b + b * c + c * a = 0

/-- The equation that P must satisfy for all a, b, c satisfying the condition -/
def SatisfiesEquation (P : RealPolynomial) : Prop :=
  ∀ (a b c : ℝ), SatisfiesCondition a b c →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

/-- The form of the polynomial we're trying to prove -/
def IsQuarticQuadratic (P : RealPolynomial) : Prop :=
  ∃ (α β : ℝ), ∀ x, P x = α * x^4 + β * x^2

theorem polynomial_characterization (P : RealPolynomial) :
  SatisfiesEquation P → IsQuarticQuadratic P :=
sorry

end polynomial_characterization_l1892_189286


namespace angle_construction_error_bound_l1892_189270

/-- Represents a 4-digit trigonometric table -/
structure TrigTable :=
  (sin : ℚ → ℚ)
  (cos : ℚ → ℚ)
  (precision : ℕ := 4)

/-- Represents the construction of a regular polygon -/
structure RegularPolygon :=
  (sides : ℕ)
  (centralAngle : ℚ)

/-- The error bound for angle construction using a 4-digit trig table -/
def angleErrorBound (p : RegularPolygon) (t : TrigTable) : ℚ := sorry

theorem angle_construction_error_bound 
  (p : RegularPolygon) 
  (t : TrigTable) 
  (h1 : p.sides = 18) 
  (h2 : p.centralAngle = 20) 
  (h3 : t.precision = 4) :
  angleErrorBound p t < 21 / 3600 := by sorry

end angle_construction_error_bound_l1892_189270


namespace pulley_system_velocity_l1892_189232

/-- A simple pulley system with two loads and a lever -/
structure PulleySystem where
  /-- Velocity of the left load in m/s -/
  v : ℝ
  /-- Velocity of the right load in m/s -/
  u : ℝ

/-- The pulley system satisfies the given conditions -/
def satisfies_conditions (sys : PulleySystem) : Prop :=
  sys.v = 0.5 ∧ 
  -- The strings are inextensible and weightless, and the lever is rigid
  -- (These conditions are implicitly assumed in the relationship between u and v)
  sys.u = 2/7

theorem pulley_system_velocity : 
  ∀ (sys : PulleySystem), satisfies_conditions sys → sys.u = 2/7 := by
  sorry

end pulley_system_velocity_l1892_189232


namespace equation_solutions_l1892_189278

theorem equation_solutions :
  (∀ x : ℝ, 9 * x^2 - 25 = 0 ↔ x = 5/3 ∨ x = -5/3) ∧
  (∀ x : ℝ, (x + 1)^3 - 27 = 0 ↔ x = 2) := by
  sorry

end equation_solutions_l1892_189278


namespace special_circle_equation_l1892_189276

/-- A circle symmetric about the y-axis, passing through (1,0), 
    and divided by the x-axis into arc lengths with ratio 1:2 -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  symmetric_about_y_axis : center.1 = 0
  passes_through_1_0 : (1 - center.1)^2 + (0 - center.2)^2 = radius^2
  arc_ratio : Real.cos (Real.pi / 3) = center.2 / radius

/-- The equation of the special circle -/
def circle_equation (c : SpecialCircle) (x y : ℝ) : Prop :=
  x^2 + (y - c.center.2)^2 = c.radius^2

theorem special_circle_equation (c : SpecialCircle) :
  ∃ a : ℝ, a = Real.sqrt 3 / 3 ∧
    (∀ x y : ℝ, circle_equation c x y ↔ x^2 + (y - a)^2 = 4/3 ∨ x^2 + (y + a)^2 = 4/3) :=
sorry

end special_circle_equation_l1892_189276


namespace root_sum_zero_l1892_189297

theorem root_sum_zero (a b : ℝ) : 
  (Complex.I + 1) ^ 2 + a * (Complex.I + 1) + b = 0 → a + b = 0 := by
  sorry

end root_sum_zero_l1892_189297


namespace yunas_marbles_l1892_189215

/-- Yuna's marble problem -/
theorem yunas_marbles (M : ℕ) : 
  (((M - 12 + 5) / 2 : ℚ) + 3 : ℚ) = 17 → M = 35 := by
  sorry

end yunas_marbles_l1892_189215


namespace max_value_T_l1892_189227

theorem max_value_T (a b c : ℝ) (ha : 1 ≤ a) (ha' : a ≤ 2) 
                     (hb : 1 ≤ b) (hb' : b ≤ 2)
                     (hc : 1 ≤ c) (hc' : c ≤ 2) : 
  (∃ (x y z : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 1 ≤ y ∧ y ≤ 2) (hz : 1 ≤ z ∧ z ≤ 2), 
    (x - y)^2018 + (y - z)^2018 + (z - x)^2018 = 2) ∧ 
  (∀ (x y z : ℝ), 1 ≤ x ∧ x ≤ 2 → 1 ≤ y ∧ y ≤ 2 → 1 ≤ z ∧ z ≤ 2 → 
    (x - y)^2018 + (y - z)^2018 + (z - x)^2018 ≤ 2) :=
by sorry

end max_value_T_l1892_189227


namespace last_four_digits_of_5_power_2017_l1892_189217

theorem last_four_digits_of_5_power_2017 (h1 : 5^5 % 10000 = 3125) 
                                         (h2 : 5^6 % 10000 = 5625) 
                                         (h3 : 5^7 % 10000 = 8125) : 
  5^2017 % 10000 = 3125 := by
sorry

end last_four_digits_of_5_power_2017_l1892_189217


namespace solve_system_l1892_189236

theorem solve_system (p q : ℚ) (eq1 : 5 * p + 3 * q = 10) (eq2 : 3 * p + 5 * q = 20) : p = -5/8 := by
  sorry

end solve_system_l1892_189236


namespace perception_permutations_count_l1892_189248

/-- The number of letters in the word "PERCEPTION" -/
def total_letters : ℕ := 10

/-- The number of repeating letters (E, P, I, N) in "PERCEPTION" -/
def repeating_letters : Finset ℕ := {2, 2, 2, 2}

/-- The number of distinct permutations of the letters in "PERCEPTION" -/
def perception_permutations : ℕ := total_letters.factorial / (repeating_letters.prod (λ x => x.factorial))

theorem perception_permutations_count :
  perception_permutations = 226800 := by sorry

end perception_permutations_count_l1892_189248


namespace line_intersection_x_axis_l1892_189266

/-- A line passing through two points intersects the x-axis --/
theorem line_intersection_x_axis 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂) 
  (h_point1 : x₁ = 8 ∧ y₁ = 2) 
  (h_point2 : x₂ = 4 ∧ y₂ = 6) :
  ∃ x : ℝ, x = 10 ∧ 
    (y₂ - y₁) * (x - x₁) = (x₂ - x₁) * (0 - y₁) :=
by sorry

end line_intersection_x_axis_l1892_189266


namespace jenny_total_wins_l1892_189219

/-- The number of games Jenny played against Mark -/
def games_with_mark : ℕ := 10

/-- The number of games Mark won against Jenny -/
def marks_wins : ℕ := 1

/-- The number of games Jenny played against Jill -/
def games_with_jill : ℕ := 2 * games_with_mark

/-- The percentage of games Jill won against Jenny -/
def jills_win_percentage : ℚ := 75 / 100

theorem jenny_total_wins : 
  (games_with_mark - marks_wins) + 
  (games_with_jill - (jills_win_percentage * games_with_jill).num) = 14 := by
sorry

end jenny_total_wins_l1892_189219


namespace cone_volume_l1892_189218

/-- The volume of a cone with height equal to its radius, where the radius is √m and m is a rational number -/
theorem cone_volume (m : ℚ) (h : m > 0) : 
  let r : ℝ := Real.sqrt m
  let volume := (1/3 : ℝ) * Real.pi * r^2 * r
  volume = (1/3 : ℝ) * Real.pi * m^(3/2) := by
  sorry

end cone_volume_l1892_189218


namespace class_mean_calculation_l1892_189201

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group2_students : ℕ) 
  (group1_mean : ℚ) (group2_mean : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 40 →
  group2_students = 10 →
  group1_mean = 85/100 →
  group2_mean = 80/100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 84/100 := by
sorry

#eval (40 * (85/100) + 10 * (80/100)) / 50

end class_mean_calculation_l1892_189201


namespace tens_digit_of_8_pow_2023_l1892_189221

theorem tens_digit_of_8_pow_2023 : ∃ k : ℕ, 8^2023 = 100 * k + 12 :=
sorry

end tens_digit_of_8_pow_2023_l1892_189221


namespace isabel_paper_count_l1892_189262

/-- Given that Isabel bought some paper, used some, and has some left, 
    prove that the initial amount is the sum of used and left amounts. -/
theorem isabel_paper_count (initial used left : ℕ) 
  (h1 : used = 156)
  (h2 : left = 744)
  (h3 : initial = used + left) : 
  initial = 900 := by sorry

end isabel_paper_count_l1892_189262


namespace arrangement_count_10_l1892_189244

/-- The number of ways to choose a president, vice-president, and committee from a group. -/
def arrangementCount (n : ℕ) : ℕ :=
  let presidentChoices := n
  let vicePresidentChoices := n - 1
  let officerArrangements := presidentChoices * vicePresidentChoices
  let remainingPeople := n - 2
  let committeeArrangements := remainingPeople.choose 3
  officerArrangements * committeeArrangements

/-- Theorem stating the number of arrangements for a group of 10 people. -/
theorem arrangement_count_10 : arrangementCount 10 = 5040 := by
  sorry

end arrangement_count_10_l1892_189244


namespace perpendicular_line_m_value_l1892_189251

/-- Given a line passing through points (m, 3) and (1, m) that is perpendicular
    to a line with slope -1, prove that m = 2. -/
theorem perpendicular_line_m_value (m : ℝ) : 
  (((m - 3) / (1 - m) = 1) ∧ (1 * (-1) = -1)) → m = 2 := by
  sorry

end perpendicular_line_m_value_l1892_189251


namespace abs_difference_implies_abs_inequality_l1892_189264

theorem abs_difference_implies_abs_inequality (a_n l : ℝ) :
  |a_n - l| > 1 → |a_n| > 1 - |l| := by
  sorry

end abs_difference_implies_abs_inequality_l1892_189264


namespace sin_cos_identity_l1892_189281

theorem sin_cos_identity : 
  Real.sin (34 * π / 180) * Real.sin (26 * π / 180) - 
  Real.cos (34 * π / 180) * Real.cos (26 * π / 180) = -1/2 := by
  sorry

end sin_cos_identity_l1892_189281


namespace john_weekly_production_l1892_189238

/-- Calculates the number of widgets John makes in a week -/
def widgets_per_week (widgets_per_hour : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  widgets_per_hour * hours_per_day * days_per_week

/-- Proves that John makes 800 widgets per week -/
theorem john_weekly_production : 
  widgets_per_week 20 8 5 = 800 := by
  sorry

end john_weekly_production_l1892_189238
