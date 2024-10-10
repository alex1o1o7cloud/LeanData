import Mathlib

namespace eccentricity_equation_roots_l3541_354199

/-- The cubic equation whose roots are the eccentricities of a hyperbola, an ellipse, and a parabola -/
def eccentricity_equation (x : ℝ) : Prop :=
  2 * x^3 - 7 * x^2 + 7 * x - 2 = 0

/-- Definition of eccentricity for an ellipse -/
def is_ellipse_eccentricity (e : ℝ) : Prop :=
  0 ≤ e ∧ e < 1

/-- Definition of eccentricity for a parabola -/
def is_parabola_eccentricity (e : ℝ) : Prop :=
  e = 1

/-- Definition of eccentricity for a hyperbola -/
def is_hyperbola_eccentricity (e : ℝ) : Prop :=
  e > 1

/-- The theorem stating that the roots of the equation correspond to the eccentricities of the three conic sections -/
theorem eccentricity_equation_roots :
  ∃ (e₁ e₂ e₃ : ℝ),
    eccentricity_equation e₁ ∧
    eccentricity_equation e₂ ∧
    eccentricity_equation e₃ ∧
    is_ellipse_eccentricity e₁ ∧
    is_parabola_eccentricity e₂ ∧
    is_hyperbola_eccentricity e₃ :=
  sorry

end eccentricity_equation_roots_l3541_354199


namespace cubic_monotonic_and_odd_l3541_354113

def f (x : ℝ) : ℝ := x^3

theorem cubic_monotonic_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (-x) = -f x) := by
sorry

end cubic_monotonic_and_odd_l3541_354113


namespace range_of_a_l3541_354177

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x ∈ Set.Ioo (-1) 1 ∧ a * x^2 - 1 ≥ 0) → a > 1 := by
  sorry

end range_of_a_l3541_354177


namespace square_in_triangle_angle_sum_l3541_354139

/-- The sum of angles in a triangle --/
def triangle_angle_sum : ℝ := 180

/-- The interior angle of an equilateral triangle --/
def equilateral_triangle_angle : ℝ := 60

/-- The sum of angles on a straight line --/
def straight_line_angle_sum : ℝ := 180

/-- The angle of a right angle (in a square) --/
def right_angle : ℝ := 90

/-- Configuration of a square inscribed in an equilateral triangle --/
structure SquareInTriangle where
  x : ℝ  -- Angle between square side and triangle side
  y : ℝ  -- Angle between square side and triangle side
  p : ℝ  -- Complementary angle to x in the triangle
  q : ℝ  -- Complementary angle to y in the triangle

/-- Theorem: The sum of x and y in the SquareInTriangle configuration is 150° --/
theorem square_in_triangle_angle_sum (config : SquareInTriangle) : 
  config.x + config.y = 150 := by
  sorry

end square_in_triangle_angle_sum_l3541_354139


namespace coin_stack_solution_l3541_354164

/-- Thickness of a nickel in millimeters -/
def nickel_thickness : ℚ := 2.05

/-- Thickness of a quarter in millimeters -/
def quarter_thickness : ℚ := 1.65

/-- Height of the stack in millimeters -/
def stack_height : ℚ := 16.5

theorem coin_stack_solution :
  ∃! (n q : ℕ), 
    n * nickel_thickness + q * quarter_thickness = stack_height ∧
    n + q = 9 := by sorry

end coin_stack_solution_l3541_354164


namespace range_of_sin6_plus_cos4_l3541_354138

theorem range_of_sin6_plus_cos4 :
  ∀ x : ℝ, 0 ≤ Real.sin x ^ 6 + Real.cos x ^ 4 ∧
  Real.sin x ^ 6 + Real.cos x ^ 4 ≤ 1 ∧
  (∃ y : ℝ, Real.sin y ^ 6 + Real.cos y ^ 4 = 0) ∧
  (∃ z : ℝ, Real.sin z ^ 6 + Real.cos z ^ 4 = 1) :=
by sorry

end range_of_sin6_plus_cos4_l3541_354138


namespace quadratic_minimum_l3541_354190

theorem quadratic_minimum (x : ℝ) : x^2 - 6*x + 5 ≥ -4 ∧ ∃ y : ℝ, y^2 - 6*y + 5 = -4 := by
  sorry

end quadratic_minimum_l3541_354190


namespace arithmetic_sequence_length_l3541_354153

/-- The number of terms in the arithmetic sequence 2.5, 7.5, 12.5, ..., 57.5, 62.5 -/
def sequenceLength : ℕ := 13

/-- The first term of the sequence -/
def firstTerm : ℚ := 2.5

/-- The last term of the sequence -/
def lastTerm : ℚ := 62.5

/-- The common difference of the sequence -/
def commonDifference : ℚ := 5

theorem arithmetic_sequence_length :
  sequenceLength = (lastTerm - firstTerm) / commonDifference + 1 := by
  sorry

end arithmetic_sequence_length_l3541_354153


namespace perpendicular_slope_l3541_354175

/-- Given a line with equation 5x - 2y = 10, the slope of a perpendicular line is -2/5 -/
theorem perpendicular_slope (x y : ℝ) :
  (5 * x - 2 * y = 10) → 
  (slope_of_perpendicular_line : ℝ) = -2/5 := by
  sorry

end perpendicular_slope_l3541_354175


namespace max_profit_at_10_max_profit_value_l3541_354107

/-- Profit function for location A -/
def L₁ (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def L₂ (x : ℝ) : ℝ := 2 * x

/-- Total profit function -/
def totalProfit (x : ℝ) : ℝ := L₁ x + L₂ (15 - x)

/-- The maximum profit is achieved when selling 10 cars in location A -/
theorem max_profit_at_10 :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 15 → totalProfit x ≤ totalProfit 10 :=
sorry

/-- The maximum profit is 45.6 -/
theorem max_profit_value : totalProfit 10 = 45.6 :=
sorry

end max_profit_at_10_max_profit_value_l3541_354107


namespace slower_speed_calculation_l3541_354188

theorem slower_speed_calculation (distance : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  distance = 50 →
  faster_speed = 14 →
  extra_distance = 20 →
  ∃ slower_speed : ℝ,
    slower_speed > 0 ∧
    distance / slower_speed = distance / faster_speed + extra_distance / faster_speed ∧
    slower_speed = 10 :=
by sorry

end slower_speed_calculation_l3541_354188


namespace lcm_of_6_10_15_l3541_354154

theorem lcm_of_6_10_15 : Nat.lcm (Nat.lcm 6 10) 15 = 30 := by
  sorry

end lcm_of_6_10_15_l3541_354154


namespace tennis_uniform_numbers_l3541_354168

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem tennis_uniform_numbers 
  (e f g h : ℕ) 
  (e_birthday today_date g_birthday : ℕ)
  (h_two_digit_prime : is_two_digit_prime e ∧ is_two_digit_prime f ∧ is_two_digit_prime g ∧ is_two_digit_prime h)
  (h_sum_all : e + f + g + h = e_birthday)
  (h_sum_ef : e + f = today_date)
  (h_sum_gf : g + f = g_birthday)
  (h_sum_hg : h + g = e_birthday) :
  h = 19 := by
  sorry

end tennis_uniform_numbers_l3541_354168


namespace candy_division_l3541_354162

theorem candy_division (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (hpq : p < q) (hqr : q < r) :
  (∃ n : ℕ, n > 0 ∧ n * (r + q - 2 * p) = 39) →
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x + y + (r - p) = 10) →
  (∃ z : ℕ, z > 0 ∧ 18 - 3 * p = 9) →
  (p = 3 ∧ q = 6 ∧ r = 13) := by
sorry

end candy_division_l3541_354162


namespace parabola_shift_theorem_l3541_354160

/-- Represents a parabola in the form y = (x - h)² + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Shifts a point horizontally -/
def shift_horizontal (p : Point) (shift : ℝ) : Point :=
  { x := p.x + shift, y := p.y }

/-- Shifts a point vertically -/
def shift_vertical (p : Point) (shift : ℝ) : Point :=
  { x := p.x, y := p.y + shift }

/-- The vertex of a parabola -/
def vertex (p : Parabola) : Point :=
  { x := p.h, y := p.k }

theorem parabola_shift_theorem (p : Parabola) :
  p.h = 2 ∧ p.k = 3 →
  (shift_vertical (shift_horizontal (vertex p) (-3)) (-5)) = { x := -1, y := -2 } := by
  sorry

end parabola_shift_theorem_l3541_354160


namespace repeating_decimal_equals_fraction_l3541_354193

/-- The repeating decimal 0.51246246246... -/
def repeating_decimal : ℚ := 
  51246 / 100000 + (246 / 100000) * (1 / (1 - 1 / 1000))

/-- The fraction representation -/
def fraction : ℚ := 511734 / 99900

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = fraction := by sorry

end repeating_decimal_equals_fraction_l3541_354193


namespace village_population_l3541_354144

/-- Given a village population with specific demographic percentages,
    calculate the total population. -/
theorem village_population (adult_percentage : ℝ) (adult_women_percentage : ℝ)
    (adult_women_count : ℕ) :
    adult_percentage = 0.9 →
    adult_women_percentage = 0.6 →
    adult_women_count = 21600 →
    ∃ total_population : ℕ,
      total_population = 40000 ∧
      (adult_percentage * adult_women_percentage * total_population : ℝ) = adult_women_count :=
by
  sorry

end village_population_l3541_354144


namespace return_journey_speed_l3541_354150

/-- Calculates the average speed of a return journey given the conditions of the problem -/
theorem return_journey_speed 
  (morning_time : ℝ) 
  (evening_time : ℝ) 
  (morning_speed : ℝ) 
  (h1 : morning_time = 1) 
  (h2 : evening_time = 1.5) 
  (h3 : morning_speed = 30) : 
  (morning_speed * morning_time) / evening_time = 20 :=
by
  sorry

#check return_journey_speed

end return_journey_speed_l3541_354150


namespace cube_root_equation_solution_difference_l3541_354149

theorem cube_root_equation_solution_difference : ∃ x₁ x₂ : ℝ,
  (x₁ ≠ x₂) ∧
  ((9 - x₁^2 / 4)^(1/3 : ℝ) = -3) ∧
  ((9 - x₂^2 / 4)^(1/3 : ℝ) = -3) ∧
  (abs (x₁ - x₂) = 24) :=
by sorry

end cube_root_equation_solution_difference_l3541_354149


namespace prop1_prop4_l3541_354136

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (perp_to_plane : Line → Plane → Prop)

-- Proposition 1
theorem prop1 (a b c : Line) :
  parallel a b → perpendicular b c → perpendicular a c :=
sorry

-- Proposition 4
theorem prop4 (a b : Line) (α : Plane) :
  perp_to_plane a α → contained_in b α → perpendicular a b :=
sorry

end prop1_prop4_l3541_354136


namespace third_degree_polynomial_specific_value_l3541_354101

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- The property that the absolute value of g at certain points equals 10 -/
def HasSpecificValues (g : ThirdDegreePolynomial) : Prop :=
  |g (-1)| = 10 ∧ |g 0| = 10 ∧ |g 2| = 10 ∧ |g 4| = 10 ∧ |g 5| = 10 ∧ |g 8| = 10

/-- The theorem statement -/
theorem third_degree_polynomial_specific_value (g : ThirdDegreePolynomial) 
  (h : HasSpecificValues g) : |g 3| = 22.5 := by
  sorry

end third_degree_polynomial_specific_value_l3541_354101


namespace power_in_denominator_l3541_354110

theorem power_in_denominator (x : ℕ) : (10 ^ 655 / 10 ^ x = 100000) → x = 650 := by
  sorry

end power_in_denominator_l3541_354110


namespace exist_two_N_l3541_354102

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the condition for point M
def M_condition (x y : ℝ) : Prop :=
  Real.sqrt ((x+1)^2 + y^2) + Real.sqrt ((x-1)^2 + y^2) = 2 * Real.sqrt 2

-- Define the line l
def line_l (x : ℝ) : Prop := x = -1/2

-- Define the property that N is the midpoint of AB
def is_midpoint (N A B : ℝ × ℝ) : Prop :=
  N.1 = (A.1 + B.1) / 2 ∧ N.2 = (A.2 + B.2) / 2

-- Define the property that PQ is perpendicular bisector of AB
def is_perp_bisector (P Q A B : ℝ × ℝ) : Prop :=
  (P.1 - Q.1) * (A.1 - B.1) + (P.2 - Q.2) * (A.2 - B.2) = 0 ∧
  (P.1 + Q.1) / 2 = (A.1 + B.1) / 2 ∧
  (P.2 + Q.2) / 2 = (A.2 + B.2) / 2

-- Define the property that (1,0) is on the circle with diameter PQ
def on_circle_PQ (P Q : ℝ × ℝ) : Prop :=
  (1 - P.1) * (1 - Q.1) + (-P.2) * (-Q.2) = 0

-- Main theorem
theorem exist_two_N :
  ∃ N1 N2 : ℝ × ℝ,
    N1 ≠ N2 ∧
    line_l N1.1 ∧ line_l N2.1 ∧
    (∃ A B P Q : ℝ × ℝ,
      E A.1 A.2 ∧ E B.1 B.2 ∧ E P.1 P.2 ∧ E Q.1 Q.2 ∧
      is_midpoint N1 A B ∧
      is_perp_bisector P Q A B ∧
      on_circle_PQ P Q) ∧
    (∃ A B P Q : ℝ × ℝ,
      E A.1 A.2 ∧ E B.1 B.2 ∧ E P.1 P.2 ∧ E Q.1 Q.2 ∧
      is_midpoint N2 A B ∧
      is_perp_bisector P Q A B ∧
      on_circle_PQ P Q) ∧
    N1 = (-1/2, Real.sqrt 19 / 19) ∧
    N2 = (-1/2, -Real.sqrt 19 / 19) ∧
    (∀ N : ℝ × ℝ,
      line_l N.1 →
      (∃ A B P Q : ℝ × ℝ,
        E A.1 A.2 ∧ E B.1 B.2 ∧ E P.1 P.2 ∧ E Q.1 Q.2 ∧
        is_midpoint N A B ∧
        is_perp_bisector P Q A B ∧
        on_circle_PQ P Q) →
      N = N1 ∨ N = N2) :=
sorry

end exist_two_N_l3541_354102


namespace min_n_is_minimum_l3541_354176

/-- The minimum positive integer n for which the expansion of (2x - 1/∛x)^n contains a constant term -/
def min_n : ℕ := 4

/-- Predicate to check if the expansion of (2x - 1/∛x)^n contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, n = 4 * r / 3 ∧ r > 0

/-- Theorem stating that min_n is the minimum positive integer satisfying the condition -/
theorem min_n_is_minimum :
  (∀ k : ℕ, k > 0 ∧ k < min_n → ¬(has_constant_term k)) ∧
  has_constant_term min_n :=
sorry

end min_n_is_minimum_l3541_354176


namespace fiftiethTermIs346_l3541_354141

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
def fiftiethTerm : ℤ := arithmeticSequenceTerm 3 7 50

theorem fiftiethTermIs346 : fiftiethTerm = 346 := by
  sorry

end fiftiethTermIs346_l3541_354141


namespace abc_expression_value_l3541_354115

theorem abc_expression_value (a b c : ℚ) 
  (ha : a^2 = 9)
  (hb : abs b = 4)
  (hc : c^3 = 27)
  (hab : a * b < 0)
  (hbc : b * c > 0) :
  a * b - b * c + c * a = -33 := by
sorry

end abc_expression_value_l3541_354115


namespace perpendicular_planes_necessary_not_sufficient_l3541_354167

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perp_line_plane (m : Line) (α : Plane) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def perp_plane_plane (α β : Plane) : Prop := sorry

/-- Definition of necessary but not sufficient condition -/
def necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem perpendicular_planes_necessary_not_sufficient 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) (h2 : α ≠ β) 
  (h3 : perp_line_plane m α) (h4 : line_in_plane n β) :
  necessary_not_sufficient (perp_plane_plane α β) (parallel m n) := by
  sorry

end perpendicular_planes_necessary_not_sufficient_l3541_354167


namespace circle_center_on_line_l3541_354143

/-- Given a circle x^2 + y^2 + Dx + Ey = 0 with center on the line x + y = l, prove D + E = -2 -/
theorem circle_center_on_line (D E l : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + D*x + E*y = 0 ∧ x + y = l) → D + E = -2 := by
  sorry

end circle_center_on_line_l3541_354143


namespace min_value_of_expression_min_value_is_nine_min_value_achieved_l3541_354197

theorem min_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_perp : m * n + (1 - n) * 1 = 0) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y + (1 - y) * 1 = 0 → 1/x + 4*y ≥ 1/m + 4*n :=
by sorry

theorem min_value_is_nine (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_perp : m * n + (1 - n) * 1 = 0) : 
  1/m + 4*n ≥ 9 :=
by sorry

theorem min_value_achieved (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_perp : m * n + (1 - n) * 1 = 0) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y + (1 - y) * 1 = 0 ∧ 1/x + 4*y = 9 :=
by sorry

end min_value_of_expression_min_value_is_nine_min_value_achieved_l3541_354197


namespace arithmetic_sequence_ratio_l3541_354179

/-- Given two arithmetic sequences, prove the ratio of their 5th terms -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℝ) (S T : ℕ → ℝ) :
  (∀ n, S n / T n = (7 * n) / (n + 3)) →  -- Given condition
  (∀ n, S n = (n / 2) * (a 1 + a n)) →    -- Definition of S_n for arithmetic sequence
  (∀ n, T n = (n / 2) * (b 1 + b n)) →    -- Definition of T_n for arithmetic sequence
  a 5 / b 5 = 21 / 4 := by
  sorry


end arithmetic_sequence_ratio_l3541_354179


namespace cube_sum_divisibility_l3541_354111

theorem cube_sum_divisibility (x y z : ℤ) (h : x^3 + y^3 = z^3) :
  3 ∣ x ∨ 3 ∣ y ∨ 3 ∣ z := by
  sorry

end cube_sum_divisibility_l3541_354111


namespace base_seven_digits_of_2000_l3541_354116

theorem base_seven_digits_of_2000 : ∃ n : ℕ, 
  (7^(n-1) ≤ 2000) ∧ (2000 < 7^n) ∧ (n = 4) := by
  sorry

end base_seven_digits_of_2000_l3541_354116


namespace probability_of_specific_draw_l3541_354100

def total_balls : ℕ := 18
def red_balls : ℕ := 4
def yellow_balls : ℕ := 5
def green_balls : ℕ := 6
def blue_balls : ℕ := 3
def drawn_balls : ℕ := 4

def favorable_outcomes : ℕ := Nat.choose green_balls 2 * Nat.choose red_balls 1 * Nat.choose blue_balls 1

def total_outcomes : ℕ := Nat.choose total_balls drawn_balls

theorem probability_of_specific_draw :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 17 :=
sorry

end probability_of_specific_draw_l3541_354100


namespace power_multiplication_l3541_354109

theorem power_multiplication (a : ℝ) : a^3 * a = a^4 := by
  sorry

end power_multiplication_l3541_354109


namespace sequence_growth_l3541_354195

/-- A sequence of integers satisfying the given conditions -/
def Sequence (a : ℕ → ℤ) : Prop :=
  a 1 > a 0 ∧ a 1 > 0 ∧ ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r

/-- The main theorem -/
theorem sequence_growth (a : ℕ → ℤ) (h : Sequence a) : a 100 > 2^99 := by
  sorry

end sequence_growth_l3541_354195


namespace range_of_b_l3541_354130

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O1 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 4
def line_P (x y b : ℝ) : Prop := x + Real.sqrt 3 * y - b = 0

-- Define the condition for P satisfying PB = 2PA
def condition_P (x y : ℝ) : Prop := x^2 + y^2 + (8/3) * x - 16/3 = 0

-- Theorem statement
theorem range_of_b :
  ∀ b : ℝ, (∃! (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧ 
    line_P p1.1 p1.2 b ∧ 
    line_P p2.1 p2.2 b ∧ 
    condition_P p1.1 p1.2 ∧ 
    condition_P p2.1 p2.2) ↔ 
  -20/3 < b ∧ b < 4 := by
sorry

end range_of_b_l3541_354130


namespace die_volume_l3541_354119

theorem die_volume (side_area : ℝ) (h : side_area = 64) : 
  side_area^(3/2) = 512 := by
  sorry

end die_volume_l3541_354119


namespace shift_left_theorem_l3541_354112

/-- Represents a quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original quadratic function y = x^2 -/
def original : QuadraticFunction := ⟨1, 0, 0⟩

/-- Shifts a quadratic function to the left by h units -/
def shift_left (f : QuadraticFunction) (h : ℝ) : QuadraticFunction :=
  ⟨f.a, f.b + 2 * f.a * h, f.c + f.b * h + f.a * h^2⟩

/-- The shifted quadratic function -/
def shifted : QuadraticFunction := shift_left original 1

theorem shift_left_theorem :
  shifted = ⟨1, 2, 1⟩ := by sorry

end shift_left_theorem_l3541_354112


namespace rectangle_length_l3541_354156

theorem rectangle_length (L B : ℝ) (h1 : L / B = 25 / 16) (h2 : L * B = 200^2) : L = 250 := by
  sorry

end rectangle_length_l3541_354156


namespace top_three_probability_correct_l3541_354189

/-- Represents a knockout tournament with 64 teams. -/
structure Tournament :=
  (teams : Fin 64 → ℕ)
  (distinct_skills : ∀ i j, i ≠ j → teams i ≠ teams j)

/-- The probability of the top three teams finishing in order of their skill levels. -/
def top_three_probability (t : Tournament) : ℚ :=
  512 / 1953

/-- Theorem stating the probability of the top three teams finishing in order of their skill levels. -/
theorem top_three_probability_correct (t : Tournament) : 
  top_three_probability t = 512 / 1953 := by
  sorry

end top_three_probability_correct_l3541_354189


namespace four_at_seven_l3541_354146

-- Define the binary operation @
def binaryOp (a b : ℤ) : ℤ := 4 * a - 2 * b

-- Theorem statement
theorem four_at_seven : binaryOp 4 7 = 2 := by
  sorry

end four_at_seven_l3541_354146


namespace widest_opening_is_f₃_l3541_354104

/-- The quadratic function with the widest opening -/
def widest_opening (f₁ f₂ f₃ f₄ : ℝ → ℝ) : Prop :=
  ∃ (a₁ a₂ a₃ a₄ : ℝ),
    (∀ x, f₁ x = -10 * x^2) ∧
    (∀ x, f₂ x = 2 * x^2) ∧
    (∀ x, f₃ x = (1/100) * x^2) ∧
    (∀ x, f₄ x = -x^2) ∧
    (abs a₃ < abs a₄ ∧ abs a₄ < abs a₂ ∧ abs a₂ < abs a₁)

/-- Theorem stating that f₃ has the widest opening -/
theorem widest_opening_is_f₃ (f₁ f₂ f₃ f₄ : ℝ → ℝ) :
  widest_opening f₁ f₂ f₃ f₄ → (∀ x, f₃ x = (1/100) * x^2) := by
  sorry

end widest_opening_is_f₃_l3541_354104


namespace problem_solution_l3541_354137

theorem problem_solution : 
  (101 * 99 = 9999) ∧ 
  (32 * 2^2 + 14 * 2^3 + 10 * 2^4 = 400) := by
  sorry

end problem_solution_l3541_354137


namespace at_least_one_positive_discriminant_l3541_354161

theorem at_least_one_positive_discriminant 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (4 * b^2 - 4 * a * c > 0) ∨ 
  (4 * c^2 - 4 * a * b > 0) ∨ 
  (4 * a^2 - 4 * b * c > 0) :=
sorry

end at_least_one_positive_discriminant_l3541_354161


namespace shirt_tie_outfits_l3541_354145

theorem shirt_tie_outfits (num_shirts : ℕ) (num_ties : ℕ) 
  (h1 : num_shirts = 6) (h2 : num_ties = 5) : 
  num_shirts * num_ties = 30 := by
  sorry

end shirt_tie_outfits_l3541_354145


namespace second_meeting_time_l3541_354121

-- Define the pool and swimmers
def Pool : Type := Unit
def Swimmer : Type := Unit

-- Define the time to meet in the center
def time_to_center : ℝ := 1.5

-- Define the function to calculate the time for the second meeting
def time_to_second_meeting (p : Pool) (s1 s2 : Swimmer) : ℝ :=
  2 * time_to_center + time_to_center

-- Theorem statement
theorem second_meeting_time (p : Pool) (s1 s2 : Swimmer) :
  time_to_second_meeting p s1 s2 = 4.5 := by
  sorry


end second_meeting_time_l3541_354121


namespace pool_volume_l3541_354133

/-- The volume of a cylindrical pool minus a central cylindrical pillar -/
theorem pool_volume (pool_diameter : ℝ) (pool_depth : ℝ) (pillar_diameter : ℝ) (pillar_depth : ℝ)
  (h1 : pool_diameter = 20)
  (h2 : pool_depth = 5)
  (h3 : pillar_diameter = 4)
  (h4 : pillar_depth = 5) :
  (π * (pool_diameter / 2)^2 * pool_depth) - (π * (pillar_diameter / 2)^2 * pillar_depth) = 480 * π := by
  sorry

#check pool_volume

end pool_volume_l3541_354133


namespace min_value_and_inequality_inequality_holds_l3541_354182

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1 - x^2) / x^2

theorem min_value_and_inequality (a : ℝ) :
  (∃ (x_min : ℝ), x_min > 0 ∧ ∀ (x : ℝ), x > 0 → f a x ≥ f a x_min ∧ f a x_min = 0) ↔ a = 2 :=
sorry

theorem inequality_holds (x : ℝ) (h : x > 0) : f 2 x ≥ 1 / x - Real.exp (1 - x) :=
sorry

end min_value_and_inequality_inequality_holds_l3541_354182


namespace cylinder_surface_area_l3541_354108

/-- The total surface area of a cylinder with height 12 and radius 4 is 128π. -/
theorem cylinder_surface_area :
  let h : ℝ := 12
  let r : ℝ := 4
  let base_area : ℝ := π * r^2
  let lateral_area : ℝ := 2 * π * r * h
  let total_area : ℝ := 2 * base_area + lateral_area
  total_area = 128 * π := by sorry

end cylinder_surface_area_l3541_354108


namespace defective_shipped_percentage_l3541_354123

/-- The percentage of units that are defective -/
def defective_percentage : ℝ := 6

/-- The percentage of defective units that are shipped for sale -/
def shipped_percentage : ℝ := 4

/-- The result we want to prove -/
def result : ℝ := 0.24

theorem defective_shipped_percentage :
  (defective_percentage / 100) * (shipped_percentage / 100) * 100 = result := by
  sorry

end defective_shipped_percentage_l3541_354123


namespace pennys_bakery_revenue_l3541_354166

/-- The revenue calculation for Penny's bakery --/
theorem pennys_bakery_revenue : 
  ∀ (price_per_slice : ℕ) (slices_per_pie : ℕ) (number_of_pies : ℕ),
    price_per_slice = 7 →
    slices_per_pie = 6 →
    number_of_pies = 7 →
    price_per_slice * slices_per_pie * number_of_pies = 294 := by
  sorry

end pennys_bakery_revenue_l3541_354166


namespace unpainted_cubes_count_l3541_354163

/-- Represents a large cube composed of unit cubes -/
structure LargeCube where
  side_length : ℕ
  total_units : ℕ
  painted_on_opposite_faces : ℕ
  painted_on_other_faces : ℕ

/-- Calculates the number of unpainted unit cubes in the large cube -/
def unpainted_cubes (c : LargeCube) : ℕ :=
  c.total_units - (2 * c.painted_on_opposite_faces + 4 * c.painted_on_other_faces - 8)

/-- The theorem to be proved -/
theorem unpainted_cubes_count (c : LargeCube) 
  (h1 : c.side_length = 6)
  (h2 : c.total_units = 216)
  (h3 : c.painted_on_opposite_faces = 16)
  (h4 : c.painted_on_other_faces = 9) :
  unpainted_cubes c = 156 := by
  sorry

end unpainted_cubes_count_l3541_354163


namespace necessary_but_not_sufficient_l3541_354191

theorem necessary_but_not_sufficient :
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (x - 3) < 0)) ∧
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) :=
sorry

end necessary_but_not_sufficient_l3541_354191


namespace west_distance_notation_l3541_354147

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to convert distance and direction to a signed number
def signedDistance (distance : ℝ) (direction : Direction) : ℝ :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

-- State the theorem
theorem west_distance_notation :
  signedDistance 6 Direction.East = 6 →
  signedDistance 10 Direction.West = -10 := by
  sorry

end west_distance_notation_l3541_354147


namespace functional_equation_identity_l3541_354134

theorem functional_equation_identity (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, f (m + f n) = f m + n) : 
  ∀ n : ℕ, f n = n := by
sorry

end functional_equation_identity_l3541_354134


namespace pages_per_day_l3541_354114

theorem pages_per_day (chapters : ℕ) (total_pages : ℕ) (days : ℕ) 
  (h1 : chapters = 41) 
  (h2 : total_pages = 450) 
  (h3 : days = 30) :
  total_pages / days = 15 := by
  sorry

end pages_per_day_l3541_354114


namespace geometric_series_sum_l3541_354148

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum : 
  let a : ℚ := 1/5
  let r : ℚ := -1/3
  let n : ℕ := 6
  geometric_sum a r n = 182/1215 := by sorry

end geometric_series_sum_l3541_354148


namespace sum_of_digits_of_sum_of_digits_of_large_number_l3541_354171

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number has exactly ten billion digits -/
def has_ten_billion_digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_sum_of_digits_of_large_number 
  (A : ℕ) 
  (h1 : has_ten_billion_digits A) 
  (h2 : A % 9 = 0) : 
  let B := sum_of_digits A
  let C := sum_of_digits B
  sum_of_digits C = 9 := by sorry

end sum_of_digits_of_sum_of_digits_of_large_number_l3541_354171


namespace f_properties_l3541_354198

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a^x / (a^x + 1)

-- Main theorem
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- 1. The range of f(x) is (0, 1)
  (∀ x, 0 < f a x ∧ f a x < 1) ∧
  -- 2. If the maximum value of f(x) on [-1, 2] is 3/4, then a = √3 or a = 1/3
  (Set.Icc (-1) 2 ⊆ f a ⁻¹' Set.Iio (3/4) → a = Real.sqrt 3 ∨ a = 1/3) :=
by sorry

end

end f_properties_l3541_354198


namespace acme_cheaper_at_min_shirts_l3541_354151

/-- Acme T-Shirt Company's pricing function -/
def acme_cost (x : ℕ) : ℝ := 60 + 8 * x

/-- Gamma T-Shirt Company's pricing function -/
def gamma_cost (x : ℕ) : ℝ := 12 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_for_acme : ℕ := 16

theorem acme_cheaper_at_min_shirts :
  acme_cost min_shirts_for_acme < gamma_cost min_shirts_for_acme ∧
  ∀ n : ℕ, n < min_shirts_for_acme → acme_cost n ≥ gamma_cost n :=
by sorry

end acme_cheaper_at_min_shirts_l3541_354151


namespace average_of_numbers_l3541_354173

def number1 : Nat := 8642097531
def number2 : Nat := 6420875319
def number3 : Nat := 4208653197
def number4 : Nat := 2086431975
def number5 : Nat := 864219753

def numbers : List Nat := [number1, number2, number3, number4, number5]

theorem average_of_numbers : 
  (numbers.sum / numbers.length : Rat) = 4444455555 := by sorry

end average_of_numbers_l3541_354173


namespace arithmetic_sequence_seventh_term_l3541_354124

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a2 : a 2 = 2)
  (h_sum : a 4 + a 5 = 12) :
  a 7 = 10 := by
  sorry

end arithmetic_sequence_seventh_term_l3541_354124


namespace twenty_mps_equals_72_kmph_l3541_354152

/-- Conversion from meters per second to kilometers per hour -/
def mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * 3.6

/-- Theorem: 20 mps is equal to 72 kmph -/
theorem twenty_mps_equals_72_kmph :
  mps_to_kmph 20 = 72 := by
  sorry

#eval mps_to_kmph 20

end twenty_mps_equals_72_kmph_l3541_354152


namespace circle_equation_and_tangent_and_symmetry_l3541_354120

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 5

-- Define the line that contains the center of C
def center_line (x y : ℝ) : Prop := 2*x - y - 7 = 0

-- Define the points A and B where C intersects the y-axis
def point_A : ℝ × ℝ := (0, -4)
def point_B : ℝ × ℝ := (0, -2)

-- Define the tangent line l
def line_l (k x y : ℝ) : Prop := k*x - y + k = 0

-- Define the line l₁ for symmetry
def line_l1 (x y : ℝ) : Prop := y = 2*x + 1

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x + 22/5)^2 + (y - 1/5)^2 = 5

theorem circle_equation_and_tangent_and_symmetry :
  ∃ (k : ℝ),
    (∀ x y, circle_C x y ↔ (x - 2)^2 + (y + 3)^2 = 5) ∧
    (k = (-9 + Real.sqrt 65) / 4 ∨ k = (-9 - Real.sqrt 65) / 4) ∧
    (∀ x y, symmetric_circle x y ↔ (x + 22/5)^2 + (y - 1/5)^2 = 5) :=
by sorry

end circle_equation_and_tangent_and_symmetry_l3541_354120


namespace total_area_smaller_than_4pi_R_squared_l3541_354158

variable (R x y z : ℝ)

/-- Three circles with radii x, y, and z touch each other externally -/
axiom circles_touch_externally : True

/-- The centers of the three circles lie on a fourth circle with radius R -/
axiom centers_on_fourth_circle : True

/-- The radius R of the fourth circle is related to x, y, and z by Heron's formula -/
axiom heron_formula : R = (x + y) * (y + z) * (z + x) / (4 * Real.sqrt ((x + y + z) * x * y * z))

/-- The total area of the three circle disks is smaller than 4πR² -/
theorem total_area_smaller_than_4pi_R_squared :
  x^2 + y^2 + z^2 < 4 * R^2 := by sorry

end total_area_smaller_than_4pi_R_squared_l3541_354158


namespace fraction_inequality_solution_set_l3541_354117

theorem fraction_inequality_solution_set :
  {x : ℝ | (x - 1) / (1 - 2*x) ≥ 0} = Set.Ioo (1/2) 1 ∪ {1} :=
by sorry

end fraction_inequality_solution_set_l3541_354117


namespace multiple_of_2007_cube_difference_l3541_354140

theorem multiple_of_2007_cube_difference (k : ℕ+) :
  (∃ a : ℤ, ∃ m : ℤ, (a + k.val : ℤ)^3 - a^3 = 2007 * m) ↔ ∃ n : ℕ, k.val = 669 * n :=
sorry

end multiple_of_2007_cube_difference_l3541_354140


namespace marble_game_winner_l3541_354169

/-- Represents the distribution of marbles in a single game -/
structure GameDistribution where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the total marbles each player has after all games -/
structure FinalMarbles where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The main theorem statement -/
theorem marble_game_winner
  (p q r : ℕ)
  (h_p_lt_q : p < q)
  (h_q_lt_r : q < r)
  (h_p_pos : 0 < p)
  (h_sum : p + q + r = 13)
  (final_marbles : FinalMarbles)
  (h_final_a : final_marbles.a = 20)
  (h_final_b : final_marbles.b = 10)
  (h_final_c : final_marbles.c = 9)
  (h_b_last : ∃ (g1 g2 : GameDistribution), g1.b + g2.b + r = 10)
  : ∃ (g1 g2 : GameDistribution),
    g1.c = q ∧ g2.c ≠ q ∧
    g1.a + g2.a + final_marbles.a - 20 = p + q + r ∧
    g1.b + g2.b + final_marbles.b - 10 = p + q + r ∧
    g1.c + g2.c + final_marbles.c - 9 = p + q + r :=
sorry

end marble_game_winner_l3541_354169


namespace find_Y_l3541_354159

theorem find_Y : ∃ Y : ℚ, (19 + Y / 151) * 151 = 2912 → Y = 43 := by
  sorry

end find_Y_l3541_354159


namespace company_picnic_attendance_l3541_354127

theorem company_picnic_attendance 
  (total_employees : ℝ) 
  (total_men : ℝ) 
  (total_women : ℝ) 
  (women_picnic_attendance : ℝ) 
  (total_picnic_attendance : ℝ) 
  (h1 : women_picnic_attendance = 0.4 * total_women)
  (h2 : total_men = 0.3 * total_employees)
  (h3 : total_women = total_employees - total_men)
  (h4 : total_picnic_attendance = 0.34 * total_employees)
  : (total_picnic_attendance - women_picnic_attendance) / total_men = 0.2 := by
  sorry

end company_picnic_attendance_l3541_354127


namespace range_of_a_l3541_354106

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, -1 ≤ x₀ ∧ x₀ ≤ 1 ∧ 2 * a * x₀^2 + 2 * x₀ - 3 - a = 0) → 
  (a ≥ 1 ∨ a ≤ (-3 - Real.sqrt 7) / 2) := by
  sorry

end range_of_a_l3541_354106


namespace sixth_term_of_geometric_sequence_l3541_354192

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem sixth_term_of_geometric_sequence (a₁ a₄ : ℝ) (h₁ : a₁ = 8) (h₂ : a₄ = 64) :
  ∃ r : ℝ, geometric_sequence a₁ r 4 = a₄ ∧ geometric_sequence a₁ r 6 = 256 := by
sorry

end sixth_term_of_geometric_sequence_l3541_354192


namespace joan_socks_total_l3541_354126

theorem joan_socks_total (total_socks : ℕ) (white_socks : ℕ) (blue_socks : ℕ) :
  white_socks = (2 : ℚ) / 3 * total_socks →
  blue_socks = total_socks - white_socks →
  blue_socks = 60 →
  total_socks = 180 := by
  sorry

end joan_socks_total_l3541_354126


namespace multiplier_problem_l3541_354170

theorem multiplier_problem (a b : ℝ) (h1 : 4 * a = b) (h2 : b = 30) (h3 : 40 * a * b = 1800) :
  ∃ m : ℝ, m * b = 30 ∧ m = 5 := by
sorry

end multiplier_problem_l3541_354170


namespace geometric_series_sum_proof_l3541_354180

/-- The sum of the infinite geometric series 5/3 - 5/6 + 5/18 - 5/54 + ... -/
def geometric_series_sum : ℚ := 10/9

/-- The first term of the geometric series -/
def a : ℚ := 5/3

/-- The common ratio of the geometric series -/
def r : ℚ := -1/2

theorem geometric_series_sum_proof :
  geometric_series_sum = a / (1 - r) :=
by sorry

end geometric_series_sum_proof_l3541_354180


namespace triangle_side_difference_l3541_354157

theorem triangle_side_difference (x : ℕ) : 
  (x + 8 > 10) ∧ (x + 10 > 8) ∧ (8 + 10 > x) →
  (∃ (max min : ℕ), 
    (∀ y : ℕ, (y + 8 > 10) ∧ (y + 10 > 8) ∧ (8 + 10 > y) → y ≤ max ∧ y ≥ min) ∧
    (max - min = 14)) :=
by sorry

end triangle_side_difference_l3541_354157


namespace binary_1101011_equals_107_l3541_354172

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101011_equals_107 :
  binary_to_decimal [true, true, false, true, false, true, true] = 107 := by
  sorry

end binary_1101011_equals_107_l3541_354172


namespace investment_proof_l3541_354132

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof of the investment problem -/
theorem investment_proof (principal : ℝ) (rate : ℝ) (time : ℕ) 
  (h1 : principal = 6000)
  (h2 : rate = 0.1)
  (h3 : time = 2) :
  compound_interest principal rate time = 7260 := by
  sorry

end investment_proof_l3541_354132


namespace chord_equation_l3541_354122

/-- The equation of a line that is a chord of the ellipse x^2 + 4y^2 = 36 and is bisected at (4, 2) -/
theorem chord_equation (x y : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    -- Points (x₁, y₁) and (x₂, y₂) lie on the ellipse
    x₁^2 + 4*y₁^2 = 36 ∧ x₂^2 + 4*y₂^2 = 36 ∧
    -- (4, 2) is the midpoint of the chord
    (x₁ + x₂)/2 = 4 ∧ (y₁ + y₂)/2 = 2 ∧
    -- (x, y) is on the line containing the chord
    ∃ (t : ℝ), x = x₁ + t*(x₂ - x₁) ∧ y = y₁ + t*(y₂ - y₁)) →
  x + 2*y - 8 = 0 := by
sorry

end chord_equation_l3541_354122


namespace triangle_inequality_l3541_354181

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b + b * c + c * a ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) := by
  sorry

end triangle_inequality_l3541_354181


namespace digit_sum_properties_l3541_354125

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem digit_sum_properties :
  (∀ N : ℕ, S N ≤ 8 * S (8 * N)) ∧
  (∀ r q : ℕ, ∃ c_k : ℚ, c_k > 0 ∧
    (∀ N : ℕ, S (2^r * 5^q * N) / S N ≥ c_k) ∧
    c_k = 1 / S (2^q * 5^r) ∧
    (∀ c : ℚ, c > c_k → ∃ N : ℕ, S (2^r * 5^q * N) / S N < c)) ∧
  (∀ k : ℕ, (∃ r q : ℕ, k = 2^r * 5^q) ∨
    (∀ c : ℚ, c > 0 → ∃ N : ℕ, S (k * N) / S N < c)) :=
sorry

end digit_sum_properties_l3541_354125


namespace streaming_service_fee_l3541_354128

/-- Given a fixed monthly fee and a charge per hour for extra content,
    if the total for one month is $18.60 and the total for another month
    with triple the extra content usage is $32.40,
    then the fixed monthly fee is $11.70. -/
theorem streaming_service_fee (x y : ℝ)
  (feb_bill : x + y = 18.60)
  (mar_bill : x + 3*y = 32.40) :
  x = 11.70 := by
  sorry

end streaming_service_fee_l3541_354128


namespace price_change_theorem_l3541_354142

theorem price_change_theorem (initial_price : ℝ) (price_increase : ℝ) 
  (discount1 : ℝ) (discount2 : ℝ) :
  price_increase = 32 ∧ discount1 = 10 ∧ discount2 = 15 →
  let increased_price := initial_price * (1 + price_increase / 100)
  let after_discount1 := increased_price * (1 - discount1 / 100)
  let final_price := after_discount1 * (1 - discount2 / 100)
  (final_price - initial_price) / initial_price * 100 = 0.98 := by
sorry

end price_change_theorem_l3541_354142


namespace carries_tshirt_purchase_l3541_354184

/-- The cost of a single t-shirt in dollars -/
def tshirt_cost : ℝ := 9.15

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 22

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℝ := tshirt_cost * num_tshirts

/-- Theorem stating that the total cost of Carrie's t-shirt purchase is $201.30 -/
theorem carries_tshirt_purchase : total_cost = 201.30 := by
  sorry

end carries_tshirt_purchase_l3541_354184


namespace least_five_digit_congruent_to_6_mod_17_l3541_354185

theorem least_five_digit_congruent_to_6_mod_17 :
  (∀ n : ℕ, 10000 ≤ n ∧ n < 10017 → ¬(n % 17 = 6)) ∧
  10017 % 17 = 6 := by
  sorry

end least_five_digit_congruent_to_6_mod_17_l3541_354185


namespace sum_angles_less_than_1100_l3541_354186

/-- Represents the angle measurement scenario with a car and a fence -/
structure AngleMeasurement where
  carSpeed : ℝ  -- Car speed in km/h
  fenceLength : ℝ  -- Fence length in meters
  measurementInterval : ℝ  -- Measurement interval in seconds

/-- Calculates the sum of angles measured -/
def sumOfAngles (scenario : AngleMeasurement) : ℝ :=
  sorry  -- Proof omitted

/-- Theorem stating that the sum of angles is less than 1100 degrees -/
theorem sum_angles_less_than_1100 (scenario : AngleMeasurement) 
  (h1 : scenario.carSpeed = 60)
  (h2 : scenario.fenceLength = 100)
  (h3 : scenario.measurementInterval = 1) :
  sumOfAngles scenario < 1100 := by
  sorry  -- Proof omitted

end sum_angles_less_than_1100_l3541_354186


namespace sin_angle_equality_l3541_354118

theorem sin_angle_equality (α : Real) (h : Real.sin (π + α) = -1/2) : 
  Real.sin (4*π - α) = -1/2 := by
  sorry

end sin_angle_equality_l3541_354118


namespace ivan_nails_purchase_l3541_354187

-- Define the cost of nails per 100 grams in each store
def cost_store1 : ℝ := 180
def cost_store2 : ℝ := 120

-- Define the amount Ivan was short in the first store
def short_amount : ℝ := 1430

-- Define the change Ivan received in the second store
def change_amount : ℝ := 490

-- Define the function to calculate the cost of nails in kilograms
def cost_per_kg (cost_per_100g : ℝ) : ℝ := cost_per_100g * 10

-- Define the amount of nails Ivan bought in kilograms
def nails_bought : ℝ := 3.2

-- Theorem statement
theorem ivan_nails_purchase :
  (cost_per_kg cost_store1 * nails_bought - (cost_per_kg cost_store2 * nails_bought + change_amount) = short_amount) ∧
  (nails_bought = 3.2) :=
by sorry

end ivan_nails_purchase_l3541_354187


namespace circle_k_range_l3541_354165

/-- Represents the equation of a potential circle -/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + 5*k = 0

/-- Checks if the equation represents a valid circle -/
def is_circle (k : ℝ) : Prop :=
  ∃ (x₀ y₀ r : ℝ), r > 0 ∧ ∀ (x y : ℝ),
    circle_equation x y k ↔ (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The theorem stating the range of k for which the equation represents a circle -/
theorem circle_k_range :
  ∀ k : ℝ, is_circle k ↔ k < 1 :=
sorry

end circle_k_range_l3541_354165


namespace apple_cost_theorem_l3541_354129

/-- The cost of apples given a rate per half dozen -/
def appleCost (halfDozenRate : ℚ) (dozens : ℚ) : ℚ :=
  dozens * (2 * halfDozenRate)

theorem apple_cost_theorem (halfDozenRate : ℚ) :
  halfDozenRate = (4.80 : ℚ) →
  appleCost halfDozenRate 4 = (38.40 : ℚ) :=
by
  sorry

#eval appleCost (4.80 : ℚ) 4

end apple_cost_theorem_l3541_354129


namespace parking_lot_tires_l3541_354183

/-- Calculates the total number of tires in a parking lot with various vehicles -/
def total_tires (cars motorcycles trucks bicycles unicycles strollers : ℕ) 
  (cars_extra_tire bicycles_flat : ℕ) (unicycles_extra : ℕ) : ℕ :=
  -- 4-wheel drive cars
  (cars * 5 + cars_extra_tire) + 
  -- Motorcycles
  (motorcycles * 4) + 
  -- 6-wheel trucks
  (trucks * 7) + 
  -- Bicycles
  (bicycles * 2 - bicycles_flat) + 
  -- Unicycles
  (unicycles + unicycles_extra) + 
  -- Baby strollers
  (strollers * 4)

/-- Theorem stating the total number of tires in the parking lot -/
theorem parking_lot_tires : 
  total_tires 30 20 10 5 3 2 4 3 1 = 323 := by
  sorry

end parking_lot_tires_l3541_354183


namespace max_digit_count_is_24_l3541_354135

def apartment_numbers : List Nat := 
  (List.range 46).map (· + 90) ++ (List.range 46).map (· + 190)

def digit_count (d : Nat) (n : Nat) : Nat :=
  if n = 0 then
    if d = 0 then 1 else 0
  else
    digit_count d (n / 10) + if n % 10 = d then 1 else 0

def count_digit (d : Nat) (numbers : List Nat) : Nat :=
  numbers.foldl (fun acc n => acc + digit_count d n) 0

theorem max_digit_count_is_24 :
  (List.range 10).foldl (fun acc d => max acc (count_digit d apartment_numbers)) 0 = 24 := by
  sorry

end max_digit_count_is_24_l3541_354135


namespace volleyball_tournament_l3541_354155

theorem volleyball_tournament (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end volleyball_tournament_l3541_354155


namespace rectangle_breadth_l3541_354196

theorem rectangle_breadth (square_area : ℝ) (rectangle_area : ℝ) : 
  square_area = 16 → 
  rectangle_area = 220 → 
  ∃ (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ),
    circle_radius = Real.sqrt square_area ∧
    rectangle_length = 5 * circle_radius ∧
    rectangle_area = rectangle_length * rectangle_breadth ∧
    rectangle_breadth = 11 := by
  sorry

end rectangle_breadth_l3541_354196


namespace jane_has_nine_cans_l3541_354194

/-- The number of sunflower seeds Jane has -/
def total_seeds : ℕ := 54

/-- The number of seeds Jane places in each can -/
def seeds_per_can : ℕ := 6

/-- The number of cans Jane has -/
def number_of_cans : ℕ := total_seeds / seeds_per_can

/-- Proof that Jane has 9 cans -/
theorem jane_has_nine_cans : number_of_cans = 9 := by
  sorry

end jane_has_nine_cans_l3541_354194


namespace sqrt_16_equals_4_l3541_354103

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end sqrt_16_equals_4_l3541_354103


namespace total_molecular_weight_l3541_354174

/-- Atomic weight in g/mol -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "Ca" => 40.08
  | "I"  => 126.90
  | "Na" => 22.99
  | "Cl" => 35.45
  | "K"  => 39.10
  | "S"  => 32.06
  | "O"  => 16.00
  | _    => 0  -- Default case

/-- Molecular weight of a compound in g/mol -/
def molecular_weight (compound : String) : ℝ :=
  match compound with
  | "CaI2" => atomic_weight "Ca" + 2 * atomic_weight "I"
  | "NaCl" => atomic_weight "Na" + atomic_weight "Cl"
  | "K2SO4" => 2 * atomic_weight "K" + atomic_weight "S" + 4 * atomic_weight "O"
  | _      => 0  -- Default case

/-- Total weight of a given number of moles of a compound in grams -/
def total_weight (compound : String) (moles : ℝ) : ℝ :=
  moles * molecular_weight compound

/-- Theorem: The total molecular weight of 10 moles of CaI2, 7 moles of NaCl, and 15 moles of K2SO4 is 5961.78 grams -/
theorem total_molecular_weight : 
  total_weight "CaI2" 10 + total_weight "NaCl" 7 + total_weight "K2SO4" 15 = 5961.78 := by
  sorry

end total_molecular_weight_l3541_354174


namespace q_range_l3541_354131

def q (x : ℝ) : ℝ := (3 * x^2 + 1)^2

theorem q_range :
  ∀ y : ℝ, y ∈ Set.range q ↔ y ≥ 1 := by sorry

end q_range_l3541_354131


namespace line_vector_to_slope_intercept_l3541_354178

/-- Given a line in vector form, prove it's equivalent to slope-intercept form -/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 4) + (5 : ℝ) * (y - 1) = 0 ↔ 
  y = -(2/5 : ℝ) * x + (13/5 : ℝ) := by
  sorry

end line_vector_to_slope_intercept_l3541_354178


namespace workforce_reduction_l3541_354105

theorem workforce_reduction (initial_employees : ℕ) : 
  (initial_employees : ℝ) * 0.85 * 0.75 = 182 → 
  initial_employees = 285 :=
by
  sorry

end workforce_reduction_l3541_354105
