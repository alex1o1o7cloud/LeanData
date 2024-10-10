import Mathlib

namespace inequality_equivalence_l1441_144116

theorem inequality_equivalence (x : ℝ) : 
  Real.sqrt ((1 / (2 - x) + 1) ^ 2) ≥ 2 ↔ 
  (x ≥ 1 ∧ x < 2) ∨ (x > 2 ∧ x ≤ 7/3) :=
by sorry

end inequality_equivalence_l1441_144116


namespace natural_solutions_3x_plus_4y_eq_12_l1441_144171

theorem natural_solutions_3x_plus_4y_eq_12 :
  {(x, y) : ℕ × ℕ | 3 * x + 4 * y = 12} = {(4, 0), (0, 3)} := by
  sorry

end natural_solutions_3x_plus_4y_eq_12_l1441_144171


namespace power_product_equality_l1441_144180

theorem power_product_equality : 2^4 * 3^2 * 5^2 * 7^2 = 176400 := by
  sorry

end power_product_equality_l1441_144180


namespace definite_integral_equality_l1441_144199

theorem definite_integral_equality : 
  let a : Real := 0
  let b : Real := Real.arcsin (Real.sqrt (7/8))
  let f (x : Real) := (6 * Real.sin x ^ 2) / (4 + 3 * Real.cos (2 * x))
  ∫ x in a..b, f x = (Real.sqrt 7 * Real.pi) / 4 - Real.arctan (Real.sqrt 7) := by
  sorry

end definite_integral_equality_l1441_144199


namespace sqrt_squared_2a_minus_1_l1441_144151

theorem sqrt_squared_2a_minus_1 (a : ℝ) (h : a ≥ (1/2 : ℝ)) :
  Real.sqrt ((2*a - 1)^2) = 2*a - 1 := by
  sorry

end sqrt_squared_2a_minus_1_l1441_144151


namespace room_freezer_temp_difference_l1441_144106

-- Define the temperatures
def freezer_temp : Int := -4
def room_temp : Int := 18

-- Define the temperature difference function
def temp_difference (room : Int) (freezer : Int) : Int :=
  room - freezer

-- Theorem to prove
theorem room_freezer_temp_difference :
  temp_difference room_temp freezer_temp = 22 := by
  sorry

end room_freezer_temp_difference_l1441_144106


namespace water_added_to_fill_tank_l1441_144104

/-- Proves that the amount of water added to fill a tank is 16 gallons, given the initial state and capacity. -/
theorem water_added_to_fill_tank (initial_fraction : ℚ) (full_capacity : ℕ) : 
  initial_fraction = 1/3 → full_capacity = 24 → (1 - initial_fraction) * full_capacity = 16 := by
  sorry

end water_added_to_fill_tank_l1441_144104


namespace satisfying_polynomial_form_l1441_144142

/-- A polynomial with real coefficients satisfying the given equality for all real a, b, c -/
def SatisfyingPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, p (a + b - 2*c) + p (b + c - 2*a) + p (c + a - 2*b) = 
               3 * (p (a - b) + p (b - c) + p (c - a))

/-- The theorem stating that any satisfying polynomial has the form a₂x² + a₁x -/
theorem satisfying_polynomial_form (p : ℝ → ℝ) (h : SatisfyingPolynomial p) :
  ∃ a₂ a₁ : ℝ, ∀ x : ℝ, p x = a₂ * x^2 + a₁ * x :=
sorry

end satisfying_polynomial_form_l1441_144142


namespace otimes_inequality_iff_interval_l1441_144175

/-- Custom binary operation ⊗ on real numbers -/
def otimes (a b : ℝ) : ℝ := a * b + 2 * a + b

/-- Theorem stating the equivalence between the inequality and the interval -/
theorem otimes_inequality_iff_interval (x : ℝ) :
  otimes x (x - 2) < 0 ↔ -2 < x ∧ x < 1 :=
sorry

end otimes_inequality_iff_interval_l1441_144175


namespace first_digit_change_largest_l1441_144160

def original : ℚ := 0.1234567

def change_digit (n : ℚ) (pos : ℕ) : ℚ :=
  n + (8 - (n * 10^pos % 10)) / 10^pos

theorem first_digit_change_largest :
  ∀ pos : ℕ, pos > 0 → change_digit original 0 ≥ change_digit original pos :=
by
  sorry

end first_digit_change_largest_l1441_144160


namespace min_value_polynomial_l1441_144138

theorem min_value_polynomial (x y : ℝ) : 
  ∀ a b : ℝ, 5 * a^2 - 4 * a * b + 4 * b^2 + 12 * a + 25 ≥ 16 :=
by sorry

end min_value_polynomial_l1441_144138


namespace adult_ticket_cost_l1441_144139

/-- Proves that the cost of an adult ticket is $9, given the conditions of the problem -/
theorem adult_ticket_cost (child_ticket_cost : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (children_tickets : ℕ) :
  child_ticket_cost = 6 →
  total_tickets = 225 →
  total_revenue = 1875 →
  children_tickets = 50 →
  (total_revenue - child_ticket_cost * children_tickets) / (total_tickets - children_tickets) = 9 := by
sorry

end adult_ticket_cost_l1441_144139


namespace probability_of_shaded_triangle_l1441_144188

/-- Given a set of triangles where some are shaded, this theorem proves
    the probability of selecting a shaded triangle when each triangle
    has an equal probability of being selected. -/
theorem probability_of_shaded_triangle 
  (total_triangles : ℕ) 
  (shaded_triangles : ℕ) 
  (h1 : total_triangles = 8) 
  (h2 : shaded_triangles = 4) 
  (h3 : shaded_triangles ≤ total_triangles) :
  (shaded_triangles : ℚ) / total_triangles = 1 / 2 := by
  sorry

end probability_of_shaded_triangle_l1441_144188


namespace min_value_expression_equality_condition_l1441_144135

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y)^2 + (y + 1/(2*x))^2 ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y)^2 + (y + 1/(2*x))^2 = 3 + 2 * Real.sqrt 2 ↔ 
  x = Real.sqrt 2 / 2 ∧ y = 1 :=
by sorry

end min_value_expression_equality_condition_l1441_144135


namespace solution_set_of_inequality_l1441_144197

-- Define an even function f: ℝ → ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define an increasing function on (-∞, 0]
def increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

-- Theorem statement
theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_inc : increasing_on_neg f) :
  {x : ℝ | f (x - 1) ≥ f 1} = Set.Icc 0 2 := by
  sorry

end solution_set_of_inequality_l1441_144197


namespace cubic_equation_solution_l1441_144127

theorem cubic_equation_solution (x : ℝ) : 
  x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ↔ x = 6 :=
by sorry

end cubic_equation_solution_l1441_144127


namespace speed_conversion_l1441_144178

/-- Proves that a speed of 0.8 km/h, when expressed as a fraction in m/s with numerator 8, has a denominator of 36 -/
theorem speed_conversion (speed_kmh : ℚ) (speed_ms_num : ℕ) : 
  speed_kmh = 0.8 → speed_ms_num = 8 → 
  ∃ (speed_ms_den : ℕ), 
    (speed_kmh * 1000 / 3600 = speed_ms_num / speed_ms_den) ∧ 
    speed_ms_den = 36 := by
  sorry

end speed_conversion_l1441_144178


namespace equation_solution_l1441_144189

theorem equation_solution : ∃ x : ℝ, (24 - 4 = 3 + x) ∧ (x = 17) := by
  sorry

end equation_solution_l1441_144189


namespace least_addition_for_divisibility_l1441_144111

theorem least_addition_for_divisibility :
  ∃ (n : ℕ), n = 8 ∧ 
  (∀ (m : ℕ), (1056 + m) % 28 = 0 → m ≥ n) ∧
  (1056 + n) % 28 = 0 :=
by sorry

end least_addition_for_divisibility_l1441_144111


namespace min_value_problem_l1441_144108

theorem min_value_problem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hmin : ∀ x : ℝ, |x + a| + |x - b| + c ≥ 4) :
  (a + b + c = 4) ∧ 
  (∀ a' b' c' : ℝ, a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 → 
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) :=
by sorry

end min_value_problem_l1441_144108


namespace common_point_exists_l1441_144124

-- Define the basic structures
structure Ray where
  origin : ℝ × ℝ
  direction : ℝ × ℝ

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the given conditions
def intersect_point : ℝ × ℝ := sorry

def ray1 : Ray := sorry
def ray2 : Ray := sorry

def a : ℝ := sorry
axiom a_positive : 0 < a

-- Define the circle properties
def circle_passes_through (c : Circle) (p : ℝ × ℝ) : Prop := sorry

def circle_intersects_ray (c : Circle) (r : Ray) : ℝ × ℝ := sorry

def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem common_point_exists :
  ∀ (c : Circle),
    circle_passes_through c intersect_point ∧
    ∃ (B C : ℝ × ℝ),
      B = circle_intersects_ray c ray1 ∧
      C = circle_intersects_ray c ray2 ∧
      distance intersect_point B + distance intersect_point C = a
    →
    ∃ (Z : ℝ × ℝ), Z ≠ intersect_point ∧
      ∀ (c' : Circle),
        circle_passes_through c' intersect_point ∧
        ∃ (B' C' : ℝ × ℝ),
          B' = circle_intersects_ray c' ray1 ∧
          C' = circle_intersects_ray c' ray2 ∧
          distance intersect_point B' + distance intersect_point C' = a
        →
        circle_passes_through c' Z :=
sorry

end common_point_exists_l1441_144124


namespace max_value_of_s_l1441_144192

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 12)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 24) :
  s ≤ 3 + 3 * Real.sqrt 5 := by
  sorry

end max_value_of_s_l1441_144192


namespace locus_of_M_constant_ratio_l1441_144119

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define a point on the ellipse
variable (P : ℝ × ℝ)
axiom P_on_ellipse : ellipse P.1 P.2

-- Define point M
def M (P : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define point N
def N (P : ℝ × ℝ) : ℝ × ℝ := sorry

theorem locus_of_M (P : ℝ × ℝ) (h : ellipse P.1 P.2) : 
  (M P).1 = -8 := by sorry

theorem constant_ratio (P : ℝ × ℝ) (h : ellipse P.1 P.2) :
  ‖N P - F₁‖ / ‖M P - F₁‖ = 1/2 := by sorry

end locus_of_M_constant_ratio_l1441_144119


namespace lesser_number_l1441_144107

theorem lesser_number (x y : ℝ) (sum_eq : x + y = 60) (diff_eq : x - y = 10) : 
  min x y = 25 := by sorry

end lesser_number_l1441_144107


namespace sum_47_58_base5_l1441_144132

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_47_58_base5 : toBase5 (47 + 58) = [4, 1, 0] := by
  sorry

end sum_47_58_base5_l1441_144132


namespace chord_length_line_circle_specific_chord_length_l1441_144125

/-- The length of the chord cut off by a line on a circle -/
theorem chord_length_line_circle (a b c d e f : ℝ) (h1 : a ≠ 0 ∨ b ≠ 0) :
  let line := {(x, y) : ℝ × ℝ | a * x + b * y = c}
  let circle := {(x, y) : ℝ × ℝ | (x - d)^2 + (y - e)^2 = f^2}
  let center := (d, e)
  let radius := f
  let dist_center_to_line := |a * d + b * e - c| / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (radius^2 - dist_center_to_line^2) = 6 * Real.sqrt 11 / 5 :=
by sorry

/-- The specific case for the given problem -/
theorem specific_chord_length :
  let line := {(x, y) : ℝ × ℝ | 3 * x + 4 * y = 7}
  let circle := {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 4}
  let center := (2, 0)
  let radius := 2
  let dist_center_to_line := |3 * 2 + 4 * 0 - 7| / Real.sqrt (3^2 + 4^2)
  2 * Real.sqrt (radius^2 - dist_center_to_line^2) = 6 * Real.sqrt 11 / 5 :=
by sorry

end chord_length_line_circle_specific_chord_length_l1441_144125


namespace max_sqrt_sum_l1441_144198

theorem max_sqrt_sum (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 17) :
  Real.sqrt (x + 34) + Real.sqrt (17 - x) + Real.sqrt (2 * x) ≤ Real.sqrt 51 + Real.sqrt 34 :=
by sorry

end max_sqrt_sum_l1441_144198


namespace expression_integer_iff_special_form_l1441_144179

def expression (n : ℤ) : ℝ :=
  (n + (n^2 + 1).sqrt)^(1/3) + (n - (n^2 + 1).sqrt)^(1/3)

theorem expression_integer_iff_special_form (n : ℤ) :
  ∃ (k : ℤ), k > 0 ∧ expression n = k ↔ ∃ (m : ℤ), m > 0 ∧ n = m * (m^2 + 3) / 2 :=
sorry

end expression_integer_iff_special_form_l1441_144179


namespace certain_negative_integer_l1441_144176

theorem certain_negative_integer (a b : ℤ) (x : ℤ) : 
  (-11 * a < 0) →
  (x < 0) →
  (x * b < 0) →
  ((-11 * a * x) * (x * b) + a * b = 89) →
  x = -1 :=
by sorry

end certain_negative_integer_l1441_144176


namespace parabola_vertex_l1441_144154

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 3 * (x - 1)^2 + 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = 3(x-1)^2 + 2 is (1, 2) -/
theorem parabola_vertex : 
  ∀ x : ℝ, parabola x ≥ parabola (vertex.1) ∧ parabola (vertex.1) = vertex.2 := by
  sorry

end parabola_vertex_l1441_144154


namespace quadratic_rewrite_sum_l1441_144143

/-- Given a quadratic expression 8k^2 - 12k + 20, when rewritten in the form d(k + r)^2 + s
    where d, r, and s are constants, prove that r + s = 14.75 -/
theorem quadratic_rewrite_sum (k : ℝ) : 
  ∃ (d r s : ℝ), (∀ k, 8 * k^2 - 12 * k + 20 = d * (k + r)^2 + s) ∧ r + s = 14.75 := by
sorry

end quadratic_rewrite_sum_l1441_144143


namespace ellipse_line_intersection_theorem_l1441_144190

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2 + 2*y^2 = 8

-- Define the line
def Line (k m x y : ℝ) : Prop := y = k*x + m

-- Define the intersection points
def Intersects (k m : ℝ) (P Q : ℝ × ℝ) : Prop :=
  Line k m P.1 P.2 ∧ Line k m Q.1 Q.2 ∧
  Ellipse P.1 P.2 ∧ Ellipse Q.1 Q.2

-- Define the x-axis and y-axis intersection points
def AxisIntersections (k m : ℝ) (C D : ℝ × ℝ) : Prop :=
  C = (-m/k, 0) ∧ D = (0, m)

-- Define the trisection condition
def Trisection (O P Q C D : ℝ × ℝ) : Prop :=
  (D.1 - O.1, D.2 - O.2) = (1/3 * (P.1 - O.1), 1/3 * (P.2 - O.2)) + (2/3 * (Q.1 - O.1), 2/3 * (Q.2 - O.2)) ∧
  (C.1 - O.1, C.2 - O.2) = (1/3 * (Q.1 - O.1), 1/3 * (Q.2 - O.2)) + (2/3 * (P.1 - O.1), 2/3 * (P.2 - O.2))

theorem ellipse_line_intersection_theorem :
  ∃ (k m : ℝ) (P Q C D : ℝ × ℝ),
    Intersects k m P Q ∧
    AxisIntersections k m C D ∧
    Trisection (0, 0) P Q C D :=
  sorry

end ellipse_line_intersection_theorem_l1441_144190


namespace remainder_theorem_l1441_144184

theorem remainder_theorem (d : ℕ) (r : ℕ) : d > 1 →
  (1059 % d = r ∧ 1417 % d = r ∧ 2312 % d = r) →
  d - r = 15 := by
sorry

end remainder_theorem_l1441_144184


namespace mcdonald_farm_eggs_l1441_144115

/-- Calculates the total number of eggs needed per month for a community -/
def total_eggs_per_month (saly_weekly : ℕ) (ben_weekly : ℕ) (weeks_per_month : ℕ) : ℕ :=
  let ked_weekly := ben_weekly / 2
  let total_weekly := saly_weekly + ben_weekly + ked_weekly
  total_weekly * weeks_per_month

/-- Proves that the total eggs needed per month is 124 given the specific requirements -/
theorem mcdonald_farm_eggs : total_eggs_per_month 10 14 4 = 124 := by
  sorry

end mcdonald_farm_eggs_l1441_144115


namespace cyclic_sum_inequality_l1441_144140

theorem cyclic_sum_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : a * b + b * c + c * a = a * b * c) : 
  (a^4 + b^4) / (a * b * (a^3 + b^3)) + 
  (b^4 + c^4) / (b * c * (b^3 + c^3)) + 
  (c^4 + a^4) / (c * a * (c^3 + a^3)) ≥ 1 := by
sorry

end cyclic_sum_inequality_l1441_144140


namespace factorial_ratio_42_40_l1441_144112

theorem factorial_ratio_42_40 : Nat.factorial 42 / Nat.factorial 40 = 1722 := by
  sorry

end factorial_ratio_42_40_l1441_144112


namespace equilateral_triangle_centroid_sum_l1441_144194

/-- An equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 1

/-- The centroid of a triangle -/
class Centroid (T : Type) where
  point : T → ℝ × ℝ

/-- The length of a segment from a vertex to the centroid -/
def vertex_to_centroid_length (t : EquilateralTriangle) : ℝ := sorry

theorem equilateral_triangle_centroid_sum 
  (t : EquilateralTriangle) 
  [Centroid EquilateralTriangle] : 
  3 * vertex_to_centroid_length t = Real.sqrt 3 := by sorry

end equilateral_triangle_centroid_sum_l1441_144194


namespace power_function_is_odd_l1441_144121

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (c : ℝ) (α : ℝ), ∀ x, f x = c * x^α

-- Define an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem power_function_is_odd (α : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (α - 2) * x^α
  isPowerFunction f → isOddFunction f := by
  sorry


end power_function_is_odd_l1441_144121


namespace afternoon_rowing_count_l1441_144193

/-- The number of campers who went rowing in the afternoon -/
def afternoon_rowing (morning_rowing hiking total : ℕ) : ℕ :=
  total - (morning_rowing + hiking)

/-- Theorem stating that 26 campers went rowing in the afternoon -/
theorem afternoon_rowing_count :
  afternoon_rowing 41 4 71 = 26 := by
  sorry

end afternoon_rowing_count_l1441_144193


namespace intersection_of_A_and_B_l1441_144105

-- Define the universal set I
def I : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x * (x - 1) ≥ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | x > 1} := by sorry

end intersection_of_A_and_B_l1441_144105


namespace erik_pie_amount_l1441_144118

theorem erik_pie_amount (frank_pie : ℝ) (erik_extra : ℝ) 
  (h1 : frank_pie = 0.3333333333333333)
  (h2 : erik_extra = 0.3333333333333333) :
  frank_pie + erik_extra = 0.6666666666666666 := by
  sorry

end erik_pie_amount_l1441_144118


namespace basketball_games_left_to_play_l1441_144181

theorem basketball_games_left_to_play 
  (games_played : ℕ) 
  (win_percentage : ℚ) 
  (additional_losses : ℕ) 
  (final_win_percentage : ℚ) :
  games_played = 40 →
  win_percentage = 70 / 100 →
  additional_losses = 8 →
  final_win_percentage = 60 / 100 →
  ∃ (games_left : ℕ), games_left = 7 ∧ 
    (games_played * win_percentage).floor + (games_played + games_left - (games_played * win_percentage).floor - additional_losses) = 
    (final_win_percentage * (games_played + games_left)).floor :=
by sorry

end basketball_games_left_to_play_l1441_144181


namespace quadratic_inequality_theorem_l1441_144102

-- Define the quadratic inequality
def quadratic_inequality (a b x : ℝ) : Prop :=
  2 * a * x^2 + 4 * x + b ≤ 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | x = -1/a}

-- State the theorem
theorem quadratic_inequality_theorem (a b : ℝ) 
  (h1 : ∀ x, x ∈ solution_set a ↔ quadratic_inequality a b x)
  (h2 : a > b) :
  (ab = 2) ∧ 
  (∀ a b, (2*a + b^3) / (2 - b^2) ≥ 4) ∧
  (∃ a b, (2*a + b^3) / (2 - b^2) = 4) :=
sorry

end quadratic_inequality_theorem_l1441_144102


namespace grandpas_initial_tomatoes_l1441_144114

-- Define the number of tomatoes that grew during vacation
def tomatoes_grown : ℕ := 3564

-- Define the multiplication factor for tomato growth
def growth_factor : ℕ := 100

-- Define the function to calculate the initial number of tomatoes
def initial_tomatoes : ℕ := (tomatoes_grown + growth_factor - 1) / growth_factor

-- Theorem statement
theorem grandpas_initial_tomatoes :
  initial_tomatoes = 36 :=
sorry

end grandpas_initial_tomatoes_l1441_144114


namespace corn_purchase_proof_l1441_144162

/-- The cost of corn in cents per pound -/
def corn_cost : ℚ := 99

/-- The cost of beans in cents per pound -/
def bean_cost : ℚ := 45

/-- The total weight of corn and beans in pounds -/
def total_weight : ℚ := 24

/-- The total cost in cents -/
def total_cost : ℚ := 1809

/-- The number of pounds of corn bought -/
def corn_pounds : ℚ := 13.5

theorem corn_purchase_proof :
  ∃ (bean_pounds : ℚ),
    bean_pounds + corn_pounds = total_weight ∧
    bean_cost * bean_pounds + corn_cost * corn_pounds = total_cost :=
by sorry

end corn_purchase_proof_l1441_144162


namespace smallest_six_consecutive_number_max_six_consecutive_with_perfect_square_F_l1441_144117

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Checks if a FourDigitNumber has distinct non-zero digits -/
def isValidFourDigitNumber (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  0 < a ∧ a ≤ 9 ∧
  0 < b ∧ b ≤ 9 ∧
  0 < c ∧ c ≤ 9 ∧
  0 < d ∧ d ≤ 9 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Checks if a FourDigitNumber is a "six-consecutive number" -/
def isSixConsecutive (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  (a + b) * (c + d) = 60

/-- Calculates F(M) for a FourDigitNumber -/
def F (n : FourDigitNumber) : Int :=
  let (a, b, c, d) := n
  (a * 10 + d) - (b * 10 + c) - ((a * 10 + c) - (b * 10 + d))

/-- Converts a FourDigitNumber to its integer representation -/
def toInt (n : FourDigitNumber) : Nat :=
  let (a, b, c, d) := n
  a * 1000 + b * 100 + c * 10 + d

theorem smallest_six_consecutive_number :
  ∃ (M : FourDigitNumber),
    isValidFourDigitNumber M ∧
    isSixConsecutive M ∧
    (∀ (N : FourDigitNumber),
      isValidFourDigitNumber N → isSixConsecutive N →
      toInt M ≤ toInt N) ∧
    toInt M = 1369 := by sorry

theorem max_six_consecutive_with_perfect_square_F :
  ∃ (N : FourDigitNumber),
    isValidFourDigitNumber N ∧
    isSixConsecutive N ∧
    (∃ (k : Nat), F N = k * k) ∧
    (∀ (M : FourDigitNumber),
      isValidFourDigitNumber M → isSixConsecutive M →
      (∃ (j : Nat), F M = j * j) →
      toInt M ≤ toInt N) ∧
    toInt N = 9613 := by sorry

end smallest_six_consecutive_number_max_six_consecutive_with_perfect_square_F_l1441_144117


namespace total_passengers_l1441_144149

theorem total_passengers (on_time : ℕ) (late : ℕ) (h1 : on_time = 14507) (h2 : late = 213) :
  on_time + late = 14720 := by
  sorry

end total_passengers_l1441_144149


namespace AC_greater_than_CK_l1441_144195

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = BC ∧ AC = 2 * Real.sqrt 7 ∧ AB = 8

-- Define point D as the foot of the height from B
def HeightFoot (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - D.1) * (A.1 - C.1) + (B.2 - D.2) * (A.2 - C.2) = 0 ∧
  D.1 = (A.1 + C.1) / 2 ∧ D.2 = (A.2 + C.2) / 2

-- Define point K on BD
def PointK (B D K : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t = 2/5 ∧ K = (t * B.1 + (1 - t) * D.1, t * B.2 + (1 - t) * D.2)

-- Main theorem
theorem AC_greater_than_CK (A B C D K : ℝ × ℝ) :
  Triangle A B C → HeightFoot A B C D → PointK B D K →
  Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) > Real.sqrt ((C.1 - K.1)^2 + (C.2 - K.2)^2) := by
  sorry

end AC_greater_than_CK_l1441_144195


namespace poles_for_given_plot_l1441_144133

/-- Calculates the number of poles needed for a side of a plot -/
def polesForSide (length : ℕ) (spacing : ℕ) : ℕ :=
  (length / spacing) + 1

/-- Represents a trapezoidal plot with given side lengths and pole spacings -/
structure TrapezoidalPlot where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  side4 : ℕ
  spacing1 : ℕ
  spacing2 : ℕ

/-- Calculates the total number of poles needed for a trapezoidal plot -/
def totalPoles (plot : TrapezoidalPlot) : ℕ :=
  polesForSide plot.side1 plot.spacing1 +
  polesForSide plot.side2 plot.spacing2 +
  polesForSide plot.side3 plot.spacing1 +
  polesForSide plot.side4 plot.spacing2

/-- The main theorem stating that the number of poles for the given plot is 40 -/
theorem poles_for_given_plot :
  let plot := TrapezoidalPlot.mk 60 30 50 40 5 4
  totalPoles plot = 40 := by
  sorry


end poles_for_given_plot_l1441_144133


namespace largest_n_with_conditions_l1441_144109

theorem largest_n_with_conditions : 
  ∃ (n : ℕ), n = 25 ∧ 
  (∃ (k : ℕ), n^2 = (k+1)^4 - k^4) ∧ 
  (∃ (b : ℕ), 3*n + 100 = b^2) ∧
  (∀ (m : ℕ), m > n → 
    (∀ (j : ℕ), m^2 ≠ (j+1)^4 - j^4) ∨ 
    (∀ (c : ℕ), 3*m + 100 ≠ c^2)) :=
by sorry

end largest_n_with_conditions_l1441_144109


namespace intersection_q_complement_p_l1441_144126

-- Define the sets
def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x^2 ≥ 9}
def Q : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_q_complement_p :
  Q ∩ (U \ P) = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end intersection_q_complement_p_l1441_144126


namespace ellipse_param_sum_l1441_144145

/-- An ellipse with given foci and distance sum -/
structure Ellipse where
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  distance_sum : ℝ

/-- Standard form parameters of an ellipse -/
structure EllipseParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Calculate the parameters of the ellipse given its foci and distance sum -/
def calculate_ellipse_params (e : Ellipse) : EllipseParams :=
  sorry

/-- The main theorem to be proved -/
theorem ellipse_param_sum (e : Ellipse) (p : EllipseParams) :
  e.f1 = (0, 1) →
  e.f2 = (6, 1) →
  e.distance_sum = 10 →
  p = calculate_ellipse_params e →
  p.h + p.k + p.a + p.b = 13 :=
sorry

end ellipse_param_sum_l1441_144145


namespace inequality_one_inequality_two_l1441_144155

-- Inequality 1: (x+1)^2 + 3(x+1) - 4 > 0
theorem inequality_one (x : ℝ) : 
  (x + 1)^2 + 3*(x + 1) - 4 > 0 ↔ x < -5 ∨ x > 0 := by sorry

-- Inequality 2: x^4 - 2x^2 + 1 > x^2 - 1
theorem inequality_two (x : ℝ) : 
  x^4 - 2*x^2 + 1 > x^2 - 1 ↔ x < -Real.sqrt 2 ∨ (-1 < x ∧ x < 1) ∨ x > Real.sqrt 2 := by sorry

end inequality_one_inequality_two_l1441_144155


namespace simplify_expression_l1441_144165

theorem simplify_expression (x : ℝ) : 7*x + 8 - 3*x + 14 = 4*x + 22 := by
  sorry

end simplify_expression_l1441_144165


namespace complex_fraction_equality_l1441_144128

theorem complex_fraction_equality : (2 - I) / (2 + I) = 3/5 - 4/5 * I :=
by sorry

end complex_fraction_equality_l1441_144128


namespace athlete_arrangement_and_allocation_l1441_144147

/-- The number of male athletes -/
def num_male_athletes : ℕ := 4

/-- The number of female athletes -/
def num_female_athletes : ℕ := 3

/-- The total number of athletes -/
def total_athletes : ℕ := num_male_athletes + num_female_athletes

/-- The number of ways to arrange the athletes with all female athletes together -/
def arrangement_count : ℕ := (Nat.factorial (num_male_athletes + 1)) * (Nat.factorial num_female_athletes)

/-- The number of ways to allocate male athletes to two venues -/
def allocation_count : ℕ := Nat.choose num_male_athletes 1 + Nat.choose num_male_athletes 2

theorem athlete_arrangement_and_allocation :
  arrangement_count = 720 ∧ allocation_count = 10 := by sorry


end athlete_arrangement_and_allocation_l1441_144147


namespace gasoline_spending_increase_l1441_144166

theorem gasoline_spending_increase (P Q : ℝ) (P_new Q_new : ℝ) : 
  P_new = 1.20 * P →
  Q_new = 0.90 * Q →
  P_new * Q_new = 1.08 * (P * Q) :=
sorry

end gasoline_spending_increase_l1441_144166


namespace smallest_term_divisible_by_billion_l1441_144130

def geometric_sequence (a₁ : ℚ) (a₂ : ℚ) (n : ℕ) : ℚ :=
  a₁ * (a₂ / a₁) ^ (n - 1)

def is_divisible_by_billion (q : ℚ) : Prop :=
  ∃ k : ℤ, q = k * 10^9

theorem smallest_term_divisible_by_billion :
  let a₁ := 5 / 8
  let a₂ := 50
  (∀ n < 9, ¬ is_divisible_by_billion (geometric_sequence a₁ a₂ n)) ∧
  is_divisible_by_billion (geometric_sequence a₁ a₂ 9) :=
by sorry

end smallest_term_divisible_by_billion_l1441_144130


namespace inverse_as_linear_combination_l1441_144131

def N : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 0, 4]

theorem inverse_as_linear_combination :
  ∃ (c d : ℚ), c = -1/12 ∧ d = 7/12 ∧ N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end inverse_as_linear_combination_l1441_144131


namespace prob_red_then_black_value_l1441_144144

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (hTotal : total_cards = 52)
  (hRed : red_cards = 26)
  (hBlack : black_cards = 26)
  (hSum : red_cards + black_cards = total_cards)

/-- The probability of drawing a red card first and a black card second -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) / d.total_cards * (d.black_cards : ℚ) / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a red card first and a black card second -/
theorem prob_red_then_black_value (d : Deck) : prob_red_then_black d = 13 / 51 := by
  sorry

end prob_red_then_black_value_l1441_144144


namespace scores_mode_is_9_l1441_144164

def scores : List Nat := [7, 10, 9, 8, 7, 9, 9, 8]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem scores_mode_is_9 : mode scores = 9 := by sorry

end scores_mode_is_9_l1441_144164


namespace alien_species_count_l1441_144173

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def alienSpeciesBase7 : List Nat := [5, 1, 2]

/-- The theorem stating that the base 7 number 215₇ is equal to 110 in base 10 --/
theorem alien_species_count : base7ToBase10 alienSpeciesBase7 = 110 := by
  sorry

end alien_species_count_l1441_144173


namespace smallest_sum_reciprocals_l1441_144196

theorem smallest_sum_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10 → 
  ∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 → 
  (x : ℕ) + y ≤ (a : ℕ) + b :=
by sorry

end smallest_sum_reciprocals_l1441_144196


namespace equation_solution_l1441_144163

theorem equation_solution (x : ℝ) : 
  (4 * x - 3 > 0) → 
  (Real.sqrt (4 * x - 3) + 12 / Real.sqrt (4 * x - 3) = 8) ↔ 
  (x = 7/4 ∨ x = 39/4) := by
sorry

end equation_solution_l1441_144163


namespace restaurant_group_cost_l1441_144136

/-- Calculates the total cost for a group to eat at a restaurant where kids eat free. -/
def group_meal_cost (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

/-- Proves that the total cost for a group of 11 people with 2 kids is $72,
    given that adult meals cost $8 and kids eat free. -/
theorem restaurant_group_cost :
  group_meal_cost 11 2 8 = 72 := by
  sorry

end restaurant_group_cost_l1441_144136


namespace test_score_calculation_l1441_144122

theorem test_score_calculation (total_marks : ℕ) (percentage : ℚ) : 
  total_marks = 50 → percentage = 80 / 100 → (percentage * total_marks : ℚ) = 40 := by
  sorry

end test_score_calculation_l1441_144122


namespace max_parts_5x5_grid_l1441_144174

/-- Represents a partition of a grid into parts with different areas -/
def GridPartition (n : ℕ) := List ℕ

/-- The sum of areas in a partition should equal the total grid area -/
def validPartition (g : ℕ) (p : GridPartition g) : Prop :=
  p.sum = g * g ∧ p.Nodup

/-- The maximum number of parts in a valid partition of a 5x5 grid -/
theorem max_parts_5x5_grid :
  (∃ (p : GridPartition 5), validPartition 5 p ∧ p.length = 6) ∧
  (∀ (p : GridPartition 5), validPartition 5 p → p.length ≤ 6) := by
  sorry

#check max_parts_5x5_grid

end max_parts_5x5_grid_l1441_144174


namespace stationery_box_sheets_l1441_144100

/-- Represents a box of stationery --/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Represents Alice's usage of stationery --/
def alice_usage (box : StationeryBox) : Prop :=
  box.sheets - 2 * box.envelopes = 80

/-- Represents Bob's usage of stationery --/
def bob_usage (box : StationeryBox) : Prop :=
  4 * box.envelopes = box.sheets ∧ box.envelopes ≥ 35

theorem stationery_box_sheets :
  ∃ (box : StationeryBox), alice_usage box ∧ bob_usage box ∧ box.sheets = 160 := by
  sorry

#check stationery_box_sheets

end stationery_box_sheets_l1441_144100


namespace root_product_equals_negative_183_l1441_144161

-- Define the polynomial h
def h (y : ℝ) : ℝ := y^5 - y^3 + 2*y + 3

-- Define the polynomial p
def p (y : ℝ) : ℝ := y^2 - 3

-- State the theorem
theorem root_product_equals_negative_183 
  (y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (h_roots : h y₁ = 0 ∧ h y₂ = 0 ∧ h y₃ = 0 ∧ h y₄ = 0 ∧ h y₅ = 0) :
  p y₁ * p y₂ * p y₃ * p y₄ * p y₅ = -183 :=
sorry

end root_product_equals_negative_183_l1441_144161


namespace circle_P_properties_l1441_144177

/-- Given a circle P with center (a, b) and radius R -/
theorem circle_P_properties (a b R : ℝ) :
  R^2 - b^2 = 2 →
  R^2 - a^2 = 3 →
  (∃ x y : ℝ, y^2 - x^2 = 1) ∧
  (|b - a| = 1 →
    ((∃ x y : ℝ, x^2 + (y - 1)^2 = 3) ∨
     (∃ x y : ℝ, x^2 + (y + 1)^2 = 3))) :=
by sorry

end circle_P_properties_l1441_144177


namespace monochromatic_rectangle_exists_l1441_144129

/-- A color represented as a natural number -/
def Color := ℕ

/-- A point in the grid -/
structure GridPoint where
  x : ℕ
  y : ℕ
  h_x : x ≤ 12
  h_y : y ≤ 12

/-- A coloring of the grid -/
def GridColoring := GridPoint → Color

/-- A rectangle in the grid -/
structure Rectangle where
  x1 : ℕ
  y1 : ℕ
  x2 : ℕ
  y2 : ℕ
  h_x1 : x1 ≤ 12
  h_y1 : y1 ≤ 12
  h_x2 : x2 ≤ 12
  h_y2 : y2 ≤ 12
  h_distinct : (x1 ≠ x2 ∧ y1 ≠ y2) ∨ (x1 ≠ x2 ∧ y1 = y2) ∨ (x1 = x2 ∧ y1 ≠ y2)

/-- The theorem stating that there exists a monochromatic rectangle -/
theorem monochromatic_rectangle_exists (coloring : GridColoring) :
  ∃ (r : Rectangle) (c : Color),
    coloring ⟨r.x1, r.y1, r.h_x1, r.h_y1⟩ = c ∧
    coloring ⟨r.x1, r.y2, r.h_x1, r.h_y2⟩ = c ∧
    coloring ⟨r.x2, r.y1, r.h_x2, r.h_y1⟩ = c ∧
    coloring ⟨r.x2, r.y2, r.h_x2, r.h_y2⟩ = c :=
  sorry

end monochromatic_rectangle_exists_l1441_144129


namespace circle_equation_from_diameter_l1441_144168

/-- Given two points A and B as the endpoints of a circle's diameter, 
    prove that the equation of the circle is (x - 1)² + (y - 2)² = 25 -/
theorem circle_equation_from_diameter (A B : ℝ × ℝ) : 
  A = (-3, -1) → B = (5, 5) → 
  ∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 25 ↔ 
    ((x - (-3))^2 + (y - (-1))^2 = ((5 - (-3))^2 + (5 - (-1))^2) / 4 ∧ 
     (x - 5)^2 + (y - 5)^2 = ((5 - (-3))^2 + (5 - (-1))^2) / 4) :=
by sorry


end circle_equation_from_diameter_l1441_144168


namespace complex_modulus_l1441_144153

theorem complex_modulus (i : ℂ) (z : ℂ) : 
  i^2 = -1 → z = (i + 1) / (1 - i)^2 → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_l1441_144153


namespace parallel_lines_corresponding_angles_not_always_supplementary_l1441_144182

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of corresponding angles
def corresponding_angles (a1 a2 : Angle) (l1 l2 : Line) : Prop := sorry

-- Define the concept of supplementary angles
def supplementary (a1 a2 : Angle) : Prop := sorry

-- The theorem to be proven
theorem parallel_lines_corresponding_angles_not_always_supplementary :
  ¬ ∀ (l1 l2 : Line) (a1 a2 : Angle), 
    parallel l1 l2 → corresponding_angles a1 a2 l1 l2 → supplementary a1 a2 := by
  sorry

end parallel_lines_corresponding_angles_not_always_supplementary_l1441_144182


namespace intersecting_lines_slope_product_l1441_144170

/-- Given two lines in the xy-plane that intersect at a 30° angle, 
    where the slope of one line is 3 times the slope of the other, 
    the product of their slopes is 1. -/
theorem intersecting_lines_slope_product (m₁ m₂ : ℝ) : 
  m₂ = 3 * m₁ → 
  (|((m₂ - m₁) / (1 + m₁ * m₂))|) = Real.tan (30 * π / 180) → 
  m₁ * m₂ = 1 := by
sorry

end intersecting_lines_slope_product_l1441_144170


namespace red_light_probability_l1441_144158

theorem red_light_probability (p : ℝ) (h1 : p = 1 / 3) :
  let probability_green := 1 - p
  let probability_red := p
  let probability_first_red_at_second := probability_green * probability_red
  probability_first_red_at_second = 2 / 9 := by
  sorry

end red_light_probability_l1441_144158


namespace complex_equation_solution_l1441_144157

theorem complex_equation_solution (z : ℂ) (h : (3 + z) * Complex.I = 1) : z = -3 - Complex.I := by
  sorry

end complex_equation_solution_l1441_144157


namespace arithmetic_computation_l1441_144134

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 10 * 2 / 5 = 26 := by
  sorry

end arithmetic_computation_l1441_144134


namespace intersection_points_on_circle_l1441_144156

/-- The parabolas y = (x - 2)^2 and x + 6 = (y + 1)^2 intersect at points that lie on a circle with radius squared 5/2 -/
theorem intersection_points_on_circle :
  ∃ (c : ℝ × ℝ) (r : ℝ),
    (∀ (p : ℝ × ℝ),
      (p.2 = (p.1 - 2)^2 ∧ p.1 + 6 = (p.2 + 1)^2) →
      (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    r^2 = 5/2 := by
  sorry

end intersection_points_on_circle_l1441_144156


namespace third_root_unity_sum_l1441_144110

theorem third_root_unity_sum (z : ℂ) (h1 : z^3 - 1 = 0) (h2 : z ≠ 1) :
  z^100 + z^101 + z^102 + z^103 + z^104 = 0 := by
  sorry

end third_root_unity_sum_l1441_144110


namespace quadratic_inequality_solutions_l1441_144183

theorem quadratic_inequality_solutions (a : ℝ) :
  (a = -1 → {x : ℝ | a * x^2 + 5 * x + 6 > 0} = {x : ℝ | -1 < x ∧ x < 6}) ∧
  ({x : ℝ | a * x^2 + 5 * x + 6 > 0} = {x : ℝ | x < -3 ∨ x > -2} → a = 1) :=
sorry

end quadratic_inequality_solutions_l1441_144183


namespace function_bounds_l1441_144137

theorem function_bounds (x : ℝ) : 
  0.95 ≤ (x^4 + x^2 + 5) / ((x^2 + 1)^2) ∧ (x^4 + x^2 + 5) / ((x^2 + 1)^2) ≤ 5 := by
  sorry

end function_bounds_l1441_144137


namespace range_of_a_l1441_144120

theorem range_of_a (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 / x + 1 / y = 1) :
  (∀ a : ℝ, x + y + a > 0) ↔ ∀ a : ℝ, a > -3 - 2 * Real.sqrt 2 :=
by sorry

end range_of_a_l1441_144120


namespace sine_of_sum_angle_l1441_144172

theorem sine_of_sum_angle (θ : Real) :
  (∃ (x y : Real), x = -3 ∧ y = 4 ∧ 
   x = Real.cos θ * Real.sqrt (x^2 + y^2) ∧ 
   y = Real.sin θ * Real.sqrt (x^2 + y^2)) →
  Real.sin (θ + π/4) = Real.sqrt 2 / 10 := by
sorry

end sine_of_sum_angle_l1441_144172


namespace complementary_angles_l1441_144159

theorem complementary_angles (C D : ℝ) : 
  C + D = 90 →  -- C and D are complementary
  C = 5 * D →   -- C is 5 times D
  C = 75 :=     -- C is 75°
by sorry

end complementary_angles_l1441_144159


namespace arithmetic_sequence_ratio_l1441_144185

theorem arithmetic_sequence_ratio (a : ℕ+ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n : ℕ+, a (n + 2) + a (n + 1) = 2 * a n) →
  (∀ n : ℕ+, a (n + 1) = a n * q) →
  q = -2 := by
  sorry

end arithmetic_sequence_ratio_l1441_144185


namespace min_sum_m_n_min_sum_is_three_l1441_144103

theorem min_sum_m_n (m n : ℕ+) (h : 32 * m = n ^ 5) : 
  ∀ (m' n' : ℕ+), 32 * m' = n' ^ 5 → m + n ≤ m' + n' :=
by
  sorry

theorem min_sum_is_three : 
  ∃ (m n : ℕ+), 32 * m = n ^ 5 ∧ m + n = 3 :=
by
  sorry

end min_sum_m_n_min_sum_is_three_l1441_144103


namespace smallest_gcd_qr_l1441_144141

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 1050) :
  210 = Nat.gcd q r ∧ ∀ x : ℕ, x < 210 → x ≠ Nat.gcd q r :=
by sorry

end smallest_gcd_qr_l1441_144141


namespace tan_alpha_value_l1441_144187

theorem tan_alpha_value (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the third quadrant
  (Real.tan (π/4 - α) = (2/3) * Real.tan (α + π)) → 
  Real.tan α = 1/2 :=
by sorry

end tan_alpha_value_l1441_144187


namespace set_intersection_conditions_l1441_144167

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = 2*x - 1 ∧ 0 < x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) < 0}

-- State the theorem
theorem set_intersection_conditions (a : ℝ) :
  (A ∩ B a = A ↔ a ∈ Set.Ioc (-2) (-1)) ∧
  (A ∩ B a ≠ ∅ ↔ a ∈ Set.Ioo (-4) 1) :=
sorry

end set_intersection_conditions_l1441_144167


namespace parabola_perpendicular_range_l1441_144150

/-- Given a parabola y = x^2 with a fixed point A(-1, 1) and two moving points P and Q on the parabola,
    if PA ⊥ PQ, then the x-coordinate of Q is in (-∞, -3] ∪ [1, +∞) -/
theorem parabola_perpendicular_range (a x : ℝ) :
  let P : ℝ × ℝ := (a, a^2)
  let Q : ℝ × ℝ := (x, x^2)
  let A : ℝ × ℝ := (-1, 1)
  (a + 1) * (x - a) + (a^2 - 1) * (x^2 - a^2) = 0 →
  x ≤ -3 ∨ x ≥ 1 :=
by sorry

end parabola_perpendicular_range_l1441_144150


namespace scooter_repair_cost_l1441_144186

/-- Calculates the repair cost of a scooter given the conditions of the problem -/
def repair_cost (cost : ℝ) : ℝ :=
  0.1 * cost

/-- Calculates the selling price of a scooter given the conditions of the problem -/
def selling_price (cost : ℝ) : ℝ :=
  1.2 * cost

/-- Theorem stating the repair cost under the given conditions -/
theorem scooter_repair_cost :
  ∀ (cost : ℝ),
  cost > 0 →
  selling_price cost - cost = 1100 →
  repair_cost cost = 550 := by
sorry

end scooter_repair_cost_l1441_144186


namespace pete_triple_age_of_son_l1441_144169

def pete_age : ℕ := 35
def son_age : ℕ := 9
def years_until_triple : ℕ := 4

theorem pete_triple_age_of_son :
  pete_age + years_until_triple = 3 * (son_age + years_until_triple) :=
by sorry

end pete_triple_age_of_son_l1441_144169


namespace a_greater_than_one_l1441_144148

-- Define an increasing function on the real numbers
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem a_greater_than_one
  (f : ℝ → ℝ)
  (h_increasing : IncreasingFunction f)
  (h_inequality : f (a + 1) < f (2 * a)) :
  a > 1 :=
sorry

end a_greater_than_one_l1441_144148


namespace tangent_slope_implies_a_l1441_144101

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

-- Define the derivative of the curve
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_slope_implies_a (a : ℝ) :
  curve_derivative a 1 = 2 → a = -1 := by
  sorry

end tangent_slope_implies_a_l1441_144101


namespace optimal_pricing_achieves_target_profit_l1441_144191

/-- Represents the pricing and sales model for a desk lamp in a shopping mall. -/
structure LampSalesModel where
  initial_purchase_price : ℝ
  initial_selling_price : ℝ
  initial_monthly_sales : ℝ
  price_sales_slope : ℝ
  target_monthly_profit : ℝ

/-- Calculates the monthly profit for a given selling price and number of lamps sold. -/
def monthly_profit (model : LampSalesModel) (selling_price : ℝ) (lamps_sold : ℝ) : ℝ :=
  (selling_price - model.initial_purchase_price) * lamps_sold

/-- Calculates the number of lamps sold based on the selling price. -/
def lamps_sold (model : LampSalesModel) (selling_price : ℝ) : ℝ :=
  model.initial_monthly_sales - model.price_sales_slope * (selling_price - model.initial_selling_price)

/-- Theorem stating that the optimal selling price and number of lamps achieve the target monthly profit. -/
theorem optimal_pricing_achieves_target_profit (model : LampSalesModel)
  (h_model : model = {
    initial_purchase_price := 30,
    initial_selling_price := 40,
    initial_monthly_sales := 600,
    price_sales_slope := 10,
    target_monthly_profit := 10000
  })
  (optimal_price : ℝ)
  (optimal_lamps : ℝ)
  (h_price : optimal_price = 50)
  (h_lamps : optimal_lamps = 500) :
  monthly_profit model optimal_price optimal_lamps = model.target_monthly_profit :=
sorry


end optimal_pricing_achieves_target_profit_l1441_144191


namespace total_paths_is_fifteen_l1441_144152

/-- A graph representing paths between points A, B, C, and D. -/
structure PathGraph where
  paths_AB : Nat
  paths_BC : Nat
  paths_CD : Nat
  direct_AC : Nat

/-- Calculates the total number of paths from A to D in the given graph. -/
def total_paths (g : PathGraph) : Nat :=
  g.paths_AB * g.paths_BC * g.paths_CD + g.direct_AC * g.paths_CD

/-- Theorem stating that the total number of paths from A to D is 15. -/
theorem total_paths_is_fifteen (g : PathGraph) 
  (h1 : g.paths_AB = 2)
  (h2 : g.paths_BC = 2)
  (h3 : g.paths_CD = 3)
  (h4 : g.direct_AC = 1) : 
  total_paths g = 15 := by
  sorry

end total_paths_is_fifteen_l1441_144152


namespace parallelogram_vertex_sum_l1441_144146

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parallelogram in 2D space -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if two line segments are perpendicular -/
def isPerpendicular (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x) * (p4.x - p3.x) + (p2.y - p1.y) * (p4.y - p3.y) = 0

theorem parallelogram_vertex_sum (ABCD : Parallelogram) : 
  ABCD.A = Point.mk (-1) 2 →
  ABCD.B = Point.mk 3 (-1) →
  ABCD.D = Point.mk 5 7 →
  isPerpendicular ABCD.A ABCD.B ABCD.B ABCD.C →
  ABCD.C.x + ABCD.C.y = 9 := by
  sorry

end parallelogram_vertex_sum_l1441_144146


namespace hyperbola_equation_l1441_144123

/-- Given a hyperbola with eccentricity 2 and the same foci as the ellipse x²/25 + y²/9 = 1,
    prove that its equation is x²/4 - y²/12 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), x^2/25 + y^2/9 = 1 → 
      ∃ (c : ℝ), c = 4 ∧ 
        (∀ (x y : ℝ), (x + c)^2 + y^2 = 25 ∨ (x - c)^2 + y^2 = 25)) ∧
    (∃ (c : ℝ), c/a = 2 ∧ c = 4)) →
  x^2/4 - y^2/12 = 1 :=
sorry

end hyperbola_equation_l1441_144123


namespace triangle_perimeter_l1441_144113

/-- Given a triangle with sides of lengths 3, 6, and x, where x is a solution to x^2 - 7x + 12 = 0
    and satisfies the triangle inequality, prove that the perimeter of the triangle is 13. -/
theorem triangle_perimeter (x : ℝ) : 
  x^2 - 7*x + 12 = 0 →
  x + 3 > 6 →
  x + 6 > 3 →
  3 + 6 + x = 13 := by
  sorry

end triangle_perimeter_l1441_144113
