import Mathlib

namespace quadratic_roots_product_l3320_332011

theorem quadratic_roots_product (b c : ℝ) : 
  (1 : ℝ) ∈ {x : ℝ | x^2 + b*x + c = 0} ∧ 
  (-2 : ℝ) ∈ {x : ℝ | x^2 + b*x + c = 0} → 
  b * c = -2 := by
sorry

end quadratic_roots_product_l3320_332011


namespace function_root_implies_a_range_l3320_332024

theorem function_root_implies_a_range (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x - 1 = 0) → a ≥ -1 := by
  sorry

end function_root_implies_a_range_l3320_332024


namespace max_sum_squared_distances_l3320_332057

/-- The incircle of a triangle -/
def Incircle (A B O : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ θ : ℝ, P = (1 + Real.cos θ, 4/3 + Real.sin θ)}

/-- The squared distance between two points -/
def squaredDistance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem max_sum_squared_distances :
  let A : ℝ × ℝ := (3, 0)
  let B : ℝ × ℝ := (0, 4)
  let O : ℝ × ℝ := (0, 0)
  ∀ P ∈ Incircle A B O,
    squaredDistance P A + squaredDistance P B + squaredDistance P O ≤ 22 :=
by sorry

end max_sum_squared_distances_l3320_332057


namespace partner_b_share_l3320_332073

/-- Calculates the share of a partner in a partnership. -/
def calculate_share (total_profit : ℚ) (investment : ℚ) (total_investment : ℚ) : ℚ :=
  (investment / total_investment) * total_profit

theorem partner_b_share 
  (investment_a investment_b investment_c : ℚ)
  (share_a : ℚ)
  (h1 : investment_a = 7000)
  (h2 : investment_b = 11000)
  (h3 : investment_c = 18000)
  (h4 : share_a = 560) :
  calculate_share 
    ((share_a * (investment_a + investment_b + investment_c)) / investment_a)
    investment_b
    (investment_a + investment_b + investment_c) = 880 := by
  sorry

#eval calculate_share (560 * 36 / 7) 11000 36000

end partner_b_share_l3320_332073


namespace square_area_from_diagonal_l3320_332060

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s ^ 2 = 144 := by sorry

end square_area_from_diagonal_l3320_332060


namespace simplify_fraction_l3320_332031

theorem simplify_fraction : (120 : ℚ) / 180 = 2 / 3 := by
  sorry

end simplify_fraction_l3320_332031


namespace altitude_to_largerBase_ratio_l3320_332030

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- The length of the smaller base -/
  smallerBase : ℝ
  /-- The length of the larger base -/
  largerBase : ℝ
  /-- The length of the altitude -/
  altitude : ℝ
  /-- The smaller base is positive -/
  smallerBase_pos : 0 < smallerBase
  /-- The larger base is positive -/
  largerBase_pos : 0 < largerBase
  /-- The altitude is positive -/
  altitude_pos : 0 < altitude
  /-- The smaller base is less than the larger base -/
  smallerBase_lt_largerBase : smallerBase < largerBase
  /-- The smaller base equals the length of a diagonal -/
  smallerBase_eq_diagonal : smallerBase = Real.sqrt (smallerBase^2 + altitude^2)
  /-- The larger base equals twice the altitude -/
  largerBase_eq_twice_altitude : largerBase = 2 * altitude

/-- The ratio of the altitude to the larger base is 1/2 -/
theorem altitude_to_largerBase_ratio (t : IsoscelesTrapezoid) : 
  t.altitude / t.largerBase = 1 / 2 := by
  sorry

end altitude_to_largerBase_ratio_l3320_332030


namespace max_intersections_theorem_l3320_332062

/-- A convex polygon inscribed in a circle -/
structure InscribedPolygon where
  sides : ℕ
  sides_ge_3 : sides ≥ 3

/-- The maximum number of intersections between two inscribed polygons -/
def max_intersections (P₁ P₂ : InscribedPolygon) : ℕ := P₁.sides * P₂.sides

/-- Theorem: Maximum intersections between two inscribed polygons -/
theorem max_intersections_theorem (P₁ P₂ : InscribedPolygon) 
  (h : P₁.sides ≤ P₂.sides) : 
  max_intersections P₁ P₂ = P₁.sides * P₂.sides := by sorry

end max_intersections_theorem_l3320_332062


namespace circle_through_points_equation_l3320_332029

/-- A circle passing through three given points -/
structure CircleThroughPoints where
  -- Define the three points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Ensure the points are distinct
  distinct_points : A ≠ B ∧ B ≠ C ∧ A ≠ C

/-- The equation of a circle in standard form -/
def circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Main theorem: The circle through the given points has the specified equation -/
theorem circle_through_points_equation (c : CircleThroughPoints) :
  c.A = (-1, -1) →
  c.B = (-8, 0) →
  c.C = (0, 6) →
  ∃ (h k r : ℝ), 
    (h = -4 ∧ k = 3 ∧ r = 5) ∧
    (∀ x y, circle_equation h k r x y ↔ 
      ((x, y) = c.A ∨ (x, y) = c.B ∨ (x, y) = c.C)) :=
by sorry

#check circle_through_points_equation

end circle_through_points_equation_l3320_332029


namespace det_inequality_and_equality_l3320_332071

open Complex Matrix

variable {n : ℕ}

theorem det_inequality_and_equality (A : Matrix (Fin n) (Fin n) ℂ) (a : ℂ) 
  (h : A - conjTranspose A = (2 * a) • 1) : 
  (Complex.abs (det A) ≥ Complex.abs a ^ n) ∧ 
  (Complex.abs (det A) = Complex.abs a ^ n → A = a • 1) := by
  sorry

end det_inequality_and_equality_l3320_332071


namespace least_possible_smallest_integer_l3320_332053

theorem least_possible_smallest_integer
  (a b c d : ℤ) -- Four different integers
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) -- Integers are different
  (h_avg : (a + b + c + d) / 4 = 68) -- Average is 68
  (h_max : d = 90) -- Largest integer is 90
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d) -- Order of integers
  : a ≥ 5 := -- Least possible value of smallest integer is 5
by sorry

end least_possible_smallest_integer_l3320_332053


namespace tangent_lines_count_l3320_332082

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  abs (l.a * x₀ + l.b * y₀ + l.c) / Real.sqrt (l.a^2 + l.b^2) = c.radius

/-- Check if a line has equal intercepts on both axes -/
def has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

/-- The main theorem -/
theorem tangent_lines_count : 
  let c : Circle := ⟨(0, -5), 3⟩
  ∃ (lines : Finset Line), 
    lines.card = 4 ∧ 
    (∀ l ∈ lines, is_tangent l c ∧ has_equal_intercepts l) ∧
    (∀ l : Line, is_tangent l c ∧ has_equal_intercepts l → l ∈ lines) :=
sorry

end tangent_lines_count_l3320_332082


namespace chord_equation_l3320_332025

structure Curve where
  equation : ℝ → ℝ × ℝ

structure Line where
  equation : ℝ × ℝ → Prop

def parabola : Curve :=
  { equation := λ t => (4 * t^2, 4 * t) }

def point_on_curve (c : Curve) (p : ℝ × ℝ) : Prop :=
  ∃ t, c.equation t = p

def perpendicular (l1 l2 : Line) : Prop :=
  sorry

def chord_length_product (c : Curve) (l : Line) (p : ℝ × ℝ) : ℝ :=
  sorry

theorem chord_equation (c : Curve) (ab cd : Line) (p : ℝ × ℝ) :
  c = parabola →
  point_on_curve c p →
  p = (2, 2) →
  perpendicular ab cd →
  chord_length_product c ab p = chord_length_product c cd p →
  (ab.equation = λ (x, y) => y = x) ∨ 
  (ab.equation = λ (x, y) => x + y = 4) :=
sorry

end chord_equation_l3320_332025


namespace rock_collection_problem_l3320_332069

theorem rock_collection_problem (minerals_yesterday : ℕ) (gemstones : ℕ) (new_minerals : ℕ) :
  gemstones = minerals_yesterday / 2 →
  new_minerals = 6 →
  gemstones = 21 →
  minerals_yesterday + new_minerals = 48 :=
by sorry

end rock_collection_problem_l3320_332069


namespace largest_integer_less_than_150_over_7_l3320_332064

theorem largest_integer_less_than_150_over_7 : 
  (∀ n : ℤ, 7 * n < 150 → n ≤ 21) ∧ (7 * 21 < 150) := by sorry

end largest_integer_less_than_150_over_7_l3320_332064


namespace string_length_around_cylinder_specific_string_length_l3320_332065

/-- 
Given a cylindrical post with circumference C, height H, and a string making n complete loops 
around it from bottom to top, the length of the string L is given by L = n * √(C² + (H/n)²)
-/
theorem string_length_around_cylinder (C H : ℝ) (n : ℕ) (h1 : C > 0) (h2 : H > 0) (h3 : n > 0) :
  let L := n * Real.sqrt (C^2 + (H/n)^2)
  L = n * Real.sqrt (C^2 + (H/n)^2) := by sorry

/-- 
For the specific case where C = 6, H = 18, and n = 3, prove that the string length is 18√2
-/
theorem specific_string_length :
  let C : ℝ := 6
  let H : ℝ := 18
  let n : ℕ := 3
  let L := n * Real.sqrt (C^2 + (H/n)^2)
  L = 18 * Real.sqrt 2 := by sorry

end string_length_around_cylinder_specific_string_length_l3320_332065


namespace sequence_limit_l3320_332004

def a (n : ℕ) : ℚ := (7 * n + 4) / (2 * n + 1)

theorem sequence_limit : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 7/2| < ε := by
  sorry

end sequence_limit_l3320_332004


namespace fifteenth_student_age_l3320_332070

theorem fifteenth_student_age 
  (total_students : Nat) 
  (group1_students : Nat) 
  (group2_students : Nat) 
  (total_average_age : ℝ) 
  (group1_average_age : ℝ) 
  (group2_average_age : ℝ) 
  (h1 : total_students = 15)
  (h2 : group1_students = 8)
  (h3 : group2_students = 6)
  (h4 : total_average_age = 15)
  (h5 : group1_average_age = 14)
  (h6 : group2_average_age = 16)
  (h7 : group1_students + group2_students + 1 = total_students) :
  (total_students : ℝ) * total_average_age - 
  ((group1_students : ℝ) * group1_average_age + (group2_students : ℝ) * group2_average_age) = 17 := by
  sorry


end fifteenth_student_age_l3320_332070


namespace dave_tickets_l3320_332077

/-- The number of tickets Dave has at the end of the scenario -/
def final_tickets (initial_win : ℕ) (spent : ℕ) (later_win : ℕ) : ℕ :=
  initial_win - spent + later_win

/-- Theorem stating that Dave ends up with 16 tickets -/
theorem dave_tickets : final_tickets 11 5 10 = 16 := by
  sorry

end dave_tickets_l3320_332077


namespace min_operations_for_square_l3320_332043

-- Define the points
variable (A B C D : Point)

-- Define the operations
def measure_distance (P Q : Point) : ℝ := sorry
def compare_numbers (x y : ℝ) : Bool := sorry

-- Define what it means for ABCD to be a square
def is_square (A B C D : Point) : Prop :=
  let AB := measure_distance A B
  let BC := measure_distance B C
  let CD := measure_distance C D
  let DA := measure_distance D A
  let AC := measure_distance A C
  let BD := measure_distance B D
  (AB = BC) ∧ (BC = CD) ∧ (CD = DA) ∧ (AC = BD)

-- The theorem to prove
theorem min_operations_for_square (A B C D : Point) :
  ∃ (n : ℕ), n = 7 ∧ 
  (∀ (m : ℕ), m < n → ¬∃ (algorithm : Unit → Bool), 
    (algorithm () = true ↔ is_square A B C D)) :=
sorry

end min_operations_for_square_l3320_332043


namespace negation_of_quadratic_inequality_l3320_332002

theorem negation_of_quadratic_inequality :
  (∃ x : ℝ, x^2 - x + 3 ≤ 0) ↔ ¬(∀ x : ℝ, x^2 - x + 3 > 0) :=
by sorry

end negation_of_quadratic_inequality_l3320_332002


namespace square_sum_from_conditions_l3320_332095

theorem square_sum_from_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 36) 
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := by
sorry

end square_sum_from_conditions_l3320_332095


namespace two_bedroom_units_l3320_332055

theorem two_bedroom_units (total_units : ℕ) (one_bedroom_cost two_bedroom_cost : ℕ) (total_cost : ℕ) :
  total_units = 12 →
  one_bedroom_cost = 360 →
  two_bedroom_cost = 450 →
  total_cost = 4950 →
  ∃ (one_bedroom_count two_bedroom_count : ℕ),
    one_bedroom_count + two_bedroom_count = total_units ∧
    one_bedroom_count * one_bedroom_cost + two_bedroom_count * two_bedroom_cost = total_cost ∧
    two_bedroom_count = 7 :=
by sorry

end two_bedroom_units_l3320_332055


namespace tangent_lines_to_circle_l3320_332044

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space represented by the equation (x - h)² + (y - k)² = r² -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a line is tangent to a circle -/
def tangent (l : Line) (c : Circle) : Prop :=
  (c.h * l.a + c.k * l.b + l.c)^2 = (l.a^2 + l.b^2) * c.r^2

theorem tangent_lines_to_circle (given_line : Line) (c : Circle) :
  given_line = Line.mk 2 (-1) 1 ∧ c = Circle.mk 0 0 (Real.sqrt 5) →
  ∃ (l1 l2 : Line),
    l1 = Line.mk 2 (-1) 5 ∧
    l2 = Line.mk 2 (-1) (-5) ∧
    parallel l1 given_line ∧
    parallel l2 given_line ∧
    tangent l1 c ∧
    tangent l2 c ∧
    ∀ (l : Line), parallel l given_line ∧ tangent l c → l = l1 ∨ l = l2 :=
by sorry

end tangent_lines_to_circle_l3320_332044


namespace units_digit_of_5_pow_17_times_4_l3320_332017

theorem units_digit_of_5_pow_17_times_4 : ∃ n : ℕ, 5^17 * 4 = 10 * n :=
by sorry

end units_digit_of_5_pow_17_times_4_l3320_332017


namespace largest_divisor_of_sequence_l3320_332020

theorem largest_divisor_of_sequence :
  ∃ (x : ℕ), x = 18 ∧ 
  (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧
  (∀ z : ℕ, z > x → ∃ w : ℕ, ¬(z ∣ (7^w + 12*w - 1))) := by
  sorry

end largest_divisor_of_sequence_l3320_332020


namespace soccer_basketball_difference_l3320_332079

theorem soccer_basketball_difference :
  let soccer_boxes : ℕ := 8
  let basketball_boxes : ℕ := 5
  let balls_per_box : ℕ := 12
  let total_soccer_balls := soccer_boxes * balls_per_box
  let total_basketballs := basketball_boxes * balls_per_box
  total_soccer_balls - total_basketballs = 36 :=
by sorry

end soccer_basketball_difference_l3320_332079


namespace largest_three_digit_product_l3320_332045

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem largest_three_digit_product (n x y : ℕ) : 
  100 ≤ n ∧ n < 1000 ∧                  -- n is a three-digit number
  n = x * y * (5 * x + 2 * y) ∧         -- n is the product of x, y, and (5x+2y)
  x < 10 ∧ y < 10 ∧                     -- x and y are less than 10
  is_composite (5 * x + 2 * y) →        -- (5x+2y) is composite
  n ≤ 336 :=                            -- The largest possible value of n is 336
by sorry

end largest_three_digit_product_l3320_332045


namespace white_circle_area_on_cube_l3320_332092

/-- Represents the problem of calculating the area of a white circle on a cube face --/
theorem white_circle_area_on_cube (edge_length : ℝ) (green_paint_area : ℝ) : 
  edge_length = 12 → 
  green_paint_area = 432 → 
  (6 * edge_length^2 - green_paint_area) / 6 = 72 := by
  sorry

#check white_circle_area_on_cube

end white_circle_area_on_cube_l3320_332092


namespace BaSO4_molecular_weight_l3320_332050

/-- The atomic weight of Barium in g/mol -/
def Ba_weight : ℝ := 137.327

/-- The atomic weight of Sulfur in g/mol -/
def S_weight : ℝ := 32.065

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 15.999

/-- The number of Oxygen atoms in BaSO4 -/
def O_count : ℕ := 4

/-- The molecular weight of BaSO4 in g/mol -/
def BaSO4_weight : ℝ := Ba_weight + S_weight + O_count * O_weight

theorem BaSO4_molecular_weight : BaSO4_weight = 233.388 := by
  sorry

end BaSO4_molecular_weight_l3320_332050


namespace unique_persistent_number_l3320_332083

/-- Definition of a persistent number -/
def isPersistent (T : ℝ) : Prop :=
  ∀ a b c d : ℝ, a ≠ 0 → a ≠ 1 → b ≠ 0 → b ≠ 1 → c ≠ 0 → c ≠ 1 → d ≠ 0 → d ≠ 1 →
    (a + b + c + d = T ∧ 1/a + 1/b + 1/c + 1/d = T) →
    1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T

/-- Theorem: There exists a unique persistent number, and it equals 2 -/
theorem unique_persistent_number :
  ∃! T : ℝ, isPersistent T ∧ T = 2 := by
  sorry

end unique_persistent_number_l3320_332083


namespace intersection_on_y_axis_l3320_332033

/-- Proves that the intersection point of two specific polynomial graphs is on the y-axis -/
theorem intersection_on_y_axis (a b c : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃! x y : ℝ, (a * x^2 + b^2 * x^3 + c = y) ∧ (a * x^2 - b^2 * x^3 + c = y) ∧ x = 0 ∧ y = c := by
  sorry

end intersection_on_y_axis_l3320_332033


namespace initial_eggs_correct_l3320_332048

/-- The number of eggs initially in the basket -/
def initial_eggs : ℕ := 14

/-- The number of eggs remaining after a customer buys eggs -/
def remaining_eggs (n : ℕ) (eggs : ℕ) : ℕ :=
  eggs - (eggs / 2 + 1)

/-- Theorem stating that the initial number of eggs satisfies the given conditions -/
theorem initial_eggs_correct : 
  let eggs1 := remaining_eggs initial_eggs initial_eggs
  let eggs2 := remaining_eggs eggs1 eggs1
  let eggs3 := remaining_eggs eggs2 eggs2
  eggs3 = 0 := by sorry

end initial_eggs_correct_l3320_332048


namespace x_minus_p_equals_five_minus_two_p_l3320_332016

theorem x_minus_p_equals_five_minus_two_p (x p : ℝ) 
  (h1 : |x - 5| = p) (h2 : x < 5) : x - p = 5 - 2*p := by
  sorry

end x_minus_p_equals_five_minus_two_p_l3320_332016


namespace family_ages_l3320_332099

theorem family_ages :
  ∀ (dad mom kolya tanya : ℕ),
    dad = mom + 4 →
    kolya = tanya + 4 →
    2 * kolya = dad →
    dad + mom + kolya + tanya = 130 →
    dad = 46 ∧ mom = 42 ∧ kolya = 23 ∧ tanya = 19 :=
by
  sorry

end family_ages_l3320_332099


namespace ice_rinks_and_ski_resorts_2019_l3320_332090

/-- The number of ice skating rinks in 2019 -/
def ice_rinks_2019 : ℕ := 830

/-- The number of ski resorts in 2019 -/
def ski_resorts_2019 : ℕ := 400

/-- The total number of ice skating rinks and ski resorts in 2019 -/
def total_2019 : ℕ := 1230

/-- The total number of ice skating rinks and ski resorts in 2022 -/
def total_2022 : ℕ := 2560

/-- The increase in ice skating rinks from 2019 to 2022 -/
def ice_rinks_increase : ℕ := 212

/-- The increase in ski resorts from 2019 to 2022 -/
def ski_resorts_increase : ℕ := 288

theorem ice_rinks_and_ski_resorts_2019 :
  ice_rinks_2019 + ski_resorts_2019 = total_2019 ∧
  2 * ice_rinks_2019 + ice_rinks_increase + ski_resorts_2019 + ski_resorts_increase = total_2022 := by
  sorry

end ice_rinks_and_ski_resorts_2019_l3320_332090


namespace min_distance_Q_to_C_l3320_332072

noncomputable def A : ℝ × ℝ := (-1, 2)
noncomputable def B : ℝ × ℝ := (0, 1)

def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 4}

def l₁ : Set (ℝ × ℝ) := {q : ℝ × ℝ | 3 * q.1 - 4 * q.2 + 12 = 0}

theorem min_distance_Q_to_C :
  ∀ Q ∈ l₁, ∃ M ∈ C, ∀ M' ∈ C, dist Q M ≤ dist Q M' ∧ dist Q M ≥ Real.sqrt 5 :=
sorry

end min_distance_Q_to_C_l3320_332072


namespace diamond_equation_solution_l3320_332075

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := 3 * a + 3 * b^2

-- Theorem statement
theorem diamond_equation_solution :
  ∀ a : ℝ, diamond a 4 = 75 → a = 9 := by
  sorry

end diamond_equation_solution_l3320_332075


namespace system_two_solutions_l3320_332014

/-- The system of equations has exactly two solutions when a is in the specified interval -/
theorem system_two_solutions (a b : ℝ) : 
  (∃ x y : ℝ, 
    Real.arcsin ((a - y) / 3) = Real.arcsin ((4 - x) / 4) ∧
    x^2 + y^2 - 8*x - 8*y = b) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    Real.arcsin ((a - y₁) / 3) = Real.arcsin ((4 - x₁) / 4) ∧
    x₁^2 + y₁^2 - 8*x₁ - 8*y₁ = b ∧
    Real.arcsin ((a - y₂) / 3) = Real.arcsin ((4 - x₂) / 4) ∧
    x₂^2 + y₂^2 - 8*x₂ - 8*y₂ = b ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
    ∀ x₃ y₃ : ℝ, 
      Real.arcsin ((a - y₃) / 3) = Real.arcsin ((4 - x₃) / 4) ∧
      x₃^2 + y₃^2 - 8*x₃ - 8*y₃ = b →
      (x₃ = x₁ ∧ y₃ = y₁) ∨ (x₃ = x₂ ∧ y₃ = y₂)) ↔
  -13/3 < a ∧ a < 37/3 :=
by sorry

end system_two_solutions_l3320_332014


namespace mighty_l_league_teams_l3320_332046

/-- The number of teams in the league -/
def n : ℕ := 8

/-- The total number of games played -/
def total_games : ℕ := 28

/-- Formula for the number of games in a round-robin tournament -/
def games (x : ℕ) : ℕ := x * (x - 1) / 2

theorem mighty_l_league_teams :
  (n ≥ 2) ∧ (games n = total_games) := by sorry

end mighty_l_league_teams_l3320_332046


namespace multiplicative_inverse_301_mod_401_l3320_332081

theorem multiplicative_inverse_301_mod_401 : ∃ x : ℤ, 0 ≤ x ∧ x < 401 ∧ (301 * x) % 401 = 1 :=
  by
  use 397
  sorry

end multiplicative_inverse_301_mod_401_l3320_332081


namespace lindsay_workout_weight_l3320_332084

/-- Represents the resistance of exercise bands in pounds -/
structure Band where
  resistance : ℕ

/-- Represents a workout exercise with associated weights -/
structure Exercise where
  bands : List Band
  legWeights : ℕ
  additionalWeight : ℕ

/-- Calculates the total weight for an exercise -/
def totalWeight (e : Exercise) : ℕ :=
  (e.bands.map (λ b => b.resistance)).sum + 2 * e.legWeights + e.additionalWeight

/-- Lindsey's workout session -/
def lindseyWorkout : Prop :=
  let bandA : Band := ⟨7⟩
  let bandB : Band := ⟨5⟩
  let bandC : Band := ⟨3⟩
  let squats : Exercise := ⟨[bandA, bandB, bandC], 10, 15⟩
  let lunges : Exercise := ⟨[bandA, bandC], 8, 18⟩
  totalWeight squats + totalWeight lunges = 94

theorem lindsay_workout_weight : lindseyWorkout := by
  sorry

end lindsay_workout_weight_l3320_332084


namespace smaller_omelette_has_three_eggs_l3320_332049

/-- Represents the number of eggs in a smaller omelette -/
def smaller_omelette_eggs : ℕ := sorry

/-- Represents the number of eggs in a larger omelette -/
def larger_omelette_eggs : ℕ := 4

/-- Represents the number of smaller omelettes ordered in the first hour -/
def first_hour_smaller : ℕ := 5

/-- Represents the number of larger omelettes ordered in the second hour -/
def second_hour_larger : ℕ := 7

/-- Represents the number of smaller omelettes ordered in the third hour -/
def third_hour_smaller : ℕ := 3

/-- Represents the number of larger omelettes ordered in the fourth hour -/
def fourth_hour_larger : ℕ := 8

/-- Represents the total number of eggs used -/
def total_eggs : ℕ := 84

/-- Theorem stating that the number of eggs in a smaller omelette is 3 -/
theorem smaller_omelette_has_three_eggs :
  smaller_omelette_eggs = 3 :=
by
  have h1 : first_hour_smaller * smaller_omelette_eggs +
            second_hour_larger * larger_omelette_eggs +
            third_hour_smaller * smaller_omelette_eggs +
            fourth_hour_larger * larger_omelette_eggs = total_eggs := sorry
  sorry

end smaller_omelette_has_three_eggs_l3320_332049


namespace stating_total_dark_triangles_formula_l3320_332093

/-- 
Given a sequence of figures formed by an increasing number of dark equilateral triangles,
this function represents the total number of dark triangles used in the first n figures.
-/
def total_dark_triangles (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- 
Theorem stating that the total number of dark triangles used in the first n figures
of the sequence is (n(n+1)(n+2))/6.
-/
theorem total_dark_triangles_formula (n : ℕ) :
  total_dark_triangles n = n * (n + 1) * (n + 2) / 6 := by
  sorry

end stating_total_dark_triangles_formula_l3320_332093


namespace duck_pond_problem_l3320_332098

theorem duck_pond_problem (small_pond : ℕ) (large_pond : ℕ) 
  (green_small : ℚ) (green_large : ℚ) (total_green : ℚ) :
  large_pond = 50 →
  green_small = 1/5 →
  green_large = 3/25 →
  total_green = 3/20 →
  green_small * small_pond.cast + green_large * large_pond.cast = 
    total_green * (small_pond.cast + large_pond.cast) →
  small_pond = 30 := by
sorry

end duck_pond_problem_l3320_332098


namespace inlet_pipe_fill_rate_l3320_332037

/-- Given a tank with specified properties, prove the inlet pipe's fill rate --/
theorem inlet_pipe_fill_rate 
  (tank_capacity : ℝ) 
  (leak_empty_time : ℝ) 
  (combined_empty_time : ℝ) 
  (h1 : tank_capacity = 4320)
  (h2 : leak_empty_time = 6)
  (h3 : combined_empty_time = 8) :
  let leak_rate := tank_capacity / leak_empty_time
  let net_empty_rate := tank_capacity / combined_empty_time
  let inlet_rate := net_empty_rate + leak_rate
  inlet_rate / 60 = 21 := by sorry

end inlet_pipe_fill_rate_l3320_332037


namespace train_length_l3320_332097

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 18 → ∃ (length : ℝ), 
  (length ≥ 299.5 ∧ length ≤ 300.5) ∧ 
  length = speed * (1000 / 3600) * time := by
  sorry


end train_length_l3320_332097


namespace marble_problem_l3320_332041

theorem marble_problem (x y : ℕ) : 
  (y - 4 = 2 * (x + 4)) → 
  (y + 2 = 11 * (x - 2)) → 
  (y = 20 ∧ x = 4) :=
by sorry

end marble_problem_l3320_332041


namespace linear_function_unique_solution_l3320_332047

/-- A linear function is a function of the form f(x) = mx + b for some constants m and b. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_unique_solution (f : ℝ → ℝ) 
  (h_linear : LinearFunction f) (h1 : f 2 = 1) (h2 : f (-1) = -5) :
  ∀ x, f x = 2 * x - 3 := by
  sorry

end linear_function_unique_solution_l3320_332047


namespace prize_distribution_l3320_332036

theorem prize_distribution (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  n^k = 8000 := by
  sorry

end prize_distribution_l3320_332036


namespace iron_wire_remainder_l3320_332007

theorem iron_wire_remainder (total_length : ℚ) : 
  total_length > 0 → 
  total_length - (2/9 * total_length) - (3/9 * total_length) = 4/9 * total_length := by
sorry

end iron_wire_remainder_l3320_332007


namespace least_valid_number_l3320_332021

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is the least positive integer divisible by 17 with digit sum 17 -/
def is_least_valid (m : ℕ) : Prop :=
  m > 0 ∧ m % 17 = 0 ∧ digit_sum m = 17 ∧
  ∀ k : ℕ, 0 < k ∧ k < m → ¬(k % 17 = 0 ∧ digit_sum k = 17)

theorem least_valid_number : is_least_valid 476 := by sorry

end least_valid_number_l3320_332021


namespace power_equality_implies_y_equals_two_l3320_332022

theorem power_equality_implies_y_equals_two : 
  ∀ y : ℝ, (3 : ℝ)^6 = 27^y → y = 2 := by
sorry

end power_equality_implies_y_equals_two_l3320_332022


namespace phi_value_for_even_shifted_function_l3320_332012

/-- Given a function f and a real number φ, proves that if f(x) = (1/2) * sin(2x + π/6)
    and f(x - φ) is an even function, then φ = -π/6 -/
theorem phi_value_for_even_shifted_function 
  (f : ℝ → ℝ) 
  (φ : ℝ) 
  (h1 : ∀ x, f x = (1/2) * Real.sin (2*x + π/6))
  (h2 : ∀ x, f (x - φ) = f (φ - x)) :
  φ = -π/6 := by
  sorry


end phi_value_for_even_shifted_function_l3320_332012


namespace sqrt_three_irrational_l3320_332006

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_three_irrational_l3320_332006


namespace white_towels_count_l3320_332038

theorem white_towels_count (green_towels : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  green_towels = 35 → given_away = 34 → remaining = 22 → 
  ∃ white_towels : ℕ, white_towels = 21 ∧ green_towels + white_towels - given_away = remaining :=
by
  sorry

end white_towels_count_l3320_332038


namespace extreme_value_and_range_l3320_332085

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x + (1 - Real.log x) / x

theorem extreme_value_and_range (a : ℝ) :
  (∀ x > 0, f 0 x ≥ -1 / Real.exp 2) ∧
  (∀ x > 0, f 0 x = -1 / Real.exp 2 → x = Real.exp 2) ∧
  (∀ x > 0, f a x ≥ 1 ↔ a ≥ 1 / Real.exp 2) :=
by sorry

end extreme_value_and_range_l3320_332085


namespace min_value_reciprocal_sum_l3320_332010

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 1 → 1/x + 1/y ≥ 3 + 2*Real.sqrt 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ 1/x + 1/y = 3 + 2*Real.sqrt 2) :=
by sorry

end min_value_reciprocal_sum_l3320_332010


namespace correct_num_browsers_l3320_332087

/-- The number of browsers James had on his computer. -/
def num_browsers : ℕ := 2

/-- The number of windows per browser. -/
def windows_per_browser : ℕ := 3

/-- The number of tabs per window. -/
def tabs_per_window : ℕ := 10

/-- The total number of tabs in all browsers. -/
def total_tabs : ℕ := 60

/-- Theorem stating that the number of browsers is correct given the conditions. -/
theorem correct_num_browsers :
  num_browsers * windows_per_browser * tabs_per_window = total_tabs :=
by sorry

end correct_num_browsers_l3320_332087


namespace apple_cost_l3320_332042

theorem apple_cost (total_cost : ℝ) (initial_dozen : ℕ) (target_dozen : ℕ) :
  total_cost = 62.40 ∧ initial_dozen = 8 ∧ target_dozen = 5 →
  (target_dozen : ℝ) * (total_cost / initial_dozen) = 39.00 :=
by sorry

end apple_cost_l3320_332042


namespace triangle_area_is_four_l3320_332003

/-- The area of the triangle formed by the intersection of lines y = x + 2, y = -x + 8, and y = 3 -/
def triangleArea : ℝ := 4

/-- The first line equation: y = x + 2 -/
def line1 (x y : ℝ) : Prop := y = x + 2

/-- The second line equation: y = -x + 8 -/
def line2 (x y : ℝ) : Prop := y = -x + 8

/-- The third line equation: y = 3 -/
def line3 (x y : ℝ) : Prop := y = 3

theorem triangle_area_is_four :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    line1 x₁ y₁ ∧ line3 x₁ y₁ ∧
    line2 x₂ y₂ ∧ line3 x₂ y₂ ∧
    line1 x₃ y₃ ∧ line2 x₃ y₃ ∧
    triangleArea = 4 := by
  sorry

end triangle_area_is_four_l3320_332003


namespace ornament_profit_theorem_l3320_332023

-- Define the cost and selling prices
def costPriceA : ℝ := 2000
def costPriceB : ℝ := 1500
def sellingPriceA : ℝ := 2500
def sellingPriceB : ℝ := 1800

-- Define the total number of ornaments and maximum budget
def totalOrnaments : ℕ := 20
def maxBudget : ℝ := 36000

-- Define the profit function
def profitFunction (x : ℝ) : ℝ := 200 * x + 6000

-- Theorem statement
theorem ornament_profit_theorem :
  -- Condition 1: Cost price difference
  (costPriceA - costPriceB = 500) →
  -- Condition 2: Equal quantity purchased
  (40000 / costPriceA = 30000 / costPriceB) →
  -- Condition 3: Budget constraint
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ totalOrnaments → costPriceA * x + costPriceB * (totalOrnaments - x) ≤ maxBudget) →
  -- Conclusion 1: Correct profit function
  (∀ x : ℝ, profitFunction x = (sellingPriceA - costPriceA) * x + (sellingPriceB - costPriceB) * (totalOrnaments - x)) ∧
  -- Conclusion 2: Maximum profit
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ totalOrnaments ∧ 
    ∀ y : ℝ, 0 ≤ y ∧ y ≤ totalOrnaments → profitFunction x ≥ profitFunction y) ∧
  profitFunction 12 = 8400 := by
  sorry

end ornament_profit_theorem_l3320_332023


namespace temperature_at_4km_l3320_332089

def temperature_at_altitude (ground_temp : ℝ) (altitude : ℝ) : ℝ :=
  ground_temp - 5 * altitude

theorem temperature_at_4km (ground_temp : ℝ) (h1 : ground_temp = 15) : 
  temperature_at_altitude ground_temp 4 = -5 := by
  sorry

end temperature_at_4km_l3320_332089


namespace quadratic_rewrite_l3320_332009

theorem quadratic_rewrite (b : ℝ) (h1 : b > 0) : 
  (∃ n : ℝ, ∀ x : ℝ, x^2 + b*x + 60 = (x + n)^2 + 16) → b = 4 * Real.sqrt 11 := by
sorry

end quadratic_rewrite_l3320_332009


namespace max_crosses_on_10x11_board_l3320_332027

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a cross shape -/
structure CrossShape :=
  (size : ℕ := 3)
  (coverage : ℕ := 5)

/-- Defines the maximum number of non-overlapping cross shapes on a chessboard -/
def max_non_overlapping_crosses (board : Chessboard) (cross : CrossShape) : ℕ := sorry

/-- Theorem stating the maximum number of non-overlapping cross shapes on a 10x11 chessboard -/
theorem max_crosses_on_10x11_board :
  ∃ (board : Chessboard) (cross : CrossShape),
    board.rows = 10 ∧ board.cols = 11 ∧
    cross.size = 3 ∧ cross.coverage = 5 ∧
    max_non_overlapping_crosses board cross = 15 := by sorry

end max_crosses_on_10x11_board_l3320_332027


namespace gunther_working_time_l3320_332061

/-- Gunther's typing speed in words per minute -/
def typing_speed : ℚ := 160 / 3

/-- Total words Gunther types in a working day -/
def total_words : ℕ := 25600

/-- Gunther's working time in minutes -/
def working_time : ℕ := 480

theorem gunther_working_time :
  (total_words : ℚ) / typing_speed = working_time := by sorry

end gunther_working_time_l3320_332061


namespace min_gymnasts_is_30_l3320_332067

/-- Represents the total number of handshakes in a gymnastics meet -/
def total_handshakes : ℕ := 465

/-- Calculates the number of handshakes given the number of gymnasts -/
def handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2 + n

/-- Proves that 30 is the minimum number of gymnasts that satisfies the conditions -/
theorem min_gymnasts_is_30 :
  ∀ n : ℕ, n > 0 → n % 2 = 0 → handshakes n = total_handshakes → n ≥ 30 :=
by sorry

end min_gymnasts_is_30_l3320_332067


namespace mrs_heine_biscuits_l3320_332039

theorem mrs_heine_biscuits (num_dogs : ℕ) (biscuits_per_dog : ℕ) 
  (h1 : num_dogs = 2) (h2 : biscuits_per_dog = 3) : 
  num_dogs * biscuits_per_dog = 6 := by
  sorry

end mrs_heine_biscuits_l3320_332039


namespace cost_effectiveness_theorem_l3320_332001

/-- Represents the cost of a plan based on the number of students -/
def plan_cost (students : ℕ) (teacher_free : Bool) (discount : ℚ) : ℚ :=
  if teacher_free then
    25 * students
  else
    25 * discount * (students + 1)

/-- Determines which plan is more cost-effective based on the number of students -/
def cost_effective_plan (students : ℕ) : String :=
  let plan1_cost := plan_cost students true 1
  let plan2_cost := plan_cost students false (4/5)
  if plan1_cost < plan2_cost then "Plan 1"
  else if plan1_cost > plan2_cost then "Plan 2"
  else "Both plans are equally cost-effective"

theorem cost_effectiveness_theorem (students : ℕ) :
  cost_effective_plan students =
    if students < 4 then "Plan 1"
    else if students > 4 then "Plan 2"
    else "Both plans are equally cost-effective" :=
  sorry

end cost_effectiveness_theorem_l3320_332001


namespace straight_lines_parabolas_disjoint_l3320_332066

-- Define the set of all straight lines
def StraightLines : Set (ℝ → ℝ) := {f | ∃ a b : ℝ, ∀ x, f x = a * x + b}

-- Define the set of all parabolas
def Parabolas : Set (ℝ → ℝ) := {f | ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c}

-- Theorem statement
theorem straight_lines_parabolas_disjoint : StraightLines ∩ Parabolas = ∅ := by
  sorry

end straight_lines_parabolas_disjoint_l3320_332066


namespace exists_valid_numbering_scheme_l3320_332013

/-- Represents a numbering scheme for 7 pins and 7 holes -/
def NumberingScheme := Fin 7 → Fin 7

/-- Checks if a numbering scheme satisfies the condition for a given rotation -/
def isValidForRotation (scheme : NumberingScheme) (rotation : Fin 7) : Prop :=
  ∃ k : Fin 7, scheme k = (k + rotation : Fin 7)

/-- The main theorem stating that there exists a valid numbering scheme -/
theorem exists_valid_numbering_scheme :
  ∃ scheme : NumberingScheme, ∀ rotation : Fin 7, isValidForRotation scheme rotation := by
  sorry

end exists_valid_numbering_scheme_l3320_332013


namespace average_height_calculation_l3320_332028

theorem average_height_calculation (total_members : ℕ) (average_height : ℝ) 
  (two_member_height : ℝ) (remaining_members : ℕ) :
  total_members = 11 →
  average_height = 145.7 →
  two_member_height = 142.1 →
  remaining_members = total_members - 2 →
  (total_members * average_height - 2 * two_member_height) / remaining_members = 146.5 := by
sorry

end average_height_calculation_l3320_332028


namespace farmer_cows_l3320_332000

theorem farmer_cows (initial_cows : ℕ) (added_cows : ℕ) (sold_fraction : ℚ) 
  (h1 : initial_cows = 51)
  (h2 : added_cows = 5)
  (h3 : sold_fraction = 1 / 4) :
  initial_cows + added_cows - ⌊(initial_cows + added_cows : ℚ) * sold_fraction⌋ = 42 := by
  sorry

end farmer_cows_l3320_332000


namespace max_daily_revenue_l3320_332054

def price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

def sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def daily_revenue (t : ℕ) : ℝ :=
  price t * sales_volume t

theorem max_daily_revenue :
  ∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ daily_revenue t = 1125 ∧
  ∀ s : ℕ, 0 < s ∧ s ≤ 30 → daily_revenue s ≤ daily_revenue t :=
sorry

end max_daily_revenue_l3320_332054


namespace luna_has_seventeen_badges_l3320_332052

/-- The number of spelling badges Luna has, given the total number of badges and the number of badges Hermione and Celestia have. -/
def luna_badges (total : ℕ) (hermione : ℕ) (celestia : ℕ) : ℕ :=
  total - (hermione + celestia)

/-- Theorem stating that Luna has 17 spelling badges given the conditions in the problem. -/
theorem luna_has_seventeen_badges :
  luna_badges 83 14 52 = 17 := by
  sorry

end luna_has_seventeen_badges_l3320_332052


namespace smallest_n_congruence_three_satisfies_congruence_three_is_smallest_l3320_332086

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 23 * n ≡ 789 [MOD 8] → n ≥ 3 :=
by sorry

theorem three_satisfies_congruence : 23 * 3 ≡ 789 [MOD 8] :=
by sorry

theorem three_is_smallest (m : ℕ) : m > 0 ∧ 23 * m ≡ 789 [MOD 8] → m ≥ 3 :=
by sorry

end smallest_n_congruence_three_satisfies_congruence_three_is_smallest_l3320_332086


namespace tuning_day_method_pi_approximation_l3320_332032

/-- The "Tuning Day Method" for approximating a real number -/
def tuningDayMethod (a b c d : ℕ) : ℚ := (b + d) / (a + c)

/-- Apply the Tuning Day Method n times -/
def applyTuningDayMethod (n : ℕ) (a b c d : ℕ) : ℚ :=
  match n with
  | 0 => b / a
  | n+1 => tuningDayMethod a b c d

theorem tuning_day_method_pi_approximation :
  applyTuningDayMethod 3 10 31 5 16 = 22 / 7 := by
  sorry


end tuning_day_method_pi_approximation_l3320_332032


namespace prize_problem_solution_l3320_332026

/-- Represents the cost and quantity of pens and notebooks --/
structure PrizeInfo where
  pen_cost : ℚ
  notebook_cost : ℚ
  total_prizes : ℕ
  max_total_cost : ℚ

/-- Theorem stating the solution to the prize problem --/
theorem prize_problem_solution (info : PrizeInfo) 
  (h1 : 2 * info.pen_cost + 3 * info.notebook_cost = 62)
  (h2 : 5 * info.pen_cost + info.notebook_cost = 90)
  (h3 : info.total_prizes = 80)
  (h4 : info.max_total_cost = 1100) :
  info.pen_cost = 16 ∧ 
  info.notebook_cost = 10 ∧ 
  (∀ m : ℕ, m * info.pen_cost + (info.total_prizes - m) * info.notebook_cost ≤ info.max_total_cost → m ≤ 50) :=
by sorry


end prize_problem_solution_l3320_332026


namespace total_purchase_cost_l3320_332080

/-- Represents the price of a single small pack -/
def small_pack_price : ℚ := 387 / 100

/-- Represents the price of a single large pack -/
def large_pack_price : ℚ := 549 / 100

/-- Calculates the cost of small packs with bulk pricing -/
def small_pack_cost (n : ℕ) : ℚ :=
  if n ≥ 10 then n * small_pack_price * (1 - 1/10)
  else if n ≥ 5 then 5 * small_pack_price * (1 - 1/20) + (n - 5) * small_pack_price
  else n * small_pack_price

/-- Calculates the cost of large packs with bulk pricing -/
def large_pack_cost (n : ℕ) : ℚ :=
  if n ≥ 6 then n * large_pack_price * (1 - 3/20)
  else if n ≥ 3 then 3 * large_pack_price * (1 - 7/100) + (n - 3) * large_pack_price
  else n * large_pack_price

/-- Theorem stating the total cost of the purchase -/
theorem total_purchase_cost :
  (small_pack_cost 8 + large_pack_cost 4) * 100 = 5080 := by
  sorry

end total_purchase_cost_l3320_332080


namespace square_perimeter_from_area_l3320_332008

theorem square_perimeter_from_area (s : Real) (area : Real) (perimeter : Real) :
  (s ^ 2 = area) → (area = 36) → (perimeter = 4 * s) → (perimeter = 24) := by
  sorry

end square_perimeter_from_area_l3320_332008


namespace right_pentagonal_prism_characterization_cone_characterization_l3320_332056

-- Define geometric shapes
def RightPentagonalPrism : Type := sorry
def Cone : Type := sorry

-- Define properties of shapes
def has_seven_faces (shape : Type) : Prop := sorry
def has_two_parallel_congruent_pentagons (shape : Type) : Prop := sorry
def has_congruent_rectangle_faces (shape : Type) : Prop := sorry
def formed_by_rotating_isosceles_triangle (shape : Type) : Prop := sorry
def rotated_180_degrees (shape : Type) : Prop := sorry
def rotated_around_height_line (shape : Type) : Prop := sorry

-- Theorem 1
theorem right_pentagonal_prism_characterization (shape : Type) :
  has_seven_faces shape ∧
  has_two_parallel_congruent_pentagons shape ∧
  has_congruent_rectangle_faces shape →
  shape = RightPentagonalPrism :=
sorry

-- Theorem 2
theorem cone_characterization (shape : Type) :
  formed_by_rotating_isosceles_triangle shape ∧
  rotated_180_degrees shape ∧
  rotated_around_height_line shape →
  shape = Cone :=
sorry

end right_pentagonal_prism_characterization_cone_characterization_l3320_332056


namespace loan_principal_calculation_l3320_332058

/-- Calculates the principal amount given the interest rate, time, and total interest. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℕ) : ℚ :=
  (interest : ℚ) * 100 / (rate * (time : ℚ))

/-- Proves that for a loan with 12% p.a. simple interest, if the interest after 3 years
    is Rs. 5400, then the principal amount borrowed was Rs. 15000. -/
theorem loan_principal_calculation (rate : ℚ) (time : ℕ) (interest : ℕ) :
  rate = 12 → time = 3 → interest = 5400 →
  calculate_principal rate time interest = 15000 := by
  sorry

#eval calculate_principal 12 3 5400

end loan_principal_calculation_l3320_332058


namespace min_abs_sum_with_constraints_l3320_332015

theorem min_abs_sum_with_constraints (α β γ : ℝ) 
  (sum_constraint : α + β + γ = 2)
  (product_constraint : α * β * γ = 4) :
  ∃ v : ℝ, v = 6 ∧ ∀ α' β' γ' : ℝ, 
    (α' + β' + γ' = 2) → (α' * β' * γ' = 4) → 
    v ≤ |α'| + |β'| + |γ'| :=
by sorry

end min_abs_sum_with_constraints_l3320_332015


namespace hadassah_additional_paintings_l3320_332096

/-- Calculates the number of additional paintings given initial and total painting information -/
def additional_paintings (initial_paintings : ℕ) (initial_time : ℕ) (total_time : ℕ) : ℕ :=
  let painting_rate := initial_paintings / initial_time
  let additional_time := total_time - initial_time
  painting_rate * additional_time

/-- Proves that Hadassah painted 20 additional paintings -/
theorem hadassah_additional_paintings :
  additional_paintings 12 6 16 = 20 := by
  sorry

end hadassah_additional_paintings_l3320_332096


namespace solve_class_selection_problem_l3320_332088

/-- The number of students who selected only "Selected Lectures on Geometric Proofs" -/
def students_only_geometric_proofs (total : ℕ) (both : ℕ) (difference : ℕ) : ℕ :=
  let geometric := (total + both - difference) / 2
  geometric - both

/-- The main theorem -/
theorem solve_class_selection_problem :
  let total := 54  -- Total number of students
  let both := 6    -- Number of students who selected both topics
  let difference := 8  -- Difference between selections
  students_only_geometric_proofs total both difference = 20 := by
  sorry

#eval students_only_geometric_proofs 54 6 8

end solve_class_selection_problem_l3320_332088


namespace all_vertices_integer_l3320_332076

/-- A cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℤ × ℤ × ℤ

/-- Predicate to check if four vertices form a valid cube face -/
def is_valid_face (v₁ v₂ v₃ v₄ : ℤ × ℤ × ℤ) : Prop := sorry

/-- Predicate to check if four vertices are non-coplanar -/
def are_non_coplanar (v₁ v₂ v₃ v₄ : ℤ × ℤ × ℤ) : Prop := sorry

/-- Theorem: If four non-coplanar vertices of a cube have integer coordinates, 
    then all vertices of the cube have integer coordinates -/
theorem all_vertices_integer (c : Cube) 
  (h₁ : is_valid_face (c.vertices 0) (c.vertices 1) (c.vertices 2) (c.vertices 3))
  (h₂ : are_non_coplanar (c.vertices 0) (c.vertices 1) (c.vertices 2) (c.vertices 3)) :
  ∀ i, ∃ (x y z : ℤ), c.vertices i = (x, y, z) := by
  sorry


end all_vertices_integer_l3320_332076


namespace walking_delay_bus_miss_time_l3320_332063

/-- Given a usual walking time and a reduced speed factor, calculates the delay in reaching the destination. -/
theorem walking_delay (usual_time : ℝ) (speed_factor : ℝ) : 
  usual_time > 0 → 
  speed_factor > 0 → 
  speed_factor < 1 → 
  (usual_time / speed_factor) - usual_time = usual_time * (1 / speed_factor - 1) :=
by sorry

/-- Proves that walking at 4/5 of the usual speed, with a usual time of 24 minutes, results in a 6-minute delay. -/
theorem bus_miss_time (usual_time : ℝ) (h1 : usual_time = 24) : 
  (usual_time / (4/5)) - usual_time = 6 :=
by sorry

end walking_delay_bus_miss_time_l3320_332063


namespace remainder_sum_l3320_332074

theorem remainder_sum (n : ℤ) : n % 20 = 13 → (n % 4 + n % 5 = 4) := by
  sorry

end remainder_sum_l3320_332074


namespace set_A_characterization_l3320_332059

def A : Set ℝ := {a | ∃! x, (x + a) / (x^2 - 1) = 1}

theorem set_A_characterization : A = {-1, 1, -5/4} := by sorry

end set_A_characterization_l3320_332059


namespace rectangle_width_l3320_332019

/-- Given a rectangle with perimeter 48 cm and width 2 cm shorter than length, prove width is 11 cm -/
theorem rectangle_width (length width : ℝ) : 
  (2 * length + 2 * width = 48) →  -- Perimeter condition
  (width = length - 2) →           -- Width-length relation
  (width = 11) :=                  -- Conclusion to prove
by
  sorry

end rectangle_width_l3320_332019


namespace equation_solution_l3320_332005

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (8 * x)^12 = (16 * x)^6 ↔ x = 1/4 := by sorry

end equation_solution_l3320_332005


namespace calendar_puzzle_l3320_332078

def date_behind (letter : Char) (base : ℕ) : ℕ :=
  match letter with
  | 'A' => base
  | 'B' => base + 1
  | 'C' => base + 2
  | 'D' => base + 3
  | 'E' => base + 4
  | 'F' => base + 5
  | 'G' => base + 6
  | _ => base

theorem calendar_puzzle (base : ℕ) :
  ∃ (x : Char), (date_behind 'B' base + date_behind x base = 2 * date_behind 'A' base + 6) ∧ x = 'F' :=
by sorry

end calendar_puzzle_l3320_332078


namespace lily_cups_count_l3320_332018

/-- Represents Gina's cup painting rates and order details -/
structure PaintingOrder where
  rose_rate : ℕ  -- Roses painted per hour
  lily_rate : ℕ  -- Lilies painted per hour
  rose_order : ℕ  -- Number of rose cups ordered
  total_pay : ℕ  -- Total payment for the order in dollars
  hourly_rate : ℕ  -- Gina's hourly rate in dollars

/-- Calculates the number of lily cups in the order -/
def lily_cups (order : PaintingOrder) : ℕ :=
  let total_hours := order.total_pay / order.hourly_rate
  let rose_hours := order.rose_order / order.rose_rate
  let lily_hours := total_hours - rose_hours
  lily_hours * order.lily_rate

/-- Theorem stating that for the given order, the number of lily cups is 14 -/
theorem lily_cups_count (order : PaintingOrder) 
  (h1 : order.rose_rate = 6)
  (h2 : order.lily_rate = 7)
  (h3 : order.rose_order = 6)
  (h4 : order.total_pay = 90)
  (h5 : order.hourly_rate = 30) :
  lily_cups order = 14 := by
  sorry

#eval lily_cups { rose_rate := 6, lily_rate := 7, rose_order := 6, total_pay := 90, hourly_rate := 30 }

end lily_cups_count_l3320_332018


namespace average_weight_problem_l3320_332051

/-- Given the average weights of three people and two of them, along with the weight of one person,
    prove that the average weight of the other two is as stated. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 → 
  (a + b) / 2 = 40 → 
  b = 33 → 
  (b + c) / 2 = 44 := by
sorry

end average_weight_problem_l3320_332051


namespace order_of_three_trig_expressions_l3320_332091

theorem order_of_three_trig_expressions :
  Real.arcsin (3/4) < Real.arccos (1/5) ∧ Real.arccos (1/5) < 1 + Real.arctan (2/3) := by
  sorry

end order_of_three_trig_expressions_l3320_332091


namespace lcm_gcf_product_20_90_l3320_332094

theorem lcm_gcf_product_20_90 : Nat.lcm 20 90 * Nat.gcd 20 90 = 1800 := by
  sorry

end lcm_gcf_product_20_90_l3320_332094


namespace rectangle_ratio_is_two_l3320_332068

/-- Geometric arrangement of squares and rectangles -/
structure SquareFrame where
  inner_side : ℝ
  outer_side : ℝ
  rect_short : ℝ
  rect_long : ℝ
  area_ratio : outer_side^2 = 9 * inner_side^2
  outer_side_composition : outer_side = inner_side + 2 * rect_short
  inner_side_composition : inner_side + rect_long = outer_side

/-- Theorem: The ratio of the longer side to the shorter side of each rectangle is 2 -/
theorem rectangle_ratio_is_two (frame : SquareFrame) :
  frame.rect_long / frame.rect_short = 2 := by
  sorry

end rectangle_ratio_is_two_l3320_332068


namespace min_intersection_cardinality_l3320_332040

-- Define the sets A, B, and C
variable (A B C : Set α)

-- Define the cardinality function
variable (card : Set α → ℕ)

-- Define the conditions
variable (h1 : card A = 50)
variable (h2 : card B = 50)
variable (h3 : card (A ∩ B) = 45)
variable (h4 : card (B ∩ C) = 40)
variable (h5 : card A + card B + card C = card (A ∪ B ∪ C))

-- State the theorem
theorem min_intersection_cardinality :
  ∃ (x : ℕ), x = card (A ∩ B ∩ C) ∧ 
  (∀ (y : ℕ), y = card (A ∩ B ∩ C) → x ≤ y) ∧
  x = 21 := by
sorry

end min_intersection_cardinality_l3320_332040


namespace box_volume_calculation_l3320_332035

/-- Given a rectangular metallic sheet and squares cut from each corner, calculate the volume of the resulting box. -/
theorem box_volume_calculation (sheet_length sheet_width cut_square_side : ℝ) 
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_square_side = 4) :
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = 4480 := by
  sorry

#check box_volume_calculation

end box_volume_calculation_l3320_332035


namespace simplify_fraction_a_l3320_332034

theorem simplify_fraction_a (a b c d : ℝ) :
  (3 * a^4 * c + 2 * a^4 * d - 3 * b^4 * c - 2 * b^4 * d) /
  ((9 * c^2 * (a - b) - 4 * d^2 * (a - b)) * ((a + b)^2 - 2 * a * b)) =
  (a + b) / (3 * c - 2 * d) :=
sorry


end simplify_fraction_a_l3320_332034
