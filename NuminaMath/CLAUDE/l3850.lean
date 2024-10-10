import Mathlib

namespace negation_equivalence_l3850_385025

theorem negation_equivalence :
  (¬ ∃ (x y : ℝ), 2*x + 3*y + 3 < 0) ↔ (∀ (x y : ℝ), 2*x + 3*y + 3 ≥ 0) :=
by sorry

end negation_equivalence_l3850_385025


namespace triplet_equality_l3850_385076

theorem triplet_equality (p q r s t u v : ℤ) : 
  -- Formulation 1
  (q + r + (p + r) + (2*p + 2*q + r) = r + (p + 2*q + r) + (2*p + q + r)) ∧
  ((q + r)^2 + (p + r)^2 + (2*p + 2*q + r)^2 = r^2 + (p + 2*q + r)^2 + (2*p + q + r)^2) ∧
  -- Formulation 2
  (u*v = s*t → 
    (s + t + (u + v) = u + v + (s + t)) ∧
    (s^2 + t^2 + (u + v)^2 = u^2 + v^2 + (s + t)^2)) := by
  sorry

end triplet_equality_l3850_385076


namespace field_trip_girls_fraction_l3850_385053

theorem field_trip_girls_fraction (total_students : ℕ) (h_positive : total_students > 0) :
  let girls : ℚ := total_students / 2
  let boys : ℚ := total_students / 2
  let girls_on_trip : ℚ := (4 / 5) * girls
  let boys_on_trip : ℚ := (3 / 4) * boys
  let total_on_trip : ℚ := girls_on_trip + boys_on_trip
  (girls_on_trip / total_on_trip) = 16 / 31 := by
  sorry

end field_trip_girls_fraction_l3850_385053


namespace direction_vector_of_determinant_line_l3850_385041

/-- Given a line in 2D space defined by the determinant equation |x y; 2 1| = 3,
    prove that (-2, -1) is a direction vector of this line. -/
theorem direction_vector_of_determinant_line :
  let line := {(x, y) : ℝ × ℝ | x - 2*y = 3}
  ((-2 : ℝ), -1) ∈ {v : ℝ × ℝ | ∃ (t : ℝ), ∀ (p q : ℝ × ℝ), p ∈ line → q ∈ line → ∃ (s : ℝ), q.1 - p.1 = s * v.1 ∧ q.2 - p.2 = s * v.2} :=
by sorry

end direction_vector_of_determinant_line_l3850_385041


namespace inequality_range_l3850_385094

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 4 * x + a > 1 - 2 * x^2) → a > 2 := by
  sorry

end inequality_range_l3850_385094


namespace rectangular_field_area_increase_l3850_385090

theorem rectangular_field_area_increase 
  (original_length : ℝ) 
  (original_width : ℝ) 
  (length_increase : ℝ) : 
  original_length = 20 →
  original_width = 5 →
  length_increase = 10 →
  (original_length + length_increase) * original_width - original_length * original_width = 50 := by
  sorry

end rectangular_field_area_increase_l3850_385090


namespace inequality_implies_range_l3850_385084

theorem inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, a * Real.sin x - Real.cos x ^ 2 ≤ 3) →
  -3 ≤ a ∧ a ≤ 3 := by
  sorry

end inequality_implies_range_l3850_385084


namespace square_difference_area_l3850_385061

theorem square_difference_area (a b : ℝ) (h : a > b) :
  (a ^ 2 - b ^ 2 : ℝ) = (Real.sqrt (a ^ 2 - b ^ 2)) ^ 2 := by
  sorry

end square_difference_area_l3850_385061


namespace linear_system_solution_inequality_system_solution_l3850_385044

-- Part 1: System of linear equations
theorem linear_system_solution :
  let x : ℝ := 5
  let y : ℝ := 1
  (x - 5 * y = 0) ∧ (3 * x + 2 * y = 17) := by sorry

-- Part 2: System of inequalities
theorem inequality_system_solution :
  ∀ x : ℝ, x < -1/5 →
    (2 * (x - 2) ≤ 3 - x) ∧ (1 - (2 * x + 1) / 3 > x + 1) := by sorry

end linear_system_solution_inequality_system_solution_l3850_385044


namespace negative_three_star_negative_two_nested_star_op_l3850_385008

-- Define the custom operation
def star_op (a b : ℤ) : ℤ := a^2 - b + a * b

-- Theorem statements
theorem negative_three_star_negative_two : star_op (-3) (-2) = 17 := by sorry

theorem nested_star_op : star_op (-2) (star_op (-3) (-2)) = -47 := by sorry

end negative_three_star_negative_two_nested_star_op_l3850_385008


namespace satisfaction_survey_stats_l3850_385003

def data : List ℝ := [34, 35, 35, 36]

theorem satisfaction_survey_stats (median mode mean variance : ℝ) :
  median = 35 ∧
  mode = 35 ∧
  mean = 35 ∧
  variance = 0.5 := by
  sorry

end satisfaction_survey_stats_l3850_385003


namespace system_of_equations_solution_l3850_385027

theorem system_of_equations_solution (a b c d e f g : ℚ) : 
  a + b + c + d + e = 1 →
  b + c + d + e + f = 2 →
  c + d + e + f + g = 3 →
  d + e + f + g + a = 4 →
  e + f + g + a + b = 5 →
  f + g + a + b + c = 6 →
  g + a + b + c + d = 7 →
  g = 13/3 := by
sorry

end system_of_equations_solution_l3850_385027


namespace evaluate_expression_l3850_385054

theorem evaluate_expression : (3^10 + 3^7) / (3^10 - 3^7) = 14/13 := by
  sorry

end evaluate_expression_l3850_385054


namespace difference_of_squares_specific_values_l3850_385005

theorem difference_of_squares_specific_values :
  let x : ℤ := 10
  let y : ℤ := 15
  (x - y) * (x + y) = -125 := by
sorry

end difference_of_squares_specific_values_l3850_385005


namespace red_balls_count_l3850_385010

theorem red_balls_count (x : ℕ) (h : (4 : ℝ) / (x + 4) = (1 : ℝ) / 5) : x = 16 := by
  sorry

end red_balls_count_l3850_385010


namespace probability_yellow_second_marble_l3850_385015

/-- Represents the contents of a bag of marbles -/
structure BagContents where
  color1 : String
  count1 : ℕ
  color2 : String
  count2 : ℕ

/-- Calculates the probability of drawing a specific color from a bag -/
def probDrawColor (bag : BagContents) (color : String) : ℚ :=
  if color = bag.color1 then
    bag.count1 / (bag.count1 + bag.count2)
  else if color = bag.color2 then
    bag.count2 / (bag.count1 + bag.count2)
  else
    0

theorem probability_yellow_second_marble 
  (bagX : BagContents)
  (bagY : BagContents)
  (bagZ : BagContents)
  (h1 : bagX = { color1 := "white", count1 := 5, color2 := "black", count2 := 5 })
  (h2 : bagY = { color1 := "yellow", count1 := 8, color2 := "blue", count2 := 2 })
  (h3 : bagZ = { color1 := "yellow", count1 := 3, color2 := "blue", count2 := 4 })
  : probDrawColor bagX "white" * probDrawColor bagY "yellow" +
    probDrawColor bagX "black" * probDrawColor bagZ "yellow" = 43/70 := by
  sorry


end probability_yellow_second_marble_l3850_385015


namespace right_triangle_shorter_leg_l3850_385071

theorem right_triangle_shorter_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 25             -- Shorter leg length
  := by sorry

end right_triangle_shorter_leg_l3850_385071


namespace primitive_pythagorean_triple_parity_l3850_385098

theorem primitive_pythagorean_triple_parity (a b c : ℕ+) 
  (h1 : a^2 + b^2 = c^2)
  (h2 : Nat.gcd a.val (Nat.gcd b.val c.val) = 1) :
  (Even a.val ∧ Odd b.val) ∨ (Odd a.val ∧ Even b.val) := by
sorry

end primitive_pythagorean_triple_parity_l3850_385098


namespace min_reciprocal_sum_l3850_385032

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (min : ℝ), min = 4 ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y ≥ min)) ∧
  (1/a + 1/b = 4 ↔ a = 1/2) :=
by sorry

end min_reciprocal_sum_l3850_385032


namespace multiply_25_26_8_multiply_divide_340_40_17_sum_products_15_l3850_385042

-- Part 1
theorem multiply_25_26_8 : 25 * 26 * 8 = 5200 := by sorry

-- Part 2
theorem multiply_divide_340_40_17 : 340 * 40 / 17 = 800 := by sorry

-- Part 3
theorem sum_products_15 : 440 * 15 + 480 * 15 + 79 * 15 + 15 = 15000 := by sorry

end multiply_25_26_8_multiply_divide_340_40_17_sum_products_15_l3850_385042


namespace coreys_weekend_goal_l3850_385017

/-- Corey's goal for the number of golf balls to find every weekend -/
def coreys_goal (saturday_balls sunday_balls remaining_balls : ℕ) : ℕ :=
  saturday_balls + sunday_balls + remaining_balls

/-- Theorem stating Corey's goal for the number of golf balls to find every weekend -/
theorem coreys_weekend_goal :
  coreys_goal 16 18 14 = 48 := by
  sorry

end coreys_weekend_goal_l3850_385017


namespace min_value_abc_l3850_385024

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b * c = 4 * (a + b)) : 
  a + b + c ≥ 8 ∧ (a + b + c = 8 ↔ a = 2 ∧ b = 2) :=
by sorry

end min_value_abc_l3850_385024


namespace marble_transfer_result_l3850_385055

/-- Represents the marble transfer game between A and B -/
def marbleTransfer (a b n : ℕ) : Prop :=
  -- Initial conditions
  b < a ∧
  -- After 2n transfers, A has b marbles
  -- The ratio of initial marbles (a) to final marbles (b) is given by the formula
  (a : ℚ) / b = (2 * (4^n + 1)) / (1 - 4^n)

/-- Theorem stating the result of the marble transfer game -/
theorem marble_transfer_result {a b n : ℕ} (h : marbleTransfer a b n) :
  (a : ℚ) / b = (2 * (4^n + 1)) / (1 - 4^n) :=
by
  sorry

#check marble_transfer_result

end marble_transfer_result_l3850_385055


namespace moving_circle_trajectory_l3850_385031

/-- The trajectory of the center of a moving circle externally tangent to a fixed circle and the y-axis -/
theorem moving_circle_trajectory (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ 
    -- The moving circle is externally tangent to (x-2)^2 + y^2 = 1
    ((x - 2)^2 + y^2 = (r + 1)^2) ∧ 
    -- The moving circle is tangent to the y-axis
    (x = r)) →
  y^2 = 6*x - 3 := by
sorry

end moving_circle_trajectory_l3850_385031


namespace independence_and_polynomial_value_l3850_385073

/-- The algebraic expression is independent of x -/
def is_independent_of_x (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (2 - 2*b) * x^2 + (a + 3) * x - 6*y + 7 = -6*y + 7

/-- The value of the polynomial given a and b -/
def polynomial_value (a b : ℝ) : ℝ :=
  3*(a^2 - 2*a*b - b^2) - (4*a^2 + a*b + b^2)

theorem independence_and_polynomial_value :
  ∃ a b : ℝ, is_independent_of_x a b ∧ a = -3 ∧ b = 1 ∧ polynomial_value a b = 8 := by
  sorry

end independence_and_polynomial_value_l3850_385073


namespace isosceles_trapezoid_base_ratio_l3850_385093

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The length of the larger base -/
  largerBase : ℝ
  /-- The length of the smaller base -/
  smallerBase : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The left segment of the larger base divided by the height -/
  leftSegment : ℝ
  /-- The right segment of the larger base divided by the height -/
  rightSegment : ℝ
  /-- The larger base is positive -/
  largerBase_pos : 0 < largerBase
  /-- The smaller base is positive -/
  smallerBase_pos : 0 < smallerBase
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The sum of segments equals the larger base -/
  segment_sum : leftSegment + rightSegment = largerBase
  /-- The ratio of segments is 2:3 -/
  segment_ratio : leftSegment / rightSegment = 2 / 3

/-- 
If the height of an isosceles trapezoid divides the larger base into segments 
with a ratio of 2:3, then the ratio of the larger base to the smaller base is 5:1
-/
theorem isosceles_trapezoid_base_ratio (t : IsoscelesTrapezoid) : 
  t.largerBase / t.smallerBase = 5 := by
  sorry

end isosceles_trapezoid_base_ratio_l3850_385093


namespace product_equals_57_over_168_l3850_385009

def product : ℚ :=
  (2^3 - 1) / (2^3 + 1) *
  (3^3 - 1) / (3^3 + 1) *
  (4^3 - 1) / (4^3 + 1) *
  (5^3 - 1) / (5^3 + 1) *
  (6^3 - 1) / (6^3 + 1) *
  (7^3 - 1) / (7^3 + 1)

theorem product_equals_57_over_168 : product = 57 / 168 := by
  sorry

end product_equals_57_over_168_l3850_385009


namespace light_glow_theorem_l3850_385066

/-- The number of times a light glows in a given time interval -/
def glowCount (interval : ℕ) (period : ℕ) : ℕ :=
  interval / period

/-- The number of times all lights glow simultaneously in a given time interval -/
def simultaneousGlowCount (interval : ℕ) (periodA periodB periodC : ℕ) : ℕ :=
  interval / (lcm (lcm periodA periodB) periodC)

theorem light_glow_theorem (totalInterval : ℕ) (periodA periodB periodC : ℕ)
    (h1 : totalInterval = 4969)
    (h2 : periodA = 18)
    (h3 : periodB = 24)
    (h4 : periodC = 30) :
    glowCount totalInterval periodA = 276 ∧
    glowCount totalInterval periodB = 207 ∧
    glowCount totalInterval periodC = 165 ∧
    simultaneousGlowCount totalInterval periodA periodB periodC = 13 := by
  sorry

end light_glow_theorem_l3850_385066


namespace sin_cos_sum_equals_sqrt3_over_2_l3850_385052

theorem sin_cos_sum_equals_sqrt3_over_2 :
  Real.sin (47 * π / 180) * Real.sin (103 * π / 180) +
  Real.sin (43 * π / 180) * Real.cos (77 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_sum_equals_sqrt3_over_2_l3850_385052


namespace expand_product_l3850_385043

theorem expand_product (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * ((7 / y) - 14 * y^3 + 21) = 3 / y - 6 * y^3 + 9 := by
  sorry

end expand_product_l3850_385043


namespace paper_fold_sum_l3850_385079

-- Define the fold line
def fold_line (x y : ℝ) : Prop := y = x

-- Define the mapping of points
def maps_to (x1 y1 x2 y2 : ℝ) : Prop :=
  fold_line ((x1 + x2) / 2) ((y1 + y2) / 2) ∧
  (y2 - y1) = -(x2 - x1)

-- Main theorem
theorem paper_fold_sum (m n : ℝ) :
  maps_to 0 5 5 0 →  -- (0,5) maps to (5,0)
  maps_to 8 4 m n →  -- (8,4) maps to (m,n)
  m + n = 12 := by
sorry

end paper_fold_sum_l3850_385079


namespace bus_problem_l3850_385074

/-- Calculates the final number of people on a bus given initial count and changes -/
def final_bus_count (initial : ℕ) (getting_on : ℕ) (getting_off : ℕ) : ℕ :=
  initial + getting_on - getting_off

/-- Theorem stating that given 22 initial people, 4 getting on, and 8 getting off, 
    the final count is 18 -/
theorem bus_problem : final_bus_count 22 4 8 = 18 := by
  sorry

end bus_problem_l3850_385074


namespace prime_sum_of_squares_divisibility_l3850_385036

theorem prime_sum_of_squares_divisibility (p : ℕ) (h_prime : Prime p) 
  (h_sum : ∃ a : ℕ, 2 * p = a^2 + (a+1)^2 + (a+2)^2 + (a+3)^2) : 
  36 ∣ (p - 7) := by
sorry

end prime_sum_of_squares_divisibility_l3850_385036


namespace negative_abs_of_negative_one_l3850_385047

theorem negative_abs_of_negative_one : -|-1| = -1 := by
  sorry

end negative_abs_of_negative_one_l3850_385047


namespace estimate_larger_than_original_l3850_385029

theorem estimate_larger_than_original 
  (x y a b ε : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxy : x > y) 
  (ha : a > 1) 
  (hb : b > 1) 
  (hab : a > b) 
  (hε : ε > 0) : 
  (a * x + ε) - (b * y - ε) > a * x - b * y := by
  sorry

end estimate_larger_than_original_l3850_385029


namespace line_slope_from_parametric_equation_l3850_385081

/-- Given a line l with parametric equations x = 1 - (3/5)t and y = (4/5)t,
    prove that the slope of the line is -4/3 -/
theorem line_slope_from_parametric_equation :
  ∀ (l : ℝ → ℝ × ℝ),
  (∀ t, l t = (1 - 3/5 * t, 4/5 * t)) →
  (∃ m b, ∀ x y, (x, y) ∈ Set.range l → y = m * x + b) →
  (∃ m b, ∀ x y, (x, y) ∈ Set.range l → y = m * x + b ∧ m = -4/3) :=
by sorry

end line_slope_from_parametric_equation_l3850_385081


namespace fourth_term_is_54_l3850_385067

/-- A positive geometric sequence with specific properties -/
structure SpecialGeometricSequence where
  a : ℕ → ℝ
  is_positive : ∀ n, a n > 0
  is_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q
  first_term : a 1 = 2
  arithmetic_mean : a 2 + 4 = (a 1 + a 3) / 2

/-- The fourth term of the special geometric sequence is 54 -/
theorem fourth_term_is_54 (seq : SpecialGeometricSequence) : seq.a 4 = 54 := by
  sorry

end fourth_term_is_54_l3850_385067


namespace exactly_eighteen_pairs_l3850_385035

/-- Predicate to check if a pair of natural numbers satisfies the given conditions -/
def satisfies_conditions (a b : ℕ) : Prop :=
  (b ∣ (5 * a - 3)) ∧ (a ∣ (5 * b - 1))

/-- The number of pairs of natural numbers satisfying the conditions -/
def number_of_pairs : ℕ := 18

/-- Theorem stating that there are exactly 18 pairs satisfying the conditions -/
theorem exactly_eighteen_pairs :
  (∃! (s : Finset (ℕ × ℕ)), s.card = number_of_pairs ∧
    ∀ (pair : ℕ × ℕ), pair ∈ s ↔ satisfies_conditions pair.1 pair.2) :=
sorry

end exactly_eighteen_pairs_l3850_385035


namespace school_demographics_l3850_385089

theorem school_demographics (total_students : ℕ) (avg_age_boys avg_age_girls avg_age_school : ℚ) : 
  total_students = 640 →
  avg_age_boys = 12 →
  avg_age_girls = 11 →
  avg_age_school = 47/4 →
  ∃ (num_girls : ℕ), num_girls = 160 ∧ 
    (total_students - num_girls) * avg_age_boys + num_girls * avg_age_girls = total_students * avg_age_school :=
by sorry

end school_demographics_l3850_385089


namespace polynomial_equation_l3850_385000

-- Define polynomials over real numbers
variable (x : ℝ)

-- Define f(x) and h(x) as polynomials
def f (x : ℝ) : ℝ := x^4 + 2*x^3 - x^2 - 4*x + 1
def h (x : ℝ) : ℝ := -x^4 - 2*x^3 + 4*x^2 + 9*x - 5

-- State the theorem
theorem polynomial_equation :
  f x + h x = 3*x^2 + 5*x - 4 := by sorry

end polynomial_equation_l3850_385000


namespace cos_300_degrees_l3850_385033

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end cos_300_degrees_l3850_385033


namespace range_of_a_l3850_385001

-- Define an odd function that is monotonically increasing on [0, +∞)
def is_odd_and_increasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 ≤ x ∧ x < y → f x < f y)

-- Theorem statement
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_odd_incr : is_odd_and_increasing f) 
  (h_ineq : f (2 - a^2) + f a > 0) : 
  -1 < a ∧ a < 2 := by
  sorry

end range_of_a_l3850_385001


namespace additional_cakes_count_l3850_385097

/-- Represents the number of cakes Baker initially made -/
def initial_cakes : ℕ := 62

/-- Represents the number of cakes Baker sold -/
def sold_cakes : ℕ := 144

/-- Represents the number of cakes Baker still has -/
def remaining_cakes : ℕ := 67

/-- Theorem stating the number of additional cakes Baker made -/
theorem additional_cakes_count : 
  ∃ x : ℕ, initial_cakes + x - sold_cakes = remaining_cakes ∧ x = 149 := by
  sorry

end additional_cakes_count_l3850_385097


namespace problem_solution_l3850_385048

theorem problem_solution (x y : ℚ) (h1 : x - y = 8) (h2 : x + 2*y = 10) : x = 26/3 := by
  sorry

end problem_solution_l3850_385048


namespace floor_sqrt_sum_eq_floor_sqrt_sum_l3850_385075

theorem floor_sqrt_sum_eq_floor_sqrt_sum (n : ℕ) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ := by
  sorry

end floor_sqrt_sum_eq_floor_sqrt_sum_l3850_385075


namespace sum_of_differences_l3850_385026

def T : Finset ℕ := Finset.range 9

def M : ℕ := Finset.sum T (λ x => Finset.sum T (λ y => if x > y then 3^x - 3^y else 0))

theorem sum_of_differences (T : Finset ℕ) (M : ℕ) :
  T = Finset.range 9 →
  M = Finset.sum T (λ x => Finset.sum T (λ y => if x > y then 3^x - 3^y else 0)) →
  M = 68896 := by
  sorry

end sum_of_differences_l3850_385026


namespace regular_15gon_symmetry_sum_l3850_385088

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  n_pos : 0 < n

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle of rotational symmetry (in degrees) for a regular polygon -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

/-- Theorem: For a regular 15-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees) is 39 -/
theorem regular_15gon_symmetry_sum :
  ∀ (p : RegularPolygon 15),
    (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 39 := by
  sorry

end regular_15gon_symmetry_sum_l3850_385088


namespace magnitude_of_b_l3850_385065

def vector_a : ℝ × ℝ := (1, -2)
def vector_sum : ℝ × ℝ := (0, 2)

def vector_b : ℝ × ℝ := (vector_sum.1 - vector_a.1, vector_sum.2 - vector_a.2)

theorem magnitude_of_b : Real.sqrt ((vector_b.1)^2 + (vector_b.2)^2) = Real.sqrt 17 := by
  sorry

end magnitude_of_b_l3850_385065


namespace miss_one_out_of_three_l3850_385077

def free_throw_probability : ℝ := 0.9

theorem miss_one_out_of_three (p : ℝ) (hp : p = free_throw_probability) :
  p * p * (1 - p) + p * (1 - p) * p + (1 - p) * p * p = 0.243 := by
  sorry

end miss_one_out_of_three_l3850_385077


namespace fraction_equations_l3850_385060

theorem fraction_equations : 
  (5 / 6 - 2 / 3 = 1 / 6) ∧
  (1 / 2 + 1 / 4 = 3 / 4) ∧
  (9 / 7 - 7 / 21 = 17 / 21) ∧
  (4 / 8 - 1 / 4 = 3 / 8) := by
sorry

end fraction_equations_l3850_385060


namespace cos_squared_pi_fourth_minus_alpha_l3850_385070

theorem cos_squared_pi_fourth_minus_alpha (α : ℝ) 
  (h : Real.sin α - Real.cos α = 4/3) : 
  Real.cos (π/4 - α)^2 = 1/9 := by sorry

end cos_squared_pi_fourth_minus_alpha_l3850_385070


namespace random_co_captains_probability_l3850_385011

def team_sizes : List Nat := [4, 5, 6, 7]
def co_captains_per_team : Nat := 3

def prob_both_co_captains (n : Nat) : Rat :=
  (co_captains_per_team.choose 2) / (n.choose 2)

theorem random_co_captains_probability :
  (1 / team_sizes.length : Rat) *
  (team_sizes.map prob_both_co_captains).sum = 2/7 := by sorry

end random_co_captains_probability_l3850_385011


namespace ellipse_focal_property_l3850_385069

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 4 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Theorem statement
theorem ellipse_focal_property :
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧  -- A and B are on the ellipse
  (∃ (t : ℝ), B = F2 + t • (A - F2)) ∧  -- A, B, and F2 are collinear
  ‖A - B‖ = 8 →  -- Distance between A and B is 8
  ‖A - F1‖ + ‖B - F1‖ = 2 := by sorry

end ellipse_focal_property_l3850_385069


namespace salary_change_percentage_l3850_385045

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let decreased_salary := initial_salary * (1 - 0.4)
  let final_salary := decreased_salary * (1 + 0.4)
  (initial_salary - final_salary) / initial_salary = 0.16 := by
sorry

end salary_change_percentage_l3850_385045


namespace buyer_count_solution_l3850_385056

/-- The number of buyers in a grocery store over three days -/
structure BuyerCount where
  dayBeforeYesterday : ℕ
  yesterday : ℕ
  today : ℕ

/-- Conditions for the buyer count problem -/
def BuyerCountProblem (b : BuyerCount) : Prop :=
  b.today = b.yesterday + 40 ∧
  b.yesterday = b.dayBeforeYesterday / 2 ∧
  b.dayBeforeYesterday + b.yesterday + b.today = 140

theorem buyer_count_solution :
  ∃ b : BuyerCount, BuyerCountProblem b ∧ b.dayBeforeYesterday = 67 := by
  sorry

end buyer_count_solution_l3850_385056


namespace complement_intersection_theorem_l3850_385012

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_theorem :
  (U \ (A ∩ B)) = {1, 4, 5} := by sorry

end complement_intersection_theorem_l3850_385012


namespace flagpole_height_l3850_385022

/-- Given a flagpole that breaks and folds over in half, with its tip 2 feet above the ground
    and the break point 7 feet from the base, prove that its original height was 16 feet. -/
theorem flagpole_height (H : ℝ) : 
  (H - 7 - 2 = 7) →  -- The folded part equals the standing part
  (H = 16) :=
by sorry

end flagpole_height_l3850_385022


namespace original_number_proof_l3850_385083

theorem original_number_proof (n k : ℕ) : 
  (n + k = 3200) → 
  (k ≥ 0) →
  (k < 8) →
  (3200 % 8 = 0) →
  ((n + k) % 8 = 0) →
  (∀ m : ℕ, m < k → (n + m) % 8 ≠ 0) →
  n = 3199 := by
sorry

end original_number_proof_l3850_385083


namespace ellipse_properties_l3850_385086

-- Define the ellipse C
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the focal distance
def focal_distance (c : ℝ) : Prop :=
  c = 2 * Real.sqrt 3

-- Define the intersection points with y-axis
def y_intersections (b : ℝ) : Prop :=
  b = 1

-- Define the standard form of the ellipse
def standard_form (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop :=
  e = Real.sqrt 3 / 2

-- Define the range of x-coordinate for point P
def x_range (x : ℝ) : Prop :=
  24 / 13 < x ∧ x ≤ 2

-- Define the maximum value of |EF|
def max_ef (ef : ℝ) : Prop :=
  ef = 1

theorem ellipse_properties (a b c : ℝ) :
  a > b ∧ b > 0 ∧
  focal_distance c ∧
  y_intersections b →
  (∃ x y, ellipse x y a b ∧ standard_form x y) ∧
  (∃ e, eccentricity e) ∧
  (∃ x, x_range x) ∧
  (∃ ef, max_ef ef) :=
sorry

end ellipse_properties_l3850_385086


namespace sin_315_degrees_l3850_385040

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_315_degrees_l3850_385040


namespace group_size_l3850_385087

/-- An international group consisting of Chinese, Americans, and Australians -/
structure InternationalGroup where
  chinese : ℕ
  americans : ℕ
  australians : ℕ

/-- The total number of people in the group -/
def InternationalGroup.total (group : InternationalGroup) : ℕ :=
  group.chinese + group.americans + group.australians

theorem group_size (group : InternationalGroup) 
  (h1 : group.chinese = 22)
  (h2 : group.americans = 16)
  (h3 : group.australians = 11) :
  group.total = 49 := by
  sorry

#check group_size

end group_size_l3850_385087


namespace intersection_point_l3850_385062

/-- The point of intersection of two lines in a 2D plane. -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- First line: y = 2x -/
def line1 (p : IntersectionPoint) : Prop := p.y = 2 * p.x

/-- Second line: x + y = 3 -/
def line2 (p : IntersectionPoint) : Prop := p.x + p.y = 3

/-- The intersection point of the two lines -/
def intersection : IntersectionPoint := ⟨1, 2⟩

/-- Theorem: The point (1, 2) is the unique intersection of the lines y = 2x and x + y = 3 -/
theorem intersection_point :
  line1 intersection ∧ line2 intersection ∧
  ∀ p : IntersectionPoint, line1 p ∧ line2 p → p = intersection :=
sorry

end intersection_point_l3850_385062


namespace farmer_loss_l3850_385019

/-- Represents the total weight of onions in pounds -/
def total_weight : ℝ := 100

/-- Represents the market price per pound of onions in dollars -/
def market_price : ℝ := 3

/-- Represents the dealer's price per pound for both leaves and whites in dollars -/
def dealer_price : ℝ := 1.5

/-- Theorem stating the farmer's loss -/
theorem farmer_loss : 
  total_weight * market_price - total_weight * dealer_price = 150 := by
  sorry

end farmer_loss_l3850_385019


namespace fourth_derivative_of_f_l3850_385095

open Real

noncomputable def f (x : ℝ) : ℝ := exp (1 - 2*x) * sin (2 + 3*x)

theorem fourth_derivative_of_f (x : ℝ) :
  (deriv^[4] f) x = -119 * exp (1 - 2*x) * sin (2 + 3*x) + 120 * exp (1 - 2*x) * cos (2 + 3*x) :=
by sorry

end fourth_derivative_of_f_l3850_385095


namespace smallest_consecutive_even_number_l3850_385099

theorem smallest_consecutive_even_number (n : ℕ) : 
  (n % 2 = 0) →  -- n is even
  (n + (n + 2) + (n + 4) = 162) →  -- sum of three consecutive even numbers is 162
  n = 52 :=  -- the smallest number is 52
by sorry

end smallest_consecutive_even_number_l3850_385099


namespace abc_value_l3850_385051

theorem abc_value (a b c : ℝ) 
  (sum_eq : a + b + c = 4)
  (sum_prod_eq : b * c + c * a + a * b = 5)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 10) : 
  a * b * c = 2 := by
sorry

end abc_value_l3850_385051


namespace inequality_solution_l3850_385057

theorem inequality_solution (n : Int) :
  n ∈ ({-1, 0, 1, 2, 3} : Set Int) →
  ((-1/2 : ℚ)^n > (-1/5 : ℚ)^n) ↔ (n = -1 ∨ n = 2) :=
by sorry

end inequality_solution_l3850_385057


namespace merchant_discount_percentage_l3850_385096

/-- Calculates the discount percentage for a merchant's pricing strategy -/
theorem merchant_discount_percentage
  (markup_percentage : ℝ)
  (profit_percentage : ℝ)
  (h_markup : markup_percentage = 50)
  (h_profit : profit_percentage = 35)
  : ∃ (discount_percentage : ℝ),
    discount_percentage = 10 ∧
    (1 + markup_percentage / 100) * (1 - discount_percentage / 100) = 1 + profit_percentage / 100 :=
by sorry

end merchant_discount_percentage_l3850_385096


namespace marble_241_is_blue_l3850_385021

/-- Represents the color of a marble -/
inductive MarbleColor
| Blue
| Red
| Green

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 14 with
  | 0 | 1 | 2 | 3 | 4 | 5 => MarbleColor.Blue
  | 6 | 7 | 8 | 9 | 10 => MarbleColor.Red
  | _ => MarbleColor.Green

/-- Theorem: The 241st marble in the sequence is blue -/
theorem marble_241_is_blue : marbleColor 241 = MarbleColor.Blue := by
  sorry

end marble_241_is_blue_l3850_385021


namespace circle_equation_k_range_l3850_385091

/-- Proves that for the equation x^2 + y^2 - 2x + 2k + 3 = 0 to represent a circle,
    k must be in the range (-∞, -1). -/
theorem circle_equation_k_range :
  ∀ (k : ℝ), (∃ (x y : ℝ), x^2 + y^2 - 2*x + 2*k + 3 = 0 ∧ 
    ∃ (h r : ℝ), ∀ (x' y' : ℝ), (x' - h)^2 + (y' - r)^2 = (x - h)^2 + (y - r)^2) 
  ↔ k < -1 := by
  sorry

end circle_equation_k_range_l3850_385091


namespace y_range_for_x_condition_l3850_385023

theorem y_range_for_x_condition (x y : ℝ) : 
  (4 * x + y = 1) → ((-1 < x ∧ x ≤ 2) ↔ (-7 ≤ y ∧ y < -3)) := by
  sorry

end y_range_for_x_condition_l3850_385023


namespace ab_minus_a_minus_b_even_l3850_385046

def S : Set ℕ := {1, 3, 5, 7, 9}

theorem ab_minus_a_minus_b_even (a b : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hab : a ≠ b) :
  Even (a * b - a - b) :=
by
  sorry

end ab_minus_a_minus_b_even_l3850_385046


namespace simplify_expressions_l3850_385039

open Real

theorem simplify_expressions (θ : ℝ) :
  (sqrt (1 - 2 * sin (135 * π / 180) * cos (135 * π / 180))) / 
  (sin (135 * π / 180) + sqrt (1 - sin (135 * π / 180) ^ 2)) = 1 ∧
  (sin (θ - 5 * π) * cos (-π / 2 - θ) * cos (8 * π - θ)) / 
  (sin (θ - 3 * π / 2) * sin (-θ - 4 * π)) = -sin (θ - 5 * π) := by
  sorry

end simplify_expressions_l3850_385039


namespace tangerines_per_day_l3850_385013

theorem tangerines_per_day 
  (initial : ℕ) 
  (days : ℕ) 
  (remaining : ℕ) 
  (h1 : initial > remaining) 
  (h2 : days > 0) : 
  (initial - remaining) / days = (initial - remaining) / days :=
by sorry

end tangerines_per_day_l3850_385013


namespace project_completion_time_l3850_385014

/-- Represents the completion of an engineering project --/
structure Project where
  initialWorkers : ℕ
  initialWorkCompleted : ℚ
  initialDuration : ℕ
  additionalWorkers : ℕ

/-- Calculates the total days required to complete the project --/
def totalDays (p : Project) : ℕ :=
  let totalWorkers := p.initialWorkers + p.additionalWorkers
  let remainingWork := 1 - p.initialWorkCompleted
  let initialWorkRate := p.initialWorkCompleted / p.initialDuration
  let totalWorkRate := initialWorkRate * totalWorkers / p.initialWorkers
  p.initialDuration + (remainingWork / totalWorkRate).ceil.toNat

/-- Theorem stating that for the given project parameters, the total days to complete is 70 --/
theorem project_completion_time (p : Project) 
  (h1 : p.initialWorkers = 6)
  (h2 : p.initialWorkCompleted = 1/3)
  (h3 : p.initialDuration = 35)
  (h4 : p.additionalWorkers = 6) :
  totalDays p = 70 := by
  sorry

#eval totalDays { initialWorkers := 6, initialWorkCompleted := 1/3, initialDuration := 35, additionalWorkers := 6 }

end project_completion_time_l3850_385014


namespace nicks_age_l3850_385004

theorem nicks_age (N : ℝ) : 
  (N + (N + 6)) / 2 + 5 = 21 → N = 13 := by
  sorry

end nicks_age_l3850_385004


namespace absolute_value_squared_l3850_385020

theorem absolute_value_squared (a b : ℝ) : |a| < b → a^2 < b^2 := by
  sorry

end absolute_value_squared_l3850_385020


namespace triangle_to_hexagon_proportionality_l3850_385006

-- Define the original triangle
structure Triangle where
  x : Real
  y : Real
  z : Real
  angle_sum : x + y + z = 180

-- Define the resulting hexagon
structure Hexagon where
  a : Real -- Length of vector a
  b : Real -- Length of vector b
  c : Real -- Length of vector c
  u : Real -- Length of vector u
  v : Real -- Length of vector v
  w : Real -- Length of vector w
  angle1 : Real -- (x-1)°
  angle2 : Real -- 181°
  angle3 : Real
  angle4 : Real
  angle5 : Real
  angle6 : Real
  angle_sum : angle1 + angle2 + angle3 + angle4 + angle5 + angle6 = 720
  non_convex : angle2 > 180

-- Define the transformation from triangle to hexagon
def transform (t : Triangle) (h : Hexagon) : Prop :=
  h.angle1 = t.x - 1 ∧ h.angle2 = 181

-- Theorem to prove
theorem triangle_to_hexagon_proportionality (t : Triangle) (h : Hexagon) 
  (trans : transform t h) : 
  ∃ (k : Real), k > 0 ∧ 
    h.a / t.x = h.b / t.y ∧ 
    h.b / t.y = h.c / t.z ∧ 
    h.c / t.z = k :=
  sorry

end triangle_to_hexagon_proportionality_l3850_385006


namespace symmetric_circle_l3850_385068

/-- Given a circle C1 with equation (x+2)^2+(y-1)^2=5,
    prove that its symmetric circle C2 with respect to the origin (0,0)
    has the equation (x-2)^2+(y+1)^2=5 -/
theorem symmetric_circle (x y : ℝ) :
  (∀ x y, (x + 2)^2 + (y - 1)^2 = 5) →
  (∃ C2 : Set (ℝ × ℝ), C2 = {(x, y) | (x - 2)^2 + (y + 1)^2 = 5} ∧
    ∀ (p : ℝ × ℝ), p ∈ C2 ↔ (-p.1, -p.2) ∈ {(x, y) | (x + 2)^2 + (y - 1)^2 = 5}) :=
by sorry

end symmetric_circle_l3850_385068


namespace log_product_equation_l3850_385028

theorem log_product_equation (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 10) = 4 → x = 10000 := by
  sorry

end log_product_equation_l3850_385028


namespace complex_fraction_equality_l3850_385058

theorem complex_fraction_equality : (3 - I) / (1 - I) = 2 + I := by
  sorry

end complex_fraction_equality_l3850_385058


namespace function_increasing_l3850_385063

theorem function_increasing (f : ℝ → ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁) : 
  StrictMono f := by
  sorry

end function_increasing_l3850_385063


namespace incorrect_number_correction_l3850_385018

theorem incorrect_number_correction (n : ℕ) (incorrect_avg correct_avg incorrect_num : ℚ) 
  (h1 : n = 10)
  (h2 : incorrect_avg = 16)
  (h3 : incorrect_num = 25)
  (h4 : correct_avg = 17) :
  let correct_num := incorrect_num - (n * correct_avg - n * incorrect_avg)
  correct_num = 15 := by sorry

end incorrect_number_correction_l3850_385018


namespace translated_point_sum_zero_l3850_385016

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the translation function
def translate (p : Point) (dx dy : ℝ) : Point :=
  (p.1 + dx, p.2 + dy)

theorem translated_point_sum_zero :
  let A : Point := (-1, 2)
  let B : Point := translate (translate A 1 0) 0 (-2)
  B.1 + B.2 = 0 := by sorry

end translated_point_sum_zero_l3850_385016


namespace unicorn_to_witch_ratio_l3850_385030

/-- Represents the number of votes for each cake type -/
structure CakeVotes where
  unicorn : ℕ
  witch : ℕ
  dragon : ℕ

/-- The conditions of the baking contest voting -/
def baking_contest (votes : CakeVotes) : Prop :=
  votes.dragon = votes.witch + 25 ∧
  votes.witch = 7 ∧
  votes.unicorn + votes.witch + votes.dragon = 60

theorem unicorn_to_witch_ratio (votes : CakeVotes) 
  (h : baking_contest votes) : 
  votes.unicorn / votes.witch = 3 :=
sorry

end unicorn_to_witch_ratio_l3850_385030


namespace oak_grove_total_books_l3850_385007

def public_library_books : ℕ := 1986
def school_library_books : ℕ := 5106

theorem oak_grove_total_books :
  public_library_books + school_library_books = 7092 :=
by sorry

end oak_grove_total_books_l3850_385007


namespace opposite_solutions_value_of_m_l3850_385038

theorem opposite_solutions_value_of_m : ∀ (x y m : ℝ),
  (3 * x + 5 * y = 2) →
  (2 * x + 7 * y = m - 18) →
  (x = -y) →
  m = 23 := by
  sorry

end opposite_solutions_value_of_m_l3850_385038


namespace nine_triangles_perimeter_l3850_385082

theorem nine_triangles_perimeter (large_perimeter : ℝ) (num_small_triangles : ℕ) 
  (h1 : large_perimeter = 120)
  (h2 : num_small_triangles = 9) :
  ∃ (small_perimeter : ℝ), 
    small_perimeter * num_small_triangles = large_perimeter ∧ 
    small_perimeter = 40 := by
  sorry

end nine_triangles_perimeter_l3850_385082


namespace shopping_mall_pricing_l3850_385064

/-- Shopping mall pricing problem -/
theorem shopping_mall_pricing
  (purchase_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_monthly_sales : ℝ)
  (sales_increase_rate : ℝ)
  (target_monthly_profit : ℝ)
  (h1 : purchase_price = 280)
  (h2 : initial_selling_price = 360)
  (h3 : initial_monthly_sales = 60)
  (h4 : sales_increase_rate = 5)
  (h5 : target_monthly_profit = 7200) :
  ∃ (price_reduction : ℝ),
    price_reduction = 60 ∧
    (initial_selling_price - price_reduction - purchase_price) *
    (initial_monthly_sales + sales_increase_rate * price_reduction) =
    target_monthly_profit :=
by sorry

end shopping_mall_pricing_l3850_385064


namespace tangent_line_at_pi_over_2_l3850_385002

noncomputable def f (x : ℝ) := Real.sin x - 2 * Real.cos x

theorem tangent_line_at_pi_over_2 :
  let Q : ℝ × ℝ := (π / 2, 1)
  let m : ℝ := Real.cos (π / 2) + 2 * Real.sin (π / 2)
  let tangent_line (x : ℝ) := m * (x - Q.1) + Q.2
  ∀ x, tangent_line x = 2 * x - π + 1 :=
by sorry

end tangent_line_at_pi_over_2_l3850_385002


namespace geometric_sequence_product_l3850_385037

/-- The constant term in the expansion of (x + 1/x)^4 -/
def constant_term : ℕ := 6

/-- Represents a geometric sequence -/
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n * a m

theorem geometric_sequence_product
  (a : ℕ → ℕ)
  (h_geo : geometric_sequence a)
  (h_a5 : a 5 = constant_term) :
  a 3 * a 7 = 36 := by
sorry

end geometric_sequence_product_l3850_385037


namespace x_intercept_of_line_l3850_385085

/-- The x-intercept of a line is the point where the line crosses the x-axis (i.e., where y = 0) -/
def x_intercept (a b c : ℚ) : ℚ × ℚ :=
  let x := c / a
  (x, 0)

/-- The line equation is in the form ax + by = c -/
def line_equation (a b c : ℚ) (x y : ℚ) : Prop :=
  a * x + b * y = c

theorem x_intercept_of_line :
  x_intercept 5 (-7) 35 = (7, 0) ∧
  line_equation 5 (-7) 35 (x_intercept 5 (-7) 35).1 (x_intercept 5 (-7) 35).2 :=
sorry

end x_intercept_of_line_l3850_385085


namespace sufficient_condition_implies_a_range_l3850_385050

theorem sufficient_condition_implies_a_range :
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) →
  a ∈ Set.Ici 3 := by
  sorry

end sufficient_condition_implies_a_range_l3850_385050


namespace six_to_six_sum_l3850_385034

theorem six_to_six_sum : (6^6 : ℕ) + 6^6 + 6^6 + 6^6 + 6^6 + 6^6 = 6^7 := by
  sorry

end six_to_six_sum_l3850_385034


namespace jonessas_take_home_pay_l3850_385059

/-- Calculates the take-home pay given the total pay and tax rate -/
def takeHomePay (totalPay : ℝ) (taxRate : ℝ) : ℝ :=
  totalPay * (1 - taxRate)

/-- Proves that Jonessa's take-home pay is $450 -/
theorem jonessas_take_home_pay :
  let totalPay : ℝ := 500
  let taxRate : ℝ := 0.1
  takeHomePay totalPay taxRate = 450 := by
sorry

end jonessas_take_home_pay_l3850_385059


namespace prob_hit_135_prob_hit_exactly_3_l3850_385072

-- Define the probability of hitting the target
def hit_probability : ℚ := 3 / 5

-- Define the number of shots
def num_shots : ℕ := 5

-- Theorem for the first part
theorem prob_hit_135 : 
  (hit_probability * (1 - hit_probability) * hit_probability * (1 - hit_probability) * hit_probability) = 108 / 3125 := by
  sorry

-- Theorem for the second part
theorem prob_hit_exactly_3 :
  (Nat.choose num_shots 3 : ℚ) * hit_probability ^ 3 * (1 - hit_probability) ^ 2 = 216 / 625 := by
  sorry

end prob_hit_135_prob_hit_exactly_3_l3850_385072


namespace compound_mass_proof_l3850_385080

/-- The atomic mass of Carbon in g/mol -/
def atomic_mass_C : ℝ := 12.01

/-- The atomic mass of Hydrogen in g/mol -/
def atomic_mass_H : ℝ := 1.008

/-- The atomic mass of Oxygen in g/mol -/
def atomic_mass_O : ℝ := 16.00

/-- The atomic mass of Nitrogen in g/mol -/
def atomic_mass_N : ℝ := 14.01

/-- The atomic mass of Bromine in g/mol -/
def atomic_mass_Br : ℝ := 79.90

/-- The molecular formula of the compound -/
def compound_formula := "C8H10O2NBr2"

/-- The number of moles of the compound -/
def moles_compound : ℝ := 3

/-- The total mass of the compound in grams -/
def total_mass : ℝ := 938.91

/-- Theorem stating that the total mass of 3 moles of C8H10O2NBr2 is 938.91 grams -/
theorem compound_mass_proof :
  moles_compound * (8 * atomic_mass_C + 10 * atomic_mass_H + 2 * atomic_mass_O + atomic_mass_N + 2 * atomic_mass_Br) = total_mass := by
  sorry

end compound_mass_proof_l3850_385080


namespace tank_capacity_proof_l3850_385078

def tank_capacity (oil_bought : ℕ) (oil_in_tank : ℕ) : ℕ :=
  oil_bought + oil_in_tank

theorem tank_capacity_proof (oil_bought : ℕ) (oil_in_tank : ℕ) 
  (h1 : oil_bought = 728) (h2 : oil_in_tank = 24) : 
  tank_capacity oil_bought oil_in_tank = 752 := by
  sorry

#check tank_capacity_proof

end tank_capacity_proof_l3850_385078


namespace fence_coloring_theorem_l3850_385092

/-- A coloring of a fence is valid if any two boards separated by exactly 2, 3, or 5 boards
    are painted in different colors. -/
def is_valid_coloring (coloring : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, (coloring i ≠ coloring (i + 3)) ∧
           (coloring i ≠ coloring (i + 4)) ∧
           (coloring i ≠ coloring (i + 6))

/-- The minimum number of colors required to paint the fence -/
def min_colors : ℕ := 3

theorem fence_coloring_theorem :
  (∃ coloring : ℕ → ℕ, is_valid_coloring coloring ∧ (∀ i : ℕ, coloring i < min_colors)) ∧
  (∀ n : ℕ, n < min_colors → ¬∃ coloring : ℕ → ℕ, is_valid_coloring coloring ∧ (∀ i : ℕ, coloring i < n)) :=
sorry

end fence_coloring_theorem_l3850_385092


namespace car_sale_profit_percentage_l3850_385049

theorem car_sale_profit_percentage (P : ℝ) : 
  let buying_price := 0.80 * P
  let selling_price := 1.16 * P
  ((selling_price - buying_price) / buying_price) * 100 = 45 := by
sorry

end car_sale_profit_percentage_l3850_385049
