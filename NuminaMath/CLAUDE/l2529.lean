import Mathlib

namespace NUMINAMATH_CALUDE_exists_number_not_in_progressions_l2529_252923

/-- Represents a geometric progression of natural numbers -/
structure GeometricProgression where
  first_term : ℕ
  common_ratio : ℕ
  h_positive : common_ratio > 1

/-- Checks if a natural number is in a geometric progression -/
def isInProgression (n : ℕ) (gp : GeometricProgression) : Prop :=
  ∃ k : ℕ, n = gp.first_term * gp.common_ratio ^ k

/-- The main theorem -/
theorem exists_number_not_in_progressions (progressions : Fin 100 → GeometricProgression) :
  ∃ n : ℕ, ∀ i : Fin 100, ¬ isInProgression n (progressions i) := by
  sorry


end NUMINAMATH_CALUDE_exists_number_not_in_progressions_l2529_252923


namespace NUMINAMATH_CALUDE_tree_F_height_l2529_252937

/-- The heights of six trees in a town square -/
structure TreeHeights where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

/-- The conditions for the tree heights problem -/
def tree_height_conditions (h : TreeHeights) : Prop :=
  h.A = 150 ∧
  h.B = 2/3 * h.A ∧
  h.C = 1/2 * h.B ∧
  h.D = h.C + 25 ∧
  h.E = 0.4 * h.A ∧
  h.F = (h.B + h.D) / 2

/-- Theorem stating that Tree F is 87.5 feet tall -/
theorem tree_F_height (h : TreeHeights) 
  (hc : tree_height_conditions h) : h.F = 87.5 := by
  sorry


end NUMINAMATH_CALUDE_tree_F_height_l2529_252937


namespace NUMINAMATH_CALUDE_tangent_lines_through_point_l2529_252951

-- Define the curve
def f (x : ℝ) : ℝ := x^2

-- Define the point P
def P : ℝ × ℝ := (3, 5)

-- Define the two lines
def line1 (x : ℝ) : ℝ := 2*x - 1
def line2 (x : ℝ) : ℝ := 10*x - 25

theorem tangent_lines_through_point :
  ∀ m b : ℝ,
  (∃ x₀ : ℝ, 
    -- The line y = mx + b passes through P(3, 5)
    m * 3 + b = 5 ∧
    -- The line is tangent to the curve at some point (x₀, f(x₀))
    m * x₀ + b = f x₀ ∧
    m = 2 * x₀) →
  ((∀ x, m * x + b = line1 x) ∨ (∀ x, m * x + b = line2 x)) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_through_point_l2529_252951


namespace NUMINAMATH_CALUDE_solution_volume_l2529_252965

/-- Given a solution with 1.5 liters of pure acid and a concentration of 30%,
    prove that the total volume of the solution is 5 liters. -/
theorem solution_volume (volume_acid : ℝ) (concentration : ℝ) :
  volume_acid = 1.5 →
  concentration = 0.30 →
  (volume_acid / concentration) = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_volume_l2529_252965


namespace NUMINAMATH_CALUDE_integral_equals_antiderivative_l2529_252925

open Real

noncomputable def f (x : ℝ) : ℝ := (x^3 - 6*x^2 + 13*x - 8) / (x*(x-2)^3)

noncomputable def F (x : ℝ) : ℝ := log (abs x) - 1 / (2*(x-2)^2)

theorem integral_equals_antiderivative (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 2) :
  deriv F x = f x :=
by sorry

end NUMINAMATH_CALUDE_integral_equals_antiderivative_l2529_252925


namespace NUMINAMATH_CALUDE_basketball_shooting_improvement_l2529_252996

theorem basketball_shooting_improvement (initial_shots : ℕ) (initial_made : ℕ) (additional_shots : ℕ) 
  (new_percentage : ℚ) (h1 : initial_shots = 45) (h2 : initial_made = 18) 
  (h3 : (initial_made : ℚ) / initial_shots = 2/5) (h4 : additional_shots = 15) 
  (h5 : new_percentage = 9/20) : 
  ∃ (additional_made : ℕ), 
    (initial_made + additional_made : ℚ) / (initial_shots + additional_shots) = new_percentage ∧ 
    additional_made = 9 := by
sorry

end NUMINAMATH_CALUDE_basketball_shooting_improvement_l2529_252996


namespace NUMINAMATH_CALUDE_log_positive_iff_greater_than_one_l2529_252985

theorem log_positive_iff_greater_than_one (x : ℝ) : x > 1 ↔ Real.log x > 0 := by
  sorry

end NUMINAMATH_CALUDE_log_positive_iff_greater_than_one_l2529_252985


namespace NUMINAMATH_CALUDE_root_sum_fraction_l2529_252913

theorem root_sum_fraction (a b c : ℝ) : 
  a^3 - 8*a^2 + 7*a - 3 = 0 → 
  b^3 - 8*b^2 + 7*b - 3 = 0 → 
  c^3 - 8*c^2 + 7*c - 3 = 0 → 
  a / (b*c + 1) + b / (a*c + 1) + c / (a*b + 1) = 17/2 := by
sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l2529_252913


namespace NUMINAMATH_CALUDE_perpendicular_planes_l2529_252988

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (is_perpendicular : Line → Plane → Prop)
variable (is_subset : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- Define m, n as different lines
variable (m n : Line)
variable (h_diff_lines : m ≠ n)

-- Define α, β as different planes
variable (α β : Plane)
variable (h_diff_planes : α ≠ β)

-- State the theorem
theorem perpendicular_planes 
  (h1 : is_perpendicular m α) 
  (h2 : is_subset m β) : 
  planes_perpendicular α β := by sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l2529_252988


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l2529_252939

theorem gcd_digits_bound (a b : ℕ) : 
  (1000000 ≤ a ∧ a < 10000000) →
  (1000000 ≤ b ∧ b < 10000000) →
  (10000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 100000000000) →
  Nat.gcd a b < 10000 := by
sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l2529_252939


namespace NUMINAMATH_CALUDE_point_on_line_implies_a_equals_negative_eight_l2529_252998

/-- A point (a, 0) lies on the line y = x + 8 -/
def point_on_line (a : ℝ) : Prop :=
  0 = a + 8

/-- Theorem: If (a, 0) lies on the line y = x + 8, then a = -8 -/
theorem point_on_line_implies_a_equals_negative_eight (a : ℝ) :
  point_on_line a → a = -8 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_implies_a_equals_negative_eight_l2529_252998


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l2529_252940

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 3

-- State the theorem
theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 15 / 5 ∧
  (∀ x : ℝ, g x = 0 → x ≤ r) ∧
  g r = 0 := by
  sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l2529_252940


namespace NUMINAMATH_CALUDE_number_relationship_l2529_252904

-- Define the numbers in their respective bases
def a : ℕ := 33
def b : ℕ := 5 * 6 + 2  -- 52 in base 6
def c : ℕ := 16 + 8 + 4 + 2 + 1  -- 11111 in base 2

-- Theorem statement
theorem number_relationship : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_l2529_252904


namespace NUMINAMATH_CALUDE_jessie_points_l2529_252993

def total_points : ℕ := 311
def some_players_points : ℕ := 188
def num_equal_scorers : ℕ := 3

theorem jessie_points : 
  (total_points - some_players_points) / num_equal_scorers = 41 := by
  sorry

end NUMINAMATH_CALUDE_jessie_points_l2529_252993


namespace NUMINAMATH_CALUDE_largest_number_with_seven_front_l2529_252942

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (n / 100 = 7) ∧ (n % 100 / 10 ≠ 7) ∧ (n % 10 ≠ 7)

theorem largest_number_with_seven_front :
  ∀ n : ℕ, is_valid_number n → n ≤ 743 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_seven_front_l2529_252942


namespace NUMINAMATH_CALUDE_lucia_dance_class_cost_l2529_252956

/-- Represents the cost calculation for Lucia's dance classes over a six-week period. -/
def dance_class_cost (hip_hop_cost ballet_cost jazz_cost salsa_cost contemporary_cost : ℚ)
  (hip_hop_freq ballet_freq jazz_freq salsa_freq contemporary_freq : ℚ)
  (extra_salsa_cost : ℚ) : ℚ :=
  hip_hop_cost * hip_hop_freq * 6 +
  ballet_cost * ballet_freq * 6 +
  jazz_cost * jazz_freq * 6 +
  salsa_cost * (6 / salsa_freq) +
  contemporary_cost * (6 / contemporary_freq) +
  extra_salsa_cost

/-- Proves that the total cost of Lucia's dance classes for a six-week period is $465.50. -/
theorem lucia_dance_class_cost :
  dance_class_cost 10.50 12.25 8.75 15 10 3 2 1 2 3 12 = 465.50 := by
  sorry

end NUMINAMATH_CALUDE_lucia_dance_class_cost_l2529_252956


namespace NUMINAMATH_CALUDE_maya_books_last_week_l2529_252957

/-- The number of pages in each book Maya reads. -/
def pages_per_book : ℕ := 300

/-- The total number of pages Maya read over two weeks. -/
def total_pages : ℕ := 4500

/-- The ratio of pages read this week compared to last week. -/
def week_ratio : ℕ := 2

/-- The number of books Maya read last week. -/
def books_last_week : ℕ := 5

theorem maya_books_last_week :
  books_last_week * pages_per_book * (week_ratio + 1) = total_pages :=
sorry

end NUMINAMATH_CALUDE_maya_books_last_week_l2529_252957


namespace NUMINAMATH_CALUDE_equation_has_root_in_interval_l2529_252991

theorem equation_has_root_in_interval (t : ℝ) (h : t ∈ ({6, 7, 8, 9} : Set ℝ)) :
  ∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^4 - t*x + 1/t = 0 := by
  sorry

#check equation_has_root_in_interval

end NUMINAMATH_CALUDE_equation_has_root_in_interval_l2529_252991


namespace NUMINAMATH_CALUDE_no_solution_exists_l2529_252902

/-- Sum of digits of a natural number in decimal notation -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There does not exist a natural number n such that n * s(n) = 20222022 -/
theorem no_solution_exists : ¬ ∃ n : ℕ, n * sum_of_digits n = 20222022 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2529_252902


namespace NUMINAMATH_CALUDE_zoo_revenue_example_l2529_252915

/-- Calculates the total money made by a zoo over two days given the number of children and adults each day and the ticket prices. -/
def zoo_revenue (child_price adult_price : ℕ) (mon_children mon_adults tues_children tues_adults : ℕ) : ℕ :=
  (mon_children * child_price + mon_adults * adult_price) +
  (tues_children * child_price + tues_adults * adult_price)

/-- Theorem stating that the zoo made $61 in total for both days. -/
theorem zoo_revenue_example : zoo_revenue 3 4 7 5 4 2 = 61 := by
  sorry

end NUMINAMATH_CALUDE_zoo_revenue_example_l2529_252915


namespace NUMINAMATH_CALUDE_tomatoes_rotted_l2529_252962

def initial_shipment : ℕ := 1000
def saturday_sales : ℕ := 300
def monday_shipment : ℕ := 2 * initial_shipment
def tuesday_ready : ℕ := 2500

theorem tomatoes_rotted (rotted : ℕ) : 
  rotted = initial_shipment - saturday_sales + monday_shipment - tuesday_ready := by sorry

end NUMINAMATH_CALUDE_tomatoes_rotted_l2529_252962


namespace NUMINAMATH_CALUDE_equation_equivalence_l2529_252970

theorem equation_equivalence (a b c : ℝ) (h : a + c = 2 * b) : 
  a^2 + 8 * b * c = (2 * b + c)^2 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2529_252970


namespace NUMINAMATH_CALUDE_a_3_range_l2529_252958

def is_convex_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (a n + a (n + 2)) / 2 ≤ a (n + 1)

def b (n : ℕ) : ℝ := n^2 - 6*n + 10

theorem a_3_range (a : ℕ → ℝ) :
  is_convex_sequence a →
  a 1 = 1 →
  a 10 = 28 →
  (∀ n : ℕ, 1 ≤ n → n < 10 → |a n - b n| ≤ 20) →
  7 ≤ a 3 ∧ a 3 ≤ 19 := by
sorry

end NUMINAMATH_CALUDE_a_3_range_l2529_252958


namespace NUMINAMATH_CALUDE_arithmetic_progression_perfect_squares_l2529_252948

theorem arithmetic_progression_perfect_squares :
  ∃ (a b c : ℤ),
    b - a = c - b ∧
    ∃ (x y z : ℤ),
      a + b = x^2 ∧
      a + c = y^2 ∧
      b + c = z^2 ∧
      a = 482 ∧
      b = 3362 ∧
      c = 6242 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_perfect_squares_l2529_252948


namespace NUMINAMATH_CALUDE_right_triangle_sides_l2529_252928

/-- A right triangle with perimeter 30 and height to hypotenuse 6 has sides 10, 7.5, and 12.5 -/
theorem right_triangle_sides (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) : 
  a^2 + b^2 = c^2 →  -- right triangle condition
  a + b + c = 30 →   -- perimeter condition
  a * b = 6 * c →    -- height to hypotenuse condition
  ((a = 10 ∧ b = 7.5 ∧ c = 12.5) ∨ (a = 7.5 ∧ b = 10 ∧ c = 12.5)) := by
  sorry

#check right_triangle_sides

end NUMINAMATH_CALUDE_right_triangle_sides_l2529_252928


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l2529_252916

theorem complex_fraction_equals_i :
  let i : ℂ := Complex.I
  (1 + i) / (1 - i) = i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l2529_252916


namespace NUMINAMATH_CALUDE_alissa_presents_l2529_252982

theorem alissa_presents (ethan_presents : ℕ) (alissa_additional : ℕ) :
  ethan_presents = 31 →
  alissa_additional = 22 →
  ethan_presents + alissa_additional = 53 :=
by sorry

end NUMINAMATH_CALUDE_alissa_presents_l2529_252982


namespace NUMINAMATH_CALUDE_siblings_combined_weight_l2529_252941

/-- Given Antonio's weight and the difference between his and his sister's weight,
    calculate their combined weight. -/
theorem siblings_combined_weight (antonio_weight sister_weight_diff : ℕ) :
  antonio_weight = 50 →
  sister_weight_diff = 12 →
  antonio_weight + (antonio_weight - sister_weight_diff) = 88 := by
  sorry

#check siblings_combined_weight

end NUMINAMATH_CALUDE_siblings_combined_weight_l2529_252941


namespace NUMINAMATH_CALUDE_parallelogram_smaller_angle_l2529_252921

theorem parallelogram_smaller_angle (smaller_angle larger_angle : ℝ) : 
  larger_angle = smaller_angle + 120 →
  smaller_angle + larger_angle + smaller_angle + larger_angle = 360 →
  smaller_angle = 30 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_smaller_angle_l2529_252921


namespace NUMINAMATH_CALUDE_banana_bread_theorem_l2529_252938

/-- The number of bananas needed to make one loaf of banana bread -/
def bananas_per_loaf : ℕ := 4

/-- The number of loaves made on Monday -/
def monday_loaves : ℕ := 3

/-- The number of loaves made on Tuesday -/
def tuesday_loaves : ℕ := 2 * monday_loaves

/-- The total number of bananas used over two days -/
def total_bananas : ℕ := bananas_per_loaf * (monday_loaves + tuesday_loaves)

theorem banana_bread_theorem : total_bananas = 36 := by
  sorry

end NUMINAMATH_CALUDE_banana_bread_theorem_l2529_252938


namespace NUMINAMATH_CALUDE_inverse_f_69_l2529_252922

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 6

-- State the theorem
theorem inverse_f_69 : f⁻¹ 69 = (21 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_inverse_f_69_l2529_252922


namespace NUMINAMATH_CALUDE_smallest_prime_angle_in_special_right_triangle_l2529_252931

-- Define a structure for a right triangle with two acute angles
structure RightTriangle where
  angle1 : ℝ
  angle2 : ℝ
  sum_less_than_45 : angle1 + angle2 < 45
  angles_positive : 0 < angle1 ∧ 0 < angle2

-- Define a predicate for primality (approximate for real numbers)
def is_prime_approx (x : ℝ) : Prop := sorry

-- Define the theorem
theorem smallest_prime_angle_in_special_right_triangle :
  ∀ (t : RightTriangle),
    is_prime_approx t.angle1 →
    is_prime_approx t.angle2 →
    ∃ (smaller_angle : ℝ),
      smaller_angle = min t.angle1 t.angle2 ∧
      smaller_angle ≥ 2.3 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_angle_in_special_right_triangle_l2529_252931


namespace NUMINAMATH_CALUDE_congruence_solution_l2529_252977

theorem congruence_solution (n : ℤ) : 13 * n ≡ 8 [ZMOD 47] → n ≡ 29 [ZMOD 47] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2529_252977


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2529_252961

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 1) :
  let e := Real.sqrt (1 + 1 / a^2)
  1 < e ∧ e < Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2529_252961


namespace NUMINAMATH_CALUDE_haley_marbles_l2529_252999

/-- The number of boys in Haley's class who love to play marbles -/
def num_boys : ℕ := 11

/-- The number of marbles Haley gives to each boy -/
def marbles_per_boy : ℕ := 9

/-- Theorem stating the total number of marbles Haley had -/
theorem haley_marbles : num_boys * marbles_per_boy = 99 := by
  sorry

end NUMINAMATH_CALUDE_haley_marbles_l2529_252999


namespace NUMINAMATH_CALUDE_zero_function_satisfies_equation_l2529_252920

theorem zero_function_satisfies_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f x - f y) → (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_zero_function_satisfies_equation_l2529_252920


namespace NUMINAMATH_CALUDE_mary_next_birthday_age_l2529_252955

theorem mary_next_birthday_age 
  (mary_age sally_age danielle_age : ℝ)
  (h1 : mary_age = 1.25 * sally_age)
  (h2 : sally_age = 0.7 * danielle_age)
  (h3 : mary_age + sally_age + danielle_age = 36) :
  ⌊mary_age⌋ + 1 = 13 :=
by sorry

end NUMINAMATH_CALUDE_mary_next_birthday_age_l2529_252955


namespace NUMINAMATH_CALUDE_series_solution_l2529_252959

/-- The sum of the infinite series 1 + 3x + 6x^2 + ... -/
noncomputable def S (x : ℝ) : ℝ := 1 / (1 - x)^3

/-- Theorem: If S(x) = 4, then x = 1 - 1/∛4 -/
theorem series_solution (x : ℝ) (h : S x = 4) : x = 1 - 1 / Real.rpow 4 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_series_solution_l2529_252959


namespace NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l2529_252903

/-- Given two curves y = x^2 - 1 and y = 1 + x^3 with perpendicular tangents at x = x_0,
    prove that x_0 = -1 / ∛6 -/
theorem perpendicular_tangents_intersection (x_0 : ℝ) :
  (2 * x_0) * (3 * x_0^2) = -1 →
  x_0 = -1 / Real.rpow 6 (1/3) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_intersection_l2529_252903


namespace NUMINAMATH_CALUDE_ratio_problem_l2529_252912

theorem ratio_problem (x y : ℝ) (h : (8 * x - 5 * y) / (11 * x - 3 * y) = 2 / 3) : 
  x / y = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2529_252912


namespace NUMINAMATH_CALUDE_circle_division_theorem_l2529_252936

/-- The number of regions a circle is divided into when n points are chosen on its circumference
    and all possible chords are drawn, assuming no three chords are concurrent. -/
def circle_regions (n : ℕ) : ℚ :=
  (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24

/-- Theorem stating that the number of regions a circle is divided into when n points are chosen
    on its circumference and all possible chords are drawn (assuming no three chords are concurrent)
    is equal to (n^4 - 6n^3 + 23n^2 - 18n + 24) / 24. -/
theorem circle_division_theorem (n : ℕ) :
  let points_on_circle := n
  let all_chords_drawn := True
  let no_three_chords_concurrent := True
  circle_regions n = (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24 := by
  sorry

#eval circle_regions 5  -- Example evaluation

end NUMINAMATH_CALUDE_circle_division_theorem_l2529_252936


namespace NUMINAMATH_CALUDE_marianne_age_always_12_more_than_bella_l2529_252935

/-- Represents the age difference between Marianne and Bella -/
def age_difference : ℕ := 12

/-- Marianne's age when Bella is 8 years old -/
def marianne_age_when_bella_8 : ℕ := 20

/-- Bella's age when Marianne is 30 years old -/
def bella_age_when_marianne_30 : ℕ := 18

/-- Marianne's age as a function of Bella's age -/
def marianne_age (bella_age : ℕ) : ℕ := bella_age + age_difference

theorem marianne_age_always_12_more_than_bella :
  ∀ (bella_age : ℕ),
    marianne_age bella_age = bella_age + age_difference :=
by
  sorry

#check marianne_age_always_12_more_than_bella

end NUMINAMATH_CALUDE_marianne_age_always_12_more_than_bella_l2529_252935


namespace NUMINAMATH_CALUDE_remainder_777_pow_777_mod_13_l2529_252978

theorem remainder_777_pow_777_mod_13 : 777^777 % 13 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_777_pow_777_mod_13_l2529_252978


namespace NUMINAMATH_CALUDE_biology_enrollment_percentage_l2529_252905

theorem biology_enrollment_percentage (total_students : ℕ) (not_enrolled : ℕ) : 
  total_students = 880 → not_enrolled = 462 → 
  (((total_students - not_enrolled : ℚ) / total_students) * 100 : ℚ) = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_biology_enrollment_percentage_l2529_252905


namespace NUMINAMATH_CALUDE_brothers_money_l2529_252986

theorem brothers_money (a₁ a₂ a₃ a₄ : ℚ) :
  a₁ + a₂ + a₃ + a₄ = 48 ∧
  a₁ + 3 = a₂ - 3 ∧
  a₁ + 3 = 3 * a₃ ∧
  a₁ + 3 = a₄ / 3 →
  a₁ = 6 ∧ a₂ = 12 ∧ a₃ = 3 ∧ a₄ = 27 := by
sorry

end NUMINAMATH_CALUDE_brothers_money_l2529_252986


namespace NUMINAMATH_CALUDE_opposite_face_is_t_l2529_252908

-- Define the faces of the cube
inductive Face : Type
  | p | q | r | s | t | u

-- Define the cube structure
structure Cube where
  top : Face
  right : Face
  left : Face
  bottom : Face
  front : Face
  back : Face

-- Define the conditions of the problem
def problem_cube : Cube :=
  { top := Face.p
  , right := Face.q
  , left := Face.r
  , bottom := Face.t  -- We'll prove this is correct
  , front := Face.s   -- Arbitrary assignment for remaining faces
  , back := Face.u }  -- Arbitrary assignment for remaining faces

-- Theorem statement
theorem opposite_face_is_t (c : Cube) :
  c.top = Face.p → c.right = Face.q → c.left = Face.r → c.bottom = Face.t :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_face_is_t_l2529_252908


namespace NUMINAMATH_CALUDE_find_extreme_stone_l2529_252932

/-- A stone with a specific weight -/
structure Stone where
  weight : ℝ

/-- A two-tiered balance scale that can compare two pairs of stones -/
def TwoTieredScale (stones : Finset Stone) : Prop :=
  ∀ (a b c d : Stone), a ∈ stones → b ∈ stones → c ∈ stones → d ∈ stones →
    ((a.weight + b.weight) > (c.weight + d.weight)) ∨
    ((a.weight + b.weight) < (c.weight + d.weight))

/-- The theorem stating that we can find either the heaviest or the lightest stone -/
theorem find_extreme_stone
  (stones : Finset Stone)
  (h_count : stones.card = 10)
  (h_distinct_weights : ∀ (a b : Stone), a ∈ stones → b ∈ stones → a ≠ b → a.weight ≠ b.weight)
  (h_distinct_sums : ∀ (a b c d : Stone), a ∈ stones → b ∈ stones → c ∈ stones → d ∈ stones →
    (a ≠ b ∧ c ≠ d ∧ (a, b) ≠ (c, d) ∧ (a, b) ≠ (d, c)) →
    a.weight + b.weight ≠ c.weight + d.weight)
  (h_scale : TwoTieredScale stones) :
  (∃ (s : Stone), s ∈ stones ∧ ∀ (t : Stone), t ∈ stones → s.weight ≥ t.weight) ∨
  (∃ (s : Stone), s ∈ stones ∧ ∀ (t : Stone), t ∈ stones → s.weight ≤ t.weight) :=
sorry

end NUMINAMATH_CALUDE_find_extreme_stone_l2529_252932


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2529_252945

theorem quadratic_inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ (-9 < m ∧ m < -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2529_252945


namespace NUMINAMATH_CALUDE_max_value_inequality_l2529_252954

theorem max_value_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + 2*c = 2) :
  (a*b)/(a+b) + (a*c)/(a+c) + (b*c)/(b+c) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2529_252954


namespace NUMINAMATH_CALUDE_gardener_work_days_l2529_252943

/-- Calculates the number of days a gardener works on a rose bush replanting project. -/
theorem gardener_work_days
  (num_rose_bushes : ℕ)
  (cost_per_rose_bush : ℚ)
  (gardener_hourly_wage : ℚ)
  (gardener_hours_per_day : ℕ)
  (soil_cubic_feet : ℕ)
  (soil_cost_per_cubic_foot : ℚ)
  (total_project_cost : ℚ)
  (h1 : num_rose_bushes = 20)
  (h2 : cost_per_rose_bush = 150)
  (h3 : gardener_hourly_wage = 30)
  (h4 : gardener_hours_per_day = 5)
  (h5 : soil_cubic_feet = 100)
  (h6 : soil_cost_per_cubic_foot = 5)
  (h7 : total_project_cost = 4100) :
  (total_project_cost - (num_rose_bushes * cost_per_rose_bush + soil_cubic_feet * soil_cost_per_cubic_foot)) / (gardener_hourly_wage * gardener_hours_per_day) = 4 := by
  sorry


end NUMINAMATH_CALUDE_gardener_work_days_l2529_252943


namespace NUMINAMATH_CALUDE_second_player_wins_l2529_252968

/-- Represents the state of the game board as a list of integers -/
def GameBoard := List Nat

/-- The initial game board with 2022 ones -/
def initialBoard : GameBoard := List.replicate 2022 1

/-- A player in the game -/
inductive Player
| First
| Second

/-- The result of a game -/
inductive GameResult
| FirstWin
| SecondWin
| Draw

/-- A move in the game, represented by the index of the first number to be replaced -/
def Move := Nat

/-- Apply a move to the game board -/
def applyMove (board : GameBoard) (move : Move) : GameBoard :=
  sorry

/-- Check if a player has won -/
def hasWon (board : GameBoard) : Bool :=
  sorry

/-- Check if the game is a draw -/
def isDraw (board : GameBoard) : Bool :=
  sorry

/-- A strategy for a player -/
def Strategy := GameBoard → Move

/-- The second player's strategy -/
def secondPlayerStrategy : Strategy :=
  sorry

/-- The game result when both players play optimally -/
def gameResult (firstStrategy secondStrategy : Strategy) : GameResult :=
  sorry

/-- Theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∃ (secondStrategy : Strategy),
    ∀ (firstStrategy : Strategy),
      gameResult firstStrategy secondStrategy = GameResult.SecondWin :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l2529_252968


namespace NUMINAMATH_CALUDE_quadratic_equation_from_roots_l2529_252918

theorem quadratic_equation_from_roots (x₁ x₂ : ℝ) (hx₁ : x₁ = 3) (hx₂ : x₂ = -4) :
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ (x - x₁) * (x - x₂) = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_roots_l2529_252918


namespace NUMINAMATH_CALUDE_rain_gear_needed_l2529_252930

structure WeatherForecast where
  rain_probability : ℝ
  rain_probability_valid : 0 ≤ rain_probability ∧ rain_probability ≤ 1

def high_possibility (p : ℝ) : Prop := p > 0.5

theorem rain_gear_needed (forecast : WeatherForecast) 
  (h : forecast.rain_probability = 0.95) : 
  high_possibility forecast.rain_probability :=
by
  sorry

#check rain_gear_needed

end NUMINAMATH_CALUDE_rain_gear_needed_l2529_252930


namespace NUMINAMATH_CALUDE_fraction_in_sections_sum_l2529_252997

/-- The fraction of students in the band who are in either the trumpet section or the trombone section -/
def fraction_in_sections (trumpet_fraction trombone_fraction : ℝ) : ℝ :=
  trumpet_fraction + trombone_fraction

/-- Theorem: Given that 0.5 of the students are in the trumpet section and 0.12 of the students are in the trombone section,
    the fraction of students in either the trumpet section or the trombone section is 0.62 -/
theorem fraction_in_sections_sum :
  fraction_in_sections 0.5 0.12 = 0.62 := by
  sorry

end NUMINAMATH_CALUDE_fraction_in_sections_sum_l2529_252997


namespace NUMINAMATH_CALUDE_vector_properties_l2529_252987

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (a b : V)

-- Define the conditions
def non_collinear (a b : V) : Prop := ¬ ∃ (k : ℝ), a = k • b
def same_starting_point (a b : V) : Prop := True  -- This is implicitly assumed in the vector space
def equal_magnitude (a b : V) : Prop := ‖a‖ = ‖b‖
def angle_60_degrees (a b : V) : Prop := inner a b = (1/2 : ℝ) * ‖a‖ * ‖b‖

-- Define the theorem
theorem vector_properties
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : non_collinear V a b)
  (h4 : same_starting_point V a b)
  (h5 : equal_magnitude V a b)
  (h6 : angle_60_degrees V a b) :
  (∃ (k : ℝ), k • ((1/2 : ℝ) • b - a) = (1/3 : ℝ) • b - (2/3 : ℝ) • a) ∧
  (∀ (t : ℝ), ‖a - (1/2 : ℝ) • b‖ ≤ ‖a - t • b‖) :=
sorry

end NUMINAMATH_CALUDE_vector_properties_l2529_252987


namespace NUMINAMATH_CALUDE_order_of_even_increasing_function_l2529_252917

-- Define an even function f on ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define an increasing function on [0, +∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Main theorem
theorem order_of_even_increasing_function (f : ℝ → ℝ) 
  (h_even : even_function f) (h_incr : increasing_on_nonneg f) :
  f (-2) < f 3 ∧ f 3 < f (-π) :=
by
  sorry


end NUMINAMATH_CALUDE_order_of_even_increasing_function_l2529_252917


namespace NUMINAMATH_CALUDE_prime_square_plus_13_divisibility_l2529_252976

theorem prime_square_plus_13_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ k : ℕ, p^2 + 13 = 2*k + 2 := by
sorry

end NUMINAMATH_CALUDE_prime_square_plus_13_divisibility_l2529_252976


namespace NUMINAMATH_CALUDE_elina_mean_is_92_5_l2529_252900

/-- The set of all test scores -/
def all_scores : Finset ℕ := {78, 85, 88, 91, 92, 95, 96, 99, 101, 103}

/-- The number of Jason's scores -/
def jason_count : ℕ := 6

/-- The number of Elina's scores -/
def elina_count : ℕ := 4

/-- Jason's mean score -/
def jason_mean : ℚ := 93

/-- The sum of all scores -/
def total_sum : ℕ := Finset.sum all_scores id

theorem elina_mean_is_92_5 :
  (total_sum - jason_count * jason_mean) / elina_count = 92.5 := by
  sorry

end NUMINAMATH_CALUDE_elina_mean_is_92_5_l2529_252900


namespace NUMINAMATH_CALUDE_smallest_x_cos_equality_l2529_252907

theorem smallest_x_cos_equality : ∃ x : ℝ, 
  x > 30 ∧ 
  Real.cos (x * Real.pi / 180) = Real.cos ((2 * x + 10) * Real.pi / 180) ∧
  x < 117 ∧
  ∀ y : ℝ, y > 30 ∧ 
    Real.cos (y * Real.pi / 180) = Real.cos ((2 * y + 10) * Real.pi / 180) → 
    y ≥ x ∧
  ⌈x⌉ = 117 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_cos_equality_l2529_252907


namespace NUMINAMATH_CALUDE_discount_order_difference_discount_order_difference_proof_l2529_252969

/-- Proves that the difference between applying 25% off then $5 off, and applying $5 off then 25% off, on a $30 item, is 125 cents. -/
theorem discount_order_difference : ℝ → Prop :=
  fun original_price : ℝ =>
    let first_discount : ℝ := 5
    let second_discount_rate : ℝ := 0.25
    let price_25_then_5 := (original_price * (1 - second_discount_rate)) - first_discount
    let price_5_then_25 := (original_price - first_discount) * (1 - second_discount_rate)
    original_price = 30 →
    (price_25_then_5 - price_5_then_25) * 100 = 125

/-- The proof of the theorem. -/
theorem discount_order_difference_proof : discount_order_difference 30 := by
  sorry

end NUMINAMATH_CALUDE_discount_order_difference_discount_order_difference_proof_l2529_252969


namespace NUMINAMATH_CALUDE_smallest_number_with_55_divisors_l2529_252983

/-- The number of divisors of n = p₁^k₁ * p₂^k₂ * ... * pₘ^kₘ is (k₁+1)(k₂+1)...(kₘ+1) -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n has exactly 55 divisors -/
def has_55_divisors (n : ℕ) : Prop := num_divisors n = 55

theorem smallest_number_with_55_divisors :
  ∃ (n : ℕ), has_55_divisors n ∧ ∀ (m : ℕ), has_55_divisors m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_55_divisors_l2529_252983


namespace NUMINAMATH_CALUDE_binomial_10_0_l2529_252909

theorem binomial_10_0 : (10 : ℕ).choose 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_0_l2529_252909


namespace NUMINAMATH_CALUDE_percent_value_in_quarters_l2529_252989

theorem percent_value_in_quarters : 
  let num_dimes : ℕ := 40
  let num_quarters : ℕ := 30
  let num_nickels : ℕ := 10
  let value_dime : ℕ := 10
  let value_quarter : ℕ := 25
  let value_nickel : ℕ := 5
  let total_value : ℕ := num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel
  let quarter_value : ℕ := num_quarters * value_quarter
  (quarter_value : ℚ) / (total_value : ℚ) * 100 = 62.5
  := by sorry

end NUMINAMATH_CALUDE_percent_value_in_quarters_l2529_252989


namespace NUMINAMATH_CALUDE_circle_graph_proportion_l2529_252964

theorem circle_graph_proportion (total_degrees : ℝ) (sector_degrees : ℝ) 
  (h1 : total_degrees = 360) 
  (h2 : sector_degrees = 180) : 
  sector_degrees / total_degrees = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_circle_graph_proportion_l2529_252964


namespace NUMINAMATH_CALUDE_expand_binomials_l2529_252995

theorem expand_binomials (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l2529_252995


namespace NUMINAMATH_CALUDE_expression_value_l2529_252906

-- Define opposite numbers
def opposite (m n : ℝ) : Prop := m + n = 0

-- Define reciprocal numbers
def reciprocal (p q : ℝ) : Prop := p * q = 1

-- Theorem statement
theorem expression_value 
  (m n p q : ℝ) 
  (h1 : opposite m n) 
  (h2 : m ≠ n) 
  (h3 : reciprocal p q) : 
  (m + n) / m + 2 * p * q - m / n = 3 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2529_252906


namespace NUMINAMATH_CALUDE_mistaken_operation_l2529_252901

/-- Given an operation O on real numbers that results in a 99% error
    compared to multiplying by 10, prove that O(x) = 0.1 * x for all x. -/
theorem mistaken_operation (O : ℝ → ℝ) (h : ∀ x : ℝ, O x = 0.01 * (10 * x)) :
  ∀ x : ℝ, O x = 0.1 * x := by
sorry

end NUMINAMATH_CALUDE_mistaken_operation_l2529_252901


namespace NUMINAMATH_CALUDE_complex_power_six_l2529_252914

theorem complex_power_six (i : ℂ) (h : i^2 = -1) : (1 + i)^6 = -8*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_six_l2529_252914


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2529_252910

/-- Given a circle C with equation 2x^2 + 3y - 25 = -y^2 + 12x + 4,
    where (a,b) is the center and r is the radius,
    prove that a + b + r = 6.744 -/
theorem circle_center_radius_sum (x y a b r : ℝ) : 
  (2 * x^2 + 3 * y - 25 = -y^2 + 12 * x + 4) →
  ((x - a)^2 + (y - b)^2 = r^2) →
  (a + b + r = 6.744) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2529_252910


namespace NUMINAMATH_CALUDE_inequality_proof_l2529_252911

theorem inequality_proof (x : ℝ) (h : x ≥ 4) :
  Real.sqrt (x - 3) - Real.sqrt (x - 1) > Real.sqrt (x - 4) - Real.sqrt (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2529_252911


namespace NUMINAMATH_CALUDE_min_value_xy_min_value_xy_achieved_l2529_252973

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y + 6 = x*y) :
  x*y ≥ 18 := by
  sorry

theorem min_value_xy_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2*x + y + 6 = x*y ∧ x*y < 18 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_min_value_xy_achieved_l2529_252973


namespace NUMINAMATH_CALUDE_jeopardy_episode_length_l2529_252967

/-- The length of one episode of Jeopardy in minutes -/
def jeopardy_length : ℝ := sorry

/-- The length of one episode of Wheel of Fortune in minutes -/
def wheel_of_fortune_length : ℝ := sorry

/-- The total number of episodes James watched -/
def total_episodes : ℕ := sorry

/-- The total time James spent watching TV in minutes -/
def total_watch_time : ℝ := sorry

theorem jeopardy_episode_length :
  jeopardy_length = 20 ∧
  wheel_of_fortune_length = 2 * jeopardy_length ∧
  total_episodes = 4 ∧
  total_watch_time = 120 ∧
  total_watch_time = 2 * jeopardy_length + 2 * wheel_of_fortune_length :=
by sorry

end NUMINAMATH_CALUDE_jeopardy_episode_length_l2529_252967


namespace NUMINAMATH_CALUDE_bakery_pie_division_l2529_252919

theorem bakery_pie_division (total_pie : ℚ) (num_friends : ℕ) : 
  total_pie = 5/8 → num_friends = 4 → total_pie / num_friends = 5/32 := by
  sorry

end NUMINAMATH_CALUDE_bakery_pie_division_l2529_252919


namespace NUMINAMATH_CALUDE_line_of_symmetry_l2529_252929

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the symmetry property
axiom symmetry_property : ∀ x, g x = g (3 - x)

-- Define what it means for a line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- State the theorem
theorem line_of_symmetry :
  (∀ x, g x = g (3 - x)) → is_axis_of_symmetry g 1.5 :=
by sorry

end NUMINAMATH_CALUDE_line_of_symmetry_l2529_252929


namespace NUMINAMATH_CALUDE_base7_even_digits_528_l2529_252949

/-- Converts a natural number to its base-7 representation as a list of digits -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

theorem base7_even_digits_528 :
  countEvenDigits (toBase7 528) = 0 := by
  sorry

end NUMINAMATH_CALUDE_base7_even_digits_528_l2529_252949


namespace NUMINAMATH_CALUDE_average_age_parents_and_children_l2529_252984

theorem average_age_parents_and_children (num_children : ℕ) (num_parents : ℕ) 
  (avg_age_children : ℝ) (avg_age_parents : ℝ) :
  num_children = 40 →
  num_parents = 60 →
  avg_age_children = 12 →
  avg_age_parents = 35 →
  (num_children * avg_age_children + num_parents * avg_age_parents) / (num_children + num_parents) = 25.8 := by
  sorry

end NUMINAMATH_CALUDE_average_age_parents_and_children_l2529_252984


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l2529_252963

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt (9 * x - 4) + 15 / Real.sqrt (9 * x - 4) = 8) ↔ (x = 29 / 9 ∨ x = 13 / 9) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l2529_252963


namespace NUMINAMATH_CALUDE_peanut_seed_germination_l2529_252974

/-- The probability of at least k successes in n independent Bernoulli trials -/
def prob_at_least (n k : ℕ) (p : ℝ) : ℝ := sorry

/-- The probability of exactly k successes in n independent Bernoulli trials -/
def prob_exactly (n k : ℕ) (p : ℝ) : ℝ := sorry

theorem peanut_seed_germination :
  let n : ℕ := 4
  let k : ℕ := 2
  let p : ℝ := 4/5
  prob_at_least n k p = 608/625 := by sorry

end NUMINAMATH_CALUDE_peanut_seed_germination_l2529_252974


namespace NUMINAMATH_CALUDE_base_conversion_problem_l2529_252966

-- Define a function to convert a number from base n to decimal
def to_decimal (digits : List Nat) (n : Nat) : Nat :=
  digits.enum.foldr (fun (i, digit) acc => acc + digit * n ^ i) 0

-- Define the problem statement
theorem base_conversion_problem (n : Nat) (d : Nat) :
  n > 0 →  -- n is a positive integer
  d < 10 →  -- d is a digit
  to_decimal [4, 5, d] n = 392 →  -- 45d in base n equals 392
  to_decimal [4, 5, 7] n = to_decimal [2, 1, d, 5] 7 →  -- 457 in base n equals 21d5 in base 7
  n + d = 12 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l2529_252966


namespace NUMINAMATH_CALUDE_ratio_equality_l2529_252972

theorem ratio_equality (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_eq : y / (x - y) = (x - y) / z ∧ (x - y) / z = z / (x + y)) :
  x / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2529_252972


namespace NUMINAMATH_CALUDE_c_less_than_a_l2529_252992

theorem c_less_than_a (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0)
  (h1 : c / (a + b) = 2) (h2 : c / (b - a) = 3) : c < a := by
  sorry

end NUMINAMATH_CALUDE_c_less_than_a_l2529_252992


namespace NUMINAMATH_CALUDE_point_on_positive_x_axis_l2529_252933

theorem point_on_positive_x_axis (m : ℝ) : 
  let x := m^2 + Real.pi
  let y := 0
  x > 0 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_point_on_positive_x_axis_l2529_252933


namespace NUMINAMATH_CALUDE_jellybean_probability_l2529_252950

def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 3
def white_jellybeans : ℕ := 4
def jellybeans_picked : ℕ := 4

theorem jellybean_probability :
  (Nat.choose red_jellybeans 3 * Nat.choose (blue_jellybeans + white_jellybeans) 1) /
  Nat.choose total_jellybeans jellybeans_picked = 14 / 99 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l2529_252950


namespace NUMINAMATH_CALUDE_nabla_ratio_equals_eight_l2529_252980

-- Define the ∇ operation for positive integers m < n
def nabla (m n : ℕ) (h1 : 0 < m) (h2 : m < n) : ℕ :=
  (n - m + 1) * (m + n) / 2

-- Theorem statement
theorem nabla_ratio_equals_eight :
  nabla 22 26 (by norm_num) (by norm_num) / nabla 4 6 (by norm_num) (by norm_num) = 8 := by
  sorry

end NUMINAMATH_CALUDE_nabla_ratio_equals_eight_l2529_252980


namespace NUMINAMATH_CALUDE_time_after_1456_minutes_l2529_252994

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem time_after_1456_minutes :
  let start_time : Time := ⟨6, 0, by sorry⟩
  let elapsed_minutes : Nat := 1456
  let end_time : Time := addMinutes start_time elapsed_minutes
  end_time = ⟨6, 16, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_time_after_1456_minutes_l2529_252994


namespace NUMINAMATH_CALUDE_cycle_selling_price_l2529_252944

theorem cycle_selling_price (cost_price : ℝ) (loss_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 1400 → 
  loss_percentage = 5 → 
  selling_price = cost_price * (1 - loss_percentage / 100) → 
  selling_price = 1330 := by
sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l2529_252944


namespace NUMINAMATH_CALUDE_johns_candy_cost_l2529_252990

/-- The amount John pays for candy bars after sharing the cost with Dave -/
def johnsPay (totalBars : ℕ) (daveBars : ℕ) (originalPrice : ℚ) (discountRate : ℚ) : ℚ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  let totalCost := totalBars * discountedPrice
  let johnBars := totalBars - daveBars
  johnBars * discountedPrice

/-- Theorem stating that John pays $11.20 for his share of the candy bars -/
theorem johns_candy_cost :
  johnsPay 20 6 1 (20 / 100) = 11.2 := by
  sorry

end NUMINAMATH_CALUDE_johns_candy_cost_l2529_252990


namespace NUMINAMATH_CALUDE_duck_problem_solution_l2529_252979

/-- Represents the duck population problem --/
def duck_problem (initial_flock : ℕ) (killed_per_year : ℕ) (born_per_year : ℕ) 
                 (other_flock : ℕ) (combined_flock : ℕ) : Prop :=
  ∃ y : ℕ, 
    initial_flock + (born_per_year - killed_per_year) * y + other_flock = combined_flock

/-- Theorem stating the solution to the duck population problem --/
theorem duck_problem_solution : 
  duck_problem 100 20 30 150 300 → 
  ∃ y : ℕ, y = 5 ∧ duck_problem 100 20 30 150 300 := by
  sorry

#check duck_problem_solution

end NUMINAMATH_CALUDE_duck_problem_solution_l2529_252979


namespace NUMINAMATH_CALUDE_composite_sum_l2529_252924

theorem composite_sum (a b : ℕ+) (h : 34 * a = 43 * b) : 
  ∃ (k m : ℕ) (hk : k > 1) (hm : m > 1), a + b = k * m := by
sorry

end NUMINAMATH_CALUDE_composite_sum_l2529_252924


namespace NUMINAMATH_CALUDE_max_quotient_value_l2529_252926

theorem max_quotient_value (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) :
  (∀ x y, 300 ≤ x ∧ x ≤ 500 ∧ 800 ≤ y ∧ y ≤ 1600 → y / x ≤ 16 / 3) ∧
  (∃ x y, 300 ≤ x ∧ x ≤ 500 ∧ 800 ≤ y ∧ y ≤ 1600 ∧ y / x = 16 / 3) :=
sorry

end NUMINAMATH_CALUDE_max_quotient_value_l2529_252926


namespace NUMINAMATH_CALUDE_solution_to_modular_equation_l2529_252960

theorem solution_to_modular_equation :
  ∃ x : ℤ, (7 * x + 2) % 15 = 11 % 15 ∧ x % 15 = 12 % 15 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_modular_equation_l2529_252960


namespace NUMINAMATH_CALUDE_fish_in_tank_l2529_252953

theorem fish_in_tank (total : ℕ) (blue : ℕ) (spotted : ℕ) : 
  3 * blue = total → 
  2 * spotted = blue → 
  spotted = 10 → 
  total = 60 := by
sorry

end NUMINAMATH_CALUDE_fish_in_tank_l2529_252953


namespace NUMINAMATH_CALUDE_count_arrangements_l2529_252946

/-- Represents the number of students --/
def totalStudents : ℕ := 6

/-- Represents the number of boys --/
def numBoys : ℕ := 3

/-- Represents the number of girls --/
def numGirls : ℕ := 3

/-- Represents whether girls are allowed at the ends --/
def girlsAtEnds : Prop := False

/-- Represents whether girls A and B can stand next to girl C --/
def girlsABNextToC : Prop := False

/-- The number of valid arrangements --/
def validArrangements : ℕ := 72

/-- Theorem stating the number of valid arrangements --/
theorem count_arrangements :
  (totalStudents = numBoys + numGirls) →
  (numBoys = 3) →
  (numGirls = 3) →
  girlsAtEnds = False →
  girlsABNextToC = False →
  validArrangements = 72 := by
  sorry

end NUMINAMATH_CALUDE_count_arrangements_l2529_252946


namespace NUMINAMATH_CALUDE_nells_baseball_cards_l2529_252934

/-- Nell's card collection problem -/
theorem nells_baseball_cards
  (initial_ace : ℕ)
  (final_ace final_baseball : ℕ)
  (ace_difference baseball_difference : ℕ)
  (h1 : initial_ace = 18)
  (h2 : final_ace = 55)
  (h3 : final_baseball = 178)
  (h4 : baseball_difference = 123)
  (h5 : final_baseball = final_ace + baseball_difference)
  : final_baseball + baseball_difference = 301 := by
  sorry

#check nells_baseball_cards

end NUMINAMATH_CALUDE_nells_baseball_cards_l2529_252934


namespace NUMINAMATH_CALUDE_snack_combinations_l2529_252927

def num_items : ℕ := 4
def items_to_choose : ℕ := 2

theorem snack_combinations : 
  Nat.choose num_items items_to_choose = 6 := by sorry

end NUMINAMATH_CALUDE_snack_combinations_l2529_252927


namespace NUMINAMATH_CALUDE_lemon_cupcakes_total_l2529_252971

theorem lemon_cupcakes_total (cupcakes_at_home : ℕ) (boxes_given : ℕ) (cupcakes_per_box : ℕ) : 
  cupcakes_at_home = 2 → boxes_given = 17 → cupcakes_per_box = 3 →
  cupcakes_at_home + boxes_given * cupcakes_per_box = 53 := by
  sorry

end NUMINAMATH_CALUDE_lemon_cupcakes_total_l2529_252971


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2529_252981

/-- Given a group of 6 persons, if replacing one person with a new person weighing 74 kg
    increases the average weight by 1.5 kg, then the weight of the person being replaced is 65 kg. -/
theorem weight_of_replaced_person (group_size : ℕ) (new_person_weight : ℝ) (average_increase : ℝ) :
  group_size = 6 →
  new_person_weight = 74 →
  average_increase = 1.5 →
  ∃ (original_average : ℝ) (replaced_person_weight : ℝ),
    group_size * (original_average + average_increase) =
    group_size * original_average - replaced_person_weight + new_person_weight ∧
    replaced_person_weight = 65 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2529_252981


namespace NUMINAMATH_CALUDE_hilton_final_marbles_l2529_252952

/-- Calculates the final number of marbles Hilton has after a series of events -/
def hiltons_marbles (initial : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial + found - lost + 2 * lost

/-- Theorem stating that given the initial conditions, Hilton ends up with 42 marbles -/
theorem hilton_final_marbles :
  hiltons_marbles 26 6 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_hilton_final_marbles_l2529_252952


namespace NUMINAMATH_CALUDE_decimal_existence_l2529_252975

theorem decimal_existence :
  (∃ (a b : ℚ), 3.5 < a ∧ a < 3.6 ∧ 3.5 < b ∧ b < 3.6 ∧ a ≠ b) ∧
  (∃ (x y z : ℚ), 0 < x ∧ x < 0.1 ∧ 0 < y ∧ y < 0.1 ∧ 0 < z ∧ z < 0.1 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) :=
by sorry

end NUMINAMATH_CALUDE_decimal_existence_l2529_252975


namespace NUMINAMATH_CALUDE_original_fraction_l2529_252947

theorem original_fraction (x y : ℚ) :
  (x > 0) →
  (y > 0) →
  ((1.2 * x) / (0.75 * y) = 2 / 15) →
  (x / y = 1 / 12) := by
sorry

end NUMINAMATH_CALUDE_original_fraction_l2529_252947
