import Mathlib

namespace NUMINAMATH_CALUDE_color_one_third_square_l2471_247149

theorem color_one_third_square (n : ℕ) (k : ℕ) : n = 18 ∧ k = 6 → Nat.choose n k = 18564 := by
  sorry

end NUMINAMATH_CALUDE_color_one_third_square_l2471_247149


namespace NUMINAMATH_CALUDE_no_ripe_oranges_harvested_l2471_247166

/-- Given the daily harvest of unripe oranges and the total after 6 days,
    prove that no ripe oranges are harvested daily. -/
theorem no_ripe_oranges_harvested
  (unripe_daily : ℕ)
  (total_unripe_6days : ℕ)
  (h1 : unripe_daily = 65)
  (h2 : total_unripe_6days = 390)
  (h3 : unripe_daily * 6 = total_unripe_6days) :
  0 = (total_unripe_6days - unripe_daily * 6) / 6 :=
by sorry

end NUMINAMATH_CALUDE_no_ripe_oranges_harvested_l2471_247166


namespace NUMINAMATH_CALUDE_old_computer_wattage_is_1500_l2471_247105

/-- The wattage of John's old computer --/
def old_computer_wattage : ℝ := 1500

/-- The price increase of electricity --/
def electricity_price_increase : ℝ := 0.25

/-- The wattage increase of the new computer compared to the old one --/
def new_computer_wattage_increase : ℝ := 0.5

/-- The old price of electricity in dollars per kilowatt-hour --/
def old_electricity_price : ℝ := 0.12

/-- The cost to run the old computer for 50 hours in dollars --/
def old_computer_cost_50_hours : ℝ := 9

/-- The number of hours the old computer runs --/
def run_hours : ℝ := 50

/-- Theorem stating that the old computer's wattage is 1500 watts --/
theorem old_computer_wattage_is_1500 :
  old_computer_wattage = 
    (old_computer_cost_50_hours / run_hours) / old_electricity_price * 1000 :=
by sorry

end NUMINAMATH_CALUDE_old_computer_wattage_is_1500_l2471_247105


namespace NUMINAMATH_CALUDE_amelia_tuesday_distance_l2471_247180

/-- The distance Amelia drove on Tuesday -/
def tuesday_distance (total_distance monday_distance remaining_distance : ℕ) : ℕ :=
  total_distance - (monday_distance + remaining_distance)

theorem amelia_tuesday_distance :
  tuesday_distance 8205 907 6716 = 582 := by
  sorry

end NUMINAMATH_CALUDE_amelia_tuesday_distance_l2471_247180


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l2471_247129

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 15 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 289) ∧
  (has_no_small_prime_factors 289) ∧
  (∀ m : ℕ, m < 289 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l2471_247129


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l2471_247115

theorem smallest_n_satisfying_condition : ∃ n : ℕ, 
  (n > 1) ∧ 
  (∀ p : ℕ, 2 ≤ p ∧ p ≤ 10 → p ∣ (n^(p-1) - 1)) ∧
  (∀ m : ℕ, 1 < m ∧ m < n → ∃ q : ℕ, 2 ≤ q ∧ q ≤ 10 ∧ ¬(q ∣ (m^(q-1) - 1))) ∧
  n = 2521 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l2471_247115


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2471_247109

theorem inequality_equivalence (x : ℝ) : 
  (x ∈ Set.Icc (-1 : ℝ) 1) ↔ 
  (∀ (n : ℕ) (a : ℕ → ℝ), n ≥ 2 → (∀ i, i ∈ Finset.range n → a i ≥ 1) → 
    ((Finset.range n).prod (λ i => (a i + x) / 2) ≤ 
     ((Finset.range n).prod (λ i => a i) + x) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2471_247109


namespace NUMINAMATH_CALUDE_car_y_win_probability_l2471_247139

/-- The probability of car Y winning a race given specific conditions -/
theorem car_y_win_probability (total_cars : ℕ) (prob_x prob_z prob_xyz : ℝ) : 
  total_cars = 15 →
  prob_x = 1/4 →
  prob_z = 1/12 →
  prob_xyz = 0.4583333333333333 →
  ∃ (prob_y : ℝ), prob_y = 1/8 ∧ prob_x + prob_y + prob_z = prob_xyz :=
by sorry

end NUMINAMATH_CALUDE_car_y_win_probability_l2471_247139


namespace NUMINAMATH_CALUDE_gcd_plus_lcm_eq_sum_iff_divides_l2471_247163

theorem gcd_plus_lcm_eq_sum_iff_divides (x y : ℕ) :
  (Nat.gcd x y + x * y / Nat.gcd x y = x + y) ↔ (y ∣ x ∨ x ∣ y) := by
  sorry

end NUMINAMATH_CALUDE_gcd_plus_lcm_eq_sum_iff_divides_l2471_247163


namespace NUMINAMATH_CALUDE_range_of_a_l2471_247152

/-- Proposition p: The equation represents a hyperbola -/
def p (a : ℝ) : Prop := 2 - a > 0 ∧ a + 1 > 0

/-- Proposition q: The equation has real roots -/
def q (a : ℝ) : Prop := 16 + 4 * a ≥ 0

/-- The range of a given the negation of p and q is true -/
theorem range_of_a : ∀ a : ℝ, (¬p a ∧ q a) → (a ≤ -1 ∨ a ≥ 2) ∧ a ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2471_247152


namespace NUMINAMATH_CALUDE_triangle_parallelogram_altitude_relation_l2471_247194

theorem triangle_parallelogram_altitude_relation 
  (base : ℝ) 
  (triangle_area parallelogram_area : ℝ) 
  (triangle_altitude parallelogram_altitude : ℝ) 
  (h1 : triangle_area = parallelogram_area) 
  (h2 : parallelogram_altitude = 100) 
  (h3 : triangle_area = 1/2 * base * triangle_altitude) 
  (h4 : parallelogram_area = base * parallelogram_altitude) : 
  triangle_altitude = 200 := by
sorry

end NUMINAMATH_CALUDE_triangle_parallelogram_altitude_relation_l2471_247194


namespace NUMINAMATH_CALUDE_base_seven_digits_of_1200_l2471_247182

theorem base_seven_digits_of_1200 : ∃ n : ℕ, (7^(n-1) ≤ 1200 ∧ 1200 < 7^n) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_1200_l2471_247182


namespace NUMINAMATH_CALUDE_equality_of_fractions_l2471_247193

theorem equality_of_fractions (M N : ℚ) : 
  (4 : ℚ) / 7 = M / 49 ∧ (4 : ℚ) / 7 = 84 / N → M - N = -119 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l2471_247193


namespace NUMINAMATH_CALUDE_dog_drying_ratio_l2471_247164

/-- The time (in minutes) it takes to dry a short-haired dog -/
def short_hair_time : ℕ := 10

/-- The number of short-haired dogs -/
def num_short_hair : ℕ := 6

/-- The number of full-haired dogs -/
def num_full_hair : ℕ := 9

/-- The total time (in minutes) it takes to dry all dogs -/
def total_time : ℕ := 240

/-- The ratio of time to dry a full-haired dog to a short-haired dog -/
def drying_ratio : ℚ := 2

theorem dog_drying_ratio :
  ∃ (full_hair_time : ℕ),
    full_hair_time = short_hair_time * (drying_ratio.num / drying_ratio.den) ∧
    num_short_hair * short_hair_time + num_full_hair * full_hair_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_dog_drying_ratio_l2471_247164


namespace NUMINAMATH_CALUDE_company_workforce_l2471_247121

theorem company_workforce (initial_total : ℕ) : 
  (initial_total * 3 / 5 : ℚ) = initial_total * 0.6 →
  (initial_total * 3 / 5 : ℚ) / (initial_total + 28) = 0.55 →
  initial_total + 28 = 336 := by
sorry

end NUMINAMATH_CALUDE_company_workforce_l2471_247121


namespace NUMINAMATH_CALUDE_decaf_coffee_percentage_l2471_247167

/-- Calculates the percentage of decaffeinated coffee in the total stock -/
theorem decaf_coffee_percentage 
  (initial_stock : ℝ) 
  (initial_decaf_percent : ℝ)
  (additional_stock : ℝ)
  (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 20)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 50) :
  let total_stock := initial_stock + additional_stock
  let total_decaf := (initial_stock * initial_decaf_percent / 100) + 
                     (additional_stock * additional_decaf_percent / 100)
  let final_decaf_percent := (total_decaf / total_stock) * 100
  final_decaf_percent = 26 := by
sorry

end NUMINAMATH_CALUDE_decaf_coffee_percentage_l2471_247167


namespace NUMINAMATH_CALUDE_modulus_z_l2471_247147

theorem modulus_z (z : ℂ) (h : z * (1 + 2*I) = 4 + 3*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_z_l2471_247147


namespace NUMINAMATH_CALUDE_town_population_distribution_l2471_247106

theorem town_population_distribution (total_population : ℕ) 
  (h1 : total_population = 600) 
  (h2 : ∃ (males females children : ℕ), 
    males + females + children = total_population ∧ 
    children = 2 * males ∧ 
    males + females + children = 4 * males) : 
  ∃ (males : ℕ), males = 150 := by
sorry

end NUMINAMATH_CALUDE_town_population_distribution_l2471_247106


namespace NUMINAMATH_CALUDE_equal_potato_distribution_l2471_247125

theorem equal_potato_distribution (total_potatoes : ℕ) (family_members : ℕ) 
  (h1 : total_potatoes = 60) (h2 : family_members = 6) :
  total_potatoes / family_members = 10 := by
  sorry

end NUMINAMATH_CALUDE_equal_potato_distribution_l2471_247125


namespace NUMINAMATH_CALUDE_problem_statement_l2471_247161

theorem problem_statement (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum_squares : x^2 + y^2 + z^2 = 6) :
  -- Part 1: Maximum value of x + 2y + z is 6
  (∃ (max : ℝ), max = 6 ∧ x + 2*y + z ≤ max ∧ 
    ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
      x₀^2 + y₀^2 + z₀^2 = 6 ∧ x₀ + 2*y₀ + z₀ = max) ∧
  -- Part 2: If |a+1| - 2a ≥ x + 2y + z for all valid x, y, z, then a ≤ -7/3
  (∀ (a : ℝ), (∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 → 
    x'^2 + y'^2 + z'^2 = 6 → |a + 1| - 2*a ≥ x' + 2*y' + z') → a ≤ -7/3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2471_247161


namespace NUMINAMATH_CALUDE_average_difference_is_negative_13_point_5_l2471_247191

/-- Represents a school with students and teachers -/
structure School where
  num_students : ℕ
  num_teachers : ℕ
  class_sizes : List ℕ

/-- Calculates the average number of students per teacher -/
def average_students_per_teacher (school : School) : ℚ :=
  (school.class_sizes.sum : ℚ) / school.num_teachers

/-- Calculates the average number of students per student -/
def average_students_per_student (school : School) : ℚ :=
  (school.class_sizes.map (λ size => size * size)).sum / school.num_students

/-- The main theorem to be proved -/
theorem average_difference_is_negative_13_point_5 (school : School) 
  (h1 : school.num_students = 100)
  (h2 : school.num_teachers = 5)
  (h3 : school.class_sizes = [50, 20, 20, 5, 5]) :
  average_students_per_teacher school - average_students_per_student school = -13.5 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_is_negative_13_point_5_l2471_247191


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_3_sqrt_difference_times_sqrt_3_equals_neg_sqrt_6_l2471_247151

-- Part 1
theorem sqrt_expression_equals_sqrt_3 :
  Real.sqrt 12 - Real.sqrt 48 + 9 * Real.sqrt (1/3) = Real.sqrt 3 := by sorry

-- Part 2
theorem sqrt_difference_times_sqrt_3_equals_neg_sqrt_6 :
  (Real.sqrt 8 - Real.sqrt 18) * Real.sqrt 3 = -Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_3_sqrt_difference_times_sqrt_3_equals_neg_sqrt_6_l2471_247151


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l2471_247195

theorem quadratic_roots_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - 11*x + (30 + k) = 0 → x > 5) → 
  0 < k ∧ k ≤ 1/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l2471_247195


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2471_247146

theorem inequality_solution_set (x : ℝ) : 
  (1 + x) * (2 - x) * (3 + x^2) > 0 ↔ -1 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2471_247146


namespace NUMINAMATH_CALUDE_total_annual_interest_l2471_247134

def total_amount : ℝ := 1600
def interest_rate_x : ℝ := 0.06
def interest_rate_y : ℝ := 0.05
def lent_amount : ℝ := 1100
def lent_interest_rate : ℝ := 0.0500001

theorem total_annual_interest :
  ∀ x y : ℝ,
  x + y = total_amount →
  y = lent_amount →
  x * interest_rate_x + y * lent_interest_rate = 85.00011 := by
sorry

end NUMINAMATH_CALUDE_total_annual_interest_l2471_247134


namespace NUMINAMATH_CALUDE_number_ten_greater_than_negative_nine_l2471_247126

theorem number_ten_greater_than_negative_nine : -9 + 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_ten_greater_than_negative_nine_l2471_247126


namespace NUMINAMATH_CALUDE_inverse_variation_proof_l2471_247158

/-- Given that x² varies inversely with y⁴, prove that when x = 5 for y = 2, 
    then x² = 25/16 when y = 4 -/
theorem inverse_variation_proof (x y : ℝ) (h : ∃ k : ℝ, x^2 * y^4 = k) 
  (h_initial : (5 : ℝ)^2 * 2^4 = x^2 * y^4) : 
  (∃ x' : ℝ, x'^2 * 4^4 = x^2 * y^4 ∧ x'^2 = 25/16) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_proof_l2471_247158


namespace NUMINAMATH_CALUDE_container_cubes_theorem_l2471_247168

/-- Represents the dimensions of a rectangular container -/
structure ContainerDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℕ

/-- Calculate the number of cubes that can fit in the container -/
def cubesFit (container : ContainerDimensions) (cube : CubeDimensions) : ℕ :=
  (container.length / cube.side) * (container.width / cube.side) * (container.height / cube.side)

/-- Calculate the volume of the container -/
def containerVolume (container : ContainerDimensions) : ℕ :=
  container.length * container.width * container.height

/-- Calculate the volume of a single cube -/
def cubeVolume (cube : CubeDimensions) : ℕ :=
  cube.side * cube.side * cube.side

/-- Calculate the fraction of the container volume occupied by cubes -/
def occupiedFraction (container : ContainerDimensions) (cube : CubeDimensions) : ℚ :=
  (cubesFit container cube * cubeVolume cube : ℚ) / containerVolume container

theorem container_cubes_theorem (container : ContainerDimensions) (cube : CubeDimensions) 
  (h1 : container.length = 8)
  (h2 : container.width = 4)
  (h3 : container.height = 9)
  (h4 : cube.side = 2) :
  cubesFit container cube = 32 ∧ occupiedFraction container cube = 8/9 := by
  sorry

#eval cubesFit ⟨8, 4, 9⟩ ⟨2⟩
#eval occupiedFraction ⟨8, 4, 9⟩ ⟨2⟩

end NUMINAMATH_CALUDE_container_cubes_theorem_l2471_247168


namespace NUMINAMATH_CALUDE_second_polygon_sides_l2471_247175

/-- A regular polygon -/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ

/-- The perimeter of a regular polygon -/
def perimeter (p : RegularPolygon) : ℝ := p.sides * p.sideLength

theorem second_polygon_sides 
  (p1 p2 : RegularPolygon) 
  (h1 : p1.sides = 42)
  (h2 : p1.sideLength = 3 * p2.sideLength)
  (h3 : perimeter p1 = perimeter p2) :
  p2.sides = 126 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l2471_247175


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l2471_247190

theorem quadratic_equation_integer_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ x^2 - (a+8)*x + 8*a - 1 = 0 ∧ y^2 - (a+8)*y + 8*a - 1 = 0) → 
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_roots_l2471_247190


namespace NUMINAMATH_CALUDE_swim_club_members_l2471_247110

theorem swim_club_members :
  ∀ (total_members : ℕ) 
    (passed_test : ℕ) 
    (not_passed_with_course : ℕ) 
    (not_passed_without_course : ℕ),
  passed_test = (30 * total_members) / 100 →
  not_passed_with_course = 5 →
  not_passed_without_course = 30 →
  total_members = passed_test + not_passed_with_course + not_passed_without_course →
  total_members = 50 := by
sorry

end NUMINAMATH_CALUDE_swim_club_members_l2471_247110


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l2471_247150

theorem roots_quadratic_equation (α β : ℝ) : 
  (α^2 + α - 1 = 0) → (β^2 + β - 1 = 0) → α^2 + 2*β^2 + β = 4 := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l2471_247150


namespace NUMINAMATH_CALUDE_sum_base7_and_base13_equals_1109_l2471_247127

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 13 to base 10, where 'C' represents 12 -/
def base13ToBase10 (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of 536 (base 7) and 4C5 (base 13) is 1109 in base 10 -/
theorem sum_base7_and_base13_equals_1109 : 
  base7ToBase10 536 + base13ToBase10 4125 = 1109 := by sorry

end NUMINAMATH_CALUDE_sum_base7_and_base13_equals_1109_l2471_247127


namespace NUMINAMATH_CALUDE_sum_bound_l2471_247113

theorem sum_bound (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (h : (x + 3)^2 / (y + z - 4) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  21 ≤ x + y + z ∧ x + y + z ≤ 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_bound_l2471_247113


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2471_247123

theorem algebraic_expression_value (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = -1) :
  (2 * a + 3 * b - 2 * a * b) - (a + 4 * b + a * b) - (3 * a * b + 2 * b - 2 * a) = 21 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2471_247123


namespace NUMINAMATH_CALUDE_circle_equation_l2471_247170

/-- A circle with center on the x-axis, passing through the origin, and tangent to the line y = 4 has the general equation x^2 + y^2 ± 8x = 0. -/
theorem circle_equation (x y : ℝ) : 
  ∃ (a : ℝ), 
    (∀ (x₀ y₀ : ℝ), (x₀ - a)^2 + y₀^2 = 16 → y₀ ≤ 4) ∧  -- Circle is tangent to y = 4
    ((0 - a)^2 + 0^2 = 16) ∧                             -- Circle passes through origin
    (a^2 = 16) →                                         -- Center is on x-axis at distance 4 from origin
  (x^2 + y^2 + 8*x = 0 ∨ x^2 + y^2 - 8*x = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2471_247170


namespace NUMINAMATH_CALUDE_expression_minimizes_q_l2471_247155

/-- The function q in terms of x and the expression to be determined -/
def q (x : ℝ) (expression : ℝ → ℝ) : ℝ :=
  (expression x)^2 + (x + 1)^2 - 6

/-- The condition that y is least when x = 2 -/
axiom y_min_at_2 : ∀ (y : ℝ → ℝ), ∀ (x : ℝ), y 2 ≤ y x

/-- The relationship between q and y -/
axiom q_related_to_y : ∃ (y : ℝ → ℝ), ∀ (x : ℝ), q x (λ t => t - 2) = y x

/-- The theorem stating that (x - 2) minimizes q when x = 2 -/
theorem expression_minimizes_q :
  ∀ (x : ℝ), q 2 (λ t => t - 2) ≤ q x (λ t => t - 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_minimizes_q_l2471_247155


namespace NUMINAMATH_CALUDE_power_division_thirteen_l2471_247130

theorem power_division_thirteen : 13^8 / 13^5 = 2197 := by sorry

end NUMINAMATH_CALUDE_power_division_thirteen_l2471_247130


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2471_247144

/-- The sum of a geometric series with n terms, first term a, and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the specific geometric series in the problem -/
def specificSum : ℚ :=
  geometricSum (1/4) (1/4) 8

theorem geometric_series_sum :
  specificSum = 65535 / 196608 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2471_247144


namespace NUMINAMATH_CALUDE_max_servings_is_56_l2471_247118

/-- Represents the ingredients required for one serving of salad -/
structure ServingRequirement where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Represents the available ingredients in the restaurant's warehouse -/
structure AvailableIngredients where
  cucumbers : ℕ
  tomatoes : ℕ
  brynza : ℕ  -- in grams
  peppers : ℕ

/-- Calculates the maximum number of servings that can be made -/
def maxServings (req : ServingRequirement) (avail : AvailableIngredients) : ℕ :=
  min (avail.cucumbers / req.cucumbers)
      (min (avail.tomatoes / req.tomatoes)
           (min (avail.brynza / req.brynza)
                (avail.peppers / req.peppers)))

/-- Theorem stating that the maximum number of servings is 56 -/
theorem max_servings_is_56 :
  let req := ServingRequirement.mk 2 2 75 1
  let avail := AvailableIngredients.mk 117 116 4200 60
  maxServings req avail = 56 := by sorry

end NUMINAMATH_CALUDE_max_servings_is_56_l2471_247118


namespace NUMINAMATH_CALUDE_sum_square_units_digits_2023_l2471_247145

def first_odd_integers (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => 2 * i + 1)

def square_units_digit (n : ℕ) : ℕ :=
  (n ^ 2) % 10

def sum_square_units_digits (n : ℕ) : ℕ :=
  (first_odd_integers n).map square_units_digit |> List.sum

theorem sum_square_units_digits_2023 :
  sum_square_units_digits 2023 % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_square_units_digits_2023_l2471_247145


namespace NUMINAMATH_CALUDE_painting_time_for_six_stools_l2471_247189

/-- Represents the painting process for stools -/
structure StoolPainting where
  num_stools : Nat
  first_coat_time : Nat
  wait_time : Nat

/-- Calculates the minimum time required to paint all stools -/
def minimum_painting_time (sp : StoolPainting) : Nat :=
  sp.num_stools * sp.first_coat_time + sp.wait_time + sp.first_coat_time

/-- Theorem stating that the minimum time to paint 6 stools is 24 minutes -/
theorem painting_time_for_six_stools :
  let sp : StoolPainting := {
    num_stools := 6,
    first_coat_time := 2,
    wait_time := 10
  }
  minimum_painting_time sp = 24 := by
  sorry


end NUMINAMATH_CALUDE_painting_time_for_six_stools_l2471_247189


namespace NUMINAMATH_CALUDE_problem_solution_l2471_247198

theorem problem_solution (a b c d x y : ℝ) 
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : (x + 3)^2 + |y - 2| = 0) :
  2*(a + b) - 2*(c*d)^4 + (x + y)^2022 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2471_247198


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2471_247108

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference between consecutive terms -/
  d : ℝ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a + (n - 1 : ℝ) * seq.d

theorem arithmetic_sequence_seventh_term
  (seq : ArithmeticSequence)
  (h3 : seq.nthTerm 3 = 17)
  (h5 : seq.nthTerm 5 = 39) :
  seq.nthTerm 7 = 61 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2471_247108


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_two_vars_arithmetic_geometric_inequality_three_vars_l2471_247183

theorem arithmetic_geometric_inequality_two_vars (a b : ℝ) (h : a ≤ b) :
  a^2 + b^2 ≥ 2 * a * b := by sorry

theorem arithmetic_geometric_inequality_three_vars (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) :
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_two_vars_arithmetic_geometric_inequality_three_vars_l2471_247183


namespace NUMINAMATH_CALUDE_blue_box_contains_70_blueberries_l2471_247136

/-- Represents the number of blueberries in each blue box -/
def blueberries : ℕ := sorry

/-- Represents the number of strawberries in each red box -/
def strawberries : ℕ := sorry

/-- The increase in total berries when replacing a blue box with a red box -/
def total_increase : ℕ := 30

/-- The increase in difference between strawberries and blueberries when replacing a blue box with a red box -/
def difference_increase : ℕ := 100

theorem blue_box_contains_70_blueberries :
  (strawberries - blueberries = total_increase) ∧
  (strawberries = difference_increase) →
  blueberries = 70 := by sorry

end NUMINAMATH_CALUDE_blue_box_contains_70_blueberries_l2471_247136


namespace NUMINAMATH_CALUDE_class_size_l2471_247156

theorem class_size (initial_absent : ℚ) (final_absent : ℚ) (total : ℕ) : 
  initial_absent = 1 / 6 →
  final_absent = 1 / 5 →
  (initial_absent / (1 + initial_absent)) * total + 1 = (final_absent / (1 + final_absent)) * total →
  total = 42 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2471_247156


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l2471_247197

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (10000 ≤ n ∧ n ≤ 99999) ∧ 
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n = b^3) ∧
  (∀ m : ℕ, (10000 ≤ m ∧ m < n) → ¬(∃ x : ℕ, m = x^2) ∨ ¬(∃ y : ℕ, m = y^3)) ∧
  n = 15625 := by
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l2471_247197


namespace NUMINAMATH_CALUDE_basketball_weight_l2471_247165

/-- Given that eight identical basketballs weigh the same as four identical watermelons,
    and one watermelon weighs 32 pounds, prove that one basketball weighs 16 pounds. -/
theorem basketball_weight (watermelon_weight : ℝ) (basketball_weight : ℝ) : 
  watermelon_weight = 32 →
  8 * basketball_weight = 4 * watermelon_weight →
  basketball_weight = 16 := by
sorry

end NUMINAMATH_CALUDE_basketball_weight_l2471_247165


namespace NUMINAMATH_CALUDE_min_value_of_f_l2471_247137

/-- The quadratic function f(x) = x^2 - 2x - 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

/-- The minimum value of f(x) = x^2 - 2x - 1 for x ∈ ℝ is -2 -/
theorem min_value_of_f :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2471_247137


namespace NUMINAMATH_CALUDE_pascal_triangle_cube_sum_l2471_247104

/-- Pascal's Triangle interior numbers sum for row n -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- Sum of cubes of interior numbers in the fifth row -/
def fifth_row_cube_sum : ℕ := 468

/-- Sum of cubes of interior numbers in the sixth row -/
def sixth_row_cube_sum : ℕ := 14750

/-- Theorem: If the sum of cubes of interior numbers in the fifth row is 468,
    then the sum of cubes of interior numbers in the sixth row is 14750 -/
theorem pascal_triangle_cube_sum :
  fifth_row_cube_sum = 468 → sixth_row_cube_sum = 14750 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_cube_sum_l2471_247104


namespace NUMINAMATH_CALUDE_kims_sweater_difference_l2471_247107

/-- The number of sweaters Kim knit on each day of the week --/
structure WeeklySweaters where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Kim's sweater knitting for the week --/
def kimsSweaterWeek (s : WeeklySweaters) : Prop :=
  s.monday = 8 ∧
  s.tuesday > s.monday ∧
  s.wednesday = s.tuesday - 4 ∧
  s.thursday = s.tuesday - 4 ∧
  s.friday = s.monday / 2 ∧
  s.monday + s.tuesday + s.wednesday + s.thursday + s.friday = 34

theorem kims_sweater_difference (s : WeeklySweaters) 
  (h : kimsSweaterWeek s) : s.tuesday - s.monday = 2 := by
  sorry

end NUMINAMATH_CALUDE_kims_sweater_difference_l2471_247107


namespace NUMINAMATH_CALUDE_three_vertex_figure_perimeter_l2471_247178

/-- A figure with 3 vertices where the distance between any 2 vertices is 7 -/
structure ThreeVertexFigure where
  vertices : Fin 3 → ℝ × ℝ
  distance_eq_seven : ∀ i j, i ≠ j → Real.sqrt ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 7

/-- The perimeter of a ThreeVertexFigure is 21 -/
theorem three_vertex_figure_perimeter (f : ThreeVertexFigure) : 
  (Real.sqrt ((f.vertices 0).1 - (f.vertices 1).1)^2 + ((f.vertices 0).2 - (f.vertices 1).2)^2) +
  (Real.sqrt ((f.vertices 1).1 - (f.vertices 2).1)^2 + ((f.vertices 1).2 - (f.vertices 2).2)^2) +
  (Real.sqrt ((f.vertices 2).1 - (f.vertices 0).1)^2 + ((f.vertices 2).2 - (f.vertices 0).2)^2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_three_vertex_figure_perimeter_l2471_247178


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_range_l2471_247117

/-- The function f(x) defined in terms of a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3*x + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 6*a*x + 3

/-- Theorem: If f(x) has at least one extreme value point in (2, 3), then 5/4 < a < 5/3 -/
theorem extreme_value_implies_a_range (a : ℝ) :
  (∃ x : ℝ, 2 < x ∧ x < 3 ∧ f_deriv a x = 0) →
  (5/4 : ℝ) < a ∧ a < (5/3 : ℝ) :=
by
  sorry


end NUMINAMATH_CALUDE_extreme_value_implies_a_range_l2471_247117


namespace NUMINAMATH_CALUDE_even_function_increasing_interval_l2471_247143

def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + (m - 1) * x + 2

theorem even_function_increasing_interval (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  (∃ a : ℝ, ∀ x y : ℝ, x < y ∧ y ≤ 0 → f m x < f m y) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y → f m x > f m y) :=
sorry

end NUMINAMATH_CALUDE_even_function_increasing_interval_l2471_247143


namespace NUMINAMATH_CALUDE_minutes_to_seconds_l2471_247103

theorem minutes_to_seconds (minutes : ℝ) : minutes * 60 = 750 → minutes = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_seconds_l2471_247103


namespace NUMINAMATH_CALUDE_dad_steps_l2471_247135

theorem dad_steps (dad_masha_ratio : ℕ → ℕ → Prop)
                   (masha_yasha_ratio : ℕ → ℕ → Prop)
                   (masha_yasha_total : ℕ) :
  dad_masha_ratio 3 5 →
  masha_yasha_ratio 3 5 →
  masha_yasha_total = 400 →
  ∃ (dad_steps : ℕ), dad_steps = 90 := by
  sorry

end NUMINAMATH_CALUDE_dad_steps_l2471_247135


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2471_247100

theorem inscribed_square_area (circle_area : ℝ) (square_area : ℝ) : 
  circle_area = 25 * Real.pi → 
  ∃ (r : ℝ), 
    circle_area = Real.pi * r^2 ∧ 
    square_area = 2 * r^2 →
    square_area = 50 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2471_247100


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2471_247122

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- State the theorem
theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  d ≠ 0 →
  arithmetic_sequence a₁ d 5 = a₁^2 →
  a₁ * arithmetic_sequence a₁ d 21 = (arithmetic_sequence a₁ d 5)^2 →
  a₁ = 4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2471_247122


namespace NUMINAMATH_CALUDE_carlsons_original_land_size_l2471_247173

/-- Calculates the size of Carlson's original land given the cost and area of new land purchases --/
theorem carlsons_original_land_size
  (cost_land1 : ℝ)
  (cost_land2 : ℝ)
  (cost_per_sqm : ℝ)
  (total_area_after : ℝ)
  (h1 : cost_land1 = 8000)
  (h2 : cost_land2 = 4000)
  (h3 : cost_per_sqm = 20)
  (h4 : total_area_after = 900) :
  total_area_after - (cost_land1 + cost_land2) / cost_per_sqm = 300 :=
by sorry

end NUMINAMATH_CALUDE_carlsons_original_land_size_l2471_247173


namespace NUMINAMATH_CALUDE_value_of_m_l2471_247141

theorem value_of_m (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) 
  (expansion : ∀ x, (1 + m * x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6)
  (alternating_sum : a₁ - a₂ + a₃ - a₄ + a₅ - a₆ = -63) :
  m = 3 ∨ m = -1 := by
sorry

end NUMINAMATH_CALUDE_value_of_m_l2471_247141


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_l2471_247132

/-- Given a hyperbola and a circle, prove that the x-coordinate of their intersection point in the first quadrant is (√3 + 1) / 2 -/
theorem hyperbola_circle_intersection (b c : ℝ) (P : ℝ × ℝ) : 
  let (x, y) := P
  (x^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (x^2 + y^2 = c^2) →      -- Circle equation
  (x > 0 ∧ y > 0) →        -- P is in the first quadrant
  ((x - c)^2 + y^2 = (c + 2)^2) →  -- |PF1| = c + 2
  (x = (Real.sqrt 3 + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_l2471_247132


namespace NUMINAMATH_CALUDE_octahedron_containment_l2471_247199

-- Define the plane equation
def plane_equation (x y z : ℚ) (n : ℤ) : Prop :=
  (x + y + z = n) ∨ (x + y - z = n) ∨ (x - y + z = n) ∨ (x - y - z = n)

-- Define a point not on any plane
def not_on_planes (x y z : ℚ) : Prop :=
  ∀ n : ℤ, ¬ plane_equation x y z n

-- Define a point inside an octahedron
def inside_octahedron (x y z : ℚ) : Prop :=
  ∃ n : ℤ, 
    n < x + y + z ∧ x + y + z < n + 1 ∧
    n < x + y - z ∧ x + y - z < n + 1 ∧
    n < x - y + z ∧ x - y + z < n + 1 ∧
    n < -x + y + z ∧ -x + y + z < n + 1

-- The main theorem
theorem octahedron_containment (x₀ y₀ z₀ : ℚ) 
  (h : not_on_planes x₀ y₀ z₀) :
  ∃ k : ℕ, inside_octahedron (k * x₀) (k * y₀) (k * z₀) := by
  sorry

end NUMINAMATH_CALUDE_octahedron_containment_l2471_247199


namespace NUMINAMATH_CALUDE_area_S_bounds_l2471_247192

theorem area_S_bounds (t : ℝ) (k : ℤ) (h_t : t ≥ 0) (h_k : 2 ≤ k ∧ k ≤ 4) : 
  let T : ℝ := t - ⌊t⌋
  let S : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - T - 1)^2 + (p.2 - k)^2 ≤ (T + 1)^2}
  0 ≤ Real.pi * (T + 1)^2 ∧ Real.pi * (T + 1)^2 ≤ 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_area_S_bounds_l2471_247192


namespace NUMINAMATH_CALUDE_nested_function_ratio_l2471_247184

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

def g (x : ℝ) : ℝ := 2 * x^2 - x + 1

theorem nested_function_ratio :
  f (g (f 1)) / g (f (g 1)) = 6801 / 281 := by
  sorry

end NUMINAMATH_CALUDE_nested_function_ratio_l2471_247184


namespace NUMINAMATH_CALUDE_max_profit_theorem_l2471_247120

/-- Represents the profit function for a mobile phone store -/
def profit_function (x : ℝ) : ℝ := -200 * x + 140000

/-- Represents the constraint on the number of type B phones -/
def constraint (x : ℝ) : Prop := 100 - x ≤ 3 * x

/-- Theorem stating the maximum profit and optimal purchase strategy -/
theorem max_profit_theorem :
  ∃ (x : ℝ),
    x ≥ 0 ∧
    x ≤ 100 ∧
    constraint x ∧
    profit_function x = 135000 ∧
    (∀ y : ℝ, y ≥ 0 → y ≤ 100 → constraint y → profit_function y ≤ profit_function x) ∧
    x = 25 :=
  sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l2471_247120


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2471_247111

/-- A point in the second quadrant with specific properties has coordinates (-2, 1) -/
theorem point_in_second_quadrant (P : ℝ × ℝ) :
  (P.1 < 0 ∧ P.2 > 0) →  -- Second quadrant condition
  (abs P.1 = 2) →        -- |x| = 2 condition
  (P.2^2 = 1) →          -- y is square root of 1 condition
  P = (-2, 1) :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2471_247111


namespace NUMINAMATH_CALUDE_percentage_to_pass_l2471_247102

/-- Calculate the percentage needed to pass an exam. -/
theorem percentage_to_pass 
  (current_score : ℕ) 
  (shortfall : ℕ) 
  (max_marks : ℕ) 
  (h1 : current_score = 212)
  (h2 : shortfall = 13)
  (h3 : max_marks = 750) :
  (((current_score + shortfall : ℚ) / max_marks) * 100 : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l2471_247102


namespace NUMINAMATH_CALUDE_train_speed_l2471_247171

/-- Proves that a train with given passing times has a specific speed -/
theorem train_speed (pole_passing_time : ℝ) (stationary_train_length : ℝ) (stationary_train_passing_time : ℝ) :
  pole_passing_time = 10 →
  stationary_train_length = 300 →
  stationary_train_passing_time = 40 →
  ∃ (train_length : ℝ),
    train_length > 0 ∧
    train_length / pole_passing_time = (train_length + stationary_train_length) / stationary_train_passing_time ∧
    train_length / pole_passing_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l2471_247171


namespace NUMINAMATH_CALUDE_set_equality_l2471_247128

def S : Set ℤ := {x | -3 < x ∧ x < 3}

theorem set_equality : S = {-2, -1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_equality_l2471_247128


namespace NUMINAMATH_CALUDE_shoppingMallMethodIsSystematic_l2471_247179

/-- Represents a sampling method with a fixed interval and starting point -/
structure SamplingMethod where
  interval : ℕ
  start : ℕ

/-- Defines the characteristics of systematic sampling -/
def isSystematicSampling (method : SamplingMethod) : Prop :=
  method.interval > 0 ∧
  method.start > 0 ∧
  method.start ≤ method.interval

/-- The sampling method used by the shopping mall -/
def shoppingMallMethod : SamplingMethod :=
  { interval := 50,
    start := 15 }

/-- Theorem stating that the shopping mall's method is a systematic sampling method -/
theorem shoppingMallMethodIsSystematic :
  isSystematicSampling shoppingMallMethod := by
  sorry

end NUMINAMATH_CALUDE_shoppingMallMethodIsSystematic_l2471_247179


namespace NUMINAMATH_CALUDE_perpendicular_parallel_perpendicular_l2471_247187

-- Define a type for lines in 3D space
structure Line3D where
  -- You might represent a line by a point and a direction vector
  -- But for this abstract proof, we don't need to specify the internals
  mk :: 

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define parallelism for lines
def parallel (l1 l2 : Line3D) : Prop := sorry

-- The theorem to prove
theorem perpendicular_parallel_perpendicular 
  (l1 l2 l3 : Line3D) : 
  perpendicular l1 l2 → parallel l2 l3 → perpendicular l1 l3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_perpendicular_l2471_247187


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_arcseconds_to_degrees_conversion_negative_fraction_comparison_l2471_247142

-- Define the conversion factor from arcseconds to degrees
def arcseconds_to_degrees (x : ℚ) : ℚ := x / 3600

-- Theorem statements
theorem reciprocal_of_negative_three : ((-3)⁻¹ : ℚ) = -1/3 := by sorry

theorem arcseconds_to_degrees_conversion : arcseconds_to_degrees 7200 = 2 := by sorry

theorem negative_fraction_comparison : (-3/4 : ℚ) > -4/5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_arcseconds_to_degrees_conversion_negative_fraction_comparison_l2471_247142


namespace NUMINAMATH_CALUDE_power_function_decreasing_n_l2471_247114

/-- A power function f(x) = ax^n where a and n are constants and x > 0 -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), ∀ x > 0, f x = a * x^n

/-- A function f is decreasing on (0, +∞) if for all x, y in (0, +∞) with x < y, f(x) > f(y) -/
def isDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x > f y

/-- The main theorem -/
theorem power_function_decreasing_n (n : ℝ) :
  (isPowerFunction (fun x ↦ (n^2 - n - 1) * x^n)) ∧
  (isDecreasingOn (fun x ↦ (n^2 - n - 1) * x^n)) ↔
  n = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_decreasing_n_l2471_247114


namespace NUMINAMATH_CALUDE_sum_f_negative_l2471_247140

/-- A monotonically decreasing odd function -/
def MonoDecreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x > f y) ∧ (∀ x, f (-x) = -f x)

/-- Theorem: Sum of function values is negative under given conditions -/
theorem sum_f_negative
  (f : ℝ → ℝ)
  (hf : MonoDecreasingOddFunction f)
  (x₁ x₂ x₃ : ℝ)
  (h₁₂ : x₁ + x₂ > 0)
  (h₂₃ : x₂ + x₃ > 0)
  (h₃₁ : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 :=
sorry

end NUMINAMATH_CALUDE_sum_f_negative_l2471_247140


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l2471_247185

theorem parallel_lines_m_values (m : ℝ) : 
  (∃ (x y : ℝ), 2*x + (m+1)*y + 4 = 0) ∧ 
  (∃ (x y : ℝ), m*x + 3*y - 2 = 0) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 2*x₁ + (m+1)*y₁ + 4 = 0 ∧ m*x₂ + 3*y₂ - 2 = 0 → 
    (2 / (m+1) = m / 3)) →
  m = -3 ∨ m = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_m_values_l2471_247185


namespace NUMINAMATH_CALUDE_product_equals_3700_l2471_247138

theorem product_equals_3700 : 4 * 37 * 25 = 3700 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_3700_l2471_247138


namespace NUMINAMATH_CALUDE_quadratic_sum_bounds_l2471_247162

theorem quadratic_sum_bounds (a b c : ℝ) 
  (eq1 : a^2 + a*b + b^2 = 11)
  (eq2 : b^2 + b*c + c^2 = 11) :
  0 ≤ c^2 + c*a + a^2 ∧ c^2 + c*a + a^2 ≤ 44 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_bounds_l2471_247162


namespace NUMINAMATH_CALUDE_systematic_sampling_eighth_group_l2471_247116

/-- Systematic sampling function -/
def systematicSample (totalSize : ℕ) (sampleSize : ℕ) (thirdGroupNumber : ℕ) (groupNumber : ℕ) : ℕ :=
  let groupCount := totalSize / sampleSize
  let commonDifference := groupCount
  thirdGroupNumber + (groupNumber - 3) * commonDifference

/-- Theorem: In a systematic sampling of 840 employees with a sample size of 42,
    if the number drawn in the third group is 44, then the number drawn in the eighth group is 144. -/
theorem systematic_sampling_eighth_group :
  systematicSample 840 42 44 8 = 144 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_eighth_group_l2471_247116


namespace NUMINAMATH_CALUDE_tetrahedron_self_dual_cube_octahedron_dual_dodecahedron_icosahedron_dual_l2471_247196

/-- A polyhedron with faces and vertices -/
structure Polyhedron where
  faces : ℕ
  vertices : ℕ
  face_sides : ℕ
  vertex_valence : ℕ

/-- Duality relation between polyhedra -/
def is_dual (p q : Polyhedron) : Prop :=
  p.faces = q.vertices ∧ p.vertices = q.faces ∧
  p.face_sides = q.vertex_valence ∧ p.vertex_valence = q.face_sides

/-- Self-duality of a polyhedron -/
def is_self_dual (p : Polyhedron) : Prop :=
  is_dual p p

/-- Theorem: Tetrahedron is self-dual -/
theorem tetrahedron_self_dual :
  is_self_dual ⟨4, 4, 3, 3⟩ := by sorry

/-- Theorem: Cube and octahedron are dual -/
theorem cube_octahedron_dual :
  is_dual ⟨6, 8, 4, 3⟩ ⟨8, 6, 3, 4⟩ := by sorry

/-- Theorem: Dodecahedron and icosahedron are dual -/
theorem dodecahedron_icosahedron_dual :
  is_dual ⟨12, 20, 5, 3⟩ ⟨20, 12, 3, 5⟩ := by sorry

end NUMINAMATH_CALUDE_tetrahedron_self_dual_cube_octahedron_dual_dodecahedron_icosahedron_dual_l2471_247196


namespace NUMINAMATH_CALUDE_birdseed_box_content_l2471_247133

/-- The number of grams of seeds in each box of birdseed -/
def grams_per_box : ℕ := 225

/-- The number of boxes Leah bought -/
def boxes_bought : ℕ := 3

/-- The number of boxes Leah already had -/
def boxes_in_pantry : ℕ := 5

/-- The number of grams the parrot eats per week -/
def parrot_consumption : ℕ := 100

/-- The number of grams the cockatiel eats per week -/
def cockatiel_consumption : ℕ := 50

/-- The number of weeks Leah can feed her birds without going back to the store -/
def weeks_of_feeding : ℕ := 12

theorem birdseed_box_content :
  grams_per_box * (boxes_bought + boxes_in_pantry) = 
    (parrot_consumption + cockatiel_consumption) * weeks_of_feeding :=
by
  sorry

#eval grams_per_box

end NUMINAMATH_CALUDE_birdseed_box_content_l2471_247133


namespace NUMINAMATH_CALUDE_ellipse_dot_product_bound_l2471_247153

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

-- Define point M
def M : ℝ × ℝ := (0, 2)

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the function to be bounded
def f (P Q : ℝ × ℝ) : ℝ :=
  dot_product (P.1, P.2) (Q.1, Q.2) + 
  dot_product (P.1 - M.1, P.2 - M.2) (Q.1 - M.1, Q.2 - M.2)

-- Theorem statement
theorem ellipse_dot_product_bound :
  ∀ P Q : ℝ × ℝ, C P.1 P.2 → C Q.1 Q.2 →
  ∃ k : ℝ, P.2 - M.2 = k * (P.1 - M.1) ∧ Q.2 - M.2 = k * (Q.1 - M.1) →
  -20 ≤ f P Q ∧ f P Q ≤ -52/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_bound_l2471_247153


namespace NUMINAMATH_CALUDE_pig_price_calculation_l2471_247169

theorem pig_price_calculation (num_cows : ℕ) (num_pigs : ℕ) (price_per_cow : ℕ) (total_revenue : ℕ) :
  num_cows = 20 →
  num_pigs = 4 * num_cows →
  price_per_cow = 800 →
  total_revenue = 48000 →
  (total_revenue - num_cows * price_per_cow) / num_pigs = 400 := by
  sorry

end NUMINAMATH_CALUDE_pig_price_calculation_l2471_247169


namespace NUMINAMATH_CALUDE_triangle_third_side_l2471_247154

theorem triangle_third_side (a b c : ℕ) : 
  a = 3 → b = 8 → c % 2 = 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) → 
  c ≠ 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_l2471_247154


namespace NUMINAMATH_CALUDE_problem_solution_l2471_247101

theorem problem_solution (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : 
  a^2 - b^2 = 32 ∧ a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2471_247101


namespace NUMINAMATH_CALUDE_cube_paint_theorem_l2471_247174

/-- Represents a cube with edge length n -/
structure Cube (n : ℕ) where
  edge_length : n > 0

/-- Count of unit cubes with one red face -/
def one_face_count (n : ℕ) : ℕ := 6 * (n - 2)^2

/-- Count of unit cubes with two red faces -/
def two_face_count (n : ℕ) : ℕ := 12 * (n - 2)

/-- The main theorem stating the condition for n = 26 -/
theorem cube_paint_theorem (n : ℕ) (c : Cube n) :
  one_face_count n = 12 * two_face_count n ↔ n = 26 := by
  sorry

#check cube_paint_theorem

end NUMINAMATH_CALUDE_cube_paint_theorem_l2471_247174


namespace NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l2471_247157

/-- A regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_150_degrees_has_12_sides :
  ∀ n : ℕ, 
    n > 2 →
    (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_150_degrees_has_12_sides_l2471_247157


namespace NUMINAMATH_CALUDE_complex_division_result_l2471_247160

theorem complex_division_result (z : ℂ) (h : z = 1 - Complex.I * Real.sqrt 3) : 
  4 / z = 1 + Complex.I * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_division_result_l2471_247160


namespace NUMINAMATH_CALUDE_seventy_eighth_ball_is_green_l2471_247188

def ball_color (n : ℕ) : String :=
  match n % 5 with
  | 0 => "violet"
  | 1 => "red"
  | 2 => "yellow"
  | 3 => "green"
  | 4 => "blue"
  | _ => "invalid"  -- This case should never occur

theorem seventy_eighth_ball_is_green : ball_color 78 = "green" := by
  sorry

end NUMINAMATH_CALUDE_seventy_eighth_ball_is_green_l2471_247188


namespace NUMINAMATH_CALUDE_triangle_side_length_l2471_247177

theorem triangle_side_length (A B C : ℝ) (AC : ℝ) (sinA sinB : ℝ) (cosC : ℝ) :
  AC = 3 →
  3 * sinA = 2 * sinB →
  cosC = 1 / 4 →
  ∃ (AB : ℝ), AB = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2471_247177


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l2471_247181

/-- 
Given a tetrahedron with:
- a, b: lengths of two opposite edges
- d: distance between edges a and b
- φ: angle between edges a and b
- V: volume of the tetrahedron

The volume V is equal to (1/6) * a * b * d * sin(φ)
-/
theorem tetrahedron_volume 
  (a b d φ V : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hd : d > 0) 
  (hφ : 0 < φ ∧ φ < π) 
  (hV : V > 0) :
  V = (1/6) * a * b * d * Real.sin φ :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l2471_247181


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2471_247186

theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (3 * x^2 * y = 3 * 3^2 * 15) → (y = 6750) → (x = Real.sqrt 2 / 10) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2471_247186


namespace NUMINAMATH_CALUDE_right_angled_tetrahedron_volume_l2471_247131

/-- A tetrahedron with all faces being right-angled triangles and three edges of length s -/
structure RightAngledTetrahedron (s : ℝ) where
  (s_pos : s > 0)
  (all_faces_right_angled : True)  -- This is a placeholder for the condition
  (three_edges_equal : True)  -- This is a placeholder for the condition

/-- The volume of a right-angled tetrahedron -/
noncomputable def volume (t : RightAngledTetrahedron s) : ℝ :=
  (s^3 * Real.sqrt 2) / 12

/-- Theorem stating the volume of a right-angled tetrahedron -/
theorem right_angled_tetrahedron_volume (s : ℝ) (t : RightAngledTetrahedron s) :
  volume t = (s^3 * Real.sqrt 2) / 12 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_tetrahedron_volume_l2471_247131


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l2471_247176

theorem fraction_equality_sum (P Q : ℚ) : 
  (5 : ℚ) / 7 = P / 63 ∧ (5 : ℚ) / 7 = 70 / Q → P + Q = 143 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l2471_247176


namespace NUMINAMATH_CALUDE_union_of_subsets_l2471_247159

def A : Set ℕ := {1, 3}
def B : Set ℕ := {1, 2, 3}

theorem union_of_subsets :
  A ⊆ B → A ∪ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_subsets_l2471_247159


namespace NUMINAMATH_CALUDE_tino_jellybean_count_l2471_247148

/-- The number of jellybeans each person has -/
structure JellybeanCount where
  tino : ℕ
  lee : ℕ
  arnold : ℕ

/-- The conditions of the jellybean problem -/
def jellybean_conditions (j : JellybeanCount) : Prop :=
  j.tino = j.lee + 24 ∧
  j.arnold = j.lee / 2 ∧
  j.arnold = 5

/-- Theorem stating that under the given conditions, Tino has 34 jellybeans -/
theorem tino_jellybean_count (j : JellybeanCount) 
  (h : jellybean_conditions j) : j.tino = 34 := by
  sorry

end NUMINAMATH_CALUDE_tino_jellybean_count_l2471_247148


namespace NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l2471_247119

/-- Given a quadratic equation x^2 + 2x - k = 0 with real roots, prove that k ≥ -1 -/
theorem quadratic_real_roots_k_range (k : ℝ) :
  (∃ x : ℝ, x^2 + 2*x - k = 0) →
  k ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l2471_247119


namespace NUMINAMATH_CALUDE_fenced_area_calculation_l2471_247124

theorem fenced_area_calculation (length width cutout_side : ℕ) : 
  length = 20 → width = 18 → cutout_side = 4 →
  (length * width) - (cutout_side * cutout_side) = 344 := by
sorry

end NUMINAMATH_CALUDE_fenced_area_calculation_l2471_247124


namespace NUMINAMATH_CALUDE_escalator_time_l2471_247112

/-- Time taken to cover an escalator's length -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) : 
  escalator_speed = 9 →
  person_speed = 3 →
  escalator_length = 200 →
  (escalator_length / (escalator_speed + person_speed)) = 200 / (9 + 3) := by
  sorry

end NUMINAMATH_CALUDE_escalator_time_l2471_247112


namespace NUMINAMATH_CALUDE_two_roots_iff_twenty_l2471_247172

/-- The quadratic equation in x parameterized by a -/
def f (a : ℝ) (x : ℝ) : ℝ := a^2 * (x - 2) + a * (39 - 20*x) + 20

/-- The proposition that the equation has at least two distinct roots -/
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0

theorem two_roots_iff_twenty :
  ∀ a : ℝ, has_two_distinct_roots a ↔ a = 20 := by sorry

end NUMINAMATH_CALUDE_two_roots_iff_twenty_l2471_247172
