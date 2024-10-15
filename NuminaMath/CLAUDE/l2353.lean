import Mathlib

namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2353_235320

theorem fraction_sum_equality (p q r s : ℝ) 
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) + s / (40 - s) = 9) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) + 8 / (40 - s) = 7.6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2353_235320


namespace NUMINAMATH_CALUDE_butterfly_equation_equal_roots_l2353_235393

/-- A quadratic equation ax^2 + bx + c = 0 (a ≠ 0) that satisfies the "butterfly" condition (a - b + c = 0) and has two equal real roots implies a = c. -/
theorem butterfly_equation_equal_roots (a b c : ℝ) (ha : a ≠ 0) :
  (a - b + c = 0) →  -- Butterfly condition
  (b^2 - 4*a*c = 0) →  -- Condition for two equal real roots (discriminant = 0)
  a = c := by
  sorry

end NUMINAMATH_CALUDE_butterfly_equation_equal_roots_l2353_235393


namespace NUMINAMATH_CALUDE_minimal_area_circle_circle_center_on_line_l2353_235336

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

end NUMINAMATH_CALUDE_minimal_area_circle_circle_center_on_line_l2353_235336


namespace NUMINAMATH_CALUDE_cubic_floor_equation_solution_l2353_235363

theorem cubic_floor_equation_solution :
  ∃! x : ℝ, 3 * x^3 - ⌊x⌋ = 3 :=
by
  -- The unique solution is x = ∛(4/3)
  use Real.rpow (4/3) (1/3)
  sorry

end NUMINAMATH_CALUDE_cubic_floor_equation_solution_l2353_235363


namespace NUMINAMATH_CALUDE_closest_point_on_line_l2353_235300

/-- The point on the line y = 2x + 3 that is closest to (2, -1) is (-6/5, 3/5) -/
theorem closest_point_on_line (x y : ℝ) : 
  y = 2 * x + 3 →  -- line equation
  (x + 6/5)^2 + (y - 3/5)^2 ≤ (x - 2)^2 + (y + 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_closest_point_on_line_l2353_235300


namespace NUMINAMATH_CALUDE_angle_A1C1_B1C_is_60_degrees_l2353_235335

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

end NUMINAMATH_CALUDE_angle_A1C1_B1C_is_60_degrees_l2353_235335


namespace NUMINAMATH_CALUDE_budget_equipment_percentage_l2353_235392

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  transportation : ℝ
  research_development : ℝ
  utilities : ℝ
  supplies : ℝ
  salaries : ℝ
  equipment : ℝ

/-- Theorem: Given the budget allocation conditions, the percentage spent on equipment is 4% -/
theorem budget_equipment_percentage
  (budget : BudgetAllocation)
  (h1 : budget.transportation = 15)
  (h2 : budget.research_development = 9)
  (h3 : budget.utilities = 5)
  (h4 : budget.supplies = 2)
  (h5 : budget.salaries = (234 / 360) * 100)
  (h6 : budget.transportation + budget.research_development + budget.utilities +
        budget.supplies + budget.salaries + budget.equipment = 100) :
  budget.equipment = 4 := by
  sorry

end NUMINAMATH_CALUDE_budget_equipment_percentage_l2353_235392


namespace NUMINAMATH_CALUDE_f_definition_l2353_235389

-- Define the function f
def f : ℝ → ℝ := fun x => x^2 - 4

-- State the theorem
theorem f_definition : 
  (∀ x : ℝ, f (x - 2) = x^2 - 4*x) → 
  (∀ x : ℝ, f x = x^2 - 4) := by sorry

end NUMINAMATH_CALUDE_f_definition_l2353_235389


namespace NUMINAMATH_CALUDE_solutions_are_correct_l2353_235374

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 = 49
def equation2 (x : ℝ) : Prop := (2*x + 3)^2 = 4*(2*x + 3)
def equation3 (x : ℝ) : Prop := 2*x^2 + 4*x - 3 = 0
def equation4 (x : ℝ) : Prop := (x + 8)*(x + 1) = -12

-- Theorem stating the solutions are correct
theorem solutions_are_correct :
  (equation1 7 ∧ equation1 (-7)) ∧
  (equation2 (-3/2) ∧ equation2 (1/2)) ∧
  (equation3 ((-2 + Real.sqrt 10) / 2) ∧ equation3 ((-2 - Real.sqrt 10) / 2)) ∧
  (equation4 (-4) ∧ equation4 (-5)) := by sorry

end NUMINAMATH_CALUDE_solutions_are_correct_l2353_235374


namespace NUMINAMATH_CALUDE_plant_height_after_two_years_l2353_235367

/-- The height of a plant after a given number of years -/
def plant_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (4 ^ years)

/-- Theorem: A plant that quadruples its height every year and reaches 256 feet
    after 4 years will be 16 feet tall after 2 years -/
theorem plant_height_after_two_years
  (h : plant_height (plant_height 1 0) 4 = 256) :
  plant_height (plant_height 1 0) 2 = 16 := by
  sorry

#check plant_height_after_two_years

end NUMINAMATH_CALUDE_plant_height_after_two_years_l2353_235367


namespace NUMINAMATH_CALUDE_infinite_divisors_of_power_plus_one_l2353_235311

theorem infinite_divisors_of_power_plus_one (a : ℕ) (h1 : a > 1) (h2 : Even a) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, n ∣ a^n + 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_divisors_of_power_plus_one_l2353_235311


namespace NUMINAMATH_CALUDE_committee_rearrangements_l2353_235366

def word : String := "COMMITTEE"

def vowels : List Char := ['O', 'I', 'E', 'E']
def consonants : List Char := ['C', 'M', 'M', 'T', 'T']

def vowel_arrangements : ℕ := 12
def consonant_m_positions : ℕ := 10
def consonant_t_positions : ℕ := 3

theorem committee_rearrangements :
  (vowel_arrangements * consonant_m_positions * consonant_t_positions) = 360 :=
sorry

end NUMINAMATH_CALUDE_committee_rearrangements_l2353_235366


namespace NUMINAMATH_CALUDE_student_distribution_problem_l2353_235370

/-- The number of ways to distribute n students among k schools,
    where each school must have at least one student. -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  (n - 1).choose (k - 1) * k.factorial

/-- The specific problem statement -/
theorem student_distribution_problem :
  distribute_students 4 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_student_distribution_problem_l2353_235370


namespace NUMINAMATH_CALUDE_factorization_equality_l2353_235388

theorem factorization_equality (a b : ℝ) : a * b^2 - 4 * a * b + 4 * a = a * (b - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2353_235388


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2353_235395

theorem rectangular_field_area (perimeter width length : ℝ) : 
  perimeter = 100 → 
  2 * (length + width) = perimeter → 
  length = 3 * width → 
  length * width = 468.75 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2353_235395


namespace NUMINAMATH_CALUDE_hotel_room_occupancy_l2353_235387

theorem hotel_room_occupancy (num_rooms : ℕ) (towels_per_person : ℕ) (total_towels : ℕ) 
  (h1 : num_rooms = 10)
  (h2 : towels_per_person = 2)
  (h3 : total_towels = 60) :
  total_towels / towels_per_person / num_rooms = 3 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_occupancy_l2353_235387


namespace NUMINAMATH_CALUDE_bella_stamps_count_l2353_235394

/-- Represents the number of stamps of each type Bella bought -/
structure StampCounts where
  snowflake : ℕ
  truck : ℕ
  rose : ℕ
  butterfly : ℕ

/-- Calculates the total number of stamps bought -/
def totalStamps (counts : StampCounts) : ℕ :=
  counts.snowflake + counts.truck + counts.rose + counts.butterfly

/-- Theorem stating the total number of stamps Bella bought -/
theorem bella_stamps_count : ∃ (counts : StampCounts),
  (counts.snowflake : ℚ) * (105 / 100) = 1575 / 100 ∧
  counts.truck = counts.snowflake + 11 ∧
  counts.rose = counts.truck - 17 ∧
  (counts.butterfly : ℚ) = (3 / 2) * counts.rose ∧
  totalStamps counts = 64 := by
  sorry

#check bella_stamps_count

end NUMINAMATH_CALUDE_bella_stamps_count_l2353_235394


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2353_235302

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 8 = 3 / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2353_235302


namespace NUMINAMATH_CALUDE_rectangle_area_reduction_l2353_235356

theorem rectangle_area_reduction (initial_length initial_width : ℝ)
  (reduced_length reduced_width : ℝ) :
  initial_length = 5 →
  initial_width = 7 →
  reduced_length = initial_length - 2 →
  reduced_width = initial_width - 1 →
  reduced_length * initial_width = 21 →
  reduced_length * reduced_width = 18 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_reduction_l2353_235356


namespace NUMINAMATH_CALUDE_distance_and_midpoint_l2353_235329

/-- Given two points in a 2D plane, calculate their distance and midpoint -/
theorem distance_and_midpoint (p1 p2 : ℝ × ℝ) : 
  p1 = (2, 3) → p2 = (5, 9) → 
  (∃ (d : ℝ), d = 3 * Real.sqrt 5 ∧ d = Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)) ∧ 
  (∃ (m : ℝ × ℝ), m = (3.5, 6) ∧ m = ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_distance_and_midpoint_l2353_235329


namespace NUMINAMATH_CALUDE_gcd_2873_1233_l2353_235398

theorem gcd_2873_1233 : Nat.gcd 2873 1233 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2873_1233_l2353_235398


namespace NUMINAMATH_CALUDE_min_value_theorem_l2353_235325

-- Define the lines
def l₁ (m n x y : ℝ) : Prop := m * x + y + n = 0
def l₂ (x y : ℝ) : Prop := x + y - 1 = 0
def l₃ (x y : ℝ) : Prop := 3 * x - y - 7 = 0

-- Theorem statement
theorem min_value_theorem (m n : ℝ) 
  (h1 : ∃ x y : ℝ, l₁ m n x y ∧ l₂ x y ∧ l₃ x y) 
  (h2 : m * n > 0) :
  (1 / m + 2 / n) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2353_235325


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l2353_235312

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

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l2353_235312


namespace NUMINAMATH_CALUDE_total_amount_is_117_l2353_235326

/-- Represents the distribution of money among three parties -/
structure MoneyDistribution where
  x : ℝ  -- Share of x in rupees
  y : ℝ  -- Share of y in rupees
  z : ℝ  -- Share of z in rupees

/-- The conditions of the money distribution problem -/
def satisfies_conditions (d : MoneyDistribution) : Prop :=
  d.y = 27 ∧                  -- y's share is 27 rupees
  d.y = 0.45 * d.x ∧          -- y gets 45 paisa for each rupee x gets
  d.z = 0.5 * d.x             -- z gets 50 paisa for each rupee x gets

/-- The theorem stating the total amount shared -/
theorem total_amount_is_117 (d : MoneyDistribution) 
  (h : satisfies_conditions d) : 
  d.x + d.y + d.z = 117 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_is_117_l2353_235326


namespace NUMINAMATH_CALUDE_equation_solution_l2353_235378

theorem equation_solution : 
  ∃ x : ℝ, (8 * 5.4 - 0.6 * x / 1.2 = 31.000000000000004) ∧ (x = 24.4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2353_235378


namespace NUMINAMATH_CALUDE_quadratic_ratio_l2353_235340

/-- Given a quadratic function f(x) = ax² + bx + c where a > 0,
    if the solution set of f(x) > 0 is (-∞, -2) ∪ (-1, +∞),
    then the ratio a:b:c is 1:3:2 -/
theorem quadratic_ratio (a b c : ℝ) : 
  a > 0 → 
  (∀ x, a * x^2 + b * x + c > 0 ↔ x < -2 ∨ x > -1) → 
  ∃ (k : ℝ), k ≠ 0 ∧ a = k ∧ b = 3*k ∧ c = 2*k :=
sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l2353_235340


namespace NUMINAMATH_CALUDE_concept_laws_theorem_l2353_235332

/-- The probability that exactly M laws are included in the Concept -/
def prob_M_laws_included (K N M : ℕ) (p : ℝ) : ℝ :=
  Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)

/-- The expected number of laws included in the Concept -/
def expected_laws_included (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

theorem concept_laws_theorem (K N M : ℕ) (p : ℝ) 
  (hK : K > 0) (hN : N > 0) (hM : M ≤ K) (hp : 0 ≤ p ∧ p ≤ 1) :
  (prob_M_laws_included K N M p = Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)) ∧
  (expected_laws_included K N p = K * (1 - (1 - p)^N)) := by
  sorry

end NUMINAMATH_CALUDE_concept_laws_theorem_l2353_235332


namespace NUMINAMATH_CALUDE_power_zero_eq_one_negative_two_power_zero_l2353_235314

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

theorem negative_two_power_zero : (-2 : ℝ)^0 = 1 := by sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_negative_two_power_zero_l2353_235314


namespace NUMINAMATH_CALUDE_circus_cages_l2353_235323

theorem circus_cages (n : ℕ) (ways : ℕ) (h1 : n = 6) (h2 : ways = 240) :
  ∃ x : ℕ, x = 3 ∧ (n! / x! = ways) :=
by sorry

end NUMINAMATH_CALUDE_circus_cages_l2353_235323


namespace NUMINAMATH_CALUDE_rotate90_neg4_plus_2i_l2353_235308

def rotate90(z : ℂ) : ℂ := z * Complex.I

theorem rotate90_neg4_plus_2i :
  rotate90 (-4 + 2 * Complex.I) = -2 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_rotate90_neg4_plus_2i_l2353_235308


namespace NUMINAMATH_CALUDE_jack_multiple_is_ten_l2353_235347

/-- The multiple of Michael's current trophies that Jack will have in three years -/
def jack_multiple (michael_current : ℕ) (michael_increase : ℕ) (total_after : ℕ) : ℕ :=
  (total_after - (michael_current + michael_increase)) / michael_current

theorem jack_multiple_is_ten :
  jack_multiple 30 100 430 = 10 := by sorry

end NUMINAMATH_CALUDE_jack_multiple_is_ten_l2353_235347


namespace NUMINAMATH_CALUDE_tangent_slope_xe_pow_x_l2353_235331

open Real

theorem tangent_slope_xe_pow_x (e : ℝ) (h : e = exp 1) :
  let f : ℝ → ℝ := λ x ↦ x * exp x
  let df : ℝ → ℝ := λ x ↦ (1 + x) * exp x
  df 1 = 2 * e :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_xe_pow_x_l2353_235331


namespace NUMINAMATH_CALUDE_simplify_expression_l2353_235357

theorem simplify_expression (y : ℝ) : (3*y)^3 - (4*y)*(y^2) = 23*y^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2353_235357


namespace NUMINAMATH_CALUDE_muffin_mix_buyers_l2353_235317

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

end NUMINAMATH_CALUDE_muffin_mix_buyers_l2353_235317


namespace NUMINAMATH_CALUDE_five_sixths_of_twelve_fifths_l2353_235377

theorem five_sixths_of_twelve_fifths (a b c d : ℚ) : 
  a = 5 ∧ b = 6 ∧ c = 12 ∧ d = 5 → (a / b) * (c / d) = 2 := by
  sorry

end NUMINAMATH_CALUDE_five_sixths_of_twelve_fifths_l2353_235377


namespace NUMINAMATH_CALUDE_fraction_transformation_l2353_235315

theorem fraction_transformation (a b : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (a + 2 : ℚ) / (b^3 : ℚ) = a / (3 * b : ℚ) → a = 1 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l2353_235315


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l2353_235330

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

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l2353_235330


namespace NUMINAMATH_CALUDE_f_range_implies_a_range_l2353_235353

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + a else -a * (x - 2)^2 + 1

/-- The range of f(x) is (-∞, +∞) -/
def has_full_range (a : ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, f a x = y

/-- The range of a is (0, 2] -/
def a_range : Set ℝ := Set.Ioo 0 2 ∪ {2}

theorem f_range_implies_a_range :
  ∀ a : ℝ, has_full_range a → a ∈ a_range :=
sorry

end NUMINAMATH_CALUDE_f_range_implies_a_range_l2353_235353


namespace NUMINAMATH_CALUDE_red_marbles_count_l2353_235386

theorem red_marbles_count (total : ℕ) (blue : ℕ) (orange : ℕ) (red : ℕ) : 
  total = 24 →
  blue = total / 2 →
  orange = 6 →
  total = blue + orange + red →
  red = 6 := by
sorry

end NUMINAMATH_CALUDE_red_marbles_count_l2353_235386


namespace NUMINAMATH_CALUDE_halloween_candy_problem_l2353_235337

theorem halloween_candy_problem (eaten : ℕ) (pile_size : ℕ) (num_piles : ℕ) :
  eaten = 30 →
  pile_size = 8 →
  num_piles = 6 →
  eaten + (pile_size * num_piles) = 78 :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_problem_l2353_235337


namespace NUMINAMATH_CALUDE_marble_distribution_l2353_235361

theorem marble_distribution (y : ℕ) : 
  let first_friend := 2 * y + 2
  let second_friend := y
  let third_friend := 3 * y - 1
  first_friend + second_friend + third_friend = 6 * y + 1 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l2353_235361


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2353_235319

/-- The speed of a boat in still water, given its downstream travel time and distance, and the stream's speed. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 7)
  (h3 : downstream_distance = 147) :
  downstream_distance = (boat_speed + stream_speed) * downstream_time →
  boat_speed = 16 :=
by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2353_235319


namespace NUMINAMATH_CALUDE_additional_track_length_l2353_235321

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

end NUMINAMATH_CALUDE_additional_track_length_l2353_235321


namespace NUMINAMATH_CALUDE_no_factors_of_polynomial_l2353_235369

theorem no_factors_of_polynomial (x : ℝ) : 
  let p (x : ℝ) := x^4 - 4*x^2 + 16
  let f1 (x : ℝ) := x^2 + 4
  let f2 (x : ℝ) := x^2 - 1
  let f3 (x : ℝ) := x^2 + 1
  let f4 (x : ℝ) := x^2 + 3*x + 2
  (∃ (y : ℝ), p x = f1 x * y) = False ∧
  (∃ (y : ℝ), p x = f2 x * y) = False ∧
  (∃ (y : ℝ), p x = f3 x * y) = False ∧
  (∃ (y : ℝ), p x = f4 x * y) = False :=
by sorry

end NUMINAMATH_CALUDE_no_factors_of_polynomial_l2353_235369


namespace NUMINAMATH_CALUDE_largest_fraction_l2353_235365

theorem largest_fraction :
  let a := (1 : ℚ) / 3
  let b := (1 : ℚ) / 4
  let c := (3 : ℚ) / 8
  let d := (5 : ℚ) / 12
  let e := (7 : ℚ) / 24
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l2353_235365


namespace NUMINAMATH_CALUDE_wednesday_saturday_earnings_difference_l2353_235354

def total_earnings : ℝ := 5182.50
def saturday_earnings : ℝ := 2662.50

theorem wednesday_saturday_earnings_difference :
  saturday_earnings - (total_earnings - saturday_earnings) = 142.50 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_saturday_earnings_difference_l2353_235354


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2353_235376

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 2.44

/-- The number of sandwiches -/
def num_sandwiches : ℕ := 2

/-- The cost of a soda in dollars -/
def soda_cost : ℚ := 0.87

/-- The number of sodas -/
def num_sodas : ℕ := 4

/-- The total cost of the order -/
def total_cost : ℚ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem total_cost_calculation :
  total_cost = 8.36 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l2353_235376


namespace NUMINAMATH_CALUDE_probability_above_curve_l2353_235352

-- Define the set of single-digit positive integers
def SingleDigitPos : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

-- Define the condition for (a,c) to be above the curve
def AboveCurve (a c : ℕ) : Prop := ∀ x : ℝ, c > a * x^3 - c * x^2

-- Define the count of valid points
def ValidPointsCount : ℕ := 16

-- Define the total number of possible points
def TotalPointsCount : ℕ := 81

-- State the theorem
theorem probability_above_curve :
  (↑ValidPointsCount / ↑TotalPointsCount : ℚ) = 16/81 :=
sorry

end NUMINAMATH_CALUDE_probability_above_curve_l2353_235352


namespace NUMINAMATH_CALUDE_trinomial_square_l2353_235305

theorem trinomial_square (c : ℚ) : 
  (∃ b y : ℚ, ∀ x : ℚ, 9*x^2 - 21*x + c = (3*x + b + y)^2) → c = 49/4 := by
sorry

end NUMINAMATH_CALUDE_trinomial_square_l2353_235305


namespace NUMINAMATH_CALUDE_shirt_tie_combinations_l2353_235359

/-- The number of possible shirt-and-tie combinations given a set of shirts and ties with restrictions -/
theorem shirt_tie_combinations (total_shirts : ℕ) (total_ties : ℕ) (restricted_shirts : ℕ) (restricted_ties : ℕ) :
  total_shirts = 8 →
  total_ties = 7 →
  restricted_shirts = 3 →
  restricted_ties = 2 →
  total_shirts * total_ties - restricted_shirts * restricted_ties = 50 := by
sorry

end NUMINAMATH_CALUDE_shirt_tie_combinations_l2353_235359


namespace NUMINAMATH_CALUDE_odd_product_probability_l2353_235318

theorem odd_product_probability (n : ℕ) (h : n = 2020) :
  let total := n
  let odds := n / 2
  let p := (odds / total) * ((odds - 1) / (total - 1)) * ((odds - 2) / (total - 2))
  p < 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_odd_product_probability_l2353_235318


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2353_235338

theorem cos_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.cos (α + Real.pi / 3) = -2/3) : 
  Real.cos α = (Real.sqrt 15 - 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2353_235338


namespace NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l2353_235328

theorem cos_pi_fourth_plus_alpha (α : Real) 
  (h : Real.sin (π / 4 - α) = 1 / 3) : 
  Real.cos (π / 4 + α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_fourth_plus_alpha_l2353_235328


namespace NUMINAMATH_CALUDE_ellipse_parabola_configuration_eccentricity_is_half_l2353_235309

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a parabola with focal length c -/
structure Parabola where
  c : ℝ
  h_pos : 0 < c

/-- Configuration of the ellipse and parabola -/
structure Configuration where
  C₁ : Ellipse
  C₂ : Parabola
  h_focus : C₁.a * C₁.a - C₁.b * C₁.b = C₂.c * C₂.c  -- Right focus of C₁ coincides with focus of C₂
  h_center : True  -- Center of C₁ coincides with vertex of C₂ (implied by other conditions)
  h_chord_ratio : (2 * C₂.c) = 4/3 * (2 * C₁.b * C₁.b / C₁.a)  -- |CD| = 4/3 * |AB|
  h_vertices_sum : 2 * C₁.a + C₂.c = 12  -- Sum of distances from vertices of C₁ to directrix of C₂

/-- Main theorem statement -/
theorem ellipse_parabola_configuration (cfg : Configuration) :
  cfg.C₁.a * cfg.C₁.a = 16 ∧ 
  cfg.C₁.b * cfg.C₁.b = 12 ∧ 
  cfg.C₂.c = 2 :=
by sorry

/-- Corollary: Eccentricity of C₁ is 1/2 -/
theorem eccentricity_is_half (cfg : Configuration) :
  Real.sqrt (cfg.C₁.a * cfg.C₁.a - cfg.C₁.b * cfg.C₁.b) / cfg.C₁.a = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_configuration_eccentricity_is_half_l2353_235309


namespace NUMINAMATH_CALUDE_solution_ordered_pair_l2353_235310

theorem solution_ordered_pair : ∃ x y : ℝ, 
  (x + y = (7 - x) + (7 - y)) ∧ 
  (x - y = (x + 1) + (y + 1)) ∧
  x = 8 ∧ y = -1 := by
sorry

end NUMINAMATH_CALUDE_solution_ordered_pair_l2353_235310


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_common_difference_l2353_235342

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

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_common_difference_l2353_235342


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2353_235327

-- Define the conditions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, |x - 2| + |x + 2| > m

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 4 > 0

-- Define the relationship between p and q
theorem p_necessary_not_sufficient_for_q :
  (∃ m : ℝ, p m ∧ ¬q m) ∧ (∀ m : ℝ, q m → p m) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2353_235327


namespace NUMINAMATH_CALUDE_only_solutions_for_equation_l2353_235380

theorem only_solutions_for_equation (x p n : ℕ) : 
  Prime p → 2 * x * (x + 5) = p^n + 3 * (x - 1) → 
  ((x = 2 ∧ p = 5 ∧ n = 2) ∨ (x = 0 ∧ p = 3 ∧ n = 1)) := by
  sorry

end NUMINAMATH_CALUDE_only_solutions_for_equation_l2353_235380


namespace NUMINAMATH_CALUDE_area_of_extended_quadrilateral_l2353_235306

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  EFext : ℝ
  FGext : ℝ
  GHext : ℝ
  HEext : ℝ
  area : ℝ

/-- The area of quadrilateral E'F'G'H' is 57 -/
theorem area_of_extended_quadrilateral (q : ExtendedQuadrilateral) 
  (h1 : q.EF = 5)
  (h2 : q.EFext = 5)
  (h3 : q.FG = 6)
  (h4 : q.FGext = 6)
  (h5 : q.GH = 7)
  (h6 : q.GHext = 7)
  (h7 : q.HE = 10)
  (h8 : q.HEext = 10)
  (h9 : q.area = 15)
  (h10 : q.EF = q.EFext) -- Isosceles triangle condition
  : (q.area + 2 * q.area + 12 : ℝ) = 57 := by
  sorry

end NUMINAMATH_CALUDE_area_of_extended_quadrilateral_l2353_235306


namespace NUMINAMATH_CALUDE_ellipse_equation_l2353_235362

/-- 
Given an ellipse with center at the origin, one focus at (0, √50), 
and a chord intersecting the line y = 3x - 2 with midpoint x-coordinate 1/2, 
prove that the standard equation of the ellipse is x²/25 + y²/75 = 1.
-/
theorem ellipse_equation (F : ℝ × ℝ) (midpoint_x : ℝ) : 
  F = (0, Real.sqrt 50) →
  midpoint_x = 1/2 →
  ∃ (x y : ℝ), x^2/25 + y^2/75 = 1 ∧
    ∃ (x1 y1 x2 y2 : ℝ), 
      (x1^2/25 + y1^2/75 = 1) ∧
      (x2^2/25 + y2^2/75 = 1) ∧
      (y1 = 3*x1 - 2) ∧
      (y2 = 3*x2 - 2) ∧
      ((x1 + x2)/2 = midpoint_x) ∧
      ((y1 + y2)/2 = 3*midpoint_x - 2) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_equation_l2353_235362


namespace NUMINAMATH_CALUDE_triangle_area_ratio_origami_triangle_area_ratio_l2353_235368

/-- The ratio of the areas of two triangles with the same base and different heights -/
theorem triangle_area_ratio (base : ℝ) (height1 height2 : ℝ) (h_base : base > 0) 
  (h_height1 : height1 > 0) (h_height2 : height2 > 0) :
  (1 / 2 * base * height1) / (1 / 2 * base * height2) = height1 / height2 := by
  sorry

/-- The specific ratio of triangle areas for the given problem -/
theorem origami_triangle_area_ratio :
  (1 / 2 * 3 * 6.02) / (1 / 2 * 3 * 2) = 3.01 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_origami_triangle_area_ratio_l2353_235368


namespace NUMINAMATH_CALUDE_cricket_game_overs_l2353_235382

theorem cricket_game_overs (target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) (remaining_overs : ℝ) :
  target = 282 →
  initial_rate = 3.2 →
  required_rate = 6.25 →
  remaining_overs = 40 →
  ∃ x : ℝ, x = 10 ∧ initial_rate * x + required_rate * remaining_overs = target :=
by sorry

end NUMINAMATH_CALUDE_cricket_game_overs_l2353_235382


namespace NUMINAMATH_CALUDE_thirty_six_has_nine_divisors_l2353_235350

/-- The number of positive divisors of 36 -/
def num_divisors_36 : ℕ := sorry

/-- 36 has exactly 9 positive divisors -/
theorem thirty_six_has_nine_divisors : num_divisors_36 = 9 := by sorry

end NUMINAMATH_CALUDE_thirty_six_has_nine_divisors_l2353_235350


namespace NUMINAMATH_CALUDE_square_to_circle_area_ratio_l2353_235324

theorem square_to_circle_area_ratio (s : ℝ) (h : s > 0) : 
  (s^2) / (π * s^2) = 1 / π :=
by sorry

end NUMINAMATH_CALUDE_square_to_circle_area_ratio_l2353_235324


namespace NUMINAMATH_CALUDE_not_54_after_60_operations_l2353_235344

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

end NUMINAMATH_CALUDE_not_54_after_60_operations_l2353_235344


namespace NUMINAMATH_CALUDE_equal_diagonal_polygon_l2353_235322

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

end NUMINAMATH_CALUDE_equal_diagonal_polygon_l2353_235322


namespace NUMINAMATH_CALUDE_mn_length_in_isosceles_triangle_l2353_235346

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

end NUMINAMATH_CALUDE_mn_length_in_isosceles_triangle_l2353_235346


namespace NUMINAMATH_CALUDE_max_value_at_negative_one_l2353_235301

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

end NUMINAMATH_CALUDE_max_value_at_negative_one_l2353_235301


namespace NUMINAMATH_CALUDE_factor_polynomial_l2353_235371

theorem factor_polynomial (x : ℝ) : 75 * x^7 - 270 * x^13 = 15 * x^7 * (5 - 18 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2353_235371


namespace NUMINAMATH_CALUDE_overall_average_l2353_235375

theorem overall_average (n : ℕ) (avg_first : ℝ) (avg_last : ℝ) (middle : ℝ) :
  n = 25 →
  avg_first = 14 →
  avg_last = 17 →
  middle = 78 →
  (avg_first * 12 + middle + avg_last * 12) / n = 18 := by
sorry

end NUMINAMATH_CALUDE_overall_average_l2353_235375


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l2353_235339

/-- A parabola of the form y = ax^2 + 6 is tangent to the line y = 2x + 4 if and only if a = 1/2 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃ x : ℝ, ax^2 + 6 = 2*x + 4 ∧ 
   ∀ y : ℝ, y ≠ x → ay^2 + 6 ≠ 2*y + 4) ↔ 
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l2353_235339


namespace NUMINAMATH_CALUDE_bathroom_square_footage_l2353_235358

/-- Calculates the square footage of a bathroom given the number of tiles and tile size. -/
theorem bathroom_square_footage 
  (width_tiles : ℕ) 
  (length_tiles : ℕ) 
  (tile_size_inches : ℕ) 
  (h1 : width_tiles = 10) 
  (h2 : length_tiles = 20) 
  (h3 : tile_size_inches = 6) : 
  (width_tiles * length_tiles * tile_size_inches^2) / 144 = 50 := by
  sorry

#check bathroom_square_footage

end NUMINAMATH_CALUDE_bathroom_square_footage_l2353_235358


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l2353_235397

theorem power_mod_thirteen : 6^2040 ≡ 1 [ZMOD 13] := by sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l2353_235397


namespace NUMINAMATH_CALUDE_sum_of_squared_medians_l2353_235348

/-- The sum of squares of medians in a triangle with sides 13, 14, and 15 --/
theorem sum_of_squared_medians (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let m_a := (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2)
  let m_b := (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2)
  let m_c := (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)
  m_a^2 + m_b^2 + m_c^2 = 442.5 := by
  sorry

#check sum_of_squared_medians

end NUMINAMATH_CALUDE_sum_of_squared_medians_l2353_235348


namespace NUMINAMATH_CALUDE_koala_fiber_intake_l2353_235333

/-- Given that koalas absorb 40% of the fiber they eat and a particular koala
    absorbed 16 ounces of fiber in one day, prove that it ate 40 ounces of fiber. -/
theorem koala_fiber_intake (absorption_rate : ℝ) (absorbed_amount : ℝ) (total_intake : ℝ) :
  absorption_rate = 0.40 →
  absorbed_amount = 16 →
  absorbed_amount = absorption_rate * total_intake →
  total_intake = 40 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_intake_l2353_235333


namespace NUMINAMATH_CALUDE_book_words_per_page_l2353_235345

theorem book_words_per_page 
  (total_pages : ℕ) 
  (max_words_per_page : ℕ) 
  (total_words_mod : ℕ) 
  (modulus : ℕ) 
  (h1 : total_pages = 180) 
  (h2 : max_words_per_page = 150) 
  (h3 : total_words_mod = 203) 
  (h4 : modulus = 229) 
  (h5 : ∃ (words_per_page : ℕ), 
    words_per_page ≤ max_words_per_page ∧ 
    (total_pages * words_per_page) % modulus = total_words_mod) :
  ∃ (words_per_page : ℕ), words_per_page = 94 ∧ 
    words_per_page ≤ max_words_per_page ∧ 
    (total_pages * words_per_page) % modulus = total_words_mod :=
sorry

end NUMINAMATH_CALUDE_book_words_per_page_l2353_235345


namespace NUMINAMATH_CALUDE_integer_solution_theorem_l2353_235304

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

end NUMINAMATH_CALUDE_integer_solution_theorem_l2353_235304


namespace NUMINAMATH_CALUDE_binomial_coefficient_identity_a_l2353_235303

theorem binomial_coefficient_identity_a (r m k : ℕ) (h1 : k ≤ m) (h2 : m ≤ r) :
  Nat.choose r m * Nat.choose m k = Nat.choose r k * Nat.choose (r - k) (m - k) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_identity_a_l2353_235303


namespace NUMINAMATH_CALUDE_plan_y_more_economical_min_megabytes_optimal_l2353_235391

/-- Represents the cost of an internet plan in cents -/
def PlanCost (initial_fee : ℕ) (rate : ℕ) (megabytes : ℕ) : ℕ :=
  initial_fee * 100 + rate * megabytes

/-- The minimum number of megabytes for Plan Y to be more economical than Plan X -/
def MinMegabytes : ℕ := 501

theorem plan_y_more_economical :
  ∀ m : ℕ, m ≥ MinMegabytes →
    PlanCost 25 10 m < PlanCost 0 15 m :=
by
  sorry

theorem min_megabytes_optimal :
  ∀ m : ℕ, m < MinMegabytes →
    PlanCost 0 15 m ≤ PlanCost 25 10 m :=
by
  sorry

end NUMINAMATH_CALUDE_plan_y_more_economical_min_megabytes_optimal_l2353_235391


namespace NUMINAMATH_CALUDE_aitana_jayda_spending_l2353_235316

theorem aitana_jayda_spending (jayda_spent : ℚ) (total_spent : ℚ) 
  (h1 : jayda_spent = 400)
  (h2 : total_spent = 960) : 
  (total_spent - jayda_spent) / jayda_spent = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_aitana_jayda_spending_l2353_235316


namespace NUMINAMATH_CALUDE_professor_seating_theorem_l2353_235349

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

end NUMINAMATH_CALUDE_professor_seating_theorem_l2353_235349


namespace NUMINAMATH_CALUDE_mice_on_bottom_path_l2353_235381

/-- Represents the number of mice in each house --/
structure MouseDistribution where
  left : ℕ
  top : ℕ
  right : ℕ

/-- The problem setup --/
def initial_distribution : MouseDistribution := ⟨8, 3, 7⟩
def final_distribution : MouseDistribution := ⟨5, 4, 9⟩

/-- The theorem to prove --/
theorem mice_on_bottom_path :
  let bottom_path_mice := 
    (initial_distribution.left + initial_distribution.right) -
    (final_distribution.left + final_distribution.right)
  bottom_path_mice = 11 := by
  sorry


end NUMINAMATH_CALUDE_mice_on_bottom_path_l2353_235381


namespace NUMINAMATH_CALUDE_smallest_k_property_l2353_235385

theorem smallest_k_property : ∃ k : ℝ, k = 2 ∧
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → 
    (a ≤ k ∨ b ≤ k ∨ (5 / a^2 + 6 / b^3) ≤ k)) ∧
  (∀ k' : ℝ, k' < k →
    ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧
      a > k' ∧ b > k' ∧ (5 / a^2 + 6 / b^3) > k') :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_property_l2353_235385


namespace NUMINAMATH_CALUDE_d_in_N_l2353_235383

def M : Set ℤ := {x | ∃ n : ℤ, x = 3 * n}
def N : Set ℤ := {x | ∃ n : ℤ, |x| = 3 * n + 1}
def P : Set ℤ := {x | ∃ n : ℤ, x = 3 * n - 1}

theorem d_in_N (a b c : ℤ) (ha : a ∈ M) (hb : b ∈ N) (hc : c ∈ P) :
  (a - b + c) ∈ N := by
  sorry

end NUMINAMATH_CALUDE_d_in_N_l2353_235383


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2353_235351

theorem min_value_quadratic (x : ℝ) :
  ∃ (y_min : ℝ), ∀ (y : ℝ), y = 4 * x^2 + 8 * x + 12 → y ≥ y_min ∧ ∃ (x_0 : ℝ), 4 * x_0^2 + 8 * x_0 + 12 = y_min :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2353_235351


namespace NUMINAMATH_CALUDE_work_completion_time_l2353_235355

-- Define the work rates and time worked by Y
def x_rate : ℚ := 1 / 24
def y_rate : ℚ := 1 / 16
def y_days_worked : ℕ := 10

-- Define the theorem
theorem work_completion_time :
  let total_work : ℚ := 1
  let work_done_by_y : ℚ := y_rate * y_days_worked
  let remaining_work : ℚ := total_work - work_done_by_y
  let days_needed_by_x : ℚ := remaining_work / x_rate
  days_needed_by_x = 9 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2353_235355


namespace NUMINAMATH_CALUDE_inscribed_polygon_radius_l2353_235399

/-- A 12-sided convex polygon inscribed in a circle -/
structure InscribedPolygon where
  /-- The number of sides of the polygon -/
  sides : ℕ
  /-- The number of sides with length √2 -/
  short_sides : ℕ
  /-- The number of sides with length √24 -/
  long_sides : ℕ
  /-- The length of the short sides -/
  short_length : ℝ
  /-- The length of the long sides -/
  long_length : ℝ
  /-- Condition: The polygon has 12 sides -/
  sides_eq : sides = 12
  /-- Condition: There are 6 short sides -/
  short_sides_eq : short_sides = 6
  /-- Condition: There are 6 long sides -/
  long_sides_eq : long_sides = 6
  /-- Condition: The short sides have length √2 -/
  short_length_eq : short_length = Real.sqrt 2
  /-- Condition: The long sides have length √24 -/
  long_length_eq : long_length = Real.sqrt 24

/-- The theorem stating that the radius of the circle is 4√2 -/
theorem inscribed_polygon_radius (p : InscribedPolygon) : 
  ∃ (r : ℝ), r = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_polygon_radius_l2353_235399


namespace NUMINAMATH_CALUDE_identify_counterfeit_l2353_235373

/-- Represents a coin with its denomination and weight -/
structure Coin where
  denomination : Nat
  weight : Nat

/-- Represents the state of a balance scale -/
inductive Balance
  | Left
  | Right
  | Equal

/-- Represents a weighing operation on the balance scale -/
def weigh (left right : List Coin) : Balance :=
  sorry

/-- Represents the set of coins -/
def coins : List Coin :=
  [⟨1, 1⟩, ⟨2, 2⟩, ⟨3, 3⟩, ⟨5, 5⟩]

/-- Represents the counterfeit coin -/
def counterfeit : Coin :=
  sorry

/-- The main theorem stating that the counterfeit coin can be identified in two weighings -/
theorem identify_counterfeit :
  ∃ (weighing1 weighing2 : List Coin × List Coin),
    let result1 := weigh weighing1.1 weighing1.2
    let result2 := weigh weighing2.1 weighing2.2
    ∃ (identified : Coin), identified = counterfeit :=
  sorry

end NUMINAMATH_CALUDE_identify_counterfeit_l2353_235373


namespace NUMINAMATH_CALUDE_exists_carmichael_number_l2353_235307

theorem exists_carmichael_number : 
  ∃ n : ℕ, 
    n > 1 ∧ 
    ¬(Nat.Prime n) ∧ 
    ∀ a : ℤ, (a^n) % n = a % n :=
by sorry

end NUMINAMATH_CALUDE_exists_carmichael_number_l2353_235307


namespace NUMINAMATH_CALUDE_probability_different_families_l2353_235341

/-- The number of families -/
def num_families : ℕ := 6

/-- The number of members in each family -/
def members_per_family : ℕ := 3

/-- The total number of people -/
def total_people : ℕ := num_families * members_per_family

/-- The size of each group in the game -/
def group_size : ℕ := 3

/-- The probability of selecting 3 people from different families -/
theorem probability_different_families : 
  (Nat.choose num_families group_size * (members_per_family ^ group_size)) / 
  (Nat.choose total_people group_size) = 45 / 68 := by sorry

end NUMINAMATH_CALUDE_probability_different_families_l2353_235341


namespace NUMINAMATH_CALUDE_problem_statement_l2353_235334

theorem problem_statement (a b : ℝ) (h : |a + 1| + (b - 2023)^2 = 0) : a^b = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2353_235334


namespace NUMINAMATH_CALUDE_painting_time_theorem_l2353_235343

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

end NUMINAMATH_CALUDE_painting_time_theorem_l2353_235343


namespace NUMINAMATH_CALUDE_transylvanian_must_be_rational_l2353_235379

/-- Represents the state of a person's mind -/
inductive MindState
| Rational
| Lost

/-- Represents a person -/
structure Person where
  mindState : MindState

/-- Represents the claim made by a person -/
def claim (p : Person) : Prop :=
  p.mindState = MindState.Lost

/-- A person with a lost mind cannot make a truthful claim about their condition -/
axiom lost_mind_cannot_claim : ∀ (p : Person), p.mindState = MindState.Lost → ¬(claim p)

/-- The theorem to be proved -/
theorem transylvanian_must_be_rational (p : Person) (makes_claim : claim p) :
  p.mindState = MindState.Rational := by
  sorry

end NUMINAMATH_CALUDE_transylvanian_must_be_rational_l2353_235379


namespace NUMINAMATH_CALUDE_correlation_identification_l2353_235390

-- Define the concept of a relationship
def Relationship : Type := Unit

-- Define specific relationships
def age_wealth : Relationship := ()
def curve_coordinates : Relationship := ()
def apple_production_climate : Relationship := ()
def tree_diameter_height : Relationship := ()

-- Define the property of being correlational
def is_correlational : Relationship → Prop := sorry

-- Define the property of being functional
def is_functional : Relationship → Prop := sorry

-- State that functional relationships are not correlational
axiom functional_not_correlational : 
  ∀ (r : Relationship), is_functional r → ¬is_correlational r

-- State the theorem
theorem correlation_identification :
  is_correlational age_wealth ∧
  is_correlational apple_production_climate ∧
  is_correlational tree_diameter_height ∧
  is_functional curve_coordinates :=
sorry

end NUMINAMATH_CALUDE_correlation_identification_l2353_235390


namespace NUMINAMATH_CALUDE_faye_initial_giveaway_l2353_235384

/-- The number of coloring books Faye bought initially -/
def initial_books : ℝ := 48.0

/-- The number of coloring books Faye gave away after the initial giveaway -/
def additional_giveaway : ℝ := 3.0

/-- The number of coloring books Faye has left -/
def remaining_books : ℝ := 11.0

/-- The number of coloring books Faye gave away initially -/
def initial_giveaway : ℝ := initial_books - additional_giveaway - remaining_books

theorem faye_initial_giveaway : initial_giveaway = 34.0 := by
  sorry

end NUMINAMATH_CALUDE_faye_initial_giveaway_l2353_235384


namespace NUMINAMATH_CALUDE_rabbit_cat_age_ratio_l2353_235313

/-- Given the ages of a cat, dog, and rabbit, prove the ratio of rabbit's age to cat's age --/
theorem rabbit_cat_age_ratio 
  (cat_age : ℕ) 
  (dog_age : ℕ) 
  (rabbit_age : ℕ) 
  (h1 : cat_age = 8) 
  (h2 : dog_age = 12) 
  (h3 : rabbit_age * 3 = dog_age) : 
  (rabbit_age : ℚ) / cat_age = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_cat_age_ratio_l2353_235313


namespace NUMINAMATH_CALUDE_sin_product_equals_one_thirty_second_l2353_235360

theorem sin_product_equals_one_thirty_second :
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * Real.sin (72 * π / 180) * Real.sin (84 * π / 180) = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_thirty_second_l2353_235360


namespace NUMINAMATH_CALUDE_dipolia_puzzle_solution_l2353_235396

-- Define the types of people in Dipolia
inductive PersonType
| Knight
| Liar

-- Define the possible meanings of "Irgo"
inductive IrgoMeaning
| Yes
| No

-- Define the properties of knights and liars
def always_truthful (p : PersonType) : Prop :=
  p = PersonType.Knight

def always_lies (p : PersonType) : Prop :=
  p = PersonType.Liar

-- Define the scenario
structure DipoliaScenario where
  inhabitant_type : PersonType
  irgo_meaning : IrgoMeaning
  guide_truthful : Prop

-- Theorem statement
theorem dipolia_puzzle_solution (scenario : DipoliaScenario) :
  scenario.guide_truthful →
  (scenario.irgo_meaning = IrgoMeaning.Yes ∧ scenario.inhabitant_type = PersonType.Liar) :=
by sorry

end NUMINAMATH_CALUDE_dipolia_puzzle_solution_l2353_235396


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l2353_235372

theorem modular_arithmetic_problem :
  ∃ (a b : ℕ), 
    (7 * a) % 60 = 1 ∧ 
    (13 * b) % 60 = 1 ∧ 
    ((3 * a + 9 * b) % 60 : ℕ) = 42 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l2353_235372


namespace NUMINAMATH_CALUDE_average_weight_increase_l2353_235364

/-- Proves that the increase in average weight is 2.5 kg when a person weighing 65 kg
    in a group of 6 is replaced by a person weighing 80 kg. -/
theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 6 →
  old_weight = 65 →
  new_weight = 80 →
  (new_weight - old_weight) / initial_count = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2353_235364
