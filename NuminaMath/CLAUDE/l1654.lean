import Mathlib

namespace NUMINAMATH_CALUDE_remy_water_usage_l1654_165456

/-- Proves that Remy used 25 gallons of water given the conditions of the problem. -/
theorem remy_water_usage (roman : ℕ) (remy : ℕ) : 
  remy = 3 * roman + 1 →  -- Condition 1
  roman + remy = 33 →     -- Condition 2
  remy = 25 := by
sorry

end NUMINAMATH_CALUDE_remy_water_usage_l1654_165456


namespace NUMINAMATH_CALUDE_min_value_theorem_l1654_165437

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + y = 5 * x * y) :
  4 * x + 3 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 * x₀ + y₀ = 5 * x₀ * y₀ ∧ 4 * x₀ + 3 * y₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1654_165437


namespace NUMINAMATH_CALUDE_officer_selection_ways_l1654_165426

def group_members : Nat := 5
def officer_positions : Nat := 4

theorem officer_selection_ways : 
  (group_members.choose officer_positions) * (officer_positions.factorial) = 120 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_ways_l1654_165426


namespace NUMINAMATH_CALUDE_no_periodic_sum_with_periods_2_and_pi_div_2_l1654_165415

/-- A function is periodic if it takes at least two different values and there exists a positive period. -/
def Periodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ ∃ p > 0, ∀ x, f (x + p) = f x

/-- The period of a function is a positive real number p such that f(x + p) = f(x) for all x. -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

theorem no_periodic_sum_with_periods_2_and_pi_div_2 :
  ¬ ∃ (g h : ℝ → ℝ),
    Periodic g ∧ Periodic h ∧ IsPeriod g 2 ∧ IsPeriod h (π / 2) ∧ Periodic (g + h) :=
sorry

end NUMINAMATH_CALUDE_no_periodic_sum_with_periods_2_and_pi_div_2_l1654_165415


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1654_165468

def A : Set ℕ := {x | (x + 4) * (x - 5) ≤ 0}
def B : Set ℕ := {x | x < 2}
def U : Set ℕ := Set.univ

theorem intersection_complement_equality :
  A ∩ (U \ B) = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1654_165468


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l1654_165413

/-- Given a square with perimeter 60 and a right triangle with one leg 20,
    if their areas are equal, then the other leg of the triangle is 22.5 -/
theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_leg : ℝ) (other_leg : ℝ) :
  square_perimeter = 60 →
  triangle_leg = 20 →
  (square_perimeter / 4) ^ 2 = (triangle_leg * other_leg) / 2 →
  other_leg = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l1654_165413


namespace NUMINAMATH_CALUDE_second_month_sale_l1654_165487

def average_sale : ℕ := 6600
def num_months : ℕ := 6
def sale_month1 : ℕ := 6435
def sale_month3 : ℕ := 7230
def sale_month4 : ℕ := 6562
def sale_month5 : ℕ := 6855
def sale_month6 : ℕ := 5591

theorem second_month_sale :
  ∃ (sale_month2 : ℕ),
    sale_month2 = average_sale * num_months - (sale_month1 + sale_month3 + sale_month4 + sale_month5 + sale_month6) ∧
    sale_month2 = 6927 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l1654_165487


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1654_165431

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 4 : ℝ) * Real.sqrt (4 * t.leg^2 - t.base^2 : ℝ)

/-- Theorem stating the minimum common perimeter of two specific isosceles triangles -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1.base * 8 = t2.base * 9 ∧
    t1 ≠ t2 ∧
    area t1 = area t2 ∧
    perimeter t1 = perimeter t2 ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1.base * 8 = s2.base * 9 →
      s1 ≠ s2 →
      area s1 = area s2 →
      perimeter s1 = perimeter s2 →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 960 :=
  sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l1654_165431


namespace NUMINAMATH_CALUDE_square_perimeter_division_l1654_165409

/-- Represents a division of a square's perimeter into two groups of segments -/
structure SquarePerimeterDivision where
  side_length : ℝ
  group1_count : ℕ
  group2_count : ℕ
  group1_segment_length : ℝ
  group2_segment_length : ℝ

/-- Checks if the given division is valid for the square's perimeter -/
def is_valid_division (d : SquarePerimeterDivision) : Prop :=
  d.group1_count * d.group1_segment_length + d.group2_count * d.group2_segment_length = 4 * d.side_length

/-- The specific division of a square with side length 20 cm into 3 and 4 segments -/
def specific_division : SquarePerimeterDivision :=
  { side_length := 20
  , group1_count := 3
  , group2_count := 4
  , group1_segment_length := 20
  , group2_segment_length := 5 }

theorem square_perimeter_division :
  is_valid_division specific_division ∧
  specific_division.group1_segment_length = 20 ∧
  specific_division.group2_segment_length = 5 := by
  sorry

#check square_perimeter_division

end NUMINAMATH_CALUDE_square_perimeter_division_l1654_165409


namespace NUMINAMATH_CALUDE_geometric_mean_of_1_and_9_l1654_165484

def geometric_mean (a b : ℝ) : Set ℝ :=
  {x | x ^ 2 = a * b}

theorem geometric_mean_of_1_and_9 :
  geometric_mean 1 9 = {3, -3} := by sorry

end NUMINAMATH_CALUDE_geometric_mean_of_1_and_9_l1654_165484


namespace NUMINAMATH_CALUDE_sugar_problem_solution_l1654_165457

def sugar_problem (sugar_at_home : ℕ) (bags_bought : ℕ) (dozens : ℕ) 
  (sugar_per_dozen_batter : ℚ) (sugar_per_dozen_frosting : ℚ) : Prop :=
  ∃ (sugar_per_bag : ℕ),
    sugar_at_home = 3 ∧
    bags_bought = 2 ∧
    dozens = 5 ∧
    sugar_per_dozen_batter = 1 ∧
    sugar_per_dozen_frosting = 2 ∧
    sugar_at_home + bags_bought * sugar_per_bag = 
      dozens * (sugar_per_dozen_batter + sugar_per_dozen_frosting) ∧
    sugar_per_bag = 6

theorem sugar_problem_solution :
  sugar_problem 3 2 5 1 2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_problem_solution_l1654_165457


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l1654_165489

open Real

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (log x) / x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := (1 - log x) / (x^2)

theorem tangent_parallel_to_x_axis :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ f_derivative x₀ = 0 → f x₀ = 1/Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l1654_165489


namespace NUMINAMATH_CALUDE_soccer_team_average_goals_l1654_165450

/-- The average number of goals scored by the soccer team per game -/
def team_average_goals (carter_goals shelby_goals judah_goals : ℝ) : ℝ :=
  carter_goals + shelby_goals + judah_goals

/-- Theorem stating the average total number of goals scored by the team per game -/
theorem soccer_team_average_goals :
  ∃ (carter_goals shelby_goals judah_goals : ℝ),
    carter_goals = 4 ∧
    shelby_goals = carter_goals / 2 ∧
    judah_goals = 2 * shelby_goals - 3 ∧
    team_average_goals carter_goals shelby_goals judah_goals = 7 := by
  sorry


end NUMINAMATH_CALUDE_soccer_team_average_goals_l1654_165450


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l1654_165428

theorem floor_ceiling_sum (x y : ℝ) (hx : 1 < x ∧ x < 2) (hy : 3 < y ∧ y < 4) :
  ⌊x⌋ + ⌈y⌉ = 5 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l1654_165428


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l1654_165434

/-- The nth term of a geometric sequence -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The 10th term of the specific geometric sequence -/
theorem tenth_term_of_sequence :
  geometric_sequence 5 (3/4) 10 = 98415/262144 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l1654_165434


namespace NUMINAMATH_CALUDE_common_tangents_count_l1654_165498

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 10*y + 13 = 0

-- Define a function to count common tangents
def count_common_tangents (C1 C2 : ℝ → ℝ → Prop) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents C1 C2 = 1 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l1654_165498


namespace NUMINAMATH_CALUDE_angelina_walking_speed_l1654_165477

/-- Angelina's walking problem -/
theorem angelina_walking_speed 
  (home_to_grocery : ℝ) 
  (grocery_to_gym : ℝ) 
  (time_difference : ℝ) 
  (h1 : home_to_grocery = 100) 
  (h2 : grocery_to_gym = 180) 
  (h3 : time_difference = 40) :
  let v := home_to_grocery / ((grocery_to_gym / 2) / time_difference + home_to_grocery)
  2 * v = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_angelina_walking_speed_l1654_165477


namespace NUMINAMATH_CALUDE_parabola_properties_l1654_165469

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_properties (a b c m : ℝ) :
  -- Conditions
  (∃ C : ℝ, C > 0 ∧ parabola a b c C = 0) →  -- Intersects positive y-axis
  (parabola a b c 1 = 2) →                   -- Vertex at (1, 2)
  (parabola a b c (-1) = m) →                -- Passes through (-1, m)
  (m < 0) →                                  -- m is negative
  -- Conclusions
  (2 * a + b = 0) ∧                          -- Conclusion ②
  (-2 < a ∧ a < -1/2) ∧                      -- Conclusion ③
  (∀ n : ℝ, (∀ x : ℝ, parabola a b c x ≠ n) → n > 2) ∧  -- Conclusion ④
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ parabola a b c x₁ = 1 ∧ parabola a b c x₂ = 1 ∧ x₁ + x₂ = 2)  -- Conclusion ⑥
  := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1654_165469


namespace NUMINAMATH_CALUDE_roots_equation_m_value_l1654_165496

theorem roots_equation_m_value (α : ℝ) (m : ℝ) : 
  (∀ x, x^2 + 3*x + m = 0 ↔ x = 1/Real.cos α ∨ x = Real.tan α) →
  m = 20/9 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_m_value_l1654_165496


namespace NUMINAMATH_CALUDE_fourth_month_sale_l1654_165439

def average_sale : ℕ := 6600
def num_months : ℕ := 6
def sale_month1 : ℕ := 6435
def sale_month2 : ℕ := 6927
def sale_month3 : ℕ := 6855
def sale_month5 : ℕ := 6562
def sale_month6 : ℕ := 5591

theorem fourth_month_sale (x : ℕ) : 
  (sale_month1 + sale_month2 + sale_month3 + x + sale_month5 + sale_month6) / num_months = average_sale →
  x = 7230 := by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l1654_165439


namespace NUMINAMATH_CALUDE_max_value_theorem_l1654_165405

theorem max_value_theorem (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_sum : a + b + c + d ≤ 4) : 
  (a^2 * (a + b))^(1/4) + (b^2 * (b + c))^(1/4) + 
  (c^2 * (c + d))^(1/4) + (d^2 * (d + a))^(1/4) ≤ 4 * 2^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1654_165405


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_l1654_165492

/-- Given sets A and B, prove that their intersection is empty if and only if a is in the specified range -/
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  let A := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}
  let B := {x : ℝ | x < -1 ∨ x > 5}
  (A ∩ B = ∅) ↔ (a > 3 ∨ (-1/2 ≤ a ∧ a ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_l1654_165492


namespace NUMINAMATH_CALUDE_complex_equality_problem_l1654_165417

theorem complex_equality_problem (x y : ℝ) 
  (h : (x + y : ℂ) + Complex.I = 3*x + (x - y)*Complex.I) : 
  x = -1 ∧ y = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_problem_l1654_165417


namespace NUMINAMATH_CALUDE_apartment_333_on_third_floor_l1654_165447

/-- Represents a building with apartments -/
structure Building where
  floors : ℕ
  entrances : ℕ
  apartments_per_floor : ℕ

/-- Calculates the total number of apartments in the building -/
def total_apartments (b : Building) : ℕ :=
  b.floors * b.entrances * b.apartments_per_floor

/-- Calculates the floor number for a given apartment number -/
def apartment_floor (b : Building) (apartment_number : ℕ) : ℕ :=
  ((apartment_number - 1) / b.apartments_per_floor) % b.floors + 1

/-- The specific building described in the problem -/
def problem_building : Building :=
  { floors := 9
  , entrances := 10
  , apartments_per_floor := 4 }

theorem apartment_333_on_third_floor :
  apartment_floor problem_building 333 = 3 := by
  sorry

#eval apartment_floor problem_building 333

end NUMINAMATH_CALUDE_apartment_333_on_third_floor_l1654_165447


namespace NUMINAMATH_CALUDE_product_of_ab_l1654_165429

theorem product_of_ab (a b : ℝ) (h1 : a^5 * b^8 = 12) (h2 : a^8 * b^13 = 18) : a * b = 128/3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_ab_l1654_165429


namespace NUMINAMATH_CALUDE_power_3_2023_mod_5_l1654_165460

theorem power_3_2023_mod_5 : 3^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_3_2023_mod_5_l1654_165460


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_bounds_l1654_165499

theorem quadratic_roots_sum_squares_bounds (k : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 - (k - 2) * x₁ + (k^2 + 3 * k + 5) = 0 →
  x₂^2 - (k - 2) * x₂ + (k^2 + 3 * k + 5) = 0 →
  x₁ ≠ x₂ →
  ∃ (y : ℝ), y = x₁^2 + x₂^2 ∧ y ≤ 18 ∧ y ≥ 50/9 ∧
  (∃ (k₁ : ℝ), x₁^2 - (k₁ - 2) * x₁ + (k₁^2 + 3 * k₁ + 5) = 0 ∧
               x₂^2 - (k₁ - 2) * x₂ + (k₁^2 + 3 * k₁ + 5) = 0 ∧
               x₁^2 + x₂^2 = 18) ∧
  (∃ (k₂ : ℝ), x₁^2 - (k₂ - 2) * x₁ + (k₂^2 + 3 * k₂ + 5) = 0 ∧
               x₂^2 - (k₂ - 2) * x₂ + (k₂^2 + 3 * k₂ + 5) = 0 ∧
               x₁^2 + x₂^2 = 50/9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_bounds_l1654_165499


namespace NUMINAMATH_CALUDE_short_bingo_first_column_count_l1654_165459

def short_bingo_first_column_possibilities : ℕ := 360360

theorem short_bingo_first_column_count :
  (Finset.range 15).card.choose 5 = short_bingo_first_column_possibilities :=
by sorry

end NUMINAMATH_CALUDE_short_bingo_first_column_count_l1654_165459


namespace NUMINAMATH_CALUDE_bike_ride_speed_l1654_165402

theorem bike_ride_speed (joann_speed joann_time fran_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 2.5) :
  (joann_speed * joann_time) / fran_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_bike_ride_speed_l1654_165402


namespace NUMINAMATH_CALUDE_sequence_sum_equals_5923_l1654_165401

def arithmetic_sum (a1 l1 d : ℤ) : ℤ :=
  let n := (l1 - a1) / d + 1
  n * (a1 + l1) / 2

def sequence_sum : ℤ :=
  3 * (arithmetic_sum 45 93 2) + 2 * (arithmetic_sum (-4) 38 2)

theorem sequence_sum_equals_5923 : sequence_sum = 5923 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_5923_l1654_165401


namespace NUMINAMATH_CALUDE_complement_of_A_l1654_165462

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

theorem complement_of_A : 
  Set.compl A = Set.Icc (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1654_165462


namespace NUMINAMATH_CALUDE_second_number_is_13_l1654_165425

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : ℕ        -- Total number of items
  sampleSize : ℕ   -- Number of items to be sampled
  first : ℕ        -- First number drawn

/-- Calculates the nth number in a systematic sample -/
def nthNumber (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first + (s.total / s.sampleSize) * (n - 1)

/-- Theorem stating that the second number drawn is 13 -/
theorem second_number_is_13 (s : SystematicSample) 
  (h1 : s.total = 500) 
  (h2 : s.sampleSize = 50) 
  (h3 : s.first = 3) : 
  nthNumber s 2 = 13 := by
  sorry

#check second_number_is_13

end NUMINAMATH_CALUDE_second_number_is_13_l1654_165425


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1654_165406

-- Define the conditions
def condition_p (m : ℝ) : Prop := ∃ x : ℝ, |x - 1| + |x - 3| < m

def condition_q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (7 - 3*m)^x > (7 - 3*m)^y

-- State the theorem
theorem necessary_but_not_sufficient :
  (∀ m : ℝ, condition_q m → condition_p m) ∧
  (∃ m : ℝ, condition_p m ∧ ¬condition_q m) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1654_165406


namespace NUMINAMATH_CALUDE_trajectory_of_point_l1654_165452

/-- The trajectory of a point M(x, y) satisfying the given distance condition -/
theorem trajectory_of_point (x y : ℝ) :
  (((x - 2)^2 + y^2).sqrt = |x + 3| - 1) →
  (y^2 = 8 * (x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_of_point_l1654_165452


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l1654_165416

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x^2 + ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 1

theorem even_function_implies_a_zero :
  ∀ a : ℝ, IsEven (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l1654_165416


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l1654_165488

theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter : ℝ := outer_cube_edge
  let inner_cube_diagonal : ℝ := sphere_diameter
  let inner_cube_edge : ℝ := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume : ℝ := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l1654_165488


namespace NUMINAMATH_CALUDE_tim_pencil_count_l1654_165495

/-- Given that Tyrah has six times as many pencils as Sarah, Tim has eight times as many pencils as Sarah, and Tyrah has 12 pencils, prove that Tim has 16 pencils. -/
theorem tim_pencil_count (sarah_pencils : ℕ) 
  (h1 : 6 * sarah_pencils = 12)  -- Tyrah has six times as many pencils as Sarah and has 12 pencils
  (h2 : 8 * sarah_pencils = tim_pencils) : -- Tim has eight times as many pencils as Sarah
  tim_pencils = 16 := by
  sorry

end NUMINAMATH_CALUDE_tim_pencil_count_l1654_165495


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1654_165446

theorem inequality_solution_range (k : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + k*x - 1 > 0) → k > -3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1654_165446


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l1654_165404

/-- Proves that the ratio of Rahul's age to Deepak's age is 4:3 -/
theorem rahul_deepak_age_ratio : 
  ∀ (rahul_age deepak_age : ℕ),
  deepak_age = 12 →
  rahul_age + 10 = 26 →
  (rahul_age : ℚ) / (deepak_age : ℚ) = 4 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l1654_165404


namespace NUMINAMATH_CALUDE_line_points_l1654_165443

-- Define the points
def p1 : ℝ × ℝ := (8, 10)
def p2 : ℝ × ℝ := (2, -2)

-- Define the function to check if a point is on the line
def is_on_line (p : ℝ × ℝ) : Prop :=
  let m := (p1.2 - p2.2) / (p1.1 - p2.1)
  let b := p1.2 - m * p1.1
  p.2 = m * p.1 + b

-- Theorem statement
theorem line_points :
  is_on_line (5, 4) ∧
  is_on_line (1, -4) ∧
  ¬is_on_line (4, 1) ∧
  ¬is_on_line (3, -1) ∧
  ¬is_on_line (6, 7) :=
by sorry

end NUMINAMATH_CALUDE_line_points_l1654_165443


namespace NUMINAMATH_CALUDE_sector_area_from_arc_length_l1654_165420

/-- Given a circle where the arc length corresponding to a central angle of 2 radians is 4cm,
    prove that the area of the sector formed by this central angle is 4 cm². -/
theorem sector_area_from_arc_length (r : ℝ) (h : 2 * r = 4) : 
  (1 / 2) * r^2 * 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_from_arc_length_l1654_165420


namespace NUMINAMATH_CALUDE_integer_sum_problem_l1654_165418

theorem integer_sum_problem (x y : ℕ+) 
  (h1 : x.val - y.val = 8) 
  (h2 : x.val * y.val = 120) : 
  x.val + y.val = 2 * Real.sqrt 136 := by
sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l1654_165418


namespace NUMINAMATH_CALUDE_seating_arrangement_theorem_l1654_165474

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n people in a row where two specific people sit together -/
def arrangementsWithTwoTogether (n : ℕ) : ℕ := (n - 1).factorial * 2

/-- The number of people -/
def numberOfPeople : ℕ := 4

/-- The number of valid seating arrangements -/
def validArrangements : ℕ := totalArrangements numberOfPeople - arrangementsWithTwoTogether numberOfPeople

theorem seating_arrangement_theorem :
  validArrangements = 12 := by sorry

end NUMINAMATH_CALUDE_seating_arrangement_theorem_l1654_165474


namespace NUMINAMATH_CALUDE_min_side_difference_l1654_165430

theorem min_side_difference (a b c : ℕ) : 
  a + b + c = 3010 →
  a < b →
  b ≤ c →
  (∀ x y z : ℕ, x + y + z = 3010 → x < y → y ≤ z → b - a ≤ y - x) →
  b - a = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_side_difference_l1654_165430


namespace NUMINAMATH_CALUDE_cone_volume_and_surface_area_l1654_165408

/-- Given a cone with slant height 17 cm and height 15 cm, prove its volume and lateral surface area -/
theorem cone_volume_and_surface_area :
  let slant_height : ℝ := 17
  let height : ℝ := 15
  let radius : ℝ := Real.sqrt (slant_height ^ 2 - height ^ 2)
  let volume : ℝ := (1 / 3) * π * radius ^ 2 * height
  let lateral_surface_area : ℝ := π * radius * slant_height
  volume = 320 * π ∧ lateral_surface_area = 136 * π := by
  sorry


end NUMINAMATH_CALUDE_cone_volume_and_surface_area_l1654_165408


namespace NUMINAMATH_CALUDE_pigeonhole_principle_sports_choices_l1654_165451

/-- Given a set of 50 people, each making choices from three categories with 4, 3, and 2 options respectively,
    there must be at least 3 people who have made exactly the same choices for all three categories. -/
theorem pigeonhole_principle_sports_choices :
  ∀ (choices : Fin 50 → Fin 4 × Fin 3 × Fin 2),
  ∃ (c : Fin 4 × Fin 3 × Fin 2) (s₁ s₂ s₃ : Fin 50),
  s₁ ≠ s₂ ∧ s₁ ≠ s₃ ∧ s₂ ≠ s₃ ∧
  choices s₁ = c ∧ choices s₂ = c ∧ choices s₃ = c :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_principle_sports_choices_l1654_165451


namespace NUMINAMATH_CALUDE_games_left_to_play_l1654_165433

/-- Represents a round-robin tournament --/
structure Tournament where
  num_teams : Nat
  total_points : Nat
  lowest_score : Nat
  top_two_equal : Bool

/-- Calculates the total number of matches in a round-robin tournament --/
def total_matches (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the total points that will be distributed in the tournament --/
def total_tournament_points (n : Nat) : Nat :=
  2 * total_matches n

/-- Theorem: In a round-robin tournament with 9 teams, where the total points
    scored is 44, the lowest-scoring team has 1 point, and the top two teams
    have equal points, there are 14 games left to play. --/
theorem games_left_to_play (t : Tournament)
  (h1 : t.num_teams = 9)
  (h2 : t.total_points = 44)
  (h3 : t.lowest_score = 1)
  (h4 : t.top_two_equal = true) :
  total_matches t.num_teams - t.total_points / 2 = 14 := by
  sorry


end NUMINAMATH_CALUDE_games_left_to_play_l1654_165433


namespace NUMINAMATH_CALUDE_exists_monthly_increase_factor_l1654_165475

/-- The marathon distance in miles -/
def marathon_distance : ℝ := 26.3

/-- The initial running distance in miles -/
def initial_distance : ℝ := 3

/-- The number of months of training -/
def training_months : ℕ := 5

/-- Theorem stating the existence of a monthly increase factor -/
theorem exists_monthly_increase_factor :
  ∃ x : ℝ, x > 1 ∧ initial_distance * x^(training_months - 1) = marathon_distance :=
sorry

end NUMINAMATH_CALUDE_exists_monthly_increase_factor_l1654_165475


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l1654_165497

theorem cubic_equation_real_root (k : ℝ) : ∃ x : ℝ, x^3 + 3*k*x^2 + 3*k^2*x + k^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_real_root_l1654_165497


namespace NUMINAMATH_CALUDE_even_not_div_four_not_sum_consec_odd_l1654_165403

theorem even_not_div_four_not_sum_consec_odd (n : ℤ) : 
  ¬(∃ k : ℤ, 2 * (n + 1) = 4 * k + 2) :=
sorry

end NUMINAMATH_CALUDE_even_not_div_four_not_sum_consec_odd_l1654_165403


namespace NUMINAMATH_CALUDE_sum_mean_median_mode_l1654_165435

def numbers : List ℝ := [-3, -1, 0, 2, 2, 3, 3, 3, 4, 5]

def mode (l : List ℝ) : ℝ := sorry

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

theorem sum_mean_median_mode :
  mean numbers + median numbers + mode numbers = 7.3 := by sorry

end NUMINAMATH_CALUDE_sum_mean_median_mode_l1654_165435


namespace NUMINAMATH_CALUDE_ratio_of_segments_l1654_165438

-- Define the right triangle
def right_triangle (a b c r s : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ s > 0 ∧
  c^2 = a^2 + b^2 ∧
  c = r + s ∧
  a^2 = r * c ∧
  b^2 = s * c

-- Theorem statement
theorem ratio_of_segments (a b c r s : ℝ) :
  right_triangle a b c r s →
  a / b = 2 / 5 →
  r / s = 4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l1654_165438


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1654_165414

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 5 ∧ x₂ = -3/2 ∧ 
  (∀ x : ℝ, 2*x*(x-5) = 3*(5-x) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1654_165414


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l1654_165491

/-- A regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ angle : ℝ, angle = 150) →
  (180 * (n - 2) : ℝ) = (150 * n : ℝ) →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l1654_165491


namespace NUMINAMATH_CALUDE_salary_proof_l1654_165445

/-- The weekly salary of employee N -/
def N_salary : ℝ := 275

/-- The weekly salary of employee M -/
def M_salary (N_salary : ℝ) : ℝ := 1.2 * N_salary

/-- The total weekly salary for both employees -/
def total_salary : ℝ := 605

theorem salary_proof :
  N_salary + M_salary N_salary = total_salary :=
sorry

end NUMINAMATH_CALUDE_salary_proof_l1654_165445


namespace NUMINAMATH_CALUDE_james_beef_cost_l1654_165410

def beef_purchase (num_packs : ℕ) (weight_per_pack : ℝ) (price_per_pound : ℝ) : ℝ :=
  (num_packs : ℝ) * weight_per_pack * price_per_pound

theorem james_beef_cost :
  beef_purchase 5 4 5.50 = 110 := by
  sorry

end NUMINAMATH_CALUDE_james_beef_cost_l1654_165410


namespace NUMINAMATH_CALUDE_prob_roll_three_l1654_165421

/-- A fair six-sided die -/
structure FairDie :=
  (sides : Nat)
  (fair : sides = 6)

/-- The probability of rolling a specific number on a fair die -/
def prob_roll (d : FairDie) (n : Nat) : ℚ :=
  1 / d.sides

/-- The sequence of previous rolls -/
def previous_rolls : List Nat := [6, 6, 6, 6, 6, 6]

/-- Theorem: The probability of rolling a 3 on a fair six-sided die is 1/6,
    regardless of previous rolls -/
theorem prob_roll_three (d : FairDie) (prev : List Nat) :
  prob_roll d 3 = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_prob_roll_three_l1654_165421


namespace NUMINAMATH_CALUDE_petting_zoo_count_l1654_165432

/-- The number of animals Mary counted -/
def mary_count : ℕ := 130

/-- The number of animals Mary double-counted -/
def double_counted : ℕ := 19

/-- The number of animals Mary missed -/
def missed : ℕ := 39

/-- The actual number of animals in the petting zoo -/
def actual_count : ℕ := 150

theorem petting_zoo_count : 
  mary_count - double_counted + missed = actual_count := by sorry

end NUMINAMATH_CALUDE_petting_zoo_count_l1654_165432


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1654_165411

theorem circle_line_intersection :
  ∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 16 ∧ p.1 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1654_165411


namespace NUMINAMATH_CALUDE_vans_needed_for_field_trip_l1654_165470

theorem vans_needed_for_field_trip (van_capacity : ℕ) (num_students : ℕ) (num_adults : ℕ) :
  van_capacity = 5 → num_students = 25 → num_adults = 5 →
  (num_students + num_adults) / van_capacity = 6 :=
by sorry

end NUMINAMATH_CALUDE_vans_needed_for_field_trip_l1654_165470


namespace NUMINAMATH_CALUDE_floor_rate_per_square_meter_l1654_165464

/-- Given a rectangular room with length 8 m and width 4.75 m, and a total flooring cost of Rs. 34,200, the rate per square meter is Rs. 900. -/
theorem floor_rate_per_square_meter :
  let length : ℝ := 8
  let width : ℝ := 4.75
  let total_cost : ℝ := 34200
  let area : ℝ := length * width
  let rate_per_sq_meter : ℝ := total_cost / area
  rate_per_sq_meter = 900 := by sorry

end NUMINAMATH_CALUDE_floor_rate_per_square_meter_l1654_165464


namespace NUMINAMATH_CALUDE_base_76_congruence_l1654_165472

theorem base_76_congruence (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 18) 
  (h3 : (276935824 : ℤ) ≡ b [ZMOD 17]) : b = 0 ∨ b = 17 := by
  sorry

#check base_76_congruence

end NUMINAMATH_CALUDE_base_76_congruence_l1654_165472


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_ge_five_l1654_165444

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(a-1)*x + 2

/-- Theorem stating the condition for f(x) to be increasing on (-∞, 4) -/
theorem f_increasing_iff_a_ge_five (a : ℝ) :
  (∀ x y, x < y ∧ y < 4 → f a x < f a y) ↔ a ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_ge_five_l1654_165444


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1654_165436

/-- The coordinates of a point with respect to the origin are the same as its Cartesian coordinates. -/
theorem point_coordinates_wrt_origin (x y : ℝ) :
  let p : ℝ × ℝ := (x, y)
  p = p := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1654_165436


namespace NUMINAMATH_CALUDE_tomato_plant_problem_l1654_165458

theorem tomato_plant_problem (initial_tomatoes : ℕ) : 
  (initial_tomatoes : ℚ) - (1/4 * initial_tomatoes + 20 + 40 : ℚ) = 15 → 
  initial_tomatoes = 100 := by
sorry

end NUMINAMATH_CALUDE_tomato_plant_problem_l1654_165458


namespace NUMINAMATH_CALUDE_factorization_proof_l1654_165424

theorem factorization_proof (a x y : ℝ) : a * x - a * y = a * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1654_165424


namespace NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l1654_165441

theorem sqrt_inequality_solution_set (x : ℝ) :
  x + 3 ≥ 0 →
  (Real.sqrt (x + 3) > 3 - x ↔ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l1654_165441


namespace NUMINAMATH_CALUDE_constant_pace_running_time_l1654_165442

/-- Represents the time taken to run a certain distance at a constant pace -/
structure RunningPace where
  distance : ℝ
  time : ℝ

/-- Theorem: If it takes 24 minutes to run 3 miles at a constant pace, 
    then it will take 16 minutes to run 2 miles at the same pace -/
theorem constant_pace_running_time 
  (park : RunningPace) 
  (library : RunningPace) 
  (h1 : park.distance = 3) 
  (h2 : park.time = 24) 
  (h3 : library.distance = 2) 
  (h4 : park.time / park.distance = library.time / library.distance) : 
  library.time = 16 := by
sorry

end NUMINAMATH_CALUDE_constant_pace_running_time_l1654_165442


namespace NUMINAMATH_CALUDE_junior_toys_l1654_165449

theorem junior_toys (rabbits : ℕ) (monday_toys : ℕ) : 
  rabbits = 16 →
  (monday_toys + 2 * monday_toys + 4 * monday_toys + monday_toys) / rabbits = 3 →
  monday_toys = 6 := by
sorry

end NUMINAMATH_CALUDE_junior_toys_l1654_165449


namespace NUMINAMATH_CALUDE_num_boys_in_class_l1654_165490

-- Define the number of girls in the class
def num_girls : ℕ := 10

-- Define the number of ways to select 1 girl and 2 boys
def num_selections : ℕ := 1050

-- Define the function to calculate the number of ways to select 1 girl and 2 boys
def selection_ways (n : ℕ) : ℕ := num_girls * (n * (n - 1) / 2)

-- Theorem statement
theorem num_boys_in_class : ∃ (n : ℕ), n > 0 ∧ selection_ways n = num_selections :=
sorry

end NUMINAMATH_CALUDE_num_boys_in_class_l1654_165490


namespace NUMINAMATH_CALUDE_law_firm_associates_tenure_l1654_165481

/-- 
Given a law firm where:
- 30% of associates are second-year associates
- 60% of associates are not first-year associates

This theorem proves that 30% of associates have been at the firm for more than two years.
-/
theorem law_firm_associates_tenure (total : ℝ) (second_year : ℝ) (not_first_year : ℝ) 
  (h1 : second_year = 0.3 * total) 
  (h2 : not_first_year = 0.6 * total) : 
  total - (second_year + (total - not_first_year)) = 0.3 * total := by
  sorry

end NUMINAMATH_CALUDE_law_firm_associates_tenure_l1654_165481


namespace NUMINAMATH_CALUDE_frog_grasshopper_jump_difference_l1654_165479

theorem frog_grasshopper_jump_difference :
  let grasshopper_jump : ℕ := 9
  let frog_jump : ℕ := 12
  frog_jump - grasshopper_jump = 3 := by sorry

end NUMINAMATH_CALUDE_frog_grasshopper_jump_difference_l1654_165479


namespace NUMINAMATH_CALUDE_adjacent_even_sum_l1654_165427

/-- A circular arrangement of seven natural numbers -/
def CircularArrangement := Fin 7 → ℕ

/-- Two numbers in a circular arrangement are adjacent if their indices differ by 1 (mod 7) -/
def adjacent (arr : CircularArrangement) (i j : Fin 7) : Prop :=
  (i.val + 1) % 7 = j.val ∨ (j.val + 1) % 7 = i.val

/-- The main theorem: In any circular arrangement of seven natural numbers,
    there exist two adjacent numbers whose sum is even -/
theorem adjacent_even_sum (arr : CircularArrangement) :
  ∃ (i j : Fin 7), adjacent arr i j ∧ Even (arr i + arr j) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_even_sum_l1654_165427


namespace NUMINAMATH_CALUDE_dave_total_earnings_l1654_165493

def dave_earnings (hourly_wage : ℝ) (monday_hours : ℝ) (tuesday_hours : ℝ) : ℝ :=
  hourly_wage * (monday_hours + tuesday_hours)

theorem dave_total_earnings :
  dave_earnings 6 6 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_dave_total_earnings_l1654_165493


namespace NUMINAMATH_CALUDE_pentagon_coverage_is_62_5_percent_l1654_165407

/-- Represents a tiling of the plane with large squares and pentagons -/
structure PlaneTiling where
  /-- The number of smaller squares in each row/column of a large square -/
  grid_size : ℕ
  /-- The number of smaller squares that are part of pentagons in each large square -/
  pentagon_squares : ℕ

/-- The percentage of the plane enclosed by pentagons -/
def pentagon_percentage (t : PlaneTiling) : ℚ :=
  t.pentagon_squares / (t.grid_size ^ 2 : ℚ) * 100

/-- Theorem stating that the percentage of the plane enclosed by pentagons is 62.5% -/
theorem pentagon_coverage_is_62_5_percent (t : PlaneTiling) 
  (h1 : t.grid_size = 4)
  (h2 : t.pentagon_squares = 10) : 
  pentagon_percentage t = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_coverage_is_62_5_percent_l1654_165407


namespace NUMINAMATH_CALUDE_cost_function_cheaper_values_l1654_165455

def C (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 30 then 15 * n
  else if 31 ≤ n ∧ n ≤ 65 then 13 * n
  else 11 * n

theorem cost_function_cheaper_values :
  (∃ (S : Finset ℕ), S.card = 6 ∧ 
    (∀ n, n ∈ S ↔ (C (n + 1) < C n ∧ n ≥ 1))) :=
by sorry

end NUMINAMATH_CALUDE_cost_function_cheaper_values_l1654_165455


namespace NUMINAMATH_CALUDE_negation_of_existence_exp_l1654_165486

theorem negation_of_existence_exp (p : Prop) : 
  (p ↔ ∃ x : ℝ, Real.exp x < 0) → 
  (¬p ↔ ∀ x : ℝ, Real.exp x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_exp_l1654_165486


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1654_165419

theorem quadratic_function_property (a b c : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  (f 0 = f 4 ∧ f 0 > f 1) → (a > 0 ∧ 4 * a + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1654_165419


namespace NUMINAMATH_CALUDE_cos_difference_inverse_cos_tan_l1654_165473

theorem cos_difference_inverse_cos_tan (x y : ℝ) 
  (hx : x^2 ≤ 1) (hy : y > 0) : 
  Real.cos (Real.arccos (4/5) - Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_inverse_cos_tan_l1654_165473


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1654_165485

theorem mod_equivalence_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -3402 [ZMOD 10] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1654_165485


namespace NUMINAMATH_CALUDE_log_product_range_l1654_165422

theorem log_product_range :
  let y := Real.log 6 / Real.log 5 *
           Real.log 7 / Real.log 6 *
           Real.log 8 / Real.log 7 *
           Real.log 9 / Real.log 8 *
           Real.log 10 / Real.log 9
  1 < y ∧ y < 2 := by sorry

end NUMINAMATH_CALUDE_log_product_range_l1654_165422


namespace NUMINAMATH_CALUDE_tunnel_length_l1654_165440

/-- Calculates the length of a tunnel given train and journey parameters -/
theorem tunnel_length (train_length : ℝ) (exit_time : ℝ) (train_speed : ℝ) :
  train_length = 2 →
  exit_time = 5 / 60 →
  train_speed = 40 →
  (train_speed * exit_time) - train_length = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_tunnel_length_l1654_165440


namespace NUMINAMATH_CALUDE_value_of_a_l1654_165480

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

-- State the theorem
theorem value_of_a (a : ℝ) : 
  (∀ x, deriv (f a) x = 3 * a * x^2 + 6 * x) → 
  deriv (f a) (-1) = 3 → 
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l1654_165480


namespace NUMINAMATH_CALUDE_order_relation_l1654_165494

theorem order_relation (a b c : ℝ) : 
  a = Real.exp 0.2 → b = 0.2 ^ Real.exp 1 → c = Real.log 2 → b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_order_relation_l1654_165494


namespace NUMINAMATH_CALUDE_johnny_take_home_pay_is_67_32_l1654_165400

/-- Calculates Johnny's take-home pay after taxes based on his work hours and pay rates. -/
def johnny_take_home_pay (task_a_rate : ℝ) (task_b_rate : ℝ) (total_hours : ℝ) (task_a_hours : ℝ) (tax_rate : ℝ) : ℝ :=
  let task_b_hours := total_hours - task_a_hours
  let total_earnings := task_a_rate * task_a_hours + task_b_rate * task_b_hours
  let tax := tax_rate * total_earnings
  total_earnings - tax

/-- Proves that Johnny's take-home pay after taxes is $67.32 given the specified conditions. -/
theorem johnny_take_home_pay_is_67_32 :
  johnny_take_home_pay 6.75 8.25 10 4 0.12 = 67.32 := by
  sorry

end NUMINAMATH_CALUDE_johnny_take_home_pay_is_67_32_l1654_165400


namespace NUMINAMATH_CALUDE_inverse_f_at_4_equals_2_l1654_165412

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem inverse_f_at_4_equals_2 :
  ∃ (f_inv : ℝ → ℝ), (∀ x > 0, f_inv (f x) = x) ∧ (f_inv 4 = 2) := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_4_equals_2_l1654_165412


namespace NUMINAMATH_CALUDE_consecutive_episodes_probability_l1654_165448

theorem consecutive_episodes_probability (n : ℕ) (h : n = 6) :
  let total_combinations := n.choose 2
  let consecutive_pairs := n - 1
  (consecutive_pairs : ℚ) / total_combinations = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_consecutive_episodes_probability_l1654_165448


namespace NUMINAMATH_CALUDE_distribution_six_twelve_l1654_165461

/-- The number of ways to distribute distinct items among recipients --/
def distribution_ways (num_items : ℕ) (num_recipients : ℕ) : ℕ :=
  num_recipients ^ num_items

/-- Theorem: The number of ways to distribute 6 distinct items among 12 recipients is 2,985,984 --/
theorem distribution_six_twelve : distribution_ways 6 12 = 2985984 := by
  sorry

end NUMINAMATH_CALUDE_distribution_six_twelve_l1654_165461


namespace NUMINAMATH_CALUDE_string_length_problem_l1654_165466

theorem string_length_problem (total_strings : ℕ) (total_avg : ℝ) (subset_strings : ℕ) (subset_avg : ℝ) :
  total_strings = 6 →
  total_avg = 80 →
  subset_strings = 2 →
  subset_avg = 70 →
  let remaining_strings := total_strings - subset_strings
  let total_length := total_strings * total_avg
  let subset_length := subset_strings * subset_avg
  let remaining_length := total_length - subset_length
  (remaining_length / remaining_strings) = 85 := by
sorry

end NUMINAMATH_CALUDE_string_length_problem_l1654_165466


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_values_l1654_165465

theorem arithmetic_geometric_sequence_values :
  ∀ (a b c : ℝ),
  (∃ (d : ℝ), b = (a + c) / 2 ∧ c - b = b - a) →  -- arithmetic sequence condition
  (a + b + c = 12) →  -- sum condition
  (∃ (r : ℝ), (b + 2) ^ 2 = (a + 2) * (c + 5)) →  -- geometric sequence condition
  ((a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_values_l1654_165465


namespace NUMINAMATH_CALUDE_sum_of_areas_equals_AD_squared_l1654_165483

/-- A right-angled quadrilateral with the golden ratio property -/
structure GoldenQuadrilateral where
  AB : ℝ
  AD : ℝ
  right_angled : AB > 0 ∧ AD > 0
  shorter_side : AB < AD
  golden_ratio : AB / AD = (AD - AB) / AB

/-- The sum of areas of an infinite series of similar quadrilaterals -/
def sum_of_areas (q : GoldenQuadrilateral) : ℝ := q.AD ^ 2

/-- The main theorem: the sum of areas equals AD^2 -/
theorem sum_of_areas_equals_AD_squared (q : GoldenQuadrilateral) :
  sum_of_areas q = q.AD ^ 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_areas_equals_AD_squared_l1654_165483


namespace NUMINAMATH_CALUDE_video_game_expense_is_correct_l1654_165476

def total_allowance : ℚ := 50

def book_fraction : ℚ := 1/2
def toy_fraction : ℚ := 1/4
def snack_fraction : ℚ := 1/10

def video_game_expense : ℚ := total_allowance - (book_fraction * total_allowance + toy_fraction * total_allowance + snack_fraction * total_allowance)

theorem video_game_expense_is_correct : video_game_expense = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_video_game_expense_is_correct_l1654_165476


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1654_165453

def A : Set ℤ := {x : ℤ | |x| < 3}
def B : Set ℤ := {x : ℤ | |x| > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1654_165453


namespace NUMINAMATH_CALUDE_student_speed_ratio_l1654_165482

theorem student_speed_ratio :
  ∀ (distance_A distance_B time_A time_B : ℚ),
    distance_A = (6 / 5) * distance_B →
    time_B = (10 / 11) * time_A →
    (distance_A / time_A) / (distance_B / time_B) = 12 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_student_speed_ratio_l1654_165482


namespace NUMINAMATH_CALUDE_wire_length_proof_l1654_165454

theorem wire_length_proof (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 4 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 14 := by
sorry

end NUMINAMATH_CALUDE_wire_length_proof_l1654_165454


namespace NUMINAMATH_CALUDE_songs_leftover_l1654_165423

theorem songs_leftover (total_songs : ℕ) (num_playlists : ℕ) (h1 : total_songs = 2048) (h2 : num_playlists = 13) :
  total_songs % num_playlists = 7 := by
  sorry

end NUMINAMATH_CALUDE_songs_leftover_l1654_165423


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1654_165471

theorem pure_imaginary_complex_number (a : ℝ) :
  let z : ℂ := a * (a - 1) + a * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1654_165471


namespace NUMINAMATH_CALUDE_function_domain_range_equality_l1654_165463

theorem function_domain_range_equality (a : ℝ) (h1 : a > 1) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*a*x + 5
  (∀ x, f x ∈ Set.Icc 1 a ↔ x ∈ Set.Icc 1 a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_domain_range_equality_l1654_165463


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_for_x_geq_1_l1654_165478

theorem x_eq_2_sufficient_not_necessary_for_x_geq_1 :
  (∀ x : ℝ, x = 2 → x ≥ 1) ∧ ¬(∀ x : ℝ, x ≥ 1 → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_for_x_geq_1_l1654_165478


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1654_165467

theorem trigonometric_identity (α : Real) 
  (h1 : π < α ∧ α < 2*π) 
  (h2 : Real.cos (α - 7*π) = -3/5) : 
  Real.sin (3*π + α) * Real.tan (α - 7*π/2) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1654_165467
