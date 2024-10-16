import Mathlib

namespace NUMINAMATH_CALUDE_tree_distance_l3842_384248

/-- Given 8 equally spaced trees along a straight road, where the distance between
    the first and fifth tree is 100 feet, and a sign 30 feet beyond the last tree,
    the total distance between the first tree and the sign is 205 feet. -/
theorem tree_distance (n : ℕ) (d : ℝ) (s : ℝ) : 
  n = 8 → d = 100 → s = 30 → 
  (n - 1) * (d / 4) + s = 205 :=
by sorry

end NUMINAMATH_CALUDE_tree_distance_l3842_384248


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3842_384201

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_with_complement :
  A ∩ (Set.univ \ B) = {1, 5, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3842_384201


namespace NUMINAMATH_CALUDE_expression_evaluation_l3842_384295

theorem expression_evaluation :
  ((3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2) / (10^1002) = 28 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3842_384295


namespace NUMINAMATH_CALUDE_water_level_rise_l3842_384265

/-- The rise in water level when a sphere is fully immersed in a rectangular vessel --/
theorem water_level_rise (sphere_radius : ℝ) (vessel_length : ℝ) (vessel_width : ℝ) :
  sphere_radius = 10 →
  vessel_length = 30 →
  vessel_width = 25 →
  ∃ (water_rise : ℝ), abs (water_rise - 5.59) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_water_level_rise_l3842_384265


namespace NUMINAMATH_CALUDE_tetrahedrons_from_triangular_prism_l3842_384275

/-- The number of tetrahedrons that can be formed from a regular triangular prism -/
def tetrahedrons_from_prism (n : ℕ) : ℕ :=
  Nat.choose n 4 - 3

/-- Theorem stating that the number of tetrahedrons formed from a regular triangular prism with 6 vertices is 12 -/
theorem tetrahedrons_from_triangular_prism : 
  tetrahedrons_from_prism 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedrons_from_triangular_prism_l3842_384275


namespace NUMINAMATH_CALUDE_max_value_theorem_l3842_384268

theorem max_value_theorem (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 15) : 
  x + 2 * y ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3842_384268


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3842_384230

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {aₙ}, if a₄ + a₅ + a₆ = 90, then a₅ = 30 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : ArithmeticSequence a) (h2 : a 4 + a 5 + a 6 = 90) :
  a 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3842_384230


namespace NUMINAMATH_CALUDE_skill_players_count_l3842_384269

/-- Represents the football team's water consumption scenario -/
structure FootballTeam where
  cooler_capacity : ℕ
  num_linemen : ℕ
  lineman_consumption : ℕ
  skill_player_consumption : ℕ
  waiting_skill_players : ℕ

/-- Calculates the number of skill position players on the team -/
def num_skill_players (team : FootballTeam) : ℕ :=
  let remaining_water := team.cooler_capacity - team.num_linemen * team.lineman_consumption
  let drinking_skill_players := remaining_water / team.skill_player_consumption
  drinking_skill_players + team.waiting_skill_players

/-- Theorem stating the number of skill position players on the team -/
theorem skill_players_count (team : FootballTeam) 
  (h1 : team.cooler_capacity = 126)
  (h2 : team.num_linemen = 12)
  (h3 : team.lineman_consumption = 8)
  (h4 : team.skill_player_consumption = 6)
  (h5 : team.waiting_skill_players = 5) :
  num_skill_players team = 10 := by
  sorry

#eval num_skill_players {
  cooler_capacity := 126,
  num_linemen := 12,
  lineman_consumption := 8,
  skill_player_consumption := 6,
  waiting_skill_players := 5
}

end NUMINAMATH_CALUDE_skill_players_count_l3842_384269


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_minus_3i_l3842_384240

theorem imaginary_part_of_2_minus_3i :
  Complex.im (2 - 3 * Complex.I) = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_minus_3i_l3842_384240


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l3842_384262

-- Define the curve f(x) = 2x³ - 3x
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 6 * x^2 - 3

-- Theorem statement
theorem tangent_line_at_origin :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (y = m * x) →                   -- Equation of a line through (0,0)
    (∃ (t : ℝ), t ≠ 0 →
      y = f t ∧                     -- Point (t, f(t)) is on the curve
      (f t - 0) / (t - 0) = m) →    -- Slope of secant line
    m = -3                          -- Slope of the tangent line
    :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l3842_384262


namespace NUMINAMATH_CALUDE_workshop_average_salary_l3842_384270

/-- Proves that the average salary of all workers in a workshop is 8000 rupees. -/
theorem workshop_average_salary :
  let total_workers : ℕ := 21
  let technicians : ℕ := 7
  let technician_salary : ℕ := 12000
  let non_technician_salary : ℕ := 6000
  (total_workers * (technicians * technician_salary + (total_workers - technicians) * non_technician_salary)) / (total_workers * total_workers) = 8000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l3842_384270


namespace NUMINAMATH_CALUDE_union_complement_equal_l3842_384278

def U : Finset ℕ := {0,1,2,4,6,8}
def M : Finset ℕ := {0,4,6}
def N : Finset ℕ := {0,1,6}

theorem union_complement_equal : M ∪ (U \ N) = {0,2,4,6,8} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equal_l3842_384278


namespace NUMINAMATH_CALUDE_angle_four_value_l3842_384222

/-- Given an isosceles triangle and some angle relationships, prove that angle 4 is 37.5 degrees -/
theorem angle_four_value (angle1 angle2 angle3 angle4 angle5 x y : ℝ) : 
  angle1 + angle2 = 180 →
  angle3 = angle4 →
  angle3 + angle4 + angle5 = 180 →
  angle1 = 45 + x →
  angle3 = 30 + y →
  x = 2 * y →
  angle4 = 37.5 := by
sorry

end NUMINAMATH_CALUDE_angle_four_value_l3842_384222


namespace NUMINAMATH_CALUDE_marble_probability_l3842_384258

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) :
  total = 84 →
  p_white = 1/4 →
  p_green = 1/7 →
  1 - (p_white + p_green) = 17/28 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3842_384258


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l3842_384227

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 0)

-- Theorem: The line passes through the center of the circle
theorem line_passes_through_circle_center :
  line_equation (circle_center.1) (circle_center.2) := by
  sorry


end NUMINAMATH_CALUDE_line_passes_through_circle_center_l3842_384227


namespace NUMINAMATH_CALUDE_total_money_l3842_384299

theorem total_money (a b c : ℕ) 
  (h1 : a + c = 200)
  (h2 : b + c = 360)
  (h3 : c = 60) :
  a + b + c = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l3842_384299


namespace NUMINAMATH_CALUDE_sparklers_to_crackers_theorem_value_comparison_theorem_l3842_384292

/-- Represents the exchange rates between different holiday items -/
structure ExchangeRates where
  ornament_to_cracker : ℚ
  sparkler_to_garland : ℚ
  ornament_to_garland : ℚ

/-- Converts sparklers to crackers based on the given exchange rates -/
def sparklers_to_crackers (rates : ExchangeRates) (sparklers : ℚ) : ℚ :=
  let garlands := (sparklers / 5) * 2
  let ornaments := garlands * 4
  ornaments * rates.ornament_to_cracker

/-- Compares the value of ornaments and crackers to sparklers -/
def compare_values (rates : ExchangeRates) (ornaments crackers sparklers : ℚ) : Prop :=
  ornaments * rates.ornament_to_cracker + crackers > 
  (sparklers / 5) * 2 * 4 * rates.ornament_to_cracker

/-- Theorem stating the equivalence of 10 sparklers to 32 crackers -/
theorem sparklers_to_crackers_theorem (rates : ExchangeRates) : 
  sparklers_to_crackers rates 10 = 32 :=
by sorry

/-- Theorem comparing the value of 5 ornaments and 1 cracker to 2 sparklers -/
theorem value_comparison_theorem (rates : ExchangeRates) : 
  compare_values rates 5 1 2 :=
by sorry

end NUMINAMATH_CALUDE_sparklers_to_crackers_theorem_value_comparison_theorem_l3842_384292


namespace NUMINAMATH_CALUDE_infinite_fraction_value_l3842_384213

theorem infinite_fraction_value : 
  ∃ x : ℝ, x = 3 + 3 / (1 + 5 / x) ∧ x = (1 + Real.sqrt 61) / 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_fraction_value_l3842_384213


namespace NUMINAMATH_CALUDE_arcsin_one_half_eq_pi_sixth_l3842_384216

theorem arcsin_one_half_eq_pi_sixth : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_eq_pi_sixth_l3842_384216


namespace NUMINAMATH_CALUDE_chrysanthemum_arrangements_l3842_384255

/-- The number of permutations of n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange k objects out of n distinct objects. -/
def arrangements (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The number of ways to arrange 6 distinct objects in a row, 
    where two specific objects (A and B) are on the same side of a third specific object (C). -/
theorem chrysanthemum_arrangements : 
  2 * (permutations 5 + 
       arrangements 4 2 * arrangements 3 3 + 
       arrangements 2 2 * arrangements 3 3 + 
       arrangements 3 2 * arrangements 3 3) = 480 := by
  sorry


end NUMINAMATH_CALUDE_chrysanthemum_arrangements_l3842_384255


namespace NUMINAMATH_CALUDE_worker_y_fraction_l3842_384251

theorem worker_y_fraction (total_products : ℝ) (x_products y_products : ℝ) 
  (h1 : x_products + y_products = total_products)
  (h2 : 0.005 * x_products + 0.008 * y_products = 0.007 * total_products) :
  y_products / total_products = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_worker_y_fraction_l3842_384251


namespace NUMINAMATH_CALUDE_two_number_problem_l3842_384221

def is_solution (x y : ℕ) : Prop :=
  (x + y = 667) ∧ 
  (Nat.lcm x y / Nat.gcd x y = 120)

theorem two_number_problem :
  ∀ x y : ℕ, is_solution x y → 
    ((x = 115 ∧ y = 552) ∨ (x = 552 ∧ y = 115) ∨ 
     (x = 232 ∧ y = 435) ∨ (x = 435 ∧ y = 232)) :=
by sorry

end NUMINAMATH_CALUDE_two_number_problem_l3842_384221


namespace NUMINAMATH_CALUDE_pyramid_missing_number_l3842_384284

/-- Represents a row in the pyramid -/
structure PyramidRow :=
  (left : ℚ) (middle : ℚ) (right : ℚ)

/-- Represents the pyramid structure -/
structure Pyramid :=
  (top_row : PyramidRow)
  (middle_row : PyramidRow)
  (bottom_row : PyramidRow)

/-- Checks if the pyramid satisfies the product rule -/
def is_valid_pyramid (p : Pyramid) : Prop :=
  p.middle_row.left = p.top_row.left * p.top_row.middle ∧
  p.middle_row.middle = p.top_row.middle * p.top_row.right ∧
  p.bottom_row.left = p.middle_row.left * p.middle_row.middle ∧
  p.bottom_row.middle = p.middle_row.middle * p.middle_row.right

/-- The main theorem -/
theorem pyramid_missing_number :
  ∀ (p : Pyramid),
    is_valid_pyramid p →
    p.middle_row = ⟨3, 2, 5⟩ →
    p.bottom_row.left = 10 →
    p.top_row.middle = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_missing_number_l3842_384284


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_four_l3842_384242

theorem sqrt_difference_equals_four :
  Real.sqrt (9 + 4 * Real.sqrt 5) - Real.sqrt (9 - 4 * Real.sqrt 5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_four_l3842_384242


namespace NUMINAMATH_CALUDE_centroid_tetrahedron_volume_centroid_tetrahedron_volume_54_l3842_384247

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ

/-- Represents the tetrahedron formed by the centroids of the faces of another tetrahedron -/
def centroid_tetrahedron (t : RegularTetrahedron) : RegularTetrahedron :=
  sorry

/-- The volume of the centroid tetrahedron is 1/27 of the original tetrahedron's volume -/
theorem centroid_tetrahedron_volume (t : RegularTetrahedron) :
  (centroid_tetrahedron t).volume = t.volume / 27 :=
sorry

/-- Given a regular tetrahedron with volume 54, the volume of the tetrahedron
    formed by the centroids of its four faces is 2 -/
theorem centroid_tetrahedron_volume_54 :
  let t : RegularTetrahedron := ⟨54⟩
  (centroid_tetrahedron t).volume = 2 :=
sorry

end NUMINAMATH_CALUDE_centroid_tetrahedron_volume_centroid_tetrahedron_volume_54_l3842_384247


namespace NUMINAMATH_CALUDE_cubic_function_nonnegative_l3842_384261

theorem cubic_function_nonnegative (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, a * x^3 - 3 * x + 1 ≥ 0) ↔ a = 4 :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_nonnegative_l3842_384261


namespace NUMINAMATH_CALUDE_greatest_drop_in_april_l3842_384217

/-- Represents the months from January to June -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June

/-- The price change for each month -/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January  => -1.00
  | Month.February => 2.50
  | Month.March    => 0.00
  | Month.April    => -3.00
  | Month.May      => -1.50
  | Month.June     => 1.00

/-- A month has a price drop if its price change is negative -/
def has_price_drop (m : Month) : Prop :=
  price_change m < 0

/-- The greatest monthly drop in price occurred in April -/
theorem greatest_drop_in_april :
  ∀ m : Month, has_price_drop m → price_change Month.April ≤ price_change m :=
by sorry

end NUMINAMATH_CALUDE_greatest_drop_in_april_l3842_384217


namespace NUMINAMATH_CALUDE_solve_for_k_l3842_384208

theorem solve_for_k (x y k : ℝ) : 
  x = 2 → 
  y = 1 → 
  k * x - y = 3 → 
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_for_k_l3842_384208


namespace NUMINAMATH_CALUDE_no_lower_bound_l3842_384250

/-- An arithmetic sequence with three terms -/
structure ArithmeticSequence :=
  (a₁ : ℝ)
  (d : ℝ)

/-- The second term of the arithmetic sequence -/
def ArithmeticSequence.a₂ (seq : ArithmeticSequence) : ℝ := seq.a₁ + seq.d

/-- The third term of the arithmetic sequence -/
def ArithmeticSequence.a₃ (seq : ArithmeticSequence) : ℝ := seq.a₁ + 2 * seq.d

/-- The expression to be minimized -/
def expression (seq : ArithmeticSequence) : ℝ := 3 * seq.a₂ + 7 * seq.a₃

/-- The theorem stating that the expression has no lower bound -/
theorem no_lower_bound :
  ∀ (b : ℝ), ∃ (seq : ArithmeticSequence), seq.a₁ = 3 ∧ expression seq < b :=
sorry

end NUMINAMATH_CALUDE_no_lower_bound_l3842_384250


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l3842_384279

theorem least_number_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 41 = 5) ∧ 
  (n % 23 = 5) ∧ 
  (∀ m : ℕ, m > 0 → m % 41 = 5 → m % 23 = 5 → m ≥ n) ∧
  n = 948 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l3842_384279


namespace NUMINAMATH_CALUDE_incorrect_transformation_l3842_384226

theorem incorrect_transformation :
  (∀ a b : ℝ, a - 3 = b - 3 → a = b) ∧
  (∀ a b c : ℝ, c ≠ 0 → a / c = b / c → a = b) ∧
  (∀ a b c : ℝ, a = b → a / (c^2 + 1) = b / (c^2 + 1)) ∧
  ¬(∀ a b c : ℝ, a * c = b * c → a = b) := by
sorry


end NUMINAMATH_CALUDE_incorrect_transformation_l3842_384226


namespace NUMINAMATH_CALUDE_seating_arrangement_count_l3842_384241

/-- Represents the seating arrangement for two rows of seats. -/
structure SeatingArrangement where
  front_row : ℕ  -- Number of seats in the front row
  back_row : ℕ   -- Number of seats in the back row
  unavailable_front : ℕ  -- Number of unavailable seats in the front row

/-- Calculates the number of seating arrangements for two people. -/
def count_seating_arrangements (s : SeatingArrangement) : ℕ :=
  sorry

/-- The main theorem stating the number of seating arrangements. -/
theorem seating_arrangement_count :
  let s : SeatingArrangement := { front_row := 11, back_row := 12, unavailable_front := 3 }
  count_seating_arrangements s = 346 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_count_l3842_384241


namespace NUMINAMATH_CALUDE_hyperbolas_M_value_l3842_384288

/-- Two hyperbolas with the same asymptotes -/
def hyperbolas_same_asymptotes (M : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧
  (∀ (x y : ℝ), x^2/9 - y^2/16 = 1 → y = k*x ∨ y = -k*x) ∧
  (∀ (x y : ℝ), y^2/25 - x^2/M = 1 → y = k*x ∨ y = -k*x)

/-- The value of M for which the hyperbolas have the same asymptotes -/
theorem hyperbolas_M_value :
  hyperbolas_same_asymptotes (225/16) :=
sorry

end NUMINAMATH_CALUDE_hyperbolas_M_value_l3842_384288


namespace NUMINAMATH_CALUDE_leonards_age_l3842_384286

theorem leonards_age (nina jerome leonard : ℕ) 
  (h1 : leonard = nina - 4)
  (h2 : nina = jerome / 2)
  (h3 : nina + jerome + leonard = 36) : 
  leonard = 6 := by
sorry

end NUMINAMATH_CALUDE_leonards_age_l3842_384286


namespace NUMINAMATH_CALUDE_min_tank_cost_l3842_384287

/-- Represents the cost function for a rectangular water tank. -/
def tank_cost (x y : ℝ) : ℝ :=
  120 * (x * y) + 100 * (2 * 3 * x + 2 * 3 * y)

/-- Theorem stating the minimum cost for the water tank construction. -/
theorem min_tank_cost :
  let volume : ℝ := 300
  let depth : ℝ := 3
  let bottom_cost : ℝ := 120
  let wall_cost : ℝ := 100
  ∀ x y : ℝ,
    x > 0 → y > 0 →
    x * y * depth = volume →
    tank_cost x y ≥ 24000 ∧
    (x = 10 ∧ y = 10 → tank_cost x y = 24000) :=
by sorry

end NUMINAMATH_CALUDE_min_tank_cost_l3842_384287


namespace NUMINAMATH_CALUDE_range_of_m_l3842_384277

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2/x + 1/y = 1) :
  (∀ x y, x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m^2 + 2*m) ↔ -4 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3842_384277


namespace NUMINAMATH_CALUDE_independent_events_probability_l3842_384202

theorem independent_events_probability (a b : Set α) (p : Set α → ℚ) :
  (p a = 4/7) → (p b = 2/5) → (∀ x y, p (x ∩ y) = p x * p y) → p (a ∩ b) = 8/35 := by
  sorry

end NUMINAMATH_CALUDE_independent_events_probability_l3842_384202


namespace NUMINAMATH_CALUDE_two_by_one_prism_net_removable_squares_l3842_384285

/-- Represents a rectangular prism --/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a net of a rectangular prism --/
structure PrismNet where
  squares : ℕ

/-- Function to create a net from a rectangular prism --/
def createNet (prism : RectangularPrism) : PrismNet :=
  { squares := 2 * (prism.length * prism.width + prism.length * prism.height + prism.width * prism.height) }

/-- Function to count removable squares in a net --/
def countRemovableSquares (net : PrismNet) : ℕ := sorry

/-- Theorem stating that a 2×1×1 prism net has exactly 5 removable squares --/
theorem two_by_one_prism_net_removable_squares :
  let prism : RectangularPrism := { length := 2, width := 1, height := 1 }
  let net := createNet prism
  countRemovableSquares net = 5 ∧ net.squares - 1 = 9 := by sorry

end NUMINAMATH_CALUDE_two_by_one_prism_net_removable_squares_l3842_384285


namespace NUMINAMATH_CALUDE_right_triangle_angle_calculation_l3842_384291

/-- In a triangle ABC with a right angle at A, prove that x = 10/3 degrees -/
theorem right_triangle_angle_calculation (x y : ℝ) : 
  x + y = 40 →
  3 * x + 2 * y = 90 →
  x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_calculation_l3842_384291


namespace NUMINAMATH_CALUDE_decimal_places_in_expression_l3842_384253

-- Define the original number
def original_number : ℝ := 3.456789

-- Define the expression
def expression : ℝ := ((10^4 : ℝ) * original_number)^9

-- Function to count decimal places
def count_decimal_places (x : ℝ) : ℕ :=
  sorry

-- Theorem stating that the number of decimal places in the expression is 2
theorem decimal_places_in_expression :
  count_decimal_places expression = 2 := by
  sorry

end NUMINAMATH_CALUDE_decimal_places_in_expression_l3842_384253


namespace NUMINAMATH_CALUDE_polynomial_sum_l3842_384232

theorem polynomial_sum (m : ℝ) : (m^2 + m) + (-3*m) = m^2 - 2*m := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3842_384232


namespace NUMINAMATH_CALUDE_triangle_count_is_102_l3842_384218

/-- Represents a rectangle divided into a 6x2 grid with diagonal lines -/
structure GridRectangle where
  width : ℕ
  height : ℕ
  grid_width : ℕ
  grid_height : ℕ
  has_diagonals : Bool

/-- Counts the number of triangles in a GridRectangle -/
def count_triangles (rect : GridRectangle) : ℕ :=
  sorry

/-- Theorem stating that the number of triangles in the specific GridRectangle is 102 -/
theorem triangle_count_is_102 :
  ∃ (rect : GridRectangle),
    rect.width = 6 ∧
    rect.height = 2 ∧
    rect.grid_width = 6 ∧
    rect.grid_height = 2 ∧
    rect.has_diagonals = true ∧
    count_triangles rect = 102 :=
  sorry

end NUMINAMATH_CALUDE_triangle_count_is_102_l3842_384218


namespace NUMINAMATH_CALUDE_probability_at_least_one_one_l3842_384274

-- Define the number of sides on each die
def num_sides : ℕ := 8

-- Define the probability of at least one die showing 1
def prob_at_least_one_one : ℚ := 15 / 64

-- Theorem statement
theorem probability_at_least_one_one :
  let total_outcomes := num_sides * num_sides
  let outcomes_without_one := (num_sides - 1) * (num_sides - 1)
  let favorable_outcomes := total_outcomes - outcomes_without_one
  (favorable_outcomes : ℚ) / total_outcomes = prob_at_least_one_one := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_one_l3842_384274


namespace NUMINAMATH_CALUDE_monochromatic_triangle_k6_exists_no_monochromatic_triangle_k5_l3842_384233

/-- A type representing the two colors used in the graph coloring -/
inductive Color
| Red
| Blue

/-- A type representing a complete graph with n vertices -/
def CompleteGraph (n : ℕ) := Fin n → Fin n → Color

/-- A predicate that checks if a triangle is monochromatic in a given graph -/
def HasMonochromaticTriangle (g : CompleteGraph n) : Prop :=
  ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    g i j = g j k ∧ g j k = g i k

theorem monochromatic_triangle_k6 :
  ∀ (g : CompleteGraph 6), HasMonochromaticTriangle g :=
sorry

theorem exists_no_monochromatic_triangle_k5 :
  ∃ (g : CompleteGraph 5), ¬HasMonochromaticTriangle g :=
sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_k6_exists_no_monochromatic_triangle_k5_l3842_384233


namespace NUMINAMATH_CALUDE_divisibility_property_l3842_384297

theorem divisibility_property (m : ℕ+) (x : ℝ) :
  ∃ k : ℝ, (x + 1)^(2 * m.val) - x^(2 * m.val) - 2*x - 1 = k * (x * (x + 1) * (2*x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3842_384297


namespace NUMINAMATH_CALUDE_ab_range_l3842_384249

-- Define the line equation
def line_equation (a b x y : ℝ) : Prop := a * x - b * y + 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the property of bisecting the circumference
def bisects_circle (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), line_equation a b x y ∧ circle_equation x y

-- Theorem statement
theorem ab_range (a b : ℝ) : 
  bisects_circle a b → ab ∈ Set.Iic (1/8) :=
sorry

end NUMINAMATH_CALUDE_ab_range_l3842_384249


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3842_384225

def A : Set ℝ := {2, 4, 6, 8}
def B : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ 6}

theorem intersection_of_A_and_B : A ∩ B = {4, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3842_384225


namespace NUMINAMATH_CALUDE_two_bags_below_threshold_probability_l3842_384257

-- Define the normal distribution parameters
def μ : ℝ := 500
def σ : ℝ := 5

-- Define the threshold weight
def threshold : ℝ := 485

-- Define the probability of selecting one bag below the threshold
def prob_one_bag : ℝ := 0.0013

-- Theorem statement
theorem two_bags_below_threshold_probability :
  let prob_two_bags := prob_one_bag * prob_one_bag
  prob_two_bags < 2e-6 := by sorry

end NUMINAMATH_CALUDE_two_bags_below_threshold_probability_l3842_384257


namespace NUMINAMATH_CALUDE_solve_equation_l3842_384252

theorem solve_equation (x : ℝ) : 3*x - 4*x + 7*x = 210 → x = 35 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3842_384252


namespace NUMINAMATH_CALUDE_cubic_roots_arithmetic_progression_l3842_384254

/-- A cubic polynomial with coefficients a, b, and c -/
def cubic_polynomial (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The condition for the roots of a cubic polynomial to form an arithmetic progression -/
def arithmetic_progression_condition (a b c : ℝ) : Prop :=
  2 * a^3 / 27 - a * b / 3 + c = 0

/-- Theorem stating that the roots of a cubic polynomial form an arithmetic progression
    if and only if the coefficients satisfy the arithmetic progression condition -/
theorem cubic_roots_arithmetic_progression (a b c : ℝ) :
  (∃ x y z : ℝ, x - y = y - z ∧ 
    (∀ t : ℝ, cubic_polynomial a b c t = 0 ↔ t = x ∨ t = y ∨ t = z)) ↔ 
  arithmetic_progression_condition a b c :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_arithmetic_progression_l3842_384254


namespace NUMINAMATH_CALUDE_not_negative_review_A_two_positive_reviews_out_of_four_l3842_384220

-- Define the platforms
inductive Platform
| A
| B

-- Define the review types
inductive Review
| Positive
| Neutral
| Negative

-- Define the function for the number of reviews for each platform and review type
def reviewCount (p : Platform) (r : Review) : ℕ :=
  match p, r with
  | Platform.A, Review.Positive => 75
  | Platform.A, Review.Neutral => 20
  | Platform.A, Review.Negative => 5
  | Platform.B, Review.Positive => 64
  | Platform.B, Review.Neutral => 8
  | Platform.B, Review.Negative => 8

-- Define the total number of reviews for each platform
def totalReviews (p : Platform) : ℕ :=
  reviewCount p Review.Positive + reviewCount p Review.Neutral + reviewCount p Review.Negative

-- Define the probability of a review type for a given platform
def reviewProbability (p : Platform) (r : Review) : ℚ :=
  reviewCount p r / totalReviews p

-- Theorem for the probability of not receiving a negative review for platform A
theorem not_negative_review_A :
  1 - reviewProbability Platform.A Review.Negative = 19/20 := by sorry

-- Theorem for the probability of exactly 2 out of 4 randomly selected buyers giving a positive review
theorem two_positive_reviews_out_of_four :
  let pA := reviewProbability Platform.A Review.Positive
  let pB := reviewProbability Platform.B Review.Positive
  (pA^2 * (1-pB)^2) + (2 * pA * (1-pA) * pB * (1-pB)) + ((1-pA)^2 * pB^2) = 73/400 := by sorry

end NUMINAMATH_CALUDE_not_negative_review_A_two_positive_reviews_out_of_four_l3842_384220


namespace NUMINAMATH_CALUDE_opposite_sides_inequality_l3842_384223

/-- Given that point P (x₀, y₀) and point A (1, 2) are on opposite sides of the line l: 3x + 2y - 8 = 0,
    prove that 3x₀ + 2y₀ > 8 -/
theorem opposite_sides_inequality (x₀ y₀ : ℝ) : 
  (3*x₀ + 2*y₀ - 8) * (3*1 + 2*2 - 8) < 0 → 3*x₀ + 2*y₀ > 8 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_inequality_l3842_384223


namespace NUMINAMATH_CALUDE_equation_solution_l3842_384235

/-- Given an equation y = a + b / x^2, where a and b are constants,
    if y = 2 when x = -2 and y = 4 when x = -4, then a + b = -6 -/
theorem equation_solution (a b : ℝ) : 
  (2 = a + b / (-2)^2) → 
  (4 = a + b / (-4)^2) → 
  a + b = -6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3842_384235


namespace NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l3842_384276

/-- Represents a hyperbola with vertices on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a line with slope k passing through (3,0) -/
structure Line where
  k : ℝ

/-- Defines when a line intersects a hyperbola at exactly one point -/
def intersects_at_one_point (h : Hyperbola) (l : Line) : Prop :=
  ∃! x y : ℝ, x^2 / h.a^2 - y^2 / h.b^2 = 1 ∧ y = l.k * (x - 3)

/-- The main theorem to be proved -/
theorem hyperbola_intersection_theorem (h : Hyperbola) (l : Line) :
  h.a = 4 ∧ h.b = 3 →
  intersects_at_one_point h l ↔ 
    l.k = 3/4 ∨ l.k = -3/4 ∨ l.k = 3*Real.sqrt 7/7 ∨ l.k = -3*Real.sqrt 7/7 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l3842_384276


namespace NUMINAMATH_CALUDE_triangle_angle_c_value_l3842_384238

theorem triangle_angle_c_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  a^2 = 3*b^2 + 3*c^2 - 2*Real.sqrt 3*b*c*Real.sin A →
  C = π/6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_value_l3842_384238


namespace NUMINAMATH_CALUDE_women_group_size_l3842_384224

/-- The number of women in the first group -/
def first_group_size : ℕ := 6

/-- The length of cloth colored by the first group -/
def first_group_cloth_length : ℕ := 180

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 3

/-- The number of women in the second group -/
def second_group_size : ℕ := 5

/-- The length of cloth colored by the second group -/
def second_group_cloth_length : ℕ := 200

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 4

theorem women_group_size :
  first_group_size * second_group_cloth_length * first_group_days =
  second_group_size * first_group_cloth_length * second_group_days :=
by sorry

end NUMINAMATH_CALUDE_women_group_size_l3842_384224


namespace NUMINAMATH_CALUDE_total_raised_is_100_l3842_384271

/-- The amount of money raised by a local business for charity -/
def total_raised (num_tickets : ℕ) (ticket_price : ℚ) (donation_15 : ℕ) (donation_20 : ℕ) : ℚ :=
  num_tickets * ticket_price + donation_15 * 15 + donation_20 * 20

/-- Proof that the total amount raised is $100.00 -/
theorem total_raised_is_100 : total_raised 25 2 2 1 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_raised_is_100_l3842_384271


namespace NUMINAMATH_CALUDE_not_cube_sum_l3842_384294

theorem not_cube_sum (a b : ℕ) : ¬ ∃ c : ℤ, (a : ℤ)^3 + (b : ℤ)^3 + 4 = c^3 := by
  sorry

end NUMINAMATH_CALUDE_not_cube_sum_l3842_384294


namespace NUMINAMATH_CALUDE_target_destruction_probabilities_l3842_384263

/-- Represents the probability of a person hitting a target -/
def HitProbability := Fin 2 → Fin 2 → ℚ

/-- The probability of person A hitting targets -/
def probA : HitProbability := fun i j => 
  if i = j then 1/2 else 1/2

/-- The probability of person B hitting targets -/
def probB : HitProbability := fun i j => 
  if i = 0 ∧ j = 0 then 1/3
  else if i = 1 ∧ j = 1 then 2/5
  else 0

/-- The probability of a target being destroyed -/
def targetDestroyed (i : Fin 2) : ℚ :=
  probA i i * probB i i

/-- The probability of exactly one target being destroyed -/
def oneTargetDestroyed : ℚ :=
  (targetDestroyed 0) * (1 - probA 1 1) * (1 - probB 1 1) +
  (targetDestroyed 1) * (1 - probA 0 0) * (1 - probB 0 0)

theorem target_destruction_probabilities :
  (targetDestroyed 0 = 1/6) ∧
  (oneTargetDestroyed = 3/10) := by sorry

end NUMINAMATH_CALUDE_target_destruction_probabilities_l3842_384263


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3842_384200

-- Problem 1
theorem problem_1 (a b : ℝ) : 4 * a^4 * b^3 / ((-2 * a * b)^2) = a^2 * b :=
by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : (3*x - y)^2 - (3*x + 2*y) * (3*x - 2*y) = 5*y^2 - 6*x*y :=
by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3842_384200


namespace NUMINAMATH_CALUDE_sample_capacity_l3842_384228

/-- Given a sample divided into groups, prove that the sample capacity is 320
    when a certain group has a frequency of 40 and a rate of 0.125. -/
theorem sample_capacity (frequency : ℕ) (rate : ℝ) (n : ℕ) 
  (h1 : frequency = 40)
  (h2 : rate = 0.125)
  (h3 : (rate : ℝ) * n = frequency) : 
  n = 320 := by
  sorry

end NUMINAMATH_CALUDE_sample_capacity_l3842_384228


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l3842_384282

/-- The trajectory of point P given fixed points A and B -/
def trajectory (x y : ℝ) : Prop :=
  x^2 + y^2/2 = 1 ∧ x ≠ 1 ∧ x ≠ -1

/-- The line intersecting the trajectory -/
def intersecting_line (x y : ℝ) : Prop :=
  y = x + 1

/-- Theorem stating the properties of the trajectory and intersection -/
theorem trajectory_and_intersection :
  ∀ (x y : ℝ),
  (∀ (x' y' : ℝ), (y' / (x' + 1)) * (y' / (x' - 1)) = -2 → trajectory x' y') ∧
  (∃ (x1 y1 x2 y2 : ℝ),
    trajectory x1 y1 ∧ trajectory x2 y2 ∧
    intersecting_line x1 y1 ∧ intersecting_line x2 y2 ∧
    ((x1 - x2)^2 + (y1 - y2)^2)^(1/2 : ℝ) = 4 * Real.sqrt 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l3842_384282


namespace NUMINAMATH_CALUDE_solution_range_l3842_384260

def M (a : ℝ) := {x : ℝ | (a - 2) * x^2 + (2*a - 1) * x + 6 > 0}

theorem solution_range (a : ℝ) (h1 : 3 ∈ M a) (h2 : 5 ∉ M a) : 1 < a ∧ a ≤ 7/5 := by
  sorry

end NUMINAMATH_CALUDE_solution_range_l3842_384260


namespace NUMINAMATH_CALUDE_system_solution_existence_l3842_384245

theorem system_solution_existence (a b : ℤ) :
  (∃ x y : ℝ, ⌊x⌋ + 2 * y = a ∧ ⌊y⌋ + 2 * x = b) ↔
  (a + b) % 3 = 0 ∨ (a + b) % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_existence_l3842_384245


namespace NUMINAMATH_CALUDE_sqrt_calculation_l3842_384256

theorem sqrt_calculation : (Real.sqrt 8 + Real.sqrt 3) * Real.sqrt 6 - 3 * Real.sqrt 2 = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l3842_384256


namespace NUMINAMATH_CALUDE_ram_price_decrease_l3842_384259

theorem ram_price_decrease (initial_price increased_price final_price : ℝ) 
  (h1 : initial_price = 50)
  (h2 : increased_price = initial_price * 1.3)
  (h3 : final_price = 52) :
  (increased_price - final_price) / increased_price * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_ram_price_decrease_l3842_384259


namespace NUMINAMATH_CALUDE_sum_of_sqrt_ratios_geq_two_l3842_384289

theorem sum_of_sqrt_ratios_geq_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x / (y + z)) + Real.sqrt (y / (z + x)) + Real.sqrt (z / (x + y)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_ratios_geq_two_l3842_384289


namespace NUMINAMATH_CALUDE_positive_real_solution_l3842_384244

theorem positive_real_solution (x : ℝ) : 
  x > 0 → x * Real.sqrt (16 - x) + Real.sqrt (16 * x - x^3) ≥ 16 → 
  15 * x^2 + 32 * x - 256 = 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_solution_l3842_384244


namespace NUMINAMATH_CALUDE_project_payment_main_project_payment_l3842_384246

/-- Represents the project details and calculates the total payment -/
structure Project where
  q_wage : ℝ  -- Hourly wage of candidate q
  p_hours : ℝ  -- Hours required by candidate p to complete the project
  total_payment : ℝ  -- Total payment for the project

/-- Theorem stating the total payment for the project is $540 -/
theorem project_payment (proj : Project) : proj.total_payment = 540 :=
  by
  have h1 : proj.q_wage + proj.q_wage / 2 = proj.q_wage + 9 := by sorry
  have h2 : (proj.q_wage + proj.q_wage / 2) * proj.p_hours = proj.q_wage * (proj.p_hours + 10) := by sorry
  have h3 : proj.total_payment = (proj.q_wage + proj.q_wage / 2) * proj.p_hours := by sorry
  sorry

/-- Main theorem proving the project payment is $540 -/
theorem main_project_payment : ∃ (proj : Project), proj.total_payment = 540 :=
  by
  sorry

end NUMINAMATH_CALUDE_project_payment_main_project_payment_l3842_384246


namespace NUMINAMATH_CALUDE_manoj_lending_problem_l3842_384236

/-- Calculates simple interest -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem manoj_lending_problem (borrowed : ℚ) (borrowRate : ℚ) (lendRate : ℚ) (time : ℚ) (totalGain : ℚ)
  (h1 : borrowed = 3900)
  (h2 : borrowRate = 6)
  (h3 : lendRate = 9)
  (h4 : time = 3)
  (h5 : totalGain = 824.85)
  : ∃ (lentSum : ℚ), 
    lentSum = 5655 ∧ 
    simpleInterest lentSum lendRate time - simpleInterest borrowed borrowRate time = totalGain :=
sorry

end NUMINAMATH_CALUDE_manoj_lending_problem_l3842_384236


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l3842_384283

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h : c ≠ 0) : a / c = b / c → a = b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l3842_384283


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3842_384290

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, ∃ n₀ : ℤ, n₀ ≤ x^2) ↔ (∃ x₀ : ℝ, ∀ n : ℤ, n > x₀^2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3842_384290


namespace NUMINAMATH_CALUDE_isosceles_triangle_l3842_384293

theorem isosceles_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b > c ∧ b + c > a ∧ c + a > b →
  c / b = Real.cos C / Real.cos B →
  C = B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l3842_384293


namespace NUMINAMATH_CALUDE_algorithm_correctness_l3842_384229

def sum_2i (n : ℕ) : ℕ := 2 * (n * (n + 1) / 2)

theorem algorithm_correctness :
  (sum_2i 3 = 12) ∧
  (∀ m : ℕ, sum_2i m = 30 → m ≥ 5) ∧
  (sum_2i 5 = 30) := by
sorry

end NUMINAMATH_CALUDE_algorithm_correctness_l3842_384229


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l3842_384298

theorem subtraction_preserves_inequality (a b : ℝ) (h : a > b) : a - 1 > b - 1 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l3842_384298


namespace NUMINAMATH_CALUDE_profit_for_five_yuan_reduction_optimal_price_reduction_l3842_384281

/-- Represents the product details and sales dynamics -/
structure ProductSales where
  cost : ℕ  -- Cost per unit in yuan
  originalPrice : ℕ  -- Original selling price per unit in yuan
  initialSales : ℕ  -- Initial sales volume
  salesIncrease : ℕ  -- Increase in sales for every 1 yuan price reduction

/-- Calculates the profit for a given price reduction -/
def calculateProfit (p : ProductSales) (priceReduction : ℕ) : ℕ :=
  let newPrice := p.originalPrice - priceReduction
  let newSales := p.initialSales + p.salesIncrease * priceReduction
  (newPrice - p.cost) * newSales

/-- Theorem for the profit calculation with a 5 yuan price reduction -/
theorem profit_for_five_yuan_reduction (p : ProductSales) 
  (h1 : p.cost = 16) (h2 : p.originalPrice = 30) (h3 : p.initialSales = 200) (h4 : p.salesIncrease = 20) :
  calculateProfit p 5 = 2700 := by sorry

/-- Theorem for the optimal price reduction to achieve 2860 yuan profit -/
theorem optimal_price_reduction (p : ProductSales) 
  (h1 : p.cost = 16) (h2 : p.originalPrice = 30) (h3 : p.initialSales = 200) (h4 : p.salesIncrease = 20) :
  ∃ (x : ℕ), calculateProfit p x = 2860 ∧ 
    ∀ (y : ℕ), calculateProfit p y = 2860 → x ≤ y := by sorry

end NUMINAMATH_CALUDE_profit_for_five_yuan_reduction_optimal_price_reduction_l3842_384281


namespace NUMINAMATH_CALUDE_star_value_l3842_384206

-- Define the * operation for non-zero integers
def star (a b : ℤ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem star_value (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 12) (h4 : a * b = 32) :
  star a b = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l3842_384206


namespace NUMINAMATH_CALUDE_log_sum_equation_l3842_384215

theorem log_sum_equation (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  Real.log p + Real.log q = Real.log (p + q + p * q) → p = -q :=
by sorry

end NUMINAMATH_CALUDE_log_sum_equation_l3842_384215


namespace NUMINAMATH_CALUDE_distance_origin_to_line_l3842_384209

/-- The distance from the origin to a line passing through a given point with a given direction vector -/
theorem distance_origin_to_line (P : ℝ × ℝ) (n : ℝ × ℝ) : 
  P.1 = 2 ∧ P.2 = 0 ∧ n.1 = 1 ∧ n.2 = -1 →
  Real.sqrt ((P.1^2 + P.2^2) * (n.1^2 + n.2^2) - (P.1*n.1 + P.2*n.2)^2) / Real.sqrt (n.1^2 + n.2^2) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_distance_origin_to_line_l3842_384209


namespace NUMINAMATH_CALUDE_light_path_in_cube_l3842_384264

/-- Represents a point on the face of a cube -/
structure FacePoint where
  x : ℝ
  y : ℝ

/-- Represents the path of a light beam in a cube -/
def LightPath (cube_side : ℝ) (p : FacePoint) : ℝ := sorry

theorem light_path_in_cube (cube_side : ℝ) (p : FacePoint) 
  (h_side : cube_side = 10)
  (h_p : p = ⟨3, 4⟩) :
  ∃ (r s : ℕ), LightPath cube_side p = r * Real.sqrt s ∧ r + s = 55 ∧ 
  ∀ (prime : ℕ), Nat.Prime prime → ¬(s.gcd (prime ^ 2) > 1) := by
  sorry

end NUMINAMATH_CALUDE_light_path_in_cube_l3842_384264


namespace NUMINAMATH_CALUDE_castle_provisions_l3842_384296

/-- Represents the initial number of people in the castle -/
def initial_people : ℕ := sorry

/-- Represents the number of days the initial provisions last -/
def initial_days : ℕ := 90

/-- Represents the number of days after which people leave -/
def days_before_leaving : ℕ := 30

/-- Represents the number of people who leave the castle -/
def people_leaving : ℕ := 100

/-- Represents the number of days the remaining provisions last -/
def remaining_days : ℕ := 90

theorem castle_provisions :
  initial_people * initial_days = 
  (initial_people * days_before_leaving) + 
  ((initial_people - people_leaving) * remaining_days) ∧
  initial_people = 300 :=
sorry

end NUMINAMATH_CALUDE_castle_provisions_l3842_384296


namespace NUMINAMATH_CALUDE_money_saved_calculation_marcus_shopping_savings_l3842_384239

/-- Calculates the money saved when buying discounted items with sales tax --/
theorem money_saved_calculation (max_budget : ℝ) 
  (shoe_price shoe_discount : ℝ) 
  (sock_price sock_discount : ℝ) 
  (shirt_price shirt_discount : ℝ) 
  (sales_tax : ℝ) : ℝ :=
  let discounted_shoe := shoe_price * (1 - shoe_discount)
  let discounted_sock := sock_price * (1 - sock_discount)
  let discounted_shirt := shirt_price * (1 - shirt_discount)
  let total_before_tax := discounted_shoe + discounted_sock + discounted_shirt
  let final_cost := total_before_tax * (1 + sales_tax)
  let money_saved := max_budget - final_cost
  money_saved

/-- Proves that the money saved is approximately $34.22 --/
theorem marcus_shopping_savings : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |money_saved_calculation 200 120 0.3 25 0.2 55 0.1 0.08 - 34.22| < ε := by
  sorry

end NUMINAMATH_CALUDE_money_saved_calculation_marcus_shopping_savings_l3842_384239


namespace NUMINAMATH_CALUDE_x_value_from_ratio_l3842_384210

theorem x_value_from_ratio (x y : ℝ) :
  x / (x - 1) = (y^3 + 2*y - 1) / (y^3 + 2*y - 3) →
  x = (y^3 + 2*y - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_from_ratio_l3842_384210


namespace NUMINAMATH_CALUDE_unique_fraction_decomposition_l3842_384207

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Prime p) (h_odd : Odd p) :
  ∃! (m n : ℕ), m ≠ n ∧ 2 / p = 1 / n + 1 / m ∧ n = (p + 1) / 2 ∧ m = p * (p + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_decomposition_l3842_384207


namespace NUMINAMATH_CALUDE_store_traffic_proof_l3842_384243

/-- The number of people who entered the store in the first hour -/
def first_hour_entries : ℕ := 94

/-- The number of people who entered the store in the second hour -/
def second_hour_entries : ℕ := 18

/-- The number of people who left the store in the second hour -/
def second_hour_exits : ℕ := 9

/-- The number of people in the store after two hours -/
def final_count : ℕ := 76

/-- The number of people who left during the first hour -/
def first_hour_exits : ℕ := 27

theorem store_traffic_proof :
  first_hour_entries - first_hour_exits + second_hour_entries - second_hour_exits = final_count :=
by sorry

end NUMINAMATH_CALUDE_store_traffic_proof_l3842_384243


namespace NUMINAMATH_CALUDE_model1_is_best_fitting_l3842_384231

-- Define the structure for a regression model
structure RegressionModel where
  name : String
  r_squared : Real

-- Define the four models
def model1 : RegressionModel := ⟨"Model 1", 0.98⟩
def model2 : RegressionModel := ⟨"Model 2", 0.80⟩
def model3 : RegressionModel := ⟨"Model 3", 0.50⟩
def model4 : RegressionModel := ⟨"Model 4", 0.25⟩

-- Define a list of all models
def allModels : List RegressionModel := [model1, model2, model3, model4]

-- Define a function to determine if a model is the best fitting
def isBestFitting (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

-- Theorem stating that Model 1 is the best fitting model
theorem model1_is_best_fitting :
  isBestFitting model1 allModels := by
  sorry

end NUMINAMATH_CALUDE_model1_is_best_fitting_l3842_384231


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_is_three_twentythirds_l3842_384204

/-- Represents the cost of ingredients relative to raisins -/
structure RelativeCost where
  raisins : ℚ := 1
  nuts : ℚ := 4
  dried_berries : ℚ := 2

/-- Represents the composition of the mixture in pounds -/
structure MixtureComposition where
  raisins : ℚ := 3
  nuts : ℚ := 4
  dried_berries : ℚ := 2

/-- Calculates the fraction of total cost attributed to raisins -/
def raisin_cost_fraction (rc : RelativeCost) (mc : MixtureComposition) : ℚ :=
  (mc.raisins * rc.raisins) / 
  (mc.raisins * rc.raisins + mc.nuts * rc.nuts + mc.dried_berries * rc.dried_berries)

theorem raisin_cost_fraction_is_three_twentythirds 
  (rc : RelativeCost) (mc : MixtureComposition) : 
  raisin_cost_fraction rc mc = 3 / 23 := by
  sorry

end NUMINAMATH_CALUDE_raisin_cost_fraction_is_three_twentythirds_l3842_384204


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3842_384205

theorem greatest_divisor_with_remainders :
  ∃ (n : ℕ), n > 0 ∧
  1255 % n = 8 ∧
  1490 % n = 11 ∧
  ∀ (m : ℕ), m > n → (1255 % m ≠ 8 ∨ 1490 % m ≠ 11) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3842_384205


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3842_384267

/-- Given a hyperbola C and an ellipse with the following properties:
    1. C has the form x²/a² - y²/b² = 1 where a > 0 and b > 0
    2. C has an asymptote with equation y = (√5/2)x
    3. C shares a common focus with the ellipse x²/12 + y²/3 = 1
    Then the equation of C is x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) ∧ 
  (∃ (x y : ℝ), y = (Real.sqrt 5 / 2) * x) ∧
  (∃ (c : ℝ), c^2 = 3^2 ∧ c^2 = a^2 + b^2) →
  a^2 = 4 ∧ b^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3842_384267


namespace NUMINAMATH_CALUDE_cosine_of_arithmetic_sequence_l3842_384237

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem cosine_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 5 + a 9 = Real.pi) : 
  Real.cos (a 2 + a 8) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_arithmetic_sequence_l3842_384237


namespace NUMINAMATH_CALUDE_min_months_for_committee_repetition_l3842_384219

theorem min_months_for_committee_repetition 
  (total_members : Nat) 
  (women : Nat) 
  (men : Nat) 
  (committee_size : Nat) 
  (h1 : total_members = 13)
  (h2 : women = 6)
  (h3 : men = 7)
  (h4 : committee_size = 5)
  (h5 : women + men = total_members) :
  let total_committees := Nat.choose total_members committee_size
  let women_only_committees := Nat.choose women committee_size
  let men_only_committees := Nat.choose men committee_size
  let valid_committees := total_committees - women_only_committees - men_only_committees
  valid_committees + 1 = 1261 := by
  sorry

end NUMINAMATH_CALUDE_min_months_for_committee_repetition_l3842_384219


namespace NUMINAMATH_CALUDE_range_of_power_function_l3842_384203

theorem range_of_power_function (k c : ℝ) (h_k : k > 0) :
  Set.range (fun x => x^k + c) = Set.Ici (1 + c) := by sorry

end NUMINAMATH_CALUDE_range_of_power_function_l3842_384203


namespace NUMINAMATH_CALUDE_equation_solution_l3842_384211

open Real

theorem equation_solution (x : ℝ) (h : cos x ≠ 0) :
  (9 : ℝ) ^ (cos x) = (9 : ℝ) ^ (sin x) * (3 : ℝ) ^ (2 / cos x) ↔
  (∃ n : ℤ, x = n * π) ∨ (∃ k : ℤ, x = -π/4 + k * π) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l3842_384211


namespace NUMINAMATH_CALUDE_invisible_dots_count_l3842_384266

/-- The sum of numbers on a standard six-sided die -/
def dieSumOfFaces : ℕ := 21

/-- The number of dice in the stack -/
def numberOfDice : ℕ := 4

/-- The list of visible numbers on the stacked dice -/
def visibleNumbers : List ℕ := [1, 1, 2, 3, 4, 4, 5, 6]

/-- The theorem stating that the number of invisible dots is 58 -/
theorem invisible_dots_count : 
  numberOfDice * dieSumOfFaces - visibleNumbers.sum = 58 := by
  sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l3842_384266


namespace NUMINAMATH_CALUDE_jason_bought_four_dozens_l3842_384272

/-- The number of cupcakes Jason gives to each cousin -/
def cupcakes_per_cousin : ℕ := 3

/-- The number of cousins Jason has -/
def number_of_cousins : ℕ := 16

/-- The number of cupcakes in a dozen -/
def cupcakes_per_dozen : ℕ := 12

/-- Theorem: Jason bought 4 dozens of cupcakes -/
theorem jason_bought_four_dozens :
  (cupcakes_per_cousin * number_of_cousins) / cupcakes_per_dozen = 4 := by
  sorry

end NUMINAMATH_CALUDE_jason_bought_four_dozens_l3842_384272


namespace NUMINAMATH_CALUDE_rising_number_Q_l3842_384280

/-- Definition of a rising number -/
def is_rising_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = 1000*a + 100*b + 10*c + d ∧ 
  a < b ∧ b < c ∧ c < d ∧ a + d = b + c

/-- Function F as defined in the problem -/
def F (m : ℕ) : ℚ :=
  let m' := 1000*(m/10%10) + 100*(m/100%10) + 10*(m/1000) + (m%10)
  (m' - m) / 99

/-- Main theorem -/
theorem rising_number_Q (P Q : ℕ) (x y z t : ℕ) : 
  is_rising_number P ∧ 
  is_rising_number Q ∧
  P = 1000 + 100*x + 10*y + z ∧
  Q = 1000*x + 100*t + 60 + z ∧
  ∃ (k : ℤ), F P + F Q = k * 7 →
  Q = 3467 := by sorry

end NUMINAMATH_CALUDE_rising_number_Q_l3842_384280


namespace NUMINAMATH_CALUDE_chemist_problem_solution_l3842_384273

/-- Represents the purity of a salt solution as a real number between 0 and 1 -/
def Purity := { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

/-- The chemist's problem setup -/
structure ChemistProblem where
  solution1 : Purity
  solution2 : Purity
  total_amount : ℝ
  final_purity : Purity
  amount_solution1 : ℝ
  h1 : solution1.val = 0.3
  h2 : total_amount = 60
  h3 : final_purity.val = 0.5
  h4 : amount_solution1 = 40

theorem chemist_problem_solution (p : ChemistProblem) : p.solution2.val = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_chemist_problem_solution_l3842_384273


namespace NUMINAMATH_CALUDE_inequality_always_positive_l3842_384212

theorem inequality_always_positive (a : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) →
  -1/2 < a ∧ a < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_always_positive_l3842_384212


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3842_384234

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_focal : 4 * Real.sqrt 5 = 2 * Real.sqrt ((a^2 + b^2) : ℝ))
  (h_asymptote : b / a = 2) :
  a^2 = 4 ∧ b^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3842_384234


namespace NUMINAMATH_CALUDE_international_shipping_charge_l3842_384214

/-- The additional charge per letter for international shipping -/
def additional_charge (standard_postage : ℚ) (total_letters : ℕ) (international_letters : ℕ) (total_cost : ℚ) : ℚ :=
  ((total_cost - (standard_postage * total_letters)) / international_letters) * 100

theorem international_shipping_charge :
  let standard_postage : ℚ := 108 / 100  -- $1.08 in decimal form
  let total_letters : ℕ := 4
  let international_letters : ℕ := 2
  let total_cost : ℚ := 460 / 100  -- $4.60 in decimal form
  additional_charge standard_postage total_letters international_letters total_cost = 14 := by
  sorry

#eval additional_charge (108/100) 4 2 (460/100)

end NUMINAMATH_CALUDE_international_shipping_charge_l3842_384214
