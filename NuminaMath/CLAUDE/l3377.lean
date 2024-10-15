import Mathlib

namespace NUMINAMATH_CALUDE_min_pool_cost_l3377_337795

/-- Minimum cost for constructing a rectangular open-top water pool --/
theorem min_pool_cost (volume : ℝ) (depth : ℝ) (bottom_cost : ℝ) (wall_cost : ℝ) :
  volume = 8 →
  depth = 2 →
  bottom_cost = 120 →
  wall_cost = 80 →
  ∃ (cost : ℝ), cost = 1760 ∧ 
    ∀ (length width : ℝ),
      length > 0 →
      width > 0 →
      length * width * depth = volume →
      bottom_cost * length * width + wall_cost * (2 * length + 2 * width) * depth ≥ cost :=
by sorry

end NUMINAMATH_CALUDE_min_pool_cost_l3377_337795


namespace NUMINAMATH_CALUDE_zoo_trip_buses_l3377_337774

theorem zoo_trip_buses (total_students : ℕ) (students_per_bus : ℕ) (car_students : ℕ) : 
  total_students = 375 → students_per_bus = 53 → car_students = 4 →
  ((total_students - car_students + students_per_bus - 1) / students_per_bus : ℕ) = 8 := by
sorry

end NUMINAMATH_CALUDE_zoo_trip_buses_l3377_337774


namespace NUMINAMATH_CALUDE_tan_increasing_on_interval_l3377_337742

open Real

theorem tan_increasing_on_interval :
  StrictMonoOn tan (Set.Ioo (π / 2) π) := by
  sorry

end NUMINAMATH_CALUDE_tan_increasing_on_interval_l3377_337742


namespace NUMINAMATH_CALUDE_intersection_implies_m_greater_than_one_l3377_337754

/-- Given a parabola y = x^2 - x + 2 and a line y = x + m, if they intersect at two points, then m > 1 -/
theorem intersection_implies_m_greater_than_one :
  ∀ m : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁^2 - x₁ + 2 = x₁ + m) ∧ 
    (x₂^2 - x₂ + 2 = x₂ + m)) →
  m > 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_m_greater_than_one_l3377_337754


namespace NUMINAMATH_CALUDE_kaleb_spring_earnings_l3377_337736

/-- Represents Kaleb's lawn mowing business earnings and expenses -/
structure LawnMowingBusiness where
  spring_earnings : ℤ
  summer_earnings : ℤ
  supplies_cost : ℤ
  final_amount : ℤ

/-- Theorem stating Kaleb's spring earnings given the other known values -/
theorem kaleb_spring_earnings (business : LawnMowingBusiness)
  (h1 : business.summer_earnings = 50)
  (h2 : business.supplies_cost = 4)
  (h3 : business.final_amount = 50) :
  business.spring_earnings = 4 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_spring_earnings_l3377_337736


namespace NUMINAMATH_CALUDE_somu_age_problem_l3377_337731

/-- Proves that Somu was one-fifth of his father's age 5 years ago -/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_ago : ℕ) :
  somu_age = 10 →
  somu_age = father_age / 3 →
  somu_age - years_ago = (father_age - years_ago) / 5 →
  years_ago = 5 := by
  sorry

#check somu_age_problem

end NUMINAMATH_CALUDE_somu_age_problem_l3377_337731


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3377_337785

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n > 0 → 15 * n < 500 → 15 * n ≤ 495 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3377_337785


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3377_337701

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (6 * y^12 + 3 * y^11 + 6 * y^10 + 3 * y^9) =
  18 * y^13 - 3 * y^12 + 12 * y^11 - 3 * y^10 - 6 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3377_337701


namespace NUMINAMATH_CALUDE_no_real_solution_l3377_337713

theorem no_real_solution :
  ¬∃ (x y : ℝ), 4 * x^2 + 9 * y^2 - 16 * x - 36 * y + 64 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_l3377_337713


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3377_337777

theorem sin_2alpha_value (α : Real) 
  (h : Real.tan (α - π/4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3377_337777


namespace NUMINAMATH_CALUDE_parabola_directrix_l3377_337718

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := x = (y^2 - 8*y - 20) / 16

/-- The directrix equation -/
def directrix_eq (x : ℝ) : Prop := x = -6.25

/-- Theorem stating that the given directrix equation is correct for the parabola -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_eq x y → (∃ x_d : ℝ, directrix_eq x_d ∧ 
    -- Additional conditions about the relationship between the point (x,y) on the parabola
    -- and its distance to the directrix would be specified here
    True) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3377_337718


namespace NUMINAMATH_CALUDE_center_sum_l3377_337778

theorem center_sum (x y : ℝ) : 
  (∀ X Y : ℝ, X^2 + Y^2 + 4*X - 6*Y = 3 ↔ (X - x)^2 + (Y - y)^2 = 16) → 
  x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_center_sum_l3377_337778


namespace NUMINAMATH_CALUDE_range_of_m_l3377_337708

theorem range_of_m (x y m : ℝ) (h1 : 2/x + 1/y = 1) (h2 : x + y = 2 + 2*m) :
  -4 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3377_337708


namespace NUMINAMATH_CALUDE_decagon_equilateral_triangles_l3377_337722

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Count of distinct equilateral triangles in a regular polygon -/
def countDistinctEquilateralTriangles (n : ℕ) (p : RegularPolygon n) : ℕ :=
  sorry

/-- Theorem: In a ten-sided regular polygon, there are 82 distinct equilateral triangles
    with at least two vertices from the set of polygon vertices -/
theorem decagon_equilateral_triangles :
  ∀ (p : RegularPolygon 10), countDistinctEquilateralTriangles 10 p = 82 :=
by sorry

end NUMINAMATH_CALUDE_decagon_equilateral_triangles_l3377_337722


namespace NUMINAMATH_CALUDE_all_six_lines_tangent_l3377_337781

/-- A line in a plane -/
structure Line :=
  (id : ℕ)

/-- A circle in a plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Predicate to check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

/-- A set of six lines in a plane -/
def six_lines : Finset Line :=
  sorry

/-- Condition: For any three lines, there exists a fourth line such that all four are tangent to some circle -/
def four_line_tangent_condition (lines : Finset Line) : Prop :=
  ∀ (l1 l2 l3 : Line), l1 ∈ lines → l2 ∈ lines → l3 ∈ lines →
    ∃ (l4 : Line) (c : Circle), l4 ∈ lines ∧
      is_tangent l1 c ∧ is_tangent l2 c ∧ is_tangent l3 c ∧ is_tangent l4 c

/-- Theorem: If the four_line_tangent_condition holds for six lines, then all six lines are tangent to the same circle -/
theorem all_six_lines_tangent (h : four_line_tangent_condition six_lines) :
  ∃ (c : Circle), ∀ (l : Line), l ∈ six_lines → is_tangent l c :=
sorry

end NUMINAMATH_CALUDE_all_six_lines_tangent_l3377_337781


namespace NUMINAMATH_CALUDE_min_value_xy_l3377_337761

theorem min_value_xy (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h_geom : (Real.log x) * (Real.log y) = 1 / 16) : 
  x * y ≥ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_l3377_337761


namespace NUMINAMATH_CALUDE_julio_twice_james_age_l3377_337766

/-- 
Given:
- Julio is currently 36 years old
- James is currently 11 years old

Prove that in 14 years, Julio's age will be twice James's age
-/
theorem julio_twice_james_age (julio_age : ℕ) (james_age : ℕ) (years : ℕ) : 
  julio_age = 36 → james_age = 11 → years = 14 → 
  julio_age + years = 2 * (james_age + years) := by
  sorry

end NUMINAMATH_CALUDE_julio_twice_james_age_l3377_337766


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l3377_337739

/-- Given a geometric sequence where the fifth term is 48 and the sixth term is 72,
    prove that the second term of the sequence is 1152/81. -/
theorem geometric_sequence_second_term
  (a : ℚ) -- First term of the sequence
  (r : ℚ) -- Common ratio of the sequence
  (h1 : a * r^4 = 48) -- Fifth term is 48
  (h2 : a * r^5 = 72) -- Sixth term is 72
  : a * r = 1152 / 81 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l3377_337739


namespace NUMINAMATH_CALUDE_rachel_apples_l3377_337732

def initial_apples (num_trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ) : ℕ :=
  num_trees * apples_per_tree + remaining_apples

theorem rachel_apples : initial_apples 3 8 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_rachel_apples_l3377_337732


namespace NUMINAMATH_CALUDE_base_conversion_403_6_to_8_l3377_337784

/-- Converts a number from base 6 to base 10 --/
def base6_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base 10 to base 8 --/
def decimal_to_base8 (n : ℕ) : ℕ :=
  if n < 8 then n
  else (decimal_to_base8 (n / 8)) * 10 + (n % 8)

theorem base_conversion_403_6_to_8 :
  decimal_to_base8 (base6_to_decimal 403) = 223 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_403_6_to_8_l3377_337784


namespace NUMINAMATH_CALUDE_parallel_to_y_axis_fourth_quadrant_integer_a_l3377_337726

-- Define point A
def A (a : ℝ) : ℝ × ℝ := (3*a - 9, 2*a - 10)

-- Define point B
def B : ℝ × ℝ := (4, 5)

-- Theorem 1
theorem parallel_to_y_axis (a : ℝ) : 
  (A a).1 = B.1 → a = 13/3 := by sorry

-- Theorem 2
theorem fourth_quadrant_integer_a : 
  ∃ (a : ℤ), (A a).1 > 0 ∧ (A a).2 < 0 → A a = (3, -2) := by sorry

end NUMINAMATH_CALUDE_parallel_to_y_axis_fourth_quadrant_integer_a_l3377_337726


namespace NUMINAMATH_CALUDE_volleyball_match_probability_l3377_337727

/-- The probability of winning a single set for class 6 of senior year two -/
def win_prob : ℚ := 2/3

/-- The number of sets needed to win the match -/
def sets_to_win : ℕ := 3

/-- The probability of class 6 of senior year two winning by 3:0 -/
def prob_win_3_0 : ℚ := win_prob^sets_to_win

theorem volleyball_match_probability :
  prob_win_3_0 = 8/27 :=
sorry

end NUMINAMATH_CALUDE_volleyball_match_probability_l3377_337727


namespace NUMINAMATH_CALUDE_hyperbola_range_l3377_337765

/-- A function that represents the equation of a hyperbola -/
def hyperbola_equation (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / (m + 2) - y^2 / (2*m - 1) = 1

/-- The theorem stating the range of m for which the equation represents a hyperbola -/
theorem hyperbola_range (m : ℝ) :
  (∀ x y, ∃ (h : hyperbola_equation m x y), True) ↔ m < -2 ∨ m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_range_l3377_337765


namespace NUMINAMATH_CALUDE_success_permutations_l3377_337757

def word := "SUCCESS"

-- Define the counts of each letter
def s_count := 3
def c_count := 2
def u_count := 1
def e_count := 1

-- Define the total number of letters
def total_letters := s_count + c_count + u_count + e_count

-- Theorem statement
theorem success_permutations :
  (Nat.factorial total_letters) / 
  (Nat.factorial s_count * Nat.factorial c_count * Nat.factorial u_count * Nat.factorial e_count) = 420 := by
  sorry

end NUMINAMATH_CALUDE_success_permutations_l3377_337757


namespace NUMINAMATH_CALUDE_percent_of_x_is_y_l3377_337752

theorem percent_of_x_is_y (x y : ℝ) (h : 0.6 * (x - y) = 0.2 * (x + y)) : y = 0.5 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_is_y_l3377_337752


namespace NUMINAMATH_CALUDE_almond_walnut_ratio_is_five_to_two_l3377_337749

/-- Represents a mixture of nuts with almonds and walnuts. -/
structure NutMixture where
  total_weight : ℝ
  almond_weight : ℝ
  almond_parts : ℝ
  walnut_parts : ℝ

/-- The ratio of almonds to walnuts in the mixture. -/
def almond_walnut_ratio (mix : NutMixture) : ℝ × ℝ :=
  (mix.almond_parts, mix.walnut_parts)

theorem almond_walnut_ratio_is_five_to_two
  (mix : NutMixture)
  (h1 : mix.total_weight = 350)
  (h2 : mix.almond_weight = 250)
  (h3 : mix.walnut_parts = 2)
  (h4 : mix.almond_parts * mix.walnut_parts = mix.almond_weight * mix.walnut_parts) :
  almond_walnut_ratio mix = (5, 2) := by
  sorry

end NUMINAMATH_CALUDE_almond_walnut_ratio_is_five_to_two_l3377_337749


namespace NUMINAMATH_CALUDE_union_of_sets_l3377_337717

theorem union_of_sets : let A : Set ℕ := {2, 3}
                        let B : Set ℕ := {1, 2}
                        A ∪ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3377_337717


namespace NUMINAMATH_CALUDE_quadratic_max_value_l3377_337791

theorem quadratic_max_value :
  ∃ (M : ℝ), M = 34 ∧ ∀ (q : ℝ), -3 * q^2 + 18 * q + 7 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l3377_337791


namespace NUMINAMATH_CALUDE_store_earnings_is_120_l3377_337710

/-- Represents the store's sales policy and outcomes -/
structure StoreSales where
  pencil_count : ℕ
  eraser_per_pencil : ℕ
  eraser_price : ℚ
  pencil_price : ℚ

/-- Calculates the total earnings from pencil and eraser sales -/
def total_earnings (s : StoreSales) : ℚ :=
  s.pencil_count * s.pencil_price + 
  s.pencil_count * s.eraser_per_pencil * s.eraser_price

/-- Theorem stating that the store's earnings are $120 given the specified conditions -/
theorem store_earnings_is_120 (s : StoreSales) 
  (h1 : s.eraser_per_pencil = 2)
  (h2 : s.eraser_price = 1)
  (h3 : s.pencil_price = 2 * s.eraser_per_pencil * s.eraser_price)
  (h4 : s.pencil_count = 20) : 
  total_earnings s = 120 := by
  sorry

#eval total_earnings { pencil_count := 20, eraser_per_pencil := 2, eraser_price := 1, pencil_price := 4 }

end NUMINAMATH_CALUDE_store_earnings_is_120_l3377_337710


namespace NUMINAMATH_CALUDE_nadia_bought_20_roses_l3377_337702

/-- Represents the number of roses Nadia bought -/
def roses : ℕ := 20

/-- Represents the number of lilies Nadia bought -/
def lilies : ℚ := (3 / 4) * roses

/-- Cost of a single rose in dollars -/
def rose_cost : ℚ := 5

/-- Cost of a single lily in dollars -/
def lily_cost : ℚ := 2 * rose_cost

/-- Total amount spent on flowers in dollars -/
def total_spent : ℚ := 250

theorem nadia_bought_20_roses :
  roses * rose_cost + lilies * lily_cost = total_spent := by sorry

end NUMINAMATH_CALUDE_nadia_bought_20_roses_l3377_337702


namespace NUMINAMATH_CALUDE_x_is_negative_l3377_337751

theorem x_is_negative (x y : ℝ) (h1 : y ≠ 0) (h2 : y > 0) (h3 : x / y < -3) : x < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_is_negative_l3377_337751


namespace NUMINAMATH_CALUDE_optimal_discount_order_l3377_337730

/-- Proves that the optimal order of applying discounts results in an additional savings of 125 cents --/
theorem optimal_discount_order (initial_price : ℝ) (flat_discount : ℝ) (percent_discount : ℝ) :
  initial_price = 30 →
  flat_discount = 5 →
  percent_discount = 0.25 →
  ((initial_price - flat_discount) * (1 - percent_discount) - 
   (initial_price * (1 - percent_discount) - flat_discount)) * 100 = 125 := by
  sorry

end NUMINAMATH_CALUDE_optimal_discount_order_l3377_337730


namespace NUMINAMATH_CALUDE_affordable_housing_theorem_l3377_337738

/-- Represents the affordable housing investment and construction scenario -/
structure AffordableHousing where
  investment_2011 : ℝ
  area_2011 : ℝ
  total_investment : ℝ
  growth_rate : ℝ

/-- The affordable housing scenario satisfies the given conditions -/
def valid_scenario (ah : AffordableHousing) : Prop :=
  ah.investment_2011 = 200 ∧
  ah.area_2011 = 0.08 ∧
  ah.total_investment = 950 ∧
  ah.investment_2011 * (1 + ah.growth_rate + (1 + ah.growth_rate)^2) = ah.total_investment

/-- The growth rate is 50% and the total area built is 38 million square meters -/
theorem affordable_housing_theorem (ah : AffordableHousing) 
  (h : valid_scenario ah) : 
  ah.growth_rate = 0.5 ∧ 
  ah.total_investment / (ah.investment_2011 / ah.area_2011) = 38 := by
  sorry


end NUMINAMATH_CALUDE_affordable_housing_theorem_l3377_337738


namespace NUMINAMATH_CALUDE_gears_rotating_when_gear1_rotates_l3377_337776

-- Define the state of a gear (rotating or stopped)
inductive GearState
| rotating
| stopped

-- Define the gearbox with 6 gears
structure Gearbox :=
  (gear1 gear2 gear3 gear4 gear5 gear6 : GearState)

-- Define the conditions of the gearbox operation
def validGearbox (gb : Gearbox) : Prop :=
  -- Condition 1
  (gb.gear1 = GearState.rotating → gb.gear2 = GearState.rotating ∧ gb.gear5 = GearState.stopped) ∧
  -- Condition 2
  ((gb.gear2 = GearState.rotating ∨ gb.gear5 = GearState.rotating) → gb.gear4 = GearState.stopped) ∧
  -- Condition 3
  (gb.gear3 = GearState.rotating ↔ gb.gear4 = GearState.rotating) ∧
  -- Condition 4
  (gb.gear5 = GearState.rotating ∨ gb.gear6 = GearState.rotating)

-- Theorem statement
theorem gears_rotating_when_gear1_rotates (gb : Gearbox) :
  validGearbox gb →
  gb.gear1 = GearState.rotating →
  gb.gear2 = GearState.rotating ∧ gb.gear3 = GearState.rotating ∧ gb.gear6 = GearState.rotating :=
by sorry

end NUMINAMATH_CALUDE_gears_rotating_when_gear1_rotates_l3377_337776


namespace NUMINAMATH_CALUDE_range_of_a_f_lower_bound_l3377_337700

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a - 1| + |x - 2*a|

-- Theorem 1: Range of a when f(1) < 3
theorem range_of_a (a : ℝ) : f 1 a < 3 → a ∈ Set.Ioo (-2/3) (4/3) :=
sorry

-- Theorem 2: f(x) ≥ 2 when a ≥ 1 and x ∈ ℝ
theorem f_lower_bound (a x : ℝ) : a ≥ 1 → f x a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_f_lower_bound_l3377_337700


namespace NUMINAMATH_CALUDE_sugar_problem_l3377_337775

theorem sugar_problem (initial_sugar : ℝ) : 
  (initial_sugar / 4 * 3.5 = 21) → initial_sugar = 24 := by
  sorry

end NUMINAMATH_CALUDE_sugar_problem_l3377_337775


namespace NUMINAMATH_CALUDE_calculation_proof_l3377_337773

theorem calculation_proof : 
  (3/5 : ℚ) * 200 + (456/1000 : ℚ) * 875 + (7/8 : ℚ) * 320 - 
  ((5575/10000 : ℚ) * 1280 + (1/3 : ℚ) * 960) = -2349/10 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3377_337773


namespace NUMINAMATH_CALUDE_wheel_rotations_per_block_l3377_337748

theorem wheel_rotations_per_block 
  (total_blocks : ℕ) 
  (initial_rotations : ℕ) 
  (additional_rotations : ℕ) : 
  total_blocks = 8 → 
  initial_rotations = 600 → 
  additional_rotations = 1000 → 
  (initial_rotations + additional_rotations) / total_blocks = 200 := by
sorry

end NUMINAMATH_CALUDE_wheel_rotations_per_block_l3377_337748


namespace NUMINAMATH_CALUDE_elephant_donkey_weight_l3377_337745

/-- Calculates the combined weight of an elephant and a donkey in pounds -/
theorem elephant_donkey_weight (elephant_tons : ℝ) (donkey_percent_less : ℝ) : 
  elephant_tons = 3 ∧ donkey_percent_less = 90 →
  elephant_tons * 2000 + (elephant_tons * 2000 * (1 - donkey_percent_less / 100)) = 6600 := by
  sorry

end NUMINAMATH_CALUDE_elephant_donkey_weight_l3377_337745


namespace NUMINAMATH_CALUDE_min_distance_between_curve_and_line_l3377_337767

theorem min_distance_between_curve_and_line :
  ∀ (a b c d : ℝ),
  (Real.log b + 1 + a - 3 * b = 0) →
  (2 * d - c + Real.sqrt 5 = 0) →
  (∃ (m : ℝ), ∀ (a' b' c' d' : ℝ),
    (Real.log b' + 1 + a' - 3 * b' = 0) →
    (2 * d' - c' + Real.sqrt 5 = 0) →
    (a - c)^2 + (b - d)^2 ≤ (a' - c')^2 + (b' - d')^2) →
  m = 4/5 := by
sorry

end NUMINAMATH_CALUDE_min_distance_between_curve_and_line_l3377_337767


namespace NUMINAMATH_CALUDE_fewer_buses_on_river_road_l3377_337772

theorem fewer_buses_on_river_road (num_cars : ℕ) (num_buses : ℕ) : 
  num_cars = 60 →
  num_buses < num_cars →
  num_buses * 3 = num_cars →
  num_cars - num_buses = 40 := by
  sorry

end NUMINAMATH_CALUDE_fewer_buses_on_river_road_l3377_337772


namespace NUMINAMATH_CALUDE_system_solution_l3377_337756

theorem system_solution :
  let f (x y : ℝ) := y + Real.sqrt (y - 3*x) + 3*x = 12
  let g (x y : ℝ) := y^2 + y - 3*x - 9*x^2 = 144
  ∀ x y : ℝ, (f x y ∧ g x y) ↔ ((x = -4/3 ∧ y = 12) ∨ (x = -24 ∧ y = 72)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3377_337756


namespace NUMINAMATH_CALUDE_farm_entrance_fee_l3377_337769

theorem farm_entrance_fee (num_students : ℕ) (num_adults : ℕ) (student_fee : ℕ) (total_cost : ℕ) :
  num_students = 35 →
  num_adults = 4 →
  student_fee = 5 →
  total_cost = 199 →
  ∃ (adult_fee : ℕ), 
    adult_fee = 6 ∧ 
    num_students * student_fee + num_adults * adult_fee = total_cost :=
by sorry

end NUMINAMATH_CALUDE_farm_entrance_fee_l3377_337769


namespace NUMINAMATH_CALUDE_regular_polygon_with_108_degree_interior_angles_l3377_337789

theorem regular_polygon_with_108_degree_interior_angles (n : ℕ) : 
  (n ≥ 3) →  -- ensuring it's a valid polygon
  (((n - 2) * 180) / n = 108) →  -- interior angle formula
  (n = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_108_degree_interior_angles_l3377_337789


namespace NUMINAMATH_CALUDE_min_side_length_l3377_337735

theorem min_side_length (PQ PR SR SQ : ℝ) (h1 : PQ = 7) (h2 : PR = 15) (h3 : SR = 10) (h4 : SQ = 25) :
  ∀ QR : ℝ, (QR > PR - PQ ∧ QR > SQ - SR) → QR ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_min_side_length_l3377_337735


namespace NUMINAMATH_CALUDE_solution_set_range_of_m_l3377_337723

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |x + 1|

-- Theorem for part (Ⅰ)
theorem solution_set (x : ℝ) : f x ≤ 2 ↔ 0 ≤ x ∧ x ≤ 2/3 := by sorry

-- Theorem for part (Ⅱ)
theorem range_of_m (m : ℝ) : 
  (∀ x, ∃ a ∈ Set.Icc (-2) 1, f x ≥ f a + m) → m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_range_of_m_l3377_337723


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l3377_337716

/-- In a right-angled triangle ABC, the sum of two specific arctangent expressions equals π/4 -/
theorem right_triangle_arctan_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  Real.arctan (a / (Real.sqrt b + Real.sqrt c)) + Real.arctan (b / (Real.sqrt a + Real.sqrt c)) = π/4 := by
  sorry

#check right_triangle_arctan_sum

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l3377_337716


namespace NUMINAMATH_CALUDE_function_property_l3377_337787

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem function_property (f : ℝ → ℝ) 
  (h1 : is_even_function f)
  (h2 : ∀ x, f (x - 3/2) = f (x + 1/2))
  (h3 : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
sorry

end NUMINAMATH_CALUDE_function_property_l3377_337787


namespace NUMINAMATH_CALUDE_cake_cutting_l3377_337724

/-- Represents a square cake -/
structure Cake where
  side : ℕ
  pieces : ℕ

/-- The maximum number of pieces obtainable with a single straight cut -/
def max_pieces_single_cut (c : Cake) : ℕ := sorry

/-- The minimum number of straight cuts required to intersect all original pieces -/
def min_cuts_all_pieces (c : Cake) : ℕ := sorry

/-- The theorem statement -/
theorem cake_cutting (c : Cake) 
  (h1 : c.side = 4) 
  (h2 : c.pieces = 16) : 
  max_pieces_single_cut c = 23 ∧ min_cuts_all_pieces c = 3 := by sorry

end NUMINAMATH_CALUDE_cake_cutting_l3377_337724


namespace NUMINAMATH_CALUDE_exists_non_prime_l3377_337794

/-- The recurrence relation for the sequence x_n -/
def recurrence (x₀ a b : ℕ) : ℕ → ℕ
| 0 => x₀
| n + 1 => recurrence x₀ a b n * a + b

/-- Theorem: There exists a non-prime number in the sequence defined by the recurrence relation -/
theorem exists_non_prime (x₀ a b : ℕ) : ∃ n : ℕ, ¬ Nat.Prime (recurrence x₀ a b n) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_prime_l3377_337794


namespace NUMINAMATH_CALUDE_sum_of_coefficients_factorization_l3377_337788

theorem sum_of_coefficients_factorization (x y : ℝ) : 
  (∃ a b c d e f g h i j : ℤ, 
    27 * x^9 - 512 * y^9 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + i*x*y + j*y^2) ∧
    a + b + c + d + e + f + g + h + i + j = 32) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_factorization_l3377_337788


namespace NUMINAMATH_CALUDE_quadratic_linear_intersection_l3377_337728

theorem quadratic_linear_intersection
  (a d : ℝ) (x₁ x₂ : ℝ) (h_a : a ≠ 0) (h_d : d ≠ 0) (h_x : x₁ ≠ x₂)
  (y₁ : ℝ → ℝ) (y₂ : ℝ → ℝ) (y : ℝ → ℝ)
  (h_y₁ : ∀ x, y₁ x = a * (x - x₁) * (x - x₂))
  (h_y₂ : ∃ e, ∀ x, y₂ x = d * x + e)
  (h_intersect : y₂ x₁ = 0)
  (h_single_root : ∃! x, y x = 0) :
  x₂ - x₁ = d / a := by sorry

end NUMINAMATH_CALUDE_quadratic_linear_intersection_l3377_337728


namespace NUMINAMATH_CALUDE_shared_side_angle_measure_l3377_337779

-- Define the properties of the figure
def regular_pentagon (P : Set Point) : Prop := sorry

def equilateral_triangle (T : Set Point) : Prop := sorry

def share_side (P T : Set Point) : Prop := sorry

-- Define the angle we're interested in
def angle_at_vertex (T : Set Point) (v : Point) : ℝ := sorry

-- Theorem statement
theorem shared_side_angle_measure 
  (P T : Set Point) (v : Point) :
  regular_pentagon P → 
  equilateral_triangle T → 
  share_side P T → 
  angle_at_vertex T v = 6 := by sorry

end NUMINAMATH_CALUDE_shared_side_angle_measure_l3377_337779


namespace NUMINAMATH_CALUDE_highest_a_divisible_by_8_l3377_337705

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def last_three_digits (n : ℕ) : ℕ := n % 1000

theorem highest_a_divisible_by_8 :
  ∀ a : ℕ, a ≤ 9 →
    (is_divisible_by_8 (365 * 100 + a * 10 + 16) ↔ a ≤ 8) ∧
    (∀ b : ℕ, b > 8 ∧ b ≤ 9 → ¬ is_divisible_by_8 (365 * 100 + b * 10 + 16)) :=
sorry

end NUMINAMATH_CALUDE_highest_a_divisible_by_8_l3377_337705


namespace NUMINAMATH_CALUDE_sqrt_two_thirds_less_than_half_l3377_337790

theorem sqrt_two_thirds_less_than_half : (Real.sqrt 2) / 3 < 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_thirds_less_than_half_l3377_337790


namespace NUMINAMATH_CALUDE_max_cross_section_area_l3377_337770

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  sideLength : ℝ
  sideLength_pos : 0 < sideLength

/-- Represents a plane that cuts the tetrahedron parallel to two opposite edges -/
structure CuttingPlane where
  tetrahedron : RegularTetrahedron
  distanceFromEdge : ℝ
  distance_nonneg : 0 ≤ distanceFromEdge
  distance_bound : distanceFromEdge ≤ tetrahedron.sideLength

/-- The area of the cross-section formed by the cutting plane -/
def crossSectionArea (plane : CuttingPlane) : ℝ :=
  plane.distanceFromEdge * (plane.tetrahedron.sideLength - plane.distanceFromEdge)

/-- The theorem stating that the maximum cross-section area is a²/4 -/
theorem max_cross_section_area (t : RegularTetrahedron) :
  ∃ (plane : CuttingPlane), plane.tetrahedron = t ∧
  ∀ (p : CuttingPlane), p.tetrahedron = t →
  crossSectionArea p ≤ crossSectionArea plane ∧
  crossSectionArea plane = t.sideLength^2 / 4 :=
sorry

end NUMINAMATH_CALUDE_max_cross_section_area_l3377_337770


namespace NUMINAMATH_CALUDE_y_coord_Q_l3377_337733

/-- A line passing through the origin with slope 0.8 -/
def line (x : ℝ) : ℝ := 0.8 * x

/-- The x-coordinate of point Q -/
def x_coord_Q : ℝ := 6

/-- Theorem: The y-coordinate of point Q is 4.8 -/
theorem y_coord_Q : line x_coord_Q = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_y_coord_Q_l3377_337733


namespace NUMINAMATH_CALUDE_playground_children_count_l3377_337755

theorem playground_children_count : 
  ∀ (girls boys : ℕ), 
  girls = 28 → 
  boys = 35 → 
  girls + boys = 63 := by
sorry

end NUMINAMATH_CALUDE_playground_children_count_l3377_337755


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l3377_337793

/-- A geometric sequence with a₁ = 1 and a₅ = 16 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 5 = 16 ∧ ∀ n : ℕ, n ≥ 1 → ∃ r : ℝ, a (n + 1) = a n * r

/-- Theorem: In a geometric sequence with a₁ = 1 and a₅ = 16, a₃ = 4 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (h : geometric_sequence a) : a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l3377_337793


namespace NUMINAMATH_CALUDE_basketball_league_games_l3377_337714

/-- Calculates the total number of games in a basketball league -/
def total_games (n : ℕ) (regular_games_per_pair : ℕ) (knockout_games_per_team : ℕ) : ℕ :=
  let regular_season_games := (n * (n - 1) / 2) * regular_games_per_pair
  let knockout_games := n * knockout_games_per_team / 2
  regular_season_games + knockout_games

/-- Theorem: In a 12-team basketball league where each team plays 4 games against every other team
    and participates in 2 knockout matches, the total number of games is 276 -/
theorem basketball_league_games :
  total_games 12 4 2 = 276 := by
  sorry

end NUMINAMATH_CALUDE_basketball_league_games_l3377_337714


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_l3377_337746

theorem quadratic_always_nonnegative (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + 1 ≥ 0) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_l3377_337746


namespace NUMINAMATH_CALUDE_selling_price_is_correct_l3377_337712

/-- Calculates the selling price per copy of a program given the production cost,
    advertisement revenue, number of copies to be sold, and desired profit. -/
def calculate_selling_price (production_cost : ℚ) (ad_revenue : ℚ) (copies : ℕ) (desired_profit : ℚ) : ℚ :=
  (desired_profit + (production_cost * copies) - ad_revenue) / copies

theorem selling_price_is_correct : 
  let production_cost : ℚ := 70/100
  let ad_revenue : ℚ := 15000
  let copies : ℕ := 35000
  let desired_profit : ℚ := 8000
  calculate_selling_price production_cost ad_revenue copies desired_profit = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_is_correct_l3377_337712


namespace NUMINAMATH_CALUDE_range_of_a_l3377_337786

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3377_337786


namespace NUMINAMATH_CALUDE_triangle_inequality_l3377_337709

/-- Given a triangle ABC with area t, perimeter k, and circumradius R, 
    prove that 4tR ≤ (k/3)³ -/
theorem triangle_inequality (t k R : ℝ) (h_positive : t > 0 ∧ k > 0 ∧ R > 0) :
  4 * t * R ≤ (k / 3) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3377_337709


namespace NUMINAMATH_CALUDE_third_speed_calculation_l3377_337706

/-- Prove that given the conditions, the third speed is 3 km/hr -/
theorem third_speed_calculation (total_time : ℝ) (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 11 →
  total_distance = 900 →
  speed1 = 6 →
  speed2 = 9 →
  ∃ (speed3 : ℝ), speed3 = 3 ∧
    total_time = (total_distance / 3) / (speed1 * 1000 / 60) +
                 (total_distance / 3) / (speed2 * 1000 / 60) +
                 (total_distance / 3) / (speed3 * 1000 / 60) :=
by sorry


end NUMINAMATH_CALUDE_third_speed_calculation_l3377_337706


namespace NUMINAMATH_CALUDE_probability_one_pair_l3377_337799

def total_gloves : ℕ := 10
def pairs_of_gloves : ℕ := 5
def gloves_picked : ℕ := 4

def total_ways : ℕ := Nat.choose total_gloves gloves_picked

def ways_one_pair : ℕ := 
  Nat.choose pairs_of_gloves 1 * Nat.choose 2 2 * Nat.choose (total_gloves - 2) (gloves_picked - 2)

theorem probability_one_pair :
  (ways_one_pair : ℚ) / total_ways = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_one_pair_l3377_337799


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3377_337768

theorem complex_equation_solution : ∃ (z : ℂ), 3 - 2 * Complex.I * z = 7 + 4 * Complex.I * z ∧ z = (2 * Complex.I) / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3377_337768


namespace NUMINAMATH_CALUDE_dot_product_special_vectors_l3377_337725

theorem dot_product_special_vectors :
  let a : ℝ × ℝ := (Real.sin (15 * π / 180), Real.sin (75 * π / 180))
  let b : ℝ × ℝ := (Real.cos (30 * π / 180), Real.sin (30 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_special_vectors_l3377_337725


namespace NUMINAMATH_CALUDE_clock_angle_at_seven_l3377_337782

/-- The number of degrees in a full circle -/
def full_circle : ℕ := 360

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees per hour on a clock face -/
def degrees_per_hour : ℕ := full_circle / clock_hours

/-- The hour we're examining -/
def current_hour : ℕ := 7

/-- The smaller angle between the hour hand and 12 o'clock position -/
def smaller_angle : ℕ := min (current_hour * degrees_per_hour) ((clock_hours - current_hour) * degrees_per_hour)

theorem clock_angle_at_seven : smaller_angle = 150 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_seven_l3377_337782


namespace NUMINAMATH_CALUDE_mangoes_rate_per_kg_l3377_337747

/-- Given Bruce's purchase of grapes and mangoes, prove the rate per kg for mangoes. -/
theorem mangoes_rate_per_kg 
  (grapes_quantity : ℕ) 
  (grapes_rate : ℕ) 
  (mangoes_quantity : ℕ) 
  (total_paid : ℕ) 
  (h1 : grapes_quantity = 8)
  (h2 : grapes_rate = 70)
  (h3 : mangoes_quantity = 11)
  (h4 : total_paid = 1165)
  (h5 : total_paid = grapes_quantity * grapes_rate + mangoes_quantity * (total_paid - grapes_quantity * grapes_rate) / mangoes_quantity) : 
  (total_paid - grapes_quantity * grapes_rate) / mangoes_quantity = 55 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_rate_per_kg_l3377_337747


namespace NUMINAMATH_CALUDE_representative_selection_cases_l3377_337704

def number_of_female_students : ℕ := 4
def number_of_male_students : ℕ := 6

theorem representative_selection_cases :
  (number_of_female_students * number_of_male_students) = 24 :=
by sorry

end NUMINAMATH_CALUDE_representative_selection_cases_l3377_337704


namespace NUMINAMATH_CALUDE_scientific_notation_of_15000_l3377_337780

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_15000 :
  toScientificNotation 15000 = ScientificNotation.mk 1.5 4 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_15000_l3377_337780


namespace NUMINAMATH_CALUDE_equation_negative_root_l3377_337753

theorem equation_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 4^x - 2^(x-1) + a = 0) ↔ -1/2 < a ∧ a ≤ 1/16 := by
  sorry

end NUMINAMATH_CALUDE_equation_negative_root_l3377_337753


namespace NUMINAMATH_CALUDE_select_students_l3377_337764

theorem select_students (n m : ℕ) (h1 : n = 10) (h2 : m = 3) : 
  Nat.choose n m = 120 := by
  sorry

end NUMINAMATH_CALUDE_select_students_l3377_337764


namespace NUMINAMATH_CALUDE_difference_is_2_5q_minus_15_l3377_337760

/-- The difference in dimes between two people's quarter amounts -/
def difference_in_dimes (q : ℝ) : ℝ :=
  let samantha_quarters : ℝ := 3 * q + 2
  let bob_quarters : ℝ := 2 * q + 8
  let quarter_to_dime : ℝ := 2.5
  quarter_to_dime * (samantha_quarters - bob_quarters)

/-- Theorem stating the difference in dimes -/
theorem difference_is_2_5q_minus_15 (q : ℝ) :
  difference_in_dimes q = 2.5 * q - 15 := by
  sorry

end NUMINAMATH_CALUDE_difference_is_2_5q_minus_15_l3377_337760


namespace NUMINAMATH_CALUDE_min_value_fraction_l3377_337707

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hsum : b + 2*a = 8) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 8 → 2/(x*y) ≥ 2/(a*b)) ∧ 2/(a*b) = 1/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l3377_337707


namespace NUMINAMATH_CALUDE_function_property_l3377_337797

/-- Given a function f(x) = ax^5 + bx^3 + 2 where f(2) = 7, prove that f(-2) = -3 -/
theorem function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + 2
  (f 2 = 7) → (f (-2) = -3) := by
sorry

end NUMINAMATH_CALUDE_function_property_l3377_337797


namespace NUMINAMATH_CALUDE_math_team_combinations_l3377_337759

theorem math_team_combinations : ℕ := by
  -- Define the total number of girls and boys in the math club
  let total_girls : ℕ := 5
  let total_boys : ℕ := 5
  
  -- Define the number of girls and boys needed for the team
  let team_girls : ℕ := 3
  let team_boys : ℕ := 3
  
  -- Define the total team size
  let team_size : ℕ := team_girls + team_boys
  
  -- Calculate the number of ways to choose the team
  let result := (total_girls.choose team_girls) * (total_boys.choose team_boys)
  
  -- Prove that the result is equal to 100
  have h : result = 100 := by sorry
  
  -- Return the result
  exact result

end NUMINAMATH_CALUDE_math_team_combinations_l3377_337759


namespace NUMINAMATH_CALUDE_cube_max_volume_l3377_337771

/-- A cuboid with side lengths a, b, and c. -/
structure Cuboid where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The surface area of a cuboid. -/
def surfaceArea (x : Cuboid) : ℝ :=
  2 * (x.a * x.b + x.b * x.c + x.a * x.c)

/-- The volume of a cuboid. -/
def volume (x : Cuboid) : ℝ :=
  x.a * x.b * x.c

/-- Given a fixed surface area S, the cube maximizes the volume among all cuboids. -/
theorem cube_max_volume (S : ℝ) (h : 0 < S) :
  ∀ x : Cuboid, surfaceArea x = S →
    ∃ y : Cuboid, surfaceArea y = S ∧ y.a = y.b ∧ y.b = y.c ∧
      ∀ z : Cuboid, surfaceArea z = S → volume z ≤ volume y :=
by sorry

end NUMINAMATH_CALUDE_cube_max_volume_l3377_337771


namespace NUMINAMATH_CALUDE_largest_integer_solution_l3377_337798

theorem largest_integer_solution : ∃ (x : ℤ), x ≤ 20 ∧ |x - 3| = 15 ∧ ∀ (y : ℤ), y ≤ 20 ∧ |y - 3| = 15 → y ≤ x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l3377_337798


namespace NUMINAMATH_CALUDE_bicycle_shop_inventory_l3377_337750

/-- Represents the bicycle shop inventory problem --/
theorem bicycle_shop_inventory
  (initial_stock : ℕ)
  (weekly_addition : ℕ)
  (weeks : ℕ)
  (final_stock : ℕ)
  (h1 : initial_stock = 51)
  (h2 : weekly_addition = 3)
  (h3 : weeks = 4)
  (h4 : final_stock = 45) :
  initial_stock + weekly_addition * weeks - final_stock = 18 :=
by sorry

end NUMINAMATH_CALUDE_bicycle_shop_inventory_l3377_337750


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l3377_337720

theorem quadratic_root_zero (m : ℝ) : 
  (∃ x, (m - 1) * x^2 + x + m^2 - 1 = 0) ∧ 
  ((m - 1) * 0^2 + 0 + m^2 - 1 = 0) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l3377_337720


namespace NUMINAMATH_CALUDE_min_abs_alpha_plus_gamma_l3377_337758

theorem min_abs_alpha_plus_gamma :
  ∀ (α γ : ℂ),
  let g := λ (z : ℂ) => (3 + I) * z^2 + α * z + γ
  (g 1).im = 0 →
  (g I).im = 0 →
  ∃ (α₀ γ₀ : ℂ),
    (let g₀ := λ (z : ℂ) => (3 + I) * z^2 + α₀ * z + γ₀
     (g₀ 1).im = 0 ∧
     (g₀ I).im = 0 ∧
     Complex.abs α₀ + Complex.abs γ₀ = Real.sqrt 2 ∧
     ∀ (α' γ' : ℂ),
       (let g' := λ (z : ℂ) => (3 + I) * z^2 + α' * z + γ'
        (g' 1).im = 0 ∧
        (g' I).im = 0 →
        Complex.abs α' + Complex.abs γ' ≥ Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_min_abs_alpha_plus_gamma_l3377_337758


namespace NUMINAMATH_CALUDE_yellow_flowers_killed_correct_l3377_337711

/-- Represents the number of flowers of each color --/
structure FlowerCounts where
  red : ℕ
  yellow : ℕ
  orange : ℕ
  purple : ℕ

/-- Represents the problem parameters --/
structure BouquetProblem where
  seeds_per_color : ℕ
  flowers_per_bouquet : ℕ
  total_bouquets : ℕ
  killed_flowers : FlowerCounts

def yellow_flowers_killed (problem : BouquetProblem) : ℕ :=
  problem.seeds_per_color -
    (problem.total_bouquets * problem.flowers_per_bouquet -
      (problem.seeds_per_color - problem.killed_flowers.red +
       problem.seeds_per_color - problem.killed_flowers.orange +
       problem.seeds_per_color - problem.killed_flowers.purple))

theorem yellow_flowers_killed_correct (problem : BouquetProblem) :
  problem.seeds_per_color = 125 →
  problem.flowers_per_bouquet = 9 →
  problem.total_bouquets = 36 →
  problem.killed_flowers.red = 45 →
  problem.killed_flowers.orange = 30 →
  problem.killed_flowers.purple = 40 →
  yellow_flowers_killed problem = 61 := by
  sorry

#eval yellow_flowers_killed {
  seeds_per_color := 125,
  flowers_per_bouquet := 9,
  total_bouquets := 36,
  killed_flowers := {
    red := 45,
    yellow := 0,  -- This value doesn't affect the calculation
    orange := 30,
    purple := 40
  }
}

end NUMINAMATH_CALUDE_yellow_flowers_killed_correct_l3377_337711


namespace NUMINAMATH_CALUDE_bagel_cost_is_1_50_l3377_337762

/-- The cost of a cup of coffee -/
def coffee_cost : ℝ := sorry

/-- The cost of a bagel -/
def bagel_cost : ℝ := sorry

/-- Condition 1: 3 cups of coffee and 2 bagels cost $12.75 -/
axiom condition1 : 3 * coffee_cost + 2 * bagel_cost = 12.75

/-- Condition 2: 2 cups of coffee and 5 bagels cost $14.00 -/
axiom condition2 : 2 * coffee_cost + 5 * bagel_cost = 14.00

/-- Theorem: The cost of one bagel is $1.50 -/
theorem bagel_cost_is_1_50 : bagel_cost = 1.50 := by sorry

end NUMINAMATH_CALUDE_bagel_cost_is_1_50_l3377_337762


namespace NUMINAMATH_CALUDE_expression_evaluation_l3377_337744

theorem expression_evaluation :
  let x : ℚ := 3
  let f (y : ℚ) := (y + 3) / (y - 2)
  3 * (f (f x) + 3) / (f (f x) - 2) = 27 / 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3377_337744


namespace NUMINAMATH_CALUDE_teachers_class_size_l3377_337792

/-- The number of students in Teacher Yang's class -/
def num_students : ℕ := 28

theorem teachers_class_size :
  (num_students / 2 : ℕ) +     -- Half in math competition
  (num_students / 4 : ℕ) +     -- Quarter in music group
  (num_students / 7 : ℕ) +     -- One-seventh in reading room
  3 =                          -- Remaining three watching TV
  num_students :=              -- Equals total number of students
by sorry

end NUMINAMATH_CALUDE_teachers_class_size_l3377_337792


namespace NUMINAMATH_CALUDE_average_decrease_l3377_337740

theorem average_decrease (n : ℕ) (old_avg new_obs : ℚ) : 
  n = 6 → 
  old_avg = 14 → 
  new_obs = 7 → 
  old_avg - (n * old_avg + new_obs) / (n + 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_average_decrease_l3377_337740


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3377_337737

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * Real.sqrt 3 * x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 2 * Real.sqrt 3 * y + k = 0 → y = x) → 
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3377_337737


namespace NUMINAMATH_CALUDE_count_special_numbers_is_360_l3377_337743

/-- A function that counts 4-digit numbers beginning with 1 and having exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits := Finset.range 10  -- digits from 0 to 9
  let non_one_digits := digits.erase 1  -- digits excluding 1
  let case1 := 3 * non_one_digits.card * (non_one_digits.card - 1)  -- case where one of the identical digits is 1
  let case2 := 2 * non_one_digits.card * digits.card  -- case where the identical digits are not 1
  case1 + case2

/-- Theorem stating that the count of special numbers is 360 -/
theorem count_special_numbers_is_360 : count_special_numbers = 360 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_is_360_l3377_337743


namespace NUMINAMATH_CALUDE_andrews_age_l3377_337763

theorem andrews_age (a : ℕ) (g : ℕ) : 
  g = 12 * a →  -- Andrew's grandfather's age is twelve times Andrew's age
  g - a = 55 →  -- Andrew's grandfather was 55 years old when Andrew was born
  a = 5 :=       -- Andrew's age is 5 years
by sorry

end NUMINAMATH_CALUDE_andrews_age_l3377_337763


namespace NUMINAMATH_CALUDE_f_plus_3_abs_l3377_337719

noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 < x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- undefined for other x values

theorem f_plus_3_abs (x : ℝ) (hx : -3 ≤ x ∧ x ≤ 3) : 
  |f x + 3| = f x + 3 :=
by sorry

end NUMINAMATH_CALUDE_f_plus_3_abs_l3377_337719


namespace NUMINAMATH_CALUDE_reciprocal_in_fourth_quadrant_l3377_337715

theorem reciprocal_in_fourth_quadrant (i : ℂ) (z : ℂ) :
  i * i = -1 →
  z = 1 + i →
  let w := 1 / z
  0 < w.re ∧ w.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_in_fourth_quadrant_l3377_337715


namespace NUMINAMATH_CALUDE_circle_plus_equality_l3377_337734

/-- Definition of the ⊕ operation -/
def circle_plus (a b : ℝ) : ℝ := a * b + a - b

/-- Theorem stating the equality to be proved -/
theorem circle_plus_equality (a b : ℝ) : 
  circle_plus a b + circle_plus (b - a) b = b^2 - b := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_equality_l3377_337734


namespace NUMINAMATH_CALUDE_solve_system_l3377_337703

theorem solve_system (x y : ℝ) (eq1 : x + y = 15) (eq2 : x - y = 5) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3377_337703


namespace NUMINAMATH_CALUDE_pudding_cost_pudding_cost_is_two_l3377_337741

/-- The cost of each cup of pudding, given the conditions of Jane's purchase -/
theorem pudding_cost (num_ice_cream : ℕ) (num_pudding : ℕ) (ice_cream_price : ℕ) (extra_spent : ℕ) : ℕ :=
  let total_ice_cream := num_ice_cream * ice_cream_price
  let pudding_cost := (total_ice_cream - extra_spent) / num_pudding
  pudding_cost

/-- Proof that each cup of pudding costs $2 -/
theorem pudding_cost_is_two :
  pudding_cost 15 5 5 65 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pudding_cost_pudding_cost_is_two_l3377_337741


namespace NUMINAMATH_CALUDE_tan_22_5_deg_l3377_337796

theorem tan_22_5_deg (h1 : Real.pi / 4 = 2 * (22.5 * Real.pi / 180)) 
  (h2 : Real.tan (Real.pi / 4) = 1) :
  Real.tan (22.5 * Real.pi / 180) / (1 - Real.tan (22.5 * Real.pi / 180)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_l3377_337796


namespace NUMINAMATH_CALUDE_residue_1237_mod_17_l3377_337729

theorem residue_1237_mod_17 : 1237 % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_residue_1237_mod_17_l3377_337729


namespace NUMINAMATH_CALUDE_distributive_property_negative_three_l3377_337721

theorem distributive_property_negative_three (a b : ℝ) : -3 * (-a - b) = 3 * a + 3 * b := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_negative_three_l3377_337721


namespace NUMINAMATH_CALUDE_largest_angle_120_l3377_337783

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ)

-- Define properties of the triangle
def isObtuse (t : Triangle) : Prop :=
  t.P > 90 ∨ t.Q > 90 ∨ t.R > 90

def isIsosceles (t : Triangle) : Prop :=
  (t.P = t.Q) ∨ (t.Q = t.R) ∨ (t.P = t.R)

def angleP30 (t : Triangle) : Prop :=
  t.P = 30

-- Theorem statement
theorem largest_angle_120 (t : Triangle) 
  (h1 : isObtuse t) 
  (h2 : isIsosceles t) 
  (h3 : angleP30 t) : 
  max t.P (max t.Q t.R) = 120 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_120_l3377_337783
