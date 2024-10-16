import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2388_238869

def M : Set ℤ := {m : ℤ | -3 < m ∧ m < 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2388_238869


namespace NUMINAMATH_CALUDE_min_value_f_l2388_238835

theorem min_value_f (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ π) (hy : 0 ≤ y ∧ y ≤ 1) :
  (2 * y - 1) * Real.sin x + (1 - y) * Real.sin ((1 - y) * x) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_l2388_238835


namespace NUMINAMATH_CALUDE_triangle_side_b_l2388_238839

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_side_b (t : Triangle) : 
  t.C = 4 * t.A →  -- ∠C = 4∠A
  t.a = 15 →       -- side a = 15
  t.c = 60 →       -- side c = 60
  t.b = 15 * Real.sqrt (2 + Real.sqrt 2) := by
    sorry


end NUMINAMATH_CALUDE_triangle_side_b_l2388_238839


namespace NUMINAMATH_CALUDE_find_divisor_l2388_238880

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 95 ∧ quotient = 6 ∧ remainder = 5 →
  ∃ divisor : Nat, dividend = divisor * quotient + remainder ∧ divisor = 15 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2388_238880


namespace NUMINAMATH_CALUDE_bd_length_l2388_238886

-- Define the triangles and their properties
def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem bd_length (c : ℝ) :
  ∀ (AB BC AC AD BD : ℝ),
  right_triangle BC AC AB →
  right_triangle AD BD AB →
  BC = 3 →
  AC = c →
  AD = c - 1 →
  BD = Real.sqrt (2 * c + 8) := by
  sorry

end NUMINAMATH_CALUDE_bd_length_l2388_238886


namespace NUMINAMATH_CALUDE_housewife_spending_fraction_l2388_238888

theorem housewife_spending_fraction (initial_amount : ℝ) (remaining_amount : ℝ)
  (h1 : initial_amount = 150)
  (h2 : remaining_amount = 50) :
  (initial_amount - remaining_amount) / initial_amount = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_housewife_spending_fraction_l2388_238888


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l2388_238840

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l2388_238840


namespace NUMINAMATH_CALUDE_greatest_integer_radius_for_circle_l2388_238822

theorem greatest_integer_radius_for_circle (A : ℝ) (h : A < 50 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi ≤ A ∧ ∀ (s : ℕ), s * s * Real.pi ≤ A → s ≤ r ∧ r = 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_for_circle_l2388_238822


namespace NUMINAMATH_CALUDE_point_on_direct_proportion_l2388_238808

/-- A direct proportion function passing through two points -/
def DirectProportion (k : ℝ) (x y : ℝ) : Prop := y = k * x

/-- The theorem stating that if A(3,-5) and B(-6,a) lie on a direct proportion function, then a = 10 -/
theorem point_on_direct_proportion (k a : ℝ) :
  DirectProportion k 3 (-5) ∧ DirectProportion k (-6) a → a = 10 := by
  sorry

end NUMINAMATH_CALUDE_point_on_direct_proportion_l2388_238808


namespace NUMINAMATH_CALUDE_weightlifting_time_l2388_238830

def practice_duration : ℕ := 120  -- 2 hours in minutes

theorem weightlifting_time (shooting_time running_time weightlifting_time : ℕ) :
  shooting_time = practice_duration / 2 →
  running_time + weightlifting_time = practice_duration - shooting_time →
  running_time = 2 * weightlifting_time →
  weightlifting_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_weightlifting_time_l2388_238830


namespace NUMINAMATH_CALUDE_parabola_translation_l2388_238833

/-- Represents a parabola in 2D space -/
structure Parabola where
  f : ℝ → ℝ

/-- Applies a vertical translation to a parabola -/
def verticalTranslate (p : Parabola) (v : ℝ) : Parabola where
  f := fun x => p.f x + v

/-- Applies a horizontal translation to a parabola -/
def horizontalTranslate (p : Parabola) (h : ℝ) : Parabola where
  f := fun x => p.f (x + h)

/-- The original parabola y = -x^2 -/
def originalParabola : Parabola where
  f := fun x => -x^2

/-- Theorem stating that translating the parabola y = -x^2 upward by 2 units
    and to the left by 3 units results in the equation y = -(x + 3)^2 + 2 -/
theorem parabola_translation :
  (horizontalTranslate (verticalTranslate originalParabola 2) 3).f =
  fun x => -(x + 3)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l2388_238833


namespace NUMINAMATH_CALUDE_sqrt_square_abs_sqrt_neg_nine_squared_l2388_238881

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

theorem sqrt_neg_nine_squared : Real.sqrt ((-9)^2) = 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_sqrt_neg_nine_squared_l2388_238881


namespace NUMINAMATH_CALUDE_mixture_composition_l2388_238893

theorem mixture_composition (water_percent_1 water_percent_2 mixture_percent : ℝ)
  (parts_1 : ℝ) (h1 : water_percent_1 = 0.20)
  (h2 : water_percent_2 = 0.35) (h3 : parts_1 = 10)
  (h4 : mixture_percent = 0.24285714285714285) : ∃ (parts_2 : ℝ),
  parts_2 = 4 ∧ 
  (water_percent_1 * parts_1 + water_percent_2 * parts_2) / (parts_1 + parts_2) = mixture_percent :=
by sorry

end NUMINAMATH_CALUDE_mixture_composition_l2388_238893


namespace NUMINAMATH_CALUDE_investment_growth_l2388_238842

/-- The initial investment amount -/
def P : ℝ := 248.52

/-- The interest rate as a decimal -/
def r : ℝ := 0.12

/-- The number of years -/
def n : ℕ := 6

/-- The final amount -/
def A : ℝ := 500

/-- Theorem stating that the initial investment P, when compounded annually
    at rate r for n years, results in approximately the final amount A -/
theorem investment_growth (ε : ℝ) (h_ε : ε > 0) : 
  |P * (1 + r)^n - A| < ε := by
  sorry


end NUMINAMATH_CALUDE_investment_growth_l2388_238842


namespace NUMINAMATH_CALUDE_equation_solution_l2388_238870

theorem equation_solution :
  ∃ y : ℚ, (3 / y - 3 / y * y / 5 = 1.2) ∧ (y = 5 / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2388_238870


namespace NUMINAMATH_CALUDE_bus_riders_percentage_l2388_238806

/-- Represents the scenario of introducing a bus service in Johnstown --/
structure BusScenario where
  population : Nat
  car_pollution : Nat
  bus_pollution : Nat
  bus_capacity : Nat
  carbon_reduction : Nat

/-- Calculates the percentage of people who now take the bus --/
def percentage_bus_riders (scenario : BusScenario) : Rat :=
  let cars_removed := scenario.carbon_reduction / scenario.car_pollution
  (cars_removed : Rat) / scenario.population * 100

/-- Theorem stating that the percentage of people who now take the bus is 12.5% --/
theorem bus_riders_percentage (scenario : BusScenario) 
  (h1 : scenario.population = 80)
  (h2 : scenario.car_pollution = 10)
  (h3 : scenario.bus_pollution = 100)
  (h4 : scenario.bus_capacity = 40)
  (h5 : scenario.carbon_reduction = 100) :
  percentage_bus_riders scenario = 25/2 := by
  sorry

#eval percentage_bus_riders {
  population := 80,
  car_pollution := 10,
  bus_pollution := 100,
  bus_capacity := 40,
  carbon_reduction := 100
}

end NUMINAMATH_CALUDE_bus_riders_percentage_l2388_238806


namespace NUMINAMATH_CALUDE_balloon_difference_l2388_238891

/-- The number of balloons Allan brought to the park -/
def allan_initial : ℕ := 2

/-- The number of balloons Allan bought at the park -/
def allan_bought : ℕ := 3

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 6

/-- The total number of balloons Allan had in the park -/
def allan_total : ℕ := allan_initial + allan_bought

theorem balloon_difference : jake_balloons - allan_total = 1 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l2388_238891


namespace NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l2388_238883

/-- Given a point C with coordinates (x, -5) and its reflection D over the y-axis,
    the sum of all coordinate values of C and D is -10. -/
theorem sum_of_coordinates_after_reflection (x : ℝ) :
  let C : ℝ × ℝ := (x, -5)
  let D : ℝ × ℝ := (-x, -5)  -- reflection of C over y-axis
  (C.1 + C.2 + D.1 + D.2) = -10 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l2388_238883


namespace NUMINAMATH_CALUDE_tank_volume_ratio_l2388_238827

theorem tank_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (5/8 : ℝ) * V₂ →
  V₁ / V₂ = 5/6 := by
sorry

end NUMINAMATH_CALUDE_tank_volume_ratio_l2388_238827


namespace NUMINAMATH_CALUDE_simple_random_sampling_prob_std_dev_transformation_l2388_238829

/-- Simple random sampling probability -/
theorem simple_random_sampling_prob (population_size : ℕ) (sample_size : ℕ) :
  population_size = 50 → sample_size = 10 →
  (sample_size : ℝ) / (population_size : ℝ) = 0.2 := by sorry

/-- Standard deviation transformation -/
theorem std_dev_transformation (x : Fin 10 → ℝ) (σ : ℝ) :
  Real.sqrt (Finset.univ.sum (λ i => (x i - Finset.univ.sum x / 10) ^ 2) / 10) = σ →
  Real.sqrt (Finset.univ.sum (λ i => ((2 * x i - 1) - Finset.univ.sum (λ j => 2 * x j - 1) / 10) ^ 2) / 10) = 2 * σ := by sorry

end NUMINAMATH_CALUDE_simple_random_sampling_prob_std_dev_transformation_l2388_238829


namespace NUMINAMATH_CALUDE_ball_removal_probability_l2388_238825

/-- A jar containing red and blue balls -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- The probability of an event occurring during the ball removal process -/
def removal_probability (initial : Jar) (event : Jar → Prop) : ℚ :=
  sorry

/-- The event of having more blue balls than red balls -/
def more_blue_than_red (j : Jar) : Prop :=
  j.blue > j.red

/-- The theorem statement -/
theorem ball_removal_probability :
  let initial_jar := Jar.mk 8 2
  removal_probability initial_jar more_blue_than_red = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ball_removal_probability_l2388_238825


namespace NUMINAMATH_CALUDE_subtraction_of_squares_l2388_238847

theorem subtraction_of_squares (a : ℝ) : 3 * a^2 - a^2 = 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_squares_l2388_238847


namespace NUMINAMATH_CALUDE_max_pie_pieces_l2388_238860

theorem max_pie_pieces : 
  (∃ (n : ℕ), n > 0 ∧ 
    ∃ (A B : ℕ), 
      10000 ≤ A ∧ A < 100000 ∧ 
      10000 ≤ B ∧ B < 100000 ∧ 
      A = B * n ∧ 
      (∀ (i j : Fin 5), i ≠ j → (A / 10^i.val % 10) ≠ (A / 10^j.val % 10)) ∧
    ∀ (m : ℕ), m > n → 
      ¬(∃ (C D : ℕ), 
        10000 ≤ C ∧ C < 100000 ∧ 
        10000 ≤ D ∧ D < 100000 ∧ 
        C = D * m ∧ 
        (∀ (i j : Fin 5), i ≠ j → (C / 10^i.val % 10) ≠ (C / 10^j.val % 10)))) ∧
  (∃ (A B : ℕ), 
    10000 ≤ A ∧ A < 100000 ∧ 
    10000 ≤ B ∧ B < 100000 ∧ 
    A = B * 7 ∧ 
    (∀ (i j : Fin 5), i ≠ j → (A / 10^i.val % 10) ≠ (A / 10^j.val % 10))) :=
by
  sorry

end NUMINAMATH_CALUDE_max_pie_pieces_l2388_238860


namespace NUMINAMATH_CALUDE_probability_four_threes_eight_dice_l2388_238859

def probability_four_threes (n m k : ℕ) : ℚ :=
  (n.choose k : ℚ) * (1 / m) ^ k * ((m - 1) / m) ^ (n - k)

theorem probability_four_threes_eight_dice :
  probability_four_threes 8 6 4 = 43750 / 1679616 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_threes_eight_dice_l2388_238859


namespace NUMINAMATH_CALUDE_largest_unique_solution_m_l2388_238873

theorem largest_unique_solution_m (x y : ℕ) (m : ℕ) : 
  (∃! (x y : ℕ), 2005 * x + 2007 * y = m) → m ≤ 2 * 2005 * 2007 ∧ 
  (∃! (x y : ℕ), 2005 * x + 2007 * y = 2 * 2005 * 2007) :=
sorry

end NUMINAMATH_CALUDE_largest_unique_solution_m_l2388_238873


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l2388_238851

theorem subtraction_of_decimals : 3.57 - 1.45 = 2.12 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l2388_238851


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2388_238807

def M : Set ℝ := {x | (x - 1) * (x - 2) > 0}
def N : Set ℝ := {x | x^2 + x < 0}

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x ∈ N → x ∈ M) ∧
  (∃ x : ℝ, x ∈ M ∧ x ∉ N) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2388_238807


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2388_238899

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a ≥ 6 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    (a₀^2 + b₀^2) / c₀ + (a₀^2 + c₀^2) / b₀ + (b₀^2 + c₀^2) / a₀ = 6 :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_expression_l2388_238899


namespace NUMINAMATH_CALUDE_point_above_plane_l2388_238858

theorem point_above_plane (a : ℝ) : 
  (∀ x y : ℝ, x = 1 ∧ y = 2 → x + y + a > 0) ↔ a > -3 :=
sorry

end NUMINAMATH_CALUDE_point_above_plane_l2388_238858


namespace NUMINAMATH_CALUDE_min_product_abc_l2388_238864

theorem min_product_abc (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 1 → 
  a ≤ 3*b → a ≤ 3*c → b ≤ 3*a → b ≤ 3*c → c ≤ 3*a → c ≤ 3*b → 
  a * b * c ≥ 9 / 343 := by
sorry

end NUMINAMATH_CALUDE_min_product_abc_l2388_238864


namespace NUMINAMATH_CALUDE_two_number_problem_l2388_238828

theorem two_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 8.58 := by
sorry

end NUMINAMATH_CALUDE_two_number_problem_l2388_238828


namespace NUMINAMATH_CALUDE_birthday_number_proof_l2388_238819

theorem birthday_number_proof : ∃! T : ℕ+,
  (∃ x y : ℕ, 4 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧
    T ^ 2 = 40000 + x * 1000 + y * 100 + 29) ∧
  T = 223 := by
  sorry

end NUMINAMATH_CALUDE_birthday_number_proof_l2388_238819


namespace NUMINAMATH_CALUDE_product_of_cosines_l2388_238890

theorem product_of_cosines : 
  (1 + Real.cos (π / 6)) * (1 + Real.cos (π / 3)) * 
  (1 + Real.cos ((2 * π) / 3)) * (1 + Real.cos ((5 * π) / 6)) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_l2388_238890


namespace NUMINAMATH_CALUDE_simplify_polynomial_no_x_squared_l2388_238866

-- Define the polynomial
def polynomial (x m : ℝ) : ℝ := 4*x^2 - 3*x + 5 - 2*m*x^2 - x + 1

-- Define the coefficient of x^2
def coeff_x_squared (m : ℝ) : ℝ := 4 - 2*m

-- Theorem statement
theorem simplify_polynomial_no_x_squared :
  ∃ (m : ℝ), coeff_x_squared m = 0 ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_simplify_polynomial_no_x_squared_l2388_238866


namespace NUMINAMATH_CALUDE_video_game_lives_l2388_238846

theorem video_game_lives (initial_players : ℕ) (quitting_players : ℕ) (total_lives : ℕ) :
  initial_players = 8 →
  quitting_players = 3 →
  total_lives = 15 →
  (total_lives / (initial_players - quitting_players) : ℚ) = 3 := by
sorry

end NUMINAMATH_CALUDE_video_game_lives_l2388_238846


namespace NUMINAMATH_CALUDE_trapezoid_construction_uniqueness_l2388_238897

-- Define the necessary types
def Line : Type := ℝ → ℝ → Prop
def Point : Type := ℝ × ℝ
def Direction : Type := ℝ × ℝ

-- Define the trapezoid structure
structure Trapezoid where
  side1 : Line
  side2 : Line
  diag1_start : Point
  diag1_end : Point
  diag2_direction : Direction

-- Define the theorem
theorem trapezoid_construction_uniqueness 
  (side1 side2 : Line)
  (E F : Point)
  (diag2_dir : Direction) :
  ∃! (trap : Trapezoid), 
    trap.side1 = side1 ∧ 
    trap.side2 = side2 ∧ 
    trap.diag1_start = E ∧ 
    trap.diag1_end = F ∧ 
    trap.diag2_direction = diag2_dir :=
sorry

end NUMINAMATH_CALUDE_trapezoid_construction_uniqueness_l2388_238897


namespace NUMINAMATH_CALUDE_nellie_gift_wrap_sales_l2388_238887

theorem nellie_gift_wrap_sales (total_goal : ℕ) (sold_to_uncle : ℕ) (sold_to_neighbor : ℕ) (remaining_to_sell : ℕ) :
  total_goal = 45 →
  sold_to_uncle = 10 →
  sold_to_neighbor = 6 →
  remaining_to_sell = 28 →
  total_goal - remaining_to_sell - (sold_to_uncle + sold_to_neighbor) = 1 :=
by sorry

end NUMINAMATH_CALUDE_nellie_gift_wrap_sales_l2388_238887


namespace NUMINAMATH_CALUDE_sample_size_is_thirty_l2388_238844

/-- Represents the ratio of young, middle-aged, and elderly employees -/
structure EmployeeRatio :=
  (young : ℕ)
  (middle : ℕ)
  (elderly : ℕ)

/-- Calculates the total sample size given the ratio and number of young employees in the sample -/
def calculateSampleSize (ratio : EmployeeRatio) (youngInSample : ℕ) : ℕ :=
  let totalRatio := ratio.young + ratio.middle + ratio.elderly
  (youngInSample * totalRatio) / ratio.young

/-- Theorem stating that for the given ratio and number of young employees, the sample size is 30 -/
theorem sample_size_is_thirty :
  let ratio : EmployeeRatio := { young := 7, middle := 5, elderly := 3 }
  let youngInSample : ℕ := 14
  calculateSampleSize ratio youngInSample = 30 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_thirty_l2388_238844


namespace NUMINAMATH_CALUDE_twenty_eight_billion_scientific_notation_l2388_238884

/-- Represents 28 billion -/
def twenty_eight_billion : ℕ := 28000000000

/-- The scientific notation representation of 28 billion -/
def scientific_notation : ℝ := 2.8 * (10 ^ 9)

theorem twenty_eight_billion_scientific_notation : 
  (twenty_eight_billion : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_twenty_eight_billion_scientific_notation_l2388_238884


namespace NUMINAMATH_CALUDE_total_cost_theorem_l2388_238882

/-- The cost of items and their relationships -/
structure ItemCosts where
  pencil_cost : ℝ
  pen_cost : ℝ
  notebook_cost : ℝ
  pen_pencil_diff : ℝ
  notebook_pen_ratio : ℝ
  notebook_discount : ℝ
  cad_usd_rate : ℝ

/-- Calculate the total cost in USD -/
def total_cost_usd (costs : ItemCosts) : ℝ :=
  let pen_cost := costs.pencil_cost + costs.pen_pencil_diff
  let notebook_cost := costs.notebook_pen_ratio * pen_cost
  let discounted_notebook_cost := notebook_cost * (1 - costs.notebook_discount)
  let total_cad := costs.pencil_cost + pen_cost + discounted_notebook_cost
  total_cad * costs.cad_usd_rate

/-- Theorem stating the total cost in USD -/
theorem total_cost_theorem (costs : ItemCosts) 
  (h1 : costs.pencil_cost = 2)
  (h2 : costs.pen_pencil_diff = 9)
  (h3 : costs.notebook_pen_ratio = 2)
  (h4 : costs.notebook_discount = 0.15)
  (h5 : costs.cad_usd_rate = 1.25) :
  total_cost_usd costs = 39.63 := by
  sorry

#eval total_cost_usd {
  pencil_cost := 2,
  pen_cost := 11,
  notebook_cost := 22,
  pen_pencil_diff := 9,
  notebook_pen_ratio := 2,
  notebook_discount := 0.15,
  cad_usd_rate := 1.25
}

end NUMINAMATH_CALUDE_total_cost_theorem_l2388_238882


namespace NUMINAMATH_CALUDE_evaluate_expression_l2388_238865

theorem evaluate_expression : 
  (3^1005 + 4^1006)^2 - (3^1005 - 4^1006)^2 = 16 * 12^1005 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2388_238865


namespace NUMINAMATH_CALUDE_prop_c_prop_d_l2388_238898

-- Proposition C
theorem prop_c (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = 1) :
  1/(2*a) + 1/b ≥ 4 := by sorry

-- Proposition D
theorem prop_d (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 4) :
  ∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 4 →
    x^2/(x+1) + y^2/(y+1) ≥ m ∧ 
    (∃ (u v : ℝ), u > 0 ∧ v > 0 ∧ u + v = 4 ∧ u^2/(u+1) + v^2/(v+1) = m) ∧
    m = 8/3 := by sorry

end NUMINAMATH_CALUDE_prop_c_prop_d_l2388_238898


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l2388_238849

theorem polar_to_rectangular (r θ : ℝ) (h : r = 7 ∧ θ = π/3) :
  (r * Real.cos θ, r * Real.sin θ) = (3.5, 7 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l2388_238849


namespace NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_45_l2388_238826

theorem one_and_two_thirds_of_x_is_45 : ∃ x : ℚ, (5/3) * x = 45 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_45_l2388_238826


namespace NUMINAMATH_CALUDE_f_even_eq_ten_times_f_odd_l2388_238804

/-- The function f(k) counts the number of k-digit integers (including those with leading zeros)
    whose digits can be permuted to form a number divisible by 11. -/
def f (k : ℕ) : ℕ := sorry

/-- For any positive integer m, f(2m) = 10 * f(2m-1) -/
theorem f_even_eq_ten_times_f_odd (m : ℕ+) : f (2 * m) = 10 * f (2 * m - 1) := by sorry

end NUMINAMATH_CALUDE_f_even_eq_ten_times_f_odd_l2388_238804


namespace NUMINAMATH_CALUDE_porter_earnings_l2388_238845

/-- Porter's daily rate in dollars -/
def daily_rate : ℚ := 8

/-- Number of regular working days per week -/
def regular_days : ℕ := 5

/-- Number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Overtime pay rate as a multiplier of regular rate -/
def overtime_rate : ℚ := 3/2

/-- Tax deduction rate -/
def tax_rate : ℚ := 1/10

/-- Insurance and benefits deduction rate -/
def insurance_rate : ℚ := 1/20

/-- Calculate Porter's monthly earnings after deductions and overtime -/
def monthly_earnings : ℚ :=
  let regular_weekly := daily_rate * regular_days
  let overtime_daily := daily_rate * overtime_rate
  let total_weekly := regular_weekly + overtime_daily
  let monthly_before_deductions := total_weekly * weeks_per_month
  let deductions := monthly_before_deductions * (tax_rate + insurance_rate)
  monthly_before_deductions - deductions

theorem porter_earnings :
  monthly_earnings = 1768/10 := by sorry

end NUMINAMATH_CALUDE_porter_earnings_l2388_238845


namespace NUMINAMATH_CALUDE_perfect_square_expression_l2388_238896

theorem perfect_square_expression : ∃ (x : ℝ), (12.86 * 12.86 + 12.86 * 0.28 + 0.14 * 0.14) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_expression_l2388_238896


namespace NUMINAMATH_CALUDE_water_average_l2388_238821

def water_problem (day1 day2 day3 : ℕ) : Prop :=
  day1 = 215 ∧
  day2 = day1 + 76 ∧
  day3 = day2 - 53 ∧
  (day1 + day2 + day3) / 3 = 248

theorem water_average : ∃ day1 day2 day3 : ℕ, water_problem day1 day2 day3 := by
  sorry

end NUMINAMATH_CALUDE_water_average_l2388_238821


namespace NUMINAMATH_CALUDE_line_segments_form_triangle_l2388_238848

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three line segments can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that the line segments 5, 6, and 10 can form a triangle. -/
theorem line_segments_form_triangle :
  can_form_triangle 5 6 10 := by
  sorry


end NUMINAMATH_CALUDE_line_segments_form_triangle_l2388_238848


namespace NUMINAMATH_CALUDE_radical_product_equality_l2388_238862

theorem radical_product_equality : Real.sqrt 81 * Real.sqrt 16 * (64 ^ (1/4 : ℝ)) = 72 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_radical_product_equality_l2388_238862


namespace NUMINAMATH_CALUDE_smallest_value_of_sum_of_cubes_l2388_238850

theorem smallest_value_of_sum_of_cubes (u v : ℂ) 
  (h1 : Complex.abs (u + v) = 2) 
  (h2 : Complex.abs (u^2 + v^2) = 8) : 
  Complex.abs (u^3 + v^3) = 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_value_of_sum_of_cubes_l2388_238850


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_sides_l2388_238810

/-- A quadrilateral inscribed in a circle with perpendicular diagonals -/
structure InscribedQuadrilateral where
  R : ℝ  -- Radius of the circumscribed circle
  d1 : ℝ  -- Distance of first diagonal from circle center
  d2 : ℝ  -- Distance of second diagonal from circle center

/-- The sides of the quadrilateral -/
def quadrilateralSides (q : InscribedQuadrilateral) : Set ℝ :=
  {x | ∃ (n : ℤ), x = 4 * (2 * Real.sqrt 13 + n) ∨ x = 4 * (8 + 2 * n * Real.sqrt 13)}

/-- Theorem stating the sides of the quadrilateral given specific conditions -/
theorem inscribed_quadrilateral_sides 
  (q : InscribedQuadrilateral) 
  (h1 : q.R = 17) 
  (h2 : q.d1 = 8) 
  (h3 : q.d2 = 9) : 
  ∀ s, s ∈ quadrilateralSides q ↔ 
    (s = 4 * (2 * Real.sqrt 13 - 1) ∨ 
     s = 4 * (2 * Real.sqrt 13 + 1) ∨ 
     s = 4 * (8 - 2 * Real.sqrt 13) ∨ 
     s = 4 * (8 + 2 * Real.sqrt 13)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_sides_l2388_238810


namespace NUMINAMATH_CALUDE_rectangle_area_l2388_238876

/-- Given a rectangle with diagonal length x and length three times its width, 
    the area of the rectangle is 3x^2/10 -/
theorem rectangle_area (x : ℝ) : 
  ∃ (w : ℝ), w > 0 ∧ x^2 = (3*w)^2 + w^2 → 3*w^2 = (3/10) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2388_238876


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_intersection_A_B_empty_intersection_A_B_equals_A_l2388_238841

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x - m < 0}

-- Theorem 1
theorem intersection_A_complement_B (m : ℝ) : 
  m = 3 → A ∩ (Set.univ \ B m) = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2
theorem intersection_A_B_empty (m : ℝ) : 
  A ∩ B m = ∅ ↔ m ≤ -2 := by sorry

-- Theorem 3
theorem intersection_A_B_equals_A (m : ℝ) : 
  A ∩ B m = A ↔ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_intersection_A_B_empty_intersection_A_B_equals_A_l2388_238841


namespace NUMINAMATH_CALUDE_three_tangents_range_l2388_238816

/-- The function f(x) = x^3 - 3x --/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 - 3

/-- Predicate to check if a point (x, y) is on the curve y = f(x) --/
def on_curve (x y : ℝ) : Prop := y = f x

/-- Predicate to check if a line through (1, m) is tangent to the curve at some point --/
def is_tangent (m t : ℝ) : Prop := 
  ∃ x : ℝ, on_curve x (f x) ∧ (m - f 1) = f' x * (1 - x)

/-- The main theorem --/
theorem three_tangents_range (m : ℝ) : 
  (∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    is_tangent m t1 ∧ is_tangent m t2 ∧ is_tangent m t3) → 
  m > -3 ∧ m < -2 :=
sorry

end NUMINAMATH_CALUDE_three_tangents_range_l2388_238816


namespace NUMINAMATH_CALUDE_prime_sum_equation_l2388_238856

theorem prime_sum_equation (a b n : ℕ) : 
  a < b ∧ 
  Nat.Prime a ∧ 
  Nat.Prime b ∧ 
  Odd n ∧ 
  a + b * n = 487 → 
  ((a = 2 ∧ b = 5 ∧ n = 97) ∨ (a = 2 ∧ b = 97 ∧ n = 5)) :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_equation_l2388_238856


namespace NUMINAMATH_CALUDE_max_sum_under_constraints_l2388_238875

theorem max_sum_under_constraints (a b : ℝ) :
  (4 * a + 3 * b ≤ 10) →
  (3 * a + 6 * b ≤ 12) →
  a + b ≤ 14 / 5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_under_constraints_l2388_238875


namespace NUMINAMATH_CALUDE_marble_probability_l2388_238868

theorem marble_probability (total : ℕ) (blue red : ℕ) (h1 : total = 50) (h2 : blue = 5) (h3 : red = 9) :
  (red + (total - blue - red)) / total = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l2388_238868


namespace NUMINAMATH_CALUDE_subcommittee_formation_ways_l2388_238831

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem subcommittee_formation_ways :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 8
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  (choose total_republicans subcommittee_republicans) *
  (choose total_democrats subcommittee_democrats) = 11760 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_ways_l2388_238831


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2388_238894

theorem arithmetic_expression_equality : 54 + (42 / 14) + (27 * 17) - 200 - (360 / 6) + 2^4 = 272 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2388_238894


namespace NUMINAMATH_CALUDE_two_designs_are_three_fifths_l2388_238855

/-- Represents a design with a shaded region --/
structure Design where
  shaded_fraction : Rat

/-- Checks if a given fraction is equal to 3/5 --/
def is_three_fifths (f : Rat) : Bool :=
  f = 3 / 5

/-- Counts the number of designs with shaded region equal to 3/5 --/
def count_three_fifths (designs : List Design) : Nat :=
  designs.filter (fun d => is_three_fifths d.shaded_fraction) |>.length

/-- The main theorem stating that exactly 2 out of 5 given designs have 3/5 shaded area --/
theorem two_designs_are_three_fifths :
  let designs : List Design := [
    ⟨3 / 8⟩,
    ⟨12 / 20⟩,
    ⟨2 / 3⟩,
    ⟨15 / 25⟩,
    ⟨4 / 8⟩
  ]
  count_three_fifths designs = 2 := by
  sorry


end NUMINAMATH_CALUDE_two_designs_are_three_fifths_l2388_238855


namespace NUMINAMATH_CALUDE_male_students_count_l2388_238879

theorem male_students_count (total_students : ℕ) (sample_size : ℕ) (female_in_sample : ℕ) :
  total_students = 800 →
  sample_size = 40 →
  female_in_sample = 11 →
  (sample_size - female_in_sample) * total_students = 580 * sample_size :=
by sorry

end NUMINAMATH_CALUDE_male_students_count_l2388_238879


namespace NUMINAMATH_CALUDE_tile_arrangements_l2388_238800

def brown_tiles : ℕ := 1
def purple_tiles : ℕ := 2
def red_tiles : ℕ := 2
def yellow_tiles : ℕ := 3

def total_tiles : ℕ := brown_tiles + purple_tiles + red_tiles + yellow_tiles

theorem tile_arrangements :
  (Nat.factorial total_tiles) / (Nat.factorial yellow_tiles * Nat.factorial purple_tiles * Nat.factorial red_tiles * Nat.factorial brown_tiles) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l2388_238800


namespace NUMINAMATH_CALUDE_probability_at_least_two_same_8sided_dice_l2388_238885

theorem probability_at_least_two_same_8sided_dice (n : ℕ) (s : ℕ) (p : ℚ) :
  n = 5 →
  s = 8 →
  p = 1628 / 2048 →
  p = 1 - (s * (s - 1) * (s - 2) * (s - 3) * (s - 4) : ℚ) / s^n :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_two_same_8sided_dice_l2388_238885


namespace NUMINAMATH_CALUDE_emilys_spending_l2388_238843

theorem emilys_spending (X : ℝ) 
  (friday : X ≥ 0)
  (saturday : 2 * X ≥ 0)
  (sunday : 3 * X ≥ 0)
  (total : X + 2 * X + 3 * X = 120) : X = 20 := by
  sorry

end NUMINAMATH_CALUDE_emilys_spending_l2388_238843


namespace NUMINAMATH_CALUDE_same_hair_count_l2388_238812

theorem same_hair_count (population : ℕ) (hair_count : Fin population → ℕ) 
  (h1 : population > 500001) 
  (h2 : ∀ p, hair_count p ≤ 500000) : 
  ∃ p1 p2, p1 ≠ p2 ∧ hair_count p1 = hair_count p2 := by
  sorry

end NUMINAMATH_CALUDE_same_hair_count_l2388_238812


namespace NUMINAMATH_CALUDE_sine_function_parameters_l2388_238809

theorem sine_function_parameters
  (y : ℝ → ℝ)
  (a b c : ℝ)
  (h1 : ∀ x, y x = a * Real.sin (b * x + c))
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : y (π / 6) = 3)
  (h5 : ∀ x, y (x + π) = y x) :
  a = 3 ∧ b = 2 ∧ c = π / 6 := by
sorry

end NUMINAMATH_CALUDE_sine_function_parameters_l2388_238809


namespace NUMINAMATH_CALUDE_triangle_circle_area_relation_l2388_238854

theorem triangle_circle_area_relation (a b c : ℝ) (A B C : ℝ) : 
  a = 13 ∧ b = 14 ∧ c = 15 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  C ≥ A ∧ C ≥ B →
  (a + b + c) / 2 = 21 →
  Real.sqrt (21 * (21 - a) * (21 - b) * (21 - c)) = 84 →
  A + B + 84 = C := by
sorry

end NUMINAMATH_CALUDE_triangle_circle_area_relation_l2388_238854


namespace NUMINAMATH_CALUDE_total_carriages_l2388_238892

/-- The number of carriages in each town -/
structure TownCarriages where
  euston : ℕ
  norfolk : ℕ
  norwich : ℕ
  flyingScotsman : ℕ

/-- The conditions given in the problem -/
def problemConditions (t : TownCarriages) : Prop :=
  t.euston = t.norfolk + 20 ∧
  t.norwich = 100 ∧
  t.flyingScotsman = t.norwich + 20 ∧
  t.euston = 130

/-- The theorem to prove -/
theorem total_carriages (t : TownCarriages) 
  (h : problemConditions t) : 
  t.euston + t.norfolk + t.norwich + t.flyingScotsman = 460 :=
by
  sorry

end NUMINAMATH_CALUDE_total_carriages_l2388_238892


namespace NUMINAMATH_CALUDE_alex_needs_three_packs_l2388_238867

/-- The number of burgers Alex plans to cook for each guest -/
def burgers_per_guest : ℕ := 3

/-- The number of friends Alex invited -/
def total_friends : ℕ := 10

/-- The number of friends who don't eat meat -/
def non_meat_eaters : ℕ := 1

/-- The number of friends who don't eat bread -/
def non_bread_eaters : ℕ := 1

/-- The number of buns in each pack -/
def buns_per_pack : ℕ := 8

/-- The function to calculate the number of packs of buns Alex needs to buy -/
def packs_of_buns_needed : ℕ :=
  let total_guests := total_friends - non_meat_eaters
  let total_burgers := burgers_per_guest * total_guests
  let burgers_needing_buns := total_burgers - (burgers_per_guest * non_bread_eaters)
  (burgers_needing_buns + buns_per_pack - 1) / buns_per_pack

/-- Theorem stating that Alex needs to buy 3 packs of buns -/
theorem alex_needs_three_packs : packs_of_buns_needed = 3 := by
  sorry

end NUMINAMATH_CALUDE_alex_needs_three_packs_l2388_238867


namespace NUMINAMATH_CALUDE_correct_average_l2388_238838

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) 
  (misread1 misread2 misread3 : ℚ) 
  (correct1 correct2 correct3 : ℚ) :
  n = 15 ∧ 
  incorrect_avg = 62 ∧
  misread1 = 30 ∧ correct1 = 90 ∧
  misread2 = 60 ∧ correct2 = 120 ∧
  misread3 = 25 ∧ correct3 = 75 →
  (n : ℚ) * incorrect_avg + (correct1 - misread1) + (correct2 - misread2) + (correct3 - misread3) = n * (73 + 1/3) :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l2388_238838


namespace NUMINAMATH_CALUDE_fraction_value_at_2017_l2388_238834

theorem fraction_value_at_2017 :
  let x : ℤ := 2017
  (x^2 + 6*x + 9) / (x + 3) = 2020 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_at_2017_l2388_238834


namespace NUMINAMATH_CALUDE_unique_intersection_point_l2388_238878

-- Define the function g
def g (x : ℝ) : ℝ := x^3 + 3*x^2 + 9*x + 15

-- State the theorem
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (-3, -3) := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l2388_238878


namespace NUMINAMATH_CALUDE_seats_needed_l2388_238820

theorem seats_needed (total_children : ℕ) (children_per_seat : ℕ) 
  (h1 : total_children = 58) 
  (h2 : children_per_seat = 2) : 
  total_children / children_per_seat = 29 := by
sorry

end NUMINAMATH_CALUDE_seats_needed_l2388_238820


namespace NUMINAMATH_CALUDE_product_mod_25_l2388_238871

theorem product_mod_25 : ∃ n : ℕ, 0 ≤ n ∧ n < 25 ∧ (123 * 456 * 789) % 25 = n ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_25_l2388_238871


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2388_238803

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * crossing_time

/-- Proof that a train with speed 180 km/h crossing a pole in 1.9998400127989762 seconds is approximately 99.992 meters long -/
theorem train_length_proof : 
  ∃ (ε : ℝ), ε > 0 ∧ |train_length 180 1.9998400127989762 - 99.992| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l2388_238803


namespace NUMINAMATH_CALUDE_jeans_average_speed_l2388_238815

/-- Proves that Jean's average speed is 18/11 mph given the problem conditions --/
theorem jeans_average_speed :
  let trail_length : ℝ := 12
  let uphill_length : ℝ := 4
  let chantal_flat_speed : ℝ := 3
  let chantal_uphill_speed : ℝ := 1.5
  let chantal_downhill_speed : ℝ := 2.25
  let jean_delay : ℝ := 2

  let chantal_flat_time : ℝ := (trail_length - uphill_length) / chantal_flat_speed
  let chantal_uphill_time : ℝ := uphill_length / chantal_uphill_speed
  let chantal_downhill_time : ℝ := uphill_length / chantal_downhill_speed
  let chantal_total_time : ℝ := chantal_flat_time + chantal_uphill_time + chantal_downhill_time

  let jean_travel_time : ℝ := chantal_total_time - jean_delay
  let jean_travel_distance : ℝ := uphill_length

  jean_travel_distance / jean_travel_time = 18 / 11 := by sorry

end NUMINAMATH_CALUDE_jeans_average_speed_l2388_238815


namespace NUMINAMATH_CALUDE_gear_speed_proportion_l2388_238861

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear
  mesh_AB : A.teeth * A.speed = B.teeth * B.speed
  mesh_BC : B.teeth * B.speed = C.teeth * C.speed
  mesh_CD : C.teeth * C.speed = D.teeth * D.speed

/-- The theorem stating the proportion of angular speeds for the gear system -/
theorem gear_speed_proportion (sys : GearSystem) :
  ∃ (k : ℝ), k ≠ 0 ∧
    (sys.A.speed = k * (sys.B.teeth * sys.C.teeth * sys.D.teeth)) ∧
    (sys.B.speed = k * (sys.A.teeth * sys.C.teeth * sys.D.teeth)) ∧
    (sys.C.speed = k * (sys.A.teeth * sys.B.teeth * sys.D.teeth)) ∧
    (sys.D.speed = k * (sys.A.teeth * sys.B.teeth * sys.C.teeth)) :=
by
  sorry

end NUMINAMATH_CALUDE_gear_speed_proportion_l2388_238861


namespace NUMINAMATH_CALUDE_order_of_7_wrt_g_l2388_238852

def g (x : ℕ) : ℕ := x^2 % 13

def g_iter (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => g (g_iter n x)

theorem order_of_7_wrt_g :
  (∀ k < 12, g_iter k 7 ≠ 7) ∧ g_iter 12 7 = 7 :=
sorry

end NUMINAMATH_CALUDE_order_of_7_wrt_g_l2388_238852


namespace NUMINAMATH_CALUDE_line_segment_param_sum_squares_l2388_238836

/-- 
Given a line segment from (-3,5) to (4,15) parameterized by x = at + b and y = ct + d,
where -1 ≤ t ≤ 2 and t = -1 corresponds to (-3,5), prove that a² + b² + c² + d² = 790/9
-/
theorem line_segment_param_sum_squares (a b c d : ℚ) : 
  (∀ t, -1 ≤ t → t ≤ 2 → ∃ x y, x = a * t + b ∧ y = c * t + d) →
  a * (-1) + b = -3 →
  c * (-1) + d = 5 →
  a * 2 + b = 4 →
  c * 2 + d = 15 →
  a^2 + b^2 + c^2 + d^2 = 790/9 := by
sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_squares_l2388_238836


namespace NUMINAMATH_CALUDE_arrangement_count_is_240_l2388_238837

/-- The number of ways to arrange 8 distinct objects in a row,
    where the two smallest objects must be at the ends and
    the largest object must be in the middle. -/
def arrangement_count : ℕ := 240

/-- Theorem stating that the number of arrangements is 240 -/
theorem arrangement_count_is_240 : arrangement_count = 240 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_240_l2388_238837


namespace NUMINAMATH_CALUDE_smallest_valid_number_l2388_238811

def is_valid (n : ℕ) : Prop :=
  ∀ k : ℕ, 2 ≤ k → k ≤ 12 → n % k = k - 1

theorem smallest_valid_number : 
  (is_valid 27719) ∧ (∀ m : ℕ, m < 27719 → ¬(is_valid m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l2388_238811


namespace NUMINAMATH_CALUDE_cos_negative_245_deg_l2388_238857

theorem cos_negative_245_deg (a : ℝ) (h : Real.cos (25 * π / 180) = a) :
  Real.cos (-245 * π / 180) = -Real.sqrt (1 - a^2) := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_245_deg_l2388_238857


namespace NUMINAMATH_CALUDE_gumball_draw_theorem_l2388_238872

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)

/-- Represents the minimum number of gumballs to draw to guarantee 4 of the same color -/
def minDrawToGuaranteeFour (machine : GumballMachine) : ℕ :=
  sorry

/-- The theorem to be proved -/
theorem gumball_draw_theorem (machine : GumballMachine) 
  (h1 : machine.red = 9)
  (h2 : machine.white = 7)
  (h3 : machine.blue = 12) :
  minDrawToGuaranteeFour machine = 12 :=
sorry

end NUMINAMATH_CALUDE_gumball_draw_theorem_l2388_238872


namespace NUMINAMATH_CALUDE_complement_of_union_is_empty_l2388_238818

universe u

def U : Set Char := {'a', 'b', 'c', 'd', 'e'}
def N : Set Char := {'b', 'd', 'e'}
def M : Set Char := {'a', 'c', 'd'}

theorem complement_of_union_is_empty :
  (M ∪ N)ᶜ = ∅ :=
sorry

end NUMINAMATH_CALUDE_complement_of_union_is_empty_l2388_238818


namespace NUMINAMATH_CALUDE_equation_solutions_l2388_238813

theorem equation_solutions :
  let f (x : ℝ) := (8*x^2 - 20*x + 3)/(2*x - 1) + 7*x
  ∀ x : ℝ, f x = 9*x - 3 ↔ x = 1/2 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2388_238813


namespace NUMINAMATH_CALUDE_derivative_at_one_l2388_238889

theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 2 * (deriv f 1) * x) :
  deriv f 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2388_238889


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2388_238823

universe u

def U : Set ℕ := {0, 1, 3, 4, 5, 6, 8}
def A : Set ℕ := {1, 4, 5, 8}
def B : Set ℕ := {2, 6}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2388_238823


namespace NUMINAMATH_CALUDE_geometric_series_proof_l2388_238801

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_proof :
  let a : ℚ := 1/2
  let r : ℚ := -1/2
  let n : ℕ := 6
  geometric_series_sum a r n = 21/64 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_series_proof_l2388_238801


namespace NUMINAMATH_CALUDE_track_length_is_900_l2388_238895

/-- The length of a circular track where two runners meet again -/
def track_length (v1 v2 t : ℝ) : ℝ :=
  (v1 - v2) * t

/-- Theorem stating the length of the track is 900 meters -/
theorem track_length_is_900 :
  let v1 : ℝ := 30  -- Speed of Bruce in m/s
  let v2 : ℝ := 20  -- Speed of Bhishma in m/s
  let t : ℝ := 90   -- Time in seconds
  track_length v1 v2 t = 900 := by
  sorry

#eval track_length 30 20 90  -- Should output 900

end NUMINAMATH_CALUDE_track_length_is_900_l2388_238895


namespace NUMINAMATH_CALUDE_max_sum_with_divisibility_conditions_l2388_238874

theorem max_sum_with_divisibility_conditions (a b c : ℕ) : 
  a > 2022 → b > 2022 → c > 2022 →
  (c - 2022) ∣ (a + b) →
  (b - 2022) ∣ (a + c) →
  (a - 2022) ∣ (b + c) →
  a + b + c ≤ 2022 * 85 := by
sorry

end NUMINAMATH_CALUDE_max_sum_with_divisibility_conditions_l2388_238874


namespace NUMINAMATH_CALUDE_line_equation_through_two_points_l2388_238817

/-- Given two points A and B on the line 2x + 3y = 4, 
    prove that this is the equation of the line passing through these points. -/
theorem line_equation_through_two_points 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 2 * x₁ + 3 * y₁ = 4) 
  (h₂ : 2 * x₂ + 3 * y₂ = 4) : 
  ∀ (x y : ℝ), (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) → 2 * x + 3 * y = 4 := by
sorry

end NUMINAMATH_CALUDE_line_equation_through_two_points_l2388_238817


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2388_238877

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 7/12) 
  (h2 : x - y = 1/12) : 
  x^2 - y^2 = 7/144 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2388_238877


namespace NUMINAMATH_CALUDE_total_students_on_trip_l2388_238814

/-- The number of students who went on a trip to the zoo -/
def students_on_trip (num_buses : ℕ) (students_per_bus : ℕ) (students_in_cars : ℕ) : ℕ :=
  num_buses * students_per_bus + students_in_cars

/-- Theorem stating the total number of students on the trip -/
theorem total_students_on_trip :
  students_on_trip 7 56 4 = 396 := by
  sorry

end NUMINAMATH_CALUDE_total_students_on_trip_l2388_238814


namespace NUMINAMATH_CALUDE_xiao_ming_walk_relation_l2388_238824

/-- Represents the relationship between remaining distance and time walked
    for a person walking towards a destination. -/
def distance_time_relation (total_distance : ℝ) (speed : ℝ) (x : ℝ) : ℝ :=
  total_distance - speed * x

/-- Theorem stating the relationship between remaining distance and time walked
    for Xiao Ming's walk to school. -/
theorem xiao_ming_walk_relation :
  ∀ x y : ℝ, y = distance_time_relation 1200 70 x ↔ y = -70 * x + 1200 :=
by sorry

end NUMINAMATH_CALUDE_xiao_ming_walk_relation_l2388_238824


namespace NUMINAMATH_CALUDE_square_area_l2388_238863

-- Define the coordinates of the vertex and diagonal intersection
def vertex : ℝ × ℝ := (-6, -4)
def diagonal_intersection : ℝ × ℝ := (3, 2)

-- Define the theorem
theorem square_area (v : ℝ × ℝ) (d : ℝ × ℝ) (h1 : v = vertex) (h2 : d = diagonal_intersection) :
  let diagonal_length := Real.sqrt ((d.1 - v.1)^2 + (d.2 - v.2)^2)
  (diagonal_length^2) / 2 = 58.5 := by sorry

end NUMINAMATH_CALUDE_square_area_l2388_238863


namespace NUMINAMATH_CALUDE_percentage_difference_l2388_238805

theorem percentage_difference (x y : ℝ) (h : x = y * (1 - 0.4)) :
  (y - x) / x = 0.4 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l2388_238805


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l2388_238802

/-- 
If in a triangle ABC, where a, b, c are the lengths of sides opposite to angles A, B, C respectively, 
and a * cos(B) = b * cos(A), then the triangle ABC is isosceles.
-/
theorem isosceles_triangle_condition (A B C a b c : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_condition : a * Real.cos B = b * Real.cos A) :
  a = b ∨ b = c ∨ a = c := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l2388_238802


namespace NUMINAMATH_CALUDE_mona_monday_miles_l2388_238853

/-- Represents Mona's biking schedule for a week --/
structure BikingWeek where
  total_miles : ℝ
  monday_miles : ℝ
  wednesday_miles : ℝ
  saturday_miles : ℝ
  steep_trail_speed : ℝ
  flat_road_speed : ℝ
  saturday_speed_reduction : ℝ

/-- Theorem stating that Mona biked 6 miles on Monday --/
theorem mona_monday_miles (week : BikingWeek) 
  (h1 : week.total_miles = 30)
  (h2 : week.wednesday_miles = 12)
  (h3 : week.saturday_miles = 2 * week.monday_miles)
  (h4 : week.steep_trail_speed = 6)
  (h5 : week.flat_road_speed = 15)
  (h6 : week.saturday_speed_reduction = 0.2)
  (h7 : week.total_miles = week.monday_miles + week.wednesday_miles + week.saturday_miles) :
  week.monday_miles = 6 := by
  sorry

#check mona_monday_miles

end NUMINAMATH_CALUDE_mona_monday_miles_l2388_238853


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_focal_length_specific_hyperbola_l2388_238832

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is 2c, where c² = a² + b² -/
theorem hyperbola_focal_length (a b c : ℝ) (h : a > 0) (h' : b > 0) :
  (a^2 = 10) → (b^2 = 2) → (c^2 = a^2 + b^2) →
  (2 * c = 4 * Real.sqrt 3) := by
  sorry

/-- The focal length of the hyperbola x²/10 - y²/2 = 1 is 4√3 -/
theorem focal_length_specific_hyperbola :
  ∃ (a b c : ℝ), (a > 0) ∧ (b > 0) ∧
  (a^2 = 10) ∧ (b^2 = 2) ∧ (c^2 = a^2 + b^2) ∧
  (2 * c = 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_focal_length_specific_hyperbola_l2388_238832
