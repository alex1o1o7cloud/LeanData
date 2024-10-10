import Mathlib

namespace triangleAreaSum_form_l1247_124751

/-- The sum of areas of all triangles with vertices on a 2 by 3 by 4 rectangular box -/
def triangleAreaSum : ℝ := sorry

/-- The number of vertices of a rectangular box -/
def vertexCount : ℕ := 8

/-- The dimensions of the rectangular box -/
def boxDimensions : Fin 3 → ℝ
| 0 => 2
| 1 => 3
| 2 => 4
| _ => 0  -- This line is just to satisfy Lean's totality requirement

/-- Theorem stating the form of the sum of triangle areas -/
theorem triangleAreaSum_form :
  ∃ (k p : ℝ), triangleAreaSum = 168 + k * Real.sqrt p :=
sorry

end triangleAreaSum_form_l1247_124751


namespace tangent_line_intersection_at_minus_one_range_of_a_l1247_124787

/-- The function f(x) = x^3 - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The function g(x) = x^2 + a, where a is a parameter -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The derivative of g(x) -/
def g_derivative (x : ℝ) : ℝ := 2 * x

/-- Theorem stating that when x₁ = -1, a = 3 -/
theorem tangent_line_intersection_at_minus_one (a : ℝ) :
  (∃ x₂ : ℝ, f_derivative (-1) = g_derivative x₂ ∧ 
    f (-1) - f_derivative (-1) * (-1) = g a x₂ - g_derivative x₂ * x₂) →
  a = 3 :=
sorry

/-- Theorem stating that a ≥ -1 -/
theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, f_derivative x₁ = g_derivative x₂ ∧ 
    f x₁ - f_derivative x₁ * x₁ = g a x₂ - g_derivative x₂ * x₂) →
  a ≥ -1 :=
sorry

end tangent_line_intersection_at_minus_one_range_of_a_l1247_124787


namespace hyperbola_sum_l1247_124722

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = -3 →
  k = 1 →
  c = Real.sqrt 50 →
  a = 4 →
  b^2 = c^2 - a^2 →
  h + k + a + b = 2 + Real.sqrt 34 := by
  sorry

end hyperbola_sum_l1247_124722


namespace arithmetic_sequence_sum_l1247_124799

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 5 + a 6 + a 7 = 15) : 
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
sorry

end arithmetic_sequence_sum_l1247_124799


namespace isosceles_iff_equal_angle_bisectors_l1247_124776

/-- Given a triangle with sides a, b, c, and angle bisectors l_α and l_β, 
    prove that the triangle is isosceles (a = b) if and only if l_α = l_β -/
theorem isosceles_iff_equal_angle_bisectors 
  (a b c : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c)
  (l_α : ℝ := (1 / (b + c)) * Real.sqrt (b * c * ((b + c)^2 - a^2)))
  (l_β : ℝ := (1 / (c + a)) * Real.sqrt (c * a * ((c + a)^2 - b^2))) :
  a = b ↔ l_α = l_β := by
  sorry

end isosceles_iff_equal_angle_bisectors_l1247_124776


namespace intersection_of_perpendicular_lines_l1247_124704

-- Define the lines
def line1 (x y c : ℝ) : Prop := 3 * x - 4 * y = c
def line2 (x y c d : ℝ) : Prop := 8 * x + d * y = -c

-- Define perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem intersection_of_perpendicular_lines (c d : ℝ) :
  -- Lines are perpendicular
  perpendicular (3/4) (-8/d) →
  -- Lines intersect at (2, -3)
  line1 2 (-3) c →
  line2 2 (-3) c d →
  -- Then c = 18
  c = 18 := by sorry

end intersection_of_perpendicular_lines_l1247_124704


namespace common_chord_equation_l1247_124736

/-- The equation of the line containing the common chord of two circles -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 - 4 = 0) →
  (x^2 + y^2 - 4*x + 4*y - 12 = 0) →
  (x - y + 2 = 0) :=
by sorry

end common_chord_equation_l1247_124736


namespace object_with_22_opposite_directions_is_clock_l1247_124746

/-- An object with hands that can show opposite directions -/
structure ObjectWithHands :=
  (oppositeDirectionsPerDay : ℕ)

/-- Definition of a clock based on its behavior -/
def isClock (obj : ObjectWithHands) : Prop :=
  obj.oppositeDirectionsPerDay = 22

/-- Theorem stating that an object with hands showing opposite directions 22 times a day is a clock -/
theorem object_with_22_opposite_directions_is_clock (obj : ObjectWithHands) :
  obj.oppositeDirectionsPerDay = 22 → isClock obj :=
by
  sorry

#check object_with_22_opposite_directions_is_clock

end object_with_22_opposite_directions_is_clock_l1247_124746


namespace change_received_l1247_124762

/-- Represents the cost of a basic calculator in dollars -/
def basic_cost : ℕ := 8

/-- Represents the total amount of money the teacher had in dollars -/
def total_money : ℕ := 100

/-- Calculates the cost of a scientific calculator -/
def scientific_cost : ℕ := 2 * basic_cost

/-- Calculates the cost of a graphing calculator -/
def graphing_cost : ℕ := 3 * scientific_cost

/-- Calculates the total cost of buying one of each calculator -/
def total_cost : ℕ := basic_cost + scientific_cost + graphing_cost

/-- Theorem stating that the change received is $28 -/
theorem change_received : total_money - total_cost = 28 := by
  sorry

end change_received_l1247_124762


namespace special_function_18_48_l1247_124774

/-- A function satisfying the given properties -/
def special_function (f : ℕ+ → ℕ+ → ℕ+) : Prop :=
  (∀ x : ℕ+, f x x = x) ∧
  (∀ x y : ℕ+, f x y = f y x) ∧
  (∀ x y : ℕ+, (x + y) * (f x y) = x * (f x (x + y)))

/-- The main theorem -/
theorem special_function_18_48 (f : ℕ+ → ℕ+ → ℕ+) (h : special_function f) :
  f 18 48 = 48 := by
  sorry

end special_function_18_48_l1247_124774


namespace base_b_divisibility_l1247_124754

theorem base_b_divisibility (b : ℤ) : b = 7 ↔ ¬(5 ∣ (b^2 * (3*b - 2))) ∧ 
  (b = 6 ∨ b = 8 ∨ b = 9 ∨ b = 10 → 5 ∣ (b^2 * (3*b - 2))) := by
  sorry

end base_b_divisibility_l1247_124754


namespace polynomial_expansion_l1247_124714

theorem polynomial_expansion (z : ℝ) :
  (3 * z^2 + 4 * z - 5) * (4 * z^3 - 3 * z^2 + 2) =
  12 * z^5 + 7 * z^4 - 26 * z^3 + 21 * z^2 + 8 * z - 10 := by
  sorry

end polynomial_expansion_l1247_124714


namespace cindy_lisa_marble_difference_l1247_124794

theorem cindy_lisa_marble_difference :
  ∀ (lisa_initial : ℕ),
  let cindy_initial : ℕ := 20
  let cindy_after : ℕ := cindy_initial - 12
  let lisa_after : ℕ := lisa_initial + 12
  lisa_after = cindy_after + 19 →
  cindy_initial - lisa_initial = 5 :=
by
  sorry

end cindy_lisa_marble_difference_l1247_124794


namespace simplify_expression_l1247_124724

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 - 2*b + 1) - 2*b^2 = 9*b^3 - 8*b^2 + 3*b := by
  sorry

end simplify_expression_l1247_124724


namespace factorization_cubic_minus_nine_xy_squared_l1247_124721

theorem factorization_cubic_minus_nine_xy_squared (x y : ℝ) :
  x^3 - 9*x*y^2 = x*(x+3*y)*(x-3*y) := by
  sorry

end factorization_cubic_minus_nine_xy_squared_l1247_124721


namespace box_volume_formula_l1247_124735

/-- The volume of a box formed by cutting rectangles from a sheet and folding up the flaps -/
def box_volume (x y : ℝ) : ℝ :=
  (16 - 2*x) * (12 - 2*y) * y

/-- The original sheet dimensions -/
def sheet_length : ℝ := 16
def sheet_width : ℝ := 12

theorem box_volume_formula (x y : ℝ) :
  box_volume x y = 4*x*y^2 - 24*x*y + 192*y - 32*y^2 :=
by sorry

end box_volume_formula_l1247_124735


namespace number_rewriting_l1247_124789

theorem number_rewriting :
  (29800000 = 2980 * 10000) ∧ (14000000000 = 140 * 100000000) := by
  sorry

end number_rewriting_l1247_124789


namespace coins_per_stack_l1247_124755

theorem coins_per_stack (total_coins : ℕ) (num_stacks : ℕ) (coins_per_stack : ℕ) : 
  total_coins = 15 → num_stacks = 5 → total_coins = num_stacks * coins_per_stack → coins_per_stack = 3 := by
  sorry

end coins_per_stack_l1247_124755


namespace unique_solution_l1247_124798

theorem unique_solution : ∃! (x : ℝ), x > 0 ∧ (180 / x) + ((5 * 12) / x) + 80 = 81 := by
  sorry

end unique_solution_l1247_124798


namespace percentage_married_employees_l1247_124768

theorem percentage_married_employees (total : ℝ) (total_pos : 0 < total) : 
  let women_ratio : ℝ := 0.76
  let men_ratio : ℝ := 1 - women_ratio
  let married_women_ratio : ℝ := 0.6842
  let single_men_ratio : ℝ := 2/3
  let married_men_ratio : ℝ := 1 - single_men_ratio
  let married_ratio : ℝ := women_ratio * married_women_ratio + men_ratio * married_men_ratio
  married_ratio = 0.600392 :=
sorry

end percentage_married_employees_l1247_124768


namespace age_difference_l1247_124793

theorem age_difference (a b : ℕ) : 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 →  -- Ensuring a and b are single digits
  (10 * a + b) + 5 = 3 * ((10 * b + a) + 5) → -- Condition after 5 years
  (10 * a + b) - (10 * b + a) = 63 := by
sorry

end age_difference_l1247_124793


namespace line_perp_plane_if_perp_two_intersecting_lines_planes_perp_if_line_in_one_perp_other_l1247_124745

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- Theorem 1
theorem line_perp_plane_if_perp_two_intersecting_lines 
  (l : Line) (α : Plane) (m n : Line) :
  contained_in m α → contained_in n α → 
  intersect m n → 
  perpendicular l m → perpendicular l n → 
  perpendicular_line_plane l α :=
sorry

-- Theorem 2
theorem planes_perp_if_line_in_one_perp_other 
  (l : Line) (α β : Plane) :
  contained_in l β → perpendicular_line_plane l α → 
  perpendicular_plane_plane α β :=
sorry

end line_perp_plane_if_perp_two_intersecting_lines_planes_perp_if_line_in_one_perp_other_l1247_124745


namespace solve_for_a_l1247_124775

theorem solve_for_a (a b : ℚ) (h1 : b / a = 4) (h2 : b = 18 - 6 * a) : a = 9 / 5 := by
  sorry

end solve_for_a_l1247_124775


namespace bobs_hair_length_l1247_124723

/-- Calculates the final hair length after a given time period. -/
def final_hair_length (initial_length : ℝ) (growth_rate : ℝ) (time : ℝ) : ℝ :=
  initial_length + growth_rate * 12 * time

/-- Proves that Bob's hair length after 5 years is 36 inches. -/
theorem bobs_hair_length :
  let initial_length : ℝ := 6
  let growth_rate : ℝ := 0.5
  let time : ℝ := 5
  final_hair_length initial_length growth_rate time = 36 := by
  sorry

end bobs_hair_length_l1247_124723


namespace completing_square_equivalence_l1247_124708

theorem completing_square_equivalence (x : ℝ) : x^2 + 4*x - 3 = 0 ↔ (x + 2)^2 = 7 := by
  sorry

end completing_square_equivalence_l1247_124708


namespace complex_subtraction_l1247_124769

theorem complex_subtraction : (4 : ℂ) - 3*I - ((2 : ℂ) + 5*I) = (2 : ℂ) - 8*I := by
  sorry

end complex_subtraction_l1247_124769


namespace min_value_reciprocal_sum_l1247_124730

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 1) :
  (1/x + 1/(2*y) ≥ 4) ∧ ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + 2*y' = 1 ∧ 1/x' + 1/(2*y') = 4 :=
by sorry

end min_value_reciprocal_sum_l1247_124730


namespace hyperbola_asymptote_inclination_l1247_124712

/-- Given a hyperbola mx^2 - y^2 = m where m > 0, if one of its asymptotes has an angle of inclination
    that is twice the angle of inclination of the line x - √3y = 0, then m = 3. -/
theorem hyperbola_asymptote_inclination (m : ℝ) (h1 : m > 0) : 
  (∃ θ : ℝ, θ = 2 * Real.arctan (1 / Real.sqrt 3) ∧ 
             Real.tan θ = Real.sqrt m) → m = 3 := by
  sorry

end hyperbola_asymptote_inclination_l1247_124712


namespace diesel_fuel_usage_l1247_124748

/-- Given weekly spending on diesel fuel and cost per gallon, calculates the amount of diesel fuel used in two weeks -/
theorem diesel_fuel_usage
  (weekly_spending : ℝ)
  (cost_per_gallon : ℝ)
  (h1 : weekly_spending = 36)
  (h2 : cost_per_gallon = 3)
  : weekly_spending / cost_per_gallon * 2 = 24 := by
  sorry

end diesel_fuel_usage_l1247_124748


namespace retail_price_calculation_l1247_124703

theorem retail_price_calculation (W S P R : ℚ) : 
  W = 99 → 
  S = 0.9 * P → 
  R = 0.2 * W → 
  S = W + R → 
  P = 132 := by
sorry

end retail_price_calculation_l1247_124703


namespace toms_age_ratio_l1247_124734

theorem toms_age_ratio (T N : ℝ) : 
  (T > 0) →  -- Tom's age is positive
  (N > 0) →  -- N is positive (years in the past)
  (T - N > 0) →  -- Tom's age N years ago was positive
  (T - 4*N > 0) →  -- The sum of children's ages N years ago was positive
  (T - N = 3 * (T - 4*N)) →  -- Condition about ages N years ago
  T / N = 11 / 2 := by
sorry

end toms_age_ratio_l1247_124734


namespace compound_interest_calculation_l1247_124788

/-- Calculates the amount after two years of compound interest with different rates for each year. -/
def amountAfterTwoYears (initialAmount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amountAfterFirstYear := initialAmount * (1 + rate1)
  amountAfterFirstYear * (1 + rate2)

/-- Theorem stating that given the initial amount and interest rates, the final amount after two years is as calculated. -/
theorem compound_interest_calculation (initialAmount : ℝ) (rate1 : ℝ) (rate2 : ℝ) 
  (h1 : initialAmount = 9828) 
  (h2 : rate1 = 0.04) 
  (h3 : rate2 = 0.05) :
  amountAfterTwoYears initialAmount rate1 rate2 = 10732.176 := by
  sorry

#eval amountAfterTwoYears 9828 0.04 0.05

end compound_interest_calculation_l1247_124788


namespace max_score_in_range_score_2079_is_in_range_l1247_124759

def score (x : ℕ) : ℕ :=
  (if x % 3 = 0 then 3 else 0) +
  (if x % 5 = 0 then 5 else 0) +
  (if x % 7 = 0 then 7 else 0) +
  (if x % 9 = 0 then 9 else 0) +
  (if x % 11 = 0 then 11 else 0)

theorem max_score_in_range :
  ∀ x : ℕ, 2017 ≤ x ∧ x ≤ 2117 → score x ≤ score 2079 :=
by sorry

theorem score_2079 : score 2079 = 30 :=
by sorry

theorem is_in_range : 2017 ≤ 2079 ∧ 2079 ≤ 2117 :=
by sorry

end max_score_in_range_score_2079_is_in_range_l1247_124759


namespace special_divisibility_property_l1247_124715

theorem special_divisibility_property (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) :
  (∀ n : ℕ, n > 0 → a^n - n^a ≠ 0 → (a^n - n^a) ∣ (b^n - n^b)) ↔
  ((a = 2 ∧ b = 2) ∨ (a = 2 ∧ b = 4) ∨ (a = b ∧ a > 1)) :=
by sorry

end special_divisibility_property_l1247_124715


namespace females_only_in_orchestra_l1247_124737

/-- Represents the membership data for the band and orchestra --/
structure MusicGroups where
  band_females : ℕ
  band_males : ℕ
  orchestra_females : ℕ
  orchestra_males : ℕ
  both_females : ℕ
  total_members : ℕ

/-- The theorem stating the number of females in the orchestra who are not in the band --/
theorem females_only_in_orchestra (mg : MusicGroups)
  (h1 : mg.band_females = 120)
  (h2 : mg.band_males = 100)
  (h3 : mg.orchestra_females = 100)
  (h4 : mg.orchestra_males = 120)
  (h5 : mg.both_females = 80)
  (h6 : mg.total_members = 260) :
  mg.orchestra_females - mg.both_females = 20 := by
  sorry

#check females_only_in_orchestra

end females_only_in_orchestra_l1247_124737


namespace ellipse_properties_l1247_124749

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0

/-- The ellipse satisfies the given conditions -/
def EllipseConditions (e : Ellipse) : Prop :=
  (e.a ^ 2 - e.b ^ 2) / e.a ^ 2 = 3 / 4 ∧  -- eccentricity is √3/2
  e.a - (e.a ^ 2 - e.b ^ 2).sqrt = 2       -- distance from upper vertex to focus is 2

/-- Theorem stating the properties of the ellipse -/
theorem ellipse_properties (e : Ellipse) (h : EllipseConditions e) :
  e.a = 2 ∧ e.b = 1 ∧
  (∀ k : ℝ, k = 1 → 
    (∃ S : ℝ → ℝ, (∀ m : ℝ, S m ≤ 1) ∧ (∃ m : ℝ, S m = 1))) ∧
  (∀ k : ℝ, (∀ m : ℝ, ∃ C : ℝ, 
    (∀ x : ℝ, (x - m)^2 + (k * (x - m))^2 + 
      ((4 * (k^2 * m^2 - 1)) / (1 + 4 * k^2) - x)^2 + 
      (k * ((4 * (k^2 * m^2 - 1)) / (1 + 4 * k^2) - x))^2 = C)) → 
    k = 1/2 ∨ k = -1/2) := by
  sorry

end ellipse_properties_l1247_124749


namespace max_value_of_f_l1247_124765

-- Define the parabola function
def f (x : ℝ) : ℝ := -(x + 1)^2 + 3

-- Theorem: The maximum value of f is 3
theorem max_value_of_f : ∀ x : ℝ, f x ≤ 3 := by
  sorry

end max_value_of_f_l1247_124765


namespace sum_of_composite_function_evaluations_l1247_124713

def p (x : ℝ) : ℝ := 2 * |x| - 4

def q (x : ℝ) : ℝ := -|x|

def evaluation_points : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_composite_function_evaluations :
  (evaluation_points.map (λ x => q (p x))).sum = -20 := by
  sorry

end sum_of_composite_function_evaluations_l1247_124713


namespace inequality_system_solution_l1247_124718

theorem inequality_system_solution :
  let S := {x : ℝ | 2 * x - 2 > 0 ∧ 3 * (x - 1) - 7 < -2 * x}
  S = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end inequality_system_solution_l1247_124718


namespace sheela_deposit_l1247_124729

/-- Calculates the deposit amount given a monthly income and deposit percentage -/
def deposit_amount (monthly_income : ℕ) (deposit_percentage : ℚ) : ℚ :=
  (deposit_percentage * monthly_income : ℚ)

theorem sheela_deposit :
  deposit_amount 10000 (25 / 100) = 2500 := by
  sorry

end sheela_deposit_l1247_124729


namespace tan_one_condition_l1247_124709

theorem tan_one_condition (x : Real) : 
  (∃ k : Int, x = (k * Real.pi) / 4) ∧ 
  (∃ x : Real, (∃ k : Int, x = (k * Real.pi) / 4) ∧ Real.tan x ≠ 1) ∧
  (∀ x : Real, Real.tan x = 1 → ∃ k : Int, x = ((4 * k + 1) * Real.pi) / 4) :=
by sorry

end tan_one_condition_l1247_124709


namespace min_m_plus_n_l1247_124758

theorem min_m_plus_n (m n : ℕ+) (h : 75 * m = n^3) : 
  ∀ (m' n' : ℕ+), 75 * m' = n'^3 → m + n ≤ m' + n' :=
by sorry

end min_m_plus_n_l1247_124758


namespace parallel_vectors_imply_k_equals_one_l1247_124792

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is parallel to c, then k = 1 -/
theorem parallel_vectors_imply_k_equals_one (a b c : ℝ × ℝ) (k : ℝ) :
  a = (Real.sqrt 3, 1) →
  b = (0, 1) →
  c = (k, Real.sqrt 3) →
  ∃ (t : ℝ), t • (a + 2 • b) = c →
  k = 1 := by
  sorry

end parallel_vectors_imply_k_equals_one_l1247_124792


namespace binomial_12_9_l1247_124753

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end binomial_12_9_l1247_124753


namespace vector_problem_l1247_124790

/-- Given vectors a and b in ℝ², if vector c satisfies the conditions
    (c + b) ⊥ a and (c - a) ∥ b, then c = (2, 1). -/
theorem vector_problem (a b c : ℝ × ℝ) : 
  a = (1, -1) → 
  b = (1, 2) → 
  ((c.1 + b.1, c.2 + b.2) • a = 0) →  -- (c + b) ⊥ a
  (∃ k : ℝ, (c.1 - a.1, c.2 - a.2) = (k * b.1, k * b.2)) →  -- (c - a) ∥ b
  c = (2, 1) :=
sorry

end vector_problem_l1247_124790


namespace tangent_slope_at_pi_half_l1247_124783

theorem tangent_slope_at_pi_half :
  let f (x : ℝ) := Real.tan (x / 2)
  (deriv f) (π / 2) = 1 := by
  sorry

end tangent_slope_at_pi_half_l1247_124783


namespace subtraction_of_fractions_l1247_124763

theorem subtraction_of_fractions : (8 : ℚ) / 19 - (5 : ℚ) / 57 = (1 : ℚ) / 3 := by
  sorry

end subtraction_of_fractions_l1247_124763


namespace x_less_than_one_iff_x_abs_x_less_than_one_l1247_124760

theorem x_less_than_one_iff_x_abs_x_less_than_one (x : ℝ) : x < 1 ↔ x * |x| < 1 := by
  sorry

end x_less_than_one_iff_x_abs_x_less_than_one_l1247_124760


namespace line_intersects_ellipse_l1247_124771

/-- The set of possible slopes for a line with y-intercept (0,3) that intersects
    the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -Real.sqrt 2 / Real.sqrt 110 ∨ m ≥ Real.sqrt 2 / Real.sqrt 110}

/-- The equation of the line with slope m and y-intercept 3 -/
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x + 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def ellipse_equation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

theorem line_intersects_ellipse (m : ℝ) :
  m ∈ possible_slopes ↔
  ∃ x : ℝ, ellipse_equation x (line_equation m x) := by
  sorry

#check line_intersects_ellipse

end line_intersects_ellipse_l1247_124771


namespace drawer_is_translation_l1247_124796

-- Define the possible transformations
inductive Transformation
  | DrawerMovement
  | MagnifyingGlassEffect
  | ClockHandMovement
  | MirrorReflection

-- Define the properties of a translation
def isTranslation (t : Transformation) : Prop :=
  match t with
  | Transformation.DrawerMovement => true
  | _ => false

-- Theorem statement
theorem drawer_is_translation :
  ∀ t : Transformation, isTranslation t ↔ t = Transformation.DrawerMovement :=
by sorry

end drawer_is_translation_l1247_124796


namespace quadratic_inequality_range_l1247_124786

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - x + a > 0) → a > (1/2 : ℝ) := by
  sorry

end quadratic_inequality_range_l1247_124786


namespace movie_start_time_l1247_124742

-- Define the movie duration in minutes
def movie_duration : ℕ := 3 * 60

-- Define the remaining time in minutes
def remaining_time : ℕ := 36

-- Define the end time (5:44 pm) in minutes since midnight
def end_time : ℕ := 17 * 60 + 44

-- Define the start time (to be proven) in minutes since midnight
def start_time : ℕ := 15 * 60 + 20

-- Theorem statement
theorem movie_start_time :
  movie_duration - remaining_time = end_time - start_time :=
by sorry

end movie_start_time_l1247_124742


namespace mr_resty_total_units_l1247_124733

/-- Represents the number of apartment units on each floor of a building -/
def BuildingUnits := List Nat

/-- Building A's unit distribution -/
def building_a : BuildingUnits := [2, 4, 6, 8, 10, 12]

/-- Building B's unit distribution (identical to A) -/
def building_b : BuildingUnits := building_a

/-- Building C's unit distribution -/
def building_c : BuildingUnits := [3, 5, 7, 9]

/-- Calculate the total number of units in a building -/
def total_units (building : BuildingUnits) : Nat :=
  building.sum

/-- The main theorem stating the total number of apartment units Mr. Resty has -/
theorem mr_resty_total_units : 
  total_units building_a + total_units building_b + total_units building_c = 108 := by
  sorry

end mr_resty_total_units_l1247_124733


namespace unknown_number_proof_l1247_124795

theorem unknown_number_proof : 
  ∃ x : ℝ, (45 * x = 0.4 * 900) ∧ (x = 8) := by
  sorry

end unknown_number_proof_l1247_124795


namespace simplify_expression_l1247_124726

theorem simplify_expression : (2 * 10^9) - (6 * 10^7) / (2 * 10^2) = 1999700000 := by
  sorry

end simplify_expression_l1247_124726


namespace mayor_approval_probability_l1247_124761

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 4

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The probability of exactly k successes in n trials with probability p -/
def binomial_probability (p : ℝ) (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem mayor_approval_probability :
  binomial_probability p n k = 0.3456 := by
  sorry

end mayor_approval_probability_l1247_124761


namespace rectangular_field_area_l1247_124766

/-- The area of a rectangular field with length 1.2 meters and width three-fourths of the length is 1.08 square meters. -/
theorem rectangular_field_area : 
  let length : ℝ := 1.2
  let width : ℝ := (3/4) * length
  let area : ℝ := length * width
  area = 1.08 := by sorry

end rectangular_field_area_l1247_124766


namespace floor_negative_seven_halves_l1247_124732

theorem floor_negative_seven_halves : ⌊(-7 : ℚ) / 2⌋ = -4 := by
  sorry

end floor_negative_seven_halves_l1247_124732


namespace betty_bracelets_l1247_124711

/-- The number of bracelets that can be made given a total number of stones and stones per bracelet -/
def num_bracelets (total_stones : ℕ) (stones_per_bracelet : ℕ) : ℕ :=
  total_stones / stones_per_bracelet

/-- Theorem: Given 140 stones and 14 stones per bracelet, the number of bracelets is 10 -/
theorem betty_bracelets :
  num_bracelets 140 14 = 10 := by
  sorry

end betty_bracelets_l1247_124711


namespace quadratic_inequality_theorem_l1247_124747

-- Define the quadratic inequality and its solution set
def quadratic_inequality (a b : ℝ) (x : ℝ) : Prop := a * x^2 + x + b > 0
def solution_set (a b : ℝ) : Set ℝ := {x | x < -2 ∨ x > 1}

-- Define the second inequality
def second_inequality (a b c : ℝ) (x : ℝ) : Prop := a * x^2 - (c + b) * x + b * c < 0

-- Theorem statement
theorem quadratic_inequality_theorem :
  ∀ a b : ℝ, (∀ x : ℝ, quadratic_inequality a b x ↔ x ∈ solution_set a b) →
  (a = 1 ∧ b = -2) ∧
  (∀ c : ℝ, 
    (c = -2 → ∀ x : ℝ, ¬(second_inequality a b c x)) ∧
    (c > -2 → ∀ x : ℝ, second_inequality a b c x ↔ -2 < x ∧ x < c) ∧
    (c < -2 → ∀ x : ℝ, second_inequality a b c x ↔ c < x ∧ x < -2)) :=
by sorry

end quadratic_inequality_theorem_l1247_124747


namespace right_triangle_with_hypotenuse_39_l1247_124738

theorem right_triangle_with_hypotenuse_39 :
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  c = 39 →           -- Hypotenuse length is 39
  (a = 15 ∧ b = 36) ∨ (a = 36 ∧ b = 15) := by
sorry

end right_triangle_with_hypotenuse_39_l1247_124738


namespace certain_number_calculation_l1247_124782

theorem certain_number_calculation (y : ℝ) : (0.65 * 210 = 0.20 * y) → y = 682.5 := by
  sorry

end certain_number_calculation_l1247_124782


namespace rectangle_perimeter_120_l1247_124720

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- Calculate the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.length

/-- Calculate the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.length)

/-- Theorem: A rectangle with area 864 and width 12 less than length has perimeter 120 -/
theorem rectangle_perimeter_120 (r : Rectangle) 
  (h_area : r.area = 864)
  (h_width : r.width + 12 = r.length) :
  r.perimeter = 120 := by
sorry

end rectangle_perimeter_120_l1247_124720


namespace power_five_mod_eighteen_l1247_124773

theorem power_five_mod_eighteen : 5^100 % 18 = 13 := by
  sorry

end power_five_mod_eighteen_l1247_124773


namespace consecutive_odd_numbers_equation_l1247_124780

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem consecutive_odd_numbers_equation (N₁ N₂ N₃ : ℤ) : 
  is_odd N₁ ∧ is_odd N₂ ∧ is_odd N₃ ∧ 
  N₂ = N₁ + 2 ∧ N₃ = N₂ + 2 ∧
  N₁ = 9 →
  N₁ ≠ 3 * N₃ + 16 + 4 * N₂ :=
by sorry

end consecutive_odd_numbers_equation_l1247_124780


namespace min_value_cos_sin_l1247_124752

theorem min_value_cos_sin (x : ℝ) : 
  3 * Real.cos x - 4 * Real.sin x ≥ -5 ∧ 
  ∃ y : ℝ, 3 * Real.cos y - 4 * Real.sin y = -5 :=
by sorry

end min_value_cos_sin_l1247_124752


namespace next_year_with_sum_4_year_2101_is_valid_year_2101_is_smallest_l1247_124716

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isValidYear (year : Nat) : Prop :=
  year > 2020 ∧ sumOfDigits year = 4

theorem next_year_with_sum_4 :
  ∀ year, year > 2020 → sumOfDigits year = 4 → year ≥ 2101 :=
by sorry

theorem year_2101_is_valid :
  isValidYear 2101 :=
by sorry

theorem year_2101_is_smallest :
  ∀ year, isValidYear year → year ≥ 2101 :=
by sorry

end next_year_with_sum_4_year_2101_is_valid_year_2101_is_smallest_l1247_124716


namespace principal_is_8925_l1247_124702

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  (simple_interest * 100) / (rate * time)

/-- Theorem stating that given the specific conditions, the principal amount is 8925 -/
theorem principal_is_8925 :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 9
  let time : ℕ := 5
  calculate_principal simple_interest rate time = 8925 := by
sorry

end principal_is_8925_l1247_124702


namespace egyptian_fraction_sum_exists_l1247_124743

theorem egyptian_fraction_sum_exists : ∃ (b₂ b₃ b₄ b₅ b₆ b₇ : ℕ), 
  (b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₂ ≠ b₅ ∧ b₂ ≠ b₆ ∧ b₂ ≠ b₇ ∧
   b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₃ ≠ b₆ ∧ b₃ ≠ b₇ ∧
   b₄ ≠ b₅ ∧ b₄ ≠ b₆ ∧ b₄ ≠ b₇ ∧
   b₅ ≠ b₆ ∧ b₅ ≠ b₇ ∧
   b₆ ≠ b₇) ∧
  (11 : ℚ) / 13 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧
  (b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 7 ∨
   b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 8 ∨
   b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 9 ∨
   b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 10 ∨
   b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 11) :=
by sorry

end egyptian_fraction_sum_exists_l1247_124743


namespace complex_magnitude_sum_reciprocals_l1247_124767

theorem complex_magnitude_sum_reciprocals (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
sorry

end complex_magnitude_sum_reciprocals_l1247_124767


namespace fourth_grade_students_l1247_124727

theorem fourth_grade_students (initial_students : ℕ) : 
  initial_students + 11 - 5 = 37 → initial_students = 31 := by
  sorry

end fourth_grade_students_l1247_124727


namespace mike_five_dollar_bills_l1247_124725

theorem mike_five_dollar_bills (total_amount : ℕ) (bill_denomination : ℕ) 
  (h1 : total_amount = 45)
  (h2 : bill_denomination = 5) :
  total_amount / bill_denomination = 9 := by
sorry

end mike_five_dollar_bills_l1247_124725


namespace four_spheres_cover_all_rays_l1247_124719

-- Define a point in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

-- Define a ray in 3D space
structure Ray where
  origin : Point3D
  direction : Point3D

-- Function to check if a ray intersects a sphere
def ray_intersects_sphere (r : Ray) (s : Sphere) : Prop :=
  sorry

-- Theorem statement
theorem four_spheres_cover_all_rays :
  ∃ (s1 s2 s3 s4 : Sphere) (light_source : Point3D),
    ∀ (r : Ray),
      r.origin = light_source →
      ray_intersects_sphere r s1 ∨
      ray_intersects_sphere r s2 ∨
      ray_intersects_sphere r s3 ∨
      ray_intersects_sphere r s4 :=
sorry

end four_spheres_cover_all_rays_l1247_124719


namespace service_provider_assignment_l1247_124779

theorem service_provider_assignment (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) = 303600 :=
by sorry

end service_provider_assignment_l1247_124779


namespace unique_base_representation_l1247_124756

/-- The repeating base-k representation of a rational number -/
def repeatingBaseK (n d k : ℕ) : ℚ :=
  (4 : ℚ) / k + (7 : ℚ) / k^2

/-- The condition for the repeating base-k representation to equal the given fraction -/
def isValidK (k : ℕ) : Prop :=
  k > 0 ∧ repeatingBaseK 11 77 k = 11 / 77

theorem unique_base_representation :
  ∃! k : ℕ, isValidK k ∧ k = 17 :=
sorry

end unique_base_representation_l1247_124756


namespace smallest_n_congruence_l1247_124744

theorem smallest_n_congruence : 
  ∃ (n : ℕ), n > 0 ∧ 5 * n ≡ 105 [MOD 24] ∧ 
  ∀ (m : ℕ), m > 0 ∧ 5 * m ≡ 105 [MOD 24] → n ≤ m :=
by sorry

end smallest_n_congruence_l1247_124744


namespace uncle_bradley_bill_change_l1247_124781

theorem uncle_bradley_bill_change (total_amount : ℕ) (small_bill_denom : ℕ) (total_bills : ℕ) :
  total_amount = 1000 →
  small_bill_denom = 50 →
  total_bills = 13 →
  ∃ (large_bill_denom : ℕ),
    (3 * total_amount / 10 / small_bill_denom + (total_amount - 3 * total_amount / 10) / large_bill_denom = total_bills) ∧
    large_bill_denom = 100 := by
  sorry

#check uncle_bradley_bill_change

end uncle_bradley_bill_change_l1247_124781


namespace linear_combination_proof_l1247_124717

theorem linear_combination_proof (A B : Matrix (Fin 3) (Fin 3) ℤ) :
  A = ![![2, -4, 0], ![-1, 5, 1], ![0, 3, -7]] →
  B = ![![4, -1, -2], ![0, -3, 5], ![2, 0, -4]] →
  3 • A - 2 • B = ![![-2, -10, 4], ![-3, 21, -7], ![-4, 9, -13]] := by
  sorry

end linear_combination_proof_l1247_124717


namespace correct_equation_for_john_scenario_l1247_124764

/-- Represents a driving scenario with a stop -/
structure DrivingScenario where
  speed_before_stop : ℝ
  stop_duration : ℝ
  speed_after_stop : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The given driving scenario -/
def john_scenario : DrivingScenario :=
  { speed_before_stop := 60
  , stop_duration := 0.5
  , speed_after_stop := 80
  , total_distance := 200
  , total_time := 4 }

/-- The equation representing the driving scenario -/
def scenario_equation (s : DrivingScenario) (t : ℝ) : Prop :=
  s.speed_before_stop * t + s.speed_after_stop * (s.total_time - s.stop_duration - t) = s.total_distance

/-- Theorem stating that the equation correctly represents John's driving scenario -/
theorem correct_equation_for_john_scenario :
  ∀ t, scenario_equation john_scenario t ↔ 60 * t + 80 * (7/2 - t) = 200 :=
sorry

end correct_equation_for_john_scenario_l1247_124764


namespace dvd_book_total_capacity_l1247_124740

def dvd_book_capacity (current_dvds : ℕ) (additional_dvds : ℕ) : ℕ :=
  current_dvds + additional_dvds

theorem dvd_book_total_capacity :
  dvd_book_capacity 81 45 = 126 := by
  sorry

end dvd_book_total_capacity_l1247_124740


namespace equation_solution_l1247_124777

theorem equation_solution : ∃ x : ℝ, 15 * 2 = 3 + x ∧ x = 27 := by
  sorry

end equation_solution_l1247_124777


namespace exists_unique_polynomial_l1247_124700

/-- Definition of the polynomial p(x, y) -/
def p (x y : ℕ) : ℕ := (x + y)^2 + 3*x + y

/-- Statement of the theorem -/
theorem exists_unique_polynomial :
  ∀ n : ℕ, ∃! (k m : ℕ), p k m = n :=
by sorry

end exists_unique_polynomial_l1247_124700


namespace consecutive_odd_numbers_sum_product_squares_l1247_124778

theorem consecutive_odd_numbers_sum_product_squares : 
  ∃ (a : ℤ), 
    let sequence := List.range 25 |>.map (λ i => a + 2*i - 24)
    ∃ (s p : ℤ), 
      (sequence.sum = s^2) ∧ 
      (sequence.prod = p^2) ∧
      (∀ n ∈ sequence, n % 2 = 1 ∨ n % 2 = -1) := by
  sorry

#check consecutive_odd_numbers_sum_product_squares

end consecutive_odd_numbers_sum_product_squares_l1247_124778


namespace M_congruent_544_mod_1000_l1247_124739

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def M : ℕ :=
  let total_blue := 20
  let total_green := 15
  let total_slots := total_blue + 1
  let ways_to_arrange_greens := Nat.choose total_slots total_green
  let ways_to_divide_poles := total_slots
  let arrangements_with_empty_pole := Nat.choose total_blue total_green
  ways_to_divide_poles * ways_to_arrange_greens - 2 * arrangements_with_empty_pole

/-- The theorem stating that M is congruent to 544 modulo 1000 -/
theorem M_congruent_544_mod_1000 : M ≡ 544 [ZMOD 1000] := by
  sorry

end M_congruent_544_mod_1000_l1247_124739


namespace min_shift_for_sine_overlap_l1247_124797

theorem min_shift_for_sine_overlap (f g : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (2 * x + π / 3)) →
  (∀ x, g x = Real.sin (2 * x)) →
  (∀ x, f x = g (x + π / 6)) →
  (∀ x, f x = g (x + φ)) →
  φ > 0 →
  φ ≥ π / 6 :=
sorry

end min_shift_for_sine_overlap_l1247_124797


namespace student_hostel_cost_theorem_l1247_124728

/-- The cost per day for additional weeks in a student youth hostel -/
def additional_week_cost (first_week_daily_rate : ℚ) (total_days : ℕ) (total_cost : ℚ) : ℚ :=
  let first_week_cost := 7 * first_week_daily_rate
  let additional_days := total_days - 7
  let additional_cost := total_cost - first_week_cost
  additional_cost / additional_days

theorem student_hostel_cost_theorem (first_week_daily_rate : ℚ) (total_days : ℕ) (total_cost : ℚ) 
  (h1 : first_week_daily_rate = 18)
  (h2 : total_days = 23)
  (h3 : total_cost = 334) :
  additional_week_cost first_week_daily_rate total_days total_cost = 13 := by
  sorry

end student_hostel_cost_theorem_l1247_124728


namespace house_work_payment_l1247_124750

/-- Represents the total payment for work on a house --/
def total_payment (bricklayer_rate : ℝ) (electrician_rate : ℝ) (hours_worked : ℝ) : ℝ :=
  bricklayer_rate * hours_worked + electrician_rate * hours_worked

/-- Proves that the total payment for the work is $630 --/
theorem house_work_payment : 
  let bricklayer_rate : ℝ := 12
  let electrician_rate : ℝ := 16
  let hours_worked : ℝ := 22.5
  total_payment bricklayer_rate electrician_rate hours_worked = 630 := by
  sorry

#eval total_payment 12 16 22.5

end house_work_payment_l1247_124750


namespace perpendicular_bisector_of_intersecting_circles_l1247_124791

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*y - 27 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  C₁ A.1 A.2 ∧ C₂ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ B.1 B.2 ∧ A ≠ B

-- Define the perpendicular bisector
def perpendicular_bisector (x y : ℝ) : Prop := x - y = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersecting_circles 
  (A B : ℝ × ℝ) (h : intersection_points A B) :
  ∀ x y, perpendicular_bisector x y ↔ 
    (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1) ∧
    2*x = A.1 + B.1 ∧ 2*y = A.2 + B.2 :=
sorry

end perpendicular_bisector_of_intersecting_circles_l1247_124791


namespace three_digit_divisibility_by_seven_l1247_124757

/-- Represents a three-digit number where the first and last digits are the same -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNum (n : ThreeDigitNumber) : ℕ :=
  100 * n.a + 10 * n.b + n.a

theorem three_digit_divisibility_by_seven (n : ThreeDigitNumber) :
  (n.toNum % 7 = 0) ↔ ((n.a + n.b) % 7 = 0) := by
  sorry

end three_digit_divisibility_by_seven_l1247_124757


namespace ln_sqrt2_lt_ln3_div3_lt_inv_e_l1247_124707

theorem ln_sqrt2_lt_ln3_div3_lt_inv_e : 
  Real.log (Real.sqrt 2) < Real.log 3 / 3 ∧ Real.log 3 / 3 < 1 / Real.exp 1 := by
  sorry

end ln_sqrt2_lt_ln3_div3_lt_inv_e_l1247_124707


namespace smallest_c_value_l1247_124741

theorem smallest_c_value : ∃ c : ℚ, (∀ x : ℚ, (3 * x + 4) * (x - 2) = 9 * x → c ≤ x) ∧ (3 * c + 4) * (c - 2) = 9 * c ∧ c = -8/3 := by
  sorry

end smallest_c_value_l1247_124741


namespace union_of_sets_l1247_124772

theorem union_of_sets : 
  let A : Set ℕ := {0, 1, 2, 3}
  let B : Set ℕ := {1, 2, 4}
  A ∪ B = {0, 1, 2, 3, 4} := by
sorry

end union_of_sets_l1247_124772


namespace apple_pear_equivalence_l1247_124784

theorem apple_pear_equivalence (apple_value pear_value : ℚ) :
  (3/4 : ℚ) * 12 * apple_value = 10 * pear_value →
  (2/3 : ℚ) * 9 * apple_value = (20/3 : ℚ) * pear_value :=
by
  sorry

end apple_pear_equivalence_l1247_124784


namespace square_field_perimeter_l1247_124705

/-- Given a square field enclosed by posts, calculate the outer perimeter of the fence. -/
theorem square_field_perimeter
  (num_posts : ℕ)
  (post_width_inches : ℝ)
  (gap_between_posts_feet : ℝ)
  (h_num_posts : num_posts = 36)
  (h_post_width : post_width_inches = 6)
  (h_gap_between : gap_between_posts_feet = 6) :
  let post_width_feet : ℝ := post_width_inches / 12
  let side_length : ℝ := (num_posts / 4 - 1) * gap_between_posts_feet + num_posts / 4 * post_width_feet
  let perimeter : ℝ := 4 * side_length
  perimeter = 236 := by
  sorry

end square_field_perimeter_l1247_124705


namespace sum_of_squares_and_products_l1247_124770

theorem sum_of_squares_and_products (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x^2 + y^2 + z^2 = 52 → x*y + y*z + z*x = 27 → x + y + z = Real.sqrt 106 := by
  sorry

end sum_of_squares_and_products_l1247_124770


namespace custom_op_one_neg_three_l1247_124710

-- Define the custom operation ※
def custom_op (a b : ℤ) : ℤ := 2 * a * b - b^2

-- Theorem statement
theorem custom_op_one_neg_three : custom_op 1 (-3) = -15 := by
  sorry

end custom_op_one_neg_three_l1247_124710


namespace final_pen_count_l1247_124701

def pen_collection (initial : ℕ) (mike_gives : ℕ) (sharon_takes : ℕ) : ℕ :=
  let after_mike := initial + mike_gives
  let after_cindy := 2 * after_mike
  after_cindy - sharon_takes

theorem final_pen_count : pen_collection 20 22 19 = 65 := by
  sorry

end final_pen_count_l1247_124701


namespace power_sum_of_three_l1247_124706

theorem power_sum_of_three : (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 18 := by
  sorry

end power_sum_of_three_l1247_124706


namespace max_participants_is_seven_l1247_124731

/-- Represents a table tennis tournament -/
structure TableTennisTournament where
  participants : ℕ
  scores : Fin participants → Fin participants → Bool
  no_self_play : ∀ i, scores i i = false
  symmetric : ∀ i j, scores i j = !scores j i

/-- The property that for any four participants, two have the same score -/
def has_equal_scores_property (t : TableTennisTournament) : Prop :=
  ∀ (a b c d : Fin t.participants),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    ∃ (i j : Fin t.participants), i ≠ j ∧
      (t.scores a i + t.scores b i + t.scores c i + t.scores d i =
       t.scores a j + t.scores b j + t.scores c j + t.scores d j)

/-- The main theorem: maximum number of participants is 7 -/
theorem max_participants_is_seven :
  ∀ t : TableTennisTournament, has_equal_scores_property t → t.participants ≤ 7 :=
sorry

end max_participants_is_seven_l1247_124731


namespace stating_count_valid_outfits_l1247_124785

/-- Represents the number of red shirts -/
def red_shirts : ℕ := 7

/-- Represents the number of green shirts -/
def green_shirts : ℕ := 5

/-- Represents the number of pants -/
def pants : ℕ := 6

/-- Represents the number of green hats -/
def green_hats : ℕ := 9

/-- Represents the number of red hats -/
def red_hats : ℕ := 7

/-- Represents the total number of valid outfits -/
def total_outfits : ℕ := 1152

/-- 
Theorem stating that the number of valid outfits is 1152.
A valid outfit consists of one shirt, one pair of pants, and one hat,
where either the shirt and hat don't share the same color,
or the pants and hat don't share the same color.
-/
theorem count_valid_outfits : 
  (red_shirts * pants * green_hats) + 
  (green_shirts * pants * red_hats) + 
  (red_shirts * red_hats * pants) +
  (green_shirts * green_hats * pants) = total_outfits :=
sorry

end stating_count_valid_outfits_l1247_124785
