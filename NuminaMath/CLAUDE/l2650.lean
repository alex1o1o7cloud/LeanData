import Mathlib

namespace NUMINAMATH_CALUDE_intersection_point_l2650_265009

-- Define the two lines
def line1 (x y : ℝ) : Prop := x + 2 * y - 4 = 0
def line2 (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- State the theorem
theorem intersection_point :
  ∃! p : ℝ × ℝ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = (0, 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l2650_265009


namespace NUMINAMATH_CALUDE_ellipse_equation_l2650_265015

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

-- Define the conditions
theorem ellipse_equation :
  ∃ (a b : ℝ), 
    -- Condition 1: Foci on X-axis (implied by standard form)
    -- Condition 2: Major axis is three times minor axis
    a = 3 * b ∧
    -- Condition 3: Passes through (3,0)
    (3, 0) ∈ Ellipse a b ∧
    -- Condition 4: Center at origin (implied by standard form)
    -- Condition 5: Coordinate axes are axes of symmetry (implied by standard form)
    -- Condition 6: Passes through (√6,1) and (-√3,-√2)
    (Real.sqrt 6, 1) ∈ Ellipse a b ∧
    (-Real.sqrt 3, -Real.sqrt 2) ∈ Ellipse a b →
    -- Conclusion: The equation of the ellipse is x²/9 + y²/3 = 1
    Ellipse 3 (Real.sqrt 3) = {p : ℝ × ℝ | (p.1 ^ 2 / 9) + (p.2 ^ 2 / 3) = 1} := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2650_265015


namespace NUMINAMATH_CALUDE_inequality_solution_l2650_265084

theorem inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, a^(6 - x) > a^(2 + 3*x) ↔ 
    ((0 < a ∧ a < 1 ∧ x > 1) ∨ (a > 1 ∧ x < 1))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2650_265084


namespace NUMINAMATH_CALUDE_collectiveEarnings_l2650_265071

-- Define the workers and their properties
structure Worker where
  name : String
  normalHours : Float
  hourlyRate : Float
  overtimeMultiplier : Float
  actualHours : Float

-- Calculate earnings for a worker
def calculateEarnings (w : Worker) : Float :=
  let regularPay := min w.normalHours w.actualHours * w.hourlyRate
  let overtimeHours := max (w.actualHours - w.normalHours) 0
  let overtimePay := overtimeHours * w.hourlyRate * w.overtimeMultiplier
  regularPay + overtimePay

-- Define Lloyd and Casey
def lloyd : Worker := {
  name := "Lloyd"
  normalHours := 7.5
  hourlyRate := 4.50
  overtimeMultiplier := 2.0
  actualHours := 10.5
}

def casey : Worker := {
  name := "Casey"
  normalHours := 8.0
  hourlyRate := 5.00
  overtimeMultiplier := 1.5
  actualHours := 9.5
}

-- Theorem: Lloyd and Casey's collective earnings equal $112.00
theorem collectiveEarnings : calculateEarnings lloyd + calculateEarnings casey = 112.00 := by
  sorry

end NUMINAMATH_CALUDE_collectiveEarnings_l2650_265071


namespace NUMINAMATH_CALUDE_max_min_powers_l2650_265021

theorem max_min_powers (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  let M := max (max (a^a) (a^b)) (max (b^a) (b^b))
  let m := min (min (a^a) (a^b)) (min (b^a) (b^b))
  M = b^a ∧ m = a^b := by
sorry

end NUMINAMATH_CALUDE_max_min_powers_l2650_265021


namespace NUMINAMATH_CALUDE_partition_characterization_l2650_265050

/-- The set V_p for a prime p -/
def V_p (p : ℕ) : Set ℕ :=
  {k | p ∣ (k * (k + 1) / 2) ∧ k ≥ 2 * p - 1}

/-- A partition of the set {1,2,...,k} into p subsets -/
def IsValidPartition (p k : ℕ) (partition : List (List ℕ)) : Prop :=
  (partition.length = p) ∧
  (partition.join.toFinset = Finset.range k) ∧
  (∀ s ∈ partition, s.sum = (partition.head!).sum)

theorem partition_characterization (p : ℕ) (hp : Nat.Prime p) :
  ∀ k : ℕ, (∃ partition : List (List ℕ), IsValidPartition p k partition) ↔ k ∈ V_p p :=
sorry

end NUMINAMATH_CALUDE_partition_characterization_l2650_265050


namespace NUMINAMATH_CALUDE_fraction_of_360_l2650_265068

theorem fraction_of_360 : (1 / 3) * (1 / 4) * (1 / 5) * (1 / 6) * 360 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_360_l2650_265068


namespace NUMINAMATH_CALUDE_expected_sixes_is_one_third_l2650_265099

/-- The probability of rolling a 6 on a standard die -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a standard die -/
def prob_not_six : ℚ := 1 - prob_six

/-- The expected number of 6's when rolling two standard dice -/
def expected_sixes : ℚ := 
  0 * (prob_not_six ^ 2) + 
  1 * (2 * prob_six * prob_not_six) + 
  2 * (prob_six ^ 2)

theorem expected_sixes_is_one_third : expected_sixes = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_expected_sixes_is_one_third_l2650_265099


namespace NUMINAMATH_CALUDE_original_total_price_l2650_265004

theorem original_total_price (candy_price soda_price chips_price chocolate_price : ℝ) :
  candy_price = 10 ∧ soda_price = 6 ∧ chips_price = 4 ∧ chocolate_price = 2 →
  candy_price + soda_price + chips_price + chocolate_price = 22 := by
sorry

end NUMINAMATH_CALUDE_original_total_price_l2650_265004


namespace NUMINAMATH_CALUDE_technician_round_trip_theorem_l2650_265014

theorem technician_round_trip_theorem (D : ℝ) (h : D > 0) :
  let round_trip_distance := 2 * D
  let total_distance_traveled := 0.6 * round_trip_distance
  let distance_after_center := total_distance_traveled - D
  distance_after_center / D = 0.2 := by sorry

end NUMINAMATH_CALUDE_technician_round_trip_theorem_l2650_265014


namespace NUMINAMATH_CALUDE_sector_area_90_degrees_l2650_265017

/-- The area of a sector with radius 2 and central angle 90° is π. -/
theorem sector_area_90_degrees : 
  let r : ℝ := 2
  let angle : ℝ := 90
  let sector_area := (angle / 360) * π * r^2
  sector_area = π := by sorry

end NUMINAMATH_CALUDE_sector_area_90_degrees_l2650_265017


namespace NUMINAMATH_CALUDE_max_value_at_two_l2650_265063

/-- A function f(x) = ax² + 4(a-1)x - 3 defined on the interval [0,2] -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * (a - 1) * x - 3

/-- The domain of the function -/
def domain : Set ℝ := Set.Icc 0 2

theorem max_value_at_two (a : ℝ) :
  (∀ x ∈ domain, f a x ≤ f a 2) → a ∈ Set.Ici (2/3) :=
sorry

end NUMINAMATH_CALUDE_max_value_at_two_l2650_265063


namespace NUMINAMATH_CALUDE_machinery_expenditure_l2650_265077

theorem machinery_expenditure (total : ℝ) (raw_materials : ℝ) (machinery : ℝ) :
  total = 93750 →
  raw_materials = 35000 →
  machinery + raw_materials + (0.2 * total) = total →
  machinery = 40000 := by
sorry

end NUMINAMATH_CALUDE_machinery_expenditure_l2650_265077


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l2650_265076

-- Define the arithmetic sequence
def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

-- State the theorem
theorem arithmetic_sequence_common_difference_range :
  ∀ d : ℝ,
  (∀ n : ℕ, n < 6 → arithmeticSequence (-15) d n ≤ 0) ∧
  (∀ n : ℕ, n ≥ 6 → arithmeticSequence (-15) d n > 0) →
  3 < d ∧ d ≤ 15/4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l2650_265076


namespace NUMINAMATH_CALUDE_least_multiple_33_above_500_l2650_265003

theorem least_multiple_33_above_500 : 
  ∀ n : ℕ, n > 0 ∧ 33 ∣ n ∧ n > 500 → n ≥ 528 :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_33_above_500_l2650_265003


namespace NUMINAMATH_CALUDE_equation_solution_l2650_265031

theorem equation_solution :
  ∃! x : ℝ, (5 : ℝ)^x * 125^(3*x) = 625^7 ∧ x = 2.8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2650_265031


namespace NUMINAMATH_CALUDE_zoe_country_albums_l2650_265010

/-- The number of pop albums Zoe bought -/
def pop_albums : ℕ := 5

/-- The number of songs per album -/
def songs_per_album : ℕ := 3

/-- The total number of songs Zoe bought -/
def total_songs : ℕ := 24

/-- The number of country albums Zoe bought -/
def country_albums : ℕ := (total_songs - pop_albums * songs_per_album) / songs_per_album

theorem zoe_country_albums : country_albums = 3 := by
  sorry

end NUMINAMATH_CALUDE_zoe_country_albums_l2650_265010


namespace NUMINAMATH_CALUDE_distance_to_origin_l2650_265038

def z₁ : ℂ := Complex.I
def z₂ : ℂ := 1 + Complex.I

theorem distance_to_origin (z : ℂ := z₁ * z₂) :
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_origin_l2650_265038


namespace NUMINAMATH_CALUDE_blue_chips_count_l2650_265073

theorem blue_chips_count (total : ℚ) 
  (h1 : total * (1 / 10) + total * (1 / 2) + 12 = total) : 
  total * (1 / 10) = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_chips_count_l2650_265073


namespace NUMINAMATH_CALUDE_remainder_of_n_l2650_265067

theorem remainder_of_n (n : ℕ) 
  (h1 : n^2 % 5 = 1) 
  (h2 : n^3 % 5 = 4) : 
  n % 5 = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_n_l2650_265067


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l2650_265023

theorem cost_of_dozen_pens (pen_cost pencil_cost : ℚ) : 
  (3 * pen_cost + 5 * pencil_cost = 150) →
  (pen_cost = 5 * pencil_cost) →
  (12 * pen_cost = 450) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l2650_265023


namespace NUMINAMATH_CALUDE_special_ellipse_eccentricity_l2650_265000

/-- An ellipse with the property that the lines connecting the two vertices 
    on the minor axis and one of its foci are perpendicular to each other. -/
structure SpecialEllipse where
  /-- Semi-major axis length -/
  a : ℝ
  /-- Semi-minor axis length -/
  b : ℝ
  /-- Distance from center to focus -/
  c : ℝ
  /-- The ellipse satisfies a² = b² + c² -/
  h1 : a^2 = b^2 + c^2
  /-- The lines connecting the vertices on the minor axis and a focus are perpendicular -/
  h2 : b = c

/-- The eccentricity of a SpecialEllipse is √2/2 -/
theorem special_ellipse_eccentricity (E : SpecialEllipse) : 
  E.c / E.a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_eccentricity_l2650_265000


namespace NUMINAMATH_CALUDE_square_garden_perimeter_l2650_265025

theorem square_garden_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 520 → perimeter = 40 * Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_square_garden_perimeter_l2650_265025


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2650_265030

theorem trigonometric_equation_solution (x : ℝ) :
  2 * Real.cos x - 5 * Real.sin x = 3 →
  (3 * Real.sin x + 2 * Real.cos x = (-21 + 13 * Real.sqrt 145) / 58) ∨
  (3 * Real.sin x + 2 * Real.cos x = (-21 - 13 * Real.sqrt 145) / 58) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2650_265030


namespace NUMINAMATH_CALUDE_group_size_proof_l2650_265075

def group_collection (n : ℕ) : ℕ := n * n

theorem group_size_proof (total_rupees : ℕ) (h : group_collection 90 = total_rupees * 100) : 
  ∃ (n : ℕ), group_collection n = total_rupees * 100 ∧ n = 90 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l2650_265075


namespace NUMINAMATH_CALUDE_c_invests_after_eight_months_l2650_265040

/-- Represents the investment problem with three partners A, B, and C --/
structure InvestmentProblem where
  initial_investment : ℝ
  annual_gain : ℝ
  a_share : ℝ
  b_invest_time : ℕ
  c_invest_time : ℕ

/-- Calculates the time when C invests given the problem parameters --/
def calculate_c_invest_time (problem : InvestmentProblem) : ℕ :=
  let a_investment := problem.initial_investment * 12
  let b_investment := 2 * problem.initial_investment * (12 - problem.b_invest_time)
  let c_investment := 3 * problem.initial_investment * problem.c_invest_time
  let total_investment := a_investment + b_investment + c_investment
  problem.c_invest_time

/-- Theorem stating that C invests after 8 months --/
theorem c_invests_after_eight_months (problem : InvestmentProblem) 
  (h1 : problem.annual_gain = 21000)
  (h2 : problem.a_share = 7000)
  (h3 : problem.b_invest_time = 6) :
  calculate_c_invest_time problem = 8 := by
  sorry

#eval calculate_c_invest_time {
  initial_investment := 1000,
  annual_gain := 21000,
  a_share := 7000,
  b_invest_time := 6,
  c_invest_time := 8
}

end NUMINAMATH_CALUDE_c_invests_after_eight_months_l2650_265040


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2650_265011

/-- Given a geometric sequence {aₙ} with common ratio 2 and a₁ + a₃ = 5, prove that a₂ + a₄ = 10 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  a 1 + a 3 = 5 →               -- given condition
  a 2 + a 4 = 10 :=             -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2650_265011


namespace NUMINAMATH_CALUDE_fraction_division_equality_l2650_265080

theorem fraction_division_equality : (-4/9 + 1/6 - 2/3) / (-1/18) = 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l2650_265080


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2650_265018

/-- A function f: R⁺ → R⁺ satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (f x + y) * f x = f (x * y + 1)

/-- The theorem stating the solution to the functional equation -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
    (∀ x, x > 1 → f x = 1 / x) ∧ f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2650_265018


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2650_265072

theorem largest_n_divisible_by_seven (n : ℕ) : 
  n < 50000 →
  (5 * (n - 3)^5 - 3 * n^2 + 20 * n - 35) % 7 = 0 →
  n ≤ 49999 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l2650_265072


namespace NUMINAMATH_CALUDE_managers_salary_l2650_265001

/-- Given an organization with 20 employees and a manager, prove the manager's salary
    based on the change in average salary. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (avg_increase : ℝ) :
  num_employees = 20 →
  avg_salary = 1600 →
  avg_increase = 100 →
  (num_employees * avg_salary + (avg_salary + avg_increase) * (num_employees + 1)) -
    (num_employees * avg_salary) = 3700 :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l2650_265001


namespace NUMINAMATH_CALUDE_divisible_by_120_l2650_265016

theorem divisible_by_120 (n : ℕ) : ∃ k : ℤ, (n^5 : ℤ) - 5*(n^3 : ℤ) + 4*(n : ℤ) = 120*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_120_l2650_265016


namespace NUMINAMATH_CALUDE_trefoils_per_case_l2650_265049

theorem trefoils_per_case (total_boxes : ℕ) (total_cases : ℕ) (boxes_per_case : ℕ) :
  total_boxes = 24 →
  total_cases = 3 →
  total_boxes = boxes_per_case * total_cases →
  boxes_per_case = 8 := by
  sorry

end NUMINAMATH_CALUDE_trefoils_per_case_l2650_265049


namespace NUMINAMATH_CALUDE_solve_equation_l2650_265051

-- Define the function f
def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 1)

-- State the theorem
theorem solve_equation (a : ℝ) : f (f a) = f 9 + 1 → a = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2650_265051


namespace NUMINAMATH_CALUDE_project_contribution_balance_l2650_265037

/-- The contribution of the first worker to the project -/
def first_worker_contribution : ℚ := 1/3

/-- The contribution of the second worker to the project -/
def second_worker_contribution : ℚ := 1/3

/-- The contribution of the third worker to the project -/
def third_worker_contribution : ℚ := 1/3

/-- The total full-time equivalent (FTE) for the project -/
def total_fte : ℚ := 1

theorem project_contribution_balance :
  first_worker_contribution + second_worker_contribution + third_worker_contribution = total_fte :=
sorry

end NUMINAMATH_CALUDE_project_contribution_balance_l2650_265037


namespace NUMINAMATH_CALUDE_bob_benefit_reduction_l2650_265053

/-- Calculates the monthly reduction in housing benefit given a raise, work hours, and net increase --/
def monthly_benefit_reduction (raise_per_hour : ℚ) (hours_per_week : ℕ) (net_increase_per_week : ℚ) : ℚ :=
  4 * (raise_per_hour * hours_per_week - net_increase_per_week)

/-- Theorem stating that given the specific conditions, the monthly reduction in housing benefit is $60 --/
theorem bob_benefit_reduction :
  monthly_benefit_reduction (1/2) 40 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_bob_benefit_reduction_l2650_265053


namespace NUMINAMATH_CALUDE_moving_point_on_line_segment_l2650_265036

/-- Two fixed points in a plane -/
structure FixedPoints where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  distance : dist F₁ F₂ = 16

/-- A moving point M satisfying the condition |MF₁| + |MF₂| = 16 -/
def MovingPoint (fp : FixedPoints) (M : ℝ × ℝ) : Prop :=
  dist M fp.F₁ + dist M fp.F₂ = 16

/-- The theorem stating that any moving point M lies on the line segment F₁F₂ -/
theorem moving_point_on_line_segment (fp : FixedPoints) (M : ℝ × ℝ) 
    (h : MovingPoint fp M) : 
    ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • fp.F₁ + t • fp.F₂ :=
  sorry

end NUMINAMATH_CALUDE_moving_point_on_line_segment_l2650_265036


namespace NUMINAMATH_CALUDE_statue_cost_calculation_l2650_265047

/-- Given a statue sold for $750 with a 35% profit, prove that the original cost was $555.56 (rounded to two decimal places). -/
theorem statue_cost_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 750)
  (h2 : profit_percentage = 35) : 
  ∃ (original_cost : ℝ), 
    selling_price = original_cost * (1 + profit_percentage / 100) ∧ 
    (round (original_cost * 100) / 100 : ℝ) = 555.56 := by
  sorry

end NUMINAMATH_CALUDE_statue_cost_calculation_l2650_265047


namespace NUMINAMATH_CALUDE_square_perimeter_from_rearranged_rectangles_l2650_265012

/-- 
Given a square cut into four equal rectangles, which are then arranged to form a shape 
with perimeter 56, prove that the perimeter of the original square is 32.
-/
theorem square_perimeter_from_rearranged_rectangles 
  (rectangle_width : ℝ) 
  (rectangle_length : ℝ) 
  (h1 : rectangle_length = 4 * rectangle_width) 
  (h2 : 28 * rectangle_width = 56) : 
  4 * (2 * rectangle_length) = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rearranged_rectangles_l2650_265012


namespace NUMINAMATH_CALUDE_solution_is_five_l2650_265097

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := 
  x > 3 ∧ log10 (x - 3) + log10 x = 1

-- State the theorem
theorem solution_is_five : 
  ∃ (x : ℝ), equation x ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_solution_is_five_l2650_265097


namespace NUMINAMATH_CALUDE_integral_bounds_l2650_265019

theorem integral_bounds : 
  let f : ℝ → ℝ := λ x => 1 / (1 + 3 * Real.sin x ^ 2)
  let a : ℝ := 0
  let b : ℝ := Real.pi / 6
  (2 * Real.pi) / 21 ≤ ∫ x in a..b, f x ∧ ∫ x in a..b, f x ≤ Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_integral_bounds_l2650_265019


namespace NUMINAMATH_CALUDE_carrie_harvest_l2650_265006

/-- Represents the number of carrots Carrie harvested -/
def num_carrots : ℕ := 350

/-- Represents the number of tomatoes Carrie harvested -/
def num_tomatoes : ℕ := 200

/-- Represents the price of a tomato in cents -/
def tomato_price : ℕ := 100

/-- Represents the price of a carrot in cents -/
def carrot_price : ℕ := 150

/-- Represents the total revenue in cents -/
def total_revenue : ℕ := 72500

theorem carrie_harvest :
  num_tomatoes * tomato_price + num_carrots * carrot_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_carrie_harvest_l2650_265006


namespace NUMINAMATH_CALUDE_pencil_problem_l2650_265083

theorem pencil_problem (s p t : ℚ) : 
  (6 * s = 12) →
  (t = 8 * s) →
  (p = 2.5 * s + 3) →
  (t = 16 ∧ p = 8) := by
sorry

end NUMINAMATH_CALUDE_pencil_problem_l2650_265083


namespace NUMINAMATH_CALUDE_vector_parallel_problem_l2650_265091

/-- Given plane vectors a, b, c, prove that k = -8 -/
theorem vector_parallel_problem (a b c : ℝ × ℝ) (k : ℝ) : 
  a = (-1, 1) →
  b = (2, 3) →
  c = (-2, k) →
  (∃ (t : ℝ), t ≠ 0 ∧ (a.1 + b.1, a.2 + b.2) = (t * c.1, t * c.2)) →
  k = -8 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_problem_l2650_265091


namespace NUMINAMATH_CALUDE_alice_winning_strategy_l2650_265095

/-- Represents the game state with n objects and maximum removal of m objects per turn -/
structure GameState where
  n : ℕ  -- Number of objects in the pile
  m : ℕ  -- Maximum number of objects that can be removed per turn

/-- Predicate to check if a player has a winning strategy -/
def has_winning_strategy (state : GameState) : Prop :=
  ¬(state.n + 1 ∣ state.m)

/-- Theorem stating the condition for Alice to have a winning strategy -/
theorem alice_winning_strategy (state : GameState) :
  has_winning_strategy state ↔ ¬(state.n + 1 ∣ state.m) :=
sorry

end NUMINAMATH_CALUDE_alice_winning_strategy_l2650_265095


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l2650_265054

theorem geometric_arithmetic_sequence_sum (x y z : ℝ) 
  (h1 : (4 * y)^2 = 15 * x * z)  -- Geometric sequence condition
  (h2 : 2 / y = 1 / x + 1 / z)   -- Arithmetic sequence condition
  : x / z + z / x = 34 / 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_l2650_265054


namespace NUMINAMATH_CALUDE_cube_derivative_three_l2650_265066

theorem cube_derivative_three (f : ℝ → ℝ) (x₀ : ℝ) :
  (∀ x, f x = x^3) →
  (deriv f x₀ = 3) →
  (x₀ = 1 ∨ x₀ = -1) := by
sorry

end NUMINAMATH_CALUDE_cube_derivative_three_l2650_265066


namespace NUMINAMATH_CALUDE_eliza_ironing_time_l2650_265090

theorem eliza_ironing_time :
  ∀ (blouse_time : ℝ),
    (blouse_time > 0) →
    (120 / blouse_time + 180 / 20 = 17) →
    blouse_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_eliza_ironing_time_l2650_265090


namespace NUMINAMATH_CALUDE_max_sum_with_product_2310_l2650_265060

theorem max_sum_with_product_2310 :
  ∀ A B C : ℕ+,
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2310 →
  A + B + C ≤ 48 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_product_2310_l2650_265060


namespace NUMINAMATH_CALUDE_root_equations_imply_m_n_values_l2650_265094

theorem root_equations_imply_m_n_values (m n : ℝ) : 
  (∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧
    ((r1 + m) * (r1 + n) * (r1 + 8)) / ((r1 + 2)^2) = 0 ∧
    ((r2 + m) * (r2 + n) * (r2 + 8)) / ((r2 + 2)^2) = 0 ∧
    ((r3 + m) * (r3 + n) * (r3 + 8)) / ((r3 + 2)^2) = 0) →
  (∃! (r : ℝ), ((r + 2*m) * (r + 4) * (r + 10)) / ((r + n) * (r + 8)) = 0) →
  m = 1 ∧ n = 4 ∧ 50*m + n = 54 := by
sorry

end NUMINAMATH_CALUDE_root_equations_imply_m_n_values_l2650_265094


namespace NUMINAMATH_CALUDE_rita_swimming_months_l2650_265096

/-- The number of months Rita needs to fulfill her coach's requirements -/
def months_to_fulfill_requirement (total_required_hours : ℕ) (hours_already_completed : ℕ) (hours_per_month : ℕ) : ℕ :=
  (total_required_hours - hours_already_completed) / hours_per_month

/-- Proof that Rita needs 6 months to fulfill her coach's requirements -/
theorem rita_swimming_months : 
  months_to_fulfill_requirement 1500 180 220 = 6 := by
sorry

end NUMINAMATH_CALUDE_rita_swimming_months_l2650_265096


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_existence_l2650_265008

theorem quadratic_inequality_solution_existence (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 10*x + c < 0) ↔ (c > 0 ∧ c < 25) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_existence_l2650_265008


namespace NUMINAMATH_CALUDE_jones_elementary_population_l2650_265013

theorem jones_elementary_population :
  let total_students : ℕ := 360
  let boys_percentage : ℚ := 1/2
  let sample_size : ℕ := 90
  (boys_percentage * boys_percentage * total_students = sample_size) →
  total_students = 360 :=
by sorry

end NUMINAMATH_CALUDE_jones_elementary_population_l2650_265013


namespace NUMINAMATH_CALUDE_school_trip_photos_l2650_265046

theorem school_trip_photos (claire lisa robert : ℕ) : 
  lisa = 3 * claire →
  robert = claire + 24 →
  lisa = robert →
  claire = 12 := by
sorry

end NUMINAMATH_CALUDE_school_trip_photos_l2650_265046


namespace NUMINAMATH_CALUDE_lcm_of_36_and_176_l2650_265081

theorem lcm_of_36_and_176 :
  let a : ℕ := 36
  let b : ℕ := 176
  let hcf : ℕ := 16
  Nat.gcd a b = hcf →
  Nat.lcm a b = 396 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_36_and_176_l2650_265081


namespace NUMINAMATH_CALUDE_sandwich_count_l2650_265044

def num_meat : ℕ := 12
def num_cheese : ℕ := 11
def num_toppings : ℕ := 8

def sandwich_combinations : ℕ := (num_meat.choose 2) * (num_cheese.choose 2) * (num_toppings.choose 2)

theorem sandwich_count : sandwich_combinations = 101640 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_count_l2650_265044


namespace NUMINAMATH_CALUDE_g_value_at_3056_l2650_265033

theorem g_value_at_3056 (g : ℝ → ℝ) 
  (h1 : ∀ x > 0, g x > 0)
  (h2 : ∀ x y, x > y ∧ y > 0 → g (x - y) = Real.sqrt (g (x * y) + 4))
  (h3 : ∃ x y, x > y ∧ y > 0 ∧ x - y = x * y ∧ x * y = 3056) :
  g 3056 = 2 := by
sorry

end NUMINAMATH_CALUDE_g_value_at_3056_l2650_265033


namespace NUMINAMATH_CALUDE_quadratic_equation_one_l2650_265085

theorem quadratic_equation_one (x : ℝ) :
  (x + 2) * (x - 2) - 2 * (x - 3) = 3 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_l2650_265085


namespace NUMINAMATH_CALUDE_smallest_benches_arrangement_l2650_265074

theorem smallest_benches_arrangement (M : ℕ+) (n : ℕ+) : 
  (9 * M.val = n ∧ 14 * M.val = n) → M.val ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_benches_arrangement_l2650_265074


namespace NUMINAMATH_CALUDE_simplify_expression_l2650_265024

theorem simplify_expression (a b : ℝ) : a * b^2 * (-2 * a^3 * b) = -2 * a^4 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2650_265024


namespace NUMINAMATH_CALUDE_statements_correctness_l2650_265089

theorem statements_correctness :
  (∃ a b : ℝ, a > b ∧ 1/a > 1/b ∧ a*b ≤ 0) ∧
  (∀ a b c : ℝ, a > b ∧ b > 0 ∧ c < 0 → c/a > c/b) ∧
  (∃ a b : ℝ, a < b ∧ b < 0 ∧ a^2 ≥ b^2) ∧
  (∀ a b c d : ℝ, a > b ∧ b > 0 ∧ c < d ∧ d < 0 → a*c < b*d) :=
by sorry

end NUMINAMATH_CALUDE_statements_correctness_l2650_265089


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2650_265070

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := (x - 2)^2 - 4

/-- The logarithm base 1/2 -/
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

theorem quadratic_function_properties :
  (∀ x, f x ≥ -4) ∧  -- Minimum value is -4
  (f 2 = -4) ∧  -- Minimum occurs at x = 2
  (f 0 = 0) ∧  -- Passes through origin
  (∀ x, x ∈ Set.Icc (1/8 : ℝ) 2 → f (log_half x) ≥ -4) ∧  -- Minimum in the interval
  (∃ x, x ∈ Set.Icc (1/8 : ℝ) 2 ∧ f (log_half x) = -4) ∧  -- Minimum is attained
  (∀ x, x ∈ Set.Icc (1/8 : ℝ) 2 → f (log_half x) ≤ 5) ∧  -- Maximum in the interval
  (∃ x, x ∈ Set.Icc (1/8 : ℝ) 2 ∧ f (log_half x) = 5)  -- Maximum is attained
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2650_265070


namespace NUMINAMATH_CALUDE_f_is_fraction_l2650_265093

-- Define what a fraction is
def is_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (n d : ℚ → ℚ), ∀ x, x ≠ 0 → f x = (n x) / (d x) ∧ d x ≠ 0

-- Define the specific function we're examining
def f (x : ℚ) : ℚ := (x + 3) / x

-- Theorem statement
theorem f_is_fraction : is_fraction f := by sorry

end NUMINAMATH_CALUDE_f_is_fraction_l2650_265093


namespace NUMINAMATH_CALUDE_decimal_point_problem_l2650_265058

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 3 / Real.sqrt 1000 := by
sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l2650_265058


namespace NUMINAMATH_CALUDE_xy_value_l2650_265055

theorem xy_value (x y : ℝ) (h : |x - 1| + (y + 2)^2 = 0) : x * y = -2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2650_265055


namespace NUMINAMATH_CALUDE_log_inequality_l2650_265082

theorem log_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c) :
  Real.log c / Real.log a > Real.log c / Real.log b :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l2650_265082


namespace NUMINAMATH_CALUDE_zoo_animals_l2650_265035

theorem zoo_animals (b r m : ℕ) : 
  b + r + m = 300 →
  2 * b + 3 * r + 4 * m = 798 →
  r = 102 :=
by sorry

end NUMINAMATH_CALUDE_zoo_animals_l2650_265035


namespace NUMINAMATH_CALUDE_max_sum_of_counts_l2650_265065

/-- Represents the color of a card -/
inductive CardColor
  | White
  | Black
  | Red

/-- Represents a stack of cards -/
structure CardStack :=
  (cards : List CardColor)
  (white_count : Nat)
  (black_count : Nat)
  (red_count : Nat)

/-- Calculates the sum of counts for a given card stack -/
def calculate_sum (stack : CardStack) : Nat :=
  sorry

/-- Theorem stating the maximum possible sum of counts -/
theorem max_sum_of_counts (stack : CardStack) 
  (h1 : stack.cards.length = 300)
  (h2 : stack.white_count = 100)
  (h3 : stack.black_count = 100)
  (h4 : stack.red_count = 100) :
  (∀ s : CardStack, calculate_sum s ≤ calculate_sum stack) →
  calculate_sum stack = 20000 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_counts_l2650_265065


namespace NUMINAMATH_CALUDE_hiker_catchup_time_l2650_265041

/-- Proves that a hiker catches up to a motorcyclist in 48 minutes under given conditions -/
theorem hiker_catchup_time (hiker_speed : ℝ) (motorcyclist_speed : ℝ) (stop_time : ℝ) : 
  hiker_speed = 6 →
  motorcyclist_speed = 30 →
  stop_time = 12 / 60 →
  (motorcyclist_speed * stop_time - hiker_speed * stop_time) / hiker_speed * 60 = 48 := by
sorry

end NUMINAMATH_CALUDE_hiker_catchup_time_l2650_265041


namespace NUMINAMATH_CALUDE_two_plus_insertion_theorem_l2650_265005

/-- Represents a way to split a number into three parts by inserting two plus signs -/
structure ThreePartSplit (n : ℕ) :=
  (first second third : ℕ)
  (split_valid : n = first * 100000 + second * 100 + third)
  (no_rearrange : first < 100 ∧ second < 1000 ∧ third < 100)

/-- The problem statement -/
theorem two_plus_insertion_theorem :
  ∃ (split : ThreePartSplit 8789924),
    split.first + split.second + split.third = 1010 := by
  sorry

end NUMINAMATH_CALUDE_two_plus_insertion_theorem_l2650_265005


namespace NUMINAMATH_CALUDE_probability_not_purple_l2650_265039

/-- Given a bag of marbles where the odds of pulling a purple marble are 5:6,
    prove that the probability of not pulling a purple marble is 6/11. -/
theorem probability_not_purple (total : ℕ) (purple : ℕ) (not_purple : ℕ) :
  total = purple + not_purple →
  purple = 5 →
  not_purple = 6 →
  (not_purple : ℚ) / total = 6 / 11 :=
by sorry

end NUMINAMATH_CALUDE_probability_not_purple_l2650_265039


namespace NUMINAMATH_CALUDE_expression_evaluation_l2650_265064

theorem expression_evaluation : 72 + (150 / 25) + (16 * 19) - 250 - (450 / 9) = 82 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2650_265064


namespace NUMINAMATH_CALUDE_fixed_point_of_function_l2650_265056

theorem fixed_point_of_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a * (x + 1) + 2
  f (-1) = 3 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_function_l2650_265056


namespace NUMINAMATH_CALUDE_tank_inlet_rate_l2650_265020

/-- Given a tank with the following properties:
  * Capacity of 3600.000000000001 liters
  * Empties in 6 hours due to a leak
  * Empties in 8 hours when both the leak and inlet are open
  Prove that the rate at which the inlet pipe fills the tank is 150 liters per hour -/
theorem tank_inlet_rate (capacity : ℝ) (leak_time : ℝ) (combined_time : ℝ) :
  capacity = 3600.000000000001 →
  leak_time = 6 →
  combined_time = 8 →
  ∃ (inlet_rate : ℝ),
    inlet_rate = 150 ∧
    inlet_rate = (capacity / leak_time) - (capacity / combined_time) :=
by sorry

end NUMINAMATH_CALUDE_tank_inlet_rate_l2650_265020


namespace NUMINAMATH_CALUDE_grandmother_doll_count_l2650_265026

/-- Represents the number of dolls each person has -/
structure DollCounts where
  grandmother : ℕ
  sister : ℕ
  rene : ℕ

/-- Defines the conditions of the doll distribution problem -/
def validDollDistribution (d : DollCounts) : Prop :=
  d.rene = 3 * d.sister ∧
  d.sister = d.grandmother + 2 ∧
  d.rene + d.sister + d.grandmother = 258

/-- Theorem stating that the grandmother has 50 dolls -/
theorem grandmother_doll_count :
  ∀ d : DollCounts, validDollDistribution d → d.grandmother = 50 := by
  sorry

end NUMINAMATH_CALUDE_grandmother_doll_count_l2650_265026


namespace NUMINAMATH_CALUDE_base10_to_base8_2357_l2650_265042

-- Define a function to convert a base 10 number to base 8
def toBase8 (n : ℕ) : List ℕ :=
  sorry

-- Theorem stating that 2357 in base 10 is equal to 4445 in base 8
theorem base10_to_base8_2357 :
  toBase8 2357 = [4, 4, 4, 5] :=
sorry

end NUMINAMATH_CALUDE_base10_to_base8_2357_l2650_265042


namespace NUMINAMATH_CALUDE_extremum_of_g_and_range_of_a_l2650_265057

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := Real.exp x - a * x^2
def g (x : ℝ) : ℝ := Real.exp x - 2 * a * x

theorem extremum_of_g_and_range_of_a :
  (a > 0 → ∃ (x_min : ℝ), x_min = Real.log (2 * a) ∧ 
    (∀ y, g a y ≥ g a x_min) ∧ 
    g a x_min = 2 * a - 2 * a * Real.log (2 * a)) ∧
  ((∀ x ≥ 0, f a x ≥ x + (1 - x) * Real.exp x) → a ≤ 1) :=
sorry

end

end NUMINAMATH_CALUDE_extremum_of_g_and_range_of_a_l2650_265057


namespace NUMINAMATH_CALUDE_perpendicular_sum_implies_zero_l2650_265043

/-- Given vectors a and b in ℝ², if a is perpendicular to (a + b), then the second component of b is 0. -/
theorem perpendicular_sum_implies_zero (a b : ℝ × ℝ) (h : a.1 = -1 ∧ a.2 = 1 ∧ b.1 = 2) :
  (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0) → b.2 = 0 := by
  sorry

#check perpendicular_sum_implies_zero

end NUMINAMATH_CALUDE_perpendicular_sum_implies_zero_l2650_265043


namespace NUMINAMATH_CALUDE_ball_placement_count_is_42_l2650_265022

/-- The number of ways to place four distinct balls into three labeled boxes
    such that exactly one box remains empty. -/
def ballPlacementCount : ℕ := 42

/-- Theorem stating that the number of ways to place four distinct balls
    into three labeled boxes such that exactly one box remains empty is 42. -/
theorem ball_placement_count_is_42 : ballPlacementCount = 42 := by
  sorry

end NUMINAMATH_CALUDE_ball_placement_count_is_42_l2650_265022


namespace NUMINAMATH_CALUDE_fifth_term_value_l2650_265034

/-- A geometric sequence with common ratio 2 and positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem fifth_term_value (a : ℕ → ℝ) :
  GeometricSequence a → a 3 * a 11 = 16 → a 5 = 1 := by
  sorry


end NUMINAMATH_CALUDE_fifth_term_value_l2650_265034


namespace NUMINAMATH_CALUDE_slope_implies_y_value_l2650_265028

/-- Given two points A(4, y) and B(2, -3), if the slope of the line passing through these points is π/4, then y = -1 -/
theorem slope_implies_y_value (y : ℝ) :
  let A : ℝ × ℝ := (4, y)
  let B : ℝ × ℝ := (2, -3)
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = π / 4 → y = -1 := by
  sorry

end NUMINAMATH_CALUDE_slope_implies_y_value_l2650_265028


namespace NUMINAMATH_CALUDE_misha_additional_earnings_l2650_265092

/-- Represents Misha's savings situation -/
def MishaSavings (x : ℝ) (y : ℝ) (z : ℝ) : Prop :=
  x + (y / 100 * x) * z = 47

theorem misha_additional_earnings :
  ∀ (y z : ℝ), MishaSavings 34 y z → 47 - 34 = 13 := by
  sorry

end NUMINAMATH_CALUDE_misha_additional_earnings_l2650_265092


namespace NUMINAMATH_CALUDE_jim_flour_on_counter_l2650_265088

/-- The amount of flour Jim has on the kitchen counter -/
def flour_on_counter (flour_in_cupboard flour_in_pantry flour_per_loaf : ℕ) (loaves_can_bake : ℕ) : ℕ :=
  loaves_can_bake * flour_per_loaf - (flour_in_cupboard + flour_in_pantry)

/-- Theorem stating that Jim has 100g of flour on the kitchen counter -/
theorem jim_flour_on_counter :
  flour_on_counter 200 100 200 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_jim_flour_on_counter_l2650_265088


namespace NUMINAMATH_CALUDE_arithmetic_sum_33_l2650_265098

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum_33 (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 4 + a 7 = 45 →
  a 2 + a 5 + a 8 = 39 →
  a 3 + a 6 + a 9 = 33 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sum_33_l2650_265098


namespace NUMINAMATH_CALUDE_train_speed_problem_l2650_265007

theorem train_speed_problem (train_length : ℝ) (crossing_time : ℝ) : 
  train_length = 120 → 
  crossing_time = 16 → 
  ∃ (speed : ℝ), speed = 27 ∧ 
    (2 * train_length) / crossing_time * 3.6 = 2 * speed := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2650_265007


namespace NUMINAMATH_CALUDE_carnation_count_l2650_265052

theorem carnation_count (total_flowers : ℕ) (roses : ℕ) (carnations : ℕ) : 
  total_flowers = 10 → roses = 5 → total_flowers = roses + carnations → carnations = 5 := by
  sorry

end NUMINAMATH_CALUDE_carnation_count_l2650_265052


namespace NUMINAMATH_CALUDE_stone_99_is_11_l2650_265078

/-- Represents the counting pattern for 12 stones -/
def stone_count (n : ℕ) : ℕ :=
  let cycle := 22  -- The pattern repeats every 22 counts
  let within_cycle := n % cycle
  if within_cycle ≤ 12
  then within_cycle
  else 13 - (within_cycle - 11)

/-- The theorem stating that the 99th count corresponds to the 11th stone -/
theorem stone_99_is_11 : stone_count 99 = 11 := by
  sorry

#eval stone_count 99  -- This should output 11

end NUMINAMATH_CALUDE_stone_99_is_11_l2650_265078


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2650_265086

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y + 6 = 0 → x + a*(a+1)*y + (a^2-1) = 0 → 
   (a * 1 + 2 * (a*(a+1)) = 0)) → 
  (a = 0 ∨ a = -3/2) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2650_265086


namespace NUMINAMATH_CALUDE_greatest_x_value_l2650_265029

theorem greatest_x_value (x : ℝ) :
  (x^2 - x - 90) / (x - 9) = 2 / (x + 6) →
  x ≤ -8 + Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2650_265029


namespace NUMINAMATH_CALUDE_wayne_shrimp_appetizer_cost_l2650_265032

/-- Calculates the total cost of shrimp for an appetizer given the number of shrimp per guest,
    number of guests, cost per pound of shrimp, and number of shrimp per pound. -/
def shrimp_appetizer_cost (shrimp_per_guest : ℕ) (num_guests : ℕ) (cost_per_pound : ℚ) (shrimp_per_pound : ℕ) : ℚ :=
  (shrimp_per_guest * num_guests : ℚ) / shrimp_per_pound * cost_per_pound

/-- Proves that Wayne's shrimp appetizer cost is $170.00 given the specified conditions. -/
theorem wayne_shrimp_appetizer_cost :
  shrimp_appetizer_cost 5 40 17 20 = 170 :=
by sorry

end NUMINAMATH_CALUDE_wayne_shrimp_appetizer_cost_l2650_265032


namespace NUMINAMATH_CALUDE_trigonometric_values_and_difference_l2650_265045

def angle_α : ℝ := sorry

def point_P : ℝ × ℝ := (3, -4)

theorem trigonometric_values_and_difference :
  (point_P.1 = 3 ∧ point_P.2 = -4) →  -- Point P(3, -4) lies on the terminal side of angle α
  (Real.sin α * Real.cos α = 1/8) →   -- sinα*cosα = 1/8
  (π < α ∧ α < 5*π/4) →               -- π < α < 5π/4
  (Real.sin α = -4/5 ∧ 
   Real.cos α = 3/5 ∧ 
   Real.tan α = -4/3 ∧
   Real.cos α - Real.sin α = -Real.sqrt 3 / 12) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_values_and_difference_l2650_265045


namespace NUMINAMATH_CALUDE_trig_combination_l2650_265059

theorem trig_combination (x : ℝ) : 
  Real.cos (3 * x) + Real.cos (5 * x) + Real.tan (2 * x) = 
  2 * Real.cos (4 * x) * Real.cos x + Real.sin (2 * x) / Real.cos (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_trig_combination_l2650_265059


namespace NUMINAMATH_CALUDE_lines_intersect_on_ellipse_l2650_265002

/-- Two lines intersect and their intersection point lies on a specific ellipse -/
theorem lines_intersect_on_ellipse (k₁ k₂ : ℝ) (h : k₁ * k₂ + 2 = 0) :
  ∃ (x y : ℝ),
    (y = k₁ * x + 1 ∧ y = k₂ * x - 1) ∧  -- Lines intersect
    2 * x^2 + y^2 = 6 :=                 -- Intersection point on ellipse
by sorry

end NUMINAMATH_CALUDE_lines_intersect_on_ellipse_l2650_265002


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2650_265087

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 6) / 12 = 6 / (x - 12) ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2650_265087


namespace NUMINAMATH_CALUDE_westerville_gnomes_l2650_265069

theorem westerville_gnomes (ravenswood westerville : ℕ) : 
  ravenswood = 4 * westerville →
  (60 * ravenswood) / 100 = 48 →
  westerville = 20 := by
sorry

end NUMINAMATH_CALUDE_westerville_gnomes_l2650_265069


namespace NUMINAMATH_CALUDE_arithmetic_seq_inequality_l2650_265062

/-- A positive arithmetic sequence with non-zero common difference -/
structure PosArithmeticSeq where
  a : ℕ → ℝ
  pos : ∀ n, a n > 0
  diff : ∃ d ≠ 0, ∀ n, a (n + 1) = a n + d

theorem arithmetic_seq_inequality (seq : PosArithmeticSeq) : seq.a 1 * seq.a 8 < seq.a 4 * seq.a 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_inequality_l2650_265062


namespace NUMINAMATH_CALUDE_average_entry_exit_time_is_200_l2650_265061

/-- Represents the position and movement of a car and storm -/
structure CarStormSystem where
  carSpeed : ℝ
  stormRadius : ℝ
  stormSpeedSouth : ℝ
  stormSpeedEast : ℝ
  initialNorthDistance : ℝ

/-- Calculates the average of the times when the car enters and exits the storm -/
def averageEntryExitTime (system : CarStormSystem) : ℝ :=
  200

/-- Theorem stating that the average entry/exit time is 200 minutes -/
theorem average_entry_exit_time_is_200 (system : CarStormSystem) 
  (h1 : system.carSpeed = 1)
  (h2 : system.stormRadius = 60)
  (h3 : system.stormSpeedSouth = 3/4)
  (h4 : system.stormSpeedEast = 1/4)
  (h5 : system.initialNorthDistance = 150) :
  averageEntryExitTime system = 200 := by
  sorry

end NUMINAMATH_CALUDE_average_entry_exit_time_is_200_l2650_265061


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2650_265048

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 2*x > 35) ↔ (x < -5 ∨ x > 7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2650_265048


namespace NUMINAMATH_CALUDE_francis_muffins_l2650_265079

/-- The cost of breakfast for Francis and Kiera -/
def breakfast_cost : ℕ → ℕ
| m => 2 * m + 2 * 3 + 2 * 2 + 3

/-- The theorem stating that Francis had 2 muffins -/
theorem francis_muffins : 
  ∃ m : ℕ, breakfast_cost m = 17 ∧ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_francis_muffins_l2650_265079


namespace NUMINAMATH_CALUDE_polygon_angles_l2650_265027

theorem polygon_angles (n : ℕ) (h : n ≥ 3) :
  (n - 2) * 180 + (360 / n) = 1500 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angles_l2650_265027
