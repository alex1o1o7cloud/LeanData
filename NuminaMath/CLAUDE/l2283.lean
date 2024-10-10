import Mathlib

namespace functional_equation_solution_l2283_228329

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (f : RealFunction) :
  (∀ x y : ℝ, f (x - f y) = 1 - x - y) → 
  (∀ x : ℝ, f x = 1/2 - x) :=
by sorry

end functional_equation_solution_l2283_228329


namespace prize_winning_condition_xiao_feng_inequality_l2283_228360

/-- Represents the school intelligence competition --/
structure Competition where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  prize_threshold : ℤ

/-- Represents a student participating in the competition --/
structure Student where
  correct_answers : ℕ
  won_prize : Prop

/-- The specific competition described in the problem --/
def school_competition : Competition :=
  { total_questions := 20
  , correct_points := 5
  , incorrect_points := -2
  , prize_threshold := 75 }

/-- Theorem stating the condition for winning a prize --/
theorem prize_winning_condition (s : Student) (c : Competition) 
  (h1 : s.won_prize) 
  (h2 : s.correct_answers ≤ c.total_questions) :
  c.correct_points * s.correct_answers + 
  c.incorrect_points * (c.total_questions - s.correct_answers) > 
  c.prize_threshold := by
  sorry

/-- Theorem for Xiao Feng's specific case --/
theorem xiao_feng_inequality (x : ℕ) :
  x ≤ school_competition.total_questions →
  5 * x - 2 * (20 - x) > 75 := by
  sorry

end prize_winning_condition_xiao_feng_inequality_l2283_228360


namespace joel_age_when_dad_twice_as_old_l2283_228335

theorem joel_age_when_dad_twice_as_old (joel_current_age dad_current_age : ℕ) : 
  joel_current_age = 5 → 
  dad_current_age = 32 → 
  ∃ (years : ℕ), 
    dad_current_age + years = 2 * (joel_current_age + years) ∧ 
    joel_current_age + years = 27 := by
sorry

end joel_age_when_dad_twice_as_old_l2283_228335


namespace jerry_money_duration_l2283_228375

/-- The number of weeks Jerry's money will last given his earnings and weekly spending -/
def weeks_money_lasts (lawn_earnings weed_eating_earnings weekly_spending : ℕ) : ℕ :=
  (lawn_earnings + weed_eating_earnings) / weekly_spending

/-- Theorem stating that Jerry's money will last 9 weeks -/
theorem jerry_money_duration :
  weeks_money_lasts 14 31 5 = 9 := by
  sorry

end jerry_money_duration_l2283_228375


namespace reciprocal_of_negative_two_l2283_228350

theorem reciprocal_of_negative_two :
  ∀ x : ℚ, x * (-2) = 1 → x = -1/2 := by
  sorry

end reciprocal_of_negative_two_l2283_228350


namespace inequality_solution_set_l2283_228351

theorem inequality_solution_set :
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end inequality_solution_set_l2283_228351


namespace soda_cost_l2283_228364

theorem soda_cost (burger_cost soda_cost : ℕ) : 
  (3 * burger_cost + 2 * soda_cost = 450) →
  (2 * burger_cost + 3 * soda_cost = 480) →
  soda_cost = 108 := by
sorry

end soda_cost_l2283_228364


namespace cubic_polynomial_roots_l2283_228308

/-- A polynomial of the form x^3 - 2ax^2 + bx - 2a -/
def cubic_polynomial (a b : ℝ) (x : ℝ) : ℝ := x^3 - 2*a*x^2 + b*x - 2*a

/-- The condition that a polynomial has all real roots -/
def has_all_real_roots (p : ℝ → ℝ) : Prop :=
  ∃ r s t : ℝ, ∀ x : ℝ, p x = (x - r) * (x - s) * (x - t)

/-- The theorem stating the relationship between a and b for the given polynomial -/
theorem cubic_polynomial_roots (a b : ℝ) :
  (a > 0 ∧ a = 3 * Real.sqrt 3 / 2 ∧ b = 81 / 4) ↔
  (has_all_real_roots (cubic_polynomial a b) ∧
   ∀ a' > 0, has_all_real_roots (cubic_polynomial a' b) → a ≤ a') :=
sorry

end cubic_polynomial_roots_l2283_228308


namespace measure_water_l2283_228315

theorem measure_water (a : ℤ) (h : -1562 ≤ a ∧ a ≤ 1562) :
  ∃ (b c d e f : ℤ), 
    (b ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (c ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (d ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (e ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (f ∈ ({-2, -1, 0, 1, 2} : Set ℤ)) ∧
    (a = 625*b + 125*c + 25*d + 5*e + f) :=
by sorry

end measure_water_l2283_228315


namespace equivalent_discount_l2283_228312

/-- Proves that a single discount of 23.5% on $1200 results in the same final price
    as successive discounts of 15% and 10%. -/
theorem equivalent_discount (original_price : ℝ) (discount1 discount2 single_discount : ℝ) :
  original_price = 1200 →
  discount1 = 0.15 →
  discount2 = 0.10 →
  single_discount = 0.235 →
  original_price * (1 - discount1) * (1 - discount2) = original_price * (1 - single_discount) :=
by sorry

end equivalent_discount_l2283_228312


namespace pirate_treasure_l2283_228385

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 := by
  sorry

end pirate_treasure_l2283_228385


namespace john_twice_sam_age_l2283_228393

def john_age (sam_age : ℕ) : ℕ := 3 * sam_age

def sam_current_age : ℕ := 7 + 2

theorem john_twice_sam_age (years : ℕ) : 
  john_age sam_current_age + years = 2 * (sam_current_age + years) → years = 9 := by
  sorry

end john_twice_sam_age_l2283_228393


namespace polynomial_simplification_l2283_228376

theorem polynomial_simplification (r : ℝ) : 
  (2 * r^3 + 5 * r^2 + 4 * r - 3) - (r^3 + 4 * r^2 + 6 * r - 8) = r^3 + r^2 - 2 * r + 5 := by
  sorry

end polynomial_simplification_l2283_228376


namespace sine_cosine_sum_equals_sqrt3_over_2_l2283_228348

theorem sine_cosine_sum_equals_sqrt3_over_2 : 
  Real.sin (20 * π / 180) * Real.cos (40 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (40 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end sine_cosine_sum_equals_sqrt3_over_2_l2283_228348


namespace ducks_theorem_l2283_228366

def ducks_remaining (initial : ℕ) : ℕ :=
  let after_first := initial - (initial / 4)
  let after_second := after_first - (after_first / 6)
  after_second - (after_second * 3 / 10)

theorem ducks_theorem : ducks_remaining 320 = 140 := by
  sorry

end ducks_theorem_l2283_228366


namespace income_calculation_l2283_228361

/-- Calculates a person's income given the income to expenditure ratio and savings amount. -/
def calculate_income (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) : ℕ :=
  (income_ratio * savings) / (income_ratio - expenditure_ratio)

/-- Theorem stating that given the specified conditions, the person's income is 10000. -/
theorem income_calculation (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) 
  (h1 : income_ratio = 10)
  (h2 : expenditure_ratio = 7)
  (h3 : savings = 3000) :
  calculate_income income_ratio expenditure_ratio savings = 10000 := by
  sorry

end income_calculation_l2283_228361


namespace one_match_probability_l2283_228382

/-- The number of balls and boxes -/
def n : ℕ := 4

/-- The total number of ways to distribute balls into boxes -/
def total_arrangements : ℕ := n.factorial

/-- The number of ways to distribute balls with exactly one color match -/
def matching_arrangements : ℕ := n * ((n - 1).factorial)

/-- The probability of exactly one ball matching its box color -/
def probability_one_match : ℚ := matching_arrangements / total_arrangements

theorem one_match_probability :
  probability_one_match = 1/3 :=
sorry

end one_match_probability_l2283_228382


namespace whole_milk_fat_percentage_l2283_228326

theorem whole_milk_fat_percentage :
  let reduced_fat_percentage : ℚ := 2
  let reduction_percentage : ℚ := 40
  let whole_milk_fat_percentage : ℚ := reduced_fat_percentage / (1 - reduction_percentage / 100)
  whole_milk_fat_percentage = 10 / 3 := by
  sorry

end whole_milk_fat_percentage_l2283_228326


namespace rounding_accuracy_of_1_35_billion_l2283_228354

theorem rounding_accuracy_of_1_35_billion :
  ∃ n : ℕ, (1350000000 : ℕ) = n * 10000000 ∧ n % 10 ≠ 0 :=
sorry

end rounding_accuracy_of_1_35_billion_l2283_228354


namespace simplify_expression_l2283_228388

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 := by
  sorry

end simplify_expression_l2283_228388


namespace cylinder_volume_with_square_section_l2283_228334

/-- Given a cylinder with a square axial section of area 4, its volume is 2π. -/
theorem cylinder_volume_with_square_section (r h : ℝ) : 
  r * h = 2 →  -- The axial section is a square
  r * r * h = 4 →  -- The area of the square is 4
  π * r * r * h = 2 * π :=  -- The volume of the cylinder is 2π
by
  sorry

#check cylinder_volume_with_square_section

end cylinder_volume_with_square_section_l2283_228334


namespace triangle_side_length_l2283_228369

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sinB : ℝ
  area : ℝ

-- Define the conditions
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

def isArithmeticSequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

def hasSinB (t : Triangle) : Prop :=
  t.sinB = 4/5

def hasArea (t : Triangle) : Prop :=
  t.area = 3/2

-- Theorem statement
theorem triangle_side_length (t : Triangle) 
  (h1 : isValidTriangle t)
  (h2 : isArithmeticSequence t)
  (h3 : hasSinB t)
  (h4 : hasArea t) :
  t.b = 2 := by
  sorry

end triangle_side_length_l2283_228369


namespace saturday_practice_hours_l2283_228394

/-- Given a person's practice schedule, calculate the hours practiced on Saturdays -/
theorem saturday_practice_hours 
  (weekday_hours : ℕ) 
  (total_weeks : ℕ) 
  (total_practice_hours : ℕ) 
  (h1 : weekday_hours = 3)
  (h2 : total_weeks = 3)
  (h3 : total_practice_hours = 60) :
  (total_practice_hours - weekday_hours * 5 * total_weeks) / total_weeks = 5 := by
  sorry

#check saturday_practice_hours

end saturday_practice_hours_l2283_228394


namespace actual_annual_yield_actual_annual_yield_approx_l2283_228368

/-- Calculates the actual annual yield for a one-year term deposit with varying interest rates and a closing fee. -/
theorem actual_annual_yield (P : ℝ) : ℝ :=
  let first_quarter_rate := 0.12 / 4
  let second_quarter_rate := 0.08 / 4
  let third_semester_rate := 0.06 / 2
  let closing_fee_rate := 0.01
  let final_amount := P * (1 + first_quarter_rate) * (1 + second_quarter_rate) * (1 + third_semester_rate)
  let effective_final_amount := final_amount - (P * closing_fee_rate)
  (effective_final_amount / P) - 1

/-- The actual annual yield is approximately 7.2118% -/
theorem actual_annual_yield_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ ∀ (P : ℝ), P > 0 → |actual_annual_yield P - 0.072118| < ε :=
sorry

end actual_annual_yield_actual_annual_yield_approx_l2283_228368


namespace rectangular_solid_diagonal_angles_sum_l2283_228302

/-- A rectangular solid with a diagonal forming angles with edges. -/
structure RectangularSolid where
  /-- Length of the diagonal -/
  diagonal : ℝ
  /-- Angle between diagonal and first edge -/
  α : ℝ
  /-- Angle between diagonal and second edge -/
  β : ℝ
  /-- Angle between diagonal and third edge -/
  γ : ℝ
  /-- The angles are formed by the diagonal and edges of the rectangular solid -/
  angles_from_edges : True

/-- 
In a rectangular solid, if one of the diagonals forms angles α, β, and γ 
with the three edges emanating from one of the vertices, 
then cos²α + cos²β + cos²γ = 1.
-/
theorem rectangular_solid_diagonal_angles_sum 
  (rs : RectangularSolid) : Real.cos rs.α ^ 2 + Real.cos rs.β ^ 2 + Real.cos rs.γ ^ 2 = 1 := by
  sorry

end rectangular_solid_diagonal_angles_sum_l2283_228302


namespace cylinder_volume_from_rectangle_l2283_228318

/-- The volume of a cylinder obtained by rotating a rectangle about its longer side -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (length_positive : 0 < length) (width_positive : 0 < width) (length_longer : width ≤ length) :
  let radius := width / 2
  let height := length
  let volume := π * radius^2 * height
  (length = 10 ∧ width = 8) → volume = 160 * π :=
by sorry

end cylinder_volume_from_rectangle_l2283_228318


namespace min_value_sum_squares_l2283_228389

theorem min_value_sum_squares (a b : ℝ) : 
  a > 0 → b > 0 → a ≠ b → a^2 - 2015*a = b^2 - 2015*b → 
  ∀ x y : ℝ, x > 0 → y > 0 → x ≠ y → x^2 - 2015*x = y^2 - 2015*y → 
  a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 2015^2 / 2 := by
sorry

end min_value_sum_squares_l2283_228389


namespace second_shipment_weight_l2283_228373

/-- Represents the weight of couscous shipments and dishes made at a Moroccan restaurant -/
structure CouscousShipments where
  first_shipment : ℕ
  second_shipment : ℕ
  third_shipment : ℕ
  num_dishes : ℕ
  couscous_per_dish : ℕ

/-- Theorem stating the weight of the second shipment of couscous -/
theorem second_shipment_weight (c : CouscousShipments) 
  (h1 : c.first_shipment = 7)
  (h2 : c.third_shipment = 45)
  (h3 : c.num_dishes = 13)
  (h4 : c.couscous_per_dish = 5)
  (h5 : c.first_shipment + c.second_shipment + c.third_shipment = c.num_dishes * c.couscous_per_dish) :
  c.second_shipment = 13 := by
  sorry

end second_shipment_weight_l2283_228373


namespace oranges_in_box_l2283_228317

/-- The number of oranges left in a box after some are removed -/
def oranges_left (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem stating that 20 oranges are left in the box -/
theorem oranges_in_box : oranges_left 55 35 = 20 := by
  sorry

end oranges_in_box_l2283_228317


namespace probability_all_different_at_most_one_odd_l2283_228320

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (all different numbers with at most one odd) -/
def favorableOutcomes : ℕ := 60

/-- The probability of rolling three dice and getting all different numbers with at most one odd number -/
def probabilityAllDifferentAtMostOneOdd : ℚ := favorableOutcomes / totalOutcomes

theorem probability_all_different_at_most_one_odd :
  probabilityAllDifferentAtMostOneOdd = 5 / 18 := by
  sorry

end probability_all_different_at_most_one_odd_l2283_228320


namespace cab_journey_time_l2283_228323

/-- Given a cab walking at 5/6 of its usual speed and arriving 12 minutes late,
    prove that its usual time to cover the journey is 1 hour. -/
theorem cab_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
    (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (usual_speed * usual_time = (5/6 * usual_speed) * (usual_time + 1/5)) → 
  usual_time = 1 := by
  sorry

#check cab_journey_time

end cab_journey_time_l2283_228323


namespace fibonacci_period_divisibility_l2283_228372

def is_extractable (p : ℕ) : Prop := ∃ x : ℕ, x * x ≡ 5 [MOD p]

def period_length (p : ℕ) : ℕ := sorry

theorem fibonacci_period_divisibility (p : ℕ) (hp : Prime p) (hp_neq : p ≠ 2 ∧ p ≠ 5) :
  (¬is_extractable p → (period_length p) ∣ (p + 1)) ∧
  (is_extractable p → (period_length p) ∣ (p - 1)) := by
  sorry

end fibonacci_period_divisibility_l2283_228372


namespace cone_volume_from_lateral_surface_l2283_228352

/-- Given a cone whose lateral surface, when unfolded, forms a semicircle with an area of 2π,
    the volume of the cone is (√3/3)π. -/
theorem cone_volume_from_lateral_surface (l r h : ℝ) : 
  l > 0 ∧ r > 0 ∧ h > 0 ∧
  (1/2) * Real.pi * l^2 = 2 * Real.pi ∧  -- Area of semicircle is 2π
  2 * Real.pi * r = Real.pi * l ∧        -- Circumference of base equals arc length of semicircle
  h^2 + r^2 = l^2 →                      -- Pythagorean theorem
  (1/3) * Real.pi * r^2 * h = (Real.sqrt 3 / 3) * Real.pi := by
sorry


end cone_volume_from_lateral_surface_l2283_228352


namespace arithmetic_sequence_fifth_term_l2283_228316

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + 3 * a 3 + a 15 = 10) :
  a 5 = 2 := by
  sorry

end arithmetic_sequence_fifth_term_l2283_228316


namespace union_P_complement_Q_l2283_228300

open Set

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def Q : Set ℝ := {x | x^2 - 4 < 0}

-- State the theorem
theorem union_P_complement_Q : P ∪ (univ \ Q) = Iic (-2) ∪ Ici 1 := by sorry

end union_P_complement_Q_l2283_228300


namespace problem_solution_l2283_228331

theorem problem_solution (a b : ℝ) 
  (h1 : a * b = 2 * (a + b) + 1) 
  (h2 : b - a = 4) : 
  b = 7 := by
sorry

end problem_solution_l2283_228331


namespace rectangle_to_hexagon_area_l2283_228355

/-- Given a rectangle with sides of length a and 36, prove that when transformed into a hexagon
    with parallel sides of length a separated by 24, and the hexagon has the same area as the
    original rectangle, then a² = 720. -/
theorem rectangle_to_hexagon_area (a : ℝ) : 
  (0 < a) →
  (24 * a + 30 * Real.sqrt (a^2 - 36) = 36 * a) →
  a^2 = 720 := by
sorry

end rectangle_to_hexagon_area_l2283_228355


namespace six_students_three_colleges_l2283_228321

/-- The number of ways n students can apply to m colleges --/
def totalApplications (n m : ℕ) : ℕ := m^n

/-- The number of ways to apply to a subset of colleges --/
def subsetApplications (n k : ℕ) : ℕ := k^n

/-- The number of ways n students can apply to m colleges with each college receiving at least one application --/
def validApplications (n m : ℕ) : ℕ :=
  totalApplications n m - m * subsetApplications n (m-1) + (m.choose 2) * subsetApplications n (m-2)

theorem six_students_three_colleges :
  validApplications 6 3 = 540 := by sorry

end six_students_three_colleges_l2283_228321


namespace anna_cannot_afford_tour_l2283_228330

/-- Calculates the future value of an amount with compound interest -/
def futureValue (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Represents Anna's initial savings -/
def initialSavings : ℝ := 40000

/-- Represents the initial cost of the tour package -/
def initialCost : ℝ := 45000

/-- Represents the annual interest rate -/
def interestRate : ℝ := 0.05

/-- Represents the annual inflation rate -/
def inflationRate : ℝ := 0.05

/-- Represents the time period in years -/
def timePeriod : ℕ := 3

/-- Theorem stating that Anna cannot afford the tour package after 3 years -/
theorem anna_cannot_afford_tour : 
  futureValue initialSavings interestRate timePeriod < 
  futureValue initialCost inflationRate timePeriod := by
  sorry

end anna_cannot_afford_tour_l2283_228330


namespace cubic_root_sum_cubes_l2283_228386

theorem cubic_root_sum_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
  sorry

end cubic_root_sum_cubes_l2283_228386


namespace work_completion_men_count_first_group_size_l2283_228356

theorem work_completion_men_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun first_group_days second_group_men second_group_days result =>
    first_group_days * result = second_group_men * second_group_days

theorem first_group_size :
  ∃ (m : ℕ), work_completion_men_count 80 20 40 m ∧ m = 10 := by
  sorry

end work_completion_men_count_first_group_size_l2283_228356


namespace angle_calculation_l2283_228377

/-- Given an angle α with its vertex at the origin, its initial side coinciding with
    the non-negative half-axis of the x-axis, and a point P(-2, -1) on its terminal side,
    prove that 2cos²α - sin(π - 2α) = 4/5 -/
theorem angle_calculation (α : ℝ) :
  (∃ (P : ℝ × ℝ), P = (-2, -1) ∧
    P.1 = Real.cos α * Real.sqrt 5 ∧
    P.2 = Real.sin α * Real.sqrt 5) →
  2 * (Real.cos α)^2 - Real.sin (π - 2 * α) = 4/5 :=
by sorry

end angle_calculation_l2283_228377


namespace opposite_of_2023_l2283_228380

theorem opposite_of_2023 :
  ∃ x : ℤ, (2023 + x = 0) ∧ (x = -2023) :=
by sorry

end opposite_of_2023_l2283_228380


namespace one_zero_of_sin_log_l2283_228353

open Real

noncomputable def f (x : ℝ) : ℝ := sin (log x)

theorem one_zero_of_sin_log (h : ∀ x, 1 < x → x < exp π → f x = 0 → x = exp π) :
  ∃! x, 1 < x ∧ x < exp π ∧ f x = 0 :=
sorry

end one_zero_of_sin_log_l2283_228353


namespace snow_leopard_arrangement_l2283_228349

theorem snow_leopard_arrangement (n : ℕ) (h : n = 9) : 
  2 * Nat.factorial (n - 2) = 10080 := by
  sorry

end snow_leopard_arrangement_l2283_228349


namespace cube_root_properties_l2283_228371

theorem cube_root_properties :
  let n : ℕ := 59319
  let a : ℕ := 6859
  let b : ℕ := 19683
  let c : ℕ := 110592
  ∃ (x y z : ℕ),
    (10 ≤ x ∧ x < 100) ∧
    x^3 = n ∧
    x = 39 ∧
    y^3 = a ∧ y = 19 ∧
    z^3 = b ∧ z = 27 ∧
    (∃ w : ℕ, w^3 = c ∧ w = 48) :=
by sorry

end cube_root_properties_l2283_228371


namespace smallest_r_minus_p_l2283_228399

theorem smallest_r_minus_p : ∃ (p q r : ℕ+),
  (p * q * r = 362880) ∧   -- 9! = 362880
  (p < q) ∧ (q < r) ∧
  ∀ (p' q' r' : ℕ+),
    (p' * q' * r' = 362880) →
    (p' < q') → (q' < r') →
    (r - p : ℤ) ≤ (r' - p' : ℤ) ∧
  (r - p : ℤ) = 219 := by
  sorry

end smallest_r_minus_p_l2283_228399


namespace function_value_problem_l2283_228384

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x, f (x/2 - 1) = 2*x + 3) 
  (h2 : f m = 6) : 
  m = -1/4 := by sorry

end function_value_problem_l2283_228384


namespace cube_volume_problem_l2283_228338

/-- Given a cube with side length a, prove that if we form a rectangular solid
    by increasing one edge by 2, decreasing another by 1, and leaving the third unchanged,
    and if the volume of this new solid is 14 more than the original cube,
    then the volume of the original cube is 64. -/
theorem cube_volume_problem (a : ℕ) : 
  (a + 2) * (a - 1) * a = a^3 + 14 → a^3 = 64 := by
  sorry

end cube_volume_problem_l2283_228338


namespace selection_ways_eq_55_l2283_228327

/-- The number of ways to select 5 students out of 5 male and 3 female students,
    ensuring both male and female students are included. -/
def selection_ways : ℕ :=
  Nat.choose 8 5 - Nat.choose 5 5

/-- Theorem stating that the number of ways to select 5 students
    out of 5 male and 3 female students, ensuring both male and
    female students are included, is equal to 55. -/
theorem selection_ways_eq_55 : selection_ways = 55 := by
  sorry

end selection_ways_eq_55_l2283_228327


namespace notes_count_l2283_228336

theorem notes_count (total_amount : ℕ) (denominations : Fin 3 → ℕ) : 
  total_amount = 480 ∧ 
  denominations 0 = 1 ∧ 
  denominations 1 = 5 ∧ 
  denominations 2 = 10 ∧ 
  (∃ x : ℕ, (denominations 0 * x + denominations 1 * x + denominations 2 * x = total_amount)) →
  (∃ x : ℕ, x + x + x = 90) :=
by sorry

end notes_count_l2283_228336


namespace first_five_valid_numbers_l2283_228310

def is_valid (n : ℕ) : Bool :=
  n ≥ 0 ∧ n ≤ 499

def random_sequence : List ℕ :=
  [164, 785, 916, 955, 567, 199, 810, 507, 185, 128, 673, 580, 744, 395]

def first_five_valid (seq : List ℕ) : List ℕ :=
  seq.filter is_valid |> List.take 5

theorem first_five_valid_numbers :
  first_five_valid random_sequence = [164, 199, 185, 128, 395] := by
  sorry

end first_five_valid_numbers_l2283_228310


namespace quadratic_roots_expression_l2283_228305

theorem quadratic_roots_expression (p q : ℝ) : 
  (3 * p^2 - 7 * p + 4 = 0) →
  (3 * q^2 - 7 * q + 4 = 0) →
  p ≠ q →
  (5 * p^3 - 5 * q^3) / (p - q) = 185 / 9 := by
sorry

end quadratic_roots_expression_l2283_228305


namespace power_of_power_three_squared_four_l2283_228397

theorem power_of_power_three_squared_four : (3^2)^4 = 6561 := by
  sorry

end power_of_power_three_squared_four_l2283_228397


namespace rain_probability_l2283_228328

theorem rain_probability (p_friday p_saturday p_sunday : ℝ) 
  (h_friday : p_friday = 0.4)
  (h_saturday : p_saturday = 0.5)
  (h_sunday : p_sunday = 0.3)
  (h_independent : True) -- Assumption of independence
  : p_friday * p_saturday * p_sunday = 0.06 := by
  sorry

end rain_probability_l2283_228328


namespace derivative_f_at_1_l2283_228342

/-- The function f(x) = (x+1)^2 -/
def f (x : ℝ) : ℝ := (x + 1)^2

/-- The theorem stating that the derivative of f(x) at x = 1 is 4 -/
theorem derivative_f_at_1 : 
  deriv f 1 = 4 := by sorry

end derivative_f_at_1_l2283_228342


namespace square_difference_of_sum_and_diff_l2283_228332

theorem square_difference_of_sum_and_diff (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_eq : x - y = 8) : 
  x^2 - y^2 = 80 := by
sorry

end square_difference_of_sum_and_diff_l2283_228332


namespace inequality_contradiction_l2283_228390

theorem inequality_contradiction (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : a / b = c / d) : 
  ¬((a + b) / (a - b) = (c + d) / (c - d)) := by
sorry

end inequality_contradiction_l2283_228390


namespace cubic_function_extrema_condition_l2283_228367

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a + 6)

/-- Theorem: If f has both a maximum and a minimum value, then a < -3 or a > 6 -/
theorem cubic_function_extrema_condition (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ x, f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 := by
  sorry

end cubic_function_extrema_condition_l2283_228367


namespace square_cards_puzzle_l2283_228345

theorem square_cards_puzzle (n : ℕ) (h : n > 0) (eq : n^2 + 36 = (n + 1)^2 + 3) :
  n^2 + 36 = 292 := by
  sorry

end square_cards_puzzle_l2283_228345


namespace iron_aluminum_weight_difference_l2283_228346

/-- The weight difference between two metal pieces -/
def weight_difference (iron_weight aluminum_weight : Float) : Float :=
  iron_weight - aluminum_weight

/-- Theorem stating the weight difference between iron and aluminum pieces -/
theorem iron_aluminum_weight_difference :
  let iron_weight : Float := 11.17
  let aluminum_weight : Float := 0.83
  weight_difference iron_weight aluminum_weight = 10.34 := by
  sorry

end iron_aluminum_weight_difference_l2283_228346


namespace marble_probability_l2283_228363

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) 
  (h_total : total = 84)
  (h_white : p_white = 1/4)
  (h_green : p_green = 2/7) :
  1 - p_white - p_green = 13/28 := by
  sorry

end marble_probability_l2283_228363


namespace complement_of_A_in_U_l2283_228381

-- Define the universal set U
def U : Set ℕ := {x : ℕ | x ≥ 2}

-- Define set A
def A : Set ℕ := {x : ℕ | x^2 ≥ 5}

-- Theorem statement
theorem complement_of_A_in_U : (U \ A) = {2} := by sorry

end complement_of_A_in_U_l2283_228381


namespace simultaneous_equations_solution_l2283_228383

theorem simultaneous_equations_solution :
  ∀ x y : ℝ,
  (3 * x^2 + x * y - 2 * y^2 = -5 ∧ x^2 + 2 * x * y + y^2 = 1) ↔
  ((x = 3/5 ∧ y = -8/5) ∨ (x = -3/5 ∧ y = 8/5)) :=
by sorry

end simultaneous_equations_solution_l2283_228383


namespace min_sum_of_digits_prime_l2283_228303

def f (n : ℕ) : ℕ := n^2 - 69*n + 2250

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem min_sum_of_digits_prime :
  ∃ (p : ℕ), is_prime p ∧
    ∀ (q : ℕ), is_prime q →
      sum_of_digits (f (p^2 + 32)) ≤ sum_of_digits (f (q^2 + 32)) ∧
      p = 3 :=
sorry

end min_sum_of_digits_prime_l2283_228303


namespace somu_age_problem_l2283_228343

/-- Represents the problem of finding when Somu was one-fifth of his father's age -/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_ago : ℕ) : 
  somu_age = 14 →
  somu_age = father_age / 3 →
  somu_age - years_ago = (father_age - years_ago) / 5 →
  years_ago = 7 := by
  sorry

end somu_age_problem_l2283_228343


namespace tortoise_age_problem_l2283_228370

theorem tortoise_age_problem (tailor_age tortoise_age tree_age : ℕ) : 
  tailor_age + tortoise_age + tree_age = 264 →
  tailor_age = 4 * (tailor_age - tortoise_age) →
  tortoise_age = 7 * (tortoise_age - tree_age) →
  tortoise_age = 77 := by
sorry

end tortoise_age_problem_l2283_228370


namespace volleyball_team_size_l2283_228333

theorem volleyball_team_size (managers : ℕ) (employees : ℕ) (teams : ℕ) :
  managers = 23 →
  employees = 7 →
  teams = 6 →
  (managers + employees) / teams = 5 := by
sorry

end volleyball_team_size_l2283_228333


namespace min_value_of_3a_plus_b_l2283_228357

theorem min_value_of_3a_plus_b (a b : ℝ) (h : 16 * a^2 + 2 * a + 8 * a * b + b^2 - 1 = 0) :
  ∃ (m : ℝ), m = 3 * a + b ∧ m ≥ -1 ∧ ∀ (x : ℝ), (∃ (a' b' : ℝ), x = 3 * a' + b' ∧ 16 * a'^2 + 2 * a' + 8 * a' * b' + b'^2 - 1 = 0) → x ≥ m :=
sorry

end min_value_of_3a_plus_b_l2283_228357


namespace parabola_translation_l2283_228396

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
    b := p.b - 2 * p.a * h
    c := p.c + p.a * h^2 - p.b * h - v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk (-1) (-2) 0
  let translated := translate (translate original 2 0) 0 3
  y = -(x * (x + 2)) → y = -(x - 1)^2 - 2 := by
  sorry

end parabola_translation_l2283_228396


namespace triangle_properties_l2283_228306

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle)
  (h1 : Real.tan t.C = (Real.sin t.A + Real.sin t.B) / (Real.cos t.A + Real.cos t.B))
  (h2 : t.c = Real.sqrt 3) :
  t.C = π / 3 ∧ 3 < t.a^2 + t.b^2 ∧ t.a^2 + t.b^2 ≤ 6 := by
  sorry

end triangle_properties_l2283_228306


namespace ceiling_minus_x_l2283_228322

theorem ceiling_minus_x (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ f : ℝ, 0 < f ∧ f < 1 ∧ ⌈x⌉ - x = 1 - f :=
by sorry

end ceiling_minus_x_l2283_228322


namespace thursday_temperature_l2283_228311

/-- Calculates the temperature on Thursday given the temperatures for the other days of the week and the average temperature. -/
def temperature_on_thursday (sunday monday tuesday wednesday friday saturday average : ℝ) : ℝ :=
  7 * average - (sunday + monday + tuesday + wednesday + friday + saturday)

/-- Theorem stating that the temperature on Thursday is 82° given the specified conditions. -/
theorem thursday_temperature :
  temperature_on_thursday 40 50 65 36 72 26 53 = 82 := by
  sorry

end thursday_temperature_l2283_228311


namespace term_position_98_l2283_228395

/-- The sequence defined by a_n = n^2 / (n^2 + 1) -/
def a (n : ℕ) : ℚ := n^2 / (n^2 + 1)

/-- The theorem stating that 0.98 occurs at position 7 in the sequence -/
theorem term_position_98 : a 7 = 98/100 := by
  sorry

end term_position_98_l2283_228395


namespace unique_distribution_l2283_228301

structure Desserts where
  coconut : Nat
  meringue : Nat
  caramel : Nat

def total_desserts (d : Desserts) : Nat :=
  d.coconut + d.meringue + d.caramel

def is_valid_distribution (d : Desserts) : Prop :=
  total_desserts d = 10 ∧
  d.coconut < d.meringue ∧
  d.meringue < d.caramel ∧
  d.caramel ≥ 6 ∧
  (d.coconut + d.meringue ≥ 3)

theorem unique_distribution :
  ∃! d : Desserts, is_valid_distribution d ∧ d.coconut = 1 ∧ d.meringue = 2 ∧ d.caramel = 7 :=
by sorry

end unique_distribution_l2283_228301


namespace line_tangent_to_circle_l2283_228319

theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - 3)^2 = 8 → y = x + 4 → 
    ∀ x' y' : ℝ, (x' - a)^2 + (y' - 3)^2 < 8 → y' ≠ x' + 4) →
  a = 3 ∨ a = -5 :=
by sorry

end line_tangent_to_circle_l2283_228319


namespace hyperbola_equation_hyperbola_final_equation_l2283_228309

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, prove that if its eccentricity is 2
    and its asymptotes are tangent to the circle (x-a)² + y² = 3/4, then a = 1 and b = √3. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (2 : ℝ) = (Real.sqrt (a^2 + b^2)) / a  -- eccentricity is 2
  → (∃ (x y : ℝ), (y = (b/a) * x ∨ y = -(b/a) * x) ∧ (x - a)^2 + y^2 = 3/4)  -- asymptotes tangent to circle
  → a = 1 ∧ b = Real.sqrt 3 :=
by sorry

/-- The equation of the hyperbola is x² - y²/3 = 1. -/
theorem hyperbola_final_equation (x y : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (2 : ℝ) = (Real.sqrt (a^2 + b^2)) / a ∧
    (∃ (x' y' : ℝ), (y' = (b/a) * x' ∨ y' = -(b/a) * x') ∧ (x' - a)^2 + y'^2 = 3/4) ∧
    x^2 / a^2 - y^2 / b^2 = 1)
  → x^2 - y^2 / 3 = 1 :=
by sorry

end hyperbola_equation_hyperbola_final_equation_l2283_228309


namespace stratified_sampling_female_count_l2283_228374

theorem stratified_sampling_female_count 
  (male_count : ℕ) 
  (female_count : ℕ) 
  (sample_size : ℕ) 
  (h1 : male_count = 810) 
  (h2 : female_count = 540) 
  (h3 : sample_size = 200) :
  (female_count * sample_size) / (male_count + female_count) = 80 := by
  sorry

end stratified_sampling_female_count_l2283_228374


namespace fifth_inequality_l2283_228324

theorem fifth_inequality (h1 : 1 / Real.sqrt 2 < 1)
  (h2 : 1 / Real.sqrt 2 + 1 / Real.sqrt 6 < Real.sqrt 2)
  (h3 : 1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 < Real.sqrt 3) :
  1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 + 1 / Real.sqrt 20 + 1 / Real.sqrt 30 < Real.sqrt 5 := by
  sorry

end fifth_inequality_l2283_228324


namespace container_weight_problem_l2283_228337

theorem container_weight_problem (x y z : ℝ) 
  (h1 : x + y = 234)
  (h2 : y + z = 241)
  (h3 : z + x = 255) :
  x + y + z = 365 := by
sorry

end container_weight_problem_l2283_228337


namespace true_proposition_l2283_228313

/-- Proposition p: For any x ∈ ℝ, 2^x > x^2 -/
def p : Prop := ∀ x : ℝ, 2^x > x^2

/-- Proposition q: "ab > 1" is a sufficient but not necessary condition for "a > 1, b > 1" -/
def q : Prop := ∀ a b : ℝ, (a * b > 1 → (a > 1 ∧ b > 1)) ∧ ¬(∀ a b : ℝ, (a > 1 ∧ b > 1) → a * b > 1)

/-- The true proposition is ¬p ∧ ¬q -/
theorem true_proposition : ¬p ∧ ¬q := by sorry

end true_proposition_l2283_228313


namespace third_anthill_population_l2283_228341

/-- Calculates the number of ants in the next anthill given the current number of ants -/
def next_anthill_population (current_ants : ℕ) : ℕ :=
  (current_ants * 4) / 5

/-- Represents the forest with three anthills -/
structure Forest where
  anthill1 : ℕ
  anthill2 : ℕ
  anthill3 : ℕ

/-- Creates a forest with three anthills, where each subsequent anthill has 20% fewer ants -/
def create_forest (initial_ants : ℕ) : Forest :=
  let anthill2 := next_anthill_population initial_ants
  let anthill3 := next_anthill_population anthill2
  { anthill1 := initial_ants, anthill2 := anthill2, anthill3 := anthill3 }

/-- Theorem stating that in a forest with 100 ants in the first anthill, 
    the third anthill will have 64 ants -/
theorem third_anthill_population : 
  (create_forest 100).anthill3 = 64 := by sorry

end third_anthill_population_l2283_228341


namespace cos_eleven_pi_thirds_l2283_228325

theorem cos_eleven_pi_thirds : Real.cos (11 * Real.pi / 3) = 1 / 2 := by
  sorry

end cos_eleven_pi_thirds_l2283_228325


namespace sum_of_absolute_coefficients_l2283_228392

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 32 := by
  sorry

end sum_of_absolute_coefficients_l2283_228392


namespace prime_equivalence_l2283_228398

theorem prime_equivalence (k : ℕ) (h : ℕ) (n : ℕ) 
  (h_odd : Odd h) 
  (h_bound : h < 2^k) 
  (n_def : n = 2^k * h + 1) : 
  Nat.Prime n ↔ ∃ a : ℕ, a^((n-1)/2) % n = n - 1 := by
  sorry

end prime_equivalence_l2283_228398


namespace sin_five_zeros_l2283_228391

theorem sin_five_zeros (f : ℝ → ℝ) (ω : ℝ) :
  ω > 0 →
  (∀ x, f x = Real.sin (ω * x)) →
  (∃! z : Finset ℝ, z.card = 5 ∧ (∀ x ∈ z, x ∈ Set.Icc 0 (3 * Real.pi) ∧ f x = 0)) →
  ω ∈ Set.Icc (4 / 3) (5 / 3) :=
sorry

end sin_five_zeros_l2283_228391


namespace marks_lost_is_one_l2283_228344

/-- Represents an examination with given parameters -/
structure Examination where
  total_questions : ℕ
  marks_per_correct : ℕ
  total_score : ℕ
  correct_answers : ℕ

/-- Calculates the marks lost per wrong answer in an examination -/
def marks_lost_per_wrong (exam : Examination) : ℚ :=
  let wrong_answers := exam.total_questions - exam.correct_answers
  let total_marks_for_correct := exam.marks_per_correct * exam.correct_answers
  let total_marks_lost := total_marks_for_correct - exam.total_score
  total_marks_lost / wrong_answers

/-- Theorem stating that for the given examination parameters, 
    the marks lost per wrong answer is 1 -/
theorem marks_lost_is_one (exam : Examination) 
  (h1 : exam.total_questions = 60)
  (h2 : exam.marks_per_correct = 4)
  (h3 : exam.total_score = 150)
  (h4 : exam.correct_answers = 42) :
  marks_lost_per_wrong exam = 1 := by
  sorry

#eval marks_lost_per_wrong { 
  total_questions := 60, 
  marks_per_correct := 4, 
  total_score := 150, 
  correct_answers := 42 
}

end marks_lost_is_one_l2283_228344


namespace fabric_needed_l2283_228314

-- Define the constants
def yards_per_dress : ℝ := 5.5
def num_dresses : ℕ := 4
def fabric_on_hand : ℝ := 7
def feet_per_yard : ℝ := 3

-- State the theorem
theorem fabric_needed : 
  ∃ (additional_fabric : ℝ), 
    additional_fabric = num_dresses * (yards_per_dress * feet_per_yard) - fabric_on_hand ∧
    additional_fabric = 59 := by
  sorry

end fabric_needed_l2283_228314


namespace quadratic_positive_range_l2283_228347

def quadratic_function (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

theorem quadratic_positive_range (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 3 → quadratic_function a x > 0) ↔ 
  ((-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3)) :=
sorry

end quadratic_positive_range_l2283_228347


namespace complex_difference_of_eighth_powers_l2283_228387

theorem complex_difference_of_eighth_powers : (2 + Complex.I) ^ 8 - (2 - Complex.I) ^ 8 = 0 := by
  sorry

end complex_difference_of_eighth_powers_l2283_228387


namespace rain_probability_l2283_228339

theorem rain_probability (p_monday p_tuesday p_neither : ℝ) 
  (h1 : p_monday = 0.7)
  (h2 : p_tuesday = 0.55)
  (h3 : p_neither = 0.35)
  (h4 : 0 ≤ p_monday ∧ p_monday ≤ 1)
  (h5 : 0 ≤ p_tuesday ∧ p_tuesday ≤ 1)
  (h6 : 0 ≤ p_neither ∧ p_neither ≤ 1) :
  p_monday + p_tuesday - (1 - p_neither) = 0.6 := by
  sorry

end rain_probability_l2283_228339


namespace sum_of_cyclic_equations_l2283_228304

theorem sum_of_cyclic_equations (x y z : ℝ) 
  (eq1 : x + y = 1) 
  (eq2 : y + z = 1) 
  (eq3 : z + x = 1) : 
  x + y + z = 3/2 := by
sorry

end sum_of_cyclic_equations_l2283_228304


namespace fourth_power_sum_l2283_228379

theorem fourth_power_sum (a b c : ℝ) 
  (sum_1 : a + b + c = 1)
  (sum_2 : a^2 + b^2 + c^2 = 2)
  (sum_3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 25/6 := by sorry

end fourth_power_sum_l2283_228379


namespace sqrt_81_equals_3_squared_l2283_228358

theorem sqrt_81_equals_3_squared : Real.sqrt 81 = 3^2 := by
  sorry

end sqrt_81_equals_3_squared_l2283_228358


namespace simplified_expression_equals_negative_three_l2283_228340

theorem simplified_expression_equals_negative_three :
  let a : ℚ := -4
  (1 / (a - 1) + 1) / (a / (a^2 - 1)) = -3 := by sorry

end simplified_expression_equals_negative_three_l2283_228340


namespace ball_picking_probabilities_l2283_228307

/-- Represents the probability of selecting ball 3 using strategy 1 -/
def P₁ : ℚ := 1/3

/-- Represents the probability of selecting ball 3 using strategy 2 -/
def P₂ : ℚ := 1/2

/-- Represents the probability of selecting ball 3 using strategy 3 -/
def P₃ : ℚ := 2/3

/-- Theorem stating the relationships between P₁, P₂, and P₃ -/
theorem ball_picking_probabilities : P₁ < P₂ ∧ P₁ < P₃ ∧ 2 * P₁ = P₃ := by
  sorry

end ball_picking_probabilities_l2283_228307


namespace batch_size_proof_l2283_228378

theorem batch_size_proof (x : ℕ) (N : ℕ) :
  (20 * (x - 1) = N) →                   -- Condition 1
  (∃ r : ℕ, r = 20) →                    -- Original rate
  ((25 * (x - 7)) = N - 80) →            -- Condition 2 (after rate increase)
  (x = 14) →                             -- Derived from solution
  N = 280 :=                             -- Conclusion to prove
by sorry

end batch_size_proof_l2283_228378


namespace c_value_l2283_228362

theorem c_value (x y : ℝ) (h : 2 * x + 5 * y = 3) :
  let c := Real.sqrt ((4 ^ (x + 1/2)) * (32 ^ y))
  c = 4 := by
sorry

end c_value_l2283_228362


namespace inequality_proof_l2283_228365

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≤ 1) : x^6 - y^6 + 2*y^3 < π/2 := by
  sorry

end inequality_proof_l2283_228365


namespace cosine_product_in_special_sequence_l2283_228359

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem cosine_product_in_special_sequence (a₁ : ℝ) :
  let a := arithmetic_sequence a₁ (2 * Real.pi / 3)
  let S := {x | ∃ n : ℕ+, x = Real.cos (a n)}
  (∃ a b : ℝ, S = {a, b}) →
  ∃ a b : ℝ, S = {a, b} ∧ a * b = -1/2 := by
  sorry

end cosine_product_in_special_sequence_l2283_228359
