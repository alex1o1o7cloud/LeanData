import Mathlib

namespace road_repaving_l2518_251882

theorem road_repaving (total_repaved : ℕ) (repaved_today : ℕ) 
  (h1 : total_repaved = 4938)
  (h2 : repaved_today = 805) :
  total_repaved - repaved_today = 4133 := by
  sorry

end road_repaving_l2518_251882


namespace quadratic_decreasing_interval_l2518_251875

/-- A quadratic function f(x) = x^2 + bx + c -/
def quadratic (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The derivative of a quadratic function -/
def quadratic_derivative (b : ℝ) (x : ℝ) : ℝ := 2*x + b

theorem quadratic_decreasing_interval (b c : ℝ) :
  (∀ x ≤ 1, quadratic_derivative b x ≤ 0) →
  (∃ x > 1, quadratic_derivative b x > 0) →
  b = -2 := by
sorry

end quadratic_decreasing_interval_l2518_251875


namespace octal_subtraction_l2518_251836

/-- Convert a base-8 number to base-10 --/
def octal_to_decimal (n : ℕ) : ℕ :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 * 64 + d2 * 8 + d3

/-- Convert a base-10 number to base-8 --/
def decimal_to_octal (n : ℕ) : ℕ :=
  let d1 := n / 64
  let d2 := (n / 8) % 8
  let d3 := n % 8
  d1 * 100 + d2 * 10 + d3

theorem octal_subtraction :
  decimal_to_octal (octal_to_decimal 526 - octal_to_decimal 321) = 205 := by
  sorry

end octal_subtraction_l2518_251836


namespace point_in_third_quadrant_l2518_251830

theorem point_in_third_quadrant : ∃ (x y : ℝ), 
  x = Real.sin (2014 * π / 180) ∧ 
  y = Real.cos (2014 * π / 180) ∧ 
  x < 0 ∧ y < 0 := by
  sorry

end point_in_third_quadrant_l2518_251830


namespace complex_equation_solution_l2518_251839

theorem complex_equation_solution (x y : ℝ) :
  (x * Complex.I + 2 = y - Complex.I) → (x - y = -3) := by
  sorry

end complex_equation_solution_l2518_251839


namespace olivias_initial_amount_l2518_251848

/-- The amount of money Olivia had in her wallet initially -/
def initial_amount : ℕ := sorry

/-- The amount Olivia spent at the supermarket -/
def supermarket_expense : ℕ := 31

/-- The amount Olivia spent at the showroom -/
def showroom_expense : ℕ := 49

/-- The amount Olivia had left after spending -/
def remaining_amount : ℕ := 26

/-- Theorem stating that Olivia's initial amount was $106 -/
theorem olivias_initial_amount : 
  initial_amount = supermarket_expense + showroom_expense + remaining_amount := by sorry

end olivias_initial_amount_l2518_251848


namespace angle_relations_l2518_251818

def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2

theorem angle_relations (α β : ℝ) 
  (h_acute_α : acute_angle α) 
  (h_acute_β : acute_angle β) 
  (h_sin_α : Real.sin α = 3/5) 
  (h_tan_diff : Real.tan (α - β) = 1/3) : 
  Real.tan β = 1/3 ∧ 
  Real.sin (2*α - β) = (13 * Real.sqrt 10) / 50 := by
sorry

end angle_relations_l2518_251818


namespace parabola_cross_section_l2518_251833

/-- Represents a cone --/
structure Cone where
  vertex_angle : ℝ

/-- Represents a cross-section of a cone --/
structure CrossSection where
  angle_with_axis : ℝ

/-- Represents the type of curve formed by a cross-section --/
inductive CurveType
  | Circle
  | Ellipse
  | Hyperbola
  | Parabola

/-- Determines the curve type of a cross-section for a given cone --/
def cross_section_curve_type (cone : Cone) (cs : CrossSection) : CurveType :=
  sorry

/-- Theorem stating that for a cone with 90° vertex angle and 45° cross-section angle, 
    the resulting curve is a parabola --/
theorem parabola_cross_section 
  (cone : Cone) 
  (cs : CrossSection) 
  (h1 : cone.vertex_angle = 90) 
  (h2 : cs.angle_with_axis = 45) : 
  cross_section_curve_type cone cs = CurveType.Parabola := by
  sorry

end parabola_cross_section_l2518_251833


namespace single_point_conic_section_l2518_251815

theorem single_point_conic_section (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + p.2^2 + 6 * p.1 - 8 * p.2 + d = 0) → d = 19 := by
  sorry

end single_point_conic_section_l2518_251815


namespace total_apples_for_bobbing_l2518_251858

/-- The number of apples in each bucket for bobbing apples. -/
def apples_per_bucket : ℕ := 9

/-- The number of buckets Mrs. Walker needs. -/
def number_of_buckets : ℕ := 7

/-- Theorem: Mrs. Walker has 63 apples for bobbing for apples. -/
theorem total_apples_for_bobbing : 
  apples_per_bucket * number_of_buckets = 63 := by
  sorry

end total_apples_for_bobbing_l2518_251858


namespace f_equals_n_plus_one_f_1993_l2518_251867

def N₀ : Set ℕ := {n : ℕ | True}

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ n, f (f n) + f n = 2 * n + 3

theorem f_equals_n_plus_one (f : ℕ → ℕ) (h : is_valid_f f) :
  ∀ n, f n = n + 1 :=
by
  sorry

-- The original question can be answered as a corollary
theorem f_1993 (f : ℕ → ℕ) (h : is_valid_f f) :
  f 1993 = 1994 :=
by
  sorry

end f_equals_n_plus_one_f_1993_l2518_251867


namespace friend_lunch_cost_l2518_251888

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 15 → difference = 5 → friend_cost = total / 2 + difference / 2 → friend_cost = 10 := by
  sorry

end friend_lunch_cost_l2518_251888


namespace assignments_for_40_points_l2518_251800

/-- Calculates the number of assignments needed for a given number of points and assignments per point -/
def assignmentsForPoints (points : ℕ) (assignmentsPerPoint : ℕ) : ℕ :=
  points * assignmentsPerPoint

/-- Calculates the total number of assignments needed for 40 homework points -/
def totalAssignments : ℕ :=
  assignmentsForPoints 7 3 +  -- First 7 points
  assignmentsForPoints 7 4 +  -- Next 7 points (8-14)
  assignmentsForPoints 7 5 +  -- Next 7 points (15-21)
  assignmentsForPoints 7 6 +  -- Next 7 points (22-28)
  assignmentsForPoints 7 7 +  -- Next 7 points (29-35)
  assignmentsForPoints 5 8    -- Last 5 points (36-40)

/-- The theorem stating that 215 assignments are needed for 40 homework points -/
theorem assignments_for_40_points : totalAssignments = 215 := by
  sorry


end assignments_for_40_points_l2518_251800


namespace x_range_for_inequality_l2518_251854

theorem x_range_for_inequality (m : ℝ) (hm : m ∈ Set.Icc 0 1) :
  {x : ℝ | m * x^2 - 2 * x - m ≥ 2} ⊆ Set.Iic (-1) := by
sorry

end x_range_for_inequality_l2518_251854


namespace chastity_money_left_l2518_251881

/-- The amount of money Chastity was left with after buying lollipops and gummies -/
def money_left (initial_amount : ℝ) (lollipop_price : ℝ) (lollipop_count : ℕ) 
                (gummy_price : ℝ) (gummy_count : ℕ) : ℝ :=
  initial_amount - (lollipop_price * lollipop_count + gummy_price * gummy_count)

/-- Theorem stating that Chastity was left with $5 after her candy purchase -/
theorem chastity_money_left : 
  money_left 15 1.5 4 2 2 = 5 := by
  sorry

end chastity_money_left_l2518_251881


namespace sqrt_three_expression_l2518_251822

theorem sqrt_three_expression : Real.sqrt 3 * (Real.sqrt 3 - 1 / Real.sqrt 3) = 2 := by
  sorry

end sqrt_three_expression_l2518_251822


namespace choose_four_from_ten_l2518_251873

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by
  sorry

end choose_four_from_ten_l2518_251873


namespace calculate_expression_l2518_251837

theorem calculate_expression : ((18^18 / 18^17)^3 * 8^3) / 2^9 = 5832 := by
  sorry

end calculate_expression_l2518_251837


namespace seating_arrangements_l2518_251895

/-- Represents the number of seats in a row -/
def total_seats : ℕ := 12

/-- Represents the number of people to be seated -/
def num_people : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Represents the number of possible arrangements of A between the other two people -/
def a_between_arrangements : ℕ := 2

/-- Represents the number of empty seats after arranging people and mandatory empty seats -/
def remaining_empty_seats : ℕ := 8

/-- Represents the number of empty seats to be chosen from remaining empty seats -/
def seats_to_choose : ℕ := 5

/-- The main theorem stating the total number of seating arrangements -/
theorem seating_arrangements :
  a_between_arrangements * choose remaining_empty_seats seats_to_choose = 112 := by
  sorry

end seating_arrangements_l2518_251895


namespace max_value_cos_sin_l2518_251860

theorem max_value_cos_sin (θ : Real) (h : 0 ≤ θ ∧ θ ≤ Real.pi / 2) :
  (Real.cos (θ / 2))^2 * (1 - Real.sin θ) ≤ 1 :=
sorry

end max_value_cos_sin_l2518_251860


namespace memory_card_cost_l2518_251847

/-- If three identical memory cards cost $45 in total, then eight of these memory cards will cost $120. -/
theorem memory_card_cost (cost_of_three : ℝ) : cost_of_three = 45 → 8 * (cost_of_three / 3) = 120 := by
  sorry

end memory_card_cost_l2518_251847


namespace mrs_hilt_total_miles_l2518_251883

/-- The total miles run by Mrs. Hilt in a week -/
def total_miles (monday wednesday friday : ℕ) : ℕ := monday + wednesday + friday

/-- Theorem: Mrs. Hilt's total miles run in the week is 12 -/
theorem mrs_hilt_total_miles : total_miles 3 2 7 = 12 := by
  sorry

end mrs_hilt_total_miles_l2518_251883


namespace existence_of_n_with_k_prime_factors_l2518_251838

theorem existence_of_n_with_k_prime_factors (k m : ℕ) (hk : k > 0) (hm : m > 0) (hm_odd : Odd m) :
  ∃ n : ℕ, n > 0 ∧ (∃ (S : Finset ℕ), S.card ≥ k ∧ ∀ p ∈ S, Prime p ∧ p ∣ (m^n + n^m)) :=
sorry

end existence_of_n_with_k_prime_factors_l2518_251838


namespace pool_filling_trips_l2518_251817

/-- The number of trips required to fill the pool -/
def trips_to_fill_pool (caleb_gallons cynthia_gallons pool_capacity : ℕ) : ℕ :=
  (pool_capacity + caleb_gallons + cynthia_gallons - 1) / (caleb_gallons + cynthia_gallons)

/-- Theorem stating that it takes 7 trips to fill the pool -/
theorem pool_filling_trips :
  trips_to_fill_pool 7 8 105 = 7 := by sorry

end pool_filling_trips_l2518_251817


namespace shaded_area_concentric_circles_l2518_251887

theorem shaded_area_concentric_circles 
  (r₁ r₂ : ℝ) 
  (h₁ : r₁ > 0) 
  (h₂ : r₂ > r₁) 
  (h₃ : r₁ / (r₂ - r₁) = 1 / 2) 
  (h₄ : r₂ = 9) : 
  π * r₂^2 - π * r₁^2 = 72 * π := by
sorry

end shaded_area_concentric_circles_l2518_251887


namespace younger_person_age_l2518_251877

/-- Given two people with an age difference of 20 years, where 15 years ago the elder was twice as old as the younger, prove that the younger person's present age is 35 years. -/
theorem younger_person_age (younger elder : ℕ) : 
  elder - younger = 20 →
  elder - 15 = 2 * (younger - 15) →
  younger = 35 := by
  sorry

end younger_person_age_l2518_251877


namespace basketball_lineup_count_l2518_251844

theorem basketball_lineup_count :
  let total_players : ℕ := 20
  let lineup_size : ℕ := 5
  let specific_role : ℕ := 1
  let interchangeable : ℕ := 4
  total_players.choose specific_role * (total_players - specific_role).choose interchangeable = 77520 :=
by sorry

end basketball_lineup_count_l2518_251844


namespace kristoff_sticker_count_l2518_251894

/-- The number of stickers Riku has -/
def riku_stickers : ℕ := 2210

/-- The ratio of Riku's stickers to Kristoff's stickers -/
def sticker_ratio : ℕ := 25

/-- The number of stickers Kristoff has -/
def kristoff_stickers : ℕ := riku_stickers / sticker_ratio

theorem kristoff_sticker_count : kristoff_stickers = 88 := by
  sorry

end kristoff_sticker_count_l2518_251894


namespace power_equation_solution_l2518_251859

theorem power_equation_solution :
  ∃ y : ℝ, ((1/8 : ℝ) * 2^36 = 4^y) → y = 16.5 := by
  sorry

end power_equation_solution_l2518_251859


namespace family_d_members_l2518_251870

/-- Represents the number of members in each family -/
structure FamilyMembers where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ

/-- The initial number of members in each family -/
def initial : FamilyMembers :=
  { a := 7
    b := 8
    c := 10
    d := 13  -- This is what we want to prove
    e := 6
    f := 10 }

/-- The number of families -/
def numFamilies : ℕ := 6

/-- The number of members who left each family -/
def membersLeft : ℕ := 1

/-- The average number of members after some left -/
def newAverage : ℕ := 8

/-- Theorem: The initial number of members in family d is 13 -/
theorem family_d_members : initial.d = 13 := by sorry

end family_d_members_l2518_251870


namespace grocery_spending_l2518_251874

theorem grocery_spending (X : ℚ) : 
  X > 0 → X - 3 - 2 - (1/3)*(X - 5) = 18 → X = 32 := by
  sorry

end grocery_spending_l2518_251874


namespace specific_hexagon_area_l2518_251876

/-- An irregular hexagon in 2D space -/
structure IrregularHexagon where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  v5 : ℝ × ℝ
  v6 : ℝ × ℝ

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Calculate the area of an irregular hexagon -/
def hexagonArea (h : IrregularHexagon) : ℝ := sorry

/-- The specific irregular hexagon from the problem -/
def specificHexagon : IrregularHexagon :=
  { v1 := (0, 0)
  , v2 := (2, 4)
  , v3 := (5, 4)
  , v4 := (7, 0)
  , v5 := (5, -4)
  , v6 := (2, -4) }

/-- Theorem: The area of the specific irregular hexagon is 32 square units -/
theorem specific_hexagon_area :
  hexagonArea specificHexagon = 32 := by sorry

end specific_hexagon_area_l2518_251876


namespace coles_return_speed_l2518_251812

/-- Calculates the average speed for the return trip given the conditions of Cole's journey -/
theorem coles_return_speed (speed_to_work : ℝ) (total_time : ℝ) (time_to_work : ℝ) : 
  speed_to_work = 60 → 
  total_time = 2 → 
  time_to_work = 1.2 → 
  (speed_to_work * time_to_work) / (total_time - time_to_work) = 90 := by
sorry

end coles_return_speed_l2518_251812


namespace chord_length_polar_curve_l2518_251865

/-- The length of the chord AB, where A is the point (3, 0) and B is the other intersection point
    of the line x = 3 with the curve ρ = 4cosθ in polar coordinates. -/
theorem chord_length_polar_curve : ∃ (A B : ℝ × ℝ),
  A = (3, 0) ∧
  B.1 = 3 ∧
  (B.1 - 2)^2 + B.2^2 = 4 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 :=
by sorry

end chord_length_polar_curve_l2518_251865


namespace polynomial_functional_equation_l2518_251802

theorem polynomial_functional_equation (p : ℝ → ℝ) (c : ℝ) :
  (∀ x, p (p x) = x * p x + c * x^2) →
  ((p = id ∧ c = 0) ∨ (∀ x, p x = -x ∧ c = -2)) :=
sorry

end polynomial_functional_equation_l2518_251802


namespace inequality_solution_range_l2518_251855

theorem inequality_solution_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) ↔ k < Real.sqrt 6 / 6 :=
sorry

end inequality_solution_range_l2518_251855


namespace three_digit_numbers_with_repeated_digits_l2518_251810

theorem three_digit_numbers_with_repeated_digits : 
  let total_three_digit_numbers := 999 - 100 + 1
  let distinct_digit_numbers := 9 * 9 * 8
  total_three_digit_numbers - distinct_digit_numbers = 252 := by
sorry

end three_digit_numbers_with_repeated_digits_l2518_251810


namespace original_group_size_l2518_251816

/-- Given a group of men working on a project, this theorem proves that the original number of men is 30, based on the given conditions. -/
theorem original_group_size (initial_days work_days : ℕ) (absent_men : ℕ) : 
  initial_days = 10 → 
  work_days = 12 → 
  absent_men = 5 → 
  ∃ (original_size : ℕ), 
    original_size * initial_days = (original_size - absent_men) * work_days ∧ 
    original_size = 30 := by
  sorry

end original_group_size_l2518_251816


namespace wheel_turns_time_l2518_251863

theorem wheel_turns_time (turns_per_two_hours : ℕ) (h : turns_per_two_hours = 1440) :
  (6 : ℝ) * (3600 : ℝ) / (turns_per_two_hours : ℝ) * 2 = 30 := by
  sorry

end wheel_turns_time_l2518_251863


namespace derivative_at_one_l2518_251868

/-- Given a function f: ℝ → ℝ satisfying f(x) = 2x * f'(1) + 1/x for all x ≠ 0,
    prove that f'(1) = 1 -/
theorem derivative_at_one (f : ℝ → ℝ) (hf : ∀ x ≠ 0, f x = 2 * x * (deriv f 1) + 1 / x) :
  deriv f 1 = 1 := by
  sorry

end derivative_at_one_l2518_251868


namespace ray_initial_cents_l2518_251845

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of cents Ray gives to Peter -/
def cents_to_peter : ℕ := 25

/-- The number of nickels Ray has left after giving away cents -/
def nickels_left : ℕ := 4

/-- The initial number of cents Ray had -/
def initial_cents : ℕ := 95

theorem ray_initial_cents :
  initial_cents = 
    cents_to_peter + 
    (2 * cents_to_peter) + 
    (nickels_left * nickel_value) :=
by sorry

end ray_initial_cents_l2518_251845


namespace min_omega_value_l2518_251884

/-- Given that ω > 0 and the graph of y = 2cos(ωx + π/5) - 1 overlaps with itself
    after shifting right by 5π/4 units, prove that the minimum value of ω is 8/5. -/
theorem min_omega_value (ω : ℝ) (h1 : ω > 0)
  (h2 : ∀ x : ℝ, 2 * Real.cos (ω * x + π / 5) - 1 = 2 * Real.cos (ω * (x + 5 * π / 4) + π / 5) - 1) :
  ω ≥ 8 / 5 ∧ ∀ ω' > 0, (∀ x : ℝ, 2 * Real.cos (ω' * x + π / 5) - 1 = 2 * Real.cos (ω' * (x + 5 * π / 4) + π / 5) - 1) → ω' ≥ ω :=
by sorry

end min_omega_value_l2518_251884


namespace least_subtraction_for_divisibility_l2518_251880

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 14 ∧ (42398 - x) % 15 = 0 ∧ ∀ y : ℕ, y < x → (42398 - y) % 15 ≠ 0 :=
by sorry

end least_subtraction_for_divisibility_l2518_251880


namespace double_age_in_two_years_l2518_251862

/-- The number of years until a man's age is twice his son's age -/
def yearsUntilDoubleAge (sonAge manAge : ℕ) : ℕ :=
  if manAge ≤ sonAge then 0
  else (manAge - sonAge)

theorem double_age_in_two_years (sonAge manAge : ℕ) 
  (h1 : manAge = sonAge + 24)
  (h2 : sonAge = 22) :
  yearsUntilDoubleAge sonAge manAge = 2 := by
sorry

end double_age_in_two_years_l2518_251862


namespace p_is_third_degree_trinomial_l2518_251899

-- Define the polynomial
def p (x y : ℝ) : ℝ := 2 * x^2 - 3 * x * y + 5 * x * y^2

-- Theorem statement
theorem p_is_third_degree_trinomial :
  (∃ (a b c : ℝ) (f g h : ℕ → ℕ → ℕ), 
    (∀ x y, p x y = a * x^(f 0 0) * y^(f 0 1) + b * x^(g 0 0) * y^(g 0 1) + c * x^(h 0 0) * y^(h 0 1)) ∧
    (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧
    (max (f 0 0 + f 0 1) (max (g 0 0 + g 0 1) (h 0 0 + h 0 1)) = 3)) :=
by sorry


end p_is_third_degree_trinomial_l2518_251899


namespace fedora_cleaning_time_l2518_251864

/-- Represents the cleaning problem of Fedora Egorovna's stove wall. -/
def CleaningProblem (total_sections : ℕ) (cleaned_sections : ℕ) (time_spent : ℕ) : Prop :=
  let cleaning_rate := time_spent / cleaned_sections
  let total_time := total_sections * cleaning_rate
  let additional_time := total_time - time_spent
  additional_time = 192

/-- Theorem stating that given the conditions of Fedora's cleaning,
    the additional time required is 192 minutes. -/
theorem fedora_cleaning_time :
  CleaningProblem 27 3 24 :=
by
  sorry

#check fedora_cleaning_time

end fedora_cleaning_time_l2518_251864


namespace inequality_proof_l2518_251897

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  Real.sqrt (a^2 + 1/a) + Real.sqrt (b^2 + 1/b) ≥ 3 := by
  sorry

end inequality_proof_l2518_251897


namespace quadratic_sum_l2518_251849

theorem quadratic_sum (a b : ℝ) : 
  ({b} : Set ℝ) = {x : ℝ | a * x^2 - 4 * x + 1 = 0} → 
  a + b = 1/4 ∨ a + b = 9/2 := by
sorry

end quadratic_sum_l2518_251849


namespace division_remainder_problem_l2518_251841

theorem division_remainder_problem (k : ℕ+) (h : ∃ b : ℕ, 80 = b * k^2 + 8) :
  ∃ q : ℕ, 140 = q * k + 2 := by
sorry

end division_remainder_problem_l2518_251841


namespace calculator_time_saved_l2518_251825

/-- Proves that using a calculator saves 60 minutes for a 20-problem assignment -/
theorem calculator_time_saved 
  (time_with_calculator : ℕ) 
  (time_without_calculator : ℕ) 
  (num_problems : ℕ) 
  (h1 : time_with_calculator = 2)
  (h2 : time_without_calculator = 5)
  (h3 : num_problems = 20) :
  (time_without_calculator - time_with_calculator) * num_problems = 60 :=
by sorry

end calculator_time_saved_l2518_251825


namespace tank_filling_l2518_251889

theorem tank_filling (tank_capacity : ℕ) (buckets_case1 buckets_case2 capacity_case1 : ℕ) :
  tank_capacity = buckets_case1 * capacity_case1 →
  tank_capacity = buckets_case2 * (tank_capacity / buckets_case2) →
  buckets_case1 = 12 →
  capacity_case1 = 81 →
  buckets_case2 = 108 →
  tank_capacity / buckets_case2 = 9 :=
by sorry

end tank_filling_l2518_251889


namespace five_sixths_of_twelve_fifths_minus_half_l2518_251861

theorem five_sixths_of_twelve_fifths_minus_half :
  (5 / 6 : ℚ) * (12 / 5 : ℚ) - (1 / 2 : ℚ) = (3 / 2 : ℚ) := by
  sorry

end five_sixths_of_twelve_fifths_minus_half_l2518_251861


namespace f_positive_implies_a_range_l2518_251807

open Real

/-- The function f(x) defined in terms of parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 + (a - 1) * x + 1

/-- Theorem stating that if f(x) > 0 for all real x, then 1 ≤ a < 5 -/
theorem f_positive_implies_a_range (a : ℝ) :
  (∀ x : ℝ, f a x > 0) → 1 ≤ a ∧ a < 5 := by
  sorry

end f_positive_implies_a_range_l2518_251807


namespace garden_area_ratio_l2518_251871

theorem garden_area_ratio :
  ∀ (L W : ℝ),
  L / W = 5 / 4 →
  L + W = 50 →
  (L * W) / (π * (W / 2)^2) = 5 / π :=
λ L W h1 h2 => by
  sorry

end garden_area_ratio_l2518_251871


namespace parabola_focus_coordinates_l2518_251823

/-- Given a parabola x² = 2py where p > 0, with a point M(4, y₀) on the parabola,
    and the distance between M and the focus F being |MF| = 5/4 * y₀,
    prove that the coordinates of the focus F are (0, 1). -/
theorem parabola_focus_coordinates (p : ℝ) (y₀ : ℝ) (h_p : p > 0) :
  x^2 = 2*p*y →
  4^2 = 2*p*y₀ →
  (4^2 + (y₀ - p/2)^2)^(1/2) = 5/4 * y₀ →
  (0, 1) = (0, p/2) := by sorry

end parabola_focus_coordinates_l2518_251823


namespace range_of_PQ_length_l2518_251852

/-- Circle C in the Cartesian coordinate system -/
def CircleC (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 2

/-- Point A is on the x-axis -/
def PointA (x : ℝ) : Prop := true

/-- AP is tangent to circle C at point P -/
def TangentAP (A P : ℝ × ℝ) : Prop := sorry

/-- AQ is tangent to circle C at point Q -/
def TangentAQ (A Q : ℝ × ℝ) : Prop := sorry

/-- The length of segment PQ -/
def LengthPQ (P Q : ℝ × ℝ) : ℝ := sorry

theorem range_of_PQ_length :
  ∀ A P Q : ℝ × ℝ,
    PointA A.1 →
    CircleC P.1 P.2 →
    CircleC Q.1 Q.2 →
    TangentAP A P →
    TangentAQ A Q →
    (2 * Real.sqrt 14 / 3 ≤ LengthPQ P Q) ∧ (LengthPQ P Q < 2 * Real.sqrt 2) :=
by sorry

end range_of_PQ_length_l2518_251852


namespace water_balloon_ratio_l2518_251814

/-- The number of water balloons each person has -/
structure WaterBalloons where
  cynthia : ℕ
  randy : ℕ
  janice : ℕ

/-- The conditions of the problem -/
def problem_conditions (wb : WaterBalloons) : Prop :=
  wb.cynthia = 12 ∧ wb.janice = 6 ∧ wb.randy = wb.janice / 2

/-- The theorem stating the ratio of Cynthia's to Randy's water balloons -/
theorem water_balloon_ratio (wb : WaterBalloons) 
  (h : problem_conditions wb) : wb.cynthia / wb.randy = 4 := by
  sorry

end water_balloon_ratio_l2518_251814


namespace magnitude_of_vector_difference_l2518_251826

def vector_a : Fin 2 → ℝ := ![2, 1]
def vector_b : Fin 2 → ℝ := ![-2, 4]

theorem magnitude_of_vector_difference :
  Real.sqrt ((vector_a 0 - vector_b 0)^2 + (vector_a 1 - vector_b 1)^2) = 5 := by
  sorry

end magnitude_of_vector_difference_l2518_251826


namespace complex_sum_pure_imaginary_l2518_251821

theorem complex_sum_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (z₁ + z₂).re = 0 → a = -3 := by
  sorry

end complex_sum_pure_imaginary_l2518_251821


namespace grasshopper_jump_distance_l2518_251806

theorem grasshopper_jump_distance 
  (frog_distance : ℕ → ℕ → ℕ) 
  (mouse_distance : ℕ → ℕ → ℕ) 
  (grasshopper_distance : ℕ → ℕ) :
  (∀ g f, frog_distance g f = g + 32) →
  (∀ m f, mouse_distance m f = f - 26) →
  mouse_distance 31 (frog_distance (grasshopper_distance 31) 31) = 31 →
  grasshopper_distance 31 = 25 := by
sorry

end grasshopper_jump_distance_l2518_251806


namespace parallel_vectors_x_value_l2518_251843

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ → ℝ × ℝ := λ x ↦ (-2, x)
  ∀ x : ℝ, are_parallel a (b x) → x = 4 := by
  sorry

end parallel_vectors_x_value_l2518_251843


namespace biased_coin_heads_probability_l2518_251853

/-- The probability of getting heads on a single flip of a biased coin -/
theorem biased_coin_heads_probability (p : ℚ) (h : p = 3/4) : 1 - p = 1/4 := by
  sorry

end biased_coin_heads_probability_l2518_251853


namespace triangle_base_length_l2518_251804

theorem triangle_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 10) 
  (h2 : height = 5) : 
  area = (height * 4) / 2 := by
  sorry

end triangle_base_length_l2518_251804


namespace reciprocal_minus_one_l2518_251872

theorem reciprocal_minus_one (x : ℝ) : (1 / x = -1) → |-x - 1| = 0 := by
  sorry

end reciprocal_minus_one_l2518_251872


namespace smallest_valid_seating_eighteen_is_valid_smallest_seating_is_eighteen_l2518_251840

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  totalChairs : Nat
  seatedPeople : Nat

/-- Checks if a seating arrangement is valid (any new person must sit next to someone). -/
def isValidSeating (table : CircularTable) : Prop :=
  table.seatedPeople > 0 ∧ 
  table.totalChairs ≥ table.seatedPeople ∧
  table.totalChairs % table.seatedPeople = 0 ∧
  table.totalChairs / table.seatedPeople ≤ 4

/-- The theorem stating the smallest valid number of seated people for a 72-chair table. -/
theorem smallest_valid_seating :
  ∀ (table : CircularTable),
    table.totalChairs = 72 →
    isValidSeating table →
    table.seatedPeople ≥ 18 :=
by
  sorry

/-- The theorem stating that 18 is a valid seating arrangement for a 72-chair table. -/
theorem eighteen_is_valid :
  isValidSeating { totalChairs := 72, seatedPeople := 18 } :=
by
  sorry

/-- The main theorem combining the above results to prove 18 is the smallest valid seating. -/
theorem smallest_seating_is_eighteen :
  ∃ (table : CircularTable),
    table.totalChairs = 72 ∧
    table.seatedPeople = 18 ∧
    isValidSeating table ∧
    ∀ (otherTable : CircularTable),
      otherTable.totalChairs = 72 →
      isValidSeating otherTable →
      otherTable.seatedPeople ≥ table.seatedPeople :=
by
  sorry

end smallest_valid_seating_eighteen_is_valid_smallest_seating_is_eighteen_l2518_251840


namespace tan_sum_of_roots_l2518_251811

theorem tan_sum_of_roots (α β : ℝ) : 
  (∃ x y : ℝ, x^2 - 3 * Real.sqrt 3 * x + 4 = 0 ∧ 
              y^2 - 3 * Real.sqrt 3 * y + 4 = 0 ∧ 
              x = Real.tan α ∧ 
              y = Real.tan β) → 
  Real.tan (α + β) = -Real.sqrt 3 := by
sorry

end tan_sum_of_roots_l2518_251811


namespace padic_square_root_solutions_l2518_251893

/-- The number of solutions to x^2 = a in p-adic numbers is either 0 or 2 -/
theorem padic_square_root_solutions (p : ℕ) [Fact (Nat.Prime p)] (a : ℚ_[p]) :
  (∃ x y : ℚ_[p], x ^ 2 = a ∧ y ^ 2 = a ∧ x ≠ y) ∨ (∀ x : ℚ_[p], x ^ 2 ≠ a) :=
sorry

end padic_square_root_solutions_l2518_251893


namespace C_equals_46_l2518_251892

/-- Custom operation ⊕ -/
def circplus (a b : ℕ) : ℕ := a * b + 10

/-- Definition of C using the custom operation -/
def C : ℕ := circplus (circplus 1 2) 3

/-- Theorem stating that C equals 46 -/
theorem C_equals_46 : C = 46 := by
  sorry

end C_equals_46_l2518_251892


namespace frieda_prob_reach_edge_l2518_251898

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (row : Fin 4)
  (col : Fin 4)

/-- Defines the center position -/
def center : Position := ⟨1, 1⟩

/-- Checks if a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.row = 0 || p.row = 3 || p.col = 0 || p.col = 3

/-- Defines the possible moves -/
inductive Move
  | up
  | down
  | left
  | right

/-- Applies a move to a position -/
def applyMove (p : Position) (m : Move) : Position :=
  match m with
  | Move.up    => ⟨(p.row + 1) % 4, p.col⟩
  | Move.down  => ⟨(p.row - 1 + 4) % 4, p.col⟩
  | Move.left  => ⟨p.row, (p.col - 1 + 4) % 4⟩
  | Move.right => ⟨p.row, (p.col + 1) % 4⟩

/-- Calculates the probability of reaching an edge within n hops -/
def probReachEdge (n : Nat) : ℚ :=
  sorry

theorem frieda_prob_reach_edge :
  probReachEdge 3 = 5/8 :=
sorry

end frieda_prob_reach_edge_l2518_251898


namespace right_triangle_cos_c_l2518_251827

theorem right_triangle_cos_c (A B C : ℝ) (h1 : A + B + C = Real.pi) 
  (h2 : A = Real.pi / 2) (h3 : Real.sin B = 3 / 5) : Real.cos C = 3 / 5 := by
  sorry

end right_triangle_cos_c_l2518_251827


namespace largest_of_five_consecutive_sum_90_l2518_251803

theorem largest_of_five_consecutive_sum_90 :
  ∀ n : ℕ, (n + (n+1) + (n+2) + (n+3) + (n+4) = 90) → (n+4 = 20) :=
by
  sorry

end largest_of_five_consecutive_sum_90_l2518_251803


namespace karens_order_cost_l2518_251801

/-- The cost of Karen's fast-food order -/
def fast_food_order_cost (burger_price sandwich_price smoothie_price : ℕ) 
  (burger_quantity sandwich_quantity smoothie_quantity : ℕ) : ℕ :=
  burger_price * burger_quantity + 
  sandwich_price * sandwich_quantity + 
  smoothie_price * smoothie_quantity

/-- Theorem stating that Karen's fast-food order costs $17 -/
theorem karens_order_cost : 
  fast_food_order_cost 5 4 4 1 1 2 = 17 := by
  sorry

end karens_order_cost_l2518_251801


namespace amount_ratio_l2518_251846

theorem amount_ratio (total : ℕ) (r_amount : ℕ) : 
  total = 7000 →
  r_amount = 2800 →
  (r_amount : ℚ) / ((total - r_amount) : ℚ) = 2 / 3 := by
  sorry

end amount_ratio_l2518_251846


namespace f_properties_f_50_l2518_251831

/-- A cubic polynomial function satisfying specific conditions -/
def f (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the properties of the function f -/
theorem f_properties :
  f 0 = 1 ∧
  f 1 = 5 ∧
  f 2 = 13 ∧
  f 3 = 25 :=
sorry

/-- Theorem proving the value of f(50) -/
theorem f_50 : f 50 = 62676 :=
sorry

end f_properties_f_50_l2518_251831


namespace april_days_l2518_251805

/-- Proves the number of days in April based on Hannah's strawberry harvesting scenario -/
theorem april_days (daily_harvest : ℕ) (given_away : ℕ) (stolen : ℕ) (final_count : ℕ) :
  daily_harvest = 5 →
  given_away = 20 →
  stolen = 30 →
  final_count = 100 →
  (final_count + given_away + stolen) / daily_harvest = 30 := by
  sorry

#check april_days

end april_days_l2518_251805


namespace remainder_3n_div_7_l2518_251885

theorem remainder_3n_div_7 (n : Int) (h : n % 7 = 1) : (3 * n) % 7 = 3 := by
  sorry

end remainder_3n_div_7_l2518_251885


namespace factors_of_M_l2518_251879

/-- The number of natural-number factors of M, where M = 2^4 · 3^3 · 5^2 · 7^1 -/
def number_of_factors (M : ℕ) : ℕ :=
  (4 + 1) * (3 + 1) * (2 + 1) * (1 + 1)

/-- Theorem stating that the number of natural-number factors of M is 120 -/
theorem factors_of_M :
  let M : ℕ := 2^4 * 3^3 * 5^2 * 7^1
  number_of_factors M = 120 := by
  sorry

end factors_of_M_l2518_251879


namespace triangle_area_is_13_5_l2518_251808

/-- The area of a triangular region bounded by the two coordinate axes and the line 3x + y = 9 -/
def triangleArea : ℝ := 13.5

/-- The equation of the line bounding the triangular region -/
def lineEquation (x y : ℝ) : Prop := 3 * x + y = 9

/-- Theorem stating that the area of the triangular region is 13.5 square units -/
theorem triangle_area_is_13_5 :
  triangleArea = 13.5 := by sorry

end triangle_area_is_13_5_l2518_251808


namespace imaginary_part_of_z_l2518_251850

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) :
  z.im = -2 := by sorry

end imaginary_part_of_z_l2518_251850


namespace restaurant_bill_calculation_l2518_251834

theorem restaurant_bill_calculation (appetizer_cost : ℝ) (entree_cost : ℝ) (num_entrees : ℕ) (tip_percentage : ℝ) : 
  appetizer_cost = 10 ∧ 
  entree_cost = 20 ∧ 
  num_entrees = 4 ∧ 
  tip_percentage = 0.2 → 
  appetizer_cost + (entree_cost * num_entrees) + (appetizer_cost + entree_cost * num_entrees) * tip_percentage = 108 := by
  sorry

#check restaurant_bill_calculation

end restaurant_bill_calculation_l2518_251834


namespace sample_size_is_192_l2518_251824

/-- Represents the total population in the school survey --/
def total_population : ℕ := 2400

/-- Represents the number of female students in the school --/
def female_students : ℕ := 1000

/-- Represents the number of female students in the sample --/
def female_sample : ℕ := 80

/-- Calculates the sample size based on the given information --/
def sample_size : ℕ := (total_population * female_sample) / female_students

/-- Theorem stating that the sample size is 192 --/
theorem sample_size_is_192 : sample_size = 192 := by
  sorry

end sample_size_is_192_l2518_251824


namespace total_bathing_suits_l2518_251813

theorem total_bathing_suits (men_suits women_suits : ℕ) 
  (h1 : men_suits = 14797) 
  (h2 : women_suits = 4969) : 
  men_suits + women_suits = 19766 := by
  sorry

end total_bathing_suits_l2518_251813


namespace contractor_engagement_l2518_251869

/-- Contractor engagement problem -/
theorem contractor_engagement
  (daily_wage : ℝ)
  (daily_fine : ℝ)
  (total_payment : ℝ)
  (absent_days : ℕ)
  (h1 : daily_wage = 25)
  (h2 : daily_fine = 7.5)
  (h3 : total_payment = 425)
  (h4 : absent_days = 10) :
  ∃ (worked_days : ℕ) (total_days : ℕ),
    worked_days * daily_wage - absent_days * daily_fine = total_payment ∧
    total_days = worked_days + absent_days ∧
    total_days = 30 := by
  sorry

end contractor_engagement_l2518_251869


namespace students_with_two_skills_l2518_251866

theorem students_with_two_skills (total : ℕ) (cant_paint cant_write cant_music : ℕ) : 
  total = 150 →
  cant_paint = 75 →
  cant_write = 90 →
  cant_music = 45 →
  ∃ (two_skills : ℕ), two_skills = 90 ∧ 
    two_skills = (total - cant_paint) + (total - cant_write) + (total - cant_music) - total :=
by sorry

end students_with_two_skills_l2518_251866


namespace polynomial_divisibility_l2518_251820

/-- A polynomial is exactly divisible by (x-1)^3 if and only if its coefficients satisfy specific conditions -/
theorem polynomial_divisibility (a b c : ℤ) : 
  (∃ q : Polynomial ℤ, x^4 + a*x^2 + b*x + c = (x - 1)^3 * q) ↔ 
  (a = -6 ∧ b = 8 ∧ c = -3) :=
sorry


end polynomial_divisibility_l2518_251820


namespace sufficient_not_necessary_condition_l2518_251856

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, x > a → x > 2) ∧ (∃ x, x > 2 ∧ x ≤ a) → a > 2 := by
  sorry

end sufficient_not_necessary_condition_l2518_251856


namespace sphere_to_hemisphere_volume_ratio_l2518_251828

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_to_hemisphere_volume_ratio 
  (r : ℝ) -- radius of the sphere
  (k : ℝ) -- material density coefficient of the hemisphere
  (h : k = 2/3) -- given condition for k
  : (4/3 * Real.pi * r^3) / (k * 1/2 * 4/3 * Real.pi * (3*r)^3) = 2/27 := by
  sorry

end sphere_to_hemisphere_volume_ratio_l2518_251828


namespace cubic_function_properties_l2518_251842

/-- The cubic function f(x) = x³ + ax² + bx + c -/
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_properties :
  ∃ (a b c : ℝ),
    (∀ x, f' a b x = 0 → x = -1 ∨ x = 3) ∧
    (f a b c (-1) = 7) ∧
    (∀ x, f a b c x ≤ 7) ∧
    (∀ x, f a b c x ≥ f a b c 3) ∧
    (a = -3) ∧
    (b = -9) ∧
    (c = 2) ∧
    (f a b c 3 = -25) := by
  sorry

end cubic_function_properties_l2518_251842


namespace average_speed_calculation_l2518_251878

/-- Given a journey with a distance and time, calculate the average speed -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 210) (h2 : time = 35/6) :
  distance / time = 36 := by
  sorry

end average_speed_calculation_l2518_251878


namespace bees_flew_in_l2518_251857

/-- Given an initial number of bees in a hive and a final total number of bees after more fly in,
    this theorem proves that the number of bees that flew in is equal to the difference between
    the final total and the initial number. -/
theorem bees_flew_in (initial_bees final_bees : ℕ) : 
  initial_bees = 16 → final_bees = 24 → final_bees - initial_bees = 8 := by
  sorry

end bees_flew_in_l2518_251857


namespace range_of_product_l2518_251851

theorem range_of_product (x y z w : ℝ) 
  (sum_zero : x + y + z + w = 0)
  (sum_seventh_power_zero : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end range_of_product_l2518_251851


namespace key_arrangement_count_l2518_251809

/-- The number of ways to arrange n distinct objects in a circular permutation -/
def circularPermutations (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of boxes -/
def numBoxes : ℕ := 6

theorem key_arrangement_count :
  circularPermutations numBoxes = 120 :=
by sorry

end key_arrangement_count_l2518_251809


namespace point_symmetry_l2518_251832

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry about x-axis
def symmetricAboutXAxis (p1 p2 : Point2D) : Prop :=
  p1.x = p2.x ∧ p1.y = -p2.y

-- Define symmetry about y-axis
def symmetricAboutYAxis (p1 p2 : Point2D) : Prop :=
  p1.x = -p2.x ∧ p1.y = p2.y

theorem point_symmetry (M P N : Point2D) :
  symmetricAboutXAxis M P →
  symmetricAboutYAxis N M →
  N = Point2D.mk 1 2 →
  P = Point2D.mk (-1) (-2) := by
  sorry

end point_symmetry_l2518_251832


namespace group_size_l2518_251886

/-- The number of members in the group -/
def n : ℕ := sorry

/-- The total collection in paise -/
def total_paise : ℕ := 5776

/-- Each member contributes as many paise as there are members -/
axiom member_contribution : n = total_paise / n

theorem group_size : n = 76 := by sorry

end group_size_l2518_251886


namespace fraction_simplification_l2518_251896

theorem fraction_simplification :
  (1 : ℚ) / 330 + 19 / 30 = 7 / 11 := by
  sorry

end fraction_simplification_l2518_251896


namespace puzzles_sum_is_five_l2518_251829

def alphabet_value (n : ℕ) : ℤ :=
  match n % 8 with
  | 0 => 0
  | 1 => -1
  | 2 => 2
  | 3 => -1
  | 4 => 0
  | 5 => 1
  | 6 => -2
  | 7 => 1
  | _ => 0 -- This case should never occur, but Lean requires it for completeness

def letter_position (c : Char) : ℕ :=
  match c with
  | 'p' => 16
  | 'u' => 21
  | 'z' => 26
  | 'l' => 12
  | 'e' => 5
  | 's' => 19
  | _ => 0 -- Default case for other characters

theorem puzzles_sum_is_five :
  (alphabet_value (letter_position 'p') +
   alphabet_value (letter_position 'u') +
   alphabet_value (letter_position 'z') +
   alphabet_value (letter_position 'z') +
   alphabet_value (letter_position 'l') +
   alphabet_value (letter_position 'e') +
   alphabet_value (letter_position 's')) = 5 := by
  sorry

end puzzles_sum_is_five_l2518_251829


namespace events_mutually_exclusive_not_complementary_l2518_251891

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define the set of cards
inductive Card : Type
| Red : Card
| Blue : Card
| White : Card

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "Person A receives the white card"
def event_A_white (d : Distribution) : Prop := d Person.A = Card.White

-- Define the event "Person B receives the white card"
def event_B_white (d : Distribution) : Prop := d Person.B = Card.White

-- State the theorem
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(event_A_white d ∧ event_B_white d)) ∧
  (∃ d : Distribution, ¬event_A_white d ∧ ¬event_B_white d) :=
sorry

end events_mutually_exclusive_not_complementary_l2518_251891


namespace curve_is_ellipse_with_foci_on_y_axis_l2518_251890

-- Define the angle α in radians
variable (α : Real)

-- Define the condition 0° < α < 90°
axiom alpha_range : 0 < α ∧ α < Real.pi / 2

-- Define the equation of the curve
def curve_equation (x y : Real) : Prop :=
  x^2 + y^2 * Real.cos α = 1

-- State the theorem
theorem curve_is_ellipse_with_foci_on_y_axis :
  ∃ (a b : Real), a > b ∧ b > 0 ∧
  ∀ (x y : Real), curve_equation α x y ↔ (x^2 / b^2) + (y^2 / a^2) = 1 :=
sorry

end curve_is_ellipse_with_foci_on_y_axis_l2518_251890


namespace derricks_yard_length_l2518_251835

theorem derricks_yard_length (derrick_length alex_length brianne_length : ℝ) : 
  alex_length = derrick_length / 2 →
  brianne_length = 6 * alex_length →
  brianne_length = 30 →
  derrick_length = 10 := by
  sorry

end derricks_yard_length_l2518_251835


namespace parabola_segment_length_l2518_251819

/-- The length of a segment AB on a parabola y = 4x² -/
theorem parabola_segment_length :
  ∀ (x₁ x₂ y₁ y₂ : ℝ),
  y₁ = 4 * x₁^2 →
  y₂ = 4 * x₂^2 →
  ∃ (k : ℝ),
  y₁ = k * x₁ + 1/16 →
  y₂ = k * x₂ + 1/16 →
  y₁ + y₂ = 2 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (17/8)^2 :=
by
  sorry


end parabola_segment_length_l2518_251819
