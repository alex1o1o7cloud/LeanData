import Mathlib

namespace frank_first_half_correct_l3378_337848

/-- Represents the trivia game scenario -/
structure TriviaGame where
  points_per_question : ℕ
  final_score : ℕ
  second_half_correct : ℕ

/-- Calculates the number of questions answered correctly in the first half -/
def first_half_correct (game : TriviaGame) : ℕ :=
  (game.final_score - game.second_half_correct * game.points_per_question) / game.points_per_question

/-- Theorem stating that Frank answered 3 questions correctly in the first half -/
theorem frank_first_half_correct :
  let game : TriviaGame := {
    points_per_question := 3,
    final_score := 15,
    second_half_correct := 2
  }
  first_half_correct game = 3 := by
  sorry

end frank_first_half_correct_l3378_337848


namespace complex_equation_solution_l3378_337896

theorem complex_equation_solution (x : ℝ) : 
  (Complex.I * (x + Complex.I) : ℂ) = -1 + 2 * Complex.I → x = 2 := by
sorry

end complex_equation_solution_l3378_337896


namespace error_clock_correct_fraction_l3378_337810

/-- Represents a 12-hour digital clock with display errors -/
structure ErrorClock where
  /-- The clock displays '1' as '9' -/
  one_as_nine : Bool
  /-- The clock displays '2' as '5' -/
  two_as_five : Bool

/-- Calculates the fraction of the day when the clock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : ℚ :=
  sorry

/-- Theorem stating that for a clock with both display errors, 
    the fraction of correct time is 49/144 -/
theorem error_clock_correct_fraction :
  ∀ (clock : ErrorClock), 
  clock.one_as_nine ∧ clock.two_as_five → 
  correct_time_fraction clock = 49 / 144 :=
sorry

end error_clock_correct_fraction_l3378_337810


namespace parabola_directrix_l3378_337800

/-- Given a fixed point A(2,1) and a parabola y^2 = 2px (p > 0) whose focus lies on the perpendicular 
    bisector of OA, prove that the directrix of the parabola has the equation x = -5/4 -/
theorem parabola_directrix (p : ℝ) (h1 : p > 0) : 
  let A : ℝ × ℝ := (2, 1)
  let O : ℝ × ℝ := (0, 0)
  let focus : ℝ × ℝ := (p/2, 0)
  let perp_bisector (x y : ℝ) := 4*x + 2*y - 5 = 0
  let parabola (x y : ℝ) := y^2 = 2*p*x
  let directrix (x : ℝ) := x = -5/4
  (perp_bisector (focus.1) (focus.2)) →
  (∀ x y, parabola x y → (x = -p/2 ↔ directrix x)) :=
by sorry

end parabola_directrix_l3378_337800


namespace egg_ratio_is_two_to_one_l3378_337823

def egg_laying_problem (day1 day2 day3 day4 total : ℕ) : Prop :=
  day1 = 50 ∧
  day2 = 2 * day1 ∧
  day3 = day2 + 20 ∧
  total = 810 ∧
  day4 = total - (day1 + day2 + day3)

theorem egg_ratio_is_two_to_one :
  ∀ day1 day2 day3 day4 total : ℕ,
    egg_laying_problem day1 day2 day3 day4 total →
    day4 * (day1 + day2 + day3) = 2 * (day1 + day2 + day3) * (day1 + day2 + day3) :=
by
  sorry

end egg_ratio_is_two_to_one_l3378_337823


namespace cubic_intersection_line_l3378_337844

theorem cubic_intersection_line (a b c M : ℝ) : 
  a < b ∧ b < c ∧ 
  2 * (b - a) = c - b ∧
  a^3 - 84*a = M ∧
  b^3 - 84*b = M ∧
  c^3 - 84*c = M →
  M = 160 := by
sorry

end cubic_intersection_line_l3378_337844


namespace todd_spending_l3378_337808

/-- The amount Todd spent on the candy bar in cents -/
def candy_cost : ℕ := 14

/-- The amount Todd spent on the box of cookies in cents -/
def cookies_cost : ℕ := 39

/-- The total amount Todd spent in cents -/
def total_spent : ℕ := candy_cost + cookies_cost

theorem todd_spending :
  total_spent = 53 := by sorry

end todd_spending_l3378_337808


namespace cone_volume_proof_l3378_337850

noncomputable def cone_volume (slant_height : ℝ) (lateral_surface_is_semicircle : Prop) : ℝ :=
  (Real.sqrt 3 / 3) * Real.pi

theorem cone_volume_proof (slant_height : ℝ) (lateral_surface_is_semicircle : Prop) 
  (h1 : slant_height = 2)
  (h2 : lateral_surface_is_semicircle) :
  cone_volume slant_height lateral_surface_is_semicircle = (Real.sqrt 3 / 3) * Real.pi :=
by
  sorry

end cone_volume_proof_l3378_337850


namespace division_problem_l3378_337899

theorem division_problem : (107.8 : ℝ) / 11 = 9.8 := by sorry

end division_problem_l3378_337899


namespace power_greater_than_square_l3378_337846

theorem power_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end power_greater_than_square_l3378_337846


namespace jim_purchase_cost_l3378_337826

/-- The cost of a lamp in dollars -/
def lamp_cost : ℝ := 7

/-- The cost difference between a lamp and a bulb in dollars -/
def cost_difference : ℝ := 4

/-- The number of lamps bought -/
def num_lamps : ℕ := 2

/-- The number of bulbs bought -/
def num_bulbs : ℕ := 6

/-- The total cost of Jim's purchase -/
def total_cost : ℝ := num_lamps * lamp_cost + num_bulbs * (lamp_cost - cost_difference)

theorem jim_purchase_cost :
  total_cost = 32 := by sorry

end jim_purchase_cost_l3378_337826


namespace linear_equation_solution_l3378_337887

theorem linear_equation_solution : ∀ (x y : ℝ), x = 3 ∧ y = -2 → 2 * x + 3 * y = 0 := by
  sorry

end linear_equation_solution_l3378_337887


namespace b_investment_is_60000_l3378_337861

/-- Represents the investment and profit sharing structure of a business partnership --/
structure BusinessPartnership where
  total_profit : ℝ
  a_investment : ℝ
  b_investment : ℝ
  a_management_share : ℝ
  a_total_share : ℝ

/-- Theorem stating that given the conditions of the business partnership,
    B's investment is 60,000 --/
theorem b_investment_is_60000 (bp : BusinessPartnership)
  (h1 : bp.total_profit = 8800)
  (h2 : bp.a_investment = 50000)
  (h3 : bp.a_management_share = 0.125 * bp.total_profit)
  (h4 : bp.a_total_share = 4600)
  (h5 : bp.a_total_share = bp.a_management_share +
        (bp.total_profit - bp.a_management_share) * (bp.a_investment / (bp.a_investment + bp.b_investment)))
  : bp.b_investment = 60000 := by
  sorry


end b_investment_is_60000_l3378_337861


namespace class_average_score_l3378_337855

theorem class_average_score (total_students : ℕ) (present_students : ℕ) (initial_average : ℚ) (makeup_score : ℚ) :
  total_students = 40 →
  present_students = 38 →
  initial_average = 92 →
  makeup_score = 100 →
  ((initial_average * present_students + makeup_score * (total_students - present_students)) / total_students) = 92.4 := by
  sorry

end class_average_score_l3378_337855


namespace quadratic_coefficient_unique_l3378_337836

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-value of a quadratic function at a given x -/
def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

/-- The x-coordinate of the vertex of a quadratic function -/
def QuadraticFunction.vertexX (f : QuadraticFunction) : ℚ :=
  -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
def QuadraticFunction.vertexY (f : QuadraticFunction) : ℚ :=
  f.evaluate (f.vertexX)

theorem quadratic_coefficient_unique (f : QuadraticFunction) :
    f.vertexX = 2 ∧ f.vertexY = -3 ∧ f.evaluate 1 = -2 → f.a = 1 := by
  sorry

end quadratic_coefficient_unique_l3378_337836


namespace nine_pointed_star_angle_sum_l3378_337830

/-- A 9-pointed star is formed by connecting every fourth point of 9 evenly spaced points on a circle. -/
structure NinePointedStar where
  /-- The number of points on the circle -/
  num_points : ℕ
  /-- The number of points to skip when forming the star -/
  skip_points : ℕ
  /-- The number of points is 9 -/
  points_eq_nine : num_points = 9
  /-- We skip every 3 points (connect every 4th) -/
  skip_three : skip_points = 3

/-- The sum of the angles at the tips of a 9-pointed star is 540 degrees -/
theorem nine_pointed_star_angle_sum (star : NinePointedStar) : 
  (star.num_points : ℝ) * (360 / (2 * star.num_points : ℝ) * star.skip_points) = 540 := by
  sorry

end nine_pointed_star_angle_sum_l3378_337830


namespace collect_all_blocks_time_l3378_337821

/-- Represents the block collection problem --/
structure BlockCollection where
  totalBlocks : ℕ := 50
  dadPuts : ℕ := 5
  miaRemoves : ℕ := 3
  brotherRemoves : ℕ := 1
  cycleTime : ℕ := 30  -- in seconds

/-- Calculates the time in minutes to collect all blocks --/
def timeToCollectAll (bc : BlockCollection) : ℕ :=
  let netBlocksPerCycle := bc.dadPuts - (bc.miaRemoves + bc.brotherRemoves)
  let cyclesToReachAlmostAll := (bc.totalBlocks - bc.dadPuts) / netBlocksPerCycle
  let totalSeconds := (cyclesToReachAlmostAll + 1) * bc.cycleTime
  totalSeconds / 60

/-- Theorem stating that the time to collect all blocks is 23 minutes --/
theorem collect_all_blocks_time (bc : BlockCollection) :
  timeToCollectAll bc = 23 := by
  sorry

end collect_all_blocks_time_l3378_337821


namespace extended_parallelepiped_volume_2_5_6_l3378_337870

/-- The volume of points within or exactly one unit from a rectangular parallelepiped -/
def extended_parallelepiped_volume (length width height : ℝ) : ℝ :=
  (length + 2) * (width + 2) * (height + 2) - length * width * height

/-- The volume of the set of points within or exactly one unit from a 2x5x6 parallelepiped -/
theorem extended_parallelepiped_volume_2_5_6 :
  extended_parallelepiped_volume 2 5 6 = (1008 + 44 * Real.pi) / 3 := by
  sorry

end extended_parallelepiped_volume_2_5_6_l3378_337870


namespace tax_revenue_change_l3378_337839

theorem tax_revenue_change (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let new_tax := 0.86 * T
  let new_consumption := 1.15 * C
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  (new_revenue / original_revenue - 1) * 100 = -1.1 := by
sorry

end tax_revenue_change_l3378_337839


namespace new_supervisor_salary_l3378_337890

/-- Proves that the new supervisor's salary must be $870 to maintain the same average salary --/
theorem new_supervisor_salary
  (num_workers : ℕ)
  (num_total : ℕ)
  (initial_average : ℚ)
  (old_supervisor_salary : ℚ)
  (new_average : ℚ)
  (h_num_workers : num_workers = 8)
  (h_num_total : num_total = num_workers + 1)
  (h_initial_average : initial_average = 430)
  (h_old_supervisor_salary : old_supervisor_salary = 870)
  (h_new_average : new_average = initial_average)
  : ∃ (new_supervisor_salary : ℚ),
    new_supervisor_salary = 870 ∧
    (num_workers : ℚ) * initial_average + old_supervisor_salary = num_total * initial_average ∧
    (num_workers : ℚ) * new_average + new_supervisor_salary = num_total * new_average :=
by
  sorry


end new_supervisor_salary_l3378_337890


namespace parabola_translation_correct_l3378_337894

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a 2D translation -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The original parabola y = x^2 -/
def originalParabola : Parabola := { a := 1, b := 0, c := 0 }

/-- The translation of 1 unit right and 2 units up -/
def givenTranslation : Translation := { dx := 1, dy := 2 }

/-- Function to apply a translation to a parabola -/
def applyTranslation (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a
    b := -2 * p.a * t.dx + p.b
    c := p.a * t.dx^2 - p.b * t.dx + p.c + t.dy }

theorem parabola_translation_correct :
  applyTranslation originalParabola givenTranslation = { a := 1, b := -2, c := 3 } := by
  sorry

end parabola_translation_correct_l3378_337894


namespace overlap_length_l3378_337847

theorem overlap_length (L D : ℝ) (n : ℕ) (h1 : L = 98) (h2 : D = 83) (h3 : n = 6) :
  ∃ x : ℝ, x = (L - D) / n ∧ x = 2.5 := by
  sorry

end overlap_length_l3378_337847


namespace square_of_1005_l3378_337807

theorem square_of_1005 : (1005 : ℕ)^2 = 1010025 := by sorry

end square_of_1005_l3378_337807


namespace not_p_sufficient_not_necessary_for_not_q_l3378_337851

-- Define the conditions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x : ℝ) : Prop := x > 2

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, ¬(p x) → ¬(q x)) ∧ 
  (∃ x : ℝ, ¬(q x) ∧ p x) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l3378_337851


namespace cistern_wet_surface_area_l3378_337895

/-- Calculate the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  let bottom_area := length * width
  let long_sides_area := 2 * length * depth
  let short_sides_area := 2 * width * depth
  bottom_area + long_sides_area + short_sides_area

/-- Theorem stating that the total wet surface area of the given cistern is 83 square meters -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 7 4 1.25 = 83 := by
  sorry

end cistern_wet_surface_area_l3378_337895


namespace parabola_chord_through_focus_l3378_337845

/-- Given a parabola y² = 2px with p > 0, if a chord AB passes through the focus F
    such that |AF| = 2 and |BF| = 3, then p = 12/5 -/
theorem parabola_chord_through_focus (p : ℝ) (A B F : ℝ × ℝ) :
  p > 0 →
  (∀ x y, y^2 = 2*p*x) →
  F.1 = p/2 ∧ F.2 = 0 →
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = 4 →
  (B.1 - F.1)^2 + (B.2 - F.2)^2 = 9 →
  p = 12/5 := by
sorry

end parabola_chord_through_focus_l3378_337845


namespace chess_tournament_participants_l3378_337849

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 105 → n = 15 := by sorry

end chess_tournament_participants_l3378_337849


namespace largest_consecutive_sum_30_l3378_337863

def consecutive_sum (start : ℕ) (count : ℕ) : ℕ :=
  count * start + count * (count - 1) / 2

theorem largest_consecutive_sum_30 :
  (∃ (n : ℕ), n > 0 ∧ ∃ (start : ℕ), start > 0 ∧ consecutive_sum start n = 30) ∧
  (∀ (m : ℕ), m > 5 → ¬∃ (start : ℕ), start > 0 ∧ consecutive_sum start m = 30) :=
sorry

end largest_consecutive_sum_30_l3378_337863


namespace square_equals_1369_l3378_337815

theorem square_equals_1369 (x : ℤ) (h : x^2 = 1369) : (x + 1) * (x - 1) = 1368 := by
  sorry

end square_equals_1369_l3378_337815


namespace sequence_convergence_l3378_337897

def sequence_property (r : ℝ) (a : ℕ → ℤ) : Prop :=
  r ≥ 0 ∧ ∀ n, a n ≤ a (n + 2) ∧ (a (n + 2) : ℝ)^2 ≤ (a n : ℝ)^2 + r * (a (n + 1) : ℝ)

theorem sequence_convergence (r : ℝ) (a : ℕ → ℤ) (h : sequence_property r a) :
  (r ≤ 2 → ∃ N, ∀ n ≥ N, a (n + 2) = a n) ∧
  (r > 2 → ∃ a : ℕ → ℤ, sequence_property r a ∧ ∀ N, ∃ n ≥ N, a (n + 2) ≠ a n) := by
  sorry

end sequence_convergence_l3378_337897


namespace pepperoni_coverage_l3378_337858

theorem pepperoni_coverage (pizza_diameter : ℝ) (pepperoni_count : ℕ) (pepperoni_across : ℕ) :
  pizza_diameter = 18 →
  pepperoni_count = 36 →
  pepperoni_across = 9 →
  (pepperoni_count * (pizza_diameter / pepperoni_across / 2)^2 * Real.pi) / (pizza_diameter / 2)^2 / Real.pi = 4 / 9 := by
  sorry

end pepperoni_coverage_l3378_337858


namespace dabbie_turkey_cost_l3378_337834

/-- The cost of Dabbie's turkeys -/
def turkey_cost : ℕ → ℕ
| 0 => 6  -- weight of first turkey
| 1 => 9  -- weight of second turkey
| 2 => 2 * turkey_cost 1  -- weight of third turkey
| _ => 0  -- for completeness

/-- The total weight of all turkeys -/
def total_weight : ℕ := turkey_cost 0 + turkey_cost 1 + turkey_cost 2

/-- The cost per kilogram of turkey -/
def cost_per_kg : ℕ := 2

/-- The theorem stating the total cost of Dabbie's turkeys -/
theorem dabbie_turkey_cost : total_weight * cost_per_kg = 66 := by
  sorry

end dabbie_turkey_cost_l3378_337834


namespace football_count_proof_l3378_337831

/-- The cost of a single soccer ball in dollars -/
def soccer_ball_cost : ℕ := 50

/-- The cost of some footballs and 3 soccer balls in dollars -/
def first_set_cost : ℕ := 220

/-- The cost of 3 footballs and 1 soccer ball in dollars -/
def second_set_cost : ℕ := 155

/-- The number of footballs in the second set -/
def footballs_in_second_set : ℕ := 3

theorem football_count_proof : 
  ∃ (football_cost : ℕ) (footballs_in_first_set : ℕ),
    footballs_in_first_set * football_cost + 3 * soccer_ball_cost = first_set_cost ∧
    3 * football_cost + soccer_ball_cost = second_set_cost ∧
    footballs_in_second_set = 3 :=
sorry

end football_count_proof_l3378_337831


namespace total_sales_proof_l3378_337827

def robyn_sales : ℕ := 55
def lucy_sales : ℕ := 43

theorem total_sales_proof : robyn_sales + lucy_sales = 98 := by
  sorry

end total_sales_proof_l3378_337827


namespace smallest_next_divisor_after_427_l3378_337878

theorem smallest_next_divisor_after_427 (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : m % 2 = 0) (h3 : m % 427 = 0) :
  ∃ (d : ℕ), d > 427 ∧ m % d = 0 ∧ d = 434 ∧ 
  ∀ (x : ℕ), 427 < x ∧ x < 434 → m % x ≠ 0 := by
sorry

end smallest_next_divisor_after_427_l3378_337878


namespace system_of_equations_l3378_337862

theorem system_of_equations (x y a : ℝ) : 
  (3 * x + y = a + 1) → 
  (x + 3 * y = 3) → 
  (x + y > 5) → 
  (a > 16) := by
sorry

end system_of_equations_l3378_337862


namespace function_equation_zero_l3378_337859

theorem function_equation_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f y = f (f x * f y)) : 
  ∀ x : ℝ, f x = 0 := by sorry

end function_equation_zero_l3378_337859


namespace ratio_problem_l3378_337832

theorem ratio_problem (second_term : ℝ) (ratio_percent : ℝ) (first_term : ℝ) :
  second_term = 25 →
  ratio_percent = 60 →
  first_term / second_term = ratio_percent / 100 →
  first_term = 15 := by
sorry

end ratio_problem_l3378_337832


namespace encyclopedia_monthly_payment_l3378_337833

/-- Proves that the monthly payment for the encyclopedia purchase is $57 -/
theorem encyclopedia_monthly_payment
  (total_cost : ℝ)
  (down_payment : ℝ)
  (num_monthly_payments : ℕ)
  (final_payment : ℝ)
  (interest_rate : ℝ)
  (h_total_cost : total_cost = 750)
  (h_down_payment : down_payment = 300)
  (h_num_monthly_payments : num_monthly_payments = 9)
  (h_final_payment : final_payment = 21)
  (h_interest_rate : interest_rate = 0.18666666666666668)
  : ∃ (monthly_payment : ℝ),
    monthly_payment = 57 ∧
    total_cost - down_payment + (total_cost - down_payment) * interest_rate =
    monthly_payment * num_monthly_payments + final_payment := by
  sorry

end encyclopedia_monthly_payment_l3378_337833


namespace inscribed_sphere_radius_specific_prism_l3378_337884

/-- An equilateral triangular prism -/
structure EquilateralTriangularPrism where
  /-- The base side length of the prism -/
  baseSideLength : ℝ
  /-- The height of the prism -/
  height : ℝ

/-- The radius of the inscribed sphere in an equilateral triangular prism -/
def inscribedSphereRadius (prism : EquilateralTriangularPrism) : ℝ :=
  sorry

/-- Theorem: The radius of the inscribed sphere in an equilateral triangular prism
    with base side length 1 and height √2 is equal to √2/6 -/
theorem inscribed_sphere_radius_specific_prism :
  let prism : EquilateralTriangularPrism := { baseSideLength := 1, height := Real.sqrt 2 }
  inscribedSphereRadius prism = Real.sqrt 2 / 6 :=
by sorry

end inscribed_sphere_radius_specific_prism_l3378_337884


namespace badge_making_contest_tables_l3378_337843

theorem badge_making_contest_tables (stools_per_table : ℕ) (stool_legs : ℕ) (table_legs : ℕ) (total_legs : ℕ) : 
  stools_per_table = 7 → 
  stool_legs = 4 → 
  table_legs = 5 → 
  total_legs = 658 → 
  ∃ (num_tables : ℕ), num_tables = 20 ∧ 
    total_legs = stool_legs * stools_per_table * num_tables + table_legs * num_tables :=
by sorry

end badge_making_contest_tables_l3378_337843


namespace pedal_triangles_common_circumcircle_l3378_337801

/-- Triangle type -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Isotomic conjugates with respect to a triangle -/
def IsotomicConjugates (P₁ P₂ : Point) (T : Triangle) : Prop := sorry

/-- Pedal triangle of a point with respect to a triangle, given an angle -/
def PedalTriangle (P : Point) (T : Triangle) (angle : ℝ) : Triangle := sorry

/-- Circumcircle of a triangle -/
def Circumcircle (T : Triangle) : Circle := sorry

/-- Center of a circle -/
def Center (C : Circle) : Point := sorry

/-- Midpoint of a segment -/
def Midpoint (A B : Point) : Point := sorry

theorem pedal_triangles_common_circumcircle 
  (T : Triangle) (P₁ P₂ : Point) (angle : ℝ) :
  IsotomicConjugates P₁ P₂ T →
  ∃ (C : Circle), 
    Circumcircle (PedalTriangle P₁ T angle) = C ∧
    Circumcircle (PedalTriangle P₂ T angle) = C ∧
    Center C = Midpoint P₁ P₂ := by
  sorry

end pedal_triangles_common_circumcircle_l3378_337801


namespace smallest_sum_of_three_smallest_sum_is_achievable_l3378_337841

def S : Finset Int := {8, -7, 2, -4, 20}

theorem smallest_sum_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a + b + c ≥ -9 :=
by sorry

theorem smallest_sum_is_achievable :
  ∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = -9 :=
by sorry

end smallest_sum_of_three_smallest_sum_is_achievable_l3378_337841


namespace invalid_league_schedule_l3378_337879

/-- Represents a league schedule --/
structure LeagueSchedule where
  num_teams : Nat
  num_dates : Nat
  max_games_per_date : Nat

/-- Calculate the total number of games in a round-robin tournament --/
def total_games (schedule : LeagueSchedule) : Nat :=
  schedule.num_teams * (schedule.num_teams - 1) / 2

/-- Check if a schedule is valid --/
def is_valid_schedule (schedule : LeagueSchedule) : Prop :=
  total_games schedule ≤ schedule.num_dates * schedule.max_games_per_date

/-- Theorem stating that the given schedule is invalid --/
theorem invalid_league_schedule : 
  ¬ is_valid_schedule ⟨20, 5, 8⟩ := by
  sorry

#eval total_games ⟨20, 5, 8⟩

end invalid_league_schedule_l3378_337879


namespace correct_verb_forms_l3378_337802

/-- Represents the grammatical number of a subject --/
inductive GrammaticalNumber
| Singular
| Plural

/-- Represents a subject in a sentence --/
structure Subject where
  text : String
  number : GrammaticalNumber

/-- Represents a verb in a sentence --/
structure Verb where
  singular_form : String
  plural_form : String

/-- Checks if a verb agrees with a subject --/
def verb_agrees (s : Subject) (v : Verb) : Prop :=
  match s.number with
  | GrammaticalNumber.Singular => v.singular_form = "is"
  | GrammaticalNumber.Plural => v.plural_form = "want"

/-- The main theorem stating the correct verb forms for the given subjects --/
theorem correct_verb_forms 
  (subject1 : Subject)
  (subject2 : Subject)
  (h1 : subject1.text = "The number of the stamps")
  (h2 : subject2.text = "a number of people")
  (h3 : subject1.number = GrammaticalNumber.Singular)
  (h4 : subject2.number = GrammaticalNumber.Plural) :
  ∃ (v1 v2 : Verb), 
    verb_agrees subject1 v1 ∧ 
    verb_agrees subject2 v2 ∧ 
    v1.singular_form = "is" ∧ 
    v2.plural_form = "want" := by
  sorry


end correct_verb_forms_l3378_337802


namespace intersection_distance_l3378_337891

-- Define the curves and ray
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C₂ (x y θ : ℝ) : Prop := x = Real.sqrt 2 * Real.cos θ ∧ y = Real.sin θ
def ray (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x ∧ x ≥ 0

-- Define the intersection points
def point_A (x y : ℝ) : Prop := C₁ x y ∧ ray x y
def point_B (x y θ : ℝ) : Prop := C₂ x y θ ∧ ray x y

-- Theorem statement
theorem intersection_distance :
  ∀ (x₁ y₁ x₂ y₂ θ : ℝ),
  point_A x₁ y₁ → point_B x₂ y₂ θ →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = Real.sqrt 3 - 2 * Real.sqrt 10 / 5 := by
  sorry

end intersection_distance_l3378_337891


namespace koala_fiber_intake_l3378_337886

/-- Proves that if a koala absorbs 20% of the fiber it eats and it absorbed 8 ounces of fiber in one day, then the total amount of fiber the koala ate that day was 40 ounces. -/
theorem koala_fiber_intake (absorption_rate : Real) (absorbed_amount : Real) (total_intake : Real) :
  absorption_rate = 0.20 →
  absorbed_amount = 8 →
  absorbed_amount = absorption_rate * total_intake →
  total_intake = 40 := by
  sorry

end koala_fiber_intake_l3378_337886


namespace obtuse_triangle_k_range_l3378_337857

/-- An obtuse triangle ABC with sides a = k, b = k + 2, and c = k + 4 -/
structure ObtuseTriangle (k : ℝ) where
  a : ℝ := k
  b : ℝ := k + 2
  c : ℝ := k + 4
  is_obtuse : c^2 > a^2 + b^2

/-- The range of possible values for k in an obtuse triangle with sides k, k+2, k+4 -/
theorem obtuse_triangle_k_range (k : ℝ) :
  (∃ t : ObtuseTriangle k, True) ↔ 2 < k ∧ k < 6 := by
  sorry

#check obtuse_triangle_k_range

end obtuse_triangle_k_range_l3378_337857


namespace fraction_of_powers_equals_500_l3378_337893

theorem fraction_of_powers_equals_500 : (0.5 : ℝ)^4 / (0.05 : ℝ)^3 = 500 := by
  sorry

end fraction_of_powers_equals_500_l3378_337893


namespace product_remainder_l3378_337812

theorem product_remainder (a b m : ℕ) (ha : a = 1492) (hb : b = 1999) (hm : m = 500) :
  (a * b) % m = 8 := by
  sorry

end product_remainder_l3378_337812


namespace max_discount_rate_l3378_337842

theorem max_discount_rate (cost_price : ℝ) (original_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 →
  original_price = 5 →
  min_profit_margin = 0.1 →
  ∃ (max_discount : ℝ),
    max_discount = 60 ∧
    ∀ (discount : ℝ),
      discount ≤ max_discount →
      (original_price * (1 - discount / 100) - cost_price) / cost_price ≥ min_profit_margin :=
by sorry

#check max_discount_rate

end max_discount_rate_l3378_337842


namespace min_points_theorem_min_points_is_minimal_l3378_337822

/-- Represents a point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the angle APB given three points A, P, and B -/
def angle (A P B : Point) : ℝ := sorry

/-- The minimum number of points satisfying the given condition -/
def min_points : ℕ := 1993

theorem min_points_theorem (A B : Point) :
  ∀ (points : Finset Point),
    points.card ≥ min_points →
    ∃ (Pi Pj : Point), Pi ∈ points ∧ Pj ∈ points ∧ Pi ≠ Pj ∧
      |Real.sin (angle A Pi B) - Real.sin (angle A Pj B)| ≤ 1 / 1992 :=
by sorry

theorem min_points_is_minimal :
  ∀ k : ℕ, k < min_points →
    ∃ (A B : Point) (points : Finset Point),
      points.card = k ∧
      ∀ (Pi Pj : Point), Pi ∈ points → Pj ∈ points → Pi ≠ Pj →
        |Real.sin (angle A Pi B) - Real.sin (angle A Pj B)| > 1 / 1992 :=
by sorry

end min_points_theorem_min_points_is_minimal_l3378_337822


namespace deduced_card_final_card_l3378_337825

-- Define the suits and ranks
inductive Suit
| Hearts | Spades | Clubs | Diamonds

inductive Rank
| A | K | Q | J | Ten | Nine | Eight | Seven | Six | Five | Four | Three | Two

-- Define a card as a pair of suit and rank
structure Card where
  suit : Suit
  rank : Rank

-- Define the set of cards in the drawer
def drawer : List Card := [
  ⟨Suit.Hearts, Rank.A⟩, ⟨Suit.Hearts, Rank.Q⟩, ⟨Suit.Hearts, Rank.Four⟩,
  ⟨Suit.Spades, Rank.J⟩, ⟨Suit.Spades, Rank.Eight⟩, ⟨Suit.Spades, Rank.Four⟩,
  ⟨Suit.Spades, Rank.Two⟩, ⟨Suit.Spades, Rank.Seven⟩, ⟨Suit.Spades, Rank.Three⟩,
  ⟨Suit.Clubs, Rank.K⟩, ⟨Suit.Clubs, Rank.Q⟩, ⟨Suit.Clubs, Rank.Five⟩,
  ⟨Suit.Clubs, Rank.Four⟩, ⟨Suit.Clubs, Rank.Six⟩,
  ⟨Suit.Diamonds, Rank.A⟩, ⟨Suit.Diamonds, Rank.Five⟩
]

-- Define the conditions based on the conversation
def qian_first_statement (c : Card) : Prop :=
  c.rank = Rank.A ∨ c.rank = Rank.Q ∨ c.rank = Rank.Five ∨ c.rank = Rank.Four

def sun_first_statement (c : Card) : Prop :=
  c.suit = Suit.Hearts ∨ c.suit = Suit.Diamonds

def qian_second_statement (c : Card) : Prop :=
  c.rank ≠ Rank.A

-- The main theorem
theorem deduced_card :
  ∃! c : Card, c ∈ drawer ∧
    qian_first_statement c ∧
    sun_first_statement c ∧
    qian_second_statement c :=
  sorry

-- The final conclusion
theorem final_card :
  ∃! c : Card, c ∈ drawer ∧
    qian_first_statement c ∧
    sun_first_statement c ∧
    qian_second_statement c ∧
    c = ⟨Suit.Diamonds, Rank.Five⟩ :=
  sorry

end deduced_card_final_card_l3378_337825


namespace min_trig_expression_l3378_337869

theorem min_trig_expression (x : ℝ) : 
  (Real.sin x)^8 + (Real.cos x)^8 + 1 ≥ 7/18 * ((Real.sin x)^6 + (Real.cos x)^6 + 1) := by
  sorry

end min_trig_expression_l3378_337869


namespace right_triangle_angle_measure_l3378_337837

theorem right_triangle_angle_measure (A B C : ℝ) : 
  A = 90 →  -- A is the right angle (90 degrees)
  C = 3 * B →  -- C is three times B
  A + B + C = 180 →  -- Sum of angles in a triangle
  B = 22.5 :=  -- B is 22.5 degrees
by sorry

end right_triangle_angle_measure_l3378_337837


namespace simplify_expression_l3378_337803

theorem simplify_expression (a b c : ℝ) :
  (15*a + 45*b + 20*c) + (25*a - 35*b - 10*c) - (10*a + 55*b + 30*c) = 30*a - 45*b - 20*c :=
by sorry

end simplify_expression_l3378_337803


namespace consecutive_sum_product_l3378_337813

theorem consecutive_sum_product (a : ℤ) : (3*a + 3) * (3*a + 12) ≠ 111111111 := by
  sorry

end consecutive_sum_product_l3378_337813


namespace expression_simplification_and_evaluation_l3378_337877

theorem expression_simplification_and_evaluation (a b : ℝ) 
  (h : |a + 1| + (b - 1/2)^2 = 0) : 
  5 * (a^2 * b - a * b^2) - (a * b^2 + 3 * a^2 * b) = 5/2 := by
  sorry

end expression_simplification_and_evaluation_l3378_337877


namespace keith_purchases_cost_l3378_337872

/-- The total cost of Keith's purchases -/
def total_cost (rabbit_toy pet_food cage water_bottle bedding found_money : ℝ)
  (rabbit_discount cage_tax : ℝ) : ℝ :=
  let rabbit_toy_original := rabbit_toy / (1 - rabbit_discount)
  let cage_with_tax := cage * (1 + cage_tax)
  rabbit_toy + pet_food + cage_with_tax + water_bottle + bedding - found_money

/-- Theorem stating the total cost of Keith's purchases -/
theorem keith_purchases_cost :
  total_cost 6.51 5.79 12.51 4.99 7.65 1 0.1 0.08 = 37.454 := by
  sorry

end keith_purchases_cost_l3378_337872


namespace hot_sauce_servings_per_day_l3378_337809

/-- Proves the number of hot sauce servings used per day -/
theorem hot_sauce_servings_per_day 
  (serving_size : Real) 
  (jar_size : Real) 
  (duration : Nat) 
  (h1 : serving_size = 0.5)
  (h2 : jar_size = 32 - 2)
  (h3 : duration = 20) :
  (jar_size / duration) / serving_size = 3 := by
  sorry

end hot_sauce_servings_per_day_l3378_337809


namespace parallel_vectors_sum_l3378_337856

theorem parallel_vectors_sum (x y : ℝ) 
  (a : ℝ × ℝ × ℝ) (b : ℝ × ℝ × ℝ) 
  (ha : a = (2, 1, x)) (hb : b = (4, y, -1)) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ a = k • b) : 
  2 * x + y = 1 := by
  sorry

end parallel_vectors_sum_l3378_337856


namespace smaller_cuboid_height_l3378_337860

/-- Given a large cuboid and smaller cuboids with specified dimensions,
    prove that the height of each smaller cuboid is 3 meters. -/
theorem smaller_cuboid_height
  (large_length large_width large_height : ℝ)
  (small_length small_width : ℝ)
  (num_small_cuboids : ℝ)
  (h_large_length : large_length = 18)
  (h_large_width : large_width = 15)
  (h_large_height : large_height = 2)
  (h_small_length : small_length = 6)
  (h_small_width : small_width = 4)
  (h_num_small_cuboids : num_small_cuboids = 7.5)
  (h_volume_conservation : large_length * large_width * large_height =
    num_small_cuboids * small_length * small_width * (large_length * large_width * large_height / (num_small_cuboids * small_length * small_width))) :
  large_length * large_width * large_height / (num_small_cuboids * small_length * small_width) = 3 := by
  sorry

end smaller_cuboid_height_l3378_337860


namespace radio_price_rank_l3378_337805

theorem radio_price_rank (n : ℕ) (prices : Finset ℕ) (radio_price : ℕ) :
  n = 16 →
  prices.card = n + 1 →
  (∀ (p q : ℕ), p ∈ prices → q ∈ prices → p ≠ q) →
  radio_price ∈ prices →
  (prices.filter (λ p => p > radio_price)).card = 3 →
  (prices.filter (λ p => p < radio_price)).card = n - 3 :=
by sorry

end radio_price_rank_l3378_337805


namespace mathematicians_set_l3378_337888

-- Define the type for famous figures
inductive FamousFigure
| BillGates
| Gauss
| LiuXiang
| Nobel
| ChenJingrun
| ChenXingshen
| Gorky
| Einstein

-- Define the set of all famous figures
def allFigures : Set FamousFigure :=
  {FamousFigure.BillGates, FamousFigure.Gauss, FamousFigure.LiuXiang, 
   FamousFigure.Nobel, FamousFigure.ChenJingrun, FamousFigure.ChenXingshen, 
   FamousFigure.Gorky, FamousFigure.Einstein}

-- Define the property of being a mathematician
def isMathematician : FamousFigure → Prop :=
  fun figure => match figure with
  | FamousFigure.Gauss => True
  | FamousFigure.ChenJingrun => True
  | FamousFigure.ChenXingshen => True
  | _ => False

-- Theorem: The set of mathematicians is equal to {Gauss, Chen Jingrun, Chen Xingshen}
theorem mathematicians_set :
  {figure ∈ allFigures | isMathematician figure} =
  {FamousFigure.Gauss, FamousFigure.ChenJingrun, FamousFigure.ChenXingshen} :=
by sorry

end mathematicians_set_l3378_337888


namespace quadratic_completion_of_square_l3378_337892

theorem quadratic_completion_of_square (x : ℝ) :
  ∃ (a h k : ℝ), x^2 - 7*x + 6 = a*(x - h)^2 + k ∧ k = -25/4 := by
  sorry

end quadratic_completion_of_square_l3378_337892


namespace parabolas_intersection_l3378_337840

def parabola1 (x : ℝ) : ℝ := 2 * x^2 - 7 * x + 1
def parabola2 (x : ℝ) : ℝ := 8 * x^2 + 5 * x + 1

theorem parabolas_intersection :
  ∃! (s : Set (ℝ × ℝ)), s = {(-2, 23), (0, 1)} ∧
  (∀ (x y : ℝ), (x, y) ∈ s ↔ parabola1 x = y ∧ parabola2 x = y) :=
sorry

end parabolas_intersection_l3378_337840


namespace small_tile_position_l3378_337853

/-- Represents a tile in the square --/
inductive Tile
| Large : Tile  -- 1×3 tile
| Small : Tile  -- 1×1 tile

/-- Represents a position in the 7×7 square --/
structure Position :=
(row : Fin 7)
(col : Fin 7)

/-- Defines if a position is in the center or adjacent to the border --/
def is_center_or_border (p : Position) : Prop :=
  (p.row = 3 ∧ p.col = 3) ∨ 
  p.row = 0 ∨ p.row = 6 ∨ p.col = 0 ∨ p.col = 6

/-- Represents the arrangement of tiles in the square --/
def Arrangement := Position → Tile

/-- The theorem to be proved --/
theorem small_tile_position 
  (arr : Arrangement) 
  (h1 : ∃! p, arr p = Tile.Small) 
  (h2 : ∀ p, arr p = Tile.Large → 
       ∃ p1 p2, p1 ≠ p ∧ p2 ≠ p ∧ p1 ≠ p2 ∧ 
       arr p1 = Tile.Large ∧ arr p2 = Tile.Large) 
  (h3 : ∀ p, arr p = Tile.Large ∨ arr p = Tile.Small) :
  ∃ p, arr p = Tile.Small ∧ is_center_or_border p :=
sorry

end small_tile_position_l3378_337853


namespace altitude_properties_l3378_337814

-- Define the triangle ABC
def A : ℝ × ℝ := (2, -1)
def B : ℝ × ℝ := (3, 2)
def C : ℝ × ℝ := (-3, -1)

-- Define vector BC
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Define the altitude AD
def AD : ℝ × ℝ → Prop := λ D => 
  -- AD is perpendicular to BC
  (D.1 - A.1) * BC.1 + (D.2 - A.2) * BC.2 = 0 ∧
  -- D lies on line BC
  ∃ t : ℝ, D = (B.1 + t * BC.1, B.2 + t * BC.2)

-- Theorem statement
theorem altitude_properties : 
  ∃ D : ℝ × ℝ, AD D ∧ 
    ((D.1 - A.1)^2 + (D.2 - A.2)^2 = 5) ∧ 
    D = (1, 1) :=
sorry

end altitude_properties_l3378_337814


namespace product_of_three_numbers_l3378_337898

theorem product_of_three_numbers (x y z m : ℚ) : 
  x + y + z = 240 ∧ 
  9 * x = m ∧ 
  y - 11 = m ∧ 
  z + 11 = m ∧ 
  x < y ∧ 
  x < z → 
  x * y * z = 7514700 / 9 := by
sorry

end product_of_three_numbers_l3378_337898


namespace thomas_daniel_equation_l3378_337817

theorem thomas_daniel_equation (b c : ℝ) : 
  (∀ x : ℝ, |x - 4| = 3 ↔ x^2 + b*x + c = 0) → 
  b = -8 ∧ c = 7 := by
sorry

end thomas_daniel_equation_l3378_337817


namespace partnership_profit_calculation_l3378_337880

/-- Represents the investment and profit information for a partnership business --/
structure PartnershipBusiness where
  a_initial : ℕ
  a_additional : ℕ
  a_additional_time : ℕ
  b_initial : ℕ
  b_withdrawal : ℕ
  b_withdrawal_time : ℕ
  c_initial : ℕ
  c_additional : ℕ
  c_additional_time : ℕ
  total_time : ℕ
  c_profit : ℕ

/-- Calculates the total profit of the partnership business --/
def calculate_total_profit (pb : PartnershipBusiness) : ℕ :=
  sorry

/-- Theorem stating that given the specific investment conditions, 
    if C's profit is 45000, then the total profit is 103571 --/
theorem partnership_profit_calculation 
  (pb : PartnershipBusiness)
  (h1 : pb.a_initial = 5000)
  (h2 : pb.a_additional = 2000)
  (h3 : pb.a_additional_time = 4)
  (h4 : pb.b_initial = 8000)
  (h5 : pb.b_withdrawal = 1000)
  (h6 : pb.b_withdrawal_time = 4)
  (h7 : pb.c_initial = 9000)
  (h8 : pb.c_additional = 3000)
  (h9 : pb.c_additional_time = 6)
  (h10 : pb.total_time = 12)
  (h11 : pb.c_profit = 45000) :
  calculate_total_profit pb = 103571 :=
sorry

end partnership_profit_calculation_l3378_337880


namespace abc_inequality_l3378_337820

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + a*b*c = 4) :
  a^2 * b^2 + b^2 * c^2 + c^2 * a^2 + a*b*c ≤ 4 := by
  sorry

end abc_inequality_l3378_337820


namespace complex_square_l3378_337806

theorem complex_square (z : ℂ) (h : z * Complex.I = 2 + Complex.I) : z^2 = -3 - 4*Complex.I := by
  sorry

end complex_square_l3378_337806


namespace max_value_of_m_over_n_l3378_337876

theorem max_value_of_m_over_n (n : ℝ) (m : ℝ) (h_n : n > 0) :
  (∀ x > 0, Real.log x + 1 ≥ m - n / x) →
  m / n ≤ Real.exp 1 :=
by sorry

end max_value_of_m_over_n_l3378_337876


namespace computer_price_increase_l3378_337873

theorem computer_price_increase (d : ℝ) (h1 : d * 1.3 = 338) (h2 : ∃ x : ℝ, x * d = 520) : 
  ∃ x : ℝ, x * d = 520 ∧ x = 2 := by
  sorry

end computer_price_increase_l3378_337873


namespace product_odd_probability_l3378_337838

def range_start : ℕ := 5
def range_end : ℕ := 19

def is_in_range (n : ℕ) : Prop := range_start ≤ n ∧ n ≤ range_end

def total_integers : ℕ := range_end - range_start + 1

def odd_integers : ℕ := (total_integers + 1) / 2

theorem product_odd_probability :
  (odd_integers.choose 2 : ℚ) / (total_integers.choose 2) = 4 / 15 :=
sorry

end product_odd_probability_l3378_337838


namespace students_on_bleachers_l3378_337824

/-- Given a total of 26 students and a ratio of 11:13 for students on the floor to total students,
    prove that the number of students on the bleachers is 4. -/
theorem students_on_bleachers :
  ∀ (floor bleachers : ℕ),
    floor + bleachers = 26 →
    floor / (floor + bleachers : ℚ) = 11 / 13 →
    bleachers = 4 := by
  sorry

end students_on_bleachers_l3378_337824


namespace integer_congruence_problem_l3378_337811

theorem integer_congruence_problem :
  ∀ n : ℤ, 5 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [ZMOD 6] → n = 9 := by
  sorry

end integer_congruence_problem_l3378_337811


namespace arithmetic_sequence_50th_term_l3378_337882

theorem arithmetic_sequence_50th_term : 
  let start : ℤ := -48
  let diff : ℤ := 2
  let n : ℕ := 50
  let sequence := fun i : ℕ => start + diff * (i - 1)
  sequence n = 50 := by
sorry

end arithmetic_sequence_50th_term_l3378_337882


namespace max_distance_on_circle_l3378_337818

open Complex

theorem max_distance_on_circle (z : ℂ) :
  abs (z - (1 + I)) = 1 →
  (∃ (w : ℂ), abs (w - (1 + I)) = 1 ∧ abs (w - (4 + 5*I)) ≥ abs (z - (4 + 5*I))) ∧
  (∀ (w : ℂ), abs (w - (1 + I)) = 1 → abs (w - (4 + 5*I)) ≤ 6) ∧
  (∃ (w : ℂ), abs (w - (1 + I)) = 1 ∧ abs (w - (4 + 5*I)) = 6) :=
by sorry

end max_distance_on_circle_l3378_337818


namespace quadrilateral_area_l3378_337835

/-- The area of a quadrilateral with given diagonal and offsets -/
theorem quadrilateral_area (d h₁ h₂ : ℝ) (hd : d = 40) (hh₁ : h₁ = 9) (hh₂ : h₂ = 6) :
  (1 / 2 : ℝ) * d * h₁ + (1 / 2 : ℝ) * d * h₂ = 300 := by
  sorry

end quadrilateral_area_l3378_337835


namespace smallest_k_square_root_diff_l3378_337868

/-- Represents a card with a number from 1 to 2016 -/
def Card := {n : ℕ // 1 ≤ n ∧ n ≤ 2016}

/-- The property that two cards have numbers whose square roots differ by less than 1 -/
def SquareRootDiffLessThanOne (a b : Card) : Prop :=
  |Real.sqrt a.val - Real.sqrt b.val| < 1

/-- The theorem stating that 45 is the smallest number of cards guaranteeing
    two cards with square root difference less than 1 -/
theorem smallest_k_square_root_diff : 
  (∀ (S : Finset Card), S.card = 45 → 
    ∃ (a b : Card), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ SquareRootDiffLessThanOne a b) ∧
  (∀ (k : ℕ), k < 45 → 
    ∃ (S : Finset Card), S.card = k ∧
      ∀ (a b : Card), a ∈ S → b ∈ S → a ≠ b → ¬SquareRootDiffLessThanOne a b) :=
by sorry

end smallest_k_square_root_diff_l3378_337868


namespace total_toothpicks_needed_l3378_337864

/-- The number of small triangles in the base row of the large equilateral triangle. -/
def base_triangles : ℕ := 2004

/-- The total number of small triangles in the large equilateral triangle. -/
def total_triangles : ℕ := base_triangles * (base_triangles + 1) / 2

/-- The number of toothpicks needed if each side of each small triangle was unique. -/
def total_sides : ℕ := 3 * total_triangles

/-- The number of toothpicks on the boundary of the large triangle. -/
def boundary_toothpicks : ℕ := 3 * base_triangles

/-- Theorem: The total number of toothpicks needed to construct the large equilateral triangle. -/
theorem total_toothpicks_needed : 
  (total_sides / 2) + boundary_toothpicks = 3021042 := by
  sorry

end total_toothpicks_needed_l3378_337864


namespace orchard_harvest_l3378_337875

/-- Calculates the total mass of fruit harvested in an orchard -/
def total_fruit_mass (apple_trees : ℕ) (apple_yield : ℕ) (peach_trees : ℕ) (peach_yield : ℕ) : ℕ :=
  apple_trees * apple_yield + peach_trees * peach_yield

/-- Theorem stating the total mass of fruit harvested in the specific orchard -/
theorem orchard_harvest :
  total_fruit_mass 30 150 45 65 = 7425 := by
  sorry

end orchard_harvest_l3378_337875


namespace pond_to_field_area_ratio_l3378_337874

/-- Proves that the ratio of a square pond's area to a rectangular field's area is 1:8,
    given specific dimensions of the field and pond. -/
theorem pond_to_field_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    field_length = 36 →
    field_length = 2 * field_width →
    pond_side = 9 →
    (pond_side^2) / (field_length * field_width) = 1 / 8 := by
  sorry

end pond_to_field_area_ratio_l3378_337874


namespace recurring_decimal_to_fraction_l3378_337867

theorem recurring_decimal_to_fraction : 
  ∃ (x : ℚ), x = 4 + 56 / 99 ∧ x = 452 / 99 := by sorry

end recurring_decimal_to_fraction_l3378_337867


namespace stock_price_theorem_l3378_337885

/-- The face value of the stock (assumed to be $100) -/
def faceValue : ℝ := 100

/-- A's stock interest rate -/
def interestRateA : ℝ := 0.10

/-- B's stock interest rate -/
def interestRateB : ℝ := 0.12

/-- The amount B must invest to get an equally good investment -/
def bInvestment : ℝ := 115.2

/-- The price of the stock A invested in -/
def stockPriceA : ℝ := 138.24

/-- Theorem stating that given the conditions, the price of A's stock is $138.24 -/
theorem stock_price_theorem :
  let incomeA := faceValue * interestRateA
  let requiredInvestmentB := incomeA / interestRateB
  let marketPriceB := bInvestment * (faceValue / requiredInvestmentB)
  marketPriceB = stockPriceA := by
  sorry

#check stock_price_theorem

end stock_price_theorem_l3378_337885


namespace solution_verification_l3378_337865

theorem solution_verification :
  let x : ℚ := 425
  let y : ℝ := (270 + 90 * Real.sqrt 2) / 7
  (x - (11/17) * x = 150) ∧ (y - ((Real.sqrt 2)/3) * y = 90) := by sorry

end solution_verification_l3378_337865


namespace problem_solution_l3378_337866

theorem problem_solution (a b : ℤ) 
  (eq1 : 1010 * a + 1014 * b = 1018)
  (eq2 : 1012 * a + 1016 * b = 1020) : 
  a - b = -3 := by
sorry

end problem_solution_l3378_337866


namespace max_distance_point_to_line_l3378_337881

/-- The maximum distance from a point to a line --/
theorem max_distance_point_to_line : 
  let P : ℝ × ℝ := (-1, 3)
  let line_equation (k x : ℝ) := k * (x - 2)
  ∀ k : ℝ, 
  (∃ x : ℝ, abs (P.2 - line_equation k P.1) / Real.sqrt (k^2 + 1) ≤ 3 * Real.sqrt 2) ∧ 
  (∃ k₀ : ℝ, abs (P.2 - line_equation k₀ P.1) / Real.sqrt (k₀^2 + 1) = 3 * Real.sqrt 2) :=
by sorry

end max_distance_point_to_line_l3378_337881


namespace circle_equation_l3378_337852

/-- A circle C with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- The line x - 2y - 1 = 0 -/
def line (x y : ℝ) : Prop := x - 2*y - 1 = 0

/-- A point (x, y) lies on the circle C -/
def on_circle (C : Circle) (x y : ℝ) : Prop :=
  (x - C.h)^2 + (y - C.k)^2 = C.r^2

theorem circle_equation : ∃ C : Circle,
  (line C.h C.k) ∧
  (on_circle C 0 0) ∧
  (on_circle C 1 2) ∧
  (C.h = 7/4 ∧ C.k = 3/8 ∧ C.r^2 = 205/64) :=
sorry

end circle_equation_l3378_337852


namespace P_superset_Q_l3378_337804

def P : Set ℝ := {x | x < 4}
def Q : Set ℝ := {x | -2 < x ∧ x < 2}

theorem P_superset_Q : P ⊃ Q := by
  sorry

end P_superset_Q_l3378_337804


namespace refund_calculation_l3378_337829

/-- Calculates the refund amount for returned cans given specific conditions -/
theorem refund_calculation (total_cans brand_a_price brand_b_price average_price discount restocking_fee tax : ℚ)
  (h1 : total_cans = 6)
  (h2 : brand_a_price = 33 / 100)
  (h3 : brand_b_price = 40 / 100)
  (h4 : average_price = 365 / 1000)
  (h5 : discount = 20 / 100)
  (h6 : restocking_fee = 5 / 100)
  (h7 : tax = 8 / 100)
  (h8 : ∃ (brand_a_count brand_b_count : ℚ), 
    brand_a_count + brand_b_count = total_cans ∧ 
    brand_a_count * brand_a_price + brand_b_count * brand_b_price = total_cans * average_price ∧
    brand_a_count > brand_b_count) :
  ∃ (refund : ℚ), refund = 55 / 100 := by
  sorry


end refund_calculation_l3378_337829


namespace johns_allowance_l3378_337828

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℚ) : A = 32.4 ↔
  ∃ (arcade toy book candy : ℚ),
    -- Spending at the arcade
    arcade = 7 / 12 * A ∧
    -- Spending at the toy store
    toy = 5 / 9 * (A - arcade) ∧
    -- Spending at the bookstore
    book = 3 / 4 * (A - arcade - toy) ∧
    -- Spending at the candy store
    candy = 3 / 2 ∧
    -- Total spending equals the allowance
    arcade + toy + book + candy = A := by
  sorry

end johns_allowance_l3378_337828


namespace simplify_and_rationalize_fraction_l3378_337889

theorem simplify_and_rationalize_fraction :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18 + Real.sqrt 32) = (5 * Real.sqrt 2) / 36 := by
  sorry

end simplify_and_rationalize_fraction_l3378_337889


namespace field_height_rise_l3378_337854

/-- Calculates the rise in height of a field after digging a pit and spreading the removed earth --/
theorem field_height_rise (field_length field_width pit_length pit_width pit_depth : ℝ) 
  (h_field_length : field_length = 20)
  (h_field_width : field_width = 10)
  (h_pit_length : pit_length = 8)
  (h_pit_width : pit_width = 5)
  (h_pit_depth : pit_depth = 2) :
  let total_area := field_length * field_width
  let pit_area := pit_length * pit_width
  let remaining_area := total_area - pit_area
  let pit_volume := pit_length * pit_width * pit_depth
  pit_volume / remaining_area = 0.5 := by sorry

end field_height_rise_l3378_337854


namespace max_value_is_72_l3378_337883

/-- Represents a type of rock with its weight and value -/
structure Rock where
  weight : ℕ
  value : ℕ

/-- The maximum weight Carl can carry -/
def maxWeight : ℕ := 24

/-- The available types of rocks -/
def rocks : List Rock := [
  { weight := 6, value := 18 },
  { weight := 3, value := 9 },
  { weight := 2, value := 5 }
]

/-- A function to calculate the maximum value of rocks that can be carried -/
def maxValue (rocks : List Rock) (maxWeight : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum value Carl can transport is $72 -/
theorem max_value_is_72 : maxValue rocks maxWeight = 72 := by
  sorry

end max_value_is_72_l3378_337883


namespace trigonometric_identity_l3378_337871

theorem trigonometric_identity (x y : ℝ) : 
  Real.sin (x - y) * Real.cos y + Real.cos (x - y) * Real.sin y = Real.sin x := by
  sorry

end trigonometric_identity_l3378_337871


namespace production_calculation_l3378_337816

-- Define the production rate for 6 machines
def production_rate_6 : ℕ := 300

-- Define the number of machines in the original setup
def original_machines : ℕ := 6

-- Define the number of machines in the new setup
def new_machines : ℕ := 10

-- Define the duration in minutes
def duration : ℕ := 4

-- Theorem to prove
theorem production_calculation :
  (new_machines * duration * production_rate_6) / original_machines = 2000 :=
by
  sorry


end production_calculation_l3378_337816


namespace jay_and_paul_distance_l3378_337819

/-- Calculates the distance traveled given a speed and time --/
def distance (speed : ℚ) (time : ℚ) : ℚ := speed * time

/-- Proves that Jay and Paul will be 20 miles apart after walking in opposite directions for 2 hours --/
theorem jay_and_paul_distance : 
  let jay_speed : ℚ := 1 / 15  -- 1 mile per 15 minutes
  let paul_speed : ℚ := 3 / 30 -- 3 miles per 30 minutes
  let time : ℚ := 2 * 60      -- 2 hours in minutes
  distance jay_speed time + distance paul_speed time = 20 := by
sorry

end jay_and_paul_distance_l3378_337819
