import Mathlib

namespace exists_n_divisible_by_1987_l2198_219815

theorem exists_n_divisible_by_1987 : ∃ n : ℕ, (1987 : ℕ) ∣ (n^n + (n+1)^n) := by
  sorry

end exists_n_divisible_by_1987_l2198_219815


namespace symmetric_quadratic_l2198_219808

/-- A quadratic function f(x) = x² + (a-2)x + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-2)*x + 3

/-- The interval [a, b] -/
def interval (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

/-- The line of symmetry x = 1 -/
def symmetry_line : ℝ := 1

/-- The statement that the graph of f is symmetric about x = 1 on [a, b] -/
def is_symmetric (a b : ℝ) : Prop :=
  ∀ x ∈ interval a b, f a x = f a (2*symmetry_line - x)

theorem symmetric_quadratic (a b : ℝ) :
  is_symmetric a b → b = 2 :=
by sorry

end symmetric_quadratic_l2198_219808


namespace tour_group_composition_l2198_219840

/-- Given a group of 18 people where selecting one male (excluding two ineligible men) 
    and one female results in 64 different combinations, prove that there are 10 men 
    and 8 women in the group. -/
theorem tour_group_composition :
  ∀ (num_men : ℕ),
    (num_men - 2) * (18 - num_men) = 64 →
    num_men = 10 ∧ 18 - num_men = 8 := by
  sorry

end tour_group_composition_l2198_219840


namespace max_tiles_on_floor_l2198_219804

theorem max_tiles_on_floor (floor_length floor_width tile_length tile_width : ℕ) 
  (h1 : floor_length = 120) 
  (h2 : floor_width = 150) 
  (h3 : tile_length = 50) 
  (h4 : tile_width = 40) : 
  (max 
    ((floor_length / tile_length) * (floor_width / tile_width))
    ((floor_length / tile_width) * (floor_width / tile_length))) = 9 := by
  sorry

end max_tiles_on_floor_l2198_219804


namespace positive_integer_solutions_of_equation_l2198_219820

theorem positive_integer_solutions_of_equation : 
  {(x, y) : ℕ × ℕ | x + y + x * y = 2008} = 
  {(6, 286), (286, 6), (40, 48), (48, 40)} := by sorry

end positive_integer_solutions_of_equation_l2198_219820


namespace max_value_x_minus_y_l2198_219898

theorem max_value_x_minus_y :
  ∃ (max : ℝ), max = 2 * Real.sqrt 3 / 3 ∧
  (∀ x y : ℝ, 3 * (x^2 + y^2) = x + y → x - y ≤ max) ∧
  (∃ x y : ℝ, 3 * (x^2 + y^2) = x + y ∧ x - y = max) := by
  sorry

end max_value_x_minus_y_l2198_219898


namespace min_draw_theorem_l2198_219858

/-- Represents the colors of the balls in the bag -/
inductive BallColor
  | Red
  | White
  | Yellow

/-- Represents the bag of balls -/
structure BallBag where
  red : Nat
  white : Nat
  yellow : Nat

/-- The minimum number of balls to draw to guarantee two different colors -/
def minDrawDifferentColors (bag : BallBag) : Nat :=
  bag.red + 1

/-- The minimum number of balls to draw to guarantee two yellow balls -/
def minDrawTwoYellow (bag : BallBag) : Nat :=
  bag.red + bag.white + 2

/-- Theorem stating the minimum number of balls to draw for different scenarios -/
theorem min_draw_theorem (bag : BallBag) 
  (h_red : bag.red = 10) 
  (h_white : bag.white = 10) 
  (h_yellow : bag.yellow = 10) : 
  minDrawDifferentColors bag = 11 ∧ minDrawTwoYellow bag = 22 := by
  sorry

#check min_draw_theorem

end min_draw_theorem_l2198_219858


namespace product_grade_probabilities_l2198_219825

theorem product_grade_probabilities :
  ∀ (p_quality p_second : ℝ),
  p_quality = 0.98 →
  p_second = 0.21 →
  0 ≤ p_quality ∧ p_quality ≤ 1 →
  0 ≤ p_second ∧ p_second ≤ 1 →
  ∃ (p_first p_third : ℝ),
    p_first = p_quality - p_second ∧
    p_third = 1 - p_quality ∧
    p_first = 0.77 ∧
    p_third = 0.02 :=
by
  sorry

end product_grade_probabilities_l2198_219825


namespace arc_length_sixty_degree_l2198_219809

/-- Given a circle with circumference 60 feet and an arc subtended by a central angle of 60°,
    the length of the arc is 10 feet. -/
theorem arc_length_sixty_degree (circle : Real → Real → Prop) 
  (center : Real × Real) (radius : Real) :
  (2 * Real.pi * radius = 60) →  -- Circumference is 60 feet
  (∀ (θ : Real), 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
    circle (center.1 + radius * Real.cos θ) (center.2 + radius * Real.sin θ)) →
  (10 : Real) = (60 / 6) := by sorry

end arc_length_sixty_degree_l2198_219809


namespace max_value_inequality_l2198_219859

theorem max_value_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_condition : a + b + c + d ≤ 4) :
  (2*a^2 + a^2*b)^(1/4) + (2*b^2 + b^2*c)^(1/4) + 
  (2*c^2 + c^2*d)^(1/4) + (2*d^2 + d^2*a)^(1/4) ≤ 4 * 3^(1/4) :=
sorry

end max_value_inequality_l2198_219859


namespace functions_intersect_at_negative_six_l2198_219880

-- Define the two functions
def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := x - 5

-- State the theorem
theorem functions_intersect_at_negative_six : f (-6) = g (-6) := by
  sorry

end functions_intersect_at_negative_six_l2198_219880


namespace f_additive_l2198_219831

/-- A function that satisfies f(a+b) = f(a) + f(b) for all real a and b -/
def f (x : ℝ) : ℝ := 3 * x

/-- Theorem stating that f(a+b) = f(a) + f(b) for all real a and b -/
theorem f_additive (a b : ℝ) : f (a + b) = f a + f b := by
  sorry

end f_additive_l2198_219831


namespace B_power_15_minus_4_power_14_l2198_219891

def B : Matrix (Fin 2) (Fin 2) ℝ := !![4, 5; 0, 2]

theorem B_power_15_minus_4_power_14 :
  B^15 - 4 • B^14 = !![0, 5; 0, -2] := by sorry

end B_power_15_minus_4_power_14_l2198_219891


namespace total_books_l2198_219894

theorem total_books (jason_books mary_books : ℕ) 
  (h1 : jason_books = 18) 
  (h2 : mary_books = 42) : 
  jason_books + mary_books = 60 := by
  sorry

end total_books_l2198_219894


namespace club_size_after_five_years_l2198_219889

def club_growth (initial_members : ℕ) (executives : ℕ) (years : ℕ) : ℕ :=
  let regular_members := initial_members - executives
  let final_regular_members := regular_members * (2 ^ years)
  final_regular_members + executives

theorem club_size_after_five_years :
  club_growth 18 6 5 = 390 := by sorry

end club_size_after_five_years_l2198_219889


namespace sqrt_18_minus_1_bounds_l2198_219874

theorem sqrt_18_minus_1_bounds : 3 < Real.sqrt 18 - 1 ∧ Real.sqrt 18 - 1 < 4 := by
  sorry

end sqrt_18_minus_1_bounds_l2198_219874


namespace girls_to_boys_ratio_l2198_219896

theorem girls_to_boys_ratio (total_students : ℕ) (girls_present : ℕ) (boys_absent : ℕ)
  (h1 : total_students = 250)
  (h2 : girls_present = 140)
  (h3 : boys_absent = 40) :
  (girls_present : ℚ) / (total_students - girls_present - boys_absent) = 2 / 1 := by
  sorry

end girls_to_boys_ratio_l2198_219896


namespace banana_cream_pie_angle_l2198_219833

def total_students : ℕ := 48
def chocolate_preference : ℕ := 15
def apple_preference : ℕ := 9
def blueberry_preference : ℕ := 11

def remaining_students : ℕ := total_students - (chocolate_preference + apple_preference + blueberry_preference)

def banana_cream_preference : ℕ := remaining_students / 2

theorem banana_cream_pie_angle :
  (banana_cream_preference : ℝ) / total_students * 360 = 45 := by
  sorry

end banana_cream_pie_angle_l2198_219833


namespace inequality_range_l2198_219890

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, |x + a| - |x + 1| < 2 * a) ↔ a > 1/3 := by
  sorry

end inequality_range_l2198_219890


namespace orvin_max_balloons_l2198_219855

/-- Represents the maximum number of balloons Orvin can buy given his budget and the sale conditions -/
def max_balloons (regular_price_budget : ℕ) (full_price_ratio : ℕ) (discount_ratio : ℕ) : ℕ :=
  let sets := (regular_price_budget * full_price_ratio) / (full_price_ratio + discount_ratio)
  sets * 2

/-- Proves that Orvin can buy at most 52 balloons given the specified conditions -/
theorem orvin_max_balloons :
  max_balloons 40 2 1 = 52 := by
  sorry

#eval max_balloons 40 2 1

end orvin_max_balloons_l2198_219855


namespace ellipse_equation_l2198_219813

theorem ellipse_equation (a b : ℝ) (ha : a = 6) (hb : b = Real.sqrt 35) :
  (∃ x y : ℝ, x^2 / 36 + y^2 / 35 = 1) ∧ (∃ x y : ℝ, y^2 / 36 + x^2 / 35 = 1) := by
  sorry

end ellipse_equation_l2198_219813


namespace parallelogram_values_l2198_219876

/-- Represents a parallelogram EFGH with given side lengths and area formula -/
structure Parallelogram where
  x : ℝ
  y : ℝ
  ef : ℝ := 5 * x + 7
  fg : ℝ := 4 * y + 1
  gh : ℝ := 27
  he : ℝ := 19
  area : ℝ := 2 * x^2 + y^2 + 5 * x * y + 3

/-- Theorem stating the values of x, y, and area for the given parallelogram -/
theorem parallelogram_values (p : Parallelogram) :
  p.x = 4 ∧ p.y = 4.5 ∧ p.area = 145.25 := by sorry

end parallelogram_values_l2198_219876


namespace coffee_cost_calculation_coffee_cost_calculation_proof_l2198_219886

/-- The daily cost of making coffee given a coffee machine purchase and previous coffee consumption habits. -/
theorem coffee_cost_calculation (machine_cost : ℝ) (discount : ℝ) (previous_coffees_per_day : ℕ) 
  (previous_coffee_price : ℝ) (payback_days : ℕ) (daily_cost : ℝ) : Prop :=
  machine_cost = 200 ∧ 
  discount = 20 ∧
  previous_coffees_per_day = 2 ∧
  previous_coffee_price = 4 ∧
  payback_days = 36 →
  daily_cost = 3

/-- Proof of the coffee cost calculation theorem. -/
theorem coffee_cost_calculation_proof : 
  coffee_cost_calculation 200 20 2 4 36 3 := by
  sorry

end coffee_cost_calculation_coffee_cost_calculation_proof_l2198_219886


namespace course_selection_count_l2198_219895

/-- The number of courses available -/
def total_courses : ℕ := 7

/-- The number of courses each student must choose -/
def courses_to_choose : ℕ := 4

/-- The number of special courses (A and B) that cannot be chosen together -/
def special_courses : ℕ := 2

/-- The number of different course selection schemes -/
def selection_schemes : ℕ := Nat.choose total_courses courses_to_choose - 
  (Nat.choose special_courses special_courses * Nat.choose (total_courses - special_courses) (courses_to_choose - special_courses))

theorem course_selection_count : selection_schemes = 25 := by
  sorry

end course_selection_count_l2198_219895


namespace system_equation_solution_l2198_219832

theorem system_equation_solution (m : ℝ) : 
  (∃ x y : ℝ, x - y = m + 2 ∧ x + 3*y = m ∧ x + y = -2) → m = -3 := by
  sorry

end system_equation_solution_l2198_219832


namespace arithmetic_sequence_sum_l2198_219801

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 4 + a 7 = 39 →
  a 2 + a 5 + a 8 = 33 →
  a 5 + a 8 + a 11 = 15 := by
sorry

end arithmetic_sequence_sum_l2198_219801


namespace apple_cost_18_pounds_l2198_219807

/-- The cost of apples given a specific rate and weight -/
def appleCost (rate : ℚ) (rateWeight : ℚ) (weight : ℚ) : ℚ :=
  (rate * weight) / rateWeight

/-- Theorem: The cost of 18 pounds of apples at a rate of $6 for 6 pounds is $18 -/
theorem apple_cost_18_pounds :
  appleCost 6 6 18 = 18 := by
  sorry

end apple_cost_18_pounds_l2198_219807


namespace sqrt_plus_arcsin_equals_pi_half_l2198_219851

theorem sqrt_plus_arcsin_equals_pi_half (x : ℝ) :
  Real.sqrt (x * (x + 1)) + Real.arcsin (Real.sqrt (x^2 + x + 1)) = π / 2 := by
  sorry

end sqrt_plus_arcsin_equals_pi_half_l2198_219851


namespace max_score_is_31_l2198_219817

/-- Represents a problem-solving robot with a limited IQ balance. -/
structure Robot where
  iq : ℕ

/-- Represents a problem with a score. -/
structure Problem where
  score : ℕ

/-- Calculates the maximum achievable score for a robot solving a set of problems. -/
def maxAchievableScore (initialIQ : ℕ) (problems : List Problem) : ℕ :=
  sorry

/-- The theorem stating the maximum achievable score for the given conditions. -/
theorem max_score_is_31 :
  let initialIQ := 25
  let problems := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map Problem.mk
  maxAchievableScore initialIQ problems = 31 := by
  sorry

end max_score_is_31_l2198_219817


namespace max_l_shapes_in_grid_l2198_219805

/-- Represents a 6x6 grid --/
def Grid := Fin 6 → Fin 6 → Bool

/-- An L-shape tetromino --/
structure LShape :=
  (position : Fin 6 × Fin 6)
  (orientation : Fin 4)

/-- Checks if an L-shape is within the grid bounds --/
def isWithinBounds (l : LShape) : Bool :=
  sorry

/-- Checks if two L-shapes overlap --/
def doOverlap (l1 l2 : LShape) : Bool :=
  sorry

/-- Checks if a set of L-shapes is valid (within bounds and non-overlapping) --/
def isValidPlacement (shapes : List LShape) : Bool :=
  sorry

/-- The main theorem stating the maximum number of L-shapes in a 6x6 grid --/
theorem max_l_shapes_in_grid :
  ∃ (shapes : List LShape),
    shapes.length = 4 ∧
    isValidPlacement shapes ∧
    ∀ (other_shapes : List LShape),
      isValidPlacement other_shapes →
      other_shapes.length ≤ 4 :=
sorry

end max_l_shapes_in_grid_l2198_219805


namespace function_root_implies_parameter_range_l2198_219857

theorem function_root_implies_parameter_range (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ a^2 * x^2 - 2*a*x + 1 = 0) →
  a > 1 := by sorry

end function_root_implies_parameter_range_l2198_219857


namespace only_physical_education_survey_census_suitable_physical_education_survey_census_suitable_light_bulb_survey_not_census_suitable_tv_program_survey_not_census_suitable_national_height_survey_not_census_suitable_l2198_219800

/-- Represents a survey --/
inductive Survey
  | PhysicalEducationScores
  | LightBulbLifespan
  | TVProgramPreferences
  | NationalStudentHeight

/-- Checks if a survey satisfies the conditions for a census --/
def isCensusSuitable (s : Survey) : Prop :=
  match s with
  | Survey.PhysicalEducationScores => true
  | _ => false

/-- Theorem stating that only the physical education scores survey is suitable for a census --/
theorem only_physical_education_survey_census_suitable :
  ∀ s : Survey, isCensusSuitable s ↔ s = Survey.PhysicalEducationScores :=
by sorry

/-- Proof that the physical education scores survey is suitable for a census --/
theorem physical_education_survey_census_suitable :
  isCensusSuitable Survey.PhysicalEducationScores :=
by sorry

/-- Proof that the light bulb lifespan survey is not suitable for a census --/
theorem light_bulb_survey_not_census_suitable :
  ¬ isCensusSuitable Survey.LightBulbLifespan :=
by sorry

/-- Proof that the TV program preferences survey is not suitable for a census --/
theorem tv_program_survey_not_census_suitable :
  ¬ isCensusSuitable Survey.TVProgramPreferences :=
by sorry

/-- Proof that the national student height survey is not suitable for a census --/
theorem national_height_survey_not_census_suitable :
  ¬ isCensusSuitable Survey.NationalStudentHeight :=
by sorry

end only_physical_education_survey_census_suitable_physical_education_survey_census_suitable_light_bulb_survey_not_census_suitable_tv_program_survey_not_census_suitable_national_height_survey_not_census_suitable_l2198_219800


namespace matrix_power_2023_l2198_219839

def A : Matrix (Fin 2) (Fin 2) ℕ :=
  !![1, 0;
     2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1,    0;
                4046, 1] := by
  sorry

end matrix_power_2023_l2198_219839


namespace slower_train_speed_l2198_219827

/-- Proves that the speed of the slower train is 36 kmph given the conditions of the problem -/
theorem slower_train_speed 
  (faster_speed : ℝ) 
  (faster_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : faster_speed = 72) 
  (h2 : faster_length = 180) 
  (h3 : crossing_time = 18) : 
  ∃ (slower_speed : ℝ), slower_speed = 36 ∧ 
    faster_length = (faster_speed - slower_speed) * (5/18) * crossing_time :=
sorry

end slower_train_speed_l2198_219827


namespace closest_point_on_line_l2198_219838

/-- The line equation y = 2x - 4 -/
def line_equation (x : ℝ) : ℝ := 2 * x - 4

/-- The point we're finding the closest point to -/
def given_point : ℝ × ℝ := (3, 1)

/-- The claimed closest point on the line -/
def closest_point : ℝ × ℝ := (2.6, 1.2)

/-- Theorem stating that the closest_point is on the line and is the closest to given_point -/
theorem closest_point_on_line :
  (line_equation closest_point.1 = closest_point.2) ∧
  ∀ (p : ℝ × ℝ), (line_equation p.1 = p.2) →
    (closest_point.1 - given_point.1)^2 + (closest_point.2 - given_point.2)^2 ≤
    (p.1 - given_point.1)^2 + (p.2 - given_point.2)^2 :=
by sorry

end closest_point_on_line_l2198_219838


namespace sqrt_a_sqrt_a_sqrt_a_l2198_219893

theorem sqrt_a_sqrt_a_sqrt_a (a : ℝ) (ha : a > 0) : 
  Real.sqrt (a * Real.sqrt a * Real.sqrt a) = a := by sorry

end sqrt_a_sqrt_a_sqrt_a_l2198_219893


namespace inequality_range_theorem_l2198_219888

theorem inequality_range_theorem (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 5, |2 - x| + |x + 1| ≤ a) ↔ a ∈ Set.Ici 9 := by
sorry

end inequality_range_theorem_l2198_219888


namespace theta_range_l2198_219828

theorem theta_range (θ : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 * Real.cos θ - x*(1-x) + (1-x)^2 * Real.sin θ > 0) →
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12 :=
by sorry

end theta_range_l2198_219828


namespace volunteer_selection_theorem_l2198_219854

def male_students : ℕ := 5
def female_students : ℕ := 4
def total_volunteers : ℕ := 3
def schools : ℕ := 3

def selection_plans : ℕ := 420

theorem volunteer_selection_theorem :
  (male_students.choose 2 * female_students.choose 1 +
   male_students.choose 1 * female_students.choose 2) * schools.factorial = selection_plans :=
by sorry

end volunteer_selection_theorem_l2198_219854


namespace two_members_absent_l2198_219864

/-- Represents a trivia team with its properties and scoring. -/
structure TriviaTeam where
  totalMembers : ℕ
  pointsPerMember : ℕ
  totalPoints : ℕ

/-- Calculates the number of members who didn't show up for a trivia game. -/
def membersAbsent (team : TriviaTeam) : ℕ :=
  team.totalMembers - (team.totalPoints / team.pointsPerMember)

/-- Theorem stating that for the given trivia team, 2 members didn't show up. -/
theorem two_members_absent (team : TriviaTeam)
  (h1 : team.totalMembers = 5)
  (h2 : team.pointsPerMember = 6)
  (h3 : team.totalPoints = 18) :
  membersAbsent team = 2 := by
  sorry

#eval membersAbsent { totalMembers := 5, pointsPerMember := 6, totalPoints := 18 }

end two_members_absent_l2198_219864


namespace miranda_savings_l2198_219863

/-- Represents an employee at the Cheesecake factory -/
structure Employee where
  name : String
  savingsFraction : ℚ

/-- Calculates the weekly salary for an employee -/
def weeklySalary (hourlyRate : ℚ) (hoursPerDay : ℕ) (daysPerWeek : ℕ) : ℚ :=
  hourlyRate * hoursPerDay * daysPerWeek

/-- Calculates the savings for an employee over a given number of weeks -/
def savings (e : Employee) (salary : ℚ) (weeks : ℕ) : ℚ :=
  e.savingsFraction * salary * weeks

/-- Theorem: Miranda saves 1/2 of her salary -/
theorem miranda_savings
  (hourlyRate : ℚ)
  (hoursPerDay daysPerWeek weeks : ℕ)
  (robby jaylen miranda : Employee)
  (h1 : hourlyRate = 10)
  (h2 : hoursPerDay = 10)
  (h3 : daysPerWeek = 5)
  (h4 : weeks = 4)
  (h5 : robby.savingsFraction = 2/5)
  (h6 : jaylen.savingsFraction = 3/5)
  (h7 : savings robby (weeklySalary hourlyRate hoursPerDay daysPerWeek) weeks +
        savings jaylen (weeklySalary hourlyRate hoursPerDay daysPerWeek) weeks +
        savings miranda (weeklySalary hourlyRate hoursPerDay daysPerWeek) weeks = 3000) :
  miranda.savingsFraction = 1/2 := by
  sorry

end miranda_savings_l2198_219863


namespace cyclic_fraction_sum_l2198_219882

theorem cyclic_fraction_sum (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a / b + b / c + c / a = 100) : 
  b / a + c / b + a / c = -103 := by
  sorry

end cyclic_fraction_sum_l2198_219882


namespace dividend_in_terms_of_a_l2198_219866

theorem dividend_in_terms_of_a (a : ℝ) :
  let divisor := 25 * quotient
  let divisor' := 7 * remainder
  let quotient_minus_remainder := 15
  let remainder := 3 * a
  let dividend := divisor * quotient + remainder
  dividend = 225 * a^2 + 1128 * a + 5625 := by
  sorry

end dividend_in_terms_of_a_l2198_219866


namespace dinner_cost_calculation_l2198_219846

theorem dinner_cost_calculation (total_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
    (h_total : total_cost = 27.5)
    (h_tax : tax_rate = 0.1)
    (h_tip : tip_rate = 0.15) : 
  ∃ (base_cost : ℝ), 
    base_cost * (1 + tax_rate + tip_rate) = total_cost ∧ 
    base_cost = 22 := by
  sorry

end dinner_cost_calculation_l2198_219846


namespace inequality_transformation_l2198_219818

theorem inequality_transformation (a b : ℝ) (h : a > b) : -3 * a < -3 * b := by
  sorry

end inequality_transformation_l2198_219818


namespace complex_roots_count_l2198_219860

theorem complex_roots_count : ∃! (S : Finset ℂ), 
  (∀ z ∈ S, Complex.abs z < 30 ∧ Complex.exp z = (z - 1) / (z + 1)) ∧ 
  Finset.card S = 10 := by
  sorry

end complex_roots_count_l2198_219860


namespace complex_equation_solution_l2198_219877

theorem complex_equation_solution (a : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : (1 + a*i)*i = 3 + i) : a = -3 := by
  sorry

end complex_equation_solution_l2198_219877


namespace square_root_sum_equals_five_l2198_219819

theorem square_root_sum_equals_five : 
  Real.sqrt ((5 / 2 - 3 * Real.sqrt 3 / 2) ^ 2) + Real.sqrt ((5 / 2 + 3 * Real.sqrt 3 / 2) ^ 2) = 5 := by
  sorry

end square_root_sum_equals_five_l2198_219819


namespace total_balloons_l2198_219824

theorem total_balloons (tom_balloons sara_balloons : ℕ) 
  (h1 : tom_balloons = 9) 
  (h2 : sara_balloons = 8) : 
  tom_balloons + sara_balloons = 17 := by
sorry

end total_balloons_l2198_219824


namespace fraction_evaluation_l2198_219884

theorem fraction_evaluation (a b : ℝ) (h : a^2 + b^2 ≠ 0) :
  (a^4 + b^4) / (a^2 + b^2) = a^2 + b^2 - (2 * a^2 * b^2) / (a^2 + b^2) := by
  sorry

end fraction_evaluation_l2198_219884


namespace ellipse_slope_product_ellipse_fixed_point_l2198_219892

/-- Represents an ellipse with eccentricity √6/3 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_ecc : (a^2 - b^2) / a^2 = 2/3

/-- A point on the ellipse -/
def PointOnEllipse (e : Ellipse) := {p : ℝ × ℝ // p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1}

theorem ellipse_slope_product (e : Ellipse) (M A B : PointOnEllipse e) 
  (h_sym : A.val = (-B.val.1, -B.val.2)) :
  let k₁ := (A.val.2 - M.val.2) / (A.val.1 - M.val.1)
  let k₂ := (B.val.2 - M.val.2) / (B.val.1 - M.val.1)
  k₁ * k₂ = -1/3 := by sorry

theorem ellipse_fixed_point (e : Ellipse) (M A B : PointOnEllipse e)
  (h_M : M.val = (0, 1))
  (h_slopes : let k₁ := (A.val.2 - M.val.2) / (A.val.1 - M.val.1)
              let k₂ := (B.val.2 - M.val.2) / (B.val.1 - M.val.1)
              k₁ + k₂ = 3) :
  ∃ (k m : ℝ), A.val.2 = k * A.val.1 + m ∧ 
                B.val.2 = k * B.val.1 + m ∧ 
                -2/3 = k * (-2/3) + m ∧ 
                -1 = k * (-2/3) + m := by sorry

end ellipse_slope_product_ellipse_fixed_point_l2198_219892


namespace amoeba_survival_l2198_219842

/-- Represents the state of an amoeba with pseudopods and nuclei -/
structure Amoeba where
  pseudopods : Int
  nuclei : Int

/-- Mutation function for an amoeba -/
def mutate (a : Amoeba) : Amoeba :=
  { pseudopods := 2 * a.pseudopods - a.nuclei,
    nuclei := 2 * a.nuclei - a.pseudopods }

/-- Predicate to check if an amoeba is alive -/
def isAlive (a : Amoeba) : Prop :=
  a.pseudopods ≥ 0 ∧ a.nuclei ≥ 0

/-- Theorem stating that only amoebas with equal initial pseudopods and nuclei survive indefinitely -/
theorem amoeba_survival (a : Amoeba) :
  (∀ n : ℕ, isAlive ((mutate^[n]) a)) ↔ a.pseudopods = a.nuclei :=
sorry

end amoeba_survival_l2198_219842


namespace average_timing_error_l2198_219834

def total_watches : ℕ := 10

def timing_errors : List ℕ := [0, 1, 2, 3]
def error_frequencies : List ℕ := [3, 4, 2, 1]

def average_error : ℚ := 1.1

theorem average_timing_error :
  (List.sum (List.zipWith (· * ·) timing_errors error_frequencies) : ℚ) / total_watches = average_error :=
sorry

end average_timing_error_l2198_219834


namespace root_sum_squares_implies_h_abs_one_l2198_219847

theorem root_sum_squares_implies_h_abs_one (h : ℝ) : 
  (∃ r s : ℝ, r^2 + 6*h*r + 8 = 0 ∧ s^2 + 6*h*s + 8 = 0 ∧ r^2 + s^2 = 20) → 
  |h| = 1 := by
sorry

end root_sum_squares_implies_h_abs_one_l2198_219847


namespace line_slope_one_m_value_l2198_219835

/-- Given a line passing through points P(-2, m) and Q(m, 4) with a slope of 1, prove that m = 1 -/
theorem line_slope_one_m_value (m : ℝ) : 
  (4 - m) / (m + 2) = 1 → m = 1 := by
  sorry

end line_slope_one_m_value_l2198_219835


namespace convex_polygon_interior_angles_l2198_219852

theorem convex_polygon_interior_angles (n : ℕ) :
  n > 2 →
  (∃ x : ℕ, (n - 2) * 180 - x = 2000) →
  n = 14 :=
by sorry

end convex_polygon_interior_angles_l2198_219852


namespace max_trees_in_garden_l2198_219875

def garden_width : ℝ := 27.9
def tree_interval : ℝ := 3.1

theorem max_trees_in_garden : 
  ⌊garden_width / tree_interval⌋ = 9 := by sorry

end max_trees_in_garden_l2198_219875


namespace shortest_path_length_l2198_219810

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 7.5)^2 + (y - 10)^2 = 36
def circle2 (x y : ℝ) : Prop := (x - 15)^2 + (y - 5)^2 = 16

-- Define a path that avoids the circles
def valid_path (p : ℝ → ℝ × ℝ) : Prop :=
  (p 0 = (0, 0)) ∧ 
  (p 1 = (15, 20)) ∧ 
  ∀ t ∈ (Set.Icc 0 1), ¬(circle1 (p t).1 (p t).2) ∧ ¬(circle2 (p t).1 (p t).2)

-- Define the length of a path
def path_length (p : ℝ → ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem shortest_path_length :
  ∃ p, valid_path p ∧ 
    path_length p = 30.6 + 5 * Real.pi / 3 ∧
    ∀ q, valid_path q → path_length p ≤ path_length q :=
sorry

end shortest_path_length_l2198_219810


namespace grocer_average_sale_l2198_219873

/-- Given the sales figures for five months, prove that the average sale is 7800 --/
theorem grocer_average_sale
  (sale1 : ℕ) (sale2 : ℕ) (sale3 : ℕ) (sale4 : ℕ) (sale5 : ℕ)
  (h1 : sale1 = 5700)
  (h2 : sale2 = 8550)
  (h3 : sale3 = 6855)
  (h4 : sale4 = 3850)
  (h5 : sale5 = 14045) :
  (sale1 + sale2 + sale3 + sale4 + sale5) / 5 = 7800 := by
  sorry

#check grocer_average_sale

end grocer_average_sale_l2198_219873


namespace books_borrowed_by_lunch_correct_l2198_219879

/-- Represents the number of books borrowed by lunchtime -/
def books_borrowed_by_lunch : ℕ := 50

/-- Represents the initial number of books on the shelf -/
def initial_books : ℕ := 100

/-- Represents the number of books added after lunch -/
def books_added : ℕ := 40

/-- Represents the number of books borrowed by evening -/
def books_borrowed_by_evening : ℕ := 30

/-- Represents the number of books remaining by evening -/
def books_remaining : ℕ := 60

/-- Proves that the number of books borrowed by lunchtime is correct -/
theorem books_borrowed_by_lunch_correct :
  initial_books - books_borrowed_by_lunch + books_added - books_borrowed_by_evening = books_remaining :=
by sorry


end books_borrowed_by_lunch_correct_l2198_219879


namespace monic_quadratic_with_complex_root_l2198_219856

theorem monic_quadratic_with_complex_root :
  ∃! (a b : ℝ), ∀ (x : ℂ), x^2 + a*x + b = 0 ↔ x = 2 - I ∨ x = 2 + I :=
by sorry

end monic_quadratic_with_complex_root_l2198_219856


namespace prob_less_than_8_l2198_219844

/-- The probability of scoring less than 8 in a single shot, given the probabilities of hitting the 10, 9, and 8 rings. -/
theorem prob_less_than_8 (p10 p9 p8 : ℝ) (h1 : p10 = 0.3) (h2 : p9 = 0.3) (h3 : p8 = 0.2) :
  1 - p10 - p9 - p8 = 0.2 := by
  sorry

end prob_less_than_8_l2198_219844


namespace max_red_tiles_100x100_l2198_219885

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents the number of colors used for tiling -/
def num_colors : ℕ := 4

/-- Defines the property that no two tiles of the same color touch each other -/
def no_adjacent_same_color (g : Grid) : Prop := sorry

/-- The maximum number of tiles of a single color in the grid -/
def max_single_color_tiles (g : Grid) : ℕ := (g.size ^ 2) / 4

/-- Theorem stating the maximum number of red tiles in a 100x100 grid -/
theorem max_red_tiles_100x100 (g : Grid) (h1 : g.size = 100) (h2 : no_adjacent_same_color g) : 
  max_single_color_tiles g = 2500 := by sorry

end max_red_tiles_100x100_l2198_219885


namespace unique_solution_cube_difference_l2198_219814

theorem unique_solution_cube_difference (x y : ℤ) :
  (x + 2)^4 - x^4 = y^3 ↔ x = -1 ∧ y = 0 := by
  sorry

end unique_solution_cube_difference_l2198_219814


namespace marble_selection_problem_l2198_219823

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem marble_selection_problem :
  let total_marbles : ℕ := 15
  let required_marbles : ℕ := 2
  let marbles_to_choose : ℕ := 5
  let remaining_marbles : ℕ := total_marbles - required_marbles
  let additional_marbles : ℕ := marbles_to_choose - required_marbles
  choose remaining_marbles additional_marbles = 286 :=
by sorry

end marble_selection_problem_l2198_219823


namespace square_sum_difference_l2198_219837

theorem square_sum_difference (n : ℕ) : 
  (2*n+1)^2 - (2*n-1)^2 + (2*n-3)^2 - (2*n-5)^2 + (2*n-7)^2 - (2*n-9)^2 + 
  (2*n-11)^2 - (2*n-13)^2 + (2*n-15)^2 - (2*n-17)^2 + (2*n-19)^2 - 
  (2*n-21)^2 + (2*n-23)^2 - (2*n-25)^2 + (2*n-27)^2 = 389 :=
by sorry

end square_sum_difference_l2198_219837


namespace same_color_probability_l2198_219849

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (purple : Nat)
  (green : Nat)
  (orange : Nat)
  (glittery : Nat)
  (total : Nat)
  (h1 : purple + green + orange + glittery = total)
  (h2 : total = 30)

/-- The probability of rolling the same color on two identical colored dice -/
def sameProbability (d : ColoredDie) : Rat :=
  (d.purple^2 + d.green^2 + d.orange^2 + d.glittery^2) / d.total^2

/-- Two 30-sided dice with specified colored sides -/
def twoDice : ColoredDie :=
  { purple := 6
    green := 10
    orange := 12
    glittery := 2
    total := 30
    h1 := by rfl
    h2 := by rfl }

theorem same_color_probability :
  sameProbability twoDice = 71 / 225 := by
  sorry

end same_color_probability_l2198_219849


namespace laptop_sale_price_l2198_219867

def original_price : ℝ := 1000.00
def discount1 : ℝ := 0.10
def discount2 : ℝ := 0.20
def discount3 : ℝ := 0.15

theorem laptop_sale_price :
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 612.00 := by
sorry

end laptop_sale_price_l2198_219867


namespace problem_statement_l2198_219853

theorem problem_statement (x : ℝ) (h : x + 1/x = 6) :
  (x - 3)^2 + 36/((x - 3)^2) = 12.5 := by
  sorry

end problem_statement_l2198_219853


namespace lock_combination_solution_l2198_219848

/-- Represents a digit in base 12 --/
def Digit12 := Fin 12

/-- Represents a mapping from letters to digits --/
def LetterMapping := Char → Digit12

/-- Converts a number in base 12 to base 10 --/
def toBase10 (x : ℕ) : ℕ := x

/-- Checks if all characters in a string are distinct --/
def allDistinct (s : String) : Prop := sorry

/-- Converts a string to a number using the given mapping --/
def stringToNumber (s : String) (m : LetterMapping) : ℕ := sorry

/-- The main theorem --/
theorem lock_combination_solution :
  ∃! (m : LetterMapping),
    (allDistinct "VENUSISNEAR") ∧
    (stringToNumber "VENUS" m + stringToNumber "IS" m + stringToNumber "NEAR" m =
     stringToNumber "SUN" m) ∧
    (toBase10 (stringToNumber "SUN" m) = 655) := by sorry

end lock_combination_solution_l2198_219848


namespace not_divides_power_minus_one_l2198_219871

theorem not_divides_power_minus_one (n : ℕ) (h : n > 1) : ¬(n ∣ (2^n - 1)) := by
  sorry

end not_divides_power_minus_one_l2198_219871


namespace janet_tile_savings_l2198_219841

/-- Calculates the cost difference between two tile options for a given wall area and tile density -/
def tile_cost_difference (
  wall1_length wall1_width wall2_length wall2_width : ℝ)
  (tiles_per_sqft : ℝ)
  (turquoise_cost purple_cost : ℝ) : ℝ :=
  let total_area := wall1_length * wall1_width + wall2_length * wall2_width
  let total_tiles := total_area * tiles_per_sqft
  let cost_diff_per_tile := turquoise_cost - purple_cost
  total_tiles * cost_diff_per_tile

/-- The cost difference between turquoise and purple tiles for Janet's bathroom -/
theorem janet_tile_savings : 
  tile_cost_difference 5 8 7 8 4 13 11 = 768 := by
  sorry

end janet_tile_savings_l2198_219841


namespace perfume_tax_rate_l2198_219826

theorem perfume_tax_rate (price_before_tax : ℝ) (total_price : ℝ) : price_before_tax = 92 → total_price = 98.90 → (total_price - price_before_tax) / price_before_tax * 100 = 7.5 := by
  sorry

end perfume_tax_rate_l2198_219826


namespace arithmetic_mean_of_three_digit_multiples_of_8_l2198_219862

theorem arithmetic_mean_of_three_digit_multiples_of_8 :
  let first := 104  -- First three-digit multiple of 8
  let last := 992   -- Last three-digit multiple of 8
  let step := 8     -- Difference between consecutive multiples
  let count := (last - first) / step + 1  -- Number of terms in the sequence
  let sum := count * (first + last) / 2   -- Sum of arithmetic sequence
  sum / count = 548 := by
sorry

end arithmetic_mean_of_three_digit_multiples_of_8_l2198_219862


namespace total_lemons_l2198_219883

/-- Given the number of lemons for each person in terms of x, prove the total number of lemons. -/
theorem total_lemons (x : ℝ) :
  let L := x
  let J := x + 6
  let A := (4/3) * (x + 6)
  let E := (2/3) * (x + 6)
  let I := 2 * (2/3) * (x + 6)
  let N := (3/4) * x
  let O := (3/5) * (4/3) * (x + 6)
  L + J + A + E + I + N + O = (413/60) * x + 30.8 := by
sorry

end total_lemons_l2198_219883


namespace floor_sqrt_120_l2198_219836

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end floor_sqrt_120_l2198_219836


namespace circle_tangent_range_l2198_219869

/-- The range of k values for which two tangent lines can be drawn from (1, 2) to the circle x^2 + y^2 + kx + 2y + k^2 - 15 = 0 -/
theorem circle_tangent_range (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + k*x + 2*y + k^2 - 15 = 0) ∧ 
  (∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    (∃ (x₁ y₁ x₂ y₂ : ℝ), 
      x₁^2 + y₁^2 + k*x₁ + 2*y₁ + k^2 - 15 = 0 ∧
      x₂^2 + y₂^2 + k*x₂ + 2*y₂ + k^2 - 15 = 0 ∧
      (y₁ - 2) = t₁ * (x₁ - 1) ∧
      (y₂ - 2) = t₂ * (x₂ - 1))) ↔ 
  (k > -8*Real.sqrt 3/3 ∧ k < -3) ∨ (k > 2 ∧ k < 8*Real.sqrt 3/3) :=
by sorry

end circle_tangent_range_l2198_219869


namespace product_abcd_l2198_219812

theorem product_abcd (a b c d : ℚ) 
  (eq1 : 4*a - 2*b + 3*c + 5*d = 22)
  (eq2 : 2*(d+c) = b - 2)
  (eq3 : 4*b - c = a + 1)
  (eq4 : c + 1 = 2*d) :
  a * b * c * d = -30751860 / 11338912 := by
  sorry

end product_abcd_l2198_219812


namespace arithmetic_sequence_first_term_l2198_219872

/-- Given an arithmetic sequence {aₙ} with common difference d and a₃₀, find a₁ -/
theorem arithmetic_sequence_first_term
  (a : ℕ → ℚ)  -- The arithmetic sequence
  (d : ℚ)      -- Common difference
  (h1 : d = 3/4)
  (h2 : a 30 = 63/4)  -- a₃₀ = 15 3/4 = 63/4
  (h3 : ∀ n : ℕ, a (n + 1) = a n + d)  -- Definition of arithmetic sequence
  : a 1 = -6 :=
by sorry

end arithmetic_sequence_first_term_l2198_219872


namespace chairs_count_l2198_219803

-- Define the variables
variable (chair_price : ℚ) (table_price : ℚ) (num_chairs : ℕ)

-- Define the conditions
def condition1 (chair_price table_price num_chairs : ℚ) : Prop :=
  num_chairs * chair_price = num_chairs * table_price - 320

def condition2 (chair_price table_price num_chairs : ℚ) : Prop :=
  num_chairs * chair_price = (num_chairs - 5) * table_price

def condition3 (chair_price table_price : ℚ) : Prop :=
  3 * table_price = 5 * chair_price + 48

-- State the theorem
theorem chairs_count 
  (h1 : condition1 chair_price table_price num_chairs)
  (h2 : condition2 chair_price table_price num_chairs)
  (h3 : condition3 chair_price table_price) :
  num_chairs = 20 := by
  sorry

end chairs_count_l2198_219803


namespace connect_points_is_valid_l2198_219897

-- Define a type for geometric points
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for geometric drawing operations
inductive DrawingOperation
  | DrawRay (start : Point) (length : ℝ)
  | ConnectPoints (a b : Point)
  | DrawMidpoint (a b : Point)
  | DrawDistance (a b : Point)

-- Define a predicate for valid drawing operations
def IsValidDrawingOperation : DrawingOperation → Prop
  | DrawingOperation.ConnectPoints _ _ => True
  | _ => False

-- Theorem statement
theorem connect_points_is_valid :
  ∀ (a b : Point), IsValidDrawingOperation (DrawingOperation.ConnectPoints a b) :=
by sorry

end connect_points_is_valid_l2198_219897


namespace expression_evaluation_l2198_219865

theorem expression_evaluation : 
  |Real.sqrt 3 - 2| + (Real.pi - Real.sqrt 10)^0 - Real.sqrt 12 = 3 - 3 * Real.sqrt 3 := by
  sorry

end expression_evaluation_l2198_219865


namespace sum_of_digits_of_M_l2198_219850

/-- Represents a four-digit number -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  is_four_digit : 1000 ≤ d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ d1 * 1000 + d2 * 100 + d3 * 10 + d4 < 10000

/-- The value of a four-digit number -/
def FourDigitNumber.value (n : FourDigitNumber) : Nat :=
  n.d1 * 1000 + n.d2 * 100 + n.d3 * 10 + n.d4

/-- The product of digits of a four-digit number -/
def FourDigitNumber.digitProduct (n : FourDigitNumber) : Nat :=
  n.d1 * n.d2 * n.d3 * n.d4

/-- The sum of digits of a four-digit number -/
def FourDigitNumber.digitSum (n : FourDigitNumber) : Nat :=
  n.d1 + n.d2 + n.d3 + n.d4

/-- M is the greatest four-digit number whose digits have a product of 24 -/
def M : FourDigitNumber :=
  sorry

theorem sum_of_digits_of_M :
  M.digitProduct = 24 ∧ 
  (∀ n : FourDigitNumber, n.digitProduct = 24 → n.value ≤ M.value) →
  M.digitSum = 13 :=
sorry

end sum_of_digits_of_M_l2198_219850


namespace range_of_a_l2198_219868

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a+3}

theorem range_of_a (a : ℝ) : A ∪ B a = A ↔ a ≤ -4 ∨ (2 ≤ a ∧ a ≤ 3) ∨ a > 3 := by
  sorry

end range_of_a_l2198_219868


namespace power_of_power_at_three_l2198_219881

theorem power_of_power_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end power_of_power_at_three_l2198_219881


namespace unique_function_solution_l2198_219845

theorem unique_function_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) → 
  (∀ x : ℝ, f x = x) := by
sorry

end unique_function_solution_l2198_219845


namespace binomial_18_10_l2198_219806

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 42328 := by
  sorry

end binomial_18_10_l2198_219806


namespace cricketer_average_score_l2198_219811

theorem cricketer_average_score (total_matches : ℕ) (first_set : ℕ) (second_set : ℕ)
  (avg_first : ℚ) (avg_second : ℚ) :
  total_matches = first_set + second_set →
  first_set = 2 →
  second_set = 3 →
  avg_first = 40 →
  avg_second = 10 →
  (first_set * avg_first + second_set * avg_second) / total_matches = 22 :=
by sorry

end cricketer_average_score_l2198_219811


namespace even_function_extension_l2198_219887

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the function for x < 0
def f_neg (x : ℝ) : ℝ := x * (2 * x - 1)

-- Theorem statement
theorem even_function_extension 
  (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_neg : ∀ x, x < 0 → f x = f_neg x) : 
  ∀ x, x > 0 → f x = x * (2 * x + 1) := by
sorry


end even_function_extension_l2198_219887


namespace simplify_expression_l2198_219899

theorem simplify_expression (w : ℝ) : 3*w + 6*w + 9*w + 12*w + 15*w + 18 = 45*w + 18 := by
  sorry

end simplify_expression_l2198_219899


namespace dishonest_dealer_profit_percentage_l2198_219802

/-- Calculates the profit percentage of a dishonest dealer who uses a reduced weight. -/
theorem dishonest_dealer_profit_percentage 
  (claimed_weight : ℝ) 
  (actual_weight : ℝ) 
  (claimed_weight_positive : claimed_weight > 0)
  (actual_weight_positive : actual_weight > 0)
  (actual_weight_less_than_claimed : actual_weight < claimed_weight) :
  (claimed_weight - actual_weight) / actual_weight * 100 = 
  ((1000 - 780) / 780) * 100 :=
by sorry

end dishonest_dealer_profit_percentage_l2198_219802


namespace triangle_side_values_l2198_219861

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_values :
  ∀ y : ℕ+, 
    (is_valid_triangle 8 11 (y.val ^ 2)) ↔ (y = 2 ∨ y = 3 ∨ y = 4) :=
by sorry

end triangle_side_values_l2198_219861


namespace brick_length_calculation_l2198_219878

-- Define the courtyard dimensions in centimeters
def courtyard_length : ℝ := 2500  -- 25 meters = 2500 cm
def courtyard_width : ℝ := 1600   -- 16 meters = 1600 cm

-- Define the brick properties
def brick_width : ℝ := 10         -- 10 cm
def total_bricks : ℕ := 20000

-- Define the theorem
theorem brick_length_calculation :
  ∃ (brick_length : ℝ),
    brick_length > 0 ∧
    brick_length * brick_width * total_bricks = courtyard_length * courtyard_width ∧
    brick_length = 20 := by
  sorry

end brick_length_calculation_l2198_219878


namespace helmet_costs_and_profit_l2198_219870

/-- Represents the cost and sales information for helmets --/
structure HelmetData where
  costA3B4 : ℕ  -- Cost of 3 type A and 4 type B helmets
  costA6B2 : ℕ  -- Cost of 6 type A and 2 type B helmets
  basePrice : ℝ  -- Base selling price of type A helmet
  baseSales : ℕ  -- Number of helmets sold at base price
  priceIncrement : ℝ  -- Price increment
  salesDecrement : ℕ  -- Sales decrement per price increment

/-- Theorem about helmet costs and profit --/
theorem helmet_costs_and_profit (data : HelmetData)
  (h1 : data.costA3B4 = 288)
  (h2 : data.costA6B2 = 306)
  (h3 : data.basePrice = 50)
  (h4 : data.baseSales = 100)
  (h5 : data.priceIncrement = 5)
  (h6 : data.salesDecrement = 10) :
  ∃ (costA costB : ℕ) (profitFunc : ℝ → ℝ) (maxProfit : ℝ),
    costA = 36 ∧
    costB = 45 ∧
    (∀ x, 50 ≤ x ∧ x ≤ 100 → profitFunc x = -2 * x^2 + 272 * x - 7200) ∧
    maxProfit = 2048 := by
  sorry

end helmet_costs_and_profit_l2198_219870


namespace unique_solution_for_equation_l2198_219829

theorem unique_solution_for_equation : ∃! (x y : ℕ), 
  x < 10 ∧ y < 10 ∧ (10 + x) * (200 + 10 * y + 7) = 5166 := by sorry

end unique_solution_for_equation_l2198_219829


namespace circle_m_range_and_perpendicular_intersection_l2198_219821

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem circle_m_range_and_perpendicular_intersection :
  -- Part 1: If the equation represents a circle, then m ∈ (-∞, 5)
  (∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → m < 5) ∧
  -- Part 2: If the circle intersects the line and OM ⟂ ON, then m = 8/5
  (∀ m : ℝ, 
    (∃ x1 y1 x2 y2 : ℝ, 
      circle_equation x1 y1 m ∧ 
      circle_equation x2 y2 m ∧
      line_equation x1 y1 ∧ 
      line_equation x2 y2 ∧
      perpendicular x1 y1 x2 y2) → 
    m = 8/5) :=
by sorry

end circle_m_range_and_perpendicular_intersection_l2198_219821


namespace smallest_890_multiple_of_18_l2198_219843

def is_digit_890 (d : ℕ) : Prop := d = 8 ∨ d = 9 ∨ d = 0

def all_digits_890 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_digit_890 d

theorem smallest_890_multiple_of_18 :
  ∃! m : ℕ, m > 0 ∧ m % 18 = 0 ∧ all_digits_890 m ∧
  ∀ n : ℕ, n > 0 → n % 18 = 0 → all_digits_890 n → m ≤ n :=
by sorry

end smallest_890_multiple_of_18_l2198_219843


namespace fuel_cost_per_refill_l2198_219816

/-- 
Given the total fuel cost and number of refills, 
calculate the cost of one refilling.
-/
theorem fuel_cost_per_refill 
  (total_cost : ℕ) 
  (num_refills : ℕ) 
  (h1 : total_cost = 63)
  (h2 : num_refills = 3)
  : total_cost / num_refills = 21 := by
  sorry

end fuel_cost_per_refill_l2198_219816


namespace scooter_price_l2198_219830

-- Define the upfront payment and the percentage paid
def upfront_payment : ℝ := 240
def percentage_paid : ℝ := 20

-- State the theorem
theorem scooter_price : 
  (upfront_payment / (percentage_paid / 100)) = 1200 := by
  sorry

end scooter_price_l2198_219830


namespace rosa_pages_last_week_l2198_219822

-- Define the total number of pages called
def total_pages : ℝ := 18.8

-- Define the number of pages called this week
def pages_this_week : ℝ := 8.6

-- Define the number of pages called last week
def pages_last_week : ℝ := total_pages - pages_this_week

-- Theorem to prove
theorem rosa_pages_last_week : pages_last_week = 10.2 := by
  sorry

end rosa_pages_last_week_l2198_219822
