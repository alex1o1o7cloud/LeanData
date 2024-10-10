import Mathlib

namespace expand_expression_l2760_276064

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 * y + 6) = 36 * x + 48 * y + 72 := by
  sorry

end expand_expression_l2760_276064


namespace sqrt_inequality_l2760_276076

theorem sqrt_inequality : Real.sqrt 5 - Real.sqrt 6 < Real.sqrt 6 - Real.sqrt 7 := by
  sorry

end sqrt_inequality_l2760_276076


namespace probability_calculation_l2760_276045

/-- The number of volunteers -/
def num_volunteers : ℕ := 5

/-- The number of venues -/
def num_venues : ℕ := 3

/-- The total number of ways to assign volunteers to venues -/
def total_assignments : ℕ := num_venues ^ num_volunteers

/-- The number of favorable assignments (where each venue has at least one volunteer) -/
def favorable_assignments : ℕ := 150

/-- The probability that each venue has at least one volunteer -/
def probability_all_venues_covered : ℚ := favorable_assignments / total_assignments

theorem probability_calculation :
  probability_all_venues_covered = 50 / 81 :=
sorry

end probability_calculation_l2760_276045


namespace sandy_carrots_l2760_276077

def carrots_problem (initial_carrots : ℕ) (sam_took : ℕ) (sandy_left : ℕ) : Prop :=
  initial_carrots = sam_took + sandy_left

theorem sandy_carrots : ∃ initial_carrots : ℕ, carrots_problem initial_carrots 3 3 :=
  sorry

end sandy_carrots_l2760_276077


namespace no_perfect_square_ends_2012_l2760_276000

theorem no_perfect_square_ends_2012 : ∀ a : ℤ, ¬(∃ k : ℤ, a^2 = 10000 * k + 2012) := by
  sorry

end no_perfect_square_ends_2012_l2760_276000


namespace max_visible_sum_three_cubes_l2760_276022

/-- Represents a cube with six faces numbered 1, 3, 5, 7, 9, 11 -/
def Cube := Fin 6 → Nat

/-- The set of numbers on each cube -/
def cubeNumbers : Finset Nat := {1, 3, 5, 7, 9, 11}

/-- A function to calculate the sum of visible faces when stacking cubes -/
def visibleSum (c1 c2 c3 : Cube) : Nat := sorry

/-- The theorem stating the maximum visible sum when stacking three cubes -/
theorem max_visible_sum_three_cubes :
  ∃ (c1 c2 c3 : Cube),
    (∀ (i : Fin 6), c1 i ∈ cubeNumbers ∧ c2 i ∈ cubeNumbers ∧ c3 i ∈ cubeNumbers) ∧
    (∀ (c1' c2' c3' : Cube),
      (∀ (i : Fin 6), c1' i ∈ cubeNumbers ∧ c2' i ∈ cubeNumbers ∧ c3' i ∈ cubeNumbers) →
      visibleSum c1' c2' c3' ≤ visibleSum c1 c2 c3) ∧
    visibleSum c1 c2 c3 = 101 :=
  sorry

end max_visible_sum_three_cubes_l2760_276022


namespace middle_truncated_cone_volume_middle_truncated_cone_volume_is_7V_div_27_l2760_276079

/-- Given a cone with volume V whose height is divided into three equal parts by planes parallel to the base, the volume of the middle truncated cone is 7V/27. -/
theorem middle_truncated_cone_volume (V : ℝ) (h : V > 0) : ℝ :=
  let cone_volume := V
  let height_parts := 3
  let middle_truncated_cone_volume := (7 : ℝ) / 27 * V
  middle_truncated_cone_volume

/-- The volume of the middle truncated cone is 7V/27 -/
theorem middle_truncated_cone_volume_is_7V_div_27 (V : ℝ) (h : V > 0) :
  middle_truncated_cone_volume V h = (7 : ℝ) / 27 * V := by
  sorry

end middle_truncated_cone_volume_middle_truncated_cone_volume_is_7V_div_27_l2760_276079


namespace largest_divisor_of_m_squared_minus_4n_squared_l2760_276040

theorem largest_divisor_of_m_squared_minus_4n_squared (m n : ℤ) 
  (h_m_odd : Odd m) (h_n_odd : Odd n) (h_m_gt_n : m > n) : 
  (∀ k : ℤ, k ∣ (m^2 - 4*n^2) → k = 1 ∨ k = -1) :=
sorry

end largest_divisor_of_m_squared_minus_4n_squared_l2760_276040


namespace clerical_staff_fraction_l2760_276034

theorem clerical_staff_fraction (total_employees : ℕ) (f : ℚ) : 
  total_employees = 3600 →
  (2/3 : ℚ) * (f * total_employees) = (1/4 : ℚ) * (total_employees - (1/3 : ℚ) * (f * total_employees)) →
  f = 1/3 := by
  sorry

end clerical_staff_fraction_l2760_276034


namespace farm_distance_problem_l2760_276074

/-- Represents the distances between three farms -/
structure FarmDistances where
  x : ℝ  -- Distance between first and second farms
  y : ℝ  -- Distance between second and third farms
  z : ℝ  -- Distance between first and third farms

/-- Theorem stating the conditions and results for the farm distance problem -/
theorem farm_distance_problem (a : ℝ) : 
  ∃ (d : FarmDistances), 
    d.x + d.y = 4 * d.z ∧                   -- Condition 1
    d.z + d.y = d.x + a ∧                   -- Condition 2
    d.x + d.z = 85 ∧                        -- Condition 3
    0 < a ∧ a < 85 ∧                        -- Interval for a
    d.x = (340 - a) / 6 ∧                   -- Distance x
    d.y = (2 * a + 85) / 3 ∧                -- Distance y
    d.z = (170 + a) / 6 ∧                   -- Distance z
    d.x + d.y > d.z ∧ d.y + d.z > d.x ∧ d.z + d.x > d.y -- Triangle inequality
    := by sorry

end farm_distance_problem_l2760_276074


namespace knowledge_competition_probabilities_l2760_276065

/-- Represents the outcome of answering a question -/
inductive Answer
| Correct
| Incorrect

/-- Represents the state of a contestant in the competition -/
structure ContestantState where
  score : ℕ
  questions_answered : ℕ

/-- Represents the probabilities of correctly answering each question -/
structure QuestionProbabilities where
  pA : ℚ
  pB : ℚ
  pC : ℚ
  pD : ℚ

/-- Updates the contestant's state based on their answer -/
def updateState (state : ContestantState) (answer : Answer) (questionNumber : ℕ) : ContestantState :=
  match answer with
  | Answer.Correct =>
    let points := match questionNumber with
      | 1 => 1
      | 2 => 2
      | 3 => 3
      | 4 => 6
      | _ => 0
    { score := state.score + points, questions_answered := state.questions_answered + 1 }
  | Answer.Incorrect =>
    { score := state.score - 2, questions_answered := state.questions_answered + 1 }

/-- Checks if a contestant is eliminated based on their current state -/
def isEliminated (state : ContestantState) : Bool :=
  state.score < 8 || (state.questions_answered = 4 && state.score < 14)

/-- Checks if a contestant has advanced to the next round -/
def hasAdvanced (state : ContestantState) : Bool :=
  state.score ≥ 14

/-- Main theorem statement -/
theorem knowledge_competition_probabilities 
  (probs : QuestionProbabilities)
  (h1 : probs.pA = 3/4)
  (h2 : probs.pB = 1/2)
  (h3 : probs.pC = 1/3)
  (h4 : probs.pD = 1/4) :
  ∃ (advanceProb : ℚ) (ξDist : ℕ → ℚ) (ξExpected : ℚ),
    (advanceProb = 1/2) ∧ 
    (ξDist 2 = 1/8) ∧ (ξDist 3 = 1/2) ∧ (ξDist 4 = 3/8) ∧
    (ξExpected = 7/4) := by
  sorry

end knowledge_competition_probabilities_l2760_276065


namespace bicyclist_speed_increase_l2760_276027

theorem bicyclist_speed_increase (x : ℝ) : 
  (1 + x) * 1.1 = 1.43 → x = 0.3 := by sorry

end bicyclist_speed_increase_l2760_276027


namespace rotation_surface_area_theorem_l2760_276047

/-- Represents a plane curve -/
structure PlaneCurve where
  -- Add necessary fields for a plane curve

/-- Calculates the length of a plane curve -/
def curveLength (c : PlaneCurve) : ℝ :=
  sorry

/-- Calculates the distance of the center of gravity from the axis of rotation -/
def centerOfGravityDistance (c : PlaneCurve) : ℝ :=
  sorry

/-- Calculates the surface area generated by rotating a plane curve around an axis -/
def rotationSurfaceArea (c : PlaneCurve) : ℝ :=
  sorry

/-- Theorem: The surface area generated by rotating an arbitrary plane curve around an axis
    is equal to 2π times the distance of the center of gravity from the axis
    times the length of the curve -/
theorem rotation_surface_area_theorem (c : PlaneCurve) :
  rotationSurfaceArea c = 2 * Real.pi * centerOfGravityDistance c * curveLength c :=
sorry

end rotation_surface_area_theorem_l2760_276047


namespace square_circle_area_ratio_l2760_276018

theorem square_circle_area_ratio (a r : ℝ) (h : a > 0) (k : r > 0) : 
  4 * a = 2 * 2 * Real.pi * r → a^2 / (Real.pi * r^2) = Real.pi := by
  sorry

end square_circle_area_ratio_l2760_276018


namespace determinant_example_l2760_276028

/-- Definition of a second-order determinant -/
def second_order_determinant (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem: The determinant of the matrix [[2, 1], [-3, 4]] is 11 -/
theorem determinant_example : second_order_determinant 2 (-3) 1 4 = 11 := by
  sorry

end determinant_example_l2760_276028


namespace min_value_trigonometric_expression_l2760_276089

theorem min_value_trigonometric_expression (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  1 / (Real.sin θ)^2 + 9 / (Real.cos θ)^2 ≥ 16 := by
sorry

end min_value_trigonometric_expression_l2760_276089


namespace modular_congruence_unique_solution_l2760_276093

theorem modular_congruence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n < 25 ∧ 24938 ≡ n [ZMOD 25] ∧ n = 13 := by
  sorry

end modular_congruence_unique_solution_l2760_276093


namespace shark_count_l2760_276005

theorem shark_count (cape_may_sharks : ℕ) (other_beach_sharks : ℕ) : 
  cape_may_sharks = 32 → 
  cape_may_sharks = 2 * other_beach_sharks + 8 → 
  other_beach_sharks = 12 := by
sorry

end shark_count_l2760_276005


namespace divisibility_property_l2760_276019

theorem divisibility_property (m : ℕ) (hm : m > 0) :
  ∃ q : Polynomial ℤ, (x + 1)^(2*m) - x^(2*m) - 2*x - 1 = x * (x + 1) * (2*x + 1) * q :=
sorry

end divisibility_property_l2760_276019


namespace fraction_simplification_l2760_276020

theorem fraction_simplification : (4 * 5) / 10 = 2 := by
  sorry

end fraction_simplification_l2760_276020


namespace abc_inequality_l2760_276053

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) := by
  sorry

end abc_inequality_l2760_276053


namespace polar_coordinates_of_point_l2760_276052

theorem polar_coordinates_of_point (x y : ℝ) (r θ : ℝ) :
  x = -Real.sqrt 3 ∧ y = -1 →
  r = 2 ∧ θ = 7 * Real.pi / 6 →
  x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ r ≥ 0 :=
by sorry

end polar_coordinates_of_point_l2760_276052


namespace combined_distance_is_91261_136_l2760_276037

/-- The combined distance traveled by friends in feet -/
def combined_distance : ℝ :=
  let mile_to_feet : ℝ := 5280
  let yard_to_feet : ℝ := 3
  let km_to_meter : ℝ := 1000
  let meter_to_feet : ℝ := 3.28084
  let lionel_miles : ℝ := 4
  let esther_yards : ℝ := 975
  let niklaus_feet : ℝ := 1287
  let isabella_km : ℝ := 18
  let sebastian_meters : ℝ := 2400
  lionel_miles * mile_to_feet +
  esther_yards * yard_to_feet +
  niklaus_feet +
  isabella_km * km_to_meter * meter_to_feet +
  sebastian_meters * meter_to_feet

/-- Theorem stating that the combined distance traveled by friends is 91261.136 feet -/
theorem combined_distance_is_91261_136 : combined_distance = 91261.136 := by
  sorry

end combined_distance_is_91261_136_l2760_276037


namespace g_comp_three_roots_l2760_276058

/-- The function g(x) = x^2 + 8x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 8*x + d

/-- The composition of g with itself -/
def g_comp (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- The statement that g(g(x)) has exactly 3 distinct real roots -/
def has_exactly_three_roots (d : ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x : ℝ, g_comp d x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧
                    r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃

theorem g_comp_three_roots :
  ∀ d : ℝ, has_exactly_three_roots d ↔ d = -20 + 4 * Real.sqrt 14 ∨ d = -20 - 4 * Real.sqrt 14 :=
sorry

end g_comp_three_roots_l2760_276058


namespace marnie_chips_per_day_l2760_276090

/-- Calculates the number of chips Marnie eats each day starting from the second day -/
def chips_per_day (total_chips : ℕ) (first_day_chips : ℕ) (total_days : ℕ) : ℕ :=
  (total_chips - first_day_chips) / (total_days - 1)

/-- Theorem stating that Marnie eats 10 chips per day starting from the second day -/
theorem marnie_chips_per_day :
  chips_per_day 100 10 10 = 10 := by
  sorry

end marnie_chips_per_day_l2760_276090


namespace total_visitors_proof_l2760_276060

/-- The total number of visitors over two days at a tourist attraction -/
def total_visitors (m n : ℕ) : ℕ :=
  2 * m + n + 1000

/-- Theorem: The total number of visitors over two days is 2m + n + 1000 -/
theorem total_visitors_proof (m n : ℕ) : 
  total_visitors m n = 2 * m + n + 1000 := by
  sorry

end total_visitors_proof_l2760_276060


namespace adult_panda_consumption_is_138_l2760_276044

/-- The daily bamboo consumption of an adult panda -/
def adult_panda_daily_consumption : ℕ := 138

/-- The daily bamboo consumption of a baby panda -/
def baby_panda_daily_consumption : ℕ := 50

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total bamboo consumption of both pandas in a week -/
def total_weekly_consumption : ℕ := 1316

/-- Theorem stating that the adult panda's daily bamboo consumption is 138 pounds -/
theorem adult_panda_consumption_is_138 :
  adult_panda_daily_consumption = 
    (total_weekly_consumption - baby_panda_daily_consumption * days_in_week) / days_in_week :=
by sorry

end adult_panda_consumption_is_138_l2760_276044


namespace problem_statement_l2760_276081

theorem problem_statement :
  (∃ x₀ : ℝ, Real.tan x₀ = 2) ∧ ¬(∀ x : ℝ, x^2 + 2*x + 1 > 0) := by sorry

end problem_statement_l2760_276081


namespace johns_age_fraction_l2760_276036

theorem johns_age_fraction (john_age mother_age father_age : ℕ) : 
  father_age = 40 →
  father_age = mother_age + 4 →
  john_age = mother_age - 16 →
  (john_age : ℚ) / father_age = 1 / 2 := by
sorry

end johns_age_fraction_l2760_276036


namespace square_covering_theorem_l2760_276075

theorem square_covering_theorem (l : ℕ) (h1 : l > 0) : 
  (∃ n : ℕ, n > 0 ∧ 2 * n^2 = 8 * l^2 / 9 ∧ l^2 < 2 * (n + 1)^2) ↔ 
  l ∈ ({3, 6, 9, 12, 15, 18, 21, 24} : Set ℕ) :=
sorry

end square_covering_theorem_l2760_276075


namespace number_subtraction_division_l2760_276098

theorem number_subtraction_division : ∃! x : ℝ, (x - 5) / 3 = 4 := by
  sorry

end number_subtraction_division_l2760_276098


namespace intersection_M_N_l2760_276084

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x : ℕ | x - 1 ≥ 0}

theorem intersection_M_N : M ∩ N = {1, 2} := by
  sorry

end intersection_M_N_l2760_276084


namespace first_number_problem_l2760_276013

theorem first_number_problem (x y : ℤ) (h1 : y = 43) (h2 : x + 2 * y = 124) : x = 38 := by
  sorry

end first_number_problem_l2760_276013


namespace hyperbola_circle_intersection_l2760_276033

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if a circle with diameter equal to the distance between the foci
    intersects one of the hyperbola's asymptotes at point (4, 3),
    then a = 4 and b = 3 -/
theorem hyperbola_circle_intersection (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c^2 = 16 + 9 ∧ 3 = (b / a) * 4) → a = 4 ∧ b = 3 :=
sorry

end hyperbola_circle_intersection_l2760_276033


namespace intersection_of_A_and_B_l2760_276056

def A : Set ℝ := {0, 2, 4, 6}
def B : Set ℝ := {x | 3 < x ∧ x < 7}

theorem intersection_of_A_and_B : A ∩ B = {4, 6} := by sorry

end intersection_of_A_and_B_l2760_276056


namespace smallest_solution_and_ratio_l2760_276016

theorem smallest_solution_and_ratio (x : ℝ) (a b c d : ℤ) : 
  (7 * x / 8 - 1 = 4 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (x ≥ (4 - 4 * Real.sqrt 15) / 7) →
  (x = (4 - 4 * Real.sqrt 15) / 7 → a * c * d / b = -105) :=
by sorry

end smallest_solution_and_ratio_l2760_276016


namespace ababab_no_large_prime_factors_l2760_276014

theorem ababab_no_large_prime_factors (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) :
  ∀ p : ℕ, p.Prime → p ∣ (101010 * a + 10101 * b) → p ≤ 99 := by
  sorry

end ababab_no_large_prime_factors_l2760_276014


namespace system_solution_unique_l2760_276008

theorem system_solution_unique :
  ∃! (x y : ℝ), x - y = 1 ∧ 2 * x + 3 * y = 7 :=
by
  -- The proof goes here
  sorry

end system_solution_unique_l2760_276008


namespace smallest_c_inequality_l2760_276007

theorem smallest_c_inequality (c : ℝ) : 
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → Real.sqrt (x^2 + y^2) + c * |x - y| ≥ (x + y) / 2) ↔ c ≥ (1/2 : ℝ) :=
sorry

end smallest_c_inequality_l2760_276007


namespace cafe_order_combinations_l2760_276099

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- Theorem: The number of ways two people can each choose one item from a set of 15 items,
    where order matters and repetition is allowed, is equal to 225. -/
theorem cafe_order_combinations :
  menu_items ^ num_people = 225 := by sorry

end cafe_order_combinations_l2760_276099


namespace lime_juice_per_lime_l2760_276051

-- Define the variables and constants
def tablespoons_per_mocktail : ℚ := 1
def days : ℕ := 30
def limes_per_dollar : ℚ := 3
def dollars_spent : ℚ := 5

-- Define the theorem
theorem lime_juice_per_lime :
  let total_tablespoons := tablespoons_per_mocktail * days
  let total_limes := limes_per_dollar * dollars_spent
  let juice_per_lime := total_tablespoons / total_limes
  juice_per_lime = 2 := by
sorry


end lime_juice_per_lime_l2760_276051


namespace lenyas_number_l2760_276001

theorem lenyas_number (x : ℝ) : ((((x + 5) / 3) * 4) - 6) / 7 = 2 → x = 10 := by
  sorry

end lenyas_number_l2760_276001


namespace puzzle_solution_l2760_276097

/-- Given positive integers A and B less than 10 satisfying the equation 21A104 × 11 = 2B8016 × 9, 
    prove that A = 1 and B = 5. -/
theorem puzzle_solution (A B : ℕ) 
  (h1 : 0 < A ∧ A < 10) 
  (h2 : 0 < B ∧ B < 10) 
  (h3 : 21 * 100000 + A * 10000 + 104 * 11 = 2 * 100000 + B * 10000 + 8016 * 9) : 
  A = 1 ∧ B = 5 := by
sorry

end puzzle_solution_l2760_276097


namespace consecutive_integers_cube_sum_l2760_276088

theorem consecutive_integers_cube_sum : 
  ∀ n : ℕ, 
  n > 0 → 
  (n - 1) * n * (n + 1) = 8 * (3 * n) → 
  (n - 1)^3 + n^3 + (n + 1)^3 = 405 := by
sorry

end consecutive_integers_cube_sum_l2760_276088


namespace line_properties_l2760_276046

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- The y-coordinate of a point on the line given its x-coordinate -/
def y_coord (l : Line3D) (x : ℝ) : ℝ := sorry

/-- The intersection point of the line with the z=0 plane -/
def z_plane_intersection (l : Line3D) : ℝ × ℝ × ℝ := sorry

theorem line_properties (l : Line3D) 
  (h1 : l.point1 = (1, 3, 2)) 
  (h2 : l.point2 = (4, 3, -1)) : 
  y_coord l 7 = 3 ∧ z_plane_intersection l = (3, 3, 0) := by sorry

end line_properties_l2760_276046


namespace exam_comparison_l2760_276071

/-- Given a 50-question exam where Sylvia has one-fifth of incorrect answers
    and Sergio has 4 incorrect answers, prove that Sergio has 6 more correct
    answers than Sylvia. -/
theorem exam_comparison (total_questions : ℕ) (sylvia_incorrect_ratio : ℚ)
    (sergio_incorrect : ℕ) (h1 : total_questions = 50)
    (h2 : sylvia_incorrect_ratio = 1 / 5)
    (h3 : sergio_incorrect = 4) :
    (total_questions - (sylvia_incorrect_ratio * total_questions).num) -
    (total_questions - sergio_incorrect) = 6 := by
  sorry

end exam_comparison_l2760_276071


namespace cost_price_percentage_l2760_276017

-- Define the profit percent
def profit_percent : ℝ := 25

-- Define the relationship between selling price (SP) and cost price (CP)
def selling_price_relation (CP SP : ℝ) : Prop :=
  SP = CP * (1 + profit_percent / 100)

-- Theorem statement
theorem cost_price_percentage (CP SP : ℝ) :
  selling_price_relation CP SP →
  CP / SP * 100 = 80 := by
sorry

end cost_price_percentage_l2760_276017


namespace power_function_through_point_value_l2760_276067

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the theorem
theorem power_function_through_point_value :
  ∀ f : ℝ → ℝ,
  isPowerFunction f →
  f 2 = 8 →
  f 3 = 27 := by
sorry

end power_function_through_point_value_l2760_276067


namespace power_eight_divided_by_four_l2760_276035

theorem power_eight_divided_by_four (n : ℕ) : n = 8^2022 → n/4 = 4^3032 := by
  sorry

end power_eight_divided_by_four_l2760_276035


namespace expand_expression_l2760_276049

theorem expand_expression (x : ℝ) : 6 * (x - 3) * (x^2 + 4*x + 16) = 6*x^3 + 6*x^2 + 24*x - 288 := by
  sorry

end expand_expression_l2760_276049


namespace statue_of_liberty_model_height_l2760_276087

/-- The scale ratio of the model to the actual size -/
def scale_ratio : ℚ := 1 / 30

/-- The actual height of the Statue of Liberty in feet -/
def actual_height : ℕ := 305

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem statue_of_liberty_model_height :
  round_to_nearest (actual_height / scale_ratio) = 10 := by
  sorry

end statue_of_liberty_model_height_l2760_276087


namespace consecutive_odd_integers_multiplier_l2760_276024

theorem consecutive_odd_integers_multiplier :
  ∀ (n : ℤ),
  (n + 4 = 15) →
  (∃ k : ℚ, 3 * n = k * (n + 4) + 3) →
  (∃ k : ℚ, 3 * n = k * (n + 4) + 3 ∧ k = 2) :=
by sorry

end consecutive_odd_integers_multiplier_l2760_276024


namespace triangle_angle_sum_l2760_276094

theorem triangle_angle_sum (a b c : ℝ) (h1 : a + b + c = 180) 
                           (h2 : a = 85) (h3 : b = 35) : c = 60 := by
  sorry

end triangle_angle_sum_l2760_276094


namespace self_square_root_numbers_l2760_276057

theorem self_square_root_numbers : {x : ℝ | x ≥ 0 ∧ x = Real.sqrt x} = {0, 1} := by sorry

end self_square_root_numbers_l2760_276057


namespace range_of_m_l2760_276091

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop := x^2 + m*x + 1 = 0

-- Define the proposition p
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

-- Define the proposition q
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

-- Theorem statement
theorem range_of_m : 
  ∀ m : ℝ, (¬p m ∧ q m) → (1 < m ∧ m ≤ 2) :=
by sorry

end range_of_m_l2760_276091


namespace picnic_attendance_difference_picnic_attendance_difference_proof_l2760_276038

/-- Proves that there are 80 more adults than children at a picnic -/
theorem picnic_attendance_difference : ℕ → Prop :=
  fun total_persons : ℕ =>
    ∀ (men women children adults : ℕ),
      total_persons = 240 →
      men = 120 →
      men = women + 80 →
      adults = men + women →
      total_persons = men + women + children →
      adults - children = 80

-- The proof is omitted
theorem picnic_attendance_difference_proof : picnic_attendance_difference 240 := by
  sorry

end picnic_attendance_difference_picnic_attendance_difference_proof_l2760_276038


namespace rachel_homework_difference_l2760_276023

theorem rachel_homework_difference (math_pages reading_pages : ℕ) 
  (h1 : math_pages = 7) 
  (h2 : reading_pages = 3) : 
  math_pages - reading_pages = 4 := by
sorry

end rachel_homework_difference_l2760_276023


namespace curve_C_properties_l2760_276095

-- Define the curve C
def C (m n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m * p.1^2 + n * p.2^2 = 1}

-- Define what it means for a curve to be an ellipse with foci on the y-axis
def is_ellipse_with_foci_on_y_axis (S : Set (ℝ × ℝ)) : Prop :=
  sorry

-- Define what it means for a curve to be a hyperbola with given asymptotes
def is_hyperbola_with_asymptotes (S : Set (ℝ × ℝ)) (f : ℝ → ℝ) : Prop :=
  sorry

-- Define what it means for a curve to consist of two straight lines
def is_two_straight_lines (S : Set (ℝ × ℝ)) : Prop :=
  sorry

theorem curve_C_properties (m n : ℝ) :
  (m > n ∧ n > 0 → is_ellipse_with_foci_on_y_axis (C m n)) ∧
  (m * n < 0 → is_hyperbola_with_asymptotes (C m n) (λ x => Real.sqrt (-m/n) * x)) ∧
  (m = 0 ∧ n > 0 → is_two_straight_lines (C m n)) :=
  sorry

end curve_C_properties_l2760_276095


namespace value_of_a_l2760_276066

theorem value_of_a (a b d : ℝ) 
  (h1 : a + b = d) 
  (h2 : b + d = 7) 
  (h3 : d = 4) : 
  a = 1 := by
sorry

end value_of_a_l2760_276066


namespace weighted_average_plants_per_hour_l2760_276009

def total_rows : ℕ := 400
def carrot_rows : ℕ := 250
def potato_rows : ℕ := 150

def carrot_first_rows : ℕ := 100
def carrot_first_plants_per_row : ℕ := 275
def carrot_first_time : ℕ := 10

def carrot_remaining_rows : ℕ := 150
def carrot_remaining_plants_per_row : ℕ := 325
def carrot_remaining_time : ℕ := 20

def potato_first_rows : ℕ := 50
def potato_first_plants_per_row : ℕ := 300
def potato_first_time : ℕ := 12

def potato_remaining_rows : ℕ := 100
def potato_remaining_plants_per_row : ℕ := 400
def potato_remaining_time : ℕ := 18

theorem weighted_average_plants_per_hour :
  let total_plants := 
    (carrot_first_rows * carrot_first_plants_per_row + 
     carrot_remaining_rows * carrot_remaining_plants_per_row +
     potato_first_rows * potato_first_plants_per_row + 
     potato_remaining_rows * potato_remaining_plants_per_row)
  let total_time := 
    (carrot_first_time + carrot_remaining_time + 
     potato_first_time + potato_remaining_time)
  (total_plants : ℚ) / total_time = 2187.5 := by
  sorry

end weighted_average_plants_per_hour_l2760_276009


namespace probability_nine_red_in_eleven_draws_l2760_276012

/-- The probability of drawing exactly 9 red balls in 11 draws, with the 11th draw being red,
    from a bag containing 6 white balls and 3 red balls (with replacement) -/
theorem probability_nine_red_in_eleven_draws :
  let total_balls : ℕ := 9
  let red_balls : ℕ := 3
  let white_balls : ℕ := 6
  let total_draws : ℕ := 11
  let red_draws : ℕ := 9
  let p_red : ℚ := red_balls / total_balls
  let p_white : ℚ := white_balls / total_balls
  Nat.choose (total_draws - 1) (red_draws - 1) * p_red ^ red_draws * p_white ^ (total_draws - red_draws) =
    Nat.choose 10 8 * (1 / 3) ^ 9 * (2 / 3) ^ 2 :=
by sorry

end probability_nine_red_in_eleven_draws_l2760_276012


namespace polygon_sides_with_120_degree_interior_angles_l2760_276042

theorem polygon_sides_with_120_degree_interior_angles :
  ∀ (n : ℕ) (interior_angle exterior_angle : ℝ),
    interior_angle = 120 →
    exterior_angle = 180 - interior_angle →
    (n : ℝ) * exterior_angle = 360 →
    n = 6 := by
  sorry

end polygon_sides_with_120_degree_interior_angles_l2760_276042


namespace valid_three_digit_numbers_l2760_276069

def is_valid_number (abc : ℕ) : Prop :=
  let a := abc / 100
  let b := (abc / 10) % 10
  let c := abc % 10
  let cab := c * 100 + a * 10 + b
  let bca := b * 100 + c * 10 + a
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  abc ≥ 100 ∧ abc < 1000 ∧
  2 * b = a + c ∧
  (cab * abc : ℚ) = bca * bca

theorem valid_three_digit_numbers :
  ∀ abc : ℕ, is_valid_number abc → abc = 432 ∨ abc = 864 :=
sorry

end valid_three_digit_numbers_l2760_276069


namespace cubic_equation_solution_l2760_276068

theorem cubic_equation_solution :
  ∃ x : ℝ, x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3 ∧ x = 6 := by
  sorry

end cubic_equation_solution_l2760_276068


namespace problem_statement_l2760_276092

theorem problem_statement : 
  (∀ x y : ℝ, (Real.sqrt x + Real.sqrt y = 0) → (x = 0 ∧ y = 0)) ∨
  (∀ x : ℝ, (x^2 + 4*x - 5 = 0) → (x = -5)) :=
by sorry

end problem_statement_l2760_276092


namespace factorization_equality_l2760_276011

theorem factorization_equality (x : ℝ) : 4 * x^3 - x = x * (2*x + 1) * (2*x - 1) := by
  sorry

end factorization_equality_l2760_276011


namespace tv_production_theorem_l2760_276004

/-- Represents the daily TV production in a factory for a month -/
structure TVProduction where
  totalDays : Nat
  firstPeriodDays : Nat
  firstPeriodAvg : Nat
  monthlyAvg : Nat

/-- Calculates the average daily production for the last period of the month -/
def lastPeriodAvg (p : TVProduction) : Nat :=
  let lastPeriodDays := p.totalDays - p.firstPeriodDays
  let totalProduction := p.totalDays * p.monthlyAvg
  let firstPeriodProduction := p.firstPeriodDays * p.firstPeriodAvg
  (totalProduction - firstPeriodProduction) / lastPeriodDays

theorem tv_production_theorem (p : TVProduction) 
  (h1 : p.totalDays = 30)
  (h2 : p.firstPeriodDays = 25)
  (h3 : p.firstPeriodAvg = 65)
  (h4 : p.monthlyAvg = 60) :
  lastPeriodAvg p = 35 := by
  sorry

#eval lastPeriodAvg ⟨30, 25, 65, 60⟩

end tv_production_theorem_l2760_276004


namespace A_intersection_B_eq_A_l2760_276039

def A (k : ℝ) : Set ℝ := {x | k + 1 ≤ x ∧ x ≤ 2 * k}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem A_intersection_B_eq_A (k : ℝ) : A k ∩ B = A k ↔ k ∈ Set.Iic (3/2) :=
sorry

end A_intersection_B_eq_A_l2760_276039


namespace total_cars_in_week_is_450_l2760_276054

/-- The number of cars passing through a toll booth in a week -/
def totalCarsInWeek (mondayCars : ℕ) : ℕ :=
  -- Monday and Tuesday
  2 * mondayCars +
  -- Wednesday and Thursday
  2 * (2 * mondayCars) +
  -- Friday, Saturday, and Sunday
  3 * mondayCars

/-- Theorem stating that the total number of cars in a week is 450 -/
theorem total_cars_in_week_is_450 : totalCarsInWeek 50 = 450 := by
  sorry

#eval totalCarsInWeek 50

end total_cars_in_week_is_450_l2760_276054


namespace cookie_boxes_problem_l2760_276002

theorem cookie_boxes_problem (n : ℕ) : 
  (n - 7 ≥ 1) →  -- Mark sold at least one box
  (n - 2 ≥ 1) →  -- Ann sold at least one box
  (n - 3 ≥ 1) →  -- Carol sold at least one box
  ((n - 7) + (n - 2) + (n - 3) < n) →  -- Together they sold less than n boxes
  n = 6 := by
sorry

end cookie_boxes_problem_l2760_276002


namespace five_integer_chords_l2760_276070

/-- A circle with a point P inside --/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- The number of chords with integer lengths passing through P --/
def numIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The theorem statement --/
theorem five_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 17) 
  (h2 : c.distanceFromCenter = 8) : 
  numIntegerChords c = 5 := by
  sorry

end five_integer_chords_l2760_276070


namespace problem_solution_l2760_276078

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 3*x - 8
def h (r : ℝ) (x : ℝ) : ℝ := 3*x - r

theorem problem_solution :
  (f 2 = 4 ∧ g (f 2) = 4) ∧
  (∀ x : ℝ, f (g x) = g (f x) ↔ x = 2 ∨ x = 6) ∧
  (∀ r : ℝ, f (h r 2) = h r (f 2) ↔ r = 3 ∨ r = 8) := by
  sorry

end problem_solution_l2760_276078


namespace teresas_pencil_sharing_l2760_276021

/-- Proves that each sibling receives 13 pencils given the conditions of Teresa's pencil sharing problem -/
theorem teresas_pencil_sharing :
  -- Define the given conditions
  let total_pencils : ℕ := 14 + 35
  let pencils_to_keep : ℕ := 10
  let num_siblings : ℕ := 3
  let pencils_to_share : ℕ := total_pencils - pencils_to_keep
  -- Define the theorem
  pencils_to_share / num_siblings = 13 := by
  sorry

end teresas_pencil_sharing_l2760_276021


namespace production_growth_equation_l2760_276048

/-- Represents the production growth scenario of Dream Enterprise --/
def production_growth_scenario (initial_value : ℝ) (growth_rate : ℝ) : Prop :=
  let feb_value := initial_value * (1 + growth_rate)
  let mar_value := initial_value * (1 + growth_rate)^2
  mar_value - feb_value = 220000

/-- Theorem stating the correct equation for the production growth scenario --/
theorem production_growth_equation :
  production_growth_scenario 2000000 x ↔ 2000000 * (1 + x)^2 - 2000000 * (1 + x) = 220000 :=
sorry

end production_growth_equation_l2760_276048


namespace smallest_max_sum_l2760_276055

theorem smallest_max_sum (p q r s t : ℕ+) (h_sum : p + q + r + s + t = 4020) :
  let N := max (p + q) (max (q + r) (max (r + s) (s + t)))
  ∀ m : ℕ, (∀ a b c d e : ℕ+, a + b + c + d + e = 4020 →
    m ≥ max (a + b) (max (b + c) (max (c + d) (d + e)))) →
  m ≥ 1342 :=
by sorry

end smallest_max_sum_l2760_276055


namespace interest_rate_proof_l2760_276073

/-- 
Given a principal sum and an annual interest rate,
if the simple interest for 4 years is one-fifth of the principal,
then the annual interest rate is 5%.
-/
theorem interest_rate_proof (P R : ℝ) (P_pos : P > 0) : 
  (P * R * 4) / 100 = P / 5 → R = 5 := by
  sorry

end interest_rate_proof_l2760_276073


namespace greatest_common_factor_4050_12320_l2760_276085

theorem greatest_common_factor_4050_12320 : Nat.gcd 4050 12320 = 10 := by
  sorry

end greatest_common_factor_4050_12320_l2760_276085


namespace geometric_sequence_sum_l2760_276029

-- Define a geometric sequence
def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometric a →
  a 1 + a 2 + a 3 = 7 →
  a 2 + a 3 + a 4 = 14 →
  a 4 + a 5 + a 6 = 56 := by
  sorry

end geometric_sequence_sum_l2760_276029


namespace binomial_probability_problem_l2760_276096

/-- Binomial distribution -/
def binomial_distribution (n : ℕ) (p : ℝ) : ℝ → ℝ := sorry

/-- Probability of a random variable being greater than or equal to a value -/
def prob_ge (X : ℝ → ℝ) (k : ℝ) : ℝ := sorry

theorem binomial_probability_problem (p : ℝ) :
  let ξ := binomial_distribution 2 p
  let η := binomial_distribution 4 p
  prob_ge ξ 1 = 5/9 →
  prob_ge η 2 = 11/27 := by sorry

end binomial_probability_problem_l2760_276096


namespace sin_15_cos_15_double_l2760_276050

theorem sin_15_cos_15_double : 2 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end sin_15_cos_15_double_l2760_276050


namespace cookies_per_package_l2760_276026

theorem cookies_per_package
  (num_friends : ℕ)
  (num_packages : ℕ)
  (cookies_per_child : ℕ)
  (h1 : num_friends = 4)
  (h2 : num_packages = 3)
  (h3 : cookies_per_child = 15) :
  (num_friends + 1) * cookies_per_child / num_packages = 25 := by
  sorry

end cookies_per_package_l2760_276026


namespace oranges_minus_apples_difference_l2760_276080

/-- The number of apples Leif has -/
def num_apples : ℕ := 14

/-- The number of dozens of oranges Leif has -/
def dozens_oranges : ℕ := 2

/-- The number of fruits in a dozen -/
def fruits_per_dozen : ℕ := 12

/-- Calculates the total number of oranges -/
def total_oranges : ℕ := dozens_oranges * fruits_per_dozen

/-- Theorem stating the difference between oranges and apples -/
theorem oranges_minus_apples_difference : 
  total_oranges - num_apples = 10 := by sorry

end oranges_minus_apples_difference_l2760_276080


namespace no_special_polyhedron_l2760_276062

/-- A convex polyhedron. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  four_faces_share_edges : Bool

/-- Theorem stating that there does not exist a convex polyhedron with the specified properties. -/
theorem no_special_polyhedron :
  ¬ ∃ (p : ConvexPolyhedron), 
    p.vertices = 8 ∧ 
    p.edges = 12 ∧ 
    p.faces = 6 ∧ 
    p.four_faces_share_edges = true :=
by
  sorry

end no_special_polyhedron_l2760_276062


namespace briana_investment_proof_l2760_276086

def emma_investment : ℝ := 300
def emma_yield_rate : ℝ := 0.15
def briana_yield_rate : ℝ := 0.10
def years : ℕ := 2
def return_difference : ℝ := 10

def briana_investment : ℝ := 400

theorem briana_investment_proof :
  (years : ℝ) * emma_yield_rate * emma_investment - 
  (years : ℝ) * briana_yield_rate * briana_investment = return_difference :=
by sorry

end briana_investment_proof_l2760_276086


namespace draw_specific_nondefective_draw_at_least_one_defective_l2760_276082

/-- Represents the total number of products -/
def total_products : ℕ := 10

/-- Represents the number of defective products -/
def defective_products : ℕ := 2

/-- Represents the number of products drawn for inspection -/
def drawn_products : ℕ := 3

/-- Theorem for the number of ways to draw a specific non-defective product -/
theorem draw_specific_nondefective :
  (total_products - defective_products).choose (drawn_products - 1) = 36 := by sorry

/-- Theorem for the number of ways to draw at least one defective product -/
theorem draw_at_least_one_defective :
  (defective_products.choose 1 * (total_products - defective_products).choose (drawn_products - 1)) +
  (defective_products.choose 2 * (total_products - defective_products).choose (drawn_products - 2)) = 64 := by sorry

end draw_specific_nondefective_draw_at_least_one_defective_l2760_276082


namespace cubic_expansion_l2760_276006

theorem cubic_expansion (x : ℝ) : 
  3*x^3 - 10*x^2 + 13 = 3*(x-2)^3 + 8*(x-2)^2 - 4*(x-2) - 3 := by
  sorry

end cubic_expansion_l2760_276006


namespace triangle_area_on_rectangle_l2760_276083

/-- Given a rectangle of 6 units by 8 units and a triangle DEF with vertices
    D(0,2), E(6,0), and F(3,8) located on the boundary of the rectangle,
    prove that the area of triangle DEF is 21 square units. -/
theorem triangle_area_on_rectangle (D E F : ℝ × ℝ) : 
  D = (0, 2) →
  E = (6, 0) →
  F = (3, 8) →
  let rectangle_width : ℝ := 6
  let rectangle_height : ℝ := 8
  let triangle_area := abs ((D.1 * (E.2 - F.2) + E.1 * (F.2 - D.2) + F.1 * (D.2 - E.2)) / 2)
  triangle_area = 21 :=
by sorry

end triangle_area_on_rectangle_l2760_276083


namespace solve_equation_l2760_276072

theorem solve_equation (x : ℚ) : 5 * (2 * x - 3) = 3 * (3 - 4 * x) + 15 → x = 39 / 22 := by
  sorry

end solve_equation_l2760_276072


namespace binders_per_student_is_one_l2760_276063

/-- Calculates the number of binders per student given the class size, costs of supplies, and total spent -/
def bindersPerStudent (
  classSize : ℕ) 
  (penCost notebookCost binderCost highlighterCost : ℚ)
  (pensPerStudent notebooksPerStudent highlightersPerStudent : ℕ)
  (teacherDiscount totalSpent : ℚ) : ℚ :=
  let totalPenCost := classSize * pensPerStudent * penCost
  let totalNotebookCost := classSize * notebooksPerStudent * notebookCost
  let totalHighlighterCost := classSize * highlightersPerStudent * highlighterCost
  let effectiveAmount := totalSpent + teacherDiscount
  let binderSpend := effectiveAmount - (totalPenCost + totalNotebookCost + totalHighlighterCost)
  let totalBinders := binderSpend / binderCost
  totalBinders / classSize

theorem binders_per_student_is_one :
  bindersPerStudent 30 0.5 1.25 4.25 0.75 5 3 2 100 260 = 1 := by
  sorry

end binders_per_student_is_one_l2760_276063


namespace unique_solution_ab_minus_a_minus_b_equals_one_l2760_276030

theorem unique_solution_ab_minus_a_minus_b_equals_one :
  ∀ a b : ℤ, a > b ∧ b > 0 ∧ a * b - a - b = 1 → a = 3 ∧ b = 2 :=
by
  sorry

end unique_solution_ab_minus_a_minus_b_equals_one_l2760_276030


namespace max_value_of_trig_function_l2760_276003

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x => Real.sin (x + Real.pi / 18) + Real.cos (x - Real.pi / 9)
  ∃ M : ℝ, (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = Real.sqrt 3 := by
  sorry

end max_value_of_trig_function_l2760_276003


namespace age_difference_l2760_276032

/-- Given the ages of four individuals x, y, z, and w, prove that z is 1.2 decades younger than x. -/
theorem age_difference (x y z w : ℕ) : 
  (x + y = y + z + 12) → 
  (x + y + w = y + z + w + 12) → 
  (x : ℚ) - z = 12 ∧ (x - z : ℚ) / 10 = 1.2 := by
  sorry

end age_difference_l2760_276032


namespace arithmetic_sequence_property_l2760_276059

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + 2 * a 8 + a 15 = 96) →
  2 * a 9 - a 10 = 24 := by
sorry

end arithmetic_sequence_property_l2760_276059


namespace spam_email_ratio_l2760_276010

theorem spam_email_ratio (total : ℕ) (important : ℕ) (promotional_fraction : ℚ) 
  (h1 : total = 400)
  (h2 : important = 180)
  (h3 : promotional_fraction = 2/5) :
  (total - important - (total - important) * promotional_fraction : ℚ) / total = 33/100 := by
  sorry

end spam_email_ratio_l2760_276010


namespace mike_total_cards_l2760_276041

def initial_cards : ℕ := 87
def received_cards : ℕ := 13

theorem mike_total_cards : initial_cards + received_cards = 100 := by
  sorry

end mike_total_cards_l2760_276041


namespace base7_product_and_sum_l2760_276031

/-- Converts a base 7 number to decimal --/
def toDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a decimal number to base 7 --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Computes the sum of digits in a list --/
def sumDigits (n : List Nat) : Nat :=
  n.foldl (· + ·) 0

/-- The main theorem to prove --/
theorem base7_product_and_sum :
  let a := [5, 3]  -- 35 in base 7
  let b := [4, 2]  -- 24 in base 7
  let product := toBase7 (toDecimal a * toDecimal b)
  product = [6, 3, 2, 1] ∧ 
  toBase7 (sumDigits product) = [5, 1] := by
  sorry


end base7_product_and_sum_l2760_276031


namespace max_value_quadratic_l2760_276043

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 6) :
  (∀ y, 0 < y ∧ y < 6 → (6 - y) * y ≤ (6 - x) * x) → (6 - x) * x = 9 :=
by sorry

end max_value_quadratic_l2760_276043


namespace geometric_sequence_tenth_term_l2760_276025

theorem geometric_sequence_tenth_term : 
  ∀ (a : ℚ) (r : ℚ),
    a = 5 →
    a * r = 20 / 3 →
    a * r^9 = 1310720 / 19683 :=
by sorry

end geometric_sequence_tenth_term_l2760_276025


namespace cone_angle_calculation_l2760_276015

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents a cone -/
structure Cone where
  vertex : ℝ × ℝ × ℝ

/-- Configuration of spheres and cone -/
structure SphereConeConfiguration where
  sphere1 : Sphere
  sphere2 : Sphere
  sphere3 : Sphere
  cone : Cone
  spheresTouch : Bool
  coneTouchesSpheres : Bool
  vertexBetweenContacts : Bool

/-- The angle at the vertex of the cone -/
def coneAngle (config : SphereConeConfiguration) : ℝ :=
  sorry

theorem cone_angle_calculation (config : SphereConeConfiguration) 
  (h1 : config.sphere1.radius = 2)
  (h2 : config.sphere2.radius = 2)
  (h3 : config.sphere3.radius = 1)
  (h4 : config.spheresTouch = true)
  (h5 : config.coneTouchesSpheres = true)
  (h6 : config.vertexBetweenContacts = true) :
  coneAngle config = 2 * Real.arctan (1 / 8) :=
sorry

end cone_angle_calculation_l2760_276015


namespace min_trees_chopped_l2760_276061

def trees_per_sharpening : ℕ := 13
def cost_per_sharpening : ℕ := 5
def total_sharpening_cost : ℕ := 35

theorem min_trees_chopped :
  ∃ (n : ℕ), n ≥ 91 ∧ n ≥ (total_sharpening_cost / cost_per_sharpening) * trees_per_sharpening :=
by sorry

end min_trees_chopped_l2760_276061
