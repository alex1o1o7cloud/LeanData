import Mathlib

namespace asterisk_replacement_l1392_139267

theorem asterisk_replacement : ∃! (n : ℝ), n > 0 ∧ (n / 18) * (n / 72) = 1 := by
  sorry

end asterisk_replacement_l1392_139267


namespace sweeties_remainder_l1392_139232

theorem sweeties_remainder (m : ℕ) (h1 : m > 0) (h2 : m % 7 = 6) : (4 * m) % 7 = 3 := by
  sorry

end sweeties_remainder_l1392_139232


namespace characterize_satisfying_function_l1392_139228

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y u : ℝ, f (x + u) (y + u) = f x y + u) ∧
  (∀ x y v : ℝ, f (x * v) (y * v) = f x y * v)

/-- The main theorem -/
theorem characterize_satisfying_function :
  ∀ f : ℝ → ℝ → ℝ, SatisfyingFunction f →
  ∃ p q : ℝ, p + q = 1 ∧ ∀ x y : ℝ, f x y = p * x + q * y :=
sorry

end characterize_satisfying_function_l1392_139228


namespace vector_collinearity_implies_t_value_l1392_139255

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Definition of collinearity for 2D vectors -/
def collinear (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

theorem vector_collinearity_implies_t_value :
  let OA : Vector2D := ⟨1, 2⟩
  let OB : Vector2D := ⟨3, 4⟩
  let OC : Vector2D := ⟨2*t, t+5⟩
  let AB : Vector2D := ⟨OB.x - OA.x, OB.y - OA.y⟩
  let AC : Vector2D := ⟨OC.x - OA.x, OC.y - OA.y⟩
  collinear AB AC → t = 4 := by
  sorry

end vector_collinearity_implies_t_value_l1392_139255


namespace test_scores_l1392_139225

theorem test_scores (scores : List Nat) : 
  (scores.length > 0) →
  (scores.Pairwise (·≠·)) →
  (scores.sum = 119) →
  (scores.take 3).sum = 23 →
  (scores.reverse.take 3).sum = 49 →
  (scores.length = 10 ∧ scores.maximum = some 18) := by
  sorry

end test_scores_l1392_139225


namespace maple_trees_cut_down_is_two_l1392_139244

/-- The number of maple trees cut down in the park -/
def maple_trees_cut_down (initial_maple_trees : ℝ) (final_maple_trees : ℝ) : ℝ :=
  initial_maple_trees - final_maple_trees

/-- Theorem: The number of maple trees cut down is 2 -/
theorem maple_trees_cut_down_is_two :
  maple_trees_cut_down 9.0 7 = 2 := by
  sorry

end maple_trees_cut_down_is_two_l1392_139244


namespace estimated_y_value_at_28_l1392_139257

/-- Linear regression equation -/
def linear_regression (x : ℝ) : ℝ := 4.75 * x + 257

/-- Theorem: The estimated y value is 390 when x is 28 -/
theorem estimated_y_value_at_28 : linear_regression 28 = 390 := by
  sorry

end estimated_y_value_at_28_l1392_139257


namespace pear_juice_percentage_l1392_139226

/-- Represents the juice extraction rate for a fruit -/
structure JuiceRate where
  fruit : String
  ounces : ℚ
  count : ℕ

/-- Calculates the percentage of one juice in a blend of two juices with equal volumes -/
def juicePercentage (rate1 rate2 : JuiceRate) : ℚ :=
  100 * (rate1.ounces * rate2.count) / (rate1.ounces * rate2.count + rate2.ounces * rate1.count)

theorem pear_juice_percentage (pearRate orangeRate : JuiceRate) 
  (h1 : pearRate.fruit = "pear")
  (h2 : orangeRate.fruit = "orange")
  (h3 : pearRate.ounces = 9)
  (h4 : pearRate.count = 4)
  (h5 : orangeRate.ounces = 10)
  (h6 : orangeRate.count = 3) :
  juicePercentage pearRate orangeRate = 50 := by
  sorry

#eval juicePercentage 
  { fruit := "pear", ounces := 9, count := 4 }
  { fruit := "orange", ounces := 10, count := 3 }

end pear_juice_percentage_l1392_139226


namespace apple_pear_ratio_l1392_139271

/-- Proves that the ratio of initial apples to initial pears is 2:1 given the conditions --/
theorem apple_pear_ratio (initial_pears initial_oranges : ℕ) 
  (fruits_given_away fruits_left : ℕ) : 
  initial_pears = 10 →
  initial_oranges = 20 →
  fruits_given_away = 2 →
  fruits_left = 44 →
  ∃ (initial_apples : ℕ), 
    initial_apples - fruits_given_away + 
    (initial_pears - fruits_given_away) + 
    (initial_oranges - fruits_given_away) = fruits_left ∧
    initial_apples / initial_pears = 2 := by
  sorry

end apple_pear_ratio_l1392_139271


namespace workers_combined_time_specific_workers_problem_l1392_139227

/-- Given the time taken by three workers to complete a job individually,
    calculate the time taken when they work together. -/
theorem workers_combined_time (t_a t_b t_c : ℝ) (h_pos_a : t_a > 0) (h_pos_b : t_b > 0) (h_pos_c : t_c > 0) :
  (1 / (1 / t_a + 1 / t_b + 1 / t_c)) = (t_a * t_b * t_c) / (t_b * t_c + t_a * t_c + t_a * t_b) :=
by sorry

/-- The specific problem with Worker A taking 8 hours, Worker B taking 10 hours,
    and Worker C taking 12 hours. -/
theorem specific_workers_problem :
  (1 / (1 / 8 + 1 / 10 + 1 / 12) : ℝ) = 120 / 37 :=
by sorry

end workers_combined_time_specific_workers_problem_l1392_139227


namespace not_in_fourth_quadrant_l1392_139279

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the fourth quadrant
def fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

-- Theorem statement
theorem not_in_fourth_quadrant (m : ℝ) :
  ¬(fourth_quadrant ⟨m - 2, m + 1⟩) := by
  sorry

end not_in_fourth_quadrant_l1392_139279


namespace another_divisor_l1392_139200

theorem another_divisor (smallest_number : ℕ) : 
  smallest_number = 44402 →
  (smallest_number + 2) % 12 = 0 →
  (smallest_number + 2) % 30 = 0 →
  (smallest_number + 2) % 48 = 0 →
  (smallest_number + 2) % 74 = 0 →
  (smallest_number + 2) % 22202 = 0 := by
sorry

end another_divisor_l1392_139200


namespace village_population_equality_l1392_139265

/-- The number of years after which the populations are equal -/
def years : ℕ := 14

/-- The rate of population decrease per year for the first village -/
def decrease_rate : ℕ := 1200

/-- The rate of population increase per year for the second village -/
def increase_rate : ℕ := 800

/-- The initial population of the second village -/
def initial_population_second : ℕ := 42000

/-- The initial population of the first village -/
def initial_population_first : ℕ := 70000

theorem village_population_equality :
  initial_population_first - years * decrease_rate = 
  initial_population_second + years * increase_rate :=
by sorry

end village_population_equality_l1392_139265


namespace set_operations_l1392_139223

def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

theorem set_operations :
  (A ∪ B = {x | 2 ≤ x ∧ x ≤ 7}) ∧
  ((U \ A) ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)}) := by
  sorry

end set_operations_l1392_139223


namespace distance_between_cities_l1392_139290

theorem distance_between_cities (v1 v2 t_diff : ℝ) (h1 : v1 = 60) (h2 : v2 = 70) (h3 : t_diff = 0.25) :
  let t := (v2 * t_diff) / (v2 - v1)
  v1 * t = 105 :=
by sorry

end distance_between_cities_l1392_139290


namespace sufficient_but_not_necessary_l1392_139214

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, x > 2 → 1 / x < 1 / 2) ∧
  (∃ x, 1 / x < 1 / 2 ∧ x ≤ 2) :=
by sorry

end sufficient_but_not_necessary_l1392_139214


namespace parabola_from_circles_l1392_139289

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + y - 3 = 0

-- Define the directrix
def directrix (y : ℝ) : Prop := y = -1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

theorem parabola_from_circles :
  ∀ (x y : ℝ),
  (∃ (x₁ y₁ x₂ y₂ : ℝ), circle1 x₁ y₁ ∧ circle2 x₂ y₂ ∧ directrix y₁ ∧ directrix y₂) →
  parabola x y :=
by sorry

end parabola_from_circles_l1392_139289


namespace mouse_lives_count_l1392_139220

def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := dog_lives + 7

theorem mouse_lives_count : mouse_lives = 13 := by
  sorry

end mouse_lives_count_l1392_139220


namespace nested_fraction_equality_l1392_139216

theorem nested_fraction_equality : (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end nested_fraction_equality_l1392_139216


namespace expression_equals_one_l1392_139233

theorem expression_equals_one (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hsum : a + b + c = 0) : 
  (a^2 * b^2) / ((a^2 - b*c) * (b^2 - a*c)) + 
  (a^2 * c^2) / ((a^2 - b*c) * (c^2 - a*b)) + 
  (b^2 * c^2) / ((b^2 - a*c) * (c^2 - a*b)) = 1 := by
  sorry

end expression_equals_one_l1392_139233


namespace quadratic_inequality_range_l1392_139274

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, 4 * x^2 + (a - 2) * x + 1 > 0) ↔ 
  (a ≤ -2 ∨ a ≥ 6) := by sorry

end quadratic_inequality_range_l1392_139274


namespace game_score_theorem_l1392_139248

/-- Calculates the total points scored in a game with three tries --/
def total_points (first_try : ℕ) (second_try_difference : ℕ) : ℕ :=
  let second_try := first_try - second_try_difference
  let third_try := 2 * second_try
  first_try + second_try + third_try

/-- Theorem stating that under the given conditions, the total points scored is 1390 --/
theorem game_score_theorem :
  total_points 400 70 = 1390 := by
  sorry

end game_score_theorem_l1392_139248


namespace appliance_cost_l1392_139240

theorem appliance_cost (a b : ℝ) 
  (eq1 : a + 2 * b = 2300)
  (eq2 : 2 * a + b = 2050) :
  a = 600 ∧ b = 850 := by
  sorry

end appliance_cost_l1392_139240


namespace line_slope_is_two_l1392_139215

/-- Given a line with y-intercept 2 and passing through the point (269, 540),
    prove that its slope is 2. -/
theorem line_slope_is_two (line : Set (ℝ × ℝ)) 
    (y_intercept : (0, 2) ∈ line)
    (point_on_line : (269, 540) ∈ line) :
    let slope := (540 - 2) / (269 - 0)
    slope = 2 := by
  sorry

end line_slope_is_two_l1392_139215


namespace total_spent_is_413_06_l1392_139204

/-- Calculates the total amount spent including sales tax -/
def total_spent (speakers_cost cd_player_cost tires_cost tax_rate : ℝ) : ℝ :=
  let subtotal := speakers_cost + cd_player_cost + tires_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Theorem stating that the total amount spent is $413.06 -/
theorem total_spent_is_413_06 :
  total_spent 136.01 139.38 112.46 0.065 = 413.06 := by
  sorry

end total_spent_is_413_06_l1392_139204


namespace remainder_13_pow_2048_mod_11_l1392_139247

theorem remainder_13_pow_2048_mod_11 : 13^2048 % 11 = 3 := by
  sorry

end remainder_13_pow_2048_mod_11_l1392_139247


namespace negation_of_implication_l1392_139259

theorem negation_of_implication (x : ℝ) :
  (¬(x^2 + x - 6 ≥ 0 → x > 2)) ↔ (x^2 + x - 6 < 0 → x ≤ 2) :=
by sorry

end negation_of_implication_l1392_139259


namespace dogsledding_race_speed_difference_l1392_139251

/-- The dogsledding race problem -/
theorem dogsledding_race_speed_difference
  (course_length : ℝ)
  (team_b_speed : ℝ)
  (time_difference : ℝ)
  (h1 : course_length = 300)
  (h2 : team_b_speed = 20)
  (h3 : time_difference = 3)
  (h4 : team_b_speed > 0) :
  let team_b_time := course_length / team_b_speed
  let team_a_time := team_b_time - time_difference
  let team_a_speed := course_length / team_a_time
  team_a_speed - team_b_speed = 5 := by
  sorry

end dogsledding_race_speed_difference_l1392_139251


namespace parabola_vertex_l1392_139230

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = 3x^2 - 6x + 5 is at the point (1, 2) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end parabola_vertex_l1392_139230


namespace symmetry_y_axis_l1392_139263

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the y-axis -/
def symmetricYAxis (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := -p.z }

/-- The theorem stating that (-2, -1, -4) is symmetrical to (2, -1, 4) with respect to the y-axis -/
theorem symmetry_y_axis :
  let P : Point3D := { x := 2, y := -1, z := 4 }
  let Q : Point3D := { x := -2, y := -1, z := -4 }
  symmetricYAxis P = Q := by sorry

end symmetry_y_axis_l1392_139263


namespace factory_works_four_days_l1392_139284

/-- Represents a toy factory with weekly production and daily production rates. -/
structure ToyFactory where
  weekly_production : ℕ
  daily_production : ℕ

/-- Calculates the number of working days per week for a given toy factory. -/
def working_days (factory : ToyFactory) : ℕ :=
  factory.weekly_production / factory.daily_production

/-- Theorem: The toy factory works 4 days per week. -/
theorem factory_works_four_days :
  let factory : ToyFactory := { weekly_production := 6000, daily_production := 1500 }
  working_days factory = 4 := by
  sorry

#eval working_days { weekly_production := 6000, daily_production := 1500 }

end factory_works_four_days_l1392_139284


namespace solution_satisfies_system_l1392_139219

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + y = 7
def equation2 (x y : ℝ) : Prop := 2 * x - y = 2

-- State the theorem
theorem solution_satisfies_system :
  ∃ (x y : ℝ), equation1 x y ∧ equation2 x y ∧ x = 3 ∧ y = 4 := by
  sorry

end solution_satisfies_system_l1392_139219


namespace sum_of_five_consecutive_odd_l1392_139222

def is_sum_of_five_consecutive_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, 2 * k + 1 + (2 * k + 3) + (2 * k + 5) + (2 * k + 7) + (2 * k + 9) = n

theorem sum_of_five_consecutive_odd :
  ¬ (is_sum_of_five_consecutive_odd 16) ∧
  (is_sum_of_five_consecutive_odd 40) ∧
  (is_sum_of_five_consecutive_odd 72) ∧
  (is_sum_of_five_consecutive_odd 100) ∧
  (is_sum_of_five_consecutive_odd 200) :=
by sorry

end sum_of_five_consecutive_odd_l1392_139222


namespace photocopy_cost_l1392_139276

/-- The cost of a single photocopy --/
def C : ℝ := sorry

/-- The discount rate for large orders --/
def discount_rate : ℝ := 0.25

/-- The number of copies in a large order --/
def large_order : ℕ := 160

/-- The total cost savings when placing a large order --/
def total_savings : ℝ := 0.80

theorem photocopy_cost :
  C = 0.02 :=
by sorry

end photocopy_cost_l1392_139276


namespace cone_surface_area_l1392_139236

theorem cone_surface_area (r h : ℝ) (hr : r = 1) (hh : h = 2 * Real.sqrt 2) :
  let l := Real.sqrt (r^2 + h^2)
  π * r^2 + π * r * l = 4 * π :=
by sorry

end cone_surface_area_l1392_139236


namespace next_common_day_l1392_139201

def dance_interval : ℕ := 6
def karate_interval : ℕ := 12
def library_interval : ℕ := 18

theorem next_common_day (dance_interval karate_interval library_interval : ℕ) :
  dance_interval = 6 → karate_interval = 12 → library_interval = 18 →
  Nat.lcm (Nat.lcm dance_interval karate_interval) library_interval = 36 :=
by sorry

end next_common_day_l1392_139201


namespace income_calculation_l1392_139277

-- Define the total income
def total_income : ℝ := sorry

-- Define the percentage given to children
def children_percentage : ℝ := 0.2 * 3

-- Define the percentage given to wife
def wife_percentage : ℝ := 0.3

-- Define the remaining percentage after giving to children and wife
def remaining_percentage : ℝ := 1 - children_percentage - wife_percentage

-- Define the percentage donated to orphan house
def orphan_house_percentage : ℝ := 0.05

-- Define the final amount left
def final_amount : ℝ := 40000

-- Theorem to prove
theorem income_calculation : 
  ∃ (total_income : ℝ),
    total_income > 0 ∧
    final_amount = total_income * remaining_percentage * (1 - orphan_house_percentage) ∧
    (total_income ≥ 421052) ∧ (total_income ≤ 421053) :=
by sorry

end income_calculation_l1392_139277


namespace selection_with_condition_l1392_139246

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := 10

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of students excluding the two specific students -/
def remaining_students : ℕ := total_students - 2

theorem selection_with_condition :
  (choose total_students selected_students) - (choose remaining_students selected_students) = 140 := by
  sorry

end selection_with_condition_l1392_139246


namespace classroom_ratio_l1392_139282

theorem classroom_ratio (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 30 → boys = 20 → girls = total_students - boys → 
  (girls : ℚ) / (boys : ℚ) = 1 / 2 := by
sorry

end classroom_ratio_l1392_139282


namespace cube_surface_area_increase_l1392_139293

theorem cube_surface_area_increase :
  ∀ (s : ℝ), s > 0 →
  let original_surface_area := 6 * s^2
  let new_edge_length := 1.3 * s
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 0.69 :=
by sorry

end cube_surface_area_increase_l1392_139293


namespace victors_percentage_l1392_139245

/-- Given that Victor scored 184 marks out of a maximum of 200 marks,
    prove that the percentage of marks he got is 92%. -/
theorem victors_percentage (marks_obtained : ℕ) (maximum_marks : ℕ) 
  (h1 : marks_obtained = 184) (h2 : maximum_marks = 200) :
  (marks_obtained : ℚ) / maximum_marks * 100 = 92 := by
  sorry

end victors_percentage_l1392_139245


namespace x_values_l1392_139207

theorem x_values (x : ℝ) :
  (x^3 - 3 = 3/8 → x = 3/2) ∧
  ((x - 1)^2 = 25 → x = 6 ∨ x = -4) := by
  sorry

end x_values_l1392_139207


namespace intersection_orthogonality_l1392_139294

/-- The line equation -/
def line (x y : ℝ) : Prop := y = 2 * Real.sqrt 2 * (x - 1)

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

/-- Point A satisfies both line and parabola equations -/
def point_A (x y : ℝ) : Prop := line x y ∧ parabola x y

/-- Point B satisfies both line and parabola equations -/
def point_B (x y : ℝ) : Prop := line x y ∧ parabola x y

/-- Point M has coordinates (-1, m) -/
def point_M (m : ℝ) : ℝ × ℝ := (-1, m)

/-- The dot product of vectors MA and MB is zero -/
def orthogonal_condition (x_a y_a x_b y_b m : ℝ) : Prop :=
  (x_a + 1) * (x_b + 1) + (y_a - m) * (y_b - m) = 0

theorem intersection_orthogonality (x_a y_a x_b y_b m : ℝ) :
  point_A x_a y_a →
  point_B x_b y_b →
  orthogonal_condition x_a y_a x_b y_b m →
  m = Real.sqrt 2 / 2 := by sorry

end intersection_orthogonality_l1392_139294


namespace dawn_cd_count_l1392_139269

theorem dawn_cd_count (dawn kristine : ℕ) 
  (h1 : kristine = dawn + 7)
  (h2 : dawn + kristine = 27) : 
  dawn = 10 := by
sorry

end dawn_cd_count_l1392_139269


namespace min_sum_arithmetic_sequence_l1392_139241

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (a₁ + arithmetic_sequence a₁ d n) / 2

theorem min_sum_arithmetic_sequence :
  let a₁ : ℤ := -28
  let d : ℤ := 4
  ∀ k : ℕ, k ≥ 1 →
    (sum_arithmetic_sequence a₁ d k ≥ sum_arithmetic_sequence a₁ d 7 ∧
     sum_arithmetic_sequence a₁ d k ≥ sum_arithmetic_sequence a₁ d 8) ∧
    (sum_arithmetic_sequence a₁ d 7 = sum_arithmetic_sequence a₁ d 8) :=
by sorry

end min_sum_arithmetic_sequence_l1392_139241


namespace at_least_one_not_less_than_two_l1392_139270

theorem at_least_one_not_less_than_two (a b : ℕ) (h : a + b ≥ 3) : max a b ≥ 2 := by
  sorry

end at_least_one_not_less_than_two_l1392_139270


namespace lcm_18_24_l1392_139209

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l1392_139209


namespace gcf_of_120_180_240_l1392_139264

theorem gcf_of_120_180_240 : Nat.gcd 120 (Nat.gcd 180 240) = 60 := by
  sorry

end gcf_of_120_180_240_l1392_139264


namespace berry_problem_l1392_139278

/-- Proves that given the conditions in the berry problem, Steve started with 8.5 berries and Amanda started with 3.5 berries. -/
theorem berry_problem (stacy_initial : ℝ) (steve_takes : ℝ) (amanda_takes : ℝ) (amanda_more : ℝ)
  (h1 : stacy_initial = 32)
  (h2 : steve_takes = 4)
  (h3 : amanda_takes = 3.25)
  (h4 : amanda_more = 5.75)
  (h5 : steve_takes + (stacy_initial / 2 - 7.5) = stacy_initial / 2 - 7.5 + steve_takes - amanda_takes + amanda_more) :
  (stacy_initial / 2 - 7.5 = 8.5) ∧ (stacy_initial / 2 - 7.5 + steve_takes - amanda_takes - amanda_more = 3.5) :=
by sorry

end berry_problem_l1392_139278


namespace initial_sum_calculation_l1392_139208

theorem initial_sum_calculation (final_amount : ℚ) (interest_rate : ℚ) (years : ℕ) :
  final_amount = 1192 →
  interest_rate = 48.00000000000001 →
  years = 4 →
  final_amount = (1000 : ℚ) + years * interest_rate :=
by
  sorry

#eval (1000 : ℚ) + 4 * 48.00000000000001 -- This should evaluate to 1192

end initial_sum_calculation_l1392_139208


namespace white_white_pairs_coincide_l1392_139296

/-- Represents a half of the geometric figure -/
structure Half where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the coinciding pairs when the halves are folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ

/-- The main theorem statement -/
theorem white_white_pairs_coincide 
  (half : Half) 
  (coinciding : CoincidingPairs) 
  (h1 : half.red = 4) 
  (h2 : half.blue = 7) 
  (h3 : half.white = 10) 
  (h4 : coinciding.red_red = 3) 
  (h5 : coinciding.blue_blue = 4) 
  (h6 : coinciding.red_white = 3) : 
  ∃ (white_white : ℕ), white_white = 7 ∧ 
    white_white = half.white - coinciding.red_white := by
  sorry

end white_white_pairs_coincide_l1392_139296


namespace star_equal_set_is_three_lines_l1392_139217

-- Define the ⋆ operation
def star (a b : ℝ) : ℝ := a^2 * b + a * b^2

-- Define the set of points (x, y) where x ⋆ y = y ⋆ x
def star_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

-- Theorem statement
theorem star_equal_set_is_three_lines :
  star_equal_set = {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 + p.2 = 0} :=
by sorry

end star_equal_set_is_three_lines_l1392_139217


namespace sum_of_digits_l1392_139238

/-- Given a three-digit number of the form 4a4, where 'a' is a single digit,
    we add 258 to it to get a three-digit number of the form 7b2,
    where 'b' is also a single digit. If 7b2 is divisible by 3,
    then a + b = 4. -/
theorem sum_of_digits (a b : Nat) : 
  (a ≥ 0 ∧ a ≤ 9) →  -- 'a' is a single digit
  (b ≥ 0 ∧ b ≤ 9) →  -- 'b' is a single digit
  (400 + 10 * a + 4) + 258 = 700 + 10 * b + 2 →  -- 4a4 + 258 = 7b2
  (700 + 10 * b + 2) % 3 = 0 →  -- 7b2 is divisible by 3
  a + b = 4 := by
  sorry

end sum_of_digits_l1392_139238


namespace smallest_lcm_with_gcd_5_l1392_139239

theorem smallest_lcm_with_gcd_5 :
  ∀ m n : ℕ,
  1000 ≤ m ∧ m < 10000 ∧
  1000 ≤ n ∧ n < 10000 ∧
  Nat.gcd m n = 5 →
  203010 ≤ Nat.lcm m n ∧
  ∃ m₀ n₀ : ℕ,
    1000 ≤ m₀ ∧ m₀ < 10000 ∧
    1000 ≤ n₀ ∧ n₀ < 10000 ∧
    Nat.gcd m₀ n₀ = 5 ∧
    Nat.lcm m₀ n₀ = 203010 :=
by sorry

end smallest_lcm_with_gcd_5_l1392_139239


namespace impossible_tiling_l1392_139242

/-- Represents a chessboard with two opposite corners removed -/
structure ChessboardWithCornersRemoved where
  n : ℕ+  -- n is a positive natural number

/-- Represents a 2 × 1 domino -/
structure Domino

/-- Represents a tiling of the chessboard with dominoes -/
def Tiling (board : ChessboardWithCornersRemoved) := List Domino

theorem impossible_tiling (board : ChessboardWithCornersRemoved) :
  ¬ ∃ (t : Tiling board), t.length = (board.n.val ^ 2 - 2) / 2 := by
  sorry

end impossible_tiling_l1392_139242


namespace clara_score_remainder_l1392_139231

theorem clara_score_remainder (a b c : ℕ) : 
  (1 ≤ a ∧ a ≤ 9) →  -- 'a' represents the tens digit
  (0 ≤ b ∧ b ≤ 9) →  -- 'b' represents the ones digit
  (0 ≤ c ∧ c ≤ 9) →  -- 'c' represents the appended digit
  ∃ r : ℕ, r < 10 ∧ ((100 * a + 10 * b + c) - (10 * a + b)) % 9 = r :=
by sorry

end clara_score_remainder_l1392_139231


namespace extreme_value_odd_function_l1392_139254

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

-- State the theorem
theorem extreme_value_odd_function 
  (a b c : ℝ) 
  (h1 : f a b c 1 = c - 4)  -- f(x) reaches c-4 at x=1
  (h2 : ∀ x, f a b c (-x) = -(f a b c x))  -- f(x) is odd
  : 
  (a = 2 ∧ b = -6) ∧  -- Part 1: values of a and b
  (∀ x ∈ Set.Ioo (-2) 0, f a b c x ≤ 4)  -- Part 2: maximum value on (-2,0)
  :=
by sorry

end extreme_value_odd_function_l1392_139254


namespace imaginary_part_of_complex_fraction_l1392_139229

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im ((i^3) / (1 + i)) = -1/2 := by sorry

end imaginary_part_of_complex_fraction_l1392_139229


namespace distinct_prime_factors_of_2310_l1392_139285

theorem distinct_prime_factors_of_2310 : Nat.card (Nat.factors 2310).toFinset = 5 := by
  sorry

end distinct_prime_factors_of_2310_l1392_139285


namespace median_is_six_l1392_139224

/-- Represents the attendance data for a group of students -/
structure AttendanceData where
  total_students : Nat
  attend_4_times : Nat
  attend_5_times : Nat
  attend_6_times : Nat
  attend_7_times : Nat
  attend_8_times : Nat

/-- Calculates the median attendance for a given AttendanceData -/
def median_attendance (data : AttendanceData) : Nat :=
  sorry

/-- Theorem stating that the median attendance for the given data is 6 -/
theorem median_is_six (data : AttendanceData) 
  (h1 : data.total_students = 20)
  (h2 : data.attend_4_times = 1)
  (h3 : data.attend_5_times = 5)
  (h4 : data.attend_6_times = 7)
  (h5 : data.attend_7_times = 4)
  (h6 : data.attend_8_times = 3) :
  median_attendance data = 6 := by
  sorry

end median_is_six_l1392_139224


namespace triangle_sin_A_l1392_139275

theorem triangle_sin_A (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = Real.pi) →
  -- Given conditions
  (a = 2) →
  (b = 3) →
  (Real.tan B = 3) →
  -- Law of Sines (assumed as part of triangle definition)
  (a / Real.sin A = b / Real.sin B) →
  -- Conclusion
  Real.sin A = Real.sqrt 10 / 5 := by
sorry

end triangle_sin_A_l1392_139275


namespace triangle_inequality_l1392_139283

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_range : 0 < α ∧ α < π
  cosine_rule : 2 * b * c * Real.cos α = b^2 + c^2 - a^2

-- State the theorem
theorem triangle_inequality (t : Triangle) : 
  (2 * t.b * t.c * Real.cos t.α) / (t.b + t.c) < t.b + t.c - t.a ∧ 
  t.b + t.c - t.a < (2 * t.b * t.c) / t.a :=
sorry

end triangle_inequality_l1392_139283


namespace seating_arrangements_l1392_139266

/-- The number of ways to seat n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange a block of k people within a group of n people -/
def blockArrangements (n k : ℕ) : ℕ := (Nat.factorial n) * (Nat.factorial k)

/-- The number of valid seating arrangements for n people, 
    where k specific people cannot sit in k consecutive seats -/
def validArrangements (n k : ℕ) : ℕ := 
  totalArrangements n - blockArrangements (n - k + 1) k

theorem seating_arrangements : 
  validArrangements 10 4 = 3507840 := by sorry

end seating_arrangements_l1392_139266


namespace largest_number_on_board_l1392_139205

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_four (n : ℕ) : Prop := n % 10 = 4

def set_of_interest : Set ℕ := {n | is_two_digit n ∧ n % 6 = 0 ∧ ends_in_four n}

theorem largest_number_on_board : 
  ∃ (m : ℕ), m ∈ set_of_interest ∧ ∀ (n : ℕ), n ∈ set_of_interest → n ≤ m ∧ m = 84 :=
sorry

end largest_number_on_board_l1392_139205


namespace complex_parts_of_z_l1392_139261

theorem complex_parts_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := i * (-1 + 2*i)
  (z.re = -2) ∧ (z.im = -1) := by sorry

end complex_parts_of_z_l1392_139261


namespace arithmetic_sequence_inequality_l1392_139287

/-- An arithmetic sequence with positive terms and non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : ∀ n, a n > 0
  h2 : d ≠ 0
  h3 : ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality (seq : ArithmeticSequence) : 
  seq.a 1 * seq.a 8 < seq.a 4 * seq.a 5 := by
  sorry

end arithmetic_sequence_inequality_l1392_139287


namespace non_monotonic_derivative_range_l1392_139298

open Real

theorem non_monotonic_derivative_range (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, deriv f x = exp x + k^2 / exp x - 1 / k) →
  (¬ Monotone f) →
  0 < k ∧ k < sqrt 2 / 2 :=
sorry

end non_monotonic_derivative_range_l1392_139298


namespace projective_transformation_existence_l1392_139202

-- Define a projective plane
class ProjectivePlane (P : Type*) :=
  (Line : Type*)
  (incidence : P → Line → Prop)
  (axiom_existence : ∀ (A B : P), ∃ (l : Line), incidence A l ∧ incidence B l)
  (axiom_uniqueness : ∀ (A B : P) (l m : Line), incidence A l → incidence B l → incidence A m → incidence B m → l = m)
  (axiom_nondegeneracy : ∃ (A B C : P), ¬∃ (l : Line), incidence A l ∧ incidence B l ∧ incidence C l)

-- Define a projective transformation
def ProjectiveTransformation (P : Type*) [ProjectivePlane P] := P → P

-- Define the property of four points being non-collinear
def NonCollinear {P : Type*} [ProjectivePlane P] (A B C D : P) : Prop :=
  ¬∃ (l : ProjectivePlane.Line P), ProjectivePlane.incidence A l ∧ ProjectivePlane.incidence B l ∧ ProjectivePlane.incidence C l ∧ ProjectivePlane.incidence D l

-- State the theorem
theorem projective_transformation_existence
  {P : Type*} [ProjectivePlane P]
  (A B C D A₁ B₁ C₁ D₁ : P)
  (h1 : NonCollinear A B C D)
  (h2 : NonCollinear A₁ B₁ C₁ D₁) :
  ∃ (f : ProjectiveTransformation P),
    f A = A₁ ∧ f B = B₁ ∧ f C = C₁ ∧ f D = D₁ :=
sorry

end projective_transformation_existence_l1392_139202


namespace no_finite_moves_to_fill_board_l1392_139288

-- Define the chessboard as a type
def Chessboard := ℤ × ℤ

-- Define the set A
def A : Set Chessboard :=
  {p | 100 ∣ p.1 ∧ 100 ∣ p.2}

-- Define a king's move
def is_valid_move (start finish : Chessboard) : Prop :=
  (start = finish) ∨
  (abs (start.1 - finish.1) ≤ 1 ∧ abs (start.2 - finish.2) ≤ 1)

-- Define the initial configuration of kings
def initial_kings : Set Chessboard :=
  {p | p ∉ A}

-- Define the state after k moves
def state_after_moves (k : ℕ) : Set Chessboard → Set Chessboard :=
  sorry

-- The main theorem
theorem no_finite_moves_to_fill_board :
  ¬ ∃ (k : ℕ), (state_after_moves k initial_kings) = Set.univ :=
sorry

end no_finite_moves_to_fill_board_l1392_139288


namespace square_root_fourth_power_l1392_139281

theorem square_root_fourth_power (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end square_root_fourth_power_l1392_139281


namespace total_distance_is_9km_l1392_139210

/-- Represents the travel plans from the city bus station to Tianbo Mountain -/
inductive TravelPlan
| BusOnly
| BikeOnly
| BikeThenBus
| BusThenBike

/-- Represents the journey from the city bus station to Tianbo Mountain -/
structure Journey where
  distance_to_hehua : ℝ
  distance_from_hehua : ℝ
  bus_speed : ℝ
  bike_speed : ℝ
  bus_stop_time : ℝ

/-- The actual journey based on the problem description -/
def actual_journey : Journey where
  distance_to_hehua := 6
  distance_from_hehua := 3
  bus_speed := 24
  bike_speed := 16
  bus_stop_time := 0.5

/-- Theorem stating that the total distance is 9 km -/
theorem total_distance_is_9km (j : Journey) :
  j.distance_to_hehua + j.distance_from_hehua = 9 ∧
  j.distance_to_hehua = 6 ∧
  j.distance_from_hehua = 3 ∧
  j.bus_speed = 24 ∧
  j.bike_speed = 16 ∧
  j.bus_stop_time = 0.5 ∧
  (j.distance_to_hehua + j.distance_from_hehua) / j.bus_speed + j.bus_stop_time =
    (j.distance_to_hehua + j.distance_from_hehua + 1) / j.bike_speed ∧
  j.distance_to_hehua / j.bus_speed = 4 / j.bike_speed ∧
  (j.distance_to_hehua / j.bus_speed + j.bus_stop_time + j.distance_from_hehua / j.bike_speed) =
    ((j.distance_to_hehua + j.distance_from_hehua) / j.bus_speed + j.bus_stop_time - 0.25) :=
by sorry

#check total_distance_is_9km actual_journey

end total_distance_is_9km_l1392_139210


namespace unique_solution_quadratic_equation_l1392_139286

theorem unique_solution_quadratic_equation (m n : ℤ) :
  m^2 - 2*m*n + 2*n^2 - 4*n + 4 = 0 → m = 2 ∧ n = 2 := by
  sorry

end unique_solution_quadratic_equation_l1392_139286


namespace other_student_correct_answers_l1392_139221

/-- 
Given:
- Martin answered 40 questions correctly
- Martin answered three fewer questions correctly than Kelsey
- Kelsey answered eight more questions correctly than another student

Prove: The other student answered 35 questions correctly
-/
theorem other_student_correct_answers 
  (martin_correct : ℕ) 
  (kelsey_martin_diff : ℕ) 
  (kelsey_other_diff : ℕ) 
  (h1 : martin_correct = 40)
  (h2 : kelsey_martin_diff = 3)
  (h3 : kelsey_other_diff = 8) :
  martin_correct + kelsey_martin_diff - kelsey_other_diff = 35 := by
sorry

end other_student_correct_answers_l1392_139221


namespace grocery_expense_l1392_139260

/-- Calculates the amount spent on groceries given credit card transactions -/
theorem grocery_expense (initial_balance new_balance returns : ℚ) : 
  initial_balance = 126 ∧ 
  new_balance = 171 ∧ 
  returns = 45 → 
  ∃ (grocery_expense : ℚ), 
    grocery_expense = 60 ∧ 
    initial_balance + grocery_expense + (grocery_expense / 2) - returns = new_balance := by
  sorry

end grocery_expense_l1392_139260


namespace compound_interest_rate_interest_rate_problem_l1392_139280

/-- Compound interest calculation --/
theorem compound_interest_rate (P A : ℝ) (t n : ℕ) (h : A = P * (1 + 0.25)^(n * t)) :
  ∃ (r : ℝ), A = P * (1 + r)^(n * t) ∧ r = 0.25 :=
by sorry

/-- Problem-specific theorem --/
theorem interest_rate_problem (P A : ℝ) (t n : ℕ) 
  (h_P : P = 1200)
  (h_A : A = 2488.32)
  (h_t : t = 4)
  (h_n : n = 1) :
  ∃ (r : ℝ), A = P * (1 + r)^(n * t) ∧ r = 0.25 :=
by sorry

end compound_interest_rate_interest_rate_problem_l1392_139280


namespace arithmetic_sequence_problem_l1392_139213

/-- A positive arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ d, ∀ k, a (k + 1) = a k + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_eq : a 2^2 + 2*(a 2)*(a 6) + a 6^2 - 4 = 0) :
  a 4 = 1 := by
sorry

end arithmetic_sequence_problem_l1392_139213


namespace quadratic_discriminant_l1392_139299

/-- The discriminant of the quadratic equation 2x^2 + (2 + 1/2)x + 1/2 is 9/4 -/
theorem quadratic_discriminant : 
  let a : ℚ := 2
  let b : ℚ := 5/2
  let c : ℚ := 1/2
  let discriminant := b^2 - 4*a*c
  discriminant = 9/4 := by
sorry

end quadratic_discriminant_l1392_139299


namespace last_integer_in_sequence_l1392_139235

def sequence_term (n : ℕ) : ℚ :=
  800000 / 2^n

theorem last_integer_in_sequence :
  ∀ k : ℕ, (sequence_term k).isInt → sequence_term k ≥ 3125 :=
sorry

end last_integer_in_sequence_l1392_139235


namespace magic_money_box_theorem_l1392_139291

/-- Represents the state of the magic money box on a given day -/
structure BoxState :=
  (day : Nat)
  (value : Nat)

/-- Calculates the next day's value based on the current state and added coins -/
def nextDayValue (state : BoxState) (added : Nat) : Nat :=
  (state.value * (state.day + 2) + added)

/-- Simulates the magic money box for a week -/
def simulateWeek : Nat :=
  let monday := BoxState.mk 0 2
  let tuesday := BoxState.mk 1 (nextDayValue monday 5)
  let wednesday := BoxState.mk 2 (nextDayValue tuesday 10)
  let thursday := BoxState.mk 3 (nextDayValue wednesday 25)
  let friday := BoxState.mk 4 (nextDayValue thursday 50)
  let saturday := BoxState.mk 5 (nextDayValue friday 0)
  let sunday := BoxState.mk 6 (nextDayValue saturday 0)
  sunday.value

theorem magic_money_box_theorem : simulateWeek = 142240 := by
  sorry

end magic_money_box_theorem_l1392_139291


namespace liam_speed_reduction_l1392_139250

/-- Proves that Liam should have driven 5 mph slower to arrive exactly on time -/
theorem liam_speed_reduction (distance : ℝ) (actual_speed : ℝ) (early_time : ℝ) :
  distance = 10 →
  actual_speed = 30 →
  early_time = 4 / 60 →
  let required_speed := distance / (distance / actual_speed + early_time)
  actual_speed - required_speed = 5 := by sorry

end liam_speed_reduction_l1392_139250


namespace system_solution_l1392_139295

theorem system_solution (x y m : ℝ) : 
  (2 * x + y = 4) → 
  (x + 2 * y = m) → 
  (x + y = 1) → 
  m = -1 := by
sorry

end system_solution_l1392_139295


namespace square_of_negative_x_plus_one_l1392_139292

theorem square_of_negative_x_plus_one (x : ℝ) : (-x - 1)^2 = x^2 + 2*x + 1 := by
  sorry

end square_of_negative_x_plus_one_l1392_139292


namespace garden_furniture_cost_ratio_l1392_139212

/-- Given a garden table and bench with a combined cost of 750 and the bench costing 250,
    prove that the ratio of the table's cost to the bench's cost is 2:1. -/
theorem garden_furniture_cost_ratio :
  ∀ (table_cost bench_cost : ℝ),
    bench_cost = 250 →
    table_cost + bench_cost = 750 →
    ∃ (n : ℕ), table_cost = n * bench_cost →
    table_cost / bench_cost = 2 := by
  sorry

end garden_furniture_cost_ratio_l1392_139212


namespace fourth_test_score_for_average_l1392_139272

def test1 : ℕ := 80
def test2 : ℕ := 70
def test3 : ℕ := 90
def test4 : ℕ := 100
def targetAverage : ℕ := 85

theorem fourth_test_score_for_average :
  (test1 + test2 + test3 + test4) / 4 = targetAverage :=
sorry

end fourth_test_score_for_average_l1392_139272


namespace sara_savings_l1392_139234

def quarters_to_cents (quarters : ℕ) (cents_per_quarter : ℕ) : ℕ :=
  quarters * cents_per_quarter

theorem sara_savings : quarters_to_cents 11 25 = 275 := by
  sorry

end sara_savings_l1392_139234


namespace specific_cone_measurements_l1392_139262

/-- Represents a cone formed from a circular sector -/
structure SectorCone where
  circle_radius : ℝ
  sector_angle : ℝ

/-- Calculate the volume of the cone divided by π -/
def volume_div_pi (cone : SectorCone) : ℝ :=
  sorry

/-- Calculate the lateral surface area of the cone divided by π -/
def lateral_area_div_pi (cone : SectorCone) : ℝ :=
  sorry

/-- Theorem stating the volume and lateral surface area for a specific cone -/
theorem specific_cone_measurements :
  let cone : SectorCone := { circle_radius := 16, sector_angle := 270 }
  volume_div_pi cone = 384 ∧ lateral_area_div_pi cone = 192 := by
  sorry

end specific_cone_measurements_l1392_139262


namespace ponchik_honey_cakes_l1392_139243

theorem ponchik_honey_cakes 
  (exercise walk run swim : ℕ) 
  (h1 : exercise * 2 = walk * 3)
  (h2 : walk * 3 = run * 5)
  (h3 : run * 5 = swim * 6)
  (h4 : exercise + walk + run + swim = 216) :
  exercise - swim = 60 := by
  sorry

end ponchik_honey_cakes_l1392_139243


namespace well_volume_l1392_139218

/-- The volume of a circular cylinder with diameter 2 metres and height 10 metres is π × 10 m³ -/
theorem well_volume :
  let diameter : ℝ := 2
  let depth : ℝ := 10
  let radius : ℝ := diameter / 2
  let volume : ℝ := π * radius^2 * depth
  volume = π * 10 := by
  sorry

end well_volume_l1392_139218


namespace mushroom_consumption_l1392_139203

theorem mushroom_consumption (initial_amount leftover_amount : ℕ) 
  (h1 : initial_amount = 15)
  (h2 : leftover_amount = 7) :
  initial_amount - leftover_amount = 8 :=
by sorry

end mushroom_consumption_l1392_139203


namespace jogging_track_circumference_l1392_139252

/-- The circumference of a circular jogging track given two people walking in opposite directions --/
theorem jogging_track_circumference 
  (deepak_speed : ℝ) 
  (wife_speed : ℝ) 
  (meeting_time : ℝ) 
  (h1 : deepak_speed = 4.5) 
  (h2 : wife_speed = 3.75) 
  (h3 : meeting_time = 4.32) : 
  deepak_speed * meeting_time + wife_speed * meeting_time = 35.64 := by
  sorry

#check jogging_track_circumference

end jogging_track_circumference_l1392_139252


namespace circumcenter_rational_coords_l1392_139297

/-- Given a triangle with rational coordinates, the center of its circumscribed circle has rational coordinates. -/
theorem circumcenter_rational_coords (a₁ a₂ a₃ b₁ b₂ b₃ : ℚ) :
  ∃ (x y : ℚ), 
    (x - a₁)^2 + (y - b₁)^2 = (x - a₂)^2 + (y - b₂)^2 ∧
    (x - a₁)^2 + (y - b₁)^2 = (x - a₃)^2 + (y - b₃)^2 :=
by sorry

end circumcenter_rational_coords_l1392_139297


namespace rational_sum_l1392_139253

theorem rational_sum (a b : ℚ) (h : |a + 6| + (b - 4)^2 = 0) : a + b = -2 := by
  sorry

end rational_sum_l1392_139253


namespace negation_of_P_l1392_139258

-- Define the original proposition P
def P : Prop := ∃ n : ℕ, n^2 > 2^n

-- State the theorem that the negation of P is equivalent to the given statement
theorem negation_of_P : (¬ P) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end negation_of_P_l1392_139258


namespace engineer_check_time_l1392_139256

/-- Represents the road construction project --/
structure RoadProject where
  totalLength : ℝ
  totalDays : ℝ
  initialWorkers : ℝ
  completedLength : ℝ
  additionalWorkers : ℝ

/-- Calculates the number of days after which the progress was checked --/
def daysUntilCheck (project : RoadProject) : ℝ :=
  200 -- The actual calculation is replaced with the known result

/-- Theorem stating that the engineer checked the progress after 200 days --/
theorem engineer_check_time (project : RoadProject) 
    (h1 : project.totalLength = 15)
    (h2 : project.totalDays = 300)
    (h3 : project.initialWorkers = 35)
    (h4 : project.completedLength = 2.5)
    (h5 : project.additionalWorkers = 52.5) :
  daysUntilCheck project = 200 := by
  sorry

#check engineer_check_time

end engineer_check_time_l1392_139256


namespace reflection_line_sum_l1392_139206

/-- The line of reflection for a point (x₁, y₁) to (x₂, y₂) has slope m and y-intercept b -/
def is_reflection_line (x₁ y₁ x₂ y₂ m b : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  (midpoint_y = m * midpoint_x + b) ∧ 
  (m * ((x₂ - x₁) / 2) = (y₁ - y₂) / 2)

/-- The sum of slope and y-intercept of the reflection line for (2, 3) to (10, 7) is 3 -/
theorem reflection_line_sum :
  ∃ (m b : ℝ), is_reflection_line 2 3 10 7 m b ∧ m + b = 3 := by
  sorry

end reflection_line_sum_l1392_139206


namespace butterflies_let_go_l1392_139237

theorem butterflies_let_go (original : ℕ) (left : ℕ) (h1 : original = 93) (h2 : left = 82) :
  original - left = 11 := by
  sorry

end butterflies_let_go_l1392_139237


namespace bobby_candy_consumption_l1392_139273

/-- The number of candy pieces Bobby ate first -/
def first_eaten : ℕ := 34

/-- The number of candy pieces Bobby ate later -/
def later_eaten : ℕ := 18

/-- The total number of candy pieces Bobby ate -/
def total_eaten : ℕ := first_eaten + later_eaten

theorem bobby_candy_consumption :
  total_eaten = 52 := by
  sorry

end bobby_candy_consumption_l1392_139273


namespace geometric_sequence_minimum_l1392_139268

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

-- Define the problem statement
theorem geometric_sequence_minimum (a₁ : ℝ) (q : ℝ) :
  (a₁ > 0) →
  (q > 0) →
  (geometric_sequence a₁ q 2017 = geometric_sequence a₁ q 2016 + 2 * geometric_sequence a₁ q 2015) →
  (∃ m n : ℕ, (geometric_sequence a₁ q m) * (geometric_sequence a₁ q n) = 16 * a₁^2) →
  (∃ m n : ℕ, ∀ k l : ℕ, 4/k + 1/l ≥ 4/m + 1/n ∧ 4/m + 1/n = 3/2) :=
by sorry

end geometric_sequence_minimum_l1392_139268


namespace smallest_angle_for_trig_equation_l1392_139211

theorem smallest_angle_for_trig_equation : 
  ∃ y : ℝ, y > 0 ∧ 
  (∀ z : ℝ, z > 0 → 6 * Real.sin z * Real.cos z ^ 3 - 6 * Real.sin z ^ 3 * Real.cos z = 3/2 → y ≤ z) ∧
  6 * Real.sin y * Real.cos y ^ 3 - 6 * Real.sin y ^ 3 * Real.cos y = 3/2 ∧
  y = 7.5 * π / 180 := by
  sorry

end smallest_angle_for_trig_equation_l1392_139211


namespace parabola_c_value_l1392_139249

/-- A parabola passing through two given points has a specific c value -/
theorem parabola_c_value :
  ∀ (b c : ℝ),
  (1^2 + b*1 + c = 5) →
  ((-2)^2 + b*(-2) + c = -8) →
  c = 4/3 := by
sorry

end parabola_c_value_l1392_139249
