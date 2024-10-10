import Mathlib

namespace eighty_nine_degrees_is_acute_l2086_208636

-- Define what an acute angle is
def is_acute_angle (angle : ℝ) : Prop := 0 < angle ∧ angle < 90

-- State the theorem
theorem eighty_nine_degrees_is_acute : is_acute_angle 89 := by
  sorry

end eighty_nine_degrees_is_acute_l2086_208636


namespace sqrt_six_equality_l2086_208621

theorem sqrt_six_equality (r : ℝ) (h : r = Real.sqrt 2 + Real.sqrt 3) :
  Real.sqrt 6 = (r^2 - 5) / 2 := by sorry

end sqrt_six_equality_l2086_208621


namespace triangle_side_length_simplification_l2086_208666

theorem triangle_side_length_simplification 
  (a b c : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  |a + b - c| - |b - a - c| = 2*b - 2*c := by
sorry

end triangle_side_length_simplification_l2086_208666


namespace fastest_student_survey_method_l2086_208682

/-- Represents a survey method -/
inductive SurveyMethod
| Comprehensive
| Sample

/-- Represents a scenario requiring a survey -/
structure Scenario where
  description : String
  requiredMethod : SurveyMethod

/-- Represents the selection of the fastest student in a school's short-distance race -/
def fastestStudentSelection : Scenario :=
  { description := "Selecting the fastest student in a school's short-distance race",
    requiredMethod := SurveyMethod.Comprehensive }

/-- Theorem: The appropriate survey method for selecting the fastest student
    in a school's short-distance race is a comprehensive survey -/
theorem fastest_student_survey_method :
  fastestStudentSelection.requiredMethod = SurveyMethod.Comprehensive :=
by sorry


end fastest_student_survey_method_l2086_208682


namespace mork_tax_rate_l2086_208656

theorem mork_tax_rate (mork_income : ℝ) (mork_rate : ℝ) : 
  mork_rate > 0 →
  mork_income > 0 →
  (mork_rate / 100 * mork_income + 0.15 * (4 * mork_income)) / (5 * mork_income) = 0.21 →
  mork_rate = 45 := by
sorry

end mork_tax_rate_l2086_208656


namespace reservoir_after_storm_l2086_208669

/-- Represents the capacity of the reservoir in billion gallons -/
def reservoir_capacity : ℝ := 400

/-- Represents the initial amount of water in the reservoir in billion gallons -/
def initial_water : ℝ := 200

/-- Represents the amount of water added by the storm in billion gallons -/
def storm_water : ℝ := 120

/-- Theorem stating that the reservoir is 80% full after the storm -/
theorem reservoir_after_storm :
  (initial_water + storm_water) / reservoir_capacity = 0.8 := by
  sorry

end reservoir_after_storm_l2086_208669


namespace distance_after_three_minutes_l2086_208616

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

theorem distance_after_three_minutes :
  let truck_speed : ℝ := 65
  let car_speed : ℝ := 85
  let time_in_hours : ℝ := 3 / 60
  distance_between_vehicles truck_speed car_speed time_in_hours = 1 := by
  sorry

end distance_after_three_minutes_l2086_208616


namespace proposition_p_sufficient_not_necessary_for_q_l2086_208672

theorem proposition_p_sufficient_not_necessary_for_q :
  (∀ a b : ℝ, a * b ≠ 0 → a^2 + b^2 ≠ 0) ∧
  (∃ a b : ℝ, a^2 + b^2 ≠ 0 ∧ a * b = 0) :=
by sorry

end proposition_p_sufficient_not_necessary_for_q_l2086_208672


namespace smallest_number_satisfying_conditions_l2086_208617

theorem smallest_number_satisfying_conditions : ∃ n : ℕ, n > 0 ∧ 
  n % 6 = 2 ∧ n % 8 = 3 ∧ n % 9 = 4 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 2 → m % 8 = 3 → m % 9 = 4 → n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_number_satisfying_conditions_l2086_208617


namespace subset_P_l2086_208655

def P : Set ℝ := {x | x ≤ 3}

theorem subset_P : {-1} ⊆ P := by
  sorry

end subset_P_l2086_208655


namespace chord_length_l2086_208668

/-- The length of the chord cut by a circle on a line --/
theorem chord_length (x y : ℝ) : 
  (x^2 + y^2 = 9) →  -- Circle equation
  (x + y = 2 * Real.sqrt 2) →  -- Line equation
  ∃ (a b : ℝ), (a - x)^2 + (b - y)^2 = 25 ∧  -- Chord endpoints
               (a^2 + b^2 = 9) ∧  -- Endpoints on circle
               (a + b = 2 * Real.sqrt 2) :=  -- Endpoints on line
by sorry

end chord_length_l2086_208668


namespace quadratic_function_properties_l2086_208612

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties :
  a ≠ 0 →
  (∀ x : ℝ, f a b c x ≠ x) →
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) ∧
  (a + b + c = 0 → ∀ x : ℝ, f a b c (f a b c x) < x) :=
by sorry

end quadratic_function_properties_l2086_208612


namespace symmetric_circle_equation_l2086_208639

/-- Given a circle and a point of symmetry, find the equation of the symmetric circle -/
theorem symmetric_circle_equation (x y : ℝ) :
  let given_circle := (x - 2)^2 + (y - 1)^2 = 1
  let point_of_symmetry := (1, 2)
  let symmetric_circle := x^2 + (y - 3)^2 = 1
  symmetric_circle = true := by sorry

end symmetric_circle_equation_l2086_208639


namespace halloween_cleanup_time_halloween_cleanup_time_specific_l2086_208609

/-- Calculates the total cleaning time for Halloween vandalism -/
theorem halloween_cleanup_time 
  (egg_cleanup_time : ℕ) 
  (tp_cleanup_time : ℕ) 
  (num_eggs : ℕ) 
  (num_tp : ℕ) : ℕ :=
  let egg_time_seconds := egg_cleanup_time * num_eggs
  let egg_time_minutes := egg_time_seconds / 60
  let tp_time_minutes := tp_cleanup_time * num_tp
  egg_time_minutes + tp_time_minutes

/-- Proves that the total cleaning time for 60 eggs and 7 rolls of toilet paper is 225 minutes -/
theorem halloween_cleanup_time_specific : 
  halloween_cleanup_time 15 30 60 7 = 225 := by
  sorry

end halloween_cleanup_time_halloween_cleanup_time_specific_l2086_208609


namespace birds_storks_difference_l2086_208632

/-- Given the initial conditions of birds and storks on a fence, prove that there are 3 more birds than storks. -/
theorem birds_storks_difference :
  let initial_birds : ℕ := 2
  let additional_birds : ℕ := 5
  let storks : ℕ := 4
  let total_birds : ℕ := initial_birds + additional_birds
  total_birds - storks = 3 :=
by
  sorry

end birds_storks_difference_l2086_208632


namespace julio_is_ten_l2086_208693

-- Define the ages as natural numbers
def zipporah_age : ℕ := 7
def dina_age : ℕ := 51 - zipporah_age
def julio_age : ℕ := 54 - dina_age

-- State the theorem
theorem julio_is_ten : julio_age = 10 := by
  sorry

end julio_is_ten_l2086_208693


namespace line_equation_proof_l2086_208642

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def lies_on (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_equation_proof (l : Line) :
  parallel l (Line.mk 2 (-1) 1) →
  lies_on (Point.mk 1 2) l →
  l = Line.mk 2 (-1) 0 :=
by
  sorry

end line_equation_proof_l2086_208642


namespace inverse_256_mod_101_l2086_208620

theorem inverse_256_mod_101 (h : (16⁻¹ : ZMod 101) = 31) :
  (256⁻¹ : ZMod 101) = 52 := by
  sorry

end inverse_256_mod_101_l2086_208620


namespace union_of_A_and_B_l2086_208625

def A : Set ℤ := {-2, 0}
def B : Set ℤ := {-2, 3}

theorem union_of_A_and_B : A ∪ B = {-2, 0, 3} := by sorry

end union_of_A_and_B_l2086_208625


namespace driveways_shoveled_l2086_208608

-- Define the prices and quantities
def candy_bar_price : ℚ := 3/4
def candy_bar_quantity : ℕ := 2
def lollipop_price : ℚ := 1/4
def lollipop_quantity : ℕ := 4
def driveway_price : ℚ := 3/2

-- Define the total spent at the candy store
def total_spent : ℚ := candy_bar_price * candy_bar_quantity + lollipop_price * lollipop_quantity

-- Define the fraction of earnings spent
def fraction_spent : ℚ := 1/6

-- Theorem to prove
theorem driveways_shoveled :
  (total_spent / fraction_spent) / driveway_price = 10 := by
  sorry


end driveways_shoveled_l2086_208608


namespace prob_exactly_one_of_two_independent_l2086_208678

/-- The probability of exactly one of two independent events occurring -/
theorem prob_exactly_one_of_two_independent (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  p₁ * (1 - p₂) + p₂ * (1 - p₁) = 
  (p₁ + p₂) - (p₁ * p₂) := by sorry

end prob_exactly_one_of_two_independent_l2086_208678


namespace inequality_solution_l2086_208657

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 6) ≥ 1) ↔ (x ∈ Set.Icc (-4) (-2) ∪ Set.Icc 2 4) :=
by sorry

end inequality_solution_l2086_208657


namespace original_number_proof_l2086_208660

theorem original_number_proof (x : ℝ) : x * 1.1 = 550 ↔ x = 500 := by
  sorry

end original_number_proof_l2086_208660


namespace divisibility_of_quadratic_form_l2086_208695

theorem divisibility_of_quadratic_form (p a b k : ℤ) : 
  Prime p → 
  p = 3*k + 2 → 
  p ∣ (a^2 + a*b + b^2) → 
  p ∣ a ∧ p ∣ b :=
by sorry

end divisibility_of_quadratic_form_l2086_208695


namespace power_mod_seventeen_l2086_208676

theorem power_mod_seventeen : 7^2048 % 17 = 1 := by
  sorry

end power_mod_seventeen_l2086_208676


namespace line_through_point_parallel_to_given_l2086_208601

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - 3 * y + 5 = 0

-- Define the point that the new line passes through
def point : ℝ × ℝ := (-2, 1)

-- Define the new line
def new_line (x y : ℝ) : Prop := 2 * x - 3 * y + 7 = 0

-- Theorem statement
theorem line_through_point_parallel_to_given :
  (new_line point.1 point.2) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), new_line x₁ y₁ → new_line x₂ y₂ → 
    (x₂ - x₁) * 3 = (y₂ - y₁) * 2) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), given_line x₁ y₁ → given_line x₂ y₂ → 
    (x₂ - x₁) * 3 = (y₂ - y₁) * 2) :=
by sorry

end line_through_point_parallel_to_given_l2086_208601


namespace trivia_team_absentees_l2086_208624

theorem trivia_team_absentees (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  total_members = 9 →
  points_per_member = 2 →
  total_points = 12 →
  total_members - (total_points / points_per_member) = 3 := by
sorry

end trivia_team_absentees_l2086_208624


namespace fixed_point_theorem_l2086_208641

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define a point on the parabola
def point_on_parabola (p : ℝ) (x y : ℝ) : Prop := parabola p x y

-- Define the fixed points A and B
def point_A (a b : ℝ) : ℝ × ℝ := (a, b)
def point_B (a : ℝ) : ℝ × ℝ := (-a, 0)

-- Define a line passing through two points
def line_through_points (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1)

-- Define the theorem
theorem fixed_point_theorem (p a b : ℝ) (M M1 M2 : ℝ × ℝ) 
  (h1 : a * b ≠ 0)
  (h2 : b^2 ≠ 2 * p * a)
  (h3 : point_on_parabola p M.1 M.2)
  (h4 : point_on_parabola p M1.1 M1.2)
  (h5 : point_on_parabola p M2.1 M2.2)
  (h6 : line_through_points a b M.1 M.2 M1.1 M1.2)
  (h7 : line_through_points (-a) 0 M.1 M.2 M2.1 M2.2)
  (h8 : M1 ≠ M2) :
  line_through_points M1.1 M1.2 M2.1 M2.2 a (2 * p * a / b) :=
sorry

end fixed_point_theorem_l2086_208641


namespace average_monthly_sales_l2086_208653

def january_sales : ℝ := 110
def february_sales : ℝ := 90
def march_sales : ℝ := 70
def april_sales : ℝ := 130
def may_sales : ℝ := 50
def total_months : ℕ := 5

def total_sales : ℝ := january_sales + february_sales + march_sales + april_sales + may_sales

theorem average_monthly_sales :
  total_sales / total_months = 90 := by sorry

end average_monthly_sales_l2086_208653


namespace three_solutions_when_a_is_9_l2086_208626

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - a * x^2 + 6

-- Theorem statement
theorem three_solutions_when_a_is_9 :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, f 9 x = Real.sqrt 2 := by
  sorry

end three_solutions_when_a_is_9_l2086_208626


namespace divisibility_conditions_divisibility_conditions_2_divisibility_conditions_3_l2086_208645

theorem divisibility_conditions (n : ℤ) :
  (∃ k : ℤ, n = 225 * k + 99) ↔ (9 ∣ n ∧ 25 ∣ (n + 1)) :=
sorry

theorem divisibility_conditions_2 (n : ℤ) :
  (∃ k : ℤ, n = 3465 * k + 1649) ↔ (21 ∣ n ∧ 165 ∣ (n + 1)) :=
sorry

theorem divisibility_conditions_3 (n : ℤ) :
  (∃ m : ℤ, n = 900 * m + 774) ↔ (9 ∣ n ∧ 25 ∣ (n + 1) ∧ 4 ∣ (n + 2)) :=
sorry

end divisibility_conditions_divisibility_conditions_2_divisibility_conditions_3_l2086_208645


namespace line_through_parabola_vertex_l2086_208654

theorem line_through_parabola_vertex (a : ℝ) : 
  (∃! (x y : ℝ), y = x^2 + a^2 ∧ y = 2*x + a ∧ ∀ (x' : ℝ), x'^2 + a^2 ≥ y) ↔ (a = 0 ∨ a = 1) :=
sorry

end line_through_parabola_vertex_l2086_208654


namespace pocket_money_mode_and_median_l2086_208661

def pocket_money : List ℕ := [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5, 6]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem pocket_money_mode_and_median :
  mode pocket_money = 2 ∧ median pocket_money = 3 := by sorry

end pocket_money_mode_and_median_l2086_208661


namespace correct_arrangements_l2086_208615

/-- The number of ways to assign students to tasks with restrictions -/
def assignment_arrangements (n m : ℕ) (r : ℕ) : ℕ :=
  Nat.descFactorial n m - 2 * Nat.descFactorial (n - 1) (m - 1)

/-- Theorem stating the correct number of arrangements -/
theorem correct_arrangements :
  assignment_arrangements 6 4 2 = 240 := by
  sorry

end correct_arrangements_l2086_208615


namespace system_solution_l2086_208638

theorem system_solution (a b : ℝ) : 
  a^2 + b^2 = 25 ∧ 3*(a + b) - a*b = 15 ↔ 
  ((a = 0 ∧ b = 5) ∨ (a = 5 ∧ b = 0) ∨ (a = 4 ∧ b = -3) ∨ (a = -3 ∧ b = 4)) :=
by sorry

end system_solution_l2086_208638


namespace smallest_marble_count_l2086_208663

/-- Represents the number of marbles of each color -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the total number of marbles -/
def total_marbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green

/-- Checks if the given marble count satisfies the equal probability conditions -/
def satisfies_conditions (mc : MarbleCount) : Prop :=
  let r := mc.red
  let w := mc.white
  let b := mc.blue
  let g := mc.green
  (r * (r - 1) * (r - 2) * (r - 3)) / 24 = 
    (w * r * (r - 1) * (r - 2)) / 6 ∧
  (r * (r - 1) * (r - 2) * (r - 3)) / 24 = 
    (w * b * r * (r - 1)) / 2 ∧
  (w * b * r * (r - 1)) / 2 = 
    w * b * g * r

theorem smallest_marble_count : 
  ∃ (mc : MarbleCount), 
    satisfies_conditions mc ∧ 
    total_marbles mc = 21 ∧ 
    (∀ (mc' : MarbleCount), satisfies_conditions mc' → total_marbles mc' ≥ 21) := by
  sorry

end smallest_marble_count_l2086_208663


namespace car_selection_average_l2086_208659

theorem car_selection_average (num_cars : ℕ) (num_clients : ℕ) (selections_per_client : ℕ) 
  (h1 : num_cars = 18) 
  (h2 : num_clients = 18) 
  (h3 : selections_per_client = 3) : 
  (num_clients * selections_per_client) / num_cars = 3 := by
  sorry

end car_selection_average_l2086_208659


namespace expression_evaluation_l2086_208627

theorem expression_evaluation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / (a * b) - (a^2 + a * b) / (a^2 + b^2) = 
  (a^4 + b^4 + a^2 * b^2 - a^2 * b - a * b^2) / (a * b * (a^2 + b^2)) := by
  sorry

end expression_evaluation_l2086_208627


namespace perfect_square_floor_l2086_208646

theorem perfect_square_floor (a b : ℝ) : 
  (∀ n : ℕ+, ∃ k : ℕ, ⌊a * n + b⌋ = k^2) ↔ 
  (a = 0 ∧ ∃ k : ℕ, ∃ u : ℝ, b = k^2 + u ∧ 0 ≤ u ∧ u < 1) :=
sorry

end perfect_square_floor_l2086_208646


namespace intersection_A_complement_B_l2086_208667

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {-2, -1, 0}

theorem intersection_A_complement_B (x : Int) : 
  x ∈ (A ∩ (U \ B)) ↔ x = 1 := by
sorry

end intersection_A_complement_B_l2086_208667


namespace fruit_cost_theorem_l2086_208631

def fruit_prices (o g w f : ℝ) : Prop :=
  o + g + w + f = 24 ∧ f = 3 * o ∧ w = o - 2 * g

theorem fruit_cost_theorem :
  ∀ o g w f : ℝ, fruit_prices o g w f → g + w = 4.8 :=
by sorry

end fruit_cost_theorem_l2086_208631


namespace maaza_liters_l2086_208665

theorem maaza_liters (pepsi sprite cans : ℕ) (h1 : pepsi = 144) (h2 : sprite = 368) (h3 : cans = 281) :
  ∃ (M : ℕ), M + pepsi + sprite = cans * (M + pepsi + sprite) / cans ∧
  ∀ (M' : ℕ), M' + pepsi + sprite = cans * (M' + pepsi + sprite) / cans → M ≤ M' :=
by sorry

end maaza_liters_l2086_208665


namespace water_intake_glasses_l2086_208671

/-- Calculates the number of glasses of water needed to meet a daily water intake goal -/
theorem water_intake_glasses (daily_goal : ℝ) (glass_capacity : ℝ) : 
  daily_goal = 1.5 → glass_capacity = 0.250 → (daily_goal * 1000) / glass_capacity = 6 := by
  sorry

end water_intake_glasses_l2086_208671


namespace f_property_l2086_208664

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 3

-- State the theorem
theorem f_property (a b : ℝ) : f a b (-2) = 7 → f a b 2 = -13 := by
  sorry

end f_property_l2086_208664


namespace log_equation_solution_l2086_208650

theorem log_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1) :
  (Real.log x) / (Real.log (b^3)) + 3 * (Real.log b) / (Real.log x) = 2 → x = b^3 := by
  sorry

end log_equation_solution_l2086_208650


namespace factorial_division_l2086_208614

theorem factorial_division :
  (10 : ℕ).factorial / (5 : ℕ).factorial = 30240 :=
by
  -- Given: 10! = 3628800
  have h1 : (10 : ℕ).factorial = 3628800 := by sorry
  
  -- Definition of 5!
  have h2 : (5 : ℕ).factorial = 120 := by sorry
  
  -- Proof that 10! / 5! = 30240
  sorry

end factorial_division_l2086_208614


namespace billy_has_ten_fish_l2086_208697

def fish_problem (billy_fish : ℕ) : Prop :=
  let tony_fish := 3 * billy_fish
  let sarah_fish := tony_fish + 5
  let bobby_fish := 2 * sarah_fish
  billy_fish + tony_fish + sarah_fish + bobby_fish = 145

theorem billy_has_ten_fish :
  ∃ (billy_fish : ℕ), fish_problem billy_fish ∧ billy_fish = 10 := by
  sorry

end billy_has_ten_fish_l2086_208697


namespace journey_remaining_distance_l2086_208610

/-- The remaining distance to be driven in a journey. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Proof that the remaining distance is 3610 miles. -/
theorem journey_remaining_distance :
  remaining_distance 9475 5865 = 3610 := by
  sorry

end journey_remaining_distance_l2086_208610


namespace part_one_part_two_l2086_208634

/-- Definition of proposition p -/
def p (x a : ℝ) : Prop := x^2 - 2*a*x - 3*a^2 < 0

/-- Definition of proposition q -/
def q (x : ℝ) : Prop := 2 ≤ x ∧ x < 4

/-- Theorem for part (1) -/
theorem part_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 ≤ x ∧ x < 3) :=
sorry

/-- Theorem for part (2) -/
theorem part_two :
  ∀ a : ℝ, (a > 0 ∧ (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x)) ↔ (4/3 ≤ a) :=
sorry

end part_one_part_two_l2086_208634


namespace stacy_berries_l2086_208690

theorem stacy_berries (total : ℕ) (x : ℕ) : total = 1100 → x + 2*x + 8*x = total → 8*x = 800 :=
by
  sorry

end stacy_berries_l2086_208690


namespace zaras_estimate_l2086_208647

theorem zaras_estimate (x y z : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : z > 0) :
  (x + z) - (y + z) = x - y := by sorry

end zaras_estimate_l2086_208647


namespace solve_system_l2086_208603

theorem solve_system (c d : ℤ) 
  (eq1 : 5 + c = 7 - d) 
  (eq2 : 6 + d = 10 + c) : 
  5 - c = 6 := by
sorry

end solve_system_l2086_208603


namespace triangle_n_values_l2086_208688

theorem triangle_n_values :
  let valid_n (n : ℕ) : Prop :=
    3*n + 15 > 3*n + 10 ∧ 
    3*n + 10 > 4*n ∧ 
    4*n + (3*n + 10) > 3*n + 15 ∧ 
    4*n + (3*n + 15) > 3*n + 10 ∧ 
    (3*n + 10) + (3*n + 15) > 4*n
  ∃! (s : Finset ℕ), (∀ n ∈ s, valid_n n) ∧ s.card = 8 :=
by sorry

end triangle_n_values_l2086_208688


namespace sum_of_three_consecutive_cubes_divisible_by_nine_l2086_208683

theorem sum_of_three_consecutive_cubes_divisible_by_nine (n : ℕ) (h : n > 1) :
  ∃ k : ℤ, (n - 1)^3 + n^3 + (n + 1)^3 = 9 * k := by
sorry

end sum_of_three_consecutive_cubes_divisible_by_nine_l2086_208683


namespace parabola_and_tangent_lines_l2086_208675

-- Define the parabola
structure Parabola where
  -- Standard form equation: x² = 2py
  p : ℝ
  -- Vertex at origin
  vertex : (ℝ × ℝ) := (0, 0)
  -- Focus on y-axis
  focus : (ℝ × ℝ) := (0, p)

-- Define a point on the parabola
def point_on_parabola (par : Parabola) (x y : ℝ) : Prop :=
  x^2 = 2 * par.p * y

-- Define a line
structure Line where
  m : ℝ
  b : ℝ

-- Define a point on a line
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.b

-- Define when a line intersects a parabola at a single point
def single_intersection (par : Parabola) (l : Line) : Prop :=
  ∃! (x y : ℝ), point_on_parabola par x y ∧ point_on_line l x y

-- Theorem statement
theorem parabola_and_tangent_lines :
  ∃ (par : Parabola),
    -- Parabola passes through (2, 1)
    point_on_parabola par 2 1 ∧
    -- Standard equation is x² = 4y
    par.p = 2 ∧
    -- Lines x = 2 and x - y - 1 = 0 are the only lines through (2, 1)
    -- that intersect the parabola at a single point
    (∀ (l : Line),
      point_on_line l 2 1 →
      single_intersection par l ↔ (l.m = 0 ∧ l.b = 2) ∨ (l.m = 1 ∧ l.b = -1)) :=
sorry

end parabola_and_tangent_lines_l2086_208675


namespace eleventh_term_is_768_l2086_208605

/-- A geometric sequence with given conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = q * a n
  a_4 : a 4 = 6
  a_7 : a 7 = 48

/-- The 11th term of the geometric sequence is 768 -/
theorem eleventh_term_is_768 (seq : GeometricSequence) : seq.a 11 = 768 := by
  sorry

end eleventh_term_is_768_l2086_208605


namespace count_integers_satisfying_inequality_l2086_208602

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset ℤ), (∀ n : ℤ, n ∈ S ↔ (Real.sqrt (n + 1) ≤ Real.sqrt (5*n - 7) ∧ Real.sqrt (5*n - 7) < Real.sqrt (3*n + 6))) ∧ S.card = 5 :=
sorry

end count_integers_satisfying_inequality_l2086_208602


namespace equation_solution_l2086_208630

theorem equation_solution :
  ∀ y : ℝ, (2012 + y)^2 = 2*y^2 ↔ y = 2012*(Real.sqrt 2 + 1) ∨ y = -2012*(Real.sqrt 2 - 1) := by
  sorry

end equation_solution_l2086_208630


namespace students_neither_art_nor_music_l2086_208651

theorem students_neither_art_nor_music 
  (total : ℕ) (art : ℕ) (music : ℕ) (both : ℕ) :
  total = 75 →
  art = 45 →
  music = 50 →
  both = 30 →
  total - (art + music - both) = 10 :=
by
  sorry

end students_neither_art_nor_music_l2086_208651


namespace total_problems_l2086_208635

def marvin_yesterday : ℕ := 40

def marvin_today (x : ℕ) : ℕ := 3 * x

def arvin_daily (x : ℕ) : ℕ := 2 * x

theorem total_problems :
  marvin_yesterday + marvin_today marvin_yesterday +
  arvin_daily marvin_yesterday + arvin_daily (marvin_today marvin_yesterday) = 480 := by
  sorry

end total_problems_l2086_208635


namespace touring_plans_count_l2086_208644

def num_destinations : Nat := 3
def num_students : Nat := 4

def total_assignments : Nat := num_destinations ^ num_students

def assignments_without_specific_destination : Nat := (num_destinations - 1) ^ num_students

theorem touring_plans_count : 
  total_assignments - assignments_without_specific_destination = 65 := by
  sorry

end touring_plans_count_l2086_208644


namespace combination_equation_solution_l2086_208628

theorem combination_equation_solution : 
  ∃! (n : ℕ), n > 0 ∧ Nat.choose (n + 1) (n - 1) = 21 := by sorry

end combination_equation_solution_l2086_208628


namespace monopoly_produces_durable_iff_lowquality_cost_gt_six_l2086_208623

/-- Represents a coffee machine producer -/
structure Producer where
  isDurable : Bool
  cost : ℝ

/-- Represents a consumer of coffee machines -/
structure Consumer where
  benefit : ℝ
  periods : ℕ

/-- Represents the market for coffee machines -/
inductive Market
  | Monopoly
  | PerfectlyCompetitive

/-- Define the conditions for the coffee machine problem -/
def coffeeMachineProblem (c : Consumer) (pd : Producer) (pl : Producer) (m : Market) : Prop :=
  c.periods = 2 ∧ 
  c.benefit = 20 ∧
  pd.isDurable = true ∧
  pd.cost = 12 ∧
  pl.isDurable = false

/-- Theorem: A monopoly will produce only durable coffee machines if and only if 
    the average cost of producing a low-quality coffee machine is greater than 6 monetary units -/
theorem monopoly_produces_durable_iff_lowquality_cost_gt_six 
  (c : Consumer) (pd : Producer) (pl : Producer) (m : Market) :
  coffeeMachineProblem c pd pl Market.Monopoly →
  (∀ S, pl.cost = S → (pd.cost < pl.cost ↔ S > 6)) :=
sorry

end monopoly_produces_durable_iff_lowquality_cost_gt_six_l2086_208623


namespace min_value_theorem_l2086_208694

theorem min_value_theorem (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + 2*y = 1) :
  (1 / (x + 1) + 1 / y) ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end min_value_theorem_l2086_208694


namespace coins_left_l2086_208600

def pennies : ℕ := 42
def nickels : ℕ := 36
def dimes : ℕ := 15
def donated : ℕ := 66

theorem coins_left : pennies + nickels + dimes - donated = 27 := by
  sorry

end coins_left_l2086_208600


namespace circle_line_intersection_l2086_208658

/-- The number of distinct points common to the circle x^2 + y^2 = 16 and the vertical line x = 4 is one. -/
theorem circle_line_intersection :
  ∃! p : ℝ × ℝ, (p.1^2 + p.2^2 = 16) ∧ (p.1 = 4) := by sorry

end circle_line_intersection_l2086_208658


namespace not_always_prime_l2086_208684

theorem not_always_prime : ∃ n : ℤ, ¬(Nat.Prime (Int.natAbs (n^2 + n + 41))) := by sorry

end not_always_prime_l2086_208684


namespace intersection_point_l2086_208619

/-- The line defined by the equation y = -7x + 9 -/
def line (x : ℝ) : ℝ := -7 * x + 9

/-- The y-axis is defined as the set of points with x-coordinate equal to 0 -/
def y_axis (p : ℝ × ℝ) : Prop := p.1 = 0

/-- A point is on the line if its y-coordinate equals the line function at its x-coordinate -/
def on_line (p : ℝ × ℝ) : Prop := p.2 = line p.1

theorem intersection_point :
  ∃! p : ℝ × ℝ, y_axis p ∧ on_line p ∧ p = (0, 9) := by sorry

end intersection_point_l2086_208619


namespace money_collection_l2086_208649

theorem money_collection (households_per_day : ℕ) (days : ℕ) (total_amount : ℕ) :
  households_per_day = 20 →
  days = 5 →
  total_amount = 2000 →
  (households_per_day * days) / 2 * (total_amount / ((households_per_day * days) / 2)) = 40 :=
by sorry

end money_collection_l2086_208649


namespace total_students_agreed_l2086_208673

def third_grade_total : ℕ := 256
def fourth_grade_total : ℕ := 525
def fifth_grade_total : ℕ := 410
def sixth_grade_total : ℕ := 600

def third_grade_percentage : ℚ := 60 / 100
def fourth_grade_percentage : ℚ := 45 / 100
def fifth_grade_percentage : ℚ := 35 / 100
def sixth_grade_percentage : ℚ := 55 / 100

def round_to_nearest (x : ℚ) : ℕ := 
  (x + 1/2).floor.toNat

theorem total_students_agreed : 
  round_to_nearest (third_grade_percentage * third_grade_total) +
  round_to_nearest (fourth_grade_percentage * fourth_grade_total) +
  round_to_nearest (fifth_grade_percentage * fifth_grade_total) +
  round_to_nearest (sixth_grade_percentage * sixth_grade_total) = 864 := by
  sorry

end total_students_agreed_l2086_208673


namespace trigonometric_identities_l2086_208629

open Real

theorem trigonometric_identities (α : ℝ) 
  (h : (sin α - 2 * cos α) / (sin α + 2 * cos α) = 3) : 
  ((sin α + 2 * cos α) / (5 * cos α - sin α) = -2/9) ∧
  ((sin α + cos α)^2 = 9/17) := by
  sorry

end trigonometric_identities_l2086_208629


namespace inverse_of_A_l2086_208607

def A : Matrix (Fin 3) (Fin 3) ℚ := !![1, 2, 3; 0, -1, 2; 3, 0, 7]

def A_inverse : Matrix (Fin 3) (Fin 3) ℚ := !![-1/2, -1, 1/2; 3/7, -1/7, -1/7; 3/14, 3/7, -1/14]

theorem inverse_of_A : A⁻¹ = A_inverse := by sorry

end inverse_of_A_l2086_208607


namespace complex_modulus_inequality_l2086_208696

theorem complex_modulus_inequality (x y : ℝ) : 
  let z : ℂ := Complex.mk x y
  ‖z‖ ≤ |x| + |y| := by sorry

end complex_modulus_inequality_l2086_208696


namespace z_in_second_quadrant_l2086_208677

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation (1-i)z = 2i
def equation (z : ℂ) : Prop := (1 - i) * z = 2 * i

-- Define what it means for a complex number to be in the second quadrant
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- State the theorem
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ in_second_quadrant z :=
sorry

end z_in_second_quadrant_l2086_208677


namespace quadratic_complete_square_l2086_208622

theorem quadratic_complete_square (x : ℝ) : 
  (∃ (b c : ℤ), (x + b : ℝ)^2 = c ∧ x^2 - 10*x + 15 = 0) → 
  (∃ (b c : ℤ), (x + b : ℝ)^2 = c ∧ b + c = 5) :=
by sorry

end quadratic_complete_square_l2086_208622


namespace subtraction_mistake_l2086_208698

/-- Given two two-digit numbers, if the first number is misread by increasing both digits by 3
    and the incorrect subtraction results in 44, then the correct subtraction equals 11. -/
theorem subtraction_mistake (A B C D : ℕ) : 
  (A < 10) → (B < 10) → (C < 10) → (D < 10) →
  ((10 * (A + 3) + (B + 3)) - (10 * C + D) = 44) →
  ((10 * A + B) - (10 * C + D) = 11) := by
sorry

end subtraction_mistake_l2086_208698


namespace dad_steps_l2086_208643

/-- Represents the number of steps taken by each person -/
structure Steps where
  dad : ℕ
  masha : ℕ
  yasha : ℕ

/-- Defines the relationship between Dad's and Masha's steps -/
def dad_masha_ratio (s : Steps) : Prop :=
  5 * s.dad = 3 * s.masha

/-- Defines the relationship between Masha's and Yasha's steps -/
def masha_yasha_ratio (s : Steps) : Prop :=
  5 * s.masha = 3 * s.yasha

/-- States that Masha and Yasha together took 400 steps -/
def total_masha_yasha (s : Steps) : Prop :=
  s.masha + s.yasha = 400

theorem dad_steps (s : Steps) 
  (h1 : dad_masha_ratio s)
  (h2 : masha_yasha_ratio s)
  (h3 : total_masha_yasha s) :
  s.dad = 90 := by
  sorry

end dad_steps_l2086_208643


namespace sum_of_fractions_equals_sum_of_roots_l2086_208686

theorem sum_of_fractions_equals_sum_of_roots : 
  let T := 1 / (Real.sqrt 10 - Real.sqrt 8) + 
           1 / (Real.sqrt 8 - Real.sqrt 6) + 
           1 / (Real.sqrt 6 - Real.sqrt 4)
  T = (Real.sqrt 10 + 2 * Real.sqrt 8 + 2 * Real.sqrt 6 + 2) / 2 := by
  sorry

end sum_of_fractions_equals_sum_of_roots_l2086_208686


namespace cycling_distance_l2086_208604

/-- Proves that cycling at a constant rate of 4 miles per hour for 2 hours results in a total distance of 8 miles. -/
theorem cycling_distance (rate : ℝ) (time : ℝ) (distance : ℝ) : 
  rate = 4 → time = 2 → distance = rate * time → distance = 8 := by
  sorry

#check cycling_distance

end cycling_distance_l2086_208604


namespace rectangle_dimensions_l2086_208606

/-- Given a rectangular frame made with 240 cm of wire, where the ratio of
    length:width:height is 3:2:1, prove that the dimensions are 30 cm, 20 cm,
    and 10 cm respectively. -/
theorem rectangle_dimensions (total_wire : ℝ) (length width height : ℝ)
    (h1 : total_wire = 240)
    (h2 : length + width + height = total_wire / 4)
    (h3 : length = 3 * height)
    (h4 : width = 2 * height) :
    length = 30 ∧ width = 20 ∧ height = 10 := by
  sorry

end rectangle_dimensions_l2086_208606


namespace slope_intercept_sum_l2086_208691

/-- Given two points A and B on a line, prove that the sum of the line's slope and y-intercept is 10. -/
theorem slope_intercept_sum (A B : ℝ × ℝ) : 
  A = (5, 6) → B = (8, 3) → 
  let m := (B.2 - A.2) / (B.1 - A.1)
  let b := A.2 - m * A.1
  m + b = 10 := by sorry

end slope_intercept_sum_l2086_208691


namespace seven_place_value_difference_l2086_208611

def number : ℕ := 54179759

def first_seven_place_value : ℕ := 10000
def second_seven_place_value : ℕ := 10

def first_seven_value : ℕ := 7 * first_seven_place_value
def second_seven_value : ℕ := 7 * second_seven_place_value

theorem seven_place_value_difference : 
  first_seven_value - second_seven_value = 69930 := by
  sorry

end seven_place_value_difference_l2086_208611


namespace line_translation_l2086_208648

/-- Given a line y = -2x + 1, translating it upwards by 2 units results in y = -2x + 3 -/
theorem line_translation (x y : ℝ) : 
  (y = -2*x + 1) → (y + 2 = -2*x + 3) := by sorry

end line_translation_l2086_208648


namespace sum_of_squares_zero_implies_sum_l2086_208680

theorem sum_of_squares_zero_implies_sum (x y z : ℝ) : 
  2 * (x - 2)^2 + 2 * (y - 3)^2 + 2 * (z - 6)^2 = 0 → x + y + z = 11 := by
  sorry

end sum_of_squares_zero_implies_sum_l2086_208680


namespace greatest_value_cubic_inequality_l2086_208692

theorem greatest_value_cubic_inequality :
  let f : ℝ → ℝ := λ b => -b^3 + b^2 + 7*b - 10
  ∃ (max_b : ℝ), max_b = 4 + Real.sqrt 6 ∧
    (∀ b : ℝ, f b ≥ 0 → b ≤ max_b) ∧
    f max_b ≥ 0 :=
by sorry

end greatest_value_cubic_inequality_l2086_208692


namespace jane_age_proof_l2086_208662

/-- Represents Jane's age when she started babysitting -/
def start_age : ℕ := 20

/-- Represents the number of years since Jane stopped babysitting -/
def years_since_stop : ℕ := 10

/-- Represents the current age of the oldest person Jane could have babysat -/
def oldest_babysat_current_age : ℕ := 22

/-- Represents Jane's current age -/
def jane_current_age : ℕ := 34

theorem jane_age_proof :
  ∀ (jane_age : ℕ),
    jane_age ≥ start_age →
    (oldest_babysat_current_age - years_since_stop) * 2 ≤ jane_age - years_since_stop →
    jane_age = jane_current_age := by
  sorry

end jane_age_proof_l2086_208662


namespace zoo_line_theorem_l2086_208687

/-- The number of ways to arrange 6 people in a line with specific conditions -/
def zoo_line_arrangements : ℕ := 24

/-- Two fathers in a group of 6 people -/
def fathers : ℕ := 2

/-- Two mothers in a group of 6 people -/
def mothers : ℕ := 2

/-- Two children in a group of 6 people -/
def children : ℕ := 2

/-- Total number of people in the group -/
def total_people : ℕ := fathers + mothers + children

theorem zoo_line_theorem :
  zoo_line_arrangements = 24 :=
sorry

end zoo_line_theorem_l2086_208687


namespace sine_cosine_relation_l2086_208613

theorem sine_cosine_relation (x : ℝ) (h : Real.cos (5 * Real.pi / 6 - x) = 1 / 3) :
  Real.sin (x - Real.pi / 3) = 1 / 3 := by
  sorry

end sine_cosine_relation_l2086_208613


namespace parabola_directrix_l2086_208652

/-- Given a parabola with equation 16y^2 = x, its directrix equation is x = -1/64 -/
theorem parabola_directrix (x y : ℝ) : 
  (16 * y^2 = x) → (∃ (k : ℝ), k = -1/64 ∧ k = x) := by
sorry

end parabola_directrix_l2086_208652


namespace paul_total_crayons_l2086_208685

/-- The number of crayons Paul initially had -/
def initial_crayons : ℝ := 479.0

/-- The number of additional crayons Paul received -/
def additional_crayons : ℝ := 134.0

/-- The total number of crayons Paul now has -/
def total_crayons : ℝ := initial_crayons + additional_crayons

/-- Theorem stating that Paul now has 613.0 crayons -/
theorem paul_total_crayons : total_crayons = 613.0 := by
  sorry

end paul_total_crayons_l2086_208685


namespace exists_divisible_pair_l2086_208633

/-- A function that checks if a number is three-digit -/
def isThreeDigit (n : ℕ) : Prop := n ≥ 100 ∧ n ≤ 999

/-- A function that checks if a number is two-digit -/
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

/-- A function that checks if a number uses only the digits 1, 2, 3, 4, 5 -/
def usesOnlyGivenDigits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ∈ [1, 2, 3, 4, 5]

/-- The main theorem -/
theorem exists_divisible_pair :
  ∃ (a b : ℕ),
    isThreeDigit a ∧
    isTwoDigit b ∧
    usesOnlyGivenDigits a ∧
    usesOnlyGivenDigits b ∧
    a % b = 0 :=
  sorry

end exists_divisible_pair_l2086_208633


namespace concert_attendance_l2086_208674

theorem concert_attendance (adults : ℕ) 
  (h1 : 3 * adults = children)
  (h2 : 7 * adults + 3 * children = 6000) :
  adults + children = 1500 :=
by sorry

end concert_attendance_l2086_208674


namespace break_even_price_correct_l2086_208699

/-- The price per kilogram to sell fruits without loss or profit -/
def break_even_price : ℝ := 2.6

/-- The price per jin that results in a loss -/
def loss_price : ℝ := 1.2

/-- The price per jin that results in a profit -/
def profit_price : ℝ := 1.5

/-- The amount of loss when selling at loss_price -/
def loss_amount : ℝ := 4

/-- The amount of profit when selling at profit_price -/
def profit_amount : ℝ := 8

/-- Conversion factor from jin to kilogram -/
def jin_to_kg : ℝ := 0.5

theorem break_even_price_correct :
  ∃ (weight : ℝ),
    weight * (break_even_price * jin_to_kg) = weight * loss_price + loss_amount ∧
    weight * (break_even_price * jin_to_kg) = weight * profit_price - profit_amount :=
by sorry

end break_even_price_correct_l2086_208699


namespace unique_congruence_in_range_l2086_208637

theorem unique_congruence_in_range : ∃! n : ℤ,
  5 ≤ n ∧ n ≤ 10 ∧ n ≡ 10543 [ZMOD 7] ∧ n = 8 := by
  sorry

end unique_congruence_in_range_l2086_208637


namespace sum_three_numbers_l2086_208681

/-- Given three numbers a, b, and c, and a value T, satisfying the following conditions:
    1. a + b + c = 84
    2. a - 5 = T
    3. b + 9 = T
    4. 5 * c = T
    Prove that T = 40 -/
theorem sum_three_numbers (a b c T : ℝ) 
  (sum_eq : a + b + c = 84)
  (a_minus : a - 5 = T)
  (b_plus : b + 9 = T)
  (c_times : 5 * c = T) : 
  T = 40 := by
  sorry

end sum_three_numbers_l2086_208681


namespace decimal_difference_value_l2086_208640

/-- The repeating decimal 0.0̅6̅ -/
def repeating_decimal : ℚ := 2 / 33

/-- The terminating decimal 0.06 -/
def terminating_decimal : ℚ := 6 / 100

/-- The difference between the repeating decimal 0.0̅6̅ and the terminating decimal 0.06 -/
def decimal_difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference_value : decimal_difference = 2 / 3300 := by
  sorry

end decimal_difference_value_l2086_208640


namespace remainder_theorem_l2086_208689

/-- Given a polynomial q(x) satisfying specific conditions, 
    prove properties about its remainder when divided by (x - 3)(x + 2)(x - 4) -/
theorem remainder_theorem (q : ℝ → ℝ) (h1 : q 3 = 2) (h2 : q (-2) = -3) (h3 : q 4 = 6) :
  ∃ (s : ℝ → ℝ), 
    (∀ x, q x = (x - 3) * (x + 2) * (x - 4) * (q x / ((x - 3) * (x + 2) * (x - 4))) + s x) ∧
    (∀ x, s x = 1/2 * x^2 + 1/2 * x - 4) ∧
    (s 5 = 11) :=
by sorry

end remainder_theorem_l2086_208689


namespace binomial_coefficient_two_l2086_208679

theorem binomial_coefficient_two (n : ℕ) (h : n > 0) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l2086_208679


namespace unique_solution_implies_a_greater_than_one_l2086_208670

-- Define the function f(x) = 2ax^2 - x - 1
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - x - 1

-- State the theorem
theorem unique_solution_implies_a_greater_than_one :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) → a > 1 := by
  sorry

end unique_solution_implies_a_greater_than_one_l2086_208670


namespace base_b_is_seven_l2086_208618

/-- Given that in base b, the square of 22_b is 514_b, prove that b = 7 -/
theorem base_b_is_seven (b : ℕ) (h : b > 1) : 
  (2 * b + 2)^2 = 5 * b^2 + b + 4 → b = 7 := by
  sorry

end base_b_is_seven_l2086_208618
