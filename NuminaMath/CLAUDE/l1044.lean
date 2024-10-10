import Mathlib

namespace video_distribution_solution_l1044_104409

/-- Represents the problem of distributing video content across discs -/
def VideoDistribution (total_minutes : ℝ) (max_capacity : ℝ) : Prop :=
  ∃ (num_discs : ℕ) (minutes_per_disc : ℝ),
    num_discs > 0 ∧
    num_discs = ⌈total_minutes / max_capacity⌉ ∧
    minutes_per_disc = total_minutes / num_discs ∧
    minutes_per_disc ≤ max_capacity

/-- Theorem stating the solution to the video distribution problem -/
theorem video_distribution_solution :
  VideoDistribution 495 65 →
  ∃ (num_discs : ℕ) (minutes_per_disc : ℝ),
    num_discs = 8 ∧ minutes_per_disc = 61.875 := by
  sorry

#check video_distribution_solution

end video_distribution_solution_l1044_104409


namespace subset_condition_implies_a_geq_three_l1044_104476

/-- Given a > 0, if the set A is a subset of set B, then a ≥ 3 -/
theorem subset_condition_implies_a_geq_three (a : ℝ) (h : a > 0) :
  ({x : ℝ | (x - 2) * (x - 3 * a - 2) < 0} ⊆ {x : ℝ | (x - 1) * (x - a^2 - 2) < 0}) →
  a ≥ 3 := by
  sorry

end subset_condition_implies_a_geq_three_l1044_104476


namespace expense_increase_percentage_is_ten_percent_l1044_104465

/-- Calculates the percentage increase in monthly expenses given the initial salary,
    savings rate, and new savings amount. -/
def calculate_expense_increase_percentage (salary : ℚ) (savings_rate : ℚ) (new_savings : ℚ) : ℚ :=
  let original_savings := salary * savings_rate
  let original_expenses := salary - original_savings
  let additional_expense := original_savings - new_savings
  (additional_expense / original_expenses) * 100

/-- Theorem stating that for the given conditions, the expense increase percentage is 10% -/
theorem expense_increase_percentage_is_ten_percent :
  calculate_expense_increase_percentage 20000 (1/10) 200 = 10 := by
  sorry

end expense_increase_percentage_is_ten_percent_l1044_104465


namespace correct_answers_for_given_exam_l1044_104447

/-- Represents an exam with a fixed number of questions and scoring rules. -/
structure Exam where
  totalQuestions : ℕ
  correctScore : ℕ
  wrongScore : ℤ

/-- Represents a student's exam attempt. -/
structure ExamAttempt where
  exam : Exam
  correctAnswers : ℕ
  wrongAnswers : ℕ
  totalScore : ℤ

/-- Calculates the total score for an exam attempt. -/
def calculateScore (attempt : ExamAttempt) : ℤ :=
  (attempt.correctAnswers : ℤ) * attempt.exam.correctScore - attempt.wrongAnswers * (-attempt.exam.wrongScore)

/-- Theorem stating the correct number of answers for the given exam conditions. -/
theorem correct_answers_for_given_exam :
  ∀ (attempt : ExamAttempt),
    attempt.exam.totalQuestions = 75 →
    attempt.exam.correctScore = 4 →
    attempt.exam.wrongScore = -1 →
    attempt.correctAnswers + attempt.wrongAnswers = attempt.exam.totalQuestions →
    calculateScore attempt = 125 →
    attempt.correctAnswers = 40 := by
  sorry


end correct_answers_for_given_exam_l1044_104447


namespace range_of_a_l1044_104408

theorem range_of_a (p : Prop) (h : p) : 
  (∀ x : ℝ, x ∈ Set.Ioo 1 2 → Real.exp x - a ≤ 0) → a ∈ Set.Ici (Real.exp 2) :=
by
  sorry

end range_of_a_l1044_104408


namespace square_root_computation_l1044_104436

theorem square_root_computation : (3 * Real.sqrt 15625 - 5)^2 = 136900 := by
  sorry

end square_root_computation_l1044_104436


namespace equal_celsius_fahrenheit_temp_l1044_104489

/-- Converts Celsius temperature to Fahrenheit -/
def celsius_to_fahrenheit (c : ℝ) : ℝ := 1.8 * c + 32

/-- Theorem stating that there exists a unique temperature where Celsius and Fahrenheit are equal -/
theorem equal_celsius_fahrenheit_temp :
  ∃! t : ℝ, t = celsius_to_fahrenheit t :=
by
  sorry

end equal_celsius_fahrenheit_temp_l1044_104489


namespace sqrt_equation_solution_l1044_104432

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 10 → x = 105 := by
  sorry

end sqrt_equation_solution_l1044_104432


namespace rotation_180_maps_points_l1044_104444

def rotation_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem rotation_180_maps_points :
  let C : ℝ × ℝ := (3, -2)
  let D : ℝ × ℝ := (2, -5)
  let C' : ℝ × ℝ := (-3, 2)
  let D' : ℝ × ℝ := (-2, 5)
  rotation_180 C = C' ∧ rotation_180 D = D' :=
by sorry

end rotation_180_maps_points_l1044_104444


namespace perpendicular_AC_AD_l1044_104494

/-- The curve E in the xy-plane -/
def E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + 3 * p.2^2 / 4 = 1 ∧ p.1 ≠ 2 ∧ p.1 ≠ -2}

/-- Point A -/
def A : ℝ × ℝ := (-2, 0)

/-- Point Q -/
def Q : ℝ × ℝ := (-1, 0)

/-- A line with non-zero slope passing through Q -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * (p.1 + 1) ∧ m ≠ 0}

/-- The intersection points of the line and curve E -/
def intersection (m : ℝ) : Set (ℝ × ℝ) :=
  E ∩ line_through_Q m

theorem perpendicular_AC_AD (m : ℝ) 
  (hm : m ≠ 0) 
  (h_intersect : ∃ C D, C ∈ intersection m ∧ D ∈ intersection m ∧ C ≠ D) :
  ∀ C D, C ∈ intersection m → D ∈ intersection m → C ≠ D →
  (C.1 + 2) * (D.1 + 2) + C.2 * D.2 = 0 :=
sorry

end perpendicular_AC_AD_l1044_104494


namespace missile_time_equation_l1044_104422

/-- Represents the speed of the missile in Mach -/
def missile_speed : ℝ := 26

/-- Represents the conversion factor from Mach to meters per second -/
def mach_to_mps : ℝ := 340

/-- Represents the distance to the target in kilometers -/
def target_distance : ℝ := 12000

/-- Represents the time taken to reach the target in minutes -/
def time_to_target : ℝ → ℝ := λ x => x

/-- Theorem stating the equation for the time taken by the missile to reach the target -/
theorem missile_time_equation :
  ∀ x : ℝ, (missile_speed * mach_to_mps * 60 * time_to_target x) / 1000 = target_distance * 1000 :=
by sorry

end missile_time_equation_l1044_104422


namespace polynomial_equality_l1044_104475

theorem polynomial_equality (x y : ℝ) (h : x + y = -1) :
  x^4 + 5*x^3*y + x^2*y + 8*x^2*y^2 + x*y^2 + 5*x*y^3 + y^4 = 1 := by
  sorry

end polynomial_equality_l1044_104475


namespace expression_value_l1044_104407

theorem expression_value (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 1) :
  (2*a + 3*b - 2*a*b) - (a + 4*b + a*b) - (3*a*b + 2*b - 2*a) = 6 := by
  sorry

end expression_value_l1044_104407


namespace bowling_ball_weight_proof_l1044_104449

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 15.625

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 25

theorem bowling_ball_weight_proof :
  (8 * bowling_ball_weight = 5 * canoe_weight) ∧
  (4 * canoe_weight = 100) →
  bowling_ball_weight = 15.625 := by sorry

end bowling_ball_weight_proof_l1044_104449


namespace no_solution_when_k_equals_five_l1044_104453

theorem no_solution_when_k_equals_five :
  ∀ x : ℝ, x ≠ 2 → x ≠ 6 → (x - 1) / (x - 2) ≠ (x - 5) / (x - 6) :=
by sorry

end no_solution_when_k_equals_five_l1044_104453


namespace multiples_equality_l1044_104496

/-- The average of the first 7 positive multiples of 5 -/
def a : ℚ := (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7

/-- The median of the first 3 positive multiples of n -/
def b (n : ℕ+) : ℚ := 2 * n

/-- Theorem stating that if a^2 - b^2 = 0, then n = 10 -/
theorem multiples_equality (n : ℕ+) : a^2 - (b n)^2 = 0 → n = 10 := by
  sorry

end multiples_equality_l1044_104496


namespace problem_statement_l1044_104479

theorem problem_statement : 103^4 - 4*103^3 + 6*103^2 - 4*103 + 1 = 108243216 := by
  sorry

end problem_statement_l1044_104479


namespace doctors_lawyers_ratio_l1044_104443

theorem doctors_lawyers_ratio (d l : ℕ) (h_group_avg : (40 * d + 55 * l) / (d + l) = 45) : d = 2 * l := by
  sorry

end doctors_lawyers_ratio_l1044_104443


namespace line_intersection_theorem_l1044_104418

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of a line being contained in a plane
variable (contained_in : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the property of a line intersecting another line
variable (intersects : Line → Line → Prop)

theorem line_intersection_theorem 
  (a b m : Line) (α β : Plane)
  (h1 : skew a b)
  (h2 : contained_in a α)
  (h3 : contained_in b β)
  (h4 : intersect α β = m) :
  intersects m a ∨ intersects m b :=
sorry

end line_intersection_theorem_l1044_104418


namespace garden_size_l1044_104466

theorem garden_size (garden_size fruit_size vegetable_size strawberry_size : ℝ) : 
  fruit_size = vegetable_size →
  garden_size = fruit_size + vegetable_size →
  strawberry_size = fruit_size / 4 →
  strawberry_size = 8 →
  garden_size = 64 := by
sorry

end garden_size_l1044_104466


namespace fraction_is_composite_l1044_104428

theorem fraction_is_composite : ¬ Nat.Prime ((5^125 - 1) / (5^25 - 1)) := by
  sorry

end fraction_is_composite_l1044_104428


namespace range_of_a_l1044_104456

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l1044_104456


namespace stone_145_is_1_l1044_104441

/-- Represents the counting pattern for stones -/
def stone_count (n : ℕ) : ℕ :=
  if n ≤ 10 then n
  else if n ≤ 19 then 20 - n
  else stone_count ((n - 1) % 18 + 1)

/-- The theorem stating that the 145th count corresponds to the first stone -/
theorem stone_145_is_1 : stone_count 145 = 1 := by
  sorry

end stone_145_is_1_l1044_104441


namespace five_sixteenths_decimal_l1044_104437

theorem five_sixteenths_decimal : (5 : ℚ) / 16 = (3125 : ℚ) / 10000 := by sorry

end five_sixteenths_decimal_l1044_104437


namespace digit_150_of_11_13_l1044_104460

/-- The decimal representation of 11/13 has a repeating sequence of 6 digits. -/
def decimal_rep_11_13 : List Nat := [8, 4, 6, 1, 5, 3]

/-- The 150th digit after the decimal point in the decimal representation of 11/13 is 3. -/
theorem digit_150_of_11_13 : 
  decimal_rep_11_13[150 % decimal_rep_11_13.length] = 3 := by sorry

end digit_150_of_11_13_l1044_104460


namespace solve_system_l1044_104434

theorem solve_system (a b : ℚ) (h1 : a + a/4 = 3) (h2 : b - 2*a = 1) : a = 12/5 ∧ b = 29/5 := by
  sorry

end solve_system_l1044_104434


namespace fruit_basket_total_cost_l1044_104425

def fruit_basket_cost (banana_count : ℕ) (apple_count : ℕ) (strawberry_count : ℕ) (avocado_count : ℕ) (grape_bunch_count : ℕ) : ℕ :=
  let banana_price := 1
  let apple_price := 2
  let strawberry_price_per_12 := 4
  let avocado_price := 3
  let grape_half_bunch_price := 2

  banana_count * banana_price +
  apple_count * apple_price +
  (strawberry_count / 12) * strawberry_price_per_12 +
  avocado_count * avocado_price +
  grape_bunch_count * (2 * grape_half_bunch_price)

theorem fruit_basket_total_cost :
  fruit_basket_cost 4 3 24 2 1 = 28 := by
  sorry

end fruit_basket_total_cost_l1044_104425


namespace angle_315_same_terminal_side_as_negative_45_l1044_104411

-- Define a function to represent angles with the same terminal side
def sameTerminalSide (θ : ℝ) (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + θ

-- State the theorem
theorem angle_315_same_terminal_side_as_negative_45 :
  sameTerminalSide 315 (-45) := by
  sorry

end angle_315_same_terminal_side_as_negative_45_l1044_104411


namespace train_speed_ratio_l1044_104431

theorem train_speed_ratio (v1 v2 : ℝ) (t1 t2 : ℝ) (h1 : t1 = 4) (h2 : t2 = 36) :
  v1 * t1 / (v2 * t2) = 1 / 9 → v1 = v2 := by
  sorry

end train_speed_ratio_l1044_104431


namespace line_intersection_l1044_104455

def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + (a + 2) * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 = 0

def not_parallel (a : ℝ) : Prop :=
  ¬ (∃ k : ℝ, k ≠ 0 ∧ a = k * a ∧ (a + 2) = -k ∧ 1 = k * 2)

theorem line_intersection (a : ℝ) :
  not_parallel a → a = 0 ∨ a = -3 := by sorry

end line_intersection_l1044_104455


namespace base_10_1234_equals_base_7_3412_l1044_104421

def base_10_to_base_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec to_digits (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else to_digits (m / 7) ((m % 7) :: acc)
  to_digits n []

theorem base_10_1234_equals_base_7_3412 :
  base_10_to_base_7 1234 = [3, 4, 1, 2] := by sorry

end base_10_1234_equals_base_7_3412_l1044_104421


namespace cos_alpha_plus_20_eq_neg_alpha_l1044_104485

theorem cos_alpha_plus_20_eq_neg_alpha (α : ℝ) (h : Real.sin (α - 70 * Real.pi / 180) = α) :
  Real.cos (α + 20 * Real.pi / 180) = -α := by
  sorry

end cos_alpha_plus_20_eq_neg_alpha_l1044_104485


namespace media_group_arrangement_count_l1044_104438

/-- Represents the number of domestic media groups -/
def domestic_groups : ℕ := 6

/-- Represents the number of foreign media groups -/
def foreign_groups : ℕ := 3

/-- Represents the total number of media groups to be selected -/
def selected_groups : ℕ := 4

/-- Calculates the number of ways to select and arrange media groups -/
def media_group_arrangements (d : ℕ) (f : ℕ) (s : ℕ) : ℕ :=
  -- Implementation details are omitted as per the instructions
  sorry

/-- Theorem stating that the number of valid arrangements is 684 -/
theorem media_group_arrangement_count :
  media_group_arrangements domestic_groups foreign_groups selected_groups = 684 := by
  sorry

end media_group_arrangement_count_l1044_104438


namespace fifth_color_marbles_l1044_104461

/-- The number of marbles of each color in a box --/
structure MarbleCount where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  fifth : ℕ

/-- The properties of the marble counts --/
def valid_marble_count (m : MarbleCount) : Prop :=
  m.red = 25 ∧
  m.green = 3 * m.red ∧
  m.yellow = m.green / 5 ∧
  m.blue = 2 * m.yellow ∧
  m.red + m.green + m.yellow + m.blue + m.fifth = 4 * m.green

theorem fifth_color_marbles (m : MarbleCount) 
  (h : valid_marble_count m) : m.fifth = 155 := by
  sorry

end fifth_color_marbles_l1044_104461


namespace equation_equivalence_l1044_104442

theorem equation_equivalence (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 3*x^2 + x + 2 = 0 ↔ x^2 * (y^2 + y - 5) = 0 :=
by sorry

end equation_equivalence_l1044_104442


namespace no_common_terms_except_first_l1044_104474

def X : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => X (n + 1) + 2 * X n

def Y : ℕ → ℤ
  | 0 => 1
  | 1 => 7
  | (n + 2) => 2 * Y (n + 1) + 3 * Y n

theorem no_common_terms_except_first : ∀ n m : ℕ, X n = Y m → n = 0 ∧ m = 0 := by
  sorry

end no_common_terms_except_first_l1044_104474


namespace correct_stratified_sample_l1044_104412

/-- Represents the number of employees to be sampled from each category -/
structure StratifiedSample where
  business : ℕ
  management : ℕ
  logistics : ℕ

/-- Calculates the stratified sample given total employees and sample size -/
def calculateStratifiedSample (totalEmployees business management logistics sampleSize : ℕ) : StratifiedSample :=
  { business := (business * sampleSize) / totalEmployees,
    management := (management * sampleSize) / totalEmployees,
    logistics := (logistics * sampleSize) / totalEmployees }

theorem correct_stratified_sample :
  let totalEmployees : ℕ := 160
  let business : ℕ := 120
  let management : ℕ := 16
  let logistics : ℕ := 24
  let sampleSize : ℕ := 20
  let sample := calculateStratifiedSample totalEmployees business management logistics sampleSize
  sample.business = 15 ∧ sample.management = 2 ∧ sample.logistics = 3 := by
  sorry

end correct_stratified_sample_l1044_104412


namespace least_integer_satisfying_inequality_l1044_104480

theorem least_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * |x| + 4 < 19) → x ≥ -4 ∧
  ∃ y : ℤ, y = -4 ∧ (3 * |y| + 4 < 19) :=
by sorry

end least_integer_satisfying_inequality_l1044_104480


namespace correct_first_year_caps_l1044_104472

/-- The number of caps Lilith collects per month in the first year -/
def first_year_monthly_caps : ℕ := 3

/-- The number of years Lilith has been collecting caps -/
def total_years : ℕ := 5

/-- The number of caps Lilith collects per month after the first year -/
def later_years_monthly_caps : ℕ := 5

/-- The number of caps Lilith receives each Christmas -/
def christmas_caps : ℕ := 40

/-- The number of caps Lilith loses each year -/
def yearly_lost_caps : ℕ := 15

/-- The total number of caps Lilith has collected after 5 years -/
def total_caps : ℕ := 401

theorem correct_first_year_caps : 
  first_year_monthly_caps * 12 + 
  (total_years - 1) * 12 * later_years_monthly_caps + 
  total_years * christmas_caps - 
  total_years * yearly_lost_caps = total_caps := by
  sorry

end correct_first_year_caps_l1044_104472


namespace tangent_line_at_one_a_lower_bound_local_max_inequality_l1044_104403

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - 2 * a * x

def g (a : ℝ) (x : ℝ) : ℝ := f a x + (1/2) * x^2

def hasTangentLine (f : ℝ → ℝ) (x₀ y₀ m : ℝ) : Prop :=
  ∀ x, f x = m * (x - x₀) + y₀

def hasLocalMax (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x ≤ f x₀

theorem tangent_line_at_one (a : ℝ) (h : a = 2) :
  hasTangentLine (f a) 1 (-4) (-3) :=
sorry

theorem a_lower_bound (a : ℝ) (h : ∀ x > 0, f a x ≤ 2) :
  a ≥ 1 / (2 * Real.exp 3) :=
sorry

theorem local_max_inequality (a : ℝ) (x₀ : ℝ) (h : hasLocalMax (g a) x₀) :
  x₀ * f a x₀ + 1 + a * x₀^2 > 0 :=
sorry

end tangent_line_at_one_a_lower_bound_local_max_inequality_l1044_104403


namespace square_difference_ben_subtraction_l1044_104420

theorem square_difference (n : ℕ) : (n - 1)^2 = n^2 - (2*n - 1) := by sorry

theorem ben_subtraction : 49^2 = 50^2 - 99 := by sorry

end square_difference_ben_subtraction_l1044_104420


namespace sqrt_five_is_quadratic_radical_l1044_104482

/-- A number is non-negative if it's greater than or equal to zero. -/
def NonNegative (x : ℝ) : Prop := x ≥ 0

/-- A quadratic radical is an expression √x where x is non-negative. -/
def QuadraticRadical (x : ℝ) : Prop := NonNegative x

/-- Theorem: √5 is a quadratic radical. -/
theorem sqrt_five_is_quadratic_radical : QuadraticRadical 5 := by
  sorry

end sqrt_five_is_quadratic_radical_l1044_104482


namespace imaginary_unit_power_l1044_104483

theorem imaginary_unit_power (i : ℂ) : i ^ 2 = -1 → i ^ 2023 = -i := by
  sorry

end imaginary_unit_power_l1044_104483


namespace set_union_problem_l1044_104488

theorem set_union_problem (M N : Set ℕ) (x : ℕ) : 
  M = {0, x} → N = {1, 2} → M ∩ N = {2} → M ∪ N = {0, 1, 2} := by
  sorry

end set_union_problem_l1044_104488


namespace melanie_total_dimes_l1044_104471

def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4

theorem melanie_total_dimes : 
  initial_dimes + dimes_from_dad + dimes_from_mom = 19 := by
  sorry

end melanie_total_dimes_l1044_104471


namespace hypotenuse_plus_diameter_eq_sum_of_legs_l1044_104497

/-- Represents a right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  a : ℝ     -- Length of one leg
  b : ℝ     -- Length of the other leg
  c : ℝ     -- Length of the hypotenuse
  ρ : ℝ     -- Radius of the inscribed circle
  h_right : a^2 + b^2 = c^2  -- Pythagorean theorem
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ ρ > 0  -- Positive lengths

/-- 
The sum of the hypotenuse and the diameter of the inscribed circle 
is equal to the sum of the two legs in a right-angled triangle
-/
theorem hypotenuse_plus_diameter_eq_sum_of_legs 
  (t : RightTriangleWithInscribedCircle) : t.c + 2 * t.ρ = t.a + t.b := by
  sorry

end hypotenuse_plus_diameter_eq_sum_of_legs_l1044_104497


namespace sector_central_angle_l1044_104402

/-- Given a circular sector with radius 2 and area 8, 
    the radian measure of its central angle is 4. -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (angle : ℝ) : 
  r = 2 → area = 8 → area = (1/2) * r^2 * angle → angle = 4 := by sorry

end sector_central_angle_l1044_104402


namespace rectangular_plot_length_breadth_difference_l1044_104457

theorem rectangular_plot_length_breadth_difference 
  (area length breadth : ℝ)
  (h1 : area = length * breadth)
  (h2 : area = 15 * breadth)
  (h3 : breadth = 5) :
  length - breadth = 10 := by
  sorry

end rectangular_plot_length_breadth_difference_l1044_104457


namespace factorial_prime_factors_l1044_104499

theorem factorial_prime_factors (x i k m p : ℕ) : 
  x = (Finset.range 8).prod (λ n => n + 1) →
  x = 2^i * 3^k * 5^m * 7^p →
  i > 0 ∧ k > 0 ∧ m > 0 ∧ p > 0 →
  i + k + m + p = 11 := by
sorry

end factorial_prime_factors_l1044_104499


namespace initial_female_percent_calculation_l1044_104400

/-- Represents a company's workforce statistics -/
structure Workforce where
  initial_total : ℕ
  initial_female_percent : ℚ
  hired_male : ℕ
  final_total : ℕ
  final_female_percent : ℚ

/-- Theorem stating the conditions and the result to be proved -/
theorem initial_female_percent_calculation (w : Workforce) 
  (h1 : w.hired_male = 30)
  (h2 : w.final_total = 360)
  (h3 : w.final_female_percent = 55/100)
  (h4 : w.initial_total * w.initial_female_percent = w.final_total * w.final_female_percent) :
  w.initial_female_percent = 60/100 := by
  sorry

end initial_female_percent_calculation_l1044_104400


namespace sticker_distribution_theorem_l1044_104427

/-- Number of stickers -/
def n : ℕ := 10

/-- Number of sheets -/
def k : ℕ := 5

/-- Number of color options for each sheet -/
def c : ℕ := 2

/-- The number of ways to distribute n identical objects into k distinct boxes -/
def ways_to_distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) k

/-- The total number of distinct arrangements considering both sticker counts and colors -/
def total_arrangements (n k c : ℕ) : ℕ := (ways_to_distribute n k) * (c^k)

/-- The main theorem stating the total number of distinct arrangements -/
theorem sticker_distribution_theorem : total_arrangements n k c = 32032 := by
  sorry

end sticker_distribution_theorem_l1044_104427


namespace additional_space_needed_l1044_104424

theorem additional_space_needed (available_space backup_size software_size : ℕ) : 
  available_space = 28 → 
  backup_size = 26 → 
  software_size = 4 → 
  (backup_size + software_size) - available_space = 2 :=
by
  sorry

end additional_space_needed_l1044_104424


namespace root_sum_product_l1044_104410

theorem root_sum_product (p q r : ℝ) : 
  (5 * p^3 - 10 * p^2 + 17 * p - 7 = 0) ∧ 
  (5 * q^3 - 10 * q^2 + 17 * q - 7 = 0) ∧ 
  (5 * r^3 - 10 * r^2 + 17 * r - 7 = 0) → 
  p * q + p * r + q * r = 17 / 5 := by
sorry

end root_sum_product_l1044_104410


namespace fabric_order_calculation_l1044_104452

/-- The conversion factor from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- David's waist size in inches -/
def waist_size : ℝ := 38

/-- The extra allowance for waistband sewing in centimeters -/
def waistband_allowance : ℝ := 2

/-- The total length of fabric David should order in centimeters -/
def total_fabric_length : ℝ := waist_size * inch_to_cm + waistband_allowance

theorem fabric_order_calculation :
  total_fabric_length = 98.52 :=
by sorry

end fabric_order_calculation_l1044_104452


namespace factorization_of_2x_squared_minus_8_l1044_104406

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end factorization_of_2x_squared_minus_8_l1044_104406


namespace max_third_side_length_l1044_104415

theorem max_third_side_length (a b : ℝ) (ha : a = 6) (hb : b = 10) :
  ∃ (s : ℕ), s ≤ 15 ∧ 
  (∀ (t : ℕ), (t : ℝ) < a + b ∧ a < (t : ℝ) + b ∧ b < a + (t : ℝ) → t ≤ s) ∧
  ((15 : ℝ) < a + b ∧ a < 15 + b ∧ b < a + 15) :=
by sorry

end max_third_side_length_l1044_104415


namespace cloud_same_color_tangents_iff_l1044_104440

/-- A configuration of n points on a line with circumferences painted in k colors -/
structure Cloud (n k : ℕ) where
  n_ge_two : n ≥ 2
  colors : Fin k → Type
  circumferences : Fin n → Fin n → Option (Fin k)
  different_points : ∀ i j : Fin n, i ≠ j → circumferences i j ≠ none

/-- Two circumferences are mutually exterior tangent -/
def mutually_exterior_tangent (c : Cloud n k) (i j m l : Fin n) : Prop :=
  (i ≠ j ∧ m ≠ l) ∧ (i = m ∨ i = l ∨ j = m ∨ j = l)

/-- A cloud has two mutually exterior tangent circumferences of the same color -/
def has_same_color_tangents (c : Cloud n k) : Prop :=
  ∃ i j m l : Fin n, ∃ color : Fin k,
    mutually_exterior_tangent c i j m l ∧
    c.circumferences i j = some color ∧
    c.circumferences m l = some color

/-- Main theorem: characterization of n for which all (n,k)-clouds have same color tangents -/
theorem cloud_same_color_tangents_iff (k : ℕ) :
  (∀ n : ℕ, ∀ c : Cloud n k, has_same_color_tangents c) ↔ n ≥ 2^k + 1 :=
sorry

end cloud_same_color_tangents_iff_l1044_104440


namespace hyperbola_focal_length_l1044_104429

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is 2√(a² + b²) -/
theorem hyperbola_focal_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let focal_length := 2 * Real.sqrt (a^2 + b^2)
  focal_length = 2 * Real.sqrt 7 ↔ a^2 = 4 ∧ b^2 = 3 :=
by sorry

end hyperbola_focal_length_l1044_104429


namespace largest_value_l1044_104435

theorem largest_value : 
  let a := 3 + 1 + 2 + 8
  let b := 3 * 1 + 2 + 8
  let c := 3 + 1 * 2 + 8
  let d := 3 + 1 + 2 * 8
  let e := 3 * 1 * 2 * 8
  (e ≥ a) ∧ (e ≥ b) ∧ (e ≥ c) ∧ (e ≥ d) :=
by sorry

end largest_value_l1044_104435


namespace probability_female_wears_glasses_l1044_104458

/-- Given a class with female and male students, some wearing glasses, prove the probability of a randomly selected female student wearing glasses. -/
theorem probability_female_wears_glasses 
  (total_female : ℕ) 
  (total_male : ℕ) 
  (female_no_glasses : ℕ) 
  (male_with_glasses : ℕ) 
  (h1 : total_female = 18) 
  (h2 : total_male = 20) 
  (h3 : female_no_glasses = 8) 
  (h4 : male_with_glasses = 11) : 
  (total_female - female_no_glasses : ℚ) / total_female = 5 / 9 := by
  sorry

#check probability_female_wears_glasses

end probability_female_wears_glasses_l1044_104458


namespace distribute_balls_count_l1044_104433

/-- The number of ways to distribute 6 balls into 3 boxes -/
def distribute_balls : ℕ :=
  3 * (Nat.choose 4 2)

/-- Theorem stating that the number of ways to distribute the balls is 18 -/
theorem distribute_balls_count : distribute_balls = 18 := by
  sorry

end distribute_balls_count_l1044_104433


namespace number_divided_by_004_l1044_104484

theorem number_divided_by_004 :
  ∃ x : ℝ, x / 0.04 = 500.90000000000003 ∧ x = 20.036 := by
  sorry

end number_divided_by_004_l1044_104484


namespace perpendicular_vectors_k_value_l1044_104469

def a : ℝ × ℝ := (2, 1)

def b (k : ℝ) : ℝ × ℝ := (1 - 2, k - 1)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_vectors_k_value :
  ∀ k : ℝ, perpendicular a (b k) → k = 3 := by
  sorry

end perpendicular_vectors_k_value_l1044_104469


namespace hotel_room_charge_comparison_l1044_104486

theorem hotel_room_charge_comparison 
  (P R G : ℝ) 
  (h1 : P = R - 0.4 * R) 
  (h2 : P = G - 0.1 * G) : 
  (R - G) / G = 0.5 := by
sorry

end hotel_room_charge_comparison_l1044_104486


namespace diamond_equation_solution_l1044_104417

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 3 * x - y^2

-- State the theorem
theorem diamond_equation_solution :
  ∀ x : ℝ, diamond x 7 = 20 → x = 23 := by
  sorry

end diamond_equation_solution_l1044_104417


namespace book_pricing_deduction_percentage_l1044_104477

theorem book_pricing_deduction_percentage
  (cost_price : ℝ)
  (profit_percentage : ℝ)
  (list_price : ℝ)
  (h1 : cost_price = 47.50)
  (h2 : profit_percentage = 25)
  (h3 : list_price = 69.85) :
  let selling_price := cost_price * (1 + profit_percentage / 100)
  let deduction_percentage := (list_price - selling_price) / list_price * 100
  deduction_percentage = 15 := by
sorry

end book_pricing_deduction_percentage_l1044_104477


namespace checkerboard_covering_l1044_104405

/-- Represents an L-shaped piece that can cover three squares on a checkerboard -/
inductive LPiece
| mk : LPiece

/-- Represents a square on the checkerboard -/
structure Square where
  x : Nat
  y : Nat

/-- Represents a checkerboard with one square removed -/
structure Checkerboard (n : Nat) where
  sideLength : Nat
  removedSquare : Square
  validSideLength : sideLength = 2^n
  validRemovedSquare : removedSquare.x < sideLength ∧ removedSquare.y < sideLength

/-- Represents a covering of the checkerboard with L-shaped pieces -/
def Covering (n : Nat) := List (Square × Square × Square)

/-- Checks if a covering is valid for a given checkerboard -/
def isValidCovering (n : Nat) (board : Checkerboard n) (covering : Covering n) : Prop :=
  -- Each L-piece covers exactly three squares
  -- No gaps or overlaps in the covering
  -- The removed square is not covered
  sorry

/-- Main theorem: Any checkerboard with one square removed can be covered by L-shaped pieces -/
theorem checkerboard_covering (n : Nat) (h : n > 0) (board : Checkerboard n) :
  ∃ (covering : Covering n), isValidCovering n board covering :=
sorry

end checkerboard_covering_l1044_104405


namespace bus_driver_overtime_limit_l1044_104439

/-- Represents the problem of determining overtime limit for a bus driver --/
theorem bus_driver_overtime_limit 
  (regular_rate : ℝ) 
  (overtime_rate : ℝ) 
  (total_compensation : ℝ) 
  (total_hours : ℝ) 
  (h1 : regular_rate = 16)
  (h2 : overtime_rate = regular_rate * 1.75)
  (h3 : total_compensation = 864)
  (h4 : total_hours = 48) :
  ∃ (limit : ℝ), 
    limit = 40 ∧ 
    total_compensation = limit * regular_rate + (total_hours - limit) * overtime_rate :=
by sorry

end bus_driver_overtime_limit_l1044_104439


namespace tank_capacity_l1044_104423

theorem tank_capacity (initial_buckets : ℕ) (initial_capacity : ℚ) (new_buckets : ℕ) :
  initial_buckets = 26 →
  initial_capacity = 13.5 →
  new_buckets = 39 →
  (initial_buckets : ℚ) * initial_capacity / (new_buckets : ℚ) = 9 :=
by sorry

end tank_capacity_l1044_104423


namespace car_speed_l1044_104459

/-- Given a car that travels 275 miles in 5 hours, its speed is 55 miles per hour. -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 275) 
  (h2 : time = 5) 
  (h3 : speed = distance / time) : 
  speed = 55 := by
  sorry

end car_speed_l1044_104459


namespace cookie_distribution_ways_l1044_104493

/-- The number of ways to distribute cookies among students -/
def distribute_cookies (total_cookies : ℕ) (num_students : ℕ) (min_cookies : ℕ) : ℕ :=
  Nat.choose (total_cookies - num_students * min_cookies + num_students - 1) (num_students - 1)

/-- Theorem: The number of ways to distribute 30 cookies among 5 students, 
    with each student receiving at least 3 cookies, is 3876 -/
theorem cookie_distribution_ways : distribute_cookies 30 5 3 = 3876 := by
  sorry

end cookie_distribution_ways_l1044_104493


namespace line_slope_intercept_l1044_104487

/-- The line equation in vector form -/
def line_equation (x y : ℝ) : Prop :=
  (2 : ℝ) * (x - 1) + (-1 : ℝ) * (y - 5) = 0

/-- The slope-intercept form of a line -/
def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

theorem line_slope_intercept :
  ∃ (m b : ℝ), (∀ x y : ℝ, line_equation x y ↔ slope_intercept_form m b x y) ∧ m = 2 ∧ b = 3 := by
  sorry

end line_slope_intercept_l1044_104487


namespace m_range_l1044_104448

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0

def q (m : ℝ) : Prop := ∀ x : ℝ, m*x^2 - x + (1/16)*m > 0

-- Define the set of m satisfying the conditions
def S : Set ℝ := {m | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- Theorem statement
theorem m_range : S = {m | 1 < m ∧ m ≤ 2} := by sorry

end m_range_l1044_104448


namespace output_theorem_l1044_104481

/-- Represents the output of the program at each step -/
structure ProgramOutput :=
  (x : ℕ)
  (y : ℤ)

/-- The sequence of outputs from the program -/
def output_sequence : ℕ → ProgramOutput := sorry

/-- The theorem stating that when y = -10, x = 32 in the output sequence -/
theorem output_theorem :
  ∃ n : ℕ, (output_sequence n).y = -10 ∧ (output_sequence n).x = 32 := by
  sorry

end output_theorem_l1044_104481


namespace amusement_park_capacity_l1044_104463

/-- Represents the capacity of an amusement park ride -/
structure RideCapacity where
  people_per_unit : ℕ
  units : ℕ

/-- Calculates the total capacity of a ride -/
def total_capacity (ride : RideCapacity) : ℕ :=
  ride.people_per_unit * ride.units

/-- Theorem: The total capacity of three specific rides is 248 people -/
theorem amusement_park_capacity (whirling_wonderland sky_high_swings roaring_rapids : RideCapacity)
  (h1 : whirling_wonderland = ⟨12, 15⟩)
  (h2 : sky_high_swings = ⟨1, 20⟩)
  (h3 : roaring_rapids = ⟨6, 8⟩) :
  total_capacity whirling_wonderland + total_capacity sky_high_swings + total_capacity roaring_rapids = 248 := by
  sorry

end amusement_park_capacity_l1044_104463


namespace claire_earnings_l1044_104468

def total_flowers : ℕ := 400
def tulips : ℕ := 120
def white_roses : ℕ := 80
def price_per_red_rose : ℚ := 3/4

def total_roses : ℕ := total_flowers - tulips
def red_roses : ℕ := total_roses - white_roses
def red_roses_to_sell : ℕ := red_roses / 2

theorem claire_earnings :
  (red_roses_to_sell : ℚ) * price_per_red_rose = 75 := by
  sorry

end claire_earnings_l1044_104468


namespace laptop_sticker_price_l1044_104451

/-- The sticker price of a laptop satisfying certain discount conditions -/
theorem laptop_sticker_price : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (0.7 * x - (0.8 * x - 120) = 30) ∧ 
  (x = 900) := by
  sorry

end laptop_sticker_price_l1044_104451


namespace vector_OA_coordinates_l1044_104473

/-- Given that O is the origin, point A is in the second quadrant,
    |OA| = 2, and ∠xOA = 150°, prove that the coordinates of vector OA are (-√3, 1). -/
theorem vector_OA_coordinates (A : ℝ × ℝ) :
  A.1 < 0 ∧ A.2 > 0 →  -- A is in the second quadrant
  A.1^2 + A.2^2 = 4 →  -- |OA| = 2
  Real.cos (150 * π / 180) = A.1 / 2 ∧ Real.sin (150 * π / 180) = A.2 / 2 →  -- ∠xOA = 150°
  A = (-Real.sqrt 3, 1) :=
by sorry

end vector_OA_coordinates_l1044_104473


namespace average_exercise_days_l1044_104490

def exercise_data : List (Nat × Nat) := [
  (1, 1), (2, 3), (3, 2), (4, 6), (5, 8), (6, 3), (7, 2)
]

def total_exercise_days : Nat :=
  (exercise_data.map (fun (days, freq) => days * freq)).sum

def total_students : Nat :=
  (exercise_data.map (fun (_, freq) => freq)).sum

theorem average_exercise_days :
  (total_exercise_days : ℚ) / (total_students : ℚ) = 436 / 100 := by sorry

end average_exercise_days_l1044_104490


namespace camp_kids_count_l1044_104495

theorem camp_kids_count (total : ℕ) 
  (h1 : total / 2 = total / 2) -- Half of the kids are going to soccer camp
  (h2 : (total / 2) / 4 = (total / 2) / 4) -- 1/4 of soccer camp kids go in the morning
  (h3 : ((total / 2) * 3) / 4 = 750) -- 750 kids go to soccer camp in the afternoon
  : total = 2000 := by
  sorry

end camp_kids_count_l1044_104495


namespace ellipse_properties_l1044_104492

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/9 + y^2 = 1

-- Define the line that intersects the ellipse
def intersecting_line (x y : ℝ) : Prop := y = x + 2

-- Define the condition for a point to be on the circle with AB as diameter
def on_circle_AB (x y : ℝ) : Prop := 
  ∃ (x1 y1 x2 y2 : ℝ), 
    ellipse_C x1 y1 ∧ 
    ellipse_C x2 y2 ∧ 
    intersecting_line x1 y1 ∧ 
    intersecting_line x2 y2 ∧ 
    x * (x1 + x2) + y * (y1 + y2) = (x1^2 + y1^2 + x2^2 + y2^2) / 2

theorem ellipse_properties :
  (∀ x y, ellipse_C x y ↔ x^2/9 + y^2 = 1) ∧ 
  ¬(on_circle_AB 0 0) :=
sorry

end ellipse_properties_l1044_104492


namespace sara_movie_spending_l1044_104464

def movie_spending (theater_ticket_price : ℚ) (num_tickets : ℕ) (rental_price : ℚ) (purchase_price : ℚ) : ℚ :=
  theater_ticket_price * num_tickets + rental_price + purchase_price

theorem sara_movie_spending :
  let theater_ticket_price : ℚ := 10.62
  let num_tickets : ℕ := 2
  let rental_price : ℚ := 1.59
  let purchase_price : ℚ := 13.95
  movie_spending theater_ticket_price num_tickets rental_price purchase_price = 36.78 := by
sorry

end sara_movie_spending_l1044_104464


namespace inequality_and_not_all_greater_l1044_104491

theorem inequality_and_not_all_greater (m a b x y z : ℝ) : 
  m > 0 → 
  0 < x → x < 2 → 
  0 < y → y < 2 → 
  0 < z → z < 2 → 
  ((a + m * b) / (1 + m))^2 ≤ (a^2 + m * b^2) / (1 + m) ∧ 
  ¬(x * (2 - y) > 1 ∧ y * (2 - z) > 1 ∧ z * (2 - x) > 1) :=
by sorry

end inequality_and_not_all_greater_l1044_104491


namespace always_positive_product_l1044_104470

theorem always_positive_product (a b c : ℝ) (h : a > b ∧ b > c) : (a - b) * |c - b| > 0 := by
  sorry

end always_positive_product_l1044_104470


namespace division_remainder_division_remainder_is_200000_l1044_104416

theorem division_remainder : ℤ → Prop :=
  fun r => ((8 * 10^9) / (4 * 10^4)) % (10^6) = r

theorem division_remainder_is_200000 : division_remainder 200000 := by
  sorry

end division_remainder_division_remainder_is_200000_l1044_104416


namespace initial_potatoes_count_l1044_104413

/-- The number of potatoes Dan initially had in the garden --/
def initial_potatoes : ℕ := sorry

/-- The number of potatoes eaten by rabbits --/
def eaten_potatoes : ℕ := 4

/-- The number of potatoes Dan has now --/
def remaining_potatoes : ℕ := 3

/-- Theorem stating the initial number of potatoes --/
theorem initial_potatoes_count : initial_potatoes = 7 := by
  sorry

end initial_potatoes_count_l1044_104413


namespace f_increasing_range_l1044_104446

/-- The function f(x) = 2x^2 + mx - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 + m * x - 1

/-- The theorem stating the range of m for which f is increasing on (1, +∞) -/
theorem f_increasing_range (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ > 1 ∧ x₁ ≠ x₂ → (f m x₁ - f m x₂) / (x₁ - x₂) > 0) →
  m ≥ -4 :=
sorry

end f_increasing_range_l1044_104446


namespace existence_of_special_number_l1044_104404

theorem existence_of_special_number : ∃ A : ℕ, 
  (100000 ≤ A ∧ A < 1000000) ∧ 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 500000 → 
    ¬(∃ d : ℕ, d < 10 ∧ (k * A) % 1000000 = d * 111111) :=
by sorry

end existence_of_special_number_l1044_104404


namespace units_digit_of_expression_l1044_104498

theorem units_digit_of_expression : 2^2023 * 5^2024 * 11^2025 % 10 = 0 := by sorry

end units_digit_of_expression_l1044_104498


namespace fraction_simplification_l1044_104450

theorem fraction_simplification (x y z : ℝ) 
  (h1 : x > z) (h2 : z > y) (h3 : y > 0) :
  (x^z * z^y * y^x) / (z^z * y^y * x^x) = x^(z-x) * z^(y-z) * y^(x-y) := by
  sorry

end fraction_simplification_l1044_104450


namespace find_a_and_b_l1044_104462

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2008*x - 2009 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem find_a_and_b :
  ∃ (a b : ℝ), 
    (M ∪ N a b = Set.univ) ∧ 
    (M ∩ N a b = Set.Ioc 2009 2010) ∧ 
    a = 2009 ∧ 
    b = -2009 * 2010 := by
  sorry

end find_a_and_b_l1044_104462


namespace triangles_from_parallel_lines_l1044_104467

/-- The number of points on line a -/
def points_on_a : ℕ := 5

/-- The number of points on line b -/
def points_on_b : ℕ := 6

/-- The total number of triangles that can be formed -/
def total_triangles : ℕ := 135

/-- Theorem stating that the total number of triangles formed by points on two parallel lines is correct -/
theorem triangles_from_parallel_lines : 
  (points_on_a.choose 1 * points_on_b.choose 2) + (points_on_a.choose 2 * points_on_b.choose 1) = total_triangles :=
by sorry

end triangles_from_parallel_lines_l1044_104467


namespace enthalpy_combustion_10_moles_glucose_l1044_104445

/-- The standard enthalpy of combustion for glucose (C6H12O6) in kJ/mol -/
def standard_enthalpy_combustion_glucose : ℝ := -2800

/-- The number of moles of glucose -/
def moles_glucose : ℝ := 10

/-- The enthalpy of combustion for a given number of moles of glucose -/
def enthalpy_combustion (moles : ℝ) : ℝ :=
  standard_enthalpy_combustion_glucose * moles

/-- Theorem: The enthalpy of combustion for 10 moles of C6H12O6 is -28000 kJ -/
theorem enthalpy_combustion_10_moles_glucose :
  enthalpy_combustion moles_glucose = -28000 := by
  sorry

end enthalpy_combustion_10_moles_glucose_l1044_104445


namespace exactly_one_solution_l1044_104454

-- Define the function g₀
def g₀ (x : ℝ) : ℝ := x + |x - 50| - |x + 150|

-- Define the function gₙ recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 1

-- Theorem statement
theorem exactly_one_solution :
  ∃! x, g 100 x = 0 :=
sorry

end exactly_one_solution_l1044_104454


namespace set_c_not_right_triangle_set_a_right_triangle_set_b_right_triangle_set_d_right_triangle_l1044_104430

/-- A function to check if three numbers can form a right triangle -/
def can_form_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that (7, 8, 9) cannot form a right triangle -/
theorem set_c_not_right_triangle : ¬(can_form_right_triangle 7 8 9) := by sorry

/-- Theorem stating that (1, 1, √2) can form a right triangle -/
theorem set_a_right_triangle : can_form_right_triangle 1 1 (Real.sqrt 2) := by sorry

/-- Theorem stating that (5, 12, 13) can form a right triangle -/
theorem set_b_right_triangle : can_form_right_triangle 5 12 13 := by sorry

/-- Theorem stating that (1.5, 2, 2.5) can form a right triangle -/
theorem set_d_right_triangle : can_form_right_triangle 1.5 2 2.5 := by sorry

end set_c_not_right_triangle_set_a_right_triangle_set_b_right_triangle_set_d_right_triangle_l1044_104430


namespace min_days_correct_l1044_104478

/-- Represents the problem of scheduling warriors for duty --/
structure WarriorSchedule where
  total_warriors : ℕ
  min_duty : ℕ
  max_duty : ℕ
  min_days : ℕ

/-- The specific instance of the problem --/
def warrior_problem : WarriorSchedule :=
  { total_warriors := 33
  , min_duty := 9
  , max_duty := 10
  , min_days := 7 }

/-- Theorem stating that the minimum number of days is correct --/
theorem min_days_correct (w : WarriorSchedule) (h1 : w = warrior_problem) :
  ∃ (k l m : ℕ),
    k + l = w.min_days ∧
    w.min_duty * k + w.max_duty * l = w.total_warriors * m ∧
    (∀ (k' l' : ℕ), k' + l' < w.min_days →
      ¬∃ (m' : ℕ), w.min_duty * k' + w.max_duty * l' = w.total_warriors * m') :=
by sorry

end min_days_correct_l1044_104478


namespace vector_operation_result_l1044_104414

theorem vector_operation_result :
  let v1 : Fin 2 → ℝ := ![5, -3]
  let v2 : Fin 2 → ℝ := ![0, 4]
  let v3 : Fin 2 → ℝ := ![-2, 1]
  let result : Fin 2 → ℝ := ![3, -14]
  v1 - 3 • v2 + v3 = result :=
by
  sorry

end vector_operation_result_l1044_104414


namespace quadratic_root_ratio_l1044_104401

/-- Given a quadratic equation ax^2 + bx + c = 0 where a ≠ 0 and c ≠ 0,
    if one root is 4 times the other root, then b^2 / (ac) = 25/4 -/
theorem quadratic_root_ratio (a b c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  (∃ x y : ℝ, x = 4 * y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  b^2 / (a * c) = 25 / 4 := by
  sorry

end quadratic_root_ratio_l1044_104401


namespace square_ratio_side_length_l1044_104426

theorem square_ratio_side_length (area_ratio : ℚ) : 
  area_ratio = 250 / 98 →
  ∃ (a b c : ℕ), 
    (a^2 * b : ℚ) / c^2 = area_ratio ∧
    a = 5 ∧ b = 5 ∧ c = 7 ∧
    a + b + c = 17 := by
  sorry

end square_ratio_side_length_l1044_104426


namespace geometric_locus_definition_l1044_104419

-- Define a type for points in a space
variable {Point : Type*}

-- Define a predicate for the condition that points must satisfy
variable (condition : Point → Prop)

-- Define a predicate for points being on the locus
variable (on_locus : Point → Prop)

-- Statement A
def statement_A : Prop :=
  (∀ p, on_locus p → condition p) ∧ (∀ p, condition p → on_locus p)

-- Statement B
def statement_B : Prop :=
  (∀ p, ¬condition p → ¬on_locus p) ∧ ¬(∀ p, condition p → on_locus p)

-- Statement C
def statement_C : Prop :=
  ∀ p, on_locus p ↔ condition p

-- Statement D
def statement_D : Prop :=
  (∀ p, ¬on_locus p → ¬condition p) ∧ (∀ p, condition p → on_locus p)

-- Statement E
def statement_E : Prop :=
  (∀ p, on_locus p → condition p) ∧ ¬(∀ p, condition p → on_locus p)

theorem geometric_locus_definition :
  (statement_A condition on_locus ∧ 
   statement_C condition on_locus ∧ 
   statement_D condition on_locus) ∧
  (¬statement_B condition on_locus ∧ 
   ¬statement_E condition on_locus) :=
sorry

end geometric_locus_definition_l1044_104419
