import Mathlib

namespace NUMINAMATH_CALUDE_total_dolls_l4171_417174

/-- The number of dolls each person has -/
structure DollCounts where
  vera : ℕ
  sophie : ℕ
  aida : ℕ

/-- The conditions of the doll distribution -/
def doll_distribution (d : DollCounts) : Prop :=
  d.vera = 20 ∧ d.sophie = 2 * d.vera ∧ d.aida = 2 * d.sophie

/-- The theorem stating the total number of dolls -/
theorem total_dolls (d : DollCounts) (h : doll_distribution d) : 
  d.vera + d.sophie + d.aida = 140 := by
  sorry

#check total_dolls

end NUMINAMATH_CALUDE_total_dolls_l4171_417174


namespace NUMINAMATH_CALUDE_jerry_shelf_theorem_l4171_417136

/-- Calculates the total number of action figures on Jerry's shelf -/
def total_action_figures (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that the total number of action figures is the sum of initial and added figures -/
theorem jerry_shelf_theorem (initial : ℕ) (added : ℕ) :
  total_action_figures initial added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_jerry_shelf_theorem_l4171_417136


namespace NUMINAMATH_CALUDE_triangle_construction_solutions_l4171_417162

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Checks if a point is the foot of an altitude in a triangle -/
def isAltitudeFoot (P : Point2D) (T : Triangle) : Prop := sorry

/-- Checks if a point is the midpoint of a side in a triangle -/
def isMidpoint (P : Point2D) (A B : Point2D) : Prop := sorry

/-- Checks if a point is the midpoint of an altitude in a triangle -/
def isAltitudeMidpoint (P : Point2D) (T : Triangle) : Prop := sorry

/-- The main theorem statement -/
theorem triangle_construction_solutions 
  (A₀ B₁ C₂ : Point2D) : 
  ∃ (T₁ T₂ : Triangle), 
    T₁ ≠ T₂ ∧ 
    isAltitudeFoot A₀ T₁ ∧
    isAltitudeFoot A₀ T₂ ∧
    isMidpoint B₁ T₁.A T₁.C ∧
    isMidpoint B₁ T₂.A T₂.C ∧
    isAltitudeMidpoint C₂ T₁ ∧
    isAltitudeMidpoint C₂ T₂ :=
  sorry

end NUMINAMATH_CALUDE_triangle_construction_solutions_l4171_417162


namespace NUMINAMATH_CALUDE_savings_difference_l4171_417132

def original_value : ℝ := 20000

def discount_scheme_1 (x : ℝ) : ℝ :=
  x * (1 - 0.3) * (1 - 0.1) - 800

def discount_scheme_2 (x : ℝ) : ℝ :=
  x * (1 - 0.25) * (1 - 0.2) - 1000

theorem savings_difference :
  discount_scheme_1 original_value - discount_scheme_2 original_value = 800 := by
  sorry

end NUMINAMATH_CALUDE_savings_difference_l4171_417132


namespace NUMINAMATH_CALUDE_base4_odd_digits_317_l4171_417198

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

theorem base4_odd_digits_317 :
  countOddDigits (toBase4 317) = 4 := by
  sorry

end NUMINAMATH_CALUDE_base4_odd_digits_317_l4171_417198


namespace NUMINAMATH_CALUDE_train_speed_l4171_417196

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 350) (h2 : time = 7) :
  length / time = 50 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l4171_417196


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_tangents_l4171_417171

/-- The circle equation: x^2 - 2x + y^2 - 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 2*y + 1 = 0

/-- The external point P -/
def P : ℝ × ℝ := (3, 2)

/-- The cosine of the angle between two tangent lines -/
noncomputable def cos_angle_between_tangents : ℝ := 3/5

theorem cosine_of_angle_between_tangents :
  let (px, py) := P
  ∀ x y : ℝ, circle_equation x y →
  cos_angle_between_tangents = 3/5 := by sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_tangents_l4171_417171


namespace NUMINAMATH_CALUDE_bridge_toll_base_cost_l4171_417142

/-- Represents the toll calculation for a bridge -/
structure BridgeToll where
  base_cost : ℝ
  axle_cost : ℝ

/-- Calculates the toll for a given number of axles -/
def calc_toll (bt : BridgeToll) (axles : ℕ) : ℝ :=
  bt.base_cost + bt.axle_cost * (axles - 2)

/-- Represents a truck with a specific number of wheels and axles -/
structure Truck where
  total_wheels : ℕ
  front_axle_wheels : ℕ
  other_axle_wheels : ℕ

/-- Calculates the number of axles for a truck -/
def calc_axles (t : Truck) : ℕ :=
  1 + (t.total_wheels - t.front_axle_wheels) / t.other_axle_wheels

theorem bridge_toll_base_cost :
  ∃ (bt : BridgeToll),
    bt.axle_cost = 0.5 ∧
    let truck := Truck.mk 18 2 4
    let axles := calc_axles truck
    calc_toll bt axles = 5 ∧
    bt.base_cost = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_bridge_toll_base_cost_l4171_417142


namespace NUMINAMATH_CALUDE_train_crossing_time_l4171_417121

/-- Time taken for a train to cross a man walking in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 100 →
  train_speed = 54.99520038396929 →
  man_speed = 5 →
  (train_length / ((train_speed + man_speed) * 1000 / 3600)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l4171_417121


namespace NUMINAMATH_CALUDE_linear_function_k_value_l4171_417163

theorem linear_function_k_value (k : ℝ) : 
  k ≠ 0 → (1 : ℝ) = k * 3 - 2 → k = 1 := by sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l4171_417163


namespace NUMINAMATH_CALUDE_custom_op_equality_l4171_417179

/-- Custom operation ⊗ -/
def custom_op (a b : ℝ) : ℝ := a * b + a - b

/-- Theorem stating the equality of the expression and its simplified form -/
theorem custom_op_equality (a b : ℝ) : 
  custom_op a b + custom_op (b - a) b = b^2 - b := by sorry

end NUMINAMATH_CALUDE_custom_op_equality_l4171_417179


namespace NUMINAMATH_CALUDE_equation_solution_l4171_417110

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (x - 1) / (x - 2) + 1 / (2 - x) = 3 ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4171_417110


namespace NUMINAMATH_CALUDE_not_odd_function_iff_exists_neq_l4171_417146

theorem not_odd_function_iff_exists_neq (f : ℝ → ℝ) :
  (¬ ∀ x, f (-x) = -f x) ↔ ∃ x₀, f (-x₀) ≠ -f x₀ :=
sorry

end NUMINAMATH_CALUDE_not_odd_function_iff_exists_neq_l4171_417146


namespace NUMINAMATH_CALUDE_time_difference_per_question_l4171_417104

def english_questions : ℕ := 30
def math_questions : ℕ := 15
def english_time_hours : ℚ := 1
def math_time_hours : ℚ := (3/2)

def english_time_minutes : ℚ := english_time_hours * 60
def math_time_minutes : ℚ := math_time_hours * 60

def english_time_per_question : ℚ := english_time_minutes / english_questions
def math_time_per_question : ℚ := math_time_minutes / math_questions

theorem time_difference_per_question :
  math_time_per_question - english_time_per_question = 4 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_per_question_l4171_417104


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l4171_417192

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l4171_417192


namespace NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l4171_417185

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 5 = 0
def line2 (x y : ℝ) : Prop := x + y + 2 = 0
def line3 (x y : ℝ) : Prop := 3 * x + y - 1 = 0
def result_line (x y : ℝ) : Prop := 3 * x + y = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define parallel lines
def parallel_lines (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

theorem line_through_intersection_and_parallel :
  (∃ (x y : ℝ), intersection_point x y ∧ result_line x y) ∧
  parallel_lines line3 result_line :=
sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_parallel_l4171_417185


namespace NUMINAMATH_CALUDE_weekend_classes_count_l4171_417155

/-- The number of beginning diving classes offered on each day of the weekend -/
def weekend_classes : ℕ := 4

/-- The number of beginning diving classes offered on weekdays -/
def weekday_classes : ℕ := 2

/-- The number of people that can be accommodated in each class -/
def class_capacity : ℕ := 5

/-- The total number of people that can take classes in 3 weeks -/
def total_people : ℕ := 270

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

/-- The number of weeks considered -/
def weeks : ℕ := 3

theorem weekend_classes_count :
  weekend_classes * class_capacity * weekend_days_per_week * weeks +
  weekday_classes * class_capacity * weekdays_per_week * weeks = total_people :=
by sorry

end NUMINAMATH_CALUDE_weekend_classes_count_l4171_417155


namespace NUMINAMATH_CALUDE_equal_angle_locus_for_given_flagpoles_l4171_417151

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a flagpole -/
structure Flagpole where
  base : Point
  height : ℝ

/-- The locus of points with equal angles of elevation to two flagpoles -/
def equalAngleLocus (pole1 pole2 : Flagpole) : Set (Point) :=
  {p : Point | (p.x - 85/8)^2 + p.y^2 = (75/8)^2}

theorem equal_angle_locus_for_given_flagpoles :
  let pole1 : Flagpole := ⟨Point.mk (-5) 0, 5⟩
  let pole2 : Flagpole := ⟨Point.mk 5 0, 3⟩
  equalAngleLocus pole1 pole2 =
    {p : Point | (p.x - 85/8)^2 + p.y^2 = (75/8)^2} :=
by
  sorry

end NUMINAMATH_CALUDE_equal_angle_locus_for_given_flagpoles_l4171_417151


namespace NUMINAMATH_CALUDE_square_and_rectangle_area_sum_l4171_417143

/-- Given a square and a rectangle satisfying certain conditions, prove that the sum of their areas is approximately 118 square units. -/
theorem square_and_rectangle_area_sum :
  ∀ (s w : ℝ),
    s > 0 →
    w > 0 →
    s^2 + 2*w^2 = 130 →
    4*s - 2*(w + 2*w) = 20 →
    abs (s^2 + 2*w^2 - 118) < 1 :=
by
  sorry

#check square_and_rectangle_area_sum

end NUMINAMATH_CALUDE_square_and_rectangle_area_sum_l4171_417143


namespace NUMINAMATH_CALUDE_probability_prime_8_sided_die_l4171_417116

-- Define a fair 8-sided die
def fair_8_sided_die : Finset ℕ := Finset.range 8

-- Define the set of prime numbers from 1 to 8
def primes_1_to_8 : Finset ℕ := {2, 3, 5, 7}

-- Theorem: The probability of rolling a prime number on a fair 8-sided die is 1/2
theorem probability_prime_8_sided_die :
  (Finset.card primes_1_to_8 : ℚ) / (Finset.card fair_8_sided_die : ℚ) = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_probability_prime_8_sided_die_l4171_417116


namespace NUMINAMATH_CALUDE_ellipse_condition_l4171_417190

/-- Represents the equation (x^2)/(6-k) + (y^2)/(k-4) = 1 --/
def is_ellipse (k : ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ a > 0 ∧ b > 0 ∧
  ∀ x y : ℝ, x^2 / (6-k) + y^2 / (k-4) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_condition (k : ℝ) :
  (is_ellipse k → 4 < k ∧ k < 6) ∧
  (∃ k : ℝ, 4 < k ∧ k < 6 ∧ ¬is_ellipse k) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l4171_417190


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l4171_417166

theorem x_gt_one_sufficient_not_necessary_for_x_gt_zero :
  (∀ x : ℝ, x > 1 → x > 0) ∧ (∃ x : ℝ, x > 0 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l4171_417166


namespace NUMINAMATH_CALUDE_games_within_division_is_16_l4171_417100

/-- Represents a baseball league with two divisions -/
structure BaseballLeague where
  /-- Number of games played against each team in the same division -/
  n : ℕ
  /-- Number of games played against each team in the other division -/
  m : ℕ
  /-- n is greater than 3m -/
  n_gt_3m : n > 3 * m
  /-- m is greater than 6 -/
  m_gt_6 : m > 6
  /-- Total number of games each team plays is 96 -/
  total_games : 4 * n + 5 * m = 96

/-- The number of games a team plays within its own division -/
def games_within_division (league : BaseballLeague) : ℕ := 4 * league.n

/-- Theorem stating that the number of games played within a team's division is 16 -/
theorem games_within_division_is_16 (league : BaseballLeague) :
  games_within_division league = 16 := by
  sorry

end NUMINAMATH_CALUDE_games_within_division_is_16_l4171_417100


namespace NUMINAMATH_CALUDE_sqrt_product_l4171_417129

theorem sqrt_product (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_l4171_417129


namespace NUMINAMATH_CALUDE_hyperbola_properties_l4171_417148

/-- Definition of the hyperbola C -/
def C (x y : ℝ) : Prop := y = Real.sqrt 3 * (1 / (2 * x) + x / 3)

/-- C is a hyperbola -/
axiom C_is_hyperbola : ∃ (a b : ℝ), ∀ (x y : ℝ), C x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1

/-- Statement about the asymptote, focus, and intersection properties of C -/
theorem hyperbola_properties :
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → ∃ y, C x y ∧ |y| > 1/ε) ∧ 
  C 1 (Real.sqrt 3) ∧
  (∀ t : ℝ, ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ C x₁ y₁ ∧ C x₂ y₂ ∧ y₁ = x₁ + t ∧ y₂ = x₂ + t) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l4171_417148


namespace NUMINAMATH_CALUDE_initial_population_initial_population_approx_l4171_417126

/-- Calculates the initial population of a village given the population changes over 5 years and the final population. -/
theorem initial_population (final_population : ℝ) : ℝ :=
  let year1_change := 1.05
  let year2_change := 0.93
  let year3_change := 1.03
  let year4_change := 1.10
  let year5_change := 0.95
  final_population / (year1_change * year2_change * year3_change * year4_change * year5_change)

/-- The initial population of the village is approximately 10,457. -/
theorem initial_population_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |initial_population 10450 - 10457| < ε :=
sorry

end NUMINAMATH_CALUDE_initial_population_initial_population_approx_l4171_417126


namespace NUMINAMATH_CALUDE_third_butcher_delivery_l4171_417107

theorem third_butcher_delivery (package_weight : ℕ) (first_butcher : ℕ) (second_butcher : ℕ) (total_weight : ℕ) : 
  package_weight = 4 →
  first_butcher = 10 →
  second_butcher = 7 →
  total_weight = 100 →
  ∃ third_butcher : ℕ, 
    third_butcher * package_weight + first_butcher * package_weight + second_butcher * package_weight = total_weight ∧
    third_butcher = 8 :=
by sorry

end NUMINAMATH_CALUDE_third_butcher_delivery_l4171_417107


namespace NUMINAMATH_CALUDE_birth_rate_calculation_l4171_417177

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the number of population changes per day -/
def changes_per_day : ℕ := seconds_per_day / 2

/-- Represents the death rate in people per two seconds -/
def death_rate : ℕ := 2

/-- Represents the daily net population increase -/
def daily_net_increase : ℕ := 345600

/-- Represents the average birth rate in people per two seconds -/
def birth_rate : ℕ := 10

theorem birth_rate_calculation :
  (birth_rate - death_rate) * changes_per_day = daily_net_increase :=
by sorry

end NUMINAMATH_CALUDE_birth_rate_calculation_l4171_417177


namespace NUMINAMATH_CALUDE_square_sum_l4171_417139

theorem square_sum (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = -1) : x^2 + y^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_l4171_417139


namespace NUMINAMATH_CALUDE_odot_inequality_equivalence_l4171_417131

-- Define the operation ⊙
def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

-- State the theorem
theorem odot_inequality_equivalence :
  ∀ x : ℝ, odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_odot_inequality_equivalence_l4171_417131


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l4171_417105

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 + (1 + x)^6 + (1 + x)^7 + (1 + x)^8
           = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 502 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l4171_417105


namespace NUMINAMATH_CALUDE_initial_figures_correct_figure_50_l4171_417127

/-- The number of unit squares in the nth figure -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The first four figures match the given pattern -/
theorem initial_figures_correct :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 := by sorry

/-- The 50th figure contains 7651 unit squares -/
theorem figure_50 : f 50 = 7651 := by sorry

end NUMINAMATH_CALUDE_initial_figures_correct_figure_50_l4171_417127


namespace NUMINAMATH_CALUDE_lewis_items_count_l4171_417178

theorem lewis_items_count (tanya samantha lewis james : ℕ) : 
  tanya = 4 →
  samantha = 4 * tanya →
  lewis = samantha - (samantha / 3) →
  james = 2 * lewis →
  lewis = 11 := by
sorry

end NUMINAMATH_CALUDE_lewis_items_count_l4171_417178


namespace NUMINAMATH_CALUDE_system_solution_l4171_417154

/-- Given a system of equations:
    1) x = 1.12 * y + 52.8
    2) x = y + 50
    Prove that the solution is approximately x ≈ 26.67 and y ≈ -23.33 -/
theorem system_solution :
  ∃ (x y : ℝ),
    (x = 1.12 * y + 52.8) ∧
    (x = y + 50) ∧
    (abs (x - 26.67) < 0.01) ∧
    (abs (y + 23.33) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4171_417154


namespace NUMINAMATH_CALUDE_probability_of_specific_match_l4171_417130

/-- The number of teams in the tournament -/
def num_teams : ℕ := 128

/-- The probability of two specific teams playing each other in a single elimination tournament -/
def probability_of_match (n : ℕ) : ℚ :=
  (n - 1) / (n * (n - 1) / 2)

/-- Theorem: In a single elimination tournament with 128 equally strong teams,
    the probability of two specific teams playing each other is 1/64 -/
theorem probability_of_specific_match :
  probability_of_match num_teams = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_of_specific_match_l4171_417130


namespace NUMINAMATH_CALUDE_batsman_average_increase_l4171_417150

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  total_runs : Nat
  average : Rat

/-- Calculates the increase in average after a new inning -/
def average_increase (b : Batsman) (new_runs : Nat) : Rat :=
  let new_total := b.total_runs + new_runs
  let new_average : Rat := new_total / (b.innings + 1)
  new_average - b.average

/-- Theorem: The batsman's average increases by 3 -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 16 →
    (b.total_runs + 86) / 17 = 38 →
    average_increase b 86 = 3 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l4171_417150


namespace NUMINAMATH_CALUDE_deceased_member_income_family_income_problem_l4171_417183

theorem deceased_member_income 
  (initial_members : ℕ) 
  (initial_avg_income : ℚ) 
  (final_members : ℕ) 
  (final_avg_income : ℚ) : ℚ :=
  let initial_total_income := initial_members * initial_avg_income
  let final_total_income := final_members * final_avg_income
  initial_total_income - final_total_income

theorem family_income_problem 
  (h1 : initial_members = 4)
  (h2 : initial_avg_income = 840)
  (h3 : final_members = 3)
  (h4 : final_avg_income = 650) : 
  deceased_member_income initial_members initial_avg_income final_members final_avg_income = 1410 := by
  sorry

end NUMINAMATH_CALUDE_deceased_member_income_family_income_problem_l4171_417183


namespace NUMINAMATH_CALUDE_rectangle_length_eq_five_l4171_417168

/-- The length of a rectangle with width 20 cm and perimeter equal to that of a regular pentagon with side length 10 cm is 5 cm. -/
theorem rectangle_length_eq_five (width : ℝ) (pentagon_side : ℝ) (length : ℝ) : 
  width = 20 →
  pentagon_side = 10 →
  2 * (length + width) = 5 * pentagon_side →
  length = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_eq_five_l4171_417168


namespace NUMINAMATH_CALUDE_solve_equation_l4171_417167

theorem solve_equation : ∃ x : ℕ, x * 12 = 173 * 240 ∧ x = 3460 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4171_417167


namespace NUMINAMATH_CALUDE_work_completion_theorem_l4171_417194

/-- The number of days it takes the first group to complete the work -/
def days_first_group : ℕ := 25

/-- The number of men in the second group -/
def men_second_group : ℕ := 20

/-- The number of days it takes the second group to complete the work -/
def days_second_group : ℕ := 20

/-- The number of men in the first group -/
def men_first_group : ℕ := men_second_group * days_second_group / days_first_group

theorem work_completion_theorem :
  men_first_group * days_first_group = men_second_group * days_second_group ∧
  men_first_group = 16 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l4171_417194


namespace NUMINAMATH_CALUDE_substitution_result_l4171_417109

theorem substitution_result (x y : ℝ) :
  y = 2 * x + 1 ∧ 5 * x - 2 * y = 7 →
  5 * x - 4 * x - 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_substitution_result_l4171_417109


namespace NUMINAMATH_CALUDE_shaded_probability_is_one_third_l4171_417159

/-- Represents a triangle in the diagram -/
structure Triangle where
  shaded : Bool

/-- Represents the diagram with triangles -/
structure Diagram where
  triangles : List Triangle
  shaded_count : Nat
  h_more_than_five : triangles.length > 5
  h_shaded_count : shaded_count = (triangles.filter (·.shaded)).length

/-- The probability of selecting a shaded triangle -/
def shaded_probability (d : Diagram) : ℚ :=
  d.shaded_count / d.triangles.length

/-- Theorem stating the probability of selecting a shaded triangle is 1/3 -/
theorem shaded_probability_is_one_third (d : Diagram) :
  d.shaded_count = 3 ∧ d.triangles.length = 9 →
  shaded_probability d = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_probability_is_one_third_l4171_417159


namespace NUMINAMATH_CALUDE_right_triangle_trig_l4171_417141

theorem right_triangle_trig (A B C : ℝ) (h_right : A^2 + B^2 = C^2) 
  (h_hypotenuse : C = 15) (h_leg : A = 7) :
  Real.sqrt ((C^2 - A^2) / C^2) = 4 * Real.sqrt 11 / 15 ∧ A / C = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_l4171_417141


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l4171_417112

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digits_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_rearrangement (a b : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧
  (a / 10 = b % 10) ∧ (a % 10 = b / 10)

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_two_digit n ∧
    (n : ℚ) / (digits_product n : ℚ) = 16 / 3 ∧
    is_rearrangement n (n - 9) ∧
    n = 32 := by sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l4171_417112


namespace NUMINAMATH_CALUDE_total_material_calculation_l4171_417103

/-- The amount of concrete ordered in tons -/
def concrete : ℝ := 0.16666666666666666

/-- The amount of bricks ordered in tons -/
def bricks : ℝ := 0.16666666666666666

/-- The amount of stone ordered in tons -/
def stone : ℝ := 0.5

/-- The total amount of material ordered in tons -/
def total_material : ℝ := concrete + bricks + stone

theorem total_material_calculation : total_material = 0.8333333333333332 := by
  sorry

end NUMINAMATH_CALUDE_total_material_calculation_l4171_417103


namespace NUMINAMATH_CALUDE_deposit_calculation_l4171_417133

theorem deposit_calculation (remaining_amount : ℝ) (deposit_percentage : ℝ) :
  remaining_amount = 1350 ∧ deposit_percentage = 0.1 →
  (remaining_amount / (1 - deposit_percentage)) * deposit_percentage = 150 := by
sorry

end NUMINAMATH_CALUDE_deposit_calculation_l4171_417133


namespace NUMINAMATH_CALUDE_x_equals_n_l4171_417182

def x : ℕ → ℚ
  | 0 => 0
  | n + 1 => ((n^2 + n + 1) * x n + 1) / (n^2 + n + 1 - x n)

theorem x_equals_n (n : ℕ) : x n = n := by
  sorry

end NUMINAMATH_CALUDE_x_equals_n_l4171_417182


namespace NUMINAMATH_CALUDE_pyramid_height_theorem_l4171_417123

/-- Properties of the Great Pyramid of Giza --/
structure Pyramid where
  h : ℝ  -- The certain height
  height : ℝ := h + 20  -- The actual height of the pyramid
  width : ℝ := height + 234  -- The width of the pyramid

/-- Theorem about the height of the Great Pyramid of Giza --/
theorem pyramid_height_theorem (p : Pyramid) 
    (sum_condition : p.height + p.width = 1274) : 
    p.h = 1000 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_height_theorem_l4171_417123


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4171_417144

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ) : 
  (∀ n, a (n + 1) - a n = 1) →  -- arithmetic sequence with common difference 1
  (∀ n, S n = n * a 1 + n * (n - 1) / 2) →  -- sum formula for arithmetic sequence
  S 8 = 4 * S 4 →  -- given condition
  a 10 = 19 / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4171_417144


namespace NUMINAMATH_CALUDE_single_point_condition_l4171_417195

/-- The equation of the curve -/
def curve_equation (x y c : ℝ) : Prop :=
  3 * x^2 + y^2 + 6 * x - 12 * y + c = 0

/-- The curve is a single point -/
def is_single_point (c : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, curve_equation p.1 p.2 c

/-- The value of c for which the curve is a single point -/
theorem single_point_condition :
  ∃! c : ℝ, is_single_point c ∧ c = 39 :=
sorry

end NUMINAMATH_CALUDE_single_point_condition_l4171_417195


namespace NUMINAMATH_CALUDE_count_divisible_sum_l4171_417164

theorem count_divisible_sum : ∃ (S : Finset Nat), 
  (∀ n ∈ S, n > 0 ∧ (10 * n) % ((n * (n + 1)) / 2) = 0) ∧
  (∀ n : Nat, n > 0 ∧ (10 * n) % ((n * (n + 1)) / 2) = 0 → n ∈ S) ∧
  Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_sum_l4171_417164


namespace NUMINAMATH_CALUDE_race_finish_orders_l4171_417113

/-- The number of possible finish orders for a race with 4 participants and no ties -/
def finish_orders : ℕ := 24

/-- The number of participants in the race -/
def num_participants : ℕ := 4

/-- Theorem: The number of possible finish orders for a race with 4 participants and no ties is 24 -/
theorem race_finish_orders : 
  finish_orders = Nat.factorial num_participants :=
sorry

end NUMINAMATH_CALUDE_race_finish_orders_l4171_417113


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l4171_417101

theorem solve_system_of_equations (u v : ℚ) 
  (eq1 : 5 * u - 6 * v = 19)
  (eq2 : 3 * u + 5 * v = 1) : 
  u + v = 147 / 129 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l4171_417101


namespace NUMINAMATH_CALUDE_picnic_total_attendance_l4171_417122

/-- The number of persons at a picnic -/
def picnic_attendance (men women adults children : ℕ) : Prop :=
  (men = women + 20) ∧ 
  (adults = children + 20) ∧ 
  (men = 65) ∧
  (men + women + children = 200)

/-- Theorem stating the total number of persons at the picnic -/
theorem picnic_total_attendance :
  ∃ (men women adults children : ℕ),
    picnic_attendance men women adults children :=
by
  sorry

end NUMINAMATH_CALUDE_picnic_total_attendance_l4171_417122


namespace NUMINAMATH_CALUDE_area_of_specific_figure_l4171_417199

/-- A figure composed of squares and triangles -/
structure Figure where
  num_squares : ℕ
  num_triangles : ℕ

/-- The area of a figure in square centimeters -/
def area (f : Figure) : ℝ :=
  f.num_squares + (f.num_triangles * 0.5)

/-- Theorem: The area of a specific figure is 10.5 cm² -/
theorem area_of_specific_figure :
  ∃ (f : Figure), f.num_squares = 8 ∧ f.num_triangles = 5 ∧ area f = 10.5 :=
sorry

end NUMINAMATH_CALUDE_area_of_specific_figure_l4171_417199


namespace NUMINAMATH_CALUDE_january_salary_l4171_417120

/-- Represents the salary structure for a person over 5 months -/
structure SalaryStructure where
  jan : ℝ
  feb : ℝ
  mar : ℝ
  apr : ℝ
  may : ℝ
  bonus : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem january_salary (s : SalaryStructure) 
  (avg_jan_apr : (s.jan + s.feb + s.mar + s.apr) / 4 = 8000)
  (avg_feb_may : (s.feb + s.mar + s.apr + s.may) / 4 = 8400)
  (may_salary : s.may = 6500)
  (apr_raise : s.apr = 1.05 * s.feb)
  (mar_bonus : s.mar = s.feb + s.bonus) :
  s.jan = 4900 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l4171_417120


namespace NUMINAMATH_CALUDE_unique_integer_solution_l4171_417102

theorem unique_integer_solution : ∃! x : ℤ, x + 12 > 14 ∧ -3*x > -9 := by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l4171_417102


namespace NUMINAMATH_CALUDE_probability_one_boy_one_girl_l4171_417161

/-- The probability of selecting exactly one boy and one girl when randomly choosing 2 people from 2 boys and 2 girls -/
theorem probability_one_boy_one_girl (num_boys num_girls : ℕ) (h1 : num_boys = 2) (h2 : num_girls = 2) :
  let total_combinations := num_boys * num_girls + (num_boys.choose 2) + (num_girls.choose 2)
  let favorable_combinations := num_boys * num_girls
  (favorable_combinations : ℚ) / total_combinations = 2/3 := by
sorry

end NUMINAMATH_CALUDE_probability_one_boy_one_girl_l4171_417161


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l4171_417160

/-- The equation of an asymptote of the hyperbola y²/8 - x²/6 = 1 -/
theorem hyperbola_asymptote :
  ∃ (x y : ℝ), (y^2 / 8 - x^2 / 6 = 1) →
  (2 * x - Real.sqrt 3 * y = 0 ∨ 2 * x + Real.sqrt 3 * y = 0) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l4171_417160


namespace NUMINAMATH_CALUDE_inequality_proof_l4171_417118

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4171_417118


namespace NUMINAMATH_CALUDE_square_of_two_digit_is_68_l4171_417124

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_digit (n : ℕ) : ℕ := n / 1000

def last_digit (n : ℕ) : ℕ := n % 10

def middle_digits_sum (n : ℕ) : ℕ := (n / 100 % 10) + (n / 10 % 10)

theorem square_of_two_digit_is_68 (n : ℕ) (h1 : is_four_digit n) 
  (h2 : first_digit n = last_digit n)
  (h3 : first_digit n + last_digit n = middle_digits_sum n)
  (h4 : ∃ m : ℕ, 10 ≤ m ∧ m ≤ 99 ∧ m * m = n) :
  ∃ m : ℕ, m = 68 ∧ m * m = n := by
sorry

end NUMINAMATH_CALUDE_square_of_two_digit_is_68_l4171_417124


namespace NUMINAMATH_CALUDE_vector_b_value_l4171_417140

/-- Given a vector a and conditions on vector b, prove b equals (-3, 6) -/
theorem vector_b_value (a b : ℝ × ℝ) : 
  a = (1, -2) → 
  (∃ k : ℝ, k < 0 ∧ b = k • a) → 
  ‖b‖ = 3 * Real.sqrt 5 → 
  b = (-3, 6) := by
  sorry

end NUMINAMATH_CALUDE_vector_b_value_l4171_417140


namespace NUMINAMATH_CALUDE_washing_machine_cost_l4171_417170

/-- The cost of a washing machine and dryer, with a discount applied --/
theorem washing_machine_cost 
  (washing_machine_cost : ℝ) 
  (dryer_cost : ℝ) 
  (discount_rate : ℝ) 
  (total_after_discount : ℝ) :
  washing_machine_cost = 100 ∧ 
  dryer_cost = washing_machine_cost - 30 ∧
  discount_rate = 0.1 ∧
  total_after_discount = 153 ∧
  (1 - discount_rate) * (washing_machine_cost + dryer_cost) = total_after_discount →
  washing_machine_cost = 100 := by
sorry

end NUMINAMATH_CALUDE_washing_machine_cost_l4171_417170


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l4171_417108

-- Problem 1
theorem problem_one : (Real.sqrt 3)^2 + |1 - Real.sqrt 3| + ((-27 : ℝ)^(1/3)) = Real.sqrt 3 - 1 := by
  sorry

-- Problem 2
theorem problem_two : (Real.sqrt 12 - Real.sqrt (1/3)) * Real.sqrt 6 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l4171_417108


namespace NUMINAMATH_CALUDE_binary_1011011_equals_91_l4171_417173

def binary_to_decimal (binary_digits : List Bool) : ℕ :=
  binary_digits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_1011011_equals_91 :
  binary_to_decimal [true, true, false, true, true, false, true] = 91 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011011_equals_91_l4171_417173


namespace NUMINAMATH_CALUDE_number_puzzle_l4171_417138

theorem number_puzzle (a b : ℕ) : 
  a + b = 21875 →
  (a % 5 = 0 ∨ b % 5 = 0) →
  b = 10 * a + 5 →
  b - a = 17893 := by
sorry

end NUMINAMATH_CALUDE_number_puzzle_l4171_417138


namespace NUMINAMATH_CALUDE_sum_and_product_of_averages_l4171_417117

def avg1 : ℚ := (0 + 100) / 2

def avg2 : ℚ := (0 + 50) / 2

def avg3 : ℚ := 560 / 8

theorem sum_and_product_of_averages :
  avg1 + avg2 + avg3 = 145 ∧ avg1 * avg2 * avg3 = 87500 := by sorry

end NUMINAMATH_CALUDE_sum_and_product_of_averages_l4171_417117


namespace NUMINAMATH_CALUDE_price_difference_l4171_417115

def coupon_A (P : ℝ) : ℝ := 0.20 * P
def coupon_B : ℝ := 40
def coupon_C (P : ℝ) : ℝ := 0.30 * (P - 150)

def valid_price (P : ℝ) : Prop :=
  P > 150 ∧ coupon_A P ≥ max coupon_B (coupon_C P)

theorem price_difference : 
  ∃ (x y : ℝ), valid_price x ∧ valid_price y ∧
  (∀ P, valid_price P → x ≤ P ∧ P ≤ y) ∧
  y - x = 250 :=
sorry

end NUMINAMATH_CALUDE_price_difference_l4171_417115


namespace NUMINAMATH_CALUDE_george_total_earnings_l4171_417175

/-- The total amount earned by George from selling toys -/
def george_earnings (num_cars : ℕ) (price_per_car : ℕ) (lego_price : ℕ) : ℕ :=
  num_cars * price_per_car + lego_price

/-- Theorem: George earned $45 from selling 3 cars at $5 each and a set of Legos for $30 -/
theorem george_total_earnings : george_earnings 3 5 30 = 45 := by
  sorry

end NUMINAMATH_CALUDE_george_total_earnings_l4171_417175


namespace NUMINAMATH_CALUDE_power_of_fraction_five_sevenths_sixth_l4171_417158

theorem power_of_fraction_five_sevenths_sixth : (5 : ℚ) / 7 ^ 6 = 15625 / 117649 := by
  sorry

end NUMINAMATH_CALUDE_power_of_fraction_five_sevenths_sixth_l4171_417158


namespace NUMINAMATH_CALUDE_conic_section_equation_l4171_417169

/-- A conic section that satisfies specific conditions -/
structure ConicSection where
  -- The conic section passes through these two points
  point_a : (ℝ × ℝ)
  point_b : (ℝ × ℝ)
  -- The conic section shares a common asymptote with this hyperbola
  asymptote_hyperbola : (ℝ → ℝ → Prop)
  -- The conic section is a hyperbola with this focal length
  focal_length : ℝ

/-- The standard equation of a hyperbola -/
def standard_hyperbola_equation (a b : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem stating the standard equation of the conic section -/
theorem conic_section_equation (c : ConicSection)
  (h1 : c.point_a = (2, -Real.sqrt 2 / 2))
  (h2 : c.point_b = (-Real.sqrt 2, -Real.sqrt 3 / 2))
  (h3 : c.asymptote_hyperbola = standard_hyperbola_equation 5 3)
  (h4 : c.focal_length = 8) :
  (standard_hyperbola_equation 10 6 = c.asymptote_hyperbola) ∨
  (standard_hyperbola_equation 6 10 = c.asymptote_hyperbola) :=
sorry

end NUMINAMATH_CALUDE_conic_section_equation_l4171_417169


namespace NUMINAMATH_CALUDE_weighted_average_is_70_55_l4171_417187

def mathematics_score : ℝ := 76
def science_score : ℝ := 65
def social_studies_score : ℝ := 82
def english_score : ℝ := 67
def biology_score : ℝ := 55
def computer_science_score : ℝ := 89
def history_score : ℝ := 74
def geography_score : ℝ := 63
def physics_score : ℝ := 78
def chemistry_score : ℝ := 71

def mathematics_weight : ℝ := 0.20
def science_weight : ℝ := 0.15
def social_studies_weight : ℝ := 0.10
def english_weight : ℝ := 0.15
def biology_weight : ℝ := 0.10
def computer_science_weight : ℝ := 0.05
def history_weight : ℝ := 0.05
def geography_weight : ℝ := 0.10
def physics_weight : ℝ := 0.05
def chemistry_weight : ℝ := 0.05

def weighted_average : ℝ :=
  mathematics_score * mathematics_weight +
  science_score * science_weight +
  social_studies_score * social_studies_weight +
  english_score * english_weight +
  biology_score * biology_weight +
  computer_science_score * computer_science_weight +
  history_score * history_weight +
  geography_score * geography_weight +
  physics_score * physics_weight +
  chemistry_score * chemistry_weight

theorem weighted_average_is_70_55 : weighted_average = 70.55 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_is_70_55_l4171_417187


namespace NUMINAMATH_CALUDE_notebooks_ordered_l4171_417147

theorem notebooks_ordered (initial final lost : ℕ) (h1 : initial = 4) (h2 : lost = 2) (h3 : final = 8) :
  ∃ ordered : ℕ, initial + ordered - lost = final ∧ ordered = 6 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_ordered_l4171_417147


namespace NUMINAMATH_CALUDE_cabinet_can_pass_through_door_l4171_417181

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a room with given dimensions -/
def Room := Dimensions

/-- Represents a cabinet with given dimensions -/
def Cabinet := Dimensions

/-- Represents a door with given dimensions -/
structure Door where
  width : ℝ
  height : ℝ

/-- Checks if a cabinet can pass through a door -/
def can_pass_through (c : Cabinet) (d : Door) : Prop :=
  (c.width ≤ d.width ∧ c.height ≤ d.height) ∨
  (c.width ≤ d.height ∧ c.height ≤ d.width) ∨
  (c.length ≤ d.width ∧ c.height ≤ d.height) ∨
  (c.length ≤ d.height ∧ c.height ≤ d.width)

theorem cabinet_can_pass_through_door 
  (room : Room)
  (cabinet : Cabinet)
  (door : Door)
  (h_room : room = ⟨4, 2.5, 2.3⟩)
  (h_cabinet : cabinet = ⟨1.8, 0.6, 2.1⟩)
  (h_door : door = ⟨0.8, 1.9⟩) :
  can_pass_through cabinet door :=
sorry

end NUMINAMATH_CALUDE_cabinet_can_pass_through_door_l4171_417181


namespace NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l4171_417165

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (a b c d : Point3D) : ℝ := sorry

/-- Calculates the height of a tetrahedron from a vertex to the opposite face -/
def tetrahedronHeight (a b c d : Point3D) : ℝ := sorry

theorem tetrahedron_volume_and_height :
  let a₁ : Point3D := ⟨-2, -1, -1⟩
  let a₂ : Point3D := ⟨0, 3, 2⟩
  let a₃ : Point3D := ⟨3, 1, -4⟩
  let a₄ : Point3D := ⟨-4, 7, 3⟩
  (tetrahedronVolume a₁ a₂ a₃ a₄ = 70/3) ∧
  (tetrahedronHeight a₄ a₁ a₂ a₃ = 140 / Real.sqrt 1021) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l4171_417165


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_l4171_417145

theorem max_consecutive_integers_sum (k : ℕ) : 
  (∃ n : ℕ, (k : ℤ) * (2 * n + k - 1) = 2 * 3^8) →
  k ≤ 108 :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_l4171_417145


namespace NUMINAMATH_CALUDE_interactive_lines_count_l4171_417134

/-- Represents a four-digit number M with specific digit placement. -/
structure FourDigitNumber where
  a : ℕ  -- Thousands place
  b : ℕ  -- Hundreds place
  c : ℕ  -- Ones place
  h1 : c ≠ 0
  h2 : a < 10 ∧ b < 10 ∧ c < 10

/-- Calculates the value of M given its digit representation. -/
def M (n : FourDigitNumber) : ℕ :=
  1000 * n.a + 100 * n.b + 10 + n.c

/-- Calculates the value of N by moving the ones digit to the front. -/
def N (n : FourDigitNumber) : ℕ :=
  1000 * n.c + 100 * n.a + 10 * n.b + 1

/-- Defines the function F(M) = (M + N) / 11. -/
def F (n : FourDigitNumber) : ℚ :=
  (M n + N n : ℚ) / 11

/-- Predicate for the interactive line condition. -/
def IsInteractiveLine (n : FourDigitNumber) : Prop :=
  n.c = n.a + n.b

/-- The main theorem stating the number of interactive lines satisfying the condition. -/
theorem interactive_lines_count :
  (∃ (S : Finset FourDigitNumber),
    S.card = 8 ∧
    (∀ n ∈ S, IsInteractiveLine n ∧ ∃ k : ℕ, F n = 6 * k) ∧
    (∀ n : FourDigitNumber, IsInteractiveLine n → (∃ k : ℕ, F n = 6 * k) → n ∈ S)) :=
  sorry


end NUMINAMATH_CALUDE_interactive_lines_count_l4171_417134


namespace NUMINAMATH_CALUDE_car_average_speed_l4171_417191

/-- Proves that the average speed of a car traveling 140 km in the first hour
    and 40 km in the second hour is 90 km/h. -/
theorem car_average_speed : 
  let speed1 : ℝ := 140 -- Speed in km/h for the first hour
  let speed2 : ℝ := 40  -- Speed in km/h for the second hour
  let time1 : ℝ := 1    -- Time in hours for the first hour
  let time2 : ℝ := 1    -- Time in hours for the second hour
  let total_distance : ℝ := speed1 * time1 + speed2 * time2
  let total_time : ℝ := time1 + time2
  let average_speed : ℝ := total_distance / total_time
  average_speed = 90 := by
sorry

end NUMINAMATH_CALUDE_car_average_speed_l4171_417191


namespace NUMINAMATH_CALUDE_oatmeal_cookies_divisible_by_containers_l4171_417156

/-- The number of chocolate chip cookies Kiara baked -/
def chocolate_chip_cookies : ℕ := 48

/-- The number of containers Kiara wants to use -/
def num_containers : ℕ := 6

/-- The number of oatmeal cookies Kiara baked -/
def oatmeal_cookies : ℕ := sorry

/-- Theorem stating that the number of oatmeal cookies must be divisible by the number of containers -/
theorem oatmeal_cookies_divisible_by_containers :
  oatmeal_cookies % num_containers = 0 :=
sorry

end NUMINAMATH_CALUDE_oatmeal_cookies_divisible_by_containers_l4171_417156


namespace NUMINAMATH_CALUDE_theater_capacity_is_50_l4171_417188

/-- The maximum capacity of a movie theater -/
def theater_capacity (ticket_price : ℕ) (tickets_sold : ℕ) (loss_amount : ℕ) : ℕ :=
  tickets_sold + loss_amount / ticket_price

/-- Theorem: The maximum capacity of the movie theater is 50 people -/
theorem theater_capacity_is_50 :
  theater_capacity 8 24 208 = 50 := by
  sorry

end NUMINAMATH_CALUDE_theater_capacity_is_50_l4171_417188


namespace NUMINAMATH_CALUDE_fruit_seller_gain_percent_l4171_417186

theorem fruit_seller_gain_percent (cost_price selling_price : ℝ) 
  (h1 : cost_price > 0)
  (h2 : selling_price > 0)
  (h3 : 150 * selling_price - 150 * cost_price = 30 * selling_price) :
  (((150 * selling_price - 150 * cost_price) / (150 * cost_price)) * 100 = 25) :=
by sorry

end NUMINAMATH_CALUDE_fruit_seller_gain_percent_l4171_417186


namespace NUMINAMATH_CALUDE_power_of_three_implies_large_prime_factor_l4171_417180

theorem power_of_three_implies_large_prime_factor (n : ℕ+) :
  (∃ k : ℕ, 125 * n + 22 = 3^k) →
  ∃ p : ℕ, p > 100 ∧ Prime p ∧ p ∣ (125 * n + 29) :=
by sorry

end NUMINAMATH_CALUDE_power_of_three_implies_large_prime_factor_l4171_417180


namespace NUMINAMATH_CALUDE_distribution_methods_count_l4171_417153

/-- The number of ways to distribute 4 out of 7 different books to 4 students -/
def distribute_books (total_books : ℕ) (books_to_distribute : ℕ) (students : ℕ) 
  (restricted_books : ℕ) (restricted_student : ℕ) : ℕ :=
  (total_books - restricted_books) * 
  (Nat.factorial (total_books - 1) / (Nat.factorial (total_books - books_to_distribute) * 
   Nat.factorial (books_to_distribute - 1)))

/-- Theorem stating the number of distribution methods -/
theorem distribution_methods_count : 
  distribute_books 7 4 4 2 1 = 600 := by
  sorry

end NUMINAMATH_CALUDE_distribution_methods_count_l4171_417153


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l4171_417184

theorem quadratic_root_difference (a b c : ℝ) (h : b^2 - 4*a*c ≥ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a ≠ 0 ∧ a*x^2 + b*x + c = 0 ∧ r₁ * r₂ < 20 → |r₁ - r₂| = 2 :=
by
  sorry

#check quadratic_root_difference 1 (-8) 15

end NUMINAMATH_CALUDE_quadratic_root_difference_l4171_417184


namespace NUMINAMATH_CALUDE_ball_probability_l4171_417128

theorem ball_probability (total : ℕ) (red : ℕ) (purple : ℕ) 
  (h1 : total = 100) 
  (h2 : red = 9) 
  (h3 : purple = 3) : 
  (total - (red + purple)) / total = 88 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l4171_417128


namespace NUMINAMATH_CALUDE_intersection_area_is_pi_l4171_417149

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- Define set M
def M : Set (ℝ × ℝ) := {p | f p.1 + f p.2 ≤ 0}

-- Define set N
def N : Set (ℝ × ℝ) := {p | f p.1 - f p.2 ≥ 0}

-- Theorem statement
theorem intersection_area_is_pi : MeasureTheory.volume (M ∩ N) = π := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_pi_l4171_417149


namespace NUMINAMATH_CALUDE_overhead_percentage_l4171_417193

theorem overhead_percentage (purchase_price markup net_profit : ℝ) :
  purchase_price = 48 →
  markup = 45 →
  net_profit = 12 →
  (((purchase_price + markup - net_profit) - purchase_price) / purchase_price) * 100 = 68.75 := by
  sorry

end NUMINAMATH_CALUDE_overhead_percentage_l4171_417193


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l4171_417114

/-- A parabola with equation y = x^2 - 6x + c has its vertex on the x-axis if and only if c = 9 -/
theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + c = 0 ∧ ∀ y : ℝ, y^2 - 6*y + c ≥ x^2 - 6*x + c) ↔ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l4171_417114


namespace NUMINAMATH_CALUDE_calculate_expression_l4171_417172

theorem calculate_expression : (2 - 5 * (-1/2)^2) / (-1/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4171_417172


namespace NUMINAMATH_CALUDE_two_number_difference_l4171_417119

theorem two_number_difference (a b : ℕ) : 
  a + b = 22305 →
  a % 5 = 0 →
  b = (a / 10) + 3 →
  a - b = 14872 :=
by sorry

end NUMINAMATH_CALUDE_two_number_difference_l4171_417119


namespace NUMINAMATH_CALUDE_inequality_proof_l4171_417106

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a * (3 * a - 1)) / (1 + a^2) + 
  (b * (3 * b - 1)) / (1 + b^2) + 
  (c * (3 * c - 1)) / (1 + c^2) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4171_417106


namespace NUMINAMATH_CALUDE_digit_2009_is_zero_l4171_417189

/-- Represents the sequence of consecutive natural numbers starting from 1 -/
def consecutiveNaturals : ℕ → ℕ
  | 0 => 1
  | n + 1 => consecutiveNaturals n + 1

/-- Returns the nth digit in the sequence of consecutive natural numbers -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 2009th digit in the sequence is 0 -/
theorem digit_2009_is_zero : nthDigit 2009 = 0 := by sorry

end NUMINAMATH_CALUDE_digit_2009_is_zero_l4171_417189


namespace NUMINAMATH_CALUDE_simplify_polynomial_l4171_417197

theorem simplify_polynomial (s : ℝ) : (2*s^2 - 5*s + 3) - (s^2 + 4*s - 6) = s^2 - 9*s + 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l4171_417197


namespace NUMINAMATH_CALUDE_power_of_nine_l4171_417135

theorem power_of_nine (n : ℕ) (h : 3^(2*n) = 81) : 9^(n+1) = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_nine_l4171_417135


namespace NUMINAMATH_CALUDE_cubic_factorization_l4171_417176

theorem cubic_factorization (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l4171_417176


namespace NUMINAMATH_CALUDE_smallest_perimeter_circle_circle_center_on_line_l4171_417125

-- Define the points A and B
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (-1, 4)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 2 * x - y - 4 = 0

-- Define the general equation of a circle
def circle_general_eq (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Define the standard equation of a circle
def circle_standard_eq (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Theorem for the circle with smallest perimeter
theorem smallest_perimeter_circle :
  ∃ (x y : ℝ), x^2 + y^2 - 2*y - 9 = 0 ∧
  circle_general_eq x y 0 1 (5 : ℝ) ∧
  (∀ (a b r : ℝ), circle_general_eq A.1 A.2 a b r → 
   circle_general_eq B.1 B.2 a b r → 
   r^2 ≥ 10) := by sorry

-- Theorem for the circle with center on the given line
theorem circle_center_on_line :
  ∃ (x y : ℝ), (x - 3)^2 + (y - 2)^2 = 20 ∧
  circle_standard_eq x y 3 2 (2 * Real.sqrt 5) ∧
  line_eq 3 2 := by sorry

end NUMINAMATH_CALUDE_smallest_perimeter_circle_circle_center_on_line_l4171_417125


namespace NUMINAMATH_CALUDE_gcd_120_75_l4171_417157

theorem gcd_120_75 : Nat.gcd 120 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_120_75_l4171_417157


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l4171_417137

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > 2 → x > 1) ∧ ¬(x > 1 → x > 2) := by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l4171_417137


namespace NUMINAMATH_CALUDE_order_of_values_l4171_417111

/-- A function f is even if f(-x) = f(x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f is monotonically increasing on [0, +∞) if
    for all a, b ≥ 0, a < b implies f(a) < f(b) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ a b, 0 ≤ a → 0 ≤ b → a < b → f a < f b

theorem order_of_values (f : ℝ → ℝ) 
    (h_even : EvenFunction f) 
    (h_mono : MonoIncreasing f) :
    f (-π) > f 3 ∧ f 3 > f (-2) := by
  sorry

end NUMINAMATH_CALUDE_order_of_values_l4171_417111


namespace NUMINAMATH_CALUDE_insertion_possible_l4171_417152

/-- Represents a natural number with exactly 2007 digits -/
def Number2007 := { n : ℕ | 10^2006 ≤ n ∧ n < 10^2007 }

/-- Represents the operation of removing 7 digits from a number -/
def remove_seven_digits (n : Number2007) : ℕ := sorry

/-- Represents the operation of inserting 7 digits into a number -/
def insert_seven_digits (n : Number2007) : Number2007 := sorry

/-- The main theorem -/
theorem insertion_possible (a b : Number2007) :
  (∃ (c : ℕ), remove_seven_digits a = c ∧ remove_seven_digits b = c) →
  (∃ (d : Number2007), ∃ (f g : Number2007 → Number2007),
    f a = d ∧ g b = d ∧ 
    (∀ x : Number2007, ∃ y, insert_seven_digits x = f x ∧ insert_seven_digits y = g x)) :=
sorry

end NUMINAMATH_CALUDE_insertion_possible_l4171_417152
