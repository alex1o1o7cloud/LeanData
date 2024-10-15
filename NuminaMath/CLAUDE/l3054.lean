import Mathlib

namespace NUMINAMATH_CALUDE_dog_weight_difference_l3054_305410

theorem dog_weight_difference (labrador_initial : ℝ) (dachshund_initial : ℝ) 
  (growth_rate : ℝ) (h1 : labrador_initial = 40) (h2 : dachshund_initial = 12) 
  (h3 : growth_rate = 0.25) : 
  labrador_initial * (1 + growth_rate) - dachshund_initial * (1 + growth_rate) = 35 := by
  sorry

end NUMINAMATH_CALUDE_dog_weight_difference_l3054_305410


namespace NUMINAMATH_CALUDE_partition_inequality_l3054_305418

def f (n : ℕ) : ℕ := sorry

theorem partition_inequality (n : ℕ) (h : n ≥ 1) :
  f (n + 1) ≤ (f n + f (n + 2)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_partition_inequality_l3054_305418


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3054_305438

theorem quadratic_equation_properties (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c = 0 → 
    ((a + b + c = 0 → b^2 - 4*a*c ≥ 0) ∧
    ((-1 ∈ {x | a * x^2 + b * x + c = 0} ∧ 2 ∈ {x | a * x^2 + b * x + c = 0}) → 2*a + c = 0) ∧
    ((∃ x y, x ≠ y ∧ a * x^2 + c = 0 ∧ a * y^2 + c = 0) → 
      ∃ u v, u ≠ v ∧ a * u^2 + b * u + c = 0 ∧ a * v^2 + b * v + c = 0))) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3054_305438


namespace NUMINAMATH_CALUDE_pool_wall_area_ratio_l3054_305415

theorem pool_wall_area_ratio : 
  let pool_radius : ℝ := 20
  let wall_width : ℝ := 4
  let pool_area := π * pool_radius^2
  let total_area := π * (pool_radius + wall_width)^2
  let wall_area := total_area - pool_area
  wall_area / pool_area = 11 / 25 := by
sorry

end NUMINAMATH_CALUDE_pool_wall_area_ratio_l3054_305415


namespace NUMINAMATH_CALUDE_sequence_sum_l3054_305465

theorem sequence_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 9*x₂ + 25*x₃ + 49*x₄ + 81*x₅ + 121*x₆ + 169*x₇ = 2)
  (eq2 : 9*x₁ + 25*x₂ + 49*x₃ + 81*x₄ + 121*x₅ + 169*x₆ + 225*x₇ = 24)
  (eq3 : 25*x₁ + 49*x₂ + 81*x₃ + 121*x₄ + 169*x₅ + 225*x₆ + 289*x₇ = 246) :
  49*x₁ + 81*x₂ + 121*x₃ + 169*x₄ + 225*x₅ + 289*x₆ + 361*x₇ = 668 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3054_305465


namespace NUMINAMATH_CALUDE_g_of_negative_three_l3054_305486

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x - 2

-- Theorem statement
theorem g_of_negative_three : g (-3) = -17 := by
  sorry

end NUMINAMATH_CALUDE_g_of_negative_three_l3054_305486


namespace NUMINAMATH_CALUDE_expand_polynomial_l3054_305431

theorem expand_polynomial (x : ℝ) : (7 * x^3 - 5 * x + 2) * (4 * x^2) = 28 * x^5 - 20 * x^3 + 8 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l3054_305431


namespace NUMINAMATH_CALUDE_max_value_inequality_l3054_305441

theorem max_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (6 * a * b) / (9 * b^2 + a^2) + (2 * a * b) / (b^2 + a^2) ≤ 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3054_305441


namespace NUMINAMATH_CALUDE_poetic_line_contrast_l3054_305458

/-- Represents a poetic line with two parts -/
structure PoeticLine :=
  (part1 : String)
  (part2 : String)

/-- Determines if a given part of a poetic line represents stillness -/
def isStillness (part : String) : Prop :=
  sorry

/-- Determines if a given part of a poetic line represents motion -/
def isMotion (part : String) : Prop :=
  sorry

/-- Determines if a poetic line contrasts stillness and motion -/
def contrastsStillnessAndMotion (line : PoeticLine) : Prop :=
  (isStillness line.part1 ∧ isMotion line.part2) ∨ (isMotion line.part1 ∧ isStillness line.part2)

/-- The four poetic lines given in the problem -/
def lineA : PoeticLine :=
  { part1 := "The bridge echoes with the distant barking of dogs"
  , part2 := "and the courtyard is empty with people asleep" }

def lineB : PoeticLine :=
  { part1 := "The stove fire illuminates the heaven and earth"
  , part2 := "and the red stars are mixed with the purple smoke" }

def lineC : PoeticLine :=
  { part1 := "The cold trees begin to have bird activities"
  , part2 := "and the frosty bridge has no human passage yet" }

def lineD : PoeticLine :=
  { part1 := "The crane cries over the quiet Chu mountain"
  , part2 := "and the frost is white on the autumn river in the morning" }

theorem poetic_line_contrast :
  contrastsStillnessAndMotion lineA ∧
  contrastsStillnessAndMotion lineB ∧
  contrastsStillnessAndMotion lineC ∧
  ¬contrastsStillnessAndMotion lineD :=
sorry

end NUMINAMATH_CALUDE_poetic_line_contrast_l3054_305458


namespace NUMINAMATH_CALUDE_decision_box_distinguishes_l3054_305496

/-- Represents a flowchart element --/
inductive FlowchartElement
  | ProcessingBox
  | DecisionBox
  | InputOutputBox
  | StartEndBox

/-- Represents a flowchart structure --/
structure FlowchartStructure :=
  (elements : Set FlowchartElement)

/-- Definition of a conditional structure --/
def is_conditional (s : FlowchartStructure) : Prop :=
  FlowchartElement.DecisionBox ∈ s.elements ∧ 
  (∃ (b1 b2 : Set FlowchartElement), b1 ⊆ s.elements ∧ b2 ⊆ s.elements ∧ b1 ≠ b2)

/-- Definition of a sequential structure --/
def is_sequential (s : FlowchartStructure) : Prop :=
  FlowchartElement.DecisionBox ∉ s.elements

/-- Theorem: The inclusion of a decision box distinguishes conditional from sequential structures --/
theorem decision_box_distinguishes :
  ∀ (s : FlowchartStructure), 
    (is_conditional s ↔ FlowchartElement.DecisionBox ∈ s.elements) ∧
    (is_sequential s ↔ FlowchartElement.DecisionBox ∉ s.elements) :=
by sorry

end NUMINAMATH_CALUDE_decision_box_distinguishes_l3054_305496


namespace NUMINAMATH_CALUDE_range_of_a_l3054_305451

/-- The line passing through points on a 2D plane. -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point on a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if two points are on opposite sides of a line. -/
def oppositeSides (p1 p2 : Point2D) (l : Line2D) : Prop :=
  (l.a * p1.x + l.b * p1.y + l.c) * (l.a * p2.x + l.b * p2.y + l.c) < 0

/-- The theorem statement. -/
theorem range_of_a :
  ∀ a : ℝ,
  (oppositeSides (Point2D.mk 0 0) (Point2D.mk 1 1) (Line2D.mk 1 1 (-a))) ↔
  (0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3054_305451


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l3054_305417

theorem consecutive_integers_around_sqrt3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l3054_305417


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3054_305435

-- Define the equations
def equation1 (x : ℝ) : Prop := (2*x - 5)/6 - (3*x + 1)/2 = 1
def equation2 (x : ℝ) : Prop := 3*x - 7*(x - 1) = 3 - 2*(x + 3)

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = -2 := by sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3054_305435


namespace NUMINAMATH_CALUDE_expression_value_l3054_305480

theorem expression_value (x y z w : ℝ) 
  (eq1 : 4*x*z + y*w = 4) 
  (eq2 : x*w + y*z = 8) : 
  (2*x + y) * (2*z + w) = 20 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3054_305480


namespace NUMINAMATH_CALUDE_angle_QNR_is_165_l3054_305423

/-- An isosceles triangle PQR with a point N inside -/
structure IsoscelesTriangleWithPoint where
  /-- The measure of angle PRQ in degrees -/
  angle_PRQ : ℝ
  /-- The measure of angle PNR in degrees -/
  angle_PNR : ℝ
  /-- The measure of angle PRN in degrees -/
  angle_PRN : ℝ
  /-- PR = QR (isosceles condition) -/
  isosceles : True
  /-- N is in the interior of the triangle -/
  N_interior : True
  /-- Angle PRQ is 108 degrees -/
  h_PRQ : angle_PRQ = 108
  /-- Angle PNR is 9 degrees -/
  h_PNR : angle_PNR = 9
  /-- Angle PRN is 21 degrees -/
  h_PRN : angle_PRN = 21

/-- Theorem: In the given isosceles triangle with point N, angle QNR is 165 degrees -/
theorem angle_QNR_is_165 (t : IsoscelesTriangleWithPoint) : ∃ angle_QNR : ℝ, angle_QNR = 165 := by
  sorry

end NUMINAMATH_CALUDE_angle_QNR_is_165_l3054_305423


namespace NUMINAMATH_CALUDE_min_value_of_max_expression_l3054_305444

theorem min_value_of_max_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (M : ℝ), M = max (1 / (a * c) + b) (max (1 / a + b * c) (a / b + c)) ∧ M ≥ 2 ∧ 
  (∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    max (1 / (a' * c') + b') (max (1 / a' + b' * c') (a' / b' + c')) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_max_expression_l3054_305444


namespace NUMINAMATH_CALUDE_marble_difference_is_negative_21_l3054_305481

/-- The number of marbles Jonny has minus the number of marbles Marissa has -/
def marbleDifference : ℤ :=
  let mara_marbles := 12 * 2
  let markus_marbles := 2 * 13
  let jonny_marbles := 18
  let marissa_marbles := 3 * 5 + 3 * 8
  jonny_marbles - marissa_marbles

/-- Theorem stating the difference in marbles between Jonny and Marissa -/
theorem marble_difference_is_negative_21 : marbleDifference = -21 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_is_negative_21_l3054_305481


namespace NUMINAMATH_CALUDE_intersection_inequality_solution_set_l3054_305450

/-- Given a line and a hyperbola intersecting at two points, 
    prove the solution set of a related inequality. -/
theorem intersection_inequality_solution_set 
  (k₀ k b m n : ℝ) : 
  (∃ (x : ℝ), k₀ * x + b = k^2 / x ∧ 
              (x = m ∧ k₀ * m + b = -1 ∧ k^2 / m = -1) ∨
              (x = n ∧ k₀ * n + b = 2 ∧ k^2 / n = 2)) →
  {x : ℝ | x^2 > k₀ * k^2 + b * x} = {x : ℝ | x < -1 ∨ x > 2} :=
by sorry

end NUMINAMATH_CALUDE_intersection_inequality_solution_set_l3054_305450


namespace NUMINAMATH_CALUDE_newspapers_sold_l3054_305404

theorem newspapers_sold (magazines : ℕ) (total : ℕ) (newspapers : ℕ) : 
  magazines = 425 → total = 700 → newspapers = total - magazines → newspapers = 275 := by
  sorry

end NUMINAMATH_CALUDE_newspapers_sold_l3054_305404


namespace NUMINAMATH_CALUDE_probability_not_math_and_physics_is_four_fifths_l3054_305416

def subjects := 6
def selected := 3

def probability_not_math_and_physics : ℚ :=
  1 - (Nat.choose 4 1 : ℚ) / (Nat.choose subjects selected : ℚ)

theorem probability_not_math_and_physics_is_four_fifths :
  probability_not_math_and_physics = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_math_and_physics_is_four_fifths_l3054_305416


namespace NUMINAMATH_CALUDE_team_selection_with_quadruplets_l3054_305484

/-- The number of ways to choose a team with restrictions on quadruplets -/
def choose_team (total_players : ℕ) (team_size : ℕ) (quadruplets : ℕ) : ℕ :=
  Nat.choose total_players team_size - Nat.choose (total_players - quadruplets) (team_size - quadruplets)

/-- Theorem stating the number of ways to choose the team under given conditions -/
theorem team_selection_with_quadruplets :
  choose_team 16 11 4 = 3576 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_with_quadruplets_l3054_305484


namespace NUMINAMATH_CALUDE_adrianna_gum_purchase_l3054_305495

/-- The number of gum pieces Adrianna bought in the second store visit -/
def second_store_purchase (initial_gum : ℕ) (first_store_purchase : ℕ) (total_friends : ℕ) : ℕ :=
  total_friends - (initial_gum + first_store_purchase)

/-- Theorem: Given Adrianna's initial 10 pieces of gum, 3 pieces bought from the first store,
    and 15 friends who received gum, the number of gum pieces bought in the second store visit is 2. -/
theorem adrianna_gum_purchase :
  second_store_purchase 10 3 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_adrianna_gum_purchase_l3054_305495


namespace NUMINAMATH_CALUDE_trig_simplification_l3054_305476

theorem trig_simplification (α : ℝ) : 
  (2 * Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 2 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3054_305476


namespace NUMINAMATH_CALUDE_interval_equivalence_l3054_305442

theorem interval_equivalence (x : ℝ) : 
  (3/4 < x ∧ x < 4/5) ↔ (3 < 5*x + 1 ∧ 5*x + 1 < 5) ∧ (3 < 4*x ∧ 4*x < 5) :=
by sorry

end NUMINAMATH_CALUDE_interval_equivalence_l3054_305442


namespace NUMINAMATH_CALUDE_wage_ratio_is_two_to_one_l3054_305493

/-- The ratio of a man's daily wage to a woman's daily wage -/
def wage_ratio (men_wage women_wage : ℚ) : ℚ := men_wage / women_wage

/-- The total earnings of a group of workers over a period -/
def total_earnings (num_workers : ℕ) (num_days : ℕ) (daily_wage : ℚ) : ℚ :=
  (num_workers : ℚ) * (num_days : ℚ) * daily_wage

theorem wage_ratio_is_two_to_one :
  ∃ (men_wage women_wage : ℚ),
    total_earnings 40 10 men_wage = 14400 ∧
    total_earnings 40 30 women_wage = 21600 ∧
    wage_ratio men_wage women_wage = 2 := by
  sorry

end NUMINAMATH_CALUDE_wage_ratio_is_two_to_one_l3054_305493


namespace NUMINAMATH_CALUDE_problem_solution_l3054_305477

theorem problem_solution (m n : ℕ) 
  (h1 : m + 10 < n + 1) 
  (h2 : (m + (m + 4) + (m + 10) + (n + 1) + (n + 2) + 2*n) / 6 = n) 
  (h3 : ((m + 10) + (n + 1)) / 2 = n) : 
  m + n = 21 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3054_305477


namespace NUMINAMATH_CALUDE_line_intercepts_opposite_l3054_305479

/-- A line with equation (a-2)x + y - a = 0 has intercepts on the coordinate axes that are opposite numbers if and only if a = 0 or a = 1 -/
theorem line_intercepts_opposite (a : ℝ) : 
  (∃ x y : ℝ, (a - 2) * x + y - a = 0 ∧ 
   ((x = 0 ∧ y ≠ 0) ∨ (x ≠ 0 ∧ y = 0)) ∧
   (x = 0 → y = a) ∧
   (y = 0 → x = a / (a - 2)) ∧
   x = -y) ↔ 
  (a = 0 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_opposite_l3054_305479


namespace NUMINAMATH_CALUDE_composite_product_division_l3054_305472

def first_six_composite_product : ℕ := 4 * 6 * 8 * 9 * 10 * 12
def next_six_composite_product : ℕ := 14 * 15 * 16 * 18 * 20 * 21

theorem composite_product_division :
  (first_six_composite_product : ℚ) / next_six_composite_product = 1 / 49 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_division_l3054_305472


namespace NUMINAMATH_CALUDE_cubic_factorization_l3054_305411

theorem cubic_factorization (x : ℝ) : x^3 - 8*x^2 + 16*x = x*(x-4)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3054_305411


namespace NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l3054_305492

theorem sum_of_solutions_squared_equation (x : ℝ) :
  (∃ a b : ℝ, (a - 8)^2 = 64 ∧ (b - 8)^2 = 64 ∧ a + b = 16) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_squared_equation_l3054_305492


namespace NUMINAMATH_CALUDE_promotional_price_equiv_correct_method_l3054_305478

/-- Represents the promotional price calculation for books -/
def promotional_price (x : ℝ) : ℝ := 0.8 * (x - 15)

/-- Represents the correct method of calculation as described in option C -/
def correct_method (x : ℝ) : ℝ := 0.8 * (x - 15)

/-- Theorem stating that the promotional price calculation is equivalent to the correct method -/
theorem promotional_price_equiv_correct_method :
  ∀ x : ℝ, promotional_price x = correct_method x := by
  sorry

end NUMINAMATH_CALUDE_promotional_price_equiv_correct_method_l3054_305478


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l3054_305414

theorem divisibility_of_expression : ∃ k : ℤ, 27195^8 - 10887^8 + 10152^8 = 26460 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l3054_305414


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l3054_305440

/-- Given two lines l₁ and l₂ in the xy-plane, where l₁ has the equation 2x - y + 1 = 0
    and l₂ is symmetric to l₁ with respect to the line y = -x,
    prove that the equation of l₂ is x - 2y + 1 = 0 -/
theorem symmetric_line_equation :
  ∀ (l₁ l₂ : Set (ℝ × ℝ)),
  (∀ x y, (x, y) ∈ l₁ ↔ 2 * x - y + 1 = 0) →
  (∀ x y, (x, y) ∈ l₂ ↔ ∃ x' y', (x', y') ∈ l₁ ∧ x + x' = y + y') →
  (∀ x y, (x, y) ∈ l₂ ↔ x - 2 * y + 1 = 0) :=
by sorry


end NUMINAMATH_CALUDE_symmetric_line_equation_l3054_305440


namespace NUMINAMATH_CALUDE_average_of_reeyas_scores_l3054_305400

def reeyas_scores : List ℕ := [55, 67, 76, 82, 85]

theorem average_of_reeyas_scores :
  (List.sum reeyas_scores) / (List.length reeyas_scores) = 73 := by
  sorry

end NUMINAMATH_CALUDE_average_of_reeyas_scores_l3054_305400


namespace NUMINAMATH_CALUDE_buratino_arrival_time_l3054_305499

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds hours and minutes to a given time -/
def addTime (t : Time) (h : Nat) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + h * 60 + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem buratino_arrival_time :
  let departureTime : Time := { hours := 13, minutes := 40 }
  let normalJourneyTime : Real := 7.5
  let fasterJourneyTime : Real := normalJourneyTime * 4 / 5
  let timeDifference : Real := normalJourneyTime - fasterJourneyTime
  timeDifference = 1.5 →
  addTime departureTime 7 30 = { hours := 21, minutes := 10 } :=
by sorry

end NUMINAMATH_CALUDE_buratino_arrival_time_l3054_305499


namespace NUMINAMATH_CALUDE_ship_cargo_after_loading_l3054_305473

/-- The total cargo on a ship after loading additional cargo in the Bahamas -/
theorem ship_cargo_after_loading (initial_cargo additional_cargo : ℕ) :
  initial_cargo = 5973 →
  additional_cargo = 8723 →
  initial_cargo + additional_cargo = 14696 := by
  sorry

end NUMINAMATH_CALUDE_ship_cargo_after_loading_l3054_305473


namespace NUMINAMATH_CALUDE_exists_non_complementary_acute_angles_l3054_305461

-- Define what an acute angle is
def is_acute_angle (angle : ℝ) : Prop := 0 < angle ∧ angle < 90

-- Define what complementary angles are
def are_complementary (angle1 angle2 : ℝ) : Prop := angle1 + angle2 = 90

-- Theorem statement
theorem exists_non_complementary_acute_angles :
  ∃ (angle1 angle2 : ℝ), is_acute_angle angle1 ∧ is_acute_angle angle2 ∧ ¬(are_complementary angle1 angle2) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_complementary_acute_angles_l3054_305461


namespace NUMINAMATH_CALUDE_inequality_proof_l3054_305452

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  b * c / a + c * a / b + a * b / c ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3054_305452


namespace NUMINAMATH_CALUDE_coordinate_sum_of_h_l3054_305449

/-- Given a function g where g(2) = 5, and a function h where h(x) = (g(x))^2 for all x,
    the sum of the coordinates of the point (2, h(2)) is 27. -/
theorem coordinate_sum_of_h (g : ℝ → ℝ) (h : ℝ → ℝ) 
    (h_def : ∀ x, h x = (g x)^2) 
    (g_val : g 2 = 5) : 
  2 + h 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_h_l3054_305449


namespace NUMINAMATH_CALUDE_value_after_percentage_increase_l3054_305412

theorem value_after_percentage_increase 
  (x : ℝ) (p : ℝ) (y : ℝ) 
  (h1 : x = 400) 
  (h2 : p = 20) :
  y = x * (1 + p / 100) → y = 480 := by
  sorry

end NUMINAMATH_CALUDE_value_after_percentage_increase_l3054_305412


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3054_305491

theorem binomial_expansion_coefficient (x : ℝ) :
  ∃ m : ℝ, (1 + 2*x)^3 = 1 + 6*x + m*x^2 + 8*x^3 ∧ m = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3054_305491


namespace NUMINAMATH_CALUDE_subtract_fractions_l3054_305401

theorem subtract_fractions : (7/3 * 12/5) - 3/5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l3054_305401


namespace NUMINAMATH_CALUDE_least_k_for_convergence_l3054_305474

def u : ℕ → ℚ
  | 0 => 1/4
  | n + 1 => 2 * u n - 2 * (u n)^2

def L : ℚ := 1/2

theorem least_k_for_convergence :
  (∀ k : ℕ, k < 10 → |u k - L| > 1/2^1000) ∧
  |u 10 - L| ≤ 1/2^1000 := by sorry

end NUMINAMATH_CALUDE_least_k_for_convergence_l3054_305474


namespace NUMINAMATH_CALUDE_f_properties_l3054_305437

-- Define the function f
def f (a b x : ℝ) : ℝ := |x - a| + |x - b|

-- State the theorem
theorem f_properties (a b : ℝ) (h : -1 < a ∧ a < b) :
  -- Part 1
  (∀ x : ℝ, f 1 2 x ≥ Real.sin x) ∧
  -- Part 2
  {x : ℝ | f a b x < a + b + 2} = {x : ℝ | |2*x - a - b| < a + b + 2} :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l3054_305437


namespace NUMINAMATH_CALUDE_largest_gold_coins_distribution_l3054_305466

theorem largest_gold_coins_distribution (n : ℕ) : 
  n < 100 ∧ 
  n % 13 = 3 ∧ 
  (∀ m : ℕ, m < 100 ∧ m % 13 = 3 → m ≤ n) → 
  n = 94 := by
sorry

end NUMINAMATH_CALUDE_largest_gold_coins_distribution_l3054_305466


namespace NUMINAMATH_CALUDE_salesperson_allocation_l3054_305494

/-- Represents the problem of determining the number of salespersons to send to a branch office --/
theorem salesperson_allocation
  (total_salespersons : ℕ)
  (initial_avg_income : ℝ)
  (hq_income_increase : ℝ)
  (branch_income_factor : ℝ)
  (h_total : total_salespersons = 100)
  (h_hq_increase : hq_income_increase = 0.2)
  (h_branch_factor : branch_income_factor = 3.5)
  (x : ℕ) :
  (((total_salespersons - x) * (1 + hq_income_increase) * initial_avg_income ≥ 
    total_salespersons * initial_avg_income) ∧
   (x * branch_income_factor * initial_avg_income ≥ 
    0.5 * total_salespersons * initial_avg_income)) →
  (x = 15 ∨ x = 16) :=
by sorry

end NUMINAMATH_CALUDE_salesperson_allocation_l3054_305494


namespace NUMINAMATH_CALUDE_speedster_convertibles_l3054_305433

theorem speedster_convertibles (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) 
  (h1 : speedsters = (2 * total) / 3)
  (h2 : convertibles = (4 * speedsters) / 5)
  (h3 : total - speedsters = 60) :
  convertibles = 96 := by
  sorry

end NUMINAMATH_CALUDE_speedster_convertibles_l3054_305433


namespace NUMINAMATH_CALUDE_problem_solution_l3054_305467

theorem problem_solution (x : ℝ) (h1 : x > 0) (h2 : x * ⌊x⌋ = 72) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3054_305467


namespace NUMINAMATH_CALUDE_log_sqrt10_1000_sqrt10_l3054_305457

theorem log_sqrt10_1000_sqrt10 : Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt10_1000_sqrt10_l3054_305457


namespace NUMINAMATH_CALUDE_fixed_points_of_f_composition_l3054_305489

def f (x : ℝ) : ℝ := x^2 - 5*x + 1

theorem fixed_points_of_f_composition :
  ∀ x : ℝ, f (f x) = f x ↔ 
    x = (5 + Real.sqrt 21) / 2 ∨
    x = (5 - Real.sqrt 21) / 2 ∨
    x = (11 + Real.sqrt 101) / 2 ∨
    x = (11 - Real.sqrt 101) / 2 :=
by sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_composition_l3054_305489


namespace NUMINAMATH_CALUDE_chairlift_halfway_l3054_305402

def total_chairs : ℕ := 96
def current_chair : ℕ := 66

def halfway_chair (total : ℕ) (current : ℕ) : ℕ :=
  (current - total / 2 + total) % total

theorem chairlift_halfway :
  halfway_chair total_chairs current_chair = 18 := by
sorry

end NUMINAMATH_CALUDE_chairlift_halfway_l3054_305402


namespace NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l3054_305446

theorem tangent_line_to_logarithmic_curve (a : ℝ) :
  (∃ x₀ y₀ : ℝ, 
    y₀ = x₀ + 1 ∧ 
    y₀ = Real.log (x₀ + a) ∧
    (1 : ℝ) = 1 / (x₀ + a)) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l3054_305446


namespace NUMINAMATH_CALUDE_aubrey_distance_to_school_l3054_305464

/-- The distance from Aubrey's home to his school -/
def distance_to_school (journey_time : ℝ) (average_speed : ℝ) : ℝ :=
  journey_time * average_speed

/-- Theorem stating the distance from Aubrey's home to his school -/
theorem aubrey_distance_to_school :
  distance_to_school 4 22 = 88 := by
  sorry

end NUMINAMATH_CALUDE_aubrey_distance_to_school_l3054_305464


namespace NUMINAMATH_CALUDE_spring_sales_calculation_l3054_305420

-- Define the total annual sandwich sales
def total_sales : ℝ := 15

-- Define the seasonal sales
def winter_sales : ℝ := 3
def summer_sales : ℝ := 4
def fall_sales : ℝ := 5

-- Define the winter sales percentage
def winter_percentage : ℝ := 0.2

-- Theorem to prove
theorem spring_sales_calculation :
  ∃ (spring_sales : ℝ),
    winter_percentage * total_sales = winter_sales ∧
    spring_sales + summer_sales + fall_sales + winter_sales = total_sales ∧
    spring_sales = 3 := by
  sorry


end NUMINAMATH_CALUDE_spring_sales_calculation_l3054_305420


namespace NUMINAMATH_CALUDE_square_of_z_l3054_305425

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := 2 + 5 * i

-- Theorem statement
theorem square_of_z : z^2 = -21 + 20 * i := by
  sorry

end NUMINAMATH_CALUDE_square_of_z_l3054_305425


namespace NUMINAMATH_CALUDE_inequality_proof_l3054_305498

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : x > 0) (h2 : n > 0) :
  x + n^n / x^n ≥ n + 1 → n^n = n^n :=
by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l3054_305498


namespace NUMINAMATH_CALUDE_team_A_builds_22_5_meters_per_day_l3054_305443

def team_A_build_rate : ℝ → Prop := λ x => 
  (150 / x = 100 / (2 * x - 30)) ∧ (x > 0)

theorem team_A_builds_22_5_meters_per_day :
  ∃ x : ℝ, team_A_build_rate x ∧ x = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_team_A_builds_22_5_meters_per_day_l3054_305443


namespace NUMINAMATH_CALUDE_tan_identity_l3054_305439

theorem tan_identity (α : ℝ) (h : Real.tan (α + π / 6) = 2) :
  Real.tan (2 * α + 7 * π / 12) = -1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_identity_l3054_305439


namespace NUMINAMATH_CALUDE_four_Y_one_equals_27_l3054_305448

/-- Definition of the Y operation -/
def Y (a b : ℝ) : ℝ := 3 * (a^2 - 2*a*b + b^2)

/-- Theorem stating that 4 Y 1 = 27 -/
theorem four_Y_one_equals_27 : Y 4 1 = 27 := by sorry

end NUMINAMATH_CALUDE_four_Y_one_equals_27_l3054_305448


namespace NUMINAMATH_CALUDE_ship_grain_problem_l3054_305434

theorem ship_grain_problem (spilled_grain : ℕ) (remaining_grain : ℕ) 
  (h1 : spilled_grain = 49952) (h2 : remaining_grain = 918) : 
  spilled_grain + remaining_grain = 50870 := by
  sorry

end NUMINAMATH_CALUDE_ship_grain_problem_l3054_305434


namespace NUMINAMATH_CALUDE_game_probability_game_probability_value_l3054_305459

theorem game_probability : ℝ :=
  let total_outcomes : ℕ := 16 * 16
  let matching_outcomes : ℕ := 16
  let non_matching_outcomes : ℕ := total_outcomes - matching_outcomes
  (non_matching_outcomes : ℝ) / total_outcomes

theorem game_probability_value : game_probability = 15 / 16 := by sorry

end NUMINAMATH_CALUDE_game_probability_game_probability_value_l3054_305459


namespace NUMINAMATH_CALUDE_complex_z_value_l3054_305422

theorem complex_z_value (z : ℂ) : z / Complex.I = 2 - 3 * Complex.I → z = 3 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_z_value_l3054_305422


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3054_305497

theorem quadratic_inequality_solution (a c : ℝ) (h : ∀ x, (a * x^2 + 5 * x + c > 0) ↔ (1/3 < x ∧ x < 1/2)) :
  (a = -6 ∧ c = -1) ∧
  (∀ b : ℝ, 
    (∀ x, (a * x^2 + (a * c + b) * x + b * c ≥ 0) ↔ 
      ((b > 6 ∧ 1 ≤ x ∧ x ≤ b/6) ∨
       (b = 6 ∧ x = 1) ∨
       (b < 6 ∧ b/6 ≤ x ∧ x ≤ 1)))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3054_305497


namespace NUMINAMATH_CALUDE_speed_of_M_constant_l3054_305447

/-- Represents a crank-slider mechanism -/
structure CrankSlider where
  ω : ℝ  -- Angular velocity of the crank
  OA : ℝ  -- Length of OA
  AB : ℝ  -- Length of AB
  AM : ℝ  -- Length of AM

/-- The speed of point M in a crank-slider mechanism -/
def speed_of_M (cs : CrankSlider) : ℝ := cs.OA * cs.ω

/-- Theorem: The speed of point M is constant and equal to OA * ω -/
theorem speed_of_M_constant (cs : CrankSlider) 
  (h1 : cs.ω = 10)
  (h2 : cs.OA = 90)
  (h3 : cs.AB = 90)
  (h4 : cs.AM = cs.AB / 2) :
  speed_of_M cs = 900 := by
  sorry

end NUMINAMATH_CALUDE_speed_of_M_constant_l3054_305447


namespace NUMINAMATH_CALUDE_twelfth_term_is_fifteen_l3054_305490

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition : a 7 + a 9 = 16
  fourth_term : a 4 = 1

/-- The 12th term of the arithmetic sequence is 15 -/
theorem twelfth_term_is_fifteen (seq : ArithmeticSequence) : seq.a 12 = 15 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_is_fifteen_l3054_305490


namespace NUMINAMATH_CALUDE_choose_20_6_l3054_305469

theorem choose_20_6 : Nat.choose 20 6 = 2584 := by
  sorry

end NUMINAMATH_CALUDE_choose_20_6_l3054_305469


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_medians_l3054_305483

/-- If a triangle has medians of lengths 3, 4, and 6, then its perimeter is 26. -/
theorem triangle_perimeter_from_medians (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (med1 : ∃ (m : ℝ), m = 3 ∧ m^2 = (b^2 + c^2) / 4 - a^2 / 16)
  (med2 : ∃ (m : ℝ), m = 4 ∧ m^2 = (a^2 + c^2) / 4 - b^2 / 16)
  (med3 : ∃ (m : ℝ), m = 6 ∧ m^2 = (a^2 + b^2) / 4 - c^2 / 16) :
  a + b + c = 26 := by
  sorry


end NUMINAMATH_CALUDE_triangle_perimeter_from_medians_l3054_305483


namespace NUMINAMATH_CALUDE_arithmetic_sequence_variance_l3054_305430

/-- Given an arithmetic sequence with common difference d,
    prove that if the variance of the first five terms is 2, then d = ±1 -/
theorem arithmetic_sequence_variance (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  ((a 1 - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2 +
   ((a 1 + d) - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2 +
   ((a 1 + 2*d) - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2 +
   ((a 1 + 3*d) - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2 +
   ((a 1 + 4*d) - ((a 1 + (a 1 + d) + (a 1 + 2*d) + (a 1 + 3*d) + (a 1 + 4*d)) / 5))^2) / 5 = 2 →
  d = 1 ∨ d = -1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_variance_l3054_305430


namespace NUMINAMATH_CALUDE_sqrt_sum_zero_implies_power_sum_zero_l3054_305482

theorem sqrt_sum_zero_implies_power_sum_zero (a b : ℝ) :
  Real.sqrt (a + 1) + Real.sqrt (b - 1) = 0 → a^1011 + b^1011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_zero_implies_power_sum_zero_l3054_305482


namespace NUMINAMATH_CALUDE_inequality_proof_l3054_305462

-- Define the set A
def A : Set ℝ := {x | x > 1}

-- State the theorem
theorem inequality_proof (m n : ℝ) (hm : m ∈ A) (hn : n ∈ A) (h_sum : m + n = 4) :
  n^2 / (m - 1) + m^2 / (n - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3054_305462


namespace NUMINAMATH_CALUDE_morgans_blue_pens_l3054_305419

theorem morgans_blue_pens (red_pens black_pens total_pens : ℕ) 
  (h1 : red_pens = 65)
  (h2 : black_pens = 58)
  (h3 : total_pens = 168)
  : total_pens - (red_pens + black_pens) = 45 := by
  sorry

end NUMINAMATH_CALUDE_morgans_blue_pens_l3054_305419


namespace NUMINAMATH_CALUDE_unique_solution_system_l3054_305488

theorem unique_solution_system (x y : ℝ) :
  x^2 + y^2 = 2 ∧ 
  (x^2 / (2 - y)) + (y^2 / (2 - x)) = 2 →
  x = 1 ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3054_305488


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3054_305436

theorem min_value_quadratic :
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3054_305436


namespace NUMINAMATH_CALUDE_ticket_cost_calculation_l3054_305429

/-- Calculates the total amount spent on tickets given the prices and quantities -/
def total_ticket_cost (adult_price child_price : ℚ) (total_tickets child_tickets : ℕ) : ℚ :=
  let adult_tickets := total_tickets - child_tickets
  adult_price * adult_tickets + child_price * child_tickets

/-- Theorem stating that the total amount spent on tickets is $83.50 -/
theorem ticket_cost_calculation :
  total_ticket_cost (5.50 : ℚ) (3.50 : ℚ) 21 16 = (83.50 : ℚ) := by
  sorry

#eval total_ticket_cost (5.50 : ℚ) (3.50 : ℚ) 21 16

end NUMINAMATH_CALUDE_ticket_cost_calculation_l3054_305429


namespace NUMINAMATH_CALUDE_lopez_family_seating_arrangements_l3054_305428

/-- Represents the number of family members -/
def family_size : ℕ := 5

/-- Represents the number of car seats -/
def car_seats : ℕ := 5

/-- Represents the number of eligible drivers -/
def eligible_drivers : ℕ := 3

/-- Calculates the number of seating arrangements -/
def seating_arrangements (f s d : ℕ) : ℕ :=
  d * (f - 1) * Nat.factorial (f - 2)

/-- Theorem stating the number of seating arrangements for the Lopez family -/
theorem lopez_family_seating_arrangements :
  seating_arrangements family_size car_seats eligible_drivers = 72 :=
by sorry

end NUMINAMATH_CALUDE_lopez_family_seating_arrangements_l3054_305428


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3054_305470

/-- A geometric sequence with its sum -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  q : ℝ      -- The common ratio
  h1 : ∀ n, a (n + 1) = a n * q  -- Definition of geometric sequence
  h2 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)  -- Sum formula for geometric sequence

/-- The theorem statement -/
theorem geometric_sequence_ratio (seq : GeometricSequence) 
  (h3 : seq.a 3 = 4)
  (h4 : seq.S 3 = 12) :
  seq.q = 1 ∨ seq.q = -1/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3054_305470


namespace NUMINAMATH_CALUDE_annas_gold_cost_per_gram_l3054_305471

/-- Calculates the cost per gram of Anna's gold -/
theorem annas_gold_cost_per_gram 
  (gary_gold : ℝ) 
  (gary_cost_per_gram : ℝ) 
  (anna_gold : ℝ) 
  (total_cost : ℝ) 
  (h1 : gary_gold = 30)
  (h2 : gary_cost_per_gram = 15)
  (h3 : anna_gold = 50)
  (h4 : total_cost = 1450)
  (h5 : total_cost = gary_gold * gary_cost_per_gram + anna_gold * (total_cost - gary_gold * gary_cost_per_gram) / anna_gold) :
  (total_cost - gary_gold * gary_cost_per_gram) / anna_gold = 20 := by
  sorry

#check annas_gold_cost_per_gram

end NUMINAMATH_CALUDE_annas_gold_cost_per_gram_l3054_305471


namespace NUMINAMATH_CALUDE_line_circle_no_intersection_l3054_305475

theorem line_circle_no_intersection (a : ℝ) :
  (∀ x y : ℝ, x + y = a → x^2 + y^2 ≠ 1) ↔ (a > 1 ∨ a < -1) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_no_intersection_l3054_305475


namespace NUMINAMATH_CALUDE_base8_addition_and_conversion_l3054_305424

/-- Converts a base 8 number to base 10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 8 --/
def base10_to_base8 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 16 --/
def base10_to_base16 (n : ℕ) : ℕ := sorry

/-- Adds two base 8 numbers and returns the result in base 8 --/
def add_base8 (a b : ℕ) : ℕ := 
  base10_to_base8 (base8_to_base10 a + base8_to_base10 b)

theorem base8_addition_and_conversion :
  let a : ℕ := 537 -- In base 8
  let b : ℕ := 246 -- In base 8
  let sum_base8 : ℕ := add_base8 a b
  let sum_base16 : ℕ := base10_to_base16 (base8_to_base10 sum_base8)
  sum_base8 = 1005 ∧ sum_base16 = 0x205 := by sorry

end NUMINAMATH_CALUDE_base8_addition_and_conversion_l3054_305424


namespace NUMINAMATH_CALUDE_divisible_by_55_l3054_305485

theorem divisible_by_55 (n : ℤ) : 
  55 ∣ (n^2 + 3*n + 1) ↔ n % 55 = 6 ∨ n % 55 = 46 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_55_l3054_305485


namespace NUMINAMATH_CALUDE_newOp_seven_three_l3054_305453

-- Define the new operation ⊗
def newOp (p q : ℝ) : ℝ := p^2 - 2*q

-- Theorem to prove
theorem newOp_seven_three : newOp 7 3 = 43 := by
  sorry

end NUMINAMATH_CALUDE_newOp_seven_three_l3054_305453


namespace NUMINAMATH_CALUDE_centerville_library_budget_percentage_l3054_305487

/-- Proves that the percentage of Centerville's annual budget spent on the public library is 15% -/
theorem centerville_library_budget_percentage
  (library_expense : ℕ)
  (park_percentage : ℚ)
  (remaining_budget : ℕ)
  (h1 : library_expense = 3000)
  (h2 : park_percentage = 24 / 100)
  (h3 : remaining_budget = 12200)
  : ∃ (total_budget : ℕ), 
    (library_expense : ℚ) / total_budget = 15 / 100 :=
sorry

end NUMINAMATH_CALUDE_centerville_library_budget_percentage_l3054_305487


namespace NUMINAMATH_CALUDE_investment_rate_calculation_l3054_305454

/-- Prove that if Rs. 1600 is divided into two parts, where one part (P1) is Rs. 1100
    invested at 6% and the other part (P2) is the remainder, and the total annual
    interest from both parts is Rs. 85, then P2 must be invested at 3.8%. -/
theorem investment_rate_calculation (total : ℝ) (p1 : ℝ) (p2 : ℝ) (r1 : ℝ) (total_interest : ℝ) :
  total = 1600 →
  p1 = 1100 →
  p2 = total - p1 →
  r1 = 6 →
  total_interest = 85 →
  p1 * r1 / 100 + p2 * (total_interest - p1 * r1 / 100) / p2 = total_interest →
  (total_interest - p1 * r1 / 100) / p2 * 100 = 3.8 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_calculation_l3054_305454


namespace NUMINAMATH_CALUDE_incorrect_average_theorem_l3054_305407

def incorrect_average (n : ℕ) (correct_avg : ℚ) (correct_num wrong_num : ℚ) : ℚ :=
  (n * correct_avg - correct_num + wrong_num) / n

theorem incorrect_average_theorem :
  incorrect_average 10 24 76 26 = 19 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_average_theorem_l3054_305407


namespace NUMINAMATH_CALUDE_range_of_f_on_interval_range_of_a_l3054_305413

-- Define the function f(x) = x^2 - ax + 4
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 4

-- Part 1: Range of f(x) on [1, 3] when a = 3
theorem range_of_f_on_interval (x : ℝ) (h : x ∈ Set.Icc 1 3) :
  ∃ y ∈ Set.Icc (7/4) 4, y = f 3 x :=
sorry

-- Part 2: Range of values for a
theorem range_of_a (a : ℝ) (h : ∀ x ∈ Set.Icc 0 2, f a x ≤ 4) :
  a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_range_of_a_l3054_305413


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l3054_305406

/-- The wood measurement problem from "The Mathematical Classic of Sunzi" -/
theorem sunzi_wood_measurement_problem (x y : ℝ) : 
  (y - x = 4.5 ∧ y / 2 = x - 1) ↔ 
  (∃ (rope_length wood_length : ℝ),
    rope_length > wood_length ∧
    rope_length - wood_length = 4.5 ∧
    rope_length / 2 > wood_length - 1 ∧
    rope_length / 2 < wood_length) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l3054_305406


namespace NUMINAMATH_CALUDE_blackboard_numbers_l3054_305427

/-- The sum of reciprocals of the initial numbers on the blackboard -/
def initial_sum (m : ℕ) : ℚ := (2 * m) / (2 * m + 1)

/-- The operation performed in each move -/
def move (a b c : ℚ) : ℚ := (a * b * c) / (a * b + b * c + c * a)

theorem blackboard_numbers (m : ℕ) (h1 : m ≥ 2) :
  ∀ x : ℚ, 
    (∃ (nums : List ℚ), 
      (nums.length = 2) ∧ 
      (4/3 ∈ nums) ∧ 
      (x ∈ nums) ∧ 
      (1 / (4/3) + 1 / x = initial_sum m)) →
    x > 4 := by sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l3054_305427


namespace NUMINAMATH_CALUDE_prime_solution_uniqueness_l3054_305460

theorem prime_solution_uniqueness :
  ∀ p q : ℕ,
  Prime p →
  Prime q →
  p^2 - 6*p*q + q^2 + 3*q - 1 = 0 →
  (p = 17 ∧ q = 3) ∨ (p = 3 ∧ q = 17) :=
by sorry

end NUMINAMATH_CALUDE_prime_solution_uniqueness_l3054_305460


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3054_305463

theorem inequality_system_solution (x : ℝ) :
  (5 * x + 1 > 3 * (x - 1)) ∧ ((1 / 2) * x < 3) → -2 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3054_305463


namespace NUMINAMATH_CALUDE_regular_star_polygon_points_l3054_305405

/-- A regular star polygon with n points, where each point has two associated angles. -/
structure RegularStarPolygon where
  n : ℕ
  A : Fin n → ℝ
  B : Fin n → ℝ
  all_A_congruent : ∀ i j, A i = A j
  all_B_congruent : ∀ i j, B i = B j
  A_less_than_B : ∀ i, A i = B i - 20

/-- The number of points in a regular star polygon satisfying the given conditions is 18. -/
theorem regular_star_polygon_points (p : RegularStarPolygon) : p.n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_star_polygon_points_l3054_305405


namespace NUMINAMATH_CALUDE_paint_cost_per_quart_paint_cost_example_l3054_305456

/-- The cost of paint per quart for a cube with given dimensions and coverage -/
theorem paint_cost_per_quart (cube_side : ℝ) (coverage_per_quart : ℝ) (total_cost : ℝ) : ℝ :=
  let surface_area := 6 * cube_side^2
  let quarts_needed := surface_area / coverage_per_quart
  total_cost / quarts_needed

/-- The cost of paint per quart is $3.20 for the given conditions -/
theorem paint_cost_example : paint_cost_per_quart 10 120 16 = 3.20 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_per_quart_paint_cost_example_l3054_305456


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3054_305468

/-- Given an arithmetic sequence with common difference d ≠ 0,
    if a₁, a₃, a₇ form a geometric sequence, then a₁/d = 2 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  (∀ n, a (n + 1) = a n + d) →
  (∃ r, a 3 = a 1 * r ∧ a 7 = a 3 * r) →
  a 1 / d = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3054_305468


namespace NUMINAMATH_CALUDE_regular_tile_area_theorem_l3054_305455

/-- Represents the properties of a tiled wall -/
structure TiledWall where
  total_area : ℝ
  regular_tile_length : ℝ
  regular_tile_width : ℝ
  jumbo_tile_length : ℝ
  jumbo_tile_width : ℝ
  jumbo_tile_ratio : ℝ
  regular_tile_count_ratio : ℝ

/-- The area covered by regular tiles in a tiled wall -/
def regular_tile_area (wall : TiledWall) : ℝ :=
  wall.total_area * wall.regular_tile_count_ratio

/-- Theorem stating the area covered by regular tiles in a specific wall configuration -/
theorem regular_tile_area_theorem (wall : TiledWall) 
  (h1 : wall.total_area = 220)
  (h2 : wall.jumbo_tile_ratio = 1/3)
  (h3 : wall.regular_tile_count_ratio = 2/3)
  (h4 : wall.jumbo_tile_length = 3 * wall.regular_tile_length)
  (h5 : wall.jumbo_tile_width = wall.regular_tile_width)
  : regular_tile_area wall = 146.67 := by
  sorry

#check regular_tile_area_theorem

end NUMINAMATH_CALUDE_regular_tile_area_theorem_l3054_305455


namespace NUMINAMATH_CALUDE_equidistant_points_bound_l3054_305409

/-- A set of points in a plane where no three points are collinear -/
structure PointSet where
  S : Set (ℝ × ℝ)
  noncollinear : ∀ (p q r : ℝ × ℝ), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r →
    (p.1 - q.1) * (r.2 - q.2) ≠ (r.1 - q.1) * (p.2 - q.2)

/-- The property that for each point, there are k equidistant points -/
def has_k_equidistant (PS : PointSet) (k : ℕ) : Prop :=
  ∀ p ∈ PS.S, ∃ (T : Set (ℝ × ℝ)), T ⊆ PS.S ∧ T.ncard = k ∧
    ∀ q ∈ T, q ≠ p → ∃ d : ℝ, d > 0 ∧ (p.1 - q.1)^2 + (p.2 - q.2)^2 = d^2

theorem equidistant_points_bound (n k : ℕ) (h_pos : 0 < n ∧ 0 < k) (PS : PointSet)
    (h_card : PS.S.ncard = n) (h_equi : has_k_equidistant PS k) :
    k ≤ (1 : ℝ)/2 + Real.sqrt (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_points_bound_l3054_305409


namespace NUMINAMATH_CALUDE_tiles_required_for_room_floor_l3054_305432

def room_length : Real := 6.24
def room_width : Real := 4.32
def tile_side : Real := 0.30

theorem tiles_required_for_room_floor :
  ⌈(room_length * room_width) / (tile_side * tile_side)⌉ = 300 := by
  sorry

end NUMINAMATH_CALUDE_tiles_required_for_room_floor_l3054_305432


namespace NUMINAMATH_CALUDE_earring_ratio_l3054_305445

theorem earring_ratio (bella_earrings monica_earrings rachel_earrings : ℕ) :
  bella_earrings = 10 ∧
  bella_earrings = monica_earrings / 4 ∧
  bella_earrings + monica_earrings + rachel_earrings = 70 →
  monica_earrings / rachel_earrings = 2 := by
  sorry

end NUMINAMATH_CALUDE_earring_ratio_l3054_305445


namespace NUMINAMATH_CALUDE_inequalities_proof_l3054_305421

theorem inequalities_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b ≥ 2*a - b) ∧ (a^2 / b + b^2 / c + c^2 / a ≥ a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l3054_305421


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l3054_305408

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l3054_305408


namespace NUMINAMATH_CALUDE_quadratic_equations_roots_l3054_305403

theorem quadratic_equations_roots :
  (∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 3 ∧ ∀ x : ℝ, x^2 - 3*x = 0 ↔ x = x₁ ∨ x = x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 5/4 ∧ x₂ = -1 ∧ ∀ x : ℝ, 4*x^2 - x - 5 = 0 ↔ x = x₁ ∨ x = x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -2/3 ∧ ∀ x : ℝ, 3*x*(x-1) = 2-2*x ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_roots_l3054_305403


namespace NUMINAMATH_CALUDE_percentage_70_79_is_800_27_l3054_305426

/-- Represents the frequency distribution of test scores -/
structure ScoreDistribution where
  score_90_100 : Nat
  score_80_89 : Nat
  score_70_79 : Nat
  score_60_69 : Nat
  score_below_60 : Nat

/-- Calculates the percentage of students in the 70%-79% range -/
def percentage_70_79 (dist : ScoreDistribution) : Rat :=
  let total := dist.score_90_100 + dist.score_80_89 + dist.score_70_79 + dist.score_60_69 + dist.score_below_60
  (dist.score_70_79 : Rat) / total * 100

/-- The given frequency distribution -/
def history_class_distribution : ScoreDistribution :=
  { score_90_100 := 5
    score_80_89 := 7
    score_70_79 := 8
    score_60_69 := 4
    score_below_60 := 3 }

theorem percentage_70_79_is_800_27 :
  percentage_70_79 history_class_distribution = 800 / 27 := by
  sorry

end NUMINAMATH_CALUDE_percentage_70_79_is_800_27_l3054_305426
