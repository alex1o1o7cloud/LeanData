import Mathlib

namespace NUMINAMATH_CALUDE_percent_relation_l1808_180805

theorem percent_relation (a b c : ℝ) (h1 : c = 0.30 * a) (h2 : b = 1.20 * a) : 
  c = 0.25 * b := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l1808_180805


namespace NUMINAMATH_CALUDE_petyas_friends_l1808_180890

theorem petyas_friends (total_stickers : ℕ) : 
  (∃ (x : ℕ), 5 * x + 8 = total_stickers ∧ 6 * x = total_stickers + 11) → 
  (∃ (x : ℕ), x = 19 ∧ 5 * x + 8 = total_stickers ∧ 6 * x = total_stickers + 11) :=
by sorry

end NUMINAMATH_CALUDE_petyas_friends_l1808_180890


namespace NUMINAMATH_CALUDE_trig_identity_l1808_180871

theorem trig_identity (α : ℝ) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.sin (α - π / 6) ^ 2 - Real.cos (5 * π / 6 + α) = (2 + Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1808_180871


namespace NUMINAMATH_CALUDE_odd_square_sum_of_consecutive_b_l1808_180807

def a : ℕ → ℕ
  | n => if n % 2 = 1 then 4 * ((n + 1) / 2) - 2 else 4 * (n / 2) - 1

def b : ℕ → ℕ
  | n => if n % 2 = 1 then 8 * ((n - 1) / 2) + 3 else 8 * (n / 2) - 2

theorem odd_square_sum_of_consecutive_b (k : ℕ) (h : k ≥ 1) :
  ∃ n : ℕ, (2 * k + 1)^2 = b n + b (n + 1) :=
sorry

end NUMINAMATH_CALUDE_odd_square_sum_of_consecutive_b_l1808_180807


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1808_180850

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x - k + 1 = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y - k + 1 = 0 → y = x) → 
  k = 0 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1808_180850


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_divisible_by_15_l1808_180872

theorem sum_of_fifth_powers_divisible_by_15 
  (a b c d e : ℤ) 
  (h : a + b + c + d + e = 0) : 
  ∃ k : ℤ, a^5 + b^5 + c^5 + d^5 + e^5 = 15 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_divisible_by_15_l1808_180872


namespace NUMINAMATH_CALUDE_prob_b_is_point_four_l1808_180893

/-- Given two events a and b, prove that the probability of b is 0.4 -/
theorem prob_b_is_point_four (a b : Set α) (p : Set α → ℝ) 
  (h1 : p a = 2/5)
  (h2 : p (a ∩ b) = 0.16000000000000003)
  (h3 : p (a ∩ b) = p a * p b) : 
  p b = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_prob_b_is_point_four_l1808_180893


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1808_180806

theorem algebraic_expression_value (p q : ℝ) : 
  (8 * p + 2 * q + 1 = -2022) → 
  (-8 * p + -2 * q + 1 = 2024) := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1808_180806


namespace NUMINAMATH_CALUDE_inverse_of_singular_matrix_l1808_180860

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; 8, -4]

theorem inverse_of_singular_matrix :
  Matrix.det A = 0 → A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_singular_matrix_l1808_180860


namespace NUMINAMATH_CALUDE_roots_sum_bound_l1808_180836

theorem roots_sum_bound (u v : ℂ) : 
  u ≠ v → 
  u^2023 = 1 → 
  v^2023 = 1 → 
  Complex.abs (u + v) < Real.sqrt (2 + Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_bound_l1808_180836


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_fifths_l1808_180897

theorem one_thirds_in_nine_fifths : (9 : ℚ) / 5 / (1 / 3) = 27 / 5 := by sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_fifths_l1808_180897


namespace NUMINAMATH_CALUDE_pasture_rent_is_140_l1808_180829

/-- Represents the rent share of a person -/
structure RentShare where
  oxen : ℕ
  months : ℕ
  payment : ℕ

/-- Calculates the total rent of a pasture given the rent shares of three people -/
def totalRent (a b c : RentShare) : ℕ :=
  let totalOxenMonths := a.oxen * a.months + b.oxen * b.months + c.oxen * c.months
  let costPerOxenMonth := c.payment / (c.oxen * c.months)
  costPerOxenMonth * totalOxenMonths

/-- Theorem stating that the total rent of the pasture is 140 -/
theorem pasture_rent_is_140 (a b c : RentShare)
  (ha : a.oxen = 10 ∧ a.months = 7)
  (hb : b.oxen = 12 ∧ b.months = 5)
  (hc : c.oxen = 15 ∧ c.months = 3 ∧ c.payment = 36) :
  totalRent a b c = 140 := by
  sorry

end NUMINAMATH_CALUDE_pasture_rent_is_140_l1808_180829


namespace NUMINAMATH_CALUDE_four_digit_number_with_specific_factors_l1808_180815

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def count_prime_factors (n : ℕ) : ℕ := sorry
def count_non_prime_factors (n : ℕ) : ℕ := sorry
def count_total_factors (n : ℕ) : ℕ := sorry

theorem four_digit_number_with_specific_factors :
  ∃ (n : ℕ), is_four_digit n ∧ 
             count_prime_factors n = 3 ∧ 
             count_non_prime_factors n = 39 ∧ 
             count_total_factors n = 42 :=
sorry

end NUMINAMATH_CALUDE_four_digit_number_with_specific_factors_l1808_180815


namespace NUMINAMATH_CALUDE_pencil_buyers_difference_l1808_180868

/-- The cost of a pencil in cents -/
def pencil_cost : ℕ := 13

/-- The number of cents paid by eighth graders -/
def eighth_grade_total : ℕ := 208

/-- The number of cents paid by seventh graders -/
def seventh_grade_total : ℕ := 181

/-- The number of cents paid by sixth graders -/
def sixth_grade_total : ℕ := 234

/-- The number of sixth graders -/
def sixth_graders : ℕ := 45

theorem pencil_buyers_difference :
  (sixth_grade_total / pencil_cost) - (seventh_grade_total / pencil_cost) = 4 :=
by sorry

end NUMINAMATH_CALUDE_pencil_buyers_difference_l1808_180868


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1808_180842

/-- The length of the major axis of an ellipse formed by intersecting a plane with a right circular cylinder --/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem stating the length of the major axis for the given conditions --/
theorem ellipse_major_axis_length :
  major_axis_length 2 1.4 = 5.6 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1808_180842


namespace NUMINAMATH_CALUDE_equation_transformation_l1808_180851

theorem equation_transformation (x : ℝ) (y : ℝ) (h : y = (x^2 + 2) / (x + 1)) :
  ((x^2 + 2) / (x + 1) + (5*x + 5) / (x^2 + 2) = 6) ↔ (y^2 - 6*y + 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_equation_transformation_l1808_180851


namespace NUMINAMATH_CALUDE_intersection_complement_equals_two_l1808_180838

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {1, 3}

theorem intersection_complement_equals_two :
  A ∩ (U \ B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_two_l1808_180838


namespace NUMINAMATH_CALUDE_workout_difference_l1808_180814

/-- Represents Oliver's workout schedule over four days -/
structure WorkoutSchedule where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ

/-- Checks if a workout schedule satisfies the given conditions -/
def is_valid_schedule (s : WorkoutSchedule) : Prop :=
  s.monday = 4 ∧
  s.tuesday < s.monday ∧
  s.wednesday = 2 * s.monday ∧
  s.thursday = 2 * s.tuesday ∧
  s.monday + s.tuesday + s.wednesday + s.thursday = 18

/-- Theorem stating that for any valid workout schedule, 
    the difference between Monday's and Tuesday's workout time is 2 hours -/
theorem workout_difference (s : WorkoutSchedule) 
  (h : is_valid_schedule s) : s.monday - s.tuesday = 2 := by
  sorry

end NUMINAMATH_CALUDE_workout_difference_l1808_180814


namespace NUMINAMATH_CALUDE_clock_malfunction_proof_l1808_180812

/-- Represents a time in HH:MM format -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Represents a single digit change due to malfunction -/
inductive DigitChange
  | Increase
  | Decrease
  | NoChange

/-- Applies a digit change to a number -/
def applyDigitChange (n : Nat) (change : DigitChange) : Nat :=
  match change with
  | DigitChange.Increase => (n + 1) % 10
  | DigitChange.Decrease => (n + 9) % 10
  | DigitChange.NoChange => n

/-- Applies changes to all digits of a time -/
def applyChanges (t : Time) (h1 h2 m1 m2 : DigitChange) : Time :=
  let newHours := applyDigitChange (t.hours / 10) h1 * 10 + applyDigitChange (t.hours % 10) h2
  let newMinutes := applyDigitChange (t.minutes / 10) m1 * 10 + applyDigitChange (t.minutes % 10) m2
  ⟨newHours, newMinutes, sorry⟩

theorem clock_malfunction_proof :
  ∃ (original : Time) (h1 h2 m1 m2 : DigitChange),
    applyChanges original h1 h2 m1 m2 = ⟨20, 50, sorry⟩ ∧
    original = ⟨19, 49, sorry⟩ :=
  sorry

end NUMINAMATH_CALUDE_clock_malfunction_proof_l1808_180812


namespace NUMINAMATH_CALUDE_problem_solution_l1808_180834

def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

theorem problem_solution :
  (∀ x, x ∈ A ∩ B (-4) ↔ 1/2 ≤ x ∧ x < 2) ∧
  (∀ x, x ∈ A ∪ B (-4) ↔ -2 < x ∧ x ≤ 3) ∧
  (∀ a, (Aᶜ ∩ B a = B a) ↔ a ≥ -1/4) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1808_180834


namespace NUMINAMATH_CALUDE_january_employee_count_l1808_180819

/-- The number of employees in January, given the December count and percentage increase --/
def january_employees (december_count : ℕ) (percent_increase : ℚ) : ℚ :=
  (december_count : ℚ) / (1 + percent_increase)

/-- Theorem stating that given the conditions, the number of employees in January is approximately 408.7 --/
theorem january_employee_count :
  let december_count : ℕ := 470
  let percent_increase : ℚ := 15 / 100
  let january_count := january_employees december_count percent_increase
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.05 ∧ |january_count - 408.7| < ε :=
sorry

end NUMINAMATH_CALUDE_january_employee_count_l1808_180819


namespace NUMINAMATH_CALUDE_train_passes_jogger_l1808_180856

/-- Prove that a train passes a jogger in 35 seconds given the specified conditions -/
theorem train_passes_jogger (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (initial_distance : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  initial_distance = 230 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 35 := by
  sorry

end NUMINAMATH_CALUDE_train_passes_jogger_l1808_180856


namespace NUMINAMATH_CALUDE_coplanar_vectors_k_value_l1808_180844

def a : ℝ × ℝ × ℝ := (1, -1, 2)
def b : ℝ × ℝ × ℝ := (-2, 1, 0)
def c (k : ℝ) : ℝ × ℝ × ℝ := (-3, 1, k)

theorem coplanar_vectors_k_value :
  ∀ k : ℝ, (∃ x y : ℝ, c k = x • a + y • b) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_coplanar_vectors_k_value_l1808_180844


namespace NUMINAMATH_CALUDE_cube_multiplication_division_equality_l1808_180809

theorem cube_multiplication_division_equality : (12 ^ 3 * 6 ^ 3) / 432 = 864 := by
  sorry

end NUMINAMATH_CALUDE_cube_multiplication_division_equality_l1808_180809


namespace NUMINAMATH_CALUDE_positive_correlation_groups_l1808_180886

structure Variable where
  name : String

structure VariableGroup where
  var1 : Variable
  var2 : Variable

def has_positive_correlation (group : VariableGroup) : Prop :=
  sorry

def selling_price : Variable := ⟨"selling price"⟩
def sales_volume : Variable := ⟨"sales volume"⟩
def id_number : Variable := ⟨"ID number"⟩
def math_score : Variable := ⟨"math score"⟩
def breakfast_eaters : Variable := ⟨"number of people who eat breakfast daily"⟩
def stomach_diseases : Variable := ⟨"number of people with stomach diseases"⟩
def temperature : Variable := ⟨"temperature"⟩
def cold_drink_sales : Variable := ⟨"cold drink sales volume"⟩
def ebike_weight : Variable := ⟨"weight of an electric bicycle"⟩
def electricity_consumption : Variable := ⟨"electricity consumption per kilometer"⟩

def group1 : VariableGroup := ⟨selling_price, sales_volume⟩
def group2 : VariableGroup := ⟨id_number, math_score⟩
def group3 : VariableGroup := ⟨breakfast_eaters, stomach_diseases⟩
def group4 : VariableGroup := ⟨temperature, cold_drink_sales⟩
def group5 : VariableGroup := ⟨ebike_weight, electricity_consumption⟩

theorem positive_correlation_groups :
  has_positive_correlation group4 ∧ 
  has_positive_correlation group5 ∧
  ¬has_positive_correlation group1 ∧
  ¬has_positive_correlation group2 ∧
  ¬has_positive_correlation group3 :=
by sorry

end NUMINAMATH_CALUDE_positive_correlation_groups_l1808_180886


namespace NUMINAMATH_CALUDE_p_and_q_true_l1808_180841

theorem p_and_q_true (P Q : Prop) (h : ¬(P ∧ Q) = False) : P ∧ Q :=
sorry

end NUMINAMATH_CALUDE_p_and_q_true_l1808_180841


namespace NUMINAMATH_CALUDE_y_axis_symmetry_l1808_180877

/-- Given a point P(2, 1), its symmetric point P' with respect to the y-axis has coordinates (-2, 1) -/
theorem y_axis_symmetry :
  let P : ℝ × ℝ := (2, 1)
  let P' : ℝ × ℝ := (-P.1, P.2)
  P' = (-2, 1) := by sorry

end NUMINAMATH_CALUDE_y_axis_symmetry_l1808_180877


namespace NUMINAMATH_CALUDE_average_income_A_and_C_l1808_180887

/-- Given the monthly incomes of individuals A, B, and C, prove that the average income of A and C is 4200. -/
theorem average_income_A_and_C (A B C : ℕ) : 
  (A + B) / 2 = 4050 →
  (B + C) / 2 = 5250 →
  A = 3000 →
  (A + C) / 2 = 4200 := by
sorry

end NUMINAMATH_CALUDE_average_income_A_and_C_l1808_180887


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1808_180855

theorem no_integer_solutions : ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1808_180855


namespace NUMINAMATH_CALUDE_sum_of_possible_m_values_l1808_180883

theorem sum_of_possible_m_values (p q r m : ℂ) : 
  p ≠ q ∧ q ≠ r ∧ r ≠ p →
  p / (1 - q) = m ∧ q / (1 - r) = m ∧ r / (1 - p) = m →
  ∃ (m₁ m₂ m₃ : ℂ), 
    (m₁ = 0 ∨ m₁ = (1 + Complex.I * Real.sqrt 3) / 2 ∨ m₁ = (1 - Complex.I * Real.sqrt 3) / 2) ∧
    (m₂ = 0 ∨ m₂ = (1 + Complex.I * Real.sqrt 3) / 2 ∨ m₂ = (1 - Complex.I * Real.sqrt 3) / 2) ∧
    (m₃ = 0 ∨ m₃ = (1 + Complex.I * Real.sqrt 3) / 2 ∨ m₃ = (1 - Complex.I * Real.sqrt 3) / 2) ∧
    m₁ + m₂ + m₃ = 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_possible_m_values_l1808_180883


namespace NUMINAMATH_CALUDE_sqrt_8_simplification_l1808_180884

theorem sqrt_8_simplification :
  Real.sqrt 8 = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_8_simplification_l1808_180884


namespace NUMINAMATH_CALUDE_chicken_admission_problem_l1808_180861

theorem chicken_admission_problem :
  let n : ℕ := 4  -- Total number of chickens
  let k : ℕ := 2  -- Number of chickens to be admitted to evening department
  Nat.choose n k = 6 := by
  sorry

end NUMINAMATH_CALUDE_chicken_admission_problem_l1808_180861


namespace NUMINAMATH_CALUDE_boat_speed_proof_l1808_180824

/-- The speed of the stream in km/h -/
def stream_speed : ℝ := 8

/-- The distance covered downstream in km -/
def downstream_distance : ℝ := 64

/-- The distance covered upstream in km -/
def upstream_distance : ℝ := 32

/-- The speed of the boat in still water in km/h -/
def boat_speed : ℝ := 24

theorem boat_speed_proof :
  (downstream_distance / (boat_speed + stream_speed) = 
   upstream_distance / (boat_speed - stream_speed)) ∧
  (boat_speed > stream_speed) :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_proof_l1808_180824


namespace NUMINAMATH_CALUDE_common_rest_days_1000_l1808_180840

/-- Represents the work-rest cycle of a person -/
structure WorkCycle where
  workDays : ℕ
  restDays : ℕ

/-- Calculates the number of common rest days for two people within a given number of days -/
def commonRestDays (cycleA cycleB : WorkCycle) (totalDays : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of common rest days for Person A and Person B -/
theorem common_rest_days_1000 :
  let cycleA := WorkCycle.mk 3 1
  let cycleB := WorkCycle.mk 7 3
  commonRestDays cycleA cycleB 1000 = 100 := by
  sorry

end NUMINAMATH_CALUDE_common_rest_days_1000_l1808_180840


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l1808_180837

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 4600

/-- The scientific notation representation of the number -/
def scientificForm : ScientificNotation :=
  { coefficient := 4.6
    exponent := 3
    property := by sorry }

/-- Theorem stating that the scientific notation form is correct -/
theorem scientific_notation_correct :
  (scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent) = number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l1808_180837


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1808_180854

theorem triangle_angle_measure (X Y Z : ℝ) : 
  Y = 30 → Z = 3 * Y → X + Y + Z = 180 → X = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1808_180854


namespace NUMINAMATH_CALUDE_rohan_investment_is_8040_l1808_180848

/-- Represents the investment scenario described in the problem -/
structure InvestmentScenario where
  suresh_investment : ℕ
  suresh_months : ℕ
  rohan_months : ℕ
  sudhir_investment : ℕ
  sudhir_months : ℕ
  total_profit : ℕ
  rohan_sudhir_diff : ℕ

/-- Calculates Rohan's investment based on the given scenario -/
def calculate_rohan_investment (scenario : InvestmentScenario) : ℕ :=
  sorry

/-- Theorem stating that Rohan's investment is 8040 given the specific scenario -/
theorem rohan_investment_is_8040 : 
  let scenario : InvestmentScenario := {
    suresh_investment := 18000,
    suresh_months := 12,
    rohan_months := 9,
    sudhir_investment := 9000,
    sudhir_months := 8,
    total_profit := 3872,
    rohan_sudhir_diff := 352
  }
  calculate_rohan_investment scenario = 8040 := by
  sorry

end NUMINAMATH_CALUDE_rohan_investment_is_8040_l1808_180848


namespace NUMINAMATH_CALUDE_rectangle_division_parts_l1808_180866

/-- The number of parts a rectangle is divided into when split into unit squares and crossed by a diagonal -/
def rectangle_parts (width : ℕ) (height : ℕ) : ℕ :=
  width * height + width + height - Nat.gcd width height

/-- Theorem stating that a 19 cm by 65 cm rectangle divided as described results in 1318 parts -/
theorem rectangle_division_parts : rectangle_parts 19 65 = 1318 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_division_parts_l1808_180866


namespace NUMINAMATH_CALUDE_max_bing_games_and_wins_l1808_180880

/-- Represents a player in the table tennis game -/
inductive Player : Type
| jia : Player
| yi : Player
| bing : Player

/-- The game state, tracking the number of games played by each player -/
structure GameState :=
  (jia_games : ℕ)
  (yi_games : ℕ)
  (bing_games : ℕ)
  (bing_wins : ℕ)

/-- Checks if the game state is valid according to the rules -/
def is_valid_state (state : GameState) : Prop :=
  state.jia_games = 10 ∧ 
  state.yi_games = 7 ∧ 
  state.bing_games ≤ state.jia_games ∧ 
  state.bing_games ≤ state.yi_games + state.bing_wins ∧
  state.bing_wins ≤ state.bing_games

/-- The main theorem to prove -/
theorem max_bing_games_and_wins :
  ∃ (state : GameState), 
    is_valid_state state ∧ 
    ∀ (other_state : GameState), 
      is_valid_state other_state → 
      other_state.bing_games ≤ state.bing_games ∧
      other_state.bing_wins ≤ state.bing_wins ∧
      state.bing_games = 13 ∧
      state.bing_wins = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_bing_games_and_wins_l1808_180880


namespace NUMINAMATH_CALUDE_at_least_one_perpendicular_l1808_180811

structure GeometricSpace where
  Plane : Type
  Line : Type
  perpendicular_planes : Plane → Plane → Prop
  line_in_plane : Line → Plane → Prop
  perpendicular_lines : Line → Line → Prop
  line_perpendicular_to_plane : Line → Plane → Prop

variable (S : GeometricSpace)

theorem at_least_one_perpendicular
  (α β : S.Plane) (a b : S.Line)
  (h1 : S.perpendicular_planes α β)
  (h2 : S.line_in_plane a α)
  (h3 : S.line_in_plane b β)
  (h4 : S.perpendicular_lines a b) :
  S.line_perpendicular_to_plane a β ∨ S.line_perpendicular_to_plane b α :=
sorry

end NUMINAMATH_CALUDE_at_least_one_perpendicular_l1808_180811


namespace NUMINAMATH_CALUDE_right_angled_triangle_l1808_180859

theorem right_angled_triangle (A B C : Real) (h1 : 0 ≤ A ∧ A ≤ π) (h2 : 0 ≤ B ∧ B ≤ π) (h3 : 0 ≤ C ∧ C ≤ π) 
  (h4 : A + B + C = π) (h5 : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 2) : 
  A = π/2 ∨ B = π/2 ∨ C = π/2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l1808_180859


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l1808_180833

theorem trader_gain_percentage (num_sold : ℕ) (num_gain : ℕ) (h1 : num_sold = 88) (h2 : num_gain = 22) :
  (num_gain : ℚ) / num_sold * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l1808_180833


namespace NUMINAMATH_CALUDE_final_green_probability_l1808_180889

/-- Represents the total number of amoeba in the dish -/
def total_amoeba : ℕ := 10

/-- Represents the initial number of green amoeba -/
def initial_green : ℕ := 7

/-- Represents the initial number of blue amoeba -/
def initial_blue : ℕ := 3

/-- Theorem stating the probability of the final amoeba being green -/
theorem final_green_probability :
  (initial_green : ℚ) / total_amoeba = 7 / 10 :=
sorry

end NUMINAMATH_CALUDE_final_green_probability_l1808_180889


namespace NUMINAMATH_CALUDE_convention_handshakes_l1808_180845

theorem convention_handshakes (num_companies num_representatives_per_company : ℕ) :
  num_companies = 4 →
  num_representatives_per_company = 4 →
  let total_people := num_companies * num_representatives_per_company
  let handshakes_per_person := total_people - num_representatives_per_company
  (total_people * handshakes_per_person) / 2 = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_l1808_180845


namespace NUMINAMATH_CALUDE_unique_room_dimensions_l1808_180831

/-- A room with integer dimensions where the unpainted border area is four times the painted area --/
structure PaintedRoom where
  a : ℕ
  b : ℕ
  h1 : 0 < a
  h2 : 0 < b
  h3 : b > a
  h4 : 4 * ((a - 4) * (b - 4)) = a * b - (a - 4) * (b - 4)

/-- The only valid dimensions for the room are 6 by 30 feet --/
theorem unique_room_dimensions : 
  ∀ (room : PaintedRoom), room.a = 6 ∧ room.b = 30 :=
by sorry

end NUMINAMATH_CALUDE_unique_room_dimensions_l1808_180831


namespace NUMINAMATH_CALUDE_painting_selection_theorem_l1808_180808

/-- The number of traditional Chinese paintings -/
def traditional_paintings : Nat := 5

/-- The number of oil paintings -/
def oil_paintings : Nat := 2

/-- The number of watercolor paintings -/
def watercolor_paintings : Nat := 7

/-- The number of ways to choose one painting from each category -/
def choose_one_each : Nat := traditional_paintings * oil_paintings * watercolor_paintings

/-- The number of ways to choose two paintings of different types -/
def choose_two_different : Nat := 
  traditional_paintings * oil_paintings + 
  traditional_paintings * watercolor_paintings + 
  oil_paintings * watercolor_paintings

theorem painting_selection_theorem : 
  (choose_one_each = 70) ∧ (choose_two_different = 59) := by
  sorry

end NUMINAMATH_CALUDE_painting_selection_theorem_l1808_180808


namespace NUMINAMATH_CALUDE_circle_point_selection_eq_258_l1808_180835

/-- The number of ways to select 8 points from 24 equally spaced points on a circle,
    such that no two selected points have an arc length of 3 or 8 between them. -/
def circle_point_selection : ℕ :=
  2^8 + 2

/-- Proves that the number of valid selections is 258. -/
theorem circle_point_selection_eq_258 : circle_point_selection = 258 := by
  sorry

end NUMINAMATH_CALUDE_circle_point_selection_eq_258_l1808_180835


namespace NUMINAMATH_CALUDE_sugar_price_increase_l1808_180888

theorem sugar_price_increase (original_price : ℝ) (consumption_reduction : ℝ) (new_price : ℝ) : 
  original_price = 6 →
  consumption_reduction = 19.999999999999996 →
  (1 - consumption_reduction / 100) * new_price = original_price →
  new_price = 7.5 := by
sorry

end NUMINAMATH_CALUDE_sugar_price_increase_l1808_180888


namespace NUMINAMATH_CALUDE_division_remainder_l1808_180895

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 23 →
  divisor = 4 →
  quotient = 5 →
  dividend = divisor * quotient + remainder →
  remainder = 3 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l1808_180895


namespace NUMINAMATH_CALUDE_games_spent_proof_l1808_180843

def total_allowance : ℚ := 50

def books_fraction : ℚ := 1/4
def apps_fraction : ℚ := 3/10
def snacks_fraction : ℚ := 1/5

def books_spent : ℚ := total_allowance * books_fraction
def apps_spent : ℚ := total_allowance * apps_fraction
def snacks_spent : ℚ := total_allowance * snacks_fraction

def other_expenses : ℚ := books_spent + apps_spent + snacks_spent

theorem games_spent_proof : total_allowance - other_expenses = 25/2 := by sorry

end NUMINAMATH_CALUDE_games_spent_proof_l1808_180843


namespace NUMINAMATH_CALUDE_prob_less_than_8_prob_at_least_7_l1808_180874

-- Define the probabilities
def p_9_or_above : ℝ := 0.56
def p_8 : ℝ := 0.22
def p_7 : ℝ := 0.12

-- Theorem for the first question
theorem prob_less_than_8 : 1 - p_9_or_above - p_8 = 0.22 := by sorry

-- Theorem for the second question
theorem prob_at_least_7 : p_9_or_above + p_8 + p_7 = 0.9 := by sorry

end NUMINAMATH_CALUDE_prob_less_than_8_prob_at_least_7_l1808_180874


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l1808_180816

theorem square_root_of_sixteen : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l1808_180816


namespace NUMINAMATH_CALUDE_minor_premise_identification_l1808_180832

-- Define the basic propositions
def ship_departs_on_time : Prop := sorry
def ship_arrives_on_time : Prop := sorry

-- Define the syllogism structure
structure Syllogism :=
  (major_premise : Prop)
  (minor_premise : Prop)
  (conclusion : Prop)

-- Define our specific syllogism
def our_syllogism : Syllogism :=
  { major_premise := ship_departs_on_time → ship_arrives_on_time,
    minor_premise := ship_arrives_on_time,
    conclusion := ship_departs_on_time }

-- Theorem to prove
theorem minor_premise_identification :
  our_syllogism.minor_premise = ship_arrives_on_time :=
by sorry

end NUMINAMATH_CALUDE_minor_premise_identification_l1808_180832


namespace NUMINAMATH_CALUDE_circle_diameter_problem_l1808_180867

theorem circle_diameter_problem (B A C : Real) :
  B = 20 → -- Diameter of circle B is 20 cm
  A = C → -- Circles A and C have the same diameter
  (π * (B/2)^2 - 2 * π * (A/2)^2) / (π * (A/2)^2) = 5 → -- Ratio of shaded area to area of A is 5:1
  A = 2 * Real.sqrt (100/7) :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_problem_l1808_180867


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l1808_180865

/-- Given two similar right triangles, where the first triangle has one leg of 24 meters
    and a hypotenuse of 25 meters, and the second triangle has a hypotenuse of 100 meters,
    the length of the other leg of the second triangle is 28 meters. -/
theorem similar_triangles_leg_length (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a^2 + 24^2 = 25^2 →
  c^2 + d^2 = 100^2 →
  a / 25 = c / 100 →
  24 / 25 = d / 100 →
  d = 28 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l1808_180865


namespace NUMINAMATH_CALUDE_arithmetic_sequence_divisibility_l1808_180885

-- Define the arithmetic sequences
def arithmetic_seq (a₁ d : ℕ+) : ℕ → ℕ
  | n => a₁ + (n - 1) * d

theorem arithmetic_sequence_divisibility 
  (a₁ d_a b₁ d_b : ℕ+) 
  (h : ∃ (S : Set (ℕ × ℕ)), S.Infinite ∧ 
    ∀ (i j : ℕ), (i, j) ∈ S → 
      i ≤ j ∧ j ≤ i + 2021 ∧ 
      (arithmetic_seq a₁ d_a i) ∣ (arithmetic_seq b₁ d_b j)) :
  ∀ i : ℕ, ∃ j : ℕ, (arithmetic_seq a₁ d_a i) ∣ (arithmetic_seq b₁ d_b j) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_divisibility_l1808_180885


namespace NUMINAMATH_CALUDE_final_concentration_calculation_l1808_180862

/-- Calculates the final concentration of a hydrochloric acid solution after draining and adding a new solution. -/
theorem final_concentration_calculation
  (initial_amount : ℝ)
  (initial_concentration : ℝ)
  (drained_amount : ℝ)
  (added_concentration : ℝ)
  (h1 : initial_amount = 300)
  (h2 : initial_concentration = 0.20)
  (h3 : drained_amount = 25)
  (h4 : added_concentration = 0.80) :
  let initial_acid := initial_amount * initial_concentration
  let removed_acid := drained_amount * initial_concentration
  let added_acid := drained_amount * added_concentration
  let final_acid := initial_acid - removed_acid + added_acid
  let final_amount := initial_amount
  final_acid / final_amount = 0.25 := by sorry

end NUMINAMATH_CALUDE_final_concentration_calculation_l1808_180862


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_l1808_180894

theorem power_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_square_l1808_180894


namespace NUMINAMATH_CALUDE_greatest_five_digit_multiple_of_6_l1808_180864

def is_multiple_of_6 (n : ℕ) : Prop := n % 6 = 0

def digits_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def uses_digits (n : ℕ) (digits : List ℕ) : Prop :=
  (n.digits 10).toFinset = digits.toFinset

theorem greatest_five_digit_multiple_of_6 :
  ∃ (n : ℕ),
    n ≥ 10000 ∧
    n < 100000 ∧
    is_multiple_of_6 n ∧
    uses_digits n [2, 5, 6, 8, 9] ∧
    ∀ (m : ℕ),
      m ≥ 10000 →
      m < 100000 →
      is_multiple_of_6 m →
      uses_digits m [2, 5, 6, 8, 9] →
      m ≤ n ∧
    n = 98652 :=
  sorry

end NUMINAMATH_CALUDE_greatest_five_digit_multiple_of_6_l1808_180864


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l1808_180863

theorem coefficient_x_cubed_in_binomial_expansion :
  let n : ℕ := 10
  let k : ℕ := 3
  let a : ℤ := 1
  let b : ℤ := -2
  (Nat.choose n k) * b^k * a^(n-k) = -960 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_binomial_expansion_l1808_180863


namespace NUMINAMATH_CALUDE_flea_misses_point_l1808_180858

/-- Represents the number of points on the circle. -/
def n : ℕ := 101

/-- Represents the position of the flea after k jumps. -/
def flea_position (k : ℕ) : ℕ := (k * (k + 1) / 2) % n

/-- States that there exists a point that the flea never lands on. -/
theorem flea_misses_point : ∃ p : Fin n, ∀ k : ℕ, flea_position k ≠ p.val :=
sorry

end NUMINAMATH_CALUDE_flea_misses_point_l1808_180858


namespace NUMINAMATH_CALUDE_product_of_invertible_labels_l1808_180810

def is_invertible (f : ℕ → Bool) := f 2 = false ∧ f 3 = true ∧ f 4 = true ∧ f 5 = true

theorem product_of_invertible_labels (f : ℕ → Bool) (h : is_invertible f) :
  (List.filter (λ i => f i) [2, 3, 4, 5]).prod = 60 :=
by sorry

end NUMINAMATH_CALUDE_product_of_invertible_labels_l1808_180810


namespace NUMINAMATH_CALUDE_min_baking_time_three_cakes_l1808_180827

/-- Represents a cake that needs to be baked on both sides -/
structure Cake where
  side1_baked : Bool
  side2_baked : Bool

/-- Represents a baking pan that can hold up to two cakes -/
structure Pan where
  capacity : Nat
  current_cakes : List Cake

/-- The time it takes to bake one side of a cake -/
def bake_time : Nat := 1

/-- The function to calculate the minimum baking time for all cakes -/
def min_baking_time (cakes : List Cake) (pan : Pan) : Nat :=
  sorry

/-- Theorem stating that the minimum baking time for three cakes is 3 minutes -/
theorem min_baking_time_three_cakes :
  let cakes := [Cake.mk false false, Cake.mk false false, Cake.mk false false]
  let pan := Pan.mk 2 []
  min_baking_time cakes pan = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_baking_time_three_cakes_l1808_180827


namespace NUMINAMATH_CALUDE_room_width_is_seven_l1808_180822

/-- The width of a room with specific dimensions and features. -/
def room_width : ℝ :=
  let room_length : ℝ := 10
  let room_height : ℝ := 5
  let door_width : ℝ := 1
  let door_height : ℝ := 3
  let num_doors : ℕ := 2
  let large_window_width : ℝ := 2
  let large_window_height : ℝ := 1.5
  let num_large_windows : ℕ := 1
  let small_window_width : ℝ := 1
  let small_window_height : ℝ := 1.5
  let num_small_windows : ℕ := 2
  let paint_cost_per_sqm : ℝ := 3
  let total_paint_cost : ℝ := 474

  7 -- The actual width value

/-- Theorem stating that the room width is 7 meters. -/
theorem room_width_is_seven : room_width = 7 := by
  sorry

end NUMINAMATH_CALUDE_room_width_is_seven_l1808_180822


namespace NUMINAMATH_CALUDE_cold_water_time_l1808_180818

/-- The combined total time Jerry and his friends spent in the cold water pool --/
def total_time (jerry_time elaine_time george_time kramer_time : ℝ) : ℝ :=
  jerry_time + elaine_time + george_time + kramer_time

/-- Theorem stating the total time spent in the cold water pool --/
theorem cold_water_time : ∃ (jerry_time elaine_time george_time kramer_time : ℝ),
  jerry_time = 3 ∧
  elaine_time = 2 * jerry_time ∧
  george_time = (1/3) * elaine_time ∧
  kramer_time = 0 ∧
  total_time jerry_time elaine_time george_time kramer_time = 11 := by
  sorry

end NUMINAMATH_CALUDE_cold_water_time_l1808_180818


namespace NUMINAMATH_CALUDE_remainder_3042_div_98_l1808_180899

theorem remainder_3042_div_98 : 3042 % 98 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3042_div_98_l1808_180899


namespace NUMINAMATH_CALUDE_jimin_has_most_candy_left_l1808_180873

def jimin_fraction : ℚ := 1/9
def taehyung_fraction : ℚ := 1/3
def hoseok_fraction : ℚ := 1/6

theorem jimin_has_most_candy_left : 
  jimin_fraction < taehyung_fraction ∧ 
  jimin_fraction < hoseok_fraction :=
by sorry

end NUMINAMATH_CALUDE_jimin_has_most_candy_left_l1808_180873


namespace NUMINAMATH_CALUDE_sin_power_five_expansion_l1808_180898

theorem sin_power_five_expansion (b₁ b₂ b₃ b₄ b₅ : ℝ) :
  (∀ θ : ℝ, Real.sin θ ^ 5 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + 
    b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ) + b₅ * Real.sin (5 * θ)) →
  b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 = 63 / 512 := by
  sorry

end NUMINAMATH_CALUDE_sin_power_five_expansion_l1808_180898


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1808_180869

/-- 
If the quadratic equation x^2 + 6x + c = 0 has two equal real roots,
then c = 9.
-/
theorem equal_roots_quadratic (c : ℝ) : 
  (∃ x : ℝ, x^2 + 6*x + c = 0 ∧ 
   ∀ y : ℝ, y^2 + 6*y + c = 0 → y = x) → 
  c = 9 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1808_180869


namespace NUMINAMATH_CALUDE_no_four_integers_with_odd_sum_and_product_l1808_180804

theorem no_four_integers_with_odd_sum_and_product : ¬∃ (a b c d : ℤ), 
  Odd (a + b + c + d) ∧ Odd (a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_no_four_integers_with_odd_sum_and_product_l1808_180804


namespace NUMINAMATH_CALUDE_range_of_a_l1808_180825

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 5}

def B (a : ℝ) : Set ℝ := {x : ℝ | (x - a + 1) * (x - a - 1) ≤ 0}

def p (x : ℝ) : Prop := x ∈ A

def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

theorem range_of_a : 
  {a : ℝ | (∀ x, q a x → p x) ∧ (∃ x, p x ∧ ¬q a x)} = {a : ℝ | 2 ≤ a ∧ a ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1808_180825


namespace NUMINAMATH_CALUDE_largest_n_for_square_sum_l1808_180881

theorem largest_n_for_square_sum : ∃ (m : ℕ), 
  (4^995 + 4^1500 + 4^2004 = m^2) ∧ 
  (∀ (k : ℕ), k > 2004 → ¬∃ (l : ℕ), 4^995 + 4^1500 + 4^k = l^2) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_square_sum_l1808_180881


namespace NUMINAMATH_CALUDE_g_of_3_eq_6_l1808_180802

/-- The function g(x) = x^3 - 3x^2 + 2x -/
def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

/-- Theorem: The value of g(3) is 6 -/
theorem g_of_3_eq_6 : g 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_6_l1808_180802


namespace NUMINAMATH_CALUDE_quarter_probability_l1808_180817

/-- The probability of choosing a quarter from a jar containing quarters, nickels, and pennies -/
theorem quarter_probability (quarter_value nickel_value penny_value : ℚ)
  (total_quarter_value total_nickel_value total_penny_value : ℚ)
  (h_quarter : quarter_value = 25/100)
  (h_nickel : nickel_value = 5/100)
  (h_penny : penny_value = 1/100)
  (h_total_quarter : total_quarter_value = 15/2)
  (h_total_nickel : total_nickel_value = 25/2)
  (h_total_penny : total_penny_value = 15) :
  (total_quarter_value / quarter_value) / 
  ((total_quarter_value / quarter_value) + 
   (total_nickel_value / nickel_value) + 
   (total_penny_value / penny_value)) = 15/890 := by
  sorry


end NUMINAMATH_CALUDE_quarter_probability_l1808_180817


namespace NUMINAMATH_CALUDE_max_rectangular_pen_area_l1808_180849

/-- Given 60 feet of fencing, the maximum area of a rectangular pen is 225 square feet. -/
theorem max_rectangular_pen_area :
  ∀ w h : ℝ,
  w > 0 → h > 0 →
  2 * w + 2 * h = 60 →
  w * h ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangular_pen_area_l1808_180849


namespace NUMINAMATH_CALUDE_childrens_admission_fee_l1808_180896

/-- Proves that the children's admission fee is $1.50 given the problem conditions -/
theorem childrens_admission_fee (total_people : ℕ) (total_fees : ℚ) (num_children : ℕ) (adult_fee : ℚ) :
  total_people = 315 →
  total_fees = 810 →
  num_children = 180 →
  adult_fee = 4 →
  ∃ (child_fee : ℚ),
    child_fee * num_children + adult_fee * (total_people - num_children) = total_fees ∧
    child_fee = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_childrens_admission_fee_l1808_180896


namespace NUMINAMATH_CALUDE_equation_solution_l1808_180826

theorem equation_solution : ∃! x : ℝ, (3 / (x - 2) = 6 / (x - 3)) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1808_180826


namespace NUMINAMATH_CALUDE_flow_rate_reduction_l1808_180801

def original_flow_rate : ℝ := 5.0

def reduced_flow_rate (original : ℝ) : ℝ := 0.6 * original - 1

theorem flow_rate_reduction : reduced_flow_rate original_flow_rate = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_flow_rate_reduction_l1808_180801


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_for_x_geq_1_l1808_180878

theorem x_eq_2_sufficient_not_necessary_for_x_geq_1 :
  (∀ x : ℝ, x = 2 → x ≥ 1) ∧ ¬(∀ x : ℝ, x ≥ 1 → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_for_x_geq_1_l1808_180878


namespace NUMINAMATH_CALUDE_division_chain_l1808_180892

theorem division_chain : (180 / 6) / 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_division_chain_l1808_180892


namespace NUMINAMATH_CALUDE_polynomial_expansion_theorem_l1808_180882

theorem polynomial_expansion_theorem (N : ℕ) : 
  (Nat.choose N 5 = 2002) ↔ (N = 17) := by sorry

#check polynomial_expansion_theorem

end NUMINAMATH_CALUDE_polynomial_expansion_theorem_l1808_180882


namespace NUMINAMATH_CALUDE_ratio_satisfies_conditions_l1808_180830

/-- Represents the number of people in each profession --/
structure ProfessionCount where
  doctors : ℕ
  lawyers : ℕ
  engineers : ℕ

/-- Checks if the given counts satisfy the average age conditions --/
def satisfiesAverageConditions (count : ProfessionCount) : Prop :=
  let totalPeople := count.doctors + count.lawyers + count.engineers
  let totalAge := 40 * count.doctors + 50 * count.lawyers + 60 * count.engineers
  totalAge / totalPeople = 45

/-- The theorem stating that the ratio 3:6:1 satisfies the conditions --/
theorem ratio_satisfies_conditions :
  ∃ (k : ℕ), k > 0 ∧ 
    let count : ProfessionCount := ⟨3*k, 6*k, k⟩
    satisfiesAverageConditions count :=
sorry

end NUMINAMATH_CALUDE_ratio_satisfies_conditions_l1808_180830


namespace NUMINAMATH_CALUDE_not_always_perfect_square_l1808_180853

theorem not_always_perfect_square (d : ℕ) (h1 : d > 0) (h2 : d ≠ 2) (h3 : d ≠ 5) (h4 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧
  ¬∃ (k : ℕ), a * b - 1 = k * k :=
by sorry

end NUMINAMATH_CALUDE_not_always_perfect_square_l1808_180853


namespace NUMINAMATH_CALUDE_square_sequence_theorem_l1808_180879

/-- The number of nonoverlapping unit squares in the nth figure -/
def f (n : ℕ) : ℕ := 2 * n^2 + 4 * n + 3

/-- The theorem stating the properties of the sequence and the value for the 100th figure -/
theorem square_sequence_theorem :
  (f 0 = 3) ∧ (f 1 = 9) ∧ (f 2 = 19) ∧ (f 3 = 33) → f 100 = 20403 :=
by
  sorry

end NUMINAMATH_CALUDE_square_sequence_theorem_l1808_180879


namespace NUMINAMATH_CALUDE_units_digit_not_four_l1808_180821

/-- The set of numbers from which a and b are chosen -/
def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

/-- The units digit of a number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem units_digit_not_four (a b : ℕ) (ha : a ∈ S) (hb : b ∈ S) :
  unitsDigit (2^a + 5^b) ≠ 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_not_four_l1808_180821


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1808_180857

theorem consecutive_integers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 2800 → n + (n + 1) = 105 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1808_180857


namespace NUMINAMATH_CALUDE_missy_watch_time_l1808_180839

/-- The total time Missy spends watching TV, given the number of reality shows,
    the duration of each reality show, and the duration of the cartoon. -/
def total_watch_time (num_reality_shows : ℕ) (reality_show_duration : ℕ) (cartoon_duration : ℕ) : ℕ :=
  num_reality_shows * reality_show_duration + cartoon_duration

/-- Theorem stating that Missy spends 150 minutes watching TV. -/
theorem missy_watch_time :
  total_watch_time 5 28 10 = 150 := by
  sorry

end NUMINAMATH_CALUDE_missy_watch_time_l1808_180839


namespace NUMINAMATH_CALUDE_solution_of_equation_l1808_180823

theorem solution_of_equation : 
  ∃ x : ℝ, 7 * (2 * x - 3) + 4 = -3 * (2 - 5 * x) ∧ x = -11 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l1808_180823


namespace NUMINAMATH_CALUDE_arithmetic_series_first_term_l1808_180891

-- Define an arithmetic series
def ArithmeticSeries (a₁ : ℚ) (d : ℚ) : ℕ → ℚ := fun n ↦ a₁ + (n - 1 : ℚ) * d

-- Sum of first n terms of an arithmetic series
def SumArithmeticSeries (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem arithmetic_series_first_term
  (a₁ : ℚ)
  (d : ℚ)
  (h1 : SumArithmeticSeries a₁ d 60 = 240)
  (h2 : SumArithmeticSeries (ArithmeticSeries a₁ d 61) d 60 = 3600) :
  a₁ = -353/15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_first_term_l1808_180891


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l1808_180875

/-- The number of ways to arrange 2 teachers and 4 students in a row -/
def arrangementCount (n : ℕ) (m : ℕ) (k : ℕ) : ℕ :=
  if n = 2 ∧ m = 4 ∧ k = 1 then
    Nat.factorial 2 * 2 * Nat.factorial 3
  else
    0

/-- Theorem stating the correct number of arrangements -/
theorem photo_arrangement_count :
  arrangementCount 2 4 1 = 24 :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l1808_180875


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1808_180828

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x₁ x₂, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ s, s = x₁ + x₂ ∧ s = -b / a) :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 + 2010*x - (2011 + 18*x)
  (∃ x₁ x₂, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ s, s = x₁ + x₂ ∧ s = -1992) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1808_180828


namespace NUMINAMATH_CALUDE_cube_equation_solution_l1808_180876

theorem cube_equation_solution : ∃! x : ℝ, (x - 3)^3 = 27 ∧ x = 6 := by sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l1808_180876


namespace NUMINAMATH_CALUDE_upper_limit_of_multiples_l1808_180846

def average_of_multiples_of_10 (n : ℕ) : ℚ :=
  (n * (10 + n)) / (2 * n)

theorem upper_limit_of_multiples (n : ℕ) :
  n ≥ 10 → average_of_multiples_of_10 n = 55 → n = 100 := by
  sorry

end NUMINAMATH_CALUDE_upper_limit_of_multiples_l1808_180846


namespace NUMINAMATH_CALUDE_inequality_solution_l1808_180800

theorem inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 → (∀ x, a^(2*x-1) > (1/a)^(x-2) ↔ x > 1)) ∧
  (a < 1 → (∀ x, a^(2*x-1) > (1/a)^(x-2) ↔ x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1808_180800


namespace NUMINAMATH_CALUDE_prob_at_least_one_three_l1808_180870

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_sides * num_sides

/-- The number of outcomes where neither die shows the target number -/
def non_target_outcomes : ℕ := (num_sides - 1) * (num_sides - 1)

/-- The probability of at least one die showing the target number -/
def prob_at_least_one_target : ℚ := (total_outcomes - non_target_outcomes) / total_outcomes

theorem prob_at_least_one_three :
  prob_at_least_one_target = 15 / 64 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_three_l1808_180870


namespace NUMINAMATH_CALUDE_opposite_numbers_l1808_180847

-- Definition of opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Theorem to prove
theorem opposite_numbers : are_opposite (-|(-6)|) (-(-6)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_l1808_180847


namespace NUMINAMATH_CALUDE_march_greatest_drop_l1808_180820

/-- Represents the months of the year --/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June
  | July
  | August

/-- Represents the price change for each month --/
def price_change : Month → ℝ
  | Month.January  => -0.75
  | Month.February => 1.50
  | Month.March    => -3.00
  | Month.April    => 2.50
  | Month.May      => -0.25
  | Month.June     => 0.80
  | Month.July     => -2.75
  | Month.August   => -1.20

/-- Determines if a given month has the greatest price drop --/
def has_greatest_drop (m : Month) : Prop :=
  ∀ other : Month, price_change m ≤ price_change other

/-- Theorem stating that March has the greatest price drop --/
theorem march_greatest_drop : has_greatest_drop Month.March :=
sorry

end NUMINAMATH_CALUDE_march_greatest_drop_l1808_180820


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_union_equals_reals_iff_l1808_180852

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a : ℝ) : Set ℝ := {x | x - 4*a ≤ 0}

-- Part I
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x : ℝ | x < -1 ∨ (3 < x ∧ x ≤ 4)} := by sorry

-- Part II
theorem union_equals_reals_iff (a : ℝ) :
  A ∪ B a = Set.univ ↔ a ≥ 3/4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_union_equals_reals_iff_l1808_180852


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l1808_180813

theorem square_root_of_sixteen : 
  ∃ (x : ℝ), x^2 = 16 ∧ (x = 4 ∨ x = -4) :=
sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l1808_180813


namespace NUMINAMATH_CALUDE_three_step_to_one_eleven_step_to_one_l1808_180803

def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 1

def reaches_one_in (n : ℕ) (steps : ℕ) : Prop :=
  ∃ (sequence : ℕ → ℕ), 
    sequence 0 = n ∧
    sequence steps = 1 ∧
    ∀ i < steps, sequence (i + 1) = operation (sequence i)

theorem three_step_to_one :
  ∃! (s : Finset ℕ), 
    s.card = 3 ∧ 
    ∀ n, n ∈ s ↔ reaches_one_in n 3 :=
sorry

theorem eleven_step_to_one :
  ∃! (s : Finset ℕ), 
    s.card = 3 ∧ 
    ∀ n, n ∈ s ↔ reaches_one_in n 11 :=
sorry

end NUMINAMATH_CALUDE_three_step_to_one_eleven_step_to_one_l1808_180803
