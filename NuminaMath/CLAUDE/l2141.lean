import Mathlib

namespace NUMINAMATH_CALUDE_monthly_fee_plan_a_correct_l2141_214184

/-- The monthly fee for Plan A in a cell phone company's text-messaging plans. -/
def monthly_fee_plan_a : ℝ := 9

/-- The cost per text message for Plan A. -/
def cost_per_text_plan_a : ℝ := 0.25

/-- The cost per text message for Plan B. -/
def cost_per_text_plan_b : ℝ := 0.40

/-- The number of text messages at which both plans cost the same. -/
def equal_cost_messages : ℕ := 60

/-- Theorem stating that the monthly fee for Plan A is correct. -/
theorem monthly_fee_plan_a_correct :
  monthly_fee_plan_a = 
    equal_cost_messages * (cost_per_text_plan_b - cost_per_text_plan_a) :=
by sorry

end NUMINAMATH_CALUDE_monthly_fee_plan_a_correct_l2141_214184


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2141_214145

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-2, -1, 1, 2}

theorem complement_of_A_in_U : 
  {x : Int | x ∈ U ∧ x ∉ A} = {0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2141_214145


namespace NUMINAMATH_CALUDE_evaluate_expression_l2141_214120

theorem evaluate_expression : (10^8) / (2 * (10^5) * (1/2)) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2141_214120


namespace NUMINAMATH_CALUDE_A_power_difference_l2141_214131

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 0, 1]

theorem A_power_difference :
  A^20 - 2 • A^19 = !![0, 3; 0, -1] := by sorry

end NUMINAMATH_CALUDE_A_power_difference_l2141_214131


namespace NUMINAMATH_CALUDE_two_numbers_with_sum_or_diff_divisible_by_1000_l2141_214132

theorem two_numbers_with_sum_or_diff_divisible_by_1000 (S : Finset ℕ) (h : S.card = 502) :
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (1000 ∣ (a - b) ∨ 1000 ∣ (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_sum_or_diff_divisible_by_1000_l2141_214132


namespace NUMINAMATH_CALUDE_londolozi_lion_cubs_per_month_l2141_214196

/-- The number of lion cubs born per month in Londolozi -/
def lion_cubs_per_month (initial_population final_population : ℕ) (months death_rate : ℕ) : ℕ :=
  (final_population - initial_population + months * death_rate) / months

/-- Theorem stating the number of lion cubs born per month in Londolozi -/
theorem londolozi_lion_cubs_per_month :
  lion_cubs_per_month 100 148 12 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_londolozi_lion_cubs_per_month_l2141_214196


namespace NUMINAMATH_CALUDE_circle_center_correct_l2141_214117

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, return its center -/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 4 8 4 (-24) 16
  let center := findCircleCenter eq
  center.x = -1 ∧ center.y = 3 := by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l2141_214117


namespace NUMINAMATH_CALUDE_sequence_bound_l2141_214106

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ i : ℕ, i > 0 → 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j : ℕ, i > 0 → j > 0 → i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_bound_l2141_214106


namespace NUMINAMATH_CALUDE_power_of_negative_power_l2141_214198

theorem power_of_negative_power (a : ℝ) : (-a^5)^2 = a^10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_power_l2141_214198


namespace NUMINAMATH_CALUDE_judys_score_is_25_l2141_214156

/-- Represents the scoring system for a math competition -/
structure ScoringSystem where
  correctPoints : Int
  incorrectPoints : Int

/-- Represents a participant's answers in the competition -/
structure Answers where
  total : Nat
  correct : Nat
  incorrect : Nat
  unanswered : Nat

/-- Calculates the score based on the scoring system and answers -/
def calculateScore (system : ScoringSystem) (answers : Answers) : Int :=
  system.correctPoints * answers.correct + system.incorrectPoints * answers.incorrect

/-- Theorem: Judy's score in the math competition is 25 points -/
theorem judys_score_is_25 (system : ScoringSystem) (answers : Answers) :
  system.correctPoints = 2 →
  system.incorrectPoints = -1 →
  answers.total = 30 →
  answers.correct = 15 →
  answers.incorrect = 5 →
  answers.unanswered = 10 →
  calculateScore system answers = 25 := by
  sorry

#eval calculateScore { correctPoints := 2, incorrectPoints := -1 }
                     { total := 30, correct := 15, incorrect := 5, unanswered := 10 }

end NUMINAMATH_CALUDE_judys_score_is_25_l2141_214156


namespace NUMINAMATH_CALUDE_sum_of_logarithms_l2141_214192

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem sum_of_logarithms (a b : ℝ) (ha : (10 : ℝ) ^ a = 2) (hb : b = log10 5) :
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_logarithms_l2141_214192


namespace NUMINAMATH_CALUDE_modified_cube_edge_count_l2141_214193

/-- Represents a modified cube with smaller cubes removed from corners and sliced by a plane -/
structure ModifiedCube where
  originalSideLength : ℕ
  removedCubeSideLength : ℕ
  slicePlane : Bool

/-- Calculates the number of edges in the modified cube -/
def edgeCount (cube : ModifiedCube) : ℕ :=
  sorry

/-- Theorem stating that a cube with side length 5, smaller cubes of side length 2 removed from corners,
    and sliced by a plane, has 40 edges -/
theorem modified_cube_edge_count :
  let cube : ModifiedCube := {
    originalSideLength := 5,
    removedCubeSideLength := 2,
    slicePlane := true
  }
  edgeCount cube = 40 := by sorry

end NUMINAMATH_CALUDE_modified_cube_edge_count_l2141_214193


namespace NUMINAMATH_CALUDE_tangent_plane_equation_l2141_214139

-- Define the function f(x, y)
def f (x y : ℝ) : ℝ := x^2 + y^2 + 2*x + 1

-- Define the point A
def A : ℝ × ℝ := (2, 3)

-- Theorem statement
theorem tangent_plane_equation :
  let (x₀, y₀) := A
  let z₀ := f x₀ y₀
  let fx := (2 * x₀ + 2 : ℝ)  -- Partial derivative with respect to x
  let fy := (2 * y₀ : ℝ)      -- Partial derivative with respect to y
  ∀ x y z, z - z₀ = fx * (x - x₀) + fy * (y - y₀) ↔ 6*x + 6*y - z - 12 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_plane_equation_l2141_214139


namespace NUMINAMATH_CALUDE_partnership_gain_is_18000_l2141_214183

/-- Represents the annual gain of a partnership given the investments and one partner's share. -/
def partnership_annual_gain (x : ℚ) (a_share : ℚ) : ℚ :=
  let a_invest_time : ℚ := x * 12
  let b_invest_time : ℚ := 2 * x * 6
  let c_invest_time : ℚ := 3 * x * 4
  let total_invest_time : ℚ := a_invest_time + b_invest_time + c_invest_time
  (total_invest_time / a_invest_time) * a_share

/-- 
Given:
- A invests x at the beginning
- B invests 2x after 6 months
- C invests 3x after 8 months
- A's share of the gain is 6000

Prove that the total annual gain of the partnership is 18000.
-/
theorem partnership_gain_is_18000 (x : ℚ) (h : x > 0) :
  partnership_annual_gain x 6000 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_gain_is_18000_l2141_214183


namespace NUMINAMATH_CALUDE_min_value_theorem_l2141_214135

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2)^2) + (y^2 / (x - 2)^2) ≥ 10 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 2 ∧ y₀ > 2 ∧ (x₀^2 / (y₀ - 2)^2) + (y₀^2 / (x₀ - 2)^2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2141_214135


namespace NUMINAMATH_CALUDE_amount_after_two_years_l2141_214127

theorem amount_after_two_years (initial_amount : ℝ) (increase_ratio : ℝ) :
  initial_amount = 70400 →
  increase_ratio = 1 / 8 →
  initial_amount * (1 + increase_ratio)^2 = 89070 :=
by sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l2141_214127


namespace NUMINAMATH_CALUDE_division_problem_l2141_214149

theorem division_problem (a b q : ℕ) (h1 : a - b = 1365) (h2 : a = 1637) (h3 : a = b * q + 5) : q = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2141_214149


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_le_e_plus_one_zeros_product_lt_one_l2141_214169

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

theorem f_nonnegative_iff_a_le_e_plus_one (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
sorry

theorem zeros_product_lt_one (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → f a x₁ = 0 → f a x₂ = 0 → x₁ * x₂ < 1 :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_le_e_plus_one_zeros_product_lt_one_l2141_214169


namespace NUMINAMATH_CALUDE_diesel_tank_capacity_l2141_214171

/-- Given the cost of a certain volume of diesel fuel and the cost of a full tank,
    calculate the capacity of the tank in liters. -/
theorem diesel_tank_capacity 
  (fuel_volume : ℝ) 
  (fuel_cost : ℝ) 
  (full_tank_cost : ℝ) 
  (h1 : fuel_volume = 36) 
  (h2 : fuel_cost = 18) 
  (h3 : full_tank_cost = 32) : 
  (full_tank_cost / (fuel_cost / fuel_volume)) = 64 := by
  sorry

#check diesel_tank_capacity

end NUMINAMATH_CALUDE_diesel_tank_capacity_l2141_214171


namespace NUMINAMATH_CALUDE_cos_240_degrees_l2141_214141

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l2141_214141


namespace NUMINAMATH_CALUDE_fruit_distribution_l2141_214122

theorem fruit_distribution (total_strawberries total_grapes : ℕ) 
  (leftover_strawberries leftover_grapes : ℕ) :
  total_strawberries = 66 →
  total_grapes = 49 →
  leftover_strawberries = 6 →
  leftover_grapes = 4 →
  ∃ (B : ℕ), 
    B > 0 ∧
    (total_strawberries - leftover_strawberries) % B = 0 ∧
    (total_grapes - leftover_grapes) % B = 0 ∧
    B = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_distribution_l2141_214122


namespace NUMINAMATH_CALUDE_expression_never_equals_33_l2141_214168

theorem expression_never_equals_33 (x y : ℤ) :
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_expression_never_equals_33_l2141_214168


namespace NUMINAMATH_CALUDE_divisibility_criterion_l2141_214100

theorem divisibility_criterion (p : ℕ) (hp : Nat.Prime p) :
  (∀ x y : ℕ, x > 0 → y > 0 → p ∣ (x + y)^19 - x^19 - y^19) ↔ p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19 :=
sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l2141_214100


namespace NUMINAMATH_CALUDE_unique_triple_solution_l2141_214159

theorem unique_triple_solution (x y p : ℕ+) (h_prime : Nat.Prime p) 
  (h_p : p = x^2 + 1) (h_y : 2*p^2 = y^2 + 1) : 
  (x = 2 ∧ y = 7 ∧ p = 5) := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l2141_214159


namespace NUMINAMATH_CALUDE_college_class_period_length_l2141_214150

/-- Given a total time, number of periods, and time between periods, 
    calculate the length of each period. -/
def period_length (total_time : ℕ) (num_periods : ℕ) (time_between : ℕ) : ℕ :=
  (total_time - (num_periods - 1) * time_between) / num_periods

/-- Theorem stating that under the given conditions, each period is 40 minutes long. -/
theorem college_class_period_length : 
  period_length 220 5 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_college_class_period_length_l2141_214150


namespace NUMINAMATH_CALUDE_max_d_value_l2141_214138

def is_valid_number (d e : ℕ) : Prop :=
  d < 10 ∧ e < 10 ∧ 707340 + 10 * d + 4 * e ≥ 100000 ∧ 707340 + 10 * d + 4 * e < 1000000

def is_multiple_of_34 (d e : ℕ) : Prop :=
  (707340 + 10 * d + 4 * e) % 34 = 0

theorem max_d_value (d e : ℕ) :
  is_valid_number d e → is_multiple_of_34 d e → d ≤ 13 :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l2141_214138


namespace NUMINAMATH_CALUDE_eight_paths_A_to_C_l2141_214148

/-- Represents a simple directed graph with four nodes -/
structure DirectedGraph :=
  (paths_A_to_B : ℕ)
  (paths_B_to_C : ℕ)
  (paths_B_to_D : ℕ)
  (paths_D_to_C : ℕ)

/-- Calculates the total number of paths from A to C -/
def total_paths_A_to_C (g : DirectedGraph) : ℕ :=
  g.paths_A_to_B * (g.paths_B_to_C + g.paths_B_to_D * g.paths_D_to_C)

/-- Theorem stating that for the given graph configuration, there are 8 paths from A to C -/
theorem eight_paths_A_to_C :
  ∃ (g : DirectedGraph),
    g.paths_A_to_B = 2 ∧
    g.paths_B_to_C = 3 ∧
    g.paths_B_to_D = 1 ∧
    g.paths_D_to_C = 1 ∧
    total_paths_A_to_C g = 8 :=
by sorry

end NUMINAMATH_CALUDE_eight_paths_A_to_C_l2141_214148


namespace NUMINAMATH_CALUDE_exam_average_l2141_214153

theorem exam_average (n1 n2 : ℕ) (avg1 avg2 : ℚ) : 
  n1 = 15 → 
  n2 = 10 → 
  avg1 = 75 / 100 → 
  avg2 = 90 / 100 → 
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 81 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_l2141_214153


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l2141_214158

theorem quadratic_roots_reciprocal_sum (x₁ x₂ : ℝ) :
  x₁^2 - 5*x₁ + 4 = 0 →
  x₂^2 - 5*x₂ + 4 = 0 →
  x₁ ≠ x₂ →
  (1 / x₁) + (1 / x₂) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l2141_214158


namespace NUMINAMATH_CALUDE_max_factors_of_power_l2141_214105

def is_power_of_two_primes (b : ℕ) : Prop :=
  ∃ p q k l : ℕ, p.Prime ∧ q.Prime ∧ p ≠ q ∧ b = p^k * q^l

theorem max_factors_of_power (b n : ℕ) : 
  b > 0 → n > 0 → b ≤ 20 → n ≤ 20 → is_power_of_two_primes b →
  (∃ k : ℕ, k ≤ b^n ∧ (∀ m : ℕ, m ≤ b^n → Nat.card (Nat.divisors m) ≤ Nat.card (Nat.divisors k))) →
  Nat.card (Nat.divisors (b^n)) ≤ 441 :=
sorry

end NUMINAMATH_CALUDE_max_factors_of_power_l2141_214105


namespace NUMINAMATH_CALUDE_days_before_reinforcement_l2141_214161

/-- 
Given a garrison with initial provisions and a reinforcement, 
calculate the number of days that passed before the reinforcement arrived.
-/
theorem days_before_reinforcement 
  (initial_garrison : ℕ) 
  (initial_provisions : ℕ) 
  (reinforcement : ℕ) 
  (remaining_provisions : ℕ) 
  (h1 : initial_garrison = 150)
  (h2 : initial_provisions = 31)
  (h3 : reinforcement = 300)
  (h4 : remaining_provisions = 5) : 
  ∃ (x : ℕ), x = 16 ∧ 
    initial_garrison * (initial_provisions - x) = 
    (initial_garrison + reinforcement) * remaining_provisions :=
by sorry

end NUMINAMATH_CALUDE_days_before_reinforcement_l2141_214161


namespace NUMINAMATH_CALUDE_parallelogram_on_circle_l2141_214167

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if a point is on a circle
def onCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define a function to check if a quadrilateral is a parallelogram
def isParallelogram (a b c d : ℝ × ℝ) : Prop :=
  (b.1 - a.1 = d.1 - c.1) ∧ (b.2 - a.2 = d.2 - c.2)

theorem parallelogram_on_circle (a c : ℝ × ℝ) (γ : Circle) :
  ∃ (b d : ℝ × ℝ), onCircle b γ ∧ onCircle d γ ∧ isParallelogram a b c d :=
sorry

end NUMINAMATH_CALUDE_parallelogram_on_circle_l2141_214167


namespace NUMINAMATH_CALUDE_readers_intersection_l2141_214146

theorem readers_intersection (total : ℕ) (sci_fi : ℕ) (literary : ℕ) 
  (h1 : total = 250) (h2 : sci_fi = 180) (h3 : literary = 88) :
  sci_fi + literary - total = 18 := by
  sorry

end NUMINAMATH_CALUDE_readers_intersection_l2141_214146


namespace NUMINAMATH_CALUDE_sparkling_water_cost_l2141_214175

/-- The cost of sparkling water bottles for Mary Anne -/
theorem sparkling_water_cost (bottles_per_night : ℚ) (yearly_cost : ℕ) : 
  bottles_per_night = 1/5 → yearly_cost = 146 → 
  (365 : ℚ) / 5 * (yearly_cost : ℚ) / ((365 : ℚ) / 5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sparkling_water_cost_l2141_214175


namespace NUMINAMATH_CALUDE_roots_sum_fourth_powers_l2141_214121

theorem roots_sum_fourth_powers (c d : ℝ) : 
  c^2 - 6*c + 8 = 0 → 
  d^2 - 6*d + 8 = 0 → 
  c^4 + c^3*d + d^3*c + d^4 = 432 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_fourth_powers_l2141_214121


namespace NUMINAMATH_CALUDE_cricket_team_age_problem_l2141_214102

/-- Represents the age difference between the wicket keeper and the team average -/
def wicket_keeper_age_difference (team_size : ℕ) (team_average_age : ℕ) 
  (known_member_age : ℕ) (remaining_average_age : ℕ) : ℕ :=
  let total_age := team_size * team_average_age
  let remaining_total_age := (team_size - 2) * remaining_average_age
  let wicket_keeper_age := total_age - known_member_age - remaining_total_age
  wicket_keeper_age - team_average_age

theorem cricket_team_age_problem :
  wicket_keeper_age_difference 11 22 25 21 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_problem_l2141_214102


namespace NUMINAMATH_CALUDE_fixed_point_on_all_parabolas_l2141_214197

/-- A parabola of the form y = 4x^2 + 2tx - 3t, where t is a real parameter -/
def parabola (t : ℝ) (x : ℝ) : ℝ := 4 * x^2 + 2 * t * x - 3 * t

/-- The fixed point through which all parabolas pass -/
def fixed_point : ℝ × ℝ := (1, 4)

/-- Theorem stating that the fixed point lies on all parabolas -/
theorem fixed_point_on_all_parabolas :
  ∀ t : ℝ, parabola t (fixed_point.1) = fixed_point.2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_all_parabolas_l2141_214197


namespace NUMINAMATH_CALUDE_exactly_two_trains_on_time_l2141_214109

-- Define the probabilities of each train arriving on time
def P_A : ℝ := 0.8
def P_B : ℝ := 0.7
def P_C : ℝ := 0.9

-- Define the probability of exactly two trains arriving on time
def P_exactly_two : ℝ := 
  P_A * P_B * (1 - P_C) + P_A * (1 - P_B) * P_C + (1 - P_A) * P_B * P_C

-- Theorem statement
theorem exactly_two_trains_on_time : P_exactly_two = 0.398 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_trains_on_time_l2141_214109


namespace NUMINAMATH_CALUDE_skew_symmetric_determinant_nonnegative_l2141_214179

theorem skew_symmetric_determinant_nonnegative 
  (a b c d e f : ℝ) : 
  (a * f - b * e + c * d)^2 ≥ 0 := by sorry

end NUMINAMATH_CALUDE_skew_symmetric_determinant_nonnegative_l2141_214179


namespace NUMINAMATH_CALUDE_bus_children_count_l2141_214143

theorem bus_children_count (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 26 → additional = 38 → total = initial + additional → total = 64 := by
sorry

end NUMINAMATH_CALUDE_bus_children_count_l2141_214143


namespace NUMINAMATH_CALUDE_average_tv_sets_is_48_l2141_214160

/-- The average number of TV sets in 5 electronic shops -/
def average_tv_sets : ℚ :=
  let shops := 5
  let tv_sets := [20, 30, 60, 80, 50]
  (tv_sets.sum : ℚ) / shops

/-- Theorem: The average number of TV sets in the 5 electronic shops is 48 -/
theorem average_tv_sets_is_48 : average_tv_sets = 48 := by
  sorry

end NUMINAMATH_CALUDE_average_tv_sets_is_48_l2141_214160


namespace NUMINAMATH_CALUDE_sum_of_k_values_l2141_214107

theorem sum_of_k_values : ∃ (S : Finset ℕ), 
  (∀ k ∈ S, ∃ j : ℕ, j > 0 ∧ k > 0 ∧ (1 : ℚ) / j + 1 / k = (1 : ℚ) / 4) ∧
  (∀ k : ℕ, k > 0 → (∃ j : ℕ, j > 0 ∧ (1 : ℚ) / j + 1 / k = (1 : ℚ) / 4) → k ∈ S) ∧
  Finset.sum S id = 51 :=
sorry

end NUMINAMATH_CALUDE_sum_of_k_values_l2141_214107


namespace NUMINAMATH_CALUDE_valentina_share_ratio_l2141_214142

/-- The length of the burger in inches -/
def burger_length : ℚ := 12

/-- The length of each person's share in inches -/
def share_length : ℚ := 6

/-- The ratio of Valentina's share to the whole burger -/
def valentina_ratio : ℚ × ℚ := (share_length, burger_length)

theorem valentina_share_ratio :
  valentina_ratio = (1, 2) := by sorry

end NUMINAMATH_CALUDE_valentina_share_ratio_l2141_214142


namespace NUMINAMATH_CALUDE_miss_adamson_classes_l2141_214166

/-- The number of classes Miss Adamson has -/
def number_of_classes (students_per_class : ℕ) (sheets_per_student : ℕ) (total_sheets : ℕ) : ℕ :=
  total_sheets / (students_per_class * sheets_per_student)

/-- Proof that Miss Adamson has 4 classes -/
theorem miss_adamson_classes :
  number_of_classes 20 5 400 = 4 := by
  sorry

end NUMINAMATH_CALUDE_miss_adamson_classes_l2141_214166


namespace NUMINAMATH_CALUDE_smallest_cut_length_l2141_214191

theorem smallest_cut_length : 
  ∃ (x : ℕ), x > 0 ∧ x ≤ 8 ∧ 
  (∀ (y : ℕ), y > 0 → y < x → (8 - y) + (15 - y) > 17 - y) ∧
  (8 - x) + (15 - x) ≤ 17 - x ∧
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cut_length_l2141_214191


namespace NUMINAMATH_CALUDE_cannot_distinguish_normal_l2141_214110

/-- Represents the three types of people on the island -/
inductive PersonType
  | Knight
  | Liar
  | Normal

/-- Represents a statement that can be true or false -/
structure Statement where
  content : Prop

/-- A function that determines whether a given person type would make a given statement -/
def wouldMakeStatement (personType : PersonType) (statement : Statement) : Prop :=
  match personType with
  | PersonType.Knight => statement.content
  | PersonType.Liar => ¬statement.content
  | PersonType.Normal => True

/-- The main theorem stating that it's impossible to distinguish a normal person from a knight or liar with any finite number of statements -/
theorem cannot_distinguish_normal (n : ℕ) :
  ∃ (statements : Fin n → Statement),
    (∀ i, wouldMakeStatement PersonType.Normal (statements i)) ∧
    ((∀ i, wouldMakeStatement PersonType.Knight (statements i)) ∨
     (∀ i, wouldMakeStatement PersonType.Liar (statements i))) :=
sorry

end NUMINAMATH_CALUDE_cannot_distinguish_normal_l2141_214110


namespace NUMINAMATH_CALUDE_constant_c_value_l2141_214116

theorem constant_c_value : ∃ c : ℚ, 
  (∀ x : ℚ, (3 * x^3 - 5 * x^2 + 6 * x - 4) * (2 * x^2 + c * x + 8) = 
   6 * x^5 - 19 * x^4 + 40 * x^3 + c * x^2 - 32 * x + 32) ∧ 
  c = 48 / 5 := by
sorry

end NUMINAMATH_CALUDE_constant_c_value_l2141_214116


namespace NUMINAMATH_CALUDE_area_between_squares_l2141_214152

/-- The area of the region between two concentric squares -/
theorem area_between_squares (outer_side : ℝ) (inner_side : ℝ) 
  (h_outer : outer_side = 6) 
  (h_inner : inner_side = 4) :
  outer_side ^ 2 - inner_side ^ 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_area_between_squares_l2141_214152


namespace NUMINAMATH_CALUDE_smallest_class_size_l2141_214176

theorem smallest_class_size : ∃ (n : ℕ), n > 0 ∧ (∃ (k : ℕ), k > 0 ∧ 29/10 < (100 * k : ℚ)/n ∧ (100 * k : ℚ)/n < 31/10) ∧
  ∀ (m : ℕ), m > 0 → m < n → ¬(∃ (j : ℕ), j > 0 ∧ 29/10 < (100 * j : ℚ)/m ∧ (100 * j : ℚ)/m < 31/10) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2141_214176


namespace NUMINAMATH_CALUDE_marco_strawberries_weight_l2141_214172

/-- The weight of Marco's strawberries in pounds -/
def marco_weight : ℝ := 37 - 22

/-- Theorem stating that Marco's strawberries weighed 15 pounds -/
theorem marco_strawberries_weight :
  marco_weight = 15 := by sorry

end NUMINAMATH_CALUDE_marco_strawberries_weight_l2141_214172


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2141_214189

theorem absolute_value_inequality (x : ℝ) : 
  |((5 - x) / 3)| < 2 ↔ -1 < x ∧ x < 11 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2141_214189


namespace NUMINAMATH_CALUDE_calories_per_candy_bar_l2141_214133

/-- Given that there are 15 calories in 5 candy bars, prove that there are 3 calories in one candy bar. -/
theorem calories_per_candy_bar :
  let total_calories : ℕ := 15
  let total_bars : ℕ := 5
  let calories_per_bar : ℚ := total_calories / total_bars
  calories_per_bar = 3 := by sorry

end NUMINAMATH_CALUDE_calories_per_candy_bar_l2141_214133


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2141_214195

theorem floor_ceil_sum : ⌊(-3.276 : ℝ)⌋ + ⌈(-17.845 : ℝ)⌉ = -21 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2141_214195


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2141_214154

theorem quadratic_inequality (x : ℝ) : (x - 2) * (x + 2) > 0 ↔ x > 2 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2141_214154


namespace NUMINAMATH_CALUDE_square_congruent_one_iff_one_or_minus_one_l2141_214129

theorem square_congruent_one_iff_one_or_minus_one (p : Nat) (hp : Prime p) :
  ∀ a : Nat, a^2 ≡ 1 [ZMOD p] ↔ a ≡ 1 [ZMOD p] ∨ a ≡ p - 1 [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_square_congruent_one_iff_one_or_minus_one_l2141_214129


namespace NUMINAMATH_CALUDE_polygon_D_has_largest_area_l2141_214111

-- Define the basic shapes
def square_area : ℝ := 1
def isosceles_right_triangle_area : ℝ := 0.5
def parallelogram_area : ℝ := 1

-- Define the polygons
def polygon_A_area : ℝ := 3 * square_area + 2 * isosceles_right_triangle_area
def polygon_B_area : ℝ := 2 * square_area + 4 * isosceles_right_triangle_area
def polygon_C_area : ℝ := square_area + 2 * isosceles_right_triangle_area + parallelogram_area
def polygon_D_area : ℝ := 4 * square_area + parallelogram_area
def polygon_E_area : ℝ := 2 * square_area + 3 * isosceles_right_triangle_area + parallelogram_area

-- Theorem statement
theorem polygon_D_has_largest_area :
  polygon_D_area > polygon_A_area ∧
  polygon_D_area > polygon_B_area ∧
  polygon_D_area > polygon_C_area ∧
  polygon_D_area > polygon_E_area :=
by sorry

end NUMINAMATH_CALUDE_polygon_D_has_largest_area_l2141_214111


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2141_214124

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

-- State the theorem
theorem intersection_implies_a_value :
  ∀ a : ℝ, (A ∩ B a = {x | -2 ≤ x ∧ x ≤ 1}) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2141_214124


namespace NUMINAMATH_CALUDE_shirt_discount_l2141_214114

/-- Given a shirt with an original price and a sale price, calculate the discount amount. -/
def discount (original_price sale_price : ℕ) : ℕ :=
  original_price - sale_price

/-- Theorem stating that for a shirt with an original price of $22 and a sale price of $16, 
    the discount amount is $6. -/
theorem shirt_discount :
  let original_price : ℕ := 22
  let sale_price : ℕ := 16
  discount original_price sale_price = 6 := by
sorry

end NUMINAMATH_CALUDE_shirt_discount_l2141_214114


namespace NUMINAMATH_CALUDE_laptop_price_proof_l2141_214104

/-- The sticker price of the laptop -/
def stickerPrice : ℝ := 750

/-- The price at Store A after discount and rebate -/
def storePriceA (x : ℝ) : ℝ := 0.8 * x - 100

/-- The price at Store B after discount -/
def storePriceB (x : ℝ) : ℝ := 0.7 * x

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem laptop_price_proof :
  storePriceB stickerPrice - storePriceA stickerPrice = 25 :=
sorry

end NUMINAMATH_CALUDE_laptop_price_proof_l2141_214104


namespace NUMINAMATH_CALUDE_count_divisible_integers_l2141_214190

theorem count_divisible_integers : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, m > 0 ∧ (1806 : ℤ) ∣ (m^2 - 2)) ∧ 
    (∀ m : ℕ, m > 0 ∧ (1806 : ℤ) ∣ (m^2 - 2) → m ∈ S) ∧
    Finset.card S = 2 :=
by sorry

end NUMINAMATH_CALUDE_count_divisible_integers_l2141_214190


namespace NUMINAMATH_CALUDE_soldier_target_practice_l2141_214144

theorem soldier_target_practice (total_shots : ℕ) (total_score : ℕ) (tens : ℕ) (tens_score : ℕ) :
  total_shots = 10 →
  total_score = 90 →
  tens = 4 →
  tens_score = 10 →
  ∃ (sevens eights nines : ℕ),
    sevens + eights + nines = total_shots - tens ∧
    7 * sevens + 8 * eights + 9 * nines = total_score - tens * tens_score ∧
    sevens = 1 :=
by sorry

end NUMINAMATH_CALUDE_soldier_target_practice_l2141_214144


namespace NUMINAMATH_CALUDE_power_equality_l2141_214180

theorem power_equality : (3 : ℕ) ^ 20 = 243 ^ 4 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2141_214180


namespace NUMINAMATH_CALUDE_man_downstream_speed_l2141_214178

/-- Given a man's upstream speed and the stream speed, calculates his downstream speed. -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  upstream_speed + 2 * stream_speed

/-- Theorem stating that given the specific upstream speed and stream speed, 
    the downstream speed is 15 kmph. -/
theorem man_downstream_speed :
  downstream_speed 8 3.5 = 15 := by
  sorry

#eval downstream_speed 8 3.5

end NUMINAMATH_CALUDE_man_downstream_speed_l2141_214178


namespace NUMINAMATH_CALUDE_fence_cost_l2141_214165

theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 57) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let cost := perimeter * price_per_foot
  cost = 3876 := by sorry

end NUMINAMATH_CALUDE_fence_cost_l2141_214165


namespace NUMINAMATH_CALUDE_original_curve_equation_l2141_214170

/-- Given a scaling transformation and the equation of the transformed curve,
    prove the equation of the original curve. -/
theorem original_curve_equation
  (x y x' y' : ℝ) -- Real variables for coordinates
  (h1 : x' = 5 * x) -- Scaling transformation for x
  (h2 : y' = 3 * y) -- Scaling transformation for y
  (h3 : 2 * x' ^ 2 + 8 * y' ^ 2 = 1) -- Equation of transformed curve
  : 50 * x ^ 2 + 72 * y ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_original_curve_equation_l2141_214170


namespace NUMINAMATH_CALUDE_seokjin_paper_left_l2141_214164

/-- Given the initial number of sheets, number of notebooks, and pages per notebook,
    calculate the remaining sheets of paper. -/
def remaining_sheets (initial_sheets : ℕ) (num_notebooks : ℕ) (pages_per_notebook : ℕ) : ℕ :=
  initial_sheets - (num_notebooks * pages_per_notebook)

/-- Theorem stating that given 100 initial sheets, 3 notebooks with 30 pages each,
    the remaining sheets is 10. -/
theorem seokjin_paper_left : remaining_sheets 100 3 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_seokjin_paper_left_l2141_214164


namespace NUMINAMATH_CALUDE_distance_A_to_C_distance_A_to_C_is_300_l2141_214155

/-- The distance between city A and city C given the travel times and speeds of Eddy and Freddy -/
theorem distance_A_to_C (eddy_time : ℝ) (freddy_time : ℝ) (distance_A_to_B : ℝ) (speed_ratio : ℝ) : ℝ :=
  let eddy_speed := distance_A_to_B / eddy_time
  let freddy_speed := eddy_speed / speed_ratio
  freddy_speed * freddy_time

/-- The actual distance between city A and city C is 300 km -/
theorem distance_A_to_C_is_300 : distance_A_to_C 3 4 450 2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_to_C_distance_A_to_C_is_300_l2141_214155


namespace NUMINAMATH_CALUDE_lamp_purchasing_problem_l2141_214125

/-- Represents a purchasing plan for energy-saving lamps -/
structure LampPlan where
  typeA : ℕ
  typeB : ℕ
  cost : ℕ

/-- Checks if a plan satisfies the given constraints -/
def isValidPlan (plan : LampPlan) : Prop :=
  plan.typeA + plan.typeB = 50 ∧
  2 * plan.typeB ≤ plan.typeA ∧
  plan.typeA ≤ 3 * plan.typeB

/-- Calculates the cost of a plan given the prices of lamps -/
def calculateCost (priceA priceB : ℕ) (plan : LampPlan) : ℕ :=
  priceA * plan.typeA + priceB * plan.typeB

/-- Main theorem to prove -/
theorem lamp_purchasing_problem :
  ∃ (priceA priceB : ℕ) (plans : List LampPlan),
    priceA + 3 * priceB = 26 ∧
    3 * priceA + 2 * priceB = 29 ∧
    priceA = 5 ∧
    priceB = 7 ∧
    plans.length = 4 ∧
    (∀ plan ∈ plans, isValidPlan plan) ∧
    (∃ bestPlan ∈ plans,
      bestPlan.typeA = 37 ∧
      bestPlan.typeB = 13 ∧
      calculateCost priceA priceB bestPlan = 276 ∧
      ∀ plan ∈ plans, calculateCost priceA priceB bestPlan ≤ calculateCost priceA priceB plan) :=
sorry

end NUMINAMATH_CALUDE_lamp_purchasing_problem_l2141_214125


namespace NUMINAMATH_CALUDE_symmetry_wrt_y_axis_l2141_214147

/-- Given a point P with coordinates (x, y), the point symmetrical to P
    with respect to the y-axis has coordinates (-x, y) -/
def symmetrical_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (-(P.1), P.2)

theorem symmetry_wrt_y_axis :
  let P : ℝ × ℝ := (3, -5)
  symmetrical_point P = (-3, -5) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_wrt_y_axis_l2141_214147


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2141_214119

theorem arithmetic_calculation : (1 + 2) * (3 - 4) + 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2141_214119


namespace NUMINAMATH_CALUDE_expression_value_l2141_214199

theorem expression_value (m n : ℝ) 
  (h1 : m^2 + 2*m*n = 3) 
  (h2 : 2*n^2 + 3*m*n = 5) : 
  2*m^2 + 13*m*n + 6*n^2 = 21 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2141_214199


namespace NUMINAMATH_CALUDE_sqrt_three_multiplication_l2141_214108

theorem sqrt_three_multiplication : Real.sqrt 3 * (2 * Real.sqrt 3 - 2) = 6 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_multiplication_l2141_214108


namespace NUMINAMATH_CALUDE_angle_bisector_ratio_bound_angle_bisector_ratio_bound_tight_l2141_214188

/-- A triangle with sides a and b, and corresponding angle bisectors t_a and t_b -/
structure Triangle where
  a : ℝ
  b : ℝ
  t_a : ℝ
  t_b : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < t_a ∧ 0 < t_b
  h_triangle : a < b + t_b ∧ b < a + t_a ∧ t_b < a + b
  h_bisector_a : t_a < (2 * b * (a + b)) / (a + 2 * b)
  h_bisector_b : t_b < (2 * a * (a + b)) / (2 * a + b)

/-- The upper bound for the ratio of sum of angle bisectors to sum of sides is 4/3 -/
theorem angle_bisector_ratio_bound (T : Triangle) :
  (T.t_a + T.t_b) / (T.a + T.b) < 4/3 :=
sorry

/-- The upper bound 4/3 is the least possible -/
theorem angle_bisector_ratio_bound_tight :
  ∀ ε > 0, ∃ T : Triangle, (4/3 - ε) < (T.t_a + T.t_b) / (T.a + T.b) :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_ratio_bound_angle_bisector_ratio_bound_tight_l2141_214188


namespace NUMINAMATH_CALUDE_month_with_conditions_has_30_days_l2141_214182

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with its number of days and day counts -/
structure Month where
  days : Nat
  dayCounts : DayOfWeek → Nat

/-- Definition of a valid month -/
def validMonth (m : Month) : Prop :=
  (m.days ≥ 28 ∧ m.days ≤ 31) ∧
  (∀ d : DayOfWeek, m.dayCounts d = 4 ∨ m.dayCounts d = 5)

/-- The condition of more Mondays than Tuesdays -/
def moreMondays (m : Month) : Prop :=
  m.dayCounts DayOfWeek.Monday > m.dayCounts DayOfWeek.Tuesday

/-- The condition of fewer Saturdays than Sundays -/
def fewerSaturdays (m : Month) : Prop :=
  m.dayCounts DayOfWeek.Saturday < m.dayCounts DayOfWeek.Sunday

theorem month_with_conditions_has_30_days (m : Month) 
  (hValid : validMonth m) 
  (hMondays : moreMondays m) 
  (hSaturdays : fewerSaturdays m) : 
  m.days = 30 := by
  sorry

end NUMINAMATH_CALUDE_month_with_conditions_has_30_days_l2141_214182


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2141_214103

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, 4 * x^2 + (a - 2) * x + 1/4 > 0) ↔ 
  (a ≤ 0 ∨ a ≥ 4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2141_214103


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l2141_214123

theorem rationalize_and_simplify :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l2141_214123


namespace NUMINAMATH_CALUDE_intersection_slope_range_l2141_214115

/-- Given two points P and Q in the Cartesian plane, and a linear function y = kx - 1
    that intersects the extension of line segment PQ (excluding Q),
    prove that the range of k is (1/3, 3/2). -/
theorem intersection_slope_range (P Q : ℝ × ℝ) (k : ℝ) : 
  P = (-1, 1) →
  Q = (2, 2) →
  (∃ x y : ℝ, y = k * x - 1 ∧ (y - 1) / (x + 1) = (2 - 1) / (2 + 1) ∧ (x, y) ≠ Q) →
  1/3 < k ∧ k < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_slope_range_l2141_214115


namespace NUMINAMATH_CALUDE_celine_payment_l2141_214157

/-- Represents the daily charge for borrowing a book -/
def daily_charge : ℚ := 1/2

/-- Represents the number of days in May -/
def days_in_may : ℕ := 31

/-- Represents the number of books Celine borrowed -/
def books_borrowed : ℕ := 3

/-- Represents the number of days Celine kept the first book -/
def days_first_book : ℕ := 20

/-- Calculates the total amount Celine paid for borrowing the books -/
def total_amount : ℚ :=
  daily_charge * days_first_book +
  daily_charge * days_in_may * 2

theorem celine_payment :
  total_amount = 41 :=
sorry

end NUMINAMATH_CALUDE_celine_payment_l2141_214157


namespace NUMINAMATH_CALUDE_pizza_solution_l2141_214140

/-- The number of pizzas made by Craig and Heather over two days -/
def pizza_problem (craig_day1 craig_day2 heather_day1 heather_day2 : ℕ) : Prop :=
  let total := craig_day1 + craig_day2 + heather_day1 + heather_day2
  craig_day1 = 40 ∧
  craig_day2 = craig_day1 + 60 ∧
  heather_day1 = 4 * craig_day1 ∧
  total = 380 ∧
  craig_day2 - heather_day2 = 20

theorem pizza_solution :
  ∃ (craig_day1 craig_day2 heather_day1 heather_day2 : ℕ),
    pizza_problem craig_day1 craig_day2 heather_day1 heather_day2 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_solution_l2141_214140


namespace NUMINAMATH_CALUDE_feline_sanctuary_count_l2141_214177

theorem feline_sanctuary_count :
  let lions : ℕ := 12
  let tigers : ℕ := 14
  let cougars : ℕ := (lions + tigers) / 3
  lions + tigers + cougars = 34 := by
sorry

end NUMINAMATH_CALUDE_feline_sanctuary_count_l2141_214177


namespace NUMINAMATH_CALUDE_loss_percentage_book1_l2141_214163

-- Define the total cost of both books
def total_cost : ℝ := 450

-- Define the cost of the first book (sold at a loss)
def cost_book1 : ℝ := 262.5

-- Define the gain percentage on the second book
def gain_percentage : ℝ := 0.19

-- Define the function to calculate the selling price of the second book
def selling_price_book2 (cost : ℝ) : ℝ := cost * (1 + gain_percentage)

-- Define the theorem
theorem loss_percentage_book1 : 
  let cost_book2 := total_cost - cost_book1
  let sp := selling_price_book2 cost_book2
  let loss_percentage := (cost_book1 - sp) / cost_book1 * 100
  loss_percentage = 15 := by sorry

end NUMINAMATH_CALUDE_loss_percentage_book1_l2141_214163


namespace NUMINAMATH_CALUDE_stamp_cost_difference_l2141_214134

/-- The cost of a single rooster stamp -/
def rooster_stamp_cost : ℚ := 1.5

/-- The cost of a single daffodil stamp -/
def daffodil_stamp_cost : ℚ := 0.75

/-- The number of rooster stamps purchased -/
def rooster_stamp_count : ℕ := 2

/-- The number of daffodil stamps purchased -/
def daffodil_stamp_count : ℕ := 5

/-- The theorem stating the cost difference between daffodil and rooster stamps -/
theorem stamp_cost_difference : 
  (daffodil_stamp_count : ℚ) * daffodil_stamp_cost - 
  (rooster_stamp_count : ℚ) * rooster_stamp_cost = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_stamp_cost_difference_l2141_214134


namespace NUMINAMATH_CALUDE_composite_function_evaluation_l2141_214112

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 4
def g (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem composite_function_evaluation :
  g (f (g 3)) = 192 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_evaluation_l2141_214112


namespace NUMINAMATH_CALUDE_swing_rope_length_proof_l2141_214162

/-- The length of a swing rope satisfying specific conditions -/
def swing_rope_length : ℝ := 14.5

/-- The initial height of the swing's footboard off the ground -/
def initial_height : ℝ := 1

/-- The distance the swing is pushed forward -/
def push_distance : ℝ := 10

/-- The height of the person -/
def person_height : ℝ := 5

theorem swing_rope_length_proof :
  ∃ (rope_length : ℝ),
    rope_length = swing_rope_length ∧
    rope_length^2 = push_distance^2 + (rope_length - person_height + initial_height)^2 :=
by sorry

end NUMINAMATH_CALUDE_swing_rope_length_proof_l2141_214162


namespace NUMINAMATH_CALUDE_sin_has_P_pi_property_P4_central_sym_monotone_P0_P3_implies_periodic_l2141_214126

-- Definition of P(a) property
def has_P_property (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, ∃ a, f (x + a) = f (-x)

-- Statement 1
theorem sin_has_P_pi_property : has_P_property Real.sin π :=
  sorry

-- Definition of central symmetry about a point
def centrally_symmetric (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f x

-- Definition of monotonically decreasing on an interval
def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Definition of monotonically increasing on an interval
def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Statement 3
theorem P4_central_sym_monotone (f : ℝ → ℝ) 
  (h1 : has_P_property f 4)
  (h2 : centrally_symmetric f (1, 0))
  (h3 : ∃ ε > 0, monotone_decreasing_on f (-1-ε) (-1+ε)) :
  monotone_decreasing_on f (-2) (-1) ∧ monotone_increasing_on f 1 2 :=
  sorry

-- Definition of periodic function
def periodic (f : ℝ → ℝ) : Prop :=
  ∃ p ≠ 0, ∀ x, f (x + p) = f x

-- Statement 4
theorem P0_P3_implies_periodic (f g : ℝ → ℝ)
  (h1 : f ≠ 0)
  (h2 : has_P_property f 0)
  (h3 : has_P_property f 3)
  (h4 : ∀ x₁ x₂, |f x₁ - f x₂| ≥ |g x₁ - g x₂|) :
  periodic g :=
  sorry

end NUMINAMATH_CALUDE_sin_has_P_pi_property_P4_central_sym_monotone_P0_P3_implies_periodic_l2141_214126


namespace NUMINAMATH_CALUDE_range_of_a_l2141_214113

def statement_p (a : ℝ) : Prop :=
  ∀ x, x^2 + (a - 1) * x + a^2 > 0

def statement_q (a : ℝ) : Prop :=
  ∀ x y, x < y → (2 * a^2 - a)^x < (2 * a^2 - a)^y

theorem range_of_a :
  ∀ a : ℝ, (statement_p a ∨ statement_q a) ∧ ¬(statement_p a ∧ statement_q a) →
    (1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2141_214113


namespace NUMINAMATH_CALUDE_schedule_theorem_l2141_214136

/-- The number of periods in a day -/
def num_periods : ℕ := 7

/-- The number of courses to be scheduled -/
def num_courses : ℕ := 4

/-- Calculates the number of ways to schedule distinct courses in non-consecutive periods -/
def schedule_ways (periods : ℕ) (courses : ℕ) : ℕ :=
  (Nat.choose (periods - courses + 1) courses) * (Nat.factorial courses)

/-- Theorem stating that there are 1680 ways to schedule 4 distinct courses in a 7-period day
    with no two courses in consecutive periods -/
theorem schedule_theorem : 
  schedule_ways num_periods num_courses = 1680 := by
  sorry

end NUMINAMATH_CALUDE_schedule_theorem_l2141_214136


namespace NUMINAMATH_CALUDE_negation_of_set_implication_l2141_214137

theorem negation_of_set_implication (A B : Set α) :
  ¬(A ∪ B = A → A ∩ B = B) ↔ (A ∪ B ≠ A → A ∩ B ≠ B) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_set_implication_l2141_214137


namespace NUMINAMATH_CALUDE_mixture_composition_l2141_214187

/-- Represents a solution with percentages of materials A and B -/
structure Solution :=
  (percentA : ℝ)
  (percentB : ℝ)
  (sum_to_100 : percentA + percentB = 100)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solutionX : Solution)
  (solutionY : Solution)
  (finalPercentA : ℝ)

theorem mixture_composition 
  (m : Mixture)
  (hX : m.solutionX.percentA = 20 ∧ m.solutionX.percentB = 80)
  (hY : m.solutionY.percentA = 30 ∧ m.solutionY.percentB = 70)
  (hFinal : m.finalPercentA = 22) :
  100 - m.finalPercentA = 78 := by
  sorry

end NUMINAMATH_CALUDE_mixture_composition_l2141_214187


namespace NUMINAMATH_CALUDE_train_length_calculation_l2141_214181

/-- Calculates the length of a train given the speeds of two trains, the time they take to cross each other, and the length of the other train. -/
theorem train_length_calculation (v1 v2 : ℝ) (t cross_time : ℝ) (l2 : ℝ) :
  v1 = 60 →
  v2 = 40 →
  cross_time = 12.239020878329734 →
  l2 = 200 →
  (v1 + v2) * 1000 / 3600 * cross_time - l2 = 140 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2141_214181


namespace NUMINAMATH_CALUDE_x_intercept_of_specific_line_l2141_214130

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ := sorry

/-- The line passing through (4, 6) and (8, 2) -/
def specific_line : Line := { x₁ := 4, y₁ := 6, x₂ := 8, y₂ := 2 }

theorem x_intercept_of_specific_line :
  x_intercept specific_line = 10 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_specific_line_l2141_214130


namespace NUMINAMATH_CALUDE_time_to_install_remaining_windows_l2141_214174

/-- Calculates the time to install remaining windows -/
def timeToInstallRemaining (totalWindows installedWindows timePerWindow : ℕ) : ℕ :=
  (totalWindows - installedWindows) * timePerWindow

/-- Theorem: Time to install remaining windows is 18 hours -/
theorem time_to_install_remaining_windows :
  timeToInstallRemaining 9 6 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_time_to_install_remaining_windows_l2141_214174


namespace NUMINAMATH_CALUDE_pages_per_book_l2141_214118

/-- Given that Frank took 12 days to finish each book and 492 days to finish all 41 books,
    prove that each book had 492 pages. -/
theorem pages_per_book (days_per_book : ℕ) (total_days : ℕ) (total_books : ℕ) :
  days_per_book = 12 →
  total_days = 492 →
  total_books = 41 →
  (total_days / days_per_book) * days_per_book = 492 := by
  sorry

#check pages_per_book

end NUMINAMATH_CALUDE_pages_per_book_l2141_214118


namespace NUMINAMATH_CALUDE_system_solution_ratio_l2141_214173

theorem system_solution_ratio (x y c d : ℝ) :
  (4 * x - 2 * y = c) →
  (6 * y - 12 * x = d) →
  d ≠ 0 →
  (∃ x y, (4 * x - 2 * y = c) ∧ (6 * y - 12 * x = d)) →
  c / d = -1 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l2141_214173


namespace NUMINAMATH_CALUDE_cost_per_box_l2141_214101

/-- The cost per box for packaging the fine arts collection --/
theorem cost_per_box (box_length box_width box_height : ℝ)
  (total_volume min_total_cost : ℝ) :
  box_length = 20 ∧ box_width = 20 ∧ box_height = 15 ∧
  total_volume = 3060000 ∧ min_total_cost = 357 →
  (min_total_cost / (total_volume / (box_length * box_width * box_height))) = 0.70 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_box_l2141_214101


namespace NUMINAMATH_CALUDE_square_side_length_with_equal_perimeter_circle_l2141_214128

theorem square_side_length_with_equal_perimeter_circle (r : ℝ) :
  ∃ (s : ℝ), (4 * s = 2 * π * r) → (s = 3 * π / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_square_side_length_with_equal_perimeter_circle_l2141_214128


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l2141_214186

theorem smaller_root_of_equation (x : ℝ) : 
  (x - 2/3) * (x - 2/3) + (x - 2/3) * (x - 1/3) = 0 →
  (x = 1/2 ∨ x = 2/3) ∧ 1/2 < 2/3 := by
sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l2141_214186


namespace NUMINAMATH_CALUDE_sin_n_eq_cos_810_l2141_214151

theorem sin_n_eq_cos_810 (n : ℤ) (h1 : -180 ≤ n) (h2 : n ≤ 180) (h3 : Real.sin (n * π / 180) = Real.cos (810 * π / 180)) :
  n = 0 ∨ n = 180 ∨ n = -180 := by
sorry

end NUMINAMATH_CALUDE_sin_n_eq_cos_810_l2141_214151


namespace NUMINAMATH_CALUDE_bag_of_balls_l2141_214194

theorem bag_of_balls (white green yellow red purple : ℕ) 
  (h1 : white = 20)
  (h2 : green = 30)
  (h3 : yellow = 10)
  (h4 : red = 37)
  (h5 : purple = 3)
  (h6 : (white + green + yellow : ℚ) / (white + green + yellow + red + purple) = 3/5) :
  white + green + yellow + red + purple = 100 := by
  sorry

end NUMINAMATH_CALUDE_bag_of_balls_l2141_214194


namespace NUMINAMATH_CALUDE_parabola_properties_l2141_214185

-- Define the parabola function
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem parabola_properties (a b c : ℝ) :
  (parabola a b c (-2) = 0) →
  (parabola a b c (-1) = 4) →
  (parabola a b c 0 = 6) →
  (parabola a b c 1 = 6) →
  (a < 0) ∧
  (∀ x, parabola a b c x ≤ parabola a b c (1/2)) ∧
  (parabola a b c (1/2) = 25/4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2141_214185
