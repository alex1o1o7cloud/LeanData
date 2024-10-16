import Mathlib

namespace NUMINAMATH_CALUDE_maximize_subsidy_l3656_365627

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.log (x + 1) - x / 10 + 1

theorem maximize_subsidy (m : ℝ) (h_m : m > 0) :
  let max_subsidy := fun x : ℝ => x ≥ 1 ∧ x ≤ 9 ∧ ∀ y, 1 ≤ y ∧ y ≤ 9 → f m x ≥ f m y
  (m ≤ 1/5 ∧ max_subsidy 1) ∨
  (1/5 < m ∧ m < 1 ∧ max_subsidy (10*m - 1)) ∨
  (m ≥ 1 ∧ max_subsidy 9) :=
by sorry

end NUMINAMATH_CALUDE_maximize_subsidy_l3656_365627


namespace NUMINAMATH_CALUDE_courtyard_width_is_16_meters_l3656_365694

def courtyard_length : ℝ := 25
def brick_length : ℝ := 0.2
def brick_width : ℝ := 0.1
def total_bricks : ℕ := 20000

theorem courtyard_width_is_16_meters :
  let brick_area : ℝ := brick_length * brick_width
  let total_area : ℝ := (total_bricks : ℝ) * brick_area
  let courtyard_width : ℝ := total_area / courtyard_length
  courtyard_width = 16 := by sorry

end NUMINAMATH_CALUDE_courtyard_width_is_16_meters_l3656_365694


namespace NUMINAMATH_CALUDE_difference_of_bounds_l3656_365652

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := m * x + 2
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

-- Define the theorem
theorem difference_of_bounds (m : ℝ) :
  ∃ (a b : ℤ), (∀ x : ℝ, a ≤ f m x - g m x ∧ f m x - g m x ≤ b) ∧
  (∀ y : ℝ, a ≤ y ∧ y ≤ b → ∃ x : ℝ, f m x - g m x = y) →
  a - b = -2 :=
sorry

end NUMINAMATH_CALUDE_difference_of_bounds_l3656_365652


namespace NUMINAMATH_CALUDE_symmetric_axis_after_transformation_l3656_365626

/-- Given a function f(x) = √3 sin(x - π/6) + cos(x - π/6), 
    after stretching the horizontal coordinate to twice its original length 
    and shifting the graph π/6 units to the left, 
    one symmetric axis of the resulting function is at x = 5π/6 -/
theorem symmetric_axis_after_transformation (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sqrt 3 * Real.sin (x - π/6) + Real.cos (x - π/6)
  let g : ℝ → ℝ := λ x => f ((x + π/6) / 2)
  ∃ (k : ℤ), g (5*π/6 + 2*π*k) = g (5*π/6 - 2*π*k) := by
  sorry


end NUMINAMATH_CALUDE_symmetric_axis_after_transformation_l3656_365626


namespace NUMINAMATH_CALUDE_fries_ratio_l3656_365614

def sally_initial_fries : ℕ := 14
def mark_initial_fries : ℕ := 36
def sally_final_fries : ℕ := 26

def fries_mark_gave_sally : ℕ := sally_final_fries - sally_initial_fries

theorem fries_ratio :
  (fries_mark_gave_sally : ℚ) / mark_initial_fries = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fries_ratio_l3656_365614


namespace NUMINAMATH_CALUDE_inequality_proof_l3656_365617

theorem inequality_proof (a : ℝ) (ha : a > 0) :
  ((a^3 + 1) / (a^2 + 1))^2 ≥ a^2 - a + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3656_365617


namespace NUMINAMATH_CALUDE_specific_polygon_perimeter_l3656_365628

/-- A polygon that forms part of a square -/
structure PartialSquarePolygon where
  /-- The length of each visible side of the polygon -/
  visible_side_length : ℝ
  /-- The fraction of the square that the polygon occupies -/
  occupied_fraction : ℝ
  /-- Assumption that the visible side length is positive -/
  visible_side_positive : visible_side_length > 0
  /-- Assumption that the occupied fraction is between 0 and 1 -/
  occupied_fraction_valid : 0 < occupied_fraction ∧ occupied_fraction ≤ 1

/-- The perimeter of a polygon that forms part of a square -/
def perimeter (p : PartialSquarePolygon) : ℝ :=
  4 * p.visible_side_length * p.occupied_fraction

/-- Theorem stating that a polygon occupying three-fourths of a square with visible sides of 5 units has a perimeter of 15 units -/
theorem specific_polygon_perimeter :
  ∀ (p : PartialSquarePolygon),
  p.visible_side_length = 5 →
  p.occupied_fraction = 3/4 →
  perimeter p = 15 := by
  sorry

end NUMINAMATH_CALUDE_specific_polygon_perimeter_l3656_365628


namespace NUMINAMATH_CALUDE_count_ones_digits_divisible_by_six_l3656_365640

/-- A number is divisible by 6 if and only if it is divisible by both 2 and 3 -/
axiom divisible_by_six (n : ℕ) : n % 6 = 0 ↔ n % 2 = 0 ∧ n % 3 = 0

/-- The set of possible ones digits in numbers divisible by 6 -/
def ones_digits_divisible_by_six : Finset ℕ :=
  {0, 2, 4, 6, 8}

/-- The number of possible ones digits in numbers divisible by 6 is 5 -/
theorem count_ones_digits_divisible_by_six :
  Finset.card ones_digits_divisible_by_six = 5 := by sorry

end NUMINAMATH_CALUDE_count_ones_digits_divisible_by_six_l3656_365640


namespace NUMINAMATH_CALUDE_min_blocks_for_specific_wall_l3656_365698

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  length : ℕ
  height : ℕ

/-- Calculates the minimum number of blocks needed to build a wall --/
def minBlocksNeeded (wall : WallDimensions) (blocks : List BlockDimensions) : ℕ :=
  sorry

/-- Theorem: The minimum number of blocks needed for the specified wall is 404 --/
theorem min_blocks_for_specific_wall :
  let wall : WallDimensions := ⟨120, 8⟩
  let blocks : List BlockDimensions := [⟨2, 1⟩, ⟨3, 1⟩, ⟨1, 1⟩]
  minBlocksNeeded wall blocks = 404 :=
by
  sorry

#check min_blocks_for_specific_wall

end NUMINAMATH_CALUDE_min_blocks_for_specific_wall_l3656_365698


namespace NUMINAMATH_CALUDE_log_equation_implies_x_greater_than_two_l3656_365625

theorem log_equation_implies_x_greater_than_two (x : ℝ) :
  Real.log (x^2 + 5*x + 6) = Real.log ((x+1)*(x+4)) + Real.log (x-2) →
  x > 2 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_implies_x_greater_than_two_l3656_365625


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l3656_365605

/-- Calculates the sampling interval for systematic sampling -/
def samplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  let adjustedPopSize := (populationSize / sampleSize) * sampleSize
  adjustedPopSize / sampleSize

/-- Proves that the sampling interval is 10 for the given problem -/
theorem systematic_sampling_interval :
  samplingInterval 123 12 = 10 := by
  sorry

#eval samplingInterval 123 12

end NUMINAMATH_CALUDE_systematic_sampling_interval_l3656_365605


namespace NUMINAMATH_CALUDE_selling_price_calculation_l3656_365696

theorem selling_price_calculation (cost_price : ℝ) (profit_percentage : ℝ) :
  cost_price = 180 →
  profit_percentage = 15 →
  cost_price + (cost_price * (profit_percentage / 100)) = 207 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l3656_365696


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3656_365647

/-- Given two circles C and D, if an arc of 60° on C has the same length as an arc of 40° on D,
    then the ratio of the area of C to the area of D is 9/4. -/
theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0) 
  (h : C * (60 / 360) = D * (40 / 360)) : 
  (C^2 / D^2 : Real) = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3656_365647


namespace NUMINAMATH_CALUDE_louisa_travel_l3656_365606

/-- Louisa's travel problem -/
theorem louisa_travel (first_day_distance : ℝ) (speed : ℝ) (time_difference : ℝ) 
  (h1 : first_day_distance = 200)
  (h2 : speed = 50)
  (h3 : time_difference = 3)
  (h4 : first_day_distance / speed + time_difference = second_day_distance / speed) :
  second_day_distance = 350 :=
by
  sorry

end NUMINAMATH_CALUDE_louisa_travel_l3656_365606


namespace NUMINAMATH_CALUDE_inequality_proof_l3656_365634

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h1 : a^2 < 16*b*c) (h2 : b^2 < 16*c*a) (h3 : c^2 < 16*a*b) :
  a^2 + b^2 + c^2 < 2*(a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3656_365634


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3656_365651

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 24) :
  1 / x + 1 / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3656_365651


namespace NUMINAMATH_CALUDE_min_difference_of_extreme_points_l3656_365678

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2*x - 1/x - a * log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x + 2*a * log x

theorem min_difference_of_extreme_points (a : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ∈ Set.Icc 0 1 → 
  (∀ x, x ≠ x₁ → x ≠ x₂ → g a x ≥ min (g a x₁) (g a x₂)) →
  g a x₁ - g a x₂ ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_difference_of_extreme_points_l3656_365678


namespace NUMINAMATH_CALUDE_safe_code_count_l3656_365667

/-- The set of digits from 0 to 9 -/
def Digits : Finset ℕ := Finset.range 10

/-- The length of the safe code -/
def CodeLength : ℕ := 4

/-- The set of forbidden first digits -/
def ForbiddenFirstDigits : Finset ℕ := {5, 7}

/-- The number of valid safe codes -/
def ValidCodes : ℕ := 10^CodeLength - ForbiddenFirstDigits.card * 10^(CodeLength - 1)

theorem safe_code_count : ValidCodes = 9900 := by
  sorry

end NUMINAMATH_CALUDE_safe_code_count_l3656_365667


namespace NUMINAMATH_CALUDE_interview_panel_seating_l3656_365665

/-- Represents the number of players in each team --/
structure TeamSizes :=
  (team1 : Nat)
  (team2 : Nat)
  (team3 : Nat)

/-- Calculates the number of seating arrangements for players from different teams
    where teammates must sit together --/
def seatingArrangements (sizes : TeamSizes) : Nat :=
  Nat.factorial 3 * Nat.factorial sizes.team1 * Nat.factorial sizes.team2 * Nat.factorial sizes.team3

/-- Theorem stating that for the given team sizes, there are 1728 seating arrangements --/
theorem interview_panel_seating :
  seatingArrangements ⟨4, 3, 2⟩ = 1728 := by
  sorry

#eval seatingArrangements ⟨4, 3, 2⟩

end NUMINAMATH_CALUDE_interview_panel_seating_l3656_365665


namespace NUMINAMATH_CALUDE_tangent_line_and_positivity_l3656_365689

open Real

noncomputable def f (a x : ℝ) : ℝ := (x - a) * log x + (1/2) * x

theorem tangent_line_and_positivity (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (∀ x : ℝ, (f a x₀ - f a x) = (1/2) * (x₀ - x)) → a = 1) ∧
  ((1/(2*exp 1)) < a ∧ a < 2 * sqrt (exp 1) → 
    ∀ x : ℝ, x > 0 → f a x > 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_positivity_l3656_365689


namespace NUMINAMATH_CALUDE_alternating_coloring_uniform_rows_l3656_365655

/-- Represents a color in the pattern -/
inductive Color
| A
| B

/-- Represents the grid of the bracelet -/
def BraceletGrid := Fin 10 → Fin 2 → Color

/-- A coloring function that alternates colors in each column -/
def alternatingColoring : BraceletGrid :=
  fun i j => if j = 0 then Color.A else Color.B

/-- Theorem stating that the alternating coloring results in uniform rows -/
theorem alternating_coloring_uniform_rows :
  (∀ i : Fin 10, alternatingColoring i 0 = Color.A) ∧
  (∀ i : Fin 10, alternatingColoring i 1 = Color.B) := by
  sorry


end NUMINAMATH_CALUDE_alternating_coloring_uniform_rows_l3656_365655


namespace NUMINAMATH_CALUDE_op_times_oq_equals_10_l3656_365670

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + 10 = 0

-- Define line l₁
def line_l₁ (x y k : ℝ) : Prop := y = k * x

-- Define line l₂
def line_l₂ (x y : ℝ) : Prop := 3*x + 2*y + 10 = 0

-- Define the intersection points A and B of circle C and line l₁
def intersection_points (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
  line_l₁ A.1 A.2 k ∧ line_l₁ B.1 B.2 k ∧
  A ≠ B

-- Define point P as the midpoint of AB
def midpoint_P (P A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

-- Define point Q as the intersection of l₁ and l₂
def intersection_Q (Q : ℝ × ℝ) (k : ℝ) : Prop :=
  line_l₁ Q.1 Q.2 k ∧ line_l₂ Q.1 Q.2

-- State the theorem
theorem op_times_oq_equals_10 (k : ℝ) (A B P Q : ℝ × ℝ) :
  intersection_points A B k →
  midpoint_P P A B →
  intersection_Q Q k →
  ‖(P.1, P.2)‖ * ‖(Q.1, Q.2)‖ = 10 :=
sorry

end NUMINAMATH_CALUDE_op_times_oq_equals_10_l3656_365670


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_value_l3656_365688

theorem quadratic_root_implies_k_value (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k - 1 = 0 ∧ x = -1) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_value_l3656_365688


namespace NUMINAMATH_CALUDE_derivative_evaluation_l3656_365693

theorem derivative_evaluation (x : ℝ) (h : x > 0) :
  let F : ℝ → ℝ := λ x => (1 - Real.sqrt x)^2 / x
  let F' : ℝ → ℝ := λ x => -1/x^2 + 1/x^(3/2)
  F' 0.01 = -9000 := by sorry

end NUMINAMATH_CALUDE_derivative_evaluation_l3656_365693


namespace NUMINAMATH_CALUDE_lizzy_scored_67_percent_l3656_365639

/-- Represents the exam scores of four students -/
structure ExamScores where
  max_score : ℕ
  gibi_percent : ℕ
  jigi_percent : ℕ
  mike_percent : ℕ
  average_mark : ℕ

/-- Calculates Lizzy's score as a percentage -/
def lizzy_percent (scores : ExamScores) : ℕ :=
  let total_marks := scores.average_mark * 4
  let others_total := (scores.gibi_percent + scores.jigi_percent + scores.mike_percent) * scores.max_score / 100
  let lizzy_score := total_marks - others_total
  lizzy_score * 100 / scores.max_score

/-- Theorem stating that Lizzy's score is 67% given the conditions -/
theorem lizzy_scored_67_percent (scores : ExamScores)
  (h_max : scores.max_score = 700)
  (h_gibi : scores.gibi_percent = 59)
  (h_jigi : scores.jigi_percent = 55)
  (h_mike : scores.mike_percent = 99)
  (h_avg : scores.average_mark = 490) :
  lizzy_percent scores = 67 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_scored_67_percent_l3656_365639


namespace NUMINAMATH_CALUDE_log_power_equality_l3656_365682

theorem log_power_equality (a N m : ℝ) (ha : a > 0) (hN : N > 0) (hm : m ≠ 0) :
  Real.log N^m / Real.log (a^m) = Real.log N / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_log_power_equality_l3656_365682


namespace NUMINAMATH_CALUDE_x₃_value_l3656_365676

noncomputable def x₃ (x₁ x₂ : ℝ) : ℝ :=
  let y₁ := Real.exp x₁
  let y₂ := Real.exp x₂
  let yC := (2/3) * y₁ + (1/3) * y₂
  Real.log ((2/3) + (1/3) * Real.exp 2)

theorem x₃_value :
  let x₁ : ℝ := 0
  let x₂ : ℝ := 2
  let f : ℝ → ℝ := Real.exp
  x₃ x₁ x₂ = Real.log ((2/3) + (1/3) * Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_x₃_value_l3656_365676


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3656_365668

theorem arithmetic_sequence_middle_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first : a 0 = 13) 
  (h_last : a 4 = 37) : 
  a 2 = 25 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3656_365668


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_sum_product_l3656_365633

theorem cubic_polynomial_root_sum_product (α β γ : ℂ) :
  (α + β + γ = -4) →
  (α * β * γ = 14) →
  (α^3 + 4*α^2 + 5*α - 14 = 0) →
  (β^3 + 4*β^2 + 5*β - 14 = 0) →
  (γ^3 + 4*γ^2 + 5*γ - 14 = 0) →
  ∃ p q : ℂ, (α+β)^3 + p*(α+β)^2 + q*(α+β) + 34 = 0 ∧
            (β+γ)^3 + p*(β+γ)^2 + q*(β+γ) + 34 = 0 ∧
            (γ+α)^3 + p*(γ+α)^2 + q*(γ+α) + 34 = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_sum_product_l3656_365633


namespace NUMINAMATH_CALUDE_work_completion_time_l3656_365673

theorem work_completion_time (x : ℝ) : 
  (x > 0) →  -- A's completion time is positive
  (4 * (1 / x + 1 / 30) = 1 / 3) →  -- Equation from working together
  (x = 20) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3656_365673


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l3656_365601

/-- A parabola is tangent to a line if and only if the discriminant of their difference is zero -/
def is_tangent (a : ℝ) : Prop :=
  (4 : ℝ) - 12 * a = 0

/-- The value of a for which the parabola y = ax^2 + 6 is tangent to the line y = 2x + 3 -/
theorem parabola_tangent_line : ∃ (a : ℝ), is_tangent a ∧ a = (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_line_l3656_365601


namespace NUMINAMATH_CALUDE_women_fair_hair_percentage_l3656_365609

/-- Represents the percentage of fair-haired employees who are women -/
def percent_fair_haired_women : ℝ := 0.40

/-- Represents the percentage of employees who have fair hair -/
def percent_fair_haired : ℝ := 0.25

/-- Represents the percentage of employees who are women with fair hair -/
def percent_women_fair_hair : ℝ := percent_fair_haired_women * percent_fair_haired

theorem women_fair_hair_percentage :
  percent_women_fair_hair = 0.10 := by sorry

end NUMINAMATH_CALUDE_women_fair_hair_percentage_l3656_365609


namespace NUMINAMATH_CALUDE_linear_pair_angle_ratio_l3656_365657

theorem linear_pair_angle_ratio (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Both angles are positive
  a + b = 180 ∧    -- Angles form a linear pair (sum to 180°)
  a = 5 * b →      -- Angles are in ratio 5:1
  b = 30 :=        -- The smaller angle is 30°
by sorry

end NUMINAMATH_CALUDE_linear_pair_angle_ratio_l3656_365657


namespace NUMINAMATH_CALUDE_bill_value_l3656_365653

theorem bill_value (total_money : ℕ) (num_bills : ℕ) (h1 : total_money = 45) (h2 : num_bills = 9) :
  total_money / num_bills = 5 := by
  sorry

end NUMINAMATH_CALUDE_bill_value_l3656_365653


namespace NUMINAMATH_CALUDE_at_least_three_prime_factors_l3656_365672

theorem at_least_three_prime_factors
  (p : Nat)
  (h_prime : Nat.Prime p)
  (h_div : p^2 ∣ 2^(p-1) - 1)
  (n : Nat) :
  ∃ (q₁ q₂ q₃ : Nat),
    Nat.Prime q₁ ∧ Nat.Prime q₂ ∧ Nat.Prime q₃ ∧
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₂ ≠ q₃ ∧
    (q₁ ∣ (p-1) * (Nat.factorial p + 2^n)) ∧
    (q₂ ∣ (p-1) * (Nat.factorial p + 2^n)) ∧
    (q₃ ∣ (p-1) * (Nat.factorial p + 2^n)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_three_prime_factors_l3656_365672


namespace NUMINAMATH_CALUDE_second_number_proof_l3656_365697

theorem second_number_proof (d : ℕ) (n₁ n₂ x : ℕ) : 
  d = 16 →
  n₁ = 25 →
  n₂ = 105 →
  x = 41 →
  x > n₁ →
  x % d = n₁ % d →
  x % d = n₂ % d →
  ∀ y : ℕ, n₁ < y ∧ y < x → y % d ≠ n₁ % d ∨ y % d ≠ n₂ % d :=
by sorry

end NUMINAMATH_CALUDE_second_number_proof_l3656_365697


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l3656_365600

theorem no_real_solution_for_log_equation :
  ¬∃ (x : ℝ), 
    (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 7*x - 18)) ∧ 
    (x + 5 > 0) ∧ (x - 3 > 0) ∧ (x^2 - 7*x - 18 > 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l3656_365600


namespace NUMINAMATH_CALUDE_fractional_sides_eq_seven_l3656_365643

/-- A 3-dimensional polyhedron with fractional sides -/
structure Polyhedron where
  F : ℝ  -- number of fractional sides
  D : ℝ  -- number of diagonals
  h1 : D = 2 * F
  h2 : D = F * (F - 3) / 2

/-- The number of fractional sides in the polyhedron is 7 -/
theorem fractional_sides_eq_seven (P : Polyhedron) : P.F = 7 := by
  sorry

end NUMINAMATH_CALUDE_fractional_sides_eq_seven_l3656_365643


namespace NUMINAMATH_CALUDE_scenario_I_correct_scenario_II_correct_scenario_III_correct_scenario_IV_correct_l3656_365644

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 2

-- Define the total number of people
def total_people : ℕ := num_boys + num_girls

-- Define the arrangement functions for each scenario
def arrangements_I : ℕ := sorry

def arrangements_II : ℕ := sorry

def arrangements_III : ℕ := sorry

def arrangements_IV : ℕ := sorry

-- Theorem for scenario I
theorem scenario_I_correct : 
  arrangements_I = 48 := by sorry

-- Theorem for scenario II
theorem scenario_II_correct : 
  arrangements_II = 36 := by sorry

-- Theorem for scenario III
theorem scenario_III_correct : 
  arrangements_III = 60 := by sorry

-- Theorem for scenario IV
theorem scenario_IV_correct : 
  arrangements_IV = 78 := by sorry

end NUMINAMATH_CALUDE_scenario_I_correct_scenario_II_correct_scenario_III_correct_scenario_IV_correct_l3656_365644


namespace NUMINAMATH_CALUDE_prob_heads_tails_tails_l3656_365650

/-- The probability of getting heads on a fair coin flip -/
def prob_heads : ℚ := 1/2

/-- The probability of getting tails on a fair coin flip -/
def prob_tails : ℚ := 1/2

/-- The number of coin flips -/
def num_flips : ℕ := 3

/-- Theorem: The probability of getting heads on the first flip and tails on the last two flips
    when flipping a fair coin three times is 1/8 -/
theorem prob_heads_tails_tails : prob_heads * prob_tails * prob_tails = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_heads_tails_tails_l3656_365650


namespace NUMINAMATH_CALUDE_cora_cookie_purchase_l3656_365630

/-- The number of cookies Cora purchased each day in April -/
def cookies_per_day : ℕ := 3

/-- The cost of each cookie in dollars -/
def cookie_cost : ℕ := 18

/-- The total amount Cora spent on cookies in April in dollars -/
def total_spent : ℕ := 1620

/-- The number of days in April -/
def days_in_april : ℕ := 30

/-- Theorem stating that Cora purchased 3 cookies each day in April -/
theorem cora_cookie_purchase :
  cookies_per_day * days_in_april * cookie_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_cora_cookie_purchase_l3656_365630


namespace NUMINAMATH_CALUDE_product_of_specific_numbers_l3656_365684

theorem product_of_specific_numbers (x y : ℝ) 
  (h1 : x - y = 6) 
  (h2 : x^3 - y^3 = 198) : 
  x * y = 5 := by
sorry

end NUMINAMATH_CALUDE_product_of_specific_numbers_l3656_365684


namespace NUMINAMATH_CALUDE_perfect_square_3_4_4_6_5_6_l3656_365623

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem perfect_square_3_4_4_6_5_6 :
  is_perfect_square (3^4 * 4^6 * 5^6) :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_square_3_4_4_6_5_6_l3656_365623


namespace NUMINAMATH_CALUDE_ball_arrangement_count_l3656_365619

def number_of_yellow_balls : ℕ := 4
def number_of_red_balls : ℕ := 3
def total_balls : ℕ := number_of_yellow_balls + number_of_red_balls

def arrangement_count : ℕ := Nat.choose total_balls number_of_yellow_balls

theorem ball_arrangement_count :
  arrangement_count = 35 :=
by sorry

end NUMINAMATH_CALUDE_ball_arrangement_count_l3656_365619


namespace NUMINAMATH_CALUDE_alvin_marbles_l3656_365646

theorem alvin_marbles (initial_marbles : ℕ) : 
  initial_marbles - 18 + 25 = 64 → initial_marbles = 57 := by
  sorry

end NUMINAMATH_CALUDE_alvin_marbles_l3656_365646


namespace NUMINAMATH_CALUDE_division_instead_of_multiplication_error_l3656_365615

theorem division_instead_of_multiplication_error (y : ℝ) (h : y > 0) :
  (|8 * y - y / 8| / (8 * y)) * 100 = 98 := by
  sorry

end NUMINAMATH_CALUDE_division_instead_of_multiplication_error_l3656_365615


namespace NUMINAMATH_CALUDE_square_equation_solution_l3656_365656

theorem square_equation_solution : ∃! x : ℤ, (2012 + x)^2 = x^2 ∧ x = -1006 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3656_365656


namespace NUMINAMATH_CALUDE_lidless_cube_box_configurations_l3656_365622

/-- Represents a cube-shaped box without a lid -/
structure LidlessCubeBox where
  -- Add necessary fields if needed

/-- The number of possible planar unfolding configurations for a lidless cube box -/
def planarUnfoldingConfigurations (box : LidlessCubeBox) : ℕ := sorry

/-- Theorem: The number of possible planar unfolding configurations for a lidless cube box is 8 -/
theorem lidless_cube_box_configurations :
  ∀ (box : LidlessCubeBox), planarUnfoldingConfigurations box = 8 := by
  sorry

end NUMINAMATH_CALUDE_lidless_cube_box_configurations_l3656_365622


namespace NUMINAMATH_CALUDE_magnet_area_theorem_l3656_365675

/-- Represents a rectangular magnet with length and width in centimeters. -/
structure Magnet where
  length : ℝ
  width : ℝ

/-- Calculates the area of a magnet in square centimeters. -/
def area (m : Magnet) : ℝ := m.length * m.width

/-- Calculates the circumference of two identical magnets attached horizontally. -/
def totalCircumference (m : Magnet) : ℝ := 2 * (2 * m.length + 2 * m.width)

/-- Theorem: Given two identical rectangular magnets with a total circumference of 70 cm
    and a total length of 15 cm when attached horizontally, the area of one magnet is 150 cm². -/
theorem magnet_area_theorem (m : Magnet) 
    (h1 : totalCircumference m = 70)
    (h2 : 2 * m.length = 15) : 
  area m = 150 := by
  sorry

end NUMINAMATH_CALUDE_magnet_area_theorem_l3656_365675


namespace NUMINAMATH_CALUDE_equation_solutions_l3656_365645

def satisfies_equation (x y z : ℕ) : Prop :=
  x^2 + y^2 = 9 + z^2 - 2*x*y

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(0,5,4), (1,4,4), (2,3,4), (3,2,4), (4,1,4), (5,0,4), (0,3,0), (1,2,0), (2,1,0), (3,0,0)}

theorem equation_solutions :
  ∀ x y z : ℕ, satisfies_equation x y z ↔ (x, y, z) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3656_365645


namespace NUMINAMATH_CALUDE_determinant_scaling_l3656_365637

theorem determinant_scaling {x y z w : ℝ} (h : Matrix.det !![x, y; z, w] = 3) :
  Matrix.det !![3*x, 3*y; 3*z, 3*w] = 27 := by sorry

end NUMINAMATH_CALUDE_determinant_scaling_l3656_365637


namespace NUMINAMATH_CALUDE_dog_age_difference_l3656_365679

/-- The age difference between the 1st and 2nd fastest dogs -/
def age_difference (d1 d2 d3 d4 d5 : ℕ) : ℕ := d1 - d2

theorem dog_age_difference :
  ∀ d1 d2 d3 d4 d5 : ℕ,
  (d1 + d5) / 2 = 18 →  -- Average age of 1st and 5th dogs
  d1 = 10 →             -- Age of 1st dog
  d2 = d1 - 2 →         -- Age of 2nd dog
  d3 = d2 + 4 →         -- Age of 3rd dog
  d4 * 2 = d3 →         -- Age of 4th dog
  d5 = d4 + 20 →        -- Age of 5th dog
  age_difference d1 d2 = 2 := by
sorry

end NUMINAMATH_CALUDE_dog_age_difference_l3656_365679


namespace NUMINAMATH_CALUDE_gcd_n_cubed_plus_16_and_n_plus_3_l3656_365642

theorem gcd_n_cubed_plus_16_and_n_plus_3 (n : ℕ) (h : n > 8) :
  Nat.gcd (n^3 + 16) (n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cubed_plus_16_and_n_plus_3_l3656_365642


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l3656_365674

theorem range_of_a_minus_b (a b : ℝ) (ha : -1 < a ∧ a < 2) (hb : -2 < b ∧ b < 1) :
  ∃ x, -2 < x ∧ x < 4 ∧ ∃ a' b', -1 < a' ∧ a' < 2 ∧ -2 < b' ∧ b' < 1 ∧ x = a' - b' :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l3656_365674


namespace NUMINAMATH_CALUDE_weight_of_raisins_l3656_365629

/-- Given the total weight of snacks and the weight of peanuts, 
    prove that the weight of raisins is 0.4 pounds. -/
theorem weight_of_raisins (total_weight peanuts_weight : ℝ) 
  (h1 : total_weight = 0.5)
  (h2 : peanuts_weight = 0.1) : 
  total_weight - peanuts_weight = 0.4 := by
sorry

end NUMINAMATH_CALUDE_weight_of_raisins_l3656_365629


namespace NUMINAMATH_CALUDE_volume_central_region_is_one_sixth_l3656_365641

/-- Represents a unit cube in 3D space -/
structure UnitCube where
  -- Add necessary fields/axioms for a unit cube

/-- Represents a plane in 3D space -/
structure Plane where
  -- Add necessary fields/axioms for a plane

/-- Represents the central region (regular octahedron) formed by intersecting planes -/
structure CentralRegion where
  cube : UnitCube
  intersecting_planes : List Plane
  -- Add necessary conditions to ensure the planes intersect at midpoints of edges

/-- Calculate the volume of the central region in a unit cube intersected by specific planes -/
def volume_central_region (region : CentralRegion) : ℝ :=
  sorry

/-- Theorem stating that the volume of the central region is 1/6 -/
theorem volume_central_region_is_one_sixth (region : CentralRegion) :
  volume_central_region region = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_volume_central_region_is_one_sixth_l3656_365641


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3656_365620

theorem fraction_equivalence : 
  let n : ℚ := 13/2
  (4 + n) / (7 + n) = 7/9 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3656_365620


namespace NUMINAMATH_CALUDE_y_squared_plus_reciprocal_l3656_365671

theorem y_squared_plus_reciprocal (x : ℝ) (a : ℕ) (h1 : x + 1/x = 3) (h2 : a ≠ 1) (h3 : a > 0) :
  let y := x^a
  y^2 + 1/y^2 = (x^2 + 1/x^2)^a - 2*a := by
sorry

end NUMINAMATH_CALUDE_y_squared_plus_reciprocal_l3656_365671


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_equals_circle_diameters_sum_l3656_365664

/-- 
A right triangle with inscribed and circumscribed circles.
-/
structure RightTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Radii of inscribed and circumscribed circles
  r : ℝ
  R : ℝ
  -- Conditions
  right_angle : c^2 = a^2 + b^2
  c_is_diameter : c = 2 * R
  nonneg : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < r ∧ 0 < R

/-- 
In a right triangle, the sum of the legs is equal to the sum of 
the diameters of the inscribed and circumscribed circles.
-/
theorem right_triangle_leg_sum_equals_circle_diameters_sum 
  (t : RightTriangle) : t.a + t.b = 2 * t.R + 2 * t.r := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_equals_circle_diameters_sum_l3656_365664


namespace NUMINAMATH_CALUDE_quadratic_function_m_condition_l3656_365686

/-- A function f: ℝ → ℝ is quadratic if it can be written as f(x) = ax² + bx + c where a ≠ 0 -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function y = (m+1)x² + 2x + 1 -/
def f (m : ℝ) : ℝ → ℝ := λ x ↦ (m + 1) * x^2 + 2 * x + 1

theorem quadratic_function_m_condition :
  ∀ m : ℝ, is_quadratic (f m) ↔ m ≠ -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_m_condition_l3656_365686


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3656_365654

/-- The quadratic equation (m-3)x^2 - 2x + 1 = 0 has real roots if and only if m ≤ 4 and m ≠ 3. -/
theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3656_365654


namespace NUMINAMATH_CALUDE_leftover_value_is_9_65_l3656_365608

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The number of quarters in a full roll -/
def quarters_per_roll : ℕ := 42

/-- The number of dimes in a full roll -/
def dimes_per_roll : ℕ := 48

/-- Gary's quarters -/
def gary_quarters : ℕ := 127

/-- Gary's dimes -/
def gary_dimes : ℕ := 212

/-- Kim's quarters -/
def kim_quarters : ℕ := 158

/-- Kim's dimes -/
def kim_dimes : ℕ := 297

/-- Theorem: The value of leftover quarters and dimes is $9.65 -/
theorem leftover_value_is_9_65 :
  let total_quarters := gary_quarters + kim_quarters
  let total_dimes := gary_dimes + kim_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  leftover_quarters * quarter_value + leftover_dimes * dime_value = 9.65 := by
  sorry

end NUMINAMATH_CALUDE_leftover_value_is_9_65_l3656_365608


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_10001_l3656_365660

theorem largest_prime_factor_of_10001 : ∃ p : ℕ, 
  p.Prime ∧ p ∣ 10001 ∧ ∀ q : ℕ, q.Prime → q ∣ 10001 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_10001_l3656_365660


namespace NUMINAMATH_CALUDE_gcd_7800_360_minus_20_l3656_365648

theorem gcd_7800_360_minus_20 : Nat.gcd 7800 360 - 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7800_360_minus_20_l3656_365648


namespace NUMINAMATH_CALUDE_min_value_my_plus_nx_l3656_365602

theorem min_value_my_plus_nx (m n x y : ℝ) 
  (h1 : m^2 + n^2 = 1) (h2 : x^2 + y^2 = 4) : 
  ∀ z : ℝ, m * y + n * x ≥ z → z ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_my_plus_nx_l3656_365602


namespace NUMINAMATH_CALUDE_custom_op_equality_l3656_365613

/-- Custom binary operation on real numbers -/
def custom_op (a b : ℝ) : ℝ := a * b + a - b

/-- Theorem stating the equality for the given expression -/
theorem custom_op_equality (a b : ℝ) :
  custom_op a b + custom_op (b - a) b = b^2 - b :=
by sorry

end NUMINAMATH_CALUDE_custom_op_equality_l3656_365613


namespace NUMINAMATH_CALUDE_sum_of_roots_l3656_365680

theorem sum_of_roots (a b : ℝ) (ha : a * (a - 4) = 5) (hb : b * (b - 4) = 5) (hab : a ≠ b) : a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3656_365680


namespace NUMINAMATH_CALUDE_area_of_ABCD_l3656_365663

/-- Represents a rectangle with length and height -/
structure Rectangle where
  length : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.height

/-- Theorem: Area of rectangle ABCD -/
theorem area_of_ABCD (r1 r2 r3 : Rectangle) (ABCD : Rectangle) :
  area r1 + area r2 + area r3 = area ABCD →
  area r1 = 2 →
  ABCD.length = 5 →
  ABCD.height = 3 →
  area ABCD = 15 := by
  sorry

#check area_of_ABCD

end NUMINAMATH_CALUDE_area_of_ABCD_l3656_365663


namespace NUMINAMATH_CALUDE_multiple_of_larger_integer_l3656_365687

theorem multiple_of_larger_integer (s l : ℤ) (m : ℚ) : 
  s + l = 30 →
  s = 10 →
  m * l = 5 * s - 10 →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_larger_integer_l3656_365687


namespace NUMINAMATH_CALUDE_existence_implies_bound_l3656_365638

theorem existence_implies_bound :
  (∃ (m : ℝ), ∃ (x : ℝ), 4^x + m * 2^x + 1 = 0) →
  (∀ (m : ℝ), (∃ (x : ℝ), 4^x + m * 2^x + 1 = 0) → m ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_existence_implies_bound_l3656_365638


namespace NUMINAMATH_CALUDE_school_capacity_l3656_365681

theorem school_capacity (total_classrooms : ℕ) 
  (desks_type1 desks_type2 desks_type3 : ℕ) : 
  total_classrooms = 30 →
  desks_type1 = 40 →
  desks_type2 = 35 →
  desks_type3 = 28 →
  (total_classrooms / 5 * desks_type1 + 
   total_classrooms / 3 * desks_type2 + 
   (total_classrooms - total_classrooms / 5 - total_classrooms / 3) * desks_type3) = 982 :=
by
  sorry

#check school_capacity

end NUMINAMATH_CALUDE_school_capacity_l3656_365681


namespace NUMINAMATH_CALUDE_pencils_in_drawer_l3656_365603

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 72 - 45

/-- The number of pencils Nancy added to the drawer -/
def added_pencils : ℕ := 45

/-- The total number of pencils in the drawer after Nancy added more -/
def total_pencils : ℕ := 72

theorem pencils_in_drawer :
  initial_pencils + added_pencils = total_pencils :=
by sorry

end NUMINAMATH_CALUDE_pencils_in_drawer_l3656_365603


namespace NUMINAMATH_CALUDE_andrew_expenses_l3656_365677

def game_night_expenses (game_count : Nat) 
  (game_cost_1 : Nat) (game_count_1 : Nat)
  (game_cost_2 : Nat) (game_count_2 : Nat)
  (game_cost_3 : Nat) (game_count_3 : Nat)
  (snack_cost : Nat) (drink_cost : Nat) : Nat :=
  game_cost_1 * game_count_1 + game_cost_2 * game_count_2 + game_cost_3 * game_count_3 + snack_cost + drink_cost

theorem andrew_expenses : 
  game_night_expenses 7 900 3 1250 2 1500 2 2500 2000 = 12700 := by
  sorry

end NUMINAMATH_CALUDE_andrew_expenses_l3656_365677


namespace NUMINAMATH_CALUDE_polynomial_division_l3656_365685

-- Define the polynomial that can be divided by x^2 + 3x - 4
def is_divisible (a b c : ℝ) : Prop :=
  ∃ (q : ℝ → ℝ), ∀ x, x^3 + a*x^2 + b*x + c = (x^2 + 3*x - 4) * q x

-- Main theorem
theorem polynomial_division (a b c : ℝ) 
  (h : is_divisible a b c) : 
  (4*a + c = 12) ∧ 
  (2*a - 2*b - c = 14) ∧ 
  (∀ (a' b' c' : ℤ), (is_divisible (a' : ℝ) (b' : ℝ) (c' : ℝ)) → 
    c' ≥ a' ∧ a' > 1 → a' = 2 ∧ b' = -7 ∧ c' = 4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_l3656_365685


namespace NUMINAMATH_CALUDE_kendra_hat_purchase_l3656_365610

theorem kendra_hat_purchase (toy_price hat_price initial_money change toys_bought : ℕ) 
  (h1 : toy_price = 20)
  (h2 : hat_price = 10)
  (h3 : initial_money = 100)
  (h4 : toys_bought = 2)
  (h5 : change = 30) :
  (initial_money - change - toy_price * toys_bought) / hat_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_kendra_hat_purchase_l3656_365610


namespace NUMINAMATH_CALUDE_non_green_car_probability_l3656_365659

/-- The probability of selecting a non-green car from a set of 60 cars with 30 green cars is 1/2 -/
theorem non_green_car_probability (total_cars : ℕ) (green_cars : ℕ) 
  (h1 : total_cars = 60) 
  (h2 : green_cars = 30) : 
  (total_cars - green_cars : ℚ) / total_cars = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_non_green_car_probability_l3656_365659


namespace NUMINAMATH_CALUDE_dark_tile_fraction_is_five_sixteenths_l3656_365692

/-- Represents a tiled floor with a repeating pattern of dark tiles -/
structure TiledFloor :=
  (size : ℕ)  -- Size of the square floor (number of tiles per side)
  (dark_tiles_per_section : ℕ)  -- Number of dark tiles in each 4x4 section
  (total_tiles_per_section : ℕ)  -- Total number of tiles in each 4x4 section

/-- The fraction of dark tiles on the floor -/
def dark_tile_fraction (floor : TiledFloor) : ℚ :=
  (floor.dark_tiles_per_section : ℚ) / (floor.total_tiles_per_section : ℚ)

/-- Theorem stating that the fraction of dark tiles is 5/16 -/
theorem dark_tile_fraction_is_five_sixteenths (floor : TiledFloor) 
  (h1 : floor.size > 0)
  (h2 : floor.dark_tiles_per_section = 5)
  (h3 : floor.total_tiles_per_section = 16) : 
  dark_tile_fraction floor = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_is_five_sixteenths_l3656_365692


namespace NUMINAMATH_CALUDE_car_distance_traveled_l3656_365666

-- Define constants
def tire_diameter : ℝ := 15
def revolutions : ℝ := 672.1628045157456
def inches_per_mile : ℝ := 63360

-- Define the theorem
theorem car_distance_traveled (ε : ℝ) (h_ε : ε > 0) :
  ∃ (distance : ℝ), 
    abs (distance - 0.5) < ε ∧ 
    distance = (π * tire_diameter * revolutions) / inches_per_mile :=
sorry

end NUMINAMATH_CALUDE_car_distance_traveled_l3656_365666


namespace NUMINAMATH_CALUDE_class_size_l3656_365691

/-- The number of students who borrowed at least 3 books -/
def R : ℕ := 16

/-- The total number of students in the class -/
def S : ℕ := 42

theorem class_size :
  (∃ (R : ℕ),
    (0 * 2 + 1 * 12 + 2 * 12 + 3 * R) / S = 2 ∧
    S = 2 + 12 + 12 + R) →
  S = 42 := by sorry

end NUMINAMATH_CALUDE_class_size_l3656_365691


namespace NUMINAMATH_CALUDE_no_quadratic_with_discriminant_23_l3656_365621

theorem no_quadratic_with_discriminant_23 :
  ¬ ∃ (a b c : ℤ), b^2 - 4*a*c = 23 := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_with_discriminant_23_l3656_365621


namespace NUMINAMATH_CALUDE_chord_length_perpendicular_bisector_l3656_365669

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 15) :
  let c := 2 * r * Real.sqrt 3 / 2
  c = 26 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_chord_length_perpendicular_bisector_l3656_365669


namespace NUMINAMATH_CALUDE_distribute_five_balls_two_boxes_l3656_365649

/-- The number of ways to distribute n indistinguishable balls into 2 indistinguishable boxes -/
def distribute_balls (n : ℕ) : ℕ :=
  (n + 2) / 2

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_five_balls_two_boxes : distribute_balls 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_two_boxes_l3656_365649


namespace NUMINAMATH_CALUDE_jacob_dinner_calories_l3656_365636

/-- Calculates Jacob's dinner calories based on his daily goal, breakfast, lunch, and excess calories --/
theorem jacob_dinner_calories
  (daily_goal : ℕ)
  (breakfast : ℕ)
  (lunch : ℕ)
  (excess : ℕ)
  (h1 : daily_goal = 1800)
  (h2 : breakfast = 400)
  (h3 : lunch = 900)
  (h4 : excess = 600) :
  daily_goal + excess - (breakfast + lunch) = 1100 :=
by sorry

end NUMINAMATH_CALUDE_jacob_dinner_calories_l3656_365636


namespace NUMINAMATH_CALUDE_fibonacci_identity_l3656_365699

/-- Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Statement of the problem -/
theorem fibonacci_identity (k : ℤ) :
  (fib 785 + k) * (fib 787 + k) - (fib 786 + k)^2 = -1 := by
  sorry


end NUMINAMATH_CALUDE_fibonacci_identity_l3656_365699


namespace NUMINAMATH_CALUDE_star_three_five_l3656_365683

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

-- Theorem statement
theorem star_three_five : star 3 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_star_three_five_l3656_365683


namespace NUMINAMATH_CALUDE_abs_eq_iff_eq_l3656_365658

theorem abs_eq_iff_eq (x y : ℝ) : 
  (|x| = |y| → x = y) ↔ False ∧ 
  (x = y → |x| = |y|) :=
sorry

end NUMINAMATH_CALUDE_abs_eq_iff_eq_l3656_365658


namespace NUMINAMATH_CALUDE_collinear_vectors_t_value_l3656_365624

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem collinear_vectors_t_value 
  (h_non_collinear : ¬ ∃ (k : ℝ), a = k • b) 
  (h_collinear : ∃ (k : ℝ), a - t • b = k • (2 • a + b)) : 
  t = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_t_value_l3656_365624


namespace NUMINAMATH_CALUDE_division_problem_l3656_365612

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2 / 5) : 
  c / a = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3656_365612


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_equals_two_when_solution_set_is_x_leq_neg_one_l3656_365618

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Part 1
theorem solution_set_when_a_is_one (a : ℝ) (h : a = 1) :
  {x : ℝ | f a x ≥ 3 * x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} :=
sorry

-- Part 2
theorem a_equals_two_when_solution_set_is_x_leq_neg_one (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -1}) → a = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_equals_two_when_solution_set_is_x_leq_neg_one_l3656_365618


namespace NUMINAMATH_CALUDE_catering_weight_calculation_l3656_365611

/-- Calculates the total weight of catering items for an event --/
theorem catering_weight_calculation (
  silverware_weight : ℕ)
  (silverware_per_setting : ℕ)
  (plate_weight : ℕ)
  (plates_per_setting : ℕ)
  (glass_weight : ℕ)
  (glasses_per_setting : ℕ)
  (decoration_weight : ℕ)
  (num_tables : ℕ)
  (settings_per_table : ℕ)
  (backup_settings : ℕ)
  (decoration_per_table : ℕ)
  (h1 : silverware_weight = 4)
  (h2 : silverware_per_setting = 3)
  (h3 : plate_weight = 12)
  (h4 : plates_per_setting = 2)
  (h5 : glass_weight = 8)
  (h6 : glasses_per_setting = 2)
  (h7 : decoration_weight = 16)
  (h8 : num_tables = 15)
  (h9 : settings_per_table = 8)
  (h10 : backup_settings = 20)
  (h11 : decoration_per_table = 1) :
  (num_tables * settings_per_table + backup_settings) *
    (silverware_weight * silverware_per_setting +
     plate_weight * plates_per_setting +
     glass_weight * glasses_per_setting) +
  num_tables * decoration_weight * decoration_per_table = 7520 := by
  sorry

end NUMINAMATH_CALUDE_catering_weight_calculation_l3656_365611


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3656_365631

def tshirt_cost : ℝ := 9.95
def number_of_tshirts : ℕ := 20

theorem total_cost_calculation : 
  tshirt_cost * (number_of_tshirts : ℝ) = 199.00 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3656_365631


namespace NUMINAMATH_CALUDE_matrix_product_proof_l3656_365616

theorem matrix_product_proof :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 0, -4]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; -2, 2]
  A * B = !![19, -7; 8, -8] := by sorry

end NUMINAMATH_CALUDE_matrix_product_proof_l3656_365616


namespace NUMINAMATH_CALUDE_price_increase_percentage_l3656_365662

theorem price_increase_percentage (original_price : ℝ) (increase_rate : ℝ) : 
  original_price = 200 →
  increase_rate = 0.1 →
  let new_price := original_price * (1 + increase_rate)
  (new_price - original_price) / original_price = increase_rate :=
by sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l3656_365662


namespace NUMINAMATH_CALUDE_min_bills_for_payment_l3656_365632

/-- Represents the number of bills of each denomination --/
structure Bills :=
  (tens : ℕ)
  (fives : ℕ)
  (ones : ℕ)

/-- Calculates the total value of the bills --/
def billsValue (b : Bills) : ℕ :=
  10 * b.tens + 5 * b.fives + b.ones

/-- Checks if a given number of bills is valid for the payment --/
def isValidPayment (b : Bills) (amount : ℕ) : Prop :=
  b.tens ≤ 13 ∧ b.fives ≤ 11 ∧ b.ones ≤ 17 ∧ billsValue b = amount

/-- Counts the total number of bills --/
def totalBills (b : Bills) : ℕ :=
  b.tens + b.fives + b.ones

/-- The main theorem stating that the minimum number of bills required is 16 --/
theorem min_bills_for_payment :
  ∃ (b : Bills), isValidPayment b 128 ∧
  ∀ (b' : Bills), isValidPayment b' 128 → totalBills b ≤ totalBills b' :=
by sorry

end NUMINAMATH_CALUDE_min_bills_for_payment_l3656_365632


namespace NUMINAMATH_CALUDE_function_property_l3656_365695

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 3) = -f x)
  (h_f1 : f 1 = 1) :
  f 2015 + f 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3656_365695


namespace NUMINAMATH_CALUDE_jones_elementary_population_l3656_365635

theorem jones_elementary_population :
  ∀ (total_students : ℕ) (boys_percentage : ℚ),
    boys_percentage = 30 / 100 →
    (boys_percentage * total_students : ℚ).num = 90 →
    total_students = 300 :=
by sorry

end NUMINAMATH_CALUDE_jones_elementary_population_l3656_365635


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l3656_365604

theorem min_x_prime_factorization_sum (x y : ℕ+) (h : 5 * x^7 = 13 * y^11) :
  ∃ (a b c d : ℕ),
    (∀ x' : ℕ+, 5 * x'^7 = 13 * y^11 → x' ≥ x) →
    x = a^c * b^d ∧
    Prime a ∧ Prime b ∧
    a + b + c + d = 62 := by
  sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_sum_l3656_365604


namespace NUMINAMATH_CALUDE_smallest_n_is_eight_l3656_365661

/-- A geometric sequence (a_n) with given conditions -/
def geometric_sequence (x : ℝ) (a : ℕ → ℝ) : Prop :=
  x > 0 ∧
  a 1 = Real.exp x ∧
  a 2 = x ∧
  a 3 = Real.log x ∧
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = r * a n

/-- The smallest n for which a_n = 2x is 8 -/
theorem smallest_n_is_eight (x : ℝ) (a : ℕ → ℝ) 
  (h : geometric_sequence x a) : 
  (∃ n : ℕ, n ≥ 1 ∧ a n = 2 * x) ∧ 
  (∀ m : ℕ, m ≥ 1 ∧ m < 8 → a m ≠ 2 * x) ∧ 
  a 8 = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_is_eight_l3656_365661


namespace NUMINAMATH_CALUDE_regular_price_of_bread_l3656_365607

/-- The regular price of a full pound of bread, given sale conditions -/
theorem regular_price_of_bread (sale_price : ℝ) (discount_rate : ℝ) : 
  sale_price = 2 →
  discount_rate = 0.6 →
  ∃ (regular_price : ℝ), 
    regular_price = 20 ∧ 
    sale_price = (1 - discount_rate) * (regular_price / 4) :=
by sorry

end NUMINAMATH_CALUDE_regular_price_of_bread_l3656_365607


namespace NUMINAMATH_CALUDE_right_triangle_angle_sum_l3656_365690

theorem right_triangle_angle_sum (A B C : Real) : 
  (A + B + C = 180) → (C = 90) → (B = 55) → (A = 35) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_sum_l3656_365690
