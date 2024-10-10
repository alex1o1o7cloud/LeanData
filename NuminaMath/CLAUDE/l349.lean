import Mathlib

namespace linear_independence_of_polynomial_basis_l349_34990

theorem linear_independence_of_polynomial_basis :
  ∀ (α₁ α₂ α₃ α₄ : ℝ),
  (∀ x : ℝ, α₁ + α₂ * x + α₃ * x^2 + α₄ * x^3 = 0) →
  (α₁ = 0 ∧ α₂ = 0 ∧ α₃ = 0 ∧ α₄ = 0) :=
by sorry

end linear_independence_of_polynomial_basis_l349_34990


namespace gold_coin_distribution_l349_34947

theorem gold_coin_distribution (x y : ℕ) (h1 : x + y = 16) (h2 : x ≠ y) :
  ∃ k : ℕ, x^2 - y^2 = k * (x - y) → k = 16 := by
  sorry

end gold_coin_distribution_l349_34947


namespace lines_coplanar_iff_k_eq_neg_one_or_neg_one_third_l349_34924

-- Define the two lines
def line1 (r : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (-2 + r, 5 - 3*k*r, k*r)
def line2 (t : ℝ) : ℝ × ℝ × ℝ := (2*t, 2 + 2*t, -2*t)

-- Define coplanarity
def coplanar (l1 l2 : ℝ → ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b c d : ℝ), ∀ (t s : ℝ),
    a * (l1 t).1 + b * (l1 t).2.1 + c * (l1 t).2.2 + d =
    a * (l2 s).1 + b * (l2 s).2.1 + c * (l2 s).2.2 + d

-- Theorem statement
theorem lines_coplanar_iff_k_eq_neg_one_or_neg_one_third :
  ∀ k : ℝ, coplanar (line1 · k) line2 ↔ k = -1 ∨ k = -1/3 :=
sorry

end lines_coplanar_iff_k_eq_neg_one_or_neg_one_third_l349_34924


namespace green_ribbons_count_l349_34926

theorem green_ribbons_count (total : ℕ) 
  (h_red : (1 : ℚ) / 4 * total = total / 4)
  (h_blue : (3 : ℚ) / 8 * total = 3 * total / 8)
  (h_green : (1 : ℚ) / 8 * total = total / 8)
  (h_white : total - (total / 4 + 3 * total / 8 + total / 8) = 36) :
  total / 8 = 18 := by
  sorry

end green_ribbons_count_l349_34926


namespace monotonic_f_implies_a_range_l349_34963

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x - 2 else Real.log x / Real.log a

-- State the theorem
theorem monotonic_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a ≤ 3 :=
by sorry

end monotonic_f_implies_a_range_l349_34963


namespace sufficient_not_necessary_l349_34921

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem sufficient_not_necessary
  (f : ℝ → ℝ) (h : OddFunction f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ + f x₂ = 0) ∧
  (∃ x₁ x₂ : ℝ, f x₁ + f x₂ = 0 ∧ x₁ + x₂ ≠ 0) :=
by sorry

end sufficient_not_necessary_l349_34921


namespace quadratic_inequality_solution_set_l349_34998

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 + 2*x - 3 < 0 ↔ -3 < x ∧ x < 1 := by
sorry

end quadratic_inequality_solution_set_l349_34998


namespace peach_pies_l349_34985

theorem peach_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio peach_ratio : ℕ) : 
  total_pies = 36 →
  apple_ratio = 1 →
  blueberry_ratio = 4 →
  cherry_ratio = 3 →
  peach_ratio = 2 →
  (peach_ratio : ℚ) * total_pies / (apple_ratio + blueberry_ratio + cherry_ratio + peach_ratio) = 8 := by
  sorry

#check peach_pies

end peach_pies_l349_34985


namespace missing_score_is_86_l349_34954

def recorded_scores : List ℝ := [81, 73, 83, 73]
def mean : ℝ := 79.2
def total_games : ℕ := 5

theorem missing_score_is_86 :
  let total_sum := mean * total_games
  let recorded_sum := recorded_scores.sum
  total_sum - recorded_sum = 86 := by
  sorry

end missing_score_is_86_l349_34954


namespace square_side_length_l349_34930

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 12) (h2 : side > 0) (h3 : area = side ^ 2) :
  side = 2 * Real.sqrt 3 := by
sorry

end square_side_length_l349_34930


namespace gunther_cleaning_time_l349_34906

/-- Gunther's apartment cleaning problem -/
theorem gunther_cleaning_time (free_time : ℕ) (vacuum_time : ℕ) (dust_time : ℕ) (mop_time : ℕ) 
  (num_cats : ℕ) (remaining_time : ℕ) : 
  free_time = 3 * 60 → 
  vacuum_time = 45 →
  dust_time = 60 →
  mop_time = 30 →
  num_cats = 3 →
  remaining_time = 30 →
  (free_time - remaining_time - vacuum_time - dust_time - mop_time) / num_cats = 5 := by
  sorry

end gunther_cleaning_time_l349_34906


namespace cost_per_cow_l349_34908

/-- Calculates the cost per cow given Timothy's expenses --/
theorem cost_per_cow (land_acres : ℕ) (land_cost_per_acre : ℕ)
  (house_cost : ℕ) (num_cows : ℕ) (num_chickens : ℕ)
  (chicken_cost : ℕ) (solar_install_hours : ℕ)
  (solar_install_rate : ℕ) (solar_equipment_cost : ℕ)
  (total_cost : ℕ) :
  land_acres = 30 →
  land_cost_per_acre = 20 →
  house_cost = 120000 →
  num_cows = 20 →
  num_chickens = 100 →
  chicken_cost = 5 →
  solar_install_hours = 6 →
  solar_install_rate = 100 →
  solar_equipment_cost = 6000 →
  total_cost = 147700 →
  (total_cost - (land_acres * land_cost_per_acre + house_cost + 
    num_chickens * chicken_cost + 
    solar_install_hours * solar_install_rate + solar_equipment_cost)) / num_cows = 1000 :=
by sorry

end cost_per_cow_l349_34908


namespace custom_op_solution_l349_34911

/-- Custom operation for integers -/
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that if y12 = 110 using the custom operation, then y = 11 -/
theorem custom_op_solution :
  ∀ y : ℤ, customOp y 12 = 110 → y = 11 := by
  sorry

end custom_op_solution_l349_34911


namespace triangle_ratio_l349_34957

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points N, D, E, F
variable (N D E F : ℝ × ℝ)

-- Define the conditions
variable (h1 : N = ((A.1 + C.1)/2, (A.2 + C.2)/2))  -- N is midpoint of AC
variable (h2 : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 100) -- AB = 10
variable (h3 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 324) -- BC = 18
variable (h4 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1-t) * C.1, t * B.2 + (1-t) * C.2)) -- D on BC
variable (h5 : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (s * A.1 + (1-s) * B.1, s * A.2 + (1-s) * B.2)) -- E on AB
variable (h6 : ∃ r u : ℝ, F = (r * D.1 + (1-r) * E.1, r * D.2 + (1-r) * E.2) ∧
                          F = (u * A.1 + (1-u) * N.1, u * A.2 + (1-u) * N.2)) -- F is intersection of DE and AN
variable (h7 : (D.1 - B.1)^2 + (D.2 - B.2)^2 = 9 * ((E.1 - B.1)^2 + (E.2 - B.2)^2)) -- BD = 3BE

-- Theorem statement
theorem triangle_ratio :
  (F.1 - D.1)^2 + (F.2 - D.2)^2 = 1/9 * ((F.1 - E.1)^2 + (F.2 - E.2)^2) :=
sorry

end triangle_ratio_l349_34957


namespace system_1_solution_system_2_solution_l349_34983

-- System 1
theorem system_1_solution :
  ∃ (x y : ℝ), 2 * x + 3 * y = 9 ∧ x = 2 * y + 1 ∧ x = 3 ∧ y = 1 := by sorry

-- System 2
theorem system_2_solution :
  ∃ (x y : ℝ), 2 * x - y = 6 ∧ 3 * x + 2 * y = 2 ∧ x = 2 ∧ y = -2 := by sorry

end system_1_solution_system_2_solution_l349_34983


namespace cricket_bat_profit_l349_34974

/-- Calculates the profit amount for a cricket bat sale -/
theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 850 ∧ profit_percentage = 36 →
  (selling_price - selling_price / (1 + profit_percentage / 100)) = 225 := by
sorry

end cricket_bat_profit_l349_34974


namespace tan_105_degrees_l349_34929

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l349_34929


namespace binomial_expected_value_and_variance_l349_34900

/-- A random variable following a binomial distribution with n trials and probability p -/
def binomial_distribution (n : ℕ) (p : ℝ) : Type := Unit

variable (ξ : binomial_distribution 200 0.01)

/-- The expected value of a binomial distribution -/
def expected_value (n : ℕ) (p : ℝ) : ℝ := n * p

/-- The variance of a binomial distribution -/
def variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_expected_value_and_variance :
  expected_value 200 0.01 = 2 ∧ variance 200 0.01 = 1.98 := by sorry

end binomial_expected_value_and_variance_l349_34900


namespace geometric_sequence_third_term_l349_34937

theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (first_term : a 1 = 5) 
  (fifth_term : a 5 = 2025) :
  a 3 = 225 := by
sorry

end geometric_sequence_third_term_l349_34937


namespace min_value_theorem_l349_34938

theorem min_value_theorem (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 1 ∧ y > 0 ∧ x + y = 2 → 1 / (x - 1) + 1 / y ≥ 1 / (a - 1) + 1 / b) ∧
  1 / (a - 1) + 1 / b = 4 :=
sorry

end min_value_theorem_l349_34938


namespace complex_number_in_fourth_quadrant_l349_34972

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - I) / (1 + I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_fourth_quadrant_l349_34972


namespace simplify_and_evaluate_l349_34995

theorem simplify_and_evaluate :
  let x : ℚ := -1
  let y : ℚ := 1
  let expr := (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y)
  expr = -x^2 + 3*y^2 ∧ expr = 2 := by
sorry

end simplify_and_evaluate_l349_34995


namespace seating_arrangements_5_total_arrangements_l349_34923

/-- Defines the number of seating arrangements for n people -/
def seating_arrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => seating_arrangements (n + 1) + seating_arrangements n

/-- Theorem stating that the number of seating arrangements for 5 people is 8 -/
theorem seating_arrangements_5 : seating_arrangements 5 = 8 := by sorry

/-- Theorem stating that the total number of arrangements for two independent groups of 5 is 64 -/
theorem total_arrangements : seating_arrangements 5 * seating_arrangements 5 = 64 := by sorry

end seating_arrangements_5_total_arrangements_l349_34923


namespace crayons_erasers_difference_l349_34999

/-- Given the initial number of crayons and erasers, and the remaining number of crayons,
    prove that the difference between remaining crayons and erasers is 353. -/
theorem crayons_erasers_difference (initial_crayons : ℕ) (initial_erasers : ℕ) (remaining_crayons : ℕ) 
    (h1 : initial_crayons = 531)
    (h2 : initial_erasers = 38)
    (h3 : remaining_crayons = 391) : 
  remaining_crayons - initial_erasers = 353 := by
  sorry

end crayons_erasers_difference_l349_34999


namespace three_fractions_inequality_l349_34981

theorem three_fractions_inequality (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_inequality : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ ∃ (k : ℝ), k > 0 ∧ x = 2*k ∧ y = k ∧ z = k) :=
by sorry

end three_fractions_inequality_l349_34981


namespace abcd_multiplication_l349_34997

theorem abcd_multiplication (A B C D : ℕ) : 
  A < 10 → B < 10 → C < 10 → D < 10 →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (1000 * A + 100 * B + 10 * C + D) * 2 = 10000 * A + 1000 * B + 100 * C + 10 * D →
  A + B = 1 := by
sorry

end abcd_multiplication_l349_34997


namespace tangent_and_inequality_imply_m_range_l349_34912

open Real

noncomputable def f (x : ℝ) : ℝ := x / (Real.exp x)

theorem tangent_and_inequality_imply_m_range :
  (∀ x ∈ Set.Ioo (1/2) (3/2), f x < 1 / (m + 6*x - 3*x^2)) →
  m ∈ Set.Icc (-9/4) (ℯ - 3) :=
sorry

end tangent_and_inequality_imply_m_range_l349_34912


namespace maria_initial_money_l349_34977

/-- The amount of money Maria had when she left the fair -/
def money_left : ℕ := 16

/-- The difference between the amount of money Maria had when she got to the fair and when she left -/
def money_difference : ℕ := 71

/-- The amount of money Maria had when she got to the fair -/
def money_initial : ℕ := money_left + money_difference

theorem maria_initial_money : money_initial = 87 := by
  sorry

end maria_initial_money_l349_34977


namespace two_zeros_read_in_2006_06_l349_34940

-- Define a function to count the number of zeros read in a number
def countZerosRead (n : ℝ) : ℕ := sorry

-- Define the given numbers
def num1 : ℝ := 200.06
def num2 : ℝ := 20.06
def num3 : ℝ := 2006.06

-- Theorem statement
theorem two_zeros_read_in_2006_06 :
  (countZerosRead num1 < 2) ∧
  (countZerosRead num2 < 2) ∧
  (countZerosRead num3 = 2) :=
sorry

end two_zeros_read_in_2006_06_l349_34940


namespace xyz_sum_product_sqrt_l349_34945

theorem xyz_sum_product_sqrt (x y z : ℝ) 
  (h1 : y + z = 16)
  (h2 : z + x = 17)
  (h3 : x + y = 18) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 1831.78125 := by
  sorry

end xyz_sum_product_sqrt_l349_34945


namespace point_two_units_from_negative_one_l349_34964

theorem point_two_units_from_negative_one (x : ℝ) : 
  (|x - (-1)| = 2) ↔ (x = -3 ∨ x = 1) := by sorry

end point_two_units_from_negative_one_l349_34964


namespace pool_filling_solution_l349_34928

/-- Represents the pool filling problem -/
def PoolFilling (totalVolume fillRate initialTime leakRate : ℝ) : Prop :=
  let initialVolume := fillRate * initialTime
  let remainingVolume := totalVolume - initialVolume
  let netFillRate := fillRate - leakRate
  let additionalTime := remainingVolume / netFillRate
  initialTime + additionalTime = 220

/-- Theorem stating the solution to the pool filling problem -/
theorem pool_filling_solution :
  PoolFilling 4000 20 20 2 := by sorry

end pool_filling_solution_l349_34928


namespace arithmetic_sequence_ratio_l349_34965

/-- Two arithmetic sequences and their partial sums -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n, S n / T n = (7 * n + 2) / (n + 3)

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) (S T : ℕ → ℚ) 
  (h : arithmetic_sequences a b S T) : 
  (a 2 + a 20) / (b 7 + b 15) = 149 / 24 := by
sorry

end arithmetic_sequence_ratio_l349_34965


namespace test_score_calculation_l349_34951

theorem test_score_calculation (total_questions : Nat) (score : Int) 
  (h1 : total_questions = 100)
  (h2 : score = 61) :
  ∃ (correct : Nat),
    correct ≤ total_questions ∧ 
    (correct : Int) - 2 * (total_questions - correct) = score ∧ 
    correct = 87 := by
  sorry

end test_score_calculation_l349_34951


namespace circle_equation_with_radius_3_l349_34960

theorem circle_equation_with_radius_3 (c : ℝ) : 
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + c = 0 ↔ (x - 4)^2 + (y + 5)^2 = 3^2) → 
  c = 32 := by
sorry

end circle_equation_with_radius_3_l349_34960


namespace speech_competition_arrangements_l349_34962

/-- The number of students in the speech competition -/
def num_students : ℕ := 6

/-- The total number of arrangements where B and C are adjacent -/
def total_arrangements_bc_adjacent : ℕ := 240

/-- The number of arrangements where A is first or last, and B and C are adjacent -/
def arrangements_a_first_or_last : ℕ := 96

/-- The number of valid arrangements for the speech competition -/
def valid_arrangements : ℕ := total_arrangements_bc_adjacent - arrangements_a_first_or_last

theorem speech_competition_arrangements :
  valid_arrangements = 144 :=
sorry

end speech_competition_arrangements_l349_34962


namespace currency_exchange_problem_l349_34920

def exchange_rate : ℚ := 9 / 6

def spent_amount : ℕ := 45

theorem currency_exchange_problem (d : ℕ) :
  (d : ℚ) * exchange_rate - spent_amount = d →
  (d / 10 + d % 10 : ℕ) = 9 := by
  sorry

end currency_exchange_problem_l349_34920


namespace circle_equation_m_range_l349_34914

/-- 
Given an equation x^2 + y^2 + mx - 2y + 4 = 0 that represents a circle,
prove that m must be in the range (-∞, -2√3) ∪ (2√3, +∞).
-/
theorem circle_equation_m_range :
  ∀ m : ℝ, 
  (∃ x y : ℝ, x^2 + y^2 + m*x - 2*y + 4 = 0) →
  (m < -2 * Real.sqrt 3 ∨ m > 2 * Real.sqrt 3) :=
by sorry

end circle_equation_m_range_l349_34914


namespace correct_propositions_l349_34992

/-- A structure representing a plane with lines -/
structure Plane where
  /-- The type of lines in the plane -/
  Line : Type
  /-- Perpendicularity relation between lines -/
  perp : Line → Line → Prop
  /-- Parallelism relation between lines -/
  parallel : Line → Line → Prop

/-- The main theorem stating the two correct propositions -/
theorem correct_propositions (P : Plane) 
  (a b c α β γ : P.Line) : 
  (P.perp a α ∧ P.perp b β ∧ P.perp α β → P.perp a b) ∧
  (P.parallel α β ∧ P.parallel β γ ∧ P.perp a α → P.perp a γ) := by
  sorry


end correct_propositions_l349_34992


namespace f_of_three_equals_six_l349_34907

/-- Given a function f: ℝ → ℝ satisfying certain conditions, prove that f(3) = 6 -/
theorem f_of_three_equals_six (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : f 1 = 4)
  (h2 : f 2 = 9)
  (h3 : ∀ x, f x = a * x + b * x + 3) :
  f 3 = 6 := by
sorry

end f_of_three_equals_six_l349_34907


namespace fraction_zero_implies_x_negative_five_l349_34935

theorem fraction_zero_implies_x_negative_five (x : ℝ) :
  (x + 5) / (x - 2) = 0 → x = -5 := by
sorry

end fraction_zero_implies_x_negative_five_l349_34935


namespace practice_hours_until_game_l349_34916

/-- Calculates the total practice hours for a given number of weeks -/
def total_practice_hours (weeks : ℕ) : ℕ :=
  let weekday_hours := 3
  let weekday_count := 5
  let saturday_hours := 5
  let weekly_hours := weekday_hours * weekday_count + saturday_hours
  weekly_hours * weeks

/-- The number of weeks until the next game -/
def weeks_until_game : ℕ := 3

theorem practice_hours_until_game :
  total_practice_hours weeks_until_game = 60 := by
  sorry

end practice_hours_until_game_l349_34916


namespace volume_between_concentric_spheres_l349_34993

theorem volume_between_concentric_spheres :
  let r₁ : ℝ := 4  -- radius of smaller sphere
  let r₂ : ℝ := 7  -- radius of larger sphere
  let V : ℝ := (4 / 3) * Real.pi * (r₂^3 - r₁^3)  -- volume between spheres
  V = 372 * Real.pi :=
by sorry

end volume_between_concentric_spheres_l349_34993


namespace geometric_region_equivalence_l349_34956

theorem geometric_region_equivalence (x y : ℝ) :
  (x^2 + y^2 - 4 ≥ 0 ∧ x^2 - 1 ≥ 0 ∧ y^2 - 1 ≥ 0) ↔
  ((x^2 + y^2 ≥ 4) ∧ (x ≤ -1 ∨ x ≥ 1) ∧ (y ≤ -1 ∨ y ≥ 1)) :=
by sorry

end geometric_region_equivalence_l349_34956


namespace employee_age_when_hired_l349_34994

theorem employee_age_when_hired (age_when_hired : ℕ) (years_worked : ℕ) : 
  age_when_hired + years_worked = 70 →
  years_worked = 19 →
  age_when_hired = 51 := by
  sorry

end employee_age_when_hired_l349_34994


namespace solution_set_inequality_l349_34904

theorem solution_set_inequality (x : ℝ) : 
  (2 * x + 3) * (4 - x) > 0 ↔ -3/2 < x ∧ x < 4 := by
  sorry

end solution_set_inequality_l349_34904


namespace snack_spending_l349_34961

/-- The total amount spent by Robert and Teddy on snacks for their friends -/
def total_spent (pizza_price : ℕ) (pizza_quantity : ℕ) (drink_price : ℕ) (robert_drink_quantity : ℕ) 
  (hamburger_price : ℕ) (hamburger_quantity : ℕ) (teddy_drink_quantity : ℕ) : ℕ :=
  pizza_price * pizza_quantity + 
  drink_price * (robert_drink_quantity + teddy_drink_quantity) + 
  hamburger_price * hamburger_quantity

/-- Theorem stating that Robert and Teddy spend $108 in total -/
theorem snack_spending : 
  total_spent 10 5 2 10 3 6 10 = 108 := by
  sorry

end snack_spending_l349_34961


namespace expression_evaluation_l349_34939

theorem expression_evaluation : (2000^2 : ℝ) / (402^2 - 398^2) = 1250 := by
  sorry

end expression_evaluation_l349_34939


namespace molecular_weight_calculation_l349_34913

/-- Given a compound where 3 moles weigh 528 grams, prove its molecular weight is 176 grams/mole. -/
theorem molecular_weight_calculation (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 528)
  (h2 : num_moles = 3) :
  total_weight / num_moles = 176 := by
  sorry

end molecular_weight_calculation_l349_34913


namespace system_solutions_l349_34903

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  26 * x^2 - 42 * x * y + 17 * y^2 = 10 ∧
  10 * x^2 - 18 * x * y + 8 * y^2 = 6

/-- The set of solutions -/
def solutions : Set (ℝ × ℝ) :=
  {(-1, -2), (1, 2), (-11, -14), (11, 14)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ x y : ℝ, system x y ↔ (x, y) ∈ solutions := by
  sorry

end system_solutions_l349_34903


namespace conditional_probability_rain_given_east_wind_l349_34984

theorem conditional_probability_rain_given_east_wind 
  (p_east_wind : ℝ) 
  (p_east_wind_and_rain : ℝ) 
  (h1 : p_east_wind = 8/30) 
  (h2 : p_east_wind_and_rain = 7/30) : 
  p_east_wind_and_rain / p_east_wind = 7/8 := by
  sorry

end conditional_probability_rain_given_east_wind_l349_34984


namespace rose_difference_is_34_l349_34980

/-- The number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := 58

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := 24

/-- The difference in the number of red roses between Mrs. Santiago and Mrs. Garrett -/
def rose_difference : ℕ := santiago_roses - garrett_roses

theorem rose_difference_is_34 : rose_difference = 34 := by
  sorry

end rose_difference_is_34_l349_34980


namespace cosine_value_for_given_point_l349_34918

theorem cosine_value_for_given_point :
  ∀ α : Real,
  let P : Real × Real := (2 * Real.cos (120 * π / 180), Real.sqrt 2 * Real.sin (225 * π / 180))
  (Real.cos α = P.1 / Real.sqrt (P.1^2 + P.2^2) ∧
   Real.sin α = P.2 / Real.sqrt (P.1^2 + P.2^2)) →
  Real.cos α = -Real.sqrt 2 / 2 := by
  sorry

end cosine_value_for_given_point_l349_34918


namespace max_element_of_S_l349_34942

def S : Set ℚ := {x | ∃ (p q : ℕ), x = p / q ∧ q ≤ 2009 ∧ x < 1257 / 2009}

theorem max_element_of_S :
  ∃ (p₀ q₀ : ℕ), 
    (p₀ : ℚ) / q₀ ∈ S ∧ 
    (∀ (x : ℚ), x ∈ S → x ≤ (p₀ : ℚ) / q₀) ∧
    (Nat.gcd p₀ q₀ = 1) ∧
    p₀ = 229 ∧ 
    q₀ = 366 ∧ 
    p₀ + q₀ = 595 := by
  sorry

end max_element_of_S_l349_34942


namespace trapezoid_triangle_area_l349_34919

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  -- Area of the trapezoid
  area : ℝ
  -- Condition that one base is twice the other
  base_ratio : Bool
  -- Point of intersection of diagonals
  O : Point
  -- Midpoint of base AD
  P : Point
  -- Points where BP and CP intersect the diagonals
  M : Point
  N : Point

/-- The area of triangle MON in a trapezoid with specific properties -/
def area_MON (t : Trapezoid) : Set ℝ :=
  {45/4, 36/5}

/-- Theorem stating the area of triangle MON in a trapezoid with given properties -/
theorem trapezoid_triangle_area (t : Trapezoid) 
  (h1 : t.area = 405) : 
  (area_MON t).Nonempty ∧ (∀ x ∈ area_MON t, x = 45/4 ∨ x = 36/5) := by
  sorry

end trapezoid_triangle_area_l349_34919


namespace evaluate_expression_l349_34967

theorem evaluate_expression (x y z w : ℚ) 
  (hx : x = 1/4)
  (hy : y = 1/3)
  (hz : z = -2)
  (hw : w = 3) :
  x^3 * y^2 * z^2 * w = 1/48 := by
  sorry

end evaluate_expression_l349_34967


namespace blueberry_picking_l349_34969

theorem blueberry_picking (annie kathryn ben : ℕ) : 
  annie = 8 →
  kathryn = annie + 2 →
  ben = kathryn - 3 →
  annie + kathryn + ben = 25 := by
  sorry

end blueberry_picking_l349_34969


namespace difference_of_squares_l349_34948

theorem difference_of_squares (m : ℝ) : m^2 - 16 = (m + 4) * (m - 4) := by
  sorry

end difference_of_squares_l349_34948


namespace smallest_non_factor_product_l349_34946

def is_factor (n m : ℕ) : Prop := m % n = 0

theorem smallest_non_factor_product (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  is_factor a 48 → 
  is_factor b 48 → 
  ¬(is_factor (a * b) 48) → 
  (∀ (x y : ℕ), x ≠ y → x > 0 → y > 0 → is_factor x 48 → is_factor y 48 → 
    ¬(is_factor (x * y) 48) → a * b ≤ x * y) → 
  a * b = 18 :=
sorry

end smallest_non_factor_product_l349_34946


namespace stamps_ratio_after_gift_l349_34950

/-- Proves that given the initial conditions, the new ratio of Kaye's stamps to Alberto's stamps is 4:3 -/
theorem stamps_ratio_after_gift (x : ℕ) 
  (h1 : 5 * x - 12 = 3 * x + 12 + 32) : 
  (5 * x - 12) / (3 * x + 12) = 4 / 3 := by
  sorry

#check stamps_ratio_after_gift

end stamps_ratio_after_gift_l349_34950


namespace optimal_garden_dimensions_l349_34971

/-- Represents a rectangular garden with one side along a house wall. -/
structure Garden where
  width : ℝ  -- Width of the garden (perpendicular to the house)
  length : ℝ  -- Length of the garden (parallel to the house)

/-- Calculates the area of a rectangular garden. -/
def Garden.area (g : Garden) : ℝ := g.width * g.length

/-- Calculates the cost of fencing for three sides of the garden. -/
def Garden.fenceCost (g : Garden) : ℝ := 10 * (g.length + 2 * g.width)

/-- Theorem stating the optimal dimensions of the garden. -/
theorem optimal_garden_dimensions (houseLength : ℝ) (totalFenceCost : ℝ) :
  houseLength = 300 → totalFenceCost = 2000 →
  ∃ (g : Garden),
    g.fenceCost = totalFenceCost ∧
    g.length = 100 ∧
    ∀ (g' : Garden), g'.fenceCost = totalFenceCost → g.area ≥ g'.area :=
sorry

end optimal_garden_dimensions_l349_34971


namespace right_triangle_hypotenuse_l349_34982

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (a = 5 ∧ b = 12) ∨ (a = 5 ∧ c = 12) ∨ (b = 5 ∧ c = 12) →
  a^2 + b^2 = c^2 →
  c = 12 ∨ c = 13 := by
sorry

end right_triangle_hypotenuse_l349_34982


namespace tournament_score_sum_l349_34989

/-- A round-robin tournament with three players -/
structure Tournament :=
  (players : Fin 3 → ℕ)

/-- The scoring system for the tournament -/
def score (result : ℕ) : ℕ :=
  match result with
  | 0 => 2  -- win
  | 1 => 1  -- draw
  | _ => 0  -- loss

/-- The theorem stating that the sum of all players' scores is always 6 -/
theorem tournament_score_sum (t : Tournament) : 
  (t.players 0) + (t.players 1) + (t.players 2) = 6 :=
sorry

end tournament_score_sum_l349_34989


namespace cd_price_difference_l349_34973

theorem cd_price_difference (album_price book_price : ℝ) (h1 : album_price = 20) (h2 : book_price = 18) : 
  let cd_price := book_price - 4
  (album_price - cd_price) / album_price * 100 = 30 := by
sorry

end cd_price_difference_l349_34973


namespace nonSimilar500PointedStars_l349_34941

/-- The number of non-similar regular n-pointed stars -/
def nonSimilarStars (n : ℕ) : ℕ :=
  (n.totient - 2) / 2

/-- Theorem: The number of non-similar regular 500-pointed stars is 99 -/
theorem nonSimilar500PointedStars : nonSimilarStars 500 = 99 := by
  sorry

#eval nonSimilarStars 500  -- This should evaluate to 99

end nonSimilar500PointedStars_l349_34941


namespace polynomial_root_sum_l349_34944

/-- A polynomial with real coefficients -/
def g (p q r s : ℝ) (x : ℂ) : ℂ := x^4 + p*x^3 + q*x^2 + r*x + s

/-- Theorem: If g(3i) = 0 and g(1+2i) = 0, then p + q + r + s = 39 -/
theorem polynomial_root_sum (p q r s : ℝ) : 
  g p q r s (3*I) = 0 → g p q r s (1 + 2*I) = 0 → p + q + r + s = 39 := by
  sorry

end polynomial_root_sum_l349_34944


namespace friend_payment_ratio_l349_34936

def james_meal : ℚ := 16
def friend_meal : ℚ := 14
def tip_percentage : ℚ := 20 / 100
def james_total_paid : ℚ := 21

def total_bill : ℚ := james_meal + friend_meal
def tip : ℚ := total_bill * tip_percentage
def total_bill_with_tip : ℚ := total_bill + tip
def james_share : ℚ := james_total_paid - tip
def friend_payment : ℚ := total_bill - james_share

theorem friend_payment_ratio :
  friend_payment / total_bill_with_tip = 5 / 12 := by
  sorry

end friend_payment_ratio_l349_34936


namespace ginas_college_expenses_l349_34931

/-- Calculates the total college expenses for Gina -/
def total_college_expenses (credits : ℕ) (cost_per_credit : ℕ) (num_textbooks : ℕ) (cost_per_textbook : ℕ) (facilities_fee : ℕ) : ℕ :=
  credits * cost_per_credit + num_textbooks * cost_per_textbook + facilities_fee

/-- Proves that Gina's total college expenses are $7100 -/
theorem ginas_college_expenses :
  total_college_expenses 14 450 5 120 200 = 7100 := by
  sorry

end ginas_college_expenses_l349_34931


namespace max_non_managers_l349_34958

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 9 →
  (managers : ℚ) / non_managers > 7 / 32 →
  non_managers ≤ 41 :=
by sorry

end max_non_managers_l349_34958


namespace circle_representation_l349_34917

theorem circle_representation (a : ℝ) :
  ∃ h k r, ∀ x y : ℝ,
    x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1 = 0 ↔
    (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0 :=
by sorry

end circle_representation_l349_34917


namespace youngest_child_age_exists_l349_34979

/-- Represents the ages of the four children -/
structure ChildrenAges where
  twin_age : ℕ
  child1_age : ℕ
  child2_age : ℕ

/-- Calculates the total bill for the dinner -/
def calculate_bill (ages : ChildrenAges) : ℚ :=
  (30 * (25 : ℚ) / 100) + 
  ((2 * ages.twin_age + ages.child1_age + ages.child2_age) * (55 : ℚ) / 100)

theorem youngest_child_age_exists : ∃ (ages : ChildrenAges), 
  calculate_bill ages = 1510 / 100 ∧
  ages.twin_age ≠ ages.child1_age ∧
  ages.twin_age ≠ ages.child2_age ∧
  ages.child1_age ≠ ages.child2_age ∧
  min ages.twin_age (min ages.child1_age ages.child2_age) = 1 := by
  sorry

end youngest_child_age_exists_l349_34979


namespace zinc_copper_ratio_theorem_l349_34968

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents a mixture of zinc and copper -/
structure Mixture where
  total_weight : ℝ
  zinc_weight : ℝ

/-- Calculates the ratio of zinc to copper in a mixture -/
def zinc_copper_ratio (m : Mixture) : Ratio :=
  sorry

/-- The given mixture of zinc and copper -/
def given_mixture : Mixture :=
  { total_weight := 74
    zinc_weight := 33.3 }

/-- Theorem stating the correct ratio of zinc to copper in the given mixture -/
theorem zinc_copper_ratio_theorem :
  zinc_copper_ratio given_mixture = Ratio.mk 333 407 :=
  sorry

end zinc_copper_ratio_theorem_l349_34968


namespace discount_percentage_is_correct_l349_34978

/-- Calculates the discount percentage given the purchase price, selling price for 10% profit, and additional costs --/
def calculate_discount_percentage (purchase_price selling_price_for_profit transport_cost installation_cost : ℚ) : ℚ :=
  let labelled_price := selling_price_for_profit / 1.1
  let discount_amount := labelled_price - purchase_price
  (discount_amount / labelled_price) * 100

/-- Theorem stating that the discount percentage is equal to (500/23)% given the problem conditions --/
theorem discount_percentage_is_correct :
  let purchase_price : ℚ := 13500
  let selling_price_for_profit : ℚ := 18975
  let transport_cost : ℚ := 125
  let installation_cost : ℚ := 250
  calculate_discount_percentage purchase_price selling_price_for_profit transport_cost installation_cost = 500 / 23 := by
  sorry

#eval calculate_discount_percentage 13500 18975 125 250

end discount_percentage_is_correct_l349_34978


namespace negative_difference_l349_34991

theorem negative_difference (a b : ℝ) : -(a - b) = -a + b := by
  sorry

end negative_difference_l349_34991


namespace kevins_age_l349_34909

theorem kevins_age (vanessa_age : ℕ) (future_years : ℕ) (ratio : ℕ) :
  vanessa_age = 2 →
  future_years = 5 →
  ratio = 3 →
  ∃ kevin_age : ℕ, kevin_age + future_years = ratio * (vanessa_age + future_years) ∧ kevin_age = 16 :=
by sorry

end kevins_age_l349_34909


namespace p_necessary_not_sufficient_for_q_l349_34949

theorem p_necessary_not_sufficient_for_q : 
  (∀ x : ℝ, |x - 1| < 2 → x + 1 ≥ 0) ∧ 
  (∃ x : ℝ, x + 1 ≥ 0 ∧ |x - 1| ≥ 2) := by
  sorry

end p_necessary_not_sufficient_for_q_l349_34949


namespace total_books_l349_34943

theorem total_books (x : ℚ) : ℚ := by
  -- Betty's books
  let betty_books := x

  -- Sister's books: x + (1/4)x
  let sister_books := x + (1/4) * x

  -- Cousin's books: 2 * (x + (1/4)x)
  let cousin_books := 2 * (x + (1/4) * x)

  -- Total books
  let total := betty_books + sister_books + cousin_books

  -- Prove that total = (19/4)x
  sorry

end total_books_l349_34943


namespace function_is_even_l349_34902

/-- A function satisfying the given functional equation -/
class FunctionalEquation (f : ℝ → ℝ) : Prop where
  eq : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b
  not_zero : ∃ x : ℝ, f x ≠ 0

/-- The main theorem: if f satisfies the functional equation, then it is even -/
theorem function_is_even (f : ℝ → ℝ) [FunctionalEquation f] : ∀ x : ℝ, f (-x) = f x := by
  sorry

end function_is_even_l349_34902


namespace space_divided_by_five_spheres_l349_34934

/-- Maximum number of regions a sphere can be divided by n circles -/
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => a (n + 1) + 2 * (n + 1)

/-- Maximum number of regions space can be divided by n spheres -/
def b : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => b (n + 1) + a (n + 1)

theorem space_divided_by_five_spheres :
  b 5 = 22 := by sorry

end space_divided_by_five_spheres_l349_34934


namespace lucy_sales_l349_34901

/-- Given the total number of packs sold and Robyn's sales, calculate Lucy's sales. -/
theorem lucy_sales (total : ℕ) (robyn : ℕ) (h1 : total = 98) (h2 : robyn = 55) :
  total - robyn = 43 := by
  sorry

end lucy_sales_l349_34901


namespace smallest_n_for_factors_l349_34927

theorem smallest_n_for_factors (k : ℕ) : 
  (∀ m : ℕ, m > 0 → (5^2 ∣ m * 2^k * 6^2 * 7^3) → (3^3 ∣ m * 2^k * 6^2 * 7^3) → m ≥ 75) ∧
  (5^2 ∣ 75 * 2^k * 6^2 * 7^3) ∧
  (3^3 ∣ 75 * 2^k * 6^2 * 7^3) :=
sorry

end smallest_n_for_factors_l349_34927


namespace cubic_integer_root_l349_34933

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_nonzero : a ≠ 0

/-- Evaluation of a cubic polynomial at a point -/
def CubicPolynomial.eval (P : CubicPolynomial) (x : ℤ) : ℤ :=
  P.a * x^3 + P.b * x^2 + P.c * x + P.d

/-- The property that xP(x) = yP(y) for infinitely many integer pairs (x,y) with x ≠ y -/
def InfinitelyManySolutions (P : CubicPolynomial) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ n < |x| ∧ n < |y| ∧ x * P.eval x = y * P.eval y

theorem cubic_integer_root (P : CubicPolynomial) 
    (h : InfinitelyManySolutions P) : 
    ∃ k : ℤ, P.eval k = 0 := by
  sorry

end cubic_integer_root_l349_34933


namespace quadratic_sum_l349_34953

/-- The quadratic function under consideration -/
def f (x : ℝ) : ℝ := 4 * x^2 - 48 * x - 128

/-- The same quadratic function in completed square form -/
def g (x : ℝ) (a b c : ℝ) : ℝ := a * (x + b)^2 + c

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, f x = g x a b c) → a + b + c = -274 := by sorry

end quadratic_sum_l349_34953


namespace annes_bowling_score_l349_34952

theorem annes_bowling_score (annes_score bob_score : ℕ) : 
  annes_score = bob_score + 50 →
  (annes_score + bob_score) / 2 = 150 →
  annes_score = 175 := by
sorry

end annes_bowling_score_l349_34952


namespace min_value_product_l349_34976

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (x + 3 * y) * (y + 3 * z) * (x * z + 2) ≥ 96 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (x₀ + 3 * y₀) * (y₀ + 3 * z₀) * (x₀ * z₀ + 2) = 96 := by
  sorry

end min_value_product_l349_34976


namespace custard_pie_pieces_l349_34922

/-- Proves that the number of pieces a custard pie is cut into is 6, given the conditions of the bakery problem. -/
theorem custard_pie_pieces : ℕ :=
  let pumpkin_pieces : ℕ := 8
  let pumpkin_price : ℕ := 5
  let custard_price : ℕ := 6
  let pumpkin_pies_sold : ℕ := 4
  let custard_pies_sold : ℕ := 5
  let total_revenue : ℕ := 340

  have h1 : pumpkin_pieces * pumpkin_price * pumpkin_pies_sold + custard_price * custard_pies_sold * custard_pie_pieces = total_revenue := by sorry

  custard_pie_pieces
where
  custard_pie_pieces : ℕ := 6

#check custard_pie_pieces

end custard_pie_pieces_l349_34922


namespace cube_root_square_l349_34905

theorem cube_root_square (x : ℝ) : (x + 5) ^ (1/3 : ℝ) = 3 → (x + 5)^2 = 729 := by
  sorry

end cube_root_square_l349_34905


namespace smaller_integer_problem_l349_34996

theorem smaller_integer_problem (a b : ℕ+) : 
  (a : ℕ) + 8 = (b : ℕ) → a * b = 80 → (a : ℕ) = 2 := by
  sorry

end smaller_integer_problem_l349_34996


namespace rectangle_dimensions_l349_34915

theorem rectangle_dimensions :
  ∀ w l : ℝ,
  w > 0 →
  l > 0 →
  l = 2 * w →
  w * l = (1/2) * (2 * (w + l)) →
  w = (3/2) ∧ l = 3 :=
by
  sorry

end rectangle_dimensions_l349_34915


namespace fourth_power_of_nested_root_l349_34986

theorem fourth_power_of_nested_root : 
  let x := Real.sqrt (2 + Real.sqrt (3 + Real.sqrt 4))
  x^4 = 9 + 4 * Real.sqrt 5 := by
sorry

end fourth_power_of_nested_root_l349_34986


namespace ruth_apples_l349_34988

theorem ruth_apples (x : ℕ) : x - 5 = 84 → x = 89 := by sorry

end ruth_apples_l349_34988


namespace sum_10_is_negative_15_l349_34925

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * a 1 + (n * (n - 1) / 2 : ℝ) * (a 2 - a 1)
  S_3 : S 3 = 6
  S_6 : S 6 = 3

/-- The sum of the first 10 terms is -15 -/
theorem sum_10_is_negative_15 (seq : ArithmeticSequence) : seq.S 10 = -15 := by
  sorry

end sum_10_is_negative_15_l349_34925


namespace percentage_problem_l349_34959

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 36 → P = 30 := by
  sorry

end percentage_problem_l349_34959


namespace division_problem_l349_34955

theorem division_problem (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 161)
  (h2 : quotient = 10)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) :
  divisor = 16 := by
sorry

end division_problem_l349_34955


namespace average_not_equal_given_l349_34970

theorem average_not_equal_given (numbers : List ℝ) (given_average : ℝ) : 
  numbers = [12, 13, 14, 510, 520, 530, 1115, 1, 1252140, 2345] →
  given_average = 858.5454545454545 →
  (numbers.sum / numbers.length : ℝ) ≠ given_average := by
sorry

end average_not_equal_given_l349_34970


namespace paper_towel_savings_l349_34966

/-- Calculates the percent savings per roll when buying a package of rolls compared to individual rolls -/
def percent_savings_per_roll (package_price : ℚ) (package_size : ℕ) (individual_price : ℚ) : ℚ :=
  let package_price_per_roll := package_price / package_size
  let savings_per_roll := individual_price - package_price_per_roll
  (savings_per_roll / individual_price) * 100

/-- Theorem: The percent savings per roll for a 12-roll package priced at $9 compared to
    buying 12 rolls individually at $1 each is 25% -/
theorem paper_towel_savings :
  percent_savings_per_roll 9 12 1 = 25 := by
  sorry

end paper_towel_savings_l349_34966


namespace sticker_collection_total_l349_34910

/-- The number of stickers Karl has -/
def karl_stickers : ℕ := 25

/-- The number of stickers Ryan has -/
def ryan_stickers : ℕ := karl_stickers + 20

/-- The number of stickers Ben has -/
def ben_stickers : ℕ := ryan_stickers - 10

/-- The total number of stickers placed in the book -/
def total_stickers : ℕ := karl_stickers + ryan_stickers + ben_stickers

theorem sticker_collection_total :
  total_stickers = 105 := by sorry

end sticker_collection_total_l349_34910


namespace handshake_problem_l349_34932

theorem handshake_problem (a b : ℕ) : 
  a + b = 20 →
  (a * (a - 1)) / 2 + (b * (b - 1)) / 2 = 106 →
  a * b = 84 := by
sorry

end handshake_problem_l349_34932


namespace rhombus_perimeter_l349_34987

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := by
  sorry


end rhombus_perimeter_l349_34987


namespace line_intercepts_sum_l349_34975

/-- Given a line 3x - 4y + k = 0, if the sum of its x-intercept and y-intercept is 2, then k = -24 -/
theorem line_intercepts_sum (k : ℝ) : 
  (∃ x y : ℝ, 3*x - 4*y + k = 0 ∧ x + y = 2) → k = -24 := by
  sorry

end line_intercepts_sum_l349_34975
