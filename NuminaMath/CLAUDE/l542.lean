import Mathlib

namespace mixture_problem_l542_54213

/-- A mixture problem involving milk and water ratios -/
theorem mixture_problem (x : ℝ) (h1 : x > 0) : 
  (4 * x) / x = 4 →                  -- Initial ratio of milk to water is 4:1
  (4 * x) / (x + 9) = 2 →            -- Final ratio after adding 9 litres of water is 2:1
  5 * x = 45 :=                      -- Initial volume of the mixture is 45 litres
by
  sorry

end mixture_problem_l542_54213


namespace meal_combinations_count_l542_54221

/-- The number of items in Menu A -/
def menu_a_items : ℕ := 15

/-- The number of items in Menu B -/
def menu_b_items : ℕ := 12

/-- The total number of possible meal combinations -/
def total_combinations : ℕ := menu_a_items * menu_b_items

/-- Theorem stating that the total number of meal combinations is 180 -/
theorem meal_combinations_count : total_combinations = 180 := by
  sorry

end meal_combinations_count_l542_54221


namespace inequality_proof_l542_54225

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_product : a * b * c = 1) : 
  a^2 + b^2 + c^2 + 3 ≥ 2 * (1/a + 1/b + 1/c) := by
  sorry

end inequality_proof_l542_54225


namespace school_election_votes_l542_54251

theorem school_election_votes (eliot_votes shaun_votes other_votes : ℕ) : 
  eliot_votes = 2 * shaun_votes →
  shaun_votes = 5 * other_votes →
  eliot_votes = 160 →
  other_votes = 16 := by
sorry

end school_election_votes_l542_54251


namespace point_on_x_axis_with_distance_3_l542_54292

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance from a point to the y-axis
def distToYAxis (p : Point2D) : ℝ := |p.x|

-- Theorem statement
theorem point_on_x_axis_with_distance_3 (P : Point2D) :
  P.y = 0 ∧ distToYAxis P = 3 → P = ⟨3, 0⟩ ∨ P = ⟨-3, 0⟩ := by
  sorry

end point_on_x_axis_with_distance_3_l542_54292


namespace bike_riders_proportion_l542_54206

theorem bike_riders_proportion (total_students bus_riders walkers : ℕ) 
  (h1 : total_students = 92)
  (h2 : bus_riders = 20)
  (h3 : walkers = 27) :
  (total_students - bus_riders - walkers : ℚ) / (total_students - bus_riders : ℚ) = 45 / 72 :=
by sorry

end bike_riders_proportion_l542_54206


namespace cpu_sales_count_l542_54273

/-- Represents the sales data for a hardware store for one week -/
structure HardwareSales where
  graphics_cards : Nat
  hard_drives : Nat
  cpus : Nat
  ram_pairs : Nat
  graphics_card_price : Nat
  hard_drive_price : Nat
  cpu_price : Nat
  ram_pair_price : Nat
  total_earnings : Nat

/-- Theorem stating that given the sales data, the number of CPUs sold is 8 -/
theorem cpu_sales_count (sales : HardwareSales) : 
  sales.graphics_cards = 10 ∧
  sales.hard_drives = 14 ∧
  sales.ram_pairs = 4 ∧
  sales.graphics_card_price = 600 ∧
  sales.hard_drive_price = 80 ∧
  sales.cpu_price = 200 ∧
  sales.ram_pair_price = 60 ∧
  sales.total_earnings = 8960 →
  sales.cpus = 8 := by
  sorry

#check cpu_sales_count

end cpu_sales_count_l542_54273


namespace x_limit_properties_l542_54222

noncomputable def x : ℕ → ℝ
  | 0 => Real.sqrt 6
  | n + 1 => x n + 3 * Real.sqrt (x n) + (n + 1 : ℝ) / Real.sqrt (x n)

theorem x_limit_properties :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |n / x n| < ε) ∧
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |n^2 / x n - 4/9| < ε) := by
  sorry

end x_limit_properties_l542_54222


namespace max_speed_theorem_l542_54230

/-- Represents a pair of observed values (speed, defective products) -/
structure Observation where
  speed : ℝ
  defects : ℝ

/-- The regression line equation -/
def regression_line (slope : ℝ) (intercept : ℝ) (x : ℝ) : ℝ :=
  slope * x + intercept

/-- Theorem: Maximum speed given observations and max defects -/
theorem max_speed_theorem (observations : List Observation) 
  (max_defects : ℝ) (slope : ℝ) (intercept : ℝ) :
  observations = [
    ⟨8, 5⟩, ⟨12, 8⟩, ⟨14, 9⟩, ⟨16, 11⟩
  ] →
  max_defects = 10 →
  slope = 51 / 70 →
  intercept = -6 / 7 →
  (∀ x, regression_line slope intercept x ≤ max_defects → x ≤ 14) ∧
  regression_line slope intercept 14 ≤ max_defects :=
by sorry

end max_speed_theorem_l542_54230


namespace tangent_line_slope_l542_54271

theorem tangent_line_slope (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^4 + m*x
  let f' : ℝ → ℝ := λ x ↦ 4*x^3 + m
  let tangent_slope : ℝ := f' (-1)
  (2 * (-1) + f (-1) + 3 = 0) ∧ (tangent_slope = -2) → m = 2 := by
  sorry

end tangent_line_slope_l542_54271


namespace nickel_difference_l542_54282

/-- The number of cents in a nickel -/
def cents_per_nickel : ℕ := 5

/-- The total number of cents Ray has initially -/
def ray_initial_cents : ℕ := 175

/-- The number of cents Ray gives to Peter -/
def cents_to_peter : ℕ := 30

/-- Calculates the number of nickels given a number of cents -/
def cents_to_nickels (cents : ℕ) : ℕ := cents / cents_per_nickel

/-- Theorem stating the difference in nickels between Randi and Peter -/
theorem nickel_difference : 
  cents_to_nickels (2 * cents_to_peter) - cents_to_nickels cents_to_peter = 6 := by
  sorry

end nickel_difference_l542_54282


namespace monotone_function_characterization_l542_54201

/-- A monotone function from integers to integers -/
def MonotoneIntFunction (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, x ≤ y → f x ≤ f y

/-- The functional equation that f must satisfy -/
def SatisfiesFunctionalEquation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (x^2005 + y^2005) = (f x)^2005 + (f y)^2005

/-- The main theorem statement -/
theorem monotone_function_characterization (f : ℤ → ℤ) 
  (hm : MonotoneIntFunction f) (hf : SatisfiesFunctionalEquation f) :
  (∀ x : ℤ, f x = x) ∨ (∀ x : ℤ, f x = -x) :=
sorry

end monotone_function_characterization_l542_54201


namespace tan_C_in_special_triangle_l542_54290

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_of_angles : A + B + C = Real.pi

-- Define the theorem
theorem tan_C_in_special_triangle (t : Triangle) (h1 : Real.tan t.A = 1) (h2 : Real.tan t.B = 2) : 
  Real.tan t.C = 3 := by
  sorry

end tan_C_in_special_triangle_l542_54290


namespace arccos_of_neg_one_equals_pi_l542_54203

theorem arccos_of_neg_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end arccos_of_neg_one_equals_pi_l542_54203


namespace circle_radius_l542_54255

theorem circle_radius (x y : ℝ) : 
  (∃ r, r > 0 ∧ ∀ x y, x^2 + y^2 - 4*x + 2*y + 2 = 0 ↔ (x - 2)^2 + (y + 1)^2 = r^2) →
  (∃ r, r > 0 ∧ ∀ x y, x^2 + y^2 - 4*x + 2*y + 2 = 0 ↔ (x - 2)^2 + (y + 1)^2 = r^2 ∧ r = Real.sqrt 3) :=
by sorry

end circle_radius_l542_54255


namespace solution_pairs_l542_54241

theorem solution_pairs (x y p : ℕ) (hp : Nat.Prime p) :
  x > 0 ∧ y > 0 ∧ x ≤ y ∧ (x + y) * (x * y - 1) = p * (x * y + 1) →
  (x = 3 ∧ y = 5 ∧ p = 7) ∨ (∃ q : ℕ, Nat.Prime q ∧ x = 1 ∧ y = q + 1 ∧ p = q) :=
sorry

end solution_pairs_l542_54241


namespace pythagorean_triple_identification_l542_54258

def is_pythagorean_triple (a b c : ℚ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_identification :
  ¬ is_pythagorean_triple 5 8 12 ∧
  is_pythagorean_triple 30 40 50 ∧
  ¬ is_pythagorean_triple 9 13 15 ∧
  ¬ is_pythagorean_triple (1/6) (1/8) (1/10) :=
by sorry

end pythagorean_triple_identification_l542_54258


namespace expand_complex_product_l542_54238

theorem expand_complex_product (x : ℂ) : (x + Complex.I) * (x - 7) = x^2 - 7*x + Complex.I*x - 7*Complex.I := by
  sorry

end expand_complex_product_l542_54238


namespace expected_value_of_game_l542_54231

def roll_value (n : ℕ) : ℝ :=
  if n % 2 = 0 then 3 * n else 0

def fair_8_sided_die : Finset ℕ := Finset.range 8

theorem expected_value_of_game : 
  (fair_8_sided_die.sum (λ i => (roll_value (i + 1)) / 8)) = 7.5 := by
  sorry

end expected_value_of_game_l542_54231


namespace reciprocals_product_l542_54219

theorem reciprocals_product (a b : ℝ) (h : a * b = 1) : 4 * a * b = 4 := by
  sorry

end reciprocals_product_l542_54219


namespace fraction_inequality_l542_54200

theorem fraction_inequality (a b c d p q : ℕ+) 
  (h1 : a * d - b * c = 1)
  (h2 : (a : ℚ) / b > (p : ℚ) / q)
  (h3 : (p : ℚ) / q > (c : ℚ) / d) : 
  q ≥ b + d ∧ (q = b + d → p = a + c) := by
  sorry

end fraction_inequality_l542_54200


namespace sixteenth_number_with_digit_sum_13_l542_54295

/-- A function that returns the sum of digits of a positive integer -/
def digit_sum (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nth_number_with_digit_sum_13 (n : ℕ+) : ℕ+ := sorry

/-- Theorem stating that the 16th number with digit sum 13 is 247 -/
theorem sixteenth_number_with_digit_sum_13 : 
  nth_number_with_digit_sum_13 16 = 247 := by sorry

end sixteenth_number_with_digit_sum_13_l542_54295


namespace max_sum_under_constraints_l542_54204

theorem max_sum_under_constraints (a b : ℝ) :
  4 * a + 3 * b ≤ 10 →
  3 * a + 5 * b ≤ 11 →
  a + b ≤ 156 / 55 := by
sorry

end max_sum_under_constraints_l542_54204


namespace reinforcement_size_l542_54283

/-- Calculates the size of a reinforcement given initial garrison size, initial provisions duration,
    time passed before reinforcement, and remaining provisions duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                            (time_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let provisions_left := initial_garrison * (initial_duration - time_passed)
  let total_men := provisions_left / remaining_duration
  total_men - initial_garrison

/-- Proves that given the specified conditions, the reinforcement size is 300 men. -/
theorem reinforcement_size :
  calculate_reinforcement 150 31 16 5 = 300 := by
  sorry

end reinforcement_size_l542_54283


namespace trapezoid_segment_length_l542_54249

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ

/-- The theorem statement -/
theorem trapezoid_segment_length 
  (ABCD : Trapezoid) 
  (h_ratio : (ABCD.AB / ABCD.CD) = (5 / 2)) 
  (h_sum : ABCD.AB + ABCD.CD = 280) : 
  ABCD.AB = 200 := by
  sorry

#check trapezoid_segment_length

end trapezoid_segment_length_l542_54249


namespace exam_grading_problem_l542_54211

theorem exam_grading_problem (X : ℝ) 
  (monday_graded : X * 0.6 = X - (X * 0.4))
  (tuesday_graded : X * 0.4 * 0.75 = X * 0.4 - (X * 0.1))
  (wednesday_remaining : X * 0.1 = 12) :
  X = 120 := by
sorry

end exam_grading_problem_l542_54211


namespace sum_of_a_and_b_l542_54235

def f (x : ℝ) := x^3 + 3*x - 1

theorem sum_of_a_and_b (a b : ℝ) (h1 : f (a - 3) = -3) (h2 : f (b - 3) = 1) : a + b = 6 := by
  sorry

end sum_of_a_and_b_l542_54235


namespace cricket_run_rate_theorem_l542_54223

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  targetRuns : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let runsScored := game.firstPartRunRate * game.firstPartOvers
  let runsNeeded := game.targetRuns - runsScored
  runsNeeded / remainingOvers

/-- Theorem stating the required run rate for the given scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.totalOvers = 45)
  (h2 : game.firstPartOvers = 10)
  (h3 : game.firstPartRunRate = 3.5)
  (h4 : game.targetRuns = 350) :
  requiredRunRate game = 9 := by
  sorry

end cricket_run_rate_theorem_l542_54223


namespace number_of_male_students_l542_54229

theorem number_of_male_students 
  (total_candidates : ℕ)
  (male_students : ℕ)
  (female_students : ℕ)
  (selected_male : ℕ)
  (selected_female : ℕ)
  (num_camps : ℕ)
  (total_schemes : ℕ) :
  total_candidates = 10 →
  male_students + female_students = total_candidates →
  male_students > female_students →
  selected_male = 2 →
  selected_female = 2 →
  num_camps = 3 →
  total_schemes = 3240 →
  (male_students.choose selected_male * female_students.choose selected_female * 
   (selected_male + selected_female).choose num_camps * num_camps.factorial = total_schemes) →
  male_students = 6 := by
sorry

end number_of_male_students_l542_54229


namespace janice_earnings_l542_54239

/-- Represents Janice's work schedule and earnings --/
structure WorkSchedule where
  regularDays : ℕ
  regularPayPerDay : ℕ
  overtimeShifts : ℕ
  overtimePay : ℕ

/-- Calculates the total earnings for the week --/
def totalEarnings (schedule : WorkSchedule) : ℕ :=
  schedule.regularDays * schedule.regularPayPerDay + schedule.overtimeShifts * schedule.overtimePay

/-- Janice's work schedule for the week --/
def janiceSchedule : WorkSchedule :=
  { regularDays := 5
  , regularPayPerDay := 30
  , overtimeShifts := 3
  , overtimePay := 15 }

/-- Theorem stating that Janice's total earnings for the week equal $195 --/
theorem janice_earnings : totalEarnings janiceSchedule = 195 := by
  sorry

end janice_earnings_l542_54239


namespace complex_combination_equality_l542_54267

/-- Given complex numbers A, M, S, P, and Q, prove that their combination equals 6 - 5i -/
theorem complex_combination_equality (A M S P Q : ℂ) : 
  A = 5 - 4*I ∧ 
  M = -5 + 2*I ∧ 
  S = 2*I ∧ 
  P = 3 ∧ 
  Q = 1 + I → 
  A - M + S - P - Q = 6 - 5*I :=
by sorry

end complex_combination_equality_l542_54267


namespace power_of_product_cube_l542_54297

theorem power_of_product_cube (m : ℝ) : (2 * m^2)^3 = 8 * m^6 := by
  sorry

end power_of_product_cube_l542_54297


namespace concert_revenue_l542_54287

theorem concert_revenue (ticket_price : ℝ) (first_group_size : ℕ) (second_group_size : ℕ) 
  (first_discount : ℝ) (second_discount : ℝ) (total_buyers : ℕ) :
  ticket_price = 20 →
  first_group_size = 10 →
  second_group_size = 20 →
  first_discount = 0.4 →
  second_discount = 0.15 →
  total_buyers = 56 →
  let first_group_revenue := first_group_size * (ticket_price * (1 - first_discount))
  let second_group_revenue := second_group_size * (ticket_price * (1 - second_discount))
  let remaining_buyers := total_buyers - first_group_size - second_group_size
  let remaining_revenue := remaining_buyers * ticket_price
  let total_revenue := first_group_revenue + second_group_revenue + remaining_revenue
  total_revenue = 980 := by sorry

end concert_revenue_l542_54287


namespace correct_relative_pronouns_l542_54270

/-- A type representing relative pronouns -/
inductive RelativePronoun
  | What
  | Where
  | That
  | Which

/-- A function that checks if a relative pronoun introduces a defining clause without an antecedent -/
def introduces_defining_clause_without_antecedent (rp : RelativePronoun) : Prop :=
  match rp with
  | RelativePronoun.What => True
  | _ => False

/-- A function that checks if a relative pronoun introduces a clause describing a location or circumstance -/
def introduces_location_clause (rp : RelativePronoun) : Prop :=
  match rp with
  | RelativePronoun.Where => True
  | _ => False

theorem correct_relative_pronouns :
  ∃ (rp1 rp2 : RelativePronoun),
    introduces_defining_clause_without_antecedent rp1 ∧
    introduces_location_clause rp2 ∧
    rp1 = RelativePronoun.What ∧
    rp2 = RelativePronoun.Where :=
by
  sorry

end correct_relative_pronouns_l542_54270


namespace other_root_of_quadratic_l542_54281

theorem other_root_of_quadratic (b : ℝ) : 
  (1 : ℝ)^2 + b*(1 : ℝ) - 2 = 0 → 
  ∃ (x : ℝ), x ≠ 1 ∧ x^2 + b*x - 2 = 0 ∧ x = -2 := by
sorry

end other_root_of_quadratic_l542_54281


namespace parabola_vertex_l542_54247

/-- Given a quadratic function f(x) = -x^2 + cx + d where f(x) ≤ 0 has solutions [1,∞) and (-∞,-7],
    prove that the vertex of the parabola is (-3, 16) -/
theorem parabola_vertex (c d : ℝ) 
    (h1 : ∀ x ≥ 1, -x^2 + c*x + d ≤ 0)
    (h2 : ∀ x ≤ -7, -x^2 + c*x + d ≤ 0)
    (h3 : ∃ x > -7, -x^2 + c*x + d > 0)
    (h4 : ∃ x < 1, -x^2 + c*x + d > 0) :
    let f := fun x => -x^2 + c*x + d
    let vertex := (-3, 16)
    ∀ x, f x ≤ f vertex.1 ∧ f vertex.1 = vertex.2 :=
by sorry

end parabola_vertex_l542_54247


namespace episode_length_proof_l542_54218

/-- Represents the length of a single episode in minutes -/
def episode_length : ℕ := 33

/-- Represents the total number of episodes watched in a week -/
def total_episodes : ℕ := 8

/-- Represents the minutes watched on Monday -/
def monday_minutes : ℕ := 138

/-- Represents the minutes watched on Thursday -/
def thursday_minutes : ℕ := 21

/-- Represents the number of episodes watched on Friday -/
def friday_episodes : ℕ := 2

/-- Represents the minutes watched over the weekend -/
def weekend_minutes : ℕ := 105

/-- Proves that the given episode length satisfies the conditions of the problem -/
theorem episode_length_proof : 
  monday_minutes + thursday_minutes + weekend_minutes = total_episodes * episode_length := by
  sorry

end episode_length_proof_l542_54218


namespace polynomial_roots_degree_zero_l542_54286

theorem polynomial_roots_degree_zero (F : Type*) [Field F] :
  ∀ (P : Polynomial F),
    (∃ (S : Finset F), (∀ x ∈ S, P.eval x = 0) ∧ S.card > P.degree) →
    P = 0 := by
  sorry

end polynomial_roots_degree_zero_l542_54286


namespace equation_represents_point_l542_54210

theorem equation_represents_point :
  ∀ x y : ℝ, (Real.sqrt (x - 2) + (y + 2)^2 = 0) ↔ (x = 2 ∧ y = -2) :=
by sorry

end equation_represents_point_l542_54210


namespace triangle_area_rational_l542_54228

theorem triangle_area_rational (x₁ x₂ y₂ : ℤ) :
  ∃ (a b : ℤ), b ≠ 0 ∧ (1/2 : ℚ) * |x₁ + x₂ - x₁*y₂ - x₂*y₂| = a / b :=
sorry

end triangle_area_rational_l542_54228


namespace distribute_planets_l542_54240

/-- The number of ways to distribute units among distinct objects --/
def distribute_units (total_units : ℕ) (earth_like : ℕ) (mars_like : ℕ) (earth_units : ℕ) (mars_units : ℕ) : ℕ :=
  sorry

theorem distribute_planets :
  distribute_units 15 7 8 3 1 = 2961 :=
sorry

end distribute_planets_l542_54240


namespace markers_given_l542_54269

theorem markers_given (initial : ℕ) (total : ℕ) (given : ℕ) : 
  initial = 217 → total = 326 → given = total - initial → given = 109 := by
sorry

end markers_given_l542_54269


namespace sin_neg_seven_pi_sixth_l542_54285

theorem sin_neg_seven_pi_sixth : Real.sin (-7 * π / 6) = 1 / 2 := by
  sorry

end sin_neg_seven_pi_sixth_l542_54285


namespace f_is_quadratic_l542_54262

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 2x + 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- Theorem: f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry

end f_is_quadratic_l542_54262


namespace min_value_expression_l542_54253

theorem min_value_expression :
  ∀ x y : ℝ,
  (Real.sqrt (2 * (1 + Real.cos (2 * x))) - Real.sqrt (9 - Real.sqrt 7) * Real.sin x + 1) *
  (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos y - Real.cos (2 * y)) ≥ -19 ∧
  ∃ x₀ y₀ : ℝ,
  (Real.sqrt (2 * (1 + Real.cos (2 * x₀))) - Real.sqrt (9 - Real.sqrt 7) * Real.sin x₀ + 1) *
  (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos y₀ - Real.cos (2 * y₀)) = -19 :=
by sorry

end min_value_expression_l542_54253


namespace quadratic_increasing_iff_m_gt_one_l542_54268

/-- A quadratic function of the form y = x^2 + (m-3)x + m + 1 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-3)*x + m + 1

/-- The derivative of the quadratic function with respect to x -/
def quadratic_derivative (m : ℝ) (x : ℝ) : ℝ := 2*x + (m-3)

theorem quadratic_increasing_iff_m_gt_one (m : ℝ) :
  (∀ x > 1, ∀ h > 0, quadratic_function m (x + h) > quadratic_function m x) ↔ m > 1 :=
sorry

end quadratic_increasing_iff_m_gt_one_l542_54268


namespace value_of_c_l542_54220

theorem value_of_c : 1996 * 19971997 - 1995 * 19961996 = 3995992 := by
  sorry

end value_of_c_l542_54220


namespace current_rate_calculation_l542_54277

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  downstream : ℝ
  upstream : ℝ
  stillWater : ℝ

/-- Calculates the rate of the current given rowing speeds -/
def currentRate (speed : RowingSpeed) : ℝ :=
  speed.downstream - speed.stillWater

theorem current_rate_calculation (speed : RowingSpeed) 
  (h1 : speed.downstream = 24)
  (h2 : speed.upstream = 7)
  (h3 : speed.stillWater = 15.5) :
  currentRate speed = 8.5 := by
  sorry

#eval currentRate { downstream := 24, upstream := 7, stillWater := 15.5 }

end current_rate_calculation_l542_54277


namespace cos_135_degrees_l542_54261

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end cos_135_degrees_l542_54261


namespace enfeoffment_probability_l542_54299

/-- The number of nobility levels in ancient China --/
def nobility_levels : ℕ := 5

/-- The probability that two people are not enfeoffed at the same level --/
def prob_different_levels : ℚ := 4/5

/-- Theorem stating the probability of two people being enfeoffed at different levels --/
theorem enfeoffment_probability :
  (1 : ℚ) - (nobility_levels : ℚ) / (nobility_levels^2 : ℚ) = prob_different_levels :=
by sorry

end enfeoffment_probability_l542_54299


namespace eva_last_when_start_vasya_l542_54212

/-- Represents the children in the circle -/
inductive Child : Type
| Anya : Child
| Borya : Child
| Vasya : Child
| Gena : Child
| Dasha : Child
| Eva : Child
| Zhenya : Child

/-- The number of children in the circle -/
def num_children : Nat := 7

/-- The step size for elimination -/
def step_size : Nat := 3

/-- Function to determine the last remaining child given a starting position -/
def last_remaining (start : Child) : Child :=
  sorry

/-- Theorem stating that starting from Vasya results in Eva being the last remaining -/
theorem eva_last_when_start_vasya :
  last_remaining Child.Vasya = Child.Eva :=
sorry

end eva_last_when_start_vasya_l542_54212


namespace power_difference_evaluation_l542_54264

theorem power_difference_evaluation : (3^4)^3 - (4^3)^4 = -16245775 := by sorry

end power_difference_evaluation_l542_54264


namespace circle_through_pole_equation_l542_54259

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  center : PolarPoint
  radius : ℝ

/-- The equation of a circle in polar coordinates -/
def circleEquation (c : PolarCircle) (p : PolarPoint) : Prop :=
  p.r = 2 * c.radius * Real.cos (p.θ - c.center.θ)

theorem circle_through_pole_equation 
  (c : PolarCircle) 
  (h1 : c.center = PolarPoint.mk (Real.sqrt 2) 0) 
  (h2 : c.radius = Real.sqrt 2) :
  ∀ (p : PolarPoint), circleEquation c p ↔ p.r = 2 * Real.sqrt 2 * Real.cos p.θ :=
by sorry

end circle_through_pole_equation_l542_54259


namespace tims_change_l542_54224

def initial_amount : ℚ := 1.50
def candy_cost : ℚ := 0.45
def chips_cost : ℚ := 0.65
def toy_cost : ℚ := 0.40
def discount_rate : ℚ := 0.10

def total_snacks_cost : ℚ := candy_cost + chips_cost
def discounted_snacks_cost : ℚ := total_snacks_cost * (1 - discount_rate)
def total_cost : ℚ := discounted_snacks_cost + toy_cost
def change : ℚ := initial_amount - total_cost

theorem tims_change : change = 0.11 := by sorry

end tims_change_l542_54224


namespace original_group_size_l542_54288

theorem original_group_size (n : ℕ) (W : ℝ) : 
  W = n * 35 →
  W + 40 = (n + 1) * 36 →
  n = 4 := by
sorry

end original_group_size_l542_54288


namespace expected_lotus_seed_zongzi_l542_54280

theorem expected_lotus_seed_zongzi 
  (total_zongzi : ℕ) 
  (lotus_seed_zongzi : ℕ) 
  (selected_zongzi : ℕ) 
  (h1 : total_zongzi = 180) 
  (h2 : lotus_seed_zongzi = 54) 
  (h3 : selected_zongzi = 10) :
  (selected_zongzi : ℚ) * (lotus_seed_zongzi : ℚ) / (total_zongzi : ℚ) = 3 := by
  sorry

end expected_lotus_seed_zongzi_l542_54280


namespace g_composition_points_sum_l542_54236

/-- Given a function g with specific values, prove the existence of points on g(g(x)) with a certain sum property -/
theorem g_composition_points_sum (g : ℝ → ℝ) 
  (h1 : g 2 = 4) (h2 : g 3 = 2) (h3 : g 4 = 6) :
  ∃ (p q r s : ℝ), g (g p) = q ∧ g (g r) = s ∧ p * q + r * s = 24 := by
  sorry

end g_composition_points_sum_l542_54236


namespace no_roots_of_composition_if_no_roots_l542_54294

/-- A quadratic polynomial -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Theorem: If p(x) = x has no real roots, then p(p(x)) = x has no real roots -/
theorem no_roots_of_composition_if_no_roots (a b c : ℝ) :
  (∀ x : ℝ, QuadraticPolynomial a b c x ≠ x) →
  (∀ x : ℝ, QuadraticPolynomial a b c (QuadraticPolynomial a b c x) ≠ x) := by
  sorry


end no_roots_of_composition_if_no_roots_l542_54294


namespace sum_base4_to_base10_l542_54208

/-- Converts a base 4 number represented as a list of digits to base 10 -/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The sum of 2213₄, 2703₄, and 1531₄ in base 10 is 309 -/
theorem sum_base4_to_base10 :
  base4ToBase10 [3, 1, 2, 2] + base4ToBase10 [3, 0, 7, 2] + base4ToBase10 [1, 3, 5, 1] = 309 := by
  sorry

#eval base4ToBase10 [3, 1, 2, 2] + base4ToBase10 [3, 0, 7, 2] + base4ToBase10 [1, 3, 5, 1]

end sum_base4_to_base10_l542_54208


namespace sum_bound_l542_54279

theorem sum_bound (w x y z : ℝ) 
  (sum_zero : w + x + y + z = 0) 
  (sum_squares_one : w^2 + x^2 + y^2 + z^2 = 1) : 
  -1 ≤ w*x + x*y + y*z + z*w ∧ w*x + x*y + y*z + z*w ≤ 0 := by
  sorry

end sum_bound_l542_54279


namespace parabola_intersection_sum_l542_54274

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (x y : ℝ) : Prop :=
  ∃ (t : ℝ), x = 1 + t ∧ y = t

-- Define the intersection points
structure IntersectionPoint where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y
  on_line : line_through_focus x y

-- Theorem statement
theorem parabola_intersection_sum (A B : IntersectionPoint) :
  (A.x - (-1))^2 + (A.y)^2 + (B.x - (-1))^2 + (B.y)^2 = 64 →
  A.x + B.x = 6 := by sorry

end parabola_intersection_sum_l542_54274


namespace negation_of_universal_statement_l542_54234

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end negation_of_universal_statement_l542_54234


namespace ellipse_tangent_quadrilateral_area_l542_54275

/-- Given an ellipse with equation 9x^2 + 25y^2 = 225, 
    the area of the quadrilateral formed by tangents at the parameter endpoints is 62.5 -/
theorem ellipse_tangent_quadrilateral_area :
  let a : ℝ := 5  -- semi-major axis
  let b : ℝ := 3  -- semi-minor axis
  ∀ x y : ℝ, 9 * x^2 + 25 * y^2 = 225 →
  let area := 2 * a^3 / Real.sqrt (a^2 - b^2)
  area = 62.5 := by
sorry


end ellipse_tangent_quadrilateral_area_l542_54275


namespace minimum_packages_for_equal_shipment_l542_54291

theorem minimum_packages_for_equal_shipment (sarah_capacity : Nat) (ryan_capacity : Nat) (emily_capacity : Nat)
  (h1 : sarah_capacity = 18)
  (h2 : ryan_capacity = 11)
  (h3 : emily_capacity = 15) :
  Nat.lcm (Nat.lcm sarah_capacity ryan_capacity) emily_capacity = 990 :=
by sorry

end minimum_packages_for_equal_shipment_l542_54291


namespace factor_theorem_application_l542_54250

theorem factor_theorem_application (d : ℚ) :
  (∀ x : ℚ, (x - 3) ∣ (x^3 + 3*x^2 + d*x + 8)) → d = -62/3 := by
  sorry

end factor_theorem_application_l542_54250


namespace inequality_proof_l542_54226

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^2 + x*y + y^2 ≤ 3*(x - Real.sqrt (x*y) + y)^2 := by
  sorry

end inequality_proof_l542_54226


namespace queue_arrangements_l542_54254

/-- Represents the number of people in each category -/
def num_fathers : ℕ := 2
def num_mothers : ℕ := 2
def num_children : ℕ := 2

/-- The total number of people -/
def total_people : ℕ := num_fathers + num_mothers + num_children

/-- Represents the constraint that fathers must be at the beginning and end -/
def fathers_fixed : ℕ := 2

/-- Represents the number of units to arrange between fathers (2 mothers and 1 children unit) -/
def units_between : ℕ := num_mothers + 1

/-- Represents the number of ways to arrange children within their unit -/
def children_arrangements : ℕ := Nat.factorial num_children

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := fathers_fixed * Nat.factorial units_between * children_arrangements

/-- Theorem stating that the number of possible arrangements is 24 -/
theorem queue_arrangements : total_arrangements = 24 := by
  sorry

end queue_arrangements_l542_54254


namespace max_x_value_l542_54215

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 7) 
  (prod_sum_eq : x * y + x * z + y * z = 12) : 
  x ≤ (14 + 2 * Real.sqrt 46) / 6 := by
sorry

end max_x_value_l542_54215


namespace problem_solution_l542_54293

/-- The probability that person A can solve the problem within half an hour -/
def prob_A : ℚ := 1/2

/-- The probability that person B can solve the problem within half an hour -/
def prob_B : ℚ := 1/3

/-- The probability that neither A nor B solves the problem -/
def prob_neither_solves : ℚ := (1 - prob_A) * (1 - prob_B)

/-- The probability that the problem is solved -/
def prob_problem_solved : ℚ := 1 - prob_neither_solves

theorem problem_solution :
  prob_neither_solves = 1/3 ∧ prob_problem_solved = 2/3 := by
  sorry

end problem_solution_l542_54293


namespace weighted_average_is_38_5_l542_54242

/-- Represents the marks in different subjects -/
structure Marks where
  mathematics : ℝ
  physics : ℝ
  chemistry : ℝ
  biology : ℝ

/-- Calculates the weighted average of Mathematics, Chemistry, and Biology marks -/
def weightedAverage (m : Marks) : ℝ :=
  0.4 * m.mathematics + 0.3 * m.chemistry + 0.3 * m.biology

/-- Theorem stating that under given conditions, the weighted average is 38.5 -/
theorem weighted_average_is_38_5 (m : Marks) :
  m.mathematics + m.physics + m.biology = 90 ∧
  m.chemistry = m.physics + 10 ∧
  m.biology = m.chemistry - 5 →
  weightedAverage m = 38.5 := by
  sorry

#eval weightedAverage { mathematics := 85, physics := 0, chemistry := 10, biology := 5 }

end weighted_average_is_38_5_l542_54242


namespace rectangle_to_cylinders_volume_ratio_l542_54260

theorem rectangle_to_cylinders_volume_ratio :
  let rectangle_width : ℝ := 7
  let rectangle_length : ℝ := 10
  let cylinder1_radius : ℝ := rectangle_width / (2 * Real.pi)
  let cylinder1_height : ℝ := rectangle_length
  let cylinder1_volume : ℝ := Real.pi * cylinder1_radius^2 * cylinder1_height
  let cylinder2_radius : ℝ := rectangle_length / (2 * Real.pi)
  let cylinder2_height : ℝ := rectangle_width
  let cylinder2_volume : ℝ := Real.pi * cylinder2_radius^2 * cylinder2_height
  let larger_volume : ℝ := max cylinder1_volume cylinder2_volume
  let smaller_volume : ℝ := min cylinder1_volume cylinder2_volume
  larger_volume / smaller_volume = 10 / 7 := by sorry

end rectangle_to_cylinders_volume_ratio_l542_54260


namespace expression_bounds_l542_54296

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
  4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ∧
  Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := by
  sorry


end expression_bounds_l542_54296


namespace total_gray_trees_count_l542_54246

/-- Represents an aerial photo with tree counts -/
structure AerialPhoto where
  totalTrees : ℕ
  whiteTrees : ℕ

/-- Calculates the number of trees in the gray area of a photo -/
def grayTrees (photo : AerialPhoto) : ℕ :=
  photo.totalTrees - photo.whiteTrees

theorem total_gray_trees_count 
  (photo1 photo2 photo3 : AerialPhoto)
  (h1 : photo1.totalTrees = 100)
  (h2 : photo1.whiteTrees = 82)
  (h3 : photo2.totalTrees = 90)
  (h4 : photo2.whiteTrees = 82)
  (h5 : photo3.whiteTrees = 75)
  (h6 : photo1.totalTrees = photo2.totalTrees)
  (h7 : photo2.totalTrees = photo3.totalTrees) :
  grayTrees photo1 + grayTrees photo2 + grayTrees photo3 = 26 := by
  sorry

end total_gray_trees_count_l542_54246


namespace cos_function_property_l542_54248

theorem cos_function_property (x : ℝ) (n : ℤ) (f : ℝ → ℝ) 
  (h : f (Real.sin x) = Real.sin ((4 * ↑n + 1) * x)) :
  f (Real.cos x) = Real.cos ((4 * ↑n + 1) * x) := by
  sorry

end cos_function_property_l542_54248


namespace no_geometric_sequence_sqrt235_l542_54266

theorem no_geometric_sequence_sqrt235 :
  ¬∃ (m n : ℕ) (q : ℝ), m > n ∧ n > 1 ∧ q > 0 ∧
    Real.sqrt 3 = q ^ n * Real.sqrt 2 ∧
    Real.sqrt 5 = q ^ m * Real.sqrt 2 := by
  sorry

end no_geometric_sequence_sqrt235_l542_54266


namespace largest_divisor_of_expression_l542_54252

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (60 : ℤ) = Nat.gcd 25920 213840 ∧
  ∀ (k : ℤ), k ∣ (15*x + 9) * (15*x + 15) * (15*x + 21) → k ≤ 60 := by
  sorry

end largest_divisor_of_expression_l542_54252


namespace square_plus_double_sqrt2_minus_1_l542_54284

theorem square_plus_double_sqrt2_minus_1 :
  let x : ℝ := Real.sqrt 2 - 1
  x^2 + 2*x = 1 := by sorry

end square_plus_double_sqrt2_minus_1_l542_54284


namespace sqrt_450_simplification_l542_54205

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplification_l542_54205


namespace reflection_about_x_axis_l542_54276

theorem reflection_about_x_axis (a : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    A = (3, a) ∧ 
    B = (3, 4) ∧ 
    A.1 = B.1 ∧ 
    A.2 = -B.2) → 
  a = -4 := by sorry

end reflection_about_x_axis_l542_54276


namespace sally_seashells_l542_54263

theorem sally_seashells (total tom jessica : ℕ) (h1 : total = 21) (h2 : tom = 7) (h3 : jessica = 5) :
  total - tom - jessica = 9 := by
  sorry

end sally_seashells_l542_54263


namespace simplify_fraction_l542_54245

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) : 
  (2 / (x^2 - 2*x + 1) - 1 / (x^2 - x)) / ((x + 1) / (2*x^2 - 2*x)) = 2 / (x - 1) :=
by sorry

end simplify_fraction_l542_54245


namespace leftover_coin_value_l542_54243

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The number of quarters in a complete roll -/
def quarters_per_roll : ℕ := 35

/-- The number of dimes in a complete roll -/
def dimes_per_roll : ℕ := 55

/-- James' quarters -/
def james_quarters : ℕ := 97

/-- James' dimes -/
def james_dimes : ℕ := 173

/-- Lindsay's quarters -/
def lindsay_quarters : ℕ := 141

/-- Lindsay's dimes -/
def lindsay_dimes : ℕ := 289

/-- The total number of quarters -/
def total_quarters : ℕ := james_quarters + lindsay_quarters

/-- The total number of dimes -/
def total_dimes : ℕ := james_dimes + lindsay_dimes

theorem leftover_coin_value :
  (total_quarters % quarters_per_roll : ℚ) * quarter_value +
  (total_dimes % dimes_per_roll : ℚ) * dime_value = 92 / 10 := by
  sorry

end leftover_coin_value_l542_54243


namespace altitude_sum_of_triangle_l542_54209

/-- The sum of altitudes of a triangle formed by the line 15x + 3y = 45 and the coordinate axes --/
theorem altitude_sum_of_triangle (x y : ℝ) : 
  (15 * x + 3 * y = 45) →  -- Line equation
  ∃ (a b c : ℝ), -- Altitudes
    (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) ∧ -- Altitudes are non-negative
    (a + b + c = (18 * Real.sqrt 26 + 15) / Real.sqrt 26) ∧ -- Sum of altitudes
    (∃ (x₁ y₁ : ℝ), 15 * x₁ + 3 * y₁ = 45 ∧ x₁ ≥ 0 ∧ y₁ ≥ 0) -- Triangle exists in the first quadrant
    :=
by sorry

end altitude_sum_of_triangle_l542_54209


namespace smallest_integer_in_ratio_l542_54233

theorem smallest_integer_in_ratio (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 60 →
  2 * a = 3 * b →
  3 * b = 5 * c →
  a = 6 := by
sorry

end smallest_integer_in_ratio_l542_54233


namespace sin_15_minus_sin_75_fourth_power_l542_54207

theorem sin_15_minus_sin_75_fourth_power :
  Real.sin (15 * π / 180) ^ 4 - Real.sin (75 * π / 180) ^ 4 = -(Real.sqrt 3) / 2 := by
  sorry

end sin_15_minus_sin_75_fourth_power_l542_54207


namespace f_2013_pi_third_l542_54257

open Real

noncomputable def f₀ (x : ℝ) : ℝ := sin x - cos x

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => f₀ x
  | n + 1 => deriv (f n) x

theorem f_2013_pi_third : f 2013 (π/3) = (1 + Real.sqrt 3) / 2 := by sorry

end f_2013_pi_third_l542_54257


namespace power_mod_45_l542_54265

theorem power_mod_45 : 14^100 % 45 = 31 := by
  sorry

end power_mod_45_l542_54265


namespace favorite_books_probability_l542_54227

variable (n : ℕ) (k : ℕ)

def P (n k : ℕ) : ℚ := (k.factorial * (n - k + 1).factorial) / n.factorial

theorem favorite_books_probability (h : k ≤ n) :
  (∀ m, m ≤ n → P n k ≥ P n m) ↔ (k = 1 ∨ k = n) ∧
  (n % 2 = 0 → P n k ≤ P n (n / 2)) ∧
  (n % 2 ≠ 0 → P n k ≤ P n ((n + 1) / 2)) :=
sorry

end favorite_books_probability_l542_54227


namespace complex_reciprocal_sum_l542_54244

theorem complex_reciprocal_sum (z z₁ z₂ : ℂ) : 
  z₁ = 5 + 10*I → z₂ = 3 - 4*I → (1 : ℂ)/z = 1/z₁ + 1/z₂ → z = 5 - (5/2)*I :=
by
  sorry

end complex_reciprocal_sum_l542_54244


namespace system_solution_l542_54217

theorem system_solution :
  ∃! (x y : ℚ), 2 * x - 3 * y = 5 ∧ 4 * x - 6 * y = 10 ∧ x + 2 * y = 7 ∧ x = 31/7 ∧ y = 9/7 := by
  sorry

end system_solution_l542_54217


namespace pages_read_difference_l542_54256

/-- The number of weeks required for Janet to read 2100 more pages than Belinda,
    given that Janet reads 80 pages a day and Belinda reads 30 pages a day. -/
theorem pages_read_difference (janet_daily : ℕ) (belinda_daily : ℕ) (total_difference : ℕ) :
  janet_daily = 80 →
  belinda_daily = 30 →
  total_difference = 2100 →
  (total_difference / ((janet_daily - belinda_daily) * 7) : ℚ) = 6 := by
  sorry

end pages_read_difference_l542_54256


namespace parabola_equation_from_hyperbola_l542_54214

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the parabola -/
theorem parabola_equation_from_hyperbola (x y : ℝ) :
  (x^2 / 3 - y^2 = 1) →  -- Given hyperbola equation
  (∃ (p : ℝ), 
    (p > 0) ∧  -- p is positive for a right-opening parabola
    ((2 : ℝ) = p / 2) ∧  -- Focus of parabola is at (2, 0), which is (p/2, 0) in standard form
    (y^2 = 2 * p * x))  -- Standard form of parabola equation
  →
  y^2 = 8 * x  -- Conclusion: specific equation of the parabola
:= by sorry

end parabola_equation_from_hyperbola_l542_54214


namespace no_country_with_100_roads_and_3_per_city_l542_54298

theorem no_country_with_100_roads_and_3_per_city :
  ¬ ∃ (n : ℕ), 3 * n = 200 :=
by sorry

end no_country_with_100_roads_and_3_per_city_l542_54298


namespace max_triangle_area_in_rectangle_l542_54216

/-- The maximum area of a right triangle with a 30° angle inside a 12x5 rectangle -/
theorem max_triangle_area_in_rectangle :
  let rectangle_length : ℝ := 12
  let rectangle_width : ℝ := 5
  let angle : ℝ := 30 * π / 180  -- 30° in radians
  ∃ (triangle_area : ℝ),
    triangle_area = 25 * Real.sqrt 3 / 4 ∧
    ∀ (a : ℝ), a ≤ rectangle_width →
      a * (2 * a) / 2 ≤ triangle_area :=
by sorry

end max_triangle_area_in_rectangle_l542_54216


namespace average_equation_solution_l542_54202

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((2*x + 5) + (8*x + 3) + (3*x + 8)) = 5*x - 10 → x = 23 := by
  sorry

end average_equation_solution_l542_54202


namespace increasing_function_bounds_l542_54237

theorem increasing_function_bounds (k : ℕ+) (f : ℕ+ → ℕ+) 
  (h_increasing : ∀ m n : ℕ+, m < n → f m < f n)
  (h_composition : ∀ n : ℕ+, f (f n) = k * n) :
  ∀ n : ℕ+, (2 * k : ℚ) / (k + 1) * n ≤ f n ∧ (f n : ℚ) ≤ (k + 1) / 2 * n :=
by sorry

end increasing_function_bounds_l542_54237


namespace sin_difference_product_l542_54278

theorem sin_difference_product (a b : ℝ) : 
  Real.sin (2 * a + b) - Real.sin (2 * a - b) = 2 * Real.cos (2 * a) * Real.sin b := by
  sorry

end sin_difference_product_l542_54278


namespace haley_initial_trees_l542_54272

/-- The number of trees that died during the typhoon -/
def trees_died : ℕ := 2

/-- The number of trees left after the typhoon -/
def trees_left : ℕ := 10

/-- The initial number of trees before the typhoon -/
def initial_trees : ℕ := trees_left + trees_died

theorem haley_initial_trees : initial_trees = 12 := by
  sorry

end haley_initial_trees_l542_54272


namespace marbles_problem_l542_54232

theorem marbles_problem (total : ℕ) (given_to_sister : ℕ) : 
  (given_to_sister = total / 6) →
  (given_to_sister = 9) →
  (total - (total / 2 + total / 6) = 18) :=
by sorry

end marbles_problem_l542_54232


namespace toy_position_l542_54289

/-- Given a row of toys, this function calculates the position from the left
    based on the total number of toys and the position from the right. -/
def position_from_left (total : ℕ) (position_from_right : ℕ) : ℕ :=
  total - position_from_right + 1

/-- Theorem stating that in a row of 19 toys, 
    if a toy is 8th from the right, it is 12th from the left. -/
theorem toy_position (total : ℕ) (position_from_right : ℕ) 
  (h1 : total = 19) (h2 : position_from_right = 8) : 
  position_from_left total position_from_right = 12 := by
  sorry

end toy_position_l542_54289
