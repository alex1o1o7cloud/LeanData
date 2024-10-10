import Mathlib

namespace painted_cubes_l4024_402414

theorem painted_cubes (total_cubes : ℕ) (unpainted_cubes : ℕ) (side_length : ℕ) : 
  unpainted_cubes = 24 →
  side_length = 5 →
  total_cubes = side_length^3 →
  total_cubes - unpainted_cubes = 101 := by
  sorry

end painted_cubes_l4024_402414


namespace pencils_given_l4024_402485

theorem pencils_given (initial : ℕ) (final : ℕ) (given : ℕ) : 
  initial = 51 → final = 57 → given = final - initial → given = 6 := by
  sorry

end pencils_given_l4024_402485


namespace coefficient_x_squared_in_expansion_l4024_402493

theorem coefficient_x_squared_in_expansion :
  (Finset.range 5).sum (fun k => (Nat.choose 4 k : ℤ) * (-2)^(4 - k) * (if k = 2 then 1 else 0)) = 24 := by
  sorry

end coefficient_x_squared_in_expansion_l4024_402493


namespace find_z_l4024_402484

theorem find_z (x y z : ℚ) 
  (h1 : x = (1/3) * y) 
  (h2 : y = (1/4) * z) 
  (h3 : x + y = 16) : 
  z = 48 := by
sorry

end find_z_l4024_402484


namespace min_max_f_l4024_402412

def a : ℕ := 2001

def A : Set (ℕ × ℕ) :=
  {p | let m := p.1
       let n := p.2
       m < 2 * a ∧
       (2 * n) ∣ (2 * a * m - m^2 + n^2) ∧
       n^2 - m^2 + 2 * m * n ≤ 2 * a * (n - m)}

def f (p : ℕ × ℕ) : ℚ :=
  let m := p.1
  let n := p.2
  (2 * a * m - m^2 - m * n) / n

theorem min_max_f :
  ∃ (min max : ℚ), min = 2 ∧ max = 3750 ∧
  (∀ p ∈ A, min ≤ f p ∧ f p ≤ max) ∧
  (∃ p₁ ∈ A, f p₁ = min) ∧
  (∃ p₂ ∈ A, f p₂ = max) :=
sorry

end min_max_f_l4024_402412


namespace distance_between_given_lines_l4024_402476

/-- Two parallel lines in 2D space -/
structure ParallelLines where
  a : ℝ × ℝ  -- Point on first line
  b : ℝ × ℝ  -- Point on second line
  d : ℝ × ℝ  -- Direction vector (same for both lines)

/-- The distance between two parallel lines -/
def distance (lines : ParallelLines) : ℝ :=
  sorry

/-- Theorem stating that the distance between the given parallel lines is 0 -/
theorem distance_between_given_lines :
  let lines : ParallelLines := {
    a := (3, -4),
    b := (2, -1),
    d := (-1, 3)
  }
  distance lines = 0 := by sorry

end distance_between_given_lines_l4024_402476


namespace quadratic_inequality_l4024_402421

theorem quadratic_inequality (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by
  sorry

end quadratic_inequality_l4024_402421


namespace line_not_in_second_quadrant_iff_l4024_402481

/-- The line equation in terms of a, x, and y -/
def line_equation (a x y : ℝ) : Prop :=
  (3*a - 1)*x + (2 - a)*y - 1 = 0

/-- The second quadrant -/
def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- The line does not pass through the second quadrant -/
def not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y, line_equation a x y → ¬ second_quadrant x y

/-- The main theorem -/
theorem line_not_in_second_quadrant_iff (a : ℝ) :
  not_in_second_quadrant a ↔ a ≥ 2 := by sorry

end line_not_in_second_quadrant_iff_l4024_402481


namespace opposite_reciprocal_fraction_l4024_402489

theorem opposite_reciprocal_fraction (a b c d : ℝ) 
  (h1 : a + b = 0) -- a and b are opposite numbers
  (h2 : c * d = 1) -- c and d are reciprocals
  : (5*a + 5*b - 7*c*d) / ((-c*d)^3) = 7 := by sorry

end opposite_reciprocal_fraction_l4024_402489


namespace inequality_solution_set_l4024_402409

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ 
  a ∈ Set.Ioc (-3/5) 1 :=
by sorry

end inequality_solution_set_l4024_402409


namespace age_ratio_proof_l4024_402454

theorem age_ratio_proof (parent_age son_age : ℕ) : 
  parent_age = 45 →
  son_age = 15 →
  parent_age + 5 = (5/2) * (son_age + 5) →
  parent_age / son_age = 3 := by
sorry

end age_ratio_proof_l4024_402454


namespace units_digit_of_m_squared_plus_two_to_m_l4024_402458

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : (m^2 + 2^m) % 10 = 3 :=
sorry

end units_digit_of_m_squared_plus_two_to_m_l4024_402458


namespace equation_solution_l4024_402465

theorem equation_solution (x y r s : ℚ) : 
  (3 * x + 2 * y = 16) → 
  (5 * x + 3 * y = 26) → 
  (r = x) → 
  (s = y) → 
  (r - s = 2) := by
sorry

end equation_solution_l4024_402465


namespace evaluate_expression_l4024_402437

theorem evaluate_expression (x y : ℝ) (hx : x = 4) (hy : y = 2) :
  y * (y - 2 * x)^2 = 72 := by
  sorry

end evaluate_expression_l4024_402437


namespace quadratic_equation_properties_l4024_402434

theorem quadratic_equation_properties :
  ∀ (k : ℝ), 
  -- The equation has two distinct real roots
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 2 * x₁^2 + k * x₁ - 1 = 0 ∧ 2 * x₂^2 + k * x₂ - 1 = 0) ∧
  -- When one root is -1, the other is 1/2 and k = 1
  (2 * (-1)^2 + k * (-1) - 1 = 0 → k = 1 ∧ 2 * (1/2)^2 + 1 * (1/2) - 1 = 0) :=
by sorry


end quadratic_equation_properties_l4024_402434


namespace triangle_angle_A_is_30_degrees_l4024_402473

theorem triangle_angle_A_is_30_degrees 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : a^2 - b^2 = Real.sqrt 3 * b * c)
  (h2 : Real.sin C = 2 * Real.sqrt 3 * Real.sin B)
  (h3 : 0 < A ∧ A < π)
  (h4 : 0 < B ∧ B < π)
  (h5 : 0 < C ∧ C < π)
  (h6 : A + B + C = π)
  (h7 : a / Real.sin A = b / Real.sin B)
  (h8 : b / Real.sin B = c / Real.sin C)
  : A = π / 6 := by
  sorry

end triangle_angle_A_is_30_degrees_l4024_402473


namespace half_abs_diff_cubes_20_15_l4024_402428

theorem half_abs_diff_cubes_20_15 : 
  (1/2 : ℝ) * |20^3 - 15^3| = 2312.5 := by sorry

end half_abs_diff_cubes_20_15_l4024_402428


namespace parallel_lines_a_equals_one_l4024_402423

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- Given two parallel lines y = ax - 2 and y = (2-a)x + 1, prove that a = 1 -/
theorem parallel_lines_a_equals_one :
  (∀ x y : ℝ, y = a * x - 2 ↔ y = (2 - a) * x + 1) → a = 1 := by
  sorry

end parallel_lines_a_equals_one_l4024_402423


namespace integer_roots_quadratic_l4024_402478

theorem integer_roots_quadratic (a : ℤ) : 
  (∃ x y : ℤ, x^2 - a*x + 9*a = 0 ∧ y^2 - a*y + 9*a = 0 ∧ x ≠ y) ↔ 
  a ∈ ({100, -64, 48, -12, 36, 0} : Set ℤ) :=
sorry

end integer_roots_quadratic_l4024_402478


namespace sin_510_degrees_l4024_402408

theorem sin_510_degrees : Real.sin (510 * π / 180) = 1 / 2 := by
  sorry

end sin_510_degrees_l4024_402408


namespace weight_lifting_duration_l4024_402450

-- Define the total practice time in minutes
def total_practice_time : ℕ := 120

-- Define the time spent on running and weight lifting combined
def run_lift_time : ℕ := total_practice_time / 2

-- Define the relationship between running and weight lifting time
def weight_lifting_time (x : ℕ) : Prop := 
  x + 2 * x = run_lift_time

-- Theorem statement
theorem weight_lifting_duration : 
  ∃ x : ℕ, weight_lifting_time x ∧ x = 20 := by sorry

end weight_lifting_duration_l4024_402450


namespace complex_sum_real_necessary_not_sufficient_l4024_402461

theorem complex_sum_real_necessary_not_sufficient (z₁ z₂ : ℂ) :
  (∃ (a b : ℝ), z₁ = a + b * I ∧ z₂ = a - b * I) → (z₁ + z₂).im = 0 ∧
  ¬(∀ z₁ z₂ : ℂ, (z₁ + z₂).im = 0 → ∃ (a b : ℝ), z₁ = a + b * I ∧ z₂ = a - b * I) :=
by sorry

end complex_sum_real_necessary_not_sufficient_l4024_402461


namespace arithmetic_expression_evaluation_l4024_402400

theorem arithmetic_expression_evaluation : 2 + 7 * 3 - 4 + 8 / 2 = 23 := by
  sorry

end arithmetic_expression_evaluation_l4024_402400


namespace marnie_bracelets_l4024_402460

def beads_per_bracelet : ℕ := 65

def total_beads : ℕ :=
  5 * 50 + 2 * 100 + 3 * 75 + 4 * 125

theorem marnie_bracelets :
  (total_beads / beads_per_bracelet : ℕ) = 18 := by
  sorry

end marnie_bracelets_l4024_402460


namespace coefficient_m4n4_in_expansion_l4024_402435

theorem coefficient_m4n4_in_expansion : ∀ m n : ℕ,
  (Nat.choose 8 4 : ℕ) = 70 := by sorry

end coefficient_m4n4_in_expansion_l4024_402435


namespace total_students_suggestion_l4024_402472

theorem total_students_suggestion (mashed_potatoes bacon tomatoes : ℕ) 
  (h1 : mashed_potatoes = 324)
  (h2 : bacon = 374)
  (h3 : tomatoes = 128) :
  mashed_potatoes + bacon + tomatoes = 826 := by
  sorry

end total_students_suggestion_l4024_402472


namespace parallel_vectors_t_value_l4024_402418

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_t_value :
  let m : ℝ × ℝ := (2, 8)
  let n : ℝ → ℝ × ℝ := fun t ↦ (-4, t)
  ∀ t : ℝ, are_parallel m (n t) → t = -16 :=
by
  sorry

end parallel_vectors_t_value_l4024_402418


namespace coconut_moving_theorem_l4024_402486

/-- The number of coconuts Barbie can carry in one trip -/
def barbie_capacity : ℕ := 4

/-- The number of coconuts Bruno can carry in one trip -/
def bruno_capacity : ℕ := 8

/-- The number of trips Barbie and Bruno make together -/
def num_trips : ℕ := 12

/-- The total number of coconuts Barbie and Bruno can move -/
def total_coconuts : ℕ := (barbie_capacity + bruno_capacity) * num_trips

theorem coconut_moving_theorem : total_coconuts = 144 := by
  sorry

end coconut_moving_theorem_l4024_402486


namespace tip_percentage_calculation_l4024_402462

theorem tip_percentage_calculation (total_bill : ℝ) (billy_tip : ℝ) (billy_percentage : ℝ) :
  total_bill = 50 →
  billy_tip = 8 →
  billy_percentage = 0.8 →
  (billy_tip / billy_percentage) / total_bill = 0.2 := by
  sorry

end tip_percentage_calculation_l4024_402462


namespace noMoreThanOneHead_atLeastTwoHeads_mutually_exclusive_l4024_402487

/-- Represents the outcome of tossing 3 coins -/
inductive CoinToss
  | HHH
  | HHT
  | HTH
  | THH
  | HTT
  | THT
  | TTH
  | TTT

/-- The event of having no more than one head -/
def noMoreThanOneHead (outcome : CoinToss) : Prop :=
  match outcome with
  | CoinToss.HTT | CoinToss.THT | CoinToss.TTH | CoinToss.TTT => True
  | _ => False

/-- The event of having at least two heads -/
def atLeastTwoHeads (outcome : CoinToss) : Prop :=
  match outcome with
  | CoinToss.HHH | CoinToss.HHT | CoinToss.HTH | CoinToss.THH => True
  | _ => False

/-- Theorem stating that "No more than one head" and "At least two heads" are mutually exclusive -/
theorem noMoreThanOneHead_atLeastTwoHeads_mutually_exclusive :
  ∀ (outcome : CoinToss), ¬(noMoreThanOneHead outcome ∧ atLeastTwoHeads outcome) :=
by
  sorry

end noMoreThanOneHead_atLeastTwoHeads_mutually_exclusive_l4024_402487


namespace sum_of_f_l4024_402411

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

theorem sum_of_f (x : ℝ) : f (-x) + f (1 + x) = Real.sqrt 2 / 2 := by
  sorry

end sum_of_f_l4024_402411


namespace population_halving_time_island_l4024_402446

/-- The time it takes for a population to halve given initial population and net emigration rate -/
def time_to_halve_population (initial_population : ℕ) (net_emigration_rate_per_500 : ℚ) : ℚ :=
  let net_emigration_rate := (initial_population : ℚ) / 500 * net_emigration_rate_per_500
  (initial_population : ℚ) / (2 * net_emigration_rate)

theorem population_halving_time_island (ε : ℚ) :
  ∃ (δ : ℚ), δ > 0 ∧ |time_to_halve_population 5000 35 - 7.14| < δ → δ < ε :=
sorry

end population_halving_time_island_l4024_402446


namespace bowtie_equation_solution_l4024_402426

-- Define the ⋈ operation
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ h : ℝ, bowtie 5 h = 11 ∧ h = 30 :=
sorry

end bowtie_equation_solution_l4024_402426


namespace f_at_negative_one_l4024_402488

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^2 + 1

-- Theorem statement
theorem f_at_negative_one : f (-1) = -1 := by
  sorry

end f_at_negative_one_l4024_402488


namespace sum_of_factors_72_l4024_402477

def sum_of_factors (n : ℕ) : ℕ := sorry

theorem sum_of_factors_72 : sum_of_factors 72 = 195 := by
  sorry

end sum_of_factors_72_l4024_402477


namespace line_slope_intercept_product_l4024_402455

/-- Given a line with y-intercept 3 and slope -3/2, prove that the product of its slope and y-intercept is -9/2 -/
theorem line_slope_intercept_product :
  ∀ (m b : ℚ),
    b = 3 →
    m = -3/2 →
    m * b = -9/2 := by
  sorry

end line_slope_intercept_product_l4024_402455


namespace i_power_2013_l4024_402496

theorem i_power_2013 (i : ℂ) (h : i^2 = -1) : i^2013 = i := by
  sorry

end i_power_2013_l4024_402496


namespace other_factor_in_product_l4024_402413

theorem other_factor_in_product (w : ℕ+) (x : ℕ+) : 
  w = 156 →
  (∃ k : ℕ+, k * w * x = 2^5 * 3^3 * 13^2) →
  x = 936 := by
sorry

end other_factor_in_product_l4024_402413


namespace root_absolute_value_greater_than_four_l4024_402479

theorem root_absolute_value_greater_than_four (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + p*r₁ + 16 = 0 → 
  r₂^2 + p*r₂ + 16 = 0 → 
  (abs r₁ > 4) ∨ (abs r₂ > 4) := by
sorry

end root_absolute_value_greater_than_four_l4024_402479


namespace count_propositions_l4024_402424

-- Define a function to check if a statement is a proposition
def isProposition (s : String) : Bool :=
  match s with
  | "|x+2|" => false
  | "-5 ∈ ℤ" => true
  | "π ∉ ℝ" => true
  | "{0} ∈ ℕ" => true
  | _ => false

-- Define the list of statements
def statements : List String := ["|x+2|", "-5 ∈ ℤ", "π ∉ ℝ", "{0} ∈ ℕ"]

-- Theorem to prove
theorem count_propositions :
  (statements.filter isProposition).length = 3 := by sorry

end count_propositions_l4024_402424


namespace apple_ratio_l4024_402401

/-- Prove that the ratio of Harry's apples to Tim's apples is 1:2 -/
theorem apple_ratio :
  ∀ (martha_apples tim_apples harry_apples : ℕ),
    martha_apples = 68 →
    tim_apples = martha_apples - 30 →
    harry_apples = 19 →
    (harry_apples : ℚ) / tim_apples = 1 / 2 := by
  sorry

end apple_ratio_l4024_402401


namespace nut_problem_l4024_402482

theorem nut_problem (sue_nuts : ℕ) (harry_nuts : ℕ) (bill_nuts : ℕ) 
  (h1 : sue_nuts = 48)
  (h2 : harry_nuts = 2 * sue_nuts)
  (h3 : bill_nuts = 6 * harry_nuts) :
  bill_nuts + harry_nuts = 672 := by
sorry

end nut_problem_l4024_402482


namespace bar_chart_suitable_for_rope_skipping_l4024_402402

/-- Represents different types of statistical charts -/
inductive StatisticalChart
  | BarChart
  | LineChart
  | PieChart

/-- Represents a dataset of rope skipping scores -/
structure RopeSkippingData where
  scores : List Nat

/-- Defines the property of a chart being suitable for representing discrete data points -/
def suitableForDiscreteData (chart : StatisticalChart) : Prop :=
  match chart with
  | StatisticalChart.BarChart => True
  | _ => False

/-- Theorem stating that a bar chart is suitable for representing rope skipping scores -/
theorem bar_chart_suitable_for_rope_skipping (data : RopeSkippingData) :
  suitableForDiscreteData StatisticalChart.BarChart :=
by sorry

end bar_chart_suitable_for_rope_skipping_l4024_402402


namespace final_alcohol_percentage_l4024_402468

/-- Calculates the final alcohol percentage after adding pure alcohol to a solution -/
theorem final_alcohol_percentage
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_alcohol : ℝ)
  (h_initial_volume : initial_volume = 6)
  (h_initial_percentage : initial_percentage = 0.25)
  (h_added_alcohol : added_alcohol = 3) :
  let initial_alcohol := initial_volume * initial_percentage
  let total_alcohol := initial_alcohol + added_alcohol
  let final_volume := initial_volume + added_alcohol
  let final_percentage := total_alcohol / final_volume
  final_percentage = 0.5 := by sorry

end final_alcohol_percentage_l4024_402468


namespace optimal_choice_is_96_l4024_402466

def count_rectangles (perimeter : ℕ) : ℕ :=
  if perimeter % 2 = 0 then
    (perimeter / 2 - 1) / 2
  else
    0

theorem optimal_choice_is_96 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 97 → count_rectangles n ≤ count_rectangles 96 :=
by sorry

end optimal_choice_is_96_l4024_402466


namespace log_difference_cubes_l4024_402453

theorem log_difference_cubes (x y : ℝ) (a : ℝ) (h : Real.log x - Real.log y = a) :
  Real.log ((x / 2) ^ 3) - Real.log ((y / 2) ^ 3) = 3 * a := by
  sorry

end log_difference_cubes_l4024_402453


namespace union_of_A_and_B_l4024_402475

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 3*x - 2*a^2 + 4 = 0}

-- State the theorem
theorem union_of_A_and_B (a : ℝ) :
  (A a ∩ B a = {1}) →
  ((a = 2 ∧ A a ∪ B a = {-4, 1}) ∨ (a = -2 ∧ A a ∪ B a = {-4, -3, 1})) :=
by sorry

end union_of_A_and_B_l4024_402475


namespace polynomial_invariant_is_constant_l4024_402419

/-- A polynomial function from ℝ×ℝ to ℝ×ℝ -/
def PolynomialRR : Type := (ℝ × ℝ) → (ℝ × ℝ)

/-- The property that P(x,y) = P(x+y,x-y) for all x,y ∈ ℝ -/
def HasInvariantProperty (P : PolynomialRR) : Prop :=
  ∀ x y : ℝ, P (x, y) = P (x + y, x - y)

/-- The theorem stating that any polynomial with the invariant property is constant -/
theorem polynomial_invariant_is_constant (P : PolynomialRR) 
  (h : HasInvariantProperty P) : 
  ∃ a b : ℝ, ∀ x y : ℝ, P (x, y) = (a, b) := by
  sorry

end polynomial_invariant_is_constant_l4024_402419


namespace complement_of_union_is_four_l4024_402456

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_is_four :
  (U \ (A ∪ B)) = {4} := by sorry

end complement_of_union_is_four_l4024_402456


namespace nigels_winnings_l4024_402406

/-- The amount of money Nigel won initially -/
def initial_winnings : ℝ := sorry

/-- The amount Nigel gave away -/
def amount_given_away : ℝ := 25

/-- The amount Nigel's mother gave him -/
def amount_from_mother : ℝ := 80

/-- The extra amount Nigel has compared to twice his initial winnings -/
def extra_amount : ℝ := 10

theorem nigels_winnings :
  initial_winnings - amount_given_away + amount_from_mother = 
  2 * initial_winnings + extra_amount ∧ initial_winnings = 45 := by
  sorry

end nigels_winnings_l4024_402406


namespace max_cake_pieces_l4024_402430

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 20

/-- The size of the small piece in inches -/
def small_piece_size : ℕ := 2

/-- The area of the large cake in square inches -/
def large_cake_area : ℕ := large_cake_size * large_cake_size

/-- The area of a small piece in square inches -/
def small_piece_area : ℕ := small_piece_size * small_piece_size

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := large_cake_area / small_piece_area

theorem max_cake_pieces : max_pieces = 100 := by
  sorry

end max_cake_pieces_l4024_402430


namespace sum_of_greatest_b_values_l4024_402443

theorem sum_of_greatest_b_values (b : ℝ) : 
  4 * b^4 - 41 * b^2 + 100 = 0 → 
  ∃ (b1 b2 : ℝ), b1 ≥ b2 ∧ b2 ≥ 0 ∧ 
    (4 * b1^4 - 41 * b1^2 + 100 = 0) ∧ 
    (4 * b2^4 - 41 * b2^2 + 100 = 0) ∧ 
    b1 + b2 = 4.5 ∧
    ∀ (x : ℝ), (4 * x^4 - 41 * x^2 + 100 = 0) → x ≤ b1 :=
sorry

end sum_of_greatest_b_values_l4024_402443


namespace jason_pokemon_cards_l4024_402448

theorem jason_pokemon_cards (initial_cards : ℕ) (bought_cards : ℕ) :
  initial_cards = 1342 →
  bought_cards = 536 →
  initial_cards - bought_cards = 806 :=
by sorry

end jason_pokemon_cards_l4024_402448


namespace percentage_of_men_l4024_402491

/-- The percentage of employees who are men, given picnic attendance data. -/
theorem percentage_of_men (men_attendance : Real) (women_attendance : Real) (total_attendance : Real)
  (h1 : men_attendance = 0.2)
  (h2 : women_attendance = 0.4)
  (h3 : total_attendance = 0.29000000000000004) :
  ∃ (men_percentage : Real),
    men_percentage * men_attendance + (1 - men_percentage) * women_attendance = total_attendance ∧
    men_percentage = 0.55 := by
  sorry

end percentage_of_men_l4024_402491


namespace soft_drink_bottles_l4024_402432

theorem soft_drink_bottles (small_bottles : ℕ) : 
  (small_bottles : ℝ) * 0.89 + 15000 * 0.88 = 18540 → 
  small_bottles = 6000 := by
  sorry

end soft_drink_bottles_l4024_402432


namespace savings_distribution_l4024_402492

/-- Calculates the amount each child receives from the couple's savings --/
theorem savings_distribution (husband_contribution : ℕ) (wife_contribution : ℕ)
  (husband_interval : ℕ) (wife_interval : ℕ) (months : ℕ) (days_per_month : ℕ)
  (savings_percentage : ℚ) (num_children : ℕ) :
  husband_contribution = 450 →
  wife_contribution = 315 →
  husband_interval = 10 →
  wife_interval = 5 →
  months = 8 →
  days_per_month = 30 →
  savings_percentage = 3/4 →
  num_children = 6 →
  (((months * days_per_month / husband_interval) * husband_contribution +
    (months * days_per_month / wife_interval) * wife_contribution) *
    savings_percentage / num_children : ℚ) = 3240 := by
  sorry

end savings_distribution_l4024_402492


namespace missing_number_proof_l4024_402436

theorem missing_number_proof (x : ℝ) : 
  x * 54 = 75625 → 
  ⌊x + 0.5⌋ = 1400 := by
sorry

end missing_number_proof_l4024_402436


namespace no_perfect_squares_l4024_402433

theorem no_perfect_squares (x y z t : ℕ+) : 
  (x * y : ℤ) - (z * t : ℤ) = (x : ℤ) + y ∧ 
  (x : ℤ) + y = (z : ℤ) + t → 
  ¬(∃ (a c : ℕ+), (x * y : ℤ) = (a * a : ℤ) ∧ (z * t : ℤ) = (c * c : ℤ)) :=
by sorry

end no_perfect_squares_l4024_402433


namespace function_property_l4024_402449

def is_solution_set (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x, x ∈ S ↔ (x > 0 ∧ f x + f (x - 8) ≤ 2)

theorem function_property (f : ℝ → ℝ) (h1 : ∀ x y, x > 0 → y > 0 → f (x * y) = f x + f y)
  (h2 : ∀ x y, x > 0 → y > 0 → x < y → f x < f y) (h3 : f 3 = 1) :
  is_solution_set f (Set.Ioo 8 9) :=
sorry

end function_property_l4024_402449


namespace hack_represents_8634_l4024_402416

-- Define the mapping of letters to digits
def letter_to_digit : Char → Nat
| 'Q' => 0
| 'U' => 1
| 'I' => 2
| 'C' => 3
| 'K' => 4
| 'M' => 5
| 'A' => 6
| 'T' => 7
| 'H' => 8
| 'S' => 9
| _ => 0  -- Default case for other characters

-- Define the code word
def code_word : List Char := ['H', 'A', 'C', 'K']

-- Theorem to prove
theorem hack_represents_8634 :
  (code_word.map letter_to_digit).foldl (fun acc d => acc * 10 + d) 0 = 8634 := by
  sorry

end hack_represents_8634_l4024_402416


namespace hair_reaches_floor_simultaneously_l4024_402407

/-- Represents the growth rate of a person or their hair -/
structure GrowthRate where
  rate : ℝ

/-- Represents a person with their growth rate and hair growth rate -/
structure Person where
  growth : GrowthRate
  hairGrowth : GrowthRate

/-- The rate at which the distance from hair to floor decreases -/
def hairToFloorRate (p : Person) : ℝ :=
  p.hairGrowth.rate - p.growth.rate

theorem hair_reaches_floor_simultaneously
  (katya alena : Person)
  (h1 : katya.hairGrowth.rate = 2 * katya.growth.rate)
  (h2 : alena.growth.rate = katya.hairGrowth.rate)
  (h3 : alena.hairGrowth.rate = 1.5 * alena.growth.rate) :
  hairToFloorRate katya = hairToFloorRate alena :=
sorry

end hair_reaches_floor_simultaneously_l4024_402407


namespace base_conversion_256_to_base_5_l4024_402417

def base_five_to_decimal (a b c d : ℕ) : ℕ :=
  a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0

theorem base_conversion_256_to_base_5 :
  base_five_to_decimal 2 0 1 1 = 256 := by
  sorry

end base_conversion_256_to_base_5_l4024_402417


namespace cindy_marbles_l4024_402405

theorem cindy_marbles (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ) :
  initial_marbles = 500 →
  friends = 4 →
  marbles_per_friend = 80 →
  4 * (initial_marbles - friends * marbles_per_friend) = 720 := by
  sorry

end cindy_marbles_l4024_402405


namespace divisibility_in_base_system_l4024_402422

theorem divisibility_in_base_system : ∃! (b : ℕ), b ≥ 8 ∧ (∃ (q : ℕ), 7 * b + 2 = q * (2 * b^2 + 7 * b + 5)) ∧ b = 8 := by
  sorry

end divisibility_in_base_system_l4024_402422


namespace fixed_points_of_moving_circle_l4024_402420

/-- The equation of the moving circle -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 4*m*y + 6*m - 2 = 0

/-- A point is a fixed point if it satisfies the circle equation for all m -/
def is_fixed_point (x y : ℝ) : Prop :=
  ∀ m : ℝ, circle_equation x y m

theorem fixed_points_of_moving_circle :
  (is_fixed_point 1 1 ∧ is_fixed_point (1/5) (7/5)) ∧
  ∀ x y : ℝ, is_fixed_point x y → (x = 1 ∧ y = 1) ∨ (x = 1/5 ∧ y = 7/5) :=
sorry

end fixed_points_of_moving_circle_l4024_402420


namespace product_expansion_l4024_402495

theorem product_expansion (x : ℝ) : 
  (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end product_expansion_l4024_402495


namespace inequality_proof_l4024_402431

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  1 / Real.sqrt (x + y) + 1 / Real.sqrt (y + z) + 1 / Real.sqrt (z + x) 
  ≤ 1 / Real.sqrt (2 * x * y * z) := by
sorry

end inequality_proof_l4024_402431


namespace vacation_savings_theorem_l4024_402471

/-- Calculates the number of months needed to reach a savings goal -/
def months_to_goal (goal : ℕ) (current : ℕ) (monthly : ℕ) : ℕ :=
  ((goal - current) + monthly - 1) / monthly

theorem vacation_savings_theorem (goal current monthly : ℕ) 
  (h1 : goal = 5000)
  (h2 : current = 2900)
  (h3 : monthly = 700) :
  months_to_goal goal current monthly = 3 := by
  sorry

end vacation_savings_theorem_l4024_402471


namespace not_right_triangle_l4024_402483

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the ratio condition
def ratio_condition (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.A = 5 * k ∧ t.B = 12 * k ∧ t.C = 13 * k

-- Theorem statement
theorem not_right_triangle (t : Triangle) (h : ratio_condition t) : 
  t.A ≠ 90 ∧ t.B ≠ 90 ∧ t.C ≠ 90 := by
  sorry

end not_right_triangle_l4024_402483


namespace tommy_balloons_l4024_402403

/-- The number of balloons Tommy has after receiving more from his mom -/
def total_balloons (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that Tommy's total balloons is the sum of his initial balloons and additional balloons -/
theorem tommy_balloons (initial : ℕ) (additional : ℕ) :
  total_balloons initial additional = initial + additional := by
  sorry

end tommy_balloons_l4024_402403


namespace f_property_f_1001_eq_1_f_1002_eq_1_l4024_402467

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def has_prime_divisor (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ p ∣ n

theorem f_property (f : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 1 →
    ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

theorem f_1001_eq_1 (f : ℕ → ℤ) : Prop := f 1001 = 1

theorem f_1002_eq_1 (f : ℕ → ℤ) : f_property f → f_1001_eq_1 f → f 1002 = 1 := by
  sorry

end f_property_f_1001_eq_1_f_1002_eq_1_l4024_402467


namespace extra_digit_sum_l4024_402442

theorem extra_digit_sum (x y : ℕ) (a : Fin 10) :
  x + y = 23456 →
  (10 * x + a.val) + y = 55555 →
  a.val = 5 :=
by sorry

end extra_digit_sum_l4024_402442


namespace parabola_circle_theorem_l4024_402447

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  a : ℝ
  eq : (y : ℝ) → (x : ℝ) → Prop := fun y x => y^2 = 4 * a * x

/-- Represents a circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (p.a, 0)

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop := fun x => x = -p.a

/-- The chord length of the intersection between a parabola and its directrix -/
def chordLength (p : Parabola) : ℝ := sorry

/-- The standard equation of a circle -/
def standardEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem parabola_circle_theorem (p : Parabola) (c : Circle) :
  p.a = 1 →
  c.center = focus p →
  chordLength p = 6 →
  ∀ x y, standardEquation c x y ↔ (x - 1)^2 + y^2 = 13 := by sorry

end parabola_circle_theorem_l4024_402447


namespace final_sum_after_transformation_l4024_402425

theorem final_sum_after_transformation (x y S : ℝ) (h : x + y = S) :
  3 * (x + 4) + 3 * (y + 4) = 3 * S + 24 := by
  sorry

end final_sum_after_transformation_l4024_402425


namespace sphere_volume_inscribed_cylinder_l4024_402441

-- Define the radius of the base of the cylinder
def r : ℝ := 15

-- Define the radius of the sphere
def sphere_radius : ℝ := r + 2

-- Define the height of the cylinder
def cylinder_height : ℝ := r + 1

-- State the theorem
theorem sphere_volume_inscribed_cylinder :
  let volume := (4 / 3) * Real.pi * sphere_radius ^ 3
  (2 * sphere_radius) ^ 2 = (2 * r) ^ 2 + cylinder_height ^ 2 →
  volume = 6550 * (2 / 3) * Real.pi := by
  sorry


end sphere_volume_inscribed_cylinder_l4024_402441


namespace seashells_needed_l4024_402463

def current_seashells : ℕ := 19
def goal_seashells : ℕ := 25

theorem seashells_needed : goal_seashells - current_seashells = 6 := by
  sorry

end seashells_needed_l4024_402463


namespace specific_test_result_l4024_402457

/-- Represents a test with a given number of questions, points for correct and incorrect answers --/
structure Test where
  total_questions : ℕ
  points_correct : ℤ
  points_incorrect : ℤ

/-- Represents the result of a test --/
structure TestResult where
  test : Test
  correct_answers : ℕ
  final_score : ℤ

/-- Theorem stating that for a specific test configuration, 
    if the final score is 0, then the number of correct answers is 10 --/
theorem specific_test_result (t : Test) (r : TestResult) : 
  t.total_questions = 26 ∧ 
  t.points_correct = 8 ∧ 
  t.points_incorrect = -5 ∧ 
  r.test = t ∧
  r.correct_answers + (t.total_questions - r.correct_answers) = t.total_questions ∧
  r.final_score = r.correct_answers * t.points_correct + (t.total_questions - r.correct_answers) * t.points_incorrect ∧
  r.final_score = 0 →
  r.correct_answers = 10 := by
sorry

end specific_test_result_l4024_402457


namespace arithmetic_sequence_property_l4024_402444

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum : a 1 - a 9 + a 17 = 7) :
  a 3 + a 15 = 14 := by
  sorry

end arithmetic_sequence_property_l4024_402444


namespace games_for_512_players_l4024_402445

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  num_players : ℕ
  num_players_pos : 0 < num_players

/-- The number of games needed to determine the champion in a single-elimination tournament -/
def games_to_champion (t : SingleEliminationTournament) : ℕ :=
  t.num_players - 1

/-- Theorem: In a single-elimination tournament with 512 players, 511 games are needed to determine the champion -/
theorem games_for_512_players :
  let t : SingleEliminationTournament := ⟨512, by norm_num⟩
  games_to_champion t = 511 := by
  sorry

end games_for_512_players_l4024_402445


namespace cades_remaining_marbles_l4024_402480

/-- Represents the number of marbles Cade has left after giving some away -/
def marblesLeft (initial : Nat) (givenAway : Nat) : Nat :=
  initial - givenAway

/-- Theorem stating that Cade has 79 marbles left -/
theorem cades_remaining_marbles :
  marblesLeft 87 8 = 79 := by
  sorry

end cades_remaining_marbles_l4024_402480


namespace problem_l4024_402429

def l₁ (x y : ℝ) : Prop := x - 2*y + 3 = 0

def l₂ (x y : ℝ) : Prop := 2*x + y + 3 = 0

def perpendicular (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c d : ℝ, (∀ x y, f x y ↔ a*x + b*y = c) ∧
                 (∀ x y, g x y ↔ d*x - a*y = 0)

def p : Prop := ¬(perpendicular l₁ l₂)

def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ + 2 > Real.exp x₀

theorem problem : (¬p) ∧ q := by sorry

end problem_l4024_402429


namespace parametric_curve_length_l4024_402451

/-- The parametric curve described by (x,y) = (3 sin t, 3 cos t) for t ∈ [0, 2π] -/
def parametric_curve : Set (ℝ × ℝ) :=
  {p | ∃ t ∈ Set.Icc 0 (2 * Real.pi), p = (3 * Real.sin t, 3 * Real.cos t)}

/-- The length of a curve -/
noncomputable def curve_length (c : Set (ℝ × ℝ)) : ℝ := sorry

theorem parametric_curve_length :
  curve_length parametric_curve = 6 * Real.pi := by sorry

end parametric_curve_length_l4024_402451


namespace y_plus_two_over_y_l4024_402469

theorem y_plus_two_over_y (y : ℝ) (h : 5 = y^2 + 4/y^2) : 
  y + 2/y = 3 ∨ y + 2/y = -3 := by
sorry

end y_plus_two_over_y_l4024_402469


namespace first_studio_students_l4024_402490

theorem first_studio_students (total : ℕ) (second : ℕ) (third : ℕ) 
  (h1 : total = 376)
  (h2 : second = 135)
  (h3 : third = 131) :
  total - (second + third) = 110 := by
  sorry

end first_studio_students_l4024_402490


namespace delivery_distances_l4024_402497

/-- Represents the direction of travel --/
inductive Direction
  | North
  | South

/-- Represents a location relative to the supermarket --/
structure Location where
  distance : ℝ
  direction : Direction

/-- Calculates the distance between two locations --/
def distanceBetween (a b : Location) : ℝ :=
  match a.direction, b.direction with
  | Direction.North, Direction.North => abs (a.distance - b.distance)
  | Direction.South, Direction.South => abs (a.distance - b.distance)
  | _, _ => a.distance + b.distance

/-- Calculates the round trip distance to a location --/
def roundTripDistance (loc : Location) : ℝ :=
  2 * loc.distance

theorem delivery_distances (unitA unitB unitC : Location) 
  (hA : unitA = { distance := 30, direction := Direction.South })
  (hB : unitB = { distance := 50, direction := Direction.South })
  (hC : unitC = { distance := 15, direction := Direction.North }) :
  distanceBetween unitA unitC = 45 ∧ 
  roundTripDistance unitB + 3 * roundTripDistance unitC = 190 := by
  sorry


end delivery_distances_l4024_402497


namespace green_hats_count_l4024_402427

theorem green_hats_count (total_hats : ℕ) (blue_cost green_cost total_cost : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_cost = 6)
  (h3 : green_cost = 7)
  (h4 : total_cost = 550) :
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_cost * blue_hats + green_cost * green_hats = total_cost ∧
    green_hats = 40 :=
by sorry

end green_hats_count_l4024_402427


namespace expected_potato_yield_l4024_402470

/-- Calculates the expected potato yield from a rectangular garden --/
theorem expected_potato_yield
  (garden_length_steps : ℕ)
  (garden_width_steps : ℕ)
  (step_length_feet : ℝ)
  (potato_yield_per_sqft : ℝ)
  (h1 : garden_length_steps = 18)
  (h2 : garden_width_steps = 25)
  (h3 : step_length_feet = 3)
  (h4 : potato_yield_per_sqft = 1/3) :
  (garden_length_steps : ℝ) * step_length_feet *
  (garden_width_steps : ℝ) * step_length_feet *
  potato_yield_per_sqft = 1350 := by
  sorry

end expected_potato_yield_l4024_402470


namespace subtracted_value_l4024_402452

theorem subtracted_value (N V : ℝ) (h1 : N = 740) (h2 : N / 4 - V = 10) : V = 175 := by
  sorry

end subtracted_value_l4024_402452


namespace base_10_sum_45_l4024_402410

/-- The sum of single-digit numbers in base b -/
def sum_single_digits (b : ℕ) : ℕ := (b - 1) * b / 2

/-- Checks if a number in base b has 5 as its units digit -/
def has_units_digit_5 (n : ℕ) (b : ℕ) : Prop := n % b = 5

theorem base_10_sum_45 :
  ∃ (b : ℕ), b > 1 ∧ sum_single_digits b = 45 ∧ has_units_digit_5 (sum_single_digits b) b ∧ b = 10 := by
  sorry

end base_10_sum_45_l4024_402410


namespace fraction_scaling_l4024_402499

theorem fraction_scaling (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (6 * x + 6 * y) / ((6 * x) * (6 * y)) = (1 / 6) * ((x + y) / (x * y)) := by
  sorry

end fraction_scaling_l4024_402499


namespace literary_society_book_exchange_l4024_402474

/-- The number of books exchanged in a Literary Society book sharing ceremony -/
def books_exchanged (x : ℕ) : ℕ := x * (x - 1)

/-- The theorem stating that for a group of x students where each student gives one book to every
    other student, and a total of 210 books are exchanged, the equation x(x-1) = 210 holds -/
theorem literary_society_book_exchange (x : ℕ) (h : books_exchanged x = 210) :
  x * (x - 1) = 210 := by sorry

end literary_society_book_exchange_l4024_402474


namespace mark_cookies_sold_l4024_402415

theorem mark_cookies_sold (n : ℕ) (mark_sold ann_sold : ℕ) : 
  n = 12 →
  mark_sold < n →
  ann_sold = n - 2 →
  mark_sold ≥ 1 →
  ann_sold ≥ 1 →
  mark_sold + ann_sold < n →
  mark_sold = n - 11 :=
by sorry

end mark_cookies_sold_l4024_402415


namespace complement_of_A_in_U_l4024_402439

-- Define the universal set U
def U : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Define set A
def A : Set ℝ := {x | 1 / x ≥ 1}

-- State the theorem
theorem complement_of_A_in_U : 
  (U \ A) = {x | -1 < x ∧ x ≤ 0} :=
sorry

end complement_of_A_in_U_l4024_402439


namespace basketball_campers_count_l4024_402438

theorem basketball_campers_count (total_campers soccer_campers football_campers : ℕ) 
  (h1 : total_campers = 88)
  (h2 : soccer_campers = 32)
  (h3 : football_campers = 32) :
  total_campers - soccer_campers - football_campers = 24 := by
  sorry

end basketball_campers_count_l4024_402438


namespace condition_necessary_not_sufficient_l4024_402459

theorem condition_necessary_not_sufficient :
  (∃ a b : ℝ, a + b ≠ 3 ∧ (a = 1 ∧ b = 2)) ∧
  (∀ a b : ℝ, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  (∃ a b : ℝ, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) :=
by sorry

end condition_necessary_not_sufficient_l4024_402459


namespace two_digit_sum_divisible_by_17_l4024_402440

/-- A function that reverses a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

/-- A predicate that checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_sum_divisible_by_17 :
  ∀ A : ℕ, is_two_digit A →
    (A + reverse_digits A) % 17 = 0 ↔ A = 89 ∨ A = 98 := by
  sorry

end two_digit_sum_divisible_by_17_l4024_402440


namespace greatest_difference_l4024_402404

/-- A type representing a chessboard arrangement of numbers 1 to 400 -/
def Arrangement := Fin 20 → Fin 20 → Fin 400

/-- The property that an arrangement has two numbers in the same row or column differing by at least N -/
def HasDifference (arr : Arrangement) (N : ℕ) : Prop :=
  ∃ (i j k : Fin 20), (arr i j).val + N ≤ (arr i k).val ∨ (arr j i).val + N ≤ (arr k i).val

/-- The theorem stating that 209 is the greatest natural number satisfying the given condition -/
theorem greatest_difference : 
  (∀ (arr : Arrangement), HasDifference arr 209) ∧ 
  ¬(∀ (arr : Arrangement), HasDifference arr 210) :=
sorry

end greatest_difference_l4024_402404


namespace garrett_granola_bars_l4024_402494

/-- Proves that Garrett bought 6 oatmeal raisin granola bars -/
theorem garrett_granola_bars :
  ∀ (total peanut oatmeal_raisin : ℕ),
    total = 14 →
    peanut = 8 →
    total = peanut + oatmeal_raisin →
    oatmeal_raisin = 6 := by
  sorry

end garrett_granola_bars_l4024_402494


namespace vacation_cost_difference_l4024_402498

theorem vacation_cost_difference (total_cost : ℕ) (initial_people : ℕ) (new_people : ℕ) 
  (h1 : total_cost = 360) 
  (h2 : initial_people = 3) 
  (h3 : new_people = 4) : 
  (total_cost / initial_people) - (total_cost / new_people) = 30 := by
sorry

end vacation_cost_difference_l4024_402498


namespace sum_of_squares_difference_l4024_402464

theorem sum_of_squares_difference (a b : ℕ+) (h : a.val^2 - b.val^4 = 2009) : 
  a.val + b.val = 47 := by
  sorry

end sum_of_squares_difference_l4024_402464
