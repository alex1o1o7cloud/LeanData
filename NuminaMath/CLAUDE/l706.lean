import Mathlib

namespace problem_solution_l706_70654

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then x^2 - 4 else |x - 3| + a

theorem problem_solution (a : ℝ) :
  f a (f a (Real.sqrt 6)) = 3 → a = 2 := by
  sorry

end problem_solution_l706_70654


namespace root_in_interval_l706_70651

noncomputable section

variables (a b : ℝ) (h : b ≥ 2*a) (h' : a > 0)

def f (x : ℝ) := 2*(a^x) - b^x

theorem root_in_interval :
  ∃ x, x ∈ Set.Ioo 0 1 ∧ f a b x = 0 :=
sorry

end

end root_in_interval_l706_70651


namespace complex_division_proof_l706_70663

theorem complex_division_proof : (1 + 3 * Complex.I) / (1 - Complex.I) = -1 + 2 * Complex.I := by
  sorry

end complex_division_proof_l706_70663


namespace find_b_l706_70674

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A (b : ℝ) : Set ℝ := {x | 1 ≤ x ∧ x < b}

-- Define the complement of A in U
def complement_A (b : ℝ) : Set ℝ := {x | x < 1 ∨ x ≥ 2}

-- Theorem statement
theorem find_b : ∃ b : ℝ, A b = Set.compl (complement_A b) := by sorry

end find_b_l706_70674


namespace smallest_integer_with_remainders_l706_70673

theorem smallest_integer_with_remainders : ∃ N : ℕ, 
  N > 0 ∧
  N % 5 = 2 ∧
  N % 6 = 3 ∧
  N % 7 = 4 ∧
  N % 11 = 9 ∧
  (∀ M : ℕ, M > 0 ∧ M % 5 = 2 ∧ M % 6 = 3 ∧ M % 7 = 4 ∧ M % 11 = 9 → N ≤ M) ∧
  N = 207 := by
sorry

end smallest_integer_with_remainders_l706_70673


namespace simultaneous_arrival_l706_70681

/-- Represents a point on the shore of the circular lake -/
structure Pier where
  point : ℝ × ℝ

/-- Represents a boat with a starting position and speed -/
structure Boat where
  start : Pier
  speed : ℝ

/-- Represents the circular lake with four piers -/
structure Lake where
  k : Pier
  l : Pier
  p : Pier
  q : Pier

/-- Represents the collision point of two boats -/
def collision_point (b1 b2 : Boat) (dest1 dest2 : Pier) : ℝ × ℝ := sorry

/-- Time taken for a boat to reach its destination -/
def time_to_destination (b : Boat) (dest : Pier) : ℝ := sorry

/-- Main theorem: If boats collide when going to opposite piers,
    they will reach swapped destinations simultaneously -/
theorem simultaneous_arrival (lake : Lake) (boat : Boat) (rowboat : Boat) :
  let x := collision_point boat rowboat lake.p lake.q
  boat.start = lake.k →
  rowboat.start = lake.l →
  time_to_destination boat lake.q = time_to_destination rowboat lake.p := by
  sorry

end simultaneous_arrival_l706_70681


namespace defective_product_probability_l706_70650

theorem defective_product_probability 
  (total_products : ℕ) 
  (genuine_products : ℕ) 
  (defective_products : ℕ) 
  (h1 : total_products = genuine_products + defective_products)
  (h2 : genuine_products = 16)
  (h3 : defective_products = 4) :
  let prob_second_defective : ℚ := defective_products - 1 / (total_products - 1)
  prob_second_defective = 3 / 19 := by
sorry

end defective_product_probability_l706_70650


namespace divisors_of_n_squared_less_than_n_not_dividing_n_l706_70644

def n : ℕ := 2^33 * 5^21

-- Function to count divisors of a number
def count_divisors (m : ℕ) : ℕ := sorry

-- Function to count divisors of m less than n
def count_divisors_less_than (m n : ℕ) : ℕ := sorry

theorem divisors_of_n_squared_less_than_n_not_dividing_n :
  count_divisors_less_than (n^2) n - count_divisors n = 692 := by sorry

end divisors_of_n_squared_less_than_n_not_dividing_n_l706_70644


namespace min_distance_parallel_lines_l706_70627

/-- The minimum distance between two parallel lines -/
theorem min_distance_parallel_lines : 
  let line1 := {(x, y) : ℝ × ℝ | 3 * x + 4 * y - 10 = 0}
  let line2 := {(x, y) : ℝ × ℝ | 6 * x + 8 * y + 5 = 0}
  ∃ d : ℝ, d = (5 : ℝ) / 2 ∧ 
    ∀ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ line1 → Q ∈ line2 → 
      d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by
  sorry

end min_distance_parallel_lines_l706_70627


namespace linear_function_passes_through_points_l706_70678

/-- A linear function y = kx - k passing through (-1, 4) also passes through (1, 0) -/
theorem linear_function_passes_through_points :
  ∃ k : ℝ, (k * (-1) - k = 4) ∧ (k * 1 - k = 0) :=
by sorry

end linear_function_passes_through_points_l706_70678


namespace geometric_sum_first_six_terms_l706_70625

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_six_terms :
  let a : ℚ := 1/2
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 4095/6144 := by
sorry

end geometric_sum_first_six_terms_l706_70625


namespace shaded_area_is_nine_l706_70687

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the spinner shape -/
structure Spinner where
  center : Point
  armLength : ℕ

/-- Represents the entire shaded shape -/
structure ShadedShape where
  spinner : Spinner
  cornerSquares : List Point

/-- Calculates the area of the shaded shape -/
def shadedArea (shape : ShadedShape) : ℕ :=
  let spinnerArea := 2 * shape.spinner.armLength * 2 + 1
  let cornerSquaresArea := shape.cornerSquares.length
  spinnerArea + cornerSquaresArea

/-- The theorem to be proved -/
theorem shaded_area_is_nine :
  ∀ (shape : ShadedShape),
    shape.spinner.center = ⟨3, 3⟩ →
    shape.spinner.armLength = 1 →
    shape.cornerSquares.length = 4 →
    shadedArea shape = 9 := by
  sorry

end shaded_area_is_nine_l706_70687


namespace like_terms_imply_exponent_relation_l706_70619

/-- Given that -25a^(2m)b and 7a^4b^(3-n) are like terms, prove that 2m - n = 2 -/
theorem like_terms_imply_exponent_relation (a b : ℝ) (m n : ℕ) 
  (h : ∃ (k : ℝ), -25 * a^(2*m) * b = k * (7 * a^4 * b^(3-n))) : 
  2 * m - n = 2 :=
sorry

end like_terms_imply_exponent_relation_l706_70619


namespace amusement_park_total_cost_l706_70615

/-- Represents the total cost for a group of children at an amusement park -/
def amusement_park_cost (num_children : ℕ) 
  (ferris_wheel_cost ferris_wheel_participants : ℕ)
  (roller_coaster_cost roller_coaster_participants : ℕ)
  (merry_go_round_cost : ℕ)
  (bumper_cars_cost bumper_cars_participants : ℕ)
  (haunted_house_cost haunted_house_participants : ℕ)
  (log_flume_cost log_flume_participants : ℕ)
  (ice_cream_cost ice_cream_participants : ℕ)
  (hot_dog_cost hot_dog_participants : ℕ)
  (pizza_cost pizza_participants : ℕ)
  (pretzel_cost pretzel_participants : ℕ)
  (cotton_candy_cost cotton_candy_participants : ℕ)
  (soda_cost soda_participants : ℕ) : ℕ :=
  ferris_wheel_cost * ferris_wheel_participants +
  roller_coaster_cost * roller_coaster_participants +
  merry_go_round_cost * num_children +
  bumper_cars_cost * bumper_cars_participants +
  haunted_house_cost * haunted_house_participants +
  log_flume_cost * log_flume_participants +
  ice_cream_cost * ice_cream_participants +
  hot_dog_cost * hot_dog_participants +
  pizza_cost * pizza_participants +
  pretzel_cost * pretzel_participants +
  cotton_candy_cost * cotton_candy_participants +
  soda_cost * soda_participants

/-- The total cost for the group of children at the amusement park is $286 -/
theorem amusement_park_total_cost : 
  amusement_park_cost 10 5 6 7 4 3 4 7 6 5 8 3 8 4 6 5 4 3 5 2 3 6 2 7 = 286 := by
  sorry

end amusement_park_total_cost_l706_70615


namespace min_a_for_inequality_solution_set_inequality_l706_70629

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2) * abs (x - 2)

-- Theorem for part (1)
theorem min_a_for_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc (-3) 1, f x ≤ a) ↔ a ≥ 4 := by sorry

-- Theorem for part (2)
theorem solution_set_inequality :
  {x : ℝ | f x > 3 * x} = {x : ℝ | x > 4} ∪ {x : ℝ | -4 < x ∧ x < 1} := by sorry

end min_a_for_inequality_solution_set_inequality_l706_70629


namespace estimate_pi_l706_70631

theorem estimate_pi (n : ℕ) (m : ℕ) (h1 : n = 120) (h2 : m = 34) :
  let π_estimate : ℚ := 4 * (m : ℚ) / (n : ℚ) + 2
  π_estimate = 47 / 15 := by
  sorry

end estimate_pi_l706_70631


namespace angle_bisectors_may_not_form_triangle_l706_70617

/-- Given a triangle with sides a = 2, b = 3, and c < 5, 
    prove that its angle bisectors may not satisfy the triangle inequality -/
theorem angle_bisectors_may_not_form_triangle :
  ∃ (c : ℝ), c < 5 ∧ 
  ∃ (ℓa ℓb ℓc : ℝ),
    (ℓa + ℓb ≤ ℓc ∨ ℓa + ℓc ≤ ℓb ∨ ℓb + ℓc ≤ ℓa) ∧
    ℓa = 3 / (1 + 2 / 7 * 3) ∧
    ℓb = 2 / (2 + 3 / 8 * 2) ∧
    ℓc = 0 := by
  sorry


end angle_bisectors_may_not_form_triangle_l706_70617


namespace frustum_smaller_base_radius_l706_70655

/-- A frustum with the given properties has a smaller base radius of 7 -/
theorem frustum_smaller_base_radius (r : ℝ) : 
  r > 0 → -- r is positive (implicit in the problem context)
  (2 * π * r) * 3 = 2 * π * (3 * r) → -- one circumference is three times the other
  3 = 3 → -- slant height is 3
  π * (r + 3 * r) * 3 = 84 * π → -- lateral surface area formula
  r = 7 := by
sorry


end frustum_smaller_base_radius_l706_70655


namespace random_sampling_correct_l706_70603

/-- Represents a random number table row -/
def RandomTableRow := List Nat

/-- Checks if a number is a valid bag number (000-799) -/
def isValidBagNumber (n : Nat) : Bool :=
  n >= 0 && n <= 799

/-- Extracts valid bag numbers from a list of numbers -/
def extractValidBagNumbers (numbers : List Nat) : List Nat :=
  numbers.filter isValidBagNumber

/-- Represents the given random number table row -/
def givenRow : RandomTableRow :=
  [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79]

/-- The expected result -/
def expectedResult : List Nat := [785, 567, 199, 507, 175]

theorem random_sampling_correct :
  let startIndex := 6  -- 7th column (0-based index)
  let relevantNumbers := givenRow.drop startIndex
  let validBagNumbers := extractValidBagNumbers relevantNumbers
  validBagNumbers.take 5 = expectedResult := by sorry

end random_sampling_correct_l706_70603


namespace roof_length_width_difference_l706_70679

-- Define the trapezoidal roof
structure TrapezoidalRoof where
  width : ℝ
  length : ℝ
  height : ℝ
  area : ℝ

-- Define the conditions of the problem
def roof_conditions (roof : TrapezoidalRoof) : Prop :=
  roof.length = 3 * roof.width ∧
  roof.height = 25 ∧
  roof.area = 675 ∧
  roof.area = (1 / 2) * (roof.width + roof.length) * roof.height

-- Theorem to prove
theorem roof_length_width_difference (roof : TrapezoidalRoof) 
  (h : roof_conditions roof) : roof.length - roof.width = 27 := by
  sorry


end roof_length_width_difference_l706_70679


namespace sum_with_radical_conjugate_l706_70684

theorem sum_with_radical_conjugate : 
  let x : ℝ := 15 - Real.sqrt 500
  let y : ℝ := 15 + Real.sqrt 500
  x + y = 30 := by sorry

end sum_with_radical_conjugate_l706_70684


namespace correct_reading_of_6005_l706_70634

/-- Represents the correct way to read a number between 1000 and 9999 -/
def ReadNumber (n : ℕ) : String :=
  sorry

/-- The correct reading of 6005 -/
def Correct6005Reading : String :=
  ReadNumber 6005

/-- The incorrect reading of 6005 -/
def Incorrect6005Reading : String :=
  "six thousand zero zero five"

theorem correct_reading_of_6005 : 
  Correct6005Reading ≠ Incorrect6005Reading :=
sorry

end correct_reading_of_6005_l706_70634


namespace parabola_line_intersection_slope_range_l706_70640

theorem parabola_line_intersection_slope_range :
  ∀ k : ℝ,
  (∃ A B : ℝ × ℝ, A ≠ B ∧
    (A.2)^2 = 4 * A.1 ∧
    (B.2)^2 = 4 * B.1 ∧
    A.2 = k * (A.1 + 2) ∧
    B.2 = k * (B.1 + 2)) ↔
  (k ∈ Set.Ioo (- Real.sqrt 2 / 2) 0 ∪ Set.Ioo 0 (Real.sqrt 2 / 2)) :=
by sorry

end parabola_line_intersection_slope_range_l706_70640


namespace volume_ratio_is_two_l706_70699

/-- Represents the state of an ideal gas -/
structure GasState where
  volume : ℝ
  pressure : ℝ
  temperature : ℝ

/-- Represents a closed cycle of an ideal gas -/
structure GasCycle where
  initial : GasState
  state2 : GasState
  state3 : GasState

/-- The conditions of the gas cycle -/
class CycleConditions (cycle : GasCycle) where
  isobaric_1_2 : cycle.state2.pressure = cycle.initial.pressure
  volume_increase_1_2 : cycle.state2.volume = 4 * cycle.initial.volume
  isothermal_2_3 : cycle.state3.temperature = cycle.state2.temperature
  pressure_increase_2_3 : cycle.state3.pressure > cycle.state2.pressure
  compression_3_1 : ∃ (γ : ℝ), cycle.initial.temperature = γ * cycle.initial.volume^2

/-- The theorem to be proved -/
theorem volume_ratio_is_two 
  (cycle : GasCycle) 
  [conditions : CycleConditions cycle] : 
  cycle.state3.volume = 2 * cycle.initial.volume := by
  sorry

end volume_ratio_is_two_l706_70699


namespace remainder_101_power_50_mod_100_l706_70690

theorem remainder_101_power_50_mod_100 : 101^50 % 100 = 1 := by
  sorry

end remainder_101_power_50_mod_100_l706_70690


namespace area_of_region_l706_70667

-- Define the region
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 - 1 ≤ p.2 ∧ p.2 ≤ Real.sqrt (1 - p.1^2)}

-- State the theorem
theorem area_of_region :
  MeasureTheory.volume R = π / 2 + 1 := by
  sorry

end area_of_region_l706_70667


namespace minimum_economic_loss_l706_70642

def repair_times : List ℕ := [12, 17, 8, 18, 23, 30, 14]
def num_workers : ℕ := 3
def loss_per_minute : ℕ := 2

def distribute_work (times : List ℕ) (workers : ℕ) : List ℕ :=
  sorry

def calculate_waiting_time (distribution : List ℕ) : ℕ :=
  sorry

def economic_loss (waiting_time : ℕ) (loss_per_minute : ℕ) : ℕ :=
  sorry

theorem minimum_economic_loss :
  economic_loss (calculate_waiting_time (distribute_work repair_times num_workers)) loss_per_minute = 364 := by
  sorry

end minimum_economic_loss_l706_70642


namespace find_b_value_l706_70660

/-- The cube of a and the fourth root of b vary inversely -/
def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a^3 * b^(1/4) = k

theorem find_b_value (a b : ℝ) :
  inverse_relation a b →
  (3: ℝ)^3 * (256 : ℝ)^(1/4) = a^3 * b^(1/4) →
  a * b = 81 →
  b = 16 := by
sorry

end find_b_value_l706_70660


namespace quadratic_roots_range_quadratic_roots_value_l706_70633

/-- The quadratic equation x^2 + 3x + k - 2 = 0 -/
def quadratic (x k : ℝ) : Prop := x^2 + 3*x + k - 2 = 0

/-- The equation has real roots -/
def has_real_roots (k : ℝ) : Prop := ∃ x : ℝ, quadratic x k

/-- The roots of the equation satisfy (x_1 + 1)(x_2 + 1) = -1 -/
def roots_condition (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic x₁ k ∧ quadratic x₂ k ∧ (x₁ + 1) * (x₂ + 1) = -1

theorem quadratic_roots_range (k : ℝ) :
  has_real_roots k → k ≤ 17/4 :=
sorry

theorem quadratic_roots_value (k : ℝ) :
  has_real_roots k → roots_condition k → k = 3 :=
sorry

end quadratic_roots_range_quadratic_roots_value_l706_70633


namespace product_remainder_l706_70630

theorem product_remainder (a b c d e : ℕ) (h1 : a = 12457) (h2 : b = 12463) (h3 : c = 12469) (h4 : d = 12473) (h5 : e = 12479) :
  (a * b * c * d * e) % 18 = 3 := by
sorry

end product_remainder_l706_70630


namespace range_of_slopes_tangent_line_equation_l706_70646

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

-- Theorem for the range of slopes
theorem range_of_slopes :
  ∀ x ∈ Set.Icc (-2) 1, -3 ≤ f' x ∧ f' x ≤ 9 :=
sorry

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ a b : ℝ,
    f' a = -3 ∧
    f a = b ∧
    (∀ x y : ℝ, line_l x y → (3*x + y + 6 = 0 → x = a ∧ y = b)) :=
sorry

end range_of_slopes_tangent_line_equation_l706_70646


namespace exists_sum_of_scores_with_two_ways_l706_70632

/-- Represents a scoring configuration for the modified AMC test. -/
structure ScoringConfig where
  total_questions : ℕ
  correct_points : ℕ
  unanswered_points : ℕ
  incorrect_points : ℕ

/-- Represents an answer combination for the test. -/
structure AnswerCombination where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- Calculates the score for a given answer combination under a specific scoring config. -/
def calculate_score (config : ScoringConfig) (answers : AnswerCombination) : ℕ :=
  answers.correct * config.correct_points + answers.unanswered * config.unanswered_points + answers.incorrect * config.incorrect_points

/-- Checks if an answer combination is valid for a given total number of questions. -/
def is_valid_combination (total_questions : ℕ) (answers : AnswerCombination) : Prop :=
  answers.correct + answers.unanswered + answers.incorrect = total_questions

/-- Defines the specific scoring configuration for the problem. -/
def amc_scoring : ScoringConfig :=
  { total_questions := 20
  , correct_points := 7
  , unanswered_points := 3
  , incorrect_points := 0 }

/-- Theorem stating the existence of a sum of scores meeting the problem criteria. -/
theorem exists_sum_of_scores_with_two_ways :
  ∃ (sum : ℕ), 
    (∃ (scores : List ℕ),
      (∀ score ∈ scores, 
        score ≤ 140 ∧ 
        (∃ (ways : List AnswerCombination), 
          ways.length = 2 ∧
          (∀ way ∈ ways, 
            is_valid_combination amc_scoring.total_questions way ∧
            calculate_score amc_scoring way = score))) ∧
      sum = scores.sum) := by
  sorry


end exists_sum_of_scores_with_two_ways_l706_70632


namespace simplify_expression_l706_70657

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 6) - (2*x + 6)*(3*x - 2) = -3*x^2 - 12 := by
  sorry

end simplify_expression_l706_70657


namespace line_intercepts_sum_l706_70618

/-- Given a line with equation y - 7 = -3(x - 5), 
    the sum of its x-intercept and y-intercept is 88/3 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y - 7 = -3 * (x - 5)) → 
  (∃ x_int y_int : ℝ, 
    (y_int - 7 = -3 * (x_int - 5)) ∧ 
    (0 - 7 = -3 * (x_int - 5)) ∧ 
    (y_int - 7 = -3 * (0 - 5)) ∧ 
    (x_int + y_int = 88 / 3)) := by
  sorry

end line_intercepts_sum_l706_70618


namespace tina_total_time_l706_70628

/-- Calculates the total time for Tina to clean keys, let them dry, take breaks, and complete her assignment -/
def total_time (total_keys : ℕ) (keys_to_clean : ℕ) (clean_time_per_key : ℕ) (dry_time_per_key : ℕ) (break_interval : ℕ) (break_duration : ℕ) (assignment_time : ℕ) : ℕ :=
  let cleaning_time := keys_to_clean * clean_time_per_key
  let drying_time := total_keys * dry_time_per_key
  let break_count := total_keys / break_interval
  let break_time := break_count * break_duration
  cleaning_time + drying_time + break_time + assignment_time

/-- Proves that given the conditions in the problem, the total time is 541 minutes -/
theorem tina_total_time :
  total_time 30 29 7 10 5 3 20 = 541 := by
  sorry

end tina_total_time_l706_70628


namespace point_2023_coordinates_l706_70637

/-- The y-coordinate of the nth point in the sequence -/
def y_coord (n : ℕ) : ℤ :=
  match n % 4 with
  | 0 => 0
  | 1 => 1
  | 2 => 0
  | 3 => -1
  | _ => 0  -- This case is technically unreachable, but Lean requires it

/-- The sequence of points as described in the problem -/
def point_sequence (n : ℕ) : ℕ × ℤ :=
  (n, y_coord (n + 1))

theorem point_2023_coordinates :
  point_sequence 2022 = (2022, 0) := by sorry

end point_2023_coordinates_l706_70637


namespace trig_expression_eval_l706_70622

/-- Proves that the given trigonometric expression evaluates to -4√3 --/
theorem trig_expression_eval :
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) /
  (4 * (Real.cos (12 * π / 180))^2 * Real.sin (12 * π / 180) - 2 * Real.sin (12 * π / 180)) =
  -4 * Real.sqrt 3 := by
  sorry

end trig_expression_eval_l706_70622


namespace angle_between_vectors_l706_70683

def vector1 : Fin 3 → ℝ := ![1, 3, -2]
def vector2 : Fin 3 → ℝ := ![4, -2, 1]

theorem angle_between_vectors :
  let dot_product := (vector1 0) * (vector2 0) + (vector1 1) * (vector2 1) + (vector1 2) * (vector2 2)
  let magnitude1 := Real.sqrt ((vector1 0)^2 + (vector1 1)^2 + (vector1 2)^2)
  let magnitude2 := Real.sqrt ((vector2 0)^2 + (vector2 1)^2 + (vector2 2)^2)
  dot_product / (magnitude1 * magnitude2) = -2 / (7 * Real.sqrt 3) :=
by sorry

end angle_between_vectors_l706_70683


namespace f_inequality_l706_70641

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 1/x + 2 * Real.sin x

theorem f_inequality (x : ℝ) (hx : x > 0) :
  f (1 - x) > f x ↔ 0 < x ∧ x < 1/2 := by sorry

end f_inequality_l706_70641


namespace vector_difference_magnitude_l706_70691

/-- Given two vectors in ℝ², prove that the magnitude of their difference is 5 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by sorry

end vector_difference_magnitude_l706_70691


namespace acute_angles_are_in_first_quadrant_l706_70643

/- Definition of acute angle -/
def is_acute_angle (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

/- Definition of angle in the first quadrant -/
def is_in_first_quadrant (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

/- Theorem stating that acute angles are angles in the first quadrant -/
theorem acute_angles_are_in_first_quadrant :
  ∀ θ : Real, is_acute_angle θ → is_in_first_quadrant θ := by
  sorry

#check acute_angles_are_in_first_quadrant

end acute_angles_are_in_first_quadrant_l706_70643


namespace r_value_when_n_is_3_l706_70658

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^n + n
  let r : ℕ := 3^s - n^2
  r = 177138 := by sorry

end r_value_when_n_is_3_l706_70658


namespace geometric_sequence_property_l706_70662

/-- Given a geometric sequence {a_n} where a₂a₆ + a₄² = π, prove that a₃a₅ = π/2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_eq : a 2 * a 6 + a 4 * a 4 = Real.pi) : a 3 * a 5 = Real.pi / 2 := by
  sorry

end geometric_sequence_property_l706_70662


namespace incorrect_exponent_equality_l706_70695

theorem incorrect_exponent_equality : (-2)^2 ≠ -(2^2) :=
by
  -- Assuming the other equalities are true
  have h1 : 2^0 = 1 := by sorry
  have h2 : (-5)^3 = -(5^3) := by sorry
  have h3 : (-1/2)^3 = -1/8 := by sorry
  
  -- Proof that (-2)^2 ≠ -(2^2)
  sorry

end incorrect_exponent_equality_l706_70695


namespace complex_fraction_theorem_l706_70606

theorem complex_fraction_theorem (x y : ℂ) 
  (h : (x + y) / (x - y) - (x - y) / (x + y) = 3) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = -2.871 := by
  sorry

end complex_fraction_theorem_l706_70606


namespace sandys_puppies_l706_70665

/-- Given that Sandy initially had 8 puppies and gave away 4,
    prove that she now has 4 puppies. -/
theorem sandys_puppies :
  let initial_puppies : ℕ := 8
  let puppies_given_away : ℕ := 4
  let remaining_puppies := initial_puppies - puppies_given_away
  remaining_puppies = 4 :=
by sorry

end sandys_puppies_l706_70665


namespace quadratic_unique_solution_l706_70638

theorem quadratic_unique_solution (a b : ℝ) : 
  (∃! x, 16 * x^2 + a * x + b = 0) → 
  a^2 = 4 * b → 
  a = 0 ∧ b = 0 := by
sorry

end quadratic_unique_solution_l706_70638


namespace field_width_calculation_l706_70647

/-- Proves that the width of each field is 250 meters -/
theorem field_width_calculation (num_fields : ℕ) (field_length : ℝ) (total_area_km2 : ℝ) :
  num_fields = 8 →
  field_length = 300 →
  total_area_km2 = 0.6 →
  ∃ (width : ℝ), width = 250 ∧ 
    (num_fields * field_length * width = total_area_km2 * 1000000) :=
by sorry

end field_width_calculation_l706_70647


namespace quadratic_point_order_l706_70697

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 2*x + 2

-- Define the points A, B, and C
def A : ℝ × ℝ := (-2, f (-2))
def B : ℝ × ℝ := (-1, f (-1))
def C : ℝ × ℝ := (2, f 2)

-- State the theorem
theorem quadratic_point_order :
  A.2 < B.2 ∧ B.2 < C.2 :=
sorry

end quadratic_point_order_l706_70697


namespace money_spent_on_blades_l706_70689

def total_earned : ℕ := 42
def game_price : ℕ := 8
def num_games : ℕ := 4

theorem money_spent_on_blades : 
  total_earned - (game_price * num_games) = 10 := by
  sorry

end money_spent_on_blades_l706_70689


namespace pencil_count_l706_70645

theorem pencil_count (mitchell_pencils : ℕ) (difference : ℕ) : mitchell_pencils = 30 → difference = 6 →
  mitchell_pencils + (mitchell_pencils - difference) = 54 := by
  sorry

end pencil_count_l706_70645


namespace quadratic_solution_sum_l706_70672

/-- Represents a quadratic equation ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Represents a solution in the form (m ± √n)/p -/
structure QuadraticSolution where
  m : ℚ
  n : ℕ
  p : ℕ

/-- Check if three numbers are coprime -/
def are_coprime (m : ℚ) (n p : ℕ) : Prop :=
  Nat.gcd (Nat.gcd (Int.natAbs (Rat.num m)) n) p = 1

/-- The main theorem -/
theorem quadratic_solution_sum (eq : QuadraticEquation) (sol : QuadraticSolution) :
  eq.a = 3 ∧ eq.b = -7 ∧ eq.c = 3 ∧
  are_coprime sol.m sol.n sol.p ∧
  (∃ x : ℚ, x * (3 * x - 7) = -3 ∧ 
    (x = (sol.m + Real.sqrt sol.n) / sol.p ∨ 
     x = (sol.m - Real.sqrt sol.n) / sol.p)) →
  sol.m + sol.n + sol.p = 26 := by
  sorry

end quadratic_solution_sum_l706_70672


namespace seconds_in_3h45m_is_13500_l706_70624

/-- Converts hours to minutes -/
def hours_to_minutes (h : ℕ) : ℕ := h * 60

/-- Converts minutes to seconds -/
def minutes_to_seconds (m : ℕ) : ℕ := m * 60

/-- The number of seconds in 3 hours and 45 minutes -/
def seconds_in_3h45m : ℕ := minutes_to_seconds (hours_to_minutes 3 + 45)

theorem seconds_in_3h45m_is_13500 : seconds_in_3h45m = 13500 := by
  sorry

end seconds_in_3h45m_is_13500_l706_70624


namespace train_passing_platform_time_l706_70611

/-- Calculates the time taken for a train to pass a platform -/
theorem train_passing_platform_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 720)
  (h2 : train_speed_kmh = 72)
  (h3 : platform_length = 280) : 
  (train_length + platform_length) / (train_speed_kmh * 1000 / 3600) = 50 :=
sorry

end train_passing_platform_time_l706_70611


namespace t_level_quasi_increasing_range_l706_70693

/-- Definition of t-level quasi-increasing function -/
def is_t_level_quasi_increasing (f : ℝ → ℝ) (t : ℝ) (M : Set ℝ) : Prop :=
  ∀ x ∈ M, (x + t) ∈ M ∧ f (x + t) ≥ f x

/-- The function we're considering -/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- The interval we're considering -/
def M : Set ℝ := {x | x ≥ 1}

/-- The main theorem -/
theorem t_level_quasi_increasing_range :
  {t : ℝ | is_t_level_quasi_increasing f t M} = {t : ℝ | t ≥ 1} := by sorry

end t_level_quasi_increasing_range_l706_70693


namespace first_candidate_percentage_is_70_percent_l706_70649

/-- The percentage of votes the first candidate received in an election with two candidates -/
def first_candidate_percentage (total_votes : ℕ) (second_candidate_votes : ℕ) : ℚ :=
  (total_votes - second_candidate_votes : ℚ) / total_votes * 100

/-- Theorem stating that the first candidate received 70% of the votes -/
theorem first_candidate_percentage_is_70_percent :
  first_candidate_percentage 800 240 = 70 := by
  sorry

end first_candidate_percentage_is_70_percent_l706_70649


namespace matching_socks_probability_l706_70613

def gray_socks : ℕ := 10
def white_socks : ℕ := 8
def blue_socks : ℕ := 6

def total_socks : ℕ := gray_socks + white_socks + blue_socks

theorem matching_socks_probability :
  (Nat.choose gray_socks 2 + Nat.choose white_socks 2 + Nat.choose blue_socks 2) /
  Nat.choose total_socks 2 = 22 / 69 :=
by sorry

end matching_socks_probability_l706_70613


namespace pool_capacity_l706_70604

theorem pool_capacity (C : ℝ) 
  (h1 : 0.55 * C + 300 = 0.85 * C) : C = 1000 := by
  sorry

end pool_capacity_l706_70604


namespace wire_length_proof_l706_70692

theorem wire_length_proof (piece1 piece2 piece3 piece4 : ℝ) 
  (ratio_condition : piece1 / piece4 = 5 / 2 ∧ piece2 / piece4 = 7 / 2 ∧ piece3 / piece4 = 3 / 2)
  (shortest_piece : piece4 = 16) : 
  piece1 + piece2 + piece3 + piece4 = 136 := by
sorry

end wire_length_proof_l706_70692


namespace increasing_prime_sequence_ones_digit_l706_70688

/-- A sequence of four increasing prime numbers with common difference 4 and first term greater than 3 -/
def IncreasingPrimeSequence (p₁ p₂ p₃ p₄ : ℕ) : Prop :=
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧
  p₁ > 3 ∧
  p₂ = p₁ + 4 ∧
  p₃ = p₂ + 4 ∧
  p₄ = p₃ + 4

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

theorem increasing_prime_sequence_ones_digit
  (p₁ p₂ p₃ p₄ : ℕ) (h : IncreasingPrimeSequence p₁ p₂ p₃ p₄) :
  onesDigit p₁ = 9 := by
  sorry

end increasing_prime_sequence_ones_digit_l706_70688


namespace quadratic_equation_real_solutions_l706_70652

theorem quadratic_equation_real_solutions (x y z : ℝ) :
  (∃ z, 16 * z^2 + 4 * x * y * z + (y^2 - 3) = 0) ↔ x ≤ -2 ∨ x ≥ 2 := by
  sorry

end quadratic_equation_real_solutions_l706_70652


namespace cone_trajectory_length_l706_70639

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cone -/
structure Cone where
  base_side_length : ℝ
  apex : Point3D
  base_center : Point3D

/-- The theorem statement -/
theorem cone_trajectory_length 
  (c : Cone) 
  (h_base_side : c.base_side_length = 2) 
  (M : Point3D) 
  (h_M_midpoint : M = Point3D.mk 0 0 ((c.apex.z - c.base_center.z) / 2)) 
  (A : Point3D) 
  (h_A_on_base : A.z = c.base_center.z ∧ (A.x - c.base_center.x)^2 + (A.y - c.base_center.y)^2 = 1) 
  (P : Point3D → Prop) 
  (h_P_on_base : ∀ p, P p → p.z = c.base_center.z ∧ (p.x - c.base_center.x)^2 + (p.y - c.base_center.y)^2 ≤ 1) 
  (h_AM_perp_MP : ∀ p, P p → (M.x - A.x) * (p.x - M.x) + (M.y - A.y) * (p.y - M.y) + (M.z - A.z) * (p.z - M.z) = 0) :
  (∃ l : ℝ, l = Real.sqrt 7 / 2 ∧ 
    ∀ ε > 0, ∃ δ > 0, ∀ p q, P p → P q → abs (p.x - q.x) < δ ∧ abs (p.y - q.y) < δ → 
      abs (Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2) - l) < ε) :=
by sorry

end cone_trajectory_length_l706_70639


namespace regular_polygon_interior_exterior_angle_relation_l706_70675

theorem regular_polygon_interior_exterior_angle_relation (n : ℕ) :
  (n ≥ 3) →
  ((n - 2) * 180 : ℝ) = 2 * 360 →
  n = 6 :=
by sorry

end regular_polygon_interior_exterior_angle_relation_l706_70675


namespace track_laying_equation_l706_70612

theorem track_laying_equation (x : ℝ) (h : x > 0) :
  (6000 / x - 6000 / (x + 20) = 15) ↔
  (∃ (original_days revised_days : ℝ),
    original_days > 0 ∧
    revised_days > 0 ∧
    original_days = 6000 / x ∧
    revised_days = 6000 / (x + 20) ∧
    original_days - revised_days = 15) :=
by sorry

end track_laying_equation_l706_70612


namespace tan_inequality_equiv_l706_70698

theorem tan_inequality_equiv (x : ℝ) : 
  Real.tan (2 * x - π / 4) ≤ 1 ↔ 
  ∃ k : ℤ, k * π / 2 - π / 8 < x ∧ x ≤ k * π / 2 + π / 4 := by
sorry

end tan_inequality_equiv_l706_70698


namespace function_inequality_l706_70676

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x, x * deriv f x ≥ 0) : 
  f (-1) + f 1 ≥ 2 * f 0 := by
  sorry

end function_inequality_l706_70676


namespace arithmetic_sum_modulo_15_l706_70605

/-- The sum of an arithmetic sequence modulo m -/
def arithmetic_sum_mod (a₁ aₙ d n m : ℕ) : ℕ :=
  ((n * (a₁ + aₙ)) / 2) % m

/-- The number of terms in an arithmetic sequence -/
def arithmetic_terms (a₁ aₙ d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

theorem arithmetic_sum_modulo_15 :
  let a₁ := 2  -- First term
  let aₙ := 102  -- Last term
  let d := 5   -- Common difference
  let m := 15  -- Modulus
  let n := arithmetic_terms a₁ aₙ d
  arithmetic_sum_mod a₁ aₙ d n m = 6 := by
sorry

end arithmetic_sum_modulo_15_l706_70605


namespace probability_divisible_by_5_l706_70607

/-- A three-digit number is an integer between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The count of three-digit numbers divisible by 5. -/
def CountDivisibleBy5 : ℕ := 180

/-- The total count of three-digit numbers. -/
def TotalThreeDigitNumbers : ℕ := 900

/-- The probability of a randomly chosen three-digit number being divisible by 5 is 1/5. -/
theorem probability_divisible_by_5 :
  (CountDivisibleBy5 : ℚ) / TotalThreeDigitNumbers = 1 / 5 := by
  sorry


end probability_divisible_by_5_l706_70607


namespace abs_difference_over_sum_equals_sqrt_three_sevenths_l706_70666

theorem abs_difference_over_sum_equals_sqrt_three_sevenths
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 5*a*b) :
  |((a - b) / (a + b))| = Real.sqrt (3/7) :=
by sorry

end abs_difference_over_sum_equals_sqrt_three_sevenths_l706_70666


namespace sqrt_t6_plus_t4_l706_70659

theorem sqrt_t6_plus_t4 (t : ℝ) : Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) := by
  sorry

end sqrt_t6_plus_t4_l706_70659


namespace quadratic_inequality_solution_set_l706_70608

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x ≤ 0} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

end quadratic_inequality_solution_set_l706_70608


namespace dust_retention_proof_l706_70664

/-- The average dust retention of a ginkgo leaf in milligrams per year. -/
def ginkgo_retention : ℝ := 40

/-- The average dust retention of a locust leaf in milligrams per year. -/
def locust_retention : ℝ := 22

/-- The number of ginkgo leaves. -/
def num_ginkgo_leaves : ℕ := 50000

theorem dust_retention_proof :
  -- Condition 1: Ginkgo retention is 4mg less than twice locust retention
  ginkgo_retention = 2 * locust_retention - 4 ∧
  -- Condition 2: Total retention of ginkgo and locust is 62mg
  ginkgo_retention + locust_retention = 62 ∧
  -- Result 1: Ginkgo retention is 40mg
  ginkgo_retention = 40 ∧
  -- Result 2: Locust retention is 22mg
  locust_retention = 22 ∧
  -- Result 3: Total retention of 50,000 ginkgo leaves is 2kg
  (ginkgo_retention * num_ginkgo_leaves) / 1000000 = 2 :=
by sorry

end dust_retention_proof_l706_70664


namespace line_l_passes_through_M_line_l1_properties_l706_70696

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  (2 + m) * x + (1 - 2 * m) * y + 4 - 3 * m = 0

-- Define the point M
def point_M : ℝ × ℝ := (-1, -2)

-- Define the line l1
def line_l1 (x y : ℝ) : Prop :=
  2 * x + y + 4 = 0

-- Theorem 1: Line l passes through point M for all real m
theorem line_l_passes_through_M :
  ∀ m : ℝ, line_l m (point_M.1) (point_M.2) := by sorry

-- Theorem 2: Line l1 passes through point M and is bisected by M
theorem line_l1_properties :
  line_l1 (point_M.1) (point_M.2) ∧
  ∃ (A B : ℝ × ℝ),
    (A.1 = 0 ∨ A.2 = 0) ∧
    (B.1 = 0 ∨ B.2 = 0) ∧
    line_l1 A.1 A.2 ∧
    line_l1 B.1 B.2 ∧
    ((A.1 + B.1) / 2 = point_M.1 ∧ (A.2 + B.2) / 2 = point_M.2) := by sorry

end line_l_passes_through_M_line_l1_properties_l706_70696


namespace roses_cut_l706_70614

theorem roses_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) 
  (h1 : initial_roses = 13)
  (h2 : initial_orchids = 84)
  (h3 : final_roses = 14)
  (h4 : final_orchids = 91) :
  final_roses - initial_roses = 1 := by
  sorry

end roses_cut_l706_70614


namespace price_reduction_equation_l706_70653

-- Define the original price
def original_price : ℝ := 200

-- Define the final price after reductions
def final_price : ℝ := 162

-- Define the average percentage reduction
variable (x : ℝ)

-- Theorem statement
theorem price_reduction_equation :
  original_price * (1 - x)^2 = final_price :=
sorry

end price_reduction_equation_l706_70653


namespace two_derived_point_of_neg_two_three_original_point_from_three_derived_k_range_for_distance_condition_l706_70686

/-- Definition of k-derived point -/
def k_derived_point (k : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 + k * P.2, k * P.1 + P.2)

/-- Theorem 1: The 2-derived point of (-2,3) is (4, -1) -/
theorem two_derived_point_of_neg_two_three :
  k_derived_point 2 (-2, 3) = (4, -1) := by sorry

/-- Theorem 2: If the 3-derived point of P is (9,11), then P is (3,2) -/
theorem original_point_from_three_derived :
  ∀ P : ℝ × ℝ, k_derived_point 3 P = (9, 11) → P = (3, 2) := by sorry

/-- Theorem 3: For a point P(0,b) on the positive y-axis, its k-derived point P'(kb,b) 
    has |kb| ≥ 5b if and only if k ≥ 5 or k ≤ -5 -/
theorem k_range_for_distance_condition :
  ∀ k b : ℝ, b > 0 → (|k * b| ≥ 5 * b ↔ k ≥ 5 ∨ k ≤ -5) := by sorry

end two_derived_point_of_neg_two_three_original_point_from_three_derived_k_range_for_distance_condition_l706_70686


namespace peanut_butter_cost_l706_70601

/-- The cost of the jar of peanut butter given the cost of bread, initial money, and money left over -/
theorem peanut_butter_cost
  (bread_cost : ℝ)
  (bread_quantity : ℕ)
  (initial_money : ℝ)
  (money_left : ℝ)
  (h1 : bread_cost = 2.25)
  (h2 : bread_quantity = 3)
  (h3 : initial_money = 14)
  (h4 : money_left = 5.25) :
  initial_money - money_left - bread_cost * bread_quantity = 2 :=
by sorry

end peanut_butter_cost_l706_70601


namespace consecutive_integers_product_sum_l706_70648

theorem consecutive_integers_product_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end consecutive_integers_product_sum_l706_70648


namespace union_and_intersection_of_A_and_B_l706_70610

variable (a : ℝ)

def A : Set ℝ := {x | (x - 3) * (x - a) = 0}
def B : Set ℝ := {x | (x - 4) * (x - 1) = 0}

theorem union_and_intersection_of_A_and_B :
  (a = 3 → (A a ∪ B = {1, 3, 4} ∧ A a ∩ B = ∅)) ∧
  (a = 1 → (A a ∪ B = {1, 3, 4} ∧ A a ∩ B = {1})) ∧
  (a = 4 → (A a ∪ B = {1, 3, 4} ∧ A a ∩ B = {4})) ∧
  (a ≠ 1 ∧ a ≠ 3 ∧ a ≠ 4 → (A a ∪ B = {1, 3, 4, a} ∧ A a ∩ B = ∅)) :=
by sorry

end union_and_intersection_of_A_and_B_l706_70610


namespace linear_equation_exponent_values_l706_70602

theorem linear_equation_exponent_values (m n : ℤ) : 
  (∀ x y : ℝ, ∃ a b c : ℝ, a ≠ 0 ∨ b ≠ 0 ∧ a * x + b * y + c = 5 * x^(3*m-2*n) - 2 * y^(n-m) + 11) →
  m = 0 ∧ n = 1 := by
sorry

end linear_equation_exponent_values_l706_70602


namespace cookie_problem_l706_70661

theorem cookie_problem (total_cookies : ℕ) (total_nuts : ℕ) (nuts_per_cookie : ℕ) 
  (h_total_cookies : total_cookies = 60)
  (h_total_nuts : total_nuts = 72)
  (h_nuts_per_cookie : nuts_per_cookie = 2)
  (h_quarter_nuts : (total_cookies / 4 : ℚ) = (total_cookies - (total_nuts / nuts_per_cookie) : ℕ)) :
  (((total_cookies - (total_cookies / 4) - (total_nuts / nuts_per_cookie - total_cookies / 4)) / total_cookies : ℚ) * 100 = 40) := by
  sorry

end cookie_problem_l706_70661


namespace product_of_sum_and_difference_l706_70635

theorem product_of_sum_and_difference (x y : ℝ) : 
  x + y = 50 → x - y = 6 → x * y = 616 := by
sorry

end product_of_sum_and_difference_l706_70635


namespace initial_girls_count_l706_70680

theorem initial_girls_count (initial_total : ℕ) (initial_girls : ℕ) : 
  (initial_girls = initial_total / 2) →
  (initial_girls - 3) * 10 = 4 * (initial_total + 1) →
  (initial_girls - 4) * 20 = 7 * (initial_total + 2) →
  initial_girls = 17 := by
sorry

end initial_girls_count_l706_70680


namespace inequality_proof_l706_70685

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c)/(a + 2*b + c) + 4*b/(a + b + 2*c) - 8*c/(a + b + 3*c) ≥ -17 + 12*Real.sqrt 2 := by
  sorry

end inequality_proof_l706_70685


namespace worker_a_completion_time_l706_70668

/-- 
Given two workers a and b who can complete a work in 4 days together, 
and in 8/3 days when working simultaneously,
prove that worker a alone can complete the work in 8 days.
-/
theorem worker_a_completion_time 
  (total_time : ℝ) 
  (combined_time : ℝ) 
  (ha : total_time = 4) 
  (hb : combined_time = 8/3) : 
  ∃ (a_time : ℝ), a_time = 8 := by
  sorry

end worker_a_completion_time_l706_70668


namespace square_root_problem_l706_70694

theorem square_root_problem (h1 : Real.sqrt 99225 = 315) (h2 : Real.sqrt x = 3.15) : x = 9.9225 := by
  sorry

end square_root_problem_l706_70694


namespace integer_expression_l706_70670

theorem integer_expression (n : ℕ) : ∃ k : ℤ, (n^5 : ℚ) / 5 + (n^3 : ℚ) / 3 + (7 * n : ℚ) / 15 = k := by
  sorry

end integer_expression_l706_70670


namespace polynomial_evaluation_l706_70682

theorem polynomial_evaluation (a : ℝ) (h : a = 2) : (7*a^2 - 20*a + 5) * (3*a - 4) = -14 := by
  sorry

end polynomial_evaluation_l706_70682


namespace min_value_of_fraction_l706_70609

theorem min_value_of_fraction (x : ℝ) (h : x > 10) :
  x^2 / (x - 10) ≥ 30 ∧ ∃ y > 10, y^2 / (y - 10) = 30 := by
  sorry

end min_value_of_fraction_l706_70609


namespace hundred_with_five_twos_l706_70626

theorem hundred_with_five_twos :
  ∃ (a b c d e : ℕ), a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2 ∧
  (a * b * c / d - e / d = 100) := by
  sorry

end hundred_with_five_twos_l706_70626


namespace inequality_proof_l706_70621

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum : x*y + y*z + z*x = 1) : 
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5/2 := by
  sorry

end inequality_proof_l706_70621


namespace symmetric_rook_placements_8x8_l706_70677

/-- Represents a chessboard configuration with rooks placed symmetrically --/
structure SymmetricRookPlacement where
  board_size : Nat
  num_rooks : Nat
  is_symmetric : Bool

/-- Counts the number of symmetric rook placements on a chessboard --/
def count_symmetric_rook_placements (config : SymmetricRookPlacement) : Nat :=
  sorry

/-- Theorem stating the number of symmetric rook placements for 8 rooks on an 8x8 chessboard --/
theorem symmetric_rook_placements_8x8 :
  count_symmetric_rook_placements ⟨8, 8, true⟩ = 139448 := by
  sorry

end symmetric_rook_placements_8x8_l706_70677


namespace seashells_given_theorem_l706_70623

/-- The number of seashells Tim gave to Sara -/
def seashells_given_to_sara (initial_seashells final_seashells : ℕ) : ℕ :=
  initial_seashells - final_seashells

/-- Theorem stating that the number of seashells given to Sara is the difference between
    the initial and final counts of seashells Tim has -/
theorem seashells_given_theorem (initial_seashells final_seashells : ℕ) 
    (h : initial_seashells ≥ final_seashells) :
  seashells_given_to_sara initial_seashells final_seashells = initial_seashells - final_seashells :=
by
  sorry

#eval seashells_given_to_sara 679 507

end seashells_given_theorem_l706_70623


namespace vector_at_negative_three_l706_70616

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  -- The vector on the line at parameter t
  vector : ℝ → ℝ × ℝ

/-- The given parameterized line satisfying the problem conditions -/
def given_line : ParameterizedLine :=
  { vector := sorry }

theorem vector_at_negative_three :
  given_line.vector 1 = (4, 5) →
  given_line.vector 5 = (12, -11) →
  given_line.vector (-3) = (-4, 21) := by
  sorry

end vector_at_negative_three_l706_70616


namespace candy_bar_cost_l706_70636

theorem candy_bar_cost (num_members : ℕ) (avg_sold_per_member : ℕ) (total_earnings : ℚ) :
  num_members = 20 →
  avg_sold_per_member = 8 →
  total_earnings = 80 →
  (total_earnings / (num_members * avg_sold_per_member : ℚ)) = 0.5 := by
  sorry

end candy_bar_cost_l706_70636


namespace last_part_distance_calculation_l706_70600

/-- Calculates the distance of the last part of a trip given the total distance,
    first part distance, speeds for different parts, and average speed. -/
def last_part_distance (total_distance first_part_distance first_part_speed
                        average_speed last_part_speed : ℝ) : ℝ :=
  total_distance - first_part_distance

theorem last_part_distance_calculation (total_distance : ℝ) (first_part_distance : ℝ)
    (first_part_speed : ℝ) (average_speed : ℝ) (last_part_speed : ℝ)
    (h1 : total_distance = 100)
    (h2 : first_part_distance = 30)
    (h3 : first_part_speed = 60)
    (h4 : average_speed = 40)
    (h5 : last_part_speed = 35) :
  last_part_distance total_distance first_part_distance first_part_speed average_speed last_part_speed = 70 := by
sorry

end last_part_distance_calculation_l706_70600


namespace expression_simplification_and_evaluation_l706_70669

theorem expression_simplification_and_evaluation (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4*a + 4) / (a + 1)) = (2 + a) / (2 - a) ∧
  (2 + 1) / (2 - 1) = 3 :=
by sorry

end expression_simplification_and_evaluation_l706_70669


namespace sequence_periodicity_l706_70656

theorem sequence_periodicity (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a (n + 2) = |a (n + 5)| - a (n + 4)) : 
  ∃ N : ℕ, ∀ n ≥ N, a n = a (n + 9) :=
sorry

end sequence_periodicity_l706_70656


namespace population_growth_theorem_l706_70620

/-- Calculates the population after a given number of years -/
def population_after_years (initial_population : ℕ) (birth_rate : ℚ) (death_rate : ℚ) (years : ℕ) : ℚ :=
  match years with
  | 0 => initial_population
  | n + 1 => 
    let prev_population := population_after_years initial_population birth_rate death_rate n
    prev_population + prev_population * birth_rate - prev_population * death_rate

/-- The population after 2 years is approximately 53045 -/
theorem population_growth_theorem : 
  let initial_population : ℕ := 50000
  let birth_rate : ℚ := 43 / 1000
  let death_rate : ℚ := 13 / 1000
  let years : ℕ := 2
  ⌊population_after_years initial_population birth_rate death_rate years⌋ = 53045 := by
  sorry


end population_growth_theorem_l706_70620


namespace hyperbola_equation_for_given_parameters_l706_70671

/-- Represents a hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ  -- Half of the real axis length
  e : ℝ  -- Eccentricity

/-- The equation of a hyperbola with given parameters -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / (h.a^2 * (h.e^2 - 1)) = 1

theorem hyperbola_equation_for_given_parameters :
  let h : Hyperbola := { a := 3, e := 5/3 }
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2 / 9 - y^2 / 16 = 1 := by
  sorry

end hyperbola_equation_for_given_parameters_l706_70671
