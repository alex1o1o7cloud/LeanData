import Mathlib

namespace NUMINAMATH_CALUDE_dish_sets_budget_l1573_157330

theorem dish_sets_budget (total_budget : ℕ) (sets_at_20 : ℕ) (price_per_set : ℕ) :
  total_budget = 6800 →
  sets_at_20 = 178 →
  price_per_set = 20 →
  total_budget - (sets_at_20 * price_per_set) = 3240 :=
by sorry

end NUMINAMATH_CALUDE_dish_sets_budget_l1573_157330


namespace NUMINAMATH_CALUDE_income_relationship_l1573_157313

theorem income_relationship (juan tim mart : ℝ) 
  (h1 : mart = 1.4 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mart = 0.84 * juan := by
sorry

end NUMINAMATH_CALUDE_income_relationship_l1573_157313


namespace NUMINAMATH_CALUDE_m_minus_reciprocal_l1573_157337

theorem m_minus_reciprocal (m : ℝ) (h : m^2 + 3*m = -1) : m - 1/(m+1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_m_minus_reciprocal_l1573_157337


namespace NUMINAMATH_CALUDE_adjacent_repeat_percentage_is_16_l1573_157334

/-- The count of three-digit numbers -/
def three_digit_count : ℕ := 900

/-- The count of three-digit numbers with adjacent repeated digits -/
def adjacent_repeat_count : ℕ := 144

/-- The percentage of three-digit numbers with adjacent repeated digits -/
def adjacent_repeat_percentage : ℚ := adjacent_repeat_count / three_digit_count * 100

/-- Theorem stating that the percentage of three-digit numbers with adjacent repeated digits is 16.0% -/
theorem adjacent_repeat_percentage_is_16 :
  ⌊adjacent_repeat_percentage * 10⌋ / 10 = 16 :=
sorry

end NUMINAMATH_CALUDE_adjacent_repeat_percentage_is_16_l1573_157334


namespace NUMINAMATH_CALUDE_square_bricks_count_square_bricks_count_proof_l1573_157339

theorem square_bricks_count : ℕ → Prop :=
  fun total =>
    ∃ (length width : ℕ),
      -- Condition 1: length to width ratio is 6:5
      6 * width = 5 * length ∧
      -- Condition 2: rectangle arrangement leaves 43 bricks
      length * width + 43 = total ∧
      -- Condition 3: increasing both dimensions by 1 results in 68 bricks short
      (length + 1) * (width + 1) = total - 68 ∧
      -- The total number of bricks is 3043
      total = 3043

-- The proof of the theorem
theorem square_bricks_count_proof : square_bricks_count 3043 := by
  sorry

end NUMINAMATH_CALUDE_square_bricks_count_square_bricks_count_proof_l1573_157339


namespace NUMINAMATH_CALUDE_smallest_number_proof_l1573_157363

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →                    -- Three positive integers
  (a + b + c) / 3 = 30 →                     -- Arithmetic mean is 30
  b = 28 →                                   -- Median is 28
  c = b + 7 →                                -- Largest number is 7 more than median
  a ≤ b ∧ b ≤ c →                            -- b is the median
  a = 27                                     -- Smallest number is 27
:= by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l1573_157363


namespace NUMINAMATH_CALUDE_count_distinct_digits_eq_2352_l1573_157325

/-- The count of integers between 2000 and 9999 with four distinct digits, none of which is 5 -/
def count_distinct_digits : ℕ :=
  let first_digit := 7  -- 2, 3, 4, 6, 7, 8, 9
  let second_digit := 8 -- 0, 1, 2, 3, 4, 6, 7, 8, 9 (excluding the first digit)
  let third_digit := 7  -- remaining digits excluding 5 and the first two chosen
  let fourth_digit := 6 -- remaining digits excluding 5 and the first three chosen
  first_digit * second_digit * third_digit * fourth_digit

theorem count_distinct_digits_eq_2352 : count_distinct_digits = 2352 := by
  sorry

end NUMINAMATH_CALUDE_count_distinct_digits_eq_2352_l1573_157325


namespace NUMINAMATH_CALUDE_cos_225_degrees_l1573_157329

theorem cos_225_degrees : Real.cos (225 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l1573_157329


namespace NUMINAMATH_CALUDE_area_is_two_l1573_157351

open Real MeasureTheory

noncomputable def area_bounded_by_curves : ℝ :=
  ∫ x in (1/Real.exp 1)..Real.exp 1, (1/x)

theorem area_is_two : area_bounded_by_curves = 2 := by
  sorry

end NUMINAMATH_CALUDE_area_is_two_l1573_157351


namespace NUMINAMATH_CALUDE_inequality_proof_l1573_157366

theorem inequality_proof (x : ℝ) (h : x > 0) : 1/x + 4*x^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1573_157366


namespace NUMINAMATH_CALUDE_fourth_roll_eight_prob_l1573_157308

-- Define the probabilities for the fair die
def fair_die_prob : ℚ := 1 / 8

-- Define the probabilities for the biased die
def biased_die_prob_eight : ℚ := 3 / 4
def biased_die_prob_other : ℚ := 1 / 28

-- Define the probability of selecting each die
def die_selection_prob : ℚ := 1 / 2

-- Define the number of rolls
def num_rolls : ℕ := 4

-- Define the number of sides on each die
def num_sides : ℕ := 8

-- Define the theorem
theorem fourth_roll_eight_prob :
  let p_fair_three_eights : ℚ := fair_die_prob ^ 3
  let p_biased_three_eights : ℚ := biased_die_prob_eight ^ 3
  let p_three_eights : ℚ := die_selection_prob * p_fair_three_eights + die_selection_prob * p_biased_three_eights
  let p_fair_given_three_eights : ℚ := (die_selection_prob * p_fair_three_eights) / p_three_eights
  let p_biased_given_three_eights : ℚ := (die_selection_prob * p_biased_three_eights) / p_three_eights
  let p_fourth_eight : ℚ := p_fair_given_three_eights * fair_die_prob + p_biased_given_three_eights * biased_die_prob_eight
  p_fourth_eight = 1297 / 1736 :=
by sorry

end NUMINAMATH_CALUDE_fourth_roll_eight_prob_l1573_157308


namespace NUMINAMATH_CALUDE_one_true_proposition_l1573_157353

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop :=
  a^2 + 2*a*b + b^2 + a + b - 2 ≠ 0 → a + b ≠ 1

-- Define the inverse of the proposition
def inverse_proposition (a b : ℝ) : Prop :=
  a + b ≠ 1 → a^2 + 2*a*b + b^2 + a + b - 2 ≠ 0

-- Define the negation of the proposition
def negation_proposition (a b : ℝ) : Prop :=
  a^2 + 2*a*b + b^2 + a + b - 2 = 0 → a + b = 1

-- Define the contrapositive of the proposition
def contrapositive_proposition (a b : ℝ) : Prop :=
  a + b = 1 → a^2 + 2*a*b + b^2 + a + b - 2 = 0

-- Theorem statement
theorem one_true_proposition :
  ∃! p : (ℝ → ℝ → Prop), 
    (p = inverse_proposition ∨ p = negation_proposition ∨ p = contrapositive_proposition) ∧
    (∀ a b : ℝ, p a b) :=
  sorry

end NUMINAMATH_CALUDE_one_true_proposition_l1573_157353


namespace NUMINAMATH_CALUDE_decimal_point_movement_l1573_157318

theorem decimal_point_movement (x : ℝ) : x / 100 = x - 1.485 ↔ x = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_movement_l1573_157318


namespace NUMINAMATH_CALUDE_polygon_sides_l1573_157301

/-- Given a polygon with sum of interior angles equal to 1080°, prove it has 8 sides -/
theorem polygon_sides (sum_interior_angles : ℝ) (h : sum_interior_angles = 1080) : 
  (sum_interior_angles / 180 + 2 : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1573_157301


namespace NUMINAMATH_CALUDE_derivative_of_product_l1573_157309

theorem derivative_of_product (x : ℝ) :
  deriv (fun x => (3 * x^2 - 4*x) * (2*x + 1)) x = 18 * x^2 - 10 * x - 4 := by
sorry

end NUMINAMATH_CALUDE_derivative_of_product_l1573_157309


namespace NUMINAMATH_CALUDE_new_person_weight_l1573_157372

/-- Given a group of 8 persons where one person weighing 65 kg is replaced by a new person,
    and the average weight of the group increases by 2.5 kg,
    prove that the weight of the new person is 85 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 85 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1573_157372


namespace NUMINAMATH_CALUDE_train_length_l1573_157354

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  speed = 72 * (5/18) → 
  platform_length = 250 → 
  crossing_time = 24 → 
  speed * crossing_time - platform_length = 230 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1573_157354


namespace NUMINAMATH_CALUDE_four_digit_count_l1573_157312

-- Define the range of four-digit numbers
def four_digit_start : ℕ := 1000
def four_digit_end : ℕ := 9999

-- Theorem statement
theorem four_digit_count : 
  (Finset.range (four_digit_end - four_digit_start + 1)).card = 9000 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_count_l1573_157312


namespace NUMINAMATH_CALUDE_g_range_l1573_157319

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x/3))^2 + (Real.pi/4) * Real.arcsin (x/3) - (Real.arcsin (x/3))^2 + (Real.pi^2/16) * (x^2 + 2*x + 3)

theorem g_range : 
  ∀ x ∈ Set.Icc (-3 : ℝ) 3, 
    g x ∈ Set.Icc (Real.pi^2/4) ((15*Real.pi^2/16) + (Real.pi/4)*Real.arcsin 1) ∧
    ∃ y ∈ Set.Icc (-3 : ℝ) 3, g y = Real.pi^2/4 ∧
    ∃ z ∈ Set.Icc (-3 : ℝ) 3, g z = (15*Real.pi^2/16) + (Real.pi/4)*Real.arcsin 1 :=
by sorry

end NUMINAMATH_CALUDE_g_range_l1573_157319


namespace NUMINAMATH_CALUDE_pie_chart_most_suitable_l1573_157375

/-- Represents a component of milk with its percentage -/
structure MilkComponent where
  name : String
  percentage : Float

/-- Represents a type of graph -/
inductive GraphType
  | PieChart
  | BarGraph
  | LineGraph
  | ScatterPlot

/-- Determines if a list of percentages sums to 100% (allowing for small floating-point errors) -/
def sumsToWhole (components : List MilkComponent) : Bool :=
  let sum := components.map (·.percentage) |>.sum
  sum > 99.99 && sum < 100.01

/-- Determines if a graph type is suitable for representing percentages of a whole -/
def isSuitableForPercentages (graphType : GraphType) : Bool :=
  match graphType with
  | GraphType.PieChart => true
  | _ => false

/-- Theorem: A pie chart is the most suitable graph type for representing milk components -/
theorem pie_chart_most_suitable (components : List MilkComponent) 
  (h_components : components = [
    ⟨"Water", 82⟩, 
    ⟨"Protein", 4.3⟩, 
    ⟨"Fat", 6⟩, 
    ⟨"Lactose", 7⟩, 
    ⟨"Other", 0.7⟩
  ])
  (h_sum : sumsToWhole components) :
  ∀ (graphType : GraphType), 
    isSuitableForPercentages graphType → graphType = GraphType.PieChart :=
by sorry

end NUMINAMATH_CALUDE_pie_chart_most_suitable_l1573_157375


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1573_157396

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 + i) / (3 - i) = (1 + 2*i) / 5 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1573_157396


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_m_geq_6_l1573_157388

-- Define the conditions p and q as functions
def p (x : ℝ) : Prop := (x - 1) / x ≤ 0
def q (x m : ℝ) : Prop := 4^x + 2^x - m ≤ 0

-- State the theorem
theorem sufficient_condition_implies_m_geq_6 :
  (∀ x m : ℝ, p x → q x m) → ∀ m : ℝ, m ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_m_geq_6_l1573_157388


namespace NUMINAMATH_CALUDE_rose_work_days_l1573_157360

/-- Proves that if John completes a work in 320 days, and both John and Rose together complete
    the same work in 192 days, then Rose completes the work alone in 384 days. -/
theorem rose_work_days (john_days : ℕ) (together_days : ℕ) (rose_days : ℕ) : 
  john_days = 320 → together_days = 192 → 
  1 / john_days + 1 / rose_days = 1 / together_days → 
  rose_days = 384 := by
sorry

end NUMINAMATH_CALUDE_rose_work_days_l1573_157360


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l1573_157316

/-- A line in the 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept) + 0)

/-- Theorem: For a line with slope -3 and x-intercept (7,0), the y-intercept is (0,21). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -3, x_intercept := 7 }
  y_intercept l = (0, 21) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l1573_157316


namespace NUMINAMATH_CALUDE_average_age_problem_l1573_157391

theorem average_age_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 29 →
  b = 23 →
  (a + c) / 2 = 32 := by
sorry

end NUMINAMATH_CALUDE_average_age_problem_l1573_157391


namespace NUMINAMATH_CALUDE_total_eggs_is_63_l1573_157338

/-- The number of Easter eggs Hannah found -/
def hannah_eggs : ℕ := 42

/-- The number of Easter eggs Helen found -/
def helen_eggs : ℕ := hannah_eggs / 2

/-- The total number of Easter eggs in the yard -/
def total_eggs : ℕ := hannah_eggs + helen_eggs

/-- Theorem stating that the total number of Easter eggs in the yard is 63 -/
theorem total_eggs_is_63 : total_eggs = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_is_63_l1573_157338


namespace NUMINAMATH_CALUDE_second_number_in_sequence_l1573_157369

/-- The second number in the sequence of numbers that, when divided by 7, 9, and 11,
    always leaves a remainder of 5, given that 1398 - 22 = 1376 is the first such number. -/
theorem second_number_in_sequence (first_number : ℕ) (h1 : first_number = 1376) :
  ∃ (second_number : ℕ),
    second_number > first_number ∧
    second_number % 7 = 5 ∧
    second_number % 9 = 5 ∧
    second_number % 11 = 5 ∧
    ∀ (n : ℕ), first_number < n ∧ n < second_number →
      (n % 7 ≠ 5 ∨ n % 9 ≠ 5 ∨ n % 11 ≠ 5) :=
by sorry

end NUMINAMATH_CALUDE_second_number_in_sequence_l1573_157369


namespace NUMINAMATH_CALUDE_factor_condition_l1573_157327

/-- A quadratic trinomial can be factored using the cross multiplication method if
    there exist two integers that multiply to give the constant term and add up to
    the coefficient of x. -/
def can_be_factored_by_cross_multiplication (a b c : ℤ) : Prop :=
  ∃ (p q : ℤ), p * q = c ∧ p + q = b

/-- If x^2 + kx + 5 can be factored using the cross multiplication method,
    then k = 6 or k = -6 -/
theorem factor_condition (k : ℤ) :
  can_be_factored_by_cross_multiplication 1 k 5 → k = 6 ∨ k = -6 := by
  sorry

end NUMINAMATH_CALUDE_factor_condition_l1573_157327


namespace NUMINAMATH_CALUDE_middle_number_proof_l1573_157373

theorem middle_number_proof (x y : ℝ) : 
  (3*x)^2 + (2*x)^2 + (5*x)^2 = 1862 →
  3*x + 2*x + 5*x + 4*y + 7*y = 155 →
  2*x = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l1573_157373


namespace NUMINAMATH_CALUDE_angle_calculation_l1573_157397

-- Define the triangles and angles
def Triangle (a b c : ℝ) := a + b + c = 180

-- Theorem statement
theorem angle_calculation (T1_angle1 T1_angle2 T2_angle1 T2_angle2 α β : ℝ) 
  (h1 : Triangle T1_angle1 T1_angle2 (180 - α))
  (h2 : Triangle T2_angle1 T2_angle2 β)
  (h3 : T1_angle1 = 70)
  (h4 : T1_angle2 = 50)
  (h5 : T2_angle1 = 45)
  (h6 : T2_angle2 = 50) :
  α = 120 ∧ β = 85 := by
  sorry

end NUMINAMATH_CALUDE_angle_calculation_l1573_157397


namespace NUMINAMATH_CALUDE_digital_earth_prospects_l1573_157365

/-- Represents the prospects of digital Earth applications -/
structure DigitalEarthProspects where
  spatialLab : Bool  -- Provides a digital spatial laboratory
  decisionMaking : Bool  -- Government decision-making can fully rely on it
  urbanManagement : Bool  -- Provides a basis for urban management
  predictable : Bool  -- The development is predictable

/-- The correct prospects of digital Earth applications -/
def correctProspects : DigitalEarthProspects :=
  { spatialLab := true
    decisionMaking := false
    urbanManagement := true
    predictable := false }

/-- Theorem stating the correct prospects of digital Earth applications -/
theorem digital_earth_prospects :
  (correctProspects.spatialLab = true) ∧
  (correctProspects.urbanManagement = true) ∧
  (correctProspects.decisionMaking = false) ∧
  (correctProspects.predictable = false) := by
  sorry


end NUMINAMATH_CALUDE_digital_earth_prospects_l1573_157365


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1573_157381

theorem arithmetic_sequence_difference (a b c : ℚ) : 
  (∃ d : ℚ, d = (9 - 2) / 4 ∧ 
             a = 2 + d ∧ 
             b = 2 + 2*d ∧ 
             c = 2 + 3*d ∧ 
             9 = 2 + 4*d) → 
  c - a = 3.5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1573_157381


namespace NUMINAMATH_CALUDE_diagonals_bisect_implies_parallelogram_parallelogram_right_angle_implies_rectangle_l1573_157387

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Define the property of diagonals bisecting each other
def diagonals_bisect (q : Quadrilateral) : Prop := sorry

-- Define the property of having a right angle
def has_right_angle (q : Quadrilateral) : Prop := sorry

-- Theorem 1: If a quadrilateral has diagonals that bisect each other, then it is a parallelogram
theorem diagonals_bisect_implies_parallelogram (q : Quadrilateral) :
  diagonals_bisect q → is_parallelogram q :=
sorry

-- Theorem 2: If a parallelogram has one right angle, then it is a rectangle
theorem parallelogram_right_angle_implies_rectangle (q : Quadrilateral) :
  is_parallelogram q → has_right_angle q → is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_diagonals_bisect_implies_parallelogram_parallelogram_right_angle_implies_rectangle_l1573_157387


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_equation_l1573_157358

theorem equal_roots_quadratic_equation (x m : ℝ) : 
  (∃ r : ℝ, ∀ x, x^2 - 5*x + m = 0 ↔ x = r) → m = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_equation_l1573_157358


namespace NUMINAMATH_CALUDE_sock_pairs_l1573_157364

/-- Given 3 pairs of socks, calculate the number of ways to choose 2 socks from different pairs -/
theorem sock_pairs (n : ℕ) (h : n = 3) : 
  (n * (n - 1)) / 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_l1573_157364


namespace NUMINAMATH_CALUDE_roses_picked_second_correct_l1573_157386

-- Define the problem parameters
def initial_roses : ℝ := 37.0
def first_picking : ℝ := 16.0
def final_total : ℕ := 72

-- Define the function to calculate roses picked in the second picking
def roses_picked_second (initial : ℝ) (first : ℝ) (total : ℕ) : ℝ :=
  (total : ℝ) - (initial + first)

-- Theorem statement
theorem roses_picked_second_correct :
  roses_picked_second initial_roses first_picking final_total = 19.0 := by
  sorry

end NUMINAMATH_CALUDE_roses_picked_second_correct_l1573_157386


namespace NUMINAMATH_CALUDE_min_value_of_F_l1573_157368

/-- The feasible region defined by the given constraints -/
def FeasibleRegion (x₁ x₂ : ℝ) : Prop :=
  2 - 2*x₁ - x₂ ≥ 0 ∧
  2 - x₁ + x₂ ≥ 0 ∧
  5 - x₁ - x₂ ≥ 0 ∧
  x₁ ≥ 0 ∧
  x₂ ≥ 0

/-- The objective function to be minimized -/
def F (x₁ x₂ : ℝ) : ℝ := x₂ - x₁

/-- Theorem stating that the minimum value of F in the feasible region is -2 -/
theorem min_value_of_F :
  ∀ x₁ x₂ : ℝ, FeasibleRegion x₁ x₂ → F x₁ x₂ ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_F_l1573_157368


namespace NUMINAMATH_CALUDE_max_omega_for_monotonic_sin_l1573_157385

/-- The maximum value of ω for which f(x) = sin(ωx) is monotonic on (-π/4, π/4) -/
theorem max_omega_for_monotonic_sin (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.sin (ω * x)) →
  ω > 0 →
  (∀ x y, -π/4 < x ∧ x < y ∧ y < π/4 → (f x < f y ∨ f x > f y)) →
  ω ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_omega_for_monotonic_sin_l1573_157385


namespace NUMINAMATH_CALUDE_chicken_ratio_is_two_to_one_l1573_157326

/-- The number of chickens in the coop -/
def chickens_in_coop : ℕ := 14

/-- The number of chickens free ranging -/
def chickens_free_ranging : ℕ := 52

/-- The number of chickens in the run -/
def chickens_in_run : ℕ := (chickens_free_ranging + 4) / 2

/-- The ratio of chickens in the run to chickens in the coop -/
def chicken_ratio : ℚ := chickens_in_run / chickens_in_coop

theorem chicken_ratio_is_two_to_one : chicken_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_chicken_ratio_is_two_to_one_l1573_157326


namespace NUMINAMATH_CALUDE_smallest_distance_between_points_on_circles_l1573_157382

theorem smallest_distance_between_points_on_circles (z w : ℂ) 
  (hz : Complex.abs (z - (2 - 4*Complex.I)) = 2)
  (hw : Complex.abs (w - (5 + 6*Complex.I)) = 4) :
  ∃ (m : ℝ), m = Real.sqrt 109 - 6 ∧ ∀ (z' w' : ℂ), 
    Complex.abs (z' - (2 - 4*Complex.I)) = 2 → 
    Complex.abs (w' - (5 + 6*Complex.I)) = 4 → 
    m ≤ Complex.abs (z' - w') :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_points_on_circles_l1573_157382


namespace NUMINAMATH_CALUDE_sequence_sum_l1573_157394

theorem sequence_sum (a b c d : ℕ+) : 
  (∃ r : ℚ, r > 1 ∧ b = a * r ∧ c = a * r^2) →  -- geometric progression
  (∃ k : ℤ, c - b = k ∧ d - c = k) →            -- arithmetic progression
  d = a + 40 →                                  -- difference between first and last term
  a + b + c + d = 104 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l1573_157394


namespace NUMINAMATH_CALUDE_local_extrema_of_f_l1573_157376

open Real

/-- The function f(x) = x^3 - 3x^2 - 9x -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

/-- The second derivative of f(x) -/
def f'' (x : ℝ) : ℝ := 6*x - 6

theorem local_extrema_of_f :
  ∃ (x : ℝ), x ∈ Set.Ioo (-2 : ℝ) 2 ∧
  IsLocalMax f x ∧
  f x = 5 ∧
  (∀ y ∈ Set.Ioo (-2 : ℝ) 2, ¬IsLocalMin f y) := by
  sorry

#check local_extrema_of_f

end NUMINAMATH_CALUDE_local_extrema_of_f_l1573_157376


namespace NUMINAMATH_CALUDE_square_roots_problem_l1573_157328

theorem square_roots_problem (m : ℝ) (a : ℝ) (h1 : m > 0) 
  (h2 : (a + 6)^2 = m) (h3 : (2*a - 9)^2 = m) :
  a = 1 ∧ m = 49 ∧ ∀ x : ℝ, a*x^2 - 16 = 0 ↔ x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1573_157328


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1573_157342

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) = q * a n) 
  (S : ℕ → ℝ) 
  (h_sum : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) 
  (h_eq1 : 2 * (a 6) = 3 * (S 4) + 1) 
  (h_eq2 : a 7 = 3 * (S 5) + 1) : 
  q = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1573_157342


namespace NUMINAMATH_CALUDE_percentage_difference_l1573_157355

theorem percentage_difference : 
  (67.5 / 100 * 250) - (52.3 / 100 * 180) = 74.61 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1573_157355


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_00625_l1573_157392

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_00625 :
  toScientificNotation 0.00625 = ScientificNotation.mk 6.25 (-3) sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_00625_l1573_157392


namespace NUMINAMATH_CALUDE_prize_logic_l1573_157307

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (answered_all_correctly : Student → Prop)
variable (got_prize : Student → Prop)

-- State the theorem
theorem prize_logic (h : ∀ s : Student, answered_all_correctly s → got_prize s) :
  ∀ s : Student, ¬(got_prize s) → ¬(answered_all_correctly s) :=
by
  sorry

end NUMINAMATH_CALUDE_prize_logic_l1573_157307


namespace NUMINAMATH_CALUDE_vector_at_negative_one_l1573_157384

/-- A line parameterized by t in 3D space -/
structure ParametricLine where
  point_at : ℝ → (ℝ × ℝ × ℝ)

/-- The vector at a given t value -/
def vector_at (line : ParametricLine) (t : ℝ) : (ℝ × ℝ × ℝ) :=
  line.point_at t

theorem vector_at_negative_one
  (line : ParametricLine)
  (h0 : vector_at line 0 = (2, 1, 5))
  (h1 : vector_at line 1 = (5, 0, 2)) :
  vector_at line (-1) = (-1, 2, 8) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_one_l1573_157384


namespace NUMINAMATH_CALUDE_y₁_greater_than_y₂_l1573_157349

/-- A linear function f(x) = 8x - 1 --/
def f (x : ℝ) : ℝ := 8 * x - 1

/-- The x-coordinate of P₁ --/
def x₁ : ℝ := 3

/-- The x-coordinate of P₂ --/
def x₂ : ℝ := 2

/-- The y-coordinate of P₁ --/
def y₁ : ℝ := f x₁

/-- The y-coordinate of P₂ --/
def y₂ : ℝ := f x₂

theorem y₁_greater_than_y₂ : y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_greater_than_y₂_l1573_157349


namespace NUMINAMATH_CALUDE_units_digit_of_23_power_23_l1573_157322

theorem units_digit_of_23_power_23 : (23^23) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_23_power_23_l1573_157322


namespace NUMINAMATH_CALUDE_power_eight_seven_thirds_l1573_157362

theorem power_eight_seven_thirds : (8 : ℝ) ^ (7/3) = 128 := by sorry

end NUMINAMATH_CALUDE_power_eight_seven_thirds_l1573_157362


namespace NUMINAMATH_CALUDE_quadratic_root_in_interval_l1573_157359

/-- A quadratic function f(x) = ax^2 + bx + c has a root in the interval (-2, 0),
    given that 2a + c/2 > b and c < 0. -/
theorem quadratic_root_in_interval (a b c : ℝ) (h1 : 2 * a + c / 2 > b) (h2 : c < 0) :
  ∃ x : ℝ, x ∈ Set.Ioo (-2 : ℝ) 0 ∧ a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_interval_l1573_157359


namespace NUMINAMATH_CALUDE_preceding_sum_40_times_l1573_157371

theorem preceding_sum_40_times (n : ℕ) : 
  (n ≠ 0) → ((n * (n - 1)) / 2 = 40 * n) → n = 81 := by
  sorry

end NUMINAMATH_CALUDE_preceding_sum_40_times_l1573_157371


namespace NUMINAMATH_CALUDE_triangle_with_altitudes_9_12_18_has_right_angle_l1573_157393

/-- A triangle with altitudes of lengths 9, 12, and 18 has a right angle as its largest angle. -/
theorem triangle_with_altitudes_9_12_18_has_right_angle :
  ∀ (a b c : ℝ) (α β γ : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c →
  α + β + γ = π →
  9 * a = 12 * b ∧ 12 * b = 18 * c →
  (∃ (h : ℝ), h * a = 9 ∧ h * b = 12 ∧ h * c = 18) →
  max α (max β γ) = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_altitudes_9_12_18_has_right_angle_l1573_157393


namespace NUMINAMATH_CALUDE_min_value_of_squares_l1573_157345

theorem min_value_of_squares (t : ℝ) :
  ∃ (a b : ℝ), 2 * a + 3 * b = t ∧
  ∀ (x y : ℝ), 2 * x + 3 * y = t → a^2 + b^2 ≤ x^2 + y^2 ∧
  a^2 + b^2 = (13 * t^2) / 169 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_squares_l1573_157345


namespace NUMINAMATH_CALUDE_exists_N_average_twelve_l1573_157321

theorem exists_N_average_twelve : ∃ N : ℝ, 11 < N ∧ N < 21 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_exists_N_average_twelve_l1573_157321


namespace NUMINAMATH_CALUDE_traci_flour_amount_l1573_157380

/-- The amount of flour Harris has in grams -/
def harris_flour : ℕ := 400

/-- The amount of flour needed for one cake in grams -/
def flour_per_cake : ℕ := 100

/-- The number of cakes Traci created -/
def traci_cakes : ℕ := 9

/-- The number of cakes Harris created -/
def harris_cakes : ℕ := 9

/-- The amount of flour Traci brought from her own house in grams -/
def traci_flour : ℕ := 1400

theorem traci_flour_amount :
  traci_flour = (flour_per_cake * (traci_cakes + harris_cakes)) - harris_flour :=
by sorry

end NUMINAMATH_CALUDE_traci_flour_amount_l1573_157380


namespace NUMINAMATH_CALUDE_grandma_olga_grandchildren_l1573_157367

-- Define the number of daughters and sons
def num_daughters : ℕ := 3
def num_sons : ℕ := 3

-- Define the number of children for each daughter and son
def sons_per_daughter : ℕ := 6
def daughters_per_son : ℕ := 5

-- Define the total number of grandchildren
def total_grandchildren : ℕ := num_daughters * sons_per_daughter + num_sons * daughters_per_son

-- Theorem statement
theorem grandma_olga_grandchildren : total_grandchildren = 33 := by
  sorry

end NUMINAMATH_CALUDE_grandma_olga_grandchildren_l1573_157367


namespace NUMINAMATH_CALUDE_basic_computer_price_l1573_157395

/-- Proves that the price of a basic computer is $1500 given certain conditions. -/
theorem basic_computer_price (basic_price printer_price : ℕ) : 
  (basic_price + printer_price = 2500) →
  (printer_price = (basic_price + 500 + printer_price) / 3) →
  basic_price = 1500 := by
  sorry

#check basic_computer_price

end NUMINAMATH_CALUDE_basic_computer_price_l1573_157395


namespace NUMINAMATH_CALUDE_marker_notebook_cost_l1573_157361

theorem marker_notebook_cost :
  ∀ (m n : ℕ),
  (10 * m + 5 * n = 120) →
  (m > n) →
  (m = 10 ∧ n = 4) →
  (m + n = 14) :=
by sorry

end NUMINAMATH_CALUDE_marker_notebook_cost_l1573_157361


namespace NUMINAMATH_CALUDE_concert_admission_revenue_l1573_157346

theorem concert_admission_revenue :
  let total_attendance : ℕ := 578
  let adult_price : ℚ := 2
  let child_price : ℚ := (3/2)
  let num_adults : ℕ := 342
  let num_children : ℕ := total_attendance - num_adults
  let total_revenue : ℚ := (num_adults : ℚ) * adult_price + (num_children : ℚ) * child_price
  total_revenue = 1038 :=
by sorry

end NUMINAMATH_CALUDE_concert_admission_revenue_l1573_157346


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1573_157398

/-- The ratio of the area to the square of the perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let perimeter : ℝ := 3 * side_length
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  area / (perimeter ^ 2) = Real.sqrt 3 / 36 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1573_157398


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l1573_157303

/-- Given a line with equation y = -2x - 3, prove that its symmetric line
    with respect to the y-axis has the equation y = 2x - 3 -/
theorem symmetric_line_wrt_y_axis (x y : ℝ) :
  (y = -2*x - 3) → (∃ (x' y' : ℝ), y' = 2*x' - 3 ∧ x' = -x ∧ y' = y) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l1573_157303


namespace NUMINAMATH_CALUDE_prob_win_match_value_l1573_157374

/-- Probability of player A winning a single game -/
def p : ℝ := 0.6

/-- Probability of player A winning the match in a best of 3 games -/
def prob_win_match : ℝ := p * p * (1 - p) + p * (1 - p) * p + p * p

/-- Theorem stating that the probability of player A winning the match is 0.648 -/
theorem prob_win_match_value : prob_win_match = 0.648 := by sorry

end NUMINAMATH_CALUDE_prob_win_match_value_l1573_157374


namespace NUMINAMATH_CALUDE_log_quarter_of_sixteen_eq_neg_two_l1573_157324

-- Define the logarithm function for base 1/4
noncomputable def log_quarter (x : ℝ) : ℝ := Real.log x / Real.log (1/4)

-- State the theorem
theorem log_quarter_of_sixteen_eq_neg_two :
  log_quarter 16 = -2 := by
  sorry

end NUMINAMATH_CALUDE_log_quarter_of_sixteen_eq_neg_two_l1573_157324


namespace NUMINAMATH_CALUDE_fraction_equality_l1573_157356

theorem fraction_equality (a b : ℝ) (h1 : b ≠ 0) (h2 : 2*a ≠ b) (h3 : a/b = 2/3) : 
  b/(2*a - b) = 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1573_157356


namespace NUMINAMATH_CALUDE_sin_18_degrees_l1573_157390

theorem sin_18_degrees : Real.sin (18 * π / 180) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_degrees_l1573_157390


namespace NUMINAMATH_CALUDE_outfits_count_l1573_157317

/-- The number of outfits with different colored shirts and hats -/
def num_outfits : ℕ :=
  let red_shirts := 5
  let green_shirts := 5
  let pants := 6
  let green_hats := 8
  let red_hats := 8
  (red_shirts * pants * green_hats) + (green_shirts * pants * red_hats)

theorem outfits_count : num_outfits = 480 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l1573_157317


namespace NUMINAMATH_CALUDE_hot_chocolate_servings_l1573_157399

/-- Represents the recipe requirements for 6 servings --/
structure Recipe :=
  (chocolate : ℚ)
  (sugar : ℚ)
  (milk : ℚ)
  (vanilla : ℚ)

/-- Represents the available ingredients --/
structure Available :=
  (chocolate : ℚ)
  (sugar : ℚ)
  (milk : ℚ)
  (vanilla : ℚ)

/-- Calculates the number of servings possible for a given ingredient --/
def servings_for_ingredient (required : ℚ) (available : ℚ) : ℚ :=
  (available / required) * 6

/-- Finds the minimum number of servings possible across all ingredients --/
def max_servings (recipe : Recipe) (available : Available) : ℚ :=
  min
    (servings_for_ingredient recipe.chocolate available.chocolate)
    (min
      (servings_for_ingredient recipe.sugar available.sugar)
      (min
        (servings_for_ingredient recipe.milk available.milk)
        (servings_for_ingredient recipe.vanilla available.vanilla)))

theorem hot_chocolate_servings
  (recipe : Recipe)
  (available : Available)
  (h_recipe : recipe = { chocolate := 3, sugar := 1/2, milk := 6, vanilla := 3/2 })
  (h_available : available = { chocolate := 8, sugar := 3, milk := 15, vanilla := 5 }) :
  max_servings recipe available = 15 := by
  sorry

end NUMINAMATH_CALUDE_hot_chocolate_servings_l1573_157399


namespace NUMINAMATH_CALUDE_max_daily_profit_l1573_157347

/-- The daily profit function for a store selling an item -/
noncomputable def daily_profit (x : ℕ) : ℝ :=
  if x ≥ 1 ∧ x ≤ 30 then
    -x^2 + 52*x + 620
  else if x ≥ 31 ∧ x ≤ 60 then
    -40*x + 2480
  else
    0

/-- The maximum daily profit and the day it occurs -/
theorem max_daily_profit :
  ∃ (max_profit : ℝ) (max_day : ℕ),
    max_profit = 1296 ∧
    max_day = 26 ∧
    (∀ x : ℕ, x ≥ 1 ∧ x ≤ 60 → daily_profit x ≤ max_profit) ∧
    daily_profit max_day = max_profit :=
by sorry

end NUMINAMATH_CALUDE_max_daily_profit_l1573_157347


namespace NUMINAMATH_CALUDE_line_equation_for_triangle_l1573_157305

/-- Given a line passing through (a, 0) that forms a triangle with area T' in the first quadrant,
    prove that its equation is 2T'x - a^2y + 2aT' = 0 --/
theorem line_equation_for_triangle (a T' : ℝ) (h_a : a > 0) (h_T' : T' > 0) :
  ∃ (x y : ℝ → ℝ), ∀ t : ℝ,
    (x t = a ∧ y t = 0) ∨
    (x t = 0 ∧ y t = 2 * T' / a) ∨
    (x t ≥ 0 ∧ y t ≥ 0 ∧ 2 * T' * x t - a^2 * y t + 2 * a * T' = 0) :=
sorry

end NUMINAMATH_CALUDE_line_equation_for_triangle_l1573_157305


namespace NUMINAMATH_CALUDE_correct_investment_structure_l1573_157323

/-- Represents the investment structure of a company --/
structure InvestmentStructure where
  initial_investors : ℕ
  initial_contribution : ℕ

/-- Checks if the investment structure satisfies the given conditions --/
def satisfies_conditions (s : InvestmentStructure) : Prop :=
  let contribution_1 := s.initial_contribution + 10000
  let contribution_2 := s.initial_contribution + 30000
  (s.initial_investors - 10) * contribution_1 = s.initial_investors * s.initial_contribution ∧
  (s.initial_investors - 25) * contribution_2 = s.initial_investors * s.initial_contribution

/-- Theorem stating the correct investment structure --/
theorem correct_investment_structure :
  ∃ (s : InvestmentStructure), s.initial_investors = 100 ∧ s.initial_contribution = 90000 ∧ satisfies_conditions s := by
  sorry

end NUMINAMATH_CALUDE_correct_investment_structure_l1573_157323


namespace NUMINAMATH_CALUDE_person_b_correct_probability_l1573_157378

theorem person_b_correct_probability 
  (prob_a_correct : ℝ) 
  (prob_b_correct_given_a_incorrect : ℝ) 
  (h1 : prob_a_correct = 0.4) 
  (h2 : prob_b_correct_given_a_incorrect = 0.5) : 
  (1 - prob_a_correct) * prob_b_correct_given_a_incorrect = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_person_b_correct_probability_l1573_157378


namespace NUMINAMATH_CALUDE_number_of_men_l1573_157344

theorem number_of_men (M W B : ℕ) (Ww Wb : ℚ) : 
  M * 6 = W * Ww ∧ 
  W * Ww = 7 * B * Wb ∧ 
  M * 6 + W * Ww + B * Wb = 90 →
  M = 5 := by
sorry

end NUMINAMATH_CALUDE_number_of_men_l1573_157344


namespace NUMINAMATH_CALUDE_r_daily_earning_l1573_157357

/-- Given the daily earnings of three individuals p, q, and r, prove that r earns 40 per day. -/
theorem r_daily_earning (p q r : ℕ) : 
  (9 * (p + q + r) = 1890) →
  (5 * (p + r) = 600) →
  (7 * (q + r) = 910) →
  r = 40 := by
  sorry

end NUMINAMATH_CALUDE_r_daily_earning_l1573_157357


namespace NUMINAMATH_CALUDE_roots_condition_l1573_157300

-- Define the quadratic function F(x)
def F (R l a x : ℝ) := 2 * R * x^2 - (l^2 + 4 * a * R) * x + 2 * R * a^2

-- Define the conditions for the roots to be between 0 and 2R
def roots_between_0_and_2R (R l a : ℝ) : Prop :=
  (0 < a ∧ a < 2 * R ∧ l^2 < (2 * R - a)^2) ∨
  (-2 * R < a ∧ a < 0 ∧ l^2 < (2 * R - a)^2)

-- Theorem statement
theorem roots_condition (R l a : ℝ) (hR : R > 0) (hl : l > 0) (ha : a ≠ 0) :
  (∀ x, F R l a x = 0 → 0 < x ∧ x < 2 * R) ↔ roots_between_0_and_2R R l a := by
  sorry

end NUMINAMATH_CALUDE_roots_condition_l1573_157300


namespace NUMINAMATH_CALUDE_reflect_point_D_l1573_157332

/-- Reflect a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflect a point across the line y = x - 1 -/
def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2 + 1, p.1 - 1)

/-- The main theorem stating that reflecting D(5,0) across y-axis and then y=x-1 results in (-1,4) -/
theorem reflect_point_D : 
  let D : ℝ × ℝ := (5, 0)
  let D' := reflect_y_axis D
  let D'' := reflect_line D'
  D'' = (-1, 4) := by sorry

end NUMINAMATH_CALUDE_reflect_point_D_l1573_157332


namespace NUMINAMATH_CALUDE_watch_cost_price_l1573_157310

theorem watch_cost_price (loss_price gain_price cost_price : ℝ) 
  (h1 : loss_price = 0.88 * cost_price)
  (h2 : gain_price = 1.08 * cost_price)
  (h3 : gain_price - loss_price = 350) : 
  cost_price = 1750 := by
sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1573_157310


namespace NUMINAMATH_CALUDE_probability_a_and_b_selected_l1573_157335

-- Define the total number of students
def total_students : ℕ := 5

-- Define the number of students to be selected
def selected_students : ℕ := 3

-- Define a function to calculate combinations
def combination (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Theorem statement
theorem probability_a_and_b_selected :
  (combination (total_students - 2) (selected_students - 2)) / 
  (combination total_students selected_students) = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_a_and_b_selected_l1573_157335


namespace NUMINAMATH_CALUDE_razorback_total_profit_l1573_157370

/-- The Razorback shop's sales during the Arkansas and Texas Tech game -/
def razorback_sales : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun tshirt_profit jersey_profit hat_profit keychain_profit
      tshirts_sold jerseys_sold hats_sold keychains_sold =>
    tshirt_profit * tshirts_sold +
    jersey_profit * jerseys_sold +
    hat_profit * hats_sold +
    keychain_profit * keychains_sold

theorem razorback_total_profit :
  razorback_sales 62 99 45 25 183 31 142 215 = 26180 := by
  sorry

end NUMINAMATH_CALUDE_razorback_total_profit_l1573_157370


namespace NUMINAMATH_CALUDE_complex_subtraction_l1573_157302

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 4 - I) :
  a - 2*b = -3 - I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1573_157302


namespace NUMINAMATH_CALUDE_cosine_sine_inequality_l1573_157348

theorem cosine_sine_inequality (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = π / 2) : 
  Real.cos a + Real.cos b + Real.cos c > Real.sin a + Real.sin b + Real.sin c := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_inequality_l1573_157348


namespace NUMINAMATH_CALUDE_article_price_l1573_157311

theorem article_price (profit_percentage : ℝ) (profit_amount : ℝ) (original_price : ℝ) : 
  profit_percentage = 40 →
  profit_amount = 560 →
  original_price * (1 + profit_percentage / 100) - original_price = profit_amount →
  original_price = 1400 := by
sorry

end NUMINAMATH_CALUDE_article_price_l1573_157311


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l1573_157340

theorem quadratic_complete_square (x : ℝ) : 
  (∃ (b c : ℤ), (x + b : ℝ)^2 = c ∧ x^2 - 10*x + 15 = 0) → 
  (∃ (b c : ℤ), (x + b : ℝ)^2 = c ∧ b + c = 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l1573_157340


namespace NUMINAMATH_CALUDE_egyptian_fraction_iff_prime_divisor_l1573_157341

theorem egyptian_fraction_iff_prime_divisor (n : ℕ) :
  (Odd n ∧ n > 0) →
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (4 : ℚ) / n = 1 / a + 1 / b) ↔
  ∃ (p : ℕ), Prime p ∧ p ∣ n ∧ p % 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_egyptian_fraction_iff_prime_divisor_l1573_157341


namespace NUMINAMATH_CALUDE_equation_solutions_l1573_157314

theorem equation_solutions :
  (∀ x : ℝ, 2 * x^2 + 1 = 3 * x ↔ x = 1 ∨ x = 1/2) ∧
  (∀ x : ℝ, (2*x - 1)^2 = (3 - x)^2 ↔ x = -2 ∨ x = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1573_157314


namespace NUMINAMATH_CALUDE_tangent_line_at_one_inequality_holds_l1573_157336

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2*a * Real.log x + (a-2)*x

-- Part 1: Tangent line equation when a = 1
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 4*x + 2*y - 3 = 0 :=
sorry

-- Part 2: Inequality holds for a ≤ -1/2
theorem inequality_holds (a : ℝ) (h : a ≤ -1/2) :
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
    (f a x₂ - f a x₁) / (x₂ - x₁) > a :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_inequality_holds_l1573_157336


namespace NUMINAMATH_CALUDE_sin_pi_6_minus_2alpha_l1573_157315

theorem sin_pi_6_minus_2alpha (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (Real.cos (α - π/6), 1/2))
  (hb : b = (1, -2 * Real.sin α))
  (hab : a.1 * b.1 + a.2 * b.2 = 1/3) :
  Real.sin (π/6 - 2*α) = -7/9 := by
sorry

end NUMINAMATH_CALUDE_sin_pi_6_minus_2alpha_l1573_157315


namespace NUMINAMATH_CALUDE_complex_power_sum_l1573_157389

theorem complex_power_sum (i : ℂ) (h1 : i^2 = -1) (h2 : i^4 = 1) :
  i^15 + i^22 + i^29 + i^36 + i^43 = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1573_157389


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l1573_157352

theorem min_value_expression (x : ℝ) : 
  (x^2 + 11) / Real.sqrt (x^2 + 5) ≥ 2 * Real.sqrt 6 := by
  sorry

theorem lower_bound_achievable : 
  ∃ x : ℝ, (x^2 + 11) / Real.sqrt (x^2 + 5) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l1573_157352


namespace NUMINAMATH_CALUDE_product_ABC_l1573_157343

theorem product_ABC (m : ℝ) : 
  let A := 4 * m
  let B := m - (1/4 : ℝ)
  let C := m + (1/4 : ℝ)
  A * B * C = 4 * m^3 - (1/4 : ℝ) * m :=
by sorry

end NUMINAMATH_CALUDE_product_ABC_l1573_157343


namespace NUMINAMATH_CALUDE_word_count_correct_l1573_157333

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the word -/
def word_length : ℕ := 5

/-- The number of positions that can vary (middle letters) -/
def varying_positions : ℕ := word_length - 2

/-- The number of five-letter words where the first and last letters are the same -/
def num_words : ℕ := alphabet_size * (alphabet_size ^ varying_positions)

theorem word_count_correct : num_words = 456976 := by sorry

end NUMINAMATH_CALUDE_word_count_correct_l1573_157333


namespace NUMINAMATH_CALUDE_intersection_x_coord_l1573_157350

/-- Two lines intersect at point (a, b) -/
def intersection_point (a b : ℝ) : Prop :=
  b = 13 ∧ b = 3 * a + 1

/-- Theorem: The x-coordinate of the intersection point is 4 -/
theorem intersection_x_coord :
  ∀ a b : ℝ, intersection_point a b → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coord_l1573_157350


namespace NUMINAMATH_CALUDE_custom_op_five_two_l1573_157331

-- Define the custom operation
def custom_op (a b : ℕ) : ℕ := 3*a + 4*b - a*b

-- State the theorem
theorem custom_op_five_two : custom_op 5 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_five_two_l1573_157331


namespace NUMINAMATH_CALUDE_min_value_problem_l1573_157306

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  ∃ (y : ℝ), y = (a + 1 / (2015 * a)) * (b + 1 / (2015 * b)) ∧
    y ≥ (2 * Real.sqrt 2016 - 2) / 2015 ∧
    (∀ (z : ℝ), z = (a + 1 / (2015 * a)) * (b + 1 / (2015 * b)) → z ≥ y) := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l1573_157306


namespace NUMINAMATH_CALUDE_sin_360_degrees_l1573_157383

theorem sin_360_degrees : Real.sin (2 * Real.pi) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_360_degrees_l1573_157383


namespace NUMINAMATH_CALUDE_jobber_pricing_jobber_pricing_example_l1573_157304

theorem jobber_pricing (original_price : ℝ) (purchase_discount : ℝ) (desired_gain : ℝ) (sale_discount : ℝ) : ℝ :=
  let purchase_price := original_price * (1 - purchase_discount)
  let selling_price := purchase_price * (1 + desired_gain)
  let marked_price := selling_price / (1 - sale_discount)
  marked_price

theorem jobber_pricing_example : jobber_pricing 24 0.125 (1/3) 0.2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_jobber_pricing_jobber_pricing_example_l1573_157304


namespace NUMINAMATH_CALUDE_xiaopang_mom_money_l1573_157379

/-- The price of apples per kilogram -/
def apple_price : ℝ := 5

/-- The amount of money Xiaopang's mom had -/
def total_money : ℝ := 21.5

/-- The amount of apples Xiaopang's mom wanted to buy initially -/
def initial_amount : ℝ := 5

/-- The amount of apples Xiaopang's mom actually bought -/
def actual_amount : ℝ := 4

/-- The amount of money Xiaopang's mom was short for the initial amount -/
def short_amount : ℝ := 3.5

/-- The amount of money Xiaopang's mom had left after buying the actual amount -/
def left_amount : ℝ := 1.5

theorem xiaopang_mom_money :
  total_money = actual_amount * apple_price + left_amount ∧
  total_money = initial_amount * apple_price - short_amount :=
by sorry

end NUMINAMATH_CALUDE_xiaopang_mom_money_l1573_157379


namespace NUMINAMATH_CALUDE_f_order_magnitude_l1573_157377

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x < y → f x < f y

-- State the theorem
theorem f_order_magnitude 
  (h1 : is_even f) 
  (h2 : is_increasing_on_nonneg f) : 
  f (-π) > f 3 ∧ f 3 > f (-2) :=
sorry

end NUMINAMATH_CALUDE_f_order_magnitude_l1573_157377


namespace NUMINAMATH_CALUDE_max_value_of_b_l1573_157320

theorem max_value_of_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1 / b - 1 / a) :
  b ≤ 1 / 3 ∧ ∃ (b₀ : ℝ), b₀ > 0 ∧ b₀ = 1 / 3 ∧ ∃ (a₀ : ℝ), a₀ > 0 ∧ a₀ + 3 * b₀ = 1 / b₀ - 1 / a₀ :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_b_l1573_157320
