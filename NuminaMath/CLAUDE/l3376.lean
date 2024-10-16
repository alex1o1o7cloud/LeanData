import Mathlib

namespace NUMINAMATH_CALUDE_solve_for_b_l3376_337629

/-- Given a system of equations and its solution, prove the value of b. -/
theorem solve_for_b (a : ℝ) : 
  (∃ x y : ℝ, a * x - 2 * y = 1 ∧ 2 * x + b * y = 5) →
  (∃ x y : ℝ, x = 1 ∧ y = a ∧ a * x - 2 * y = 1 ∧ 2 * x + b * y = 5) →
  b = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l3376_337629


namespace NUMINAMATH_CALUDE_train_car_speed_ratio_l3376_337677

/-- Given a bus that travels 320 km in 5 hours, and its speed is 4/5 of the train's speed,
    and a car that travels 525 km in 7 hours, prove that the ratio of the train's speed
    to the car's speed is 16:15 -/
theorem train_car_speed_ratio :
  ∀ (bus_speed train_speed car_speed : ℝ),
    bus_speed = 320 / 5 →
    bus_speed = (4 / 5) * train_speed →
    car_speed = 525 / 7 →
    train_speed / car_speed = 16 / 15 := by
  sorry

end NUMINAMATH_CALUDE_train_car_speed_ratio_l3376_337677


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l3376_337610

theorem perfect_square_quadratic (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 12*x + k = (x + a)^2) ↔ k = 36 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l3376_337610


namespace NUMINAMATH_CALUDE_remaining_flour_l3376_337695

def flour_needed (total_required : ℕ) (already_added : ℕ) : ℕ :=
  total_required - already_added

theorem remaining_flour :
  flour_needed 9 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_remaining_flour_l3376_337695


namespace NUMINAMATH_CALUDE_optimal_distribution_is_best_l3376_337601

/-- Represents the distribution of coins proposed by a logician -/
structure Distribution :=
  (logician1 : ℕ)
  (logician2 : ℕ)
  (logician3 : ℕ)

/-- The total number of coins to be distributed -/
def totalCoins : ℕ := 10

/-- The number of coins given to an eliminated logician -/
def eliminationCoins : ℕ := 2

/-- Checks if a distribution is valid (sums to the total number of coins) -/
def isValidDistribution (d : Distribution) : Prop :=
  d.logician1 + d.logician2 + d.logician3 = totalCoins

/-- Represents the approval of a distribution by a logician -/
def approves (logician : ℕ) (d : Distribution) : Prop :=
  match logician with
  | 1 => d.logician1 ≥ eliminationCoins
  | 2 => d.logician2 ≥ eliminationCoins
  | 3 => d.logician3 ≥ eliminationCoins
  | _ => False

/-- Checks if a distribution receives majority approval -/
def hasApproval (d : Distribution) : Prop :=
  (approves 1 d ∧ (approves 2 d ∨ approves 3 d)) ∨
  (approves 2 d ∧ approves 3 d)

/-- The optimal distribution strategy for Logician 1 -/
def optimalDistribution : Distribution :=
  { logician1 := 9, logician2 := 0, logician3 := 1 }

/-- Theorem stating that the optimal distribution is valid and maximizes Logician 1's gain -/
theorem optimal_distribution_is_best :
  isValidDistribution optimalDistribution ∧
  hasApproval optimalDistribution ∧
  ∀ d : Distribution,
    isValidDistribution d ∧ hasApproval d →
    d.logician1 ≤ optimalDistribution.logician1 :=
sorry


end NUMINAMATH_CALUDE_optimal_distribution_is_best_l3376_337601


namespace NUMINAMATH_CALUDE_total_waiting_time_bounds_l3376_337673

/-- 
Represents the total waiting time for a queue with Slowpokes and Quickies.
m: number of Slowpokes
n: number of Quickies
a: time taken by a Quickie
b: time taken by a Slowpoke
-/
def TotalWaitingTime (m n : ℕ) (a b : ℝ) : Prop :=
  let total := m + n
  ∀ (t_min t_max t_exp : ℝ),
    b > a →
    t_min = a * (n.choose 2) + a * m * n + b * (m.choose 2) →
    t_max = a * (n.choose 2) + b * m * n + b * (m.choose 2) →
    t_exp = (total.choose 2 : ℝ) * (b * m + a * n) / total →
    (t_min ≤ t_exp ∧ t_exp ≤ t_max)

theorem total_waiting_time_bounds {m n : ℕ} {a b : ℝ} :
  TotalWaitingTime m n a b :=
sorry

end NUMINAMATH_CALUDE_total_waiting_time_bounds_l3376_337673


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3376_337655

theorem max_value_of_expression (x y z : ℝ) 
  (h : 2 * x^2 + y^2 + z^2 = 2 * x - 4 * y + 2 * x * z - 5) : 
  ∃ (M : ℝ), M = 4 ∧ ∀ (a b c : ℝ), 2 * a^2 + b^2 + c^2 = 2 * a - 4 * b + 2 * a * c - 5 → 
  a - b + c ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3376_337655


namespace NUMINAMATH_CALUDE_petya_sum_theorem_l3376_337681

/-- Represents Petya's operation on the board numbers -/
def petyaOperation (x y z : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  (x, y, z - 1, x * y)

/-- Represents the invariant property throughout Petya's operations -/
def invariant (x y z : ℕ) (sum : ℕ) : Prop :=
  x * y * z = sum + x * y * z

/-- The main theorem stating that the sum of products on the paper
    equals the initial product of board numbers when process terminates -/
theorem petya_sum_theorem (x y z : ℕ) :
  ∃ (n : ℕ) (sum : ℕ),
    (∃ (a b : ℕ), a * b * 0 = n) ∧
    invariant x y z sum ∧
    sum = x * y * z := by
  sorry

end NUMINAMATH_CALUDE_petya_sum_theorem_l3376_337681


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3376_337634

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 2 = 1 →
  a 5 * a 6 = 4 →
  a 3 * a 4 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3376_337634


namespace NUMINAMATH_CALUDE_bird_stork_difference_l3376_337616

theorem bird_stork_difference (initial_birds storks additional_birds : ℕ) :
  initial_birds = 3 →
  storks = 4 →
  additional_birds = 2 →
  (initial_birds + additional_birds) - storks = 1 := by
  sorry

end NUMINAMATH_CALUDE_bird_stork_difference_l3376_337616


namespace NUMINAMATH_CALUDE_approximation_accuracy_l3376_337654

theorem approximation_accuracy : 
  abs (84 * Real.sqrt 7 - 222 * (2 / (2 + 7))) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_approximation_accuracy_l3376_337654


namespace NUMINAMATH_CALUDE_output_value_scientific_notation_l3376_337656

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem output_value_scientific_notation :
  toScientificNotation 110000000000 = ScientificNotation.mk 1.1 10 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_output_value_scientific_notation_l3376_337656


namespace NUMINAMATH_CALUDE_hakimi_age_l3376_337682

/-- Given three friends Hakimi, Jared, and Molly, this theorem proves Hakimi's age
    based on the given conditions. -/
theorem hakimi_age (hakimi_age jared_age molly_age : ℕ) : 
  (hakimi_age + jared_age + molly_age) / 3 = 40 →  -- Average age is 40
  jared_age = hakimi_age + 10 →  -- Jared is 10 years older than Hakimi
  molly_age = 30 →  -- Molly's age is 30
  hakimi_age = 40 :=  -- Hakimi's age is 40
by
  sorry

end NUMINAMATH_CALUDE_hakimi_age_l3376_337682


namespace NUMINAMATH_CALUDE_pump_fill_time_l3376_337619

/-- The time it takes to fill the tank with the leak present -/
def fill_time_with_leak : ℝ := 3

/-- The time it takes for the leak to drain the full tank -/
def leak_drain_time : ℝ := 5.999999999999999

/-- The time it takes for the pump to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 2

theorem pump_fill_time :
  (1 / fill_time_without_leak) - (1 / leak_drain_time) = (1 / fill_time_with_leak) :=
sorry

end NUMINAMATH_CALUDE_pump_fill_time_l3376_337619


namespace NUMINAMATH_CALUDE_diagonal_triangle_area_l3376_337657

/-- Represents a rectangular prism with given face areas -/
structure RectangularPrism where
  face_area_1 : ℝ
  face_area_2 : ℝ
  face_area_3 : ℝ

/-- Calculates the area of the triangle formed by the diagonals of the prism's faces -/
noncomputable def triangle_area (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem stating that for a rectangular prism with face areas 24, 30, and 32,
    the triangle formed by the diagonals of these faces has an area of 25 -/
theorem diagonal_triangle_area :
  let prism : RectangularPrism := ⟨24, 30, 32⟩
  triangle_area prism = 25 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_triangle_area_l3376_337657


namespace NUMINAMATH_CALUDE_three_consecutive_free_throws_l3376_337694

/-- The probability of scoring a single free throw -/
def free_throw_probability : ℝ := 0.7

/-- The number of consecutive free throws -/
def num_throws : ℕ := 3

/-- The probability of scoring in three consecutive free throws -/
def three_consecutive_probability : ℝ := free_throw_probability ^ num_throws

theorem three_consecutive_free_throws :
  three_consecutive_probability = 0.343 := by
  sorry

end NUMINAMATH_CALUDE_three_consecutive_free_throws_l3376_337694


namespace NUMINAMATH_CALUDE_triangle_properties_l3376_337622

noncomputable def angle_A (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C + (1/2) * c = b ∧ a = 1 → A = Real.pi / 3

def perimeter_range (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C + (1/2) * c = b ∧ a = 1 →
  let l := a + b + c
  2 < l ∧ l ≤ 3

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  angle_A A B C a b c ∧ perimeter_range A B C a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3376_337622


namespace NUMINAMATH_CALUDE_inverse_sum_equality_l3376_337607

-- Define the function g and its inverse
variable (g : ℝ → ℝ)
variable (g_inv : ℝ → ℝ)

-- Define the given conditions
axiom g_inverse : Function.LeftInverse g_inv g ∧ Function.RightInverse g_inv g
axiom g_4 : g 4 = 6
axiom g_6 : g 6 = 3
axiom g_7 : g 7 = 4

-- State the theorem
theorem inverse_sum_equality :
  g_inv (g_inv 4 + g_inv 6) = g_inv 11 :=
sorry

end NUMINAMATH_CALUDE_inverse_sum_equality_l3376_337607


namespace NUMINAMATH_CALUDE_rolling_semicircle_distance_l3376_337674

/-- The distance traveled by the center of a rolling semi-circle -/
theorem rolling_semicircle_distance (r : ℝ) (h : r = 4 / Real.pi) :
  2 * Real.pi * r / 2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_rolling_semicircle_distance_l3376_337674


namespace NUMINAMATH_CALUDE_largest_odd_in_sum_not_exceeding_200_l3376_337650

/-- The sum of the first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- The nth odd number -/
def nthOddNumber (n : ℕ) : ℕ := 2*n - 1

theorem largest_odd_in_sum_not_exceeding_200 :
  ∃ n : ℕ, sumOddNumbers n ≤ 200 ∧ 
           sumOddNumbers (n + 1) > 200 ∧ 
           nthOddNumber n = 27 :=
sorry

end NUMINAMATH_CALUDE_largest_odd_in_sum_not_exceeding_200_l3376_337650


namespace NUMINAMATH_CALUDE_sampling_suitability_l3376_337604

/-- Represents a sampling scenario --/
structure SamplingScenario where
  population : ℕ
  sample_size : ℕ
  (valid : sample_size ≤ population)

/-- Determines if a sampling scenario is suitable for simple random sampling --/
def suitable_for_simple_random_sampling (scenario : SamplingScenario) : Prop :=
  scenario.sample_size ≤ 10 ∧ scenario.population ≤ 100

/-- Determines if a sampling scenario is suitable for systematic sampling --/
def suitable_for_systematic_sampling (scenario : SamplingScenario) : Prop :=
  scenario.sample_size > 10 ∧ scenario.population > 100

/-- The first sampling scenario --/
def scenario1 : SamplingScenario where
  population := 10
  sample_size := 2
  valid := by norm_num

/-- The second sampling scenario --/
def scenario2 : SamplingScenario where
  population := 1000
  sample_size := 50
  valid := by norm_num

/-- Theorem stating that the first scenario is suitable for simple random sampling
    and the second scenario is suitable for systematic sampling --/
theorem sampling_suitability :
  suitable_for_simple_random_sampling scenario1 ∧
  suitable_for_systematic_sampling scenario2 := by
  sorry


end NUMINAMATH_CALUDE_sampling_suitability_l3376_337604


namespace NUMINAMATH_CALUDE_latus_rectum_for_parabola_l3376_337617

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = -1/6 * x^2

/-- The equation of the latus rectum -/
def latus_rectum_equation (y : ℝ) : Prop := y = 3/2

/-- Theorem: The latus rectum equation for the given parabola -/
theorem latus_rectum_for_parabola :
  ∀ x y : ℝ, parabola_equation x y → latus_rectum_equation y :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_for_parabola_l3376_337617


namespace NUMINAMATH_CALUDE_fraction_equality_l3376_337640

-- Define the @ operation
def at_op (a b : ℚ) : ℚ := a * b - b^2

-- Define the # operation
def hash_op (a b : ℚ) : ℚ := a + b - 2 * a * b^2

-- Theorem statement
theorem fraction_equality : (at_op 8 3) / (hash_op 8 3) = -15 / 133 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3376_337640


namespace NUMINAMATH_CALUDE_tan_pi_over_3n_irrational_l3376_337693

theorem tan_pi_over_3n_irrational (n : ℕ) (hn : n > 0) : 
  Irrational (Real.tan (π / (3 * ↑n))) :=
sorry

end NUMINAMATH_CALUDE_tan_pi_over_3n_irrational_l3376_337693


namespace NUMINAMATH_CALUDE_diamond_calculation_l3376_337625

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement
theorem diamond_calculation :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29/132 :=
by sorry

end NUMINAMATH_CALUDE_diamond_calculation_l3376_337625


namespace NUMINAMATH_CALUDE_second_candidate_percentage_l3376_337672

/-- Represents an exam with total marks and passing marks. -/
structure Exam where
  totalMarks : ℝ
  passingMarks : ℝ

/-- Represents a candidate's performance in the exam. -/
structure Candidate where
  marksObtained : ℝ

def Exam.firstCandidateCondition (e : Exam) : Prop :=
  0.3 * e.totalMarks = e.passingMarks - 50

def Exam.secondCandidateCondition (e : Exam) (c : Candidate) : Prop :=
  c.marksObtained = e.passingMarks + 25

/-- The theorem stating the percentage of marks obtained by the second candidate. -/
theorem second_candidate_percentage (e : Exam) (c : Candidate) :
  e.passingMarks = 199.99999999999997 →
  e.firstCandidateCondition →
  e.secondCandidateCondition c →
  c.marksObtained / e.totalMarks = 0.45 := by
  sorry


end NUMINAMATH_CALUDE_second_candidate_percentage_l3376_337672


namespace NUMINAMATH_CALUDE_factorial_equation_unique_solution_l3376_337614

theorem factorial_equation_unique_solution :
  ∃! m : ℕ+, (Nat.factorial 6) * (Nat.factorial 11) = 18 * (Nat.factorial m.val) * 2 :=
by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_unique_solution_l3376_337614


namespace NUMINAMATH_CALUDE_x_one_value_l3376_337685

theorem x_one_value (x₁ x₂ x₃ : Real) 
  (h1 : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 0.8)
  (h2 : (1-x₁)^2 + (x₁-x₂)^2 + (x₂-x₃)^2 + x₃^2 = 1/3) :
  x₁ = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_x_one_value_l3376_337685


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l3376_337652

theorem largest_three_digit_congruence :
  ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    40 * n ≡ 140 [MOD 320] ∧
    ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 40 * m ≡ 140 [MOD 320]) → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l3376_337652


namespace NUMINAMATH_CALUDE_no_square_base_l3376_337653

theorem no_square_base (b : ℕ) (h : b > 0) : ¬∃ (n : ℕ), b^2 + 3*b + 2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_square_base_l3376_337653


namespace NUMINAMATH_CALUDE_linear_function_inequality_l3376_337645

theorem linear_function_inequality (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = a * x + b) →
  (∀ x, f (f x) ≥ x - 3) ↔
  ((a = -1 ∧ b ∈ Set.univ) ∨ (a = 1 ∧ b ≥ -3/2)) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_inequality_l3376_337645


namespace NUMINAMATH_CALUDE_slope_angle_vertical_line_l3376_337690

-- Define a vertical line
def vertical_line (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = a}

-- Define the slope angle of a line
def slope_angle (L : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem slope_angle_vertical_line :
  slope_angle (vertical_line 2) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_vertical_line_l3376_337690


namespace NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l3376_337670

/-- Calculates the amount John paid out of pocket for a new computer and accessories,
    given the costs and the sale of his PlayStation. -/
theorem johns_out_of_pocket_expense (computer_cost accessories_cost playstation_value : ℕ)
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : playstation_value = 400) :
  computer_cost + accessories_cost - (playstation_value * 80 / 100) = 580 := by
  sorry

#check johns_out_of_pocket_expense

end NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l3376_337670


namespace NUMINAMATH_CALUDE_max_n_for_positive_an_l3376_337636

theorem max_n_for_positive_an (n : ℕ) : 
  (∀ k : ℕ, k > n → (19 : ℤ) - 2 * k ≤ 0) ∧ 
  ((19 : ℤ) - 2 * n > 0) → 
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_max_n_for_positive_an_l3376_337636


namespace NUMINAMATH_CALUDE_consecutive_composite_sequence_l3376_337648

theorem consecutive_composite_sequence (n : ℕ) : ∃ r : ℕ, ∀ k ∈ Finset.range n, ¬(Nat.Prime (r + k + 1)) :=
sorry

end NUMINAMATH_CALUDE_consecutive_composite_sequence_l3376_337648


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l3376_337663

theorem absolute_value_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 5*x*y) :
  |((x+y)/(x-y))| = Real.sqrt (7/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l3376_337663


namespace NUMINAMATH_CALUDE_class_fund_problem_l3376_337679

theorem class_fund_problem (m : ℕ) (x : ℕ) (y : ℚ) :
  m < 400 →
  38 ≤ x →
  x < 50 →
  x * y = m →
  (x + 12) * (y - 2) = m →
  x = 42 ∧ y = 9 := by
  sorry

end NUMINAMATH_CALUDE_class_fund_problem_l3376_337679


namespace NUMINAMATH_CALUDE_farm_animals_difference_l3376_337671

theorem farm_animals_difference (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 5 * initial_cows →
  (initial_horses - 15) / (initial_cows + 15) = 17 / 7 →
  (initial_horses - 15) - (initial_cows + 15) = 50 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_difference_l3376_337671


namespace NUMINAMATH_CALUDE_jerry_candy_boxes_l3376_337644

theorem jerry_candy_boxes (initial boxes_sold boxes_left : ℕ) :
  boxes_sold = 5 →
  boxes_left = 5 →
  initial = boxes_sold + boxes_left →
  initial = 10 :=
by sorry

end NUMINAMATH_CALUDE_jerry_candy_boxes_l3376_337644


namespace NUMINAMATH_CALUDE_pricing_equation_l3376_337647

/-- 
Given an item with:
- cost price x (in yuan)
- markup percentage m (as a decimal)
- discount percentage d (as a decimal)
- final selling price s (in yuan)

This theorem states that the equation relating these values is:
x * (1 + m) * (1 - d) = s
-/
theorem pricing_equation (x m d s : ℝ) 
  (markup : m = 0.3)
  (discount : d = 0.2)
  (selling_price : s = 2080) :
  x * (1 + m) * (1 - d) = s :=
sorry

end NUMINAMATH_CALUDE_pricing_equation_l3376_337647


namespace NUMINAMATH_CALUDE_pizza_slices_left_l3376_337669

theorem pizza_slices_left (total_slices : ℕ) (fraction_eaten : ℚ) : 
  total_slices = 16 → fraction_eaten = 3/4 → total_slices - (total_slices * fraction_eaten).floor = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l3376_337669


namespace NUMINAMATH_CALUDE_multiple_solutions_exist_four_wheelers_not_unique_l3376_337680

/-- Represents the number of wheels on a vehicle -/
inductive WheelCount
  | two
  | four

/-- Represents the parking lot with 2 wheelers and 4 wheelers -/
structure ParkingLot where
  twoWheelers : ℕ
  fourWheelers : ℕ

/-- Calculates the total number of wheels in the parking lot -/
def totalWheels (lot : ParkingLot) : ℕ :=
  2 * lot.twoWheelers + 4 * lot.fourWheelers

/-- Theorem stating that multiple solutions exist for a given total wheel count -/
theorem multiple_solutions_exist (totalWheelCount : ℕ) :
  ∃ (lot1 lot2 : ParkingLot), lot1 ≠ lot2 ∧ totalWheels lot1 = totalWheelCount ∧ totalWheels lot2 = totalWheelCount :=
sorry

/-- Theorem stating that the number of 4 wheelers cannot be uniquely determined -/
theorem four_wheelers_not_unique (totalWheelCount : ℕ) :
  ¬∃! (fourWheelerCount : ℕ), ∃ (twoWheelerCount : ℕ), totalWheels {twoWheelers := twoWheelerCount, fourWheelers := fourWheelerCount} = totalWheelCount :=
sorry

end NUMINAMATH_CALUDE_multiple_solutions_exist_four_wheelers_not_unique_l3376_337680


namespace NUMINAMATH_CALUDE_gasoline_needed_for_distance_l3376_337635

/-- Given a car with fuel efficiency and a known fuel consumption for a specific distance,
    calculate the amount of gasoline needed for any distance. -/
theorem gasoline_needed_for_distance (fuel_efficiency : ℝ) (known_distance : ℝ) (known_gasoline : ℝ) (distance : ℝ) :
  fuel_efficiency = 20 →
  known_distance = 130 →
  known_gasoline = 6.5 →
  known_distance / known_gasoline = fuel_efficiency →
  distance / fuel_efficiency = distance / 20 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_needed_for_distance_l3376_337635


namespace NUMINAMATH_CALUDE_student_average_score_l3376_337627

theorem student_average_score (M P C : ℕ) : 
  M + P = 50 → C = P + 20 → (M + C) / 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_student_average_score_l3376_337627


namespace NUMINAMATH_CALUDE_sarah_initial_followers_l3376_337624

/-- Represents the number of followers gained by Sarah in a week -/
structure WeeklyGain where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the data for a student's social media followers -/
structure StudentData where
  school_size : ℕ
  initial_followers : ℕ
  weekly_gain : WeeklyGain

theorem sarah_initial_followers (susy sarah : StudentData) 
  (h1 : susy.school_size = 800)
  (h2 : sarah.school_size = 300)
  (h3 : susy.initial_followers = 100)
  (h4 : sarah.weekly_gain.first = 90)
  (h5 : sarah.weekly_gain.second = sarah.weekly_gain.first / 3)
  (h6 : sarah.weekly_gain.third = sarah.weekly_gain.second / 3)
  (h7 : max (susy.initial_followers + susy.weekly_gain.first + susy.weekly_gain.second + susy.weekly_gain.third)
            (sarah.initial_followers + sarah.weekly_gain.first + sarah.weekly_gain.second + sarah.weekly_gain.third) = 180) :
  sarah.initial_followers = 50 := by
  sorry


end NUMINAMATH_CALUDE_sarah_initial_followers_l3376_337624


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l3376_337662

noncomputable section

-- Define the * operation
def ast (x₁ x₂ : ℝ) : ℝ := (x₁ + x₂)^2 - (x₁ - x₂)^2

-- Define the point P
def P (x a : ℝ) : ℝ × ℝ := (x, Real.sqrt (ast x a))

-- Theorem statement
theorem trajectory_is_parabola (a : ℝ) (h₁ : a > 0) :
  ∃ k c : ℝ, ∀ x : ℝ, x ≥ 0 → (P x a).2^2 = k * (P x a).1 + c :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l3376_337662


namespace NUMINAMATH_CALUDE_linear_system_solution_l3376_337667

-- Define the determinant function
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the system of linear equations
def system (x y : ℝ) : Prop := 2 * x + y = 1 ∧ 3 * x - 2 * y = 12

-- State the theorem
theorem linear_system_solution :
  let D := det2x2 2 1 3 (-2)
  let Dx := det2x2 1 1 12 (-2)
  let Dy := det2x2 2 1 3 12
  D = -7 ∧ Dx = -14 ∧ Dy = 21 ∧ system (Dx / D) (Dy / D) ∧ system 2 (-3) := by
  sorry


end NUMINAMATH_CALUDE_linear_system_solution_l3376_337667


namespace NUMINAMATH_CALUDE_expression_equals_101_15_closest_integer_is_6_l3376_337675

-- Define the expression
def expression : ℚ := (4 * 10^150 + 4 * 10^152) / (3 * 10^151 + 3 * 10^151)

-- Theorem stating that the expression equals 101/15
theorem expression_equals_101_15 : expression = 101 / 15 := by sorry

-- Function to find the closest integer to a rational number
def closest_integer (q : ℚ) : ℤ := 
  ⌊q + 1/2⌋

-- Theorem stating that the closest integer to the expression is 6
theorem closest_integer_is_6 : closest_integer expression = 6 := by sorry

end NUMINAMATH_CALUDE_expression_equals_101_15_closest_integer_is_6_l3376_337675


namespace NUMINAMATH_CALUDE_new_shape_perimeter_l3376_337612

/-- The perimeter of the new shape formed by cutting an isosceles triangle from a square and reattaching it outside --/
theorem new_shape_perimeter (square_perimeter : ℝ) (triangle_side : ℝ) : 
  square_perimeter = 64 →
  triangle_side = square_perimeter / 4 →
  square_perimeter / 4 + square_perimeter / 4 + square_perimeter / 4 + square_perimeter / 4 + square_perimeter / 4 = 80 :=
by sorry

end NUMINAMATH_CALUDE_new_shape_perimeter_l3376_337612


namespace NUMINAMATH_CALUDE_sum_of_fractions_simplest_form_l3376_337608

theorem sum_of_fractions_simplest_form : 
  (6 : ℚ) / 7 + (7 : ℚ) / 9 = (103 : ℚ) / 63 ∧ 
  ∀ n d : ℤ, (n : ℚ) / d = (103 : ℚ) / 63 → (n.gcd d = 1 → n = 103 ∧ d = 63) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_simplest_form_l3376_337608


namespace NUMINAMATH_CALUDE_arccos_cos_three_l3376_337606

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_three_l3376_337606


namespace NUMINAMATH_CALUDE_fourth_month_sales_l3376_337661

def sales_problem (m1 m2 m3 m5 m6 average : ℕ) : Prop :=
  ∃ m4 : ℕ, (m1 + m2 + m3 + m4 + m5 + m6) / 6 = average

theorem fourth_month_sales :
  sales_problem 6435 6927 6855 6562 7391 6900 →
  ∃ m4 : ℕ, m4 = 7230 ∧ (6435 + 6927 + 6855 + m4 + 6562 + 7391) / 6 = 6900 :=
by sorry

end NUMINAMATH_CALUDE_fourth_month_sales_l3376_337661


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3376_337676

/-- The common ratio of the infinite geometric series 8/10 - 6/15 + 36/225 - ... is -1/2 -/
theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 8 / 10
  let a₂ : ℚ := -6 / 15
  let a₃ : ℚ := 36 / 225
  ∃ r : ℚ, r = a₂ / a₁ ∧ r = a₃ / a₂ ∧ r = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3376_337676


namespace NUMINAMATH_CALUDE_third_largest_three_digit_with_eight_ones_l3376_337639

/-- Given a list of digits, returns all three-digit numbers that can be formed using exactly three of those digits. -/
def threeDigitNumbers (digits : List Nat) : List Nat := sorry

/-- Checks if a number has 8 in the ones place. -/
def hasEightInOnes (n : Nat) : Bool := sorry

/-- The third largest element in a list of natural numbers. -/
def thirdLargest (numbers : List Nat) : Nat := sorry

theorem third_largest_three_digit_with_eight_ones : 
  let digits := [0, 1, 4, 8]
  let validNumbers := (threeDigitNumbers digits).filter hasEightInOnes
  thirdLargest validNumbers = 148 := by sorry

end NUMINAMATH_CALUDE_third_largest_three_digit_with_eight_ones_l3376_337639


namespace NUMINAMATH_CALUDE_chipped_marbles_possibilities_l3376_337632

def marble_counts : List Nat := [15, 18, 20, 22, 24, 27, 30, 32, 35, 37]

def total_marbles : Nat := marble_counts.sum

theorem chipped_marbles_possibilities :
  ∀ n : Nat, n ∈ marble_counts →
  (total_marbles - n) % 5 = 0 →
  n % 5 = 0 →
  n ∈ [15, 20, 30, 35] :=
by sorry

end NUMINAMATH_CALUDE_chipped_marbles_possibilities_l3376_337632


namespace NUMINAMATH_CALUDE_min_value_theorem_l3376_337666

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (heq : a * b = 1 / 2) :
  (4 * a^2 + b^2 + 1) / (2 * a - b) ≥ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3376_337666


namespace NUMINAMATH_CALUDE_intersection_with_complement_l3376_337687

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_with_complement : P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l3376_337687


namespace NUMINAMATH_CALUDE_smallest_possible_b_l3376_337628

theorem smallest_possible_b (a b : ℝ) 
  (h1 : 2 < a ∧ a < b) 
  (h2 : 2 + a ≤ b) 
  (h3 : 1/a + 1/b ≤ 1/2) : 
  b ≥ 3 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l3376_337628


namespace NUMINAMATH_CALUDE_side_b_value_triangle_area_l3376_337698

-- Define the triangle ABC
def triangle_ABC (A B C : Real) (a b c : Real) : Prop :=
  -- Add conditions here
  a = 3 ∧ 
  Real.cos A = Real.sqrt 6 / 3 ∧
  B = A + Real.pi / 2

-- Theorem for the value of side b
theorem side_b_value (A B C : Real) (a b c : Real) 
  (h : triangle_ABC A B C a b c) : b = 3 * Real.sqrt 2 := by
  sorry

-- Theorem for the area of triangle ABC
theorem triangle_area (A B C : Real) (a b c : Real) 
  (h : triangle_ABC A B C a b c) : 
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_side_b_value_triangle_area_l3376_337698


namespace NUMINAMATH_CALUDE_root_values_l3376_337697

theorem root_values (a b c d k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * k^4 + b * k^3 + c * k^2 + d * k + a = 0)
  (h2 : a * k^3 + b * k^2 + c * k + d = 0) :
  k = Complex.I^(1/4) ∨ k = -Complex.I^(1/4) ∨ k = Complex.I^(3/4) ∨ k = -Complex.I^(3/4) :=
sorry

end NUMINAMATH_CALUDE_root_values_l3376_337697


namespace NUMINAMATH_CALUDE_louise_oranges_l3376_337665

theorem louise_oranges (num_boxes : ℕ) (oranges_per_box : ℕ) 
  (h1 : num_boxes = 7) 
  (h2 : oranges_per_box = 6) : 
  num_boxes * oranges_per_box = 42 := by
  sorry

end NUMINAMATH_CALUDE_louise_oranges_l3376_337665


namespace NUMINAMATH_CALUDE_minimum_red_marbles_l3376_337611

theorem minimum_red_marbles (r w g : ℕ) : 
  g ≥ (2 * w) / 3 →
  g ≤ r / 4 →
  w + g ≥ 72 →
  (∀ r' : ℕ, (∃ w' g' : ℕ, g' ≥ (2 * w') / 3 ∧ g' ≤ r' / 4 ∧ w' + g' ≥ 72) → r' ≥ r) →
  r = 120 := by
sorry

end NUMINAMATH_CALUDE_minimum_red_marbles_l3376_337611


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l3376_337686

-- Define the lines and circle
def line_l2 (x y : ℝ) : Prop := x + 3*y + 1 = 0
def line_l1 (x y : ℝ) : Prop := 3*x - y = 0
def circle_C (x y a : ℝ) : Prop := x^2 + y^2 - 2*a*x - 2*a*y = 1 - 2*a^2

-- Define the theorem
theorem circle_center_coordinates (a : ℝ) :
  a > 0 →
  (∃ M N : ℝ × ℝ, line_l1 M.1 M.2 ∧ line_l1 N.1 N.2 ∧ circle_C M.1 M.2 a ∧ circle_C N.1 N.2 a) →
  (∀ x y : ℝ, line_l2 x y → (∀ u v : ℝ, line_l1 u v → u*x + v*y = 0)) →
  (∃ C : ℝ × ℝ, C.1 = a ∧ C.2 = a ∧ circle_C C.1 C.2 a) →
  (∃ M N : ℝ × ℝ, line_l1 M.1 M.2 ∧ line_l1 N.1 N.2 ∧ circle_C M.1 M.2 a ∧ circle_C N.1 N.2 a ∧
    (M.1 - a) * (N.1 - a) + (M.2 - a) * (N.2 - a) = 0) →
  a = Real.sqrt 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l3376_337686


namespace NUMINAMATH_CALUDE_complex_real_condition_l3376_337683

theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := (m - 2 : ℂ) + (m^2 - 3*m + 2 : ℂ) * Complex.I
  (z ≠ 0 ∧ z.im = 0) → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3376_337683


namespace NUMINAMATH_CALUDE_translation_of_quadratic_l3376_337649

/-- The original quadratic function -/
def g (x : ℝ) : ℝ := -2 * x^2

/-- The translated quadratic function -/
def f (x : ℝ) : ℝ := -2 * x^2 - 12 * x - 16

/-- The vertex of the original function g -/
def vertex_g : ℝ × ℝ := (0, 0)

/-- The vertex of the translated function f -/
def vertex_f : ℝ × ℝ := (-3, 2)

/-- Theorem stating that f is the translation of g -/
theorem translation_of_quadratic :
  ∀ x : ℝ, f x = g (x + 3) + 2 :=
sorry

end NUMINAMATH_CALUDE_translation_of_quadratic_l3376_337649


namespace NUMINAMATH_CALUDE_candy_mixture_cost_l3376_337633

/-- Given a mixture of two types of candy, prove the cost of the second type. -/
theorem candy_mixture_cost
  (total_mixture : ℝ)
  (first_candy_weight : ℝ)
  (first_candy_cost : ℝ)
  (second_candy_weight : ℝ)
  (mixture_cost : ℝ)
  (h1 : total_mixture = first_candy_weight + second_candy_weight)
  (h2 : total_mixture = 45)
  (h3 : first_candy_weight = 15)
  (h4 : first_candy_cost = 8)
  (h5 : second_candy_weight = 30)
  (h6 : mixture_cost = 6) :
  ∃ (second_candy_cost : ℝ),
    second_candy_cost = 5 ∧
    total_mixture * mixture_cost =
      first_candy_weight * first_candy_cost +
      second_candy_weight * second_candy_cost :=
by sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_l3376_337633


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3376_337618

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + a*x + 1 ≥ 0) → a ∈ Set.Ioi 2 ∪ Set.Iio (-2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3376_337618


namespace NUMINAMATH_CALUDE_complete_square_with_integer_l3376_337658

theorem complete_square_with_integer (x : ℝ) : 
  ∃ (k : ℤ) (a : ℝ), x^2 - 6*x + 20 = (x - a)^2 + k ∧ k = 11 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_with_integer_l3376_337658


namespace NUMINAMATH_CALUDE_whiteboard_washing_l3376_337623

theorem whiteboard_washing (kids : ℕ) (whiteboards : ℕ) (time : ℕ) :
  kids = 4 →
  whiteboards = 3 →
  time = 20 →
  (1 : ℝ) * 160 * whiteboards = kids * time * 6 :=
by sorry

end NUMINAMATH_CALUDE_whiteboard_washing_l3376_337623


namespace NUMINAMATH_CALUDE_green_beads_count_l3376_337696

/-- The number of green beads initially in a container -/
def initial_green_beads (total : ℕ) (brown red taken left : ℕ) : ℕ :=
  total - brown - red

/-- Theorem stating the number of green beads initially in the container -/
theorem green_beads_count (brown red taken left : ℕ) 
  (h1 : brown = 2)
  (h2 : red = 3)
  (h3 : taken = 2)
  (h4 : left = 4) :
  initial_green_beads (taken + left) brown red taken left = 1 := by
  sorry

#check green_beads_count

end NUMINAMATH_CALUDE_green_beads_count_l3376_337696


namespace NUMINAMATH_CALUDE_seven_mondays_in_45_days_l3376_337651

/-- The number of Mondays in the first 45 days of a year that starts on a Monday. -/
def mondaysIn45Days (yearStartsOnMonday : Bool) : ℕ :=
  if yearStartsOnMonday then 7 else 0

/-- Theorem stating that if a year starts on a Monday, there are 7 Mondays in the first 45 days. -/
theorem seven_mondays_in_45_days (yearStartsOnMonday : Bool) 
  (h : yearStartsOnMonday = true) : mondaysIn45Days yearStartsOnMonday = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_mondays_in_45_days_l3376_337651


namespace NUMINAMATH_CALUDE_mrs_hilt_fountain_trips_l3376_337638

/-- Calculates the total distance walked to a water fountain given the one-way distance and number of trips -/
def total_distance_walked (one_way_distance : ℕ) (num_trips : ℕ) : ℕ :=
  2 * one_way_distance * num_trips

/-- Proves that given a distance of 30 feet from desk to fountain and 4 trips to the fountain, the total distance walked is 240 feet -/
theorem mrs_hilt_fountain_trips :
  total_distance_walked 30 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_fountain_trips_l3376_337638


namespace NUMINAMATH_CALUDE_kia_vehicles_count_l3376_337646

theorem kia_vehicles_count (total : ℕ) (dodge : ℕ) (hyundai : ℕ) (kia : ℕ) : 
  total = 400 →
  dodge = total / 2 →
  hyundai = dodge / 2 →
  kia = total - (dodge + hyundai) →
  kia = 100 := by
  sorry

end NUMINAMATH_CALUDE_kia_vehicles_count_l3376_337646


namespace NUMINAMATH_CALUDE_mark_takes_tablets_for_12_hours_l3376_337609

/-- Represents the number of hours Mark takes Tylenol tablets -/
def hours_taking_tablets (tablets_per_dose : ℕ) (mg_per_tablet : ℕ) (hours_between_doses : ℕ) (total_grams : ℕ) : ℕ :=
  (total_grams * 1000) / (tablets_per_dose * mg_per_tablet) * hours_between_doses

/-- Theorem stating that Mark takes the tablets for 12 hours -/
theorem mark_takes_tablets_for_12_hours :
  hours_taking_tablets 2 500 4 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mark_takes_tablets_for_12_hours_l3376_337609


namespace NUMINAMATH_CALUDE_rectangle_breadth_l3376_337642

/-- 
Given a rectangle where:
1. The area is 24 times its breadth
2. The difference between the length and the breadth is 10 meters
Prove that the breadth is 14 meters
-/
theorem rectangle_breadth (length breadth : ℝ) 
  (h1 : length * breadth = 24 * breadth) 
  (h2 : length - breadth = 10) : 
  breadth = 14 := by
sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l3376_337642


namespace NUMINAMATH_CALUDE_intersection_point_l3376_337678

/-- The quadratic function f(x) = x^2 - 5x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 5*x + 1

/-- The y-axis is the set of points with x-coordinate 0 -/
def yAxis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}

theorem intersection_point :
  (0, 1) ∈ yAxis ∧ f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l3376_337678


namespace NUMINAMATH_CALUDE_product_inequality_l3376_337688

theorem product_inequality (a b c d : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d)
  (hab : a + b = 2) (hcd : c + d = 2) : 
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3376_337688


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3376_337602

theorem max_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → -1/(2*x) - 2/y ≤ -1/(2*a) - 2/b) ∧
  (-1/(2*a) - 2/b = -9/2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3376_337602


namespace NUMINAMATH_CALUDE_specific_structure_surface_area_l3376_337643

/-- Represents a complex structure composed of unit cubes -/
structure CubeStructure where
  num_cubes : ℕ
  height : ℕ
  length : ℕ
  width : ℕ

/-- Calculates the surface area of a cube structure -/
def surface_area (s : CubeStructure) : ℕ :=
  2 * (s.length * s.width + s.length * s.height + s.width * s.height)

/-- Theorem stating that a specific cube structure has a surface area of 84 square units -/
theorem specific_structure_surface_area :
  ∃ (s : CubeStructure), s.num_cubes = 15 ∧ s.height = 4 ∧ s.length = 5 ∧ s.width = 3 ∧
  surface_area s = 84 :=
sorry

end NUMINAMATH_CALUDE_specific_structure_surface_area_l3376_337643


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_pyramid_l3376_337600

theorem lateral_surface_area_of_pyramid (sin_alpha : ℝ) (diagonal_section_area : ℝ) :
  sin_alpha = 15 / 17 →
  diagonal_section_area = 3 * Real.sqrt 34 →
  (4 * diagonal_section_area) / (2 * Real.sqrt ((1 + (-Real.sqrt (1 - sin_alpha^2))) / 2)) = 68 :=
by sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_pyramid_l3376_337600


namespace NUMINAMATH_CALUDE_one_valid_placement_l3376_337615

/-- Represents the number of pegs of each color -/
structure PegCounts where
  purple : Nat
  yellow : Nat
  red : Nat
  green : Nat
  blue : Nat

/-- Represents a hexagonal peg board -/
structure HexBoard where
  rows : Nat
  columns : Nat

/-- Counts the number of valid peg placements -/
def countValidPlacements (board : HexBoard) (pegs : PegCounts) : Nat :=
  sorry

/-- Theorem stating that there is exactly one valid placement -/
theorem one_valid_placement (board : HexBoard) (pegs : PegCounts) : 
  board.rows = 6 ∧ board.columns = 6 ∧ 
  pegs.purple = 6 ∧ pegs.yellow = 5 ∧ pegs.red = 4 ∧ pegs.green = 3 ∧ pegs.blue = 2 →
  countValidPlacements board pegs = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_valid_placement_l3376_337615


namespace NUMINAMATH_CALUDE_system_solution_l3376_337664

theorem system_solution (x y z : ℚ) : 
  (1/x + 1/y = 6) ∧ (1/y + 1/z = 4) ∧ (1/z + 1/x = 5) → 
  (x = 2/7) ∧ (y = 2/5) ∧ (z = 2/3) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3376_337664


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3376_337660

open Real

theorem trigonometric_identity (α : ℝ) 
  (h1 : sin α < 0) 
  (h2 : cos α < 0) 
  (h3 : cos (75 * π / 180 + α) = 1/3) : 
  cos (105 * π / 180 - α) + sin (α - 105 * π / 180) = (2 * sqrt 2 - 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3376_337660


namespace NUMINAMATH_CALUDE_one_pole_inside_l3376_337641

/-- Represents a non-convex polygon fence -/
structure Fence where
  is_non_convex : Bool

/-- Represents a power line with poles -/
structure PowerLine where
  total_poles : Nat

/-- Represents a spy walking around the fence -/
structure Spy where
  counted_poles : Nat

/-- Theorem stating that given the conditions, there is one pole inside the fence -/
theorem one_pole_inside (fence : Fence) (power_line : PowerLine) (spy : Spy) :
  fence.is_non_convex ∧
  power_line.total_poles = 36 ∧
  spy.counted_poles = 2015 →
  ∃ (poles_inside : Nat), poles_inside = 1 :=
sorry

end NUMINAMATH_CALUDE_one_pole_inside_l3376_337641


namespace NUMINAMATH_CALUDE_decreasing_functions_a_range_l3376_337684

/-- Given two functions f and g, prove that if they are both decreasing on [1,2],
    then the parameter a is in the interval (0,1]. -/
theorem decreasing_functions_a_range 
  (f g : ℝ → ℝ) 
  (hf : f = fun x ↦ -x^2 + 2*a*x) 
  (hg : g = fun x ↦ a / (x + 1)) 
  (hf_decreasing : ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x < y → f x > f y) 
  (hg_decreasing : ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x < y → g x > g y) 
  : a ∈ Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_functions_a_range_l3376_337684


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3376_337699

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 6 → a^2 > 36) ∧ (∃ a, a^2 > 36 ∧ a ≤ 6) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3376_337699


namespace NUMINAMATH_CALUDE_series_sum_l3376_337620

/-- The sum of the infinite series ∑(n=1 to ∞) (5n-1)/(3^n) is equal to 13/6 -/
theorem series_sum : (∑' n : ℕ, (5 * n - 1 : ℝ) / (3 : ℝ) ^ n) = 13 / 6 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l3376_337620


namespace NUMINAMATH_CALUDE_garden_perimeter_l3376_337605

/-- The total perimeter of a rectangular garden with an attached triangular flower bed -/
theorem garden_perimeter (garden_length garden_width triangle_height : ℝ) 
  (hl : garden_length = 15)
  (hw : garden_width = 10)
  (ht : triangle_height = 6) :
  2 * (garden_length + garden_width) + 
  (Real.sqrt (garden_length^2 + triangle_height^2) + triangle_height) - 
  garden_length = 41 + Real.sqrt 261 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l3376_337605


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l3376_337613

theorem ice_cream_sundaes (total_flavors : ℕ) (h_total : total_flavors = 8) :
  let required_flavor := 1
  let sundae_size := 2
  let max_sundaes := total_flavors - required_flavor
  max_sundaes = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l3376_337613


namespace NUMINAMATH_CALUDE_odd_function_property_l3376_337621

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : f 3 - f 2 = 1) :
  f (-2) - f (-3) = 1 := by sorry

end NUMINAMATH_CALUDE_odd_function_property_l3376_337621


namespace NUMINAMATH_CALUDE_correct_algorithm_structures_l3376_337689

-- Define the possible algorithm structures
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop
  | Flow
  | Nested

-- Define a function that checks if a list of structures is correct
def isCorrectStructureList (list : List AlgorithmStructure) : Prop :=
  list = [AlgorithmStructure.Sequential, AlgorithmStructure.Conditional, AlgorithmStructure.Loop]

-- State the theorem
theorem correct_algorithm_structures :
  isCorrectStructureList [AlgorithmStructure.Sequential, AlgorithmStructure.Conditional, AlgorithmStructure.Loop] :=
by sorry


end NUMINAMATH_CALUDE_correct_algorithm_structures_l3376_337689


namespace NUMINAMATH_CALUDE_range_of_m_l3376_337626

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 4 ∧ x^2 - 2*x + 5 - m < 0) → m > 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3376_337626


namespace NUMINAMATH_CALUDE_existence_of_twin_prime_divisors_l3376_337630

theorem existence_of_twin_prime_divisors :
  ∃ (n : ℕ) (p₁ p₂ : ℕ), 
    Odd n ∧ 
    0 < n ∧
    Prime p₁ ∧ 
    Prime p₂ ∧ 
    (2^n - 1) % p₁ = 0 ∧ 
    (2^n - 1) % p₂ = 0 ∧ 
    p₁ - p₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_twin_prime_divisors_l3376_337630


namespace NUMINAMATH_CALUDE_infinitely_many_odd_floor_squares_l3376_337637

theorem infinitely_many_odd_floor_squares (α : ℝ) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, Odd ⌊n^2 * α⌋ :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_odd_floor_squares_l3376_337637


namespace NUMINAMATH_CALUDE_first_competitor_distance_l3376_337691

/-- The long jump competition with four competitors -/
structure LongJumpCompetition where
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ

/-- The conditions of the long jump competition -/
def validCompetition (c : LongJumpCompetition) : Prop :=
  c.second = c.first + 1 ∧
  c.third = c.second - 2 ∧
  c.fourth = c.third + 3 ∧
  c.fourth = 24

/-- Theorem: In a valid long jump competition, the first competitor jumped 22 feet -/
theorem first_competitor_distance (c : LongJumpCompetition) 
  (h : validCompetition c) : c.first = 22 := by
  sorry

#check first_competitor_distance

end NUMINAMATH_CALUDE_first_competitor_distance_l3376_337691


namespace NUMINAMATH_CALUDE_even_mono_increasing_range_l3376_337659

/-- A function f is even if f(x) = f(-x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f is monotonically increasing on [0, +∞) if
    for all x ≥ 0 and y ≥ 0, x < y implies f(x) < f(y) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x < f y

/-- The main theorem -/
theorem even_mono_increasing_range 
  (f : ℝ → ℝ) (h1 : EvenFunction f) (h2 : MonoIncreasing f) :
  {x : ℝ | f (1 - x) < f 2} = Set.Ioo (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_even_mono_increasing_range_l3376_337659


namespace NUMINAMATH_CALUDE_mary_peter_ratio_l3376_337631

/-- Represents the amount of chestnuts picked by each person in kilograms -/
structure ChestnutPickers where
  mary : ℝ
  peter : ℝ
  lucy : ℝ

/-- The conditions of the chestnut picking problem -/
def chestnut_problem (c : ChestnutPickers) : Prop :=
  c.mary = 12 ∧
  c.lucy = c.peter + 2 ∧
  c.mary + c.peter + c.lucy = 26

/-- The theorem stating the ratio of Mary's chestnuts to Peter's is 2:1 -/
theorem mary_peter_ratio (c : ChestnutPickers) :
  chestnut_problem c → c.mary / c.peter = 2 := by
  sorry

#check mary_peter_ratio

end NUMINAMATH_CALUDE_mary_peter_ratio_l3376_337631


namespace NUMINAMATH_CALUDE_parkway_soccer_players_l3376_337692

theorem parkway_soccer_players (total_students : ℕ) (boys : ℕ) (girls_not_playing : ℕ) 
  (h1 : total_students = 420)
  (h2 : boys = 312)
  (h3 : girls_not_playing = 63)
  (h4 : (82 : ℚ) / 100 * (total_students - (total_students - boys - girls_not_playing)) = boys - (total_students - boys - girls_not_playing)) :
  total_students - (total_students - boys - girls_not_playing) = 250 := by
  sorry

end NUMINAMATH_CALUDE_parkway_soccer_players_l3376_337692


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l3376_337668

theorem smallest_square_containing_circle (r : ℝ) (h : r = 4) :
  (2 * r) ^ 2 = 64 := by sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l3376_337668


namespace NUMINAMATH_CALUDE_chord_equation_l3376_337603

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 144

-- Define point P
def P : ℝ × ℝ := (3, 2)

-- Define a chord passing through P with P as its midpoint
def is_chord_midpoint (x y : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    ellipse x1 y1 ∧ ellipse x2 y2 ∧
    (x1 + x2) / 2 = P.1 ∧ (y1 + y2) / 2 = P.2 ∧
    x = (x2 - x1) ∧ y = (y2 - y1)

-- Theorem statement
theorem chord_equation :
  ∀ (x y : ℝ), is_chord_midpoint x y → 2 * x + 3 * y = 12 :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l3376_337603
