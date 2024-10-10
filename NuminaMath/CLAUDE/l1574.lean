import Mathlib

namespace matrix_power_not_identity_l1574_157429

/-- Given a 5x5 complex matrix A with trace 0 and invertible I₅ - A, A⁵ ≠ I₅ -/
theorem matrix_power_not_identity
  (A : Matrix (Fin 5) (Fin 5) ℂ)
  (h_trace : Matrix.trace A = 0)
  (h_invertible : IsUnit (1 - A)) :
  A ^ 5 ≠ 1 := by
  sorry

end matrix_power_not_identity_l1574_157429


namespace game_result_l1574_157489

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 ∧ n % 3 = 0 then 6
  else if n % 3 = 0 then 3
  else if n % 2 = 0 then 2
  else 1

def allie_rolls : List ℕ := [5, 6, 1, 2, 3]
def betty_rolls : List ℕ := [6, 1, 1, 2, 3]

def calculate_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result : 
  calculate_points allie_rolls * calculate_points betty_rolls = 169 := by
  sorry

end game_result_l1574_157489


namespace subset_condition_implies_a_range_l1574_157401

theorem subset_condition_implies_a_range (a : ℝ) : 
  (Finset.powerset {2 * a, a^2 - a}).card = 4 → a ≠ 0 ∧ a ≠ 3 := by
  sorry

end subset_condition_implies_a_range_l1574_157401


namespace larger_square_perimeter_l1574_157468

theorem larger_square_perimeter
  (small_square_perimeter : ℝ)
  (shaded_area : ℝ)
  (h1 : small_square_perimeter = 72)
  (h2 : shaded_area = 160) :
  let small_side := small_square_perimeter / 4
  let small_area := small_side ^ 2
  let large_area := small_area + shaded_area
  let large_side := Real.sqrt large_area
  let large_perimeter := 4 * large_side
  large_perimeter = 88 := by
sorry

end larger_square_perimeter_l1574_157468


namespace complex_arithmetic_evaluation_l1574_157488

theorem complex_arithmetic_evaluation :
  6 - 5 * (7 - (Real.sqrt 16 + 2)^2) * 3 = -429 := by
  sorry

end complex_arithmetic_evaluation_l1574_157488


namespace problem_1_2_l1574_157416

theorem problem_1_2 :
  (2 * Real.sqrt 6 + 2 / 3) * Real.sqrt 3 - Real.sqrt 32 = 2 * Real.sqrt 2 + 2 * Real.sqrt 3 / 3 ∧
  (Real.sqrt 7 + 2) * (Real.sqrt 7 - 2) - (Real.sqrt 45 + Real.sqrt 20) / Real.sqrt 5 = -10 :=
by sorry

end problem_1_2_l1574_157416


namespace square_circle_perimeter_equality_l1574_157453

theorem square_circle_perimeter_equality (x : ℝ) :
  (4 * x = 2 * π * 5) → x = (5 * π) / 2 := by
  sorry

end square_circle_perimeter_equality_l1574_157453


namespace elevator_exit_probability_l1574_157470

/-- The number of floors where people can exit the elevator -/
def num_floors : ℕ := 9

/-- The probability that two people exit the elevator on different floors -/
def prob_different_floors : ℚ := 8 / 9

theorem elevator_exit_probability :
  (num_floors : ℚ) * (num_floors - 1) / (num_floors * num_floors) = prob_different_floors := by
  sorry

end elevator_exit_probability_l1574_157470


namespace triangle_ratio_proof_l1574_157456

theorem triangle_ratio_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  b^2 = a * c →
  a^2 + b * c = c^2 + a * c →
  c / (b * Real.sin B) = 2 * Real.sqrt 3 / 3 :=
by sorry

end triangle_ratio_proof_l1574_157456


namespace jessica_bank_balance_l1574_157486

theorem jessica_bank_balance (B : ℝ) : 
  B > 0 → 
  200 = (2/5) * B → 
  let remaining := B - 200
  let deposit := (1/5) * remaining
  remaining + deposit = 360 := by
sorry

end jessica_bank_balance_l1574_157486


namespace largest_non_attainable_sum_l1574_157412

/-- The set of coin denominations in Limonia -/
def coin_denominations (n : ℕ) : Set ℕ :=
  {3*n - 1, 6*n + 1, 6*n + 4, 6*n + 7}

/-- A sum is attainable if it can be formed using the given coin denominations -/
def is_attainable (n : ℕ) (sum : ℕ) : Prop :=
  ∃ (a b c d : ℕ), sum = a*(3*n - 1) + b*(6*n + 1) + c*(6*n + 4) + d*(6*n + 7)

/-- The largest non-attainable sum in Limonia -/
def largest_non_attainable (n : ℕ) : ℕ := 6*n^2 + 4*n - 5

/-- Theorem: The largest non-attainable sum in Limonia is 6n^2 + 4n - 5 -/
theorem largest_non_attainable_sum (n : ℕ) :
  (∀ k > largest_non_attainable n, is_attainable n k) ∧
  ¬(is_attainable n (largest_non_attainable n)) := by
  sorry

end largest_non_attainable_sum_l1574_157412


namespace number_base_conversion_l1574_157458

theorem number_base_conversion :
  ∃! (x y z b : ℕ),
    (x * b^2 + y * b + z = 1989) ∧
    (b^2 ≤ 1989) ∧
    (1989 < b^3) ∧
    (x + y + z = 27) ∧
    (0 ≤ x) ∧ (x < b) ∧
    (0 ≤ y) ∧ (y < b) ∧
    (0 ≤ z) ∧ (z < b) ∧
    (x = 5 ∧ y = 9 ∧ z = 13 ∧ b = 19) := by
  sorry

end number_base_conversion_l1574_157458


namespace cereal_eating_time_l1574_157421

/-- The time it takes for Mr. Fat and Mr. Thin to eat 5 pounds of cereal together -/
theorem cereal_eating_time (fat_rate thin_rate : ℚ) (total_cereal : ℚ) : 
  fat_rate = 1 / 15 →
  thin_rate = 1 / 45 →
  total_cereal = 5 →
  (total_cereal / (fat_rate + thin_rate) : ℚ) = 56.25 := by
  sorry

end cereal_eating_time_l1574_157421


namespace wrapper_cap_difference_l1574_157442

/-- Represents Danny's collection of bottle caps and wrappers -/
structure Collection where
  caps : ℕ
  wrappers : ℕ

/-- The number of bottle caps and wrappers Danny found at the park -/
def park_find : Collection :=
  { caps := 15, wrappers := 18 }

/-- Danny's current collection -/
def current_collection : Collection :=
  { caps := 35, wrappers := 67 }

/-- The theorem stating the difference between wrappers and bottle caps in Danny's collection -/
theorem wrapper_cap_difference :
  current_collection.wrappers - current_collection.caps = 32 :=
by sorry

end wrapper_cap_difference_l1574_157442


namespace volleyball_basketball_soccer_arrangement_l1574_157484

def num_stadiums : ℕ := 4
def num_competitions : ℕ := 3

def total_arrangements : ℕ := num_stadiums ^ num_competitions

def arrangements_all_same : ℕ := num_stadiums

theorem volleyball_basketball_soccer_arrangement :
  total_arrangements - arrangements_all_same = 60 :=
by sorry

end volleyball_basketball_soccer_arrangement_l1574_157484


namespace juan_has_64_marbles_l1574_157482

/-- The number of marbles Connie has -/
def connie_marbles : ℕ := 39

/-- The number of additional marbles Juan has compared to Connie -/
def juan_extra_marbles : ℕ := 25

/-- The total number of marbles Juan has -/
def juan_marbles : ℕ := connie_marbles + juan_extra_marbles

theorem juan_has_64_marbles : juan_marbles = 64 := by
  sorry

end juan_has_64_marbles_l1574_157482


namespace hyperbola_vertex_distance_l1574_157474

/-- The distance between the vertices of the hyperbola x^2/144 - y^2/49 = 1 is 24 -/
theorem hyperbola_vertex_distance :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2/144 - y^2/49 = 1}
  ∃ (v1 v2 : ℝ × ℝ), v1 ∈ hyperbola ∧ v2 ∈ hyperbola ∧ ‖v1 - v2‖ = 24 :=
by sorry

end hyperbola_vertex_distance_l1574_157474


namespace fraction_unchanged_l1574_157471

theorem fraction_unchanged (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x) / (2 * (x + y)) = x / (x + y) := by
  sorry

end fraction_unchanged_l1574_157471


namespace part_one_part_two_l1574_157499

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem part_one (a : ℝ) (h : a ≤ 2) :
  {x : ℝ | f a x ≥ 2} = {x : ℝ | x ≤ 1/2 ∨ x ≥ 5/2} := by sorry

-- Part 2
theorem part_two :
  {a : ℝ | a > 1 ∧ ∀ x, f a x + |x - 1| ≥ 1} = {a : ℝ | a ≥ 2} := by sorry

end part_one_part_two_l1574_157499


namespace radio_loss_percentage_l1574_157441

/-- Given the cost price and selling price of a radio, prove the loss percentage. -/
theorem radio_loss_percentage
  (cost_price : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 2400)
  (h2 : selling_price = 2100) :
  (cost_price - selling_price) / cost_price * 100 = 12.5 := by
  sorry

end radio_loss_percentage_l1574_157441


namespace max_value_of_a_l1574_157406

theorem max_value_of_a (a b c : ℝ) (sum_eq : a + b + c = 3) (prod_sum_eq : a * b + a * c + b * c = 3) :
  a ≤ 1 + Real.sqrt 2 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 3 ∧ a₀ * b₀ + a₀ * c₀ + b₀ * c₀ = 3 ∧ a₀ = 1 + Real.sqrt 2 :=
by sorry

end max_value_of_a_l1574_157406


namespace roots_quadratic_equation_l1574_157419

theorem roots_quadratic_equation (m p q c : ℝ) (a b : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + c/b)^2 - p*(a + c/b) + q = 0) →
  ((b + c/a)^2 - p*(b + c/a) + q = 0) →
  (q = 3 + 2*c + c^2/3) :=
by sorry

end roots_quadratic_equation_l1574_157419


namespace sample_is_extracurricular_homework_l1574_157444

/-- Represents a student in the survey -/
structure Student where
  id : Nat
  hasExtracurricularHomework : Bool

/-- Represents the survey conducted by the middle school -/
structure Survey where
  totalPopulation : Finset Student
  selectedSample : Finset Student
  sampleSize : Nat

/-- Definition of a valid survey -/
def validSurvey (s : Survey) : Prop :=
  s.totalPopulation.card = 1800 ∧
  s.selectedSample.card = 300 ∧
  s.selectedSample ⊆ s.totalPopulation ∧
  s.sampleSize = s.selectedSample.card

/-- Definition of the sample in the survey -/
def sampleDefinition (s : Survey) : Finset Student :=
  s.selectedSample.filter (λ student => student.hasExtracurricularHomework)

/-- Theorem stating that the sample is the extracurricular homework of 300 students -/
theorem sample_is_extracurricular_homework (s : Survey) (h : validSurvey s) :
  sampleDefinition s = s.selectedSample :=
sorry


end sample_is_extracurricular_homework_l1574_157444


namespace max_min_difference_c_l1574_157466

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 6) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 18) : 
  ∃ (c_max c_min : ℝ), 
    (∀ x : ℝ, (∃ y z : ℝ, x + y + z = 6 ∧ x^2 + y^2 + z^2 = 18) → c_min ≤ x ∧ x ≤ c_max) ∧
    c_max - c_min = 4 :=
sorry

end max_min_difference_c_l1574_157466


namespace arithmetic_sequence_a1_l1574_157463

/-- An arithmetic sequence with specific conditions -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  (∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 3 = -6 ∧
  a 7 = a 5 + 4

/-- Theorem stating that under given conditions, a_1 = -10 -/
theorem arithmetic_sequence_a1 (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = -10 := by
  sorry

end arithmetic_sequence_a1_l1574_157463


namespace quadratic_function_range_l1574_157420

/-- Given a quadratic function f(x) = x^2 + ax + b, where a and b are real numbers,
    and sets A and B defined as follows:
    A = { x ∈ ℝ | f(x) ≤ 0 }
    B = { x ∈ ℝ | f(f(x)) ≤ 3 }
    If A = B ≠ ∅, then the range of a is [2√3, 6). -/
theorem quadratic_function_range (a b : ℝ) :
  let f := fun x : ℝ => x^2 + a*x + b
  let A := {x : ℝ | f x ≤ 0}
  let B := {x : ℝ | f (f x) ≤ 3}
  A = B ∧ A.Nonempty → a ∈ Set.Icc (2 * Real.sqrt 3) 6 := by
  sorry

end quadratic_function_range_l1574_157420


namespace retailer_profit_percent_l1574_157494

/-- Calculates the profit percent for a retailer given the purchase price, overhead expenses, and selling price. -/
theorem retailer_profit_percent
  (purchase_price : ℚ)
  (overhead_expenses : ℚ)
  (selling_price : ℚ)
  (h1 : purchase_price = 225)
  (h2 : overhead_expenses = 15)
  (h3 : selling_price = 300) :
  (selling_price - (purchase_price + overhead_expenses)) / (purchase_price + overhead_expenses) * 100 = 25 := by
  sorry

end retailer_profit_percent_l1574_157494


namespace angle_sum_equality_counterexample_l1574_157411

theorem angle_sum_equality_counterexample :
  ∃ (angle1 angle2 : ℝ), 
    angle1 + angle2 = 90 ∧ angle1 = angle2 :=
by sorry

end angle_sum_equality_counterexample_l1574_157411


namespace budgets_equal_in_1996_l1574_157480

/-- Represents the year when the budgets of projects Q and V are equal -/
def year_budgets_equal (initial_q initial_v increase_q decrease_v : ℕ) : ℕ :=
  let n := (initial_v - initial_q) / (increase_q + decrease_v)
  1990 + n

/-- Theorem stating that the budgets of projects Q and V are equal in 1996 -/
theorem budgets_equal_in_1996 :
  year_budgets_equal 540000 780000 30000 10000 = 1996 := by
  sorry

#eval year_budgets_equal 540000 780000 30000 10000

end budgets_equal_in_1996_l1574_157480


namespace inverse_proportion_m_range_l1574_157461

/-- Given an inverse proportion function y = (1-m)/x passing through points (1, y₁) and (2, y₂),
    where y₁ > y₂, prove that m < 1 -/
theorem inverse_proportion_m_range (y₁ y₂ m : ℝ) : 
  y₁ = 1 - m → 
  y₂ = (1 - m) / 2 → 
  y₁ > y₂ → 
  m < 1 := by
  sorry

end inverse_proportion_m_range_l1574_157461


namespace quadratic_perfect_square_l1574_157490

theorem quadratic_perfect_square (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 50*x + c = (x + a)^2) → c = 625 := by
  sorry

end quadratic_perfect_square_l1574_157490


namespace shaded_area_calculation_l1574_157457

theorem shaded_area_calculation (r : Real) (h : r = 1) : 
  6 * (π * r^2) + 4 * (1/2 * π * r^2) = 8 * π := by
  sorry

end shaded_area_calculation_l1574_157457


namespace baron_weights_partition_l1574_157440

/-- A set of weights satisfying the Baron's conditions -/
def BaronWeights : Type := 
  { s : Finset ℕ // s.card = 50 ∧ ∀ x ∈ s, x ≤ 100 ∧ Even (s.sum id) }

/-- The proposition that the weights can be partitioned into two subsets with equal sums -/
def CanPartition (weights : BaronWeights) : Prop :=
  ∃ (s₁ s₂ : Finset ℕ), s₁ ∪ s₂ = weights.val ∧ s₁ ∩ s₂ = ∅ ∧ s₁.sum id = s₂.sum id

/-- The theorem stating that any set of weights satisfying the Baron's conditions can be partitioned -/
theorem baron_weights_partition (weights : BaronWeights) : CanPartition weights := by
  sorry


end baron_weights_partition_l1574_157440


namespace odd_primes_with_eight_factors_l1574_157432

theorem odd_primes_with_eight_factors (w y : ℕ) : 
  Nat.Prime w → 
  Nat.Prime y → 
  w < y → 
  Odd w → 
  Odd y → 
  (Finset.card (Nat.divisors (2 * w * y)) = 8) → 
  w = 3 := by
sorry

end odd_primes_with_eight_factors_l1574_157432


namespace mara_bags_count_l1574_157446

/-- Prove that Mara has 12 bags given the conditions of the marble problem -/
theorem mara_bags_count : ∀ (x : ℕ), 
  (x * 2 + 2 = 2 * 13) → x = 12 := by
  sorry

end mara_bags_count_l1574_157446


namespace sqrt_of_sqrt_81_l1574_157433

theorem sqrt_of_sqrt_81 : ∃ (x : ℝ), x^2 = 81 ∧ (x = 3 ∨ x = -3) := by
  sorry

end sqrt_of_sqrt_81_l1574_157433


namespace equation_solution_l1574_157426

theorem equation_solution : ∃ x : ℚ, (1 / 4 + 8 / x = 13 / x + 1 / 8) ∧ x = 40 := by
  sorry

end equation_solution_l1574_157426


namespace probability_at_least_two_same_l1574_157427

theorem probability_at_least_two_same (n : Nat) (s : Nat) :
  n = 8 →
  s = 8 →
  (1 - (Nat.factorial n) / (s^n : ℚ)) = 415 / 416 := by
  sorry

end probability_at_least_two_same_l1574_157427


namespace sequence_differences_l1574_157476

def a (n : ℕ) : ℕ := n^2 + 1

def first_difference (n : ℕ) : ℕ := a (n + 1) - a n

def second_difference (n : ℕ) : ℕ := first_difference (n + 1) - first_difference n

def third_difference (n : ℕ) : ℕ := second_difference (n + 1) - second_difference n

theorem sequence_differences :
  (∀ n : ℕ, first_difference n = 2*n + 1) ∧
  (∀ n : ℕ, second_difference n = 2) ∧
  (∀ n : ℕ, third_difference n = 0) := by
  sorry

end sequence_differences_l1574_157476


namespace parallelogram_bisector_slope_l1574_157497

/-- A parallelogram with given vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- A line passing through the origin -/
structure Line where
  slope : ℝ

/-- Predicate to check if a line cuts a parallelogram into two congruent polygons -/
def cuts_into_congruent_polygons (p : Parallelogram) (l : Line) : Prop :=
  sorry

/-- The main theorem -/
theorem parallelogram_bisector_slope (p : Parallelogram) (l : Line) :
  p.v1 = (5, 20) →
  p.v2 = (5, 50) →
  p.v3 = (20, 100) →
  p.v4 = (20, 70) →
  cuts_into_congruent_polygons p l →
  l.slope = 40 / 9 := by
  sorry

end parallelogram_bisector_slope_l1574_157497


namespace fundraising_average_contribution_l1574_157407

/-- Proves that the average contribution required from the remaining targeted people
    is $400 / 0.36, given the conditions of the fundraising problem. -/
theorem fundraising_average_contribution
  (total_amount : ℝ) 
  (total_people : ℝ) 
  (h1 : total_amount > 0)
  (h2 : total_people > 0)
  (h3 : 0.6 * total_amount = 0.4 * total_people * 400) :
  (0.4 * total_amount) / (0.6 * total_people) = 400 / 0.36 := by
sorry

end fundraising_average_contribution_l1574_157407


namespace oranges_per_box_l1574_157477

theorem oranges_per_box (total_oranges : ℕ) (total_boxes : ℕ) 
  (h1 : total_oranges = 2650) 
  (h2 : total_boxes = 265) :
  total_oranges / total_boxes = 10 := by
sorry

end oranges_per_box_l1574_157477


namespace rhombuses_in_grid_of_25_l1574_157479

/-- Represents a triangular grid of equilateral triangles -/
structure TriangularGrid where
  side_length : ℕ
  total_triangles : ℕ

/-- Calculates the number of rhombuses in a triangular grid -/
def count_rhombuses (grid : TriangularGrid) : ℕ :=
  3 * (grid.side_length - 1) * grid.side_length

/-- Theorem: In a triangular grid with 25 triangles (5 per side), there are 30 rhombuses -/
theorem rhombuses_in_grid_of_25 :
  let grid : TriangularGrid := { side_length := 5, total_triangles := 25 }
  count_rhombuses grid = 30 := by
  sorry


end rhombuses_in_grid_of_25_l1574_157479


namespace single_point_ellipse_l1574_157428

theorem single_point_ellipse (c : ℝ) : 
  (∃! p : ℝ × ℝ, 4 * p.1^2 + p.2^2 + 16 * p.1 - 6 * p.2 + c = 0) → c = 7 := by
  sorry

end single_point_ellipse_l1574_157428


namespace fraction_equality_sum_l1574_157422

theorem fraction_equality_sum (C D : ℝ) :
  (∀ x : ℝ, x ≠ 3 ∧ x ≠ 5 →
    (D * x - 17) / (x^2 - 8*x + 15) = C / (x - 3) + 5 / (x - 5)) →
  C + D = 29/5 := by
sorry

end fraction_equality_sum_l1574_157422


namespace no_negative_exponents_l1574_157443

theorem no_negative_exponents (a b c d : ℤ) 
  (h : (4 : ℝ)^a + (4 : ℝ)^b = (5 : ℝ)^c + (5 : ℝ)^d + 1) : 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d := by
  sorry

end no_negative_exponents_l1574_157443


namespace power_equation_solution_l1574_157452

theorem power_equation_solution (m : ℤ) : (7 : ℝ) ^ (2 * m) = (1 / 7 : ℝ) ^ (m - 30) → m = 10 := by
  sorry

end power_equation_solution_l1574_157452


namespace total_baseball_cards_l1574_157430

-- Define the number of people
def num_people : Nat := 4

-- Define the number of cards each person has
def cards_per_person : Nat := 3

-- Theorem to prove
theorem total_baseball_cards : num_people * cards_per_person = 12 := by
  sorry

end total_baseball_cards_l1574_157430


namespace train_passing_time_l1574_157410

/-- The time taken for a person to walk the length of a train, given the times it takes for the train to pass the person in opposite and same directions. -/
theorem train_passing_time (t₁ t₂ : ℝ) (h₁ : t₁ > 0) (h₂ : t₂ > 0) (h₃ : t₂ > t₁) : 
  let t₃ := (2 * t₁ * t₂) / (t₂ - t₁)
  t₁ = 1 ∧ t₂ = 2 → t₃ = 4 :=
by sorry

end train_passing_time_l1574_157410


namespace sale_price_comparison_l1574_157400

theorem sale_price_comparison (x : ℝ) (h : x > 0) : x * 1.3 * 0.85 > x * 1.1 := by
  sorry

end sale_price_comparison_l1574_157400


namespace open_box_volume_l1574_157413

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume
  (sheet_length : ℝ)
  (sheet_width : ℝ)
  (cut_square_side : ℝ)
  (h_sheet_length : sheet_length = 46)
  (h_sheet_width : sheet_width = 36)
  (h_cut_square_side : cut_square_side = 8) :
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = 4800 :=
by sorry

end open_box_volume_l1574_157413


namespace girls_together_arrangement_person_not_in_middle_l1574_157449

-- Define the number of boys and girls
def num_boys : ℕ := 4
def num_girls : ℕ := 3
def total_people : ℕ := num_boys + num_girls

-- Define permutation and combination functions
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement A
theorem girls_together_arrangement :
  (A num_girls num_girls) * (A (num_boys + 1) (num_boys + 1)) =
  A num_girls num_girls * A 5 5 := by sorry

-- Statement C
theorem person_not_in_middle :
  (C (total_people - 1) 1) * (A (total_people - 1) (total_people - 1)) =
  C 6 1 * A 6 6 := by sorry

end girls_together_arrangement_person_not_in_middle_l1574_157449


namespace boys_height_correction_l1574_157414

theorem boys_height_correction (n : ℕ) (initial_avg wrong_height actual_avg : ℝ) : 
  n = 35 →
  initial_avg = 183 →
  wrong_height = 166 →
  actual_avg = 181 →
  ∃ (correct_height : ℝ), 
    correct_height = wrong_height + (n * initial_avg - n * actual_avg) ∧
    correct_height = 236 :=
by sorry

end boys_height_correction_l1574_157414


namespace circle_theorem_l1574_157447

structure Circle where
  center : Point
  radius : ℝ

structure Angle where
  vertex : Point
  ray1 : Point
  ray2 : Point

def parallel (l1 l2 : Line) : Prop := sorry

def diameter (c : Circle) (l : Line) : Prop := sorry

def inscribed_angle (c : Circle) (a : Angle) : Prop := sorry

def angle_measure (a : Angle) : ℝ := sorry

theorem circle_theorem (c : Circle) (F B D C A : Point) 
  (FB DC AB FD : Line) (AFB ABF BCD : Angle) :
  diameter c FB →
  parallel FB DC →
  parallel AB FD →
  angle_measure AFB / angle_measure ABF = 3 / 4 →
  inscribed_angle c BCD →
  angle_measure BCD = 330 / 7 := by sorry

end circle_theorem_l1574_157447


namespace shadow_boundary_is_constant_l1574_157462

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The xy-plane -/
def xyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Light source position -/
def lightSource : Point3D := ⟨0, -4, 3⟩

/-- The sphere in the problem -/
def problemSphere : Sphere := ⟨⟨0, 0, 2⟩, 2⟩

/-- A point on the boundary of the shadow -/
structure ShadowBoundaryPoint where
  x : ℝ
  y : ℝ

/-- The boundary function of the shadow -/
def shadowBoundary (p : ShadowBoundaryPoint) : Prop :=
  p.y = -19/4

theorem shadow_boundary_is_constant (s : Sphere) (l : Point3D) :
  s = problemSphere →
  l = lightSource →
  ∀ p : ShadowBoundaryPoint, shadowBoundary p := by
  sorry

#check shadow_boundary_is_constant

end shadow_boundary_is_constant_l1574_157462


namespace product_minimum_value_l1574_157465

-- Define the functions h and k
def h : ℝ → ℝ := sorry
def k : ℝ → ℝ := sorry

-- State the theorem
theorem product_minimum_value (x : ℝ) :
  (∀ x, -3 ≤ h x ∧ h x ≤ 4) →
  (∀ x, -1 ≤ k x ∧ k x ≤ 3) →
  -12 ≤ h x * k x :=
sorry

end product_minimum_value_l1574_157465


namespace min_value_product_squares_l1574_157493

theorem min_value_product_squares (a b c d e f g h i j k l m n o p : ℝ) 
  (h1 : a * b * c * d = 16)
  (h2 : e * f * g * h = 16)
  (h3 : i * j * k * l = 16)
  (h4 : m * n * o * p = 16) :
  (a * e * i * m)^2 + (b * f * j * n)^2 + (c * g * k * o)^2 + (d * h * l * p)^2 ≥ 1024 :=
sorry

end min_value_product_squares_l1574_157493


namespace books_loaned_out_correct_loaned_books_l1574_157403

/-- Proves that the number of books loaned out is 160 given the initial and final book counts and return rate -/
theorem books_loaned_out 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (return_rate : ℚ) : ℕ :=
  let loaned_books := (initial_books - final_books) / (1 - return_rate)
  160

/-- The number of books loaned out is 160 -/
theorem correct_loaned_books : books_loaned_out 300 244 (65/100) = 160 := by
  sorry

end books_loaned_out_correct_loaned_books_l1574_157403


namespace solve_for_q_l1574_157485

theorem solve_for_q : ∀ (k r q : ℚ),
  (4 / 5 : ℚ) = k / 90 →
  (4 / 5 : ℚ) = (k + r) / 105 →
  (4 / 5 : ℚ) = (q - r) / 150 →
  q = 132 := by
sorry

end solve_for_q_l1574_157485


namespace ferry_river_crossing_l1574_157405

/-- Two ferries crossing a river problem -/
theorem ferry_river_crossing (W : ℝ) : 
  W > 0 → -- Width of the river is positive
  (∃ (d₁ d₂ : ℝ), 
    d₁ = 700 ∧ -- First meeting point is 700 feet from one shore
    d₁ + d₂ = W ∧ -- Sum of distances at first meeting equals river width
    W + 400 + (W + (W - 400)) = 3 * W ∧ -- Total distance at second meeting
    2 * (W + 700) = 3 * W) → -- Relationship between meetings and river width
  W = 1400 := by
sorry

end ferry_river_crossing_l1574_157405


namespace house_sale_profit_l1574_157437

/-- Calculates the net profit from a house sale and repurchase --/
def netProfit (initialValue : ℝ) (sellProfit : ℝ) (buyLoss : ℝ) : ℝ :=
  let sellPrice := initialValue * (1 + sellProfit)
  let buyPrice := sellPrice * (1 - buyLoss)
  sellPrice - buyPrice

/-- Theorem stating that the net profit is $1725 given the specified conditions --/
theorem house_sale_profit :
  netProfit 15000 0.15 0.10 = 1725 := by
  sorry

#eval netProfit 15000 0.15 0.10

end house_sale_profit_l1574_157437


namespace unique_composite_with_square_predecessor_divisors_l1574_157481

/-- A natural number is composite if it has a proper divisor greater than 1 -/
def IsComposite (n : ℕ) : Prop := ∃ d : ℕ, 1 < d ∧ d < n ∧ n % d = 0

/-- A natural number is a perfect square if it's equal to some integer squared -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- Property: for every natural divisor d of n, d-1 is a perfect square -/
def HasSquarePredecessorDivisors (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 0 → n % d = 0 → IsPerfectSquare (d - 1)

theorem unique_composite_with_square_predecessor_divisors :
  ∃! n : ℕ, IsComposite n ∧ HasSquarePredecessorDivisors n ∧ n = 10 :=
sorry

end unique_composite_with_square_predecessor_divisors_l1574_157481


namespace sqrt_eight_equals_two_sqrt_two_l1574_157408

theorem sqrt_eight_equals_two_sqrt_two : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_eight_equals_two_sqrt_two_l1574_157408


namespace z_in_fourth_quadrant_l1574_157475

/-- The complex number z is in the fourth quadrant of the complex plane -/
theorem z_in_fourth_quadrant : 
  let i : ℂ := Complex.I
  let z : ℂ := 1 + (1 - i) / (1 + i)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end z_in_fourth_quadrant_l1574_157475


namespace division_problem_l1574_157478

theorem division_problem (L S Q : ℝ) : 
  L - S = 1356 →
  S = 268.2 →
  L = S * Q + 15 →
  Q = 6 := by
sorry

end division_problem_l1574_157478


namespace somu_present_age_l1574_157459

/-- Somu's age -/
def somu_age : ℕ := sorry

/-- Somu's father's age -/
def father_age : ℕ := sorry

/-- Somu's age is one-third of his father's age -/
axiom current_age_ratio : somu_age = father_age / 3

/-- 5 years ago, Somu's age was one-fifth of his father's age -/
axiom past_age_ratio : somu_age - 5 = (father_age - 5) / 5

theorem somu_present_age : somu_age = 10 := by sorry

end somu_present_age_l1574_157459


namespace literature_class_b_count_l1574_157436

/-- In a literature class with the given grade distribution, prove the number of B grades. -/
theorem literature_class_b_count (total : ℕ) (p_a p_b p_c : ℝ) (b_count : ℕ) : 
  total = 25 →
  p_a = 0.8 * p_b →
  p_c = 1.2 * p_b →
  p_a + p_b + p_c = 1 →
  b_count = ⌊(total : ℝ) / 3⌋ →
  b_count = 8 := by
sorry

end literature_class_b_count_l1574_157436


namespace probability_all_red_by_fourth_draw_specific_l1574_157417

/-- The probability of drawing all red balls exactly by the 4th draw -/
def probability_all_red_by_fourth_draw (white_balls red_balls : ℕ) : ℚ :=
  let total_balls := white_balls + red_balls
  let prob_white := white_balls / total_balls
  let prob_red := red_balls / total_balls
  prob_white^3 * prob_red

/-- Theorem stating the probability of drawing all red balls exactly by the 4th draw
    given 8 white balls and 2 red balls in a bag -/
theorem probability_all_red_by_fourth_draw_specific :
  probability_all_red_by_fourth_draw 8 2 = 217/5000 := by
  sorry

end probability_all_red_by_fourth_draw_specific_l1574_157417


namespace percentage_of_red_cars_l1574_157473

theorem percentage_of_red_cars (total_cars : ℕ) (honda_cars : ℕ) 
  (honda_red_percentage : ℚ) (non_honda_red_percentage : ℚ) :
  total_cars = 9000 →
  honda_cars = 5000 →
  honda_red_percentage = 90 / 100 →
  non_honda_red_percentage = 225 / 1000 →
  (((honda_red_percentage * honda_cars) + 
    (non_honda_red_percentage * (total_cars - honda_cars))) / total_cars) * 100 = 60 := by
  sorry

end percentage_of_red_cars_l1574_157473


namespace coefficient_of_x_in_triple_expansion_l1574_157425

theorem coefficient_of_x_in_triple_expansion (x : ℝ) : 
  let expansion := (1 + x)^3 + (1 + x)^3 + (1 + x)^3
  ∃ a b c d : ℝ, expansion = a + 9*x + b*x^2 + c*x^3 + d*x^4 :=
by sorry

end coefficient_of_x_in_triple_expansion_l1574_157425


namespace solution_in_interval_l1574_157491

theorem solution_in_interval :
  ∃ x₀ ∈ Set.Ioo 2 3, Real.log x₀ + x₀ - 4 = 0 := by sorry

end solution_in_interval_l1574_157491


namespace parallel_vectors_x_value_l1574_157450

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (2, x + 2)
  are_parallel a b → x = 4 := by
  sorry

end parallel_vectors_x_value_l1574_157450


namespace triple_involution_properties_l1574_157418

/-- A function f: ℝ → ℝ satisfying f(f(f(x))) = x for all x ∈ ℝ -/
def triple_involution (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f (f x)) = x

theorem triple_involution_properties (f : ℝ → ℝ) (h : triple_involution f) :
  (∀ x y : ℝ, f x = f y → x = y) ∧ 
  (¬ (∀ x y : ℝ, x < y → f x > f y)) ∧
  ((∀ x y : ℝ, x < y → f x < f y) → ∀ x : ℝ, f x = x) :=
by sorry

end triple_involution_properties_l1574_157418


namespace selection_plans_count_l1574_157409

def number_of_people : ℕ := 6
def number_of_cities : ℕ := 4
def number_to_select : ℕ := 4
def restricted_people : ℕ := 2
def restricted_city : ℕ := 1

theorem selection_plans_count :
  (number_of_people * (number_of_people - 1) * (number_of_people - 2) * (number_of_people - 3)) -
  (restricted_people * ((number_of_people - 1) * (number_of_people - 2) * (number_of_people - 3))) = 240 := by
  sorry

end selection_plans_count_l1574_157409


namespace victors_class_size_l1574_157472

theorem victors_class_size (total_skittles : ℕ) (skittles_per_classmate : ℕ) 
  (h1 : total_skittles = 25)
  (h2 : skittles_per_classmate = 5) :
  total_skittles / skittles_per_classmate = 5 :=
by sorry

end victors_class_size_l1574_157472


namespace james_painting_fraction_l1574_157423

/-- If a person can paint a wall in a given time, this function calculates
    the fraction of the wall they can paint in a shorter time period. -/
def fractionPainted (totalTime minutes : ℚ) : ℚ :=
  minutes / totalTime

theorem james_painting_fraction :
  fractionPainted 60 15 = 1/4 := by sorry

end james_painting_fraction_l1574_157423


namespace cube_root_squared_eq_81_l1574_157460

theorem cube_root_squared_eq_81 (x : ℝ) :
  (x ^ (1/3)) ^ 2 = 81 → x = 729 := by
  sorry

end cube_root_squared_eq_81_l1574_157460


namespace count_positive_rationals_l1574_157431

def numbers : List ℚ := [-2023, 1/100, 3/2, 0, 1/5]

theorem count_positive_rationals : 
  (numbers.filter (λ x => x > 0)).length = 3 := by sorry

end count_positive_rationals_l1574_157431


namespace probability_five_green_marbles_l1574_157483

def num_green_marbles : ℕ := 6
def num_purple_marbles : ℕ := 4
def total_marbles : ℕ := num_green_marbles + num_purple_marbles
def num_draws : ℕ := 8
def num_green_draws : ℕ := 5

def probability_green : ℚ := num_green_marbles / total_marbles
def probability_purple : ℚ := num_purple_marbles / total_marbles

def combinations : ℕ := Nat.choose num_draws num_green_draws

theorem probability_five_green_marbles :
  (combinations : ℚ) * probability_green ^ num_green_draws * probability_purple ^ (num_draws - num_green_draws) =
  56 * (6/10)^5 * (4/10)^3 :=
sorry

end probability_five_green_marbles_l1574_157483


namespace assignment_operation_l1574_157495

theorem assignment_operation (A : Int) : A = 15 → -A + 5 = -10 := by
  sorry

end assignment_operation_l1574_157495


namespace water_formation_l1574_157438

/-- Represents the balanced chemical equation for the reaction between NH4Cl and NaOH -/
structure ChemicalReaction where
  nh4cl : ℕ
  naoh : ℕ
  h2o : ℕ
  balanced : nh4cl = naoh ∧ nh4cl = h2o

/-- Calculates the moles of water produced in the reaction -/
def waterProduced (reaction : ChemicalReaction) (nh4cl_moles : ℕ) (naoh_moles : ℕ) : ℕ :=
  min nh4cl_moles naoh_moles

theorem water_formation (reaction : ChemicalReaction) 
  (h1 : reaction.nh4cl = 1 ∧ reaction.naoh = 1 ∧ reaction.h2o = 1) 
  (h2 : nh4cl_moles = 3) 
  (h3 : naoh_moles = 3) : 
  waterProduced reaction nh4cl_moles naoh_moles = 3 := by
  sorry

end water_formation_l1574_157438


namespace part_one_part_two_l1574_157467

/-- Given vectors in R^2 -/
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (0, 1)
def c : ℝ × ℝ := (1, -2)

/-- The theorem for the first part of the problem -/
theorem part_one : ∃ (m n : ℝ), a = m • b + n • c ∧ m = 3 ∧ n = 2 := by sorry

/-- The theorem for the second part of the problem -/
theorem part_two : 
  (∃ (d : ℝ × ℝ), ∃ (k : ℝ), k ≠ 0 ∧ (a + d) = k • (b + c)) ∧ 
  (∀ (d : ℝ × ℝ), (∃ (k : ℝ), k ≠ 0 ∧ (a + d) = k • (b + c)) → 
    Real.sqrt 2 / 2 ≤ Real.sqrt (d.1^2 + d.2^2)) ∧
  (∃ (d : ℝ × ℝ), (∃ (k : ℝ), k ≠ 0 ∧ (a + d) = k • (b + c)) ∧ 
    Real.sqrt (d.1^2 + d.2^2) = Real.sqrt 2 / 2) := by sorry

end part_one_part_two_l1574_157467


namespace absolute_value_sum_greater_than_one_l1574_157448

theorem absolute_value_sum_greater_than_one (x y : ℝ) :
  y ≤ -2 → abs x + abs y > 1 := by
  sorry

end absolute_value_sum_greater_than_one_l1574_157448


namespace surface_generates_solid_by_rotation_l1574_157451

/-- A right-angled triangle -/
structure RightTriangle where
  /-- The triangle has a right angle -/
  has_right_angle : Bool

/-- A cone -/
structure Cone where
  /-- The cone is formed by rotation -/
  formed_by_rotation : Bool

/-- Rotation of a triangle around one of its perpendicular sides -/
def rotate_triangle (t : RightTriangle) : Cone :=
  { formed_by_rotation := true }

/-- A theorem stating that rotating a right-angled triangle around one of its perpendicular sides
    demonstrates that a surface can generate a solid through rotation -/
theorem surface_generates_solid_by_rotation (t : RightTriangle) :
  ∃ (c : Cone), c = rotate_triangle t ∧ c.formed_by_rotation :=
by sorry

end surface_generates_solid_by_rotation_l1574_157451


namespace geometric_sequence_minimum_value_l1574_157439

theorem geometric_sequence_minimum_value 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) = 3 * a n) 
  (m n : ℕ) 
  (h_product : a m * a n = 9 * (a 2)^2) :
  (∀ k l : ℕ, 2 / k + 1 / (2 * l) ≥ 3 / 4) ∧ 
  (∃ k l : ℕ, 2 / k + 1 / (2 * l) = 3 / 4) := by
sorry

end geometric_sequence_minimum_value_l1574_157439


namespace coefficient_x4_is_negative_15_l1574_157435

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 5*(x^3 - 2*x^4) + 3*(x^2 - x^4 + 2*x^6) - (2*x^4 + 5*x^3)

/-- The coefficient of x^4 in the simplified expression -/
def coefficient_x4 : ℝ := -15

/-- Theorem stating that the coefficient of x^4 in the simplified expression is -15 -/
theorem coefficient_x4_is_negative_15 :
  ∃ (f : ℝ → ℝ), ∀ x, expression x = f x + coefficient_x4 * x^4 ∧ 
  ∀ n, n ≠ 4 → (∃ c, ∀ x, f x = c * x^n + (f x - c * x^n)) :=
sorry

end coefficient_x4_is_negative_15_l1574_157435


namespace unique_solution_for_exponential_equation_l1574_157404

theorem unique_solution_for_exponential_equation :
  ∀ (a b n p : ℕ), 
    p.Prime → 
    2^a + p^b = n^(p-1) → 
    (a = 0 ∧ b = 1 ∧ n = 2 ∧ p = 3) := by
  sorry

end unique_solution_for_exponential_equation_l1574_157404


namespace meeting_handshakes_l1574_157434

theorem meeting_handshakes (total_handshakes : ℕ) 
  (h1 : total_handshakes = 159) : ∃ (people second_handshakes : ℕ),
  people * (people - 1) / 2 + second_handshakes = total_handshakes ∧
  people = 18 ∧ 
  second_handshakes = 6 := by
sorry

end meeting_handshakes_l1574_157434


namespace field_goal_percentage_l1574_157469

theorem field_goal_percentage (total_attempts : ℕ) (miss_ratio : ℚ) (wide_right : ℕ) : 
  total_attempts = 60 →
  miss_ratio = 1/4 →
  wide_right = 3 →
  (wide_right : ℚ) / (miss_ratio * total_attempts) * 100 = 20 := by
  sorry

end field_goal_percentage_l1574_157469


namespace hemisphere_surface_area_l1574_157402

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) :
  π * r^2 = 225 * π → 2 * π * r^2 + π * r^2 = 675 * π := by
  sorry

end hemisphere_surface_area_l1574_157402


namespace intersection_implies_solution_l1574_157498

/-- Two lines intersecting at a point imply the solution to a related system of equations -/
theorem intersection_implies_solution (b k : ℝ) : 
  (∃ (x y : ℝ), y = -3*x + b ∧ y = -k*x + 1 ∧ x = 1 ∧ y = -2) →
  (∀ (x y : ℝ), 3*x + y = b ∧ k*x + y = 1 ↔ x = 1 ∧ y = -2) :=
by sorry

end intersection_implies_solution_l1574_157498


namespace emily_marbles_problem_l1574_157487

theorem emily_marbles_problem (emily_initial : ℕ) (emily_final : ℕ) : 
  emily_initial = 6 →
  emily_final = 8 →
  ∃ (additional_marbles : ℕ),
    emily_final = emily_initial + 2 * emily_initial - 
      ((emily_initial + 2 * emily_initial) / 2 + additional_marbles) ∧
    additional_marbles = 1 :=
by sorry

end emily_marbles_problem_l1574_157487


namespace even_function_sum_l1574_157496

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_sum (f : ℝ → ℝ) (h1 : is_even_function f) (h2 : f 4 = 3) :
  f 4 + f (-4) = 6 := by
  sorry

end even_function_sum_l1574_157496


namespace coprime_sum_not_divides_power_sum_l1574_157464

theorem coprime_sum_not_divides_power_sum
  (x y n : ℕ)
  (h_coprime : Nat.Coprime x y)
  (h_positive : 0 < x ∧ 0 < y)
  (h_not_one : x * y ≠ 1)
  (h_even : Even n)
  (h_pos : 0 < n) :
  ¬ (x + y ∣ x^n + y^n) :=
sorry

end coprime_sum_not_divides_power_sum_l1574_157464


namespace expected_boy_girl_pairs_l1574_157454

/-- The number of boys in the line -/
def num_boys : ℕ := 9

/-- The number of girls in the line -/
def num_girls : ℕ := 15

/-- The total number of people in the line -/
def total_people : ℕ := num_boys + num_girls

/-- The number of adjacent pairs in the line -/
def num_pairs : ℕ := total_people - 1

/-- The probability of a boy-girl pair at any given position -/
def prob_boy_girl_pair : ℚ := (2 * (num_boys - 1) * (num_girls - 1)) / ((total_people - 2) * (total_people - 3))

theorem expected_boy_girl_pairs :
  (num_pairs : ℚ) * prob_boy_girl_pair = 920 / 77 := by sorry

end expected_boy_girl_pairs_l1574_157454


namespace quadratic_inequality_range_l1574_157424

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 6*a*x + 1 < 0) → a ∈ Set.Icc (-1/3) (1/3) := by
  sorry

end quadratic_inequality_range_l1574_157424


namespace max_intersections_theorem_l1574_157455

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ

/-- Calculates the maximum number of intersections between two convex polygons -/
def max_intersections (P₁ P₂ : ConvexPolygon) (k : ℕ) : ℕ :=
  k * P₂.sides

/-- Theorem stating the maximum number of intersections between two convex polygons -/
theorem max_intersections_theorem 
  (P₁ P₂ : ConvexPolygon) 
  (k : ℕ) 
  (h₁ : P₁.sides ≤ P₂.sides) 
  (h₂ : k ≤ P₁.sides) : 
  max_intersections P₁ P₂ k = k * P₂.sides :=
by
  sorry

#check max_intersections_theorem

end max_intersections_theorem_l1574_157455


namespace max_value_negative_x_min_value_greater_than_negative_one_l1574_157445

-- Problem 1
theorem max_value_negative_x (x : ℝ) (hx : x < 0) :
  (x^2 + x + 1) / x ≤ -1 :=
sorry

-- Problem 2
theorem min_value_greater_than_negative_one (x : ℝ) (hx : x > -1) :
  ((x + 5) * (x + 2)) / (x + 1) ≥ 9 :=
sorry

end max_value_negative_x_min_value_greater_than_negative_one_l1574_157445


namespace greater_number_proof_l1574_157492

theorem greater_number_proof (x y : ℝ) (h1 : x > y) (h2 : x * y = 2688) (h3 : x + y - (x - y) = 64) : x = 84 := by
  sorry

end greater_number_proof_l1574_157492


namespace smaller_number_problem_l1574_157415

theorem smaller_number_problem (x y : ℝ) (h_sum : x + y = 18) (h_product : x * y = 80) :
  min x y = 8 := by
  sorry

end smaller_number_problem_l1574_157415
