import Mathlib

namespace expected_waiting_time_correct_l1255_125567

/-- Represents the arrival time of a bus in minutes past 8:00 AM -/
def BusArrivalTime := Fin 120

/-- Represents the probability distribution of bus arrivals -/
def BusDistribution := BusArrivalTime → ℝ

/-- The first bus arrives randomly between 8:00 and 9:00 -/
def firstBusDistribution : BusDistribution := sorry

/-- The second bus arrives randomly between 9:00 and 10:00 -/
def secondBusDistribution : BusDistribution := sorry

/-- The passenger arrival time in minutes past 8:00 AM -/
def passengerArrivalTime : ℕ := 20

/-- Expected waiting time function -/
def expectedWaitingTime (firstBus secondBus : BusDistribution) (passengerTime : ℕ) : ℝ := sorry

theorem expected_waiting_time_correct :
  expectedWaitingTime firstBusDistribution secondBusDistribution passengerArrivalTime = 160 / 3 := by
  sorry

end expected_waiting_time_correct_l1255_125567


namespace min_diagonal_of_rectangle_l1255_125535

/-- Given a rectangle ABCD with perimeter 24 inches, 
    the minimum possible length of its diagonal AC is 6√2 inches. -/
theorem min_diagonal_of_rectangle (l w : ℝ) : 
  (l + w = 12) →  -- perimeter condition: 2l + 2w = 24, simplified
  ∃ (d : ℝ), d = Real.sqrt (l^2 + w^2) ∧ 
              d ≥ 6 * Real.sqrt 2 ∧
              (∀ l' w' : ℝ, l' + w' = 12 → 
                Real.sqrt (l'^2 + w'^2) ≥ 6 * Real.sqrt 2) :=
by sorry

end min_diagonal_of_rectangle_l1255_125535


namespace final_values_after_assignments_l1255_125566

/-- This theorem proves that after a series of assignments, 
    the final values of a and b are both 4. -/
theorem final_values_after_assignments :
  let a₀ : ℕ := 3
  let b₀ : ℕ := 4
  let a₁ : ℕ := b₀
  let b₁ : ℕ := a₁
  (a₁ = 4 ∧ b₁ = 4) :=
by sorry

#check final_values_after_assignments

end final_values_after_assignments_l1255_125566


namespace tree_height_l1255_125531

/-- The height of a tree given specific conditions involving a rope and a person. -/
theorem tree_height (rope_ground_distance : ℝ) (person_distance : ℝ) (person_height : ℝ) 
  (h1 : rope_ground_distance = 4)
  (h2 : person_distance = 3)
  (h3 : person_height = 1.6)
  (h4 : person_distance < rope_ground_distance) : 
  ∃ (tree_height : ℝ), tree_height = 6.4 := by
  sorry


end tree_height_l1255_125531


namespace coffee_table_price_correct_l1255_125533

/-- Represents the price of the coffee table -/
def coffee_table_price : ℝ := 429.24

/-- Calculates the total cost before discount and tax -/
def total_before_discount_and_tax (coffee_table_price : ℝ) : ℝ :=
  1250 + 2 * 425 + 350 + 200 + coffee_table_price

/-- Calculates the discounted total -/
def discounted_total (coffee_table_price : ℝ) : ℝ :=
  0.9 * total_before_discount_and_tax coffee_table_price

/-- Calculates the final invoice amount after tax -/
def final_invoice_amount (coffee_table_price : ℝ) : ℝ :=
  1.06 * discounted_total coffee_table_price

/-- Theorem stating that the calculated coffee table price results in the given final invoice amount -/
theorem coffee_table_price_correct :
  final_invoice_amount coffee_table_price = 2937.60 := by
  sorry


end coffee_table_price_correct_l1255_125533


namespace max_value_theorem_max_value_achieved_l1255_125576

theorem max_value_theorem (A M C : ℕ) (h : A + M + C = 15) :
  2 * (A * M * C) + A * M + M * C + C * A ≤ 325 :=
by sorry

theorem max_value_achieved (A M C : ℕ) (h : A + M + C = 15) :
  ∃ A M C, A + M + C = 15 ∧ 2 * (A * M * C) + A * M + M * C + C * A = 325 :=
by sorry

end max_value_theorem_max_value_achieved_l1255_125576


namespace congruence_problem_l1255_125569

theorem congruence_problem (x : ℤ) 
  (h1 : (4 + x) % 16 = 9)
  (h2 : (6 + x) % 36 = 16)
  (h3 : (8 + x) % 64 = 36) :
  x % 48 = 37 := by
  sorry

end congruence_problem_l1255_125569


namespace cone_generatrix_length_l1255_125514

/-- Given a cone with base radius √2 and lateral surface that unfolds into a semicircle,
    the length of the generatrix is 2√2. -/
theorem cone_generatrix_length (r : ℝ) (l : ℝ) : 
  r = Real.sqrt 2 → 
  2 * Real.pi * r = Real.pi * l → 
  l = 2 * Real.sqrt 2 := by
  sorry

end cone_generatrix_length_l1255_125514


namespace distance_and_speed_l1255_125501

-- Define the variables
def distance : ℝ := sorry
def speed_second_car : ℝ := sorry
def speed_first_car : ℝ := sorry
def speed_third_car : ℝ := sorry

-- Define the relationships between the speeds
axiom speed_diff_first_second : speed_first_car = speed_second_car + 4
axiom speed_diff_second_third : speed_second_car = speed_third_car + 6

-- Define the time differences
axiom time_diff_first_second : distance / speed_first_car = distance / speed_second_car - 3 / 60
axiom time_diff_second_third : distance / speed_second_car = distance / speed_third_car - 5 / 60

-- Theorem to prove
theorem distance_and_speed : distance = 120 ∧ speed_second_car = 96 := by
  sorry

end distance_and_speed_l1255_125501


namespace quadratic_function_properties_l1255_125542

def f (a x : ℝ) := a * x^2 + x + 1

theorem quadratic_function_properties (a : ℝ) (h_a : a > 0) :
  -- Part 1: Maximum value in the interval [-4, -2]
  (∀ x ∈ Set.Icc (-4) (-2), f a x ≤ (if a ≤ 1/6 then 4*a - 1 else 16*a - 3)) ∧
  (∃ x ∈ Set.Icc (-4) (-2), f a x = (if a ≤ 1/6 then 4*a - 1 else 16*a - 3)) ∧
  -- Part 2: Maximum value of a given root conditions
  (∀ x₁ x₂ : ℝ, f a x₁ = 0 → f a x₂ = 0 → x₁ ≠ x₂ → x₁ / x₂ ∈ Set.Icc (1/10) 10 → a ≤ 1/4) ∧
  (∃ x₁ x₂ : ℝ, f (1/4) x₁ = 0 ∧ f (1/4) x₂ = 0 ∧ x₁ ≠ x₂ ∧ x₁ / x₂ ∈ Set.Icc (1/10) 10) :=
by sorry

end quadratic_function_properties_l1255_125542


namespace simplify_and_evaluate_1_simplify_and_evaluate_2_l1255_125515

-- Problem 1
theorem simplify_and_evaluate_1 : 2 * Real.sqrt 3 * 31.5 * 612 = 6 := by sorry

-- Problem 2
theorem simplify_and_evaluate_2 : 
  (Real.log 3 / Real.log 4 - Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 1/4 := by sorry

end simplify_and_evaluate_1_simplify_and_evaluate_2_l1255_125515


namespace maisy_earnings_difference_l1255_125523

/-- Represents Maisy's job details -/
structure Job where
  hours : ℕ
  wage : ℕ
  bonus : ℕ

/-- Calculates the weekly earnings for a job -/
def weekly_earnings (job : Job) : ℕ :=
  job.hours * job.wage + job.bonus

/-- Theorem: Maisy earns $15 more per week at her new job -/
theorem maisy_earnings_difference :
  let current_job : Job := ⟨8, 10, 0⟩
  let new_job : Job := ⟨4, 15, 35⟩
  weekly_earnings new_job - weekly_earnings current_job = 15 :=
by sorry

end maisy_earnings_difference_l1255_125523


namespace complex_fraction_real_l1255_125500

theorem complex_fraction_real (a : ℝ) : 
  (((1 : ℂ) + a * I) / ((2 : ℂ) + I)).im = 0 → a = (1 : ℝ) / 2 := by
  sorry

end complex_fraction_real_l1255_125500


namespace diamond_expression_evaluation_l1255_125524

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- State the theorem
theorem diamond_expression_evaluation :
  let x := diamond (diamond 2 3) 4
  let y := diamond 2 (diamond 3 4)
  x - y = -29/132 := by sorry

end diamond_expression_evaluation_l1255_125524


namespace sqrt_x_plus_2_equals_2_implies_x_plus_2_squared_equals_16_l1255_125544

theorem sqrt_x_plus_2_equals_2_implies_x_plus_2_squared_equals_16 (x : ℝ) :
  Real.sqrt (x + 2) = 2 → (x + 2)^2 = 16 := by
  sorry

end sqrt_x_plus_2_equals_2_implies_x_plus_2_squared_equals_16_l1255_125544


namespace sugar_problem_l1255_125502

/-- Calculates the remaining sugar after a bag is torn -/
def remaining_sugar (initial_sugar : ℕ) (num_bags : ℕ) (torn_bags : ℕ) : ℕ :=
  let sugar_per_bag := initial_sugar / num_bags
  let intact_sugar := sugar_per_bag * (num_bags - torn_bags)
  let torn_bag_sugar := sugar_per_bag / 2
  intact_sugar + (torn_bag_sugar * torn_bags)

/-- Theorem stating that 21 kilos of sugar remain after one bag is torn -/
theorem sugar_problem :
  remaining_sugar 24 4 1 = 21 := by
  sorry

end sugar_problem_l1255_125502


namespace problem_solution_l1255_125564

-- Define the propositions
def proposition_A (a : ℝ) : Prop :=
  ∀ x, x^2 + (2*a - 1)*x + a^2 > 0

def proposition_B (a : ℝ) : Prop :=
  ∀ x y, x < y → (a^2 - 1)^x > (a^2 - 1)^y

-- Define the theorem
theorem problem_solution :
  (∀ a : ℝ, (proposition_A a ∨ proposition_B a) ↔ (a < -1 ∧ a > -Real.sqrt 2) ∨ a > 1/4) ∧
  (∀ a : ℝ, a < -1 ∧ a > -Real.sqrt 2 → a^3 + 1 < a^2 + a) :=
sorry

end problem_solution_l1255_125564


namespace cubic_equation_roots_difference_l1255_125516

theorem cubic_equation_roots_difference (x : ℝ) : 
  (64 * x^3 - 144 * x^2 + 92 * x - 15 = 0) →
  (∃ a d : ℝ, {a - d, a, a + d} ⊆ {x | 64 * x^3 - 144 * x^2 + 92 * x - 15 = 0}) →
  (∃ r₁ r₂ r₃ : ℝ, 
    r₁ < r₂ ∧ r₂ < r₃ ∧
    {r₁, r₂, r₃} = {x | 64 * x^3 - 144 * x^2 + 92 * x - 15 = 0} ∧
    r₃ - r₁ = 1) :=
by sorry

end cubic_equation_roots_difference_l1255_125516


namespace pyramid_surface_area_l1255_125547

-- Define the pyramid
structure RectangularPyramid where
  baseSideLength : ℝ
  sideEdgeLength : ℝ

-- Define the properties of the pyramid
def isPyramidOnSphere (p : RectangularPyramid) : Prop :=
  (p.baseSideLength ^ 2 + p.baseSideLength ^ 2 + p.sideEdgeLength ^ 2) / 4 = 1

def hasSquareBase (p : RectangularPyramid) : Prop :=
  p.baseSideLength ^ 2 + p.baseSideLength ^ 2 = 2

def sideEdgesPerpendicular (p : RectangularPyramid) : Prop :=
  p.sideEdgeLength ^ 2 + p.baseSideLength ^ 2 / 2 = 1

-- Define the surface area calculation
def surfaceArea (p : RectangularPyramid) : ℝ :=
  p.baseSideLength ^ 2 + 4 * p.baseSideLength * p.sideEdgeLength

-- Theorem statement
theorem pyramid_surface_area (p : RectangularPyramid) :
  isPyramidOnSphere p → hasSquareBase p → sideEdgesPerpendicular p →
  surfaceArea p = 2 + 4 * Real.sqrt 2 := by
  sorry

end pyramid_surface_area_l1255_125547


namespace boys_without_notebooks_l1255_125521

/-- Proves the number of boys without notebooks in Ms. Johnson's class -/
theorem boys_without_notebooks
  (total_boys : ℕ)
  (students_with_notebooks : ℕ)
  (girls_with_notebooks : ℕ)
  (h1 : total_boys = 24)
  (h2 : students_with_notebooks = 30)
  (h3 : girls_with_notebooks = 17) :
  total_boys - (students_with_notebooks - girls_with_notebooks) = 11 :=
by sorry

end boys_without_notebooks_l1255_125521


namespace integral_x_sin_ax_over_x2_plus_k2_l1255_125554

/-- The integral of x*sin(ax)/(x^2 + k^2) from 0 to infinity equals (π/2)*e^(-ak) for positive a and k -/
theorem integral_x_sin_ax_over_x2_plus_k2 (a k : ℝ) (ha : a > 0) (hk : k > 0) :
  ∫ (x : ℝ) in Set.Ici 0, (x * Real.sin (a * x)) / (x^2 + k^2) = (Real.pi / 2) * Real.exp (-a * k) := by
  sorry

end integral_x_sin_ax_over_x2_plus_k2_l1255_125554


namespace parallel_planes_line_parallel_parallel_planes_perpendicular_line_not_always_perpendicular_planes_perpendicular_line_parallel_not_always_perpendicular_planes_line_perpendicular_l1255_125503

-- Define the basic types
variable (P : Type) -- Type for planes
variable (L : Type) -- Type for lines

-- Define the relations
variable (parallel_planes : P → P → Prop)
variable (perpendicular_planes : P → P → Prop)
variable (line_in_plane : L → P → Prop)
variable (line_parallel_to_plane : L → P → Prop)
variable (line_perpendicular_to_plane : L → P → Prop)

-- State the theorems
theorem parallel_planes_line_parallel
  (p1 p2 : P) (l : L)
  (h1 : parallel_planes p1 p2)
  (h2 : line_in_plane l p1) :
  line_parallel_to_plane l p2 :=
sorry

theorem parallel_planes_perpendicular_line
  (p1 p2 : P) (l : L)
  (h1 : parallel_planes p1 p2)
  (h2 : line_perpendicular_to_plane l p1) :
  line_perpendicular_to_plane l p2 :=
sorry

theorem not_always_perpendicular_planes_perpendicular_line_parallel
  (p1 p2 : P) (l : L) :
  ¬ (∀ (h1 : perpendicular_planes p1 p2)
      (h2 : line_perpendicular_to_plane l p1),
      line_parallel_to_plane l p2) :=
sorry

theorem not_always_perpendicular_planes_line_perpendicular
  (p1 p2 : P) (l : L) :
  ¬ (∀ (h1 : perpendicular_planes p1 p2)
      (h2 : line_in_plane l p1),
      line_perpendicular_to_plane l p2) :=
sorry

end parallel_planes_line_parallel_parallel_planes_perpendicular_line_not_always_perpendicular_planes_perpendicular_line_parallel_not_always_perpendicular_planes_line_perpendicular_l1255_125503


namespace skating_minutes_tenth_day_l1255_125574

def minutes_per_day_first_5 : ℕ := 75
def days_first_period : ℕ := 5
def minutes_per_day_next_3 : ℕ := 120
def days_second_period : ℕ := 3
def total_days : ℕ := 10
def target_average : ℕ := 95

theorem skating_minutes_tenth_day : 
  ∃ (x : ℕ), 
    (minutes_per_day_first_5 * days_first_period + 
     minutes_per_day_next_3 * days_second_period + x) / total_days = target_average ∧
    x = 215 := by
  sorry

end skating_minutes_tenth_day_l1255_125574


namespace max_product_at_three_l1255_125568

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n-1)

def product_of_terms (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  (a₁ * r^((n-1)/2))^n

theorem max_product_at_three (a₁ r : ℝ) (h₁ : a₁ = 3) (h₂ : r = 2/5) :
  ∀ k : ℕ, k ≠ 0 → product_of_terms a₁ r 3 ≥ product_of_terms a₁ r k :=
by sorry

end max_product_at_three_l1255_125568


namespace smallest_non_factor_product_l1255_125549

theorem smallest_non_factor_product (a b : ℕ+) : 
  a ≠ b →
  a ∣ 48 →
  b ∣ 48 →
  ¬(a * b ∣ 48) →
  (∀ (c d : ℕ+), c ≠ d → c ∣ 48 → d ∣ 48 → ¬(c * d ∣ 48) → a * b ≤ c * d) →
  a * b = 18 :=
by sorry

end smallest_non_factor_product_l1255_125549


namespace clouds_weather_relationship_l1255_125579

/-- Represents the contingency table data --/
structure ContingencyTable where
  clouds_rain : Nat
  clouds_no_rain : Nat
  no_clouds_rain : Nat
  no_clouds_no_rain : Nat

/-- Represents the χ² test result --/
structure ChiSquareTest where
  calculated_value : Real
  critical_value : Real

/-- Theorem stating the relationship between clouds at sunset and nighttime weather --/
theorem clouds_weather_relationship (data : ContingencyTable) (test : ChiSquareTest) :
  data.clouds_rain + data.clouds_no_rain + data.no_clouds_rain + data.no_clouds_no_rain = 100 →
  data.clouds_rain + data.no_clouds_rain = 50 →
  data.clouds_no_rain + data.no_clouds_no_rain = 50 →
  test.calculated_value > test.critical_value →
  ∃ (relationship : Prop), relationship := by
  sorry

#check clouds_weather_relationship

end clouds_weather_relationship_l1255_125579


namespace maple_trees_planted_proof_l1255_125559

/-- The number of maple trees planted in a park -/
def maple_trees_planted (initial : ℕ) (final : ℕ) : ℕ := final - initial

/-- Theorem stating that 11 maple trees were planted -/
theorem maple_trees_planted_proof :
  let initial_trees : ℕ := 53
  let final_trees : ℕ := 64
  maple_trees_planted initial_trees final_trees = 11 := by
  sorry

end maple_trees_planted_proof_l1255_125559


namespace same_last_digit_count_l1255_125589

def has_same_last_digit (x : ℕ) : Bool :=
  x % 10 = (64 - x) % 10

def count_same_last_digit : ℕ :=
  (List.range 63).filter (λ x => has_same_last_digit (x + 1)) |>.length

theorem same_last_digit_count : count_same_last_digit = 13 := by
  sorry

end same_last_digit_count_l1255_125589


namespace leah_saves_fifty_cents_l1255_125590

/-- Represents the daily savings of Leah in dollars -/
def leah_daily_savings : ℝ := sorry

/-- Represents Josiah's total savings in dollars -/
def josiah_total_savings : ℝ := 0.25 * 24

/-- Represents Leah's total savings in dollars -/
def leah_total_savings : ℝ := leah_daily_savings * 20

/-- Represents Megan's total savings in dollars -/
def megan_total_savings : ℝ := 2 * leah_daily_savings * 12

/-- The total amount saved by all three children -/
def total_savings : ℝ := 28

/-- Theorem stating that Leah's daily savings amount to $0.50 -/
theorem leah_saves_fifty_cents :
  josiah_total_savings + leah_total_savings + megan_total_savings = total_savings →
  leah_daily_savings = 0.50 := by sorry

end leah_saves_fifty_cents_l1255_125590


namespace f_composed_three_roots_l1255_125593

/-- A quadratic function f(x) = x^2 + 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

/-- The composition of f with itself -/
def f_composed (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Predicate to check if a function has exactly 3 distinct real roots -/
def has_three_distinct_real_roots (g : ℝ → ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x : ℝ, g x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃

theorem f_composed_three_roots :
  ∀ c : ℝ, has_three_distinct_real_roots (f_composed c) ↔ c = (11 - Real.sqrt 13) / 2 :=
sorry

end f_composed_three_roots_l1255_125593


namespace reciprocal_equals_self_l1255_125582

theorem reciprocal_equals_self (x : ℚ) : x = x⁻¹ ↔ x = 1 ∨ x = -1 := by
  sorry

end reciprocal_equals_self_l1255_125582


namespace max_binomial_probability_l1255_125517

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem max_binomial_probability :
  ∃ (k : ℕ), k ≤ 5 ∧
  ∀ (j : ℕ), j ≤ 5 →
    binomial_probability 5 k (1/4) ≥ binomial_probability 5 j (1/4) ∧
  k = 1 :=
sorry

end max_binomial_probability_l1255_125517


namespace largest_common_divisor_problem_l1255_125526

theorem largest_common_divisor_problem : Nat.gcd (69 - 5) (86 - 6) = 16 := by
  sorry

end largest_common_divisor_problem_l1255_125526


namespace parallel_lines_distance_l1255_125571

/-- Given two lines l₁ and l₂ in the form x + ay = 1 and ax + y = 1 respectively,
    if they are parallel, then the distance between them is √2. -/
theorem parallel_lines_distance (a : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | x + a * y = 1}
  let l₂ := {(x, y) : ℝ × ℝ | a * x + y = 1}
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l₁ → (x₂, y₂) ∈ l₂ → 
    (y₂ - y₁) / (x₂ - x₁) = (y₁ - y₂) / (x₁ - x₂)) →
  (∃ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ l₁ ∧ p₂ ∈ l₂ ∧ 
    Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) = Real.sqrt 2) :=
by sorry

end parallel_lines_distance_l1255_125571


namespace circumscribed_sphere_volume_l1255_125563

theorem circumscribed_sphere_volume (cube_surface_area : ℝ) (h : cube_surface_area = 24) :
  let cube_edge := Real.sqrt (cube_surface_area / 6)
  let sphere_radius := cube_edge * Real.sqrt 3 / 2
  (4 / 3) * Real.pi * sphere_radius ^ 3 = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end circumscribed_sphere_volume_l1255_125563


namespace abs_x_squared_minus_4x_plus_3_lt_6_l1255_125507

theorem abs_x_squared_minus_4x_plus_3_lt_6 (x : ℝ) :
  |x^2 - 4*x + 3| < 6 ↔ 1 < x ∧ x < 3 := by
  sorry

end abs_x_squared_minus_4x_plus_3_lt_6_l1255_125507


namespace chemical_mixture_volume_l1255_125561

theorem chemical_mixture_volume : 
  ∀ (initial_volume : ℝ),
  initial_volume > 0 →
  0.30 * initial_volume + 20 = 0.44 * (initial_volume + 20) →
  initial_volume = 80 :=
λ initial_volume h_positive h_equation =>
  sorry

end chemical_mixture_volume_l1255_125561


namespace rationalize_denominator_l1255_125532

theorem rationalize_denominator : 
  (Real.sqrt 18 + Real.sqrt 8) / (Real.sqrt 12 + Real.sqrt 8) = 2.5 * Real.sqrt 6 - 4 := by
  sorry

end rationalize_denominator_l1255_125532


namespace quadratic_root_relation_l1255_125543

theorem quadratic_root_relation (p q : ℤ) : 
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x = 4*y) ∧ 
  (abs p < 100) ∧ (abs q < 100) ↔ 
  ((p = 5 ∨ p = -5) ∧ q = 4) ∨
  ((p = 10 ∨ p = -10) ∧ q = 16) ∨
  ((p = 15 ∨ p = -15) ∧ q = 36) ∨
  ((p = 20 ∨ p = -20) ∧ q = 64) :=
by sorry

end quadratic_root_relation_l1255_125543


namespace theater_ticket_income_theater_income_proof_l1255_125510

/-- Calculate the total ticket income for a theater -/
theorem theater_ticket_income 
  (total_seats : ℕ) 
  (adult_price child_price : ℚ) 
  (children_count : ℕ) : ℚ :=
  let adult_count : ℕ := total_seats - children_count
  let adult_income : ℚ := adult_count * adult_price
  let child_income : ℚ := children_count * child_price
  adult_income + child_income

/-- Prove that the total ticket income for the given theater scenario is $510.00 -/
theorem theater_income_proof :
  theater_ticket_income 200 3 (3/2) 60 = 510 := by
  sorry

end theater_ticket_income_theater_income_proof_l1255_125510


namespace square_kilometer_conversion_time_conversion_l1255_125580

-- Define the conversion rates
def sq_km_to_hectares : ℝ := 100
def hour_to_minutes : ℝ := 60

-- Define the problem statements
def problem1 (sq_km : ℝ) (whole_sq_km : ℕ) (hectares : ℕ) : Prop :=
  sq_km = whole_sq_km + hectares / sq_km_to_hectares

def problem2 (hours : ℝ) (whole_hours : ℕ) (minutes : ℕ) : Prop :=
  hours = whole_hours + minutes / hour_to_minutes

-- Theorem statements
theorem square_kilometer_conversion :
  problem1 7.05 7 500 := by sorry

theorem time_conversion :
  problem2 6.7 6 42 := by sorry

end square_kilometer_conversion_time_conversion_l1255_125580


namespace quadratic_inequality_l1255_125528

theorem quadratic_inequality (x : ℝ) : x^2 + 3*x < 10 ↔ -5 < x ∧ x < 2 := by
  sorry

end quadratic_inequality_l1255_125528


namespace cookies_per_box_type1_is_12_l1255_125584

/-- Represents the number of cookies in a box of the first type -/
def cookies_per_box_type1 : ℕ := 12

/-- Represents the number of cookies in a box of the second type -/
def cookies_per_box_type2 : ℕ := 20

/-- Represents the number of cookies in a box of the third type -/
def cookies_per_box_type3 : ℕ := 16

/-- Represents the number of boxes sold of the first type -/
def boxes_sold_type1 : ℕ := 50

/-- Represents the number of boxes sold of the second type -/
def boxes_sold_type2 : ℕ := 80

/-- Represents the number of boxes sold of the third type -/
def boxes_sold_type3 : ℕ := 70

/-- Represents the total number of cookies sold -/
def total_cookies_sold : ℕ := 3320

/-- Theorem stating that the number of cookies in each box of the first type is 12 -/
theorem cookies_per_box_type1_is_12 :
  cookies_per_box_type1 * boxes_sold_type1 +
  cookies_per_box_type2 * boxes_sold_type2 +
  cookies_per_box_type3 * boxes_sold_type3 = total_cookies_sold :=
by sorry

end cookies_per_box_type1_is_12_l1255_125584


namespace special_angle_calculation_l1255_125570

theorem special_angle_calculation :
  let tan30 := Real.sqrt 3 / 3
  let cos60 := 1 / 2
  let sin45 := Real.sqrt 2 / 2
  Real.sqrt 3 * tan30 + 2 * cos60 - Real.sqrt 2 * sin45 = 1 := by sorry

end special_angle_calculation_l1255_125570


namespace find_b_l1255_125577

theorem find_b (a b c : ℤ) (eq1 : a + 5 = b) (eq2 : 5 + b = c) (eq3 : b + c = a) : b = -10 := by
  sorry

end find_b_l1255_125577


namespace triangle_height_sum_bound_l1255_125522

/-- For a triangle with side lengths a ≤ b ≤ c, heights h_a, h_b, h_c,
    semiperimeter p, and circumradius R, the sum of heights is bounded. -/
theorem triangle_height_sum_bound (a b c h_a h_b h_c p R : ℝ) :
  a ≤ b → b ≤ c → a > 0 → b > 0 → c > 0 →
  p = (a + b + c) / 2 →
  R > 0 →
  h_a + h_b + h_c ≤ (3 * b * (a^2 + a*c + c^2)) / (4 * p * R) := by
  sorry

end triangle_height_sum_bound_l1255_125522


namespace employee_discount_percentage_l1255_125592

theorem employee_discount_percentage
  (wholesale_cost : ℝ)
  (markup_percentage : ℝ)
  (employee_paid_price : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : markup_percentage = 20)
  (h3 : employee_paid_price = 168) :
  let retail_price := wholesale_cost * (1 + markup_percentage / 100)
  let discount_amount := retail_price - employee_paid_price
  let discount_percentage := (discount_amount / retail_price) * 100
  discount_percentage = 30 := by
sorry

end employee_discount_percentage_l1255_125592


namespace f_neg_two_equals_thirteen_l1255_125586

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^4 + b * x^2 - x + 1

-- State the theorem
theorem f_neg_two_equals_thirteen (a b : ℝ) (h : f a b 2 = 9) : f a b (-2) = 13 := by
  sorry

end f_neg_two_equals_thirteen_l1255_125586


namespace consecutive_odd_integers_sum_l1255_125598

theorem consecutive_odd_integers_sum (a : ℤ) : 
  (∃ (b c : ℤ), b = a + 2 ∧ c = a + 4 ∧ a + c = 100) →
  a + (a + 2) + (a + 4) = 150 := by
sorry

end consecutive_odd_integers_sum_l1255_125598


namespace train_length_l1255_125540

/-- The length of a train given specific conditions -/
theorem train_length (t : ℝ) (v : ℝ) (b : ℝ) (h1 : t = 30) (h2 : v = 45) (h3 : b = 205) :
  v * (1000 / 3600) * t - b = 170 :=
by sorry

end train_length_l1255_125540


namespace maths_fraction_in_class_l1255_125578

theorem maths_fraction_in_class (total_students : ℕ) 
  (maths_and_history_students : ℕ) :
  total_students = 25 →
  maths_and_history_students = 20 →
  ∃ (maths_fraction : ℚ),
    maths_fraction * total_students +
    (1 / 3 : ℚ) * (total_students - maths_fraction * total_students) +
    (total_students - maths_fraction * total_students - 
     (1 / 3 : ℚ) * (total_students - maths_fraction * total_students)) = total_students ∧
    maths_fraction * total_students +
    (total_students - maths_fraction * total_students - 
     (1 / 3 : ℚ) * (total_students - maths_fraction * total_students)) = maths_and_history_students ∧
    maths_fraction = 2 / 5 := by
  sorry

end maths_fraction_in_class_l1255_125578


namespace enclosed_area_is_five_twelfths_l1255_125575

noncomputable def f (x : ℝ) : ℝ := x^(1/2)
noncomputable def g (x : ℝ) : ℝ := x^3

theorem enclosed_area_is_five_twelfths :
  ∫ x in (0)..(1), (f x - g x) = 5/12 := by
  sorry

end enclosed_area_is_five_twelfths_l1255_125575


namespace tv_price_reduction_l1255_125511

/-- Proves that a price reduction resulting in a 75% increase in sales and a 31.25% increase in total sale value implies a 25% price reduction -/
theorem tv_price_reduction (P : ℝ) (N : ℝ) (x : ℝ) 
  (h1 : P > 0) 
  (h2 : N > 0) 
  (h3 : x > 0) 
  (h4 : x < 100) 
  (h5 : P * (1 - x / 100) * (N * 1.75) = P * N * 1.3125) : 
  x = 25 := by
sorry

end tv_price_reduction_l1255_125511


namespace find_unknown_number_l1255_125560

theorem find_unknown_number : ∃ x : ℝ, (20 + 40 + 60) / 3 = ((10 + 70 + x) / 3) + 4 := by
  sorry

end find_unknown_number_l1255_125560


namespace book_page_sum_problem_l1255_125504

theorem book_page_sum_problem :
  ∃ (n : ℕ) (p : ℕ), 
    0 < n ∧ 
    1 ≤ p ∧ 
    p ≤ n ∧ 
    n * (n + 1) / 2 + p = 2550 ∧ 
    p = 65 := by
  sorry

end book_page_sum_problem_l1255_125504


namespace quadratic_one_solution_l1255_125557

theorem quadratic_one_solution (a : ℝ) : 
  (∃! x : ℝ, x^2 + a*x + 1 = 0) ↔ (a = 2 ∨ a = -2) := by
sorry

end quadratic_one_solution_l1255_125557


namespace curve_properties_l1255_125545

-- Define the curve
def curve (x y : ℝ) : Prop := x^3 + x*y + y^3 = 3

-- Define symmetry with respect to y = -x
def symmetric_about_neg_x (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ f (-y) (-x)

-- Define a point being on the curve
def point_on_curve (x y : ℝ) : Prop := curve x y

-- Define the concept of a curve approaching a line
def approaches_line (f : ℝ → ℝ → Prop) (m b : ℝ) : Prop :=
  ∀ ε > 0, ∃ M > 0, ∀ x y, f x y → (|x| > M ∨ |y| > M) → |y - (m*x + b)| < ε

theorem curve_properties :
  symmetric_about_neg_x curve ∧
  point_on_curve (Real.rpow 3 (1/3 : ℝ)) 0 ∧
  point_on_curve 1 1 ∧
  point_on_curve 0 (Real.rpow 3 (1/3 : ℝ)) ∧
  approaches_line curve (-1) 0 :=
sorry

end curve_properties_l1255_125545


namespace solve_system_1_solve_system_2_l1255_125506

-- First system of equations
theorem solve_system_1 (x y : ℝ) : 
  x - y - 1 = 0 ∧ 4 * (x - y) - y = 0 → x = 5 ∧ y = 4 := by
  sorry

-- Second system of equations
theorem solve_system_2 (x y : ℝ) :
  3 * x - y - 2 = 0 ∧ (6 * x - 2 * y + 1) / 5 + 3 * y = 10 → x = 5 / 3 ∧ y = 3 := by
  sorry

end solve_system_1_solve_system_2_l1255_125506


namespace quadratic_roots_property_l1255_125518

theorem quadratic_roots_property : ∀ m n : ℝ, 
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ x = m ∨ x = n) → 
  m + n - m*n = 5 := by
  sorry

end quadratic_roots_property_l1255_125518


namespace round_37_259_to_thousandth_l1255_125534

/-- Represents a repeating decimal with a whole number part and a repeating fractional part. -/
structure RepeatingDecimal where
  wholePart : ℕ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest thousandth. -/
def roundToThousandth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The repeating decimal 37.259259... -/
def num : RepeatingDecimal :=
  { wholePart := 37, repeatingPart := 259 }

theorem round_37_259_to_thousandth :
  roundToThousandth num = 37259 / 1000 :=
sorry

end round_37_259_to_thousandth_l1255_125534


namespace circle_and_m_range_l1255_125509

-- Define the circle S
def circle_S (x y : ℝ) := (x - 4)^2 + (y - 4)^2 = 25

-- Define the line that contains the center of S
def center_line (x y : ℝ) := 2*x - y - 4 = 0

-- Define the intersecting line
def intersecting_line (x y m : ℝ) := x + y - m = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (7, 8)
def point_B : ℝ × ℝ := (8, 7)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem circle_and_m_range :
  ∀ (m : ℝ),
  (∃ (C D : ℝ × ℝ), 
    circle_S C.1 C.2 ∧ 
    circle_S D.1 D.2 ∧
    intersecting_line C.1 C.2 m ∧
    intersecting_line D.1 D.2 m ∧
    -- Angle COD is obtuse
    (C.1 * D.1 + C.2 * D.2 < 0)) →
  circle_S point_A.1 point_A.2 ∧
  circle_S point_B.1 point_B.2 ∧
  (∃ (center : ℝ × ℝ), center_line center.1 center.2 ∧ circle_S center.1 center.2) →
  1 < m ∧ m < 7 :=
sorry

end circle_and_m_range_l1255_125509


namespace inequality_reversal_l1255_125591

theorem inequality_reversal (x y : ℝ) (h : x < y) : ¬(-2 * x < -2 * y) := by
  sorry

end inequality_reversal_l1255_125591


namespace students_on_pullout_couch_l1255_125508

theorem students_on_pullout_couch (total_students : ℕ) (num_rooms : ℕ) (students_per_bed : ℕ) (beds_per_room : ℕ) :
  total_students = 30 →
  num_rooms = 6 →
  students_per_bed = 2 →
  beds_per_room = 2 →
  (total_students / num_rooms - students_per_bed * beds_per_room : ℕ) = 1 :=
by sorry

end students_on_pullout_couch_l1255_125508


namespace gunther_tractor_finance_l1255_125520

/-- Calculates the total amount financed for a loan with no interest -/
def total_financed (monthly_payment : ℕ) (payment_duration_years : ℕ) : ℕ :=
  monthly_payment * (payment_duration_years * 12)

/-- Theorem stating that for Gunther's tractor loan, the total financed amount is $9000 -/
theorem gunther_tractor_finance :
  total_financed 150 5 = 9000 := by
  sorry

end gunther_tractor_finance_l1255_125520


namespace comparison_square_and_power_l1255_125588

theorem comparison_square_and_power (n : ℕ) (h : n ≥ 3) : (n + 1)^2 < 3^n := by
  sorry

end comparison_square_and_power_l1255_125588


namespace simplify_and_evaluate_l1255_125519

theorem simplify_and_evaluate : 
  let x : ℝ := -2
  (2 * x + 1) * (x - 2) - (2 - x)^2 = -4 := by sorry

end simplify_and_evaluate_l1255_125519


namespace pen_price_calculation_l1255_125527

/-- Given the total number of pens purchased -/
def num_pens : ℕ := 30

/-- Given the total number of pencils purchased -/
def num_pencils : ℕ := 75

/-- Given the total cost of pens and pencils -/
def total_cost : ℝ := 750

/-- Given the average price of a pencil -/
def avg_price_pencil : ℝ := 2

/-- The average price of a pen -/
def avg_price_pen : ℝ := 20

theorem pen_price_calculation :
  (num_pens : ℝ) * avg_price_pen + (num_pencils : ℝ) * avg_price_pencil = total_cost :=
by sorry

end pen_price_calculation_l1255_125527


namespace frequency_limit_theorem_l1255_125583

/-- A fair coin toss experiment -/
structure CoinToss where
  /-- The number of tosses -/
  n : ℕ
  /-- The number of heads -/
  heads : ℕ
  /-- The number of heads is less than or equal to the number of tosses -/
  heads_le_n : heads ≤ n

/-- The frequency of heads in a coin toss experiment -/
def frequency (ct : CoinToss) : ℚ :=
  ct.heads / ct.n

/-- The limit of the frequency of heads as the number of tosses approaches infinity -/
theorem frequency_limit_theorem :
  ∀ ε > 0, ∃ N : ℕ, ∀ ct : CoinToss, ct.n ≥ N → |frequency ct - 1/2| < ε :=
sorry

end frequency_limit_theorem_l1255_125583


namespace stockholm_uppsala_distance_l1255_125536

/-- The scale of the map in km per cm -/
def map_scale : ℝ := 10

/-- The distance between Stockholm and Uppsala on the map in cm -/
def map_distance : ℝ := 45

/-- The actual distance between Stockholm and Uppsala in km -/
def actual_distance : ℝ := map_distance * map_scale

theorem stockholm_uppsala_distance :
  actual_distance = 450 := by sorry

end stockholm_uppsala_distance_l1255_125536


namespace area_of_rectangle_l1255_125572

/-- A square with two points on its sides forming a rectangle --/
structure SquareWithRectangle where
  -- Side length of the square
  side : ℝ
  -- Ratio of PT to PQ
  pt_ratio : ℝ
  -- Ratio of SU to SR
  su_ratio : ℝ
  -- Assumptions
  side_pos : 0 < side
  pt_ratio_pos : 0 < pt_ratio
  pt_ratio_lt_one : pt_ratio < 1
  su_ratio_pos : 0 < su_ratio
  su_ratio_lt_one : su_ratio < 1

/-- The perimeter of the rectangle PTUS --/
def rectangle_perimeter (s : SquareWithRectangle) : ℝ :=
  2 * (s.side * s.pt_ratio + s.side * s.su_ratio)

/-- The area of the rectangle PTUS --/
def rectangle_area (s : SquareWithRectangle) : ℝ :=
  (s.side * s.pt_ratio) * (s.side * s.su_ratio)

/-- Theorem: If PQRS is a square, T on PQ with PT:TQ = 1:2, U on SR with SU:UR = 1:2,
    and the perimeter of PTUS is 40 cm, then the area of PTUS is 75 cm² --/
theorem area_of_rectangle (s : SquareWithRectangle)
    (h_pt : s.pt_ratio = 1/3)
    (h_su : s.su_ratio = 1/3)
    (h_perimeter : rectangle_perimeter s = 40) :
    rectangle_area s = 75 := by
  sorry

end area_of_rectangle_l1255_125572


namespace hash_two_three_four_l1255_125552

/-- The # operation for real numbers -/
def hash (r s t : ℝ) : ℝ := r + s + t + r*s + r*t + s*t + r*s*t

/-- Theorem stating that 2 # 3 # 4 = 59 -/
theorem hash_two_three_four : hash 2 3 4 = 59 := by
  sorry

end hash_two_three_four_l1255_125552


namespace problem_proof_l1255_125599

theorem problem_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ x*y > a*b) → a*b ≤ 1/8 ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → 1/x + 8/y ≥ 25) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → x^2 + 4*y^2 ≥ 1/2) :=
by sorry

end problem_proof_l1255_125599


namespace total_apples_l1255_125597

/-- Represents the number of apples in a pack -/
def apples_per_pack : ℕ := 4

/-- Represents the number of packs bought -/
def packs_bought : ℕ := 2

/-- Theorem stating that buying 2 packs of 4 apples each results in 8 apples total -/
theorem total_apples : apples_per_pack * packs_bought = 8 := by
  sorry

end total_apples_l1255_125597


namespace johns_work_hours_l1255_125581

/-- Calculates the number of hours John works every other day given his wage, raise percentage, total earnings, and days in a month. -/
theorem johns_work_hours 
  (former_wage : ℝ)
  (raise_percentage : ℝ)
  (total_earnings : ℝ)
  (days_in_month : ℕ)
  (h1 : former_wage = 20)
  (h2 : raise_percentage = 30)
  (h3 : total_earnings = 4680)
  (h4 : days_in_month = 30) :
  let new_wage := former_wage * (1 + raise_percentage / 100)
  let working_days := days_in_month / 2
  let total_hours := total_earnings / new_wage
  let hours_per_working_day := total_hours / working_days
  hours_per_working_day = 12 := by sorry

end johns_work_hours_l1255_125581


namespace division_problem_l1255_125555

theorem division_problem (x y : ℕ+) 
  (h1 : x = 10 * y + 3)
  (h2 : 2 * x = 21 * y + 1) :
  11 * y - x = 2 := by
  sorry

end division_problem_l1255_125555


namespace range_of_x_when_a_is_neg_three_range_of_a_when_p_equiv_q_l1255_125553

-- Define the conditions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 6*x + 8 ≤ 0

-- Theorem for part (1)
theorem range_of_x_when_a_is_neg_three :
  ∀ x : ℝ, (p x (-3) ∧ q x) ↔ -4 ≤ x ∧ x < -3 :=
sorry

-- Theorem for part (2)
theorem range_of_a_when_p_equiv_q :
  ∀ a : ℝ, (∀ x : ℝ, p x a ↔ q x) ↔ -2 < a ∧ a < -4/3 :=
sorry

end range_of_x_when_a_is_neg_three_range_of_a_when_p_equiv_q_l1255_125553


namespace max_x_value_l1255_125525

theorem max_x_value : 
  ∃ (x_max : ℝ), 
    (∀ x : ℝ, ((5*x - 20)/(4*x - 5))^2 + ((5*x - 20)/(4*x - 5)) = 20 → x ≤ x_max) ∧
    ((5*x_max - 20)/(4*x_max - 5))^2 + ((5*x_max - 20)/(4*x_max - 5)) = 20 ∧
    x_max = 9/5 :=
by sorry

end max_x_value_l1255_125525


namespace bill_total_is_95_l1255_125537

/-- Represents a person's order at the restaurant -/
structure Order where
  appetizer_share : ℚ
  drinks_cost : ℚ
  dessert_cost : ℚ

/-- Calculates the total cost of an order -/
def total_cost (order : Order) : ℚ :=
  order.appetizer_share + order.drinks_cost + order.dessert_cost

/-- Represents the restaurant bill -/
def restaurant_bill (mary nancy fred steve : Order) : Prop :=
  let appetizer_total : ℚ := 28
  let appetizer_share : ℚ := appetizer_total / 4
  mary.appetizer_share = appetizer_share ∧
  nancy.appetizer_share = appetizer_share ∧
  fred.appetizer_share = appetizer_share ∧
  steve.appetizer_share = appetizer_share ∧
  mary.drinks_cost = 14 ∧
  nancy.drinks_cost = 11 ∧
  fred.drinks_cost = 12 ∧
  steve.drinks_cost = 6 ∧
  mary.dessert_cost = 8 ∧
  nancy.dessert_cost = 0 ∧
  fred.dessert_cost = 10 ∧
  steve.dessert_cost = 6

theorem bill_total_is_95 (mary nancy fred steve : Order) 
  (h : restaurant_bill mary nancy fred steve) : 
  total_cost mary + total_cost nancy + total_cost fred + total_cost steve = 95 := by
  sorry

end bill_total_is_95_l1255_125537


namespace binary_to_decimal_conversion_l1255_125548

/-- Converts a binary digit to its decimal value -/
def binaryToDecimal (digit : ℕ) (position : ℤ) : ℚ :=
  (digit : ℚ) * (2 : ℚ) ^ position

/-- Represents the binary number 111.11 -/
def binaryNumber : List (ℕ × ℤ) :=
  [(1, 2), (1, 1), (1, 0), (1, -1), (1, -2)]

/-- Theorem: The binary number 111.11 is equal to 7.75 in decimal -/
theorem binary_to_decimal_conversion :
  (binaryNumber.map (fun (digit, position) => binaryToDecimal digit position)).sum = 7.75 := by
  sorry

end binary_to_decimal_conversion_l1255_125548


namespace negation_of_forall_nonnegative_square_l1255_125529

theorem negation_of_forall_nonnegative_square (p : Prop) : 
  (p ↔ ∀ x : ℝ, x^2 ≥ 0) → (¬p ↔ ∃ x : ℝ, x^2 < 0) := by
  sorry

end negation_of_forall_nonnegative_square_l1255_125529


namespace is_projection_matrix_l1255_125512

def projection_matrix (M : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  M * M = M

theorem is_projection_matrix : 
  let M : Matrix (Fin 2) (Fin 2) ℚ := !![9/34, 25/34; 3/5, 15/34]
  projection_matrix M := by
  sorry

end is_projection_matrix_l1255_125512


namespace min_value_expression_l1255_125594

theorem min_value_expression (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ + x₂ = 1) :
  3 * x₁ / x₂ + 1 / (x₁ * x₂) ≥ 6 ∧ 
  ∃ x₁' x₂' : ℝ, x₁' > 0 ∧ x₂' > 0 ∧ x₁' + x₂' = 1 ∧ 3 * x₁' / x₂' + 1 / (x₁' * x₂') = 6 :=
by sorry

end min_value_expression_l1255_125594


namespace calculate_expression_l1255_125550

theorem calculate_expression : 3^2 * 7 + 5 * 4^2 - 45 / 3 = 128 := by
  sorry

end calculate_expression_l1255_125550


namespace four_digit_numbers_with_5_or_7_l1255_125538

def four_digit_numbers : ℕ := 9000

def digits_without_5_or_7 : ℕ := 8

def first_digit_options : ℕ := 7

def numbers_without_5_or_7 : ℕ := first_digit_options * (digits_without_5_or_7 ^ 3)

theorem four_digit_numbers_with_5_or_7 :
  four_digit_numbers - numbers_without_5_or_7 = 5416 :=
by sorry

end four_digit_numbers_with_5_or_7_l1255_125538


namespace tv_sales_decrease_l1255_125541

theorem tv_sales_decrease (original_price original_quantity : ℝ) 
  (price_increase : ℝ) (revenue_increase : ℝ) (sales_decrease : ℝ) :
  price_increase = 0.4 →
  revenue_increase = 0.12 →
  (1 + price_increase) * (1 - sales_decrease) = 1 + revenue_increase →
  sales_decrease = 0.2 := by
sorry

end tv_sales_decrease_l1255_125541


namespace range_of_a_l1255_125539

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (Real.exp x - a)^2 + x^2 - 2*a*x + a^2 ≤ 1/2) → a = 1/2 := by
  sorry

end range_of_a_l1255_125539


namespace profit_share_difference_example_l1255_125595

/-- Calculates the difference between profit shares of two partners given investments and one partner's profit share. -/
def profit_share_difference (inv_a inv_b inv_c b_profit : ℚ) : ℚ :=
  let total_inv := inv_a + inv_b + inv_c
  let total_profit := (total_inv / inv_b) * b_profit
  let a_share := (inv_a / total_inv) * total_profit
  let c_share := (inv_c / total_inv) * total_profit
  c_share - a_share

/-- Proves that the difference between profit shares of a and c is 600 given the specified investments and b's profit share. -/
theorem profit_share_difference_example : 
  profit_share_difference 8000 10000 12000 1500 = 600 := by
  sorry


end profit_share_difference_example_l1255_125595


namespace bricks_per_course_l1255_125585

/-- Proves that the number of bricks in each course is 400 --/
theorem bricks_per_course (initial_courses : ℕ) (added_courses : ℕ) (total_bricks : ℕ) :
  initial_courses = 3 →
  added_courses = 2 →
  total_bricks = 1800 →
  ∃ (bricks_per_course : ℕ),
    bricks_per_course * (initial_courses + added_courses) - bricks_per_course / 2 = total_bricks ∧
    bricks_per_course = 400 := by
  sorry

end bricks_per_course_l1255_125585


namespace motorcycle_speed_l1255_125573

/-- Motorcycle trip problem -/
theorem motorcycle_speed (total_distance : ℝ) (ab_distance : ℝ) (bc_distance : ℝ)
  (inclination_angle : ℝ) (total_avg_speed : ℝ) (ab_time_ratio : ℝ) :
  total_distance = ab_distance + bc_distance →
  bc_distance = ab_distance / 2 →
  ab_distance = 120 →
  inclination_angle = 10 →
  total_avg_speed = 30 →
  ab_time_ratio = 3 →
  ∃ (bc_avg_speed : ℝ), bc_avg_speed = 40 := by
  sorry

#check motorcycle_speed

end motorcycle_speed_l1255_125573


namespace sandys_age_l1255_125596

theorem sandys_age (sandy_age molly_age : ℕ) 
  (h1 : molly_age = sandy_age + 18) 
  (h2 : sandy_age * 9 = molly_age * 7) : 
  sandy_age = 63 := by
  sorry

end sandys_age_l1255_125596


namespace cone_volume_l1255_125546

/-- The volume of a cone with base diameter 12 cm and slant height 10 cm is 96π cubic centimeters -/
theorem cone_volume (π : ℝ) (diameter : ℝ) (slant_height : ℝ) : 
  diameter = 12 → slant_height = 10 → 
  (1 / 3) * π * ((diameter / 2) ^ 2) * (Real.sqrt (slant_height ^ 2 - (diameter / 2) ^ 2)) = 96 * π := by
  sorry


end cone_volume_l1255_125546


namespace anna_ate_14_apples_l1255_125513

def apples_tuesday : ℕ := 4

def apples_wednesday (tuesday : ℕ) : ℕ := 2 * tuesday

def apples_thursday (tuesday : ℕ) : ℕ := tuesday / 2

def total_apples (tuesday wednesday thursday : ℕ) : ℕ := 
  tuesday + wednesday + thursday

theorem anna_ate_14_apples : 
  total_apples apples_tuesday 
               (apples_wednesday apples_tuesday) 
               (apples_thursday apples_tuesday) = 14 := by
  sorry

end anna_ate_14_apples_l1255_125513


namespace sum_digits_first_1500_even_l1255_125587

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The sum of digits for all even numbers from 2 to n -/
def sum_digits_even (n : ℕ) : ℕ := sorry

/-- The 1500th positive even integer -/
def nth_even (n : ℕ) : ℕ := 2 * n

theorem sum_digits_first_1500_even :
  sum_digits_even (nth_even 1500) = 5448 := by sorry

end sum_digits_first_1500_even_l1255_125587


namespace product_equals_32_l1255_125565

theorem product_equals_32 : 
  (1 / 2) * 4 * (1 / 8) * 16 * (1 / 32) * 64 * (1 / 128) * 256 * (1 / 512) * 1024 = 32 := by
  sorry

end product_equals_32_l1255_125565


namespace mikeys_leaves_theorem_l1255_125530

/-- The number of leaves that blew away given initial and remaining leaf counts -/
def leaves_blown_away (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that 244 leaves blew away given the initial and remaining counts -/
theorem mikeys_leaves_theorem (initial : ℕ) (remaining : ℕ)
  (h1 : initial = 356)
  (h2 : remaining = 112) :
  leaves_blown_away initial remaining = 244 := by
sorry

end mikeys_leaves_theorem_l1255_125530


namespace inequality_solution_set_l1255_125556

theorem inequality_solution_set (x : ℝ) :
  (3 * x + 2 ≥ 5) ↔ (x ≥ 1) := by sorry

end inequality_solution_set_l1255_125556


namespace min_fourth_integer_l1255_125505

theorem min_fourth_integer (A B C D : ℕ+) : 
  (A + B + C + D : ℚ) / 4 = 16 →
  A = 3 * B →
  B = C - 2 →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  D ≥ 52 :=
by sorry

end min_fourth_integer_l1255_125505


namespace cubic_roots_relation_l1255_125551

theorem cubic_roots_relation (a b c r s t : ℝ) : 
  (∀ x, x^3 + 3*x^2 + 4*x - 11 = (x - a) * (x - b) * (x - c)) →
  (∀ x, x^3 + r*x^2 + s*x + t = (x - (a + b)) * (x - (b + c)) * (x - (c + a))) →
  t = 23 := by
sorry

end cubic_roots_relation_l1255_125551


namespace fraction_sum_l1255_125558

theorem fraction_sum : 3/8 + 9/12 = 9/8 := by
  sorry

end fraction_sum_l1255_125558


namespace eighth_term_is_one_thirty_second_l1255_125562

/-- The sequence defined by a_n = (-1)^n * n / 2^n -/
def a (n : ℕ) : ℚ := (-1)^n * n / 2^n

/-- The 8th term of the sequence is 1/32 -/
theorem eighth_term_is_one_thirty_second : a 8 = 1 / 32 := by
  sorry

end eighth_term_is_one_thirty_second_l1255_125562
