import Mathlib

namespace NUMINAMATH_CALUDE_sum_reciprocal_bound_l3428_342816

theorem sum_reciprocal_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a + b = 2) :
  c / a + c / b ≥ 2 * c ∧ ∀ ε > 0, ∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ c / a' + c / b' > ε := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_bound_l3428_342816


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l3428_342834

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ω.1 ∨ ω.2

-- Define the event of missing the target both times
def miss_both_times (ω : Ω) : Prop :=
  ¬ω.1 ∧ ¬ω.2

-- Theorem stating that the events are mutually exclusive
theorem mutually_exclusive_events :
  ∀ ω : Ω, ¬(hit_at_least_once ω ∧ miss_both_times ω) :=
by
  sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l3428_342834


namespace NUMINAMATH_CALUDE_bf_length_l3428_342839

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the conditions
variable (h1 : (A.1 = C.1 ∧ A.2 = D.2) ∧ (C.1 = D.1 ∧ C.2 = B.2))  -- right angles at A and C
variable (h2 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (1 - t) • A + t • C)  -- E is on AC
variable (h3 : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F = (1 - s) • A + s • C)  -- F is on AC
variable (h4 : (D.1 - E.1) * (C.1 - A.1) + (D.2 - E.2) * (C.2 - A.2) = 0)  -- DE perpendicular to AC
variable (h5 : (B.1 - F.1) * (C.1 - A.1) + (B.2 - F.2) * (C.2 - A.2) = 0)  -- BF perpendicular to AC
variable (h6 : Real.sqrt ((A.1 - E.1)^2 + (A.2 - E.2)^2) = 4)  -- AE = 4
variable (h7 : Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = 4)  -- DE = 4
variable (h8 : Real.sqrt ((C.1 - E.1)^2 + (C.2 - E.2)^2) = 6)  -- CE = 6

-- Theorem statement
theorem bf_length : Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_bf_length_l3428_342839


namespace NUMINAMATH_CALUDE_kangaroo_problem_l3428_342838

/-- Represents the number of days required to reach a target number of kangaroos -/
def daysToReach (initial : ℕ) (daily : ℕ) (target : ℕ) : ℕ :=
  if initial ≥ target then 0
  else ((target - initial) + (daily - 1)) / daily

theorem kangaroo_problem :
  let kameronKangaroos : ℕ := 100
  let bertInitial : ℕ := 20
  let bertDaily : ℕ := 2
  let christinaInitial : ℕ := 45
  let christinaDaily : ℕ := 3
  let davidInitial : ℕ := 10
  let davidDaily : ℕ := 5
  
  max (daysToReach bertInitial bertDaily kameronKangaroos)
      (max (daysToReach christinaInitial christinaDaily kameronKangaroos)
           (daysToReach davidInitial davidDaily kameronKangaroos)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_problem_l3428_342838


namespace NUMINAMATH_CALUDE_pascal_row_10_sum_l3428_342817

/-- The sum of numbers in a row of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

/-- Theorem: The sum of numbers in Row 10 of Pascal's Triangle is 1024 -/
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row_10_sum_l3428_342817


namespace NUMINAMATH_CALUDE_fraction_equality_l3428_342878

theorem fraction_equality (b : ℕ+) : 
  (b : ℚ) / ((b : ℚ) + 35) = 869 / 1000 → b = 232 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3428_342878


namespace NUMINAMATH_CALUDE_sports_club_membership_l3428_342846

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ)
  (h1 : total = 40)
  (h2 : badminton = 20)
  (h3 : tennis = 18)
  (h4 : both = 3) :
  total - (badminton + tennis - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_membership_l3428_342846


namespace NUMINAMATH_CALUDE_complex_number_properties_l3428_342826

variable (a : ℝ)
variable (b : ℝ)
def z : ℂ := a + Complex.I

theorem complex_number_properties :
  (∀ z, Complex.abs z = 1 → a = 0) ∧
  (∀ z, (z / (1 + Complex.I)).im = 0 → a = 1) ∧
  (∀ z b, z^2 + b*z + 2 = 0 → ((a = 1 ∧ b = -2) ∨ (a = -1 ∧ b = 2))) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3428_342826


namespace NUMINAMATH_CALUDE_negative_integer_sum_square_twelve_l3428_342873

theorem negative_integer_sum_square_twelve (M : ℤ) : 
  M < 0 → M^2 + M = 12 → M = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_sum_square_twelve_l3428_342873


namespace NUMINAMATH_CALUDE_hyperbola_other_asymptote_l3428_342833

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- X-coordinate of the foci -/
  foci_x : ℝ

/-- Represents the equation of a line in the form y = mx + b -/
structure LineEquation where
  m : ℝ
  b : ℝ

/-- The other asymptote of a hyperbola given one asymptote and the x-coordinate of the foci -/
def other_asymptote (h : Hyperbola) : LineEquation :=
  { m := -2, b := -16 }

theorem hyperbola_other_asymptote (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x ↦ 2 * x) 
  (h2 : h.foci_x = -4) : 
  other_asymptote h = { m := -2, b := -16 } := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_other_asymptote_l3428_342833


namespace NUMINAMATH_CALUDE_max_value_expression_l3428_342845

theorem max_value_expression (a b c d : ℝ) 
  (ha : -7.5 ≤ a ∧ a ≤ 7.5)
  (hb : -7.5 ≤ b ∧ b ≤ 7.5)
  (hc : -7.5 ≤ c ∧ c ≤ 7.5)
  (hd : -7.5 ≤ d ∧ d ≤ 7.5) :
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 240 ∧ 
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 
    (-7.5 ≤ a₀ ∧ a₀ ≤ 7.5) ∧
    (-7.5 ≤ b₀ ∧ b₀ ≤ 7.5) ∧
    (-7.5 ≤ c₀ ∧ c₀ ≤ 7.5) ∧
    (-7.5 ≤ d₀ ∧ d₀ ≤ 7.5) ∧
    (a₀ + 2*b₀ + c₀ + 2*d₀ - a₀*b₀ - b₀*c₀ - c₀*d₀ - d₀*a₀) = 240 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3428_342845


namespace NUMINAMATH_CALUDE_bills_steps_correct_l3428_342860

/-- The length of each step Bill takes, in metres -/
def step_length : ℚ := 1/2

/-- The total distance Bill walks, in metres -/
def total_distance : ℚ := 12

/-- The number of steps Bill takes to walk the total distance -/
def number_of_steps : ℕ := 24

/-- Theorem stating that the number of steps Bill takes is correct -/
theorem bills_steps_correct : 
  (step_length * number_of_steps : ℚ) = total_distance :=
by sorry

end NUMINAMATH_CALUDE_bills_steps_correct_l3428_342860


namespace NUMINAMATH_CALUDE_fridge_cost_difference_l3428_342814

theorem fridge_cost_difference (total_budget : ℕ) (tv_cost : ℕ) (computer_cost : ℕ) 
  (h1 : total_budget = 1600)
  (h2 : tv_cost = 600)
  (h3 : computer_cost = 250)
  (h4 : ∃ fridge_cost : ℕ, fridge_cost > computer_cost ∧ 
        fridge_cost + tv_cost + computer_cost = total_budget) :
  ∃ fridge_cost : ℕ, fridge_cost - computer_cost = 500 := by
sorry

end NUMINAMATH_CALUDE_fridge_cost_difference_l3428_342814


namespace NUMINAMATH_CALUDE_correct_average_l3428_342895

theorem correct_average (numbers : Finset ℕ) (incorrect_sum : ℕ) (incorrect_number correct_number : ℕ) :
  numbers.card = 10 →
  incorrect_sum / numbers.card = 19 →
  incorrect_number = 26 →
  correct_number = 76 →
  (incorrect_sum - incorrect_number + correct_number) / numbers.card = 24 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l3428_342895


namespace NUMINAMATH_CALUDE_negative_cube_squared_l3428_342806

theorem negative_cube_squared (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l3428_342806


namespace NUMINAMATH_CALUDE_percentage_of_sum_l3428_342866

theorem percentage_of_sum (x y : ℝ) (P : ℝ) : 
  (0.5 * (x - y) = (P / 100) * (x + y)) → 
  (y = 0.42857142857142854 * x) → 
  (P = 20) := by
sorry

end NUMINAMATH_CALUDE_percentage_of_sum_l3428_342866


namespace NUMINAMATH_CALUDE_symmetry_implies_a_equals_one_l3428_342823

/-- A function f: ℝ → ℝ is symmetric about the line x = c if f(c + t) = f(c - t) for all t ∈ ℝ -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ t, f (c + t) = f (c - t)

/-- The main theorem: If sin x + a cos x is symmetric about x = π/4, then a = 1 -/
theorem symmetry_implies_a_equals_one (a : ℝ) :
  SymmetricAbout (fun x ↦ Real.sin x + a * Real.cos x) (π/4) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_a_equals_one_l3428_342823


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_l3428_342803

theorem integral_sqrt_one_minus_x_squared (f : ℝ → ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f x = Real.sqrt (1 - x^2)) →
  (∫ x in Set.Icc (-1) 1, f x) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_l3428_342803


namespace NUMINAMATH_CALUDE_components_upper_bound_l3428_342801

/-- Represents a square grid with diagonals --/
structure DiagonalGrid (n : ℕ) where
  size : n > 8
  cells : Fin n → Fin n → Bool
  -- True represents one diagonal, False represents the other

/-- Counts the number of connected components in the grid --/
def countComponents (g : DiagonalGrid n) : ℕ := sorry

/-- Theorem stating that the number of components is not greater than n²/4 --/
theorem components_upper_bound (n : ℕ) (g : DiagonalGrid n) :
  countComponents g ≤ n^2 / 4 := by sorry

end NUMINAMATH_CALUDE_components_upper_bound_l3428_342801


namespace NUMINAMATH_CALUDE_community_families_count_l3428_342885

theorem community_families_count :
  let families_with_two_dogs : ℕ := 15
  let families_with_one_dog : ℕ := 20
  let total_animals : ℕ := 80
  let total_dogs : ℕ := families_with_two_dogs * 2 + families_with_one_dog
  let total_cats : ℕ := total_animals - total_dogs
  let families_with_cats : ℕ := total_cats / 2
  families_with_two_dogs + families_with_one_dog + families_with_cats = 50 :=
by sorry

end NUMINAMATH_CALUDE_community_families_count_l3428_342885


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_l3428_342804

-- Define the geometric sequences and their properties
def geometric_sequence (k a₂ a₃ : ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 1 ∧ a₂ = k * p ∧ a₃ = k * p^2

-- Theorem statement
theorem sum_of_common_ratios 
  (k a₂ a₃ b₂ b₃ : ℝ) 
  (h₁ : geometric_sequence k a₂ a₃)
  (h₂ : geometric_sequence k b₂ b₃)
  (h₃ : ∃ p r : ℝ, p ≠ r ∧ 
    a₂ = k * p ∧ a₃ = k * p^2 ∧ 
    b₂ = k * r ∧ b₃ = k * r^2)
  (h₄ : a₃ - b₃ = 3 * (a₂ - b₂))
  : ∃ p r : ℝ, p + r = 3 ∧ 
    a₂ = k * p ∧ a₃ = k * p^2 ∧ 
    b₂ = k * r ∧ b₃ = k * r^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_l3428_342804


namespace NUMINAMATH_CALUDE_cylinder_intersection_area_l3428_342871

/-- Represents a cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Represents the area of a surface formed by intersecting a cylinder with a plane --/
def intersectionArea (c : Cylinder) (arcAngle : ℝ) : ℝ := sorry

theorem cylinder_intersection_area :
  let c : Cylinder := { radius := 7, height := 9 }
  let arcAngle : ℝ := 150 * (π / 180)  -- Convert degrees to radians
  intersectionArea c arcAngle = 62.4 * π + 112 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_cylinder_intersection_area_l3428_342871


namespace NUMINAMATH_CALUDE_dianas_biking_distance_l3428_342811

/-- Diana's biking problem -/
theorem dianas_biking_distance
  (initial_speed : ℝ)
  (initial_time : ℝ)
  (tired_speed : ℝ)
  (total_time : ℝ)
  (h1 : initial_speed = 3)
  (h2 : initial_time = 2)
  (h3 : tired_speed = 1)
  (h4 : total_time = 6) :
  initial_speed * initial_time + tired_speed * (total_time - initial_time) = 10 :=
by sorry

end NUMINAMATH_CALUDE_dianas_biking_distance_l3428_342811


namespace NUMINAMATH_CALUDE_tangent_line_at_2_monotonicity_intervals_l3428_342896

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x

-- Theorem for the tangent line equation
theorem tangent_line_at_2 :
  ∃ (m c : ℝ), ∀ x y, y = m * (x - 2) + f 2 ↔ 12 * x - y - 17 = 0 := by sorry

-- Theorem for intervals of monotonicity
theorem monotonicity_intervals :
  (∀ x, x < 0 → (f' x > 0)) ∧
  (∀ x, 0 < x ∧ x < 1 → (f' x < 0)) ∧
  (∀ x, x > 1 → (f' x > 0)) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_monotonicity_intervals_l3428_342896


namespace NUMINAMATH_CALUDE_angle_complement_supplement_l3428_342868

theorem angle_complement_supplement (x : ℝ) : 
  (90 - x) = 4 * (180 - x) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_supplement_l3428_342868


namespace NUMINAMATH_CALUDE_expansion_and_reduction_l3428_342837

theorem expansion_and_reduction : 
  (234 * 205 = 47970) ∧ (86400 / 300 = 288) := by
  sorry

end NUMINAMATH_CALUDE_expansion_and_reduction_l3428_342837


namespace NUMINAMATH_CALUDE_salary_left_unspent_l3428_342880

/-- The fraction of salary spent in the first week -/
def first_week_spending : ℚ := 1/4

/-- The fraction of salary spent in each of the following three weeks -/
def other_weeks_spending : ℚ := 1/5

/-- The number of weeks after the first week -/
def remaining_weeks : ℕ := 3

/-- Theorem: Given the spending conditions, the fraction of salary left unspent at the end of the month is 3/20 -/
theorem salary_left_unspent :
  1 - (first_week_spending + remaining_weeks * other_weeks_spending) = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_salary_left_unspent_l3428_342880


namespace NUMINAMATH_CALUDE_candies_left_l3428_342886

def candies_bought_tuesday : ℕ := 3
def candies_bought_thursday : ℕ := 5
def candies_bought_friday : ℕ := 2
def candies_eaten : ℕ := 6

theorem candies_left : 
  candies_bought_tuesday + candies_bought_thursday + candies_bought_friday - candies_eaten = 4 := by
  sorry

end NUMINAMATH_CALUDE_candies_left_l3428_342886


namespace NUMINAMATH_CALUDE_electricity_scientific_notation_l3428_342864

-- Define the number of kilowatt-hours
def electricity_delivered : ℝ := 105.9e9

-- Theorem to prove the scientific notation
theorem electricity_scientific_notation :
  electricity_delivered = 1.059 * (10 : ℝ)^10 := by
  sorry

end NUMINAMATH_CALUDE_electricity_scientific_notation_l3428_342864


namespace NUMINAMATH_CALUDE_complete_square_with_integer_l3428_342852

theorem complete_square_with_integer (y : ℝ) : 
  ∃ (k : ℤ) (a : ℝ), y^2 + 12*y + 44 = (y + a)^2 + k ∧ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_with_integer_l3428_342852


namespace NUMINAMATH_CALUDE_intersection_distance_l3428_342818

/-- Given ω > 0, if the distance between the two closest intersection points 
    of y = 4sin(ωx) and y = 4cos(ωx) is 6, then ω = π/2 -/
theorem intersection_distance (ω : Real) (h1 : ω > 0) : 
  (∃ x₁ x₂ : Real, 
    x₁ ≠ x₂ ∧ 
    4 * Real.sin (ω * x₁) = 4 * Real.cos (ω * x₁) ∧
    4 * Real.sin (ω * x₂) = 4 * Real.cos (ω * x₂) ∧
    ∀ x : Real, 4 * Real.sin (ω * x) = 4 * Real.cos (ω * x) → 
      (x = x₁ ∨ x = x₂ ∨ |x - x₁| ≥ |x₁ - x₂| ∧ |x - x₂| ≥ |x₁ - x₂|) ∧
    (x₁ - x₂)^2 = 36) →
  ω = π / 2 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_l3428_342818


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3428_342882

theorem partial_fraction_decomposition (M₁ M₂ : ℚ) : 
  (∀ x : ℚ, x ≠ 1 → x ≠ 3 → (45 * x - 34) / (x^2 - 4*x + 3) = M₁ / (x - 1) + M₂ / (x - 3)) →
  M₁ * M₂ = -1111 / 4 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3428_342882


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3428_342802

theorem election_votes_theorem (candidates : ℕ) (winner_percentage : ℝ) (majority : ℕ) 
  (h1 : candidates = 4)
  (h2 : winner_percentage = 0.7)
  (h3 : majority = 3000) :
  ∃ total_votes : ℕ, 
    (↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = majority) ∧ 
    total_votes = 7500 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l3428_342802


namespace NUMINAMATH_CALUDE_distance_is_35_over_13_l3428_342805

def point : ℝ × ℝ × ℝ := (0, -1, 4)
def line_point : ℝ × ℝ × ℝ := (-3, 2, 5)
def line_direction : ℝ × ℝ × ℝ := (4, 1, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ := 
  sorry

theorem distance_is_35_over_13 : 
  distance_to_line point line_point line_direction = 35 / 13 := by sorry

end NUMINAMATH_CALUDE_distance_is_35_over_13_l3428_342805


namespace NUMINAMATH_CALUDE_paint_calculation_l3428_342875

theorem paint_calculation (total_paint : ℚ) : 
  (1/4 * total_paint + 1/3 * (3/4 * total_paint) = 180) → total_paint = 360 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l3428_342875


namespace NUMINAMATH_CALUDE_taller_tree_height_l3428_342843

theorem taller_tree_height (h₁ h₂ : ℝ) : 
  h₂ = h₁ + 20 →  -- One tree is 20 feet taller than the other
  h₁ / h₂ = 5 / 7 →  -- The heights are in the ratio 5:7
  h₂ = 70 :=  -- The height of the taller tree is 70 feet
by sorry

end NUMINAMATH_CALUDE_taller_tree_height_l3428_342843


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3428_342820

theorem quadratic_equation_solution :
  ∃! x : ℚ, x > 0 ∧ 3 * x^2 + 11 * x - 20 = 0 :=
by
  use 4/3
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3428_342820


namespace NUMINAMATH_CALUDE_intersection_M_N_l3428_342867

-- Define set M
def M : Set ℝ := {x | x^2 + 2*x - 3 = 0}

-- Define set N
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (2^x - 1/2)}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3428_342867


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l3428_342883

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  averageBeforeLastInning : Rat
  lastInningScore : Nat
  averageIncrease : Rat

/-- Calculates the new average after the last inning -/
def newAverage (stats : BatsmanStats) : Rat :=
  (stats.totalRuns + stats.lastInningScore) / stats.innings

/-- Theorem: Given the conditions, prove that the new average is 23 -/
theorem batsman_average_theorem (stats : BatsmanStats) 
  (h1 : stats.innings = 17)
  (h2 : stats.lastInningScore = 87)
  (h3 : stats.averageIncrease = 4)
  (h4 : newAverage stats = stats.averageBeforeLastInning + stats.averageIncrease) :
  newAverage stats = 23 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_theorem_l3428_342883


namespace NUMINAMATH_CALUDE_base_10_to_7_2023_l3428_342844

/-- Converts a base 10 number to base 7 --/
def toBase7 (n : Nat) : List Nat :=
  sorry

/-- Converts a list of digits in base 7 to a natural number --/
def fromBase7 (digits : List Nat) : Nat :=
  sorry

theorem base_10_to_7_2023 :
  toBase7 2023 = [5, 6, 2, 0] ∧ fromBase7 [5, 6, 2, 0] = 2023 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_7_2023_l3428_342844


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l3428_342832

open Real

theorem max_value_trig_expression :
  ∃ (M : ℝ), M = Real.sqrt 10 + 3 ∧
  (∀ x : ℝ, cos x + 3 * sin x + tan x ≤ M) ∧
  (∃ x : ℝ, cos x + 3 * sin x + tan x = M) := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l3428_342832


namespace NUMINAMATH_CALUDE_distance_to_origin_of_complex_number_l3428_342825

theorem distance_to_origin_of_complex_number : ∃ (z : ℂ), 
  z = (2 * Complex.I) / (1 - Complex.I) ∧ 
  Complex.abs z = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_distance_to_origin_of_complex_number_l3428_342825


namespace NUMINAMATH_CALUDE_determinant_trig_matrix_l3428_342841

theorem determinant_trig_matrix (α β : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.cos (α + β), Real.sin (α + β), -Real.sin α],
    ![-Real.sin β, Real.cos β, 0],
    ![Real.sin α * Real.cos β, Real.sin α * Real.sin β, Real.cos α]
  ]
  Matrix.det M = 1 := by sorry

end NUMINAMATH_CALUDE_determinant_trig_matrix_l3428_342841


namespace NUMINAMATH_CALUDE_increase_by_percentage_l3428_342877

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 50 → percentage = 120 → result = initial * (1 + percentage / 100) → result = 110 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l3428_342877


namespace NUMINAMATH_CALUDE_additional_money_needed_l3428_342819

def phone_cost : ℝ := 1300
def percentage_owned : ℝ := 40

theorem additional_money_needed : 
  phone_cost - (percentage_owned / 100) * phone_cost = 780 := by
  sorry

end NUMINAMATH_CALUDE_additional_money_needed_l3428_342819


namespace NUMINAMATH_CALUDE_ratio_squares_sum_l3428_342830

theorem ratio_squares_sum (a b c : ℝ) : 
  a / b = 3 / 2 ∧ 
  c / b = 5 / 2 ∧ 
  b = 14 → 
  a^2 + b^2 + c^2 = 1862 := by
sorry

end NUMINAMATH_CALUDE_ratio_squares_sum_l3428_342830


namespace NUMINAMATH_CALUDE_park_area_is_102400_l3428_342815

/-- Represents a rectangular park -/
structure RectangularPark where
  length : ℝ
  breadth : ℝ
  ratio : length = 4 * breadth

/-- Calculates the perimeter of the park -/
def perimeter (park : RectangularPark) : ℝ :=
  2 * (park.length + park.breadth)

/-- Calculates the area of the park -/
def area (park : RectangularPark) : ℝ :=
  park.length * park.breadth

/-- Theorem: The area of the park is 102400 square meters -/
theorem park_area_is_102400 (park : RectangularPark) 
    (h_perimeter : perimeter park = 12 * 8 / 60 * 1000) : 
    area park = 102400 := by
  sorry


end NUMINAMATH_CALUDE_park_area_is_102400_l3428_342815


namespace NUMINAMATH_CALUDE_student_walking_speed_l3428_342888

/-- 
Given two students walking towards each other:
- They start 350 meters apart
- They walk for 100 seconds until they meet
- The first student walks at 1.6 m/s
Prove that the second student's speed is 1.9 m/s
-/
theorem student_walking_speed 
  (initial_distance : ℝ) 
  (time : ℝ) 
  (speed1 : ℝ) 
  (h1 : initial_distance = 350)
  (h2 : time = 100)
  (h3 : speed1 = 1.6) :
  ∃ speed2 : ℝ, 
    speed2 = 1.9 ∧ 
    speed1 * time + speed2 * time = initial_distance := by
  sorry

end NUMINAMATH_CALUDE_student_walking_speed_l3428_342888


namespace NUMINAMATH_CALUDE_last_number_in_sequence_l3428_342851

theorem last_number_in_sequence (x : ℕ) : 
  1000 + 20 + 1000 + 30 + 1000 + 40 + 1000 + x = 4100 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_last_number_in_sequence_l3428_342851


namespace NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l3428_342874

theorem sqrt_eight_div_sqrt_two_equals_two :
  Real.sqrt 8 / Real.sqrt 2 = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l3428_342874


namespace NUMINAMATH_CALUDE_min_stamps_for_39_cents_l3428_342828

theorem min_stamps_for_39_cents : 
  ∃ (c f : ℕ), 3 * c + 5 * f = 39 ∧ 
  c + f = 9 ∧ 
  ∀ (c' f' : ℕ), 3 * c' + 5 * f' = 39 → c + f ≤ c' + f' :=
by sorry

end NUMINAMATH_CALUDE_min_stamps_for_39_cents_l3428_342828


namespace NUMINAMATH_CALUDE_dans_seashells_l3428_342869

def seashells_problem (initial_seashells : ℕ) (remaining_seashells : ℕ) : Prop :=
  initial_seashells ≥ remaining_seashells →
  ∃ (given_seashells : ℕ), given_seashells = initial_seashells - remaining_seashells

theorem dans_seashells : seashells_problem 56 22 := by
  sorry

end NUMINAMATH_CALUDE_dans_seashells_l3428_342869


namespace NUMINAMATH_CALUDE_cards_per_student_l3428_342893

/-- Given that Joseph had 357 cards initially, has 15 students, and had 12 cards left after distribution,
    prove that the number of cards given to each student is 23. -/
theorem cards_per_student (total_cards : Nat) (num_students : Nat) (remaining_cards : Nat)
    (h1 : total_cards = 357)
    (h2 : num_students = 15)
    (h3 : remaining_cards = 12) :
    (total_cards - remaining_cards) / num_students = 23 :=
by sorry

end NUMINAMATH_CALUDE_cards_per_student_l3428_342893


namespace NUMINAMATH_CALUDE_op_theorem_l3428_342857

/-- The type representing elements in our set -/
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

/-- The operation ⊕ -/
def op : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.four

theorem op_theorem : 
  op (op Element.four Element.one) (op Element.two Element.three) = Element.three :=
by sorry

end NUMINAMATH_CALUDE_op_theorem_l3428_342857


namespace NUMINAMATH_CALUDE_F_r_properties_l3428_342842

/-- Represents a point in the cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the polygon F_r -/
def F_r (r : ℝ) : Set Point :=
  {p : Point | p.x^2 + p.y^2 = r^2 ∧ (p.x * p.y)^2 = 1}

/-- The area of the polygon F_r as a function of r -/
noncomputable def area (r : ℝ) : ℝ :=
  sorry

/-- Predicate to check if a polygon is regular -/
def is_regular (s : Set Point) : Prop :=
  sorry

theorem F_r_properties :
  ∃ (A : ℝ → ℝ),
    (∀ r, A r = area r) ∧
    is_regular (F_r 1) ∧
    ∀ r > 1, is_regular (F_r r) := by
  sorry

end NUMINAMATH_CALUDE_F_r_properties_l3428_342842


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l3428_342807

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the origin
def origin : Point := (0, 0)

-- Define the given point
def given_point : Point := (2, -6)

-- Theorem stating that the coordinates of the given point with respect to the origin are (2, -6)
theorem coordinates_wrt_origin (p : Point) : p = given_point → p = (2, -6) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l3428_342807


namespace NUMINAMATH_CALUDE_unique_prime_for_equiangular_polygons_l3428_342870

theorem unique_prime_for_equiangular_polygons :
  ∃! k : ℕ, 
    Prime k ∧ 
    k > 1 ∧
    ∃ (x n₁ n₂ : ℕ),
      -- Angle formula for P1
      x = 180 - 360 / n₁ ∧ 
      -- Angle formula for P2
      k * x = 180 - 360 / n₂ ∧ 
      -- Angles must be positive and less than 180°
      0 < x ∧ x < 180 ∧
      0 < k * x ∧ k * x < 180 ∧
      -- Number of sides must be at least 3
      n₁ ≥ 3 ∧ n₂ ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_for_equiangular_polygons_l3428_342870


namespace NUMINAMATH_CALUDE_sum_removal_proof_l3428_342897

theorem sum_removal_proof : 
  let original_sum := (1/2 : ℚ) + 1/3 + 1/4 + 1/6 + 1/8 + 1/9 + 1/12
  let removed_terms := (1/8 : ℚ) + 1/9
  original_sum - removed_terms = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_removal_proof_l3428_342897


namespace NUMINAMATH_CALUDE_square_perimeter_l3428_342827

theorem square_perimeter (A : ℝ) (h : A = 625) :
  2 * (4 * Real.sqrt A) = 200 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_l3428_342827


namespace NUMINAMATH_CALUDE_pear_weight_proof_l3428_342854

/-- The weight of one pear in grams -/
def pear_weight : ℝ := 120

theorem pear_weight_proof :
  let apple_weight : ℝ := 530
  let apple_count : ℕ := 12
  let pear_count : ℕ := 8
  let weight_difference : ℝ := 5400
  apple_count * apple_weight = pear_count * pear_weight + weight_difference →
  pear_weight = 120 := by
sorry

end NUMINAMATH_CALUDE_pear_weight_proof_l3428_342854


namespace NUMINAMATH_CALUDE_complex_vector_sum_l3428_342809

theorem complex_vector_sum (z₁ z₂ z₃ : ℂ) (x y : ℝ) :
  z₁ = -1 + 2*Complex.I →
  z₂ = 1 - Complex.I →
  z₃ = 3 - 2*Complex.I →
  z₃ = x * z₁ + y * z₂ →
  x + y = 5 := by sorry

end NUMINAMATH_CALUDE_complex_vector_sum_l3428_342809


namespace NUMINAMATH_CALUDE_arithmetic_sequence_tangent_l3428_342899

/-- Given an arithmetic sequence {a_n} where S_n is the sum of its first n terms,
    prove that if S_11 = 22π/3, then tan(a_6) = -√3 -/
theorem arithmetic_sequence_tangent (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of S_n for arithmetic sequence
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Definition of arithmetic sequence
  S 11 = 22 * Real.pi / 3 →             -- Given condition
  Real.tan (a 6) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_tangent_l3428_342899


namespace NUMINAMATH_CALUDE_product_of_smallest_primes_l3428_342821

def smallest_one_digit_primes : List Nat := [2, 3]
def smallest_two_digit_prime : Nat := 11

theorem product_of_smallest_primes :
  (smallest_one_digit_primes.prod) * smallest_two_digit_prime = 66 := by
  sorry

end NUMINAMATH_CALUDE_product_of_smallest_primes_l3428_342821


namespace NUMINAMATH_CALUDE_initial_students_count_l3428_342853

/-- The number of students at the start of the year. -/
def initial_students : ℕ := sorry

/-- The number of students who left during the year. -/
def students_left : ℕ := 5

/-- The number of new students who came during the year. -/
def new_students : ℕ := 8

/-- The number of students at the end of the year. -/
def final_students : ℕ := 11

/-- Theorem stating that the initial number of students is 8. -/
theorem initial_students_count :
  initial_students = final_students - (new_students - students_left) := by sorry

end NUMINAMATH_CALUDE_initial_students_count_l3428_342853


namespace NUMINAMATH_CALUDE_line_of_symmetry_l3428_342861

/-- Definition of circle O -/
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Definition of circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 4 = 0

/-- Definition of line l -/
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

/-- Theorem stating that line l is the line of symmetry for circles O and C -/
theorem line_of_symmetry :
  ∀ (x y : ℝ), line_l x y → (∃ (x' y' : ℝ), circle_O x' y' ∧ circle_C x y ∧
    x' = 2*x - x ∧ y' = 2*y - y) :=
sorry

end NUMINAMATH_CALUDE_line_of_symmetry_l3428_342861


namespace NUMINAMATH_CALUDE_mrs_blue_tomato_yield_l3428_342863

/-- Represents the dimensions of a rectangular vegetable patch in steps -/
structure PatchDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected tomato yield from a vegetable patch -/
def expected_tomato_yield (dimensions : PatchDimensions) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  (dimensions.length : ℝ) * step_length * (dimensions.width : ℝ) * step_length * yield_per_sqft

/-- Theorem stating the expected tomato yield for Mrs. Blue's vegetable patch -/
theorem mrs_blue_tomato_yield :
  let dimensions : PatchDimensions := ⟨18, 25⟩
  let step_length : ℝ := 3
  let yield_per_sqft : ℝ := 3 / 4
  expected_tomato_yield dimensions step_length yield_per_sqft = 3037.5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_blue_tomato_yield_l3428_342863


namespace NUMINAMATH_CALUDE_temperature_difference_l3428_342892

theorem temperature_difference (highest lowest : ℤ) (h1 : highest = 12) (h2 : lowest = -1) :
  highest - lowest = 13 := by sorry

end NUMINAMATH_CALUDE_temperature_difference_l3428_342892


namespace NUMINAMATH_CALUDE_element_in_two_pairs_l3428_342855

/-- A system of elements and pairs satisfying the given conditions -/
structure PairSystem (n : ℕ) where
  -- The set of elements
  elements : Fin n → Type
  -- The set of pairs
  pairs : Fin n → Set (Fin n)
  -- Two pairs share exactly one element iff they form a pair
  share_condition : ∀ i j : Fin n, 
    (∃! k : Fin n, k ∈ pairs i ∧ k ∈ pairs j) ↔ j ∈ pairs i

/-- Every element is in exactly two pairs -/
theorem element_in_two_pairs {n : ℕ} (sys : PairSystem n) :
  ∀ k : Fin n, ∃! (i j : Fin n), i ≠ j ∧ k ∈ sys.pairs i ∧ k ∈ sys.pairs j :=
sorry

end NUMINAMATH_CALUDE_element_in_two_pairs_l3428_342855


namespace NUMINAMATH_CALUDE_square_difference_problem_l3428_342812

theorem square_difference_problem (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) :
  |x^2 - y^2| = 108 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_problem_l3428_342812


namespace NUMINAMATH_CALUDE_total_wheels_at_station_l3428_342858

/-- The number of trains at the station -/
def num_trains : ℕ := 4

/-- The number of carriages per train -/
def carriages_per_train : ℕ := 4

/-- The number of wheel rows per carriage -/
def wheel_rows_per_carriage : ℕ := 3

/-- The number of wheels per row -/
def wheels_per_row : ℕ := 5

/-- The total number of wheels at the train station -/
def total_wheels : ℕ := num_trains * carriages_per_train * wheel_rows_per_carriage * wheels_per_row

theorem total_wheels_at_station :
  total_wheels = 240 :=
by sorry

end NUMINAMATH_CALUDE_total_wheels_at_station_l3428_342858


namespace NUMINAMATH_CALUDE_odd_function_with_period_4_sum_l3428_342898

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_with_period_4_sum (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) (h_period : has_period f 4) :
  f 2005 + f 2006 + f 2007 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_with_period_4_sum_l3428_342898


namespace NUMINAMATH_CALUDE_simplify_cube_roots_l3428_342829

theorem simplify_cube_roots : 
  (1 + 27) ^ (1/3) * (1 + 27 ^ (1/3)) ^ (1/3) = 112 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_roots_l3428_342829


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3428_342836

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence condition
  (a 1 + a 2 = 16) →                        -- first given condition
  (a 3 + a 4 = 24) →                        -- second given condition
  (a 7 + a 8 = 54) :=                       -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3428_342836


namespace NUMINAMATH_CALUDE_derivative_f_at_neg_one_l3428_342859

noncomputable def f (x : ℝ) : ℝ := (1 + x) * (2 + x^2)^(1/2) * (3 + x^3)^(1/3)

theorem derivative_f_at_neg_one :
  deriv f (-1) = Real.sqrt 3 * 2^(1/3) :=
sorry

end NUMINAMATH_CALUDE_derivative_f_at_neg_one_l3428_342859


namespace NUMINAMATH_CALUDE_f_not_mapping_l3428_342884

-- Define set A
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

-- Define set B
def B : Set ℝ := {y : ℝ | 0 ≤ y ∧ y ≤ 8}

-- Define the correspondence rule f
def f (x : ℝ) : ℝ := 4

-- Theorem stating that f is not a mapping from A to B
theorem f_not_mapping : ¬(∀ x ∈ A, f x ∈ B) :=
sorry

end NUMINAMATH_CALUDE_f_not_mapping_l3428_342884


namespace NUMINAMATH_CALUDE_quadratic_sum_of_constants_l3428_342881

theorem quadratic_sum_of_constants (x : ℝ) : ∃ (a b c : ℝ),
  (6 * x^2 + 48 * x + 162 = a * (x + b)^2 + c) ∧ (a + b + c = 76) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_constants_l3428_342881


namespace NUMINAMATH_CALUDE_league_games_and_weeks_l3428_342847

/-- Represents a sports league --/
structure League where
  num_teams : ℕ
  games_per_week : ℕ

/-- Calculates the total number of games in a round-robin tournament --/
def total_games (league : League) : ℕ :=
  league.num_teams * (league.num_teams - 1) / 2

/-- Calculates the minimum number of weeks required to complete all games --/
def min_weeks (league : League) : ℕ :=
  (total_games league + league.games_per_week - 1) / league.games_per_week

/-- Theorem about the number of games and weeks in a specific league --/
theorem league_games_and_weeks :
  let league := League.mk 15 7
  total_games league = 105 ∧ min_weeks league = 15 := by
  sorry


end NUMINAMATH_CALUDE_league_games_and_weeks_l3428_342847


namespace NUMINAMATH_CALUDE_accounting_majors_l3428_342813

theorem accounting_majors (p q r s t u : ℕ) : 
  p * q * r * s * t * u = 51030 → 
  1 < p → p < q → q < r → r < s → s < t → t < u → 
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_accounting_majors_l3428_342813


namespace NUMINAMATH_CALUDE_prob_exceeds_175_l3428_342889

/-- The probability that a randomly selected student's height is less than 160cm -/
def prob_less_than_160 : ℝ := 0.2

/-- The probability that a randomly selected student's height is between 160cm and 175cm -/
def prob_between_160_and_175 : ℝ := 0.5

/-- Theorem: Given the probabilities of a student's height being less than 160cm and between 160cm and 175cm,
    the probability of a student's height exceeding 175cm is 0.3 -/
theorem prob_exceeds_175 : 1 - (prob_less_than_160 + prob_between_160_and_175) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_prob_exceeds_175_l3428_342889


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_negative_two_l3428_342831

theorem sqrt_expression_equals_negative_two :
  Real.sqrt 24 + (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) - (Real.sqrt 3 + Real.sqrt 2)^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_negative_two_l3428_342831


namespace NUMINAMATH_CALUDE_journal_problem_formula_l3428_342835

def f (x y : ℕ) : ℕ := 5 * x + 60 * (y - 1970) - 4

theorem journal_problem_formula (x y : ℕ) 
  (hx : 1 ≤ x ∧ x ≤ 12) (hy : 1970 ≤ y ∧ y ≤ 1989) : 
  (f 1 1970 = 1) ∧
  (∀ x' y', 1 ≤ x' ∧ x' < 12 → f (x' + 1) y' = f x' y' + 5) ∧
  (∀ y', f 1 (y' + 1) = f 1 y' + 60) →
  f x y = 5 * x + 60 * (y - 1970) - 4 :=
by sorry

end NUMINAMATH_CALUDE_journal_problem_formula_l3428_342835


namespace NUMINAMATH_CALUDE_fraction_division_problem_l3428_342865

theorem fraction_division_problem : (3/7 + 1/3) / (2/5) = 40/21 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_problem_l3428_342865


namespace NUMINAMATH_CALUDE_parallelogram_area_l3428_342824

-- Define the parallelogram ABCD
variable (A B C D : Point)

-- Define point E as midpoint of BC
variable (E : Point)

-- Define point F on AD
variable (F : Point)

-- Define the area function
variable (area : Set Point → ℝ)

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : Point) : Prop := sorry

-- Define E as midpoint of BC
def is_midpoint (E B C : Point) : Prop := sorry

-- Define the condition DF = 2FC
def segment_ratio (D F C : Point) : Prop := sorry

-- Define triangles
def triangle (P Q R : Point) : Set Point := sorry

-- Define parallelogram
def parallelogram (A B C D : Point) : Set Point := sorry

-- Theorem statement
theorem parallelogram_area 
  (h1 : is_parallelogram A B C D)
  (h2 : is_midpoint E B C)
  (h3 : segment_ratio D F C)
  (h4 : area (triangle A F C) + area (triangle A B E) = 10) :
  area (parallelogram A B C D) = 24 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3428_342824


namespace NUMINAMATH_CALUDE_at_least_one_negative_l3428_342856

theorem at_least_one_negative (a b : ℝ) (h : a + b < 0) :
  a < 0 ∨ b < 0 := by sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l3428_342856


namespace NUMINAMATH_CALUDE_unique_natural_solution_l3428_342890

theorem unique_natural_solution :
  ∀ n : ℕ, n ≠ 0 → (2 * n - 1 / (n^5 : ℚ) = 3 - 2 / (n : ℚ)) ↔ n = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_natural_solution_l3428_342890


namespace NUMINAMATH_CALUDE_range_of_a_l3428_342894

/-- The solution set to the inequality a^2 - 4a + 3 < 0 -/
def P (a : ℝ) : Prop := a^2 - 4*a + 3 < 0

/-- The real number a for which (a-2)x^2 + 2(a-2)x - 4 < 0 holds for all real numbers x -/
def Q (a : ℝ) : Prop := ∀ x : ℝ, (a-2)*x^2 + 2*(a-2)*x - 4 < 0

/-- Given P ∨ Q is true, the range of values for the real number a is -2 < a < 3 -/
theorem range_of_a (a : ℝ) (h : P a ∨ Q a) : -2 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3428_342894


namespace NUMINAMATH_CALUDE_problem_statement_l3428_342862

theorem problem_statement (x : ℚ) : 5 * x - 10 = 15 * x + 5 → 5 * (x + 3) = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3428_342862


namespace NUMINAMATH_CALUDE_estimations_correct_l3428_342891

/-- A function that performs rounding to the nearest hundred. -/
def roundToHundred (n : ℤ) : ℤ :=
  (n + 50) / 100 * 100

/-- The acceptable error margin for estimation. -/
def ε : ℚ := 100

/-- Theorem stating that the estimations are correct within the error margin. -/
theorem estimations_correct :
  let e1 := |212 + 384 - roundToHundred 212 - roundToHundred 384|
  let e2 := |903 - 497 - (roundToHundred 903 - roundToHundred 497)|
  let e3 := |206 + 3060 - roundToHundred 206 - roundToHundred 3060|
  let e4 := |523 + 386 - roundToHundred 523 - roundToHundred 386|
  (e1 ≤ ε) ∧ (e2 ≤ ε) ∧ (e3 ≤ ε) ∧ (e4 ≤ ε) := by
  sorry

end NUMINAMATH_CALUDE_estimations_correct_l3428_342891


namespace NUMINAMATH_CALUDE_polynomial_roots_l3428_342848

theorem polynomial_roots : 
  let f : ℝ → ℝ := λ x => (x + 1998) * (x + 1999) * (x + 2000) * (x + 2001) + 1
  ∀ x : ℝ, f x = 0 ↔ x = -1999.5 - Real.sqrt 5 / 2 ∨ x = -1999.5 + Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3428_342848


namespace NUMINAMATH_CALUDE_fraction_division_l3428_342822

theorem fraction_division : (4/9) / (5/8) = 32/45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l3428_342822


namespace NUMINAMATH_CALUDE_system_solution_iff_a_in_interval_l3428_342800

/-- The system of equations has a solution for some b if and only if a is in the interval (-8, 7] -/
theorem system_solution_iff_a_in_interval (a : ℝ) : 
  (∃ (b x y : ℝ), x^2 + y^2 + 2*a*(a - x - y) = 64 ∧ y = 7 / ((x + b)^2 + 1)) ↔ 
  -8 < a ∧ a ≤ 7 := by sorry

end NUMINAMATH_CALUDE_system_solution_iff_a_in_interval_l3428_342800


namespace NUMINAMATH_CALUDE_children_tickets_count_l3428_342808

/-- Proves that the number of children's tickets is 21 given the ticket prices, total amount paid, and total number of tickets. -/
theorem children_tickets_count (adult_price child_price total_amount total_tickets : ℕ)
  (h_adult_price : adult_price = 8)
  (h_child_price : child_price = 5)
  (h_total_amount : total_amount = 201)
  (h_total_tickets : total_tickets = 33) :
  ∃ (adult_count child_count : ℕ),
    adult_count + child_count = total_tickets ∧
    adult_count * adult_price + child_count * child_price = total_amount ∧
    child_count = 21 :=
by sorry

end NUMINAMATH_CALUDE_children_tickets_count_l3428_342808


namespace NUMINAMATH_CALUDE_cody_tickets_l3428_342850

theorem cody_tickets (initial : ℝ) (lost : ℝ) (spent : ℝ) : 
  initial = 49.0 → lost = 6.0 → spent = 25.0 → initial - lost - spent = 18.0 :=
by sorry

end NUMINAMATH_CALUDE_cody_tickets_l3428_342850


namespace NUMINAMATH_CALUDE_car_selling_problem_l3428_342876

/-- Calculates the net amount Chris receives from each buyer's offer --/
def net_amount (asking_price : ℝ) (inspection_cost : ℝ) (headlight_cost : ℝ) 
  (tire_cost : ℝ) (battery_cost : ℝ) (discount_rate : ℝ) (paint_job_rate : ℝ) : ℝ × ℝ × ℝ :=
  let first_offer := asking_price - inspection_cost
  let second_offer := asking_price - (headlight_cost + tire_cost + battery_cost)
  let discounted_price := asking_price * (1 - discount_rate)
  let third_offer := discounted_price - (discounted_price * paint_job_rate)
  (first_offer, second_offer, third_offer)

/-- Theorem statement for the car selling problem --/
theorem car_selling_problem (asking_price : ℝ) (inspection_rate : ℝ) (headlight_cost : ℝ) 
  (tire_rate : ℝ) (battery_rate : ℝ) (discount_rate : ℝ) (paint_job_rate : ℝ) :
  asking_price = 5200 ∧
  inspection_rate = 1/10 ∧
  headlight_cost = 80 ∧
  tire_rate = 3 ∧
  battery_rate = 2 ∧
  discount_rate = 15/100 ∧
  paint_job_rate = 1/5 →
  let (first, second, third) := net_amount asking_price (asking_price * inspection_rate) 
    headlight_cost (headlight_cost * tire_rate) (headlight_cost * tire_rate * battery_rate) 
    discount_rate paint_job_rate
  max first (max second third) - min first (min second third) = 1144 := by
  sorry


end NUMINAMATH_CALUDE_car_selling_problem_l3428_342876


namespace NUMINAMATH_CALUDE_find_number_l3428_342810

theorem find_number : ∃ x : ℝ, (0.4 * x - 30 = 50) ∧ (x = 200) := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3428_342810


namespace NUMINAMATH_CALUDE_sum_of_digits_mod_9_triple_sum_of_digits_4444_power_l3428_342887

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Property: sum of digits is congruent to the number modulo 9 -/
theorem sum_of_digits_mod_9 (n : ℕ) : sum_of_digits n % 9 = n % 9 := sorry

/-- Main theorem -/
theorem triple_sum_of_digits_4444_power :
  let N := 4444^4444
  let f := sum_of_digits
  f (f (f N)) = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_mod_9_triple_sum_of_digits_4444_power_l3428_342887


namespace NUMINAMATH_CALUDE_circle_kinetic_energy_l3428_342849

/-- 
Given a circle with radius R and a point P on its diameter AB, where PC is a semicord perpendicular to AB,
if three unit masses move along PA, PB, and PC with constant velocities reaching A, B, and C respectively in one unit of time,
and the total kinetic energy expended is a^2 units, then:
1. The distance of P from A is R ± √(2a^2 - 3R^2)
2. The value of a^2 must satisfy 3/2 * R^2 ≤ a^2 < 2R^2
-/
theorem circle_kinetic_energy (R a : ℝ) (h : R > 0) :
  let PA : ℝ → ℝ := λ x => x
  let PB : ℝ → ℝ := λ x => 2 * R - x
  let PC : ℝ → ℝ := λ x => Real.sqrt (x * (2 * R - x))
  let kinetic_energy : ℝ → ℝ := λ x => (PA x)^2 / 2 + (PB x)^2 / 2 + (PC x)^2 / 2
  ∃ x : ℝ, 0 < x ∧ x < 2 * R ∧ kinetic_energy x = a^2 →
    (x = R + Real.sqrt (2 * a^2 - 3 * R^2) ∨ x = R - Real.sqrt (2 * a^2 - 3 * R^2)) ∧
    3 / 2 * R^2 ≤ a^2 ∧ a^2 < 2 * R^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_kinetic_energy_l3428_342849


namespace NUMINAMATH_CALUDE_equation_solution_l3428_342872

theorem equation_solution (x : ℝ) :
  (x^2 - 7*x + 6)/(x - 1) + (2*x^2 + 7*x - 6)/(2*x - 1) = 1 ∧ 
  x ≠ 1 ∧ 
  x ≠ 1/2 →
  x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3428_342872


namespace NUMINAMATH_CALUDE_lemons_removed_is_thirty_l3428_342840

/-- Represents the number of lemons picked by each person and eaten by animals --/
structure LemonCounts where
  sally : ℕ
  mary : ℕ
  tom : ℕ
  eaten : ℕ

/-- Calculates the total number of lemons removed from the tree --/
def total_lemons_removed (counts : LemonCounts) : ℕ :=
  counts.sally + counts.mary + counts.tom + counts.eaten

/-- Theorem stating that the total number of lemons removed is 30 --/
theorem lemons_removed_is_thirty : 
  ∀ (counts : LemonCounts), 
  counts.sally = 7 → 
  counts.mary = 9 → 
  counts.tom = 12 → 
  counts.eaten = 2 → 
  total_lemons_removed counts = 30 := by
  sorry


end NUMINAMATH_CALUDE_lemons_removed_is_thirty_l3428_342840


namespace NUMINAMATH_CALUDE_cosine_sum_equals_one_l3428_342879

theorem cosine_sum_equals_one (α β : ℝ) :
  ((Real.cos α * Real.cos (β / 2)) / Real.cos (α + β / 2) +
   (Real.cos β * Real.cos (α / 2)) / Real.cos (β + α / 2) = 1) →
  Real.cos α + Real.cos β = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_equals_one_l3428_342879
