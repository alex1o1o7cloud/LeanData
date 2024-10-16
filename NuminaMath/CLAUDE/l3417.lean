import Mathlib

namespace NUMINAMATH_CALUDE_weed_pulling_rate_is_11_l3417_341792

-- Define the hourly rates and hours worked
def mowing_rate : ℝ := 6
def mulch_rate : ℝ := 9
def mowing_hours : ℝ := 63
def weed_hours : ℝ := 9
def mulch_hours : ℝ := 10
def total_earnings : ℝ := 567

-- Define the function to calculate total earnings
def calculate_earnings (weed_rate : ℝ) : ℝ :=
  mowing_rate * mowing_hours + weed_rate * weed_hours + mulch_rate * mulch_hours

-- Theorem statement
theorem weed_pulling_rate_is_11 :
  ∃ (weed_rate : ℝ), calculate_earnings weed_rate = total_earnings ∧ weed_rate = 11 := by
  sorry

end NUMINAMATH_CALUDE_weed_pulling_rate_is_11_l3417_341792


namespace NUMINAMATH_CALUDE_crate_stacking_probability_l3417_341719

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the probability of stacking crates to a specific height -/
def stackProbability (dimensions : CrateDimensions) (numCrates : ℕ) (targetHeight : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the probability of stacking 15 crates to 50ft -/
theorem crate_stacking_probability :
  let dimensions : CrateDimensions := ⟨2, 3, 5⟩
  stackProbability dimensions 15 50 = 1162161 / 14348907 := by
  sorry

end NUMINAMATH_CALUDE_crate_stacking_probability_l3417_341719


namespace NUMINAMATH_CALUDE_line_intersection_l3417_341734

/-- Two lines in 2D space -/
structure Line2D where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line2D) : Prop :=
  ∃ (c : ℝ), l1.direction = (c * l2.direction.1, c * l2.direction.2)

/-- The problem statement -/
theorem line_intersection (p : ℝ) : 
  let line1 : Line2D := ⟨(2, 3), (5, -8)⟩
  let line2 : Line2D := ⟨(-1, 4), (3, p)⟩
  parallel line1 line2 → p = -24/5 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_l3417_341734


namespace NUMINAMATH_CALUDE_correct_regression_equation_l3417_341710

-- Define the variables and their properties
variable (x y : ℝ)
variable (mean_x mean_y : ℝ)

-- Define the negative correlation between x and y
variable (negative_correlation : ∃ (r : ℝ), r < 0 ∧ Correlation x y = r)

-- Define the sample means
variable (sample_mean_x : mean_x = 4)
variable (sample_mean_y : mean_y = 6.5)

-- Define the linear regression equation
def regression_equation (x : ℝ) : ℝ := -2 * x + 14.5

-- Theorem statement
theorem correct_regression_equation :
  negative_correlation →
  sample_mean_x →
  sample_mean_y →
  regression_equation mean_x = mean_y :=
sorry

end NUMINAMATH_CALUDE_correct_regression_equation_l3417_341710


namespace NUMINAMATH_CALUDE_water_displaced_by_cube_l3417_341707

/-- The volume of water displaced by a partially submerged cube in a cylinder -/
theorem water_displaced_by_cube (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ)
  (h_cube_side : cube_side = 10)
  (h_cylinder_radius : cylinder_radius = 5)
  (h_cylinder_height : cylinder_height = 12) :
  ∃ (v : ℝ), v = 75 * Real.sqrt 3 ∧ v^2 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_water_displaced_by_cube_l3417_341707


namespace NUMINAMATH_CALUDE_folded_paper_perimeter_ratio_l3417_341773

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem folded_paper_perimeter_ratio :
  let original_side : ℝ := 8
  let large_rect : Rectangle := { width := original_side, height := original_side / 2 }
  let small_rect : Rectangle := { width := original_side / 2, height := original_side / 2 }
  (perimeter small_rect) / (perimeter large_rect) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_perimeter_ratio_l3417_341773


namespace NUMINAMATH_CALUDE_expand_expression_l3417_341770

theorem expand_expression (x y : ℝ) : (2*x + 15) * (3*y + 5) = 6*x*y + 10*x + 45*y + 75 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3417_341770


namespace NUMINAMATH_CALUDE_three_petal_clover_percentage_l3417_341745

theorem three_petal_clover_percentage 
  (total_clovers : ℕ)
  (two_petal_percentage : ℝ)
  (four_petal_percentage : ℝ)
  (total_earnings : ℕ)
  (h1 : total_clovers = 200)
  (h2 : two_petal_percentage = 24)
  (h3 : four_petal_percentage = 1)
  (h4 : total_earnings = 554) :
  100 - two_petal_percentage - four_petal_percentage = 75 := by
  sorry

end NUMINAMATH_CALUDE_three_petal_clover_percentage_l3417_341745


namespace NUMINAMATH_CALUDE_sum_n_value_l3417_341762

/-- An arithmetic sequence {a_n} satisfying given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  condition1 : a 3 * a 7 = -16
  condition2 : a 4 + a 6 = 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n : ℤ) * (seq.a 1 + seq.a n) / 2

/-- Theorem stating the possible values for the sum of the first n terms -/
theorem sum_n_value (seq : ArithmeticSequence) (n : ℕ) :
  sum_n seq n = n * (n - 9) ∨ sum_n seq n = -n * (n - 9) := by
  sorry


end NUMINAMATH_CALUDE_sum_n_value_l3417_341762


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_range_l3417_341795

theorem absolute_value_inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, |x + 1| - |x - 2| ≥ a^2 - 4*a) ↔ a ∈ Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_range_l3417_341795


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l3417_341700

theorem cow_chicken_problem (c h : ℕ) : 
  (4 * c + 2 * h = 2 * (c + h) + 18) → c = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l3417_341700


namespace NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l3417_341733

/-- Given two cubic polynomials with two distinct common roots, prove that a = -6 and b = -7 -/
theorem common_roots_cubic_polynomials (a b : ℝ) : 
  (∃ r s : ℝ, r ≠ s ∧ 
    r^3 + a*r^2 + 17*r + 10 = 0 ∧ 
    s^3 + a*s^2 + 17*s + 10 = 0 ∧
    r^3 + b*r^2 + 20*r + 12 = 0 ∧ 
    s^3 + b*s^2 + 20*s + 12 = 0) →
  a = -6 ∧ b = -7 := by
sorry

end NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l3417_341733


namespace NUMINAMATH_CALUDE_x_intercept_is_two_l3417_341781

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def xIntercept (l : Line) : ℝ :=
  sorry

/-- The theorem stating that the x-intercept of the given line is 2 -/
theorem x_intercept_is_two :
  let l : Line := { x₁ := 1, y₁ := -2, x₂ := 5, y₂ := 6 }
  xIntercept l = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_is_two_l3417_341781


namespace NUMINAMATH_CALUDE_nancy_tortilla_chips_nancy_final_chips_l3417_341752

/-- Calculates the number of tortilla chips Nancy has left after sharing with her family members -/
theorem nancy_tortilla_chips (initial_chips : ℝ) (brother_chips : ℝ) 
  (sister_fraction : ℝ) (cousin_percent : ℝ) : ℝ :=
  let remaining_after_brother := initial_chips - brother_chips
  let sister_chips := sister_fraction * remaining_after_brother
  let remaining_after_sister := remaining_after_brother - sister_chips
  let cousin_chips := (cousin_percent / 100) * remaining_after_sister
  let final_chips := remaining_after_sister - cousin_chips
  final_chips

/-- Proves that Nancy has 18.75 tortilla chips left for herself -/
theorem nancy_final_chips : 
  nancy_tortilla_chips 50 12.5 (1/3) 25 = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_nancy_tortilla_chips_nancy_final_chips_l3417_341752


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l3417_341711

/-- Represents the total number of products -/
def total_products : ℕ := 5

/-- Represents the number of qualified products -/
def qualified_products : ℕ := 3

/-- Represents the number of unqualified products -/
def unqualified_products : ℕ := 2

/-- Represents the number of products randomly selected -/
def selected_products : ℕ := 2

/-- Event A: Exactly 1 unqualified product is selected -/
def event_A : Prop := sorry

/-- Event B: Exactly 2 qualified products are selected -/
def event_B : Prop := sorry

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutually_exclusive (e1 e2 : Prop) : Prop := ¬(e1 ∧ e2)

/-- Two events are contradictory if one must occur when the other does not -/
def contradictory (e1 e2 : Prop) : Prop := (e1 ↔ ¬e2)

theorem events_mutually_exclusive_not_contradictory :
  mutually_exclusive event_A event_B ∧ ¬contradictory event_A event_B := by sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_contradictory_l3417_341711


namespace NUMINAMATH_CALUDE_even_square_iff_even_l3417_341704

theorem even_square_iff_even (p : ℕ) : Even p ↔ Even (p^2) := by
  sorry

end NUMINAMATH_CALUDE_even_square_iff_even_l3417_341704


namespace NUMINAMATH_CALUDE_nickys_pace_l3417_341769

/-- Prove Nicky's pace given the race conditions -/
theorem nickys_pace (race_length : ℝ) (head_start : ℝ) (cristina_pace : ℝ) (catch_up_time : ℝ)
  (h1 : race_length = 500)
  (h2 : head_start = 12)
  (h3 : cristina_pace = 5)
  (h4 : catch_up_time = 30) :
  cristina_pace = catch_up_time * race_length / (catch_up_time * cristina_pace) :=
by sorry

end NUMINAMATH_CALUDE_nickys_pace_l3417_341769


namespace NUMINAMATH_CALUDE_smallest_integer_theorem_l3417_341760

def is_divisible (n m : ℕ) : Prop := m ∣ n

def smallest_integer_with_divisors (excluded : List ℕ) : ℕ :=
  let divisors := (List.range 31).filter (λ x => x ∉ excluded)
  divisors.foldl Nat.lcm 1

theorem smallest_integer_theorem :
  let n := smallest_integer_with_divisors [17, 19]
  (∀ k ∈ List.range 31, k ≠ 17 → k ≠ 19 → is_divisible n k) ∧
  (∀ m < n, ∃ k ∈ List.range 31, k ≠ 17 ∧ k ≠ 19 ∧ ¬is_divisible m k) ∧
  n = 122522400 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_theorem_l3417_341760


namespace NUMINAMATH_CALUDE_A_intersect_B_is_empty_l3417_341799

def A : Set ℝ := {0, 1, 2}

def B : Set ℝ := {x : ℝ | (x + 1) * (x + 2) ≤ 0}

theorem A_intersect_B_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_is_empty_l3417_341799


namespace NUMINAMATH_CALUDE_correct_arrangement_l3417_341771

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def valid_arrangement (arr : List ℕ) : Prop :=
  arr.length = 6 ∧
  (∀ n, n ∈ arr → n ∈ [1, 2, 3, 4, 5, 6]) ∧
  (∀ i, i < 3 → is_perfect_square (arr[2*i]! * arr[2*i+1]!))

theorem correct_arrangement :
  valid_arrangement [4, 2, 5, 3, 6, 1] :=
by sorry

end NUMINAMATH_CALUDE_correct_arrangement_l3417_341771


namespace NUMINAMATH_CALUDE_sum_of_two_equals_third_l3417_341716

theorem sum_of_two_equals_third (a b c x y : ℝ) 
  (h1 : (a + x)⁻¹ = 6)
  (h2 : (b + y)⁻¹ = 3)
  (h3 : (c + x + y)⁻¹ = 2) : 
  c = a + b := by sorry

end NUMINAMATH_CALUDE_sum_of_two_equals_third_l3417_341716


namespace NUMINAMATH_CALUDE_difference_of_squares_l3417_341785

theorem difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3417_341785


namespace NUMINAMATH_CALUDE_problem_solution_l3417_341736

theorem problem_solution : 
  let M : ℚ := 2007 / 3
  let N : ℚ := M / 3
  let X : ℚ := M - N
  X = 446 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3417_341736


namespace NUMINAMATH_CALUDE_min_value_problem_l3417_341791

theorem min_value_problem (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 4)
  (h5 : y^2 = x^2 + 2) (h6 : z^2 = y^2 + 2) :
  ∃ (min_val : ℝ), min_val = 4 - 2 * Real.sqrt 3 ∧ 
  ∀ (x' y' z' : ℝ), 0 ≤ x' ∧ x' ≤ y' ∧ y' ≤ z' ∧ z' ≤ 4 ∧ 
  y'^2 = x'^2 + 2 ∧ z'^2 = y'^2 + 2 → z' - x' ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l3417_341791


namespace NUMINAMATH_CALUDE_pool_visitors_l3417_341775

theorem pool_visitors (total_earned : ℚ) (cost_per_person : ℚ) (amount_left : ℚ) 
  (h1 : total_earned = 30)
  (h2 : cost_per_person = 5/2)
  (h3 : amount_left = 5) :
  (total_earned - amount_left) / cost_per_person = 10 := by
  sorry

end NUMINAMATH_CALUDE_pool_visitors_l3417_341775


namespace NUMINAMATH_CALUDE_kameron_has_100_kangaroos_l3417_341797

/-- The number of kangaroos Bert currently has -/
def bert_initial : ℕ := 20

/-- The number of days until Bert has the same number of kangaroos as Kameron -/
def days : ℕ := 40

/-- The number of kangaroos Bert buys per day -/
def bert_rate : ℕ := 2

/-- The number of kangaroos Kameron has -/
def kameron_kangaroos : ℕ := bert_initial + days * bert_rate

theorem kameron_has_100_kangaroos : kameron_kangaroos = 100 := by
  sorry

end NUMINAMATH_CALUDE_kameron_has_100_kangaroos_l3417_341797


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3417_341713

/-- The surface area of a sphere, given specific conditions for a hemisphere --/
theorem sphere_surface_area (r : ℝ) 
  (h1 : π * r^2 = 3)  -- area of the base of the hemisphere
  (h2 : 3 * π * r^2 = 9)  -- total surface area of the hemisphere
  : 4 * π * r^2 = 12 := by
  sorry

#check sphere_surface_area

end NUMINAMATH_CALUDE_sphere_surface_area_l3417_341713


namespace NUMINAMATH_CALUDE_sqrt_three_fourths_equals_sqrt_three_over_two_l3417_341764

theorem sqrt_three_fourths_equals_sqrt_three_over_two :
  Real.sqrt (3 / 4) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_fourths_equals_sqrt_three_over_two_l3417_341764


namespace NUMINAMATH_CALUDE_quadratic_function_value_l3417_341712

-- Define the quadratic function f(x) = x² - ax + b
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 - a*x + b

-- State the theorem
theorem quadratic_function_value (a b : ℝ) :
  f a b 1 = -1 → f a b 2 = 2 → f a b (-4) = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l3417_341712


namespace NUMINAMATH_CALUDE_four_numbers_theorem_l3417_341706

def satisfies_condition (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  x₁ + x₂ * x₃ * x₄ = 2 ∧
  x₂ + x₁ * x₃ * x₄ = 2 ∧
  x₃ + x₁ * x₂ * x₄ = 2 ∧
  x₄ + x₁ * x₂ * x₃ = 2

theorem four_numbers_theorem :
  ∀ x₁ x₂ x₃ x₄ : ℝ,
    satisfies_condition x₁ x₂ x₃ x₄ ↔
      ((x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
       (x₁ = 3 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = -1) ∨
       (x₁ = -1 ∧ x₂ = 3 ∧ x₃ = -1 ∧ x₄ = -1) ∨
       (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = 3 ∧ x₄ = -1) ∨
       (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = 3)) :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_theorem_l3417_341706


namespace NUMINAMATH_CALUDE_davonte_mercedes_difference_l3417_341753

/-- Proves that Davonte ran 2 kilometers farther than Mercedes -/
theorem davonte_mercedes_difference (jonathan_distance : ℝ) 
  (h1 : jonathan_distance = 7.5)
  (mercedes_distance : ℝ) 
  (h2 : mercedes_distance = 2 * jonathan_distance)
  (davonte_distance : ℝ)
  (h3 : mercedes_distance + davonte_distance = 32) :
  davonte_distance - mercedes_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_davonte_mercedes_difference_l3417_341753


namespace NUMINAMATH_CALUDE_dress_designs_count_l3417_341721

/-- The number of available fabric colors -/
def num_colors : ℕ := 3

/-- The number of available patterns -/
def num_patterns : ℕ := 4

/-- The total number of possible dress designs -/
def total_designs : ℕ := num_colors * num_patterns

/-- Theorem stating that the total number of possible dress designs is 12 -/
theorem dress_designs_count : total_designs = 12 := by
  sorry

end NUMINAMATH_CALUDE_dress_designs_count_l3417_341721


namespace NUMINAMATH_CALUDE_relationship_xyz_l3417_341732

theorem relationship_xyz (x y z : ℝ) 
  (hx : x = (0.5 : ℝ) ^ (0.5 : ℝ))
  (hy : y = (0.5 : ℝ) ^ (1.3 : ℝ))
  (hz : z = (1.3 : ℝ) ^ (0.5 : ℝ)) :
  y < x ∧ x < z :=
by sorry

end NUMINAMATH_CALUDE_relationship_xyz_l3417_341732


namespace NUMINAMATH_CALUDE_tims_change_l3417_341754

/-- The amount of change Tim will get after buying a candy bar -/
def change (initial_amount : ℕ) (price : ℕ) : ℕ :=
  initial_amount - price

/-- Theorem: Tim's change is 5 cents -/
theorem tims_change : change 50 45 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tims_change_l3417_341754


namespace NUMINAMATH_CALUDE_employment_percentage_l3417_341731

theorem employment_percentage (population : ℝ) 
  (h1 : population > 0)
  (h2 : (80 : ℝ) / 100 * population = employed_males)
  (h3 : (1 : ℝ) / 3 * total_employed = employed_females)
  (h4 : employed_males + employed_females = total_employed) :
  total_employed / population = (60 : ℝ) / 100 := by
sorry

end NUMINAMATH_CALUDE_employment_percentage_l3417_341731


namespace NUMINAMATH_CALUDE_triangle_properties_l3417_341730

theorem triangle_properties (a b c : ℝ) 
  (h : |a - Real.sqrt 7| + Real.sqrt (b - 5) + (c - 4 * Real.sqrt 2)^2 = 0) :
  a = Real.sqrt 7 ∧ 
  b = 5 ∧ 
  c = 4 * Real.sqrt 2 ∧
  a^2 + b^2 = c^2 ∧
  a + b > c ∧
  b + c > a ∧
  c + a > b := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3417_341730


namespace NUMINAMATH_CALUDE_equation_solutions_l3417_341735

theorem equation_solutions : 
  ∀ x : ℝ, (4 * (3 * x)^2 + 3 * x + 5 = 3 * (8 * x^2 + 3 * x + 3)) ↔ 
  (x = (1 + Real.sqrt 19) / 4 ∨ x = (1 - Real.sqrt 19) / 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3417_341735


namespace NUMINAMATH_CALUDE_cell_division_after_three_hours_l3417_341715

/-- Represents the number of cells after a given number of 30-minute periods -/
def cells_after_periods (n : ℕ) : ℕ := 2^n

/-- Represents the number of 30-minute periods in a given number of hours -/
def periods_in_hours (hours : ℕ) : ℕ := 2 * hours

theorem cell_division_after_three_hours :
  cells_after_periods (periods_in_hours 3) = 64 := by
  sorry

#eval cells_after_periods (periods_in_hours 3)

end NUMINAMATH_CALUDE_cell_division_after_three_hours_l3417_341715


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3417_341746

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- First term of the sequence -/
def a : ℚ := 1/3

/-- Common ratio of the sequence -/
def r : ℚ := 1/3

/-- Number of terms to sum -/
def n : ℕ := 8

theorem geometric_sequence_sum :
  geometric_sum a r n = 3280/6561 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3417_341746


namespace NUMINAMATH_CALUDE_employee_transportation_difference_l3417_341784

/-- Proves the difference between employees who drive and those who take public transportation -/
theorem employee_transportation_difference
  (total_employees : ℕ)
  (drive_percentage : ℚ)
  (public_transport_fraction : ℚ)
  (h_total : total_employees = 200)
  (h_drive : drive_percentage = 3/5)
  (h_public : public_transport_fraction = 1/2) :
  (drive_percentage * total_employees : ℚ) -
  (public_transport_fraction * (total_employees - drive_percentage * total_employees) : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_employee_transportation_difference_l3417_341784


namespace NUMINAMATH_CALUDE_shower_water_usage_l3417_341724

/-- The total water usage of Roman and Remy's showers -/
theorem shower_water_usage (R : ℝ) 
  (h1 : 3 * R + 1 = 25) : R + (3 * R + 1) = 33 := by
  sorry

end NUMINAMATH_CALUDE_shower_water_usage_l3417_341724


namespace NUMINAMATH_CALUDE_figure_area_solution_l3417_341768

theorem figure_area_solution (x : ℝ) : 
  (3 * x)^2 + (6 * x)^2 + (1/2 * 3 * x * 6 * x) = 1950 → 
  x = (5 * Real.sqrt 13) / 3 := by
sorry

end NUMINAMATH_CALUDE_figure_area_solution_l3417_341768


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l3417_341728

theorem smallest_number_with_remainders : ∃! x : ℕ, 
  x ≥ 0 ∧ 
  x % 5 = 2 ∧ 
  x % 7 = 3 ∧ 
  x % 11 = 7 ∧
  ∀ y : ℕ, y ≥ 0 ∧ y % 5 = 2 ∧ y % 7 = 3 ∧ y % 11 = 7 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l3417_341728


namespace NUMINAMATH_CALUDE_g_negative_one_equals_three_l3417_341725

-- Define an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the theorem
theorem g_negative_one_equals_three
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_g_def : ∀ x, g x = f x + 2)
  (h_g_one : g 1 = 1) :
  g (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_negative_one_equals_three_l3417_341725


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l3417_341748

theorem consecutive_even_integers_sum (n : ℤ) : 
  n % 2 = 0 ∧ n * (n + 2) * (n + 4) = 3360 → n + (n + 2) + (n + 4) = 48 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l3417_341748


namespace NUMINAMATH_CALUDE_sin_120_degrees_l3417_341742

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l3417_341742


namespace NUMINAMATH_CALUDE_new_person_weight_l3417_341741

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 80 kg -/
theorem new_person_weight :
  weight_of_new_person 6 2.5 65 = 80 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3417_341741


namespace NUMINAMATH_CALUDE_watermelon_sharing_l3417_341786

/-- The number of people that can share one watermelon -/
def people_per_watermelon : ℕ := 8

/-- The number of watermelons available -/
def num_watermelons : ℕ := 4

/-- The total number of people that can share the watermelons -/
def total_people : ℕ := people_per_watermelon * num_watermelons

theorem watermelon_sharing :
  total_people = 32 :=
by sorry

end NUMINAMATH_CALUDE_watermelon_sharing_l3417_341786


namespace NUMINAMATH_CALUDE_parabola_focus_and_intersection_l3417_341708

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line
def line (n : ℝ) (x y : ℝ) : Prop := x = Real.sqrt 3 * y + n

-- Define the point E
def point_E : ℝ × ℝ := (4, 4)

-- Define the point D
def point_D (n : ℝ) : ℝ × ℝ := (n, 0)

-- Theorem statement
theorem parabola_focus_and_intersection
  (p : ℝ)
  (n : ℝ)
  (h1 : parabola p (point_E.1) (point_E.2))
  (h2 : ∃ (A B : ℝ × ℝ), A ≠ B ∧ A ≠ point_E ∧ B ≠ point_E ∧
        parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
        line n A.1 A.2 ∧ line n B.1 B.2)
  (h3 : ∃ (A B : ℝ × ℝ), 
        parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
        line n A.1 A.2 ∧ line n B.1 B.2 ∧
        (A.1 - (point_D n).1)^2 + A.2^2 * ((B.1 - (point_D n).1)^2 + B.2^2) = 64) :
  (p = 2 ∧ n = 4) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_and_intersection_l3417_341708


namespace NUMINAMATH_CALUDE_rectangular_field_width_l3417_341788

theorem rectangular_field_width :
  ∀ (width length perimeter : ℝ),
    length = (7 / 5) * width →
    perimeter = 2 * length + 2 * width →
    perimeter = 360 →
    width = 75 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l3417_341788


namespace NUMINAMATH_CALUDE_no_real_solutions_l3417_341782

/-- 
Theorem: The system x^3 + y^3 = 2 and y = kx + d has no real solutions (x,y) 
if and only if k = -1 and 0 < d < 2√2.
-/
theorem no_real_solutions (k d : ℝ) : 
  (∀ x y : ℝ, x^3 + y^3 ≠ 2 ∨ y ≠ k*x + d) ↔ (k = -1 ∧ 0 < d ∧ d < 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_no_real_solutions_l3417_341782


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3417_341778

/-- Represents a repeating decimal with a single digit repeating -/
def SingleDigitRepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + repeating / 9

/-- Represents a repeating decimal with two digits repeating -/
def TwoDigitRepeatingDecimal (whole : ℚ) (repeating : ℕ) : ℚ :=
  whole + repeating / 99

theorem sum_of_repeating_decimals :
  (SingleDigitRepeatingDecimal 0 2) + (TwoDigitRepeatingDecimal 0 4) = 26 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3417_341778


namespace NUMINAMATH_CALUDE_product_increase_thirteen_times_l3417_341790

theorem product_increase_thirteen_times :
  ∃ (a b c d e f g : ℕ),
    (a - 3) * (b - 3) * (c - 3) * (d - 3) * (e - 3) * (f - 3) * (g - 3) = 13 * (a * b * c * d * e * f * g) :=
by sorry

end NUMINAMATH_CALUDE_product_increase_thirteen_times_l3417_341790


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3417_341756

theorem negative_fraction_comparison : -5/6 < -7/9 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3417_341756


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3417_341702

def m : ℕ := 55555555
def n : ℕ := 5555555555

theorem gcd_of_specific_numbers : Nat.gcd m n = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l3417_341702


namespace NUMINAMATH_CALUDE_binomial_12_choose_6_l3417_341779

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_6_l3417_341779


namespace NUMINAMATH_CALUDE_courier_net_pay_rate_l3417_341774

def travel_time : ℝ := 3
def speed : ℝ := 65
def fuel_efficiency : ℝ := 28
def payment_rate : ℝ := 0.55
def gasoline_cost : ℝ := 2.50

theorem courier_net_pay_rate : 
  let total_distance := travel_time * speed
  let gasoline_used := total_distance / fuel_efficiency
  let earnings := payment_rate * total_distance
  let gasoline_expense := gasoline_cost * gasoline_used
  let net_earnings := earnings - gasoline_expense
  let net_rate_per_hour := net_earnings / travel_time
  ⌊net_rate_per_hour⌋ = 30 := by sorry

end NUMINAMATH_CALUDE_courier_net_pay_rate_l3417_341774


namespace NUMINAMATH_CALUDE_scramble_word_count_l3417_341796

/-- The number of letters in the extended Kobish alphabet -/
def alphabet_size : ℕ := 21

/-- The maximum length of a word -/
def max_word_length : ℕ := 4

/-- Calculates the number of words of a given length that contain the letter B at least once -/
def words_with_b (length : ℕ) : ℕ :=
  alphabet_size^length - (alphabet_size - 1)^length

/-- The total number of valid words in the Scramble language -/
def total_valid_words : ℕ :=
  words_with_b 1 + words_with_b 2 + words_with_b 3 + words_with_b 4

theorem scramble_word_count : total_valid_words = 35784 := by
  sorry

end NUMINAMATH_CALUDE_scramble_word_count_l3417_341796


namespace NUMINAMATH_CALUDE_product_mod_seven_l3417_341703

theorem product_mod_seven : (2007 * 2008 * 2009 * 2010) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l3417_341703


namespace NUMINAMATH_CALUDE_max_distance_between_functions_l3417_341751

theorem max_distance_between_functions : ∃ (C : ℝ),
  C = Real.sqrt 5 ∧ 
  ∀ x : ℝ, |2 * Real.sin x - Real.sin (π / 2 - x)| ≤ C ∧
  ∃ x₀ : ℝ, |2 * Real.sin x₀ - Real.sin (π / 2 - x₀)| = C :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_functions_l3417_341751


namespace NUMINAMATH_CALUDE_n_good_lower_bound_two_is_seven_good_l3417_341726

/-- A tournament between n players where each player plays against every other player once --/
structure Tournament (n : ℕ) where
  result : Fin n → Fin n → Bool
  irreflexive : ∀ i, result i i = false
  antisymmetric : ∀ i j, result i j = !result j i

/-- A number k is n-good if there exists a tournament where for any k players, 
    there is another player who has lost to all of them --/
def is_n_good (n k : ℕ) : Prop :=
  ∃ t : Tournament n, ∀ (s : Finset (Fin n)) (hs : s.card = k),
    ∃ p : Fin n, p ∉ s ∧ ∀ q ∈ s, t.result q p = true

/-- The main theorem: For any n-good number k, n ≥ 2^(k+1) - 1 --/
theorem n_good_lower_bound (n k : ℕ) (h : is_n_good n k) : n ≥ 2^(k+1) - 1 :=
  sorry

/-- The smallest n for which 2 is n-good is 7 --/
theorem two_is_seven_good : 
  (is_n_good 7 2) ∧ (∀ m < 7, ¬ is_n_good m 2) :=
  sorry

end NUMINAMATH_CALUDE_n_good_lower_bound_two_is_seven_good_l3417_341726


namespace NUMINAMATH_CALUDE_danny_wrappers_found_l3417_341798

theorem danny_wrappers_found (initial_caps : ℕ) (found_caps : ℕ) (total_caps : ℕ) (total_wrappers : ℕ) 
  (h1 : initial_caps = 6)
  (h2 : found_caps = 22)
  (h3 : total_caps = 28)
  (h4 : total_wrappers = 63)
  (h5 : found_caps = total_caps - initial_caps)
  : ∃ (found_wrappers : ℕ), found_wrappers = 22 ∧ total_wrappers ≥ found_wrappers :=
by
  sorry

#check danny_wrappers_found

end NUMINAMATH_CALUDE_danny_wrappers_found_l3417_341798


namespace NUMINAMATH_CALUDE_seconds_in_hours_l3417_341772

theorem seconds_in_hours : 
  (∀ (hours : ℝ), hours * 60 * 60 = hours * 3600) →
  3.5 * 3600 = 12600 := by sorry

end NUMINAMATH_CALUDE_seconds_in_hours_l3417_341772


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3417_341729

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (k : ℚ), k * (a.val * Real.sqrt 6 + b.val * Real.sqrt 8) / c.val = 
    Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) →
  (∀ (x y z : ℕ+), (∃ (l : ℚ), l * (x.val * Real.sqrt 6 + y.val * Real.sqrt 8) / z.val = 
    Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) → 
    c.val ≤ z.val) →
  a.val + b.val + c.val = 192 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3417_341729


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_20_l3417_341759

/-- A four-digit number is a natural number between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

/-- Theorem stating that 9920 is the largest four-digit number whose digits sum to 20 -/
theorem largest_four_digit_sum_20 :
  FourDigitNumber 9920 ∧
  sumOfDigits 9920 = 20 ∧
  ∀ n : ℕ, FourDigitNumber n → sumOfDigits n = 20 → n ≤ 9920 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_20_l3417_341759


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3417_341714

theorem cost_price_calculation (selling_price : ℚ) (profit_percentage : ℚ) : 
  selling_price = 600 → profit_percentage = 60 → 
  ∃ (cost_price : ℚ), cost_price = 375 ∧ selling_price = cost_price * (1 + profit_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3417_341714


namespace NUMINAMATH_CALUDE_clothing_expense_l3417_341747

theorem clothing_expense (total_spent adidas_original nike skechers puma adidas clothes : ℝ) 
  (h_total : total_spent = 12000)
  (h_nike : nike = 2 * adidas)
  (h_skechers : adidas = 1/3 * skechers)
  (h_puma : puma = 3/4 * nike)
  (h_adidas_original : adidas_original = 900)
  (h_adidas_discount : adidas = adidas_original * 0.9)
  (h_sum : total_spent = nike + adidas + skechers + puma + clothes) :
  clothes = 5925 := by
sorry


end NUMINAMATH_CALUDE_clothing_expense_l3417_341747


namespace NUMINAMATH_CALUDE_roof_area_l3417_341727

theorem roof_area (width length : ℝ) : 
  width > 0 → 
  length > 0 → 
  length = 4 * width → 
  length - width = 39 → 
  width * length = 676 := by
sorry

end NUMINAMATH_CALUDE_roof_area_l3417_341727


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3417_341750

theorem complex_number_in_first_quadrant (z : ℂ) (h : z * (4 + I) = 3 + I) : 
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3417_341750


namespace NUMINAMATH_CALUDE_rays_grocery_bill_l3417_341743

/-- Calculates the total grocery bill for Ray's purchase with a store rewards discount --/
theorem rays_grocery_bill :
  let meat_price : ℚ := 5
  let crackers_price : ℚ := 3.5
  let vegetable_price : ℚ := 2
  let vegetable_quantity : ℕ := 4
  let cheese_price : ℚ := 3.5
  let discount_rate : ℚ := 0.1

  let total_before_discount : ℚ := 
    meat_price + crackers_price + (vegetable_price * vegetable_quantity) + cheese_price
  
  let discount_amount : ℚ := total_before_discount * discount_rate
  
  let final_bill : ℚ := total_before_discount - discount_amount

  final_bill = 18 :=
by
  sorry


end NUMINAMATH_CALUDE_rays_grocery_bill_l3417_341743


namespace NUMINAMATH_CALUDE_triangle_angle_B_l3417_341737

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_B (t : Triangle) :
  t.a = 2 ∧ t.b = 3 ∧ t.A = π/4 → t.B = π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l3417_341737


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3417_341780

theorem quadratic_factorization (x : ℝ) : 12 * x^2 + 8 * x - 4 = 4 * (3 * x - 1) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3417_341780


namespace NUMINAMATH_CALUDE_range_of_a_l3417_341749

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x - 4 < 0) ↔ -4 < a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3417_341749


namespace NUMINAMATH_CALUDE_granola_bars_per_box_l3417_341740

theorem granola_bars_per_box 
  (num_kids : ℕ) 
  (bars_per_kid : ℕ) 
  (num_boxes : ℕ) 
  (h1 : num_kids = 30) 
  (h2 : bars_per_kid = 2) 
  (h3 : num_boxes = 5) :
  (num_kids * bars_per_kid) / num_boxes = 12 := by
sorry

end NUMINAMATH_CALUDE_granola_bars_per_box_l3417_341740


namespace NUMINAMATH_CALUDE_max_regions_lines_theorem_max_regions_circles_theorem_l3417_341718

/-- The maximum number of regions in a plane divided by n lines -/
def max_regions_lines (n : ℕ) : ℕ := (n^2 + n + 2) / 2

/-- The maximum number of regions in a plane divided by n circles -/
def max_regions_circles (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem: The maximum number of regions in a plane divided by n lines is (n^2 + n + 2) / 2 -/
theorem max_regions_lines_theorem (n : ℕ) :
  max_regions_lines n = (n^2 + n + 2) / 2 := by sorry

/-- Theorem: The maximum number of regions in a plane divided by n circles is n^2 - n + 2 -/
theorem max_regions_circles_theorem (n : ℕ) :
  max_regions_circles n = n^2 - n + 2 := by sorry

end NUMINAMATH_CALUDE_max_regions_lines_theorem_max_regions_circles_theorem_l3417_341718


namespace NUMINAMATH_CALUDE_square_circle_ratio_l3417_341722

theorem square_circle_ratio (s r : ℝ) (h : s > 0 ∧ r > 0) :
  s^2 / (π * r^2) = 250 / 196 →
  ∃ (a b c : ℕ), (a : ℝ) * Real.sqrt b / c = s / r ∧ a = 5 ∧ b = 10 ∧ c = 14 ∧ a + b + c = 29 :=
by sorry

end NUMINAMATH_CALUDE_square_circle_ratio_l3417_341722


namespace NUMINAMATH_CALUDE_range_of_b_minus_a_l3417_341761

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem range_of_b_minus_a (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, -1 ≤ f x ∧ f x ≤ 3) →
  (∃ x ∈ Set.Icc a b, f x = -1) →
  (∃ x ∈ Set.Icc a b, f x = 3) →
  2 ≤ b - a ∧ b - a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_minus_a_l3417_341761


namespace NUMINAMATH_CALUDE_solution_sets_l3417_341717

-- Define the set A as (-∞, 1)
def A : Set ℝ := Set.Iio 1

-- Define the solution set B
def B (a : ℝ) : Set ℝ :=
  if a < -1 then Set.Icc a (-1)
  else if a = -1 then {-1}
  else if -1 < a ∧ a < 0 then Set.Icc (-1) a
  else ∅

-- Theorem statement
theorem solution_sets (a : ℝ) (h1 : A = {x | a * x + (-2 * a) > 0}) :
  B a = {x | (a * x - (-2 * a)) * (x - a) ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_l3417_341717


namespace NUMINAMATH_CALUDE_yeast_growth_30_minutes_l3417_341794

/-- The number of yeast cells after a given number of 5-minute intervals -/
def yeast_population (initial_population : ℕ) (intervals : ℕ) : ℕ :=
  initial_population * 2^intervals

/-- Theorem: After 30 minutes (6 intervals), the yeast population will be 3200 -/
theorem yeast_growth_30_minutes :
  yeast_population 50 6 = 3200 := by
  sorry

end NUMINAMATH_CALUDE_yeast_growth_30_minutes_l3417_341794


namespace NUMINAMATH_CALUDE_final_value_calculation_l3417_341789

theorem final_value_calculation (initial_number : ℕ) : 
  initial_number = 10 → 3 * (2 * initial_number + 8) = 84 := by
  sorry

end NUMINAMATH_CALUDE_final_value_calculation_l3417_341789


namespace NUMINAMATH_CALUDE_simplify_expression_l3417_341757

theorem simplify_expression (m n : ℝ) : m - n - (m - n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3417_341757


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3417_341766

theorem simplify_trig_expression :
  2 * Real.sqrt (1 + Real.sin 4) + Real.sqrt (2 + 2 * Real.cos 4) = 2 * Real.sin 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3417_341766


namespace NUMINAMATH_CALUDE_concert_attendance_l3417_341767

theorem concert_attendance (total_tickets : ℕ) 
  (before_start : ℚ) (after_first_song : ℚ) (during_middle : ℕ) 
  (h1 : total_tickets = 900)
  (h2 : before_start = 3/4)
  (h3 : after_first_song = 5/9)
  (h4 : during_middle = 80) : 
  total_tickets - (before_start * total_tickets + 
    after_first_song * (total_tickets - before_start * total_tickets) + 
    during_middle) = 20 := by
sorry

end NUMINAMATH_CALUDE_concert_attendance_l3417_341767


namespace NUMINAMATH_CALUDE_dans_remaining_money_is_14_02_l3417_341720

/-- Calculates the remaining money after Dan's shopping trip -/
def dans_remaining_money (initial_money : ℚ) (candy_price : ℚ) (candy_count : ℕ) 
  (toy_price : ℚ) (toy_discount : ℚ) (sales_tax : ℚ) : ℚ :=
  let candy_total := candy_price * candy_count
  let discounted_toy := toy_price * (1 - toy_discount)
  let subtotal := candy_total + discounted_toy
  let total_with_tax := subtotal * (1 + sales_tax)
  initial_money - total_with_tax

/-- Theorem stating that Dan's remaining money after shopping is $14.02 -/
theorem dans_remaining_money_is_14_02 :
  dans_remaining_money 45 4 4 15 0.1 0.05 = 14.02 := by
  sorry

#eval dans_remaining_money 45 4 4 15 0.1 0.05

end NUMINAMATH_CALUDE_dans_remaining_money_is_14_02_l3417_341720


namespace NUMINAMATH_CALUDE_symmetric_quadratic_inequality_l3417_341787

/-- A quadratic function with positive leading coefficient and symmetric about x = 2 -/
def SymmetricQuadratic (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, a > 0 ∧ ∀ x, f x = a * x^2 + b * x + c) ∧
  (∀ x, f (2 + x) = f (2 - x))

theorem symmetric_quadratic_inequality
  (f : ℝ → ℝ) (h : SymmetricQuadratic f) (x : ℝ) :
  f (1 - 2 * x^2) < f (1 + 2 * x - x^2) → -2 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_quadratic_inequality_l3417_341787


namespace NUMINAMATH_CALUDE_triangle_properties_l3417_341758

/-- Triangle ABC with vertices A(1,3), B(3,1), and C(-1,0) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The specific triangle ABC given in the problem -/
def triangleABC : Triangle := {
  A := (1, 3)
  B := (3, 1)
  C := (-1, 0)
}

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to get the line equation of side AB -/
def getLineAB (t : Triangle) : LineEquation := sorry

/-- Function to calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

theorem triangle_properties (t : Triangle) (h : t = triangleABC) : 
  getLineAB t = { a := 1, b := 1, c := -4 } ∧ triangleArea t = 5 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3417_341758


namespace NUMINAMATH_CALUDE_prob_at_least_one_pair_not_three_of_kind_l3417_341744

/-- The number of faces on a standard die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The total number of possible outcomes when rolling the dice -/
def total_outcomes : ℕ := num_faces ^ num_dice

/-- The number of ways to choose a pair from the dice -/
def ways_to_choose_pair : ℕ := Nat.choose num_dice 2

/-- The number of successful outcomes (at least one pair but not a three-of-a-kind) -/
def successful_outcomes : ℕ := num_faces^3 * 25

/-- The probability of rolling at least one pair but not a three-of-a-kind -/
def probability : ℚ := successful_outcomes / total_outcomes

theorem prob_at_least_one_pair_not_three_of_kind : probability = 25 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_pair_not_three_of_kind_l3417_341744


namespace NUMINAMATH_CALUDE_sum_of_squares_l3417_341739

/-- Given a system of equations, prove that x² + y² + z² = 29 -/
theorem sum_of_squares (x y z : ℝ) 
  (eq1 : 2*x + y + 4*x*y + 6*x*z = -6)
  (eq2 : y + 2*z + 2*x*y + 6*y*z = 4)
  (eq3 : x - z + 2*x*z - 4*y*z = -3) :
  x^2 + y^2 + z^2 = 29 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3417_341739


namespace NUMINAMATH_CALUDE_boxes_with_neither_l3417_341755

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (crayons : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : markers = 8)
  (h3 : crayons = 5)
  (h4 : both = 3) :
  total - (markers + crayons - both) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l3417_341755


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3417_341738

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 8 * x - 5) - (2 * x^3 + x^2 + 6 * x - 7) = x^3 + 3 * x^2 + 2 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3417_341738


namespace NUMINAMATH_CALUDE_n_squared_not_divides_factorial_l3417_341701

theorem n_squared_not_divides_factorial (n : ℕ) :
  ¬(n^2 ∣ n!) ↔ n = 4 ∨ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_n_squared_not_divides_factorial_l3417_341701


namespace NUMINAMATH_CALUDE_two_distinct_integer_roots_l3417_341763

theorem two_distinct_integer_roots (r : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ r^2 * x^2 + 2*r*x + 4 = 28*r^2 ∧ r^2 * y^2 + 2*r*y + 4 = 28*r^2) ↔ 
  (r = 1 ∨ r = -1 ∨ r = 1/2 ∨ r = -1/2 ∨ r = 1/3 ∨ r = -1/3) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_integer_roots_l3417_341763


namespace NUMINAMATH_CALUDE_unknown_number_problem_l3417_341777

theorem unknown_number_problem : ∃ x : ℝ, 0.5 * 56 = 0.3 * x + 13 ∧ x = 50 := by sorry

end NUMINAMATH_CALUDE_unknown_number_problem_l3417_341777


namespace NUMINAMATH_CALUDE_bowlingPrizeOrders_l3417_341765

/-- Represents the number of bowlers in the tournament -/
def numBowlers : ℕ := 7

/-- Represents the number of playoff matches -/
def numMatches : ℕ := 6

/-- The number of possible outcomes for each match -/
def outcomesPerMatch : ℕ := 2

/-- Calculates the total number of possible prize orders -/
def totalPossibleOrders : ℕ := outcomesPerMatch ^ numMatches

/-- Proves that the number of different possible prize orders is 64 -/
theorem bowlingPrizeOrders : totalPossibleOrders = 64 := by
  sorry

end NUMINAMATH_CALUDE_bowlingPrizeOrders_l3417_341765


namespace NUMINAMATH_CALUDE_probability_same_tune_is_one_fourth_l3417_341793

/-- A defective toy train that produces two different tunes at random -/
structure DefectiveToyTrain :=
  (tunes : Fin 2 → String)

/-- The probability of the defective toy train producing 3 music tunes of the same type -/
def probability_same_tune (train : DefectiveToyTrain) : ℚ :=
  1 / 4

/-- Theorem stating that the probability of producing 3 music tunes of the same type is 1/4 -/
theorem probability_same_tune_is_one_fourth (train : DefectiveToyTrain) :
  probability_same_tune train = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_tune_is_one_fourth_l3417_341793


namespace NUMINAMATH_CALUDE_circle_properties_l3417_341783

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*y - 36 = -y^2 + 12*x + 16

def is_center (a b : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = (2 * Real.sqrt 23)^2

theorem circle_properties :
  ∃ a b : ℝ,
    is_center a b ∧
    a = 6 ∧
    b = 2 ∧
    a + b + 2 * Real.sqrt 23 = 8 + 2 * Real.sqrt 23 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3417_341783


namespace NUMINAMATH_CALUDE_point_reflection_origin_l3417_341723

/-- Given a point P(4, -3) in the Cartesian coordinate system,
    its coordinates with respect to the origin are (-4, 3). -/
theorem point_reflection_origin : 
  let P : ℝ × ℝ := (4, -3)
  let P_reflected : ℝ × ℝ := (-4, 3)
  P_reflected = (-(P.1), -(P.2)) :=
by sorry

end NUMINAMATH_CALUDE_point_reflection_origin_l3417_341723


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3417_341705

theorem inequality_solution_set (a : ℝ) (ha : a < 0) :
  {x : ℝ | a * x^2 - (a + 2) * x + 2 ≥ 0} = {x : ℝ | 2/a ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3417_341705


namespace NUMINAMATH_CALUDE_stationery_box_content_l3417_341709

theorem stationery_box_content : ∃ (S E : ℕ), S - E = 80 ∧ S = 4 * E ∧ S = 320 := by
  sorry

end NUMINAMATH_CALUDE_stationery_box_content_l3417_341709


namespace NUMINAMATH_CALUDE_product_not_divisible_by_72_l3417_341776

def S : Finset Nat := {4, 8, 18, 28, 36, 49, 56}

theorem product_not_divisible_by_72 (a b : Nat) (ha : a ∈ S) (hb : b ∈ S) (hab : a ≠ b) :
  ¬(72 ∣ a * b) := by
  sorry

#check product_not_divisible_by_72

end NUMINAMATH_CALUDE_product_not_divisible_by_72_l3417_341776
