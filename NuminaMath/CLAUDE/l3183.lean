import Mathlib

namespace sum_min_max_cubic_quartic_l3183_318381

theorem sum_min_max_cubic_quartic (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 8)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 18) : 
  let f := fun (x y z w : ℝ) => 3 * (x^3 + y^3 + z^3 + w^3) - 2 * (x^4 + y^4 + z^4 + w^4)
  ∃ (m M : ℝ), (∀ (x y z w : ℝ), (x + y + z + w = 8 ∧ x^2 + y^2 + z^2 + w^2 = 18) → 
    m ≤ f x y z w ∧ f x y z w ≤ M) ∧ m + M = 29 :=
sorry

end sum_min_max_cubic_quartic_l3183_318381


namespace unknown_number_theorem_l3183_318332

theorem unknown_number_theorem (X : ℝ) : 30 = 0.50 * X + 10 → X = 40 := by
  sorry

end unknown_number_theorem_l3183_318332


namespace equation_solutions_l3183_318314

theorem equation_solutions (b c : ℝ) : 
  (∀ x : ℝ, (|x - 4| = 3) ↔ (x^2 + b*x + c = 0)) → 
  (b = -8 ∧ c = 7) := by
sorry

end equation_solutions_l3183_318314


namespace oak_trees_planted_l3183_318389

/-- Given the initial and final number of oak trees in a park, 
    prove that the number of new trees planted is their difference -/
theorem oak_trees_planted (initial final : ℕ) (h : final ≥ initial) :
  final - initial = final - initial :=
by sorry

end oak_trees_planted_l3183_318389


namespace edwards_remaining_money_l3183_318319

/-- 
Given that Edward had $18 initially and spent $16, 
prove that his remaining money is $2.
-/
theorem edwards_remaining_money :
  let initial_amount : ℕ := 18
  let spent_amount : ℕ := 16
  let remaining_amount : ℕ := initial_amount - spent_amount
  remaining_amount = 2 := by sorry

end edwards_remaining_money_l3183_318319


namespace cube_split_l3183_318371

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The nth odd number -/
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

theorem cube_split (m : ℕ) (h1 : m > 1) :
  (∃ k : ℕ, k > 0 ∧ k < m ∧ nth_odd (triangular_number k + 1) = 59) →
  m = 8 := by sorry

end cube_split_l3183_318371


namespace max_abs_z_given_condition_l3183_318362

theorem max_abs_z_given_condition (z : ℂ) (h : Complex.abs (z + Complex.I) + Complex.abs (z - Complex.I) = 2) : 
  Complex.abs z ≤ 1 := by
  sorry

end max_abs_z_given_condition_l3183_318362


namespace square_perimeter_problem_l3183_318383

theorem square_perimeter_problem (perimeter_I perimeter_II : ℝ) 
  (h1 : perimeter_I = 16)
  (h2 : perimeter_II = 36)
  (side_I : ℝ) (side_II : ℝ) (side_III : ℝ)
  (h3 : side_I = perimeter_I / 4)
  (h4 : side_II = perimeter_II / 4)
  (h5 : side_III = Real.sqrt (side_I * side_II))
  (perimeter_III : ℝ)
  (h6 : perimeter_III = 4 * side_III) :
  perimeter_III = 24 := by
sorry

end square_perimeter_problem_l3183_318383


namespace cube_sum_divisible_by_nine_l3183_318353

theorem cube_sum_divisible_by_nine (n : ℕ+) :
  ∃ k : ℤ, n^3 + (n+1)^3 + (n+2)^3 = 9 * k := by
sorry

end cube_sum_divisible_by_nine_l3183_318353


namespace triangle_formation_l3183_318330

/-- Triangle inequality check for three sides -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 3 3 6 ∧
  ¬ can_form_triangle 2 5 7 ∧
  can_form_triangle 4 5 6 :=
sorry

end triangle_formation_l3183_318330


namespace geckos_sold_last_year_l3183_318325

theorem geckos_sold_last_year (x : ℕ) : 
  x + 2 * x = 258 → x = 86 := by
  sorry

end geckos_sold_last_year_l3183_318325


namespace triangle_area_l3183_318393

theorem triangle_area (a b c : ℝ) (A : ℝ) :
  A = π / 3 →  -- 60° in radians
  a = Real.sqrt 3 →
  b + c = 3 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2 := by
  sorry

end triangle_area_l3183_318393


namespace period_of_cosine_l3183_318385

theorem period_of_cosine (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ Real.cos ((3 * x) / 4)
  ∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x ∧ T = (8 * Real.pi) / 3 :=
by
  sorry

end period_of_cosine_l3183_318385


namespace specific_triangle_toothpicks_l3183_318378

/-- Represents the configuration of a large equilateral triangle made of small triangles --/
structure TriangleConfig where
  rows : Nat
  base_triangles : Nat
  double_count_start : Nat

/-- Calculates the total number of toothpicks required for a given triangle configuration --/
def total_toothpicks (config : TriangleConfig) : Nat :=
  sorry

/-- Theorem stating that the specific configuration requires 1617 toothpicks --/
theorem specific_triangle_toothpicks :
  let config : TriangleConfig := {
    rows := 5,
    base_triangles := 100,
    double_count_start := 2
  }
  total_toothpicks config = 1617 := by
  sorry

end specific_triangle_toothpicks_l3183_318378


namespace triangle_area_is_16_l3183_318320

/-- The area of the triangle formed by the intersection of three lines -/
def triangleArea (line1 line2 line3 : ℝ → ℝ) : ℝ := sorry

/-- Line 1: y = 6 -/
def line1 : ℝ → ℝ := fun x ↦ 6

/-- Line 2: y = 2 + x -/
def line2 : ℝ → ℝ := fun x ↦ 2 + x

/-- Line 3: y = 2 - x -/
def line3 : ℝ → ℝ := fun x ↦ 2 - x

theorem triangle_area_is_16 : triangleArea line1 line2 line3 = 16 := by sorry

end triangle_area_is_16_l3183_318320


namespace bus_driver_compensation_l3183_318345

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_rate_increase : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 16)
  (h2 : regular_hours = 40)
  (h3 : overtime_rate_increase = 0.75)
  (h4 : total_hours = 52) :
  let overtime_hours := total_hours - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  regular_pay + overtime_pay = 976 := by
sorry


end bus_driver_compensation_l3183_318345


namespace decimal_expression_simplification_l3183_318377

theorem decimal_expression_simplification :
  (0.00001 * (0.01)^2 * 1000) / 0.001 = 10^(-3) := by
  sorry

end decimal_expression_simplification_l3183_318377


namespace solution_set_inequality_l3183_318324

theorem solution_set_inequality (x : ℝ) : (2*x + 1) / (x + 1) < 1 ↔ -1 < x ∧ x < 0 := by
  sorry

end solution_set_inequality_l3183_318324


namespace arithmetic_sequence_a2_l3183_318366

/-- An arithmetic sequence {aₙ} -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_a2 (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 5 = 15) 
  (h_a6 : a 6 = 7) : 
  a 2 = 8 := by
sorry

end arithmetic_sequence_a2_l3183_318366


namespace function_inequality_implies_m_bound_l3183_318310

theorem function_inequality_implies_m_bound (f g : ℝ → ℝ) (m : ℝ) 
  (hf : ∀ x, f x = x^2)
  (hg : ∀ x, g x = (1/2)^x - m)
  (h : ∀ x₁ ∈ Set.Icc 0 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g x₂) :
  m ≥ 1/4 := by
sorry

end function_inequality_implies_m_bound_l3183_318310


namespace circle_region_area_l3183_318360

/-- Given a circle with radius 36 and two chords of length 90 intersecting at a point 12 units from the center,
    the area of one of the regions formed can be expressed as 216π, which is equivalent to aπ - b√c
    where a + b + c = 216 and a, b, c are positive integers with c not divisible by the square of any prime. -/
theorem circle_region_area (r : ℝ) (chord_length : ℝ) (intersection_distance : ℝ)
  (h_radius : r = 36)
  (h_chord : chord_length = 90)
  (h_intersection : intersection_distance = 12) :
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (∀ (p : ℕ), Prime p → c % (p^2) ≠ 0) ∧
    (a + b + c = 216) ∧
    (Real.pi * (a : ℝ) - (b : ℝ) * Real.sqrt (c : ℝ) = 216 * Real.pi) :=
by sorry

end circle_region_area_l3183_318360


namespace cos_squared_alpha_plus_pi_fourth_l3183_318394

theorem cos_squared_alpha_plus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + π / 4) ^ 2 = 1 / 6 := by
  sorry

end cos_squared_alpha_plus_pi_fourth_l3183_318394


namespace quadratic_function_properties_l3183_318346

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_function_properties (a b : ℝ) :
  f a b 0 = 6 ∧ f a b 1 = 5 →
  (∀ x, f a b x = x^2 - 2*x + 6) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a b x ≥ 5) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a b x ≤ 14) ∧
  (∃ x ∈ Set.Icc (-2) 2, f a b x = 5) ∧
  (∃ x ∈ Set.Icc (-2) 2, f a b x = 14) :=
by sorry

end quadratic_function_properties_l3183_318346


namespace nth_inequality_l3183_318343

theorem nth_inequality (x : ℝ) (n : ℕ) (h : x > 0) : 
  x + (n^n : ℝ) / x^n ≥ n + 1 := by
sorry

end nth_inequality_l3183_318343


namespace unique_cut_l3183_318358

/-- Represents a cut of the original number -/
structure Cut where
  pos1 : Nat
  pos2 : Nat
  valid : pos1 < pos2 ∧ pos2 < 5

/-- Checks if a given cut produces the required difference -/
def isValidCut (c : Cut) : Prop :=
  let part1 := 12345 / (10^(5 - c.pos1))
  let part2 := (12345 / (10^(5 - c.pos2))) % (10^(c.pos2 - c.pos1))
  let part3 := 12345 % (10^(5 - c.pos2))
  (part1 * 10^4 + part2 * 10^(5 - c.pos2) + part3) -
  (part2 * 10^4 + part3 * 10^c.pos1 + part1) = 28926

theorem unique_cut : 
  ∃! c : Cut, isValidCut c ∧ c.pos1 = 1 ∧ c.pos2 = 4 :=
sorry

end unique_cut_l3183_318358


namespace min_dot_product_l3183_318317

theorem min_dot_product (a b : ℝ × ℝ) (h : |3 * (a.1 * b.1 + a.2 * b.2)| ≤ 4) :
  ∃ (c d : ℝ × ℝ), c.1 * d.1 + c.2 * d.2 ≥ a.1 * b.1 + a.2 * b.2 ∧ 
  |3 * (c.1 * d.1 + c.2 * d.2)| ≤ 4 ∧ 
  c.1 * d.1 + c.2 * d.2 = -4/3 := by
sorry

end min_dot_product_l3183_318317


namespace rohan_join_time_is_seven_l3183_318322

/-- Represents the investment scenario and profit distribution --/
structure InvestmentScenario where
  suresh_investment : ℕ
  rohan_investment : ℕ
  sudhir_investment : ℕ
  total_profit : ℕ
  rohan_sudhir_diff : ℕ
  total_months : ℕ
  sudhir_join_time : ℕ

/-- Calculates the number of months after which Rohan joined the business --/
def calculate_rohan_join_time (scenario : InvestmentScenario) : ℕ :=
  sorry

/-- Theorem stating that Rohan joined after 7 months --/
theorem rohan_join_time_is_seven (scenario : InvestmentScenario) 
  (h1 : scenario.suresh_investment = 18000)
  (h2 : scenario.rohan_investment = 12000)
  (h3 : scenario.sudhir_investment = 9000)
  (h4 : scenario.total_profit = 3795)
  (h5 : scenario.rohan_sudhir_diff = 345)
  (h6 : scenario.total_months = 12)
  (h7 : scenario.sudhir_join_time = 8) : 
  calculate_rohan_join_time scenario = 7 :=
sorry

end rohan_join_time_is_seven_l3183_318322


namespace boat_current_speed_l3183_318316

/-- Proves that given a boat with a speed of 15 km/hr in still water,
    traveling 3.6 km downstream in 12 minutes, the rate of the current is 3 km/hr. -/
theorem boat_current_speed (boat_speed : ℝ) (downstream_distance : ℝ) (time_minutes : ℝ) 
  (h1 : boat_speed = 15)
  (h2 : downstream_distance = 3.6)
  (h3 : time_minutes = 12) : 
  let time_hours : ℝ := time_minutes / 60
  let current_speed : ℝ := downstream_distance / time_hours - boat_speed
  current_speed = 3 := by
  sorry

end boat_current_speed_l3183_318316


namespace tetrahedron_volume_l3183_318309

/-- Tetrahedron with given edge lengths -/
structure Tetrahedron where
  AB : ℝ
  AC : ℝ
  AD : ℝ
  BC : ℝ
  BD : ℝ
  CD : ℝ

/-- Volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- The theorem to be proved -/
theorem tetrahedron_volume (t : Tetrahedron) 
  (h1 : t.AB = 4)
  (h2 : t.AC = 5)
  (h3 : t.AD = 6)
  (h4 : t.BC = 2 * Real.sqrt 7)
  (h5 : t.BD = 5)
  (h6 : t.CD = Real.sqrt 34) :
  volume t = 6 * Real.sqrt 1301 := by sorry

end tetrahedron_volume_l3183_318309


namespace AMC9_paths_count_l3183_318305

/-- Represents the layout of the AMC9 puzzle --/
structure AMC9Layout where
  start_A : Nat
  adjacent_Ms : Nat
  adjacent_Cs : Nat
  Cs_with_two_9s : Nat
  Cs_with_one_9 : Nat

/-- Calculates the number of paths in the AMC9 puzzle --/
def count_AMC9_paths (layout : AMC9Layout) : Nat :=
  layout.adjacent_Ms * 
  (layout.Cs_with_two_9s * 2 + layout.Cs_with_one_9 * 1)

/-- Theorem stating that the number of paths in the AMC9 puzzle is 20 --/
theorem AMC9_paths_count :
  ∀ (layout : AMC9Layout),
  layout.start_A = 1 →
  layout.adjacent_Ms = 4 →
  layout.adjacent_Cs = 3 →
  layout.Cs_with_two_9s = 2 →
  layout.Cs_with_one_9 = 1 →
  count_AMC9_paths layout = 20 := by
  sorry

end AMC9_paths_count_l3183_318305


namespace magician_earnings_proof_l3183_318399

def magician_earnings (price : ℕ) (initial_decks : ℕ) (final_decks : ℕ) : ℕ :=
  (initial_decks - final_decks) * price

theorem magician_earnings_proof (price : ℕ) (initial_decks : ℕ) (final_decks : ℕ) 
  (h1 : price = 2)
  (h2 : initial_decks = 5)
  (h3 : final_decks = 3) :
  magician_earnings price initial_decks final_decks = 4 := by
  sorry

end magician_earnings_proof_l3183_318399


namespace range_of_m_l3183_318326

/-- Given conditions for the problem -/
structure ProblemConditions (m : ℝ) :=
  (h1 : ∃ x : ℝ, (x^2 + 1) * (x^2 - 8*x - 20) ≤ 0)
  (h2 : ∃ x : ℝ, x^2 - 2*x + 1 - m^2 ≤ 0)
  (h3 : m > 0)
  (h4 : ∀ x : ℝ, (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m))
  (h5 : ∃ x : ℝ, (x < -2 ∨ x > 10) ∧ ¬(x < 1 - m ∨ x > 1 + m))

/-- The main theorem stating the range of m -/
theorem range_of_m (m : ℝ) (h : ProblemConditions m) : m ≥ 9 :=
sorry

end range_of_m_l3183_318326


namespace parallel_lines_problem_l3183_318368

/-- The number of parallelograms formed by the intersection of two sets of parallel lines -/
def num_parallelograms (n : ℕ) (m : ℕ) : ℕ := n.choose 2 * m.choose 2

/-- The theorem statement -/
theorem parallel_lines_problem (n : ℕ) :
  num_parallelograms n 8 = 420 ↔ n = 6 := by sorry

end parallel_lines_problem_l3183_318368


namespace cuboid_volume_l3183_318354

/-- Given a cuboid with face areas 3, 5, and 15 sharing a common vertex, its volume is 15 -/
theorem cuboid_volume (a b c : ℝ) 
  (h1 : a * b = 3) 
  (h2 : a * c = 5) 
  (h3 : b * c = 15) : 
  a * b * c = 15 := by
  sorry

#check cuboid_volume

end cuboid_volume_l3183_318354


namespace framed_painting_perimeter_l3183_318311

/-- The perimeter of a framed rectangular painting -/
theorem framed_painting_perimeter
  (height : ℕ) -- Height of the painting
  (width : ℕ) -- Width of the painting
  (frame_width : ℕ) -- Width of the frame
  (h1 : height = 12)
  (h2 : width = 15)
  (h3 : frame_width = 3) :
  2 * (height + 2 * frame_width + width + 2 * frame_width) = 78 :=
by sorry

end framed_painting_perimeter_l3183_318311


namespace factorial_difference_quotient_l3183_318301

theorem factorial_difference_quotient : (Nat.factorial 13 - Nat.factorial 12) / Nat.factorial 10 = 1584 := by
  sorry

end factorial_difference_quotient_l3183_318301


namespace equal_function_values_l3183_318396

/-- Given a function f(x) = ax^2 - 2ax + 1 where a > 1, prove that f(x₁) = f(x₂) when x₁ < x₂ and x₁ + x₂ = 1 + a -/
theorem equal_function_values
  (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 1)
  (hx : x₁ < x₂)
  (hsum : x₁ + x₂ = 1 + a)
  : a * x₁^2 - 2*a*x₁ + 1 = a * x₂^2 - 2*a*x₂ + 1 :=
by sorry

end equal_function_values_l3183_318396


namespace range_of_m_l3183_318313

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (B m).Nonempty ∧ B m ⊆ A → 2 ≤ m ∧ m ≤ 3 := by sorry

end range_of_m_l3183_318313


namespace unique_solution_l3183_318344

def system_solution (x y : ℝ) : Prop :=
  x + y = 1 ∧ x - y = -1

theorem unique_solution : 
  ∃! p : ℝ × ℝ, system_solution p.1 p.2 ∧ p = (0, 1) :=
sorry

end unique_solution_l3183_318344


namespace min_value_expression_l3183_318384

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 9*a*b + 9*b^2 + 3*c^2 ≥ 60 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 27 ∧ 
    a₀^2 + 9*a₀*b₀ + 9*b₀^2 + 3*c₀^2 = 60 :=
by sorry

end min_value_expression_l3183_318384


namespace square_perimeter_problem_l3183_318367

theorem square_perimeter_problem (A B C : ℝ) : 
  -- A, B, and C represent the side lengths of squares A, B, and C respectively
  (4 * A = 20) →  -- Perimeter of A is 20 units
  (4 * B = 32) →  -- Perimeter of B is 32 units
  (C = A / 2 + 2 * B) →  -- Side length of C definition
  (4 * C = 74) -- Perimeter of C is 74 units
  := by sorry

end square_perimeter_problem_l3183_318367


namespace sector_area_l3183_318334

/-- Given a circular sector with circumference 8 and central angle 2 radians, its area is 4. -/
theorem sector_area (c : ℝ) (θ : ℝ) (h1 : c = 8) (h2 : θ = 2) :
  let r := c / (2 + 2 * Real.pi)
  (1/2) * r^2 * θ = 4 := by sorry

end sector_area_l3183_318334


namespace roses_in_garden_l3183_318369

theorem roses_in_garden (rows : ℕ) (roses_per_row : ℕ) 
  (red_fraction : ℚ) (white_fraction : ℚ) :
  rows = 10 →
  roses_per_row = 20 →
  red_fraction = 1/2 →
  white_fraction = 3/5 →
  (rows * roses_per_row * (1 - red_fraction) * (1 - white_fraction) : ℚ) = 40 := by
  sorry

end roses_in_garden_l3183_318369


namespace sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3183_318375

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let roots := {x : ℝ | a * x^2 + b * x + c = 0}
  (∃ x y : ℝ, roots = {x, y}) →
  (∃ s : ℝ, ∀ z ∈ roots, ∃ w ∈ roots, z + w = s) →
  (∃ s : ℝ, ∀ z ∈ roots, ∃ w ∈ roots, z + w = s) → s = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let roots := {x : ℝ | x^2 - 6*x + 8 = 0}
  (∃ x y : ℝ, roots = {x, y}) →
  (∃ s : ℝ, ∀ z ∈ roots, ∃ w ∈ roots, z + w = s) →
  s = 6 :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3183_318375


namespace four_digit_sum_4360_l3183_318308

def is_valid_insertion (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧ 
    ((a = 2 ∧ b = 1 ∧ d = 5) ∨ (a = 2 ∧ c = 1 ∧ d = 5))

theorem four_digit_sum_4360 :
  ∀ (n₁ n₂ : ℕ), is_valid_insertion n₁ → is_valid_insertion n₂ → n₁ + n₂ = 4360 →
    ((n₁ = 2195 ∧ n₂ = 2165) ∨ (n₁ = 2185 ∧ n₂ = 2175) ∨ (n₁ = 2215 ∧ n₂ = 2145)) :=
by sorry

end four_digit_sum_4360_l3183_318308


namespace income_increase_percentage_l3183_318333

theorem income_increase_percentage 
  (initial_income : ℝ) 
  (initial_expenditure_ratio : ℝ) 
  (expenditure_increase_ratio : ℝ) 
  (savings_increase_ratio : ℝ) 
  (income_increase_ratio : ℝ)
  (h1 : initial_expenditure_ratio = 0.75)
  (h2 : expenditure_increase_ratio = 1.1)
  (h3 : savings_increase_ratio = 1.5)
  (h4 : income_increase_ratio > 0)
  : income_increase_ratio = 1.2 := by
  sorry

#check income_increase_percentage

end income_increase_percentage_l3183_318333


namespace inequality_proof_l3183_318306

theorem inequality_proof (a b c d p q : ℝ) 
  (h1 : a * b + c * d = 2 * p * q)
  (h2 : a * c ≥ p ^ 2)
  (h3 : p > 0) : 
  b * d ≤ q ^ 2 := by
sorry

end inequality_proof_l3183_318306


namespace christinas_walking_speed_l3183_318361

/-- The problem of finding Christina's walking speed -/
theorem christinas_walking_speed 
  (initial_distance : ℝ) 
  (jacks_speed : ℝ) 
  (lindys_speed : ℝ) 
  (lindys_total_distance : ℝ) 
  (h1 : initial_distance = 270) 
  (h2 : jacks_speed = 4) 
  (h3 : lindys_speed = 8) 
  (h4 : lindys_total_distance = 240) : 
  ∃ (christinas_speed : ℝ), christinas_speed = 5 := by
  sorry

#check christinas_walking_speed

end christinas_walking_speed_l3183_318361


namespace dodecahedron_volume_greater_than_icosahedron_l3183_318352

/-- A regular dodecahedron -/
structure Dodecahedron where
  radius : ℝ
  volume : ℝ

/-- A regular icosahedron -/
structure Icosahedron where
  radius : ℝ
  volume : ℝ

/-- The volume of a dodecahedron inscribed in a sphere is greater than 
    the volume of an icosahedron inscribed in the same sphere -/
theorem dodecahedron_volume_greater_than_icosahedron 
  (D : Dodecahedron) (I : Icosahedron) (h : D.radius = I.radius) :
  D.volume > I.volume := by
  sorry

end dodecahedron_volume_greater_than_icosahedron_l3183_318352


namespace probability_sum_greater_than_six_l3183_318359

def dice_outcomes : ℕ := 6 * 6

def favorable_outcomes : ℕ := 21

theorem probability_sum_greater_than_six :
  (favorable_outcomes : ℚ) / dice_outcomes = 7 / 12 :=
sorry

end probability_sum_greater_than_six_l3183_318359


namespace ravi_overall_profit_l3183_318386

/-- Calculates the overall profit for Ravi's purchases and sales -/
theorem ravi_overall_profit (refrigerator_cost mobile_cost : ℝ)
  (refrigerator_loss_percent mobile_profit_percent : ℝ) :
  refrigerator_cost = 15000 →
  mobile_cost = 8000 →
  refrigerator_loss_percent = 4 →
  mobile_profit_percent = 10 →
  let refrigerator_selling_price := refrigerator_cost * (1 - refrigerator_loss_percent / 100)
  let mobile_selling_price := mobile_cost * (1 + mobile_profit_percent / 100)
  let total_cost := refrigerator_cost + mobile_cost
  let total_selling_price := refrigerator_selling_price + mobile_selling_price
  total_selling_price - total_cost = 200 :=
by sorry

end ravi_overall_profit_l3183_318386


namespace triangle_side_value_l3183_318379

theorem triangle_side_value (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  a * b = 60 →
  (1 / 2) * a * b * Real.sin C = 15 * Real.sqrt 3 →
  R = Real.sqrt 3 →
  c = 2 * R * Real.sin C →
  c = 3 := by
sorry

end triangle_side_value_l3183_318379


namespace complex_equation_l3183_318390

theorem complex_equation (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : z = 1 - Complex.I := by
  sorry

end complex_equation_l3183_318390


namespace root_product_sum_l3183_318350

theorem root_product_sum (p q r : ℝ) : 
  (4 * p^3 - 6 * p^2 + 17 * p - 10 = 0) →
  (4 * q^3 - 6 * q^2 + 17 * q - 10 = 0) →
  (4 * r^3 - 6 * r^2 + 17 * r - 10 = 0) →
  p * q + p * r + q * r = 17/4 := by
sorry

end root_product_sum_l3183_318350


namespace min_green_tiles_l3183_318338

/-- Represents the colors of tiles --/
inductive Color
  | Red
  | Orange
  | Yellow
  | Green
  | Blue
  | Indigo

/-- Represents the number of tiles for each color --/
structure TileCount where
  red : ℕ
  orange : ℕ
  yellow : ℕ
  green : ℕ
  blue : ℕ
  indigo : ℕ

/-- The total number of tiles --/
def total_tiles : ℕ := 100

/-- Checks if the tile count satisfies all constraints --/
def satisfies_constraints (tc : TileCount) : Prop :=
  tc.red + tc.orange + tc.yellow + tc.green + tc.blue + tc.indigo = total_tiles ∧
  tc.indigo ≥ tc.red + tc.orange + tc.yellow + tc.green + tc.blue ∧
  tc.blue ≥ tc.red + tc.orange + tc.yellow + tc.green ∧
  tc.green ≥ tc.red + tc.orange + tc.yellow ∧
  tc.yellow ≥ tc.red + tc.orange ∧
  tc.orange ≥ tc.red

/-- Checks if one tile count is preferred over another according to the client's preferences --/
def is_preferred (tc1 tc2 : TileCount) : Prop :=
  tc1.red > tc2.red ∨
  (tc1.red = tc2.red ∧ tc1.orange > tc2.orange) ∨
  (tc1.red = tc2.red ∧ tc1.orange = tc2.orange ∧ tc1.yellow > tc2.yellow) ∨
  (tc1.red = tc2.red ∧ tc1.orange = tc2.orange ∧ tc1.yellow = tc2.yellow ∧ tc1.green > tc2.green) ∨
  (tc1.red = tc2.red ∧ tc1.orange = tc2.orange ∧ tc1.yellow = tc2.yellow ∧ tc1.green = tc2.green ∧ tc1.blue > tc2.blue) ∨
  (tc1.red = tc2.red ∧ tc1.orange = tc2.orange ∧ tc1.yellow = tc2.yellow ∧ tc1.green = tc2.green ∧ tc1.blue = tc2.blue ∧ tc1.indigo > tc2.indigo)

/-- The theorem to be proved --/
theorem min_green_tiles :
  ∃ (optimal : TileCount),
    satisfies_constraints optimal ∧
    optimal.green = 13 ∧
    ∀ (tc : TileCount), satisfies_constraints tc → ¬is_preferred tc optimal :=
by sorry

end min_green_tiles_l3183_318338


namespace max_fraction_sum_l3183_318387

def DigitSet : Set Nat := {2, 3, 4, 5, 6, 7, 8, 9}

def ValidOptions : Set Rat := {2/17, 3/17, 17/72, 25/72, 13/36}

theorem max_fraction_sum (A B C D : Nat) :
  A ∈ DigitSet → B ∈ DigitSet → C ∈ DigitSet → D ∈ DigitSet →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (A : Rat) / B + (C : Rat) / D ∈ ValidOptions →
  ∀ (X Y Z W : Nat), X ∈ DigitSet → Y ∈ DigitSet → Z ∈ DigitSet → W ∈ DigitSet →
    X ≠ Y → X ≠ Z → X ≠ W → Y ≠ Z → Y ≠ W → Z ≠ W →
    (X : Rat) / Y + (Z : Rat) / W ∈ ValidOptions →
    (X : Rat) / Y + (Z : Rat) / W ≤ (A : Rat) / B + (C : Rat) / D →
  (A : Rat) / B + (C : Rat) / D = 25 / 72 :=
sorry

end max_fraction_sum_l3183_318387


namespace sum_ratio_equals_four_sevenths_l3183_318327

theorem sum_ratio_equals_four_sevenths 
  (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 16)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 49)
  (sum_products : a*x + b*y + c*z = 28) :
  (a + b + c) / (x + y + z) = 4/7 := by
sorry

end sum_ratio_equals_four_sevenths_l3183_318327


namespace rock_age_count_l3183_318341

/-- The set of digits used to form the rock's age -/
def rock_age_digits : Finset Nat := {2, 3, 7, 9}

/-- The number of occurrences of each digit in the rock's age -/
def digit_occurrences : Nat → Nat
  | 2 => 3
  | 3 => 1
  | 7 => 1
  | 9 => 1
  | _ => 0

/-- The set of odd digits that can start the rock's age -/
def odd_start_digits : Finset Nat := {3, 7, 9}

/-- The length of the rock's age in digits -/
def age_length : Nat := 6

/-- The number of possibilities for the rock's age -/
def rock_age_possibilities : Nat := 60

theorem rock_age_count :
  (Finset.card odd_start_digits) *
  (Nat.factorial (age_length - 1)) /
  (Nat.factorial (digit_occurrences 2)) =
  rock_age_possibilities := by sorry

end rock_age_count_l3183_318341


namespace simplify_expression_l3183_318355

theorem simplify_expression (x : ℝ) : 105 * x - 58 * x = 47 * x := by
  sorry

end simplify_expression_l3183_318355


namespace equation_solution_l3183_318351

theorem equation_solution :
  ∃ x : ℝ, (2*x - 1)^2 - (1 - 3*x)^2 = 5*(1 - x)*(x + 1) ∧ x = 5/2 := by
  sorry

end equation_solution_l3183_318351


namespace factorization_proof_l3183_318395

theorem factorization_proof (x y : ℝ) : 91 * x^7 - 273 * x^14 * y^3 = 91 * x^7 * (1 - 3 * x^7 * y^3) := by
  sorry

end factorization_proof_l3183_318395


namespace perpendicular_slope_correct_l3183_318363

-- Define the slope of the given line
def given_line_slope : ℚ := 3 / 4

-- Define the slope of the perpendicular line
def perpendicular_slope : ℚ := -4 / 3

-- Theorem stating that the perpendicular slope is correct
theorem perpendicular_slope_correct :
  perpendicular_slope = -1 / given_line_slope :=
by sorry

end perpendicular_slope_correct_l3183_318363


namespace hayley_stickers_l3183_318380

/-- The number of Hayley's close friends who like stickers. -/
def num_friends : ℕ := 9

/-- The number of stickers each friend would receive if distributed equally. -/
def stickers_per_friend : ℕ := 8

/-- The total number of stickers Hayley has. -/
def total_stickers : ℕ := num_friends * stickers_per_friend

theorem hayley_stickers : total_stickers = 72 := by
  sorry

end hayley_stickers_l3183_318380


namespace square_area_6cm_l3183_318328

theorem square_area_6cm (side_length : ℝ) (h : side_length = 6) :
  side_length * side_length = 36 := by
  sorry

end square_area_6cm_l3183_318328


namespace remainder_problem_l3183_318370

theorem remainder_problem (n : ℕ) : 
  n % 12 = 22 → 
  ((n % 34) + (n % 12)) % 12 = 10 → 
  n % 34 = 10 := by
sorry

end remainder_problem_l3183_318370


namespace relationship_abc_l3183_318302

theorem relationship_abc : 
  let a := Real.log 2
  let b := 5^(-1/2 : ℝ)
  let c := (1/4 : ℝ) * ∫ x in (0 : ℝ)..(π : ℝ), Real.sin x
  b < c ∧ c < a := by sorry

end relationship_abc_l3183_318302


namespace students_without_A_l3183_318329

theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (both : ℕ)
  (h_total : total = 40)
  (h_history : history = 10)
  (h_math : math = 18)
  (h_both : both = 6) :
  total - (history + math - both) = 18 :=
by sorry

end students_without_A_l3183_318329


namespace david_pushups_difference_l3183_318336

/-- Proves that David did 17 more push-ups than Zachary given the conditions in the problem -/
theorem david_pushups_difference (david_crunches zachary_pushups zachary_crunches : ℕ) 
  (h1 : david_crunches = 45)
  (h2 : zachary_pushups = 34)
  (h3 : zachary_crunches = 62)
  (h4 : david_crunches + 17 = zachary_crunches) : 
  ∃ (david_pushups : ℕ), david_pushups - zachary_pushups = 17 := by
sorry

end david_pushups_difference_l3183_318336


namespace exists_unsolvable_configuration_l3183_318372

/-- Represents a chessboard with integers -/
def Chessboard := Matrix (Fin 2018) (Fin 2019) ℤ

/-- Represents a set of selected cells on the chessboard -/
def SelectedCells := Set (Fin 2018 × Fin 2019)

/-- Performs one step of the operation on the chessboard -/
def perform_operation (board : Chessboard) (selected : SelectedCells) : Chessboard :=
  sorry

/-- Checks if all numbers on the board are equal -/
def all_equal (board : Chessboard) : Prop :=
  sorry

/-- Theorem stating that there exists a chessboard configuration where it's impossible to make all numbers equal -/
theorem exists_unsolvable_configuration :
  ∃ (initial_board : Chessboard),
    ∀ (operations : List SelectedCells),
      ¬(all_equal (operations.foldl perform_operation initial_board)) :=
sorry

end exists_unsolvable_configuration_l3183_318372


namespace darnell_initial_fabric_l3183_318347

/-- Calculates the initial amount of fabric Darnell had --/
def initial_fabric (square_side : ℕ) (wide_length wide_width : ℕ) (tall_length tall_width : ℕ)
  (num_square num_wide num_tall : ℕ) (fabric_left : ℕ) : ℕ :=
  let square_area := square_side * square_side
  let wide_area := wide_length * wide_width
  let tall_area := tall_length * tall_width
  let total_used := square_area * num_square + wide_area * num_wide + tall_area * num_tall
  total_used + fabric_left

/-- Theorem stating that Darnell initially had 1000 square feet of fabric --/
theorem darnell_initial_fabric :
  initial_fabric 4 5 3 3 5 16 20 10 294 = 1000 := by
  sorry

end darnell_initial_fabric_l3183_318347


namespace square_cut_diagonal_length_l3183_318365

theorem square_cut_diagonal_length (s x : ℝ) : 
  s > 0 → 
  x > 0 → 
  x^2 = 72 → 
  s^2 = 2 * x^2 → 
  (s - 2*x)^2 + (s - 2*x)^2 = 12^2 := by
sorry

end square_cut_diagonal_length_l3183_318365


namespace trajectory_of_Q_l3183_318342

/-- Given a line segment PQ with midpoint M(0,4) and P moving along x + y - 2 = 0,
    prove that the trajectory of Q is x + y - 6 = 0 -/
theorem trajectory_of_Q (P Q : ℝ × ℝ) (t : ℝ) : 
  let M := (0, 4)
  let P := (t, 2 - t)  -- parametric form of x + y - 2 = 0
  let Q := (2 * M.1 - P.1, 2 * M.2 - P.2)  -- Q is symmetric to P with respect to M
  Q.1 + Q.2 = 6 := by
  sorry

end trajectory_of_Q_l3183_318342


namespace binary_to_octal_conversion_l3183_318374

/-- Convert a binary number represented as a list of bits to decimal --/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Convert a decimal number to octal --/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_to_octal_conversion :
  let binary := [false, false, true, true, true, false, true, true]
  let decimal := binary_to_decimal binary.reverse
  let octal := decimal_to_octal decimal
  decimal = 220 ∧ octal = [3, 3, 4] := by
  sorry

#eval binary_to_decimal [false, false, true, true, true, false, true, true].reverse
#eval decimal_to_octal 220

end binary_to_octal_conversion_l3183_318374


namespace expansion_distinct_terms_l3183_318376

/-- The number of distinct terms in the expansion of (x+y+z+w)(p+q+r+s+t) -/
def distinctTerms (x y z w p q r s t : ℝ) : ℕ :=
  4 * 5

theorem expansion_distinct_terms (x y z w p q r s t : ℝ) 
  (h : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
       p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t) :
  distinctTerms x y z w p q r s t = 20 := by
  sorry

end expansion_distinct_terms_l3183_318376


namespace total_students_count_l3183_318357

/-- The number of students per team -/
def students_per_team : ℕ := 18

/-- The number of teams -/
def number_of_teams : ℕ := 9

/-- The total number of students -/
def total_students : ℕ := students_per_team * number_of_teams

theorem total_students_count : total_students = 162 := by
  sorry

end total_students_count_l3183_318357


namespace complex_division_l3183_318398

theorem complex_division (i : ℂ) (h : i^2 = -1) : 
  (1 - 2*i) / (1 + i) = -1/2 - 3/2*i := by
  sorry

end complex_division_l3183_318398


namespace percentage_difference_l3183_318323

theorem percentage_difference : 
  (68.5 / 100 * 825) - (34.25 / 100 * 1620) = 10.275 := by
  sorry

end percentage_difference_l3183_318323


namespace books_borrowed_by_lunchtime_correct_books_borrowed_l3183_318348

theorem books_borrowed_by_lunchtime 
  (initial_books : ℕ) 
  (books_added : ℕ) 
  (books_borrowed_evening : ℕ) 
  (books_remaining : ℕ) : ℕ :=
  let books_borrowed_lunchtime := 
    initial_books + books_added - books_borrowed_evening - books_remaining
  books_borrowed_lunchtime

#check @books_borrowed_by_lunchtime

theorem correct_books_borrowed (
  initial_books : ℕ) 
  (books_added : ℕ) 
  (books_borrowed_evening : ℕ) 
  (books_remaining : ℕ) 
  (h1 : initial_books = 100) 
  (h2 : books_added = 40) 
  (h3 : books_borrowed_evening = 30) 
  (h4 : books_remaining = 60) :
  books_borrowed_by_lunchtime initial_books books_added books_borrowed_evening books_remaining = 50 := by
  sorry

end books_borrowed_by_lunchtime_correct_books_borrowed_l3183_318348


namespace intersection_of_A_and_B_l3183_318339

def A : Set Char := {'a', 'b', 'c', 'd', 'e'}
def B : Set Char := {'d', 'f', 'g'}

theorem intersection_of_A_and_B :
  A ∩ B = {'d'} := by
  sorry

end intersection_of_A_and_B_l3183_318339


namespace point_on_segment_coordinates_l3183_318304

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line segment between two other points -/
def lies_on_segment (p q r : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    p.x = q.x + t * (r.x - q.x) ∧
    p.y = q.y + t * (r.y - q.y)

theorem point_on_segment_coordinates :
  let K : Point := ⟨4, 2⟩
  let M : Point := ⟨10, 11⟩
  let L : Point := ⟨6, w⟩
  lies_on_segment L K M → w = 5 := by
sorry

end point_on_segment_coordinates_l3183_318304


namespace x_intercept_of_specific_line_l3183_318335

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ := 
  sorry

/-- The specific line passing through (-1, 1) and (0, 3) -/
def specific_line : Line := { x₁ := -1, y₁ := 1, x₂ := 0, y₂ := 3 }

theorem x_intercept_of_specific_line : 
  x_intercept specific_line = -3/2 := by sorry

end x_intercept_of_specific_line_l3183_318335


namespace domain_of_h_l3183_318321

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-12) 6

-- Define the function h in terms of f
def h (x : ℝ) : ℝ := f (-3 * x)

-- State the theorem about the domain of h
theorem domain_of_h :
  {x : ℝ | h x ∈ Set.range f} = Set.Icc (-2) 4 := by sorry

end domain_of_h_l3183_318321


namespace chess_square_exists_l3183_318307

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 100x100 table of colored cells -/
def Table := Fin 100 → Fin 100 → Color

/-- Checks if a cell is on the border of the table -/
def isBorder (i j : Fin 100) : Prop :=
  i = 0 || i = 99 || j = 0 || j = 99

/-- Checks if a 2x2 square starting at (i,j) contains cells of two colors -/
def hasTwoColors (t : Table) (i j : Fin 100) : Prop :=
  ∃ (c₁ c₂ : Color), c₁ ≠ c₂ ∧
    ((t i j = c₁ ∧ t (i+1) j = c₂) ∨
     (t i j = c₁ ∧ t i (j+1) = c₂) ∨
     (t i j = c₁ ∧ t (i+1) (j+1) = c₂) ∨
     (t (i+1) j = c₁ ∧ t i (j+1) = c₂) ∨
     (t (i+1) j = c₁ ∧ t (i+1) (j+1) = c₂) ∨
     (t i (j+1) = c₁ ∧ t (i+1) (j+1) = c₂))

/-- Checks if a 2x2 square starting at (i,j) is colored in chess order -/
def isChessOrder (t : Table) (i j : Fin 100) : Prop :=
  (t i j = Color.Black ∧ t (i+1) j = Color.White ∧ t i (j+1) = Color.White ∧ t (i+1) (j+1) = Color.Black) ∨
  (t i j = Color.White ∧ t (i+1) j = Color.Black ∧ t i (j+1) = Color.Black ∧ t (i+1) (j+1) = Color.White)

theorem chess_square_exists (t : Table) 
  (border_black : ∀ i j, isBorder i j → t i j = Color.Black)
  (two_colors : ∀ i j, hasTwoColors t i j) :
  ∃ i j, isChessOrder t i j := by
  sorry

end chess_square_exists_l3183_318307


namespace function_inequality_l3183_318300

theorem function_inequality (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : ∀ x, f x = |Real.log x|)
  (h4 : f a > f c) (h5 : f c > f b) : 
  (a - 1) * (c - 1) > 0 := by
  sorry

end function_inequality_l3183_318300


namespace parabola_intersection_l3183_318373

/-- First parabola equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- Second parabola equation -/
def g (x : ℝ) : ℝ := -x^2 + 6 * x + 8

/-- Theorem stating that (-0.5, 4.75) and (3, 17) are the only intersection points -/
theorem parabola_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = -0.5 ∧ y = 4.75) ∨ (x = 3 ∧ y = 17)) := by
  sorry

end parabola_intersection_l3183_318373


namespace min_value_abs_function_l3183_318303

theorem min_value_abs_function (x : ℝ) :
  ∀ x, |x - 1| + |x - 2| - |x - 3| ≥ -1 ∧ ∃ x, |x - 1| + |x - 2| - |x - 3| = -1 :=
by sorry

end min_value_abs_function_l3183_318303


namespace fathers_age_fathers_age_is_52_l3183_318349

theorem fathers_age (sons_age_5_years_ago : ℕ) (years_passed : ℕ) : ℕ :=
  let sons_current_age := sons_age_5_years_ago + years_passed
  2 * sons_current_age

theorem fathers_age_is_52 : fathers_age 21 5 = 52 := by
  sorry

end fathers_age_fathers_age_is_52_l3183_318349


namespace sum_x_y_equals_negative_one_l3183_318388

theorem sum_x_y_equals_negative_one (x y : ℝ) (h : |x - 1| + (y + 2)^2 = 0) : x + y = -1 := by
  sorry

end sum_x_y_equals_negative_one_l3183_318388


namespace juice_cans_bought_l3183_318356

-- Define the original price of ice cream
def original_ice_cream_price : ℚ := 12

-- Define the discount on ice cream
def ice_cream_discount : ℚ := 2

-- Define the price of juice
def juice_price : ℚ := 2

-- Define the number of cans in a set of juice
def cans_per_set : ℕ := 5

-- Define the total cost
def total_cost : ℚ := 24

-- Define the number of ice cream tubs bought
def ice_cream_tubs : ℕ := 2

-- Theorem to prove
theorem juice_cans_bought : ℕ := by
  -- The proof goes here
  sorry

#check juice_cans_bought

end juice_cans_bought_l3183_318356


namespace shopkeeper_profit_l3183_318331

theorem shopkeeper_profit (cost_price : ℝ) : cost_price > 0 →
  let marked_price := cost_price * 1.2
  let selling_price := marked_price * 0.85
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 2 := by
sorry

end shopkeeper_profit_l3183_318331


namespace circle_rolling_in_triangle_l3183_318337

theorem circle_rolling_in_triangle (a b c : ℝ) (r : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) (h4 : r = 2) :
  let k := (a + b + c - 6 * r) / (a + b + c)
  (k * a + k * b + k * c) = 220 / 7 :=
by sorry

end circle_rolling_in_triangle_l3183_318337


namespace eight_div_repeating_third_eq_24_l3183_318312

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1 / 3

/-- The result of 8 divided by 0.333... --/
def result : ℚ := 8 / repeating_third

/-- Theorem stating that 8 divided by 0.333... equals 24 --/
theorem eight_div_repeating_third_eq_24 : result = 24 := by sorry

end eight_div_repeating_third_eq_24_l3183_318312


namespace monotone_f_range_l3183_318392

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a^x else x^2 + 4/x + a * Real.log x

theorem monotone_f_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a ≤ 5 := by sorry

end monotone_f_range_l3183_318392


namespace x_minus_y_value_l3183_318397

theorem x_minus_y_value (x y : ℝ) (h1 : x^2 = 9) (h2 : |y| = 4) (h3 : x < y) :
  x - y = -7 ∨ x - y = -1 := by
sorry

end x_minus_y_value_l3183_318397


namespace class_composition_l3183_318318

/-- The number of girls in the class -/
def num_girls : ℕ := 13

/-- The percentage of girls after adding one boy -/
def girls_percentage : ℚ := 52 / 100

/-- The original number of boys in the class -/
def original_boys : ℕ := 11

theorem class_composition :
  (num_girls : ℚ) / ((original_boys : ℚ) + 1 + num_girls) = girls_percentage := by
  sorry

end class_composition_l3183_318318


namespace rice_weight_per_container_l3183_318315

/-- 
Given a bag of rice weighing sqrt(50) pounds divided equally into 7 containers,
prove that the weight of rice in each container, in ounces, is (80 * sqrt(2)) / 7,
assuming 1 pound = 16 ounces.
-/
theorem rice_weight_per_container 
  (total_weight : ℝ) 
  (num_containers : ℕ) 
  (pounds_to_ounces : ℝ) 
  (h1 : total_weight = Real.sqrt 50)
  (h2 : num_containers = 7)
  (h3 : pounds_to_ounces = 16) :
  (total_weight / num_containers) * pounds_to_ounces = (80 * Real.sqrt 2) / 7 := by
  sorry

end rice_weight_per_container_l3183_318315


namespace tangent_line_perpendicular_and_equation_l3183_318382

/-- The tangent line to y = x^4 at (1, 1) is perpendicular to x + 4y - 8 = 0 and has equation 4x - y - 3 = 0 -/
theorem tangent_line_perpendicular_and_equation (x y : ℝ) : 
  let f : ℝ → ℝ := fun x => x^4
  let tangent_slope : ℝ := (deriv f) 1
  let perpendicular_line_slope : ℝ := -1/4
  let tangent_equation : ℝ → ℝ → Prop := fun x y => 4*x - y - 3 = 0
  tangent_slope * perpendicular_line_slope = -1 ∧
  tangent_equation 1 1 ∧
  (∀ x y, tangent_equation x y ↔ y - 1 = tangent_slope * (x - 1)) := by
sorry

end tangent_line_perpendicular_and_equation_l3183_318382


namespace polynomial_division_remainder_l3183_318340

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 = (X^2 + 3*X + 2) * q + (-15*X - 14) := by sorry

end polynomial_division_remainder_l3183_318340


namespace gcf_of_12_and_16_l3183_318364

theorem gcf_of_12_and_16 (n : ℕ) : 
  n = 12 → Nat.lcm n 16 = 48 → Nat.gcd n 16 = 4 := by
  sorry

end gcf_of_12_and_16_l3183_318364


namespace simon_is_10_years_old_l3183_318391

/-- Simon's age given Alvin's age and their relationship -/
def simon_age (alvin_age : ℕ) (age_difference : ℕ) : ℕ :=
  alvin_age / 2 - age_difference

theorem simon_is_10_years_old :
  simon_age 30 5 = 10 := by
  sorry

end simon_is_10_years_old_l3183_318391
