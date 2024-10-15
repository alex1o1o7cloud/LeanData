import Mathlib

namespace NUMINAMATH_CALUDE_unique_x_for_volume_l1156_115694

/-- A function representing the volume of the rectangular prism -/
def volume (x : ℕ) : ℕ := (x + 3) * (x - 3) * (x^2 + 9)

/-- The theorem stating that there is exactly one positive integer x satisfying the conditions -/
theorem unique_x_for_volume :
  ∃! x : ℕ, x > 3 ∧ volume x < 500 :=
sorry

end NUMINAMATH_CALUDE_unique_x_for_volume_l1156_115694


namespace NUMINAMATH_CALUDE_smallest_three_types_sixty_nine_includes_three_types_l1156_115636

/-- Represents a type of tree in the grove -/
inductive TreeType
  | Birch
  | Spruce
  | Pine
  | Aspen

/-- Represents the grove of trees -/
structure Grove :=
  (trees : Finset ℕ)
  (type : ℕ → TreeType)
  (total_count : trees.card = 100)
  (four_types_in_85 : ∀ s : Finset ℕ, s ⊆ trees → s.card = 85 → 
    (∃ i ∈ s, type i = TreeType.Birch) ∧
    (∃ i ∈ s, type i = TreeType.Spruce) ∧
    (∃ i ∈ s, type i = TreeType.Pine) ∧
    (∃ i ∈ s, type i = TreeType.Aspen))

/-- The main theorem stating the smallest number of trees that must include at least three types -/
theorem smallest_three_types (g : Grove) : 
  ∀ n < 69, ∃ s : Finset ℕ, s ⊆ g.trees ∧ s.card = n ∧ 
    (∃ t1 t2 : TreeType, ∀ i ∈ s, g.type i = t1 ∨ g.type i = t2) :=
by sorry

/-- The theorem stating that 69 trees always include at least three types -/
theorem sixty_nine_includes_three_types (g : Grove) :
  ∀ s : Finset ℕ, s ⊆ g.trees → s.card = 69 → 
    ∃ t1 t2 t3 : TreeType, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧
    (∃ i ∈ s, g.type i = t1) ∧ (∃ i ∈ s, g.type i = t2) ∧ (∃ i ∈ s, g.type i = t3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_types_sixty_nine_includes_three_types_l1156_115636


namespace NUMINAMATH_CALUDE_octal_to_decimal_fraction_l1156_115606

theorem octal_to_decimal_fraction (c d : ℕ) : 
  (5 * 8^2 + 4 * 8 + 7 = 300 + 10 * c + d) → 
  (0 ≤ c) → (c ≤ 9) → (0 ≤ d) → (d ≤ 9) →
  (c * d) / 12 = 5 / 4 := by sorry

end NUMINAMATH_CALUDE_octal_to_decimal_fraction_l1156_115606


namespace NUMINAMATH_CALUDE_race_head_start_l1156_115651

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (30 / 17) * Vb) :
  ∃ H : ℝ, H = (13 / 30) * L ∧ L / Va = (L - H) / Vb :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l1156_115651


namespace NUMINAMATH_CALUDE_number_difference_theorem_l1156_115614

theorem number_difference_theorem (x : ℝ) : x - (3 / 5) * x = 64 → x = 160 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_theorem_l1156_115614


namespace NUMINAMATH_CALUDE_largest_solution_floor_equation_l1156_115683

theorem largest_solution_floor_equation :
  let f (x : ℝ) := ⌊x⌋ = 10 + 50 * (x - ⌊x⌋)
  ∃ (max_sol : ℝ), f max_sol ∧ max_sol = 59.98 ∧ ∀ y, f y → y ≤ max_sol :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_floor_equation_l1156_115683


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_sum_of_squares_l1156_115613

/-- A hyperbola centered at the origin -/
structure Hyperbola where
  a : ℝ
  equation : ∀ (x y : ℝ), x^2 - y^2 = a^2

/-- A circle with center at the origin -/
structure Circle where
  r : ℝ
  equation : ∀ (x y : ℝ), x^2 + y^2 = r^2

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance of a point from the origin -/
def distance_from_origin (p : Point) : ℝ := p.x^2 + p.y^2

theorem hyperbola_circle_intersection_sum_of_squares 
  (h : Hyperbola) (c : Circle) (P Q R S : Point) :
  (P.x^2 - P.y^2 = h.a^2) →
  (Q.x^2 - Q.y^2 = h.a^2) →
  (R.x^2 - R.y^2 = h.a^2) →
  (S.x^2 - S.y^2 = h.a^2) →
  (P.x^2 + P.y^2 = c.r^2) →
  (Q.x^2 + Q.y^2 = c.r^2) →
  (R.x^2 + R.y^2 = c.r^2) →
  (S.x^2 + S.y^2 = c.r^2) →
  distance_from_origin P + distance_from_origin Q + 
  distance_from_origin R + distance_from_origin S = 4 * c.r^2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_sum_of_squares_l1156_115613


namespace NUMINAMATH_CALUDE_circle_line_intersection_sum_l1156_115650

/-- Given a circle with radius 4 centered at the origin and a line y = √3x - 4
    intersecting the circle at points A and B, the sum of the length of segment AB
    and the length of the larger arc AB is (16π/3) + 4√3. -/
theorem circle_line_intersection_sum (A B : ℝ × ℝ) : 
  let r : ℝ := 4
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let line := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 * x - 4}
  A ∈ circle ∧ A ∈ line ∧ B ∈ circle ∧ B ∈ line ∧ A ≠ B →
  let segment_length := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let angle := Real.arccos ((2 * r^2 - segment_length^2) / (2 * r^2))
  let arc_length := (2 * π - angle) * r
  segment_length + arc_length = (16 * π / 3) + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_sum_l1156_115650


namespace NUMINAMATH_CALUDE_total_money_collected_is_960_l1156_115646

/-- Calculates the total money collected from admission receipts for a play. -/
def totalMoneyCollected (totalPeople : Nat) (adultPrice : Nat) (childPrice : Nat) (numAdults : Nat) : Nat :=
  let numChildren := totalPeople - numAdults
  adultPrice * numAdults + childPrice * numChildren

/-- Theorem stating that the total money collected is 960 dollars given the specified conditions. -/
theorem total_money_collected_is_960 :
  totalMoneyCollected 610 2 1 350 = 960 := by
  sorry

end NUMINAMATH_CALUDE_total_money_collected_is_960_l1156_115646


namespace NUMINAMATH_CALUDE_poetry_class_attendance_l1156_115688

/-- The number of people who initially attended the poetry class. -/
def initial_attendees : ℕ := 45

/-- The number of people who arrived late to the class. -/
def late_arrivals : ℕ := 15

/-- The number of lollipops given away by the teacher. -/
def lollipops_given : ℕ := 12

/-- The ratio of attendees to lollipops. -/
def attendee_lollipop_ratio : ℕ := 5

theorem poetry_class_attendance :
  (initial_attendees + late_arrivals) / attendee_lollipop_ratio = lollipops_given :=
by sorry

end NUMINAMATH_CALUDE_poetry_class_attendance_l1156_115688


namespace NUMINAMATH_CALUDE_glass_bottles_in_second_scenario_l1156_115669

/-- The weight of a glass bottle in grams -/
def glass_weight : ℕ := 200

/-- The weight of a plastic bottle in grams -/
def plastic_weight : ℕ := 50

/-- The number of glass bottles in the first scenario -/
def first_scenario_bottles : ℕ := 3

/-- The number of plastic bottles in the second scenario -/
def second_scenario_plastic : ℕ := 5

/-- The total weight in the first scenario in grams -/
def first_scenario_weight : ℕ := 600

/-- The total weight in the second scenario in grams -/
def second_scenario_weight : ℕ := 1050

/-- The weight difference between a glass and plastic bottle in grams -/
def weight_difference : ℕ := 150

theorem glass_bottles_in_second_scenario :
  ∃ x : ℕ, 
    first_scenario_bottles * glass_weight = first_scenario_weight ∧
    glass_weight = plastic_weight + weight_difference ∧
    x * glass_weight + second_scenario_plastic * plastic_weight = second_scenario_weight ∧
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_glass_bottles_in_second_scenario_l1156_115669


namespace NUMINAMATH_CALUDE_solution_set_m_zero_range_of_m_x_in_2_3_l1156_115698

-- Define the inequality
def inequality (x m : ℝ) : Prop := x * abs (x - m) - 2 ≥ m

-- Part 1: Solution set when m = 0
theorem solution_set_m_zero :
  {x : ℝ | inequality x 0} = {x : ℝ | x ≥ Real.sqrt 2} := by sorry

-- Part 2: Range of m when x ∈ [2, 3]
theorem range_of_m_x_in_2_3 :
  {m : ℝ | ∀ x ∈ Set.Icc 2 3, inequality x m} = 
  {m : ℝ | m ≤ 2/3 ∨ m ≥ 6} := by sorry

end NUMINAMATH_CALUDE_solution_set_m_zero_range_of_m_x_in_2_3_l1156_115698


namespace NUMINAMATH_CALUDE_expected_winnings_is_one_l1156_115689

/-- Represents the possible outcomes of the dice roll -/
inductive Outcome
| Star
| Moon
| Sun

/-- The probability of each outcome -/
def probability (o : Outcome) : ℚ :=
  match o with
  | Outcome.Star => 1/4
  | Outcome.Moon => 1/2
  | Outcome.Sun => 1/4

/-- The winnings (or losses) associated with each outcome -/
def winnings (o : Outcome) : ℤ :=
  match o with
  | Outcome.Star => 2
  | Outcome.Moon => 4
  | Outcome.Sun => -6

/-- The expected winnings from rolling the dice once -/
def expected_winnings : ℚ :=
  (probability Outcome.Star * winnings Outcome.Star) +
  (probability Outcome.Moon * winnings Outcome.Moon) +
  (probability Outcome.Sun * winnings Outcome.Sun)

theorem expected_winnings_is_one : expected_winnings = 1 := by
  sorry

end NUMINAMATH_CALUDE_expected_winnings_is_one_l1156_115689


namespace NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l1156_115665

theorem tan_fifteen_pi_fourths : Real.tan (15 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_pi_fourths_l1156_115665


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1156_115619

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 5*x < 10 ↔ -5 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1156_115619


namespace NUMINAMATH_CALUDE_y_minimum_range_l1156_115637

def y (x : ℝ) : ℝ := |x^2 - 1| + |2*x^2 - 1| + |3*x^2 - 1|

theorem y_minimum_range :
  ∀ x : ℝ, y x ≥ 1 ∧
  (y x = 1 ↔ (x ∈ Set.Icc (-Real.sqrt (1/2)) (-Real.sqrt (1/3)) ∪ 
              Set.Icc (Real.sqrt (1/3)) (Real.sqrt (1/2)))) :=
sorry

end NUMINAMATH_CALUDE_y_minimum_range_l1156_115637


namespace NUMINAMATH_CALUDE_inequality_proof_l1156_115620

theorem inequality_proof (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) : b/a + a/b > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1156_115620


namespace NUMINAMATH_CALUDE_linear_function_condition_l1156_115652

/-- A linear function f(x) = ax + b satisfying f⁽¹⁰⁾(x) ≥ 1024x + 1023 
    must have a = 2 and b ≥ 1, or a = -2 and b ≤ -3 -/
theorem linear_function_condition (a b : ℝ) (h : ∀ x, a^10 * x + b * (a^10 - 1) / (a - 1) ≥ 1024 * x + 1023) :
  (a = 2 ∧ b ≥ 1) ∨ (a = -2 ∧ b ≤ -3) := by sorry

end NUMINAMATH_CALUDE_linear_function_condition_l1156_115652


namespace NUMINAMATH_CALUDE_sum_of_stacks_with_green_3_and_4_l1156_115675

/-- Represents a card with a color and a number -/
structure Card where
  color : String
  number : Nat

/-- Represents a stack of cards -/
structure Stack where
  green : Card
  orange : Option Card

/-- Checks if a stack is valid according to the problem rules -/
def isValidStack (s : Stack) : Bool :=
  match s.orange with
  | none => true
  | some o => s.green.number ≤ o.number

/-- The set of all green cards -/
def greenCards : List Card :=
  [1, 2, 3, 4, 5].map (λ n => ⟨"green", n⟩)

/-- The set of all orange cards -/
def orangeCards : List Card :=
  [2, 3, 4, 5].map (λ n => ⟨"orange", n⟩)

/-- Calculates the sum of numbers in a stack -/
def stackSum (s : Stack) : Nat :=
  s.green.number + match s.orange with
  | none => 0
  | some o => o.number

/-- The main theorem to prove -/
theorem sum_of_stacks_with_green_3_and_4 :
  ∃ (s₁ s₂ : Stack),
    s₁.green.number = 3 ∧
    s₂.green.number = 4 ∧
    s₁.green ∈ greenCards ∧
    s₂.green ∈ greenCards ∧
    (∀ o₁ ∈ s₁.orange, o₁ ∈ orangeCards) ∧
    (∀ o₂ ∈ s₂.orange, o₂ ∈ orangeCards) ∧
    isValidStack s₁ ∧
    isValidStack s₂ ∧
    stackSum s₁ + stackSum s₂ = 14 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_stacks_with_green_3_and_4_l1156_115675


namespace NUMINAMATH_CALUDE_parabola_properties_l1156_115692

/-- Parabola passing through given points with specific properties -/
theorem parabola_properties (a b : ℝ) (m : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + 1) →
  (-2 = a * 1^2 + b * 1 + 1) →
  (13 = a * (-2)^2 + b * (-2) + 1) →
  (∃ y₁ y₂, y₁ = a * 5^2 + b * 5 + 1 ∧ 
            y₂ = a * m^2 + b * m + 1 ∧ 
            y₂ = 12 - y₁) →
  (a = 1 ∧ b = -4 ∧ m = -1) := by
sorry

end NUMINAMATH_CALUDE_parabola_properties_l1156_115692


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1156_115678

theorem water_tank_capacity (initial_fraction : ℚ) (final_fraction : ℚ) (added_amount : ℚ) (capacity : ℚ) : 
  initial_fraction = 1/7 →
  final_fraction = 1/5 →
  added_amount = 5 →
  initial_fraction * capacity + added_amount = final_fraction * capacity →
  capacity = 87.5 := by
sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1156_115678


namespace NUMINAMATH_CALUDE_simplify_expression_l1156_115644

theorem simplify_expression : (2^3002 * 3^3004) / 6^3003 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1156_115644


namespace NUMINAMATH_CALUDE_root_sum_cubes_l1156_115673

theorem root_sum_cubes (r s t : ℝ) : 
  (6 * r^3 + 1506 * r + 3009 = 0) →
  (6 * s^3 + 1506 * s + 3009 = 0) →
  (6 * t^3 + 1506 * t + 3009 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 1504.5 := by
sorry

end NUMINAMATH_CALUDE_root_sum_cubes_l1156_115673


namespace NUMINAMATH_CALUDE_both_miss_probability_l1156_115640

/-- The probability that both shooters miss the target given their individual hit probabilities -/
theorem both_miss_probability (p_hit_A p_hit_B : ℝ) (h_A : p_hit_A = 0.85) (h_B : p_hit_B = 0.8) :
  (1 - p_hit_A) * (1 - p_hit_B) = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_both_miss_probability_l1156_115640


namespace NUMINAMATH_CALUDE_train_length_calculation_l1156_115666

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 15 → ∃ (length : ℝ), abs (length - 500) < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1156_115666


namespace NUMINAMATH_CALUDE_apple_problem_l1156_115626

theorem apple_problem (initial_apples : ℕ) (sold_to_jill_percent : ℚ) (sold_to_june_percent : ℚ) (given_to_teacher : ℕ) : 
  initial_apples = 150 →
  sold_to_jill_percent = 30 / 100 →
  sold_to_june_percent = 20 / 100 →
  given_to_teacher = 2 →
  initial_apples - 
    (↑initial_apples * sold_to_jill_percent).floor - 
    ((↑initial_apples - (↑initial_apples * sold_to_jill_percent).floor) * sold_to_june_percent).floor - 
    given_to_teacher = 82 :=
by sorry

end NUMINAMATH_CALUDE_apple_problem_l1156_115626


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1156_115674

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 20 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 + a 7 + a 9 + a 11 = 20

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : 
  a 1 + a 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1156_115674


namespace NUMINAMATH_CALUDE_largest_term_binomial_expansion_largest_term_specific_case_l1156_115696

theorem largest_term_binomial_expansion (n : ℕ) (x : ℝ) (h : x > 0) :
  let A : ℕ → ℝ := λ k => (n.choose k) * x^k
  ∃ k : ℕ, k ≤ n ∧ ∀ j : ℕ, j ≤ n → A k ≥ A j :=
by
  sorry

theorem largest_term_specific_case :
  let n : ℕ := 500
  let x : ℝ := 0.3
  let A : ℕ → ℝ := λ k => (n.choose k) * x^k
  ∃ k : ℕ, k = 125 ∧ ∀ j : ℕ, j ≤ n → A k ≥ A j :=
by
  sorry

end NUMINAMATH_CALUDE_largest_term_binomial_expansion_largest_term_specific_case_l1156_115696


namespace NUMINAMATH_CALUDE_volume_ratio_minimum_l1156_115679

noncomputable section

/-- The volume ratio of a cone to its circumscribed cylinder, given the sine of the cone's half-angle -/
def volume_ratio (s : ℝ) : ℝ := (1 + s)^3 / (6 * s * (1 - s^2))

/-- The theorem stating that the volume ratio is minimized when sin(θ) = 1/3 -/
theorem volume_ratio_minimum :
  ∀ s : ℝ, 0 < s → s < 1 →
  volume_ratio s ≥ 4/3 ∧
  (volume_ratio s = 4/3 ↔ s = 1/3) :=
sorry

end

end NUMINAMATH_CALUDE_volume_ratio_minimum_l1156_115679


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l1156_115622

theorem quadratic_equation_from_means (a b : ℝ) 
  (h_arithmetic_mean : (a + b) / 2 = 10)
  (h_geometric_mean : Real.sqrt (a * b) = 24) :
  ∀ x, x^2 - 20*x + 576 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l1156_115622


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l1156_115609

/-- A rectangular prism with given surface area and total edge length has a specific interior diagonal length -/
theorem rectangular_prism_diagonal (a b c : ℝ) 
  (h_surface_area : 2 * (a * b + a * c + b * c) = 54)
  (h_edge_length : 4 * (a + b + c) = 40) :
  ∃ d : ℝ, d^2 = a^2 + b^2 + c^2 ∧ d = Real.sqrt 46 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l1156_115609


namespace NUMINAMATH_CALUDE_function_derivative_at_midpoint_negative_l1156_115604

open Real

theorem function_derivative_at_midpoint_negative 
  (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) 
  (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hf : ∀ x, f x = log x - a * x + 1) 
  (hz : f x₁ = 0 ∧ f x₂ = 0) : 
  deriv f ((x₁ + x₂) / 2) < 0 := by
  sorry


end NUMINAMATH_CALUDE_function_derivative_at_midpoint_negative_l1156_115604


namespace NUMINAMATH_CALUDE_austin_started_with_80_l1156_115693

/-- The amount of money Austin started with, given the conditions of the problem. -/
def austin_starting_amount : ℚ :=
  let num_robots : ℕ := 7
  let robot_cost : ℚ := 875 / 100
  let total_tax : ℚ := 722 / 100
  let change : ℚ := 1153 / 100
  num_robots * robot_cost + total_tax + change

/-- Theorem stating that Austin started with $80. -/
theorem austin_started_with_80 : austin_starting_amount = 80 := by
  sorry

end NUMINAMATH_CALUDE_austin_started_with_80_l1156_115693


namespace NUMINAMATH_CALUDE_garden_fence_length_l1156_115681

theorem garden_fence_length (side_length : ℝ) (h : side_length = 28) : 
  4 * side_length = 112 := by
  sorry

end NUMINAMATH_CALUDE_garden_fence_length_l1156_115681


namespace NUMINAMATH_CALUDE_experiment_sequences_l1156_115690

def num_procedures : ℕ → ℕ
  | n => 4 * Nat.factorial (n - 3)

theorem experiment_sequences (n : ℕ) (h : n ≥ 3) : num_procedures n = 96 := by
  sorry

end NUMINAMATH_CALUDE_experiment_sequences_l1156_115690


namespace NUMINAMATH_CALUDE_sum_of_squares_l1156_115697

theorem sum_of_squares (a b c : ℝ) 
  (h1 : a * b + b * c + a * c = 131) 
  (h2 : a + b + c = 22) : 
  a^2 + b^2 + c^2 = 222 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1156_115697


namespace NUMINAMATH_CALUDE_intersection_dot_product_l1156_115667

/-- Given a line and a parabola that intersect at points A and B, 
    and the focus of the parabola F, prove that the dot product 
    of vectors FA and FB is -11. -/
theorem intersection_dot_product 
  (A B : ℝ × ℝ) 
  (hA : A.2 = 2 * A.1 - 2 ∧ A.2^2 = 8 * A.1) 
  (hB : B.2 = 2 * B.1 - 2 ∧ B.2^2 = 8 * B.1) 
  (hAB_distinct : A ≠ B) : 
  let F : ℝ × ℝ := (2, 0)
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = -11 := by
  sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l1156_115667


namespace NUMINAMATH_CALUDE_rectangle_area_sum_l1156_115695

theorem rectangle_area_sum : 
  let rect1 := 7 * 8
  let rect2 := 5 * 3
  let rect3 := 2 * 8
  let rect4 := 2 * 7
  let rect5 := 4 * 4
  rect1 + rect2 + rect3 + rect4 + rect5 = 117 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_sum_l1156_115695


namespace NUMINAMATH_CALUDE_f_monotonic_decreasing_on_interval_l1156_115654

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- State the theorem
theorem f_monotonic_decreasing_on_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x > f y :=
sorry

end NUMINAMATH_CALUDE_f_monotonic_decreasing_on_interval_l1156_115654


namespace NUMINAMATH_CALUDE_triangle_side_length_l1156_115632

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area of triangle is √3
  (B = Real.pi / 3) →  -- B = 60°
  (a^2 + c^2 = 3*a*c) →  -- Given condition
  (b = 2 * Real.sqrt 2) :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1156_115632


namespace NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l1156_115624

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

theorem thirty_factorial_trailing_zeros :
  trailing_zeros 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l1156_115624


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l1156_115616

theorem normal_distribution_std_dev (μ σ : ℝ) : 
  μ = 55 → μ - 3 * σ > 48 → σ < 7/3 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l1156_115616


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l1156_115661

def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

theorem alex_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 60) : 
  min_additional_coins num_friends initial_coins = 60 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l1156_115661


namespace NUMINAMATH_CALUDE_geometric_transformations_l1156_115635

-- Define the basic geometric entities
structure Point

structure Line

structure Surface

structure Body

-- Define the movement operation
def moves (a : Type) (b : Type) : Prop :=
  ∃ (x : a), ∃ (y : b), true

-- Theorem statement
theorem geometric_transformations :
  (moves Point Line) ∧
  (moves Line Surface) ∧
  (moves Surface Body) := by
  sorry

end NUMINAMATH_CALUDE_geometric_transformations_l1156_115635


namespace NUMINAMATH_CALUDE_inequality_solution_l1156_115623

theorem inequality_solution (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  {x : ℝ | (x - a) * (x - 1/a) < 0} = {x : ℝ | a < x ∧ x < 1/a} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1156_115623


namespace NUMINAMATH_CALUDE_pea_patch_fraction_l1156_115680

theorem pea_patch_fraction (radish_patch : ℝ) (pea_patch : ℝ) (fraction : ℝ) : 
  radish_patch = 15 →
  pea_patch = 2 * radish_patch →
  fraction * pea_patch = 5 →
  fraction = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_pea_patch_fraction_l1156_115680


namespace NUMINAMATH_CALUDE_evaluate_expression_l1156_115691

theorem evaluate_expression : (-2 : ℤ) ^ (3 ^ 2) + 2 ^ (3 ^ 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1156_115691


namespace NUMINAMATH_CALUDE_x_value_l1156_115664

theorem x_value (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1156_115664


namespace NUMINAMATH_CALUDE_sterilization_tank_solution_l1156_115615

/-- Represents the sterilization tank problem --/
def sterilization_tank_problem (initial_volume : ℝ) (drained_volume : ℝ) (final_concentration : ℝ) (initial_concentration : ℝ) : Prop :=
  let remaining_volume := initial_volume - drained_volume
  remaining_volume * initial_concentration + drained_volume = initial_volume * final_concentration

/-- Theorem stating the solution to the sterilization tank problem --/
theorem sterilization_tank_solution :
  sterilization_tank_problem 100 3.0612244898 0.05 0.02 := by
  sorry

end NUMINAMATH_CALUDE_sterilization_tank_solution_l1156_115615


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l1156_115649

/-- Given a cube with face perimeter of 40 cm, its volume is 1000 cubic centimeters. -/
theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (volume : ℝ) :
  face_perimeter = 40 → volume = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l1156_115649


namespace NUMINAMATH_CALUDE_area_of_triangle_NOI_l1156_115648

/-- Triangle PQR with given side lengths -/
structure TrianglePQR where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  side_lengths : PQ = 15 ∧ PR = 8 ∧ QR = 17

/-- Point O is the circumcenter of triangle PQR -/
def is_circumcenter (O : ℝ × ℝ) (t : TrianglePQR) : Prop :=
  sorry

/-- Point I is the incenter of triangle PQR -/
def is_incenter (I : ℝ × ℝ) (t : TrianglePQR) : Prop :=
  sorry

/-- Point N is the center of a circle tangent to sides PQ, PR, and the circumcircle -/
def is_tangent_circle_center (N : ℝ × ℝ) (t : TrianglePQR) (O : ℝ × ℝ) : Prop :=
  sorry

/-- Calculate the area of a triangle given three points -/
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  sorry

/-- Main theorem: The area of triangle NOI is 5 -/
theorem area_of_triangle_NOI (t : TrianglePQR) (O I N : ℝ × ℝ) 
  (hO : is_circumcenter O t) 
  (hI : is_incenter I t)
  (hN : is_tangent_circle_center N t O) : 
  triangle_area N O I = 5 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_NOI_l1156_115648


namespace NUMINAMATH_CALUDE_arccos_cos_three_l1156_115610

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_three_l1156_115610


namespace NUMINAMATH_CALUDE_problem_solution_l1156_115641

theorem problem_solution (a b : ℝ) 
  (h1 : 5 + a = 6 - b) 
  (h2 : 6 + b = 9 + a) : 
  5 - a = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1156_115641


namespace NUMINAMATH_CALUDE_sock_pair_count_l1156_115601

/-- The number of ways to choose a pair of socks with different colors -/
def different_color_pairs (white brown blue red : ℕ) : ℕ :=
  white * brown + white * blue + white * red +
  brown * blue + brown * red +
  blue * red

/-- Theorem: Given 5 white socks, 5 brown socks, 2 blue socks, and 1 red sock,
    the number of ways to choose a pair of socks with different colors is 57 -/
theorem sock_pair_count :
  different_color_pairs 5 5 2 1 = 57 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l1156_115601


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l1156_115645

theorem divisible_by_eleven (n : ℕ) : n < 10 → (123 * 100000 + n * 1000 + 789) % 11 = 0 ↔ n = 10 % 11 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l1156_115645


namespace NUMINAMATH_CALUDE_pigsy_fruits_l1156_115634

def process (n : ℕ) : ℕ := 
  (n / 2 + 2) / 2

theorem pigsy_fruits : ∃ x : ℕ, process (process (process (process x))) = 5 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_pigsy_fruits_l1156_115634


namespace NUMINAMATH_CALUDE_prob_head_fair_coin_l1156_115685

/-- A fair coin with two sides. -/
structure FairCoin where
  sides : Fin 2
  prob_head : ℝ
  prob_tail : ℝ
  sum_to_one : prob_head + prob_tail = 1
  equal_prob : prob_head = prob_tail

/-- The probability of getting a head in a fair coin toss is 1/2. -/
theorem prob_head_fair_coin (c : FairCoin) : c.prob_head = 1/2 := by
  sorry

#check prob_head_fair_coin

end NUMINAMATH_CALUDE_prob_head_fair_coin_l1156_115685


namespace NUMINAMATH_CALUDE_divisibility_property_l1156_115686

theorem divisibility_property (a b : ℕ+) : ∃ n : ℕ+, (a : ℕ) ∣ (b : ℕ)^(n : ℕ) - (n : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1156_115686


namespace NUMINAMATH_CALUDE_odd_function_value_l1156_115605

-- Define an odd function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem odd_function_value :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  (∀ x < 0, f x = 2^x) →      -- f(x) = 2^x for x < 0
  f (Real.log 9 / Real.log 4) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_odd_function_value_l1156_115605


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_l1156_115670

-- Define the triangle ABC
structure Triangle where
  a : ℝ  -- side length opposite to angle A
  b : ℝ  -- side length opposite to angle B
  c : ℝ  -- side length opposite to angle C
  A : ℝ  -- angle A in radians
  B : ℝ  -- angle B in radians
  C : ℝ  -- angle C in radians

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ Real.cos t.B = 4/5

-- Theorem 1
theorem theorem_1 (t : Triangle) (h : triangle_conditions t) (h_A : t.A = Real.pi/6) :
  t.a = 5/3 := by sorry

-- Theorem 2
theorem theorem_2 (t : Triangle) (h : triangle_conditions t) 
  (h_area : (1/2) * t.a * t.c * Real.sin t.B = 3) :
  t.a = Real.sqrt 10 ∧ t.c = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_l1156_115670


namespace NUMINAMATH_CALUDE_joker_king_probability_l1156_115642

/-- A deck of cards with jokers -/
structure Deck :=
  (total_cards : ℕ)
  (num_jokers : ℕ)
  (num_kings : ℕ)

/-- The probability of drawing a joker first and a king second -/
def joker_king_prob (d : Deck) : ℚ :=
  (d.num_jokers : ℚ) / d.total_cards * (d.num_kings : ℚ) / (d.total_cards - 1)

/-- The modified 54-card deck -/
def modified_deck : Deck :=
  { total_cards := 54,
    num_jokers := 2,
    num_kings := 4 }

theorem joker_king_probability :
  joker_king_prob modified_deck = 8 / 1431 := by
  sorry

end NUMINAMATH_CALUDE_joker_king_probability_l1156_115642


namespace NUMINAMATH_CALUDE_complex_equation_roots_l1156_115655

theorem complex_equation_roots : 
  let z₁ : ℂ := 3.5 - I
  let z₂ : ℂ := -2.5 + I
  (z₁^2 - z₁ = 6 - 6*I) ∧ (z₂^2 - z₂ = 6 - 6*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l1156_115655


namespace NUMINAMATH_CALUDE_max_value_in_region_D_l1156_115627

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the asymptotes
def asymptote1 (x y : ℝ) : Prop := y = x
def asymptote2 (x y : ℝ) : Prop := y = -x

-- Define the bounding line
def boundingLine (x : ℝ) : Prop := x = 3

-- Define the region D
def regionD (x y : ℝ) : Prop :=
  x ≤ 3 ∧ y ≤ x ∧ y ≥ -x

-- Define the objective function
def objectiveFunction (x y : ℝ) : ℝ := x + 4*y

-- Theorem statement
theorem max_value_in_region_D :
  ∃ (x y : ℝ), regionD x y ∧
  ∀ (x' y' : ℝ), regionD x' y' →
  objectiveFunction x y ≥ objectiveFunction x' y' ∧
  objectiveFunction x y = 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_in_region_D_l1156_115627


namespace NUMINAMATH_CALUDE_unique_divisor_with_remainder_sum_l1156_115671

theorem unique_divisor_with_remainder_sum (a b c : ℕ) : ∃! n : ℕ,
  n > 3 ∧
  ∃ x y z r s t : ℕ,
    63 = n * x + r ∧
    91 = n * y + s ∧
    130 = n * z + t ∧
    r + s + t = 26 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_with_remainder_sum_l1156_115671


namespace NUMINAMATH_CALUDE_sum_of_ace_l1156_115672

/-- Given 5 children with player numbers, prove that the sum of numbers for A, C, and E is 24 -/
theorem sum_of_ace (a b c d e : ℕ) : 
  a + b + c + d + e = 35 →
  b + c = 13 →
  a + b + c + e = 31 →
  b + c + e = 21 →
  b = 7 →
  a + c + e = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ace_l1156_115672


namespace NUMINAMATH_CALUDE_pepperjack_cheese_probability_l1156_115617

theorem pepperjack_cheese_probability :
  let cheddar : ℕ := 15
  let mozzarella : ℕ := 30
  let pepperjack : ℕ := 45
  let total : ℕ := cheddar + mozzarella + pepperjack
  (pepperjack : ℚ) / (total : ℚ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pepperjack_cheese_probability_l1156_115617


namespace NUMINAMATH_CALUDE_expression_evaluation_l1156_115633

theorem expression_evaluation : -30 + 5 * (9 / (3 + 3)) = -22.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1156_115633


namespace NUMINAMATH_CALUDE_quadratic_radicals_combination_l1156_115663

theorem quadratic_radicals_combination (a : ℝ) : 
  (∃ k : ℝ, k * (1 + a) = 4 - 2*a ∧ k > 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radicals_combination_l1156_115663


namespace NUMINAMATH_CALUDE_reading_reward_pie_chart_l1156_115657

theorem reading_reward_pie_chart (agree disagree neutral : ℕ) 
  (h_ratio : (agree : ℚ) / (disagree : ℚ) = 7 / 2 ∧ (agree : ℚ) / (neutral : ℚ) = 7 / 1) :
  (360 : ℚ) * (agree : ℚ) / ((agree : ℚ) + (disagree : ℚ) + (neutral : ℚ)) = 252 := by
  sorry

end NUMINAMATH_CALUDE_reading_reward_pie_chart_l1156_115657


namespace NUMINAMATH_CALUDE_deepak_age_l1156_115607

/-- Proves that Deepak's present age is 21 years given the conditions -/
theorem deepak_age (rahul_future_age : ℕ) (years_difference : ℕ) (ratio_rahul : ℕ) (ratio_deepak : ℕ) :
  rahul_future_age = 34 →
  years_difference = 6 →
  ratio_rahul = 4 →
  ratio_deepak = 3 →
  (rahul_future_age - years_difference) * ratio_deepak = 21 * ratio_rahul :=
by
  sorry

#check deepak_age

end NUMINAMATH_CALUDE_deepak_age_l1156_115607


namespace NUMINAMATH_CALUDE_office_paper_cost_l1156_115618

/-- Represents a type of bond paper -/
structure BondPaper where
  sheets_per_ream : ℕ
  cost_per_ream : ℕ

/-- Calculates the number of reams needed, rounding up -/
def reams_needed (sheets_required : ℕ) (paper : BondPaper) : ℕ :=
  (sheets_required + paper.sheets_per_ream - 1) / paper.sheets_per_ream

/-- Calculates the cost for a given number of reams -/
def cost_for_reams (reams : ℕ) (paper : BondPaper) : ℕ :=
  reams * paper.cost_per_ream

theorem office_paper_cost :
  let type_a : BondPaper := ⟨500, 27⟩
  let type_b : BondPaper := ⟨400, 24⟩
  let type_c : BondPaper := ⟨300, 18⟩
  let total_sheets : ℕ := 5000
  let min_a_sheets : ℕ := 2500
  let min_b_sheets : ℕ := 1500
  let remaining_sheets : ℕ := total_sheets - min_a_sheets - min_b_sheets
  let reams_a : ℕ := reams_needed min_a_sheets type_a
  let reams_b : ℕ := reams_needed min_b_sheets type_b
  let reams_c : ℕ := reams_needed remaining_sheets type_c
  let total_cost : ℕ := cost_for_reams reams_a type_a +
                        cost_for_reams reams_b type_b +
                        cost_for_reams reams_c type_c
  total_cost = 303 := by
  sorry

end NUMINAMATH_CALUDE_office_paper_cost_l1156_115618


namespace NUMINAMATH_CALUDE_present_age_of_B_prove_present_age_of_B_l1156_115602

theorem present_age_of_B : ℕ → ℕ → Prop :=
  fun a b =>
    (a + 30 = 2 * (b - 30)) →  -- In 30 years, A will be twice as old as B was 30 years ago
    (a = b + 5) →              -- A is now 5 years older than B
    (b = 95)                   -- B's current age is 95 years

-- The proof of the theorem
theorem prove_present_age_of_B : ∃ a b : ℕ, present_age_of_B a b :=
  sorry

end NUMINAMATH_CALUDE_present_age_of_B_prove_present_age_of_B_l1156_115602


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1156_115608

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a, prove that if a₃ + a₅ = 12 - a₇, then a₁ + a₉ = 8 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (h1 : a 3 + a 5 = 12 - a 7) : a 1 + a 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1156_115608


namespace NUMINAMATH_CALUDE_exactly_three_solutions_l1156_115629

-- Define the greatest integer function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (floor x : ℝ) / x else 0

-- State the theorem
theorem exactly_three_solutions (a : ℝ) (h : 3/4 < a ∧ a ≤ 4/5) :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, x > 0 ∧ f x = a :=
sorry

end NUMINAMATH_CALUDE_exactly_three_solutions_l1156_115629


namespace NUMINAMATH_CALUDE_range_of_a_l1156_115687

def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem range_of_a (a : ℝ) (h : A a ∪ B a = Set.univ) : a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1156_115687


namespace NUMINAMATH_CALUDE_product_prs_l1156_115682

theorem product_prs (p r s : ℕ) : 
  4^p + 4^3 = 272 → 
  3^r + 27 = 81 → 
  2^s + 7^2 = 1024 → 
  p * r * s = 160 := by
sorry

end NUMINAMATH_CALUDE_product_prs_l1156_115682


namespace NUMINAMATH_CALUDE_fence_price_per_foot_l1156_115638

/-- Given a square plot with area and total fencing cost, calculate the price per foot of fencing --/
theorem fence_price_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 289) 
  (h2 : total_cost = 3740) : 
  total_cost / (4 * Real.sqrt area) = 55 := by
  sorry

end NUMINAMATH_CALUDE_fence_price_per_foot_l1156_115638


namespace NUMINAMATH_CALUDE_xy_value_l1156_115677

theorem xy_value (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) : x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1156_115677


namespace NUMINAMATH_CALUDE_share_distribution_theorem_l1156_115662

/-- Represents the share distribution problem among three children -/
def ShareDistribution (anusha_share babu_share esha_share k : ℚ) : Prop :=
  -- Total amount is 378
  anusha_share + babu_share + esha_share = 378 ∧
  -- Anusha's share is 84
  anusha_share = 84 ∧
  -- 12 times Anusha's share equals k times Babu's share
  12 * anusha_share = k * babu_share ∧
  -- k times Babu's share equals 6 times Esha's share
  k * babu_share = 6 * esha_share

/-- The main theorem stating that given the conditions, k equals 4 -/
theorem share_distribution_theorem :
  ∀ (anusha_share babu_share esha_share k : ℚ),
  ShareDistribution anusha_share babu_share esha_share k →
  k = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_share_distribution_theorem_l1156_115662


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l1156_115631

/-- Given a geometric sequence {a_n} where a_5 = 7 and a_8 = 56, 
    prove that the general formula is a_n = (7/32) * 2^n -/
theorem geometric_sequence_formula (a : ℕ → ℝ) 
  (h1 : a 5 = 7) 
  (h2 : a 8 = 56) 
  (h_geom : ∀ n m : ℕ, a (n + m) = a n * (a (n + 1) / a n) ^ m) :
  ∃ q : ℝ, ∀ n : ℕ, a n = (7 / 32) * 2^n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l1156_115631


namespace NUMINAMATH_CALUDE_circular_sum_equivalence_l1156_115611

/-- 
Given integers n > m > 1 arranged in a circle, s_i is the sum of m integers 
starting at the i-th position moving clockwise, and t_i is the sum of the 
remaining n-m integers. f(a, b) is the number of elements i in {1, 2, ..., n} 
such that s_i ≡ a (mod 4) and t_i ≡ b (mod 4).
-/
def f (n m : ℕ) (a b : ℕ) : ℕ := sorry

/-- The main theorem to be proved -/
theorem circular_sum_equivalence (n m : ℕ) (h1 : n > m) (h2 : m > 1) :
  f n m 1 3 ≡ f n m 3 1 [MOD 4] ↔ Even (f n m 2 2) := by sorry

end NUMINAMATH_CALUDE_circular_sum_equivalence_l1156_115611


namespace NUMINAMATH_CALUDE_exists_valid_painting_33_exists_valid_painting_32_l1156_115628

/-- Represents a cell on the board -/
structure Cell :=
  (x : Fin 7)
  (y : Fin 7)

/-- Checks if two cells are adjacent -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.x = c2.x ∧ c1.y.val + 1 = c2.y.val) ∨
  (c1.x = c2.x ∧ c1.y.val = c2.y.val + 1) ∨
  (c1.x.val + 1 = c2.x.val ∧ c1.y = c2.y) ∨
  (c1.x.val = c2.x.val + 1 ∧ c1.y = c2.y)

/-- A valid painting sequence -/
def ValidPainting (seq : List Cell) : Prop :=
  seq.length > 0 ∧
  ∀ i j, 0 < i ∧ i < seq.length → 0 ≤ j ∧ j < i - 1 →
    (adjacent (seq.get ⟨i, sorry⟩) (seq.get ⟨i-1, sorry⟩) ∧
     ¬adjacent (seq.get ⟨i, sorry⟩) (seq.get ⟨j, sorry⟩))

/-- Main theorem: There exists a valid painting of 33 cells -/
theorem exists_valid_painting_33 :
  ∃ (seq : List Cell), seq.length = 33 ∧ ValidPainting seq :=
sorry

/-- Corollary: There exists a valid painting of 32 cells -/
theorem exists_valid_painting_32 :
  ∃ (seq : List Cell), seq.length = 32 ∧ ValidPainting seq :=
sorry

end NUMINAMATH_CALUDE_exists_valid_painting_33_exists_valid_painting_32_l1156_115628


namespace NUMINAMATH_CALUDE_polynomial_factor_value_theorem_l1156_115684

theorem polynomial_factor_value_theorem (h k : ℝ) : 
  (∀ x : ℝ, (x + 2) * (x - 1) * (x + 3) ∣ (3 * x^4 - 2 * h * x^2 + h * x + k)) →
  |3 * h - 2 * k| = 11 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_value_theorem_l1156_115684


namespace NUMINAMATH_CALUDE_tables_in_hall_l1156_115653

theorem tables_in_hall : ℕ :=
  let total_legs : ℕ := 724
  let stools_per_table : ℕ := 8
  let stool_legs : ℕ := 4
  let table_legs : ℕ := 5

  have h : ∃ (t : ℕ), t * (stools_per_table * stool_legs + table_legs) = total_legs :=
    sorry

  have unique : ∀ (t : ℕ), t * (stools_per_table * stool_legs + table_legs) = total_legs → t = 19 :=
    sorry

  19

/- Proof omitted -/

end NUMINAMATH_CALUDE_tables_in_hall_l1156_115653


namespace NUMINAMATH_CALUDE_baking_powder_difference_l1156_115600

def baking_powder_yesterday : ℝ := 0.4
def baking_powder_today : ℝ := 0.3

theorem baking_powder_difference :
  baking_powder_yesterday - baking_powder_today = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_baking_powder_difference_l1156_115600


namespace NUMINAMATH_CALUDE_problem_statement_l1156_115699

theorem problem_statement (x y : ℝ) (h1 : x - y > -x) (h2 : x + y > y) : x > 0 ∧ y < 2*x := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1156_115699


namespace NUMINAMATH_CALUDE_fraction_of_men_left_l1156_115621

/-- Represents the movie screening scenario -/
structure MovieScreening where
  total_guests : ℕ
  women : ℕ
  men : ℕ
  children : ℕ
  children_left : ℕ
  people_stayed : ℕ

/-- The specific movie screening instance from the problem -/
def problem_screening : MovieScreening :=
  { total_guests := 50
  , women := 25
  , men := 15
  , children := 10
  , children_left := 4
  , people_stayed := 43
  }

/-- Theorem stating that the fraction of men who left is 1/5 -/
theorem fraction_of_men_left (s : MovieScreening) 
  (h1 : s.total_guests = 50)
  (h2 : s.women = s.total_guests / 2)
  (h3 : s.men = 15)
  (h4 : s.children = s.total_guests - s.women - s.men)
  (h5 : s.children_left = 4)
  (h6 : s.people_stayed = 43) :
  (s.total_guests - s.people_stayed - s.children_left) / s.men = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_men_left_l1156_115621


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_equality_l1156_115625

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (1 / x + 1 / y) ≥ 3 / 2 + Real.sqrt 2 :=
by sorry

theorem min_reciprocal_sum_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (1 / x + 1 / y = 3 / 2 + Real.sqrt 2) ↔ (x = 2 / 3 ∧ y = 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_equality_l1156_115625


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l1156_115639

theorem largest_integer_inequality : ∀ y : ℤ, y ≤ 3 ↔ (y : ℚ) / 4 + 6 / 7 < 7 / 4 := by sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l1156_115639


namespace NUMINAMATH_CALUDE_nell_card_count_l1156_115647

/-- The number of cards Nell has after receiving cards from Jeff -/
def total_cards (initial : Float) (received : Float) : Float :=
  initial + received

/-- Theorem stating that Nell's total cards is the sum of her initial cards and received cards -/
theorem nell_card_count (initial : Float) (received : Float) :
  total_cards initial received = initial + received := by sorry

end NUMINAMATH_CALUDE_nell_card_count_l1156_115647


namespace NUMINAMATH_CALUDE_range_of_a_l1156_115668

-- Define the polynomials p and q
def p (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1
def q (x a : ℝ) : ℝ := x^2 - (2 * a + 1) * x + a^2 + a

-- Define the condition for p
def p_condition (x : ℝ) : Prop := p x ≤ 0

-- Define the condition for q
def q_condition (x a : ℝ) : Prop := q x a ≤ 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p_condition x → q_condition x a) ∧
  (∃ x, q_condition x a ∧ ¬p_condition x)

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1156_115668


namespace NUMINAMATH_CALUDE_distance_sum_bounded_l1156_115676

/-- The sum of squared distances from a point on an ellipse to four fixed points is bounded -/
theorem distance_sum_bounded (x y : ℝ) :
  (x / 2)^2 + (y / 3)^2 = 1 →
  32 ≤ (x - 1)^2 + (y - Real.sqrt 3)^2 +
       (x + Real.sqrt 3)^2 + (y - 1)^2 +
       (x + 1)^2 + (y + Real.sqrt 3)^2 +
       (x - Real.sqrt 3)^2 + (y + 1)^2 ∧
  (x - 1)^2 + (y - Real.sqrt 3)^2 +
  (x + Real.sqrt 3)^2 + (y - 1)^2 +
  (x + 1)^2 + (y + Real.sqrt 3)^2 +
  (x - Real.sqrt 3)^2 + (y + 1)^2 ≤ 52 := by
  sorry


end NUMINAMATH_CALUDE_distance_sum_bounded_l1156_115676


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1156_115643

def A : Set ℝ := {x | x^2 - 1 ≥ 0}
def B : Set ℝ := {x | 0 < x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1156_115643


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1156_115603

/-- Represents a geometric sequence. -/
structure GeometricSequence where
  a : ℕ → ℝ
  r : ℝ
  h1 : ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence {a_n} with a_1 = 2 and a_3a_5 = 4a_6^2, prove that a_3 = 1. -/
theorem geometric_sequence_problem (seq : GeometricSequence)
  (h2 : seq.a 1 = 2)
  (h3 : seq.a 3 * seq.a 5 = 4 * (seq.a 6)^2) :
  seq.a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1156_115603


namespace NUMINAMATH_CALUDE_student_council_distribution_l1156_115658

/-- The number of ways to distribute n indistinguishable items among k distinguishable bins,
    with each bin containing at least 1 item. -/
def distribute_with_minimum (n k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- Theorem stating that there are 252 ways to distribute 11 positions among 6 classes
    with at least 1 position per class. -/
theorem student_council_distribution : distribute_with_minimum 11 6 = 252 := by
  sorry

end NUMINAMATH_CALUDE_student_council_distribution_l1156_115658


namespace NUMINAMATH_CALUDE_sequence_minimum_l1156_115630

theorem sequence_minimum (n : ℤ) : ∃ (m : ℤ), ∀ (n : ℤ), n^2 - 8*n + 5 ≥ m ∧ ∃ (k : ℤ), k^2 - 8*k + 5 = m :=
sorry

end NUMINAMATH_CALUDE_sequence_minimum_l1156_115630


namespace NUMINAMATH_CALUDE_greatest_y_value_l1156_115660

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -9) : 
  ∀ z : ℤ, (∃ w : ℤ, w * z + 3 * w + 2 * z = -9) → z ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_greatest_y_value_l1156_115660


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1156_115612

theorem sqrt_product_equality : 
  2 * Real.sqrt 3 * (1.5 ^ (1/3)) * (12 ^ (1/6)) = 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1156_115612


namespace NUMINAMATH_CALUDE_three_power_greater_than_n_plus_two_times_two_power_l1156_115656

theorem three_power_greater_than_n_plus_two_times_two_power (n : ℕ) (h : n > 2) :
  3^n > (n + 2) * 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_three_power_greater_than_n_plus_two_times_two_power_l1156_115656


namespace NUMINAMATH_CALUDE_edward_additional_spending_l1156_115659

def edward_spending (initial_amount spent_first final_amount : ℕ) : ℕ :=
  initial_amount - spent_first - final_amount

theorem edward_additional_spending :
  edward_spending 34 9 17 = 8 := by
  sorry

end NUMINAMATH_CALUDE_edward_additional_spending_l1156_115659
