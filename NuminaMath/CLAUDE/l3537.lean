import Mathlib

namespace disk_rotation_on_clock_face_l3537_353707

theorem disk_rotation_on_clock_face (clock_radius disk_radius : ℝ) 
  (h1 : clock_radius = 30)
  (h2 : disk_radius = 15)
  (h3 : disk_radius = clock_radius / 2) :
  let initial_position := 0 -- 12 o'clock
  let final_position := π -- 6 o'clock (π radians)
  ∃ (θ : ℝ), 
    θ * disk_radius = final_position * clock_radius ∧ 
    θ % (2 * π) = 0 := by
  sorry

end disk_rotation_on_clock_face_l3537_353707


namespace map_scale_conversion_l3537_353799

/-- Given a map scale where 8 cm represents 40 km, prove that 20 cm represents 100 km -/
theorem map_scale_conversion (map_scale : ℝ → ℝ) 
  (h1 : map_scale 8 = 40) -- 8 cm represents 40 km
  (h2 : ∀ x y : ℝ, map_scale (x + y) = map_scale x + map_scale y) -- Linear scaling
  (h3 : ∀ x : ℝ, map_scale x ≥ 0) -- Non-negative scaling
  : map_scale 20 = 100 := by sorry

end map_scale_conversion_l3537_353799


namespace least_subtraction_for_divisibility_problem_solution_l3537_353714

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x ≤ d - 1 ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 8 ∧ (42398 - x) % 15 = 0 ∧ ∀ (y : ℕ), y < x → (42398 - y) % 15 ≠ 0 :=
sorry

end least_subtraction_for_divisibility_problem_solution_l3537_353714


namespace average_of_solutions_is_zero_l3537_353751

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 28}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ (x : ℝ), x ∈ solutions → x = x₁ ∨ x = x₂ :=
by sorry

end average_of_solutions_is_zero_l3537_353751


namespace dilation_and_shift_result_l3537_353705

/-- Represents a complex number -/
structure ComplexNumber where
  re : ℝ
  im : ℝ

/-- Applies a dilation to a complex number -/
def dilate (center : ComplexNumber) (scale : ℝ) (z : ComplexNumber) : ComplexNumber :=
  { re := center.re + scale * (z.re - center.re),
    im := center.im + scale * (z.im - center.im) }

/-- Shifts a complex number by another complex number -/
def shift (z : ComplexNumber) (s : ComplexNumber) : ComplexNumber :=
  { re := z.re - s.re,
    im := z.im - s.im }

/-- The main theorem to be proved -/
theorem dilation_and_shift_result :
  let initial := ComplexNumber.mk 1 (-2)
  let center := ComplexNumber.mk 1 2
  let scale := 2
  let shiftAmount := ComplexNumber.mk 3 4
  let dilated := dilate center scale initial
  let final := shift dilated shiftAmount
  final = ComplexNumber.mk (-2) (-10) := by
  sorry

end dilation_and_shift_result_l3537_353705


namespace alice_has_winning_strategy_l3537_353748

/-- A game played on a complete graph -/
structure Graph :=
  (n : ℕ)  -- number of vertices
  (is_complete : n > 0)

/-- A player in the game -/
inductive Player
| Alice
| Bob

/-- A move in the game -/
structure Move :=
  (player : Player)
  (edges_oriented : ℕ)

/-- The game state -/
structure GameState :=
  (graph : Graph)
  (moves : List Move)
  (remaining_edges : ℕ)

/-- Alice's strategy -/
def alice_strategy (state : GameState) : Move :=
  { player := Player.Alice, edges_oriented := 1 }

/-- Bob's strategy -/
def bob_strategy (state : GameState) (m : ℕ) : Move :=
  { player := Player.Bob, edges_oriented := m }

/-- The winning condition for Alice -/
def alice_wins (final_state : GameState) : Prop :=
  ∃ (cycle : List ℕ), cycle.length > 0 ∧ cycle.Nodup

/-- The main theorem -/
theorem alice_has_winning_strategy :
  ∀ (g : Graph),
    g.n = 2014 →
    ∀ (bob_moves : GameState → ℕ),
      (∀ (state : GameState), 1 ≤ bob_moves state ∧ bob_moves state ≤ 1000) →
      ∃ (final_state : GameState),
        final_state.graph = g ∧
        final_state.remaining_edges = 0 ∧
        alice_wins final_state :=
  sorry

end alice_has_winning_strategy_l3537_353748


namespace kolya_optimal_strategy_l3537_353765

/-- Represents the three methods Kolya can choose from -/
inductive Method
  | largest_smallest
  | two_middle
  | choice_with_payment

/-- Represents a division of nuts -/
structure NutDivision where
  a₁ : ℕ
  a₂ : ℕ
  b₁ : ℕ
  b₂ : ℕ

/-- Calculates the number of nuts Kolya gets for a given method and division -/
def nuts_for_kolya (m : Method) (d : NutDivision) : ℕ :=
  match m with
  | Method.largest_smallest => max d.a₁ d.b₁ + min d.a₂ d.b₂
  | Method.two_middle => d.a₁ + d.a₂ + d.b₁ + d.b₂ - (max d.a₁ (max d.a₂ (max d.b₁ d.b₂))) - (min d.a₁ (min d.a₂ (min d.b₁ d.b₂)))
  | Method.choice_with_payment => max (max d.a₁ d.b₁ + min d.a₂ d.b₂) (d.a₁ + d.a₂ + d.b₁ + d.b₂ - (max d.a₁ (max d.a₂ (max d.b₁ d.b₂))) - (min d.a₁ (min d.a₂ (min d.b₁ d.b₂)))) - 1

/-- Theorem stating the existence of most and least advantageous methods for Kolya -/
theorem kolya_optimal_strategy (n : ℕ) (h : n ≥ 2) :
  ∃ (best worst : Method) (d : NutDivision),
    (d.a₁ + d.a₂ + d.b₁ + d.b₂ = 2*n + 1) ∧
    (d.a₁ ≥ 1 ∧ d.a₂ ≥ 1 ∧ d.b₁ ≥ 1 ∧ d.b₂ ≥ 1) ∧
    (∀ m : Method, nuts_for_kolya best d ≥ nuts_for_kolya m d) ∧
    (∀ m : Method, nuts_for_kolya worst d ≤ nuts_for_kolya m d) :=
  sorry

end kolya_optimal_strategy_l3537_353765


namespace line_segment_polar_equation_l3537_353771

/-- The polar equation of the line segment y = 1 - x where 0 ≤ x ≤ 1 -/
theorem line_segment_polar_equation (θ : Real) (ρ : Real) :
  (0 ≤ θ) ∧ (θ ≤ Real.pi / 2) →
  (ρ * Real.cos θ + ρ * Real.sin θ = 1) ↔
  (ρ * Real.sin θ = 1 - ρ * Real.cos θ) ∧
  (0 ≤ ρ * Real.cos θ) ∧ (ρ * Real.cos θ ≤ 1) :=
by sorry

end line_segment_polar_equation_l3537_353771


namespace locus_of_points_l3537_353795

/-- Two lines in a plane --/
structure TwoLines where
  l₁ : Set (ℝ × ℝ)
  l₃ : Set (ℝ × ℝ)

/-- Distance from a point to a line --/
def distanceToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

/-- Translate a line by a distance --/
def translateLine (l : Set (ℝ × ℝ)) (d : ℝ) : Set (ℝ × ℝ) := sorry

/-- Angle bisector of two lines --/
def angleBisector (l1 l2 : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- The theorem statement --/
theorem locus_of_points (lines : TwoLines) (a : ℝ) :
  ∀ (M : ℝ × ℝ), 
    (distanceToLine M lines.l₁ + distanceToLine M lines.l₃ = a) →
    ∃ (d : ℝ), M ∈ angleBisector lines.l₁ (translateLine lines.l₃ d) := by
  sorry

end locus_of_points_l3537_353795


namespace employee_payment_l3537_353706

theorem employee_payment (total : ℝ) (x y : ℝ) (h1 : total = 572) (h2 : x = 1.2 * y) (h3 : total = x + y) : y = 260 :=
by
  sorry

end employee_payment_l3537_353706


namespace equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solutions_l3537_353773

-- Equation 1
theorem equation_one_solutions (x : ℝ) : 
  (x + 2)^2 = 2*x + 4 ↔ x = 0 ∨ x = -2 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) : 
  x^2 - 2*x - 5 = 0 ↔ x = 1 + Real.sqrt 6 ∨ x = 1 - Real.sqrt 6 := by sorry

-- Equation 3
theorem equation_three_solutions (x : ℝ) : 
  x^2 - 5*x - 6 = 0 ↔ x = -1 ∨ x = 6 := by sorry

-- Equation 4
theorem equation_four_solutions (x : ℝ) : 
  (x + 3)^2 = (1 - 2*x)^2 ↔ x = -2/3 ∨ x = 4 := by sorry

end equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solutions_l3537_353773


namespace arrangements_eight_athletes_three_consecutive_l3537_353754

/-- The number of tracks and athletes -/
def n : ℕ := 8

/-- The number of specified athletes that must be in consecutive tracks -/
def k : ℕ := 3

/-- The number of ways to arrange n athletes on n tracks, 
    where k specified athletes must be in consecutive tracks -/
def arrangements (n k : ℕ) : ℕ := sorry

/-- Theorem stating the correct number of arrangements for the given problem -/
theorem arrangements_eight_athletes_three_consecutive : 
  arrangements n k = 4320 := by sorry

end arrangements_eight_athletes_three_consecutive_l3537_353754


namespace gcd_75_225_l3537_353786

theorem gcd_75_225 : Nat.gcd 75 225 = 75 := by
  sorry

end gcd_75_225_l3537_353786


namespace largest_non_sum_of_composites_l3537_353767

def isComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def isSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, isComposite a ∧ isComposite b ∧ a + b = n

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → isSumOfTwoComposites n) ∧
  ¬isSumOfTwoComposites 11 :=
sorry

end largest_non_sum_of_composites_l3537_353767


namespace min_operations_for_two_pints_l3537_353787

/-- Represents the state of the two vessels -/
structure VesselState :=
  (v7 : ℕ)
  (v11 : ℕ)

/-- Represents an operation on the vessels -/
inductive Operation
  | Fill7
  | Fill11
  | Empty7
  | Empty11
  | Pour7To11
  | Pour11To7

/-- Applies an operation to a vessel state -/
def applyOperation (state : VesselState) (op : Operation) : VesselState :=
  match op with
  | Operation.Fill7 => ⟨7, state.v11⟩
  | Operation.Fill11 => ⟨state.v7, 11⟩
  | Operation.Empty7 => ⟨0, state.v11⟩
  | Operation.Empty11 => ⟨state.v7, 0⟩
  | Operation.Pour7To11 => 
      let amount := min state.v7 (11 - state.v11)
      ⟨state.v7 - amount, state.v11 + amount⟩
  | Operation.Pour11To7 => 
      let amount := min state.v11 (7 - state.v7)
      ⟨state.v7 + amount, state.v11 - amount⟩

/-- Checks if a sequence of operations results in 2 pints in either vessel -/
def isValidSolution (ops : List Operation) : Prop :=
  let finalState := ops.foldl applyOperation ⟨0, 0⟩
  finalState.v7 = 2 ∨ finalState.v11 = 2

/-- The main theorem stating that 14 is the minimum number of operations -/
theorem min_operations_for_two_pints :
  (∃ (ops : List Operation), ops.length = 14 ∧ isValidSolution ops) ∧
  (∀ (ops : List Operation), ops.length < 14 → ¬isValidSolution ops) :=
sorry

end min_operations_for_two_pints_l3537_353787


namespace pauls_journey_time_l3537_353708

theorem pauls_journey_time (paul_time : ℝ) : 
  (paul_time + 7 * (paul_time + 2) = 46) → paul_time = 4 := by
  sorry

end pauls_journey_time_l3537_353708


namespace school_cafeteria_discussion_l3537_353755

theorem school_cafeteria_discussion (students_like : ℕ) (students_dislike : ℕ) : 
  students_like = 383 → students_dislike = 431 → students_like + students_dislike = 814 :=
by sorry

end school_cafeteria_discussion_l3537_353755


namespace john_needs_four_planks_l3537_353758

/-- The number of planks John needs for the house wall -/
def num_planks (total_nails : ℕ) (nails_per_plank : ℕ) (additional_nails : ℕ) : ℕ :=
  (total_nails - additional_nails) / nails_per_plank

/-- Theorem stating that John needs 4 planks for the house wall -/
theorem john_needs_four_planks :
  num_planks 43 7 15 = 4 := by
  sorry

end john_needs_four_planks_l3537_353758


namespace tiffany_sunscreen_cost_l3537_353724

/-- Calculates the cost of sunscreen for a beach visit given the specified parameters. -/
def sunscreenCost (reapplyInterval : ℕ) (amountPerApplication : ℕ) (bottleSize : ℕ) (bottleCost : ℚ) (visitDuration : ℕ) : ℚ :=
  let applications := visitDuration / reapplyInterval
  let totalAmount := applications * amountPerApplication
  let bottlesNeeded := (totalAmount + bottleSize - 1) / bottleSize  -- Ceiling division
  bottlesNeeded * bottleCost

/-- Theorem stating that the sunscreen cost for Tiffany's beach visit is $7. -/
theorem tiffany_sunscreen_cost :
  sunscreenCost 2 3 12 (7/2) 16 = 7 := by
  sorry

#eval sunscreenCost 2 3 12 (7/2) 16

end tiffany_sunscreen_cost_l3537_353724


namespace power_function_increasing_m_l3537_353769

/-- A function f is a power function if it can be written as f(x) = ax^n for some constants a and n, where a ≠ 0 -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^n

/-- A function f is increasing on (0, +∞) if for any x1 < x2 in (0, +∞), f(x1) < f(x2) -/
def isIncreasingOnPositiveReals (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, 0 < x1 → x1 < x2 → f x1 < f x2

/-- The main theorem -/
theorem power_function_increasing_m (m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (m^2 - m - 5) * x^m
  isPowerFunction f ∧ isIncreasingOnPositiveReals f → m = 3 := by
  sorry

end power_function_increasing_m_l3537_353769


namespace solution_set_is_real_solution_set_is_empty_solution_set_has_element_l3537_353721

-- Define the quadratic expression
def f (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 * a - 3

-- Define the solution set for the inequality
def solution_set (a : ℝ) : Set ℝ := {x | f a x < 0}

-- Theorem 1: The solution set is ℝ iff a ∈ (-∞, 0]
theorem solution_set_is_real : ∀ a : ℝ, solution_set a = Set.univ ↔ a ≤ 0 := by sorry

-- Theorem 2: The solution set is ∅ iff a ∈ [3, +∞)
theorem solution_set_is_empty : ∀ a : ℝ, solution_set a = ∅ ↔ a ≥ 3 := by sorry

-- Theorem 3: There is at least one real solution iff a ∈ (-∞, 3)
theorem solution_set_has_element : ∀ a : ℝ, (∃ x : ℝ, x ∈ solution_set a) ↔ a < 3 := by sorry

end solution_set_is_real_solution_set_is_empty_solution_set_has_element_l3537_353721


namespace groups_with_pair_fraction_l3537_353764

-- Define the number of people
def n : ℕ := 6

-- Define the size of each group
def k : ℕ := 3

-- Define the function to calculate combinations
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem groups_with_pair_fraction :
  C (n - 2) (k - 2) / C n k = 1 / 5 := by sorry

end groups_with_pair_fraction_l3537_353764


namespace product_of_four_numbers_l3537_353735

theorem product_of_four_numbers (a b c d : ℚ) : 
  a + b + c + d = 36 →
  a = 3 * (b + c + d) →
  b = 5 * c →
  d = (1/2) * c →
  a * b * c * d = 178.5 := by
sorry

end product_of_four_numbers_l3537_353735


namespace cubic_root_approximation_bound_l3537_353789

theorem cubic_root_approximation_bound :
  ∃ (c : ℝ), c > 0 ∧ ∀ (m n : ℤ), n ≥ 1 →
    |2^(1/3 : ℝ) - (m : ℝ) / (n : ℝ)| > c / (n : ℝ)^3 := by
  sorry

end cubic_root_approximation_bound_l3537_353789


namespace simplify_and_evaluate_evaluate_expression_l3537_353716

theorem simplify_and_evaluate (a b : ℝ) :
  (a - b)^2 - 2*a*(a + b) + (a + 2*b)*(a - 2*b) = -4*a*b - 3*b^2 :=
by sorry

theorem evaluate_expression :
  let a : ℝ := -1
  let b : ℝ := 4
  (a - b)^2 - 2*a*(a + b) + (a + 2*b)*(a - 2*b) = -32 :=
by sorry

end simplify_and_evaluate_evaluate_expression_l3537_353716


namespace fence_cost_l3537_353783

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 55) :
  4 * price_per_foot * Real.sqrt area = 3740 := by
  sorry

end fence_cost_l3537_353783


namespace price_reduction_for_1200_profit_no_solution_for_1600_profit_l3537_353733

-- Define the initial conditions
def initial_sales : ℕ := 30
def initial_profit : ℕ := 40
def sales_increase_rate : ℕ := 2

-- Define the profit function
def daily_profit (price_reduction : ℝ) : ℝ :=
  (initial_profit - price_reduction) * (initial_sales + sales_increase_rate * price_reduction)

-- Theorem for part 1
theorem price_reduction_for_1200_profit :
  ∃ (x : ℝ), x > 0 ∧ daily_profit x = 1200 ∧ 
  (∀ (y : ℝ), y > 0 ∧ y ≠ x → daily_profit y ≠ 1200) :=
sorry

-- Theorem for part 2
theorem no_solution_for_1600_profit :
  ¬∃ (x : ℝ), daily_profit x = 1600 :=
sorry

end price_reduction_for_1200_profit_no_solution_for_1600_profit_l3537_353733


namespace sum_of_integers_l3537_353700

theorem sum_of_integers (w x y z : ℤ) 
  (eq1 : w - x + y = 7)
  (eq2 : x - y + z = 8)
  (eq3 : y - z + w = 4)
  (eq4 : z - w + x = 3) :
  w + x + y + z = 11 := by
sorry

end sum_of_integers_l3537_353700


namespace at_least_one_not_less_than_two_l3537_353784

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end at_least_one_not_less_than_two_l3537_353784


namespace two_solutions_iff_a_gt_neg_one_l3537_353737

/-- The equation has exactly two solutions if and only if a > -1 -/
theorem two_solutions_iff_a_gt_neg_one (a : ℝ) :
  (∃! x y, x ≠ y ∧ x^2 + 2*x + 2*|x+1| = a ∧ y^2 + 2*y + 2*|y+1| = a) ↔ a > -1 := by
  sorry

end two_solutions_iff_a_gt_neg_one_l3537_353737


namespace rectangle_new_length_l3537_353763

/-- Given a rectangle with original length 18 cm and breadth 10 cm,
    if the breadth is changed to 7.2 cm while maintaining the same area,
    the new length will be 25 cm. -/
theorem rectangle_new_length (original_length original_breadth new_breadth new_length : ℝ) :
  original_length = 18 ∧
  original_breadth = 10 ∧
  new_breadth = 7.2 ∧
  original_length * original_breadth = new_length * new_breadth →
  new_length = 25 := by
  sorry


end rectangle_new_length_l3537_353763


namespace expression_evaluation_l3537_353712

theorem expression_evaluation : 
  (4+8-16+32+64-128+256)/(8+16-32+64+128-256+512) = 1/2 := by
sorry

end expression_evaluation_l3537_353712


namespace sugar_consumption_reduction_l3537_353761

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h_initial : initial_price = 10)
  (h_new : new_price = 13) :
  let reduction_percentage := (new_price - initial_price) / initial_price * 100
  reduction_percentage = 30 := by
  sorry

end sugar_consumption_reduction_l3537_353761


namespace apartments_per_floor_l3537_353703

theorem apartments_per_floor 
  (stories : ℕ) 
  (people_per_apartment : ℕ) 
  (total_people : ℕ) 
  (h1 : stories = 25)
  (h2 : people_per_apartment = 2)
  (h3 : total_people = 200) :
  (total_people / (stories * people_per_apartment) : ℚ) = 4 := by
  sorry

end apartments_per_floor_l3537_353703


namespace hyperbola_theorem_l3537_353723

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the asymptote condition
def asymptote_condition (a b : ℝ) : Prop :=
  b / a = Real.sqrt 3

-- Define the focus condition
def focus_condition (c : ℝ) : Prop :=
  Real.sqrt ((1 + c)^2 + 3) = 2

-- Define the point condition
def point_condition (x y c : ℝ) : Prop :=
  Real.sqrt ((x + c)^2 + y^2) = 5/2

-- Main theorem
theorem hyperbola_theorem (a b c : ℝ) (x y : ℝ) :
  hyperbola a b x y →
  asymptote_condition a b →
  focus_condition c →
  point_condition x y c →
  Real.sqrt ((x - c)^2 + y^2) = 9/2 :=
by sorry

end hyperbola_theorem_l3537_353723


namespace union_of_A_and_B_complement_A_intersect_B_l3537_353701

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 10} := by sorry

-- Theorem for (∁A) ∩ B
theorem complement_A_intersect_B : (Aᶜ) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10} := by sorry

end union_of_A_and_B_complement_A_intersect_B_l3537_353701


namespace mountain_climb_speed_l3537_353711

-- Define the parameters
def total_time : ℝ := 20
def total_distance : ℝ := 80

-- Define the variables
variable (v : ℝ)  -- Speed on the first day
variable (t : ℝ)  -- Time spent on the first day

-- Define the theorem
theorem mountain_climb_speed :
  -- Conditions
  (t + (t - 2) + (t + 1) = total_time) →
  (v * t + (v + 0.5) * (t - 2) + (v - 0.5) * (t + 1) = total_distance) →
  -- Conclusion
  (v + 0.5 = 4.575) :=
by
  sorry  -- Proof omitted

end mountain_climb_speed_l3537_353711


namespace inscribed_circle_radius_l3537_353739

theorem inscribed_circle_radius (PQ QR : Real) (h1 : PQ = 15) (h2 : QR = 8) : 
  let PR := Real.sqrt (PQ^2 + QR^2)
  let s := (PQ + QR + PR) / 2
  let area := PQ * QR / 2
  area / s = 3 := by sorry

end inscribed_circle_radius_l3537_353739


namespace initial_bottles_count_l3537_353718

/-- The number of bottles Jason buys -/
def jason_bottles : ℕ := 5

/-- The number of bottles Harry buys -/
def harry_bottles : ℕ := 6

/-- The number of bottles left on the shelf after purchases -/
def remaining_bottles : ℕ := 24

/-- The initial number of bottles on the shelf -/
def initial_bottles : ℕ := jason_bottles + harry_bottles + remaining_bottles

theorem initial_bottles_count : initial_bottles = 35 := by
  sorry

end initial_bottles_count_l3537_353718


namespace min_weighings_three_l3537_353781

/-- Represents the outcome of a weighing --/
inductive WeighingOutcome
  | Equal : WeighingOutcome
  | LeftHeavier : WeighingOutcome
  | RightHeavier : WeighingOutcome

/-- Represents a coin --/
inductive Coin
  | Real : Coin
  | Fake : Coin

/-- Represents a weighing strategy --/
def WeighingStrategy := List (List Coin × List Coin)

/-- The total number of coins --/
def totalCoins : Nat := 2023

/-- The number of fake coins --/
def fakeCoins : Nat := 2

/-- The number of real coins --/
def realCoins : Nat := totalCoins - fakeCoins

/-- A function that determines the outcome of a weighing --/
def weighOutcome (left right : List Coin) : WeighingOutcome := sorry

/-- A function that determines if a strategy is valid --/
def isValidStrategy (strategy : WeighingStrategy) : Prop := sorry

/-- A function that determines if a strategy solves the problem --/
def solvesProblem (strategy : WeighingStrategy) : Prop := sorry

/-- The main theorem stating that the minimum number of weighings is 3 --/
theorem min_weighings_three :
  ∃ (strategy : WeighingStrategy),
    strategy.length = 3 ∧
    isValidStrategy strategy ∧
    solvesProblem strategy ∧
    ∀ (other : WeighingStrategy),
      isValidStrategy other →
      solvesProblem other →
      other.length ≥ 3 := by sorry

end min_weighings_three_l3537_353781


namespace total_pencils_is_52_l3537_353749

/-- The number of pencils in a pack -/
def pencils_per_pack : ℕ := 12

/-- The number of packs Jimin has -/
def jimin_packs : ℕ := 2

/-- The number of individual pencils Jimin has -/
def jimin_individual : ℕ := 7

/-- The number of packs Yuna has -/
def yuna_packs : ℕ := 1

/-- The number of individual pencils Yuna has -/
def yuna_individual : ℕ := 9

/-- The total number of pencils Jimin and Yuna have -/
def total_pencils : ℕ := 
  jimin_packs * pencils_per_pack + jimin_individual +
  yuna_packs * pencils_per_pack + yuna_individual

theorem total_pencils_is_52 : total_pencils = 52 := by
  sorry

end total_pencils_is_52_l3537_353749


namespace binomial_coefficient_minus_two_divisible_by_prime_l3537_353720

theorem binomial_coefficient_minus_two_divisible_by_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ k : ℤ, (2 * p).factorial / (p.factorial * p.factorial) - 2 = k * p :=
sorry

end binomial_coefficient_minus_two_divisible_by_prime_l3537_353720


namespace absolute_value_subtraction_l3537_353709

theorem absolute_value_subtraction : 2 - |(-3)| = -1 := by
  sorry

end absolute_value_subtraction_l3537_353709


namespace simplify_expression_l3537_353742

theorem simplify_expression (α : ℝ) (h : π < α ∧ α < (3*π)/2) :
  Real.sqrt (1/2 + 1/2 * Real.sqrt (1/2 + 1/2 * Real.cos (2*α))) = Real.sin (α/2) := by
  sorry

end simplify_expression_l3537_353742


namespace rhombus_square_diagonals_l3537_353719

-- Define a rhombus
structure Rhombus :=
  (sides_equal : ∀ s1 s2 : ℝ, s1 = s2)
  (diagonals_perpendicular : Bool)

-- Define a square as a special case of rhombus
structure Square extends Rhombus :=
  (all_angles_right : Bool)

-- Theorem statement
theorem rhombus_square_diagonals :
  ∃ (r : Rhombus), ¬(∀ d1 d2 : ℝ, d1 = d2) ∧
  ∀ (s : Square), ∀ d1 d2 : ℝ, d1 = d2 :=
sorry

end rhombus_square_diagonals_l3537_353719


namespace bee_paths_count_l3537_353760

/-- Represents the number of beehives in the row -/
def n : ℕ := 6

/-- Represents the possible moves of the bee -/
inductive BeeMove
  | Right
  | UpperRight
  | LowerRight

/-- Represents a path of the bee as a list of moves -/
def BeePath := List BeeMove

/-- Checks if a path is valid (ends at hive number 6) -/
def isValidPath (path : BeePath) : Bool :=
  sorry

/-- Counts the number of valid paths to hive number 6 -/
def countValidPaths : ℕ :=
  sorry

/-- Theorem: The number of valid paths to hive number 6 is 21 -/
theorem bee_paths_count : countValidPaths = 21 := by
  sorry

end bee_paths_count_l3537_353760


namespace bara_numbers_l3537_353766

theorem bara_numbers (a b : ℤ) (h1 : a ≠ b) 
  (h2 : (a + b) + (a - b) + a * b + a / b = -100)
  (h3 : (a - b) + a * b + a / b = -100) :
  (a = -9 ∧ b = 9) ∨ (a = 11 ∧ b = -11) := by
sorry

end bara_numbers_l3537_353766


namespace sum_of_specific_numbers_l3537_353702

theorem sum_of_specific_numbers : 
  22000000 + 22000 + 2200 + 22 = 22024222 := by sorry

end sum_of_specific_numbers_l3537_353702


namespace geometric_sequence_sum_l3537_353779

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 3 = 3 →
  a 6 = 1 / 9 →
  a 4 + a 5 = 4 / 3 := by
  sorry

end geometric_sequence_sum_l3537_353779


namespace amy_and_noah_total_books_l3537_353757

/-- The number of books owned by different people -/
structure BookCounts where
  maddie : ℕ
  luisa : ℕ
  amy : ℕ
  noah : ℕ

/-- The conditions of the book counting problem -/
def BookProblemConditions (bc : BookCounts) : Prop :=
  bc.maddie = 15 ∧
  bc.luisa = 18 ∧
  bc.amy + bc.luisa = bc.maddie + 9 ∧
  bc.noah = bc.amy / 3

/-- The theorem stating that under the given conditions, Amy and Noah have 8 books in total -/
theorem amy_and_noah_total_books (bc : BookCounts) 
  (h : BookProblemConditions bc) : bc.amy + bc.noah = 8 := by
  sorry

end amy_and_noah_total_books_l3537_353757


namespace triangle_two_solutions_l3537_353776

theorem triangle_two_solutions (a b : ℝ) (A : ℝ) :
  a = 6 →
  b = 6 * Real.sqrt 3 →
  A = π / 6 →
  ∃! (c₁ c₂ B₁ B₂ C₁ C₂ : ℝ),
    (c₁ = 12 ∧ B₁ = π / 3 ∧ C₁ = π / 2) ∧
    (c₂ = 6 ∧ B₂ = 2 * π / 3 ∧ C₂ = π / 6) ∧
    (∀ c B C : ℝ,
      (c = c₁ ∧ B = B₁ ∧ C = C₁) ∨
      (c = c₂ ∧ B = B₂ ∧ C = C₂)) :=
by sorry

end triangle_two_solutions_l3537_353776


namespace fractional_equation_positive_root_l3537_353793

theorem fractional_equation_positive_root (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (x + 5) / (x - 3) = 2 - m / (3 - x)) → m = 8 := by
  sorry

end fractional_equation_positive_root_l3537_353793


namespace problem_solution_l3537_353713

theorem problem_solution : ∃! x : ℝ, (0.8 * x) = ((4 / 5) * 25 + 28) := by
  sorry

end problem_solution_l3537_353713


namespace number_problem_l3537_353756

theorem number_problem : ∃ x : ℚ, (x / 6) * 12 = 12 ∧ x = 6 := by
  sorry

end number_problem_l3537_353756


namespace josh_bought_six_cds_l3537_353770

/-- Represents the shopping problem where Josh buys films, books, and CDs. -/
def ShoppingProblem (num_films num_books total_spent : ℕ) (film_cost book_cost cd_cost : ℚ) : Prop :=
  ∃ (num_cds : ℕ),
    (num_films : ℚ) * film_cost + (num_books : ℚ) * book_cost + (num_cds : ℚ) * cd_cost = total_spent

/-- Proves that Josh bought 6 CDs given the problem conditions. -/
theorem josh_bought_six_cds :
  ShoppingProblem 9 4 79 5 4 3 → (∃ (num_cds : ℕ), num_cds = 6) :=
by
  sorry

#check josh_bought_six_cds

end josh_bought_six_cds_l3537_353770


namespace share_ratio_proof_l3537_353790

theorem share_ratio_proof (total : ℝ) (c_share : ℝ) (f : ℝ) :
  total = 700 →
  c_share = 400 →
  0 < f →
  f ≤ 1 →
  total = f^2 * c_share + f * c_share + c_share →
  (f^2 * c_share) / (f * c_share) = 1 / 2 ∧
  (f * c_share) / c_share = 1 / 2 :=
by sorry

end share_ratio_proof_l3537_353790


namespace taxi_fare_problem_l3537_353794

/-- The fare structure for a taxi ride -/
structure TaxiFare where
  fixedCharge : ℝ
  ratePerMile : ℝ

/-- Calculate the total fare for a given distance -/
def totalFare (fare : TaxiFare) (distance : ℝ) : ℝ :=
  fare.fixedCharge + fare.ratePerMile * distance

/-- The problem statement -/
theorem taxi_fare_problem (fare : TaxiFare) 
  (h1 : totalFare fare 80 = 200)
  (h2 : fare.fixedCharge = 20) :
  totalFare fare 100 = 245 := by
  sorry


end taxi_fare_problem_l3537_353794


namespace complement_of_beta_l3537_353727

-- Define angles α and β
variable (α β : Real)

-- Define the conditions
def complementary : Prop := α + β = 180
def alpha_greater : Prop := α > β

-- Define the complement of an angle
def complement (θ : Real) : Real := 90 - θ

-- State the theorem
theorem complement_of_beta (h1 : complementary α β) (h2 : alpha_greater α β) :
  complement β = (α - β) / 2 := by
  sorry

end complement_of_beta_l3537_353727


namespace sum_of_powers_l3537_353732

theorem sum_of_powers (x : ℝ) (h : x + 1/x = 4) : x^6 + 1/x^6 = 2702 := by
  sorry

end sum_of_powers_l3537_353732


namespace factorization_of_polynomial_l3537_353798

theorem factorization_of_polynomial (x : ℝ) :
  x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3)*(8*x^2 - x + 3) ∧
  (∀ a b c d e f : ℤ, (-8*x^2 + x - 3) = a*x^2 + b*x + c ∧
                       (8*x^2 - x + 3) = d*x^2 + e*x + f ∧
                       a < d) :=
by sorry

end factorization_of_polynomial_l3537_353798


namespace ap_80th_term_l3537_353777

/-- An arithmetic progression (AP) with given properties -/
structure AP where
  /-- Sum of the first 20 terms -/
  sum20 : ℚ
  /-- Sum of the first 60 terms -/
  sum60 : ℚ
  /-- The property that sum20 = 200 -/
  sum20_eq : sum20 = 200
  /-- The property that sum60 = 180 -/
  sum60_eq : sum60 = 180

/-- The 80th term of the AP -/
def term80 (ap : AP) : ℚ := -573/40

/-- Theorem stating that the 80th term of the AP with given properties is -573/40 -/
theorem ap_80th_term (ap : AP) : term80 ap = -573/40 := by
  sorry

end ap_80th_term_l3537_353777


namespace min_value_3m_plus_n_l3537_353717

/-- Given a triangle ABC with point G satisfying the centroid condition,
    and points M on AB and N on AC with specific vector relationships,
    prove that the minimum value of 3m + n is 4/3 + 2√3/3 -/
theorem min_value_3m_plus_n (A B C G M N : ℝ × ℝ) (m n : ℝ) :
  (G.1 - A.1 + G.1 - B.1 + G.1 - C.1 = 0 ∧
   G.2 - A.2 + G.2 - B.2 + G.2 - C.2 = 0) →
  (∃ t : ℝ, M = (1 - t) • A + t • B ∧
            N = (1 - t) • A + t • C) →
  (M.1 - A.1 = m * (B.1 - A.1) ∧
   M.2 - A.2 = m * (B.2 - A.2)) →
  (N.1 - A.1 = n * (C.1 - A.1) ∧
   N.2 - A.2 = n * (C.2 - A.2)) →
  m > 0 →
  n > 0 →
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 3 * m + n ≤ 3 * m' + n') →
  3 * m + n = 4/3 + 2 * Real.sqrt 3 / 3 :=
by sorry

end min_value_3m_plus_n_l3537_353717


namespace no_prime_solution_l3537_353775

theorem no_prime_solution : ¬∃ p : ℕ, Nat.Prime p ∧ p^3 + 6*p^2 + 4*p + 28 = 6*p^2 + 17*p + 6 := by
  sorry

end no_prime_solution_l3537_353775


namespace triangle_angle_inequality_l3537_353729

/-- Given a triangle ABC with side lengths a and b, and angles A, B, and C,
    prove that C > B > A when a = 1, b = √3, A = 30°, and B is acute. -/
theorem triangle_angle_inequality (a b : ℝ) (A B C : ℝ) : 
  a = 1 → 
  b = Real.sqrt 3 → 
  A = π / 6 → 
  0 < B ∧ B < π / 2 → 
  a * Real.sin B = b * Real.sin A →
  A + B + C = π →
  C > B ∧ B > A := by
  sorry

end triangle_angle_inequality_l3537_353729


namespace p_or_q_can_be_either_l3537_353722

theorem p_or_q_can_be_either (p q : Prop) (h : ¬(p ∧ q)) : 
  (∃ b : Bool, (p ∨ q) = b) ∧ (∃ b : Bool, (p ∨ q) ≠ b) := by
sorry

end p_or_q_can_be_either_l3537_353722


namespace sum_of_largest_and_smallest_odd_l3537_353752

def isOdd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

def inRange (n : ℕ) : Prop := 5 ≤ n ∧ n ≤ 12

theorem sum_of_largest_and_smallest_odd : 
  ∃ (a b : ℕ), 
    isOdd a ∧ isOdd b ∧ 
    inRange a ∧ inRange b ∧
    (∀ x, isOdd x ∧ inRange x → a ≤ x ∧ x ≤ b) ∧
    a + b = 16 :=
by sorry

end sum_of_largest_and_smallest_odd_l3537_353752


namespace solution_concentration_l3537_353731

/-- Theorem: Concentration of solution to be added to achieve target concentration --/
theorem solution_concentration 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (drain_volume : ℝ) 
  (final_concentration : ℝ) 
  (h1 : initial_volume = 50)
  (h2 : initial_concentration = 0.6)
  (h3 : drain_volume = 35)
  (h4 : final_concentration = 0.46)
  : ∃ (x : ℝ), 
    (initial_volume - drain_volume) * initial_concentration + drain_volume * x = 
    initial_volume * final_concentration ∧ 
    x = 0.4 := by
  sorry

end solution_concentration_l3537_353731


namespace points_three_units_from_negative_one_l3537_353747

theorem points_three_units_from_negative_one : 
  ∀ x : ℝ, abs (x - (-1)) = 3 ↔ x = 2 ∨ x = -4 := by sorry

end points_three_units_from_negative_one_l3537_353747


namespace four_digit_number_proof_l3537_353743

/-- Represents a four-digit number -/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Returns the largest number that can be formed by rearranging the digits of a given number -/
def largest_rearrangement (n : FourDigitNumber) : ℕ :=
  sorry

/-- Returns the smallest number that can be formed by rearranging the digits of a given number -/
def smallest_rearrangement (n : FourDigitNumber) : ℕ :=
  sorry

/-- Checks if a number has any digit equal to 0 -/
def has_zero_digit (n : ℕ) : Bool :=
  sorry

theorem four_digit_number_proof :
  ∃ (A : FourDigitNumber),
    largest_rearrangement A = A.value + 7668 ∧
    smallest_rearrangement A = A.value - 594 ∧
    ¬ has_zero_digit A.value ∧
    A.value = 1963 :=
  sorry

end four_digit_number_proof_l3537_353743


namespace pythagorean_triple_has_even_number_l3537_353782

theorem pythagorean_triple_has_even_number (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  Even a ∨ Even b ∨ Even c := by
  sorry

end pythagorean_triple_has_even_number_l3537_353782


namespace quadratic_equation_m_value_l3537_353759

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m + 2) * x^(m^2 - 2) + 2 * x + 1 = a * x^2 + b * x + c) ∧ 
  (m + 2 ≠ 0) → 
  m = 2 := by
  sorry

end quadratic_equation_m_value_l3537_353759


namespace clock_synchronization_l3537_353715

/-- The number of minutes in 12 hours -/
def minutes_in_12_hours : ℕ := 12 * 60

/-- The number of minutes Arthur's clock gains per day -/
def arthur_gain : ℕ := 15

/-- The number of minutes Oleg's clock gains per day -/
def oleg_gain : ℕ := 12

/-- The number of days it takes for Arthur's clock to gain 12 hours -/
def arthur_days : ℕ := minutes_in_12_hours / arthur_gain

/-- The number of days it takes for Oleg's clock to gain 12 hours -/
def oleg_days : ℕ := minutes_in_12_hours / oleg_gain

theorem clock_synchronization :
  Nat.lcm arthur_days oleg_days = 240 := by sorry

end clock_synchronization_l3537_353715


namespace front_view_of_stack_map_l3537_353774

/-- Represents a stack map with four columns --/
structure StackMap :=
  (A : ℕ)
  (B : ℕ)
  (C : ℕ)
  (D : ℕ)

/-- Represents the front view of a stack map --/
def FrontView := List ℕ

/-- Computes the front view of a stack map --/
def computeFrontView (sm : StackMap) : FrontView :=
  [sm.A, sm.B, sm.C, sm.D]

/-- Theorem: The front view of the given stack map is [3, 5, 2, 4] --/
theorem front_view_of_stack_map :
  let sm : StackMap := { A := 3, B := 5, C := 2, D := 4 }
  computeFrontView sm = [3, 5, 2, 4] := by
  sorry

end front_view_of_stack_map_l3537_353774


namespace alberts_age_l3537_353741

theorem alberts_age (dad_age : ℕ) (h1 : dad_age = 48) : ∃ (albert_age : ℕ),
  (albert_age = 15) ∧ 
  (dad_age - 4 = 4 * (albert_age - 4)) :=
by
  sorry

end alberts_age_l3537_353741


namespace part1_solution_set_part2_solution_set_l3537_353740

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part1_solution_set :
  {x : ℝ | f 1 x < |2*x - 1| - 1} = {x : ℝ | x < -1 ∨ x > 1} := by sorry

-- Part 2
theorem part2_solution_set :
  ∀ x ∈ Set.Ioo (-2) 1, {a : ℝ | |x - 1| > |2*x - a - 1| - f a x} = Set.Iic (-2) := by sorry

end part1_solution_set_part2_solution_set_l3537_353740


namespace fresh_fruits_ratio_l3537_353734

/-- Represents the quantity of fruits in the store -/
structure FruitQuantity where
  pineapples : ℕ
  apples : ℕ
  oranges : ℕ

/-- Represents the spoilage rates of fruits -/
structure SpoilageRate where
  pineapples : ℚ
  apples : ℚ
  oranges : ℚ

def initialQuantity : FruitQuantity :=
  { pineapples := 200, apples := 300, oranges := 100 }

def soldQuantity : FruitQuantity :=
  { pineapples := 56, apples := 128, oranges := 22 }

def spoilageRate : SpoilageRate :=
  { pineapples := 1/10, apples := 15/100, oranges := 1/20 }

def remainingFruits (initial : FruitQuantity) (sold : FruitQuantity) : FruitQuantity :=
  { pineapples := initial.pineapples - sold.pineapples,
    apples := initial.apples - sold.apples,
    oranges := initial.oranges - sold.oranges }

def spoiledFruits (remaining : FruitQuantity) (rate : SpoilageRate) : FruitQuantity :=
  { pineapples := (remaining.pineapples : ℚ) * rate.pineapples |> round |> Int.toNat,
    apples := (remaining.apples : ℚ) * rate.apples |> round |> Int.toNat,
    oranges := (remaining.oranges : ℚ) * rate.oranges |> round |> Int.toNat }

def freshFruits (remaining : FruitQuantity) (spoiled : FruitQuantity) : FruitQuantity :=
  { pineapples := remaining.pineapples - spoiled.pineapples,
    apples := remaining.apples - spoiled.apples,
    oranges := remaining.oranges - spoiled.oranges }

theorem fresh_fruits_ratio :
  let remaining := remainingFruits initialQuantity soldQuantity
  let spoiled := spoiledFruits remaining spoilageRate
  let fresh := freshFruits remaining spoiled
  fresh.pineapples = 130 ∧ fresh.apples = 146 ∧ fresh.oranges = 74 := by sorry

end fresh_fruits_ratio_l3537_353734


namespace triangle_shape_l3537_353768

theorem triangle_shape (A : ℝ) (hA : 0 < A ∧ A < π) 
  (h : Real.sin A + Real.cos A = 12/25) : A > π/2 := by
  sorry

end triangle_shape_l3537_353768


namespace f_comparison_l3537_353796

def f (a b x : ℝ) := a * x^2 - 2 * b * x + 1

theorem f_comparison (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_increasing : ∀ x y, x ≤ y → y ≤ 0 → f a b x ≤ f a b y) :
  f a b (a - 2) < f a b (b + 1) :=
by sorry

end f_comparison_l3537_353796


namespace prob_at_least_two_tails_in_three_flips_prob_at_least_two_tails_in_three_flips_is_half_l3537_353785

/-- The probability of getting at least two tails in three independent flips of a fair coin -/
theorem prob_at_least_two_tails_in_three_flips : ℝ :=
  let p_head : ℝ := 1/2  -- probability of getting heads on a single flip
  let p_tail : ℝ := 1 - p_head  -- probability of getting tails on a single flip
  let p_all_heads : ℝ := p_head ^ 3  -- probability of getting all heads
  let p_one_tail : ℝ := 3 * p_head ^ 2 * p_tail  -- probability of getting exactly one tail
  1 - (p_all_heads + p_one_tail)

/-- The probability of getting at least two tails in three independent flips of a fair coin is 1/2 -/
theorem prob_at_least_two_tails_in_three_flips_is_half :
  prob_at_least_two_tails_in_three_flips = 1/2 := by
  sorry

end prob_at_least_two_tails_in_three_flips_prob_at_least_two_tails_in_three_flips_is_half_l3537_353785


namespace train_length_l3537_353745

/-- The length of a train given its speed, the speed of a man walking in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 60.994720422366214 →
  man_speed = 5 →
  passing_time = 6 →
  ∃ (train_length : ℝ), 
    109.98 < train_length ∧ train_length < 110 :=
by sorry


end train_length_l3537_353745


namespace sam_has_148_balls_l3537_353762

-- Define the number of tennis balls for each person
def lily_balls : ℕ := 84

-- Define Frodo's tennis balls in terms of Lily's
def frodo_balls : ℕ := (lily_balls * 135 + 50) / 100

-- Define Brian's tennis balls in terms of Frodo's
def brian_balls : ℕ := (frodo_balls * 35 + 5) / 10

-- Define Sam's tennis balls
def sam_balls : ℕ := ((frodo_balls + lily_balls) * 3 + 2) / 4

-- Theorem statement
theorem sam_has_148_balls : sam_balls = 148 := by
  sorry

end sam_has_148_balls_l3537_353762


namespace complex_square_l3537_353730

theorem complex_square : (1 - Complex.I) ^ 2 = -2 * Complex.I := by
  sorry

end complex_square_l3537_353730


namespace opponent_total_score_l3537_353788

/-- Represents the score of a basketball game -/
structure GameScore where
  team : ℕ
  opponent : ℕ

/-- Calculates the total opponent score given a list of game scores -/
def totalOpponentScore (games : List GameScore) : ℕ :=
  games.foldr (fun game acc => game.opponent + acc) 0

theorem opponent_total_score : 
  ∃ (games : List GameScore),
    games.length = 12 ∧ 
    (∀ g ∈ games, 1 ≤ g.team ∧ g.team ≤ 12) ∧
    (games.filter (fun g => g.opponent = g.team + 2)).length = 6 ∧
    (∀ g ∈ games.filter (fun g => g.opponent ≠ g.team + 2), g.team = 3 * g.opponent) ∧
    totalOpponentScore games = 50 := by
  sorry


end opponent_total_score_l3537_353788


namespace xibing_purchase_problem_l3537_353772

/-- Xibing purchase problem -/
theorem xibing_purchase_problem 
  (initial_price : ℚ) 
  (person_a_spent : ℚ) 
  (person_b_spent : ℚ) 
  (box_difference : ℕ) 
  (price_reduction : ℚ) :
  person_a_spent = 2400 →
  person_b_spent = 3000 →
  box_difference = 10 →
  price_reduction = 20 →
  ∃ (person_a_boxes : ℕ),
    person_a_boxes = 40 ∧
    initial_price = person_a_spent / person_a_boxes ∧
    initial_price = person_b_spent / (person_a_boxes + box_difference) ∧
    (2 * person_a_spent) / (person_a_boxes + person_a_spent / (initial_price - price_reduction)) = 48 ∧
    (person_b_spent + (initial_price - price_reduction) * (person_a_boxes + box_difference)) / 
      (2 * (person_a_boxes + box_difference)) = 50 := by
  sorry

end xibing_purchase_problem_l3537_353772


namespace intersection_of_A_and_B_l3537_353728

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = -x^2 + 2*x + 2}
def B : Set ℝ := {y | ∃ x, y = 2^x - 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {y | -1 < y ∧ y ≤ 3} := by sorry

end intersection_of_A_and_B_l3537_353728


namespace quadratic_inequality_l3537_353726

/-- A quadratic function with axis of symmetry at x = 2 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_inequality (b c : ℝ) :
  (∀ x, f b c (2 - x) = f b c (2 + x)) →  -- axis of symmetry at x = 2
  (∀ x₁ x₂, x₁ < x₂ → f b c x₁ > f b c x₂ → f b c x₂ > f b c (2*x₂ - x₁)) →  -- opens upwards
  f b c 2 < f b c 1 ∧ f b c 1 < f b c 4 :=
sorry

end quadratic_inequality_l3537_353726


namespace dodecahedron_edge_probability_l3537_353750

/-- A regular dodecahedron -/
structure RegularDodecahedron :=
  (vertices : ℕ)
  (edges_per_vertex : ℕ)
  (h_vertices : vertices = 20)
  (h_edges_per_vertex : edges_per_vertex = 3)

/-- The probability of two randomly chosen vertices being connected by an edge -/
def edge_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

theorem dodecahedron_edge_probability (d : RegularDodecahedron) :
  edge_probability d = 3 / 19 :=
by sorry

end dodecahedron_edge_probability_l3537_353750


namespace line_inclination_theorem_l3537_353780

/-- Given a line ax + by + c = 0 with inclination angle α, and sin α + cos α = 0, then a - b = 0 -/
theorem line_inclination_theorem (a b c : ℝ) (α : ℝ) : 
  (∃ x y, a * x + b * y + c = 0) →  -- line exists
  (Real.tan α = -a / b) →           -- definition of inclination angle
  (Real.sin α + Real.cos α = 0) →   -- given condition
  a - b = 0 := by
sorry

end line_inclination_theorem_l3537_353780


namespace license_plate_count_l3537_353704

/-- The number of consonants in the English alphabet -/
def num_consonants : ℕ := 20

/-- The number of vowels in the English alphabet (including Y) -/
def num_vowels : ℕ := 6

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_plates : ℕ := num_consonants * num_vowels * num_consonants * num_digits

theorem license_plate_count : total_plates = 24000 := by
  sorry

end license_plate_count_l3537_353704


namespace arithmetic_operations_l3537_353725

theorem arithmetic_operations :
  ((-3) + (-1) = -4) ∧
  (0 - 11 = -11) ∧
  (97 - (-3) = 100) ∧
  ((-7) * 5 = -35) ∧
  ((-8) / (-1/4) = 32) ∧
  ((-2/3)^3 = -8/27) := by
sorry

end arithmetic_operations_l3537_353725


namespace expand_and_simplify_product_l3537_353753

theorem expand_and_simplify_product (x : ℝ) : 
  (5 * x + 3) * (2 * x^2 - x + 4) = 10 * x^3 + x^2 + 17 * x + 12 := by
  sorry

end expand_and_simplify_product_l3537_353753


namespace sushi_lollipops_l3537_353744

theorem sushi_lollipops (x y : ℕ) : x + y = 27 :=
  by
    have h1 : x + y = 5 + (3 * 5) + 7 := by sorry
    have h2 : 5 + (3 * 5) + 7 = 27 := by sorry
    rw [h1, h2]

end sushi_lollipops_l3537_353744


namespace hyperbola_foci_distance_l3537_353746

/-- The distance between the foci of a hyperbola given by the equation 9x^2 - 27x - 16y^2 - 32y = 72 -/
theorem hyperbola_foci_distance : 
  let equation := fun (x y : ℝ) => 9 * x^2 - 27 * x - 16 * y^2 - 32 * y - 72
  ∃ c : ℝ, c > 0 ∧ 
    (∀ x y : ℝ, equation x y = 0 → 
      ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
        ((x - 3/2)^2 / a^2) - ((y + 1)^2 / b^2) = 1 ∧
        c^2 = a^2 + b^2) ∧
    2 * c = Real.sqrt 41775 / 12 :=
sorry

end hyperbola_foci_distance_l3537_353746


namespace sum_of_absolute_coefficients_l3537_353797

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (2*x - 1)^6 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 729 := by
  sorry

end sum_of_absolute_coefficients_l3537_353797


namespace system_solution_iff_m_neq_one_l3537_353738

/-- The system of equations has at least one solution if and only if m ≠ 1 -/
theorem system_solution_iff_m_neq_one (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 1 := by
  sorry

end system_solution_iff_m_neq_one_l3537_353738


namespace exists_efficient_coin_ordering_strategy_l3537_353791

/-- A strategy for ordering coins by weight using a balance scale. -/
structure CoinOrderingStrategy where
  /-- The number of coins to be ordered -/
  num_coins : Nat
  /-- The expected number of weighings required by the strategy -/
  expected_weighings : ℝ

/-- A weighing action compares two coins and determines which is heavier -/
def weighing_action (coin1 coin2 : Nat) : Bool := sorry

/-- Theorem stating that there exists a strategy for ordering 4 coins with expected weighings < 4.8 -/
theorem exists_efficient_coin_ordering_strategy :
  ∃ (strategy : CoinOrderingStrategy),
    strategy.num_coins = 4 ∧
    strategy.expected_weighings < 4.8 := by sorry

end exists_efficient_coin_ordering_strategy_l3537_353791


namespace total_accidents_l3537_353792

/-- Represents the accident rate for a highway -/
structure AccidentRate where
  accidents : ℕ
  vehicles : ℕ

/-- Calculates the number of accidents for a given traffic volume -/
def calculateAccidents (rate : AccidentRate) (traffic : ℕ) : ℕ :=
  (rate.accidents * traffic + rate.vehicles - 1) / rate.vehicles

theorem total_accidents (highwayA_rate : AccidentRate) (highwayB_rate : AccidentRate) (highwayC_rate : AccidentRate)
  (highwayA_traffic : ℕ) (highwayB_traffic : ℕ) (highwayC_traffic : ℕ) :
  highwayA_rate = ⟨200, 100000000⟩ →
  highwayB_rate = ⟨150, 50000000⟩ →
  highwayC_rate = ⟨100, 150000000⟩ →
  highwayA_traffic = 2000000000 →
  highwayB_traffic = 1500000000 →
  highwayC_traffic = 2500000000 →
  calculateAccidents highwayA_rate highwayA_traffic +
  calculateAccidents highwayB_rate highwayB_traffic +
  calculateAccidents highwayC_rate highwayC_traffic = 10168 := by
  sorry

end total_accidents_l3537_353792


namespace fraction_problem_l3537_353710

theorem fraction_problem : 
  ∃ (x y : ℚ), x / y > 0 ∧ y ≠ 0 ∧ ((377 / 13) / 29) * (x / y) / 2 = 1 / 8 ∧ x / y = 1 / 4 :=
by sorry

end fraction_problem_l3537_353710


namespace inequality_order_l3537_353778

theorem inequality_order (a b : ℝ) (ha : a = 6) (hb : b = 3) :
  (a + 3*b) / 4 < (a^2 * b)^(1/3) ∧ (a^2 * b)^(1/3) < (a + 3*b)^2 / (4*(a + b)) := by
  sorry

end inequality_order_l3537_353778


namespace sum_of_factors_40_l3537_353736

theorem sum_of_factors_40 : (Finset.filter (· ∣ 40) (Finset.range 41)).sum id = 90 := by
  sorry

end sum_of_factors_40_l3537_353736
