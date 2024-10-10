import Mathlib

namespace count_a_values_correct_l1505_150578

/-- The number of integer values of a for which (a-1)x^2 + 2x - a - 1 = 0 has integer roots for x -/
def count_a_values : ℕ := 5

/-- The equation has integer roots for x -/
def has_integer_roots (a : ℤ) : Prop :=
  ∃ x : ℤ, (a - 1) * x^2 + 2 * x - a - 1 = 0

/-- There are exactly 5 integer values of a for which the equation has integer roots -/
theorem count_a_values_correct :
  (∃ S : Finset ℤ, S.card = count_a_values ∧ 
    (∀ a : ℤ, a ∈ S ↔ has_integer_roots a)) ∧
  (∀ T : Finset ℤ, (∀ a : ℤ, a ∈ T ↔ has_integer_roots a) → T.card ≤ count_a_values) :=
sorry

end count_a_values_correct_l1505_150578


namespace grasshopper_impossibility_l1505_150538

/-- A point in the 2D plane with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Check if a move from p1 to p2 is parallel to the line segment from p3 to p4 -/
def parallel_move (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x) * (p4.y - p3.y) = (p2.y - p1.y) * (p4.x - p3.x)

/-- A valid move in the grasshopper game -/
inductive ValidMove : List Point → List Point → Prop where
  | move (p1 p2 p3 p1' : Point) (rest : List Point) :
      parallel_move p1 p1' p2 p3 →
      ValidMove [p1, p2, p3] (p1' :: rest)

/-- A sequence of valid moves -/
def ValidMoveSequence : List Point → List Point → Prop :=
  Relation.ReflTransGen ValidMove

/-- The main theorem: impossibility of reaching the final configuration -/
theorem grasshopper_impossibility :
  ¬∃ (final : List Point),
    ValidMoveSequence [Point.mk 1 0, Point.mk 0 0, Point.mk 0 1] final ∧
    final = [Point.mk 0 0, Point.mk (-1) (-1), Point.mk 1 1] :=
sorry


end grasshopper_impossibility_l1505_150538


namespace sum_of_roots_quadratic_l1505_150518

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 3*x₁ - 4 = 0) → (x₂^2 - 3*x₂ - 4 = 0) → (x₁ + x₂ = 3) :=
by sorry

end sum_of_roots_quadratic_l1505_150518


namespace experienced_sailors_monthly_earnings_l1505_150590

theorem experienced_sailors_monthly_earnings :
  let total_sailors : ℕ := 17
  let inexperienced_sailors : ℕ := 5
  let experienced_sailors : ℕ := total_sailors - inexperienced_sailors
  let inexperienced_hourly_wage : ℚ := 10
  let wage_increase_ratio : ℚ := 1 / 5
  let experienced_hourly_wage : ℚ := inexperienced_hourly_wage * (1 + wage_increase_ratio)
  let weekly_hours : ℕ := 60
  let weeks_per_month : ℕ := 4
  
  experienced_sailors * experienced_hourly_wage * weekly_hours * weeks_per_month = 34560 :=
by sorry

end experienced_sailors_monthly_earnings_l1505_150590


namespace lines_intersect_l1505_150586

/-- Two lines in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Define when two lines are intersecting -/
def are_intersecting (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a

/-- The problem statement -/
theorem lines_intersect : 
  let line1 : Line2D := ⟨3, -2, 5⟩
  let line2 : Line2D := ⟨1, 3, 10⟩
  are_intersecting line1 line2 := by
  sorry

end lines_intersect_l1505_150586


namespace cone_volume_from_cylinder_l1505_150539

/-- Given a cylinder with volume 72π cm³, prove that a cone with the same radius
    and half the height has a volume of 12π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  π * r^2 * h = 72 * π →
  (1/3) * π * r^2 * (h/2) = 12 * π := by
sorry

end cone_volume_from_cylinder_l1505_150539


namespace geometric_sequence_common_ratio_l1505_150594

theorem geometric_sequence_common_ratio 
  (a₁ : ℝ) (r : ℝ) 
  (h₁ : a₁ ≠ 0) 
  (h₂ : r > 0) 
  (h₃ : ∀ n m : ℕ, n ≠ m → a₁ * r^n ≠ a₁ * r^m) 
  (h₄ : ∃ d : ℝ, a₁ * r^3 - a₁ * r = d ∧ a₁ * r^4 - a₁ * r^3 = d) : 
  r = (1 + Real.sqrt 5) / 2 := by
sorry

end geometric_sequence_common_ratio_l1505_150594


namespace farm_animals_problem_l1505_150564

theorem farm_animals_problem :
  ∃! (s c : ℕ), s > 0 ∧ c > 0 ∧ 28 * s + 27 * c = 1200 ∧ c > s :=
by sorry

end farm_animals_problem_l1505_150564


namespace expression_value_l1505_150577

theorem expression_value (m n : ℤ) (h : m - n = 2) : 2*m^2 - 4*m*n + 2*n^2 - 1 = 7 := by
  sorry

end expression_value_l1505_150577


namespace binomial_coefficient_divisible_by_prime_l1505_150529

theorem binomial_coefficient_divisible_by_prime (p k : ℕ) :
  Nat.Prime p → 1 ≤ k → k < p →
  ∃ m : ℕ, Nat.choose p k = m * p := by
  sorry

end binomial_coefficient_divisible_by_prime_l1505_150529


namespace shopping_with_refund_l1505_150504

/-- Calculates the remaining money after shopping with a partial refund --/
theorem shopping_with_refund 
  (initial_amount : ℕ) 
  (sweater_cost t_shirt_cost shoes_cost : ℕ) 
  (refund_percentage : ℚ) : 
  initial_amount = 74 →
  sweater_cost = 9 →
  t_shirt_cost = 11 →
  shoes_cost = 30 →
  refund_percentage = 90 / 100 →
  initial_amount - (sweater_cost + t_shirt_cost + (shoes_cost * (1 - refund_percentage))) = 51 := by
  sorry

end shopping_with_refund_l1505_150504


namespace triangle_inequality_four_points_l1505_150579

-- Define a metric space
variable {X : Type*} [MetricSpace X]

-- Define four points in the metric space
variable (A B C D : X)

-- State the theorem
theorem triangle_inequality_four_points :
  dist A D ≤ dist A B + dist B C + dist C D := by
  sorry

end triangle_inequality_four_points_l1505_150579


namespace sufficient_not_necessary_condition_l1505_150505

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) :=
by sorry

end sufficient_not_necessary_condition_l1505_150505


namespace shirt_store_profit_optimization_l1505_150533

/-- Represents the daily profit function for a shirt store -/
def daily_profit (x : ℝ) : ℝ := (20 + 2*x) * (40 - x)

/-- Represents the price reduction that achieves a specific daily profit -/
def price_reduction_for_profit (target_profit : ℝ) : ℝ :=
  20 -- The actual value should be solved from the equation, but we're using the known result

/-- Represents the price reduction that maximizes daily profit -/
def optimal_price_reduction : ℝ := 15

/-- The maximum daily profit achieved at the optimal price reduction -/
def max_daily_profit : ℝ := 1250

theorem shirt_store_profit_optimization :
  (daily_profit (price_reduction_for_profit 1200) = 1200) ∧
  (∀ x : ℝ, daily_profit x ≤ max_daily_profit) ∧
  (daily_profit optimal_price_reduction = max_daily_profit) := by
  sorry


end shirt_store_profit_optimization_l1505_150533


namespace range_of_M_M_lower_bound_l1505_150550

theorem range_of_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) :
  let M := (1/a - 1) * (1/b - 1) * (1/c - 1)
  ∀ x : ℝ, x ≥ 8 → ∃ a' b' c' : ℝ, 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a' + b' + c' = 1 ∧
    (1/a' - 1) * (1/b' - 1) * (1/c' - 1) = x :=
by sorry

theorem M_lower_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) :
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 :=
by sorry

end range_of_M_M_lower_bound_l1505_150550


namespace largest_multiple_of_eight_less_than_neg_63_l1505_150598

theorem largest_multiple_of_eight_less_than_neg_63 :
  ∀ n : ℤ, n * 8 < -63 → n * 8 ≤ -64 :=
by
  sorry

end largest_multiple_of_eight_less_than_neg_63_l1505_150598


namespace lcm_of_prime_and_nondivisor_l1505_150561

theorem lcm_of_prime_and_nondivisor (p n : ℕ) (hp : Nat.Prime p) (hn : ¬(n ∣ p)) :
  Nat.lcm p n = p * n :=
by sorry

end lcm_of_prime_and_nondivisor_l1505_150561


namespace sum_of_five_consecutive_even_integers_l1505_150527

theorem sum_of_five_consecutive_even_integers (m : ℤ) (h : Even m) :
  m + (m + 2) + (m + 4) + (m + 6) + (m + 8) = 5 * m + 20 := by
  sorry

end sum_of_five_consecutive_even_integers_l1505_150527


namespace periodic_sin_and_empty_subset_l1505_150596

-- Define a periodic function
def isPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = f x

-- Define the sine function
noncomputable def sin : ℝ → ℝ := Real.sin

-- Define set A
variable (A : Set ℝ)

-- Theorem statement
theorem periodic_sin_and_empty_subset (A : Set ℝ) : 
  (isPeriodic sin) ∧ (∅ ⊆ A) := by sorry

end periodic_sin_and_empty_subset_l1505_150596


namespace blown_out_sand_dunes_with_treasure_l1505_150566

theorem blown_out_sand_dunes_with_treasure :
  let sand_dunes_remain_prob : ℚ := 1 / 3
  let sand_dunes_with_coupon_prob : ℚ := 2 / 3
  let total_blown_out_dunes : ℕ := 5
  let both_treasure_and_coupon_prob : ℚ := 8 / 90
  ∃ (treasure_dunes : ℕ),
    (treasure_dunes : ℚ) / total_blown_out_dunes * sand_dunes_with_coupon_prob = both_treasure_and_coupon_prob ∧
    treasure_dunes = 20 :=
by sorry

end blown_out_sand_dunes_with_treasure_l1505_150566


namespace chip_exits_at_A2_l1505_150584

-- Define the grid size
def gridSize : Nat := 4

-- Define the possible directions
inductive Direction
| Up
| Down
| Left
| Right

-- Define a cell position
structure Position where
  row : Nat
  col : Nat

-- Define the state of the game
structure GameState where
  chipPosition : Position
  arrows : Array (Array Direction)

-- Define the initial state
def initialState : GameState := sorry

-- Define a function to get the next position based on current position and direction
def nextPosition (pos : Position) (dir : Direction) : Position := sorry

-- Define a function to flip the direction
def flipDirection (dir : Direction) : Direction := sorry

-- Define a function to make a move
def makeMove (state : GameState) : GameState := sorry

-- Define a function to check if a position is out of bounds
def isOutOfBounds (pos : Position) : Bool := sorry

-- Define a function to simulate the game until the chip exits
def simulateUntilExit (state : GameState) : Position := sorry

-- The main theorem to prove
theorem chip_exits_at_A2 :
  let finalPos := simulateUntilExit initialState
  finalPos = Position.mk 0 1 := sorry

end chip_exits_at_A2_l1505_150584


namespace partial_fraction_decomposition_l1505_150547

theorem partial_fraction_decomposition (x A B C : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 4) :
  (6 * x) / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2 ↔ 
  A = 3 ∧ B = -3 ∧ C = -6 := by
  sorry

end partial_fraction_decomposition_l1505_150547


namespace propositions_truth_l1505_150591

theorem propositions_truth : 
  (∀ x : ℝ, x^2 - x + 1 ≥ 0) ∧ 
  (∃ x : ℝ, x^2 - 4 = 0) := by
  sorry

end propositions_truth_l1505_150591


namespace sum_and_cube_sum_divisibility_l1505_150532

theorem sum_and_cube_sum_divisibility (x y : ℤ) :
  (6 ∣ (x + y)) ↔ (6 ∣ (x^3 + y^3)) := by sorry

end sum_and_cube_sum_divisibility_l1505_150532


namespace polar_coords_of_negative_two_plus_two_i_l1505_150548

/-- The polar coordinates of a complex number z = -(2+2i) -/
theorem polar_coords_of_negative_two_plus_two_i :
  ∃ (r : ℝ) (θ : ℝ) (k : ℤ),
    r = 2 * Real.sqrt 2 ∧
    θ = 5 * Real.pi / 4 + 2 * k * Real.pi ∧
    Complex.exp (θ * Complex.I) * r = -(2 + 2 * Complex.I) :=
by sorry

end polar_coords_of_negative_two_plus_two_i_l1505_150548


namespace max_value_quadratic_l1505_150510

/-- The maximum value of y = -x^2 + 4x + 3, where x is a real number, is 7. -/
theorem max_value_quadratic :
  ∃ (y_max : ℝ), y_max = 7 ∧ ∀ (x : ℝ), -x^2 + 4*x + 3 ≤ y_max :=
sorry

end max_value_quadratic_l1505_150510


namespace triangle_inequality_l1505_150514

/-- Proves that for any triangle with side lengths a, b, c, and area S,
    the inequality (ab + ac + bc) / (4S) ≥ √3 holds true. -/
theorem triangle_inequality (a b c S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ S > 0) 
    (h_triangle : S = 1/4 * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))) :
  (a * b + a * c + b * c) / (4 * S) ≥ Real.sqrt 3 := by
  sorry

end triangle_inequality_l1505_150514


namespace ribbon_cutting_l1505_150544

-- Define the lengths of the two ribbons
def ribbon1_length : ℕ := 28
def ribbon2_length : ℕ := 16

-- Define the function to calculate the maximum length of shorter ribbons
def max_short_ribbon_length (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the function to calculate the total number of shorter ribbons
def total_short_ribbons (a b c : ℕ) : ℕ := (a + b) / c

-- Theorem statement
theorem ribbon_cutting :
  (max_short_ribbon_length ribbon1_length ribbon2_length = 4) ∧
  (total_short_ribbons ribbon1_length ribbon2_length (max_short_ribbon_length ribbon1_length ribbon2_length) = 11) :=
by sorry

end ribbon_cutting_l1505_150544


namespace abc_inequality_l1505_150588

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 / (1 + a^2) + b^2 / (1 + b^2) + c^2 / (1 + c^2) = 1) :
  a * b * c ≤ Real.sqrt 2 / 4 := by
  sorry

end abc_inequality_l1505_150588


namespace function_inequality_l1505_150589

-- Define the function f on the non-zero real numbers
variable (f : ℝ → ℝ)

-- Define the condition that f is twice differentiable
variable (hf : TwiceDifferentiable ℝ f)

-- Define the condition that f''(x) - f(x)/x > 0 for all non-zero x
variable (h : ∀ x : ℝ, x ≠ 0 → (deriv^[2] f) x - f x / x > 0)

-- State the theorem
theorem function_inequality : 3 * f 4 > 4 * f 3 := by sorry

end function_inequality_l1505_150589


namespace john_star_wars_spending_l1505_150595

/-- Calculates the total money spent on Star Wars toys --/
def total_spent (group_a_cost group_b_cost : ℝ) 
                (group_a_discount group_b_discount : ℝ) 
                (group_a_tax group_b_tax lightsaber_tax : ℝ) : ℝ :=
  let group_a_discounted := group_a_cost * (1 - group_a_discount)
  let group_b_discounted := group_b_cost * (1 - group_b_discount)
  let group_a_total := group_a_discounted * (1 + group_a_tax)
  let group_b_total := group_b_discounted * (1 + group_b_tax)
  let other_toys_total := group_a_total + group_b_total
  let lightsaber_cost := 2 * other_toys_total
  let lightsaber_total := lightsaber_cost * (1 + lightsaber_tax)
  other_toys_total + lightsaber_total

/-- The total amount John spent on Star Wars toys is $4008.312 --/
theorem john_star_wars_spending :
  total_spent 900 600 0.15 0.25 0.06 0.09 0.04 = 4008.312 := by
  sorry

end john_star_wars_spending_l1505_150595


namespace line_outside_plane_iff_at_most_one_point_l1505_150559

-- Define the basic types
variable (L : Type*) -- Type for lines
variable (P : Type*) -- Type for planes

-- Define the relationships between lines and planes
variable (parallel : L → P → Prop)
variable (intersects : L → P → Prop)
variable (within : L → P → Prop)
variable (outside : L → P → Prop)

-- Define the number of common points
variable (common_points : L → P → ℕ)

-- Theorem statement
theorem line_outside_plane_iff_at_most_one_point 
  (l : L) (p : P) : 
  outside l p ↔ common_points l p ≤ 1 := by sorry

end line_outside_plane_iff_at_most_one_point_l1505_150559


namespace right_triangle_third_side_l1505_150560

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a = 6 → b = 8 →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c = 10 ∨ c = 2 * Real.sqrt 7 := by
  sorry

end right_triangle_third_side_l1505_150560


namespace unique_solution_in_interval_l1505_150517

theorem unique_solution_in_interval (x : ℝ) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  ((2 - Real.sin (2 * x)) * Real.sin (x + Real.pi / 4) = 1) ↔
  (x = Real.pi / 4) := by
  sorry

end unique_solution_in_interval_l1505_150517


namespace uncle_ben_farm_l1505_150568

def farm_problem (total_chickens : ℕ) (non_laying_hens : ℕ) (eggs_per_hen : ℕ) (total_eggs : ℕ) : Prop :=
  ∃ (roosters hens : ℕ),
    roosters + hens = total_chickens ∧
    3 * (hens - non_laying_hens) = total_eggs ∧
    roosters = 39

theorem uncle_ben_farm :
  farm_problem 440 15 3 1158 :=
sorry

end uncle_ben_farm_l1505_150568


namespace all_days_happy_l1505_150534

theorem all_days_happy (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end all_days_happy_l1505_150534


namespace existence_implies_range_l1505_150509

theorem existence_implies_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ Real.exp x * (x - a) < 1) → a > -1 := by
  sorry

end existence_implies_range_l1505_150509


namespace sqrt_equality_implies_t_value_l1505_150592

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, (Real.sqrt (3 * Real.sqrt (t - 3)) = (8 - t) ^ (1/4)) → t = 7/2 := by
  sorry

end sqrt_equality_implies_t_value_l1505_150592


namespace investment_rate_calculation_l1505_150507

theorem investment_rate_calculation (total_investment : ℝ) (invested_at_18_percent : ℝ) (total_interest : ℝ) :
  total_investment = 22000 →
  invested_at_18_percent = 7000 →
  total_interest = 3360 →
  let remaining_investment := total_investment - invested_at_18_percent
  let interest_from_18_percent := invested_at_18_percent * 0.18
  let remaining_interest := total_interest - interest_from_18_percent
  let unknown_rate := remaining_interest / remaining_investment
  unknown_rate = 0.14 := by sorry

end investment_rate_calculation_l1505_150507


namespace geometric_sequence_ratio_2018_2013_l1505_150597

/-- A geometric sequence with common ratio q > 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio_2018_2013 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : GeometricSequence a q)
  (h_sum : a 1 + a 6 = 8)
  (h_prod : a 3 * a 4 = 12) :
  a 2018 / a 2013 = 3 := by
sorry

end geometric_sequence_ratio_2018_2013_l1505_150597


namespace inverse_proportion_l1505_150536

theorem inverse_proportion (x₁ x₂ y₁ y₂ : ℝ) (h1 : x₁ ≠ 0) (h2 : x₂ ≠ 0) (h3 : y₁ ≠ 0) (h4 : y₂ ≠ 0)
  (h5 : ∃ k : ℝ, ∀ x y : ℝ, x * y = k) (h6 : x₁ / x₂ = 4 / 5) :
  y₁ / y₂ = 5 / 4 := by
sorry

end inverse_proportion_l1505_150536


namespace tan_105_degrees_l1505_150556

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l1505_150556


namespace translation_preserves_vector_translation_problem_l1505_150530

/-- A translation in 2D space -/
structure Translation (α : Type*) [Add α] :=
  (dx dy : α)

/-- Apply a translation to a point -/
def apply_translation {α : Type*} [Add α] (t : Translation α) (p : α × α) : α × α :=
  (p.1 + t.dx, p.2 + t.dy)

theorem translation_preserves_vector {α : Type*} [AddCommGroup α] 
  (t : Translation α) (a b c d : α × α) :
  apply_translation t a = c →
  apply_translation t b = d →
  c.1 - a.1 = d.1 - b.1 ∧ c.2 - a.2 = d.2 - b.2 :=
sorry

/-- The main theorem to prove -/
theorem translation_problem :
  ∃ (t : Translation ℤ),
    apply_translation t (-1, 4) = (3, 6) ∧
    apply_translation t (-3, 2) = (1, 4) :=
sorry

end translation_preserves_vector_translation_problem_l1505_150530


namespace max_value_on_unit_circle_l1505_150585

theorem max_value_on_unit_circle (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (M : ℝ), M = 7 ∧ ∀ (a b : ℝ), a^2 + b^2 = 1 → a^2 + 4*b + 3 ≤ M :=
by sorry

end max_value_on_unit_circle_l1505_150585


namespace parabola_chord_intersection_l1505_150522

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a parabola in the form y^2 = 2px -/
structure Parabola where
  p : ℝ

def Parabola.contains (p : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * p.p * pt.x

def Line.contains (l : Line) (pt : Point) : Prop :=
  pt.y = l.m * pt.x + l.b

def perpendicular (l1 l2 : Line) : Prop :=
  l1.m * l2.m = -1

theorem parabola_chord_intersection (p : Parabola) (m : Point) (d e : Point) :
  p.p = 2 →
  p.contains m →
  m.y = 4 →
  ∃ (l_md l_me l_de : Line),
    l_md.contains m ∧ l_md.contains d ∧
    l_me.contains m ∧ l_me.contains e ∧
    l_de.contains d ∧ l_de.contains e ∧
    perpendicular l_md l_me →
    l_de.contains (Point.mk 8 (-4)) := by
  sorry

end parabola_chord_intersection_l1505_150522


namespace sarah_savings_l1505_150543

def savings_schedule : List ℕ := [5, 5, 5, 5, 10, 10, 10, 10, 20, 20, 20, 20]

theorem sarah_savings : (savings_schedule.sum = 140) := by
  sorry

end sarah_savings_l1505_150543


namespace divisibility_of_power_difference_l1505_150581

theorem divisibility_of_power_difference (a b c k q : ℕ) (n : ℤ) :
  a ≥ 1 →
  b ≥ 1 →
  c ≥ 1 →
  k ≥ 1 →
  n = a^(c^k) - b^(c^k) →
  (∃ (p : List ℕ), (∀ x ∈ p, Nat.Prime x) ∧ p.length ≥ q ∧ (∀ x ∈ p, c % x = 0)) →
  ∃ (r : List ℕ), (∀ x ∈ r, Nat.Prime x) ∧ r.length ≥ q * k ∧ (∀ x ∈ r, n % x = 0) :=
by sorry

end divisibility_of_power_difference_l1505_150581


namespace total_hamburgers_calculation_l1505_150571

/-- Calculates the total number of hamburgers bought given the total amount spent,
    costs of single and double burgers, and the number of double burgers bought. -/
theorem total_hamburgers_calculation 
  (total_spent : ℚ)
  (single_burger_cost : ℚ)
  (double_burger_cost : ℚ)
  (double_burgers_bought : ℕ)
  (h1 : total_spent = 66.5)
  (h2 : single_burger_cost = 1)
  (h3 : double_burger_cost = 1.5)
  (h4 : double_burgers_bought = 33) :
  ∃ (single_burgers_bought : ℕ),
    single_burgers_bought + double_burgers_bought = 50 ∧
    total_spent = single_burger_cost * single_burgers_bought + double_burger_cost * double_burgers_bought :=
by sorry


end total_hamburgers_calculation_l1505_150571


namespace factorization_proof_l1505_150503

theorem factorization_proof (x : ℝ) : 221 * x^2 + 68 * x + 17 = 17 * (13 * x^2 + 4 * x + 1) := by
  sorry

end factorization_proof_l1505_150503


namespace base9_4318_equals_3176_l1505_150593

/-- Converts a base-9 digit to its decimal (base-10) value. -/
def base9ToDecimal (digit : ℕ) : ℕ := digit

/-- Converts a base-9 number to its decimal (base-10) equivalent. -/
def convertBase9ToDecimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ (digits.length - 1 - i))) 0

theorem base9_4318_equals_3176 :
  convertBase9ToDecimal [4, 3, 1, 8] = 3176 := by
  sorry

#eval convertBase9ToDecimal [4, 3, 1, 8]

end base9_4318_equals_3176_l1505_150593


namespace course_selection_proof_l1505_150554

def total_courses : ℕ := 9
def courses_to_choose : ℕ := 4
def conflicting_courses : ℕ := 3
def other_courses : ℕ := total_courses - conflicting_courses

def selection_schemes : ℕ := 
  (conflicting_courses.choose 1 * other_courses.choose (courses_to_choose - 1)) +
  (other_courses.choose courses_to_choose)

theorem course_selection_proof : selection_schemes = 75 := by
  sorry

end course_selection_proof_l1505_150554


namespace janet_lives_l1505_150557

theorem janet_lives (initial_lives lost_lives gained_lives : ℕ) :
  initial_lives ≥ lost_lives →
  initial_lives - lost_lives + gained_lives =
    initial_lives + gained_lives - lost_lives :=
by
  sorry

#check janet_lives 38 16 32

end janet_lives_l1505_150557


namespace smallest_n_for_polygon_cuts_l1505_150516

theorem smallest_n_for_polygon_cuts : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 → (m - 2) % 31 = 0 ∧ (m - 2) % 65 = 0 → n ≤ m) ∧
  (n - 2) % 31 = 0 ∧ 
  (n - 2) % 65 = 0 ∧ 
  n = 2017 := by
  sorry

end smallest_n_for_polygon_cuts_l1505_150516


namespace hike_attendance_l1505_150535

/-- The number of cars used for the hike -/
def num_cars : ℕ := 7

/-- The number of people in each car -/
def people_per_car : ℕ := 4

/-- The number of taxis used for the hike -/
def num_taxis : ℕ := 10

/-- The number of people in each taxi -/
def people_per_taxi : ℕ := 6

/-- The number of vans used for the hike -/
def num_vans : ℕ := 4

/-- The number of people in each van -/
def people_per_van : ℕ := 5

/-- The number of buses used for the hike -/
def num_buses : ℕ := 3

/-- The number of people in each bus -/
def people_per_bus : ℕ := 20

/-- The number of minibuses used for the hike -/
def num_minibuses : ℕ := 2

/-- The number of people in each minibus -/
def people_per_minibus : ℕ := 8

/-- The total number of people who went on the hike -/
def total_people : ℕ := 
  num_cars * people_per_car + 
  num_taxis * people_per_taxi + 
  num_vans * people_per_van + 
  num_buses * people_per_bus + 
  num_minibuses * people_per_minibus

theorem hike_attendance : total_people = 184 := by
  sorry

end hike_attendance_l1505_150535


namespace square_equality_necessary_not_sufficient_l1505_150569

theorem square_equality_necessary_not_sufficient :
  (∀ x y : ℝ, x = y → x^2 = y^2) ∧
  ¬(∀ x y : ℝ, x^2 = y^2 → x = y) :=
by sorry

end square_equality_necessary_not_sufficient_l1505_150569


namespace inequality_proof_l1505_150587

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_condition : a * b + b * c + c * a ≤ 1) :
  a + b + c + Real.sqrt 3 ≥ 8 * a * b * c * (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) :=
by sorry

end inequality_proof_l1505_150587


namespace basketball_winning_percentage_l1505_150573

theorem basketball_winning_percentage (total_games season_games remaining_games first_wins : ℕ)
  (h1 : total_games = season_games + remaining_games)
  (h2 : season_games = 75)
  (h3 : remaining_games = 45)
  (h4 : first_wins = 60)
  (h5 : total_games = 120) :
  (∃ x : ℕ, x = 36 ∧ (first_wins + x : ℚ) / total_games = 4/5) :=
sorry

end basketball_winning_percentage_l1505_150573


namespace units_digit_problem_l1505_150567

theorem units_digit_problem : (8 * 18 * 1988 - 8^3) % 10 = 0 := by
  sorry

end units_digit_problem_l1505_150567


namespace second_chapter_longer_l1505_150572

/-- A book with two chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ

/-- The difference in pages between two chapters -/
def page_difference (b : Book) : ℕ := b.chapter2_pages - b.chapter1_pages

theorem second_chapter_longer (b : Book) 
  (h1 : b.chapter1_pages = 37) 
  (h2 : b.chapter2_pages = 80) : 
  page_difference b = 43 := by
  sorry

end second_chapter_longer_l1505_150572


namespace all_positive_l1505_150502

theorem all_positive (a b c : ℝ) 
  (sum_pos : a + b + c > 0) 
  (sum_prod_pos : a * b + b * c + c * a > 0) 
  (prod_pos : a * b * c > 0) : 
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end all_positive_l1505_150502


namespace complement_of_A_l1505_150553

def A : Set ℝ := {x | x ≥ 3} ∪ {x | x < -1}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | -1 ≤ x ∧ x < 3} := by sorry

end complement_of_A_l1505_150553


namespace gcd_repeated_digit_numbers_l1505_150552

/-- A six-digit integer formed by repeating a positive three-digit integer -/
def repeatedDigitNumber (m : ℕ) : Prop :=
  100 ≤ m ∧ m < 1000 ∧ ∃ n : ℕ, n = 1001 * m

/-- The greatest common divisor of all six-digit integers formed by repeating a positive three-digit integer is 1001 -/
theorem gcd_repeated_digit_numbers :
  ∃ d : ℕ, d > 0 ∧ (∀ n : ℕ, repeatedDigitNumber n → d ∣ n) ∧
  ∀ k : ℕ, k > 0 → (∀ n : ℕ, repeatedDigitNumber n → k ∣ n) → k ∣ d :=
by sorry

end gcd_repeated_digit_numbers_l1505_150552


namespace krista_savings_exceed_target_l1505_150524

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (r^n - 1) / (r - 1)

/-- The first day Krista deposits money -/
def initialDeposit : ℚ := 3

/-- The ratio by which Krista increases her deposit each day -/
def depositRatio : ℚ := 3

/-- The amount Krista wants to exceed in cents -/
def targetAmount : ℚ := 2000

theorem krista_savings_exceed_target :
  (∀ k < 7, geometricSum initialDeposit depositRatio k ≤ targetAmount) ∧
  geometricSum initialDeposit depositRatio 7 > targetAmount :=
sorry

end krista_savings_exceed_target_l1505_150524


namespace max_cos_product_l1505_150506

theorem max_cos_product (α β γ : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) (h3 : 0 < γ ∧ γ < π/2) 
  (h4 : Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1) :
  Real.cos α * Real.cos β * Real.cos γ ≤ 2 * Real.sqrt 6 / 9 := by
  sorry

end max_cos_product_l1505_150506


namespace increasing_interval_of_sine_l1505_150540

theorem increasing_interval_of_sine (f : ℝ → ℝ) (h : f = λ x => Real.sin (2 * x + π / 6)) :
  ∀ x ∈ Set.Icc (-π / 3) (π / 6), ∀ y ∈ Set.Icc (-π / 3) (π / 6),
    x < y → f x < f y :=
by sorry

end increasing_interval_of_sine_l1505_150540


namespace smallest_number_of_ducks_l1505_150574

/-- Represents the number of birds in a flock for each type --/
structure FlockSize where
  duck : ℕ
  crane : ℕ
  heron : ℕ

/-- Represents the number of flocks for each type of bird --/
structure FlockCount where
  duck : ℕ
  crane : ℕ
  heron : ℕ

/-- The main theorem stating the smallest number of ducks observed --/
theorem smallest_number_of_ducks 
  (flock_size : FlockSize)
  (flock_count : FlockCount)
  (h1 : flock_size.duck = 13)
  (h2 : flock_size.crane = 17)
  (h3 : flock_size.heron = 11)
  (h4 : flock_size.duck * flock_count.duck = flock_size.crane * flock_count.crane)
  (h5 : 6 * (flock_size.duck * flock_count.duck) = 5 * (flock_size.heron * flock_count.heron))
  (h6 : 3 * (flock_size.crane * flock_count.crane) = 8 * (flock_size.heron * flock_count.heron))
  (h7 : ∀ c : FlockCount, 
    (c.duck < flock_count.duck ∨ c.crane < flock_count.crane ∨ c.heron < flock_count.heron) →
    (flock_size.duck * c.duck ≠ flock_size.crane * c.crane ∨
     6 * (flock_size.duck * c.duck) ≠ 5 * (flock_size.heron * c.heron) ∨
     3 * (flock_size.crane * c.crane) ≠ 8 * (flock_size.heron * c.heron))) :
  flock_size.duck * flock_count.duck = 520 := by
  sorry


end smallest_number_of_ducks_l1505_150574


namespace grasshopper_jump_distance_l1505_150570

/-- The jumping distances of animals in a contest -/
structure JumpingContest where
  mouse_jump : ℕ
  frog_jump : ℕ
  grasshopper_jump : ℕ
  mouse_frog_diff : frog_jump = mouse_jump + 12
  grasshopper_frog_diff : grasshopper_jump = frog_jump + 19

/-- Theorem: In a jumping contest where the mouse jumped 8 inches, 
    the mouse jumped 12 inches less than the frog, 
    and the grasshopper jumped 19 inches farther than the frog, 
    the grasshopper jumped 39 inches. -/
theorem grasshopper_jump_distance (contest : JumpingContest) 
  (h_mouse_jump : contest.mouse_jump = 8) : 
  contest.grasshopper_jump = 39 := by
  sorry


end grasshopper_jump_distance_l1505_150570


namespace collinear_vectors_y_value_l1505_150545

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-4, y)
  collinear a b → y = -6 := by
sorry

end collinear_vectors_y_value_l1505_150545


namespace championship_ties_l1505_150562

/-- Represents the number of points and games for a hockey team -/
structure HockeyTeam where
  wins : ℕ
  ties : ℕ
  totalPoints : ℕ

/-- Calculates the total points for a hockey team -/
def calculatePoints (team : HockeyTeam) : ℕ :=
  3 * team.wins + 2 * team.ties

theorem championship_ties (team : HockeyTeam) 
  (h1 : team.totalPoints = 85)
  (h2 : team.wins = team.ties + 15)
  (h3 : calculatePoints team = team.totalPoints) :
  team.ties = 8 := by
sorry

end championship_ties_l1505_150562


namespace problem_solution_l1505_150537

/-- f(n) denotes the nth positive integer which is not a perfect square -/
def f (n : ℕ) : ℕ := sorry

/-- Applies the function f n times -/
def iterateF (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | m + 1 => f (iterateF m x)

theorem problem_solution :
  ∃ (n : ℕ), n > 0 ∧ iterateF 2013 n = 2014^2 + 1 ∧ n = 6077248 := by
  sorry

end problem_solution_l1505_150537


namespace eliminate_x_l1505_150599

/-- Represents a linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The system of equations -/
def system : (LinearEquation × LinearEquation) :=
  ({ a := 6, b := 2, c := 4 },
   { a := 3, b := -3, c := -6 })

/-- Operation that combines two equations -/
def combineEquations (eq1 eq2 : LinearEquation) (k : ℝ) : LinearEquation :=
  { a := eq1.a - k * eq2.a,
    b := eq1.b - k * eq2.b,
    c := eq1.c - k * eq2.c }

/-- Theorem stating that the specified operation eliminates x -/
theorem eliminate_x :
  let (eq1, eq2) := system
  let result := combineEquations eq1 eq2 2
  result.a = 0 := by sorry

end eliminate_x_l1505_150599


namespace complex_sum_zero_l1505_150501

theorem complex_sum_zero : 
  let z : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I
  1 + z + z^2 = 0 := by
  sorry

end complex_sum_zero_l1505_150501


namespace expression_evaluation_l1505_150500

theorem expression_evaluation : 
  let a := 12
  let b := 14
  let c := 18
  let numerator := a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)
  let denominator := a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)
  numerator / denominator = 44 := by
sorry


end expression_evaluation_l1505_150500


namespace arithmetic_sequence_theorem_l1505_150512

def is_arithmetic_sequence (seq : Fin 4 → ℝ) : Prop :=
  ∃ (a d : ℝ), seq 0 = a - d ∧ seq 1 = a ∧ seq 2 = a + d ∧ seq 3 = a + 2*d

def sum_is_26 (seq : Fin 4 → ℝ) : Prop :=
  (seq 0) + (seq 1) + (seq 2) + (seq 3) = 26

def middle_product_is_40 (seq : Fin 4 → ℝ) : Prop :=
  (seq 1) * (seq 2) = 40

theorem arithmetic_sequence_theorem (seq : Fin 4 → ℝ) :
  is_arithmetic_sequence seq ∧ sum_is_26 seq ∧ middle_product_is_40 seq →
  (seq 0 = 2 ∧ seq 1 = 5 ∧ seq 2 = 8 ∧ seq 3 = 11) ∨
  (seq 0 = 11 ∧ seq 1 = 8 ∧ seq 2 = 5 ∧ seq 3 = 2) :=
by
  sorry

end arithmetic_sequence_theorem_l1505_150512


namespace consecutive_points_length_l1505_150563

/-- Given 5 consecutive points on a straight line, prove that ae = 21 -/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (c - b = 3 * (d - c)) →  -- bc = 3 cd
  (e - d = 8) →            -- de = 8
  (b - a = 5) →            -- ab = 5
  (c - a = 11) →           -- ac = 11
  (e - a = 21) :=          -- ae = 21
by sorry

end consecutive_points_length_l1505_150563


namespace fourth_power_of_cube_of_third_smallest_prime_l1505_150580

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem fourth_power_of_cube_of_third_smallest_prime :
  (nthSmallestPrime 3) ^ 3 ^ 4 = 244140625 := by sorry

end fourth_power_of_cube_of_third_smallest_prime_l1505_150580


namespace sugar_pack_weight_l1505_150565

/-- Given the total sugar, number of packs, and leftover sugar, calculates the weight of each pack. -/
def packWeight (totalSugar : ℕ) (numPacks : ℕ) (leftoverSugar : ℕ) : ℕ :=
  (totalSugar - leftoverSugar) / numPacks

/-- Proves that given 3020 grams of total sugar, 12 packs, and 20 grams of leftover sugar, 
    the weight of each pack is 250 grams. -/
theorem sugar_pack_weight :
  packWeight 3020 12 20 = 250 := by
  sorry

end sugar_pack_weight_l1505_150565


namespace equation_identity_l1505_150508

theorem equation_identity (a b c x : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  c * ((x - a) * (x - b)) / ((c - a) * (c - b)) +
  b * ((x - a) * (x - c)) / ((b - a) * (b - c)) +
  a * ((x - b) * (x - c)) / ((a - b) * (a - c)) = x := by
  sorry

end equation_identity_l1505_150508


namespace painting_selection_theorem_l1505_150528

/-- The number of traditional Chinese paintings -/
def traditional_paintings : Nat := 6

/-- The number of oil paintings -/
def oil_paintings : Nat := 4

/-- The number of watercolor paintings -/
def watercolor_paintings : Nat := 5

/-- The number of ways to select one painting from each type -/
def select_one_each : Nat := traditional_paintings * oil_paintings * watercolor_paintings

/-- The number of ways to select two paintings of different types -/
def select_two_different : Nat :=
  traditional_paintings * oil_paintings +
  traditional_paintings * watercolor_paintings +
  oil_paintings * watercolor_paintings

theorem painting_selection_theorem :
  select_one_each = 120 ∧ select_two_different = 74 := by
  sorry

end painting_selection_theorem_l1505_150528


namespace enclosed_area_semicircles_l1505_150583

/-- Given a semicircle with radius R and its diameter divided into parts 2r and 2(R-r),
    the area enclosed between the three semicircles (the original and two smaller ones)
    is equal to π r(R-r) -/
theorem enclosed_area_semicircles (R r : ℝ) (h1 : 0 < R) (h2 : 0 < r) (h3 : r < R) :
  let original_area := π * R^2 / 2
  let small_area1 := π * r^2 / 2
  let small_area2 := π * (R-r)^2 / 2
  original_area - small_area1 - small_area2 = π * r * (R-r) :=
by sorry

end enclosed_area_semicircles_l1505_150583


namespace mechanic_worked_five_and_half_hours_l1505_150558

/-- Calculates the number of hours a mechanic worked given the total cost, part costs, labor rate, and break time. -/
def mechanic_work_hours (total_cost parts_cost labor_rate_per_minute break_minutes : ℚ) : ℚ :=
  ((total_cost - parts_cost) / labor_rate_per_minute - break_minutes) / 60

/-- Proves that the mechanic worked 5.5 hours given the problem conditions. -/
theorem mechanic_worked_five_and_half_hours :
  let total_cost : ℚ := 220
  let parts_cost : ℚ := 2 * 20
  let labor_rate_per_minute : ℚ := 0.5
  let break_minutes : ℚ := 30
  mechanic_work_hours total_cost parts_cost labor_rate_per_minute break_minutes = 5.5 := by
  sorry


end mechanic_worked_five_and_half_hours_l1505_150558


namespace lara_overtakes_darla_l1505_150541

/-- The length of the circular track in meters -/
def track_length : ℝ := 500

/-- The speed ratio of Lara to Darla -/
def speed_ratio : ℝ := 1.2

/-- The number of laps completed by Lara when she first overtakes Darla -/
def laps_completed : ℝ := 6

/-- Theorem stating that Lara completes 6 laps when she first overtakes Darla -/
theorem lara_overtakes_darla :
  ∃ (t : ℝ), t > 0 ∧ speed_ratio * t * track_length = t * track_length + track_length ∧
  laps_completed = speed_ratio * t * track_length / track_length :=
sorry

end lara_overtakes_darla_l1505_150541


namespace fahrenheit_95_equals_celsius_35_l1505_150549

-- Define the conversion function from Fahrenheit to Celsius
def fahrenheit_to_celsius (f : ℚ) : ℚ := (f - 32) * (5/9)

-- Theorem statement
theorem fahrenheit_95_equals_celsius_35 : fahrenheit_to_celsius 95 = 35 := by
  sorry

end fahrenheit_95_equals_celsius_35_l1505_150549


namespace family_eating_habits_l1505_150546

theorem family_eating_habits (only_veg only_nonveg total_veg : ℕ) 
  (h1 : only_veg = 16)
  (h2 : only_nonveg = 9)
  (h3 : total_veg = 28) :
  total_veg - only_veg = 12 := by
  sorry

end family_eating_habits_l1505_150546


namespace middle_number_of_three_consecutive_squares_l1505_150551

theorem middle_number_of_three_consecutive_squares (n : ℕ) : 
  n^2 + (n+1)^2 + (n+2)^2 = 2030 → n + 1 = 26 := by
  sorry

end middle_number_of_three_consecutive_squares_l1505_150551


namespace right_triangle_square_areas_l1505_150515

theorem right_triangle_square_areas : ∀ (A B C : ℝ),
  (A = 6^2) →
  (B = 8^2) →
  (C = 10^2) →
  A + B = C :=
by
  sorry

end right_triangle_square_areas_l1505_150515


namespace optimalPlan_is_most_cost_effective_l1505_150531

/-- Represents a vehicle type with its capacity and cost -/
structure VehicleType where
  peopleCapacity : ℕ
  luggageCapacity : ℕ
  cost : ℕ

/-- Represents a rental plan -/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

def totalStudents : ℕ := 290
def totalLuggage : ℕ := 100
def totalVehicles : ℕ := 8

def typeA : VehicleType := ⟨40, 10, 2000⟩
def typeB : VehicleType := ⟨30, 20, 1800⟩

def isValidPlan (plan : RentalPlan) : Prop :=
  plan.typeA + plan.typeB = totalVehicles ∧
  plan.typeA * typeA.peopleCapacity + plan.typeB * typeB.peopleCapacity ≥ totalStudents ∧
  plan.typeA * typeA.luggageCapacity + plan.typeB * typeB.luggageCapacity ≥ totalLuggage

def planCost (plan : RentalPlan) : ℕ :=
  plan.typeA * typeA.cost + plan.typeB * typeB.cost

def optimalPlan : RentalPlan := ⟨5, 3⟩

theorem optimalPlan_is_most_cost_effective :
  isValidPlan optimalPlan ∧
  ∀ plan, isValidPlan plan → planCost optimalPlan ≤ planCost plan :=
sorry

end optimalPlan_is_most_cost_effective_l1505_150531


namespace forty_students_not_enrolled_l1505_150513

/-- The number of students not enrolled in any language course -/
def students_not_enrolled (total students_french students_german students_spanish
  students_french_german students_french_spanish students_german_spanish
  students_all_three : ℕ) : ℕ :=
  total - (students_french + students_german + students_spanish
           - students_french_german - students_french_spanish - students_german_spanish
           + students_all_three)

/-- Theorem stating that 40 students are not enrolled in any language course -/
theorem forty_students_not_enrolled :
  students_not_enrolled 150 60 50 40 20 15 10 5 = 40 := by
  sorry

end forty_students_not_enrolled_l1505_150513


namespace fraction_addition_simplification_l1505_150526

theorem fraction_addition_simplification :
  5 / 462 + 23 / 42 = 43 / 77 := by sorry

end fraction_addition_simplification_l1505_150526


namespace homework_problem_l1505_150523

theorem homework_problem (p t : ℕ) (h1 : p > 15) (h2 : p * t = (2*p - 6) * (t - 3)) : p * t = 126 := by
  sorry

end homework_problem_l1505_150523


namespace doris_hourly_rate_l1505_150520

/-- Doris's hourly rate for babysitting -/
def hourly_rate : ℝ := 20

/-- Minimum amount Doris needs to earn in 3 weeks -/
def minimum_earnings : ℝ := 1200

/-- Number of hours Doris babysits on weekdays -/
def weekday_hours : ℝ := 3

/-- Number of hours Doris babysits on Saturdays -/
def saturday_hours : ℝ := 5

/-- Number of weekdays in a week -/
def weekdays_per_week : ℝ := 5

/-- Number of Saturdays in a week -/
def saturdays_per_week : ℝ := 1

/-- Number of weeks Doris needs to work to earn minimum_earnings -/
def weeks_to_earn : ℝ := 3

theorem doris_hourly_rate :
  hourly_rate = minimum_earnings / (weeks_to_earn * (weekdays_per_week * weekday_hours + saturdays_per_week * saturday_hours)) := by
  sorry

end doris_hourly_rate_l1505_150520


namespace sin_210_deg_l1505_150521

/-- The sine of 210 degrees is equal to -1/2 --/
theorem sin_210_deg : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end sin_210_deg_l1505_150521


namespace power_sum_problem_l1505_150576

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 85) :
  a * x^5 + b * y^5 = 7025 / 29 := by
sorry

end power_sum_problem_l1505_150576


namespace stating_chess_tournament_players_l1505_150582

/-- The number of players in the chess tournament. -/
def num_players : ℕ := 11

/-- The total number of games played in the tournament. -/
def total_games : ℕ := 132

/-- 
Theorem stating that the number of players in the chess tournament is 11,
given the conditions of the problem.
-/
theorem chess_tournament_players :
  (∀ n : ℕ, n > 0 → 2 * n * (n - 1) = total_games) → num_players = 11 :=
by sorry

end stating_chess_tournament_players_l1505_150582


namespace salary_decrease_increase_l1505_150519

theorem salary_decrease_increase (original_salary : ℝ) (h : original_salary > 0) :
  let decreased_salary := original_salary * 0.5
  let final_salary := decreased_salary * 1.5
  final_salary = original_salary * 0.75 ∧ 
  (original_salary - final_salary) / original_salary = 0.25 :=
by sorry

end salary_decrease_increase_l1505_150519


namespace fraction_equality_l1505_150575

theorem fraction_equality (a b c : ℝ) (h1 : b ≠ c) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
  (a - b) / (b - c) = a / c ↔ 1 / b = (1 / a + 1 / c) / 2 := by
  sorry

end fraction_equality_l1505_150575


namespace distribute_8_3_non_empty_different_l1505_150555

/-- The number of ways to distribute n different balls into k different boxes --/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n different balls into k different boxes,
    where each box contains at least one ball --/
def distributeNonEmpty (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n different balls into k different boxes,
    where each box contains at least one ball and the number of balls in each box is different --/
def distributeNonEmptyDifferent (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 8 different balls into 3 different boxes,
    where each box contains at least one ball and the number of balls in each box is different,
    is equal to 2688 --/
theorem distribute_8_3_non_empty_different :
  distributeNonEmptyDifferent 8 3 = 2688 := by sorry

end distribute_8_3_non_empty_different_l1505_150555


namespace classroom_seating_arrangements_l1505_150542

/-- Represents a seating arrangement of students in a classroom -/
structure SeatingArrangement where
  rows : Nat
  cols : Nat
  boys : Nat
  girls : Nat

/-- Calculates the number of valid seating arrangements -/
def validArrangements (s : SeatingArrangement) : Nat :=
  2 * (Nat.factorial s.boys) * (Nat.factorial s.girls)

/-- Theorem stating the number of valid seating arrangements
    for the given classroom configuration -/
theorem classroom_seating_arrangements :
  let s : SeatingArrangement := {
    rows := 5,
    cols := 6,
    boys := 15,
    girls := 15
  }
  validArrangements s = 2 * (Nat.factorial 15) * (Nat.factorial 15) :=
by
  sorry

end classroom_seating_arrangements_l1505_150542


namespace expansion_coefficient_constraint_l1505_150511

theorem expansion_coefficient_constraint (k : ℕ+) :
  (15 : ℝ) * (k : ℝ)^4 < 120 → k = 1 := by
  sorry

end expansion_coefficient_constraint_l1505_150511


namespace printer_fraction_of_total_l1505_150525

/-- The price of the printer as a fraction of the total price with an enhanced computer -/
theorem printer_fraction_of_total (basic_computer_price printer_price enhanced_computer_price total_price_basic total_price_enhanced : ℚ) : 
  total_price_basic = basic_computer_price + printer_price →
  enhanced_computer_price = basic_computer_price + 500 →
  basic_computer_price = 2000 →
  total_price_enhanced = enhanced_computer_price + printer_price →
  printer_price / total_price_enhanced = 1 / 6 := by
  sorry

end printer_fraction_of_total_l1505_150525
