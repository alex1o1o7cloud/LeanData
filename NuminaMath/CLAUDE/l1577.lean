import Mathlib

namespace NUMINAMATH_CALUDE_range_of_m_l1577_157776

-- Define the sets A and B
def A (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}
def B := {x : ℝ | x^2 - 2*x - 15 ≤ 0}

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∃ x, x ∈ A m) ∧ -- A is non-empty
  (∃ x, x ∈ B) ∧ -- B is non-empty
  (∀ x, x ∈ A m → x ∈ B) → -- A ⊆ B
  2 ≤ m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1577_157776


namespace NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l1577_157735

-- Define propositions P and Q
def P (a b : ℝ) : Prop := a^2 + b^2 > 2*a*b
def Q (a b : ℝ) : Prop := |a + b| < |a| + |b|

-- Theorem statement
theorem P_necessary_not_sufficient_for_Q :
  (∀ a b : ℝ, Q a b → P a b) ∧
  (∃ a b : ℝ, P a b ∧ ¬(Q a b)) :=
sorry

end NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l1577_157735


namespace NUMINAMATH_CALUDE_multiply_and_add_l1577_157721

theorem multiply_and_add : 19 * 42 + 81 * 19 = 2337 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_l1577_157721


namespace NUMINAMATH_CALUDE_fraction_equality_l1577_157761

theorem fraction_equality : (10^9 : ℚ) / (2 * 5^2 * 10^3) = 20000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1577_157761


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l1577_157774

/-- Proves that adding 1.5 liters of 90% alcohol solution to 6 liters of 40% alcohol solution 
    results in a final mixture that is 50% alcohol. -/
theorem alcohol_mixture_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.4
  let added_concentration : ℝ := 0.9
  let added_volume : ℝ := 1.5
  let final_concentration : ℝ := 0.5
  let final_volume : ℝ := initial_volume + added_volume
  let initial_alcohol : ℝ := initial_volume * initial_concentration
  let added_alcohol : ℝ := added_volume * added_concentration
  let total_alcohol : ℝ := initial_alcohol + added_alcohol
  total_alcohol = final_volume * final_concentration :=
by
  sorry


end NUMINAMATH_CALUDE_alcohol_mixture_proof_l1577_157774


namespace NUMINAMATH_CALUDE_sum_ends_in_zero_squares_same_last_digit_l1577_157710

theorem sum_ends_in_zero_squares_same_last_digit (a b : ℤ) :
  (a + b) % 10 = 0 → (a ^ 2) % 10 = (b ^ 2) % 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_ends_in_zero_squares_same_last_digit_l1577_157710


namespace NUMINAMATH_CALUDE_tornado_distance_ratio_l1577_157703

/-- Given the distances traveled by various objects in a tornado, prove the ratio of
    the lawn chair's distance to the car's distance. -/
theorem tornado_distance_ratio :
  ∀ (car_distance lawn_chair_distance birdhouse_distance : ℝ),
  car_distance = 200 →
  birdhouse_distance = 1200 →
  birdhouse_distance = 3 * lawn_chair_distance →
  lawn_chair_distance / car_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_tornado_distance_ratio_l1577_157703


namespace NUMINAMATH_CALUDE_candy_mixture_proof_l1577_157734

/-- Proves that mixing 1 pound of candy A with 4 pounds of candy B
    results in 5 pounds of mixture costing $2 per pound -/
theorem candy_mixture_proof :
  let candy_a_price : ℝ := 3.20
  let candy_b_price : ℝ := 1.70
  let candy_a_amount : ℝ := 1
  let candy_b_amount : ℝ := 4
  let total_amount : ℝ := candy_a_amount + candy_b_amount
  let total_cost : ℝ := candy_a_price * candy_a_amount + candy_b_price * candy_b_amount
  let mixture_price_per_pound : ℝ := total_cost / total_amount
  total_amount = 5 ∧ mixture_price_per_pound = 2 := by
sorry

end NUMINAMATH_CALUDE_candy_mixture_proof_l1577_157734


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1577_157769

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, 7 * x^2 + m * x = -6) ∧ (7 * 3^2 + m * 3 = -6) → 
  (7 * (2/7)^2 + m * (2/7) = -6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1577_157769


namespace NUMINAMATH_CALUDE_tangent_line_theorem_l1577_157743

theorem tangent_line_theorem (a b : ℝ) : 
  (∀ x y : ℝ, y = x^2 + a*x + b) →
  (∀ x y : ℝ, x - y + 1 = 0 ↔ y = b ∧ x = 0) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_theorem_l1577_157743


namespace NUMINAMATH_CALUDE_correct_calculation_l1577_157740

theorem correct_calculation (x : ℤ) (h : x - 32 = 33) : x + 32 = 97 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1577_157740


namespace NUMINAMATH_CALUDE_king_will_be_checked_l1577_157705

/-- Represents a chess piece -/
inductive Piece
| King
| Rook

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents the state of the chessboard -/
structure ChessboardState :=
  (kingPos : Position)
  (rookPositions : List Position)

/-- Represents a move in the game -/
inductive Move
| KingMove (newPos : Position)
| RookMove (oldPos : Position) (newPos : Position)

/-- The game ends when the king is in check or reaches the top-right corner -/
def gameEnded (state : ChessboardState) : Prop :=
  (state.kingPos.x = 20 ∧ state.kingPos.y = 20) ∨
  state.rookPositions.any (λ rookPos => rookPos.x = state.kingPos.x ∨ rookPos.y = state.kingPos.y)

/-- A valid game sequence -/
def ValidGameSequence : List Move → Prop :=
  sorry

/-- The theorem to be proved -/
theorem king_will_be_checked
  (initialState : ChessboardState)
  (h1 : initialState.kingPos = ⟨1, 1⟩)
  (h2 : initialState.rookPositions.length = 10)
  (h3 : ∀ pos ∈ initialState.rookPositions, pos.x ≤ 20 ∧ pos.y ≤ 20) :
  ∀ (moves : List Move), ValidGameSequence moves →
    ∃ (n : Nat), let finalState := (moves.take n).foldl (λ s m => sorry) initialState
                 gameEnded finalState :=
sorry

end NUMINAMATH_CALUDE_king_will_be_checked_l1577_157705


namespace NUMINAMATH_CALUDE_john_task_completion_l1577_157796

-- Define the start time and end time of the first three tasks
def start_time : Nat := 9 * 60  -- 9:00 AM in minutes
def end_three_tasks : Nat := 12 * 60 + 15  -- 12:15 PM in minutes

-- Define the number of tasks
def num_tasks : Nat := 4

-- Define the theorem
theorem john_task_completion 
  (h1 : end_three_tasks - start_time = (num_tasks - 1) * ((end_three_tasks - start_time) / (num_tasks - 1)))
  (h2 : (end_three_tasks - start_time) % (num_tasks - 1) = 0) :
  end_three_tasks + ((end_three_tasks - start_time) / (num_tasks - 1)) = 13 * 60 + 20 := by
sorry


end NUMINAMATH_CALUDE_john_task_completion_l1577_157796


namespace NUMINAMATH_CALUDE_divisibility_of_power_tower_plus_one_l1577_157730

theorem divisibility_of_power_tower_plus_one (a : ℕ) : 
  ∃ n : ℕ, ∀ k : ℕ, a ∣ n^(n^k) + 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_power_tower_plus_one_l1577_157730


namespace NUMINAMATH_CALUDE_janet_masud_sibling_ratio_l1577_157793

/-- The number of Masud's siblings -/
def masud_siblings : ℕ := 60

/-- The number of Carlos' siblings -/
def carlos_siblings : ℕ := (3 * masud_siblings) / 4

/-- The number of Janet's siblings -/
def janet_siblings : ℕ := carlos_siblings + 135

/-- The ratio of Janet's siblings to Masud's siblings -/
def sibling_ratio : ℚ := janet_siblings / masud_siblings

theorem janet_masud_sibling_ratio :
  sibling_ratio = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_janet_masud_sibling_ratio_l1577_157793


namespace NUMINAMATH_CALUDE_ali_baba_cave_theorem_l1577_157771

/-- Represents the state of a barrel (herring head up or down) -/
inductive BarrelState
| Up
| Down

/-- Represents a configuration of n barrels -/
def Configuration (n : ℕ) := Fin n → BarrelState

/-- Represents a move by Ali Baba -/
def Move (n : ℕ) := Fin n → Bool

/-- Apply a move to a configuration -/
def applyMove (n : ℕ) (config : Configuration n) (move : Move n) : Configuration n :=
  fun i => if move i then match config i with
    | BarrelState.Up => BarrelState.Down
    | BarrelState.Down => BarrelState.Up
  else config i

/-- Check if all barrels are in the same state -/
def allSameState (n : ℕ) (config : Configuration n) : Prop :=
  (∀ i : Fin n, config i = BarrelState.Up) ∨ (∀ i : Fin n, config i = BarrelState.Down)

/-- Ali Baba can win in a finite number of moves -/
def canWin (n : ℕ) : Prop :=
  ∃ (strategy : ℕ → Move n), ∀ (initialConfig : Configuration n),
    ∃ (k : ℕ), allSameState n (Nat.rec initialConfig (fun i config => applyMove n config (strategy i)) k)

/-- n is a power of 2 -/
def isPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem ali_baba_cave_theorem (n : ℕ) :
  canWin n ↔ isPowerOfTwo n :=
sorry

end NUMINAMATH_CALUDE_ali_baba_cave_theorem_l1577_157771


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l1577_157770

theorem simplify_sqrt_sum (a : ℝ) (h : 3 < a ∧ a < 5) : 
  Real.sqrt ((a - 2)^2) + Real.sqrt ((a - 8)^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l1577_157770


namespace NUMINAMATH_CALUDE_negative_number_identification_l1577_157759

theorem negative_number_identification :
  let a := -(-2)
  let b := abs (-2)
  let c := (-2)^2
  let d := (-2)^3
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d < 0) := by sorry

end NUMINAMATH_CALUDE_negative_number_identification_l1577_157759


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1577_157784

theorem quadratic_real_roots (n : ℕ+) :
  (∃ x : ℝ, x^2 - 4*x + n.val = 0) ↔ n.val ∈ ({1, 2, 3, 4} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1577_157784


namespace NUMINAMATH_CALUDE_max_numbers_summing_to_1000_with_distinct_digit_sums_l1577_157712

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a list of natural numbers has pairwise distinct sums of digits -/
def hasPairwiseDistinctDigitSums (list : List ℕ) : Prop := sorry

/-- The maximum number of natural numbers summing to 1000 with pairwise distinct digit sums -/
theorem max_numbers_summing_to_1000_with_distinct_digit_sums :
  ∃ (list : List ℕ),
    list.sum = 1000 ∧
    hasPairwiseDistinctDigitSums list ∧
    list.length = 19 ∧
    ∀ (other_list : List ℕ),
      other_list.sum = 1000 →
      hasPairwiseDistinctDigitSums other_list →
      other_list.length ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_max_numbers_summing_to_1000_with_distinct_digit_sums_l1577_157712


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l1577_157789

theorem root_sum_reciprocal (p q r : ℂ) : 
  (p^3 - 2*p + 3 = 0) → 
  (q^3 - 2*q + 3 = 0) → 
  (r^3 - 2*r + 3 = 0) → 
  (1/(p+2) + 1/(q+2) + 1/(r+2) = -10) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l1577_157789


namespace NUMINAMATH_CALUDE_b_current_age_l1577_157792

/-- Given two people A and B, where:
    1) In 10 years, A will be twice as old as B was 10 years ago.
    2) A is currently 8 years older than B.
    This theorem proves that B's current age is 38 years. -/
theorem b_current_age (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10))
  (h2 : a = b + 8) : 
  b = 38 := by
sorry

end NUMINAMATH_CALUDE_b_current_age_l1577_157792


namespace NUMINAMATH_CALUDE_least_k_cube_divisible_by_120_l1577_157749

theorem least_k_cube_divisible_by_120 : 
  ∃ k : ℕ+, k.val = 30 ∧ 
  (∀ m : ℕ+, m.val < k.val → ¬(120 ∣ m.val^3)) ∧ 
  (120 ∣ k.val^3) := by
  sorry

end NUMINAMATH_CALUDE_least_k_cube_divisible_by_120_l1577_157749


namespace NUMINAMATH_CALUDE_cafeteria_bill_calculation_l1577_157794

/-- Calculates the total cost for Mell and her friends at the cafeteria --/
def cafeteria_bill (coffee_price ice_cream_price cake_price : ℚ) 
  (discount_rate tax_rate : ℚ) : ℚ :=
  let mell_order := 2 * coffee_price + cake_price
  let friend_order := 2 * coffee_price + cake_price + ice_cream_price
  let total_before_discount := mell_order + 2 * friend_order
  let discounted_total := total_before_discount * (1 - discount_rate)
  let final_total := discounted_total * (1 + tax_rate)
  final_total

/-- Theorem stating that the total bill for Mell and her friends is $47.69 --/
theorem cafeteria_bill_calculation : 
  cafeteria_bill 4 3 7 (15/100) (10/100) = 47.69 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_bill_calculation_l1577_157794


namespace NUMINAMATH_CALUDE_power_difference_l1577_157766

theorem power_difference (a m n : ℝ) (hm : a^m = 12) (hn : a^n = 3) : a^(m-n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l1577_157766


namespace NUMINAMATH_CALUDE_fourth_power_roots_l1577_157739

theorem fourth_power_roots (p q : ℝ) (r₁ r₂ : ℂ) : 
  (r₁^2 + p*r₁ + q = 0) → 
  (r₂^2 + p*r₂ + q = 0) → 
  (r₁^4)^2 + ((p^2 - 2*q)^2 - 2*q^2)*(r₁^4) + q^4 = 0 ∧
  (r₂^4)^2 + ((p^2 - 2*q)^2 - 2*q^2)*(r₂^4) + q^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_roots_l1577_157739


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1577_157718

theorem quadratic_roots_sum (α β : ℝ) : 
  (α^2 + 3*α - 7 = 0) → 
  (β^2 + 3*β - 7 = 0) → 
  α^2 + 4*α + β = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1577_157718


namespace NUMINAMATH_CALUDE_trig_identity_l1577_157701

theorem trig_identity (θ : Real) (h : Real.tan (θ + π/4) = 2) :
  Real.sin θ^2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ^2 = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1577_157701


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l1577_157711

theorem cuboid_surface_area (h : ℝ) (sum_edges : ℝ) (surface_area : ℝ) : 
  sum_edges = 100 ∧ 
  20 * h = sum_edges ∧ 
  surface_area = 2 * (2*h * 2*h + 2*h * h + 2*h * h) → 
  surface_area = 400 := by
sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l1577_157711


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1577_157720

/-- Proves that a rectangular field with sides in ratio 3:4 and fencing cost of 98 rupees at 25 paise per metre has an area of 9408 square meters -/
theorem rectangular_field_area (length width : ℝ) (cost_per_metre : ℚ) (total_cost : ℚ) : 
  length / width = 4 / 3 →
  cost_per_metre = 25 / 100 →
  total_cost = 98 →
  2 * (length + width) * cost_per_metre = total_cost →
  length * width = 9408 := by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_rectangular_field_area_l1577_157720


namespace NUMINAMATH_CALUDE_function_positive_implies_m_bound_l1577_157755

open Real

theorem function_positive_implies_m_bound (m : ℝ) :
  (∀ x : ℝ, x > 0 → (Real.exp x / x - m * x) > 0) →
  m < Real.exp 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_function_positive_implies_m_bound_l1577_157755


namespace NUMINAMATH_CALUDE_min_value_at_one_min_value_is_constant_l1577_157706

/-- A quadratic function f(x) = x^2 + (a+2)x + b symmetric about x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + (a+2)*x + b

/-- The property of f being symmetric about x = 1 -/
def symmetric_about_one (a b : ℝ) : Prop :=
  ∀ x, f a b (1 + x) = f a b (1 - x)

/-- The minimum value of f -/
def min_value (a b : ℝ) : ℝ := f a b 1

theorem min_value_at_one (a b : ℝ) 
  (h : symmetric_about_one a b) : 
  ∀ x, f a b x ≥ min_value a b :=
sorry

theorem min_value_is_constant (a b : ℝ) 
  (h : symmetric_about_one a b) : 
  ∃ c, min_value a b = c :=
sorry

end NUMINAMATH_CALUDE_min_value_at_one_min_value_is_constant_l1577_157706


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1577_157757

theorem complex_equation_solution :
  ∀ z : ℂ, z * (1 - Complex.I) = (1 + Complex.I)^3 → z = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1577_157757


namespace NUMINAMATH_CALUDE_sqrt_of_36_l1577_157795

theorem sqrt_of_36 : Real.sqrt 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_36_l1577_157795


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1577_157777

/-- Given a geometric sequence where the first three terms are a-1, a+1, and a+4,
    prove that the general term formula is a_n = 4 × (3/2)^(n-1) -/
theorem geometric_sequence_general_term (a : ℝ) (n : ℕ) :
  (a - 1 : ℝ) * (a + 4 : ℝ) = (a + 1 : ℝ)^2 →
  ∃ (seq : ℕ → ℝ), seq 1 = a - 1 ∧ seq 2 = a + 1 ∧ seq 3 = a + 4 ∧
    (∀ k : ℕ, seq (k + 1) / seq k = seq 2 / seq 1) →
    ∀ m : ℕ, seq m = 4 * (3/2 : ℝ)^(m - 1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1577_157777


namespace NUMINAMATH_CALUDE_triangle_property_l1577_157713

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.A = 2 * t.B) 
  (h2 : t.A = 3 * t.B) : 
  t.a^2 = t.b * (t.b + t.c) ∧ 
  t.c^2 = (1 / t.b) * (t.a - t.b) * (t.a^2 - t.b^2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_property_l1577_157713


namespace NUMINAMATH_CALUDE_dollar_neg_three_four_l1577_157714

-- Define the $ operation
def dollar (x y : ℤ) : ℤ := x * (y + 2) + x * y

-- Theorem statement
theorem dollar_neg_three_four : dollar (-3) 4 = -30 := by
  sorry

end NUMINAMATH_CALUDE_dollar_neg_three_four_l1577_157714


namespace NUMINAMATH_CALUDE_triangles_in_4x6_grid_l1577_157729

/-- Represents a grid with vertical and horizontal sections -/
structure Grid :=
  (vertical_sections : ℕ)
  (horizontal_sections : ℕ)

/-- Calculates the number of triangles in a grid with diagonal lines -/
def count_triangles (g : Grid) : ℕ :=
  let small_right_triangles := g.vertical_sections * g.horizontal_sections
  let medium_right_triangles := 2 * (g.vertical_sections - 1) * (g.horizontal_sections - 1)
  let large_isosceles_triangles := g.horizontal_sections - 1
  small_right_triangles + medium_right_triangles + large_isosceles_triangles

/-- Theorem: The number of triangles in a 4x6 grid is 90 -/
theorem triangles_in_4x6_grid :
  count_triangles { vertical_sections := 4, horizontal_sections := 6 } = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_4x6_grid_l1577_157729


namespace NUMINAMATH_CALUDE_marla_errand_time_l1577_157733

/-- The time Marla spends driving one way to her son's school -/
def drive_time : ℕ := 20

/-- The time Marla spends attending parent-teacher night -/
def meeting_time : ℕ := 70

/-- The total time Marla spends on the errand -/
def total_time : ℕ := 2 * drive_time + meeting_time

/-- Theorem stating that the total time Marla spends on the errand is 110 minutes -/
theorem marla_errand_time : total_time = 110 := by
  sorry

end NUMINAMATH_CALUDE_marla_errand_time_l1577_157733


namespace NUMINAMATH_CALUDE_fish_ratio_calculation_l1577_157727

/-- The ratio of tagged fish to total fish in a second catch -/
def fish_ratio (tagged_first : ℕ) (total_second : ℕ) (tagged_second : ℕ) (total_pond : ℕ) : ℚ :=
  tagged_second / total_second

/-- Theorem stating the ratio of tagged fish to total fish in the second catch -/
theorem fish_ratio_calculation :
  let tagged_first : ℕ := 40
  let total_second : ℕ := 50
  let tagged_second : ℕ := 2
  let total_pond : ℕ := 1000
  fish_ratio tagged_first total_second tagged_second total_pond = 1 / 25 := by
  sorry


end NUMINAMATH_CALUDE_fish_ratio_calculation_l1577_157727


namespace NUMINAMATH_CALUDE_zero_last_to_appear_l1577_157737

-- Define the Fibonacci sequence modulo 9
def fibMod9 : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => (fibMod9 n + fibMod9 (n + 1)) % 9

-- Define a function to check if a digit has appeared in the sequence up to n
def digitAppeared (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ fibMod9 k = d

-- Define a function to check if all digits from 0 to 8 have appeared
def allDigitsAppeared (n : ℕ) : Prop :=
  ∀ d, d ≤ 8 → digitAppeared d n

-- The main theorem
theorem zero_last_to_appear :
  ∃ n, allDigitsAppeared n ∧
    ¬(∃ k < n, allDigitsAppeared k) ∧
    fibMod9 n = 0 :=
  sorry

end NUMINAMATH_CALUDE_zero_last_to_appear_l1577_157737


namespace NUMINAMATH_CALUDE_binomial_coefficient_16_10_l1577_157726

theorem binomial_coefficient_16_10 :
  (Nat.choose 15 8 = 6435) →
  (Nat.choose 15 9 = 5005) →
  (Nat.choose 17 10 = 19448) →
  Nat.choose 16 10 = 8008 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_16_10_l1577_157726


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l1577_157797

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) 
  (h1 : L > 0) (h2 : W > 0) (h3 : x > 0) :
  L * (1 + x / 100) * W * 0.95 = L * W * 1.045 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l1577_157797


namespace NUMINAMATH_CALUDE_indistinguishable_balls_in_boxes_l1577_157758

/-- The number of partitions of n indistinguishable objects into k or fewer non-empty parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- The balls are indistinguishable -/
def balls : ℕ := 4

/-- The boxes are indistinguishable -/
def boxes : ℕ := 4

theorem indistinguishable_balls_in_boxes : partition_count balls boxes = 5 := by sorry

end NUMINAMATH_CALUDE_indistinguishable_balls_in_boxes_l1577_157758


namespace NUMINAMATH_CALUDE_lice_check_time_l1577_157750

/-- Calculates the time required for lice checks in an elementary school -/
theorem lice_check_time (kindergarteners : ℕ) (first_graders : ℕ) (second_graders : ℕ) (third_graders : ℕ) 
  (time_per_check : ℕ) (h1 : kindergarteners = 26) (h2 : first_graders = 19) (h3 : second_graders = 20) 
  (h4 : third_graders = 25) (h5 : time_per_check = 2) : 
  (kindergarteners + first_graders + second_graders + third_graders) * time_per_check / 60 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lice_check_time_l1577_157750


namespace NUMINAMATH_CALUDE_correct_classification_l1577_157754

-- Define the set of statement numbers
def StatementNumbers : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define the function that classifies numbers as precise or approximate
def classify : Nat → Bool
| 1 => true  -- Xiao Ming's books (precise)
| 2 => true  -- War cost (precise)
| 3 => true  -- DVD sales (precise)
| 4 => false -- Brain cells (approximate)
| 5 => true  -- Xiao Hong's score (precise)
| 6 => false -- Coal reserves (approximate)
| _ => false -- For completeness

-- Theorem statement
theorem correct_classification :
  {n ∈ StatementNumbers | classify n = true} = {1, 2, 3, 5} ∧
  {n ∈ StatementNumbers | classify n = false} = {4, 6} := by
  sorry


end NUMINAMATH_CALUDE_correct_classification_l1577_157754


namespace NUMINAMATH_CALUDE_bob_has_31_pennies_l1577_157724

/-- The number of pennies Alex has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Bob has -/
def bob_pennies : ℕ := sorry

/-- If Alex gives Bob a penny, Bob will have four times as many pennies as Alex has -/
axiom condition1 : bob_pennies + 1 = 4 * (alex_pennies - 1)

/-- If Bob gives Alex a penny, Bob will have three times as many pennies as Alex has -/
axiom condition2 : bob_pennies - 1 = 3 * (alex_pennies + 1)

/-- Bob has 31 pennies -/
theorem bob_has_31_pennies : bob_pennies = 31 := by sorry

end NUMINAMATH_CALUDE_bob_has_31_pennies_l1577_157724


namespace NUMINAMATH_CALUDE_tournament_result_l1577_157748

-- Define the tournament structure
structure Tournament :=
  (teams : Fin 9 → ℕ)  -- Each team's points
  (t1_wins : ℕ)
  (t1_draws : ℕ)
  (t1_losses : ℕ)
  (t9_wins : ℕ)
  (t9_draws : ℕ)
  (t9_losses : ℕ)

-- Define the conditions of the tournament
def valid_tournament (t : Tournament) : Prop :=
  t.teams 0 = 3 * t.t1_wins + t.t1_draws ∧  -- T1's score
  t.teams 8 = t.t9_draws ∧  -- T9's score
  t.t1_wins = 3 ∧ t.t1_draws = 4 ∧ t.t1_losses = 1 ∧
  t.t9_wins = 0 ∧ t.t9_draws = 5 ∧ t.t9_losses = 3 ∧
  (∀ i j, i < j → t.teams i > t.teams j) ∧  -- Strict ordering
  (∀ i, t.teams i ≤ 24)  -- Maximum possible points

-- Define the theorem
theorem tournament_result (t : Tournament) (h : valid_tournament t) :
  (¬ ∃ (t3_defeats_t4 : Bool), t.teams 2 > t.teams 3) ∧
  (∃ (t4_defeats_t3 : Bool), t.teams 3 > t.teams 2) :=
sorry

end NUMINAMATH_CALUDE_tournament_result_l1577_157748


namespace NUMINAMATH_CALUDE_inequality_solution_l1577_157709

theorem inequality_solution (x : ℝ) :
  (∃ a : ℝ, a ∈ Set.Icc (-1) 2 ∧ (2 - a) * x^3 + (1 - 2*a) * x^2 - 6*x + 5 + 4*a - a^2 < 0) ↔
  (x < -2 ∨ (0 < x ∧ x < 1) ∨ x > 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1577_157709


namespace NUMINAMATH_CALUDE_erased_number_is_202_l1577_157751

-- Define the sequence of consecutive positive integers
def consecutive_sequence (n : ℕ) : List ℕ := List.range n

-- Define the function to calculate the sum of the first n positive integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the function to calculate the average of the remaining numbers after erasing x
def average_after_erasing (n : ℕ) (x : ℕ) : ℚ :=
  (sum_first_n n - x) / (n - 1 : ℚ)

-- The theorem to prove
theorem erased_number_is_202 (n : ℕ) (x : ℕ) :
  x ∈ consecutive_sequence n →
  average_after_erasing n x = 151 / 3 →
  x = 202 := by
  sorry

end NUMINAMATH_CALUDE_erased_number_is_202_l1577_157751


namespace NUMINAMATH_CALUDE_midpoint_complex_numbers_l1577_157700

theorem midpoint_complex_numbers : 
  let a : ℂ := (1 : ℂ) / (1 + Complex.I)
  let b : ℂ := (1 : ℂ) / (1 - Complex.I)
  let c : ℂ := (a + b) / 2
  c = (1 : ℂ) / 2 := by sorry

end NUMINAMATH_CALUDE_midpoint_complex_numbers_l1577_157700


namespace NUMINAMATH_CALUDE_robot_path_lengths_l1577_157722

/-- Represents the direction the robot is facing -/
inductive Direction
| North
| East
| South
| West

/-- Represents a point in the plane -/
structure Point where
  x : Int
  y : Int

/-- Represents the state of the robot -/
structure RobotState where
  position : Point
  direction : Direction

/-- The robot's path -/
def RobotPath := List RobotState

/-- Function to check if a path is valid according to the problem conditions -/
def is_valid_path (path : RobotPath) : Bool :=
  sorry

/-- Function to check if a path returns to the starting point -/
def returns_to_start (path : RobotPath) : Bool :=
  sorry

/-- Function to check if a path visits any point more than once -/
def no_revisits (path : RobotPath) : Bool :=
  sorry

/-- Theorem stating the possible path lengths for the robot -/
theorem robot_path_lengths :
  ∀ (n : Nat), 
    (∃ (path : RobotPath), 
      path.length = n ∧ 
      is_valid_path path ∧ 
      returns_to_start path ∧ 
      no_revisits path) ↔ 
    (∃ (k : Nat), n = 4 * k ∧ k ≥ 3) :=
  sorry

end NUMINAMATH_CALUDE_robot_path_lengths_l1577_157722


namespace NUMINAMATH_CALUDE_line_through_points_l1577_157788

/-- A line passing through given points -/
structure Line where
  a : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.a * x + l.b

/-- The main theorem -/
theorem line_through_points : ∃ (l : Line),
  l.contains 2 8 ∧
  l.contains 5 17 ∧
  l.contains 8 26 ∧
  l.contains 34 104 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1577_157788


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l1577_157799

theorem cube_volume_surface_area (y : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 6*y ∧ 6*s^2 = 2*y) → y = 5832 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l1577_157799


namespace NUMINAMATH_CALUDE_new_person_weight_l1577_157767

theorem new_person_weight (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) :
  n = 10 →
  avg_increase = 2.5 →
  old_weight = 65 →
  (n : ℝ) * avg_increase + old_weight = 90 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1577_157767


namespace NUMINAMATH_CALUDE_floor_of_e_equals_two_l1577_157772

theorem floor_of_e_equals_two : ⌊Real.exp 1⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_e_equals_two_l1577_157772


namespace NUMINAMATH_CALUDE_competition_scores_l1577_157702

theorem competition_scores (x y z w : ℝ) 
  (hA : x = (y + z + w) / 3 + 2)
  (hB : y = (x + z + w) / 3 - 3)
  (hC : z = (x + y + w) / 3 + 3) :
  (x + y + z) / 3 - w = 2 := by sorry

end NUMINAMATH_CALUDE_competition_scores_l1577_157702


namespace NUMINAMATH_CALUDE_arithmetic_operations_l1577_157708

theorem arithmetic_operations :
  (-10 + 2 = -8) ∧
  (-6 - 3 = -9) ∧
  ((-4) * 6 = -24) ∧
  ((-15) / 5 = -3) ∧
  ((-4)^2 / 2 = 8) ∧
  (|(-2)| - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l1577_157708


namespace NUMINAMATH_CALUDE_greatest_difference_l1577_157763

theorem greatest_difference (x y : ℤ) 
  (hx : 5 < x ∧ x < 8) 
  (hy : 8 < y ∧ y < 13) 
  (hxdiv : x % 3 = 0) 
  (hydiv : y % 3 = 0) : 
  (∀ a b : ℤ, 5 < a ∧ a < 8 ∧ 8 < b ∧ b < 13 ∧ a % 3 = 0 ∧ b % 3 = 0 → b - a ≤ y - x) ∧ y - x = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_difference_l1577_157763


namespace NUMINAMATH_CALUDE_kareem_has_largest_number_l1577_157781

def jose_final (start : ℕ) : ℕ :=
  ((start - 2) * 4) + 5

def thuy_final (start : ℕ) : ℕ :=
  ((start * 3) - 3) - 4

def kareem_final (start : ℕ) : ℕ :=
  ((start - 3) + 4) * 3

theorem kareem_has_largest_number :
  kareem_final 20 > jose_final 15 ∧ kareem_final 20 > thuy_final 15 :=
by sorry

end NUMINAMATH_CALUDE_kareem_has_largest_number_l1577_157781


namespace NUMINAMATH_CALUDE_necessary_not_implies_sufficient_l1577_157725

theorem necessary_not_implies_sufficient (A B : Prop) : 
  (A → B) → ¬(∀ A B, (A → B) → (B → A)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_implies_sufficient_l1577_157725


namespace NUMINAMATH_CALUDE_cloth_cost_price_l1577_157717

/-- Given a trader sells cloth with the following conditions:
    - Total length of cloth sold is 75 meters
    - Total selling price is Rs. 4950
    - Profit per meter is Rs. 15
    This theorem proves that the cost price per meter is Rs. 51 -/
theorem cloth_cost_price (total_length : ℕ) (total_selling_price : ℕ) (profit_per_meter : ℕ) :
  total_length = 75 →
  total_selling_price = 4950 →
  profit_per_meter = 15 →
  (total_selling_price - total_length * profit_per_meter) / total_length = 51 :=
by sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l1577_157717


namespace NUMINAMATH_CALUDE_simplify_expression_l1577_157704

theorem simplify_expression (c : ℝ) : ((3 * c + 5) - 3 * c) / 2 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1577_157704


namespace NUMINAMATH_CALUDE_driver_total_stops_l1577_157780

/-- The total number of stops made by a delivery driver -/
def total_stops (initial_stops additional_stops : ℕ) : ℕ :=
  initial_stops + additional_stops

/-- Theorem: The delivery driver made 7 stops in total -/
theorem driver_total_stops :
  total_stops 3 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_driver_total_stops_l1577_157780


namespace NUMINAMATH_CALUDE_sum_of_squares_equality_l1577_157785

variables {a b c x y z : ℝ}

theorem sum_of_squares_equality 
  (h1 : x / a + y / b + z / c = 5)
  (h2 : a / x + b / y + c / z = 0)
  : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equality_l1577_157785


namespace NUMINAMATH_CALUDE_chest_value_is_35000_l1577_157765

/-- Represents the pirate treasure distribution problem -/
structure PirateTreasure where
  total_pirates : ℕ
  total_chests : ℕ
  pirates_with_chests : ℕ
  contribution_per_chest : ℕ

/-- The specific instance of the pirate treasure problem -/
def pirate_problem : PirateTreasure := {
  total_pirates := 7
  total_chests := 5
  pirates_with_chests := 5
  contribution_per_chest := 10000
}

/-- Calculates the value of one chest based on the given problem parameters -/
def chest_value (p : PirateTreasure) : ℕ :=
  let total_contribution := p.pirates_with_chests * p.contribution_per_chest
  let pirates_without_chests := p.total_pirates - p.pirates_with_chests
  let compensation_per_pirate := total_contribution / pirates_without_chests
  p.total_pirates * compensation_per_pirate / p.total_chests

/-- Theorem stating that the chest value for the given problem is 35000 -/
theorem chest_value_is_35000 : chest_value pirate_problem = 35000 := by
  sorry

end NUMINAMATH_CALUDE_chest_value_is_35000_l1577_157765


namespace NUMINAMATH_CALUDE_pencil_buyers_difference_l1577_157745

theorem pencil_buyers_difference (pencil_cost : ℕ) 
  (h1 : pencil_cost > 0)
  (h2 : 234 % pencil_cost = 0)
  (h3 : 312 % pencil_cost = 0) :
  312 / pencil_cost - 234 / pencil_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_pencil_buyers_difference_l1577_157745


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1577_157786

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Define the property that a_1 and a_19 are roots of x^2 - 10x + 16 = 0
def roots_property (a : ℕ → ℝ) : Prop :=
  a 1 ^ 2 - 10 * a 1 + 16 = 0 ∧ a 19 ^ 2 - 10 * a 19 + 16 = 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  roots_property a →
  a 8 * a 10 * a 12 = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1577_157786


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l1577_157747

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed of the swimmer. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 5.17 km/h. -/
theorem swimmer_speed_in_still_water :
  ∃ (s : SwimmerSpeeds),
    (effectiveSpeed s true * 5 = 36) ∧
    (effectiveSpeed s false * 7 = 22) ∧
    (s.swimmer = 5.17) := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l1577_157747


namespace NUMINAMATH_CALUDE_sea_turtle_shell_age_l1577_157716

/-- Converts an octal digit to decimal --/
def octal_to_decimal (digit : Nat) : Nat :=
  if digit < 8 then digit else 0

/-- Converts an octal number to decimal --/
def octal_to_decimal_full (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + octal_to_decimal digit * 8^i) 0

theorem sea_turtle_shell_age :
  octal_to_decimal_full [4, 5, 7, 3] = 2028 := by
  sorry

end NUMINAMATH_CALUDE_sea_turtle_shell_age_l1577_157716


namespace NUMINAMATH_CALUDE_correct_plate_set_is_valid_l1577_157753

/-- Represents a plate with a count of bacteria -/
structure Plate where
  count : ℕ
  deriving Repr

/-- Represents a set of plates used in the dilution spread plate method -/
structure PlateSet where
  plates : List Plate
  dilutionFactor : ℕ
  deriving Repr

/-- Checks if a plate count is valid (between 30 and 300) -/
def isValidCount (count : ℕ) : Bool :=
  30 ≤ count ∧ count ≤ 300

/-- Checks if a plate set is valid for the dilution spread plate method -/
def isValidPlateSet (ps : PlateSet) : Bool :=
  ps.plates.length ≥ 3 ∧ 
  ps.plates.all (fun p => isValidCount p.count) ∧
  ps.dilutionFactor = 10^6

/-- Calculates the average count of a plate set -/
def averageCount (ps : PlateSet) : ℚ :=
  let total : ℚ := ps.plates.foldl (fun acc p => acc + p.count) 0
  total / ps.plates.length

/-- The correct plate set for the problem -/
def correctPlateSet : PlateSet :=
  { plates := [⟨210⟩, ⟨240⟩, ⟨250⟩],
    dilutionFactor := 10^6 }

theorem correct_plate_set_is_valid :
  isValidPlateSet correctPlateSet ∧ 
  averageCount correctPlateSet = 233 :=
sorry

end NUMINAMATH_CALUDE_correct_plate_set_is_valid_l1577_157753


namespace NUMINAMATH_CALUDE_proportion_equality_l1577_157790

theorem proportion_equality (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l1577_157790


namespace NUMINAMATH_CALUDE_min_value_polynomial_l1577_157756

theorem min_value_polynomial (x : ℝ) :
  ∃ (min : ℝ), min = 2022 - (5 + Real.sqrt 5) / 2 ∧
  ∀ y : ℝ, (y + 1) * (y + 2) * (y + 3) * (y + 4) + y + 2023 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_polynomial_l1577_157756


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l1577_157783

/-- The constant term in the expansion of (√5/5 * x^2 + 1/x)^6 is 3 -/
theorem constant_term_binomial_expansion :
  let a := Real.sqrt 5 / 5
  let b := 1
  let n := 6
  let k := 4  -- The value of k where x^(2n-3k) = x^0
  (Nat.choose n k) * a^(n-k) * b^k = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l1577_157783


namespace NUMINAMATH_CALUDE_trajectory_equation_l1577_157707

/-- The trajectory of a point M that satisfies |MF₁| + |MF₂| = 10, where F₁ = (-3, 0) and F₂ = (3, 0) -/
theorem trajectory_equation (M : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  F₁ = (-3, 0) →
  F₂ = (3, 0) →
  ‖M - F₁‖ + ‖M - F₂‖ = 10 →
  (M.1^2 / 25 + M.2^2 / 16 = 1) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1577_157707


namespace NUMINAMATH_CALUDE_expand_product_l1577_157798

theorem expand_product (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * (4 / x - 5 * x^2 + 20 / x^3) = 3 / x - 15 * x^2 / 4 + 15 / x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1577_157798


namespace NUMINAMATH_CALUDE_triangle_height_l1577_157787

/-- Given a triangle with base 6 and area 24, prove its height is 8 -/
theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 6 → 
  area = 24 → 
  area = 1/2 * base * height → 
  height = 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l1577_157787


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l1577_157728

theorem paper_clip_distribution (total_clips : ℕ) (num_boxes : ℕ) (clips_per_box : ℕ) :
  total_clips = 81 →
  num_boxes = 9 →
  total_clips = num_boxes * clips_per_box →
  clips_per_box = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l1577_157728


namespace NUMINAMATH_CALUDE_dark_lord_sword_distribution_l1577_157773

/-- Calculates the weight of swords each orc must carry given the total weight,
    number of squads, and orcs per squad. -/
def weight_per_orc (total_weight : ℕ) (num_squads : ℕ) (orcs_per_squad : ℕ) : ℚ :=
  total_weight / (num_squads * orcs_per_squad)

/-- Proves that given 1200 pounds of swords, 10 squads, and 8 orcs per squad,
    each orc must carry 15 pounds of swords. -/
theorem dark_lord_sword_distribution :
  weight_per_orc 1200 10 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_dark_lord_sword_distribution_l1577_157773


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_number_of_boys_is_correct_l1577_157732

/-- The number of boys in a class given certain weight conditions -/
theorem number_of_boys_in_class : ℕ :=
  let initial_average : ℚ := 58.4
  let misread_weight : ℕ := 56
  let correct_weight : ℕ := 68
  let correct_average : ℚ := 59
  20

theorem number_of_boys_is_correct (n : ℕ) :
  (n : ℚ) * 58.4 + (68 - 56) = n * 59 → n = number_of_boys_in_class :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_number_of_boys_is_correct_l1577_157732


namespace NUMINAMATH_CALUDE_triangle_nature_l1577_157741

theorem triangle_nature (a b c : ℝ) (h_ratio : a / b = 3 / 4 ∧ b / c = 4 / 5)
  (h_perimeter : a + b + c = 36) : a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_nature_l1577_157741


namespace NUMINAMATH_CALUDE_visible_bird_legs_count_l1577_157782

theorem visible_bird_legs_count :
  let crows : ℕ := 4
  let pigeons : ℕ := 3
  let flamingos : ℕ := 5
  let sparrows : ℕ := 8
  let crow_legs : ℕ := 2
  let pigeon_legs : ℕ := 2
  let flamingo_legs : ℕ := 3
  let sparrow_legs : ℕ := 2
  crows * crow_legs + pigeons * pigeon_legs + flamingos * flamingo_legs + sparrows * sparrow_legs = 45 :=
by sorry

end NUMINAMATH_CALUDE_visible_bird_legs_count_l1577_157782


namespace NUMINAMATH_CALUDE_cylinder_volume_l1577_157791

/-- The volume of a cylinder whose lateral surface unfolds into a square with side length 4 -/
theorem cylinder_volume (h : Real) (r : Real) : 
  h = 4 ∧ 2 * Real.pi * r = 4 → Real.pi * r^2 * h = 16 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l1577_157791


namespace NUMINAMATH_CALUDE_masters_proportion_in_team_l1577_157760

/-- Represents a team of juniors and masters in a shooting tournament. -/
structure ShootingTeam where
  juniors : ℕ
  masters : ℕ

/-- Calculates the proportion of masters in the team. -/
def mastersProportion (team : ShootingTeam) : ℚ :=
  team.masters / (team.juniors + team.masters)

/-- The theorem stating the proportion of masters in the team under given conditions. -/
theorem masters_proportion_in_team (team : ShootingTeam) 
  (h1 : 22 * team.juniors + 47 * team.masters = 41 * (team.juniors + team.masters)) :
  mastersProportion team = 19 / 25 := by
  sorry

#eval (19 : ℚ) / 25  -- To verify that 19/25 is indeed equal to 0.76

end NUMINAMATH_CALUDE_masters_proportion_in_team_l1577_157760


namespace NUMINAMATH_CALUDE_x_intercept_distance_l1577_157764

/-- Two lines with slopes 4 and 6 intersecting at (8,12) have x-intercepts with distance 1 -/
theorem x_intercept_distance (line1 line2 : ℝ → ℝ) : 
  (∀ x, line1 x = 4*x - 20) →  -- Equation of line with slope 4
  (∀ x, line2 x = 6*x - 36) →  -- Equation of line with slope 6
  line1 8 = 12 →              -- Lines intersect at (8,12)
  line2 8 = 12 →              -- Lines intersect at (8,12)
  ∃ x1 x2, line1 x1 = 0 ∧ line2 x2 = 0 ∧ |x2 - x1| = 1 := by
sorry

end NUMINAMATH_CALUDE_x_intercept_distance_l1577_157764


namespace NUMINAMATH_CALUDE_friends_score_l1577_157723

/-- Given that Edward and his friend scored a total of 13 points in basketball,
    and Edward scored 7 points, prove that Edward's friend scored 6 points. -/
theorem friends_score (total : ℕ) (edward : ℕ) (friend : ℕ)
    (h1 : total = 13)
    (h2 : edward = 7)
    (h3 : total = edward + friend) :
  friend = 6 := by
sorry

end NUMINAMATH_CALUDE_friends_score_l1577_157723


namespace NUMINAMATH_CALUDE_expression_meets_requirements_l1577_157738

/-- Represents an algebraic expression -/
inductive AlgebraicExpression
  | constant (n : ℚ)
  | variable (name : String)
  | product (e1 e2 : AlgebraicExpression)
  | power (base : AlgebraicExpression) (exponent : ℕ)
  | fraction (numerator denominator : AlgebraicExpression)
  | negation (e : AlgebraicExpression)

/-- Checks if an algebraic expression meets the standard writing requirements -/
def meetsWritingRequirements (e : AlgebraicExpression) : Prop :=
  match e with
  | AlgebraicExpression.constant _ => true
  | AlgebraicExpression.variable _ => true
  | AlgebraicExpression.product e1 e2 => meetsWritingRequirements e1 ∧ meetsWritingRequirements e2
  | AlgebraicExpression.power base exponent => meetsWritingRequirements base ∧ exponent > 0
  | AlgebraicExpression.fraction num den => meetsWritingRequirements num ∧ meetsWritingRequirements den
  | AlgebraicExpression.negation e => meetsWritingRequirements e

/-- The expression -1/3 * x^2 * y -/
def expression : AlgebraicExpression :=
  AlgebraicExpression.negation
    (AlgebraicExpression.fraction
      (AlgebraicExpression.constant 1)
      (AlgebraicExpression.constant 3))

theorem expression_meets_requirements :
  meetsWritingRequirements expression :=
sorry


end NUMINAMATH_CALUDE_expression_meets_requirements_l1577_157738


namespace NUMINAMATH_CALUDE_perfect_squares_equivalence_l1577_157775

theorem perfect_squares_equivalence (n : ℕ+) :
  (∃ k : ℕ, 2 * n + 1 = k^2) ∧ (∃ t : ℕ, 3 * n + 1 = t^2) ↔
  (∃ k : ℕ, n + 1 = k^2 + (k + 1)^2) ∧ 
  (∃ t : ℕ, n + 1 = (t - 1)^2 + 2 * t^2 ∨ n + 1 = (t + 1)^2 + 2 * t^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_equivalence_l1577_157775


namespace NUMINAMATH_CALUDE_problem_1_l1577_157762

theorem problem_1 (m : ℝ) : m * m^3 + (-m^2)^3 / m^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1577_157762


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1577_157736

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  (X^4 - 1) * (X^2 - 1) = (X^2 + X + 1) * q + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1577_157736


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_prism_l1577_157752

/-- The volume of a sphere that circumscribes a rectangular prism with dimensions 2 × 1 × 1 is √6π -/
theorem sphere_volume_circumscribing_prism :
  let l : ℝ := 2
  let w : ℝ := 1
  let h : ℝ := 1
  let diagonal := Real.sqrt (l^2 + w^2 + h^2)
  let radius := diagonal / 2
  let volume := (4/3) * Real.pi * radius^3
  volume = Real.sqrt 6 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_prism_l1577_157752


namespace NUMINAMATH_CALUDE_maximal_value_S_l1577_157719

theorem maximal_value_S (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0)
  (hsum : a + b + c + d = 100) :
  let S := (a / (b + 7))^(1/3) + (b / (c + 7))^(1/3) + (c / (d + 7))^(1/3) + (d / (a + 7))^(1/3)
  S ≤ 8 / 7^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_maximal_value_S_l1577_157719


namespace NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l1577_157778

/-- Given a 2x2 matrix B with elements [[4, 5], [3, m]], prove that if B^(-1) = j * B, 
    then m = -4 and j = 1/31 -/
theorem matrix_inverse_scalar_multiple 
  (B : Matrix (Fin 2) (Fin 2) ℝ)
  (h_B : B = !![4, 5; 3, m])
  (h_inv : B⁻¹ = j • B) :
  m = -4 ∧ j = 1 / 31 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_scalar_multiple_l1577_157778


namespace NUMINAMATH_CALUDE_N_subset_M_l1577_157744

-- Define the sets M and N
def M : Set ℝ := {x | |x| ≤ 1}
def N : Set ℝ := {y | ∃ x, y = 2^x ∧ x ≤ 0}

-- State the theorem
theorem N_subset_M : N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_N_subset_M_l1577_157744


namespace NUMINAMATH_CALUDE_quadratic_inequality_roots_l1577_157742

theorem quadratic_inequality_roots (b : ℝ) : 
  (∀ x, -x^2 + b*x - 12 < 0 ↔ x < 3 ∨ x > 7) → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_roots_l1577_157742


namespace NUMINAMATH_CALUDE_pairball_playing_time_l1577_157715

theorem pairball_playing_time (n : ℕ) (total_time : ℕ) (h1 : n = 7) (h2 : total_time = 105) :
  let players_per_game : ℕ := 2
  let total_child_minutes : ℕ := players_per_game * total_time
  let time_per_child : ℕ := total_child_minutes / n
  time_per_child = 30 := by sorry

end NUMINAMATH_CALUDE_pairball_playing_time_l1577_157715


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1577_157731

theorem arithmetic_calculations :
  (24 - |(-2)| + (-16) - 8 = -2) ∧
  ((-2) * (3/2) / (-3/4) * 4 = 16) ∧
  (-1^2016 - (1 - 0.5) / 3 * (2 - (-3)^2) = 1/6) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1577_157731


namespace NUMINAMATH_CALUDE_league_games_count_l1577_157768

/-- The number of games played in a league season -/
def games_in_season (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * k

theorem league_games_count :
  games_in_season 20 7 = 1330 := by
  sorry

end NUMINAMATH_CALUDE_league_games_count_l1577_157768


namespace NUMINAMATH_CALUDE_product_and_divisibility_l1577_157746

theorem product_and_divisibility (n : ℕ) : 
  n = 3 → 
  ((n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720) ∧ 
  ¬(720 % 11 = 0) := by
  sorry

end NUMINAMATH_CALUDE_product_and_divisibility_l1577_157746


namespace NUMINAMATH_CALUDE_rhombus_side_length_l1577_157779

/-- A rhombus with area K and one diagonal three times the length of the other has side length √(5K/3) -/
theorem rhombus_side_length (K : ℝ) (d₁ d₂ s : ℝ) (h₁ : K > 0) (h₂ : d₁ > 0) (h₃ : d₂ > 0) (h₄ : s > 0) :
  d₂ = 3 * d₁ →
  K = (1/2) * d₁ * d₂ →
  s^2 = (d₁/2)^2 + (d₂/2)^2 →
  s = Real.sqrt ((5 * K) / 3) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l1577_157779
