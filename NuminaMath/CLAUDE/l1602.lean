import Mathlib

namespace mikey_leaves_total_l1602_160247

theorem mikey_leaves_total (initial_leaves additional_leaves : Float) : 
  initial_leaves = 356.0 → additional_leaves = 112.0 → 
  initial_leaves + additional_leaves = 468.0 := by
  sorry

end mikey_leaves_total_l1602_160247


namespace parallel_lines_k_value_l1602_160216

/-- Given two parallel lines x - ky - k = 0 and y = k(x-1), prove that k = -1 -/
theorem parallel_lines_k_value (k : ℝ) : 
  (∀ x y : ℝ, x - k * y - k = 0 ↔ y = k * (x - 1)) → 
  (∀ x y : ℝ, x - k * y - k = 0 → ∃ c : ℝ, y = (1/k) * x + c) →
  k ≠ 0 →
  k = -1 := by
  sorry

end parallel_lines_k_value_l1602_160216


namespace sandys_shorts_expense_l1602_160260

/-- Given Sandy's shopping expenses, calculate the amount spent on shorts -/
theorem sandys_shorts_expense (total shirt jacket : ℚ)
  (h_total : total = 33.56)
  (h_shirt : shirt = 12.14)
  (h_jacket : jacket = 7.43) :
  total - shirt - jacket = 13.99 := by
  sorry

end sandys_shorts_expense_l1602_160260


namespace calculation_proof_l1602_160276

theorem calculation_proof : 5 * (-2) + Real.pi ^ 0 + (-1) ^ 2023 - 2 ^ 3 = -18 := by
  sorry

end calculation_proof_l1602_160276


namespace fred_card_spending_l1602_160225

-- Define the costs of each type of card
def football_pack_cost : ℝ := 2.73
def pokemon_pack_cost : ℝ := 4.01
def baseball_deck_cost : ℝ := 8.95

-- Define the number of packs/decks bought
def football_packs : ℕ := 2
def pokemon_packs : ℕ := 1
def baseball_decks : ℕ := 1

-- Define the total cost function
def total_cost : ℝ := 
  (football_pack_cost * football_packs) + 
  (pokemon_pack_cost * pokemon_packs) + 
  (baseball_deck_cost * baseball_decks)

-- Theorem statement
theorem fred_card_spending : total_cost = 18.42 := by
  sorry

end fred_card_spending_l1602_160225


namespace probability_two_females_l1602_160262

/-- The probability of selecting two females from a group of contestants -/
theorem probability_two_females (total : ℕ) (females : ℕ) (males : ℕ) :
  total = females + males →
  females = 5 →
  males = 3 →
  (Nat.choose females 2 : ℚ) / (Nat.choose total 2 : ℚ) = 5 / 14 := by
  sorry

end probability_two_females_l1602_160262


namespace smallest_prime_factors_difference_l1602_160252

theorem smallest_prime_factors_difference (n : Nat) (h : n = 296045) :
  ∃ (p q : Nat), Prime p ∧ Prime q ∧ p < q ∧ p ∣ n ∧ q ∣ n ∧
  (∀ r : Nat, Prime r → r ∣ n → p ≤ r) ∧
  (∀ r : Nat, Prime r → r ∣ n → r ≠ p → q ≤ r) ∧
  q - p = 4 :=
by sorry

end smallest_prime_factors_difference_l1602_160252


namespace max_crosses_4x10_impossible_5x10_l1602_160213

/-- Represents a rectangular table with crosses placed in its cells -/
structure CrossTable (m n : ℕ) :=
  (crosses : Fin m → Fin n → Bool)

/-- Checks if the number of crosses in a given row is odd -/
def has_odd_row_crosses (t : CrossTable m n) (row : Fin m) : Prop :=
  (Finset.filter (λ col => t.crosses row col) (Finset.univ : Finset (Fin n))).card % 2 = 1

/-- Checks if the number of crosses in a given column is odd -/
def has_odd_col_crosses (t : CrossTable m n) (col : Fin n) : Prop :=
  (Finset.filter (λ row => t.crosses row col) (Finset.univ : Finset (Fin m))).card % 2 = 1

/-- Checks if all rows and columns have odd number of crosses -/
def all_odd_crosses (t : CrossTable m n) : Prop :=
  (∀ row, has_odd_row_crosses t row) ∧ (∀ col, has_odd_col_crosses t col)

/-- Counts the total number of crosses in the table -/
def total_crosses (t : CrossTable m n) : ℕ :=
  (Finset.filter (λ (row, col) => t.crosses row col) (Finset.univ : Finset (Fin m × Fin n))).card

theorem max_crosses_4x10 :
  (∃ (t : CrossTable 4 10), all_odd_crosses t ∧ total_crosses t = 30) ∧
  (∀ (t : CrossTable 4 10), all_odd_crosses t → total_crosses t ≤ 30) :=
sorry

theorem impossible_5x10 :
  ¬∃ (t : CrossTable 5 10), all_odd_crosses t :=
sorry

end max_crosses_4x10_impossible_5x10_l1602_160213


namespace initial_bacteria_count_l1602_160290

/-- The number of bacteria after a given number of tripling events -/
def bacteria_count (initial_count : ℕ) (tripling_events : ℕ) : ℕ :=
  initial_count * (3 ^ tripling_events)

/-- The number of tripling events in a given number of seconds -/
def tripling_events (seconds : ℕ) : ℕ :=
  seconds / 20

theorem initial_bacteria_count :
  ∃ (initial_count : ℕ),
    bacteria_count initial_count (tripling_events 180) = 275562 ∧
    initial_count = 14 :=
by
  sorry

end initial_bacteria_count_l1602_160290


namespace geometry_statements_l1602_160267

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def parallel (l1 l2 : Line) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def parallel_plane (p1 p2 : Plane) : Prop := sorry
def perpendicular_plane (p1 p2 : Plane) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry

theorem geometry_statements 
  (a b : Line) (α β : Plane) : 
  -- Statement 2
  (perpendicular a b ∧ perpendicular a α ∧ ¬contained_in b α → parallel b α) ∧
  -- Statement 3
  (perpendicular_plane α β ∧ perpendicular a α ∧ perpendicular b β → perpendicular a b) ∧
  -- Statement 1 (not necessarily true)
  ¬(parallel a b ∧ contained_in b α → parallel a α ∨ contained_in a α) ∧
  -- Statement 4 (not necessarily true)
  ¬(skew a b ∧ contained_in a α ∧ contained_in b β → parallel_plane α β) :=
sorry

end geometry_statements_l1602_160267


namespace arithmetic_calculation_l1602_160201

theorem arithmetic_calculation : (4 + 4 + 6) / 3 - 2 / 3 = 4 := by
  sorry

end arithmetic_calculation_l1602_160201


namespace inequalities_for_negative_fractions_l1602_160205

theorem inequalities_for_negative_fractions (a b : ℝ) 
  (h1 : -1 < a) (h2 : a < b) (h3 : b < 0) : 
  (1 / a > 1 / b) ∧ 
  (a^2 + b^2 > 2*a*b) ∧ 
  (a + 1/a > b + 1/b) := by
  sorry

end inequalities_for_negative_fractions_l1602_160205


namespace triangle_problem_l1602_160289

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  AB : Real
  BC : Real
  AC : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  (2 * Real.sin t.B * Real.cos t.A = Real.sin (t.A + t.C)) →  -- Given condition
  (t.BC = 2) →  -- Given condition
  (1/2 * t.AB * t.AC * Real.sin t.A = Real.sqrt 3) →  -- Area condition
  (t.A = Real.pi / 3 ∧ t.AB = 2) := by  -- Conclusion
sorry  -- Proof is omitted

end triangle_problem_l1602_160289


namespace square_difference_formula_l1602_160282

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8 / 15) (h2 : x - y = 2 / 15) : 
  x^2 - y^2 = 16 / 225 := by
sorry

end square_difference_formula_l1602_160282


namespace sufficient_not_necessary_l1602_160239

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, (abs x = x → x^2 ≥ -x)) ∧
  (∃ x, x^2 ≥ -x ∧ abs x ≠ x) :=
by sorry

end sufficient_not_necessary_l1602_160239


namespace mountain_paths_theorem_l1602_160266

/-- The number of paths leading to the summit from the east side -/
def east_paths : ℕ := 3

/-- The number of paths leading to the summit from the west side -/
def west_paths : ℕ := 2

/-- The total number of paths leading to the summit -/
def total_paths : ℕ := east_paths + west_paths

/-- The number of different ways for tourists to go up and come down the mountain -/
def different_ways : ℕ := total_paths * total_paths

theorem mountain_paths_theorem : different_ways = 25 := by
  sorry

end mountain_paths_theorem_l1602_160266


namespace weekend_ice_cream_total_l1602_160299

/-- The total amount of ice cream consumed by 4 roommates over a weekend -/
def weekend_ice_cream_consumption (friday_total : ℝ) : ℝ :=
  let saturday_total := friday_total - (4 * 0.25)
  let sunday_total := 2 * saturday_total
  friday_total + saturday_total + sunday_total

/-- Theorem stating that the total ice cream consumption over the weekend is 10 pints -/
theorem weekend_ice_cream_total :
  weekend_ice_cream_consumption 3.25 = 10 := by
  sorry

#eval weekend_ice_cream_consumption 3.25

end weekend_ice_cream_total_l1602_160299


namespace existence_of_specific_values_l1602_160261

theorem existence_of_specific_values : ∃ (a b : ℝ), a * b = a^2 - a * b + b^2 ∧ a = 1 ∧ b = 1 := by
  sorry

end existence_of_specific_values_l1602_160261


namespace valid_sequences_count_l1602_160214

/-- Represents a coin arrangement in a circle -/
def CoinArrangement := List Bool

/-- Represents a move that flips two adjacent coins -/
def Move := Nat

/-- The number of coins in the circle -/
def numCoins : Nat := 8

/-- The number of moves in a sequence -/
def numMoves : Nat := 6

/-- Checks if a coin arrangement is alternating heads and tails -/
def isAlternating (arrangement : CoinArrangement) : Bool :=
  sorry

/-- Applies a move to a coin arrangement -/
def applyMove (arrangement : CoinArrangement) (move : Move) : CoinArrangement :=
  sorry

/-- Applies a sequence of moves to a coin arrangement -/
def applyMoveSequence (arrangement : CoinArrangement) (moves : List Move) : CoinArrangement :=
  sorry

/-- Counts the number of valid 6-move sequences -/
def countValidSequences : Nat :=
  sorry

theorem valid_sequences_count :
  countValidSequences = 7680 :=
sorry

end valid_sequences_count_l1602_160214


namespace window_purchase_savings_l1602_160283

/-- Calculates the cost of windows given the number of windows and the discount rule -/
def calculateCost (windowCount : ℕ) (windowPrice : ℕ) : ℕ :=
  (windowCount - windowCount / 3) * windowPrice

/-- Represents the window purchase scenario -/
theorem window_purchase_savings
  (windowPrice : ℕ)
  (daveWindowCount : ℕ)
  (dougWindowCount : ℕ)
  (h1 : windowPrice = 100)
  (h2 : daveWindowCount = 10)
  (h3 : dougWindowCount = 12) :
  calculateCost (daveWindowCount + dougWindowCount) windowPrice =
  calculateCost daveWindowCount windowPrice + calculateCost dougWindowCount windowPrice :=
by sorry

#eval calculateCost 22 100 -- Joint purchase
#eval calculateCost 10 100 + calculateCost 12 100 -- Separate purchases

end window_purchase_savings_l1602_160283


namespace find_t_value_l1602_160271

theorem find_t_value (s t : ℚ) 
  (eq1 : 15 * s + 7 * t = 210)
  (eq2 : t = 3 * s - 1) : 
  t = 205 / 12 := by
  sorry

end find_t_value_l1602_160271


namespace vector_parallel_solution_l1602_160211

/-- Represents a 2D vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Check if two vectors are parallel -/
def parallel (v w : Vec2D) : Prop :=
  ∃ (k : ℝ), v.x * w.y = k * v.y * w.x

theorem vector_parallel_solution (m : ℝ) : 
  let a : Vec2D := ⟨1, m⟩
  let b : Vec2D := ⟨2, 5⟩
  let c : Vec2D := ⟨m, 3⟩
  parallel (Vec2D.mk (a.x + c.x) (a.y + c.y)) (Vec2D.mk (a.x - b.x) (a.y - b.y)) →
  m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 := by
  sorry

end vector_parallel_solution_l1602_160211


namespace equation_solution_l1602_160263

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  2 - 9 / x + 9 / x^2 = 0 → 2 / x = 2 / 3 ∨ 2 / x = 4 / 3 := by
  sorry

end equation_solution_l1602_160263


namespace sphere_surface_area_l1602_160222

theorem sphere_surface_area (V : ℝ) (r : ℝ) (A : ℝ) : 
  V = 72 * Real.pi → 
  V = (4/3) * Real.pi * r^3 → 
  A = 4 * Real.pi * r^2 → 
  A = 36 * Real.pi * 2^(2/3) :=
by sorry

end sphere_surface_area_l1602_160222


namespace newspaper_conference_overlap_l1602_160236

theorem newspaper_conference_overlap (total : ℕ) (writers : ℕ) (editors : ℕ) 
  (h_total : total = 110)
  (h_writers : writers = 45)
  (h_editors : editors ≥ 39)
  (h_max_overlap : ∀ overlap : ℕ, overlap ≤ 26)
  (h_neither : ∀ overlap : ℕ, 2 * overlap = total - writers - editors + overlap) :
  ∃ overlap : ℕ, overlap = 26 ∧ 
    writers + editors - overlap + 2 * overlap = total ∧
    overlap = total - writers - editors + overlap :=
by sorry

end newspaper_conference_overlap_l1602_160236


namespace intersection_M_N_l1602_160229

-- Define the sets M and N
def M : Set ℝ := {x | (x + 2) * (x - 2) ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end intersection_M_N_l1602_160229


namespace car_trip_distance_l1602_160209

theorem car_trip_distance (D : ℝ) : D - (1/2) * D - (1/4) * ((1/2) * D) = 135 → D = 360 := by
  sorry

end car_trip_distance_l1602_160209


namespace geometric_progression_integers_l1602_160202

/-- A geometric progression with first term b and common ratio r -/
def GeometricProgression (b : ℤ) (r : ℚ) : ℕ → ℚ :=
  fun n => b * r ^ (n - 1)

/-- An arithmetic progression with first term a and common difference d -/
def ArithmeticProgression (a d : ℚ) : ℕ → ℚ :=
  fun n => a + (n - 1) * d

theorem geometric_progression_integers
  (b : ℤ) (r : ℚ) (a d : ℚ)
  (h_subset : ∀ n : ℕ, ∃ m : ℕ, GeometricProgression b r n = ArithmeticProgression a d m) :
  ∀ n : ℕ, ∃ k : ℤ, GeometricProgression b r n = k :=
sorry

end geometric_progression_integers_l1602_160202


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1602_160258

/-- An isosceles triangle with two sides measuring 4 and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∧ b = 9 ∧ c = 9) ∨ (a = 9 ∧ b = 4 ∧ c = 9) ∨ (a = 9 ∧ b = 9 ∧ c = 4) →
    a + b + c = 22

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof :
  ∃ (a b c : ℝ), isosceles_triangle_perimeter a b c :=
sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1602_160258


namespace sin_2x_derivative_l1602_160259

theorem sin_2x_derivative (x : ℝ) : 
  deriv (λ x => Real.sin (2 * x)) x = 2 * Real.cos (2 * x) := by
sorry

end sin_2x_derivative_l1602_160259


namespace outside_door_cost_l1602_160212

/-- Proves that the cost of each outside door is $20 -/
theorem outside_door_cost (bedroom_doors : ℕ) (outside_doors : ℕ) (total_cost : ℚ) :
  bedroom_doors = 3 →
  outside_doors = 2 →
  total_cost = 70 →
  ∃ (outside_door_cost : ℚ),
    outside_door_cost * outside_doors + (outside_door_cost / 2) * bedroom_doors = total_cost ∧
    outside_door_cost = 20 :=
by
  sorry

end outside_door_cost_l1602_160212


namespace tangent_perpendicular_max_derivative_decreasing_function_range_l1602_160248

noncomputable section

variables (a : ℝ)

def f (x : ℝ) : ℝ := a * x^2 - Real.exp x

def f_derivative (x : ℝ) : ℝ := 2 * a * x - Real.exp x

theorem tangent_perpendicular_max_derivative :
  f_derivative a 1 = 0 →
  ∀ x, f_derivative a x ≤ 0 :=
sorry

theorem decreasing_function_range :
  (∀ x₁ x₂, 0 ≤ x₁ → x₁ < x₂ → 
    f a x₂ + x₂ * (2 - 2 * Real.log 2) < f a x₁ + x₁ * (2 - 2 * Real.log 2)) →
  a ≤ 1 :=
sorry

end tangent_perpendicular_max_derivative_decreasing_function_range_l1602_160248


namespace johns_expenses_l1602_160279

/-- Given that John spent 40% of his earnings on rent and had 32% left over,
    prove that he spent 30% less on the dishwasher compared to the rent. -/
theorem johns_expenses (earnings : ℝ) (rent_percent : ℝ) (leftover_percent : ℝ)
  (h1 : rent_percent = 40)
  (h2 : leftover_percent = 32) :
  let dishwasher_percent := 100 - rent_percent - leftover_percent
  (rent_percent - dishwasher_percent) / rent_percent * 100 = 30 := by
  sorry

end johns_expenses_l1602_160279


namespace triangle_height_and_median_l1602_160274

-- Define the triangle ABC
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (4, 5)
def C : ℝ × ℝ := (0, 7)

-- Define the height from A to BC
def height_equation (x y : ℝ) : Prop := 2 * x - y - 6 = 0

-- Define the median from A to BC
def median_equation (x y : ℝ) : Prop := 6 * x + y - 18 = 0

theorem triangle_height_and_median :
  (∀ x y : ℝ, height_equation x y ↔ 
    (y - A.2) / (x - A.1) = -1 / (B.1 - C.1) / (B.2 - C.2)) ∧
  (∀ x y : ℝ, median_equation x y ↔ 
    (y - A.2) / (x - A.1) = ((B.2 + C.2) / 2 - A.2) / ((B.1 + C.1) / 2 - A.1)) :=
by sorry

end triangle_height_and_median_l1602_160274


namespace mei_age_l1602_160257

/-- Given the ages of Li, Zhang, Jung, and Mei, prove Mei's age is 13 --/
theorem mei_age (li_age zhang_age jung_age mei_age : ℕ) : 
  li_age = 12 →
  zhang_age = 2 * li_age →
  jung_age = zhang_age + 2 →
  mei_age = jung_age / 2 →
  mei_age = 13 := by
  sorry


end mei_age_l1602_160257


namespace exists_graph_clique_lt_chromatic_l1602_160238

/-- A graph type with vertices and edges -/
structure Graph where
  V : Type
  E : V → V → Prop

/-- The clique number of a graph -/
def cliqueNumber (G : Graph) : ℕ := sorry

/-- The chromatic number of a graph -/
def chromaticNumber (G : Graph) : ℕ := sorry

/-- Theorem: There exists a graph with clique number smaller than its chromatic number -/
theorem exists_graph_clique_lt_chromatic :
  ∃ (G : Graph), cliqueNumber G < chromaticNumber G := by sorry

end exists_graph_clique_lt_chromatic_l1602_160238


namespace inequality_equivalence_l1602_160207

theorem inequality_equivalence (x : ℝ) (h : x ≠ 2) :
  (x - 1) / (x - 2) ≤ 0 ↔ (x^3 - x^2 + x - 1) / (x - 2) ≤ 0 :=
by sorry

end inequality_equivalence_l1602_160207


namespace library_loans_l1602_160264

theorem library_loans (init_a init_b current_a current_b : ℕ) 
  (return_rate_a return_rate_b : ℚ) : 
  init_a = 75 → 
  init_b = 100 → 
  current_a = 54 → 
  current_b = 82 → 
  return_rate_a = 65 / 100 → 
  return_rate_b = 1 / 2 → 
  ∃ (loaned_a loaned_b : ℕ), 
    loaned_a + loaned_b = 96 ∧ 
    (init_a - current_a : ℚ) = (1 - return_rate_a) * loaned_a ∧
    (init_b - current_b : ℚ) = (1 - return_rate_b) * loaned_b :=
by sorry

end library_loans_l1602_160264


namespace mother_age_is_40_l1602_160251

/-- The age of the mother -/
def mother_age : ℕ := sorry

/-- The sum of the ages of the 7 children -/
def children_ages_sum : ℕ := sorry

/-- The age of the mother is equal to the sum of the ages of her 7 children -/
axiom mother_age_eq_children_sum : mother_age = children_ages_sum

/-- After 20 years, the sum of the ages of the children will be three times the age of the mother -/
axiom future_age_relation : children_ages_sum + 7 * 20 = 3 * (mother_age + 20)

theorem mother_age_is_40 : mother_age = 40 := by sorry

end mother_age_is_40_l1602_160251


namespace functional_inequality_l1602_160204

theorem functional_inequality (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  let f : ℝ → ℝ := λ x => (x^3 - x^2 - 1) / (2*x*(x-1))
  f x + f ((x-1)/x) ≥ 1 + x :=
by sorry

end functional_inequality_l1602_160204


namespace digit_79_is_2_l1602_160240

/-- The sequence of digits obtained by concatenating consecutive integers from 60 down to 1 -/
def digit_sequence : List Nat := sorry

/-- The 79th digit in the sequence -/
def digit_79 : Nat := sorry

/-- Theorem stating that the 79th digit in the sequence is 2 -/
theorem digit_79_is_2 : digit_79 = 2 := by sorry

end digit_79_is_2_l1602_160240


namespace max_value_w_l1602_160210

theorem max_value_w (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  ∃ (w_max : ℝ), w_max = 0 ∧ ∀ (w : ℝ), w = x^2 + y^2 - 8 * x → w ≤ w_max :=
by
  sorry

end max_value_w_l1602_160210


namespace complement_of_union_equals_specific_set_l1602_160227

-- Define the universal set U
def U : Set Int := {x | -3 < x ∧ x ≤ 4}

-- Define sets A and B
def A : Set Int := {-2, -1, 3}
def B : Set Int := {1, 2, 3}

-- State the theorem
theorem complement_of_union_equals_specific_set :
  (U \ (A ∪ B)) = {0, 4} := by sorry

end complement_of_union_equals_specific_set_l1602_160227


namespace min_value_theorem_l1602_160286

theorem min_value_theorem (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_eq : (2 / x) + (3 / y) + (5 / z) = 10) :
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (2 / a) + (3 / b) + (5 / c) = 10 →
  x^4 * y^3 * z^2 ≤ a^4 * b^3 * c^2 ∧
  x^4 * y^3 * z^2 = 390625 / 1296 := by
  sorry

end min_value_theorem_l1602_160286


namespace polynomial_division_remainder_l1602_160218

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^6 - 1) * (X^2 - 1) = (X^3 - 1) * q + (X^2 - 1) :=
sorry

end polynomial_division_remainder_l1602_160218


namespace max_distance_with_tire_switch_l1602_160280

/-- Given a car with front tires lasting 24000 km and rear tires lasting 36000 km,
    the maximum distance the car can travel by switching tires once is 48000 km. -/
theorem max_distance_with_tire_switch (front_tire_life rear_tire_life : ℕ) 
  (h1 : front_tire_life = 24000)
  (h2 : rear_tire_life = 36000) :
  ∃ (switch_point : ℕ), 
    switch_point ≤ front_tire_life ∧
    switch_point ≤ rear_tire_life ∧
    switch_point + min (front_tire_life - switch_point) (rear_tire_life - switch_point) = 48000 :=
by sorry

end max_distance_with_tire_switch_l1602_160280


namespace correction_amount_proof_l1602_160296

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | "penny" => 1
  | _ => 0

/-- Calculates the correction amount in cents given the counting errors -/
def correction_amount (y z w : ℕ) : ℤ :=
  15 * y + 4 * z - 15 * w

theorem correction_amount_proof (y z w : ℕ) :
  correction_amount y z w = 
    y * (coin_value "quarter" - coin_value "dime") +
    z * (coin_value "nickel" - coin_value "penny") -
    w * (coin_value "quarter" - coin_value "dime") :=
  sorry

end correction_amount_proof_l1602_160296


namespace base3_subtraction_l1602_160232

/-- Represents a number in base 3 as a list of digits (least significant first) -/
def Base3 : Type := List Nat

/-- Converts a base 3 number to a natural number -/
def to_nat (b : Base3) : Nat :=
  b.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Adds two base 3 numbers -/
def add (a b : Base3) : Base3 :=
  sorry

/-- Subtracts two base 3 numbers -/
def sub (a b : Base3) : Base3 :=
  sorry

theorem base3_subtraction :
  let a : Base3 := [0, 1, 0]  -- 10₃
  let b : Base3 := [1, 0, 1, 1]  -- 1101₃
  let c : Base3 := [2, 0, 1, 2]  -- 2102₃
  let d : Base3 := [2, 1, 2]  -- 212₃
  let result : Base3 := [1, 0, 1, 1]  -- 1101₃
  sub (add (add a b) c) d = result := by
  sorry

end base3_subtraction_l1602_160232


namespace square_root_fraction_equality_l1602_160285

theorem square_root_fraction_equality : 
  let x : ℝ := Real.sqrt (7 - 4 * Real.sqrt 3)
  (x^2 - 4*x + 5) / (x^2 - 4*x + 3) = 2 := by sorry

end square_root_fraction_equality_l1602_160285


namespace evaluate_expression_l1602_160231

theorem evaluate_expression : 3^2 * 4 * 6^3 * Nat.factorial 7 = 39191040 := by
  sorry

end evaluate_expression_l1602_160231


namespace pen_gain_percentage_l1602_160234

/-- 
Given that the selling price of 5 pens equals the cost price of 10 pens, 
prove that the gain percentage is 100%.
-/
theorem pen_gain_percentage (cost selling : ℝ) 
  (h : 5 * selling = 10 * cost) : 
  (selling - cost) / cost * 100 = 100 := by
  sorry

end pen_gain_percentage_l1602_160234


namespace polar_to_cartesian_l1602_160278

theorem polar_to_cartesian :
  let ρ : ℝ := 4
  let θ : ℝ := π / 3
  let x : ℝ := ρ * Real.cos θ
  let y : ℝ := ρ * Real.sin θ
  (x = 2 ∧ y = 2 * Real.sqrt 3) := by sorry

end polar_to_cartesian_l1602_160278


namespace function_identity_implies_zero_function_l1602_160244

def IsPositive (n : ℤ) : Prop := n > 0

theorem function_identity_implies_zero_function 
  (f : ℤ → ℝ) 
  (h : ∀ (n m : ℤ), IsPositive n → IsPositive m → n ≥ m → 
       f (n + m) + f (n - m) = f (3 * n)) :
  ∀ (n : ℤ), IsPositive n → f n = 0 := by
sorry

end function_identity_implies_zero_function_l1602_160244


namespace book_arrangement_theorem_l1602_160254

def arrange_books (geom_copies : ℕ) (alg_copies : ℕ) : ℕ :=
  let total_slots := geom_copies + alg_copies - 1
  let remaining_geom := geom_copies - 2
  (total_slots.choose remaining_geom) * 2

theorem book_arrangement_theorem :
  arrange_books 4 5 = 112 :=
sorry

end book_arrangement_theorem_l1602_160254


namespace repeating_decimal_as_fraction_l1602_160268

/-- The decimal representation of 0.7888... -/
def repeating_decimal : ℚ := 0.7 + (8 / 9) / 10

theorem repeating_decimal_as_fraction :
  repeating_decimal = 71 / 90 := by sorry

end repeating_decimal_as_fraction_l1602_160268


namespace congruence_problem_l1602_160220

theorem congruence_problem (x : ℤ) : 
  (5 * x + 3) % 18 = 1 → (3 * x + 8) % 18 = 14 := by
  sorry

end congruence_problem_l1602_160220


namespace people_per_column_l1602_160219

theorem people_per_column (total_people : ℕ) 
  (h1 : total_people / 60 = 8) 
  (h2 : total_people % 16 = 0) : 
  total_people / 16 = 30 := by
sorry

end people_per_column_l1602_160219


namespace path_area_l1602_160230

/-- Calculates the area of a path surrounding a rectangular field -/
theorem path_area (field_length field_width path_width : ℝ) :
  field_length = 85 ∧ 
  field_width = 55 ∧ 
  path_width = 2.5 → 
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - 
  field_length * field_width = 725 := by
  sorry

end path_area_l1602_160230


namespace inequality_solution_set_l1602_160237

def solution_set (x : ℝ) : Prop :=
  x ∈ Set.union (Set.Ioo 0 1) (Set.union (Set.Ioo 1 (2 ^ (5/7))) (Set.Ioi 4))

theorem inequality_solution_set (x : ℝ) :
  (|1 / Real.log (1/2 * x) + 2| > 3/2) ↔ solution_set x :=
sorry

end inequality_solution_set_l1602_160237


namespace f_range_l1602_160249

def f (x : ℝ) : ℝ := 256 * x^9 - 576 * x^7 + 432 * x^5 - 120 * x^3 + 9 * x

theorem f_range :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ∈ Set.Icc (-1 : ℝ) 1) ∧
  (∃ x₁ ∈ Set.Icc (-1 : ℝ) 1, f x₁ = -1) ∧
  (∃ x₂ ∈ Set.Icc (-1 : ℝ) 1, f x₂ = 1) :=
sorry

end f_range_l1602_160249


namespace triangle_side_length_l1602_160256

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions for a valid triangle here
  True

-- Define the right angle at A
def RightAngleAtA (A B C : ℝ × ℝ) : Prop :=
  -- Add condition for right angle at A
  True

-- Define the length of a side
def Length (P Q : ℝ × ℝ) : ℝ :=
  -- Add definition for length between two points
  0

-- Define tangent of an angle
def Tan (A B C : ℝ × ℝ) : ℝ :=
  -- Add definition for tangent of angle C
  0

-- Define cosine of an angle
def Cos (A B C : ℝ × ℝ) : ℝ :=
  -- Add definition for cosine of angle B
  0

theorem triangle_side_length 
  (A B C : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_right_angle : RightAngleAtA A B C)
  (h_BC_length : Length B C = 10)
  (h_tan_cos : Tan A B C = 3 * Cos A B C) :
  Length A B = 20 * Real.sqrt 2 / 3 :=
sorry

end triangle_side_length_l1602_160256


namespace no_xy_term_l1602_160253

-- Define the expression as a function of x, y, and a
def expression (x y a : ℝ) : ℝ :=
  2 * (x^2 - x*y + y^2) - (3*x^2 - a*x*y + y^2)

-- Theorem statement
theorem no_xy_term (a : ℝ) :
  (∀ x y : ℝ, ∃ k : ℝ, expression x y a = -x^2 + k + y^2) ↔ a = 2 := by
  sorry

end no_xy_term_l1602_160253


namespace age_ratio_in_two_years_l1602_160294

def son_age : ℕ := 20
def man_age : ℕ := son_age + 22

def son_age_in_two_years : ℕ := son_age + 2
def man_age_in_two_years : ℕ := man_age + 2

theorem age_ratio_in_two_years :
  man_age_in_two_years / son_age_in_two_years = 2 :=
by sorry

end age_ratio_in_two_years_l1602_160294


namespace smallest_four_digit_divisible_by_53_ending_3_l1602_160223

theorem smallest_four_digit_divisible_by_53_ending_3 :
  ∀ n : ℕ,
  1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n % 10 = 3 →
  1113 ≤ n :=
by
  sorry

end smallest_four_digit_divisible_by_53_ending_3_l1602_160223


namespace sinA_cosA_rational_l1602_160270

/-- An isosceles triangle with integer base and height -/
structure IsoscelesTriangle where
  base : ℤ
  height : ℤ

/-- The sine of angle A in an isosceles triangle -/
def sinA (t : IsoscelesTriangle) : ℚ :=
  4 * t.base * t.height^2 / (4 * t.height^2 + t.base^2)

/-- The cosine of angle A in an isosceles triangle -/
def cosA (t : IsoscelesTriangle) : ℚ :=
  (4 * t.height^2 - t.base^2) / (4 * t.height^2 + t.base^2)

/-- Theorem: In an isosceles triangle with integer base and height, 
    both sin A and cos A are rational numbers -/
theorem sinA_cosA_rational (t : IsoscelesTriangle) : 
  (∃ q : ℚ, sinA t = q) ∧ (∃ q : ℚ, cosA t = q) := by
  sorry

end sinA_cosA_rational_l1602_160270


namespace lowest_number_of_students_twenty_four_divisible_lowest_number_is_twenty_four_l1602_160255

theorem lowest_number_of_students (n : ℕ) : n > 0 ∧ 8 ∣ n ∧ 12 ∣ n → n ≥ 24 := by
  sorry

theorem twenty_four_divisible : 8 ∣ 24 ∧ 12 ∣ 24 := by
  sorry

theorem lowest_number_is_twenty_four : ∃ n : ℕ, n > 0 ∧ 8 ∣ n ∧ 12 ∣ n ∧ n = 24 := by
  sorry

end lowest_number_of_students_twenty_four_divisible_lowest_number_is_twenty_four_l1602_160255


namespace problem_solution_l1602_160233

theorem problem_solution (a b : ℚ) :
  (∀ x y : ℚ, y = a + b / x) →
  (2 = a + b / (-2)) →
  (3 = a + b / (-6)) →
  a + b = 13/2 := by
sorry

end problem_solution_l1602_160233


namespace hike_attendance_l1602_160273

theorem hike_attendance (num_cars num_taxis num_vans : ℕ) 
                        (people_per_car people_per_taxi people_per_van : ℕ) : 
  num_cars = 3 → 
  num_taxis = 6 → 
  num_vans = 2 → 
  people_per_car = 4 → 
  people_per_taxi = 6 → 
  people_per_van = 5 → 
  num_cars * people_per_car + num_taxis * people_per_taxi + num_vans * people_per_van = 58 := by
  sorry


end hike_attendance_l1602_160273


namespace fish_and_shrimp_prices_l1602_160226

/-- The regular price of fish per pound -/
def regular_fish_price : ℝ := 10

/-- The discounted price of fish per quarter-pound package -/
def discounted_fish_price : ℝ := 1.5

/-- The price of shrimp per half-pound -/
def shrimp_price : ℝ := 5

/-- The discount rate on fish -/
def discount_rate : ℝ := 0.6

theorem fish_and_shrimp_prices :
  (regular_fish_price * (1 - discount_rate) / 4 = discounted_fish_price) ∧
  (regular_fish_price = 2 * shrimp_price) :=
sorry

end fish_and_shrimp_prices_l1602_160226


namespace third_month_sales_l1602_160277

def sales_1 : ℕ := 5400
def sales_2 : ℕ := 9000
def sales_4 : ℕ := 7200
def sales_5 : ℕ := 4500
def sales_6 : ℕ := 1200
def average_sale : ℕ := 5600
def num_months : ℕ := 6

theorem third_month_sales :
  ∃ (sales_3 : ℕ),
    sales_3 = num_months * average_sale - (sales_1 + sales_2 + sales_4 + sales_5 + sales_6) ∧
    sales_3 = 6300 := by
  sorry

end third_month_sales_l1602_160277


namespace triangle_abc_properties_l1602_160287

open Real

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  c * cos (π - B) = (b - 2 * a) * sin (π / 2 - C) →
  c = sqrt 13 →
  b = 3 →
  C = π / 3 ∧ 
  (1 / 2) * a * b * sin C = 3 * sqrt 3 :=
by sorry

end triangle_abc_properties_l1602_160287


namespace larger_solution_of_quadratic_l1602_160288

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 36 = 0 → x = 9 ∨ x = 4 → 9 > 4 := by
  sorry

end larger_solution_of_quadratic_l1602_160288


namespace joan_initial_books_l1602_160284

/-- The number of books Joan initially gathered -/
def initial_books : ℕ := sorry

/-- The number of additional books Joan found -/
def additional_books : ℕ := 26

/-- The total number of books Joan has now -/
def total_books : ℕ := 59

/-- Theorem stating that the initial number of books is 33 -/
theorem joan_initial_books : 
  initial_books = total_books - additional_books :=
by sorry

end joan_initial_books_l1602_160284


namespace initial_milk_water_ratio_l1602_160217

/-- Given a mixture of milk and water, proves that the initial ratio is 4:1 --/
theorem initial_milk_water_ratio 
  (total_volume : ℝ) 
  (added_water : ℝ) 
  (final_ratio : ℝ) 
  (h1 : total_volume = 45) 
  (h2 : added_water = 11) 
  (h3 : final_ratio = 1.8) : 
  ∃ (milk water : ℝ), 
    milk + water = total_volume ∧ 
    milk / (water + added_water) = final_ratio ∧ 
    milk / water = 4 := by
sorry

end initial_milk_water_ratio_l1602_160217


namespace triangle_is_obtuse_l1602_160224

theorem triangle_is_obtuse (a b c : ℝ) (h_a : a = 5) (h_b : b = 6) (h_c : c = 8) :
  c^2 > a^2 + b^2 := by
  sorry

#check triangle_is_obtuse

end triangle_is_obtuse_l1602_160224


namespace pittsburgh_police_stations_count_l1602_160246

/-- The number of police stations in Pittsburgh -/
def pittsburgh_police_stations : ℕ := 20

/-- The number of stores in Pittsburgh -/
def pittsburgh_stores : ℕ := 2000

/-- The number of hospitals in Pittsburgh -/
def pittsburgh_hospitals : ℕ := 500

/-- The number of schools in Pittsburgh -/
def pittsburgh_schools : ℕ := 200

/-- The total number of buildings in the new city -/
def new_city_total_buildings : ℕ := 2175

theorem pittsburgh_police_stations_count :
  pittsburgh_police_stations = 20 :=
by
  have new_city_stores : ℕ := pittsburgh_stores / 2
  have new_city_hospitals : ℕ := pittsburgh_hospitals * 2
  have new_city_schools : ℕ := pittsburgh_schools - 50
  have new_city_police_stations : ℕ := pittsburgh_police_stations + 5
  
  have : new_city_stores + new_city_hospitals + new_city_schools + new_city_police_stations = new_city_total_buildings :=
    by sorry
  
  sorry -- The proof goes here

end pittsburgh_police_stations_count_l1602_160246


namespace oliver_money_theorem_l1602_160298

def oliver_money_problem (initial savings frisbee puzzle gift : ℕ) : Prop :=
  initial + savings - frisbee - puzzle + gift = 15

theorem oliver_money_theorem :
  oliver_money_problem 9 5 4 3 8 := by
  sorry

end oliver_money_theorem_l1602_160298


namespace r_equals_1464_when_n_is_1_l1602_160245

/-- Given the conditions for r and s, prove that r equals 1464 when n is 1 -/
theorem r_equals_1464_when_n_is_1 (n : ℕ) (s r : ℕ) 
  (h1 : s = 4^n + 2) 
  (h2 : r = 2 * 3^s + s) 
  (h3 : n = 1) : 
  r = 1464 := by
  sorry

end r_equals_1464_when_n_is_1_l1602_160245


namespace polar_to_cartesian_circle_l1602_160221

/-- Polar to Cartesian conversion theorem for ρ = 4cosθ -/
theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 4 * Real.cos θ) ∧ 
  (x = ρ * Real.cos θ) ∧ 
  (y = ρ * Real.sin θ) →
  (x - 2)^2 + y^2 = 4 :=
by sorry

end polar_to_cartesian_circle_l1602_160221


namespace original_average_calc_l1602_160203

/-- Given a set of 10 numbers, if increasing one number by 6 changes the average to 6.8,
    then the original average was 6.2 -/
theorem original_average_calc (S : Finset ℝ) (original_sum : ℝ) :
  Finset.card S = 10 →
  (original_sum + 6) / 10 = 6.8 →
  original_sum / 10 = 6.2 :=
by sorry

end original_average_calc_l1602_160203


namespace two_lines_with_45_degree_angle_l1602_160250

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Represents a point in 3D space -/
structure Point3D where
  -- Add necessary fields

/-- Calculates the angle between a line and a plane -/
def angle_line_plane (l : Line3D) (p : Plane3D) : Real :=
  sorry

/-- Calculates the angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : Real :=
  sorry

/-- Checks if a line passes through a point -/
def line_passes_through (l : Line3D) (p : Point3D) : Prop :=
  sorry

theorem two_lines_with_45_degree_angle 
  (a : Line3D) (α : Plane3D) (P : Point3D) 
  (h : angle_line_plane a α = 30) : 
  ∃! (l1 l2 : Line3D), 
    l1 ≠ l2 ∧
    line_passes_through l1 P ∧
    line_passes_through l2 P ∧
    angle_between_lines l1 a = 45 ∧
    angle_between_lines l2 a = 45 ∧
    angle_line_plane l1 α = 45 ∧
    angle_line_plane l2 α = 45 ∧
    (∀ l : Line3D, 
      line_passes_through l P ∧ 
      angle_between_lines l a = 45 ∧ 
      angle_line_plane l α = 45 → 
      l = l1 ∨ l = l2) :=
by
  sorry

end two_lines_with_45_degree_angle_l1602_160250


namespace right_triangle_area_l1602_160206

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 = 64) (h2 : b^2 = 49) (h3 : c^2 = 113) 
  (h4 : a^2 + b^2 = c^2) : (1/2) * a * b = 28 := by
  sorry

end right_triangle_area_l1602_160206


namespace cubic_expression_value_l1602_160297

theorem cubic_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : a^3 + 2*a^2 + 3 = 4 := by
  sorry

end cubic_expression_value_l1602_160297


namespace four_consecutive_primes_sum_l1602_160292

theorem four_consecutive_primes_sum (A B : ℕ) : 
  (A > 0) → 
  (B > 0) → 
  (Nat.Prime A) → 
  (Nat.Prime B) → 
  (Nat.Prime (A - B)) → 
  (Nat.Prime (A + B)) → 
  (∃ p q r s : ℕ, 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧
    q = p + 2 ∧ r = q + 2 ∧ s = r + 2 ∧
    ((A = p ∧ B = q) ∨ (A = q ∧ B = p) ∨ (A = r ∧ B = p) ∨ (A = s ∧ B = p))) →
  p + q + r + s = 17 :=
sorry

end four_consecutive_primes_sum_l1602_160292


namespace team_selection_count_l1602_160243

/-- The number of ways to select a team of 8 students from 10 boys and 12 girls, with at least 4 girls -/
def select_team (total_boys : ℕ) (total_girls : ℕ) (team_size : ℕ) (min_girls : ℕ) : ℕ :=
  (Nat.choose total_girls min_girls * Nat.choose total_boys (team_size - min_girls)) +
  (Nat.choose total_girls (min_girls + 1) * Nat.choose total_boys (team_size - min_girls - 1)) +
  (Nat.choose total_girls (min_girls + 2) * Nat.choose total_boys (team_size - min_girls - 2)) +
  (Nat.choose total_girls (min_girls + 3) * Nat.choose total_boys (team_size - min_girls - 3)) +
  (Nat.choose total_girls (min_girls + 4))

theorem team_selection_count :
  select_team 10 12 8 4 = 245985 :=
by sorry

end team_selection_count_l1602_160243


namespace subtraction_and_simplification_l1602_160295

theorem subtraction_and_simplification : (8 : ℚ) / 23 - (5 : ℚ) / 46 = (11 : ℚ) / 46 := by
  sorry

end subtraction_and_simplification_l1602_160295


namespace union_of_M_and_N_l1602_160269

def M : Set ℕ := {1, 2}
def N : Set ℕ := {n | ∃ a ∈ M, n = 2*a - 1}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by sorry

end union_of_M_and_N_l1602_160269


namespace kendall_driving_distance_l1602_160291

/-- The distance Kendall drove with her father -/
def distance_with_father (total_distance mother_distance : ℝ) : ℝ :=
  total_distance - mother_distance

/-- Theorem: Kendall drove 0.50 miles with her father -/
theorem kendall_driving_distance :
  distance_with_father 0.67 0.17 = 0.50 := by
  sorry

end kendall_driving_distance_l1602_160291


namespace complex_set_is_line_l1602_160241

/-- The set of complex numbers z such that (3+4i)z is real forms a line in the complex plane. -/
theorem complex_set_is_line : 
  let S : Set ℂ := {z | ∃ r : ℝ, (3 + 4*I) * z = r}
  ∃ a b : ℝ, S = {z | z.re = a * z.im + b} :=
sorry

end complex_set_is_line_l1602_160241


namespace interest_difference_implies_principal_l1602_160200

/-- Proves that for a given principal amount, interest rate, and time period,
    if the difference between compound and simple interest is 12,
    then the principal amount is 1200. -/
theorem interest_difference_implies_principal
  (P : ℝ)  -- Principal amount
  (r : ℝ)  -- Interest rate (as a decimal)
  (t : ℝ)  -- Time period in years
  (h1 : r = 0.1)  -- Interest rate is 10%
  (h2 : t = 2)    -- Time period is 2 years
  (h3 : P * (1 + r)^t - P - (P * r * t) = 12)  -- Difference between CI and SI is 12
  : P = 1200 :=
by sorry

end interest_difference_implies_principal_l1602_160200


namespace simplify_expression_l1602_160281

theorem simplify_expression : (625 : ℝ) ^ (1/4) * (225 : ℝ) ^ (1/2) = 75 := by
  sorry

end simplify_expression_l1602_160281


namespace one_fourth_of_six_point_three_l1602_160242

theorem one_fourth_of_six_point_three (x : ℚ) : x = 6.3 / 4 → x = 63 / 40 := by
  sorry

end one_fourth_of_six_point_three_l1602_160242


namespace min_value_theorem_l1602_160235

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (1 / (a + 1) + 4 / (b + 1)) ≥ 9/4 ∧
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 2 ∧ 1 / (a₀ + 1) + 4 / (b₀ + 1) = 9/4 :=
sorry

end min_value_theorem_l1602_160235


namespace opposite_of_2023_l1602_160293

theorem opposite_of_2023 : Int.neg 2023 = -2023 := by
  sorry

end opposite_of_2023_l1602_160293


namespace det_of_specific_matrix_l1602_160272

theorem det_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; 6, 3]
  Matrix.det A = 33 := by sorry

end det_of_specific_matrix_l1602_160272


namespace range_of_m_for_sufficient_not_necessary_condition_l1602_160275

/-- The range of m for which ¬p is a sufficient but not necessary condition for ¬q -/
theorem range_of_m_for_sufficient_not_necessary_condition 
  (p : ℝ → Prop) (q : ℝ → ℝ → Prop) (m : ℝ) : 
  (∀ x, p x ↔ x^2 - 8*x - 20 ≤ 0) →
  (∀ x, q x m ↔ x^2 - 2*x + 1 - m^2 ≤ 0) →
  m > 0 →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  0 < m ∧ m ≤ 3 :=
sorry

end range_of_m_for_sufficient_not_necessary_condition_l1602_160275


namespace circle_center_l1602_160215

/-- A circle passes through (0,1) and is tangent to y = x^3 at (1,1). Its center is (1/2, 7/6). -/
theorem circle_center (c : ℝ × ℝ) : 
  (∃ r : ℝ, (c.1 - 0)^2 + (c.2 - 1)^2 = r^2 ∧ (c.1 - 1)^2 + (c.2 - 1)^2 = r^2) →  -- circle passes through (0,1) and (1,1)
  (∃ t : ℝ, t ≠ 1 → (t^3 - 1) / (t - 1) = 3 * (t - 1)) →                        -- tangent to y = x^3 at (1,1)
  c = (1/2, 7/6) :=
by sorry

end circle_center_l1602_160215


namespace not_p_necessary_not_sufficient_for_q_l1602_160265

-- Define propositions p and q
def p (x : ℝ) : Prop := -1 < x ∧ x < 3
def q (x : ℝ) : Prop := x > 5

-- Theorem statement
theorem not_p_necessary_not_sufficient_for_q :
  (∀ x, q x → ¬(p x)) ∧ 
  ¬(∀ x, ¬(p x) → q x) :=
by sorry

end not_p_necessary_not_sufficient_for_q_l1602_160265


namespace least_integer_satisfying_inequality_l1602_160228

theorem least_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), |y^2 + 3*y + 10| ≤ 25 - y → x ≤ y) ∧
             |x^2 + 3*x + 10| ≤ 25 - x ∧
             x = -5 := by
  sorry

end least_integer_satisfying_inequality_l1602_160228


namespace quartic_polynomial_sum_l1602_160208

/-- A quartic polynomial with specific properties -/
structure QuarticPolynomial (k : ℝ) where
  P : ℝ → ℝ
  is_quartic : ∃ (a b c d : ℝ), ∀ x, P x = a * x^4 + b * x^3 + c * x^2 + d * x + k
  at_zero : P 0 = k
  at_one : P 1 = 3 * k
  at_neg_one : P (-1) = 4 * k

/-- The sum of the polynomial evaluated at 2 and -2 equals 82k -/
theorem quartic_polynomial_sum (k : ℝ) (p : QuarticPolynomial k) :
  p.P 2 + p.P (-2) = 82 * k := by
  sorry

end quartic_polynomial_sum_l1602_160208
