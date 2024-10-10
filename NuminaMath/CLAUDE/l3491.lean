import Mathlib

namespace unique_solution_system_l3491_349109

theorem unique_solution_system (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x * (x + y + z) = 26 ∧ y * (x + y + z) = 27 ∧ z * (x + y + z) = 28 →
  x = 26 / 9 ∧ y = 3 ∧ z = 28 / 9 := by
  sorry

end unique_solution_system_l3491_349109


namespace three_card_selections_count_l3491_349134

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- The number of ways to select and order three different cards from a standard deck -/
def ThreeCardSelections : ℕ := StandardDeck * (StandardDeck - 1) * (StandardDeck - 2)

/-- Theorem: The number of ways to select and order three different cards from a standard 52-card deck is 132600 -/
theorem three_card_selections_count : ThreeCardSelections = 132600 := by
  sorry

end three_card_selections_count_l3491_349134


namespace student_path_probability_l3491_349146

/-- Represents a point on the city map -/
structure Point where
  east : Nat
  south : Nat

/-- Calculates the number of paths between two points -/
def num_paths (start finish : Point) : Nat :=
  Nat.choose (finish.east - start.east + finish.south - start.south) (finish.east - start.east)

/-- The probability of choosing a specific path -/
def path_probability (start finish : Point) : ℚ :=
  1 / 2 ^ (finish.east - start.east + finish.south - start.south)

theorem student_path_probability :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨4, 3⟩
  let C : Point := ⟨2, 1⟩
  let D : Point := ⟨3, 2⟩
  let total_paths := num_paths A B
  let paths_through_C_and_D := num_paths A C * num_paths C D * num_paths D B
  paths_through_C_and_D / total_paths = 12 / 35 := by
  sorry

#eval num_paths ⟨0, 0⟩ ⟨4, 3⟩  -- Should output 35
#eval num_paths ⟨0, 0⟩ ⟨2, 1⟩ * num_paths ⟨2, 1⟩ ⟨3, 2⟩ * num_paths ⟨3, 2⟩ ⟨4, 3⟩  -- Should output 12

end student_path_probability_l3491_349146


namespace bob_discount_percentage_l3491_349140

def bob_bill : ℝ := 30
def kate_bill : ℝ := 25
def total_after_discount : ℝ := 53

theorem bob_discount_percentage :
  let total_before_discount := bob_bill + kate_bill
  let discount_amount := total_before_discount - total_after_discount
  let discount_percentage := (discount_amount / bob_bill) * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |discount_percentage - 6.67| < ε :=
sorry

end bob_discount_percentage_l3491_349140


namespace complex_magnitude_problem_l3491_349177

theorem complex_magnitude_problem (z : ℂ) (h : (1 - I) * z - 3 * I = 1) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l3491_349177


namespace prime_sum_squares_l3491_349133

theorem prime_sum_squares (a b c d : ℕ) : 
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d ∧ 
  a > 3 ∧ b > 6 ∧ c > 12 ∧
  a^2 - b^2 + c^2 - d^2 = 1749 →
  a^2 + b^2 + c^2 + d^2 = 2143 := by
sorry

end prime_sum_squares_l3491_349133


namespace max_value_abc_l3491_349129

theorem max_value_abc (a b c : ℝ) (h : 2 * a + 3 * b + c = 6) :
  ∃ (max : ℝ), max = 9/2 ∧ ∀ (x y z : ℝ), 2 * x + 3 * y + z = 6 → x * y + x * z + y * z ≤ max :=
sorry

end max_value_abc_l3491_349129


namespace group_collection_theorem_l3491_349130

/-- Calculates the total collection amount in rupees for a group where each member
    contributes as many paise as there are members. -/
def totalCollectionInRupees (numberOfMembers : ℕ) : ℚ :=
  (numberOfMembers * numberOfMembers : ℚ) / 100

/-- Proves that for a group of 88 members, where each member contributes as many
    paise as there are members, the total collection amount is 77.44 rupees. -/
theorem group_collection_theorem :
  totalCollectionInRupees 88 = 77.44 := by
  sorry

#eval totalCollectionInRupees 88

end group_collection_theorem_l3491_349130


namespace complex_equation_roots_l3491_349135

theorem complex_equation_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = 1 - I ∧ z₂ = -3 + I ∧ 
  (z₁^2 + 2*z₁ = 3 - 4*I) ∧ 
  (z₂^2 + 2*z₂ = 3 - 4*I) := by
sorry

end complex_equation_roots_l3491_349135


namespace rectangle_tangent_circles_l3491_349141

/-- Given a rectangle ABCD with side lengths a and b, and two externally tangent circles
    inside the rectangle, one tangent to AB and AD, the other tangent to CB and CD,
    this theorem proves properties about the distance between circle centers and
    the locus of their tangency point. -/
theorem rectangle_tangent_circles
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let d := (Real.sqrt a - Real.sqrt b) ^ 2
  let m := min a b
  let p₁ := (a - m / 2, b - m / 2)
  let p₂ := (m / 2 + Real.sqrt (2 * a * b) - b, m / 2 + Real.sqrt (2 * a * b) - a)
  ∃ (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
    -- c₁ and c₂ are the centers of the circles
    -- r₁ and r₂ are the radii of the circles
    -- The circles are inside the rectangle
    c₁.1 ∈ Set.Icc 0 a ∧ c₁.2 ∈ Set.Icc 0 b ∧
    c₂.1 ∈ Set.Icc 0 a ∧ c₂.2 ∈ Set.Icc 0 b ∧
    -- The circles are tangent to the sides of the rectangle
    c₁.1 = r₁ ∧ c₁.2 = r₁ ∧
    c₂.1 = a - r₂ ∧ c₂.2 = b - r₂ ∧
    -- The circles are externally tangent to each other
    (c₁.1 - c₂.1) ^ 2 + (c₁.2 - c₂.2) ^ 2 = (r₁ + r₂) ^ 2 ∧
    -- The distance between the centers is d
    (c₁.1 - c₂.1) ^ 2 + (c₁.2 - c₂.2) ^ 2 = d ^ 2 ∧
    -- The locus of the tangency point is a line segment
    ∃ (t : ℝ), t ∈ Set.Icc 0 1 ∧
      let p := (1 - t) • p₁ + t • p₂
      p.1 = (r₁ * (a - r₁ - r₂)) / (r₁ + r₂) + r₁ ∧
      p.2 = (r₁ * (b - r₁ - r₂)) / (r₁ + r₂) + r₁ :=
by
  sorry

end rectangle_tangent_circles_l3491_349141


namespace three_toys_picked_l3491_349181

def toy_count : ℕ := 4

def probability_yo_yo_and_ball (n : ℕ) : ℚ :=
  if n < 2 then 0
  else (Nat.choose 2 (n - 2) : ℚ) / (Nat.choose toy_count n : ℚ)

theorem three_toys_picked :
  ∃ (n : ℕ), n ≤ toy_count ∧ probability_yo_yo_and_ball n = 1/2 ∧ n = 3 :=
sorry

end three_toys_picked_l3491_349181


namespace rich_walk_distance_l3491_349128

/-- Calculates the total distance Rich walks based on the given conditions -/
def total_distance : ℝ :=
  let initial_distance := 20 + 200
  let left_turn_distance := 2 * initial_distance
  let halfway_distance := initial_distance + left_turn_distance
  let final_distance := halfway_distance + 0.5 * halfway_distance
  2 * final_distance

/-- Theorem stating that the total distance Rich walks is 1980 feet -/
theorem rich_walk_distance : total_distance = 1980 := by
  sorry

end rich_walk_distance_l3491_349128


namespace decreasing_function_implies_a_greater_than_one_l3491_349127

/-- A linear function y = mx + b decreases if and only if its slope m is negative -/
axiom decreasing_linear_function (m b : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → m * x₁ + b > m * x₂ + b) ↔ m < 0

/-- For the function y = (1-a)x + 2, if it decreases as x increases, then a > 1 -/
theorem decreasing_function_implies_a_greater_than_one (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (1 - a) * x₁ + 2 > (1 - a) * x₂ + 2) → a > 1 := by
  sorry

end decreasing_function_implies_a_greater_than_one_l3491_349127


namespace mod_equiv_problem_l3491_349103

theorem mod_equiv_problem (m : ℕ) : 
  197 * 879 ≡ m [ZMOD 60] → 0 ≤ m → m < 60 → m = 3 := by
  sorry

end mod_equiv_problem_l3491_349103


namespace salary_increase_l3491_349155

theorem salary_increase (num_employees : ℕ) (avg_salary : ℚ) (manager_salary : ℚ) :
  num_employees = 20 ∧ avg_salary = 1500 ∧ manager_salary = 14100 →
  (((num_employees : ℚ) * avg_salary + manager_salary) / (num_employees + 1 : ℚ)) - avg_salary = 600 := by
sorry

end salary_increase_l3491_349155


namespace sol_earnings_l3491_349122

def candy_sales (day : Nat) : Nat :=
  10 + 4 * (day - 1)

def total_sales : Nat :=
  (List.range 6).map (λ i => candy_sales (i + 1)) |>.sum

def earnings_cents : Nat :=
  total_sales * 10

theorem sol_earnings :
  earnings_cents / 100 = 12 := by sorry

end sol_earnings_l3491_349122


namespace max_spheres_in_frustum_l3491_349132

/-- Represents a frustum with given height and two spheres placed inside it. -/
structure Frustum :=
  (height : ℝ)
  (sphere1_radius : ℝ)
  (sphere2_radius : ℝ)

/-- Calculates the maximum number of additional spheres that can be placed in the frustum. -/
def max_additional_spheres (f : Frustum) : ℕ :=
  sorry

/-- The main theorem stating the maximum number of additional spheres. -/
theorem max_spheres_in_frustum (f : Frustum) 
  (h1 : f.height = 8)
  (h2 : f.sphere1_radius = 2)
  (h3 : f.sphere2_radius = 3)
  : max_additional_spheres f = 2 := by
  sorry

end max_spheres_in_frustum_l3491_349132


namespace cubic_polynomials_common_roots_l3491_349169

theorem cubic_polynomials_common_roots :
  ∃! (a b : ℝ), 
    (∃ (r s : ℝ) (h : r ≠ s), 
      (∀ x : ℝ, x^3 + a*x^2 + 14*x + 7 = 0 ↔ x = r ∨ x = s ∨ x^3 + a*x^2 + 14*x + 7 = 0) ∧
      (∀ x : ℝ, x^3 + b*x^2 + 21*x + 15 = 0 ↔ x = r ∨ x = s ∨ x^3 + b*x^2 + 21*x + 15 = 0)) ∧
    a = 5 ∧ b = 4 := by
  sorry

end cubic_polynomials_common_roots_l3491_349169


namespace election_probability_l3491_349151

/-- Represents an election between two candidates -/
structure Election where
  p : ℕ  -- votes for candidate A
  q : ℕ  -- votes for candidate B
  h : p > q  -- condition that p > q

/-- 
The probability that candidate A's vote count is always greater than 
candidate B's throughout the counting process 
-/
noncomputable def winning_probability (e : Election) : ℚ :=
  (e.p - e.q : ℚ) / (e.p + e.q : ℚ)

/-- 
Theorem: The probability that candidate A's vote count is always greater than 
candidate B's throughout the counting process is (p - q) / (p + q) 
-/
theorem election_probability (e : Election) : 
  winning_probability e = (e.p - e.q : ℚ) / (e.p + e.q : ℚ) := by
  sorry

/-- Example for p = 3 and q = 2 -/
example : ∃ (e : Election), e.p = 3 ∧ e.q = 2 ∧ winning_probability e = 1/5 := by
  sorry

/-- Example for p = 1010 and q = 1009 -/
example : ∃ (e : Election), e.p = 1010 ∧ e.q = 1009 ∧ winning_probability e = 1/2019 := by
  sorry

end election_probability_l3491_349151


namespace blackboard_problem_l3491_349118

/-- Represents the state of the blackboard -/
structure BoardState where
  ones : ℕ
  twos : ℕ
  threes : ℕ
  fours : ℕ

/-- Represents a single operation on the blackboard -/
inductive Operation
  | erase_123_add_4
  | erase_124_add_3
  | erase_134_add_2
  | erase_234_add_1

/-- Applies an operation to a board state -/
def applyOperation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.erase_123_add_4 => 
      { ones := state.ones - 1, twos := state.twos - 1, 
        threes := state.threes - 1, fours := state.fours + 2 }
  | Operation.erase_124_add_3 => 
      { ones := state.ones - 1, twos := state.twos - 1, 
        threes := state.threes + 2, fours := state.fours - 1 }
  | Operation.erase_134_add_2 => 
      { ones := state.ones - 1, twos := state.twos + 2, 
        threes := state.threes - 1, fours := state.fours - 1 }
  | Operation.erase_234_add_1 => 
      { ones := state.ones + 2, twos := state.twos - 1, 
        threes := state.threes - 1, fours := state.fours - 1 }

/-- Checks if the board state is in a final state (only three numbers remain) -/
def isFinalState (state : BoardState) : Bool :=
  (state.ones + state.twos + state.threes + state.fours) = 3

/-- Calculates the product of the remaining numbers -/
def productOfRemaining (state : BoardState) : ℕ :=
  (if state.ones > 0 then 1^state.ones else 1) *
  (if state.twos > 0 then 2^state.twos else 1) *
  (if state.threes > 0 then 3^state.threes else 1) *
  (if state.fours > 0 then 4^state.fours else 1)

/-- The main theorem to prove -/
theorem blackboard_problem :
  ∃ (operations : List Operation),
    let initialState : BoardState := { ones := 11, twos := 22, threes := 33, fours := 44 }
    let finalState := operations.foldl applyOperation initialState
    isFinalState finalState ∧ productOfRemaining finalState = 12 := by
  sorry


end blackboard_problem_l3491_349118


namespace dvd_rack_sequence_l3491_349100

theorem dvd_rack_sequence (rack : Fin 6 → ℕ) 
  (h1 : rack 0 = 2)
  (h2 : rack 1 = 4)
  (h4 : rack 3 = 16)
  (h5 : rack 4 = 32)
  (h6 : rack 5 = 64)
  (h_double : ∀ i : Fin 5, rack (i.succ) = 2 * rack i) :
  rack 2 = 8 := by
  sorry

end dvd_rack_sequence_l3491_349100


namespace xiaohong_fruit_money_l3491_349114

/-- The price difference between 500g of apples and 500g of pears in yuan -/
def price_difference : ℚ := 55 / 100

/-- The amount saved when buying 5 kg of apples in yuan -/
def apple_savings : ℚ := 4

/-- The amount saved when buying 6 kg of pears in yuan -/
def pear_savings : ℚ := 3

/-- The price of 1 kg of pears in yuan -/
def pear_price : ℚ := 45 / 10

theorem xiaohong_fruit_money : 
  ∃ (total : ℚ), 
    total = 6 * pear_price - pear_savings ∧ 
    total = 5 * (pear_price + 2 * price_difference) - apple_savings ∧
    total = 24 := by
  sorry

end xiaohong_fruit_money_l3491_349114


namespace discount_doubles_with_time_ratio_two_l3491_349145

/-- Represents the true discount calculation for a bill -/
structure BillDiscount where
  face_value : ℝ
  initial_discount : ℝ
  time_ratio : ℝ

/-- Calculates the discount for a different time period based on the time ratio -/
def discount_for_different_time (bill : BillDiscount) : ℝ :=
  bill.initial_discount * bill.time_ratio

/-- Theorem stating that for a bill of 110 with initial discount of 10 and time ratio of 2,
    the discount for the different time period is 20 -/
theorem discount_doubles_with_time_ratio_two :
  let bill := BillDiscount.mk 110 10 2
  discount_for_different_time bill = 20 := by
  sorry

#eval discount_for_different_time (BillDiscount.mk 110 10 2)

end discount_doubles_with_time_ratio_two_l3491_349145


namespace value_of_x_l3491_349162

theorem value_of_x : (2009^2 - 2009) / 2009 = 2008 := by
  sorry

end value_of_x_l3491_349162


namespace largest_difference_in_S_l3491_349101

def S : Set ℤ := {-20, -8, 0, 6, 10, 15, 25}

theorem largest_difference_in_S : 
  ∀ (a b : ℤ), a ∈ S → b ∈ S → (a - b) ≤ 45 ∧ ∃ (x y : ℤ), x ∈ S ∧ y ∈ S ∧ x - y = 45 :=
by sorry

end largest_difference_in_S_l3491_349101


namespace deepak_present_age_l3491_349180

/-- Represents the ages of two people with a given ratio --/
structure AgeRatio where
  x : ℕ
  rahul_age : ℕ := 4 * x
  deepak_age : ℕ := 3 * x

/-- The theorem stating Deepak's present age given the conditions --/
theorem deepak_present_age (ar : AgeRatio) 
  (h1 : ar.rahul_age + 6 = 50) : 
  ar.deepak_age = 33 := by
  sorry

#check deepak_present_age

end deepak_present_age_l3491_349180


namespace balls_in_box_perfect_square_l3491_349192

theorem balls_in_box_perfect_square (a v : ℕ) : 
  (2 * a * v : ℚ) / ((a + v) * (a + v - 1) / 2) = 1 / 2 → 
  ∃ n : ℕ, a + v = n^2 := by
sorry

end balls_in_box_perfect_square_l3491_349192


namespace total_bulbs_needed_l3491_349183

theorem total_bulbs_needed (medium_lights : ℕ) (small_bulbs : ℕ) (medium_bulbs : ℕ) (large_bulbs : ℕ) :
  medium_lights = 12 →
  small_bulbs = 1 →
  medium_bulbs = 2 →
  large_bulbs = 3 →
  (medium_lights * small_bulbs + 10) * small_bulbs +
  medium_lights * medium_bulbs +
  (2 * medium_lights) * large_bulbs = 118 := by
  sorry

end total_bulbs_needed_l3491_349183


namespace inequality_solution_set_l3491_349136

theorem inequality_solution_set (x : ℝ) : 
  (1/2 - (x - 2)/3 > 1) ↔ (x < 1/2) := by sorry

end inequality_solution_set_l3491_349136


namespace purely_imaginary_complex_number_l3491_349110

theorem purely_imaginary_complex_number (x : ℝ) : 
  let z : ℂ := Complex.mk (x^2 - 2*x - 3) (x + 1)
  (z.re = 0 ∧ z.im ≠ 0) → x = 3 := by
  sorry

end purely_imaginary_complex_number_l3491_349110


namespace pascals_triangle_51st_row_third_number_l3491_349172

theorem pascals_triangle_51st_row_third_number : 
  (Nat.choose 51 2) = 1275 := by sorry

end pascals_triangle_51st_row_third_number_l3491_349172


namespace integral_equals_two_minus_three_ln_three_l3491_349196

/-- Given that the solution set of the inequality 1 - 3/(x+a) < 0 is (-1,2),
    prove that the integral from 0 to 2 of (1 - 3/(x+a)) dx equals 2 - 3 * ln 3 -/
theorem integral_equals_two_minus_three_ln_three 
  (a : ℝ) 
  (h : Set.Ioo (-1 : ℝ) 2 = {x : ℝ | 1 - 3 / (x + a) < 0}) : 
  ∫ x in (0:ℝ)..2, (1 - 3 / (x + a)) = 2 - 3 * Real.log 3 := by
  sorry

#check integral_equals_two_minus_three_ln_three

end integral_equals_two_minus_three_ln_three_l3491_349196


namespace flour_sack_cost_l3491_349173

/-- Represents the cost and customs scenario for flour sacks --/
structure FlourScenario where
  sack_cost : ℕ  -- Cost of one sack of flour in pesetas
  customs_duty : ℕ  -- Customs duty per sack in pesetas
  truck1_sacks : ℕ := 118  -- Number of sacks in first truck
  truck2_sacks : ℕ := 40   -- Number of sacks in second truck
  truck1_left : ℕ := 10    -- Sacks left by first truck
  truck2_left : ℕ := 4     -- Sacks left by second truck
  truck1_pay : ℕ := 800    -- Additional payment by first truck
  truck2_receive : ℕ := 800  -- Amount received by second truck

/-- The theorem stating the cost of each sack of flour --/
theorem flour_sack_cost (scenario : FlourScenario) : scenario.sack_cost = 1600 :=
  by
    have h1 : scenario.sack_cost * scenario.truck1_left + scenario.truck1_pay = 
              scenario.customs_duty * (scenario.truck1_sacks - scenario.truck1_left) := by sorry
    have h2 : scenario.sack_cost * scenario.truck2_left - scenario.truck2_receive = 
              scenario.customs_duty * (scenario.truck2_sacks - scenario.truck2_left) := by sorry
    sorry  -- The proof goes here

end flour_sack_cost_l3491_349173


namespace base_number_proof_l3491_349185

theorem base_number_proof (y : ℝ) (base : ℝ) 
  (h1 : 9^y = base^16) (h2 : y = 8) : base = 3 := by
  sorry

end base_number_proof_l3491_349185


namespace cubic_polynomial_uniqueness_l3491_349161

def is_monic_cubic (q : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, q x = x^3 + a*x^2 + b*x + c

theorem cubic_polynomial_uniqueness (q : ℝ → ℂ) :
  is_monic_cubic (λ x : ℝ ↦ (q x).re) →
  q (5 - 3*I) = 0 →
  q 0 = -80 →
  ∀ x, q x = x^3 - 10*x^2 + 40*x - 80 :=
by sorry

end cubic_polynomial_uniqueness_l3491_349161


namespace correct_calculation_l3491_349189

theorem correct_calculation : -7 + 3 = -4 := by
  sorry

end correct_calculation_l3491_349189


namespace quadratic_one_solution_positive_m_value_l3491_349107

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + m * x + 36 = 0) → m = 36 ∨ m = -36 :=
by sorry

theorem positive_m_value (m : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + m * x + 36 = 0) ∧ m > 0 → m = 36 :=
by sorry

end quadratic_one_solution_positive_m_value_l3491_349107


namespace ellipse_incenter_ratio_theorem_l3491_349193

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Represents the foci of an ellipse -/
structure Foci (e : Ellipse) where
  left : Point
  right : Point

/-- Represents a triangle formed by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Represents the incenter of a triangle -/
def incenter (t : Triangle) : Point :=
  sorry -- Definition of incenter

/-- Checks if a point is on a line segment -/
def isOnSegment (p : Point) (a : Point) (b : Point) : Prop :=
  sorry -- Definition of being on a line segment

theorem ellipse_incenter_ratio_theorem
  (e : Ellipse) (m : Point) (f : Foci e) (p n : Point) :
  isOnEllipse e m →
  p = incenter (Triangle.mk m f.left f.right) →
  isOnSegment n f.left f.right →
  isOnSegment n m p →
  (m.x - n.x)^2 + (m.y - n.y)^2 > 0 →
  (n.x - p.x)^2 + (n.y - p.y)^2 > 0 →
  ∃ (r : ℝ), r > 0 ∧
    r = (m.x - n.x)^2 + (m.y - n.y)^2 / ((n.x - p.x)^2 + (n.y - p.y)^2) ∧
    r = (m.x - f.left.x)^2 + (m.y - f.left.y)^2 / ((f.left.x - p.x)^2 + (f.left.y - p.y)^2) ∧
    r = (m.x - f.right.x)^2 + (m.y - f.right.y)^2 / ((f.right.x - p.x)^2 + (f.right.y - p.y)^2) :=
by
  sorry

end ellipse_incenter_ratio_theorem_l3491_349193


namespace volume_of_region_l3491_349174

def region (x y z : ℝ) : Prop :=
  abs (x + y + z) + abs (x + y - z) ≤ 12 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

theorem volume_of_region : 
  MeasureTheory.volume {p : ℝ × ℝ × ℝ | region p.1 p.2.1 p.2.2} = 108 := by
  sorry

end volume_of_region_l3491_349174


namespace min_value_theorem_l3491_349167

theorem min_value_theorem (x y : ℝ) (h : x * y > 0) :
  ∃ (min_val : ℝ), min_val = 4 - 2 * Real.sqrt 2 ∧
  ∀ (z : ℝ), y / (x + y) + 2 * x / (2 * x + y) ≥ min_val :=
sorry

end min_value_theorem_l3491_349167


namespace concentric_circles_chord_count_l3491_349139

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle ABC is 80 degrees, then the number of segments needed to return to the starting point is 18. -/
theorem concentric_circles_chord_count (angle_ABC : ℝ) (n : ℕ) : 
  angle_ABC = 80 → n * 100 = 360 * (n / 18) → n = 18 := by sorry

end concentric_circles_chord_count_l3491_349139


namespace kenny_book_purchase_l3491_349138

def lawn_price : ℕ := 15
def video_game_price : ℕ := 45
def book_price : ℕ := 5
def lawns_mowed : ℕ := 35
def video_games_wanted : ℕ := 5

def total_earned : ℕ := lawn_price * lawns_mowed
def video_games_cost : ℕ := video_game_price * video_games_wanted
def remaining_money : ℕ := total_earned - video_games_cost

theorem kenny_book_purchase :
  remaining_money / book_price = 60 := by sorry

end kenny_book_purchase_l3491_349138


namespace basketball_substitutions_l3491_349163

/-- The number of ways to make substitutions in a basketball game --/
def substitution_ways (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  let substitutes := total_players - starting_players
  let ways_0 := 1
  let ways_1 := starting_players * substitutes
  let ways_2 := ways_1 * (starting_players - 1) * (substitutes - 1)
  let ways_3 := ways_2 * (starting_players - 2) * (substitutes - 2)
  let ways_4 := ways_3 * (starting_players - 3) * (substitutes - 3)
  ways_0 + ways_1 + ways_2 + ways_3 + ways_4

/-- The main theorem about basketball substitutions --/
theorem basketball_substitutions :
  let total_ways := substitution_ways 15 5 4
  total_ways = 648851 ∧ total_ways % 100 = 51 := by
  sorry

#eval substitution_ways 15 5 4
#eval (substitution_ways 15 5 4) % 100

end basketball_substitutions_l3491_349163


namespace rhombus_side_length_l3491_349197

/-- The side length of a rhombus given its diagonal lengths -/
theorem rhombus_side_length (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  ∃ (side : ℝ), side = 13 ∧ side^2 = (d1/2)^2 + (d2/2)^2 := by sorry

end rhombus_side_length_l3491_349197


namespace hypotenuse_length_l3491_349190

-- Define the triangle
structure RightTriangle where
  a : ℝ  -- Length of one leg
  b : ℝ  -- Length of the other leg
  c : ℝ  -- Length of the hypotenuse
  right_angled : a^2 + b^2 = c^2  -- Pythagorean theorem
  sum_of_squares : a^2 + b^2 + c^2 = 2450
  hypotenuse_relation : c = b + 10

-- Theorem statement
theorem hypotenuse_length (t : RightTriangle) : t.c = 35 := by
  sorry

end hypotenuse_length_l3491_349190


namespace geometric_sequence_problem_l3491_349112

theorem geometric_sequence_problem (a b c : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ -2 = -2 * r ∧ a = -2 * r^2 ∧ b = -2 * r^3 ∧ c = -2 * r^4 ∧ -8 = -2 * r^5) →
  b = -4 ∧ a * c = 16 := by
sorry

end geometric_sequence_problem_l3491_349112


namespace fifteenth_term_binomial_expansion_l3491_349102

theorem fifteenth_term_binomial_expansion : 
  let n : ℕ := 20
  let k : ℕ := 14
  let z : ℂ := -1 + Complex.I
  Nat.choose n k * (-1)^(n - k) * Complex.I^k = -38760 := by sorry

end fifteenth_term_binomial_expansion_l3491_349102


namespace exists_72_degree_angle_l3491_349104

/-- Represents a hexagon with angles in arithmetic progression -/
structure ArithmeticHexagon where
  a : ℝ  -- First angle in the progression
  d : ℝ  -- Common difference

/-- The sum of angles in a hexagon is 720° -/
axiom hexagon_angle_sum (h : ArithmeticHexagon) : 
  h.a + (h.a + h.d) + (h.a + 2*h.d) + (h.a + 3*h.d) + (h.a + 4*h.d) + (h.a + 5*h.d) = 720

/-- Theorem: There exists a hexagon with angles in arithmetic progression that has a 72° angle -/
theorem exists_72_degree_angle : ∃ h : ArithmeticHexagon, 
  h.a = 72 ∨ (h.a + h.d) = 72 ∨ (h.a + 2*h.d) = 72 ∨ 
  (h.a + 3*h.d) = 72 ∨ (h.a + 4*h.d) = 72 ∨ (h.a + 5*h.d) = 72 :=
sorry

end exists_72_degree_angle_l3491_349104


namespace melanie_remaining_plums_l3491_349142

/-- The number of plums Melanie picked initially -/
def initial_plums : ℕ := 7

/-- The number of plums Melanie gave away -/
def plums_given_away : ℕ := 3

/-- Theorem: Melanie has 4 plums after giving some away -/
theorem melanie_remaining_plums : 
  initial_plums - plums_given_away = 4 := by sorry

end melanie_remaining_plums_l3491_349142


namespace trigonometric_identity_l3491_349152

theorem trigonometric_identity (α β : Real) (h : α + β = Real.pi / 3) :
  Real.tan α + Real.tan β + Real.sqrt 3 * Real.tan α * Real.tan β = Real.sqrt 3 := by
  sorry

end trigonometric_identity_l3491_349152


namespace sum_of_three_numbers_l3491_349166

theorem sum_of_three_numbers (a b c : ℝ) : 
  a ≤ b → b ≤ c → b = 10 → 
  (a + b + c) / 3 = a + 5 → 
  (a + b + c) / 3 = c - 20 → 
  a + b + c = 75 := by
sorry

end sum_of_three_numbers_l3491_349166


namespace greek_yogurt_cost_per_pack_l3491_349168

theorem greek_yogurt_cost_per_pack 
  (total_packs : ℕ)
  (expired_percentage : ℚ)
  (total_refund : ℚ)
  (h1 : total_packs = 80)
  (h2 : expired_percentage = 40 / 100)
  (h3 : total_refund = 384) :
  total_refund / (expired_percentage * total_packs) = 12 := by
sorry

end greek_yogurt_cost_per_pack_l3491_349168


namespace same_gender_probability_l3491_349182

/-- The probability of selecting two students of the same gender from a group of 3 male and 2 female students -/
theorem same_gender_probability (male_students female_students : ℕ) 
  (h1 : male_students = 3)
  (h2 : female_students = 2) : 
  (Nat.choose male_students 2 + Nat.choose female_students 2) / Nat.choose (male_students + female_students) 2 = 2 / 5 := by
  sorry

end same_gender_probability_l3491_349182


namespace points_collinear_l3491_349164

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the properties of the triangle
def is_acute_angled (t : Triangle) : Prop := sorry

-- Define the angle A to be 60°
def angle_A_is_60_degrees (t : Triangle) : Prop := sorry

-- Define the orthocenter H
def orthocenter (t : Triangle) : Point := sorry

-- Define point M
def point_M (t : Triangle) (H : Point) : Point := sorry

-- Define point N
def point_N (t : Triangle) (H : Point) : Point := sorry

-- Define the circumcenter O
def circumcenter (t : Triangle) : Point := sorry

-- Define collinearity
def collinear (P Q R S : Point) : Prop := sorry

-- Theorem statement
theorem points_collinear (t : Triangle) (H : Point) (M N O : Point) :
  is_acute_angled t →
  angle_A_is_60_degrees t →
  H = orthocenter t →
  M = point_M t H →
  N = point_N t H →
  O = circumcenter t →
  collinear M N H O :=
sorry

end points_collinear_l3491_349164


namespace number_of_selection_schemes_l3491_349117

/-- The number of people to choose from -/
def total_people : ℕ := 6

/-- The number of cities to visit -/
def total_cities : ℕ := 4

/-- The number of people who cannot visit a specific city -/
def restricted_people : ℕ := 2

/-- Calculates the number of ways to select people for cities with restrictions -/
def selection_schemes (n m r : ℕ) : ℕ :=
  (n.factorial / (n - m).factorial) - 2 * ((n - 1).factorial / (n - m).factorial)

/-- The main theorem stating the number of selection schemes -/
theorem number_of_selection_schemes :
  selection_schemes total_people total_cities restricted_people = 240 := by
  sorry

end number_of_selection_schemes_l3491_349117


namespace pelt_costs_l3491_349123

/-- Proof of pelt costs given total cost and individual profits -/
theorem pelt_costs (total_cost : ℝ) (total_profit_percent : ℝ) 
  (profit_percent_1 : ℝ) (profit_percent_2 : ℝ) 
  (h1 : total_cost = 22500)
  (h2 : total_profit_percent = 40)
  (h3 : profit_percent_1 = 25)
  (h4 : profit_percent_2 = 50) :
  ∃ (cost_1 cost_2 : ℝ),
    cost_1 + cost_2 = total_cost ∧
    cost_1 * (1 + profit_percent_1 / 100) + cost_2 * (1 + profit_percent_2 / 100) 
      = total_cost * (1 + total_profit_percent / 100) ∧
    cost_1 = 9000 ∧
    cost_2 = 13500 := by
  sorry

end pelt_costs_l3491_349123


namespace product_real_iff_condition_l3491_349120

/-- For complex numbers z₁ = a + bi and z₂ = c + di, where a, b, c, and d are real numbers,
    the product z₁ * z₂ is real if and only if ad + bc = 0. -/
theorem product_real_iff_condition (a b c d : ℝ) :
  (Complex.I * Complex.I = -1) →
  let z₁ : ℂ := Complex.mk a b
  let z₂ : ℂ := Complex.mk c d
  (z₁ * z₂).im = 0 ↔ a * d + b * c = 0 := by
  sorry

end product_real_iff_condition_l3491_349120


namespace stating_probability_theorem_l3491_349176

/-- Represents the number of guests -/
def num_guests : ℕ := 3

/-- Represents the number of roll types -/
def num_roll_types : ℕ := 4

/-- Represents the total number of rolls -/
def total_rolls : ℕ := 12

/-- Represents the number of rolls per guest -/
def rolls_per_guest : ℕ := 4

/-- Represents the number of each type of roll -/
def rolls_per_type : ℕ := 3

/-- 
Calculates the probability that each guest receives one roll of each type 
when rolls are randomly distributed.
-/
def probability_all_different_rolls : ℚ := sorry

/-- 
Theorem stating that the probability of each guest receiving one roll of each type 
is equal to 2/165720
-/
theorem probability_theorem : 
  probability_all_different_rolls = 2 / 165720 := sorry

end stating_probability_theorem_l3491_349176


namespace smallest_n_properties_count_non_14_divisors_l3491_349116

def is_perfect_power (x : ℕ) (k : ℕ) : Prop :=
  ∃ y : ℕ, x = y^k

def smallest_n : ℕ :=
  sorry

theorem smallest_n_properties (n : ℕ) (hn : n = smallest_n) :
  is_perfect_power (n / 2) 2 ∧
  is_perfect_power (n / 3) 3 ∧
  is_perfect_power (n / 5) 5 ∧
  is_perfect_power (n / 7) 7 :=
  sorry

theorem count_non_14_divisors (n : ℕ) (hn : n = smallest_n) :
  (Finset.filter (fun d => ¬(14 ∣ d)) (Nat.divisors n)).card = 240 :=
  sorry

end smallest_n_properties_count_non_14_divisors_l3491_349116


namespace carpenters_completion_time_l3491_349194

/-- The time it takes for two carpenters to complete a job together -/
theorem carpenters_completion_time 
  (rate1 : ℚ) -- Work rate of the first carpenter
  (rate2 : ℚ) -- Work rate of the second carpenter
  (h1 : rate1 = 1 / 7) -- First carpenter's work rate
  (h2 : rate2 = 1 / (35/2)) -- Second carpenter's work rate
  : (1 : ℚ) / (rate1 + rate2) = 5 := by
  sorry

end carpenters_completion_time_l3491_349194


namespace bulb_selection_problem_l3491_349126

theorem bulb_selection_problem (total_bulbs : ℕ) (defective_bulbs : ℕ) 
  (prob_at_least_one_defective : ℝ) :
  total_bulbs = 22 →
  defective_bulbs = 4 →
  prob_at_least_one_defective = 0.33766233766233766 →
  ∃ n : ℕ, n = 2 ∧ 
    (1 - ((total_bulbs - defective_bulbs : ℝ) / total_bulbs) ^ n) = prob_at_least_one_defective :=
by sorry

end bulb_selection_problem_l3491_349126


namespace tangent_line_equation_l3491_349111

/-- A line passing through (2,4) and tangent to (x-1)^2 + (y-2)^2 = 1 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 2 ∨ (3 * p.1 - 4 * p.2 + 10 = 0)}

/-- The circle (x-1)^2 + (y-2)^2 = 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 1}

/-- The point (2,4) -/
def Point : ℝ × ℝ := (2, 4)

theorem tangent_line_equation :
  ∀ (L : Set (ℝ × ℝ)),
    (Point ∈ L) →
    (∃! p, p ∈ L ∩ Circle) →
    L = TangentLine :=
sorry

end tangent_line_equation_l3491_349111


namespace test_scores_analysis_l3491_349160

def benchmark : ℝ := 85

def deviations : List ℝ := [8, -3, 12, -7, -10, -4, -8, 1, 0, 10]

def actual_scores : List ℝ := deviations.map (λ x => benchmark + x)

theorem test_scores_analysis :
  let max_score := actual_scores.maximum
  let min_score := actual_scores.minimum
  let avg_score := benchmark + (deviations.sum / deviations.length)
  (max_score = 97 ∧ min_score = 75) ∧ avg_score = 84.9 := by
  sorry

end test_scores_analysis_l3491_349160


namespace no_y_term_in_polynomial_l3491_349143

theorem no_y_term_in_polynomial (x y k : ℝ) : 
  (2*x - 3*y + 4 + 3*k*x + 2*k*y - k = (2 + 3*k)*x + (-k + 4)) → k = 3/2 :=
by
  sorry

end no_y_term_in_polynomial_l3491_349143


namespace namjoon_cookies_l3491_349158

/-- The number of cookies Namjoon had initially -/
def initial_cookies : ℕ := 24

/-- The number of cookies Namjoon ate -/
def eaten_cookies : ℕ := 8

/-- The number of cookies Namjoon gave to Hoseok -/
def given_cookies : ℕ := 7

/-- The number of cookies left after eating and giving away -/
def remaining_cookies : ℕ := 9

theorem namjoon_cookies : 
  initial_cookies - eaten_cookies - given_cookies = remaining_cookies :=
by sorry

end namjoon_cookies_l3491_349158


namespace exponential_function_point_l3491_349198

theorem exponential_function_point (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^x + 1
  f 0 = 2 := by
sorry

end exponential_function_point_l3491_349198


namespace product_of_fractions_l3491_349171

/-- Prove that the product of 2/3 and 1 4/9 is equal to 26/27 -/
theorem product_of_fractions :
  (2 : ℚ) / 3 * (1 + 4 / 9) = 26 / 27 := by
  sorry

end product_of_fractions_l3491_349171


namespace area_between_parabola_and_line_l3491_349149

-- Define the parabola and line functions
def parabola (x : ℝ) : ℝ := x^2 - 1
def line (x : ℝ) : ℝ := x + 1

-- Define the region
def region : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem area_between_parabola_and_line :
  ∫ x in region, (line x - parabola x) = 9/2 := by
  sorry


end area_between_parabola_and_line_l3491_349149


namespace arithmetic_calculation_l3491_349106

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end arithmetic_calculation_l3491_349106


namespace dolphin_shark_ratio_l3491_349119

/-- The ratio of buckets fed to dolphins compared to sharks -/
def R : ℚ := 1 / 2

/-- The number of buckets fed to sharks daily -/
def shark_buckets : ℕ := 4

/-- The number of days in 3 weeks -/
def days : ℕ := 21

/-- The total number of buckets lasting 3 weeks -/
def total_buckets : ℕ := 546

theorem dolphin_shark_ratio :
  R * shark_buckets * days +
  shark_buckets * days +
  (5 * shark_buckets) * days = total_buckets := by sorry

end dolphin_shark_ratio_l3491_349119


namespace violet_hiking_time_l3491_349156

/-- Calculates the maximum hiking time for Violet and her dog given their water consumption rates and Violet's water carrying capacity. -/
theorem violet_hiking_time (violet_rate : ℝ) (dog_rate : ℝ) (water_capacity : ℝ) :
  violet_rate = 800 →
  dog_rate = 400 →
  water_capacity = 4800 →
  (water_capacity / (violet_rate + dog_rate) : ℝ) = 4 := by
  sorry

end violet_hiking_time_l3491_349156


namespace divisibility_condition_solutions_l3491_349115

theorem divisibility_condition_solutions (a b : ℕ+) :
  (a ^ 2017 + b : ℤ) % (a * b : ℤ) = 0 → (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 2 ^ 2017) := by
  sorry

end divisibility_condition_solutions_l3491_349115


namespace perpendicular_sum_l3491_349159

/-- Given vectors a and b in ℝ², if a + b is perpendicular to a, 
    then the second component of b is -1. -/
theorem perpendicular_sum (a b : ℝ × ℝ) (h : a = (1, 0)) :
  (a + b) • a = 0 → b.1 = -1 := by
  sorry

end perpendicular_sum_l3491_349159


namespace problem_statement_l3491_349147

theorem problem_statement (x y z : ℝ) 
  (h1 : 1 / x = 2 / (y + z))
  (h2 : 1 / x = 3 / (z + x))
  (h3 : 1 / x = (x^2 - y - z) / (x + y + z))
  (h4 : x ≠ 0)
  (h5 : y + z ≠ 0)
  (h6 : z + x ≠ 0)
  (h7 : x + y + z ≠ 0) :
  (z - y) / x = 2 := by
sorry

end problem_statement_l3491_349147


namespace base_r_square_property_l3491_349105

/-- A natural number x is representable as a two-digit number with identical digits in base r -/
def is_two_digit_identical (x r : ℕ) : Prop :=
  ∃ a : ℕ, 0 < a ∧ a < r ∧ x = a * (r + 1)

/-- A natural number y is representable as a four-digit number in base r with form b00b -/
def is_four_digit_b00b (y r : ℕ) : Prop :=
  ∃ b : ℕ, 0 < b ∧ b < r ∧ y = b * (r^3 + 1)

/-- The main theorem -/
theorem base_r_square_property (r : ℕ) (hr : r ≤ 100) :
  (∃ x : ℕ, is_two_digit_identical x r ∧ is_four_digit_b00b (x^2) r) →
  r = 2 ∨ r = 23 :=
by sorry

end base_r_square_property_l3491_349105


namespace choir_members_proof_l3491_349153

theorem choir_members_proof :
  ∃! n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧ n % 7 = 3 ∧ n % 11 = 6 ∧ n = 220 := by
  sorry

end choir_members_proof_l3491_349153


namespace binomial_150_150_l3491_349170

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end binomial_150_150_l3491_349170


namespace gcd_lcm_product_24_36_l3491_349184

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  sorry

end gcd_lcm_product_24_36_l3491_349184


namespace sqrt_two_subset_P_l3491_349108

def P : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem sqrt_two_subset_P : {Real.sqrt 2} ⊆ P := by sorry

end sqrt_two_subset_P_l3491_349108


namespace speaker_must_be_trulalya_l3491_349144

/-- Represents the two brothers --/
inductive Brother
| T1 -- Tralyalya
| T2 -- Trulalya

/-- Represents the two possible card suits --/
inductive Suit
| Orange
| Purple

/-- Represents the statement made by a brother --/
structure Statement where
  speaker : Brother
  claimed_suit : Suit

/-- Represents the actual state of the cards --/
structure CardState where
  T1_card : Suit
  T2_card : Suit

/-- Determines if a statement is truthful given the actual card state --/
def is_truthful (s : Statement) (cs : CardState) : Prop :=
  match s.speaker with
  | Brother.T1 => s.claimed_suit = cs.T1_card
  | Brother.T2 => s.claimed_suit = cs.T2_card

/-- The main theorem: Given the conditions, the speaker must be Trulalya (T2) --/
theorem speaker_must_be_trulalya :
  ∀ (s : Statement) (cs : CardState),
    s.claimed_suit = Suit.Purple →
    is_truthful s cs →
    s.speaker = Brother.T2 :=
by sorry

end speaker_must_be_trulalya_l3491_349144


namespace school_function_participants_l3491_349154

theorem school_function_participants (boys girls : ℕ) 
  (h1 : 2 * (boys - girls) = 3 * 400)
  (h2 : 3 * girls = 4 * 150)
  (h3 : 2 * boys + 3 * girls = 3 * 550) :
  boys + girls = 800 := by
  sorry

end school_function_participants_l3491_349154


namespace unique_modulus_of_quadratic_roots_l3491_349165

theorem unique_modulus_of_quadratic_roots :
  ∃! r : ℝ, ∃ z : ℂ, z^2 - 6*z + 34 = 0 ∧ Complex.abs z = r :=
by sorry

end unique_modulus_of_quadratic_roots_l3491_349165


namespace grid_toothpicks_l3491_349157

/-- Calculates the total number of toothpicks in a rectangular grid. -/
def total_toothpicks (height width : ℕ) : ℕ :=
  let horizontal_lines := height + 1
  let vertical_lines := width + 1
  (horizontal_lines * width) + (vertical_lines * height)

/-- Theorem stating that a 30x15 rectangular grid of toothpicks uses 945 toothpicks. -/
theorem grid_toothpicks : total_toothpicks 30 15 = 945 := by
  sorry

end grid_toothpicks_l3491_349157


namespace triangle_abc_proof_l3491_349148

theorem triangle_abc_proof (b c : ℝ) (A : Real) (hb : b = 1) (hc : c = 2) (hA : A = 60 * π / 180) :
  ∃ (a : ℝ) (B : Real),
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A ∧
    a = Real.sqrt 3 ∧
    Real.cos B = (a ^ 2 + c ^ 2 - b ^ 2) / (2 * a * c) ∧
    B = 30 * π / 180 := by
  sorry


end triangle_abc_proof_l3491_349148


namespace book_pricing_theorem_l3491_349125

-- Define the cost function
def C (n : ℕ) : ℕ :=
  if n ≤ 24 then 12 * n
  else if n ≤ 48 then 11 * n
  else 10 * n

-- Define the production cost
def production_cost : ℕ := 5

-- Define the theorem
theorem book_pricing_theorem :
  -- Part 1: Exactly 6 values of n where C(n+1) < C(n)
  (∃ (S : Finset ℕ), S.card = 6 ∧ ∀ n, n ∈ S ↔ C (n + 1) < C n) ∧
  -- Part 2: Profit range for two individuals buying 60 books
  (∀ a b : ℕ, a + b = 60 → a ≥ 1 → b ≥ 1 →
    302 ≤ C a + C b - 60 * production_cost ∧
    C a + C b - 60 * production_cost ≤ 384) :=
by sorry

end book_pricing_theorem_l3491_349125


namespace boys_to_girls_ratio_l3491_349124

theorem boys_to_girls_ratio (total_students : ℕ) (girls : ℕ) 
  (h1 : girls = 135) (h2 : total_students = 351) :
  (total_students - girls) / girls = 8 / 5 := by
  sorry

end boys_to_girls_ratio_l3491_349124


namespace intersection_when_a_is_two_range_of_a_when_intersection_is_empty_l3491_349199

def U : Set ℝ := Set.univ

def A (a : ℝ) : Set ℝ := {x : ℝ | 2 * x + a > 0}

def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

theorem intersection_when_a_is_two :
  A 2 ∩ B = {x : ℝ | x > 3} := by sorry

theorem range_of_a_when_intersection_is_empty :
  ∀ a : ℝ, A a ∩ (U \ B) = ∅ ↔ a ≤ -6 := by sorry

end intersection_when_a_is_two_range_of_a_when_intersection_is_empty_l3491_349199


namespace six_students_five_lectures_l3491_349186

/-- The number of ways students can choose lectures -/
def lecture_choices (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: 6 students choosing from 5 lectures results in 5^6 possibilities -/
theorem six_students_five_lectures :
  lecture_choices 6 5 = 5^6 := by
  sorry

end six_students_five_lectures_l3491_349186


namespace gcd_ABC_l3491_349178

-- Define the constants
def a : ℕ := 177
def b : ℕ := 173

-- Define A, B, and C using the given formulas
def A : ℕ := a^5 + (a*b) * b^3 - b^5
def B : ℕ := b^5 + (a*b) * a^3 - a^5
def C : ℕ := b^4 + (a*b)^2 + a^4

-- State the theorem
theorem gcd_ABC : 
  Nat.gcd A C = 30637 ∧ Nat.gcd B C = 30637 := by
  sorry

end gcd_ABC_l3491_349178


namespace production_theorem_l3491_349179

/-- Represents the production process with recycling --/
def max_parts_and_waste (initial_blanks : ℕ) (efficiency : ℚ) : ℕ × ℚ :=
  sorry

/-- The theorem statement --/
theorem production_theorem :
  max_parts_and_waste 20 (2/3) = (29, 1/3) := by sorry

end production_theorem_l3491_349179


namespace happy_number_512_l3491_349150

def is_happy_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k + 1)^2 - (2*k - 1)^2

theorem happy_number_512 :
  is_happy_number 512 ∧
  ¬is_happy_number 285 ∧
  ¬is_happy_number 330 ∧
  ¬is_happy_number 582 :=
sorry

end happy_number_512_l3491_349150


namespace both_make_shots_probability_l3491_349188

/-- The probability that both person A and person B make their shots -/
def prob_both_make_shots (prob_A prob_B : ℝ) : ℝ := prob_A * prob_B

theorem both_make_shots_probability :
  let prob_A : ℝ := 2/5
  let prob_B : ℝ := 1/2
  prob_both_make_shots prob_A prob_B = 1/5 := by
  sorry

end both_make_shots_probability_l3491_349188


namespace prob_not_red_or_purple_is_correct_l3491_349187

-- Define the total number of balls and the number of each color
def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 10
def yellow_balls : ℕ := 7
def red_balls : ℕ := 15
def purple_balls : ℕ := 6

-- Define the probability of choosing a ball that is neither red nor purple
def prob_not_red_or_purple : ℚ := (white_balls + green_balls + yellow_balls) / total_balls

-- Theorem statement
theorem prob_not_red_or_purple_is_correct :
  prob_not_red_or_purple = 13/20 := by sorry

end prob_not_red_or_purple_is_correct_l3491_349187


namespace total_pokemon_cards_l3491_349195

/-- The number of people with Pokemon cards -/
def num_people : ℕ := 4

/-- The number of Pokemon cards each person has -/
def cards_per_person : ℕ := 14

/-- The total number of Pokemon cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem total_pokemon_cards : total_cards = 56 := by
  sorry

end total_pokemon_cards_l3491_349195


namespace third_dog_food_consumption_l3491_349191

/-- Given information about three dogs' food consumption, prove the amount eaten by the third dog -/
theorem third_dog_food_consumption
  (total_dogs : ℕ)
  (average_consumption : ℝ)
  (first_dog_consumption : ℝ)
  (h_total_dogs : total_dogs = 3)
  (h_average : average_consumption = 15)
  (h_first_dog : first_dog_consumption = 13)
  (h_second_dog : ∃ (second_dog_consumption : ℝ), second_dog_consumption = 2 * first_dog_consumption) :
  ∃ (third_dog_consumption : ℝ),
    third_dog_consumption = total_dogs * average_consumption - (first_dog_consumption + 2 * first_dog_consumption) :=
by sorry

end third_dog_food_consumption_l3491_349191


namespace probability_green_jellybean_l3491_349113

def total_jellybeans : ℕ := 7 + 9 + 8 + 10 + 6
def green_jellybeans : ℕ := 9

theorem probability_green_jellybean :
  (green_jellybeans : ℚ) / (total_jellybeans : ℚ) = 9 / 40 := by
  sorry

end probability_green_jellybean_l3491_349113


namespace inverse_function_problem_l3491_349137

/-- Given a function h and its inverse f⁻¹, prove that 7c + 7d = 2 -/
theorem inverse_function_problem (c d : ℝ) :
  (∀ x, (7 * x - 6 : ℝ) = (Function.invFun (fun x ↦ c * x + d) x - 5)) →
  7 * c + 7 * d = 2 := by
  sorry

end inverse_function_problem_l3491_349137


namespace square_area_on_parabola_l3491_349175

/-- The area of a square with one side on y = 7 and endpoints on y = x^2 + 4x + 3 is 32 -/
theorem square_area_on_parabola : ∃ (s : ℝ),
  (∃ (x₁ x₂ : ℝ),
    x₁^2 + 4*x₁ + 3 = 7 ∧
    x₂^2 + 4*x₂ + 3 = 7 ∧
    s = |x₂ - x₁|) →
  s^2 = 32 := by
  sorry

end square_area_on_parabola_l3491_349175


namespace intersection_of_A_and_B_l3491_349121

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {x | x^2 - x - 2 > 0}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo 2 3 := by sorry

end intersection_of_A_and_B_l3491_349121


namespace lindas_lunchbox_theorem_l3491_349131

/-- Represents the cost calculation at Linda's Lunchbox -/
def lindas_lunchbox_cost (sandwich_price : ℝ) (soda_price : ℝ) (discount_rate : ℝ) 
  (discount_threshold : ℕ) (num_sandwiches : ℕ) (num_sodas : ℕ) : ℝ :=
  let total_items := num_sandwiches + num_sodas
  let subtotal := sandwich_price * num_sandwiches + soda_price * num_sodas
  if total_items ≥ discount_threshold then
    subtotal * (1 - discount_rate)
  else
    subtotal

/-- Theorem: The cost of 7 sandwiches and 5 sodas at Linda's Lunchbox is $38.7 -/
theorem lindas_lunchbox_theorem : 
  lindas_lunchbox_cost 4 3 0.1 10 7 5 = 38.7 := by
  sorry

end lindas_lunchbox_theorem_l3491_349131
