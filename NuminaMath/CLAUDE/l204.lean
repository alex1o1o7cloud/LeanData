import Mathlib

namespace decreasing_function_implies_a_nonnegative_l204_20445

-- Define the function f
def f (x a b : ℝ) : ℝ := x^2 + |x - a| + b

-- State the theorem
theorem decreasing_function_implies_a_nonnegative 
  (h : ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ ≤ 0 → f x₂ a b ≤ f x₁ a b) : 
  a ≥ 0 := by
  sorry

end decreasing_function_implies_a_nonnegative_l204_20445


namespace dance_team_size_l204_20410

theorem dance_team_size (initial_size : ℕ) (quit : ℕ) (new_members : ℕ) : 
  initial_size = 25 → quit = 8 → new_members = 13 → 
  initial_size - quit + new_members = 30 := by
  sorry

end dance_team_size_l204_20410


namespace unique_function_existence_l204_20476

theorem unique_function_existence (g : ℂ → ℂ) (ω a : ℂ) 
  (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  ∃! f : ℂ → ℂ, ∀ z : ℂ, f z + f (ω * z + a) = g z := by
  sorry

end unique_function_existence_l204_20476


namespace initial_kids_count_l204_20466

/-- The number of kids still awake after the first round of napping -/
def kids_after_first_round (initial : ℕ) : ℕ := initial / 2

/-- The number of kids still awake after the second round of napping -/
def kids_after_second_round (initial : ℕ) : ℕ := kids_after_first_round initial / 2

/-- Theorem stating that the initial number of kids ready for a nap is 20 -/
theorem initial_kids_count : ∃ (initial : ℕ), 
  kids_after_second_round initial = 5 ∧ initial = 20 := by
  sorry

#check initial_kids_count

end initial_kids_count_l204_20466


namespace quadratic_inequality_solution_l204_20453

-- Define the quadratic function
def f (x : ℝ) := x^2 + x - 2

-- Define the solution set
def solution_set := {x : ℝ | x < -2 ∨ x > 1}

-- Theorem stating that the solution set is correct
theorem quadratic_inequality_solution :
  ∀ x : ℝ, f x > 0 ↔ x ∈ solution_set :=
by sorry

end quadratic_inequality_solution_l204_20453


namespace boys_to_girls_ratio_l204_20490

theorem boys_to_girls_ratio :
  ∀ (boys girls : ℕ),
    boys = 80 →
    girls = boys + 128 →
    (boys : ℚ) / girls = 5 / 13 :=
by
  sorry

end boys_to_girls_ratio_l204_20490


namespace bob_improvement_percentage_l204_20411

/-- The percentage improvement needed to match a target time -/
def percentage_improvement (current_time target_time : ℕ) : ℚ :=
  (current_time - target_time : ℚ) / current_time * 100

/-- Bob's current mile time in seconds -/
def bob_time : ℕ := 640

/-- Bob's sister's mile time in seconds -/
def sister_time : ℕ := 320

/-- Theorem: Bob needs to improve his time by 50% to match his sister's time -/
theorem bob_improvement_percentage :
  percentage_improvement bob_time sister_time = 50 := by
  sorry

end bob_improvement_percentage_l204_20411


namespace sine_product_upper_bound_sine_product_upper_bound_achievable_l204_20439

/-- Given points A, B, and C in a coordinate plane, where A = (-8, 0), B = (8, 0), and C = (t, 6) for some real number t, the product of sines of angles CAB and CBA is at most 3/8. -/
theorem sine_product_upper_bound (t : ℝ) :
  let A : ℝ × ℝ := (-8, 0)
  let B : ℝ × ℝ := (8, 0)
  let C : ℝ × ℝ := (t, 6)
  let angle_CAB := Real.arctan ((C.2 - A.2) / (C.1 - A.1)) - Real.arctan ((B.2 - A.2) / (B.1 - A.1))
  let angle_CBA := Real.arctan ((C.2 - B.2) / (C.1 - B.1)) - Real.arctan ((A.2 - B.2) / (A.1 - B.1))
  Real.sin angle_CAB * Real.sin angle_CBA ≤ 3/8 :=
by sorry

/-- The upper bound 3/8 for the product of sines is achievable. -/
theorem sine_product_upper_bound_achievable :
  ∃ t : ℝ,
  let A : ℝ × ℝ := (-8, 0)
  let B : ℝ × ℝ := (8, 0)
  let C : ℝ × ℝ := (t, 6)
  let angle_CAB := Real.arctan ((C.2 - A.2) / (C.1 - A.1)) - Real.arctan ((B.2 - A.2) / (B.1 - A.1))
  let angle_CBA := Real.arctan ((C.2 - B.2) / (C.1 - B.1)) - Real.arctan ((A.2 - B.2) / (A.1 - B.1))
  Real.sin angle_CAB * Real.sin angle_CBA = 3/8 :=
by sorry

end sine_product_upper_bound_sine_product_upper_bound_achievable_l204_20439


namespace multiple_compounds_with_same_oxygen_percentage_l204_20499

/-- Represents a chemical compound -/
structure Compound where
  elements : List String
  massPercentages : List Float
  deriving Repr

/-- Predicate to check if a compound has 57.14% oxygen -/
def hasCorrectOxygenPercentage (c : Compound) : Prop :=
  "O" ∈ c.elements ∧ 
  let oIndex := c.elements.indexOf "O"
  c.massPercentages[oIndex]! = 57.14

/-- Theorem stating that multiple compounds can have 57.14% oxygen -/
theorem multiple_compounds_with_same_oxygen_percentage :
  ∃ (c1 c2 : Compound), c1 ≠ c2 ∧ 
    hasCorrectOxygenPercentage c1 ∧ 
    hasCorrectOxygenPercentage c2 :=
sorry

end multiple_compounds_with_same_oxygen_percentage_l204_20499


namespace isosceles_triangle_vertex_angle_l204_20450

theorem isosceles_triangle_vertex_angle (α β : ℝ) : 
  α = 50 → -- base angle is 50°
  β = 180 - 2*α → -- vertex angle formula
  β = 80 -- vertex angle is 80°
:= by sorry

end isosceles_triangle_vertex_angle_l204_20450


namespace polynomial_remainder_l204_20413

theorem polynomial_remainder (x : ℝ) : 
  (x^4 - x + 1) % (x + 3) = 85 := by
sorry

end polynomial_remainder_l204_20413


namespace bowling_ball_weight_is_18_l204_20467

-- Define the weight of one bowling ball
def bowling_ball_weight : ℝ := sorry

-- Define the weight of one kayak
def kayak_weight : ℝ := sorry

-- Theorem to prove the weight of one bowling ball
theorem bowling_ball_weight_is_18 :
  (10 * bowling_ball_weight = 6 * kayak_weight) →
  (3 * kayak_weight = 90) →
  bowling_ball_weight = 18 := by
  sorry

end bowling_ball_weight_is_18_l204_20467


namespace function_existence_l204_20496

theorem function_existence : ∃ f : ℤ → ℤ, ∀ a b : ℤ, f (a + b) + f (a * b - 1) = f a * f b + 1 := by
  sorry

end function_existence_l204_20496


namespace numbered_cube_consecutive_pairs_l204_20478

/-- Represents a cube with numbers on its faces -/
structure NumberedCube where
  numbers : Fin 6 → ℕ
  distinct : ∀ i j, i ≠ j → numbers i ≠ numbers j

/-- Checks if two faces are adjacent on a cube -/
def adjacent (f1 f2 : Fin 6) : Prop := sorry

/-- Checks if two numbers are consecutive -/
def consecutive (n1 n2 : ℕ) : Prop := n2 = n1 + 1 ∨ n1 = n2 + 1

/-- Theorem: A cube numbered with consecutive integers from 1 to 6 
    has at least two pairs of adjacent faces with consecutive numbers -/
theorem numbered_cube_consecutive_pairs (c : NumberedCube) 
  (h_range : ∀ i, c.numbers i ∈ Finset.range 6) : 
  ∃ (f1 f2 f3 f4 : Fin 6), f1 ≠ f2 ∧ f3 ≠ f4 ∧ (f1, f2) ≠ (f3, f4) ∧ 
    adjacent f1 f2 ∧ adjacent f3 f4 ∧ 
    consecutive (c.numbers f1) (c.numbers f2) ∧ 
    consecutive (c.numbers f3) (c.numbers f4) := by
  sorry

end numbered_cube_consecutive_pairs_l204_20478


namespace price_difference_is_24_l204_20415

/-- The original price of the smartphone --/
def original_price : ℚ := 800

/-- The single discount rate offered by the first store --/
def single_discount_rate : ℚ := 25 / 100

/-- The first discount rate offered by the second store --/
def first_discount_rate : ℚ := 20 / 100

/-- The second discount rate offered by the second store --/
def second_discount_rate : ℚ := 10 / 100

/-- The price after applying a single discount --/
def price_after_single_discount : ℚ := original_price * (1 - single_discount_rate)

/-- The price after applying two successive discounts --/
def price_after_successive_discounts : ℚ := 
  original_price * (1 - first_discount_rate) * (1 - second_discount_rate)

/-- Theorem stating that the difference between the two final prices is $24 --/
theorem price_difference_is_24 : 
  price_after_single_discount - price_after_successive_discounts = 24 := by
  sorry


end price_difference_is_24_l204_20415


namespace prime_product_l204_20419

theorem prime_product (p q : ℕ) : 
  Prime p ∧ Prime q ∧ Prime (q^2 - p^2) → p * q = 6 :=
by sorry

end prime_product_l204_20419


namespace circle_center_on_line_ab_range_l204_20472

theorem circle_center_on_line_ab_range :
  ∀ (a b : ℝ),
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧ a*x - b*y + 1 = 0) →
  a*b ≤ 1/8 ∧ ∀ ε > 0, ∃ (a' b' : ℝ), a'*b' < -ε :=
by sorry

end circle_center_on_line_ab_range_l204_20472


namespace smallest_positive_integer_3003m_66666n_l204_20412

theorem smallest_positive_integer_3003m_66666n : 
  (∃ (k : ℕ), k > 0 ∧ ∀ (x : ℕ), x > 0 → (∃ (m n : ℤ), x = 3003 * m + 66666 * n) → k ≤ x) ∧
  (∃ (m n : ℤ), 3 = 3003 * m + 66666 * n) :=
sorry

end smallest_positive_integer_3003m_66666n_l204_20412


namespace right_handed_players_count_l204_20482

theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) : 
  total_players = 70 →
  throwers = 31 →
  (total_players - throwers) % 3 = 0 →
  57 = throwers + (total_players - throwers) * 2 / 3 := by
  sorry

end right_handed_players_count_l204_20482


namespace complex_product_theorem_l204_20497

theorem complex_product_theorem (z₁ z₂ : ℂ) :
  z₁ = 4 + I → z₂ = 1 - 2*I → z₁ * z₂ = 6 - 7*I := by
  sorry

end complex_product_theorem_l204_20497


namespace smallest_k_inequality_l204_20458

theorem smallest_k_inequality (a b c : ℕ+) : 
  ∃ (k : ℕ+), k = 1297 ∧ 
  (16 * a^2 + 36 * b^2 + 81 * c^2) * (81 * a^2 + 36 * b^2 + 16 * c^2) < k * (a^2 + b^2 + c^2)^2 ∧
  ∀ (m : ℕ+), m < 1297 → 
  (16 * a^2 + 36 * b^2 + 81 * c^2) * (81 * a^2 + 36 * b^2 + 16 * c^2) ≥ m * (a^2 + b^2 + c^2)^2 := by
  sorry

end smallest_k_inequality_l204_20458


namespace smallest_acute_angle_in_right_triangle_l204_20431

theorem smallest_acute_angle_in_right_triangle (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 90 → (a / b) = (3 / 2) → min a b = 36 := by
  sorry

end smallest_acute_angle_in_right_triangle_l204_20431


namespace exist_50_integers_with_equal_sum_l204_20468

/-- Sum of digits function -/
def S (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + S (n / 10)

/-- Theorem statement -/
theorem exist_50_integers_with_equal_sum :
  ∃ (n : Fin 50 → ℕ), (∀ i j, i < j → n i < n j) ∧
    (∀ i j, i < j → n i + S (n i) = n j + S (n j)) :=
sorry

end exist_50_integers_with_equal_sum_l204_20468


namespace initial_column_size_l204_20423

/-- The number of people in each column initially -/
def people_per_column : ℕ := 30

/-- The total number of people -/
def total_people : ℕ := people_per_column * 16

/-- The number of columns formed when 48 people stand in each column -/
def columns_with_48 : ℕ := total_people / 48

theorem initial_column_size :
  (total_people = people_per_column * 16) ∧
  (total_people = 48 * 10) ∧
  (columns_with_48 = 10) →
  people_per_column = 30 := by
  sorry

end initial_column_size_l204_20423


namespace work_rate_problem_l204_20421

/-- Proves that given the work rates of A and B, and the combined work rate of A, B, and C,
    we can determine how long it takes C to do the work alone. -/
theorem work_rate_problem (a b c : ℝ) (h1 : a = 6) (h2 : b = 5) 
  (h3 : 1 / a + 1 / b + 1 / c = 1 / (10 / 9)) : c = 15 / 8 := by
  sorry

#eval (15 : ℚ) / 8  -- To show that 15/8 = 1.875

end work_rate_problem_l204_20421


namespace tank_plastering_cost_l204_20417

/-- Calculates the cost of plastering a rectangular tank's walls and bottom. -/
def plasteringCost (length width depth : ℝ) (costPerSquareMeter : ℝ) : ℝ :=
  let bottomArea := length * width
  let wallArea := 2 * (length * depth + width * depth)
  let totalArea := bottomArea + wallArea
  totalArea * costPerSquareMeter

/-- Theorem stating the cost of plastering a specific tank. -/
theorem tank_plastering_cost :
  let length : ℝ := 25
  let width : ℝ := 12
  let depth : ℝ := 6
  let costPerSquareMeter : ℝ := 0.75  -- 75 paise = 0.75 rupees
  plasteringCost length width depth costPerSquareMeter = 558 := by
  sorry

#eval plasteringCost 25 12 6 0.75

end tank_plastering_cost_l204_20417


namespace max_value_of_b_l204_20433

theorem max_value_of_b (a b : ℤ) : 
  (a + b)^2 + a*(a + b) + b = 0 → b ≤ 9 := by sorry

end max_value_of_b_l204_20433


namespace certain_value_proof_l204_20422

theorem certain_value_proof (x : ℤ) : 
  (∀ n : ℤ, 101 * n^2 ≤ x → n ≤ 10) ∧ 
  (∃ n : ℤ, n = 10 ∧ 101 * n^2 ≤ x) →
  x = 10100 :=
by sorry

end certain_value_proof_l204_20422


namespace largest_sum_and_simplification_l204_20480

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/4 + 1/5, 1/4 + 1/6, 1/4 + 1/3, 1/4 + 1/8, 1/4 + 1/7]
  (∀ x ∈ sums, x ≤ (1/4 + 1/3)) ∧ (1/4 + 1/3 = 7/12) := by
  sorry

end largest_sum_and_simplification_l204_20480


namespace simplify_square_roots_l204_20485

theorem simplify_square_roots : Real.sqrt 49 - Real.sqrt 144 + Real.sqrt 9 = -2 := by
  sorry

end simplify_square_roots_l204_20485


namespace parking_theorem_l204_20454

/-- Represents the number of empty parking spaces -/
def total_spaces : ℕ := 10

/-- Represents the number of cars to be parked -/
def num_cars : ℕ := 3

/-- Represents the number of empty spaces required between cars -/
def spaces_between : ℕ := 1

/-- Calculates the number of parking arrangements given the constraints -/
def parking_arrangements (total : ℕ) (cars : ℕ) (spaces : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of parking arrangements is 40 -/
theorem parking_theorem : 
  parking_arrangements total_spaces num_cars spaces_between = 40 :=
sorry

end parking_theorem_l204_20454


namespace equality_implies_product_equality_l204_20484

theorem equality_implies_product_equality (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end equality_implies_product_equality_l204_20484


namespace count_m_gons_correct_l204_20462

/-- Given integers m and n where 4 < m < n, and a regular polygon with 2n+1 sides,
    this function computes the number of convex m-gons with vertices from the polygon's vertices
    and exactly two acute interior angles. -/
def count_m_gons (m n : ℕ) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

/-- Theorem stating that count_m_gons correctly computes the number of m-gons
    satisfying the given conditions. -/
theorem count_m_gons_correct (m n : ℕ) (h1 : 4 < m) (h2 : m < n) :
  count_m_gons m n = (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1)) :=
by sorry

end count_m_gons_correct_l204_20462


namespace card_area_theorem_l204_20432

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem card_area_theorem (original : Rectangle) 
  (h1 : original.length = 3 ∧ original.width = 7)
  (h2 : ∃ (shortened : Rectangle), 
    (shortened.length = original.length ∧ shortened.width = original.width - 2) ∨
    (shortened.length = original.length - 2 ∧ shortened.width = original.width) ∧
    area shortened = 15) :
  ∃ (other_shortened : Rectangle),
    (other_shortened.length = original.length - 2 ∧ other_shortened.width = original.width) ∨
    (other_shortened.length = original.length ∧ other_shortened.width = original.width - 2) ∧
    area other_shortened ≠ area shortened ∧
    area other_shortened = 7 := by
  sorry

end card_area_theorem_l204_20432


namespace square_roots_problem_l204_20404

theorem square_roots_problem (a : ℝ) : 
  (∃ x : ℝ, (3 * a + 2)^2 = x ∧ (a + 14)^2 = x) → a = -4 := by
  sorry

end square_roots_problem_l204_20404


namespace factorization_proof_l204_20491

theorem factorization_proof (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 3) * (x + 2) := by
  sorry

end factorization_proof_l204_20491


namespace max_value_of_x_plus_2y_l204_20455

theorem max_value_of_x_plus_2y (x y : ℝ) (h : x^2 - x*y + y^2 = 1) :
  x + 2*y ≤ 2 * Real.sqrt 21 / 3 :=
sorry

end max_value_of_x_plus_2y_l204_20455


namespace initial_puppies_count_l204_20401

/-- The number of puppies Alyssa had initially -/
def initial_puppies : ℕ := 7

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 5

/-- The number of puppies Alyssa has left -/
def puppies_left : ℕ := 2

/-- Theorem stating that the initial number of puppies equals the sum of puppies given away and puppies left -/
theorem initial_puppies_count : initial_puppies = puppies_given_away + puppies_left := by
  sorry

end initial_puppies_count_l204_20401


namespace shortening_powers_l204_20403

def is_power (n : ℕ) (k : ℕ) : Prop :=
  ∃ m : ℕ, n = m^k

def shorten (n : ℕ) : ℕ :=
  n / 10

theorem shortening_powers (n : ℕ) :
  n > 1000000 →
  is_power (shorten n) 2 →
  is_power (shorten (shorten n)) 3 →
  is_power (shorten (shorten (shorten n))) 4 →
  is_power (shorten (shorten (shorten (shorten n)))) 5 →
  is_power (shorten (shorten (shorten (shorten (shorten n))))) 6 :=
by sorry

end shortening_powers_l204_20403


namespace cube_painting_problem_l204_20406

theorem cube_painting_problem (n : ℕ) : 
  n > 0 →  -- Ensure n is positive
  (4 * n^2 : ℚ) / (6 * n^3 : ℚ) = 3 / 4 → 
  n = 2 :=
by
  sorry

end cube_painting_problem_l204_20406


namespace percentage_of_amount_l204_20446

theorem percentage_of_amount (amount : ℝ) :
  (25 : ℝ) / 100 * amount = 150 → amount = 600 := by
  sorry

end percentage_of_amount_l204_20446


namespace existence_of_number_with_prime_multiples_l204_20464

theorem existence_of_number_with_prime_multiples : ∃ x : ℝ, 
  (∃ p : ℕ, Nat.Prime p ∧ (10 : ℝ) * x = p) ∧ 
  (∃ q : ℕ, Nat.Prime q ∧ (15 : ℝ) * x = q) := by
  sorry

end existence_of_number_with_prime_multiples_l204_20464


namespace remainder_theorem_l204_20416

-- Define the polynomial P(x) = x^100 - 2x^51 + 1
def P (x : ℝ) : ℝ := x^100 - 2*x^51 + 1

-- Define the divisor polynomial D(x) = x^2 - 1
def D (x : ℝ) : ℝ := x^2 - 1

-- Define the remainder polynomial R(x) = -2x + 2
def R (x : ℝ) : ℝ := -2*x + 2

-- Theorem statement
theorem remainder_theorem : 
  ∃ (Q : ℝ → ℝ), ∀ x, P x = Q x * D x + R x :=
sorry

end remainder_theorem_l204_20416


namespace peace_treaty_day_l204_20470

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define a function to get the next day of the week
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem peace_treaty_day (startDay : DayOfWeek) (daysPassed : Nat) :
  startDay = DayOfWeek.Monday ∧ daysPassed = 893 →
  advanceDay startDay daysPassed = DayOfWeek.Saturday :=
by
  sorry -- Proof omitted as per instructions


end peace_treaty_day_l204_20470


namespace final_state_digits_l204_20493

/-- Represents the state of the board as three integers -/
structure BoardState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Performs one iteration of pairwise sum replacement -/
def iterate (state : BoardState) : BoardState :=
  { a := (state.a + state.b) % 10,
    b := (state.a + state.c) % 10,
    c := (state.b + state.c) % 10 }

/-- Performs n iterations of pairwise sum replacement -/
def iterateN (n : ℕ) (state : BoardState) : BoardState :=
  match n with
  | 0 => state
  | n + 1 => iterate (iterateN n state)

/-- The main theorem to be proved -/
theorem final_state_digits (initialState : BoardState) :
  initialState.a = 1 ∧ initialState.b = 2 ∧ initialState.c = 4 →
  let finalState := iterateN 60 initialState
  (finalState.a = 6 ∧ finalState.b = 7 ∧ finalState.c = 9) ∨
  (finalState.a = 6 ∧ finalState.b = 9 ∧ finalState.c = 7) ∨
  (finalState.a = 7 ∧ finalState.b = 6 ∧ finalState.c = 9) ∨
  (finalState.a = 7 ∧ finalState.b = 9 ∧ finalState.c = 6) ∨
  (finalState.a = 9 ∧ finalState.b = 6 ∧ finalState.c = 7) ∨
  (finalState.a = 9 ∧ finalState.b = 7 ∧ finalState.c = 6) :=
by sorry

end final_state_digits_l204_20493


namespace sequence_a_property_l204_20407

def sequence_a (n : ℕ) : ℚ :=
  3 / (15 * n - 14)

theorem sequence_a_property :
  (sequence_a 1 = 3) ∧
  (∀ n : ℕ, n > 0 → 1 / (sequence_a (n + 1) + 1) - 1 / (sequence_a n) = 5) :=
by sorry

end sequence_a_property_l204_20407


namespace square_edge_sum_l204_20420

theorem square_edge_sum (u v w x : ℕ+) : 
  u * x + u * v + v * w + w * x = 15 → u + v + w + x = 8 :=
by
  sorry

end square_edge_sum_l204_20420


namespace square_area_l204_20414

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the square
def square (p1 p2 : Point2D) : ℝ := 
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2

-- Theorem statement
theorem square_area (p1 p2 : Point2D) (h : p1 = ⟨1, 2⟩ ∧ p2 = ⟨4, 6⟩) : 
  square p1 p2 = 25 := by
  sorry

end square_area_l204_20414


namespace sum_of_powers_of_i_l204_20479

def i : ℂ := Complex.I

theorem sum_of_powers_of_i : 
  (Finset.range 2015).sum (λ n => i ^ n) = i :=
sorry

end sum_of_powers_of_i_l204_20479


namespace football_field_theorem_l204_20429

/-- Represents a rectangular football field -/
structure FootballField where
  length : ℝ  -- length in centimeters
  width : ℝ   -- width in meters
  perimeter_condition : 2 * (length / 100 + width) > 350
  area_condition : (length / 100) * width < 7560

/-- Checks if a field meets international match requirements -/
def is_international_match_compliant (field : FootballField) : Prop :=
  100 ≤ field.length / 100 ∧ field.length / 100 ≤ 110 ∧
  64 ≤ field.width ∧ field.width ≤ 75

theorem football_field_theorem (field : FootballField) 
  (h_width : field.width = 70) :
  (10.5 < field.length / 100 ∧ field.length / 100 < 108) ∧
  is_international_match_compliant field := by
  sorry

end football_field_theorem_l204_20429


namespace harry_seed_cost_l204_20400

/-- The cost of seeds for Harry's garden --/
def seedCost (pumpkinPrice tomatoPrice pepperPrice : ℚ)
             (pumpkinQty tomatoQty pepperQty : ℕ) : ℚ :=
  pumpkinPrice * pumpkinQty + tomatoPrice * tomatoQty + pepperPrice * pepperQty

/-- Theorem stating the total cost of seeds for Harry --/
theorem harry_seed_cost :
  seedCost (5/2) (3/2) (9/10) 3 4 5 = 18 := by
  sorry

end harry_seed_cost_l204_20400


namespace smallest_k_with_remainder_one_l204_20442

theorem smallest_k_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 23 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 23 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  sorry

end smallest_k_with_remainder_one_l204_20442


namespace sequence_property_l204_20440

def is_arithmetic_progression (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_progression (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

def has_common_difference (a b c d : ℝ) (diff : ℝ) : Prop :=
  b - a = diff ∧ c - b = diff ∧ d - c = diff

theorem sequence_property (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ ∧ a₅ < a₆ ∧ a₆ < a₇ ∧ a₇ < a₈ →
  ((has_common_difference a₁ a₂ a₃ a₄ 4 ∧ has_common_difference a₅ a₆ a₇ a₈ 36) ∨
   (has_common_difference a₂ a₃ a₄ a₅ 4 ∧ has_common_difference a₅ a₆ a₇ a₈ 36) ∨
   (has_common_difference a₁ a₂ a₃ a₄ 4 ∧ has_common_difference a₄ a₅ a₆ a₇ 36) ∨
   (has_common_difference a₂ a₃ a₄ a₅ 4 ∧ has_common_difference a₄ a₅ a₆ a₇ 36) ∨
   (has_common_difference a₁ a₂ a₃ a₄ 36 ∧ has_common_difference a₅ a₆ a₇ a₈ 4)) →
  (is_geometric_progression a₂ a₃ a₄ a₅ ∨ is_geometric_progression a₃ a₄ a₅ a₆ ∨
   is_geometric_progression a₄ a₅ a₆ a₇) →
  a₈ = 126 ∨ a₈ = 6 :=
by sorry

end sequence_property_l204_20440


namespace sum_of_numbers_l204_20437

theorem sum_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_prod : x * y = 12) (h_recip : 1 / x = 3 * (1 / y)) : x + y = 8 := by
  sorry

end sum_of_numbers_l204_20437


namespace isosceles_triangle_base_length_l204_20435

/-- 
An isosceles triangle with congruent sides of length 7 cm and perimeter 23 cm 
has a base of length 9 cm.
-/
theorem isosceles_triangle_base_length : 
  ∀ (base : ℝ), 
  base > 0 → 
  7 + 7 + base = 23 → 
  base = 9 :=
by
  sorry

end isosceles_triangle_base_length_l204_20435


namespace toy_cost_l204_20430

/-- Given Roger's initial amount, the cost of a game, and the number of toys he can buy,
    prove that each toy costs $7. -/
theorem toy_cost (initial_amount : ℕ) (game_cost : ℕ) (num_toys : ℕ) :
  initial_amount = 68 →
  game_cost = 47 →
  num_toys = 3 →
  (initial_amount - game_cost) / num_toys = 7 := by
  sorry

end toy_cost_l204_20430


namespace exponential_function_property_l204_20456

theorem exponential_function_property (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = a^x) →
  (a > 0) →
  (abs (f 2 - f 1) = a / 2) →
  (a = 1/2 ∨ a = 3/2) :=
by sorry

end exponential_function_property_l204_20456


namespace retail_price_calculation_l204_20460

/-- The retail price of a machine, given wholesale price, discount, and profit margin -/
theorem retail_price_calculation (wholesale_price discount profit_margin : ℚ) 
  (h1 : wholesale_price = 126)
  (h2 : discount = 10/100)
  (h3 : profit_margin = 20/100)
  (h4 : profit_margin * wholesale_price + wholesale_price = (1 - discount) * retail_price) :
  retail_price = 168 := by
  sorry

#check retail_price_calculation

end retail_price_calculation_l204_20460


namespace integer_root_of_cubic_l204_20483

theorem integer_root_of_cubic (b c : ℚ) :
  (∃ x : ℝ, x^3 + b*x + c = 0 ∧ x = 2 - Real.sqrt 5) →
  (∃ r : ℤ, r^3 + b*r + c = 0) →
  (∃ r : ℤ, r^3 + b*r + c = 0 ∧ r = -4) :=
by sorry

end integer_root_of_cubic_l204_20483


namespace repeating_decimal_35_eq_fraction_l204_20402

/-- Represents a repeating decimal where the digits 35 repeat infinitely after the decimal point. -/
def repeating_decimal_35 : ℚ :=
  35 / 99

/-- The theorem states that the repeating decimal 0.353535... is equal to the fraction 35/99. -/
theorem repeating_decimal_35_eq_fraction :
  repeating_decimal_35 = 35 / 99 := by
  sorry

end repeating_decimal_35_eq_fraction_l204_20402


namespace thomson_incentive_spending_l204_20474

theorem thomson_incentive_spending (incentive : ℝ) (savings : ℝ) (f : ℝ) : 
  incentive = 240 →
  savings = 84 →
  savings = (3/4) * (incentive - f * incentive - (1/5) * incentive) →
  f = 1/3 := by
sorry

end thomson_incentive_spending_l204_20474


namespace dmv_waiting_time_l204_20465

theorem dmv_waiting_time (x : ℝ) : 
  x + (4 * x + 14) = 114 → x = 20 := by
  sorry

end dmv_waiting_time_l204_20465


namespace manny_cookie_slices_left_l204_20447

/-- Calculates the number of cookie slices left after distribution --/
def cookie_slices_left (num_pies : ℕ) (slices_per_pie : ℕ) (total_people : ℕ) (half_slice_people : ℕ) : ℕ :=
  let total_slices := num_pies * slices_per_pie
  let full_slice_people := total_people - half_slice_people
  let distributed_slices := full_slice_people + (half_slice_people / 2)
  total_slices - distributed_slices

/-- Theorem stating the number of cookie slices left in Manny's scenario --/
theorem manny_cookie_slices_left : cookie_slices_left 6 12 39 3 = 33 := by
  sorry

#eval cookie_slices_left 6 12 39 3

end manny_cookie_slices_left_l204_20447


namespace quadrilateral_diagonal_length_l204_20473

/-- Represents a quadrilateral with diagonals intersecting at a point -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem quadrilateral_diagonal_length 
  (ABCD : Quadrilateral) 
  (h1 : distance ABCD.B ABCD.O = 3)
  (h2 : distance ABCD.O ABCD.D = 9)
  (h3 : distance ABCD.A ABCD.O = 5)
  (h4 : distance ABCD.O ABCD.C = 2)
  (h5 : distance ABCD.A ABCD.B = 7) :
  distance ABCD.A ABCD.D = Real.sqrt 151 := by
  sorry

end quadrilateral_diagonal_length_l204_20473


namespace fraction_equality_l204_20489

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : a^2 + b^2 ≠ 0) (h4 : a^4 - 2*b^4 ≠ 0) 
  (h5 : a^2 * b^2 / (a^4 - 2*b^4) = 1) : 
  (a^2 - b^2) / (a^2 + b^2) = 1/3 := by
sorry

end fraction_equality_l204_20489


namespace paper_folding_perimeter_ratio_l204_20457

/-- Given a square piece of paper with side length 8 inches, when folded and cut as described,
    the ratio of the perimeter of the larger rectangle to the perimeter of one of the smaller rectangles is 3/2. -/
theorem paper_folding_perimeter_ratio :
  let initial_side_length : ℝ := 8
  let large_rectangle_length : ℝ := initial_side_length
  let large_rectangle_width : ℝ := initial_side_length / 2
  let small_rectangle_side : ℝ := initial_side_length / 2
  let large_perimeter : ℝ := 2 * (large_rectangle_length + large_rectangle_width)
  let small_perimeter : ℝ := 4 * small_rectangle_side
  large_perimeter / small_perimeter = 3 / 2 := by sorry

end paper_folding_perimeter_ratio_l204_20457


namespace sugar_percentage_after_addition_l204_20425

def initial_volume : ℝ := 340
def water_percentage : ℝ := 0.75
def kola_percentage : ℝ := 0.05
def added_sugar : ℝ := 3.2
def added_water : ℝ := 12
def added_kola : ℝ := 6.8

theorem sugar_percentage_after_addition :
  let initial_sugar_percentage : ℝ := 1 - water_percentage - kola_percentage
  let initial_sugar_volume : ℝ := initial_sugar_percentage * initial_volume
  let final_sugar_volume : ℝ := initial_sugar_volume + added_sugar
  let final_volume : ℝ := initial_volume + added_sugar + added_water + added_kola
  let final_sugar_percentage : ℝ := final_sugar_volume / final_volume
  ∃ ε > 0, |final_sugar_percentage - 0.1967| < ε :=
by sorry

end sugar_percentage_after_addition_l204_20425


namespace first_part_speed_l204_20495

def trip_length : ℝ := 12
def part_time : ℝ := 0.25  -- 15 minutes in hours
def second_part_speed : ℝ := 12
def third_part_speed : ℝ := 20

theorem first_part_speed :
  ∃ (v : ℝ), v * part_time + second_part_speed * part_time + third_part_speed * part_time = trip_length ∧ v = 16 := by
  sorry

end first_part_speed_l204_20495


namespace twelve_sixteen_twenty_pythagorean_triple_l204_20487

/-- A Pythagorean triple is a set of three positive integers (a, b, c) that satisfy a² + b² = c² -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The set {12, 16, 20} is a Pythagorean triple -/
theorem twelve_sixteen_twenty_pythagorean_triple :
  isPythagoreanTriple 12 16 20 := by
  sorry

end twelve_sixteen_twenty_pythagorean_triple_l204_20487


namespace nancy_history_marks_l204_20444

def american_literature : ℕ := 66
def home_economics : ℕ := 52
def physical_education : ℕ := 68
def art : ℕ := 89
def average_marks : ℕ := 70
def total_subjects : ℕ := 5

theorem nancy_history_marks :
  ∃ history : ℕ,
    history = average_marks * total_subjects - (american_literature + home_economics + physical_education + art) ∧
    history = 75 := by
  sorry

end nancy_history_marks_l204_20444


namespace winter_sports_camp_l204_20418

theorem winter_sports_camp (total_students : ℕ) (boys girls : ℕ) (pine_students oak_students : ℕ)
  (seventh_grade eighth_grade : ℕ) (pine_girls : ℕ) :
  total_students = 120 →
  boys = 70 →
  girls = 50 →
  pine_students = 70 →
  oak_students = 50 →
  seventh_grade = 60 →
  eighth_grade = 60 →
  pine_girls = 30 →
  pine_students / 2 = seventh_grade →
  ∃ (oak_eighth_boys : ℕ), oak_eighth_boys = 15 :=
by sorry

end winter_sports_camp_l204_20418


namespace line_through_points_l204_20409

/-- A line passing through (0, -2) and (1, 0) also passes through (7, b). Prove b = 12. -/
theorem line_through_points (b : ℝ) : 
  (∃ m c : ℝ, (0 = m * 0 + c ∧ -2 = m * 0 + c) ∧ 
              (0 = m * 1 + c) ∧ 
              (b = m * 7 + c)) → 
  b = 12 := by
  sorry

end line_through_points_l204_20409


namespace fourth_buoy_distance_l204_20434

/-- Given buoys placed at even intervals in the ocean, with the third buoy 72 meters from the beach,
    this theorem proves that the fourth buoy is 108 meters from the beach. -/
theorem fourth_buoy_distance (interval : ℝ) (h1 : interval > 0) (h2 : 3 * interval = 72) :
  4 * interval = 108 := by
  sorry

end fourth_buoy_distance_l204_20434


namespace cookies_theorem_l204_20452

/-- The number of cookies Mona brought -/
def mona_cookies : ℕ := 20

/-- The number of cookies Jasmine brought -/
def jasmine_cookies : ℕ := mona_cookies - 5

/-- The number of cookies Rachel brought -/
def rachel_cookies : ℕ := jasmine_cookies + 10

/-- The total number of cookies brought by Mona, Jasmine, and Rachel -/
def total_cookies : ℕ := mona_cookies + jasmine_cookies + rachel_cookies

theorem cookies_theorem : total_cookies = 60 := by
  sorry

end cookies_theorem_l204_20452


namespace expression_evaluation_l204_20498

theorem expression_evaluation (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) :
  let y := 1 / x + z
  (x - 1 / x) * (y + 1 / y) = ((x^2 - 1) * (1 + 2*x*z + x^2*z^2 + x^2)) / (x^2 * (1 + x*z)) :=
by sorry

end expression_evaluation_l204_20498


namespace birds_on_fence_l204_20427

theorem birds_on_fence (initial_birds : ℕ) (new_birds : ℕ) : 
  initial_birds = 1 → new_birds = 4 → initial_birds + new_birds = 5 := by
  sorry

end birds_on_fence_l204_20427


namespace lemons_for_combined_beverages_l204_20486

/-- The number of lemons needed for a given amount of lemonade and limeade -/
def lemons_needed (lemonade_gallons : ℚ) (limeade_gallons : ℚ) : ℚ :=
  let lemons_per_gallon_lemonade : ℚ := 36 / 48
  let lemons_per_gallon_limeade : ℚ := 2 * lemons_per_gallon_lemonade
  lemonade_gallons * lemons_per_gallon_lemonade + limeade_gallons * lemons_per_gallon_limeade

/-- Theorem stating the number of lemons needed for 18 gallons of combined lemonade and limeade -/
theorem lemons_for_combined_beverages :
  lemons_needed 9 9 = 81/4 := by
  sorry

#eval lemons_needed 9 9

end lemons_for_combined_beverages_l204_20486


namespace complex_determinant_equation_l204_20449

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the theorem
theorem complex_determinant_equation :
  ∀ z : ℂ, det 1 (-1) z (z * Complex.I) = 2 → z = 1 - Complex.I := by
  sorry

end complex_determinant_equation_l204_20449


namespace probability_identical_cubes_value_l204_20441

/-- Represents a cube with 8 faces, each face can be painted with one of three colors -/
structure Cube :=
  (faces : Fin 8 → Fin 3)

/-- The total number of ways to paint two cubes -/
def total_paintings : ℕ := 3^8 * 3^8

/-- The number of ways to paint two cubes so they look identical after rotation -/
def identical_paintings : ℕ := 831

/-- The probability that two cubes look identical after painting and possible rotations -/
def probability_identical_cubes : ℚ :=
  identical_paintings / total_paintings

theorem probability_identical_cubes_value :
  probability_identical_cubes = 831 / 43046721 :=
sorry

end probability_identical_cubes_value_l204_20441


namespace intersecting_circles_theorem_l204_20426

/-- Two circles intersecting at two distinct points theorem -/
theorem intersecting_circles_theorem 
  (r a b x₁ y₁ x₂ y₂ : ℝ) 
  (hr : r > 0)
  (hab : a ≠ 0 ∨ b ≠ 0)
  (hC₁_A : x₁^2 + y₁^2 = r^2)
  (hC₂_A : (x₁ + a)^2 + (y₁ + b)^2 = r^2)
  (hC₁_B : x₂^2 + y₂^2 = r^2)
  (hC₂_B : (x₂ + a)^2 + (y₂ + b)^2 = r^2)
  (hAB_distinct : (x₁, y₁) ≠ (x₂, y₂)) :
  (2*a*x₁ + 2*b*y₁ + a^2 + b^2 = 0) ∧ 
  (a*(x₁ - x₂) + b*(y₁ - y₂) = 0) ∧ 
  (x₁ + x₂ = -a ∧ y₁ + y₂ = -b) := by
  sorry

end intersecting_circles_theorem_l204_20426


namespace min_value_expression_l204_20436

theorem min_value_expression (x y : ℝ) : 
  (x * y - 2)^2 + (x + y - 1)^2 ≥ 0 ∧ 
  ∃ (a b : ℝ), (a * b - 2)^2 + (a + b - 1)^2 = 0 := by
  sorry

end min_value_expression_l204_20436


namespace scaled_determinant_l204_20461

theorem scaled_determinant (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 12 →
  Matrix.det !![3*x, 3*y; 3*z, 3*w] = 108 := by
  sorry

end scaled_determinant_l204_20461


namespace gcd_problem_l204_20459

theorem gcd_problem (h : Prime 103) : 
  Nat.gcd (103^7 + 1) (103^7 + 103^5 + 1) = 1 := by
  sorry

end gcd_problem_l204_20459


namespace equality_iff_inequality_l204_20438

theorem equality_iff_inequality (x : ℝ) : (x - 2) * (x - 3) ≤ 0 ↔ |x - 2| + |x - 3| = 1 := by
  sorry

end equality_iff_inequality_l204_20438


namespace f_properties_l204_20448

-- Define the function f(x) = x ln|x|
noncomputable def f (x : ℝ) : ℝ := x * Real.log (abs x)

-- Define the function g(x) = f(x) - m
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f x - m

theorem f_properties :
  (∀ x y, x < y ∧ x < -1/Real.exp 1 ∧ y < -1/Real.exp 1 → f x < f y) ∧
  (∀ m : ℝ, ∃ n : ℕ, n ≤ 3 ∧ (∃ s : Finset ℝ, s.card = n ∧ ∀ x ∈ s, g m x = 0) ∧
    ∀ s : Finset ℝ, (∀ x ∈ s, g m x = 0) → s.card ≤ n) :=
sorry

end f_properties_l204_20448


namespace base8_sum_3_to_100_l204_20408

/-- Converts a base 8 number to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 8 -/
def base10ToBase8 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSequenceSum (first last n : ℕ) : ℕ :=
  n * (first + last) / 2

theorem base8_sum_3_to_100 :
  let first := 3
  let last := base8ToBase10 100
  let n := last - first + 1
  base10ToBase8 (arithmeticSequenceSum first last n) = 4035 := by
  sorry

end base8_sum_3_to_100_l204_20408


namespace sam_initial_pennies_l204_20428

/-- The number of pennies Sam spent -/
def pennies_spent : ℕ := 93

/-- The number of pennies Sam has left -/
def pennies_left : ℕ := 5

/-- The initial number of pennies in Sam's bank -/
def initial_pennies : ℕ := pennies_spent + pennies_left

theorem sam_initial_pennies : initial_pennies = 98 := by
  sorry

end sam_initial_pennies_l204_20428


namespace steven_peach_count_l204_20469

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := 9

/-- The difference in peaches between Steven and Jake -/
def steven_jake_diff : ℕ := 7

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := jake_peaches + steven_jake_diff

theorem steven_peach_count : steven_peaches = 16 := by
  sorry

end steven_peach_count_l204_20469


namespace enthalpy_change_reaction_l204_20405

/-- Standard enthalpy of formation for Na₂O (s) in kJ/mol -/
def ΔH_f_Na2O : ℝ := -416

/-- Standard enthalpy of formation for H₂O (l) in kJ/mol -/
def ΔH_f_H2O : ℝ := -286

/-- Standard enthalpy of formation for NaOH (s) in kJ/mol -/
def ΔH_f_NaOH : ℝ := -427.8

/-- Standard enthalpy change of the reaction Na₂O + H₂O → 2NaOH at 298 K -/
def ΔH_reaction : ℝ := 2 * ΔH_f_NaOH - (ΔH_f_Na2O + ΔH_f_H2O)

theorem enthalpy_change_reaction :
  ΔH_reaction = -153.6 := by sorry

end enthalpy_change_reaction_l204_20405


namespace subset_of_sqrt_eleven_l204_20494

theorem subset_of_sqrt_eleven (h : Real.sqrt 11 < 2 * Real.sqrt 3) :
  {Real.sqrt 11} ⊆ {x : ℝ | |x| ≤ 2 * Real.sqrt 3} := by
  sorry

end subset_of_sqrt_eleven_l204_20494


namespace total_payment_for_bikes_l204_20443

-- Define the payment for painting a bike
def paint_payment : ℕ := 5

-- Define the additional payment for selling a bike
def sell_additional : ℕ := 8

-- Define the number of bikes
def num_bikes : ℕ := 8

-- Theorem to prove
theorem total_payment_for_bikes : 
  (paint_payment + (paint_payment + sell_additional)) * num_bikes = 144 := by
  sorry

end total_payment_for_bikes_l204_20443


namespace total_collection_theorem_l204_20471

def group_size : ℕ := 77

def member_contribution (n : ℕ) : ℕ := n

def total_collection_paise (n : ℕ) : ℕ := n * member_contribution n

def paise_to_rupees (p : ℕ) : ℚ := p / 100

theorem total_collection_theorem :
  paise_to_rupees (total_collection_paise group_size) = 59.29 := by
  sorry

end total_collection_theorem_l204_20471


namespace polynomial_not_equal_77_l204_20475

theorem polynomial_not_equal_77 (x y : ℤ) : 
  x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end polynomial_not_equal_77_l204_20475


namespace factorial_division_l204_20481

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 4 = 151200 := by
  sorry

end factorial_division_l204_20481


namespace percent_difference_z_x_of_w_l204_20488

theorem percent_difference_z_x_of_w (y q w x z : ℝ) 
  (hw : w = 0.60 * q)
  (hq : q = 0.60 * y)
  (hz : z = 0.54 * y)
  (hx : x = 1.30 * w) :
  (z - x) / w = 0.20 := by
sorry

end percent_difference_z_x_of_w_l204_20488


namespace parabola_ellipse_focus_l204_20492

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the parabola
def parabola (x y p : ℝ) : Prop := y^2 = 2 * p * x

-- Define the right focus of the ellipse
def right_focus (x y : ℝ) : Prop := ellipse x y ∧ x > 0 ∧ y = 0

-- Define the focus of the parabola
def parabola_focus (x y p : ℝ) : Prop := x = p / 2 ∧ y = 0

-- The main theorem
theorem parabola_ellipse_focus (p : ℝ) :
  p > 0 →
  (∃ x y, right_focus x y ∧ parabola_focus x y p) →
  p = 4 := by
  sorry

end parabola_ellipse_focus_l204_20492


namespace woodworker_productivity_increase_l204_20451

/-- Woodworker's productivity increase problem -/
theorem woodworker_productivity_increase
  (normal_days : ℕ)
  (normal_parts : ℕ)
  (new_days : ℕ)
  (extra_parts : ℕ)
  (h1 : normal_days = 24)
  (h2 : normal_parts = 360)
  (h3 : new_days = 22)
  (h4 : extra_parts = 80) :
  (normal_parts + extra_parts) / new_days - normal_parts / normal_days = 5 :=
by sorry

end woodworker_productivity_increase_l204_20451


namespace gaspard_empty_bags_iff_even_sum_l204_20477

/-- Represents the state of the bags -/
structure BagState where
  m : ℕ
  n : ℕ

/-- Defines the allowed operations on the bags -/
inductive Operation
  | RemoveEqual : ℕ → Operation
  | TripleOne : Bool → Operation

/-- Applies an operation to a bag state -/
def applyOperation (state : BagState) (op : Operation) : BagState :=
  match op with
  | Operation.RemoveEqual k => ⟨state.m - k, state.n - k⟩
  | Operation.TripleOne true => ⟨3 * state.m, state.n⟩
  | Operation.TripleOne false => ⟨state.m, 3 * state.n⟩

/-- Defines when a bag state is empty -/
def isEmptyState (state : BagState) : Prop :=
  state.m = 0 ∧ state.n = 0

/-- Defines when a sequence of operations can empty the bags -/
def canEmpty (initialState : BagState) : Prop :=
  ∃ (ops : List Operation), isEmptyState (ops.foldl applyOperation initialState)

/-- The main theorem: Gaspard can empty both bags iff m + n is even -/
theorem gaspard_empty_bags_iff_even_sum (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) :
    canEmpty ⟨m, n⟩ ↔ Even (m + n) := by
  sorry


end gaspard_empty_bags_iff_even_sum_l204_20477


namespace single_point_conic_section_l204_20424

theorem single_point_conic_section (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + p.2^2 + 6 * p.1 - 6 * p.2 + d = 0) → d = 12 := by
  sorry

end single_point_conic_section_l204_20424


namespace regular_polyhedron_spheres_l204_20463

/-- A regular polyhedron -/
structure RegularPolyhedron where
  -- We don't need to define the internal structure,
  -- as the problem doesn't rely on specific properties

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points -/
def distance (p q : Point3D) : ℝ :=
  sorry

/-- Distance from a point to a face of the polyhedron -/
def distanceToFace (p : Point3D) (poly : RegularPolyhedron) (face : Nat) : ℝ :=
  sorry

/-- Get a vertex of the polyhedron -/
def getVertex (poly : RegularPolyhedron) (v : Nat) : Point3D :=
  sorry

/-- Number of vertices in the polyhedron -/
def numVertices (poly : RegularPolyhedron) : Nat :=
  sorry

/-- Number of faces in the polyhedron -/
def numFaces (poly : RegularPolyhedron) : Nat :=
  sorry

/-- Theorem: For any regular polyhedron, there exists a point O such that
    1) The distance from O to all vertices is constant
    2) The distance from O to all faces is constant -/
theorem regular_polyhedron_spheres (poly : RegularPolyhedron) :
  ∃ (O : Point3D),
    (∀ (i j : Nat), i < numVertices poly → j < numVertices poly →
      distance O (getVertex poly i) = distance O (getVertex poly j)) ∧
    (∀ (i j : Nat), i < numFaces poly → j < numFaces poly →
      distanceToFace O poly i = distanceToFace O poly j) :=
by sorry

end regular_polyhedron_spheres_l204_20463
