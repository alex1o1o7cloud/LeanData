import Mathlib

namespace arrangement_count_proof_l144_14463

/-- The number of ways to arrange 4 men and 4 women into two indistinguishable groups
    of two (each containing one man and one woman) and one group of four
    (containing the remaining two men and two women) -/
def arrangement_count : ℕ := 72

/-- The number of ways to choose one man from 4 men -/
def choose_man : ℕ := 4

/-- The number of ways to choose one woman from 4 women -/
def choose_woman : ℕ := 4

/-- The number of ways to choose one man from 3 remaining men -/
def choose_remaining_man : ℕ := 3

/-- The number of ways to choose one woman from 3 remaining women -/
def choose_remaining_woman : ℕ := 3

/-- The number of ways to arrange two indistinguishable groups -/
def indistinguishable_groups : ℕ := 2

theorem arrangement_count_proof :
  arrangement_count = (choose_man * choose_woman * choose_remaining_man * choose_remaining_woman) / indistinguishable_groups :=
by sorry

end arrangement_count_proof_l144_14463


namespace permutation_inequality_solution_l144_14415

def A (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

theorem permutation_inequality_solution :
  ∃! x : ℕ+, A 8 x < 6 * A 8 (x - 2) ∧ x = 8 := by sorry

end permutation_inequality_solution_l144_14415


namespace order_of_logarithmic_expressions_l144_14431

theorem order_of_logarithmic_expressions :
  let a : ℝ := (Real.log (Real.sqrt 2)) / 2
  let b : ℝ := (Real.log 3) / 6
  let c : ℝ := 1 / (2 * Real.exp 1)
  c > b ∧ b > a := by sorry

end order_of_logarithmic_expressions_l144_14431


namespace output_is_27_l144_14439

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 2
  if step1 ≤ 22 then
    step1 + 8
  else
    step1 + 3

theorem output_is_27 : function_machine 12 = 27 := by
  sorry

end output_is_27_l144_14439


namespace cake_remainder_cake_problem_l144_14480

theorem cake_remainder (john_ate : ℚ) (emily_took_half : Bool) : ℚ :=
  by
    -- Define John's portion
    have john_portion : ℚ := 3/5
    
    -- Define the remaining portion after John ate
    have remaining_after_john : ℚ := 1 - john_portion
    
    -- Define Emily's portion
    have emily_portion : ℚ := remaining_after_john / 2
    
    -- Calculate the final remaining portion
    have final_remaining : ℚ := remaining_after_john - emily_portion
    
    -- Prove that the final remaining portion is 1/5 (20%)
    sorry

-- State the theorem
theorem cake_problem : cake_remainder (3/5) true = 1/5 :=
  by sorry

end cake_remainder_cake_problem_l144_14480


namespace new_perimeter_after_triangle_rotation_l144_14496

/-- Given a square with perimeter 48 inches and a right isosceles triangle with legs 12 inches,
    prove that removing the triangle and reattaching it results in a figure with perimeter 36 + 12√2 inches -/
theorem new_perimeter_after_triangle_rotation (square_perimeter : ℝ) (triangle_leg : ℝ) : 
  square_perimeter = 48 → triangle_leg = 12 → 
  36 + 12 * Real.sqrt 2 = square_perimeter - triangle_leg + Real.sqrt (2 * triangle_leg^2) :=
by sorry

end new_perimeter_after_triangle_rotation_l144_14496


namespace function_value_theorem_l144_14450

theorem function_value_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = Real.sqrt (2 * x + 1)) →
  f a = 5 →
  a = 12 := by
  sorry

end function_value_theorem_l144_14450


namespace third_term_value_l144_14465

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem third_term_value
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = -11)
  (h_sum : a 4 + a 6 = -6) :
  a 3 = -7 :=
sorry

end third_term_value_l144_14465


namespace negation_of_proposition_l144_14448

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 1 → x^2 - 1 > 0)) ↔ (∃ x₀ : ℝ, x₀ > 1 ∧ x₀^2 - 1 ≤ 0) :=
by sorry

end negation_of_proposition_l144_14448


namespace tree_increase_factor_l144_14428

theorem tree_increase_factor (initial_maples : ℝ) (initial_lindens : ℝ) 
  (spring_total : ℝ) (autumn_total : ℝ) : 
  initial_maples / (initial_maples + initial_lindens) = 3/5 →
  initial_maples / spring_total = 1/5 →
  initial_maples / autumn_total = 3/5 →
  autumn_total / (initial_maples + initial_lindens) = 6 :=
by
  sorry

end tree_increase_factor_l144_14428


namespace irrational_element_existence_l144_14479

open Set Real

theorem irrational_element_existence
  (a b : ℚ)
  (M : Set ℝ)
  (hab : 0 < a ∧ a < b)
  (hM : ∀ (x y : ℝ), x ∈ M → y ∈ M → Real.sqrt (x * y) ∈ M)
  (haM : (a : ℝ) ∈ M)
  (hbM : (b : ℝ) ∈ M) :
  ∀ (c d : ℝ), (a : ℝ) < c → c < d → d < (b : ℝ) →
  ∃ (m : ℝ), m ∈ M ∧ Irrational m ∧ c < m ∧ m < d :=
sorry

end irrational_element_existence_l144_14479


namespace arithmetic_expression_equality_l144_14487

theorem arithmetic_expression_equality : 5 + 16 / 4 - 3^2 = 0 := by
  sorry

end arithmetic_expression_equality_l144_14487


namespace complex_fraction_simplification_l144_14458

/-- Given that i² = -1, prove that (2-i)/(1+4i) = -2/17 - 9/17*i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (2 - i) / (1 + 4*i) = -2/17 - 9/17*i := by sorry

end complex_fraction_simplification_l144_14458


namespace oil_change_time_is_15_minutes_l144_14426

/-- Represents the time in minutes for various car maintenance tasks -/
structure CarMaintenanceTimes where
  washTime : ℕ
  oilChangeTime : ℕ
  tireChangeTime : ℕ

/-- Represents the number of tasks performed -/
structure TasksCounts where
  carsWashed : ℕ
  oilChanges : ℕ
  tireChanges : ℕ

/-- Calculates the total time spent on tasks -/
def totalTime (times : CarMaintenanceTimes) (counts : TasksCounts) : ℕ :=
  times.washTime * counts.carsWashed +
  times.oilChangeTime * counts.oilChanges +
  times.tireChangeTime * counts.tireChanges

/-- The main theorem to prove -/
theorem oil_change_time_is_15_minutes 
  (times : CarMaintenanceTimes)
  (counts : TasksCounts)
  (h1 : times.washTime = 10)
  (h2 : times.tireChangeTime = 30)
  (h3 : counts.carsWashed = 9)
  (h4 : counts.oilChanges = 6)
  (h5 : counts.tireChanges = 2)
  (h6 : totalTime times counts = 4 * 60) :
  times.oilChangeTime = 15 := by
  sorry


end oil_change_time_is_15_minutes_l144_14426


namespace eli_calculation_l144_14443

theorem eli_calculation (x : ℝ) (h : (8 * x - 7) / 5 = 63) : (5 * x - 7) / 8 = 24.28125 := by
  sorry

end eli_calculation_l144_14443


namespace sum_of_powers_l144_14491

theorem sum_of_powers (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^2013 + (y / (x + y))^2013 = -2 := by
  sorry

end sum_of_powers_l144_14491


namespace books_bought_two_years_ago_l144_14419

/-- Represents the number of books in a library over time --/
structure LibraryBooks where
  initial : ℕ  -- Initial number of books 5 years ago
  bought_two_years_ago : ℕ  -- Books bought 2 years ago
  bought_last_year : ℕ  -- Books bought last year
  donated : ℕ  -- Books donated this year
  current : ℕ  -- Current number of books

/-- Theorem stating the number of books bought two years ago --/
theorem books_bought_two_years_ago 
  (lib : LibraryBooks) 
  (h1 : lib.initial = 500)
  (h2 : lib.bought_last_year = lib.bought_two_years_ago + 100)
  (h3 : lib.donated = 200)
  (h4 : lib.current = 1000)
  (h5 : lib.current = lib.initial + lib.bought_two_years_ago + lib.bought_last_year - lib.donated) :
  lib.bought_two_years_ago = 300 := by
  sorry

#check books_bought_two_years_ago

end books_bought_two_years_ago_l144_14419


namespace problem_solution_l144_14499

theorem problem_solution (a b m n : ℚ) 
  (ha_neg : a < 0) 
  (ha_abs : |a| = 7/4) 
  (hb_recip : b⁻¹ = -3/2) 
  (hmn_opp : m = -n) : 
  4 * a / b + 3 * (m + n) = 21/2 := by
  sorry

end problem_solution_l144_14499


namespace prob_A3_given_white_l144_14406

structure Urn :=
  (white : ℕ)
  (black : ℕ)

def total_urns : ℕ := 12

def urns : Fin 4 → (ℕ × Urn)
  | 0 => (6, ⟨3, 4⟩)  -- A₁
  | 1 => (3, ⟨2, 8⟩)  -- A₂
  | 2 => (2, ⟨6, 1⟩)  -- A₃
  | 3 => (1, ⟨4, 3⟩)  -- A₄

def prob_select_urn (i : Fin 4) : ℚ :=
  (urns i).1 / total_urns

def prob_white_given_urn (i : Fin 4) : ℚ :=
  (urns i).2.white / ((urns i).2.white + (urns i).2.black)

def prob_white : ℚ :=
  Finset.sum Finset.univ (λ i => prob_select_urn i * prob_white_given_urn i)

theorem prob_A3_given_white :
  (prob_select_urn 2 * prob_white_given_urn 2) / prob_white = 30 / 73 := by
  sorry

end prob_A3_given_white_l144_14406


namespace trip_cost_is_127_l144_14475

/-- Represents a car with its specifications and trip details -/
structure Car where
  efficiency : ℝ  -- miles per gallon
  tankCapacity : ℝ  -- gallons
  initialMileage : ℝ  -- miles
  firstFillUpPrice : ℝ  -- dollars per gallon
  secondFillUpPrice : ℝ  -- dollars per gallon

/-- Calculates the total cost of a road trip given a car's specifications -/
def totalTripCost (c : Car) : ℝ :=
  c.tankCapacity * (c.firstFillUpPrice + c.secondFillUpPrice)

/-- Theorem stating that the total cost of the trip is $127.00 -/
theorem trip_cost_is_127 (c : Car) 
    (h1 : c.efficiency = 30)
    (h2 : c.tankCapacity = 20)
    (h3 : c.initialMileage = 1728)
    (h4 : c.firstFillUpPrice = 3.1)
    (h5 : c.secondFillUpPrice = 3.25) :
  totalTripCost c = 127 := by
  sorry

#eval totalTripCost { efficiency := 30, tankCapacity := 20, initialMileage := 1728, firstFillUpPrice := 3.1, secondFillUpPrice := 3.25 }

end trip_cost_is_127_l144_14475


namespace nail_sizes_sum_l144_14404

theorem nail_sizes_sum (size_2d : ℚ) (size_4d : ℚ) (size_6d : ℚ) (size_8d : ℚ) 
  (h1 : size_2d = 1/5)
  (h2 : size_4d = 3/10)
  (h3 : size_6d = 1/4)
  (h4 : size_8d = 1/8) :
  size_2d + size_4d = 1/2 := by
sorry

end nail_sizes_sum_l144_14404


namespace carolyn_sum_is_18_l144_14482

/-- Represents the game state -/
structure GameState where
  remaining : List Nat
  carolyn_sum : Nat

/-- Represents a player's move -/
inductive Move
  | Remove (n : Nat)

/-- Applies Carolyn's move to the game state -/
def apply_carolyn_move (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.Remove n =>
    { remaining := state.remaining.filter (· ≠ n),
      carolyn_sum := state.carolyn_sum + n }

/-- Applies Paul's move to the game state -/
def apply_paul_move (state : GameState) (move : List Move) : GameState :=
  match move with
  | [] => state
  | (Move.Remove n) :: rest =>
    apply_paul_move
      { remaining := state.remaining.filter (· ≠ n),
        carolyn_sum := state.carolyn_sum }
      rest

/-- Checks if a number has a divisor in the list -/
def has_divisor_in_list (n : Nat) (list : List Nat) : Bool :=
  list.any (fun m => m ≠ n && n % m == 0)

/-- Simulates the game -/
def play_game (initial_state : GameState) : Nat :=
  let state1 := apply_carolyn_move initial_state (Move.Remove 4)
  let state2 := apply_paul_move state1 [Move.Remove 1, Move.Remove 2]
  let state3 := apply_carolyn_move state2 (Move.Remove 6)
  let state4 := apply_paul_move state3 [Move.Remove 3]
  let state5 := apply_carolyn_move state4 (Move.Remove 8)
  let final_state := apply_paul_move state5 [Move.Remove 5, Move.Remove 7]
  final_state.carolyn_sum

theorem carolyn_sum_is_18 :
  let initial_state : GameState := { remaining := [1, 2, 3, 4, 5, 6, 7, 8], carolyn_sum := 0 }
  play_game initial_state = 18 := by
  sorry

end carolyn_sum_is_18_l144_14482


namespace gcd_lcm_sum_120_4620_l144_14472

theorem gcd_lcm_sum_120_4620 : Nat.gcd 120 4620 + Nat.lcm 120 4620 = 4680 := by
  sorry

end gcd_lcm_sum_120_4620_l144_14472


namespace binomial_distribution_problem_l144_14430

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The random variable X following a binomial distribution -/
def X (b : BinomialDistribution) : ℝ := sorry

/-- Expectation of a random variable -/
def expectation (X : ℝ) : ℝ := sorry

/-- Variance of a random variable -/
def variance (X : ℝ) : ℝ := sorry

theorem binomial_distribution_problem (b : BinomialDistribution) 
  (h2 : expectation (3 * X b - 9) = 27)
  (h3 : variance (3 * X b - 9) = 27) :
  b.n = 16 ∧ b.p = 3/4 := by sorry

end binomial_distribution_problem_l144_14430


namespace hyperbola_standard_equation_l144_14441

/-- The standard equation of a hyperbola given its asymptotes and shared foci with an ellipse -/
theorem hyperbola_standard_equation
  (asymptote_slope : ℝ)
  (ellipse_a : ℝ)
  (ellipse_b : ℝ)
  (h_asymptote : asymptote_slope = 2)
  (h_ellipse : ellipse_a^2 = 49 ∧ ellipse_b^2 = 24) :
  ∃ (a b : ℝ), a^2 = 25 ∧ b^2 = 100 ∧
    ∀ (x y : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 :=
by sorry

end hyperbola_standard_equation_l144_14441


namespace seeds_per_flower_bed_l144_14452

theorem seeds_per_flower_bed 
  (total_seeds : ℕ) 
  (num_flower_beds : ℕ) 
  (h1 : total_seeds = 60) 
  (h2 : num_flower_beds = 6) 
  : total_seeds / num_flower_beds = 10 := by
  sorry

end seeds_per_flower_bed_l144_14452


namespace price_reduction_percentage_l144_14447

theorem price_reduction_percentage (P : ℝ) (S : ℝ) (h1 : P > 0) (h2 : S > 0) :
  let new_sales := 1.80 * S
  let new_revenue := 1.08 * (P * S)
  let new_price := new_revenue / new_sales
  (P - new_price) / P = 0.40 := by
sorry

end price_reduction_percentage_l144_14447


namespace tangent_at_one_l144_14402

/-- A polynomial function of degree 4 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^3 + 1

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 3 * b * x^2

theorem tangent_at_one (a b : ℝ) : 
  (f a b 1 = 0 ∧ f' a b 1 = 0) ↔ (a = 3 ∧ b = -4) := by sorry

end tangent_at_one_l144_14402


namespace gretchen_scuba_trips_l144_14433

/-- The minimum number of trips required to transport a given number of objects,
    where each trip can carry a fixed number of objects. -/
def min_trips (total_objects : ℕ) (objects_per_trip : ℕ) : ℕ :=
  (total_objects + objects_per_trip - 1) / objects_per_trip

/-- Theorem stating that 6 trips are required to transport 17 objects
    when carrying 3 objects per trip. -/
theorem gretchen_scuba_trips :
  min_trips 17 3 = 6 := by
  sorry

end gretchen_scuba_trips_l144_14433


namespace min_squares_13x13_l144_14483

/-- Represents a square on a grid -/
structure GridSquare where
  size : Nat
  deriving Repr

/-- The original square size -/
def originalSize : Nat := 13

/-- A list of squares that the original square is divided into -/
def divisionList : List GridSquare := [
  {size := 6},
  {size := 5},
  {size := 4},
  {size := 3},
  {size := 2},
  {size := 2},
  {size := 1},
  {size := 1},
  {size := 1},
  {size := 1},
  {size := 1}
]

/-- The number of squares in the division -/
def numSquares : Nat := divisionList.length

/-- Checks if the division is valid (covers the entire original square) -/
def isValidDivision (list : List GridSquare) : Prop :=
  list.foldl (fun acc square => acc + square.size * square.size) 0 = originalSize * originalSize

/-- Theorem: The minimum number of squares a 13x13 square can be divided into is 11 -/
theorem min_squares_13x13 :
  (isValidDivision divisionList) ∧
  (∀ (otherList : List GridSquare), isValidDivision otherList → otherList.length ≥ numSquares) :=
sorry

end min_squares_13x13_l144_14483


namespace subtract_from_forty_squared_l144_14440

theorem subtract_from_forty_squared (n : ℕ) (h : n = 40 - 1) : n^2 = 40^2 - 79 := by
  sorry

end subtract_from_forty_squared_l144_14440


namespace altitude_and_median_equations_l144_14461

/-- Triangle with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : ℝ × ℝ := (4, 0)
  B : ℝ × ℝ := (6, 7)
  C : ℝ × ℝ := (0, 3)

/-- Line represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Altitude from B to side AC -/
def altitude (t : Triangle) : Line :=
  { a := 3, b := 2, c := -12 }

/-- Median from B to side AC -/
def median (t : Triangle) : Line :=
  { a := 4, b := -6, c := 1 }

/-- Theorem stating that the altitude and median equations are correct -/
theorem altitude_and_median_equations (t : Triangle) : 
  (altitude t = { a := 3, b := 2, c := -12 }) ∧ 
  (median t = { a := 4, b := -6, c := 1 }) := by
  sorry

end altitude_and_median_equations_l144_14461


namespace min_value_of_f_l144_14411

theorem min_value_of_f (x : ℝ) (hx : x < 0) : 
  ∃ (m : ℝ), (∀ y, y < 0 → -y - 2/y ≥ m) ∧ (∃ z, z < 0 ∧ -z - 2/z = m) ∧ m = 2 * Real.sqrt 2 := by
  sorry

end min_value_of_f_l144_14411


namespace zero_additive_identity_for_integers_l144_14412

theorem zero_additive_identity_for_integers : 
  ∃! y : ℤ, ∀ x : ℤ, y + x = x :=
by sorry

end zero_additive_identity_for_integers_l144_14412


namespace coloring_periodicity_l144_14478

-- Define a circle with n equal arcs
def Circle (n : ℕ) := Fin n

-- Define a coloring of the circle
def Coloring (n : ℕ) := Circle n → ℕ

-- Define a rotation of the circle
def rotate (n : ℕ) (k : ℕ) (i : Circle n) : Circle n :=
  ⟨(i.val + k) % n, by sorry⟩

-- Define when two arcs are identically colored
def identically_colored (n : ℕ) (c : Coloring n) (i j k l : Circle n) : Prop :=
  ∃ m : ℕ, ∀ t : ℕ, c (rotate n m ⟨(i.val + t) % n, by sorry⟩) = c ⟨(k.val + t) % n, by sorry⟩

-- Define the condition for each division point
def condition_for_each_point (n : ℕ) (c : Coloring n) : Prop :=
  ∀ k : Circle n, ∃ i j : Circle n, 
    i ≠ j ∧ 
    identically_colored n c k i k j ∧
    (∀ t : ℕ, t < i.val - k.val → c ⟨(k.val + t) % n, by sorry⟩ ≠ c ⟨(k.val + t + j.val - i.val) % n, by sorry⟩)

-- Define periodicity of the coloring
def is_periodic (n : ℕ) (c : Coloring n) : Prop :=
  ∃ p : ℕ, p > 0 ∧ p < n ∧ ∀ i : Circle n, c i = c ⟨(i.val + p) % n, by sorry⟩

-- The main theorem
theorem coloring_periodicity (n : ℕ) (c : Coloring n) :
  condition_for_each_point n c → is_periodic n c :=
by sorry

end coloring_periodicity_l144_14478


namespace ann_bill_money_problem_l144_14400

/-- Ann and Bill's money problem -/
theorem ann_bill_money_problem (bill_initial : ℕ) (transfer : ℕ) (ann_initial : ℕ) :
  bill_initial = 1111 →
  transfer = 167 →
  ann_initial + transfer = bill_initial - transfer →
  ann_initial = 777 := by
  sorry

end ann_bill_money_problem_l144_14400


namespace inequality_proof_l144_14492

theorem inequality_proof (a b c d : ℝ) (h : a > b ∧ b > c ∧ c > d) :
  1 / (a - b) + 1 / (b - c) + 1 / (c - d) ≥ 9 / (a - d) := by
  sorry

end inequality_proof_l144_14492


namespace least_subtraction_for_divisibility_problem_solution_l144_14473

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  let r := n % d
  (∀ k : Nat, k < r → ¬(d ∣ (n - k))) ∧ (d ∣ (n - r)) :=
by sorry

theorem problem_solution :
  let initial_number := 427398
  let divisor := 15
  let remainder := initial_number % divisor
  remainder = 3 ∧
  (∀ k : Nat, k < remainder → ¬(divisor ∣ (initial_number - k))) ∧
  (divisor ∣ (initial_number - remainder)) :=
by sorry

end least_subtraction_for_divisibility_problem_solution_l144_14473


namespace largest_subsequence_number_l144_14489

def original_number : ℕ := 778157260669103

def is_subsequence (sub seq : List ℕ) : Prop :=
  ∃ (l1 l2 : List ℕ), seq = l1 ++ sub ++ l2

def digits_to_nat (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

def nat_to_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

theorem largest_subsequence_number :
  let orig_digits := nat_to_digits original_number
  let result_digits := nat_to_digits 879103
  (result_digits.length = 6) ∧
  (is_subsequence result_digits orig_digits) ∧
  (∀ (other : List ℕ), other.length = 6 →
    is_subsequence other orig_digits →
    digits_to_nat other ≤ digits_to_nat result_digits) :=
by sorry

end largest_subsequence_number_l144_14489


namespace congruence_solution_l144_14442

theorem congruence_solution : ∃! n : ℕ, n < 47 ∧ (13 * n) % 47 = 9 % 47 := by
  sorry

end congruence_solution_l144_14442


namespace axis_of_symmetry_l144_14424

-- Define the parabola
def parabola (x : ℝ) : ℝ := (x - 5)^2 + 4

-- State the theorem
theorem axis_of_symmetry :
  ∀ x : ℝ, parabola (5 + x) = parabola (5 - x) := by sorry

end axis_of_symmetry_l144_14424


namespace third_competitor_hotdogs_l144_14401

/-- The number of hotdogs the third competitor can eat in a given time -/
def hotdogs_eaten_by_third (first_rate : ℕ) (second_multiplier third_multiplier time : ℕ) : ℕ :=
  first_rate * second_multiplier * third_multiplier * time

/-- Theorem: The third competitor eats 300 hotdogs in 5 minutes -/
theorem third_competitor_hotdogs :
  hotdogs_eaten_by_third 10 3 2 5 = 300 := by
  sorry

#eval hotdogs_eaten_by_third 10 3 2 5

end third_competitor_hotdogs_l144_14401


namespace triangle_abc_properties_l144_14488

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- Angles are positive
  A + B + C = π ∧ -- Sum of angles in a triangle
  0 < a ∧ 0 < b ∧ 0 < c ∧ -- Sides are positive
  b * Real.sin (2 * C) = c * Real.sin B ∧ -- Given condition
  Real.sin (B - π / 3) = 3 / 5 -- Given condition
  →
  C = π / 3 ∧ 
  Real.sin A = (4 * Real.sqrt 3 - 3) / 10 :=
by sorry

end triangle_abc_properties_l144_14488


namespace unusual_coin_probability_l144_14476

theorem unusual_coin_probability (p q : ℝ) : 
  0 ≤ p ∧ 0 ≤ q ∧ q ≤ p ∧ p + q + 1/6 = 1 ∧ 
  p^2 + q^2 + (1/6)^2 = 1/2 → 
  p = 2/3 := by sorry

end unusual_coin_probability_l144_14476


namespace min_distance_circle_to_line_l144_14464

theorem min_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*x - 2*y + 1 = 0}
  let line := {(x, y) : ℝ × ℝ | 3*x + 4*y + 8 = 0}
  (∀ p ∈ circle, ∃ q ∈ line, ∀ r ∈ line, dist p q ≤ dist p r) ∧
  (∃ p ∈ circle, ∃ q ∈ line, dist p q = 2) ∧
  (∀ p ∈ circle, ∀ q ∈ line, dist p q ≥ 2) :=
by sorry

where
  dist : ℝ × ℝ → ℝ × ℝ → ℝ := λ (x₁, y₁) (x₂, y₂) => Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

end min_distance_circle_to_line_l144_14464


namespace median_in_60_64_interval_l144_14451

/-- Represents the score intervals in the histogram --/
inductive ScoreInterval
| I50_54
| I55_59
| I60_64
| I65_69
| I70_74

/-- The frequency of scores in each interval --/
def frequency : ScoreInterval → Nat
| ScoreInterval.I50_54 => 3
| ScoreInterval.I55_59 => 5
| ScoreInterval.I60_64 => 10
| ScoreInterval.I65_69 => 15
| ScoreInterval.I70_74 => 20

/-- The total number of students --/
def totalStudents : Nat := 100

/-- The position of the median in the ordered list of scores --/
def medianPosition : Nat := (totalStudents + 1) / 2

/-- Theorem stating that the median score is in the interval 60-64 --/
theorem median_in_60_64_interval :
  ∃ k : Nat, k ≤ medianPosition ∧
  (frequency ScoreInterval.I70_74 + frequency ScoreInterval.I65_69 + frequency ScoreInterval.I60_64) ≥ k ∧
  (frequency ScoreInterval.I70_74 + frequency ScoreInterval.I65_69) < k :=
by sorry

end median_in_60_64_interval_l144_14451


namespace initial_crayons_count_l144_14474

/-- 
Given:
- initial_crayons is the number of crayons initially in the drawer
- added_crayons is the number of crayons Benny added (3)
- total_crayons is the total number of crayons after adding (12)

Prove that the initial number of crayons is 9.
-/
theorem initial_crayons_count (initial_crayons added_crayons total_crayons : ℕ) 
  (h1 : added_crayons = 3)
  (h2 : total_crayons = 12)
  (h3 : initial_crayons + added_crayons = total_crayons) : 
  initial_crayons = 9 := by
sorry

end initial_crayons_count_l144_14474


namespace log_base_10_derivative_l144_14408

theorem log_base_10_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 10) x = 1 / (x * Real.log 10) := by
sorry

end log_base_10_derivative_l144_14408


namespace charity_event_selection_methods_l144_14423

def total_students : ℕ := 10
def selected_students : ℕ := 4
def special_students : ℕ := 2  -- A and B

-- Function to calculate the number of ways to select k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem charity_event_selection_methods :
  (choose (total_students - special_students) (selected_students - special_students) +
   choose (total_students - special_students) (selected_students - 1) * special_students) = 140 :=
by sorry

end charity_event_selection_methods_l144_14423


namespace snow_fall_time_l144_14422

/-- Given that snow falls at a rate of 1 mm every 6 minutes, prove that it takes 100 hours for 1 m of snow to fall. -/
theorem snow_fall_time (rate : ℝ) (h1 : rate = 1 / 6) : (1000 / rate) / 60 = 100 := by
  sorry

end snow_fall_time_l144_14422


namespace decimal_multiplication_l144_14420

theorem decimal_multiplication (a b : ℚ) (n m : ℕ) :
  a = 0.125 →
  b = 3.84 →
  (a * 10^3).num * (b * 10^2).num = 48000 →
  a * b = 0.48 := by
sorry

end decimal_multiplication_l144_14420


namespace scarletts_oil_measurement_l144_14449

theorem scarletts_oil_measurement (initial_oil : ℝ) : 
  (initial_oil + 0.67 = 0.84) → initial_oil = 0.17 := by
  sorry

end scarletts_oil_measurement_l144_14449


namespace game_result_l144_14454

/-- Represents the state of the game, with each player's money in pence -/
structure GameState where
  adams : ℚ
  baker : ℚ
  carter : ℚ
  dobson : ℚ
  edwards : ℚ
  francis : ℚ
  gudgeon : ℚ

/-- Doubles the money of all players except the winner -/
def double_others (state : GameState) (winner : Fin 7) : GameState :=
  match winner with
  | 0 => ⟨state.adams, 2*state.baker, 2*state.carter, 2*state.dobson, 2*state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 1 => ⟨2*state.adams, state.baker, 2*state.carter, 2*state.dobson, 2*state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 2 => ⟨2*state.adams, 2*state.baker, state.carter, 2*state.dobson, 2*state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 3 => ⟨2*state.adams, 2*state.baker, 2*state.carter, state.dobson, 2*state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 4 => ⟨2*state.adams, 2*state.baker, 2*state.carter, 2*state.dobson, state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 5 => ⟨2*state.adams, 2*state.baker, 2*state.carter, 2*state.dobson, 2*state.edwards, state.francis, 2*state.gudgeon⟩
  | 6 => ⟨2*state.adams, 2*state.baker, 2*state.carter, 2*state.dobson, 2*state.edwards, 2*state.francis, state.gudgeon⟩

/-- Plays the game for all seven rounds -/
def play_game (initial_state : GameState) : GameState :=
  (List.range 7).foldl (fun state i => double_others state i) initial_state

/-- The main theorem to prove -/
theorem game_result (initial_state : GameState) 
  (h1 : initial_state.adams = 1/2)
  (h2 : initial_state.baker = 1/4)
  (h3 : initial_state.carter = 1/4)
  (h4 : initial_state.dobson = 1/4)
  (h5 : initial_state.edwards = 1/4)
  (h6 : initial_state.francis = 1/4)
  (h7 : initial_state.gudgeon = 1/4) :
  let final_state := play_game initial_state
  final_state.adams = 32 ∧
  final_state.baker = 32 ∧
  final_state.carter = 32 ∧
  final_state.dobson = 32 ∧
  final_state.edwards = 32 ∧
  final_state.francis = 32 ∧
  final_state.gudgeon = 32 := by
  sorry

end game_result_l144_14454


namespace lemonade_consumption_l144_14477

/-- Represents the lemonade consumption problem -/
theorem lemonade_consumption (x : ℝ) 
  (h1 : x > 0)  -- Ed's initial lemonade amount is positive
  (h2 : x / 2 + x / 4 + 3 = 2 * x - (x / 4 + 3)) -- Equation representing equal consumption
  : x + 2 * x = 18 := by
  sorry

#check lemonade_consumption

end lemonade_consumption_l144_14477


namespace equation_root_constraint_l144_14403

theorem equation_root_constraint (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = a * x + 1) ∧ 
  (∀ x : ℝ, x > 0 → |x| ≠ a * x + 1) → 
  -1 < a ∧ a < 1 := by
sorry

end equation_root_constraint_l144_14403


namespace square_diagonal_ratio_l144_14470

theorem square_diagonal_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) 
  (area_ratio : a^2 / b^2 = 49 / 64) : 
  (a * Real.sqrt 2) / (b * Real.sqrt 2) = 7 / 8 := by
  sorry

end square_diagonal_ratio_l144_14470


namespace tan_sum_equals_one_l144_14493

-- Define the line equation
def line_equation (x y : ℝ) (α β : ℝ) : Prop :=
  x * Real.tan α - y - 3 * Real.tan β = 0

-- Define the theorem
theorem tan_sum_equals_one (α β : ℝ) :
  (∃ (x y : ℝ), line_equation x y α β) → -- Line equation exists
  (Real.tan α = 2) →                     -- Slope is 2
  (3 * Real.tan β = -1) →                -- Y-intercept is 1
  Real.tan (α + β) = 1 :=
by sorry

end tan_sum_equals_one_l144_14493


namespace f_at_four_is_zero_l144_14405

/-- A function f satisfying the given property for all real x -/
def f : ℝ → ℝ := sorry

/-- The main property of the function f -/
axiom f_property : ∀ x : ℝ, x * f x = 2 * f (2 - x) + 1

/-- The theorem to be proved -/
theorem f_at_four_is_zero : f 4 = 0 := by sorry

end f_at_four_is_zero_l144_14405


namespace remaining_requests_after_two_weeks_l144_14485

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the number of weekdays in a week -/
def weekdaysInWeek : ℕ := 5

/-- Represents the number of weekend days in a week -/
def weekendDaysInWeek : ℕ := daysInWeek - weekdaysInWeek

/-- Represents the number of requests Maia gets on a weekday -/
def weekdayRequests : ℕ := 8

/-- Represents the number of requests Maia gets on a weekend day -/
def weekendRequests : ℕ := 5

/-- Represents the number of requests Maia works on each day (except Sunday) -/
def requestsWorkedPerDay : ℕ := 4

/-- Represents the number of weeks we're considering -/
def numberOfWeeks : ℕ := 2

/-- Represents the number of days Maia works in a week -/
def workDaysPerWeek : ℕ := daysInWeek - 1

theorem remaining_requests_after_two_weeks : 
  (weekdayRequests * weekdaysInWeek + weekendRequests * weekendDaysInWeek) * numberOfWeeks - 
  (requestsWorkedPerDay * workDaysPerWeek) * numberOfWeeks = 52 := by
  sorry

end remaining_requests_after_two_weeks_l144_14485


namespace hyperbola_eccentricity_l144_14471

/-- The eccentricity of the hyperbola (x²/4) - (y²/2) = 1 is √6/2 -/
theorem hyperbola_eccentricity : 
  let C : Set (ℝ × ℝ) := {(x, y) | x^2/4 - y^2/2 = 1}
  ∃ e : ℝ, e = Real.sqrt 6 / 2 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ C → 
      e = Real.sqrt ((x^2/4 + y^2/2) / (x^2/4)) :=
by sorry

end hyperbola_eccentricity_l144_14471


namespace proposition_variants_l144_14457

theorem proposition_variants (a b : ℝ) : 
  (∀ a b, a ≤ b → a - 2 ≤ b - 2) ∧ 
  (∀ a b, a - 2 > b - 2 → a > b) ∧ 
  (∀ a b, a - 2 ≤ b - 2 → a ≤ b) ∧ 
  ¬(∀ a b, a > b → a - 2 ≤ b - 2) := by
  sorry

end proposition_variants_l144_14457


namespace not_all_square_roots_irrational_l144_14484

theorem not_all_square_roots_irrational : ¬ (∀ x : ℝ, ∃ y : ℝ, y ^ 2 = x → ¬ (∃ a b : ℤ, x = a / b ∧ b ≠ 0)) := by
  sorry

end not_all_square_roots_irrational_l144_14484


namespace unique_set_A_l144_14468

def A : Finset ℕ := {2, 3, 4, 5}

def B : Finset ℕ := {24, 30, 40, 60}

def three_products (S : Finset ℕ) : Finset ℕ :=
  S.powerset.filter (λ s => s.card = 3) |>.image (λ s => s.prod id)

theorem unique_set_A : 
  ∀ S : Finset ℕ, S.card = 4 → three_products S = B → S = A := by
  sorry

end unique_set_A_l144_14468


namespace order_of_numbers_l144_14494

theorem order_of_numbers : 
  let a := 2 / Real.exp 2
  let b := Real.log (Real.sqrt 2)
  let c := Real.log 3 / 3
  a < b ∧ b < c := by sorry

end order_of_numbers_l144_14494


namespace diploma_monthly_pay_l144_14466

/-- The annual salary of a person with a degree -/
def annual_salary_degree : ℕ := 144000

/-- The ratio of salary between a person with a degree and a diploma holder -/
def salary_ratio : ℕ := 3

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- The monthly pay for a person holding a diploma certificate -/
def monthly_pay_diploma : ℚ := annual_salary_degree / (salary_ratio * months_per_year)

theorem diploma_monthly_pay :
  monthly_pay_diploma = 4000 := by sorry

end diploma_monthly_pay_l144_14466


namespace unique_function_satisfying_condition_l144_14445

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, is_perfect_square (f a * f (a + b) - a * b)

theorem unique_function_satisfying_condition :
  ∃! f : ℕ → ℕ, satisfies_condition f ∧ ∀ x : ℕ, f x = x :=
sorry

end unique_function_satisfying_condition_l144_14445


namespace parabola_vertex_l144_14481

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = 2 * (x - 3)^2 - 7

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (3, -7)

/-- Theorem: The vertex of the parabola y = 2(x-3)^2 - 7 is (3, -7) -/
theorem parabola_vertex : 
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end parabola_vertex_l144_14481


namespace simplify_square_roots_l144_14435

theorem simplify_square_roots : Real.sqrt 81 - Real.sqrt 144 = -3 := by
  sorry

end simplify_square_roots_l144_14435


namespace committee_count_theorem_l144_14416

/-- The number of ways to choose a committee with at least one female member -/
def committee_count (total_members : ℕ) (committee_size : ℕ) (female_members : ℕ) : ℕ :=
  Nat.choose total_members committee_size - Nat.choose (total_members - female_members) committee_size

theorem committee_count_theorem :
  committee_count 30 5 12 = 133938 := by
  sorry

end committee_count_theorem_l144_14416


namespace gcf_75_90_l144_14425

theorem gcf_75_90 : Nat.gcd 75 90 = 15 := by
  sorry

end gcf_75_90_l144_14425


namespace rectangle_area_l144_14497

theorem rectangle_area (L W : ℝ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) :
  L * W = 140 := by
  sorry

end rectangle_area_l144_14497


namespace inscribed_sphere_volume_l144_14469

/-- A right circular cone with a sphere inscribed inside it. -/
structure ConeWithSphere where
  /-- The diameter of the cone's base in inches. -/
  base_diameter : ℝ
  /-- The vertex angle of the cross-section triangle in degrees. -/
  vertex_angle : ℝ
  /-- The sphere is tangent to the sides of the cone and sits on the table. -/
  sphere_tangent : Bool

/-- The volume of the inscribed sphere in cubic inches. -/
def sphere_volume (cone : ConeWithSphere) : ℝ := sorry

/-- Theorem stating the volume of the inscribed sphere for the given conditions. -/
theorem inscribed_sphere_volume (cone : ConeWithSphere) 
  (h1 : cone.base_diameter = 24)
  (h2 : cone.vertex_angle = 90)
  (h3 : cone.sphere_tangent = true) :
  sphere_volume cone = 576 * Real.sqrt 2 * Real.pi := by sorry

end inscribed_sphere_volume_l144_14469


namespace fraction_simplification_l144_14437

theorem fraction_simplification (m : ℝ) (h : m ≠ 3 ∧ m ≠ -3) :
  (m^2 - 3*m) / (9 - m^2) = -m / (m + 3) := by
  sorry

end fraction_simplification_l144_14437


namespace complex_sum_of_powers_of_i_l144_14438

theorem complex_sum_of_powers_of_i : Complex.I + Complex.I^2 + Complex.I^3 + Complex.I^4 = 0 := by
  sorry

end complex_sum_of_powers_of_i_l144_14438


namespace quadratic_expression_value_l144_14446

theorem quadratic_expression_value (a b : ℝ) : 
  (2 : ℝ)^2 + a * 2 - 6 = 0 ∧ 
  b^2 + a * b - 6 = 0 → 
  (2 * a + b)^2023 = -1 := by
  sorry

end quadratic_expression_value_l144_14446


namespace candidate_vote_percentage_l144_14467

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (invalid_percentage : ℚ) 
  (candidate_valid_votes : ℕ) 
  (h1 : total_votes = 560000) 
  (h2 : invalid_percentage = 15/100) 
  (h3 : candidate_valid_votes = 404600) : 
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 85/100 := by
sorry

end candidate_vote_percentage_l144_14467


namespace intersection_area_l144_14462

/-- The area of a region formed by the intersection of four circles -/
theorem intersection_area (r : ℝ) (h : r = 5) : 
  ∃ (A : ℝ), A = 50 * (π - 2) ∧ 
  A = 8 * (((π * r^2) / 4) - ((r^2) / 2)) := by
  sorry

end intersection_area_l144_14462


namespace sum_of_squares_first_50_even_integers_l144_14427

theorem sum_of_squares_first_50_even_integers :
  (Finset.range 50).sum (fun i => (2 * (i + 1))^2) = 171700 := by
  sorry

end sum_of_squares_first_50_even_integers_l144_14427


namespace smallest_difference_in_triangle_l144_14414

theorem smallest_difference_in_triangle (PQ PR QR : ℕ) : 
  PQ + PR + QR = 2021 →  -- Perimeter condition
  PQ < PR →              -- PQ < PR condition
  PR = (3 * PQ) / 2 →    -- PR = 1.5 × PQ condition
  PQ > 0 ∧ PR > 0 ∧ QR > 0 →  -- Positive side lengths
  PQ + QR > PR ∧ PR + QR > PQ ∧ PQ + PR > QR →  -- Triangle inequality
  PR - PQ ≥ 204 :=
by sorry

end smallest_difference_in_triangle_l144_14414


namespace opposite_numbers_sum_l144_14429

theorem opposite_numbers_sum (a b : ℤ) : (a + b = 0) → (2006 * a + 2006 * b = 0) := by
  sorry

end opposite_numbers_sum_l144_14429


namespace a_b_product_l144_14417

def a : ℕ → ℚ
  | 0 => 1/2
  | n + 1 => 2 * a n / (1 + (a n)^2)

def b : ℕ → ℚ
  | 0 => 4
  | n + 1 => b n ^ 2 - 2 * b n + 2

def b_product : ℕ → ℚ
  | 0 => b 0
  | n + 1 => b_product n * b (n + 1)

theorem a_b_product (n : ℕ) : a (n + 1) * b (n + 1) = 2 * b_product n := by
  sorry

end a_b_product_l144_14417


namespace increasing_order_x_xx_xxx_l144_14498

theorem increasing_order_x_xx_xxx (x : ℝ) (h1 : 1 < x) (h2 : x < 1.1) :
  x < x^x ∧ x^x < x^(x^x) := by sorry

end increasing_order_x_xx_xxx_l144_14498


namespace two_colorable_l144_14486

-- Define a graph with 2000 vertices
def Graph := Fin 2000 → Set (Fin 2000)

-- Define a property that each vertex has at least one edge
def HasEdges (g : Graph) : Prop :=
  ∀ v : Fin 2000, ∃ u : Fin 2000, u ∈ g v

-- Define a coloring function
def Coloring := Fin 2000 → Bool

-- Define a valid coloring
def ValidColoring (g : Graph) (c : Coloring) : Prop :=
  ∀ v u : Fin 2000, u ∈ g v → c v ≠ c u

-- Theorem statement
theorem two_colorable (g : Graph) (h : HasEdges g) :
  ∃ c : Coloring, ValidColoring g c :=
sorry

end two_colorable_l144_14486


namespace percent_problem_l144_14409

theorem percent_problem (x : ℝ) : 2 = (4 / 100) * x → x = 50 := by
  sorry

end percent_problem_l144_14409


namespace closest_to_580_l144_14460

def problem_value : ℝ := 0.000218 * 5432000 - 500

def options : List ℝ := [520, 580, 600, 650]

theorem closest_to_580 : 
  ∀ x ∈ options, |problem_value - 580| ≤ |problem_value - x| := by
  sorry

end closest_to_580_l144_14460


namespace sum_of_coefficients_l144_14413

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
sorry

end sum_of_coefficients_l144_14413


namespace distance_between_vehicles_distance_is_300_l144_14432

/-- The distance between two vehicles l and k, given specific conditions on their speeds and travel times. -/
theorem distance_between_vehicles (speed_l : ℝ) (start_time_l start_time_k meet_time : ℕ) : ℝ :=
  let speed_k := speed_l * 1.5
  let travel_time_l := meet_time - start_time_l
  let travel_time_k := meet_time - start_time_k
  let distance_l := speed_l * travel_time_l
  let distance_k := speed_k * travel_time_k
  distance_l + distance_k

/-- The distance between vehicles l and k is 300 km under the given conditions. -/
theorem distance_is_300 : distance_between_vehicles 50 9 10 12 = 300 := by
  sorry

end distance_between_vehicles_distance_is_300_l144_14432


namespace tangent_point_circle_properties_l144_14418

/-- The equation of a circle with center (x₀, y₀) and radius r -/
def circle_equation (x₀ y₀ r : ℝ) (x y : ℝ) : Prop :=
  (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The equation of the line 2x + y = 0 -/
def line_center (x y : ℝ) : Prop :=
  2 * x + y = 0

/-- The equation of the line x + y - 1 = 0 -/
def line_tangent (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- The point (2, -1) lies on the tangent line -/
theorem tangent_point : line_tangent 2 (-1) := by sorry

theorem circle_properties (x₀ y₀ r : ℝ) :
  line_center x₀ y₀ →
  (∀ x y, circle_equation x₀ y₀ r x y ↔ (x - 1)^2 + (y + 2)^2 = 2) →
  (∃ x y, circle_equation x₀ y₀ r x y ∧ line_tangent x y) →
  circle_equation x₀ y₀ r 2 (-1) :=
by sorry

end tangent_point_circle_properties_l144_14418


namespace no_five_linked_country_with_46_airlines_l144_14444

theorem no_five_linked_country_with_46_airlines :
  ¬ ∃ (n : ℕ), n > 0 ∧ (5 * n) / 2 = 46 := by
  sorry

end no_five_linked_country_with_46_airlines_l144_14444


namespace max_value_constraint_l144_14495

theorem max_value_constraint (x y : ℝ) : 
  x^2 + y^2 = 20*x + 9*y + 9 → (4*x + 3*y ≤ 83) ∧ ∃ x y, x^2 + y^2 = 20*x + 9*y + 9 ∧ 4*x + 3*y = 83 := by
  sorry

end max_value_constraint_l144_14495


namespace fourth_ball_black_prob_l144_14434

/-- A box containing red and black balls -/
structure Box where
  red_balls : ℕ
  black_balls : ℕ

/-- The probability of selecting a black ball from a box -/
def prob_black_ball (box : Box) : ℚ :=
  box.black_balls / (box.red_balls + box.black_balls)

/-- The theorem stating the probability of the fourth ball being black -/
theorem fourth_ball_black_prob (box : Box) 
  (h1 : box.red_balls = 3) 
  (h2 : box.black_balls = 3) : 
  prob_black_ball box = 1/2 := by
  sorry

#eval prob_black_ball { red_balls := 3, black_balls := 3 }

end fourth_ball_black_prob_l144_14434


namespace pizza_slices_per_pizza_pizza_problem_l144_14421

theorem pizza_slices_per_pizza (num_pizzas : ℕ) (total_cost : ℚ) (slices_sample : ℕ) (cost_sample : ℚ) : ℚ :=
  let cost_per_slice : ℚ := cost_sample / slices_sample
  let cost_per_pizza : ℚ := total_cost / num_pizzas
  cost_per_pizza / cost_per_slice

theorem pizza_problem : pizza_slices_per_pizza 3 72 5 10 = 12 := by
  sorry

end pizza_slices_per_pizza_pizza_problem_l144_14421


namespace yvonne_word_count_l144_14436

/-- Proves that Yvonne wrote 400 words given the conditions of the research paper problem -/
theorem yvonne_word_count 
  (total_required : Nat) 
  (janna_extra : Nat) 
  (words_removed : Nat) 
  (words_to_add : Nat) 
  (h1 : total_required = 1000)
  (h2 : janna_extra = 150)
  (h3 : words_removed = 20)
  (h4 : words_to_add = 30) : 
  ∃ (yvonne_words : Nat), 
    yvonne_words + (yvonne_words + janna_extra) - words_removed + 2 * words_removed + words_to_add = total_required ∧ 
    yvonne_words = 400 := by
  sorry

#check yvonne_word_count

end yvonne_word_count_l144_14436


namespace remainder_theorem_l144_14407

-- Define the polynomial Q(x)
variable (Q : ℝ → ℝ)

-- Define the conditions
axiom Q_remainder_15 : ∃ P : ℝ → ℝ, ∀ x, Q x = P x * (x - 15) + 10
axiom Q_remainder_12 : ∃ P : ℝ → ℝ, ∀ x, Q x = P x * (x - 12) + 2

-- Theorem statement
theorem remainder_theorem :
  ∃ R : ℝ → ℝ, ∀ x, Q x = R x * ((x - 12) * (x - 15)) + (8/3 * x - 30) :=
sorry

end remainder_theorem_l144_14407


namespace liam_cycling_speed_l144_14459

/-- Given the cycling speeds of Eugene, Claire, and Liam, prove that Liam's speed is 6 miles per hour. -/
theorem liam_cycling_speed 
  (eugene_speed : ℝ) 
  (claire_speed_ratio : ℝ) 
  (liam_speed_ratio : ℝ) 
  (h1 : eugene_speed = 6)
  (h2 : claire_speed_ratio = 3/4)
  (h3 : liam_speed_ratio = 4/3) :
  liam_speed_ratio * (claire_speed_ratio * eugene_speed) = 6 :=
by sorry

end liam_cycling_speed_l144_14459


namespace square_rectangle_area_relation_l144_14490

theorem square_rectangle_area_relation : 
  ∃ (x₁ x₂ : ℝ), 
    (∀ x : ℝ, 3 * (x - 4)^2 = (x - 5) * (x + 6) ↔ x = x₁ ∨ x = x₂) ∧ 
    x₁ + x₂ = 12.5 := by
  sorry

end square_rectangle_area_relation_l144_14490


namespace angle_sum_around_point_l144_14455

theorem angle_sum_around_point (x : ℝ) : 
  (6*x + 3*x + x + x + 4*x = 360) → x = 24 := by
  sorry

end angle_sum_around_point_l144_14455


namespace king_descendants_comparison_l144_14456

theorem king_descendants_comparison :
  let pafnutius_sons := 2
  let pafnutius_two_sons := 60
  let pafnutius_one_son := 20
  let zenobius_daughters := 4
  let zenobius_three_daughters := 35
  let zenobius_one_daughter := 35

  let pafnutius_descendants := pafnutius_sons + pafnutius_two_sons * 2 + pafnutius_one_son * 1
  let zenobius_descendants := zenobius_daughters + zenobius_three_daughters * 3 + zenobius_one_daughter * 1

  zenobius_descendants > pafnutius_descendants := by sorry

end king_descendants_comparison_l144_14456


namespace volunteer_hours_per_time_l144_14453

/-- The number of times John volunteers per month -/
def volunteering_frequency : ℕ := 2

/-- The total number of hours John volunteers per year -/
def total_hours_per_year : ℕ := 72

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- Theorem stating how many hours John volunteers at a time -/
theorem volunteer_hours_per_time :
  total_hours_per_year / (volunteering_frequency * months_per_year) = 3 := by
  sorry

end volunteer_hours_per_time_l144_14453


namespace arrangements_with_fixed_order_l144_14410

/-- The number of programs --/
def total_programs : ℕ := 5

/-- The number of programs that must appear in a specific order --/
def fixed_order_programs : ℕ := 3

/-- The number of different arrangements when 3 specific programs must appear in a given order --/
def num_arrangements : ℕ := 20

/-- Theorem stating that given 5 programs with 3 in a fixed order, there are 20 different arrangements --/
theorem arrangements_with_fixed_order :
  total_programs = 5 →
  fixed_order_programs = 3 →
  num_arrangements = 20 :=
by sorry

end arrangements_with_fixed_order_l144_14410
