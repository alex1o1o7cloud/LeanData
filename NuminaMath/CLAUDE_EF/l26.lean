import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_divisor_infinitely_many_factorial_divisors_infinitely_many_prime_divisors_l26_2626

-- Part 1
theorem unique_prime_divisor (p : Nat) : 
  Nat.Prime p ∧ p ∣ (3^(p+1) - 2^(p+1)) ↔ p = 5 := by sorry

-- Part 2
theorem infinitely_many_factorial_divisors : 
  ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ n, Nat.factorial (f n) ∣ (3^((f n)-1) - 2^((f n)-1)) := by sorry

-- Part 3
theorem infinitely_many_prime_divisors : 
  ∃ f : ℕ → ℕ, StrictMono f ∧ (∀ n, Nat.Prime (f n)) ∧ 
  (∀ n, (f n) ∣ (3^(((f n)-1)/2) - 2^(((f n)-1)/2))) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_divisor_infinitely_many_factorial_divisors_infinitely_many_prime_divisors_l26_2626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_ride_time_l26_2698

/-- Represents the time it takes Clea to ride up the operating escalator when standing still -/
noncomputable def ride_time (walk_time_off : ℝ) (walk_time_on : ℝ) : ℝ :=
  walk_time_off * walk_time_on / (walk_time_off - walk_time_on)

theorem escalator_ride_time :
  ride_time 120 48 = 80 := by
  -- Unfold the definition of ride_time
  unfold ride_time
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_ride_time_l26_2698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_permutations_eq_60_l26_2600

/-- The number of distinct permutations of the letters in BANANA -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- The word BANANA has 6 letters -/
axiom banana_length : 6 = 6

/-- The word BANANA has 1 B, 2 N's, and 3 A's -/
axiom banana_composition : 
  1 = 1 ∧ 2 = 2 ∧ 3 = 3

/-- The number of distinct permutations of BANANA is 60 -/
theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_permutations_eq_60_l26_2600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_def_T_formula_l26_2651

/-- Sequence a_n defined by S_n = 1 - a_n -/
def a : ℕ+ → ℚ := sorry

/-- Sequence b_n defined as n / a_n -/
def b : ℕ+ → ℚ := sorry

/-- Sum of first n terms of sequence b_n -/
def T : ℕ+ → ℚ := sorry

/-- The sum of the first n terms of sequence a_n -/
def S (n : ℕ+) : ℚ := 1 - a n

theorem a_formula (n : ℕ+) : a n = (1 / 2) ^ (n : ℕ) := by sorry

theorem b_def (n : ℕ+) : b n = n / (a n) := by sorry

theorem T_formula (n : ℕ+) : T n = (n - 1) * 2^(n.val + 1) + 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_b_def_T_formula_l26_2651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_inscribed_angles_l26_2648

/-- Represents an inscribed angle in a circle -/
structure InscribedAngle where
  arcs : ℕ
  total_arcs : ℕ

/-- The measure of an inscribed angle in degrees -/
noncomputable def angle_measure (a : InscribedAngle) : ℝ :=
  (a.arcs : ℝ) * 360 / (2 * a.total_arcs)

theorem sum_of_inscribed_angles (x y : InscribedAngle) 
  (hx : x.arcs = 3 ∧ x.total_arcs = 16)
  (hy : y.arcs = 5 ∧ y.total_arcs = 16) :
  angle_measure x + angle_measure y = 90 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_inscribed_angles_l26_2648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l26_2667

theorem calculation_proof :
  (Real.rpow (-64 : ℝ) (1/3) + Real.sqrt 16 * Real.sqrt (9/4) + (-Real.sqrt 2)^2 = 4) ∧
  (Real.rpow 27 (1/3) - Real.sqrt 0 + Real.rpow (1/8 : ℝ) (1/3) = 3.5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proof_l26_2667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_l26_2689

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The grid size -/
def gridWidth : ℕ := 8
def gridHeight : ℕ := 5

/-- The given points -/
def A : Point := ⟨0, 4⟩
def B : Point := ⟨7, 0⟩
def C : Point := ⟨5, 3⟩
def D : Point := ⟨3, 0⟩

/-- Point E is where AB meets CD -/
noncomputable def E : Point := sorry

/-- The theorem to prove -/
theorem length_of_AE : distance A E = (5 * Real.sqrt 65) / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AE_l26_2689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_denote_loss_given_profit_notation_l26_2697

/-- Represents the financial value in yuan -/
def FinancialValue : Type := ℤ

/-- Converts a financial result (profit or loss) to a FinancialValue -/
def denoteFinancialResult (amount : ℕ) (isProfit : Bool) : FinancialValue :=
  if isProfit then (amount : ℤ) else -(amount : ℤ)

/-- The theorem stating how to denote a loss given how a profit is denoted -/
theorem denote_loss_given_profit_notation 
  (profit_amount : ℕ) (loss_amount : ℕ) 
  (h : denoteFinancialResult profit_amount true = (profit_amount : ℤ)) :
  denoteFinancialResult loss_amount false = -(loss_amount : ℤ) :=
by
  sorry

#check denote_loss_given_profit_notation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_denote_loss_given_profit_notation_l26_2697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colored_squares_l26_2650

/-- Represents a coloring of a square grid -/
def Coloring (n : ℕ) := Fin n → Fin n → Option (Fin 5)

/-- A valid coloring has the same number of occurrences for each color and no differently colored squares in the same row or column -/
def ValidColoring (c : Coloring 99) : Prop :=
  (∀ color : Fin 5, (count c color) = (count c 0)) ∧
  (∀ i j k : Fin 99, c i j ≠ none → c i k ≠ none → c i j = c i k) ∧
  (∀ i j k : Fin 99, c i j ≠ none → c k j ≠ none → c i j = c k j)
where
  count (c : Coloring 99) (color : Fin 5) : ℕ :=
    (Finset.sum (Finset.univ : Finset (Fin 99)) fun i =>
      Finset.sum (Finset.univ : Finset (Fin 99)) fun j =>
        if c i j = some color then 1 else 0)

/-- The number of colored squares in a coloring -/
def ColoredSquares (c : Coloring 99) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 99)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 99)) fun j =>
      if c i j ≠ none then 1 else 0

/-- The maximum number of colored squares in a valid coloring of a 99x99 grid with 5 colors is 1900 -/
theorem max_colored_squares :
  (∃ c : Coloring 99, ValidColoring c ∧ ColoredSquares c = 1900) ∧
  (∀ c : Coloring 99, ValidColoring c → ColoredSquares c ≤ 1900) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_colored_squares_l26_2650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l26_2690

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define point P
def P : ℝ × ℝ := (-2, 1)

-- Define a point Q on the parabola
def Q (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

-- Distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_properties :
  (∀ t : ℝ, distance (Q t) focus = |(Q t).1 + 1|) ∧
  (3 * ((P.1 + focus.1) / 2) - ((P.2 + focus.2) / 2) + 2 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l26_2690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l26_2684

theorem integer_solutions_equation : 
  {(x, y) : ℤ × ℤ | x * (x + 1) * (x + 7) * (x + 8) = y^2} = 
  {(1, 12), (1, -12), (-9, 12), (-9, -12), (0, 0), (-8, 0), (-4, -12), (-4, 12), (-1, 0), (-7, 0)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l26_2684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2006_equals_4_l26_2601

def b : ℕ → ℚ
  | 0 => 5  -- Added case for 0
  | 1 => 5
  | 2 => 4
  | n+3 => b (n+2) / b (n+1)

theorem b_2006_equals_4 : b 2006 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2006_equals_4_l26_2601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_b_value_l26_2658

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3*x - b else (2 : ℝ)^x

-- State the theorem
theorem function_b_value (b : ℝ) : f b (f b (5/6)) = 4 → b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_b_value_l26_2658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_in_cyclic_process_l26_2616

/-- Represents the work done by a monatomic ideal gas in a cyclic process -/
noncomputable def work_done_cyclic_process (T₀ : ℝ) (R : ℝ) : ℝ :=
  1/2 * R * T₀

/-- The gas constant (J/(mol·K)) -/
def R : ℝ := 8.314

theorem work_in_cyclic_process :
  let T₀ : ℝ := 320
  work_done_cyclic_process T₀ R = 665 := by
  sorry

-- Remove the #eval statement as it's causing issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_in_cyclic_process_l26_2616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_negative_three_simplify_trig_expression_l26_2638

-- Part 1
theorem tan_alpha_negative_three (α : ℝ) (h : Real.tan α = -3) :
  (Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α) = -1/8 := by sorry

-- Part 2
theorem simplify_trig_expression (α β : ℝ) :
  (2 * Real.cos α * Real.sin β + Real.sin (α - β)) / 
  (Real.cos (α - β) - 2 * Real.sin α * Real.sin β) = Real.tan (α + β) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_negative_three_simplify_trig_expression_l26_2638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_wins_l26_2662

/-- Represents a player in the game -/
inductive Player : Type where
  | White : Player
  | Black : Player
deriving DecidableEq

/-- Represents a cell on the game board -/
structure Cell where
  x : Fin 25
  y : Fin 25
deriving DecidableEq

/-- Represents the game state -/
structure GameState where
  board : Cell → Option Player
  currentPlayer : Player

/-- Checks if two cells are neighbors -/
def isNeighbor (c1 c2 : Cell) : Prop :=
  (c1.x = c2.x ∧ (c1.y = c2.y - 1 ∨ c1.y = c2.y + 1)) ∨
  (c1.y = c2.y ∧ (c1.x = c2.x - 1 ∨ c1.x = c2.x + 1))

/-- Checks if a cell is valid for placement by the current player -/
def isValidMove (state : GameState) (cell : Cell) : Prop :=
  state.board cell = none ∧
  ¬(∀ (neighbor : Cell), isNeighbor cell neighbor →
    state.board neighbor = some state.currentPlayer)

/-- Represents a winning strategy for the first player -/
def hasWinningStrategy (player : Player) : Prop :=
  ∀ (state : GameState),
    state.currentPlayer = player →
    ∃ (move : Cell), isValidMove state move ∧
      ¬∃ (opponentMove : Cell), 
        isValidMove (GameState.mk (Function.update state.board move (some player)) 
          (if player = Player.White then Player.Black else Player.White)) opponentMove

/-- The main theorem stating that White has a winning strategy -/
theorem white_wins : hasWinningStrategy Player.White := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_wins_l26_2662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_weight_l26_2623

/-- Given Bridget's weight and the difference between Bridget's and Martha's weights, 
    prove that Martha's weight is 2 pounds. -/
theorem martha_weight (bridget_weight : ℕ) (weight_difference : ℕ) (martha_weight : ℕ)
  (h1 : bridget_weight = 39)
  (h2 : weight_difference = 37)
  (h3 : bridget_weight = weight_difference + martha_weight) : 
  martha_weight = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_weight_l26_2623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_exponential_function_l26_2674

-- Define the function f(x) = 3^x
noncomputable def f (x : ℝ) : ℝ := 3^x

-- State the theorem
theorem range_of_exponential_function :
  {y : ℝ | ∃ x : ℝ, x ≥ 0 ∧ f x = y} = Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_exponential_function_l26_2674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_deriv_at_2_l26_2676

noncomputable section

variable (f : ℝ → ℝ)

-- Define the function f based on the given condition
axiom f_def : ∀ x : ℝ, f x = x^2 * (deriv f 2) + 3*x

-- State the theorem to be proved
theorem f_deriv_at_2 : deriv f 2 = -1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_deriv_at_2_l26_2676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_theorem_l26_2624

theorem stratified_sampling_theorem (total_male : ℕ) (total_female : ℕ) (sample_size : ℕ) :
  total_male = 200 →
  total_female = 300 →
  sample_size = 50 →
  let total_students := total_male + total_female
  let prob_individual := (sample_size : ℚ) / total_students
  let male_sample := (total_male : ℚ) * prob_individual
  let female_sample := (total_female : ℚ) * prob_individual
  male_sample = 20 ∧ female_sample = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_theorem_l26_2624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carousel_seats_l26_2602

theorem carousel_seats (n : ℕ) :
  (∃ (moves : List ℕ),
    moves.length = n - 1 ∧
    moves.Nodup ∧
    (∀ m ∈ moves, m < n) ∧
    (moves.sum % n = 0)) ↔ n % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carousel_seats_l26_2602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l26_2675

/-- The function f(x) = (4-3sin(x))(4-3cos(x)) -/
noncomputable def f (x : ℝ) : ℝ := (4 - 3 * Real.sin x) * (4 - 3 * Real.cos x)

/-- The minimum value of f(x) is 7/2 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ 7/2 ∧ ∃ x₀ : ℝ, f x₀ = 7/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l26_2675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_length_calculation_l26_2620

/-- Represents the properties of a uniform rod -/
structure UniformRod where
  weightPerMeter : ℝ

/-- Calculates the weight of a given length of a uniform rod -/
noncomputable def weightOfLength (rod : UniformRod) (length : ℝ) : ℝ :=
  rod.weightPerMeter * length

/-- Calculates the length of a uniform rod given its weight -/
noncomputable def lengthOfWeight (rod : UniformRod) (weight : ℝ) : ℝ :=
  weight / rod.weightPerMeter

theorem rod_length_calculation (rod : UniformRod) 
  (h1 : weightOfLength rod 11.25 = 42.75)
  (h2 : weightOfLength rod (lengthOfWeight rod 22.8) = 22.8) : 
  lengthOfWeight rod 22.8 = 6 := by
  sorry

#check rod_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rod_length_calculation_l26_2620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_blue_cells_l26_2608

/-- Represents a coloring of a triangular board. -/
def BoardColoring (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if two cells share a side. -/
def sharesside {n : ℕ} (x1 y1 x2 y2 : Fin n) : Prop :=
  (x1 = x2 ∧ y1 = y2.succ) ∨ (x1 = x2 ∧ y2 = y1.succ) ∨
  (x1 = x2.succ ∧ y1 = y2) ∨ (x2 = x1.succ ∧ y1 = y2)

/-- Checks if a coloring is valid (no two red cells share a side). -/
def validColoring (n : ℕ) (c : BoardColoring n) : Prop :=
  ∀ x1 y1 x2 y2, sharesside x1 y1 x2 y2 → (c x1 y1 = true ∨ c x2 y2 = true)

/-- Counts the number of blue cells in a coloring. -/
def blueCount (n : ℕ) (c : BoardColoring n) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin n)) fun x =>
   Finset.sum (Finset.univ : Finset (Fin n)) fun y =>
   if c x y then 0 else 1)

/-- The minimum number of blue cells needed for a valid coloring. -/
def minBlue (n : ℕ) : ℕ :=
  Nat.floor ((n^2 - 1) / 2) + 1 - (n + 1) / 2

theorem min_blue_cells (n : ℕ) (h : n > 0) :
  ∃ (c : BoardColoring n), validColoring n c ∧ blueCount n c = minBlue n ∧
  ∀ (c' : BoardColoring n), validColoring n c' → blueCount n c' ≥ minBlue n :=
sorry

#check min_blue_cells

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_blue_cells_l26_2608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_zero_f_decreasing_on_positive_reals_l26_2607

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 / x^2 + a * x

-- Theorem 1: If f(x) is even, then a = 0
theorem a_is_zero (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a x = f a (-x)) : a = 0 := by
  sorry

-- Theorem 2: f(x) = 1/x^2 is decreasing on (0, +∞)
theorem f_decreasing_on_positive_reals : 
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → (1 / x^2) > (1 / y^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_is_zero_f_decreasing_on_positive_reals_l26_2607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_magnitude_condition_l26_2641

/-- Given vectors in ℝ² -/
def a : Fin 2 → ℝ := ![1, 1]
def b : Fin 2 → ℝ := ![3, 4]

/-- Dot product of two vectors -/
def dot (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

/-- Magnitude of a vector -/
noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ := Real.sqrt ((v 0) ^ 2 + (v 1) ^ 2)

/-- First theorem: perpendicularity condition -/
theorem perpendicular_condition (k : ℝ) : 
  dot (fun i => k * (a i) + (b i)) (fun i => k * (a i) - (b i)) = 0 ↔ 
  k = (5 * Real.sqrt 2) / 2 ∨ k = -(5 * Real.sqrt 2) / 2 := by sorry

/-- Second theorem: magnitude condition -/
theorem magnitude_condition (k : ℝ) :
  magnitude (fun i => k * (a i) + 2 * (b i)) = 10 ↔ k = 0 ∨ k = -14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_magnitude_condition_l26_2641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_value_l26_2613

-- Define the power function as noncomputable
noncomputable def powerFunction (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

-- State the theorem
theorem power_function_through_point_value :
  ∀ a : ℝ, powerFunction a (Real.sqrt 2) = 2 → powerFunction a 3 = 9 :=
by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_value_l26_2613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_k_for_intersection_l26_2661

/-- The set of points (x, x^2 + k) where x is a real number -/
def S (k : ℝ) : Set (ℝ × ℝ) := {p | ∃ x, p = (x, x^2 + k)}

/-- The set of circles with centers in S and tangent to the X-axis -/
def C (k : ℝ) : Set (Set (ℝ × ℝ)) :=
  {c | ∃ (x : ℝ), c = {p | (p.1 - x)^2 + (p.2 - (x^2 + k))^2 ≤ (x^2 + k)^2}}

/-- Two circles intersect if the distance between their centers is less than or equal to the sum of their radii -/
def circles_intersect (k : ℝ) (c1 c2 : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), c1 ∈ C k ∧ c2 ∈ C k →
    (x - y)^2 + ((x^2 + k) - (y^2 + k))^2 ≤ (x^2 + k + y^2 + k)^2

/-- The theorem to be proved -/
theorem minimum_k_for_intersection :
  (∀ k > 0, ∀ c1 c2, c1 ∈ C k → c2 ∈ C k → circles_intersect k c1 c2) ↔ k ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_k_for_intersection_l26_2661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_square_plus_f_l26_2606

/-- Function that represents an n-digit number with all digits x -/
def f (n : ℕ) (x : Fin 9) : ℕ :=
  x.val * (10^n - 1) / 9

/-- Theorem stating that (f n 3)^2 + f n 2 = f (2*n) 1 for any positive n -/
theorem f_square_plus_f (n : ℕ) (hn : 0 < n) : (f n 3)^2 + f n 2 = f (2*n) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_square_plus_f_l26_2606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_prop1_and_prop4_true_l26_2644

open Real

-- Define the propositions
def proposition1 : Prop := ∀ f : ℝ → ℝ, (∀ x, f x = sin x ^ 4 - cos x ^ 4) → 
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) ∧ 
  (∀ x, f (x + π) = f x)

def proposition2 : Prop := ∀ α : ℝ, (∃ k : ℤ, α = k * π / 2) ↔ 
  (cos α = 0 ∧ sin α = 1) ∨ (cos α = 0 ∧ sin α = -1)

def proposition3 : Prop := ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
  sin x₁ = x₁ ∧ sin x₂ = x₂ ∧ sin x₃ = x₃

def proposition4 : Prop := ∀ x : ℝ, 3 * sin (2 * (x - π / 6) + π / 3) = 3 * sin (2 * x)

def proposition5 : Prop := ∀ θ : ℝ, (0 < θ ∧ θ < π / 2) ↔ sin θ > 0

-- Theorem stating that only propositions 1 and 4 are true
theorem only_prop1_and_prop4_true : 
  proposition1 ∧ proposition4 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_prop1_and_prop4_true_l26_2644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sqrt5_minus3_l26_2652

theorem quadratic_root_sqrt5_minus3 :
  ∃ (a b c : ℚ), a = 1 ∧ (Real.sqrt 5 - 3 : ℝ) ^ 2 + b * (Real.sqrt 5 - 3 : ℝ) + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sqrt5_minus3_l26_2652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_convexity_and_parameter_range_l26_2615

/-- The function f(x) = x^2 + a|x| + x -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * abs x + x

theorem f_convexity_and_parameter_range :
  ∀ a : ℝ,
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 → 
    (f a x₁ + f a x₂) / 2 ≥ f a ((x₁ + x₂) / 2)) ∧
  ((∀ x₁ x₂ : ℝ, (f a x₁ + f a x₂) / 2 ≥ f a ((x₁ + x₂) / 2)) ↔ a ≥ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_convexity_and_parameter_range_l26_2615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l26_2635

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : 0 < a) (h2 : a < Real.sqrt 2) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 2 = 1) →
  (∃ k : ℝ, k > 0 ∧ ∀ x y : ℝ, (y = k*x ∨ y = -k*x) → x^2 / a^2 - y^2 / 2 = 1) →
  (Real.arctan (Real.sqrt 2 / a) = π / 3) →
  let c := Real.sqrt (a^2 + 2)
  c / a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l26_2635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_2500_pi_l26_2699

/-- Given three concentric circles with a chord tangent to the middle and largest circles,
    calculate the area between the middle and largest circles. -/
noncomputable def area_between_circles (r_middle : ℝ) (chord_length : ℝ) : ℝ :=
  let r_largest := Real.sqrt (2 * r_middle^2)
  let area_middle := Real.pi * r_middle^2
  let area_largest := Real.pi * r_largest^2
  area_largest - area_middle

/-- The area between the middle and largest circles is 2500π square units. -/
theorem area_is_2500_pi :
  area_between_circles 50 100 = 2500 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_2500_pi_l26_2699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_sum_count_l26_2680

/-- The number of integers less than 500 that can be written as the sum of k consecutive
    positive even integers for exactly 3 values of k ≥ 1 -/
theorem consecutive_even_sum_count : ∃ (S : Finset ℕ),
  S.card = 9 ∧
  (∀ M ∈ S, M < 500) ∧
  (∀ M ∈ S, ∃ (k₁ k₂ k₃ : ℕ), k₁ < k₂ ∧ k₂ < k₃ ∧
    (∀ k, k ≥ 1 → (∃ m, M = k * (2 * m + k - 1) ∧ m ≥ 1) ↔ k ∈ ({k₁, k₂, k₃} : Finset ℕ))) ∧
  (∀ M, M < 500 →
    (∃ (k₁ k₂ k₃ : ℕ), k₁ < k₂ ∧ k₂ < k₃ ∧
      (∀ k, k ≥ 1 → (∃ m, M = k * (2 * m + k - 1) ∧ m ≥ 1) ↔ k ∈ ({k₁, k₂, k₃} : Finset ℕ))) →
    M ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_sum_count_l26_2680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_iteration_convergence_l26_2666

theorem cosine_iteration_convergence :
  ∃! t : ℝ, t ∈ Set.Ioo (-1) 1 ∧ t = Real.cos t ∧
  ∀ N₀ : ℝ, ∃ (N : ℕ → ℝ), N 0 = N₀ ∧
    (∀ j, N (j + 1) = Real.cos (N j)) ∧
    Filter.Tendsto N Filter.atTop (nhds t) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_iteration_convergence_l26_2666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_proposition_l26_2683

open Real

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ ∀ x, ¬p x := by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x - 2 > log x) ↔ (∃ x : ℝ, x - 2 ≤ log x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_proposition_l26_2683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_replaced_man_age_l26_2645

theorem replaced_man_age (initial_men : ℕ) (age_increase : ℝ) (known_man_age : ℕ) (women_avg_age : ℝ) :
  initial_men = 7 →
  age_increase = 3 →
  known_man_age = 18 →
  women_avg_age = 30.5 →
  ∃ (replaced_man_age : ℕ),
    replaced_man_age = 22 ∧
    (initial_men : ℝ) * ((1 / initial_men : ℝ) * (↑known_man_age + ↑replaced_man_age) + age_increase) =
    ((initial_men - 2) : ℝ) * ((1 / initial_men : ℝ) * ↑known_man_age) + 2 * women_avg_age :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_replaced_man_age_l26_2645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_difference_inverse_main_result_l26_2633

theorem inverse_difference_inverse : (5⁻¹ : ℚ) - (2⁻¹ : ℚ) = -3 / 10 := by
  -- Rewrite fractions
  have h1 : (5⁻¹ : ℚ) = 1 / 5 := by norm_num
  have h2 : (2⁻¹ : ℚ) = 1 / 2 := by norm_num
  
  -- Substitute
  rw [h1, h2]
  
  -- Perform the subtraction
  norm_num

theorem main_result : ((5⁻¹ : ℚ) - (2⁻¹ : ℚ))⁻¹ = -10 / 3 := by
  -- Use the previous theorem
  have h : (5⁻¹ : ℚ) - (2⁻¹ : ℚ) = -3 / 10 := inverse_difference_inverse
  
  -- Substitute and simplify
  rw [h]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_difference_inverse_main_result_l26_2633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l26_2681

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + |x| - 1
  else -x^2 - |x| + 1

theorem odd_function_property (x : ℝ) :
  f x = -f (-x) ∧
  (x > 0 → f x = x^2 + |x| - 1) ∧
  (x < 0 → f x = -x^2 - |x| + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_property_l26_2681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_iff_a_is_zero_or_four_l26_2622

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (1/2) * x - 1 else (1/2) ^ x

-- State the theorem
theorem f_equals_one_iff_a_is_zero_or_four (a : ℝ) :
  f a = 1 ↔ a = 0 ∨ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_one_iff_a_is_zero_or_four_l26_2622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l26_2668

def z (m : ℝ) : ℂ := (m^2 - 1 : ℝ) + (m^2 - m - 2 : ℝ) * Complex.I

theorem complex_number_properties (m : ℝ) :
  (z m = Complex.ofReal (z m).re ↔ m = -1 ∨ m = 2) ∧
  ((z m ≠ Complex.ofReal (z m).re) ∧ (z m).re ≠ 0 ↔ m ≠ -1 ∧ m ≠ 2) ∧
  ((z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l26_2668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marthas_savings_l26_2687

/-- Calculates the total savings for a week given a daily allowance, 
    the fraction saved on regular days, the number of regular days,
    the fraction saved on the exceptional day, and the number of exceptional days. -/
def weeklySavings (dailyAllowance : ℚ) (regularSavingFraction : ℚ) (regularDays : ℕ)
                  (exceptionalSavingFraction : ℚ) (exceptionalDays : ℕ) : ℚ :=
  dailyAllowance * (regularSavingFraction * regularDays + 
                    exceptionalSavingFraction * exceptionalDays)

/-- Proves that given the specific conditions of Martha's savings plan, 
    she saves $39 at the end of the week. -/
theorem marthas_savings : 
  weeklySavings 12 (1/2) 6 (1/4) 1 = 39 := by
  -- Unfold the definition of weeklySavings
  unfold weeklySavings
  -- Simplify the arithmetic
  simp [Rat.mul_add, Rat.mul_assoc]
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marthas_savings_l26_2687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_result_l26_2639

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^2 + 1 else -z^2 + 1

-- State the theorem
theorem f_composition_result : f (f (f (f (1 + Complex.I)))) = 378 + 336 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_result_l26_2639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_quadratic_range_l26_2642

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

-- Define monotonicity on an interval
def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x < f y ∨ ∀ z ∈ Set.Icc a b, f z = f x)

-- State the theorem
theorem monotonic_quadratic_range (m : ℝ) :
  monotonic_on (f m) (-1) 3 → m ∈ Set.Iic (-6) ∪ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_quadratic_range_l26_2642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_is_one_sixth_of_prism_l26_2637

/-- A rectangular prism with dimensions a, b, and c -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The volume of a rectangular prism -/
noncomputable def prism_volume (p : RectangularPrism) : ℝ := p.a * p.b * p.c

/-- A tetrahedron formed by specific vertices of the rectangular prism -/
inductive Tetrahedron
  | AB'CD' : Tetrahedron
  | A'DC'B : Tetrahedron

/-- The intersection volume of two specific tetrahedra within a rectangular prism -/
noncomputable def intersection_volume (p : RectangularPrism) : ℝ := 
  (1 : ℝ) / 6 * prism_volume p

/-- Theorem: The volume of the intersection of two specific tetrahedra 
    within a rectangular prism is one-sixth of the prism's volume -/
theorem intersection_volume_is_one_sixth_of_prism (p : RectangularPrism) :
  intersection_volume p = (1 : ℝ) / 6 * prism_volume p := by
  -- Unfold the definitions
  unfold intersection_volume
  unfold prism_volume
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_volume_is_one_sixth_of_prism_l26_2637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l26_2660

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_properties (ω φ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_range : φ ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2))
  (h_period : ∀ x, f ω φ (x + Real.pi) = f ω φ x)
  (h_symmetry : ∀ x, f ω φ (Real.pi / 6 + x) = f ω φ (Real.pi / 6 - x)) :
  ω = 2 ∧ 
  φ = Real.pi / 3 ∧
  (∀ x, f ω φ (2 * Real.pi / 3 + x) = f ω φ (2 * Real.pi / 3 - x)) ∧
  (∀ x ∈ Set.Icc (-Real.pi / 6) 0, ∀ y ∈ Set.Icc (-Real.pi / 6) 0, x < y → f ω φ x < f ω φ y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l26_2660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_l26_2636

open Real MeasureTheory Interval

theorem integral_bounds (I : ℝ) :
  I = ∫ x in (-π/2)..π/2, (2 / (7 - 3 * Real.cos (2 * x))) →
  π/5 < I ∧ I < π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_bounds_l26_2636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_is_negative_one_seventh_l26_2618

def initial_numbers : List ℚ :=
  (List.range 2017).map (λ i => (i - 1008 : ℚ) / 1008)

def operation (x y : ℚ) : ℚ := x + 7 * x * y + y

theorem final_number_is_negative_one_seventh :
  ∃ (sequence : List (List ℚ)),
    sequence.length = 2017 ∧
    sequence.head! = initial_numbers ∧
    (∀ i < 2016, 
      ∃ x y, 
        x ∈ sequence[i]! ∧ 
        y ∈ sequence[i]! ∧ 
        x ≠ y ∧
        sequence[i+1]! = (sequence[i]!.erase x).erase y ++ [operation x y]) ∧
    sequence.getLast! = [-1/7] :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_is_negative_one_seventh_l26_2618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_triangles_area_sum_l26_2688

/-- The side length of the cube -/
def cube_side : ℝ := 2

/-- The number of vertices in the cube -/
def num_vertices : ℕ := 8

/-- The number of faces in the cube -/
def num_faces : ℕ := 6

/-- The number of edges in the cube -/
def num_edges : ℕ := 12

/-- The area of a triangle within a face of the cube -/
noncomputable def face_triangle_area : ℝ := (cube_side * cube_side) / 2

/-- The area of a triangle perpendicular to a face of the cube -/
noncomputable def perp_triangle_area : ℝ := cube_side * (cube_side * Real.sqrt 2) / 2

/-- The area of a triangle oblique to cube edges -/
noncomputable def oblique_triangle_area : ℝ := Real.sqrt 3 / 4 * (2 * cube_side * Real.sqrt 2) ^ 2

/-- The total number of triangles within faces -/
def num_face_triangles : ℕ := num_faces * 4

/-- The total number of triangles perpendicular to faces -/
def num_perp_triangles : ℕ := num_edges * 2

/-- The total number of triangles oblique to cube edges -/
def num_oblique_triangles : ℕ := num_vertices

theorem cube_triangles_area_sum :
  ∃ (m n p : ℕ),
    (num_face_triangles : ℝ) * face_triangle_area +
    (num_perp_triangles : ℝ) * perp_triangle_area +
    (num_oblique_triangles : ℝ) * oblique_triangle_area =
    m + Real.sqrt (n : ℝ) + Real.sqrt (p : ℝ) ∧
    m + n + p = 5424 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_triangles_area_sum_l26_2688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_l26_2647

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1/4

-- State the theorem
theorem tangent_line_condition (a : ℝ) : 
  (∃ x : ℝ, f a x = 0 ∧ (deriv (f a)) x = 0) → a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_condition_l26_2647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_company_problem_l26_2634

/-- Represents the relationship between selling price and daily sales volume -/
def sales_function (x : ℝ) : ℝ := -2 * x + 400

/-- Represents the profit function without donation -/
def profit_function (x : ℝ) : ℝ := (x - 60) * (sales_function x)

/-- Represents the profit function with donation -/
def profit_function_with_donation (x : ℝ) : ℝ := (x - 70) * (sales_function x)

theorem clothing_company_problem :
  (∀ (x : ℝ), x ∈ ({80, 90, 100, 110} : Set ℝ) → sales_function x = 400 - 2*x) ∧
  (∃ (x : ℝ), x < 160 ∧ profit_function x = 8000 ∧ x = 100) ∧
  (∃ (x : ℝ), ∀ (y : ℝ), profit_function_with_donation x ≥ profit_function_with_donation y ∧ x = 135) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_company_problem_l26_2634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_unique_solution_l26_2663

/-- Given a quadratic equation 9x^2 + bx + c = 0 with only one solution and b^2 = 4c,
    prove that the solution is -b/18 -/
theorem quadratic_equation_unique_solution (b c : ℝ) :
  (∃! x, 9 * x^2 + b * x + c = 0) →
  b^2 = 4 * c →
  9 * (-b / 18)^2 + b * (-b / 18) + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_unique_solution_l26_2663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l26_2678

theorem inequality_solution_set :
  let S := {x : ℝ | (1/3: ℝ)^(2*x^2 - 1) ≤ 3^(3*x - 4)}
  S = Set.Iic (-5/2) ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l26_2678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_premise_is_false_l26_2665

-- Define the basic geometric objects
variable (Line Plane : Type)

-- Define the relationships between geometric objects
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- Define the premise we want to disprove
def incorrect_premise (Line Plane : Type)
  (parallel_line_plane : Line → Plane → Prop)
  (parallel_lines : Line → Line → Prop)
  (line_in_plane : Line → Plane → Prop) : Prop :=
  ∀ (l : Line) (p : Plane),
    parallel_line_plane l p →
    ∀ (m : Line), line_in_plane m p → parallel_lines l m

-- Theorem stating that the premise is false
theorem premise_is_false :
  ∃ (Line Plane : Type)
    (parallel_line_plane : Line → Plane → Prop)
    (parallel_lines : Line → Line → Prop)
    (line_in_plane : Line → Plane → Prop),
  ¬incorrect_premise Line Plane parallel_line_plane parallel_lines line_in_plane :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_premise_is_false_l26_2665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ω_function_l26_2611

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem smallest_ω_function 
  (ω : ℝ) (φ : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_φ_bound : |φ| < π/2) 
  (h_symmetry : ∀ x, f ω φ (x - π/6) = f ω φ (-x - π/6)) 
  (h_value : f ω φ (π/ω) = -1/2) 
  (h_smallest : ∀ ω' φ', ω' > 0 → |φ'| < π/2 → 
    (∀ x, f ω' φ' (x - π/6) = f ω' φ' (-x - π/6)) → 
    f ω' φ' (π/ω') = -1/2 → ω ≤ ω') :
  f ω φ = λ x ↦ Real.sin (4 * x + π/6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_ω_function_l26_2611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l26_2655

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  -- A is at (0, 0)
  A = (0, 0) ∧
  -- B is on the x-axis
  (B.2 = 0) ∧
  -- C is in the first quadrant
  C.1 > 0 ∧ C.2 > 0

-- Define the right angle at A
def right_angle_at_A (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the tangent of angle C
noncomputable def tan_C (A B C : ℝ × ℝ) : ℝ :=
  (C.2 - A.2) / (C.1 - A.1)

-- Define the length of BC
noncomputable def BC_length (B C : ℝ × ℝ) : ℝ :=
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)

-- Define the length of AB
noncomputable def AB_length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Define the perimeter of the triangle
noncomputable def perimeter (A B C : ℝ × ℝ) : ℝ :=
  AB_length A B + BC_length B C + Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)

-- State the theorem
theorem right_triangle_properties (A B C : ℝ × ℝ) :
  triangle_ABC A B C →
  right_angle_at_A A B C →
  tan_C A B C = 3 →
  BC_length B C = 90 →
  AB_length A B = 27 * Real.sqrt 10 ∧
  perimeter A B C = 36 * Real.sqrt 10 + 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l26_2655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_exists_min_omega_l26_2694

noncomputable def y (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3) + 2

theorem min_omega (ω : ℝ) : 
  (∀ x, y ω x = y ω (x + 4 * Real.pi / 3)) → |ω| ≥ 3 / 2 := by
  sorry

theorem exists_min_omega : 
  ∃ ω : ℝ, (∀ x, y ω x = y ω (x + 4 * Real.pi / 3)) ∧ |ω| = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_exists_min_omega_l26_2694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_rate_is_three_l26_2669

/-- Represents the rainfall scenario with a fish tank --/
structure RainfallScenario where
  start_time : ℕ  -- Start time in hours after noon (1 pm = 1)
  initial_rainfall : ℚ  -- Rainfall in the first hour
  constant_rate : ℚ  -- Rainfall rate for the next four hours
  constant_duration : ℕ  -- Duration of constant rainfall in hours
  tank_height : ℚ  -- Height of the fish tank in inches
  end_time : ℕ  -- End time in hours after noon (10 pm = 10)

/-- Calculates the required rainfall rate for the remaining time --/
noncomputable def calculate_remaining_rate (scenario : RainfallScenario) : ℚ :=
  let total_duration := scenario.end_time - scenario.start_time
  let remaining_duration := total_duration - (1 + scenario.constant_duration)
  let initial_water := scenario.initial_rainfall + scenario.constant_rate * scenario.constant_duration
  let remaining_water := scenario.tank_height - initial_water
  remaining_water / remaining_duration

/-- Theorem stating that the remaining rainfall rate is 3 inches per hour --/
theorem rainfall_rate_is_three (scenario : RainfallScenario) 
    (h1 : scenario.start_time = 1)
    (h2 : scenario.initial_rainfall = 2)
    (h3 : scenario.constant_rate = 1)
    (h4 : scenario.constant_duration = 4)
    (h5 : scenario.tank_height = 18)
    (h6 : scenario.end_time = 10) :
  calculate_remaining_rate scenario = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_rate_is_three_l26_2669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_l26_2691

theorem quadratic_roots (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt (d + 1)) / 2 ∨ x = (-7 - Real.sqrt (d + 1)) / 2) → 
  d = 48 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_l26_2691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l26_2656

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

-- State the theorem
theorem power_function_property :
  ∀ α : ℝ, power_function α 4 = 2 → power_function α (1/4) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l26_2656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_bee_perpendicular_l26_2696

/-- The point where the spider starts moving away from the bee -/
noncomputable def spider_point : ℝ × ℝ := (61/10, 67/10)

/-- The bee's location -/
def bee_location : ℝ × ℝ := (14, 5)

/-- The slope of the spider's path -/
def spider_slope : ℝ := -3

/-- The y-intercept of the spider's path -/
def spider_intercept : ℝ := 25

/-- The spider's path as a function -/
def spider_path (x : ℝ) : ℝ := spider_slope * x + spider_intercept

theorem spider_bee_perpendicular :
  let (c, d) := spider_point
  (∀ x : ℝ, spider_path x = spider_slope * x + spider_intercept) ∧
  (spider_path c = d) ∧
  ((d - 5) / (c - 14) = 1 / spider_slope) ∧
  (c + d = 1337 / 100) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_bee_perpendicular_l26_2696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_l26_2672

-- Define the function g
def g : ℝ → ℝ := sorry

-- Define the inverse of g
def g_inv : ℝ → ℝ := sorry

-- State the given conditions
axiom g_inv_of_g : ∀ x, g_inv (g x) = x
axiom g_of_g_inv : ∀ x, g (g_inv x) = x
axiom g_4 : g 4 = 3
axiom g_1 : g 1 = 6
axiom g_3 : g 3 = 2

-- State the theorem to be proved
theorem inverse_composition : g_inv (g_inv 6 + g_inv 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_l26_2672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l26_2657

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Part 1: Prove that the only real solution to (e^x)^2 = e^x + 2 is x = ln 2
theorem part1 : ∃! x : ℝ, (f x)^2 = f x + 2 ∧ x = Real.log 2 := by
  sorry

-- Part 2: Prove that for all x ∈ ℝ, x + b ≤ e^x if and only if b ≤ 1
theorem part2 : ∀ b : ℝ, (∀ x : ℝ, x + b ≤ f x) ↔ b ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l26_2657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_turning_point_l26_2649

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 2) * x + a * Real.log x

-- Define the concept of a turning point
def is_turning_point (g : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, x ≠ x₀ → (g x - (g x₀ + (deriv g x₀) * (x - x₀))) / (x - x₀) > 0

theorem f_extrema_and_turning_point :
  (∃ (x_max x_min : ℝ), x_max = 1/2 ∧ x_min = 1 ∧
    IsLocalMax (f 1) x_max ∧ IsLocalMin (f 1) x_min) ∧
  is_turning_point (f 8) 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_turning_point_l26_2649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l26_2659

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    and its left and right foci F₁ and F₂ respectively,
    if a line passing through F₁ with a slope of 30° intersects
    the right branch of the hyperbola at point M,
    and MF₂ is perpendicular to the x-axis,
    then the eccentricity of the hyperbola is √3. -/
theorem hyperbola_eccentricity (a b c : ℝ) (F₁ F₂ M : ℝ × ℝ) :
  a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ (x, y) ∈ Set.range (fun t ↦ (a * Real.cosh t, b * Real.sinh t))) ∧
  F₁.1 = -c ∧ F₂.1 = c ∧
  (M.2 - F₁.2) / (M.1 - F₁.1) = Real.tan (30 * π / 180) ∧
  M.1 > 0 ∧
  (M.2 - F₂.2) / (M.1 - F₂.1) = 0 →
  c / a = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l26_2659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l26_2695

def f (x : ℝ) : ℝ := x^3 + x

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 → f (m * Real.cos θ) + f (1 - m) > 0) →
  m < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l26_2695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_squares_l26_2603

theorem triangle_area_from_squares (a b c : ℝ) (ha : a = 121) (hb : b = 64) (hc : c = 225) :
  (1/2) * Real.sqrt a * Real.sqrt b = 44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_squares_l26_2603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_restaurants_l26_2628

/-- Calculates the number of restaurants in a building with given conditions -/
theorem number_of_restaurants (total_units : ℕ) (h1 : total_units = 300) : 
  (total_units - total_units / 2) / 2 = 75 := by
  have residential_units := total_units / 2
  have non_residential_units := total_units - residential_units
  have restaurants := non_residential_units / 2
  sorry -- Placeholder for the actual proof

#eval (300 - 300 / 2) / 2 -- This will evaluate to 75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_restaurants_l26_2628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l26_2682

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

noncomputable def sum_arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + d * (n - 1)) / 2

theorem arithmetic_sequence_sum (a₁ d : ℝ) :
  (sum_arithmetic_sequence a₁ d 3 = 3) →
  (sum_arithmetic_sequence a₁ d 6 = 15) →
  (arithmetic_sequence a₁ d 10 + arithmetic_sequence a₁ d 11 + arithmetic_sequence a₁ d 12 = 30) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l26_2682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_problem_l26_2693

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define point K
def K : ℝ × ℝ := (-4, 0)

-- Define a tangent line to the parabola
def is_tangent (p : ℝ) (x y : ℝ) : Prop :=
  ∃ t : ℝ, parabola p x y ∧ (∀ x' y', parabola p x' y' → (y' - y)^2 ≥ t^2 * ((x' - x)^2 + (y' - y)^2))

-- Define that AB passes through the focus
def passes_through_focus (p : ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, focus p = (1 - t) • A + t • B

-- Define the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

theorem parabola_tangent_problem (p : ℝ) (A B : ℝ × ℝ) :
  parabola p A.1 A.2 →
  parabola p B.1 B.2 →
  is_tangent p A.1 A.2 →
  is_tangent p B.1 B.2 →
  passes_through_focus p A B →
  triangle_area K A B = 24 →
  p = 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_problem_l26_2693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_false_iff_a_geq_sqrt2_l26_2685

theorem proposition_p_false_iff_a_geq_sqrt2 (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 (π/4), Real.sin (2*x) + Real.cos (2*x) ≤ a) ↔ a ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_p_false_iff_a_geq_sqrt2_l26_2685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_eq_two_pi_l26_2609

theorem sum_of_solutions_eq_two_pi : 
  ∃ (S : Finset ℝ), 
    (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ (1 / Real.sin x) + (1 / Real.cos x) = 4) ∧
    (S.sum id) = 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_eq_two_pi_l26_2609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l26_2605

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sin (2 * ω * x) - Real.sqrt 3

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  is_periodic f T ∧ T > 0 ∧ ∀ T' > 0, is_periodic f T' → T ≤ T'

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (2 * x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem problem_solution (ω : ℝ) (h1 : ω > 0) (h2 : smallest_positive_period (f ω) (2 * Real.pi)) :
  ω = 1/2 ∧
  ∀ k : ℤ, monotone_increasing_on (g ω) (Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l26_2605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_OBC_measure_l26_2619

-- Define the circle and points
variable (O A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the cyclic quadrilateral property
def is_cyclic_quadrilateral (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the center of the circle
def is_center_of_circle (O A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the angle measure in degrees
noncomputable def angle_measure (P Q R : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- State the theorem
theorem angle_OBC_measure
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_center : is_center_of_circle O A B C D)
  (h_CDA : angle_measure C D A = 42)
  (h_DBC : angle_measure D B C = 10) :
  angle_measure O B C = 58 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_OBC_measure_l26_2619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_half_not_covered_l26_2692

theorem sqrt2_half_not_covered : 
  ∀ (a b : ℕ), 
    0 < a → a ≤ b → 
    Nat.Coprime a b → 
    |Real.sqrt 2 / 2 - (a : ℝ) / (b : ℝ)| > 1 / (4 * (b : ℝ)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt2_half_not_covered_l26_2692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_increase_approx_l26_2632

/-- Calculates the percent increase in sales between two periods -/
noncomputable def percentIncrease (salesThisYear salesLastYear : ℝ) : ℝ :=
  ((salesThisYear - salesLastYear) / salesLastYear) * 100

/-- Represents the sales data and exchange rates for a store over two years -/
structure SalesData where
  salesThisYearEuro : ℝ
  salesLastYearDollar : ℝ
  exchangeRateThisYear : ℝ
  exchangeRateLastYear : ℝ

/-- Theorem stating the approximate percent increase in sales -/
theorem sales_increase_approx (data : SalesData)
    (h1 : data.salesThisYearEuro = 460)
    (h2 : data.salesLastYearDollar = 320)
    (h3 : data.exchangeRateThisYear = 1.10)
    (h4 : data.exchangeRateLastYear = 1.15) :
    |percentIncrease (data.salesThisYearEuro * data.exchangeRateThisYear) data.salesLastYearDollar - 58.13| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_increase_approx_l26_2632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_two_in_id_l26_2629

def valid_id (n : Nat) : Bool :=
  n ≥ 1 && n ≤ 5000 && n.repr.all (λ c => c ≠ '2')

def count_valid_ids : Nat :=
  (List.range 5000).filter (λ n => valid_id (n + 1)) |>.length

theorem probability_no_two_in_id :
  count_valid_ids / 5000 = 729 / 1250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_two_in_id_l26_2629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_non_parallel_sides_l26_2625

/-- A trapezoid with specific properties -/
structure Trapezoid where
  -- The shorter base length
  c : ℝ
  -- The ratio of non-parallel sides
  k : ℝ
  -- Assumption that c and k are positive
  c_pos : c > 0
  k_pos : k > 0

/-- The lengths of non-parallel sides in the trapezoid -/
noncomputable def non_parallel_sides (t : Trapezoid) : ℝ × ℝ :=
  ((Real.sqrt 5 * t.c) / Real.sqrt (1 + t.k^2),
   (Real.sqrt 5 * t.k * t.c) / Real.sqrt (1 + t.k^2))

theorem trapezoid_non_parallel_sides (t : Trapezoid) :
  non_parallel_sides t =
    ((Real.sqrt 5 * t.c) / Real.sqrt (1 + t.k^2),
     (Real.sqrt 5 * t.k * t.c) / Real.sqrt (1 + t.k^2)) :=
by
  -- Unfold the definition of non_parallel_sides
  unfold non_parallel_sides
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_non_parallel_sides_l26_2625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newspaper_stand_max_profit_l26_2612

/-- Represents the daily profit function for a newspaper stand --/
def daily_profit (buy_price : ℝ) (sell_price : ℝ) (return_price : ℝ) (copies_bought : ℕ) (copies_sold : ℕ) : ℝ :=
  sell_price * (copies_sold : ℝ) + return_price * ((copies_bought - copies_sold) : ℝ) - buy_price * (copies_bought : ℝ)

/-- Represents the monthly profit function for a newspaper stand --/
def monthly_profit (buy_price : ℝ) (sell_price : ℝ) (return_price : ℝ) (copies_bought : ℕ) (high_demand_days : ℕ) (low_demand_days : ℕ) (high_demand_copies : ℕ) (low_demand_copies : ℕ) : ℝ :=
  ((high_demand_days : ℝ) * daily_profit buy_price sell_price return_price copies_bought high_demand_copies) +
  ((low_demand_days : ℝ) * daily_profit buy_price sell_price return_price copies_bought low_demand_copies)

theorem newspaper_stand_max_profit :
  let buy_price : ℝ := 0.20
  let sell_price : ℝ := 0.30
  let return_price : ℝ := 0.05
  let high_demand_days : ℕ := 20
  let low_demand_days : ℕ := 10
  let high_demand_copies : ℕ := 400
  let low_demand_copies : ℕ := 250
  let total_days : ℕ := high_demand_days + low_demand_days
  ∀ copies_bought : ℕ, copies_bought ≥ low_demand_copies → copies_bought ≤ high_demand_copies →
    monthly_profit buy_price sell_price return_price copies_bought high_demand_days low_demand_days high_demand_copies low_demand_copies ≤
    monthly_profit buy_price sell_price return_price high_demand_copies high_demand_days low_demand_days high_demand_copies low_demand_copies ∧
    monthly_profit buy_price sell_price return_price high_demand_copies high_demand_days low_demand_days high_demand_copies low_demand_copies = 825 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_newspaper_stand_max_profit_l26_2612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_numbers_product_l26_2617

theorem two_numbers_product (x y : ℚ) : 
  (x - y : ℚ) / (x + y : ℚ) = 1 / 8 ∧ 
  (x + y : ℚ) / (x * y : ℚ) = 8 / 30 → 
  x * y = 400 / 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_numbers_product_l26_2617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_space_occupied_l26_2631

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℚ := d.length * d.width * d.height

/-- Represents the storage cost and payment information -/
structure StorageInfo where
  costPerBoxPerMonth : ℚ
  totalMonthlyPayment : ℚ

/-- Calculates the number of boxes stored based on the storage information -/
def numberOfBoxes (s : StorageInfo) : ℚ := s.totalMonthlyPayment / s.costPerBoxPerMonth

/-- Theorem stating that the total space occupied by the boxes is 1,080,000 cubic inches -/
theorem total_space_occupied (d : BoxDimensions) (s : StorageInfo)
    (h1 : d = ⟨15, 12, 10⟩)
    (h2 : s.costPerBoxPerMonth = 6/10)
    (h3 : s.totalMonthlyPayment = 360) :
    boxVolume d * numberOfBoxes s = 1080000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_space_occupied_l26_2631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l26_2614

noncomputable def a (m : ℝ) : ℝ × ℝ := (Real.sqrt 2, m)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

noncomputable def f (m x : ℝ) : ℝ := (a m).1 * (b x).1 + (a m).2 * (b x).2

theorem problem_solution (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x, f m x ≤ 2) (h3 : ∃ x, f m x = 2)
  (A B C : ℝ) (hC : C = π / 3)
  (hf : f m (A - π/4) + f m (B - π/4) = 12 * Real.sqrt 2 * Real.sin A * Real.sin B)
  (hc : Real.sqrt 6 = 2 * Real.sqrt 2 * Real.sin C) :
  m = Real.sqrt 2 ∧ 
  (∀ x, f m x = f m (x + 2*π)) ∧
  (1/2 * (2 * Real.sqrt 2 * Real.sin A) * (2 * Real.sqrt 2 * Real.sin B) * Real.sin C = Real.sqrt 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l26_2614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l26_2677

theorem order_of_abc (a b c : ℝ) : 
  a = Real.log 6 / Real.log 0.7 →
  b = 6^(0.7 : ℝ) →
  c = 0.7^(0.6 : ℝ) →
  b > c ∧ c > a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l26_2677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_around_circle_l26_2671

/-- The shortest path from (0,0) to (15,20) avoiding a circle -/
theorem shortest_path_around_circle : ∃ (path : ℝ), path = 2 * Real.sqrt 94 + 3 * Real.pi := by
  -- Define the start and end points
  let start : ℝ × ℝ := (0, 0)
  let end_point : ℝ × ℝ := (15, 20)

  -- Define the circle
  let circle_center : ℝ × ℝ := (7, 9)
  let circle_radius : ℝ := 6

  -- Define the function for the circle
  let circle_equation (x y : ℝ) : Prop :=
    (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2

  -- Define the shortest path that avoids the circle
  let shortest_path : ℝ := 2 * Real.sqrt 94 + 3 * Real.pi

  -- The proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_around_circle_l26_2671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_f_range_of_g_range_of_a_for_g_leq_a_l26_2630

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (2^x)

-- Theorem 1: Range of a for f(x) ≥ 2
theorem range_of_a_for_f (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 2) ↔ -2 ≤ a ∧ a ≤ 2 := by
  sorry

-- Theorem 2: Range of g(x)
theorem range_of_g (a : ℝ) :
  (∀ x : ℝ, g a x ≥ 3 - a^2/4 ∧ (a < 0 → g a x ≥ 3 - a^2/4) ∧ (a ≥ 0 → g a x > 3)) := by
  sorry

-- Theorem 3: Range of a for g(x) ≤ a to have solutions
theorem range_of_a_for_g_leq_a (a : ℝ) :
  (∃ x : ℝ, g a x ≤ a) ↔ a ≤ -6 ∨ a > 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_f_range_of_g_range_of_a_for_g_leq_a_l26_2630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_squirrels_is_125_l26_2610

/-- The number of squirrels counted by each student -/
def squirrels_counted (n : ℕ) : ℚ :=
  match n with
  | 1 => 15
  | 2 => squirrels_counted 1 * (1 + 1/3)
  | 3 => squirrels_counted 2 * (1 + 1/4)
  | 4 => squirrels_counted 3 * (1 + 1/5)
  | 5 => squirrels_counted 4 * (1 + 1/6)
  | _ => 0

/-- The total number of squirrels counted by all five students -/
def total_squirrels : ℚ :=
  (Finset.range 5).sum (λ i => squirrels_counted (i + 1))

/-- Theorem stating that the total number of squirrels counted is 125 -/
theorem total_squirrels_is_125 : total_squirrels = 125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_squirrels_is_125_l26_2610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_l26_2679

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℚ
  b : ℚ
  c : ℚ

noncomputable def Line.slope (l : Line) : ℚ := -l.a / l.b

def are_parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope ∧ l1 ≠ l2

theorem parallel_lines : 
  let line1 : Line := ⟨-3, 2, 15⟩  -- 2y-3x=15
  let line2 : Line := ⟨3, 2, 5⟩   -- 3x+2y=5
  let line3 : Line := ⟨-6, 4, 18⟩ -- 4y-6x=18
  let line4 : Line := ⟨-3, 2, 5⟩  -- -3x+2y=5
  are_parallel line1 line3 := by
  sorry

#check parallel_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_l26_2679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_l26_2646

-- Define the coefficients of the quadratic equation
noncomputable def a : ℝ := 7 + 4 * Real.sqrt 3
noncomputable def b : ℝ := 2 + Real.sqrt 3
def c : ℝ := -2

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the roots of the equation
noncomputable def root1 : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
noncomputable def root2 : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)

-- State the theorem
theorem root_difference : |root1 - root2| = 6 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_l26_2646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_condition_count_l26_2664

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (equidistant : Plane → Plane → Prop)

-- Define the lines and planes
variable (a b : Line)
variable (α β γ : Plane)

-- State the theorem
theorem parallel_planes_condition_count 
  (h1 : skew a b)
  (h2 : subset a α)
  (h3 : subset b β)
  (h4 : parallel_line_plane a β)
  (h5 : parallel_line_plane b α)
  (h6 : equidistant α β)
  (h7 : perpendicular α γ)
  (h8 : perpendicular β γ) :
  ∃! (condition : Prop), condition → parallel_plane_plane α β := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_condition_count_l26_2664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l26_2653

noncomputable def f (x y : Real) : Real :=
  Real.sqrt ((1 + x * y) / (1 + x^2)) + Real.sqrt ((1 - x * y) / (1 + y^2))

theorem f_range (x y : Real) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) :
  1 ≤ f x y ∧ f x y ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l26_2653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l26_2654

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, |x - 2| < 1 → x^2 + x - 2 > 0) ∧
  (∃ x : ℝ, x^2 + x - 2 > 0 ∧ |x - 2| ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l26_2654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diametric_points_l26_2604

structure Circle (n : ℕ) where
  length : ℕ := 6 * n
  num_points : ℕ := 3 * n
  num_arcs : ℕ := 3 * n
  arcs_length_1 : ℕ := n
  arcs_length_2 : ℕ := n
  arcs_length_3 : ℕ := n

def Line := Set (ℝ × ℝ)

def contains (l : Line) (p : ℕ) : Prop := sorry

def passes_through_center (l : Line) : Prop := sorry

def has_diametric_points (c : Circle n) : Prop :=
  ∃ p q : ℕ, p < c.num_points ∧ q < c.num_points ∧ p ≠ q ∧
    ∃ line : Line, contains line p ∧ contains line q ∧ passes_through_center line

theorem circle_diametric_points (n : ℕ) (c : Circle n) :
  has_diametric_points c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diametric_points_l26_2604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_equality_l26_2686

theorem cosine_product_equality : 
  Real.cos (π / 15) * Real.cos (2 * π / 15) * Real.cos (3 * π / 15) * Real.cos (4 * π / 15) * 
  Real.cos (5 * π / 15) * Real.cos (6 * π / 15) * Real.cos (7 * π / 15) = (1 / 2)^7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_product_equality_l26_2686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_terms_l26_2673

def S (n : ℕ) : ℤ := n^2 - 2*n + 3

def a : ℕ → ℤ
  | 0 => 0  -- Add a case for 0
  | 1 => S 1
  | n+1 => S (n+1) - S n

theorem first_three_terms :
  (a 1 = 2) ∧ (a 2 = 1) ∧ (a 3 = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_three_terms_l26_2673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l26_2640

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (4, 2)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

def vector_a (l : ℝ) : ℝ × ℝ := (2, l)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem parallel_vectors_lambda :
  ∀ l : ℝ, parallel (vector_a l) vector_AB → l = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l26_2640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_inscribed_circle_max_area_l26_2670

/-- Represents an ellipse centered at the origin -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ 0 < b
  h_a_ge_b : a ≥ b

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on the ellipse -/
def Ellipse.onEllipse (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The foci of the ellipse -/
noncomputable def Ellipse.foci (e : Ellipse) : ℝ × ℝ :=
  (e.a * e.eccentricity, -e.a * e.eccentricity)

/-- A line passing through a point -/
structure Line where
  m : ℝ
  b : ℝ

/-- A point is on the line -/
def Line.onLine (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.b

/-- The inscribed circle of a triangle -/
noncomputable def inscribedCircleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.pi * ((s - a) * (s - b) * (s - c) / s)

/-- The main theorem -/
theorem ellipse_inscribed_circle_max_area (e : Ellipse) :
  e.eccentricity = 1/2 →
  e.onEllipse 1 (3/2) →
  ∃ (l : Line),
    l.onLine e.foci.2 0 →
    ∀ (l' : Line),
      l'.onLine e.foci.2 0 →
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        e.onEllipse x₁ y₁ ∧
        e.onEllipse x₂ y₂ ∧
        l'.onLine x₁ y₁ ∧
        l'.onLine x₂ y₂ ∧
        inscribedCircleArea (Real.sqrt ((x₁ - e.foci.1)^2 + y₁^2))
                            (Real.sqrt ((x₂ - e.foci.1)^2 + y₂^2))
                            (Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2))
        ≤ 9 * Real.pi / 16 ∧
        (inscribedCircleArea (Real.sqrt ((x₁ - e.foci.1)^2 + y₁^2))
                             (Real.sqrt ((x₂ - e.foci.1)^2 + y₂^2))
                             (Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2))
         = 9 * Real.pi / 16 ↔ l' = l) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_inscribed_circle_max_area_l26_2670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_GH_in_isosceles_trapezoid_l26_2627

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  -- Points of the trapezoid
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- G is a point on AD
  G : ℝ × ℝ
  -- H is the foot of the altitude from B to AD
  H : ℝ × ℝ
  -- AD is parallel to BC
  parallelADBC : (A.2 = D.2) ∧ (B.2 = C.2)
  -- Angle at AD is π/4
  angleAD : Real.arctan ((B.2 - A.2) / (B.1 - A.1)) = π / 4
  -- Diagonals have length 8√30
  diagonalLength : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 8 * Real.sqrt 30
  -- G is at distances 8√10 and 16√10 from A and D respectively
  distanceAG : Real.sqrt ((G.1 - A.1)^2 + (G.2 - A.2)^2) = 8 * Real.sqrt 10
  distanceGD : Real.sqrt ((D.1 - G.1)^2 + (D.2 - G.2)^2) = 16 * Real.sqrt 10
  -- H is on AD
  HonAD : H.2 = A.2

/-- The distance GH in an isosceles trapezoid with specific properties is 4√30 -/
theorem distance_GH_in_isosceles_trapezoid (t : IsoscelesTrapezoid) :
  Real.sqrt ((t.G.1 - t.H.1)^2 + (t.G.2 - t.H.2)^2) = 4 * Real.sqrt 30 := by
  sorry

#check distance_GH_in_isosceles_trapezoid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_GH_in_isosceles_trapezoid_l26_2627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_little_o_of_x_l26_2643

-- Define the little-o notation
def is_little_o (f g : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x| ≤ ε * |g x|

-- State the theorem
theorem f_is_little_o_of_x (f : ℝ → ℝ) 
  (h1 : ∀ x, 0 < x → x < 1 → ∃ y, f x = y)  -- f is defined on (0, 1)
  (h2 : ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < x ∧ x < δ → |f x| < ε)  -- lim_{x → 0} f(x) = 0
  (h3 : is_little_o (λ x ↦ f x - f (x/2)) (λ x ↦ x))  -- f(x) - f(x/2) = o(x)
  : is_little_o f (λ x ↦ x) :=  -- f(x) = o(x)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_little_o_of_x_l26_2643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_number_proof_l26_2621

theorem square_number_proof (n : ℕ) (h1 : n ≥ 3) :
  let m := n * (n - 1) / 2
  ∃ (cards : Finset ℕ),
    (2 ∣ m) ∧
    cards.card = n ∧
    (∀ i ∈ cards, i ≤ m) ∧
    (∀ i j k l : ℕ, i ∈ cards → j ∈ cards → k ∈ cards → l ∈ cards → 
      i ≠ j → k ≠ l → (i + j) % m ≠ (k + l) % m) →
    ∃ k : ℕ, n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_number_proof_l26_2621
