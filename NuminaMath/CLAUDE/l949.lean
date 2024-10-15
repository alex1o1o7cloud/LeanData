import Mathlib

namespace NUMINAMATH_CALUDE_decimal_51_to_binary_binary_to_decimal_51_l949_94928

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- Converts a list of bits to a natural number -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_51_to_binary :
  to_binary 51 = [true, true, false, false, true, true] :=
by sorry

theorem binary_to_decimal_51 :
  from_binary [true, true, false, false, true, true] = 51 :=
by sorry

end NUMINAMATH_CALUDE_decimal_51_to_binary_binary_to_decimal_51_l949_94928


namespace NUMINAMATH_CALUDE_solution_is_correct_l949_94963

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(8, 3, 3, 1), (5, 4, 3, 1), (3, 2, 2, 2), (7, 6, 2, 1), (9, 5, 2, 1), (15, 4, 2, 1),
   (1, 1, 1, 7), (2, 1, 1, 5), (3, 2, 1, 3), (8, 3, 1, 2), (5, 4, 1, 2)}

def satisfies_conditions (x y z t : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧
  x * y * z = Nat.factorial t ∧
  (x + 1) * (y + 1) * (z + 1) = Nat.factorial (t + 1)

theorem solution_is_correct :
  ∀ x y z t, (x, y, z, t) ∈ solution_set ↔ satisfies_conditions x y z t := by
  sorry

end NUMINAMATH_CALUDE_solution_is_correct_l949_94963


namespace NUMINAMATH_CALUDE_gnollish_sentences_l949_94995

/-- The number of words in the Gnollish language -/
def num_words : ℕ := 4

/-- The length of a sentence in the Gnollish language -/
def sentence_length : ℕ := 3

/-- The number of invalid sentence patterns due to the restriction -/
def num_invalid_patterns : ℕ := 2

/-- The number of choices for the unrestricted word in an invalid pattern -/
def choices_for_unrestricted : ℕ := num_words

theorem gnollish_sentences :
  (num_words ^ sentence_length) - (num_invalid_patterns * choices_for_unrestricted) = 56 := by
  sorry

end NUMINAMATH_CALUDE_gnollish_sentences_l949_94995


namespace NUMINAMATH_CALUDE_inequality_proof_l949_94904

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c = d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l949_94904


namespace NUMINAMATH_CALUDE_smallest_positive_period_of_even_odd_functions_l949_94918

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function g is odd if g(x) = -g(-x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

/-- The period of a function f is p if f(x + p) = f(x) for all x -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

/-- The smallest positive period of a function -/
def SmallestPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ IsPeriod f p ∧ ∀ q, 0 < q ∧ q < p → ¬IsPeriod f q

theorem smallest_positive_period_of_even_odd_functions
  (f g : ℝ → ℝ) (c : ℝ)
  (hf : IsEven f)
  (hg : IsOdd g)
  (h : ∀ x, f x = -g (x + c))
  (hc : c > 0) :
  SmallestPositivePeriod f (4 * c) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_period_of_even_odd_functions_l949_94918


namespace NUMINAMATH_CALUDE_family_savings_l949_94945

def initial_savings : ℕ := 1147240
def income : ℕ := 509600
def expenses : ℕ := 276000

theorem family_savings : initial_savings + income - expenses = 1340840 := by
  sorry

end NUMINAMATH_CALUDE_family_savings_l949_94945


namespace NUMINAMATH_CALUDE_range_of_a_l949_94977

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (2*x - 1) / (x - 1) ≤ 0 → x^2 - (2*a + 1)*x + a*(a + 1) < 0) ∧ 
  (∃ x : ℝ, x^2 - (2*a + 1)*x + a*(a + 1) < 0 ∧ ¬((2*x - 1) / (x - 1) ≤ 0)) →
  0 ≤ a ∧ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l949_94977


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l949_94911

theorem rectangle_perimeter (a b : ℕ) : 
  a ≠ b → -- non-square condition
  a * b = 3 * (2 * a + 2 * b) → -- area equals 3 times perimeter
  2 * a + 2 * b = 36 ∨ 2 * a + 2 * b = 28 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l949_94911


namespace NUMINAMATH_CALUDE_max_correct_answers_l949_94916

theorem max_correct_answers (total_questions : Nat) (correct_points : Int) (incorrect_points : Int) (total_score : Int) :
  total_questions = 30 →
  correct_points = 4 →
  incorrect_points = -3 →
  total_score = 72 →
  ∃ (correct incorrect unanswered : Nat),
    correct + incorrect + unanswered = total_questions ∧
    correct * correct_points + incorrect * incorrect_points = total_score ∧
    correct ≤ 21 ∧
    ∀ (c i u : Nat),
      c + i + u = total_questions →
      c * correct_points + i * incorrect_points = total_score →
      c ≤ 21 :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l949_94916


namespace NUMINAMATH_CALUDE_unoccupied_volume_correct_l949_94991

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of an ice cube -/
structure IceCubeDimensions where
  side : ℕ

/-- Calculates the unoccupied volume in a tank given its dimensions, water depth, and ice cubes -/
def unoccupiedVolume (tank : TankDimensions) (waterDepth : ℕ) (iceCube : IceCubeDimensions) (numIceCubes : ℕ) : ℕ :=
  let tankVolume := tank.length * tank.width * tank.height
  let waterVolume := tank.length * tank.width * waterDepth
  let iceCubeVolume := iceCube.side * iceCube.side * iceCube.side
  let totalIceVolume := numIceCubes * iceCubeVolume
  tankVolume - (waterVolume + totalIceVolume)

/-- Theorem stating the unoccupied volume in the tank under given conditions -/
theorem unoccupied_volume_correct :
  let tank : TankDimensions := ⟨12, 12, 15⟩
  let waterDepth : ℕ := 7
  let iceCube : IceCubeDimensions := ⟨3⟩
  let numIceCubes : ℕ := 15
  unoccupiedVolume tank waterDepth iceCube numIceCubes = 747 := by
  sorry

end NUMINAMATH_CALUDE_unoccupied_volume_correct_l949_94991


namespace NUMINAMATH_CALUDE_first_day_is_friday_l949_94978

/-- Days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after n days -/
def dayAfter (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (dayAfter d m)

/-- Theorem: If the 25th day of a month is a Monday, then the 1st day of that month is a Friday -/
theorem first_day_is_friday (d : DayOfWeek) : 
  dayAfter d 24 = DayOfWeek.Monday → d = DayOfWeek.Friday :=
by
  sorry


end NUMINAMATH_CALUDE_first_day_is_friday_l949_94978


namespace NUMINAMATH_CALUDE_room_tiles_theorem_l949_94965

/-- Given a room with length and width in centimeters, 
    calculate the least number of square tiles required to cover the floor. -/
def leastNumberOfTiles (length width : ℕ) : ℕ :=
  let tileSize := Nat.gcd length width
  (length / tileSize) * (width / tileSize)

/-- Theorem stating that for a room with length 720 cm and width 432 cm,
    the least number of square tiles required is 15. -/
theorem room_tiles_theorem :
  leastNumberOfTiles 720 432 = 15 := by
  sorry

#eval leastNumberOfTiles 720 432

end NUMINAMATH_CALUDE_room_tiles_theorem_l949_94965


namespace NUMINAMATH_CALUDE_inequality_proof_l949_94933

theorem inequality_proof (α x y z : ℝ) (hα : α > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x * y + y * z + z * x = α) :
  (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ 16 * (x / z + z / x + 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l949_94933


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_a_in_range_l949_94901

def set_A (a : ℝ) : Set ℝ := {y | y > a^2 + 1 ∨ y < a}
def set_B : Set ℝ := {y | 2 ≤ y ∧ y ≤ 4}

theorem intersection_nonempty_iff_a_in_range (a : ℝ) :
  (set_A a ∩ set_B).Nonempty ↔ (-Real.sqrt 3 < a ∧ a < Real.sqrt 3) ∨ a > 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_a_in_range_l949_94901


namespace NUMINAMATH_CALUDE_edwards_initial_money_l949_94999

theorem edwards_initial_money (spent_first spent_second remaining : ℕ) : 
  spent_first = 9 → 
  spent_second = 8 → 
  remaining = 17 → 
  spent_first + spent_second + remaining = 34 :=
by sorry

end NUMINAMATH_CALUDE_edwards_initial_money_l949_94999


namespace NUMINAMATH_CALUDE_polygonal_chains_10_9_l949_94935

/-- The number of points on the circle -/
def n : ℕ := 10

/-- The number of sides in each polygonal chain -/
def sides : ℕ := 9

/-- A function that calculates the number of non-closed, non-self-intersecting 
    polygonal chains with 'sides' sides that can be formed from 'n' points on a circle -/
def polygonal_chains (n : ℕ) (sides : ℕ) : ℕ :=
  if sides ≤ n ∧ sides > 2 then
    (n * 2^(sides - 1)) / 2
  else
    0

/-- Theorem stating that the number of non-closed, non-self-intersecting 9-sided 
    polygonal chains that can be formed with 10 points on a circle as vertices is 1280 -/
theorem polygonal_chains_10_9 : polygonal_chains n sides = 1280 := by
  sorry

end NUMINAMATH_CALUDE_polygonal_chains_10_9_l949_94935


namespace NUMINAMATH_CALUDE_kendra_shirts_theorem_l949_94980

/-- Calculates the number of shirts Kendra needs for two weeks -/
def shirts_needed : ℕ :=
  let school_days := 5
  let after_school_days := 3
  let saturday_shirts := 1
  let sunday_shirts := 2
  let weeks := 2
  (school_days + after_school_days + saturday_shirts + sunday_shirts) * weeks

/-- Theorem stating that Kendra needs 22 shirts for two weeks -/
theorem kendra_shirts_theorem : shirts_needed = 22 := by
  sorry

end NUMINAMATH_CALUDE_kendra_shirts_theorem_l949_94980


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l949_94984

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 9)
  (h2 : train_speed = 45)
  (h3 : train_length = 120)
  (h4 : initial_distance = 240)
  : (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 36 :=
by
  sorry

#check train_passing_jogger_time

end NUMINAMATH_CALUDE_train_passing_jogger_time_l949_94984


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l949_94914

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 2*x + 3 - 4*x
  ∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 6 ∧ x₂ = 3 - Real.sqrt 6 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l949_94914


namespace NUMINAMATH_CALUDE_complex_sum_problem_l949_94929

theorem complex_sum_problem (p r t u : ℝ) :
  let q : ℝ := 5
  let s : ℝ := 2 * q
  t = -p - r →
  Complex.I * (q + s + u) = Complex.I * 7 →
  Complex.I * u + Complex.I = Complex.I * (-8) + Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l949_94929


namespace NUMINAMATH_CALUDE_salary_percent_increase_l949_94905

def salary_increase : ℝ := 5000
def new_salary : ℝ := 25000

theorem salary_percent_increase :
  let original_salary := new_salary - salary_increase
  let percent_increase := (salary_increase / original_salary) * 100
  percent_increase = 25 := by
sorry

end NUMINAMATH_CALUDE_salary_percent_increase_l949_94905


namespace NUMINAMATH_CALUDE_dragon_eventual_defeat_l949_94997

/-- Represents the probabilities of head growth after each cut -/
structure HeadGrowthProbabilities where
  two_heads : ℝ
  one_head : ℝ
  no_heads : ℝ

/-- The probability of eventually defeating the dragon -/
def defeat_probability (probs : HeadGrowthProbabilities) : ℝ :=
  sorry

/-- The theorem stating that the dragon will eventually be defeated -/
theorem dragon_eventual_defeat (probs : HeadGrowthProbabilities) 
  (h1 : probs.two_heads = 1/4)
  (h2 : probs.one_head = 1/3)
  (h3 : probs.no_heads = 5/12)
  (h4 : probs.two_heads + probs.one_head + probs.no_heads = 1) :
  defeat_probability probs = 1 := by
  sorry

end NUMINAMATH_CALUDE_dragon_eventual_defeat_l949_94997


namespace NUMINAMATH_CALUDE_unique_solution_equation_l949_94909

theorem unique_solution_equation (x : ℝ) : 
  (8^x * (3*x + 1) = 4) ↔ (x = 1/3) := by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l949_94909


namespace NUMINAMATH_CALUDE_min_value_of_f_l949_94989

/-- The quadratic function f(x) = 3x^2 + 8x + 15 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 8 * x + 15

/-- The minimum value of f(x) is 29/3 -/
theorem min_value_of_f : 
  ∀ x : ℝ, f x ≥ 29/3 ∧ ∃ x₀ : ℝ, f x₀ = 29/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l949_94989


namespace NUMINAMATH_CALUDE_calculation_proof_l949_94986

theorem calculation_proof : (180 : ℚ) / (15 + 12 * 3 - 9) = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l949_94986


namespace NUMINAMATH_CALUDE_stone_piles_impossible_l949_94923

/-- Represents a configuration of stone piles -/
def StonePiles := List Nat

/-- The initial configuration of stone piles -/
def initial_piles : StonePiles := [51, 49, 5]

/-- Merges two piles in the configuration -/
def merge_piles (piles : StonePiles) (i j : Nat) : StonePiles :=
  sorry

/-- Splits an even-numbered pile into two equal piles -/
def split_pile (piles : StonePiles) (i : Nat) : StonePiles :=
  sorry

/-- Checks if a configuration consists of 105 piles of 1 stone each -/
def is_final_state (piles : StonePiles) : Prop :=
  piles.length = 105 ∧ piles.all (· = 1)

/-- Represents a sequence of operations on the stone piles -/
inductive Operation
  | Merge (i j : Nat)
  | Split (i : Nat)

/-- Applies a sequence of operations to the initial configuration -/
def apply_operations (ops : List Operation) : StonePiles :=
  sorry

theorem stone_piles_impossible :
  ∀ (ops : List Operation), ¬(is_final_state (apply_operations ops)) :=
sorry

end NUMINAMATH_CALUDE_stone_piles_impossible_l949_94923


namespace NUMINAMATH_CALUDE_second_triangle_weight_l949_94966

/-- Represents an equilateral triangle with given side length and weight -/
structure EquilateralTriangle where
  side_length : ℝ
  weight : ℝ

/-- Calculate the weight of a second equilateral triangle given the properties of a first triangle -/
def calculate_second_triangle_weight (t1 t2 : EquilateralTriangle) : Prop :=
  t1.side_length = 2 ∧ 
  t1.weight = 20 ∧ 
  t2.side_length = 4 ∧ 
  t2.weight = 80

theorem second_triangle_weight (t1 t2 : EquilateralTriangle) : 
  calculate_second_triangle_weight t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_second_triangle_weight_l949_94966


namespace NUMINAMATH_CALUDE_third_divisor_l949_94983

theorem third_divisor (n : ℕ) (h1 : n = 200) 
  (h2 : ∃ k₁ k₂ k₃ k₄ : ℕ, n + 20 = 15 * k₁ ∧ n + 20 = 30 * k₂ ∧ n + 20 = 60 * k₄) 
  (h3 : ∃ x : ℕ, x ≠ 15 ∧ x ≠ 30 ∧ x ≠ 60 ∧ ∃ k : ℕ, n + 20 = x * k) : 
  ∃ x : ℕ, x = 11 ∧ x ≠ 15 ∧ x ≠ 30 ∧ x ≠ 60 ∧ ∃ k : ℕ, n + 20 = x * k :=
sorry

end NUMINAMATH_CALUDE_third_divisor_l949_94983


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l949_94937

theorem quadratic_equation_solution (y : ℝ) : 
  (((8 * y^2 + 50 * y + 5) / (3 * y + 21)) = 4 * y + 3) ↔ 
  (y = (-43 + Real.sqrt 921) / 8 ∨ y = (-43 - Real.sqrt 921) / 8) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l949_94937


namespace NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l949_94982

theorem cos_pi_third_minus_alpha (α : Real) 
  (h : Real.sin (π / 6 + α) = 1 / 3) : 
  Real.cos (π / 3 - α) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l949_94982


namespace NUMINAMATH_CALUDE_workshop_workers_count_l949_94970

/-- Proves that the total number of workers in a workshop is 14 given specific salary conditions -/
theorem workshop_workers_count : ∀ (W : ℕ) (N : ℕ),
  -- Average salary of all workers is 9000
  W * 9000 = 7 * 12000 + N * 6000 →
  -- Total workers is sum of technicians and non-technicians
  W = 7 + N →
  -- Conclusion: Total number of workers is 14
  W = 14 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_count_l949_94970


namespace NUMINAMATH_CALUDE_video_game_lives_l949_94944

/-- Calculates the total lives after completing all levels in a video game -/
def total_lives (initial : ℝ) (hard_part : ℝ) (next_level : ℝ) (extra_challenge1 : ℝ) (extra_challenge2 : ℝ) : ℝ :=
  initial + hard_part + next_level + extra_challenge1 + extra_challenge2

/-- Theorem stating that the total lives after completing all levels is 261.0 -/
theorem video_game_lives :
  let initial : ℝ := 143.0
  let hard_part : ℝ := 14.0
  let next_level : ℝ := 27.0
  let extra_challenge1 : ℝ := 35.0
  let extra_challenge2 : ℝ := 42.0
  total_lives initial hard_part next_level extra_challenge1 extra_challenge2 = 261.0 := by
  sorry


end NUMINAMATH_CALUDE_video_game_lives_l949_94944


namespace NUMINAMATH_CALUDE_bottle_arrangement_l949_94958

theorem bottle_arrangement (x : ℕ) : 
  (x^2 + 36 = (x + 1)^2 + 3) → (x^2 + 36 = 292) :=
by
  sorry

end NUMINAMATH_CALUDE_bottle_arrangement_l949_94958


namespace NUMINAMATH_CALUDE_shoe_cost_l949_94930

/-- Given a suit purchase, a discount, and a total paid amount, prove the cost of shoes. -/
theorem shoe_cost (suit_price discount total_paid : ℤ) (h1 : suit_price = 430) (h2 : discount = 100) (h3 : total_paid = 520) :
  suit_price + (total_paid + discount - suit_price) = total_paid + discount := by
  sorry

#eval 520 + 100 - 430  -- Expected output: 190

end NUMINAMATH_CALUDE_shoe_cost_l949_94930


namespace NUMINAMATH_CALUDE_sin_axis_of_symmetry_l949_94967

/-- Proves that x = π/12 is one of the axes of symmetry for the function y = sin(2x + π/3) -/
theorem sin_axis_of_symmetry :
  ∃ (k : ℤ), 2 * (π/12 : ℝ) + π/3 = π/2 + k*π := by sorry

end NUMINAMATH_CALUDE_sin_axis_of_symmetry_l949_94967


namespace NUMINAMATH_CALUDE_part_one_part_two_l949_94950

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (3 - 2*a) * x - 6

-- Part 1
theorem part_one :
  f 1 x > 0 ↔ x < -3 ∨ x > 2 :=
sorry

-- Part 2
theorem part_two (a : ℝ) (h : a < 0) :
  (a < -3/2 → (f a x < 0 ↔ x < -3/a ∨ x > 2)) ∧
  (a = -3/2 → (f a x < 0 ↔ x ≠ 2)) ∧
  (-3/2 < a → (f a x < 0 ↔ x < 2 ∨ x > -3/a)) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l949_94950


namespace NUMINAMATH_CALUDE_limit_special_function_l949_94941

/-- The limit of (7^(3x) - 3^(2x)) / (tan(x) + x^3) as x approaches 0 is ln(343/9) -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ →
    |((7^(3*x) - 3^(2*x)) / (Real.tan x + x^3)) - Real.log (343/9)| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_special_function_l949_94941


namespace NUMINAMATH_CALUDE_least_common_remainder_least_common_remainder_achieved_least_common_remainder_is_126_l949_94994

theorem least_common_remainder (n : ℕ) : n > 1 ∧ n % 25 = 1 ∧ n % 7 = 1 → n ≥ 126 := by
  sorry

theorem least_common_remainder_achieved : 126 % 25 = 1 ∧ 126 % 7 = 1 := by
  sorry

theorem least_common_remainder_is_126 : ∃ (n : ℕ), n = 126 ∧ n > 1 ∧ n % 25 = 1 ∧ n % 7 = 1 ∧ 
  ∀ (m : ℕ), m > 1 ∧ m % 25 = 1 ∧ m % 7 = 1 → m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_least_common_remainder_least_common_remainder_achieved_least_common_remainder_is_126_l949_94994


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l949_94990

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 17th term is 12 and the 18th term is 15, the 3rd term is -30. -/
theorem arithmetic_sequence_third_term 
  (a : ℕ → ℤ) 
  (h_arithmetic : isArithmeticSequence a) 
  (h_17th : a 17 = 12) 
  (h_18th : a 18 = 15) : 
  a 3 = -30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l949_94990


namespace NUMINAMATH_CALUDE_area_perimeter_relation_l949_94973

/-- A stepped shape with n rows -/
structure SteppedShape (n : ℕ) where
  (n_pos : n > 0)
  (bottom_row : ℕ)
  (bottom_odd : Odd bottom_row)
  (bottom_eq : bottom_row = 2 * n - 1)
  (top_row : ℕ)
  (top_eq : top_row = 1)

/-- The area of a stepped shape -/
def area (shape : SteppedShape n) : ℕ := n ^ 2

/-- The perimeter of a stepped shape -/
def perimeter (shape : SteppedShape n) : ℕ := 6 * n - 2

/-- The main theorem relating area and perimeter of a stepped shape -/
theorem area_perimeter_relation (shape : SteppedShape n) :
  36 * (area shape) = (perimeter shape + 2) ^ 2 := by sorry

end NUMINAMATH_CALUDE_area_perimeter_relation_l949_94973


namespace NUMINAMATH_CALUDE_planes_parallel_l949_94920

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (lineParallel : Line → Line → Prop)

-- State the theorem
theorem planes_parallel (α β γ : Plane) (a b : Line) :
  (parallel α γ ∧ parallel β γ) ∧
  (perpendicular a α ∧ perpendicular b β ∧ lineParallel a b) →
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_l949_94920


namespace NUMINAMATH_CALUDE_quadratic_equations_count_l949_94979

variable (p : ℕ) [Fact (Nat.Prime p)]

/-- The number of quadratic equations with two distinct roots in p-arithmetic -/
def two_roots (p : ℕ) : ℕ := p * (p - 1) / 2

/-- The number of quadratic equations with exactly one root in p-arithmetic -/
def one_root (p : ℕ) : ℕ := p

/-- The number of quadratic equations with no roots in p-arithmetic -/
def no_roots (p : ℕ) : ℕ := p * (p - 1) / 2

/-- The total number of distinct quadratic equations in p-arithmetic -/
def total_equations (p : ℕ) : ℕ := p^2

theorem quadratic_equations_count (p : ℕ) [Fact (Nat.Prime p)] :
  two_roots p + one_root p + no_roots p = total_equations p :=
sorry

end NUMINAMATH_CALUDE_quadratic_equations_count_l949_94979


namespace NUMINAMATH_CALUDE_class_fund_total_l949_94972

theorem class_fund_total (ten_bills : ℕ) (twenty_bills : ℕ) : 
  ten_bills = 2 * twenty_bills →
  twenty_bills = 3 →
  ten_bills * 10 + twenty_bills * 20 = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_class_fund_total_l949_94972


namespace NUMINAMATH_CALUDE_complement_of_intersection_l949_94969

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 3, 4}
def B : Set Nat := {1, 2, 3}

theorem complement_of_intersection :
  (U \ (A ∩ B)) = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l949_94969


namespace NUMINAMATH_CALUDE_sequence_inequality_l949_94959

theorem sequence_inequality (a : ℕ → ℕ) (n N : ℕ) 
  (h1 : ∀ m k, a (m + k) ≤ a m + a k) 
  (h2 : N ≥ n) :
  a n + a N ≤ n * a 1 + (N / n) * a n :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l949_94959


namespace NUMINAMATH_CALUDE_max_linear_term_bound_l949_94993

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem max_linear_term_bound {a b c : ℝ} :
  (∀ x : ℝ, |x| ≤ 1 → |quadratic_function a b c x| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |a * x + b| ≤ 2) ∧
  (∃ a b : ℝ, ∃ x : ℝ, |x| ≤ 1 ∧ |a * x + b| = 2) :=
sorry

end NUMINAMATH_CALUDE_max_linear_term_bound_l949_94993


namespace NUMINAMATH_CALUDE_reciprocal_difference_sequence_l949_94988

theorem reciprocal_difference_sequence (a : ℕ → ℚ) :
  a 1 = 1/3 ∧
  (∀ n : ℕ, n > 1 → a n = 1 / (1 - a (n-1))) →
  a 2023 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_difference_sequence_l949_94988


namespace NUMINAMATH_CALUDE_system_solution_existence_l949_94940

theorem system_solution_existence (m : ℝ) : 
  (m ≠ 1) ↔ (∃ (x y : ℝ), y = m * x + 5 ∧ y = (3 * m - 2) * x + 7) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l949_94940


namespace NUMINAMATH_CALUDE_jeromes_contacts_l949_94926

theorem jeromes_contacts (classmates : ℕ) (total_contacts : ℕ) :
  classmates = 20 →
  total_contacts = 33 →
  3 = total_contacts - (classmates + classmates / 2) :=
by sorry

end NUMINAMATH_CALUDE_jeromes_contacts_l949_94926


namespace NUMINAMATH_CALUDE_min_unboxed_balls_tennis_balls_storage_l949_94981

theorem min_unboxed_balls (total_balls : ℕ) (big_box_size small_box_size : ℕ) : ℕ :=
  let min_unboxed := total_balls % big_box_size
  let remaining_after_big := total_balls % big_box_size
  let min_unboxed_small := remaining_after_big % small_box_size
  min min_unboxed min_unboxed_small

theorem tennis_balls_storage :
  min_unboxed_balls 104 25 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_unboxed_balls_tennis_balls_storage_l949_94981


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l949_94974

theorem sum_of_squares_of_roots : ∃ (r₁ r₂ r₃ r₄ : ℝ),
  (∀ x : ℝ, (x^2 + 4*x)^2 - 2016*(x^2 + 4*x) + 2017 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
  r₁^2 + r₂^2 + r₃^2 + r₄^2 = 4048 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l949_94974


namespace NUMINAMATH_CALUDE_average_age_of_five_students_l949_94951

/-- Proves that the average age of 5 students is 14 years given the conditions of the problem -/
theorem average_age_of_five_students
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_students_with_known_average : Nat)
  (average_age_known : ℝ)
  (age_of_twelfth_student : ℕ)
  (h1 : total_students = 16)
  (h2 : average_age_all = 16)
  (h3 : num_students_with_known_average = 9)
  (h4 : average_age_known = 16)
  (h5 : age_of_twelfth_student = 42)
  : (total_students * average_age_all - num_students_with_known_average * average_age_known - age_of_twelfth_student) / (total_students - num_students_with_known_average - 1) = 14 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_five_students_l949_94951


namespace NUMINAMATH_CALUDE_race_head_start_l949_94938

theorem race_head_start (course_length : ℝ) (speed_ratio : ℝ) (head_start : ℝ) : 
  course_length = 84 →
  speed_ratio = 2 →
  course_length / speed_ratio = (course_length - head_start) / 1 →
  head_start = 42 := by
sorry

end NUMINAMATH_CALUDE_race_head_start_l949_94938


namespace NUMINAMATH_CALUDE_election_winner_percentage_l949_94987

theorem election_winner_percentage 
  (total_votes : ℕ) 
  (majority : ℕ) 
  (winning_percentage : ℚ) :
  total_votes = 6500 →
  majority = 1300 →
  winning_percentage * total_votes = (total_votes + majority) / 2 →
  winning_percentage = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l949_94987


namespace NUMINAMATH_CALUDE_adults_fed_is_22_l949_94956

/-- Represents the resources and feeding capabilities of a community center -/
structure CommunityCenter where
  soup_cans : ℕ
  bread_loaves : ℕ
  adults_per_can : ℕ
  children_per_can : ℕ
  adults_per_loaf : ℕ
  children_per_loaf : ℕ

/-- Calculates the number of adults that can be fed with remaining resources -/
def adults_fed_after_children (cc : CommunityCenter) (children_to_feed : ℕ) : ℕ :=
  let cans_for_children := (children_to_feed + cc.children_per_can - 1) / cc.children_per_can
  let remaining_cans := cc.soup_cans - cans_for_children
  let adults_fed_by_cans := remaining_cans * cc.adults_per_can
  let adults_fed_by_bread := cc.bread_loaves * cc.adults_per_loaf
  adults_fed_by_cans + adults_fed_by_bread

/-- Theorem stating that 22 adults can be fed with remaining resources -/
theorem adults_fed_is_22 (cc : CommunityCenter) (h1 : cc.soup_cans = 8) (h2 : cc.bread_loaves = 2)
    (h3 : cc.adults_per_can = 4) (h4 : cc.children_per_can = 7) (h5 : cc.adults_per_loaf = 3)
    (h6 : cc.children_per_loaf = 4) :
    adults_fed_after_children cc 24 = 22 := by
  sorry

end NUMINAMATH_CALUDE_adults_fed_is_22_l949_94956


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l949_94902

theorem unique_solution_trigonometric_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan ((150 - x) * π / 180) = 
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 110 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l949_94902


namespace NUMINAMATH_CALUDE_greatest_integer_square_thrice_plus_81_l949_94953

theorem greatest_integer_square_thrice_plus_81 :
  ∀ x : ℤ, x^2 = 3*x + 81 → x ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_square_thrice_plus_81_l949_94953


namespace NUMINAMATH_CALUDE_min_value_xy_plus_four_over_xy_l949_94954

theorem min_value_xy_plus_four_over_xy (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) : 
  ∀ z w : ℝ, z > 0 → w > 0 → z + w = 2 → x * y + 4 / (x * y) ≤ z * w + 4 / (z * w) ∧ 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ a * b + 4 / (a * b) = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_plus_four_over_xy_l949_94954


namespace NUMINAMATH_CALUDE_sum_4_equivalence_l949_94932

-- Define the type for dice outcomes
def DiceOutcome := Fin 6

-- Define the type for a pair of dice outcomes
def DicePair := DiceOutcome × DiceOutcome

-- Define the sum of a pair of dice
def diceSum (pair : DicePair) : Nat :=
  pair.1.val + pair.2.val + 2

-- Define the event ξ = 4
def sumIs4 (pair : DicePair) : Prop :=
  diceSum pair = 4

-- Define the event where one die shows 3 and the other shows 1
def oneThreeOneOne (pair : DicePair) : Prop :=
  (pair.1.val = 2 ∧ pair.2.val = 0) ∨ (pair.1.val = 0 ∧ pair.2.val = 2)

-- Define the event where both dice show 2
def bothTwo (pair : DicePair) : Prop :=
  pair.1.val = 1 ∧ pair.2.val = 1

-- Theorem: ξ = 4 is equivalent to (one die shows 3 and the other shows 1) or (both dice show 2)
theorem sum_4_equivalence (pair : DicePair) :
  sumIs4 pair ↔ oneThreeOneOne pair ∨ bothTwo pair :=
by sorry

end NUMINAMATH_CALUDE_sum_4_equivalence_l949_94932


namespace NUMINAMATH_CALUDE_parabola_properties_l949_94936

/-- A function representing a parabola -/
def f (x : ℝ) : ℝ := -x^2 + 1

/-- The parabola opens downwards -/
def opens_downwards (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a < 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The parabola intersects the y-axis at (0,1) -/
def intersects_y_axis_at_0_1 (f : ℝ → ℝ) : Prop :=
  f 0 = 1

theorem parabola_properties :
  opens_downwards f ∧ intersects_y_axis_at_0_1 f :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l949_94936


namespace NUMINAMATH_CALUDE_commute_time_difference_l949_94971

theorem commute_time_difference (distance : Real) (walk_speed : Real) (train_speed : Real) (time_difference : Real) :
  distance = 1.5 ∧ 
  walk_speed = 3 ∧ 
  train_speed = 20 ∧ 
  time_difference = 25 →
  ∃ x : Real, 
    (distance / walk_speed) * 60 = (distance / train_speed) * 60 + x + time_difference ∧ 
    x = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_commute_time_difference_l949_94971


namespace NUMINAMATH_CALUDE_reunion_handshakes_l949_94975

theorem reunion_handshakes (n : ℕ) : n > 0 → (n * (n - 1)) / 2 = 45 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_reunion_handshakes_l949_94975


namespace NUMINAMATH_CALUDE_fred_grew_four_carrots_l949_94942

/-- The number of carrots Sally grew -/
def sally_carrots : ℕ := 6

/-- The total number of carrots grown by Sally and Fred -/
def total_carrots : ℕ := 10

/-- The number of carrots Fred grew -/
def fred_carrots : ℕ := total_carrots - sally_carrots

theorem fred_grew_four_carrots : fred_carrots = 4 := by
  sorry

end NUMINAMATH_CALUDE_fred_grew_four_carrots_l949_94942


namespace NUMINAMATH_CALUDE_cans_difference_l949_94913

/-- The number of cans Sarah collected yesterday -/
def sarah_yesterday : ℕ := 50

/-- The number of additional cans Lara collected compared to Sarah yesterday -/
def lara_extra_yesterday : ℕ := 30

/-- The number of cans Sarah collected today -/
def sarah_today : ℕ := 40

/-- The number of cans Lara collected today -/
def lara_today : ℕ := 70

/-- Theorem stating the difference in total cans collected between yesterday and today -/
theorem cans_difference : 
  (sarah_yesterday + (sarah_yesterday + lara_extra_yesterday)) - (sarah_today + lara_today) = 20 := by
  sorry

end NUMINAMATH_CALUDE_cans_difference_l949_94913


namespace NUMINAMATH_CALUDE_total_pets_l949_94915

theorem total_pets (taylor_pets : ℕ) (friends_with_double : ℕ) (friends_with_two : ℕ) : 
  taylor_pets = 4 → 
  friends_with_double = 3 → 
  friends_with_two = 2 → 
  taylor_pets + friends_with_double * (2 * taylor_pets) + friends_with_two * 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_l949_94915


namespace NUMINAMATH_CALUDE_min_value_of_expression_l949_94952

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + 3 * c)) + (b / (8 * c + 4 * a)) + (9 * c / (3 * a + 2 * b)) ≥ 47 / 48 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l949_94952


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l949_94927

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 5) / (x - 3) = 4 ∧ x = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l949_94927


namespace NUMINAMATH_CALUDE_willowton_vampires_l949_94998

/-- The number of vampires after a given number of nights -/
def vampires_after_nights (initial_population : ℕ) (initial_vampires : ℕ) (turned_per_night : ℕ) (nights : ℕ) : ℕ :=
  initial_vampires + nights * (initial_vampires * turned_per_night)

/-- Theorem stating the number of vampires after two nights in Willowton -/
theorem willowton_vampires :
  vampires_after_nights 300 2 5 2 = 72 := by
  sorry

#eval vampires_after_nights 300 2 5 2

end NUMINAMATH_CALUDE_willowton_vampires_l949_94998


namespace NUMINAMATH_CALUDE_problem_statement_l949_94931

def A (n r : ℕ) : ℕ := n.factorial / (n - r).factorial

def C (n r : ℕ) : ℕ := n.factorial / (r.factorial * (n - r).factorial)

theorem problem_statement : A 6 2 + C 6 4 = 45 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l949_94931


namespace NUMINAMATH_CALUDE_howard_earnings_l949_94908

/-- Calculates the money earned from washing windows --/
def money_earned (initial_amount current_amount : ℕ) : ℕ :=
  current_amount - initial_amount

theorem howard_earnings :
  let initial_amount : ℕ := 26
  let current_amount : ℕ := 52
  money_earned initial_amount current_amount = 26 := by
  sorry

end NUMINAMATH_CALUDE_howard_earnings_l949_94908


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_4_range_of_m_l949_94964

-- Define the function f
def f (x : ℝ) := |x - 2| + 2 * |x - 1|

-- Theorem for the solution set of f(x) > 4
theorem solution_set_f_greater_than_4 :
  {x : ℝ | f x > 4} = {x : ℝ | x < 0 ∨ x > 0} :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, f x > 2 * m^2 - 7 * m + 4} = {m : ℝ | 1/2 < m ∧ m < 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_4_range_of_m_l949_94964


namespace NUMINAMATH_CALUDE_intersection_point_theorem_l949_94946

/-- A parabola that intersects the coordinate axes at three distinct points -/
structure Parabola where
  p : ℝ
  q : ℝ
  distinct_intersections : ∃ (a b : ℝ), a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧ q ≠ 0 ∧ a^2 + p*a + q = 0 ∧ b^2 + p*b + q = 0

/-- The circle passing through the three intersection points of the parabola with the coordinate axes -/
def intersection_circle (par : Parabola) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ (a b : ℝ), a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧ par.q ≠ 0 ∧
            a^2 + par.p*a + par.q = 0 ∧ b^2 + par.p*b + par.q = 0 ∧
            (x^2 + y^2) * (a*b) + x * (par.q*(a+b)) + y * (par.q*par.p) + par.q^2 = 0}

/-- Theorem: All intersection circles pass through the point (0, 1) -/
theorem intersection_point_theorem (par : Parabola) :
  (0, 1) ∈ intersection_circle par :=
sorry

end NUMINAMATH_CALUDE_intersection_point_theorem_l949_94946


namespace NUMINAMATH_CALUDE_sum_equals_thirteen_thousand_two_hundred_l949_94917

theorem sum_equals_thirteen_thousand_two_hundred : 9773 + 3427 = 13200 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_thirteen_thousand_two_hundred_l949_94917


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l949_94921

theorem absolute_value_inequality_solution (x : ℝ) :
  |2*x - 7| < 3 ↔ 2 < x ∧ x < 5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l949_94921


namespace NUMINAMATH_CALUDE_complex_value_at_angle_l949_94925

/-- The value of 1-2i at an angle of 267.5° is equal to -√2/2 -/
theorem complex_value_at_angle : 
  let z : ℂ := 1 - 2*I
  let angle : Real := 267.5 * (π / 180)  -- Convert to radians
  Complex.abs z * Complex.exp (I * angle) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_value_at_angle_l949_94925


namespace NUMINAMATH_CALUDE_find_y_value_l949_94961

theorem find_y_value (x y z : ℝ) 
  (h1 : x^2 * y = z) 
  (h2 : x / y = 36)
  (h3 : Real.sqrt (x * y) = z)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) :
  y = 1 / 14.7 := by
sorry

end NUMINAMATH_CALUDE_find_y_value_l949_94961


namespace NUMINAMATH_CALUDE_trig_functions_and_expression_l949_94922

theorem trig_functions_and_expression (α : Real) (h : Real.tan α = -Real.sqrt 3) :
  (((Real.sin α = Real.sqrt 3 / 2) ∧ (Real.cos α = -1 / 2)) ∨
   ((Real.sin α = -Real.sqrt 3 / 2) ∧ (Real.cos α = 1 / 2))) ∧
  ((Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_trig_functions_and_expression_l949_94922


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l949_94910

theorem sqrt_product_simplification :
  Real.sqrt (12 + 1/9) * Real.sqrt 3 = Real.sqrt 327 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l949_94910


namespace NUMINAMATH_CALUDE_line_equations_specific_line_equations_l949_94924

/-- Definition of a line passing through two points -/
def Line (A B : ℝ × ℝ) :=
  {(x, y) : ℝ × ℝ | (y - A.2) * (B.1 - A.1) = (x - A.1) * (B.2 - A.2)}

theorem line_equations (A B : ℝ × ℝ) (h : A ≠ B) :
  let l := Line A B
  -- Two-point form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ (y - B.2) / (A.2 - B.2) = (x - B.1) / (A.1 - B.1) ∧
  -- Point-slope form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y - B.2 = ((A.2 - B.2) / (A.1 - B.1)) * (x - B.1) ∧
  -- Slope-intercept form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y = ((A.2 - B.2) / (A.1 - B.1)) * x + (B.2 - ((A.2 - B.2) / (A.1 - B.1)) * B.1) ∧
  -- Intercept form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ x / (-B.1 + (A.1 * B.2 - A.2 * B.1) / (A.2 - B.2)) + 
                             y / ((A.1 * B.2 - A.2 * B.1) / (A.1 - B.1)) = 1 :=
by
  sorry

-- Specific instance for points A(-2, 3) and B(4, -1)
theorem specific_line_equations :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (4, -1)
  let l := Line A B
  -- Two-point form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ (y + 1) / 4 = (x - 4) / (-6) ∧
  -- Point-slope form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y + 1 = -2/3 * (x - 4) ∧
  -- Slope-intercept form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ y = -2/3 * x + 5/3 ∧
  -- Intercept form
  ∀ (x y : ℝ), (x, y) ∈ l ↔ x / (5/2) + y / (5/3) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_line_equations_specific_line_equations_l949_94924


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l949_94960

theorem smallest_integer_with_given_remainders : ∃ x : ℕ+, 
  (x : ℕ) % 6 = 5 ∧ 
  (x : ℕ) % 7 = 6 ∧ 
  (x : ℕ) % 8 = 7 ∧ 
  ∀ y : ℕ+, 
    (y : ℕ) % 6 = 5 → 
    (y : ℕ) % 7 = 6 → 
    (y : ℕ) % 8 = 7 → 
    x ≤ y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l949_94960


namespace NUMINAMATH_CALUDE_exists_valid_assignment_l949_94906

/-- Represents a 7x7 square table with four corner squares deleted -/
def Table := Fin 7 → Fin 7 → Option ℤ

/-- Checks if a position is a valid square on the table -/
def isValidSquare (row col : Fin 7) : Prop :=
  ¬((row = 0 ∧ col = 0) ∨ (row = 0 ∧ col = 6) ∨ (row = 6 ∧ col = 0) ∨ (row = 6 ∧ col = 6))

/-- Represents a Greek cross on the table -/
structure GreekCross (t : Table) where
  center_row : Fin 7
  center_col : Fin 7
  valid : isValidSquare center_row center_col ∧
          isValidSquare center_row (center_col - 1) ∧
          isValidSquare center_row (center_col + 1) ∧
          isValidSquare (center_row - 1) center_col ∧
          isValidSquare (center_row + 1) center_col

/-- Calculates the sum of integers in a Greek cross -/
def sumGreekCross (t : Table) (cross : GreekCross t) : ℤ :=
  sorry

/-- Calculates the sum of all integers in the table -/
def sumTable (t : Table) : ℤ :=
  sorry

/-- Main theorem to prove -/
theorem exists_valid_assignment :
  ∃ (t : Table), (∀ (cross : GreekCross t), sumGreekCross t cross < 0) ∧ sumTable t > 0 :=
sorry

end NUMINAMATH_CALUDE_exists_valid_assignment_l949_94906


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_value_l949_94943

theorem mean_equality_implies_y_value : 
  let nums : List ℝ := [4, 6, 10, 14]
  let mean_nums := (nums.sum) / (nums.length : ℝ)
  mean_nums = (y + 18) / 2 → y = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_value_l949_94943


namespace NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l949_94919

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The perpendicular bisector of a line segment -/
def isPerpBisector (p q : Point) (a b c : ℝ) : Prop :=
  let midpoint : Point := ⟨(p.x + q.x) / 2, (p.y + q.y) / 2⟩
  a * midpoint.x + b * midpoint.y = c ∧
  (q.y - p.y) * a = (q.x - p.x) * b

/-- The theorem stating that b = 6 given the conditions -/
theorem perpendicular_bisector_b_value :
  let p : Point := ⟨0, 0⟩
  let q : Point := ⟨4, 8⟩
  ∀ b : ℝ, isPerpBisector p q 1 1 b → b = 6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l949_94919


namespace NUMINAMATH_CALUDE_other_factor_of_60n_l949_94912

theorem other_factor_of_60n (n : ℕ) (other : ℕ) : 
  (∃ k : ℕ, 60 * n = 4 * other * k) → 
  (∀ m : ℕ, m < n → ¬∃ j : ℕ, 60 * m = 4 * other * j) → 
  n = 8 → 
  other = 120 :=
by sorry

end NUMINAMATH_CALUDE_other_factor_of_60n_l949_94912


namespace NUMINAMATH_CALUDE_factorization_x4_minus_64_l949_94900

theorem factorization_x4_minus_64 (x : ℝ) : x^4 - 64 = (x^2 - 8) * (x^2 + 8) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_64_l949_94900


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_2_dividing_32_factorial_l949_94948

/-- The largest power of 2 that divides n! -/
def largestPowerOf2DividingFactorial (n : ℕ) : ℕ :=
  (n / 2) + (n / 4) + (n / 8) + (n / 16) + (n / 32)

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

theorem ones_digit_of_largest_power_of_2_dividing_32_factorial :
  onesDigit (2^(largestPowerOf2DividingFactorial 32)) = 8 :=
sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_2_dividing_32_factorial_l949_94948


namespace NUMINAMATH_CALUDE_jake_not_drop_coffee_l949_94976

/-- The probability of Jake tripping over his dog in the morning -/
def prob_trip : ℝ := 0.4

/-- The probability of Jake dropping his coffee when he trips -/
def prob_drop_given_trip : ℝ := 0.25

/-- The probability of Jake not dropping his coffee in the morning -/
def prob_not_drop : ℝ := 1 - prob_trip * prob_drop_given_trip

theorem jake_not_drop_coffee : prob_not_drop = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_jake_not_drop_coffee_l949_94976


namespace NUMINAMATH_CALUDE_frequency_distribution_correct_for_proportions_l949_94949

/-- Represents the score ranges for the math test. -/
inductive ScoreRange
  | AboveOrEqual120
  | Between90And120
  | Between75And90
  | Between60And75
  | Below60

/-- Represents statistical methods that can be applied to analyze test scores. -/
inductive StatisticalMethod
  | SampleAndEstimate
  | CalculateAverage
  | FrequencyDistribution
  | CalculateVariance

/-- The total number of students who took the test. -/
def totalStudents : ℕ := 800

/-- Determines if a given statistical method is correct for finding proportions of students in different score ranges. -/
def isCorrectMethodForProportions (method : StatisticalMethod) : Prop :=
  method = StatisticalMethod.FrequencyDistribution

/-- Theorem stating that frequency distribution is the correct method for finding proportions of students in different score ranges. -/
theorem frequency_distribution_correct_for_proportions :
  isCorrectMethodForProportions StatisticalMethod.FrequencyDistribution :=
sorry

end NUMINAMATH_CALUDE_frequency_distribution_correct_for_proportions_l949_94949


namespace NUMINAMATH_CALUDE_sophia_transactions_l949_94996

theorem sophia_transactions (mabel anthony cal jade sophia : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 19 →
  sophia = jade + jade / 2 →
  sophia = 128 :=
by sorry

end NUMINAMATH_CALUDE_sophia_transactions_l949_94996


namespace NUMINAMATH_CALUDE_necklace_cost_proof_l949_94934

/-- The cost of a single necklace -/
def necklace_cost : ℝ := 40000

/-- The total cost of the purchase -/
def total_cost : ℝ := 240000

/-- The number of necklaces purchased -/
def num_necklaces : ℕ := 3

theorem necklace_cost_proof :
  (num_necklaces : ℝ) * necklace_cost + 3 * necklace_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_necklace_cost_proof_l949_94934


namespace NUMINAMATH_CALUDE_factorization_proof_l949_94939

theorem factorization_proof (x y : ℝ) : 9*x^2*y - y = y*(3*x + 1)*(3*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l949_94939


namespace NUMINAMATH_CALUDE_parabola_symmetry_l949_94903

theorem parabola_symmetry (x₁ x₂ y₁ y₂ m : ℝ) : 
  y₁ = 2 * x₁^2 →
  y₂ = 2 * x₂^2 →
  y₁ + y₂ = x₁ + x₂ + 2*m →
  x₁ * x₂ = -1/2 →
  m = 3/2 := by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l949_94903


namespace NUMINAMATH_CALUDE_restaurant_seating_capacity_l949_94955

theorem restaurant_seating_capacity :
  ∀ (new_tables original_tables : ℕ),
    new_tables + original_tables = 40 →
    new_tables = original_tables + 12 →
    6 * new_tables + 4 * original_tables = 212 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_seating_capacity_l949_94955


namespace NUMINAMATH_CALUDE_square_inequality_l949_94957

theorem square_inequality (a b c A B C : ℝ) 
  (h1 : b^2 < a*c) 
  (h2 : a*C - 2*b*B + c*A = 0) : 
  B^2 ≥ A*C := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l949_94957


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l949_94947

/-- Represents the sides of an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- Checks if the given sides form a valid isosceles triangle -/
def is_valid_isosceles (t : IsoscelesTriangle) : Prop :=
  t.base > 0 ∧ t.leg > 0 ∧ t.leg + t.leg > t.base

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.base + 2 * t.leg

theorem isosceles_triangle_sides (p : ℝ) (s : ℝ) :
  p = 26 ∧ s = 11 →
  ∃ (t1 t2 : IsoscelesTriangle),
    (perimeter t1 = p ∧ (t1.base = s ∨ t1.leg = s) ∧ is_valid_isosceles t1) ∧
    (perimeter t2 = p ∧ (t2.base = s ∨ t2.leg = s) ∧ is_valid_isosceles t2) ∧
    ((t1.base = 11 ∧ t1.leg = 7.5) ∨ (t1.leg = 11 ∧ t1.base = 4)) ∧
    ((t2.base = 11 ∧ t2.leg = 7.5) ∨ (t2.leg = 11 ∧ t2.base = 4)) ∧
    t1 ≠ t2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_l949_94947


namespace NUMINAMATH_CALUDE_harmonic_mean_inequality_l949_94968

theorem harmonic_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_inequality_l949_94968


namespace NUMINAMATH_CALUDE_inequality_solution_count_l949_94962

theorem inequality_solution_count : 
  ∃! (s : Finset ℤ), 
    (∀ n : ℤ, n ∈ s ↔ Real.sqrt (3*n - 1) ≤ Real.sqrt (5*n - 7) ∧ 
                       Real.sqrt (5*n - 7) < Real.sqrt (3*n + 8)) ∧ 
    s.card = 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_count_l949_94962


namespace NUMINAMATH_CALUDE_complete_residue_system_product_l949_94907

theorem complete_residue_system_product (m n : ℕ) (a : Fin m → ℤ) (b : Fin n → ℤ) :
  (∀ k : Fin (m * n), ∃ i : Fin m, ∃ j : Fin n, (a i * b j) % (m * n) = k) →
  ((∀ k : Fin m, ∃ i : Fin m, a i % m = k) ∧
   (∀ k : Fin n, ∃ j : Fin n, b j % n = k)) :=
by sorry

end NUMINAMATH_CALUDE_complete_residue_system_product_l949_94907


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l949_94985

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 6 ∧ x₂ = -2) ∧ 
  (x₁^2 - 4*x₁ - 12 = 0) ∧ 
  (x₂^2 - 4*x₂ - 12 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l949_94985


namespace NUMINAMATH_CALUDE_shelves_needed_l949_94992

/-- Given a total of 14 books, with 2 taken by a librarian and 3 books fitting on each shelf,
    prove that 4 shelves are needed to store the remaining books. -/
theorem shelves_needed (total_books : ℕ) (taken_books : ℕ) (books_per_shelf : ℕ) :
  total_books = 14 →
  taken_books = 2 →
  books_per_shelf = 3 →
  ((total_books - taken_books) / books_per_shelf : ℕ) = 4 := by
  sorry

#check shelves_needed

end NUMINAMATH_CALUDE_shelves_needed_l949_94992
