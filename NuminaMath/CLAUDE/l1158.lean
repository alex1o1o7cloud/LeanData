import Mathlib

namespace negative_fractions_comparison_l1158_115852

theorem negative_fractions_comparison : -3/4 < -2/3 := by sorry

end negative_fractions_comparison_l1158_115852


namespace inverse_of_matrix_A_l1158_115875

theorem inverse_of_matrix_A (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A * !![1, 2; 0, 6] = !![(-1), (-2); 0, 3] →
  A⁻¹ = !![(-1), 0; 0, 2] := by sorry

end inverse_of_matrix_A_l1158_115875


namespace chord_tangent_angle_l1158_115873

-- Define the circle and chord
def Circle : Type := Unit
def Chord (c : Circle) : Type := Unit

-- Define the ratio of arc division
def arc_ratio (c : Circle) (ch : Chord c) : ℚ × ℚ := (11, 16)

-- Define the angle between tangents
def angle_between_tangents (c : Circle) (ch : Chord c) : ℚ := 100 / 3

-- Theorem statement
theorem chord_tangent_angle (c : Circle) (ch : Chord c) :
  arc_ratio c ch = (11, 16) →
  angle_between_tangents c ch = 100 / 3 :=
by
  sorry

end chord_tangent_angle_l1158_115873


namespace smallest_divisible_by_five_million_l1158_115838

def geometric_sequence (a₁ : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a₁ * r^(n - 1)

def is_divisible_by (a b : ℕ) : Prop :=
  ∃ k : ℕ, a = b * k

theorem smallest_divisible_by_five_million :
  let a₁ := 2
  let a₂ := 70
  let r := a₂ / a₁
  ∀ n : ℕ, n > 0 →
    (is_divisible_by (geometric_sequence a₁ r n) 5000000 ∧
     ∀ m : ℕ, 0 < m → m < n →
       ¬ is_divisible_by (geometric_sequence a₁ r m) 5000000) →
    n = 8 :=
sorry

end smallest_divisible_by_five_million_l1158_115838


namespace max_value_of_exponential_difference_l1158_115822

theorem max_value_of_exponential_difference : 
  ∃ (M : ℝ), M = 1/4 ∧ ∀ (x : ℝ), 2^x - 16^x ≤ M :=
sorry

end max_value_of_exponential_difference_l1158_115822


namespace orange_juice_fraction_is_three_tenths_l1158_115882

/-- Represents the capacity and fill level of a pitcher -/
structure Pitcher where
  capacity : ℚ
  fillLevel : ℚ

/-- Calculates the fraction of orange juice in the mixture -/
def orangeJuiceFraction (pitchers : List Pitcher) : ℚ :=
  let totalJuice := pitchers.foldl (fun acc p => acc + p.capacity * p.fillLevel) 0
  let totalVolume := pitchers.foldl (fun acc p => acc + p.capacity) 0
  totalJuice / totalVolume

/-- Theorem stating that the fraction of orange juice in the mixture is 3/10 -/
theorem orange_juice_fraction_is_three_tenths :
  let pitchers := [
    Pitcher.mk 500 (1/5),
    Pitcher.mk 700 (3/7),
    Pitcher.mk 800 (1/4)
  ]
  orangeJuiceFraction pitchers = 3/10 := by
  sorry

end orange_juice_fraction_is_three_tenths_l1158_115882


namespace probability_of_q_section_l1158_115853

/- Define the spinner -/
def spinner_sections : ℕ := 6
def q_sections : ℕ := 2

/- Define the probability function -/
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

/- Theorem statement -/
theorem probability_of_q_section :
  probability q_sections spinner_sections = 2 / 6 := by
  sorry

end probability_of_q_section_l1158_115853


namespace rectangular_field_area_l1158_115825

/-- A rectangular field with width one-third of its length and perimeter 72 meters has an area of 243 square meters. -/
theorem rectangular_field_area (w l : ℝ) (h1 : w > 0) (h2 : l > 0) : 
  w = l / 3 → 2 * (w + l) = 72 → w * l = 243 := by
  sorry

end rectangular_field_area_l1158_115825


namespace michaels_brother_final_money_l1158_115849

/-- Given the initial conditions of Michael and his brother's money, and their subsequent actions,
    this theorem proves the final amount of money Michael's brother has. -/
theorem michaels_brother_final_money (michael_initial : ℕ) (brother_initial : ℕ) 
    (candy_cost : ℕ) (h1 : michael_initial = 42) (h2 : brother_initial = 17) 
    (h3 : candy_cost = 3) : 
    brother_initial + michael_initial / 2 - candy_cost = 35 := by
  sorry

end michaels_brother_final_money_l1158_115849


namespace pupils_in_singing_only_l1158_115874

/-- Given a class with pupils in debate and singing activities, calculate the number of pupils in singing only. -/
theorem pupils_in_singing_only
  (total : ℕ)
  (debate_only : ℕ)
  (both : ℕ)
  (h_total : total = 55)
  (h_debate_only : debate_only = 10)
  (h_both : both = 17) :
  total - debate_only - both = 45 :=
by sorry

end pupils_in_singing_only_l1158_115874


namespace function_zero_in_interval_l1158_115827

/-- The function f(x) = 2ax^2 + 2x - 3 - a has a zero in the interval [-1, 1] 
    if and only if a ≤ (-3 - √7)/2 or a ≥ 1 -/
theorem function_zero_in_interval (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * a * x^2 + 2 * x - 3 - a = 0) ↔ 
  (a ≤ (-3 - Real.sqrt 7) / 2 ∨ a ≥ 1) :=
sorry

end function_zero_in_interval_l1158_115827


namespace min_value_sum_squares_l1158_115846

theorem min_value_sum_squares (a b s : ℝ) (h : 2 * a + 2 * b = s) :
  ∃ (min : ℝ), min = s^2 / 2 ∧ ∀ (x y : ℝ), 2 * x + 2 * y = s → 2 * x^2 + 2 * y^2 ≥ min :=
sorry

end min_value_sum_squares_l1158_115846


namespace max_money_is_zero_l1158_115802

/-- Represents the state of the stone piles and A's money --/
structure GameState where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  money : ℤ

/-- Represents a move from one pile to another --/
inductive Move
  | one_to_two
  | one_to_three
  | two_to_one
  | two_to_three
  | three_to_one
  | three_to_two

/-- Applies a move to the current game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.one_to_two => 
      { pile1 := state.pile1 - 1, 
        pile2 := state.pile2 + 1, 
        pile3 := state.pile3,
        money := state.money + (state.pile2 - state.pile1 + 1) }
  | Move.one_to_three => 
      { pile1 := state.pile1 - 1, 
        pile2 := state.pile2, 
        pile3 := state.pile3 + 1,
        money := state.money + (state.pile3 - state.pile1 + 1) }
  | Move.two_to_one => 
      { pile1 := state.pile1 + 1, 
        pile2 := state.pile2 - 1, 
        pile3 := state.pile3,
        money := state.money + (state.pile1 - state.pile2 + 1) }
  | Move.two_to_three => 
      { pile1 := state.pile1, 
        pile2 := state.pile2 - 1, 
        pile3 := state.pile3 + 1,
        money := state.money + (state.pile3 - state.pile2 + 1) }
  | Move.three_to_one => 
      { pile1 := state.pile1 + 1, 
        pile2 := state.pile2, 
        pile3 := state.pile3 - 1,
        money := state.money + (state.pile1 - state.pile3 + 1) }
  | Move.three_to_two => 
      { pile1 := state.pile1, 
        pile2 := state.pile2 + 1, 
        pile3 := state.pile3 - 1,
        money := state.money + (state.pile2 - state.pile3 + 1) }

/-- Theorem: The maximum amount of money A can have when all stones return to their initial positions is 0 --/
theorem max_money_is_zero (initial : GameState) (moves : List Move) :
  (moves.foldl applyMove initial).pile1 = initial.pile1 ∧
  (moves.foldl applyMove initial).pile2 = initial.pile2 ∧
  (moves.foldl applyMove initial).pile3 = initial.pile3 →
  (moves.foldl applyMove initial).money ≤ 0 :=
sorry

end max_money_is_zero_l1158_115802


namespace picnic_task_division_l1158_115855

theorem picnic_task_division (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 3) :
  Nat.choose n k = 20 := by
  sorry

end picnic_task_division_l1158_115855


namespace sum_reciprocal_squared_bound_l1158_115870

theorem sum_reciprocal_squared_bound (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) 
  (h_sum : x₁ + x₂ + x₃ = 1) : 
  (1 / (1 + x₁^2)) + (1 / (1 + x₂^2)) + (1 / (1 + x₃^2)) ≤ 27/10 := by
  sorry

end sum_reciprocal_squared_bound_l1158_115870


namespace book_arrangement_count_l1158_115835

/-- The number of ways to arrange books of different languages on a shelf --/
def arrange_books (total : ℕ) (italian : ℕ) (german : ℕ) (french : ℕ) : ℕ :=
  Nat.factorial 3 * Nat.factorial italian * Nat.factorial german * Nat.factorial french

/-- Theorem stating the number of arrangements for the given book problem --/
theorem book_arrangement_count :
  arrange_books 11 3 3 5 = 25920 := by
  sorry

end book_arrangement_count_l1158_115835


namespace integer_solutions_count_l1158_115860

/-- The number of distinct integer values of a for which x^2 + ax + 9a = 0 has integer solutions for x -/
theorem integer_solutions_count : 
  (∃ (S : Finset ℤ), (∀ a : ℤ, (∃ x : ℤ, x^2 + a*x + 9*a = 0) ↔ a ∈ S) ∧ Finset.card S = 5) :=
by sorry

end integer_solutions_count_l1158_115860


namespace fruit_drink_volume_l1158_115899

/-- Represents a fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_ounces : ℝ

/-- Theorem stating the total volume of the fruit drink -/
theorem fruit_drink_volume (drink : FruitDrink)
  (h1 : drink.orange_percent = 0.25)
  (h2 : drink.watermelon_percent = 0.40)
  (h3 : drink.grape_ounces = 70)
  (h4 : drink.orange_percent + drink.watermelon_percent + drink.grape_ounces / drink.total = 1) :
  drink.total = 200 := by
  sorry

end fruit_drink_volume_l1158_115899


namespace max_sum_of_factors_max_sum_is_884_l1158_115824

theorem max_sum_of_factors (a b : ℕ+) : 
  a * b = 1764 → ∀ x y : ℕ+, x * y = 1764 → a + b ≥ x + y :=
by sorry

theorem max_sum_is_884 : 
  ∃ a b : ℕ+, a * b = 1764 ∧ a + b = 884 ∧ 
  (∀ x y : ℕ+, x * y = 1764 → x + y ≤ 884) :=
by sorry

end max_sum_of_factors_max_sum_is_884_l1158_115824


namespace integral_exp_sin_l1158_115872

open Real

theorem integral_exp_sin (α β : ℝ) :
  deriv (fun x => (exp (α * x) * (α * sin (β * x) - β * cos (β * x))) / (α^2 + β^2)) =
  fun x => exp (α * x) * sin (β * x) := by
sorry

end integral_exp_sin_l1158_115872


namespace odd_function_implies_a_equals_one_l1158_115862

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = ax^3 + (a-1)x^2 + x -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + (a - 1) * x^2 + x

theorem odd_function_implies_a_equals_one :
  ∀ a : ℝ, IsOdd (f a) → a = 1 := by
  sorry

end odd_function_implies_a_equals_one_l1158_115862


namespace mean_of_xyz_l1158_115863

theorem mean_of_xyz (original_mean : ℝ) (new_mean : ℝ) (x y z : ℝ) : 
  original_mean = 40 →
  new_mean = 50 →
  z = x + 10 →
  (12 * original_mean + x + y + z) / 15 = new_mean →
  (x + y + z) / 3 = 90 := by
sorry

end mean_of_xyz_l1158_115863


namespace ben_dogs_difference_l1158_115880

/-- The number of dogs Teddy has -/
def teddy_dogs : ℕ := 7

/-- The number of cats Teddy has -/
def teddy_cats : ℕ := 8

/-- The number of cats Dave has -/
def dave_cats : ℕ := teddy_cats + 13

/-- The number of dogs Dave has -/
def dave_dogs : ℕ := teddy_dogs - 5

/-- The total number of pets all three have -/
def total_pets : ℕ := 54

/-- The number of dogs Ben has -/
def ben_dogs : ℕ := total_pets - (teddy_dogs + teddy_cats + dave_dogs + dave_cats)

theorem ben_dogs_difference : ben_dogs - teddy_dogs = 9 := by
  sorry

end ben_dogs_difference_l1158_115880


namespace unit_vector_AB_l1158_115861

def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

theorem unit_vector_AB : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let magnitude := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let unit_vector := (AB.1 / magnitude, AB.2 / magnitude)
  unit_vector = (3/5, -4/5) :=
by sorry

end unit_vector_AB_l1158_115861


namespace log_equation_equivalence_l1158_115891

-- Define the logarithm function with base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_equivalence (x : ℝ) (h : x > 0) :
  lg x ^ 2 + lg (x ^ 2) = 0 ↔ lg x ^ 2 + 2 * lg x = 0 :=
by sorry

end log_equation_equivalence_l1158_115891


namespace four_point_partition_l1158_115866

/-- A point in a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A straight line in a plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a line --/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- A set of four points in a plane --/
def FourPoints := Fin 4 → Point

/-- A partition of four points into two non-empty subsets --/
structure Partition (pts : FourPoints) where
  set1 : Set (Fin 4)
  set2 : Set (Fin 4)
  partition : set1 ∪ set2 = Set.univ
  nonempty1 : set1.Nonempty
  nonempty2 : set2.Nonempty

/-- Check if a line separates two sets of points --/
def separates (l : Line) (pts : FourPoints) (p : Partition pts) : Prop :=
  (∀ i ∈ p.set1, (pts i).onLine l) ∧ (∀ i ∈ p.set2, ¬(pts i).onLine l) ∨
  (∀ i ∈ p.set1, ¬(pts i).onLine l) ∧ (∀ i ∈ p.set2, (pts i).onLine l)

/-- The main theorem --/
theorem four_point_partition (pts : FourPoints) :
  ∃ p : Partition pts, ∀ l : Line, ¬separates l pts p := by
  sorry

end four_point_partition_l1158_115866


namespace exists_real_for_special_sequence_l1158_115848

/-- A sequence of non-negative integers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, n ≤ 1999 → a n ≥ 0) ∧
  (∀ i j, i + j ≤ 1999 → a i + a j ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1)

/-- The main theorem -/
theorem exists_real_for_special_sequence (a : ℕ → ℕ) (h : SpecialSequence a) :
  ∃ x : ℝ, ∀ n : ℕ, n ≤ 1999 → a n = ⌊n * x⌋ := by
  sorry

end exists_real_for_special_sequence_l1158_115848


namespace soda_price_ratio_l1158_115815

/-- The ratio of unit prices between two soda brands -/
theorem soda_price_ratio 
  (v : ℝ) -- Volume of Brand Y soda
  (p : ℝ) -- Price of Brand Y soda
  (h_v_pos : v > 0) -- Assumption that volume is positive
  (h_p_pos : p > 0) -- Assumption that price is positive
  : (0.85 * p) / (1.35 * v) / (p / v) = 17 / 27 := by
  sorry

end soda_price_ratio_l1158_115815


namespace average_age_decrease_l1158_115830

theorem average_age_decrease (initial_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (total_students : ℕ) : 
  initial_avg = 48 →
  new_students = 120 →
  new_avg = 32 →
  total_students = 160 →
  let original_students := total_students - new_students
  let total_age := initial_avg * original_students + new_avg * new_students
  let new_avg_age := total_age / total_students
  initial_avg - new_avg_age = 12 := by
sorry

end average_age_decrease_l1158_115830


namespace union_of_A_and_B_l1158_115869

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Theorem statement
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -2 < x ∧ x < 2} := by
  sorry

end union_of_A_and_B_l1158_115869


namespace average_monthly_balance_l1158_115823

def initial_balance : ℝ := 120
def february_change : ℝ := 80
def march_change : ℝ := -50
def april_change : ℝ := 70
def may_change : ℝ := 0
def june_change : ℝ := 100
def num_months : ℕ := 6

def monthly_balances : List ℝ := [
  initial_balance,
  initial_balance + february_change,
  initial_balance + february_change + march_change,
  initial_balance + february_change + march_change + april_change,
  initial_balance + february_change + march_change + april_change + may_change,
  initial_balance + february_change + march_change + april_change + may_change + june_change
]

theorem average_monthly_balance :
  (monthly_balances.sum / num_months) = 205 := by sorry

end average_monthly_balance_l1158_115823


namespace f_properties_l1158_115892

/-- The function f(x) = -x³ + ax² + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

/-- The function g(x) = f(x) - ax² + 3 -/
def g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x - a*x^2 + 3

/-- The derivative of f(x) -/
def f_derivative (a b : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x + b

theorem f_properties (a b c : ℝ) :
  (f_derivative a b 1 = -3) ∧  -- Tangent line condition
  (f a b c 1 = -2) ∧          -- Point P(1, f(1)) condition
  (∀ x, g a b c x = -g a b c (-x)) →  -- g(x) is an odd function
  (∃ a' b' c', 
    (∀ x, f a' b' c' x = -x^3 - 2*x^2 + 4*x - 3) ∧
    (∀ x, f a' b' c' x ≥ -11) ∧
    (f a' b' c' (-2) = -11) ∧
    (∀ x, f a' b' c' x ≤ -41/27) ∧
    (f a' b' c' (2/3) = -41/27)) := by sorry

end f_properties_l1158_115892


namespace die_roll_probabilities_l1158_115897

-- Define the type for a single die roll
def DieRoll : Type := Fin 6

-- Define the sample space for two die rolls
def SampleSpace : Type := DieRoll × DieRoll

-- Define the probability measure
noncomputable def prob : Set SampleSpace → ℝ := sorry

-- Define the event "sum is 5"
def sum_is_5 (roll : SampleSpace) : Prop :=
  roll.1.val + roll.2.val + 2 = 5

-- Define the event "at least one roll is odd"
def at_least_one_odd (roll : SampleSpace) : Prop :=
  roll.1.val % 2 = 0 ∨ roll.2.val % 2 = 0

-- State the theorem
theorem die_roll_probabilities :
  (prob {roll : SampleSpace | sum_is_5 roll} = 1/9) ∧
  (prob {roll : SampleSpace | at_least_one_odd roll} = 3/4) := by sorry

end die_roll_probabilities_l1158_115897


namespace circle_area_with_radius_four_l1158_115814

theorem circle_area_with_radius_four (π : ℝ) : 
  let r : ℝ := 4
  let area := π * r^2
  area = 16 * π := by sorry

end circle_area_with_radius_four_l1158_115814


namespace cooking_probability_l1158_115842

-- Define a finite set of courses
def Courses : Type := Fin 4

-- Define a probability measure on the set of courses
def prob : Courses → ℚ := λ _ => 1 / 4

-- Theorem statement
theorem cooking_probability :
  ∀ (c : Courses), prob c = 1 / 4 :=
by sorry

end cooking_probability_l1158_115842


namespace fraction_mediant_l1158_115829

theorem fraction_mediant (r s u v : ℚ) (l m : ℕ+) 
  (h1 : 0 < r) (h2 : 0 < s) (h3 : 0 < u) (h4 : 0 < v) 
  (h5 : s * u - r * v = 1) : 
  (∀ x, r / u < x ∧ x < s / v → 
    ∃ l m : ℕ+, x = (l * r + m * s) / (l * u + m * v)) ∧
  (r / u < (l * r + m * s) / (l * u + m * v) ∧ 
   (l * r + m * s) / (l * u + m * v) < s / v) :=
sorry

end fraction_mediant_l1158_115829


namespace initial_roses_count_l1158_115859

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of roses added to the vase -/
def added_roses : ℕ := 13

/-- The total number of roses in the vase after adding -/
def total_roses : ℕ := 20

/-- Theorem stating that the initial number of roses is 7 -/
theorem initial_roses_count : initial_roses = 7 := by
  sorry

end initial_roses_count_l1158_115859


namespace quadratic_real_roots_condition_l1158_115804

/-- For a quadratic equation ax^2 + 2x + 1 = 0 to have real roots, 
    a must satisfy: a ≤ 1 and a ≠ 0 -/
theorem quadratic_real_roots_condition (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) ↔ (a ≤ 1 ∧ a ≠ 0) := by
  sorry

end quadratic_real_roots_condition_l1158_115804


namespace stickers_distribution_l1158_115876

/-- Calculates the number of stickers each of the other students received -/
def stickers_per_other_student (total_stickers : ℕ) (friends : ℕ) (stickers_per_friend : ℕ) 
  (leftover_stickers : ℕ) (total_students : ℕ) : ℕ :=
  let stickers_given_to_friends := friends * stickers_per_friend
  let total_stickers_given := total_stickers - leftover_stickers
  let stickers_for_others := total_stickers_given - stickers_given_to_friends
  let other_students := total_students - 1 - friends
  stickers_for_others / other_students

theorem stickers_distribution (total_stickers : ℕ) (friends : ℕ) (stickers_per_friend : ℕ) 
  (leftover_stickers : ℕ) (total_students : ℕ)
  (h1 : total_stickers = 50)
  (h2 : friends = 5)
  (h3 : stickers_per_friend = 4)
  (h4 : leftover_stickers = 8)
  (h5 : total_students = 17) :
  stickers_per_other_student total_stickers friends stickers_per_friend leftover_stickers total_students = 2 := by
  sorry

end stickers_distribution_l1158_115876


namespace sum_of_solutions_equation_l1158_115839

theorem sum_of_solutions_equation : ∃ (x₁ x₂ : ℝ), 
  (4 * x₁ + 3) * (3 * x₁ - 7) = 0 ∧
  (4 * x₂ + 3) * (3 * x₂ - 7) = 0 ∧
  x₁ + x₂ = 19 / 12 := by
  sorry

end sum_of_solutions_equation_l1158_115839


namespace factorization_equality_l1158_115817

theorem factorization_equality (a b : ℝ) : b^2 - a*b + a - b = (b - 1) * (b - a) := by
  sorry

end factorization_equality_l1158_115817


namespace quadratic_inequality_solution_l1158_115811

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, -1/2 < x ∧ x < 1/3 ↔ a * x^2 + b * x + 2 > 0) →
  a - b = -10 := by
sorry

end quadratic_inequality_solution_l1158_115811


namespace nine_women_eighteen_tea_l1158_115898

/-- The time (in minutes) it takes for a given number of women to drink a given amount of tea,
    given that 1.5 women drink 1.5 tea in 1.5 minutes. -/
def drinking_time (women : ℚ) (tea : ℚ) : ℚ :=
  1.5 * tea / women

/-- Theorem stating that if 1.5 women drink 1.5 tea in 1.5 minutes,
    then 9 women can drink 18 tea in 3 minutes. -/
theorem nine_women_eighteen_tea :
  drinking_time 9 18 = 3 := by
  sorry

end nine_women_eighteen_tea_l1158_115898


namespace modified_counting_game_45th_number_l1158_115845

/-- Represents the modified counting game sequence -/
def modifiedSequence (n : ℕ) : ℕ :=
  n + (n - 1) / 10

/-- The 45th number in the modified counting game is 54 -/
theorem modified_counting_game_45th_number : modifiedSequence 45 = 54 := by
  sorry

end modified_counting_game_45th_number_l1158_115845


namespace distance_between_points_l1158_115837

/-- The distance between two points given their net movements -/
theorem distance_between_points (south west : ℝ) (h : south = 30 ∧ west = 40) :
  Real.sqrt (south^2 + west^2) = 50 := by
  sorry

end distance_between_points_l1158_115837


namespace set_operations_and_inclusion_l1158_115890

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | 2*x + a ≥ 0}

-- Theorem statement
theorem set_operations_and_inclusion :
  (A ∩ B = {x | 2 ≤ x ∧ x < 3}) ∧
  (A ∪ B = {x | x ≥ -1}) ∧
  (∀ a : ℝ, B ⊆ C a → a > -4) :=
by sorry

end set_operations_and_inclusion_l1158_115890


namespace hiker_distance_l1158_115857

theorem hiker_distance (hours_day1 : ℝ) : 
  hours_day1 > 0 →
  3 * hours_day1 + 4 * (hours_day1 - 1) + 4 * hours_day1 = 62 →
  3 * hours_day1 = 18 :=
by sorry

end hiker_distance_l1158_115857


namespace marilyn_final_bottle_caps_l1158_115810

/-- Calculates the final number of bottle caps Marilyn has after a series of exchanges --/
def final_bottle_caps (initial : ℕ) (shared : ℕ) (received : ℕ) : ℕ :=
  let remaining := initial - shared + received
  remaining - remaining / 2

/-- Theorem stating that Marilyn ends up with 55 bottle caps --/
theorem marilyn_final_bottle_caps : 
  final_bottle_caps 165 78 23 = 55 := by
  sorry

end marilyn_final_bottle_caps_l1158_115810


namespace mango_seller_loss_percentage_l1158_115854

/-- Calculates the percentage of loss for a fruit seller selling mangoes -/
theorem mango_seller_loss_percentage
  (selling_price : ℝ)
  (profit_price : ℝ)
  (h1 : selling_price = 16)
  (h2 : profit_price = 21.818181818181817)
  (h3 : profit_price = 1.2 * (profit_price / 1.2)) :
  (((profit_price / 1.2) - selling_price) / (profit_price / 1.2)) * 100 = 12 :=
by sorry

end mango_seller_loss_percentage_l1158_115854


namespace pond_volume_calculation_l1158_115856

/-- The volume of a rectangular prism with given dimensions -/
def pond_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

/-- Theorem stating that the volume of the pond is 1200 cubic meters -/
theorem pond_volume_calculation :
  pond_volume 20 12 5 = 1200 := by
  sorry

end pond_volume_calculation_l1158_115856


namespace defective_product_probability_l1158_115831

/-- The probability of drawing a defective product on the second draw,
    given that the first draw was a defective product, when there are
    10 total products, 4 of which are defective, and 2 products are
    drawn successively without replacement. -/
theorem defective_product_probability :
  let total_products : ℕ := 10
  let defective_products : ℕ := 4
  let qualified_products : ℕ := total_products - defective_products
  let first_draw_defective_prob : ℚ := defective_products / total_products
  let second_draw_defective_prob : ℚ :=
    (defective_products - 1) / (total_products - 1)
  let conditional_prob : ℚ :=
    (first_draw_defective_prob * second_draw_defective_prob) / first_draw_defective_prob
  conditional_prob = 1 / 3 :=
by sorry

end defective_product_probability_l1158_115831


namespace negative_a_to_zero_power_l1158_115800

theorem negative_a_to_zero_power (a : ℝ) (h : a ≠ 0) : (-a)^0 = 1 := by
  sorry

end negative_a_to_zero_power_l1158_115800


namespace furniture_store_optimal_profit_l1158_115836

/-- Represents the furniture store's purchase and sales plan -/
structure FurnitureStore where
  a : ℝ  -- Original purchase price of dining table
  tableRetailPrice : ℝ := 270
  chairRetailPrice : ℝ := 70
  setPrice : ℝ := 500
  numTables : ℕ
  numChairs : ℕ

/-- Calculates the profit for the furniture store -/
def profit (store : FurnitureStore) : ℝ :=
  let numSets := store.numTables / 2
  let remainingTables := store.numTables - numSets
  let chairsInSets := numSets * 4
  let remainingChairs := store.numChairs - chairsInSets
  (store.setPrice - store.a - 4 * (store.a - 110)) * numSets +
  (store.tableRetailPrice - store.a) * remainingTables +
  (store.chairRetailPrice - (store.a - 110)) * remainingChairs

/-- The main theorem to be proved -/
theorem furniture_store_optimal_profit (store : FurnitureStore) :
  (600 / store.a = 160 / (store.a - 110)) →
  (store.numChairs = 5 * store.numTables + 20) →
  (store.numTables + store.numChairs ≤ 200) →
  (∃ (maxProfit : ℝ), 
    maxProfit = 7950 ∧ 
    store.a = 150 ∧ 
    store.numTables = 30 ∧ 
    store.numChairs = 170 ∧
    profit store = maxProfit ∧
    ∀ (otherStore : FurnitureStore), 
      (600 / otherStore.a = 160 / (otherStore.a - 110)) →
      (otherStore.numChairs = 5 * otherStore.numTables + 20) →
      (otherStore.numTables + otherStore.numChairs ≤ 200) →
      profit otherStore ≤ maxProfit) := by
  sorry

end furniture_store_optimal_profit_l1158_115836


namespace u_equivalence_l1158_115885

theorem u_equivalence (u : ℝ) : 
  u = 1 / (2 - Real.rpow 3 (1/3)) → 
  u = ((2 + Real.rpow 3 (1/3)) * (4 + Real.rpow 9 (1/3))) / 7 := by
sorry

end u_equivalence_l1158_115885


namespace exists_n_sum_digits_decreases_l1158_115886

-- Define the sum of digits function
def S (a : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_n_sum_digits_decreases :
  ∃ n : ℕ, n > 1000 ∧ S (2^n) > S (2^(n+1)) := by sorry

end exists_n_sum_digits_decreases_l1158_115886


namespace range_of_a_l1158_115850

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| - |x - 2| < a^2 + a + 1) →
  (a < -1 ∨ a > 0) :=
by sorry

end range_of_a_l1158_115850


namespace angle_C_measure_triangle_area_l1158_115828

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleC : ℝ

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a^2 - t.c^2 + t.b^2 = t.a * t.b

-- Theorem for part 1
theorem angle_C_measure (t : Triangle) (h : triangle_condition t) : 
  t.angleC = π / 3 := by sorry

-- Theorem for part 2
theorem triangle_area (t : Triangle) (h1 : triangle_condition t) (h2 : t.a = 3) (h3 : t.b = 3) :
  (1/2) * t.a * t.b * Real.sin t.angleC = 9 * Real.sqrt 3 / 4 := by sorry

end angle_C_measure_triangle_area_l1158_115828


namespace partition_sum_condition_l1158_115833

def sum_set (s : Finset Nat) : Nat := s.sum id

theorem partition_sum_condition (k : Nat) :
  (∃ (A B : Finset Nat), A ∩ B = ∅ ∧ A ∪ B = Finset.range k ∧ sum_set A = 2 * sum_set B) ↔
  (∃ m : Nat, m > 0 ∧ (k = 3 * m ∨ k = 3 * m - 1)) :=
by sorry

end partition_sum_condition_l1158_115833


namespace fruit_sales_problem_l1158_115879

/-- Fruit sales problem -/
theorem fruit_sales_problem 
  (purchase_price : ℝ)
  (price_10 : ℝ)
  (sales_10 : ℝ)
  (price_13 : ℝ)
  (profit_13 : ℝ)
  (h1 : purchase_price = 8)
  (h2 : price_10 = 10)
  (h3 : sales_10 = 300)
  (h4 : price_13 = 13)
  (h5 : profit_13 = 750)
  (y : ℝ → ℝ)
  (h6 : ∀ x > 0, ∃ k b : ℝ, y x = k * x + b) :
  (∃ k b : ℝ, ∀ x > 0, y x = k * x + b ∧ k = -50 ∧ b = 800) ∧
  (∃ max_price : ℝ, max_price = 12 ∧ 
    ∀ x > 0, (y x) * (x - purchase_price) ≤ (y max_price) * (max_price - purchase_price)) ∧
  (∃ max_profit : ℝ, max_profit = 800 ∧
    max_profit = (y 12) * (12 - purchase_price)) :=
by sorry

end fruit_sales_problem_l1158_115879


namespace partial_fraction_decomposition_l1158_115843

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ), ∀ (x : ℚ), x ≠ 3 → x ≠ 5 →
    (4 * x) / ((x - 5) * (x - 3)^2) = A / (x - 5) + B / (x - 3) + C / (x - 3)^2 ∧
    A = 5 ∧ B = -5 ∧ C = -6 :=
by sorry

end partial_fraction_decomposition_l1158_115843


namespace election_theorem_l1158_115864

def total_candidates : ℕ := 20
def past_officers : ℕ := 8
def positions_available : ℕ := 4

def elections_with_at_least_two_past_officers : ℕ :=
  Nat.choose past_officers 2 * Nat.choose (total_candidates - past_officers) 2 +
  Nat.choose past_officers 3 * Nat.choose (total_candidates - past_officers) 1 +
  Nat.choose past_officers 4 * Nat.choose (total_candidates - past_officers) 0

theorem election_theorem :
  elections_with_at_least_two_past_officers = 2590 :=
by sorry

end election_theorem_l1158_115864


namespace rectangle_hexagon_apothem_comparison_l1158_115826

theorem rectangle_hexagon_apothem_comparison :
  ∀ (w l : ℝ) (s : ℝ),
    w > 0 ∧ l > 0 ∧ s > 0 →
    l = 3 * w →
    w * l = 2 * (w + l) →
    3 * Real.sqrt 3 / 2 * s^2 = 6 * s →
    w / 2 = 2/3 * (s * Real.sqrt 3 / 2) :=
by sorry

end rectangle_hexagon_apothem_comparison_l1158_115826


namespace distinct_roots_equal_integer_roots_l1158_115803

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + (m + 3) * x + 2 * m

-- Part 1: Prove the equation always has two distinct real roots
theorem distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

-- Part 2: Prove the specific case has two equal integer roots
theorem equal_integer_roots : 
  ∃ x : ℤ, quadratic 2 (x : ℝ) = 0 ∧ x = -2 :=
sorry

end distinct_roots_equal_integer_roots_l1158_115803


namespace elaine_rent_percentage_l1158_115847

/-- Represents Elaine's financial situation over two years -/
structure ElaineFinances where
  last_year_earnings : ℝ
  last_year_rent_percentage : ℝ
  this_year_earnings_increase : ℝ
  this_year_rent_percentage : ℝ
  rent_increase_percentage : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem elaine_rent_percentage
  (e : ElaineFinances)
  (h1 : e.this_year_earnings_increase = 0.15)
  (h2 : e.this_year_rent_percentage = 0.30)
  (h3 : e.rent_increase_percentage = 3.45)
  : e.last_year_rent_percentage = 0.10 := by
  sorry

#check elaine_rent_percentage

end elaine_rent_percentage_l1158_115847


namespace dress_price_calculation_l1158_115894

def calculate_final_price (original_price : ℝ) (initial_discount : ℝ) (additional_discount : ℝ) (store_credit : ℝ) (sales_tax : ℝ) : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount)
  let price_after_additional_discount := price_after_initial_discount * (1 - additional_discount)
  let price_after_credit := price_after_additional_discount - store_credit
  let final_price := price_after_credit * (1 + sales_tax)
  final_price

theorem dress_price_calculation :
  calculate_final_price 50 0.3 0.2 10 0.075 = 19.35 := by
  sorry

end dress_price_calculation_l1158_115894


namespace contractor_absent_days_l1158_115881

/-- Represents the problem of calculating a contractor's absent days. -/
def ContractorProblem (total_days : ℕ) (daily_wage : ℚ) (daily_fine : ℚ) (total_amount : ℚ) : Prop :=
  ∃ (worked_days absent_days : ℕ),
    worked_days + absent_days = total_days ∧
    daily_wage * worked_days - daily_fine * absent_days = total_amount

/-- Theorem stating that given the problem conditions, the number of absent days is 10. -/
theorem contractor_absent_days :
  ContractorProblem 30 25 (15/2) 425 →
  ∃ (worked_days absent_days : ℕ),
    worked_days + absent_days = 30 ∧
    absent_days = 10 := by
  sorry

#check contractor_absent_days

end contractor_absent_days_l1158_115881


namespace betty_garden_total_l1158_115832

/-- Represents Betty's herb garden -/
structure HerbGarden where
  basil : ℕ
  oregano : ℕ

/-- The number of oregano plants is 2 more than twice the number of basil plants -/
def oregano_rule (garden : HerbGarden) : Prop :=
  garden.oregano = 2 + 2 * garden.basil

/-- Betty's garden has 5 basil plants -/
def betty_garden : HerbGarden :=
  { basil := 5, oregano := 2 + 2 * 5 }

/-- The total number of plants in the garden -/
def total_plants (garden : HerbGarden) : ℕ :=
  garden.basil + garden.oregano

theorem betty_garden_total : total_plants betty_garden = 17 := by
  sorry

end betty_garden_total_l1158_115832


namespace function_satisfying_inequality_is_constant_l1158_115889

/-- A function f: ℝ → ℝ satisfying f(x+y) ≤ f(x²+y) for all x, y ∈ ℝ is constant. -/
theorem function_satisfying_inequality_is_constant (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) ≤ f (x^2 + y)) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end function_satisfying_inequality_is_constant_l1158_115889


namespace nba_scheduling_impossibility_l1158_115808

theorem nba_scheduling_impossibility :
  ∀ (k : ℕ) (x y z : ℕ),
    k ≤ 30 ∧
    x + y + z = 1230 ∧
    82 * k = 2 * x + z →
    z ≠ (x + y + z) / 2 :=
by
  sorry

end nba_scheduling_impossibility_l1158_115808


namespace machine_output_l1158_115877

/-- The number of shirts an industrial machine can make in a minute. -/
def shirts_per_minute : ℕ := sorry

/-- The number of minutes the machine worked today. -/
def minutes_worked_today : ℕ := 12

/-- The total number of shirts made today. -/
def total_shirts_today : ℕ := 72

/-- Theorem stating that the machine can make 6 shirts per minute. -/
theorem machine_output : shirts_per_minute = 6 := by
  sorry

end machine_output_l1158_115877


namespace angle_triple_supplement_measure_l1158_115865

theorem angle_triple_supplement_measure : 
  ∃ (x : ℝ), x > 0 ∧ x < 180 ∧ x = 3 * (180 - x) ∧ x = 135 := by
  sorry

end angle_triple_supplement_measure_l1158_115865


namespace theater_sales_proof_l1158_115801

/-- Calculates the total ticket sales for a theater performance. -/
def theater_sales (adult_price child_price total_attendance children_attendance : ℕ) : ℕ :=
  let adults := total_attendance - children_attendance
  let adult_sales := adults * adult_price
  let child_sales := children_attendance * child_price
  adult_sales + child_sales

/-- Theorem stating that given the specific conditions, the theater collects $50 from ticket sales. -/
theorem theater_sales_proof :
  theater_sales 8 1 22 18 = 50 := by
  sorry

end theater_sales_proof_l1158_115801


namespace no_prime_sum_53_less_than_30_l1158_115858

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def primeSum53LessThan30 : Prop :=
  ¬∃ (p q : ℕ), isPrime p ∧ isPrime q ∧ p + q = 53 ∧ (p < 30 ∨ q < 30)

theorem no_prime_sum_53_less_than_30 : primeSum53LessThan30 := by
  sorry

end no_prime_sum_53_less_than_30_l1158_115858


namespace perpendicular_lines_slope_l1158_115841

theorem perpendicular_lines_slope (a : ℝ) : 
  (∃ x y : ℝ, y = a * x - 2) ∧ 
  (∃ x y : ℝ, y = 2 * x + 1) ∧ 
  (a * 2 = -1) → 
  a = -1/2 := by
sorry

end perpendicular_lines_slope_l1158_115841


namespace union_of_P_and_Q_l1158_115806

-- Define the sets P and Q
def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem union_of_P_and_Q : P ∪ Q = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end union_of_P_and_Q_l1158_115806


namespace b_2017_eq_1_l1158_115884

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Sequence of Fibonacci numbers modulo 3 -/
def b (n : ℕ) : ℕ := fib n % 3

/-- The sequence b has period 8 -/
axiom b_period (n : ℕ) : b (n + 8) = b n

theorem b_2017_eq_1 : b 2017 = 1 := by sorry

end b_2017_eq_1_l1158_115884


namespace fruit_cost_difference_l1158_115867

/-- Represents the cost and quantity of a fruit carton -/
structure FruitCarton where
  cost : ℚ  -- Cost in dollars
  quantity : ℚ  -- Quantity in ounces
  inv_mk : cost > 0 ∧ quantity > 0

/-- Calculates the number of cartons needed for a given amount of fruit -/
def cartonsNeeded (fruit : FruitCarton) (amount : ℚ) : ℚ :=
  amount / fruit.quantity

/-- Calculates the total cost for a given number of cartons -/
def totalCost (fruit : FruitCarton) (cartons : ℚ) : ℚ :=
  fruit.cost * cartons

/-- The main theorem to prove -/
theorem fruit_cost_difference 
  (blueberries : FruitCarton)
  (raspberries : FruitCarton)
  (batches : ℕ)
  (fruitPerBatch : ℚ)
  (h1 : blueberries.cost = 5)
  (h2 : blueberries.quantity = 6)
  (h3 : raspberries.cost = 3)
  (h4 : raspberries.quantity = 8)
  (h5 : batches = 4)
  (h6 : fruitPerBatch = 12) :
  totalCost blueberries (cartonsNeeded blueberries (batches * fruitPerBatch)) -
  totalCost raspberries (cartonsNeeded raspberries (batches * fruitPerBatch)) = 22 := by
  sorry

end fruit_cost_difference_l1158_115867


namespace mean_of_five_numbers_with_sum_three_quarters_l1158_115812

theorem mean_of_five_numbers_with_sum_three_quarters
  (a b c d e : ℝ) (h : a + b + c + d + e = 3/4) :
  (a + b + c + d + e) / 5 = 3/20 := by
  sorry

end mean_of_five_numbers_with_sum_three_quarters_l1158_115812


namespace only_proposition4_is_correct_l1158_115807

-- Define the propositions
def proposition1 : Prop := ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) - a n = 0) → (∃ r, ∀ n, a (n + 1) = r * a n)
def proposition2 : Prop := ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) = (1/2) * a n) → (∀ n, a (n + 1) < a n)
def proposition3 : Prop := ∀ a b c : ℝ, (b^2 = a * c) ↔ (∃ r, b = a * r ∧ c = b * r)
def proposition4 : Prop := ∀ a b c : ℝ, (2 * b = a + c) ↔ (∃ d, b = a + d ∧ c = b + d)

-- Theorem statement
theorem only_proposition4_is_correct :
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4 :=
sorry

end only_proposition4_is_correct_l1158_115807


namespace parallelepiped_ring_sum_exists_l1158_115819

/-- Represents a rectangular parallelepiped with dimensions a × b × c -/
structure Parallelepiped (a b c : ℕ) where
  dim_a : a > 0
  dim_b : b > 0
  dim_c : c > 0

/-- Represents an assignment of numbers to the faces of a parallelepiped -/
def FaceAssignment (a b c : ℕ) := Fin 6 → ℕ

/-- Calculates the sum of numbers in a 1-unit-wide ring around the parallelepiped -/
def ringSum (p : Parallelepiped 3 4 5) (assignment : FaceAssignment 3 4 5) : ℕ :=
  2 * (4 * assignment 0 + 5 * assignment 2 +
       3 * assignment 0 + 5 * assignment 4 +
       3 * assignment 2 + 4 * assignment 4)

/-- The main theorem stating that there exists an assignment satisfying the condition -/
theorem parallelepiped_ring_sum_exists :
  ∃ (assignment : FaceAssignment 3 4 5),
    ∀ (p : Parallelepiped 3 4 5), ringSum p assignment = 120 := by
  sorry

end parallelepiped_ring_sum_exists_l1158_115819


namespace line_circle_no_intersection_l1158_115871

/-- The range of m for which a line and circle have no intersection -/
theorem line_circle_no_intersection (m : ℝ) : 
  (∀ x y : ℝ, 3*x + 4*y + m ≠ 0 ∨ (x+1)^2 + (y-2)^2 ≠ 1) →
  m < -10 ∨ m > 0 :=
sorry

end line_circle_no_intersection_l1158_115871


namespace sector_max_area_angle_l1158_115834

/-- Given a sector with circumference 36, the radian measure of the central angle
    that maximizes the area of the sector is 2. -/
theorem sector_max_area_angle (r : ℝ) (l : ℝ) (α : ℝ) :
  2 * r + l = 36 →
  α = l / r →
  (∀ r' l' α', 2 * r' + l' = 36 → α' = l' / r' →
    r * l ≥ r' * l') →
  α = 2 := by
  sorry

end sector_max_area_angle_l1158_115834


namespace dollar_operation_result_l1158_115809

/-- Custom dollar operation -/
def dollar (a b c : ℝ) : ℝ := (a - b + c)^2

/-- Theorem statement -/
theorem dollar_operation_result (x z : ℝ) :
  dollar ((x + z)^2) ((z - x)^2) ((x - z)^2) = (x + z)^4 := by
  sorry

end dollar_operation_result_l1158_115809


namespace simplify_expression_1_simplify_expression_2_l1158_115851

-- First expression
theorem simplify_expression_1 (x y : ℝ) :
  4 * y^2 + 3 * x - 5 + 6 - 4 * x - 2 * y^2 = 2 * y^2 - x + 1 := by sorry

-- Second expression
theorem simplify_expression_2 (m n : ℝ) :
  3/2 * (m^2 - m*n) - 2 * (m*n + m^2) = -1/2 * m^2 - 7/2 * m*n := by sorry

end simplify_expression_1_simplify_expression_2_l1158_115851


namespace min_crooks_proof_l1158_115896

/-- Represents the total number of ministers -/
def total_ministers : ℕ := 100

/-- Represents the size of any subgroup of ministers that must contain at least one crook -/
def subgroup_size : ℕ := 10

/-- Represents the property that any subgroup of ministers contains at least one crook -/
def at_least_one_crook (num_crooks : ℕ) : Prop :=
  ∀ (subgroup : Finset ℕ), subgroup.card = subgroup_size → 
    (total_ministers - num_crooks < subgroup.card)

/-- The minimum number of crooks in the cabinet -/
def min_crooks : ℕ := total_ministers - (subgroup_size - 1)

theorem min_crooks_proof :
  (at_least_one_crook min_crooks) ∧ 
  (∀ k < min_crooks, ¬(at_least_one_crook k)) :=
sorry

end min_crooks_proof_l1158_115896


namespace common_solution_iff_y_eq_one_l1158_115821

/-- The first equation: x^2 + y^2 - 4 = 0 -/
def equation1 (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0

/-- The second equation: x^2 - 4y + y^2 = 0 -/
def equation2 (x y : ℝ) : Prop := x^2 - 4*y + y^2 = 0

/-- The theorem stating that the equations have common real solutions iff y = 1 -/
theorem common_solution_iff_y_eq_one :
  (∃ x : ℝ, equation1 x 1 ∧ equation2 x 1) ∧
  (∀ y : ℝ, y ≠ 1 → ¬∃ x : ℝ, equation1 x y ∧ equation2 x y) :=
sorry

end common_solution_iff_y_eq_one_l1158_115821


namespace max_min_values_l1158_115818

theorem max_min_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 2 * y = 10) :
  (Real.sqrt (3 * x) + Real.sqrt (2 * y) ≤ 2 * Real.sqrt 5) ∧
  (3 / x + 2 / y ≥ 5 / 2) :=
by sorry

end max_min_values_l1158_115818


namespace reflection_sum_coordinates_l1158_115888

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflect a point over the y-axis -/
def reflectOverYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

/-- Sum of coordinates of two points -/
def sumCoordinates (p1 p2 : Point2D) : ℝ :=
  p1.x + p1.y + p2.x + p2.y

theorem reflection_sum_coordinates (a : ℝ) :
  let C : Point2D := { x := a, y := 8 }
  let D : Point2D := reflectOverYAxis C
  sumCoordinates C D = 16 := by
  sorry

end reflection_sum_coordinates_l1158_115888


namespace linear_function_problem_l1158_115893

/-- A linear function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The inverse of f -/
def f_inv (x : ℝ) : ℝ := sorry

theorem linear_function_problem :
  (∀ x y : ℝ, ∃ a b : ℝ, f x = a * x + b) →  -- f is linear
  (∀ x : ℝ, f x = 4 * f_inv x + 8) →         -- f(x) = 4f^(-1)(x) + 8
  (f 1 = 5) →                                -- f(1) = 5
  (f 2 = 20 / 3) :=                          -- f(2) = 20/3
by sorry

end linear_function_problem_l1158_115893


namespace eight_b_value_l1158_115820

theorem eight_b_value (a b : ℚ) 
  (eq1 : 6 * a + 3 * b = 3) 
  (eq2 : b = 2 * a - 3) : 
  8 * b = -8 := by
sorry

end eight_b_value_l1158_115820


namespace digits_1498_to_1500_form_229_l1158_115816

/-- A function that generates the list of positive integers starting with 2 -/
def integerListStartingWith2 : ℕ → ℕ
| 0 => 2
| n + 1 => 
  let prev := integerListStartingWith2 n
  if prev < 10 then 20
  else if prev % 10 = 9 then prev + 11
  else prev + 1

/-- A function that returns the nth digit in the concatenated list -/
def nthDigitInList (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 1498th, 1499th, and 1500th digits form 229 -/
theorem digits_1498_to_1500_form_229 : 
  (nthDigitInList 1498) * 100 + (nthDigitInList 1499) * 10 + nthDigitInList 1500 = 229 := by sorry

end digits_1498_to_1500_form_229_l1158_115816


namespace complex_number_properties_l1158_115868

theorem complex_number_properties : ∃ (z : ℂ), 
  z = 2 / (1 - Complex.I) ∧ 
  Complex.abs z = Real.sqrt 2 ∧
  z^2 = 2 * Complex.I ∧
  z^2 - 2*z + 2 = 0 := by
  sorry

end complex_number_properties_l1158_115868


namespace decreasing_interval_of_f_l1158_115840

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 - x + 4

-- State the theorem
theorem decreasing_interval_of_f :
  ∀ x y : ℝ, x ≥ -1/2 → y > x → f y < f x :=
sorry

end decreasing_interval_of_f_l1158_115840


namespace ten_percent_of_n_l1158_115887

theorem ten_percent_of_n (n f : ℝ) (h : n - (1/4 * 2) - (1/3 * 3) - f * n = 27) : 
  (0.1 : ℝ) * n = (0.1 : ℝ) * (28.5 / (1 - f)) := by
sorry

end ten_percent_of_n_l1158_115887


namespace two_p_plus_q_l1158_115878

theorem two_p_plus_q (p q : ℚ) (h : p / q = 6 / 7) : 2 * p + q = (19 / 7) * q := by
  sorry

end two_p_plus_q_l1158_115878


namespace triangle_radii_inequality_l1158_115813

/-- For any triangle with circumradius R, inradius r, and exradii r_a, r_b, r_c,
    the inequality (r * r_a * r_b * r_c) / R^4 ≤ 27/16 holds. -/
theorem triangle_radii_inequality (R r r_a r_b r_c : ℝ) 
    (h_R : R > 0) 
    (h_r : r > 0) 
    (h_ra : r_a > 0) 
    (h_rb : r_b > 0) 
    (h_rc : r_c > 0) 
    (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
      R = (a * b * c) / (4 * (a + b + c) * (a + b - c) * (b + c - a) * (c + a - b))^(1/2) ∧
      r = (a + b - c) * (b + c - a) * (c + a - b) / (4 * (a + b + c)) ∧
      r_a = (b + c - a) / 2 ∧
      r_b = (c + a - b) / 2 ∧
      r_c = (a + b - c) / 2) :
  (r * r_a * r_b * r_c) / R^4 ≤ 27/16 := by
sorry

end triangle_radii_inequality_l1158_115813


namespace inscribed_equal_angles_is_regular_circumscribed_equal_sides_is_regular_l1158_115895

-- Define a polygon type
structure Polygon where
  vertices : List ℝ × ℝ
  sides : Nat
  is_odd : Odd sides

-- Define properties for inscribed and circumscribed polygons
def is_inscribed (p : Polygon) : Prop := sorry
def is_circumscribed (p : Polygon) : Prop := sorry

-- Define properties for equal angles and equal sides
def has_equal_angles (p : Polygon) : Prop := sorry
def has_equal_sides (p : Polygon) : Prop := sorry

-- Define what it means for a polygon to be regular
def is_regular (p : Polygon) : Prop := sorry

-- Theorem for part a
theorem inscribed_equal_angles_is_regular (p : Polygon) 
  (h_inscribed : is_inscribed p) (h_equal_angles : has_equal_angles p) : 
  is_regular p := by sorry

-- Theorem for part b
theorem circumscribed_equal_sides_is_regular (p : Polygon) 
  (h_circumscribed : is_circumscribed p) (h_equal_sides : has_equal_sides p) : 
  is_regular p := by sorry

end inscribed_equal_angles_is_regular_circumscribed_equal_sides_is_regular_l1158_115895


namespace red_rose_theatre_ticket_sales_l1158_115883

theorem red_rose_theatre_ticket_sales 
  (price_low : ℝ) 
  (price_high : ℝ) 
  (total_sales : ℝ) 
  (low_price_tickets : ℕ) 
  (h1 : price_low = 4.5)
  (h2 : price_high = 6)
  (h3 : total_sales = 1972.5)
  (h4 : low_price_tickets = 205) :
  ∃ (high_price_tickets : ℕ),
    (low_price_tickets : ℝ) * price_low + (high_price_tickets : ℝ) * price_high = total_sales ∧
    low_price_tickets + high_price_tickets = 380 :=
by sorry

end red_rose_theatre_ticket_sales_l1158_115883


namespace tangent_line_at_y_axis_l1158_115805

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x + 1)

theorem tangent_line_at_y_axis (x : ℝ) :
  let y_intercept := f 0
  let slope := (deriv f) 0
  (fun x => slope * x + y_intercept) = (fun x => 2 * Real.exp 1 * x + Real.exp 1) :=
by sorry

end tangent_line_at_y_axis_l1158_115805


namespace parabola_chord_perpendicular_bisector_l1158_115844

/-- The parabola y^2 = 8(x+2) with focus at (0, 0) -/
def parabola (x y : ℝ) : Prop := y^2 = 8*(x+2)

/-- The line y = x passing through (0, 0) -/
def line (x y : ℝ) : Prop := y = x

/-- The perpendicular bisector of a chord on the line y = x -/
def perp_bisector (x y : ℝ) : Prop := y = -x + 2*x

theorem parabola_chord_perpendicular_bisector :
  ∀ (x : ℝ),
  (∃ (y : ℝ), parabola x y ∧ line x y) →
  (∃ (P : ℝ × ℝ), P.1 = x ∧ P.2 = 0 ∧ perp_bisector P.1 P.2) →
  x = x := by sorry

end parabola_chord_perpendicular_bisector_l1158_115844
