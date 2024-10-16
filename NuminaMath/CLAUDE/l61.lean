import Mathlib

namespace NUMINAMATH_CALUDE_infinitely_many_common_terms_l61_6167

-- Define the arithmetic sequence
def a (n : ℕ) : ℤ := 3*n - 1

-- Define the geometric sequence
def b (n : ℕ) : ℕ := 2^n

-- State the properties of the sequences
axiom a2_eq_5 : a 2 = 5
axiom a8_eq_23 : a 8 = 23
axiom b1_eq_2 : b 1 = 2
axiom b_mul (s t : ℕ) : b (s + t) = b s * b t

-- Theorem statement
theorem infinitely_many_common_terms :
  ∀ m : ℕ, ∃ k : ℕ, k > m ∧ ∃ n : ℕ, b k = a n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_common_terms_l61_6167


namespace NUMINAMATH_CALUDE_opposing_team_score_l61_6179

theorem opposing_team_score (chucks_team_score : ℕ) (lead : ℕ) (opposing_team_score : ℕ) :
  chucks_team_score = 72 →
  lead = 17 →
  chucks_team_score = opposing_team_score + lead →
  opposing_team_score = 55 := by
sorry

end NUMINAMATH_CALUDE_opposing_team_score_l61_6179


namespace NUMINAMATH_CALUDE_work_completion_l61_6162

/-- Given that 36 men can complete a piece of work in 18 days, and a different number of men can
    complete the same work in 24 days, prove that the number of men in the second group is 27. -/
theorem work_completion (total_work : ℕ) (men_group1 men_group2 : ℕ) (days_group1 days_group2 : ℕ) :
  men_group1 = 36 →
  days_group1 = 18 →
  days_group2 = 24 →
  total_work = men_group1 * days_group1 →
  total_work = men_group2 * days_group2 →
  men_group2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l61_6162


namespace NUMINAMATH_CALUDE_distance_traveled_l61_6116

def initial_reading : ℝ := 212.3
def final_reading : ℝ := 372.0

theorem distance_traveled : final_reading - initial_reading = 159.7 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l61_6116


namespace NUMINAMATH_CALUDE_min_distance_ellipse_line_is_zero_l61_6114

/-- The minimum distance between a point on the ellipse x²/8 + y²/4 = 1 
    and the line x - √2 y - 4 = 0 is 0. -/
theorem min_distance_ellipse_line_is_zero :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}
  let line := {p : ℝ × ℝ | p.1 - Real.sqrt 2 * p.2 - 4 = 0}
  (∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ ellipse ∧ q ∈ line ∧ ‖p - q‖ = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_line_is_zero_l61_6114


namespace NUMINAMATH_CALUDE_train_length_l61_6138

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 32 → (speed * (5/18) * time) = 373.33 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l61_6138


namespace NUMINAMATH_CALUDE_square_difference_133_l61_6164

theorem square_difference_133 : 
  ∃ (a b c d : ℕ), 
    a * a - b * b = 133 ∧ 
    c * c - d * d = 133 ∧ 
    a > b ∧ c > d ∧ 
    (a ≠ c ∨ b ≠ d) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_133_l61_6164


namespace NUMINAMATH_CALUDE_koschei_puzzle_solvable_l61_6187

theorem koschei_puzzle_solvable (a b c : Nat) (h1 : a ∈ Finset.range 100) 
  (h2 : b ∈ Finset.range 100) (h3 : c ∈ Finset.range 100) :
  ∃! (x y z : Nat), x = 1 ∧ y = 100 ∧ z = 10000 ∧
    ∀ (S : Nat), S = a * x + b * y + c * z → 
      (S % 100 = a ∧ (S / 100) % 100 = b ∧ S / 10000 = c) :=
by sorry

end NUMINAMATH_CALUDE_koschei_puzzle_solvable_l61_6187


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l61_6141

def vector1 : Fin 3 → ℝ := ![3, 4, 5]
def vector2 (k : ℝ) : Fin 3 → ℝ := ![2, k, 3]
def vector3 (k : ℝ) : Fin 3 → ℝ := ![2, 3, k]

def matrix (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.of (λ i j => match i, j with
    | 0, 0 => 3 | 0, 1 => 2 | 0, 2 => 2
    | 1, 0 => 4 | 1, 1 => k | 1, 2 => 3
    | 2, 0 => 5 | 2, 1 => 3 | 2, 2 => k
    | _, _ => 0)

theorem parallelepiped_volume (k : ℝ) :
  k > 0 ∧ |Matrix.det (matrix k)| = 30 → k = 3 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l61_6141


namespace NUMINAMATH_CALUDE_inequality_proof_l61_6112

theorem inequality_proof (a b c : ℝ) 
  (h1 : c < b) (h2 : b < a) (h3 : a + b + c = 0) : 
  c * b^2 ≤ a * b^2 ∧ a * b > a * c := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l61_6112


namespace NUMINAMATH_CALUDE_circle_radius_proof_l61_6173

theorem circle_radius_proof (r : ℝ) (x y : ℝ) : 
  x = π * r^2 →
  y = 2 * π * r - 6 →
  x + y = 94 * π →
  r = 10 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l61_6173


namespace NUMINAMATH_CALUDE_opposite_sides_equal_implies_parallelogram_l61_6178

/-- A quadrilateral is represented by four points in a 2D plane -/
structure Quadrilateral (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C D : V)

/-- Definition of a parallelogram -/
def is_parallelogram {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (q : Quadrilateral V) : Prop :=
  q.A - q.B = q.D - q.C ∧ q.A - q.D = q.B - q.C

/-- Theorem: If opposite sides of a quadrilateral are equal, it is a parallelogram -/
theorem opposite_sides_equal_implies_parallelogram 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (q : Quadrilateral V) 
  (h1 : q.A - q.B = q.D - q.C) 
  (h2 : q.A - q.D = q.B - q.C) : 
  is_parallelogram q :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_equal_implies_parallelogram_l61_6178


namespace NUMINAMATH_CALUDE_gcd_of_45_75_90_l61_6168

theorem gcd_of_45_75_90 : Nat.gcd 45 (Nat.gcd 75 90) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_75_90_l61_6168


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l61_6127

theorem quadratic_roots_property (p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ x₁ - x₂ = 5 ∧ x₁^3 - x₂^3 = 35) →
  ((p = 1 ∧ q = -6) ∨ (p = -1 ∧ q = -6)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l61_6127


namespace NUMINAMATH_CALUDE_prob_bus_251_theorem_l61_6135

/-- Represents the bus schedule system with two routes -/
structure BusSchedule where
  interval_152 : ℕ
  interval_251 : ℕ

/-- The probability of getting on bus No. 251 given a bus schedule -/
def prob_bus_251 (schedule : BusSchedule) : ℚ :=
  5 / 14

/-- Theorem stating the probability of getting on bus No. 251 -/
theorem prob_bus_251_theorem (schedule : BusSchedule) 
  (h1 : schedule.interval_152 = 5)
  (h2 : schedule.interval_251 = 7) :
  prob_bus_251 schedule = 5 / 14 := by
  sorry

#eval prob_bus_251 ⟨5, 7⟩

end NUMINAMATH_CALUDE_prob_bus_251_theorem_l61_6135


namespace NUMINAMATH_CALUDE_statement_equivalence_l61_6108

theorem statement_equivalence (P Q R : Prop) :
  (P → (Q ∧ ¬R)) ↔ ((¬Q ∨ R) → ¬P) := by
  sorry

end NUMINAMATH_CALUDE_statement_equivalence_l61_6108


namespace NUMINAMATH_CALUDE_profit_calculation_l61_6177

-- Define the buying and selling rates
def buy_rate : ℚ := 5 / 6
def sell_rate : ℚ := 4 / 8

-- Define the target profit
def target_profit : ℚ := 120

-- Define the number of disks to be sold
def disks_to_sell : ℕ := 150

-- Theorem statement
theorem profit_calculation :
  (disks_to_sell : ℚ) * (1 / sell_rate - 1 / buy_rate) = target_profit := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l61_6177


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l61_6197

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- The binary representation of 1101101₂ -/
def binary_num : List Bool := [true, true, false, true, true, false, true]

/-- The expected quaternary representation -/
def expected_quaternary : List Nat := [3, 1, 2, 1]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_num) = expected_quaternary :=
by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l61_6197


namespace NUMINAMATH_CALUDE_matt_darius_difference_l61_6131

/-- The scores of three friends in a table football game. -/
structure Scores where
  darius : ℕ
  matt : ℕ
  marius : ℕ

/-- The conditions of the table football game. -/
def game_conditions (s : Scores) : Prop :=
  s.darius = 10 ∧
  s.marius = s.darius + 3 ∧
  s.matt > s.darius ∧
  s.darius + s.matt + s.marius = 38

/-- The theorem stating the difference between Matt's and Darius's scores. -/
theorem matt_darius_difference (s : Scores) (h : game_conditions s) : 
  s.matt - s.darius = 5 := by
  sorry

end NUMINAMATH_CALUDE_matt_darius_difference_l61_6131


namespace NUMINAMATH_CALUDE_complex_repairs_is_two_l61_6144

/-- Represents Jim's bike shop operations for a month --/
structure BikeShop where
  tire_repair_price : ℕ
  tire_repair_cost : ℕ
  tire_repairs_count : ℕ
  complex_repair_price : ℕ
  complex_repair_cost : ℕ
  retail_profit : ℕ
  fixed_expenses : ℕ
  total_profit : ℕ

/-- Calculates the number of complex repairs given the shop's operations --/
def complex_repairs_count (shop : BikeShop) : ℕ :=
  sorry

/-- Theorem stating that the number of complex repairs is 2 --/
theorem complex_repairs_is_two (shop : BikeShop) 
  (h1 : shop.tire_repair_price = 20)
  (h2 : shop.tire_repair_cost = 5)
  (h3 : shop.tire_repairs_count = 300)
  (h4 : shop.complex_repair_price = 300)
  (h5 : shop.complex_repair_cost = 50)
  (h6 : shop.retail_profit = 2000)
  (h7 : shop.fixed_expenses = 4000)
  (h8 : shop.total_profit = 3000) :
  complex_repairs_count shop = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_repairs_is_two_l61_6144


namespace NUMINAMATH_CALUDE_tangent_slope_at_2_l61_6134

-- Define the function f(x) = x^2 + 3x
def f (x : ℝ) : ℝ := x^2 + 3*x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 2*x + 3

-- Theorem statement
theorem tangent_slope_at_2 :
  f' 2 = 7 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_2_l61_6134


namespace NUMINAMATH_CALUDE_jackson_spending_money_l61_6125

/-- The amount of money earned per hour of chores -/
def money_per_hour : ℝ := 5

/-- The time spent vacuuming (in hours) -/
def vacuuming_time : ℝ := 2 * 2

/-- The time spent washing dishes (in hours) -/
def dish_washing_time : ℝ := 0.5

/-- The time spent cleaning the bathroom (in hours) -/
def bathroom_cleaning_time : ℝ := 3 * dish_washing_time

/-- The total time spent on chores (in hours) -/
def total_chore_time : ℝ := vacuuming_time + dish_washing_time + bathroom_cleaning_time

/-- The theorem stating that Jackson's earned spending money is $30 -/
theorem jackson_spending_money : money_per_hour * total_chore_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_jackson_spending_money_l61_6125


namespace NUMINAMATH_CALUDE_inequality_region_l61_6105

theorem inequality_region (x y : ℝ) : 
  Real.sqrt (x * y) ≥ x - 2 * y ↔ 
  ((x ≥ 0 ∧ y ≥ 0 ∧ y ≥ x / 2) ∨ 
   (x ≤ 0 ∧ y ≤ 0 ∧ y ≥ x / 2) ∨ 
   (x = 0 ∧ y ≥ 0) ∨ 
   (x ≥ 0 ∧ y = 0)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_region_l61_6105


namespace NUMINAMATH_CALUDE_officer_selection_count_l61_6176

def total_members : ℕ := 25
def num_officers : ℕ := 3

-- Define a structure to represent a pair of members
structure MemberPair :=
  (member1 : ℕ)
  (member2 : ℕ)

-- Define the two special pairs
def pair1 : MemberPair := ⟨1, 2⟩  -- Rachel and Simon
def pair2 : MemberPair := ⟨3, 4⟩  -- Penelope and Quentin

-- Function to calculate the number of ways to choose officers
def count_officer_choices (total : ℕ) (officers : ℕ) (pair1 pair2 : MemberPair) : ℕ := 
  sorry

-- Theorem statement
theorem officer_selection_count :
  count_officer_choices total_members num_officers pair1 pair2 = 8072 :=
sorry

end NUMINAMATH_CALUDE_officer_selection_count_l61_6176


namespace NUMINAMATH_CALUDE_x_y_values_l61_6157

theorem x_y_values (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x + y = 30) : x = 6 ∧ y = 24 := by
  sorry

end NUMINAMATH_CALUDE_x_y_values_l61_6157


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_nine_l61_6110

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_neg_two_eq_neg_nine
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x ∈ Set.Icc 1 5, f x = x^3 + 1) :
  f (-2) = -9 := by sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_nine_l61_6110


namespace NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l61_6152

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(n % p = 0)

def is_nonprime (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

theorem smallest_nonprime_with_large_factors :
  ∃ n : ℕ, is_nonprime n ∧
            has_no_prime_factors_less_than n 20 ∧
            (∀ m : ℕ, m < n → ¬(is_nonprime m ∧ has_no_prime_factors_less_than m 20)) ∧
            500 < n ∧ n ≤ 550 :=
sorry

end NUMINAMATH_CALUDE_smallest_nonprime_with_large_factors_l61_6152


namespace NUMINAMATH_CALUDE_negation_equivalence_l61_6150

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ + 1) * Real.exp x₀ ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l61_6150


namespace NUMINAMATH_CALUDE_digits_at_1100_to_1102_l61_6169

/-- Represents a list of integers starting with 2 in increasing order -/
def listStartingWith2 : List ℕ := sorry

/-- Returns the nth digit in the concatenated string of all numbers in the list -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 1100th, 1101st, and 1102nd digits are 2, 1, and 9 respectively -/
theorem digits_at_1100_to_1102 :
  (nthDigit 1100 = 2) ∧ (nthDigit 1101 = 1) ∧ (nthDigit 1102 = 9) := by sorry

end NUMINAMATH_CALUDE_digits_at_1100_to_1102_l61_6169


namespace NUMINAMATH_CALUDE_not_proportional_l61_6190

-- Define the notion of direct proportionality
def is_directly_proportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, y t = k * x t

-- Define the notion of inverse proportionality
def is_inversely_proportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

-- Define our equation
def our_equation (x y : ℝ) : Prop :=
  2 * x + 3 * y = 6

-- Theorem statement
theorem not_proportional :
  ¬ (∃ x y : ℝ → ℝ, (∀ t : ℝ, our_equation (x t) (y t)) ∧
    (is_directly_proportional x y ∨ is_inversely_proportional x y)) :=
sorry

end NUMINAMATH_CALUDE_not_proportional_l61_6190


namespace NUMINAMATH_CALUDE_smallest_perfect_square_sum_l61_6192

def consecutive_sum (n : ℕ) : ℕ := 10 * (2 * n + 19)

theorem smallest_perfect_square_sum :
  ∃ (n : ℕ), 
    (∀ (m : ℕ), m < n → ¬∃ (k : ℕ), consecutive_sum m = k^2) ∧
    (∃ (k : ℕ), consecutive_sum n = k^2) ∧
    consecutive_sum n = 1000 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_sum_l61_6192


namespace NUMINAMATH_CALUDE_combined_average_marks_l61_6155

theorem combined_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℝ) : 
  n1 = 26 → 
  n2 = 50 → 
  avg1 = 40 → 
  avg2 = 60 → 
  let total_students := n1 + n2
  let total_marks := n1 * avg1 + n2 * avg2
  abs ((total_marks / total_students) - 53.16) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_combined_average_marks_l61_6155


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt3_minus2_l61_6191

theorem rationalize_denominator_sqrt3_minus2 :
  1 / (Real.sqrt 3 - 2) = -Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt3_minus2_l61_6191


namespace NUMINAMATH_CALUDE_peter_marbles_l61_6123

theorem peter_marbles (initial_marbles lost_marbles : ℕ) 
  (h1 : initial_marbles = 33)
  (h2 : lost_marbles = 15) :
  initial_marbles - lost_marbles = 18 := by
  sorry

end NUMINAMATH_CALUDE_peter_marbles_l61_6123


namespace NUMINAMATH_CALUDE_correct_inequalities_l61_6160

/-- 
Given a student's estimated scores in Chinese and Mathematics after a mock final exam,
this theorem proves that the correct system of inequalities representing the situation is
x > 85 and y ≥ 80, where x is the Chinese score and y is the Mathematics score.
-/
theorem correct_inequalities (x y : ℝ) 
  (h1 : x > 85)  -- Chinese score is higher than 85 points
  (h2 : y ≥ 80)  -- Mathematics score is not less than 80 points
  : x > 85 ∧ y ≥ 80 := by
  sorry

end NUMINAMATH_CALUDE_correct_inequalities_l61_6160


namespace NUMINAMATH_CALUDE_equation_solution_range_l61_6151

theorem equation_solution_range (k : ℝ) :
  (∃ x : ℝ, (4 * (2015^x) - 2015^(-x)) / (2015^x - 3 * (2015^(-x))) = k) ↔ 
  (k < 1/3 ∨ k > 4) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_range_l61_6151


namespace NUMINAMATH_CALUDE_aprils_roses_l61_6174

theorem aprils_roses (price : ℕ) (remaining : ℕ) (earnings : ℕ) (initial : ℕ) : 
  price = 7 → remaining = 4 → earnings = 35 → initial * price - remaining * price = earnings → initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_aprils_roses_l61_6174


namespace NUMINAMATH_CALUDE_clara_has_68_stickers_l61_6183

/-- Calculates the number of stickers Clara has left after a series of transactions -/
def claras_stickers : ℕ :=
  let initial := 100
  let after_boy := initial - 10
  let after_teacher := after_boy + 50
  let after_classmates := after_teacher - 20
  let after_exchange := after_classmates - 15 + 30
  let to_friends := after_exchange / 2
  after_exchange - to_friends

/-- Proves that Clara ends up with 68 stickers -/
theorem clara_has_68_stickers : claras_stickers = 68 := by
  sorry

#eval claras_stickers

end NUMINAMATH_CALUDE_clara_has_68_stickers_l61_6183


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l61_6158

theorem fraction_to_decimal : (17 : ℚ) / 200 = (34 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l61_6158


namespace NUMINAMATH_CALUDE_system_solution_l61_6159

theorem system_solution (a b : ℝ) : 
  (∃ x y : ℝ, a * x - y = 4 ∧ 3 * x + b * y = 4 ∧ x = 2 ∧ y = -2) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l61_6159


namespace NUMINAMATH_CALUDE_trigonometric_fraction_simplification_l61_6185

theorem trigonometric_fraction_simplification (x : ℝ) :
  (2 + 3 * Real.sin x - 4 * Real.cos x) / (2 + 3 * Real.sin x + 2 * Real.cos x) 
  = (-1 + 3 * Real.sin (x/2) * Real.cos (x/2) + 4 * (Real.sin (x/2))^2) / 
    (2 + 3 * Real.sin (x/2) * Real.cos (x/2) - 2 * (Real.sin (x/2))^2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_fraction_simplification_l61_6185


namespace NUMINAMATH_CALUDE_random_events_charge_attraction_certain_no_impossible_events_l61_6100

-- Define the type for events
inductive Event
| GlassCups
| CannonFiring
| PhoneNumber
| ChargeAttraction
| LotteryWin

-- Define the property of being a random event
def isRandomEvent (e : Event) : Prop :=
  match e with
  | Event.GlassCups => true
  | Event.CannonFiring => true
  | Event.PhoneNumber => true
  | Event.ChargeAttraction => false
  | Event.LotteryWin => true

-- Theorem stating which events are random
theorem random_events :
  (isRandomEvent Event.GlassCups) ∧
  (isRandomEvent Event.CannonFiring) ∧
  (isRandomEvent Event.PhoneNumber) ∧
  (¬isRandomEvent Event.ChargeAttraction) ∧
  (isRandomEvent Event.LotteryWin) :=
by sorry

-- Definition of a certain event
def isCertainEvent (e : Event) : Prop :=
  match e with
  | Event.ChargeAttraction => true
  | _ => false

-- Theorem stating that charge attraction is a certain event
theorem charge_attraction_certain :
  isCertainEvent Event.ChargeAttraction :=
by sorry

-- Definition of an impossible event
def isImpossibleEvent (e : Event) : Prop := false

-- Theorem stating that none of the given events are impossible
theorem no_impossible_events :
  ∀ e : Event, ¬(isImpossibleEvent e) :=
by sorry

end NUMINAMATH_CALUDE_random_events_charge_attraction_certain_no_impossible_events_l61_6100


namespace NUMINAMATH_CALUDE_polynomial_transformation_l61_6163

theorem polynomial_transformation (g : ℝ → ℝ) :
  (∀ x, g (x^2 - 2) = x^4 - 6*x^2 + 8) →
  (∀ x, g (x^2 - 1) = x^4 - 4*x^2 + 7) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l61_6163


namespace NUMINAMATH_CALUDE_temperature_problem_l61_6102

/-- The temperature problem in three cities -/
theorem temperature_problem (T_NY : ℝ) : 
  (∃ (T_Miami T_SD : ℝ),
    T_Miami = T_NY + 10 ∧
    T_SD = T_Miami + 25 ∧
    (T_NY + T_Miami + T_SD) / 3 = 95) →
  T_NY = 80 := by
sorry

end NUMINAMATH_CALUDE_temperature_problem_l61_6102


namespace NUMINAMATH_CALUDE_inequality_proof_l61_6139

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (1 / x^2 - x) * (1 / y^2 - y) * (1 / z^2 - z) ≥ (26 / 3)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l61_6139


namespace NUMINAMATH_CALUDE_circle_properties_l61_6111

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a diameter
def diameter (c : Circle) (p q : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
  (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2 ∧
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = (2 * c.radius)^2

-- Define a point on the circle
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_properties (c : Circle) :
  (∀ p q : ℝ × ℝ, diameter c p q → ∀ r s : ℝ × ℝ, diameter c r s → 
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = (r.1 - s.1)^2 + (r.2 - s.2)^2) ∧
  (∀ p q : ℝ × ℝ, onCircle c p → onCircle c q → 
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l61_6111


namespace NUMINAMATH_CALUDE_jerrys_age_l61_6184

theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 22 → 
  mickey_age = 2 * jerry_age - 6 → 
  jerry_age = 14 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l61_6184


namespace NUMINAMATH_CALUDE_train_crossing_time_l61_6117

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 180 → 
  train_speed_kmh = 108 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l61_6117


namespace NUMINAMATH_CALUDE_square_area_ratio_l61_6132

/-- The ratio of areas between a smaller square and a larger square, given specific conditions -/
theorem square_area_ratio : ∀ (r : ℝ) (y : ℝ),
  r > 0 →  -- radius of circumscribed circle is positive
  r = 4 * Real.sqrt 2 →  -- radius of circumscribed circle
  y > 0 →  -- half side length of smaller square is positive
  y * (3 * y - 8 * Real.sqrt 2) = 0 →  -- condition for diagonal touching circle
  (2 * y)^2 / 8^2 = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l61_6132


namespace NUMINAMATH_CALUDE_face_vertex_assignment_l61_6145

-- Define a planar bipartite graph
class PlanarBipartiteGraph (G : Type) where
  -- Add necessary properties for planar bipartite graphs
  is_planar : Bool
  is_bipartite : Bool

-- Define faces and vertices of a graph
def faces (G : Type) [PlanarBipartiteGraph G] : Set G := sorry
def vertices (G : Type) [PlanarBipartiteGraph G] : Set G := sorry

-- Theorem statement
theorem face_vertex_assignment {G : Type} [PlanarBipartiteGraph G] :
  ∃ f : faces G → vertices G, Function.Injective f :=
sorry

end NUMINAMATH_CALUDE_face_vertex_assignment_l61_6145


namespace NUMINAMATH_CALUDE_angle_relation_l61_6119

theorem angle_relation (α β : Real) (x y : Real) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos (α + β) = -4/5)
  (h4 : Real.sin β = x)
  (h5 : Real.cos α = y)
  (h6 : 4/5 < x ∧ x < 1) :
  y = -4/5 * Real.sqrt (1 - x^2) + 3/5 * x := by
  sorry

#check angle_relation

end NUMINAMATH_CALUDE_angle_relation_l61_6119


namespace NUMINAMATH_CALUDE_circle_C_equation_line_MN_equation_l61_6195

-- Define the circle C
def circle_C (x y m : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + m = 0

-- Define the line that the circle is tangent to
def tangent_line (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + Real.sqrt 3 - 2 = 0

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop :=
  x + 2*y = 0

-- Theorem for the equation of circle C
theorem circle_C_equation :
  ∃ m, ∀ x y, circle_C x y m ↔ (x+2)^2 + (y-1)^2 = 4 :=
sorry

-- Theorem for the equation of line MN
theorem line_MN_equation :
  ∃ M N : ℝ × ℝ,
    (∀ x y, circle_C x y 0 → (x, y) = M ∨ (x, y) = N) ∧
    (symmetry_line M.1 M.2 ↔ symmetry_line N.1 N.2) ∧
    ((M.1 - N.1)^2 + (M.2 - N.2)^2 = 12) →
    ∃ c, ∀ x y, (2*x - y + c = 0 ∨ 2*x - y + (10 - c) = 0) ∧ c^2 = 30 :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_line_MN_equation_l61_6195


namespace NUMINAMATH_CALUDE_quadratic_equation_c_value_l61_6107

theorem quadratic_equation_c_value (c : ℝ) : 
  (∀ x : ℝ, 2*x^2 + 8*x + c = 0 ↔ x = (-8 + Real.sqrt 20) / 4 ∨ x = (-8 - Real.sqrt 20) / 4) →
  c = 5.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_c_value_l61_6107


namespace NUMINAMATH_CALUDE_rectangle_to_square_theorem_l61_6186

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- Function to calculate the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.width * r.height

/-- Function to calculate the area of a square -/
def squareArea (s : Square) : ℝ := s.side * s.side

/-- Theorem stating the relationship between the rectangle and the resulting square -/
theorem rectangle_to_square_theorem (r : Rectangle) (s : Square) (y : ℝ) :
  r.width = 10 ∧ r.height = 15 ∧
  rectangleArea r = squareArea s ∧
  y = 3 * min r.width r.height →
  y = 30 := by
  sorry

#check rectangle_to_square_theorem

end NUMINAMATH_CALUDE_rectangle_to_square_theorem_l61_6186


namespace NUMINAMATH_CALUDE_madeline_pencils_l61_6180

/-- The number of colored pencils each person has -/
structure ColoredPencils where
  cyrus : ℕ
  cheryl : ℕ
  madeline : ℕ

/-- The conditions of the problem -/
def pencil_conditions (p : ColoredPencils) : Prop :=
  p.cheryl = 3 * p.cyrus ∧
  p.madeline = p.cheryl / 2 ∧
  p.cyrus + p.cheryl + p.madeline = 231

/-- The theorem to prove -/
theorem madeline_pencils :
  ∀ p : ColoredPencils, pencil_conditions p → p.madeline = 63 :=
by sorry

end NUMINAMATH_CALUDE_madeline_pencils_l61_6180


namespace NUMINAMATH_CALUDE_range_of_a_l61_6181

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 8*x - 20 > 0 → x^2 - 2*x + 1 - a^2 > 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x + 1 - a^2 > 0 ∧ x^2 - 8*x - 20 ≤ 0) ∧ 
  a > 0 
  ↔ 0 < a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l61_6181


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l61_6188

/-- The focal length of a hyperbola with given properties -/
theorem hyperbola_focal_length (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let e := 2  -- eccentricity
  let d := Real.sqrt 3  -- distance from focus to asymptote
  2 * Real.sqrt (a^2 + b^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l61_6188


namespace NUMINAMATH_CALUDE_competitive_exam_candidates_l61_6148

theorem competitive_exam_candidates (candidates : ℕ) : 
  (candidates * 8 / 100 : ℚ) + 220 = (candidates * 12 / 100 : ℚ) →
  candidates = 5500 := by
sorry

end NUMINAMATH_CALUDE_competitive_exam_candidates_l61_6148


namespace NUMINAMATH_CALUDE_translated_line_proof_l61_6118

/-- Given a line y = 2x + 5 translated down by m units (m > 0) -/
def translated_line (x : ℝ) (m : ℝ) : ℝ := 2 * x + 5 - m

theorem translated_line_proof (m : ℝ) (h_m : m > 0) :
  (translated_line (-2) m = -6 → m = 7) ∧
  (∀ x : ℝ, translated_line x 7 < 0 ↔ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_translated_line_proof_l61_6118


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l61_6113

theorem complex_number_in_fourth_quadrant (z : ℂ) (h : (1 + Complex.I) * z = 2 - Complex.I) :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l61_6113


namespace NUMINAMATH_CALUDE_interpolation_polynomial_existence_and_uniqueness_l61_6104

theorem interpolation_polynomial_existence_and_uniqueness
  (n : ℕ) (x y : Fin n → ℝ) (h : ∀ i j : Fin n, i < j → x i < x j) :
  ∃! f : ℝ → ℝ,
    (∀ i : Fin n, f (x i) = y i) ∧
    ∃ p : Polynomial ℝ, (∀ t, f t = p.eval t) ∧ p.degree < n :=
sorry

end NUMINAMATH_CALUDE_interpolation_polynomial_existence_and_uniqueness_l61_6104


namespace NUMINAMATH_CALUDE_trig_identities_l61_6101

open Real

theorem trig_identities (α x : ℝ) (h : tan α = 2) :
  (2 * sin α - cos α) / (sin α + 2 * cos α) = 3/4 ∧
  2 * sin x ^ 2 - sin x * cos x + cos x ^ 2 = 2 - sin (2 * x) / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l61_6101


namespace NUMINAMATH_CALUDE_paint_cost_theorem_l61_6136

-- Define the paint properties
structure Paint where
  cost : Float
  coverage : Float

-- Define the cuboid dimensions
def cuboid_length : Float := 12
def cuboid_width : Float := 15
def cuboid_height : Float := 20

-- Define the paints
def paint_A : Paint := { cost := 3.20, coverage := 60 }
def paint_B : Paint := { cost := 5.50, coverage := 55 }
def paint_C : Paint := { cost := 4.00, coverage := 50 }

-- Calculate the areas of the faces
def largest_face_area : Float := 2 * cuboid_width * cuboid_height
def middle_face_area : Float := 2 * cuboid_length * cuboid_height
def smallest_face_area : Float := 2 * cuboid_length * cuboid_width

-- Calculate the number of quarts needed for each paint
def quarts_A : Float := Float.ceil (largest_face_area / paint_A.coverage)
def quarts_B : Float := Float.ceil (middle_face_area / paint_B.coverage)
def quarts_C : Float := Float.ceil (smallest_face_area / paint_C.coverage)

-- Calculate the total cost
def total_cost : Float := quarts_A * paint_A.cost + quarts_B * paint_B.cost + quarts_C * paint_C.cost

-- Theorem to prove
theorem paint_cost_theorem : total_cost = 113.50 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_theorem_l61_6136


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l61_6156

theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (remaining_players_age_diff : ℕ),
    team_size = 11 →
    captain_age = 25 →
    wicket_keeper_age_diff = 3 →
    remaining_players_age_diff = 1 →
    ∃ (team_average_age : ℚ),
      team_average_age = 22 ∧
      team_average_age * team_size =
        captain_age + (captain_age + wicket_keeper_age_diff) +
        (team_size - 2) * (team_average_age - remaining_players_age_diff) :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l61_6156


namespace NUMINAMATH_CALUDE_infinite_product_a_l61_6130

noncomputable def a : ℕ → ℚ
  | 0 => 2/3
  | n+1 => 1 + (a n - 1)^2

theorem infinite_product_a : ∏' n, a n = 1/2 := by sorry

end NUMINAMATH_CALUDE_infinite_product_a_l61_6130


namespace NUMINAMATH_CALUDE_faye_age_l61_6171

/-- Given the ages of Chad, Diana, Eduardo, and Faye, prove that Faye is 18 years old. -/
theorem faye_age (C D E F : ℕ) 
  (h1 : D = E - 2)
  (h2 : E = C + 3)
  (h3 : F = C + 4)
  (h4 : D = 15) :
  F = 18 := by
  sorry

end NUMINAMATH_CALUDE_faye_age_l61_6171


namespace NUMINAMATH_CALUDE_smallest_x_with_given_remainders_l61_6146

theorem smallest_x_with_given_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧
  ∀ (y : ℕ), y > 0 → 
    (y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7) → 
    x ≤ y ∧ 
  x = 167 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_with_given_remainders_l61_6146


namespace NUMINAMATH_CALUDE_basketball_team_starters_count_l61_6170

theorem basketball_team_starters_count :
  let total_players : ℕ := 16
  let quadruplets : ℕ := 4
  let starters : ℕ := 7
  let quadruplets_in_lineup : ℕ := 2
  let remaining_players : ℕ := total_players - quadruplets
  let remaining_starters : ℕ := starters - quadruplets_in_lineup

  (Nat.choose quadruplets quadruplets_in_lineup) *
  (Nat.choose remaining_players remaining_starters) = 4752 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_starters_count_l61_6170


namespace NUMINAMATH_CALUDE_counterexample_exists_l61_6126

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a * b > 0 ∧ 1/a ≥ 1/b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l61_6126


namespace NUMINAMATH_CALUDE_ella_video_game_spending_percentage_l61_6115

/-- Represents Ella's salary and spending on video games --/
structure EllasFinances where
  lastYearSalary : ℝ
  videoGameSpending : ℝ
  raisePercentage : ℝ
  newSalary : ℝ

/-- Theorem stating that Ella spends 40% of her salary on video games --/
theorem ella_video_game_spending_percentage
  (e : EllasFinances)
  (h1 : e.videoGameSpending = 100)
  (h2 : e.raisePercentage = 0.1)
  (h3 : e.newSalary = 275)
  (h4 : e.newSalary = e.lastYearSalary * (1 + e.raisePercentage)) :
  e.videoGameSpending / e.lastYearSalary = 0.4 := by
  sorry


end NUMINAMATH_CALUDE_ella_video_game_spending_percentage_l61_6115


namespace NUMINAMATH_CALUDE_volume_inscribed_sphere_l61_6175

/-- The volume of a sphere inscribed in a cube -/
theorem volume_inscribed_sphere (cube_volume : ℝ) (sphere_volume : ℝ) : 
  cube_volume = 343 →
  sphere_volume = (343 * Real.pi) / 6 :=
by sorry

end NUMINAMATH_CALUDE_volume_inscribed_sphere_l61_6175


namespace NUMINAMATH_CALUDE_gcd_2210_145_l61_6122

theorem gcd_2210_145 : Int.gcd 2210 145 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2210_145_l61_6122


namespace NUMINAMATH_CALUDE_solution_pairs_count_l61_6143

theorem solution_pairs_count : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    4 * p.1 + 7 * p.2 = 548 ∧ p.1 > 0 ∧ p.2 > 0) 
    (Finset.product (Finset.range 548) (Finset.range 548))).card ∧ n = 19 := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_count_l61_6143


namespace NUMINAMATH_CALUDE_martha_butterflies_l61_6161

def butterfly_collection (blue yellow black : ℕ) : Prop :=
  blue = 2 * yellow ∧ black = 5 ∧ blue = 4

theorem martha_butterflies :
  ∀ blue yellow black : ℕ,
  butterfly_collection blue yellow black →
  blue + yellow + black = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_martha_butterflies_l61_6161


namespace NUMINAMATH_CALUDE_triangle_projection_types_l61_6147

-- Define the possible projection types
inductive ProjectionType
  | Angle
  | Strip
  | TwoAnglesJoined
  | Triangle
  | AngleWithInfiniteFigure

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Function to determine if a point is on a plane
def isPointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

-- Function to determine if three points are collinear
def areCollinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ (t : ℝ), p3.x - p1.x = t * (p2.x - p1.x) ∧
              p3.y - p1.y = t * (p2.y - p1.y) ∧
              p3.z - p1.z = t * (p2.z - p1.z)

-- Define the projection function
def project (triangle : Triangle3D) (O : Point3D) (P : Plane3D) : ProjectionType :=
  sorry -- Actual implementation would go here

-- The main theorem
theorem triangle_projection_types 
  (triangle : Triangle3D) 
  (O : Point3D) 
  (P : Plane3D) 
  (h1 : ¬ isPointOnPlane O (Plane3D.mk 0 0 0 0)) -- O is not in the plane of the triangle
  (h2 : ¬ areCollinear triangle.A triangle.B triangle.C) -- ABC is a valid triangle
  : ∃ (projType : ProjectionType), project triangle O P = projType ∧ 
    (projType = ProjectionType.Angle ∨ 
     projType = ProjectionType.Strip ∨ 
     projType = ProjectionType.TwoAnglesJoined ∨ 
     projType = ProjectionType.Triangle ∨ 
     projType = ProjectionType.AngleWithInfiniteFigure) :=
  sorry


end NUMINAMATH_CALUDE_triangle_projection_types_l61_6147


namespace NUMINAMATH_CALUDE_cougar_ratio_l61_6172

theorem cougar_ratio (lions tigers total : ℕ) 
  (h1 : lions = 12)
  (h2 : tigers = 14)
  (h3 : total = 39) :
  (total - (lions + tigers)) * 2 = lions + tigers :=
by sorry

end NUMINAMATH_CALUDE_cougar_ratio_l61_6172


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l61_6165

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 2) * (Real.sqrt 9 / Real.sqrt 13) * (Real.sqrt 22 / Real.sqrt 7) = 
  (3 * Real.sqrt 20020) / 182 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l61_6165


namespace NUMINAMATH_CALUDE_joshua_fruit_profit_l61_6133

/-- Calculates the total profit in cents for Joshua's fruit sales --/
def fruit_profit (orange_qty : ℕ) (apple_qty : ℕ) (banana_qty : ℕ)
  (orange_cost : ℚ) (apple_cost : ℚ) (banana_cost : ℚ)
  (orange_sell : ℚ) (apple_sell : ℚ) (banana_sell : ℚ)
  (discount_threshold : ℕ) (discount_rate : ℚ) : ℕ :=
  let orange_total_cost := orange_qty * orange_cost
  let apple_total_cost := if apple_qty ≥ discount_threshold
    then apple_qty * (apple_cost * (1 - discount_rate))
    else apple_qty * apple_cost
  let banana_total_cost := if banana_qty ≥ discount_threshold
    then banana_qty * (banana_cost * (1 - discount_rate))
    else banana_qty * banana_cost
  let total_cost := orange_total_cost + apple_total_cost + banana_total_cost
  let total_revenue := orange_qty * orange_sell + apple_qty * apple_sell + banana_qty * banana_sell
  let profit := total_revenue - total_cost
  (profit * 100).floor.toNat

/-- Theorem stating that Joshua's profit is 2035 cents --/
theorem joshua_fruit_profit :
  fruit_profit 25 40 50 0.5 0.65 0.25 0.6 0.75 0.45 30 0.1 = 2035 := by
  sorry

end NUMINAMATH_CALUDE_joshua_fruit_profit_l61_6133


namespace NUMINAMATH_CALUDE_hairdresser_cash_register_l61_6194

theorem hairdresser_cash_register (x : ℝ) : 
  (8 * x - 70 = 0) → x = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_hairdresser_cash_register_l61_6194


namespace NUMINAMATH_CALUDE_calculation_proofs_l61_6142

theorem calculation_proofs :
  (4.4 * 25 = 110) ∧
  (13.2 * 1.1 - 8.45 = 6.07) ∧
  (76.84 * 103 - 7.684 * 30 = 7684) ∧
  ((2.8 + 3.85 / 3.5) / 3 = 1.3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l61_6142


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l61_6198

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 4 ∧ 
  (x₁^2 - 6*x₁ + 8 = 0) ∧ (x₂^2 - 6*x₂ + 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l61_6198


namespace NUMINAMATH_CALUDE_complex_linear_combination_l61_6109

theorem complex_linear_combination :
  let x : ℂ := 3 + 2*I
  let y : ℂ := 2 - 3*I
  3*x + 4*y = 17 - 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_linear_combination_l61_6109


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l61_6193

theorem rectangular_prism_volume 
  (m n Q : ℝ) 
  (m_pos : m > 0) 
  (n_pos : n > 0) 
  (Q_pos : Q > 0) : 
  let base_ratio := m / n
  let diagonal_area := Q
  let volume := (m * n * Q * Real.sqrt Q) / (m^2 + n^2)
  ∃ (a b h : ℝ), 
    a > 0 ∧ b > 0 ∧ h > 0 ∧
    a / b = base_ratio ∧
    a * a + b * b = Q ∧
    h * h = Q ∧
    a * b * h = volume :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l61_6193


namespace NUMINAMATH_CALUDE_square_diagonal_ratio_l61_6120

theorem square_diagonal_ratio (a b : ℝ) (h : b^2 / a^2 = 4) :
  (b * Real.sqrt 2) / (a * Real.sqrt 2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_ratio_l61_6120


namespace NUMINAMATH_CALUDE_smallest_three_digit_with_product_8_and_even_digit_l61_6103

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

def has_even_digit (n : ℕ) : Prop :=
  (n / 100) % 2 = 0 ∨ ((n / 10) % 10) % 2 = 0 ∨ (n % 10) % 2 = 0

theorem smallest_three_digit_with_product_8_and_even_digit :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → has_even_digit n → 124 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_with_product_8_and_even_digit_l61_6103


namespace NUMINAMATH_CALUDE_select_three_roles_from_25_l61_6124

/-- The number of ways to select three distinct roles from a squad of players. -/
def selectThreeRoles (squadSize : ℕ) : ℕ :=
  squadSize * (squadSize - 1) * (squadSize - 2)

/-- Theorem: The number of ways to select a captain, vice-captain, and goalkeeper
    from a squad of 25 players, where no player can occupy more than one role, is 13800. -/
theorem select_three_roles_from_25 : selectThreeRoles 25 = 13800 := by
  sorry

end NUMINAMATH_CALUDE_select_three_roles_from_25_l61_6124


namespace NUMINAMATH_CALUDE_first_player_wins_l61_6137

/-- Represents a position on the rectangular table -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents the game state -/
structure GameState :=
  (table : Set Position)
  (occupied : Set Position)
  (currentPlayer : Nat)

/-- Defines a valid move in the game -/
def validMove (state : GameState) (pos : Position) : Prop :=
  pos ∈ state.table ∧ pos ∉ state.occupied

/-- Defines the winning condition for a player -/
def winningStrategy (player : Nat) : Prop :=
  ∀ (state : GameState), 
    state.currentPlayer = player → 
    ∃ (move : Position), validMove state move

/-- The main theorem stating that the first player has a winning strategy -/
theorem first_player_wins :
  ∃ (strategy : GameState → Position), 
    winningStrategy 1 ∧ 
    (∀ (state : GameState), 
      state.currentPlayer = 1 → 
      validMove state (strategy state)) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l61_6137


namespace NUMINAMATH_CALUDE_express_train_meetings_l61_6182

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- The problem statement -/
theorem express_train_meetings :
  let travelTime : Nat := 210 -- 3 hours and 30 minutes in minutes
  let departureInterval : Nat := 60 -- 1 hour in minutes
  let firstDeparture : Time := ⟨6, 0⟩ -- 6:00 AM
  let expressDeparture : Time := ⟨9, 0⟩ -- 9:00 AM
  let expressArrival : Time := ⟨12, 30⟩ -- 12:30 PM (9:00 AM + 3h30m)
  
  (timeDifference firstDeparture expressDeparture / departureInterval + 1) -
  (timeDifference firstDeparture expressArrival / departureInterval + 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_express_train_meetings_l61_6182


namespace NUMINAMATH_CALUDE_family_weight_ratio_l61_6196

/-- Given the weights of three generations in a family, prove the ratio of the child's weight to the grandmother's weight -/
theorem family_weight_ratio :
  ∀ (grandmother daughter child : ℝ),
  grandmother + daughter + child = 160 →
  daughter + child = 60 →
  daughter = 40 →
  child / grandmother = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_family_weight_ratio_l61_6196


namespace NUMINAMATH_CALUDE_sequence_and_inequality_problem_l61_6128

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

def positive_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n > 0

theorem sequence_and_inequality_problem
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_pos : positive_sequence b)
  (h_a1 : a 1 = 2)
  (h_b1 : b 1 = 3)
  (h_sum1 : a 3 + b 5 = 56)
  (h_sum2 : a 5 + b 3 = 26)
  (h_ineq : ∀ n : ℕ, n > 0 → ∀ x : ℝ, -x^2 + 3*x ≤ (2 * b n) / (2 * ↑n + 1)) :
  (∀ n : ℕ, a n = 3 * ↑n - 1) ∧
  (∀ n : ℕ, b n = 3 * 2^(n-1)) ∧
  (∀ x : ℝ, (-x^2 + 3*x ≤ 2) ↔ (x ≥ 2 ∨ x ≤ 1)) :=
sorry

end NUMINAMATH_CALUDE_sequence_and_inequality_problem_l61_6128


namespace NUMINAMATH_CALUDE_alex_ate_six_ounces_l61_6189

/-- The amount of jelly beans Alex ate -/
def jelly_beans_eaten (initial : ℕ) (num_piles : ℕ) (weight_per_pile : ℕ) : ℕ :=
  initial - (num_piles * weight_per_pile)

/-- Theorem stating that Alex ate 6 ounces of jelly beans -/
theorem alex_ate_six_ounces : 
  jelly_beans_eaten 36 3 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_alex_ate_six_ounces_l61_6189


namespace NUMINAMATH_CALUDE_sum_of_ages_l61_6129

/-- Given the ages of a father and son, prove that their sum is 55 years. -/
theorem sum_of_ages (father_age son_age : ℕ) 
  (h1 : father_age = 37) 
  (h2 : son_age = 18) : 
  father_age + son_age = 55 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l61_6129


namespace NUMINAMATH_CALUDE_total_seashells_l61_6199

theorem total_seashells (day1 day2 day3 : ℕ) 
  (h1 : day1 = 27) 
  (h2 : day2 = 46) 
  (h3 : day3 = 19) : 
  day1 + day2 + day3 = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l61_6199


namespace NUMINAMATH_CALUDE_certain_number_problem_l61_6106

theorem certain_number_problem : 
  ∃ N : ℕ, (N / 5 + N + 5 = 65) ∧ (N = 50) :=
by sorry

end NUMINAMATH_CALUDE_certain_number_problem_l61_6106


namespace NUMINAMATH_CALUDE_discount_calculation_l61_6149

theorem discount_calculation (marked_price : ℝ) (discount_rate : ℝ) (num_articles : ℕ) 
  (h1 : marked_price = 15)
  (h2 : discount_rate = 0.4)
  (h3 : num_articles = 2) :
  marked_price * num_articles * (1 - discount_rate) = 18 :=
by sorry

end NUMINAMATH_CALUDE_discount_calculation_l61_6149


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l61_6153

theorem quadratic_factorization_sum (a b c : ℤ) :
  (∀ x, x^2 + 14*x + 45 = (x + a)*(x + b)) →
  (∀ x, x^2 - 19*x + 90 = (x - b)*(x - c)) →
  a + b + c = 24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l61_6153


namespace NUMINAMATH_CALUDE_range_of_k_for_two_roots_l61_6140

open Real

theorem range_of_k_for_two_roots (g : ℝ → ℝ) (k : ℝ) :
  (∀ x, g x = 2 * sin (2 * x - π / 6)) →
  (∀ x ∈ Set.Icc 0 (π / 2), (g x - k = 0 → ∃ y ∈ Set.Icc 0 (π / 2), x ≠ y ∧ g y - k = 0)) ↔
  k ∈ Set.Icc 1 2 ∧ k ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_for_two_roots_l61_6140


namespace NUMINAMATH_CALUDE_aaron_final_card_count_l61_6121

/-- Given that Aaron initially has 5 cards and finds 62 more cards,
    prove that Aaron ends up with 67 cards in total. -/
theorem aaron_final_card_count :
  let initial_cards : ℕ := 5
  let found_cards : ℕ := 62
  initial_cards + found_cards = 67 :=
by sorry

end NUMINAMATH_CALUDE_aaron_final_card_count_l61_6121


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l61_6166

/-- Parabola with vertex at origin and directrix x = -1 -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  vertex_at_origin : equation 0 0
  directrix : ℝ → Prop
  directrix_eq : ∀ x, directrix x ↔ x = -1

/-- Line passing through two points on the parabola -/
structure IntersectingLine (p : Parabola) where
  equation : ℝ → ℝ → Prop
  passes_through_focus : equation 1 0
  intersects_parabola : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    equation x₁ y₁ ∧ p.equation x₁ y₁ ∧
    equation x₂ y₂ ∧ p.equation x₂ y₂ ∧
    x₁ ≠ x₂
  midpoint_x_coord : ℝ
  midpoint_condition : ∀ (x₁ y₁ x₂ y₂ : ℝ),
    equation x₁ y₁ ∧ p.equation x₁ y₁ ∧
    equation x₂ y₂ ∧ p.equation x₂ y₂ ∧
    x₁ ≠ x₂ →
    (x₁ + x₂) / 2 = midpoint_x_coord

/-- Main theorem about the parabola and intersecting line -/
theorem parabola_and_line_properties (p : Parabola) (l : IntersectingLine p) 
    (h_midpoint : l.midpoint_x_coord = 2) :
  (∀ x y, p.equation x y ↔ y^2 = 4*x) ∧
  (∀ x y, l.equation x y ↔ (y = Real.sqrt 2 * x - Real.sqrt 2 ∨ 
                            y = -Real.sqrt 2 * x + Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l61_6166


namespace NUMINAMATH_CALUDE_rectangle_area_change_l61_6154

theorem rectangle_area_change (original_area : ℝ) : 
  original_area = 540 →
  (0.9 * 1.2 * original_area : ℝ) = 583.2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l61_6154
