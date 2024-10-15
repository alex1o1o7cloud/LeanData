import Mathlib

namespace NUMINAMATH_CALUDE_base9_to_base10_653_l711_71133

/-- Converts a base-9 number to base 10 --/
def base9_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9^i)) 0

/-- The base-9 representation of the number --/
def base9_number : List Nat := [3, 5, 6]

theorem base9_to_base10_653 :
  base9_to_base10 base9_number = 534 := by
  sorry

end NUMINAMATH_CALUDE_base9_to_base10_653_l711_71133


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l711_71111

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 2 ∧ x₂ = 2 - Real.sqrt 2 ∧
  (x₁^2 - 4*x₁ + 2 = 0) ∧ (x₂^2 - 4*x₂ + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l711_71111


namespace NUMINAMATH_CALUDE_number_added_to_2q_l711_71144

theorem number_added_to_2q (x y q : ℤ) (some_number : ℤ) : 
  x = some_number + 2 * q →
  y = 4 * q + 41 →
  (q = 7 → x = y) →
  some_number = 55 := by
sorry

end NUMINAMATH_CALUDE_number_added_to_2q_l711_71144


namespace NUMINAMATH_CALUDE_digit_puzzle_l711_71157

theorem digit_puzzle (c o u n t s : ℕ) 
  (h1 : c + o = u)
  (h2 : u + n = t)
  (h3 : t + c = s)
  (h4 : o + n + s = 12)
  (h5 : c ≠ 0)
  (h6 : o ≠ 0)
  (h7 : u ≠ 0)
  (h8 : n ≠ 0)
  (h9 : t ≠ 0)
  (h10 : s ≠ 0) :
  t = 6 := by
sorry

end NUMINAMATH_CALUDE_digit_puzzle_l711_71157


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l711_71112

/-- Given a geometric sequence {a_n} with sum S_n of the first n terms,
    if a_3 + 2a_6 = 0, then S_3/S_6 = 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- Geometric sequence condition
  (∀ n : ℕ, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- Sum formula
  a 3 + 2 * a 6 = 0 →  -- Given condition
  S 3 / S 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l711_71112


namespace NUMINAMATH_CALUDE_lashawn_double_kymbrea_after_25_months_l711_71151

/-- Represents the number of comic books in a collection after a given number of months. -/
def comic_books (initial : ℕ) (rate : ℕ) (months : ℕ) : ℕ :=
  initial + rate * months

theorem lashawn_double_kymbrea_after_25_months :
  let kymbrea_initial := 30
  let kymbrea_rate := 2
  let lashawn_initial := 10
  let lashawn_rate := 6
  let months := 25
  comic_books lashawn_initial lashawn_rate months = 
    2 * comic_books kymbrea_initial kymbrea_rate months := by
  sorry

end NUMINAMATH_CALUDE_lashawn_double_kymbrea_after_25_months_l711_71151


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l711_71136

/-- An isosceles right triangle with leg length 6 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  leg_length_eq : leg_length = 6

/-- A square inscribed in the triangle with one vertex at the right angle -/
structure InscribedSquareA (triangle : IsoscelesRightTriangle) where
  side_length : ℝ
  vertex_at_right_angle : True
  side_along_leg : True

/-- A square inscribed in the triangle with one side along the other leg -/
structure InscribedSquareB (triangle : IsoscelesRightTriangle) where
  side_length : ℝ
  side_along_leg : True

/-- The theorem statement -/
theorem inscribed_squares_ratio 
  (triangle : IsoscelesRightTriangle) 
  (square_a : InscribedSquareA triangle) 
  (square_b : InscribedSquareB triangle) : 
  square_a.side_length / square_b.side_length = 1 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l711_71136


namespace NUMINAMATH_CALUDE_q_n_limit_zero_l711_71113

def q_n (n : ℕ+) : ℕ := Nat.minFac (n + 1)

theorem q_n_limit_zero : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n : ℕ+, n.val > N → (q_n n : ℝ) / n.val < ε :=
sorry

end NUMINAMATH_CALUDE_q_n_limit_zero_l711_71113


namespace NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_64_times_sqrt_16_l711_71142

theorem fourth_root_256_times_cube_root_64_times_sqrt_16 : 
  (256 : ℝ) ^ (1/4) * (64 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/2) = 32 := by sorry

end NUMINAMATH_CALUDE_fourth_root_256_times_cube_root_64_times_sqrt_16_l711_71142


namespace NUMINAMATH_CALUDE_arithmetic_geometric_subset_l711_71127

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  a3_eq_3 : a 3 = 3
  a5_eq_6 : a 5 = 6
  geometric_subset : ∃ m, (a 3) * (a m) = (a 5)^2

/-- The theorem stating that m = 9 for the given conditions -/
theorem arithmetic_geometric_subset (seq : ArithmeticSequence) :
  ∃ m, seq.a m = 12 ∧ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_subset_l711_71127


namespace NUMINAMATH_CALUDE_ln_inequality_l711_71138

-- Define the natural logarithm function
noncomputable def f (x : ℝ) := Real.log x

-- State the theorem
theorem ln_inequality (x : ℝ) (h : x > 0) : f x ≤ x - 1 := by
  -- Define the derivative of f
  have f_deriv : ∀ x > 0, deriv f x = 1 / x := by sorry
  
  -- f(1) = 0
  have f_at_one : f 1 = 0 := by sorry
  
  -- The tangent line at x = 1 is y = x - 1
  have tangent_line : ∀ x, x - 1 = (x - 1) * (deriv f 1) + f 1 := by sorry
  
  -- The tangent line is above the graph of f for x > 0
  have tangent_above : ∀ x > 0, f x ≤ x - 1 := by sorry
  
  -- Apply the tangent_above property to prove the inequality
  exact tangent_above x h

end NUMINAMATH_CALUDE_ln_inequality_l711_71138


namespace NUMINAMATH_CALUDE_hundredthDigitOf7Over33_l711_71117

-- Define the fraction
def f : ℚ := 7 / 33

-- Define a function to get the nth digit after the decimal point
noncomputable def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem hundredthDigitOf7Over33 : nthDigitAfterDecimal f 100 = 1 := by sorry

end NUMINAMATH_CALUDE_hundredthDigitOf7Over33_l711_71117


namespace NUMINAMATH_CALUDE_shorter_diagonal_length_l711_71118

/-- Represents a trapezoid EFGH with given properties -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  side1 : ℝ
  side2 : ℝ
  acute_angles : Bool

/-- The shorter diagonal of the trapezoid -/
def shorter_diagonal (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that for a trapezoid with specific measurements, 
    the shorter diagonal has length 27 -/
theorem shorter_diagonal_length :
  ∀ t : Trapezoid, 
    t.EF = 40 ∧ 
    t.GH = 28 ∧ 
    t.side1 = 13 ∧ 
    t.side2 = 15 ∧ 
    t.acute_angles = true →
    shorter_diagonal t = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_shorter_diagonal_length_l711_71118


namespace NUMINAMATH_CALUDE_max_pairs_after_loss_l711_71129

/-- Represents the types of shoes -/
inductive ShoeType
| Sneaker
| Sandal
| Boot

/-- Represents the colors of shoes -/
inductive ShoeColor
| Red
| Blue
| Green
| Black

/-- Represents the sizes of shoes -/
inductive ShoeSize
| Size6
| Size7
| Size8

/-- Represents a shoe with its type, color, and size -/
structure Shoe :=
  (type : ShoeType)
  (color : ShoeColor)
  (size : ShoeSize)

/-- Represents the initial collection of shoes -/
def initial_collection : Finset Shoe := sorry

/-- The number of shoes lost -/
def shoes_lost : Nat := 9

/-- Theorem stating the maximum number of complete pairs after losing shoes -/
theorem max_pairs_after_loss :
  ∃ (remaining_collection : Finset Shoe),
    remaining_collection ⊆ initial_collection ∧
    (initial_collection.card - remaining_collection.card = shoes_lost) ∧
    (∀ (s : Shoe), s ∈ remaining_collection →
      ∃ (s' : Shoe), s' ∈ remaining_collection ∧ s ≠ s' ∧
        s.type = s'.type ∧ s.color = s'.color ∧ s.size = s'.size) ∧
    remaining_collection.card = 36 := by
  sorry

end NUMINAMATH_CALUDE_max_pairs_after_loss_l711_71129


namespace NUMINAMATH_CALUDE_parallelogram_height_l711_71150

/-- The height of a parallelogram with given area and base -/
theorem parallelogram_height (area base height : ℝ) 
  (h_area : area = 864) 
  (h_base : base = 36) 
  (h_formula : area = base * height) : 
  height = 24 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l711_71150


namespace NUMINAMATH_CALUDE_expression_evaluation_l711_71181

theorem expression_evaluation : 3^(0^(2^11)) + ((3^0)^2)^11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l711_71181


namespace NUMINAMATH_CALUDE_min_distance_to_one_l711_71122

open Complex

/-- Given a complex number z satisfying the equation, the minimum value of |z - 1| is √2 -/
theorem min_distance_to_one (z : ℂ) 
  (h : Complex.abs ((z^2 + 1) / (z + I)) + Complex.abs ((z^2 + 4*I - 3) / (z - I + 2)) = 4) :
  ∃ (min_dist : ℝ), (∀ (w : ℂ), Complex.abs ((w^2 + 1) / (w + I)) + Complex.abs ((w^2 + 4*I - 3) / (w - I + 2)) = 4 → 
    Complex.abs (w - 1) ≥ min_dist) ∧ min_dist = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_one_l711_71122


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reverse_composite_l711_71163

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ d ∣ n

def reverse_digits (n : ℕ) : ℕ :=
  let ones := n % 10
  let tens := n / 10
  ones * 10 + tens

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def tens_digit (n : ℕ) : ℕ := n / 10

theorem smallest_two_digit_prime_with_reverse_composite :
  ∃ (n : ℕ), 
    is_two_digit n ∧ 
    is_prime n ∧ 
    tens_digit n = 2 ∧ 
    is_composite (reverse_digits n) ∧
    (∀ m : ℕ, is_two_digit m → is_prime m → tens_digit m = 2 → 
      is_composite (reverse_digits m) → n ≤ m) ∧
    n = 23 := by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reverse_composite_l711_71163


namespace NUMINAMATH_CALUDE_trapezium_height_l711_71146

theorem trapezium_height (a b area : ℝ) (ha : a > 0) (hb : b > 0) (harea : area > 0) :
  a = 4 → b = 5 → area = 27 →
  (area = (a + b) * (area / ((a + b) / 2)) / 2) →
  area / ((a + b) / 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_height_l711_71146


namespace NUMINAMATH_CALUDE_train_distance_theorem_l711_71116

/-- The distance between two trains after 8 hours of travel, given their initial positions and speeds -/
def distance_between_trains (initial_distance : ℝ) (speed1 speed2 : ℝ) (time : ℝ) : Set ℝ :=
  let distance1 := speed1 * time
  let distance2 := speed2 * time
  let diff := distance2 - distance1
  {initial_distance + diff, initial_distance - diff}

/-- Theorem stating the distance between two trains after 8 hours -/
theorem train_distance_theorem :
  distance_between_trains 892 40 48 8 = {956, 828} :=
by sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l711_71116


namespace NUMINAMATH_CALUDE_triangle_problem_l711_71101

/-- Given a triangle ABC with the specified properties, prove AC = 5 and ∠A = 120° --/
theorem triangle_problem (A B C : ℝ) (BC AB AC : ℝ) (angleA : ℝ) :
  BC = 7 →
  AB = 3 →
  (Real.sin C) / (Real.sin B) = 3/5 →
  AC = 5 ∧ angleA = 120 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l711_71101


namespace NUMINAMATH_CALUDE_hyperbola_condition_roots_or_hyperbola_condition_l711_71187

-- Define the conditions
def has_two_distinct_positive_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
  x₁^2 + 2*m*x₁ + (m+2) = 0 ∧ x₂^2 + 2*m*x₂ + (m+2) = 0

def is_hyperbola_with_foci_on_y_axis (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2/(m+3) - y^2/(2*m-1) = 1 → 
  (m+3 < 0 ∧ 2*m-1 > 0)

-- Theorem statements
theorem hyperbola_condition (m : ℝ) : 
  is_hyperbola_with_foci_on_y_axis m → m < -3 :=
sorry

theorem roots_or_hyperbola_condition (m : ℝ) :
  (has_two_distinct_positive_roots m ∨ is_hyperbola_with_foci_on_y_axis m) ∧
  ¬(has_two_distinct_positive_roots m ∧ is_hyperbola_with_foci_on_y_axis m) →
  (-2 < m ∧ m < -1) ∨ m < -3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_roots_or_hyperbola_condition_l711_71187


namespace NUMINAMATH_CALUDE_min_value_problem_l711_71125

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 9/b = 6) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 9/y = 6 → (a + 1) * (b + 9) ≤ (x + 1) * (y + 9) ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 6 ∧ (x + 1) * (y + 9) = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l711_71125


namespace NUMINAMATH_CALUDE_sqrt_inequality_l711_71135

theorem sqrt_inequality (x : ℝ) : 0 < x → (Real.sqrt (x + 1) < 3 * x - 2 ↔ x > 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l711_71135


namespace NUMINAMATH_CALUDE_easter_egg_arrangement_l711_71188

theorem easter_egg_arrangement (yellow_eggs : Nat) (blue_eggs : Nat) 
  (min_eggs_per_basket : Nat) (min_baskets : Nat) :
  yellow_eggs = 30 →
  blue_eggs = 42 →
  min_eggs_per_basket = 6 →
  min_baskets = 3 →
  ∃ (eggs_per_basket : Nat),
    eggs_per_basket ≥ min_eggs_per_basket ∧
    eggs_per_basket ∣ yellow_eggs ∧
    eggs_per_basket ∣ blue_eggs ∧
    yellow_eggs / eggs_per_basket ≥ min_baskets ∧
    blue_eggs / eggs_per_basket ≥ min_baskets ∧
    ∀ (n : Nat),
      n > eggs_per_basket →
      ¬(n ≥ min_eggs_per_basket ∧
        n ∣ yellow_eggs ∧
        n ∣ blue_eggs ∧
        yellow_eggs / n ≥ min_baskets ∧
        blue_eggs / n ≥ min_baskets) :=
by
  sorry

end NUMINAMATH_CALUDE_easter_egg_arrangement_l711_71188


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l711_71124

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ t : ℝ, m * t^2 - m * t + 4 > 0) ↔ (0 ≤ m ∧ m < 16) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l711_71124


namespace NUMINAMATH_CALUDE_x_leq_y_l711_71130

theorem x_leq_y (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  Real.sqrt ((a - b) * (b - c)) ≤ (a - c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_leq_y_l711_71130


namespace NUMINAMATH_CALUDE_gcf_36_54_l711_71107

theorem gcf_36_54 : Nat.gcd 36 54 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcf_36_54_l711_71107


namespace NUMINAMATH_CALUDE_privateer_overtakes_at_6_08_pm_l711_71182

/-- Represents the chase scenario between a privateer and a merchantman -/
structure ChaseScenario where
  initial_distance : ℝ
  privateer_initial_speed : ℝ
  merchantman_speed : ℝ
  time_before_damage : ℝ
  new_speed_ratio_privateer : ℝ
  new_speed_ratio_merchantman : ℝ

/-- Calculates the time when the privateer overtakes the merchantman -/
def overtake_time (scenario : ChaseScenario) : ℝ :=
  sorry

/-- Theorem stating that given the specific chase scenario, the privateer overtakes the merchantman at 6:08 p.m. -/
theorem privateer_overtakes_at_6_08_pm (scenario : ChaseScenario) 
  (h1 : scenario.initial_distance = 12)
  (h2 : scenario.privateer_initial_speed = 10)
  (h3 : scenario.merchantman_speed = 7)
  (h4 : scenario.time_before_damage = 3)
  (h5 : scenario.new_speed_ratio_privateer = 13)
  (h6 : scenario.new_speed_ratio_merchantman = 12) :
  overtake_time scenario = 8.1333333333 :=
  sorry

#eval 10 + 8.1333333333  -- Should output approximately 18.1333333333, representing 6:08 p.m.

end NUMINAMATH_CALUDE_privateer_overtakes_at_6_08_pm_l711_71182


namespace NUMINAMATH_CALUDE_netGainDifference_l711_71169

/-- Represents a job candidate with their associated costs and revenue --/
structure Candidate where
  salary : ℕ
  revenue : ℕ
  trainingMonths : ℕ
  trainingCostPerMonth : ℕ
  hiringBonusPercent : ℕ

/-- Calculates the net gain for the company from a candidate --/
def netGain (c : Candidate) : ℕ :=
  c.revenue - c.salary - (c.trainingMonths * c.trainingCostPerMonth) - (c.salary * c.hiringBonusPercent / 100)

/-- The two candidates as described in the problem --/
def candidate1 : Candidate :=
  { salary := 42000
    revenue := 93000
    trainingMonths := 3
    trainingCostPerMonth := 1200
    hiringBonusPercent := 0 }

def candidate2 : Candidate :=
  { salary := 45000
    revenue := 92000
    trainingMonths := 0
    trainingCostPerMonth := 0
    hiringBonusPercent := 1 }

/-- Theorem stating the difference in net gain between the two candidates --/
theorem netGainDifference : netGain candidate1 - netGain candidate2 = 850 := by
  sorry

end NUMINAMATH_CALUDE_netGainDifference_l711_71169


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_2009_l711_71100

/-- Define the sequence a_n -/
def a_seq (a : ℕ+) : ℕ → ℕ
  | 0 => a
  | n + 1 => a_seq a n + 40^(Nat.factorial (n + 1))

/-- Theorem: The sequence a_n has infinitely many numbers divisible by 2009 -/
theorem infinitely_many_divisible_by_2009 (a : ℕ+) :
  ∀ k : ℕ, ∃ n > k, 2009 ∣ a_seq a n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_2009_l711_71100


namespace NUMINAMATH_CALUDE_trig_inequality_l711_71134

theorem trig_inequality (x y z : ℝ) (h1 : 0 < z) (h2 : z < y) (h3 : y < x) (h4 : x < π/2) :
  π/2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l711_71134


namespace NUMINAMATH_CALUDE_volume_specific_pyramid_l711_71190

/-- A triangular pyramid with specific edge lengths -/
structure TriangularPyramid where
  edge_opposite1 : ℝ
  edge_opposite2 : ℝ
  edge_other : ℝ

/-- Volume of a triangular pyramid -/
def volume (p : TriangularPyramid) : ℝ := sorry

/-- Theorem: The volume of the specific triangular pyramid is 24 cm³ -/
theorem volume_specific_pyramid :
  let p : TriangularPyramid := {
    edge_opposite1 := 4,
    edge_opposite2 := 12,
    edge_other := 7
  }
  volume p = 24 := by sorry

end NUMINAMATH_CALUDE_volume_specific_pyramid_l711_71190


namespace NUMINAMATH_CALUDE_one_third_between_one_fourth_one_sixth_l711_71198

/-- The fraction one-third of the way from a to b -/
def one_third_between (a b : ℚ) : ℚ := (2 * a + b) / 3

/-- Prove that the fraction one-third of the way from 1/4 to 1/6 is equal to 2/9 -/
theorem one_third_between_one_fourth_one_sixth :
  one_third_between (1/4) (1/6) = 2/9 := by sorry

end NUMINAMATH_CALUDE_one_third_between_one_fourth_one_sixth_l711_71198


namespace NUMINAMATH_CALUDE_evaluate_expression_l711_71115

theorem evaluate_expression : 3^0 + 9^5 / 9^3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l711_71115


namespace NUMINAMATH_CALUDE_range_of_a_l711_71199

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, |x + a| ≤ 2) → a ∈ Set.Icc (-3) 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l711_71199


namespace NUMINAMATH_CALUDE_a_value_m_minimum_l711_71194

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}

-- Theorem 1: Prove that a = 1
theorem a_value : 
  ∀ x ∈ solution_set 1, f 1 x ≤ 6 ∧
  ∀ a : ℝ, (∀ x ∈ solution_set a, f a x ≤ 6) → a = 1 :=
sorry

-- Define the function g (which is f with a = 1)
def g (x : ℝ) : ℝ := |2*x - 1| + 1

-- Theorem 2: Prove that the minimum value of m is 3.5
theorem m_minimum :
  (∃ m : ℝ, ∀ t : ℝ, g (t/2) ≤ m - g (-t)) ∧
  (∀ m : ℝ, (∀ t : ℝ, g (t/2) ≤ m - g (-t)) → m ≥ 3.5) :=
sorry

end NUMINAMATH_CALUDE_a_value_m_minimum_l711_71194


namespace NUMINAMATH_CALUDE_min_difference_theorem_l711_71159

noncomputable section

def f (x : ℝ) : ℝ := Real.exp (2 * x)
def g (x : ℝ) : ℝ := Real.log x + 1/2

theorem min_difference_theorem :
  ∃ (h : ℝ → ℝ), ∀ (x₁ : ℝ),
    (∃ (x₂ : ℝ), x₂ > 0 ∧ f x₁ = g x₂) ∧
    (∀ (x₂ : ℝ), x₂ > 0 → f x₁ = g x₂ → h x₁ ≤ x₂ - x₁) ∧
    (∃ (x₁ x₂ : ℝ), x₂ > 0 ∧ f x₁ = g x₂ ∧ h x₁ = x₂ - x₁) ∧
    (∀ (x : ℝ), h x = 1 + Real.log 2 / 2) :=
sorry

end

end NUMINAMATH_CALUDE_min_difference_theorem_l711_71159


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_2007_l711_71128

/-- The area of a quadrilateral with vertices at (1, 3), (1, 1), (3, 1), and (2007, 2008) -/
def quadrilateral_area : ℝ :=
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (3, 1)
  let D : ℝ × ℝ := (2007, 2008)
  -- Area calculation goes here
  0  -- Placeholder, replace with actual calculation

/-- Theorem stating that the area of the quadrilateral is 2007 square units -/
theorem quadrilateral_area_is_2007 : quadrilateral_area = 2007 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_2007_l711_71128


namespace NUMINAMATH_CALUDE_tarantula_legs_tarantula_leg_count_l711_71126

/-- The number of tarantulas in one egg sac -/
def tarantulas_per_sac : ℕ := 1000

/-- The number of baby tarantula legs in one less than 5 egg sacs -/
def total_legs : ℕ := 32000

/-- The number of egg sacs containing the total legs -/
def num_sacs : ℕ := 5 - 1

/-- Proves that a tarantula has 8 legs -/
theorem tarantula_legs : ℕ :=
  8

/-- Proves that the number of legs a tarantula has is 8 -/
theorem tarantula_leg_count : tarantula_legs = 8 := by
  sorry

end NUMINAMATH_CALUDE_tarantula_legs_tarantula_leg_count_l711_71126


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l711_71196

theorem smallest_positive_integer_congruence :
  ∃ (y : ℕ), y > 0 ∧ y + 3721 ≡ 803 [ZMOD 17] ∧
  ∀ (z : ℕ), z > 0 ∧ z + 3721 ≡ 803 [ZMOD 17] → y ≤ z ∧ y = 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l711_71196


namespace NUMINAMATH_CALUDE_purple_cars_count_l711_71171

theorem purple_cars_count (total : ℕ) (blue red orange yellow purple green : ℕ) : 
  total = 1423 →
  blue = 2 * red →
  red = 3 * orange →
  yellow = orange / 2 →
  yellow = 3 * purple →
  green = 5 * purple →
  blue ≥ 200 →
  red ≥ 50 →
  total = blue + red + orange + yellow + purple + green →
  purple = 20 := by
  sorry

#check purple_cars_count

end NUMINAMATH_CALUDE_purple_cars_count_l711_71171


namespace NUMINAMATH_CALUDE_two_guests_mixed_probability_l711_71189

/-- The number of guests -/
def num_guests : ℕ := 3

/-- The number of pastry types -/
def num_pastry_types : ℕ := 3

/-- The total number of pastries -/
def total_pastries : ℕ := num_guests * num_pastry_types

/-- The number of pastries each guest receives -/
def pastries_per_guest : ℕ := num_pastry_types

/-- The probability of exactly two guests receiving one of each type of pastry -/
def probability_two_guests_mixed : ℚ := 27 / 280

theorem two_guests_mixed_probability :
  probability_two_guests_mixed = 27 / 280 := by
  sorry

end NUMINAMATH_CALUDE_two_guests_mixed_probability_l711_71189


namespace NUMINAMATH_CALUDE_student_arrangements_l711_71168

/-- The number of students in the row -/
def n : ℕ := 7

/-- The number of arrangements where students A and B must stand next to each other -/
def arrangements_adjacent : ℕ := 1440

/-- The number of arrangements where students A, B, and C must not stand next to each other -/
def arrangements_not_adjacent : ℕ := 1440

/-- The number of arrangements where student A is not at the head and student B is not at the tail -/
def arrangements_not_head_tail : ℕ := 3720

theorem student_arrangements :
  (arrangements_adjacent = 1440) ∧
  (arrangements_not_adjacent = 1440) ∧
  (arrangements_not_head_tail = 3720) := by
  sorry

end NUMINAMATH_CALUDE_student_arrangements_l711_71168


namespace NUMINAMATH_CALUDE_fraction_sum_equals_2315_over_1200_l711_71178

theorem fraction_sum_equals_2315_over_1200 :
  (1/2 : ℚ) * (3/4) + (5/6) * (7/8) + (9/10) * (11/12) = 2315/1200 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_2315_over_1200_l711_71178


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l711_71141

theorem merry_go_round_revolutions 
  (r₁ : ℝ) (r₂ : ℝ) (rev₁ : ℝ) 
  (h₁ : r₁ = 36) 
  (h₂ : r₂ = 12) 
  (h₃ : rev₁ = 40) 
  (h₄ : r₁ > 0) 
  (h₅ : r₂ > 0) : 
  ∃ rev₂ : ℝ, rev₂ * r₂ = rev₁ * r₁ ∧ rev₂ = 120 :=
sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l711_71141


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l711_71106

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a n > 0 ∧ a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : is_positive_geometric_sequence a)
  (h_prod : a 2 * a 8 = 4) :
  a 5 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l711_71106


namespace NUMINAMATH_CALUDE_grape_purchases_l711_71179

theorem grape_purchases (lena_shown ira_shown combined_shown : ℝ)
  (h1 : lena_shown = 2)
  (h2 : ira_shown = 3)
  (h3 : combined_shown = 4.5) :
  ∃ (lena_actual ira_actual offset : ℝ),
    lena_actual + offset = lena_shown ∧
    ira_actual + offset = ira_shown ∧
    lena_actual + ira_actual = combined_shown ∧
    lena_actual = 1.5 ∧
    ira_actual = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_grape_purchases_l711_71179


namespace NUMINAMATH_CALUDE_w_over_y_value_l711_71109

theorem w_over_y_value (w x y : ℝ) 
  (h1 : w / x = 2 / 3)
  (h2 : (x + y) / y = 1.6) :
  w / y = 0.4 := by
sorry

end NUMINAMATH_CALUDE_w_over_y_value_l711_71109


namespace NUMINAMATH_CALUDE_ryan_english_hours_l711_71186

/-- Ryan's daily study schedule -/
structure StudySchedule where
  english_hours : ℕ
  chinese_hours : ℕ
  chinese_more_than_english : chinese_hours = english_hours + 1

/-- Ryan's actual study schedule -/
def ryans_schedule : StudySchedule where
  english_hours := 6
  chinese_hours := 7
  chinese_more_than_english := by rfl

theorem ryan_english_hours :
  ryans_schedule.english_hours = 6 :=
by sorry

end NUMINAMATH_CALUDE_ryan_english_hours_l711_71186


namespace NUMINAMATH_CALUDE_distance_after_walking_l711_71121

/-- The distance between two people walking in opposite directions for 1.5 hours -/
theorem distance_after_walking (jay_speed : ℝ) (paul_speed : ℝ) (time : ℝ) : 
  jay_speed = 0.75 * (60 / 15) →  -- Jay's speed in miles per hour
  paul_speed = 2.5 * (60 / 30) →  -- Paul's speed in miles per hour
  time = 1.5 →                    -- Time in hours
  jay_speed * time + paul_speed * time = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_walking_l711_71121


namespace NUMINAMATH_CALUDE_solve_jump_rope_problem_l711_71166

def jump_rope_problem (cindy_time betsy_time tina_time : ℝ) : Prop :=
  cindy_time = 12 ∧
  tina_time = 3 * betsy_time ∧
  tina_time = cindy_time + 6 ∧
  betsy_time / cindy_time = 1 / 2

theorem solve_jump_rope_problem :
  ∃ (betsy_time tina_time : ℝ),
    jump_rope_problem 12 betsy_time tina_time :=
by
  sorry

end NUMINAMATH_CALUDE_solve_jump_rope_problem_l711_71166


namespace NUMINAMATH_CALUDE_rock_collecting_contest_l711_71114

/-- The rock collecting contest between Sydney and Conner --/
theorem rock_collecting_contest 
  (sydney_initial : ℕ) 
  (conner_initial : ℕ) 
  (sydney_day1 : ℕ) 
  (conner_day1_multiplier : ℕ) 
  (conner_day2 : ℕ) 
  (sydney_day3_multiplier : ℕ) 
  (h1 : sydney_initial = 837) 
  (h2 : conner_initial = 723) 
  (h3 : sydney_day1 = 4) 
  (h4 : conner_day1_multiplier = 8) 
  (h5 : conner_day2 = 123) 
  (h6 : sydney_day3_multiplier = 2) : 
  ∃ (conner_day3 : ℕ), conner_day3 ≥ 27 ∧ 
    conner_initial + conner_day1_multiplier * sydney_day1 + conner_day2 + conner_day3 ≥ 
    sydney_initial + sydney_day1 + sydney_day3_multiplier * (conner_day1_multiplier * sydney_day1) :=
by
  sorry

end NUMINAMATH_CALUDE_rock_collecting_contest_l711_71114


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l711_71137

/-- A rectangular prism with length 4, width 3, and height 2 -/
structure RectangularPrism where
  length : ℕ := 4
  width : ℕ := 3
  height : ℕ := 2

/-- The number of diagonals in a rectangular prism -/
def num_diagonals (prism : RectangularPrism) : ℕ := sorry

/-- Theorem stating that a rectangular prism with length 4, width 3, and height 2 has 16 diagonals -/
theorem rectangular_prism_diagonals :
  ∀ (prism : RectangularPrism), num_diagonals prism = 16 := by sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l711_71137


namespace NUMINAMATH_CALUDE_negation_of_forall_not_prime_l711_71191

theorem negation_of_forall_not_prime :
  (¬ (∀ n : ℕ, ¬ (Nat.Prime (2^n - 2)))) ↔ (∃ n : ℕ, Nat.Prime (2^n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_not_prime_l711_71191


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l711_71177

theorem partial_fraction_decomposition_product (x : ℝ) 
  (A B C : ℝ) : 
  (x^2 - 25) / ((x - 2) * (x + 3) * (x - 4)) = 
    A / (x - 2) + B / (x + 3) + C / (x - 4) → 
  A * B * C = 1 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l711_71177


namespace NUMINAMATH_CALUDE_sum_first_ten_primes_with_units_digit_3_l711_71184

def is_prime (n : ℕ) : Prop := sorry

def has_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def first_ten_primes_with_units_digit_3 : List ℕ := sorry

theorem sum_first_ten_primes_with_units_digit_3 :
  (first_ten_primes_with_units_digit_3.foldl (· + ·) 0) = 793 := by sorry

end NUMINAMATH_CALUDE_sum_first_ten_primes_with_units_digit_3_l711_71184


namespace NUMINAMATH_CALUDE_auto_shop_discount_l711_71119

theorem auto_shop_discount (part_cost : ℕ) (num_parts : ℕ) (total_discount : ℕ) : 
  part_cost = 80 → num_parts = 7 → total_discount = 121 → 
  part_cost * num_parts - total_discount = 439 := by
  sorry

end NUMINAMATH_CALUDE_auto_shop_discount_l711_71119


namespace NUMINAMATH_CALUDE_quadratic_function_constraint_l711_71139

theorem quadratic_function_constraint (a b c : ℝ) : 
  (∃ a' ∈ Set.Icc 1 2, ∀ x ∈ Set.Icc 1 2, a' * x^2 + b * x + c ≤ 1) →
  (7 * b + 5 * c ≤ -6 ∧ ∃ b' c', 7 * b' + 5 * c' = -6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_constraint_l711_71139


namespace NUMINAMATH_CALUDE_mod_fifteen_equivalence_l711_71152

theorem mod_fifteen_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 14567 [ZMOD 15] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_fifteen_equivalence_l711_71152


namespace NUMINAMATH_CALUDE_point_coordinates_l711_71195

/-- If point A(a, a-2) lies on the x-axis, then the coordinates of point B(a+2, a-1) are (4, 1) -/
theorem point_coordinates (a : ℝ) :
  (a = 2) → (a + 2 = 4 ∧ a - 1 = 1) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_l711_71195


namespace NUMINAMATH_CALUDE_almost_square_quotient_l711_71197

/-- Definition of an almost square -/
def AlmostSquare (k : ℕ) : Prop := ∃ n : ℕ, k = n * (n + 1)

/-- Theorem: Every almost square can be expressed as a quotient of two almost squares -/
theorem almost_square_quotient (n : ℕ) : 
  ∃ a b : ℕ, AlmostSquare a ∧ AlmostSquare b ∧ n * (n + 1) = a / b := by
  sorry

end NUMINAMATH_CALUDE_almost_square_quotient_l711_71197


namespace NUMINAMATH_CALUDE_find_a_value_l711_71145

theorem find_a_value (x : ℝ) (a : ℝ) : 
  (2 * x - 3 = 5 * x - 2 * a) → (x = 1) → (a = 3) := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l711_71145


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l711_71185

/-- The distance between the foci of the ellipse 9x^2 + 16y^2 = 144 is 2√7 -/
theorem ellipse_foci_distance :
  let a : ℝ := 4
  let b : ℝ := 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  (∀ x y : ℝ, 9 * x^2 + 16 * y^2 = 144) →
  2 * c = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l711_71185


namespace NUMINAMATH_CALUDE_f_evaluation_l711_71149

def f (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x - 7

theorem f_evaluation : 2 * f 2 + 3 * f (-2) = -107 := by
  sorry

end NUMINAMATH_CALUDE_f_evaluation_l711_71149


namespace NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l711_71193

def not_in_second_quadrant (m n : ℝ) : Prop :=
  (m / n > 0) ∧ (1 / n < 0)

theorem mn_positive_necessary_not_sufficient (m n : ℝ) :
  (not_in_second_quadrant m n → m * n > 0) ∧
  ¬(m * n > 0 → not_in_second_quadrant m n) :=
sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l711_71193


namespace NUMINAMATH_CALUDE_inequality_proof_l711_71176

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a^2 + 1 / b^2 + a * b ≥ 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l711_71176


namespace NUMINAMATH_CALUDE_election_majority_proof_l711_71110

/-- 
In an election with a total of 4500 votes, where the winning candidate receives 60% of the votes,
prove that the majority of votes by which the candidate won is 900.
-/
theorem election_majority_proof (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 4500 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).num / (winning_percentage * total_votes : ℚ).den -
  ((1 - winning_percentage) * total_votes : ℚ).num / ((1 - winning_percentage) * total_votes : ℚ).den = 900 := by
  sorry

end NUMINAMATH_CALUDE_election_majority_proof_l711_71110


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l711_71103

theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (circle_eq : ∀ x y, x^2 + y^2 = c^2)
  (asymptote_eq : ∀ x, b / a * x = x)
  (point_M : ∃ x y, x^2 + y^2 = c^2 ∧ y = b / a * x ∧ x = a ∧ y = b)
  (distance_condition : ∀ x y, x^2 + y^2 = c^2 ∧ y = b / a * x → 
    Real.sqrt ((x + c)^2 + y^2) - Real.sqrt ((x - c)^2 + y^2) = 2 * b)
  (relation_abc : b^2 = a^2 - c^2)
  (eccentricity_def : c / a = e) :
  e^2 = (Real.sqrt 5 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l711_71103


namespace NUMINAMATH_CALUDE_max_value_of_a_l711_71164

/-- An odd function that is increasing on the non-negative reals -/
structure OddIncreasingFunction (f : ℝ → ℝ) : Prop where
  odd : ∀ x, f (-x) = -f x
  increasing_nonneg : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

/-- The condition relating f, a, x, and t -/
def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x t, x ∈ Set.Icc 1 2 → t ∈ Set.Icc 1 2 →
    f (x^2 + a*x + a) ≤ f (-a*t^2 - t + 1)

theorem max_value_of_a (f : ℝ → ℝ) (hf : OddIncreasingFunction f) :
  (∃ a, condition f a) → (∀ a, condition f a → a ≤ -1) ∧ (condition f (-1)) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l711_71164


namespace NUMINAMATH_CALUDE_profit_maximization_l711_71104

/-- Represents the price reduction in yuan -/
def x : ℝ := 2.5

/-- Represents the initial selling price in yuan -/
def initial_price : ℝ := 60

/-- Represents the cost price in yuan -/
def cost_price : ℝ := 40

/-- Represents the initial weekly sales in items -/
def initial_sales : ℝ := 300

/-- Represents the increase in sales for each yuan of price reduction -/
def sales_increase_rate : ℝ := 20

/-- The profit function based on the price reduction x -/
def profit_function (x : ℝ) : ℝ :=
  (initial_price - x) * (initial_sales + sales_increase_rate * x) -
  cost_price * (initial_sales + sales_increase_rate * x)

/-- The maximum profit achieved -/
def max_profit : ℝ := 6125

theorem profit_maximization :
  profit_function x = max_profit ∧
  ∀ y, 0 ≤ y ∧ y < initial_price - cost_price →
    profit_function y ≤ max_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_maximization_l711_71104


namespace NUMINAMATH_CALUDE_original_price_from_loss_and_selling_price_l711_71158

/-- Proves that if an item is sold at a 20% loss for 960 units, then its original price was 1200 units. -/
theorem original_price_from_loss_and_selling_price 
  (loss_percentage : ℝ) 
  (selling_price : ℝ) : 
  loss_percentage = 20 → 
  selling_price = 960 → 
  (1 - loss_percentage / 100) * (selling_price / (1 - loss_percentage / 100)) = 1200 :=
by sorry

end NUMINAMATH_CALUDE_original_price_from_loss_and_selling_price_l711_71158


namespace NUMINAMATH_CALUDE_yellow_roses_count_l711_71108

/-- The number of yellow roses on the third rose bush -/
def yellow_roses : ℕ := 20

theorem yellow_roses_count :
  let red_roses : ℕ := 12
  let pink_roses : ℕ := 18
  let orange_roses : ℕ := 8
  let red_picked : ℕ := red_roses / 2
  let pink_picked : ℕ := pink_roses / 2
  let orange_picked : ℕ := orange_roses / 4
  let yellow_picked : ℕ := yellow_roses / 4
  let total_picked : ℕ := 22
  red_picked + pink_picked + orange_picked + yellow_picked = total_picked →
  yellow_roses = 20 := by
sorry

end NUMINAMATH_CALUDE_yellow_roses_count_l711_71108


namespace NUMINAMATH_CALUDE_probability_sum_13_l711_71160

def die1 : Finset ℕ := {1, 2, 3, 7, 8, 9}
def die2 : Finset ℕ := {4, 5, 6, 10, 11, 12}

def sumTo13 : Finset (ℕ × ℕ) :=
  (die1.product die2).filter (fun p => p.1 + p.2 = 13)

theorem probability_sum_13 :
  (sumTo13.card : ℚ) / ((die1.card * die2.card) : ℚ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_13_l711_71160


namespace NUMINAMATH_CALUDE_BaBr2_molecular_weight_l711_71175

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The molecular weight of BaBr2 in g/mol -/
def molecular_weight_BaBr2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Br

/-- Theorem stating that the molecular weight of BaBr2 is 297.13 g/mol -/
theorem BaBr2_molecular_weight : 
  molecular_weight_BaBr2 = 297.13 := by sorry

end NUMINAMATH_CALUDE_BaBr2_molecular_weight_l711_71175


namespace NUMINAMATH_CALUDE_range_of_m_l711_71192

theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b : ℝ, a > 0 → b > 0 → (1/a + 1/b) * Real.sqrt (a^2 + b^2) ≥ 2*m - 4) :
  m ≤ 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l711_71192


namespace NUMINAMATH_CALUDE_illuminated_area_ratio_l711_71153

theorem illuminated_area_ratio (r : ℝ) (h : r > 0) :
  let sphere_radius := r
  let light_distance := 3 * r
  let illuminated_area := 2 * Real.pi * r * (r - r / 4)
  let cone_base_radius := r / 4 * Real.sqrt 15
  let cone_slant_height := r * Real.sqrt 15
  let cone_lateral_area := Real.pi * cone_base_radius * cone_slant_height
  illuminated_area / cone_lateral_area = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_illuminated_area_ratio_l711_71153


namespace NUMINAMATH_CALUDE_johns_allowance_l711_71156

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℚ) : 
  (3/5 : ℚ) * A + (1/3 : ℚ) * (A - (3/5 : ℚ) * A) + (9/10 : ℚ) = A → A = (27/8 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l711_71156


namespace NUMINAMATH_CALUDE_unique_prime_in_range_l711_71165

theorem unique_prime_in_range : ∃! n : ℕ, 
  70 ≤ n ∧ n ≤ 90 ∧ 
  Nat.gcd n 15 = 5 ∧ 
  Nat.Prime n ∧
  n = 85 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_in_range_l711_71165


namespace NUMINAMATH_CALUDE_factorial_ratio_l711_71162

theorem factorial_ratio : Nat.factorial 11 / Nat.factorial 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l711_71162


namespace NUMINAMATH_CALUDE_square_area_is_26_l711_71180

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The square defined by its four vertices -/
structure Square where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Calculate the area of a square given its vertices -/
def squareArea (sq : Square) : ℝ := 
  let dx := sq.p.x - sq.q.x
  let dy := sq.p.y - sq.q.y
  (dx * dx + dy * dy)

/-- The theorem stating that the area of the given square is 26 -/
theorem square_area_is_26 : 
  let sq := Square.mk 
    (Point.mk 2 3) 
    (Point.mk (-3) 4) 
    (Point.mk (-2) 9) 
    (Point.mk 3 8)
  squareArea sq = 26 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_26_l711_71180


namespace NUMINAMATH_CALUDE_intersection_of_curves_l711_71143

/-- Prove that if a curve C₁ defined by θ = π/6 (ρ ∈ ℝ) intersects with a curve C₂ defined by 
    x = a + √2 cos θ, y = √2 sin θ (where a > 0) at two points A and B, and the distance |AB| = 2, 
    then a = 2. -/
theorem intersection_of_curves (a : ℝ) (h_a : a > 0) : 
  ∃ (A B : ℝ × ℝ),
    (∃ (ρ₁ ρ₂ : ℝ), 
      A.1 = ρ₁ * Real.cos (π/6) ∧ A.2 = ρ₁ * Real.sin (π/6) ∧
      B.1 = ρ₂ * Real.cos (π/6) ∧ B.2 = ρ₂ * Real.sin (π/6)) ∧
    (∃ (θ₁ θ₂ : ℝ),
      A.1 = a + Real.sqrt 2 * Real.cos θ₁ ∧ A.2 = Real.sqrt 2 * Real.sin θ₁ ∧
      B.1 = a + Real.sqrt 2 * Real.cos θ₂ ∧ B.2 = Real.sqrt 2 * Real.sin θ₂) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 →
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_intersection_of_curves_l711_71143


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l711_71161

theorem square_circle_area_ratio (s r : ℝ) (h : s > 0) (k : r > 0) 
  (perimeter_relation : 4 * s = 2 * π * r) : 
  (s^2) / (π * r^2) = π := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l711_71161


namespace NUMINAMATH_CALUDE_factorization_3x_squared_minus_12_l711_71172

theorem factorization_3x_squared_minus_12 (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3x_squared_minus_12_l711_71172


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l711_71170

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Theorem: In an arithmetic sequence, if a₈ = 20 and S₇ = 56, then a₁₂ = 32 -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
    (h₁ : seq.a 8 = 20)
    (h₂ : seq.S 7 = 56) :
  seq.a 12 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l711_71170


namespace NUMINAMATH_CALUDE_divisibility_properties_l711_71132

theorem divisibility_properties :
  (∃ k : ℤ, 2^41 + 1 = 83 * k) ∧
  (∃ m : ℤ, 2^70 + 3^70 = 13 * m) ∧
  (∃ n : ℤ, 2^60 - 1 = 20801 * n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_properties_l711_71132


namespace NUMINAMATH_CALUDE_max_stores_visited_l711_71148

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (double_visitors : ℕ) (total_shoppers : ℕ) 
  (h1 : total_stores = 8)
  (h2 : total_visits = 23)
  (h3 : double_visitors = 8)
  (h4 : total_shoppers = 12)
  (h5 : double_visitors * 2 ≤ total_visits)
  (h6 : ∀ n : ℕ, n ≤ total_shoppers → n > 0) :
  ∃ max_visits : ℕ, max_visits = 7 ∧ 
    (∀ person : ℕ, person ≤ total_shoppers → 
      ∃ visits : ℕ, visits ≤ max_visits ∧ 
        (double_visitors * 2 + (total_shoppers - double_visitors) * visits ≤ total_visits)) :=
by sorry

end NUMINAMATH_CALUDE_max_stores_visited_l711_71148


namespace NUMINAMATH_CALUDE_clara_dina_age_difference_l711_71173

theorem clara_dina_age_difference : ∃! n : ℕ+, ∃ C D : ℕ+,
  C = D + n ∧
  C - 1 = 3 * (D - 1) ∧
  C = D^3 + 1 ∧
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_clara_dina_age_difference_l711_71173


namespace NUMINAMATH_CALUDE_triangular_array_sum_of_digits_l711_71147

/-- The sum of the first n positive integers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The number of rows in the triangular array -/
def N : ℕ := 77

theorem triangular_array_sum_of_digits :
  (triangular_number N = 3003) ∧ (sum_of_digits N = 14) := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_sum_of_digits_l711_71147


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l711_71120

theorem ratio_x_to_y (x y : ℝ) (h : y = 0.2 * x) : x / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l711_71120


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l711_71174

theorem complex_on_imaginary_axis (a : ℝ) : 
  (Complex.I * ((2 * a + Complex.I) * (1 + Complex.I))).re = 0 → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l711_71174


namespace NUMINAMATH_CALUDE_bbq_ice_cost_chad_bbq_ice_cost_l711_71167

/-- The cost of ice for a BBQ given the number of people, ice needed per person, and ice price --/
theorem bbq_ice_cost (people : ℕ) (ice_per_person : ℕ) (pack_size : ℕ) (pack_price : ℚ) : ℚ :=
  let total_ice := people * ice_per_person
  let packs_needed := (total_ice + pack_size - 1) / pack_size  -- Ceiling division
  packs_needed * pack_price

/-- Proof that the cost of ice for Chad's BBQ is $9.00 --/
theorem chad_bbq_ice_cost : bbq_ice_cost 15 2 10 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_bbq_ice_cost_chad_bbq_ice_cost_l711_71167


namespace NUMINAMATH_CALUDE_car_trade_profit_percentage_l711_71102

/-- Calculates the profit percentage on the original price when a trader buys a car at a discount and sells it at an increase. -/
theorem car_trade_profit_percentage 
  (original_price : ℝ) 
  (discount_rate : ℝ) 
  (increase_rate : ℝ) 
  (h1 : original_price > 0)
  (h2 : discount_rate = 0.20)
  (h3 : increase_rate = 0.50) : 
  (((1 - discount_rate) * (1 + increase_rate) - 1) * 100 = 20) := by
sorry

end NUMINAMATH_CALUDE_car_trade_profit_percentage_l711_71102


namespace NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l711_71154

theorem root_in_interval_implies_m_range (m : ℝ) :
  (∃ x, x^2 + 2*x - m = 0 ∧ 2 < x ∧ x < 3) →
  (8 < m ∧ m < 15) := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l711_71154


namespace NUMINAMATH_CALUDE_expression_simplification_l711_71123

theorem expression_simplification :
  (4 * 6) / (12 * 14) * (3 * 5 * 7 * 9) / (4 * 6 * 8) * 7 = 45 / 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l711_71123


namespace NUMINAMATH_CALUDE_wheat_mixture_profit_percentage_wheat_profit_approximately_30_percent_l711_71183

/-- Calculates the profit percentage for a wheat mixture sale --/
theorem wheat_mixture_profit_percentage 
  (wheat1_weight : ℝ) (wheat1_price : ℝ) 
  (wheat2_weight : ℝ) (wheat2_price : ℝ) 
  (selling_price : ℝ) : ℝ :=
  let total_cost := wheat1_weight * wheat1_price + wheat2_weight * wheat2_price
  let total_weight := wheat1_weight + wheat2_weight
  let selling_total := total_weight * selling_price
  let profit := selling_total - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage

/-- Proves that the profit percentage is approximately 30% --/
theorem wheat_profit_approximately_30_percent : 
  abs (wheat_mixture_profit_percentage 30 11.50 20 14.25 16.38 - 30) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_wheat_mixture_profit_percentage_wheat_profit_approximately_30_percent_l711_71183


namespace NUMINAMATH_CALUDE_product_sum_and_32_l711_71105

theorem product_sum_and_32 : (12 + 25 + 52 + 21) * 32 = 3520 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_and_32_l711_71105


namespace NUMINAMATH_CALUDE_triangle_properties_l711_71131

-- Define a structure for our triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define our main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b * Real.sin t.A = (3 * t.b - t.c) * Real.sin t.B)
  (h2 : t.a + t.b + t.c = 8) :
  (2 * Real.sin t.A = 3 * Real.sin t.B → t.c = 3) ∧
  (t.a = t.c → Real.cos (2 * t.B) = 17 / 81) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l711_71131


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l711_71140

/-- Given a function f(x) = (1/3)x^3 + x^2 - ax + 3a that is monotonically increasing
    in the interval [1, 2], prove that a ≤ 3 -/
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, Monotone (fun x => (1/3) * x^3 + x^2 - a*x + 3*a)) →
  a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l711_71140


namespace NUMINAMATH_CALUDE_integers_with_consecutive_twos_l711_71155

def fibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def valid_integers (n : ℕ) : ℕ := 2^n

def integers_without_consecutive_twos (n : ℕ) : ℕ := fibonacci (n - 1)

theorem integers_with_consecutive_twos (n : ℕ) : 
  valid_integers n - integers_without_consecutive_twos n = 880 → n = 10 := by
  sorry

#eval valid_integers 10 - integers_without_consecutive_twos 10

end NUMINAMATH_CALUDE_integers_with_consecutive_twos_l711_71155
