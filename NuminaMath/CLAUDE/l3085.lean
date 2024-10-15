import Mathlib

namespace NUMINAMATH_CALUDE_prob_irrational_not_adjacent_l3085_308509

/-- The number of rational terms in the expansion of (x + 2/√x)^6 -/
def num_rational_terms : ℕ := 4

/-- The number of irrational terms in the expansion of (x + 2/√x)^6 -/
def num_irrational_terms : ℕ := 3

/-- The total number of terms in the expansion -/
def total_terms : ℕ := num_rational_terms + num_irrational_terms

/-- The probability that irrational terms are not adjacent in the expansion of (x + 2/√x)^6 -/
theorem prob_irrational_not_adjacent : 
  (Nat.factorial num_rational_terms * (Nat.factorial (num_rational_terms + 1)) / 
   Nat.factorial num_irrational_terms) / 
  (Nat.factorial total_terms) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_irrational_not_adjacent_l3085_308509


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l3085_308503

theorem smallest_integer_solution : 
  (∀ x : ℤ, x < 1 → (x : ℚ) / 4 + 3 / 7 ≤ 2 / 3) ∧ 
  (1 : ℚ) / 4 + 3 / 7 > 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l3085_308503


namespace NUMINAMATH_CALUDE_solution_for_x_l3085_308550

theorem solution_for_x (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0)
  (eq1 : x + 1 / z = 15) (eq2 : z + 1 / x = 9 / 20) :
  x = (15 + 5 * Real.sqrt 11) / 2 ∨ x = (15 - 5 * Real.sqrt 11) / 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_for_x_l3085_308550


namespace NUMINAMATH_CALUDE_polynomial_value_at_three_l3085_308587

theorem polynomial_value_at_three : 
  let x : ℝ := 3
  x^6 - 6*x^2 + 7*x = 696 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_three_l3085_308587


namespace NUMINAMATH_CALUDE_correct_sunset_time_l3085_308560

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

def addTime (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + d.minutes + (t.hours + d.hours) * 60
  { hours := totalMinutes / 60 % 24,
    minutes := totalMinutes % 60 }

def sunsetTime (sunrise : Time) (daylight : Duration) : Time :=
  addTime sunrise daylight

theorem correct_sunset_time :
  let sunrise : Time := { hours := 16, minutes := 35 }
  let daylight : Duration := { hours := 9, minutes := 48 }
  sunsetTime sunrise daylight = { hours := 2, minutes := 23 } := by
  sorry

end NUMINAMATH_CALUDE_correct_sunset_time_l3085_308560


namespace NUMINAMATH_CALUDE_expression_simplification_l3085_308573

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 - 1) :
  (x / (x - 1) - 1) / ((x^2 - 1) / (x^2 - 2*x + 1)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3085_308573


namespace NUMINAMATH_CALUDE_ax5_plus_by5_l3085_308582

theorem ax5_plus_by5 (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 80) :
  a * x^5 + b * y^5 = 6200 / 29 := by
sorry

end NUMINAMATH_CALUDE_ax5_plus_by5_l3085_308582


namespace NUMINAMATH_CALUDE_fifth_term_ratio_l3085_308554

/-- Given two arithmetic sequences {a_n} and {b_n} with sums S_n and T_n respectively -/
def arithmetic_sequences (a b : ℕ → ℝ) (S T : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2 ∧ T n = (n * (b 1 + b n)) / 2

/-- The ratio of sums S_n and T_n is 2n / (3n + 1) -/
def sum_ratio (S T : ℕ → ℝ) : Prop :=
  ∀ n, S n / T n = (2 * n : ℝ) / (3 * n + 1)

/-- The main theorem: given the conditions, prove a_5 / b_5 = 9 / 14 -/
theorem fifth_term_ratio (a b : ℕ → ℝ) (S T : ℕ → ℝ)
    (h1 : arithmetic_sequences a b S T) (h2 : sum_ratio S T) :
    a 5 / b 5 = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_ratio_l3085_308554


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3085_308549

theorem polynomial_factorization (a b c : ℝ) : 
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 = 
  (a - b) * (b - c) * (c - a) * ((a + b)^2 + (b + c)^2 + (c + a)^2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3085_308549


namespace NUMINAMATH_CALUDE_sum_lent_is_1000_l3085_308502

/-- The sum lent in rupees -/
def sum_lent : ℝ := 1000

/-- The annual interest rate as a percentage -/
def interest_rate : ℝ := 5

/-- The time period in years -/
def time_period : ℝ := 5

/-- The difference between the sum lent and the interest after the time period -/
def interest_difference : ℝ := 750

/-- Theorem stating that the sum lent is 1000 rupees given the problem conditions -/
theorem sum_lent_is_1000 :
  sum_lent = 1000 ∧
  interest_rate = 5 ∧
  time_period = 5 ∧
  interest_difference = 750 ∧
  sum_lent * interest_rate * time_period / 100 = sum_lent - interest_difference :=
by sorry

end NUMINAMATH_CALUDE_sum_lent_is_1000_l3085_308502


namespace NUMINAMATH_CALUDE_simplify_fraction_l3085_308526

theorem simplify_fraction (m n : ℝ) (h : n ≠ 0) : m * n / (n ^ 2) = m / n := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3085_308526


namespace NUMINAMATH_CALUDE_remainder_sum_l3085_308581

theorem remainder_sum (a b : ℤ) 
  (ha : a % 80 = 74) 
  (hb : b % 120 = 114) : 
  (a + b) % 40 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3085_308581


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_three_elevenths_l3085_308528

/-- The repeating decimal 0.27̄ -/
def repeating_decimal : ℚ := 27 / 99

theorem repeating_decimal_equals_three_elevenths : 
  repeating_decimal = 3 / 11 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_three_elevenths_l3085_308528


namespace NUMINAMATH_CALUDE_crayon_boxes_l3085_308511

theorem crayon_boxes (total_crayons : ℕ) (full_boxes : ℕ) (loose_crayons : ℕ) (friend_crayons : ℕ) :
  total_crayons = 85 →
  full_boxes = 5 →
  loose_crayons = 5 →
  friend_crayons = 27 →
  (total_crayons - loose_crayons) / full_boxes = 16 →
  ((loose_crayons + friend_crayons) + ((total_crayons - loose_crayons) / full_boxes - 1)) / ((total_crayons - loose_crayons) / full_boxes) = 2 := by
  sorry

#check crayon_boxes

end NUMINAMATH_CALUDE_crayon_boxes_l3085_308511


namespace NUMINAMATH_CALUDE_open_box_volume_is_24000_l3085_308563

/-- Represents the dimensions of a rectangular sheet -/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a parallelogram cut from corners -/
structure ParallelogramCut where
  base : ℝ
  height : ℝ

/-- Calculates the volume of the open box created from a sheet with given dimensions and corner cuts -/
def openBoxVolume (sheet : SheetDimensions) (cut : ParallelogramCut) : ℝ :=
  (sheet.length - 2 * cut.base) * (sheet.width - 2 * cut.base) * cut.height

/-- Theorem stating that the volume of the open box is 24000 m^3 -/
theorem open_box_volume_is_24000 (sheet : SheetDimensions) (cut : ParallelogramCut)
    (h1 : sheet.length = 100)
    (h2 : sheet.width = 50)
    (h3 : cut.base = 10)
    (h4 : cut.height = 10) :
    openBoxVolume sheet cut = 24000 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_is_24000_l3085_308563


namespace NUMINAMATH_CALUDE_distribute_five_contestants_three_companies_l3085_308522

/-- The number of ways to distribute contestants among companies -/
def distribute_contestants (num_contestants : ℕ) (num_companies : ℕ) : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem: The number of ways to distribute 5 contestants among 3 companies,
    where each company must have at least 1 and at most 2 contestants, is 90 -/
theorem distribute_five_contestants_three_companies :
  distribute_contestants 5 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_contestants_three_companies_l3085_308522


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3085_308516

theorem polynomial_divisibility (m : ℤ) : 
  ∃ k : ℤ, (4*m + 5)^2 - 9 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3085_308516


namespace NUMINAMATH_CALUDE_average_songs_theorem_l3085_308594

/-- Represents a band's performance schedule --/
structure BandPerformance where
  repertoire : ℕ
  first_set : ℕ
  second_set : ℕ
  encores : ℕ

/-- Calculates the average number of songs for the remaining sets --/
def average_remaining_songs (b : BandPerformance) : ℚ :=
  let songs_played := b.first_set + b.second_set + b.encores
  let remaining_songs := b.repertoire - songs_played
  let remaining_sets := 3
  (remaining_songs : ℚ) / remaining_sets

/-- Theorem stating the average number of songs for the remaining sets --/
theorem average_songs_theorem (b : BandPerformance) 
  (h1 : b.repertoire = 50)
  (h2 : b.first_set = 8)
  (h3 : b.second_set = 12)
  (h4 : b.encores = 4) :
  average_remaining_songs b = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_songs_theorem_l3085_308594


namespace NUMINAMATH_CALUDE_max_sum_xyz_l3085_308541

theorem max_sum_xyz (x y z : ℕ+) (h1 : x < y) (h2 : y < z) (h3 : x + x * y + x * y * z = 37) :
  x + y + z ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_max_sum_xyz_l3085_308541


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l3085_308531

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l3085_308531


namespace NUMINAMATH_CALUDE_encoded_equation_unique_solution_l3085_308556

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- The encoded equation -/
def EncodedEquation (Δ square triangle circle : Digit) : Prop :=
  ∃ (base : TwoDigitNumber) (result : ThreeDigitNumber),
    base.val = 10 * Δ.val + square.val ∧
    result.val = 100 * square.val + 10 * circle.val + square.val ∧
    base.val ^ triangle.val = result.val

theorem encoded_equation_unique_solution :
  ∃! (Δ square triangle circle : Digit), EncodedEquation Δ square triangle circle :=
sorry

end NUMINAMATH_CALUDE_encoded_equation_unique_solution_l3085_308556


namespace NUMINAMATH_CALUDE_inequality_always_true_l3085_308593

theorem inequality_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l3085_308593


namespace NUMINAMATH_CALUDE_james_weight_vest_savings_l3085_308565

/-- The savings James makes by buying a separate vest and plates instead of a discounted 200-pound weight vest -/
theorem james_weight_vest_savings 
  (separate_vest_cost : ℝ) 
  (plate_weight : ℝ) 
  (cost_per_pound : ℝ) 
  (full_vest_cost : ℝ) 
  (discount : ℝ)
  (h1 : separate_vest_cost = 250)
  (h2 : plate_weight = 200)
  (h3 : cost_per_pound = 1.2)
  (h4 : full_vest_cost = 700)
  (h5 : discount = 100) :
  full_vest_cost - discount - (separate_vest_cost + plate_weight * cost_per_pound) = 110 := by
  sorry

end NUMINAMATH_CALUDE_james_weight_vest_savings_l3085_308565


namespace NUMINAMATH_CALUDE_workshop_workers_workshop_workers_proof_l3085_308578

/-- The total number of workers in a workshop with given salary conditions -/
theorem workshop_workers : ℕ :=
  let avg_salary : ℚ := 8000
  let num_technicians : ℕ := 7
  let avg_salary_technicians : ℚ := 18000
  let avg_salary_others : ℚ := 6000
  42

/-- Proof that the total number of workers is 42 -/
theorem workshop_workers_proof : workshop_workers = 42 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_workshop_workers_proof_l3085_308578


namespace NUMINAMATH_CALUDE_acid_dilution_l3085_308548

/-- Given a p% solution of acid with volume p ounces (where p > 45),
    adding y ounces of water to create a (2p/3)% solution results in y = p/2 -/
theorem acid_dilution (p : ℝ) (y : ℝ) (h₁ : p > 45) :
  (p^2 / 100 = (2 * p / 300) * (p + y)) → y = p / 2 := by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l3085_308548


namespace NUMINAMATH_CALUDE_solution_system_equations_l3085_308515

theorem solution_system_equations (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_order : a > b ∧ b > c ∧ c > d) :
  ∃ (x y z t : ℝ),
    (|a - b| * y + |a - c| * z + |a - d| * t = 1) ∧
    (|b - a| * x + |b - c| * z + |b - d| * t = 1) ∧
    (|c - a| * x + |c - b| * y + |c - d| * t = 1) ∧
    (|d - a| * x + |d - b| * y + |d - c| * z = 1) ∧
    (x = 1 / (a - d)) ∧
    (y = 0) ∧
    (z = 0) ∧
    (t = 1 / (a - d)) := by
  sorry

end NUMINAMATH_CALUDE_solution_system_equations_l3085_308515


namespace NUMINAMATH_CALUDE_odds_calculation_l3085_308527

theorem odds_calculation (x : ℝ) (h : (x / (x + 5)) = 0.375) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_odds_calculation_l3085_308527


namespace NUMINAMATH_CALUDE_multiples_of_seven_l3085_308572

theorem multiples_of_seven (a b : ℤ) (q : Set ℤ) : 
  (∃ k₁ k₂ : ℤ, a = 14 * k₁ ∧ b = 14 * k₂) →
  q = {x : ℤ | a ≤ x ∧ x ≤ b} →
  (Finset.filter (fun x => x % 14 = 0) (Finset.Icc a b)).card = 12 →
  (Finset.filter (fun x => x % 7 = 0) (Finset.Icc a b)).card = 24 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_seven_l3085_308572


namespace NUMINAMATH_CALUDE_angle_half_in_second_quadrant_l3085_308580

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2

def is_in_second_quadrant (θ : Real) : Prop :=
  ∃ k : ℤ, k * Real.pi + Real.pi / 2 < θ ∧ θ < k * Real.pi + Real.pi

theorem angle_half_in_second_quadrant (α : Real) 
  (h1 : is_in_third_quadrant α) 
  (h2 : |Real.cos (α/2)| = -Real.cos (α/2)) : 
  is_in_second_quadrant (α/2) := by
  sorry


end NUMINAMATH_CALUDE_angle_half_in_second_quadrant_l3085_308580


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2017_l3085_308501

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_2017 
  (seq : ArithmeticSequence) 
  (h1 : sum_n seq 2011 = -2011) 
  (h2 : seq.a 1012 = 3) : 
  sum_n seq 2017 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2017_l3085_308501


namespace NUMINAMATH_CALUDE_smallest_quotient_two_digit_numbers_l3085_308519

theorem smallest_quotient_two_digit_numbers :
  ∀ a b : ℕ,
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧
    a ≠ b →
    (10 * a + b : ℚ) / (a + b) ≥ 1.9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_quotient_two_digit_numbers_l3085_308519


namespace NUMINAMATH_CALUDE_sticks_per_hour_to_stay_warm_l3085_308590

/-- The number of sticks of wood produced by chopping up furniture -/
def sticks_per_furniture : Nat → Nat
| 0 => 6  -- chairs
| 1 => 9  -- tables
| 2 => 2  -- stools
| _ => 0  -- other furniture (not considered)

/-- The number of each type of furniture Mary chopped up -/
def furniture_count : Nat → Nat
| 0 => 18  -- chairs
| 1 => 6   -- tables
| 2 => 4   -- stools
| _ => 0   -- other furniture (not considered)

/-- The number of hours Mary can keep warm -/
def warm_hours : Nat := 34

/-- Calculates the total number of sticks of wood Mary has -/
def total_sticks : Nat :=
  (sticks_per_furniture 0 * furniture_count 0) +
  (sticks_per_furniture 1 * furniture_count 1) +
  (sticks_per_furniture 2 * furniture_count 2)

/-- The theorem to prove -/
theorem sticks_per_hour_to_stay_warm :
  total_sticks / warm_hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_sticks_per_hour_to_stay_warm_l3085_308590


namespace NUMINAMATH_CALUDE_bus_capacity_theorem_l3085_308512

/-- Represents the capacity of a bus in terms of children -/
def bus_capacity (rows : ℕ) (children_per_row : ℕ) : ℕ :=
  rows * children_per_row

/-- Theorem stating that a bus with 9 rows and 4 children per row can accommodate 36 children -/
theorem bus_capacity_theorem :
  bus_capacity 9 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_theorem_l3085_308512


namespace NUMINAMATH_CALUDE_jacket_sale_profit_l3085_308589

/-- Represents a jacket sale with its cost and selling price -/
structure JacketSale where
  cost : ℝ
  sellingPrice : ℝ

/-- Calculates the profit or loss from a jacket sale -/
def profit (sale : JacketSale) : ℝ := sale.sellingPrice - sale.cost

theorem jacket_sale_profit :
  ∀ (jacket1 jacket2 : JacketSale),
    jacket1.sellingPrice = 80 ∧
    jacket2.sellingPrice = 80 ∧
    jacket1.sellingPrice = jacket1.cost * 1.6 ∧
    jacket2.sellingPrice = jacket2.cost * 0.8 →
    profit jacket1 + profit jacket2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jacket_sale_profit_l3085_308589


namespace NUMINAMATH_CALUDE_box_dimensions_sum_l3085_308585

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Defines the properties of the rectangular box -/
def validBox (d : BoxDimensions) : Prop :=
  d.A * d.B = 18 ∧ d.A * d.C = 32 ∧ d.B * d.C = 50

/-- Theorem stating that the sum of dimensions is approximately 57.28 -/
theorem box_dimensions_sum (d : BoxDimensions) (h : validBox d) :
  ∃ ε > 0, |d.A + d.B + d.C - 57.28| < ε :=
sorry

end NUMINAMATH_CALUDE_box_dimensions_sum_l3085_308585


namespace NUMINAMATH_CALUDE_abs_minus_2010_l3085_308533

theorem abs_minus_2010 : |(-2010 : ℤ)| = 2010 := by
  sorry

end NUMINAMATH_CALUDE_abs_minus_2010_l3085_308533


namespace NUMINAMATH_CALUDE_max_slope_no_lattice_points_l3085_308583

theorem max_slope_no_lattice_points :
  let max_a : ℚ := 25 / 49
  ∀ a : ℚ, (∀ m x y : ℚ,
    (1 / 2 < m) → (m < a) →
    (0 < x) → (x ≤ 50) →
    (y = m * x + 3) →
    (∃ n : ℤ, x = ↑n) →
    (∃ n : ℤ, y = ↑n) →
    False) →
  a ≤ max_a :=
by sorry

end NUMINAMATH_CALUDE_max_slope_no_lattice_points_l3085_308583


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3085_308598

theorem rhombus_longer_diagonal (side_length shorter_diagonal : ℝ) :
  side_length = 65 ∧ shorter_diagonal = 72 →
  ∃ longer_diagonal : ℝ, longer_diagonal = 108 ∧
  longer_diagonal^2 + shorter_diagonal^2 = 4 * side_length^2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3085_308598


namespace NUMINAMATH_CALUDE_unique_solution_for_odd_prime_l3085_308536

theorem unique_solution_for_odd_prime (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (m n : ℕ), 
    (2 : ℚ) / p = 1 / m + 1 / n ∧
    m > n ∧
    m = p * (p + 1) / 2 ∧
    n = 2 / (p + 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_odd_prime_l3085_308536


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3085_308538

theorem polynomial_remainder (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 7*x^3 + 9*x^2 + 16*x - 13
  f 3 = 8 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3085_308538


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_defined_l3085_308520

theorem sqrt_x_minus_3_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_defined_l3085_308520


namespace NUMINAMATH_CALUDE_hemisphere_chord_length_l3085_308558

theorem hemisphere_chord_length (R : ℝ) (h : R = 20) : 
  let chord_length := 2 * R * Real.sqrt 2 / 2
  chord_length = 20 * Real.sqrt 2 := by
  sorry

#check hemisphere_chord_length

end NUMINAMATH_CALUDE_hemisphere_chord_length_l3085_308558


namespace NUMINAMATH_CALUDE_factorization_equality_l3085_308567

theorem factorization_equality (a b : ℝ) : 3 * a * b^2 + a^2 * b = a * b * (3 * b + a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3085_308567


namespace NUMINAMATH_CALUDE_angle_between_skew_medians_l3085_308597

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- A median of a face in a regular tetrahedron -/
structure FaceMedian (t : RegularTetrahedron a) where
  start_vertex : ℝ × ℝ × ℝ
  end_point : ℝ × ℝ × ℝ

/-- The angle between two vectors in ℝ³ -/
def angle_between (v w : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Two face medians are skew if they're not on the same face -/
def are_skew_medians (m1 m2 : FaceMedian t) : Prop := sorry

theorem angle_between_skew_medians (t : RegularTetrahedron a) 
  (m1 m2 : FaceMedian t) (h : are_skew_medians m1 m2) : 
  angle_between (m1.end_point - m1.start_vertex) (m2.end_point - m2.start_vertex) = Real.arccos (1/6) := by
  sorry

end NUMINAMATH_CALUDE_angle_between_skew_medians_l3085_308597


namespace NUMINAMATH_CALUDE_pet_ownership_l3085_308559

theorem pet_ownership (total : ℕ) (dog_owners : ℕ) (cat_owners : ℕ) 
  (h1 : total = 50)
  (h2 : dog_owners = 28)
  (h3 : cat_owners = 35)
  (h4 : dog_owners + cat_owners - total ≤ dog_owners)
  (h5 : dog_owners + cat_owners - total ≤ cat_owners) :
  dog_owners + cat_owners - total = 13 := by
sorry

end NUMINAMATH_CALUDE_pet_ownership_l3085_308559


namespace NUMINAMATH_CALUDE_moose_population_canada_l3085_308506

/-- The moose population in Canada, given the ratio of moose to beavers to humans -/
theorem moose_population_canada (total_humans : ℕ) (moose_to_beaver : ℕ) (beaver_to_human : ℕ) :
  total_humans = 38000000 →
  moose_to_beaver = 2 →
  beaver_to_human = 19 →
  (total_humans / (moose_to_beaver * beaver_to_human) : ℚ) = 1000000 := by
  sorry

#check moose_population_canada

end NUMINAMATH_CALUDE_moose_population_canada_l3085_308506


namespace NUMINAMATH_CALUDE_weight_lifting_multiple_l3085_308586

theorem weight_lifting_multiple (rodney roger ron : ℕ) (m : ℕ) : 
  rodney + roger + ron = 239 →
  rodney = 2 * roger →
  roger = m * ron - 7 →
  rodney = 146 →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_weight_lifting_multiple_l3085_308586


namespace NUMINAMATH_CALUDE_negation_of_forall_gt_negation_of_gt_is_le_negation_of_forall_x_squared_gt_1_minus_2x_l3085_308555

theorem negation_of_forall_gt (P : ℝ → Prop) :
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) :=
by sorry

theorem negation_of_gt_is_le {a b : ℝ} :
  ¬(a > b) ↔ (a ≤ b) :=
by sorry

theorem negation_of_forall_x_squared_gt_1_minus_2x :
  (¬∀ x : ℝ, x^2 > 1 - 2*x) ↔ (∃ x : ℝ, x^2 ≤ 1 - 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_gt_negation_of_gt_is_le_negation_of_forall_x_squared_gt_1_minus_2x_l3085_308555


namespace NUMINAMATH_CALUDE_park_tree_count_l3085_308518

/-- The number of dogwood trees in the park after 5 days of planting and one uprooting event -/
def final_tree_count (initial : ℕ) (day1 : ℕ) (day5 : ℕ) : ℕ :=
  let day2 := day1 / 2
  let day3 := day2 * 4
  let day4 := 5  -- Trees replaced due to uprooting
  initial + day1 + day2 + day3 + day4 + day5

/-- Theorem stating the final number of trees in the park -/
theorem park_tree_count : final_tree_count 39 24 15 = 143 := by
  sorry

end NUMINAMATH_CALUDE_park_tree_count_l3085_308518


namespace NUMINAMATH_CALUDE_pigeonhole_principle_sports_l3085_308568

theorem pigeonhole_principle_sports (n : ℕ) (h : n = 50) :
  ∃ (same_choices : ℕ), same_choices ≥ 3 ∧
  (∀ (choices : Fin n → Fin 4 × Fin 3 × Fin 2),
   ∃ (subset : Finset (Fin n)),
   subset.card = same_choices ∧
   ∀ (i j : Fin n), i ∈ subset → j ∈ subset → choices i = choices j) :=
by sorry

end NUMINAMATH_CALUDE_pigeonhole_principle_sports_l3085_308568


namespace NUMINAMATH_CALUDE_infinitely_many_linear_combinations_l3085_308552

/-- An infinite sequence of positive integers where each element is strictly greater than the previous one. -/
def StrictlyIncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ k, a k < a (k + 1)

/-- The property that an element of the sequence can be expressed as a linear combination of two distinct earlier elements. -/
def CanBeExpressedAsLinearCombination (a : ℕ → ℕ) (m p q : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ p ≠ q ∧ a m = x * a p + y * a q

/-- The main theorem stating that infinitely many elements of the sequence can be expressed as linear combinations of two distinct earlier elements. -/
theorem infinitely_many_linear_combinations (a : ℕ → ℕ) 
    (h : StrictlyIncreasingSequence a) :
    ∀ N, ∃ m, m > N ∧ ∃ p q, CanBeExpressedAsLinearCombination a m p q := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_linear_combinations_l3085_308552


namespace NUMINAMATH_CALUDE_lcm_1362_918_l3085_308591

theorem lcm_1362_918 : Nat.lcm 1362 918 = 69462 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1362_918_l3085_308591


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l3085_308595

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The problem statement -/
theorem pirate_loot_sum : 
  let silverware := base5ToBase10 [4, 1, 2, 3]
  let gemstones := base5ToBase10 [2, 2, 0, 3]
  let fine_silk := base5ToBase10 [2, 0, 2]
  silverware + gemstones + fine_silk = 873 := by
  sorry


end NUMINAMATH_CALUDE_pirate_loot_sum_l3085_308595


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3085_308539

theorem problem_1 : (1 - Real.sqrt 2) ^ 0 - 2 * Real.sin (π / 4) + (Real.sqrt 2) ^ 2 = 3 - Real.sqrt 2 := by
  sorry

theorem problem_2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) :
  (1 - 2 / (x - 1)) / ((x^2 - 6*x + 9) / (x^2 - 1)) = (x + 1) / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3085_308539


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3085_308535

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_a6 : a 6 = 5) 
  (h_a10 : a 10 = 6) : 
  ∃ d : ℚ, d = 1/4 ∧ ∀ n : ℕ, a (n + 1) = a n + d := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3085_308535


namespace NUMINAMATH_CALUDE_sum_of_powers_of_two_l3085_308517

theorem sum_of_powers_of_two : 1 + 1/2 + 1/4 + 1/8 = 15/8 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_two_l3085_308517


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l3085_308571

def p (x : ℤ) : ℤ := x^3 - 4*x^2 - 11*x + 24

theorem integer_roots_of_cubic :
  {x : ℤ | p x = 0} = {-1, -2, 3} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l3085_308571


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3085_308576

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 4 * x + y = 12) 
  (h2 : x + 4 * y = 18) : 
  20 * x^2 + 24 * x * y + 20 * y^2 = 468 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3085_308576


namespace NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l3085_308507

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point is on the right branch of the hyperbola -/
def isOnRightBranch (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1 ∧ p.x > 0

/-- Checks if a point is on the given line -/
def isOnLine (p : Point) : Prop :=
  p.y = Real.sqrt 3 / 3 * p.x - 2

/-- The main theorem to be proved -/
theorem hyperbola_intersection_theorem (h : Hyperbola) 
    (hA : isOnRightBranch h A ∧ isOnLine A)
    (hB : isOnRightBranch h B ∧ isOnLine B)
    (hC : isOnRightBranch h C) :
    h.a = 2 * Real.sqrt 3 →
    h.b = Real.sqrt 3 →
    C.x = 4 * Real.sqrt 3 →
    C.y = 3 →
    ∃ m : ℝ, m = 4 ∧ A.x + B.x = m * C.x ∧ A.y + B.y = m * C.y := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_intersection_theorem_l3085_308507


namespace NUMINAMATH_CALUDE_triangle_legs_theorem_l3085_308592

/-- A point inside a right angle -/
structure PointInRightAngle where
  /-- Distance from the point to one side of the angle -/
  dist1 : ℝ
  /-- Distance from the point to the other side of the angle -/
  dist2 : ℝ

/-- A triangle formed by a line through a point in a right angle -/
structure TriangleInRightAngle where
  /-- The point inside the right angle -/
  point : PointInRightAngle
  /-- The area of the triangle -/
  area : ℝ

/-- The legs of a right triangle -/
structure RightTriangleLegs where
  /-- Length of one leg -/
  leg1 : ℝ
  /-- Length of the other leg -/
  leg2 : ℝ

/-- Theorem about the legs of a specific triangle in a right angle -/
theorem triangle_legs_theorem (t : TriangleInRightAngle)
    (h1 : t.point.dist1 = 4)
    (h2 : t.point.dist2 = 8)
    (h3 : t.area = 100) :
    (∃ l : RightTriangleLegs, (l.leg1 = 40 ∧ l.leg2 = 5) ∨ (l.leg1 = 10 ∧ l.leg2 = 20)) :=
  sorry

end NUMINAMATH_CALUDE_triangle_legs_theorem_l3085_308592


namespace NUMINAMATH_CALUDE_line_circle_no_intersection_l3085_308524

/-- The line 3x + 4y = 12 and the circle x^2 + y^2 = 4 have no intersection points in the real plane. -/
theorem line_circle_no_intersection :
  ¬ ∃ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_no_intersection_l3085_308524


namespace NUMINAMATH_CALUDE_equation_solution_l3085_308599

theorem equation_solution (x : ℝ) : 
  (|Real.cos x| + Real.cos (3 * x)) / (Real.sin x * Real.cos (2 * x)) = -2 * Real.sqrt 3 ↔ 
  (∃ k : ℤ, x = 2 * Real.pi / 3 + 2 * k * Real.pi ∨ 
            x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ 
            x = -Real.pi / 6 + 2 * k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3085_308599


namespace NUMINAMATH_CALUDE_b_10_equals_64_l3085_308542

/-- Sequences a and b satisfying the given conditions -/
def sequences_a_b (a b : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ 
  ∀ n : ℕ, (a n) * (a (n + 1)) = 2^n ∧
           (a n) + (a (n + 1)) = b n

/-- The main theorem to prove -/
theorem b_10_equals_64 (a b : ℕ → ℝ) (h : sequences_a_b a b) : 
  b 10 = 64 := by
  sorry

end NUMINAMATH_CALUDE_b_10_equals_64_l3085_308542


namespace NUMINAMATH_CALUDE_area_at_stage_8_l3085_308574

/-- Calculates the number of squares added up to a given stage -/
def squaresAdded (stage : ℕ) : ℕ :=
  (stage + 1) / 2

/-- The side length of each square in inches -/
def squareSideLength : ℕ := 4

/-- Calculates the area of the figure at a given stage -/
def areaAtStage (stage : ℕ) : ℕ :=
  (squaresAdded stage) * (squareSideLength * squareSideLength)

/-- Proves that the area of the figure at Stage 8 is 64 square inches -/
theorem area_at_stage_8 : areaAtStage 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_8_l3085_308574


namespace NUMINAMATH_CALUDE_delores_remaining_money_l3085_308525

/-- Calculates the remaining money after purchases -/
def remaining_money (initial : ℕ) (computer : ℕ) (printer : ℕ) : ℕ :=
  initial - (computer + printer)

/-- Proves that Delores has $10 left after her purchases -/
theorem delores_remaining_money :
  remaining_money 450 400 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_delores_remaining_money_l3085_308525


namespace NUMINAMATH_CALUDE_first_half_speed_l3085_308569

/-- Proves that given a journey of 3600 miles completed in 30 hours, 
    where the second half is traveled at 180 mph, 
    the average speed for the first half of the journey is 90 mph. -/
theorem first_half_speed (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 3600 →
  total_time = 30 →
  second_half_speed = 180 →
  (total_distance / 2) / (total_time - (total_distance / 2) / second_half_speed) = 90 :=
by sorry

end NUMINAMATH_CALUDE_first_half_speed_l3085_308569


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l3085_308532

theorem sum_reciprocals_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 ∧ ∀ M : ℝ, ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ a' + b' = 2 ∧ 1 / a' + 1 / b' > M :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l3085_308532


namespace NUMINAMATH_CALUDE_next_joint_tutoring_day_l3085_308534

def jaclyn_schedule : ℕ := 3
def marcelle_schedule : ℕ := 4
def susanna_schedule : ℕ := 6
def wanda_schedule : ℕ := 7

theorem next_joint_tutoring_day :
  Nat.lcm jaclyn_schedule (Nat.lcm marcelle_schedule (Nat.lcm susanna_schedule wanda_schedule)) = 84 := by
  sorry

end NUMINAMATH_CALUDE_next_joint_tutoring_day_l3085_308534


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l3085_308577

theorem smallest_k_no_real_roots :
  ∀ k : ℤ, (∀ x : ℝ, 3*x*(k*x-5) - 2*x^2 + 8 ≠ 0) → k ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l3085_308577


namespace NUMINAMATH_CALUDE_odd_function_iff_condition_l3085_308561

def f (x a b : ℝ) : ℝ := x * |x + a| + b

theorem odd_function_iff_condition (a b : ℝ) :
  (∀ x, f x a b = -f (-x) a b) ↔ a^2 + b^2 = 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_iff_condition_l3085_308561


namespace NUMINAMATH_CALUDE_milk_remaining_l3085_308588

theorem milk_remaining (initial : ℚ) (given : ℚ) (remaining : ℚ) : 
  initial = 8 → given = 18/7 → remaining = initial - given → remaining = 38/7 := by
  sorry

end NUMINAMATH_CALUDE_milk_remaining_l3085_308588


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_l3085_308544

/-- The stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease)

/-- Theorem stating the final stock price after two years -/
theorem stock_price_after_two_years :
  final_stock_price 50 1.5 0.3 = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_after_two_years_l3085_308544


namespace NUMINAMATH_CALUDE_both_false_sufficient_not_necessary_l3085_308545

-- Define simple propositions a and b
variable (a b : Prop)

-- Define the statements
def both_false : Prop := ¬a ∧ ¬b
def either_false : Prop := ¬a ∨ ¬b

-- Theorem statement
theorem both_false_sufficient_not_necessary :
  (both_false a b → either_false a b) ∧
  ¬(either_false a b → both_false a b) :=
sorry

end NUMINAMATH_CALUDE_both_false_sufficient_not_necessary_l3085_308545


namespace NUMINAMATH_CALUDE_max_value_a_l3085_308546

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 150) : 
  a ≤ 8924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 8924 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 150 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l3085_308546


namespace NUMINAMATH_CALUDE_investment_income_calculation_l3085_308521

/-- Calculates the annual income from an investment in shares given the investment amount,
    share face value, quoted price, and dividend rate. -/
def annual_income (investment : ℚ) (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) : ℚ :=
  (investment / quoted_price) * (face_value * dividend_rate)

/-- Theorem stating that for the given investment scenario, the annual income is 728 -/
theorem investment_income_calculation :
  let investment : ℚ := 4940
  let face_value : ℚ := 10
  let quoted_price : ℚ := 9.5
  let dividend_rate : ℚ := 14 / 100
  annual_income investment face_value quoted_price dividend_rate = 728 := by
  sorry


end NUMINAMATH_CALUDE_investment_income_calculation_l3085_308521


namespace NUMINAMATH_CALUDE_third_box_weight_l3085_308508

/-- Given two boxes with weights and their weight difference, prove the weight of the third box -/
theorem third_box_weight (weight_first : ℝ) (weight_diff : ℝ) : 
  weight_first = 2 → weight_diff = 11 → weight_first + weight_diff = 13 := by
  sorry

end NUMINAMATH_CALUDE_third_box_weight_l3085_308508


namespace NUMINAMATH_CALUDE_total_cost_of_books_l3085_308504

-- Define the number of books for each category
def animal_books : ℕ := 10
def space_books : ℕ := 1
def train_books : ℕ := 3

-- Define the cost per book
def cost_per_book : ℕ := 16

-- Define the total number of books
def total_books : ℕ := animal_books + space_books + train_books

-- Theorem to prove
theorem total_cost_of_books : total_books * cost_per_book = 224 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_books_l3085_308504


namespace NUMINAMATH_CALUDE_derivative_at_negative_one_l3085_308566

/-- Given a function f(x) = ax^4 + bx^2 + c where f'(1) = 2, prove that f'(-1) = -2 -/
theorem derivative_at_negative_one (a b c : ℝ) :
  let f := fun x : ℝ => a * x^4 + b * x^2 + c
  (deriv f) 1 = 2 → (deriv f) (-1) = -2 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_negative_one_l3085_308566


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l3085_308543

theorem least_number_for_divisibility (n m : ℕ) (hn : n = 1056) (hm : m = 23) :
  ∃ k : ℕ, k > 0 ∧ k ≤ m ∧ (n + k) % m = 0 ∧ ∀ j : ℕ, 0 < j ∧ j < k → (n + j) % m ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l3085_308543


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l3085_308547

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (36^2 + 45^2) ∧ ∀ q : ℕ, Nat.Prime q → q ∣ (36^2 + 45^2) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l3085_308547


namespace NUMINAMATH_CALUDE_shortening_theorem_l3085_308575

/-- A sequence of digits where each digit is 0 or 9 -/
def DigitSequence := List (Fin 2)

/-- The length of the original sequence -/
def originalLength : Nat := 2015

/-- The probability of a digit being the same as the previous one -/
def sameDigitProb : Real := 0.1

/-- The probability of a digit being different from the previous one -/
def differentDigitProb : Real := 0.9

/-- The shortening operation on a digit sequence -/
def shortenSequence (seq : DigitSequence) : DigitSequence :=
  sorry

/-- The probability that the sequence will shorten by exactly one digit -/
def shortenByOneProb (n : Nat) : Real :=
  sorry

/-- The expected length of the new sequence after shortening -/
def expectedNewLength (n : Nat) : Real :=
  sorry

theorem shortening_theorem :
  ∃ (ε : Real),
    ε > 0 ∧
    ε < 1e-89 ∧
    abs (shortenByOneProb originalLength - 1.564e-90) < ε ∧
    abs (expectedNewLength originalLength - 1813.6) < ε :=
  sorry

end NUMINAMATH_CALUDE_shortening_theorem_l3085_308575


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_solution_l3085_308557

theorem at_least_one_equation_has_solution (a b c : ℝ) : 
  ¬(c^2 > a^2 + b^2 ∧ b^2 - 16*a*c < 0) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_solution_l3085_308557


namespace NUMINAMATH_CALUDE_unvisited_route_count_l3085_308529

/-- The number of ways to distribute four families among four routes with one route unvisited -/
def unvisited_route_scenarios : ℕ := 144

/-- The number of families -/
def num_families : ℕ := 4

/-- The number of available routes -/
def num_routes : ℕ := 4

theorem unvisited_route_count :
  unvisited_route_scenarios = 
    (Nat.choose num_families 2) * (Nat.factorial num_routes) / Nat.factorial (num_routes - 3) :=
sorry

end NUMINAMATH_CALUDE_unvisited_route_count_l3085_308529


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3085_308564

theorem pizza_toppings_combinations (n : ℕ) (h : n = 8) : 
  n + n.choose 2 + n.choose 3 = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3085_308564


namespace NUMINAMATH_CALUDE_fish_count_l3085_308530

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := 10

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 12

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := lilly_fish + rosy_fish

theorem fish_count : total_fish = 22 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l3085_308530


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3085_308537

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b (m : ℝ) : Fin 2 → ℝ := ![m, 1]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem perpendicular_vectors_m_value :
  ∀ m : ℝ, dot_product a (b m) = 0 → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3085_308537


namespace NUMINAMATH_CALUDE_factory_price_decrease_and_sales_optimization_l3085_308523

/-- The average decrease rate in factory price over two years -/
def average_decrease_rate : ℝ := 0.1

/-- The price reduction that maximizes sales while maintaining the target profit -/
def price_reduction : ℝ := 15

/-- The initial factory price in 2019 -/
def initial_price : ℝ := 200

/-- The final factory price in 2021 -/
def final_price : ℝ := 162

/-- The initial number of pieces sold per day at the original price -/
def initial_sales : ℝ := 20

/-- The increase in sales for every 5 yuan reduction in price -/
def sales_increase_rate : ℝ := 2

/-- The target daily profit after price reduction -/
def target_profit : ℝ := 1150

theorem factory_price_decrease_and_sales_optimization :
  (initial_price * (1 - average_decrease_rate)^2 = final_price) ∧
  ((initial_price - final_price - price_reduction) * 
   (initial_sales + sales_increase_rate * price_reduction) = target_profit) ∧
  (∀ m : ℝ, m > price_reduction → 
   ((initial_price - final_price - m) * (initial_sales + sales_increase_rate * m) ≠ target_profit)) :=
by sorry

end NUMINAMATH_CALUDE_factory_price_decrease_and_sales_optimization_l3085_308523


namespace NUMINAMATH_CALUDE_pool_water_removal_l3085_308553

/-- Calculates the volume of water removed from a rectangular pool in gallons -/
def water_removed (length width height : ℝ) (conversion_factor : ℝ) : ℝ :=
  length * width * height * conversion_factor

theorem pool_water_removal :
  let length : ℝ := 60
  let width : ℝ := 10
  let height : ℝ := 0.5
  let conversion_factor : ℝ := 7.5
  water_removed length width height conversion_factor = 2250 := by
sorry

end NUMINAMATH_CALUDE_pool_water_removal_l3085_308553


namespace NUMINAMATH_CALUDE_alan_told_seven_jokes_l3085_308514

/-- The number of jokes Jessy told on Saturday -/
def jessy_jokes : ℕ := 11

/-- The number of jokes Alan told on Saturday -/
def alan_jokes : ℕ := sorry

/-- The total number of jokes both told over two Saturdays -/
def total_jokes : ℕ := 54

/-- Theorem stating that Alan told 7 jokes on Saturday -/
theorem alan_told_seven_jokes :
  alan_jokes = 7 ∧
  jessy_jokes + alan_jokes + 2 * jessy_jokes + 2 * alan_jokes = total_jokes :=
sorry

end NUMINAMATH_CALUDE_alan_told_seven_jokes_l3085_308514


namespace NUMINAMATH_CALUDE_max_red_socks_l3085_308584

theorem max_red_socks (x y : ℕ) : 
  x + y ≤ 2017 →
  (x * (x - 1) + y * (y - 1)) / ((x + y) * (x + y - 1)) = 1 / 2 →
  x ≤ 990 :=
by sorry

end NUMINAMATH_CALUDE_max_red_socks_l3085_308584


namespace NUMINAMATH_CALUDE_triangle_theorem_l3085_308562

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle. -/
theorem triangle_theorem (t : Triangle) 
  (h1 : 3 * t.a * Real.cos t.A - t.c * Real.cos t.B + t.b * Real.cos t.C = 0)
  (h2 : t.a = 2 * Real.sqrt 3)
  (h3 : Real.cos t.B + Real.cos t.C = 2 * Real.sqrt 3 / 3) :
  Real.cos t.A = 1/3 ∧ t.c = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3085_308562


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_formula_l3085_308505

/-- An arithmetic-geometric sequence -/
def ArithGeomSeq (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ n, a (n + 1) = a n * q

/-- The general term formula for the sequence -/
def GeneralTerm (a : ℕ → ℝ) : Prop :=
  (∃ (c : ℝ), ∀ n, a n = 125 * (2/5)^(n-1)) ∨
  (∃ (c : ℝ), ∀ n, a n = 8 * (5/2)^(n-1))

theorem arithmetic_geometric_sequence_formula (a : ℕ → ℝ) 
  (h1 : ArithGeomSeq a)
  (h2 : a 1 + a 4 = 133)
  (h3 : a 2 + a 3 = 70) :
  GeneralTerm a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_formula_l3085_308505


namespace NUMINAMATH_CALUDE_third_side_length_l3085_308500

-- Define a right-angled triangle with side lengths a, b, and x
structure RightTriangle where
  a : ℝ
  b : ℝ
  x : ℝ
  is_right : a^2 + b^2 = x^2 ∨ a^2 + x^2 = b^2 ∨ x^2 + b^2 = a^2

-- Define the theorem
theorem third_side_length (t : RightTriangle) :
  (t.a - 3)^2 + |t.b - 4| = 0 → t.x = 5 ∨ t.x = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_l3085_308500


namespace NUMINAMATH_CALUDE_square_division_theorem_l3085_308551

/-- Represents a cell in the square array -/
structure Cell where
  x : Nat
  y : Nat

/-- Represents a rectangle in the square array -/
structure Rectangle where
  top_left : Cell
  bottom_right : Cell

/-- Represents the state of a cell (pink or not pink) -/
inductive CellState
  | Pink
  | NotPink

/-- Represents the square array -/
def SquareArray (n : Nat) := Fin n → Fin n → CellState

/-- Checks if a rectangle contains exactly one pink cell -/
def containsOnePinkCell (arr : SquareArray n) (rect : Rectangle) : Prop := sorry

/-- Checks if a list of rectangles forms a valid division of the square -/
def isValidDivision (n : Nat) (rectangles : List Rectangle) : Prop := sorry

/-- The main theorem -/
theorem square_division_theorem (n : Nat) (arr : SquareArray n) :
  (∃ (i j : Fin n), arr i j = CellState.Pink) →
  ∃ (rectangles : List Rectangle), 
    isValidDivision n rectangles ∧ 
    ∀ rect ∈ rectangles, containsOnePinkCell arr rect :=
sorry

end NUMINAMATH_CALUDE_square_division_theorem_l3085_308551


namespace NUMINAMATH_CALUDE_intersection_coordinate_product_prove_intersection_coordinate_product_l3085_308510

/-- The product of coordinates of intersection points of two specific circles is 8 -/
theorem intersection_coordinate_product : ℝ → Prop := fun r =>
  ∀ x y : ℝ,
  (x^2 - 4*x + y^2 - 8*y + 20 = 0 ∧ x^2 - 6*x + y^2 - 8*y + 25 = 0) →
  r = x * y ∧ r = 8

/-- Proof of the theorem -/
theorem prove_intersection_coordinate_product :
  ∃ r : ℝ, intersection_coordinate_product r :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_coordinate_product_prove_intersection_coordinate_product_l3085_308510


namespace NUMINAMATH_CALUDE_toby_photo_shoot_l3085_308513

/-- The number of photos Toby took in the photo shoot -/
def photos_in_shoot (initial : ℕ) (deleted_bad : ℕ) (cat_pics : ℕ) (deleted_after : ℕ) (final : ℕ) : ℕ :=
  final - (initial - deleted_bad + cat_pics - deleted_after)

theorem toby_photo_shoot :
  photos_in_shoot 63 7 15 3 84 = 16 := by
  sorry

end NUMINAMATH_CALUDE_toby_photo_shoot_l3085_308513


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l3085_308596

theorem units_digit_of_quotient (n : ℕ) : (2^2023 + 3^2023) % 7 = 0 → 
  (2^2023 + 3^2023) / 7 % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l3085_308596


namespace NUMINAMATH_CALUDE_max_value_of_function_max_value_achieved_l3085_308540

theorem max_value_of_function (x : ℝ) (h : x < 0) : x + 4/x ≤ -4 := by
  sorry

theorem max_value_achieved (x : ℝ) (h : x < 0) : ∃ x₀, x₀ < 0 ∧ x₀ + 4/x₀ = -4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_max_value_achieved_l3085_308540


namespace NUMINAMATH_CALUDE_sons_age_l3085_308570

theorem sons_age (son_age man_age : ℕ) : 
  (man_age = son_age + 20) →
  (man_age + 2 = 2 * (son_age + 2)) →
  son_age = 18 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3085_308570


namespace NUMINAMATH_CALUDE_segment_bisection_l3085_308579

-- Define the angle
structure Angle where
  C : Point
  K : Point
  L : Point

-- Define the condition for a point to be inside an angle
def InsideAngle (α : Angle) (O : Point) : Prop := sorry

-- Define the condition for a point to be on a line
def OnLine (P Q R : Point) : Prop := sorry

-- Define the midpoint of a segment
def Midpoint (M A B : Point) : Prop := sorry

-- Main theorem
theorem segment_bisection (α : Angle) (O : Point) 
  (h : InsideAngle α O) : 
  ∃ (A B : Point), 
    OnLine α.C α.K A ∧ 
    OnLine α.C α.L B ∧ 
    Midpoint O A B :=
  sorry

end NUMINAMATH_CALUDE_segment_bisection_l3085_308579
