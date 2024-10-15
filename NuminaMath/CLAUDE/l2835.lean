import Mathlib

namespace NUMINAMATH_CALUDE_specific_hyperbola_real_axis_length_l2835_283525

/-- A hyperbola with given asymptotes and passing through a specific point -/
structure Hyperbola where
  -- The hyperbola passes through this point
  point : ℝ × ℝ
  -- The equations of the asymptotes
  asymptote1 : ℝ → ℝ → ℝ
  asymptote2 : ℝ → ℝ → ℝ

/-- The length of the real axis of a hyperbola -/
def realAxisLength (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem stating the length of the real axis of the specific hyperbola -/
theorem specific_hyperbola_real_axis_length :
  ∃ (h : Hyperbola),
    h.point = (5, -2) ∧
    h.asymptote1 = (λ x y => x - 2*y) ∧
    h.asymptote2 = (λ x y => x + 2*y) ∧
    realAxisLength h = 6 :=
  sorry

end NUMINAMATH_CALUDE_specific_hyperbola_real_axis_length_l2835_283525


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_682_l2835_283552

theorem sin_n_equals_cos_682 :
  ∃ n : ℤ, -120 ≤ n ∧ n ≤ 120 ∧ Real.sin (n * π / 180) = Real.cos (682 * π / 180) ∧ n = 128 := by
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_682_l2835_283552


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2835_283581

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- Properties of the specific arithmetic sequence -/
def SpecificSequence (seq : ArithmeticSequence) : Prop :=
  seq.S 5 < seq.S 6 ∧ seq.S 6 = seq.S 7 ∧ seq.S 7 > seq.S 8

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : SpecificSequence seq) : 
  seq.d < 0 ∧ 
  seq.S 9 < seq.S 5 ∧ 
  seq.a 7 = 0 ∧ 
  (∀ n, seq.S n ≤ seq.S 6 ∧ seq.S n ≤ seq.S 7) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2835_283581


namespace NUMINAMATH_CALUDE_inverse_f_at_142_l2835_283542

def f (x : ℝ) : ℝ := 5 * x^3 + 7

theorem inverse_f_at_142 : f⁻¹ 142 = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_142_l2835_283542


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l2835_283520

theorem quadratic_root_sum (p r : ℝ) (hp : p > 0) (hr : r > 0) :
  (∀ x : ℝ, x^2 - p*x + r = 0 → ∃ y : ℝ, y^2 - p*y + r = 0 ∧ x + y = 8) →
  r = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l2835_283520


namespace NUMINAMATH_CALUDE_polynomial_roots_product_l2835_283577

theorem polynomial_roots_product (d e : ℤ) : 
  (∀ r : ℝ, r^2 = r + 1 → r^6 = d*r + e) → d*e = 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_product_l2835_283577


namespace NUMINAMATH_CALUDE_q_age_is_40_l2835_283568

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Given two people P and Q, proves that Q's age is 40 years
    under the specified conditions -/
theorem q_age_is_40 (P Q : Person) :
  (∃ (y : ℕ), P.age = 3 * (Q.age - y) ∧ P.age - y = Q.age) →
  P.age + Q.age = 100 →
  Q.age = 40 := by
sorry


end NUMINAMATH_CALUDE_q_age_is_40_l2835_283568


namespace NUMINAMATH_CALUDE_divisibility_condition_l2835_283501

theorem divisibility_condition (n : ℕ) : 
  (∃ k : ℤ, (7 * n + 5 : ℤ) = k * (3 * n - 1)) ↔ n = 1 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2835_283501


namespace NUMINAMATH_CALUDE_stamps_needed_tara_stamps_problem_l2835_283541

theorem stamps_needed (current_stamps : ℕ) (stamps_per_sheet : ℕ) : ℕ :=
  stamps_per_sheet - (current_stamps % stamps_per_sheet)

theorem tara_stamps_problem : stamps_needed 38 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_stamps_needed_tara_stamps_problem_l2835_283541


namespace NUMINAMATH_CALUDE_trapezoid_properties_l2835_283504

/-- Represents a trapezoid ABCD with AB and CD as parallel bases (AB < CD) -/
structure Trapezoid where
  AD : ℝ  -- Length of larger base
  BC : ℝ  -- Length of smaller base
  AB : ℝ  -- Length of shorter leg
  midline : ℝ  -- Length of midline
  midpoint_segment : ℝ  -- Length of segment connecting midpoints of bases
  angle1 : ℝ  -- Angle at one end of larger base (in degrees)
  angle2 : ℝ  -- Angle at other end of larger base (in degrees)

/-- Theorem stating the properties of the specific trapezoid in the problem -/
theorem trapezoid_properties (T : Trapezoid) 
  (h1 : T.midline = 5)
  (h2 : T.midpoint_segment = 3)
  (h3 : T.angle1 = 30)
  (h4 : T.angle2 = 60) :
  T.AD = 8 ∧ T.BC = 2 ∧ T.AB = 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_properties_l2835_283504


namespace NUMINAMATH_CALUDE_total_net_buried_bones_l2835_283540

/-- Represents the types of bones Barkley receives --/
inductive BoneType
  | A
  | B
  | C

/-- Represents Barkley's bone statistics over 5 months --/
structure BoneStats where
  received : Nat
  buried : Nat
  eaten : Nat

/-- Calculates the net buried bones for a given BoneStats --/
def netBuried (stats : BoneStats) : Nat :=
  stats.buried - stats.eaten

/-- Defines Barkley's bone statistics for each type over 5 months --/
def barkleyStats : BoneType → BoneStats
  | BoneType.A => { received := 50, buried := 30, eaten := 3 }
  | BoneType.B => { received := 30, buried := 16, eaten := 2 }
  | BoneType.C => { received := 20, buried := 10, eaten := 2 }

/-- Theorem: The total net number of buried bones after 5 months is 49 --/
theorem total_net_buried_bones :
  (netBuried (barkleyStats BoneType.A) +
   netBuried (barkleyStats BoneType.B) +
   netBuried (barkleyStats BoneType.C)) = 49 := by
  sorry


end NUMINAMATH_CALUDE_total_net_buried_bones_l2835_283540


namespace NUMINAMATH_CALUDE_midpoint_of_specific_segment_l2835_283583

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Calculate the midpoint of two points in polar coordinates -/
def polarMidpoint (p1 p2 : PolarPoint) : PolarPoint :=
  sorry

theorem midpoint_of_specific_segment :
  let p1 : PolarPoint := ⟨10, π/4⟩
  let p2 : PolarPoint := ⟨10, 3*π/4⟩
  let midpoint := polarMidpoint p1 p2
  midpoint.r = 5 * Real.sqrt 2 ∧ midpoint.θ = π/2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_of_specific_segment_l2835_283583


namespace NUMINAMATH_CALUDE_exists_divisible_by_sum_of_digits_remainder_at_most_15_exists_number_for_remainder_l2835_283594

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n ≤ 99 }

/-- Sum of digits of a two-digit number -/
def sumOfDigits (n : TwoDigitNumber) : ℕ :=
  n.val / 10 + n.val % 10

/-- Theorem 1: There exists a two-digit number divisible by the sum of its digits -/
theorem exists_divisible_by_sum_of_digits :
  ∃ n : TwoDigitNumber, n.val % (sumOfDigits n) = 0 :=
sorry

/-- Theorem 2: The remainder when a two-digit number is divided by the sum of its digits is at most 15 -/
theorem remainder_at_most_15 (n : TwoDigitNumber) :
  n.val % (sumOfDigits n) ≤ 15 :=
sorry

/-- Theorem 3: For any remainder r ≤ 12, there exists a two-digit number that produces that remainder -/
theorem exists_number_for_remainder (r : ℕ) (h : r ≤ 12) :
  ∃ n : TwoDigitNumber, n.val % (sumOfDigits n) = r :=
sorry

end NUMINAMATH_CALUDE_exists_divisible_by_sum_of_digits_remainder_at_most_15_exists_number_for_remainder_l2835_283594


namespace NUMINAMATH_CALUDE_bisection_method_result_l2835_283562

def f (x : ℝ) := x^3 - 3*x + 1

theorem bisection_method_result :
  ∃ x₀ : ℝ, f x₀ = 0 ∧ 0 < x₀ ∧ x₀ < 1 →
  ∃ a b : ℝ, 1/4 < a ∧ a < x₀ ∧ x₀ < b ∧ b < 1/2 ∧
    f a * f b < 0 ∧
    ∀ c ∈ Set.Ioo (0 : ℝ) 1, f c * f (1/2) ≤ 0 → c ≤ 1/2 ∧
    ∀ d ∈ Set.Ioo (0 : ℝ) (1/2), f d * f (1/4) ≤ 0 → 1/4 ≤ d :=
by sorry

end NUMINAMATH_CALUDE_bisection_method_result_l2835_283562


namespace NUMINAMATH_CALUDE_matrix_equality_zero_l2835_283566

open Matrix

theorem matrix_equality_zero (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ) 
  (h1 : A * B = B) (h2 : det (A - 1) ≠ 0) : B = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_zero_l2835_283566


namespace NUMINAMATH_CALUDE_problem_solution_l2835_283572

def f (x a : ℝ) := |x - a| * x + |x - 2| * (x - a)

theorem problem_solution :
  (∀ x, f x 1 < 0 ↔ x ∈ Set.Iio 1) ∧
  (∀ a, (∀ x, x ∈ Set.Iio 1 → f x a < 0) ↔ a ∈ Set.Ici 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2835_283572


namespace NUMINAMATH_CALUDE_youngest_child_age_l2835_283517

/-- Given a family where:
    1. 10 years ago, the average age of 4 members was 24 years
    2. Two children were born with an age difference of 2 years
    3. The present average age of the family (now 6 members) is still 24 years
    Prove that the present age of the youngest child is 3 years -/
theorem youngest_child_age
  (past_average_age : ℕ)
  (past_family_size : ℕ)
  (years_passed : ℕ)
  (present_average_age : ℕ)
  (present_family_size : ℕ)
  (age_difference : ℕ)
  (h1 : past_average_age = 24)
  (h2 : past_family_size = 4)
  (h3 : years_passed = 10)
  (h4 : present_average_age = 24)
  (h5 : present_family_size = 6)
  (h6 : age_difference = 2) :
  ∃ (youngest_age : ℕ), youngest_age = 3 ∧
    present_average_age * present_family_size =
    (past_average_age * past_family_size + years_passed * past_family_size + youngest_age + (youngest_age + age_difference)) :=
by sorry

end NUMINAMATH_CALUDE_youngest_child_age_l2835_283517


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2835_283500

/-- Represents a digit in the set {1, 2, 3, 4, 5, 6} -/
def Digit := Fin 6

/-- Represents the multiplication problem AB × C = DEF -/
def IsValidMultiplication (a b c d e f : Digit) : Prop :=
  (a.val + 1) * 10 + (b.val + 1) = (d.val + 1) * 100 + (e.val + 1) * 10 + (f.val + 1)

/-- All digits are distinct -/
def AreDistinct (a b c d e f : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

theorem multiplication_puzzle :
  ∀ (a b c d e f : Digit),
    IsValidMultiplication a b c d e f →
    AreDistinct a b c d e f →
    c.val = 2 :=
by sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2835_283500


namespace NUMINAMATH_CALUDE_rhombus_diagonal_sum_l2835_283565

/-- A rhombus with specific properties -/
structure Rhombus where
  longer_diagonal : ℝ
  shorter_diagonal : ℝ
  area : ℝ
  diagonal_diff : longer_diagonal - shorter_diagonal = 4
  area_eq : area = 6
  positive_diagonals : longer_diagonal > 0 ∧ shorter_diagonal > 0

/-- The sum of diagonals in a rhombus with given properties is 8 -/
theorem rhombus_diagonal_sum (r : Rhombus) : r.longer_diagonal + r.shorter_diagonal = 8 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_sum_l2835_283565


namespace NUMINAMATH_CALUDE_new_average_height_l2835_283564

/-- Calculates the new average height of a class after some students leave and others join. -/
theorem new_average_height
  (initial_size : ℕ)
  (initial_avg : ℝ)
  (left_size : ℕ)
  (left_avg : ℝ)
  (joined_size : ℕ)
  (joined_avg : ℝ)
  (h_initial_size : initial_size = 35)
  (h_initial_avg : initial_avg = 180)
  (h_left_size : left_size = 7)
  (h_left_avg : left_avg = 120)
  (h_joined_size : joined_size = 7)
  (h_joined_avg : joined_avg = 140)
  : (initial_size * initial_avg - left_size * left_avg + joined_size * joined_avg) / initial_size = 184 := by
  sorry

end NUMINAMATH_CALUDE_new_average_height_l2835_283564


namespace NUMINAMATH_CALUDE_triangle_lattice_distance_product_l2835_283514

theorem triangle_lattice_distance_product (x y : ℝ) 
  (hx : ∃ (a b : ℤ), x^2 = a^2 + a*b + b^2)
  (hy : ∃ (c d : ℤ), y^2 = c^2 + c*d + d^2) :
  ∃ (e f : ℤ), (x*y)^2 = e^2 + e*f + f^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_lattice_distance_product_l2835_283514


namespace NUMINAMATH_CALUDE_simplify_expression_l2835_283591

theorem simplify_expression (a : ℝ) : 2*a + 1 - (1 - a) = 3*a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2835_283591


namespace NUMINAMATH_CALUDE_parabola_through_point_l2835_283511

-- Define a parabola
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  -- ax² + by² = c

-- Define the point (1, -2)
def point : ℝ × ℝ := (1, -2)

-- Theorem statement
theorem parabola_through_point :
  ∃ (p1 p2 : Parabola),
    (p1.a = 0 ∧ p1.b = 1 ∧ p1.c = 4 ∧ p1.a * point.1^2 + p1.b * point.2^2 = p1.c) ∨
    (p2.a = 1 ∧ p2.b = -1/2 ∧ p2.c = 0 ∧ p2.a * point.1^2 + p2.b * point.2 = p2.c) :=
by sorry

end NUMINAMATH_CALUDE_parabola_through_point_l2835_283511


namespace NUMINAMATH_CALUDE_original_profit_percentage_l2835_283502

/-- Calculates the profit percentage given the cost price and selling price -/
def profitPercentage (costPrice sellingPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

theorem original_profit_percentage
  (costPrice : ℚ)
  (sellingPrice : ℚ)
  (h1 : costPrice = 80)
  (h2 : profitPercentage (costPrice * (1 - 0.2)) (sellingPrice - 16.8) = 30) :
  profitPercentage costPrice sellingPrice = 25 := by
  sorry

end NUMINAMATH_CALUDE_original_profit_percentage_l2835_283502


namespace NUMINAMATH_CALUDE_number_multiplication_l2835_283531

theorem number_multiplication (x : ℝ) : x - 7 = 9 → x * 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_l2835_283531


namespace NUMINAMATH_CALUDE_B_equals_roster_l2835_283544

def A : Set Int := {-2, 2, 3, 4}

def B : Set Int := {x | ∃ t ∈ A, x = t^2}

theorem B_equals_roster : B = {4, 9, 16} := by sorry

end NUMINAMATH_CALUDE_B_equals_roster_l2835_283544


namespace NUMINAMATH_CALUDE_matthew_crackers_l2835_283585

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 55

/-- The number of cakes Matthew had -/
def cakes : ℕ := 34

/-- The number of friends Matthew gave crackers and cakes to -/
def friends : ℕ := 11

/-- The number of crackers each person ate -/
def crackers_eaten_per_person : ℕ := 2

theorem matthew_crackers :
  (cakes / friends = initial_crackers / friends) ∧
  (friends * crackers_eaten_per_person + friends * (cakes / friends) = initial_crackers) :=
by sorry

end NUMINAMATH_CALUDE_matthew_crackers_l2835_283585


namespace NUMINAMATH_CALUDE_square_pentagon_side_ratio_l2835_283560

theorem square_pentagon_side_ratio :
  ∀ (s_s s_p : ℝ),
  s_s > 0 → s_p > 0 →
  s_s^2 = (5 * s_p^2 * (Real.sqrt 5 + 1)) / 8 →
  s_p / s_s = Real.sqrt (8 / (5 * (Real.sqrt 5 + 1))) :=
by sorry

end NUMINAMATH_CALUDE_square_pentagon_side_ratio_l2835_283560


namespace NUMINAMATH_CALUDE_library_visitors_l2835_283533

/-- Proves that the average number of visitors on non-Sunday days is 240 --/
theorem library_visitors (sunday_visitors : ℕ) (total_days : ℕ) (sundays : ℕ) (avg_visitors : ℕ) :
  sunday_visitors = 510 →
  total_days = 30 →
  sundays = 5 →
  avg_visitors = 285 →
  (sundays * sunday_visitors + (total_days - sundays) * 
    ((total_days * avg_visitors - sundays * sunday_visitors) / (total_days - sundays))) 
    / total_days = avg_visitors →
  (total_days * avg_visitors - sundays * sunday_visitors) / (total_days - sundays) = 240 := by
sorry

#eval (30 * 285 - 5 * 510) / (30 - 5)  -- Should output 240

end NUMINAMATH_CALUDE_library_visitors_l2835_283533


namespace NUMINAMATH_CALUDE_max_sock_price_l2835_283599

theorem max_sock_price (total_money : ℕ) (entrance_fee : ℕ) (num_socks : ℕ) (tax_rate : ℚ) :
  total_money = 180 →
  entrance_fee = 3 →
  num_socks = 20 →
  tax_rate = 6 / 100 →
  ∃ (max_price : ℕ), 
    max_price = 8 ∧
    (max_price : ℚ) * num_socks * (1 + tax_rate) + entrance_fee ≤ total_money ∧
    ∀ (price : ℕ), 
      price > max_price → 
      (price : ℚ) * num_socks * (1 + tax_rate) + entrance_fee > total_money :=
by sorry

end NUMINAMATH_CALUDE_max_sock_price_l2835_283599


namespace NUMINAMATH_CALUDE_egyptian_fraction_sum_l2835_283584

theorem egyptian_fraction_sum : ∃ (b₂ b₃ b₄ b₅ b₆ : ℕ),
  (4 : ℚ) / 9 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 ∧
  b₂ < 2 ∧ b₃ < 3 ∧ b₄ < 4 ∧ b₅ < 5 ∧ b₆ < 6 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ = 9 ∧
  b₂ ≠ b₃ ∧ b₂ ≠ b₄ ∧ b₂ ≠ b₅ ∧ b₂ ≠ b₆ ∧
  b₃ ≠ b₄ ∧ b₃ ≠ b₅ ∧ b₃ ≠ b₆ ∧
  b₄ ≠ b₅ ∧ b₄ ≠ b₆ ∧
  b₅ ≠ b₆ := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_sum_l2835_283584


namespace NUMINAMATH_CALUDE_stapler_equation_l2835_283510

theorem stapler_equation (sheets : ℕ) (time_first time_combined : ℝ) (time_second : ℝ) :
  sheets > 0 ∧ time_first > 0 ∧ time_combined > 0 ∧ time_second > 0 →
  (sheets / time_first + sheets / time_second = sheets / time_combined) ↔
  (1 / time_first + 1 / time_second = 1 / time_combined) :=
by sorry

end NUMINAMATH_CALUDE_stapler_equation_l2835_283510


namespace NUMINAMATH_CALUDE_probability_three_dice_divisible_by_10_l2835_283526

-- Define a die as having 6 faces
def die_faces : ℕ := 6

-- Define a function to check if a number is divisible by 2
def divisible_by_2 (n : ℕ) : Prop := n % 2 = 0

-- Define a function to check if a number is divisible by 5
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Define a function to check if a product is divisible by 10
def product_divisible_by_10 (a b c : ℕ) : Prop :=
  divisible_by_2 (a * b * c) ∧ divisible_by_5 (a * b * c)

-- Define the probability of the event
def probability_divisible_by_10 : ℚ :=
  (144 : ℚ) / (die_faces ^ 3 : ℚ)

-- State the theorem
theorem probability_three_dice_divisible_by_10 :
  probability_divisible_by_10 = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_three_dice_divisible_by_10_l2835_283526


namespace NUMINAMATH_CALUDE_matrix_from_eigenvectors_l2835_283586

theorem matrix_from_eigenvectors (A : Matrix (Fin 2) (Fin 2) ℝ) :
  (A.mulVec (![1, -3]) = ![-1, 3]) →
  (A.mulVec (![1, 1]) = ![3, 3]) →
  A = !![2, 1; 3, 0] := by
sorry

end NUMINAMATH_CALUDE_matrix_from_eigenvectors_l2835_283586


namespace NUMINAMATH_CALUDE_problem_statement_l2835_283596

theorem problem_statement (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ (4/x + 1/y ≥ 4/m + 1/n)) ∧
  (4/m + 1/n ≥ 9) ∧
  (Real.sqrt m + Real.sqrt n ≤ Real.sqrt 2) ∧
  (m > n → 1/(m-1) < 1/(n-1)) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2835_283596


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2835_283569

theorem simplify_fraction_product : (240 / 12) * (5 / 150) * (12 / 3) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2835_283569


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2835_283507

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 12,
    prove that the fourth term is 6. -/
theorem arithmetic_sequence_fourth_term (a d : ℝ) 
  (h : (a + 2*d) + (a + 4*d) = 12) : a + 3*d = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2835_283507


namespace NUMINAMATH_CALUDE_benzene_formation_enthalpy_l2835_283559

-- Define the substances
def C : Type := Unit
def H₂ : Type := Unit
def C₂H₂ : Type := Unit
def C₆H₆ : Type := Unit

-- Define the states
inductive State
| Gas
| Liquid
| Graphite

-- Define a reaction
structure Reaction :=
  (reactants : List (Type × State × ℕ))
  (products : List (Type × State × ℕ))
  (heat_effect : ℝ)

-- Given reactions
def reaction1 : Reaction :=
  ⟨[(C₂H₂, State.Gas, 1)], [(C, State.Graphite, 2), (H₂, State.Gas, 1)], 226.7⟩

def reaction2 : Reaction :=
  ⟨[(C₂H₂, State.Gas, 3)], [(C₆H₆, State.Liquid, 1)], 631.1⟩

def reaction3 : Reaction :=
  ⟨[(C₆H₆, State.Liquid, 1)], [(C₆H₆, State.Liquid, 1)], -33.9⟩

-- Standard enthalpy of formation
def standard_enthalpy_of_formation (substance : Type) (state : State) : ℝ := sorry

-- Theorem statement
theorem benzene_formation_enthalpy :
  standard_enthalpy_of_formation C₆H₆ State.Liquid = -82.9 :=
sorry

end NUMINAMATH_CALUDE_benzene_formation_enthalpy_l2835_283559


namespace NUMINAMATH_CALUDE_rhombus_area_l2835_283578

/-- The area of a rhombus with side length 5 cm and an interior angle of 60 degrees is 12.5√3 square centimeters. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 5) (h2 : θ = π / 3) :
  s * s * Real.sin θ = 25 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l2835_283578


namespace NUMINAMATH_CALUDE_function_is_constant_one_l2835_283508

/-- A function satisfying the given conditions is constant and equal to 1 -/
theorem function_is_constant_one (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = 2 - f x) 
  (h2 : ∀ x, f (x + 3) ≥ f x) : 
  ∀ x, f x = 1 := by sorry

end NUMINAMATH_CALUDE_function_is_constant_one_l2835_283508


namespace NUMINAMATH_CALUDE_lansing_elementary_students_l2835_283512

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := 247

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := num_schools * students_per_school

theorem lansing_elementary_students :
  total_students = 6175 :=
sorry

end NUMINAMATH_CALUDE_lansing_elementary_students_l2835_283512


namespace NUMINAMATH_CALUDE_part1_part2_l2835_283537

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

-- Part 1
theorem part1 (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ x, x ∈ Set.Icc 1 a ↔ f a x ∈ Set.Icc 1 a) : 
  a = 2 := by sorry

-- Part 2
theorem part2 (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ x₁ x₂, x₁ ∈ Set.Icc 1 (a+1) → x₂ ∈ Set.Icc 1 (a+1) → 
    |f a x₁ - f a x₂| ≤ 4) : 
  1 < a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2835_283537


namespace NUMINAMATH_CALUDE_geometric_distribution_sum_to_one_l2835_283573

/-- The probability mass function for a geometric distribution -/
def geometric_pmf (p : ℝ) (m : ℕ) : ℝ := (1 - p) ^ (m - 1) * p

/-- Theorem: The sum of probabilities for a geometric distribution equals 1 -/
theorem geometric_distribution_sum_to_one (p : ℝ) (hp : 0 < p) (hp' : p < 1) :
  ∑' m : ℕ, geometric_pmf p m = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_distribution_sum_to_one_l2835_283573


namespace NUMINAMATH_CALUDE_probability_product_not_odd_l2835_283567

/-- Represents a standard six-sided die -/
def Die : Type := Fin 6

/-- The set of all possible outcomes when rolling two dice -/
def TwoDiceOutcomes : Type := Die × Die

/-- Predicate to check if a number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- Predicate to check if the product of two die rolls is not odd -/
def productNotOdd (roll : TwoDiceOutcomes) : Prop :=
  ¬isOdd ((roll.1.val + 1) * (roll.2.val + 1))

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := 36

/-- The number of outcomes where the product is not odd -/
def favorableOutcomes : ℕ := 27

theorem probability_product_not_odd :
  (favorableOutcomes : ℚ) / totalOutcomes = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_product_not_odd_l2835_283567


namespace NUMINAMATH_CALUDE_marble_probability_l2835_283545

theorem marble_probability (a b c : ℕ) : 
  a + b + c = 97 →
  (a * (a - 1) + b * (b - 1) + c * (c - 1)) / (97 * 96) = 5 / 12 →
  (a^2 + b^2 + c^2) / 97^2 = 41 / 97 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l2835_283545


namespace NUMINAMATH_CALUDE_angle_sets_relation_l2835_283535

-- Define the sets A, B, and C
def A : Set ℝ := {θ | ∃ k : ℤ, 2 * k * Real.pi < θ ∧ θ < 2 * k * Real.pi + Real.pi / 2}
def B : Set ℝ := {θ | 0 < θ ∧ θ < Real.pi / 2}
def C : Set ℝ := {θ | θ < Real.pi / 2}

-- State the theorem
theorem angle_sets_relation : B ∪ C = C := by
  sorry

end NUMINAMATH_CALUDE_angle_sets_relation_l2835_283535


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l2835_283503

theorem correct_mean_calculation (n : ℕ) (incorrect_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) :
  n = 30 ∧ 
  incorrect_mean = 150 ∧ 
  incorrect_value = 135 ∧ 
  correct_value = 165 →
  (n * incorrect_mean - incorrect_value + correct_value) / n = 151 := by sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l2835_283503


namespace NUMINAMATH_CALUDE_basil_daytime_cookies_l2835_283557

/-- Represents the number of cookies Basil gets per day -/
structure BasilCookies where
  morning : ℚ
  evening : ℚ
  daytime : ℕ

/-- Represents the cookie box information -/
structure CookieBox where
  cookies_per_box : ℕ
  boxes_needed : ℕ
  days_lasting : ℕ

theorem basil_daytime_cookies 
  (basil_cookies : BasilCookies)
  (cookie_box : CookieBox)
  (h1 : basil_cookies.morning = 1/2)
  (h2 : basil_cookies.evening = 1/2)
  (h3 : cookie_box.cookies_per_box = 45)
  (h4 : cookie_box.boxes_needed = 2)
  (h5 : cookie_box.days_lasting = 30) :
  basil_cookies.daytime = 2 :=
sorry

end NUMINAMATH_CALUDE_basil_daytime_cookies_l2835_283557


namespace NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l2835_283582

theorem arithmetic_mean_sqrt2 :
  (Real.sqrt 2 + 1 + (Real.sqrt 2 - 1)) / 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sqrt2_l2835_283582


namespace NUMINAMATH_CALUDE_cubic_roots_bound_l2835_283532

-- Define the polynomial
def cubic_polynomial (p q x : ℝ) : ℝ := x^3 + p*x + q

-- Define the condition for roots not exceeding 1 in modulus
def roots_within_unit_circle (p q : ℝ) : Prop :=
  ∀ x : ℂ, cubic_polynomial p q x.re = 0 → Complex.abs x ≤ 1

-- Theorem statement
theorem cubic_roots_bound (p q : ℝ) :
  roots_within_unit_circle p q ↔ p > abs q - 1 :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_bound_l2835_283532


namespace NUMINAMATH_CALUDE_union_with_complement_l2835_283546

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set P
def P : Set Nat := {1, 2}

-- Define set Q
def Q : Set Nat := {1, 3}

-- Theorem statement
theorem union_with_complement :
  P ∪ (U \ Q) = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_union_with_complement_l2835_283546


namespace NUMINAMATH_CALUDE_farm_animals_percentage_l2835_283530

theorem farm_animals_percentage (cows ducks pigs : ℕ) : 
  cows = 20 →
  pigs = (ducks + cows) / 5 →
  cows + ducks + pigs = 60 →
  (ducks - cows : ℚ) / cows * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_percentage_l2835_283530


namespace NUMINAMATH_CALUDE_exists_common_element_l2835_283536

/-- A collection of 1978 sets, each containing 40 elements -/
def SetCollection := Fin 1978 → Finset (Fin (1978 * 40))

/-- The property that any two sets in the collection have exactly one common element -/
def OneCommonElement (collection : SetCollection) : Prop :=
  ∀ i j, i ≠ j → (collection i ∩ collection j).card = 1

/-- The theorem stating that there exists an element in all sets of the collection -/
theorem exists_common_element (collection : SetCollection)
  (h1 : ∀ i, (collection i).card = 40)
  (h2 : OneCommonElement collection) :
  ∃ x, ∀ i, x ∈ collection i :=
sorry

end NUMINAMATH_CALUDE_exists_common_element_l2835_283536


namespace NUMINAMATH_CALUDE_triangle_shape_not_unique_l2835_283515

/-- A triangle with sides a, b, c and angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The shape of a triangle is not uniquely determined by the product of two sides and the angle between them --/
theorem triangle_shape_not_unique (p : ℝ) (γ : ℝ) :
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ t1.a * t1.b = p ∧ t1.C = γ ∧ t2.a * t2.b = p ∧ t2.C = γ :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_not_unique_l2835_283515


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_ratio_l2835_283590

theorem equilateral_triangle_area_ratio :
  ∀ s : ℝ,
  s > 0 →
  let small_triangle_area := (s^2 * Real.sqrt 3) / 4
  let large_triangle_side := 3 * s
  let large_triangle_area := (large_triangle_side^2 * Real.sqrt 3) / 4
  (3 * small_triangle_area) / large_triangle_area = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_ratio_l2835_283590


namespace NUMINAMATH_CALUDE_square_overlap_ratio_l2835_283593

theorem square_overlap_ratio (a b : ℝ) 
  (h1 : 0.52 * a^2 = a^2 - (a^2 - 0.73 * b^2)) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  a / b = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_square_overlap_ratio_l2835_283593


namespace NUMINAMATH_CALUDE_christmas_to_birthday_ratio_l2835_283529

def total_presents : ℕ := 90
def christmas_presents : ℕ := 60

theorem christmas_to_birthday_ratio :
  (christmas_presents : ℚ) / (total_presents - christmas_presents : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_christmas_to_birthday_ratio_l2835_283529


namespace NUMINAMATH_CALUDE_evaluate_polynomial_at_negative_two_l2835_283597

theorem evaluate_polynomial_at_negative_two :
  let y : ℤ := -2
  y^3 - y^2 + 2*y + 4 = -12 := by
sorry

end NUMINAMATH_CALUDE_evaluate_polynomial_at_negative_two_l2835_283597


namespace NUMINAMATH_CALUDE_equal_charges_at_300_minutes_l2835_283549

/-- Represents a mobile phone plan -/
structure PhonePlan where
  monthly_fee : ℝ
  call_rate : ℝ

/-- Calculates the monthly bill for a given plan and call duration -/
def monthly_bill (plan : PhonePlan) (duration : ℝ) : ℝ :=
  plan.monthly_fee + plan.call_rate * duration

/-- The Unicom company's phone plans -/
def plan_a : PhonePlan := { monthly_fee := 15, call_rate := 0.1 }
def plan_b : PhonePlan := { monthly_fee := 0, call_rate := 0.15 }

theorem equal_charges_at_300_minutes : 
  ∃ (duration : ℝ), duration = 300 ∧ 
    monthly_bill plan_a duration = monthly_bill plan_b duration := by
  sorry

end NUMINAMATH_CALUDE_equal_charges_at_300_minutes_l2835_283549


namespace NUMINAMATH_CALUDE_integer_condition_l2835_283558

theorem integer_condition (x : ℝ) : 
  (∀ x : ℤ, ∃ y : ℤ, 2 * (x : ℝ) + 1 = y) ∧ 
  (∃ x : ℝ, ∃ y : ℤ, 2 * x + 1 = y ∧ ¬∃ z : ℤ, x = z) :=
sorry

end NUMINAMATH_CALUDE_integer_condition_l2835_283558


namespace NUMINAMATH_CALUDE_monic_quartic_problem_l2835_283556

-- Define a monic quartic polynomial
def monicQuartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + f 0

-- State the theorem
theorem monic_quartic_problem (f : ℝ → ℝ) 
  (h_monic : monicQuartic f)
  (h_neg2 : f (-2) = 0)
  (h_3 : f 3 = -9)
  (h_neg4 : f (-4) = -16)
  (h_5 : f 5 = -25) :
  f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_monic_quartic_problem_l2835_283556


namespace NUMINAMATH_CALUDE_expand_polynomial_l2835_283579

theorem expand_polynomial (x : ℝ) : 
  (x - 2) * (x + 2) * (x^3 + 3*x + 1) = x^5 - x^3 + x^2 - 12*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l2835_283579


namespace NUMINAMATH_CALUDE_probability_three_digit_l2835_283513

def set_start : ℕ := 60
def set_end : ℕ := 1000

def three_digit_start : ℕ := 100
def three_digit_end : ℕ := 999

def total_numbers : ℕ := set_end - set_start + 1
def three_digit_numbers : ℕ := three_digit_end - (three_digit_start - 1)

theorem probability_three_digit :
  (three_digit_numbers : ℚ) / total_numbers = 901 / 941 := by sorry

end NUMINAMATH_CALUDE_probability_three_digit_l2835_283513


namespace NUMINAMATH_CALUDE_remainder_of_sum_is_zero_l2835_283592

-- Define the arithmetic sequence
def arithmeticSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Define the sum of the arithmetic sequence
def sumArithmeticSequence (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ := n * (a₁ + aₙ) / 2

-- Theorem statement
theorem remainder_of_sum_is_zero :
  let a₁ := 3
  let aₙ := 309
  let d := 6
  let n := (aₙ - a₁) / d + 1
  (sumArithmeticSequence a₁ aₙ n) % 6 = 0 := by
    sorry

end NUMINAMATH_CALUDE_remainder_of_sum_is_zero_l2835_283592


namespace NUMINAMATH_CALUDE_social_practice_arrangements_l2835_283554

def number_of_teachers : Nat := 2
def number_of_students : Nat := 6
def teachers_per_group : Nat := 1
def students_per_group : Nat := 3
def number_of_groups : Nat := 2

theorem social_practice_arrangements :
  (number_of_teachers.choose teachers_per_group) *
  (number_of_students.choose students_per_group) = 40 := by
  sorry

end NUMINAMATH_CALUDE_social_practice_arrangements_l2835_283554


namespace NUMINAMATH_CALUDE_triangle_and_star_operations_l2835_283587

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := a^2 - a * b

-- Define the star operation
def star (a b : ℚ) : ℚ := 3 * a * b - b^2

theorem triangle_and_star_operations : 
  (triangle (-3 : ℚ) 5 = 24) ∧ 
  (star (-4 : ℚ) (triangle 2 3) = 20) := by
  sorry

end NUMINAMATH_CALUDE_triangle_and_star_operations_l2835_283587


namespace NUMINAMATH_CALUDE_davids_crunches_l2835_283509

theorem davids_crunches (zachary_crunches : ℕ) (david_less_crunches : ℕ) 
  (h1 : zachary_crunches = 17)
  (h2 : david_less_crunches = 13) :
  zachary_crunches - david_less_crunches = 4 := by
  sorry

end NUMINAMATH_CALUDE_davids_crunches_l2835_283509


namespace NUMINAMATH_CALUDE_min_value_expression_l2835_283576

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x = y) :
  ∃ (min : ℝ), min = 0 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a = b →
    (a + 1/b) * (a + 1/b - 2) + (b + 1/a) * (b + 1/a - 2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2835_283576


namespace NUMINAMATH_CALUDE_triangle_side_length_l2835_283543

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  b = 7 →
  c = 6 →
  Real.cos (B - C) = 15/16 →
  a = 5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2835_283543


namespace NUMINAMATH_CALUDE_meeting_attendees_l2835_283547

theorem meeting_attendees (total_handshakes : ℕ) (h : total_handshakes = 66) :
  ∃ (n : ℕ), n * (n - 1) / 2 = total_handshakes ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_meeting_attendees_l2835_283547


namespace NUMINAMATH_CALUDE_triangle_centroid_property_l2835_283524

open Real

variable (A B C Q G' : ℝ × ℝ)

def is_inside_triangle (P A B C : ℝ × ℝ) : Prop := sorry

def distance_squared (P Q : ℝ × ℝ) : ℝ := 
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem triangle_centroid_property :
  is_inside_triangle G' A B C →
  G' = ((1/4 : ℝ) • A + (1/4 : ℝ) • B + (1/2 : ℝ) • C) →
  distance_squared Q A + distance_squared Q B + distance_squared Q C = 
  4 * distance_squared Q G' + distance_squared G' A + distance_squared G' B + distance_squared G' C :=
by sorry

end NUMINAMATH_CALUDE_triangle_centroid_property_l2835_283524


namespace NUMINAMATH_CALUDE_fruit_shop_results_l2835_283516

/-- Represents the fruit inventory and pricing information for a shopkeeper --/
structure FruitShop where
  totalFruits : Nat
  oranges : Nat
  bananas : Nat
  apples : Nat
  rottenOrangesPercent : Rat
  rottenBananasPercent : Rat
  rottenApplesPercent : Rat
  orangePurchasePrice : Rat
  bananaPurchasePrice : Rat
  applePurchasePrice : Rat
  orangeSellingPrice : Rat
  bananaSellingPrice : Rat
  appleSellingPrice : Rat

/-- Calculates the percentage of fruits in good condition and the overall profit --/
def calculateResults (shop : FruitShop) : (Rat × Rat) :=
  sorry

/-- Theorem stating the correct percentage of good fruits and overall profit --/
theorem fruit_shop_results (shop : FruitShop) 
  (h1 : shop.totalFruits = 1000)
  (h2 : shop.oranges = 600)
  (h3 : shop.bananas = 300)
  (h4 : shop.apples = 100)
  (h5 : shop.rottenOrangesPercent = 15/100)
  (h6 : shop.rottenBananasPercent = 8/100)
  (h7 : shop.rottenApplesPercent = 20/100)
  (h8 : shop.orangePurchasePrice = 60/100)
  (h9 : shop.bananaPurchasePrice = 30/100)
  (h10 : shop.applePurchasePrice = 1)
  (h11 : shop.orangeSellingPrice = 120/100)
  (h12 : shop.bananaSellingPrice = 60/100)
  (h13 : shop.appleSellingPrice = 150/100) :
  calculateResults shop = (866/1000, 3476/10) := by
  sorry


end NUMINAMATH_CALUDE_fruit_shop_results_l2835_283516


namespace NUMINAMATH_CALUDE_treasure_chest_age_conversion_l2835_283580

/-- Converts a number from base 8 to base 10 --/
def base8ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The age of the treasure chest in base 8 --/
def treasureChestAgeBase8 : Nat × Nat × Nat := (3, 4, 7)

theorem treasure_chest_age_conversion :
  let (h, t, o) := treasureChestAgeBase8
  base8ToBase10 h t o = 231 := by
  sorry

end NUMINAMATH_CALUDE_treasure_chest_age_conversion_l2835_283580


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2835_283595

theorem quadratic_equation_unique_solution :
  ∃! x : ℝ, x^2 + 2*x + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l2835_283595


namespace NUMINAMATH_CALUDE_no_integer_root_l2835_283539

theorem no_integer_root (P : ℤ → ℤ) (h_poly : ∀ x y : ℤ, (x - y) ∣ (P x - P y)) 
  (h1 : P 1 = 10) (h_neg1 : P (-1) = 22) (h0 : P 0 = 4) :
  ∀ r : ℤ, P r ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_no_integer_root_l2835_283539


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l2835_283561

-- Define the complex number z(a)
def z (a : ℝ) : ℂ := (a - 1) * (a + 2) + (a + 3) * Complex.I

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

-- State the theorem
theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → isPurelyImaginary (z a)) ∧
  ¬(∀ a : ℝ, isPurelyImaginary (z a) → a = 1) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l2835_283561


namespace NUMINAMATH_CALUDE_initial_pennies_equation_l2835_283528

/-- Given that Sam spent some pennies and has some left, prove that his initial number of pennies
    is equal to the sum of pennies spent and pennies left. -/
theorem initial_pennies_equation (initial spent left : ℕ) : 
  spent = 93 → left = 5 → initial = spent + left := by sorry

end NUMINAMATH_CALUDE_initial_pennies_equation_l2835_283528


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2835_283571

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) :
  area = (1/2) * leg1 * leg2 →
  leg1 = 30 →
  area = 150 →
  leg2^2 + leg1^2 = hypotenuse^2 →
  leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2835_283571


namespace NUMINAMATH_CALUDE_f_positive_range_l2835_283563

/-- A function f that is strictly increasing for x > 0 and symmetric about the y-axis -/
noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x => (a * Real.exp x + b) * (x - 2)

/-- The theorem stating the range of m for which f(2-m) > 0 -/
theorem f_positive_range (a b : ℝ) :
  (∀ x > 0, Monotone (f a b)) →
  (∀ x, f a b x = f a b (-x)) →
  {m : ℝ | f a b (2 - m) > 0} = {m : ℝ | m < 0 ∨ m > 4} := by
  sorry

end NUMINAMATH_CALUDE_f_positive_range_l2835_283563


namespace NUMINAMATH_CALUDE_cube_of_sum_and_reciprocal_l2835_283588

theorem cube_of_sum_and_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 3) :
  (a + 1/a)^3 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_sum_and_reciprocal_l2835_283588


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2835_283574

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n → n ≥ 102 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2835_283574


namespace NUMINAMATH_CALUDE_largest_ball_radius_is_four_l2835_283534

/-- Represents a torus in 3D space --/
structure Torus where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  inner_radius : ℝ
  outer_radius : ℝ

/-- Represents a spherical ball in 3D space --/
structure Ball where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- The torus described in the problem --/
def problem_torus : Torus :=
  { center := (4, 0, 1)
    radius := 1
    inner_radius := 3
    outer_radius := 5 }

/-- 
  Given a torus sitting on the xy-plane, returns the radius of the largest
  spherical ball that can be placed on top of the center of the torus and
  still touch the horizontal plane
--/
def largest_ball_radius (t : Torus) : ℝ :=
  sorry

theorem largest_ball_radius_is_four :
  largest_ball_radius problem_torus = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_ball_radius_is_four_l2835_283534


namespace NUMINAMATH_CALUDE_square_sum_of_xy_l2835_283548

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 83)
  (h2 : x^2 * y + x * y^2 = 1056) : 
  x^2 + y^2 = 458 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_xy_l2835_283548


namespace NUMINAMATH_CALUDE_daffodil_cost_is_65_cents_l2835_283506

/-- Represents the cost of bulbs and garden space --/
structure BulbGarden where
  totalSpace : ℕ
  crocusCost : ℚ
  totalBudget : ℚ
  crocusCount : ℕ

/-- Calculates the cost of each daffodil bulb --/
def daffodilCost (g : BulbGarden) : ℚ :=
  let crocusTotalCost := g.crocusCost * g.crocusCount
  let remainingBudget := g.totalBudget - crocusTotalCost
  let daffodilCount := g.totalSpace - g.crocusCount
  remainingBudget / daffodilCount

/-- Theorem stating the cost of each daffodil bulb --/
theorem daffodil_cost_is_65_cents (g : BulbGarden)
  (h1 : g.totalSpace = 55)
  (h2 : g.crocusCost = 35/100)
  (h3 : g.totalBudget = 2915/100)
  (h4 : g.crocusCount = 22) :
  daffodilCost g = 65/100 := by
  sorry

#eval daffodilCost { totalSpace := 55, crocusCost := 35/100, totalBudget := 2915/100, crocusCount := 22 }

end NUMINAMATH_CALUDE_daffodil_cost_is_65_cents_l2835_283506


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2835_283523

/-- An arithmetic sequence with given first two terms -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = -1 ∧ a 2 = 2 ∧ ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h : arithmetic_sequence a) :
  a 4 + a 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2835_283523


namespace NUMINAMATH_CALUDE_valid_selling_price_l2835_283505

/-- Represents the business model for Oleg's water heater production --/
structure WaterHeaterBusiness where
  units_sold : ℕ
  variable_cost : ℕ
  fixed_cost : ℕ
  desired_profit : ℕ
  selling_price : ℕ

/-- Calculates the total revenue given the number of units sold and the selling price --/
def total_revenue (b : WaterHeaterBusiness) : ℕ :=
  b.units_sold * b.selling_price

/-- Calculates the total cost given the number of units sold, variable cost, and fixed cost --/
def total_cost (b : WaterHeaterBusiness) : ℕ :=
  b.units_sold * b.variable_cost + b.fixed_cost

/-- Checks if the selling price satisfies the business requirements --/
def is_valid_price (b : WaterHeaterBusiness) : Prop :=
  total_revenue b ≥ total_cost b + b.desired_profit

/-- Theorem stating that the calculated selling price satisfies the business requirements --/
theorem valid_selling_price :
  let b : WaterHeaterBusiness := {
    units_sold := 5000,
    variable_cost := 800,
    fixed_cost := 1000000,
    desired_profit := 1500000,
    selling_price := 1300
  }
  is_valid_price b ∧ b.selling_price ≥ 0 :=
by sorry


end NUMINAMATH_CALUDE_valid_selling_price_l2835_283505


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l2835_283550

/-- Given an ellipse with equation x^2/23 + y^2/32 = 1, its focal length is 6. -/
theorem ellipse_focal_length : ∀ (x y : ℝ), x^2/23 + y^2/32 = 1 → ∃ (c : ℝ), c = 3 ∧ 2*c = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l2835_283550


namespace NUMINAMATH_CALUDE_total_berries_l2835_283553

/-- The number of berries each person has -/
structure Berries where
  stacy : ℕ
  steve : ℕ
  skylar : ℕ

/-- The conditions of the berry distribution -/
def berry_conditions (b : Berries) : Prop :=
  b.stacy = 4 * b.steve ∧ 
  b.steve = 2 * b.skylar ∧ 
  b.stacy = 800

/-- The theorem stating the total number of berries -/
theorem total_berries (b : Berries) (h : berry_conditions b) : 
  b.stacy + b.steve + b.skylar = 1100 := by
  sorry

end NUMINAMATH_CALUDE_total_berries_l2835_283553


namespace NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l2835_283519

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-3/10, -9/10)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := y = 3 * x

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := y + 3 = -7 * x

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_both_lines :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y := by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point := by sorry

end NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l2835_283519


namespace NUMINAMATH_CALUDE_dilation_rotation_composition_l2835_283522

def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

def rotation_90_ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -1; 1, 0]

def combined_transformation : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -4; 4, 0]

theorem dilation_rotation_composition :
  combined_transformation = rotation_90_ccw * dilation_matrix 4 := by
  sorry

end NUMINAMATH_CALUDE_dilation_rotation_composition_l2835_283522


namespace NUMINAMATH_CALUDE_min_value_of_ab_l2835_283527

theorem min_value_of_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h_seq : (1/4 * Real.log a) * (Real.log b) = (1/4)^2) : 
  (∀ x y : ℝ, x > 1 → y > 1 → (1/4 * Real.log x) * (Real.log y) = (1/4)^2 → a * b ≤ x * y) ∧ 
  a * b = Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_ab_l2835_283527


namespace NUMINAMATH_CALUDE_big_bottles_sold_percentage_l2835_283521

theorem big_bottles_sold_percentage
  (small_initial : Nat)
  (big_initial : Nat)
  (small_sold_percent : Rat)
  (total_remaining : Nat)
  (h1 : small_initial = 6000)
  (h2 : big_initial = 15000)
  (h3 : small_sold_percent = 12 / 100)
  (h4 : total_remaining = 18180)
  : (big_initial - (total_remaining - (small_initial - small_initial * small_sold_percent))) / big_initial = 14 / 100 := by
  sorry

end NUMINAMATH_CALUDE_big_bottles_sold_percentage_l2835_283521


namespace NUMINAMATH_CALUDE_circle_ratio_l2835_283598

theorem circle_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : π * b^2 - π * a^2 = 5 * (π * a^2)) : 
  a / b = 1 / Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l2835_283598


namespace NUMINAMATH_CALUDE_two_point_form_equation_l2835_283551

/-- Two-point form equation of a line passing through two points -/
theorem two_point_form_equation (x1 y1 x2 y2 : ℝ) :
  let A : ℝ × ℝ := (x1, y1)
  let B : ℝ × ℝ := (x2, y2)
  x1 = 5 ∧ y1 = 6 ∧ x2 = -1 ∧ y2 = 2 →
  ∀ (x y : ℝ), (y - y1) / (y2 - y1) = (x - x1) / (x2 - x1) ↔
  (y - 6) / (2 - 6) = (x - 5) / (-1 - 5) :=
by sorry

end NUMINAMATH_CALUDE_two_point_form_equation_l2835_283551


namespace NUMINAMATH_CALUDE_triangle_equilateral_l2835_283589

/-- A triangle with side lengths a, b, and c satisfying specific conditions is equilateral. -/
theorem triangle_equilateral (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^4 = b^4 + c^4 - b^2*c^2) (h5 : b^4 = a^4 + c^4 - a^2*c^2) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l2835_283589


namespace NUMINAMATH_CALUDE_chocolates_per_student_l2835_283518

theorem chocolates_per_student (n : ℕ) :
  (∀ (students : ℕ), students * n < 288 → students ≤ 9) ∧
  (∀ (students : ℕ), students * n > 300 → students ≥ 10) →
  n = 31 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_per_student_l2835_283518


namespace NUMINAMATH_CALUDE_horner_third_intermediate_value_l2835_283570

def horner_polynomial (a : List ℚ) (x : ℚ) : ℚ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def horner_intermediate (a : List ℚ) (x : ℚ) (n : ℕ) : ℚ :=
  (a.take (n + 1)).foldl (fun acc coeff => acc * x + coeff) 0

theorem horner_third_intermediate_value :
  let f (x : ℚ) := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64
  let coeffs := [1, -12, 60, -160, 240, -192, 64]
  let x := 2
  horner_intermediate coeffs x 3 = -80 := by sorry

end NUMINAMATH_CALUDE_horner_third_intermediate_value_l2835_283570


namespace NUMINAMATH_CALUDE_sequence_sum_l2835_283575

theorem sequence_sum (n x y : ℝ) : 
  (3 + 16 + 33 + (n + 1) + x + y) / 6 = 25 → n + x + y = 97 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l2835_283575


namespace NUMINAMATH_CALUDE_divisibility_property_l2835_283555

theorem divisibility_property (q : ℕ) (h1 : Nat.Prime q) (h2 : q % 2 = 1) :
  ∃ k : ℤ, (q + 1 : ℤ)^(q - 1) - 1 = k * q :=
sorry

end NUMINAMATH_CALUDE_divisibility_property_l2835_283555


namespace NUMINAMATH_CALUDE_student_count_l2835_283538

theorem student_count (rank_right rank_left : ℕ) 
  (h1 : rank_right = 13) 
  (h2 : rank_left = 8) : 
  rank_right + rank_left - 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2835_283538
