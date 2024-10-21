import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_gt_cos_range_l1300_130009

theorem sin_gt_cos_range (x : ℝ) : 
  x ∈ Set.Icc 0 (2 * Real.pi) → 
  (Real.sin x > Real.cos x ↔ x ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_gt_cos_range_l1300_130009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_inverse_sum_l1300_130099

/-- A plane intersecting the coordinate axes -/
structure IntersectingPlane where
  /-- Distance from the origin to the plane -/
  distance : ℝ
  /-- x-intercept of the plane -/
  a : ℝ
  /-- y-intercept of the plane -/
  b : ℝ
  /-- z-intercept of the plane (fixed at 5) -/
  c : ℝ
  distance_eq : distance = 2
  c_eq : c = 5

/-- The centroid of a triangle with vertices (0,0,0), (a,0,0), and (0,b,0) -/
noncomputable def centroid (plane : IntersectingPlane) : ℝ × ℝ × ℝ :=
  (plane.a / 3, plane.b / 3, plane.c / 3)

/-- The main theorem -/
theorem centroid_inverse_sum (plane : IntersectingPlane) :
  let (p, q, r) := centroid plane
  1 / p^2 + 1 / q^2 + 1 / r^2 = 369 / 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_inverse_sum_l1300_130099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_framing_for_specific_picture_l1300_130093

/-- Calculates the minimum number of linear feet of framing needed for a picture with given dimensions, enlargement factor, and border width. -/
def min_framing_feet (width : ℚ) (height : ℚ) (enlarge_factor : ℚ) (border_width : ℚ) : ℕ :=
  let enlarged_width := width * enlarge_factor
  let enlarged_height := height * enlarge_factor
  let framed_width := enlarged_width + 2 * border_width
  let framed_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (framed_width + framed_height)
  let perimeter_feet := perimeter_inches / 12
  Int.ceil perimeter_feet |>.toNat

/-- Theorem stating that for a 5-inch by 8-inch picture, doubled in size and with a 3-inch border, 
    the minimum framing needed is 7 feet. -/
theorem min_framing_for_specific_picture : 
  min_framing_feet 5 8 2 3 = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_framing_for_specific_picture_l1300_130093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l1300_130045

-- Define the space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the points
variable (A B C O : V)

-- Define the conditions
variable (m : ℝ)
variable (lambda : ℝ)

-- Collinearity of A, B, and C
variable (h_collinear : ∃ (t : ℝ), B - A = t • (C - A))

-- O is not on the line
variable (h_O_not_on_line : ¬∃ (s : ℝ), O - A = s • (C - A))

-- Vector equation
variable (h_vector_eq : m • (O - A) - 2 • (O - B) + (O - C) = 0)

-- BA = λAC
variable (h_BA_lambda_AC : B - A = lambda • (C - A))

-- Theorem statement
theorem lambda_value : lambda = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l1300_130045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_one_l1300_130082

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * (2:ℝ)^x - (2:ℝ)^(-x))

theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 1 := by
  intro h
  -- The proof goes here
  sorry

#check even_function_implies_a_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_one_l1300_130082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_arithmetic_sequence_l1300_130003

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

def is_arithmetic_progression (a b c : ℕ) : Prop :=
  b - a = c - b

theorem fibonacci_arithmetic_sequence (n : ℕ) :
  is_arithmetic_progression (fib n) (fib (n + 3)) (fib (n + 5)) ∧
  n + (n + 3) + (n + 5) = 2500 →
  n = 831 := by
  sorry

#check fibonacci_arithmetic_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_arithmetic_sequence_l1300_130003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2009_value_l1300_130080

def my_sequence (x : ℕ → ℚ) : Prop :=
  x 1 = 2 ∧ ∀ n : ℕ, n ≥ 1 → (n + 1) * x (n + 1) = x n + n

theorem sequence_2009_value (x : ℕ → ℚ) (h : my_sequence x) : 
  x 2009 = (Nat.factorial 2010 + 1) / Nat.factorial 2010 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2009_value_l1300_130080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequence_exists_l1300_130070

def is_valid_sequence (s : List Nat) : Prop :=
  s.length = 10 ∧
  s.toFinset = Finset.range 10 ∧
  ∀ i : Fin s.length, i.val > 0 → (s.take i.val).sum % s[i] = 0

theorem valid_sequence_exists : ∃ s : List Nat, is_valid_sequence s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequence_exists_l1300_130070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1300_130041

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the xy-plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (l : Line) : ℝ :=
  abs (l.a * p.1 + l.b * p.2 + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  distance_point_to_line c.center l = c.radius

theorem tangent_line_equation (c : Circle) (l2 : Line) :
  c = Circle.mk (0, -1) 1 →
  l2 = Line.mk 3 4 (-6) →
  ∀ l1 : Line,
    parallel l1 l2 →
    is_tangent l1 c →
    (l1 = Line.mk 3 4 (-1) ∨ l1 = Line.mk 3 4 9) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1300_130041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l1300_130084

theorem arithmetic_sequence_middle_term (a₁ a₃ y : ℝ) : 
  a₁ = 3^2 → a₃ = 3^4 → (a₃ - y = y - a₁) → y = 45 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l1300_130084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_is_prime_l1300_130065

def first_ten_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun m => m <= 1 || n % (m + 1) ≠ 0)

def valid_pair (p q : Nat) : Bool :=
  p ∈ first_ten_primes && q ∈ first_ten_primes && p ≠ q && is_prime (p + q)

def count_valid_pairs : Nat :=
  (first_ten_primes.map (fun p => 
    (first_ten_primes.filter (fun q => valid_pair p q)).length
  )).sum / 2

theorem probability_sum_is_prime :
  count_valid_pairs / (first_ten_primes.length.choose 2) = 1 / 9 := by
  sorry

#eval count_valid_pairs
#eval first_ten_primes.length.choose 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_is_prime_l1300_130065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1300_130001

/-- Calculates the principal given simple interest, rate, and time -/
noncomputable def calculate_principal (simple_interest : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  simple_interest * 100 / (rate * time)

/-- Proves that the given simple interest, rate, and time result in the specified principal -/
theorem principal_calculation :
  let simple_interest : ℝ := 6016.75
  let rate : ℝ := 8
  let time : ℝ := 5
  let calculated_principal := calculate_principal simple_interest rate time
  calculated_principal = 15041.875 := by
  -- Unfold the definition of calculate_principal
  unfold calculate_principal
  -- Perform the calculation
  norm_num
  -- Complete the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l1300_130001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1300_130057

/-- A cubic function parameterized by m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (4*m - 1) * x^2 + (15*m^2 - 2*m - 7) * x + 2

/-- The derivative of f with respect to x -/
noncomputable def f' (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*(4*m - 1)*x + (15*m^2 - 2*m - 7)

/-- f is increasing on ℝ -/
def is_increasing (m : ℝ) : Prop := ∀ x, f' m x > 0

/-- The main theorem: if f is increasing on ℝ, then m is in [2, 4] -/
theorem m_range (m : ℝ) (h : is_increasing m) : m ∈ Set.Icc 2 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1300_130057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bert_next_to_dana_carl_not_between_bert_and_dana_dana_in_four_carl_in_one_l1300_130053

-- Define the type for seats
inductive Seat : Type where
  | one : Seat
  | two : Seat
  | three : Seat
  | four : Seat

-- Define the type for people
inductive Person : Type where
  | Abby : Person
  | Bert : Person
  | Carl : Person
  | Dana : Person

-- Define a function to represent the seating arrangement
def seating : Seat → Person := sorry

-- Define the conditions as theorems
theorem bert_next_to_dana : 
  (seating Seat.three = Person.Bert ∧ seating Seat.four = Person.Dana) ∨
  (seating Seat.four = Person.Dana ∧ seating Seat.three = Person.Bert) := sorry

theorem carl_not_between_bert_and_dana :
  ¬(seating Seat.two = Person.Carl ∧ seating Seat.three = Person.Bert ∧ seating Seat.four = Person.Dana) := sorry

theorem dana_in_four : seating Seat.four = Person.Dana := sorry

-- The theorem to prove
theorem carl_in_one : seating Seat.one = Person.Carl := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bert_next_to_dana_carl_not_between_bert_and_dana_dana_in_four_carl_in_one_l1300_130053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1300_130032

noncomputable def f (x : ℝ) := x^2
noncomputable def g (x m : ℝ) := (1/2)^x - m

theorem min_m_value (h : ∀ x₁ ∈ Set.Icc (-1) 3, ∃ x₂ ∈ Set.Icc 0 2, f x₁ ≥ g x₂ m) : 
  m ≥ 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1300_130032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l1300_130048

/-- A hyperbola passing through (1,1) with b/a = √2 has the standard equation x²/(1/2) - y² = 1 or y²/(1/2) - x² = 1 -/
theorem hyperbola_standard_equation (h : Real → Real → Prop) 
  (passes_through : h 1 1)
  (ratio : ∃ (a b : Real), b / a = Real.sqrt 2 ∧ 
    (∀ x y, h x y ↔ (x^2 / a^2 - y^2 / b^2 = 1 ∨ y^2 / a^2 - x^2 / b^2 = 1))) :
  (∃ x y, h x y ↔ x^2 / (1/2) - y^2 = 1) ∨
  (∃ x y, h x y ↔ y^2 / (1/2) - x^2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l1300_130048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exceeding_1000_l1300_130069

noncomputable def geometric_progression (n : ℕ) : ℝ := (5 : ℝ) ^ ((n : ℝ) / 7)

def product_exceeds_1000 (n : ℕ) : Prop :=
  (5 : ℝ) ^ ((n * (n + 1) : ℝ) / (2 * 7)) > 1000

theorem smallest_n_exceeding_1000 :
  ∀ k : ℕ, k < 7 → ¬(product_exceeds_1000 k) ∧ product_exceeds_1000 7 :=
by
  sorry

#check smallest_n_exceeding_1000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_exceeding_1000_l1300_130069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l1300_130010

/-- Line represented by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to check if a point is below a line -/
def is_below (p : Point) (l : Line) : Prop :=
  p.y ≤ l.m * p.x + l.b

/-- Function to check if a point is between two lines -/
def is_between (p : Point) (l1 l2 : Line) : Prop :=
  is_below p l1 ∧ ¬is_below p l2

/-- Function to calculate the area of a triangle formed by a line and the axes -/
noncomputable def triangle_area (l : Line) : ℝ :=
  (l.b * l.b) / (2 * (-l.m))

/-- The main theorem -/
theorem probability_between_lines (p q : Line)
  (hp : p.m = -2 ∧ p.b = 8)
  (hq : q.m = -3 ∧ q.b = 8) :
  (triangle_area p - triangle_area q) / triangle_area p = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l1300_130010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1300_130094

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

def geometric_sequence (b : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, q > 0 ∧ ∀ n : ℕ, b (n + 1) = q * b n

def S (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i ↦ a (i + 1))

def C (a b : ℕ → ℚ) (n : ℕ) : ℚ :=
  if n % 2 = 1 then 2 else -2 * a n / b n

def T (a b : ℕ → ℚ) (n : ℕ) : ℚ :=
  (Finset.range (2 * n)).sum (λ i ↦ C a b (i + 1))

theorem sequence_properties
  (a b : ℕ → ℚ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h1 : b 1 = -2 * a 1)
  (h2 : b 1 = 2)
  (h3 : a 3 + b 2 = -1)
  (h4 : S a 3 + 2 * b 3 = 7) :
  (∀ n : ℕ, a n = -2 * n + 1) ∧
  (∀ n : ℕ, b n = 2^n) ∧
  (∀ n : ℕ, T a b n = 26/9 - (12*n + 13) / (9 * 2^(2*n - 1)) + 2*n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1300_130094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_characterization_l1300_130038

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 + 2*m) * x^(m^2 + m - 1)

-- Define what it means for f to be a power function
def is_power_function (m : ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f m x = a * x^b

-- Define what it means for f to be a direct proportionality function
def is_direct_proportional (m : ℝ) : Prop :=
  ∃ (k : ℝ), ∀ x, f m x = k * x

-- Define what it means for f to be an inverse proportionality function
def is_inverse_proportional (m : ℝ) : Prop :=
  ∃ (k : ℝ), ∀ x, f m x = k / x

-- The main theorem
theorem f_characterization :
  (∀ m : ℝ, is_power_function m ↔ (m = -1 - Real.sqrt 2 ∨ m = -1 + Real.sqrt 2)) ∧
  (∀ m : ℝ, is_direct_proportional m ↔ m = 1) ∧
  (∀ m : ℝ, is_inverse_proportional m ↔ m = -1) := by
  sorry

#check f_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_characterization_l1300_130038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_cost_is_three_l1300_130033

/-- The cost of a dozen apples -/
def apple_dozen_cost : ℝ := 0

/-- The cost of a bunch of bananas -/
def banana_bunch_cost : ℝ := 0

/-- Tony's purchase: 2 dozen apples and some bananas -/
def tony_purchase : ℝ := 2 * apple_dozen_cost + banana_bunch_cost

/-- Arnold's purchase: 1 dozen apples and the same number of bananas as Tony -/
def arnold_purchase : ℝ := apple_dozen_cost + banana_bunch_cost

theorem banana_cost_is_three :
  tony_purchase = 7 ∧ arnold_purchase = 5 → banana_bunch_cost = 3 := by
  intro h
  sorry

#check banana_cost_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_cost_is_three_l1300_130033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l1300_130044

theorem absolute_value_inequality (x : ℝ) : 
  (abs (abs (x - 2) - 1) ≤ 1) ↔ (0 ≤ x ∧ x ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l1300_130044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_breadth_to_length_ratio_l1300_130050

/-- A rectangular hall with given dimensions -/
structure RectangularHall where
  length : ℝ
  breadth : ℝ
  area : ℝ

/-- The properties of our specific hall -/
noncomputable def our_hall : RectangularHall where
  length := 60
  breadth := 2400 / 60
  area := 2400

/-- The theorem stating the ratio of breadth to length -/
theorem breadth_to_length_ratio :
  our_hall.breadth / our_hall.length = 2 / 3 := by
  -- Unfold the definition of our_hall
  unfold our_hall
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_breadth_to_length_ratio_l1300_130050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_divisible_by_primes_and_nine_l1300_130068

def fourth_prime : ℕ := 7
def fifth_prime : ℕ := 11
def sixth_prime : ℕ := 13

theorem least_divisible_by_primes_and_nine :
  ∃ (n : ℕ), n = 9009 ∧
    n > 0 ∧
    n % fourth_prime = 0 ∧
    n % fifth_prime = 0 ∧
    n % sixth_prime = 0 ∧
    n % 9 = 0 ∧
    ∀ (m : ℕ), m > 0 ∧
      m % fourth_prime = 0 ∧
      m % fifth_prime = 0 ∧
      m % sixth_prime = 0 ∧
      m % 9 = 0 →
      n ≤ m :=
by
  use 9009
  constructor
  · rfl
  constructor
  · norm_num
  repeat (constructor; norm_num)
  intro m hm
  sorry  -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_divisible_by_primes_and_nine_l1300_130068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_quotient_partition_l1300_130061

def is_valid_partition (A B : Finset Nat) : Prop :=
  A ∪ B = Finset.range 9 \ {0} ∧ A ∩ B = ∅ ∧ A.card > 0 ∧ B.card > 0

def product (s : Finset Nat) : Nat :=
  s.prod id

theorem min_quotient_partition :
  ∀ A B : Finset Nat,
    is_valid_partition A B →
    (product A) % (product B) = 0 →
    70 ≤ (product A) / (product B) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_quotient_partition_l1300_130061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_ratio_l1300_130059

theorem root_sum_ratio (k₁ k₂ : ℝ) : 
  (∃ a b : ℝ, (∃ k : ℝ, k*(3*a^2 - 5*a) + 2*a + 7 = 0 ∧ k*(3*b^2 - 5*b) + 2*b + 7 = 0) ∧
   a/b + b/a = 3/4 ∧
   k₁*(3*a^2 - 5*a) + 2*a + 7 = 0 ∧ k₁*(3*b^2 - 5*b) + 2*b + 7 = 0 ∧
   k₂*(3*a^2 - 5*a) + 2*a + 7 = 0 ∧ k₂*(3*b^2 - 5*b) + 2*b + 7 = 0) →
  k₁/k₂ + k₂/k₁ = 2.6923 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_ratio_l1300_130059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_microbial_counting_statements_l1300_130098

-- Define the set of all statements
inductive MicrobialCountingStatements
  | statement1
  | statement2
  | statement3
  | statement4

-- Define the property of being a correct statement
def IsCorrectStatement : MicrobialCountingStatements → Prop
  | MicrobialCountingStatements.statement1 => True  -- Statement ① is correct
  | MicrobialCountingStatements.statement2 => False -- Statement ② is incorrect
  | MicrobialCountingStatements.statement3 => True  -- Statement ③ is correct
  | MicrobialCountingStatements.statement4 => False -- Statement ④ is incorrect

-- Define the set of correct statements
def CorrectStatements : Set MicrobialCountingStatements :=
  {s | IsCorrectStatement s}

-- Theorem to prove
theorem correct_microbial_counting_statements :
  CorrectStatements = {MicrobialCountingStatements.statement1, MicrobialCountingStatements.statement3} := by
  sorry

#check correct_microbial_counting_statements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_microbial_counting_statements_l1300_130098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_division_exists_l1300_130040

/-- Represents a 6x6 grid --/
structure Grid where
  cells : Fin 6 → Fin 6 → Option Bool

/-- Represents a position on the grid --/
structure Position where
  row : Fin 6
  col : Fin 6

/-- Represents a 3x3 part of the grid --/
structure GridPart where
  topLeft : Position

def Grid.hasCircle (g : Grid) (p : GridPart) : Prop :=
  ∃ (i j : Fin 3), g.cells (i + p.topLeft.row) (j + p.topLeft.col) = some true

def Grid.hasStar (g : Grid) (p : GridPart) : Prop :=
  ∃ (i j : Fin 3), g.cells (i + p.topLeft.row) (j + p.topLeft.col) = some false

def Grid.validPart (g : Grid) (p : GridPart) : Prop :=
  g.hasCircle p ∧ g.hasStar p

def Grid.validDivision (g : Grid) (p1 p2 p3 p4 : GridPart) : Prop :=
  g.validPart p1 ∧ g.validPart p2 ∧ g.validPart p3 ∧ g.validPart p4 ∧
  p1.topLeft = ⟨0, 0⟩ ∧ p2.topLeft = ⟨0, 3⟩ ∧ p3.topLeft = ⟨3, 0⟩ ∧ p4.topLeft = ⟨3, 3⟩

theorem grid_division_exists :
  ∃ (g : Grid) (p1 p2 p3 p4 : GridPart), g.validDivision p1 p2 p3 p4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_division_exists_l1300_130040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_values_l1300_130037

theorem sum_of_x_values : 
  ∃ (x₁ x₂ : ℝ), |x₁ - 25| = 50 ∧ |x₂ - 25| = 50 ∧ x₁ + x₂ = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_values_l1300_130037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_monochromatic_triangle_l1300_130060

/-- A complete graph K6 where each edge is independently colored red or blue -/
def K6ColoredGraph := Fin 6 → Fin 6 → Bool

/-- The probability of an edge being red (or blue) -/
def edgeProbability : ℚ := 1/2

/-- The probability of a triangle being monochromatic -/
def monochromaticTriangleProbability : ℚ := 1/4

/-- The number of triangles in K6 -/
def numTriangles : ℕ := 20

/-- The theorem to prove -/
theorem probability_of_monochromatic_triangle (g : K6ColoredGraph) :
  (255 : ℚ)/256 = 1 - (1 - monochromaticTriangleProbability) ^ numTriangles :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_monochromatic_triangle_l1300_130060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_parallel_to_l_l1300_130083

structure GeometricSpace where
  Point : Type
  Line : Type
  Plane : Type
  perpendicular_line_plane : Line → Plane → Prop
  perpendicular_line_line : Line → Line → Prop
  skew : Line → Line → Prop
  intersect : Plane → Plane → Prop
  parallel : Line → Line → Prop
  not_subset : Line → Plane → Prop

theorem intersection_parallel_to_l 
  (S : GeometricSpace) 
  (m n l : S.Line) 
  (α β : S.Plane) :
  S.skew m n →
  S.perpendicular_line_plane m α →
  S.perpendicular_line_plane n β →
  S.perpendicular_line_line l m →
  S.perpendicular_line_line l n →
  S.not_subset l α →
  S.not_subset l β →
  ∃ (i : S.Line), S.intersect α β ∧ S.parallel i l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_parallel_to_l_l1300_130083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_concerts_for_musicians_l1300_130011

/-- Represents a concert configuration --/
structure Concert where
  performers : Finset Nat
  listeners : Finset Nat

/-- The problem setup --/
structure MusicianProblem where
  total_musicians : Nat
  concerts : List Concert

/-- Each musician listens to all others --/
def all_musicians_listen (p : MusicianProblem) : Prop :=
  ∀ m, m < p.total_musicians → 
    ∀ other, other < p.total_musicians ∧ other ≠ m →
      ∃ c ∈ p.concerts, m ∈ c.listeners ∧ other ∈ c.performers

/-- In each concert, some perform and others listen --/
def valid_concerts (p : MusicianProblem) : Prop :=
  ∀ c ∈ p.concerts, 
    c.performers.card + c.listeners.card = p.total_musicians ∧
    c.performers.card > 0 ∧ c.listeners.card > 0

/-- The main theorem --/
theorem min_concerts_for_musicians :
  ∃ p : MusicianProblem, 
    all_musicians_listen p ∧ 
    valid_concerts p ∧ 
    p.concerts.length = 4 ∧
    (∀ p' : MusicianProblem, all_musicians_listen p' → valid_concerts p' → 
      p'.concerts.length ≥ 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_concerts_for_musicians_l1300_130011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_odd_function_l1300_130043

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * (Real.cos x)^2 - 2 * Real.sin x * Real.cos x - Real.sqrt 3

-- Define the translated function
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := f (x + t)

-- State the theorem
theorem min_translation_for_odd_function :
  ∃ (t : ℝ), t > 0 ∧
  (∀ (x : ℝ), g t x = -g t (-x)) ∧
  (∀ (t' : ℝ), t' > 0 ∧ (∀ (x : ℝ), g t' x = -g t' (-x)) → t ≤ t') ∧
  t = Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_odd_function_l1300_130043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1300_130081

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin (2 * x) + b * Real.cos (2 * x)

theorem monotonic_increase_interval 
  (a b : ℝ) 
  (h1 : a * b ≠ 0)
  (h2 : ∀ x : ℝ, f a b x ≤ |f a b (π/6)|)
  (h3 : f a b (π/2) > 0) :
  ∃ (k : ℤ), StrictMonoOn (f a b) (Set.Icc (k * π + π/6) (k * π + 2*π/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l1300_130081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_divisibility_l1300_130021

def is_nine_digit_number (n : ℕ) : Prop :=
  n ≥ 100000000 ∧ n < 1000000000

def replace_digit (n : ℕ) (i : ℕ) (new_digit : ℕ) : ℕ :=
  let digits := List.range 9 |> List.map (λ j => (n / (10 ^ j)) % 10)
  let new_digits := digits.set i new_digit
  new_digits.foldl (λ acc d => acc * 10 + d) 0

theorem nine_digit_divisibility (d e f : ℕ) (hd : is_nine_digit_number d)
  (he : is_nine_digit_number e) (hf : is_nine_digit_number f) :
  (∀ i : Fin 9, (replace_digit d i.val ((e / (10 ^ i.val)) % 10)) % 7 = 0) →
  (∀ i : Fin 9, (replace_digit e i.val ((f / (10 ^ i.val)) % 10)) % 7 = 0) →
  ∀ i : Fin 9, ((d / (10 ^ i.val)) % 10 - (f / (10 ^ i.val)) % 10) % 7 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_digit_divisibility_l1300_130021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_f_two_zeros_l1300_130000

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (2 * x) + (a - 2) * Real.exp x - x

theorem f_minimum_value :
  ∃ x₀ : ℝ, IsMinOn (f 2) Set.univ x₀ ∧ f 2 x₀ = 1/2 + Real.log 2 := by sorry

theorem f_two_zeros (a : ℝ) :
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ 0 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_f_two_zeros_l1300_130000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_optimism_indicator_l1300_130046

theorem survey_optimism_indicator 
  (a b c : ℤ) 
  (total : a + b + c = 100) 
  (m_def : a + b / 2 = 40) : 
  a - c = -20 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_optimism_indicator_l1300_130046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unit_segments_in_plane_l1300_130055

/-- Represents a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A configuration of 4 points in a plane -/
structure Configuration where
  A1 : Point
  A2 : Point
  A3 : Point
  A4 : Point

/-- Predicate to check if all distances between points are at least 1 -/
def validConfiguration (c : Configuration) : Prop :=
  distance c.A1 c.A2 ≥ 1 ∧
  distance c.A1 c.A3 ≥ 1 ∧
  distance c.A1 c.A4 ≥ 1 ∧
  distance c.A2 c.A3 ≥ 1 ∧
  distance c.A2 c.A4 ≥ 1 ∧
  distance c.A3 c.A4 ≥ 1

/-- Count the number of unit-length segments in a configuration -/
noncomputable def countUnitSegments (c : Configuration) : ℕ :=
  (if distance c.A1 c.A2 = 1 then 1 else 0) +
  (if distance c.A1 c.A3 = 1 then 1 else 0) +
  (if distance c.A1 c.A4 = 1 then 1 else 0) +
  (if distance c.A2 c.A3 = 1 then 1 else 0) +
  (if distance c.A2 c.A4 = 1 then 1 else 0) +
  (if distance c.A3 c.A4 = 1 then 1 else 0)

/-- The main theorem to be proven -/
theorem max_unit_segments_in_plane :
  ∀ c : Configuration, validConfiguration c → countUnitSegments c ≤ 5 ∧
  ∃ c' : Configuration, validConfiguration c' ∧ countUnitSegments c' = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unit_segments_in_plane_l1300_130055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_salary_proof_l1300_130029

/-- The number of players on a team -/
def num_players : ℕ := 18

/-- The minimum salary for each player -/
def min_salary : ℕ := 12000

/-- The maximum total salary for all players on a team -/
def max_total_salary : ℕ := 480000

/-- The maximum possible salary for a single player -/
def max_single_salary : ℕ := 276000

theorem max_salary_proof :
  ∃ (salaries : Fin num_players → ℕ),
    (∀ i, salaries i ≥ min_salary) ∧
    (Finset.sum Finset.univ salaries ≤ max_total_salary) ∧
    (∀ i, salaries i ≤ max_single_salary) ∧
    (∃ j, salaries j = max_single_salary) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_salary_proof_l1300_130029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_TUW_l1300_130030

noncomputable section

-- Define the rectangle PQRS
def Rectangle (P Q R S : ℝ × ℝ) : Prop :=
  P.1 = Q.1 ∧ P.2 = S.2 ∧ Q.2 = R.2 ∧ S.1 = R.1

-- Define midpoint
def Midpoint (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define the area of a triangle given its vertices
def TriangleArea (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_TUW :
  ∀ (P Q R S T U V W : ℝ × ℝ),
  Rectangle P Q R S →
  Q.1 - P.1 = 2 →
  R.2 - Q.2 = 3 →
  Midpoint Q R T →
  Midpoint R S U →
  Midpoint S P V →
  Midpoint V T W →
  TriangleArea T U W = 0.75 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_TUW_l1300_130030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l1300_130067

/-- Calculates the minimum discount percentage that can be offered given the purchase price, 
    original selling price, and minimum profit margin. -/
noncomputable def min_discount (purchase_price selling_price : ℝ) (min_profit_margin : ℝ) : ℝ :=
  (1 - (purchase_price * (1 + min_profit_margin)) / selling_price) * 100

/-- Theorem stating that for the given conditions, the minimum discount is 70% -/
theorem discount_calculation (purchase_price selling_price min_profit_margin : ℝ) 
  (h1 : purchase_price = 800)
  (h2 : selling_price = 1200)
  (h3 : min_profit_margin = 0.05) :
  min_discount purchase_price selling_price min_profit_margin = 70 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval min_discount 800 1200 0.05

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_calculation_l1300_130067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l1300_130085

/-- The area of the region bounded by two lines and the y-axis --/
noncomputable def bounded_area (line1 line2 : ℝ → ℝ) : ℝ :=
  let x_intersect := (line2 0 - line1 0) / ((line1 1 - line1 0) - (line2 1 - line2 0))
  let y_intersect := line1 x_intersect
  let base := |line2 0 - line1 0|
  let height := x_intersect
  (1 / 2) * base * height

/-- First line equation --/
def line1 (x : ℝ) : ℝ := 3 * x - 6

/-- Second line equation --/
def line2 (x : ℝ) : ℝ := -2 * x + 14

theorem area_of_bounded_region :
  bounded_area line1 line2 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l1300_130085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1300_130018

noncomputable section

/-- Parabola with equation y² = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Tangent line with equation x - y + 1 = 0 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Distance from a point to the tangent line -/
noncomputable def distToTangent (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 + 1| / Real.sqrt 2

/-- Sum of distances from two points to the tangent line -/
noncomputable def sumOfDistances (p q : ℝ × ℝ) : ℝ :=
  distToTangent p + distToTangent q

theorem min_sum_distances :
  ∃ (minDist : ℝ),
    minDist = 3 * Real.sqrt 2 / 2 ∧
    ∀ (p q : ℝ × ℝ),
      p ∈ Parabola → q ∈ Parabola →
      ∃ (m : Set (ℝ × ℝ)), Focus ∈ m ∧ p ∈ m ∧ q ∈ m →
        minDist ≤ sumOfDistances p q :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_l1300_130018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l1300_130028

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The distance from the center to a focus of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- Theorem: If the maximum distance from a point on the ellipse to a focus is 5
    and the minimum distance is 1, then the length of the minor axis is 2√5 -/
theorem ellipse_minor_axis_length
  (e : Ellipse)
  (h_max : e.a + focal_distance e = 5)
  (h_min : e.a - focal_distance e = 1) :
  2 * e.b = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minor_axis_length_l1300_130028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_fixed_line_l1300_130097

noncomputable section

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Foci of the ellipse -/
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

/-- Point that the ellipse passes through -/
def P : ℝ × ℝ := (1, -3/2)

/-- Point D -/
def D : ℝ × ℝ := (4, 0)

/-- Left and right vertices of the ellipse -/
def A₁ : ℝ × ℝ := (-2, 0)
def A₂ : ℝ × ℝ := (2, 0)

/-- Definition of a line through D not parallel to x-axis -/
def line_through_D (m : ℝ) (x : ℝ) : ℝ := m * (x - D.1)

/-- Intersection points of the line with the ellipse -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ p.2 = line_through_D m p.1}

/-- Theorem: The intersection of A₁E and A₂F always lies on x = 1 -/
theorem intersection_on_fixed_line (m : ℝ) (E F : ℝ × ℝ) 
  (hE : E ∈ intersection_points m) (hF : F ∈ intersection_points m) 
  (hm : m ≠ 0) : 
  ∃ (y : ℝ), 
    (A₁.1 - E.1) * (1 - A₁.1) * (F.2 - A₂.2) = 
    (A₂.1 - F.1) * (1 - A₂.1) * (E.2 - A₁.2) ∧
    (1, y) = (1, (E.2 * (1 - A₁.1)) / (E.1 - A₁.1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_fixed_line_l1300_130097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_satisfies_equation_l1300_130089

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Theorem: If a point P(x,y,z) is equidistant from A(1,0,1) and B(2,3,-1),
    then 2x + 3y - 4z = 6 -/
theorem equidistant_point_satisfies_equation (p : Point3D) :
  let a : Point3D := ⟨1, 0, 1⟩
  let b : Point3D := ⟨2, 3, -1⟩
  distance p a = distance p b →
  2 * p.x + 3 * p.y - 4 * p.z = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_satisfies_equation_l1300_130089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_sale_loss_percentage_l1300_130077

/-- Calculates the loss percentage when an article is sold at a fraction of its original price -/
theorem article_sale_loss_percentage 
  (original_price : ℝ) 
  (cost_price : ℝ) 
  (original_gain_percent : ℝ) 
  (sale_fraction : ℝ) : 
  original_price = cost_price * (1 + original_gain_percent / 100) →
  sale_fraction = 2 / 3 →
  original_gain_percent = 35 →
  (cost_price - sale_fraction * original_price) / cost_price * 100 = 10 := by
  sorry

#check article_sale_loss_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_sale_loss_percentage_l1300_130077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_finds_train_prob_l1300_130008

/-- Represents time in minutes after 1:00 PM -/
def Time := Fin 60

/-- The train's arrival time -/
noncomputable def trainArrival : Time := sorry

/-- Susan's arrival time -/
noncomputable def susanArrival : Time := sorry

/-- Predicate that determines if Susan finds the train at the station -/
def susanFindsTrain (t s : Time) : Prop :=
  (s.val ≥ t.val) ∧ (s.val < t.val + 30)

/-- The probability space for this problem -/
def Ω : Type := Time × Time

/-- The probability measure for this problem -/
noncomputable def P : Set Ω → ℝ := sorry

theorem susan_finds_train_prob :
  P {ω : Ω | susanFindsTrain ω.1 ω.2} = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_susan_finds_train_prob_l1300_130008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_perimeter_l1300_130016

/-- Given a sector with radius 1 and area π/3, its perimeter is (2π/3) + 2 -/
theorem sector_perimeter (sector_area : Real) (h : sector_area = π / 3) : 
  (2 * π / 3) + 2 = 2 * 1 + 2 * sector_area / 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_perimeter_l1300_130016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l1300_130087

theorem smallest_a_value (a b : ℕ) (h : 96 * a ^ 2 = b ^ 3) : 
  ∀ x : ℕ, x > 0 ∧ (∃ y : ℕ, y > 0 ∧ 96 * x ^ 2 = y ^ 3) → a ≤ x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l1300_130087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_supporting_both_policies_l1300_130052

/-- Given a set of students participating in a poll about two policies, 
    this theorem proves the number of students supporting both policies. -/
theorem students_supporting_both_policies 
  (U : Finset Nat) -- The set of all students
  (A : Finset Nat) -- The set of students supporting policy A
  (B : Finset Nat) -- The set of students supporting policy B
  (h1 : Finset.card U = 220) -- Total number of students
  (h2 : Finset.card A = 165) -- Number of students supporting policy A
  (h3 : Finset.card B = 140) -- Number of students supporting policy B
  (h4 : Finset.card (U \ (A ∪ B)) = 40) -- Number of students opposing both policies
  : Finset.card (A ∩ B) = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_supporting_both_policies_l1300_130052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_triple_l1300_130006

-- Define the conditions for the ordered triple (a, b, c)
def ValidTriple (a b c : ℕ) : Prop :=
  a ≥ 2 ∧ b ≥ 1 ∧ c ≥ 0 ∧ (Real.logb a b = c^3) ∧ a + b + c = 100

-- Theorem statement
theorem unique_valid_triple : ∃! (t : ℕ × ℕ × ℕ), ValidTriple t.1 t.2.1 t.2.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_triple_l1300_130006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baseball_card_value_decrease_l1300_130090

theorem baseball_card_value_decrease (x : ℝ) : 
  (∃ (initial_value : ℝ), initial_value > 0 ∧
    let first_year_value := initial_value * (1 - x / 100);
    let second_year_value := first_year_value * (1 - 10 / 100);
    second_year_value = initial_value * (1 - 46 / 100)) →
  x = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_baseball_card_value_decrease_l1300_130090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_legendre_formula_correct_l1300_130024

noncomputable def legendre_formula (p n : ℕ) : ℕ :=
  Finset.sum (Finset.range (Nat.log p n + 1)) (fun i => n / p ^ i)

theorem legendre_formula_correct (p n : ℕ) (hp : Nat.Prime p) :
  (legendre_formula p n) = Finset.sup (Finset.range (n + 1)) (fun k => if p ^ k ∣ Nat.factorial n then k else 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_legendre_formula_correct_l1300_130024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_30_degrees_l1300_130058

/-- The area of a figure formed by rotating a semicircle around one of its endpoints -/
noncomputable def rotated_semicircle_area (R : ℝ) (α : ℝ) : ℝ :=
  (1 / 2) * (2 * R)^2 * α

/-- Theorem: The area of the figure formed by rotating a semicircle of radius R
    around one of its endpoints by an angle of 30° (π/6 radians) is equal to (π * R^2) / 3 -/
theorem rotated_semicircle_area_30_degrees (R : ℝ) (h : R > 0) :
  rotated_semicircle_area R (Real.pi / 6) = (Real.pi * R^2) / 3 := by
  sorry

#check rotated_semicircle_area_30_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_30_degrees_l1300_130058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1300_130056

def expression (e f g h : ℕ) : ℕ := e * f^g - h

theorem max_expression_value :
  ∃ (e f g h : ℕ),
    e ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    f ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    g ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    h ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ f ≠ g ∧ f ≠ h ∧ g ≠ h ∧
    ∀ (a b c d : ℕ),
      a ∈ ({1, 2, 3, 4} : Set ℕ) →
      b ∈ ({1, 2, 3, 4} : Set ℕ) →
      c ∈ ({1, 2, 3, 4} : Set ℕ) →
      d ∈ ({1, 2, 3, 4} : Set ℕ) →
      a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
      expression e f g h ≥ expression a b c d ∧
      expression e f g h = 127 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1300_130056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l1300_130066

theorem negation_of_sin_inequality :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x₀ : ℝ, Real.sin x₀ > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l1300_130066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1300_130039

def sequenceA (n : ℕ) : ℚ :=
  (2 * n - 1) / n

theorem sequence_formula (a : ℕ → ℚ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n ≥ 1 → (a n - 3) * a (n + 1) - a n + 4 = 0) : 
  ∀ n : ℕ, n ≥ 1 → a n = sequenceA n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1300_130039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_divisible_power_minus_base_l1300_130004

theorem non_divisible_power_minus_base (m : ℕ) (h : m ≠ 1) :
  ∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℕ, ¬(p ∣ (n^m - m)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_divisible_power_minus_base_l1300_130004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_2_equals_8_l1300_130078

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

-- Define the inverse function of f
noncomputable def f_inv (x : ℝ) : ℝ := (3 * x - 4) / x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 1 / (f_inv x) + 7

-- Theorem statement
theorem g_of_2_equals_8 : g 2 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_2_equals_8_l1300_130078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_function_l1300_130017

-- Define the positive rationals
def PositiveRationals := {q : ℚ // q > 0}

-- Define the nth prime number
noncomputable def nthPrime (n : ℕ) : ℕ := sorry

-- Instance for multiplication of PositiveRationals
instance : Mul PositiveRationals where
  mul a b := ⟨a.val * b.val, sorry⟩

-- Instance for division of PositiveRationals
instance : Div PositiveRationals where
  div a b := ⟨a.val / b.val, sorry⟩

-- State the existence of the function f
theorem exists_special_function :
  ∃ (f : PositiveRationals → PositiveRationals),
    (∀ (x y : PositiveRationals), f (x * f y) = (f x) / y) ∧
    (∀ (k : ℕ), k ≥ 1 → f ⟨nthPrime (2 * k), sorry⟩ = ⟨nthPrime (2 * k + 1), sorry⟩) ∧
    (∀ (k : ℕ), k ≥ 1 → f ⟨nthPrime (2 * k + 1), sorry⟩ = ⟨(nthPrime (2 * k))⁻¹, sorry⟩) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_function_l1300_130017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_tests_l1300_130025

/-- Represents the status of a person with respect to the sceptervirus -/
inductive PersonStatus
| Safe
| Infected
| Healed
deriving Inhabited

/-- Represents the result of a sceptervirus test -/
inductive TestResult
| VirusPositive
| AntibodyPositive
| Neutral
deriving Inhabited

/-- A function that determines the test result based on a list of person statuses -/
def testResult (statuses : List PersonStatus) : TestResult := 
  sorry

/-- Theorem stating that n-1 tests are not sufficient to determine if all citizens are safe -/
theorem insufficient_tests (n : Nat) (h : n > 0) :
  ∃ (citizenStatuses : List PersonStatus),
    citizenStatuses.length = n ∧
    ∃ (tests : List (List Nat)),
      tests.length = n - 1 ∧
      (∀ test ∈ tests, test.all (· < n)) ∧
      (∀ test ∈ tests, testResult (test.map (fun i => citizenStatuses[i]!)) = TestResult.Neutral) ∧
      ¬(∀ status ∈ citizenStatuses, status = PersonStatus.Safe) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_insufficient_tests_l1300_130025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1300_130042

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - 2 * Real.cos x)

theorem domain_of_f :
  ∀ x : ℝ, f x ∈ Set.range f ↔ 
  ∃ k : ℤ, π/3 + 2*k*π ≤ x ∧ x ≤ 5*π/3 + 2*k*π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1300_130042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sqrt3_cos_eq_one_l1300_130035

theorem sin_sqrt3_cos_eq_one (x : ℝ) :
  Real.sin x + Real.sqrt 3 * Real.cos x = 1 ↔ ∃ k : ℤ, x = k * Real.pi + (-1)^k * (Real.pi / 6) - Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sqrt3_cos_eq_one_l1300_130035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l1300_130005

-- Define an odd function f on ℝ
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 3*x - 1 else -(2^(-x) + 3*(-x) - 1)

-- State the theorem
theorem odd_function_value : f (-2) = -9 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the if-then-else expression
  simp
  -- Evaluate the expression
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l1300_130005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_plus_y_less_than_4_l1300_130064

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- A point inside the square --/
structure PointInSquare (s : Square) where
  point : ℝ × ℝ
  inside : point.1 ≥ s.bottomLeft.1 ∧ 
           point.1 ≤ s.bottomLeft.1 + s.sideLength ∧
           point.2 ≥ s.bottomLeft.2 ∧ 
           point.2 ≤ s.bottomLeft.2 + s.sideLength

/-- The probability of an event occurring in a uniform distribution --/
noncomputable def probability (totalArea : ℝ) (eventArea : ℝ) : ℝ :=
  eventArea / totalArea

/-- The theorem to be proved --/
theorem probability_x_plus_y_less_than_4 (s : Square) 
  (h1 : s.bottomLeft = (0, 0)) 
  (h2 : s.sideLength = 3) : 
  probability (s.sideLength ^ 2) 
    (s.sideLength ^ 2 - (s.sideLength - 1) ^ 2 / 2) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_plus_y_less_than_4_l1300_130064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l1300_130031

noncomputable section

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define the points A and B
def point_A : ℝ × ℝ := (2, 0)
def point_B : ℝ × ℝ := (1, -Real.sqrt 3)

-- Define the line y = x
def line_y_eq_x : Set (ℝ × ℝ) := {p | p.1 = p.2}

-- Define the point that line l passes through
def point_on_l : ℝ × ℝ := (1, Real.sqrt 3 / 3)

-- Define the length of the chord
def chord_length : ℝ := 2 * Real.sqrt 3

-- Theorem statement
theorem circle_and_line_properties :
  (point_A ∈ circle_C) ∧
  (point_B ∈ circle_C) ∧
  (∃ c ∈ circle_C, c ∈ line_y_eq_x) ∧
  (∃ l : Set (ℝ × ℝ),
    (point_on_l ∈ l) ∧
    (∃ p q : ℝ × ℝ, p ≠ q ∧ p ∈ l ∧ q ∈ l ∧ p ∈ circle_C ∧ q ∈ circle_C ∧
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) ∧
    ((∀ x y : ℝ, (x, y) ∈ l ↔ x = 1) ∨
     (∃ k : ℝ, ∀ x y : ℝ, (x, y) ∈ l ↔ y = -Real.sqrt 3 / 3 * x + 2 * Real.sqrt 3 / 3))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_properties_l1300_130031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l1300_130002

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + c) / (x^2 - x - 6)

theorem exactly_one_vertical_asymptote (c : ℝ) :
  (∃! x, ¬ ∃ y, f c x = y) ↔ (c = -3 ∨ c = -8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l1300_130002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1300_130022

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (3*x + 4) / (x - 2)

-- State the theorem
theorem solution_set_of_inequality :
  ∀ x : ℝ, x ≠ 2 → (f x ≥ 4 ↔ x > 2 ∧ x ≤ 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1300_130022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l1300_130079

theorem cube_surface_area_increase (s : ℝ) (h : s > 0) : 
  (6 * (1.2 * s)^2 - 6 * s^2) / (6 * s^2) = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l1300_130079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_body_fat_sigma_approximation_l1300_130062

/-- Represents the cumulative distribution function of the standard normal distribution -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- The body fat percentage distribution parameters -/
structure BodyFatDistribution where
  μ : ℝ
  σ : ℝ

/-- Calculates the probability that a value from the given normal distribution is less than or equal to x -/
noncomputable def normalCDF (d : BodyFatDistribution) (x : ℝ) : ℝ :=
  Φ ((x - d.μ) / d.σ)

/-- Approximate equality for real numbers -/
def approx_equal (x y : ℝ) : Prop := sorry

notation:50 a " ≈ " b => approx_equal a b

theorem body_fat_sigma_approximation :
  ∀ d : BodyFatDistribution,
    d.μ = 0.2 →
    normalCDF d 0.17 = 0.16 →
    d.σ ≈ 0.03 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_body_fat_sigma_approximation_l1300_130062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducibility_l1300_130092

theorem fraction_irreducibility (n : ℕ) : 
  (Int.gcd (2 * n^2 + 11 * n - 18 : ℤ) (n + 7 : ℤ) = 1) ↔ n % 3 = 0 ∨ n % 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_irreducibility_l1300_130092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_points_expression_l1300_130026

theorem linear_function_points_expression : 
  ∀ (a b c : ℝ),
  (∃ (f : ℝ → ℝ), f = λ x ↦ a * x + b) →
  a * 3 + b = 8 →
  a * (-2) + b = 3 →
  a * (-3) + b = c →
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_points_expression_l1300_130026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_l1300_130071

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 - x

noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x - 1

theorem function_and_range (a b : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, f a b x = f a b x) ∧
  (f' a b 1 = 0) ∧ 
  (f' a b 2 = 0) ∧
  (∃ m : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc (-2) 0 ∧ x₂ ∈ Set.Icc (-2) 0 ∧
    (1/6 : ℝ) * x₁^3 - (3/4 : ℝ) * x₁^2 - 2 * x₁ - m = 0 ∧
    (1/6 : ℝ) * x₂^3 - (3/4 : ℝ) * x₂^2 - 2 * x₂ - m = 0) →
  (∀ x : ℝ, f a b x = -(1/6 : ℝ) * x^3 + (3/4 : ℝ) * x^2 - x) ∧
  (∃ m : ℝ, 0 ≤ m ∧ m < 13/12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_l1300_130071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_diagonal_angles_sum_lt_pi_l1300_130049

/-- Predicate to represent that α, β, γ are angles in a rectangular parallelepiped -/
def IsRectangularParallelepiped (α β γ : ℝ) : Prop :=
  0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2 ∧ 0 < γ ∧ γ < Real.pi/2

/-- Predicate to represent that α, β, γ are angles formed by the diagonal with edges -/
def DiagonalAngles (α β γ : ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  α = Real.arccos (a / Real.sqrt (a^2 + b^2 + c^2)) ∧
  β = Real.arccos (b / Real.sqrt (a^2 + b^2 + c^2)) ∧
  γ = Real.arccos (c / Real.sqrt (a^2 + b^2 + c^2))

/-- The sum of angles formed by the diagonal of a rectangular parallelepiped with its edges is less than π. -/
theorem parallelepiped_diagonal_angles_sum_lt_pi 
  (α β γ : ℝ) 
  (h_parallelepiped : IsRectangularParallelepiped α β γ) 
  (h_diagonal_angles : DiagonalAngles α β γ) : 
  α + β + γ < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_diagonal_angles_sum_lt_pi_l1300_130049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_y_axis_l1300_130007

/-- A point on a parabola with given distance to focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 8*x
  dist_to_focus : Real.sqrt ((x - 2)^2 + y^2) = 6

/-- The distance from a point on the parabola to the y-axis is 4 -/
theorem parabola_point_distance_to_y_axis (P : ParabolaPoint) : P.x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_y_axis_l1300_130007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1300_130047

noncomputable def f (x : ℝ) : ℝ := (x^3 - 64) / (x - 8)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 8} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1300_130047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minoxidil_mixture_concentration_l1300_130019

/-- Calculates the final concentration of a solution after mixing two solutions with different concentrations -/
noncomputable def final_concentration (v1 : ℝ) (c1 : ℝ) (v2 : ℝ) (c2 : ℝ) : ℝ :=
  ((v1 * c1 + v2 * c2) / (v1 + v2)) * 100

/-- Theorem stating that mixing 70 ml of 2% Minoxidil solution with 35 ml of 5% Minoxidil solution results in a 3% Minoxidil solution -/
theorem minoxidil_mixture_concentration :
  final_concentration 70 2 35 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minoxidil_mixture_concentration_l1300_130019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_ratio_l1300_130014

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the lengths
def AB : ℝ := 8
def AC : ℝ := 6
def CB : ℝ := 2
def CD : ℝ := 2

-- Define the areas
noncomputable def area_large_semicircle : ℝ := (1/2) * Real.pi * (AB/2)^2
noncomputable def area_AC_semicircle : ℝ := (1/2) * Real.pi * (AC/2)^2
noncomputable def area_CB_semicircle : ℝ := (1/2) * Real.pi * (CB/2)^2
noncomputable def shaded_area : ℝ := area_large_semicircle - (area_AC_semicircle + area_CB_semicircle)
noncomputable def area_CD_circle : ℝ := Real.pi * CD^2

-- The theorem to prove
theorem shaded_area_ratio :
  shaded_area / area_CD_circle = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_ratio_l1300_130014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equality_condition_l1300_130012

theorem sqrt_equality_condition (a b c : ℕ+) :
  Real.sqrt (2 * (a : ℝ) + (b : ℝ) / (c : ℝ)) = 2 * (a : ℝ) * Real.sqrt ((b : ℝ) / (c : ℝ)) ↔
  (c : ℝ) = (b : ℝ) * (4 * (a : ℝ)^2 - 1) / (2 * (a : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equality_condition_l1300_130012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_remainder_difference_l1300_130073

/-- The number of coefficients in the expansion of (1+x)^38 that leave a remainder of 1 when divided by 3 -/
def N₁ : ℕ := 8

/-- The number of coefficients in the expansion of (1+x)^38 that leave a remainder of 2 when divided by 3 -/
def N₂ : ℕ := 4

/-- The expansion of (1+x)^38 in ascending powers of x -/
noncomputable def expansion : Polynomial ℚ := (1 + Polynomial.X)^38

theorem coefficient_remainder_difference : N₁ - N₂ = 4 := by
  -- The proof goes here
  sorry

#eval N₁ - N₂  -- This will evaluate to 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_remainder_difference_l1300_130073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_jensen_is_affine_l1300_130036

/-- Jensen's functional equation -/
def satisfies_jensen (f : ℝ → ℝ) : Prop :=
  ∀ x y t : ℝ, t ∈ Set.Ioo 0 1 → f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y

/-- A function is monotonic if it preserves order -/
def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

theorem monotonic_jensen_is_affine (f : ℝ → ℝ) 
  (h_mono : monotonic f) (h_jensen : satisfies_jensen f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_jensen_is_affine_l1300_130036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_range_l1300_130074

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x - a * x^2 else a^x

-- Define the decreasing property for a function
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- Theorem statement
theorem f_decreasing_iff_a_in_range (a : ℝ) :
  is_decreasing (f a) ↔ (1/2 ≤ a ∧ a < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_range_l1300_130074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_sum_l1300_130096

noncomputable def b (n : ℕ) : ℝ := 2 * n

noncomputable def s (n : ℕ) : ℝ := n * (b 1 + b n) / 2

noncomputable def a (n : ℕ) : ℝ := (b n / 2) * (Real.sqrt 2) ^ (b n)

noncomputable def T (n : ℕ) : ℝ := (n + 1) * 2^(n + 1) + 2

theorem arithmetic_sequence_and_sum :
  (∀ n : ℕ, n ≥ 1 → b (n + 1) > b n) ∧  -- monotonically increasing
  b 3 = 6 ∧
  ∃ r : ℝ, r > 0 ∧ b 4 / (Real.sqrt (s 5 + 2)) = (Real.sqrt (s 5 + 2)) / b 2 ∧  -- geometric sequence condition
  (∀ n : ℕ, n ≥ 1 → b n = 2 * n) ∧  -- prove b_n = 2n
  (∀ n : ℕ, n ≥ 1 → T n = (n + 1) * 2^(n + 1) + 2)  -- prove T_n formula
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_sum_l1300_130096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_is_47_class_size_l1300_130095

/-- Represents the number of students who borrowed at least 3 books -/
def R : ℕ → ℕ := λ S => S - 25

/-- Represents the total number of students in the class -/
def total_students (R : ℕ) : ℕ := 25 + R

/-- The average number of books borrowed per student -/
def average_books : ℚ := 2

/-- Theorem stating that the total number of students is 47 -/
theorem total_students_is_47 (S : ℕ) :
  (0 * 5 + 1 * 12 + 2 * 8 + 3 * R S : ℚ) / S = average_books →
  S = 47 := by
  sorry

/-- Main theorem proving the total number of students in the class -/
theorem class_size : ∃ (S : ℕ), S = 47 ∧
  ((0 * 5 + 1 * 12 + 2 * 8 + 3 * (S - 25) : ℚ) / S = 2) ∧
  (5 + 12 + 8 + (S - 25) = S) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_is_47_class_size_l1300_130095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_neg_eight_l1300_130054

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 3^x - 1 else -3^(-x) + 1

-- State the theorem
theorem f_neg_two_equals_neg_eight :
  (∀ x, f (-x) = -f x) →  -- f is odd
  f (-2) = -8 :=
by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_two_equals_neg_eight_l1300_130054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_railway_stone_counts_l1300_130051

/-- Represents a stone marker on the railway line -/
structure StoneMarker where
  position : ℕ
  left_distance : ℚ
  right_distance : ℚ

/-- The railway line with its properties and stone markers -/
structure RailwayLine where
  length : ℚ
  stones : List StoneMarker

/-- Counts stones with markings where only two different digits appear -/
def count_two_digit_stones (railway : RailwayLine) : ℕ :=
  sorry

/-- Counts stones with markings where no digit appears more than once -/
def count_unique_digit_stones (railway : RailwayLine) : ℕ :=
  sorry

theorem railway_stone_counts
  (railway : RailwayLine)
  (h1 : railway.length = 777/10)
  (h2 : ∀ s ∈ railway.stones, s.position % 100 = 0)
  (h3 : ∀ s ∈ railway.stones, s.left_distance + s.right_distance = railway.length)
  (h4 : ∀ s ∈ railway.stones, (s.left_distance * 10).isInt ∧ (s.right_distance * 10).isInt) :
  count_two_digit_stones railway = 32 ∧
  count_unique_digit_stones railway = 304 := by
  sorry

#eval (777 : ℚ) / 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_railway_stone_counts_l1300_130051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excellent_team_probability_minimum_rounds_l1300_130091

/-- Probability of student A answering each question correctly -/
noncomputable def p₁ : ℝ := 3/4

/-- Probability of student B answering each question correctly -/
noncomputable def p₂ : ℝ := 2/3

/-- Probability of being an "excellent team" in one round -/
noncomputable def excellent_team_prob : ℝ := 2 * p₁^2 * p₂ * (1 - p₂) + 2 * p₂^2 * p₁ * (1 - p₁) + p₁^2 * p₂^2

/-- The sum of probabilities for part 2 -/
noncomputable def prob_sum : ℝ := 6/5

/-- Maximum probability of being an "excellent team" in one round for part 2 -/
noncomputable def max_prob : ℝ := 297/625

/-- Number of times they want to be an "excellent team" -/
def target_excellent : ℕ := 9

theorem excellent_team_probability :
  excellent_team_prob = 2/3 := by
  sorry

theorem minimum_rounds (n : ℕ) :
  (∃ p₁' p₂' : ℝ, p₁' + p₂' = prob_sum ∧ 
   max_prob * n ≥ target_excellent) →
  n ≥ 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excellent_team_probability_minimum_rounds_l1300_130091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_four_value_l1300_130027

noncomputable def M₀ : ℝ := 24

noncomputable def M (t : ℝ) : ℝ := M₀ * (1.2 ^ (-t / 2))

theorem M_four_value :
  (∀ t, HasDerivAt (M) (-1/2 * Real.log 1.2 * M₀ * (1.2 ^ (-t / 2))) t) →
  (HasDerivAt M (-10 * Real.log 1.2) 2) →
  M 4 = 50/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_four_value_l1300_130027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_l1300_130076

-- Define the curve f(x) = x^3 - 2x^2 + 1
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

-- Define points P and Q
def P : ℝ × ℝ := (1, 0)
def Q : ℝ × ℝ := (2, 1)

-- Define the tangent line l₁ at point P
def l₁ (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the two possible tangent lines l₂ passing through Q
def l₂₁ (y : ℝ) : Prop := y = 1
def l₂₂ (x y : ℝ) : Prop := 4*x - y - 7 = 0

theorem tangent_lines_theorem :
  (∀ x y : ℝ, l₁ x y ↔ (y - P.2 = f' P.1 * (x - P.1))) ∧
  ((∀ y : ℝ, l₂₁ y → (∃ x₀ : ℝ, y - f x₀ = f' x₀ * (Q.1 - x₀) ∧ y = Q.2)) ∨
   (∀ x y : ℝ, l₂₂ x y → (∃ x₀ : ℝ, y - f x₀ = f' x₀ * (x - x₀) ∧ x = Q.1 ∧ y = Q.2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_theorem_l1300_130076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1300_130015

open Real

theorem trigonometric_equation_solution (x : ℝ) :
  Real.sin x * Real.sin (2 * x) * Real.sin (3 * x) + Real.cos x * Real.cos (2 * x) * Real.cos (3 * x) = 1 →
  ∃ k : ℤ, x = k * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l1300_130015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1300_130072

open Real

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := (cos x)^2 + Real.sqrt 3 * sin x * cos x

-- State the theorem
theorem f_range :
  ∀ y ∈ Set.range (f ∘ (fun x => x * π)),
    0 ≤ y ∧ y ≤ (Real.sqrt 3 + 1) / 2 ∧
    ∃ x ∈ Set.Icc (-1/6 : ℝ) (1/4 : ℝ), y = f (x * π) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1300_130072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l1300_130075

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- The magnitude (Euclidean norm) of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem parallel_vectors_magnitude :
  let m : ℝ × ℝ := (-1, 2)
  let n : ℝ × ℝ := (2, -4)
  parallel m n → magnitude n = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_magnitude_l1300_130075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_difference_inverse_l1300_130020

theorem inverse_difference_inverse : ((5 : ℝ)⁻¹ - (4 : ℝ)⁻¹)⁻¹ = -20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_difference_inverse_l1300_130020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1300_130034

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation x^2 = 4y -/
def Parabola := {p : Point | p.x^2 = 4 * p.y}

/-- The focus of the parabola -/
def F : Point := ⟨0, 1⟩

/-- The given point A -/
def A : Point := ⟨-1, 8⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The theorem to be proved -/
theorem min_distance_sum (P : Point) (h : P ∈ Parabola) :
  ∃ (min : ℝ), ∀ (Q : Point), Q ∈ Parabola → distance P A + distance P F ≥ min ∧ min = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1300_130034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_99_l1300_130013

/-- A geometric sequence with common ratio 2 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n

/-- The sum of every third term from a₁ to a₉₇ is 11 -/
def sum_of_every_third_term (a : ℕ → ℝ) : Prop :=
  (Finset.range 33).sum (λ i ↦ a (3 * i + 1)) = 11

/-- The sum of the first 99 terms -/
def sum_99 (a : ℕ → ℝ) : ℝ :=
  (Finset.range 99).sum a

theorem geometric_sequence_sum_99 (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : sum_of_every_third_term a) : 
  sum_99 a = 77 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_99_l1300_130013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_conditions_l1300_130063

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def contains_digit_9 : ℕ → Prop
  | 0 => False
  | n+1 => n % 10 = 9 ∨ contains_digit_9 n

def is_of_form (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = 2^a * 5^b * 3^c

theorem smallest_n_with_conditions :
  (∀ m : ℕ, m < 5120 →
    ¬(is_terminating_decimal m ∧ contains_digit_9 m ∧ is_of_form m)) ∧
  (is_terminating_decimal 5120 ∧ contains_digit_9 5120 ∧ is_of_form 5120) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_conditions_l1300_130063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_field_is_potential_l1300_130088

-- Define the vector space
variable (x y z : ℝ)

-- Define the scalar charge and distance
variable (q : ℝ)
noncomputable def r (x y z : ℝ) : ℝ := Real.sqrt (x^2 + y^2 + z^2)

-- Define the electric field vector
noncomputable def E (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := r x y z
  (q * x / r^3, q * y / r^3, q * z / r^3)

-- Define the potential function
noncomputable def φ (x y z : ℝ) : ℝ := -q / r x y z

-- Define the gradient of φ
noncomputable def grad_φ (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := r x y z
  (-q * x / r^3, -q * y / r^3, -q * z / r^3)

-- Theorem statement
theorem electric_field_is_potential (x y z : ℝ) : E x y z = grad_φ x y z := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_field_is_potential_l1300_130088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_wins_symmetric_petya_wins_asymmetric_l1300_130023

/-- Represents the state of the caramel game -/
structure GameState where
  bag1 : ℕ
  bag2 : ℕ

/-- Determines if a move is valid in the caramel game -/
def validMove (state : GameState) (move : ℕ) : Prop :=
  (move > 0) ∧ (move ≤ state.bag1 ∨ move ≤ state.bag2)

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : ℕ) : GameState :=
  if move ≤ state.bag1 then
    { bag1 := state.bag1 - move, bag2 := state.bag2 + move }
  else
    { bag1 := state.bag1 + move, bag2 := state.bag2 - move }

/-- Determines if the game is over (one bag is empty) -/
def gameOver (state : GameState) : Prop :=
  state.bag1 = 0 ∨ state.bag2 = 0

/-- Represents a winning strategy for a player -/
def winningStrategy (initialState : GameState) (playerMoves : ℕ → ℕ) : Prop :=
  ∀ (opponentMoves : ℕ → ℕ),
    ∃ (gameSequence : ℕ → GameState),
      gameSequence 0 = initialState ∧
      (∀ n, gameSequence (n + 1) = 
        if n % 2 = 0 
        then applyMove (gameSequence n) (playerMoves n)
        else applyMove (gameSequence n) (opponentMoves n)) ∧
      ∃ n, gameOver (gameSequence n) ∧ n % 2 = 0

theorem vasya_wins_symmetric : 
  ∃ (strategy : ℕ → ℕ), winningStrategy ⟨2021, 2021⟩ strategy := by
  sorry

theorem petya_wins_asymmetric : 
  ∃ (strategy : ℕ → ℕ), winningStrategy ⟨1000, 2000⟩ strategy := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_wins_symmetric_petya_wins_asymmetric_l1300_130023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangements_l1300_130086

/-- The number of ways to arrange n distinct objects taken k at a time -/
def A (n k : ℕ) : ℕ := sorry

/-- The number of male actors -/
def num_male_actors : ℕ := 4

/-- The number of female actors -/
def num_female_actors : ℕ := 5

/-- The total number of actors -/
def total_actors : ℕ := num_male_actors + num_female_actors

theorem photo_arrangements :
  ∃ (x : ℕ), 
    (A num_female_actors num_female_actors) * (A (num_female_actors + 1) num_male_actors) -
    2 * (A (num_female_actors - 1) (num_female_actors - 1)) * (A num_female_actors num_male_actors) = x ∧
    x = (A num_female_actors num_female_actors) * (A (num_female_actors + 1) num_male_actors) -
        2 * (A (num_female_actors - 1) (num_female_actors - 1)) * (A num_female_actors num_male_actors) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_photo_arrangements_l1300_130086
