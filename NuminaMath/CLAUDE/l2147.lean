import Mathlib

namespace g_zero_value_l2147_214754

def f (x : ℝ) : ℝ := 2 * x + 3

theorem g_zero_value (g : ℝ → ℝ) (h : ∀ x, g (x + 2) = f x) : g 0 = -1 := by
  sorry

end g_zero_value_l2147_214754


namespace book_pages_count_l2147_214799

/-- Represents the number of pages read in a day period --/
structure ReadingPeriod where
  days : ℕ
  pagesPerDay : ℕ

/-- Calculates the total pages read in a period --/
def totalPages (period : ReadingPeriod) : ℕ :=
  period.days * period.pagesPerDay

/-- Represents Robert's reading schedule --/
def robertReading : List ReadingPeriod :=
  [{ days := 3, pagesPerDay := 28 },
   { days := 3, pagesPerDay := 35 },
   { days := 3, pagesPerDay := 42 }]

/-- The number of pages Robert read on the last day --/
def lastDayPages : ℕ := 15

/-- Theorem stating the total number of pages in the book --/
theorem book_pages_count :
  (robertReading.map totalPages).sum + lastDayPages = 330 := by
  sorry

end book_pages_count_l2147_214799


namespace intersection_implies_value_l2147_214764

theorem intersection_implies_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, a^2+1, 2*a-1}
  (A ∩ B = {-3}) → a = -1 := by
sorry

end intersection_implies_value_l2147_214764


namespace odd_function_sum_l2147_214752

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_sum : ∀ x, f (2 + x) + f (2 - x) = 0)
  (h_f1 : f 1 = 9) :
  f 2010 + f 2011 + f 2012 = -9 := by
sorry

end odd_function_sum_l2147_214752


namespace sqrt_equation_solution_l2147_214757

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (7 + 3 * z) = 15 :=
by
  -- The proof would go here
  sorry

end sqrt_equation_solution_l2147_214757


namespace circle_ratio_l2147_214771

theorem circle_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : c^2 - a^2 = 4 * a^2) 
  (h2 : b^2 = (a^2 + c^2) / 2) : 
  a / c = 1 / Real.sqrt 5 := by
  sorry

end circle_ratio_l2147_214771


namespace special_parallelogram_perimeter_l2147_214720

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  /-- The length of the perpendicular from one vertex to the opposite side -/
  perpendicular : ℝ
  /-- The length of one diagonal -/
  diagonal : ℝ

/-- Theorem: The perimeter of a special parallelogram is 36 -/
theorem special_parallelogram_perimeter 
  (P : SpecialParallelogram) 
  (h1 : P.perpendicular = 12) 
  (h2 : P.diagonal = 15) : 
  Real.sqrt ((P.diagonal ^ 2 - P.perpendicular ^ 2) / 4) * 4 = 36 := by
  sorry

#check special_parallelogram_perimeter

end special_parallelogram_perimeter_l2147_214720


namespace lcm_of_231_and_300_l2147_214728

theorem lcm_of_231_and_300 (lcm hcf : ℕ) (a b : ℕ) : 
  hcf = 30 → a = 231 → b = 300 → lcm * hcf = a * b → lcm = 2310 := by
  sorry

end lcm_of_231_and_300_l2147_214728


namespace greatest_divisor_four_consecutive_integers_l2147_214725

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  (∃ k : ℕ, k > 12 ∧ (∀ m : ℕ, m > 0 → k ∣ (m * (m + 1) * (m + 2) * (m + 3)))) →
  False :=
sorry

end greatest_divisor_four_consecutive_integers_l2147_214725


namespace minimum_peanuts_min_peanuts_is_25_l2147_214730

theorem minimum_peanuts : ℕ → Prop :=
  fun n => (n % 3 = 1) ∧ 
           ((n - 1) / 3 - 1) % 3 = 0 ∧ 
           (((n - 1) / 3 - 1 - 1) / 3 - 1) % 3 = 0

theorem min_peanuts_is_25 : minimum_peanuts 25 ∧ ∀ m < 25, ¬minimum_peanuts m := by
  sorry

end minimum_peanuts_min_peanuts_is_25_l2147_214730


namespace knight_probability_after_2023_moves_l2147_214716

/-- Knight's move on an infinite chessboard -/
def KnightMove (a b : ℤ) : Set (ℤ × ℤ) :=
  {(a+1, b+2), (a+1, b-2), (a-1, b+2), (a-1, b-2),
   (a+2, b+1), (a+2, b-1), (a-2, b+1), (a-2, b-1)}

/-- Probability space for knight's moves -/
def KnightProbSpace : Type := ℤ × ℤ

/-- Probability measure for knight's moves -/
noncomputable def KnightProb : KnightProbSpace → ℝ := sorry

/-- The set of positions (a, b) where a ≡ 4 (mod 8) and b ≡ 5 (mod 8) -/
def TargetPositions : Set (ℤ × ℤ) :=
  {(a, b) | a % 8 = 4 ∧ b % 8 = 5}

/-- The probability of the knight being at a target position after n moves -/
noncomputable def ProbAtTargetAfterMoves (n : ℕ) : ℝ := sorry

theorem knight_probability_after_2023_moves :
  ProbAtTargetAfterMoves 2023 = 1/32 - 1/2^2027 := by sorry

end knight_probability_after_2023_moves_l2147_214716


namespace count_eights_theorem_l2147_214797

/-- Count of digit 8 appearances in integers from 1 to 800 -/
def count_eights : ℕ := 160

/-- The upper bound of the integer range -/
def upper_bound : ℕ := 800

/-- Counts the occurrences of a specific digit in a given range of integers -/
def count_digit_occurrences (digit : ℕ) (lower : ℕ) (upper : ℕ) : ℕ :=
  sorry

theorem count_eights_theorem :
  count_digit_occurrences 8 1 upper_bound = count_eights :=
sorry

end count_eights_theorem_l2147_214797


namespace infinite_representable_elements_l2147_214788

def is_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, a (i + 1) > a i

theorem infinite_representable_elements 
  (a : ℕ → ℕ) 
  (h_increasing : is_increasing_sequence a) :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧
    ∃ (x y h k : ℕ), 
      0 < h ∧ h < k ∧ k < m ∧
      a m = x * a h + y * a k :=
sorry

end infinite_representable_elements_l2147_214788


namespace total_teachers_is_182_l2147_214785

/-- Represents the number of teachers in different categories and survey selections -/
structure SchoolTeachers where
  senior : ℕ
  intermediate : ℕ
  survey_total : ℕ
  survey_other : ℕ

/-- Calculates the total number of teachers in the school -/
def total_teachers (s : SchoolTeachers) : ℕ :=
  s.senior + s.intermediate + (s.survey_total - (s.survey_other + s.senior + s.intermediate))

/-- Theorem stating that given the specific numbers, the total teachers is 182 -/
theorem total_teachers_is_182 (s : SchoolTeachers) 
  (h1 : s.senior = 26)
  (h2 : s.intermediate = 104)
  (h3 : s.survey_total = 56)
  (h4 : s.survey_other = 16) :
  total_teachers s = 182 := by
  sorry

#eval total_teachers ⟨26, 104, 56, 16⟩

end total_teachers_is_182_l2147_214785


namespace concyclic_projections_l2147_214768

-- Define a circle and a point on a plane
variable (Circle : Type) (Point : Type)

-- Define a function to check if points are concyclic
variable (are_concyclic : Circle → List Point → Prop)

-- Define a function for orthogonal projection
variable (orthogonal_projection : Point → Point → Point → Point)

-- Theorem statement
theorem concyclic_projections
  (A B C D A' B' C' D' : Point) (circle : Circle) :
  are_concyclic circle [A, B, C, D] →
  A' = orthogonal_projection A B D →
  C' = orthogonal_projection C B D →
  B' = orthogonal_projection B A C →
  D' = orthogonal_projection D A C →
  ∃ (circle' : Circle), are_concyclic circle' [A', B', C', D'] :=
by sorry

end concyclic_projections_l2147_214768


namespace triangle_side_length_l2147_214729

theorem triangle_side_length (a b c : Real) (angle_A angle_B : Real) :
  angle_A = 30 * Real.pi / 180 →
  angle_B = 45 * Real.pi / 180 →
  c = 8 →
  b = 4 * Real.sqrt 2 :=
by sorry

end triangle_side_length_l2147_214729


namespace infinite_matrices_squared_zero_l2147_214700

/-- The set of 2x2 real matrices B satisfying B^2 = 0 is infinite -/
theorem infinite_matrices_squared_zero :
  Set.Infinite {B : Matrix (Fin 2) (Fin 2) ℝ | B * B = 0} := by
  sorry

end infinite_matrices_squared_zero_l2147_214700


namespace p_2017_equals_14_l2147_214705

/-- Function that calculates the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ := sorry

/-- Function that calculates the number of digits of a positive integer -/
def numberOfDigits (n : ℕ+) : ℕ := sorry

/-- Function P(n) as defined in the problem -/
def P (n : ℕ+) : ℕ := sumOfDigits n + numberOfDigits n

/-- Theorem stating that P(2017) = 14 -/
theorem p_2017_equals_14 : P 2017 = 14 := by sorry

end p_2017_equals_14_l2147_214705


namespace M_mod_1000_l2147_214738

/-- The number of distinguishable flagpoles -/
def num_flagpoles : ℕ := 2

/-- The total number of flags -/
def total_flags : ℕ := 21

/-- The number of blue flags -/
def blue_flags : ℕ := 12

/-- The number of green flags -/
def green_flags : ℕ := 9

/-- The minimum number of flags required on each flagpole -/
def min_flags_per_pole : ℕ := 3

/-- The function to calculate the number of distinguishable arrangements -/
def M : ℕ := sorry

/-- Theorem stating the remainder when M is divided by 1000 -/
theorem M_mod_1000 : M % 1000 = 596 := by sorry

end M_mod_1000_l2147_214738


namespace point_a_in_second_quadrant_l2147_214782

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the second quadrant -/
def in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The specific point we're considering -/
def point_a : Point :=
  { x := -1, y := 2 }

/-- Theorem stating that point_a is in the second quadrant -/
theorem point_a_in_second_quadrant : in_second_quadrant point_a := by
  sorry

end point_a_in_second_quadrant_l2147_214782


namespace polynomial_coefficients_l2147_214775

-- Define the polynomial
def p (x a₄ a₃ a₂ a₁ a₀ : ℝ) : ℝ := (x + 2)^5 - (x + 1)^5 - (a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀)

-- State the theorem
theorem polynomial_coefficients :
  ∃ (a₄ a₃ a₂ : ℝ), ∀ x, p x a₄ a₃ a₂ 75 31 = 0 :=
by sorry

end polynomial_coefficients_l2147_214775


namespace polynomial_equation_solutions_l2147_214761

-- Define the polynomials p and q
def p (x : ℂ) : ℂ := x^5 + x
def q (x : ℂ) : ℂ := x^5 + x^2

-- Define a primitive third root of unity
noncomputable def ε : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

-- Define the set of solution pairs
def solution_pairs : Set (ℂ × ℂ) :=
  {(ε, 1 - ε), (ε^2, 1 - ε^2), 
   ((1 + Complex.I * Real.sqrt 3) / 2, (1 - Complex.I * Real.sqrt 3) / 2),
   ((1 - Complex.I * Real.sqrt 3) / 2, (1 + Complex.I * Real.sqrt 3) / 2)}

-- State the theorem
theorem polynomial_equation_solutions :
  ∀ w z : ℂ, w ≠ z → (p w = p z ∧ q w = q z) ↔ (w, z) ∈ solution_pairs :=
by sorry

end polynomial_equation_solutions_l2147_214761


namespace min_cuts_for_polygons_l2147_214779

/-- Represents the number of sides in the target polygons -/
def target_sides : Nat := 20

/-- Represents the number of target polygons to be created -/
def num_polygons : Nat := 3

/-- Represents the initial number of vertices in the rectangular sheet -/
def initial_vertices : Nat := 4

/-- Represents the maximum increase in vertices per cut -/
def max_vertex_increase : Nat := 4

/-- Theorem stating the minimum number of cuts required -/
theorem min_cuts_for_polygons : 
  ∃ (n : Nat), n = 50 ∧ 
  (∀ m : Nat, m < n → 
    (m + 1) * initial_vertices + m * max_vertex_increase < 
    num_polygons * target_sides + 3 * (m + 1 - num_polygons)) ∧
  ((n + 1) * initial_vertices + n * max_vertex_increase ≥ 
    num_polygons * target_sides + 3 * (n + 1 - num_polygons)) := by
  sorry

end min_cuts_for_polygons_l2147_214779


namespace triangle_construction_l2147_214792

-- Define the necessary structures and properties
structure Point where
  x : ℝ
  y : ℝ

def nonCollinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) ≠ (C.x - A.x) * (B.y - A.y)

def isMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def isOrthocenter (P A B C : Point) : Prop :=
  ((B.y - A.y) * (P.x - A.x) + (A.x - B.x) * (P.y - A.y) = 0) ∧
  ((C.y - B.y) * (P.x - B.x) + (B.x - C.x) * (P.y - B.y) = 0) ∧
  ((A.y - C.y) * (P.x - C.x) + (C.x - A.x) * (P.y - C.y) = 0)

-- State the theorem
theorem triangle_construction (M N P : Point) (h : nonCollinear M N P) :
  ∃ (A B C : Point),
    (isMidpoint M A B ∨ isMidpoint M B C ∨ isMidpoint M A C) ∧
    (isMidpoint N A B ∨ isMidpoint N B C ∨ isMidpoint N A C) ∧
    (isMidpoint M A B → isMidpoint N A C ∨ isMidpoint N B C) ∧
    (isMidpoint M B C → isMidpoint N A B ∨ isMidpoint N A C) ∧
    (isMidpoint M A C → isMidpoint N A B ∨ isMidpoint N B C) ∧
    isOrthocenter P A B C :=
  sorry

end triangle_construction_l2147_214792


namespace sam_digits_of_pi_l2147_214702

theorem sam_digits_of_pi (carlos mina sam : ℕ) : 
  sam = carlos + 6 →
  mina = 6 * carlos →
  mina = 24 →
  sam = 10 := by sorry

end sam_digits_of_pi_l2147_214702


namespace system_solution_l2147_214740

theorem system_solution (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y z : ℝ),
    (x + a*y + a^2*z + a^3 = 0) ∧
    (x + b*y + b^2*z + b^3 = 0) ∧
    (x + c*y + c^2*z + c^3 = 0) ∧
    (x = -a*b*c) ∧
    (y = a*b + b*c + c*a) ∧
    (z = -(a + b + c)) :=
by sorry

end system_solution_l2147_214740


namespace repeated_two_digit_divisible_by_101_l2147_214748

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // n ≥ 10 ∧ n < 100 }

/-- Constructs a four-digit number by repeating a two-digit number -/
def repeat_two_digit (n : TwoDigitNumber) : ℕ :=
  100 * n.val + n.val

theorem repeated_two_digit_divisible_by_101 (n : TwoDigitNumber) :
  (repeat_two_digit n) % 101 = 0 := by
  sorry

end repeated_two_digit_divisible_by_101_l2147_214748


namespace inequality_equivalence_f_less_than_one_l2147_214760

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part I: Equivalence of the inequality
theorem inequality_equivalence (x : ℝ) : f x < x + 1 ↔ 0 < x ∧ x < 2 := by
  sorry

-- Part II: Prove f(x) < 1 under given conditions
theorem f_less_than_one (x y : ℝ) 
  (h1 : |x - y - 1| ≤ 1/3) (h2 : |2*y + 1| ≤ 1/6) : f x < 1 := by
  sorry

end inequality_equivalence_f_less_than_one_l2147_214760


namespace sequence_problem_l2147_214789

theorem sequence_problem (a : Fin 100 → ℚ) :
  (∀ i : Fin 98, a (Fin.succ i) = a i * a (Fin.succ (Fin.succ i))) →
  (∀ i : Fin 100, a i ≠ 0) →
  a 0 = 2018 →
  a 99 = 1 / 2018 := by
sorry

end sequence_problem_l2147_214789


namespace circular_arrangement_students_l2147_214717

/-- Given a circular arrangement of students, if the 10th and 45th positions
    are opposite each other, then the total number of students is 70. -/
theorem circular_arrangement_students (n : ℕ) : n = 70 :=
  by
  -- Assume the 10th and 45th positions are opposite each other
  have h1 : 45 - 10 = n / 2 := by sorry
  
  -- The total number of students is twice the difference between opposite positions
  have h2 : n = 2 * (45 - 10) := by sorry
  
  -- Prove that n equals 70
  sorry

end circular_arrangement_students_l2147_214717


namespace nelly_outbid_multiple_l2147_214765

def joes_bid : ℕ := 160000
def nellys_bid : ℕ := 482000
def additional_amount : ℕ := 2000

theorem nelly_outbid_multiple : 
  (nellys_bid - joes_bid - additional_amount) / joes_bid = 2 := by
  sorry

end nelly_outbid_multiple_l2147_214765


namespace symmetry_of_product_l2147_214758

def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, f (-x) = -f x

def IsEvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, g (-x) = g x

theorem symmetry_of_product (f g : ℝ → ℝ) 
    (hf : IsOddFunction f) (hg : IsEvenFunction g) : 
    IsOddFunction (fun x ↦ f x * g x) := by
  sorry

end symmetry_of_product_l2147_214758


namespace sum_of_repeating_decimals_l2147_214736

/-- Represents a repeating decimal with a single repeating digit -/
def single_repeating_decimal (n : ℕ) : ℚ :=
  n / 9

/-- Represents a repeating decimal with two repeating digits -/
def double_repeating_decimal (n : ℕ) : ℚ :=
  n / 99

/-- The sum of 0.3̄ and 0.0̄2̄ equals 35/99 -/
theorem sum_of_repeating_decimals :
  single_repeating_decimal 3 + double_repeating_decimal 2 = 35 / 99 := by
  sorry

end sum_of_repeating_decimals_l2147_214736


namespace largest_coefficient_binomial_expansion_l2147_214791

theorem largest_coefficient_binomial_expansion :
  ∃ (k : ℕ) (c : ℚ), 
    (k = 3 ∧ c = 160) ∧
    ∀ (j : ℕ) (d : ℚ), 
      (Nat.choose 6 j * (2 ^ j)) ≤ (Nat.choose 6 k * (2 ^ k)) :=
by sorry

end largest_coefficient_binomial_expansion_l2147_214791


namespace limit_implies_range_l2147_214795

theorem limit_implies_range (a : ℝ) : 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |3^n / (3^(n+1) + (a+1)^n) - 1/3| < ε) → 
  a ∈ Set.Ioo (-4 : ℝ) 2 := by
sorry

end limit_implies_range_l2147_214795


namespace f_properties_l2147_214753

def f (x : ℝ) := x^3 - x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, x < -Real.sqrt 3 / 3 → ∀ y, x < y → f x < f y) ∧
  (∀ x, x > Real.sqrt 3 / 3 → ∀ y, x < y → f x < f y) :=
sorry

end f_properties_l2147_214753


namespace quadratic_polynomial_form_l2147_214713

/-- A quadratic polynomial with specific properties -/
structure QuadraticPolynomial where
  p : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c
  value_at_neg_two : p (-2) = 8
  asymptotes : Set ℝ
  asymptotes_def : asymptotes = {-2, 2}
  is_asymptote : ∀ x ∈ asymptotes, ∀ ε > 0, ∃ δ > 0, ∀ y, |y - x| < δ → |1 / (p y)| > 1 / ε

/-- The theorem stating the specific form of the quadratic polynomial -/
theorem quadratic_polynomial_form (f : QuadraticPolynomial) : f.p = λ x => -2 * x^2 + 8 := by
  sorry

end quadratic_polynomial_form_l2147_214713


namespace unique_triple_exists_l2147_214737

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem unique_triple_exists :
  ∃! (x y z : ℕ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    Nat.lcm x y = 90 ∧
    Nat.lcm x z = 720 ∧
    Nat.lcm y z = 1000 ∧
    x < y ∧ y < z ∧
    (is_square x ∨ is_square y ∨ is_square z) :=
by sorry

end unique_triple_exists_l2147_214737


namespace distance_between_points_l2147_214767

theorem distance_between_points : Real.sqrt ((24 - 0)^2 + (0 - 10)^2) = 26 := by
  sorry

end distance_between_points_l2147_214767


namespace two_numbers_with_special_properties_l2147_214770

theorem two_numbers_with_special_properties : ∃ (a b : ℕ), 
  a ≠ b ∧
  a > 9 ∧ b > 9 ∧
  (a + b) / 2 ≥ 10 ∧ (a + b) / 2 ≤ 99 ∧
  Nat.sqrt (a * b) ≥ 10 ∧ Nat.sqrt (a * b) ≤ 99 ∧
  (a = 98 ∧ b = 32 ∨ a = 32 ∧ b = 98) :=
by sorry

end two_numbers_with_special_properties_l2147_214770


namespace distance_theorem_l2147_214794

/-- The distance between Maxwell's and Brad's homes --/
def distance_between_homes (maxwell_speed : ℝ) (brad_speed : ℝ) (brad_delay : ℝ) (total_time : ℝ) : ℝ :=
  maxwell_speed * total_time + brad_speed * (total_time - brad_delay)

/-- Theorem stating the distance between Maxwell's and Brad's homes --/
theorem distance_theorem (maxwell_speed : ℝ) (brad_speed : ℝ) (brad_delay : ℝ) (total_time : ℝ)
  (h1 : maxwell_speed = 4)
  (h2 : brad_speed = 6)
  (h3 : brad_delay = 1)
  (h4 : total_time = 2) :
  distance_between_homes maxwell_speed brad_speed brad_delay total_time = 14 := by
  sorry

#check distance_theorem

end distance_theorem_l2147_214794


namespace tan_sum_half_angles_l2147_214733

theorem tan_sum_half_angles (p q : ℝ) 
  (h1 : Real.cos p + Real.cos q = 1/3)
  (h2 : Real.sin p + Real.sin q = 5/13) : 
  Real.tan ((p + q)/2) = 15/13 := by
  sorry

end tan_sum_half_angles_l2147_214733


namespace sqrt_x_minus_one_meaningful_l2147_214739

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) → x ≥ 1 := by sorry

end sqrt_x_minus_one_meaningful_l2147_214739


namespace binomial_coefficient_divisible_by_prime_l2147_214776

theorem binomial_coefficient_divisible_by_prime (p k : ℕ) :
  Nat.Prime p → 1 ≤ k → k ≤ p - 1 → p ∣ Nat.choose p k := by
  sorry

end binomial_coefficient_divisible_by_prime_l2147_214776


namespace chastity_lollipop_cost_l2147_214774

def lollipop_cost (initial_money : ℚ) (remaining_money : ℚ) (num_lollipops : ℕ) (num_gummy_packs : ℕ) (gummy_pack_cost : ℚ) : ℚ :=
  ((initial_money - remaining_money) - (num_gummy_packs * gummy_pack_cost)) / num_lollipops

theorem chastity_lollipop_cost :
  lollipop_cost 15 5 4 2 2 = (3/2) :=
sorry

end chastity_lollipop_cost_l2147_214774


namespace chocolate_bar_weight_l2147_214722

/-- Proves that given a 2-kilogram box containing 16 chocolate bars, 
    each chocolate bar weighs 125 grams. -/
theorem chocolate_bar_weight :
  let box_weight_kg : ℕ := 2
  let bars_per_box : ℕ := 16
  let grams_per_kg : ℕ := 1000
  let box_weight_g : ℕ := box_weight_kg * grams_per_kg
  let bar_weight_g : ℕ := box_weight_g / bars_per_box
  bar_weight_g = 125 := by
  sorry

end chocolate_bar_weight_l2147_214722


namespace staircase_region_perimeter_l2147_214715

/-- Represents the staircase-shaped region with an adjoined right triangle -/
structure StaircaseRegion where
  staircase_side_length : ℝ
  staircase_side_count : ℕ
  triangle_leg1 : ℝ
  triangle_leg2 : ℝ
  total_area : ℝ

/-- Calculates the perimeter of the StaircaseRegion -/
def calculate_perimeter (region : StaircaseRegion) : ℝ :=
  sorry

/-- Theorem stating the perimeter of the specific StaircaseRegion -/
theorem staircase_region_perimeter :
  let region : StaircaseRegion := {
    staircase_side_length := 2,
    staircase_side_count := 10,
    triangle_leg1 := 3,
    triangle_leg2 := 4,
    total_area := 150
  }
  calculate_perimeter region = 81.77 := by
  sorry

end staircase_region_perimeter_l2147_214715


namespace least_n_satisfying_inequality_l2147_214743

theorem least_n_satisfying_inequality :
  ∀ n : ℕ+, n < 4 → (1 : ℚ) / n - (1 : ℚ) / (n + 1) ≥ (1 : ℚ) / 15 ∧
  (1 : ℚ) / 4 - (1 : ℚ) / 5 < (1 : ℚ) / 15 :=
by sorry

end least_n_satisfying_inequality_l2147_214743


namespace quadratic_function_properties_l2147_214778

/-- A quadratic function with specific properties -/
def QuadraticFunction (m n : ℝ) : ℝ → ℝ := fun x ↦ m * x^2 - 2 * m * x + n + 1

/-- The derived function f based on g -/
def DerivedFunction (g : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := fun x ↦ g x + (2 - a) * x

/-- The theorem statement -/
theorem quadratic_function_properties 
  (m n : ℝ) 
  (h_m : m > 0)
  (h_max : ∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, QuadraticFunction m n x ≥ QuadraticFunction m n y)
  (h_min : ∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, QuadraticFunction m n x ≤ QuadraticFunction m n y)
  (h_max_val : ∃ x ∈ Set.Icc 0 3, QuadraticFunction m n x = 4)
  (h_min_val : ∃ x ∈ Set.Icc 0 3, QuadraticFunction m n x = 0)
  : 
  (∀ x, QuadraticFunction m n x = x^2 - 2*x + 1) ∧
  (∃ a, (a = -5 ∨ a = 4) ∧ 
    (∃ x ∈ Set.Icc (-1) 2, ∀ y ∈ Set.Icc (-1) 2, 
      DerivedFunction (QuadraticFunction m n) a x ≤ DerivedFunction (QuadraticFunction m n) a y) ∧
    (∃ x ∈ Set.Icc (-1) 2, DerivedFunction (QuadraticFunction m n) a x = -3)) :=
by sorry

end quadratic_function_properties_l2147_214778


namespace black_men_tshirt_cost_l2147_214731

/-- Represents the cost of t-shirts and number of employees --/
structure TShirtData where
  white_men_cost : ℝ
  black_men_cost : ℝ
  total_employees : ℕ
  total_spent : ℝ

/-- Theorem stating the cost of black men's t-shirts --/
theorem black_men_tshirt_cost (data : TShirtData) 
  (h1 : data.white_men_cost = 20)
  (h2 : data.total_employees = 40)
  (h3 : data.total_spent = 660)
  (h4 : ∃ (n : ℕ), n * 4 = data.total_employees) :
  data.black_men_cost = 18 := by
  sorry

end black_men_tshirt_cost_l2147_214731


namespace hyperbola_equation_l2147_214784

/-- A hyperbola with given asymptotes and passing through a specific point -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- The x-coordinate of the point the hyperbola passes through -/
  point_x : ℝ
  /-- The y-coordinate of the point the hyperbola passes through -/
  point_y : ℝ

/-- The equation of the hyperbola satisfies the given conditions -/
theorem hyperbola_equation (h : Hyperbola) 
  (h_asymptote : h.asymptote_slope = 1/2)
  (h_point : h.point_x = 4 ∧ h.point_y = Real.sqrt 2) :
  ∃ (f : ℝ → ℝ → Prop), 
    (∀ x y, f x y ↔ x^2/8 - y^2/2 = 1) ∧ 
    (f h.point_x h.point_y) ∧
    (∀ x, f x (h.asymptote_slope * x) ∨ f x (-h.asymptote_slope * x)) :=
by sorry

end hyperbola_equation_l2147_214784


namespace intersection_M_N_l2147_214777

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 + 1}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 1}

theorem intersection_M_N : M ∩ N = {(0, 1), (1, 2)} := by
  sorry

end intersection_M_N_l2147_214777


namespace point_coordinates_l2147_214704

/-- Given a point P in the Cartesian coordinate system, prove its coordinates. -/
theorem point_coordinates :
  ∀ m : ℝ,
  let P : ℝ × ℝ := (-m - 1, 2 * m + 1)
  (P.1 < 0 ∧ P.2 > 0) →  -- P is in the second quadrant
  P.2 = 5 →              -- Distance from M to x-axis is 5
  P = (-3, 5) :=         -- Coordinates of P are (-3, 5)
by
  sorry


end point_coordinates_l2147_214704


namespace max_value_sqrt_quadratic_l2147_214742

theorem max_value_sqrt_quadratic :
  ∃ (max : ℝ), max = 9/2 ∧
  ∀ a : ℝ, -6 ≤ a ∧ a ≤ 3 →
    Real.sqrt ((3 - a) * (a + 6)) ≤ max :=
by sorry

end max_value_sqrt_quadratic_l2147_214742


namespace coin_problem_l2147_214746

theorem coin_problem (x y : ℕ) : 
  x + y = 12 →
  5 * x + 10 * y = 90 →
  x = 6 ∧ y = 6 :=
by sorry

end coin_problem_l2147_214746


namespace coles_return_speed_l2147_214759

/-- Calculates the average speed of the return journey given the conditions of Cole's trip -/
theorem coles_return_speed (total_time : Real) (outbound_time : Real) (outbound_speed : Real) :
  total_time = 2 ∧ outbound_time = 72 / 60 ∧ outbound_speed = 70 →
  (2 * outbound_speed * outbound_time) / (total_time - outbound_time) = 105 := by
  sorry

#check coles_return_speed

end coles_return_speed_l2147_214759


namespace women_average_age_l2147_214762

theorem women_average_age 
  (n : ℕ) 
  (A : ℝ) 
  (age1 age2 : ℕ) 
  (h1 : n = 8) 
  (h2 : age1 = 20) 
  (h3 : age2 = 22) 
  (h4 : (n * A - (age1 + age2 : ℝ) + (W1 + W2)) / n = A + 2) :
  (W1 + W2) / 2 = 29 :=
by sorry

end women_average_age_l2147_214762


namespace fair_walking_distance_l2147_214727

theorem fair_walking_distance (total_distance : ℝ) (short_segment : ℝ) 
  (h1 : total_distance = 0.75)
  (h2 : short_segment = 0.08)
  (h3 : ∃ x : ℝ, total_distance = 2 * x + short_segment) :
  ∃ x : ℝ, x = 0.335 ∧ total_distance = 2 * x + short_segment :=
sorry

end fair_walking_distance_l2147_214727


namespace angle_measure_in_acute_triangle_l2147_214766

theorem angle_measure_in_acute_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π ∧
  b = 2 * a * Real.sin B →
  A = π/6 := by
sorry

end angle_measure_in_acute_triangle_l2147_214766


namespace ball_count_theorem_l2147_214734

theorem ball_count_theorem (total : ℕ) (red_freq black_freq : ℚ) : 
  total = 120 ∧ 
  red_freq = 15/100 ∧ 
  black_freq = 45/100 → 
  total - (red_freq * total).floor - (black_freq * total).floor = 48 := by
  sorry

end ball_count_theorem_l2147_214734


namespace unique_square_divisible_by_three_in_range_l2147_214714

theorem unique_square_divisible_by_three_in_range : 
  ∃! x : ℕ, 
    (∃ n : ℕ, x = n * n) ∧ 
    (∃ k : ℕ, x = 3 * k) ∧ 
    90 < x ∧ x < 150 := by
  sorry

end unique_square_divisible_by_three_in_range_l2147_214714


namespace scientific_notation_8350_l2147_214718

theorem scientific_notation_8350 : 
  8350 = 8.35 * (10 : ℝ)^3 := by sorry

end scientific_notation_8350_l2147_214718


namespace abcd_power_2018_l2147_214721

theorem abcd_power_2018 (a b c d : ℝ) 
  (ha : (5 : ℝ) ^ a = 4)
  (hb : (4 : ℝ) ^ b = 3)
  (hc : (3 : ℝ) ^ c = 2)
  (hd : (2 : ℝ) ^ d = 5) :
  (a * b * c * d) ^ 2018 = 1 := by
  sorry

end abcd_power_2018_l2147_214721


namespace range_of_M_l2147_214747

theorem range_of_M (a θ : ℝ) (ha : a ≠ 0) :
  let M := (a^2 - a * Real.sin θ + 1) / (a^2 - a * Real.cos θ + 1)
  (4 - Real.sqrt 7) / 3 ≤ M ∧ M ≤ (4 + Real.sqrt 7) / 3 := by
  sorry

end range_of_M_l2147_214747


namespace max_distance_between_circles_l2147_214769

/-- The maximum distance between a point on circle1 and a point on circle2 -/
theorem max_distance_between_circles (M N : ℝ × ℝ) : 
  (∃ x y, M = (x, y) ∧ (x - 3/2)^2 + y^2 = 23/4) →
  (∃ x y, N = (x, y) ∧ (x + 5)^2 + y^2 = 1) →
  (∀ M' N', 
    (∃ x y, M' = (x, y) ∧ (x - 3/2)^2 + y^2 = 23/4) →
    (∃ x y, N' = (x, y) ∧ (x + 5)^2 + y^2 = 1) →
    Real.sqrt ((M'.1 - N'.1)^2 + (M'.2 - N'.2)^2) ≤ Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)) →
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = (15 + Real.sqrt 23) / 2 := by
sorry

end max_distance_between_circles_l2147_214769


namespace hugh_initial_candy_l2147_214763

/-- The amount of candy Hugh had initially -/
def hugh_candy : ℕ := sorry

/-- The amount of candy Tommy had initially -/
def tommy_candy : ℕ := 6

/-- The amount of candy Melany had initially -/
def melany_candy : ℕ := 7

/-- The amount of candy each person had after sharing equally -/
def shared_candy : ℕ := 7

/-- The number of people sharing the candy -/
def num_people : ℕ := 3

theorem hugh_initial_candy :
  hugh_candy = 8 :=
by
  sorry

end hugh_initial_candy_l2147_214763


namespace a_plus_b_value_l2147_214719

theorem a_plus_b_value (a b : ℝ) (h1 : a^2 = 4) (h2 : b^2 = 9) (h3 : a * b < 0) :
  a + b = 1 ∨ a + b = -1 := by
sorry

end a_plus_b_value_l2147_214719


namespace marts_income_percentage_l2147_214751

theorem marts_income_percentage (juan tim mart : ℝ) : 
  tim = 0.6 * juan →
  mart = 0.9599999999999999 * juan →
  (mart - tim) / tim * 100 = 60 :=
by
  sorry

end marts_income_percentage_l2147_214751


namespace circle_tangency_l2147_214707

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

theorem circle_tangency (m : ℝ) : 
  externally_tangent (m, 0) (0, 2) (|m|) 1 → m = 3/2 ∨ m = -3/2 := by
  sorry

#check circle_tangency

end circle_tangency_l2147_214707


namespace motorboat_speed_calculation_l2147_214796

/-- The flood flow speed in kilometers per hour -/
def flood_speed : ℝ := 10

/-- The downstream distance in kilometers -/
def downstream_distance : ℝ := 2

/-- The upstream distance in kilometers -/
def upstream_distance : ℝ := 1.2

/-- The maximum speed of the motorboat in still water in kilometers per hour -/
def motorboat_speed : ℝ := 40

theorem motorboat_speed_calculation :
  (downstream_distance / (motorboat_speed + flood_speed) = 
   upstream_distance / (motorboat_speed - flood_speed)) ∧
  motorboat_speed = 40 := by sorry

end motorboat_speed_calculation_l2147_214796


namespace sector_area_l2147_214745

/-- The area of a sector with radius 10 cm and central angle 120° is (100π/3) cm² -/
theorem sector_area (r : ℝ) (θ : ℝ) : 
  r = 10 → θ = 2 * π / 3 → (1/2) * r^2 * θ = (100 * π) / 3 := by
  sorry

end sector_area_l2147_214745


namespace quadratic_functions_theorem_l2147_214793

/-- A quadratic function -/
def QuadraticFunction := ℝ → ℝ

/-- Condition that a function is quadratic -/
def IsQuadratic (f : QuadraticFunction) : Prop := 
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The x-coordinate of the vertex of a quadratic function -/
def VertexX (f : QuadraticFunction) : ℝ := sorry

/-- The x-intercepts of a function -/
def XIntercepts (f : QuadraticFunction) : Set ℝ := sorry

theorem quadratic_functions_theorem 
  (f g : QuadraticFunction)
  (hf : IsQuadratic f)
  (hg : IsQuadratic g)
  (h_relation : ∀ x, g x = -f (75 - x))
  (h_vertex : VertexX f ∈ XIntercepts g)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h_intercepts : {x₁, x₂, x₃, x₄} ⊆ XIntercepts f ∪ XIntercepts g)
  (h_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄)
  (h_diff : x₃ - x₂ = 120) :
  x₄ - x₁ = 360 + 240 * Real.sqrt 2 := by
  sorry

end quadratic_functions_theorem_l2147_214793


namespace tangent_line_slope_l2147_214726

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the tangent line
def tangent_line (a : ℝ) (x : ℝ) : ℝ := a*x + 16

theorem tangent_line_slope (a : ℝ) :
  (∃ x₀ : ℝ, f x₀ = tangent_line a x₀ ∧
    ∀ x : ℝ, x ≠ x₀ → f x ≠ tangent_line a x) →
  a = 9 :=
sorry

end tangent_line_slope_l2147_214726


namespace greatest_root_of_f_l2147_214781

noncomputable def f (x : ℝ) : ℝ := 16 * x^4 - 8 * x^3 + 9 * x^2 - 3 * x + 1

theorem greatest_root_of_f :
  ∃ (r : ℝ), r = 0.5 ∧ f r = 0 ∧ ∀ (x : ℝ), f x = 0 → x ≤ r :=
sorry

end greatest_root_of_f_l2147_214781


namespace ufo_convention_attendees_l2147_214755

theorem ufo_convention_attendees :
  ∀ (total male female : ℕ),
    total = 120 →
    male = female + 4 →
    total = male + female →
    male = 62 := by
  sorry

end ufo_convention_attendees_l2147_214755


namespace bug_probability_after_six_steps_l2147_214723

/-- Represents a vertex of the tetrahedron -/
inductive Vertex : Type
| A : Vertex
| B : Vertex
| C : Vertex
| D : Vertex

/-- The probability of the bug being at a given vertex after n steps -/
def prob_at_vertex (v : Vertex) (n : ℕ) : ℚ :=
  sorry

/-- The probability of the bug choosing a non-opposite vertex -/
def prob_non_opposite : ℚ := 1/2

/-- The probability of the bug choosing the opposite vertex -/
def prob_opposite : ℚ := 1/6

/-- The edge length of the tetrahedron -/
def edge_length : ℝ := 1

theorem bug_probability_after_six_steps :
  prob_at_vertex Vertex.A 6 = 53/324 := by
  sorry

end bug_probability_after_six_steps_l2147_214723


namespace sqrt_product_equality_l2147_214783

theorem sqrt_product_equality : Real.sqrt 121 * Real.sqrt 49 * Real.sqrt 11 = 77 * Real.sqrt 11 := by
  sorry

end sqrt_product_equality_l2147_214783


namespace opposite_expressions_l2147_214750

theorem opposite_expressions (x : ℝ) : (x + 1) + (3 * x - 5) = 0 ↔ x = 1 := by
  sorry

end opposite_expressions_l2147_214750


namespace total_population_l2147_214710

def population_problem (springfield_population greenville_population : ℕ) : Prop :=
  springfield_population = 482653 ∧
  greenville_population = springfield_population - 119666 ∧
  springfield_population + greenville_population = 845640

theorem total_population :
  ∃ (springfield_population greenville_population : ℕ),
    population_problem springfield_population greenville_population :=
by
  sorry

end total_population_l2147_214710


namespace equation_solution_l2147_214756

theorem equation_solution (x : ℝ) : 3 - 1 / (1 - x) = 2 * (1 / (1 - x)) → x = 0 := by
  sorry

end equation_solution_l2147_214756


namespace johnson_family_seating_theorem_l2147_214780

def johnson_family_seating (n m : ℕ) : ℕ :=
  Nat.factorial (n + m) - (Nat.factorial n * Nat.factorial m)

theorem johnson_family_seating_theorem :
  johnson_family_seating 5 4 = 360000 := by
  sorry

end johnson_family_seating_theorem_l2147_214780


namespace circle_radius_from_area_l2147_214724

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 64 * Real.pi) : 
  A = Real.pi * r^2 → r = 8 := by
sorry

end circle_radius_from_area_l2147_214724


namespace strawberry_calculation_l2147_214798

theorem strawberry_calculation (initial : ℝ) (sold : ℝ) (given_away : ℝ) (eaten : ℝ) 
  (h1 : initial = 120.5)
  (h2 : sold = 8.25)
  (h3 : given_away = 33.5)
  (h4 : eaten = 4.3) :
  initial - sold - given_away - eaten = 74.45 := by
  sorry

end strawberry_calculation_l2147_214798


namespace complex_fraction_equality_l2147_214732

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x + y) / (x - y) + (x - y) / (x + y) = 2) : 
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 2 := by
  sorry

end complex_fraction_equality_l2147_214732


namespace exponent_division_l2147_214790

theorem exponent_division (a : ℝ) : 2 * a^3 / a = 2 * a^2 := by
  sorry

end exponent_division_l2147_214790


namespace cake_and_icing_sum_l2147_214701

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the cake piece -/
structure CakePiece where
  top : List Point3D
  height : ℝ

/-- Calculates the volume of the cake piece -/
def cakeVolume (piece : CakePiece) : ℝ :=
  sorry

/-- Calculates the area of icing on the cake piece -/
def icingArea (piece : CakePiece) : ℝ :=
  sorry

/-- The main theorem -/
theorem cake_and_icing_sum (R P N : Point3D) (piece : CakePiece) :
  R.x = 0 ∧ R.y = 0 ∧ R.z = 3 ∧
  P.x = 3 ∧ P.y = 0 ∧ P.z = 3 ∧
  N.x = 2 ∧ N.y = 0 ∧ N.z = 3 ∧
  piece.top = [R, N, P] ∧
  piece.height = 3 →
  cakeVolume piece + icingArea piece = 13 := by
  sorry

end cake_and_icing_sum_l2147_214701


namespace salary_savings_percentage_l2147_214735

/-- Represents the percentage of salary saved -/
def P : ℝ := by sorry

theorem salary_savings_percentage :
  let S : ℝ := 20000  -- Monthly salary in Rs.
  let increase_factor : ℝ := 1.1  -- 10% increase in expenses
  let new_savings : ℝ := 200  -- New monthly savings in Rs.
  S - increase_factor * (S - P / 100 * S) = new_savings →
  P = 10 := by sorry

end salary_savings_percentage_l2147_214735


namespace cos_squared_pi_twelfth_plus_one_l2147_214712

theorem cos_squared_pi_twelfth_plus_one :
  2 * (Real.cos (π / 12))^2 + 1 = 2 + Real.sqrt 3 / 2 := by
  sorry

end cos_squared_pi_twelfth_plus_one_l2147_214712


namespace fraction_power_simplification_l2147_214706

theorem fraction_power_simplification :
  9 * (1 / 7)^4 = 9 / 2401 :=
by sorry

end fraction_power_simplification_l2147_214706


namespace inner_segments_sum_l2147_214708

theorem inner_segments_sum (perimeter_quadrilaterals perimeter_triangles perimeter_ABC : ℝ) 
  (h1 : perimeter_quadrilaterals = 25)
  (h2 : perimeter_triangles = 20)
  (h3 : perimeter_ABC = 19) :
  let total_perimeter := perimeter_quadrilaterals + perimeter_triangles
  let inner_segments := total_perimeter - perimeter_ABC
  inner_segments / 2 = 13 := by sorry

end inner_segments_sum_l2147_214708


namespace orthocenter_tangents_collinear_l2147_214741

/-- Representation of a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Representation of a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Definition of an acute-angled triangle -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Definition of the orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point := sorry

/-- Definition of a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Definition of a tangent line to a circle -/
def isTangent (p : Point) (c : Circle) : Prop := sorry

/-- Definition of collinearity -/
def areCollinear (p1 p2 p3 : Point) : Prop := sorry

/-- Main theorem -/
theorem orthocenter_tangents_collinear 
  (t : Triangle) 
  (h_acute : isAcuteAngled t) 
  (H : Point) 
  (h_ortho : H = orthocenter t) 
  (c : Circle) 
  (h_circle : c.center = Point.mk ((t.B.x + t.C.x) / 2) ((t.B.y + t.C.y) / 2) ∧ 
              c.radius = (((t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2)^(1/2)) / 2) 
  (P Q : Point) 
  (h_tangent_P : isTangent P c ∧ (∃ k : ℝ, P = Point.mk (t.A.x + k * (P.x - t.A.x)) (t.A.y + k * (P.y - t.A.y))))
  (h_tangent_Q : isTangent Q c ∧ (∃ k : ℝ, Q = Point.mk (t.A.x + k * (Q.x - t.A.x)) (t.A.y + k * (Q.y - t.A.y))))
  : areCollinear P H Q := 
sorry

end orthocenter_tangents_collinear_l2147_214741


namespace sum_of_a_and_b_l2147_214744

theorem sum_of_a_and_b (a b : ℝ) : 
  |a - 1/2| + |b + 5| = 0 → a + b = -9/2 := by
sorry

end sum_of_a_and_b_l2147_214744


namespace consecutive_non_prime_non_prime_power_l2147_214773

/-- For any positive integer n, there exists a positive integer k such that 
    for all i in {1, ..., n}, k + i is neither prime nor a prime power. -/
theorem consecutive_non_prime_non_prime_power (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ¬(Nat.Prime (k + i)) ∧ ¬(∃ p m : ℕ, Nat.Prime p ∧ 1 < m ∧ k + i = p^m) :=
sorry

end consecutive_non_prime_non_prime_power_l2147_214773


namespace arithmetic_sequence_count_l2147_214749

theorem arithmetic_sequence_count (a₁ : ℝ) (aₙ : ℝ) (d : ℝ) (n : ℕ) :
  a₁ = 2.5 ∧ aₙ = 68.5 ∧ d = 6 →
  aₙ = a₁ + (n - 1) * d →
  n = 12 := by
  sorry

end arithmetic_sequence_count_l2147_214749


namespace union_of_A_and_B_l2147_214787

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- State the theorem
theorem union_of_A_and_B : 
  A ∪ B = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end union_of_A_and_B_l2147_214787


namespace cos_angle_through_point_l2147_214786

/-- Given an angle α whose initial side is the positive x-axis and whose terminal side
    passes through the point (4, -3), prove that cos(α) = 4/5 -/
theorem cos_angle_through_point :
  ∀ α : ℝ,
  (∃ (x y : ℝ), x = 4 ∧ y = -3 ∧ 
    (Real.cos α * x - Real.sin α * y = x) ∧
    (Real.sin α * x + Real.cos α * y = y)) →
  Real.cos α = 4/5 := by
  sorry

end cos_angle_through_point_l2147_214786


namespace job_selection_probability_l2147_214772

theorem job_selection_probability (jamie_prob tom_prob : ℚ) 
  (h1 : jamie_prob = 2 / 3)
  (h2 : tom_prob = 5 / 7) :
  jamie_prob * tom_prob = 10 / 21 := by
sorry

end job_selection_probability_l2147_214772


namespace sixty_one_invalid_l2147_214709

/-- Represents the seat numbers of selected students -/
def selected_seats : List Nat := [5, 16, 27, 38, 49]

/-- The number of selected students -/
def num_selected : Nat := 5

/-- Checks if the given number can be the total number of students in the class -/
def is_valid_class_size (x : Nat) : Prop :=
  ∃ k, x = k * (num_selected - 1) + selected_seats.head!

/-- Theorem stating that 61 cannot be the number of students in the class -/
theorem sixty_one_invalid : ¬ is_valid_class_size 61 := by
  sorry


end sixty_one_invalid_l2147_214709


namespace mod_eight_congruence_l2147_214703

theorem mod_eight_congruence (m : ℕ) : 
  12^7 % 8 = m → 0 ≤ m → m < 8 → m = 0 := by
  sorry

end mod_eight_congruence_l2147_214703


namespace multiplicative_inverse_800_mod_7801_l2147_214711

theorem multiplicative_inverse_800_mod_7801 
  (h1 : 28^2 + 195^2 = 197^2) -- Pythagorean triple condition
  : ∃ n : ℕ, n < 7801 ∧ (800 * n) % 7801 = 1 :=
by
  -- The proof would go here
  sorry

end multiplicative_inverse_800_mod_7801_l2147_214711
