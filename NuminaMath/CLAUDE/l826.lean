import Mathlib

namespace parallel_vectors_imply_x_value_l826_82645

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (k * a.1 = b.1 ∧ k * a.2 = b.2)

/-- Theorem: If (1, 2) is parallel to (x, -4), then x = -2 -/
theorem parallel_vectors_imply_x_value :
  ∀ x : ℝ, parallel (1, 2) (x, -4) → x = -2 := by
  sorry

end parallel_vectors_imply_x_value_l826_82645


namespace final_cow_count_l826_82633

def cow_count (initial : ℕ) (died : ℕ) (sold : ℕ) (increase : ℕ) (bought : ℕ) (gift : ℕ) : ℕ :=
  initial - died - sold + increase + bought + gift

theorem final_cow_count :
  cow_count 39 25 6 24 43 8 = 83 := by
  sorry

end final_cow_count_l826_82633


namespace complex_power_evaluation_l826_82673

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_evaluation :
  3 * i ^ 44 - 2 * i ^ 333 = 3 - 2 * i :=
by sorry

end complex_power_evaluation_l826_82673


namespace min_value_quadratic_l826_82603

theorem min_value_quadratic (x y : ℝ) :
  y = 3 * x^2 + 6 * x + 9 →
  ∀ z : ℝ, y ≥ 6 ∧ ∃ w : ℝ, 3 * w^2 + 6 * w + 9 = 6 :=
by sorry

end min_value_quadratic_l826_82603


namespace expression_simplification_l826_82615

theorem expression_simplification (x : ℝ) : 
  3 * x - 5 * (2 + x) + 6 * (2 - x) - 7 * (2 + 3 * x) = -29 * x - 12 := by
  sorry

end expression_simplification_l826_82615


namespace range_of_f_l826_82639

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = { y | y ≥ 2 } := by
  sorry

end range_of_f_l826_82639


namespace three_fifths_equivalence_l826_82657

/-- Proves the equivalence of various representations of 3/5 -/
theorem three_fifths_equivalence :
  (3 : ℚ) / 5 = 12 / 20 ∧
  (3 : ℚ) / 5 = (10 : ℚ) / (50 / 3) ∧
  (3 : ℚ) / 5 = 60 / 100 ∧
  (3 : ℚ) / 5 = 0.60 ∧
  (3 : ℚ) / 5 = 60 / 100 := by
  sorry

#check three_fifths_equivalence

end three_fifths_equivalence_l826_82657


namespace students_in_same_group_l826_82636

/-- The number of interest groups -/
def num_groups : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 2

/-- The probability of a student joining any specific group -/
def prob_join_group : ℚ := 1 / num_groups

/-- The probability of both students being in the same group -/
def prob_same_group : ℚ := 1 / num_groups

theorem students_in_same_group : 
  prob_same_group = 1 / num_groups :=
sorry

end students_in_same_group_l826_82636


namespace rectangle_perimeter_l826_82671

theorem rectangle_perimeter (a b : ℝ) (h1 : a * b = 24) (h2 : a^2 + b^2 = 11^2) : 
  2 * (a + b) = 26 := by
sorry

end rectangle_perimeter_l826_82671


namespace common_factor_proof_l826_82608

def p (x : ℝ) := x^2 - 2*x - 3
def q (x : ℝ) := x^2 - 6*x + 9
def common_factor (x : ℝ) := x - 3

theorem common_factor_proof :
  ∀ x : ℝ, (∃ k₁ k₂ : ℝ, p x = common_factor x * k₁ ∧ q x = common_factor x * k₂) :=
sorry

end common_factor_proof_l826_82608


namespace mystery_book_shelves_l826_82667

theorem mystery_book_shelves (books_per_shelf : ℕ) (picture_book_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 4 →
  picture_book_shelves = 3 →
  total_books = 32 →
  (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf = 5 :=
by sorry

end mystery_book_shelves_l826_82667


namespace expression_one_proof_l826_82674

theorem expression_one_proof : 1 + (-2) + |(-2) - 3| - 5 = -1 := by
  sorry

end expression_one_proof_l826_82674


namespace sphere_radii_difference_l826_82689

theorem sphere_radii_difference (r₁ r₂ : ℝ) 
  (h_surface : 4 * π * (r₁^2 - r₂^2) = 48 * π) 
  (h_circumference : 2 * π * (r₁ + r₂) = 12 * π) : 
  |r₁ - r₂| = 2 := by
sorry

end sphere_radii_difference_l826_82689


namespace min_blue_eyes_and_water_bottle_l826_82649

theorem min_blue_eyes_and_water_bottle 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (water_bottle : ℕ) 
  (h1 : total_students = 35) 
  (h2 : blue_eyes = 18) 
  (h3 : water_bottle = 25) : 
  ∃ (both : ℕ), both ≥ 8 ∧ 
    both ≤ blue_eyes ∧ 
    both ≤ water_bottle ∧ 
    (∀ (x : ℕ), x < both → 
      x > blue_eyes - (total_students - water_bottle) ∨ 
      x > water_bottle - (total_students - blue_eyes)) :=
by sorry

end min_blue_eyes_and_water_bottle_l826_82649


namespace extraneous_root_implies_m_value_l826_82656

/-- Given a fractional equation (x - 3) / (x - 1) = m / (x - 1),
    if x = 1 is an extraneous root, then m = -2 -/
theorem extraneous_root_implies_m_value :
  ∀ (x m : ℝ), 
    (x - 3) / (x - 1) = m / (x - 1) →
    (1 : ℝ) ≠ 1 →  -- This represents that x = 1 is an extraneous root
    m = -2 := by
  sorry


end extraneous_root_implies_m_value_l826_82656


namespace distance_from_point_to_x_axis_l826_82661

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distance_to_x_axis (x y : ℝ) : ℝ :=
  |y|

theorem distance_from_point_to_x_axis :
  distance_to_x_axis 3 (-4) = 4 := by
  sorry

end distance_from_point_to_x_axis_l826_82661


namespace overlap_area_is_one_l826_82601

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by its vertices -/
structure Square where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- A triangle defined by its vertices -/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Calculate the area of overlap between a square and a triangle -/
def areaOfOverlap (s : Square) (t : Triangle) : ℝ := sorry

/-- The main theorem stating that the area of overlap is 1 square unit -/
theorem overlap_area_is_one :
  let s := Square.mk
    (Point.mk 0 0)
    (Point.mk 0 2)
    (Point.mk 2 2)
    (Point.mk 2 0)
  let t := Triangle.mk
    (Point.mk 2 2)
    (Point.mk 0 1)
    (Point.mk 1 0)
  areaOfOverlap s t = 1 := by sorry

end overlap_area_is_one_l826_82601


namespace complex_modulus_one_l826_82697

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : 
  Complex.abs z = 1 := by sorry

end complex_modulus_one_l826_82697


namespace simplify_expression_l826_82666

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  (20 * x^2) * (5 * x) * (1 / (2 * x)^2) * (2 * x)^2 = 100 * x^3 := by
  sorry

end simplify_expression_l826_82666


namespace square_of_105_l826_82617

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by sorry

end square_of_105_l826_82617


namespace coprime_iff_no_common_prime_factor_l826_82695

theorem coprime_iff_no_common_prime_factor (a b : ℕ) : 
  Nat.gcd a b = 1 ↔ ¬ ∃ (p : ℕ), Nat.Prime p ∧ p ∣ a ∧ p ∣ b := by
  sorry

end coprime_iff_no_common_prime_factor_l826_82695


namespace mod_difference_of_powers_l826_82642

theorem mod_difference_of_powers (n : ℕ) : 45^1537 - 25^1537 ≡ 4 [MOD 8] := by
  sorry

end mod_difference_of_powers_l826_82642


namespace chord_equation_through_bisection_point_l826_82625

/-- Given a parabola y² = 6x and a chord passing through point P(4, 1) that is bisected at P,
    prove that the equation of the line l on which this chord lies is 3x - y - 11 = 0. -/
theorem chord_equation_through_bisection_point (x y : ℝ) :
  (∀ x y, y^2 = 6*x) →  -- Parabola equation
  (∃ x₁ y₁ x₂ y₂ : ℝ,   -- Existence of two points on the parabola
    y₁^2 = 6*x₁ ∧ y₂^2 = 6*x₂ ∧
    (4 = (x₁ + x₂) / 2) ∧ (1 = (y₁ + y₂) / 2)) →  -- P(4,1) is midpoint
  (3*x - y - 11 = 0) :=  -- Equation of the line
by sorry

end chord_equation_through_bisection_point_l826_82625


namespace five_digit_division_l826_82630

/-- A five-digit number -/
def FiveDigitNumber (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A four-digit number -/
def FourDigitNumber (m : ℕ) : Prop :=
  1000 ≤ m ∧ m ≤ 9999

/-- m is formed by removing the middle digit of n -/
def MiddleDigitRemoved (n m : ℕ) : Prop :=
  FiveDigitNumber n ∧ FourDigitNumber m ∧
  ∃ (a b c d e : ℕ), n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
                     m = a * 1000 + b * 100 + d * 10 + e

theorem five_digit_division (n m : ℕ) :
  FiveDigitNumber n → MiddleDigitRemoved n m →
  (∃ k : ℕ, n = k * m) ↔ ∃ a : ℕ, 10 ≤ a ∧ a ≤ 99 ∧ n = a * 1000 := by
  sorry

end five_digit_division_l826_82630


namespace victors_hourly_rate_l826_82611

theorem victors_hourly_rate (hours_worked : ℕ) (total_earned : ℕ) 
  (h1 : hours_worked = 10) 
  (h2 : total_earned = 60) : 
  total_earned / hours_worked = 6 := by
  sorry

end victors_hourly_rate_l826_82611


namespace max_value_of_f_in_interval_l826_82694

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_value_of_f_in_interval :
  ∃ (m : ℝ), m = 2 ∧ 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x ≤ m) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f x = m) :=
by sorry

end max_value_of_f_in_interval_l826_82694


namespace four_digit_perfect_cubes_divisible_by_16_l826_82685

theorem four_digit_perfect_cubes_divisible_by_16 :
  (∃! (count : ℕ), ∃ (S : Finset ℕ),
    S.card = count ∧
    (∀ n ∈ S, 1000 ≤ n ∧ n ≤ 9999) ∧
    (∀ n ∈ S, ∃ m : ℕ, n = m^3) ∧
    (∀ n ∈ S, n % 16 = 0) ∧
    (∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m^3) ∧ n % 16 = 0 → n ∈ S)) ∧
  count = 3 :=
sorry

end four_digit_perfect_cubes_divisible_by_16_l826_82685


namespace impossible_all_positive_4x4_impossible_all_positive_8x8_l826_82604

/-- Represents a grid of signs -/
def Grid (n : Nat) := Fin n → Fin n → Bool

/-- Represents a line (row, column, or diagonal) in the grid -/
inductive Line (n : Nat)
| Row : Fin n → Line n
| Col : Fin n → Line n
| Diag : Bool → Line n

/-- Flips the signs along a given line in the grid -/
def flipLine (n : Nat) (g : Grid n) (l : Line n) : Grid n :=
  sorry

/-- Checks if all signs in the grid are positive -/
def allPositive (n : Nat) (g : Grid n) : Prop :=
  ∀ i j, g i j = true

/-- Initial configuration for the 8x8 grid with one negative sign -/
def initialConfig : Grid 8 :=
  sorry

/-- Theorem for the 4x4 grid -/
theorem impossible_all_positive_4x4 (g : Grid 4) :
  ¬∃ (flips : List (Line 4)), allPositive 4 (flips.foldl (flipLine 4) g) :=
  sorry

/-- Theorem for the 8x8 grid -/
theorem impossible_all_positive_8x8 :
  ¬∃ (flips : List (Line 8)), allPositive 8 (flips.foldl (flipLine 8) initialConfig) :=
  sorry

end impossible_all_positive_4x4_impossible_all_positive_8x8_l826_82604


namespace airport_distance_l826_82699

-- Define the problem parameters
def initial_speed : ℝ := 45
def speed_increase : ℝ := 20
def late_time : ℝ := 0.75  -- 45 minutes in hours
def early_time : ℝ := 0.25  -- 15 minutes in hours

-- Define the theorem
theorem airport_distance : ∃ (d : ℝ), d = 241.875 ∧ 
  ∃ (t : ℝ), 
    d = initial_speed * (t + late_time) ∧
    d - initial_speed = (initial_speed + speed_increase) * (t - (1 + early_time)) :=
by
  sorry


end airport_distance_l826_82699


namespace max_value_f_on_interval_l826_82682

def f (x : ℝ) := x^3 - 3*x^2 + 2

theorem max_value_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1 : ℝ) 1 ∧ 
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f x ≤ f c) ∧
  f c = 2 := by
sorry

end max_value_f_on_interval_l826_82682


namespace sqrt_simplification_l826_82644

theorem sqrt_simplification :
  Real.sqrt 32 - Real.sqrt 18 + Real.sqrt 8 = 3 * Real.sqrt 2 := by
  sorry

end sqrt_simplification_l826_82644


namespace purely_imaginary_complex_number_l826_82607

theorem purely_imaginary_complex_number (i : ℂ) (a : ℝ) : 
  i * i = -1 → 
  (∃ (k : ℝ), (1 + a * i) / (2 - i) = k * i) → 
  a = 2 :=
sorry

end purely_imaginary_complex_number_l826_82607


namespace collinear_vectors_l826_82638

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-2, -1]
def c : Fin 2 → ℝ := ![1, 2]

def is_collinear (v w : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, v 0 * w 1 = t * v 1 * w 0

theorem collinear_vectors (k : ℝ) :
  is_collinear (fun i => a i + k * b i) c ↔ k = -1/3 := by
  sorry

end collinear_vectors_l826_82638


namespace enclosed_area_theorem_l826_82679

/-- The area enclosed by a curve composed of 9 congruent circular arcs, each with length π/3,
    centered at the vertices of a regular hexagon with side length 3 -/
def enclosed_area (arc_length : Real) (num_arcs : Nat) (hexagon_side : Real) : Real :=
  sorry

/-- The theorem stating the enclosed area for the given conditions -/
theorem enclosed_area_theorem :
  enclosed_area (π/3) 9 3 = (27 * Real.sqrt 3) / 2 + (3 * π) / 8 := by
  sorry

end enclosed_area_theorem_l826_82679


namespace max_area_2014_l826_82681

/-- A polygon drawn on a grid with sides following grid lines -/
structure GridPolygon where
  perimeter : ℕ
  sides_follow_grid : Bool

/-- The maximum area of a grid polygon given its perimeter -/
def max_area (p : GridPolygon) : ℕ :=
  (p.perimeter / 4)^2 - if p.perimeter % 4 == 2 then 1/4 else 0

/-- Theorem stating the maximum area of a grid polygon with perimeter 2014 -/
theorem max_area_2014 :
  ∀ (p : GridPolygon), p.perimeter = 2014 → p.sides_follow_grid → max_area p = 253512 := by
  sorry


end max_area_2014_l826_82681


namespace even_composite_ratio_l826_82634

def first_five_even_composites : List Nat := [4, 6, 8, 10, 12]
def next_five_even_composites : List Nat := [14, 16, 18, 20, 22]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem even_composite_ratio :
  (product_of_list first_five_even_composites) / 
  (product_of_list next_five_even_composites) = 1 / 42 := by
  sorry

end even_composite_ratio_l826_82634


namespace fraction_addition_simplification_l826_82614

theorem fraction_addition_simplification : (2 : ℚ) / 5 + (3 : ℚ) / 15 = (3 : ℚ) / 5 := by
  sorry

end fraction_addition_simplification_l826_82614


namespace necessary_not_sufficient_condition_l826_82646

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, x^2 - x < 0 → -1 < x ∧ x < 1) ∧
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ ¬(x^2 - x < 0)) := by
  sorry

end necessary_not_sufficient_condition_l826_82646


namespace min_value_theorem_l826_82632

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (a + b) * (a + c) = 4) : 
  2 * a + b + c ≥ 4 ∧ (2 * a + b + c = 4 ↔ b = c) := by
  sorry

end min_value_theorem_l826_82632


namespace continuous_additive_function_is_linear_l826_82691

-- Define the property of the function
def SatisfiesAdditiveProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

-- State the theorem
theorem continuous_additive_function_is_linear
  (f : ℝ → ℝ)
  (hf_cont : Continuous f)
  (hf_add : SatisfiesAdditiveProperty f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end continuous_additive_function_is_linear_l826_82691


namespace arithmetic_sequence_2017th_term_l826_82619

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum of the first n terms
  sum_property : ∀ n : ℕ, S n = n * (a 1 + a n) / 2
  arithmetic_property : ∀ n m : ℕ, a (n + m) - a n = m * (a 2 - a 1)

/-- Theorem stating the property of the 2017th term of the arithmetic sequence -/
theorem arithmetic_sequence_2017th_term 
  (seq : ArithmeticSequence)
  (h1 : seq.a 1 = -2017)
  (h2 : seq.S 2007 / 2007 - seq.S 2005 / 2005 = 2) :
  seq.a 2017 = 2015 :=
sorry

end arithmetic_sequence_2017th_term_l826_82619


namespace inequalities_count_l826_82616

theorem inequalities_count (a c : ℝ) (h : a * c < 0) :
  ∃! n : ℕ, n = (Bool.toNat (a / c < 0) +
                 Bool.toNat (a * c^2 < 0) +
                 Bool.toNat (a^2 * c < 0) +
                 Bool.toNat (c^3 * a < 0) +
                 Bool.toNat (c * a^3 < 0)) ∧ n = 3 := by
  sorry

end inequalities_count_l826_82616


namespace floyd_jumps_exist_l826_82690

def sum_of_decimal_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_decimal_digits (n / 10)

def floyd_sequence : ℕ → ℕ
  | 0 => 90
  | n + 1 => 2 * (10^(n + 2)) - 28

theorem floyd_jumps_exist :
  ∃ (a : ℕ → ℕ), (∀ n > 0, a n ≤ 2 * a (n - 1)) ∧
                 (∀ i j, i ≠ j → sum_of_decimal_digits (a i) ≠ sum_of_decimal_digits (a j)) :=
by
  sorry


end floyd_jumps_exist_l826_82690


namespace milo_running_distance_l826_82602

def cory_speed : ℝ := 12

theorem milo_running_distance
  (h1 : cory_speed = 12)
  (h2 : ∃ milo_skateboard_speed : ℝ, cory_speed = 2 * milo_skateboard_speed)
  (h3 : ∃ milo_running_speed : ℝ, milo_skateboard_speed = 2 * milo_running_speed)
  : ∃ distance : ℝ, distance = 2 * milo_running_speed ∧ distance = 6 :=
by
  sorry

end milo_running_distance_l826_82602


namespace airport_exchange_rate_l826_82692

theorem airport_exchange_rate (euros : ℝ) (official_rate : ℝ) (airport_rate_factor : ℝ) :
  euros = 70 →
  official_rate = 5 →
  airport_rate_factor = 5 / 7 →
  (euros / official_rate) * airport_rate_factor = 10 := by
  sorry

end airport_exchange_rate_l826_82692


namespace greatest_three_digit_base7_divisible_by_7_l826_82688

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 7 number --/
def isThreeDigitBase7 (n : ℕ) : Prop := sorry

/-- The greatest 3-digit base 7 number --/
def greatestThreeDigitBase7 : ℕ := 666

theorem greatest_three_digit_base7_divisible_by_7 :
  isThreeDigitBase7 greatestThreeDigitBase7 ∧
  base7ToDecimal greatestThreeDigitBase7 % 7 = 0 ∧
  ∀ n : ℕ, isThreeDigitBase7 n ∧ base7ToDecimal n % 7 = 0 →
    base7ToDecimal n ≤ base7ToDecimal greatestThreeDigitBase7 :=
sorry

end greatest_three_digit_base7_divisible_by_7_l826_82688


namespace complex_product_magnitude_l826_82622

theorem complex_product_magnitude (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 3 * Complex.I →
  x = 6 * Real.sqrt 6 :=
by sorry

end complex_product_magnitude_l826_82622


namespace compound_propositions_true_l826_82626

-- Define proposition P
def P : Prop := ∀ x y : ℝ, x > y → -x > -y

-- Define proposition Q
def Q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Theorem to prove
theorem compound_propositions_true : (¬P ∨ ¬Q) ∧ ((¬P) ∨ Q) := by
  sorry

end compound_propositions_true_l826_82626


namespace trajectory_equation_constant_distance_fixed_point_l826_82665

/-- The trajectory of point P given the conditions -/
def trajectory (x y : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ -2 ∧ x^2 / 4 + y^2 = 1

/-- The line l intersecting the trajectory -/
def line_l (k m x y : ℝ) : Prop :=
  y = k * x + m

/-- Points M and N are on both the trajectory and line l -/
def intersection_points (x₁ y₁ x₂ y₂ k m : ℝ) : Prop :=
  trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
  line_l k m x₁ y₁ ∧ line_l k m x₂ y₂ ∧
  (x₁, y₁) ≠ (x₂, y₂)

/-- OM is perpendicular to ON -/
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

/-- The slopes of BM and BN satisfy the given condition -/
def slope_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₁ / (x₁ - 2)) * (y₂ / (x₂ - 2)) = -1/4

theorem trajectory_equation (x y : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (y / (x + 2)) * (y / (x - 2)) = -1/4 ↔ trajectory x y :=
sorry

theorem constant_distance (k m x₁ y₁ x₂ y₂ : ℝ)
  (h : intersection_points x₁ y₁ x₂ y₂ k m)
  (h_perp : perpendicular x₁ y₁ x₂ y₂) :
  |m| / Real.sqrt (1 + k^2) = 2 * Real.sqrt 5 / 5 :=
sorry

theorem fixed_point (k m x₁ y₁ x₂ y₂ : ℝ)
  (h : intersection_points x₁ y₁ x₂ y₂ k m)
  (h_slope : slope_condition x₁ y₁ x₂ y₂) :
  m = 0 :=
sorry

end trajectory_equation_constant_distance_fixed_point_l826_82665


namespace average_of_a_and_b_l826_82676

theorem average_of_a_and_b (a b : ℝ) : 
  (3 + 5 + 7 + a + b) / 5 = 15 → (a + b) / 2 = 30 := by
  sorry

end average_of_a_and_b_l826_82676


namespace recipe_flour_amount_l826_82610

def recipe_flour (total_sugar : ℕ) (added_sugar : ℕ) (flour_sugar_diff : ℕ) : ℕ :=
  (total_sugar - added_sugar) + flour_sugar_diff

theorem recipe_flour_amount : recipe_flour 6 4 7 = 9 := by
  sorry

end recipe_flour_amount_l826_82610


namespace distance_on_segment_triangle_inequality_l826_82648

/-- Custom distance function for points in 2D space -/
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₂ - x₁| + |y₂ - y₁|

/-- Theorem: If C is on the line segment AB, then AC + CB = AB -/
theorem distance_on_segment (x₁ y₁ x₂ y₂ x y : ℝ) 
  (h_x : min x₁ x₂ ≤ x ∧ x ≤ max x₁ x₂) 
  (h_y : min y₁ y₂ ≤ y ∧ y ≤ max y₁ y₂) :
  distance x₁ y₁ x y + distance x y x₂ y₂ = distance x₁ y₁ x₂ y₂ := by sorry

/-- Theorem: For any triangle ABC, AC + CB > AB -/
theorem triangle_inequality (x₁ y₁ x₂ y₂ x y : ℝ) :
  distance x₁ y₁ x y + distance x y x₂ y₂ ≥ distance x₁ y₁ x₂ y₂ := by sorry

end distance_on_segment_triangle_inequality_l826_82648


namespace smallest_solution_of_equation_l826_82612

theorem smallest_solution_of_equation : 
  ∃ x : ℝ, x^4 - 50*x^2 + 625 = 0 ∧ 
  (∀ y : ℝ, y^4 - 50*y^2 + 625 = 0 → x ≤ y) ∧ 
  x = -5 :=
sorry

end smallest_solution_of_equation_l826_82612


namespace eggs_equal_rice_cost_l826_82628

/-- The cost of a pound of rice in dollars -/
def rice_cost : ℝ := 0.36

/-- The cost of an egg in dollars -/
def egg_cost : ℝ := rice_cost

/-- The cost of half a liter of kerosene in dollars -/
def kerosene_cost : ℝ := 8 * egg_cost

/-- The number of eggs that cost the same as a pound of rice -/
def eggs_per_rice : ℕ := 1

theorem eggs_equal_rice_cost : eggs_per_rice = 1 := by
  sorry

end eggs_equal_rice_cost_l826_82628


namespace point_in_quadrants_I_and_II_l826_82609

-- Define the quadrants
def QuadrantI (x y : ℝ) : Prop := x > 0 ∧ y > 0
def QuadrantII (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the inequalities
def Inequality1 (x y : ℝ) : Prop := y > -3 * x
def Inequality2 (x y : ℝ) : Prop := y > x + 2

-- Theorem statement
theorem point_in_quadrants_I_and_II (x y : ℝ) :
  Inequality1 x y ∧ Inequality2 x y → QuadrantI x y ∨ QuadrantII x y :=
by sorry

end point_in_quadrants_I_and_II_l826_82609


namespace max_area_at_midline_l826_82650

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a line parallel to AC
def ParallelLine (t : Triangle) (M N : ℝ × ℝ) : Prop :=
  -- Add appropriate condition for parallel lines
  sorry

-- Define the rectangle MNPQ
structure Rectangle (t : Triangle) :=
  (M N P Q : ℝ × ℝ)
  (parallel : ParallelLine t M N)

-- Define the area of a rectangle
def area (r : Rectangle t) : ℝ :=
  sorry

-- Define the midline of a triangle
def Midline (t : Triangle) (M N : ℝ × ℝ) : Prop :=
  -- Add appropriate condition for midline
  sorry

-- Theorem statement
theorem max_area_at_midline (t : Triangle) :
  ∀ (r : Rectangle t), 
    Midline t r.M r.N → 
    ∀ (r' : Rectangle t), area r ≥ area r' :=
sorry

end max_area_at_midline_l826_82650


namespace rachel_picked_four_apples_l826_82647

/-- The number of apples Rachel picked from her tree -/
def apples_picked (initial_apples remaining_apples : ℕ) : ℕ :=
  initial_apples - remaining_apples

/-- Theorem: Rachel picked 4 apples -/
theorem rachel_picked_four_apples :
  apples_picked 7 3 = 4 := by
  sorry

end rachel_picked_four_apples_l826_82647


namespace cube_of_m_equals_64_l826_82680

theorem cube_of_m_equals_64 (m : ℕ) (h : 3^m = 81) : m^3 = 64 := by
  sorry

end cube_of_m_equals_64_l826_82680


namespace hours_to_minutes_l826_82684

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Define the number of hours Ava watched television
def hours_watched : ℕ := 4

-- Theorem to prove
theorem hours_to_minutes :
  hours_watched * minutes_per_hour = 240 := by
  sorry

end hours_to_minutes_l826_82684


namespace expression_evaluation_l826_82698

theorem expression_evaluation : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) + 1 / 4 = 37 / 60 := by
  sorry

end expression_evaluation_l826_82698


namespace min_additional_cells_for_symmetry_l826_82637

/-- Represents a cell in the rectangle --/
structure Cell where
  x : ℤ
  y : ℤ

/-- Represents the rectangle --/
structure Rectangle where
  width : ℕ
  height : ℕ
  center : Cell

/-- The set of initially colored cells --/
def initialColoredCells : Finset Cell := sorry

/-- Function to determine if two cells are symmetric about the center --/
def isSymmetric (c1 c2 : Cell) (center : Cell) : Prop := sorry

/-- Function to count the number of additional cells needed for symmetry --/
def additionalCellsForSymmetry (rect : Rectangle) (initial : Finset Cell) : ℕ := sorry

/-- Theorem stating that the minimum number of additional cells to color is 7 --/
theorem min_additional_cells_for_symmetry (rect : Rectangle) : 
  additionalCellsForSymmetry rect initialColoredCells = 7 := by sorry

end min_additional_cells_for_symmetry_l826_82637


namespace solve_cubic_equation_l826_82624

theorem solve_cubic_equation (x : ℝ) : 
  (x^3 * 6^3) / 432 = 864 → x = 12 := by
  sorry

end solve_cubic_equation_l826_82624


namespace equation_solution_l826_82643

theorem equation_solution : ∃ x : ℝ, (x^2 + 3*x + 4) / (x^2 - 3*x + 2) = x + 6 := by
  use 1
  -- Proof goes here
  sorry

#check equation_solution

end equation_solution_l826_82643


namespace laptop_sticker_price_l826_82655

theorem laptop_sticker_price :
  ∀ (sticker_price : ℝ),
    (0.8 * sticker_price - 120 = 0.7 * sticker_price - 18) →
    sticker_price = 1020 := by
  sorry

end laptop_sticker_price_l826_82655


namespace remainder_theorem_l826_82659

theorem remainder_theorem (n : ℕ) 
  (h1 : n % 22 = 7) 
  (h2 : n % 33 = 18) : 
  n % 66 = 51 := by
sorry

end remainder_theorem_l826_82659


namespace isosceles_triangle_from_rope_l826_82618

/-- Represents the sides of an isosceles triangle --/
structure IsoscelesTriangle where
  short : ℝ
  long : ℝ
  isIsosceles : long = 2 * short

/-- Checks if the given sides form a valid triangle --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem to be proved --/
theorem isosceles_triangle_from_rope (t : IsoscelesTriangle) :
  t.short + t.long + t.long = 20 →
  is_valid_triangle t.short t.long t.long →
  t.short = 4 ∧ t.long = 8 := by
  sorry

end isosceles_triangle_from_rope_l826_82618


namespace simplify_sqrt_product_l826_82631

theorem simplify_sqrt_product : 
  Real.sqrt (3 * 5) * Real.sqrt (5^4 * 3^3) = 45 * Real.sqrt 5 := by sorry

end simplify_sqrt_product_l826_82631


namespace ear_muffs_total_l826_82687

/-- The number of ear muffs bought before December -/
def before_december : ℕ := 1346

/-- The number of ear muffs bought during December -/
def during_december : ℕ := 6444

/-- The total number of ear muffs bought -/
def total_ear_muffs : ℕ := before_december + during_december

theorem ear_muffs_total : total_ear_muffs = 7790 := by
  sorry

end ear_muffs_total_l826_82687


namespace find_M_l826_82641

theorem find_M (x y z M : ℚ) 
  (sum_eq : x + y + z = 120)
  (x_dec : x - 10 = M)
  (y_inc : y + 10 = M)
  (z_mul : 10 * z = M) :
  M = 400 / 7 := by
  sorry

end find_M_l826_82641


namespace triangle_area_inequality_l826_82635

/-- Given two triangles with sides a₁ ≤ b₁ ≤ c and a₂ ≤ b₂ ≤ c, and equal smallest angles α,
    the area of a triangle with sides (a₁ + a₂), (b₁ + b₂), and (c + c) is no less than
    twice the sum of the areas of the original triangles. -/
theorem triangle_area_inequality
  (a₁ b₁ c a₂ b₂ : ℝ) (α : ℝ)
  (h₁ : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c)
  (h₂ : 0 < a₂ ∧ 0 < b₂)
  (h₃ : a₁ ≤ b₁ ∧ b₁ ≤ c)
  (h₄ : a₂ ≤ b₂ ∧ b₂ ≤ c)
  (h₅ : 0 < α ∧ α < π)
  (area₁ : ℝ := (1/2) * b₁ * c * Real.sin α)
  (area₂ : ℝ := (1/2) * b₂ * c * Real.sin α)
  (new_area : ℝ := (1/2) * (b₁ + b₂) * (2*c) * Real.sin (min α π/2)) :
  new_area ≥ 2 * (area₁ + area₂) :=
by sorry


end triangle_area_inequality_l826_82635


namespace pigeonhole_principle_interns_l826_82670

theorem pigeonhole_principle_interns (n : ℕ) (h : n > 0) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧
  (∃ (f : Fin n → ℕ), (∀ x, f x < n - 1) ∧ f i = k ∧ f j = k) :=
sorry

end pigeonhole_principle_interns_l826_82670


namespace fifteenth_term_ratio_l826_82678

-- Define the sums of arithmetic sequences
def U (n : ℕ) (c f : ℚ) : ℚ := n * (2 * c + (n - 1) * f) / 2
def V (n : ℕ) (g h : ℚ) : ℚ := n * (2 * g + (n - 1) * h) / 2

-- Define the ratio condition
def ratio_condition (n : ℕ) (c f g h : ℚ) : Prop :=
  U n c f / V n g h = (5 * n^2 + 3 * n + 2) / (3 * n^2 + 2 * n + 30)

-- Define the 15th term of each sequence
def term_15 (c f : ℚ) : ℚ := c + 14 * f

-- Theorem statement
theorem fifteenth_term_ratio 
  (c f g h : ℚ) 
  (h1 : ∀ (n : ℕ), n > 0 → ratio_condition n c f g h) :
  term_15 c f / term_15 g h = 125 / 99 := by
  sorry

end fifteenth_term_ratio_l826_82678


namespace specific_shiny_penny_last_probability_l826_82627

/-- The number of shiny pennies in the box -/
def shiny_pennies : ℕ := 4

/-- The number of dull pennies in the box -/
def dull_pennies : ℕ := 4

/-- The total number of pennies in the box -/
def total_pennies : ℕ := shiny_pennies + dull_pennies

/-- The probability of drawing a specific shiny penny last -/
def prob_specific_shiny_last : ℚ := 1 / 2

theorem specific_shiny_penny_last_probability :
  prob_specific_shiny_last = (Nat.choose (total_pennies - 1) (shiny_pennies - 1)) / (Nat.choose total_pennies shiny_pennies) :=
by sorry

end specific_shiny_penny_last_probability_l826_82627


namespace group_size_is_eight_l826_82662

/-- The number of people in a group, given certain weight conditions -/
def number_of_people : ℕ :=
  let weight_increase_per_person : ℕ := 5
  let weight_difference : ℕ := 75 - 35
  weight_difference / weight_increase_per_person

theorem group_size_is_eight :
  number_of_people = 8 :=
by
  -- Proof goes here
  sorry

#eval number_of_people  -- Should output 8

end group_size_is_eight_l826_82662


namespace decagon_adjacent_vertices_probability_l826_82696

theorem decagon_adjacent_vertices_probability :
  let n : ℕ := 10  -- number of vertices in a decagon
  let adjacent_pairs : ℕ := 2  -- number of adjacent vertices for any chosen vertex
  let total_choices : ℕ := n - 1  -- total number of choices for the second vertex
  (adjacent_pairs : ℚ) / total_choices = 2 / 9 := by
  sorry

end decagon_adjacent_vertices_probability_l826_82696


namespace series_solution_l826_82606

/-- The sum of the infinite geometric series with first term a and common ratio r -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The given series as a function of k -/
noncomputable def givenSeries (k : ℝ) : ℝ :=
  4 + geometricSum ((4 + k) / 5) (1 / 5)

theorem series_solution :
  ∃ k : ℝ, givenSeries k = 10 ∧ k = 16 := by sorry

end series_solution_l826_82606


namespace average_parking_cost_senior_student_l826_82686

/-- Calculates the average hourly parking cost for a senior citizen or student
    parking for 9 hours on a weekend, given the specified fee structure. -/
theorem average_parking_cost_senior_student (base_cost : ℝ) (additional_hourly_rate : ℝ)
  (weekend_surcharge : ℝ) (discount_rate : ℝ) (parking_duration : ℕ) :
  base_cost = 20 →
  additional_hourly_rate = 1.75 →
  weekend_surcharge = 5 →
  discount_rate = 0.1 →
  parking_duration = 9 →
  let total_cost := base_cost + (parking_duration - 2 : ℕ) * additional_hourly_rate + weekend_surcharge
  let discounted_cost := total_cost * (1 - discount_rate)
  let average_hourly_cost := discounted_cost / parking_duration
  average_hourly_cost = 3.725 := by
sorry

end average_parking_cost_senior_student_l826_82686


namespace min_cost_grass_seed_l826_82600

/-- Represents a bag of grass seed -/
structure GrassSeedBag where
  weight : Nat
  price : Rat
  deriving Repr

/-- Calculates the total weight of a list of bags -/
def totalWeight (bags : List GrassSeedBag) : Nat :=
  bags.foldl (fun acc bag => acc + bag.weight) 0

/-- Calculates the total cost of a list of bags -/
def totalCost (bags : List GrassSeedBag) : Rat :=
  bags.foldl (fun acc bag => acc + bag.price) 0

/-- Checks if a list of bags satisfies the purchase conditions -/
def isValidPurchase (bags : List GrassSeedBag) : Prop :=
  totalWeight bags ≥ 65 ∧
  totalWeight bags ≤ 80 ∧
  bags.length ≤ 5 ∧
  bags.length ≥ 4 ∧
  (∃ b ∈ bags, b.weight = 5) ∧
  (∃ b ∈ bags, b.weight = 10) ∧
  (∃ b ∈ bags, b.weight = 25) ∧
  (∃ b ∈ bags, b.weight = 40)

theorem min_cost_grass_seed :
  let bags := [
    GrassSeedBag.mk 5 (13.85),
    GrassSeedBag.mk 10 (20.43),
    GrassSeedBag.mk 25 (32.20),
    GrassSeedBag.mk 40 (54.30)
  ]
  ∀ purchase : List GrassSeedBag,
    isValidPurchase purchase →
    totalCost purchase ≥ 120.78 :=
by sorry

end min_cost_grass_seed_l826_82600


namespace duty_arrangements_count_l826_82658

def staff_count : ℕ := 7
def days_count : ℕ := 7
def restricted_days : ℕ := 2
def restricted_staff : ℕ := 2

theorem duty_arrangements_count : 
  (staff_count.factorial) / ((staff_count - days_count).factorial) *
  ((days_count - restricted_days).factorial) / 
  ((days_count - restricted_days - restricted_staff).factorial) = 2400 := by
  sorry

end duty_arrangements_count_l826_82658


namespace range_of_a_l826_82677

theorem range_of_a (a : ℝ) : (∃ x : ℝ, Real.exp (2 * x) - (a - 3) * Real.exp x + 4 - 3 * a > 0) → a ≤ 4 / 3 := by
  sorry

end range_of_a_l826_82677


namespace find_m_l826_82663

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the theorem
theorem find_m (z m : ℂ) (h1 : det z i m i = 1 - 2*I) (h2 : z.re = 0) : m = 2 := by
  sorry

end find_m_l826_82663


namespace alice_game_theorem_l826_82693

/-- The game state, representing the positions of the red and blue beads -/
structure GameState where
  red : ℚ
  blue : ℚ

/-- The move function that updates the game state -/
def move (r : ℚ) (state : GameState) (k : ℤ) (moveRed : Bool) : GameState :=
  if moveRed then
    { red := state.blue + r^k * (state.red - state.blue), blue := state.blue }
  else
    { red := state.red, blue := state.red + r^k * (state.blue - state.red) }

/-- Predicate to check if a rational number is of the form (b+1)/b for 1 ≤ b ≤ 1010 -/
def isValidR (r : ℚ) : Prop :=
  ∃ b : ℕ, 1 ≤ b ∧ b ≤ 1010 ∧ r = (b + 1) / b

/-- Main theorem statement -/
theorem alice_game_theorem (r : ℚ) (hr : r > 1) :
  (∃ (moves : List (ℤ × Bool)), moves.length ≤ 2021 ∧
    (moves.foldl (λ state (k, moveRed) => move r state k moveRed)
      { red := 0, blue := 1 }).red = 1) ↔
  isValidR r :=
sorry

end alice_game_theorem_l826_82693


namespace min_value_x_plus_81_over_x_l826_82621

theorem min_value_x_plus_81_over_x (x : ℝ) (h : x > 0) : 
  x + 81 / x ≥ 18 ∧ ∃ y > 0, y + 81 / y = 18 := by
  sorry

end min_value_x_plus_81_over_x_l826_82621


namespace book_organization_time_l826_82672

theorem book_organization_time (time_A time_B joint_time : ℝ) 
  (h1 : time_A = 6)
  (h2 : time_B = 8)
  (h3 : joint_time = 2)
  (h4 : joint_time * (1 / time_A + 1 / time_B) + 1 / time_A * remaining_time = 1) :
  remaining_time = 5/2 :=
by sorry

end book_organization_time_l826_82672


namespace tip_calculation_correct_l826_82620

/-- Calculates the tip amount for a family's salon visit -/
def calculate_tip (womens_haircut_price : ℚ) 
                  (childrens_haircut_price : ℚ) 
                  (teens_haircut_price : ℚ) 
                  (num_women : ℕ) 
                  (num_children : ℕ) 
                  (num_teens : ℕ) 
                  (hair_treatment_price : ℚ)
                  (tip_percentage : ℚ) : ℚ :=
  let total_cost := womens_haircut_price * num_women +
                    childrens_haircut_price * num_children +
                    teens_haircut_price * num_teens +
                    hair_treatment_price
  tip_percentage * total_cost

theorem tip_calculation_correct :
  calculate_tip 40 30 35 1 2 1 20 (1/4) = 155/4 :=
by sorry

end tip_calculation_correct_l826_82620


namespace square_sum_given_sum_and_product_l826_82683

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 3) (h2 : x * y = 1) : x^2 + y^2 = 7 := by
  sorry

end square_sum_given_sum_and_product_l826_82683


namespace library_wall_arrangement_l826_82651

/-- Proves that the maximum number of desk-bookcase pairs on a 15m wall leaves 3m of space --/
theorem library_wall_arrangement (wall_length : ℝ) (desk_length : ℝ) (bookcase_length : ℝ) 
  (space_between : ℝ) (h1 : wall_length = 15) (h2 : desk_length = 2) 
  (h3 : bookcase_length = 1.5) (h4 : space_between = 0.5) : 
  ∃ (n : ℕ) (leftover : ℝ), 
    n * (desk_length + bookcase_length + space_between) + leftover = wall_length ∧ 
    leftover = 3 ∧ 
    ∀ m : ℕ, m > n → m * (desk_length + bookcase_length + space_between) > wall_length := by
  sorry

end library_wall_arrangement_l826_82651


namespace expand_expression_l826_82669

theorem expand_expression (x : ℝ) : 25 * (3 * x - 4) = 75 * x - 100 := by
  sorry

end expand_expression_l826_82669


namespace inscribed_circle_radius_l826_82623

/-- An isosceles triangle with an inscribed circle -/
structure IsoscelesTriangleWithInscribedCircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The diameter of the circle is along the base of the triangle -/
  diameter_along_base : Bool

/-- Theorem: The radius of the inscribed circle in the given isosceles triangle is 120/13 -/
theorem inscribed_circle_radius 
  (triangle : IsoscelesTriangleWithInscribedCircle) 
  (h1 : triangle.base = 20) 
  (h2 : triangle.height = 24) 
  (h3 : triangle.diameter_along_base = true) : 
  triangle.radius = 120 / 13 := by
  sorry

end inscribed_circle_radius_l826_82623


namespace min_value_fraction_min_value_is_four_min_value_achieved_min_value_fraction_is_four_l826_82652

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 2) : 
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 → (x + y) / (x * y * z) ≤ (a + b) / (a * b * c) :=
by sorry

theorem min_value_is_four (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 2) : 
  (x + y) / (x * y * z) ≥ 4 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 2) : 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 ∧ (a + b) / (a * b * c) = 4 :=
by sorry

theorem min_value_fraction_is_four :
  ∃ m : ℝ, m = 4 ∧ 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 2 → (x + y) / (x * y * z) ≥ m) ∧
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 2 ∧ (a + b) / (a * b * c) = m) :=
by sorry

end min_value_fraction_min_value_is_four_min_value_achieved_min_value_fraction_is_four_l826_82652


namespace h_properties_l826_82654

-- Define the functions f, g, and h
def f : ℝ → ℝ := λ x => x

-- g is symmetric to f with respect to y = x
def g : ℝ → ℝ := λ x => x

def h : ℝ → ℝ := λ x => g (1 - |x|)

-- Theorem statement
theorem h_properties :
  (∀ x, h x = h (-x)) ∧  -- h is an even function
  (∃ m, ∀ x, h x ≥ m ∧ ∃ x₀, h x₀ = m ∧ m = 0) -- The minimum value of h is 0
  := by sorry

end h_properties_l826_82654


namespace greatest_x_value_l826_82660

theorem greatest_x_value (x : ℝ) : 
  x ≠ 6 → x ≠ -4 → (x^2 - 3*x - 18) / (x - 6) = 2 / (x + 4) → 
  x ≤ -2 ∧ ∃ y : ℝ, y ≠ 6 ∧ y ≠ -4 ∧ (y^2 - 3*y - 18) / (y - 6) = 2 / (y + 4) ∧ y = -2 :=
sorry

end greatest_x_value_l826_82660


namespace divisibility_problem_l826_82668

theorem divisibility_problem : ∃ (a b : ℕ), 
  (7^3 ∣ a^2 + a*b + b^2) ∧ 
  ¬(7 ∣ a) ∧ 
  ¬(7 ∣ b) ∧
  a = 1 ∧ 
  b = 18 :=
by sorry

end divisibility_problem_l826_82668


namespace minimum_third_term_l826_82605

def SallySequence (a : ℕ → ℕ) : Prop :=
  (∀ n ≥ 3, a n = a (n - 1) + a (n - 2)) ∧
  (a 8 = 400)

theorem minimum_third_term (a : ℕ → ℕ) (h : SallySequence a) :
  ∃ (m : ℕ), (∀ (b : ℕ → ℕ), SallySequence b → a 3 ≤ b 3) ∧ (a 3 = m) ∧ (m = 35) := by
  sorry

end minimum_third_term_l826_82605


namespace exist_prime_sum_30_l826_82664

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- State the theorem
theorem exist_prime_sum_30 : ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ p + q = 30 := by
  sorry

end exist_prime_sum_30_l826_82664


namespace role_assignment_count_l826_82629

/-- The number of ways to assign roles in a play. -/
def assign_roles (num_men num_women : ℕ) : ℕ :=
  let male_role_assignments := num_men
  let female_role_assignments := num_women * (num_women - 1)
  let specific_role_assignment := 1
  let remaining_actors := (num_men - 1) + num_women
  let remaining_role_assignments := remaining_actors * (remaining_actors - 1)
  male_role_assignments * female_role_assignments * specific_role_assignment * remaining_role_assignments

/-- Theorem stating the number of ways to assign roles in the given scenario. -/
theorem role_assignment_count :
  assign_roles 6 7 = 27720 :=
by sorry

end role_assignment_count_l826_82629


namespace circle_symmetry_and_properties_l826_82653

-- Define the circle C1 and line l
def C1 (m : ℝ) (x y : ℝ) : Prop := (x + 1)^2 + (y - 3*m - 3)^2 = 4*m^2
def l (m : ℝ) (x y : ℝ) : Prop := y = x + m + 2

-- Define the circle C2
def C2 (m : ℝ) (x y : ℝ) : Prop := (x - 2*m - 1)^2 + (y - m - 1)^2 = 4*m^2

-- Define the line on which centers of C2 lie
def centerLine (x y : ℝ) : Prop := x - 2*y + 1 = 0

-- Define the common tangent line
def commonTangent (x y : ℝ) : Prop := y = -3/4 * x + 7/4

theorem circle_symmetry_and_properties 
  (m : ℝ) (h : m ≠ 0) :
  (∀ x y, C2 m x y ↔ 
    ∃ x' y', C1 m x' y' ∧ l m ((x + x') / 2) ((y + y') / 2)) ∧ 
  (∀ m x y, C2 m x y → centerLine x y) ∧
  (∀ m x y, C2 m x y → ∃ x₀ y₀, commonTangent x₀ y₀ ∧ 
    (x₀ - x)^2 + (y₀ - y)^2 = ((x - (2*m + 1))^2 + (y - (m + 1))^2) / 4) :=
sorry

end circle_symmetry_and_properties_l826_82653


namespace vector_sum_magnitude_l826_82675

/-- The ellipse equation -/
def on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The line equation -/
def on_line (x y : ℝ) : Prop := 4*x - 2*y - 3 = 0

/-- Symmetry about the line -/
def symmetric_about_line (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), on_line x₀ y₀ ∧ x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem vector_sum_magnitude (x₁ y₁ x₂ y₂ : ℝ) :
  on_ellipse x₁ y₁ → on_ellipse x₂ y₂ → symmetric_about_line x₁ y₁ x₂ y₂ →
  (x₁ + x₂)^2 + (y₁ + y₂)^2 = 5 :=
by sorry

end vector_sum_magnitude_l826_82675


namespace sam_fish_count_l826_82640

/-- Represents the number of fish a person has -/
structure FishCount where
  goldfish : ℕ
  guppies : ℕ
  angelfish : ℕ

def Lilly : FishCount :=
  { goldfish := 10, guppies := 15, angelfish := 0 }

def Rosy : FishCount :=
  { goldfish := 12, guppies := 8, angelfish := 5 }

def Sam : FishCount :=
  { goldfish := Rosy.goldfish - 3, guppies := 2 * Lilly.guppies, angelfish := 0 }

def guppiesTransferred : ℕ := Lilly.guppies / 2

def LillyAfterTransfer : FishCount :=
  { Lilly with guppies := Lilly.guppies - guppiesTransferred }

def SamAfterTransfer : FishCount :=
  { Sam with guppies := Sam.guppies + guppiesTransferred }

def totalFish (fc : FishCount) : ℕ :=
  fc.goldfish + fc.guppies + fc.angelfish

theorem sam_fish_count :
  totalFish SamAfterTransfer = 46 := by sorry

end sam_fish_count_l826_82640


namespace boys_count_in_class_l826_82613

theorem boys_count_in_class (total : ℕ) (boy_ratio girl_ratio : ℕ) (h1 : total = 49) (h2 : boy_ratio = 4) (h3 : girl_ratio = 3) :
  (total * boy_ratio) / (boy_ratio + girl_ratio) = 28 := by
  sorry

end boys_count_in_class_l826_82613
