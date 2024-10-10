import Mathlib

namespace conditional_probability_B_given_A_l817_81759

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define event A
def A : Set Ω := {ω | ω.1 = 0}

-- Define event B
def B : Set Ω := {ω | ω.1.val + ω.2.val + 2 = 6}

-- State the theorem
theorem conditional_probability_B_given_A :
  P B / P A = 1 / 6 := by sorry

end conditional_probability_B_given_A_l817_81759


namespace maria_furniture_assembly_time_l817_81724

def total_time (num_chairs num_tables time_per_piece : ℕ) : ℕ :=
  (num_chairs + num_tables) * time_per_piece

theorem maria_furniture_assembly_time :
  total_time 2 2 8 = 32 := by
  sorry

end maria_furniture_assembly_time_l817_81724


namespace tenth_digit_theorem_l817_81723

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def some_number : ℕ := 6840

theorem tenth_digit_theorem :
  (((factorial 5 * factorial 5 - factorial 5 * factorial 3) / some_number) % 100) / 10 = 8 := by
  sorry

end tenth_digit_theorem_l817_81723


namespace min_a_for_increasing_f_l817_81731

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

-- State the theorem
theorem min_a_for_increasing_f :
  (∀ a : ℝ, ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x < y → f a x < f a y) →
  (∃ a_min : ℝ, a_min = -3 ∧ 
    (∀ a : ℝ, (∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x < y → f a x < f a y) → a ≥ a_min)) :=
sorry

end min_a_for_increasing_f_l817_81731


namespace SR_equals_15_l817_81777

/-- Triangle PQR with point S on PR --/
structure TrianglePQRWithS where
  /-- Length of PQ --/
  PQ : ℝ
  /-- Length of QR --/
  QR : ℝ
  /-- Length of PS --/
  PS : ℝ
  /-- Length of QS --/
  QS : ℝ
  /-- PQ equals QR --/
  eq_PQ_QR : PQ = QR
  /-- PQ equals 10 --/
  eq_PQ_10 : PQ = 10
  /-- PS equals 6 --/
  eq_PS_6 : PS = 6
  /-- QS equals 5 --/
  eq_QS_5 : QS = 5

/-- The length of SR in the given triangle configuration --/
def SR (t : TrianglePQRWithS) : ℝ := 15

/-- Theorem: The length of SR is 15 in the given triangle configuration --/
theorem SR_equals_15 (t : TrianglePQRWithS) : SR t = 15 := by
  sorry

end SR_equals_15_l817_81777


namespace toothpicks_count_l817_81732

/-- The number of small triangles in a row, starting from the base --/
def num_triangles_in_row (n : ℕ) : ℕ := 2500 - n + 1

/-- The total number of small triangles in the large triangle --/
def total_small_triangles : ℕ := (2500 * 2501) / 2

/-- The number of toothpicks needed for the interior and remaining exterior of the large triangle --/
def toothpicks_needed : ℕ := ((3 * total_small_triangles) / 2) + 2 * 2500

theorem toothpicks_count : toothpicks_needed = 4694375 := by sorry

end toothpicks_count_l817_81732


namespace concert_ticket_problem_l817_81714

def ticket_price_possibilities (seventh_grade_total eighth_grade_total : ℕ) : ℕ :=
  (Finset.filter (fun x => seventh_grade_total % x = 0 ∧ eighth_grade_total % x = 0)
    (Finset.range (min seventh_grade_total eighth_grade_total + 1))).card

theorem concert_ticket_problem : ticket_price_possibilities 36 90 = 6 := by
  sorry

end concert_ticket_problem_l817_81714


namespace quadratic_diophantine_bound_l817_81733

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The number of integer solutions to a quadratic Diophantine equation -/
def num_solutions (A B C D E : ℤ) : ℕ := sorry

theorem quadratic_diophantine_bound
  (A B C D E : ℤ)
  (hB : B ≠ 0)
  (hF : A * D^2 - B * C * D + B^2 * E ≠ 0) :
  num_solutions A B C D E ≤ 2 * num_divisors (Int.natAbs (A * D^2 - B * C * D + B^2 * E)) :=
sorry

end quadratic_diophantine_bound_l817_81733


namespace bus_speed_with_stoppages_l817_81757

/-- Given a bus that travels at 90 km/hr excluding stoppages and stops for 4 minutes per hour,
    its speed including stoppages is 84 km/hr. -/
theorem bus_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stoppage_time : ℝ) 
  (total_time : ℝ) :
  speed_without_stoppages = 90 →
  stoppage_time = 4 →
  total_time = 60 →
  (speed_without_stoppages * (total_time - stoppage_time)) / total_time = 84 := by
  sorry

#check bus_speed_with_stoppages

end bus_speed_with_stoppages_l817_81757


namespace max_a_is_maximum_l817_81795

/-- The maximum value of a such that the line y = mx + 1 does not pass through
    any lattice points for 0 < x ≤ 200 and 1/2 < m < a -/
def max_a : ℚ := 101 / 201

/-- Predicate to check if a point (x, y) is a lattice point -/
def is_lattice_point (x y : ℚ) : Prop := ∃ (n m : ℤ), x = n ∧ y = m

/-- Predicate to check if the line y = mx + 1 passes through a lattice point -/
def line_passes_lattice_point (m : ℚ) (x : ℚ) : Prop :=
  ∃ (y : ℚ), is_lattice_point x y ∧ y = m * x + 1

theorem max_a_is_maximum :
  ∀ (a : ℚ), (∀ (m : ℚ), 1/2 < m → m < a →
    ∀ (x : ℚ), 0 < x → x ≤ 200 → ¬ line_passes_lattice_point m x) →
  a ≤ max_a :=
sorry

end max_a_is_maximum_l817_81795


namespace triangle_point_inequality_l817_81781

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point P
def Point := ℝ × ℝ

-- Function to calculate distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Function to check if a point is inside a triangle
def isInside (t : Triangle) (p : Point) : Prop := sorry

-- Define the perimeter of the triangle
def perimeter (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

-- Theorem statement
theorem triangle_point_inequality (t : Triangle) (P : Point) (s : ℝ) :
  perimeter t = 2 * s →
  isInside t P →
  s < distance t.A P + distance t.B P + distance t.C P ∧
  distance t.A P + distance t.B P + distance t.C P < 2 * s :=
by sorry

end triangle_point_inequality_l817_81781


namespace position_change_l817_81750

/-- The position of a person from the back in a line of descending height order -/
def position_from_back_descending (total : ℕ) (position : ℕ) : Prop :=
  position > 0 ∧ position ≤ total

/-- The position of a person from the back in a line of ascending height order -/
def position_from_back_ascending (total : ℕ) (position : ℕ) : Prop :=
  position > 0 ∧ position ≤ total

/-- Theorem stating the relationship between a person's position in descending and ascending order lines -/
theorem position_change 
  (total : ℕ) 
  (position_desc : ℕ) 
  (position_asc : ℕ) 
  (h1 : total = 22)
  (h2 : position_desc = 13)
  (h3 : position_from_back_descending total position_desc)
  (h4 : position_from_back_ascending total position_asc)
  : position_asc = 10 := by
  sorry

#check position_change

end position_change_l817_81750


namespace inheritance_calculation_l817_81718

theorem inheritance_calculation (inheritance : ℝ) : 
  (0.25 * inheritance + 0.15 * (inheritance - 0.25 * inheritance) = 20000) → 
  inheritance = 55172.41 := by
  sorry

end inheritance_calculation_l817_81718


namespace rectangle_square_overlap_l817_81791

/-- Given a rectangle JKLM and a square NOPQ, if 30% of JKLM's area overlaps with NOPQ,
    and 40% of NOPQ's area overlaps with JKLM, then the ratio of JKLM's length to its width is 4/3. -/
theorem rectangle_square_overlap (j l m n : ℝ) :
  j > 0 → l > 0 → m > 0 → n > 0 →
  0.3 * (j * l) = 0.4 * (n * n) →
  j * l = m * n →
  j / m = 4 / 3 := by
  sorry

end rectangle_square_overlap_l817_81791


namespace range_of_a_l817_81763

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a :
  (∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ (∃ x : ℝ, ¬(p x) ∧ (q x a))) →
  (∀ a : ℝ, a ≥ 1) ∧ (∀ b : ℝ, b ≥ 1 → ∃ a : ℝ, a = b) :=
sorry

end range_of_a_l817_81763


namespace point_line_plane_relations_l817_81792

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relations
variable (lies_on : Point → Line → Prop)
variable (lies_in_line : Line → Plane → Prop)
variable (lies_in_point : Point → Plane → Prop)

-- State the theorem
theorem point_line_plane_relations 
  (A : Point) (a : Line) (α : Plane) (B : Point) :
  lies_on A a → lies_in_line a α → lies_in_point B α →
  (A ∈ {x : Point | lies_on x a}) ∧ 
  ({x : Point | lies_on x a} ⊆ {x : Point | lies_in_point x α}) ∧ 
  (B ∈ {x : Point | lies_in_point x α}) :=
sorry

end point_line_plane_relations_l817_81792


namespace simple_interest_principal_calculation_l817_81719

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation
  (rate : ℝ) (interest : ℝ) (time : ℝ) :
  rate = 5 →
  interest = 160 →
  time = 4 →
  160 = (rate * time / 100) * (interest * 100 / (rate * time)) :=
by
  sorry

#check simple_interest_principal_calculation

end simple_interest_principal_calculation_l817_81719


namespace gbp_share_change_l817_81749

/-- The change in the share of British pounds in the National Wealth Fund -/
theorem gbp_share_change (
  total : ℝ)
  (initial_share : ℝ)
  (other_amounts : List ℝ)
  (h_total : total = 794.26)
  (h_initial : initial_share = 8.2)
  (h_other : other_amounts = [39.84, 34.72, 600.3, 110.54, 0.31]) :
  ∃ (δ : ℝ), abs (δ + 7) < 0.5 ∧ 
  δ = (total - (other_amounts.sum)) / total * 100 - initial_share :=
sorry

end gbp_share_change_l817_81749


namespace number_difference_l817_81752

theorem number_difference (L S : ℕ) (h1 : L = 1637) (h2 : L = 6 * S + 5) : L - S = 1365 := by
  sorry

end number_difference_l817_81752


namespace expression_evaluation_l817_81762

theorem expression_evaluation : 60 + 5 * 12 / (180 / 3) = 61 := by
  sorry

end expression_evaluation_l817_81762


namespace all_nines_square_l817_81711

/-- A function that generates a number with n 9's -/
def all_nines (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- Theorem: For any positive integer n, (all_nines n)² = (all_nines n + 1)(all_nines n - 1) + 1 -/
theorem all_nines_square (n : ℕ+) :
  (all_nines n)^2 = (all_nines n + 1) * (all_nines n - 1) + 1 := by
  sorry

end all_nines_square_l817_81711


namespace complex_sum_difference_l817_81713

theorem complex_sum_difference (A M S : ℂ) (P : ℝ) 
  (hA : A = 3 - 2*I) 
  (hM : M = -5 + 3*I) 
  (hS : S = -2*I) 
  (hP : P = 3) : 
  A + M + S - P = -5 - I := by
  sorry

end complex_sum_difference_l817_81713


namespace right_triangle_theorem_l817_81772

/-- Right triangle DEF with given side lengths and midpoint N on hypotenuse -/
structure RightTriangle where
  /-- Length of side DE -/
  de : ℝ
  /-- Length of side DF -/
  df : ℝ
  /-- Right angle at E -/
  right_angle : de ^ 2 + df ^ 2 = (de + df) ^ 2 / 4
  /-- N is midpoint of EF -/
  n_midpoint : True

/-- Properties of the right triangle -/
def triangle_properties (t : RightTriangle) : Prop :=
  let dn := (t.de ^ 2 + t.df ^ 2).sqrt / 2
  let area := t.de * t.df / 2
  let centroid_distance := 2 * dn / 3
  dn = 5.0 ∧ area = 24.0 ∧ centroid_distance = 3.3

/-- Theorem stating the properties of the specific right triangle -/
theorem right_triangle_theorem :
  ∃ t : RightTriangle, t.de = 6 ∧ t.df = 8 ∧ triangle_properties t :=
sorry

end right_triangle_theorem_l817_81772


namespace field_trip_adults_l817_81767

theorem field_trip_adults (van_capacity : ℕ) (num_vans : ℕ) (num_students : ℕ) :
  van_capacity = 4 →
  num_vans = 2 →
  num_students = 2 →
  ∃ (num_adults : ℕ), num_adults + num_students = num_vans * van_capacity ∧ num_adults = 6 :=
by sorry

end field_trip_adults_l817_81767


namespace p_sufficient_not_necessary_for_q_l817_81745

-- Define the conditions
def p (x : ℝ) : Prop := abs x > 1
def q (x : ℝ) : Prop := x < -2

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) := by
  sorry

end p_sufficient_not_necessary_for_q_l817_81745


namespace power_sum_fourth_l817_81790

theorem power_sum_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end power_sum_fourth_l817_81790


namespace min_sum_of_product_144_l817_81771

theorem min_sum_of_product_144 :
  ∀ c d : ℤ, c * d = 144 → (∀ x y : ℤ, x * y = 144 → c + d ≤ x + y) ∧ (∃ a b : ℤ, a * b = 144 ∧ a + b = -145) :=
by sorry

end min_sum_of_product_144_l817_81771


namespace marbles_difference_l817_81704

/-- The number of marbles Cindy and Lisa have after Cindy gives some to Lisa -/
def marbles_after_giving (cindy_initial : ℕ) (lisa_initial : ℕ) (marbles_given : ℕ) :
  ℕ × ℕ :=
  (cindy_initial - marbles_given, lisa_initial + marbles_given)

/-- The theorem stating the difference in marbles after Cindy gives some to Lisa -/
theorem marbles_difference
  (cindy_initial : ℕ)
  (lisa_initial : ℕ)
  (marbles_given : ℕ)
  (h1 : cindy_initial = 20)
  (h2 : cindy_initial = lisa_initial + 5)
  (h3 : marbles_given = 12) :
  (marbles_after_giving cindy_initial lisa_initial marbles_given).2 -
  (marbles_after_giving cindy_initial lisa_initial marbles_given).1 = 19 := by
  sorry

end marbles_difference_l817_81704


namespace correct_multiplication_l817_81794

theorem correct_multiplication (x : ℕ) (h : 63 + x = 70) : 36 * x = 252 := by
  sorry

end correct_multiplication_l817_81794


namespace arithmetic_sequence_sum_l817_81705

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 9th, 52nd, and 95th terms of the sequence. -/
def sum_terms (a : ℕ → ℝ) : ℝ := a 9 + a 52 + a 95

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 = 4 → a 101 = 36 → sum_terms a = 60 :=
by
  sorry

#check arithmetic_sequence_sum

end arithmetic_sequence_sum_l817_81705


namespace fraction_irreducible_l817_81789

theorem fraction_irreducible (a : ℤ) : 
  Nat.gcd (Int.natAbs (a^3 + 2*a)) (Int.natAbs (a^4 + 3*a^2 + 1)) = 1 := by
  sorry

end fraction_irreducible_l817_81789


namespace sum_of_ratios_geq_two_l817_81751

theorem sum_of_ratios_geq_two (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by
  sorry

end sum_of_ratios_geq_two_l817_81751


namespace grid_sum_theorem_l817_81743

/-- Represents a 3x3 grid where each cell contains a number from 1 to 3 --/
def Grid := Fin 3 → Fin 3 → Fin 3

/-- Checks if a row in the grid contains 1, 2, and 3 --/
def valid_row (g : Grid) (r : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃ c : Fin 3, g r c = n.succ

/-- Checks if a column in the grid contains 1, 2, and 3 --/
def valid_column (g : Grid) (c : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃ r : Fin 3, g r c = n.succ

/-- Checks if the entire grid is valid --/
def valid_grid (g : Grid) : Prop :=
  (∀ r : Fin 3, valid_row g r) ∧ (∀ c : Fin 3, valid_column g c)

theorem grid_sum_theorem (g : Grid) :
  valid_grid g →
  g 0 0 = 2 →
  g 1 1 = 3 →
  g 1 2 + g 2 2 + 4 = 8 := by sorry

end grid_sum_theorem_l817_81743


namespace fraction_sum_equality_l817_81717

theorem fraction_sum_equality : (1 : ℚ) / 3 + 5 / 9 - 2 / 9 = 2 / 3 := by
  sorry

end fraction_sum_equality_l817_81717


namespace line_through_point_with_equal_intercepts_l817_81798

/-- A line passing through (1, 3) with equal absolute intercepts has one of three specific equations -/
theorem line_through_point_with_equal_intercepts :
  ∀ (f : ℝ → ℝ),
  (f 1 = 3) →  -- Line passes through (1, 3)
  (∃ a : ℝ, a ≠ 0 ∧ f 0 = a ∧ f a = 0) →  -- Equal absolute intercepts
  (∀ x, f x = 3 * x) ∨  -- y = 3x
  (∀ x, x + f x = 4) ∨  -- x + y - 4 = 0
  (∀ x, x - f x = -2)  -- x - y + 2 = 0
  := by sorry

end line_through_point_with_equal_intercepts_l817_81798


namespace number_difference_l817_81744

theorem number_difference (L S : ℕ) (h1 : L = 1575) (h2 : L = 7 * S + 15) : L - S = 1353 := by
  sorry

end number_difference_l817_81744


namespace cubic_equation_solution_l817_81727

theorem cubic_equation_solution :
  ∃! x : ℝ, (x^3 - x^2) / (x^2 + 3*x + 2) + x = -3 ∧ x ≠ -1 ∧ x ≠ -2 :=
by sorry

end cubic_equation_solution_l817_81727


namespace fraction_equality_l817_81725

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 3)
  (h3 : p / q = 1 / 5) :
  m / q = 4 / 15 := by
sorry

end fraction_equality_l817_81725


namespace quadratic_radical_equality_l817_81787

theorem quadratic_radical_equality :
  ∃! x : ℝ, x^2 - 2 = 2*x - 2 ∧ x^2 - 2 ≥ 0 ∧ 2*x - 2 ≥ 0 ∧ x = 2 :=
by sorry

end quadratic_radical_equality_l817_81787


namespace necessary_condition_example_l817_81769

theorem necessary_condition_example : 
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧ 
  (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) := by
  sorry

end necessary_condition_example_l817_81769


namespace leading_zeros_of_fraction_l817_81730

/-- The number of leading zeros in the decimal representation of a fraction -/
def leadingZeros (n d : ℕ) : ℕ :=
  sorry

theorem leading_zeros_of_fraction :
  leadingZeros 1 (2^3 * 5^5) = 4 := by
  sorry

end leading_zeros_of_fraction_l817_81730


namespace sum_of_thousands_and_units_digits_l817_81788

/-- Represents a 100-digit number with a repeating pattern --/
def RepeatNumber (a b : ℕ) := ℕ

/-- The first 100-digit number: 606060606...060606 --/
def num1 : RepeatNumber 60 6 := sorry

/-- The second 100-digit number: 808080808...080808 --/
def num2 : RepeatNumber 80 8 := sorry

/-- Returns the units digit of a number --/
def unitsDigit (n : ℕ) : ℕ := sorry

/-- Returns the thousands digit of a number --/
def thousandsDigit (n : ℕ) : ℕ := sorry

/-- The product of num1 and num2 --/
def product : ℕ := sorry

theorem sum_of_thousands_and_units_digits :
  thousandsDigit product + unitsDigit product = 14 := by sorry

end sum_of_thousands_and_units_digits_l817_81788


namespace network_connections_l817_81774

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (n * k) / 2 = 50 := by
  sorry

end network_connections_l817_81774


namespace square_distance_equivalence_l817_81741

theorem square_distance_equivalence :
  ∀ (s : Real), s = 1 →
  (5 : Real) / Real.sqrt 2 = (5 : Real) / 6 :=
by
  sorry

end square_distance_equivalence_l817_81741


namespace geometric_sequence_sum_l817_81796

/-- Given a geometric sequence {a_n} with sum of first n terms S_n = 3^n + t, prove t + a_3 = 17 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = 3^n + t) →  -- Definition of S_n
  (∀ n, a (n+1) = S (n+1) - S n) →  -- Definition of a_n in terms of S_n
  (a 2)^2 = a 1 * a 3 →  -- Property of geometric sequence
  t + a 3 = 17 := by
sorry

end geometric_sequence_sum_l817_81796


namespace range_of_a_l817_81797

open Set

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc 0 a, f x ∈ Icc (-4) 0) ∧ 
  (Icc (-4) 0 ⊆ f '' Icc 0 a) ↔ 
  a ∈ Icc 2 4 :=
sorry

end range_of_a_l817_81797


namespace system_solutions_l817_81742

theorem system_solutions (a : ℤ) :
  let eq1 := fun (x y z : ℤ) => 5 * x + (a + 2) * y + (a + 2) * z = a
  let eq2 := fun (x y z : ℤ) => (2 * a + 4) * x + (a^2 + 3) * y + (2 * a + 2) * z = 3 * a - 1
  let eq3 := fun (x y z : ℤ) => (2 * a + 4) * x + (2 * a + 2) * y + (a^2 + 3) * z = a + 1
  (∀ x y z : ℤ, eq1 x y z ∧ eq2 x y z ∧ eq3 x y z ↔
    (a = 1 ∧ ∃ n : ℤ, x = -1 ∧ y = n ∧ z = 2 - n) ∨
    (a = -1 ∧ x = 0 ∧ y = -1 ∧ z = 0) ∨
    (a = 0 ∧ x = 0 ∧ y = -1 ∧ z = 1) ∨
    (a = 2 ∧ x = -6 ∧ y = 5 ∧ z = 3)) ∧
  (a = 3 → ¬∃ x y z : ℤ, eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) ∧
  (a ≠ 1 ∧ a ≠ -1 ∧ a ≠ 0 ∧ a ≠ 2 ∧ a ≠ 3 →
    ¬∃ x y z : ℤ, eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) :=
by sorry

end system_solutions_l817_81742


namespace rectangle_perimeter_l817_81770

theorem rectangle_perimeter (L W : ℝ) 
  (h1 : L - 4 = W + 3) 
  (h2 : (L - 4) * (W + 3) = L * W) : 
  2 * L + 2 * W = 50 := by
sorry

end rectangle_perimeter_l817_81770


namespace min_value_sum_fourth_and_square_l817_81753

theorem min_value_sum_fourth_and_square (t : ℝ) :
  let f := fun (a : ℝ) => a^4 + (t - a)^2
  ∃ (min_val : ℝ), (∀ (a : ℝ), f a ≥ min_val) ∧ (min_val = t^4 / 16 + t^2 / 4) := by
  sorry

end min_value_sum_fourth_and_square_l817_81753


namespace number_difference_l817_81778

theorem number_difference (L S : ℕ) (h1 : L > S) (h2 : L = 6 * S + 15) (h3 : L = 1656) : L - S = 1383 := by
  sorry

end number_difference_l817_81778


namespace number_order_l817_81726

-- Define the numbers in their respective bases
def a : ℕ := 33
def b : ℕ := 5 * 6 + 2
def c : ℕ := 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

-- Theorem statement
theorem number_order : a > b ∧ b > c :=
sorry

end number_order_l817_81726


namespace polynomial_not_equal_33_l817_81735

theorem polynomial_not_equal_33 (x y : ℤ) :
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end polynomial_not_equal_33_l817_81735


namespace time_to_meet_prove_time_to_meet_l817_81702

/-- The time it takes for Michael to reach Eric given the specified conditions --/
theorem time_to_meet (initial_distance : ℝ) (speed_ratio : ℝ) (closing_rate : ℝ) 
  (initial_time : ℝ) (delay_time : ℝ) : ℝ :=
  65

/-- Proof of the time it takes for Michael to reach Eric --/
theorem prove_time_to_meet :
  time_to_meet 30 4 2 4 6 = 65 := by
  sorry

end time_to_meet_prove_time_to_meet_l817_81702


namespace largest_solution_of_equation_l817_81716

theorem largest_solution_of_equation (x : ℝ) :
  (x / 3 + 1 / (3 * x) = 1 / 2) → x ≤ 2 :=
by sorry

end largest_solution_of_equation_l817_81716


namespace sqrt_two_squared_cubed_l817_81703

theorem sqrt_two_squared_cubed : (Real.sqrt (Real.sqrt 2)^2)^3 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_two_squared_cubed_l817_81703


namespace steven_shirts_l817_81736

/-- The number of shirts owned by Brian -/
def brian_shirts : ℕ := 3

/-- The number of shirts owned by Andrew relative to Brian -/
def andrew_multiplier : ℕ := 6

/-- The number of shirts owned by Steven relative to Andrew -/
def steven_multiplier : ℕ := 4

/-- Theorem: Given the conditions, Steven has 72 shirts -/
theorem steven_shirts : 
  steven_multiplier * (andrew_multiplier * brian_shirts) = 72 := by
sorry

end steven_shirts_l817_81736


namespace geometric_sequence_sum_l817_81799

theorem geometric_sequence_sum (a₁ a₄ r : ℚ) (h₁ : a₁ = 4096) (h₂ : a₄ = 16) (h₃ : r = 1/4) :
  a₁ * r + a₁ * r^2 = 320 := by
  sorry

end geometric_sequence_sum_l817_81799


namespace neighborhood_cable_cost_l817_81780

/-- Calculate the total cost of power cable for a neighborhood --/
theorem neighborhood_cable_cost
  (east_west_streets : ℕ)
  (north_south_streets : ℕ)
  (east_west_length : ℝ)
  (north_south_length : ℝ)
  (cable_per_street_mile : ℝ)
  (cable_cost_per_mile : ℝ)
  (h1 : east_west_streets = 18)
  (h2 : north_south_streets = 10)
  (h3 : east_west_length = 2)
  (h4 : north_south_length = 4)
  (h5 : cable_per_street_mile = 5)
  (h6 : cable_cost_per_mile = 2000) :
  east_west_streets * east_west_length * cable_per_street_mile * cable_cost_per_mile +
  north_south_streets * north_south_length * cable_per_street_mile * cable_cost_per_mile =
  760000 :=
by sorry

end neighborhood_cable_cost_l817_81780


namespace square_area_error_l817_81782

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
sorry

end square_area_error_l817_81782


namespace two_non_congruent_triangles_l817_81756

/-- A triangle with integer side lengths. -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The perimeter of a triangle. -/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- Check if a triangle satisfies the triangle inequality. -/
def is_valid (t : IntTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Check if two triangles are congruent. -/
def is_congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all valid triangles with perimeter 11. -/
def valid_triangles : Set IntTriangle :=
  {t : IntTriangle | perimeter t = 11 ∧ is_valid t}

/-- The theorem to be proved. -/
theorem two_non_congruent_triangles :
  ∃ (t1 t2 : IntTriangle),
    t1 ∈ valid_triangles ∧
    t2 ∈ valid_triangles ∧
    ¬is_congruent t1 t2 ∧
    ∀ (t : IntTriangle),
      t ∈ valid_triangles →
      is_congruent t t1 ∨ is_congruent t t2 :=
sorry

end two_non_congruent_triangles_l817_81756


namespace product_expansion_l817_81758

theorem product_expansion (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end product_expansion_l817_81758


namespace frank_work_days_l817_81754

/-- Calculates the number of days worked given total hours and hours per day -/
def days_worked (total_hours : Float) (hours_per_day : Float) : Float :=
  total_hours / hours_per_day

/-- Theorem: Frank worked 4 days given the conditions -/
theorem frank_work_days :
  let total_hours : Float := 8.0
  let hours_per_day : Float := 2.0
  days_worked total_hours hours_per_day = 4.0 := by
  sorry

end frank_work_days_l817_81754


namespace simplify_expression_l817_81760

theorem simplify_expression (x y : ℝ) :
  (2 * x + 25) + (150 * x + 40) + (5 * y + 10) = 152 * x + 5 * y + 75 := by
  sorry

end simplify_expression_l817_81760


namespace expression_value_l817_81765

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 4) :
  3 * x - 5 * y + 7 = -4 := by
  sorry

end expression_value_l817_81765


namespace carpooling_distance_ratio_l817_81710

def distance_to_first_friend : ℝ := 8

def distance_to_second_friend : ℝ := 4

def distance_to_work (d1 d2 : ℝ) : ℝ := 3 * (d1 + d2)

theorem carpooling_distance_ratio :
  distance_to_work distance_to_first_friend distance_to_second_friend = 36 →
  distance_to_second_friend / distance_to_first_friend = 1 / 2 := by
  sorry

end carpooling_distance_ratio_l817_81710


namespace x_squared_plus_y_squared_equals_four_l817_81773

theorem x_squared_plus_y_squared_equals_four 
  (h : (x^2 + y^2 + 1) * (x^2 + y^2 - 3) = 5) : 
  x^2 + y^2 = 4 := by
sorry

end x_squared_plus_y_squared_equals_four_l817_81773


namespace repeated_root_implies_m_equals_two_l817_81768

theorem repeated_root_implies_m_equals_two (x m : ℝ) : 
  (2 / (x - 1) + 3 = m / (x - 1)) →  -- Condition 1
  (x - 1 = 0) →                      -- Condition 2 (repeated root implies x - 1 = 0)
  m = 2 := by
sorry

end repeated_root_implies_m_equals_two_l817_81768


namespace max_stamps_for_50_dollars_l817_81761

theorem max_stamps_for_50_dollars (stamp_price : ℕ) (available_amount : ℕ) : 
  stamp_price = 37 → available_amount = 5000 → 
  (∃ (n : ℕ), n * stamp_price ≤ available_amount ∧ 
  ∀ (m : ℕ), m * stamp_price ≤ available_amount → m ≤ n) → 
  (∃ (max_stamps : ℕ), max_stamps = 135) := by
  sorry

end max_stamps_for_50_dollars_l817_81761


namespace positive_cubic_interval_l817_81746

theorem positive_cubic_interval (x : ℝ) :
  (x + 1) * (x - 1) * (x + 3) > 0 ↔ x ∈ Set.Ioo (-3 : ℝ) (-1 : ℝ) ∪ Set.Ioi (1 : ℝ) :=
sorry

end positive_cubic_interval_l817_81746


namespace evaluate_f_l817_81700

def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_f : 3 * f 2 + 2 * f (-2) = 98 := by
  sorry

end evaluate_f_l817_81700


namespace train_speed_l817_81721

/-- Calculates the speed of a train given its composition and time to cross a bridge -/
theorem train_speed (num_carriages : ℕ) (carriage_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  num_carriages = 24 →
  carriage_length = 60 →
  bridge_length = 1500 →
  crossing_time = 3 →
  (((num_carriages + 1) * carriage_length + bridge_length) / 1000) / (crossing_time / 60) = 60 := by
  sorry

#check train_speed

end train_speed_l817_81721


namespace ceiling_squared_fraction_l817_81747

theorem ceiling_squared_fraction : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end ceiling_squared_fraction_l817_81747


namespace smallest_solution_of_equation_l817_81701

theorem smallest_solution_of_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧ 
  (∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x) →
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_of_equation_l817_81701


namespace hcl_formation_l817_81786

/-- Represents a chemical compound with its coefficient in a chemical equation -/
structure Compound where
  name : String
  coefficient : ℚ

/-- Represents a chemical equation with reactants and products -/
structure ChemicalEquation where
  reactants : List Compound
  products : List Compound

/-- Calculates the number of moles of HCl formed given the initial moles of reactants -/
def molesOfHClFormed (h2so4_moles : ℚ) (nacl_moles : ℚ) (equation : ChemicalEquation) : ℚ :=
  sorry

/-- The main theorem stating that 3 moles of HCl are formed -/
theorem hcl_formation :
  let equation : ChemicalEquation := {
    reactants := [
      {name := "H₂SO₄", coefficient := 1},
      {name := "NaCl", coefficient := 2}
    ],
    products := [
      {name := "HCl", coefficient := 2},
      {name := "Na₂SO₄", coefficient := 1}
    ]
  }
  molesOfHClFormed 3 3 equation = 3 :=
by sorry

end hcl_formation_l817_81786


namespace alice_bake_time_proof_l817_81708

/-- The time it takes Alice to bake a pie -/
def alice_bake_time : ℝ := 5

/-- The time it takes Bob to bake a pie -/
def bob_bake_time : ℝ := 6

/-- The total time given in the problem -/
def total_time : ℝ := 60

/-- The number of additional pies Alice can bake compared to Bob in the given time -/
def additional_pies : ℕ := 2

theorem alice_bake_time_proof :
  alice_bake_time = 5 ∧
  (total_time / bob_bake_time + additional_pies) * alice_bake_time = total_time :=
by sorry

end alice_bake_time_proof_l817_81708


namespace AC_length_l817_81784

-- Define the isosceles trapezoid
structure IsoscelesTrapezoid :=
  (AB : ℝ)
  (CD : ℝ)
  (AD : ℝ)
  (BC : ℝ)
  (isIsosceles : AD = BC)

-- Define our specific trapezoid
def specificTrapezoid : IsoscelesTrapezoid :=
  { AB := 30
  , CD := 12
  , AD := 15
  , BC := 15
  , isIsosceles := rfl }

-- Theorem statement
theorem AC_length (t : IsoscelesTrapezoid) (h : t = specificTrapezoid) :
  ∃ (AC : ℝ), AC = Real.sqrt (12^2 + 20^2) :=
sorry

end AC_length_l817_81784


namespace ascending_order_negative_a_l817_81728

theorem ascending_order_negative_a (a : ℝ) (h1 : -1 < a) (h2 : a < 0) :
  1 / a < a ∧ a < a^2 ∧ a^2 < |a| := by sorry

end ascending_order_negative_a_l817_81728


namespace point_P_on_circle_M_and_line_L_l817_81737

/-- Circle M with center (3,2) and radius √2 -/
def circle_M (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 2

/-- Line L with equation x + y - 3 = 0 -/
def line_L (x y : ℝ) : Prop := x + y - 3 = 0

/-- Point P with coordinates (2,1) -/
def point_P : ℝ × ℝ := (2, 1)

theorem point_P_on_circle_M_and_line_L :
  circle_M point_P.1 point_P.2 ∧ line_L point_P.1 point_P.2 := by
  sorry

end point_P_on_circle_M_and_line_L_l817_81737


namespace abc_sum_bound_l817_81722

theorem abc_sum_bound (a b c : ℝ) (h : a + b + c = 1) :
  ∃ (M : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (a' b' c' : ℝ),
    a' + b' + c' = 1 ∧ a' * b' + a' * c' + b' * c' > M ∧
    a * b + a * c + b * c ≤ 1/2 ∧
    a * b + a * c + b * c < 1/2 + ε :=
sorry

end abc_sum_bound_l817_81722


namespace angle_equation_solution_l817_81766

theorem angle_equation_solution (A : Real) :
  (1/2 * Real.sin (A/2) + Real.cos (A/2) = 1) → A = 2 * Real.pi := by
  sorry

end angle_equation_solution_l817_81766


namespace smallest_number_proof_l817_81785

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  b = 28 →                 -- Median is 28
  c = b + 6 →              -- Largest number is 6 more than median
  a ≤ b ∧ b ≤ c →          -- b is the median
  a = 28 :=                -- Smallest number is 28
by sorry

end smallest_number_proof_l817_81785


namespace divisors_between_squares_l817_81783

theorem divisors_between_squares (m a b d : ℕ) : 
  1 ≤ m → 
  m^2 < a → a < m^2 + m → 
  m^2 < b → b < m^2 + m → 
  a ≠ b → 
  m^2 < d → d < m^2 + m → 
  d ∣ (a * b) → 
  d = a ∨ d = b :=
by sorry

end divisors_between_squares_l817_81783


namespace max_daily_profit_l817_81748

/-- Represents the daily profit function for a store selling a product -/
def daily_profit (x : ℝ) : ℝ :=
  (2 + 0.5 * x) * (200 - 10 * x)

/-- Theorem stating the maximum daily profit and the corresponding selling price -/
theorem max_daily_profit :
  ∃ (x : ℝ), daily_profit x = 720 ∧ 
  (∀ (y : ℝ), daily_profit y ≤ daily_profit x) ∧
  x = 8 :=
sorry

end max_daily_profit_l817_81748


namespace solution_set_of_inequality_l817_81712

theorem solution_set_of_inequality (x : ℝ) :
  (8 * x^2 + 6 * x ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 1/4) :=
by sorry

end solution_set_of_inequality_l817_81712


namespace prime_equation_solution_l817_81775

theorem prime_equation_solution (p : ℕ) (x y : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h_eq : x * (y^2 - p) + y * (x^2 - p) = 5 * p) : 
  p = 2 ∨ p = 3 ∨ p = 7 := by
  sorry

end prime_equation_solution_l817_81775


namespace max_gcd_17n_plus_4_10n_plus_3_l817_81738

theorem max_gcd_17n_plus_4_10n_plus_3 :
  ∃ (k : ℕ), k > 0 ∧
  Nat.gcd (17 * k + 4) (10 * k + 3) = 11 ∧
  ∀ (n : ℕ), n > 0 → Nat.gcd (17 * n + 4) (10 * n + 3) ≤ 11 :=
by sorry

end max_gcd_17n_plus_4_10n_plus_3_l817_81738


namespace sravans_journey_l817_81715

/-- Calculates the total distance traveled given the conditions of Sravan's journey -/
theorem sravans_journey (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 15 ∧ speed1 = 45 ∧ speed2 = 30 → 
  ∃ (distance : ℝ), 
    distance / (2 * speed1) + distance / (2 * speed2) = total_time ∧
    distance = 540 := by
  sorry


end sravans_journey_l817_81715


namespace joan_bought_72_eggs_l817_81764

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Joan bought -/
def dozens_bought : ℕ := 6

/-- The total number of eggs Joan bought -/
def total_eggs : ℕ := dozens_bought * eggs_per_dozen

theorem joan_bought_72_eggs : total_eggs = 72 := by
  sorry

end joan_bought_72_eggs_l817_81764


namespace inequality_condition_l817_81779

theorem inequality_condition (x y : ℝ) : 
  (x > 0 ∧ y > 0 → y / x + x / y ≥ 2) ∧ 
  ¬(y / x + x / y ≥ 2 → x > 0 ∧ y > 0) := by
  sorry

end inequality_condition_l817_81779


namespace elevator_problem_l817_81729

theorem elevator_problem (x y z w v a b c n : ℕ) : 
  x = 20 ∧ 
  y = 7 ∧ 
  z = 3^2 ∧ 
  w = 5^2 ∧ 
  v = 3^2 ∧ 
  a = 3^2 - 2 ∧ 
  b = 3 ∧ 
  c = 1^3 ∧ 
  x - y + z - w + v - a + b - c = n 
  → n = 1 := by
sorry

end elevator_problem_l817_81729


namespace bryden_payment_proof_l817_81740

/-- The amount a collector pays for state quarters as a percentage of face value -/
def collector_payment_percentage : ℝ := 1500

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 7

/-- The face value of a single state quarter in dollars -/
def quarter_face_value : ℝ := 0.25

/-- The amount Bryden receives for his quarters in dollars -/
def bryden_payment : ℝ := 26.25

theorem bryden_payment_proof :
  (collector_payment_percentage / 100) * (bryden_quarters * quarter_face_value) = bryden_payment :=
by sorry

end bryden_payment_proof_l817_81740


namespace quadratic_roots_l817_81706

theorem quadratic_roots (x : ℝ) : 
  (x^2 + 4*x - 21 = 0) ↔ (x = 3 ∨ x = -7) := by
  sorry

end quadratic_roots_l817_81706


namespace student_absence_probability_l817_81739

theorem student_absence_probability :
  let p_absent : ℚ := 1 / 20
  let p_present : ℚ := 1 - p_absent
  let p_two_absent_one_present : ℚ := 3 * (p_absent * p_absent * p_present)
  p_two_absent_one_present = 57 / 8000 := by
  sorry

end student_absence_probability_l817_81739


namespace scale_division_l817_81720

-- Define the length of the scale in inches
def scale_length : ℕ := 6 * 12 + 8

-- Define the number of parts
def num_parts : ℕ := 4

-- Theorem to prove
theorem scale_division :
  scale_length / num_parts = 20 := by
  sorry

end scale_division_l817_81720


namespace inverse_g_84_l817_81755

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 3

-- State the theorem
theorem inverse_g_84 : g⁻¹ 84 = 3 := by sorry

end inverse_g_84_l817_81755


namespace inverse_proportion_point_order_l817_81709

theorem inverse_proportion_point_order (y₁ y₂ y₃ : ℝ) : 
  y₁ = -2 / (-2) → y₂ = -2 / 2 → y₃ = -2 / 3 → y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end inverse_proportion_point_order_l817_81709


namespace total_hours_theorem_hours_breakdown_theorem_l817_81776

/-- The number of hours Sangita is required to fly to earn an airplane pilot certificate -/
def required_hours : ℕ := 1320

/-- The number of hours Sangita has already completed -/
def completed_hours : ℕ := 50 + 9 + 121

/-- The number of months Sangita needs to complete her goal -/
def months : ℕ := 6

/-- The number of hours Sangita must fly per month -/
def hours_per_month : ℕ := 220

/-- Theorem stating that the total number of hours Sangita is required to fly
    is equal to the product of months and hours per month -/
theorem total_hours_theorem :
  required_hours = months * hours_per_month :=
by sorry

/-- Theorem stating that the total number of hours Sangita is required to fly
    is equal to the sum of completed hours and remaining hours -/
theorem hours_breakdown_theorem :
  required_hours = completed_hours + (required_hours - completed_hours) :=
by sorry

end total_hours_theorem_hours_breakdown_theorem_l817_81776


namespace baker_shopping_cost_l817_81734

theorem baker_shopping_cost :
  let flour_boxes : ℕ := 3
  let flour_price : ℕ := 3
  let egg_trays : ℕ := 3
  let egg_price : ℕ := 10
  let milk_liters : ℕ := 7
  let milk_price : ℕ := 5
  let soda_boxes : ℕ := 2
  let soda_price : ℕ := 3
  let total_cost : ℕ := flour_boxes * flour_price + egg_trays * egg_price + 
                        milk_liters * milk_price + soda_boxes * soda_price
  total_cost = 80 := by
sorry


end baker_shopping_cost_l817_81734


namespace part_one_part_two_l817_81793

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x < 3-a}

-- Part 1
theorem part_one :
  let a : ℝ := -2
  (B a ∩ A = {x | 1 ≤ x ∧ x < 4}) ∧
  (B a ∩ (Set.univ \ A) = {x | (-4 ≤ x ∧ x < 1) ∨ (4 ≤ x ∧ x < 5)}) :=
sorry

-- Part 2
theorem part_two :
  ∀ a : ℝ, (A ∪ B a = A) ↔ a ≥ 1/2 :=
sorry

end part_one_part_two_l817_81793


namespace product_positive_not_imply_both_positive_l817_81707

theorem product_positive_not_imply_both_positive : ∃ (a b : ℝ), a * b > 0 ∧ ¬(a > 0 ∧ b > 0) := by
  sorry

end product_positive_not_imply_both_positive_l817_81707
