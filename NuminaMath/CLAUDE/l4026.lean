import Mathlib

namespace arithmetic_sequence_probability_l4026_402626

-- Define the set of numbers
def S : Set Nat := Finset.range 20

-- Define a function to check if three numbers form an arithmetic sequence
def isArithmeticSequence (a b c : Nat) : Prop := a + c = 2 * b

-- Define the total number of ways to choose 3 numbers from 20
def totalCombinations : Nat := Nat.choose 20 3

-- Define the number of valid arithmetic sequences
def validSequences : Nat := 90

-- State the theorem
theorem arithmetic_sequence_probability :
  (validSequences : ℚ) / totalCombinations = 1 / 38 := by sorry

end arithmetic_sequence_probability_l4026_402626


namespace triangle_on_axes_zero_volume_l4026_402676

/-- Given a triangle ABC with sides of length 8, 6, and 10, where each vertex is on a positive axis,
    prove that the volume of tetrahedron OABC (where O is the origin) is 0. -/
theorem triangle_on_axes_zero_volume (a b c : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 →  -- vertices on positive axes
  a^2 + b^2 = 64 →  -- AB = 8
  b^2 + c^2 = 36 →  -- BC = 6
  c^2 + a^2 = 100 →  -- CA = 10
  (1/6 : ℝ) * a * b * c = 0 :=  -- volume of tetrahedron OABC
by sorry


end triangle_on_axes_zero_volume_l4026_402676


namespace vector_relations_l4026_402695

/-- Two vectors in R² -/
def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

/-- Perpendicular vectors have dot product zero -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Parallel vectors have proportional components -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_relations (x : ℝ) :
  (perpendicular (a x) (b x) → x = 3 ∨ x = -1) ∧
  (parallel (a x) (b x) → x = 0 ∨ x = -2) := by
  sorry

end vector_relations_l4026_402695


namespace symmetric_point_coordinates_l4026_402625

/-- A point in a 2D Cartesian coordinate system. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis. -/
def symmetricXAxis (p q : Point) : Prop :=
  q.x = p.x ∧ q.y = -p.y

/-- The theorem stating that if Q is symmetric to P(-3, 2) with respect to the x-axis,
    then Q has coordinates (-3, -2). -/
theorem symmetric_point_coordinates :
  let p : Point := ⟨-3, 2⟩
  let q : Point := ⟨-3, -2⟩
  symmetricXAxis p q → q = ⟨-3, -2⟩ := by
  sorry


end symmetric_point_coordinates_l4026_402625


namespace nines_in_hundred_l4026_402694

def count_nines (n : ℕ) : ℕ :=
  (n / 10) + (n - n / 10 * 10)

theorem nines_in_hundred : count_nines 100 = 20 := by
  sorry

end nines_in_hundred_l4026_402694


namespace three_numbers_average_l4026_402624

theorem three_numbers_average (a b c : ℝ) 
  (h1 : a + (b + c) / 2 = 65)
  (h2 : b + (a + c) / 2 = 69)
  (h3 : c + (a + b) / 2 = 76)
  : (a + b + c) / 3 = 35 := by
sorry

end three_numbers_average_l4026_402624


namespace sufficient_not_necessary_l4026_402667

def M : Set ℝ := {y | 0 < y ∧ y < 1}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem sufficient_not_necessary : 
  (∀ x, x ∈ M → x ∈ N) ∧ 
  (∃ x, x ∈ N ∧ x ∉ M) :=
by sorry

end sufficient_not_necessary_l4026_402667


namespace find_divisor_l4026_402670

theorem find_divisor (n s : ℕ) (hn : n = 5264) (hs : s = 11) :
  let d := n - s
  (d ∣ d) ∧ (∀ m : ℕ, m < s → ¬(d ∣ (n - m))) → d = 5253 :=
by sorry

end find_divisor_l4026_402670


namespace exists_valid_permutation_l4026_402650

/-- A permutation of numbers from 1 to 200 -/
def Permutation := Fin 200 → Fin 200

/-- Check if a permutation satisfies the adjacent difference condition -/
def ValidPermutation (p : Permutation) : Prop :=
  ∀ i : Fin 199, (p (i + 1) - p i).val = 3 ∨ (p (i + 1) - p i).val = 5 ∨
                 (p i - p (i + 1)).val = 3 ∨ (p i - p (i + 1)).val = 5

/-- Theorem stating the existence of a valid permutation -/
theorem exists_valid_permutation : ∃ p : Permutation, ValidPermutation p :=
  sorry

end exists_valid_permutation_l4026_402650


namespace probability_of_purple_marble_l4026_402690

theorem probability_of_purple_marble (blue_prob green_prob purple_prob : ℝ) : 
  blue_prob = 0.35 →
  green_prob = 0.45 →
  blue_prob + green_prob + purple_prob = 1 →
  purple_prob = 0.2 := by
sorry

end probability_of_purple_marble_l4026_402690


namespace quadratic_roots_relation_l4026_402633

theorem quadratic_roots_relation (m n p : ℝ) : 
  m ≠ 0 → n ≠ 0 → p ≠ 0 →
  (∃ s₁ s₂ : ℝ, (s₁ * s₂ = m) ∧ 
               (s₁ + s₂ = -p) ∧
               ((3 * s₁) * (3 * s₂) = n)) →
  n / p = -27 := by
sorry

end quadratic_roots_relation_l4026_402633


namespace quadratic_inequality_solution_l4026_402603

theorem quadratic_inequality_solution (a b : ℝ) 
  (h1 : (1 : ℝ) / 3 * 1 = -1 / a) 
  (h2 : (1 : ℝ) / 3 + 1 = -b / a) 
  (h3 : a < 0) : 
  a + b = 1 := by
sorry

end quadratic_inequality_solution_l4026_402603


namespace quadratic_equation_equal_coefficients_l4026_402632

/-- A quadratic equation with coefficients forming an arithmetic sequence and reciprocal roots -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  roots_reciprocal : ∃ (r s : ℝ), r * s = 1 ∧ a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0
  coeff_arithmetic : b - a = c - b

/-- The coefficients of a quadratic equation with reciprocal roots and coefficients in arithmetic sequence are equal -/
theorem quadratic_equation_equal_coefficients (eq : QuadraticEquation) : eq.a = eq.b ∧ eq.b = eq.c := by
  sorry

end quadratic_equation_equal_coefficients_l4026_402632


namespace three_digit_number_divisibility_l4026_402681

theorem three_digit_number_divisibility (a b c : Nat) : 
  a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ a + b + c = 7 →
  (100 * a + 10 * b + c) % 7 = 0 ↔ b = c :=
by sorry

end three_digit_number_divisibility_l4026_402681


namespace sqrt_inequality_l4026_402647

theorem sqrt_inequality (x : ℝ) (h : x ≥ -3) :
  Real.sqrt (x + 5) - Real.sqrt (x + 3) > Real.sqrt (x + 6) - Real.sqrt (x + 4) := by
  sorry

end sqrt_inequality_l4026_402647


namespace carols_invitations_l4026_402698

/-- Given that Carol bought packages of invitations, prove that the number of friends she can invite is equal to the product of invitations per package and the number of packages. -/
theorem carols_invitations (invitations_per_package : ℕ) (num_packages : ℕ) :
  invitations_per_package = 9 →
  num_packages = 5 →
  invitations_per_package * num_packages = 45 := by
  sorry

#check carols_invitations

end carols_invitations_l4026_402698


namespace molecular_weight_Al_OH_3_value_l4026_402641

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The molecular weight of Al(OH)3 in g/mol -/
def molecular_weight_Al_OH_3 : ℝ := 
  atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H

/-- Theorem stating that the molecular weight of Al(OH)3 is 78.01 g/mol -/
theorem molecular_weight_Al_OH_3_value : 
  molecular_weight_Al_OH_3 = 78.01 := by sorry

end molecular_weight_Al_OH_3_value_l4026_402641


namespace positive_real_inequality_l4026_402616

theorem positive_real_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 1) :
  a * (a - 1) + b * (b - 1) + c * (c - 1) ≥ 0 := by
  sorry

end positive_real_inequality_l4026_402616


namespace faye_coloring_books_l4026_402642

theorem faye_coloring_books (given_away_first : ℝ) (given_away_second : ℝ) (remaining : ℕ) 
  (h1 : given_away_first = 34.0)
  (h2 : given_away_second = 3.0)
  (h3 : remaining = 11) :
  given_away_first + given_away_second + remaining = 48.0 := by
  sorry

end faye_coloring_books_l4026_402642


namespace subtract_problem_l4026_402627

theorem subtract_problem (x : ℕ) (h : 913 - x = 514) : 514 - x = 115 := by
  sorry

end subtract_problem_l4026_402627


namespace quadratic_root_relation_l4026_402696

/-- Given two quadratic equations and a relationship between their roots, prove the value of k. -/
theorem quadratic_root_relation (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 9 = 0 → ∃ y : ℝ, y^2 - k*y + 9 = 0 ∧ y = x + 3) →
  k = -3 :=
by sorry

end quadratic_root_relation_l4026_402696


namespace eulers_formula_l4026_402602

/-- A connected planar graph -/
structure ConnectedPlanarGraph where
  s : ℕ  -- number of vertices
  f : ℕ  -- number of faces
  a : ℕ  -- number of edges

/-- Euler's formula for connected planar graphs -/
theorem eulers_formula (G : ConnectedPlanarGraph) : G.f = G.a - G.s + 2 := by
  sorry

end eulers_formula_l4026_402602


namespace arithmetic_sequence_12th_term_l4026_402657

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions, prove that the 12th term is 14. -/
theorem arithmetic_sequence_12th_term
  (seq : ArithmeticSequence)
  (h1 : seq.a 7 + seq.a 9 = 15)
  (h2 : seq.a 4 = 1) :
  seq.a 12 = 14 := by
  sorry

end arithmetic_sequence_12th_term_l4026_402657


namespace four_at_six_equals_twenty_l4026_402689

-- Define the @ operation
def at_operation (a b : ℤ) : ℤ := 4*a - 2*b + a^2

-- Theorem statement
theorem four_at_six_equals_twenty : at_operation 4 6 = 20 := by
  sorry

end four_at_six_equals_twenty_l4026_402689


namespace arithmetic_sequence_sum_l4026_402621

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 4 + a 6 = 3 →
  a 1 + a 3 + a 5 + a 7 = 4 :=
by
  sorry

end arithmetic_sequence_sum_l4026_402621


namespace josh_marbles_l4026_402660

/-- The number of marbles Josh has after receiving marbles from Jack -/
def total_marbles (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Josh has 42 marbles after receiving marbles from Jack -/
theorem josh_marbles :
  let initial_marbles : ℕ := 22
  let marbles_from_jack : ℕ := 20
  total_marbles initial_marbles marbles_from_jack = 42 := by
sorry

end josh_marbles_l4026_402660


namespace deposit_percentage_l4026_402622

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) : 
  deposit = 130 → remaining = 1170 → (deposit / (deposit + remaining)) * 100 = 10 := by
  sorry

end deposit_percentage_l4026_402622


namespace quadratic_inequality_equivalence_l4026_402654

theorem quadratic_inequality_equivalence (x : ℝ) :
  3 * x^2 + x - 2 < 0 ↔ -1 < x ∧ x < 2/3 := by
  sorry

end quadratic_inequality_equivalence_l4026_402654


namespace base_conversion_sum_l4026_402643

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 13 to base 10 -/
def base13_to_base10 (n : ℕ) : ℕ := sorry

theorem base_conversion_sum :
  let base8_num := 357
  let base13_num := 4 * 13^2 + 12 * 13 + 13
  (base8_to_base10 base8_num) + (base13_to_base10 base13_num) = 1084 := by
  sorry

end base_conversion_sum_l4026_402643


namespace two_shirts_per_package_l4026_402666

/-- Given a number of packages and a total number of t-shirts,
    calculate the number of t-shirts per package. -/
def tShirtsPerPackage (packages : ℕ) (totalShirts : ℕ) : ℚ :=
  totalShirts / packages

/-- Theorem stating that given 28 packages and 56 total t-shirts,
    the number of t-shirts per package is 2. -/
theorem two_shirts_per_package :
  tShirtsPerPackage 28 56 = 2 := by
  sorry

end two_shirts_per_package_l4026_402666


namespace ball_in_hole_within_six_bounces_l4026_402659

/-- Represents a point on the table -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hole on the table -/
structure Hole where
  location : Point

/-- Represents a rectangular table with holes -/
structure Table where
  length : ℝ
  width : ℝ
  holes : List Hole

/-- Represents a ball's trajectory -/
structure Trajectory where
  start : Point
  bounces : List Point

/-- Function to check if a trajectory ends in a hole within n bounces -/
def endsInHole (traj : Trajectory) (table : Table) (n : ℕ) : Prop :=
  ∃ (h : Hole), h ∈ table.holes ∧ traj.bounces.length ≤ n ∧ traj.bounces.getLast? = some h.location

/-- The main theorem -/
theorem ball_in_hole_within_six_bounces 
  (table : Table) 
  (a b c : Point) : 
  table.length = 8 ∧ 
  table.width = 5 ∧ 
  table.holes.length = 4 →
  ∃ (start : Point) (traj : Trajectory), 
    (start = a ∨ start = b ∨ start = c) ∧
    traj.start = start ∧
    endsInHole traj table 6 :=
sorry

end ball_in_hole_within_six_bounces_l4026_402659


namespace wilson_prime_l4026_402646

theorem wilson_prime (n : ℕ) (h : n > 1) (h_div : n ∣ (Nat.factorial (n - 1) + 1)) : Nat.Prime n := by
  sorry

end wilson_prime_l4026_402646


namespace two_lines_at_45_degrees_l4026_402620

/-- The equation represents two lines that intersect at a 45° angle when k = 80 -/
theorem two_lines_at_45_degrees (x y : ℝ) :
  let k : ℝ := 80
  let equation := x^2 + x*y - 6*y^2 - 20*x - 20*y + k
  ∃ (l₁ l₂ : ℝ → ℝ → Prop),
    (∀ x y, equation = 0 ↔ (l₁ x y ∨ l₂ x y)) ∧
    (∃ x₀ y₀, l₁ x₀ y₀ ∧ l₂ x₀ y₀) ∧
    (∀ x₀ y₀, l₁ x₀ y₀ ∧ l₂ x₀ y₀ → 
      ∃ (v₁ v₂ : ℝ × ℝ),
        (v₁.1 ≠ 0 ∨ v₁.2 ≠ 0) ∧
        (v₂.1 ≠ 0 ∨ v₂.2 ≠ 0) ∧
        (v₁.1 * v₂.1 + v₁.2 * v₂.2) / (Real.sqrt (v₁.1^2 + v₁.2^2) * Real.sqrt (v₂.1^2 + v₂.2^2)) = Real.cos (π/4)) :=
by sorry


end two_lines_at_45_degrees_l4026_402620


namespace pickle_problem_l4026_402683

/-- Pickle Problem -/
theorem pickle_problem (jars cucumbers initial_vinegar pickles_per_cucumber pickles_per_jar remaining_vinegar : ℕ)
  (h1 : jars = 4)
  (h2 : cucumbers = 10)
  (h3 : initial_vinegar = 100)
  (h4 : pickles_per_cucumber = 6)
  (h5 : pickles_per_jar = 12)
  (h6 : remaining_vinegar = 60) :
  (initial_vinegar - remaining_vinegar) / jars = 10 := by
  sorry


end pickle_problem_l4026_402683


namespace b_work_time_l4026_402673

/-- The time it takes for worker a to complete the work alone -/
def a_time : ℝ := 14

/-- The time it takes for workers a and b to complete the work together -/
def ab_time : ℝ := 5.833333333333333

/-- The time it takes for worker b to complete the work alone -/
def b_time : ℝ := 10

/-- The total amount of work to be completed -/
def total_work : ℝ := 1

theorem b_work_time : 
  (1 / a_time + 1 / b_time = 1 / ab_time) ∧
  (1 / b_time = 1 / total_work) := by
  sorry

end b_work_time_l4026_402673


namespace other_integer_is_30_l4026_402623

theorem other_integer_is_30 (a b : ℤ) (h1 : 3 * a + 2 * b = 135) (h2 : a = 25 ∨ b = 25) : 
  (a ≠ 25 → b = 30) ∧ (b ≠ 25 → a = 30) := by
  sorry

end other_integer_is_30_l4026_402623


namespace min_tuple_c_value_l4026_402691

def is_valid_tuple (a b c d e f : ℕ) : Prop :=
  a + 2*b + 6*c + 30*d + 210*e + 2310*f = 2^15

def tuple_sum (a b c d e f : ℕ) : ℕ :=
  a + b + c + d + e + f

theorem min_tuple_c_value :
  ∃ (a b c d e f : ℕ),
    is_valid_tuple a b c d e f ∧
    (∀ (a' b' c' d' e' f' : ℕ),
      is_valid_tuple a' b' c' d' e' f' →
      tuple_sum a b c d e f ≤ tuple_sum a' b' c' d' e' f') ∧
    c = 1 := by sorry

end min_tuple_c_value_l4026_402691


namespace prob_blank_one_shot_prob_blank_three_shots_prob_away_from_vertices_l4026_402675

-- Define the number of bullets and the number of blanks
def total_bullets : ℕ := 4
def blank_bullets : ℕ := 1

-- Define the number of shots
def shots : ℕ := 3

-- Define the side length of the equilateral triangle
def side_length : ℝ := 10

-- Theorem for the probability of shooting a blank in one shot
theorem prob_blank_one_shot : 
  (blank_bullets : ℝ) / total_bullets = 1 / 4 := by sorry

-- Theorem for the probability of a blank appearing in 3 shots
theorem prob_blank_three_shots : 
  1 - (total_bullets - blank_bullets : ℝ) * (total_bullets - blank_bullets - 1) * (total_bullets - blank_bullets - 2) / 
    (total_bullets * (total_bullets - 1) * (total_bullets - 2)) = 3 / 4 := by sorry

-- Theorem for the probability of all shots being more than 1 unit away from vertices
theorem prob_away_from_vertices (triangle_area : ℝ) (h : triangle_area = side_length^2 * Real.sqrt 3 / 4) :
  1 - (3 * π / 2) / triangle_area = 1 - Real.sqrt 3 * π / 150 := by sorry

end prob_blank_one_shot_prob_blank_three_shots_prob_away_from_vertices_l4026_402675


namespace solve_system_l4026_402697

theorem solve_system (a b : ℚ) 
  (eq1 : 5 + 2 * a = 6 - 3 * b) 
  (eq2 : 3 + 4 * b = 10 + 2 * a) : 
  5 - 2 * a = 4 := by
  sorry

end solve_system_l4026_402697


namespace product_mod_seven_l4026_402692

theorem product_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end product_mod_seven_l4026_402692


namespace cosine_BHD_value_l4026_402688

structure RectangularPrism where
  DHG : Real
  FHB : Real

def cosine_BHD (prism : RectangularPrism) : Real :=
  sorry

theorem cosine_BHD_value (prism : RectangularPrism) 
  (h1 : prism.DHG = Real.pi / 4)
  (h2 : prism.FHB = Real.pi / 3) :
  cosine_BHD prism = Real.sqrt 6 / 4 := by
  sorry

end cosine_BHD_value_l4026_402688


namespace positive_real_inequality_l4026_402612

theorem positive_real_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * (a^2 + b*c)) / (b + c) + (b * (b^2 + c*a)) / (c + a) + (c * (c^2 + a*b)) / (a + b) ≥ a*b + b*c + c*a := by
  sorry

end positive_real_inequality_l4026_402612


namespace sqrt_six_irrational_l4026_402619

theorem sqrt_six_irrational : Irrational (Real.sqrt 6) := by
  sorry

end sqrt_six_irrational_l4026_402619


namespace factor_x10_minus_1024_l4026_402607

theorem factor_x10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x-2)*(x+2)*(x^4 + 2*x^3 + 4*x^2 + 8*x + 16)*(x^4 - 2*x^3 + 4*x^2 - 8*x + 16) := by
  sorry

end factor_x10_minus_1024_l4026_402607


namespace marble_arrangement_l4026_402637

/-- Represents the number of green marbles -/
def green_marbles : Nat := 4

/-- Represents the number of red marbles -/
def red_marbles : Nat := 3

/-- Represents the maximum number of blue marbles that can be used to create a balanced arrangement -/
def m : Nat := 5

/-- Represents the total number of slots where blue marbles can be placed -/
def total_slots : Nat := green_marbles + red_marbles + 1

/-- Calculates the number of ways to arrange the marbles -/
def N : Nat := Nat.choose (m + total_slots - 1) m

/-- Theorem stating the properties of the marble arrangement -/
theorem marble_arrangement :
  (N % 1000 = 287) ∧
  (∀ k : Nat, k > m → Nat.choose (k + total_slots - 1) k % 1000 ≠ 287) := by
  sorry

end marble_arrangement_l4026_402637


namespace absolute_value_sum_zero_implies_product_l4026_402604

theorem absolute_value_sum_zero_implies_product (x y : ℝ) :
  |x - 1| + |y + 3| = 0 → (x + 1) * (y - 3) = -12 := by
  sorry

end absolute_value_sum_zero_implies_product_l4026_402604


namespace log3_derivative_l4026_402630

theorem log3_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 3) x = 1 / (x * Real.log 3) := by
sorry

end log3_derivative_l4026_402630


namespace midpoint_trajectory_l4026_402618

/-- The trajectory of the midpoint of a line segment with one fixed endpoint and the other moving on a circle -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ x₀ y₀ : ℝ, (x₀ + 1)^2 + y₀^2 = 4 ∧ x = (x₀ + 4)/2 ∧ y = (y₀ + 3)/2) → 
  (x - 3/2)^2 + (y - 3/2)^2 = 1 := by
sorry

end midpoint_trajectory_l4026_402618


namespace tea_profit_percentage_l4026_402653

/-- Given a tea mixture and sale price, calculate the profit percentage -/
theorem tea_profit_percentage
  (tea1_weight : ℝ)
  (tea1_cost : ℝ)
  (tea2_weight : ℝ)
  (tea2_cost : ℝ)
  (sale_price : ℝ)
  (h1 : tea1_weight = 80)
  (h2 : tea1_cost = 15)
  (h3 : tea2_weight = 20)
  (h4 : tea2_cost = 20)
  (h5 : sale_price = 20.8) :
  let total_cost := tea1_weight * tea1_cost + tea2_weight * tea2_cost
  let total_weight := tea1_weight + tea2_weight
  let total_sale := total_weight * sale_price
  let profit := total_sale - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 30 := by
sorry

end tea_profit_percentage_l4026_402653


namespace equation_satisfied_l4026_402693

theorem equation_satisfied (x y : ℝ) (hx : x = 5) (hy : y = 3) : 2 * x - 3 * y = 1 := by
  sorry

end equation_satisfied_l4026_402693


namespace cube_triangles_area_sum_l4026_402665

/-- Represents a 3D point in a cube --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space --/
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

/-- The vertices of a 2x2x2 cube --/
def cubeVertices : List Point3D := sorry

/-- Calculates the area of a triangle in 3D space --/
def triangleArea (t : Triangle3D) : ℝ := sorry

/-- Generates all possible triangles from the cube vertices --/
def allTriangles : List Triangle3D := sorry

/-- Expresses a real number in the form m + √n + √p --/
structure SqrtForm where
  m : ℤ
  n : ℤ
  p : ℤ

/-- Converts a real number to SqrtForm --/
def toSqrtForm (r : ℝ) : SqrtForm := sorry

/-- The main theorem --/
theorem cube_triangles_area_sum :
  let totalArea := (allTriangles.map triangleArea).sum
  let sqrtForm := toSqrtForm totalArea
  sqrtForm.m + sqrtForm.n + sqrtForm.p = 121 := by sorry

end cube_triangles_area_sum_l4026_402665


namespace min_obtuse_angles_convex_octagon_l4026_402614

/-- A convex octagon -/
structure ConvexOctagon where
  -- We don't need to define the structure explicitly for this problem

/-- The number of angles in an octagon -/
def num_angles : ℕ := 8

/-- The sum of exterior angles in any polygon -/
def sum_exterior_angles : ℕ := 360

/-- Theorem: In a convex octagon, the minimum number of obtuse interior angles is 5 -/
theorem min_obtuse_angles_convex_octagon (O : ConvexOctagon) : 
  ∃ (n : ℕ), n ≥ 5 ∧ n = (num_angles - (sum_exterior_angles / 90)) := by
  sorry

end min_obtuse_angles_convex_octagon_l4026_402614


namespace area_above_x_axis_half_total_l4026_402664

-- Define the parallelogram PQRS
def P : ℝ × ℝ := (4, 4)
def Q : ℝ × ℝ := (-2, -2)
def R : ℝ × ℝ := (-8, -2)
def S : ℝ × ℝ := (-2, 4)

-- Define a function to calculate the area of a parallelogram
def parallelogramArea (a b c d : ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the area of the part of the parallelogram above the x-axis
def areaAboveXAxis (a b c d : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_above_x_axis_half_total : 
  areaAboveXAxis P Q R S = (1/2) * parallelogramArea P Q R S := by sorry

end area_above_x_axis_half_total_l4026_402664


namespace cost_price_calculation_l4026_402628

/-- Proves that if an article is sold for $1200 with a 20% profit, then the cost price is $1000. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 1200 ∧ profit_percentage = 20 →
  (selling_price = (100 + profit_percentage) / 100 * 1000) := by
  sorry

end cost_price_calculation_l4026_402628


namespace nonzero_y_solution_l4026_402631

theorem nonzero_y_solution (y : ℝ) (hy : y ≠ 0) (h : (3 * y)^5 = (9 * y)^4) : y = 27 := by
  sorry

end nonzero_y_solution_l4026_402631


namespace rectangle_circle_area_ratio_l4026_402649

/-- A rectangle and a circle that intersect in a specific way -/
structure RectangleCircleIntersection where
  /-- The diameter of the circle -/
  d : ℝ
  /-- Assumption that the diameter is positive -/
  d_pos : d > 0
  /-- The length of the longer side of the rectangle -/
  long_side : ℝ
  /-- The length of the shorter side of the rectangle -/
  short_side : ℝ
  /-- The longer side of the rectangle is twice the diameter of the circle -/
  long_side_eq : long_side = 2 * d
  /-- The shorter side of the rectangle is equal to the diameter of the circle -/
  short_side_eq : short_side = d

/-- The ratio of the area of the rectangle to the area of the circle is 8/π -/
theorem rectangle_circle_area_ratio (rc : RectangleCircleIntersection) :
  (rc.long_side * rc.short_side) / (π * (rc.d / 2)^2) = 8 / π := by
  sorry

end rectangle_circle_area_ratio_l4026_402649


namespace quadratic_sum_l4026_402672

-- Define the quadratic function
def f (x : ℝ) : ℝ := -8 * x^2 + 16 * x + 320

-- Define the completed square form
def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x + b)^2 + c

theorem quadratic_sum : ∃ a b c : ℝ, 
  (∀ x, f x = g a b c x) ∧ 
  (a + b + c = 319) := by sorry

end quadratic_sum_l4026_402672


namespace a_can_be_any_real_l4026_402655

theorem a_can_be_any_real : ∀ (a b c d : ℤ), 
  b > 0 → d < 0 → (a : ℚ) / b > (c : ℚ) / d → 
  (∃ (x : ℝ), x > 0 ∧ (∃ (a : ℤ), (a : ℚ) / b > (c : ℚ) / d ∧ (a : ℝ) = x)) ∧
  (∃ (y : ℝ), y < 0 ∧ (∃ (a : ℤ), (a : ℚ) / b > (c : ℚ) / d ∧ (a : ℝ) = y)) ∧
  (∃ (a : ℤ), (a : ℚ) / b > (c : ℚ) / d ∧ a = 0) :=
by sorry

end a_can_be_any_real_l4026_402655


namespace complex_fraction_sum_l4026_402600

theorem complex_fraction_sum (a b : ℝ) : 
  (1 + 2*I) / (1 + I) = a + b*I → a + b = 2 := by sorry

end complex_fraction_sum_l4026_402600


namespace inequality_proof_l4026_402611

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 1) :
  (1 / x^2 + x) * (1 / y^2 + y) * (1 / z^2 + z) ≥ (28/3)^3 := by
  sorry

end inequality_proof_l4026_402611


namespace book_arrangement_count_book_arrangement_theorem_l4026_402661

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange 3 different mathematics books and 3 different Chinese books
    on a shelf, such that books of the same type are not adjacent. -/
theorem book_arrangement_count : ℕ := 
  2 * permutations 3 * permutations 3

/-- Prove that the number of ways to arrange 3 different mathematics books and 3 different Chinese books
    on a shelf, such that books of the same type are not adjacent, is equal to 72. -/
theorem book_arrangement_theorem : book_arrangement_count = 72 := by
  sorry

end book_arrangement_count_book_arrangement_theorem_l4026_402661


namespace cubic_three_distinct_roots_in_interval_l4026_402606

/-- A cubic equation x^3 + px + q = 0 has three distinct roots in (-2, 4) if and only if
    its coefficients p and q satisfy the given conditions. -/
theorem cubic_three_distinct_roots_in_interval
  (p q : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    -2 < x₁ ∧ x₁ < 4 ∧ -2 < x₂ ∧ x₂ < 4 ∧ -2 < x₃ ∧ x₃ < 4 ∧
    x₁^3 + p*x₁ + q = 0 ∧ x₂^3 + p*x₂ + q = 0 ∧ x₃^3 + p*x₃ + q = 0) ↔
  (4*p^3 + 27*q^2 < 0 ∧ -4*p - 64 < q ∧ q < 2*p + 8) :=
sorry

end cubic_three_distinct_roots_in_interval_l4026_402606


namespace triangle_third_side_length_l4026_402668

theorem triangle_third_side_length (a b c : ℝ) (θ : ℝ) (h1 : a = 9) (h2 : b = 11) (h3 : θ = 150 * π / 180) :
  c^2 = a^2 + b^2 - 2*a*b*Real.cos θ → c = Real.sqrt (202 + 99 * Real.sqrt 3) :=
sorry

end triangle_third_side_length_l4026_402668


namespace sum_70_terms_is_negative_350_l4026_402682

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference

/-- Sum of first n terms of an arithmetic progression -/
def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

/-- Theorem: For an arithmetic progression with specific properties, 
    the sum of its first 70 terms is -350 -/
theorem sum_70_terms_is_negative_350 
  (ap : ArithmeticProgression)
  (h1 : sum_n_terms ap 20 = 200)
  (h2 : sum_n_terms ap 50 = 50) :
  sum_n_terms ap 70 = -350 := by
  sorry

end sum_70_terms_is_negative_350_l4026_402682


namespace probability_sum_10_l4026_402613

def die_faces : Nat := 6

def total_outcomes : Nat := die_faces ^ 3

def favorable_outcomes : Nat := 30

theorem probability_sum_10 : 
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 := by
  sorry

end probability_sum_10_l4026_402613


namespace boulevard_painting_cost_l4026_402652

/-- Represents a side of the boulevard with house numbers -/
structure BoulevardSide where
  start : ℕ
  diff : ℕ
  count : ℕ

/-- Calculates the sum of digits for all numbers in an arithmetic sequence -/
def sumOfDigits (side : BoulevardSide) : ℕ :=
  sorry

/-- The total cost of painting house numbers on both sides of the boulevard -/
def totalCost (eastSide westSide : BoulevardSide) : ℕ :=
  sumOfDigits eastSide + sumOfDigits westSide

theorem boulevard_painting_cost :
  let eastSide : BoulevardSide := { start := 5, diff := 7, count := 25 }
  let westSide : BoulevardSide := { start := 2, diff := 5, count := 25 }
  totalCost eastSide westSide = 113 :=
sorry

end boulevard_painting_cost_l4026_402652


namespace initial_orchids_l4026_402678

/-- Proves that the initial number of orchids in the vase was 2, given that there are now 21 orchids
    in the vase after 19 orchids were added. -/
theorem initial_orchids (final_orchids : ℕ) (added_orchids : ℕ) 
  (h1 : final_orchids = 21) 
  (h2 : added_orchids = 19) : 
  final_orchids - added_orchids = 2 := by
  sorry

end initial_orchids_l4026_402678


namespace interest_calculation_time_l4026_402629

-- Define the given values
def simple_interest : ℚ := 345/100
def principal : ℚ := 23
def rate_paise : ℚ := 5

-- Convert rate from paise to rupees
def rate : ℚ := rate_paise / 100

-- Define the simple interest formula
def calculate_time (si p r : ℚ) : ℚ := si / (p * r)

-- State the theorem
theorem interest_calculation_time :
  calculate_time simple_interest principal rate = 3 := by
  sorry

end interest_calculation_time_l4026_402629


namespace clara_weight_l4026_402634

/-- Given two positive real numbers representing weights in pounds,
    prove that one of them (Clara's weight) is equal to 960/7 pounds,
    given the specified conditions. -/
theorem clara_weight (alice_weight clara_weight : ℝ) 
  (h1 : alice_weight > 0)
  (h2 : clara_weight > 0)
  (h3 : alice_weight + clara_weight = 240)
  (h4 : clara_weight - alice_weight = alice_weight / 3) :
  clara_weight = 960 / 7 := by
  sorry

end clara_weight_l4026_402634


namespace sum_of_eleven_terms_l4026_402677

def a (n : ℕ) : ℤ := 1 - 2 * n

def S (n : ℕ) : ℤ := n * (a 1 + a n) / 2

def sequence_sum (n : ℕ) : ℚ := 
  Finset.sum (Finset.range n) (λ i => S (i + 1) / (i + 1))

theorem sum_of_eleven_terms : sequence_sum 11 = -66 := by sorry

end sum_of_eleven_terms_l4026_402677


namespace value_of_b_l4026_402685

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 4) : b = 4 := by
  sorry

end value_of_b_l4026_402685


namespace angela_action_figures_l4026_402636

theorem angela_action_figures (initial : ℕ) (sold_fraction : ℚ) (given_fraction : ℚ) : 
  initial = 24 → 
  sold_fraction = 1/4 → 
  given_fraction = 1/3 → 
  initial - (initial * sold_fraction).floor - ((initial - (initial * sold_fraction).floor) * given_fraction).floor = 12 := by
  sorry

end angela_action_figures_l4026_402636


namespace parallel_vectors_imply_y_equals_5_l4026_402638

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, prove that if they are parallel, then y = 5 -/
theorem parallel_vectors_imply_y_equals_5 :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, y + 1)
  parallel a b → y = 5 := by
sorry

end parallel_vectors_imply_y_equals_5_l4026_402638


namespace right_triangle_angle_measure_l4026_402610

/-- In a right triangle ABC where angle C is 90° and tan A is √3, angle A measures 60°. -/
theorem right_triangle_angle_measure (A B C : Real) (h1 : A + B + C = 180) 
  (h2 : C = 90) (h3 : Real.tan A = Real.sqrt 3) : A = 60 := by
  sorry

end right_triangle_angle_measure_l4026_402610


namespace city_households_l4026_402605

/-- The number of deer that entered the city -/
def num_deer : ℕ := 100

/-- The number of households in the city -/
def num_households : ℕ := 75

theorem city_households : 
  (num_households < num_deer) ∧ 
  (4 * num_households = 3 * num_deer) := by
  sorry

end city_households_l4026_402605


namespace pumpkin_patch_pie_filling_l4026_402679

/-- Represents the number of cans of pie filling produced from small and large pumpkins -/
def cans_of_pie_filling (small_pumpkins : ℕ) (large_pumpkins : ℕ) : ℕ :=
  (small_pumpkins / 2) + large_pumpkins

theorem pumpkin_patch_pie_filling :
  let small_pumpkins : ℕ := 50
  let large_pumpkins : ℕ := 33
  let total_sales : ℕ := 120
  let small_price : ℕ := 3
  let large_price : ℕ := 5
  cans_of_pie_filling small_pumpkins large_pumpkins = 58 := by
  sorry

#eval cans_of_pie_filling 50 33

end pumpkin_patch_pie_filling_l4026_402679


namespace count_congruent_integers_l4026_402615

theorem count_congruent_integers (n : ℕ) : 
  (Finset.filter (fun x => x > 0 ∧ x < 2000 ∧ x % 13 = 3) (Finset.range 2000)).card = 154 := by
  sorry

end count_congruent_integers_l4026_402615


namespace square_sum_of_xy_l4026_402617

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 83)
  (h2 : x^2 * y + x * y^2 = 1056) : 
  x^2 + y^2 = 458 := by sorry

end square_sum_of_xy_l4026_402617


namespace negative_result_l4026_402645

theorem negative_result : 1 - 9 < 0 := by
  sorry

#check negative_result

end negative_result_l4026_402645


namespace certain_amount_of_seconds_l4026_402658

/-- Given that 12 is to a certain amount of seconds as 16 is to 8 minutes,
    prove that the certain amount of seconds is 360. -/
theorem certain_amount_of_seconds : ∃ X : ℝ, 
  (12 / X = 16 / (8 * 60)) ∧ (X = 360) := by
  sorry

end certain_amount_of_seconds_l4026_402658


namespace bottles_not_in_crates_l4026_402686

/-- Represents the number of bottles that can be held by each crate size -/
structure CrateCapacity where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of crates of each size -/
structure CrateCount where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculate the total capacity of all crates -/
def totalCrateCapacity (capacity : CrateCapacity) (count : CrateCount) : Nat :=
  capacity.small * count.small + capacity.medium * count.medium + capacity.large * count.large

/-- Calculate the number of bottles that will not be placed in a crate -/
def bottlesNotInCrates (totalBottles : Nat) (capacity : CrateCapacity) (count : CrateCount) : Nat :=
  totalBottles - totalCrateCapacity capacity count

/-- Theorem stating that 50 bottles will not be placed in a crate -/
theorem bottles_not_in_crates : 
  let totalBottles : Nat := 250
  let capacity : CrateCapacity := { small := 8, medium := 12, large := 20 }
  let count : CrateCount := { small := 5, medium := 5, large := 5 }
  bottlesNotInCrates totalBottles capacity count = 50 := by
  sorry

end bottles_not_in_crates_l4026_402686


namespace min_k_value_l4026_402644

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- AB is the length of side AB
  AB : ℝ
  -- h is the height of the trapezoid
  h : ℝ
  -- E and F are midpoints of AD and AB respectively
  -- CD = 2AB (implied by the structure)

/-- The area difference between triangle CDG and quadrilateral AEGF -/
def areaDifference (t : Trapezoid) : ℝ :=
  2 * t.AB * t.h - t.AB * t.h

/-- The area of the trapezoid ABCD -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  3 * t.AB * t.h

/-- Main theorem: The minimum value of k is 8 -/
theorem min_k_value (t : Trapezoid) (k : ℕ+) 
    (h1 : areaDifference t = k / 24)
    (h2 : ∃ n : ℕ, trapezoidArea t = n) : 
  k ≥ 8 ∧ ∃ (t : Trapezoid) (k : ℕ+), k = 8 ∧ areaDifference t = k / 24 ∧ ∃ (n : ℕ), trapezoidArea t = n :=
by
  sorry

end min_k_value_l4026_402644


namespace football_points_sum_l4026_402601

theorem football_points_sum : 
  let zach_points : Float := 42.0
  let ben_points : Float := 21.0
  let sarah_points : Float := 18.5
  let emily_points : Float := 27.5
  zach_points + ben_points + sarah_points + emily_points = 109.0 := by
  sorry

end football_points_sum_l4026_402601


namespace symmetric_point_xoz_l4026_402648

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The plane xOz in three-dimensional space -/
def PlaneXOZ : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- Symmetry with respect to the plane xOz -/
def symmetricXOZ (p : Point3D) : Point3D :=
  ⟨p.x, -p.y, p.z⟩

theorem symmetric_point_xoz :
  let A : Point3D := ⟨-3, 2, -4⟩
  symmetricXOZ A = ⟨-3, -2, -4⟩ := by
  sorry

end symmetric_point_xoz_l4026_402648


namespace factorization_proof_l4026_402680

theorem factorization_proof (x y : ℝ) : -2*x*y^2 + 4*x*y - 2*x = -2*x*(y - 1)^2 := by
  sorry

end factorization_proof_l4026_402680


namespace max_pieces_of_cake_l4026_402608

/-- The size of the cake in inches -/
def cake_size : ℕ := 16

/-- The size of each piece in inches -/
def piece_size : ℕ := 4

/-- The area of the cake in square inches -/
def cake_area : ℕ := cake_size * cake_size

/-- The area of each piece in square inches -/
def piece_area : ℕ := piece_size * piece_size

/-- The maximum number of pieces that can be cut from the cake -/
def max_pieces : ℕ := cake_area / piece_area

theorem max_pieces_of_cake :
  max_pieces = 16 :=
sorry

end max_pieces_of_cake_l4026_402608


namespace unique_solution_quadratic_equation_l4026_402640

theorem unique_solution_quadratic_equation :
  ∃! x : ℝ, (3012 + x)^2 = x^2 ∧ x = -1506 := by sorry

end unique_solution_quadratic_equation_l4026_402640


namespace multiple_of_six_is_multiple_of_three_l4026_402684

theorem multiple_of_six_is_multiple_of_three (n : ℤ) :
  (∃ k : ℤ, n = 6 * k) → (∃ m : ℤ, n = 3 * m) := by
  sorry

end multiple_of_six_is_multiple_of_three_l4026_402684


namespace q_satisfies_conditions_l4026_402687

/-- The cubic polynomial q(x) that satisfies given conditions -/
def q (x : ℝ) : ℝ := 4 * x^3 - 19 * x^2 + 5 * x + 6

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  q 0 = 6 ∧ q 1 = -4 ∧ q 2 = 0 ∧ q 3 = 10 := by
  sorry

end q_satisfies_conditions_l4026_402687


namespace unique_prime_pair_l4026_402639

theorem unique_prime_pair : ∃! p : ℕ, Prime p ∧ Prime (p + 15) := by sorry

end unique_prime_pair_l4026_402639


namespace debbie_large_boxes_l4026_402635

def large_box_tape : ℕ := 5
def medium_box_tape : ℕ := 3
def small_box_tape : ℕ := 2
def medium_boxes_packed : ℕ := 8
def small_boxes_packed : ℕ := 5
def total_tape_used : ℕ := 44

theorem debbie_large_boxes :
  ∃ (large_boxes : ℕ),
    large_boxes * large_box_tape +
    medium_boxes_packed * medium_box_tape +
    small_boxes_packed * small_box_tape = total_tape_used ∧
    large_boxes = 2 :=
by sorry

end debbie_large_boxes_l4026_402635


namespace cos_neg_seventeen_pi_fourths_l4026_402669

theorem cos_neg_seventeen_pi_fourths : Real.cos (-17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_neg_seventeen_pi_fourths_l4026_402669


namespace quadratic_root_l4026_402662

theorem quadratic_root (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - 3 * x - k = 0 ∧ x = 1) → 
  (∃ y : ℝ, 2 * y^2 - 3 * y - k = 0 ∧ y = 1/2) := by
sorry

end quadratic_root_l4026_402662


namespace no_intersection_l4026_402663

def f (x : ℝ) := |3 * x + 6|
def g (x : ℝ) := -|4 * x - 3|

theorem no_intersection :
  ∀ x : ℝ, f x ≠ g x :=
sorry

end no_intersection_l4026_402663


namespace intersection_of_A_and_B_l4026_402674

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l4026_402674


namespace not_adjacent_in_sorted_consecutive_l4026_402609

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def sorted_by_digit_sum (a b : ℕ) : Prop :=
  (sum_of_digits a < sum_of_digits b) ∨ 
  (sum_of_digits a = sum_of_digits b ∧ a ≤ b)

theorem not_adjacent_in_sorted_consecutive (start : ℕ) : 
  ¬ ∃ i : ℕ, i < 99 ∧ 
    (sorted_by_digit_sum (start + i) 2010 ∧ sorted_by_digit_sum 2010 2011 ∧ sorted_by_digit_sum 2011 (start + (i + 1))) ∨
    (sorted_by_digit_sum (start + i) 2011 ∧ sorted_by_digit_sum 2011 2010 ∧ sorted_by_digit_sum 2010 (start + (i + 1))) :=
sorry

end not_adjacent_in_sorted_consecutive_l4026_402609


namespace sum_of_integers_l4026_402656

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 10) (h3 : x * y = 80) : x + y = 20 := by
  sorry

end sum_of_integers_l4026_402656


namespace puppies_per_cage_l4026_402651

theorem puppies_per_cage 
  (initial_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : initial_puppies = 18) 
  (h2 : sold_puppies = 3) 
  (h3 : num_cages = 3) 
  : (initial_puppies - sold_puppies) / num_cages = 5 := by
  sorry

end puppies_per_cage_l4026_402651


namespace callum_points_l4026_402671

theorem callum_points (total_matches : ℕ) (krishna_win_ratio : ℚ) (points_per_win : ℕ) : 
  total_matches = 8 →
  krishna_win_ratio = 3/4 →
  points_per_win = 10 →
  (total_matches - (krishna_win_ratio * total_matches).num) * points_per_win = 20 := by
  sorry

end callum_points_l4026_402671


namespace half_circle_roll_midpoint_path_length_l4026_402699

/-- The length of the path traveled by the midpoint of a half-circle's diameter when rolled along a straight line -/
theorem half_circle_roll_midpoint_path_length 
  (diameter : ℝ) 
  (h_diameter : diameter = 4 / Real.pi) : 
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let path_length := circumference / 2
  path_length = 2 := by sorry

end half_circle_roll_midpoint_path_length_l4026_402699
