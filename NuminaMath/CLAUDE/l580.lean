import Mathlib

namespace remove_one_gives_average_seven_point_five_l580_58019

def original_list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13]

def remove_element (l : List ℕ) (n : ℕ) : List ℕ :=
  l.filter (λ x => x ≠ n)

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem remove_one_gives_average_seven_point_five :
  average (remove_element original_list 1) = 7.5 := by
  sorry

end remove_one_gives_average_seven_point_five_l580_58019


namespace sequence_sum_1993_l580_58032

def sequence_sum (n : ℕ) : ℤ :=
  let group_sum := 5
  let num_groups := n / 5
  ↑num_groups * group_sum

theorem sequence_sum_1993 :
  sequence_sum 1993 = 1990 :=
by sorry

end sequence_sum_1993_l580_58032


namespace fraction_equality_implies_sum_l580_58088

theorem fraction_equality_implies_sum (A B : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 →
    (B * x - 23) / (x^2 - 9*x + 20) = A / (x - 4) + 5 / (x - 5)) →
  A + B = 11/9 := by
sorry

end fraction_equality_implies_sum_l580_58088


namespace number_problem_l580_58021

theorem number_problem (x : ℚ) : x - (3/5) * x = 64 → x = 160 := by
  sorry

end number_problem_l580_58021


namespace min_value_fraction_l580_58029

theorem min_value_fraction (x : ℝ) (h : x > 9) : 
  x^2 / (x - 9) ≥ 36 ∧ ∃ y > 9, y^2 / (y - 9) = 36 := by
  sorry

end min_value_fraction_l580_58029


namespace smallest_number_l580_58044

theorem smallest_number (S : Set ℤ) (h : S = {-3, 2, -2, 0}) : 
  ∃ m ∈ S, ∀ x ∈ S, m ≤ x ∧ m = -3 := by
  sorry

end smallest_number_l580_58044


namespace probability_two_defective_tubes_l580_58023

/-- The probability of selecting two defective tubes without replacement from a consignment of picture tubes -/
theorem probability_two_defective_tubes (total : ℕ) (defective : ℕ) 
  (h1 : total = 20) (h2 : defective = 5) (h3 : defective < total) :
  (defective : ℚ) / total * ((defective - 1) : ℚ) / (total - 1) = 1 / 19 := by
  sorry

end probability_two_defective_tubes_l580_58023


namespace max_n_is_largest_l580_58087

/-- Represents the sum of digits of a natural number -/
def S (a : ℕ) : ℕ := sorry

/-- Checks if all digits of a natural number are distinct -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- The maximum natural number satisfying the given conditions -/
def max_n : ℕ := 3210

theorem max_n_is_largest :
  ∀ n : ℕ, 
  has_distinct_digits n → 
  S (3 * n) = 3 * S n → 
  n ≤ max_n :=
sorry

end max_n_is_largest_l580_58087


namespace jake_fewer_than_steven_peach_difference_is_twelve_l580_58045

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 19

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := 7

/-- Jake has fewer peaches than Steven -/
theorem jake_fewer_than_steven : jake_peaches < steven_peaches := by sorry

/-- The difference between Steven's and Jake's peaches -/
def peach_difference : ℕ := steven_peaches - jake_peaches

/-- Prove that the difference between Steven's and Jake's peaches is 12 -/
theorem peach_difference_is_twelve : peach_difference = 12 := by sorry

end jake_fewer_than_steven_peach_difference_is_twelve_l580_58045


namespace point_on_line_l580_58056

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- A point lies on a line if and only if it can be expressed as a linear combination of two points on that line. -/
theorem point_on_line (O A B X : V) :
  (∃ t : ℝ, X - O = t • (A - O) + (1 - t) • (B - O)) ↔
  ∃ s : ℝ, X - A = s • (B - A) :=
sorry

end point_on_line_l580_58056


namespace det_A_squared_minus_2A_l580_58034

/-- Given a 2x2 matrix A, prove that det(A^2 - 2A) = 25 -/
theorem det_A_squared_minus_2A (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A = ![![1, 3], ![2, 1]]) : 
  Matrix.det (A ^ 2 - 2 • A) = 25 := by
sorry

end det_A_squared_minus_2A_l580_58034


namespace solution_range_l580_58074

-- Define the operation @
def op (p q : ℝ) : ℝ := p + q - p * q

-- Define the main theorem
theorem solution_range (m : ℝ) :
  (∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 
    op 2 (x₁ : ℝ) > 0 ∧ op (x₁ : ℝ) 3 ≤ m ∧
    op 2 (x₂ : ℝ) > 0 ∧ op (x₂ : ℝ) 3 ≤ m ∧
    (∀ (x : ℤ), x ≠ x₁ ∧ x ≠ x₂ → op 2 (x : ℝ) ≤ 0 ∨ op (x : ℝ) 3 > m)) →
  3 ≤ m ∧ m < 5 :=
sorry

end solution_range_l580_58074


namespace direct_proportional_function_points_l580_58060

/-- A direct proportional function passing through (2, -3) also passes through (4, -6) -/
theorem direct_proportional_function_points : ∃ (k : ℝ), k * 2 = -3 ∧ k * 4 = -6 := by
  sorry

end direct_proportional_function_points_l580_58060


namespace reflection_theorem_l580_58090

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflect a point with respect to another point -/
def reflect (p : Point3D) (center : Point3D) : Point3D :=
  { x := 2 * center.x - p.x,
    y := 2 * center.y - p.y,
    z := 2 * center.z - p.z }

/-- Perform a sequence of reflections -/
def reflectSequence (p : Point3D) (centers : List Point3D) : Point3D :=
  centers.foldl reflect p

theorem reflection_theorem (A O₁ O₂ O₃ : Point3D) :
  reflectSequence (reflectSequence A [O₁, O₂, O₃]) [O₁, O₂, O₃] = A := by
  sorry


end reflection_theorem_l580_58090


namespace valid_three_digit_numbers_count_l580_58064

/-- The count of three-digit numbers without exactly two identical adjacent digits -/
def validThreeDigitNumbers : ℕ :=
  let totalThreeDigitNumbers := 900
  let excludedNumbers := 162
  totalThreeDigitNumbers - excludedNumbers

/-- Theorem stating that the count of valid three-digit numbers is 738 -/
theorem valid_three_digit_numbers_count :
  validThreeDigitNumbers = 738 := by
  sorry

end valid_three_digit_numbers_count_l580_58064


namespace modular_arithmetic_problem_l580_58091

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), 
    (7 * a) % 80 = 1 ∧ 
    (13 * b) % 80 = 1 ∧ 
    ((3 * a + 9 * b) % 80) = 2 := by
  sorry

end modular_arithmetic_problem_l580_58091


namespace binomial_8_4_l580_58086

theorem binomial_8_4 : Nat.choose 8 4 = 70 := by
  sorry

end binomial_8_4_l580_58086


namespace complex_fraction_equation_l580_58025

theorem complex_fraction_equation (y : ℚ) : 
  3 + 1 / (1 + 1 / (3 + 3 / (4 + y))) = 169 / 53 → y = -605 / 119 := by
  sorry

end complex_fraction_equation_l580_58025


namespace least_n_for_inequality_l580_58099

theorem least_n_for_inequality : 
  (∀ n : ℕ, n > 0 → (1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15 → n ≥ 4) ∧ 
  ((1 : ℚ) / 4 - (1 : ℚ) / 5 < (1 : ℚ) / 15) := by
  sorry

end least_n_for_inequality_l580_58099


namespace smallest_integer_with_remainder_one_l580_58024

theorem smallest_integer_with_remainder_one : ∃ m : ℕ, 
  (m > 1) ∧ 
  (m % 5 = 1) ∧ 
  (m % 7 = 1) ∧ 
  (m % 3 = 1) ∧ 
  (∀ n : ℕ, n > 1 → n % 5 = 1 → n % 7 = 1 → n % 3 = 1 → m ≤ n) ∧
  (m = 106) := by
sorry

end smallest_integer_with_remainder_one_l580_58024


namespace base_10_to_base_7_l580_58016

theorem base_10_to_base_7 (n : ℕ) (h : n = 947) :
  ∃ (a b c d : ℕ),
    n = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
    a = 2 ∧ b = 5 ∧ c = 2 ∧ d = 2 :=
by sorry

end base_10_to_base_7_l580_58016


namespace maries_speed_l580_58059

/-- Given that Marie can bike 372 miles in 31 hours, prove that her speed is 12 miles per hour. -/
theorem maries_speed (distance : ℝ) (time : ℝ) (h1 : distance = 372) (h2 : time = 31) :
  distance / time = 12 := by
  sorry

end maries_speed_l580_58059


namespace max_sum_of_rolls_l580_58093

def is_valid_roll_set (rolls : List Nat) : Prop :=
  rolls.length = 24 ∧
  (∀ n : Nat, n ≥ 1 ∧ n ≤ 6 → n ∈ rolls) ∧
  (∀ n : Nat, n ≥ 2 ∧ n ≤ 6 → rolls.count 1 > rolls.count n)

def sum_of_rolls (rolls : List Nat) : Nat :=
  rolls.sum

theorem max_sum_of_rolls :
  ∀ rolls : List Nat,
    is_valid_roll_set rolls →
    sum_of_rolls rolls ≤ 90 :=
by sorry

end max_sum_of_rolls_l580_58093


namespace smallest_n_for_Q_less_than_threshold_l580_58017

def Q (n : ℕ) : ℚ := 4 / ((n + 2) * (n + 3))

theorem smallest_n_for_Q_less_than_threshold : 
  (∃ n : ℕ, Q n < 1/4022) ∧ 
  (∀ m : ℕ, m < 62 → Q m ≥ 1/4022) ∧ 
  Q 62 < 1/4022 := by
  sorry

end smallest_n_for_Q_less_than_threshold_l580_58017


namespace eccentricity_classification_l580_58057

theorem eccentricity_classification (x₁ x₂ : ℝ) : 
  2 * x₁^2 - 5 * x₁ + 2 = 0 →
  2 * x₂^2 - 5 * x₂ + 2 = 0 →
  x₁ ≠ x₂ →
  ((0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂) ∨ (0 < x₂ ∧ x₂ < 1 ∧ 1 < x₁)) :=
by sorry

end eccentricity_classification_l580_58057


namespace number_theory_problem_no_solution_2014_l580_58073

theorem number_theory_problem (a x y : ℕ+) (h : x ≠ y) :
  a * x + Nat.gcd a x + Nat.lcm a x ≠ a * y + Nat.gcd a y + Nat.lcm a y :=
by sorry

theorem no_solution_2014 :
  ¬∃ (a b : ℕ+), a * b + Nat.gcd a b + Nat.lcm a b = 2014 :=
by sorry

end number_theory_problem_no_solution_2014_l580_58073


namespace jacksons_email_deletion_l580_58097

theorem jacksons_email_deletion (initial_deletion : ℕ) (first_received : ℕ) 
  (second_received : ℕ) (final_received : ℕ) (final_inbox : ℕ) :
  initial_deletion = 50 →
  first_received = 15 →
  second_received = 5 →
  final_received = 10 →
  final_inbox = 30 →
  ∃ (second_deletion : ℕ), 
    second_deletion = 50 ∧
    final_inbox = first_received + second_received + final_received - initial_deletion - second_deletion :=
by sorry

end jacksons_email_deletion_l580_58097


namespace negation_of_implication_l580_58027

theorem negation_of_implication :
  (¬(∀ x : ℝ, x > 1 → x > 0)) ↔ (∀ x : ℝ, x ≤ 1 → x ≤ 0) := by
  sorry

end negation_of_implication_l580_58027


namespace choir_members_count_l580_58077

theorem choir_members_count : ∃! n : ℕ, 
  200 < n ∧ n < 300 ∧ 
  (∃ k : ℕ, n + 4 = 10 * k) ∧
  (∃ m : ℕ, n + 5 = 11 * m) := by
  sorry

end choir_members_count_l580_58077


namespace new_students_average_age_l580_58026

/-- Proves that the average age of new students is 32 years given the problem conditions. -/
theorem new_students_average_age
  (original_average : ℕ)
  (original_strength : ℕ)
  (new_students : ℕ)
  (average_decrease : ℕ)
  (h1 : original_average = 40)
  (h2 : original_strength = 12)
  (h3 : new_students = 12)
  (h4 : average_decrease = 4) :
  (original_average * original_strength + new_students * 32) / (original_strength + new_students) =
  original_average - average_decrease :=
by sorry


end new_students_average_age_l580_58026


namespace x_power_y_value_l580_58008

theorem x_power_y_value (x y : ℝ) (h : |x + 2*y| + (y - 3)^2 = 0) : x^y = -216 := by
  sorry

end x_power_y_value_l580_58008


namespace chocolate_purchase_l580_58066

theorem chocolate_purchase (boxes_bought : ℕ) (price_per_box : ℚ) (boxes_given : ℕ) 
  (pieces_per_box : ℕ) (discount_percent : ℚ) : 
  boxes_bought = 12 → 
  price_per_box = 4 → 
  boxes_given = 7 → 
  pieces_per_box = 6 → 
  discount_percent = 15 / 100 → 
  ∃ (amount_paid : ℚ) (pieces_remaining : ℕ), 
    amount_paid = 40.80 ∧ 
    pieces_remaining = 30 :=
by sorry

end chocolate_purchase_l580_58066


namespace door_rod_equation_l580_58047

theorem door_rod_equation (x : ℝ) : (x - 4)^2 + (x - 2)^2 = x^2 := by
  sorry

end door_rod_equation_l580_58047


namespace max_composite_sum_l580_58009

/-- A positive integer is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- The sum of a list of natural numbers. -/
def ListSum (L : List ℕ) : ℕ :=
  L.foldl (· + ·) 0

/-- A list of natural numbers is a valid decomposition if all its elements are composite
    and their sum is 2013. -/
def IsValidDecomposition (L : List ℕ) : Prop :=
  (∀ n ∈ L, IsComposite n) ∧ ListSum L = 2013

theorem max_composite_sum :
  (∃ L : List ℕ, IsValidDecomposition L ∧ L.length = 502) ∧
  (∀ L : List ℕ, IsValidDecomposition L → L.length ≤ 502) := by
  sorry

end max_composite_sum_l580_58009


namespace coefficient_of_quadratic_term_l580_58018

/-- The coefficient of the quadratic term in a quadratic equation ax^2 + bx + c = 0 -/
def quadratic_coefficient (a b c : ℝ) : ℝ := a

theorem coefficient_of_quadratic_term :
  quadratic_coefficient (-5) 5 6 = -5 := by sorry

end coefficient_of_quadratic_term_l580_58018


namespace least_number_with_remainders_l580_58084

theorem least_number_with_remainders : ∃! n : ℕ, 
  n > 0 ∧
  n % 34 = 4 ∧
  n % 48 = 6 ∧
  n % 5 = 2 ∧
  ∀ m : ℕ, m > 0 ∧ m % 34 = 4 ∧ m % 48 = 6 ∧ m % 5 = 2 → n ≤ m :=
by
  use 4082
  sorry

end least_number_with_remainders_l580_58084


namespace complex_square_simplification_l580_58080

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (5 - 3 * i)^2 = 16 - 30 * i :=
by sorry

end complex_square_simplification_l580_58080


namespace lisas_large_spoons_lisas_large_spoons_is_ten_l580_58089

/-- Calculates the number of large spoons in Lisa's new cutlery set -/
theorem lisas_large_spoons (num_children : ℕ) (baby_spoons_per_child : ℕ) 
  (decorative_spoons : ℕ) (new_teaspoons : ℕ) (total_spoons : ℕ) : ℕ :=
  let kept_spoons := num_children * baby_spoons_per_child + decorative_spoons
  let known_spoons := kept_spoons + new_teaspoons
  total_spoons - known_spoons

/-- Proves that the number of large spoons in Lisa's new cutlery set is 10 -/
theorem lisas_large_spoons_is_ten :
  lisas_large_spoons 4 3 2 15 39 = 10 := by
  sorry

end lisas_large_spoons_lisas_large_spoons_is_ten_l580_58089


namespace mean_of_smallest_elements_l580_58010

/-- The arithmetic mean of the smallest elements of all r-element subsets of {1, 2, ..., n} -/
def F (n r : ℕ) : ℚ :=
  sorry

/-- Theorem stating that F(n,r) = (n+1)/(r+1) for 1 ≤ r ≤ n -/
theorem mean_of_smallest_elements (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) :
  F n r = (n + 1 : ℚ) / (r + 1 : ℚ) := by
  sorry

end mean_of_smallest_elements_l580_58010


namespace pencil_problem_l580_58036

/-- Given the initial number of pencils, the number of containers, and the number of pencils that can be evenly distributed after receiving more, calculate the number of additional pencils received. -/
def additional_pencils (initial : ℕ) (containers : ℕ) (even_distribution : ℕ) : ℕ :=
  containers * even_distribution - initial

/-- Prove that given the specific conditions in the problem, the number of additional pencils is 30. -/
theorem pencil_problem : additional_pencils 150 5 36 = 30 := by
  sorry

end pencil_problem_l580_58036


namespace base_10_515_equals_base_6_2215_l580_58039

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (a b c d : ℕ) : ℕ :=
  a * 6^3 + b * 6^2 + c * 6^1 + d * 6^0

/-- Theorem stating that 515 in base 10 is equal to 2215 in base 6 --/
theorem base_10_515_equals_base_6_2215 :
  515 = base6ToBase10 2 2 1 5 := by
  sorry

end base_10_515_equals_base_6_2215_l580_58039


namespace sixth_term_equals_23_l580_58011

/-- Given a sequence with general term a(n) = 4n - 1, prove that a(6) = 23 -/
theorem sixth_term_equals_23 (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 1) : a 6 = 23 := by
  sorry

end sixth_term_equals_23_l580_58011


namespace prob_at_least_one_women_pair_l580_58094

/-- The number of young men in the group -/
def num_men : ℕ := 6

/-- The number of young women in the group -/
def num_women : ℕ := 6

/-- The total number of people in the group -/
def total_people : ℕ := num_men + num_women

/-- The number of pairs formed -/
def num_pairs : ℕ := total_people / 2

/-- The total number of ways to pair up the group -/
def total_pairings : ℕ := (total_people.factorial) / (2^num_pairs * num_pairs.factorial)

/-- The number of ways to pair up without any all-women pairs -/
def pairings_without_women_pairs : ℕ := num_women.factorial

/-- The probability of at least one pair consisting of two women -/
theorem prob_at_least_one_women_pair :
  (total_pairings - pairings_without_women_pairs) / total_pairings = 9675 / 10395 :=
sorry

end prob_at_least_one_women_pair_l580_58094


namespace arithmetic_sequence_inequality_l580_58076

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_inequality
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : d ≠ 0)
  (h3 : ∀ n : ℕ, a n > 0) :
  a 1 * a 8 < a 4 * a 5 := by
sorry

end arithmetic_sequence_inequality_l580_58076


namespace regular_pyramid_volume_l580_58096

-- Define the properties of the pyramid
structure RegularPyramid where
  l : ℝ  -- lateral edge length
  interior_angle_sum : ℝ  -- sum of interior angles of the base polygon
  lateral_angle : ℝ  -- angle between lateral edge and height

-- Define the theorem
theorem regular_pyramid_volume 
  (p : RegularPyramid) 
  (h1 : p.interior_angle_sum = 720)
  (h2 : p.lateral_angle = 30) : 
  ∃ (v : ℝ), v = (3 * p.l ^ 3) / 16 := by
  sorry

end regular_pyramid_volume_l580_58096


namespace r_amount_unchanged_l580_58040

/-- Represents the financial situation of three friends P, Q, and R. -/
structure FriendFinances where
  p : ℝ  -- Amount with P
  q : ℝ  -- Amount with Q
  r : ℝ  -- Amount with R

/-- The total amount among the three friends is 4000. -/
def total_amount (f : FriendFinances) : Prop :=
  f.p + f.q + f.r = 4000

/-- R has two-thirds of the total amount with P and Q. -/
def r_two_thirds_pq (f : FriendFinances) : Prop :=
  f.r = (2/3) * (f.p + f.q)

/-- The ratio of amount with P to amount with Q is 3:2. -/
def p_q_ratio (f : FriendFinances) : Prop :=
  f.p / f.q = 3/2

/-- 10% of P's amount will be donated to charity. -/
def charity_donation (f : FriendFinances) : ℝ :=
  0.1 * f.p

/-- Theorem stating that R's amount remains unchanged after P's charity donation. -/
theorem r_amount_unchanged (f : FriendFinances) 
  (h1 : total_amount f) 
  (h2 : r_two_thirds_pq f) 
  (h3 : p_q_ratio f) : 
  f.r = 1600 :=
sorry

end r_amount_unchanged_l580_58040


namespace proposition_false_iff_m_in_range_l580_58082

/-- The proposition is false for all real x when m is in [2,6) -/
theorem proposition_false_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, (m - 2) * x^2 + (m - 2) * x + 1 > 0) ↔ (2 ≤ m ∧ m < 6) :=
by sorry

end proposition_false_iff_m_in_range_l580_58082


namespace largest_four_digit_sum_20_l580_58079

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_four_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 20 → n ≤ 9920 :=
by sorry

end largest_four_digit_sum_20_l580_58079


namespace probability_of_one_out_of_four_l580_58051

theorem probability_of_one_out_of_four (S : Finset α) (h : S.card = 4) :
  ∀ a ∈ S, (1 : ℝ) / S.card = (1 : ℝ) / 4 := by
  sorry

end probability_of_one_out_of_four_l580_58051


namespace alyssa_grapes_cost_l580_58049

/-- The amount Alyssa paid for grapes -/
def grapesCost (totalSpent refund : ℚ) : ℚ := totalSpent + refund

/-- Proof that Alyssa paid $12.08 for grapes -/
theorem alyssa_grapes_cost : 
  let totalSpent : ℚ := 223/100
  let cherryRefund : ℚ := 985/100
  grapesCost totalSpent cherryRefund = 1208/100 := by
  sorry

end alyssa_grapes_cost_l580_58049


namespace teal_color_perception_l580_58020

theorem teal_color_perception (total : ℕ) (kinda_blue : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 150)
  (h2 : kinda_blue = 90)
  (h3 : both = 45)
  (h4 : neither = 25) :
  ∃ kinda_green : ℕ, kinda_green = 80 ∧ 
  kinda_green = total - (kinda_blue - both) - neither :=
by sorry

end teal_color_perception_l580_58020


namespace dave_remaining_candy_l580_58046

/-- The number of chocolate candy boxes Dave bought -/
def total_boxes : ℕ := 12

/-- The number of boxes Dave gave to his little brother -/
def given_boxes : ℕ := 5

/-- The number of candy pieces in each box -/
def pieces_per_box : ℕ := 3

/-- The number of candy pieces Dave still has -/
def remaining_pieces : ℕ := (total_boxes - given_boxes) * pieces_per_box

theorem dave_remaining_candy : remaining_pieces = 21 := by
  sorry

end dave_remaining_candy_l580_58046


namespace sequence_properties_l580_58069

def sequence_a (n : ℕ) : ℚ :=
  if n = 0 then 1 else 1 / (2 * n - 1)

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → 1 / (2 * sequence_a (n + 1)) = 1 / (2 * sequence_a n) + 1) →
  (∀ n : ℕ, n > 0 → 1 / sequence_a (n + 1) - 1 / sequence_a n = 2) ∧
  (∀ n : ℕ, n > 0 → sequence_a n = 1 / (2 * n - 1)) ∧
  (∀ n : ℕ, n > 0 → 
    (Finset.range n).sum (λ i => sequence_a i * sequence_a (i + 1)) = n / (2 * n + 1)) ∧
  (∀ n : ℕ, n > 0 → 
    ((Finset.range n).sum (λ i => sequence_a i * sequence_a (i + 1)) > 16 / 33) ↔ n > 16) :=
by sorry

end sequence_properties_l580_58069


namespace square_divisors_count_l580_58092

-- Define a function to count divisors
def count_divisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem square_divisors_count (n : ℕ) : 
  count_divisors n = 4 → count_divisors (n^2) = 7 := by sorry

end square_divisors_count_l580_58092


namespace pizza_topping_cost_l580_58068

/-- The cost per pizza in dollars -/
def cost_per_pizza : ℚ := 10

/-- The number of pizzas ordered -/
def num_pizzas : ℕ := 3

/-- The total number of toppings across all pizzas -/
def total_toppings : ℕ := 4

/-- The tip amount in dollars -/
def tip : ℚ := 5

/-- The total cost of the order including tip in dollars -/
def total_cost : ℚ := 39

/-- The cost per topping in dollars -/
def cost_per_topping : ℚ := 1

theorem pizza_topping_cost :
  cost_per_pizza * num_pizzas + cost_per_topping * total_toppings + tip = total_cost :=
sorry

end pizza_topping_cost_l580_58068


namespace nathan_basketball_games_l580_58037

/-- Calculates the number of basketball games played given the number of air hockey games,
    the cost per game, and the total tokens used. -/
def basketball_games (air_hockey_games : ℕ) (cost_per_game : ℕ) (total_tokens : ℕ) : ℕ :=
  (total_tokens - air_hockey_games * cost_per_game) / cost_per_game

/-- Proves that Nathan played 4 basketball games given the problem conditions. -/
theorem nathan_basketball_games :
  basketball_games 2 3 18 = 4 := by
  sorry

#eval basketball_games 2 3 18

end nathan_basketball_games_l580_58037


namespace geometric_sequence_min_value_l580_58052

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_min_value (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  (2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) →
  (∃ m : ℝ, m = 54 ∧ ∀ x : ℝ, 2 * a 8 + a 7 ≥ m) :=
sorry

end geometric_sequence_min_value_l580_58052


namespace sin_six_arcsin_one_third_l580_58013

theorem sin_six_arcsin_one_third :
  Real.sin (6 * Real.arcsin (1/3)) = 191 * Real.sqrt 2 / 729 := by
  sorry

end sin_six_arcsin_one_third_l580_58013


namespace tower_arrangements_l580_58050

def num_red_cubes : ℕ := 2
def num_blue_cubes : ℕ := 4
def num_green_cubes : ℕ := 3
def tower_height : ℕ := 8

def remaining_cubes : ℕ := tower_height - 1
def remaining_blue_cubes : ℕ := num_blue_cubes - 1
def remaining_red_cubes : ℕ := num_red_cubes
def remaining_green_cubes : ℕ := num_green_cubes - 1

theorem tower_arrangements :
  (remaining_cubes.factorial) / (remaining_blue_cubes.factorial * remaining_red_cubes.factorial * remaining_green_cubes.factorial) = 210 := by
  sorry

end tower_arrangements_l580_58050


namespace average_monthly_growth_rate_equation_l580_58070

/-- Represents the average monthly growth rate of profit from January to March -/
def monthly_growth_rate : ℝ → Prop :=
  fun x => 3 * (1 + x)^2 = 3.63

/-- The profit in January -/
def january_profit : ℝ := 30000

/-- The profit in March -/
def march_profit : ℝ := 36300

/-- Theorem stating the equation for the average monthly growth rate -/
theorem average_monthly_growth_rate_equation :
  ∃ x : ℝ, monthly_growth_rate x ∧
    march_profit = january_profit * (1 + x)^2 :=
  sorry

end average_monthly_growth_rate_equation_l580_58070


namespace equality_of_ratios_implies_equality_of_squares_l580_58078

theorem equality_of_ratios_implies_equality_of_squares
  (x y z : ℝ) (h : x / y = 3 / z) :
  9 * y^2 = x^2 * z^2 :=
by sorry

end equality_of_ratios_implies_equality_of_squares_l580_58078


namespace m_upper_bound_l580_58071

theorem m_upper_bound (f : ℝ → ℝ) (m : ℝ) :
  (∀ x > 0, f x = Real.exp x + Real.exp (-x)) →
  (∀ x > 0, m * f x ≤ Real.exp (-x) + m - 1) →
  m ≤ -1/3 := by
  sorry

end m_upper_bound_l580_58071


namespace expression_equals_five_l580_58007

theorem expression_equals_five :
  (π + Real.sqrt 3) ^ 0 + (-2) ^ 2 + |(-1/2)| - Real.sin (30 * π / 180) = 5 := by
  sorry

end expression_equals_five_l580_58007


namespace interest_rate_20_percent_l580_58058

-- Define the compound interest function
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r) ^ t

-- State the theorem
theorem interest_rate_20_percent 
  (P : ℝ) 
  (r : ℝ) 
  (h1 : compound_interest P r 3 = 3000) 
  (h2 : compound_interest P r 4 = 3600) :
  r = 0.2 :=
sorry

end interest_rate_20_percent_l580_58058


namespace joan_has_six_balloons_l580_58033

/-- The number of orange balloons Joan has now, given she initially had 8 and lost 2. -/
def joans_balloons : ℕ := 8 - 2

/-- Theorem stating that Joan has 6 orange balloons now. -/
theorem joan_has_six_balloons : joans_balloons = 6 := by
  sorry

end joan_has_six_balloons_l580_58033


namespace sticker_count_l580_58054

/-- Given a number of stickers per page and a number of pages, 
    calculate the total number of stickers -/
def total_stickers (stickers_per_page : ℕ) (num_pages : ℕ) : ℕ :=
  stickers_per_page * num_pages

/-- Theorem: The total number of stickers is 220 when there are 10 stickers per page and 22 pages -/
theorem sticker_count : total_stickers 10 22 = 220 := by
  sorry

end sticker_count_l580_58054


namespace solve_scooter_price_l580_58030

def scooter_price_problem (upfront_percentage : ℚ) (upfront_amount : ℚ) (num_installments : ℕ) : Prop :=
  let total_price : ℚ := upfront_amount / upfront_percentage * 100
  let remaining_amount : ℚ := total_price * (1 - upfront_percentage)
  let installment_amount : ℚ := remaining_amount / num_installments
  (upfront_percentage = 20/100) ∧ 
  (upfront_amount = 240) ∧ 
  (num_installments = 12) ∧
  (total_price = 1200) ∧ 
  (installment_amount = 80)

theorem solve_scooter_price : 
  ∃ (upfront_percentage : ℚ) (upfront_amount : ℚ) (num_installments : ℕ),
    scooter_price_problem upfront_percentage upfront_amount num_installments :=
by
  sorry

end solve_scooter_price_l580_58030


namespace dave_trays_first_table_l580_58075

/-- The number of trays Dave can carry per trip -/
def trays_per_trip : ℕ := 9

/-- The number of trips Dave made -/
def total_trips : ℕ := 8

/-- The number of trays Dave picked up from the second table -/
def trays_from_second_table : ℕ := 55

/-- The number of trays Dave picked up from the first table -/
def trays_from_first_table : ℕ := trays_per_trip * total_trips - trays_from_second_table

theorem dave_trays_first_table : trays_from_first_table = 17 := by
  sorry

end dave_trays_first_table_l580_58075


namespace remainder_three_divisor_l580_58041

theorem remainder_three_divisor (n : ℕ) (h : n = 1680) (h9 : n % 9 = 0) :
  ∃ m : ℕ, m = 1677 ∧ n % m = 3 :=
by sorry

end remainder_three_divisor_l580_58041


namespace cosine_product_sqrt_value_l580_58004

theorem cosine_product_sqrt_value :
  Real.sqrt ((3 - Real.cos (π / 9) ^ 2) * (3 - Real.cos (2 * π / 9) ^ 2) * (3 - Real.cos (4 * π / 9) ^ 2)) = 9 * Real.sqrt 3 / 8 := by
  sorry

end cosine_product_sqrt_value_l580_58004


namespace mark_tree_count_l580_58028

/-- Calculates the final number of trees after planting and removing sessions -/
def final_tree_count (x y : ℕ) (plant_rate remove_rate : ℕ) : ℤ :=
  let days : ℕ := y / plant_rate
  let removed : ℕ := days * remove_rate
  (x : ℤ) + (y : ℤ) - (removed : ℤ)

/-- Theorem stating the final number of trees after Mark's planting session -/
theorem mark_tree_count (x : ℕ) : final_tree_count x 12 2 3 = (x : ℤ) - 6 := by
  sorry

end mark_tree_count_l580_58028


namespace divisor_exists_l580_58095

def N : ℕ := 123456789101112131415161718192021222324252627282930313233343536373839404142434481

theorem divisor_exists : ∃ D : ℕ, D > 0 ∧ N % D = 36 := by
  sorry

end divisor_exists_l580_58095


namespace solution_set_f_geq_2_min_value_f_l580_58005

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

-- Theorem for the solution set of f(x) ≥ 2
theorem solution_set_f_geq_2 :
  {x : ℝ | f x ≥ 2} = {x : ℝ | x ≤ -7} ∪ {x : ℝ | x ≥ 5/3} := by sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (x : ℝ), f x = -9/2 ∧ ∀ (y : ℝ), f y ≥ -9/2 := by sorry

end solution_set_f_geq_2_min_value_f_l580_58005


namespace tile_ratio_l580_58015

theorem tile_ratio (total : Nat) (yellow purple white : Nat)
  (h_total : total = 20)
  (h_yellow : yellow = 3)
  (h_purple : purple = 6)
  (h_white : white = 7) :
  (total - (yellow + purple + white)) / yellow = 4 / 3 := by
  sorry

end tile_ratio_l580_58015


namespace sin_30_degrees_l580_58098

open Real

theorem sin_30_degrees :
  sin (π / 6) = 1 / 2 :=
by
  sorry

end sin_30_degrees_l580_58098


namespace deposit_percentage_l580_58022

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) :
  deposit = 50 →
  remaining = 950 →
  (deposit / (deposit + remaining)) * 100 = 5 := by
sorry

end deposit_percentage_l580_58022


namespace cubic_root_sum_l580_58003

theorem cubic_root_sum (p q r : ℝ) : 
  (p^3 - 2*p - 2 = 0) → 
  (q^3 - 2*q - 2 = 0) → 
  (r^3 - 2*r - 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -24 := by sorry

end cubic_root_sum_l580_58003


namespace prop_2_prop_4_l580_58001

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Theorem for proposition ②
theorem prop_2 (l : Line) (α β : Plane) :
  perpendicular l α → parallel l β → plane_perpendicular α β := by sorry

-- Theorem for proposition ④
theorem prop_4 (α β γ : Plane) :
  plane_perpendicular α β → plane_parallel α γ → plane_perpendicular γ β := by sorry

end prop_2_prop_4_l580_58001


namespace fraction_calculation_l580_58085

theorem fraction_calculation : (8 / 15 - 7 / 9) + 3 / 4 = 1 / 2 := by
  sorry

end fraction_calculation_l580_58085


namespace intersection_implies_sum_l580_58062

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 4*x < 0}
def N (m : ℝ) : Set ℝ := {x | m < x ∧ x < 5}

-- State the theorem
theorem intersection_implies_sum (m n : ℝ) : 
  M ∩ N m = {x | 3 < x ∧ x < n} → m + n = 7 := by
  sorry

end intersection_implies_sum_l580_58062


namespace mixed_groups_count_l580_58006

/-- Represents the chess club structure and game results -/
structure ChessClub where
  total_children : ℕ
  total_groups : ℕ
  children_per_group : ℕ
  boy_vs_boy_games : ℕ
  girl_vs_girl_games : ℕ

/-- Calculates the number of mixed groups in the chess club -/
def mixed_groups (club : ChessClub) : ℕ :=
  let total_games := club.total_groups * (club.children_per_group.choose 2)
  let mixed_games := total_games - club.boy_vs_boy_games - club.girl_vs_girl_games
  mixed_games / 2

/-- Theorem stating that the number of mixed groups is 23 -/
theorem mixed_groups_count (club : ChessClub) 
  (h1 : club.total_children = 90)
  (h2 : club.total_groups = 30)
  (h3 : club.children_per_group = 3)
  (h4 : club.boy_vs_boy_games = 30)
  (h5 : club.girl_vs_girl_games = 14) :
  mixed_groups club = 23 := by
  sorry

#eval mixed_groups ⟨90, 30, 3, 30, 14⟩

end mixed_groups_count_l580_58006


namespace f_two_eq_zero_iff_r_eq_neg_38_l580_58002

/-- The function f(x) as defined in the problem -/
def f (x r : ℝ) : ℝ := 2 * x^4 + x^3 + x^2 - 3 * x + r

/-- Theorem stating that f(2) = 0 if and only if r = -38 -/
theorem f_two_eq_zero_iff_r_eq_neg_38 : ∀ r : ℝ, f 2 r = 0 ↔ r = -38 := by sorry

end f_two_eq_zero_iff_r_eq_neg_38_l580_58002


namespace absolute_value_inequality_implies_range_l580_58031

theorem absolute_value_inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, |2*x - 1| + |x + 2| ≥ a^2 + (1/2)*a + 2) →
  -1 ≤ a ∧ a ≤ 1/2 := by
  sorry

end absolute_value_inequality_implies_range_l580_58031


namespace nonagon_diagonals_l580_58014

/-- The number of distinct diagonals in a convex nonagon -/
def diagonals_in_nonagon : ℕ := 27

/-- A convex polygon with 9 sides -/
structure Nonagon where
  sides : ℕ
  convex : Bool
  is_nonagon : sides = 9 ∧ convex = true

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonals (n : Nonagon) : diagonals_in_nonagon = 27 := by
  sorry

end nonagon_diagonals_l580_58014


namespace quadratic_factorization_l580_58061

theorem quadratic_factorization (b : ℤ) : 
  (∃ (c d e f : ℤ), 45 * y^2 + b * y + 45 = (c * y + d) * (e * y + f)) →
  ∃ (k : ℤ), b = 2 * k :=
by sorry

end quadratic_factorization_l580_58061


namespace floor_sum_equals_four_l580_58043

theorem floor_sum_equals_four (x y : ℝ) : 
  (⌊x⌋^2 + ⌊y⌋^2 = 4) ↔ 
  ((2 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 1) ∨
   (-2 ≤ x ∧ x < -1 ∧ 0 ≤ y ∧ y < 1) ∨
   (0 ≤ x ∧ x < 1 ∧ 2 ≤ y ∧ y < 3) ∨
   (0 ≤ x ∧ x < 1 ∧ -2 ≤ y ∧ y < -1)) :=
by sorry

end floor_sum_equals_four_l580_58043


namespace fraction_arithmetic_l580_58048

theorem fraction_arithmetic : (2 : ℚ) / 9 * 4 / 5 - 1 / 45 = 7 / 45 := by
  sorry

end fraction_arithmetic_l580_58048


namespace square_division_correct_l580_58067

/-- Represents the state of the square division process after n iterations -/
structure SquareDivision (n : ℕ) where
  /-- The number of remaining squares -/
  remaining_squares : ℕ
  /-- The side length of each remaining square -/
  side_length : ℚ
  /-- The total area of removed squares -/
  removed_area : ℚ

/-- The result of the square division process after n iterations -/
def square_division_result (n : ℕ) : SquareDivision n :=
  { remaining_squares := 8^n,
    side_length := 1 / 3^n,
    removed_area := 1 - (8/9)^n }

/-- Theorem stating the correctness of the square division result -/
theorem square_division_correct (n : ℕ) :
  (square_division_result n).remaining_squares = 8^n ∧
  (square_division_result n).side_length = 1 / 3^n ∧
  (square_division_result n).removed_area = 1 - (8/9)^n :=
by sorry

end square_division_correct_l580_58067


namespace sum_of_roots_equals_p_l580_58081

/-- Given a quadratic equation x^2 - px + 2q = 0 where p and q are its roots and both non-zero,
    the sum of the roots is equal to p. -/
theorem sum_of_roots_equals_p (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
    (h : ∀ x, x^2 - p*x + 2*q = 0 ↔ x = p ∨ x = q) : 
  p + q = p := by
  sorry

end sum_of_roots_equals_p_l580_58081


namespace petes_number_l580_58000

theorem petes_number (x : ℝ) : 5 * (3 * x - 6) = 195 → x = 15 := by
  sorry

end petes_number_l580_58000


namespace original_height_is_100_l580_58035

/-- The rebound factor of the ball -/
def rebound_factor : ℝ := 0.5

/-- The total travel distance when the ball touches the floor for the third time -/
def total_distance : ℝ := 250

/-- Calculates the total travel distance for a ball dropped from height h -/
def calculate_total_distance (h : ℝ) : ℝ :=
  h + 2 * h * rebound_factor + 2 * h * rebound_factor^2

/-- Theorem stating that the original height is 100 cm -/
theorem original_height_is_100 :
  ∃ h : ℝ, h > 0 ∧ calculate_total_distance h = total_distance ∧ h = 100 :=
sorry

end original_height_is_100_l580_58035


namespace grid_walk_probability_l580_58038

def grid_walk (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * (grid_walk (x-1) y + grid_walk x (y-1) + grid_walk (x-1) (y-1))

theorem grid_walk_probability :
  ∃ (m n : ℕ), 
    m > 0 ∧ 
    n > 0 ∧ 
    ¬(3 ∣ m) ∧ 
    grid_walk 5 5 = m / (3^n : ℚ) ∧ 
    m + n = 1186 :=
sorry

end grid_walk_probability_l580_58038


namespace bill_amount_correct_l580_58012

/-- The amount of the bill in dollars -/
def bill_amount : ℝ := 26

/-- The percentage a bad tipper tips -/
def bad_tip_percent : ℝ := 0.05

/-- The percentage a good tipper tips -/
def good_tip_percent : ℝ := 0.20

/-- The difference between a good tip and a bad tip in dollars -/
def tip_difference : ℝ := 3.90

theorem bill_amount_correct : 
  (good_tip_percent - bad_tip_percent) * bill_amount = tip_difference := by
  sorry

end bill_amount_correct_l580_58012


namespace max_n_for_coprime_with_prime_l580_58065

/-- A function that checks if a list of integers is pairwise coprime -/
def IsPairwiseCoprime (list : List Int) : Prop :=
  ∀ i j, i ≠ j → i ∈ list → j ∈ list → Int.gcd i j = 1

/-- A function that checks if a number is prime -/
def IsPrime (n : Int) : Prop :=
  n > 1 ∧ ∀ m, 1 < m → m < n → ¬(n % m = 0)

/-- The main theorem -/
theorem max_n_for_coprime_with_prime : 
  (∀ (list : List Int), list.length = 5 → 
    (∀ x ∈ list, x ≥ 1 ∧ x ≤ 48) → 
    IsPairwiseCoprime list → 
    (∃ x ∈ list, IsPrime x)) ∧ 
  (∃ (list : List Int), list.length = 5 ∧ 
    (∀ x ∈ list, x ≥ 1 ∧ x ≤ 49) ∧ 
    IsPairwiseCoprime list ∧ 
    (∀ x ∈ list, ¬IsPrime x)) := by
  sorry

end max_n_for_coprime_with_prime_l580_58065


namespace minimum_cards_to_turn_l580_58053

/-- Represents a card with a letter on one side and a number on the other -/
structure Card where
  letter : Char
  number : Nat

/-- Checks if a character is a vowel -/
def isVowel (c : Char) : Bool :=
  c ∈ ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']

/-- Checks if a number is even -/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/-- Checks if a card satisfies the condition: 
    if it has a vowel, it must have an even number -/
def satisfiesCondition (card : Card) : Bool :=
  ¬(isVowel card.letter) || isEven card.number

/-- Represents the set of cards on the table -/
def cardSet : Finset Card := sorry

/-- The number of cards that need to be turned over -/
def cardsToTurn : Nat := sorry

theorem minimum_cards_to_turn : 
  (∀ card ∈ cardSet, satisfiesCondition card) → cardsToTurn = 3 := by
  sorry

end minimum_cards_to_turn_l580_58053


namespace max_triangle_side_length_l580_58072

theorem max_triangle_side_length (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- Three different side lengths
  a + b + c = 30 →         -- Perimeter is 30
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a + b > c ∧ b + c > a ∧ a + c > b →  -- Triangle inequality
  a ≤ 14 ∧ b ≤ 14 ∧ c ≤ 14 :=
by sorry

#check max_triangle_side_length

end max_triangle_side_length_l580_58072


namespace business_value_l580_58063

theorem business_value (man_share : ℚ) (sold_fraction : ℚ) (sale_price : ℕ) :
  man_share = 1/3 →
  sold_fraction = 3/5 →
  sale_price = 15000 →
  (sale_price : ℚ) / sold_fraction / man_share = 75000 :=
by sorry

end business_value_l580_58063


namespace smallest_inverse_undefined_twenty_two_satisfies_smallest_inverse_undefined_is_22_l580_58055

theorem smallest_inverse_undefined (a : ℕ) : a > 0 ∧ 
  (∀ x : ℕ, x * a % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * a % 77 ≠ 1) → 
  a ≥ 22 := by
sorry

theorem twenty_two_satisfies : 
  (∀ x : ℕ, x * 22 % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * 22 % 77 ≠ 1) := by
sorry

theorem smallest_inverse_undefined_is_22 : 
  ∃! a : ℕ, a > 0 ∧ 
  (∀ x : ℕ, x * a % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * a % 77 ≠ 1) ∧ 
  ∀ b : ℕ, b > 0 ∧ 
  (∀ x : ℕ, x * b % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * b % 77 ≠ 1) → 
  a ≤ b := by
sorry

end smallest_inverse_undefined_twenty_two_satisfies_smallest_inverse_undefined_is_22_l580_58055


namespace exactly_two_solutions_l580_58042

/-- The number of integer solutions to the given system of equations -/
def num_solutions : ℕ := 2

/-- The system of equations -/
def system (x y z : ℤ) : Prop :=
  x^2 - 4*x*y + 3*y^2 - z^2 = 40 ∧
  -x^2 + 4*y*z + 3*z^2 = 47 ∧
  x^2 + 2*x*y + 9*z^2 = 110

/-- Theorem stating that there are exactly 2 solutions to the system -/
theorem exactly_two_solutions :
  (∃! (solutions : Finset (ℤ × ℤ × ℤ)), solutions.card = num_solutions ∧
    ∀ (x y z : ℤ), (x, y, z) ∈ solutions ↔ system x y z) :=
sorry

end exactly_two_solutions_l580_58042


namespace triangle_inequality_check_l580_58083

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_inequality_check :
  can_form_triangle 9 6 13 ∧
  ¬(can_form_triangle 6 8 16) ∧
  ¬(can_form_triangle 18 9 8) ∧
  ¬(can_form_triangle 3 5 9) :=
sorry

end triangle_inequality_check_l580_58083
