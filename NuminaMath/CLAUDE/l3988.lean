import Mathlib

namespace cube_root_cubed_equals_identity_l3988_398848

theorem cube_root_cubed_equals_identity (x : ℝ) : (x^(1/3))^3 = x := by sorry

end cube_root_cubed_equals_identity_l3988_398848


namespace smallest_with_2023_divisors_l3988_398872

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Checks if a number is divisible by another -/
def is_divisible (a b : ℕ) : Prop := sorry

theorem smallest_with_2023_divisors :
  ∃ (n m k : ℕ),
    n > 0 ∧
    num_divisors n = 2023 ∧
    n = m * 6^k ∧
    ¬ is_divisible m 6 ∧
    (∀ (n' m' k' : ℕ),
      n' > 0 →
      num_divisors n' = 2023 →
      n' = m' * 6^k' →
      ¬ is_divisible m' 6 →
      n ≤ n') ∧
    m + k = 745 :=
  sorry

end smallest_with_2023_divisors_l3988_398872


namespace triangle_x_theorem_l3988_398857

/-- A function that checks if three side lengths can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of positive integer values of x for which a triangle with sides 5, 12, and x^2 exists -/
def triangle_x_values : Set ℕ+ :=
  {x : ℕ+ | is_triangle 5 12 (x.val ^ 2)}

theorem triangle_x_theorem : triangle_x_values = {3, 4} := by
  sorry

end triangle_x_theorem_l3988_398857


namespace operation_on_81_divided_by_3_l3988_398860

theorem operation_on_81_divided_by_3 : ∃ f : ℝ → ℝ, (f 81) / 3 = 3 := by sorry

end operation_on_81_divided_by_3_l3988_398860


namespace quadratic_polynomial_satisfies_conditions_l3988_398889

-- Define the quadratic polynomial p(x)
def p (x : ℚ) : ℚ := (12/7) * x^2 + (36/7) * x - 216/7

-- Theorem stating that p(x) satisfies the given conditions
theorem quadratic_polynomial_satisfies_conditions :
  p (-6) = 0 ∧ p 3 = 0 ∧ p 1 = -24 :=
by sorry

end quadratic_polynomial_satisfies_conditions_l3988_398889


namespace intersection_area_implies_m_values_l3988_398859

def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

def Line (m x y : ℝ) : Prop := x - m*y + 2 = 0

def AreaABO (m : ℝ) : ℝ := 2

theorem intersection_area_implies_m_values (m : ℝ) :
  (∃ A B : ℝ × ℝ, 
    Circle A.1 A.2 ∧ Circle B.1 B.2 ∧ 
    Line m A.1 A.2 ∧ Line m B.1 B.2 ∧
    AreaABO m = 2) →
  m = 1 ∨ m = -1 :=
sorry

end intersection_area_implies_m_values_l3988_398859


namespace range_of_m_range_of_x_l3988_398858

-- Define the function f
def f (m x : ℝ) : ℝ := m * x^2 - m * x - 6 + m

-- Part 1
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x < 0) → m < 6/7 :=
sorry

-- Part 2
theorem range_of_x (x : ℝ) :
  (∀ m ∈ Set.Icc (-2) 2, f m x < 0) → -1 < x ∧ x < 2 :=
sorry

end range_of_m_range_of_x_l3988_398858


namespace polynomial_factorization_l3988_398846

/-- A polynomial in x and y with a parameter n -/
def polynomial (n : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + 2*x + n*y - n

/-- Predicate to check if a polynomial can be factored into two linear factors with integer coefficients -/
def has_linear_factors (p : ℤ → ℤ → ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), ∀ x y, p x y = (a*x + b*y + c) * (d*x + e*y + f)

theorem polynomial_factorization (n : ℤ) :
  has_linear_factors (polynomial n) ↔ n = 0 :=
sorry

end polynomial_factorization_l3988_398846


namespace reflection_about_y_axis_example_l3988_398824

/-- Given a point in 3D space, return its reflection about the y-axis -/
def reflect_about_y_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, -z)

/-- The reflection of point (3, -2, 1) about the y-axis is (-3, -2, -1) -/
theorem reflection_about_y_axis_example : 
  reflect_about_y_axis (3, -2, 1) = (-3, -2, -1) := by
  sorry

#check reflection_about_y_axis_example

end reflection_about_y_axis_example_l3988_398824


namespace self_repeating_mod_1000_numbers_l3988_398850

/-- A three-digit number that remains unchanged when raised to any natural power modulo 1000 -/
def self_repeating_mod_1000 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∀ k : ℕ, k > 0 → n^k ≡ n [MOD 1000]

/-- The only three-digit numbers that remain unchanged when raised to any natural power modulo 1000 are 625 and 376 -/
theorem self_repeating_mod_1000_numbers :
  ∀ n : ℕ, self_repeating_mod_1000 n ↔ n = 625 ∨ n = 376 := by sorry

end self_repeating_mod_1000_numbers_l3988_398850


namespace long_jump_challenge_l3988_398802

/-- Represents a student in the long jump challenge -/
structure Student where
  success_prob : ℚ
  deriving Repr

/-- Calculates the probability of a student achieving excellence -/
def excellence_prob (s : Student) : ℚ :=
  s.success_prob + (1 - s.success_prob) * s.success_prob

/-- Calculates the probability of a student achieving a good rating -/
def good_prob (s : Student) : ℚ :=
  1 - excellence_prob s

/-- The probability that exactly two out of three students achieve a good rating -/
def prob_two_good (s1 s2 s3 : Student) : ℚ :=
  excellence_prob s1 * good_prob s2 * good_prob s3 +
  good_prob s1 * excellence_prob s2 * good_prob s3 +
  good_prob s1 * good_prob s2 * excellence_prob s3

theorem long_jump_challenge (s1 s2 s3 : Student)
  (h1 : s1.success_prob = 3/4)
  (h2 : s2.success_prob = 1/2)
  (h3 : s3.success_prob = 1/3) :
  prob_two_good s1 s2 s3 = 77/576 := by
  sorry

#eval prob_two_good ⟨3/4⟩ ⟨1/2⟩ ⟨1/3⟩

end long_jump_challenge_l3988_398802


namespace original_avg_age_l3988_398856

def original_class_size : ℕ := 8
def new_students_size : ℕ := 8
def new_students_avg_age : ℕ := 32
def avg_age_decrease : ℕ := 4

theorem original_avg_age (original_avg : ℕ) :
  (original_avg * original_class_size + new_students_avg_age * new_students_size) / 
  (original_class_size + new_students_size) = original_avg - avg_age_decrease →
  original_avg = 40 :=
by sorry

end original_avg_age_l3988_398856


namespace conference_handshakes_l3988_398867

/-- The number of handshakes in a conference where each person shakes hands with every other person exactly once. -/
def num_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that in a conference of 10 people where each person shakes hands with every other person exactly once, there are 45 handshakes. -/
theorem conference_handshakes : num_handshakes 10 = 45 := by
  sorry

end conference_handshakes_l3988_398867


namespace quadratic_function_unique_form_l3988_398879

/-- Given a quadratic function f(x) = x^2 + ax + b that intersects the x-axis at (1, 0) 
    and has its axis of symmetry at x = 2, prove that f(x) = x^2 - 4x + 3 -/
theorem quadratic_function_unique_form (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^2 + a*x + b)
    (h2 : f 1 = 0)
    (h3 : -a/2 = 2) : 
  ∀ x, f x = x^2 - 4*x + 3 := by
sorry


end quadratic_function_unique_form_l3988_398879


namespace states_fraction_l3988_398840

/-- Given 30 total states and 15 states joining during a specific decade,
    prove that the fraction of states joining in that decade is 1/2. -/
theorem states_fraction (total_states : ℕ) (decade_states : ℕ) 
    (h1 : total_states = 30) 
    (h2 : decade_states = 15) : 
    (decade_states : ℚ) / total_states = 1 / 2 := by
  sorry

end states_fraction_l3988_398840


namespace discount_percentage_proof_l3988_398838

theorem discount_percentage_proof (total_cost : ℝ) (num_shirts : ℕ) (discounted_price : ℝ) :
  total_cost = 60 ∧ num_shirts = 3 ∧ discounted_price = 12 →
  (1 - discounted_price / (total_cost / num_shirts)) * 100 = 40 := by
  sorry

end discount_percentage_proof_l3988_398838


namespace point_trajectory_l3988_398897

/-- The trajectory of a point P satisfying |PA| + |PB| = 5, where A(0,0) and B(5,0) are fixed points -/
theorem point_trajectory (P : ℝ × ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (5, 0)
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 5 →
  P.2 = 0 ∧ 0 ≤ P.1 ∧ P.1 ≤ 5 := by
  sorry

end point_trajectory_l3988_398897


namespace x_squared_plus_inverse_l3988_398855

theorem x_squared_plus_inverse (x : ℝ) (h : 47 = x^6 + 1 / x^6) : x^2 + 1 / x^2 = 9 := by
  sorry

end x_squared_plus_inverse_l3988_398855


namespace triangle_problem_l3988_398820

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_problem (a b c : ℝ) 
  (h : Real.sqrt (8 - a) + Real.sqrt (a - 8) = abs (c - 17) + b^2 - 30*b + 225) :
  a = 8 ∧ 
  b = 15 ∧ 
  c = 17 ∧
  triangle_inequality a b c ∧
  a^2 + b^2 = c^2 ∧
  a + b + c = 40 ∧
  a * b / 2 = 60 := by
  sorry

end triangle_problem_l3988_398820


namespace sqrt_difference_equals_seven_l3988_398853

theorem sqrt_difference_equals_seven : Real.sqrt (36 + 64) - Real.sqrt (25 - 16) = 7 := by
  sorry

end sqrt_difference_equals_seven_l3988_398853


namespace problem_solution_l3988_398886

theorem problem_solution (x : ℝ) : 
  (x - 9)^3 / (x + 4) = 27 → (x^2 - 12*x + 15) / (x - 2) = -20.1 := by
sorry

end problem_solution_l3988_398886


namespace yard_length_theorem_l3988_398845

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 14 trees planted at equal distances, 
    with one tree at each end, and a distance of 21 meters between consecutive trees, 
    is equal to 273 meters. -/
theorem yard_length_theorem : 
  yard_length 14 21 = 273 := by
  sorry

end yard_length_theorem_l3988_398845


namespace class_test_average_l3988_398827

theorem class_test_average (class_size : ℝ) (h_positive : class_size > 0) : 
  let group_a := 0.15 * class_size
  let group_b := 0.50 * class_size
  let group_c := class_size - group_a - group_b
  let score_a := 100
  let score_b := 78
  ∃ score_c : ℝ,
    (group_a * score_a + group_b * score_b + group_c * score_c) / class_size = 76.05 ∧
    score_c = 63 :=
by sorry

end class_test_average_l3988_398827


namespace factorial_sum_equation_l3988_398854

theorem factorial_sum_equation (x y : ℕ) (z : ℤ) 
  (h_odd : ∃ k : ℤ, z = 2 * k + 1)
  (h_eq : x.factorial + y.factorial = 48 * z + 2017) :
  ((x = 1 ∧ y = 6 ∧ z = -27) ∨
   (x = 6 ∧ y = 1 ∧ z = -27) ∨
   (x = 1 ∧ y = 7 ∧ z = 63) ∨
   (x = 7 ∧ y = 1 ∧ z = 63)) :=
by sorry

end factorial_sum_equation_l3988_398854


namespace expression_simplification_l3988_398844

theorem expression_simplification :
  ∀ q : ℚ, ((7*q+3)-3*q*2)*4+(5-2/4)*(8*q-12) = 40*q - 42 := by
  sorry

end expression_simplification_l3988_398844


namespace elena_book_purchase_l3988_398830

theorem elena_book_purchase (total_money : ℝ) (total_books : ℕ) (book_price : ℝ) 
  (h1 : book_price > 0) 
  (h2 : total_books > 0)
  (h3 : total_money / 3 = (total_books / 2 : ℝ) * book_price) : 
  total_money - total_books * book_price = total_money / 3 := by
sorry

end elena_book_purchase_l3988_398830


namespace equation_solution_l3988_398833

theorem equation_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + 1 = d + Real.sqrt (a + b + c - d) →
  d = 5/4 := by
sorry

end equation_solution_l3988_398833


namespace rectangle_area_l3988_398835

theorem rectangle_area (side : ℝ) (diagonal : ℝ) (area : ℝ) : 
  side = 15 → diagonal = 17 → area = side * (Real.sqrt (diagonal^2 - side^2)) → area = 120 :=
by sorry

end rectangle_area_l3988_398835


namespace consecutive_negative_integers_product_sum_l3988_398849

theorem consecutive_negative_integers_product_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = -105 := by
  sorry

end consecutive_negative_integers_product_sum_l3988_398849


namespace quadratic_completion_square_l3988_398869

theorem quadratic_completion_square (x : ℝ) :
  4 * x^2 - 8 * x - 128 = 0 →
  ∃ (r : ℝ), (x + r)^2 = 33 :=
by
  sorry

end quadratic_completion_square_l3988_398869


namespace a_value_l3988_398899

def A (a : ℝ) : Set ℝ := {-1, a}
def B (a b : ℝ) : Set ℝ := {3^a, b}

theorem a_value (a b : ℝ) :
  A a ∪ B a b = {-1, 0, 1} → a = 0 := by
  sorry

end a_value_l3988_398899


namespace sqrt_product_equality_l3988_398884

theorem sqrt_product_equality : Real.sqrt 48 * Real.sqrt 27 * Real.sqrt 8 * Real.sqrt 3 = 72 * Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l3988_398884


namespace joanne_coins_l3988_398811

def coins_problem (first_hour : ℕ) (next_two_hours : ℕ) (fourth_hour : ℕ) (total_after : ℕ) : Prop :=
  let total_collected := first_hour + 2 * next_two_hours + fourth_hour
  total_collected - total_after = 15

theorem joanne_coins : coins_problem 15 35 50 120 := by
  sorry

end joanne_coins_l3988_398811


namespace school_sections_theorem_l3988_398837

/-- Calculates the total number of sections needed in a school with given constraints -/
def totalSections (numBoys numGirls : ℕ) (maxBoysPerSection maxGirlsPerSection : ℕ) (numSubjects : ℕ) : ℕ :=
  let boySections := (numBoys + maxBoysPerSection - 1) / maxBoysPerSection * numSubjects
  let girlSections := (numGirls + maxGirlsPerSection - 1) / maxGirlsPerSection * numSubjects
  boySections + girlSections

/-- Theorem stating that the total number of sections is 87 under the given constraints -/
theorem school_sections_theorem :
  totalSections 408 192 24 16 3 = 87 := by
  sorry

end school_sections_theorem_l3988_398837


namespace angle_D_measure_l3988_398874

-- Define the hexagon and its angles
structure Hexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

-- Define the properties of the hexagon
def is_convex_hexagon_with_properties (h : Hexagon) : Prop :=
  h.A = h.B ∧ h.B = h.C ∧  -- A, B, C are congruent
  h.D = h.E ∧ h.E = h.F ∧  -- D, E, F are congruent
  h.A + 30 = h.D ∧         -- A is 30° less than D
  h.A + h.B + h.C + h.D + h.E + h.F = 720  -- Sum of angles in a hexagon

-- Theorem statement
theorem angle_D_measure (h : Hexagon) 
  (hprop : is_convex_hexagon_with_properties h) : h.D = 135 := by
  sorry

end angle_D_measure_l3988_398874


namespace rectangle_area_theorem_l3988_398852

/-- The area of a rectangle with perimeter 20 meters and one side length x meters --/
def rectangle_area (x : ℝ) : ℝ := x * (10 - x)

/-- Theorem: The area of a rectangle with perimeter 20 meters and one side length x meters is x(10 - x) square meters --/
theorem rectangle_area_theorem (x : ℝ) (h : x > 0 ∧ x < 10) : 
  rectangle_area x = x * (10 - x) ∧ 
  2 * (x + (10 - x)) = 20 := by
  sorry

#check rectangle_area_theorem

end rectangle_area_theorem_l3988_398852


namespace arithmetic_calculation_l3988_398829

theorem arithmetic_calculation : 4 * (8 - 3)^2 / 5 = 20 := by
  sorry

end arithmetic_calculation_l3988_398829


namespace min_four_digit_number_l3988_398809

/-- Represents a four-digit number ABCD -/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Returns the first two digits (AB) of a four-digit number -/
def first_two_digits (n : FourDigitNumber) : ℕ :=
  n.value / 100

/-- Returns the last two digits (CD) of a four-digit number -/
def last_two_digits (n : FourDigitNumber) : ℕ :=
  n.value % 100

/-- The property that ABCD + AB × CD is a multiple of 1111 -/
def satisfies_condition (n : FourDigitNumber) : Prop :=
  ∃ k : ℕ, n.value + (first_two_digits n) * (last_two_digits n) = 1111 * k

theorem min_four_digit_number :
  ∀ n : FourDigitNumber, satisfies_condition n → n.value ≥ 1729 :=
by sorry

end min_four_digit_number_l3988_398809


namespace brand_preference_ratio_l3988_398831

theorem brand_preference_ratio (total : ℕ) (brand_x : ℕ) (h1 : total = 400) (h2 : brand_x = 360) :
  (brand_x : ℚ) / (total - brand_x : ℚ) = 9 / 1 := by
  sorry

end brand_preference_ratio_l3988_398831


namespace distinct_subscription_selections_l3988_398822

def number_of_providers : ℕ := 25
def number_of_siblings : ℕ := 4

theorem distinct_subscription_selections :
  (number_of_providers - 0) *
  (number_of_providers - 1) *
  (number_of_providers - 2) *
  (number_of_providers - 3) = 303600 := by
  sorry

end distinct_subscription_selections_l3988_398822


namespace sphere_distance_to_plane_l3988_398871

/-- The distance from the center of a sphere to the plane formed by three points on its surface. -/
def distance_center_to_plane (r : ℝ) (a b c : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for a sphere of radius 13 with three points on its surface forming
    a triangle with side lengths 6, 8, and 10, the distance from the center to the plane
    containing the triangle is 12. -/
theorem sphere_distance_to_plane :
  distance_center_to_plane 13 6 8 10 = 12 := by
  sorry

end sphere_distance_to_plane_l3988_398871


namespace binomial_ratio_sum_l3988_398882

theorem binomial_ratio_sum (n k : ℕ+) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k+1) : ℚ) = 2/3 ∧ 
  (Nat.choose n (k+1) : ℚ) / (Nat.choose n (k+2) : ℚ) = 3/4 ∧
  (∀ m l : ℕ+, (Nat.choose m l : ℚ) / (Nat.choose m (l+1) : ℚ) = 2/3 ∧ 
               (Nat.choose m (l+1) : ℚ) / (Nat.choose m (l+2) : ℚ) = 3/4 → 
               m = n ∧ l = k) →
  n + k = 47 := by
sorry

end binomial_ratio_sum_l3988_398882


namespace blue_to_red_ratio_l3988_398810

def cube_side_length : ℕ := 13

def red_face_area : ℕ := 6 * cube_side_length^2

def total_face_area : ℕ := 6 * cube_side_length^3

def blue_face_area : ℕ := total_face_area - red_face_area

theorem blue_to_red_ratio :
  blue_face_area / red_face_area = 12 := by
  sorry

end blue_to_red_ratio_l3988_398810


namespace product_of_square_roots_l3988_398881

theorem product_of_square_roots (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (8 * q^3) * Real.sqrt (7 * q^5) = 29 * q^4 * Real.sqrt 840 := by
  sorry

end product_of_square_roots_l3988_398881


namespace max_value_of_quadratic_l3988_398865

theorem max_value_of_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 1/3) :
  x * (1 - 3*x) ≤ 1/12 :=
sorry

end max_value_of_quadratic_l3988_398865


namespace inequality_proof_l3988_398813

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  Real.sqrt (x + (y - z)^2 / 12) + Real.sqrt (y + (x - z)^2 / 12) + Real.sqrt (z + (x - y)^2 / 12) ≤ Real.sqrt 3 := by
  sorry

end inequality_proof_l3988_398813


namespace jelly_bean_probabilities_l3988_398863

/-- Represents the number of jelly beans of each color in the bag -/
structure JellyBeanBag where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  purple : ℕ

/-- Calculates the total number of jelly beans in the bag -/
def totalJellyBeans (bag : JellyBeanBag) : ℕ :=
  bag.red + bag.green + bag.yellow + bag.blue + bag.purple

/-- The specific bag of jelly beans described in the problem -/
def ourBag : JellyBeanBag :=
  { red := 10, green := 12, yellow := 15, blue := 18, purple := 5 }

theorem jelly_bean_probabilities :
  let total := totalJellyBeans ourBag
  (ourBag.purple : ℚ) / total = 1 / 12 ∧
  ((ourBag.blue + ourBag.purple : ℚ) / total = 23 / 60) := by
  sorry

end jelly_bean_probabilities_l3988_398863


namespace sport_to_standard_ratio_l3988_398801

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard : DrinkRatio :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

/-- The sport formulation ratio -/
def sport : DrinkRatio :=
  { flavoring := (15 : ℚ) / 60,
    corn_syrup := 1,
    water := 15 }

/-- The ratio of flavoring to corn syrup for a given formulation -/
def flavoring_to_corn_syrup_ratio (d : DrinkRatio) : ℚ :=
  d.flavoring / d.corn_syrup

theorem sport_to_standard_ratio :
  flavoring_to_corn_syrup_ratio sport / flavoring_to_corn_syrup_ratio standard = 3 := by
  sorry

end sport_to_standard_ratio_l3988_398801


namespace dallas_current_age_l3988_398888

/-- Proves Dallas's current age given the relationships between family members' ages --/
theorem dallas_current_age (dallas_last_year darcy_last_year darcy_current dexter_current : ℕ) 
  (h1 : dallas_last_year = 3 * darcy_last_year)
  (h2 : darcy_current = 2 * dexter_current)
  (h3 : dexter_current = 8) :
  dallas_last_year + 1 = 46 := by
  sorry

#check dallas_current_age

end dallas_current_age_l3988_398888


namespace square_area_increase_l3988_398896

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.5 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end square_area_increase_l3988_398896


namespace units_digit_of_sum_factorials_1000_l3988_398875

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_of_sum_factorials_1000 :
  sum_of_factorials 1000 % 10 = 3 := by
  sorry

end units_digit_of_sum_factorials_1000_l3988_398875


namespace one_seven_two_eight_gt_one_roundness_of_1728_l3988_398861

/-- Roundness of an integer greater than 1 is the sum of exponents in its prime factorization -/
def roundness (n : ℕ) : ℕ :=
  sorry

/-- 1728 is greater than 1 -/
theorem one_seven_two_eight_gt_one : 1728 > 1 :=
  sorry

/-- The roundness of 1728 is 9 -/
theorem roundness_of_1728 : roundness 1728 = 9 :=
  sorry

end one_seven_two_eight_gt_one_roundness_of_1728_l3988_398861


namespace expression_simplification_l3988_398803

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (((x + 2) / (x - 2) + (x - x^2) / (x^2 - 4*x + 4)) / ((x - 4) / (x - 2))) = 1 :=
by sorry

end expression_simplification_l3988_398803


namespace paint_combinations_l3988_398817

/-- The number of color options available -/
def num_colors : ℕ := 5

/-- The number of painting method options available -/
def num_methods : ℕ := 4

/-- The total number of combinations of color and painting method -/
def total_combinations : ℕ := num_colors * num_methods

theorem paint_combinations : total_combinations = 20 := by
  sorry

end paint_combinations_l3988_398817


namespace fair_attendance_percentage_l3988_398873

/-- The percent of projected attendance that was the actual attendance --/
theorem fair_attendance_percentage (A : ℝ) (V W : ℝ) : 
  let projected_attendance := 1.25 * A
  let actual_attendance := 0.8 * A
  (actual_attendance / projected_attendance) * 100 = 64 := by
  sorry

end fair_attendance_percentage_l3988_398873


namespace grandfather_grandson_ages_l3988_398866

theorem grandfather_grandson_ages :
  ∀ (x y a b : ℕ),
    x > 70 →
    x - a = 10 * (y - a) →
    x + b = 8 * (y + b) →
    x = 71 ∧ y = 8 :=
by
  sorry

end grandfather_grandson_ages_l3988_398866


namespace regression_equation_equivalence_l3988_398832

/-- Conversion factor from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- Conversion factor from pounds to kilograms -/
def pound_to_kg : ℝ := 0.45

/-- Slope of the regression equation in imperial units (pounds per inch) -/
def imperial_slope : ℝ := 4

/-- Intercept of the regression equation in imperial units (pounds) -/
def imperial_intercept : ℝ := -130

/-- Predicted weight in imperial units (pounds) given height in inches -/
def predicted_weight_imperial (height : ℝ) : ℝ :=
  imperial_slope * height + imperial_intercept

/-- Predicted weight in metric units (kg) given height in cm -/
def predicted_weight_metric (height : ℝ) : ℝ :=
  0.72 * height - 58.5

theorem regression_equation_equivalence :
  ∀ height_inch : ℝ,
  let height_cm := height_inch * inch_to_cm
  predicted_weight_metric height_cm =
    predicted_weight_imperial height_inch * pound_to_kg := by
  sorry

end regression_equation_equivalence_l3988_398832


namespace dad_has_three_eyes_l3988_398800

/-- A monster family with a specified number of eyes for each member -/
structure MonsterFamily where
  mom_eyes : ℕ
  dad_eyes : ℕ
  num_kids : ℕ
  kid_eyes : ℕ
  total_eyes : ℕ

/-- Theorem stating that given the conditions of the monster family, the dad must have 3 eyes -/
theorem dad_has_three_eyes (family : MonsterFamily)
  (h1 : family.mom_eyes = 1)
  (h2 : family.num_kids = 3)
  (h3 : family.kid_eyes = 4)
  (h4 : family.total_eyes = 16) :
  family.dad_eyes = 3 := by
  sorry


end dad_has_three_eyes_l3988_398800


namespace work_completion_time_l3988_398821

/-- 
Given:
- A can complete the work in 60 days
- A and B together can complete the work in 15 days

Prove that B can complete the work alone in 20 days
-/
theorem work_completion_time (a b : ℝ) (h1 : a = 60) (h2 : 1 / a + 1 / b = 1 / 15) : b = 20 := by
  sorry

end work_completion_time_l3988_398821


namespace ratio_of_percentages_l3988_398878

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.30 * Q)
  (hQ : Q = 0.20 * P)
  (hN : N = 0.50 * P)
  (hP : P ≠ 0) :
  M / N = 3 / 25 := by
sorry

end ratio_of_percentages_l3988_398878


namespace order_of_f_values_l3988_398868

noncomputable def f (x : ℝ) : ℝ := 2 / (4^x) - x

noncomputable def a : ℝ := 0
noncomputable def b : ℝ := Real.log 2 / Real.log 0.4
noncomputable def c : ℝ := Real.log 3 / Real.log 4

theorem order_of_f_values :
  f a < f c ∧ f c < f b := by sorry

end order_of_f_values_l3988_398868


namespace circle_dot_product_bound_l3988_398880

theorem circle_dot_product_bound :
  ∀ (A : ℝ × ℝ),
  A.1^2 + (A.2 - 1)^2 = 1 →
  -2 ≤ (A.1 * 2 + A.2 * 0) ∧ (A.1 * 2 + A.2 * 0) ≤ 2 := by
  sorry

end circle_dot_product_bound_l3988_398880


namespace highest_sample_number_l3988_398885

/-- Given a systematic sample from a population, calculate the highest number in the sample. -/
theorem highest_sample_number
  (total_students : Nat)
  (sample_size : Nat)
  (first_sample : Nat)
  (h1 : total_students = 54)
  (h2 : sample_size = 6)
  (h3 : first_sample = 5)
  (h4 : sample_size > 0)
  (h5 : total_students ≥ sample_size)
  : first_sample + (sample_size - 1) * (total_students / sample_size) = 50 := by
  sorry

#check highest_sample_number

end highest_sample_number_l3988_398885


namespace danny_steve_time_ratio_l3988_398864

/-- The time it takes Danny to reach Steve's house, in minutes -/
def danny_time : ℝ := 35

/-- The time it takes Steve to reach Danny's house, in minutes -/
def steve_time : ℝ := 70

/-- The extra time it takes Steve to reach the halfway point compared to Danny, in minutes -/
def extra_time : ℝ := 17.5

theorem danny_steve_time_ratio :
  danny_time / steve_time = 1 / 2 ∧
  steve_time / 2 = danny_time / 2 + extra_time :=
sorry

end danny_steve_time_ratio_l3988_398864


namespace point_conditions_imply_m_value_l3988_398841

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the point P based on the parameter m -/
def P (m : ℝ) : Point :=
  { x := 3 - m, y := 2 * m + 6 }

/-- Condition: P is in the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Condition: P is equidistant from the coordinate axes -/
def equidistant_from_axes (p : Point) : Prop :=
  abs p.x = abs p.y

/-- Theorem: If P(3-m, 2m+6) is in the fourth quadrant and equidistant from the axes, then m = -9 -/
theorem point_conditions_imply_m_value :
  ∀ m : ℝ, in_fourth_quadrant (P m) ∧ equidistant_from_axes (P m) → m = -9 :=
by sorry

end point_conditions_imply_m_value_l3988_398841


namespace target_digit_is_seven_l3988_398839

/-- The decimal representation of 13/481 -/
def decimal_rep : ℚ := 13 / 481

/-- The length of the repeating sequence in the decimal representation -/
def repeat_length : ℕ := 3

/-- The position of the digit we're looking for -/
def target_position : ℕ := 222

/-- The function that returns the nth digit after the decimal point -/
noncomputable def nth_digit (n : ℕ) : ℕ := 
  sorry

theorem target_digit_is_seven : nth_digit target_position = 7 := by
  sorry

end target_digit_is_seven_l3988_398839


namespace average_diesel_cost_approx_9_94_l3988_398891

/-- Represents the diesel purchase data for a single year -/
structure YearlyPurchase where
  litres : ℝ
  pricePerLitre : ℝ

/-- Calculates the total cost for a year including delivery fees and taxes -/
def yearlyTotalCost (purchase : YearlyPurchase) (deliveryFee : ℝ) (taxes : ℝ) : ℝ :=
  purchase.litres * purchase.pricePerLitre + deliveryFee + taxes

/-- Theorem: The average cost per litre of diesel over three years is approximately 9.94 -/
theorem average_diesel_cost_approx_9_94 
  (year1 : YearlyPurchase)
  (year2 : YearlyPurchase)
  (year3 : YearlyPurchase)
  (deliveryFee : ℝ)
  (taxes : ℝ)
  (h1 : year1.litres = 520 ∧ year1.pricePerLitre = 8.5)
  (h2 : year2.litres = 540 ∧ year2.pricePerLitre = 9)
  (h3 : year3.litres = 560 ∧ year3.pricePerLitre = 9.5)
  (h4 : deliveryFee = 200)
  (h5 : taxes = 300) :
  let totalCost := yearlyTotalCost year1 deliveryFee taxes + 
                   yearlyTotalCost year2 deliveryFee taxes + 
                   yearlyTotalCost year3 deliveryFee taxes
  let totalLitres := year1.litres + year2.litres + year3.litres
  let averageCost := totalCost / totalLitres
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |averageCost - 9.94| < ε :=
by sorry

end average_diesel_cost_approx_9_94_l3988_398891


namespace complex_equation_sum_l3988_398834

theorem complex_equation_sum (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  Complex.mk a b = Complex.mk 1 1 * Complex.mk 2 (-1) →
  a + b = 4 := by
  sorry

end complex_equation_sum_l3988_398834


namespace opposite_of_negative_fifth_l3988_398883

theorem opposite_of_negative_fifth : -(-(1/5 : ℚ)) = 1/5 := by sorry

end opposite_of_negative_fifth_l3988_398883


namespace log_equation_solution_l3988_398825

theorem log_equation_solution (x : ℝ) (h : x > 0) : 
  2 * (Real.log x / Real.log 6) = 1 - (Real.log 3 / Real.log 6) ↔ x = Real.sqrt 2 := by
  sorry

end log_equation_solution_l3988_398825


namespace c_rent_share_is_72_l3988_398807

/-- Represents a person renting the pasture -/
structure Renter where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of a renter in ox-months -/
def share (r : Renter) : ℕ := r.oxen * r.months

/-- Represents the pasture rental scenario -/
structure PastureRental where
  a : Renter
  b : Renter
  c : Renter
  totalRent : ℕ

/-- Calculates the total share of all renters -/
def totalShare (pr : PastureRental) : ℕ :=
  share pr.a + share pr.b + share pr.c

/-- Calculates the rent share for a specific renter -/
def rentShare (pr : PastureRental) (r : Renter) : ℚ :=
  (share r : ℚ) / (totalShare pr : ℚ) * pr.totalRent

theorem c_rent_share_is_72 (pr : PastureRental) : 
  pr.a = { oxen := 10, months := 7 } →
  pr.b = { oxen := 12, months := 5 } →
  pr.c = { oxen := 15, months := 3 } →
  pr.totalRent = 280 →
  rentShare pr pr.c = 72 := by
  sorry

#check c_rent_share_is_72

end c_rent_share_is_72_l3988_398807


namespace heptagon_angle_measure_l3988_398893

/-- In a heptagon GEOMETRY, prove that if ∠G ≅ ∠E ≅ ∠R, ∠O is supplementary to ∠Y,
    and assuming ∠M ≅ ∠T ≅ ∠R, then ∠R = 144° -/
theorem heptagon_angle_measure (GEOMETRY : Type) 
  (G E O M T R Y : GEOMETRY → ℝ) :
  (∀ x : GEOMETRY, G x = E x ∧ E x = R x) →  -- ∠G ≅ ∠E ≅ ∠R
  (∀ x : GEOMETRY, O x + Y x = 180) →        -- ∠O is supplementary to ∠Y
  (∀ x : GEOMETRY, M x = T x ∧ T x = R x) →  -- Assumption: ∠M ≅ ∠T ≅ ∠R
  (∀ x : GEOMETRY, G x + E x + O x + M x + T x + R x + Y x = 900) →  -- Sum of angles in heptagon
  (∀ x : GEOMETRY, R x = 144) :=
by
  sorry


end heptagon_angle_measure_l3988_398893


namespace quadratic_symmetry_l3988_398890

/-- A quadratic function with coefficients a, b, and c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x, quadratic a b c (x + 4.5) = quadratic a b c (4.5 - x)) →
  quadratic a b c (-9) = 1 →
  quadratic a b c 18 = 1 := by
  sorry

end quadratic_symmetry_l3988_398890


namespace balloon_count_total_l3988_398816

/-- Calculate the total number of balloons for each color given Sara's and Sandy's balloons -/
theorem balloon_count_total 
  (R1 G1 B1 Y1 O1 R2 G2 B2 Y2 O2 : ℕ) 
  (h1 : R1 = 31) (h2 : G1 = 15) (h3 : B1 = 12) (h4 : Y1 = 18) (h5 : O1 = 10)
  (h6 : R2 = 24) (h7 : G2 = 7)  (h8 : B2 = 14) (h9 : Y2 = 20) (h10 : O2 = 16) :
  R1 + R2 = 55 ∧ 
  G1 + G2 = 22 ∧ 
  B1 + B2 = 26 ∧ 
  Y1 + Y2 = 38 ∧ 
  O1 + O2 = 26 := by
  sorry

end balloon_count_total_l3988_398816


namespace range_of_f_l3988_398894

-- Define the function
def f (x : ℝ) : ℝ := |x^2 - 4| - 3*x

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -6 ≤ y ∧ y ≤ 25/4} :=
by sorry

end range_of_f_l3988_398894


namespace smallest_ends_in_7_div_by_5_l3988_398804

def ends_in_7 (n : ℕ) : Prop := n % 10 = 7

theorem smallest_ends_in_7_div_by_5 : 
  ∃ (n : ℕ), n > 0 ∧ ends_in_7 n ∧ n % 5 = 0 ∧ 
  ∀ (m : ℕ), m > 0 → ends_in_7 m → m % 5 = 0 → m ≥ n :=
by
  use 37
  sorry

end smallest_ends_in_7_div_by_5_l3988_398804


namespace salary_calculation_l3988_398815

/-- Represents the man's original monthly salary in Rupees -/
def original_salary : ℝ := sorry

/-- The man's original savings rate as a decimal -/
def savings_rate : ℝ := 0.20

/-- The man's original rent expense rate as a decimal -/
def rent_rate : ℝ := 0.40

/-- The man's original utilities expense rate as a decimal -/
def utilities_rate : ℝ := 0.30

/-- The man's original groceries expense rate as a decimal -/
def groceries_rate : ℝ := 0.20

/-- The increase rate for rent as a decimal -/
def rent_increase : ℝ := 0.15

/-- The increase rate for utilities as a decimal -/
def utilities_increase : ℝ := 0.20

/-- The increase rate for groceries as a decimal -/
def groceries_increase : ℝ := 0.10

/-- The reduced savings amount in Rupees -/
def reduced_savings : ℝ := 180

theorem salary_calculation :
  original_salary * savings_rate - reduced_savings =
  original_salary * (rent_rate * (1 + rent_increase) +
                     utilities_rate * (1 + utilities_increase) +
                     groceries_rate * (1 + groceries_increase)) -
  original_salary * (rent_rate + utilities_rate + groceries_rate) ∧
  original_salary = 3000 :=
sorry

end salary_calculation_l3988_398815


namespace items_left_in_store_l3988_398877

theorem items_left_in_store (ordered : ℕ) (sold : ℕ) (in_storeroom : ℕ) 
  (h_ordered : ordered = 4458)
  (h_sold : sold = 1561)
  (h_storeroom : in_storeroom = 575)
  (h_damaged : ⌊(5 : ℝ) / 100 * ordered⌋ = 222) : 
  ordered - sold - ⌊(5 : ℝ) / 100 * ordered⌋ + in_storeroom = 3250 := by
  sorry

end items_left_in_store_l3988_398877


namespace max_value_x_5_minus_4x_l3988_398818

theorem max_value_x_5_minus_4x (x : ℝ) (h1 : 0 < x) (h2 : x < 5/4) :
  x * (5 - 4*x) ≤ 25/16 := by
  sorry

end max_value_x_5_minus_4x_l3988_398818


namespace choir_arrangement_min_choir_size_l3988_398898

theorem choir_arrangement (n : ℕ) : (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) → n ≥ 990 :=
sorry

theorem min_choir_size : ∃ (n : ℕ), n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n = 990 :=
sorry

end choir_arrangement_min_choir_size_l3988_398898


namespace smallest_a_is_2_pow_16_l3988_398842

/-- The number of factors of a natural number n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- The smallest natural number satisfying the given condition -/
def smallest_a : ℕ := sorry

/-- The theorem statement -/
theorem smallest_a_is_2_pow_16 :
  (∀ a : ℕ, num_factors (a^2) = num_factors a + 16 → a ≥ smallest_a) ∧
  num_factors (smallest_a^2) = num_factors smallest_a + 16 ∧
  smallest_a = 2^16 := by sorry

end smallest_a_is_2_pow_16_l3988_398842


namespace root_product_theorem_l3988_398836

theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ x₆ : ℂ) : 
  (x₁^6 - x₁^3 + 1 = 0) → 
  (x₂^6 - x₂^3 + 1 = 0) → 
  (x₃^6 - x₃^3 + 1 = 0) → 
  (x₄^6 - x₄^3 + 1 = 0) → 
  (x₅^6 - x₅^3 + 1 = 0) → 
  (x₆^6 - x₆^3 + 1 = 0) → 
  (x₁^2 - 3) * (x₂^2 - 3) * (x₃^2 - 3) * (x₄^2 - 3) * (x₅^2 - 3) * (x₆^2 - 3) = 757 := by
  sorry

end root_product_theorem_l3988_398836


namespace solution_set_part_i_range_of_a_part_ii_l3988_398862

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |x - 3|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x + |2*x - 5| ≥ 6} = {x : ℝ | x ≥ 4 ∨ x ≤ 0} := by sorry

-- Part II
theorem range_of_a_part_ii (h : Set.Icc (-1 : ℝ) 2 ⊆ Set.range (g a)) :
  a ≤ 1 ∨ a ≥ 5 := by sorry

end solution_set_part_i_range_of_a_part_ii_l3988_398862


namespace absolute_value_equation_solution_l3988_398876

theorem absolute_value_equation_solution :
  let f : ℝ → ℝ := λ x => |2*x + 4|
  let g : ℝ → ℝ := λ x => 1 - 3*x + x^2
  let solution1 : ℝ := (5 + Real.sqrt 37) / 2
  let solution2 : ℝ := (5 - Real.sqrt 37) / 2
  (∀ x : ℝ, f x = g x ↔ x = solution1 ∨ x = solution2) :=
by sorry

end absolute_value_equation_solution_l3988_398876


namespace element_in_set_M_l3988_398847

def U : Finset Nat := {1, 2, 3, 4, 5}

theorem element_in_set_M (M : Finset Nat) 
  (h : (U \ M) = {1, 2}) : 3 ∈ M := by
  sorry

end element_in_set_M_l3988_398847


namespace sam_has_46_balloons_l3988_398828

/-- Given the number of red balloons Fred and Dan have, and the total number of red balloons,
    calculate the number of red balloons Sam has. -/
def sams_balloons (fred_balloons dan_balloons total_balloons : ℕ) : ℕ :=
  total_balloons - (fred_balloons + dan_balloons)

/-- Theorem stating that under the given conditions, Sam has 46 red balloons. -/
theorem sam_has_46_balloons :
  sams_balloons 10 16 72 = 46 := by
  sorry

end sam_has_46_balloons_l3988_398828


namespace root_in_interval_l3988_398826

def f (x : ℝ) := x^3 - x - 1

theorem root_in_interval :
  ∃ r ∈ Set.Icc 1.25 1.5, f r = 0 :=
by
  sorry

end root_in_interval_l3988_398826


namespace polynomial_equality_l3988_398806

theorem polynomial_equality (P : ℝ → ℝ) :
  (∀ a b c : ℝ, P (a + b - 2 * c) + P (b + c - 2 * a) + P (c + a - 2 * b) = 
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)) →
  ∃ a b : ℝ, ∀ x : ℝ, P x = a * x^2 + b * x :=
by sorry

end polynomial_equality_l3988_398806


namespace number_problem_l3988_398870

theorem number_problem (x : ℚ) : (x / 6) * 12 = 10 → x = 5 := by
  sorry

end number_problem_l3988_398870


namespace endpoint_sum_endpoint_sum_proof_l3988_398851

/-- Given a line segment with one endpoint (6, 2) and midpoint (5, 7),
    the sum of the coordinates of the other endpoint is 16. -/
theorem endpoint_sum : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop :=
  fun endpoint1 midpoint endpoint2 =>
    endpoint1 = (6, 2) ∧
    midpoint = (5, 7) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 + endpoint2.2 = 16

/-- Proof of the theorem -/
theorem endpoint_sum_proof : ∃ (endpoint2 : ℝ × ℝ), endpoint_sum (6, 2) (5, 7) endpoint2 :=
  sorry

end endpoint_sum_endpoint_sum_proof_l3988_398851


namespace volume_of_enlarged_box_l3988_398805

/-- Represents a rectangular box with length l, width w, and height h -/
structure Box where
  l : ℝ
  w : ℝ
  h : ℝ

/-- Theorem: Volume of enlarged box -/
theorem volume_of_enlarged_box (box : Box) 
  (volume_eq : box.l * box.w * box.h = 5000)
  (surface_area_eq : 2 * (box.l * box.w + box.w * box.h + box.l * box.h) = 1800)
  (edge_sum_eq : 4 * (box.l + box.w + box.h) = 210) :
  (box.l + 2) * (box.w + 2) * (box.h + 2) = 7018 := by
  sorry

end volume_of_enlarged_box_l3988_398805


namespace binomial_probability_problem_l3988_398823

/-- A binomial distribution with n trials and probability p -/
def binomial_distribution (n : ℕ) (p : ℝ) : Type := Unit

/-- The probability of getting at least k successes in a binomial distribution -/
def P_at_least (dist : binomial_distribution n p) (k : ℕ) : ℝ := sorry

theorem binomial_probability_problem 
  (p : ℝ) 
  (ξ : binomial_distribution 2 p) 
  (η : binomial_distribution 4 p) 
  (h : P_at_least ξ 1 = 5/9) :
  P_at_least η 2 = 11/27 := by
  sorry

end binomial_probability_problem_l3988_398823


namespace enough_money_for_jump_ropes_l3988_398812

/-- The cost of a single jump rope in yuan -/
def jump_rope_cost : ℕ := 8

/-- The number of jump ropes to be purchased -/
def num_jump_ropes : ℕ := 31

/-- The amount of money available in yuan -/
def available_money : ℕ := 250

/-- Theorem stating that the class has enough money to buy the jump ropes -/
theorem enough_money_for_jump_ropes :
  jump_rope_cost * num_jump_ropes ≤ available_money := by
  sorry

end enough_money_for_jump_ropes_l3988_398812


namespace f_properties_l3988_398892

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -a * Real.log x + (a + 1) * x - 0.5 * x^2

theorem f_properties (a : ℝ) (h_a : a > 0) :
  -- Monotonicity property
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ < x₂ ∧ f a x₁ < f a x₂) ∧
  (∃ (x₃ x₄ : ℝ), x₃ > 0 ∧ x₄ > 0 ∧ x₃ < x₄ ∧ f a x₃ > f a x₄) ∧
  -- Maximum value of b
  (∀ b : ℝ, (∀ x : ℝ, x > 0 → f a x ≥ -0.5 * x^2 + a * x + b) →
    b ≤ 0.5 * (1 + Real.log 2)) ∧
  (∃ b : ℝ, b = 0.5 * (1 + Real.log 2) ∧
    (∀ x : ℝ, x > 0 → f (0.5) x ≥ -0.5 * x^2 + 0.5 * x + b)) :=
by sorry

end f_properties_l3988_398892


namespace eleanor_childless_descendants_l3988_398819

/-- Eleanor's family structure -/
structure EleanorFamily where
  daughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The number of Eleanor's daughters and granddaughters with no daughters -/
def childless_descendants (f : EleanorFamily) : ℕ :=
  f.total_descendants - f.daughters_with_children

/-- Theorem stating the number of Eleanor's daughters and granddaughters with no daughters -/
theorem eleanor_childless_descendants :
  ∀ f : EleanorFamily,
  f.daughters = 8 →
  f.total_descendants = 43 →
  f.daughters_with_children * 7 = f.total_descendants - f.daughters →
  childless_descendants f = 38 := by
  sorry

end eleanor_childless_descendants_l3988_398819


namespace cos_product_equals_one_l3988_398887

theorem cos_product_equals_one : 8 * Real.cos (4 * Real.pi / 9) * Real.cos (2 * Real.pi / 9) * Real.cos (Real.pi / 9) = 1 := by
  sorry

end cos_product_equals_one_l3988_398887


namespace combined_degrees_l3988_398814

theorem combined_degrees (summer_degrees jolly_degrees : ℕ) : 
  summer_degrees = 150 → 
  summer_degrees = jolly_degrees + 5 → 
  summer_degrees + jolly_degrees = 295 := by
sorry

end combined_degrees_l3988_398814


namespace binary_101110_equals_46_l3988_398895

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_101110_equals_46 :
  binary_to_decimal [false, true, true, true, true, false, true] = 46 := by
  sorry

end binary_101110_equals_46_l3988_398895


namespace larger_number_l3988_398808

theorem larger_number (x y : ℝ) (h1 : x + y = 28) (h2 : x - y = 4) : max x y = 16 := by
  sorry

end larger_number_l3988_398808


namespace cuboid_diagonal_squared_l3988_398843

/-- The square of the diagonal of a cuboid equals the sum of the squares of its length, width, and height. -/
theorem cuboid_diagonal_squared (l w h d : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  d^2 = l^2 + w^2 + h^2 :=
by sorry

end cuboid_diagonal_squared_l3988_398843
