import Mathlib

namespace starting_lineup_theorem_l2727_272765

def total_players : ℕ := 18
def goalie_count : ℕ := 1
def regular_players_count : ℕ := 10
def captain_count : ℕ := 1

def starting_lineup_count : ℕ :=
  total_players * (Nat.choose (total_players - goalie_count) regular_players_count) * regular_players_count

theorem starting_lineup_theorem :
  starting_lineup_count = 34928640 := by sorry

end starting_lineup_theorem_l2727_272765


namespace range_of_a_for_P_and_Q_l2727_272742

theorem range_of_a_for_P_and_Q (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) ↔ 
  a ≤ -2 ∨ a = 1 := by sorry

end range_of_a_for_P_and_Q_l2727_272742


namespace tan_150_degrees_l2727_272718

theorem tan_150_degrees : Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end tan_150_degrees_l2727_272718


namespace juice_boxes_for_school_year_l2727_272758

/-- Calculate the total number of juice boxes needed for a school year -/
def total_juice_boxes (num_children : ℕ) (school_days_per_week : ℕ) (weeks_in_school_year : ℕ) : ℕ :=
  num_children * school_days_per_week * weeks_in_school_year

/-- Theorem: Given the specific conditions, the total number of juice boxes needed is 375 -/
theorem juice_boxes_for_school_year :
  let num_children : ℕ := 3
  let school_days_per_week : ℕ := 5
  let weeks_in_school_year : ℕ := 25
  total_juice_boxes num_children school_days_per_week weeks_in_school_year = 375 := by
  sorry


end juice_boxes_for_school_year_l2727_272758


namespace equation_solution_l2727_272755

theorem equation_solution :
  ∀ x : ℝ, x + 36 / (x - 4) = -9 ↔ x = 0 ∨ x = -5 := by
  sorry

end equation_solution_l2727_272755


namespace smallest_difference_l2727_272781

theorem smallest_difference (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) :
  ∃ (m : ℤ), m = 4 ∧ ∀ (c d : ℤ), c + d < 11 → c > 6 → c - d ≥ m :=
by sorry

end smallest_difference_l2727_272781


namespace range_of_negative_values_l2727_272767

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x y, x < y ∧ y ≤ 0 → f x > f y

-- State the theorem
theorem range_of_negative_values (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_decreasing : decreasing_on_neg f) 
  (h_zero : f 3 = 0) : 
  {x : ℝ | f x < 0} = Set.Ioo (-3) 3 := by
  sorry

end range_of_negative_values_l2727_272767


namespace complement_of_A_l2727_272748

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x + 1 > 0}

theorem complement_of_A (x : ℝ) : x ∈ (Set.univ \ A) ↔ x ≤ -1 := by sorry

end complement_of_A_l2727_272748


namespace least_product_of_three_primes_greater_than_50_l2727_272791

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to find the nth prime number greater than a given value
def nthPrimeGreaterThan (n : ℕ) (start : ℕ) : ℕ :=
  sorry

theorem least_product_of_three_primes_greater_than_50 :
  ∃ p q r : ℕ,
    isPrime p ∧ isPrime q ∧ isPrime r ∧
    p > 50 ∧ q > 50 ∧ r > 50 ∧
    p < q ∧ q < r ∧
    p * q * r = 191557 ∧
    ∀ a b c : ℕ,
      isPrime a ∧ isPrime b ∧ isPrime c ∧
      a > 50 ∧ b > 50 ∧ c > 50 ∧
      a < b ∧ b < c →
      a * b * c ≥ 191557 :=
by sorry

end least_product_of_three_primes_greater_than_50_l2727_272791


namespace cubic_inequality_and_fraction_bound_l2727_272739

theorem cubic_inequality_and_fraction_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^3 + y^3 ≥ x^2*y + y^2*x) ∧
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a/b^2 + b/a^2 ≥ m/2 * (1/a + 1/b)) → m ≤ 2) :=
by sorry

end cubic_inequality_and_fraction_bound_l2727_272739


namespace problem_statement_l2727_272754

theorem problem_statement : (-1)^2023 - Real.tan (π/3) + (Real.sqrt 5 - 1)^0 + |-(Real.sqrt 3)| = 0 := by
  sorry

end problem_statement_l2727_272754


namespace paul_lost_crayons_l2727_272761

/-- Given information about Paul's crayons --/
def paul_crayons (initial : ℕ) (given_to_friends : ℕ) (difference : ℕ) : Prop :=
  ∃ (lost : ℕ), 
    initial ≥ given_to_friends + lost ∧
    given_to_friends = lost + difference

/-- Theorem stating the number of crayons Paul lost --/
theorem paul_lost_crayons :
  paul_crayons 589 571 410 → ∃ (lost : ℕ), lost = 161 := by
  sorry

end paul_lost_crayons_l2727_272761


namespace product_of_repeating_decimal_and_eight_l2727_272703

/-- Represents the repeating decimal 0.456̄ -/
def repeating_decimal : ℚ := 456 / 999

theorem product_of_repeating_decimal_and_eight :
  repeating_decimal * 8 = 1216 / 333 := by
  sorry

end product_of_repeating_decimal_and_eight_l2727_272703


namespace complex_magnitude_reciprocal_one_plus_i_l2727_272740

theorem complex_magnitude_reciprocal_one_plus_i :
  let i : ℂ := Complex.I
  let z : ℂ := 1 / (1 + i)
  Complex.abs z = Real.sqrt 2 / 2 := by
    sorry

end complex_magnitude_reciprocal_one_plus_i_l2727_272740


namespace bridge_crossing_time_l2727_272777

/-- Proves that a man walking at 5 km/hr takes 15 minutes to cross a 1250-meter bridge -/
theorem bridge_crossing_time :
  let walking_speed : ℝ := 5  -- km/hr
  let bridge_length : ℝ := 1250  -- meters
  let crossing_time : ℝ := 15  -- minutes
  
  walking_speed * 1000 / 60 * crossing_time = bridge_length :=
by sorry

end bridge_crossing_time_l2727_272777


namespace price_change_l2727_272760

theorem price_change (original_price : ℝ) (h : original_price > 0) :
  original_price * (1 + 0.02) * (1 - 0.02) < original_price :=
by
  sorry

end price_change_l2727_272760


namespace arithmetic_sequence_problem_l2727_272701

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 = 8)
  (h_sum2 : a 2 + a 4 = 12) :
  (∀ n : ℕ, a n = 2 * n) ∧
  (∃ n : ℕ, n * (n + 1) = 420) :=
sorry

end arithmetic_sequence_problem_l2727_272701


namespace parallel_line_k_value_l2727_272776

/-- A line passing through points (4, -5) and (k, 23) is parallel to the line 3x - 4y = 12 -/
theorem parallel_line_k_value (k : ℚ) : 
  (∃ (m b : ℚ), (m * 4 + b = -5) ∧ (m * k + b = 23) ∧ (m = 3/4)) → k = 124/3 := by
  sorry

end parallel_line_k_value_l2727_272776


namespace millet_cost_is_60_cents_l2727_272711

/-- Represents the cost of millet seed per pound -/
def millet_cost : ℝ := sorry

/-- The total weight of millet seed in pounds -/
def millet_weight : ℝ := 100

/-- The cost of sunflower seeds per pound -/
def sunflower_cost : ℝ := 1.10

/-- The total weight of sunflower seeds in pounds -/
def sunflower_weight : ℝ := 25

/-- The desired cost per pound of the mixture -/
def mixture_cost_per_pound : ℝ := 0.70

/-- The total weight of the mixture -/
def total_weight : ℝ := millet_weight + sunflower_weight

/-- Theorem stating that the cost of millet seed per pound is $0.60 -/
theorem millet_cost_is_60_cents :
  millet_cost = 0.60 :=
by
  sorry

end millet_cost_is_60_cents_l2727_272711


namespace isosceles_triangle_vertex_angle_l2727_272717

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We don't need to define all properties of an isosceles triangle,
  -- just the ones relevant to our problem
  vertex_angle : ℝ
  base_angle : ℝ
  is_valid : vertex_angle + 2 * base_angle = 180

-- Define our theorem
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.vertex_angle = 70 ∨ triangle.base_angle = 70) : 
  triangle.vertex_angle = 40 ∨ triangle.vertex_angle = 70 := by
  sorry


end isosceles_triangle_vertex_angle_l2727_272717


namespace expand_polynomial_l2727_272722

theorem expand_polynomial (x : ℝ) : (x + 3) * (x + 8) = x^2 + 11*x + 24 := by
  sorry

end expand_polynomial_l2727_272722


namespace plane_relation_l2727_272710

-- Define the concept of a plane
class Plane

-- Define the concept of a line
class Line

-- Define the parallelism relation between planes
def parallel (α β : Plane) : Prop := sorry

-- Define the intersection relation between planes
def intersects (α β : Plane) : Prop := sorry

-- Define the relation of a line being parallel to a plane
def line_parallel_to_plane (l : Line) (β : Plane) : Prop := sorry

-- Define the property of having infinitely many parallel lines
def has_infinitely_many_parallel_lines (α β : Plane) : Prop :=
  ∃ (S : Set Line), Set.Infinite S ∧ ∀ l ∈ S, line_parallel_to_plane l β

-- State the theorem
theorem plane_relation (α β : Plane) :
  has_infinitely_many_parallel_lines α β → parallel α β ∨ intersects α β :=
sorry

end plane_relation_l2727_272710


namespace problem_solution_l2727_272783

theorem problem_solution (x : ℝ) : 
  (7/11) * (5/13) * x = 48 → (315/100) * x = 617.4 := by
  sorry

end problem_solution_l2727_272783


namespace range_of_m_l2727_272713

theorem range_of_m : ∀ m : ℝ,
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0 ↔ m > 2) →
  (¬∃ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 = 0 ↔ 1 < m ∧ m < 3) →
  ((m > 2 ∨ (1 < m ∧ m < 3)) ∧ ¬(m > 2 ∧ 1 < m ∧ m < 3)) →
  m ≥ 3 ∨ (1 < m ∧ m ≤ 2) :=
by sorry

end range_of_m_l2727_272713


namespace problem_solution_l2727_272737

theorem problem_solution (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
  sorry

end problem_solution_l2727_272737


namespace positive_integer_solutions_count_l2727_272705

theorem positive_integer_solutions_count :
  let n : ℕ := 30
  let k : ℕ := 3
  (Nat.choose (n - 1) (k - 1) : ℕ) = 406 := by sorry

end positive_integer_solutions_count_l2727_272705


namespace sine_of_angle_l2727_272708

theorem sine_of_angle (α : Real) (m : Real) (h1 : m ≠ 0) 
  (h2 : Real.sqrt 3 / Real.sqrt (3 + m^2) = m / 6) 
  (h3 : Real.cos α = m / 6) : 
  Real.sin α = Real.sqrt 3 / 2 := by
sorry

end sine_of_angle_l2727_272708


namespace handshake_count_l2727_272779

theorem handshake_count (num_gremlins num_imps : ℕ) (h1 : num_gremlins = 30) (h2 : num_imps = 20) :
  let gremlin_handshakes := num_gremlins.choose 2
  let gremlin_imp_handshakes := num_gremlins * num_imps
  gremlin_handshakes + gremlin_imp_handshakes = 1035 := by
  sorry

#check handshake_count

end handshake_count_l2727_272779


namespace bruce_eggs_l2727_272763

theorem bruce_eggs (initial_eggs lost_eggs : ℕ) : 
  initial_eggs ≥ lost_eggs → 
  initial_eggs - lost_eggs = initial_eggs - lost_eggs :=
by
  sorry

#check bruce_eggs 75 70

end bruce_eggs_l2727_272763


namespace smallest_third_term_geometric_progression_l2727_272750

theorem smallest_third_term_geometric_progression (a b c : ℝ) : 
  a = 5 ∧ 
  b - a = c - b ∧ 
  (5 * (c + 27) = (b + 9)^2) →
  c + 27 ≥ 16 - 4 * Real.sqrt 7 :=
by sorry

end smallest_third_term_geometric_progression_l2727_272750


namespace isosceles_60_similar_l2727_272702

/-- An isosceles triangle with a 60° interior angle -/
structure IsoscelesTriangle60 where
  -- We represent the triangle by its three angles
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  -- The triangle is isosceles
  isIsosceles : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3)
  -- One of the angles is 60°
  has60Degree : angle1 = 60 ∨ angle2 = 60 ∨ angle3 = 60
  -- The sum of angles in a triangle is 180°
  sumIs180 : angle1 + angle2 + angle3 = 180

/-- Two isosceles triangles with a 60° interior angle are similar -/
theorem isosceles_60_similar (t1 t2 : IsoscelesTriangle60) : 
  t1.angle1 = t2.angle1 ∧ t1.angle2 = t2.angle2 ∧ t1.angle3 = t2.angle3 := by
  sorry

end isosceles_60_similar_l2727_272702


namespace pet_store_birds_l2727_272731

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 6

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 6

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 2

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds : total_birds = 48 := by
  sorry

end pet_store_birds_l2727_272731


namespace magnitude_a_plus_2b_l2727_272709

/-- Given two vectors a and b in ℝ², prove that |a + 2b| = √17 under certain conditions. -/
theorem magnitude_a_plus_2b (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 1)
  (h2 : ‖b‖ = 2)
  (h3 : a - b = (Real.sqrt 2, Real.sqrt 3)) :
  ‖a + 2 • b‖ = Real.sqrt 17 := by
  sorry

end magnitude_a_plus_2b_l2727_272709


namespace negative_390_same_terminal_side_as_330_l2727_272786

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + β

theorem negative_390_same_terminal_side_as_330 :
  same_terminal_side (-390) 330 :=
sorry

end negative_390_same_terminal_side_as_330_l2727_272786


namespace custom_op_result_l2727_272738

/-- Custom operation * for non-zero integers -/
def custom_op (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

/-- Theorem stating the result of the custom operation given specific conditions -/
theorem custom_op_result (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 11) (h4 : a * b = 24) :
  custom_op a b = 11 / 24 := by
  sorry

#check custom_op_result

end custom_op_result_l2727_272738


namespace spinner_probability_l2727_272775

theorem spinner_probability (p_A p_B p_C p_DE : ℚ) : 
  p_A = 1/3 →
  p_B = 1/6 →
  p_C = p_DE →
  p_A + p_B + p_C + p_DE = 1 →
  p_C = 1/4 := by
sorry

end spinner_probability_l2727_272775


namespace systematic_sampling_l2727_272700

theorem systematic_sampling (total_students : Nat) (num_groups : Nat) (first_group_number : Nat) (target_group : Nat) :
  total_students = 480 →
  num_groups = 30 →
  first_group_number = 5 →
  target_group = 8 →
  (target_group - 1) * (total_students / num_groups) + first_group_number = 117 := by
  sorry

end systematic_sampling_l2727_272700


namespace percentage_below_eight_l2727_272733

/-- Proves that the percentage of students below 8 years of age is 20% -/
theorem percentage_below_eight (total : ℕ) (eight_years : ℕ) (above_eight : ℕ) 
  (h1 : total = 25)
  (h2 : eight_years = 12)
  (h3 : above_eight = 2 * eight_years / 3)
  : (total - eight_years - above_eight) * 100 / total = 20 := by
  sorry

#check percentage_below_eight

end percentage_below_eight_l2727_272733


namespace iron_cubes_melting_l2727_272757

theorem iron_cubes_melting (s1 s2 s3 s_large : ℝ) : 
  s1 = 1 ∧ s2 = 6 ∧ s3 = 8 → 
  s_large^3 = s1^3 + s2^3 + s3^3 →
  s_large = 9 := by
sorry

end iron_cubes_melting_l2727_272757


namespace negation_of_existential_quantifier_l2727_272725

theorem negation_of_existential_quantifier :
  (¬ ∃ x : ℝ, x^2 ≤ |x|) ↔ (∀ x : ℝ, x^2 > |x|) := by
  sorry

end negation_of_existential_quantifier_l2727_272725


namespace quadratic_root_problem_l2727_272752

theorem quadratic_root_problem (m : ℝ) : 
  (3 * (1 : ℝ)^2 + m * 1 - 7 = 0) → 
  (∃ x : ℝ, x ≠ 1 ∧ 3 * x^2 + m * x - 7 = 0 ∧ x = -7/3) := by
  sorry

end quadratic_root_problem_l2727_272752


namespace simplify_fraction_l2727_272719

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end simplify_fraction_l2727_272719


namespace cab_driver_average_income_l2727_272747

def day1_income : ℕ := 300
def day2_income : ℕ := 150
def day3_income : ℕ := 750
def day4_income : ℕ := 400
def day5_income : ℕ := 500
def num_days : ℕ := 5

theorem cab_driver_average_income :
  (day1_income + day2_income + day3_income + day4_income + day5_income) / num_days = 420 := by
  sorry

end cab_driver_average_income_l2727_272747


namespace ratio_equation_solution_l2727_272788

theorem ratio_equation_solution (a b : ℚ) 
  (h1 : b / a = 4) 
  (h2 : b = 20 - 7 * a) : 
  a = 20 / 11 := by
sorry

end ratio_equation_solution_l2727_272788


namespace gcd_sum_problem_l2727_272768

def is_valid (a b c : ℕ+) : Prop :=
  Nat.gcd a.val (Nat.gcd b.val c.val) = 1 ∧
  Nat.gcd a.val (b.val + c.val) > 1 ∧
  Nat.gcd b.val (c.val + a.val) > 1 ∧
  Nat.gcd c.val (a.val + b.val) > 1

theorem gcd_sum_problem :
  (∃ a b c : ℕ+, is_valid a b c ∧ a.val + b.val + c.val = 2015) ∧
  (∀ a b c : ℕ+, is_valid a b c → a.val + b.val + c.val ≥ 30) ∧
  (∃ a b c : ℕ+, is_valid a b c ∧ a.val + b.val + c.val = 30) := by
  sorry

end gcd_sum_problem_l2727_272768


namespace divisibility_by_power_of_three_l2727_272706

def sequence_a (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 4 → a n = a (n - 1) - min (a (n - 2)) (a (n - 3))

theorem divisibility_by_power_of_three (a : ℕ → ℤ) (h : sequence_a a) :
  ∀ k : ℕ, k > 0 → ∃ n : ℕ, (3^k : ℤ) ∣ a n :=
sorry

end divisibility_by_power_of_three_l2727_272706


namespace dereks_lowest_score_l2727_272778

theorem dereks_lowest_score (test1 test2 test3 test4 : ℕ) : 
  test1 = 85 →
  test2 = 78 →
  test1 ≤ 100 →
  test2 ≤ 100 →
  test3 ≤ 100 →
  test4 ≤ 100 →
  test3 ≥ 60 →
  test4 ≥ 60 →
  (test1 + test2 + test3 + test4) / 4 = 84 →
  (min test3 test4 = 73 ∨ min test3 test4 > 73) :=
by sorry

end dereks_lowest_score_l2727_272778


namespace complement_intersection_A_B_l2727_272707

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {3, 5}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {1, 2, 4, 5} :=
by sorry

end complement_intersection_A_B_l2727_272707


namespace function_cuts_x_axis_l2727_272798

theorem function_cuts_x_axis : ∃ x : ℝ, x > 0 ∧ Real.log x + 2 * x = 0 := by sorry

end function_cuts_x_axis_l2727_272798


namespace total_money_l2727_272727

theorem total_money (a b c : ℕ) : 
  a + c = 200 → b + c = 310 → c = 10 → a + b + c = 500 := by
  sorry

end total_money_l2727_272727


namespace range_of_a_for_quadratic_inequality_l2727_272771

theorem range_of_a_for_quadratic_inequality :
  ∀ a : ℝ, (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) := by sorry

end range_of_a_for_quadratic_inequality_l2727_272771


namespace infinite_sqrt_two_plus_l2727_272712

theorem infinite_sqrt_two_plus (x : ℝ) : x > 0 ∧ x^2 = 2 + x → x = 2 := by
  sorry

end infinite_sqrt_two_plus_l2727_272712


namespace john_total_cost_l2727_272766

/-- The total cost for John to raise a child and pay for university tuition -/
def total_cost_for_john : ℕ := 
  let cost_per_year_first_8 := 10000
  let years_first_period := 8
  let years_second_period := 10
  let university_tuition := 250000
  let first_period_cost := cost_per_year_first_8 * years_first_period
  let second_period_cost := 2 * cost_per_year_first_8 * years_second_period
  let total_cost := first_period_cost + second_period_cost + university_tuition
  total_cost / 2

/-- Theorem stating that the total cost for John is $265,000 -/
theorem john_total_cost : total_cost_for_john = 265000 := by
  sorry

end john_total_cost_l2727_272766


namespace contrapositive_proof_l2727_272745

theorem contrapositive_proof (a b : ℝ) :
  (∀ a b, a > b → a - 5 > b - 5) ↔ (∀ a b, a - 5 ≤ b - 5 → a ≤ b) :=
by sorry

end contrapositive_proof_l2727_272745


namespace alpha_square_greater_beta_square_l2727_272704

theorem alpha_square_greater_beta_square 
  (α β : ℝ) 
  (h1 : α ∈ Set.Icc (-Real.pi/2) (Real.pi/2))
  (h2 : β ∈ Set.Icc (-Real.pi/2) (Real.pi/2))
  (h3 : α * Real.sin α - β * Real.sin β > 0) : 
  α^2 > β^2 := by
sorry

end alpha_square_greater_beta_square_l2727_272704


namespace arithmetic_expression_evaluation_l2727_272792

theorem arithmetic_expression_evaluation :
  5 * 7 + 9 * 4 - 30 / 3 + 2^2 = 65 := by
  sorry

end arithmetic_expression_evaluation_l2727_272792


namespace repetitions_for_99_cubes_impossible_2016_cubes_l2727_272762

/-- The number of cubes after x repetitions of the cutting process -/
def num_cubes (x : ℕ) : ℕ := 7 * x + 1

/-- Theorem stating that 14 repetitions are needed to obtain 99 cubes -/
theorem repetitions_for_99_cubes : ∃ x : ℕ, num_cubes x = 99 ∧ x = 14 := by sorry

/-- Theorem stating that it's impossible to obtain 2016 cubes -/
theorem impossible_2016_cubes : ¬∃ x : ℕ, num_cubes x = 2016 := by sorry

end repetitions_for_99_cubes_impossible_2016_cubes_l2727_272762


namespace tensor_A_B_l2727_272784

-- Define the ⊗ operation
def tensor (M N : Set ℝ) : Set ℝ := (M ∪ N) \ (M ∩ N)

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2 - 2*x + 3}

-- Theorem statement
theorem tensor_A_B : tensor A B = {1, 3} := by sorry

end tensor_A_B_l2727_272784


namespace number_of_appliances_l2727_272723

/-- Proves that the number of appliances in a batch is 34, given the purchase price,
    selling price, and total profit. -/
theorem number_of_appliances (purchase_price selling_price total_profit : ℕ) : 
  purchase_price = 230 →
  selling_price = 250 →
  total_profit = 680 →
  (total_profit / (selling_price - purchase_price) : ℕ) = 34 :=
by
  sorry

end number_of_appliances_l2727_272723


namespace subjective_collection_not_set_l2727_272749

-- Define a type for objects in a textbook
structure TextbookObject where
  id : Nat

-- Define a property that determines if an object belongs to a collection
def belongsToCollection (P : TextbookObject → Prop) (obj : TextbookObject) : Prop :=
  P obj

-- Define what it means for a collection to have a clear, objective criterion
def hasClearCriterion (P : TextbookObject → Prop) : Prop :=
  ∀ (obj1 obj2 : TextbookObject), obj1 = obj2 → (P obj1 ↔ P obj2)

-- Define what it means for a collection to be subjective
def isSubjective (P : TextbookObject → Prop) : Prop :=
  ∃ (obj1 obj2 : TextbookObject), obj1 = obj2 ∧ (P obj1 ↔ ¬P obj2)

-- Theorem: A collection with subjective criteria cannot form a well-defined set
theorem subjective_collection_not_set (P : TextbookObject → Prop) :
  isSubjective P → ¬(hasClearCriterion P) :=
by
  sorry

#check subjective_collection_not_set

end subjective_collection_not_set_l2727_272749


namespace sin_sixty_degrees_times_two_l2727_272772

theorem sin_sixty_degrees_times_two : 2 * Real.sin (π / 3) = Real.sqrt 3 := by
  sorry

end sin_sixty_degrees_times_two_l2727_272772


namespace right_triangle_area_in_circle_l2727_272795

/-- The area of a right triangle inscribed in a circle -/
theorem right_triangle_area_in_circle (r : ℝ) (h : r = 5) :
  let a : ℝ := 5 * (10 / 13)
  let b : ℝ := 12 * (10 / 13)
  let c : ℝ := 13 * (10 / 13)
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem
  (c = 2 * r) →        -- Diameter is the hypotenuse
  (1/2 * a * b = 6000/169) :=
by sorry

end right_triangle_area_in_circle_l2727_272795


namespace ten_sentences_per_paragraph_l2727_272782

/-- Represents the reading speed in sentences per hour -/
def reading_speed : ℕ := 200

/-- Represents the number of pages in the book -/
def pages : ℕ := 50

/-- Represents the number of paragraphs per page -/
def paragraphs_per_page : ℕ := 20

/-- Represents the total time taken to read the book in hours -/
def total_reading_time : ℕ := 50

/-- Calculates the number of sentences per paragraph -/
def sentences_per_paragraph : ℕ :=
  (reading_speed * total_reading_time) / (pages * paragraphs_per_page)

/-- Theorem stating that there are 10 sentences per paragraph -/
theorem ten_sentences_per_paragraph : sentences_per_paragraph = 10 := by
  sorry

end ten_sentences_per_paragraph_l2727_272782


namespace related_transitive_l2727_272721

/-- A function is great if it satisfies the given condition for all nonnegative integers m and n. -/
def IsGreat (f : ℕ → ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, f (m + 1) (n + 1) * f m n - f (m + 1) n * f m (n + 1) = 1

/-- Two sequences are related (∼) if there exists a great function satisfying the given conditions. -/
def Related (A B : ℕ → ℤ) : Prop :=
  ∃ f : ℕ → ℕ → ℤ, IsGreat f ∧ (∀ n : ℕ, f n 0 = A n ∧ f 0 n = B n)

/-- The main theorem to be proved. -/
theorem related_transitive (A B C D : ℕ → ℤ) 
  (hAB : Related A B) (hBC : Related B C) (hCD : Related C D) : Related D A := by
  sorry

end related_transitive_l2727_272721


namespace sin_three_pi_halves_l2727_272799

theorem sin_three_pi_halves : Real.sin (3 * Real.pi / 2) = -1 := by
  sorry

end sin_three_pi_halves_l2727_272799


namespace point_inside_circle_l2727_272796

theorem point_inside_circle (a : ℝ) : 
  let P : ℝ × ℝ := (5*a + 1, 12*a)
  ((P.1 - 1)^2 + P.2^2 < 1) ↔ (abs a < 1/13) :=
sorry

end point_inside_circle_l2727_272796


namespace gas_price_difference_l2727_272735

/-- Represents the price difference per gallon between two states --/
def price_difference (nc_price va_price : ℚ) : ℚ := va_price - nc_price

/-- Proves the price difference per gallon between Virginia and North Carolina --/
theorem gas_price_difference 
  (nc_gallons va_gallons : ℚ) 
  (nc_price : ℚ) 
  (total_spent : ℚ) :
  nc_gallons = 10 →
  va_gallons = 10 →
  nc_price = 2 →
  total_spent = 50 →
  price_difference nc_price ((total_spent - nc_gallons * nc_price) / va_gallons) = 1 := by
  sorry


end gas_price_difference_l2727_272735


namespace field_trip_total_cost_l2727_272726

/-- Calculates the total cost of a field trip for multiple classes --/
def field_trip_cost (num_classes : ℕ) (students_per_class : ℕ) (adults_per_class : ℕ) 
                    (student_fee : ℚ) (adult_fee : ℚ) : ℚ :=
  let total_students := num_classes * students_per_class
  let total_adults := num_classes * adults_per_class
  (total_students : ℚ) * student_fee + (total_adults : ℚ) * adult_fee

/-- Theorem stating the total cost of the field trip --/
theorem field_trip_total_cost : 
  field_trip_cost 4 40 5 (11/2) (13/2) = 1010 := by
  sorry

#eval field_trip_cost 4 40 5 (11/2) (13/2)

end field_trip_total_cost_l2727_272726


namespace susan_cats_proof_l2727_272780

/-- The number of cats Bob has -/
def bob_cats : ℕ := 3

/-- The number of cats Susan gives away -/
def cats_given_away : ℕ := 4

/-- The difference in cats between Susan and Bob after Susan gives some away -/
def cat_difference : ℕ := 14

/-- Susan's initial number of cats -/
def susan_initial_cats : ℕ := 25

theorem susan_cats_proof :
  susan_initial_cats = bob_cats + cats_given_away + cat_difference := by
  sorry

end susan_cats_proof_l2727_272780


namespace triangle_ABC_properties_l2727_272751

/-- In triangle ABC, prove that given specific conditions, angle A and the area of the triangle can be determined. -/
theorem triangle_ABC_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given condition
  2 * Real.cos A * (b * Real.cos C + c * Real.cos B) = a →
  -- Additional conditions
  a = Real.sqrt 7 →
  b + c = 5 →
  -- Conclusions
  A = π / 3 ∧ 
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by sorry

end triangle_ABC_properties_l2727_272751


namespace square_inequality_negative_l2727_272790

theorem square_inequality_negative (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end square_inequality_negative_l2727_272790


namespace log_inequality_cube_inequality_l2727_272794

theorem log_inequality_cube_inequality (a b : ℝ) :
  (∀ a b, Real.log a < Real.log b → a^3 < b^3) ∧
  (∃ a b, a^3 < b^3 ∧ ¬(Real.log a < Real.log b)) :=
sorry

end log_inequality_cube_inequality_l2727_272794


namespace asymptote_sum_l2727_272728

/-- Given a rational function y = x / (x^3 + Ax^2 + Bx + C) with integer coefficients A, B, C,
    if the graph has vertical asymptotes at x = -3, 0, 3, then A + B + C = -9 -/
theorem asymptote_sum (A B C : ℤ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 →
    ∃ y : ℝ, y = x / (x^3 + A * x^2 + B * x + C)) →
  (A + B + C = -9) := by
  sorry

end asymptote_sum_l2727_272728


namespace shifted_function_equals_g_l2727_272741

-- Define the original function
def f (x : ℝ) : ℝ := -3 * x + 2

-- Define the shifted function
def g (x : ℝ) : ℝ := -3 * x - 1

-- Define the vertical shift
def shift : ℝ := 3

-- Theorem statement
theorem shifted_function_equals_g :
  ∀ x : ℝ, f x - shift = g x :=
by
  sorry

end shifted_function_equals_g_l2727_272741


namespace jo_bob_max_height_l2727_272759

/-- Represents the state of a hot air balloon ride -/
structure BalloonRide where
  ascent_rate : ℝ
  descent_rate : ℝ
  first_pull_time : ℝ
  release_time : ℝ
  second_pull_time : ℝ

/-- Calculates the maximum height reached during a balloon ride -/
def max_height (ride : BalloonRide) : ℝ :=
  let first_ascent := ride.ascent_rate * ride.first_pull_time
  let descent := ride.descent_rate * ride.release_time
  let second_ascent := ride.ascent_rate * ride.second_pull_time
  first_ascent - descent + second_ascent

/-- Theorem stating the maximum height reached in Jo-Bob's balloon ride -/
theorem jo_bob_max_height :
  let ride : BalloonRide := {
    ascent_rate := 50,
    descent_rate := 10,
    first_pull_time := 15,
    release_time := 10,
    second_pull_time := 15
  }
  max_height ride = 1400 := by
  sorry


end jo_bob_max_height_l2727_272759


namespace shaded_area_theorem_l2727_272732

/-- Represents a rectangle with diagonals divided into 12 equal segments -/
structure DividedRectangle where
  blank_area : ℝ
  total_area : ℝ

/-- The theorem stating the relationship between blank and shaded areas -/
theorem shaded_area_theorem (rect : DividedRectangle) 
  (h1 : rect.blank_area = 10) 
  (h2 : rect.total_area = rect.blank_area + 14) : 
  rect.total_area - rect.blank_area = 14 := by
  sorry

#check shaded_area_theorem

end shaded_area_theorem_l2727_272732


namespace football_games_per_month_l2727_272756

theorem football_games_per_month 
  (total_games : ℕ) 
  (num_months : ℕ) 
  (h1 : total_games = 323) 
  (h2 : num_months = 17) 
  (h3 : total_games % num_months = 0) : 
  total_games / num_months = 19 := by
sorry


end football_games_per_month_l2727_272756


namespace rotation_equivalence_l2727_272729

/-- 
Given a point that is rotated 450 degrees clockwise and x degrees counterclockwise 
about the same center to reach the same final position, prove that x = 270 degrees,
assuming x < 360.
-/
theorem rotation_equivalence (x : ℝ) : 
  (450 % 360 : ℝ) = (360 - x) % 360 → x < 360 → x = 270 := by
  sorry

end rotation_equivalence_l2727_272729


namespace greatest_multiple_of_four_cubed_less_than_2000_l2727_272736

theorem greatest_multiple_of_four_cubed_less_than_2000 :
  ∃ (x : ℕ), x % 4 = 0 ∧ x^3 < 2000 ∧ ∀ (y : ℕ), y % 4 = 0 ∧ y^3 < 2000 → y ≤ x :=
by
  -- The proof goes here
  sorry

end greatest_multiple_of_four_cubed_less_than_2000_l2727_272736


namespace slope_product_constant_l2727_272720

/-- The trajectory C -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1 ∧ p.2 ≠ 0}

/-- The line y = kx -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1}

theorem slope_product_constant
  (M : ℝ × ℝ) (h_M : M ∈ C)
  (k : ℝ)
  (A B : ℝ × ℝ) (h_A : A ∈ C ∩ Line k) (h_B : B ∈ C ∩ Line k)
  (h_AB : A.1 = -B.1 ∧ A.2 = -B.2)
  (h_MA : M.1 ≠ A.1) (h_MB : M.1 ≠ B.1) :
  let K_MA := (M.2 - A.2) / (M.1 - A.1)
  let K_MB := (M.2 - B.2) / (M.1 - B.1)
  K_MA * K_MB = -1/4 :=
sorry

end slope_product_constant_l2727_272720


namespace parabola_shift_l2727_272744

def original_function (x : ℝ) : ℝ := -2 * (x + 1)^2 + 5

def shift_left (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f (x + shift)

def shift_down (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f x - shift

def final_function (x : ℝ) : ℝ := -2 * (x + 3)^2 + 1

theorem parabola_shift :
  ∀ x : ℝ, shift_down (shift_left original_function 2) 4 x = final_function x :=
by sorry

end parabola_shift_l2727_272744


namespace parallel_vectors_m_value_l2727_272724

/-- Two 2D vectors are parallel if their corresponding components are proportional -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (-1, 1)
  let b : ℝ × ℝ := (3, m)
  parallel a b → m = -3 := by
  sorry

end parallel_vectors_m_value_l2727_272724


namespace six_eight_ten_pythagorean_triple_l2727_272773

/-- A Pythagorean triple is a set of three positive integers (a, b, c) where a² + b² = c² --/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The set (6, 8, 10) is a Pythagorean triple --/
theorem six_eight_ten_pythagorean_triple : is_pythagorean_triple 6 8 10 := by
  sorry

end six_eight_ten_pythagorean_triple_l2727_272773


namespace consecutive_integers_sum_l2727_272797

theorem consecutive_integers_sum (n : ℕ) (hn : n > 0) :
  (∃ m : ℤ, (Finset.range n).sum (λ i => m - i) = n) ↔ n % 2 = 1 :=
by sorry

end consecutive_integers_sum_l2727_272797


namespace monotonic_function_a_range_l2727_272789

/-- Given that f(x) = ln x + a/x is monotonically increasing on [2, +∞), 
    prove that the range of values for a is (-∞, 2] -/
theorem monotonic_function_a_range (a : ℝ) :
  (∀ x ≥ 2, Monotone (fun x => Real.log x + a / x)) ↔ a ≤ 2 :=
sorry

end monotonic_function_a_range_l2727_272789


namespace basketball_team_starters_count_l2727_272730

def total_players : ℕ := 18
def num_triplets : ℕ := 3
def num_twins : ℕ := 2
def num_starters : ℕ := 7
def triplets_in_lineup : ℕ := 2
def twins_in_lineup : ℕ := 1

def remaining_players : ℕ := total_players - num_triplets - num_twins

theorem basketball_team_starters_count :
  (Nat.choose num_triplets triplets_in_lineup) *
  (Nat.choose num_twins twins_in_lineup) *
  (Nat.choose remaining_players (num_starters - triplets_in_lineup - twins_in_lineup)) = 4290 := by
  sorry

end basketball_team_starters_count_l2727_272730


namespace male_athletes_to_sample_l2727_272774

theorem male_athletes_to_sample (total_athletes : ℕ) (female_athletes : ℕ) (selection_prob : ℚ) :
  total_athletes = 98 →
  female_athletes = 42 →
  selection_prob = 2 / 7 →
  (total_athletes - female_athletes) * selection_prob = 16 := by
  sorry

end male_athletes_to_sample_l2727_272774


namespace grape_rate_proof_l2727_272746

theorem grape_rate_proof (grapes_kg mangoes_kg mangoes_rate total_paid : ℕ) 
  (h1 : grapes_kg = 8)
  (h2 : mangoes_kg = 9)
  (h3 : mangoes_rate = 65)
  (h4 : total_paid = 1145)
  : ∃ (grape_rate : ℕ), grape_rate * grapes_kg + mangoes_kg * mangoes_rate = total_paid ∧ grape_rate = 70 := by
  sorry

end grape_rate_proof_l2727_272746


namespace average_rounds_is_four_l2727_272716

/-- Represents the distribution of golf rounds played by members -/
structure GolfRoundsDistribution where
  rounds : Fin 6 → ℕ
  members : Fin 6 → ℕ

/-- Calculates the average number of rounds played, rounded to the nearest whole number -/
def averageRoundsRounded (dist : GolfRoundsDistribution) : ℕ :=
  let totalRounds := (Finset.range 6).sum (λ i => dist.rounds i * dist.members i)
  let totalMembers := (Finset.range 6).sum (λ i => dist.members i)
  (totalRounds + totalMembers / 2) / totalMembers

/-- The specific distribution given in the problem -/
def givenDistribution : GolfRoundsDistribution where
  rounds := λ i => i.val + 1
  members := ![4, 3, 5, 6, 2, 7]

theorem average_rounds_is_four :
  averageRoundsRounded givenDistribution = 4 := by
  sorry

end average_rounds_is_four_l2727_272716


namespace vector_dot_product_l2727_272785

/-- Given two 2D vectors a and b, prove that their dot product is -18. -/
theorem vector_dot_product (a b : ℝ × ℝ) : 
  a = (1, -3) → b = (3, 7) → a.1 * b.1 + a.2 * b.2 = -18 := by
  sorry

end vector_dot_product_l2727_272785


namespace arccos_lt_arcsin_iff_l2727_272770

theorem arccos_lt_arcsin_iff (x : ℝ) : 
  Real.arccos x < Real.arcsin x ↔ x ∈ Set.Ioo (1 / Real.sqrt 2) 1 :=
by
  sorry

end arccos_lt_arcsin_iff_l2727_272770


namespace quadratic_inequality_solution_set_l2727_272764

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, 2 * x^2 - x - 1 > 0 ↔ x < -1/2 ∨ x > 1 := by
  sorry

end quadratic_inequality_solution_set_l2727_272764


namespace car_travel_distance_l2727_272714

/-- Proves that a car traveling at 70 kmh for a certain time covers a distance of 105 km,
    given that if it had traveled 35 kmh faster, the trip would have lasted 30 minutes less. -/
theorem car_travel_distance :
  ∀ (time : ℝ),
  time > 0 →
  let distance := 70 * time
  let faster_time := time - 0.5
  let faster_speed := 70 + 35
  distance = faster_speed * faster_time →
  distance = 105 :=
by
  sorry

end car_travel_distance_l2727_272714


namespace cosine_value_for_given_point_l2727_272734

theorem cosine_value_for_given_point (α : Real) :
  (∃ r : Real, r > 0 ∧ r^2 = 1 + 3) →
  (1, -Real.sqrt 3) ∈ {(x, y) | x = r * Real.cos α ∧ y = r * Real.sin α} →
  Real.cos α = 1/2 := by
sorry

end cosine_value_for_given_point_l2727_272734


namespace spade_operation_result_l2727_272787

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_operation_result : spade 1.5 (spade 2.5 (spade 4.5 6)) = 0.5 := by
  sorry

end spade_operation_result_l2727_272787


namespace mary_cake_flour_l2727_272753

/-- Given a recipe that requires a certain amount of flour and the amount already added,
    calculate the remaining amount of flour needed. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- The problem statement -/
theorem mary_cake_flour : remaining_flour 9 3 = 6 := by
  sorry

end mary_cake_flour_l2727_272753


namespace cube_surface_area_l2727_272715

/-- The surface area of a cube with edge length 6a is 216a² -/
theorem cube_surface_area (a : ℝ) : 
  let edge_length : ℝ := 6 * a
  let surface_area : ℝ := 6 * (edge_length ^ 2)
  surface_area = 216 * (a ^ 2) := by
  sorry

end cube_surface_area_l2727_272715


namespace california_permutations_count_l2727_272793

/-- The number of distinct permutations of a word with repeated letters -/
def wordPermutations (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The number of distinct permutations of CALIFORNIA -/
def californiaPermutations : ℕ := wordPermutations 10 [3, 2]

theorem california_permutations_count :
  californiaPermutations = 302400 := by
  sorry

end california_permutations_count_l2727_272793


namespace tshirt_company_profit_l2727_272769

/-- Calculates the daily profit of a t-shirt company given specific conditions -/
theorem tshirt_company_profit (
  num_employees : ℕ
  ) (shirts_per_employee : ℕ
  ) (shift_hours : ℕ
  ) (hourly_wage : ℚ
  ) (per_shirt_bonus : ℚ
  ) (shirt_price : ℚ
  ) (nonemployee_expenses : ℚ
  ) (h1 : num_employees = 20
  ) (h2 : shirts_per_employee = 20
  ) (h3 : shift_hours = 8
  ) (h4 : hourly_wage = 12
  ) (h5 : per_shirt_bonus = 5
  ) (h6 : shirt_price = 35
  ) (h7 : nonemployee_expenses = 1000
  ) : (num_employees * shirts_per_employee * shirt_price) -
      (num_employees * shift_hours * hourly_wage +
       num_employees * shirts_per_employee * per_shirt_bonus +
       nonemployee_expenses) = 9080 := by
  sorry


end tshirt_company_profit_l2727_272769


namespace complex_square_condition_l2727_272743

theorem complex_square_condition (a b : ℝ) : 
  (∃ a b : ℝ, (Complex.I : ℂ)^2 = -1 ∧ (a + b * Complex.I)^2 = 2 * Complex.I ∧ ¬(a = 1 ∧ b = 1)) ∧
  ((a = 1 ∧ b = 1) → (a + b * Complex.I)^2 = 2 * Complex.I) :=
sorry

end complex_square_condition_l2727_272743
