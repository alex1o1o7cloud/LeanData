import Mathlib

namespace integer_roots_of_polynomial_l785_78540

def polynomial (x : ℤ) : ℤ := x^3 - 4*x^2 - 14*x + 24

theorem integer_roots_of_polynomial :
  {x : ℤ | polynomial x = 0} = {-4, -3, 3} := by sorry

end integer_roots_of_polynomial_l785_78540


namespace quadratic_roots_proof_l785_78502

theorem quadratic_roots_proof (x₁ x₂ : ℝ) : x₁ = -1 ∧ x₂ = 6 →
  (x₁^2 - 5*x₁ - 6 = 0) ∧ (x₂^2 - 5*x₂ - 6 = 0) := by
  sorry

end quadratic_roots_proof_l785_78502


namespace ghee_mixture_original_quantity_l785_78515

/-- Proves that the original quantity of a ghee mixture is 10 kg given specific conditions -/
theorem ghee_mixture_original_quantity :
  ∀ (x : ℝ),
  (0.6 * x = x - 0.4 * x) →  -- 60% pure ghee, 40% vanaspati in original mixture
  (0.2 * (x + 10) = 0.4 * x) →  -- 20% vanaspati after adding 10 kg pure ghee
  x = 10 :=
by
  sorry

end ghee_mixture_original_quantity_l785_78515


namespace f_is_even_and_increasing_l785_78557

def f (x : ℝ) := -x^2

theorem f_is_even_and_increasing :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, x < y ∧ y ≤ 0 → f x < f y) :=
sorry

end f_is_even_and_increasing_l785_78557


namespace arithmetic_sequence_product_l785_78565

theorem arithmetic_sequence_product (a : ℝ) (d : ℝ) : 
  (a + 6 * d = 20) → (d = 2) → (a * (a + d) = 80) := by
  sorry

end arithmetic_sequence_product_l785_78565


namespace purple_candies_count_l785_78570

/-- The number of purple candies in a box of rainbow nerds -/
def purple_candies : ℕ := 10

/-- The number of yellow candies in a box of rainbow nerds -/
def yellow_candies : ℕ := purple_candies + 4

/-- The number of green candies in a box of rainbow nerds -/
def green_candies : ℕ := yellow_candies - 2

/-- The total number of candies in the box -/
def total_candies : ℕ := 36

/-- Theorem stating that the number of purple candies is 10 -/
theorem purple_candies_count : 
  purple_candies = 10 ∧ 
  yellow_candies = purple_candies + 4 ∧ 
  green_candies = yellow_candies - 2 ∧ 
  purple_candies + yellow_candies + green_candies = total_candies :=
by sorry

end purple_candies_count_l785_78570


namespace infinitely_many_solutions_iff_m_eq_two_l785_78586

/-- A system of linear equations in x and y with parameter m -/
structure LinearSystem (m : ℝ) where
  eq1 : ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ
  h1 : ∀ x y, eq1 x y = m * x + 4 * y - (m + 2)
  h2 : ∀ x y, eq2 x y = x + m * y - m

/-- The system has infinitely many solutions -/
def HasInfinitelySolutions (sys : LinearSystem m) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ sys.eq1 x₁ y₁ = 0 ∧ sys.eq2 x₁ y₁ = 0 ∧ sys.eq1 x₂ y₂ = 0 ∧ sys.eq2 x₂ y₂ = 0

/-- The main theorem: the system has infinitely many solutions iff m = 2 -/
theorem infinitely_many_solutions_iff_m_eq_two (m : ℝ) (sys : LinearSystem m) :
  HasInfinitelySolutions sys ↔ m = 2 := by
  sorry

end infinitely_many_solutions_iff_m_eq_two_l785_78586


namespace problem_solution_l785_78567

theorem problem_solution (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x * y = 3) 
  (h3 : x^3 - x^2 - 4*x + 4 = 0) 
  (h4 : y^3 - y^2 - 4*y + 4 = 0) : 
  x + x^3/y^2 + y^3/x^2 + y = 174 := by sorry

end problem_solution_l785_78567


namespace solution_set_for_a_equals_one_solution_range_for_a_l785_78531

-- Define the function f(x) = |x-a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | |x - 1| > (1/2) * (x + 1)} = {x : ℝ | x > 3 ∨ x < 1/3} := by sorry

-- Part 2
theorem solution_range_for_a :
  ∀ a : ℝ, (∃ x : ℝ, |x - a| + |x - 2| ≤ 3) ↔ -1 ≤ a ∧ a ≤ 5 := by sorry

end solution_set_for_a_equals_one_solution_range_for_a_l785_78531


namespace smallest_prime_factor_in_C_l785_78541

def C : Set Nat := {33, 35, 37, 39, 41}

def has_smallest_prime_factor (n : Nat) (s : Set Nat) : Prop :=
  n ∈ s ∧ ∀ m ∈ s, (Nat.minFac n ≤ Nat.minFac m)

theorem smallest_prime_factor_in_C :
  has_smallest_prime_factor 33 C ∧ has_smallest_prime_factor 39 C ∧
  ∀ x ∈ C, has_smallest_prime_factor x C → (x = 33 ∨ x = 39) :=
sorry

end smallest_prime_factor_in_C_l785_78541


namespace tan_product_special_angles_l785_78551

theorem tan_product_special_angles :
  let A : Real := 30 * π / 180
  let B : Real := 60 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = 4 + 2 * Real.sqrt 3 := by
  sorry

end tan_product_special_angles_l785_78551


namespace subtracted_value_l785_78544

theorem subtracted_value (x : ℤ) (h : 282 = x + 133) : x - 11 = 138 := by
  sorry

end subtracted_value_l785_78544


namespace unique_perpendicular_line_l785_78506

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

structure Line3D where
  point : Point3D
  direction : Point3D

-- Define perpendicularity between a line and a plane
def isPerpendicular (l : Line3D) (p : Plane) : Prop :=
  l.direction.x * p.a + l.direction.y * p.b + l.direction.z * p.c = 0

-- State the theorem
theorem unique_perpendicular_line 
  (P : Point3D) (π : Plane) : 
  ∃! l : Line3D, l.point = P ∧ isPerpendicular l π :=
sorry

end unique_perpendicular_line_l785_78506


namespace column_arrangement_l785_78578

theorem column_arrangement (total_people : ℕ) 
  (h1 : total_people = 30 * 16) 
  (h2 : ∃ (people_per_column : ℕ), total_people = people_per_column * 10) : 
  ∃ (people_per_column : ℕ), total_people = people_per_column * 10 ∧ people_per_column = 48 :=
by sorry

end column_arrangement_l785_78578


namespace obtuse_triangle_necessary_not_sufficient_l785_78577

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- Definition of an obtuse triangle --/
def isObtuse (t : Triangle) : Prop :=
  t.a^2 + t.b^2 < t.c^2 ∨ t.b^2 + t.c^2 < t.a^2 ∨ t.c^2 + t.a^2 < t.b^2

theorem obtuse_triangle_necessary_not_sufficient :
  (∀ t : Triangle, isObtuse t → (t.a^2 + t.b^2 < t.c^2 ∨ t.b^2 + t.c^2 < t.a^2 ∨ t.c^2 + t.a^2 < t.b^2)) ∧
  (∃ t : Triangle, (t.a^2 + t.b^2 < t.c^2 ∨ t.b^2 + t.c^2 < t.a^2 ∨ t.c^2 + t.a^2 < t.b^2) ∧ ¬isObtuse t) :=
by sorry

end obtuse_triangle_necessary_not_sufficient_l785_78577


namespace contrapositive_false_l785_78598

theorem contrapositive_false : 
  ¬(∀ x : ℝ, x^2 - 1 = 0 → x = 1) :=
sorry

end contrapositive_false_l785_78598


namespace arithmetic_sequence_sum_l785_78564

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 2 + a 3 = 9) →
  (a 4 + a 5 + a 6 = 27) →
  (a 7 + a 8 + a 9 = 45) :=
by
  sorry

end arithmetic_sequence_sum_l785_78564


namespace polynomial_equation_solution_l785_78587

theorem polynomial_equation_solution (p : ℝ → ℝ) :
  (∀ x : ℝ, p (5 * x)^2 - 3 = p (5 * x^2 + 1)) →
  (p = λ _ ↦ (1 + Real.sqrt 13) / 2) ∨ (p = λ _ ↦ (1 - Real.sqrt 13) / 2) :=
by sorry

end polynomial_equation_solution_l785_78587


namespace pineapple_weight_l785_78574

theorem pineapple_weight (P : ℝ) 
  (h1 : P > 0)
  (h2 : P / 6 + 2 / 5 * (5 / 6 * P) + 2 / 3 * (P / 2) + 120 = P) : 
  P = 720 := by
  sorry

end pineapple_weight_l785_78574


namespace train_bridge_crossing_time_l785_78573

/-- Proves that a train of given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 250) 
  (h2 : train_speed_kmph = 72) 
  (h3 : bridge_length = 1250) : 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 75 := by
  sorry

#check train_bridge_crossing_time

end train_bridge_crossing_time_l785_78573


namespace probability_girl_grade4_l785_78536

/-- The probability of selecting a girl from grade 4 in a school playground -/
theorem probability_girl_grade4 (g3 b3 g4 b4 g5 b5 : ℕ) : 
  g3 = 28 → b3 = 35 → g4 = 45 → b4 = 42 → g5 = 38 → b5 = 51 →
  (g4 : ℚ) / (g3 + b3 + g4 + b4 + g5 + b5) = 45 / 239 := by
  sorry

end probability_girl_grade4_l785_78536


namespace smallest_sum_of_squares_l785_78596

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 187 → ∃ (a b : ℕ), a^2 - b^2 = 187 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 205 :=
by sorry

end smallest_sum_of_squares_l785_78596


namespace point_outside_intersecting_line_l785_78510

/-- A line ax + by = 1 intersects a unit circle if and only if the distance
    from the origin to the line is less than 1 -/
def line_intersects_circle (a b : ℝ) : Prop :=
  (|1| / Real.sqrt (a^2 + b^2)) < 1

/-- A point (x,y) is outside the unit circle if its distance from the origin is greater than 1 -/
def point_outside_circle (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) > 1

theorem point_outside_intersecting_line (a b : ℝ) :
  line_intersects_circle a b → point_outside_circle a b :=
by sorry

end point_outside_intersecting_line_l785_78510


namespace line_equation_proof_l785_78597

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a function to check if two lines are parallel
def linesParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- Theorem statement
theorem line_equation_proof (l : Line) (p : Point) (given_line : Line) :
  pointOnLine p l ∧ 
  p = Point.mk 0 3 ∧ 
  linesParallel l given_line ∧ 
  given_line = Line.mk 1 (-1) (-1) →
  l = Line.mk 1 (-1) 3 :=
by sorry

end line_equation_proof_l785_78597


namespace complex_modulus_equation_l785_78556

theorem complex_modulus_equation (t : ℝ) : 
  t > 0 → Complex.abs (8 + 3 * t * Complex.I) = 13 → t = Real.sqrt 105 / 3 := by
  sorry

end complex_modulus_equation_l785_78556


namespace van_rental_cost_equation_l785_78534

theorem van_rental_cost_equation (x : ℝ) (h : x > 2) :
  180 / (x - 2) - 180 / x = 3 :=
sorry

end van_rental_cost_equation_l785_78534


namespace rectangular_prism_volume_l785_78508

theorem rectangular_prism_volume (a b c : ℕ) 
  (h1 : 4 * ((a - 2) + (b - 2) + (c - 2)) = 40)
  (h2 : 2 * ((a - 2) * (b - 2) + (a - 2) * (c - 2) + (b - 2) * (c - 2)) = 66) :
  a * b * c = 150 := by
  sorry

end rectangular_prism_volume_l785_78508


namespace cubic_factorization_sum_of_squares_l785_78517

theorem cubic_factorization_sum_of_squares (p q r s t u : ℤ) :
  (∀ x : ℤ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210 :=
by sorry

end cubic_factorization_sum_of_squares_l785_78517


namespace mustard_at_third_table_l785_78543

theorem mustard_at_third_table 
  (first_table : Real) 
  (second_table : Real) 
  (total_mustard : Real) 
  (h1 : first_table = 0.25)
  (h2 : second_table = 0.25)
  (h3 : total_mustard = 0.88) :
  total_mustard - (first_table + second_table) = 0.38 := by
sorry

end mustard_at_third_table_l785_78543


namespace triangle_area_from_squares_l785_78585

theorem triangle_area_from_squares (a b c : ℝ) (h1 : a^2 = 64) (h2 : b^2 = 121) (h3 : c^2 = 169)
  (h4 : a^2 + b^2 = c^2) : (1/2) * a * b = 44 := by
  sorry

end triangle_area_from_squares_l785_78585


namespace painting_time_theorem_l785_78512

def time_for_lily : ℕ := 5
def time_for_rose : ℕ := 7
def time_for_orchid : ℕ := 3
def time_for_vine : ℕ := 2

def num_lilies : ℕ := 17
def num_roses : ℕ := 10
def num_orchids : ℕ := 6
def num_vines : ℕ := 20

def total_time : ℕ := time_for_lily * num_lilies + time_for_rose * num_roses + 
                       time_for_orchid * num_orchids + time_for_vine * num_vines

theorem painting_time_theorem : total_time = 213 := by
  sorry

end painting_time_theorem_l785_78512


namespace dividing_line_halves_area_l785_78559

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the L-shaped region -/
def LShapedRegion : Set Point := {p | 
  (0 ≤ p.x ∧ p.x ≤ 4 ∧ 0 ≤ p.y ∧ p.y ≤ 4) ∨
  (4 < p.x ∧ p.x ≤ 7 ∧ 0 ≤ p.y ∧ p.y ≤ 2)
}

/-- Calculates the area of a region -/
noncomputable def area (s : Set Point) : ℝ := sorry

/-- The line y = (5/7)x -/
def dividingLine (p : Point) : Prop := p.y = (5/7) * p.x

/-- Regions above and below the dividing line -/
def upperRegion : Set Point := {p ∈ LShapedRegion | p.y ≥ (5/7) * p.x}
def lowerRegion : Set Point := {p ∈ LShapedRegion | p.y ≤ (5/7) * p.x}

theorem dividing_line_halves_area : 
  area upperRegion = area lowerRegion := by sorry

end dividing_line_halves_area_l785_78559


namespace product_of_sums_geq_one_l785_78599

theorem product_of_sums_geq_one (a b c d : ℝ) 
  (h1 : a + b = 1) (h2 : c * d = 1) : 
  (a * c + b * d) * (a * d + b * c) ≥ 1 := by sorry

end product_of_sums_geq_one_l785_78599


namespace high_five_problem_l785_78521

theorem high_five_problem (n : ℕ) (h : n > 0) :
  (∀ (person : Fin n), (person.val < n → 2 * 2021 = n - 1)) →
  (n = 4043 ∧ Nat.choose n 3 = 11024538580) := by
  sorry

end high_five_problem_l785_78521


namespace three_std_dev_below_mean_l785_78550

/-- Given a distribution with mean 16.2 and standard deviation 2.3,
    the value 3 standard deviations below the mean is 9.3 -/
theorem three_std_dev_below_mean (μ : ℝ) (σ : ℝ) 
  (h_mean : μ = 16.2) (h_std_dev : σ = 2.3) :
  μ - 3 * σ = 9.3 := by
  sorry

end three_std_dev_below_mean_l785_78550


namespace unique_divisor_property_l785_78581

def divisor_count (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem unique_divisor_property : ∃! n : ℕ, n > 0 ∧ n = 100 * divisor_count n :=
  sorry

end unique_divisor_property_l785_78581


namespace distance_between_specific_planes_l785_78572

/-- The distance between two planes given by their equations -/
def distance_between_planes (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ :=
  sorry

/-- Theorem: The distance between the planes x - 2y + 2z = 9 and 2x - 4y + 4z = 18 is 0 -/
theorem distance_between_specific_planes :
  distance_between_planes 1 (-2) 2 9 2 (-4) 4 18 = 0 := by
  sorry

end distance_between_specific_planes_l785_78572


namespace divide_fractions_l785_78569

theorem divide_fractions : (7 : ℚ) / 3 / ((5 : ℚ) / 4) = 28 / 15 := by sorry

end divide_fractions_l785_78569


namespace paula_candy_problem_l785_78523

theorem paula_candy_problem (initial_candies : ℕ) (num_friends : ℕ) (candies_per_friend : ℕ)
  (h1 : initial_candies = 20)
  (h2 : num_friends = 6)
  (h3 : candies_per_friend = 4) :
  num_friends * candies_per_friend - initial_candies = 4 := by
  sorry

end paula_candy_problem_l785_78523


namespace alternating_number_composite_l785_78537

def alternating_number (k : ℕ) : ℕ := 
  (10^(2*k+1) - 1) / 99

theorem alternating_number_composite (k : ℕ) (h : k ≥ 2) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ alternating_number k = a * b :=
sorry

end alternating_number_composite_l785_78537


namespace expression_simplification_l785_78582

theorem expression_simplification :
  1 + (1 : ℝ) / (1 + Real.sqrt 2) - 1 / (1 - Real.sqrt 5) =
  1 + (-Real.sqrt 2 - Real.sqrt 5) / (1 + Real.sqrt 2 - Real.sqrt 5 - Real.sqrt 10) := by
  sorry

end expression_simplification_l785_78582


namespace max_quotient_value_l785_78509

theorem max_quotient_value (a b : ℝ) 
  (ha : 210 ≤ a ∧ a ≤ 430) 
  (hb : 590 ≤ b ∧ b ≤ 1190) : 
  (∀ x y, 210 ≤ x ∧ x ≤ 430 ∧ 590 ≤ y ∧ y ≤ 1190 → y / x ≤ 1190 / 210) :=
by sorry

end max_quotient_value_l785_78509


namespace student_A_more_stable_l785_78563

/-- Represents a student's jumping rope performance -/
structure JumpRopePerformance where
  average_score : ℝ
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines when one performance is more stable than another -/
def more_stable (a b : JumpRopePerformance) : Prop :=
  a.variance < b.variance

theorem student_A_more_stable (
  student_A student_B : JumpRopePerformance
) (h1 : student_A.average_score = student_B.average_score)
  (h2 : student_A.variance = 0.06)
  (h3 : student_B.variance = 0.35) :
  more_stable student_A student_B :=
sorry

end student_A_more_stable_l785_78563


namespace petes_backward_speed_l785_78528

/-- Pete's backward walking speed problem -/
theorem petes_backward_speed (petes_hand_speed tracy_cartwheel_speed susans_speed petes_backward_speed : ℝ) : 
  petes_hand_speed = 2 →
  petes_hand_speed = (1 / 4) * tracy_cartwheel_speed →
  tracy_cartwheel_speed = 2 * susans_speed →
  petes_backward_speed = 3 * susans_speed →
  petes_backward_speed = 12 := by
  sorry

end petes_backward_speed_l785_78528


namespace sequence_range_l785_78546

theorem sequence_range (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_recur : ∀ n, a (n + 1) ≥ 2 * a n + 1) 
  (h_bound : ∀ n, a n < 2^(n + 1)) : 
  0 < a 1 ∧ a 1 ≤ 3 := by
  sorry

end sequence_range_l785_78546


namespace angle_with_special_complement_supplement_l785_78539

theorem angle_with_special_complement_supplement : ∀ x : ℝ,
  (90 - x = (1 / 3) * (180 - x)) → x = 45 := by sorry

end angle_with_special_complement_supplement_l785_78539


namespace largest_prime_factor_3434_l785_78518

def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_3434 : largest_prime_factor 3434 = 7 := by sorry

end largest_prime_factor_3434_l785_78518


namespace yard_raking_time_l785_78542

theorem yard_raking_time (your_time brother_time together_time : ℝ) 
  (h1 : brother_time = 45)
  (h2 : together_time = 18)
  (h3 : 1 / your_time + 1 / brother_time = 1 / together_time) :
  your_time = 30 := by
  sorry

end yard_raking_time_l785_78542


namespace marble_probability_difference_l785_78529

theorem marble_probability_difference :
  let total_marbles : ℕ := 4000
  let red_marbles : ℕ := 1500
  let black_marbles : ℕ := 2500
  let p_same : ℚ := (red_marbles.choose 2 + black_marbles.choose 2) / total_marbles.choose 2
  let p_different : ℚ := (red_marbles * black_marbles) / total_marbles.choose 2
  |p_same - p_different| = 3 / 50 := by
sorry

end marble_probability_difference_l785_78529


namespace triangle_property_l785_78595

open Real

theorem triangle_property (A B C a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  b * cos C + sqrt 3 * b * sin C = a + c →
  B = π / 3 ∧
  (b = sqrt 3 → -sqrt 3 < 2 * a - c ∧ 2 * a - c < 2 * sqrt 3) := by
  sorry


end triangle_property_l785_78595


namespace half_power_five_decimal_l785_78579

theorem half_power_five_decimal : (1/2)^5 = 0.03125 := by
  sorry

end half_power_five_decimal_l785_78579


namespace prob_drawing_10_red_in_12_draws_l785_78593

-- Define the number of white and red balls
def white_balls : ℕ := 5
def red_balls : ℕ := 3
def total_balls : ℕ := white_balls + red_balls

-- Define the probability of drawing a red ball
def prob_red : ℚ := red_balls / total_balls

-- Define the probability of drawing a white ball
def prob_white : ℚ := white_balls / total_balls

-- Define the number of draws
def total_draws : ℕ := 12

-- Define the number of red balls needed to stop
def red_balls_to_stop : ℕ := 10

-- Define the probability of the event
def prob_event : ℚ := (Nat.choose (total_draws - 1) (red_balls_to_stop - 1)) * 
                      (prob_red ^ red_balls_to_stop) * 
                      (prob_white ^ (total_draws - red_balls_to_stop))

-- Theorem statement
theorem prob_drawing_10_red_in_12_draws : 
  prob_event = (Nat.choose 11 9) * ((3 / 8) ^ 10) * ((5 / 8) ^ 2) :=
sorry

end prob_drawing_10_red_in_12_draws_l785_78593


namespace square_land_area_l785_78547

/-- A square land plot with side length 30 units has an area of 900 square units. -/
theorem square_land_area (side_length : ℝ) (h1 : side_length = 30) :
  side_length * side_length = 900 := by
  sorry

end square_land_area_l785_78547


namespace paint_for_sun_l785_78500

/-- The amount of paint left for the sun, given Mary's and Mike's usage --/
def paint_left_for_sun (mary_paint : ℝ) (mike_extra_paint : ℝ) (total_paint : ℝ) : ℝ :=
  total_paint - (mary_paint + (mary_paint + mike_extra_paint))

/-- Theorem stating the amount of paint left for the sun --/
theorem paint_for_sun :
  paint_left_for_sun 3 2 13 = 5 := by
  sorry

end paint_for_sun_l785_78500


namespace total_fruits_is_107_l785_78552

-- Define the number of oranges and apples picked by George and Amelia
def george_oranges : ℕ := 45
def amelia_apples : ℕ := 15
def george_amelia_apple_diff : ℕ := 5
def george_amelia_orange_diff : ℕ := 18

-- Define the number of apples picked by George
def george_apples : ℕ := amelia_apples + george_amelia_apple_diff

-- Define the number of oranges picked by Amelia
def amelia_oranges : ℕ := george_oranges - george_amelia_orange_diff

-- Define the total number of fruits picked
def total_fruits : ℕ := george_oranges + george_apples + amelia_oranges + amelia_apples

-- Theorem statement
theorem total_fruits_is_107 : total_fruits = 107 := by sorry

end total_fruits_is_107_l785_78552


namespace complement_of_union_l785_78558

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 5}

theorem complement_of_union :
  (U \ (A ∪ B)) = {2, 4} := by
  sorry

end complement_of_union_l785_78558


namespace line_shift_theorem_l785_78530

-- Define the original line
def original_line (x : ℝ) : ℝ := -2 * x + 1

-- Define the shift amount
def shift : ℝ := 2

-- Define the shifted line
def shifted_line (x : ℝ) : ℝ := original_line (x + shift)

-- Theorem statement
theorem line_shift_theorem :
  ∀ x : ℝ, shifted_line x = -2 * x - 3 := by
  sorry

end line_shift_theorem_l785_78530


namespace drink_expense_l785_78522

def initial_amount : ℝ := 9
def final_amount : ℝ := 6
def additional_expense : ℝ := 1.25

theorem drink_expense : 
  initial_amount - final_amount - additional_expense = 1.75 := by
  sorry

end drink_expense_l785_78522


namespace probability_five_consecutive_heads_eight_flips_l785_78592

/-- A sequence of coin flips -/
def CoinFlipSequence := List Bool

/-- The length of a coin flip sequence -/
def sequenceLength : CoinFlipSequence → Nat :=
  List.length

/-- Checks if a sequence has at least n consecutive heads -/
def hasConsecutiveHeads (n : Nat) : CoinFlipSequence → Bool :=
  sorry

/-- All possible outcomes of flipping a coin n times -/
def allOutcomes (n : Nat) : List CoinFlipSequence :=
  sorry

/-- Count of sequences with at least n consecutive heads -/
def countConsecutiveHeads (n : Nat) (totalFlips : Nat) : Nat :=
  sorry

/-- Probability of getting at least n consecutive heads in m flips -/
def probabilityConsecutiveHeads (n : Nat) (m : Nat) : Rat :=
  sorry

theorem probability_five_consecutive_heads_eight_flips :
  probabilityConsecutiveHeads 5 8 = 23 / 256 :=
sorry

end probability_five_consecutive_heads_eight_flips_l785_78592


namespace hexagon_largest_angle_l785_78548

theorem hexagon_largest_angle (a b c d e f : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  b / a = 3 / 2 →
  c / a = 3 / 2 →
  d / a = 2 →
  e / a = 2 →
  f / a = 5 / 2 →
  a + b + c + d + e + f = 720 →
  f = 1200 / 7 :=
by sorry

end hexagon_largest_angle_l785_78548


namespace parabola_ellipse_coincident_foci_l785_78533

/-- Given a parabola and an ellipse, proves that if their foci coincide, then the parameter of the parabola is 4. -/
theorem parabola_ellipse_coincident_foci (p : ℝ) : 
  (∀ x y, y^2 = 2*p*x → x^2/6 + y^2/2 = 1 → x = p/2 ∧ x = 2) → p = 4 := by
  sorry

end parabola_ellipse_coincident_foci_l785_78533


namespace rectangular_box_diagonals_l785_78527

theorem rectangular_box_diagonals 
  (a b c : ℝ) 
  (surface_area : 2 * (a * b + b * c + c * a) = 166) 
  (edge_sum : 4 * (a + b + c) = 64) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 12 * Real.sqrt 10 := by
  sorry

end rectangular_box_diagonals_l785_78527


namespace percent_students_with_cats_l785_78501

/-- Given a school with 500 students where 75 students own cats,
    prove that 15% of the students own cats. -/
theorem percent_students_with_cats :
  let total_students : ℕ := 500
  let cat_owners : ℕ := 75
  (cat_owners : ℚ) / total_students * 100 = 15 := by
  sorry

end percent_students_with_cats_l785_78501


namespace geometric_series_sum_l785_78535

theorem geometric_series_sum : ∀ (a r : ℝ) (n : ℕ),
  a = 2 → r = 3 → n = 6 →
  a * (r^n - 1) / (r - 1) = 728 := by
  sorry

end geometric_series_sum_l785_78535


namespace percentage_of_material_A_in_first_solution_l785_78503

/-- Given two solutions and their mixture, proves the percentage of material A in the first solution -/
theorem percentage_of_material_A_in_first_solution 
  (x : ℝ) -- Percentage of material A in the first solution
  (h1 : x + 80 = 100) -- First solution composition
  (h2 : 30 + 70 = 100) -- Second solution composition
  (h3 : 0.8 * x + 0.2 * 30 = 22) -- Mixture composition
  : x = 20 := by
  sorry

end percentage_of_material_A_in_first_solution_l785_78503


namespace fibonacci_7_l785_78516

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_7 : fibonacci 6 = 13 := by
  sorry

end fibonacci_7_l785_78516


namespace football_team_throwers_l785_78526

/-- Represents the number of throwers on a football team. -/
def num_throwers : ℕ := 52

/-- Represents the total number of players on the football team. -/
def total_players : ℕ := 70

/-- Represents the total number of right-handed players on the team. -/
def right_handed_players : ℕ := 64

theorem football_team_throwers :
  num_throwers = 52 ∧
  total_players = 70 ∧
  right_handed_players = 64 ∧
  num_throwers ≤ total_players ∧
  num_throwers ≤ right_handed_players ∧
  (total_players - num_throwers) % 3 = 0 ∧
  right_handed_players = num_throwers + 2 * ((total_players - num_throwers) / 3) :=
by sorry

end football_team_throwers_l785_78526


namespace triple_area_right_triangle_l785_78553

/-- Given a right triangle with hypotenuse a+b and legs a and b, 
    the area of a triangle that is three times the area of this right triangle is 3/2ab. -/
theorem triple_area_right_triangle (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  3 * (1/2 * a * b) = 3/2 * a * b := by sorry

end triple_area_right_triangle_l785_78553


namespace sum_of_polynomials_l785_78519

/-- Given polynomials f, g, and h, prove their sum equals the simplified polynomial -/
theorem sum_of_polynomials (x : ℝ) :
  let f := fun x : ℝ => -4 * x^2 + 2 * x - 5
  let g := fun x : ℝ => -6 * x^2 + 4 * x - 9
  let h := fun x : ℝ => 6 * x^2 + 6 * x + 2
  f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end sum_of_polynomials_l785_78519


namespace percentage_passed_l785_78590

def total_students : ℕ := 800
def failed_students : ℕ := 520

theorem percentage_passed : 
  (((total_students - failed_students) : ℚ) / total_students) * 100 = 35 := by
  sorry

end percentage_passed_l785_78590


namespace min_balls_for_target_color_l785_78505

def orange_balls : ℕ := 26
def purple_balls : ℕ := 21
def brown_balls : ℕ := 20
def gray_balls : ℕ := 15
def silver_balls : ℕ := 12
def golden_balls : ℕ := 10

def target_count : ℕ := 17

theorem min_balls_for_target_color :
  ∃ (n : ℕ), 
    (∀ (m : ℕ), m < n → 
      ∃ (o p b g s g' : ℕ), 
        o + p + b + g + s + g' = m ∧ 
        o ≤ orange_balls ∧ 
        p ≤ purple_balls ∧ 
        b ≤ brown_balls ∧ 
        g ≤ gray_balls ∧ 
        s ≤ silver_balls ∧ 
        g' ≤ golden_balls ∧
        o < target_count ∧ 
        p < target_count ∧ 
        b < target_count ∧ 
        g < target_count ∧ 
        s < target_count ∧ 
        g' < target_count) ∧
    (∀ (o p b g s g' : ℕ), 
      o + p + b + g + s + g' = n → 
      o ≤ orange_balls → 
      p ≤ purple_balls → 
      b ≤ brown_balls → 
      g ≤ gray_balls → 
      s ≤ silver_balls → 
      g' ≤ golden_balls →
      o ≥ target_count ∨ 
      p ≥ target_count ∨ 
      b ≥ target_count ∨ 
      g ≥ target_count ∨ 
      s ≥ target_count ∨ 
      g' ≥ target_count) ∧
    n = 86 :=
by sorry

end min_balls_for_target_color_l785_78505


namespace roots_exist_when_q_positive_no_integer_roots_when_q_negative_l785_78545

-- Define the quadratic equations
def equation1 (p q x : ℤ) : Prop := x^2 - p*x + q = 0
def equation2 (p q x : ℤ) : Prop := x^2 - (p+1)*x + q = 0

-- Theorem for q > 0
theorem roots_exist_when_q_positive (q : ℤ) (hq : q > 0) :
  ∃ (p x1 x2 x3 x4 : ℤ), 
    equation1 p q x1 ∧ equation1 p q x2 ∧
    equation2 p q x3 ∧ equation2 p q x4 :=
sorry

-- Theorem for q < 0
theorem no_integer_roots_when_q_negative (q : ℤ) (hq : q < 0) :
  ¬∃ (p x1 x2 x3 x4 : ℤ), 
    equation1 p q x1 ∧ equation1 p q x2 ∧
    equation2 p q x3 ∧ equation2 p q x4 :=
sorry

end roots_exist_when_q_positive_no_integer_roots_when_q_negative_l785_78545


namespace opposite_of_neg_three_l785_78513

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem stating that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 := by
  sorry

end opposite_of_neg_three_l785_78513


namespace bicycle_cost_price_l785_78571

theorem bicycle_cost_price (final_price : ℝ) (profit_percentage : ℝ) : 
  final_price = 225 →
  profit_percentage = 25 →
  ∃ (original_cost : ℝ), 
    original_cost * (1 + profit_percentage / 100)^2 = final_price ∧
    original_cost = 144 := by
  sorry

end bicycle_cost_price_l785_78571


namespace max_product_constrained_l785_78589

theorem max_product_constrained (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_constraint : 3 * x + 8 * y = 72) : 
  x * y ≤ 54 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 * x₀ + 8 * y₀ = 72 ∧ x₀ * y₀ = 54 :=
sorry

end max_product_constrained_l785_78589


namespace binary_sum_equals_638_l785_78576

def binary_to_decimal (b : ℕ) : ℕ := 2^b - 1

theorem binary_sum_equals_638 :
  (binary_to_decimal 9) + (binary_to_decimal 7) = 638 := by
  sorry

end binary_sum_equals_638_l785_78576


namespace polynomial_simplification_l785_78514

theorem polynomial_simplification (x : ℝ) :
  3 * x^3 + 4 * x^2 + 2 * x + 5 - (2 * x^3 - 5 * x^2 + x - 3) + (x^3 - 2 * x^2 - 4 * x + 6) =
  2 * x^3 + 7 * x^2 - 3 * x + 14 := by
  sorry

end polynomial_simplification_l785_78514


namespace shirt_tie_belt_combinations_l785_78549

/-- Given a number of shirts, ties, and belts, calculates the total number of
    shirt-and-tie or shirt-and-belt combinations -/
def total_combinations (shirts : ℕ) (ties : ℕ) (belts : ℕ) : ℕ :=
  shirts * ties + shirts * belts

/-- Theorem stating that with 7 shirts, 6 ties, and 4 belts, 
    the total number of combinations is 70 -/
theorem shirt_tie_belt_combinations :
  total_combinations 7 6 4 = 70 := by
  sorry

end shirt_tie_belt_combinations_l785_78549


namespace stream_speed_is_one_l785_78575

/-- Represents the speed of a boat in still water and the speed of a stream. -/
structure BoatProblem where
  boat_speed : ℝ
  stream_speed : ℝ

/-- Given the conditions of the problem, proves that the stream speed is 1 km/h. -/
theorem stream_speed_is_one
  (bp : BoatProblem)
  (h1 : bp.boat_speed + bp.stream_speed = 100 / 10)
  (h2 : bp.boat_speed - bp.stream_speed = 200 / 25) :
  bp.stream_speed = 1 := by
  sorry

end stream_speed_is_one_l785_78575


namespace smallest_a_divisible_by_65_l785_78504

theorem smallest_a_divisible_by_65 :
  ∃ (a : ℕ), a > 0 ∧ 
  (∀ (n : ℤ), 65 ∣ (5 * n^13 + 13 * n^5 + 9 * a * n)) ∧
  (∀ (b : ℕ), b > 0 → b < a → 
    ∃ (m : ℤ), ¬(65 ∣ (5 * m^13 + 13 * m^5 + 9 * b * m))) ∧
  a = 63 :=
sorry

end smallest_a_divisible_by_65_l785_78504


namespace right_triangle_arithmetic_progression_inradius_l785_78580

theorem right_triangle_arithmetic_progression_inradius (a b c d : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a < b ∧ b < c →  -- Ordered side lengths
  a^2 + b^2 = c^2 →  -- Right triangle (Pythagorean theorem)
  b = a + d ∧ c = a + 2*d →  -- Arithmetic progression
  d = (a*b*c) / (a + b + c)  -- d equals inradius
  := by sorry

end right_triangle_arithmetic_progression_inradius_l785_78580


namespace carrot_stick_calories_prove_carrot_stick_calories_l785_78532

theorem carrot_stick_calories : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_calories burger_calories cookie_count cookie_calories carrot_stick_count carrot_stick_calories =>
    (total_calories = burger_calories + cookie_count * cookie_calories + carrot_stick_count * carrot_stick_calories) →
    (total_calories = 750) →
    (burger_calories = 400) →
    (cookie_count = 5) →
    (cookie_calories = 50) →
    (carrot_stick_count = 5) →
    (carrot_stick_calories = 20)

theorem prove_carrot_stick_calories : carrot_stick_calories 750 400 5 50 5 20 :=
by sorry

end carrot_stick_calories_prove_carrot_stick_calories_l785_78532


namespace semicircle_area_l785_78538

theorem semicircle_area (d : ℝ) (h : d = 11) : 
  (1/2) * π * (d/2)^2 = (121/8) * π := by sorry

end semicircle_area_l785_78538


namespace sum_of_roots_equal_one_l785_78520

theorem sum_of_roots_equal_one : 
  ∃ (x₁ x₂ : ℝ), (x₁ + 2) * (x₁ - 3) = 16 ∧ 
                 (x₂ + 2) * (x₂ - 3) = 16 ∧ 
                 x₁ + x₂ = 1 := by
  sorry

end sum_of_roots_equal_one_l785_78520


namespace remaining_bottle_caps_l785_78562

-- Define the initial number of bottle caps
def initial_caps : ℕ := 34

-- Define the number of bottle caps eaten
def eaten_caps : ℕ := 8

-- Theorem to prove
theorem remaining_bottle_caps : initial_caps - eaten_caps = 26 := by
  sorry

end remaining_bottle_caps_l785_78562


namespace systematic_sampling_theorem_l785_78568

/-- Systematic sampling function -/
def systematicSample (totalEmployees : ℕ) (sampleSize : ℕ) (startingNumber : ℕ) (sampleIndex : ℕ) : ℕ :=
  startingNumber + (sampleIndex - 1) * (totalEmployees / sampleSize)

/-- Theorem: If the 5th sample is 23 in a systematic sampling of 40 from 200, then the 8th sample is 38 -/
theorem systematic_sampling_theorem (totalEmployees sampleSize startingNumber : ℕ) 
    (h1 : totalEmployees = 200)
    (h2 : sampleSize = 40)
    (h3 : systematicSample totalEmployees sampleSize startingNumber 5 = 23) :
  systematicSample totalEmployees sampleSize startingNumber 8 = 38 := by
  sorry

#eval systematicSample 200 40 3 5  -- Should output 23
#eval systematicSample 200 40 3 8  -- Should output 38

end systematic_sampling_theorem_l785_78568


namespace ratio_equality_l785_78594

theorem ratio_equality (x : ℝ) : (0.60 / x = 6 / 2) → x = 0.2 := by
  sorry

end ratio_equality_l785_78594


namespace i_pow_2006_l785_78554

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the properties of i
axiom i_pow_1 : i^1 = i
axiom i_pow_2 : i^2 = -1
axiom i_pow_3 : i^3 = -i
axiom i_pow_4 : i^4 = 1
axiom i_pow_5 : i^5 = i

-- Theorem to prove
theorem i_pow_2006 : i^2006 = -1 := by
  sorry

end i_pow_2006_l785_78554


namespace sin_40_tan_10_minus_sqrt_3_l785_78560

theorem sin_40_tan_10_minus_sqrt_3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end sin_40_tan_10_minus_sqrt_3_l785_78560


namespace line_point_distance_l785_78584

/-- Given five points O, A, B, C, D on a line, with Q and P also on the line,
    prove that OP = 2q under the given conditions. -/
theorem line_point_distance (a b c d q : ℝ) : 
  ∀ (x : ℝ), 
  (0 < a) → (a < b) → (b < c) → (c < d) →  -- Points are in order
  (0 < q) → (q < d) →  -- Q is on the line
  (b ≤ x) → (x ≤ c) →  -- P is between B and C
  ((a - x) / (x - d) = (b - x) / (x - c)) →  -- AP : PD = BP : PC
  (x = 2 * q) →  -- P is twice as far from O as Q is
  x = 2 * q := by
  sorry

end line_point_distance_l785_78584


namespace stratified_sampling_arts_students_l785_78588

theorem stratified_sampling_arts_students 
  (total_students : ℕ) 
  (arts_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 1000)
  (h2 : arts_students = 200)
  (h3 : sample_size = 100) :
  (arts_students : ℚ) / total_students * sample_size = 20 := by
sorry

end stratified_sampling_arts_students_l785_78588


namespace fraction_equality_l785_78591

theorem fraction_equality (a b : ℝ) (h : (a - b) / a = 2 / 3) : b / a = 1 / 3 := by
  sorry

end fraction_equality_l785_78591


namespace sum_of_xy_l785_78583

theorem sum_of_xy (x y : ℕ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x < 30) 
  (h4 : y < 30) 
  (h5 : x + y + x * y = 119) : 
  x + y = 20 := by
sorry

end sum_of_xy_l785_78583


namespace camp_cedar_counselors_l785_78525

/-- The number of counselors needed at Camp Cedar -/
def counselors_needed (num_boys : ℕ) (girl_to_boy_ratio : ℕ) (children_per_counselor : ℕ) : ℕ :=
  let num_girls := num_boys * girl_to_boy_ratio
  let total_children := num_boys + num_girls
  total_children / children_per_counselor

/-- Theorem stating the number of counselors needed at Camp Cedar -/
theorem camp_cedar_counselors :
  counselors_needed 40 3 8 = 20 := by
  sorry

#eval counselors_needed 40 3 8

end camp_cedar_counselors_l785_78525


namespace max_length_valid_progression_l785_78507

/-- An arithmetic progression of natural numbers. -/
structure ArithmeticProgression :=
  (first : ℕ)
  (diff : ℕ)
  (len : ℕ)

/-- Check if a natural number contains the digit 9. -/
def containsNine (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 9

/-- An arithmetic progression satisfying the given conditions. -/
def ValidProgression (ap : ArithmeticProgression) : Prop :=
  ap.diff ≠ 0 ∧
  ∀ i : ℕ, i < ap.len → ¬containsNine (ap.first + i * ap.diff)

/-- The main theorem: The maximum length of a valid progression is 72. -/
theorem max_length_valid_progression :
  ∀ ap : ArithmeticProgression, ValidProgression ap → ap.len ≤ 72 :=
sorry

end max_length_valid_progression_l785_78507


namespace trigonometric_equality_l785_78511

theorem trigonometric_equality : 
  4 * Real.sin (30 * π / 180) - Real.sqrt 2 * Real.cos (45 * π / 180) - 
  Real.sqrt 3 * Real.tan (30 * π / 180) + 2 * Real.sin (60 * π / 180) = Real.sqrt 3 := by
  sorry

end trigonometric_equality_l785_78511


namespace garden_area_l785_78561

theorem garden_area (total_posts : ℕ) (post_spacing : ℕ) 
  (h1 : total_posts = 20)
  (h2 : post_spacing = 4)
  (h3 : ∃ (short_posts long_posts : ℕ), 
    short_posts > 1 ∧ 
    long_posts > 1 ∧ 
    short_posts + long_posts = total_posts / 2 + 2 ∧ 
    long_posts = 2 * short_posts) :
  ∃ (width length : ℕ), 
    width * length = 336 ∧ 
    width = post_spacing * (short_posts - 1) ∧ 
    length = post_spacing * (long_posts - 1) :=
by sorry

#check garden_area

end garden_area_l785_78561


namespace production_difference_formula_l785_78524

/-- Represents the widget production scenario for David --/
structure WidgetProduction where
  /-- Widgets produced per hour on Monday --/
  w : ℕ
  /-- Hours worked on Monday --/
  t : ℕ
  /-- Relationship between w and t --/
  w_eq_3t : w = 3 * t

/-- Calculates the difference in widget production between Monday and Tuesday --/
def productionDifference (p : WidgetProduction) : ℕ :=
  let monday_production := p.w * p.t
  let tuesday_production := (p.w + 6) * (p.t - 3)
  monday_production - tuesday_production

/-- Theorem stating the difference in widget production --/
theorem production_difference_formula (p : WidgetProduction) :
  productionDifference p = 3 * p.t + 18 := by
  sorry

#check production_difference_formula

end production_difference_formula_l785_78524


namespace green_peaches_count_l785_78555

/-- Given a basket of peaches, prove the number of green peaches. -/
theorem green_peaches_count (red : ℕ) (green : ℕ) : 
  red = 7 → green = red + 1 → green = 8 := by
  sorry

end green_peaches_count_l785_78555


namespace fill_675_cans_l785_78566

/-- A machine that fills paint cans at a specific rate -/
structure PaintMachine where
  cans_per_batch : ℕ
  minutes_per_batch : ℕ

/-- Calculate the time needed to fill a given number of cans -/
def time_to_fill (machine : PaintMachine) (total_cans : ℕ) : ℕ := 
  (total_cans * machine.minutes_per_batch + machine.cans_per_batch - 1) / machine.cans_per_batch

/-- Theorem: The given machine takes 36 minutes to fill 675 cans -/
theorem fill_675_cans (machine : PaintMachine) 
  (h1 : machine.cans_per_batch = 150) 
  (h2 : machine.minutes_per_batch = 8) : 
  time_to_fill machine 675 = 36 := by sorry

end fill_675_cans_l785_78566
