import Mathlib

namespace min_value_a_l2575_257526

theorem min_value_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ (x y : ℝ), x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 9) : 
  a ≥ 4 ∧ ∀ (ε : ℝ), ε > 0 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y) * (1/(x) + (4 - ε)/y) < 9 :=
sorry

end min_value_a_l2575_257526


namespace euler_family_mean_age_l2575_257522

def euler_family_ages : List ℕ := [8, 8, 12, 12, 10, 14]

theorem euler_family_mean_age :
  (euler_family_ages.sum : ℚ) / euler_family_ages.length = 32 / 3 := by
  sorry

end euler_family_mean_age_l2575_257522


namespace lcm_is_perfect_square_l2575_257563

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a*b) % (a*b*(a - b)) = 0) :
  Nat.lcm a b = (Nat.gcd a b)^2 := by
  sorry

end lcm_is_perfect_square_l2575_257563


namespace reciprocal_of_sum_diff_fractions_l2575_257583

theorem reciprocal_of_sum_diff_fractions : 
  (1 / (1/3 + 1/4 - 1/12) : ℚ) = 2 := by
  sorry

end reciprocal_of_sum_diff_fractions_l2575_257583


namespace other_x_axis_point_on_circle_l2575_257575

def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

theorem other_x_axis_point_on_circle :
  let C : Set (ℝ × ℝ) := Circle (0, 0) 16
  (16, 0) ∈ C →
  (-16, 0) ∈ C ∧ (-16, 0).2 = 0 := by
  sorry

end other_x_axis_point_on_circle_l2575_257575


namespace vector_difference_magnitude_l2575_257500

theorem vector_difference_magnitude : ∃ x : ℝ,
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![x, 4]
  (∃ k : ℝ, a = k • b) →
  Real.sqrt ((a 0 - b 0)^2 + (a 1 - b 1)^2) = 3 * Real.sqrt 5 := by
  sorry

end vector_difference_magnitude_l2575_257500


namespace absolute_value_square_inequality_l2575_257551

theorem absolute_value_square_inequality {a b : ℝ} (h : |a| < b) : a^2 < b^2 := by
  sorry

end absolute_value_square_inequality_l2575_257551


namespace reinforcement_size_l2575_257554

/-- Calculates the size of reinforcement given initial garrison size, initial provisions duration,
    time passed before reinforcement, and remaining provisions duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                            (time_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_duration
  let provisions_left := total_provisions - (initial_garrison * time_passed)
  let reinforcement := (provisions_left / remaining_duration) - initial_garrison
  reinforcement

/-- Theorem stating that given the specific conditions of the problem,
    the calculated reinforcement size is 3000. -/
theorem reinforcement_size :
  calculate_reinforcement 2000 65 15 20 = 3000 := by
  sorry

end reinforcement_size_l2575_257554


namespace negation_statement_1_negation_statement_2_negation_statement_3_l2575_257572

-- Define the set of prime numbers
def isPrime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)
def P : Set ℕ := {p : ℕ | isPrime p}

-- Statement 1
theorem negation_statement_1 :
  (∀ n : ℕ, ∃ p ∈ P, n ≤ p) ↔ (∃ n : ℕ, ∀ p ∈ P, p ≤ n) :=
sorry

-- Statement 2
theorem negation_statement_2 :
  (∀ n : ℤ, ∃! p : ℤ, n + p = 0) ↔ (∃ n : ℤ, ∀ p : ℤ, n + p ≠ 0) :=
sorry

-- Statement 3
theorem negation_statement_3 :
  (∃ y : ℝ, ∀ x : ℝ, ∃ c : ℝ, x * y = c) ↔
  (∀ y : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * y ≠ x₂ * y) :=
sorry

end negation_statement_1_negation_statement_2_negation_statement_3_l2575_257572


namespace factorization_proof_l2575_257599

theorem factorization_proof (x : ℝ) : 
  (x^2 - 1) * (x^4 + x^2 + 1) - (x^3 + 1)^2 = -2 * (x + 1) * (x^2 - x + 1) := by
  sorry

end factorization_proof_l2575_257599


namespace smallest_nonprime_no_small_factors_is_529_l2575_257545

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_no_prime_factors_less_than (n k : ℕ) : Prop :=
  ∀ p, is_prime p → p < k → ¬(n % p = 0)

def smallest_nonprime_no_small_factors : ℕ → Prop
| n => ¬(is_prime n) ∧ 
       n > 1 ∧ 
       has_no_prime_factors_less_than n 20 ∧
       ∀ m, 1 < m → m < n → ¬(¬(is_prime m) ∧ has_no_prime_factors_less_than m 20)

theorem smallest_nonprime_no_small_factors_is_529 :
  ∃ n, smallest_nonprime_no_small_factors n ∧ n = 529 :=
sorry

end smallest_nonprime_no_small_factors_is_529_l2575_257545


namespace steves_gum_pieces_l2575_257598

/-- Given Todd's initial and final number of gum pieces, prove that the number of gum pieces
    Steve gave Todd is equal to the difference between the final and initial numbers. -/
theorem steves_gum_pieces (todd_initial todd_final steve_gave : ℕ) 
    (h1 : todd_initial = 38)
    (h2 : todd_final = 54)
    (h3 : todd_final = todd_initial + steve_gave) :
  steve_gave = todd_final - todd_initial := by
  sorry

end steves_gum_pieces_l2575_257598


namespace opposites_sum_zero_l2575_257597

theorem opposites_sum_zero (a b : ℚ) : a + b = 0 → a = -b := by
  sorry

end opposites_sum_zero_l2575_257597


namespace power_calculation_l2575_257564

theorem power_calculation : 16^12 * 8^8 / 2^60 = 4096 := by
  sorry

end power_calculation_l2575_257564


namespace seashell_sum_total_seashells_l2575_257509

theorem seashell_sum : Int → Int → Int → Int
  | sam, joan, alex => sam + joan + alex

theorem total_seashells (sam joan alex : Int) 
  (h1 : sam = 35) (h2 : joan = 18) (h3 : alex = 27) : 
  seashell_sum sam joan alex = 80 := by
  sorry

end seashell_sum_total_seashells_l2575_257509


namespace quadratic_minimum_quadratic_minimum_achievable_l2575_257532

theorem quadratic_minimum (x : ℝ) : 7 * x^2 - 28 * x + 1425 ≥ 1397 :=
sorry

theorem quadratic_minimum_achievable : ∃ x : ℝ, 7 * x^2 - 28 * x + 1425 = 1397 :=
sorry

end quadratic_minimum_quadratic_minimum_achievable_l2575_257532


namespace outbound_speed_calculation_l2575_257537

-- Define the problem parameters
def distance : ℝ := 19.999999999999996
def return_speed : ℝ := 4
def total_time : ℝ := 5.8

-- Define the theorem
theorem outbound_speed_calculation :
  ∃ (v : ℝ), v > 0 ∧ (distance / v + distance / return_speed = total_time) → v = 25 := by
  sorry

end outbound_speed_calculation_l2575_257537


namespace sqrt_equation_solution_l2575_257543

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 2) = 8 → x = 66 := by
  sorry

end sqrt_equation_solution_l2575_257543


namespace rows_of_nine_l2575_257504

/-- Given 74 people seated in rows of either 7 or 9 seats, with all seats occupied,
    there are exactly 2 rows seating 9 people. -/
theorem rows_of_nine (total_people : ℕ) (rows_of_seven : ℕ) (rows_of_nine : ℕ) : 
  total_people = 74 →
  total_people = 7 * rows_of_seven + 9 * rows_of_nine →
  rows_of_nine = 2 := by
  sorry

end rows_of_nine_l2575_257504


namespace inverse_proposition_false_l2575_257580

theorem inverse_proposition_false : 
  ¬ (∀ a b c : ℝ, a > b → a * c^2 > b * c^2) := by
sorry

end inverse_proposition_false_l2575_257580


namespace jack_afternoon_emails_l2575_257584

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 4

/-- The total number of emails Jack received in the day -/
def total_emails : ℕ := 5

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := total_emails - morning_emails

theorem jack_afternoon_emails :
  afternoon_emails = 1 :=
sorry

end jack_afternoon_emails_l2575_257584


namespace percentage_difference_l2575_257529

theorem percentage_difference (A B C y : ℝ) : 
  A = B + C → 
  B > C → 
  C > 0 → 
  B = C * (1 + y / 100) → 
  y = 100 * ((B - C) / C) :=
by sorry

end percentage_difference_l2575_257529


namespace corner_with_same_color_l2575_257541

/-- Definition of a "corner" figure -/
def Corner (square : Fin 2017 → Fin 2017 → Fin 120) : Prop :=
  ∃ (i j : Fin 2017) (dir : Bool),
    let horizontal := if dir then (fun k => square i (j + k)) else (fun k => square (i + k) j)
    let vertical := if dir then (fun k => square (i + k) j) else (fun k => square i (j + k))
    (∀ k : Fin 10, horizontal k ∈ Set.range horizontal) ∧
    (∀ k : Fin 10, vertical k ∈ Set.range vertical) ∧
    (square i j ∈ Set.range horizontal ∪ Set.range vertical)

/-- The main theorem -/
theorem corner_with_same_color (square : Fin 2017 → Fin 2017 → Fin 120) :
  ∃ (corner : Corner square), 
    ∃ (c1 c2 : Fin 2017 × Fin 2017), c1 ≠ c2 ∧ 
      square c1.1 c1.2 = square c2.1 c2.2 :=
sorry

end corner_with_same_color_l2575_257541


namespace f_monotonicity_and_inequality_l2575_257538

noncomputable def f (x : ℝ) := x / Real.exp x

theorem f_monotonicity_and_inequality :
  (∀ x y, x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f y < f x) ∧
  (∀ x, x > 0 → Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x)) :=
sorry

end f_monotonicity_and_inequality_l2575_257538


namespace cube_ratio_equals_27_l2575_257510

theorem cube_ratio_equals_27 : (81000 : ℚ)^3 / (27000 : ℚ)^3 = 27 := by sorry

end cube_ratio_equals_27_l2575_257510


namespace angle_bisector_product_theorem_l2575_257506

/-- Given a triangle with sides a, b, c, internal angle bisectors fa, fb, fc, and area T,
    this theorem states that the product of the angle bisectors divided by the product of the sides
    is equal to four times the area multiplied by the sum of the sides,
    divided by the product of the pairwise sums of the sides. -/
theorem angle_bisector_product_theorem
  (a b c fa fb fc T : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b)
  (h_bisectors : fa > 0 ∧ fb > 0 ∧ fc > 0)
  (h_area : T > 0) :
  (fa * fb * fc) / (a * b * c) = 4 * T * (a + b + c) / ((a + b) * (b + c) * (a + c)) :=
sorry

end angle_bisector_product_theorem_l2575_257506


namespace f_geq_a_iff_a_in_range_l2575_257513

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

theorem f_geq_a_iff_a_in_range (a : ℝ) :
  (∀ x ≥ (1/2 : ℝ), f a x ≥ a) ↔ a ≤ 1 :=
by sorry

end f_geq_a_iff_a_in_range_l2575_257513


namespace triangle_inequality_last_three_terms_l2575_257524

/-- An arithmetic sequence with positive terms and positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = a n + d

/-- Triangle inequality for the last three terms of a four-term arithmetic sequence -/
theorem triangle_inequality_last_three_terms
  (a : ℕ → ℝ) (d : ℝ) (h : ArithmeticSequence a d) :
  a 2 + a 3 > a 4 ∧ a 2 + a 4 > a 3 ∧ a 3 + a 4 > a 2 :=
sorry

end triangle_inequality_last_three_terms_l2575_257524


namespace chicken_wings_distribution_l2575_257560

theorem chicken_wings_distribution (num_friends : ℕ) (pre_cooked : ℕ) (additional_cooked : ℕ) :
  num_friends = 4 →
  pre_cooked = 9 →
  additional_cooked = 7 →
  (pre_cooked + additional_cooked) / num_friends = 4 :=
by sorry

end chicken_wings_distribution_l2575_257560


namespace x_value_proof_l2575_257556

theorem x_value_proof (x : ℚ) (h : 2/3 - 1/4 = 4/x) : x = 48/5 := by
  sorry

end x_value_proof_l2575_257556


namespace currency_notes_existence_l2575_257577

theorem currency_notes_existence : 
  ∃ (x y z : ℕ), x + 5*y + 10*z = 480 ∧ x + y + z = 90 := by
  sorry

end currency_notes_existence_l2575_257577


namespace garlic_cloves_left_is_600_l2575_257540

/-- The number of garlic cloves Maria has left after using some for a feast -/
def garlic_cloves_left : ℕ :=
  let kitchen_initial := 750
  let pantry_initial := 450
  let basement_initial := 300
  let kitchen_used := 500
  let pantry_used := 230
  let basement_used := 170
  (kitchen_initial - kitchen_used) + (pantry_initial - pantry_used) + (basement_initial - basement_used)

theorem garlic_cloves_left_is_600 : garlic_cloves_left = 600 := by
  sorry

end garlic_cloves_left_is_600_l2575_257540


namespace equation_solution_l2575_257591

theorem equation_solution (y : ℝ) : 
  (|y - 4|^2 + 3*y = 14) ↔ (y = (5 + Real.sqrt 17)/2 ∨ y = (5 - Real.sqrt 17)/2) := by
sorry

end equation_solution_l2575_257591


namespace system_elimination_l2575_257508

theorem system_elimination (x y : ℝ) : 
  (x + y = 5 ∧ x - y = 2) → 
  (∃ k : ℝ, (x + y) + (x - y) = k ∧ y ≠ k / 2) ∧ 
  (∃ m : ℝ, (x + y) - (x - y) = m ∧ x ≠ m / 2) :=
by sorry

end system_elimination_l2575_257508


namespace billiard_path_to_top_left_l2575_257520

/-- Represents a point in a 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Represents a rectangular lattice -/
structure RectangularLattice where
  width : ℕ
  height : ℕ

def billiardTable : RectangularLattice := { width := 1965, height := 26 }

/-- Checks if a point is on the top edge of the lattice -/
def isTopEdge (l : RectangularLattice) (p : LatticePoint) : Prop :=
  p.x = 0 ∧ p.y = l.height

/-- Represents a line with slope 1 starting from (0, 0) -/
def slopeLine (x : ℤ) : LatticePoint :=
  { x := x, y := x }

theorem billiard_path_to_top_left :
  ∃ (n : ℕ), isTopEdge billiardTable (slopeLine (n * billiardTable.width)) := by
  sorry

end billiard_path_to_top_left_l2575_257520


namespace solution_set_of_inequality_range_of_a_l2575_257555

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 5|

-- Theorem for part I
theorem solution_set_of_inequality (x : ℝ) :
  (f x ≤ x + 10) ↔ (x ∈ Set.Icc (-2) 14) :=
sorry

-- Theorem for part II
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a - (x - 2)^2) ↔ (a ∈ Set.Iic 6) :=
sorry

end solution_set_of_inequality_range_of_a_l2575_257555


namespace purely_imaginary_complex_number_l2575_257530

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) * (1 - 2 * Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → a = -1/2 := by
  sorry

end purely_imaginary_complex_number_l2575_257530


namespace erased_number_proof_l2575_257585

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  x ≤ n →
  (n * (n + 1) / 2 - x) / (n - 1) = 866 / 19 →
  x = 326 :=
sorry

end erased_number_proof_l2575_257585


namespace least_odd_integer_given_mean_l2575_257590

theorem least_odd_integer_given_mean (integers : List Int) : 
  integers.length = 10 ∧ 
  (∀ i ∈ integers, i % 2 = 1) ∧ 
  (∀ i j, i ∈ integers → j ∈ integers → i ≠ j → |i - j| % 2 = 0) ∧
  (integers.sum / integers.length : ℚ) = 154 →
  integers.minimum? = some 144 := by
sorry

end least_odd_integer_given_mean_l2575_257590


namespace bike_trip_distance_difference_l2575_257581

-- Define the parameters of the problem
def total_time : ℝ := 6
def alberto_speed : ℝ := 12
def bjorn_speed : ℝ := 10
def bjorn_rest_time : ℝ := 1

-- Define the distances traveled by Alberto and Bjorn
def alberto_distance : ℝ := alberto_speed * total_time
def bjorn_distance : ℝ := bjorn_speed * (total_time - bjorn_rest_time)

-- State the theorem
theorem bike_trip_distance_difference :
  alberto_distance - bjorn_distance = 22 := by sorry

end bike_trip_distance_difference_l2575_257581


namespace intersecting_chords_theorem_l2575_257550

-- Define a circle
variable (circle : Type) [MetricSpace circle]

-- Define the chords and intersection point
variable (chord1 chord2 : Set circle)
variable (P : circle)

-- Define the segments of the first chord
variable (PA PB : ℝ)

-- Define the ratio of the segments of the second chord
variable (r : ℚ)

-- State the theorem
theorem intersecting_chords_theorem 
  (h1 : P ∈ chord1 ∩ chord2)
  (h2 : PA = 12)
  (h3 : PB = 18)
  (h4 : r = 3 / 8)
  : ∃ (PC PD : ℝ), PC + PD = 33 ∧ PC / PD = r := by
  sorry

end intersecting_chords_theorem_l2575_257550


namespace no_solution_set_characterization_l2575_257516

/-- The quadratic function f(x) = x² - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The set of values k for which f(x) = k has no real solutions -/
def no_solution_set : Set ℝ := {k | ∀ x, f x ≠ k}

/-- Theorem stating that the no_solution_set is equivalent to {k | k < 1} -/
theorem no_solution_set_characterization :
  no_solution_set = {k | k < 1} := by sorry

end no_solution_set_characterization_l2575_257516


namespace binomial_coefficient_recurrence_l2575_257505

theorem binomial_coefficient_recurrence (n r : ℕ) (h1 : n > 0) (h2 : r > 0) (h3 : n > r) :
  Nat.choose n r = Nat.choose (n - 1) r + Nat.choose (n - 1) (r - 1) := by
  sorry

end binomial_coefficient_recurrence_l2575_257505


namespace compound_interest_rate_proof_l2575_257573

/-- Proves that the given conditions result in the specified annual interest rate -/
theorem compound_interest_rate_proof 
  (principal : ℝ) 
  (time : ℝ) 
  (compounding_frequency : ℝ) 
  (compound_interest : ℝ) 
  (h1 : principal = 50000)
  (h2 : time = 2)
  (h3 : compounding_frequency = 2)
  (h4 : compound_interest = 4121.608)
  : ∃ (rate : ℝ), 
    (abs (rate - 0.0398) < 0.0001) ∧ 
    (principal * (1 + rate / compounding_frequency) ^ (compounding_frequency * time) = 
     principal + compound_interest) :=
by sorry


end compound_interest_rate_proof_l2575_257573


namespace horner_v₂_for_specific_polynomial_v₂_value_at_10_l2575_257544

/-- Horner's Rule for a polynomial of degree 4 -/
def horner_rule (a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((a₄ * x + a₃) * x + a₂) * x + a₁ * x + a₀

/-- The second intermediate value in Horner's Rule calculation -/
def v₂ (a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  (a₄ * x + a₃) * x + a₂

theorem horner_v₂_for_specific_polynomial (x : ℝ) :
  v₂ 3 1 0 2 4 x = 3 * x * x + x := by sorry

theorem v₂_value_at_10 :
  v₂ 3 1 0 2 4 10 = 310 := by sorry

end horner_v₂_for_specific_polynomial_v₂_value_at_10_l2575_257544


namespace unique_congruence_l2575_257528

theorem unique_congruence (n : ℤ) : 
  12 ≤ n ∧ n ≤ 18 ∧ n ≡ 9001 [ZMOD 7] → n = 13 := by
sorry

end unique_congruence_l2575_257528


namespace green_toads_count_l2575_257535

/-- The number of green toads per acre -/
def green_toads_per_acre : ℕ := 8

/-- The ratio of green toads to brown toads -/
def green_to_brown_ratio : ℚ := 1 / 25

/-- The fraction of brown toads that are spotted -/
def spotted_brown_fraction : ℚ := 1 / 4

/-- The number of spotted brown toads per acre -/
def spotted_brown_per_acre : ℕ := 50

theorem green_toads_count :
  green_toads_per_acre = 8 :=
sorry

end green_toads_count_l2575_257535


namespace line_points_k_value_l2575_257557

/-- Given a line with equation x - 5/2y + 1 = 0 and two points (m, n) and (m + 1/2, n + 1/k) on this line,
    prove that k = 3/5 -/
theorem line_points_k_value (m n k : ℝ) :
  (m - 5/2 * n + 1 = 0) →
  (m + 1/2 - 5/2 * (n + 1/k) + 1 = 0) →
  k = 3/5 := by
  sorry


end line_points_k_value_l2575_257557


namespace system_of_equations_solution_l2575_257533

theorem system_of_equations_solution :
  (∀ p q : ℚ, p + q = 4 ∧ 2 * p - q = 5 → p = 3 ∧ q = 1) ∧
  (∀ v t : ℚ, 2 * v + t = 3 ∧ 3 * v - 2 * t = 3 → v = 9 / 7 ∧ t = 3 / 7) := by
  sorry

end system_of_equations_solution_l2575_257533


namespace tan_alpha_fourth_quadrant_l2575_257547

theorem tan_alpha_fourth_quadrant (α : Real) : 
  (π / 2 < α ∧ α < 2 * π) →  -- α is in the fourth quadrant
  (Real.cos (π / 2 + α) = 4 / 5) → 
  Real.tan α = -4 / 3 := by
sorry

end tan_alpha_fourth_quadrant_l2575_257547


namespace root_sum_squared_plus_triple_plus_other_root_l2575_257503

theorem root_sum_squared_plus_triple_plus_other_root (α β : ℝ) : 
  α^2 + 2*α - 2024 = 0 → β^2 + 2*β - 2024 = 0 → α^2 + 3*α + β = 2022 :=
by sorry

end root_sum_squared_plus_triple_plus_other_root_l2575_257503


namespace window_width_is_four_l2575_257552

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangular surface -/
def rectangleArea (d : Dimensions) : ℝ := d.length * d.width

/-- Calculates the perimeter of a rectangular surface -/
def rectanglePerimeter (d : Dimensions) : ℝ := 2 * (d.length + d.width)

/-- Represents the properties of the room and whitewashing job -/
structure RoomProperties where
  roomDimensions : Dimensions
  doorDimensions : Dimensions
  windowHeight : ℝ
  numWindows : ℕ
  costPerSquareFoot : ℝ
  totalCost : ℝ

/-- Theorem: The width of each window is 4 feet -/
theorem window_width_is_four (props : RoomProperties) 
  (h1 : props.roomDimensions = ⟨25, 15, 12⟩)
  (h2 : props.doorDimensions = ⟨6, 3, 0⟩)
  (h3 : props.windowHeight = 3)
  (h4 : props.numWindows = 3)
  (h5 : props.costPerSquareFoot = 8)
  (h6 : props.totalCost = 7248) : 
  ∃ w : ℝ, w = 4 ∧ 
    props.totalCost = props.costPerSquareFoot * 
      (rectanglePerimeter props.roomDimensions * props.roomDimensions.height - 
       rectangleArea props.doorDimensions - 
       props.numWindows * (w * props.windowHeight)) := by
  sorry

end window_width_is_four_l2575_257552


namespace haleigh_leggings_count_l2575_257587

/-- The number of dogs Haleigh has -/
def num_dogs : ℕ := 4

/-- The number of cats Haleigh has -/
def num_cats : ℕ := 3

/-- The number of legs each animal (dog or cat) has -/
def legs_per_animal : ℕ := 4

/-- The number of legs covered by one pair of leggings -/
def legs_per_legging : ℕ := 2

/-- The total number of pairs of leggings needed for Haleigh's pets -/
def total_leggings : ℕ := 
  (num_dogs * legs_per_animal + num_cats * legs_per_animal) / legs_per_legging

theorem haleigh_leggings_count : total_leggings = 14 := by
  sorry

end haleigh_leggings_count_l2575_257587


namespace fraction_sum_integer_l2575_257588

theorem fraction_sum_integer (n : ℕ) (h1 : n > 0) 
  (h2 : ∃ k : ℤ, (1 : ℚ) / 2 + 1 / 3 + 1 / 5 + 1 / n = k) : n = 30 := by
  sorry

end fraction_sum_integer_l2575_257588


namespace gcd_problem_l2575_257570

def gcd_operation (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_problem :
  gcd_operation (gcd_operation (gcd_operation 20 16) (gcd_operation 18 24)) 1 = 2 := by
  sorry

end gcd_problem_l2575_257570


namespace line_points_property_l2575_257521

theorem line_points_property (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = -2 * x₁ + 3)
  (h2 : y₂ = -2 * x₂ + 3)
  (h3 : y₃ = -2 * x₃ + 3)
  (h4 : x₁ < x₂)
  (h5 : x₂ < x₃)
  (h6 : x₂ * x₃ < 0) :
  y₁ * y₂ > 0 := by
  sorry

end line_points_property_l2575_257521


namespace product_of_four_numbers_l2575_257527

theorem product_of_four_numbers (a b c d : ℝ) : 
  ((a + b + c + d) / 4 = 7.1) →
  (2.5 * a = b - 1.2) →
  (b - 1.2 = c + 4.8) →
  (c + 4.8 = 0.25 * d) →
  (a * b * c * d = 49.6) := by
sorry

end product_of_four_numbers_l2575_257527


namespace third_fraction_numerator_l2575_257546

/-- Given three fractions where the sum is 3.0035428163476343,
    the first fraction is 2007/2999, the second is 8001/5998,
    and the third has a denominator of 3999,
    prove that the numerator of the third fraction is 4002. -/
theorem third_fraction_numerator :
  let sum : ℚ := 3.0035428163476343
  let frac1 : ℚ := 2007 / 2999
  let frac2 : ℚ := 8001 / 5998
  let denom3 : ℕ := 3999
  ∃ (num3 : ℕ), (frac1 + frac2 + (num3 : ℚ) / denom3 = sum) ∧ num3 = 4002 := by
  sorry


end third_fraction_numerator_l2575_257546


namespace solution_set_inequality_l2575_257566

theorem solution_set_inequality (x : ℝ) : 
  x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 :=
by sorry

end solution_set_inequality_l2575_257566


namespace deck_width_l2575_257592

/-- Given a rectangular deck with the following properties:
  * length is 30 feet
  * total cost per square foot (including construction and sealant) is $4
  * total payment is $4800
  prove that the width of the deck is 40 feet -/
theorem deck_width (length : ℝ) (cost_per_sqft : ℝ) (total_cost : ℝ) :
  length = 30 →
  cost_per_sqft = 4 →
  total_cost = 4800 →
  (length * (total_cost / cost_per_sqft)) / length = 40 := by
  sorry

end deck_width_l2575_257592


namespace largest_angle_in_triangle_l2575_257562

theorem largest_angle_in_triangle (x y z : ℝ) : 
  x = 60 → y = 70 → x + y + z = 180 → 
  ∃ max_angle : ℝ, max_angle = 70 ∧ max_angle ≥ x ∧ max_angle ≥ y ∧ max_angle ≥ z :=
by sorry

end largest_angle_in_triangle_l2575_257562


namespace floor_of_5_7_l2575_257525

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by
  sorry

end floor_of_5_7_l2575_257525


namespace div_value_problem_l2575_257511

theorem div_value_problem (a b d : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / d = 2 / 5) : 
  d / a = 5 / 6 := by
  sorry

end div_value_problem_l2575_257511


namespace jumping_contest_l2575_257574

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump squirrel_jump : ℕ)
  (grasshopper_obstacle frog_obstacle mouse_obstacle squirrel_obstacle : ℕ)
  (h1 : grasshopper_jump = 19)
  (h2 : grasshopper_obstacle = 3)
  (h3 : frog_jump = grasshopper_jump + 10)
  (h4 : frog_obstacle = 0)
  (h5 : mouse_jump = frog_jump + 20)
  (h6 : mouse_obstacle = 5)
  (h7 : squirrel_jump = mouse_jump - 7)
  (h8 : squirrel_obstacle = 2) :
  (mouse_jump - mouse_obstacle) - (grasshopper_jump - grasshopper_obstacle) = 28 := by
  sorry

#check jumping_contest

end jumping_contest_l2575_257574


namespace rectangle_division_l2575_257515

theorem rectangle_division (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (area1 : a * b = 18) (area2 : a * c = 27) (area3 : b * d = 12) :
  c * d = 93 := by
  sorry

end rectangle_division_l2575_257515


namespace equation_solution_l2575_257558

theorem equation_solution :
  let f (x : ℝ) := x + 3 = 4 / (x - 2)
  ∀ x : ℝ, x ≠ 2 → (f x ↔ (x = (-1 + Real.sqrt 41) / 2 ∨ x = (-1 - Real.sqrt 41) / 2)) :=
by sorry

end equation_solution_l2575_257558


namespace z_magnitude_l2575_257501

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem z_magnitude : 
  ((z - 2) * i = 1 + i) → Complex.abs z = Real.sqrt 10 := by
  sorry

end z_magnitude_l2575_257501


namespace train_passing_jogger_time_l2575_257507

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h_jogger_speed : jogger_speed = 9 * (1000 / 3600))
  (h_train_speed : train_speed = 45 * (1000 / 3600))
  (h_train_length : train_length = 120)
  (h_initial_distance : initial_distance = 250) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 37 := by
sorry

end train_passing_jogger_time_l2575_257507


namespace find_k_l2575_257519

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 5 * x + 6
def g (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 - k * x + 1

-- State the theorem
theorem find_k : ∃ k : ℝ, f 5 - g k 5 = 30 ∧ k = -10 := by
  sorry

end find_k_l2575_257519


namespace percent_decrease_l2575_257549

theorem percent_decrease (original_price sale_price : ℝ) (h : original_price > 0) :
  let decrease := original_price - sale_price
  let percent_decrease := (decrease / original_price) * 100
  original_price = 100 ∧ sale_price = 75 → percent_decrease = 25 := by
  sorry

end percent_decrease_l2575_257549


namespace marbles_cost_calculation_l2575_257593

/-- The amount spent on marbles when the total spent on toys is known, along with the costs of a football and baseball. -/
def marbles_cost (total_spent football_cost baseball_cost : ℚ) : ℚ :=
  total_spent - (football_cost + baseball_cost)

/-- Theorem stating that the cost of marbles is the difference between the total spent and the sum of football and baseball costs. -/
theorem marbles_cost_calculation (total_spent football_cost baseball_cost : ℚ) 
  (h1 : total_spent = 20.52)
  (h2 : football_cost = 4.95)
  (h3 : baseball_cost = 6.52) : 
  marbles_cost total_spent football_cost baseball_cost = 9.05 := by
sorry

end marbles_cost_calculation_l2575_257593


namespace impossible_sum_and_reciprocal_sum_l2575_257542

theorem impossible_sum_and_reciprocal_sum (x y z : ℝ) :
  x + y + z = 0 ∧ 1/x + 1/y + 1/z = 0 →
  x^1988 + y^1988 + z^1988 = 1/x^1988 + 1/y^1988 + 1/z^1988 :=
by sorry

end impossible_sum_and_reciprocal_sum_l2575_257542


namespace sequence_ratio_values_l2575_257589

/-- Two sequences where one is arithmetic and the other is geometric -/
structure SequencePair :=
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arithmetic : (∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d) ∨ 
                  (∃ d : ℝ, ∀ n : ℕ, b (n + 1) - b n = d))
  (h_geometric : (∃ r : ℝ, ∀ n : ℕ, a (n + 1) / a n = r) ∨ 
                 (∃ r : ℝ, ∀ n : ℕ, b (n + 1) / b n = r))

/-- The theorem stating the possible values of a_3 / b_3 -/
theorem sequence_ratio_values (s : SequencePair)
  (h1 : s.a 1 = s.b 1)
  (h2 : s.a 2 / s.b 2 = 2)
  (h4 : s.a 4 / s.b 4 = 8) :
  s.a 3 / s.b 3 = -5 ∨ s.a 3 / s.b 3 = -16/5 := by
  sorry

end sequence_ratio_values_l2575_257589


namespace length_width_difference_l2575_257579

/-- A rectangular hall with width being half of its length and area of 128 sq. m -/
structure RectangularHall where
  length : ℝ
  width : ℝ
  width_half_length : width = length / 2
  area_128 : length * width = 128

/-- The difference between length and width of the hall is 8 meters -/
theorem length_width_difference (hall : RectangularHall) : hall.length - hall.width = 8 := by
  sorry

end length_width_difference_l2575_257579


namespace people_per_column_l2575_257567

theorem people_per_column (total_people : ℕ) 
  (h1 : total_people = 30 * 16) 
  (h2 : total_people = 15 * (total_people / 15)) : 
  total_people / 15 = 32 := by
  sorry

end people_per_column_l2575_257567


namespace union_A_B_range_of_a_l2575_257559

-- Define sets A, B, and C
def A : Set ℝ := {x | -4 ≤ x ∧ x ≤ 0}
def B : Set ℝ := {x | x > -2}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem 1: A ∪ B = {x | x ≥ -4}
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -4} := by sorry

-- Theorem 2: Range of a is [-4, -1]
theorem range_of_a : 
  (∀ a : ℝ, C a ∩ A = C a) ↔ a ∈ Set.Icc (-4) (-1) := by sorry

end union_A_B_range_of_a_l2575_257559


namespace greatest_divisor_with_remainders_l2575_257586

theorem greatest_divisor_with_remainders (n : ℕ) : 
  (∃ q₁ : ℕ, 6215 = 144 * q₁ + 23) ∧
  (∃ q₂ : ℕ, 7373 = 144 * q₂ + 29) ∧
  (∀ m : ℕ, m > 144 → 
    (∃ q₃ q₄ : ℕ, 6215 = m * q₃ + 23 ∧ 7373 = m * q₄ + 29) → False) :=
by sorry

end greatest_divisor_with_remainders_l2575_257586


namespace right_triangle_legs_sum_l2575_257514

theorem right_triangle_legs_sum (a b c : ℕ) : 
  a + 1 = b →                -- legs are consecutive integers
  a^2 + b^2 = 41^2 →         -- Pythagorean theorem with hypotenuse 41
  a + b = 59 :=              -- sum of legs is 59
by sorry

end right_triangle_legs_sum_l2575_257514


namespace max_coefficient_of_expansion_l2575_257596

theorem max_coefficient_of_expansion : 
  ∃ (a b c d : ℕ+), 
    (∀ x : ℝ, (6 * x + 3)^3 = a * x^3 + b * x^2 + c * x + d) ∧ 
    (max a (max b (max c d)) = 324) := by
  sorry

end max_coefficient_of_expansion_l2575_257596


namespace max_value_of_function_l2575_257594

theorem max_value_of_function (x : ℝ) (h : x < 5/4) :
  ∃ (max_y : ℝ), max_y = 1 ∧ ∀ y, y = 4*x - 2 + 1/(4*x - 5) → y ≤ max_y :=
by sorry

end max_value_of_function_l2575_257594


namespace handshake_theorem_l2575_257568

/-- The number of handshakes in a group where each person shakes hands with a fixed number of others -/
def total_handshakes (n : ℕ) (k : ℕ) : ℕ := n * k / 2

/-- Theorem: In a group of 30 people, where each person shakes hands with exactly 3 others, 
    the total number of handshakes is 45 -/
theorem handshake_theorem : 
  total_handshakes 30 3 = 45 := by
  sorry

end handshake_theorem_l2575_257568


namespace andy_cookies_l2575_257517

/-- Represents the number of cookies taken by each basketball team member -/
def basketballTeamCookies (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of cookies taken by all basketball team members -/
def totalTeamCookies (teamSize : ℕ) : ℕ :=
  (teamSize * (basketballTeamCookies 1 + basketballTeamCookies teamSize)) / 2

theorem andy_cookies (initialCookies brotherCookies teamSize : ℕ) 
  (h1 : initialCookies = 72)
  (h2 : brotherCookies = 5)
  (h3 : teamSize = 8)
  (h4 : totalTeamCookies teamSize + brotherCookies < initialCookies) :
  initialCookies - (totalTeamCookies teamSize + brotherCookies) = 3 := by
  sorry

end andy_cookies_l2575_257517


namespace base7_4513_equals_1627_l2575_257534

/-- Converts a base-7 digit to its base-10 equivalent --/
def base7ToBase10Digit (d : ℕ) : ℕ := d

/-- Converts a list of base-7 digits to a base-10 number --/
def base7ToBase10 (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

theorem base7_4513_equals_1627 :
  base7ToBase10 [3, 1, 5, 4] = 1627 := by sorry

end base7_4513_equals_1627_l2575_257534


namespace contest_probability_l2575_257582

theorem contest_probability (p q : ℝ) (h_p : p = 2/3) (h_q : q = 1/3) :
  ∃ n : ℕ, n > 0 ∧ (p ^ n < 0.05) ∧ ∀ m : ℕ, m > 0 → m < n → p ^ m ≥ 0.05 :=
sorry

end contest_probability_l2575_257582


namespace level3_available_spots_l2575_257576

/-- Represents a parking level in a multi-story parking lot -/
structure ParkingLevel where
  totalSpots : ℕ
  parkedCars : ℕ
  reservedParkedCars : ℕ

/-- Calculates the available non-reserved parking spots on a given level -/
def availableNonReservedSpots (level : ParkingLevel) : ℕ :=
  level.totalSpots - (level.parkedCars - level.reservedParkedCars)

/-- Theorem stating that the available non-reserved parking spots on level 3 is 450 -/
theorem level3_available_spots :
  let level3 : ParkingLevel := {
    totalSpots := 480,
    parkedCars := 45,
    reservedParkedCars := 15
  }
  availableNonReservedSpots level3 = 450 := by
  sorry

end level3_available_spots_l2575_257576


namespace company_fund_problem_l2575_257548

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) :
  (80 * n = initial_fund + 8) →
  (70 * n + 160 = initial_fund) →
  initial_fund = 1352 := by
  sorry

end company_fund_problem_l2575_257548


namespace circle_equation_through_points_l2575_257578

theorem circle_equation_through_points : 
  let equation (x y : ℝ) := x^2 + y^2 - 4*x - 6*y
  ∀ (x y : ℝ), 
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) → 
    equation x y = 0 := by
  sorry

end circle_equation_through_points_l2575_257578


namespace trig_identity_l2575_257539

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos (2 * α) + Real.sin (2 * α)) = 5 := by
sorry

end trig_identity_l2575_257539


namespace tuesday_greatest_diff_greatest_diff_day_is_tuesday_l2575_257518

-- Define the temperature difference for each day
def monday_diff : ℤ := 5 - 2
def tuesday_diff : ℤ := 4 - (-1)
def wednesday_diff : ℤ := 0 - (-4)

-- Theorem stating that Tuesday has the greatest temperature difference
theorem tuesday_greatest_diff : 
  tuesday_diff > monday_diff ∧ tuesday_diff > wednesday_diff :=
by
  sorry

-- Define a function to get the day with the greatest temperature difference
def day_with_greatest_diff : String :=
  if tuesday_diff > monday_diff ∧ tuesday_diff > wednesday_diff then
    "Tuesday"
  else if monday_diff > tuesday_diff ∧ monday_diff > wednesday_diff then
    "Monday"
  else
    "Wednesday"

-- Theorem stating that the day with the greatest temperature difference is Tuesday
theorem greatest_diff_day_is_tuesday : 
  day_with_greatest_diff = "Tuesday" :=
by
  sorry

end tuesday_greatest_diff_greatest_diff_day_is_tuesday_l2575_257518


namespace lawrence_county_houses_l2575_257561

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := 574

/-- The total number of houses in Lawrence County after the housing boom -/
def total_houses : ℕ := houses_before + houses_built

theorem lawrence_county_houses : total_houses = 2000 := by
  sorry

end lawrence_county_houses_l2575_257561


namespace lcm_of_9_12_15_l2575_257512

theorem lcm_of_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end lcm_of_9_12_15_l2575_257512


namespace product_a4_b4_l2575_257571

theorem product_a4_b4 (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) 
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
sorry

end product_a4_b4_l2575_257571


namespace lollipop_distribution_l2575_257595

theorem lollipop_distribution (num_kids : ℕ) (additional_lollipops : ℕ) (initial_lollipops : ℕ) : 
  num_kids = 42 → 
  additional_lollipops = 22 → 
  (initial_lollipops + additional_lollipops) % num_kids = 0 → 
  initial_lollipops < num_kids → 
  initial_lollipops = 62 := by
sorry

end lollipop_distribution_l2575_257595


namespace garden_usable_area_l2575_257502

/-- Calculate the usable area of a rectangular garden with a square pond in one corner -/
theorem garden_usable_area 
  (garden_length : ℝ) 
  (garden_width : ℝ) 
  (pond_side : ℝ) 
  (h1 : garden_length = 20) 
  (h2 : garden_width = 18) 
  (h3 : pond_side = 4) : 
  garden_length * garden_width - pond_side * pond_side = 344 := by
  sorry

#check garden_usable_area

end garden_usable_area_l2575_257502


namespace solve_worker_problem_l2575_257569

/-- Represents the work rate of one person -/
structure WorkRate where
  rate : ℝ

/-- Represents a group of workers -/
structure WorkerGroup where
  men : ℕ
  women : ℕ

/-- Calculates the total work rate of a group -/
def totalWorkRate (g : WorkerGroup) (m w : WorkRate) : ℝ :=
  (g.men : ℝ) * m.rate + (g.women : ℝ) * w.rate

theorem solve_worker_problem (m w : WorkRate) : ∃ x : ℕ,
  let group1 := WorkerGroup.mk 3 8
  let group2 := WorkerGroup.mk x 2
  let group3 := WorkerGroup.mk 3 2
  totalWorkRate group1 m w = totalWorkRate group2 m w ∧
  totalWorkRate group3 m w = (4/7 : ℝ) * totalWorkRate group1 m w ∧
  x = 6 := by
  sorry

end solve_worker_problem_l2575_257569


namespace one_thirds_in_nine_thirds_l2575_257523

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 / 3) = 9 := by sorry

end one_thirds_in_nine_thirds_l2575_257523


namespace square_sum_primes_l2575_257553

theorem square_sum_primes (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r)
  (h1 : ∃ a : ℕ, pq + 1 = a^2)
  (h2 : ∃ b : ℕ, pr + 1 = b^2)
  (h3 : ∃ c : ℕ, qr - p = c^2) :
  ∃ d : ℕ, p + 2*q*r + 2 = d^2 := by
sorry

end square_sum_primes_l2575_257553


namespace vartan_recreation_spending_l2575_257531

theorem vartan_recreation_spending :
  ∀ (last_week_wages : ℝ) (last_week_percent : ℝ),
  last_week_percent > 0 →
  let this_week_wages := 0.9 * last_week_wages
  let last_week_spending := (last_week_percent / 100) * last_week_wages
  let this_week_spending := 0.3 * this_week_wages
  this_week_spending = 1.8 * last_week_spending →
  last_week_percent = 15 := by
sorry

end vartan_recreation_spending_l2575_257531


namespace coin_circumference_diameter_ratio_l2575_257565

theorem coin_circumference_diameter_ratio :
  let diameter : ℝ := 100
  let circumference : ℝ := 314
  circumference / diameter = 3.14 := by sorry

end coin_circumference_diameter_ratio_l2575_257565


namespace base_ten_to_five_235_l2575_257536

/-- Converts a number from base 10 to base 5 -/
def toBaseFive (n : ℕ) : List ℕ :=
  sorry

theorem base_ten_to_five_235 :
  toBaseFive 235 = [1, 4, 2, 0] :=
sorry

end base_ten_to_five_235_l2575_257536
