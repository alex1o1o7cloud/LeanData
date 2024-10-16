import Mathlib

namespace NUMINAMATH_CALUDE_circle_equation_l2616_261697

/-- The equation of a circle with center (-1, 2) and radius 4 -/
theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (-1, 2)
  let radius : ℝ := 4
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ (x + 1)^2 + (y - 2)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2616_261697


namespace NUMINAMATH_CALUDE_john_mary_distance_difference_l2616_261687

/-- The width of the streets in feet -/
def street_width : ℕ := 15

/-- The side length of a block in feet -/
def block_side_length : ℕ := 300

/-- The perimeter of a square -/
def square_perimeter (side_length : ℕ) : ℕ := 4 * side_length

theorem john_mary_distance_difference :
  square_perimeter (block_side_length + 2 * street_width) - square_perimeter block_side_length = 120 := by
  sorry

end NUMINAMATH_CALUDE_john_mary_distance_difference_l2616_261687


namespace NUMINAMATH_CALUDE_joe_pocket_transfer_l2616_261690

/-- Represents the money transfer problem with Joe's pockets --/
def MoneyTransferProblem (total initial_left transfer_amount : ℚ) : Prop :=
  let initial_right := total - initial_left
  let after_quarter_left := initial_left - (initial_left / 4)
  let after_quarter_right := initial_right + (initial_left / 4)
  let final_left := after_quarter_left - transfer_amount
  let final_right := after_quarter_right + transfer_amount
  (total = 200) ∧ 
  (initial_left = 160) ∧ 
  (final_left = final_right) ∧
  (transfer_amount > 0)

theorem joe_pocket_transfer : 
  ∃ (transfer_amount : ℚ), MoneyTransferProblem 200 160 transfer_amount ∧ transfer_amount = 20 := by
  sorry

end NUMINAMATH_CALUDE_joe_pocket_transfer_l2616_261690


namespace NUMINAMATH_CALUDE_semicircle_radius_equals_rectangle_area_l2616_261656

theorem semicircle_radius_equals_rectangle_area (width length : ℝ) (h1 : width = 3) (h2 : length = 8) :
  let rectangle_area := width * length
  let semicircle_area (r : ℝ) := (π * r^2) / 2
  ∃ r : ℝ, semicircle_area r = rectangle_area ∧ r = Real.sqrt (48 / π) :=
by sorry

end NUMINAMATH_CALUDE_semicircle_radius_equals_rectangle_area_l2616_261656


namespace NUMINAMATH_CALUDE_zero_point_implies_a_range_l2616_261694

theorem zero_point_implies_a_range (a : ℝ) : 
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, a * x + 1 - 2 * a = 0) → 
  a ∈ Set.Ioo (1/3 : ℝ) 1 := by
sorry

end NUMINAMATH_CALUDE_zero_point_implies_a_range_l2616_261694


namespace NUMINAMATH_CALUDE_frog_jump_distance_l2616_261670

/-- The jumping contest between a grasshopper and a frog -/
theorem frog_jump_distance (grasshopper_jump : ℕ) (frog_extra_distance : ℕ) : 
  grasshopper_jump = 9 → frog_extra_distance = 3 → 
  grasshopper_jump + frog_extra_distance = 12 := by
  sorry

end NUMINAMATH_CALUDE_frog_jump_distance_l2616_261670


namespace NUMINAMATH_CALUDE_joes_fast_food_cost_l2616_261653

theorem joes_fast_food_cost : 
  let sandwich_cost : ℕ := 4
  let soda_cost : ℕ := 3
  let num_sandwiches : ℕ := 3
  let num_sodas : ℕ := 5
  (num_sandwiches * sandwich_cost + num_sodas * soda_cost) = 27 := by
sorry

end NUMINAMATH_CALUDE_joes_fast_food_cost_l2616_261653


namespace NUMINAMATH_CALUDE_equal_angles_necessary_not_sufficient_l2616_261696

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  sorry -- Definition of a square

-- Define the property of having four equal interior angles
def has_four_equal_angles (q : Quadrilateral) : Prop :=
  sorry -- Definition of having four equal interior angles

theorem equal_angles_necessary_not_sufficient :
  (∀ q : Quadrilateral, is_square q → has_four_equal_angles q) ∧
  (∃ q : Quadrilateral, has_four_equal_angles q ∧ ¬is_square q) :=
sorry

end NUMINAMATH_CALUDE_equal_angles_necessary_not_sufficient_l2616_261696


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_zero_l2616_261628

theorem sum_of_fifth_powers_zero (a b c : ℚ) 
  (sum_zero : a + b + c = 0) 
  (sum_cubes_nonzero : a^3 + b^3 + c^3 ≠ 0) : 
  a^5 + b^5 + c^5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_zero_l2616_261628


namespace NUMINAMATH_CALUDE_inequality_proof_l2616_261651

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^3 / (a^3 + 2*b^2)) + (b^3 / (b^3 + 2*c^2)) + (c^3 / (c^3 + 2*a^2)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2616_261651


namespace NUMINAMATH_CALUDE_amc10_paths_l2616_261692

/-- The number of 'M's adjacent to the central 'A' -/
def num_m_adj_a : ℕ := 4

/-- The number of 'C's adjacent to each 'M' -/
def num_c_adj_m : ℕ := 4

/-- The number of '10's adjacent to each 'C' -/
def num_10_adj_c : ℕ := 5

/-- The total number of paths to spell "AMC10" -/
def total_paths : ℕ := num_m_adj_a * num_c_adj_m * num_10_adj_c

theorem amc10_paths : total_paths = 80 := by
  sorry

end NUMINAMATH_CALUDE_amc10_paths_l2616_261692


namespace NUMINAMATH_CALUDE_extended_segment_coordinates_l2616_261658

/-- Given points A and B, and a point C on the line extending AB such that BC = 1/2 * AB,
    prove that the coordinates of C are (12, 12). -/
theorem extended_segment_coordinates :
  let A : ℝ × ℝ := (3, 3)
  let B : ℝ × ℝ := (9, 9)
  let C : ℝ × ℝ := (12, 12)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  BC.1 = (1/2) * AB.1 ∧ BC.2 = (1/2) * AB.2 :=
by sorry

end NUMINAMATH_CALUDE_extended_segment_coordinates_l2616_261658


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2616_261637

theorem fraction_equals_zero (x : ℝ) (h : x = 1) : (2 * x - 2) / (x - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2616_261637


namespace NUMINAMATH_CALUDE_number_count_proof_l2616_261606

theorem number_count_proof (avg_all : ℝ) (avg_pair1 avg_pair2 avg_pair3 : ℝ) 
  (h1 : avg_all = 5.40)
  (h2 : avg_pair1 = 5.2)
  (h3 : avg_pair2 = 5.80)
  (h4 : avg_pair3 = 5.200000000000003) :
  (2 * avg_pair1 + 2 * avg_pair2 + 2 * avg_pair3) / avg_all = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_count_proof_l2616_261606


namespace NUMINAMATH_CALUDE_smallest_triangle_side_l2616_261672

theorem smallest_triangle_side : ∃ (t : ℕ), 
  (∀ (s : ℕ), s < t → ¬(7 < s + 13 ∧ 13 < 7 + s ∧ s < 7 + 13)) ∧ 
  (7 < t + 13 ∧ 13 < 7 + t ∧ t < 7 + 13) :=
by sorry

end NUMINAMATH_CALUDE_smallest_triangle_side_l2616_261672


namespace NUMINAMATH_CALUDE_bumper_car_line_problem_l2616_261625

theorem bumper_car_line_problem (initial_people : ℕ) (people_left : ℕ) (total_people : ℕ) : 
  initial_people = 9 →
  people_left = 6 →
  total_people = 18 →
  total_people - (initial_people - people_left) = 15 := by
sorry

end NUMINAMATH_CALUDE_bumper_car_line_problem_l2616_261625


namespace NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l2616_261676

theorem contrapositive_square_sum_zero (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l2616_261676


namespace NUMINAMATH_CALUDE_salary_calculation_correct_l2616_261620

/-- Calculates the salary after three months of increases -/
def salary_after_three_months (initial_salary : ℝ) (first_month_increase : ℝ) : ℝ :=
  let month1 := initial_salary * (1 + first_month_increase)
  let month2 := month1 * (1 + 2 * first_month_increase)
  let month3 := month2 * (1 + 4 * first_month_increase)
  month3

/-- Theorem stating that the salary after three months matches the expected value -/
theorem salary_calculation_correct : 
  salary_after_three_months 2000 0.05 = 2772 := by
  sorry


end NUMINAMATH_CALUDE_salary_calculation_correct_l2616_261620


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l2616_261640

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone. -/
def valid_seating (table : CircularTable) : Prop :=
  table.seated_people > 0 ∧ 
  table.seated_people ≤ table.total_chairs ∧
  ∀ k : ℕ, k < table.total_chairs → ∃ i j : ℕ, 
    i < table.seated_people ∧ 
    j < table.seated_people ∧ 
    (k - i) % table.total_chairs ≤ 2 ∧ 
    (j - k) % table.total_chairs ≤ 2

/-- The theorem stating the smallest number of people that can be seated. -/
theorem smallest_valid_seating :
  ∀ n : ℕ, n < 20 → ¬(valid_seating ⟨60, n⟩) ∧ 
  valid_seating ⟨60, 20⟩ :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_l2616_261640


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l2616_261673

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 36 = 0 → x ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l2616_261673


namespace NUMINAMATH_CALUDE_inequality_implication_l2616_261604

theorem inequality_implication (x y : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_ineq : 4 * Real.log x + 2 * Real.log (2 * y) ≥ x^2 + 8 * y - 4) : 
  x * y = Real.sqrt 2 / 4 ∧ x + 2 * y = 1 / 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2616_261604


namespace NUMINAMATH_CALUDE_angle_degree_proof_l2616_261610

theorem angle_degree_proof (x : ℝ) : 
  x = 2 * (180 - x) + 30 → x = 130 := by
  sorry

end NUMINAMATH_CALUDE_angle_degree_proof_l2616_261610


namespace NUMINAMATH_CALUDE_initial_oranges_count_l2616_261633

/-- The number of oranges initially in the bin -/
def initial_oranges : ℕ := sorry

/-- The number of oranges thrown away -/
def thrown_away : ℕ := 37

/-- The number of new oranges added -/
def new_oranges : ℕ := 7

/-- The final number of oranges in the bin -/
def final_oranges : ℕ := 10

/-- Theorem stating that the initial number of oranges was 40 -/
theorem initial_oranges_count : initial_oranges = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_oranges_count_l2616_261633


namespace NUMINAMATH_CALUDE_min_M_and_F_M_l2616_261685

def is_k_multiple (n : ℕ) (k : ℤ) : Prop :=
  let thousands := n / 1000
  let hundreds := (n / 100) % 10
  let tens := (n / 10) % 10
  let units := n % 10
  (thousands + hundreds : ℤ) = k * (tens - units)

def swap_hundreds_tens (n : ℕ) : ℕ :=
  let thousands := n / 1000
  let hundreds := (n / 100) % 10
  let tens := (n / 10) % 10
  let units := n % 10
  thousands * 1000 + tens * 100 + hundreds * 10 + units

def F (m : ℕ) : ℚ :=
  let a : ℕ := (m + 1) / 2
  let b : ℕ := (m - 1) / 2
  (a : ℚ) / b

theorem min_M_and_F_M :
  ∃ (M : ℕ),
    M ≥ 1000 ∧ M < 10000 ∧
    is_k_multiple M 4 ∧
    is_k_multiple (M - 4) (-3) ∧
    is_k_multiple (swap_hundreds_tens M) 4 ∧
    (∀ (N : ℕ), N ≥ 1000 ∧ N < 10000 ∧
      is_k_multiple N 4 ∧
      is_k_multiple (N - 4) (-3) ∧
      is_k_multiple (swap_hundreds_tens N) 4 →
      M ≤ N) ∧
    M = 6663 ∧
    F M = 3332 / 3331 := by sorry

end NUMINAMATH_CALUDE_min_M_and_F_M_l2616_261685


namespace NUMINAMATH_CALUDE_isabel_music_purchase_l2616_261614

theorem isabel_music_purchase (country_albums : ℕ) (pop_albums : ℕ) (songs_per_album : ℕ) : 
  country_albums = 4 → pop_albums = 5 → songs_per_album = 8 → 
  (country_albums + pop_albums) * songs_per_album = 72 := by
sorry

end NUMINAMATH_CALUDE_isabel_music_purchase_l2616_261614


namespace NUMINAMATH_CALUDE_stock_loss_percentage_l2616_261627

theorem stock_loss_percentage 
  (total_stock : ℝ) 
  (profit_percentage : ℝ) 
  (profit_stock_ratio : ℝ) 
  (overall_loss : ℝ) :
  total_stock = 22500 →
  profit_percentage = 10 →
  profit_stock_ratio = 20 →
  overall_loss = 450 →
  ∃ (loss_percentage : ℝ),
    loss_percentage = 5 ∧
    overall_loss = (loss_percentage / 100 * (100 - profit_stock_ratio) / 100 * total_stock) - 
                   (profit_percentage / 100 * profit_stock_ratio / 100 * total_stock) := by
  sorry

end NUMINAMATH_CALUDE_stock_loss_percentage_l2616_261627


namespace NUMINAMATH_CALUDE_election_result_theorem_l2616_261667

/-- Represents the result of an election with five candidates -/
structure ElectionResult where
  total_votes : ℕ
  candidate1_votes : ℕ
  candidate2_votes : ℕ
  candidate3_votes : ℕ
  candidate4_votes : ℕ
  candidate5_votes : ℕ

/-- Theorem stating the election result given the conditions -/
theorem election_result_theorem (er : ElectionResult) : 
  er.candidate1_votes = (30 * er.total_votes) / 100 ∧
  er.candidate2_votes = (20 * er.total_votes) / 100 ∧
  er.candidate3_votes = 3000 ∧
  er.candidate3_votes = (15 * er.total_votes) / 100 ∧
  er.candidate4_votes = (25 * er.total_votes) / 100 ∧
  er.candidate5_votes = 2 * er.candidate3_votes →
  er.total_votes = 20000 ∧
  er.candidate1_votes = 6000 ∧
  er.candidate2_votes = 4000 ∧
  er.candidate3_votes = 3000 ∧
  er.candidate4_votes = 5000 ∧
  er.candidate5_votes = 6000 :=
by
  sorry

end NUMINAMATH_CALUDE_election_result_theorem_l2616_261667


namespace NUMINAMATH_CALUDE_gcd_840_1785_f_2_equals_62_l2616_261649

-- Define the polynomial f(x) = 2x⁴ + 3x³ + 5x - 4
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

-- Theorem for the GCD of 840 and 1785
theorem gcd_840_1785 : Nat.gcd 840 1785 = 105 := by sorry

-- Theorem for the value of f(2)
theorem f_2_equals_62 : f 2 = 62 := by sorry

end NUMINAMATH_CALUDE_gcd_840_1785_f_2_equals_62_l2616_261649


namespace NUMINAMATH_CALUDE_unique_divisible_by_44_l2616_261618

/-- Represents a six-digit number in the form 5n7264 where n is a single digit -/
def sixDigitNumber (n : Nat) : Nat :=
  500000 + 10000 * n + 7264

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (a b : Nat) : Prop :=
  ∃ k, a = b * k

/-- Theorem stating that 517264 is the only number in the form 5n7264 
    (where n is a single digit) that is divisible by 44 -/
theorem unique_divisible_by_44 : 
  ∀ n : Nat, n < 10 → 
    (isDivisibleBy (sixDigitNumber n) 44 ↔ n = 1) := by
  sorry

#check unique_divisible_by_44

end NUMINAMATH_CALUDE_unique_divisible_by_44_l2616_261618


namespace NUMINAMATH_CALUDE_probability_non_defective_second_draw_l2616_261607

def total_products : ℕ := 100
def defective_products : ℕ := 3

theorem probability_non_defective_second_draw :
  let remaining_total := total_products - 1
  let remaining_defective := defective_products - 1
  let remaining_non_defective := remaining_total - remaining_defective
  (remaining_non_defective : ℚ) / remaining_total = 97 / 99 :=
sorry

end NUMINAMATH_CALUDE_probability_non_defective_second_draw_l2616_261607


namespace NUMINAMATH_CALUDE_fibonacci_sum_theorem_l2616_261621

/-- Definition of the Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of the Fibonacci series divided by powers of 10 -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / (10 : ℝ) ^ n

/-- Theorem stating that the sum of Fₙ/10ⁿ from n=0 to infinity equals 10/89 -/
theorem fibonacci_sum_theorem : fibSum = 10 / 89 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_sum_theorem_l2616_261621


namespace NUMINAMATH_CALUDE_no_eight_consecutive_odd_exponent_primes_l2616_261666

theorem no_eight_consecutive_odd_exponent_primes :
  ∀ n : ℕ, ∃ k : ℕ, k ∈ Finset.range 8 ∧
  ∃ p : ℕ, Prime p ∧ ∃ m : ℕ, m > 0 ∧ 2 ∣ m ∧ p ^ m ∣ (n + k) := by
  sorry

end NUMINAMATH_CALUDE_no_eight_consecutive_odd_exponent_primes_l2616_261666


namespace NUMINAMATH_CALUDE_point_q_coordinates_l2616_261678

/-- A point on the unit circle --/
structure PointOnUnitCircle where
  x : ℝ
  y : ℝ
  on_circle : x^2 + y^2 = 1

/-- The arc length between two points on the unit circle --/
def arcLength (p q : PointOnUnitCircle) : ℝ := sorry

theorem point_q_coordinates :
  ∀ (p q : PointOnUnitCircle),
  p.x = 1 ∧ p.y = 0 →
  arcLength p q = π / 3 →
  q.x = 1 / 2 ∧ q.y = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_point_q_coordinates_l2616_261678


namespace NUMINAMATH_CALUDE_ladder_height_l2616_261655

theorem ladder_height (ladder_length : Real) (base_distance : Real) (height : Real) :
  ladder_length = 13 →
  base_distance = 5 →
  ladder_length^2 = height^2 + base_distance^2 →
  height = 12 :=
by sorry

end NUMINAMATH_CALUDE_ladder_height_l2616_261655


namespace NUMINAMATH_CALUDE_quadratic_function_derivative_l2616_261613

theorem quadratic_function_derivative (a c : ℝ) :
  (∀ x, deriv (fun x => a * x^2 + c) x = 2 * a * x) →
  deriv (fun x => a * x^2 + c) 1 = 2 →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_derivative_l2616_261613


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_range_of_a_l2616_261617

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

-- Part 1
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B 1 = {x : ℝ | -1 < x ∧ x < 1} := by sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  B a ∩ A = B a → a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_range_of_a_l2616_261617


namespace NUMINAMATH_CALUDE_geometry_exam_average_score_l2616_261698

/-- Represents a student in the geometry exam -/
structure Student where
  name : String
  mistakes : ℕ
  score : ℚ

/-- Represents the geometry exam -/
structure GeometryExam where
  totalProblems : ℕ
  firstSectionProblems : ℕ
  firstSectionPoints : ℕ
  secondSectionPoints : ℕ
  firstSectionDeduction : ℕ
  secondSectionDeduction : ℕ

theorem geometry_exam_average_score 
  (exam : GeometryExam)
  (madeline leo brent nicholas : Student)
  (h_exam : exam.totalProblems = 15 ∧ 
            exam.firstSectionProblems = 5 ∧ 
            exam.firstSectionPoints = 3 ∧ 
            exam.secondSectionPoints = 1 ∧
            exam.firstSectionDeduction = 2 ∧
            exam.secondSectionDeduction = 1)
  (h_madeline : madeline.mistakes = 2)
  (h_leo : leo.mistakes = 2 * madeline.mistakes)
  (h_brent : brent.score = 25 ∧ brent.mistakes = leo.mistakes + 1)
  (h_nicholas : nicholas.mistakes = 3 * madeline.mistakes ∧ 
                nicholas.score = brent.score - 5) :
  (madeline.score + leo.score + brent.score + nicholas.score) / 4 = 22.25 := by
  sorry

end NUMINAMATH_CALUDE_geometry_exam_average_score_l2616_261698


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l2616_261615

theorem chicken_rabbit_problem (x y : ℕ) : 
  (x + y = 35 ∧ 2 * x + 4 * y = 94) ↔ 
  (x + y = 35 ∧ x * 2 + y * 4 = 94) := by sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l2616_261615


namespace NUMINAMATH_CALUDE_sequence_properties_l2616_261661

def S (n : ℕ) : ℤ := n^2 - 9*n

def a (n : ℕ) : ℤ := 2*n - 10

theorem sequence_properties :
  (∀ n, S (n+1) - S n = a (n+1)) ∧
  (∃! k : ℕ, k > 0 ∧ 5 < a k ∧ a k < 8) ∧
  (∀ k : ℕ, k > 0 ∧ 5 < a k ∧ a k < 8 → k = 8) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2616_261661


namespace NUMINAMATH_CALUDE_collinearity_condition_for_linear_combination_l2616_261677

/-- Given points O, A, B are not collinear, and vector OP = m * vector OA + n * vector OB,
    points A, P, B are collinear if and only if m + n = 1 -/
theorem collinearity_condition_for_linear_combination
  (O A B P : EuclideanSpace ℝ (Fin 3))
  (m n : ℝ)
  (h_not_collinear : ¬ Collinear ℝ {O, A, B})
  (h_linear_combination : P - O = m • (A - O) + n • (B - O)) :
  Collinear ℝ {A, P, B} ↔ m + n = 1 := by sorry

end NUMINAMATH_CALUDE_collinearity_condition_for_linear_combination_l2616_261677


namespace NUMINAMATH_CALUDE_prob_only_one_AB_qualifies_prob_at_least_one_qualifies_l2616_261626

-- Define the probabilities for each student passing each round
def prob_written_A : ℚ := 2/3
def prob_written_B : ℚ := 1/2
def prob_written_C : ℚ := 3/4
def prob_interview_A : ℚ := 1/2
def prob_interview_B : ℚ := 2/3
def prob_interview_C : ℚ := 1/3

-- Define the probability of each student qualifying for the finals
def prob_qualify_A : ℚ := prob_written_A * prob_interview_A
def prob_qualify_B : ℚ := prob_written_B * prob_interview_B
def prob_qualify_C : ℚ := prob_written_C * prob_interview_C

-- Theorem for the first question
theorem prob_only_one_AB_qualifies :
  (prob_qualify_A * (1 - prob_qualify_B) + (1 - prob_qualify_A) * prob_qualify_B) = 4/9 := by
  sorry

-- Theorem for the second question
theorem prob_at_least_one_qualifies :
  (1 - (1 - prob_qualify_A) * (1 - prob_qualify_B) * (1 - prob_qualify_C)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_only_one_AB_qualifies_prob_at_least_one_qualifies_l2616_261626


namespace NUMINAMATH_CALUDE_stella_annual_income_after_tax_l2616_261616

/-- Calculates Stella's annual income after tax deduction --/
theorem stella_annual_income_after_tax :
  let base_salary : ℕ := 3500
  let bonuses : List ℕ := [1200, 600, 1500, 900, 1200]
  let paid_months : ℕ := 10
  let tax_rate : ℚ := 1 / 20

  let total_base_salary := base_salary * paid_months
  let total_bonuses := bonuses.sum
  let total_income := total_base_salary + total_bonuses
  let tax_deduction := (total_income : ℚ) * tax_rate
  let annual_income_after_tax := (total_income : ℚ) - tax_deduction

  annual_income_after_tax = 38380 := by
  sorry

end NUMINAMATH_CALUDE_stella_annual_income_after_tax_l2616_261616


namespace NUMINAMATH_CALUDE_jerome_theorem_l2616_261650

def jerome_problem (initial_money : ℝ) : Prop :=
  let half_money : ℝ := 43
  let meg_amount : ℝ := 8
  let bianca_amount : ℝ := 3 * meg_amount
  let after_meg_bianca : ℝ := initial_money - meg_amount - bianca_amount
  let nathan_amount : ℝ := after_meg_bianca / 2
  let after_nathan : ℝ := after_meg_bianca - nathan_amount
  let charity_percentage : ℝ := 0.2
  let charity_amount : ℝ := charity_percentage * after_nathan
  let final_amount : ℝ := after_nathan - charity_amount

  (initial_money / 2 = half_money) ∧
  (final_amount = 21.60)

theorem jerome_theorem : 
  ∃ (initial_money : ℝ), jerome_problem initial_money :=
by
  sorry

end NUMINAMATH_CALUDE_jerome_theorem_l2616_261650


namespace NUMINAMATH_CALUDE_good_2013_implies_good_20_l2616_261680

/-- A sequence of positive integers is non-decreasing -/
def IsNonDecreasingSeq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

/-- A number is good if it can be expressed as i/a_i for some index i -/
def IsGood (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∃ i : ℕ, n = i / a i

theorem good_2013_implies_good_20 (a : ℕ → ℕ) 
  (h_nondec : IsNonDecreasingSeq a) 
  (h_2013 : IsGood 2013 a) : 
  IsGood 20 a :=
sorry

end NUMINAMATH_CALUDE_good_2013_implies_good_20_l2616_261680


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l2616_261683

theorem modular_arithmetic_problem : ∃ (a b : ℤ), 
  (7 * a) % 60 = 1 ∧ 
  (13 * b) % 60 = 1 ∧ 
  (3 * a + 9 * b) % 60 = 42 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l2616_261683


namespace NUMINAMATH_CALUDE_M_elements_l2616_261662

def M : Set (ℕ × ℕ) := {p | p.1 + p.2 ≤ 1}

theorem M_elements : M = {(0, 0), (0, 1), (1, 0)} := by
  sorry

end NUMINAMATH_CALUDE_M_elements_l2616_261662


namespace NUMINAMATH_CALUDE_range_of_m_for_inequality_l2616_261674

theorem range_of_m_for_inequality (m : ℝ) : 
  (∃ x : ℝ, Real.sqrt ((x + m) ^ 2) + Real.sqrt ((x - 1) ^ 2) ≤ 3) ↔ 
  -4 ≤ m ∧ m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_inequality_l2616_261674


namespace NUMINAMATH_CALUDE_compound_inequality_solution_l2616_261693

theorem compound_inequality_solution (x : ℝ) :
  (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 6) ↔ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_compound_inequality_solution_l2616_261693


namespace NUMINAMATH_CALUDE_max_value_expression_l2616_261622

theorem max_value_expression :
  ∃ (x y : ℝ),
    ∀ (a b : ℝ),
      (Real.sqrt (9 - Real.sqrt 7) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 1) *
      (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos y - Real.cos (2 * y)) ≤
      (Real.sqrt (9 - Real.sqrt 7) * Real.sin a - Real.sqrt (2 * (1 + Real.cos (2 * a))) - 1) *
      (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos b - Real.cos (2 * b)) →
      (Real.sqrt (9 - Real.sqrt 7) * Real.sin a - Real.sqrt (2 * (1 + Real.cos (2 * a))) - 1) *
      (3 + 2 * Real.sqrt (13 - Real.sqrt 7) * Real.cos b - Real.cos (2 * b)) = 24 - 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2616_261622


namespace NUMINAMATH_CALUDE_x_value_l2616_261609

theorem x_value : ∃ x : ℝ, (x = (1/x) * (-x) - 5) ∧ (x^2 - 3*x + 2 ≥ 0) ∧ (x = -6) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2616_261609


namespace NUMINAMATH_CALUDE_min_value_expression_l2616_261682

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1 / 2) :
  x^3 + 4*x*y + 16*y^3 + 8*y*z + 3*z^3 ≥ 18 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 / 2 ∧
    x₀^3 + 4*x₀*y₀ + 16*y₀^3 + 8*y₀*z₀ + 3*z₀^3 = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2616_261682


namespace NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_four_l2616_261603

theorem sqrt_plus_square_zero_implies_diff_four (m n : ℝ) : 
  Real.sqrt (m - 3) + (n + 1)^2 = 0 → m - n = 4 := by
sorry

end NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_four_l2616_261603


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2616_261699

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^3 + y^3 + z^3 ≥ x^2 * Real.sqrt (y*z) + y^2 * Real.sqrt (z*x) + z^2 * Real.sqrt (x*y) :=
sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2616_261699


namespace NUMINAMATH_CALUDE_triangle_properties_l2616_261664

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  let m : Fin 2 → ℝ := ![tan A + tan C, sqrt 3]
  let n : Fin 2 → ℝ := ![tan A * tan C - 1, 1]
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  b = 2 →
  (∃ (k : ℝ), m = k • n) →
  (B = π / 3 ∧
   (∀ S : ℝ, S = 1/2 * a * c * sin B → S ≤ sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2616_261664


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2616_261665

/-- The perimeter of a rectangular field with area 800 square meters and width 20 meters is 120 meters. -/
theorem rectangle_perimeter (area width : ℝ) (h_area : area = 800) (h_width : width = 20) :
  2 * (area / width + width) = 120 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2616_261665


namespace NUMINAMATH_CALUDE_power_sum_of_i_l2616_261611

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^203 = -2*i := by
  sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l2616_261611


namespace NUMINAMATH_CALUDE_not_obtainable_2013201420152016_l2616_261602

/-- Represents the state of the board -/
structure Board :=
  (left : ℕ)
  (right : ℕ)

/-- Represents a single operation on the board -/
def operate (b : Board) : Board :=
  { left := b.left * b.right,
    right := b.left^3 + b.right^3 }

/-- Checks if a number is obtainable on the board -/
def is_obtainable (n : ℕ) : Prop :=
  ∃ (b : Board), ∃ (k : ℕ), 
    (Nat.iterate operate k { left := 21, right := 8 }).left = n ∨
    (Nat.iterate operate k { left := 21, right := 8 }).right = n

/-- The main theorem stating that 2013201420152016 is not obtainable -/
theorem not_obtainable_2013201420152016 : 
  ¬ is_obtainable 2013201420152016 := by
  sorry


end NUMINAMATH_CALUDE_not_obtainable_2013201420152016_l2616_261602


namespace NUMINAMATH_CALUDE_martin_crayons_l2616_261632

theorem martin_crayons (total_boxes : ℕ) (crayons_per_box : ℕ) (boxes_with_missing : ℕ) (missing_per_box : ℕ) :
  total_boxes = 8 →
  crayons_per_box = 7 →
  boxes_with_missing = 3 →
  missing_per_box = 2 →
  total_boxes * crayons_per_box - boxes_with_missing * missing_per_box = 50 :=
by sorry

end NUMINAMATH_CALUDE_martin_crayons_l2616_261632


namespace NUMINAMATH_CALUDE_jessie_weight_before_jogging_l2616_261654

def weight_before_jogging (weight_after_first_week weight_lost_first_week : ℕ) : ℕ :=
  weight_after_first_week + weight_lost_first_week

theorem jessie_weight_before_jogging 
  (weight_after_first_week : ℕ) 
  (weight_lost_first_week : ℕ) 
  (h1 : weight_after_first_week = 36)
  (h2 : weight_lost_first_week = 56) : 
  weight_before_jogging weight_after_first_week weight_lost_first_week = 92 := by
  sorry

end NUMINAMATH_CALUDE_jessie_weight_before_jogging_l2616_261654


namespace NUMINAMATH_CALUDE_actual_car_body_mass_l2616_261671

/-- Represents the scale factor between the model and the actual car body. -/
def scaleFactor : ℝ := 10

/-- Represents the mass of the model car body in kilograms. -/
def modelMass : ℝ := 1.5

/-- Calculates the mass of the actual car body given the scale factor and model mass. -/
def actualMass (s : ℝ) (m : ℝ) : ℝ := s^3 * m

/-- Theorem stating that the mass of the actual car body is 1500 kg. -/
theorem actual_car_body_mass :
  actualMass scaleFactor modelMass = 1500 := by
  sorry

end NUMINAMATH_CALUDE_actual_car_body_mass_l2616_261671


namespace NUMINAMATH_CALUDE_students_getting_B_l2616_261663

theorem students_getting_B (grade_A : ℚ) (grade_C : ℚ) (grade_D : ℚ) (grade_F : ℚ) (passing_grade : ℚ) :
  grade_A = 1/4 →
  grade_C = 1/8 →
  grade_D = 1/12 →
  grade_F = 1/24 →
  passing_grade = 0.875 →
  grade_A + grade_C + grade_D + grade_F + 3/8 = passing_grade :=
by sorry

end NUMINAMATH_CALUDE_students_getting_B_l2616_261663


namespace NUMINAMATH_CALUDE_f_inequality_l2616_261638

-- Define f as a differentiable function on ℝ
variable (f : ℝ → ℝ)

-- State the condition that f'(x) - f(x) < 0 for all x ∈ ℝ
variable (h : ∀ x : ℝ, (deriv f) x - f x < 0)

-- Define e as the mathematical constant e
noncomputable def e : ℝ := Real.exp 1

-- State the theorem to be proved
theorem f_inequality : e * f 2015 > f 2016 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_l2616_261638


namespace NUMINAMATH_CALUDE_expression_evaluation_l2616_261675

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 - 1
  (x + 3) * (x - 3) - x * (x - 2) = 2 * Real.sqrt 2 - 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2616_261675


namespace NUMINAMATH_CALUDE_cards_in_hospital_l2616_261669

/-- Proves that the number of get well cards Mariela received while in the hospital is 403 -/
theorem cards_in_hospital (total_cards : ℕ) (cards_after_home : ℕ) 
  (h1 : total_cards = 690) 
  (h2 : cards_after_home = 287) : 
  total_cards - cards_after_home = 403 := by
  sorry

end NUMINAMATH_CALUDE_cards_in_hospital_l2616_261669


namespace NUMINAMATH_CALUDE_nonCoplanarChoices_eq_141_l2616_261686

/-- The number of ways to choose 4 non-coplanar points from the vertices and midpoints of a tetrahedron -/
def nonCoplanarChoices : ℕ :=
  Nat.choose 10 4 - (4 * Nat.choose 6 4 + 6 + 3)

/-- Theorem stating that the number of ways to choose 4 non-coplanar points
    from the vertices and midpoints of a tetrahedron is 141 -/
theorem nonCoplanarChoices_eq_141 : nonCoplanarChoices = 141 := by
  sorry

end NUMINAMATH_CALUDE_nonCoplanarChoices_eq_141_l2616_261686


namespace NUMINAMATH_CALUDE_hillarys_craft_price_l2616_261631

/-- Proves that the price of each craft is $12 given the conditions of Hillary's sales and deposits -/
theorem hillarys_craft_price :
  ∀ (price : ℕ),
  (3 * price + 7 = 18 + 25) →
  price = 12 := by
sorry

end NUMINAMATH_CALUDE_hillarys_craft_price_l2616_261631


namespace NUMINAMATH_CALUDE_cereal_original_price_l2616_261646

def initial_money : ℝ := 60
def celery_price : ℝ := 5
def bread_price : ℝ := 8
def milk_original_price : ℝ := 10
def milk_discount : ℝ := 0.1
def potato_price : ℝ := 1
def potato_quantity : ℕ := 6
def money_left : ℝ := 26
def cereal_discount : ℝ := 0.5

theorem cereal_original_price :
  let milk_price := milk_original_price * (1 - milk_discount)
  let potato_total := potato_price * potato_quantity
  let spent_on_known_items := celery_price + bread_price + milk_price + potato_total
  let total_spent := initial_money - money_left
  let cereal_discounted_price := total_spent - spent_on_known_items
  cereal_discounted_price / (1 - cereal_discount) = 12 := by sorry

end NUMINAMATH_CALUDE_cereal_original_price_l2616_261646


namespace NUMINAMATH_CALUDE_two_tap_system_solution_l2616_261601

/-- Represents the time it takes for a tap to fill a tank -/
structure TapTime where
  minutes : ℝ
  positive : minutes > 0

/-- Represents a system of two taps filling a tank -/
structure TwoTapSystem where
  tapA : TapTime
  tapB : TapTime
  timeDifference : tapA.minutes = tapB.minutes + 22
  combinedTime : (1 / tapA.minutes + 1 / tapB.minutes) * 60 = 1

theorem two_tap_system_solution (system : TwoTapSystem) :
  system.tapB.minutes = 110 ∧ system.tapA.minutes = 132 := by
  sorry

end NUMINAMATH_CALUDE_two_tap_system_solution_l2616_261601


namespace NUMINAMATH_CALUDE_equipment_marked_price_marked_price_approx_58_82_l2616_261695

/-- The marked price of equipment given specific buying and selling conditions --/
theorem equipment_marked_price (original_price : ℝ) (buying_discount : ℝ) 
  (desired_gain : ℝ) (selling_discount : ℝ) : ℝ :=
  let cost_price := original_price * (1 - buying_discount)
  let selling_price := cost_price * (1 + desired_gain)
  selling_price / (1 - selling_discount)

/-- The marked price of equipment is approximately 58.82 given the specific conditions --/
theorem marked_price_approx_58_82 : 
  ∃ ε > 0, |equipment_marked_price 50 0.2 0.25 0.15 - 58.82| < ε :=
sorry

end NUMINAMATH_CALUDE_equipment_marked_price_marked_price_approx_58_82_l2616_261695


namespace NUMINAMATH_CALUDE_triangle_area_l2616_261668

theorem triangle_area (base height : Real) (h1 : base = 8.4) (h2 : height = 5.8) :
  (base * height) / 2 = 24.36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2616_261668


namespace NUMINAMATH_CALUDE_meeting_gender_ratio_l2616_261681

theorem meeting_gender_ratio (total_population : ℕ) (females_attending : ℕ) : 
  total_population = 300 →
  females_attending = 50 →
  (total_population / 2 - females_attending) / females_attending = 2 := by
  sorry

end NUMINAMATH_CALUDE_meeting_gender_ratio_l2616_261681


namespace NUMINAMATH_CALUDE_haley_trees_died_l2616_261642

/-- The number of trees that died in a typhoon given the initial number of trees and the number of trees remaining. -/
def trees_died (initial_trees remaining_trees : ℕ) : ℕ :=
  initial_trees - remaining_trees

/-- Proof that 5 trees died in the typhoon given the conditions in Haley's problem. -/
theorem haley_trees_died : trees_died 17 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_haley_trees_died_l2616_261642


namespace NUMINAMATH_CALUDE_borrowed_amount_l2616_261624

theorem borrowed_amount (X : ℝ) : 
  (X + 0.1 * X = 110) → X = 100 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_amount_l2616_261624


namespace NUMINAMATH_CALUDE_car_journey_downhill_distance_l2616_261689

/-- Proves that a car traveling 100 km uphill at 30 km/hr and an unknown distance downhill
    at 60 km/hr, with an average speed of 36 km/hr for the entire journey,
    travels 50 km downhill. -/
theorem car_journey_downhill_distance
  (uphill_speed : ℝ) (downhill_speed : ℝ) (uphill_distance : ℝ) (average_speed : ℝ)
  (h1 : uphill_speed = 30)
  (h2 : downhill_speed = 60)
  (h3 : uphill_distance = 100)
  (h4 : average_speed = 36)
  : ∃ (downhill_distance : ℝ),
    (uphill_distance + downhill_distance) / ((uphill_distance / uphill_speed) + (downhill_distance / downhill_speed)) = average_speed
    ∧ downhill_distance = 50 :=
by sorry

end NUMINAMATH_CALUDE_car_journey_downhill_distance_l2616_261689


namespace NUMINAMATH_CALUDE_jane_ate_12_swirls_l2616_261619

/-- Given a number of cinnamon swirls and people, calculate how many swirls each person ate. -/
def swirls_per_person (total_swirls : ℕ) (num_people : ℕ) : ℕ :=
  total_swirls / num_people

/-- Theorem stating that Jane ate 12 cinnamon swirls. -/
theorem jane_ate_12_swirls (total_swirls : ℕ) (num_people : ℕ) 
  (h1 : total_swirls = 120) 
  (h2 : num_people = 10) :
  swirls_per_person total_swirls num_people = 12 := by
  sorry

#eval swirls_per_person 120 10

end NUMINAMATH_CALUDE_jane_ate_12_swirls_l2616_261619


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_range_l2616_261643

/-- The slope of a chord of the ellipse x^2 + y^2/4 = 1 whose midpoint lies on the line segment
    between (1/2, 1/2) and (1/2, 1) is between -4 and -2. -/
theorem ellipse_chord_slope_range :
  ∀ (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ),
  (x₁^2 + y₁^2/4 = 1) →  -- P(x₁, y₁) is on the ellipse
  (x₂^2 + y₂^2/4 = 1) →  -- Q(x₂, y₂) is on the ellipse
  (x₀ = (x₁ + x₂)/2) →   -- x-coordinate of midpoint
  (y₀ = (y₁ + y₂)/2) →   -- y-coordinate of midpoint
  (x₀ = 1/2) →           -- midpoint x-coordinate is on AB
  (1/2 ≤ y₀ ∧ y₀ ≤ 1) →  -- midpoint y-coordinate is between A and B
  (-4 ≤ -(y₁ - y₂)/(x₁ - x₂) ∧ -(y₁ - y₂)/(x₁ - x₂) ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_range_l2616_261643


namespace NUMINAMATH_CALUDE_exponential_function_coefficient_l2616_261608

def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ (b c : ℝ), b > 0 ∧ b ≠ 1 ∧ (∀ x, f x = c * b^x)

theorem exponential_function_coefficient (a : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  is_exponential_function (λ x => (a^2 - 3*a + 3) * a^x) →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_exponential_function_coefficient_l2616_261608


namespace NUMINAMATH_CALUDE_area_difference_sheets_l2616_261647

/-- The difference in combined area (front and back) between a square sheet of paper
    with side length 11 inches and a rectangular sheet of paper measuring 5.5 inches
    by 11 inches is equal to 121 square inches. -/
theorem area_difference_sheets : 
  let square_sheet_side : ℝ := 11
  let rect_sheet_length : ℝ := 11
  let rect_sheet_width : ℝ := 5.5
  let square_sheet_area : ℝ := 2 * square_sheet_side * square_sheet_side
  let rect_sheet_area : ℝ := 2 * rect_sheet_length * rect_sheet_width
  square_sheet_area - rect_sheet_area = 121 := by
sorry

end NUMINAMATH_CALUDE_area_difference_sheets_l2616_261647


namespace NUMINAMATH_CALUDE_dog_bone_collection_l2616_261635

theorem dog_bone_collection (initial_bones : ℝ) (found_multiplier : ℝ) (given_away : ℝ) (return_fraction : ℝ) : 
  initial_bones = 425.5 →
  found_multiplier = 3.5 →
  given_away = 127.25 →
  return_fraction = 1/4 →
  let total_after_finding := initial_bones + found_multiplier * initial_bones
  let total_after_giving := total_after_finding - given_away
  let returned_bones := return_fraction * given_away
  let final_total := total_after_giving + returned_bones
  final_total = 1819.3125 := by
sorry

end NUMINAMATH_CALUDE_dog_bone_collection_l2616_261635


namespace NUMINAMATH_CALUDE_total_coins_after_addition_initial_ratio_final_ratio_l2616_261659

/-- Represents a coin collection with gold and silver coins -/
structure CoinCollection where
  gold : ℕ
  silver : ℕ

/-- The initial state of the coin collection -/
def initial_collection : CoinCollection :=
  { gold := 30, silver := 90 }

/-- The final state of the coin collection after adding 15 gold coins -/
def final_collection : CoinCollection :=
  { gold := initial_collection.gold + 15, silver := initial_collection.silver }

/-- Theorem stating the total number of coins after the addition -/
theorem total_coins_after_addition :
  final_collection.gold + final_collection.silver = 135 := by
  sorry

/-- Theorem for the initial ratio of gold to silver coins -/
theorem initial_ratio :
  initial_collection.gold * 3 = initial_collection.silver := by
  sorry

/-- Theorem for the final ratio of gold to silver coins -/
theorem final_ratio :
  final_collection.gold * 2 = final_collection.silver := by
  sorry

end NUMINAMATH_CALUDE_total_coins_after_addition_initial_ratio_final_ratio_l2616_261659


namespace NUMINAMATH_CALUDE_irrational_product_l2616_261639

-- Define the property of being irrational
def IsIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Define the property of being rational
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem irrational_product (x : ℝ) : 
  IsIrrational x → 
  IsRational ((x - 2) * (x + 6)) → 
  IsIrrational ((x + 2) * (x - 6)) := by
  sorry

end NUMINAMATH_CALUDE_irrational_product_l2616_261639


namespace NUMINAMATH_CALUDE_xy_value_l2616_261634

theorem xy_value (x y : ℝ) 
  (h : (x / (1 - Complex.I)) - (y / (1 - 2 * Complex.I)) = (5 : ℝ) / (1 - 3 * Complex.I)) : 
  x * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2616_261634


namespace NUMINAMATH_CALUDE_power_of_power_l2616_261605

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2616_261605


namespace NUMINAMATH_CALUDE_equal_area_intersection_sum_l2616_261645

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (a b c d : Point) : ℚ :=
  (1/2) * abs (a.x * b.y + b.x * c.y + c.x * d.y + d.x * a.y
             - (b.x * a.y + c.x * b.y + d.x * c.y + a.x * d.y))

/-- Checks if a fraction is in its lowest terms -/
def isLowestTerms (p q : ℤ) : Prop :=
  ∀ (d : ℤ), d > 1 → ¬(d ∣ p ∧ d ∣ q)

/-- Main theorem -/
theorem equal_area_intersection_sum (p q r s : ℤ) :
  let a := Point.mk 0 0
  let b := Point.mk 1 3
  let c := Point.mk 4 4
  let d := Point.mk 5 0
  let intersectionPoint := Point.mk (p/q) (r/s)
  quadrilateralArea a b intersectionPoint d = quadrilateralArea b c d intersectionPoint →
  isLowestTerms p q →
  isLowestTerms r s →
  p + q + r + s = 200 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_intersection_sum_l2616_261645


namespace NUMINAMATH_CALUDE_sqrt_neg_four_squared_l2616_261660

theorem sqrt_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_four_squared_l2616_261660


namespace NUMINAMATH_CALUDE_hourly_rate_approximation_l2616_261623

/-- Calculates the hourly rate based on given salary information and work schedule. -/
def calculate_hourly_rate (base_salary : ℚ) (commission_rate : ℚ) (total_sales : ℚ) 
  (performance_bonus : ℚ) (deductions : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) 
  (weeks_per_month : ℕ) : ℚ :=
  let total_earnings := base_salary + (commission_rate * total_sales) + performance_bonus - deductions
  let total_hours := hours_per_day * days_per_week * weeks_per_month
  total_earnings / total_hours

/-- Proves that the hourly rate is approximately $3.86 given the specified conditions. -/
theorem hourly_rate_approximation :
  let base_salary : ℚ := 576
  let commission_rate : ℚ := 3 / 100
  let total_sales : ℚ := 4000
  let performance_bonus : ℚ := 75
  let deductions : ℚ := 30
  let hours_per_day : ℕ := 8
  let days_per_week : ℕ := 6
  let weeks_per_month : ℕ := 4
  let hourly_rate := calculate_hourly_rate base_salary commission_rate total_sales 
    performance_bonus deductions hours_per_day days_per_week weeks_per_month
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |hourly_rate - 386/100| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_hourly_rate_approximation_l2616_261623


namespace NUMINAMATH_CALUDE_monthly_growth_rate_price_reduction_l2616_261636

-- Define the sales data
def august_sales : ℕ := 50000
def october_sales : ℕ := 72000

-- Define the pricing and sales data
def cost_price : ℚ := 40
def original_price : ℚ := 80
def initial_daily_sales : ℕ := 20
def sales_increase_rate : ℚ := 4  -- 2 units per $0.5 decrease = 4 units per $1 decrease
def desired_daily_profit : ℚ := 1400

-- Part 1: Monthly average growth rate
theorem monthly_growth_rate :
  ∃ (x : ℝ), x ≥ 0 ∧ x ≤ 1 ∧ 
  (↑august_sales * (1 + x)^2 : ℝ) = october_sales ∧
  x = 0.2 := by sorry

-- Part 2: Price reduction for promotion
theorem price_reduction :
  ∃ (y : ℚ), y > 0 ∧ y < original_price - cost_price ∧
  (original_price - y - cost_price) * (initial_daily_sales + sales_increase_rate * y) = desired_daily_profit ∧
  y = 30 := by sorry

end NUMINAMATH_CALUDE_monthly_growth_rate_price_reduction_l2616_261636


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2616_261648

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a → a 2 = 4 → a 6 = 16 → a 3 + a 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2616_261648


namespace NUMINAMATH_CALUDE_fraction_inequality_l2616_261612

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) : a / b < a / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2616_261612


namespace NUMINAMATH_CALUDE_omega_sum_l2616_261652

theorem omega_sum (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68 + ω^72 + ω^76 + ω^80 = -ω^2 := by
  sorry

end NUMINAMATH_CALUDE_omega_sum_l2616_261652


namespace NUMINAMATH_CALUDE_remaining_note_denomination_l2616_261657

theorem remaining_note_denomination 
  (total_amount : ℕ) 
  (total_notes : ℕ) 
  (fifty_notes : ℕ) 
  (fifty_denomination : ℕ) :
  total_amount = 10350 →
  total_notes = 90 →
  fifty_notes = 77 →
  fifty_denomination = 50 →
  ∃ (remaining_denomination : ℕ),
    remaining_denomination * (total_notes - fifty_notes) = 
      total_amount - (fifty_notes * fifty_denomination) ∧
    remaining_denomination = 500 := by
  sorry

end NUMINAMATH_CALUDE_remaining_note_denomination_l2616_261657


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2616_261684

-- Define the function f
def f (x a : ℝ) := |x - a| - |2 * x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x 2 + 3 ≥ 0} = {x : ℝ | -4 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x ∈ Set.Icc 1 3, f x a ≤ 3} = Set.Icc (-3) 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2616_261684


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2616_261630

def U : Set ℕ := {x | 0 ≤ x ∧ x < 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {x ∈ U | x^2 + 4 = 5*x}

theorem complement_union_theorem : 
  (U \ A) ∪ (U \ B) = {0, 2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2616_261630


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2616_261641

theorem complex_magnitude_problem (z : ℂ) (h : (4 + 3*Complex.I) * (z - 3*Complex.I) = 25) : 
  Complex.abs z = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2616_261641


namespace NUMINAMATH_CALUDE_triangle_midpoint_dot_product_l2616_261629

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  AB = 10 ∧ AC = 6 ∧ BC = 8

-- Define the midpoint
def Midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define the dot product
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem triangle_midpoint_dot_product 
  (A B C M : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : Midpoint M A B) : 
  DotProduct (M.1 - C.1, M.2 - C.2) (A.1 - C.1, A.2 - C.2) + 
  DotProduct (M.1 - C.1, M.2 - C.2) (B.1 - C.1, B.2 - C.2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_midpoint_dot_product_l2616_261629


namespace NUMINAMATH_CALUDE_first_discount_percentage_l2616_261679

/-- Proves that the first discount percentage is 10% given the conditions of the problem -/
theorem first_discount_percentage 
  (list_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) :
  list_price = 70 →
  final_price = 59.85 →
  second_discount = 0.05 →
  ∃ (first_discount : ℝ),
    final_price = list_price * (1 - first_discount) * (1 - second_discount) ∧
    first_discount = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l2616_261679


namespace NUMINAMATH_CALUDE_abc_sum_l2616_261600

theorem abc_sum (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a * b = 2 * (a + b)) (hbc : b * c = 3 * (b + c)) (hca : c * a = 4 * (c + a)) :
  a + b + c = 1128 / 35 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_l2616_261600


namespace NUMINAMATH_CALUDE_point_B_coordinate_l2616_261691

def point_A : ℝ := -1

theorem point_B_coordinate (point_B : ℝ) (h : |point_B - point_A| = 3) :
  point_B = 2 ∨ point_B = -4 := by
  sorry

end NUMINAMATH_CALUDE_point_B_coordinate_l2616_261691


namespace NUMINAMATH_CALUDE_horner_method_v2_l2616_261644

/-- Horner's method for polynomial evaluation -/
def horner_v2 (x : ℤ) : ℤ := x^2 + 6

/-- The polynomial f(x) = x^6 + 6x^4 + 9x^2 + 208 -/
def f (x : ℤ) : ℤ := x^6 + 6*x^4 + 9*x^2 + 208

theorem horner_method_v2 :
  horner_v2 (-4) = 22 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v2_l2616_261644


namespace NUMINAMATH_CALUDE_shirley_eggs_l2616_261688

/-- The number of eggs Shirley starts with -/
def initial_eggs : ℕ := 98

/-- The number of eggs Shirley buys -/
def bought_eggs : ℕ := 8

/-- The total number of eggs Shirley ends with -/
def total_eggs : ℕ := initial_eggs + bought_eggs

theorem shirley_eggs : total_eggs = 106 := by
  sorry

end NUMINAMATH_CALUDE_shirley_eggs_l2616_261688
