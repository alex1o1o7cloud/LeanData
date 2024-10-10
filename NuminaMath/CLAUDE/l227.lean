import Mathlib

namespace smallest_dual_palindrome_seventeen_is_dual_palindrome_smallest_dual_palindrome_is_17_l227_22710

/-- Checks if a number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from base 10 to another base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 15 → 
  (isPalindrome n 2 ∧ isPalindrome n 4) → 
  n ≥ 17 := by sorry

theorem seventeen_is_dual_palindrome : 
  isPalindrome 17 2 ∧ isPalindrome 17 4 := by sorry

theorem smallest_dual_palindrome_is_17 : 
  ∀ n : ℕ, n > 15 → 
  (isPalindrome n 2 ∧ isPalindrome n 4) → 
  n = 17 := by sorry

end smallest_dual_palindrome_seventeen_is_dual_palindrome_smallest_dual_palindrome_is_17_l227_22710


namespace quadratic_equation_roots_l227_22727

theorem quadratic_equation_roots (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 9 ∧ x * y = 20) →
  m + n = 87 := by
sorry

end quadratic_equation_roots_l227_22727


namespace solutions_eq1_solutions_eq2_l227_22762

-- First equation
theorem solutions_eq1 (x : ℝ) : x^2 - 2*x - 3 = 0 ↔ x = 3 ∨ x = -1 := by sorry

-- Second equation
theorem solutions_eq2 (x : ℝ) : x*(x-2) + x - 2 = 0 ↔ x = -1 ∨ x = 2 := by sorry

end solutions_eq1_solutions_eq2_l227_22762


namespace road_travel_cost_l227_22731

/-- Calculates the cost of traveling two intersecting roads on a rectangular lawn. -/
theorem road_travel_cost
  (lawn_length lawn_width road_width : ℕ)
  (cost_per_sqm : ℚ)
  (h1 : lawn_length = 80)
  (h2 : lawn_width = 60)
  (h3 : road_width = 10)
  (h4 : cost_per_sqm = 4) :
  (((lawn_length * road_width + lawn_width * road_width - road_width * road_width) : ℚ) * cost_per_sqm) = 5200 := by
  sorry

end road_travel_cost_l227_22731


namespace joshua_friends_count_l227_22775

/-- Given that Joshua gave 40 Skittles to each friend and the total number of Skittles given is 200,
    prove that the number of friends Joshua gave Skittles to is 5. -/
theorem joshua_friends_count (skittles_per_friend : ℕ) (total_skittles : ℕ) 
    (h1 : skittles_per_friend = 40) 
    (h2 : total_skittles = 200) : 
  total_skittles / skittles_per_friend = 5 := by
sorry

end joshua_friends_count_l227_22775


namespace quiz_probabilities_l227_22712

/-- Represents the quiz with multiple-choice and true/false questions -/
structure Quiz where
  total_questions : ℕ
  multiple_choice : ℕ
  true_false : ℕ

/-- Calculates the probability of A drawing a multiple-choice question and B drawing a true/false question -/
def prob_a_multiple_b_true_false (q : Quiz) : ℚ :=
  (q.multiple_choice * q.true_false) / (q.total_questions * (q.total_questions - 1))

/-- Calculates the probability of at least one of A or B drawing a multiple-choice question -/
def prob_at_least_one_multiple (q : Quiz) : ℚ :=
  1 - (q.true_false * (q.true_false - 1)) / (q.total_questions * (q.total_questions - 1))

theorem quiz_probabilities (q : Quiz) 
  (h1 : q.total_questions = 10)
  (h2 : q.multiple_choice = 6)
  (h3 : q.true_false = 4) :
  prob_a_multiple_b_true_false q = 4 / 15 ∧ 
  prob_at_least_one_multiple q = 13 / 15 := by
  sorry


end quiz_probabilities_l227_22712


namespace roots_of_cubic_equation_l227_22753

variable (a b c d α β : ℝ)

def original_quadratic (x : ℝ) : ℝ := x^2 - (a + d)*x + (a*d - b*c)

def new_quadratic (x : ℝ) : ℝ := x^2 - (a^3 + d^3 + 3*a*b*c + 3*b*c*d)*x + (a*d - b*c)^3

theorem roots_of_cubic_equation 
  (h1 : original_quadratic α = 0)
  (h2 : original_quadratic β = 0) :
  new_quadratic (α^3) = 0 ∧ new_quadratic (β^3) = 0 := by
  sorry

end roots_of_cubic_equation_l227_22753


namespace collector_problem_l227_22735

/-- The number of items in the collection --/
def n : ℕ := 10

/-- The probability of finding each item --/
def p : ℝ := 0.1

/-- The probability of having exactly k items missing in the second collection
    when the first collection is complete --/
def prob_missing (k : ℕ) : ℝ := sorry

theorem collector_problem :
  (prob_missing 1 = prob_missing 2) ∧
  (∀ k ∈ Finset.range 9, prob_missing (k + 2) > prob_missing (k + 3)) :=
sorry

end collector_problem_l227_22735


namespace stating_equal_cost_guests_proof_l227_22717

/-- The number of guests for which the costs of renting Caesar's and Venus Hall are equal -/
def equal_cost_guests : ℕ := 60

/-- The room rental cost for Caesar's -/
def caesars_rental : ℕ := 800

/-- The per-meal cost for Caesar's -/
def caesars_per_meal : ℕ := 30

/-- The room rental cost for Venus Hall -/
def venus_rental : ℕ := 500

/-- The per-meal cost for Venus Hall -/
def venus_per_meal : ℕ := 35

/-- 
Theorem stating that the number of guests for which the costs of renting 
Caesar's and Venus Hall are equal is 60, given the rental and per-meal costs for each venue.
-/
theorem equal_cost_guests_proof :
  caesars_rental + caesars_per_meal * equal_cost_guests = 
  venus_rental + venus_per_meal * equal_cost_guests :=
by sorry

end stating_equal_cost_guests_proof_l227_22717


namespace largest_odd_integer_in_range_l227_22720

theorem largest_odd_integer_in_range : 
  ∃ (x : ℤ), (x % 2 = 1) ∧ (1/4 < x/6) ∧ (x/6 < 7/9) ∧
  ∀ (y : ℤ), (y % 2 = 1) ∧ (1/4 < y/6) ∧ (y/6 < 7/9) → y ≤ x :=
by
  -- The proof goes here
  sorry

end largest_odd_integer_in_range_l227_22720


namespace consecutive_integers_sum_l227_22780

theorem consecutive_integers_sum (x y : ℤ) : 
  (y = x + 1) → (x < Real.sqrt 5 + 1) → (Real.sqrt 5 + 1 < y) → x + y = 7 := by
  sorry

end consecutive_integers_sum_l227_22780


namespace selling_price_ratio_l227_22736

theorem selling_price_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end selling_price_ratio_l227_22736


namespace smaller_root_of_quadratic_l227_22760

theorem smaller_root_of_quadratic (x : ℚ) : 
  (x - 2/3) * (x - 5/6) + (x - 2/3) * (x - 2/3) - 1 = 0 →
  x = -1/12 ∨ x = 4/3 ∧ 
  -1/12 < 4/3 :=
sorry

end smaller_root_of_quadratic_l227_22760


namespace sum_of_circle_circumferences_l227_22771

/-- The sum of circumferences of an infinite series of circles inscribed in an equilateral triangle -/
theorem sum_of_circle_circumferences (r : ℝ) (h : r = 1) : 
  (2 * π * r) + (3 * (2 * π * r * (∑' n, (1/3)^n))) = 5 * π :=
sorry

end sum_of_circle_circumferences_l227_22771


namespace difference_of_squares_72_48_l227_22726

theorem difference_of_squares_72_48 : 72^2 - 48^2 = 2880 := by
  sorry

end difference_of_squares_72_48_l227_22726


namespace intersection_equals_open_interval_l227_22798

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- Define the open interval (-1, 1)
def openInterval : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Theorem statement
theorem intersection_equals_open_interval : M ∩ N = openInterval := by
  sorry

end intersection_equals_open_interval_l227_22798


namespace smallest_staircase_steps_l227_22766

theorem smallest_staircase_steps : ∃ (n : ℕ), 
  n > 20 ∧ 
  n % 6 = 5 ∧ 
  n % 7 = 1 ∧ 
  (∀ m : ℕ, m > 20 ∧ m % 6 = 5 ∧ m % 7 = 1 → m ≥ n) ∧
  n = 29 := by
sorry

end smallest_staircase_steps_l227_22766


namespace inequality_solution_set_l227_22796

theorem inequality_solution_set : 
  {x : ℤ | (x + 3)^3 ≤ 8} = {x : ℤ | x ≤ -1} := by sorry

end inequality_solution_set_l227_22796


namespace average_blanket_price_l227_22724

/-- The average price of blankets given specific purchase conditions -/
theorem average_blanket_price : 
  let blanket_group1 := (3, 100)  -- (quantity, price)
  let blanket_group2 := (5, 150)
  let blanket_group3 := (2, 275)  -- 550 / 2 = 275
  let total_blankets := blanket_group1.1 + blanket_group2.1 + blanket_group3.1
  let total_cost := blanket_group1.1 * blanket_group1.2 + 
                    blanket_group2.1 * blanket_group2.2 + 
                    blanket_group3.1 * blanket_group3.2
  (total_cost / total_blankets : ℚ) = 160 := by
  sorry

end average_blanket_price_l227_22724


namespace granola_bars_eaten_by_parents_l227_22761

theorem granola_bars_eaten_by_parents (total : ℕ) (children : ℕ) (per_child : ℕ) 
  (h1 : total = 200) 
  (h2 : children = 6) 
  (h3 : per_child = 20) : 
  total - (children * per_child) = 80 :=
by sorry

end granola_bars_eaten_by_parents_l227_22761


namespace always_greater_than_m_l227_22745

theorem always_greater_than_m (m : ℚ) : m + 2 > m := by
  sorry

end always_greater_than_m_l227_22745


namespace problem_solution_l227_22788

theorem problem_solution (x y : ℝ) : 
  x / y = 6 / 3 → y = 27 → x = 54 := by sorry

end problem_solution_l227_22788


namespace percentage_increase_l227_22707

theorem percentage_increase (N : ℝ) (P : ℝ) : 
  N = 40 →
  N + (P / 100) * N - (N - (30 / 100) * N) = 22 →
  P = 25 := by
sorry

end percentage_increase_l227_22707


namespace expected_value_S_squared_l227_22708

/-- ω is a primitive 2018th root of unity -/
def ω : ℂ :=
  sorry

/-- The set of complex numbers from which subsets are chosen -/
def complexSet : Finset ℂ :=
  sorry

/-- S is the sum of elements in a randomly chosen subset of complexSet -/
def S : Finset ℂ → ℂ :=
  sorry

/-- The expected value of |S|² -/
def expectedValueS : ℝ :=
  sorry

theorem expected_value_S_squared :
  expectedValueS = 1009 / 2 :=
sorry

end expected_value_S_squared_l227_22708


namespace angle_Z_measure_l227_22769

-- Define the triangle and its angles
def Triangle (X Y W Z : ℝ) : Prop :=
  -- Conditions
  X = 34 ∧ Y = 53 ∧ W = 43 ∧
  -- Additional properties of a triangle
  X > 0 ∧ Y > 0 ∧ W > 0 ∧ Z > 0 ∧
  -- Sum of angles in the larger triangle is 180°
  X + Y + W + Z = 180

-- Theorem statement
theorem angle_Z_measure (X Y W Z : ℝ) (h : Triangle X Y W Z) : Z = 130 := by
  sorry

end angle_Z_measure_l227_22769


namespace pete_total_books_matt_year2_increase_l227_22709

/-- The number of books Matt read in the first year -/
def matt_year1 : ℕ := 50

/-- The number of books Matt read in the second year -/
def matt_year2 : ℕ := 75

/-- The number of books Pete read in the first year -/
def pete_year1 : ℕ := 2 * matt_year1

/-- The number of books Pete read in the second year -/
def pete_year2 : ℕ := 2 * pete_year1

/-- Theorem stating that Pete read 300 books across both years -/
theorem pete_total_books : pete_year1 + pete_year2 = 300 := by
  sorry

/-- Verification that Matt's second year reading increased by 50% -/
theorem matt_year2_increase : matt_year2 = (3 * matt_year1) / 2 := by
  sorry

end pete_total_books_matt_year2_increase_l227_22709


namespace trapezoid_bisector_length_l227_22749

/-- 
Given a trapezoid with parallel sides of length a and c,
the length of a segment parallel to these sides that bisects the trapezoid's area
is √((a² + c²) / 2).
-/
theorem trapezoid_bisector_length (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  ∃ x : ℝ, x > 0 ∧ x^2 = (a^2 + c^2) / 2 ∧
  (∀ m : ℝ, m > 0 → (a + c) * m / 2 = (x + c) * (2 * m / (c + x)) / 2 + (x + a) * (2 * m / (a + x)) / 2) :=
by sorry

end trapezoid_bisector_length_l227_22749


namespace pop_survey_result_l227_22701

/-- Given a survey of 600 people where the central angle for "Pop" is 270°
    (to the nearest whole degree), prove that 450 people chose "Pop". -/
theorem pop_survey_result (total : ℕ) (angle : ℕ) (h_total : total = 600) (h_angle : angle = 270) :
  ∃ (pop : ℕ), pop = 450 ∧ 
  (pop : ℝ) / total * 360 ≥ angle - 0.5 ∧
  (pop : ℝ) / total * 360 < angle + 0.5 :=
by sorry

end pop_survey_result_l227_22701


namespace intersection_condition_l227_22795

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3 * a^2 * x + 1

/-- The statement that f(x) intersects y = 3 at only one point -/
def intersects_once (a : ℝ) : Prop :=
  ∃! x : ℝ, f a x = 3

/-- The main theorem to be proved -/
theorem intersection_condition :
  ∀ a : ℝ, intersects_once a ↔ -1 < a ∧ a < 1 := by sorry

end intersection_condition_l227_22795


namespace unique_zero_in_interval_l227_22774

def f (x : ℝ) := -x^2 + 4*x - 4

theorem unique_zero_in_interval :
  ∃! x : ℝ, x ∈ Set.Icc 1 3 ∧ f x = 0 :=
sorry

end unique_zero_in_interval_l227_22774


namespace pyramid_blocks_l227_22728

/-- Calculates the number of blocks in a pyramid layer given the number in the layer above -/
def blocks_in_layer (blocks_above : ℕ) : ℕ := 3 * blocks_above

/-- Calculates the total number of blocks in a pyramid with the given number of layers -/
def total_blocks (layers : ℕ) : ℕ :=
  match layers with
  | 0 => 0
  | n + 1 => (blocks_in_layer^[n] 1) + total_blocks n

theorem pyramid_blocks :
  total_blocks 4 = 40 :=
by sorry

end pyramid_blocks_l227_22728


namespace symmetry_condition_l227_22706

/-- Given a curve y = (2px + q) / (rx - 2s) where p, q, r, s are nonzero real numbers,
    if the line y = x is an axis of symmetry for this curve, then r - 2s = 0. -/
theorem symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (2*p*x + q) / (r*x - 2*s) ↔ x = (2*p*y + q) / (r*y - 2*s)) →
  r - 2*s = 0 := by
  sorry

end symmetry_condition_l227_22706


namespace sum_of_digits_N_l227_22784

/-- The smallest positive integer whose digits have a product of 1728 -/
def N : ℕ := sorry

/-- The product of the digits of N is 1728 -/
axiom N_digit_product : (N.digits 10).prod = 1728

/-- N is the smallest such positive integer -/
axiom N_smallest (m : ℕ) : m > 0 → (m.digits 10).prod = 1728 → m ≥ N

/-- The sum of the digits of N is 28 -/
theorem sum_of_digits_N : (N.digits 10).sum = 28 := by sorry

end sum_of_digits_N_l227_22784


namespace x_minus_2y_bounds_l227_22793

theorem x_minus_2y_bounds (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) :
  0 ≤ x - 2*y ∧ x - 2*y ≤ 10 := by
  sorry

end x_minus_2y_bounds_l227_22793


namespace property_width_l227_22791

/-- Proves that the width of a rectangular property is 1000 feet given specific conditions -/
theorem property_width (property_length : ℝ) (garden_area : ℝ) 
  (h1 : property_length = 2250)
  (h2 : garden_area = 28125)
  (h3 : ∃ (property_width : ℝ), 
    garden_area = (property_width / 8) * (property_length / 10)) :
  ∃ (property_width : ℝ), property_width = 1000 := by
  sorry

end property_width_l227_22791


namespace light_distance_scientific_notation_l227_22776

/-- The speed of light in kilometers per second -/
def speed_of_light : ℝ := 300000

/-- The time in seconds -/
def time : ℝ := 10

/-- The distance traveled by light in the given time -/
def distance : ℝ := speed_of_light * time

/-- The exponent in the scientific notation of the distance -/
def n : ℕ := 6

theorem light_distance_scientific_notation :
  ∃ (a : ℝ), a > 0 ∧ a < 10 ∧ distance = a * (10 : ℝ) ^ n :=
sorry

end light_distance_scientific_notation_l227_22776


namespace simplify_and_ratio_l227_22725

theorem simplify_and_ratio (m : ℝ) : 
  (6*m + 12) / 6 = m + 2 ∧ (1 : ℝ) / 2 = 1 / 2 := by
  sorry

end simplify_and_ratio_l227_22725


namespace triangle_properties_l227_22737

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Area condition
  (b^2 / (3 * Real.sin B)) = (1/2) * a * c * Real.sin B →
  -- Given condition
  Real.cos A * Real.cos C = 1/6 →
  -- Prove these statements
  Real.sin A * Real.sin C = 2/3 ∧ B = π/3 := by sorry

end triangle_properties_l227_22737


namespace integral_f_equals_five_sixths_l227_22741

-- Define the piecewise function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x ≥ 0 ∧ x ≤ 1 then x^2
  else if x > 1 ∧ x ≤ 2 then 2 - x
  else 0  -- Define a value for x outside [0,2] to make f total

-- State the theorem
theorem integral_f_equals_five_sixths :
  ∫ x in (0)..(2), f x = 5/6 := by sorry

end integral_f_equals_five_sixths_l227_22741


namespace yunas_math_score_l227_22759

theorem yunas_math_score (score1 score2 : ℝ) (h1 : (score1 + score2) / 2 = 92) 
  (h2 : ∃ (score3 : ℝ), (score1 + score2 + score3) / 3 = 94) : 
  ∃ (score3 : ℝ), score3 = 98 ∧ (score1 + score2 + score3) / 3 = 94 := by
  sorry

end yunas_math_score_l227_22759


namespace smallest_multiple_l227_22742

theorem smallest_multiple (n : ℕ) : 
  (∃ k : ℕ, n = 32 * k) ∧ 
  (∃ m : ℕ, n - 6 = 97 * m) ∧
  (∀ x : ℕ, x < n → ¬((∃ k : ℕ, x = 32 * k) ∧ (∃ m : ℕ, x - 6 = 97 * m))) →
  n = 2528 := by
  sorry

end smallest_multiple_l227_22742


namespace contrapositive_truth_l227_22778

theorem contrapositive_truth : 
  (∀ x : ℝ, (x^2 < 1 → -1 < x ∧ x < 1)) ↔ 
  (∀ x : ℝ, (x ≤ -1 ∨ 1 ≤ x) → x^2 ≥ 1) :=
by sorry

end contrapositive_truth_l227_22778


namespace line_slope_point_value_l227_22756

theorem line_slope_point_value (m : ℝ) : 
  m > 0 → 
  (((m - 5) / (2 - m)) = Real.sqrt 2) → 
  m = 2 + 3 * Real.sqrt 2 := by
  sorry

end line_slope_point_value_l227_22756


namespace train_length_problem_l227_22790

/-- The length of two trains passing each other on parallel tracks --/
theorem train_length_problem (faster_speed slower_speed : ℝ) (passing_time : ℝ) : 
  faster_speed = 50 * (5/18) →
  slower_speed = 36 * (5/18) →
  passing_time = 36 →
  ∃ (train_length : ℝ), train_length = 70 ∧ 
    2 * train_length = (faster_speed - slower_speed) * passing_time :=
by sorry


end train_length_problem_l227_22790


namespace triangle_movement_path_length_l227_22757

/-- Represents the movement of a triangle inside a square -/
structure TriangleMovement where
  square_side : ℝ
  triangle_side : ℝ
  initial_rotation_radius : ℝ
  final_rotation_radius : ℝ
  initial_rotation_angle : ℝ
  final_rotation_angle : ℝ

/-- Calculates the total path traversed by vertex P -/
def total_path_length (m : TriangleMovement) : ℝ :=
  m.initial_rotation_radius * m.initial_rotation_angle +
  m.final_rotation_radius * m.final_rotation_angle

/-- The theorem to be proved -/
theorem triangle_movement_path_length :
  ∀ (m : TriangleMovement),
  m.square_side = 6 ∧
  m.triangle_side = 3 ∧
  m.initial_rotation_radius = m.triangle_side ∧
  m.final_rotation_radius = (m.square_side / 2 + m.triangle_side / 2) ∧
  m.initial_rotation_angle = Real.pi ∧
  m.final_rotation_angle = 2 * Real.pi →
  total_path_length m = 12 * Real.pi :=
by sorry

end triangle_movement_path_length_l227_22757


namespace quadratic_equation_roots_l227_22792

theorem quadratic_equation_roots (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h1 : p^2 + p*p + q = 0) (h2 : q^2 + p*q + q = 0) : p = 1 ∧ q = -2 := by
  sorry

end quadratic_equation_roots_l227_22792


namespace max_value_expression_l227_22730

theorem max_value_expression (y : ℝ) (h : y > 0) :
  (y^2 + 3 - Real.sqrt (y^4 + 9)) / y ≤ 4 * Real.sqrt 6 - 6 := by
  sorry

end max_value_expression_l227_22730


namespace parabola_intersection_theorem_l227_22703

structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  h_p_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x
  h_focus : focus = (1, 0)

structure Line where
  m : ℝ
  b : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ y = m*x + b

def intersect (C : Parabola) (l : Line) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | C.eq p.1 p.2 ∧ l.eq p.1 p.2}

def perpendicular (A B O : ℝ × ℝ) : Prop :=
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0

theorem parabola_intersection_theorem (C : Parabola) (l : Line) 
  (A B : ℝ × ℝ) (h_AB : A ∈ intersect C l ∧ B ∈ intersect C l) 
  (h_perp : perpendicular A B (0, 0)) :
  ∃ (T : ℝ × ℝ), 
    (∃ (k : ℝ), ∀ (X : ℝ × ℝ), X ∈ intersect C l → 
      (X.2 / (X.1 - 4) + X.2 / (X.1 - T.1) = k)) ∧
    T = (-4, 0) ∧ 
    k = 0 := by
  sorry

end parabola_intersection_theorem_l227_22703


namespace inscribed_rectangle_semicircle_radius_l227_22770

/-- Given a rectangle inscribed in a semi-circle with specific properties,
    prove that the radius of the semi-circle is 23.625 cm. -/
theorem inscribed_rectangle_semicircle_radius 
  (perimeter : ℝ) 
  (width : ℝ) 
  (length : ℝ) 
  (h1 : perimeter = 126)
  (h2 : length = 3 * width)
  (h3 : perimeter = 2 * length + 2 * width) : 
  (length / 2 : ℝ) = 23.625 := by sorry

end inscribed_rectangle_semicircle_radius_l227_22770


namespace remainder_1234567_div_256_l227_22758

theorem remainder_1234567_div_256 : 1234567 % 256 = 45 := by
  sorry

end remainder_1234567_div_256_l227_22758


namespace circle_and_chord_theorem_l227_22744

/-- The polar coordinate equation of a circle C that passes through the point (√2, π/4)
    and has its center at the intersection of the polar axis and the line ρ sin(θ - π/3) = -√3/2 -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The length of the chord intercepted by the line θ = π/3 on the circle C defined by ρ = 2cos(θ) -/
def chord_length : ℝ := 1

theorem circle_and_chord_theorem :
  /- Circle C passes through (√2, π/4) -/
  (circle_equation (Real.sqrt 2) (π / 4)) ∧
  /- The center of C is at the intersection of the polar axis and ρ sin(θ - π/3) = -√3/2 -/
  (∃ ρ₀ : ℝ, ρ₀ * Real.sin (0 - π / 3) = -Real.sqrt 3 / 2) ∧
  /- The polar coordinate equation of circle C is ρ = 2cos(θ) -/
  (∀ ρ θ : ℝ, circle_equation ρ θ ↔ ρ = 2 * Real.cos θ) ∧
  /- The length of the chord intercepted by θ = π/3 on circle C is 1 -/
  chord_length = 1 := by
    sorry

end circle_and_chord_theorem_l227_22744


namespace number_equation_solution_l227_22754

theorem number_equation_solution : 
  ∃ x : ℝ, x^2 + 95 = (x - 20)^2 ∧ x = 7.625 := by
  sorry

end number_equation_solution_l227_22754


namespace trig_identities_l227_22738

/-- Given tan α = 2, prove two trigonometric identities -/
theorem trig_identities (α : Real) (h : Real.tan α = 2) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 6/13 ∧
  3 * Real.sin α^2 + 3 * Real.sin α * Real.cos α - 2 * Real.cos α^2 = 16/5 := by
  sorry

end trig_identities_l227_22738


namespace isosceles_triangle_perimeter_l227_22711

/-- An isosceles triangle with side lengths satisfying x^2 - 5x + 6 = 0 has perimeter 7 or 8 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  (a^2 - 5*a + 6 = 0) →
  (b^2 - 5*b + 6 = 0) →
  (a = b ∨ a = c ∨ b = c) →  -- isosceles condition
  (a + b > c ∧ a + c > b ∧ b + c > a) →  -- triangle inequality
  (a + b + c = 7 ∨ a + b + c = 8) :=
by sorry

end isosceles_triangle_perimeter_l227_22711


namespace infinitely_many_non_representable_l227_22777

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- The statement that there exist infinitely many positive integers which cannot be written as a^(d(a)) + b^(d(b)) -/
theorem infinitely_many_non_representable : 
  ∃ (S : Set ℕ+), Set.Infinite S ∧ 
    ∀ (n : ℕ+), n ∈ S → 
      ∀ (a b : ℕ+), n ≠ a ^ (num_divisors a) + b ^ (num_divisors b) := by
  sorry

end infinitely_many_non_representable_l227_22777


namespace regular_polygons_ratio_l227_22715

/-- The interior angle of a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ := 180 - 360 / n

/-- The theorem statement -/
theorem regular_polygons_ratio (r k : ℕ) : 
  (r > 2 ∧ k > 2) →  -- Ensure polygons have at least 3 sides
  (interior_angle r / interior_angle k = 5 / 3) →
  (r = 2 * k) →
  (r = 8 ∧ k = 4) :=
by sorry

end regular_polygons_ratio_l227_22715


namespace arithmetic_expression_equality_l227_22783

theorem arithmetic_expression_equality : 1 + 6 * 2 - 3 + 5 * 4 / 2 = 20 := by
  sorry

end arithmetic_expression_equality_l227_22783


namespace square_equality_implies_m_equals_four_l227_22722

theorem square_equality_implies_m_equals_four (n m : ℝ) :
  (∀ x : ℝ, (x + n)^2 = x^2 + 4*x + m) → m = 4 := by
  sorry

end square_equality_implies_m_equals_four_l227_22722


namespace marathon_calories_burned_l227_22797

/-- Represents a cycling ride with its distance relative to the base distance -/
structure Ride :=
  (distance : ℝ)

/-- Calculates the adjusted distance for a ride given the actual distance and base distance -/
def adjustedDistance (actualDistance : ℝ) (baseDistance : ℝ) : ℝ :=
  actualDistance - baseDistance

/-- Calculates the total calories burned given a list of rides, base distance, and calorie burn rate -/
def totalCaloriesBurned (rides : List Ride) (baseDistance : ℝ) (caloriesPerKm : ℝ) : ℝ :=
  (rides.map (λ ride => ride.distance + baseDistance)).sum * caloriesPerKm

theorem marathon_calories_burned 
  (rides : List Ride)
  (baseDistance : ℝ)
  (caloriesPerKm : ℝ)
  (h1 : rides.length = 10)
  (h2 : baseDistance = 15)
  (h3 : caloriesPerKm = 20)
  (h4 : rides[3].distance = adjustedDistance 16.5 baseDistance)
  (h5 : rides[6].distance = adjustedDistance 14.1 baseDistance)
  : totalCaloriesBurned rides baseDistance caloriesPerKm = 3040 := by
  sorry

end marathon_calories_burned_l227_22797


namespace total_cost_is_49_27_l227_22755

/-- Represents the cost of tickets for a family outing to a theme park -/
def theme_park_tickets : ℝ → Prop :=
  λ total_cost : ℝ =>
    ∃ (regular_price : ℝ),
      -- A senior ticket (30% discount) costs $7.50
      0.7 * regular_price = 7.5 ∧
      -- Total cost calculation
      total_cost = 2 * 7.5 + -- Two senior tickets
                   2 * regular_price + -- Two regular tickets
                   2 * (0.6 * regular_price) -- Two children tickets (40% discount)

/-- The total cost for all tickets is $49.27 -/
theorem total_cost_is_49_27 : theme_park_tickets 49.27 := by
  sorry

end total_cost_is_49_27_l227_22755


namespace rectangular_parallelepiped_theorem_l227_22743

/-- Represents a rectangular parallelepiped -/
structure RectParallelepiped where
  base_side : ℝ
  cos_angle : ℝ

/-- Represents a vector configuration -/
structure VectorConfig where
  a_magnitude : ℝ
  a_dot_e : ℝ

theorem rectangular_parallelepiped_theorem (rp : RectParallelepiped) (vc : VectorConfig) :
  rp.base_side = 2 * Real.sqrt 2 →
  rp.cos_angle = Real.sqrt 3 / 3 →
  vc.a_magnitude = 2 * Real.sqrt 6 →
  vc.a_dot_e = 2 * Real.sqrt 2 →
  (∃ (sphere_surface_area : ℝ), sphere_surface_area = 24 * Real.pi) ∧
  (∃ (min_value : ℝ), min_value = 2 * Real.sqrt 2) := by
  sorry

#check rectangular_parallelepiped_theorem

end rectangular_parallelepiped_theorem_l227_22743


namespace negative_quadratic_range_l227_22700

theorem negative_quadratic_range (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ (a > 2 ∨ a < -2) := by sorry

end negative_quadratic_range_l227_22700


namespace inequality_solution_set_l227_22786

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the inequality
def inequality (x : ℝ) : Prop := log_half (2*x + 1) ≥ log_half 3

-- Define the solution set
def solution_set : Set ℝ := Set.Ioc (-1/2) 1

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end inequality_solution_set_l227_22786


namespace intersection_M_N_l227_22746

def M : Set ℝ := {x | x^2 > 1}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, 2} := by
  sorry

end intersection_M_N_l227_22746


namespace geli_workout_days_l227_22763

/-- Calculates the total number of push-ups for a given number of days -/
def totalPushUps (initialPushUps : ℕ) (dailyIncrease : ℕ) (days : ℕ) : ℕ :=
  days * initialPushUps + (days * (days - 1) * dailyIncrease) / 2

/-- Proves that Geli works out 3 times a week -/
theorem geli_workout_days : 
  ∃ (days : ℕ), days > 0 ∧ totalPushUps 10 5 days = 45 ∧ days = 3 := by
  sorry

#eval totalPushUps 10 5 3

end geli_workout_days_l227_22763


namespace floor_neg_five_thirds_l227_22704

theorem floor_neg_five_thirds : ⌊(-5/3 : ℚ)⌋ = -2 := by
  sorry

end floor_neg_five_thirds_l227_22704


namespace jeans_original_cost_l227_22772

/-- The original cost of jeans before discounts -/
def original_cost : ℝ := 49

/-- The summer discount as a percentage -/
def summer_discount : ℝ := 0.5

/-- The additional Wednesday discount in dollars -/
def wednesday_discount : ℝ := 10

/-- The final price after all discounts -/
def final_price : ℝ := 14.5

/-- Theorem stating that the original cost is correct given the discounts and final price -/
theorem jeans_original_cost :
  final_price = original_cost * (1 - summer_discount) - wednesday_discount := by
  sorry


end jeans_original_cost_l227_22772


namespace algebraic_expression_equality_l227_22785

theorem algebraic_expression_equality (x y : ℝ) (h : x - 2*y = 3) : 
  4*y + 1 - 2*x = -5 := by
  sorry

end algebraic_expression_equality_l227_22785


namespace ratio_transitivity_l227_22747

theorem ratio_transitivity (a b c : ℚ) 
  (hab : a / b = 8 / 3) 
  (hbc : b / c = 1 / 5) : 
  a / c = 8 / 15 := by
  sorry

end ratio_transitivity_l227_22747


namespace photographers_selection_l227_22719

theorem photographers_selection (n m : ℕ) (h1 : n = 10) (h2 : m = 3) :
  Nat.choose n m = 120 := by
  sorry

end photographers_selection_l227_22719


namespace population_growth_problem_l227_22794

theorem population_growth_problem (initial_population : ℝ) (final_population : ℝ) (second_year_decrease : ℝ) :
  initial_population = 10000 →
  final_population = 9600 →
  second_year_decrease = 20 →
  ∃ first_year_increase : ℝ,
    first_year_increase = 20 ∧
    final_population = initial_population * (1 + first_year_increase / 100) * (1 - second_year_decrease / 100) :=
by sorry

end population_growth_problem_l227_22794


namespace A_3_2_equals_13_l227_22721

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2_equals_13 : A 3 2 = 13 := by sorry

end A_3_2_equals_13_l227_22721


namespace table_chair_price_ratio_l227_22750

/-- The price ratio of tables to chairs in a store -/
theorem table_chair_price_ratio :
  ∀ (chair_price table_price : ℝ),
  chair_price > 0 →
  table_price > 0 →
  2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price) →
  table_price = 7 * chair_price :=
by
  sorry

end table_chair_price_ratio_l227_22750


namespace one_language_speakers_l227_22702

theorem one_language_speakers (total : ℕ) (latin french spanish : ℕ) (none : ℕ) 
  (latin_french latin_spanish french_spanish : ℕ) (all_three : ℕ) 
  (h1 : total = 40)
  (h2 : latin = 20)
  (h3 : french = 22)
  (h4 : spanish = 15)
  (h5 : none = 5)
  (h6 : latin_french = 8)
  (h7 : latin_spanish = 6)
  (h8 : french_spanish = 4)
  (h9 : all_three = 3) :
  total - none - (latin_french + latin_spanish + french_spanish - 2 * all_three) - all_three = 20 := by
  sorry

#check one_language_speakers

end one_language_speakers_l227_22702


namespace mushroom_remainder_l227_22799

theorem mushroom_remainder (initial : ℕ) (consumed : ℕ) (remaining : ℕ) : 
  initial = 15 → consumed = 8 → remaining = initial - consumed → remaining = 7 := by
  sorry

end mushroom_remainder_l227_22799


namespace prob_even_sum_l227_22764

/-- Probability of selecting an even number from the first wheel -/
def P_even1 : ℚ := 2/3

/-- Probability of selecting an odd number from the first wheel -/
def P_odd1 : ℚ := 1/3

/-- Probability of selecting an even number from the second wheel -/
def P_even2 : ℚ := 1/2

/-- Probability of selecting an odd number from the second wheel -/
def P_odd2 : ℚ := 1/2

/-- The probability of selecting an even sum from two wheels with the given probability distributions -/
theorem prob_even_sum : P_even1 * P_even2 + P_odd1 * P_odd2 = 1/2 := by
  sorry


end prob_even_sum_l227_22764


namespace fraction_power_product_l227_22729

theorem fraction_power_product :
  (8 / 9 : ℚ)^3 * (5 / 3 : ℚ)^3 = 64000 / 19683 := by
  sorry

end fraction_power_product_l227_22729


namespace translate_line_2x_minus_1_l227_22723

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically -/
def translate_line (l : Line) (units : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + units }

/-- The theorem stating that translating y = 2x - 1 by 2 units 
    upward results in y = 2x + 1 -/
theorem translate_line_2x_minus_1 :
  let original_line : Line := { slope := 2, intercept := -1 }
  let translated_line := translate_line original_line 2
  translated_line = { slope := 2, intercept := 1 } := by
  sorry

end translate_line_2x_minus_1_l227_22723


namespace expression_value_l227_22789

/-- Given that when x = 30, the value of ax³ + bx - 7 is 9,
    prove that the value of ax³ + bx + 2 when x = -30 is -14 -/
theorem expression_value (a b : ℝ) : 
  (30^3 * a + 30 * b - 7 = 9) → 
  ((-30)^3 * a + (-30) * b + 2 = -14) :=
by sorry

end expression_value_l227_22789


namespace expression_value_l227_22787

theorem expression_value (x y : ℝ) (h : 2 * y - x = 5) :
  5 * (x - 2 * y)^2 + 3 * (x - 2 * y) + 10 = 120 := by
  sorry

end expression_value_l227_22787


namespace subtraction_of_large_numbers_l227_22713

theorem subtraction_of_large_numbers :
  10000000000000 - (5555555555555 * 2) = -1111111111110 := by
  sorry

end subtraction_of_large_numbers_l227_22713


namespace brick_height_l227_22748

/-- The surface area of a rectangular prism given its length, width, and height. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem stating that a rectangular prism with length 10, width 4, and surface area 136 has height 2. -/
theorem brick_height : ∃ (h : ℝ), h > 0 ∧ surface_area 10 4 h = 136 → h = 2 := by
  sorry

end brick_height_l227_22748


namespace functional_equation_solution_l227_22782

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y : ℚ, f (x + f y) = f x * f y) →
  (∀ x : ℚ, f x = 0) ∨ (∀ x : ℚ, f x = 1) := by
  sorry

end functional_equation_solution_l227_22782


namespace initial_capacity_correct_l227_22767

/-- The capacity of each bucket in the set of 20 buckets -/
def initial_bucket_capacity : ℝ := 13.5

/-- The number of buckets in the initial set -/
def initial_bucket_count : ℕ := 20

/-- The capacity of each bucket in the set of 30 buckets -/
def new_bucket_capacity : ℝ := 9

/-- The number of buckets in the new set -/
def new_bucket_count : ℕ := 30

/-- The theorem states that the initial bucket capacity is correct -/
theorem initial_capacity_correct : 
  initial_bucket_capacity * initial_bucket_count = new_bucket_capacity * new_bucket_count := by
  sorry

end initial_capacity_correct_l227_22767


namespace intersection_perpendicular_tangents_l227_22781

open Real

theorem intersection_perpendicular_tangents (a : ℝ) : 
  ∃ (x : ℝ), 0 < x ∧ x < π / 2 ∧ 
  2 * sin x = a * cos x ∧
  (2 * cos x) * (-a * sin x) = -1 →
  a = 2 * sqrt 3 / 3 := by
sorry

end intersection_perpendicular_tangents_l227_22781


namespace probability_one_pair_one_triplet_proof_l227_22733

/-- The probability of rolling six standard six-sided dice and getting exactly
    one pair, one triplet, and the remaining dice showing different values. -/
def probability_one_pair_one_triplet : ℚ := 25 / 162

/-- The number of possible outcomes when rolling six standard six-sided dice. -/
def total_outcomes : ℕ := 6^6

/-- The number of successful outcomes (one pair, one triplet, remaining different). -/
def successful_outcomes : ℕ := 7200

theorem probability_one_pair_one_triplet_proof :
  probability_one_pair_one_triplet = successful_outcomes / total_outcomes :=
by sorry

end probability_one_pair_one_triplet_proof_l227_22733


namespace euler_product_theorem_l227_22773

theorem euler_product_theorem (z₁ z₂ : ℂ) :
  z₁ = Complex.exp (Complex.I * (Real.pi / 3)) →
  z₂ = Complex.exp (Complex.I * (Real.pi / 6)) →
  z₁ * z₂ = Complex.I := by
  sorry

end euler_product_theorem_l227_22773


namespace x_plus_2y_equals_20_l227_22765

theorem x_plus_2y_equals_20 (x y : ℝ) (hx : x = 10) (hy : y = 5) : x + 2 * y = 20 := by
  sorry

end x_plus_2y_equals_20_l227_22765


namespace quadratic_complete_square_l227_22705

/-- Given a quadratic function y = x^2 - 2x + 3, prove it can be expressed as y = (x + m)^2 + h
    where m = -1 and h = 2 -/
theorem quadratic_complete_square :
  ∃ (m h : ℝ), ∀ (x y : ℝ),
    y = x^2 - 2*x + 3 → y = (x + m)^2 + h ∧ m = -1 ∧ h = 2 := by
  sorry

end quadratic_complete_square_l227_22705


namespace power_function_increasing_iff_m_eq_two_l227_22752

/-- A power function f(x) = (m^2 - m - 1)x^m is increasing on (0, +∞) if and only if m = 2 -/
theorem power_function_increasing_iff_m_eq_two (m : ℝ) :
  (∀ x > 0, StrictMono (fun x => (m^2 - m - 1) * x^m)) ↔ m = 2 :=
by sorry

end power_function_increasing_iff_m_eq_two_l227_22752


namespace midpoint_of_fractions_l227_22751

theorem midpoint_of_fractions :
  (1 / 6 + 1 / 12) / 2 = 1 / 8 := by
  sorry

end midpoint_of_fractions_l227_22751


namespace rabbit_jumps_l227_22714

def N (a : ℤ) : ℕ :=
  sorry

theorem rabbit_jumps (a : ℤ) : Odd (N a) ↔ a = 1 ∨ a = -1 := by
  sorry

end rabbit_jumps_l227_22714


namespace sine_cosine_relation_l227_22768

theorem sine_cosine_relation (α : ℝ) (h : Real.cos (α + π / 12) = 1 / 5) :
  Real.sin (α + 7 * π / 12) = 1 / 5 := by
  sorry

end sine_cosine_relation_l227_22768


namespace dust_particles_problem_l227_22716

theorem dust_particles_problem (initial_dust : ℕ) : 
  (initial_dust / 10 + 223 = 331) → initial_dust = 1080 := by
  sorry

end dust_particles_problem_l227_22716


namespace four_times_angle_triangle_l227_22734

theorem four_times_angle_triangle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  (α = 40 ∧ β = 4 * γ) ∨ (α = 40 ∧ γ = 4 * β) ∨ (β = 40 ∧ α = 4 * γ) →  -- One angle is 40° and another is 4 times the third
  ((β = 130 ∧ γ = 10) ∨ (β = 112 ∧ γ = 28)) ∨ 
  ((α = 130 ∧ γ = 10) ∨ (α = 112 ∧ γ = 28)) ∨ 
  ((α = 130 ∧ β = 10) ∨ (α = 112 ∧ β = 28)) :=
by sorry

end four_times_angle_triangle_l227_22734


namespace inequality_solution_l227_22718

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) > 1 / 5) ↔ 
  (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) := by sorry

end inequality_solution_l227_22718


namespace quadratic_coefficient_value_l227_22732

theorem quadratic_coefficient_value (b : ℝ) (n : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 88 = (x + n)^2 + 16) → 
  b = 12 * Real.sqrt 2 := by
sorry

end quadratic_coefficient_value_l227_22732


namespace boys_on_trip_l227_22739

/-- Calculates the number of boys on a family trip given the specified conditions. -/
def number_of_boys (adults : ℕ) (total_eggs : ℕ) (eggs_per_adult : ℕ) (girls : ℕ) (eggs_per_girl : ℕ) : ℕ :=
  let eggs_for_children := total_eggs - adults * eggs_per_adult
  let eggs_for_girls := girls * eggs_per_girl
  let eggs_for_boys := eggs_for_children - eggs_for_girls
  let eggs_per_boy := eggs_per_girl + 1
  eggs_for_boys / eggs_per_boy

/-- Theorem stating that the number of boys on the trip is 10 under the given conditions. -/
theorem boys_on_trip :
  number_of_boys 3 (3 * 12) 3 7 1 = 10 := by
  sorry

end boys_on_trip_l227_22739


namespace dirk_profit_l227_22740

/-- Calculates the profit for selling amulets at a Ren Faire --/
def amulet_profit (days : ℕ) (amulets_per_day : ℕ) (sell_price : ℕ) (cost_price : ℕ) (faire_fee_percent : ℕ) : ℕ :=
  let total_amulets := days * amulets_per_day
  let revenue := total_amulets * sell_price
  let faire_fee := revenue * faire_fee_percent / 100
  let revenue_after_fee := revenue - faire_fee
  let total_cost := total_amulets * cost_price
  revenue_after_fee - total_cost

/-- Theorem stating that Dirk's profit is 300 dollars --/
theorem dirk_profit :
  amulet_profit 2 25 40 30 10 = 300 := by
  sorry

end dirk_profit_l227_22740


namespace triangle_side_possible_value_l227_22779

theorem triangle_side_possible_value (a : ℤ) : 
  (a > 0) → 
  (7 + 3 > a) → 
  (7 + a > 3) → 
  (3 + a > 7) → 
  (a = 8) → 
  ∃ (x y z : ℝ), x = 7 ∧ y = a ∧ z = 3 ∧ x + y > z ∧ x + z > y ∧ y + z > x :=
by sorry

end triangle_side_possible_value_l227_22779
