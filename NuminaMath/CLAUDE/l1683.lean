import Mathlib

namespace NUMINAMATH_CALUDE_fraction_subtraction_theorem_l1683_168362

theorem fraction_subtraction_theorem : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_theorem_l1683_168362


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l1683_168382

/-- 
Given a point P with polar coordinates (r, θ), 
this theorem states that its Cartesian coordinates are (r cos(θ), r sin(θ)).
-/
theorem polar_to_cartesian (r θ : ℝ) : 
  let p : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)
  ∃ (x y : ℝ), p = (x, y) ∧ 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l1683_168382


namespace NUMINAMATH_CALUDE_divisor_power_difference_l1683_168327

theorem divisor_power_difference (k : ℕ) : 
  (15 ^ k ∣ 759325) → 3 ^ k - k ^ 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_power_difference_l1683_168327


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l1683_168324

theorem triangle_ABC_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  Real.sqrt 3 * b * Real.cos A = Real.sin A * (a * Real.cos C + c * Real.cos A) →
  a = 2 * Real.sqrt 3 →
  (5 * Real.sqrt 3) / 4 = (1 / 2) * a * b * Real.sin C →
  (A = π / 3) ∧ (a + b + c = 5 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l1683_168324


namespace NUMINAMATH_CALUDE_textbook_weight_difference_l1683_168308

/-- Given the weights of four textbooks, prove that the difference between
    the sum of the middle two weights and the difference between the
    largest and smallest weights is 2.5 pounds. -/
theorem textbook_weight_difference
  (chemistry_weight geometry_weight calculus_weight biology_weight : ℝ)
  (h1 : chemistry_weight = 7.125)
  (h2 : geometry_weight = 0.625)
  (h3 : calculus_weight = 5.25)
  (h4 : biology_weight = 3.75)
  : (calculus_weight + biology_weight) - (chemistry_weight - geometry_weight) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_textbook_weight_difference_l1683_168308


namespace NUMINAMATH_CALUDE_mixed_numbers_sum_range_l1683_168368

theorem mixed_numbers_sum_range : 
  let a : ℚ := 3 + 1 / 9
  let b : ℚ := 4 + 1 / 3
  let c : ℚ := 6 + 1 / 21
  let sum : ℚ := a + b + c
  13.5 < sum ∧ sum < 14 := by
sorry

end NUMINAMATH_CALUDE_mixed_numbers_sum_range_l1683_168368


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_ratio_l1683_168391

def sum_odd_from_3 (n : ℕ) : ℕ := n^2 + 2*n

def sum_even (n : ℕ) : ℕ := n*(n+1)

theorem smallest_n_satisfying_ratio : 
  ∀ n : ℕ, n > 0 → (n < 51 → (sum_odd_from_3 n : ℚ) / sum_even n ≠ 49/50) ∧
  (sum_odd_from_3 51 : ℚ) / sum_even 51 = 49/50 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_ratio_l1683_168391


namespace NUMINAMATH_CALUDE_cos_pi_third_plus_two_alpha_l1683_168300

theorem cos_pi_third_plus_two_alpha (α : Real) 
  (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.cos (π / 3 + 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_plus_two_alpha_l1683_168300


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_with_16_factors_l1683_168332

-- Define a function to count the number of factors of a natural number
def count_factors (n : ℕ) : ℕ := sorry

-- Define a function to check if a number is five digits
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

-- Define the main theorem
theorem smallest_five_digit_multiple_with_16_factors : 
  ∀ n : ℕ, is_five_digit n → n % 2014 = 0 → 
  count_factors (n % 1000) = 16 → n ≥ 24168 := by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_with_16_factors_l1683_168332


namespace NUMINAMATH_CALUDE_nonnegative_difference_of_roots_l1683_168326

theorem nonnegative_difference_of_roots (x : ℝ) : 
  let roots := {r : ℝ | r^2 + 6*r + 8 = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_nonnegative_difference_of_roots_l1683_168326


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l1683_168312

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l1683_168312


namespace NUMINAMATH_CALUDE_students_to_add_l1683_168378

theorem students_to_add (current_students : ℕ) (teachers : ℕ) (h1 : current_students = 1049) (h2 : teachers = 9) :
  ∃ (students_to_add : ℕ), 
    students_to_add = 4 ∧
    (current_students + students_to_add) % teachers = 0 ∧
    ∀ (n : ℕ), n < students_to_add → (current_students + n) % teachers ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_students_to_add_l1683_168378


namespace NUMINAMATH_CALUDE_inequality_proof_l1683_168394

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / Real.sqrt (x * y) ≤ x / y + y / x :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1683_168394


namespace NUMINAMATH_CALUDE_data_entry_team_size_l1683_168392

theorem data_entry_team_size :
  let rudy_speed := 64
  let joyce_speed := 76
  let gladys_speed := 91
  let lisa_speed := 80
  let mike_speed := 89
  let team_average := 80
  let total_speed := rudy_speed + joyce_speed + gladys_speed + lisa_speed + mike_speed
  (total_speed / team_average : ℚ) = 5 := by sorry

end NUMINAMATH_CALUDE_data_entry_team_size_l1683_168392


namespace NUMINAMATH_CALUDE_line_intersects_y_axis_l1683_168372

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line passing through two points
def Line (p1 p2 : Point2D) :=
  {p : Point2D | ∃ t : ℝ, p.x = p1.x + t * (p2.x - p1.x) ∧ p.y = p1.y + t * (p2.y - p1.y)}

-- The given points
def p1 : Point2D := ⟨2, 9⟩
def p2 : Point2D := ⟨4, 15⟩

-- The y-axis
def yAxis : Set Point2D := {p : Point2D | p.x = 0}

-- The intersection point
def intersectionPoint : Point2D := ⟨0, 3⟩

-- The theorem to prove
theorem line_intersects_y_axis :
  intersectionPoint ∈ Line p1 p2 ∩ yAxis := by sorry

end NUMINAMATH_CALUDE_line_intersects_y_axis_l1683_168372


namespace NUMINAMATH_CALUDE_hannah_stocking_stuffers_l1683_168356

/-- The number of candy canes per stocking -/
def candy_canes : Nat := 4

/-- The number of beanie babies per stocking -/
def beanie_babies : Nat := 2

/-- The number of books per stocking -/
def books : Nat := 1

/-- The number of kids Hannah has -/
def num_kids : Nat := 3

/-- The total number of stocking stuffers Hannah buys -/
def total_stuffers : Nat := (candy_canes + beanie_babies + books) * num_kids

theorem hannah_stocking_stuffers :
  total_stuffers = 21 := by sorry

end NUMINAMATH_CALUDE_hannah_stocking_stuffers_l1683_168356


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l1683_168329

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  Complex.abs ((z - 2)^2 * (z + 2)) ≤ 16 * Real.sqrt 2 ∧
  ∃ w : ℂ, Complex.abs w = 2 ∧ Complex.abs ((w - 2)^2 * (w + 2)) = 16 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l1683_168329


namespace NUMINAMATH_CALUDE_solution_characterization_l1683_168334

/-- The set of polynomials that satisfy the given condition -/
def SolutionSet : Set (Polynomial ℤ) :=
  {f | f = Polynomial.monomial 3 1 + Polynomial.monomial 2 1 + Polynomial.monomial 1 1 + Polynomial.monomial 0 1 ∨
       f = Polynomial.monomial 3 1 + Polynomial.monomial 2 2 + Polynomial.monomial 1 2 + Polynomial.monomial 0 2 ∨
       f = Polynomial.monomial 3 2 + Polynomial.monomial 2 1 + Polynomial.monomial 1 2 + Polynomial.monomial 0 1 ∨
       f = Polynomial.monomial 3 2 + Polynomial.monomial 2 2 + Polynomial.monomial 1 1 + Polynomial.monomial 0 2}

/-- The condition that f must satisfy -/
def SatisfiesCondition (f : Polynomial ℤ) : Prop :=
  ∃ g h : Polynomial ℤ, f^4 + 2*f + 2 = (Polynomial.monomial 4 1 + 2*Polynomial.monomial 2 1 + 2)*g + 3*h

theorem solution_characterization :
  ∀ f : Polynomial ℤ, (f ∈ SolutionSet ↔ (SatisfiesCondition f ∧ 
    ∀ f' : Polynomial ℤ, SatisfiesCondition f' → (Polynomial.degree f' ≥ Polynomial.degree f))) :=
sorry

end NUMINAMATH_CALUDE_solution_characterization_l1683_168334


namespace NUMINAMATH_CALUDE_problem_solution_l1683_168345

theorem problem_solution (a b c x y z : ℝ) 
  (h1 : a * (x / c) + b * (y / a) + c * (z / b) = 5)
  (h2 : c / x + a / y + b / z = 0)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  x^2 / c^2 + y^2 / a^2 + z^2 / b^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1683_168345


namespace NUMINAMATH_CALUDE_students_liking_both_subjects_l1683_168393

theorem students_liking_both_subjects 
  (total_students : ℕ) 
  (art_students : ℕ) 
  (science_students : ℕ) 
  (h1 : total_students = 45)
  (h2 : art_students = 42)
  (h3 : science_students = 40)
  (h4 : art_students ≤ total_students)
  (h5 : science_students ≤ total_students) :
  art_students + science_students - total_students = 37 := by
sorry

end NUMINAMATH_CALUDE_students_liking_both_subjects_l1683_168393


namespace NUMINAMATH_CALUDE_blue_face_probability_l1683_168328

structure Octahedron :=
  (total_faces : ℕ)
  (blue_faces : ℕ)
  (red_faces : ℕ)
  (green_faces : ℕ)
  (face_sum : blue_faces + red_faces + green_faces = total_faces)

def roll_probability (o : Octahedron) : ℚ :=
  o.blue_faces / o.total_faces

theorem blue_face_probability (o : Octahedron) 
  (h1 : o.total_faces = 8)
  (h2 : o.blue_faces = 4)
  (h3 : o.red_faces = 3)
  (h4 : o.green_faces = 1) :
  roll_probability o = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_blue_face_probability_l1683_168328


namespace NUMINAMATH_CALUDE_common_root_and_parameter_l1683_168342

theorem common_root_and_parameter :
  ∃ (x p : ℚ), 
    x = -5 ∧ 
    p = 14/3 ∧ 
    p = -(x^2 - x - 2) / (x - 1) ∧ 
    p = -(x^2 + 2*x - 1) / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_common_root_and_parameter_l1683_168342


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l1683_168335

theorem smallest_number_with_given_remainders : ∃ (n : ℕ), 
  (n % 19 = 9 ∧ n % 23 = 7) ∧ 
  (∀ m : ℕ, m % 19 = 9 ∧ m % 23 = 7 → n ≤ m) ∧
  n = 161 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l1683_168335


namespace NUMINAMATH_CALUDE_daniel_purchase_cost_l1683_168397

/-- The total cost of items bought by Daniel -/
def total_cost (tax_amount : ℚ) (tax_rate : ℚ) (tax_free_cost : ℚ) : ℚ :=
  (tax_amount / tax_rate) + tax_free_cost

/-- Theorem stating the total cost of items Daniel bought -/
theorem daniel_purchase_cost :
  let tax_amount : ℚ := 30 / 100  -- 30 paise = 0.30 rupees
  let tax_rate : ℚ := 6 / 100     -- 6%
  let tax_free_cost : ℚ := 347 / 10  -- Rs. 34.7
  total_cost tax_amount tax_rate tax_free_cost = 397 / 10 := by
  sorry

#eval total_cost (30/100) (6/100) (347/10)

end NUMINAMATH_CALUDE_daniel_purchase_cost_l1683_168397


namespace NUMINAMATH_CALUDE_somu_age_problem_l1683_168315

theorem somu_age_problem (somu_age father_age : ℕ) : 
  somu_age = father_age / 3 →
  somu_age - 6 = (father_age - 6) / 5 →
  somu_age = 12 := by
sorry

end NUMINAMATH_CALUDE_somu_age_problem_l1683_168315


namespace NUMINAMATH_CALUDE_complex_magnitude_l1683_168385

theorem complex_magnitude (a b : ℝ) (h : (1 + 2*a*Complex.I) * Complex.I = 1 - b*Complex.I) :
  Complex.abs (a + b*Complex.I) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1683_168385


namespace NUMINAMATH_CALUDE_bucket_weight_l1683_168388

theorem bucket_weight (c d : ℝ) : ℝ :=
  let weight_three_quarters : ℝ := c
  let weight_one_third : ℝ := d
  let weight_full : ℝ := (8 * c / 5) - (3 * d / 5)
  weight_full

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l1683_168388


namespace NUMINAMATH_CALUDE_fair_prize_division_l1683_168371

/-- Represents the state of the game --/
structure GameState where
  player1_wins : ℕ
  player2_wins : ℕ

/-- Calculates the probability of a player winning the game from a given state --/
def win_probability (state : GameState) : ℚ :=
  1 - (1/2) ^ (6 - state.player1_wins)

/-- Theorem stating the fair division of the prize --/
theorem fair_prize_division (state : GameState) 
  (h1 : state.player1_wins = 5)
  (h2 : state.player2_wins = 3) :
  let p1_prob := win_probability state
  let p2_prob := 1 - p1_prob
  (p1_prob : ℚ) / p2_prob = 7 / 1 := by sorry

end NUMINAMATH_CALUDE_fair_prize_division_l1683_168371


namespace NUMINAMATH_CALUDE_joan_football_games_l1683_168376

/-- The number of football games Joan went to this year -/
def games_this_year : ℕ := 4

/-- The total number of football games Joan went to -/
def total_games : ℕ := 13

/-- The number of football games Joan went to last year -/
def games_last_year : ℕ := total_games - games_this_year

theorem joan_football_games : games_last_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_football_games_l1683_168376


namespace NUMINAMATH_CALUDE_coffee_shop_solution_l1683_168331

/-- Represents the coffee shop scenario with Alice and Bob -/
def coffee_shop_scenario (x : ℝ) : Prop :=
  let alice_initial := x
  let bob_initial := 1.25 * x
  let alice_consumed := 0.75 * x
  let bob_consumed := 0.75 * (1.25 * x)
  let alice_remaining := 0.25 * x
  let bob_remaining := 1.25 * x - 0.75 * (1.25 * x)
  let alice_gives := 0.5 * alice_remaining + 1
  let alice_final := alice_consumed - alice_gives
  let bob_final := bob_consumed + alice_gives
  (alice_final = bob_final) ∧
  (alice_initial + bob_initial = 9)

/-- Theorem stating that there exists a solution to the coffee shop scenario -/
theorem coffee_shop_solution : ∃ x : ℝ, coffee_shop_scenario x := by
  sorry


end NUMINAMATH_CALUDE_coffee_shop_solution_l1683_168331


namespace NUMINAMATH_CALUDE_zuzkas_number_l1683_168336

theorem zuzkas_number : ∃! n : ℕ, 
  10000 ≤ n ∧ n < 100000 ∧ 
  10 * n + 1 = 3 * (100000 + n) := by
sorry

end NUMINAMATH_CALUDE_zuzkas_number_l1683_168336


namespace NUMINAMATH_CALUDE_unique_solution_for_reciprocal_squares_sum_l1683_168398

theorem unique_solution_for_reciprocal_squares_sum (x y z t : ℕ+) :
  (1 : ℚ) / x^2 + (1 : ℚ) / y^2 + (1 : ℚ) / z^2 + (1 : ℚ) / t^2 = 1 →
  (x = 2 ∧ y = 2 ∧ z = 2 ∧ t = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_reciprocal_squares_sum_l1683_168398


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1683_168301

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y + 10) :
  x + y = 14 ∨ x + y = -2 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1683_168301


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_max_l1683_168386

theorem triangle_cosine_sum_max (a b c : ℝ) (x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hxyz : x + y + z = π) : 
  (∃ (x y z : ℝ), x + y + z = π ∧ 
    a * Real.cos x + b * Real.cos y + c * Real.cos z ≤ (1/2) * (a*b/c + a*c/b + b*c/a)) :=
sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_max_l1683_168386


namespace NUMINAMATH_CALUDE_lock_settings_count_l1683_168311

/-- The number of digits on each dial of the lock -/
def num_digits : ℕ := 8

/-- The number of dials on the lock -/
def num_dials : ℕ := 4

/-- The number of different settings possible for the lock -/
def num_settings : ℕ := 1680

/-- Theorem stating that the number of different settings for the lock
    with the given conditions is equal to 1680 -/
theorem lock_settings_count :
  (num_digits.factorial) / ((num_digits - num_dials).factorial) = num_settings :=
sorry

end NUMINAMATH_CALUDE_lock_settings_count_l1683_168311


namespace NUMINAMATH_CALUDE_smallest_urn_satisfying_condition_l1683_168319

/-- An urn contains marbles of five colors: red, white, blue, green, and yellow. -/
structure Urn :=
  (red : ℕ)
  (white : ℕ)
  (blue : ℕ)
  (green : ℕ)
  (yellow : ℕ)

/-- The total number of marbles in the urn -/
def Urn.total (u : Urn) : ℕ := u.red + u.white + u.blue + u.green + u.yellow

/-- The probability of drawing five red marbles -/
def Urn.prob_five_red (u : Urn) : ℚ :=
  (u.red.choose 5 : ℚ) / (u.total.choose 5)

/-- The probability of drawing one white, one blue, and three red marbles -/
def Urn.prob_one_white_one_blue_three_red (u : Urn) : ℚ :=
  ((u.white.choose 1) * (u.blue.choose 1) * (u.red.choose 3) : ℚ) / (u.total.choose 5)

/-- The probability of drawing one white, one blue, one green, and two red marbles -/
def Urn.prob_one_white_one_blue_one_green_two_red (u : Urn) : ℚ :=
  ((u.white.choose 1) * (u.blue.choose 1) * (u.green.choose 1) * (u.red.choose 2) : ℚ) / (u.total.choose 5)

/-- The probability of drawing one marble of each color except yellow -/
def Urn.prob_one_each_except_yellow (u : Urn) : ℚ :=
  ((u.white.choose 1) * (u.blue.choose 1) * (u.green.choose 1) * (u.red.choose 1) : ℚ) / (u.total.choose 5)

/-- The probability of drawing one marble of each color -/
def Urn.prob_one_each (u : Urn) : ℚ :=
  ((u.white.choose 1) * (u.blue.choose 1) * (u.green.choose 1) * (u.red.choose 1) * (u.yellow.choose 1) : ℚ) / (u.total.choose 5)

/-- The urn satisfies the equal probability condition -/
def Urn.satisfies_condition (u : Urn) : Prop :=
  u.prob_five_red = u.prob_one_white_one_blue_three_red ∧
  u.prob_five_red = u.prob_one_white_one_blue_one_green_two_red ∧
  u.prob_five_red = u.prob_one_each_except_yellow ∧
  u.prob_five_red = u.prob_one_each

theorem smallest_urn_satisfying_condition :
  ∃ (u : Urn), u.satisfies_condition ∧ u.total = 14 ∧ ∀ (v : Urn), v.satisfies_condition → u.total ≤ v.total :=
sorry

end NUMINAMATH_CALUDE_smallest_urn_satisfying_condition_l1683_168319


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1683_168340

theorem ratio_of_sum_and_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1683_168340


namespace NUMINAMATH_CALUDE_sum_first_three_special_sequence_l1683_168360

/-- An arithmetic sequence with given fourth, fifth, and sixth terms -/
def ArithmeticSequence (a₄ a₅ a₆ : ℤ) : ℕ → ℤ :=
  fun n => a₄ + (n - 4) * (a₅ - a₄)

/-- The sum of the first three terms of an arithmetic sequence -/
def SumFirstThree (seq : ℕ → ℤ) : ℤ :=
  seq 1 + seq 2 + seq 3

theorem sum_first_three_special_sequence :
  let seq := ArithmeticSequence 4 7 10
  SumFirstThree seq = -6 := by sorry

end NUMINAMATH_CALUDE_sum_first_three_special_sequence_l1683_168360


namespace NUMINAMATH_CALUDE_simple_interest_rate_l1683_168318

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time_years simple_interest : ℝ) :
  principal = 10000 →
  time_years = 1 →
  simple_interest = 400 →
  (simple_interest / (principal * time_years)) * 100 = 4 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l1683_168318


namespace NUMINAMATH_CALUDE_cheese_distribution_l1683_168344

/-- Represents the amount of cheese bought by the first n customers -/
def S (n : ℕ) : ℚ := 20 * n / (n + 10)

/-- The total amount of cheese available -/
def total_cheese : ℚ := 20

theorem cheese_distribution (n : ℕ) (h : n ≤ 10) :
  (total_cheese - S n = 10 * (S n / n)) ∧
  (∀ k : ℕ, k ≤ n → S k - S (k-1) > 0) ∧
  (S 10 = 10) := by sorry

#check cheese_distribution

end NUMINAMATH_CALUDE_cheese_distribution_l1683_168344


namespace NUMINAMATH_CALUDE_bills_toilet_paper_supply_l1683_168307

/-- Theorem: Bill's Toilet Paper Supply

Given:
- Bill uses the bathroom 3 times a day
- Bill uses 5 squares of toilet paper each time
- Each roll has 300 squares of toilet paper
- Bill's toilet paper supply will last for 20000 days

Prove that Bill has 1000 rolls of toilet paper.
-/
theorem bills_toilet_paper_supply 
  (bathroom_visits_per_day : ℕ) 
  (squares_per_visit : ℕ) 
  (squares_per_roll : ℕ) 
  (supply_duration_days : ℕ) 
  (h1 : bathroom_visits_per_day = 3)
  (h2 : squares_per_visit = 5)
  (h3 : squares_per_roll = 300)
  (h4 : supply_duration_days = 20000) :
  (bathroom_visits_per_day * squares_per_visit * supply_duration_days) / squares_per_roll = 1000 := by
  sorry

#check bills_toilet_paper_supply

end NUMINAMATH_CALUDE_bills_toilet_paper_supply_l1683_168307


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l1683_168304

/-- A parabola is defined by its vertex, directrix, and focus. -/
structure Parabola where
  vertex : ℝ × ℝ
  directrix : ℝ
  focus : ℝ × ℝ

/-- Given a parabola with vertex at (2,0) and directrix x = -1, its focus is at (5,0). -/
theorem parabola_focus_coordinates :
  ∀ p : Parabola, p.vertex = (2, 0) ∧ p.directrix = -1 → p.focus = (5, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l1683_168304


namespace NUMINAMATH_CALUDE_max_basketballs_proof_l1683_168370

/-- The maximum number of basketballs that can be purchased given the constraints -/
def max_basketballs : ℕ := 26

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 40

/-- The cost of each basketball in dollars -/
def basketball_cost : ℕ := 80

/-- The cost of each soccer ball in dollars -/
def soccer_ball_cost : ℕ := 50

/-- The total budget in dollars -/
def total_budget : ℕ := 2800

theorem max_basketballs_proof :
  (∀ x : ℕ, 
    x ≤ total_balls ∧ 
    (basketball_cost * x + soccer_ball_cost * (total_balls - x) ≤ total_budget) →
    x ≤ max_basketballs) ∧
  (basketball_cost * max_basketballs + soccer_ball_cost * (total_balls - max_basketballs) ≤ total_budget) :=
sorry

end NUMINAMATH_CALUDE_max_basketballs_proof_l1683_168370


namespace NUMINAMATH_CALUDE_candy_box_price_increase_l1683_168347

theorem candy_box_price_increase (current_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) : 
  current_price = 10 ∧ 
  increase_percentage = 25 ∧ 
  current_price = original_price * (1 + increase_percentage / 100) →
  original_price = 8 := by
sorry

end NUMINAMATH_CALUDE_candy_box_price_increase_l1683_168347


namespace NUMINAMATH_CALUDE_no_positive_solution_l1683_168363

theorem no_positive_solution :
  ¬ ∃ (a b c d : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
    a * d + b = c ∧
    Real.sqrt a * Real.sqrt d + Real.sqrt b = Real.sqrt c :=
by sorry

end NUMINAMATH_CALUDE_no_positive_solution_l1683_168363


namespace NUMINAMATH_CALUDE_mean_home_runs_l1683_168325

def players_5 : ℕ := 4
def players_6 : ℕ := 3
def players_7 : ℕ := 2
def players_9 : ℕ := 1
def players_11 : ℕ := 1

def total_players : ℕ := players_5 + players_6 + players_7 + players_9 + players_11

def total_home_runs : ℕ := 5 * players_5 + 6 * players_6 + 7 * players_7 + 9 * players_9 + 11 * players_11

theorem mean_home_runs : 
  (total_home_runs : ℚ) / (total_players : ℚ) = 6.545454545 := by
  sorry

end NUMINAMATH_CALUDE_mean_home_runs_l1683_168325


namespace NUMINAMATH_CALUDE_rachel_age_when_emily_half_her_age_l1683_168359

def emily_current_age : ℕ := 20
def rachel_current_age : ℕ := 24

theorem rachel_age_when_emily_half_her_age :
  ∃ (x : ℕ), 
    (rachel_current_age - x = 2 * (emily_current_age - x)) ∧
    (rachel_current_age - x = 8) := by
  sorry

end NUMINAMATH_CALUDE_rachel_age_when_emily_half_her_age_l1683_168359


namespace NUMINAMATH_CALUDE_solve_baguette_problem_l1683_168389

def baguette_problem (batches_per_day : ℕ) (baguettes_per_batch : ℕ) 
  (sold_after_first : ℕ) (sold_after_second : ℕ) (left_at_end : ℕ) : Prop :=
  let total_baguettes := batches_per_day * baguettes_per_batch
  let sold_first_two := sold_after_first + sold_after_second
  let sold_after_third := total_baguettes - sold_first_two - left_at_end
  sold_after_third = 49

theorem solve_baguette_problem : 
  baguette_problem 3 48 37 52 6 := by sorry

end NUMINAMATH_CALUDE_solve_baguette_problem_l1683_168389


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l1683_168369

/-- Given an arithmetic sequence of 20 terms with first term 2 and last term 59,
    prove that the 5th term is 14. -/
theorem fifth_term_of_arithmetic_sequence :
  ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence
  a 0 = 2 →                            -- first term is 2
  a 19 = 59 →                          -- last term (20th term) is 59
  a 4 = 14 :=                          -- 5th term (index 4) is 14
by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l1683_168369


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1683_168321

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = -4 ∧ x₂ = -4.5 ∧ 
  (∀ x : ℝ, x^2 + 6*x + 8 = -(x + 4)*(x + 7) ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1683_168321


namespace NUMINAMATH_CALUDE_sector_angle_measure_l1683_168320

theorem sector_angle_measure (r : ℝ) (α : ℝ) 
  (h1 : α * r = 2)  -- arc length = 2
  (h2 : (1/2) * α * r^2 = 2)  -- area = 2
  : α = 1 := by
sorry

end NUMINAMATH_CALUDE_sector_angle_measure_l1683_168320


namespace NUMINAMATH_CALUDE_investment_return_percentage_l1683_168310

/-- Calculates the return percentage for a two-venture investment --/
def calculate_return_percentage (total_investment : ℚ) (investment1 : ℚ) (investment2 : ℚ) 
  (profit_percentage1 : ℚ) (loss_percentage2 : ℚ) : ℚ :=
  let profit1 := investment1 * profit_percentage1
  let loss2 := investment2 * loss_percentage2
  let net_income := profit1 - loss2
  (net_income / total_investment) * 100

/-- Theorem stating that the return percentage is 6.5% for the given investment scenario --/
theorem investment_return_percentage : 
  calculate_return_percentage 25000 16250 16250 (15/100) (5/100) = 13/2 :=
by sorry

end NUMINAMATH_CALUDE_investment_return_percentage_l1683_168310


namespace NUMINAMATH_CALUDE_hyperbola_intersection_perpendicular_l1683_168348

-- Define the hyperbola C₁
def C₁ (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a line with slope 1
def Line (x y b : ℝ) : Prop := y = x + b

-- Define the tangency condition
def IsTangent (b : ℝ) : Prop := b^2 = 2

-- Define the perpendicularity of two vectors
def Perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem hyperbola_intersection_perpendicular 
  (x₁ y₁ x₂ y₂ b : ℝ) : 
  C₁ x₁ y₁ → C₁ x₂ y₂ → 
  Line x₁ y₁ b → Line x₂ y₂ b → 
  IsTangent b → 
  Perpendicular x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_perpendicular_l1683_168348


namespace NUMINAMATH_CALUDE_sally_lost_balloons_l1683_168337

/-- Given that Sally initially had 9 orange balloons and now has 7 orange balloons,
    prove that she lost 2 orange balloons. -/
theorem sally_lost_balloons (initial : Nat) (current : Nat) 
    (h1 : initial = 9) (h2 : current = 7) : initial - current = 2 := by
  sorry

end NUMINAMATH_CALUDE_sally_lost_balloons_l1683_168337


namespace NUMINAMATH_CALUDE_linear_equation_condition_l1683_168379

theorem linear_equation_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ k b : ℝ, (a - 6) * x - y^(a - 6) = k * y + b) → a = 7 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l1683_168379


namespace NUMINAMATH_CALUDE_election_votes_l1683_168351

theorem election_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (60 : ℚ) / 100 * total_votes - (40 : ℚ) / 100 * total_votes = 288) : 
  (60 : ℚ) / 100 * total_votes = 864 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l1683_168351


namespace NUMINAMATH_CALUDE_least_subtraction_l1683_168313

theorem least_subtraction (n : ℕ) : n = 10 ↔ 
  (∀ m : ℕ, m < n → ¬(
    (2590 - n) % 9 = 6 ∧ 
    (2590 - n) % 11 = 6 ∧ 
    (2590 - n) % 13 = 6
  )) ∧
  (2590 - n) % 9 = 6 ∧ 
  (2590 - n) % 11 = 6 ∧ 
  (2590 - n) % 13 = 6 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_l1683_168313


namespace NUMINAMATH_CALUDE_teal_color_survey_l1683_168373

theorem teal_color_survey (total : ℕ) (more_blue : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 150)
  (h_more_blue : more_blue = 90)
  (h_both : both = 45)
  (h_neither : neither = 20) :
  ∃ (more_green : ℕ), more_green = 85 ∧ 
    total = more_blue + more_green - both + neither :=
by sorry

end NUMINAMATH_CALUDE_teal_color_survey_l1683_168373


namespace NUMINAMATH_CALUDE_cubic_root_inequality_l1683_168396

theorem cubic_root_inequality (p q x : ℝ) : x^3 + p*x + q = 0 → 4*q*x ≤ p^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_inequality_l1683_168396


namespace NUMINAMATH_CALUDE_triangle_problem_l1683_168380

open Real

theorem triangle_problem (a b c A B C : ℝ) 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sine_law : a / sin A = b / sin B ∧ b / sin B = c / sin C)
  (h_condition : Real.sqrt 3 * b * sin C = c * cos B + c) : 
  B = π / 3 ∧ 
  (b^2 = a * c → 2 / tan A + 1 / tan C = 2 * Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1683_168380


namespace NUMINAMATH_CALUDE_ittymangnark_catch_l1683_168339

/-- Represents the number of fish each family member and pet receives -/
structure FishDistribution where
  ittymangnark : ℕ
  kingnook : ℕ
  oomyapeck : ℕ
  yurraknalik : ℕ
  ankaq : ℕ
  nanuq : ℕ

/-- Represents the distribution of fish eyes -/
structure EyeDistribution where
  oomyapeck : ℕ
  yurraknalik : ℕ
  ankaq : ℕ
  nanuq : ℕ

/-- Theorem stating that given the fish and eye distribution, Ittymangnark caught 21 fish -/
theorem ittymangnark_catch (fish : FishDistribution) (eyes : EyeDistribution) :
  fish.ittymangnark = 3 →
  fish.kingnook = 4 →
  fish.oomyapeck = 1 →
  fish.yurraknalik = 2 →
  fish.ankaq = 1 →
  fish.nanuq = 3 →
  eyes.oomyapeck = 24 →
  eyes.yurraknalik = 4 →
  eyes.ankaq = 6 →
  eyes.nanuq = 8 →
  fish.ittymangnark + fish.kingnook + fish.oomyapeck + fish.yurraknalik + fish.ankaq + fish.nanuq = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_ittymangnark_catch_l1683_168339


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1683_168375

theorem polynomial_divisibility (A B : ℝ) : 
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^103 + A*x + B = 0) → A + B = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1683_168375


namespace NUMINAMATH_CALUDE_real_complex_condition_l1683_168387

theorem real_complex_condition (a : ℝ) : 
  (Complex.I * (a - 1)^2 + 4*a).im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_complex_condition_l1683_168387


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_OBEC_area_of_quadrilateral_OBEC_proof_l1683_168390

/-- A line with slope -3 passing through points A, B, and E -/
structure Line1 where
  slope : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  E : ℝ × ℝ

/-- Another line passing through points C, D, and E -/
structure Line2 where
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- Definition of the problem setup -/
def ProblemSetup (l1 : Line1) (l2 : Line2) : Prop :=
  l1.slope = -3 ∧
  l1.A.1 > 0 ∧ l1.A.2 = 0 ∧
  l1.B.1 = 0 ∧ l1.B.2 > 0 ∧
  l1.E = (3, 3) ∧
  l2.C = (6, 0) ∧
  l2.D.1 = 0 ∧ l2.D.2 ≠ 0 ∧
  l2.E = (3, 3)

/-- The main theorem to prove -/
theorem area_of_quadrilateral_OBEC (l1 : Line1) (l2 : Line2) 
  (h : ProblemSetup l1 l2) : ℝ :=
  22.5

/-- Proof of the theorem -/
theorem area_of_quadrilateral_OBEC_proof (l1 : Line1) (l2 : Line2) 
  (h : ProblemSetup l1 l2) : 
  area_of_quadrilateral_OBEC l1 l2 h = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_OBEC_area_of_quadrilateral_OBEC_proof_l1683_168390


namespace NUMINAMATH_CALUDE_cube_surface_area_approx_l1683_168361

-- Define the dimensions of the rectangular prism
def prism_length : ℝ := 10
def prism_width : ℝ := 5
def prism_height : ℝ := 24

-- Define the volume of the rectangular prism
def prism_volume : ℝ := prism_length * prism_width * prism_height

-- Define the edge length of the cube with the same volume
def cube_edge : ℝ := (prism_volume) ^ (1/3)

-- Define the surface area of the cube
def cube_surface_area : ℝ := 6 * (cube_edge ^ 2)

-- Theorem stating that the surface area of the cube is approximately 677.76 square inches
theorem cube_surface_area_approx :
  ∃ ε > 0, |cube_surface_area - 677.76| < ε :=
sorry

end NUMINAMATH_CALUDE_cube_surface_area_approx_l1683_168361


namespace NUMINAMATH_CALUDE_odd_sided_polygon_indivisible_l1683_168399

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry -- Additional conditions to ensure the polygon is regular

/-- The diameter of a polygon -/
def diameter (p : RegularPolygon n) : ℝ := sorry

/-- A division of a polygon into two parts -/
structure Division (p : RegularPolygon n) where
  part1 : Set (Fin n)
  part2 : Set (Fin n)
  is_partition : part1 ∪ part2 = univ ∧ part1 ∩ part2 = ∅

/-- The diameter of a part of a polygon -/
def part_diameter (p : RegularPolygon n) (part : Set (Fin n)) : ℝ := sorry

theorem odd_sided_polygon_indivisible (n : ℕ) (h : Odd n) (p : RegularPolygon n) :
  ∀ d : Division p, 
    part_diameter p d.part1 = diameter p ∨ part_diameter p d.part2 = diameter p := by
  sorry

end NUMINAMATH_CALUDE_odd_sided_polygon_indivisible_l1683_168399


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l1683_168358

open Real

/-- The function f(x) = x ln x is monotonically decreasing on the interval (0, 1/e) -/
theorem f_decreasing_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < (1 : ℝ)/Real.exp 1 →
  x₁ * log x₁ > x₂ * log x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l1683_168358


namespace NUMINAMATH_CALUDE_probability_no_distinct_roots_l1683_168352

def is_valid_pair (b c : ℤ) : Prop :=
  b.natAbs ≤ 4 ∧ c.natAbs ≤ 4 ∧ c ≥ 0

def has_distinct_real_roots (b c : ℤ) : Prop :=
  b^2 - 4*c > 0

def total_valid_pairs : ℕ := 45

def pairs_without_distinct_roots : ℕ := 27

theorem probability_no_distinct_roots :
  (pairs_without_distinct_roots : ℚ) / total_valid_pairs = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_no_distinct_roots_l1683_168352


namespace NUMINAMATH_CALUDE_trapezoid_area_is_400_l1683_168367

-- Define the trapezoid and square properties
def trapezoid_base1 : ℝ := 50
def trapezoid_base2 : ℝ := 30
def num_trapezoids : ℕ := 4
def outer_square_area : ℝ := 2500

-- Theorem statement
theorem trapezoid_area_is_400 :
  let outer_square_side : ℝ := trapezoid_base1
  let inner_square_side : ℝ := trapezoid_base2
  let inner_square_area : ℝ := inner_square_side ^ 2
  let total_trapezoid_area : ℝ := outer_square_area - inner_square_area
  let single_trapezoid_area : ℝ := total_trapezoid_area / num_trapezoids
  single_trapezoid_area = 400 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_400_l1683_168367


namespace NUMINAMATH_CALUDE_younger_person_age_l1683_168357

theorem younger_person_age (y e : ℕ) : 
  e = y + 20 →                  -- The ages differ by 20 years
  e - 8 = 5 * (y - 8) →         -- 8 years ago, elder was 5 times younger's age
  y = 13                        -- The younger person's age is 13
  := by sorry

end NUMINAMATH_CALUDE_younger_person_age_l1683_168357


namespace NUMINAMATH_CALUDE_product_of_primes_with_conditions_l1683_168341

theorem product_of_primes_with_conditions :
  ∃ (p q r : ℕ), 
    Prime p ∧ Prime q ∧ Prime r ∧
    (r - q = 2 * p) ∧
    (r * q + p^2 = 676) ∧
    (p * q * r = 2001) := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_with_conditions_l1683_168341


namespace NUMINAMATH_CALUDE_pineapples_cost_theorem_l1683_168306

/-- The cost relationship between bananas, apples, and pineapples -/
structure FruitCosts where
  banana_to_apple : ℚ    -- 5 bananas = 3 apples
  apple_to_pineapple : ℚ  -- 9 apples = 6 pineapples

/-- The number of pineapples that cost the same as 30 bananas -/
def pineapples_equal_to_30_bananas (costs : FruitCosts) : ℚ :=
  30 * (costs.apple_to_pineapple / 9) * (3 / 5)

theorem pineapples_cost_theorem (costs : FruitCosts) 
  (h1 : costs.banana_to_apple = 3 / 5)
  (h2 : costs.apple_to_pineapple = 6 / 9) :
  pineapples_equal_to_30_bananas costs = 12 := by
  sorry

end NUMINAMATH_CALUDE_pineapples_cost_theorem_l1683_168306


namespace NUMINAMATH_CALUDE_min_multiplications_proof_twelve_numbers_multiplications_l1683_168365

/-- The minimum number of multiplications needed to multiply n numbers -/
def min_multiplications (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

/-- Theorem stating that the minimum number of multiplications for n ≥ 2 numbers is n-1 -/
theorem min_multiplications_proof (n : ℕ) (h : n ≥ 2) :
  min_multiplications n = n - 1 := by
  sorry

/-- Corollary for the specific case of 12 numbers -/
theorem twelve_numbers_multiplications :
  min_multiplications 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_min_multiplications_proof_twelve_numbers_multiplications_l1683_168365


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l1683_168374

theorem complex_magnitude_theorem (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 3 / w = s) : 
  Complex.abs w = (3 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l1683_168374


namespace NUMINAMATH_CALUDE_refrigerator_price_l1683_168338

theorem refrigerator_price (P : ℝ) 
  (h1 : 1.1 * P = 21725)  -- Selling price for 10% profit
  (h2 : 0.8 * P + 125 + 250 = 16175)  -- Price paid by buyer
  : True :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_price_l1683_168338


namespace NUMINAMATH_CALUDE_parabola_max_triangle_area_l1683_168353

/-- Given a parabola y = ax^2 + bx + c with a ≠ 0, intersecting the x-axis at A and B
    and the y-axis at C, with its vertex on y = -1, and ABC forming a right triangle,
    prove that the maximum area of triangle ABC is 1. -/
theorem parabola_max_triangle_area (a b c : ℝ) (ha : a ≠ 0) : 
  let f := fun x => a * x^2 + b * x + c
  let vertex_y := -1
  let A := {x : ℝ | f x = 0 ∧ x < 0}
  let B := {x : ℝ | f x = 0 ∧ x > 0}
  let C := (0, c)
  (∃ x, f x = vertex_y) →
  (∃ x₁ ∈ A, ∃ x₂ ∈ B, c^2 = (-x₁) * x₂) →
  (∀ S : ℝ, S = (1/2) * |c| * |x₂ - x₁| → S ≤ 1) ∧ 
  (∃ S : ℝ, S = (1/2) * |c| * |x₂ - x₁| ∧ S = 1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_max_triangle_area_l1683_168353


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_chord_length_l1683_168333

/-- Given a parabola and a circle, prove the length of the chord formed by their intersection -/
theorem parabola_circle_intersection_chord_length :
  ∀ (p : ℝ) (x y : ℝ → ℝ),
    p > 0 →
    (∀ t, y t ^ 2 = 2 * p * x t) →
    (∀ t, (x t - 1) ^ 2 + (y t + 2) ^ 2 = 9) →
    x 0 = 1 ∧ y 0 = -2 →
    ∃ (a b : ℝ), a ≠ b ∧
      x a = -1 ∧ x b = -1 ∧
      (x a - 1) ^ 2 + (y a + 2) ^ 2 = 9 ∧
      (x b - 1) ^ 2 + (y b + 2) ^ 2 = 9 ∧
      (y a - y b) ^ 2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_chord_length_l1683_168333


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1683_168322

theorem simplify_and_rationalize : 
  (Real.sqrt 7 / Real.sqrt 3) * (Real.sqrt 8 / Real.sqrt 5) * (Real.sqrt 9 / Real.sqrt 7) = 2 * Real.sqrt 30 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1683_168322


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l1683_168349

theorem mistaken_multiplication (x y : ℕ) : 
  x ≥ 1000000 ∧ x ≤ 9999999 ∧
  y ≥ 1000000 ∧ y ≤ 9999999 ∧
  (10^7 : ℕ) * x + y = 3 * x * y →
  x = 3333333 ∧ y = 3333334 := by
sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l1683_168349


namespace NUMINAMATH_CALUDE_art_kit_student_ratio_is_two_to_one_l1683_168383

/-- Represents the art class scenario --/
structure ArtClass where
  students : ℕ
  art_kits : ℕ
  total_artworks : ℕ

/-- Calculates the ratio of art kits to students --/
def art_kit_student_ratio (ac : ArtClass) : Rat :=
  ac.art_kits / ac.students

/-- Theorem stating the ratio of art kits to students is 2:1 --/
theorem art_kit_student_ratio_is_two_to_one (ac : ArtClass) 
  (h1 : ac.students = 10)
  (h2 : ac.art_kits = 20)
  (h3 : ac.total_artworks = 35)
  (h4 : ∃ (n : ℕ), 2 * n = ac.students ∧ 
       n * 3 + n * 4 = ac.total_artworks) : 
  art_kit_student_ratio ac = 2 := by
  sorry

end NUMINAMATH_CALUDE_art_kit_student_ratio_is_two_to_one_l1683_168383


namespace NUMINAMATH_CALUDE_two_lines_theorem_l1683_168303

/-- Two lines in the plane -/
structure TwoLines where
  l₁ : ℝ → ℝ → ℝ
  l₂ : ℝ → ℝ → ℝ
  a : ℝ
  b : ℝ
  h₁ : ∀ x y, l₁ x y = a * x - b * y + 4
  h₂ : ∀ x y, l₂ x y = (a - 1) * x + y + 2

/-- Scenario 1: l₁ passes through (-3,-1) and is perpendicular to l₂ -/
def scenario1 (lines : TwoLines) : Prop :=
  lines.l₁ (-3) (-1) = 0 ∧ 
  (lines.a / lines.b) * (1 - lines.a) = -1

/-- Scenario 2: l₁ is parallel to l₂ and has y-intercept -3 -/
def scenario2 (lines : TwoLines) : Prop :=
  lines.a / lines.b = 1 - lines.a ∧
  4 / lines.b = -3

theorem two_lines_theorem (lines : TwoLines) :
  (scenario1 lines → lines.a = 2 ∧ lines.b = 2) ∧
  (scenario2 lines → lines.a = 4 ∧ lines.b = -4/3) := by
  sorry

end NUMINAMATH_CALUDE_two_lines_theorem_l1683_168303


namespace NUMINAMATH_CALUDE_quadratic_sum_l1683_168302

/-- The quadratic expression 20x^2 + 240x + 3200 can be written as a(x+b)^2+c -/
def quadratic (x : ℝ) : ℝ := 20*x^2 + 240*x + 3200

/-- The completed square form of the quadratic -/
def completed_square (x a b c : ℝ) : ℝ := a*(x+b)^2 + c

theorem quadratic_sum : 
  ∃ (a b c : ℝ), (∀ x, quadratic x = completed_square x a b c) ∧ (a + b + c = 2506) := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1683_168302


namespace NUMINAMATH_CALUDE_product_equals_sum_of_squares_l1683_168323

theorem product_equals_sum_of_squares 
  (nums : List ℕ) 
  (count : nums.length = 116) 
  (sum_of_squares : (nums.map (λ x => x^2)).sum = 144) : 
  nums.prod = 144 := by
sorry

end NUMINAMATH_CALUDE_product_equals_sum_of_squares_l1683_168323


namespace NUMINAMATH_CALUDE_function_passes_through_point_l1683_168309

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 2
  f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l1683_168309


namespace NUMINAMATH_CALUDE_elise_puzzle_cost_l1683_168314

def puzzle_cost (initial_money savings comic_cost final_money : ℕ) : ℕ :=
  initial_money + savings - comic_cost - final_money

theorem elise_puzzle_cost : puzzle_cost 8 13 2 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_elise_puzzle_cost_l1683_168314


namespace NUMINAMATH_CALUDE_no_prime_multiple_of_four_in_range_l1683_168381

theorem no_prime_multiple_of_four_in_range : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 50 → ¬(4 ∣ n ∧ Nat.Prime n ∧ n > 10) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_multiple_of_four_in_range_l1683_168381


namespace NUMINAMATH_CALUDE_smallest_bob_number_l1683_168350

def alice_number : Nat := 30

def has_all_prime_factors_of (n m : Nat) : Prop :=
  ∀ p : Nat, Nat.Prime p → p ∣ n → p ∣ m

def has_additional_prime_factor (n m : Nat) : Prop :=
  ∃ p : Nat, Nat.Prime p ∧ p ∣ m ∧ ¬(p ∣ n)

theorem smallest_bob_number :
  ∃ bob_number : Nat,
    has_all_prime_factors_of alice_number bob_number ∧
    has_additional_prime_factor alice_number bob_number ∧
    (∀ m : Nat, m < bob_number →
      ¬(has_all_prime_factors_of alice_number m ∧
        has_additional_prime_factor alice_number m)) ∧
    bob_number = 210 := by
  sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l1683_168350


namespace NUMINAMATH_CALUDE_amalie_coin_spending_l1683_168355

/-- Proof that Amalie spends 3/4 of her coins on toys -/
theorem amalie_coin_spending :
  ∀ (elsa_coins amalie_coins : ℕ),
    -- The ratio of Elsa's coins to Amalie's coins is 10:45
    elsa_coins * 45 = amalie_coins * 10 →
    -- The total number of coins they have is 440
    elsa_coins + amalie_coins = 440 →
    -- Amalie remains with 90 coins after spending
    ∃ (spent_coins : ℕ),
      spent_coins ≤ amalie_coins ∧
      amalie_coins - spent_coins = 90 →
    -- The fraction of coins Amalie spends on toys is 3/4
    (spent_coins : ℚ) / amalie_coins = 3 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_amalie_coin_spending_l1683_168355


namespace NUMINAMATH_CALUDE_cyclist_return_speed_l1683_168395

/-- Proves that given the conditions of the cyclist's trip, the average speed for the return trip is 9 miles per hour. -/
theorem cyclist_return_speed (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) : 
  total_distance = 36 →
  speed1 = 12 →
  speed2 = 10 →
  total_time = 7.3 →
  (total_distance / speed1 + total_distance / speed2 + total_distance / 9 = total_time) := by
sorry

end NUMINAMATH_CALUDE_cyclist_return_speed_l1683_168395


namespace NUMINAMATH_CALUDE_rectangle_side_lengths_l1683_168305

/-- Given a rectangle DRAK with area 44, rectangle DUPE with area 64,
    and polygon DUPLAK with area 92, this theorem proves that there are
    only three possible sets of integer side lengths for the polygon. -/
theorem rectangle_side_lengths :
  ∀ (dr de du dk pl la : ℕ),
    dr * de = 16 →
    dr * dk = 44 →
    du * de = 64 →
    dk - de = la →
    du - dr = pl →
    (dr, de, du, dk, pl, la) ∈ ({(1, 16, 4, 44, 3, 28), (2, 8, 8, 22, 6, 14), (4, 4, 16, 11, 12, 7)} : Set (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_side_lengths_l1683_168305


namespace NUMINAMATH_CALUDE_website_earnings_l1683_168346

/-- John's website earnings problem -/
theorem website_earnings (visits_per_month : ℕ) (days_per_month : ℕ) (earnings_per_visit : ℚ)
  (h1 : visits_per_month = 30000)
  (h2 : days_per_month = 30)
  (h3 : earnings_per_visit = 1 / 100) :
  (visits_per_month : ℚ) * earnings_per_visit / days_per_month = 10 := by
  sorry

end NUMINAMATH_CALUDE_website_earnings_l1683_168346


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1683_168343

theorem cyclic_sum_inequality (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (sum_one : x + y + z = 1) :
  x^2 * y + y^2 * z + z^2 * x ≤ 4/27 ∧ 
  (x^2 * y + y^2 * z + z^2 * x = 4/27 ↔ 
    ((x = 2/3 ∧ y = 1/3 ∧ z = 0) ∨ 
     (x = 0 ∧ y = 2/3 ∧ z = 1/3) ∨ 
     (x = 1/3 ∧ y = 0 ∧ z = 2/3))) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1683_168343


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1683_168317

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space using the general form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b + l1.b * l2.a = 0

-- Define a point being on a line
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- The main theorem
theorem perpendicular_line_through_point :
  let P : Point2D := ⟨1, -2⟩
  let given_line : Line2D := ⟨1, -3, 2⟩
  let result_line : Line2D := ⟨3, 1, -1⟩
  perpendicular given_line result_line ∧ point_on_line P result_line := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1683_168317


namespace NUMINAMATH_CALUDE_some_number_exists_l1683_168377

theorem some_number_exists : ∃ N : ℝ, 
  (2 * ((3.6 * 0.48 * N) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002) ∧ 
  (abs (N - 2.5) < 0.0000000000000005) := by
  sorry

end NUMINAMATH_CALUDE_some_number_exists_l1683_168377


namespace NUMINAMATH_CALUDE_probability_ace_of_hearts_fifth_l1683_168354

-- Define the number of cards in a standard deck
def standard_deck_size : ℕ := 52

-- Define a function to calculate the probability of a specific card in a specific position
def probability_specific_card_in_position (deck_size : ℕ) : ℚ :=
  1 / deck_size

-- Theorem statement
theorem probability_ace_of_hearts_fifth : 
  probability_specific_card_in_position standard_deck_size = 1 / 52 := by
  sorry

end NUMINAMATH_CALUDE_probability_ace_of_hearts_fifth_l1683_168354


namespace NUMINAMATH_CALUDE_total_profit_calculation_l1683_168364

/-- Calculates the total profit given investments and one partner's share --/
def calculate_total_profit (anand_investment deepak_investment deepak_share : ℚ) : ℚ :=
  let total_parts := anand_investment + deepak_investment
  let deepak_parts := deepak_investment
  deepak_share * total_parts / deepak_parts

/-- The total profit is 1380.48 given the investments and Deepak's share --/
theorem total_profit_calculation (anand_investment deepak_investment deepak_share : ℚ) 
  (h1 : anand_investment = 2250)
  (h2 : deepak_investment = 3200)
  (h3 : deepak_share = 810.28) :
  calculate_total_profit anand_investment deepak_investment deepak_share = 1380.48 := by
  sorry

#eval calculate_total_profit 2250 3200 810.28

end NUMINAMATH_CALUDE_total_profit_calculation_l1683_168364


namespace NUMINAMATH_CALUDE_dennis_rocks_l1683_168384

theorem dennis_rocks (initial_rocks : ℕ) : 
  (initial_rocks / 2 + 2 = 7) → initial_rocks = 10 := by
sorry

end NUMINAMATH_CALUDE_dennis_rocks_l1683_168384


namespace NUMINAMATH_CALUDE_system_solutions_l1683_168330

theorem system_solutions :
  let solutions := [(2, 1), (0, -3), (-6, 9)]
  ∀ (x y : ℝ),
    (x + |y| = 3 ∧ 2*|x| - y = 3) ↔ (x, y) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l1683_168330


namespace NUMINAMATH_CALUDE_original_price_satisfies_conditions_l1683_168316

/-- The original price of merchandise satisfying given conditions -/
def original_price : ℝ := 175

/-- The loss when sold at 60% of the original price -/
def loss_at_60_percent : ℝ := 20

/-- The gain when sold at 80% of the original price -/
def gain_at_80_percent : ℝ := 15

/-- Theorem stating that the original price satisfies the given conditions -/
theorem original_price_satisfies_conditions : 
  (0.6 * original_price + loss_at_60_percent = 0.8 * original_price - gain_at_80_percent) := by
  sorry

end NUMINAMATH_CALUDE_original_price_satisfies_conditions_l1683_168316


namespace NUMINAMATH_CALUDE_bertha_family_without_children_l1683_168366

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The number of Bertha's daughters and granddaughters without daughters -/
def daughters_without_children (f : BerthaFamily) : ℕ :=
  f.total_descendants - f.daughters_with_children

/-- Theorem stating the number of Bertha's daughters and granddaughters without daughters -/
theorem bertha_family_without_children (f : BerthaFamily) 
  (h1 : f.daughters = 5)
  (h2 : f.total_descendants = 25)
  (h3 : f.daughters_with_children * 5 = f.total_descendants - f.daughters) :
  daughters_without_children f = 21 := by
  sorry

#check bertha_family_without_children

end NUMINAMATH_CALUDE_bertha_family_without_children_l1683_168366
