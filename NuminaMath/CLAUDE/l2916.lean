import Mathlib

namespace NUMINAMATH_CALUDE_number_of_people_is_fifteen_l2916_291608

theorem number_of_people_is_fifteen (x : ℕ) (y : ℕ) : 
  (12 * x + 3 = y) → 
  (13 * x - 12 = y) → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_number_of_people_is_fifteen_l2916_291608


namespace NUMINAMATH_CALUDE_unique_quadratic_with_prime_roots_l2916_291622

theorem unique_quadratic_with_prime_roots (a : ℝ) (ha : a > 0) :
  (∃! k : ℝ, ∃ p q : ℕ, 
    Prime p ∧ Prime q ∧ 
    (∀ x : ℝ, x^2 + (k^2 + a*k)*x + (1999 + k^2 + a*k) = 0 ↔ x = p ∨ x = q)) ↔ 
  a = 2 * Real.sqrt 502 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_with_prime_roots_l2916_291622


namespace NUMINAMATH_CALUDE_unique_sums_count_l2916_291606

def bag_A : Finset ℕ := {1, 4, 5, 8}
def bag_B : Finset ℕ := {2, 3, 7, 9}

theorem unique_sums_count : 
  Finset.card ((bag_A.product bag_B).image (λ (p : ℕ × ℕ) => p.1 + p.2)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_sums_count_l2916_291606


namespace NUMINAMATH_CALUDE_substitution_ways_soccer_l2916_291698

/-- The number of ways a coach can make substitutions in a soccer game -/
def substitution_ways (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  let substitutes := total_players - starting_players
  let no_sub := 1
  let one_sub := starting_players * substitutes
  let two_sub := one_sub * (starting_players - 1) * (substitutes + 1)
  let three_sub := two_sub * (starting_players - 2) * (substitutes + 2)
  let four_sub := three_sub * (starting_players - 3) * (substitutes + 3)
  (no_sub + one_sub + two_sub + three_sub + four_sub) % 1000

theorem substitution_ways_soccer : 
  substitution_ways 25 14 4 = 
  (1 + 14 * 11 + 14 * 11 * 13 * 12 + 14 * 11 * 13 * 12 * 12 * 13 + 
   14 * 11 * 13 * 12 * 12 * 13 * 11 * 14) % 1000 := by
  sorry

end NUMINAMATH_CALUDE_substitution_ways_soccer_l2916_291698


namespace NUMINAMATH_CALUDE_other_divisor_of_h_l2916_291647

def is_divisor (d n : ℕ) : Prop := n % d = 0

theorem other_divisor_of_h (h a b c : ℕ) : 
  h > 0 →
  is_divisor 225 h →
  h = 2^a * 3^b * 5^c →
  a > 0 →
  b > 0 →
  c > 0 →
  a + b + c ≥ 8 →
  (∀ a' b' c' : ℕ, a' > 0 → b' > 0 → c' > 0 → a' + b' + c' < a + b + c → ¬(h = 2^a' * 3^b' * 5^c')) →
  ∃ d : ℕ, d ≠ 225 ∧ is_divisor d h ∧ d = 16 :=
by sorry

end NUMINAMATH_CALUDE_other_divisor_of_h_l2916_291647


namespace NUMINAMATH_CALUDE_cos_225_degrees_l2916_291665

theorem cos_225_degrees : Real.cos (225 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l2916_291665


namespace NUMINAMATH_CALUDE_mary_remaining_sheep_l2916_291603

/-- Calculates the number of sheep Mary has left after distributing to her relatives --/
def remaining_sheep (initial : ℕ) : ℕ :=
  let after_sister := initial - (initial / 4)
  let after_brother := after_sister - (after_sister / 3)
  after_brother - (after_brother / 6)

/-- Theorem stating that Mary will have 500 sheep remaining --/
theorem mary_remaining_sheep :
  remaining_sheep 1200 = 500 := by
  sorry

end NUMINAMATH_CALUDE_mary_remaining_sheep_l2916_291603


namespace NUMINAMATH_CALUDE_abs_inequality_l2916_291667

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l2916_291667


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2916_291634

/-- The transformation from y = -2x^2 + 4x + 1 to y = -2x^2 -/
theorem quadratic_transformation (f g : ℝ → ℝ) (h_f : f = λ x => -2*x^2 + 4*x + 1) (h_g : g = λ x => -2*x^2) : 
  (∃ (a b : ℝ), ∀ x, f x = g (x + a) + b) ∧ 
  (∃ (vertex_f vertex_g : ℝ × ℝ), 
    vertex_f = (1, 3) ∧ 
    vertex_g = (0, 0) ∧ 
    vertex_f.1 - vertex_g.1 = 1 ∧ 
    vertex_f.2 - vertex_g.2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2916_291634


namespace NUMINAMATH_CALUDE_last_draw_same_color_prob_l2916_291671

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 2

/-- Represents the number of colors of marbles -/
def num_colors : ℕ := 3

/-- Represents the total number of marbles -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- Represents the number of people drawing marbles -/
def num_people : ℕ := 3

/-- Represents the number of marbles each person draws -/
def marbles_per_draw : ℕ := 2

/-- Calculates the probability of the last person drawing two marbles of the same color -/
def prob_last_draw_same_color : ℚ :=
  (num_colors * (Nat.choose (total_marbles - 2 * marbles_per_draw) marbles_per_draw)) /
  (Nat.choose total_marbles marbles_per_draw * 
   Nat.choose (total_marbles - marbles_per_draw) marbles_per_draw)

theorem last_draw_same_color_prob :
  prob_last_draw_same_color = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_last_draw_same_color_prob_l2916_291671


namespace NUMINAMATH_CALUDE_m_minus_reciprocal_l2916_291677

theorem m_minus_reciprocal (m : ℝ) (h : m^2 + 3*m = -1) : m - 1/(m+1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_m_minus_reciprocal_l2916_291677


namespace NUMINAMATH_CALUDE_square_roots_problem_l2916_291664

theorem square_roots_problem (m : ℝ) (a : ℝ) (h1 : m > 0) 
  (h2 : (a + 6)^2 = m) (h3 : (2*a - 9)^2 = m) :
  a = 1 ∧ m = 49 ∧ ∀ x : ℝ, a*x^2 - 16 = 0 ↔ x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l2916_291664


namespace NUMINAMATH_CALUDE_distance_between_points_l2916_291683

theorem distance_between_points : Real.sqrt ((8 - 2)^2 + (-5 - 3)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2916_291683


namespace NUMINAMATH_CALUDE_age_difference_l2916_291626

/-- Given the ages of Taehyung, his father, and his mother, prove that the age difference
between the father and mother is equal to Taehyung's age. -/
theorem age_difference (taehyung_age : ℕ) (father_age : ℕ) (mother_age : ℕ)
  (h1 : taehyung_age = 9)
  (h2 : father_age = 5 * taehyung_age)
  (h3 : mother_age = 4 * taehyung_age) :
  father_age - mother_age = taehyung_age :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l2916_291626


namespace NUMINAMATH_CALUDE_john_completion_time_l2916_291696

/-- The number of days it takes for Rose to complete the work alone -/
def rose_days : ℝ := 480

/-- The number of days it takes for John and Rose to complete the work together -/
def joint_days : ℝ := 192

/-- The number of days it takes for John to complete the work alone -/
def john_days : ℝ := 320

/-- Theorem stating that given Rose's and joint completion times, John's completion time is 320 days -/
theorem john_completion_time : 
  (1 / john_days + 1 / rose_days = 1 / joint_days) → john_days = 320 :=
by sorry

end NUMINAMATH_CALUDE_john_completion_time_l2916_291696


namespace NUMINAMATH_CALUDE_rectangular_box_sum_l2916_291660

theorem rectangular_box_sum (A B C : ℝ) 
  (h1 : A * B = 30)
  (h2 : A * C = 50)
  (h3 : B * C = 90) :
  A + B + C = 58 * Real.sqrt 15 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_sum_l2916_291660


namespace NUMINAMATH_CALUDE_second_diff_constant_correct_y_value_l2916_291694

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the sequence of x values
def x_seq (x₁ d : ℝ) (n : ℕ) : ℝ := x₁ + n * d

-- Define the sequence of y values
def y_seq (a b c x₁ d : ℝ) (n : ℕ) : ℝ := quadratic a b c (x_seq x₁ d n)

-- Define the first difference sequence
def delta_seq (a b c x₁ d : ℝ) (n : ℕ) : ℝ :=
  y_seq a b c x₁ d (n + 1) - y_seq a b c x₁ d n

-- Define the second difference sequence
def delta2_seq (a b c x₁ d : ℝ) (n : ℕ) : ℝ :=
  delta_seq a b c x₁ d (n + 1) - delta_seq a b c x₁ d n

-- Theorem: The second difference is constant
theorem second_diff_constant (a b c x₁ d : ℝ) (h : a ≠ 0) :
  ∃ k, ∀ n, delta2_seq a b c x₁ d n = k :=
sorry

-- Given y values
def given_y_values : List ℝ := [51, 107, 185, 285, 407, 549, 717]

-- Find the incorrect y value and its correct value
def find_incorrect_y (ys : List ℝ) : Option (ℕ × ℝ) :=
sorry

-- Theorem: The identified incorrect y value is 549 and should be 551
theorem correct_y_value :
  find_incorrect_y given_y_values = some (5, 551) :=
sorry

end NUMINAMATH_CALUDE_second_diff_constant_correct_y_value_l2916_291694


namespace NUMINAMATH_CALUDE_parallelograms_in_divided_triangle_l2916_291684

/-- The number of parallelograms formed in a triangle with sides divided into n equal parts -/
def num_parallelograms (n : ℕ) : ℕ :=
  3 * (Nat.choose (n + 2) 4)

/-- Theorem stating the number of parallelograms in a divided triangle -/
theorem parallelograms_in_divided_triangle (n : ℕ) :
  num_parallelograms n = 3 * (Nat.choose (n + 2) 4) :=
by sorry

end NUMINAMATH_CALUDE_parallelograms_in_divided_triangle_l2916_291684


namespace NUMINAMATH_CALUDE_orange_probability_l2916_291624

/-- Given a box of fruit with apples and oranges, calculate the probability of selecting an orange -/
theorem orange_probability (apples oranges : ℕ) (h1 : apples = 20) (h2 : oranges = 10) :
  (oranges : ℚ) / ((apples : ℚ) + (oranges : ℚ)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_probability_l2916_291624


namespace NUMINAMATH_CALUDE_first_tv_width_l2916_291679

/-- Proves that the width of the first TV is 24 inches given the specified conditions. -/
theorem first_tv_width : 
  ∀ (W : ℝ),
  (672 / (W * 16) = 1152 / (48 * 32) + 1) →
  W = 24 := by
sorry

end NUMINAMATH_CALUDE_first_tv_width_l2916_291679


namespace NUMINAMATH_CALUDE_intersection_trajectory_l2916_291676

/-- The trajectory of the intersection point of two rotating rods -/
theorem intersection_trajectory (a : ℝ) (h : a ≠ 0) :
  ∃ (x y : ℝ), 
    (∃ (b b₁ : ℝ), b * b₁ = a^2 ∧ b ≠ 0 ∧ b₁ ≠ 0) →
    (y = -b / a * (x - a) ∧ y = b₁ / a * (x + a)) →
    x^2 + y^2 = a^2 ∧ -a < x ∧ x < a :=
by sorry

end NUMINAMATH_CALUDE_intersection_trajectory_l2916_291676


namespace NUMINAMATH_CALUDE_expression_evaluation_l2916_291673

theorem expression_evaluation : -20 + 7 * ((8 - 2) / 3) = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2916_291673


namespace NUMINAMATH_CALUDE_circle_radius_zero_circle_equation_point_circle_radius_l2916_291621

theorem circle_radius_zero (x y : ℝ) :
  x^2 + 8*x + y^2 - 10*y + 41 = 0 → (x + 4)^2 + (y - 5)^2 = 0 :=
by sorry

theorem circle_equation_point (x y : ℝ) :
  (x + 4)^2 + (y - 5)^2 = 0 → x = -4 ∧ y = 5 :=
by sorry

theorem circle_radius (x y : ℝ) :
  x^2 + 8*x + y^2 - 10*y + 41 = 0 → ∃! (center : ℝ × ℝ), center = (-4, 5) ∧ (x - center.1)^2 + (y - center.2)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_zero_circle_equation_point_circle_radius_l2916_291621


namespace NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l2916_291670

theorem sandy_marks_per_correct_sum :
  ∀ (total_sums : ℕ) (total_marks : ℤ) (correct_sums : ℕ) (marks_lost_per_incorrect : ℤ) (marks_per_correct : ℤ),
    total_sums = 30 →
    total_marks = 60 →
    correct_sums = 24 →
    marks_lost_per_incorrect = 2 →
    (marks_per_correct * correct_sums : ℤ) - (marks_lost_per_incorrect * (total_sums - correct_sums) : ℤ) = total_marks →
    marks_per_correct = 3 :=
by sorry

end NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l2916_291670


namespace NUMINAMATH_CALUDE_det_dilation_matrix_l2916_291650

/-- A 3x3 matrix representing a dilation with scale factor 5 -/
def E : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ => 5)

/-- Theorem stating that the determinant of E is 125 -/
theorem det_dilation_matrix : Matrix.det E = 125 := by
  sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_l2916_291650


namespace NUMINAMATH_CALUDE_range_of_f_l2916_291691

-- Define the function f(x) = -x²
def f (x : ℝ) : ℝ := -x^2

-- Define the domain
def domain : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }

-- Theorem statement
theorem range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | -9 ≤ y ∧ y ≤ 0 } :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l2916_291691


namespace NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l2916_291646

theorem square_sum_from_product_and_sum (x y : ℝ) 
  (h1 : x * y = 12) 
  (h2 : x + y = 10) : 
  x^2 + y^2 = 76 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l2916_291646


namespace NUMINAMATH_CALUDE_total_fat_served_l2916_291695

/-- The amount of fat in ounces for a single herring -/
def herring_fat : ℕ := 40

/-- The amount of fat in ounces for a single eel -/
def eel_fat : ℕ := 20

/-- The amount of fat in ounces for a single pike -/
def pike_fat : ℕ := eel_fat + 10

/-- The number of fish of each type served -/
def fish_count : ℕ := 40

/-- The total amount of fat served in ounces -/
def total_fat : ℕ := fish_count * (herring_fat + eel_fat + pike_fat)

theorem total_fat_served :
  total_fat = 3600 := by sorry

end NUMINAMATH_CALUDE_total_fat_served_l2916_291695


namespace NUMINAMATH_CALUDE_negation_of_existence_l2916_291623

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2916_291623


namespace NUMINAMATH_CALUDE_min_a_value_l2916_291637

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 ≤ 0}

-- Define the set B
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}

-- Theorem statement
theorem min_a_value (a : ℝ) : 
  (A a ∪ B = A a) → a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l2916_291637


namespace NUMINAMATH_CALUDE_remainder_98_pow_24_mod_100_l2916_291663

theorem remainder_98_pow_24_mod_100 : 98^24 % 100 = 16 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_pow_24_mod_100_l2916_291663


namespace NUMINAMATH_CALUDE_power_division_negative_x_l2916_291675

theorem power_division_negative_x (x : ℝ) : (-x)^8 / (-x)^4 = x^4 := by sorry

end NUMINAMATH_CALUDE_power_division_negative_x_l2916_291675


namespace NUMINAMATH_CALUDE_vectors_linearly_dependent_iff_l2916_291662

/-- Two vectors in ℝ² -/
def v1 : Fin 2 → ℝ := ![2, 5]
def v2 (m : ℝ) : Fin 2 → ℝ := ![4, m]

/-- Definition of linear dependence for two vectors -/
def linearlyDependent (u v : Fin 2 → ℝ) : Prop :=
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (∀ i, a * u i + b * v i = 0)

/-- Theorem: The vectors v1 and v2 are linearly dependent iff m = 10 -/
theorem vectors_linearly_dependent_iff (m : ℝ) :
  linearlyDependent v1 (v2 m) ↔ m = 10 := by
  sorry

end NUMINAMATH_CALUDE_vectors_linearly_dependent_iff_l2916_291662


namespace NUMINAMATH_CALUDE_omega_value_for_max_sine_l2916_291699

theorem omega_value_for_max_sine (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 1) :
  (∀ x ∈ Set.Icc 0 (π / 3), 2 * Real.sin (ω * x) ≤ Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (π / 3), 2 * Real.sin (ω * x) = Real.sqrt 2) →
  ω = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_omega_value_for_max_sine_l2916_291699


namespace NUMINAMATH_CALUDE_g_range_l2916_291688

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x/3))^2 + (Real.pi/4) * Real.arcsin (x/3) - (Real.arcsin (x/3))^2 + (Real.pi^2/16) * (x^2 + 2*x + 3)

theorem g_range : 
  ∀ x ∈ Set.Icc (-3 : ℝ) 3, 
    g x ∈ Set.Icc (Real.pi^2/4) ((15*Real.pi^2/16) + (Real.pi/4)*Real.arcsin 1) ∧
    ∃ y ∈ Set.Icc (-3 : ℝ) 3, g y = Real.pi^2/4 ∧
    ∃ z ∈ Set.Icc (-3 : ℝ) 3, g z = (15*Real.pi^2/16) + (Real.pi/4)*Real.arcsin 1 :=
by sorry

end NUMINAMATH_CALUDE_g_range_l2916_291688


namespace NUMINAMATH_CALUDE_least_multiple_of_35_greater_than_450_l2916_291692

theorem least_multiple_of_35_greater_than_450 : ∀ n : ℕ, n > 0 ∧ 35 ∣ n ∧ n > 450 → n ≥ 455 := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_35_greater_than_450_l2916_291692


namespace NUMINAMATH_CALUDE_slope_of_line_l2916_291693

/-- The slope of the line 4x + 7y = 28 is -4/7 -/
theorem slope_of_line (x y : ℝ) : 4 * x + 7 * y = 28 → (y - 4) / x = -4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l2916_291693


namespace NUMINAMATH_CALUDE_cone_volume_from_slant_and_height_l2916_291638

/-- The volume of a cone given its slant height and height --/
theorem cone_volume_from_slant_and_height 
  (slant_height : ℝ) 
  (height : ℝ) 
  (h_slant : slant_height = 15) 
  (h_height : height = 9) : 
  (1/3 : ℝ) * Real.pi * (slant_height^2 - height^2) * height = 432 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_slant_and_height_l2916_291638


namespace NUMINAMATH_CALUDE_weeks_passed_l2916_291672

/-- Prove that the number of weeks that have already passed is 4 --/
theorem weeks_passed
  (watch_cost : ℕ)
  (weekly_allowance : ℕ)
  (current_savings : ℕ)
  (weeks_left : ℕ)
  (h1 : watch_cost = 100)
  (h2 : weekly_allowance = 5)
  (h3 : current_savings = 20)
  (h4 : weeks_left = 16)
  (h5 : current_savings + weeks_left * weekly_allowance = watch_cost) :
  current_savings / weekly_allowance = 4 := by
  sorry


end NUMINAMATH_CALUDE_weeks_passed_l2916_291672


namespace NUMINAMATH_CALUDE_existence_of_special_set_l2916_291656

theorem existence_of_special_set (n : ℕ) (p : ℕ) (h_n : n ≥ 2) (h_p : Nat.Prime p) (h_div : p ∣ n) :
  ∃ (A : Fin n → ℕ), ∀ (i j : Fin n) (S : Finset (Fin n)), S.card = p →
    (A i * A j) ∣ (S.sum A) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_set_l2916_291656


namespace NUMINAMATH_CALUDE_sum_of_digits_of_triangular_array_rows_l2916_291602

/-- The number of coins in a triangular array with n rows -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem stating that the sum of digits of N is 14, where N is the number of rows in a triangular array containing 3003 coins -/
theorem sum_of_digits_of_triangular_array_rows :
  ∃ N : ℕ, triangular_sum N = 3003 ∧ sum_of_digits N = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_triangular_array_rows_l2916_291602


namespace NUMINAMATH_CALUDE_max_pieces_is_72_l2916_291651

/-- Represents a rectangular cake with dimensions m and n -/
structure Cake where
  m : ℕ+
  n : ℕ+

/-- Calculates the number of pieces in the two central rows -/
def central_pieces (c : Cake) : ℕ := (c.m - 4) * (c.n - 4)

/-- Calculates the number of pieces on the perimeter -/
def perimeter_pieces (c : Cake) : ℕ := 2 * c.m + 2 * c.n - 4

/-- Checks if the cake satisfies the chef's condition -/
def satisfies_condition (c : Cake) : Prop :=
  central_pieces c = perimeter_pieces c

/-- Calculates the total number of pieces -/
def total_pieces (c : Cake) : ℕ := c.m * c.n

/-- States that the maximum number of pieces satisfying the condition is 72 -/
theorem max_pieces_is_72 :
  ∃ (c : Cake), satisfies_condition c ∧
    total_pieces c = 72 ∧
    ∀ (c' : Cake), satisfies_condition c' → total_pieces c' ≤ 72 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_is_72_l2916_291651


namespace NUMINAMATH_CALUDE_hannah_grocery_cost_l2916_291681

theorem hannah_grocery_cost 
  (total_cost : ℝ)
  (cookie_price : ℝ)
  (carrot_price : ℝ)
  (cabbage_price : ℝ)
  (orange_price : ℝ)
  (h1 : cookie_price + carrot_price + cabbage_price + orange_price = total_cost)
  (h2 : orange_price = 3 * cookie_price)
  (h3 : cabbage_price = cookie_price - carrot_price)
  (h4 : total_cost = 24) :
  carrot_price + cabbage_price = 24 / 5 := by
sorry

end NUMINAMATH_CALUDE_hannah_grocery_cost_l2916_291681


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2916_291658

def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (N a ⊆ M) ↔ (a = -1 ∨ a = 0 ∨ a = 2/3) :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2916_291658


namespace NUMINAMATH_CALUDE_f_lower_bound_a_range_l2916_291643

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x + 1| + 2 * |x + 2|

-- Theorem for part (I)
theorem f_lower_bound : ∀ x : ℝ, f x ≥ 5 := by sorry

-- Theorem for part (II)
theorem a_range (a : ℝ) : 
  (∀ x : ℝ, 15 - 2 * f x < a^2 + 9 / (a^2 + 1)) ↔ a ≠ Real.sqrt 2 ∧ a ≠ -Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_a_range_l2916_291643


namespace NUMINAMATH_CALUDE_apple_purchase_cost_l2916_291612

/-- The price of apples per kilogram before discount -/
def original_price : ℝ := 5

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.4

/-- The quantity of apples in kilograms -/
def quantity : ℝ := 10

/-- Calculates the discounted price per kilogram -/
def discounted_price : ℝ := original_price * (1 - discount_rate)

/-- Calculates the total cost for the given quantity of apples -/
def total_cost : ℝ := discounted_price * quantity

/-- Theorem stating that the total cost for 10 kilograms of apples with a 40% discount is $30 -/
theorem apple_purchase_cost : total_cost = 30 := by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_cost_l2916_291612


namespace NUMINAMATH_CALUDE_sum_of_diagonals_l2916_291666

/-- A hexagon inscribed in a circle -/
structure InscribedHexagon where
  -- Sides of the hexagon
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side6 : ℝ
  -- Diagonals from vertex A
  diag1 : ℝ
  diag2 : ℝ
  diag3 : ℝ
  -- Assumption that the hexagon is inscribed in a circle
  inscribed : True

/-- The theorem about the sum of diagonals in a specific inscribed hexagon -/
theorem sum_of_diagonals (h : InscribedHexagon) 
    (h1 : h.side1 = 70)
    (h2 : h.side2 = 90)
    (h3 : h.side3 = 90)
    (h4 : h.side4 = 90)
    (h5 : h.side5 = 90)
    (h6 : h.side6 = 50) :
    h.diag1 + h.diag2 + h.diag3 = 376 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_diagonals_l2916_291666


namespace NUMINAMATH_CALUDE_larger_number_problem_l2916_291686

theorem larger_number_problem (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2916_291686


namespace NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l2916_291641

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function f(x) = 1 + log_a(x-1)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + log a (x - 1)

-- State the theorem
theorem fixed_point_of_logarithmic_function (a : ℝ) (h : a > 0 ∧ a ≠ 1) : f a 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l2916_291641


namespace NUMINAMATH_CALUDE_nabla_calculation_l2916_291618

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem nabla_calculation : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l2916_291618


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l2916_291609

def M : Set ℝ := {x | 0 < x ∧ x ≤ 4}
def N : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) :=
by sorry

end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l2916_291609


namespace NUMINAMATH_CALUDE_max_value_of_b_l2916_291689

theorem max_value_of_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1 / b - 1 / a) :
  b ≤ 1 / 3 ∧ ∃ (b₀ : ℝ), b₀ > 0 ∧ b₀ = 1 / 3 ∧ ∃ (a₀ : ℝ), a₀ > 0 ∧ a₀ + 3 * b₀ = 1 / b₀ - 1 / a₀ :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_b_l2916_291689


namespace NUMINAMATH_CALUDE_magic_trick_result_l2916_291652

theorem magic_trick_result (x : ℚ) : ((2 * x + 8) / 4) - (x / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_magic_trick_result_l2916_291652


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2916_291620

theorem contrapositive_equivalence (p q : Prop) :
  (q → p) → (¬p → ¬q) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2916_291620


namespace NUMINAMATH_CALUDE_percentage_comparison_l2916_291619

theorem percentage_comparison (base : ℝ) (first second : ℝ) 
  (h1 : first = base * 1.71)
  (h2 : second = base * 1.80) :
  first / second * 100 = 95 := by
  sorry

end NUMINAMATH_CALUDE_percentage_comparison_l2916_291619


namespace NUMINAMATH_CALUDE_initial_average_calculation_l2916_291611

theorem initial_average_calculation (n : ℕ) (correct_sum wrong_sum : ℝ) 
  (h1 : n = 10)
  (h2 : correct_sum / n = 24)
  (h3 : wrong_sum = correct_sum - 10) :
  wrong_sum / n = 23 := by
sorry

end NUMINAMATH_CALUDE_initial_average_calculation_l2916_291611


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l2916_291644

/-- The equation of a hyperbola with parameter k -/
def hyperbola_equation (x y k : ℝ) : Prop :=
  x^2 / (1 - 2*k) - y^2 / (k - 2) = 1

/-- The condition that the hyperbola has foci on the y-axis -/
def foci_on_y_axis (k : ℝ) : Prop :=
  1 - 2*k < 0 ∧ k - 2 < 0

/-- Theorem: If the equation represents a hyperbola with foci on the y-axis,
    then k is in the open interval (1/2, 2) -/
theorem hyperbola_k_range (k : ℝ) :
  (∃ x y : ℝ, hyperbola_equation x y k) →
  foci_on_y_axis k →
  k ∈ Set.Ioo (1/2 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l2916_291644


namespace NUMINAMATH_CALUDE_circle_radii_in_square_l2916_291605

theorem circle_radii_in_square (r : ℝ) : 
  r > 0 →  -- radius is positive
  r < 1/4 →  -- each circle fits in a corner
  (∀ (i j : Fin 4), i ≠ j → 
    (∃ (x y : ℝ), x^2 + y^2 = (2*r)^2 ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1)) →  -- circles touch
  (∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = (2*r)^2 ∧
      (x₂ - x₃)^2 + (y₂ - y₃)^2 = (2*r)^2 ∧
      (x₃ - x₁)^2 + (y₃ - y₁)^2 > (2*r)^2)) →  -- only two circles touch each other
  1 - Real.sqrt 2 / 2 < r ∧ r < 2 - Real.sqrt 2 / 2 - Real.sqrt (4 - 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radii_in_square_l2916_291605


namespace NUMINAMATH_CALUDE_largest_four_digit_multiple_of_9_with_digit_sum_27_l2916_291697

/-- Returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if n is a four-digit number, false otherwise -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_four_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 9990 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_multiple_of_9_with_digit_sum_27_l2916_291697


namespace NUMINAMATH_CALUDE_supermarket_spending_l2916_291669

theorem supermarket_spending (total : ℚ) (candy : ℚ) : 
  total = 24 →
  (1/4 : ℚ) * total + (1/3 : ℚ) * total + (1/6 : ℚ) * total + candy = total →
  candy = 8 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_l2916_291669


namespace NUMINAMATH_CALUDE_toms_dog_age_l2916_291616

theorem toms_dog_age (brother_age dog_age : ℕ) : 
  brother_age = 4 * dog_age →
  brother_age + 6 = 30 →
  dog_age + 6 = 12 := by
sorry

end NUMINAMATH_CALUDE_toms_dog_age_l2916_291616


namespace NUMINAMATH_CALUDE_store_clearance_sale_l2916_291625

/-- Calculates the amount owed to creditors after a store's clearance sale --/
theorem store_clearance_sale 
  (total_items : ℕ) 
  (original_price : ℝ) 
  (discount_percent : ℝ) 
  (sold_percent : ℝ) 
  (remaining_amount : ℝ) 
  (h1 : total_items = 2000)
  (h2 : original_price = 50)
  (h3 : discount_percent = 0.8)
  (h4 : sold_percent = 0.9)
  (h5 : remaining_amount = 3000) : 
  (total_items : ℝ) * sold_percent * (original_price * (1 - discount_percent)) - remaining_amount = 15000 := by
  sorry

end NUMINAMATH_CALUDE_store_clearance_sale_l2916_291625


namespace NUMINAMATH_CALUDE_game_a_more_likely_than_game_b_l2916_291649

def prob_heads : ℚ := 3/4
def prob_tails : ℚ := 1/4

def prob_game_a : ℚ := prob_heads^4

def prob_game_b : ℚ := (prob_heads * prob_tails)^3

theorem game_a_more_likely_than_game_b : prob_game_a > prob_game_b := by
  sorry

end NUMINAMATH_CALUDE_game_a_more_likely_than_game_b_l2916_291649


namespace NUMINAMATH_CALUDE_notebook_purchase_possible_l2916_291636

theorem notebook_purchase_possible : ∃ x y : ℤ, 16 * x + 27 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_notebook_purchase_possible_l2916_291636


namespace NUMINAMATH_CALUDE_bank_coin_value_l2916_291607

/-- Proves that the total value of coins in a bank is 555 cents -/
theorem bank_coin_value : 
  let total_coins : ℕ := 70
  let nickel_count : ℕ := 29
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  let dime_count : ℕ := total_coins - nickel_count
  total_coins = nickel_count + dime_count →
  nickel_count * nickel_value + dime_count * dime_value = 555 := by
  sorry

end NUMINAMATH_CALUDE_bank_coin_value_l2916_291607


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2916_291617

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -3) : x^3 + y^3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2916_291617


namespace NUMINAMATH_CALUDE_sets_inclusion_l2916_291668

-- Define the sets M, N, and P
def M : Set ℝ := {θ | ∃ k : ℤ, θ = k * Real.pi / 4}
def N : Set ℝ := {x | Real.cos (2 * x) = 0}
def P : Set ℝ := {a | Real.sin (2 * a) = 1}

-- State the theorem
theorem sets_inclusion : P ⊆ N ∧ N ⊆ M := by sorry

end NUMINAMATH_CALUDE_sets_inclusion_l2916_291668


namespace NUMINAMATH_CALUDE_faster_speed_proof_l2916_291680

/-- Proves that the faster speed is 12 kmph given the problem conditions -/
theorem faster_speed_proof (distance : ℝ) (slow_speed : ℝ) (late_time : ℝ) (early_time : ℝ) 
  (h1 : distance = 24)
  (h2 : slow_speed = 9)
  (h3 : late_time = 1/3)  -- 20 minutes in hours
  (h4 : early_time = 1/3) -- 20 minutes in hours
  : ∃ (fast_speed : ℝ), 
    distance / slow_speed - distance / fast_speed = late_time + early_time ∧ 
    fast_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_faster_speed_proof_l2916_291680


namespace NUMINAMATH_CALUDE_fishbowl_count_l2916_291631

theorem fishbowl_count (total_fish : ℕ) (fish_per_bowl : ℕ) (h1 : total_fish = 6003) (h2 : fish_per_bowl = 23) :
  total_fish / fish_per_bowl = 261 := by
sorry

end NUMINAMATH_CALUDE_fishbowl_count_l2916_291631


namespace NUMINAMATH_CALUDE_function_value_comparison_l2916_291640

theorem function_value_comparison (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : 0 < a ∧ a < 3) 
  (hx : x₁ < x₂) 
  (hsum : x₁ + x₂ = 1 - a) : 
  let f := fun x => a * x^2 + 2 * a * x + 4
  f x₁ < f x₂ := by
sorry

end NUMINAMATH_CALUDE_function_value_comparison_l2916_291640


namespace NUMINAMATH_CALUDE_complex_product_quadrant_l2916_291642

theorem complex_product_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_product_quadrant_l2916_291642


namespace NUMINAMATH_CALUDE_difference_of_squares_l2916_291613

theorem difference_of_squares (a b : ℝ) : (a - b) * (-b - a) = b^2 - a^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2916_291613


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2916_291659

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 8 / x + 1 / y = 1) : x + 2 * y ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2916_291659


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_90_l2916_291690

/-- The number of ways to distribute 5 college students among 3 freshman classes -/
def allocation_schemes : ℕ :=
  let n_students : ℕ := 5
  let n_classes : ℕ := 3
  let min_per_class : ℕ := 1
  let max_per_class : ℕ := 2
  -- The actual calculation is not implemented, just returning the correct result
  90

/-- Theorem stating that the number of allocation schemes is 90 -/
theorem allocation_schemes_eq_90 : allocation_schemes = 90 := by
  -- The proof is not implemented
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_eq_90_l2916_291690


namespace NUMINAMATH_CALUDE_tree_space_calculation_l2916_291627

/-- The space taken up by each tree in square feet -/
def tree_space : ℝ := 1

/-- The total length of the road in feet -/
def road_length : ℝ := 148

/-- The number of trees to be planted -/
def num_trees : ℕ := 8

/-- The space between each tree in feet -/
def space_between : ℝ := 20

theorem tree_space_calculation :
  tree_space * num_trees + space_between * (num_trees - 1) = road_length := by
  sorry

end NUMINAMATH_CALUDE_tree_space_calculation_l2916_291627


namespace NUMINAMATH_CALUDE_power_sum_inequality_l2916_291645

theorem power_sum_inequality (a b c : ℝ) (n : ℕ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hn : 0 < n) :
  a^n + b^n + c^n ≥ a * b^(n-1) + b * c^(n-1) + c * a^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l2916_291645


namespace NUMINAMATH_CALUDE_no_solution_exists_l2916_291682

theorem no_solution_exists : ¬∃ (x : ℤ), x^2 = 3*x + 75 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2916_291682


namespace NUMINAMATH_CALUDE_square_with_inscribed_semicircles_l2916_291614

theorem square_with_inscribed_semicircles (square_side : ℝ) (semicircle_count : ℕ) : 
  square_side = 4 → 
  semicircle_count = 4 → 
  (square_side^2 - semicircle_count * (π * (square_side/2)^2 / 2)) = 16 - 8*π := by
sorry

end NUMINAMATH_CALUDE_square_with_inscribed_semicircles_l2916_291614


namespace NUMINAMATH_CALUDE_square_roots_and_cube_root_l2916_291632

theorem square_roots_and_cube_root (x a : ℝ) (hx : x > 0) : 
  ((2*a - 1)^2 = x ∧ (-a + 2)^2 = x ∧ 2*a - 1 ≠ -a + 2) →
  (a = -1 ∧ x = 9 ∧ (4*x + 9*a)^(1/3) = 3) :=
by sorry

end NUMINAMATH_CALUDE_square_roots_and_cube_root_l2916_291632


namespace NUMINAMATH_CALUDE_inequality_solution_l2916_291639

theorem inequality_solution (x : ℝ) : 
  (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ↔ 
  (0 < x ∧ x ≤ 1/4) ∨ (1 < x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2916_291639


namespace NUMINAMATH_CALUDE_smartphone_price_proof_l2916_291661

def laptop_price : ℕ := 600
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def total_paid : ℕ := 3000
def change_received : ℕ := 200

theorem smartphone_price_proof :
  ∃ (smartphone_price : ℕ),
    smartphone_price * num_smartphones + laptop_price * num_laptops = total_paid - change_received ∧
    smartphone_price = 400 := by
  sorry

end NUMINAMATH_CALUDE_smartphone_price_proof_l2916_291661


namespace NUMINAMATH_CALUDE_unique_fraction_l2916_291628

theorem unique_fraction : ∃! (m n : ℕ), 
  m < 10 ∧ n < 10 ∧ 
  n = m^2 - 1 ∧
  (m + 2 : ℚ) / (n + 2) > 1/3 ∧
  (m - 3 : ℚ) / (n - 3) < 1/10 ∧
  m = 3 ∧ n = 8 := by
sorry

end NUMINAMATH_CALUDE_unique_fraction_l2916_291628


namespace NUMINAMATH_CALUDE_function_range_theorem_l2916_291633

open Real

theorem function_range_theorem (a : ℝ) (m n p : ℝ) : 
  let f := fun (x : ℝ) => -x^3 + 3*x + a
  (m ≠ n ∧ n ≠ p ∧ m ≠ p) →
  (f m = 2022 ∧ f n = 2022 ∧ f p = 2022) →
  (2020 < a ∧ a < 2024) := by
sorry

end NUMINAMATH_CALUDE_function_range_theorem_l2916_291633


namespace NUMINAMATH_CALUDE_problem_body_surface_area_l2916_291654

/-- Represents a three-dimensional geometric body -/
structure GeometricBody where
  -- Add necessary fields to represent the geometric body
  -- This is a placeholder as we don't have specific information about the structure

/-- Calculates the surface area of a geometric body -/
noncomputable def surfaceArea (body : GeometricBody) : ℝ :=
  sorry -- Actual calculation would go here

/-- Represents the specific geometric body from the problem -/
def problemBody : GeometricBody :=
  sorry -- Construction of the specific body would go here

/-- Theorem stating that the surface area of the problem's geometric body is 40 -/
theorem problem_body_surface_area :
    surfaceArea problemBody = 40 := by
  sorry

end NUMINAMATH_CALUDE_problem_body_surface_area_l2916_291654


namespace NUMINAMATH_CALUDE_peas_corn_difference_l2916_291629

/-- The number of cans of peas Beth bought -/
def peas : ℕ := 35

/-- The number of cans of corn Beth bought -/
def corn : ℕ := 10

/-- The difference between the number of cans of peas and twice the number of cans of corn -/
def difference : ℕ := peas - 2 * corn

theorem peas_corn_difference : difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_peas_corn_difference_l2916_291629


namespace NUMINAMATH_CALUDE_sandy_age_l2916_291630

theorem sandy_age (sandy_age molly_age : ℕ) : 
  molly_age = sandy_age + 12 →
  sandy_age * 9 = molly_age * 7 →
  sandy_age = 42 := by
sorry

end NUMINAMATH_CALUDE_sandy_age_l2916_291630


namespace NUMINAMATH_CALUDE_total_eggs_is_63_l2916_291678

/-- The number of Easter eggs Hannah found -/
def hannah_eggs : ℕ := 42

/-- The number of Easter eggs Helen found -/
def helen_eggs : ℕ := hannah_eggs / 2

/-- The total number of Easter eggs in the yard -/
def total_eggs : ℕ := hannah_eggs + helen_eggs

/-- Theorem stating that the total number of Easter eggs in the yard is 63 -/
theorem total_eggs_is_63 : total_eggs = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_is_63_l2916_291678


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2916_291655

/-- Given a triangle with inradius 3 cm and area 30 cm², its perimeter is 20 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 3 → A = 30 → A = r * (p / 2) → p = 20 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2916_291655


namespace NUMINAMATH_CALUDE_square_from_triangles_even_count_l2916_291674

-- Define the triangle type
structure Triangle :=
  (side1 : ℕ)
  (side2 : ℕ)
  (side3 : ℕ)

-- Define the properties of our specific triangle
def SpecificTriangle : Triangle :=
  { side1 := 3, side2 := 4, side3 := 5 }

-- Define the area of the triangle
def triangleArea (t : Triangle) : ℚ :=
  (t.side1 * t.side2 : ℚ) / 2

-- Define a function to check if a number is even
def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Main theorem
theorem square_from_triangles_even_count :
  ∀ n : ℕ, n > 0 →
  (∃ a : ℕ, a > 0 ∧ (a : ℚ)^2 = n * triangleArea SpecificTriangle) →
  isEven n :=
sorry

end NUMINAMATH_CALUDE_square_from_triangles_even_count_l2916_291674


namespace NUMINAMATH_CALUDE_sum_of_products_l2916_291657

theorem sum_of_products : 64 * 46 + 73 * 37 + 82 * 28 + 91 * 19 = 9670 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l2916_291657


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l2916_291648

theorem acute_triangle_properties (A B C a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle
  A + B + C = π ∧  -- Sum of angles in a triangle
  Real.sqrt ((1 - Real.cos (2 * C)) / 2) + Real.sin (B - A) = 2 * Real.sin (2 * A) ∧  -- Given equation
  c ≥ max a b ∧  -- AB is the longest side
  a = Real.sin A ∧ b = Real.sin B ∧ c = Real.sin C  -- Law of sines
  →
  a / b = 1 / 2 ∧ 0 < Real.cos C ∧ Real.cos C ≤ 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l2916_291648


namespace NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l2916_291685

theorem permutations_of_eight_distinct_objects : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l2916_291685


namespace NUMINAMATH_CALUDE_basketball_team_average_weight_l2916_291653

/-- Given a basketball team with boys and girls, calculate the average weight of all players. -/
theorem basketball_team_average_weight 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (avg_weight_boys : ℚ) 
  (avg_weight_girls : ℚ) 
  (h_num_boys : num_boys = 8) 
  (h_num_girls : num_girls = 5) 
  (h_avg_weight_boys : avg_weight_boys = 160) 
  (h_avg_weight_girls : avg_weight_girls = 130) : 
  (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 148 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_average_weight_l2916_291653


namespace NUMINAMATH_CALUDE_baby_grasshoppers_count_l2916_291635

/-- The number of grasshoppers Jemma found on the African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := 31

/-- The number of baby grasshoppers under the plant -/
def baby_grasshoppers : ℕ := total_grasshoppers - grasshoppers_on_plant

theorem baby_grasshoppers_count : baby_grasshoppers = 24 := by
  sorry

end NUMINAMATH_CALUDE_baby_grasshoppers_count_l2916_291635


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2916_291615

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 1 ∧ 
  (∀ (m : ℕ), (13294 - m) % 97 = 0 → m ≥ n) ∧
  (13294 - n) % 97 = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2916_291615


namespace NUMINAMATH_CALUDE_least_bench_sections_l2916_291687

/-- Represents the capacity of a single bench section -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Proves that 6 is the least positive integer N such that N bench sections
    can hold an equal number of adults and children -/
theorem least_bench_sections (capacity : BenchCapacity)
    (h_adults : capacity.adults = 8)
    (h_children : capacity.children = 12) :
    (∃ N : Nat, N > 0 ∧
      ∃ x : Nat, x > 0 ∧
        N * capacity.adults = x ∧
        N * capacity.children = x ∧
        ∀ M : Nat, M > 0 → M < N →
          ¬∃ y : Nat, y > 0 ∧
            M * capacity.adults = y ∧
            M * capacity.children = y) →
    (∃ N : Nat, N = 6 ∧ N > 0 ∧
      ∃ x : Nat, x > 0 ∧
        N * capacity.adults = x ∧
        N * capacity.children = x ∧
        ∀ M : Nat, M > 0 → M < N →
          ¬∃ y : Nat, y > 0 ∧
            M * capacity.adults = y ∧
            M * capacity.children = y) :=
  sorry

end NUMINAMATH_CALUDE_least_bench_sections_l2916_291687


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2916_291604

theorem largest_angle_in_pentagon (A B C D E : ℝ) : 
  A = 70 → 
  B = 120 → 
  C = D → 
  E = 3 * C - 30 → 
  A + B + C + D + E = 540 → 
  max A (max B (max C (max D E))) = 198 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2916_291604


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2916_291601

/-- A geometric sequence with common ratio r -/
def geometricSequence (r : ℝ) : ℕ → ℝ := fun n => r^(n-1)

/-- The third term of a geometric sequence -/
def a₃ (r : ℝ) : ℝ := geometricSequence r 3

/-- The seventh term of a geometric sequence -/
def a₇ (r : ℝ) : ℝ := geometricSequence r 7

/-- The fifth term of a geometric sequence -/
def a₅ (r : ℝ) : ℝ := geometricSequence r 5

theorem geometric_sequence_fifth_term (r : ℝ) :
  (a₃ r)^2 - 4*(a₃ r) + 3 = 0 ∧ (a₇ r)^2 - 4*(a₇ r) + 3 = 0 → a₅ r = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2916_291601


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2916_291610

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 4 ∧ (9671 - k) % 5 = 0 ∧ ∀ (m : ℕ), m < k → (9671 - m) % 5 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2916_291610


namespace NUMINAMATH_CALUDE_blue_candy_count_l2916_291600

theorem blue_candy_count (total : ℕ) (red : ℕ) (h1 : total = 11567) (h2 : red = 792) :
  total - red = 10775 := by
  sorry

end NUMINAMATH_CALUDE_blue_candy_count_l2916_291600
