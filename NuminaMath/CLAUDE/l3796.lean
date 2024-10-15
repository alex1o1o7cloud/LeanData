import Mathlib

namespace NUMINAMATH_CALUDE_davis_class_groups_l3796_379626

/-- The number of groups in Miss Davis's class -/
def number_of_groups (sticks_per_group : ℕ) (initial_sticks : ℕ) (remaining_sticks : ℕ) : ℕ :=
  (initial_sticks - remaining_sticks) / sticks_per_group

/-- Theorem stating the number of groups in Miss Davis's class -/
theorem davis_class_groups :
  number_of_groups 15 170 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_davis_class_groups_l3796_379626


namespace NUMINAMATH_CALUDE_sticker_distribution_l3796_379613

/-- The number of ways to partition n identical objects into at most k parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 30 ways to partition 10 identical objects into at most 5 parts -/
theorem sticker_distribution : partition_count 10 5 = 30 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l3796_379613


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_17_l3796_379667

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (t.b - 5)^2 + |t.c - 7| = 0 ∧ |t.a - 3| = 2

-- Define the perimeter
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Theorem statement
theorem triangle_perimeter_is_17 :
  ∀ t : Triangle, satisfies_conditions t → perimeter t = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_17_l3796_379667


namespace NUMINAMATH_CALUDE_selling_price_loss_l3796_379625

/-- Represents the ratio of selling price to cost price -/
def price_ratio : ℚ := 2 / 5

/-- The loss percentage when selling price is less than cost price -/
def loss_percent (r : ℚ) : ℚ := (1 - r) * 100

theorem selling_price_loss :
  price_ratio = 2 / 5 →
  loss_percent price_ratio = 60 := by
sorry

end NUMINAMATH_CALUDE_selling_price_loss_l3796_379625


namespace NUMINAMATH_CALUDE_solutions_count_3x_2y_802_l3796_379688

theorem solutions_count_3x_2y_802 : 
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 2 * p.2 = 802 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 803) (Finset.range 402))).card = 133 := by
  sorry

end NUMINAMATH_CALUDE_solutions_count_3x_2y_802_l3796_379688


namespace NUMINAMATH_CALUDE_camp_children_count_l3796_379690

/-- The number of children currently in the camp -/
def current_children : ℕ := 25

/-- The percentage of boys currently in the camp -/
def boys_percentage : ℚ := 85/100

/-- The number of boys to be added -/
def boys_added : ℕ := 50

/-- The desired percentage of girls after adding boys -/
def desired_girls_percentage : ℚ := 5/100

theorem camp_children_count :
  (boys_percentage * current_children).num = 
    (desired_girls_percentage * (current_children + boys_added)).num * 
    ((1 - boys_percentage) * current_children).den := by sorry

end NUMINAMATH_CALUDE_camp_children_count_l3796_379690


namespace NUMINAMATH_CALUDE_powderman_distance_approximation_l3796_379660

/-- The speed of the powderman in yards per second -/
def powderman_speed : ℝ := 8

/-- The time in seconds when the powderman hears the blast -/
def time_of_hearing : ℝ := 30.68

/-- The distance the powderman runs in yards -/
def distance_run : ℝ := powderman_speed * time_of_hearing

theorem powderman_distance_approximation :
  ∃ ε > 0, abs (distance_run - 245) < ε := by sorry

end NUMINAMATH_CALUDE_powderman_distance_approximation_l3796_379660


namespace NUMINAMATH_CALUDE_correct_decision_probability_l3796_379608

-- Define the probability of a consultant giving a correct opinion
def p_correct : ℝ := 0.8

-- Define the number of consultants
def n_consultants : ℕ := 3

-- Define the probability of making a correct decision
def p_correct_decision : ℝ :=
  (Nat.choose n_consultants 2) * p_correct^2 * (1 - p_correct) +
  (Nat.choose n_consultants 3) * p_correct^3

-- Theorem statement
theorem correct_decision_probability :
  p_correct_decision = 0.896 := by sorry

end NUMINAMATH_CALUDE_correct_decision_probability_l3796_379608


namespace NUMINAMATH_CALUDE_incorrect_propositions_are_one_and_three_l3796_379698

-- Define a proposition as a structure with an id and a correctness value
structure Proposition :=
  (id : Nat)
  (isCorrect : Bool)

-- Define our set of propositions
def propositions : List Proposition := [
  ⟨1, false⟩,  -- Three points determine a plane
  ⟨2, true⟩,   -- A rectangle is a plane figure
  ⟨3, false⟩,  -- Three lines intersecting in pairs determine a plane
  ⟨4, true⟩    -- Two intersecting planes divide the space into four regions
]

-- Define a function to get incorrect propositions
def getIncorrectPropositions (props : List Proposition) : List Nat :=
  (props.filter (λ p => !p.isCorrect)).map Proposition.id

-- Theorem statement
theorem incorrect_propositions_are_one_and_three :
  getIncorrectPropositions propositions = [1, 3] := by
  sorry

end NUMINAMATH_CALUDE_incorrect_propositions_are_one_and_three_l3796_379698


namespace NUMINAMATH_CALUDE_trivia_team_groups_l3796_379664

theorem trivia_team_groups (total : ℕ) (not_picked : ℕ) (num_groups : ℕ) 
  (h1 : total = 17) 
  (h2 : not_picked = 5) 
  (h3 : num_groups = 3) :
  (total - not_picked) / num_groups = 4 :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_groups_l3796_379664


namespace NUMINAMATH_CALUDE_fraction_product_subtraction_l3796_379692

theorem fraction_product_subtraction : (1/2 : ℚ) * (1/3 : ℚ) * (1/6 : ℚ) * 72 - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_subtraction_l3796_379692


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3796_379646

-- Define the propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, |x| + |x - 1| > m
def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(5 - 2*m)^x) > (-(5 - 2*m)^y)

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, p m ∧ q m) ∧ (∃ m : ℝ, q m ∧ ¬(p m)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3796_379646


namespace NUMINAMATH_CALUDE_infinite_k_sin_k_greater_than_C_l3796_379647

theorem infinite_k_sin_k_greater_than_C :
  ∀ C : ℝ, ∃ S : Set ℤ, (Set.Infinite S) ∧ (∀ k ∈ S, (k : ℝ) * Real.sin k > C) := by
  sorry

end NUMINAMATH_CALUDE_infinite_k_sin_k_greater_than_C_l3796_379647


namespace NUMINAMATH_CALUDE_land_tax_calculation_l3796_379665

/-- Calculates the land tax for a given plot --/
def calculate_land_tax (area : ℝ) (cadastral_value_per_acre : ℝ) (tax_rate : ℝ) : ℝ :=
  area * cadastral_value_per_acre * tax_rate

/-- Proves that the land tax for the given conditions is 4500 rubles --/
theorem land_tax_calculation :
  let area : ℝ := 15
  let cadastral_value_per_acre : ℝ := 100000
  let tax_rate : ℝ := 0.003
  calculate_land_tax area cadastral_value_per_acre tax_rate = 4500 := by
  sorry

#eval calculate_land_tax 15 100000 0.003

end NUMINAMATH_CALUDE_land_tax_calculation_l3796_379665


namespace NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l3796_379640

theorem triangle_angle_not_all_greater_than_60 :
  ¬ ∀ (a b c : ℝ), 
    (a > 0) → (b > 0) → (c > 0) → 
    (a + b + c = 180) → 
    (a > 60 ∧ b > 60 ∧ c > 60) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_not_all_greater_than_60_l3796_379640


namespace NUMINAMATH_CALUDE_solution_pairs_l3796_379699

theorem solution_pairs : ∃! (s : Set (ℝ × ℝ)), 
  s = {(1 + Real.sqrt 2, 1 - Real.sqrt 2), (1 - Real.sqrt 2, 1 + Real.sqrt 2)} ∧
  ∀ (x y : ℝ), (x, y) ∈ s ↔ 
    (x^2 + y^2 = (6 - x^2) + (6 - y^2)) ∧ 
    (x^2 - y^2 = (x - 2)^2 + (y - 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_solution_pairs_l3796_379699


namespace NUMINAMATH_CALUDE_block_distance_is_200_l3796_379644

/-- The distance of one time around the block -/
def block_distance : ℝ := sorry

/-- The number of times Johnny runs around the block -/
def johnny_laps : ℕ := 4

/-- The number of times Mickey runs around the block -/
def mickey_laps : ℕ := johnny_laps / 2

/-- The average distance run by Johnny and Mickey -/
def average_distance : ℝ := 600

theorem block_distance_is_200 :
  block_distance = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_block_distance_is_200_l3796_379644


namespace NUMINAMATH_CALUDE_expand_square_root_two_l3796_379612

theorem expand_square_root_two (a b : ℚ) : (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2 → a + b = 70 := by
  sorry

end NUMINAMATH_CALUDE_expand_square_root_two_l3796_379612


namespace NUMINAMATH_CALUDE_division_by_three_remainder_l3796_379685

theorem division_by_three_remainder (n : ℤ) : 
  (n % 3 ≠ 0) → (n % 3 = 1 ∨ n % 3 = 2) :=
by sorry

end NUMINAMATH_CALUDE_division_by_three_remainder_l3796_379685


namespace NUMINAMATH_CALUDE_white_animals_count_l3796_379666

theorem white_animals_count (total : ℕ) (black : ℕ) (white : ℕ) : 
  total = 13 → black = 6 → white = total - black → white = 7 := by sorry

end NUMINAMATH_CALUDE_white_animals_count_l3796_379666


namespace NUMINAMATH_CALUDE_parabola_vertex_l3796_379605

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := y^2 - 8*x + 6*y + 17 = 0

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, -3)

/-- Theorem: The vertex of the parabola y^2 - 8x + 6y + 17 = 0 is at the point (1, -3) -/
theorem parabola_vertex : 
  ∀ x y : ℝ, parabola_eq x y → (x, y) = vertex ∨ ∃ t : ℝ, parabola_eq (x + t) y :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3796_379605


namespace NUMINAMATH_CALUDE_sqrt_11_diamond_sqrt_11_l3796_379614

-- Define the ¤ operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt_11_diamond_sqrt_11 : diamond (Real.sqrt 11) (Real.sqrt 11) = 44 := by sorry

end NUMINAMATH_CALUDE_sqrt_11_diamond_sqrt_11_l3796_379614


namespace NUMINAMATH_CALUDE_hyperbola_and_slopes_l3796_379671

-- Define the hyperbola E
def E (b : ℝ) (x y : ℝ) : Prop := x^2 - y^2/b^2 = 1

-- Define point P
def P : ℝ × ℝ := (-2, -3)

-- Define point Q
def Q : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem hyperbola_and_slopes 
  (b : ℝ) 
  (h1 : b > 0) 
  (h2 : E b P.1 P.2) 
  (A B : ℝ × ℝ) 
  (h3 : A ≠ P ∧ B ≠ P ∧ A ≠ B) 
  (h4 : ∃ k : ℝ, A.2 = k * A.1 - 1 ∧ B.2 = k * B.1 - 1) 
  (h5 : E b A.1 A.2 ∧ E b B.1 B.2) :
  (b^2 = 3) ∧ 
  (((A.2 - P.2) / (A.1 - P.1)) + ((B.2 - P.2) / (B.1 - P.1)) = 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_and_slopes_l3796_379671


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_square_imaginary_part_of_one_minus_two_i_squared_l3796_379693

theorem imaginary_part_of_complex_square : ℂ → ℝ
  | ⟨re, im⟩ => im

theorem imaginary_part_of_one_minus_two_i_squared :
  imaginary_part_of_complex_square ((1 - 2 * Complex.I) ^ 2) = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_square_imaginary_part_of_one_minus_two_i_squared_l3796_379693


namespace NUMINAMATH_CALUDE_square_sum_difference_l3796_379687

theorem square_sum_difference (n : ℕ) : n^2 + (n+1)^2 - (n+2)^2 = n*(n-2) - 3 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_difference_l3796_379687


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3796_379638

theorem quadratic_roots_sum_of_squares (a b s p : ℝ) : 
  a^2 + b^2 = 15 → 
  s = a + b → 
  p = a * b → 
  (∀ x, x^2 - s*x + p = 0 ↔ x = a ∨ x = b) → 
  15 = s^2 - 2*p := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l3796_379638


namespace NUMINAMATH_CALUDE_sum_of_costs_equals_power_l3796_379691

/-- An antipalindromic sequence of A's and B's -/
def AntipalindromicSequence : Type := List Bool

/-- The cost of a sequence is the product of positions of A's -/
def cost (seq : AntipalindromicSequence) : ℕ := sorry

/-- The set of all antipalindromic sequences of length 2020 -/
def allAntipalindromic2020 : Set AntipalindromicSequence := sorry

/-- The sum of costs of all antipalindromic sequences of length 2020 -/
def sumOfCosts : ℕ := sorry

/-- Main theorem: The sum of costs equals 2021^1010 -/
theorem sum_of_costs_equals_power :
  sumOfCosts = 2021^1010 := by sorry

end NUMINAMATH_CALUDE_sum_of_costs_equals_power_l3796_379691


namespace NUMINAMATH_CALUDE_cookie_ratio_l3796_379643

/-- Given a total of 14 bags, 28 cookies, and 2 bags of cookies,
    prove that the ratio of cookies in each bag to the total number of cookies is 1:2 -/
theorem cookie_ratio (total_bags : ℕ) (total_cookies : ℕ) (cookie_bags : ℕ)
  (h1 : total_bags = 14)
  (h2 : total_cookies = 28)
  (h3 : cookie_bags = 2) :
  (total_cookies / cookie_bags) / total_cookies = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cookie_ratio_l3796_379643


namespace NUMINAMATH_CALUDE_sum_product_bounds_l3796_379683

theorem sum_product_bounds (x y z : ℝ) (h : x + y + z = 3) :
  -3/2 ≤ x*y + x*z + y*z ∧ x*y + x*z + y*z ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_product_bounds_l3796_379683


namespace NUMINAMATH_CALUDE_ned_short_sleeve_shirts_l3796_379653

/-- The number of short sleeve shirts Ned had to wash -/
def short_sleeve_shirts : ℕ := sorry

/-- The number of long sleeve shirts Ned had to wash -/
def long_sleeve_shirts : ℕ := 21

/-- The number of shirts Ned washed before school started -/
def washed_shirts : ℕ := 29

/-- The number of shirts Ned did not wash -/
def unwashed_shirts : ℕ := 1

/-- The total number of shirts Ned had to wash -/
def total_shirts : ℕ := washed_shirts + unwashed_shirts

theorem ned_short_sleeve_shirts :
  short_sleeve_shirts = total_shirts - long_sleeve_shirts :=
by sorry

end NUMINAMATH_CALUDE_ned_short_sleeve_shirts_l3796_379653


namespace NUMINAMATH_CALUDE_ping_pong_practice_time_l3796_379637

theorem ping_pong_practice_time 
  (total_students : ℕ) 
  (practicing_simultaneously : ℕ) 
  (total_time : ℕ) 
  (h1 : total_students = 5)
  (h2 : practicing_simultaneously = 2)
  (h3 : total_time = 90) :
  (total_time * practicing_simultaneously) / total_students = 36 :=
by sorry

end NUMINAMATH_CALUDE_ping_pong_practice_time_l3796_379637


namespace NUMINAMATH_CALUDE_tangent_length_l3796_379684

/-- The circle C with equation x^2 + y^2 - 2x - 6y + 9 = 0 -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 9 = 0

/-- The point P on the x-axis -/
def point_P : ℝ × ℝ := (1, 0)

/-- The length of the tangent from P to circle C is 2√2 -/
theorem tangent_length : 
  ∃ (t : ℝ × ℝ), 
    circle_C t.1 t.2 ∧ 
    ((t.1 - point_P.1)^2 + (t.2 - point_P.2)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_tangent_length_l3796_379684


namespace NUMINAMATH_CALUDE_max_value_expression_l3796_379697

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 4 - Real.sqrt (x^4 + 16)) / x ≤ 2 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3796_379697


namespace NUMINAMATH_CALUDE_square_sum_equals_one_l3796_379632

theorem square_sum_equals_one (x y : ℝ) :
  (x^2 + y^2 + 1)^2 - 4 = 0 → x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_one_l3796_379632


namespace NUMINAMATH_CALUDE_inequality_implication_l3796_379634

theorem inequality_implication (a b : ℝ) : a < b → -a + 3 > -b + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3796_379634


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3796_379629

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 6 - 3*x) → x ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3796_379629


namespace NUMINAMATH_CALUDE_pages_to_read_on_day_three_l3796_379645

theorem pages_to_read_on_day_three 
  (total_pages : ℕ) 
  (pages_day_one : ℕ) 
  (pages_day_two : ℕ) 
  (h1 : total_pages = 100)
  (h2 : pages_day_one = 35)
  (h3 : pages_day_two = pages_day_one - 5) :
  total_pages - (pages_day_one + pages_day_two) = 35 := by
  sorry

end NUMINAMATH_CALUDE_pages_to_read_on_day_three_l3796_379645


namespace NUMINAMATH_CALUDE_four_inch_cube_painted_faces_l3796_379609

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Counts the number of 1-inch cubes with at least two painted faces in a painted n×n×n cube -/
def count_painted_cubes (c : Cube n) : ℕ :=
  sorry

/-- Theorem: In a 4x4x4 painted cube, there are 56 1-inch cubes with at least two painted faces -/
theorem four_inch_cube_painted_faces :
  ∃ (c : Cube 4), count_painted_cubes c = 56 := by
  sorry

end NUMINAMATH_CALUDE_four_inch_cube_painted_faces_l3796_379609


namespace NUMINAMATH_CALUDE_intersection_condition_l3796_379642

theorem intersection_condition (m : ℝ) : 
  let A := {x : ℝ | x^2 - 3*x + 2 = 0}
  let C := {x : ℝ | x^2 - m*x + 2 = 0}
  (A ∩ C = C) → (m = 3 ∨ -2*Real.sqrt 2 < m ∧ m < 2*Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_condition_l3796_379642


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3796_379603

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x + x^2

theorem inequality_solution_set (x : ℝ) :
  (x ∈ Set.Ioo (Real.exp (-1)) (Real.exp 1)) ↔
  (f (Real.log x) + f (Real.log (1/x)) < 2 * f 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3796_379603


namespace NUMINAMATH_CALUDE_mode_is_highest_rectangle_middle_l3796_379600

/-- Represents a frequency distribution histogram --/
structure FrequencyHistogram where
  -- Add necessary fields here

/-- The mode of a frequency distribution --/
def mode (h : FrequencyHistogram) : ℝ :=
  sorry

/-- The middle position of the highest rectangle in a frequency histogram --/
def highestRectangleMiddle (h : FrequencyHistogram) : ℝ :=
  sorry

/-- Theorem stating that the mode corresponds to the middle of the highest rectangle --/
theorem mode_is_highest_rectangle_middle (h : FrequencyHistogram) :
  mode h = highestRectangleMiddle h :=
sorry

end NUMINAMATH_CALUDE_mode_is_highest_rectangle_middle_l3796_379600


namespace NUMINAMATH_CALUDE_bug_return_probability_l3796_379662

/-- Represents the probability of the bug being at its starting vertex after n moves -/
def P : ℕ → ℚ
| 0 => 1
| n + 1 => (2 : ℚ) / 3 * P n

/-- The probability of returning to the starting vertex on the 10th move -/
def probability_10th_move : ℚ := P 10

theorem bug_return_probability :
  probability_10th_move = 1024 / 59049 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l3796_379662


namespace NUMINAMATH_CALUDE_head_circumference_ratio_l3796_379623

theorem head_circumference_ratio :
  let jack_circumference : ℝ := 12
  let charlie_circumference : ℝ := 9 + (jack_circumference / 2)
  let bill_circumference : ℝ := 10
  bill_circumference / charlie_circumference = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_head_circumference_ratio_l3796_379623


namespace NUMINAMATH_CALUDE_min_value_theorem_l3796_379681

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / (x + 2)) + (1 / (y + 1)) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3796_379681


namespace NUMINAMATH_CALUDE_unique_integer_complex_sixth_power_l3796_379663

def complex_sixth_power_is_integer (n : ℤ) : Prop :=
  ∃ m : ℤ, (n + Complex.I) ^ 6 = m

theorem unique_integer_complex_sixth_power :
  ∃! n : ℤ, complex_sixth_power_is_integer n :=
sorry

end NUMINAMATH_CALUDE_unique_integer_complex_sixth_power_l3796_379663


namespace NUMINAMATH_CALUDE_zoo_total_revenue_l3796_379606

def monday_children : Nat := 7
def monday_adults : Nat := 5
def tuesday_children : Nat := 4
def tuesday_adults : Nat := 2
def child_ticket_cost : Nat := 3
def adult_ticket_cost : Nat := 4

theorem zoo_total_revenue : 
  (monday_children + tuesday_children) * child_ticket_cost + 
  (monday_adults + tuesday_adults) * adult_ticket_cost = 61 := by
  sorry

#eval (monday_children + tuesday_children) * child_ticket_cost + 
      (monday_adults + tuesday_adults) * adult_ticket_cost

end NUMINAMATH_CALUDE_zoo_total_revenue_l3796_379606


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3796_379604

/-- A trinomial of the form ax² + bx + c is a perfect square if there exist real numbers p and q
    such that ax² + bx + c = (px + q)² for all x. -/
def IsPerfectSquare (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial (k : ℝ) :
  IsPerfectSquare 4 k 9 → k = 12 ∨ k = -12 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3796_379604


namespace NUMINAMATH_CALUDE_quadratic_real_roots_iff_k_eq_one_l3796_379620

/-- 
A quadratic equation ax^2 + bx + c = 0 has real roots if and only if its discriminant b^2 - 4ac is non-negative.
-/
def has_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c ≥ 0

/--
Given the quadratic equation kx^2 - 3x + 2 = 0, where k is a non-negative integer,
the equation has real roots if and only if k = 1.
-/
theorem quadratic_real_roots_iff_k_eq_one :
  ∀ k : ℕ, has_real_roots k (-3) 2 ↔ k = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_iff_k_eq_one_l3796_379620


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_negative_seven_l3796_379679

theorem sqrt_expression_equals_negative_seven :
  (Real.sqrt 15)^2 / Real.sqrt 3 * (1 / Real.sqrt 3) - Real.sqrt 6 * Real.sqrt 24 = -7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_negative_seven_l3796_379679


namespace NUMINAMATH_CALUDE_exact_sequence_2007_l3796_379650

/-- An exact sequence of integers. -/
def ExactSequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n > m → a n ^ 2 - a m ^ 2 = a (n - m) * a (n + m)

/-- The 2007th term of the exact sequence with given initial conditions. -/
theorem exact_sequence_2007 (a : ℕ → ℤ) 
    (h_exact : ExactSequence a) 
    (h_init1 : a 1 = 1) 
    (h_init2 : a 2 = 0) : 
  a 2007 = -1 := by
  sorry

end NUMINAMATH_CALUDE_exact_sequence_2007_l3796_379650


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l3796_379601

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 4) (h2 : l^2 + w^2 = d^2) :
  l * w = (20 / 41) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l3796_379601


namespace NUMINAMATH_CALUDE_fraction_simplification_l3796_379668

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 3 + 5 * Real.sqrt 48) = (5 * Real.sqrt 3) / 84 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3796_379668


namespace NUMINAMATH_CALUDE_variance_best_for_stability_l3796_379618

/-- Represents a statistical measure -/
inductive StatMeasure
  | Mode
  | Variance
  | Mean
  | Frequency

/-- Represents an athlete's performance data -/
structure AthleteData where
  results : List Float
  len : Nat
  h_len : len = 10

/-- Assesses the stability of performance based on a statistical measure -/
def assessStability (measure : StatMeasure) (data : AthleteData) : Bool :=
  sorry

/-- Theorem stating that variance is the most suitable measure for assessing stability -/
theorem variance_best_for_stability (data : AthleteData) :
  ∀ (m : StatMeasure), m ≠ StatMeasure.Variance →
    assessStability StatMeasure.Variance data = true ∧
    assessStability m data = false :=
  sorry

end NUMINAMATH_CALUDE_variance_best_for_stability_l3796_379618


namespace NUMINAMATH_CALUDE_pen_distribution_l3796_379636

theorem pen_distribution (total_pencils : ℕ) (num_students : ℕ) (num_pens : ℕ) :
  total_pencils = 928 →
  num_students = 16 →
  total_pencils % num_students = 0 →
  num_pens % num_students = 0 →
  ∃ k : ℕ, num_pens = 16 * k :=
by sorry

end NUMINAMATH_CALUDE_pen_distribution_l3796_379636


namespace NUMINAMATH_CALUDE_illumination_theorem_l3796_379675

/-- Represents a direction: North, South, East, or West -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a spotlight with a position and direction -/
structure Spotlight where
  position : Point
  direction : Direction

/-- Represents the configuration of 4 spotlights -/
def SpotlightConfiguration := Fin 4 → Spotlight

/-- Checks if a point is illuminated by a spotlight -/
def isIlluminated (p : Point) (s : Spotlight) : Prop :=
  match s.direction with
  | Direction.North => p.y ≥ s.position.y
  | Direction.South => p.y ≤ s.position.y
  | Direction.East => p.x ≥ s.position.x
  | Direction.West => p.x ≤ s.position.x

/-- The main theorem: there exists a configuration that illuminates the entire plane -/
theorem illumination_theorem (p1 p2 p3 p4 : Point) :
  ∃ (config : SpotlightConfiguration),
    ∀ (p : Point), ∃ (i : Fin 4), isIlluminated p (config i) := by
  sorry


end NUMINAMATH_CALUDE_illumination_theorem_l3796_379675


namespace NUMINAMATH_CALUDE_union_complement_equality_l3796_379633

def I : Set ℤ := {x | x^2 < 9}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_complement_equality : A ∪ (I \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_complement_equality_l3796_379633


namespace NUMINAMATH_CALUDE_sum_of_g_and_h_l3796_379648

theorem sum_of_g_and_h (a b c d e f g h : ℝ) 
  (avg_abc : (a + b + c) / 3 = 103 / 3)
  (avg_def : (d + e + f) / 3 = 375 / 6)
  (avg_all : (a + b + c + d + e + f + g + h) / 8 = 23 / 2) :
  g + h = -198.5 := by sorry

end NUMINAMATH_CALUDE_sum_of_g_and_h_l3796_379648


namespace NUMINAMATH_CALUDE_original_photo_dimensions_l3796_379607

/-- Represents the dimensions of a rectangular photo frame --/
structure PhotoFrame where
  width : ℕ
  height : ℕ

/-- Calculates the number of squares needed for a frame --/
def squares_for_frame (frame : PhotoFrame) : ℕ :=
  2 * (frame.width + frame.height)

/-- Theorem stating the dimensions of the original photo --/
theorem original_photo_dimensions 
  (original_squares : ℕ) 
  (cut_squares : ℕ) 
  (h1 : original_squares = 1812)
  (h2 : cut_squares = 2018) :
  ∃ (frame : PhotoFrame), 
    squares_for_frame frame = original_squares ∧ 
    frame.width = 803 ∧ 
    frame.height = 101 ∧
    cut_squares - original_squares = 2 * frame.height :=
sorry


end NUMINAMATH_CALUDE_original_photo_dimensions_l3796_379607


namespace NUMINAMATH_CALUDE_extended_parallelepiped_volume_l3796_379624

/-- The volume of a set described by a rectangular parallelepiped extended by unit radius cylinders and spheres -/
theorem extended_parallelepiped_volume :
  let l : ℝ := 2  -- length
  let w : ℝ := 3  -- width
  let h : ℝ := 6  -- height
  let r : ℝ := 1  -- radius of extension

  -- Volume of the original parallelepiped
  let v_box := l * w * h

  -- Volume of outward projecting parallelepipeds
  let v_out := 2 * (r * w * h + r * l * h + r * l * w)

  -- Volume of quarter-cylinders along edges
  let edge_length := 2 * (l + w + h)
  let v_cyl := (π * r^2 / 4) * edge_length

  -- Volume of eighth-spheres at vertices
  let v_sph := 8 * ((4 / 3) * π * r^3 / 8)

  -- Total volume
  let v_total := v_box + v_out + v_cyl + v_sph

  v_total = (324 + 70 * π) / 3 :=
by sorry

end NUMINAMATH_CALUDE_extended_parallelepiped_volume_l3796_379624


namespace NUMINAMATH_CALUDE_intersection_problem_l3796_379680

/-- The problem statement as a theorem -/
theorem intersection_problem (m b k : ℝ) : 
  b ≠ 0 →
  7 = 2 * m + b →
  (∃ y₁ y₂ : ℝ, 
    y₁ = k^2 + 8*k + 7 ∧
    y₂ = m*k + b ∧
    |y₁ - y₂| = 4) →
  m = 6 ∧ b = -5 := by
sorry

end NUMINAMATH_CALUDE_intersection_problem_l3796_379680


namespace NUMINAMATH_CALUDE_square_root_of_1024_l3796_379630

theorem square_root_of_1024 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1024_l3796_379630


namespace NUMINAMATH_CALUDE_joseph_kyle_distance_difference_l3796_379641

theorem joseph_kyle_distance_difference : 
  let joseph_speed : ℝ := 50
  let joseph_time : ℝ := 2.5
  let kyle_speed : ℝ := 62
  let kyle_time : ℝ := 2
  let joseph_distance := joseph_speed * joseph_time
  let kyle_distance := kyle_speed * kyle_time
  joseph_distance - kyle_distance = 1 := by
sorry

end NUMINAMATH_CALUDE_joseph_kyle_distance_difference_l3796_379641


namespace NUMINAMATH_CALUDE_triangle_on_hyperbola_l3796_379616

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := y = 1 / x

-- Define a point on the hyperbola
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y

-- Define parallel lines
def parallel (p1 p2 p3 p4 : PointOnHyperbola) : Prop :=
  (p2.y - p1.y) / (p2.x - p1.x) = (p4.y - p3.y) / (p4.x - p3.x)

-- Define the theorem
theorem triangle_on_hyperbola
  (A B C A₁ B₁ C₁ : PointOnHyperbola)
  (h1 : parallel A B A₁ B₁)
  (h2 : parallel B C B₁ C₁) :
  parallel A C₁ A₁ C := by
  sorry

end NUMINAMATH_CALUDE_triangle_on_hyperbola_l3796_379616


namespace NUMINAMATH_CALUDE_abs_cube_complex_l3796_379610

/-- The absolute value of (3 + √7i)^3 is equal to 64, where i is the imaginary unit. -/
theorem abs_cube_complex : Complex.abs ((3 + Complex.I * Real.sqrt 7) ^ 3) = 64 := by
  sorry

end NUMINAMATH_CALUDE_abs_cube_complex_l3796_379610


namespace NUMINAMATH_CALUDE_parabola_properties_l3796_379615

/-- Parabola properties -/
theorem parabola_properties (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 - (a + 1) * x
  (f 2 = 0) →
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = -4 ∧ f x₁ = f x₂ → a = -1/5) ∧
  (∀ (x₁ x₂ : ℝ), x₁ > x₂ ∧ x₂ ≥ -2 ∧ f x₁ < f x₂ → -1/5 ≤ a ∧ a < 0) ∧
  (∃ (x : ℝ), x = 1 ∧ ∀ (y : ℝ), f (x + y) = f (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3796_379615


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l3796_379669

theorem quadratic_rewrite_sum (x : ℝ) : ∃ (a b c : ℝ),
  (6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) ∧ (a + b + c = 171) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l3796_379669


namespace NUMINAMATH_CALUDE_initial_chairs_count_l3796_379678

theorem initial_chairs_count (initial_chairs : ℕ) 
  (h1 : initial_chairs - (initial_chairs - 3) = 12) : initial_chairs = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_chairs_count_l3796_379678


namespace NUMINAMATH_CALUDE_total_ants_is_twenty_l3796_379654

/-- The number of ants found by Abe -/
def abe_ants : ℕ := 4

/-- The number of ants found by Beth -/
def beth_ants : ℕ := abe_ants + abe_ants / 2

/-- The number of ants found by CeCe -/
def cece_ants : ℕ := 2 * abe_ants

/-- The number of ants found by Duke -/
def duke_ants : ℕ := abe_ants / 2

/-- The total number of ants found by all four children -/
def total_ants : ℕ := abe_ants + beth_ants + cece_ants + duke_ants

/-- Theorem stating that the total number of ants found is 20 -/
theorem total_ants_is_twenty : total_ants = 20 := by sorry

end NUMINAMATH_CALUDE_total_ants_is_twenty_l3796_379654


namespace NUMINAMATH_CALUDE_nested_cube_root_l3796_379694

theorem nested_cube_root (N : ℝ) (h : N > 1) :
  (N * (N * (N * N^(1/3))^(1/3))^(1/3))^(1/3) = N^(40/81) := by
  sorry

end NUMINAMATH_CALUDE_nested_cube_root_l3796_379694


namespace NUMINAMATH_CALUDE_science_fair_teams_l3796_379655

theorem science_fair_teams (total_students : Nat) (red_hats : Nat) (green_hats : Nat) 
  (total_teams : Nat) (red_red_teams : Nat) : 
  total_students = 144 →
  red_hats = 63 →
  green_hats = 81 →
  total_teams = 72 →
  red_red_teams = 28 →
  red_hats + green_hats = total_students →
  ∃ (green_green_teams : Nat), green_green_teams = 37 ∧ 
    green_green_teams + red_red_teams + (total_students - 2 * red_red_teams - 2 * green_green_teams) / 2 = total_teams :=
by
  sorry

end NUMINAMATH_CALUDE_science_fair_teams_l3796_379655


namespace NUMINAMATH_CALUDE_trip_cost_is_1050_l3796_379635

-- Define the distances and costs
def distance_AB : ℝ := 4000
def distance_BC : ℝ := 3000
def bus_rate : ℝ := 0.15
def plane_rate : ℝ := 0.12
def plane_booking_fee : ℝ := 120

-- Define the total trip cost function
def total_trip_cost : ℝ :=
  (distance_AB * plane_rate + plane_booking_fee) + (distance_BC * bus_rate)

-- Theorem statement
theorem trip_cost_is_1050 : total_trip_cost = 1050 := by
  sorry

end NUMINAMATH_CALUDE_trip_cost_is_1050_l3796_379635


namespace NUMINAMATH_CALUDE_no_rectangle_with_sum_76_l3796_379674

theorem no_rectangle_with_sum_76 : ¬∃ (w : ℕ), w > 0 ∧ 2 * w^2 + 6 * w = 76 := by
  sorry

end NUMINAMATH_CALUDE_no_rectangle_with_sum_76_l3796_379674


namespace NUMINAMATH_CALUDE_ratio_problem_l3796_379631

theorem ratio_problem (x : ℝ) : 0.75 / x = 5 / 7 → x = 1.05 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3796_379631


namespace NUMINAMATH_CALUDE_factorization_of_x_squared_minus_x_l3796_379657

theorem factorization_of_x_squared_minus_x (x : ℝ) : x^2 - x = x * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_squared_minus_x_l3796_379657


namespace NUMINAMATH_CALUDE_paul_sandwich_consumption_l3796_379689

/-- Calculates the number of sandwiches eaten in one 3-day cycle -/
def sandwiches_per_cycle (initial : ℕ) : ℕ :=
  initial + 2 * initial + 4 * initial

/-- Calculates the total number of sandwiches eaten in a given number of days -/
def total_sandwiches (days : ℕ) (initial : ℕ) : ℕ :=
  (days / 3) * sandwiches_per_cycle initial + 
  if days % 3 = 1 then initial
  else if days % 3 = 2 then initial + 2 * initial
  else 0

theorem paul_sandwich_consumption :
  total_sandwiches 6 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_paul_sandwich_consumption_l3796_379689


namespace NUMINAMATH_CALUDE_parabola_focus_l3796_379649

/-- A parabola is defined by its coefficients a, b, and c in the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola is a point (h, k) -/
structure Focus where
  h : ℝ
  k : ℝ

/-- Given a parabola y = -2x^2 - 4x + 1, its focus is at (-1, 23/8) -/
theorem parabola_focus (p : Parabola) (f : Focus) :
  p.a = -2 ∧ p.b = -4 ∧ p.c = 1 →
  f.h = -1 ∧ f.k = 23/8 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l3796_379649


namespace NUMINAMATH_CALUDE_triangle_angle_relations_l3796_379661

theorem triangle_angle_relations (a b c : ℝ) (α β γ : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (0 < α) ∧ (0 < β) ∧ (0 < γ) ∧ 
  (α + β + γ = Real.pi) ∧
  (c^2 = a^2 + 2 * b^2 * Real.cos β) →
  ((γ = β / 2 + Real.pi / 2 ∧ α = Real.pi / 2 - 3 * β / 2 ∧ 0 < β ∧ β < Real.pi / 3) ∨
   (α = β / 2 ∧ γ = Real.pi - 3 * β / 2 ∧ 0 < β ∧ β < 2 * Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_relations_l3796_379661


namespace NUMINAMATH_CALUDE_percentage_women_after_hiring_l3796_379696

/-- Percentage of women in a multinational company after new hires --/
theorem percentage_women_after_hiring (country_a_initial : ℕ) (country_b_initial : ℕ)
  (country_a_men_ratio : ℚ) (country_b_women_ratio : ℚ)
  (country_a_new_hires : ℕ) (country_b_new_hires : ℕ)
  (country_a_new_men_ratio : ℚ) (country_b_new_women_ratio : ℚ)
  (h1 : country_a_initial = 90)
  (h2 : country_b_initial = 150)
  (h3 : country_a_men_ratio = 2/3)
  (h4 : country_b_women_ratio = 3/5)
  (h5 : country_a_new_hires = 5)
  (h6 : country_b_new_hires = 8)
  (h7 : country_a_new_men_ratio = 3/5)
  (h8 : country_b_new_women_ratio = 1/2) :
  ∃ (percentage : ℚ), abs (percentage - 4980/10000) < 1/1000 ∧
  percentage = (country_a_initial * (1 - country_a_men_ratio) + country_b_initial * country_b_women_ratio +
    country_a_new_hires * (1 - country_a_new_men_ratio) + country_b_new_hires * country_b_new_women_ratio) /
    (country_a_initial + country_b_initial + country_a_new_hires + country_b_new_hires) * 100 :=
by
  sorry


end NUMINAMATH_CALUDE_percentage_women_after_hiring_l3796_379696


namespace NUMINAMATH_CALUDE_min_blue_eyes_and_lunch_box_l3796_379602

theorem min_blue_eyes_and_lunch_box 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (lunch_box : ℕ) 
  (h1 : total_students = 35) 
  (h2 : blue_eyes = 15) 
  (h3 : lunch_box = 25) :
  ∃ (overlap : ℕ), 
    overlap ≥ 5 ∧ 
    overlap ≤ blue_eyes ∧ 
    overlap ≤ lunch_box ∧ 
    (∀ (x : ℕ), x < overlap → 
      x + (total_students - lunch_box) < blue_eyes ∨ 
      x + (total_students - blue_eyes) < lunch_box) :=
by sorry

end NUMINAMATH_CALUDE_min_blue_eyes_and_lunch_box_l3796_379602


namespace NUMINAMATH_CALUDE_gcf_and_multiples_of_90_and_135_l3796_379676

theorem gcf_and_multiples_of_90_and_135 :
  ∃ (gcf : ℕ), 
    (Nat.gcd 90 135 = gcf) ∧ 
    (gcf = 45) ∧
    (45 ∣ gcf) ∧ 
    (90 ∣ gcf) ∧ 
    (135 ∣ gcf) := by
  sorry

end NUMINAMATH_CALUDE_gcf_and_multiples_of_90_and_135_l3796_379676


namespace NUMINAMATH_CALUDE_sphere_ratio_theorem_l3796_379651

/-- Given two spheres with radii r₁ and r₂ where r₁ : r₂ = 1 : 3, 
    prove that their surface areas are in the ratio 1:9 
    and their volumes are in the ratio 1:27 -/
theorem sphere_ratio_theorem (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 3) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 9 ∧
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 1 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_ratio_theorem_l3796_379651


namespace NUMINAMATH_CALUDE_coach_number_divisibility_l3796_379652

/-- A function that checks if a number is of the form aabb, abba, or abab -/
def isValidFormat (n : ℕ) : Prop :=
  ∃ a b : ℕ, 
    (n = a * 1000 + a * 100 + b * 10 + b) ∨ 
    (n = a * 1000 + b * 100 + b * 10 + a) ∨ 
    (n = a * 1000 + b * 100 + a * 10 + b)

/-- The set of possible ages of the children -/
def childrenAges : Set ℕ := {3, 4, 5, 6, 7, 8, 9, 10, 11}

/-- The theorem to be proved -/
theorem coach_number_divisibility 
  (N : ℕ) 
  (h1 : isValidFormat N) 
  (h2 : ∀ (x : ℕ), x ∈ childrenAges → x ≠ 10 → N % x = 0) 
  (h3 : N % 10 ≠ 0) 
  (h4 : 1000 ≤ N ∧ N < 10000) : 
  ∃ (a b : ℕ), N = 7000 + 700 + 40 + 4 := by
  sorry

end NUMINAMATH_CALUDE_coach_number_divisibility_l3796_379652


namespace NUMINAMATH_CALUDE_carrie_text_messages_l3796_379619

/-- The number of text messages Carrie sends to her brother on Saturday -/
def saturday_messages : ℕ := 5

/-- The number of text messages Carrie sends to her brother on Sunday -/
def sunday_messages : ℕ := 5

/-- The number of text messages Carrie sends to her brother on each weekday -/
def weekday_messages : ℕ := 2

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weeks we're considering -/
def total_weeks : ℕ := 4

/-- The total number of text messages Carrie sends to her brother over the given period -/
def total_messages : ℕ := 
  total_weeks * (saturday_messages + sunday_messages + weekdays_per_week * weekday_messages)

theorem carrie_text_messages : total_messages = 80 := by
  sorry

end NUMINAMATH_CALUDE_carrie_text_messages_l3796_379619


namespace NUMINAMATH_CALUDE_fewer_threes_for_hundred_l3796_379611

-- Define a type for arithmetic expressions
inductive Expr
  | num : Int → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

-- Function to evaluate an expression
def eval : Expr → Int
  | Expr.num n => n
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

-- Function to count the number of threes in an expression
def countThrees : Expr → Nat
  | Expr.num 3 => 1
  | Expr.num _ => 0
  | Expr.add e1 e2 => countThrees e1 + countThrees e2
  | Expr.sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.div e1 e2 => countThrees e1 + countThrees e2

-- Theorem: There exists an expression using fewer than ten threes that evaluates to 100
theorem fewer_threes_for_hundred : ∃ e : Expr, eval e = 100 ∧ countThrees e < 10 := by
  sorry


end NUMINAMATH_CALUDE_fewer_threes_for_hundred_l3796_379611


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3796_379628

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

-- State the theorem
theorem vertex_of_quadratic :
  ∃ (a h k : ℝ), (∀ x, f x = a * (x - h)^2 + k) ∧ (f h = k) ∧ (∀ x, f x ≤ k) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3796_379628


namespace NUMINAMATH_CALUDE_altitude_length_is_one_l3796_379682

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point lies on the parabola y = x^2 -/
def onParabola (p : Point) : Prop :=
  p.y = p.x^2

/-- Checks if a line segment is parallel to the x-axis -/
def parallelToXAxis (p1 p2 : Point) : Prop :=
  p1.y = p2.y

/-- Checks if a triangle is right-angled -/
def isRightTriangle (t : Triangle) : Prop :=
  let AC := (t.C.x - t.A.x, t.C.y - t.A.y)
  let BC := (t.C.x - t.B.x, t.C.y - t.B.y)
  AC.1 * BC.1 + AC.2 * BC.2 = 0

/-- Calculates the length of the altitude from C to AB -/
def altitudeLength (t : Triangle) : ℝ :=
  t.A.y - t.C.y

/-- The main theorem -/
theorem altitude_length_is_one (t : Triangle) :
  isRightTriangle t →
  onParabola t.A ∧ onParabola t.B ∧ onParabola t.C →
  parallelToXAxis t.A t.B →
  altitudeLength t = 1 := by
  sorry

end NUMINAMATH_CALUDE_altitude_length_is_one_l3796_379682


namespace NUMINAMATH_CALUDE_solve_system_l3796_379656

theorem solve_system (y z : ℝ) 
  (h1 : y^2 - 6*y + 9 = 0) 
  (h2 : y + z = 11) : 
  y = 3 ∧ z = 8 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l3796_379656


namespace NUMINAMATH_CALUDE_cuboids_painted_equals_five_l3796_379617

/-- The number of outer faces on each cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of faces painted -/
def total_faces_painted : ℕ := 30

/-- The number of cuboids painted -/
def num_cuboids : ℕ := total_faces_painted / faces_per_cuboid

theorem cuboids_painted_equals_five :
  num_cuboids = 5 :=
by sorry

end NUMINAMATH_CALUDE_cuboids_painted_equals_five_l3796_379617


namespace NUMINAMATH_CALUDE_diana_remaining_paint_l3796_379622

/-- The amount of paint required for one statue in gallons -/
def paint_per_statue : ℚ := 1/8

/-- The number of statues Diana can paint with the remaining paint -/
def statues_to_paint : ℕ := 7

/-- The total amount of paint Diana has remaining in gallons -/
def remaining_paint : ℚ := paint_per_statue * statues_to_paint

theorem diana_remaining_paint : remaining_paint = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_diana_remaining_paint_l3796_379622


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3796_379627

theorem quadratic_roots_sum (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 2023 = 0 → 
  x₂^2 + x₂ - 2023 = 0 → 
  x₁^2 + 2*x₁ + x₂ = 2022 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3796_379627


namespace NUMINAMATH_CALUDE_dog_owners_count_l3796_379673

-- Define the sets of people owning each type of pet
def C : Finset ℕ := sorry
def D : Finset ℕ := sorry
def R : Finset ℕ := sorry

-- Define the theorem
theorem dog_owners_count :
  (C ∪ D ∪ R).card = 60 ∧
  C.card = 30 ∧
  R.card = 16 ∧
  ((C ∩ D) ∪ (C ∩ R) ∪ (D ∩ R)).card - (C ∩ D ∩ R).card = 12 ∧
  (C ∩ D ∩ R).card = 7 →
  D.card = 40 := by
sorry


end NUMINAMATH_CALUDE_dog_owners_count_l3796_379673


namespace NUMINAMATH_CALUDE_sum_18_probability_l3796_379677

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The target sum we're aiming for -/
def target_sum : ℕ := 18

/-- The probability of rolling a sum of 18 with four standard 6-faced dice -/
def probability_sum_18 : ℚ := 5 / 216

/-- Theorem stating that the probability of rolling a sum of 18 with four standard 6-faced dice is 5/216 -/
theorem sum_18_probability : 
  probability_sum_18 = (num_favorable_outcomes : ℚ) / (num_faces ^ num_dice : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_sum_18_probability_l3796_379677


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3796_379658

theorem simplify_trig_expression (θ : Real) (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = 2 * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3796_379658


namespace NUMINAMATH_CALUDE_max_non_fiction_books_l3796_379670

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_non_fiction_books :
  ∀ (fiction non_fiction : ℕ) (p : ℕ),
    fiction + non_fiction = 100 →
    fiction = non_fiction + p →
    is_prime p →
    non_fiction ≤ 49 :=
by sorry

end NUMINAMATH_CALUDE_max_non_fiction_books_l3796_379670


namespace NUMINAMATH_CALUDE_expansion_coefficients_l3796_379672

theorem expansion_coefficients (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) 
  (h : (2*(x-1)-1)^9 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
                       a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9) : 
  a₂ = -144 ∧ a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 2 := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l3796_379672


namespace NUMINAMATH_CALUDE_prime_sum_47_l3796_379695

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define the property we want to prove
def no_prime_sum_47 : Prop :=
  ∀ p q : ℕ, is_prime p → is_prime q → p + q ≠ 47

-- State the theorem
theorem prime_sum_47 : no_prime_sum_47 :=
sorry

end NUMINAMATH_CALUDE_prime_sum_47_l3796_379695


namespace NUMINAMATH_CALUDE_min_a_for_decreasing_h_range_a_for_p_greater_q_l3796_379639

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := x + 4 * a / x - 1
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x
def h (a : ℝ) (x : ℝ) : ℝ := f a x - g a x
def p (x : ℝ) : ℝ := (2 - x^3) * Real.exp x
def q (a : ℝ) (x : ℝ) : ℝ := g a x / x + 2

-- Part I: Minimum value of a for h to be decreasing on [1,3]
theorem min_a_for_decreasing_h : 
  (∀ x ∈ Set.Icc 1 3, ∀ y ∈ Set.Icc 1 3, x ≤ y → h (9/7) x ≥ h (9/7) y) ∧
  (∀ a < 9/7, ∃ x ∈ Set.Icc 1 3, ∃ y ∈ Set.Icc 1 3, x < y ∧ h a x < h a y) :=
sorry

-- Part II: Range of a for p(x₁) > q(x₂) to hold for any x₁, x₂ ∈ (0,1)
theorem range_a_for_p_greater_q :
  (∀ a ≥ 0, ∀ x₁ ∈ Set.Ioo 0 1, ∀ x₂ ∈ Set.Ioo 0 1, p x₁ > q a x₂) ∧
  (∀ a < 0, ∃ x₁ ∈ Set.Ioo 0 1, ∃ x₂ ∈ Set.Ioo 0 1, p x₁ ≤ q a x₂) :=
sorry

end NUMINAMATH_CALUDE_min_a_for_decreasing_h_range_a_for_p_greater_q_l3796_379639


namespace NUMINAMATH_CALUDE_second_smallest_dimension_is_twelve_l3796_379659

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical pillar -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder can fit upright in a crate -/
def cylinderFitsInCrate (c : Cylinder) (d : CrateDimensions) : Prop :=
  (2 * c.radius ≤ d.length ∧ 2 * c.radius ≤ d.width) ∨
  (2 * c.radius ≤ d.length ∧ 2 * c.radius ≤ d.height) ∨
  (2 * c.radius ≤ d.width ∧ 2 * c.radius ≤ d.height)

/-- The theorem stating that the second smallest dimension of the crate is 12 feet -/
theorem second_smallest_dimension_is_twelve
  (d : CrateDimensions)
  (h1 : d.length = 6)
  (h2 : d.height = 12)
  (h3 : d.width > 0)
  (c : Cylinder)
  (h4 : c.radius = 6)
  (h5 : cylinderFitsInCrate c d) :
  d.width = 12 ∨ d.width = 12 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_dimension_is_twelve_l3796_379659


namespace NUMINAMATH_CALUDE_triangle_properties_l3796_379621

/-- Given a triangle ABC with the following properties:
  BC = √5
  AC = 3
  sin C = 2 sin A
  Prove that:
  1. AB = 2√5
  2. sin(A - π/4) = -√10/10
-/
theorem triangle_properties (A B C : ℝ) (h1 : BC = Real.sqrt 5) (h2 : AC = 3)
    (h3 : Real.sin C = 2 * Real.sin A) :
  AB = 2 * Real.sqrt 5 ∧ Real.sin (A - π/4) = -(Real.sqrt 10)/10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3796_379621


namespace NUMINAMATH_CALUDE_parabola_chord_length_l3796_379686

/-- Given a parabola y² = 4x with a chord passing through its focus and endpoints A(x₁, y₁) and B(x₂, y₂),
    if x₁ + x₂ = 6, then the length of AB is 8. -/
theorem parabola_chord_length (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 4*x₁ → y₂^2 = 4*x₂ → x₁ + x₂ = 6 → 
  ∃ (AB : ℝ), AB = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ∧ AB = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_chord_length_l3796_379686
