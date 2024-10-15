import Mathlib

namespace NUMINAMATH_CALUDE_equal_division_theorem_l1166_116643

theorem equal_division_theorem (total : ℕ) (people : ℕ) (share : ℕ) : 
  total = 2400 → people = 4 → share * people = total → share = 600 := by
  sorry

end NUMINAMATH_CALUDE_equal_division_theorem_l1166_116643


namespace NUMINAMATH_CALUDE_average_of_three_l1166_116666

theorem average_of_three (M : ℝ) (h1 : 12 < M) (h2 : M < 25) :
  let avg := (8 + 15 + M) / 3
  (avg = 12 ∨ avg = 15) ∧ avg ≠ 18 ∧ avg ≠ 20 ∧ avg ≠ 23 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_l1166_116666


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1166_116619

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, 3 * X^5 + X^4 + 3 = (X - 2)^2 * q + (13 * X - 9) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1166_116619


namespace NUMINAMATH_CALUDE_polynomial_comparison_l1166_116635

theorem polynomial_comparison : ∀ x : ℝ, (x - 3) * (x - 2) > (x + 1) * (x - 6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_comparison_l1166_116635


namespace NUMINAMATH_CALUDE_community_age_is_35_l1166_116629

/-- Represents the average age of a community given specific demographic information. -/
def community_average_age (women_ratio men_ratio : ℚ) (women_avg_age men_avg_age children_avg_age : ℚ) (children_ratio : ℚ) : ℚ :=
  let total_population := women_ratio + men_ratio + children_ratio * men_ratio
  let total_age := women_ratio * women_avg_age + men_ratio * men_avg_age + children_ratio * men_ratio * children_avg_age
  total_age / total_population

/-- Theorem stating that the average age of the community is 35 years given the specified conditions. -/
theorem community_age_is_35 :
  community_average_age 3 2 40 36 10 (1/3) = 35 := by
  sorry

end NUMINAMATH_CALUDE_community_age_is_35_l1166_116629


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_l1166_116616

/-- The line equation 4y - 3x = 16 intersects the x-axis at (-16/3, 0) -/
theorem line_intersects_x_axis :
  let line := λ x y : ℚ => 4 * y - 3 * x = 16
  let x_axis := λ x y : ℚ => y = 0
  let intersection_point := (-16/3, 0)
  line intersection_point.1 intersection_point.2 ∧ x_axis intersection_point.1 intersection_point.2 := by
  sorry


end NUMINAMATH_CALUDE_line_intersects_x_axis_l1166_116616


namespace NUMINAMATH_CALUDE_largest_smallest_factor_l1166_116645

theorem largest_smallest_factor (a b c : ℕ+) : 
  a * b * c = 2160 → 
  ∃ (x : ℕ+), x ≤ a ∧ x ≤ b ∧ x ≤ c ∧ 
  (∀ (y : ℕ+), y ≤ a ∧ y ≤ b ∧ y ≤ c → y ≤ x) ∧ 
  x ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_largest_smallest_factor_l1166_116645


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1166_116605

theorem cube_sum_reciprocal (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 = 970 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1166_116605


namespace NUMINAMATH_CALUDE_franklin_gathering_handshakes_l1166_116640

/-- Represents a gathering of married couples -/
structure Gathering where
  couples : Nat
  men : Nat
  women : Nat
  total_people : Nat

/-- Calculates the number of handshakes in the gathering -/
def handshakes (g : Gathering) : Nat :=
  let men_handshakes := g.men.choose 2
  let men_women_handshakes := g.men * (g.women - 1)
  men_handshakes + men_women_handshakes

theorem franklin_gathering_handshakes :
  ∀ g : Gathering,
    g.couples = 15 →
    g.men = g.couples →
    g.women = g.couples →
    g.total_people = g.men + g.women →
    handshakes g = 315 := by
  sorry

#eval handshakes { couples := 15, men := 15, women := 15, total_people := 30 }

end NUMINAMATH_CALUDE_franklin_gathering_handshakes_l1166_116640


namespace NUMINAMATH_CALUDE_sandy_age_l1166_116611

theorem sandy_age (S M : ℕ) (h1 : M = S + 14) (h2 : S / M = 7 / 9) : S = 49 := by
  sorry

end NUMINAMATH_CALUDE_sandy_age_l1166_116611


namespace NUMINAMATH_CALUDE_tan_105_degrees_l1166_116657

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l1166_116657


namespace NUMINAMATH_CALUDE_seven_n_implies_n_is_sum_of_squares_l1166_116625

theorem seven_n_implies_n_is_sum_of_squares (n : ℤ) (A B : ℤ) (h : 7 * n = A^2 + 3 * B^2) :
  ∃ (a b : ℤ), n = a^2 + 3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_seven_n_implies_n_is_sum_of_squares_l1166_116625


namespace NUMINAMATH_CALUDE_factor_bound_l1166_116600

/-- The number of ways to factor a positive integer into a product of integers greater than 1 -/
def f (k : ℕ) : ℕ := sorry

/-- Theorem: For any positive integer n > 1 and any prime factor p of n,
    the number of ways to factor n is less than or equal to n/p -/
theorem factor_bound {n p : ℕ} (h1 : n > 1) (h2 : p.Prime) (h3 : p ∣ n) : f n ≤ n / p := by
  sorry

end NUMINAMATH_CALUDE_factor_bound_l1166_116600


namespace NUMINAMATH_CALUDE_profit_reduction_theorem_l1166_116634

/-- Initial daily sales -/
def initial_sales : ℕ := 30

/-- Initial profit per unit in yuan -/
def initial_profit_per_unit : ℕ := 50

/-- Sales increase per yuan of price reduction -/
def sales_increase_rate : ℕ := 2

/-- Calculate daily profit based on price reduction -/
def daily_profit (price_reduction : ℝ) : ℝ :=
  (initial_profit_per_unit - price_reduction) * (initial_sales + sales_increase_rate * price_reduction)

/-- Price reduction needed for a specific daily profit -/
def price_reduction_for_profit (target_profit : ℝ) : ℝ :=
  20  -- This is the value we want to prove

/-- Price reduction for maximum profit -/
def price_reduction_for_max_profit : ℝ :=
  17.5  -- This is the value we want to prove

theorem profit_reduction_theorem :
  daily_profit (price_reduction_for_profit 2100) = 2100 ∧
  ∀ x, daily_profit x ≤ daily_profit price_reduction_for_max_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_reduction_theorem_l1166_116634


namespace NUMINAMATH_CALUDE_no_lines_satisfying_conditions_l1166_116652

-- Define the plane and points A and B
def Plane : Type := ℝ × ℝ
def A : Plane := sorry
def B : Plane := sorry

-- Define the distance between two points in the plane
def distance (p q : Plane) : ℝ := sorry

-- Define a line in the plane
def Line : Type := Plane → Prop

-- Define the distance from a point to a line
def point_to_line_distance (p : Plane) (l : Line) : ℝ := sorry

-- Define the angle between two lines
def angle_between_lines (l1 l2 : Line) : ℝ := sorry

-- Define the line y = x
def y_equals_x : Line := sorry

-- State the theorem
theorem no_lines_satisfying_conditions :
  ∀ (l : Line),
    distance A B = 8 →
    point_to_line_distance A l = 3 →
    point_to_line_distance B l = 4 →
    angle_between_lines l y_equals_x = π/4 →
    False :=
sorry

end NUMINAMATH_CALUDE_no_lines_satisfying_conditions_l1166_116652


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l1166_116650

def q (x : ℝ) : ℝ := -x^3 + 2*x^2 + 3*x

theorem q_satisfies_conditions :
  (q 3 = 0) ∧ 
  (q (-1) = 0) ∧ 
  (∃ (a b c : ℝ), ∀ x, q x = a*x^3 + b*x^2 + c*x) ∧
  (q 4 = -20) := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l1166_116650


namespace NUMINAMATH_CALUDE_polar_to_rectangular_coordinates_l1166_116603

theorem polar_to_rectangular_coordinates :
  let r : ℝ := 4
  let θ : ℝ := 5 * π / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x = -2 * Real.sqrt 3 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_coordinates_l1166_116603


namespace NUMINAMATH_CALUDE_gcd_4288_9277_l1166_116694

theorem gcd_4288_9277 : Int.gcd 4288 9277 = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_4288_9277_l1166_116694


namespace NUMINAMATH_CALUDE_point_transformation_l1166_116676

/-- Rotates a point (x, y) by 180° counterclockwise around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

/-- Reflects a point (x, y) about the line y = x -/
def reflectAboutYEqualsX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

/-- The main theorem -/
theorem point_transformation (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate180 a b 2 3
  let final := reflectAboutYEqualsX rotated.1 rotated.2
  final = (5, -1) → b - a = -4 := by
sorry

end NUMINAMATH_CALUDE_point_transformation_l1166_116676


namespace NUMINAMATH_CALUDE_average_after_removal_l1166_116661

theorem average_after_removal (numbers : Finset ℝ) (sum : ℝ) :
  Finset.card numbers = 12 →
  sum / 12 = 72 →
  60 ∈ numbers →
  80 ∈ numbers →
  ((sum - 60 - 80) / 10 : ℝ) = 72.4 := by
  sorry

end NUMINAMATH_CALUDE_average_after_removal_l1166_116661


namespace NUMINAMATH_CALUDE_evening_pages_read_l1166_116658

/-- Given a person who reads books with the following conditions:
  * Reads twice a day (morning and evening)
  * Reads 5 pages in the morning
  * Reads at this rate for a week (7 days)
  * Reads a total of 105 pages in a week
This theorem proves that the number of pages read in the evening is 10. -/
theorem evening_pages_read (morning_pages : ℕ) (total_pages : ℕ) (days : ℕ) :
  morning_pages = 5 →
  days = 7 →
  total_pages = 105 →
  ∃ (evening_pages : ℕ), 
    days * (morning_pages + evening_pages) = total_pages ∧ 
    evening_pages = 10 := by
  sorry

end NUMINAMATH_CALUDE_evening_pages_read_l1166_116658


namespace NUMINAMATH_CALUDE_dexter_card_boxes_l1166_116665

theorem dexter_card_boxes (x : ℕ) : 
  (15 * x + 20 * (x - 3) = 255) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_dexter_card_boxes_l1166_116665


namespace NUMINAMATH_CALUDE_gracies_height_l1166_116684

/-- Proves that Gracie's height is 56 inches given the relationships between Gracie, Grayson, and Griffin's heights. -/
theorem gracies_height (griffin_height grayson_height gracie_height : ℕ) : 
  griffin_height = 61 →
  grayson_height = griffin_height + 2 →
  gracie_height = grayson_height - 7 →
  gracie_height = 56 :=
by sorry

end NUMINAMATH_CALUDE_gracies_height_l1166_116684


namespace NUMINAMATH_CALUDE_school_girls_count_l1166_116692

theorem school_girls_count (total_students sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : sample_size = 200)
  (h3 : ∃ (girls_in_sample : ℕ), 
    girls_in_sample + (girls_in_sample + 10) = sample_size) :
  ∃ (girls_in_school : ℕ), 
    girls_in_school = (950 : ℕ) ∧ 
    (girls_in_school : ℚ) / total_students = 
      ((sample_size / 2 - 5) : ℚ) / sample_size :=
by sorry

end NUMINAMATH_CALUDE_school_girls_count_l1166_116692


namespace NUMINAMATH_CALUDE_max_binomial_coeff_x_minus_2_pow_5_l1166_116672

theorem max_binomial_coeff_x_minus_2_pow_5 :
  (Finset.range 6).sup (fun k => Nat.choose 5 k) = 10 := by
  sorry

end NUMINAMATH_CALUDE_max_binomial_coeff_x_minus_2_pow_5_l1166_116672


namespace NUMINAMATH_CALUDE_equation_solve_for_n_l1166_116610

theorem equation_solve_for_n (s P k c n : ℝ) (h1 : c > 0) (h2 : P = s / (c * (1 + k)^n)) :
  n = Real.log (s / (P * c)) / Real.log (1 + k) := by
  sorry

end NUMINAMATH_CALUDE_equation_solve_for_n_l1166_116610


namespace NUMINAMATH_CALUDE_domain_of_sqrt_fraction_l1166_116681

theorem domain_of_sqrt_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 3)) / (Real.sqrt (8 - x))) ↔ x ∈ Set.Ici (-3) ∩ Set.Iio 8 := by
sorry

end NUMINAMATH_CALUDE_domain_of_sqrt_fraction_l1166_116681


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l1166_116602

def f (x : ℝ) := -x^3

theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l1166_116602


namespace NUMINAMATH_CALUDE_quadratic_solution_l1166_116669

theorem quadratic_solution : ∃ x : ℝ, x^2 - 5*x + 6 = 0 ↔ x = 2 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1166_116669


namespace NUMINAMATH_CALUDE_highest_points_fewer_wins_l1166_116631

/-- Represents a football team in the tournament -/
structure Team :=
  (id : Nat)
  (wins : Nat)
  (draws : Nat)
  (losses : Nat)

/-- Calculates the points for a team based on their wins and draws -/
def points (t : Team) : Nat :=
  3 * t.wins + t.draws

/-- Represents the tournament results -/
structure TournamentResult :=
  (teams : Finset Team)
  (team_count : Nat)
  (hteam_count : teams.card = team_count)

/-- Theorem stating that it's possible for a team to have the highest points but fewer wins -/
theorem highest_points_fewer_wins (tr : TournamentResult) 
  (h_six_teams : tr.team_count = 6) : 
  ∃ (t1 t2 : Team), t1 ∈ tr.teams ∧ t2 ∈ tr.teams ∧ 
    (∀ t ∈ tr.teams, points t1 ≥ points t) ∧
    t1.wins < t2.wins :=
  sorry

end NUMINAMATH_CALUDE_highest_points_fewer_wins_l1166_116631


namespace NUMINAMATH_CALUDE_binary_calculation_theorem_l1166_116620

/-- Represents a binary number as a list of bits (least significant bit first) -/
def BinaryNum := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNum) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal number to its binary representation -/
def decimal_to_binary (n : ℕ) : BinaryNum :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : BinaryNum :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNum) : BinaryNum :=
  decimal_to_binary (binary_to_decimal a * binary_to_decimal b)

/-- Divides a binary number by 2^n (equivalent to right shift by n) -/
def binary_divide_by_power_of_two (b : BinaryNum) (n : ℕ) : BinaryNum :=
  decimal_to_binary (binary_to_decimal b / 2^n)

theorem binary_calculation_theorem :
  let a : BinaryNum := [false, true, false, true, false, false, true, true]  -- 11001010₂
  let b : BinaryNum := [false, true, false, true, true]                      -- 11010₂
  let divisor : BinaryNum := [false, false, true]                            -- 100₂
  binary_divide_by_power_of_two (binary_multiply a b) 2 =
  [false, false, true, false, true, true, true, true, false, false]          -- 1001110100₂
  := by sorry

end NUMINAMATH_CALUDE_binary_calculation_theorem_l1166_116620


namespace NUMINAMATH_CALUDE_first_dog_takes_one_more_than_second_l1166_116691

def dog_bone_problem (second_dog_bones : ℕ) : Prop :=
  let first_dog_bones := 3
  let third_dog_bones := 2 * second_dog_bones
  let fourth_dog_bones := 1
  let fifth_dog_bones := 2 * fourth_dog_bones
  first_dog_bones + second_dog_bones + third_dog_bones + fourth_dog_bones + fifth_dog_bones = 12

theorem first_dog_takes_one_more_than_second :
  ∃ (second_dog_bones : ℕ), dog_bone_problem second_dog_bones ∧ 3 = second_dog_bones + 1 := by
  sorry

end NUMINAMATH_CALUDE_first_dog_takes_one_more_than_second_l1166_116691


namespace NUMINAMATH_CALUDE_probability_of_selection_l1166_116615

/-- The number of students -/
def n : ℕ := 5

/-- The number of students to be chosen -/
def k : ℕ := 2

/-- Binomial coefficient -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The probability of selecting 2 students out of 5, where student A is selected and student B is not -/
theorem probability_of_selection : 
  (choose (n - 2) (k - 1) : ℚ) / (choose n k) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_l1166_116615


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l1166_116673

theorem cube_volume_ratio (edge_q : ℝ) (edge_p : ℝ) (h : edge_p = 3 * edge_q) :
  (edge_q ^ 3) / (edge_p ^ 3) = 1 / 27 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l1166_116673


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1166_116627

-- Define the arithmetic sequence
def arithmetic_seq (a₁ a₂ : ℝ) : Prop :=
  ∃ d : ℝ, a₂ - a₁ = d ∧ a₁ - (-2) = d ∧ (-8) - a₂ = d

-- Define the geometric sequence
def geometric_seq (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ r : ℝ, b₁ / (-2) = r ∧ b₂ / b₁ = r ∧ b₃ / b₂ = r ∧ (-8) / b₃ = r

theorem arithmetic_geometric_ratio
  (a₁ a₂ b₁ b₂ b₃ : ℝ)
  (h_arith : arithmetic_seq a₁ a₂)
  (h_geom : geometric_seq b₁ b₂ b₃) :
  (a₂ - a₁) / b₂ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1166_116627


namespace NUMINAMATH_CALUDE_initial_avg_equals_correct_avg_l1166_116638

-- Define the number of elements
def n : ℕ := 10

-- Define the correct average
def correct_avg : ℚ := 22

-- Define the difference between the correct and misread value
def misread_diff : ℤ := 10

-- Theorem statement
theorem initial_avg_equals_correct_avg :
  let correct_sum := n * correct_avg
  let initial_sum := correct_sum - misread_diff
  initial_sum / n = correct_avg := by
sorry

end NUMINAMATH_CALUDE_initial_avg_equals_correct_avg_l1166_116638


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1166_116697

-- Define set A
def A : Set ℝ := {x | -1 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 3}

-- Define set B
def B : Set ℝ := {x | x ≠ 0 ∧ (x - 2) / x ≤ 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1166_116697


namespace NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l1166_116614

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- State the theorem
theorem union_equality_iff_a_in_range (a : ℝ) :
  M a ∪ N = N ↔ a ∈ Set.Icc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_union_equality_iff_a_in_range_l1166_116614


namespace NUMINAMATH_CALUDE_zero_multiple_of_all_primes_l1166_116644

theorem zero_multiple_of_all_primes : ∃! x : ℤ, ∀ p : ℕ, Nat.Prime p → ∃ k : ℤ, x = k * p :=
sorry

end NUMINAMATH_CALUDE_zero_multiple_of_all_primes_l1166_116644


namespace NUMINAMATH_CALUDE_function_properties_l1166_116678

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / x

theorem function_properties (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∀ m n : ℝ, m > 0 ∧ n > 0 ∧ m < n ∧ f a m = 2*m ∧ f a n = 2*n → a > 2 * Real.sqrt 2) ∧
  ((∀ x : ℝ, x ∈ Set.Icc (1/3) (1/2) → x^2 * |f a x| ≤ 1) → a ∈ Set.Icc (-2) 6) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1166_116678


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l1166_116653

theorem no_positive_integer_solution :
  ¬ ∃ (x : ℕ), (x > 0) ∧ ((5 * x + 1) / (x - 1) > 2 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l1166_116653


namespace NUMINAMATH_CALUDE_circular_garden_radius_l1166_116624

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1 / 6) * π * r^2 → r = 12 := by
  sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l1166_116624


namespace NUMINAMATH_CALUDE_election_votes_proof_l1166_116613

/-- The total number of votes in a school election where Emily received 45 votes, 
    which accounted for 25% of the total votes. -/
def total_votes : ℕ := 180

/-- Emily's votes in the election -/
def emily_votes : ℕ := 45

/-- The percentage of total votes that Emily received -/
def emily_percentage : ℚ := 25 / 100

theorem election_votes_proof : 
  total_votes = emily_votes / emily_percentage :=
by sorry

end NUMINAMATH_CALUDE_election_votes_proof_l1166_116613


namespace NUMINAMATH_CALUDE_sine_inequalities_l1166_116622

theorem sine_inequalities :
  (∀ x : ℝ, |Real.sin (2 * x)| ≤ 2 * |Real.sin x|) ∧
  (∀ n : ℕ, n > 0 → ∀ x : ℝ, |Real.sin (n * x)| ≤ n * |Real.sin x|) := by
  sorry

end NUMINAMATH_CALUDE_sine_inequalities_l1166_116622


namespace NUMINAMATH_CALUDE_fraction_unchanged_l1166_116682

theorem fraction_unchanged (a b : ℝ) : (2 * (7 * a)) / ((7 * a) + (7 * b)) = (2 * a) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l1166_116682


namespace NUMINAMATH_CALUDE_sector_perimeter_l1166_116606

/-- Given a circular sector with area 2 and central angle 4 radians, its perimeter is 6. -/
theorem sector_perimeter (r : ℝ) (h1 : r > 0) : 
  (1/2 * r * (4 * r) = 2) → (4 * r + 2 * r = 6) := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l1166_116606


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1166_116688

theorem sqrt_product_simplification (y : ℝ) (hy : y > 0) :
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1166_116688


namespace NUMINAMATH_CALUDE_colonization_combinations_l1166_116623

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 8

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := 7

/-- Represents the colonization effort required for an Earth-like planet -/
def earth_like_effort : ℕ := 2

/-- Represents the colonization effort required for a Mars-like planet -/
def mars_like_effort : ℕ := 1

/-- Represents the total available colonization effort -/
def total_effort : ℕ := 18

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Theorem stating the number of distinct combinations of planets that can be fully colonized -/
theorem colonization_combinations : 
  (choose earth_like_planets 8 * choose mars_like_planets 2) +
  (choose earth_like_planets 7 * choose mars_like_planets 4) +
  (choose earth_like_planets 6 * choose mars_like_planets 6) = 497 := by
  sorry

end NUMINAMATH_CALUDE_colonization_combinations_l1166_116623


namespace NUMINAMATH_CALUDE_sum_equals_zero_l1166_116646

theorem sum_equals_zero : 1 + 1 - 2 + 3 + 5 - 8 + 13 + 21 - 34 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_zero_l1166_116646


namespace NUMINAMATH_CALUDE_intersection_M_N_l1166_116677

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set N
def N : Set ℝ := {x | Real.log x / Real.log 2 < 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1166_116677


namespace NUMINAMATH_CALUDE_cubic_square_fraction_inequality_l1166_116671

theorem cubic_square_fraction_inequality {s r : ℝ} (hs : 0 < s) (hr : 0 < r) (hsr : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_square_fraction_inequality_l1166_116671


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l1166_116655

theorem minimum_value_theorem (x y m : ℝ) :
  x > 0 →
  y > 0 →
  (4 / x + 9 / y = m) →
  (∀ a b : ℝ, a > 0 → b > 0 → 4 / a + 9 / b = m → x + y ≤ a + b) →
  x + y = 5 / 6 →
  m = 30 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l1166_116655


namespace NUMINAMATH_CALUDE_union_A_B_complement_B_intersect_A_C_subset_A_implies_a_leq_3_l1166_116630

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x > 2}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- Theorem statements
theorem union_A_B : A ∪ B = {x | x > 2} := by sorry

theorem complement_B_intersect_A : (𝓤 \ B) ∩ A = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

theorem C_subset_A_implies_a_leq_3 (a : ℝ) : C a ⊆ A → a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_B_intersect_A_C_subset_A_implies_a_leq_3_l1166_116630


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l1166_116687

theorem polynomial_division_quotient :
  ∀ (x : ℝ), x ≠ 1 →
  (x^6 + 8) / (x - 1) = x^5 + x^4 + x^3 + x^2 + x + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l1166_116687


namespace NUMINAMATH_CALUDE_income_comparison_l1166_116674

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.8)
  (h2 : mary = tim * 1.6) : 
  mary = juan * 1.28 := by sorry

end NUMINAMATH_CALUDE_income_comparison_l1166_116674


namespace NUMINAMATH_CALUDE_inequality_proof_l1166_116663

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  x*y/z + y*z/x + x*z/y ≥ Real.sqrt 3 ∧ 
  (x*y/z + y*z/x + x*z/y = Real.sqrt 3 ↔ x = y ∧ y = z ∧ z = Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1166_116663


namespace NUMINAMATH_CALUDE_area_MNKP_l1166_116675

/-- The area of quadrilateral MNKP given the area of quadrilateral ABCD -/
theorem area_MNKP (S_ABCD : ℝ) (h1 : S_ABCD = (180 + 50 * Real.sqrt 3) / 6)
  (h2 : ∃ S_MNKP : ℝ, S_MNKP = S_ABCD / 2) :
  ∃ S_MNKP : ℝ, S_MNKP = (90 + 25 * Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_area_MNKP_l1166_116675


namespace NUMINAMATH_CALUDE_parabola_and_line_intersection_l1166_116621

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line y = x + m -/
structure Line where
  m : ℝ

/-- The distance from a point to the y-axis -/
def distToAxis (pt : Point) : ℝ := |pt.x|

theorem parabola_and_line_intersection
  (para : Parabola)
  (A : Point)
  (l : Line)
  (h1 : A.y^2 = 2 * para.p * A.x) -- A is on the parabola
  (h2 : A.x = 2) -- x-coordinate of A is 2
  (h3 : distToAxis A = 4) -- distance from A to axis is 4
  (h4 : ∃ (P Q : Point), P ≠ Q ∧
        P.y^2 = 2 * para.p * P.x ∧ Q.y^2 = 2 * para.p * Q.x ∧
        P.y = P.x + l.m ∧ Q.y = Q.x + l.m) -- l intersects parabola at distinct P and Q
  (h5 : ∃ (P Q : Point), P ≠ Q ∧
        P.y^2 = 2 * para.p * P.x ∧ Q.y^2 = 2 * para.p * Q.x ∧
        P.y = P.x + l.m ∧ Q.y = Q.x + l.m ∧
        P.x * Q.x + P.y * Q.y = 0) -- OP ⊥ OQ
  : para.p = 4 ∧ l.m = -8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_line_intersection_l1166_116621


namespace NUMINAMATH_CALUDE_sarah_bottle_caps_l1166_116608

/-- Given that Sarah initially had 26 bottle caps and now has 29 in total,
    prove that she bought 3 bottle caps. -/
theorem sarah_bottle_caps (initial : ℕ) (total : ℕ) (bought : ℕ) 
    (h1 : initial = 26) 
    (h2 : total = 29) 
    (h3 : total = initial + bought) : bought = 3 := by
  sorry

end NUMINAMATH_CALUDE_sarah_bottle_caps_l1166_116608


namespace NUMINAMATH_CALUDE_equation_solution_l1166_116696

noncomputable def f (x a : ℝ) : ℝ := Real.sqrt ((x + 2)^2 + 4 * a^2 - 4) + Real.sqrt ((x - 2)^2 + 4 * a^2 - 4)

theorem equation_solution (a b : ℝ) (h : b ≥ 0) :
  (∀ x, f x a = 4 * b → (b ∈ Set.Icc 0 1 ∪ Set.Ioi 1 → x = 0) ∧
                        (b = 1 → x ∈ Set.Icc (-2) 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1166_116696


namespace NUMINAMATH_CALUDE_leo_marbles_l1166_116656

theorem leo_marbles (total_marbles : ℕ) (marbles_per_pack : ℕ) 
  (manny_fraction : ℚ) (neil_fraction : ℚ) :
  total_marbles = 400 →
  marbles_per_pack = 10 →
  manny_fraction = 1/4 →
  neil_fraction = 1/8 →
  (total_marbles / marbles_per_pack : ℚ) * (1 - manny_fraction - neil_fraction) = 25 := by
  sorry

end NUMINAMATH_CALUDE_leo_marbles_l1166_116656


namespace NUMINAMATH_CALUDE_quadratic_polynomial_value_at_10_l1166_116604

-- Define a quadratic polynomial
def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the divisibility condition
def divisibility_condition (q : ℝ → ℝ) : Prop :=
  ∃ (p : ℝ → ℝ), ∀ (x : ℝ), q x^3 - 3*x = p x * (x - 2) * (x + 2) * (x - 5)

theorem quadratic_polynomial_value_at_10 
  (a b c : ℝ) 
  (h : divisibility_condition (quadratic_polynomial a b c)) :
  quadratic_polynomial a b c 10 = (96 * Real.rpow 15 (1/3) - 135 * Real.rpow 6 (1/3)) / 21 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_polynomial_value_at_10_l1166_116604


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l1166_116685

def a : Fin 2 → ℝ := ![2, 3]
def b : Fin 2 → ℝ := ![-1, 2]

theorem collinear_vectors_m_value :
  ∃ (m : ℝ), ∃ (k : ℝ),
    (k ≠ 0) ∧
    (∀ i : Fin 2, k * (m * a i + 4 * b i) = (a i - 2 * b i)) →
    m = -2 :=
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l1166_116685


namespace NUMINAMATH_CALUDE_f_lower_bound_a_range_l1166_116641

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x + 2*a + 3|

-- Theorem 1: For all real x and a, f(x) ≥ 2
theorem f_lower_bound (x a : ℝ) : f x a ≥ 2 := by
  sorry

-- Theorem 2: If f(-3/2) < 3, then -1 < a < 0
theorem a_range (a : ℝ) : f (-3/2) a < 3 → -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_a_range_l1166_116641


namespace NUMINAMATH_CALUDE_fixed_point_satisfies_line_equation_fixed_point_is_unique_l1166_116683

/-- The line equation as a function of m, x, and y -/
def line_equation (m x y : ℝ) : ℝ := (3*m + 4)*x + (5 - 2*m)*y + 7*m - 6

/-- The fixed point through which all lines pass -/
def fixed_point : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the fixed point satisfies the line equation for all real m -/
theorem fixed_point_satisfies_line_equation :
  ∀ (m : ℝ), line_equation m fixed_point.1 fixed_point.2 = 0 := by sorry

/-- Theorem stating that the fixed point is unique -/
theorem fixed_point_is_unique :
  ∀ (x y : ℝ), (∀ (m : ℝ), line_equation m x y = 0) → (x, y) = fixed_point := by sorry

end NUMINAMATH_CALUDE_fixed_point_satisfies_line_equation_fixed_point_is_unique_l1166_116683


namespace NUMINAMATH_CALUDE_right_triangle_complex_roots_l1166_116659

theorem right_triangle_complex_roots : 
  ∃! (s : Finset ℂ), 
    (∀ z ∈ s, z ≠ 0 ∧ (z.re * (z^3).re + z.im * (z^3).im = 0)) ∧ 
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_complex_roots_l1166_116659


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l1166_116670

theorem unique_positive_integer_solution : ∃! (x : ℕ), x > 0 ∧ x - x^2 + 29 = 526 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l1166_116670


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1166_116690

theorem smallest_prime_divisor_of_sum (p : ℕ → ℕ → ℕ) :
  (∃ (d : ℕ), d ∣ (p 3 15 + p 11 21) ∧ Prime d) →
  (∀ (d : ℕ), d ∣ (p 3 15 + p 11 21) ∧ Prime d → d ≥ 2) →
  (∃ (d : ℕ), d ∣ (p 3 15 + p 11 21) ∧ Prime d ∧ d = 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1166_116690


namespace NUMINAMATH_CALUDE_workers_savings_l1166_116637

theorem workers_savings (monthly_pay : ℝ) (savings_fraction : ℝ) 
  (h1 : savings_fraction = 1 / 7)
  (h2 : savings_fraction > 0)
  (h3 : savings_fraction < 1) : 
  12 * (savings_fraction * monthly_pay) = 2 * ((1 - savings_fraction) * monthly_pay) := by
  sorry

end NUMINAMATH_CALUDE_workers_savings_l1166_116637


namespace NUMINAMATH_CALUDE_abs_sum_leq_sum_abs_l1166_116601

theorem abs_sum_leq_sum_abs (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  |a| + |b| ≤ |a + b| := by
sorry

end NUMINAMATH_CALUDE_abs_sum_leq_sum_abs_l1166_116601


namespace NUMINAMATH_CALUDE_product_units_digit_base6_l1166_116668

/-- The units digit in base 6 of a number -/
def unitsDigitBase6 (n : ℕ) : ℕ := n % 6

/-- The product of 168 and 59 -/
def product : ℕ := 168 * 59

theorem product_units_digit_base6 :
  unitsDigitBase6 product = 0 := by sorry

end NUMINAMATH_CALUDE_product_units_digit_base6_l1166_116668


namespace NUMINAMATH_CALUDE_expression_factorization_l1166_116680

theorem expression_factorization (a : ℝ) :
  (9 * a^4 + 105 * a^3 - 15 * a^2 + 1) - (-2 * a^4 + 3 * a^3 - 4 * a^2 + 2 * a - 5) =
  (a - 3) * (11 * a^2 * (a + 1) - 2) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l1166_116680


namespace NUMINAMATH_CALUDE_weekly_pill_count_l1166_116651

/-- Calculates the total number of pills taken in a week given daily intake of different types of pills -/
theorem weekly_pill_count 
  (insulin_daily : ℕ) 
  (blood_pressure_daily : ℕ) 
  (anticonvulsant_multiplier : ℕ) :
  insulin_daily = 2 →
  blood_pressure_daily = 3 →
  anticonvulsant_multiplier = 2 →
  (insulin_daily + blood_pressure_daily + anticonvulsant_multiplier * blood_pressure_daily) * 7 = 77 := by
  sorry

#check weekly_pill_count

end NUMINAMATH_CALUDE_weekly_pill_count_l1166_116651


namespace NUMINAMATH_CALUDE_function_properties_l1166_116667

def f (abc : ℝ) (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - abc

theorem function_properties (a b c abc : ℝ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : f abc a = 0) (h4 : f abc b = 0) (h5 : f abc c = 0) :
  (f abc 0) * (f abc 1) < 0 ∧ (f abc 0) * (f abc 3) > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l1166_116667


namespace NUMINAMATH_CALUDE_ferris_wheel_rides_l1166_116686

/-- The number of times Will rode the Ferris wheel during the day -/
def daytime_rides : ℕ := 7

/-- The number of times Will rode the Ferris wheel at night -/
def nighttime_rides : ℕ := 6

/-- The total number of times Will rode the Ferris wheel -/
def total_rides : ℕ := daytime_rides + nighttime_rides

theorem ferris_wheel_rides : total_rides = 13 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_rides_l1166_116686


namespace NUMINAMATH_CALUDE_tan_product_values_l1166_116679

theorem tan_product_values (a b : ℝ) :
  7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 2) = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = 4 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -4 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_values_l1166_116679


namespace NUMINAMATH_CALUDE_softball_players_count_l1166_116664

theorem softball_players_count (total : ℕ) (cricket : ℕ) (hockey : ℕ) (football : ℕ) 
  (h1 : total = 50)
  (h2 : cricket = 12)
  (h3 : hockey = 17)
  (h4 : football = 11) :
  total - (cricket + hockey + football) = 10 := by
  sorry

end NUMINAMATH_CALUDE_softball_players_count_l1166_116664


namespace NUMINAMATH_CALUDE_initial_tomatoes_correct_l1166_116628

/-- Represents the initial number of tomatoes in the garden -/
def initial_tomatoes : ℕ := 175

/-- Represents the initial number of potatoes in the garden -/
def initial_potatoes : ℕ := 77

/-- Represents the number of potatoes picked -/
def picked_potatoes : ℕ := 172

/-- Represents the total number of tomatoes and potatoes left after picking -/
def remaining_total : ℕ := 80

/-- Theorem stating that the initial number of tomatoes is correct given the conditions -/
theorem initial_tomatoes_correct : 
  initial_tomatoes + initial_potatoes - picked_potatoes = remaining_total :=
by sorry


end NUMINAMATH_CALUDE_initial_tomatoes_correct_l1166_116628


namespace NUMINAMATH_CALUDE_remaining_trip_time_l1166_116693

/-- Proves the remaining time of a trip given specific conditions -/
theorem remaining_trip_time 
  (total_time : ℝ) 
  (original_speed : ℝ) 
  (first_part_time : ℝ) 
  (first_part_speed : ℝ) 
  (remaining_speed : ℝ) 
  (h1 : total_time = 7.25)
  (h2 : original_speed = 50)
  (h3 : first_part_time = 2)
  (h4 : first_part_speed = 80)
  (h5 : remaining_speed = 40) :
  let total_distance := total_time * original_speed
  let first_part_distance := first_part_time * first_part_speed
  let remaining_distance := total_distance - first_part_distance
  remaining_distance / remaining_speed = 5.0625 := by
  sorry

end NUMINAMATH_CALUDE_remaining_trip_time_l1166_116693


namespace NUMINAMATH_CALUDE_sales_theorem_l1166_116607

def sales_problem (sales1 sales2 sales3 sales5 sales6 : ℕ) (average : ℕ) : Prop :=
  let total_sales := average * 6
  let known_sales := sales1 + sales2 + sales3 + sales5 + sales6
  let sales4 := total_sales - known_sales
  sales4 = 11707

theorem sales_theorem :
  sales_problem 5266 5768 5922 6029 4937 5600 :=
by
  sorry

end NUMINAMATH_CALUDE_sales_theorem_l1166_116607


namespace NUMINAMATH_CALUDE_smallest_cube_ending_576_l1166_116612

theorem smallest_cube_ending_576 : 
  ∀ n : ℕ, n > 0 → n < 706 → n^3 % 1000 ≠ 576 ∧ 706^3 % 1000 = 576 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_576_l1166_116612


namespace NUMINAMATH_CALUDE_midline_leg_relation_l1166_116698

/-- A right triangle with legs a and b, and midlines K₁ and K₂. -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  K₁ : ℝ
  K₂ : ℝ
  a_positive : 0 < a
  b_positive : 0 < b
  K₁_eq : K₁^2 = (a/2)^2 + b^2
  K₂_eq : K₂^2 = a^2 + (b/2)^2

/-- The main theorem about the relationship between midlines and leg in a right triangle. -/
theorem midline_leg_relation (t : RightTriangle) : 16 * t.K₂^2 - 4 * t.K₁^2 = 15 * t.a^2 := by
  sorry

end NUMINAMATH_CALUDE_midline_leg_relation_l1166_116698


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1166_116609

theorem cubic_roots_sum_cubes (p q r : ℝ) : 
  (3 * p^3 + 4 * p^2 - 200 * p + 5 = 0) →
  (3 * q^3 + 4 * q^2 - 200 * q + 5 = 0) →
  (3 * r^3 + 4 * r^2 - 200 * r + 5 = 0) →
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 24 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l1166_116609


namespace NUMINAMATH_CALUDE_line_equation_l1166_116689

/-- The equation of a line with slope 2 passing through the point (0, 3) is y = 2x + 3 -/
theorem line_equation (l : Set (ℝ × ℝ)) (slope : ℝ) (point : ℝ × ℝ) : 
  slope = 2 → 
  point = (0, 3) → 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ y = 2*x + 3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1166_116689


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_l1166_116662

-- Define the sets A and B
def A : Set ℝ := {x | (x + 3) / (x - 7) < 0}
def B : Set ℝ := {x | |x - 4| ≤ 6}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 7} := by sorry

-- Theorem for part (2)
theorem complement_union_A_B : (A ∪ B)ᶜ = {x : ℝ | x ≤ -3 ∨ x > 10} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_l1166_116662


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l1166_116699

-- Define the circles
def circle_C1 (m : ℝ) (x y : ℝ) : Prop := (x - m)^2 + (y + 2)^2 = 9
def circle_C2 (m : ℝ) (x y : ℝ) : Prop := (x + 1)^2 + (y - m)^2 = 4

-- Define the condition for internal tangency
def internally_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C1 m x y ∧ circle_C2 m x y

-- Theorem statement
theorem circles_internally_tangent :
  ∀ m : ℝ, internally_tangent m ↔ m = -2 ∨ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l1166_116699


namespace NUMINAMATH_CALUDE_count_digit_six_is_280_l1166_116642

/-- Count of digit 6 in integers from 100 to 999 -/
def count_digit_six : ℕ :=
  let hundreds := 100  -- 600 to 699
  let tens := 9 * 10   -- 10 numbers per hundred, 9 hundreds
  let ones := 9 * 10   -- 10 numbers per hundred, 9 hundreds
  hundreds + tens + ones

/-- The count of digit 6 in integers from 100 to 999 is 280 -/
theorem count_digit_six_is_280 : count_digit_six = 280 := by
  sorry

end NUMINAMATH_CALUDE_count_digit_six_is_280_l1166_116642


namespace NUMINAMATH_CALUDE_square_of_105_l1166_116617

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_square_of_105_l1166_116617


namespace NUMINAMATH_CALUDE_haley_trees_count_l1166_116636

/-- The number of trees that died after the typhoon -/
def dead_trees : ℕ := 5

/-- The number of trees left after the typhoon -/
def remaining_trees : ℕ := 12

/-- The total number of trees Haley grew -/
def total_trees : ℕ := dead_trees + remaining_trees

theorem haley_trees_count : total_trees = 17 := by
  sorry

end NUMINAMATH_CALUDE_haley_trees_count_l1166_116636


namespace NUMINAMATH_CALUDE_custom_operations_result_l1166_116626

def star (a b : ℤ) : ℤ := a + b - 1

def hash (a b : ℤ) : ℤ := a * b - 1

theorem custom_operations_result : (star (star 6 8) (hash 3 5)) = 26 := by
  sorry

end NUMINAMATH_CALUDE_custom_operations_result_l1166_116626


namespace NUMINAMATH_CALUDE_valentines_remaining_l1166_116633

theorem valentines_remaining (initial : ℕ) (children neighbors coworkers : ℕ) :
  initial ≥ children + neighbors + coworkers →
  initial - (children + neighbors + coworkers) =
  initial - children - neighbors - coworkers :=
by sorry

end NUMINAMATH_CALUDE_valentines_remaining_l1166_116633


namespace NUMINAMATH_CALUDE_win_sector_area_l1166_116695

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/4) :
  p * (π * r^2) = 36 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l1166_116695


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l1166_116647

theorem value_of_a_minus_b (a b : ℚ) 
  (eq1 : 3015 * a + 3021 * b = 3025)
  (eq2 : 3017 * a + 3023 * b = 3027) : 
  a - b = -7/3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l1166_116647


namespace NUMINAMATH_CALUDE_unique_c_for_complex_magnitude_l1166_116632

theorem unique_c_for_complex_magnitude : ∃! c : ℝ, Complex.abs (1 - 2 * c * Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_for_complex_magnitude_l1166_116632


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1166_116639

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 - x = 0} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1166_116639


namespace NUMINAMATH_CALUDE_sons_age_l1166_116660

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 27 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 25 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l1166_116660


namespace NUMINAMATH_CALUDE_anglets_in_sixth_circle_is_6000_l1166_116649

-- Define constants
def full_circle_degrees : ℕ := 360
def anglets_per_degree : ℕ := 100

-- Define the number of anglets in a sixth of a circle
def anglets_in_sixth_circle : ℕ := (full_circle_degrees / 6) * anglets_per_degree

-- Theorem statement
theorem anglets_in_sixth_circle_is_6000 : anglets_in_sixth_circle = 6000 := by
  sorry

end NUMINAMATH_CALUDE_anglets_in_sixth_circle_is_6000_l1166_116649


namespace NUMINAMATH_CALUDE_equation_is_linear_l1166_116654

/-- Definition of a linear equation with two variables -/
def is_linear_equation_two_vars (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation 3x = 2y -/
def equation (x y : ℝ) : Prop := 3 * x = 2 * y

/-- Theorem: The equation 3x = 2y is a linear equation with two variables -/
theorem equation_is_linear : is_linear_equation_two_vars equation := by
  sorry

end NUMINAMATH_CALUDE_equation_is_linear_l1166_116654


namespace NUMINAMATH_CALUDE_complex_argument_bounds_l1166_116618

variable (b : ℝ) (hb : b ≠ 0)
variable (y : ℂ)

theorem complex_argument_bounds :
  (Complex.abs (b * y + y⁻¹) = Real.sqrt 2) →
  (Complex.arg y = π / 4 ∨ Complex.arg y = 7 * π / 4) ∧
  (∀ z : ℂ, Complex.abs (b * z + z⁻¹) = Real.sqrt 2 →
    π / 4 ≤ Complex.arg z ∧ Complex.arg z ≤ 7 * π / 4) :=
by sorry

end NUMINAMATH_CALUDE_complex_argument_bounds_l1166_116618


namespace NUMINAMATH_CALUDE_team_selection_count_l1166_116648

/-- The number of ways to select a team of 8 members with an equal number of boys and girls
    from a group of 10 boys and 12 girls. -/
def select_team (total_boys : ℕ) (total_girls : ℕ) (team_size : ℕ) : ℕ :=
  Nat.choose total_boys (team_size / 2) * Nat.choose total_girls (team_size / 2)

/-- Theorem stating that the number of ways to select the team is 103950. -/
theorem team_selection_count :
  select_team 10 12 8 = 103950 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l1166_116648
