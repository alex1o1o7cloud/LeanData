import Mathlib

namespace NUMINAMATH_CALUDE_incenter_distance_l358_35831

/-- Given a triangle PQR with sides PQ = 12, PR = 13, QR = 15, and incenter J, 
    the length of PJ is 7√2. -/
theorem incenter_distance (P Q R J : ℝ × ℝ) : 
  let d := (λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
  (d P Q = 12) → (d P R = 13) → (d Q R = 15) → 
  (J.1 = (P.1 + Q.1 + R.1) / 3) → (J.2 = (P.2 + Q.2 + R.2) / 3) →
  d P J = 7 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_incenter_distance_l358_35831


namespace NUMINAMATH_CALUDE_unique_solution_condition_l358_35858

/-- The equation has exactly one solution if and only if a is in the specified set -/
theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, a * |2 + x| + (x^2 + x - 12) / (x + 4) = 0) ↔ 
  (a ∈ Set.Ioc (-1) 1 ∪ {7/2}) := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l358_35858


namespace NUMINAMATH_CALUDE_slower_train_speed_l358_35837

/-- Proves that the speed of the slower train is 36 km/hr given the conditions of the problem --/
theorem slower_train_speed (faster_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) :
  faster_speed = 46 →
  passing_time = 36 →
  train_length = 50 →
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * passing_time * (1000 / 3600) = 2 * train_length :=
by
  sorry


end NUMINAMATH_CALUDE_slower_train_speed_l358_35837


namespace NUMINAMATH_CALUDE_middle_part_length_l358_35802

/-- Given a road of length 28 km divided into three parts, if the distance between
    the midpoints of the outer parts is 16 km, then the length of the middle part is 4 km. -/
theorem middle_part_length
  (total_length : ℝ)
  (part1 part2 part3 : ℝ)
  (h_total : total_length = 28)
  (h_parts : part1 + part2 + part3 = total_length)
  (h_midpoints : |((part1 + part2 + part3/2) - part1/2)| = 16) :
  part2 = 4 := by
sorry

end NUMINAMATH_CALUDE_middle_part_length_l358_35802


namespace NUMINAMATH_CALUDE_set_equals_interval_l358_35886

-- Define the set S as {x | -1 < x ≤ 3}
def S : Set ℝ := {x | -1 < x ∧ x ≤ 3}

-- Define the interval (-1,3]
def I : Set ℝ := Set.Ioc (-1) 3

-- Theorem statement
theorem set_equals_interval : S = I := by sorry

end NUMINAMATH_CALUDE_set_equals_interval_l358_35886


namespace NUMINAMATH_CALUDE_solution_set_implies_k_empty_solution_set_implies_k_range_l358_35855

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2*x + 3*k

-- Part 1
theorem solution_set_implies_k (k : ℝ) :
  (∀ x, f k x < 0 ↔ x < -3 ∨ x > -1) → k = -1/2 := by sorry

-- Part 2
theorem empty_solution_set_implies_k_range (k : ℝ) :
  (∀ x, ¬(f k x < 0)) → 0 < k ∧ k ≤ Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_implies_k_empty_solution_set_implies_k_range_l358_35855


namespace NUMINAMATH_CALUDE_sum_of_constants_l358_35885

/-- Given constants a, b, and c satisfying the conditions, prove that a + 2b + 3c = 65 -/
theorem sum_of_constants (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ x < -4 ∨ |x - 25| ≤ 2)
  (h2 : a < b) : 
  a + 2*b + 3*c = 65 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_constants_l358_35885


namespace NUMINAMATH_CALUDE_ellipse_trace_l358_35832

/-- Given a complex number z with |z| = 3, the locus of points (x, y) satisfying z + 2/z = x + yi forms an ellipse -/
theorem ellipse_trace (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), z + 2 / z = x + y * Complex.I ↔ x^2 / a^2 + y^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_trace_l358_35832


namespace NUMINAMATH_CALUDE_original_number_is_ten_l358_35818

theorem original_number_is_ten : 
  ∃ x : ℝ, (2 * x + 5 = x / 2 + 20) ∧ (x = 10) := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_ten_l358_35818


namespace NUMINAMATH_CALUDE_triangle_inequality_satisfied_l358_35811

theorem triangle_inequality_satisfied (a b c : ℝ) (ha : a = 8) (hb : b = 8) (hc : c = 15) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_satisfied_l358_35811


namespace NUMINAMATH_CALUDE_prime_factor_sum_l358_35859

theorem prime_factor_sum (w x y z : ℕ) 
  (h : 2^w * 3^x * 5^y * 11^z = 825) : 
  w + 2*x + 3*y + 4*z = 12 := by
sorry

end NUMINAMATH_CALUDE_prime_factor_sum_l358_35859


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l358_35828

/-- Given a linear function y = 2x + 1 and two points (-1, y₁) and (3, y₂) on its graph,
    prove that y₁ < y₂ -/
theorem y1_less_than_y2 (y₁ y₂ : ℝ) : 
  (y₁ = 2 * (-1) + 1) → (y₂ = 2 * 3 + 1) → y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l358_35828


namespace NUMINAMATH_CALUDE_rationalize_denominator_l358_35899

theorem rationalize_denominator :
  (Real.sqrt 18 + Real.sqrt 2) / (Real.sqrt 3 + Real.sqrt 2) = 4 * (Real.sqrt 6 - 2) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l358_35899


namespace NUMINAMATH_CALUDE_sunflower_majority_on_wednesday_sunflower_proportion_increases_l358_35881

/-- Represents the amount of sunflower seeds on a given day -/
def sunflower_seeds (day : ℕ) : ℝ :=
  3 * (1 - (0.8 ^ day))

/-- Represents the total amount of seeds on any day after adding new seeds -/
def total_seeds : ℝ := 2

/-- Theorem stating that Wednesday (day 3) is the first day when sunflower seeds exceed half of total seeds -/
theorem sunflower_majority_on_wednesday :
  (∀ d < 3, sunflower_seeds d ≤ total_seeds / 2) ∧
  (sunflower_seeds 3 > total_seeds / 2) :=
by sorry

/-- Helper theorem: The proportion of sunflower seeds increases each day -/
theorem sunflower_proportion_increases (d : ℕ) :
  sunflower_seeds d < sunflower_seeds (d + 1) :=
by sorry

end NUMINAMATH_CALUDE_sunflower_majority_on_wednesday_sunflower_proportion_increases_l358_35881


namespace NUMINAMATH_CALUDE_marbles_cost_calculation_l358_35883

/-- The amount spent on marbles, given the total spent on toys and the cost of a football -/
def marbles_cost (total_spent : ℚ) (football_cost : ℚ) : ℚ :=
  total_spent - football_cost

/-- Theorem stating that the amount spent on marbles is $6.59 -/
theorem marbles_cost_calculation :
  marbles_cost 12.30 5.71 = 6.59 := by
  sorry

end NUMINAMATH_CALUDE_marbles_cost_calculation_l358_35883


namespace NUMINAMATH_CALUDE_distance_equals_speed_times_time_l358_35808

/-- The distance between Patrick's house and Aaron's house -/
def distance : ℝ := 14

/-- The time Patrick spent jogging -/
def time : ℝ := 2

/-- Patrick's jogging speed -/
def speed : ℝ := 7

/-- Theorem stating that the distance is equal to speed multiplied by time -/
theorem distance_equals_speed_times_time : distance = speed * time := by
  sorry

end NUMINAMATH_CALUDE_distance_equals_speed_times_time_l358_35808


namespace NUMINAMATH_CALUDE_super_champion_tournament_24_teams_l358_35887

/-- The number of games played in a tournament with a given number of teams --/
def tournament_games (n : ℕ) : ℕ :=
  n - 1

/-- The total number of games in a tournament with a "Super Champion" game --/
def super_champion_tournament (n : ℕ) : ℕ :=
  tournament_games n + 1

/-- Theorem: A tournament with 24 teams and a "Super Champion" game has 24 total games --/
theorem super_champion_tournament_24_teams :
  super_champion_tournament 24 = 24 := by
  sorry

end NUMINAMATH_CALUDE_super_champion_tournament_24_teams_l358_35887


namespace NUMINAMATH_CALUDE_stockholm_malmo_distance_l358_35856

/-- The road distance between Stockholm and Malmo in kilometers -/
def road_distance (map_distance : ℝ) (scale : ℝ) (road_factor : ℝ) : ℝ :=
  map_distance * scale * road_factor

/-- Theorem: The road distance between Stockholm and Malmo is 1380 km -/
theorem stockholm_malmo_distance :
  road_distance 120 10 1.15 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_malmo_distance_l358_35856


namespace NUMINAMATH_CALUDE_perfect_square_condition_l358_35864

theorem perfect_square_condition (Z K : ℤ) : 
  (500 < Z ∧ Z < 1000) →
  K > 1 →
  Z = K * K^2 →
  (∃ n : ℤ, Z = n^2) →
  K = 9 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l358_35864


namespace NUMINAMATH_CALUDE_triangle_inequality_in_necklace_l358_35852

theorem triangle_inequality_in_necklace :
  ∀ (a : ℕ → ℕ),
  (∀ n, 290 ≤ a n ∧ a n ≤ 2023) →
  (∀ m n, m ≠ n → a m ≠ a n) →
  ∃ i, a i + a (i + 1) > a (i + 2) ∧
       a i + a (i + 2) > a (i + 1) ∧
       a (i + 1) + a (i + 2) > a i :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_in_necklace_l358_35852


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_at_midpoint_l358_35846

theorem line_intersects_ellipse_at_midpoint (x y : ℝ) :
  let P : ℝ × ℝ := (1, 1)
  let ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1
  let line (x y : ℝ) : Prop := 4*x + 9*y = 13
  (∀ x y, line x y → (x, y) = P ∨ ellipse x y) ∧
  (∃ A B : ℝ × ℝ, A ≠ B ∧ line A.1 A.2 ∧ line B.1 B.2 ∧ 
    ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
    ((A.1 + B.1)/2, (A.2 + B.2)/2) = P) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_at_midpoint_l358_35846


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l358_35826

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 231 → n = 22 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l358_35826


namespace NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_two_l358_35844

theorem abs_ratio_eq_sqrt_two (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 6*a*b) :
  |((a + b) / (a - b))| = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_two_l358_35844


namespace NUMINAMATH_CALUDE_min_value_on_line_l358_35876

theorem min_value_on_line (x y : ℝ) : 
  x + y = 4 → ∀ a b : ℝ, a + b = 4 → x^2 + y^2 ≤ a^2 + b^2 ∧ ∃ c d : ℝ, c + d = 4 ∧ c^2 + d^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_line_l358_35876


namespace NUMINAMATH_CALUDE_yellow_shirt_pairs_l358_35813

theorem yellow_shirt_pairs (total_students : ℕ) (blue_shirts : ℕ) (yellow_shirts : ℕ) 
  (total_pairs : ℕ) (blue_pairs : ℕ) 
  (h1 : total_students = 132)
  (h2 : blue_shirts = 57)
  (h3 : yellow_shirts = 75)
  (h4 : total_pairs = 66)
  (h5 : blue_pairs = 23)
  (h6 : total_students = blue_shirts + yellow_shirts)
  (h7 : blue_pairs ≤ total_pairs) :
  ∃ yellow_pairs : ℕ, yellow_pairs = total_pairs - blue_pairs - (blue_shirts - 2 * blue_pairs) :=
by sorry

end NUMINAMATH_CALUDE_yellow_shirt_pairs_l358_35813


namespace NUMINAMATH_CALUDE_molecular_weight_c4h10_l358_35871

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The number of Carbon atoms in C4H10 -/
def carbon_count : ℕ := 4

/-- The number of Hydrogen atoms in C4H10 -/
def hydrogen_count : ℕ := 10

/-- The number of moles of C4H10 -/
def mole_count : ℝ := 6

/-- Theorem: The molecular weight of 6 moles of C4H10 is 348.72 grams -/
theorem molecular_weight_c4h10 :
  (carbon_weight * carbon_count + hydrogen_weight * hydrogen_count) * mole_count = 348.72 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_c4h10_l358_35871


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l358_35833

def f (x : ℤ) : ℤ := x^3 - 4*x^2 - 14*x + 24

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, f x = 0 ↔ x = -1 ∨ x = 3 ∨ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l358_35833


namespace NUMINAMATH_CALUDE_percentage_of_employed_females_l358_35829

-- Define the given percentages
def total_employed_percent : ℝ := 64
def employed_males_percent : ℝ := 46

-- Define the theorem
theorem percentage_of_employed_females :
  (total_employed_percent - employed_males_percent) / total_employed_percent * 100 = 28.125 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_employed_females_l358_35829


namespace NUMINAMATH_CALUDE_sqrt_sum_given_diff_l358_35880

theorem sqrt_sum_given_diff (y : ℝ) : 
  Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4 → 
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_given_diff_l358_35880


namespace NUMINAMATH_CALUDE_decreasing_reciprocal_function_l358_35889

theorem decreasing_reciprocal_function (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : 0 < x₂) (h3 : x₁ < x₂) :
  (1 : ℝ) / x₁ > (1 : ℝ) / x₂ := by
  sorry

end NUMINAMATH_CALUDE_decreasing_reciprocal_function_l358_35889


namespace NUMINAMATH_CALUDE_carl_practice_hours_l358_35868

/-- The number of weeks Carl practices -/
def total_weeks : ℕ := 8

/-- The required average hours per week -/
def required_average : ℕ := 15

/-- The hours practiced in the first 7 weeks -/
def first_seven_weeks : List ℕ := [14, 16, 12, 18, 15, 13, 17]

/-- The sum of hours practiced in the first 7 weeks -/
def sum_first_seven : ℕ := first_seven_weeks.sum

/-- The number of hours Carl must practice in the 8th week -/
def hours_eighth_week : ℕ := 15

theorem carl_practice_hours :
  (sum_first_seven + hours_eighth_week) / total_weeks = required_average :=
sorry

end NUMINAMATH_CALUDE_carl_practice_hours_l358_35868


namespace NUMINAMATH_CALUDE_anna_sandwiches_l358_35867

theorem anna_sandwiches (slices_per_sandwich : ℕ) (current_slices : ℕ) (additional_slices : ℕ) :
  slices_per_sandwich = 3 →
  current_slices = 31 →
  additional_slices = 119 →
  (current_slices + additional_slices) / slices_per_sandwich = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_anna_sandwiches_l358_35867


namespace NUMINAMATH_CALUDE_smallest_n_for_coloring_property_l358_35834

def is_valid_coloring (n : ℕ) (coloring : ℕ → Bool) : Prop :=
  ∀ x y z w, x ≤ n ∧ y ≤ n ∧ z ≤ n ∧ w ≤ n →
    coloring x = coloring y ∧ coloring y = coloring z ∧ coloring z = coloring w →
    x + y + z ≠ w

theorem smallest_n_for_coloring_property : 
  (∀ n < 11, ∃ coloring, is_valid_coloring n coloring) ∧
  (∀ coloring, ¬ is_valid_coloring 11 coloring) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_coloring_property_l358_35834


namespace NUMINAMATH_CALUDE_simplify_expression_l358_35891

theorem simplify_expression (b c : ℝ) :
  3 * b * (3 * b^3 + 2 * b) - 2 * b^2 + c * (3 * b^2 - c) = 9 * b^4 + 4 * b^2 + 3 * b^2 * c - c^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l358_35891


namespace NUMINAMATH_CALUDE_remainder_problem_l358_35872

theorem remainder_problem (D : ℕ) (h1 : D = 13) (h2 : 698 % D = 9) (h3 : (242 + 698) % D = 4) :
  242 % D = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l358_35872


namespace NUMINAMATH_CALUDE_laptop_price_exceeds_savings_l358_35850

/-- Proves that for any initial laptop price greater than 0, 
    after 2 years of 6% annual price increase, 
    the laptop price will exceed 56358 rubles -/
theorem laptop_price_exceeds_savings (P₀ : ℝ) (h : P₀ > 0) : 
  P₀ * (1 + 0.06)^2 > 56358 := by
  sorry

#check laptop_price_exceeds_savings

end NUMINAMATH_CALUDE_laptop_price_exceeds_savings_l358_35850


namespace NUMINAMATH_CALUDE_number_of_boys_l358_35819

theorem number_of_boys (total : ℕ) (boys : ℕ) : 
  total = 900 →
  (total - boys : ℚ) = (boys : ℚ) * (total : ℚ) / 100 →
  boys = 90 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l358_35819


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_half_l358_35841

theorem arctan_sum_equals_pi_half (n : ℕ+) :
  Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/n) = π/2 ↔ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_half_l358_35841


namespace NUMINAMATH_CALUDE_toucan_count_l358_35804

theorem toucan_count (initial_toucans : ℕ) (joining_toucans : ℕ) : 
  initial_toucans = 2 → joining_toucans = 1 → initial_toucans + joining_toucans = 3 := by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l358_35804


namespace NUMINAMATH_CALUDE_max_constant_inequality_l358_35840

theorem max_constant_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x^2 + y^2 = 1) :
  ∃ (c : ℝ), c = 1/2 ∧ x^6 + y^6 ≥ c*x*y ∧ ∀ (d : ℝ), (∀ (a b : ℝ), a > 0 → b > 0 → a^2 + b^2 = 1 → a^6 + b^6 ≥ d*a*b) → d ≤ c :=
by sorry

end NUMINAMATH_CALUDE_max_constant_inequality_l358_35840


namespace NUMINAMATH_CALUDE_johns_father_age_difference_l358_35890

/-- Given John's age and the sum of John and his father's ages, 
    prove the difference between John's father's age and twice John's age. -/
theorem johns_father_age_difference (john_age : ℕ) (sum_ages : ℕ) 
    (h1 : john_age = 15)
    (h2 : john_age + (2 * john_age + sum_ages - john_age) = sum_ages)
    (h3 : sum_ages = 77) : 
  (2 * john_age + sum_ages - john_age) - 2 * john_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_johns_father_age_difference_l358_35890


namespace NUMINAMATH_CALUDE_octal_computation_l358_35806

/-- Converts a decimal number to its octal representation -/
def toOctal (n : ℕ) : ℕ := sorry

/-- Multiplies two octal numbers -/
def octalMultiply (a b : ℕ) : ℕ := sorry

/-- Divides an octal number by another octal number -/
def octalDivide (a b : ℕ) : ℕ := sorry

theorem octal_computation : 
  let a := toOctal 254
  let b := toOctal 170
  let c := toOctal 4
  octalDivide (octalMultiply a b) c = 3156 := by sorry

end NUMINAMATH_CALUDE_octal_computation_l358_35806


namespace NUMINAMATH_CALUDE_diamond_ratio_equals_three_fifths_l358_35843

-- Define the ♢ operation
def diamond (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem diamond_ratio_equals_three_fifths :
  (diamond 5 3) / (diamond 3 5 : ℚ) = 3/5 := by sorry

end NUMINAMATH_CALUDE_diamond_ratio_equals_three_fifths_l358_35843


namespace NUMINAMATH_CALUDE_student_count_proof_l358_35877

theorem student_count_proof (initial_avg : ℝ) (new_student_weight : ℝ) (final_avg : ℝ) :
  initial_avg = 28 →
  new_student_weight = 7 →
  final_avg = 27.3 →
  (∃ n : ℕ, (n : ℝ) * initial_avg + new_student_weight = (n + 1 : ℝ) * final_avg ∧ n = 29) :=
by sorry

end NUMINAMATH_CALUDE_student_count_proof_l358_35877


namespace NUMINAMATH_CALUDE_sum_interior_angles_decagon_l358_35860

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A decagon is a polygon with 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The sum of the interior angles of a decagon is 1440 degrees -/
theorem sum_interior_angles_decagon :
  sum_interior_angles decagon_sides = 1440 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_decagon_l358_35860


namespace NUMINAMATH_CALUDE_rabbit_fur_genetics_l358_35827

/-- Represents the phase of meiotic division --/
inductive MeioticPhase
  | LateFirst
  | LateSecond

/-- Represents the fur length gene --/
inductive FurGene
  | Long
  | Short

/-- Represents a rabbit's genotype for fur length --/
structure RabbitGenotype where
  allele1 : FurGene
  allele2 : FurGene

/-- Represents the genetic characteristics of rabbit fur --/
structure RabbitFurGenetics where
  totalGenes : ℕ
  genesPerOocyte : ℕ
  nucleotideTypes : ℕ
  separationPhase : MeioticPhase

def isHeterozygous (genotype : RabbitGenotype) : Prop :=
  genotype.allele1 ≠ genotype.allele2

def maxShortFurOocytes (genetics : RabbitFurGenetics) (genotype : RabbitGenotype) : ℕ :=
  genetics.totalGenes / genetics.genesPerOocyte

theorem rabbit_fur_genetics 
  (genetics : RabbitFurGenetics) 
  (genotype : RabbitGenotype) :
  isHeterozygous genotype →
  genetics.totalGenes = 20 →
  genetics.genesPerOocyte = 4 →
  genetics.nucleotideTypes = 4 →
  genetics.separationPhase = MeioticPhase.LateFirst →
  maxShortFurOocytes genetics genotype = 5 :=
by sorry

end NUMINAMATH_CALUDE_rabbit_fur_genetics_l358_35827


namespace NUMINAMATH_CALUDE_square_properties_l358_35863

/-- Given a square with diagonal length 12√2 cm, prove its perimeter and inscribed circle area -/
theorem square_properties (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  let perimeter := 4 * s
  let r := s / 2
  let inscribed_circle_area := Real.pi * r^2
  (perimeter = 48 ∧ inscribed_circle_area = 36 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_square_properties_l358_35863


namespace NUMINAMATH_CALUDE_boat_license_combinations_l358_35857

/-- The number of possible letters for the first character of a boat license -/
def numLetters : ℕ := 3

/-- The number of possible digits for each of the six numeric positions in a boat license -/
def numDigits : ℕ := 10

/-- The number of numeric positions in a boat license -/
def numPositions : ℕ := 6

/-- The total number of unique boat license combinations -/
def totalCombinations : ℕ := numLetters * (numDigits ^ numPositions)

theorem boat_license_combinations :
  totalCombinations = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_combinations_l358_35857


namespace NUMINAMATH_CALUDE_problem_solution_l358_35805

theorem problem_solution (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : a ^ b = b ^ a) (h4 : b = 27 * a) : a = (27 : ℝ) ^ (1 / 26) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l358_35805


namespace NUMINAMATH_CALUDE_angle_double_complement_measure_l358_35810

theorem angle_double_complement_measure : ∀ x : ℝ, 
  (x = 2 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_double_complement_measure_l358_35810


namespace NUMINAMATH_CALUDE_common_number_in_list_l358_35892

theorem common_number_in_list (l : List ℝ) 
  (h_length : l.length = 7)
  (h_avg_first : (l.take 4).sum / 4 = 7)
  (h_avg_last : (l.drop 3).sum / 4 = 9)
  (h_avg_all : l.sum / 7 = 8) :
  ∃ x ∈ l.take 4 ∩ l.drop 3, x = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_number_in_list_l358_35892


namespace NUMINAMATH_CALUDE_ages_solution_l358_35848

/-- Represents the ages of three persons --/
structure Ages where
  youngest : ℕ
  middle : ℕ
  eldest : ℕ

/-- Checks if the given ages satisfy the problem conditions --/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.eldest = ages.middle + 16 ∧
  ages.middle = ages.youngest + 8 ∧
  ages.eldest - 6 = 3 * (ages.youngest - 6) ∧
  ages.eldest - 6 = 2 * (ages.middle - 6)

/-- Theorem stating that the ages 18, 26, and 42 satisfy the problem conditions --/
theorem ages_solution :
  ∃ (ages : Ages), satisfiesConditions ages ∧ 
    ages.youngest = 18 ∧ ages.middle = 26 ∧ ages.eldest = 42 := by
  sorry

end NUMINAMATH_CALUDE_ages_solution_l358_35848


namespace NUMINAMATH_CALUDE_waterfall_flow_rate_l358_35898

/-- The waterfall problem -/
theorem waterfall_flow_rate 
  (basin_capacity : ℝ) 
  (leak_rate : ℝ) 
  (fill_time : ℝ) 
  (h1 : basin_capacity = 260) 
  (h2 : leak_rate = 4) 
  (h3 : fill_time = 13) : 
  ∃ (flow_rate : ℝ), flow_rate = 24 ∧ 
  fill_time * flow_rate - fill_time * leak_rate = basin_capacity :=
sorry

end NUMINAMATH_CALUDE_waterfall_flow_rate_l358_35898


namespace NUMINAMATH_CALUDE_max_gcd_of_product_7200_l358_35842

theorem max_gcd_of_product_7200 :
  ∃ (a b : ℕ), a * b = 7200 ∧ 
  ∀ (c d : ℕ), c * d = 7200 → Nat.gcd c d ≤ 60 ∧
  Nat.gcd a b = 60 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_of_product_7200_l358_35842


namespace NUMINAMATH_CALUDE_arithmetic_progression_result_l358_35861

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℕ → ℝ  -- The nth term of the progression
  S : ℕ → ℝ  -- The sum of the first n terms

/-- Theorem stating the result for the given arithmetic progression -/
theorem arithmetic_progression_result (ap : ArithmeticProgression) 
  (h1 : ap.a 1 + ap.a 3 = 5)
  (h2 : ap.S 4 = 20) :
  (ap.S 8 - 2 * ap.S 4) / (ap.S 6 - ap.S 4 - ap.S 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_result_l358_35861


namespace NUMINAMATH_CALUDE_rob_unique_cards_rob_doubles_ratio_jess_doubles_ratio_alex_doubles_ratio_l358_35853

-- Define the friends
structure Friend where
  name : String
  total_cards : ℕ
  doubles : ℕ

-- Define the problem setup
def rob : Friend := { name := "Rob", total_cards := 24, doubles := 8 }
def jess : Friend := { name := "Jess", total_cards := 0, doubles := 40 } -- total_cards unknown
def alex : Friend := { name := "Alex", total_cards := 0, doubles := 0 } -- both unknown

-- Theorem: Rob has 16 unique cards
theorem rob_unique_cards :
  rob.total_cards - rob.doubles = 16 :=
by
  sorry

-- Conditions from the problem
theorem rob_doubles_ratio :
  3 * rob.doubles = rob.total_cards :=
by
  sorry

theorem jess_doubles_ratio :
  jess.doubles = 5 * rob.doubles :=
by
  sorry

theorem alex_doubles_ratio (alex_total : ℕ) :
  4 * alex.doubles = alex_total :=
by
  sorry

end NUMINAMATH_CALUDE_rob_unique_cards_rob_doubles_ratio_jess_doubles_ratio_alex_doubles_ratio_l358_35853


namespace NUMINAMATH_CALUDE_quadratic_uniqueness_l358_35869

/-- A quadratic function is uniquely determined by three distinct points -/
theorem quadratic_uniqueness (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) :
  ∃! (a b c : ℝ), 
    y₁ = a * x₁^2 + b * x₁ + c ∧
    y₂ = a * x₂^2 + b * x₂ + c ∧
    y₃ = a * x₃^2 + b * x₃ + c := by
  sorry

#check quadratic_uniqueness

end NUMINAMATH_CALUDE_quadratic_uniqueness_l358_35869


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l358_35874

theorem decimal_sum_to_fraction : 
  0.4 + 0.05 + 0.006 + 0.0007 + 0.00008 = 22839 / 50000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l358_35874


namespace NUMINAMATH_CALUDE_bus_profit_analysis_l358_35807

/-- Represents the monthly profit of a bus service -/
def monthly_profit (passengers : ℕ) : ℤ :=
  2 * passengers - 4000

theorem bus_profit_analysis :
  let break_even := 2000
  let profit_4230 := monthly_profit 4230
  -- 1. Independent variable is passengers, dependent is profit (implicit in the function definition)
  -- 2. Break-even point
  monthly_profit break_even = 0 ∧
  -- 3. Profit for 4230 passengers
  profit_4230 = 4460 := by
  sorry

end NUMINAMATH_CALUDE_bus_profit_analysis_l358_35807


namespace NUMINAMATH_CALUDE_only_four_solutions_l358_35825

/-- A digit is a natural number from 0 to 9. -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- Convert a repeating decimal 0.aaaaa... to a fraction a/9. -/
def repeatingDecimalToFraction (a : Digit) : ℚ := a.val / 9

/-- The property that a pair of digits (a,b) satisfies √(0.aaaaa...) = 0.bbbbb... -/
def SatisfiesEquation (a b : Digit) : Prop :=
  (repeatingDecimalToFraction b) ^ 2 = repeatingDecimalToFraction a

/-- The theorem stating that only four specific digit pairs satisfy the equation. -/
theorem only_four_solutions :
  ∀ a b : Digit, SatisfiesEquation a b ↔ 
    ((a.val = 0 ∧ b.val = 0) ∨
     (a.val = 1 ∧ b.val = 3) ∨
     (a.val = 4 ∧ b.val = 6) ∨
     (a.val = 9 ∧ b.val = 9)) :=
by sorry

end NUMINAMATH_CALUDE_only_four_solutions_l358_35825


namespace NUMINAMATH_CALUDE_trash_in_classrooms_l358_35845

theorem trash_in_classrooms (total_trash : ℕ) (outside_trash : ℕ) 
  (h1 : total_trash = 1576) 
  (h2 : outside_trash = 1232) : 
  total_trash - outside_trash = 344 := by
  sorry

end NUMINAMATH_CALUDE_trash_in_classrooms_l358_35845


namespace NUMINAMATH_CALUDE_difference_of_squares_l358_35865

theorem difference_of_squares (x y : ℝ) : x^2 - 25*y^2 = (x - 5*y) * (x + 5*y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l358_35865


namespace NUMINAMATH_CALUDE_smallest_factor_l358_35893

theorem smallest_factor (n : ℕ) : 
  (∀ m : ℕ, m > 0 → 
    (2^4 ∣ 1452 * m) ∧ 
    (3^3 ∣ 1452 * m) ∧ 
    (13^3 ∣ 1452 * m) → 
    m ≥ n) ↔ 
  n = 676 := by
sorry

end NUMINAMATH_CALUDE_smallest_factor_l358_35893


namespace NUMINAMATH_CALUDE_monkey_climb_l358_35870

/-- The height of the tree climbed by the monkey -/
def tree_height : ℕ := 21

/-- The net progress of the monkey per hour -/
def net_progress_per_hour : ℕ := 1

/-- The time taken by the monkey to reach the top of the tree -/
def total_hours : ℕ := 19

/-- The distance the monkey hops up in the last hour -/
def last_hop : ℕ := 3

theorem monkey_climb :
  tree_height = net_progress_per_hour * (total_hours - 1) + last_hop := by
  sorry

end NUMINAMATH_CALUDE_monkey_climb_l358_35870


namespace NUMINAMATH_CALUDE_smallest_odd_triangle_perimeter_l358_35816

/-- A triangle with consecutive odd integer side lengths. -/
structure OddTriangle where
  a : ℕ
  is_odd : Odd a
  satisfies_inequality : a + (a + 2) > (a + 4) ∧ a + (a + 4) > (a + 2) ∧ (a + 2) + (a + 4) > a

/-- The perimeter of an OddTriangle. -/
def perimeter (t : OddTriangle) : ℕ := t.a + (t.a + 2) + (t.a + 4)

/-- The statement to be proven. -/
theorem smallest_odd_triangle_perimeter :
  (∃ t : OddTriangle, ∀ t' : OddTriangle, perimeter t ≤ perimeter t') ∧
  (∃ t : OddTriangle, perimeter t = 15) :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_triangle_perimeter_l358_35816


namespace NUMINAMATH_CALUDE_slope_determines_m_l358_35815

theorem slope_determines_m (m : ℝ) : 
  let A : ℝ × ℝ := (-m, 6)
  let B : ℝ × ℝ := (1, 3*m)
  (B.2 - A.2) / (B.1 - A.1) = 12 → m = -2 := by
sorry

end NUMINAMATH_CALUDE_slope_determines_m_l358_35815


namespace NUMINAMATH_CALUDE_certain_number_problem_l358_35822

theorem certain_number_problem : ∃ x : ℝ, 0.85 * x = (4/5 * 25) + 14 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l358_35822


namespace NUMINAMATH_CALUDE_coloring_satisfies_conditions_l358_35851

-- Define the color type
inductive Color
  | White
  | Red
  | Black

-- Define the coloring function
def color (x y : Int) : Color :=
  if (x + y) % 2 = 0 then Color.Red
  else if x % 2 = 1 && y % 2 = 0 then Color.White
  else Color.Black

-- Define a lattice point
structure LatticePoint where
  x : Int
  y : Int

-- Define a property that a color appears infinitely many times on infinitely many horizontal lines
def infiniteOccurrence (c : Color) : Prop :=
  ∀ (n : Nat), ∃ (m : Int), ∀ (k : Int), ∃ (x : Int), 
    color x (m + k * n) = c

-- Define the parallelogram property
def parallelogramProperty : Prop :=
  ∀ (A B C : LatticePoint),
    color A.x A.y = Color.White →
    color B.x B.y = Color.Red →
    color C.x C.y = Color.Black →
    ∃ (D : LatticePoint),
      color D.x D.y = Color.Red ∧
      D.x - C.x = A.x - B.x ∧
      D.y - C.y = A.y - B.y

-- The main theorem
theorem coloring_satisfies_conditions :
  (∀ c : Color, infiniteOccurrence c) ∧ parallelogramProperty :=
sorry

end NUMINAMATH_CALUDE_coloring_satisfies_conditions_l358_35851


namespace NUMINAMATH_CALUDE_total_running_time_l358_35888

def track_length : ℝ := 500
def num_laps : ℕ := 7
def first_section_length : ℝ := 200
def second_section_length : ℝ := 300
def first_section_speed : ℝ := 5
def second_section_speed : ℝ := 6

theorem total_running_time :
  (num_laps : ℝ) * (first_section_length / first_section_speed + second_section_length / second_section_speed) = 630 :=
by sorry

end NUMINAMATH_CALUDE_total_running_time_l358_35888


namespace NUMINAMATH_CALUDE_half_squared_is_quarter_l358_35878

theorem half_squared_is_quarter : (1/2)^2 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_half_squared_is_quarter_l358_35878


namespace NUMINAMATH_CALUDE_parabola_intersection_l358_35839

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 18
def g (x : ℝ) : ℝ := x^2 - 2 * x + 4

-- Define the intersection points
def p₁ : ℝ × ℝ := (-2, 12)
def p₂ : ℝ × ℝ := (5.5, 23.25)

-- Theorem statement
theorem parabola_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x, y) = p₁ ∨ (x, y) = p₂) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l358_35839


namespace NUMINAMATH_CALUDE_count_divisible_numbers_l358_35882

theorem count_divisible_numbers (n : ℕ) : 
  (Finset.filter (fun k => (k^2 - 1) % 291 = 0) (Finset.range (291000 + 1))).card = 4000 :=
sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_l358_35882


namespace NUMINAMATH_CALUDE_range_of_f_l358_35823

def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

theorem range_of_f :
  Set.range f = Set.Icc (-8) 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l358_35823


namespace NUMINAMATH_CALUDE_smallest_integer_x_l358_35897

theorem smallest_integer_x : ∃ x : ℤ, 
  (∀ z : ℤ, (7 - 5*z < 25 ∧ 10 - 3*z > 6) → x ≤ z) ∧ 
  (7 - 5*x < 25 ∧ 10 - 3*x > 6) ∧
  x = -3 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_x_l358_35897


namespace NUMINAMATH_CALUDE_one_third_of_x_l358_35814

theorem one_third_of_x (x y : ℚ) : 
  x / y = 15 / 3 → y = 24 → x / 3 = 40 := by sorry

end NUMINAMATH_CALUDE_one_third_of_x_l358_35814


namespace NUMINAMATH_CALUDE_telethon_total_money_telethon_specific_case_l358_35895

/-- Calculates the total money raised in a telethon with varying hourly rates --/
theorem telethon_total_money (first_period_hours : ℕ) (second_period_hours : ℕ) 
  (first_period_rate : ℕ) (rate_increase_percent : ℕ) : ℕ :=
  let total_hours := first_period_hours + second_period_hours
  let first_period_total := first_period_hours * first_period_rate
  let second_period_rate := first_period_rate + (first_period_rate * rate_increase_percent / 100)
  let second_period_total := second_period_hours * second_period_rate
  first_period_total + second_period_total

/-- Proves that the telethon raises $144,000 given the specific conditions --/
theorem telethon_specific_case : 
  telethon_total_money 12 14 5000 20 = 144000 := by
  sorry

end NUMINAMATH_CALUDE_telethon_total_money_telethon_specific_case_l358_35895


namespace NUMINAMATH_CALUDE_simplify_fraction_l358_35817

theorem simplify_fraction : (75 : ℚ) / 225 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l358_35817


namespace NUMINAMATH_CALUDE_problem_solution_l358_35854

noncomputable section

variables (a b c x : ℝ)

def f (x : ℝ) (b : ℝ) : ℝ := |x + b^2| - |-x + 1|
def g (x a b c : ℝ) : ℝ := |x + a^2 + c^2| + |x - 2*b^2|

theorem problem_solution (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a*b + b*c + a*c = 1) :
  (∀ x, f x 1 ≥ 1 ↔ x ∈ Set.Ici (1/2)) ∧
  (∀ x, f x b ≤ g x a b c) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l358_35854


namespace NUMINAMATH_CALUDE_locus_of_T_is_tangents_to_C_perp_to_L_l358_35849

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Represents a point in the plane -/
def Point := ℝ × ℝ

/-- The fixed circle C -/
def C : Circle := sorry

/-- The line L passing through the center of C -/
def L : Line := sorry

/-- A variable point P on L -/
def P : Point := sorry

/-- The circle K centered at P and passing through the center of C -/
def K : Circle := sorry

/-- A point T on K where a common tangent to C and K meets K -/
def T : Point := sorry

/-- The locus of point T -/
def locus_of_T : Set Point := sorry

/-- The pair of tangents to C which are perpendicular to L -/
def tangents_to_C_perp_to_L : Set Point := sorry

theorem locus_of_T_is_tangents_to_C_perp_to_L :
  locus_of_T = tangents_to_C_perp_to_L := by sorry

end NUMINAMATH_CALUDE_locus_of_T_is_tangents_to_C_perp_to_L_l358_35849


namespace NUMINAMATH_CALUDE_equation_solution_l358_35847

theorem equation_solution : ∃! y : ℝ, (128 : ℝ) ^ (y + 1) / (8 : ℝ) ^ (y + 1) = (64 : ℝ) ^ (3 * y - 2) ∧ y = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l358_35847


namespace NUMINAMATH_CALUDE_log_a_equals_three_l358_35894

theorem log_a_equals_three (a : ℝ) (h1 : a > 0) (h2 : a^(2/3) = 4/9) : 
  Real.log a / Real.log (2/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_a_equals_three_l358_35894


namespace NUMINAMATH_CALUDE_shirt_cost_l358_35862

theorem shirt_cost (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 86) : S = 24 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l358_35862


namespace NUMINAMATH_CALUDE_triangle_inequality_with_constant_l358_35830

theorem triangle_inequality_with_constant (k : ℕ) : 
  (k > 0) →
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 →
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
    a + b > c ∧ b + c > a ∧ c + a > b) ↔
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_constant_l358_35830


namespace NUMINAMATH_CALUDE_fraction_equality_l358_35896

theorem fraction_equality (X : ℚ) : (2/5 : ℚ) * (5/9 : ℚ) * X = 0.11111111111111112 → X = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l358_35896


namespace NUMINAMATH_CALUDE_special_function_property_l358_35875

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧
  (∀ x : ℝ, f (x + 5) ≥ f x + 5) ∧
  (∀ x : ℝ, f (x + 1) ≤ f x + 1)

/-- The function g defined in terms of f -/
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 1 - x

/-- The main theorem -/
theorem special_function_property (f : ℝ → ℝ) (hf : special_function f) :
  g f 2009 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l358_35875


namespace NUMINAMATH_CALUDE_solve_system_l358_35835

/-- Given a system of equations, prove that y and z have specific values -/
theorem solve_system (x y z : ℚ) 
  (eq1 : (x + y) / (z - x) = 9/2)
  (eq2 : (y + z) / (y - x) = 5)
  (eq3 : x = 43/4) :
  y = 305/17 ∧ z = 1165/68 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l358_35835


namespace NUMINAMATH_CALUDE_total_puppies_eq_sum_l358_35820

/-- The number of puppies Alyssa's dog had -/
def total_puppies : ℕ := 23

/-- The number of puppies Alyssa gave to her friends -/
def puppies_given_away : ℕ := 15

/-- The number of puppies Alyssa kept for herself -/
def puppies_kept : ℕ := 8

/-- Theorem stating that the total number of puppies is the sum of puppies given away and kept -/
theorem total_puppies_eq_sum : total_puppies = puppies_given_away + puppies_kept := by
  sorry

end NUMINAMATH_CALUDE_total_puppies_eq_sum_l358_35820


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l358_35836

theorem two_digit_number_difference (a b : ℕ) : 
  a ≥ 1 → a ≤ 9 → b ≤ 9 → (10 * a + b) - (10 * b + a) = 45 → a - b = 5 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l358_35836


namespace NUMINAMATH_CALUDE_ABCD_requires_16_bits_l358_35800

/-- Represents a base-16 digit --/
def Hex : Type := Fin 16

/-- Represents a base-16 number with 4 digits --/
def HexNumber := Fin 4 → Hex

/-- Converts a HexNumber to its decimal (base-10) representation --/
def toDecimal (h : HexNumber) : ℕ :=
  (h 0).val * 16^3 + (h 1).val * 16^2 + (h 2).val * 16^1 + (h 3).val * 16^0

/-- The specific HexNumber ABCD --/
def ABCD : HexNumber :=
  fun i => match i with
    | 0 => ⟨10, by norm_num⟩
    | 1 => ⟨11, by norm_num⟩
    | 2 => ⟨12, by norm_num⟩
    | 3 => ⟨13, by norm_num⟩

/-- Number of bits required to represent a natural number --/
def bitsRequired (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem ABCD_requires_16_bits :
  bitsRequired (toDecimal ABCD) = 16 :=
sorry

end NUMINAMATH_CALUDE_ABCD_requires_16_bits_l358_35800


namespace NUMINAMATH_CALUDE_tom_shares_problem_l358_35824

theorem tom_shares_problem (initial_cost : ℕ) (sold_shares : ℕ) (sold_price : ℕ) (total_profit : ℕ) :
  initial_cost = 3 →
  sold_shares = 10 →
  sold_price = 4 →
  total_profit = 40 →
  ∃ (initial_shares : ℕ), 
    initial_shares = sold_shares ∧
    sold_shares * (sold_price - initial_cost) = total_profit :=
by sorry

end NUMINAMATH_CALUDE_tom_shares_problem_l358_35824


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l358_35838

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (180 - 360 / n : ℝ) = 160 → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l358_35838


namespace NUMINAMATH_CALUDE_tablecloth_extension_theorem_l358_35821

/-- Represents a circular table with a square tablecloth placed on it. -/
structure TableWithCloth where
  /-- Diameter of the circular table in meters -/
  table_diameter : ℝ
  /-- Side length of the square tablecloth in meters -/
  cloth_side_length : ℝ
  /-- Extension of one corner beyond the table edge in meters -/
  corner1_extension : ℝ
  /-- Extension of an adjacent corner beyond the table edge in meters -/
  corner2_extension : ℝ

/-- Calculates the extensions of the remaining two corners of the tablecloth. -/
def calculate_remaining_extensions (t : TableWithCloth) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct extensions for the given table and tablecloth configuration. -/
theorem tablecloth_extension_theorem (t : TableWithCloth) 
  (h1 : t.table_diameter = 0.6)
  (h2 : t.cloth_side_length = 1)
  (h3 : t.corner1_extension = 0.5)
  (h4 : t.corner2_extension = 0.3) :
  calculate_remaining_extensions t = (0.33, 0.52) :=
by sorry

end NUMINAMATH_CALUDE_tablecloth_extension_theorem_l358_35821


namespace NUMINAMATH_CALUDE_subset_sum_property_l358_35884

theorem subset_sum_property (n : ℕ) (hn : n > 1) :
  ∀ (S : Finset ℕ), S ⊆ Finset.range (2 * n) → S.card = n + 2 →
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b = c :=
by sorry

end NUMINAMATH_CALUDE_subset_sum_property_l358_35884


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_36_l358_35879

def is_divisible_by_36 (n : ℕ) : Prop := n % 36 = 0

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def form_number (a b : ℕ) : ℕ := 90000 + 1000 * a + 650 + b

theorem five_digit_divisible_by_36 :
  ∀ a b : ℕ,
    is_single_digit a →
    is_single_digit b →
    is_divisible_by_36 (form_number a b) →
    ((a = 5 ∧ b = 2) ∨ (a = 1 ∧ b = 6)) :=
sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_36_l358_35879


namespace NUMINAMATH_CALUDE_total_rainfall_l358_35801

def rainfall_problem (sunday monday tuesday : ℕ) : Prop :=
  (tuesday = 2 * monday) ∧
  (monday = sunday + 3) ∧
  (sunday = 4)

theorem total_rainfall : 
  ∀ sunday monday tuesday : ℕ, 
  rainfall_problem sunday monday tuesday → 
  sunday + monday + tuesday = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_l358_35801


namespace NUMINAMATH_CALUDE_expression_evaluation_l358_35812

theorem expression_evaluation (b : ℕ) (h : b = 2) : (b^3 * b^4) - b^2 = 124 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l358_35812


namespace NUMINAMATH_CALUDE_painted_cells_count_l358_35809

-- Define the structure of the grid
structure Grid :=
  (k : ℕ)
  (l : ℕ)

-- Define the properties of the grid
def valid_grid (g : Grid) : Prop :=
  g.k = 2 ∧ g.l = 37

-- Define the number of white cells
def white_cells (g : Grid) : ℕ :=
  g.k * g.l

-- Define the total number of cells
def total_cells (g : Grid) : ℕ :=
  (2 * g.k + 1) * (2 * g.l + 1)

-- Define the number of painted cells
def painted_cells (g : Grid) : ℕ :=
  total_cells g - white_cells g

-- The main theorem
theorem painted_cells_count (g : Grid) :
  valid_grid g → white_cells g = 74 → painted_cells g = 301 :=
by
  sorry


end NUMINAMATH_CALUDE_painted_cells_count_l358_35809


namespace NUMINAMATH_CALUDE_jane_circle_impossibility_l358_35803

theorem jane_circle_impossibility : ¬ ∃ (a : Fin 2024 → ℕ+),
  (∀ i : Fin 2024, ∃ j : Fin 2024, a i * a (i + 1) = Nat.factorial (j + 1)) ∧
  (∀ k : Fin 2024, ∃ i : Fin 2024, a i * a (i + 1) = Nat.factorial (k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_jane_circle_impossibility_l358_35803


namespace NUMINAMATH_CALUDE_limit_P_div_B_l358_35866

/-- The number of ways to make n cents using quarters, dimes, nickels, and pennies -/
def P (n : ℕ) : ℕ := sorry

/-- The number of ways to make n cents using dollar bills, quarters, dimes, and nickels -/
def B (n : ℕ) : ℕ := sorry

/-- The value of a penny in cents -/
def penny : ℕ := 1

/-- The value of a nickel in cents -/
def nickel : ℕ := 5

/-- The value of a dime in cents -/
def dime : ℕ := 10

/-- The value of a quarter in cents -/
def quarter : ℕ := 25

/-- The value of a dollar bill in cents -/
def dollar : ℕ := 100

/-- The limit of P_n / B_n as n approaches infinity -/
theorem limit_P_div_B :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((P n : ℝ) / (B n : ℝ)) - (1 / 20)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_P_div_B_l358_35866


namespace NUMINAMATH_CALUDE_solution_set_f_gt_5_range_of_a_l358_35873

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- Theorem for the solution set of f(x) > 5
theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -3 ∨ x > 2} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 - 2*a) → -1 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_5_range_of_a_l358_35873
