import Mathlib

namespace total_salaries_l1868_186829

/-- The total amount of A and B's salaries given the specified conditions -/
theorem total_salaries (A_salary B_salary : ℝ) : 
  A_salary = 1500 →
  A_salary * 0.05 = B_salary * 0.15 →
  A_salary + B_salary = 2000 := by
  sorry

end total_salaries_l1868_186829


namespace largest_three_digit_in_pascal_l1868_186833

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Pascal's triangle entry at row n and position k -/
def pascal (n k : ℕ) : ℕ := binomial n k

/-- The largest three-digit number -/
def largest_three_digit : ℕ := 999

/-- The row where the largest three-digit number first appears -/
def first_appearance_row : ℕ := 1000

/-- The position in the row where the largest three-digit number first appears -/
def first_appearance_pos : ℕ := 500

theorem largest_three_digit_in_pascal :
  (∀ n k, n < first_appearance_row → pascal n k ≤ largest_three_digit) ∧
  pascal first_appearance_row first_appearance_pos = largest_three_digit ∧
  (∀ n k, n > first_appearance_row → pascal n k > largest_three_digit) :=
sorry

end largest_three_digit_in_pascal_l1868_186833


namespace complex_fraction_value_l1868_186896

theorem complex_fraction_value : 
  let i : ℂ := Complex.I
  (1 + Real.sqrt 3 * i)^2 / (Real.sqrt 3 * i - 1) = 2 := by sorry

end complex_fraction_value_l1868_186896


namespace exists_k_for_1001_free_ends_l1868_186840

/-- Represents the process of drawing segments as described in the problem -/
structure SegmentDrawing where
  initial_segment : Unit  -- Represents the initial segment OA
  branch_factor : Nat     -- Number of segments drawn from each point (5 in this case)
  free_ends : Nat         -- Number of free ends

/-- Calculates the number of free ends after k iterations of drawing segments -/
def free_ends_after_iterations (k : Nat) : Nat :=
  1 + 4 * k

/-- Theorem stating that it's possible to have exactly 1001 free ends -/
theorem exists_k_for_1001_free_ends :
  ∃ k : Nat, free_ends_after_iterations k = 1001 := by
  sorry

#check exists_k_for_1001_free_ends

end exists_k_for_1001_free_ends_l1868_186840


namespace intersection_point_l1868_186809

theorem intersection_point (x y : ℚ) : 
  (8 * x - 5 * y = 10) ∧ (6 * x + 2 * y = 22) ↔ (x = 65/23 ∧ y = -137/23) :=
by sorry

end intersection_point_l1868_186809


namespace kim_cousins_count_l1868_186835

theorem kim_cousins_count (gum_per_cousin : ℕ) (total_gum : ℕ) (h1 : gum_per_cousin = 5) (h2 : total_gum = 20) :
  total_gum / gum_per_cousin = 4 := by
sorry

end kim_cousins_count_l1868_186835


namespace child_b_share_l1868_186811

theorem child_b_share (total_money : ℝ) (ratios : Fin 5 → ℝ) : 
  total_money = 10800 ∧ 
  ratios 0 = 0.5 ∧ 
  ratios 1 = 1.5 ∧ 
  ratios 2 = 2.25 ∧ 
  ratios 3 = 3.5 ∧ 
  ratios 4 = 4.25 → 
  (ratios 1 * total_money) / (ratios 0 + ratios 1 + ratios 2 + ratios 3 + ratios 4) = 1350 := by
sorry

end child_b_share_l1868_186811


namespace divide_100_by_0_25_l1868_186866

theorem divide_100_by_0_25 : (100 : ℝ) / 0.25 = 400 := by
  sorry

end divide_100_by_0_25_l1868_186866


namespace extremum_implies_slope_l1868_186876

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := (x - 2) * (x^2 + c)

-- Define the derivative of f(x)
def f' (c : ℝ) (x : ℝ) : ℝ := (x^2 + c) + (x - 2) * (2 * x)

theorem extremum_implies_slope (c : ℝ) :
  (∃ k, f' c 1 = k ∧ k = 0) → f' c (-1) = 8 := by
  sorry

end extremum_implies_slope_l1868_186876


namespace necessary_but_not_sufficient_condition_l1868_186894

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > 2 → x > 1) ∧ ¬(x > 1 → x > 2) := by
  sorry

end necessary_but_not_sufficient_condition_l1868_186894


namespace satisfactory_grade_fraction_l1868_186815

-- Define the grade categories
inductive Grade
| A
| B
| C
| D
| F

-- Define a function to check if a grade is satisfactory
def isSatisfactory (g : Grade) : Bool :=
  match g with
  | Grade.A => true
  | Grade.B => true
  | Grade.C => true
  | _ => false

-- Define the distribution of grades
def gradeDistribution : List (Grade × Nat) :=
  [(Grade.A, 6), (Grade.B, 5), (Grade.C, 7), (Grade.D, 4), (Grade.F, 6)]

-- Theorem to prove
theorem satisfactory_grade_fraction :
  let totalStudents := (gradeDistribution.map (·.2)).sum
  let satisfactoryStudents := (gradeDistribution.filter (isSatisfactory ·.1)).map (·.2) |>.sum
  (satisfactoryStudents : Rat) / totalStudents = 9 / 14 := by
  sorry


end satisfactory_grade_fraction_l1868_186815


namespace divisibility_by_hundred_l1868_186885

theorem divisibility_by_hundred (N : ℕ) : 
  N = 2^5 * 3^2 * 7 * 75 → 100 ∣ N := by
  sorry

end divisibility_by_hundred_l1868_186885


namespace player_positions_satisfy_distances_l1868_186803

/-- Represents the positions of four soccer players on a number line -/
def PlayerPositions : Fin 4 → ℝ
  | 0 => 0
  | 1 => 1
  | 2 => 4
  | 3 => 6

/-- Calculates the distance between two players -/
def distance (i j : Fin 4) : ℝ :=
  |PlayerPositions i - PlayerPositions j|

/-- The set of required pairwise distances -/
def RequiredDistances : Set ℝ := {1, 2, 3, 4, 5, 6}

/-- Theorem stating that the player positions satisfy the required distances -/
theorem player_positions_satisfy_distances :
  ∀ i j : Fin 4, i ≠ j → distance i j ∈ RequiredDistances :=
sorry

end player_positions_satisfy_distances_l1868_186803


namespace recurrence_solution_l1868_186802

-- Define the recurrence relation
def a : ℕ → ℤ
  | 0 => 3
  | n + 1 => 2 * a n + 2^(n + 1)

-- State the theorem
theorem recurrence_solution (n : ℕ) : a n = (n + 3) * 2^n := by
  sorry

end recurrence_solution_l1868_186802


namespace total_time_is_14_25_years_l1868_186868

def time_to_get_in_shape : ℕ := 2 * 12  -- 2 years in months
def time_to_learn_climbing : ℕ := 2 * time_to_get_in_shape
def time_for_survival_skills : ℕ := 9
def time_for_photography : ℕ := 3
def downtime : ℕ := 1
def time_for_summits : List ℕ := [4, 5, 6, 8, 7, 9, 10]
def time_to_learn_diving : ℕ := 13
def time_for_cave_diving : ℕ := 2 * 12  -- 2 years in months

theorem total_time_is_14_25_years :
  let total_months : ℕ := time_to_get_in_shape + time_to_learn_climbing +
                          time_for_survival_skills + time_for_photography +
                          downtime + (time_for_summits.sum) +
                          time_to_learn_diving + time_for_cave_diving
  (total_months : ℚ) / 12 = 14.25 := by
  sorry

end total_time_is_14_25_years_l1868_186868


namespace a_plus_b_equals_seven_l1868_186822

theorem a_plus_b_equals_seven (a b : ℝ) (h : ∀ x, a * (x + b) = 3 * x + 12) : a + b = 7 := by
  sorry

end a_plus_b_equals_seven_l1868_186822


namespace triangle_side_range_l1868_186893

/-- Given a triangle ABC where c = √2 and a cos C = c sin A, 
    the length of side BC is in the range (√2, 2) -/
theorem triangle_side_range (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 2 →
  a * Real.cos C = c * Real.sin A →
  ∃ (BC : ℝ), BC > Real.sqrt 2 ∧ BC < 2 :=
by sorry

end triangle_side_range_l1868_186893


namespace pure_imaginary_complex_number_l1868_186878

theorem pure_imaginary_complex_number (a : ℝ) :
  (Complex.I * (a - 2) : ℂ).re = 0 → a = 1 := by
  sorry

end pure_imaginary_complex_number_l1868_186878


namespace expected_girls_left_10_7_l1868_186867

/-- The expected number of girls standing to the left of all boys in a random arrangement -/
def expected_girls_left (num_boys num_girls : ℕ) : ℚ :=
  num_girls / (num_boys + 1)

/-- Theorem: In a random arrangement of 10 boys and 7 girls, 
    the expected number of girls standing to the left of all boys is 7/11 -/
theorem expected_girls_left_10_7 : 
  expected_girls_left 10 7 = 7 / 11 := by
  sorry

end expected_girls_left_10_7_l1868_186867


namespace purple_pairs_coincide_l1868_186849

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  yellow : ℕ
  green : ℕ
  purple : ℕ

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  yellow_yellow : ℕ
  green_green : ℕ
  yellow_purple : ℕ
  purple_purple : ℕ

/-- The main theorem to prove -/
theorem purple_pairs_coincide 
  (counts : TriangleCounts)
  (pairs : CoincidingPairs)
  (h1 : counts.yellow = 4)
  (h2 : counts.green = 6)
  (h3 : counts.purple = 10)
  (h4 : pairs.yellow_yellow = 3)
  (h5 : pairs.green_green = 4)
  (h6 : pairs.yellow_purple = 3) :
  pairs.purple_purple = 5 := by
  sorry

end purple_pairs_coincide_l1868_186849


namespace joan_seashells_l1868_186886

/-- Given 245 initial seashells, prove that after giving 3/5 to Mike and 2/5 of the remainder to Lisa, Joan is left with 59 seashells. -/
theorem joan_seashells (initial_seashells : ℕ) (mike_fraction : ℚ) (lisa_fraction : ℚ) :
  initial_seashells = 245 →
  mike_fraction = 3 / 5 →
  lisa_fraction = 2 / 5 →
  initial_seashells - (initial_seashells * mike_fraction).floor -
    ((initial_seashells - (initial_seashells * mike_fraction).floor) * lisa_fraction).floor = 59 := by
  sorry


end joan_seashells_l1868_186886


namespace bridge_length_l1868_186818

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 235 := by
  sorry

end bridge_length_l1868_186818


namespace max_m_inequality_l1868_186812

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (m : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y ≥ m/(2*x + y)) ∧
  (∀ (n : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y ≥ n/(2*x + y)) → n ≤ 9) :=
by sorry

end max_m_inequality_l1868_186812


namespace three_primes_in_list_l1868_186864

def number_list : List Nat := [11, 12, 13, 14, 15, 16, 17]

theorem three_primes_in_list :
  (number_list.filter Nat.Prime).length = 3 := by
  sorry

end three_primes_in_list_l1868_186864


namespace soda_preference_l1868_186816

/-- Given a survey of 520 people and a central angle of 144° for "Soda" preference
    in a pie chart, prove that 208 people favor "Soda". -/
theorem soda_preference (total : ℕ) (angle : ℝ) (h1 : total = 520) (h2 : angle = 144) :
  (angle / 360 : ℝ) * total = 208 := by
  sorry

end soda_preference_l1868_186816


namespace union_of_M_and_N_l1868_186853

def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by
  sorry

end union_of_M_and_N_l1868_186853


namespace arctan_sum_three_four_l1868_186892

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by
  sorry

end arctan_sum_three_four_l1868_186892


namespace sum_of_smaller_angles_is_180_l1868_186842

/-- A convex pentagon with all diagonals drawn. -/
structure ConvexPentagonWithDiagonals where
  -- We don't need to define the specific properties here, just the structure

/-- The sum of the smaller angles formed by intersecting diagonals in a convex pentagon. -/
def sumOfSmallerAngles (p : ConvexPentagonWithDiagonals) : ℝ := sorry

/-- Theorem: The sum of the smaller angles formed by intersecting diagonals in a convex pentagon is always 180°. -/
theorem sum_of_smaller_angles_is_180 (p : ConvexPentagonWithDiagonals) :
  sumOfSmallerAngles p = 180 := by sorry

end sum_of_smaller_angles_is_180_l1868_186842


namespace f_greater_than_f_prime_plus_three_halves_l1868_186828

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - log x) + (2 * x - 1) / x^2

theorem f_greater_than_f_prime_plus_three_halves (x : ℝ) (hx : x ∈ Set.Icc 1 2) :
  f 1 x > (deriv (f 1)) x + 3/2 := by
  sorry

end f_greater_than_f_prime_plus_three_halves_l1868_186828


namespace cubic_polynomial_value_at_5_l1868_186873

/-- A cubic polynomial satisfying specific conditions -/
def cubicPolynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, p x = a * x^3 + b * x^2 + c * x + d) ∧
  (p 1 = 1) ∧ (p 2 = 1/8) ∧ (p 3 = 1/27) ∧ (p 4 = 1/64)

/-- Theorem stating that a cubic polynomial satisfying the given conditions has p(5) = 0 -/
theorem cubic_polynomial_value_at_5 (p : ℝ → ℝ) (h : cubicPolynomial p) : p 5 = 0 := by
  sorry

end cubic_polynomial_value_at_5_l1868_186873


namespace michelle_gas_problem_l1868_186819

/-- Michelle's gas problem -/
theorem michelle_gas_problem (gas_left gas_used : ℚ) 
  (h1 : gas_left = 0.17)
  (h2 : gas_used = 0.33) : 
  gas_left + gas_used = 0.50 := by
  sorry

end michelle_gas_problem_l1868_186819


namespace orange_sales_l1868_186801

theorem orange_sales (alice_oranges emily_oranges total_oranges : ℕ) : 
  alice_oranges = 120 →
  alice_oranges = 2 * emily_oranges →
  total_oranges = alice_oranges + emily_oranges →
  total_oranges = 180 := by
  sorry

end orange_sales_l1868_186801


namespace percy_swimming_hours_l1868_186807

/-- Percy's swimming schedule and total hours over 4 weeks -/
theorem percy_swimming_hours :
  let weekday_hours : ℕ := 2 -- 1 hour before school + 1 hour after school
  let weekdays_per_week : ℕ := 5
  let weekend_hours : ℕ := 3
  let weekend_days : ℕ := 2
  let weeks : ℕ := 4
  
  let total_hours_per_week : ℕ := weekday_hours * weekdays_per_week + weekend_hours * weekend_days
  let total_hours_four_weeks : ℕ := total_hours_per_week * weeks
  
  total_hours_four_weeks = 64
  := by sorry

end percy_swimming_hours_l1868_186807


namespace unique_divisible_by_twelve_l1868_186895

/-- A function that constructs a four-digit number in the form x27x -/
def constructNumber (x : Nat) : Nat :=
  1000 * x + 270 + x

/-- Predicate to check if a number is a single digit -/
def isSingleDigit (n : Nat) : Prop :=
  n ≥ 0 ∧ n ≤ 9

theorem unique_divisible_by_twelve :
  ∃! x : Nat, isSingleDigit x ∧ (constructNumber x) % 12 = 0 ∧ x = 6 := by
  sorry

end unique_divisible_by_twelve_l1868_186895


namespace constraint_implies_sum_equals_nine_l1868_186884

open Real

/-- The maximum value of xy + xz + yz given the constraint -/
noncomputable def N : ℝ := sorry

/-- The minimum value of xy + xz + yz given the constraint -/
noncomputable def n : ℝ := sorry

/-- Theorem stating that N + 8n = 9 given the constraint -/
theorem constraint_implies_sum_equals_nine :
  ∀ x y z : ℝ, 3 * (x + y + z) = x^2 + y^2 + z^2 → N + 8 * n = 9 := by
  sorry

end constraint_implies_sum_equals_nine_l1868_186884


namespace grid_ball_probability_l1868_186880

theorem grid_ball_probability
  (a b c r : ℝ)
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_a_gt_b : a > b)
  (h_r_lt_b_half : r < b / 2)
  (h_strip_width : 2 * c = Real.sqrt ((a + b)^2 / 4 + a * b) - (a + b) / 2)
  : (a - 2 * r) * (b - 2 * r) / ((a + 2 * c) * (b + 2 * c)) ≤ 1 / 2 :=
by sorry

end grid_ball_probability_l1868_186880


namespace min_value_trigonometric_expression_l1868_186899

theorem min_value_trigonometric_expression (γ δ : ℝ) :
  (3 * Real.cos γ + 4 * Real.sin δ - 7)^2 + (3 * Real.sin γ + 4 * Real.cos δ - 12)^2 ≥ 81 ∧
  ∃ (γ₀ δ₀ : ℝ), (3 * Real.cos γ₀ + 4 * Real.sin δ₀ - 7)^2 + (3 * Real.sin γ₀ + 4 * Real.cos δ₀ - 12)^2 = 81 :=
by sorry

end min_value_trigonometric_expression_l1868_186899


namespace fence_construction_l1868_186841

/-- A fence construction problem -/
theorem fence_construction (panels : ℕ) (sheets_per_panel : ℕ) (beams_per_panel : ℕ) 
  (rods_per_sheet : ℕ) (total_rods : ℕ) :
  panels = 10 →
  sheets_per_panel = 3 →
  beams_per_panel = 2 →
  rods_per_sheet = 10 →
  total_rods = 380 →
  (total_rods - panels * sheets_per_panel * rods_per_sheet) / (panels * beams_per_panel) = 4 :=
by sorry

end fence_construction_l1868_186841


namespace distribute_problems_l1868_186891

theorem distribute_problems (n m : ℕ) (hn : n = 7) (hm : m = 15) :
  (Nat.choose m n) * (Nat.factorial n) = 32432400 := by
  sorry

end distribute_problems_l1868_186891


namespace vector_magnitude_range_l1868_186857

theorem vector_magnitude_range (a b : EuclideanSpace ℝ (Fin 3)) :
  (norm b = 2) → (norm a = 2 * norm (b - a)) → (4/3 : ℝ) ≤ norm a ∧ norm a ≤ 4 := by
  sorry

end vector_magnitude_range_l1868_186857


namespace ceiling_sqrt_169_l1868_186888

theorem ceiling_sqrt_169 : ⌈Real.sqrt 169⌉ = 13 := by
  sorry

end ceiling_sqrt_169_l1868_186888


namespace solution_exists_l1868_186821

theorem solution_exists (x : ℝ) (h : x = 5) : ∃ some_number : ℝ, (x / 5) + some_number = 4 := by
  sorry

end solution_exists_l1868_186821


namespace sum_powers_i_2047_l1868_186887

def imaginary_unit_sum (i : ℂ) : ℕ → ℂ
  | 0 => 1
  | n + 1 => i^(n + 1) + imaginary_unit_sum i n

theorem sum_powers_i_2047 (i : ℂ) (h : i^2 = -1) :
  imaginary_unit_sum i 2047 = 0 := by
  sorry

end sum_powers_i_2047_l1868_186887


namespace bakers_ovens_l1868_186861

/-- Baker's bread production problem -/
theorem bakers_ovens :
  let loaves_per_hour_per_oven : ℕ := 5
  let weekday_hours : ℕ := 5
  let weekday_count : ℕ := 5
  let weekend_hours : ℕ := 2
  let weekend_count : ℕ := 2
  let weeks : ℕ := 3
  let total_loaves : ℕ := 1740
  
  let weekly_hours := weekday_hours * weekday_count + weekend_hours * weekend_count
  let weekly_loaves_per_oven := weekly_hours * loaves_per_hour_per_oven
  let total_loaves_per_oven := weekly_loaves_per_oven * weeks
  
  total_loaves / total_loaves_per_oven = 4 := by
  sorry


end bakers_ovens_l1868_186861


namespace new_person_weight_l1868_186875

/-- Given a group of 8 people, when one person weighing 20 kg is replaced by a new person,
    and the average weight increases by 2.5 kg, the weight of the new person is 40 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_removed : Real) (avg_increase : Real) :
  initial_count = 8 →
  weight_removed = 20 →
  avg_increase = 2.5 →
  (initial_count : Real) * avg_increase + weight_removed = 40 :=
by sorry

end new_person_weight_l1868_186875


namespace sqrt_square_eq_abs_sqrt_three_squared_l1868_186834

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by sorry

theorem sqrt_three_squared : Real.sqrt (3 ^ 2) = 3 := by sorry

end sqrt_square_eq_abs_sqrt_three_squared_l1868_186834


namespace quadratic_equations_solutions_l1868_186881

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ x = 2 - Real.sqrt 5 ∨ x = 2 + Real.sqrt 5) ∧
  (∀ x : ℝ, 3*x^2 - 5*x + 1 = 0 ↔ x = (5 - Real.sqrt 13) / 6 ∨ x = (5 + Real.sqrt 13) / 6) := by
  sorry

end quadratic_equations_solutions_l1868_186881


namespace right_triangle_parity_l1868_186851

theorem right_triangle_parity (a b c : ℕ) (h_right : a^2 + b^2 = c^2) :
  (Even a ∧ Even b ∧ Even c) ∨
  ((Odd a ∧ Even b ∧ Odd c) ∨ (Even a ∧ Odd b ∧ Odd c)) :=
sorry

end right_triangle_parity_l1868_186851


namespace area_bounded_region_area_is_four_l1868_186800

/-- The area of the region bounded by x = 2, y = 2, x = 0, and y = 0 is 4 -/
theorem area_bounded_region : ℝ :=
  let x_bound : ℝ := 2
  let y_bound : ℝ := 2
  x_bound * y_bound

#check area_bounded_region

theorem area_is_four : area_bounded_region = 4 := by sorry

end area_bounded_region_area_is_four_l1868_186800


namespace equation_solution_l1868_186813

theorem equation_solution (a b x : ℤ) : 
  (a * x^2 + b * x + 14)^2 + (b * x^2 + a * x + 8)^2 = 0 →
  a = -6 ∧ b = -5 ∧ x = -2 := by
sorry

end equation_solution_l1868_186813


namespace slope_MN_constant_l1868_186848

/-- Definition of curve C -/
def curve_C (x y : ℝ) : Prop := y^2 = 4*x + 4 ∧ x ≥ 0

/-- Definition of point D on curve C -/
def point_D : ℝ × ℝ := (0, 2)

/-- Definition of complementary slopes -/
def complementary_slopes (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -1

/-- Theorem: The slope of line MN is constant and equal to -1 -/
theorem slope_MN_constant (k : ℝ) (M N : ℝ × ℝ) :
  curve_C M.1 M.2 →
  curve_C N.1 N.2 →
  curve_C point_D.1 point_D.2 →
  complementary_slopes k (-k) →
  (M.2 - point_D.2) = k * (M.1 - point_D.1) →
  (N.2 - point_D.2) = (-k) * (N.1 - point_D.1) →
  M ≠ point_D →
  N ≠ point_D →
  (N.2 - M.2) / (N.1 - M.1) = -1 := by
  sorry

end slope_MN_constant_l1868_186848


namespace sum_of_powers_l1868_186823

theorem sum_of_powers (a b : ℝ) : 
  a + b = 1 →
  a^2 + b^2 = 3 →
  a^3 + b^3 = 4 →
  a^4 + b^4 = 7 →
  a^5 + b^5 = 11 →
  a^10 + b^10 = 123 := by
  sorry

end sum_of_powers_l1868_186823


namespace sequence_sum_l1868_186865

theorem sequence_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
sorry

end sequence_sum_l1868_186865


namespace triangle_property_l1868_186897

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.a / t.b * Real.cos t.C + t.c / (2 * t.b) = 1) 
  (h2 : t.A + t.B + t.C = π) 
  (h3 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0) 
  (h4 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0) :
  (t.A = π / 3) ∧ 
  (t.a = 1 → ∃ l : Real, l > 2 ∧ l ≤ 3 ∧ l = t.a + t.b + t.c) := by
  sorry


end triangle_property_l1868_186897


namespace female_elementary_students_l1868_186870

theorem female_elementary_students (total_students : ℕ) (non_elementary_girls : ℕ) 
  (h1 : total_students = 30)
  (h2 : non_elementary_girls = 7) :
  total_students / 2 - non_elementary_girls = 8 := by
  sorry

end female_elementary_students_l1868_186870


namespace knicks_win_probability_l1868_186855

/-- The probability of winning a single game for the Heat -/
def p : ℚ := 3/5

/-- The probability of winning a single game for the Knicks -/
def q : ℚ := 1 - p

/-- The number of games needed to win the tournament -/
def games_to_win : ℕ := 3

/-- The total number of games in the tournament -/
def total_games : ℕ := 5

/-- The probability of the Knicks winning the tournament in exactly 5 games -/
def knicks_win_prob : ℚ :=
  (Nat.choose 4 2 : ℚ) * q^2 * p^2 * q

theorem knicks_win_probability :
  knicks_win_prob = 432/3125 :=
sorry

end knicks_win_probability_l1868_186855


namespace max_operation_value_l1868_186869

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def operation (n : ℕ) : ℕ := 3 * (300 - n)

theorem max_operation_value :
  ∃ (m : ℕ), (∀ (n : ℕ), is_two_digit n → operation n ≤ m) ∧ (∃ (k : ℕ), is_two_digit k ∧ operation k = m) ∧ m = 870 :=
sorry

end max_operation_value_l1868_186869


namespace modulus_of_complex_number_l1868_186898

theorem modulus_of_complex_number (z : ℂ) : z = 2 / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_complex_number_l1868_186898


namespace equation_solution_l1868_186804

theorem equation_solution (x : ℚ) : 
  (x + 10) / (x - 4) = (x - 4) / (x + 8) → x = -32 / 13 := by
sorry

end equation_solution_l1868_186804


namespace amp_composition_l1868_186808

-- Define the operations
def amp (x : ℤ) : ℤ := 10 - x
def amp_prefix (x : ℤ) : ℤ := x - 10

-- State the theorem
theorem amp_composition : amp_prefix (amp 15) = -15 := by
  sorry

end amp_composition_l1868_186808


namespace books_movies_difference_l1868_186856

def books_read : ℕ := 17
def movies_watched : ℕ := 21

theorem books_movies_difference :
  (books_read : ℤ) - movies_watched = -4 :=
sorry

end books_movies_difference_l1868_186856


namespace hadley_walk_distance_l1868_186820

/-- The distance Hadley walked to the pet store -/
def distance_to_pet_store : ℝ := 1

/-- The distance Hadley walked to the grocery store -/
def distance_to_grocery : ℝ := 2

/-- The distance Hadley walked back home -/
def distance_back_home : ℝ := 4 - 1

/-- The total distance Hadley walked -/
def total_distance : ℝ := 6

theorem hadley_walk_distance :
  distance_to_grocery + distance_to_pet_store + distance_back_home = total_distance :=
by sorry

end hadley_walk_distance_l1868_186820


namespace marion_score_l1868_186845

theorem marion_score (total_items : Nat) (ella_incorrect : Nat) (marion_additional : Nat) :
  total_items = 40 →
  ella_incorrect = 4 →
  marion_additional = 6 →
  (total_items - ella_incorrect) / 2 + marion_additional = 24 := by
  sorry

end marion_score_l1868_186845


namespace inheritance_calculation_l1868_186874

theorem inheritance_calculation (inheritance : ℝ) : 
  inheritance * 0.25 + (inheritance - inheritance * 0.25) * 0.15 = 15000 → 
  inheritance = 41379 := by
sorry

end inheritance_calculation_l1868_186874


namespace square_of_difference_l1868_186883

theorem square_of_difference (x : ℝ) : (x - 1)^2 = x^2 + 1 - 2*x := by
  sorry

end square_of_difference_l1868_186883


namespace non_congruent_triangles_count_l1868_186858

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A 2x4 array of points -/
def PointArray : Array (Array Point) :=
  #[#[{x := 0, y := 0}, {x := 1, y := 0}, {x := 2, y := 0}, {x := 3, y := 0}],
    #[{x := 0, y := 1}, {x := 1, y := 1}, {x := 2, y := 1}, {x := 3, y := 1}]]

/-- Check if two triangles are congruent -/
def are_congruent (t1 t2 : Array Point) : Prop := sorry

/-- Count non-congruent triangles in the point array -/
def count_non_congruent_triangles (arr : Array (Array Point)) : ℕ := sorry

/-- Theorem: The number of non-congruent triangles in the given 2x4 array is 3 -/
theorem non_congruent_triangles_count :
  count_non_congruent_triangles PointArray = 3 := by sorry

end non_congruent_triangles_count_l1868_186858


namespace product_of_smallest_primes_l1868_186879

def smallest_two_digit_prime_1 : Nat := 11
def smallest_two_digit_prime_2 : Nat := 13
def smallest_three_digit_prime : Nat := 101

theorem product_of_smallest_primes :
  smallest_two_digit_prime_1 * smallest_two_digit_prime_2 * smallest_three_digit_prime = 14443 := by
  sorry

end product_of_smallest_primes_l1868_186879


namespace gcd_72_120_l1868_186827

theorem gcd_72_120 : Nat.gcd 72 120 = 24 := by
  sorry

end gcd_72_120_l1868_186827


namespace ellipse_focal_chord_properties_l1868_186846

/-- An ellipse with eccentricity e and a line segment PQ passing through its left focus -/
structure EllipseWithFocalChord where
  e : ℝ
  b : ℝ
  hb : b > 0
  pq_not_vertical : True  -- Represents that PQ is not perpendicular to x-axis
  equilateral_exists : True  -- Represents that there exists R making PQR equilateral

/-- The range of eccentricity and slope of PQ for an ellipse with a special focal chord -/
theorem ellipse_focal_chord_properties (E : EllipseWithFocalChord) :
  E.e > Real.sqrt 3 / 3 ∧ E.e < 1 ∧
  ∃ (k : ℝ), (k = 1 / Real.sqrt (3 * E.e^2 - 1) ∨ k = -1 / Real.sqrt (3 * E.e^2 - 1)) :=
by sorry

end ellipse_focal_chord_properties_l1868_186846


namespace selling_price_with_gain_l1868_186826

/-- Given an article with a cost price where a $10 gain represents a 10% gain, 
    prove that the selling price is $110. -/
theorem selling_price_with_gain (cost_price : ℝ) 
  (h1 : cost_price > 0)
  (h2 : 10 / cost_price = 0.1) : 
  cost_price + 10 = 110 := by
  sorry

#check selling_price_with_gain

end selling_price_with_gain_l1868_186826


namespace expression_evaluation_l1868_186843

theorem expression_evaluation :
  let x : ℝ := (Real.pi - 3) ^ 0
  let y : ℝ := (-1/3)⁻¹
  ((2*x - y)^2 - (y + 2*x) * (y - 2*x)) / (-1/2 * x) = -40 :=
by sorry

end expression_evaluation_l1868_186843


namespace inner_circle_radius_l1868_186838

theorem inner_circle_radius (r : ℝ) : 
  r > 0 →
  (π * ((10 : ℝ)^2 - (0.5 * r)^2) = 3.25 * π * (8^2 - r^2)) →
  r = 6 := by
sorry

end inner_circle_radius_l1868_186838


namespace correct_seat_ratio_l1868_186850

/-- The ratio of coach class seats to first-class seats in an airplane -/
def seat_ratio (total_seats first_class_seats : ℕ) : ℚ × ℚ :=
  let coach_seats := total_seats - first_class_seats
  (coach_seats, first_class_seats)

/-- Theorem stating the correct ratio of coach to first-class seats -/
theorem correct_seat_ratio :
  seat_ratio 387 77 = (310, 77) := by
  sorry

#eval seat_ratio 387 77

end correct_seat_ratio_l1868_186850


namespace price_increase_theorem_l1868_186825

theorem price_increase_theorem (original_price : ℝ) (original_price_pos : original_price > 0) :
  let price_a := original_price * 1.2 * 1.15
  let price_b := original_price * 1.3 * 0.9
  let price_c := original_price * 1.25 * 1.2
  let total_increase := (price_a + price_b + price_c) - 3 * original_price
  let percent_increase := total_increase / (3 * original_price) * 100
  percent_increase = 35 := by
sorry

end price_increase_theorem_l1868_186825


namespace integral_2sqrt_minus_sin_l1868_186805

open MeasureTheory Interval Real

theorem integral_2sqrt_minus_sin : ∫ x in (-1)..1, (2 * Real.sqrt (1 - x^2) - Real.sin x) = π := by
  sorry

end integral_2sqrt_minus_sin_l1868_186805


namespace right_triangle_angles_l1868_186854

theorem right_triangle_angles (A B C : Real) (h1 : A + B + C = 180) (h2 : C = 90) (h3 : A = 50) : B = 40 := by
  sorry

end right_triangle_angles_l1868_186854


namespace crazy_silly_school_books_l1868_186832

/-- The number of different books in the 'crazy silly school' series -/
def num_books : ℕ := 16

/-- The number of movies watched -/
def movies_watched : ℕ := 19

/-- The difference between movies watched and books read -/
def movie_book_difference : ℕ := 3

theorem crazy_silly_school_books :
  num_books = movies_watched - movie_book_difference :=
by sorry

end crazy_silly_school_books_l1868_186832


namespace arc_length_45_degrees_l1868_186814

/-- Given a circle with circumference 90 meters, prove that an arc subtending a 45° angle at the center has a length of 11.25 meters. -/
theorem arc_length_45_degrees (D : Real) (EF : Real) :
  D = 90 →  -- circumference of the circle
  EF = (45 / 360) * D →  -- arc length is proportional to the angle it subtends
  EF = 11.25 := by
sorry

end arc_length_45_degrees_l1868_186814


namespace least_integer_x_l1868_186882

theorem least_integer_x (x : ℤ) : (∀ y : ℤ, |3 * y + 5| ≤ 21 → y ≥ -8) ∧ |3 * (-8) + 5| ≤ 21 := by
  sorry

end least_integer_x_l1868_186882


namespace school_location_minimizes_distance_l1868_186877

/-- Represents the distance between two villages in kilometers -/
def village_distance : ℝ := 3

/-- Represents the number of students in village A -/
def students_A : ℕ := 300

/-- Represents the number of students in village B -/
def students_B : ℕ := 200

/-- Represents the distance from village A to the school -/
def school_distance (x : ℝ) : ℝ := x

/-- Calculates the total distance traveled by all students -/
def total_distance (x : ℝ) : ℝ :=
  students_A * school_distance x + students_B * (village_distance - school_distance x)

/-- Theorem: The total distance is minimized when the school is built in village A -/
theorem school_location_minimizes_distance :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ village_distance →
    total_distance 0 ≤ total_distance x :=
by sorry

end school_location_minimizes_distance_l1868_186877


namespace second_hand_store_shirt_price_l1868_186863

/-- The price of a shirt sold to the second-hand store -/
def shirt_price : ℚ := 4

theorem second_hand_store_shirt_price :
  let pants_sold : ℕ := 3
  let shorts_sold : ℕ := 5
  let shirts_sold : ℕ := 5
  let pants_price : ℚ := 5
  let shorts_price : ℚ := 3
  let new_shirts_bought : ℕ := 2
  let new_shirt_price : ℚ := 10
  let remaining_money : ℚ := 30

  shirt_price * shirts_sold + 
  pants_price * pants_sold + 
  shorts_price * shorts_sold = 
  remaining_money + new_shirt_price * new_shirts_bought := by sorry

end second_hand_store_shirt_price_l1868_186863


namespace square_neq_iff_neq_and_neq_neg_l1868_186831

theorem square_neq_iff_neq_and_neq_neg (x y : ℝ) :
  x^2 ≠ y^2 ↔ x ≠ y ∧ x ≠ -y := by
  sorry

end square_neq_iff_neq_and_neq_neg_l1868_186831


namespace quadratic_function_value_l1868_186852

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_value (a b c : ℝ) (h : a ≠ 0) :
  f a b c (-3) = 7 →
  f a b c (-2) = 0 →
  f a b c 0 = -8 →
  f a b c 1 = -9 →
  f a b c 3 = -5 →
  f a b c 5 = 7 →
  f a b c 2 = -8 := by
  sorry

end quadratic_function_value_l1868_186852


namespace canoe_production_sum_l1868_186847

/-- Represents the number of canoes built in the first month -/
def first_month_canoes : ℕ := 7

/-- Represents the ratio of canoes built between consecutive months -/
def monthly_ratio : ℕ := 3

/-- Represents the number of months considered -/
def num_months : ℕ := 6

/-- Calculates the sum of a geometric sequence -/
def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem canoe_production_sum :
  geometric_sum first_month_canoes monthly_ratio num_months = 2548 := by
  sorry

end canoe_production_sum_l1868_186847


namespace special_ellipse_properties_l1868_186871

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_angle : c / a = Real.sqrt 3 / 2
  h_dist : a + c = 2 + Real.sqrt 3

/-- The line passing through a focus of the ellipse -/
structure FocusLine where
  m : ℝ

/-- The theorem statement -/
theorem special_ellipse_properties (e : SpecialEllipse) (l : FocusLine) :
  (e.a = 2 ∧ e.b = 1) ∧
  (l.m = Real.sqrt 2 ∨ l.m = -Real.sqrt 2) := by
  sorry

end special_ellipse_properties_l1868_186871


namespace area_of_triangle_DBC_l1868_186806

/-- Given points A, B, C, D, and E in a coordinate plane, where D and E are midpoints of AB and BC respectively, prove that the area of triangle DBC is 30 square units. -/
theorem area_of_triangle_DBC (A B C D E : ℝ × ℝ) : 
  A = (0, 10) → 
  B = (0, 0) → 
  C = (12, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) → 
  (1 / 2) * (C.1 - B.1) * D.2 = 30 := by
  sorry

end area_of_triangle_DBC_l1868_186806


namespace trig_identity_for_point_l1868_186844

/-- Given a point P on the terminal side of angle α with coordinates (4a, -3a) where a < 0,
    prove that 2sin(α) + cos(α) = 2/5 -/
theorem trig_identity_for_point (a : ℝ) (α : ℝ) (h : a < 0) :
  let x : ℝ := 4 * a
  let y : ℝ := -3 * a
  let r : ℝ := Real.sqrt (x^2 + y^2)
  2 * (y / r) + (x / r) = 2 / 5 := by sorry

end trig_identity_for_point_l1868_186844


namespace not_right_triangle_3_4_5_squared_l1868_186890

-- Define a function to check if three numbers can form a right triangle
def isRightTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Theorem stating that 3^2, 4^2, 5^2 cannot form a right triangle
theorem not_right_triangle_3_4_5_squared :
  ¬ isRightTriangle (3^2) (4^2) (5^2) := by
  sorry


end not_right_triangle_3_4_5_squared_l1868_186890


namespace base_b_subtraction_divisibility_other_bases_divisible_l1868_186836

theorem base_b_subtraction_divisibility (b : ℤ) : 
  b = 6 ↔ (b^3 - 3*b^2 + 3*b - 2) % 5 ≠ 0 := by sorry

theorem other_bases_divisible : 
  ∀ b ∈ ({5, 7, 9, 10} : Set ℤ), (b^3 - 3*b^2 + 3*b - 2) % 5 = 0 := by sorry

end base_b_subtraction_divisibility_other_bases_divisible_l1868_186836


namespace raine_initial_payment_l1868_186830

/-- The price of a bracelet in dollars -/
def bracelet_price : ℕ := 15

/-- The price of a gold heart necklace in dollars -/
def necklace_price : ℕ := 10

/-- The price of a personalized coffee mug in dollars -/
def mug_price : ℕ := 20

/-- The number of bracelets Raine bought -/
def bracelets_bought : ℕ := 3

/-- The number of gold heart necklaces Raine bought -/
def necklaces_bought : ℕ := 2

/-- The number of personalized coffee mugs Raine bought -/
def mugs_bought : ℕ := 1

/-- The amount of change Raine received in dollars -/
def change_received : ℕ := 15

/-- The theorem stating the amount Raine initially gave -/
theorem raine_initial_payment : 
  bracelet_price * bracelets_bought + 
  necklace_price * necklaces_bought + 
  mug_price * mugs_bought + 
  change_received = 100 := by
  sorry

end raine_initial_payment_l1868_186830


namespace hot_air_balloon_problem_l1868_186860

theorem hot_air_balloon_problem (initial_balloons : ℕ) 
  (h1 : initial_balloons = 200)
  (h2 : initial_balloons > 0) : 
  let first_blown_up := initial_balloons / 5
  let second_blown_up := 2 * first_blown_up
  let total_blown_up := first_blown_up + second_blown_up
  initial_balloons - total_blown_up = 80 := by
sorry

end hot_air_balloon_problem_l1868_186860


namespace four_inch_cube_multi_painted_l1868_186817

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  painted_faces : ℕ
  hsl : side_length = n
  hpf : painted_faces = 6

/-- Represents a smaller cube cut from a larger cube -/
structure SmallCube where
  painted_faces : ℕ

/-- Function to count cubes with at least two painted faces -/
def count_multi_painted_cubes (n : ℕ) : ℕ :=
  8 + 12 * (n - 2)

/-- Theorem statement -/
theorem four_inch_cube_multi_painted (c : Cube 4) :
  count_multi_painted_cubes c.side_length = 40 :=
sorry

end four_inch_cube_multi_painted_l1868_186817


namespace no_four_distinct_naturals_power_sum_equality_l1868_186872

theorem no_four_distinct_naturals_power_sum_equality :
  ¬∃ (x y z t : ℕ), x ≠ y ∧ x ≠ z ∧ x ≠ t ∧ y ≠ z ∧ y ≠ t ∧ z ≠ t ∧ x^x + y^y = z^z + t^t :=
by sorry

end no_four_distinct_naturals_power_sum_equality_l1868_186872


namespace perfect_square_prime_exponents_l1868_186824

theorem perfect_square_prime_exponents (p q r : Nat) : 
  Prime p ∧ Prime q ∧ Prime r → 
  (∃ (n : Nat), p^q + p^r = n^2) ↔ 
  ((p = 2 ∧ q = 2 ∧ r = 5) ∨ 
   (p = 2 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 3 ∧ q = 2 ∧ r = 3) ∨ 
   (p = 3 ∧ q = 3 ∧ r = 2) ∨ 
   (p = 2 ∧ q = r ∧ q ≥ 3 ∧ Odd q)) := by
  sorry

#check perfect_square_prime_exponents

end perfect_square_prime_exponents_l1868_186824


namespace magazine_budget_cut_percentage_l1868_186889

/-- Given a company's yearly magazine subscription cost and desired budget cut,
    calculate the percentage reduction in the budget. -/
theorem magazine_budget_cut_percentage
  (original_cost : ℝ)
  (budget_cut : ℝ)
  (h_original_cost : original_cost = 940)
  (h_budget_cut : budget_cut = 611) :
  (budget_cut / original_cost) * 100 = 65 := by
sorry

end magazine_budget_cut_percentage_l1868_186889


namespace simplify_star_expression_l1868_186862

/-- Custom binary operation ※ for rational numbers -/
def star (a b : ℚ) : ℚ := 2 * a - b

/-- Theorem stating the equivalence of the expression and its simplified form -/
theorem simplify_star_expression (x y : ℚ) : 
  star (star (x - y) (x + y)) (-3 * y) = 2 * x - 3 * y :=
sorry

end simplify_star_expression_l1868_186862


namespace tangent_line_of_conic_section_l1868_186839

/-- Conic section equation -/
def ConicSection (A B C D E F : ℝ) (x y : ℝ) : Prop :=
  A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0

/-- Tangent line equation -/
def TangentLine (A B C D E F x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  2 * A * x₀ * x + B * (x₀ * y + x * y₀) + 2 * C * y₀ * y + 
  D * (x₀ + x) + E * (y₀ + y) + 2 * F = 0

theorem tangent_line_of_conic_section 
  (A B C D E F x₀ y₀ : ℝ) :
  ConicSection A B C D E F x₀ y₀ →
  ∃ ε > 0, ∀ x y : ℝ, 
    0 < (x - x₀)^2 + (y - y₀)^2 ∧ (x - x₀)^2 + (y - y₀)^2 < ε^2 →
    ConicSection A B C D E F x y →
    TangentLine A B C D E F x₀ y₀ x y := by
  sorry

end tangent_line_of_conic_section_l1868_186839


namespace remainder_3_pow_210_mod_17_l1868_186837

theorem remainder_3_pow_210_mod_17 : (3^210 : ℕ) % 17 = 9 := by
  sorry

end remainder_3_pow_210_mod_17_l1868_186837


namespace total_books_read_formula_l1868_186859

/-- The total number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  let books_per_month := 5
  let months_per_year := 12
  let books_per_student_per_year := books_per_month * months_per_year
  books_per_student_per_year * c * s

/-- Theorem stating the total number of books read by the entire student body in one year -/
theorem total_books_read_formula (c s : ℕ) :
  total_books_read c s = 60 * c * s :=
by sorry

end total_books_read_formula_l1868_186859


namespace perfect_square_trinomial_k_l1868_186810

/-- A trinomial ax^2 + bxy + cy^2 is a perfect square if and only if b^2 = 4ac -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  b^2 = 4*a*c

/-- The value of k for which 9x^2 - kxy + 4y^2 is a perfect square trinomial -/
theorem perfect_square_trinomial_k : 
  ∃ (k : ℝ), is_perfect_square_trinomial 9 (-k) 4 ∧ (k = 12 ∨ k = -12) :=
sorry

end perfect_square_trinomial_k_l1868_186810
