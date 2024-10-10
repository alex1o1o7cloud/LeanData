import Mathlib

namespace inverse_value_l2056_205637

-- Define a function f with an inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the conditions
axiom has_inverse : Function.RightInverse f_inv f ∧ Function.LeftInverse f_inv f
axiom f_at_2 : f 2 = -1

-- State the theorem
theorem inverse_value : f_inv (-1) = 2 := by sorry

end inverse_value_l2056_205637


namespace smallest_three_digit_number_with_divisibility_properties_l2056_205638

theorem smallest_three_digit_number_with_divisibility_properties :
  ∃ (n : ℕ), 
    100 ≤ n ∧ n ≤ 999 ∧
    (n - 7) % 7 = 0 ∧
    (n - 8) % 8 = 0 ∧
    (n - 9) % 9 = 0 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < n →
      ¬((m - 7) % 7 = 0 ∧ (m - 8) % 8 = 0 ∧ (m - 9) % 9 = 0) :=
by
  -- The proof goes here
  sorry

end smallest_three_digit_number_with_divisibility_properties_l2056_205638


namespace orange_apple_weight_equivalence_l2056_205683

/-- Given that 14 oranges weigh the same as 10 apples, 
    prove that 42 oranges weigh the same as 30 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℝ),
  orange_weight > 0 →
  apple_weight > 0 →
  14 * orange_weight = 10 * apple_weight →
  42 * orange_weight = 30 * apple_weight :=
by
  sorry

#check orange_apple_weight_equivalence

end orange_apple_weight_equivalence_l2056_205683


namespace f_tangent_perpendicular_range_l2056_205650

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * (m + exp (-x))

theorem f_tangent_perpendicular_range :
  ∃ (a b : Set ℝ), a = Set.Ioo 0 (exp (-2)) ∧
  (∀ m : ℝ, m ∈ a ↔ 
    ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
      (deriv (f m)) x₁ = 0 ∧ 
      (deriv (f m)) x₂ = 0) :=
sorry

end f_tangent_perpendicular_range_l2056_205650


namespace square_plus_one_geq_double_l2056_205605

theorem square_plus_one_geq_double (x : ℝ) : x^2 + 1 ≥ 2*x := by
  sorry

end square_plus_one_geq_double_l2056_205605


namespace closest_to_200_l2056_205680

def problem_value : ℝ := 2.54 * 7.89 * (4.21 + 5.79)

def options : List ℝ := [150, 200, 250, 300, 350]

theorem closest_to_200 :
  ∀ x ∈ options, x ≠ 200 → |problem_value - 200| < |problem_value - x| :=
by sorry

end closest_to_200_l2056_205680


namespace total_protest_days_equals_29_625_l2056_205603

/-- Calculates the total number of days spent at four protests -/
def total_protest_days (first_protest : ℝ) (second_increase : ℝ) (third_increase : ℝ) (fourth_increase : ℝ) : ℝ :=
  let second_protest := first_protest * (1 + second_increase)
  let third_protest := second_protest * (1 + third_increase)
  let fourth_protest := third_protest * (1 + fourth_increase)
  first_protest + second_protest + third_protest + fourth_protest

/-- Theorem stating that the total number of days spent at four protests equals 29.625 -/
theorem total_protest_days_equals_29_625 :
  total_protest_days 4 0.25 0.5 0.75 = 29.625 := by
  sorry

end total_protest_days_equals_29_625_l2056_205603


namespace children_attendance_l2056_205690

/-- Proves the number of children attending a concert given ticket prices and total revenue -/
theorem children_attendance (adult_price : ℕ) (adult_count : ℕ) (total_revenue : ℕ) : 
  adult_price = 26 →
  adult_count = 183 →
  total_revenue = 5122 →
  ∃ (child_count : ℕ), 
    adult_price * adult_count + (adult_price / 2) * child_count = total_revenue ∧
    child_count = 28 := by
  sorry

end children_attendance_l2056_205690


namespace space_diagonals_of_specific_polyhedron_l2056_205652

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ
  pentagon_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  sorry

/-- The main theorem stating that a convex polyhedron with given properties has 310 space diagonals -/
theorem space_diagonals_of_specific_polyhedron :
  ∃ Q : ConvexPolyhedron,
    Q.vertices = 30 ∧
    Q.edges = 70 ∧
    Q.faces = 40 ∧
    Q.triangular_faces = 20 ∧
    Q.quadrilateral_faces = 15 ∧
    Q.pentagon_faces = 5 ∧
    space_diagonals Q = 310 :=
  sorry

end space_diagonals_of_specific_polyhedron_l2056_205652


namespace isosceles_triangle_l2056_205694

theorem isosceles_triangle (A B C : ℝ) (h_sum : A + B + C = π) :
  let f := fun x : ℝ => x^2 - x * Real.cos A * Real.cos B + 2 * Real.sin (C/2)^2
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = (1/2) * x₁ * x₂ → A = B := by
  sorry

end isosceles_triangle_l2056_205694


namespace min_value_a_l2056_205626

theorem min_value_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → 2 * a * Real.exp (2 * x) - Real.log x + Real.log a ≥ 0) →
  a ≥ 1 / (2 * Real.exp 1) :=
by sorry

end min_value_a_l2056_205626


namespace x_0_value_l2056_205649

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem x_0_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 2) → x₀ = Real.exp 1 := by
  sorry

end x_0_value_l2056_205649


namespace candidate_vote_percentage_l2056_205615

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 357000) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 75 / 100 :=
by sorry

end candidate_vote_percentage_l2056_205615


namespace new_man_weight_l2056_205632

theorem new_man_weight (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) :
  n = 10 →
  avg_increase = 2.5 →
  replaced_weight = 68 →
  (n : ℝ) * avg_increase + replaced_weight = 93 :=
by
  sorry

end new_man_weight_l2056_205632


namespace solve_for_y_l2056_205601

theorem solve_for_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 8) : y = -4 := by
  sorry

end solve_for_y_l2056_205601


namespace cricket_bat_profit_l2056_205659

/-- Proves that the profit from selling a cricket bat is approximately $215.29 --/
theorem cricket_bat_profit (selling_price : ℝ) (profit_percentage : ℝ) (h1 : selling_price = 850) (h2 : profit_percentage = 33.85826771653544) :
  ∃ (profit : ℝ), abs (profit - 215.29) < 0.01 := by
  sorry

end cricket_bat_profit_l2056_205659


namespace quadrilateral_to_parallelogram_l2056_205620

-- Define the points
variable (A B C D E F O : ℝ × ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def is_midpoint (M X Y : ℝ × ℝ) : Prop := sorry

def segments_intersect (P Q R S : ℝ × ℝ) (I : ℝ × ℝ) : Prop := sorry

def divides_into_three_equal_parts (P Q R S : ℝ × ℝ) : Prop := sorry

def is_parallelogram (A B C D : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem quadrilateral_to_parallelogram 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_midpoint_E : is_midpoint E A D)
  (h_midpoint_F : is_midpoint F B C)
  (h_intersect : segments_intersect C E D F O)
  (h_divide_AO : divides_into_three_equal_parts A O C D)
  (h_divide_BO : divides_into_three_equal_parts B O C D) :
  is_parallelogram A B C D :=
sorry

end quadrilateral_to_parallelogram_l2056_205620


namespace fifteen_points_max_planes_l2056_205678

def max_planes (n : ℕ) : ℕ := n.choose 3

theorem fifteen_points_max_planes :
  max_planes 15 = 455 :=
by sorry

end fifteen_points_max_planes_l2056_205678


namespace simplify_radical_product_l2056_205656

theorem simplify_radical_product (y z : ℝ) :
  Real.sqrt (50 * y) * Real.sqrt (18 * z) * Real.sqrt (32 * y) = 40 * y * Real.sqrt (2 * z) :=
by sorry

end simplify_radical_product_l2056_205656


namespace farmer_randy_cotton_acres_l2056_205670

/-- The number of acres a single tractor can plant in one day -/
def acres_per_day : ℕ := 68

/-- The number of tractors working for the first two days -/
def tractors_first_two_days : ℕ := 2

/-- The number of tractors working for the last three days -/
def tractors_last_three_days : ℕ := 7

/-- The number of days in the first period -/
def first_period_days : ℕ := 2

/-- The number of days in the second period -/
def second_period_days : ℕ := 3

/-- The total number of acres Farmer Randy needs to have planted -/
def total_acres : ℕ := 1700

theorem farmer_randy_cotton_acres :
  total_acres = 
    acres_per_day * first_period_days * tractors_first_two_days + 
    acres_per_day * second_period_days * tractors_last_three_days :=
by sorry

end farmer_randy_cotton_acres_l2056_205670


namespace correct_answer_l2056_205684

theorem correct_answer (x : ℝ) (h : 3 * x = 90) : x / 3 = 10 := by
  sorry

end correct_answer_l2056_205684


namespace double_angle_sine_15_degrees_l2056_205685

theorem double_angle_sine_15_degrees :
  2 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end double_angle_sine_15_degrees_l2056_205685


namespace distance_and_angle_from_origin_l2056_205675

/-- In a rectangular coordinate system, for a point (12, 5): -/
theorem distance_and_angle_from_origin :
  let x : ℝ := 12
  let y : ℝ := 5
  let distance := Real.sqrt (x^2 + y^2)
  let angle := Real.arctan (y / x)
  (distance = 13 ∧ angle = Real.arctan (5 / 12)) := by
  sorry

end distance_and_angle_from_origin_l2056_205675


namespace garden_theorem_l2056_205677

def garden_problem (initial_plants : ℕ) (day1_eaten : ℕ) (day3_eaten : ℕ) : ℕ :=
  let remaining_day1 := initial_plants - day1_eaten
  let remaining_day2 := remaining_day1 / 2
  remaining_day2 - day3_eaten

theorem garden_theorem :
  garden_problem 30 20 1 = 4 := by
  sorry

#eval garden_problem 30 20 1

end garden_theorem_l2056_205677


namespace min_time_35_minutes_l2056_205699

/-- Represents a rectangular parallelepiped -/
structure Brick where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a point moving on the surface of a brick -/
structure MovingPoint where
  v_x : ℝ → ℝ
  v_y : ℝ → ℝ
  velocity_constraint : ∀ t, (v_x t)^2 + 4*(v_y t)^2 = 1

/-- The minimum time for a point to travel from one vertex of the lower base
    to the opposite vertex of the upper base of a brick -/
def min_travel_time (b : Brick) (p : MovingPoint) : ℝ := sorry

/-- The theorem stating the minimum travel time for the given problem -/
theorem min_time_35_minutes (b : Brick) (p : MovingPoint)
    (h1 : b.length = 28)
    (h2 : b.width = 9)
    (h3 : b.height = 6) :
  min_travel_time b p = 35 := by sorry

end min_time_35_minutes_l2056_205699


namespace min_groups_for_class_l2056_205665

/-- Given a class of 30 students and a maximum group size of 12,
    proves that the minimum number of equal-sized groups is 3. -/
theorem min_groups_for_class (total_students : ℕ) (max_group_size : ℕ) :
  total_students = 30 →
  max_group_size = 12 →
  ∃ (group_size : ℕ), 
    group_size ≤ max_group_size ∧
    total_students % group_size = 0 ∧
    (total_students / group_size = 3) ∧
    ∀ (other_size : ℕ), 
      other_size ≤ max_group_size →
      total_students % other_size = 0 →
      total_students / other_size ≥ 3 :=
by sorry

end min_groups_for_class_l2056_205665


namespace boys_without_calculators_l2056_205671

theorem boys_without_calculators (total_students : Nat) (boys : Nat) (students_with_calculators : Nat) (girls_with_calculators : Nat)
  (h1 : total_students = 30)
  (h2 : boys = 20)
  (h3 : students_with_calculators = 25)
  (h4 : girls_with_calculators = 18)
  : total_students - boys - (students_with_calculators - girls_with_calculators) = 13 := by
  sorry

end boys_without_calculators_l2056_205671


namespace triangle_solutions_l2056_205602

theorem triangle_solutions (a b : ℝ) (B : ℝ) :
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  B = 45 * π / 180 →
  ∃ (A C c : ℝ),
    ((A = 60 * π / 180 ∧ C = 75 * π / 180 ∧ c = (Real.sqrt 2 + Real.sqrt 6) / 2) ∨
     (A = 120 * π / 180 ∧ C = 15 * π / 180 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2)) ∧
    A + B + C = π ∧
    a / Real.sin A = b / Real.sin B ∧
    a / Real.sin A = c / Real.sin C :=
by sorry

end triangle_solutions_l2056_205602


namespace function_properties_l2056_205667

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom cond1 : ∀ x, f (10 + x) = f (10 - x)
axiom cond2 : ∀ x, f (20 - x) = -f (20 + x)

-- Theorem to prove
theorem function_properties :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 40) = f x) :=
by sorry

end function_properties_l2056_205667


namespace domain_of_f_l2056_205612

noncomputable def f (x : ℝ) : ℝ := (3 * x^2) / Real.sqrt (1 - x) + Real.log (3 * x + 1) / Real.log 10

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1/3 < x ∧ x < 1} := by sorry

end domain_of_f_l2056_205612


namespace cube_sum_l2056_205600

theorem cube_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end cube_sum_l2056_205600


namespace remainder_mod_24_l2056_205654

theorem remainder_mod_24 (m : ℤ) (h : m % 288 = 47) : m % 24 = 23 := by
  sorry

end remainder_mod_24_l2056_205654


namespace triangle_condition_equivalent_to_m_gt_2_l2056_205628

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

-- Define the interval [0,2]
def I : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Define the triangle inequality condition
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem triangle_condition_equivalent_to_m_gt_2 :
  (∀ m : ℝ, (∀ a b c : ℝ, a ∈ I → b ∈ I → c ∈ I → a ≠ b → b ≠ c → a ≠ c →
    triangle_inequality (f m a) (f m b) (f m c)) ↔ m > 2) :=
sorry

end triangle_condition_equivalent_to_m_gt_2_l2056_205628


namespace power_sum_and_division_equals_82_l2056_205636

theorem power_sum_and_division_equals_82 : 2^0 + 9^5 / 9^3 = 82 := by
  sorry

end power_sum_and_division_equals_82_l2056_205636


namespace smallest_x_congruence_and_divisible_l2056_205698

theorem smallest_x_congruence_and_divisible (x : ℕ) : x = 45 ↔ 
  (x > 0 ∧ 
   (x + 6721) % 12 = 3458 % 12 ∧ 
   x % 5 = 0 ∧
   ∀ y : ℕ, y > 0 → (y + 6721) % 12 = 3458 % 12 → y % 5 = 0 → y ≥ x) :=
by sorry

end smallest_x_congruence_and_divisible_l2056_205698


namespace division_problem_l2056_205608

theorem division_problem (dividend quotient divisor remainder n : ℕ) : 
  dividend = 86 →
  remainder = 6 →
  divisor = 5 * quotient →
  divisor = 3 * remainder + n →
  dividend = divisor * quotient + remainder →
  n = 2 := by
sorry

end division_problem_l2056_205608


namespace distinct_prime_factors_of_divisor_sum_450_l2056_205644

/-- The sum of positive divisors of a natural number n -/
noncomputable def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
noncomputable def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of distinct prime factors of the sum of positive divisors of 450 is 2 -/
theorem distinct_prime_factors_of_divisor_sum_450 : 
  num_distinct_prime_factors (sum_of_divisors 450) = 2 := by sorry

end distinct_prime_factors_of_divisor_sum_450_l2056_205644


namespace range_of_a_l2056_205674

/-- A quadratic function y = x^2 + 2(a-1)x + 2 -/
def f (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The function decreases monotonically on (-∞, 4] -/
def decreases_on_left (a : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ ∧ x₂ ≤ 4 → f a x₁ ≥ f a x₂

/-- The function increases monotonically on [5, +∞) -/
def increases_on_right (a : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 5 ≤ x₁ ∧ x₁ ≤ x₂ → f a x₁ ≤ f a x₂

/-- The range of a given the monotonicity conditions -/
theorem range_of_a (a : ℝ) 
  (h1 : decreases_on_left a) 
  (h2 : increases_on_right a) : 
  -4 ≤ a ∧ a ≤ -3 := by
  sorry

end range_of_a_l2056_205674


namespace union_of_sets_l2056_205661

theorem union_of_sets : 
  let A : Set ℕ := {1, 3, 7, 8}
  let B : Set ℕ := {1, 5, 8}
  A ∪ B = {1, 3, 5, 7, 8} := by
sorry

end union_of_sets_l2056_205661


namespace ajays_income_l2056_205682

/-- Ajay's monthly income in Rupees -/
def monthly_income : ℝ := 90000

/-- Percentage of income spent on household items -/
def household_percentage : ℝ := 0.50

/-- Percentage of income spent on clothes -/
def clothes_percentage : ℝ := 0.25

/-- Percentage of income spent on medicines -/
def medicines_percentage : ℝ := 0.15

/-- Amount saved in Rupees -/
def savings : ℝ := 9000

theorem ajays_income :
  monthly_income * household_percentage +
  monthly_income * clothes_percentage +
  monthly_income * medicines_percentage +
  savings = monthly_income :=
by sorry

end ajays_income_l2056_205682


namespace quadratic_coefficient_l2056_205672

theorem quadratic_coefficient (a : ℝ) : 
  (a * (1/2)^2 + 9 * (1/2) - 5 = 0) → a = 2 := by
  sorry

end quadratic_coefficient_l2056_205672


namespace specific_value_problem_l2056_205645

theorem specific_value_problem (x : ℕ) (specific_value : ℕ) 
  (h1 : 15 * x = specific_value) (h2 : x = 11) : 
  specific_value = 165 := by
  sorry

end specific_value_problem_l2056_205645


namespace x_minus_y_value_l2056_205687

theorem x_minus_y_value (x y : ℝ) 
  (hx : |x| = 4)
  (hy : |y| = 2)
  (hxy : x * y < 0) :
  x - y = 6 ∨ x - y = -6 := by
sorry

end x_minus_y_value_l2056_205687


namespace parallel_line_through_point_l2056_205681

/-- Given a line L1 with equation 3x - 6y = 9, prove that the line L2 with equation y = (1/2)x - 1
    is parallel to L1 and passes through the point (2,0). -/
theorem parallel_line_through_point (x y : ℝ) : 
  (3 * x - 6 * y = 9) →  -- Equation of line L1
  (y = (1/2) * x - 1) →  -- Equation of line L2
  (∃ m b : ℝ, y = m * x + b ∧ m = 1/2) →  -- L2 is in slope-intercept form with slope 1/2
  (0 = (1/2) * 2 - 1) →  -- L2 passes through (2,0)
  (∀ x₁ y₁ x₂ y₂ : ℝ, (3 * x₁ - 6 * y₁ = 9 ∧ 3 * x₂ - 6 * y₂ = 9) → 
    ((y₂ - y₁) / (x₂ - x₁) = 1/2)) →  -- Slope of L1 is 1/2
  (y = (1/2) * x - 1)  -- Conclusion: equation of L2
  := by sorry

end parallel_line_through_point_l2056_205681


namespace max_leftover_apples_l2056_205609

theorem max_leftover_apples (n : ℕ) (h : n > 0) : 
  ∃ (m : ℕ), m > 0 ∧ m < n ∧ 
  ∀ (total : ℕ), total ≥ n * (total / n) + m → total / n = (total - m) / n :=
by
  sorry

end max_leftover_apples_l2056_205609


namespace mehki_age_l2056_205629

/-- Given the ages of Zrinka, Jordyn, and Mehki, prove Mehki's age is 22 years. -/
theorem mehki_age (zrinka jordyn mehki : ℕ) 
  (h1 : mehki = jordyn + 10)
  (h2 : jordyn = 2 * zrinka)
  (h3 : zrinka = 6) : 
  mehki = 22 := by
sorry

end mehki_age_l2056_205629


namespace flower_count_l2056_205676

theorem flower_count (num_bees : ℕ) (num_flowers : ℕ) : 
  num_bees = 3 → num_bees = num_flowers - 2 → num_flowers = 5 := by
  sorry

end flower_count_l2056_205676


namespace range_of_a_l2056_205634

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

-- State the theorem
theorem range_of_a (a : ℝ) (h : f a ≤ f 2) : a ∈ Set.Icc (-2) 2 := by
  sorry

end range_of_a_l2056_205634


namespace min_value_of_expression_l2056_205619

theorem min_value_of_expression (a b c : ℤ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (m : ℤ), m = 3 ∧ ∀ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z →
    3*x^2 + 2*y^2 + 4*z^2 - x*y - 3*y*z - 5*z*x ≥ m :=
by sorry

end min_value_of_expression_l2056_205619


namespace max_imaginary_part_of_roots_l2056_205647

theorem max_imaginary_part_of_roots (z : ℂ) (θ : ℝ) :
  z^12 - z^9 + z^6 - z^3 + 1 = 0 →
  -π/2 ≤ θ ∧ θ ≤ π/2 →
  z.im = Real.sin θ →
  z.im ≤ Real.sin (84 * π / 180) :=
by sorry

end max_imaginary_part_of_roots_l2056_205647


namespace quadratic_function_sign_l2056_205622

/-- Given a quadratic function f(x) = x^2 - x + a, where f(-m) < 0,
    prove that f(m+1) is negative. -/
theorem quadratic_function_sign (a m : ℝ) : 
  let f := λ x : ℝ => x^2 - x + a
  f (-m) < 0 → f (m + 1) < 0 := by sorry

end quadratic_function_sign_l2056_205622


namespace position_of_2008_l2056_205664

/-- Define the position of a number in the pattern -/
structure Position where
  row : Nat
  column : Nat

/-- Function to calculate the position of a number in the pattern -/
noncomputable def calculatePosition (n : Nat) : Position :=
  sorry  -- The actual implementation would go here

/-- Theorem stating that 2008 is in row 18, column 45 -/
theorem position_of_2008 : calculatePosition 2008 = ⟨18, 45⟩ := by
  sorry

#check position_of_2008

end position_of_2008_l2056_205664


namespace invariant_parity_and_final_digit_l2056_205617

/-- Represents the count of each digit -/
structure DigitCounts where
  zeros : ℕ
  ones : ℕ
  twos : ℕ

/-- Represents the possible operations on the board -/
inductive Operation
  | replaceZeroOne
  | replaceOneTwo
  | replaceZeroTwo

/-- Applies an operation to the digit counts -/
def applyOperation (counts : DigitCounts) (op : Operation) : DigitCounts :=
  match op with
  | Operation.replaceZeroOne => ⟨counts.zeros - 1, counts.ones - 1, counts.twos + 1⟩
  | Operation.replaceOneTwo => ⟨counts.zeros + 1, counts.ones - 1, counts.twos - 1⟩
  | Operation.replaceZeroTwo => ⟨counts.zeros - 1, counts.ones + 1, counts.twos - 1⟩

/-- The parity of the sum of digit counts -/
def sumParity (counts : DigitCounts) : ℕ :=
  (counts.zeros + counts.ones + counts.twos) % 2

/-- The final remaining digit -/
def finalDigit (initialCounts : DigitCounts) : ℕ :=
  if initialCounts.zeros % 2 ≠ initialCounts.ones % 2 ∧ initialCounts.zeros % 2 ≠ initialCounts.twos % 2 then 0
  else if initialCounts.ones % 2 ≠ initialCounts.zeros % 2 ∧ initialCounts.ones % 2 ≠ initialCounts.twos % 2 then 1
  else 2

theorem invariant_parity_and_final_digit (initialCounts : DigitCounts) (ops : List Operation) :
  (sumParity initialCounts = sumParity (ops.foldl applyOperation initialCounts)) ∧
  (finalDigit initialCounts = finalDigit (ops.foldl applyOperation initialCounts)) :=
sorry

end invariant_parity_and_final_digit_l2056_205617


namespace range_of_a_l2056_205621

-- Define the custom operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + 1) < 1) →
  -2 < a ∧ a < 2 :=
by sorry

end range_of_a_l2056_205621


namespace simplify_and_evaluate_one_l2056_205635

theorem simplify_and_evaluate_one (x y : ℚ) :
  x = 1/2 ∧ y = -1 →
  (1 * (2*x + y) * (2*x - y)) - 4*x*(x - y) = -3 := by
  sorry

end simplify_and_evaluate_one_l2056_205635


namespace A_intersect_B_l2056_205641

def A : Set ℕ := {0, 1, 2, 3}

def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a}

theorem A_intersect_B : A ∩ B = {0, 2} := by sorry

end A_intersect_B_l2056_205641


namespace work_done_by_combined_forces_l2056_205660

/-- Work done by combined forces -/
theorem work_done_by_combined_forces
  (F₁ : ℝ × ℝ)
  (F₂ : ℝ × ℝ)
  (S : ℝ × ℝ)
  (h₁ : F₁ = (Real.log 2, Real.log 2))
  (h₂ : F₂ = (Real.log 5, Real.log 2))
  (h₃ : S = (2 * Real.log 5, 1)) :
  (F₁.1 + F₂.1) * S.1 + (F₁.2 + F₂.2) * S.2 = 2 := by
  sorry

#check work_done_by_combined_forces

end work_done_by_combined_forces_l2056_205660


namespace friday_to_thursday_ratio_l2056_205686

def thursday_sales : ℝ := 210
def saturday_sales : ℝ := 150
def average_daily_sales : ℝ := 260

theorem friday_to_thursday_ratio :
  let total_sales := average_daily_sales * 3
  let friday_sales := total_sales - thursday_sales - saturday_sales
  friday_sales / thursday_sales = 2 := by sorry

end friday_to_thursday_ratio_l2056_205686


namespace binomial_12_choose_3_l2056_205625

theorem binomial_12_choose_3 : Nat.choose 12 3 = 220 := by
  sorry

end binomial_12_choose_3_l2056_205625


namespace cubic_harmonic_mean_root_condition_l2056_205614

/-- 
Given a cubic equation ax³ + bx² + cx + d = 0 with a ≠ 0 and d ≠ 0,
if one of its roots is equal to the harmonic mean of the other two roots,
then the coefficients satisfy the equation 27ad² - 9bcd + 2c³ = 0.
-/
theorem cubic_harmonic_mean_root_condition (a b c d : ℝ) 
  (ha : a ≠ 0) (hd : d ≠ 0) : 
  (∃ x₁ x₂ x₃ : ℝ, 
    (a * x₁^3 + b * x₁^2 + c * x₁ + d = 0) ∧ 
    (a * x₂^3 + b * x₂^2 + c * x₂ + d = 0) ∧ 
    (a * x₃^3 + b * x₃^2 + c * x₃ + d = 0) ∧ 
    (x₂ = 2 * x₁ * x₃ / (x₁ + x₃))) →
  27 * a * d^2 - 9 * b * c * d + 2 * c^3 = 0 := by
  sorry

end cubic_harmonic_mean_root_condition_l2056_205614


namespace M_mod_51_l2056_205643

def M : ℕ := sorry

theorem M_mod_51 : M % 51 = 34 := by sorry

end M_mod_51_l2056_205643


namespace curve_properties_l2056_205630

-- Define the curve y = ax^3 + bx
def curve (a b x : ℝ) : ℝ := a * x^3 + b * x

-- Define the derivative of the curve
def curve_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 + b

theorem curve_properties (a b : ℝ) :
  curve a b 2 = 2 ∧ 
  curve_derivative a b 2 = 9 →
  a * b = -3 ∧
  Set.Icc (-3/2 : ℝ) 3 ⊆ Set.Icc (-2 : ℝ) 18 ∧
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3/2 : ℝ) 3 ∧ 
             x₂ ∈ Set.Icc (-3/2 : ℝ) 3 ∧
             curve a b x₁ = -2 ∧
             curve a b x₂ = 18 := by
  sorry

end curve_properties_l2056_205630


namespace sequence_inequality_l2056_205668

-- Define the sequence a_n
def a (n k : ℤ) : ℝ := |n - k| + |n + 2*k|

-- State the theorem
theorem sequence_inequality (k : ℤ) :
  (∀ n : ℕ, a n k ≥ a 3 k) ∧ (a 3 k = a 4 k) →
  k ≤ -2 ∨ k ≥ 4 :=
by sorry

end sequence_inequality_l2056_205668


namespace arithmetic_sequence_geometric_mean_l2056_205624

/-- An arithmetic sequence with common difference d and first term 2d -/
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 2*d + (n - 1)*d

/-- The value of k for which a_k is the geometric mean of a_1 and a_{2k+1} -/
def k_value : ℕ := 3

theorem arithmetic_sequence_geometric_mean (d : ℝ) (h : d ≠ 0) :
  let a := arithmetic_sequence d
  (a k_value)^2 = a 1 * a (2*k_value + 1) := by
  sorry


end arithmetic_sequence_geometric_mean_l2056_205624


namespace solve_equation_one_solve_equation_two_l2056_205655

-- Equation 1
theorem solve_equation_one : 
  let f : ℝ → ℝ := λ x => 2 * x^2 - 4 * x - 3
  ∃ x₁ x₂ : ℝ, x₁ = 1 + (Real.sqrt 10) / 2 ∧ 
              x₂ = 1 - (Real.sqrt 10) / 2 ∧ 
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

-- Equation 2
theorem solve_equation_two :
  let g : ℝ → ℝ := λ x => (x^2 + x)^2 - x^2 - x - 30
  ∃ x₁ x₂ : ℝ, x₁ = -3 ∧ x₂ = 2 ∧ 
              g x₁ = 0 ∧ g x₂ = 0 ∧
              ∀ x : ℝ, g x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end solve_equation_one_solve_equation_two_l2056_205655


namespace tank_filling_ratio_l2056_205633

/-- Proves that the ratio of time B works alone to total time is 0.5 -/
theorem tank_filling_ratio : 
  ∀ (t_A t_B t_total : ℝ),
  t_A > 0 → t_B > 0 → t_total > 0 →
  (1 / t_A + 1 / t_B = 1 / 24) →
  t_total = 29.999999999999993 →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ t_total ∧ 
    t / t_B + (t_total - t) / 24 = 1) →
  ∃ t : ℝ, t / t_total = 0.5 := by
sorry

end tank_filling_ratio_l2056_205633


namespace system_of_equations_solution_fractional_equation_solution_l2056_205658

-- Problem 1: System of equations
theorem system_of_equations_solution :
  ∃! (x y : ℝ), x - y = 2 ∧ 2*x + y = 7 :=
sorry

-- Problem 2: Fractional equation
theorem fractional_equation_solution :
  ∃! y : ℝ, y ≠ 1 ∧ 3 / (1 - y) = y / (y - 1) - 5 :=
sorry

end system_of_equations_solution_fractional_equation_solution_l2056_205658


namespace regression_analysis_l2056_205631

structure RegressionData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  original_slope : ℝ
  original_intercept : ℝ
  x_mean : ℝ
  new_slope : ℝ

def positive_correlation (data : RegressionData) : Prop :=
  data.new_slope > 0

def new_regression_equation (data : RegressionData) : Prop :=
  ∃ new_intercept : ℝ, new_intercept = data.x_mean * (data.original_slope - data.new_slope) + data.original_intercept + 1

def decreased_rate_of_increase (data : RegressionData) : Prop :=
  data.new_slope < data.original_slope

theorem regression_analysis (data : RegressionData) 
  (h1 : data.original_slope = 2)
  (h2 : data.original_intercept = -1)
  (h3 : data.x_mean = 3)
  (h4 : data.new_slope = 1.2) :
  positive_correlation data ∧ 
  new_regression_equation data ∧ 
  decreased_rate_of_increase data := by
  sorry

end regression_analysis_l2056_205631


namespace triangle_abc_theorem_l2056_205689

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_abc_theorem (t : Triangle) 
  (h1 : t.b * Real.sin t.C = Real.sqrt 3)
  (h2 : t.B = π / 4)
  (h3 : (1 / 2) * t.a * t.c * Real.sin t.B = 9 / 2) :
  t.c = Real.sqrt 6 ∧ t.b = Real.sqrt 15 := by
  sorry

end triangle_abc_theorem_l2056_205689


namespace min_value_geometric_sequence_l2056_205613

/-- A geometric sequence of positive real numbers -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem min_value_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 1 * a 20 = 100) :
  ∃ m : ℝ, m = 20 ∧ ∀ x : ℝ, (a 7 + a 14 ≥ x ∧ (∃ y : ℝ, a 7 = y ∧ a 14 = y → a 7 + a 14 = x)) → x ≥ m :=
sorry

end min_value_geometric_sequence_l2056_205613


namespace prime_product_l2056_205627

theorem prime_product (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → 3 * p + 7 * q = 41 → (p + 1) * (q - 1) = 12 := by
  sorry

end prime_product_l2056_205627


namespace cistern_water_depth_l2056_205616

/-- Proves that for a cistern with given dimensions and wet surface area, the water depth is 1.25 meters -/
theorem cistern_water_depth
  (length : ℝ)
  (width : ℝ)
  (total_wet_area : ℝ)
  (h_length : length = 12)
  (h_width : width = 4)
  (h_wet_area : total_wet_area = 88)
  : ∃ (depth : ℝ), depth = 1.25 ∧ total_wet_area = length * width + 2 * depth * (length + width) :=
by sorry

end cistern_water_depth_l2056_205616


namespace jills_net_salary_l2056_205692

/-- Calculates the net monthly salary given the discretionary income ratio and remaining amount --/
def calculate_net_salary (discretionary_ratio : ℚ) (vacation_ratio : ℚ) (savings_ratio : ℚ) 
  (socializing_ratio : ℚ) (remaining_amount : ℚ) : ℚ :=
  remaining_amount / (discretionary_ratio * (1 - (vacation_ratio + savings_ratio + socializing_ratio)))

/-- Proves that given the specified conditions, Jill's net monthly salary is $3700 --/
theorem jills_net_salary :
  let discretionary_ratio : ℚ := 1/5
  let vacation_ratio : ℚ := 30/100
  let savings_ratio : ℚ := 20/100
  let socializing_ratio : ℚ := 35/100
  let remaining_amount : ℚ := 111
  calculate_net_salary discretionary_ratio vacation_ratio savings_ratio socializing_ratio remaining_amount = 3700 := by
  sorry

#eval calculate_net_salary (1/5) (30/100) (20/100) (35/100) 111

end jills_net_salary_l2056_205692


namespace algebraic_expression_value_l2056_205606

theorem algebraic_expression_value (x : ℝ) : 
  3 * x^2 - 2 * x - 1 = 2 → -9 * x^2 + 6 * x - 1 = -10 := by
  sorry

end algebraic_expression_value_l2056_205606


namespace cubic_minus_linear_factorization_l2056_205640

theorem cubic_minus_linear_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end cubic_minus_linear_factorization_l2056_205640


namespace circle_tangent_sum_radii_l2056_205653

/-- A circle with center C(r,r) is tangent to the positive x-axis and y-axis,
    and externally tangent to another circle centered at (5,0) with radius 2.
    The sum of all possible radii of the circle with center C is 14. -/
theorem circle_tangent_sum_radii :
  ∀ r : ℝ,
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = 14) :=
by sorry

end circle_tangent_sum_radii_l2056_205653


namespace min_value_sum_l2056_205662

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b^2)) + (b / (4 * c^3)) + (c / (5 * a^4)) ≥ 1 ∧
  ((a / (3 * b^2)) + (b / (4 * c^3)) + (c / (5 * a^4)) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end min_value_sum_l2056_205662


namespace cone_surface_area_ratio_l2056_205618

/-- Given a cone whose lateral surface development forms a sector with a central angle of 120° and radius l, 
    prove that the ratio of its total surface area to its lateral surface area is 4:3. -/
theorem cone_surface_area_ratio (l : ℝ) (h : l > 0) : 
  let r := l / 3
  let lateral_area := π * l * r
  let base_area := π * r^2
  let total_area := lateral_area + base_area
  (total_area / lateral_area : ℝ) = 4 / 3 := by
sorry

end cone_surface_area_ratio_l2056_205618


namespace subset_condition_l2056_205611

theorem subset_condition (a : ℝ) : 
  {x : ℝ | a ≤ x ∧ x < 7} ⊆ {x : ℝ | 2 < x ∧ x < 10} ↔ a > 2 := by
  sorry

end subset_condition_l2056_205611


namespace variance_2xi_plus_3_l2056_205696

variable (ξ : ℝ → ℝ)

-- D represents the variance operator
def D (X : ℝ → ℝ) : ℝ := sorry

-- Given condition
axiom variance_xi : D ξ = 2

-- Theorem to prove
theorem variance_2xi_plus_3 : D (fun ω => 2 * ξ ω + 3) = 8 := by sorry

end variance_2xi_plus_3_l2056_205696


namespace combined_transformation_correct_l2056_205623

def dilation_matrix (scale : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![scale, 0; 0, scale]

def reflection_x_axis : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0; 0, -1]

def combined_transformation : Matrix (Fin 2) (Fin 2) ℝ :=
  !![5, 0; 0, -5]

theorem combined_transformation_correct :
  combined_transformation = reflection_x_axis * dilation_matrix 5 := by
  sorry

end combined_transformation_correct_l2056_205623


namespace inequality_proof_l2056_205693

theorem inequality_proof (a b c : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) : 
  (a / (b + c + 1)) + (b / (c + a + 1)) + (c / (a + b + 1)) + (1 - a) * (1 - b) * (1 - c) ≤ 1 := by
sorry

end inequality_proof_l2056_205693


namespace slips_with_three_l2056_205673

/-- Given a bag of 20 slips with numbers 3 or 8, prove the number of 3s when expected value is 6 -/
theorem slips_with_three (total : ℕ) (value_one value_two : ℕ) (expected_value : ℚ) : 
  total = 20 →
  value_one = 3 →
  value_two = 8 →
  expected_value = 6 →
  ∃ (num_value_one : ℕ),
    num_value_one ≤ total ∧
    (num_value_one : ℚ) / total * value_one + (total - num_value_one : ℚ) / total * value_two = expected_value ∧
    num_value_one = 8 :=
by sorry

end slips_with_three_l2056_205673


namespace box_two_neg_one_zero_l2056_205646

def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

theorem box_two_neg_one_zero : box 2 (-1) 0 = -1/2 := by
  sorry

end box_two_neg_one_zero_l2056_205646


namespace two_oplus_neg_three_l2056_205648

/-- The ⊕ operation for rational numbers -/
def oplus (α β : ℚ) : ℚ := α * β + 1

/-- Theorem stating that 2 ⊕ (-3) = -5 -/
theorem two_oplus_neg_three : oplus 2 (-3) = -5 := by
  sorry

end two_oplus_neg_three_l2056_205648


namespace fifth_power_sum_equality_l2056_205642

theorem fifth_power_sum_equality : ∃! (n : ℕ), n > 0 ∧ 120^5 + 105^5 + 78^5 + 33^5 = n^5 := by
  sorry

end fifth_power_sum_equality_l2056_205642


namespace even_decreasing_function_inequality_l2056_205663

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is decreasing on (0, +∞) if for all x, y ∈ (0, +∞),
    x < y implies f(x) > f(y) -/
def IsDecreasingOnPositiveReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x > f y

theorem even_decreasing_function_inequality
  (f : ℝ → ℝ)
  (heven : IsEven f)
  (hdecr : IsDecreasingOnPositiveReals f) :
  f (-5) < f (-4) ∧ f (-4) < f 3 :=
sorry

end even_decreasing_function_inequality_l2056_205663


namespace circle_equation_correct_l2056_205679

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define the center of the circle
def center : ℝ × ℝ := (1, 1)

-- Define the point that the circle passes through
def point_on_circle : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem circle_equation_correct :
  -- The circle passes through the point (1, 0)
  circle_equation point_on_circle.1 point_on_circle.2 ∧
  -- The center is at the intersection of x=1 and x+y=2
  center.1 = 1 ∧ center.1 + center.2 = 2 ∧
  -- The equation represents a circle with the given center
  ∀ x y : ℝ, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = 1 :=
by
  sorry

end circle_equation_correct_l2056_205679


namespace tree_height_problem_l2056_205691

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₁ = h₂ + 16 →  -- One tree is 16 feet taller than the other
  h₂ / h₁ = 3 / 4 →  -- The heights are in the ratio 3:4
  h₁ = 64 :=  -- The taller tree is 64 feet tall
by
  sorry

end tree_height_problem_l2056_205691


namespace percentage_of_70_to_125_l2056_205651

theorem percentage_of_70_to_125 : ∃ p : ℚ, p = 70 / 125 * 100 ∧ p = 56 := by
  sorry

end percentage_of_70_to_125_l2056_205651


namespace marks_age_in_five_years_l2056_205607

theorem marks_age_in_five_years :
  ∀ (amy_age mark_age : ℕ),
    amy_age = 15 →
    mark_age = amy_age + 7 →
    mark_age + 5 = 27 :=
by sorry

end marks_age_in_five_years_l2056_205607


namespace unattainable_value_l2056_205657

theorem unattainable_value (x : ℝ) (y : ℝ) (h : x ≠ -4/3) : 
  y = (1 - x) / (3 * x + 4) → y ≠ -1/3 :=
by sorry

end unattainable_value_l2056_205657


namespace c_necessary_not_sufficient_l2056_205669

-- Define the proposition p
def p (x : ℝ) : Prop := x^2 - x < 0

-- Define the condition c
def c (x : ℝ) : Prop := -1 < x ∧ x < 1

-- Theorem stating that c is a necessary but not sufficient condition for p
theorem c_necessary_not_sufficient :
  (∀ x : ℝ, p x → c x) ∧ 
  (∃ x : ℝ, c x ∧ ¬(p x)) :=
sorry

end c_necessary_not_sufficient_l2056_205669


namespace sum_of_reciprocal_equations_l2056_205639

theorem sum_of_reciprocal_equations (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 4)
  (h2 : x⁻¹ - y⁻¹ = -6) : 
  x + y = -4/5 := by
sorry

end sum_of_reciprocal_equations_l2056_205639


namespace unmanned_supermarket_prices_l2056_205688

/-- Represents the unit price of keychains in yuan -/
def keychain_price : ℝ := 24

/-- Represents the unit price of plush toys in yuan -/
def plush_toy_price : ℝ := 36

/-- The total number of items bought -/
def total_items : ℕ := 15

/-- The total amount spent on keychains in yuan -/
def total_keychain_cost : ℝ := 240

/-- The total amount spent on plush toys in yuan -/
def total_plush_toy_cost : ℝ := 180

theorem unmanned_supermarket_prices :
  (total_keychain_cost / keychain_price + total_plush_toy_cost / plush_toy_price = total_items) ∧
  (plush_toy_price = 1.5 * keychain_price) := by
  sorry

end unmanned_supermarket_prices_l2056_205688


namespace cistern_leak_time_l2056_205666

/-- Represents the cistern problem -/
def CisternProblem (capacity : ℝ) (tapRate : ℝ) (timeWithTap : ℝ) : Prop :=
  let leakRate := capacity / timeWithTap + tapRate
  let timeWithoutTap := capacity / leakRate
  timeWithoutTap = 20

/-- Theorem stating the solution to the cistern problem -/
theorem cistern_leak_time :
  CisternProblem 480 4 24 := by sorry

end cistern_leak_time_l2056_205666


namespace total_is_sum_of_eaten_and_saved_l2056_205610

/-- The number of strawberries Micah picked in total -/
def total_strawberries : ℕ := sorry

/-- The number of strawberries Micah ate -/
def eaten_strawberries : ℕ := 6

/-- The number of strawberries Micah saved for his mom -/
def saved_strawberries : ℕ := 18

/-- Theorem stating that the total number of strawberries is the sum of eaten and saved strawberries -/
theorem total_is_sum_of_eaten_and_saved : 
  total_strawberries = eaten_strawberries + saved_strawberries :=
by sorry

end total_is_sum_of_eaten_and_saved_l2056_205610


namespace similar_triangle_longest_side_l2056_205695

/-- Given two similar triangles, where the first triangle has sides of 8, 10, and 12,
    and the second triangle has a perimeter of 150, prove that the longest side
    of the second triangle is 60. -/
theorem similar_triangle_longest_side
  (triangle1 : ℝ × ℝ × ℝ)
  (triangle2 : ℝ × ℝ × ℝ)
  (h_triangle1 : triangle1 = (8, 10, 12))
  (h_similar : ∃ (k : ℝ), triangle2 = (8*k, 10*k, 12*k))
  (h_perimeter : triangle2.1 + triangle2.2.1 + triangle2.2.2 = 150)
  : triangle2.2.2 = 60 := by
  sorry

end similar_triangle_longest_side_l2056_205695


namespace triangle_abc_properties_l2056_205697

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- Triangle inequality
  cos B = 2/5 →
  sin A * cos B - (2*c - cos A) * sin B = 0 →
  b = 1/2 ∧
  ∀ a' c', 0 < a' ∧ 0 < c' →
    a' + b + c' ≤ Real.sqrt 30 / 6 + 1/2 :=
by sorry

end triangle_abc_properties_l2056_205697


namespace imaginary_part_of_i_times_one_plus_i_l2056_205604

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_part_of_i_times_one_plus_i :
  (i * (1 + i)).im = 1 := by sorry

end imaginary_part_of_i_times_one_plus_i_l2056_205604
