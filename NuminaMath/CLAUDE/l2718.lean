import Mathlib

namespace sufficient_not_necessary_l2718_271833

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 2 → a^2 > 2*a) ∧ 
  (∃ a, a ≤ 2 ∧ a^2 > 2*a) := by
sorry

end sufficient_not_necessary_l2718_271833


namespace sum_of_digits_9ab_l2718_271829

/-- The sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A number consisting of n repetitions of a digit d in base 10 -/
def repeatedDigit (d : ℕ) (n : ℕ) : ℕ := sorry

theorem sum_of_digits_9ab : 
  let a := repeatedDigit 6 2023
  let b := repeatedDigit 4 2023
  sumOfDigits (9 * a * b) = 20225 := by sorry

end sum_of_digits_9ab_l2718_271829


namespace square_root_of_square_l2718_271851

theorem square_root_of_square (x : ℝ) (h : x = 25) : Real.sqrt (x ^ 2) = |x| := by
  sorry

end square_root_of_square_l2718_271851


namespace shirt_cost_theorem_l2718_271822

theorem shirt_cost_theorem (first_shirt_cost second_shirt_cost total_cost : ℕ) : 
  first_shirt_cost = 15 →
  first_shirt_cost = second_shirt_cost + 6 →
  total_cost = first_shirt_cost + second_shirt_cost →
  total_cost = 24 :=
by
  sorry

end shirt_cost_theorem_l2718_271822


namespace specific_journey_distance_l2718_271838

/-- A journey with two parts at different speeds -/
structure Journey where
  total_time : ℝ
  first_part_time : ℝ
  first_part_speed : ℝ
  second_part_speed : ℝ
  (first_part_time_valid : first_part_time > 0 ∧ first_part_time < total_time)
  (speeds_positive : first_part_speed > 0 ∧ second_part_speed > 0)

/-- Calculate the total distance of a journey -/
def total_distance (j : Journey) : ℝ :=
  j.first_part_speed * j.first_part_time + 
  j.second_part_speed * (j.total_time - j.first_part_time)

/-- The specific journey described in the problem -/
def specific_journey : Journey where
  total_time := 8
  first_part_time := 4
  first_part_speed := 4
  second_part_speed := 2
  first_part_time_valid := by sorry
  speeds_positive := by sorry

/-- Theorem stating that the total distance of the specific journey is 24 km -/
theorem specific_journey_distance : 
  total_distance specific_journey = 24 := by sorry

end specific_journey_distance_l2718_271838


namespace orange_count_orange_count_problem_l2718_271816

theorem orange_count (initial_apples : ℕ) (removed_oranges : ℕ) 
  (apple_percentage : ℚ) (initial_oranges : ℕ) : Prop :=
  initial_apples = 14 →
  removed_oranges = 20 →
  apple_percentage = 7/10 →
  initial_apples / (initial_apples + initial_oranges - removed_oranges) = apple_percentage →
  initial_oranges = 26

/-- The theorem states that given the conditions from the problem,
    the initial number of oranges in the box is 26. -/
theorem orange_count_problem : 
  ∃ (initial_oranges : ℕ), orange_count 14 20 (7/10) initial_oranges :=
sorry

end orange_count_orange_count_problem_l2718_271816


namespace cube_edge_length_l2718_271882

/-- A prism made up of six squares -/
structure Cube where
  edge_length : ℝ
  edge_sum : ℝ

/-- The sum of the lengths of all edges is 72 cm -/
def total_edge_length (c : Cube) : Prop :=
  c.edge_sum = 72

/-- Theorem: If the sum of the lengths of all edges is 72 cm, 
    then the length of one edge is 6 cm -/
theorem cube_edge_length (c : Cube) 
    (h : total_edge_length c) : c.edge_length = 6 := by
  sorry

end cube_edge_length_l2718_271882


namespace equations_represent_same_curve_l2718_271885

-- Define the two equations
def equation1 (x y : ℝ) : Prop := |y| = |x|
def equation2 (x y : ℝ) : Prop := y^2 = x^2

-- Theorem statement
theorem equations_represent_same_curve :
  ∀ (x y : ℝ), equation1 x y ↔ equation2 x y := by
  sorry

end equations_represent_same_curve_l2718_271885


namespace abs_diff_eq_diff_implies_leq_l2718_271865

theorem abs_diff_eq_diff_implies_leq (x y : ℝ) : |x - y| = y - x → x ≤ y := by
  sorry

end abs_diff_eq_diff_implies_leq_l2718_271865


namespace infinite_factorial_solutions_l2718_271897

theorem infinite_factorial_solutions :
  ∃ f : ℕ → ℕ × ℕ × ℕ, ∀ n : ℕ,
    let (x, y, z) := f n
    x > 1 ∧ y > 1 ∧ z > 1 ∧ Nat.factorial x * Nat.factorial y = Nat.factorial z :=
by sorry

end infinite_factorial_solutions_l2718_271897


namespace necessary_not_sufficient_l2718_271862

theorem necessary_not_sufficient (a b : ℝ) : 
  (a > b → a > b - 1) ∧ ¬(a > b - 1 → a > b) := by sorry

end necessary_not_sufficient_l2718_271862


namespace orange_boxes_weight_l2718_271846

/-- Calculates the total weight of oranges in three boxes given their capacities, fill ratios, and orange weights. -/
theorem orange_boxes_weight (capacity1 capacity2 capacity3 : ℕ)
                            (fill1 fill2 fill3 : ℚ)
                            (weight1 weight2 weight3 : ℚ) :
  capacity1 = 80 →
  capacity2 = 50 →
  capacity3 = 60 →
  fill1 = 3/4 →
  fill2 = 3/5 →
  fill3 = 2/3 →
  weight1 = 1/4 →
  weight2 = 3/10 →
  weight3 = 2/5 →
  (capacity1 * fill1 * weight1 + capacity2 * fill2 * weight2 + capacity3 * fill3 * weight3 : ℚ) = 40 := by
  sorry

#eval (80 * (3/4) * (1/4) + 50 * (3/5) * (3/10) + 60 * (2/3) * (2/5) : ℚ)

end orange_boxes_weight_l2718_271846


namespace not_always_parallel_to_plane_l2718_271802

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem not_always_parallel_to_plane
  (a b : Line) (α : Plane)
  (diff : a ≠ b)
  (h1 : subset b α)
  (h2 : parallel_lines a b) :
  ¬(∀ a b α, subset b α → parallel_lines a b → parallel_line_plane a α) :=
sorry

end not_always_parallel_to_plane_l2718_271802


namespace solve_for_q_l2718_271815

theorem solve_for_q (x y q : ℚ) 
  (h1 : (7 : ℚ) / 8 = x / 96)
  (h2 : (7 : ℚ) / 8 = (x + y) / 104)
  (h3 : (7 : ℚ) / 8 = (q - y) / 144) :
  q = 133 := by sorry

end solve_for_q_l2718_271815


namespace a_left_after_ten_days_l2718_271840

/-- The number of days it takes A to complete the work -/
def days_A : ℝ := 30

/-- The number of days it takes B to complete the work -/
def days_B : ℝ := 30

/-- The number of days B worked after A left -/
def days_B_worked : ℝ := 10

/-- The number of days C worked to finish the work -/
def days_C_worked : ℝ := 10

/-- The number of days it takes C to complete the whole work -/
def days_C : ℝ := 29.999999999999996

/-- The theorem stating that A left the work after 10 days -/
theorem a_left_after_ten_days :
  ∃ (x : ℝ),
    x > 0 ∧
    x / days_A + days_B_worked / days_B + days_C_worked / days_C = 1 ∧
    x = 10 := by
  sorry

end a_left_after_ten_days_l2718_271840


namespace radii_and_circles_regions_l2718_271854

/-- The number of regions created by radii and concentric circles inside a larger circle -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem stating that 16 radii and 10 concentric circles create 176 regions -/
theorem radii_and_circles_regions :
  num_regions 16 10 = 176 := by
  sorry

end radii_and_circles_regions_l2718_271854


namespace average_of_abc_is_three_l2718_271880

theorem average_of_abc_is_three (A B C : ℚ) 
  (eq1 : 101 * C - 202 * A = 404)
  (eq2 : 101 * B + 303 * A = 505)
  (eq3 : 101 * A + 101 * B + 101 * C = 303) :
  (A + B + C) / 3 = 3 := by
  sorry

end average_of_abc_is_three_l2718_271880


namespace system_of_equations_solution_l2718_271868

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x - y = 6) ∧ (x + 2 * y = -2) :=
by
  -- The proof goes here
  sorry

end system_of_equations_solution_l2718_271868


namespace skating_speed_ratio_l2718_271849

/-- The ratio of skating speeds between a father and son -/
theorem skating_speed_ratio (v_f v_s : ℝ) (h1 : v_f > 0) (h2 : v_s > 0) 
  (h3 : v_f > v_s) (h4 : (v_f + v_s) / (v_f - v_s) = 5) : v_f / v_s = 3/2 :=
by sorry

end skating_speed_ratio_l2718_271849


namespace sqrt_a_minus_one_is_rational_square_l2718_271800

theorem sqrt_a_minus_one_is_rational_square (a b : ℚ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + 4*a^2*b = 4*a^2 + b^4) :
  ∃ q : ℚ, Real.sqrt a - 1 = q^2 := by
  sorry

end sqrt_a_minus_one_is_rational_square_l2718_271800


namespace truck_travel_distance_l2718_271893

/-- 
Given a truck that travels:
- x miles north
- 30 miles east
- x miles north again
And ends up 50 miles from the starting point,
prove that x must equal 20.
-/
theorem truck_travel_distance (x : ℝ) : 
  (2 * x)^2 + 30^2 = 50^2 → x = 20 := by
  sorry

end truck_travel_distance_l2718_271893


namespace smallest_cube_ending_392_l2718_271836

theorem smallest_cube_ending_392 :
  ∃ (n : ℕ), n > 0 ∧ n^3 ≡ 392 [ZMOD 1000] ∧ ∀ (m : ℕ), m > 0 ∧ m^3 ≡ 392 [ZMOD 1000] → n ≤ m :=
by
  use 48
  sorry

end smallest_cube_ending_392_l2718_271836


namespace min_sum_equal_last_three_digits_l2718_271875

theorem min_sum_equal_last_three_digits (m n : ℕ) : 
  m ≥ 1 → n > m → 
  (1978^n - 1978^m) % 1000 = 0 → 
  (∀ k l : ℕ, k ≥ 1 → l > k → (1978^l - 1978^k) % 1000 = 0 → m + n ≤ k + l) → 
  m + n = 106 := by
sorry

end min_sum_equal_last_three_digits_l2718_271875


namespace lilith_cap_collection_l2718_271892

/-- Calculates the number of caps Lilith has collected after a given number of years -/
def caps_collected (years : ℕ) : ℕ :=
  let first_year_caps := 3 * 12
  let subsequent_years_caps := 5 * 12 * (years - 1)
  let christmas_caps := 40 * years
  let total_caps := first_year_caps + subsequent_years_caps + christmas_caps
  let lost_caps := 15 * years
  total_caps - lost_caps

/-- Theorem stating that Lilith has collected 401 caps after 5 years -/
theorem lilith_cap_collection : caps_collected 5 = 401 := by
  sorry

end lilith_cap_collection_l2718_271892


namespace eight_person_lineup_l2718_271831

theorem eight_person_lineup : Nat.factorial 8 = 40320 := by
  sorry

end eight_person_lineup_l2718_271831


namespace right_triangle_sin_A_l2718_271842

theorem right_triangle_sin_A (A B C : Real) (AB BC : Real) :
  -- Right triangle ABC with ∠BAC = 90°
  A + B + C = 180 →
  A = 90 →
  -- Side lengths
  AB = 15 →
  BC = 20 →
  -- Definition of sin A in a right triangle
  Real.sin A = Real.sqrt 7 / 4 := by sorry

end right_triangle_sin_A_l2718_271842


namespace zeros_product_greater_than_e_squared_l2718_271837

theorem zeros_product_greater_than_e_squared (k : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
  (Real.log x₁ = k * x₁) → (Real.log x₂ = k * x₂) →
  x₁ * x₂ > Real.exp 2 := by
sorry

end zeros_product_greater_than_e_squared_l2718_271837


namespace basketball_team_starters_l2718_271888

theorem basketball_team_starters (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) (quad_starters : ℕ) :
  total_players = 12 →
  quadruplets = 4 →
  starters = 5 →
  quad_starters = 2 →
  (Nat.choose quadruplets quad_starters) * (Nat.choose (total_players - quadruplets) (starters - quad_starters)) = 336 :=
by sorry

end basketball_team_starters_l2718_271888


namespace unique_solution_equation_l2718_271824

theorem unique_solution_equation : 
  ∃! (x y z : ℕ+), 1 + 2^(x.val) + 3^(y.val) = z.val^3 :=
by
  -- The proof would go here
  sorry

end unique_solution_equation_l2718_271824


namespace cubic_roots_sum_squares_l2718_271871

theorem cubic_roots_sum_squares (p q r : ℝ) (x₁ x₂ x₃ : ℝ) : 
  (∀ x, x^3 - p*x^2 + q*x - r = 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  x₁^2 + x₂^2 + x₃^2 = p^2 - 2*q := by
  sorry

end cubic_roots_sum_squares_l2718_271871


namespace cars_meet_time_l2718_271867

-- Define the highway length
def highway_length : ℝ := 175

-- Define the speeds of the two cars
def speed_car1 : ℝ := 25
def speed_car2 : ℝ := 45

-- Define the meeting time
def meeting_time : ℝ := 2.5

-- Theorem statement
theorem cars_meet_time :
  speed_car1 * meeting_time + speed_car2 * meeting_time = highway_length :=
by sorry


end cars_meet_time_l2718_271867


namespace mod_nine_equivalence_l2718_271896

theorem mod_nine_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] := by sorry

end mod_nine_equivalence_l2718_271896


namespace function_value_at_five_l2718_271814

theorem function_value_at_five (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 2 * x^2 - 1) : 
  f 5 = 13/3 := by
sorry

end function_value_at_five_l2718_271814


namespace intersection_circles_properties_l2718_271877

/-- Given two circles O₁ and O₂ with equations x² + y² - 2x = 0 and x² + y² + 2x - 4y = 0 respectively,
    prove that their intersection points A and B satisfy:
    1. The line AB has equation x - y = 0
    2. The perpendicular bisector of AB has equation x + y - 1 = 0 -/
theorem intersection_circles_properties (x y : ℝ) :
  let O₁ := {(x, y) | x^2 + y^2 - 2*x = 0}
  let O₂ := {(x, y) | x^2 + y^2 + 2*x - 4*y = 0}
  let A := (x₀, y₀)
  let B := (x₁, y₁)
  ∀ x₀ y₀ x₁ y₁,
    (x₀, y₀) ∈ O₁ ∧ (x₀, y₀) ∈ O₂ ∧
    (x₁, y₁) ∈ O₁ ∧ (x₁, y₁) ∈ O₂ ∧
    (x₀, y₀) ≠ (x₁, y₁) →
    (x - y = 0 ↔ ∃ t, x = (1-t)*x₀ + t*x₁ ∧ y = (1-t)*y₀ + t*y₁) ∧
    (x + y - 1 = 0 ↔ (x - 1)^2 + y^2 = (x + 1)^2 + (y - 2)^2) :=
by sorry

end intersection_circles_properties_l2718_271877


namespace distance_to_square_center_l2718_271810

-- Define the right triangle ABC
structure RightTriangle where
  a : ℝ  -- length of BC
  b : ℝ  -- length of AC
  h : 0 < a ∧ 0 < b  -- positive lengths

-- Define the square ABDE on the hypotenuse
structure SquareOnHypotenuse (t : RightTriangle) where
  center : ℝ × ℝ  -- coordinates of the center of the square

-- Theorem statement
theorem distance_to_square_center (t : RightTriangle) (s : SquareOnHypotenuse t) :
  Real.sqrt ((s.center.1 ^ 2) + (s.center.2 ^ 2)) = (t.a + t.b) / Real.sqrt 2 := by
  sorry

end distance_to_square_center_l2718_271810


namespace complex_ln_def_l2718_271804

-- Define the complex logarithm function
def complex_ln (z : ℂ) : Set ℂ :=
  {w : ℂ | ∃ k : ℤ, w = Complex.log (Complex.abs z) + Complex.I * (Complex.arg z + 2 * k * Real.pi)}

-- State the theorem
theorem complex_ln_def (z : ℂ) :
  ∀ w ∈ complex_ln z, Complex.exp w = z :=
by sorry

end complex_ln_def_l2718_271804


namespace letters_with_both_in_given_alphabet_l2718_271879

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet where
  total : ℕ
  line_no_dot : ℕ
  dot_no_line : ℕ
  all_have_dot_or_line : Bool

/-- The number of letters containing both a dot and a straight line -/
def letters_with_both (a : Alphabet) : ℕ :=
  a.total - a.line_no_dot - a.dot_no_line

/-- Theorem stating the number of letters with both dot and line in the given alphabet -/
theorem letters_with_both_in_given_alphabet :
  ∀ (a : Alphabet),
    a.total = 60 ∧
    a.line_no_dot = 36 ∧
    a.dot_no_line = 4 ∧
    a.all_have_dot_or_line = true →
    letters_with_both a = 20 := by
  sorry


end letters_with_both_in_given_alphabet_l2718_271879


namespace pairs_count_l2718_271894

/-- S(n) denotes the sum of the digits of a natural number n -/
def S (n : ℕ) : ℕ := sorry

/-- The number of pairs <m, n> satisfying the given conditions -/
def count_pairs : ℕ := sorry

theorem pairs_count :
  count_pairs = 99 ∧
  ∀ m n : ℕ,
    m < 100 →
    n < 100 →
    m > n →
    m + S n = n + 2 * S m →
    (m, n) ∈ (Finset.filter (fun p : ℕ × ℕ => 
      p.1 < 100 ∧
      p.2 < 100 ∧
      p.1 > p.2 ∧
      p.1 + S p.2 = p.2 + 2 * S p.1)
    (Finset.product (Finset.range 100) (Finset.range 100))) :=
by sorry

end pairs_count_l2718_271894


namespace f_g_one_eq_one_solution_set_eq_two_l2718_271898

-- Define the domain of x
inductive X : Type
| one : X
| two : X
| three : X

-- Define functions f and g
def f : X → ℕ
| X.one => 1
| X.two => 3
| X.three => 1

def g : X → ℕ
| X.one => 3
| X.two => 2
| X.three => 1

-- Define composition of f and g
def f_comp_g (x : X) : ℕ := f (match g x with
  | 1 => X.one
  | 2 => X.two
  | 3 => X.three
  | _ => X.one)

def g_comp_f (x : X) : ℕ := g (match f x with
  | 1 => X.one
  | 2 => X.two
  | 3 => X.three
  | _ => X.one)

theorem f_g_one_eq_one : f_comp_g X.one = 1 := by sorry

theorem solution_set_eq_two :
  (∀ x : X, f_comp_g x > g_comp_f x ↔ x = X.two) := by sorry

end f_g_one_eq_one_solution_set_eq_two_l2718_271898


namespace triangle_side_calculation_l2718_271895

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) :
  a = 6 →
  c = 4 →
  Real.sin (B / 2) = Real.sqrt 3 / 3 →
  b = 6 := by
  sorry

end triangle_side_calculation_l2718_271895


namespace log2_7_value_l2718_271817

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Given conditions
variable (m n : ℝ)
variable (h1 : lg 5 = m)
variable (h2 : lg 7 = n)

-- Theorem to prove
theorem log2_7_value : Real.log 7 / Real.log 2 = n / (1 - m) := by
  sorry

end log2_7_value_l2718_271817


namespace jills_first_bus_ride_time_l2718_271870

/-- Jill's journey to the library -/
def jills_journey (first_bus_wait : ℕ) (second_bus_ride : ℕ) (first_bus_ride : ℕ) : Prop :=
  second_bus_ride = (first_bus_wait + first_bus_ride) / 2

theorem jills_first_bus_ride_time :
  ∃ (first_bus_ride : ℕ),
    jills_journey 12 21 first_bus_ride ∧
    first_bus_ride = 30 := by
  sorry

end jills_first_bus_ride_time_l2718_271870


namespace trapezoid_circle_properties_l2718_271884

/-- Represents a trapezoid ABCD with a circle centered at P on AB and tangent to BC and AD -/
structure Trapezoid :=
  (AB CD BC AD : ℝ)
  (AP : ℝ)
  (r : ℝ)

/-- The theorem stating the properties of the trapezoid and circle -/
theorem trapezoid_circle_properties (T : Trapezoid) :
  T.AB = 105 ∧
  T.BC = 65 ∧
  T.CD = 27 ∧
  T.AD = 80 ∧
  T.AP = 175 / 3 ∧
  T.r = 35 / 6 :=
sorry

end trapezoid_circle_properties_l2718_271884


namespace meaningful_expression_range_l2718_271827

-- Define the condition for the expression to be meaningful
def is_meaningful (x : ℝ) : Prop := x > 3

-- Theorem statement
theorem meaningful_expression_range :
  ∀ x : ℝ, is_meaningful x ↔ x > 3 := by
  sorry

end meaningful_expression_range_l2718_271827


namespace smallest_divisible_by_10_11_12_13_eight_five_eight_zero_divisible_smallest_positive_integer_divisible_by_10_11_12_13_l2718_271881

theorem smallest_divisible_by_10_11_12_13 : 
  ∀ n : ℕ, n > 0 ∧ 10 ∣ n ∧ 11 ∣ n ∧ 12 ∣ n ∧ 13 ∣ n → n ≥ 8580 := by
  sorry

theorem eight_five_eight_zero_divisible :
  10 ∣ 8580 ∧ 11 ∣ 8580 ∧ 12 ∣ 8580 ∧ 13 ∣ 8580 := by
  sorry

theorem smallest_positive_integer_divisible_by_10_11_12_13 :
  ∃! n : ℕ, n > 0 ∧ 
    (∀ m : ℕ, m > 0 ∧ 10 ∣ m ∧ 11 ∣ m ∧ 12 ∣ m ∧ 13 ∣ m → n ≤ m) ∧
    10 ∣ n ∧ 11 ∣ n ∧ 12 ∣ n ∧ 13 ∣ n ∧ n = 8580 := by
  sorry

end smallest_divisible_by_10_11_12_13_eight_five_eight_zero_divisible_smallest_positive_integer_divisible_by_10_11_12_13_l2718_271881


namespace focal_radii_common_points_l2718_271874

/-- An ellipse and hyperbola sharing the same foci -/
structure EllipseHyperbola where
  a : ℝ  -- semi-major axis of the ellipse
  e : ℝ  -- semi-major axis of the hyperbola

/-- The focal radii of the common points of an ellipse and hyperbola sharing the same foci -/
def focal_radii (eh : EllipseHyperbola) : ℝ × ℝ :=
  (eh.a + eh.e, eh.a - eh.e)

/-- Theorem: The focal radii of the common points of an ellipse and hyperbola 
    sharing the same foci are a + e and a - e -/
theorem focal_radii_common_points (eh : EllipseHyperbola) :
  focal_radii eh = (eh.a + eh.e, eh.a - eh.e) := by
  sorry

end focal_radii_common_points_l2718_271874


namespace sine_curve_intersection_l2718_271808

theorem sine_curve_intersection (A a : ℝ) (h1 : A > 0) (h2 : a > 0) :
  (∃ x1 x2 x3 x4 : ℝ, 
    0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 ≤ 2 * π ∧
    A * Real.sin x1 + a = 2 ∧
    A * Real.sin x2 + a = -1 ∧
    A * Real.sin x3 + a = -1 ∧
    A * Real.sin x4 + a = 2 ∧
    (x2 - x1) = (x4 - x3) ∧
    x2 ≠ x1) →
  a = 1/2 ∧ A > 3/2 := by
sorry

end sine_curve_intersection_l2718_271808


namespace solution_set_equiv_l2718_271834

/-- The solution set of ax^2 + 2ax > 0 -/
def SolutionSet (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * a * x > 0}

/-- The proposition that 0 < a < 1 -/
def q (a : ℝ) : Prop := 0 < a ∧ a < 1

/-- The theorem stating that q is necessary and sufficient for the solution set to be ℝ -/
theorem solution_set_equiv (a : ℝ) : SolutionSet a = Set.univ ↔ q a := by sorry

end solution_set_equiv_l2718_271834


namespace base_five_product_theorem_l2718_271830

def base_five_to_decimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (5 ^ i)) 0

def decimal_to_base_five (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec convert (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else convert (m / 5) ((m % 5) :: acc)
  convert n []

def base_five_multiply (a b : List Nat) : List Nat :=
  decimal_to_base_five ((base_five_to_decimal a) * (base_five_to_decimal b))

theorem base_five_product_theorem :
  base_five_multiply [1, 3, 1] [2, 1] = [2, 2, 1, 2] := by sorry

end base_five_product_theorem_l2718_271830


namespace find_a_value_l2718_271821

def f (x a : ℝ) : ℝ := |x + 1| + |x - a|

theorem find_a_value (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, f x a ≥ 5 ↔ x ≤ -2 ∨ x > 3) : a = 2 := by
  sorry

end find_a_value_l2718_271821


namespace chess_tournament_games_l2718_271823

/-- The number of games in a chess tournament where each player plays twice with every other player -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: In a chess tournament with 18 players, where each player plays twice with every other player, the total number of games played is 612 -/
theorem chess_tournament_games :
  tournament_games 18 * 2 = 612 := by
  sorry

end chess_tournament_games_l2718_271823


namespace twelve_boys_handshakes_l2718_271803

/-- The number of handshakes when n boys each shake hands exactly once with every other boy -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 12 boys, where each boy shakes hands exactly once with each of the others, 
    the total number of handshakes is 66 -/
theorem twelve_boys_handshakes : handshakes 12 = 66 := by
  sorry

end twelve_boys_handshakes_l2718_271803


namespace sum_of_square_roots_bound_l2718_271819

theorem sum_of_square_roots_bound (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_one : x + y + z = 1) : 
  Real.sqrt (7 * x + 3) + Real.sqrt (7 * y + 3) + Real.sqrt (7 * z + 3) ≤ 7 := by
  sorry

end sum_of_square_roots_bound_l2718_271819


namespace min_distance_circle_line_l2718_271878

/-- The minimum distance between a circle and a line --/
theorem min_distance_circle_line :
  let circle := {p : ℝ × ℝ | (p.1 + 2)^2 + p.2^2 = 4}
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  (∀ p ∈ circle, ∀ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ 3 * Real.sqrt 2 - 2) ∧
  (∃ p ∈ circle, ∃ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 3 * Real.sqrt 2 - 2) :=
by sorry

end min_distance_circle_line_l2718_271878


namespace unique_square_divisible_by_12_l2718_271811

theorem unique_square_divisible_by_12 :
  ∃! x : ℕ, (∃ y : ℕ, x = y^2) ∧ 
            (12 ∣ x) ∧ 
            100 ≤ x ∧ x ≤ 200 :=
by
  -- The proof goes here
  sorry

end unique_square_divisible_by_12_l2718_271811


namespace entrepreneur_raised_12000_l2718_271857

/-- Represents the crowdfunding levels and backers for an entrepreneur's business effort -/
structure CrowdfundingCampaign where
  highest_level : ℕ
  second_level : ℕ
  lowest_level : ℕ
  highest_backers : ℕ
  second_backers : ℕ
  lowest_backers : ℕ

/-- Calculates the total amount raised in a crowdfunding campaign -/
def total_raised (campaign : CrowdfundingCampaign) : ℕ :=
  campaign.highest_level * campaign.highest_backers +
  campaign.second_level * campaign.second_backers +
  campaign.lowest_level * campaign.lowest_backers

/-- Theorem stating that the entrepreneur raised $12000 -/
theorem entrepreneur_raised_12000 :
  ∀ (campaign : CrowdfundingCampaign),
  campaign.highest_level = 5000 ∧
  campaign.second_level = campaign.highest_level / 10 ∧
  campaign.lowest_level = campaign.second_level / 10 ∧
  campaign.highest_backers = 2 ∧
  campaign.second_backers = 3 ∧
  campaign.lowest_backers = 10 →
  total_raised campaign = 12000 :=
sorry

end entrepreneur_raised_12000_l2718_271857


namespace v_2003_equals_5_l2718_271839

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 2
| 4 => 1
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 5
| n + 1 => g (v n)

-- Theorem to prove
theorem v_2003_equals_5 : v 2003 = 5 := by
  sorry


end v_2003_equals_5_l2718_271839


namespace constant_b_value_l2718_271876

theorem constant_b_value (x y : ℝ) (b : ℝ) 
  (h1 : (7 * x + b * y) / (x - 2 * y) = 29)
  (h2 : x / (2 * y) = 3 / 2) : 
  b = 8 := by
  sorry

end constant_b_value_l2718_271876


namespace goldfish_count_correct_l2718_271889

/-- The number of Goldfish Layla has -/
def num_goldfish : ℕ := 2

/-- The total amount of food Layla gives to all her fish -/
def total_food : ℕ := 12

/-- The number of Swordtails Layla has -/
def num_swordtails : ℕ := 3

/-- The amount of food each Swordtail gets -/
def food_per_swordtail : ℕ := 2

/-- The number of Guppies Layla has -/
def num_guppies : ℕ := 8

/-- The amount of food each Guppy gets -/
def food_per_guppy : ℚ := 1/2

/-- The amount of food each Goldfish gets -/
def food_per_goldfish : ℕ := 1

/-- Theorem stating that the number of Goldfish is correct given the conditions -/
theorem goldfish_count_correct : 
  total_food = num_swordtails * food_per_swordtail + 
               num_guppies * food_per_guppy + 
               num_goldfish * food_per_goldfish :=
by sorry

end goldfish_count_correct_l2718_271889


namespace probability_of_one_each_l2718_271845

def shirts : ℕ := 6
def shorts : ℕ := 8
def socks : ℕ := 7
def hats : ℕ := 3

def total_items : ℕ := shirts + shorts + socks + hats

theorem probability_of_one_each : 
  (shirts.choose 1 * shorts.choose 1 * socks.choose 1 * hats.choose 1) / total_items.choose 4 = 72 / 91 := by
  sorry

end probability_of_one_each_l2718_271845


namespace one_third_of_270_l2718_271812

theorem one_third_of_270 : (1 / 3 : ℚ) * 270 = 90 := by
  sorry

end one_third_of_270_l2718_271812


namespace problem_1_problem_2_problem_3_l2718_271872

-- Given identity
axiom identity (a b : ℝ) : (a - b) * (a + b) = a^2 - b^2

-- Theorem 1
theorem problem_1 : (2 - 1) * (2 + 1) = 3 := by sorry

-- Theorem 2
theorem problem_2 : (2 + 1) * (2^2 + 1) = 15 := by sorry

-- Helper function to generate the product series
def product_series (n : ℕ) : ℝ :=
  if n = 0 then 2 + 1
  else (2^(2^n) + 1) * product_series (n-1)

-- Theorem 3
theorem problem_3 : product_series 5 = 2^64 - 1 := by sorry

end problem_1_problem_2_problem_3_l2718_271872


namespace oliver_final_amount_l2718_271828

def oliver_money_left (initial_amount savings frisbee_cost puzzle_cost birthday_gift : ℕ) : ℕ :=
  initial_amount + savings - frisbee_cost - puzzle_cost + birthday_gift

theorem oliver_final_amount :
  oliver_money_left 9 5 4 3 8 = 15 := by
  sorry

end oliver_final_amount_l2718_271828


namespace f_eq_g_l2718_271832

-- Define the two functions
def f (x : ℝ) : ℝ := x^2 - 4*x + 6
def g (x : ℝ) : ℝ := (x - 2)^2 + 2

-- Theorem stating that f and g are equal for all real x
theorem f_eq_g : ∀ x : ℝ, f x = g x := by
  sorry

end f_eq_g_l2718_271832


namespace f_difference_l2718_271848

/-- Given a function f defined as f(n) = 1/3 * n * (n+1) * (n+2),
    prove that f(r) - f(r-1) = r * (r+1) for any real number r. -/
theorem f_difference (r : ℝ) : 
  let f (n : ℝ) := (1/3) * n * (n+1) * (n+2)
  f r - f (r-1) = r * (r+1) := by
sorry

end f_difference_l2718_271848


namespace sin_15_deg_squared_value_l2718_271853

theorem sin_15_deg_squared_value : 
  7/16 - 7/8 * (Real.sin (15 * π / 180))^2 = 7 * Real.sqrt 3 / 32 := by
  sorry

end sin_15_deg_squared_value_l2718_271853


namespace complex_point_in_third_quadrant_l2718_271843

/-- Given a complex number z = 1 + i, prove that the real and imaginary parts of (5/z^2) - z are both negative -/
theorem complex_point_in_third_quadrant (z : ℂ) (h : z = 1 + Complex.I) :
  (Complex.re ((5 / z^2) - z) < 0) ∧ (Complex.im ((5 / z^2) - z) < 0) := by
  sorry

end complex_point_in_third_quadrant_l2718_271843


namespace no_three_digit_even_sum_27_l2718_271820

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is even -/
def is_even (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has three digits -/
def is_three_digit (n : ℕ) : Prop := sorry

theorem no_three_digit_even_sum_27 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ is_even n ∧ digit_sum n = 27 := by sorry

end no_three_digit_even_sum_27_l2718_271820


namespace rachel_total_books_l2718_271805

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves with mystery books -/
def mystery_shelves : ℕ := 6

/-- The number of shelves with picture books -/
def picture_shelves : ℕ := 2

/-- The total number of books Rachel has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem rachel_total_books : total_books = 72 := by
  sorry

end rachel_total_books_l2718_271805


namespace attendance_difference_l2718_271899

def football_game_attendance (saturday_attendance : ℕ) : Prop :=
  let monday_attendance : ℕ := saturday_attendance - saturday_attendance / 4
  let wednesday_attendance : ℕ := monday_attendance + monday_attendance / 2
  let friday_attendance : ℕ := saturday_attendance + monday_attendance
  let thursday_attendance : ℕ := 45
  let sunday_attendance : ℕ := saturday_attendance - saturday_attendance * 15 / 100
  let total_attendance : ℕ := saturday_attendance + monday_attendance + wednesday_attendance + 
                               thursday_attendance + friday_attendance + sunday_attendance
  let expected_attendance : ℕ := 350
  total_attendance - expected_attendance = 133

theorem attendance_difference : 
  football_game_attendance 80 :=
sorry

end attendance_difference_l2718_271899


namespace intersection_complement_eq_l2718_271866

def M : Set ℝ := {-1, 0, 1, 3}
def N : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement_eq : M ∩ (Set.univ \ N) = {0, 1} := by
  sorry

end intersection_complement_eq_l2718_271866


namespace enclosure_blocks_count_l2718_271852

/-- Calculates the number of cubical blocks used to create a cuboidal enclosure --/
def cubicalBlocksCount (length width height thickness : ℕ) : ℕ :=
  length * width * height - (length - 2 * thickness) * (width - 2 * thickness) * (height - thickness)

/-- Theorem stating that the number of cubical blocks for the given dimensions is 372 --/
theorem enclosure_blocks_count :
  cubicalBlocksCount 15 8 7 1 = 372 := by
  sorry

end enclosure_blocks_count_l2718_271852


namespace f2_form_l2718_271890

/-- A quadratic function with coefficients a, b, and c. -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The reflection of a function about the y-axis. -/
def reflect_y (g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ g (-x)

/-- The reflection of a function about the line y = 1. -/
def reflect_y_eq_1 (g : ℝ → ℝ) : ℝ → ℝ := λ x ↦ 2 - g x

/-- Theorem stating the form of f2 after two reflections of f. -/
theorem f2_form (a b c : ℝ) (ha : a ≠ 0) :
  let f1 := reflect_y (f a b c)
  let f2 := reflect_y_eq_1 f1
  ∀ x, f2 x = -a * x^2 + b * x + (2 - c) :=
sorry

end f2_form_l2718_271890


namespace adult_dog_cost_l2718_271859

/-- The cost to prepare animals for adoption -/
structure AdoptionCost where
  cat : ℕ → ℕ     -- Cost for cats
  dog : ℕ → ℕ     -- Cost for adult dogs
  puppy : ℕ → ℕ   -- Cost for puppies

/-- The theorem stating the cost for each adult dog -/
theorem adult_dog_cost (c : AdoptionCost) 
  (h1 : c.cat 1 = 50)
  (h2 : c.puppy 1 = 150)
  (h3 : c.cat 2 + c.dog 3 + c.puppy 2 = 700) :
  c.dog 1 = 100 := by
  sorry


end adult_dog_cost_l2718_271859


namespace inverse_65_mod_66_l2718_271861

theorem inverse_65_mod_66 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 65 ∧ (65 * x) % 66 = 1 :=
by
  use 65
  sorry

end inverse_65_mod_66_l2718_271861


namespace function_range_l2718_271856

/-- Given a function f(x) = x^2 + ax + 3 - a, if f(x) ≥ 0 for all x in [-2, 2],
    then a is in the range [-7, 2]. -/
theorem function_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + a*x + 3 - a ≥ 0) → 
  a ∈ Set.Icc (-7 : ℝ) 2 := by
  sorry

end function_range_l2718_271856


namespace candy_spent_is_10_l2718_271855

-- Define the total amount spent
def total_spent : ℝ := 150

-- Define the fractions spent on each category
def fruits_veg_fraction : ℚ := 1/2
def meat_fraction : ℚ := 1/3
def bakery_fraction : ℚ := 1/10

-- Define the theorem
theorem candy_spent_is_10 :
  let remaining_fraction : ℚ := 1 - (fruits_veg_fraction + meat_fraction + bakery_fraction)
  (remaining_fraction : ℝ) * total_spent = 10 := by sorry

end candy_spent_is_10_l2718_271855


namespace range_of_t_range_of_t_lower_bound_l2718_271886

def A : Set ℝ := {x | -3 < x ∧ x < 7}
def B (t : ℝ) : Set ℝ := {x | t + 1 < x ∧ x < 2*t - 1}

theorem range_of_t (t : ℝ) : B t ⊆ A → t ≤ 4 :=
  sorry

theorem range_of_t_lower_bound : ∀ ε > 0, ∃ t : ℝ, t ≤ 4 - ε ∧ B t ⊆ A :=
  sorry

end range_of_t_range_of_t_lower_bound_l2718_271886


namespace absolute_value_inequality_l2718_271891

theorem absolute_value_inequality (x : ℝ) :
  abs (x - 2) + abs (x + 3) < 8 ↔ -9/2 < x ∧ x < 7/2 := by
  sorry

end absolute_value_inequality_l2718_271891


namespace x_squared_minus_two_is_quadratic_l2718_271818

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² - 2 = 0 is a quadratic equation -/
theorem x_squared_minus_two_is_quadratic :
  is_quadratic_equation (λ x : ℝ ↦ x^2 - 2) :=
by
  sorry


end x_squared_minus_two_is_quadratic_l2718_271818


namespace number_of_boys_l2718_271813

theorem number_of_boys (girls : ℕ) (groups : ℕ) (members_per_group : ℕ) 
  (h1 : girls = 12)
  (h2 : groups = 7)
  (h3 : members_per_group = 3) :
  groups * members_per_group - girls = 9 := by
sorry

end number_of_boys_l2718_271813


namespace paths_in_10x5_grid_avoiding_point_l2718_271847

/-- The number of paths in a grid that avoid a specific point -/
def grid_paths_avoiding_point (m n a b c d : ℕ) : ℕ :=
  Nat.choose (m + n) n - Nat.choose (a + b) b * Nat.choose ((m - a) + (n - b)) (n - b)

/-- Theorem stating the number of paths in a 10x5 grid from (0,0) to (10,5) avoiding (5,3) -/
theorem paths_in_10x5_grid_avoiding_point : 
  grid_paths_avoiding_point 10 5 5 3 5 2 = 1827 := by
  sorry

end paths_in_10x5_grid_avoiding_point_l2718_271847


namespace pursuer_catches_target_l2718_271887

/-- Represents a point on an infinite straight line --/
structure Point where
  position : ℝ

/-- Represents a moving object on the line --/
structure MovingObject where
  initialPosition : Point
  speed : ℝ
  direction : Bool  -- True for positive direction, False for negative

/-- The pursuer (police car) --/
def pursuer : MovingObject :=
  { initialPosition := { position := 0 },
    speed := 1,
    direction := true }

/-- The target (stolen car) --/
def target : MovingObject :=
  { initialPosition := { position := 0 },  -- Initial position unknown
    speed := 0.9,
    direction := true }  -- Direction unknown

/-- Theorem stating that the pursuer will eventually catch the target --/
theorem pursuer_catches_target :
  ∃ (t : ℝ), t > 0 ∧ 
  (pursuer.initialPosition.position + t * pursuer.speed = 
   target.initialPosition.position + t * target.speed ∨
   pursuer.initialPosition.position - t * pursuer.speed = 
   target.initialPosition.position - t * target.speed) :=
sorry

end pursuer_catches_target_l2718_271887


namespace yujeong_drank_most_l2718_271873

/-- Represents the amount of water drunk by each person in liters. -/
structure WaterConsumption where
  eunji : ℚ
  yujeong : ℚ
  yuna : ℚ

/-- Determines who drank the most water given the water consumption of three people. -/
def who_drank_most (consumption : WaterConsumption) : String :=
  if consumption.yujeong > consumption.eunji ∧ consumption.yujeong > consumption.yuna then
    "Yujeong"
  else if consumption.eunji > consumption.yujeong ∧ consumption.eunji > consumption.yuna then
    "Eunji"
  else
    "Yuna"

/-- Theorem stating that Yujeong drank the most water given the specific amounts. -/
theorem yujeong_drank_most :
  who_drank_most ⟨(1/2), (7/10), (6/10)⟩ = "Yujeong" := by
  sorry

#eval who_drank_most ⟨(1/2), (7/10), (6/10)⟩

end yujeong_drank_most_l2718_271873


namespace minimum_shoeing_time_l2718_271883

theorem minimum_shoeing_time 
  (num_blacksmiths : ℕ) 
  (num_horses : ℕ) 
  (time_per_horseshoe : ℕ) 
  (horseshoes_per_horse : ℕ) : 
  num_blacksmiths = 48 →
  num_horses = 60 →
  time_per_horseshoe = 5 →
  horseshoes_per_horse = 4 →
  (num_horses * horseshoes_per_horse * time_per_horseshoe) / num_blacksmiths = 25 :=
by sorry

end minimum_shoeing_time_l2718_271883


namespace power_equality_l2718_271835

theorem power_equality (x y : ℕ) :
  2 * (3^8)^2 * (2^3)^2 * 3 = 2^x * 3^y → x = 7 ∧ y = 17 := by
  sorry

end power_equality_l2718_271835


namespace lcm_factor_problem_l2718_271863

theorem lcm_factor_problem (A B : ℕ+) (X : ℕ) : 
  A = 225 →
  Nat.gcd A B = 15 →
  Nat.lcm A B = 15 * X →
  X = 15 := by
sorry

end lcm_factor_problem_l2718_271863


namespace function_simplification_l2718_271850

theorem function_simplification (x : ℝ) : 
  Real.sqrt (4 * Real.sin x ^ 4 - 2 * Real.cos (2 * x) + 3) + 
  Real.sqrt (4 * Real.cos x ^ 4 + 2 * Real.cos (2 * x) + 3) = 4 := by
sorry

end function_simplification_l2718_271850


namespace smallest_bound_for_cubic_inequality_l2718_271806

theorem smallest_bound_for_cubic_inequality :
  ∃ (M : ℝ), (∀ (a b c : ℝ),
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2) ∧
  (∀ (M' : ℝ), (∀ (a b c : ℝ),
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M' * (a^2 + b^2 + c^2)^2) →
    M ≤ M') ∧
  M = (9 * Real.sqrt 2) / 32 :=
by sorry

end smallest_bound_for_cubic_inequality_l2718_271806


namespace geometric_sequence_a4_l2718_271844

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: In a geometric sequence where a_2 = 4 and a_6 = 16, a_4 = 8 -/
theorem geometric_sequence_a4 (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a) 
    (h_a2 : a 2 = 4) 
    (h_a6 : a 6 = 16) : 
  a 4 = 8 := by
  sorry


end geometric_sequence_a4_l2718_271844


namespace multiply_mixed_number_l2718_271864

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by sorry

end multiply_mixed_number_l2718_271864


namespace parking_theorem_l2718_271860

/-- The number of parking spaces -/
def total_spaces : ℕ := 7

/-- The number of cars to be parked -/
def num_cars : ℕ := 3

/-- The number of spaces that must remain empty and connected -/
def empty_spaces : ℕ := 4

/-- The number of possible positions for the block of empty spaces -/
def empty_block_positions : ℕ := total_spaces - empty_spaces + 1

/-- The number of distinct parking arrangements -/
def parking_arrangements : ℕ := empty_block_positions * (Nat.factorial num_cars)

theorem parking_theorem : parking_arrangements = 24 := by
  sorry

end parking_theorem_l2718_271860


namespace number_problem_l2718_271825

theorem number_problem (n : ℝ) : (40 / 100) * (3 / 5) * n = 36 → n = 150 := by
  sorry

end number_problem_l2718_271825


namespace circle_area_difference_l2718_271807

theorem circle_area_difference (π : ℝ) (h_π : π > 0) : 
  let R := 18 / π  -- Radius of larger circle
  let r := R / 2   -- Radius of smaller circle
  (π * R^2 - π * r^2) = 243 / π := by
sorry

end circle_area_difference_l2718_271807


namespace carl_kevin_historical_difference_l2718_271869

/-- A stamp collector's collection --/
structure StampCollection where
  total : ℕ
  international : ℕ
  historical : ℕ
  animal : ℕ

/-- Carl's stamp collection --/
def carl : StampCollection :=
  { total := 125
  , international := 45
  , historical := 60
  , animal := 20 }

/-- Kevin's stamp collection --/
def kevin : StampCollection :=
  { total := 95
  , international := 30
  , historical := 50
  , animal := 15 }

/-- The difference in historical stamps between two collections --/
def historicalStampDifference (c1 c2 : StampCollection) : ℕ :=
  c1.historical - c2.historical

/-- Theorem stating the difference in historical stamps between Carl and Kevin --/
theorem carl_kevin_historical_difference :
  historicalStampDifference carl kevin = 10 := by
  sorry

end carl_kevin_historical_difference_l2718_271869


namespace power_of_negative_cube_l2718_271858

theorem power_of_negative_cube (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end power_of_negative_cube_l2718_271858


namespace divide_into_three_unequal_groups_divide_into_three_equal_groups_divide_among_three_people_l2718_271809

-- Define the number of books
def n : ℕ := 6

-- Theorem for the first question
theorem divide_into_three_unequal_groups :
  (n.choose 1) * ((n - 1).choose 2) * ((n - 3).choose 3) = 60 := by sorry

-- Theorem for the second question
theorem divide_into_three_equal_groups :
  (n.choose 2 * (n - 2).choose 2 * (n - 4).choose 2) / 6 = 15 := by sorry

-- Theorem for the third question
theorem divide_among_three_people :
  n.choose 2 * (n - 2).choose 2 * (n - 4).choose 2 = 90 := by sorry

end divide_into_three_unequal_groups_divide_into_three_equal_groups_divide_among_three_people_l2718_271809


namespace g_value_at_4_l2718_271801

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 - 2*x + 3

-- Define the properties of g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d) ∧  -- g is a cubic polynomial
  (g 0 = 3) ∧  -- g(0) = 3
  (∀ r : ℝ, f r = 0 → ∃ s : ℝ, g s = 0 ∧ s = r^2)  -- roots of g are squares of roots of f

-- State the theorem
theorem g_value_at_4 (g : ℝ → ℝ) (h : g_properties g) : g 4 = -75 := by
  sorry

end g_value_at_4_l2718_271801


namespace inequality_proof_l2718_271841

theorem inequality_proof (a b c : ℝ) : 
  (a + b) * (a + b - 2 * c) + (b + c) * (b + c - 2 * a) + (c + a) * (c + a - 2 * b) ≥ 0 := by
  sorry

end inequality_proof_l2718_271841


namespace number_of_cows_farm_cows_l2718_271826

/-- The number of cows in a farm given their husk consumption -/
theorem number_of_cows (total_bags : ℕ) (total_days : ℕ) (cow_days : ℕ) : ℕ :=
  total_bags * cow_days / total_days

/-- Proof that there are 26 cows in the farm -/
theorem farm_cows : number_of_cows 26 26 26 = 26 := by
  sorry

end number_of_cows_farm_cows_l2718_271826
