import Mathlib

namespace min_width_for_rectangular_area_l2950_295014

theorem min_width_for_rectangular_area :
  ∀ w : ℝ,
  w > 0 →
  w * (w + 18) ≥ 150 →
  (∀ x : ℝ, x > 0 ∧ x * (x + 18) ≥ 150 → x ≥ w) →
  w = 6 :=
by sorry

end min_width_for_rectangular_area_l2950_295014


namespace quadratic_equation_roots_l2950_295041

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + k = 0 ∧ x = 4) → 
  (∃ y : ℝ, y^2 - 3*y + k = 0 ∧ y = -1) ∧ k = -4 :=
by sorry

end quadratic_equation_roots_l2950_295041


namespace same_gender_probability_same_school_probability_l2950_295095

-- Define the schools and their teacher compositions
def school_A : Nat := 3
def school_A_males : Nat := 2
def school_A_females : Nat := 1

def school_B : Nat := 3
def school_B_males : Nat := 1
def school_B_females : Nat := 2

def total_teachers : Nat := school_A + school_B

-- Theorem for the first question
theorem same_gender_probability :
  (school_A_males * school_B_males + school_A_females * school_B_females) /
  (school_A * school_B) = 4 / 9 :=
by sorry

-- Theorem for the second question
theorem same_school_probability :
  (school_A * (school_A - 1) / 2 + school_B * (school_B - 1) / 2) /
  (total_teachers * (total_teachers - 1) / 2) = 2 / 5 :=
by sorry

end same_gender_probability_same_school_probability_l2950_295095


namespace instances_in_one_hour_l2950_295093

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The interval in seconds at which the device records data -/
def recording_interval : ℕ := 5

/-- Proves that the number of 5-second intervals in one hour is equal to 720 -/
theorem instances_in_one_hour :
  (seconds_per_minute * minutes_per_hour) / recording_interval = 720 := by
  sorry

end instances_in_one_hour_l2950_295093


namespace dolls_in_small_box_l2950_295001

theorem dolls_in_small_box :
  let big_box_count : ℕ := 5
  let big_box_dolls : ℕ := 7
  let small_box_count : ℕ := 9
  let total_dolls : ℕ := 71
  let small_box_dolls : ℕ := (total_dolls - big_box_count * big_box_dolls) / small_box_count
  small_box_dolls = 4 := by sorry

end dolls_in_small_box_l2950_295001


namespace xy_geq_ac_plus_bd_l2950_295064

theorem xy_geq_ac_plus_bd (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (hx : x = Real.sqrt (a^2 + b^2)) (hy : y = Real.sqrt (c^2 + d^2)) : x * y ≥ a * c + b * d :=
by sorry

end xy_geq_ac_plus_bd_l2950_295064


namespace container_water_percentage_l2950_295056

theorem container_water_percentage (capacity : ℝ) (added_water : ℝ) (final_fraction : ℝ) 
  (h1 : capacity = 120)
  (h2 : added_water = 54)
  (h3 : final_fraction = 3/4) :
  let initial_percentage := (final_fraction * capacity - added_water) / capacity * 100
  initial_percentage = 30 := by
sorry

end container_water_percentage_l2950_295056


namespace inscribed_circle_rectangle_area_l2950_295083

theorem inscribed_circle_rectangle_area :
  ∀ (r : ℝ) (l w : ℝ),
    r = 7 →
    l / w = 3 →
    w = 2 * r →
    l * w = 588 :=
by
  sorry

end inscribed_circle_rectangle_area_l2950_295083


namespace trig_identity_l2950_295025

theorem trig_identity : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) + 
  Real.sqrt 3 * Real.sin (10 * π / 180) * Real.tan (70 * π / 180) - 
  2 * Real.sin (50 * π / 180) = 2 := by
  sorry

end trig_identity_l2950_295025


namespace kite_plot_area_l2950_295011

/-- The scale of the map in miles per inch -/
def scale : ℚ := 200 / 2

/-- The length of the first diagonal on the map in inches -/
def diagonal1_map : ℚ := 2

/-- The length of the second diagonal on the map in inches -/
def diagonal2_map : ℚ := 10

/-- The area of a kite given its diagonals -/
def kite_area (d1 d2 : ℚ) : ℚ := (1 / 2) * d1 * d2

/-- The theorem stating that the area of the kite-shaped plot is 100,000 square miles -/
theorem kite_plot_area : 
  kite_area (diagonal1_map * scale) (diagonal2_map * scale) = 100000 := by
  sorry

end kite_plot_area_l2950_295011


namespace james_lego_collection_l2950_295008

/-- Represents the number of Legos in James' collection -/
def initial_collection : ℕ := sorry

/-- Represents the number of Legos James uses for his castle -/
def used_legos : ℕ := sorry

/-- Represents the number of Legos put back in the box -/
def legos_in_box : ℕ := 245

/-- Represents the number of missing Legos -/
def missing_legos : ℕ := 5

theorem james_lego_collection :
  (initial_collection = 500) ∧
  (used_legos = initial_collection / 2) ∧
  (legos_in_box + missing_legos = initial_collection - used_legos) :=
sorry

end james_lego_collection_l2950_295008


namespace complementary_angle_adjustment_l2950_295048

theorem complementary_angle_adjustment (a b : ℝ) (h1 : a + b = 90) (h2 : a / b = 1 / 2) :
  let a' := a * 1.2
  let b' := 90 - a'
  (b - b') / b = 0.1 := by sorry

end complementary_angle_adjustment_l2950_295048


namespace walnut_trees_in_park_l2950_295098

theorem walnut_trees_in_park (current_trees : ℕ) : 
  (current_trees + 44 = 77) → current_trees = 33 := by
  sorry

end walnut_trees_in_park_l2950_295098


namespace mary_total_spending_l2950_295081

-- Define the amounts spent on each item
def shirt_cost : ℚ := 13.04
def jacket_cost : ℚ := 12.27

-- Define the total cost
def total_cost : ℚ := shirt_cost + jacket_cost

-- Theorem to prove
theorem mary_total_spending : total_cost = 25.31 := by
  sorry

end mary_total_spending_l2950_295081


namespace square_sum_from_difference_and_product_l2950_295059

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 49) 
  (h2 : x * y = -12) : 
  x^2 + y^2 = 25 := by
sorry

end square_sum_from_difference_and_product_l2950_295059


namespace set_intersection_and_union_l2950_295051

def A (a : ℝ) : Set ℝ := {2, 3, a^2 + 4*a + 2}

def B (a : ℝ) : Set ℝ := {0, 7, 2 - a, a^2 + 4*a - 2}

theorem set_intersection_and_union (a : ℝ) :
  A a ∩ B a = {3, 7} → a = 1 ∧ A a ∪ B a = {0, 1, 2, 3, 7} := by
  sorry

end set_intersection_and_union_l2950_295051


namespace domain_of_f_l2950_295094

noncomputable def f (x : ℝ) := Real.log (2 * (Real.cos x)^2 - 1)

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

theorem domain_of_f : domain f = {x : ℝ | ∃ k : ℤ, k * Real.pi - Real.pi / 4 < x ∧ x < k * Real.pi + Real.pi / 4} := by sorry

end domain_of_f_l2950_295094


namespace reflection_direction_vector_l2950_295035

/-- Given a particle moving in a plane along direction u = (1,2) and reflecting off a line l
    to move in direction v = (-2,1) according to the optical principle,
    one possible direction vector of line l is ω = (1,-3). -/
theorem reflection_direction_vector :
  let u : ℝ × ℝ := (1, 2)
  let v : ℝ × ℝ := (-2, 1)
  ∃ ω : ℝ × ℝ, ω = (1, -3) ∧
    (∀ k : ℝ, (k - 2) / (1 + 2*k) = (-1/2 - k) / (1 - 1/2*k) → k = -3) ∧
    (∀ θ₁ θ₂ : ℝ, θ₁ = θ₂ → 
      (u.2 / u.1 - ω.2 / ω.1) / (1 + (u.2 / u.1) * (ω.2 / ω.1)) =
      (v.2 / v.1 - ω.2 / ω.1) / (1 + (v.2 / v.1) * (ω.2 / ω.1))) :=
by sorry

end reflection_direction_vector_l2950_295035


namespace problem_1_l2950_295003

theorem problem_1 : (-1)^3 + |2 - Real.sqrt 5| + (Real.pi / 2 - 1.57)^0 + Real.sqrt 20 = 3 * Real.sqrt 5 - 2 := by
  sorry

end problem_1_l2950_295003


namespace parking_lot_spaces_l2950_295082

theorem parking_lot_spaces (total_spaces : ℕ) (full_ratio compact_ratio : ℕ) 
  (h1 : total_spaces = 450)
  (h2 : full_ratio = 11)
  (h3 : compact_ratio = 4) :
  (total_spaces * full_ratio) / (full_ratio + compact_ratio) = 330 := by
  sorry

end parking_lot_spaces_l2950_295082


namespace quadratic_inequality_solution_l2950_295088

-- Define the quadratic function
def f (a b x : ℝ) := a * x^2 + x + b

-- Define the solution set of the original inequality
def solution_set (a b : ℝ) : Set ℝ := {x | x < -2 ∨ x > 1}

-- Define the new quadratic function
def g (c x : ℝ) := x^2 - (c - 2) * x - 2 * c

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ a b : ℝ, (∀ x : ℝ, f a b x > 0 ↔ x ∈ solution_set a b) →
  (a = 1 ∧ b = -2) ∧
  (∀ c : ℝ, 
    (c = -2 → {x : ℝ | g c x < 0} = ∅) ∧
    (c > -2 → {x : ℝ | g c x < 0} = Set.Ioo (-2) c) ∧
    (c < -2 → {x : ℝ | g c x < 0} = Set.Ioo c (-2))) :=
by sorry

end quadratic_inequality_solution_l2950_295088


namespace P_identity_l2950_295037

def P (n : ℕ) : ℕ := (n + 1).factorial / n.factorial

def oddProduct (n : ℕ) : ℕ :=
  Finset.prod (Finset.range n) (fun i => 2 * i + 1)

theorem P_identity (n : ℕ) : P n = 2^n * oddProduct n := by
  sorry

end P_identity_l2950_295037


namespace max_value_a_l2950_295043

theorem max_value_a (a b c d : ℤ) 
  (h1 : a < 2 * b) 
  (h2 : b < 3 * c) 
  (h3 : c < 4 * d) 
  (h4 : d < 100) : 
  a ≤ 2367 ∧ ∃ (a' b' c' d' : ℤ), a' = 2367 ∧ a' < 2 * b' ∧ b' < 3 * c' ∧ c' < 4 * d' ∧ d' < 100 :=
by
  sorry

end max_value_a_l2950_295043


namespace nested_fraction_evaluation_l2950_295028

theorem nested_fraction_evaluation :
  1 / (1 + 1 / (2 + 1 / (3 + 1 / 4))) = 30 / 43 := by
  sorry

end nested_fraction_evaluation_l2950_295028


namespace subsets_and_proper_subsets_of_S_l2950_295019

def S : Set ℕ := {0, 1, 2}

theorem subsets_and_proper_subsets_of_S :
  (Finset.powerset {0, 1, 2} = {∅, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}}) ∧
  (Finset.powerset {0, 1, 2} \ {{0, 1, 2}} = {∅, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}}) := by
  sorry

end subsets_and_proper_subsets_of_S_l2950_295019


namespace cathys_money_proof_l2950_295031

/-- Calculates the total amount of money Cathy has after receiving contributions from her parents. -/
def cathys_total_money (initial : ℕ) (dads_contribution : ℕ) : ℕ :=
  initial + dads_contribution + 2 * dads_contribution

/-- Proves that Cathy's total money is 87 given the initial conditions. -/
theorem cathys_money_proof :
  cathys_total_money 12 25 = 87 := by
  sorry

#eval cathys_total_money 12 25

end cathys_money_proof_l2950_295031


namespace car_speed_problem_l2950_295033

/-- Proves that if a car traveling at 600 km/h takes 2 seconds longer to cover 1 km 
    than it would at speed v km/h, then v = 900 km/h. -/
theorem car_speed_problem (v : ℝ) : 
  (1 / (600 / 3600) - 1 / (v / 3600) = 2) → v = 900 := by
  sorry

#check car_speed_problem

end car_speed_problem_l2950_295033


namespace power_function_not_in_second_quadrant_l2950_295047

def f (x : ℝ) : ℝ := x

theorem power_function_not_in_second_quadrant :
  (∀ x : ℝ, x < 0 → f x ≤ 0) ∧
  (∀ x : ℝ, f x = x) :=
sorry

end power_function_not_in_second_quadrant_l2950_295047


namespace cube_coloring_ways_octahedron_coloring_ways_l2950_295055

-- Define the number of colors for each shape
def cube_colors : ℕ := 6
def octahedron_colors : ℕ := 8

-- Define the number of faces for each shape
def cube_faces : ℕ := 6
def octahedron_faces : ℕ := 8

-- Theorem for coloring the cube
theorem cube_coloring_ways :
  (cube_colors.factorial / (cube_colors - cube_faces).factorial) = 30 := by
  sorry

-- Theorem for coloring the octahedron
theorem octahedron_coloring_ways :
  (octahedron_colors.factorial / (octahedron_colors - octahedron_faces).factorial) = 1680 := by
  sorry

end cube_coloring_ways_octahedron_coloring_ways_l2950_295055


namespace A_single_element_A_at_most_one_element_l2950_295053

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 - x + a + 2 = 0}

-- Theorem 1: A contains only one element iff a ∈ {0, -2+√5, -2-√5}
theorem A_single_element (a : ℝ) :
  (∃! x, x ∈ A a) ↔ a = 0 ∨ a = -2 + Real.sqrt 5 ∨ a = -2 - Real.sqrt 5 :=
sorry

-- Theorem 2: A contains at most one element iff a ∈ (-∞, -2-√5] ∪ {0} ∪ [-2+√5, +∞)
theorem A_at_most_one_element (a : ℝ) :
  (∀ x y, x ∈ A a → y ∈ A a → x = y) ↔
  a ≤ -2 - Real.sqrt 5 ∨ a = 0 ∨ a ≥ -2 + Real.sqrt 5 :=
sorry

end A_single_element_A_at_most_one_element_l2950_295053


namespace students_remaining_l2950_295063

theorem students_remaining (groups : ℕ) (students_per_group : ℕ) (left_early : ℕ) : 
  groups = 5 → students_per_group = 12 → left_early = 7 → 
  groups * students_per_group - left_early = 53 := by
  sorry

end students_remaining_l2950_295063


namespace yellow_preference_l2950_295076

/-- Proves that 9 students like yellow best given the survey conditions --/
theorem yellow_preference (total_students : ℕ) (total_girls : ℕ) 
  (h_total : total_students = 30)
  (h_girls : total_girls = 18)
  (h_green : total_students / 2 = total_students - total_students / 2)
  (h_pink : total_girls / 3 = total_girls - 2 * (total_girls / 3)) :
  total_students - (total_students / 2 + total_girls / 3) = 9 := by
  sorry

#check yellow_preference

end yellow_preference_l2950_295076


namespace complex_number_equal_parts_l2950_295086

theorem complex_number_equal_parts (b : ℝ) : 
  let z : ℂ := (3 - b * Complex.I) / (2 + Complex.I)
  z.re = z.im ↔ b = -9 := by
sorry

end complex_number_equal_parts_l2950_295086


namespace a_bounds_l2950_295023

/-- Given a linear equation y = ax + 1/3 where x and y are bounded,
    prove that a is bounded between -1/3 and 2/3. -/
theorem a_bounds (a : ℝ) : 
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ y = a * x + 1/3) →
  -1/3 ≤ a ∧ a ≤ 2/3 := by
  sorry

#check a_bounds

end a_bounds_l2950_295023


namespace smallest_five_digit_divisor_l2950_295002

def is_valid_seven_digit_number (n : ℕ) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧
  (n / 100 % 10 = 2 * (n / 1000000)) ∧
  (n / 10 % 10 = 2 * (n / 100000 % 10)) ∧
  (n % 10 = 2 * (n / 10000 % 10)) ∧
  (n / 1000 % 10 = 0)

theorem smallest_five_digit_divisor :
  ∃ (n : ℕ), is_valid_seven_digit_number n ∧ n % 10002 = 0 ∧
  ∀ (m : ℕ), 10000 ≤ m ∧ m < 10002 → ¬(∃ (k : ℕ), is_valid_seven_digit_number k ∧ k % m = 0) :=
sorry

end smallest_five_digit_divisor_l2950_295002


namespace common_roots_product_l2950_295024

/-- Given two cubic equations with two common roots, prove that the product of these common roots is 10√[3]{2} -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (u v w t : ℝ),
    (u^3 + C*u + 20 = 0) ∧ 
    (v^3 + C*v + 20 = 0) ∧ 
    (w^3 + C*w + 20 = 0) ∧
    (u^3 + D*u^2 + 100 = 0) ∧ 
    (v^3 + D*v^2 + 100 = 0) ∧ 
    (t^3 + D*t^2 + 100 = 0) ∧
    (u ≠ v) ∧ 
    (u * v = 10 * Real.rpow 2 (1/3)) := by
  sorry

end common_roots_product_l2950_295024


namespace symmetric_function_zero_l2950_295061

/-- A function f: ℝ → ℝ satisfying specific symmetry properties -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2*x + 2) = -f (-2*x - 2)) ∧ 
  (∀ x, f (x + 1) = f (-x + 1))

/-- Theorem stating that for a function with the given symmetry properties, f(4) = 0 -/
theorem symmetric_function_zero (f : ℝ → ℝ) (h : SymmetricFunction f) : f 4 = 0 := by
  sorry

end symmetric_function_zero_l2950_295061


namespace horner_method_v2_l2950_295077

def horner_polynomial (x : ℝ) : ℝ := 2*x^6 + 3*x^5 + 5*x^3 + 6*x^2 + 7*x + 8

def horner_v0 : ℝ := 2
def horner_v1 (x : ℝ) : ℝ := horner_v0 * x + 3
def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x + 0

theorem horner_method_v2 :
  horner_v2 2 = 14 ∧ horner_polynomial 2 = horner_v2 2 :=
sorry

end horner_method_v2_l2950_295077


namespace union_A_B_equals_open_interval_l2950_295052

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x}

-- Theorem statement
theorem union_A_B_equals_open_interval :
  A ∪ B = Set.Ioo (-2 : ℝ) 4 := by sorry

end union_A_B_equals_open_interval_l2950_295052


namespace unique_solution_l2950_295075

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f (x + y)) = f x * f y + f x + f y + x * y

/-- The theorem stating that f(1) = 1 is the only solution -/
theorem unique_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 1 = 1 := by
  sorry

end unique_solution_l2950_295075


namespace arithmetic_mean_difference_l2950_295058

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 22) : 
  r - p = 24 := by
sorry

end arithmetic_mean_difference_l2950_295058


namespace two_digit_square_l2950_295046

/-- Given distinct digits a, b, c, prove that the two-digit number 'ab' is 21 -/
theorem two_digit_square (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c →
  b = 1 →
  10 * a + b < 100 →
  100 * c + 10 * c + b > 300 →
  (10 * a + b)^2 = 100 * c + 10 * c + b →
  10 * a + b = 21 := by
sorry

end two_digit_square_l2950_295046


namespace distance_Y_to_GH_l2950_295005

-- Define the square
def Square (t : ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ t ∧ 0 ≤ p.2 ∧ p.2 ≤ t}

-- Define the half-circle centered at E
def ArcE (t : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = (t/2)^2 ∧ 0 ≤ p.1 ∧ 0 ≤ p.2}

-- Define the half-circle centered at F
def ArcF (t : ℝ) := {p : ℝ × ℝ | (p.1 - t)^2 + p.2^2 = (3*t/2)^2 ∧ p.1 ≤ t ∧ 0 ≤ p.2}

-- Define the intersection point Y
def Y (t : ℝ) := {p : ℝ × ℝ | p ∈ ArcE t ∧ p ∈ ArcF t ∧ p ∈ Square t}

-- Theorem statement
theorem distance_Y_to_GH (t : ℝ) (h : t > 0) :
  ∀ y ∈ Y t, t - y.2 = t :=
sorry

end distance_Y_to_GH_l2950_295005


namespace distinct_prime_factors_of_300_l2950_295042

theorem distinct_prime_factors_of_300 : Nat.card (Nat.factors 300).toFinset = 3 := by
  sorry

end distinct_prime_factors_of_300_l2950_295042


namespace ed_lost_seven_marbles_l2950_295006

/-- Represents the number of marbles each person has -/
structure MarbleCount where
  doug : ℕ
  ed : ℕ
  tim : ℕ

/-- The initial state of marble distribution -/
def initial_state (d : ℕ) : MarbleCount :=
  { doug := d
  , ed := d + 19
  , tim := d - 10 }

/-- The final state of marble distribution after transactions -/
def final_state (d : ℕ) (l : ℕ) : MarbleCount :=
  { doug := d
  , ed := d + 8
  , tim := d }

/-- Theorem stating that Ed lost 7 marbles -/
theorem ed_lost_seven_marbles (d : ℕ) :
  ∃ l : ℕ, 
    (initial_state d).ed - l - 4 = (final_state d l).ed ∧
    (initial_state d).tim + 4 + 3 = (final_state d l).tim ∧
    l = 7 := by
  sorry

#check ed_lost_seven_marbles

end ed_lost_seven_marbles_l2950_295006


namespace last_three_average_l2950_295045

theorem last_three_average (numbers : List ℝ) : 
  numbers.length = 7 → 
  numbers.sum / 7 = 60 → 
  (numbers.take 4).sum / 4 = 55 → 
  (numbers.drop 4).sum / 3 = 200 / 3 := by
sorry

end last_three_average_l2950_295045


namespace scout_troop_profit_l2950_295013

/-- The profit calculation for a scout troop selling candy bars -/
theorem scout_troop_profit : 
  let total_bars : ℕ := 1500
  let cost_price : ℚ := 1 / 3  -- price per bar when buying more than 800
  let selling_price : ℚ := 1 / 2  -- price per bar when selling
  let total_cost : ℚ := total_bars * cost_price
  let total_revenue : ℚ := total_bars * selling_price
  let profit : ℚ := total_revenue - total_cost
  profit = 250 := by sorry

end scout_troop_profit_l2950_295013


namespace george_carries_two_buckets_l2950_295074

/-- The number of buckets George can carry each round -/
def george_buckets : ℕ := 2

/-- The number of buckets Harry can carry each round -/
def harry_buckets : ℕ := 3

/-- The total number of buckets needed to fill the pool -/
def total_buckets : ℕ := 110

/-- The number of rounds needed to fill the pool -/
def total_rounds : ℕ := 22

theorem george_carries_two_buckets :
  george_buckets = 2 ∧
  harry_buckets * total_rounds + george_buckets * total_rounds = total_buckets :=
sorry

end george_carries_two_buckets_l2950_295074


namespace max_value_implies_m_equals_four_l2950_295036

theorem max_value_implies_m_equals_four (x y m : ℝ) : 
  x > 1 →
  y ≥ x →
  y ≤ 2 * x →
  x + y ≤ 1 →
  (∀ x' y' : ℝ, y' ≥ x' → y' ≤ 2 * x' → x' + y' ≤ 1 → x' + m * y' ≤ x + m * y) →
  x + m * y = 3 →
  m = 4 := by
sorry

end max_value_implies_m_equals_four_l2950_295036


namespace original_ratio_of_boarders_to_day_students_l2950_295020

/-- Proof of the original ratio of boarders to day students -/
theorem original_ratio_of_boarders_to_day_students :
  let initial_boarders : ℕ := 120
  let new_boarders : ℕ := 30
  let total_boarders : ℕ := initial_boarders + new_boarders
  let day_students : ℕ := 2 * total_boarders
  (initial_boarders : ℚ) / day_students = 1 / (5 / 2) := by
  sorry

end original_ratio_of_boarders_to_day_students_l2950_295020


namespace faye_age_l2950_295049

/-- Represents the ages of the individuals in the problem --/
structure Ages where
  diana : ℕ
  eduardo : ℕ
  chad : ℕ
  faye : ℕ
  george : ℕ

/-- The conditions of the problem --/
def valid_ages (a : Ages) : Prop :=
  a.diana + 2 = a.eduardo ∧
  a.eduardo = a.chad + 6 ∧
  a.faye = a.chad + 4 ∧
  a.george + 5 = a.chad ∧
  a.diana = 16

/-- The theorem to prove --/
theorem faye_age (a : Ages) (h : valid_ages a) : a.faye = 16 := by
  sorry

#check faye_age

end faye_age_l2950_295049


namespace digit_sum_problem_l2950_295070

/-- Given single-digit integers a and b satisfying certain conditions, prove their sum is 7 --/
theorem digit_sum_problem (a b : ℕ) : 
  a < 10 → b < 10 → (4 * a) % 10 = 6 → 3 * b * 10 + 4 * a = 116 → a + b = 7 := by
  sorry

end digit_sum_problem_l2950_295070


namespace probability_continuous_stripe_is_one_fourth_l2950_295062

/-- A regular tetrahedron with painted stripes on its faces -/
structure StripedTetrahedron where
  /-- The number of faces of the tetrahedron -/
  num_faces : Nat
  /-- The number of possible stripe configurations per face -/
  stripe_configs : Nat
  /-- The probability of a continuous stripe pattern given all faces have intersecting stripes -/
  prob_continuous_intersecting : ℚ

/-- The probability of at least one continuous stripe pattern encircling the tetrahedron -/
def probability_continuous_stripe (t : StripedTetrahedron) : ℚ :=
  2 * (1 / 2) ^ 3

theorem probability_continuous_stripe_is_one_fourth (t : StripedTetrahedron) 
    (h1 : t.num_faces = 4)
    (h2 : t.stripe_configs = 2)
    (h3 : t.prob_continuous_intersecting = 1 / 16) :
  probability_continuous_stripe t = 1 / 4 := by
  sorry

end probability_continuous_stripe_is_one_fourth_l2950_295062


namespace base_conversion_subtraction_l2950_295015

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The problem statement -/
theorem base_conversion_subtraction :
  let base7_number := [3, 0, 4, 2, 5]  -- 52403 in base 7 (least significant digit first)
  let base5_number := [5, 4, 3, 0, 2]  -- 20345 in base 5 (least significant digit first)
  toBase10 base7_number 7 - toBase10 base5_number 5 = 11540 := by
sorry

end base_conversion_subtraction_l2950_295015


namespace min_value_expression_l2950_295012

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / (2 * a)) + (a / (b + 1)) ≥ 5/4 :=
by sorry

end min_value_expression_l2950_295012


namespace sum_first_150_remainder_l2950_295079

theorem sum_first_150_remainder (n : Nat) (sum : Nat) : 
  n = 150 → 
  sum = n * (n + 1) / 2 → 
  sum % 11300 = 25 := by
  sorry

end sum_first_150_remainder_l2950_295079


namespace unique_cuddly_number_l2950_295078

/-- A two-digit positive integer is cuddly if it equals the sum of its nonzero tens digit and the square of its units digit. -/
def IsCuddly (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n = (n / 10) + (n % 10)^2

/-- There exists exactly one cuddly two-digit positive integer. -/
theorem unique_cuddly_number : ∃! n : ℕ, IsCuddly n :=
  sorry

end unique_cuddly_number_l2950_295078


namespace circle_area_through_points_l2950_295090

/-- The area of a circle with center R(1, 2) passing through S(-7, 6) is 80π -/
theorem circle_area_through_points : 
  let R : ℝ × ℝ := (1, 2)
  let S : ℝ × ℝ := (-7, 6)
  let radius := Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)
  π * radius^2 = 80 * π := by sorry

end circle_area_through_points_l2950_295090


namespace line_intersection_symmetry_l2950_295071

/-- Given a line y = -x + m intersecting the x-axis at A, prove that when moved 6 units left
    to intersect the x-axis at A', if A' is symmetric to A about the origin, then m = 3. -/
theorem line_intersection_symmetry (m : ℝ) : 
  let A : ℝ × ℝ := (m, 0)
  let A' : ℝ × ℝ := (m - 6, 0)
  (A'.1 = -A.1 ∧ A'.2 = -A.2) → m = 3 := by
sorry

end line_intersection_symmetry_l2950_295071


namespace min_value_and_sum_l2950_295080

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

-- State the theorem
theorem min_value_and_sum (a b : ℝ) : 
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) ∧ m = 2) ∧
  (a^2 + b^2 = 2 → 1 / (a^2 + 1) + 4 / (b^2 + 1) ≥ 9/4) :=
sorry

end min_value_and_sum_l2950_295080


namespace cos_to_sin_shift_l2950_295054

theorem cos_to_sin_shift (x : ℝ) : 
  3 * Real.cos (2 * x - π / 4) = 3 * Real.sin (2 * (x + π / 8)) := by
  sorry

end cos_to_sin_shift_l2950_295054


namespace product_ab_equals_ten_l2950_295089

theorem product_ab_equals_ten (a b c : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) 
  (h3 : a + b + c = 21) : 
  a * b = 10 := by
sorry

end product_ab_equals_ten_l2950_295089


namespace inequality_proof_l2950_295066

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Define the set M
def M : Set ℝ := {x | f x < 4}

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 
  2 * |a + b| < |4 + a * b| := by
  sorry

end inequality_proof_l2950_295066


namespace banana_permutations_l2950_295085

/-- The number of permutations of a multiset -/
def multiset_permutations (n : ℕ) (frequencies : List ℕ) : ℕ :=
  Nat.factorial n / (List.prod (List.map Nat.factorial frequencies))

/-- Theorem: The number of distinct permutations of BANANA is 60 -/
theorem banana_permutations :
  multiset_permutations 6 [3, 2, 1] = 60 := by
  sorry

end banana_permutations_l2950_295085


namespace new_crew_member_weight_l2950_295018

/-- Given a crew of 10 oarsmen, prove that replacing a 53 kg member with a new member
    that increases the average weight by 1.8 kg results in the new member weighing 71 kg. -/
theorem new_crew_member_weight (crew_size : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  crew_size = 10 →
  weight_increase = 1.8 →
  replaced_weight = 53 →
  (crew_size : ℝ) * weight_increase + replaced_weight = 71 := by
  sorry

end new_crew_member_weight_l2950_295018


namespace function_zeros_imply_k_range_l2950_295016

open Real

theorem function_zeros_imply_k_range (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, f x = (log x) / x - k * x) →
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 1/ℯ ≤ x₁ ∧ x₁ ≤ ℯ^2 ∧ 1/ℯ ≤ x₂ ∧ x₂ ≤ ℯ^2 ∧ f x₁ = 0 ∧ f x₂ = 0) →
  2/ℯ^4 ≤ k ∧ k < 1/(2*ℯ) :=
by sorry

end function_zeros_imply_k_range_l2950_295016


namespace sin_squared_sum_6_to_174_l2950_295009

theorem sin_squared_sum_6_to_174 : 
  (Finset.range 29).sum (fun k => Real.sin ((6 * k + 6 : ℕ) * π / 180) ^ 2) = 31 / 2 := by
  sorry

end sin_squared_sum_6_to_174_l2950_295009


namespace equation_solution_l2950_295022

theorem equation_solution : ∃ x : ℝ, 
  x = 160 + 64 * Real.sqrt 6 ∧ 
  Real.sqrt (2 + Real.sqrt (3 + Real.sqrt x)) = (2 + Real.sqrt x) ^ (1/4) := by
  sorry

end equation_solution_l2950_295022


namespace unique_solution_system_l2950_295067

theorem unique_solution_system (a b : ℕ+) 
  (h1 : a^(b:ℕ) + 3 = b^(a:ℕ)) 
  (h2 : 3 * a^(b:ℕ) = b^(a:ℕ) + 13) : 
  a = 2 ∧ b = 3 := by
sorry

end unique_solution_system_l2950_295067


namespace inequality_equivalence_l2950_295029

def f (x : ℝ) : ℝ := x * abs x

theorem inequality_equivalence (m : ℝ) : 
  (∀ x ≥ 1, f (x + m) + m * f x < 0) ↔ m ∈ Set.Iic (-1 : ℝ) := by
  sorry

end inequality_equivalence_l2950_295029


namespace minute_hand_angle_for_110_minutes_l2950_295087

/-- The angle turned by the minute hand when the hour hand moves for a given time -/
def minuteHandAngle (hours : ℝ) (minutes : ℝ) : ℝ :=
  -(hours * 360 + minutes * 6)

/-- Theorem: When the hour hand moves for 1 hour and 50 minutes, 
    the angle turned by the minute hand is -660° -/
theorem minute_hand_angle_for_110_minutes : 
  minuteHandAngle 1 50 = -660 := by sorry

end minute_hand_angle_for_110_minutes_l2950_295087


namespace maintenance_cost_third_year_l2950_295072

/-- Represents the maintenance cost function for factory equipment -/
def maintenance_cost (x : ℝ) : ℝ := 0.8 * x + 1.5

/-- Proves that the maintenance cost for equipment in its third year is 3.9 ten thousand yuan -/
theorem maintenance_cost_third_year :
  maintenance_cost 3 = 3.9 := by
  sorry

end maintenance_cost_third_year_l2950_295072


namespace sum_of_digits_1729_base_8_l2950_295091

/-- Converts a natural number to its base 8 representation -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Sums the digits in a list of natural numbers -/
def sumDigits (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

/-- The sum of digits of 1729 in base 8 is equal to 7 -/
theorem sum_of_digits_1729_base_8 :
  sumDigits (toBase8 1729) = 7 := by
  sorry

end sum_of_digits_1729_base_8_l2950_295091


namespace min_abs_z_on_locus_l2950_295034

theorem min_abs_z_on_locus (z : ℂ) (h : Complex.abs (z - (0 : ℂ) + 4*I) + Complex.abs (z - 5) = 7) : 
  ∃ (w : ℂ), Complex.abs (w - (0 : ℂ) + 4*I) + Complex.abs (w - 5) = 7 ∧ 
  (∀ (v : ℂ), Complex.abs (v - (0 : ℂ) + 4*I) + Complex.abs (v - 5) = 7 → Complex.abs w ≤ Complex.abs v) ∧
  Complex.abs w = 20 / 7 :=
sorry

end min_abs_z_on_locus_l2950_295034


namespace cannot_transform_to_target_l2950_295000

/-- Represents a parabola equation in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines a simple transformation of a parabola --/
inductive SimpleTransformation
  | right : SimpleTransformation  -- Move 2 units right
  | up : SimpleTransformation     -- Move 1 unit up

/-- Applies a simple transformation to a parabola --/
def applyTransformation (p : Parabola) (t : SimpleTransformation) : Parabola :=
  match t with
  | SimpleTransformation.right => { a := p.a, b := p.b - 2 * p.a, c := p.c + p.a }
  | SimpleTransformation.up => { a := p.a, b := p.b, c := p.c + 1 }

/-- Applies a sequence of simple transformations to a parabola --/
def applyTransformations (p : Parabola) (ts : List SimpleTransformation) : Parabola :=
  ts.foldl applyTransformation p

theorem cannot_transform_to_target : 
  ∀ (ts : List SimpleTransformation),
    ts.length = 2 → 
    applyTransformations { a := 1, b := 6, c := 5 } ts ≠ { a := 1, b := 0, c := 1 } :=
sorry

end cannot_transform_to_target_l2950_295000


namespace incircle_center_locus_is_mid_distance_strip_l2950_295038

/-- Represents a line in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents the incircle of a triangle -/
structure Incircle :=
  (center : Point)
  (radius : ℝ)

/-- Three parallel lines on a plane -/
def parallel_lines : List Line := sorry

/-- A triangle with vertices on the parallel lines -/
def triangle_on_lines (t : Triangle) : Prop := sorry

/-- The incircle of a triangle -/
def incircle_of_triangle (t : Triangle) : Incircle := sorry

/-- The strip between the mid-distances of outer and middle lines -/
def mid_distance_strip : Set Point := sorry

/-- The geometric locus of incircle centers -/
def incircle_center_locus : Set Point := sorry

/-- Theorem: The geometric locus of the centers of incircles of triangles with vertices on three parallel lines
    is the strip bound by lines parallel and in the mid-distance between the outer and mid lines -/
theorem incircle_center_locus_is_mid_distance_strip :
  incircle_center_locus = mid_distance_strip := by sorry

end incircle_center_locus_is_mid_distance_strip_l2950_295038


namespace det_A_eq_58_l2950_295027

def A : Matrix (Fin 2) (Fin 2) ℝ := !![10, 4; -2, 5]

theorem det_A_eq_58 : Matrix.det A = 58 := by sorry

end det_A_eq_58_l2950_295027


namespace intersection_implies_sum_of_slopes_is_five_l2950_295004

/-- Given two sets A and B in R^2, defined by linear equations,
    prove that if their intersection is a single point (2, 5),
    then the sum of their slopes is 5. -/
theorem intersection_implies_sum_of_slopes_is_five 
  (a b : ℝ) 
  (A : Set (ℝ × ℝ)) 
  (B : Set (ℝ × ℝ)) 
  (h1 : A = {p : ℝ × ℝ | p.2 = a * p.1 + 1})
  (h2 : B = {p : ℝ × ℝ | p.2 = p.1 + b})
  (h3 : A ∩ B = {(2, 5)}) : 
  a + b = 5 := by
sorry

end intersection_implies_sum_of_slopes_is_five_l2950_295004


namespace b_17_value_l2950_295084

/-- A sequence where consecutive terms are roots of a quadratic equation -/
def special_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, (a n)^2 - n * (a n) + (b n) = 0 ∧
       (a (n + 1))^2 - n * (a (n + 1)) + (b n) = 0

theorem b_17_value (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h : special_sequence a b) (h10 : a 10 = 7) : b 17 = 66 := by
  sorry

end b_17_value_l2950_295084


namespace tax_revenue_change_l2950_295099

theorem tax_revenue_change 
  (original_tax : ℝ) 
  (original_consumption : ℝ) 
  (tax_reduction_rate : ℝ) 
  (consumption_increase_rate : ℝ) 
  (h1 : tax_reduction_rate = 0.16) 
  (h2 : consumption_increase_rate = 0.15) : 
  let new_tax := original_tax * (1 - tax_reduction_rate)
  let new_consumption := original_consumption * (1 + consumption_increase_rate)
  let original_revenue := original_tax * original_consumption
  let new_revenue := new_tax * new_consumption
  (new_revenue - original_revenue) / original_revenue = -0.034 :=
by sorry

end tax_revenue_change_l2950_295099


namespace theater_seats_l2950_295097

/-- The number of seats in the nth row of the theater -/
def seats_in_row (n : ℕ) : ℕ := 2 * n + 26

/-- The total number of seats in the theater -/
def total_seats (rows : ℕ) : ℕ :=
  (seats_in_row 1 + seats_in_row rows) * rows / 2

/-- Theorem stating the total number of seats in the theater -/
theorem theater_seats :
  total_seats 20 = 940 := by sorry

end theater_seats_l2950_295097


namespace sisyphus_stones_l2950_295030

/-- The minimum number of operations to move n stones to the rightmost square -/
def minOperations (n : ℕ) : ℕ :=
  (Finset.range n).sum fun k => (n + k) / (k + 1)

/-- The problem statement -/
theorem sisyphus_stones (n : ℕ) (h : n > 0) :
  ∀ (ops : ℕ), 
    (∃ (final_state : Fin (n + 1) → ℕ), 
      (final_state (Fin.last n) = n) ∧ 
      (∀ i < n, final_state i = 0) ∧
      (∃ (initial_state : Fin (n + 1) → ℕ),
        (initial_state 0 = n) ∧
        (∀ i > 0, initial_state i = 0) ∧
        (∃ (moves : Fin ops → Fin (n + 1) × Fin (n + 1)),
          (∀ m, (moves m).1 < (moves m).2) ∧
          (∀ m, (moves m).2.val - (moves m).1.val ≤ initial_state (moves m).1)))) →
    ops ≥ minOperations n :=
by sorry

end sisyphus_stones_l2950_295030


namespace project_completion_time_l2950_295044

/-- The number of days B takes to complete the project alone -/
def B_days : ℝ := 30

/-- The total number of days the project takes when A and B work together, with A quitting 15 days before completion -/
def total_days : ℝ := 21

/-- The number of days before project completion that A quits -/
def A_quit_days : ℝ := 15

/-- The number of days A takes to complete the project alone -/
def A_days : ℝ := 20

theorem project_completion_time :
  A_days = 20 :=
by sorry

end project_completion_time_l2950_295044


namespace quadratic_transformation_l2950_295069

theorem quadratic_transformation (x : ℝ) :
  x^2 - 6*x - 1 = 0 ↔ (x + 3)^2 = 10 :=
by sorry

end quadratic_transformation_l2950_295069


namespace product_cube_square_l2950_295017

theorem product_cube_square : ((-1 : ℤ)^3) * ((-2 : ℤ)^2) = -4 := by sorry

end product_cube_square_l2950_295017


namespace volunteers_selection_theorem_l2950_295092

theorem volunteers_selection_theorem :
  let n : ℕ := 5  -- Total number of volunteers
  let k : ℕ := 2  -- Number of people to be sent to each location
  let locations : ℕ := 2  -- Number of locations
  Nat.choose n k * Nat.choose (n - k) k = 30 := by
  sorry

end volunteers_selection_theorem_l2950_295092


namespace gumballs_to_todd_l2950_295032

/-- Represents the distribution of gumballs among friends --/
structure GumballDistribution where
  total : ℕ
  remaining : ℕ
  todd : ℕ
  alisha : ℕ
  bobby : ℕ

/-- Checks if a gumball distribution satisfies the given conditions --/
def isValidDistribution (d : GumballDistribution) : Prop :=
  d.total = 45 ∧
  d.remaining = 6 ∧
  d.alisha = 2 * d.todd ∧
  d.bobby = 4 * d.alisha - 5 ∧
  d.total = d.todd + d.alisha + d.bobby + d.remaining

theorem gumballs_to_todd (d : GumballDistribution) :
  isValidDistribution d → d.todd = 4 := by
  sorry

end gumballs_to_todd_l2950_295032


namespace milkshake_production_l2950_295065

/-- Augustus's milkshake production rate per hour -/
def augustus_rate : ℕ := 3

/-- Luna's milkshake production rate per hour -/
def luna_rate : ℕ := 7

/-- The number of hours Augustus and Luna have been making milkshakes -/
def hours_worked : ℕ := 8

/-- The total number of milkshakes made by Augustus and Luna -/
def total_milkshakes : ℕ := (augustus_rate + luna_rate) * hours_worked

theorem milkshake_production :
  total_milkshakes = 80 := by sorry

end milkshake_production_l2950_295065


namespace sixth_term_value_l2950_295010

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  a_1_eq_4 : a 1 = 4
  a_3_eq_prod : a 3 = a 2 * a 4

/-- The sixth term of the geometric sequence is either 1/8 or -1/8 -/
theorem sixth_term_value (seq : GeometricSequence) : 
  seq.a 6 = 1/8 ∨ seq.a 6 = -1/8 := by
  sorry

end sixth_term_value_l2950_295010


namespace clothing_prices_l2950_295007

-- Define the original prices
def original_sweater_price : ℝ := 43.11
def original_shirt_price : ℝ := original_sweater_price - 7.43
def original_pants_price : ℝ := 2 * original_shirt_price

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the total cost after discount
def total_cost : ℝ := 143.67

-- Theorem statement
theorem clothing_prices :
  (original_shirt_price = 35.68) ∧
  (original_sweater_price = 43.11) ∧
  (original_pants_price = 71.36) ∧
  (original_shirt_price + (1 - discount_rate) * original_sweater_price + original_pants_price = total_cost) := by
  sorry

end clothing_prices_l2950_295007


namespace function_maximum_value_l2950_295060

theorem function_maximum_value (x : ℝ) (h : x < 1/2) :
  ∃ M : ℝ, M = -1 ∧ ∀ y : ℝ, y = 2*x + 1/(2*x - 1) → y ≤ M := by
  sorry

end function_maximum_value_l2950_295060


namespace greatest_multiple_of_5_and_6_less_than_1000_l2950_295068

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n : ℕ,
  n < 1000 ∧ 
  5 ∣ n ∧ 
  6 ∣ n ∧ 
  ∀ m : ℕ, m < 1000 → 5 ∣ m → 6 ∣ m → m ≤ n :=
by
  -- The proof goes here
  sorry

end greatest_multiple_of_5_and_6_less_than_1000_l2950_295068


namespace gianna_savings_l2950_295073

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (n : ℕ) (a₁ d : ℚ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℕ) * d)

/-- Gianna's savings problem -/
theorem gianna_savings :
  arithmetic_sum 365 39 2 = 147095 := by
  sorry

end gianna_savings_l2950_295073


namespace matrix_inverse_proof_l2950_295039

def A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -5; -4, 3]
def A_inv : Matrix (Fin 2) (Fin 2) ℤ := !![3, 5; 4, 7]

theorem matrix_inverse_proof :
  IsUnit (Matrix.det A) ∧ A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end matrix_inverse_proof_l2950_295039


namespace area_after_shortening_l2950_295050

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle --/
def original : Rectangle := ⟨5, 7⟩

/-- Shortens either the length or the width of a rectangle by 2 --/
def shorten (r : Rectangle) (shortenLength : Bool) : Rectangle :=
  if shortenLength then ⟨r.length - 2, r.width⟩ else ⟨r.length, r.width - 2⟩

theorem area_after_shortening :
  (area (shorten original true) = 21 ∧ area (shorten original false) = 25) ∨
  (area (shorten original true) = 25 ∧ area (shorten original false) = 21) :=
by sorry

end area_after_shortening_l2950_295050


namespace parallel_vectors_difference_l2950_295057

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_difference (x : ℝ) :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (x - 2, -2)
  are_parallel a b → a - b = (-2, 1) := by
sorry

end parallel_vectors_difference_l2950_295057


namespace pyramid_volume_theorem_l2950_295026

/-- Represents a pyramid with a square base ABCD and vertex E -/
structure Pyramid where
  baseArea : ℝ
  triangleABEArea : ℝ
  triangleCDEArea : ℝ

/-- Calculates the volume of the pyramid -/
def pyramidVolume (p : Pyramid) : ℝ :=
  sorry

/-- Theorem stating the volume of the pyramid with given conditions -/
theorem pyramid_volume_theorem (p : Pyramid) 
  (h1 : p.baseArea = 256)
  (h2 : p.triangleABEArea = 128)
  (h3 : p.triangleCDEArea = 96) :
  pyramidVolume p = 1194 + 2/3 := by
  sorry

end pyramid_volume_theorem_l2950_295026


namespace same_number_on_four_dice_l2950_295040

theorem same_number_on_four_dice : 
  let n : ℕ := 6  -- number of sides on each die
  let k : ℕ := 4  -- number of dice
  (1 : ℚ) / n^(k-1) = (1 : ℚ) / 216 :=
by sorry

end same_number_on_four_dice_l2950_295040


namespace max_value_condition_l2950_295021

theorem max_value_condition (x y : ℝ) : 
  (2 * x^2 - y^2 + 3/2 ≤ 1 ∧ y^4 + 4*x + 2 ≤ 1) ↔ 
  ((x = -1/2 ∧ y = 1) ∨ (x = -1/2 ∧ y = -1)) :=
by sorry

end max_value_condition_l2950_295021


namespace library_wage_calculation_l2950_295096

/-- Represents the weekly work schedule and earnings of a student with two part-time jobs -/
structure WorkSchedule where
  library_hours : ℝ
  construction_hours : ℝ
  library_wage : ℝ
  construction_wage : ℝ
  total_earnings : ℝ

/-- Theorem stating the library wage given the problem conditions -/
theorem library_wage_calculation (w : WorkSchedule) :
  w.library_hours = 10 ∧
  w.construction_hours = 15 ∧
  w.construction_wage = 15 ∧
  w.library_hours + w.construction_hours = 25 ∧
  w.total_earnings ≥ 300 ∧
  w.total_earnings = w.library_hours * w.library_wage + w.construction_hours * w.construction_wage →
  w.library_wage = 7.5 := by
  sorry

#check library_wage_calculation

end library_wage_calculation_l2950_295096
