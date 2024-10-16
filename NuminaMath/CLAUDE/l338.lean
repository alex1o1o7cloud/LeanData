import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_integers_squares_sum_l338_33876

theorem consecutive_integers_squares_sum : ∃ a : ℕ,
  (a > 0) ∧
  ((a - 1) * a * (a + 1) = 8 * (3 * a)) ∧
  ((a - 1)^2 + a^2 + (a + 1)^2 = 77) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_squares_sum_l338_33876


namespace NUMINAMATH_CALUDE_proposition_1_proposition_2_false_proposition_3_l338_33806

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations and operations
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersect : Line → Line → Prop → Prop)
variable (point_on : Prop → Line → Prop)
variable (coplanar : Line → Line → Prop)

-- Theorem 1
theorem proposition_1 (m l : Line) (α : Plane) (A : Prop) :
  contains α m →
  perpendicular l α →
  point_on A l →
  ¬point_on A m →
  ¬coplanar l m :=
sorry

-- Theorem 2
theorem proposition_2_false (l m : Line) (α β : Plane) :
  ¬(∀ (l m : Line) (α β : Plane),
    parallel l α →
    parallel m β →
    parallel_planes α β →
    parallel_lines l m) :=
sorry

-- Theorem 3
theorem proposition_3 (l m : Line) (α β : Plane) (A : Prop) :
  contains α l →
  contains α m →
  intersect l m A →
  parallel l β →
  parallel m β →
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_2_false_proposition_3_l338_33806


namespace NUMINAMATH_CALUDE_randy_mango_trees_l338_33866

theorem randy_mango_trees :
  ∀ (mango coconut : ℕ),
  coconut = mango / 2 - 5 →
  mango + coconut = 85 →
  mango = 60 := by
sorry

end NUMINAMATH_CALUDE_randy_mango_trees_l338_33866


namespace NUMINAMATH_CALUDE_student_calculation_error_l338_33821

def correct_calculation : ℚ := (3/4 * 16 - 7/8 * 8) / (3/10 - 1/8)

def incorrect_calculation : ℚ := (3/4 * 16 - 7/8 * 8) * (3/5)

def percentage_error (correct incorrect : ℚ) : ℚ :=
  abs (correct - incorrect) / correct * 100

theorem student_calculation_error :
  abs (percentage_error correct_calculation incorrect_calculation - 89.47) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_error_l338_33821


namespace NUMINAMATH_CALUDE_election_percentage_l338_33817

theorem election_percentage (total_votes : ℕ) (vote_difference : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 5500 →
  vote_difference = 1650 →
  candidate_percentage = 35 / 100 →
  (candidate_percentage * total_votes : ℚ) + 
  (candidate_percentage * total_votes : ℚ) + vote_difference = total_votes :=
by sorry

end NUMINAMATH_CALUDE_election_percentage_l338_33817


namespace NUMINAMATH_CALUDE_max_x_minus_y_l338_33837

theorem max_x_minus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) :
  ∃ (m : ℝ), m = 1 / (2 * Real.sqrt 3) ∧ ∀ (a b : ℝ), 3 * (a^2 + b^2) = a + b → (a - b) ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l338_33837


namespace NUMINAMATH_CALUDE_sequence_contains_24_l338_33815

theorem sequence_contains_24 : ∃ n : ℕ+, n * (n + 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_sequence_contains_24_l338_33815


namespace NUMINAMATH_CALUDE_even_increasing_relation_l338_33839

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y → f x < f y

theorem even_increasing_relation (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_incr : increasing_on_nonneg f) :
  f π > f (-3) ∧ f (-3) > f (-2) :=
sorry

end NUMINAMATH_CALUDE_even_increasing_relation_l338_33839


namespace NUMINAMATH_CALUDE_limit_sine_cosine_ratio_l338_33856

theorem limit_sine_cosine_ratio : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → 
    |(1 + Real.sin (2*x) - Real.cos (2*x)) / (1 - Real.sin (2*x) - Real.cos (2*x)) + 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_sine_cosine_ratio_l338_33856


namespace NUMINAMATH_CALUDE_hike_attendance_l338_33834

theorem hike_attendance (num_cars num_taxis num_vans : ℕ) 
                        (people_per_car people_per_taxi people_per_van : ℕ) : 
  num_cars = 3 → 
  num_taxis = 6 → 
  num_vans = 2 → 
  people_per_car = 4 → 
  people_per_taxi = 6 → 
  people_per_van = 5 → 
  num_cars * people_per_car + num_taxis * people_per_taxi + num_vans * people_per_van = 58 := by
  sorry


end NUMINAMATH_CALUDE_hike_attendance_l338_33834


namespace NUMINAMATH_CALUDE_f_expression_range_f_transformed_is_l338_33879

/-- A quadratic function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The property that f(0) = 0 -/
axiom f_zero : f 0 = 0

/-- The property that f(x+1) = f(x) + x + 1 for all x -/
axiom f_next (x : ℝ) : f (x + 1) = f x + x + 1

/-- f is a quadratic function -/
axiom f_quadratic : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

/-- Theorem: f(x) = (1/2)x^2 + (1/2)x -/
theorem f_expression : ∀ x, f x = (1/2) * x^2 + (1/2) * x := sorry

/-- The range of y = f(x^2 - 2) -/
def range_f_transformed : Set ℝ := {y | ∃ x, y = f (x^2 - 2)}

/-- Theorem: The range of y = f(x^2 - 2) is [-1/8, +∞) -/
theorem range_f_transformed_is : range_f_transformed = {y | y ≥ -1/8} := sorry

end NUMINAMATH_CALUDE_f_expression_range_f_transformed_is_l338_33879


namespace NUMINAMATH_CALUDE_davids_crunches_l338_33872

/-- Given that David did 17 less crunches than Zachary, and Zachary did 62 crunches,
    prove that David did 45 crunches. -/
theorem davids_crunches (zachary_crunches : ℕ) (david_difference : ℤ) 
  (h1 : zachary_crunches = 62)
  (h2 : david_difference = -17) :
  zachary_crunches + david_difference = 45 :=
by sorry

end NUMINAMATH_CALUDE_davids_crunches_l338_33872


namespace NUMINAMATH_CALUDE_triangle_side_length_l338_33802

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area S_ABC = 1/4(a^2 + b^2 - c^2), b = 1, and a = √2, then c = 1. -/
theorem triangle_side_length (a b c : ℝ) (h_area : (a^2 + b^2 - c^2) / 4 = a * b * Real.sin (π/4) / 2)
  (h_b : b = 1) (h_a : a = Real.sqrt 2) : c = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l338_33802


namespace NUMINAMATH_CALUDE_sequence_difference_l338_33889

theorem sequence_difference (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, a (n + 1) - a n - n = 0) : 
  a 2017 - a 2016 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l338_33889


namespace NUMINAMATH_CALUDE_sqrt_two_cos_sin_equality_l338_33833

theorem sqrt_two_cos_sin_equality (x : ℝ) :
  Real.sqrt 2 * (Real.cos (2 * x))^4 - Real.sqrt 2 * (Real.sin (2 * x))^4 = Real.cos (2 * x) + Real.sin (2 * x) →
  ∃ k : ℤ, x = Real.pi * (4 * k - 1) / 8 := by
sorry

end NUMINAMATH_CALUDE_sqrt_two_cos_sin_equality_l338_33833


namespace NUMINAMATH_CALUDE_regular_polygon_with_900_degree_sum_l338_33840

theorem regular_polygon_with_900_degree_sum (n : ℕ) : 
  n > 2 → (n - 2) * 180 = 900 → n = 7 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_with_900_degree_sum_l338_33840


namespace NUMINAMATH_CALUDE_bob_arrival_probability_bob_arrival_probability_value_l338_33843

/-- The probability that Bob arrived before 3:45 PM given that Alice arrived after him,
    when both arrive randomly between 3:00 PM and 4:00 PM. -/
theorem bob_arrival_probability : ℝ :=
  let total_time := 60 -- minutes
  let bob_early_time := 45 -- minutes
  let total_area := (total_time ^ 2) / 2 -- area where Alice arrives after Bob
  let early_area := (bob_early_time ^ 2) / 2 -- area where Bob is early and Alice is after
  early_area / total_area

/-- The probability is equal to 9/16 -/
theorem bob_arrival_probability_value : bob_arrival_probability = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_bob_arrival_probability_bob_arrival_probability_value_l338_33843


namespace NUMINAMATH_CALUDE_watch_cost_calculation_l338_33871

/-- The cost of a watch, given the amount saved and the additional amount needed. -/
def watch_cost (saved : ℕ) (additional_needed : ℕ) : ℕ :=
  saved + additional_needed

/-- Theorem: The cost of the watch is $55, given Connie saved $39 and needs $16 more. -/
theorem watch_cost_calculation : watch_cost 39 16 = 55 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_calculation_l338_33871


namespace NUMINAMATH_CALUDE_total_peaches_sum_l338_33809

/-- The total number of peaches after picking more -/
def total_peaches (initial : Float) (picked : Float) : Float :=
  initial + picked

/-- Theorem: The total number of peaches is the sum of initial and picked peaches -/
theorem total_peaches_sum (initial picked : Float) :
  total_peaches initial picked = initial + picked := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_sum_l338_33809


namespace NUMINAMATH_CALUDE_system_solution_l338_33894

theorem system_solution :
  ∃ (x y z : ℚ),
    (7 * x - 3 * y + 2 * z = 4) ∧
    (2 * x + 8 * y - z = 1) ∧
    (3 * x - 4 * y + 5 * z = 7) ∧
    (x = 1262 / 913) ∧
    (y = -59 / 83) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l338_33894


namespace NUMINAMATH_CALUDE_contractor_fine_calculation_l338_33808

/-- Proves that the fine for each day of absence is Rs. 7.5 given the conditions of the contractor problem -/
theorem contractor_fine_calculation (total_days : ℕ) (daily_wage : ℚ) (total_received : ℚ) (absent_days : ℕ) :
  total_days = 30 →
  daily_wage = 25 →
  total_received = 685 →
  absent_days = 2 →
  ∃ (daily_fine : ℚ), daily_fine = 7.5 ∧
    total_received = daily_wage * (total_days - absent_days) - daily_fine * absent_days :=
by sorry

end NUMINAMATH_CALUDE_contractor_fine_calculation_l338_33808


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l338_33886

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ) :
  failed_english = 44 →
  failed_both = 22 →
  passed_both = 44 →
  ∃ failed_hindi : ℝ, failed_hindi = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l338_33886


namespace NUMINAMATH_CALUDE_parallelogram_properties_independence_l338_33829

/-- A parallelogram with potentially equal sides and/or right angles -/
structure Parallelogram where
  has_equal_sides : Bool
  has_right_angles : Bool

/-- Theorem: There exist parallelograms with equal sides but not right angles, 
    and parallelograms with right angles but not equal sides -/
theorem parallelogram_properties_independence :
  ∃ (p q : Parallelogram), 
    (p.has_equal_sides ∧ ¬p.has_right_angles) ∧
    (q.has_right_angles ∧ ¬q.has_equal_sides) :=
by
  sorry


end NUMINAMATH_CALUDE_parallelogram_properties_independence_l338_33829


namespace NUMINAMATH_CALUDE_max_distance_for_given_tires_l338_33805

/-- Represents the maximum distance a car can travel with tire switching -/
def max_distance (front_tire_life : ℕ) (rear_tire_life : ℕ) : ℕ :=
  min front_tire_life rear_tire_life

/-- Theorem stating the maximum distance a car can travel with specific tire lifespans -/
theorem max_distance_for_given_tires :
  max_distance 42000 56000 = 42000 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_for_given_tires_l338_33805


namespace NUMINAMATH_CALUDE_dog_to_cats_ratio_is_two_to_one_l338_33892

/-- The weight of Christine's first cat in pounds -/
def cat1_weight : ℕ := 7

/-- The weight of Christine's second cat in pounds -/
def cat2_weight : ℕ := 10

/-- The combined weight of Christine's cats in pounds -/
def cats_combined_weight : ℕ := cat1_weight + cat2_weight

/-- The weight of Christine's dog in pounds -/
def dog_weight : ℕ := 34

/-- The ratio of the dog's weight to the combined weight of the cats -/
def dog_to_cats_ratio : ℚ := dog_weight / cats_combined_weight

theorem dog_to_cats_ratio_is_two_to_one :
  dog_to_cats_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_dog_to_cats_ratio_is_two_to_one_l338_33892


namespace NUMINAMATH_CALUDE_pigeonhole_divisibility_l338_33881

theorem pigeonhole_divisibility (n : ℕ+) (a : Fin (n + 1) → ℤ) :
  ∃ i j : Fin (n + 1), i ≠ j ∧ (n : ℤ) ∣ (a i - a j) := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_divisibility_l338_33881


namespace NUMINAMATH_CALUDE_power_sum_of_i_l338_33875

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^47 = -2*i := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l338_33875


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l338_33895

theorem sum_of_cubes_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum_squares : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l338_33895


namespace NUMINAMATH_CALUDE_union_equals_reals_subset_of_complement_l338_33861

-- Define the sets A and B
def A : Set ℝ := {x | x < 0 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2*a}

-- Theorem for part (1)
theorem union_equals_reals (a : ℝ) : 
  A ∪ B a = Set.univ ↔ a ∈ Set.Iic 0 :=
sorry

-- Theorem for part (2)
theorem subset_of_complement (a : ℝ) :
  B a ⊆ -A ↔ a ∈ {x | x ≥ 1/2} :=
sorry

end NUMINAMATH_CALUDE_union_equals_reals_subset_of_complement_l338_33861


namespace NUMINAMATH_CALUDE_hidden_numbers_puzzle_l338_33891

theorem hidden_numbers_puzzle (x y : ℕ) :
  x^2 + y^2 = 65 ∧
  x + y ≥ 10 ∧
  (∀ a b : ℕ, a^2 + b^2 = 65 ∧ a + b ≥ 10 → (a = x ∧ b = y) ∨ (a = y ∧ b = x)) →
  ((x = 7 ∧ y = 4) ∨ (x = 4 ∧ y = 7)) :=
by sorry

end NUMINAMATH_CALUDE_hidden_numbers_puzzle_l338_33891


namespace NUMINAMATH_CALUDE_brother_father_age_ratio_l338_33859

/-- Represents the ages of family members and total family age --/
structure FamilyAges where
  total : ℕ
  father : ℕ
  mother : ℕ
  sister : ℕ
  kaydence : ℕ

/-- Theorem stating the ratio of brother's age to father's age --/
theorem brother_father_age_ratio (f : FamilyAges) 
  (h1 : f.total = 200)
  (h2 : f.father = 60)
  (h3 : f.mother = f.father - 2)
  (h4 : f.sister = 40)
  (h5 : f.kaydence = 12) :
  ∃ (brother_age : ℕ), 
    brother_age = f.total - (f.father + f.mother + f.sister + f.kaydence) ∧
    2 * brother_age = f.father :=
by
  sorry

end NUMINAMATH_CALUDE_brother_father_age_ratio_l338_33859


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l338_33816

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l338_33816


namespace NUMINAMATH_CALUDE_combinations_permutations_relation_l338_33863

/-- The number of combinations of n elements taken k at a time -/
def C (n k : ℕ) : ℕ := sorry

/-- The number of permutations of k elements from an n-element set -/
def A (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of combinations is equal to the number of permutations divided by k factorial -/
theorem combinations_permutations_relation (n k : ℕ) : C n k = A n k / k! := by
  sorry

end NUMINAMATH_CALUDE_combinations_permutations_relation_l338_33863


namespace NUMINAMATH_CALUDE_min_shots_to_hit_ship_l338_33855

/-- Represents a point on the game board -/
structure Point where
  x : Fin 10
  y : Fin 10

/-- Represents a ship on the game board -/
inductive Ship
  | Horizontal : Fin 10 → Fin 7 → Ship
  | Vertical : Fin 7 → Fin 10 → Ship

/-- Checks if a point is on a ship -/
def pointOnShip (p : Point) (s : Ship) : Prop :=
  match s with
  | Ship.Horizontal row col => p.y = row ∧ col ≤ p.x ∧ p.x < col + 4
  | Ship.Vertical row col => p.x = col ∧ row ≤ p.y ∧ p.y < row + 4

/-- The theorem to be proved -/
theorem min_shots_to_hit_ship :
  ∃ (shots : Finset Point),
    shots.card = 14 ∧
    ∀ (s : Ship), ∃ (p : Point), p ∈ shots ∧ pointOnShip p s ∧
    ∀ (shots' : Finset Point),
      shots'.card < 14 →
      ∃ (s : Ship), ∀ (p : Point), p ∈ shots' → ¬pointOnShip p s :=
by sorry

end NUMINAMATH_CALUDE_min_shots_to_hit_ship_l338_33855


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_system_l338_33801

theorem smallest_solution_congruence_system (x : ℕ) : x = 1309 ↔ 
  (x > 0) ∧
  (3 * x ≡ 9 [MOD 12]) ∧ 
  (5 * x + 4 ≡ 14 [MOD 7]) ∧ 
  (4 * x - 3 ≡ 2 * x + 5 [MOD 17]) ∧ 
  (x ≡ 4 [MOD 11]) ∧
  (∀ y : ℕ, y > 0 → 
    (3 * y ≡ 9 [MOD 12]) → 
    (5 * y + 4 ≡ 14 [MOD 7]) → 
    (4 * y - 3 ≡ 2 * y + 5 [MOD 17]) → 
    (y ≡ 4 [MOD 11]) → 
    y ≥ x) :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_system_l338_33801


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l338_33870

open Set

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- State the theorem
theorem solution_set_of_inequality 
  (h1 : ∀ x, f' x < f x) 
  (h2 : f 1 = Real.exp 1) :
  {x : ℝ | f (Real.log x) > x} = Ioo 0 (Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l338_33870


namespace NUMINAMATH_CALUDE_problem_statement_l338_33857

theorem problem_statement (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) :
  (1/3) * x^8 * y^5 - x^2 * y = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l338_33857


namespace NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_45_l338_33865

theorem one_and_two_thirds_of_x_is_45 (x : ℚ) : (5 / 3 : ℚ) * x = 45 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_45_l338_33865


namespace NUMINAMATH_CALUDE_car_speed_problem_l338_33831

/-- Proves that given the conditions of the car problem, the speed of Car X is approximately 33.87 mph -/
theorem car_speed_problem (speed_y : ℝ) (time_diff : ℝ) (distance_x : ℝ) :
  speed_y = 42 →
  time_diff = 72 / 60 →
  distance_x = 210 →
  ∃ (speed_x : ℝ), 
    speed_x > 0 ∧ 
    speed_x * (distance_x / speed_y + time_diff) = distance_x ∧ 
    (abs (speed_x - 33.87) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l338_33831


namespace NUMINAMATH_CALUDE_min_value_expression_l338_33874

theorem min_value_expression (x y : ℝ) :
  x^2 - 6*x*Real.sin y - 9*(Real.cos y)^2 ≥ -9 ∧
  ∃ (x y : ℝ), x^2 - 6*x*Real.sin y - 9*(Real.cos y)^2 = -9 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l338_33874


namespace NUMINAMATH_CALUDE_triangle_inequality_reciprocal_l338_33869

theorem triangle_inequality_reciprocal (a b c : ℝ) 
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  1 / (a + c) + 1 / (b + c) > 1 / (a + b) ∧
  1 / (a + c) + 1 / (a + b) > 1 / (b + c) ∧
  1 / (b + c) + 1 / (a + b) > 1 / (a + c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_reciprocal_l338_33869


namespace NUMINAMATH_CALUDE_triangle_height_and_median_l338_33835

-- Define the triangle ABC
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (4, 5)
def C : ℝ × ℝ := (0, 7)

-- Define the height from A to BC
def height_equation (x y : ℝ) : Prop := 2 * x - y - 6 = 0

-- Define the median from A to BC
def median_equation (x y : ℝ) : Prop := 6 * x + y - 18 = 0

theorem triangle_height_and_median :
  (∀ x y : ℝ, height_equation x y ↔ 
    (y - A.2) / (x - A.1) = -1 / (B.1 - C.1) / (B.2 - C.2)) ∧
  (∀ x y : ℝ, median_equation x y ↔ 
    (y - A.2) / (x - A.1) = ((B.2 + C.2) / 2 - A.2) / ((B.1 + C.1) / 2 - A.1)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_and_median_l338_33835


namespace NUMINAMATH_CALUDE_bread_cost_is_1_1_l338_33828

/-- The cost of each bread given the conditions of the problem -/
def bread_cost (total_breads : ℕ) (num_people : ℕ) (compensation : ℚ) : ℚ :=
  (compensation * 2 * num_people) / total_breads

/-- Theorem stating that the cost of each bread is 1.1 yuan -/
theorem bread_cost_is_1_1 :
  bread_cost 12 3 (22/10) = 11/10 := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_is_1_1_l338_33828


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l338_33836

/-- Given that 3/4 of 12 bananas are worth 9 oranges, 
    prove that 1/3 of 9 bananas are worth 3 oranges -/
theorem banana_orange_equivalence 
  (h : (3/4 : ℚ) * 12 * (banana_value : ℚ) = 9 * (orange_value : ℚ)) :
  (1/3 : ℚ) * 9 * banana_value = 3 * orange_value :=
by sorry


end NUMINAMATH_CALUDE_banana_orange_equivalence_l338_33836


namespace NUMINAMATH_CALUDE_f_composition_at_pi_l338_33842

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x - 1 else Real.sin x - 2

theorem f_composition_at_pi : f (f Real.pi) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_at_pi_l338_33842


namespace NUMINAMATH_CALUDE_square_roots_problem_l338_33896

theorem square_roots_problem (a : ℝ) (n : ℝ) :
  n > 0 ∧ 
  (∃ x y : ℝ, x * x = n ∧ y * y = n ∧ x = a ∧ y = 2 * a - 6) →
  a = 6 ∧ 
  n = 36 ∧
  (∃ b : ℝ, b * b * b = 10 * 2 + 7 ∧ b = 3) :=
by sorry

end NUMINAMATH_CALUDE_square_roots_problem_l338_33896


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l338_33803

/-- A line passing through the point (-3, -1) and parallel to x - 3y - 1 = 0 has the equation x - 3y = 0 -/
theorem parallel_line_through_point : 
  ∀ (l : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ x - 3*y = 0) →
    (-3, -1) ∈ l →
    (∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ (t : ℝ), x = t ∧ y = (t - 1) / 3) →
    True :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l338_33803


namespace NUMINAMATH_CALUDE_roots_sum_squared_plus_double_plus_other_l338_33873

theorem roots_sum_squared_plus_double_plus_other (a b : ℝ) : 
  a^2 + a - 2023 = 0 → b^2 + b - 2023 = 0 → a^2 + 2*a + b = 2022 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_squared_plus_double_plus_other_l338_33873


namespace NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l338_33887

theorem cone_sphere_volume_ratio (r : ℝ) (h : ℝ) : 
  r > 0 → h = 2 * r → 
  (1 / 3 * π * r^2 * h) / (4 / 3 * π * r^3) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_ratio_l338_33887


namespace NUMINAMATH_CALUDE_apple_tree_yield_l338_33827

theorem apple_tree_yield (total : ℕ) : 
  (total / 5 : ℚ) +             -- First day
  (2 * (total / 5) : ℚ) +       -- Second day
  (total / 5 + 20 : ℚ) +        -- Third day
  20 = total →                  -- Remaining apples
  total = 200 := by
sorry

end NUMINAMATH_CALUDE_apple_tree_yield_l338_33827


namespace NUMINAMATH_CALUDE_difference_of_fractions_numerator_l338_33800

theorem difference_of_fractions_numerator : 
  let a := 2024
  let b := 2023
  let diff := a / b - b / a
  let p := (a^2 - b^2) / (a * b)
  p = 4047 := by sorry

end NUMINAMATH_CALUDE_difference_of_fractions_numerator_l338_33800


namespace NUMINAMATH_CALUDE_trip_time_difference_car_trip_time_difference_l338_33814

/-- Calculates the time difference in minutes between two trips of different distances traveled at the same speed. -/
theorem trip_time_difference (speed : ℝ) (distance1 : ℝ) (distance2 : ℝ) : 
  speed > 0 → distance1 > 0 → distance2 > 0 →
  (distance2 - distance1) / speed * 60 = (distance2 / speed - distance1 / speed) * 60 := by
  sorry

/-- Proves that the difference in time between a 420-mile trip and a 360-mile trip, both traveled at 60 miles per hour, is 60 minutes. -/
theorem car_trip_time_difference : 
  let speed : ℝ := 60
  let distance1 : ℝ := 360
  let distance2 : ℝ := 420
  (distance2 - distance1) / speed * 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_car_trip_time_difference_l338_33814


namespace NUMINAMATH_CALUDE_fourth_rectangle_perimeter_l338_33897

theorem fourth_rectangle_perimeter 
  (a b c d : ℝ) 
  (h1 : 2 * (c + b) = 6) 
  (h2 : 2 * (a + c) = 10) 
  (h3 : 2 * (a + d) = 12) : 
  2 * (b + d) = 8 := by
sorry

end NUMINAMATH_CALUDE_fourth_rectangle_perimeter_l338_33897


namespace NUMINAMATH_CALUDE_bleacher_sets_l338_33838

theorem bleacher_sets (total_fans : ℕ) (fans_per_set : ℕ) (h1 : total_fans = 2436) (h2 : fans_per_set = 812) :
  total_fans / fans_per_set = 3 :=
by sorry

end NUMINAMATH_CALUDE_bleacher_sets_l338_33838


namespace NUMINAMATH_CALUDE_stratified_sampling_most_suitable_l338_33868

/-- Represents the age groups in the population -/
inductive AgeGroup
  | Elderly
  | MiddleAged
  | Young

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Other

/-- Represents the population structure -/
structure Population where
  total : Nat
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Determines the most suitable sampling method given a population and sample size -/
def mostSuitableSamplingMethod (pop : Population) (sampleSize : Nat) : SamplingMethod :=
  sorry

/-- The theorem stating that stratified sampling is the most suitable method for the given population and sample size -/
theorem stratified_sampling_most_suitable :
  let pop : Population := { total := 163, elderly := 28, middleAged := 54, young := 81 }
  let sampleSize : Nat := 36
  mostSuitableSamplingMethod pop sampleSize = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_suitable_l338_33868


namespace NUMINAMATH_CALUDE_ratio_is_five_thirds_l338_33846

/-- Given a diagram with triangles, some shaded and some unshaded -/
structure TriangleDiagram where
  shaded : ℕ
  unshaded : ℕ

/-- The ratio of shaded to unshaded triangles -/
def shaded_unshaded_ratio (d : TriangleDiagram) : ℚ :=
  d.shaded / d.unshaded

theorem ratio_is_five_thirds (d : TriangleDiagram) 
  (h1 : d.shaded = 5) 
  (h2 : d.unshaded = 3) : 
  shaded_unshaded_ratio d = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_is_five_thirds_l338_33846


namespace NUMINAMATH_CALUDE_sum_has_48_divisors_l338_33877

def sum_of_numbers : ℕ := 9240 + 8820

theorem sum_has_48_divisors : Nat.card (Nat.divisors sum_of_numbers) = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_has_48_divisors_l338_33877


namespace NUMINAMATH_CALUDE_gathering_attendance_l338_33880

theorem gathering_attendance (W S B : ℕ) (hW : W = 26) (hS : S = 22) (hB : B = 17) :
  W + S - B = 31 := by sorry

end NUMINAMATH_CALUDE_gathering_attendance_l338_33880


namespace NUMINAMATH_CALUDE_max_k_value_l338_33853

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, (1/m + 2/(1-2*m) ≥ k) → k ≤ 8) ∧
  (∃ k : ℝ, k = 8 ∧ 1/m + 2/(1-2*m) ≥ k) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l338_33853


namespace NUMINAMATH_CALUDE_team_a_min_wins_l338_33890

theorem team_a_min_wins (total_games : ℕ) (lost_games : ℕ) (min_points : ℕ) 
  (win_points draw_points lose_points : ℕ) :
  total_games = 5 →
  lost_games = 1 →
  min_points = 7 →
  win_points = 3 →
  draw_points = 1 →
  lose_points = 0 →
  ∃ (won_games : ℕ),
    won_games ≥ 2 ∧
    won_games + lost_games ≤ total_games ∧
    won_games * win_points + (total_games - won_games - lost_games) * draw_points > min_points :=
by sorry

end NUMINAMATH_CALUDE_team_a_min_wins_l338_33890


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l338_33860

/-- The diagonal of a rectangle with length 40√3 cm and width 30√3 cm is 50√3 cm. -/
theorem rectangle_diagonal : 
  let length : ℝ := 40 * Real.sqrt 3
  let width : ℝ := 30 * Real.sqrt 3
  let diagonal : ℝ := Real.sqrt (length^2 + width^2)
  diagonal = 50 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l338_33860


namespace NUMINAMATH_CALUDE_min_c_for_unique_solution_l338_33878

/-- The system of equations -/
def system (x y c : ℝ) : Prop :=
  8 * (x + 7)^4 + (y - 4)^4 = c ∧ (x + 4)^4 + 8 * (y - 7)^4 = c

/-- The existence of a unique solution for the system -/
def has_unique_solution (c : ℝ) : Prop :=
  ∃! x y, system x y c

/-- The theorem stating the minimum value of c for a unique solution -/
theorem min_c_for_unique_solution :
  ∀ c, has_unique_solution c → c ≥ 24 ∧ has_unique_solution 24 :=
sorry

end NUMINAMATH_CALUDE_min_c_for_unique_solution_l338_33878


namespace NUMINAMATH_CALUDE_inequality_solution_set_l338_33898

def solution_set (x : ℝ) : Prop :=
  x ∈ Set.union (Set.Ioo 0 1) (Set.union (Set.Ioo 1 (2 ^ (5/7))) (Set.Ioi 4))

theorem inequality_solution_set (x : ℝ) :
  (|1 / Real.log (1/2 * x) + 2| > 3/2) ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l338_33898


namespace NUMINAMATH_CALUDE_total_noodles_and_pirates_l338_33883

theorem total_noodles_and_pirates (pirates : ℕ) (noodle_difference : ℕ) : 
  pirates = 45 → noodle_difference = 7 → pirates + (pirates - noodle_difference) = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_noodles_and_pirates_l338_33883


namespace NUMINAMATH_CALUDE_marin_apples_l338_33847

theorem marin_apples (donald_apples : ℕ) (total_apples : ℕ) 
  (h1 : donald_apples = 2)
  (h2 : total_apples = 11) :
  ∃ marin_apples : ℕ, marin_apples + donald_apples = total_apples ∧ marin_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_marin_apples_l338_33847


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l338_33823

-- Define the given line
def given_line (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define the point through which the perpendicular line passes
def point : ℝ × ℝ := (-1, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  (∀ x y : ℝ, perpendicular_line x y ↔ 
    (∃ m b : ℝ, y = m*x + b ∧ 
      (perpendicular_line point.1 point.2) ∧
      (m * (1/2) = -1))) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l338_33823


namespace NUMINAMATH_CALUDE_relationship_abxy_l338_33832

theorem relationship_abxy (a b x y : ℚ) 
  (eq1 : x + y = a + b) 
  (ineq1 : y - x < a - b) 
  (ineq2 : b > a) : 
  y < a ∧ a < b ∧ b < x :=
sorry

end NUMINAMATH_CALUDE_relationship_abxy_l338_33832


namespace NUMINAMATH_CALUDE_line_symmetry_l338_33884

-- Define the original line
def original_line (x y : ℝ) : Prop := 2 * x - y + 4 = 0

-- Define the axis of symmetry
def axis_of_symmetry (x : ℝ) : Prop := x = 1

-- Define the symmetric line
def symmetric_line (x y : ℝ) : Prop := 2 * x + y - 8 = 0

-- Theorem statement
theorem line_symmetry :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), original_line x₀ y₀ ∧ axis_of_symmetry ((x + x₀) / 2)) →
  symmetric_line x y :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l338_33884


namespace NUMINAMATH_CALUDE_folded_rectangle_EF_length_l338_33862

/-- A rectangle ABCD with side lengths AB = 4 and BC = 8 is folded so that A and C coincide,
    forming a new shape ABEFD. This function calculates the length of EF. -/
def foldedRectangleEFLength (AB BC : ℝ) : ℝ :=
  4

/-- Theorem stating that for a rectangle ABCD with AB = 4 and BC = 8, when folded so that
    A and C coincide to form ABEFD, the length of EF is 4. -/
theorem folded_rectangle_EF_length :
  foldedRectangleEFLength 4 8 = 4 := by
  sorry

#check folded_rectangle_EF_length

end NUMINAMATH_CALUDE_folded_rectangle_EF_length_l338_33862


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l338_33888

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = n^2 + 3n,
    prove that the sum of the 6th, 7th, and 8th terms is 48. -/
theorem sum_of_specific_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
    (h : ∀ n, S n = n^2 + 3*n) :
  a 6 + a 7 + a 8 = 48 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l338_33888


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l338_33885

def M : Set ℤ := {0, 1, 2, -1}
def N : Set ℤ := {0, 1, 2, 3}

theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l338_33885


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l338_33813

theorem contrapositive_equivalence (a b : ℝ) :
  (((a + b ≠ 3) → (a ≠ 1 ∨ b ≠ 2)) ↔ ((a = 1 ∧ b = 2) → (a + b = 3))) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l338_33813


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l338_33844

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The general term of the sequence. -/
def a (n : ℕ) (p q : ℝ) : ℝ := p * n + q

theorem sequence_is_arithmetic (p q : ℝ) :
  IsArithmeticSequence (a · p q) := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l338_33844


namespace NUMINAMATH_CALUDE_area_of_region_l338_33820

-- Define the region
def region (x y : ℝ) : Prop :=
  |x - 2*y^2| + x + 2*y^2 ≤ 8 - 4*y

-- Define symmetry about Y-axis
def symmetricAboutYAxis (S : Set (ℝ × ℝ)) : Prop :=
  ∀ x y, (x, y) ∈ S ↔ (-x, y) ∈ S

-- Theorem statement
theorem area_of_region :
  ∃ S : Set (ℝ × ℝ),
    (∀ x y, (x, y) ∈ S ↔ region x y) ∧
    symmetricAboutYAxis S ∧
    MeasureTheory.volume S = 30 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l338_33820


namespace NUMINAMATH_CALUDE_equation_solution_set_l338_33841

theorem equation_solution_set : 
  {x : ℝ | |x^2 - 5*x + 6| = x + 2} = {3 + Real.sqrt 5, 3 - Real.sqrt 5} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_set_l338_33841


namespace NUMINAMATH_CALUDE_inequality_proof_l338_33804

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_abc : a + b + c = 1) :
  (7 + 2*b) / (1 + a) + (7 + 2*c) / (1 + b) + (7 + 2*a) / (1 + c) ≥ 69/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l338_33804


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l338_33826

theorem rectangle_perimeter (w : ℝ) (h1 : w > 0) :
  let l := 3 * w
  let d := 8 * Real.sqrt 10
  d^2 = l^2 + w^2 →
  2 * l + 2 * w = 64 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l338_33826


namespace NUMINAMATH_CALUDE_extremum_when_a_zero_range_of_m_l338_33811

noncomputable section

def g (a x : ℝ) : ℝ := (2 - a) * Real.log x

def h (a x : ℝ) : ℝ := Real.log x + a * x^2

def f (a x : ℝ) : ℝ := g a x + (deriv (h a)) x

theorem extremum_when_a_zero :
  let f₀ := f 0
  ∃ (x_min : ℝ), x_min = 1/2 ∧ 
    (∀ x > 0, f₀ x ≥ f₀ x_min) ∧
    f₀ x_min = 2 - 2 * Real.log 2 ∧
    (∀ M : ℝ, ∃ x > 0, f₀ x > M) :=
sorry

theorem range_of_m (a : ℝ) (h : -8 < a ∧ a < -2) :
  let m_lower := 2 / (3 * Real.exp 2) - 4
  ∀ m > m_lower,
    ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 1 3 →
      |f a x₁ - f a x₂| > (m + Real.log 3) * a - 2 * Real.log 3 + 2/3 * Real.log (-a) :=
sorry

end NUMINAMATH_CALUDE_extremum_when_a_zero_range_of_m_l338_33811


namespace NUMINAMATH_CALUDE_negative_square_times_a_l338_33845

theorem negative_square_times_a (a : ℝ) : -a^2 * a = -a^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_times_a_l338_33845


namespace NUMINAMATH_CALUDE_abs_sum_equals_eight_l338_33807

theorem abs_sum_equals_eight (x : ℝ) (θ : ℝ) (h : Real.log x / Real.log 3 = 1 + Real.sin θ) :
  |x - 1| + |x - 9| = 8 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_equals_eight_l338_33807


namespace NUMINAMATH_CALUDE_floor_squared_sum_four_l338_33882

theorem floor_squared_sum_four (x y : ℝ) : 
  (Int.floor x)^2 + (Int.floor y)^2 = 4 ↔ 
    ((2 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 1) ∨
     (0 ≤ x ∧ x < 1 ∧ 2 ≤ y ∧ y < 3) ∨
     (-2 ≤ x ∧ x < -1 ∧ 0 ≤ y ∧ y < 1) ∨
     (0 ≤ x ∧ x < 1 ∧ -2 ≤ y ∧ y < -1)) :=
by sorry

end NUMINAMATH_CALUDE_floor_squared_sum_four_l338_33882


namespace NUMINAMATH_CALUDE_number_ordering_l338_33858

theorem number_ordering : 7^8 < 3^15 ∧ 3^15 < 4^12 ∧ 4^12 < 8^10 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l338_33858


namespace NUMINAMATH_CALUDE_asymptotes_sum_l338_33822

theorem asymptotes_sum (A B C : ℤ) : 
  (∀ x, x^3 + A*x^2 + B*x + C = (x + 1)*(x - 3)*(x - 4)) → 
  A + B + C = 11 := by
sorry

end NUMINAMATH_CALUDE_asymptotes_sum_l338_33822


namespace NUMINAMATH_CALUDE_milk_expense_l338_33852

/-- Given Mr. Kishore's savings and expenses, prove the amount spent on milk -/
theorem milk_expense (savings : ℕ) (rent groceries education petrol misc : ℕ) 
  (h1 : savings = 2350)
  (h2 : rent = 5000)
  (h3 : groceries = 4500)
  (h4 : education = 2500)
  (h5 : petrol = 2000)
  (h6 : misc = 5650)
  (h7 : savings = (1 / 10 : ℚ) * (savings / (1 / 10 : ℚ))) :
  ∃ (milk : ℕ), milk = 1500 ∧ 
    (9 / 10 : ℚ) * (savings / (1 / 10 : ℚ)) = 
    (rent + groceries + education + petrol + misc + milk) :=
by sorry

end NUMINAMATH_CALUDE_milk_expense_l338_33852


namespace NUMINAMATH_CALUDE_joseph_total_distance_l338_33818

/-- Joseph's daily running distance in meters -/
def daily_distance : ℕ := 900

/-- Number of days Joseph ran -/
def days_run : ℕ := 3

/-- Total distance Joseph ran -/
def total_distance : ℕ := daily_distance * days_run

/-- Theorem: Joseph's total running distance is 2700 meters -/
theorem joseph_total_distance : total_distance = 2700 := by
  sorry

end NUMINAMATH_CALUDE_joseph_total_distance_l338_33818


namespace NUMINAMATH_CALUDE_storm_water_deposit_l338_33867

theorem storm_water_deposit (original_content : ℝ) (pre_storm_percentage : ℝ) (post_storm_percentage : ℝ) : 
  original_content = 200 * 10^9 →
  pre_storm_percentage = 0.5 →
  post_storm_percentage = 0.8 →
  (post_storm_percentage * (original_content / pre_storm_percentage)) - original_content = 120 * 10^9 := by
sorry

end NUMINAMATH_CALUDE_storm_water_deposit_l338_33867


namespace NUMINAMATH_CALUDE_prob_one_rectification_prob_at_least_one_closed_l338_33854

-- Define the number of canteens
def num_canteens : ℕ := 4

-- Define the probability of passing inspection before rectification
def prob_pass_before : ℝ := 0.5

-- Define the probability of passing inspection after rectification
def prob_pass_after : ℝ := 0.8

-- Theorem for the probability that exactly one canteen needs rectification
theorem prob_one_rectification :
  (num_canteens.choose 1 : ℝ) * prob_pass_before^(num_canteens - 1) * (1 - prob_pass_before) = 0.25 := by
  sorry

-- Theorem for the probability that at least one canteen is closed
theorem prob_at_least_one_closed :
  1 - (1 - (1 - prob_pass_before) * (1 - prob_pass_after))^num_canteens = 0.34 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_rectification_prob_at_least_one_closed_l338_33854


namespace NUMINAMATH_CALUDE_exists_graph_clique_lt_chromatic_l338_33899

/-- A graph type with vertices and edges -/
structure Graph where
  V : Type
  E : V → V → Prop

/-- The clique number of a graph -/
def cliqueNumber (G : Graph) : ℕ := sorry

/-- The chromatic number of a graph -/
def chromaticNumber (G : Graph) : ℕ := sorry

/-- Theorem: There exists a graph with clique number smaller than its chromatic number -/
theorem exists_graph_clique_lt_chromatic :
  ∃ (G : Graph), cliqueNumber G < chromaticNumber G := by sorry

end NUMINAMATH_CALUDE_exists_graph_clique_lt_chromatic_l338_33899


namespace NUMINAMATH_CALUDE_fraction_sum_bound_l338_33819

theorem fraction_sum_bound (a b c : ℕ+) (h : (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ < 1) :
  (a : ℚ)⁻¹ + (b : ℚ)⁻¹ + (c : ℚ)⁻¹ ≤ 41 / 42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_bound_l338_33819


namespace NUMINAMATH_CALUDE_complex_equation_solution_l338_33851

theorem complex_equation_solution (z : ℂ) (h : Complex.I * (z + 2 * Complex.I) = 1) : z = -3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l338_33851


namespace NUMINAMATH_CALUDE_prism_volume_l338_33849

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 10)
  (h2 : b * c = 15)
  (h3 : c * a = 18) :
  a * b * c = 30 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l338_33849


namespace NUMINAMATH_CALUDE_circle_equation_l338_33824

/-- The equation of a circle passing through points A(1, -1) and B(-1, 1) with center on the line x + y - 2 = 0 -/
theorem circle_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (-1, 1)
  let center_line (p : ℝ × ℝ) := p.1 + p.2 - 2 = 0
  let circle_eq (p : ℝ × ℝ) := (p.1 - 1)^2 + (p.2 - 1)^2 = 4
  let on_circle (p : ℝ × ℝ) := circle_eq p
  ∃ (c : ℝ × ℝ), 
    center_line c ∧ 
    on_circle A ∧ 
    on_circle B ∧ 
    on_circle (x, y) ↔ circle_eq (x, y) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l338_33824


namespace NUMINAMATH_CALUDE_triangle_3_4_5_l338_33825

/-- A triangle can be formed from three line segments if the sum of the lengths of any two sides is greater than the length of the remaining side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that the line segments 3, 4, and 5 can form a triangle. -/
theorem triangle_3_4_5 : can_form_triangle 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_3_4_5_l338_33825


namespace NUMINAMATH_CALUDE_f_less_than_g_implies_a_bound_l338_33812

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 1

noncomputable def g (x : ℝ) : ℝ := Real.log x - x / 4 + 3 / (4 * x)

theorem f_less_than_g_implies_a_bound (a : ℝ) :
  (∀ x₁ > 0, ∃ x₂ > 1, f a x₁ < g x₂) →
  a > (1 / 3) * Real.exp (1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_f_less_than_g_implies_a_bound_l338_33812


namespace NUMINAMATH_CALUDE_smallest_value_w_cube_plus_z_cube_l338_33864

theorem smallest_value_w_cube_plus_z_cube (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 18) :
  Complex.abs (w^3 + z^3) = 50 := by
sorry

end NUMINAMATH_CALUDE_smallest_value_w_cube_plus_z_cube_l338_33864


namespace NUMINAMATH_CALUDE_positive_distinct_solutions_l338_33848

/-- Given a system of equations, prove the necessary and sufficient conditions for positive and distinct solutions -/
theorem positive_distinct_solutions (a b x y z : ℝ) :
  x + y + z = a →
  x^2 + y^2 + z^2 = b^2 →
  x * y = z^2 →
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ (a > 0 ∧ b^2 < a^2 ∧ a^2 < 3 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_positive_distinct_solutions_l338_33848


namespace NUMINAMATH_CALUDE_queen_spade_probability_l338_33830

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)

/-- Represents a Queen card -/
def is_queen (card : Nat × Nat) : Prop := card.1 = 12

/-- Represents a Spade card -/
def is_spade (card : Nat × Nat) : Prop := card.2 = 3

/-- The probability of drawing a Queen as the first card and a Spade as the second card -/
def queen_spade_prob (d : Deck) : ℚ :=
  18 / 221

theorem queen_spade_probability (d : Deck) :
  queen_spade_prob d = 18 / 221 :=
sorry

end NUMINAMATH_CALUDE_queen_spade_probability_l338_33830


namespace NUMINAMATH_CALUDE_graph_not_simple_l338_33893

-- Define the equation
def equation (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2 + 1

-- Define the set of points satisfying the equation
def graph : Set (ℝ × ℝ) := {p | equation p.1 p.2}

-- Theorem stating that the graph is not any of the given options
theorem graph_not_simple : 
  (graph ≠ ∅) ∧ 
  (∃ p q : ℝ × ℝ, p ∈ graph ∧ q ∈ graph ∧ p ≠ q) ∧ 
  (¬∃ a b : ℝ, graph = {p | p.2 = a * p.1 + b} ∪ {p | p.2 = a * p.1 + (b + 1)}) ∧
  (¬∃ c r : ℝ, graph = {p | (p.1 - c)^2 + (p.2 - c)^2 = r^2}) ∧
  (graph ≠ Set.univ) :=
sorry

end NUMINAMATH_CALUDE_graph_not_simple_l338_33893


namespace NUMINAMATH_CALUDE_or_true_if_one_true_l338_33850

theorem or_true_if_one_true (p q : Prop) (h : p ∨ q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_or_true_if_one_true_l338_33850


namespace NUMINAMATH_CALUDE_rational_fraction_implication_l338_33810

theorem rational_fraction_implication (x : ℝ) :
  (∃ a : ℚ, x / (x^2 + x + 1) = a) →
  (∃ b : ℚ, x^2 / (x^4 + x^2 + 1) = b) :=
by sorry

end NUMINAMATH_CALUDE_rational_fraction_implication_l338_33810
