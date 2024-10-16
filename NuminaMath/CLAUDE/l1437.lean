import Mathlib

namespace NUMINAMATH_CALUDE_point_c_value_l1437_143738

/-- Represents a point on a number line --/
structure Point where
  value : ℝ

/-- The distance between two points on a number line --/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_c_value (a b c : Point) :
  a.value = -1 →
  distance a b = 11 →
  b.value > a.value →
  distance b c = 5 →
  c.value = 5 ∨ c.value = -5 := by
  sorry

end NUMINAMATH_CALUDE_point_c_value_l1437_143738


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1437_143744

/-- A continuous function from positive reals to positive reals satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ 
  (∀ x, x > 0 → f x > 0) ∧
  ∀ x y, x > 0 → y > 0 → f (x + y) * (f x + f y) = f x * f y

/-- The theorem stating that any function satisfying the functional equation has the form f(x) = 1/(αx) for some α > 0 -/
theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h : FunctionalEquation f) :
  ∃ α : ℝ, α > 0 ∧ ∀ x, x > 0 → f x = 1 / (α * x) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1437_143744


namespace NUMINAMATH_CALUDE_gretchen_to_rachelle_ratio_l1437_143723

def pennies_problem (rachelle gretchen rocky : ℕ) : Prop :=
  rachelle = 180 ∧
  rocky = gretchen / 3 ∧
  rachelle + gretchen + rocky = 300

theorem gretchen_to_rachelle_ratio :
  ∀ rachelle gretchen rocky : ℕ,
  pennies_problem rachelle gretchen rocky →
  gretchen * 2 = rachelle :=
by
  sorry

end NUMINAMATH_CALUDE_gretchen_to_rachelle_ratio_l1437_143723


namespace NUMINAMATH_CALUDE_distribute_five_balls_to_three_children_l1437_143764

/-- The number of ways to distribute n identical balls to k children,
    with each child receiving at least one ball -/
def distribute_balls (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 6 ways to distribute 5 identical balls to 3 children,
    with each child receiving at least one ball -/
theorem distribute_five_balls_to_three_children :
  distribute_balls 5 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_to_three_children_l1437_143764


namespace NUMINAMATH_CALUDE_diophantine_equation_implication_l1437_143776

theorem diophantine_equation_implication (a b : ℤ) 
  (ha : ¬ ∃ (n : ℤ), a = n^2) 
  (hb : ¬ ∃ (n : ℤ), b = n^2) :
  (∃ (x0 y0 z0 w0 : ℤ), x0^2 - a*y0^2 - b*z0^2 + a*b*w0^2 = 0 ∧ (x0, y0, z0, w0) ≠ (0, 0, 0, 0)) →
  (∃ (x1 y1 z1 : ℤ), x1^2 - a*y1^2 - b*z1^2 = 0 ∧ (x1, y1, z1) ≠ (0, 0, 0)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_implication_l1437_143776


namespace NUMINAMATH_CALUDE_cos_240_degrees_l1437_143758

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l1437_143758


namespace NUMINAMATH_CALUDE_second_boy_speed_l1437_143774

/-- Given two boys walking in the same direction, with the first boy's speed at 5.5 kmph,
    and they are 20 km apart after 10 hours, prove that the second boy's speed is 7.5 kmph. -/
theorem second_boy_speed (v : ℝ) 
  (h1 : (v - 5.5) * 10 = 20) : v = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_second_boy_speed_l1437_143774


namespace NUMINAMATH_CALUDE_fraction_five_thirteenths_digit_sum_l1437_143777

theorem fraction_five_thirteenths_digit_sum : 
  ∃ (a b c d : ℕ), 
    (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧ 
    (5 : ℚ) / 13 = (a * 1000 + b * 100 + c * 10 + d) / 9999 ∧ 
    a + b + c + d = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_five_thirteenths_digit_sum_l1437_143777


namespace NUMINAMATH_CALUDE_honor_guard_subsets_l1437_143720

theorem honor_guard_subsets (n : ℕ) (h : n = 60) :
  Finset.card (Finset.powerset (Finset.range n)) = 2^n := by sorry

end NUMINAMATH_CALUDE_honor_guard_subsets_l1437_143720


namespace NUMINAMATH_CALUDE_boys_age_l1437_143751

/-- Given a person whose current age is twice what it was 5 years ago, prove that their current age is 10 years. -/
theorem boys_age (age : ℕ) (h : age = 2 * (age - 5)) : age = 10 := by
  sorry

end NUMINAMATH_CALUDE_boys_age_l1437_143751


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l1437_143734

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 100) : 
  a * b = -3 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l1437_143734


namespace NUMINAMATH_CALUDE_fraction_of_women_l1437_143798

/-- Proves that the fraction of women in a room is 1/4 given the specified conditions -/
theorem fraction_of_women (total_people : ℕ) (married_fraction : ℚ) (max_unmarried_women : ℕ) : 
  total_people = 80 →
  married_fraction = 3/4 →
  max_unmarried_women = 20 →
  (max_unmarried_women : ℚ) / total_people = 1/4 := by
  sorry

#check fraction_of_women

end NUMINAMATH_CALUDE_fraction_of_women_l1437_143798


namespace NUMINAMATH_CALUDE_fractional_equation_root_l1437_143743

theorem fractional_equation_root (m : ℝ) : 
  (∃ x : ℝ, x ≠ 4 ∧ (3 / (x - 4) + (x + m) / (4 - x) = 1)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l1437_143743


namespace NUMINAMATH_CALUDE_coplanar_condition_l1437_143778

open Real

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the origin and points
variable (O A B C D : V)

-- Define the condition for coplanarity
def are_coplanar (A B C D : V) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧ 
  a • (A - O) + b • (B - O) + c • (C - O) + d • (D - O) = 0

-- State the theorem
theorem coplanar_condition (k' : ℝ) :
  (4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k' • (D - O) = 0) →
  (are_coplanar O A B C D ↔ k' = -7) := by
  sorry

end NUMINAMATH_CALUDE_coplanar_condition_l1437_143778


namespace NUMINAMATH_CALUDE_cooking_time_per_potato_l1437_143746

theorem cooking_time_per_potato 
  (total_potatoes : ℕ) 
  (cooked_potatoes : ℕ) 
  (time_for_rest : ℕ) 
  (h1 : total_potatoes = 16) 
  (h2 : cooked_potatoes = 7) 
  (h3 : time_for_rest = 45) : 
  (time_for_rest : ℚ) / ((total_potatoes - cooked_potatoes) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cooking_time_per_potato_l1437_143746


namespace NUMINAMATH_CALUDE_set_operations_and_range_l1437_143716

def U := Set ℝ

def A : Set ℝ := {x | x ≥ 3}

def B : Set ℝ := {x | x^2 - 8*x + 7 ≤ 0}

def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem set_operations_and_range :
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x : ℝ | x ≥ 1}) ∧
  (∀ a : ℝ, C a ∪ A = A → a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l1437_143716


namespace NUMINAMATH_CALUDE_fraction_sum_integer_implies_not_divisible_by_three_l1437_143718

theorem fraction_sum_integer_implies_not_divisible_by_three (n : ℕ+) 
  (h : ∃ (k : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 8 + (1 : ℚ) / n.val = k) : 
  ¬(3 ∣ n.val) := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_implies_not_divisible_by_three_l1437_143718


namespace NUMINAMATH_CALUDE_f_monotone_increasing_and_g_bound_l1437_143737

open Real

noncomputable def f (x : ℝ) : ℝ := log x - (2 * x) / (x + 2)

noncomputable def g (x : ℝ) : ℝ := f x - 4 / (x + 2)

theorem f_monotone_increasing_and_g_bound (a : ℝ) :
  (∀ x > 0, Monotone f) ∧
  (∀ x > 0, g x < x + a ↔ a > -3) := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_and_g_bound_l1437_143737


namespace NUMINAMATH_CALUDE_sqrt_two_plus_pi_irrational_l1437_143714

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ (p : ℝ) / (q : ℝ)

-- State the theorem
theorem sqrt_two_plus_pi_irrational :
  IsIrrational (Real.sqrt 2) → IsIrrational π → IsIrrational (Real.sqrt 2 + π) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_pi_irrational_l1437_143714


namespace NUMINAMATH_CALUDE_ellipse_foci_coordinates_l1437_143771

/-- 
Given an ellipse with equation mx^2 + ny^2 + mn = 0, where m < n < 0,
prove that the coordinates of its foci are (0, ±√(n-m)).
-/
theorem ellipse_foci_coordinates 
  (m n : ℝ) 
  (h1 : m < n) 
  (h2 : n < 0) : 
  let equation := fun (x y : ℝ) => m * x^2 + n * y^2 + m * n
  ∃ c : ℝ, c > 0 ∧ 
    (∀ x y : ℝ, equation x y = 0 → 
      ((x = 0 ∧ y = c) ∨ (x = 0 ∧ y = -c)) ∧ 
      c^2 = n - m) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_coordinates_l1437_143771


namespace NUMINAMATH_CALUDE_subset_condition_implies_a_values_l1437_143717

theorem subset_condition_implies_a_values (a : ℝ) : 
  let A : Set ℝ := {x | x^2 = 1}
  let B : Set ℝ := {x | a * x = 1}
  B ⊆ A → a ∈ ({-1, 0, 1} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_subset_condition_implies_a_values_l1437_143717


namespace NUMINAMATH_CALUDE_problem_statement_l1437_143773

theorem problem_statement (a b : ℝ) (h : 2 * a - 3 * b = 5) :
  4 * a^2 - 9 * b^2 - 30 * b + 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1437_143773


namespace NUMINAMATH_CALUDE_building_has_seven_floors_l1437_143792

/-- Represents a building with floors -/
structure Building where
  totalFloors : ℕ
  ninasFloor : ℕ
  shurasFloor : ℕ

/-- Calculates the distance of Shura's mistaken path -/
def mistakenPathDistance (b : Building) : ℕ :=
  (b.totalFloors - b.ninasFloor) + (b.totalFloors - b.shurasFloor)

/-- Calculates the distance of Shura's direct path -/
def directPathDistance (b : Building) : ℕ :=
  if b.ninasFloor ≥ b.shurasFloor then b.ninasFloor - b.shurasFloor
  else b.shurasFloor - b.ninasFloor

/-- Theorem stating the conditions and conclusion about the building -/
theorem building_has_seven_floors :
  ∃ (b : Building),
    b.ninasFloor = 6 ∧
    b.totalFloors > b.ninasFloor ∧
    (mistakenPathDistance b : ℚ) = 1.5 * (directPathDistance b : ℚ) ∧
    b.totalFloors = 7 := by
  sorry

end NUMINAMATH_CALUDE_building_has_seven_floors_l1437_143792


namespace NUMINAMATH_CALUDE_plane_intersection_line_properties_l1437_143705

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (intersect_at : Plane → Plane → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (intersects : Line → Line → Point → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem plane_intersection_line_properties
  (α β : Plane) (l m : Line) (P : Point)
  (h1 : intersect_at α β l)
  (h2 : contains α m)
  (h3 : intersects m l P) :
  (∃ (n : Line), contains β n ∧ perpendicular m n) ∧
  (¬∃ (k : Line), contains β k ∧ parallel m k) :=
sorry

end NUMINAMATH_CALUDE_plane_intersection_line_properties_l1437_143705


namespace NUMINAMATH_CALUDE_friend_bikes_count_l1437_143725

/-- The number of bicycles Ignatius owns -/
def ignatius_bikes : ℕ := 4

/-- The number of tires on a bicycle -/
def tires_per_bike : ℕ := 2

/-- The number of tires on Ignatius's bikes -/
def ignatius_tires : ℕ := ignatius_bikes * tires_per_bike

/-- The total number of tires on the friend's cycles -/
def friend_total_tires : ℕ := 3 * ignatius_tires

/-- The number of tires on a unicycle -/
def unicycle_tires : ℕ := 1

/-- The number of tires on a tricycle -/
def tricycle_tires : ℕ := 3

/-- The number of tires on the friend's non-bicycle cycles -/
def friend_non_bike_tires : ℕ := unicycle_tires + tricycle_tires

/-- The number of tires on the friend's bicycles -/
def friend_bike_tires : ℕ := friend_total_tires - friend_non_bike_tires

theorem friend_bikes_count : (friend_bike_tires / tires_per_bike) = 10 := by
  sorry

end NUMINAMATH_CALUDE_friend_bikes_count_l1437_143725


namespace NUMINAMATH_CALUDE_last_tree_distance_l1437_143768

/-- The distance between the last pair of trees in a yard with a specific planting pattern -/
theorem last_tree_distance (yard_length : ℕ) (num_trees : ℕ) (first_distance : ℕ) (increment : ℕ) :
  yard_length = 1200 →
  num_trees = 117 →
  first_distance = 5 →
  increment = 2 →
  (num_trees - 1) * (2 * first_distance + (num_trees - 2) * increment) ≤ 2 * yard_length →
  first_distance + (num_trees - 2) * increment = 235 :=
by sorry

end NUMINAMATH_CALUDE_last_tree_distance_l1437_143768


namespace NUMINAMATH_CALUDE_tuesday_kids_l1437_143790

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 11

/-- The additional number of kids Julia played with on Tuesday compared to Monday -/
def additional_kids : ℕ := 1

/-- Theorem stating the number of kids Julia played with on Tuesday -/
theorem tuesday_kids : monday_kids + additional_kids = 12 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_kids_l1437_143790


namespace NUMINAMATH_CALUDE_system_solution_unique_l1437_143755

theorem system_solution_unique : 
  ∃! (x y : ℝ), (x + 3 * y = 1) ∧ (3 * x + y = -5) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1437_143755


namespace NUMINAMATH_CALUDE_f_properties_l1437_143709

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b / x^2

theorem f_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := f a b
  ∃ (min_value : ℝ) (min_point : ℝ),
    (∀ x, x ≠ 0 → f x ≥ min_value) ∧
    (f min_point = min_value) ∧
    (min_value = 2 * Real.sqrt (a * b)) ∧
    (min_point = Real.sqrt (Real.sqrt (b / a))) ∧
    (∀ x, f (-x) = f x) ∧
    (∀ x y, 0 < x ∧ x < y ∧ y < min_point → f x > f y) ∧
    (∀ x y, min_point < x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1437_143709


namespace NUMINAMATH_CALUDE_symmetric_complex_numbers_l1437_143793

theorem symmetric_complex_numbers (z₁ z₂ : ℂ) :
  (z₁ = 2 - 3*I) →
  (z₁ + z₂ = 0) →
  z₂ = -2 + 3*I := by
sorry

end NUMINAMATH_CALUDE_symmetric_complex_numbers_l1437_143793


namespace NUMINAMATH_CALUDE_train_speed_l1437_143795

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 1600) (h2 : time = 40) :
  length / time = 40 := by
sorry

end NUMINAMATH_CALUDE_train_speed_l1437_143795


namespace NUMINAMATH_CALUDE_range_of_a_l1437_143715

-- Define the propositions p and q
def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def q (x a : ℝ) : Prop := x^2 - (2*a + 1) * x + a * (a + 1) ≤ 0

-- Define the set A for proposition p
def A : Set ℝ := {x | p x}

-- Define the set B for proposition q
def B (a : ℝ) : Set ℝ := {x | q x a}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ A → x ∈ B a) ∧ (∃ x, x ∈ B a ∧ x ∉ A) → 0 ≤ a ∧ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1437_143715


namespace NUMINAMATH_CALUDE_seventh_term_ratio_l1437_143724

/-- Two arithmetic sequences and their properties -/
structure ArithmeticSequencePair where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  S : ℕ → ℚ  -- Sum of first n terms of sequence a
  T : ℕ → ℚ  -- Sum of first n terms of sequence b
  h : ∀ n, S n / T n = (3 * n - 2) / (2 * n + 1)  -- Given condition

/-- Theorem stating the relation between the 7th terms of the sequences -/
theorem seventh_term_ratio (seq : ArithmeticSequencePair) : seq.a 7 / seq.b 7 = 37 / 27 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_ratio_l1437_143724


namespace NUMINAMATH_CALUDE_ian_painted_faces_l1437_143762

/-- The number of faces of a cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The number of cuboids Ian painted -/
def number_of_cuboids : ℕ := 8

/-- The total number of faces painted by Ian -/
def total_faces_painted : ℕ := faces_per_cuboid * number_of_cuboids

theorem ian_painted_faces :
  total_faces_painted = 48 :=
by sorry

end NUMINAMATH_CALUDE_ian_painted_faces_l1437_143762


namespace NUMINAMATH_CALUDE_expression_value_l1437_143719

theorem expression_value : 
  let a : ℝ := Real.sqrt 3 - Real.sqrt 2
  let b : ℝ := Real.sqrt 3 + Real.sqrt 2
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) + 2*a*(b - a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1437_143719


namespace NUMINAMATH_CALUDE_diagonal_shorter_than_midpoint_distance_l1437_143726

-- Define the quadrilateral ABCD
variables {A B C D : EuclideanSpace ℝ (Fin 2)}

-- Define the property that a circle through three points is tangent to a line segment
def is_tangent_circle (P Q R S T : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ),
    dist center P = radius ∧ dist center Q = radius ∧ dist center R = radius ∧
    dist center S = radius ∧ dist S T = dist center S + dist center T

-- State the theorem
theorem diagonal_shorter_than_midpoint_distance
  (h1 : is_tangent_circle A B C C D)
  (h2 : is_tangent_circle A C D A B) :
  dist A C < (dist A D + dist B C) / 2 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_shorter_than_midpoint_distance_l1437_143726


namespace NUMINAMATH_CALUDE_parabola_equation_l1437_143702

/-- A parabola with directrix y = 1/2 and focus on the negative half of the y-axis has the standard equation x² = -2y -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) (h2 : p / 2 = 1 / 2) :
  ∀ x y : ℝ, (x^2 = -2 * p * y) ↔ (x^2 = -2 * y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1437_143702


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1437_143740

theorem linear_equation_solution :
  ∃ x : ℝ, 2 * x - 1 = 1 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1437_143740


namespace NUMINAMATH_CALUDE_pizza_solution_l1437_143757

/-- The number of pizzas made by Craig and Heather over two days -/
def pizza_problem (craig_day1 craig_day2 heather_day1 heather_day2 : ℕ) : Prop :=
  let total := craig_day1 + craig_day2 + heather_day1 + heather_day2
  craig_day1 = 40 ∧
  craig_day2 = craig_day1 + 60 ∧
  heather_day1 = 4 * craig_day1 ∧
  total = 380 ∧
  craig_day2 - heather_day2 = 20

theorem pizza_solution :
  ∃ (craig_day1 craig_day2 heather_day1 heather_day2 : ℕ),
    pizza_problem craig_day1 craig_day2 heather_day1 heather_day2 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_solution_l1437_143757


namespace NUMINAMATH_CALUDE_investment_more_profitable_l1437_143775

/-- Represents the initial price of buckwheat in rubles per kilogram -/
def initial_price : ℝ := 70

/-- Represents the final price of buckwheat in rubles per kilogram -/
def final_price : ℝ := 85

/-- Represents the annual interest rate for deposits in 2015 -/
def interest_rate_2015 : ℝ := 0.16

/-- Represents the annual interest rate for deposits in 2016 -/
def interest_rate_2016 : ℝ := 0.10

/-- Represents the annual interest rate for two-year deposits -/
def interest_rate_two_year : ℝ := 0.15

/-- Calculates the value after two years of annual deposits -/
def value_annual_deposits (initial : ℝ) : ℝ :=
  initial * (1 + interest_rate_2015) * (1 + interest_rate_2016)

/-- Calculates the value after a two-year deposit -/
def value_two_year_deposit (initial : ℝ) : ℝ :=
  initial * (1 + interest_rate_two_year) ^ 2

/-- Theorem stating that investing the initial price would yield more than the final price -/
theorem investment_more_profitable :
  (value_annual_deposits initial_price > final_price) ∧
  (value_two_year_deposit initial_price > final_price) := by
  sorry


end NUMINAMATH_CALUDE_investment_more_profitable_l1437_143775


namespace NUMINAMATH_CALUDE_smallest_k_for_divisibility_by_10_l1437_143770

def is_largest_prime_with_2005_digits (p : ℕ) : Prop :=
  Nat.Prime p ∧ 
  (10^2004 ≤ p) ∧ 
  (p < 10^2005) ∧ 
  ∀ q, Nat.Prime q → (10^2004 ≤ q) → (q < 10^2005) → q ≤ p

theorem smallest_k_for_divisibility_by_10 (p : ℕ) 
  (h : is_largest_prime_with_2005_digits p) : 
  (∃ k : ℕ, k > 0 ∧ (10 ∣ (p^2 - k))) ∧
  (∀ k : ℕ, k > 0 → (10 ∣ (p^2 - k)) → k ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_divisibility_by_10_l1437_143770


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1437_143788

theorem polynomial_divisibility (P : ℤ → ℤ) (n : ℤ) 
  (h1 : ∃ k1 : ℤ, P n = 3 * k1)
  (h2 : ∃ k2 : ℤ, P (n + 1) = 3 * k2)
  (h3 : ∃ k3 : ℤ, P (n + 2) = 3 * k3)
  (h_poly : ∀ x y : ℤ, ∃ a b c : ℤ, P (x + y) = P x + a * y + b * y^2 + c * y^3) :
  ∀ m : ℤ, ∃ k : ℤ, P m = 3 * k :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1437_143788


namespace NUMINAMATH_CALUDE_sum_of_sequences_l1437_143742

def sequence1 : List ℕ := [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
def sequence2 : List ℕ := [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

theorem sum_of_sequences : (sequence1.sum + sequence2.sum = 1000) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l1437_143742


namespace NUMINAMATH_CALUDE_abs_diff_of_sum_and_product_l1437_143704

theorem abs_diff_of_sum_and_product (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (prod_eq : x * y = 221) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_abs_diff_of_sum_and_product_l1437_143704


namespace NUMINAMATH_CALUDE_sheep_distribution_l1437_143750

theorem sheep_distribution (n : ℕ) (k : ℕ) 
  (h1 : 2000 ≤ n ∧ n ≤ 2100) 
  (h2 : k < n) 
  (h3 : 2 * k * (n - k) = n * (n - 1)) : n = 2025 := by
  sorry

end NUMINAMATH_CALUDE_sheep_distribution_l1437_143750


namespace NUMINAMATH_CALUDE_modified_fibonacci_series_sum_l1437_143763

/-- Modified Fibonacci sequence -/
def G : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | (n + 2) => G (n + 1) + G n

/-- The sum of the series G_n / 5^n from n = 0 to infinity -/
noncomputable def series_sum : ℚ := ∑' n, G n / (5 : ℚ) ^ n

/-- Theorem stating that the sum of the series G_n / 5^n from n = 0 to infinity equals 50/19 -/
theorem modified_fibonacci_series_sum : series_sum = 50 / 19 := by sorry

end NUMINAMATH_CALUDE_modified_fibonacci_series_sum_l1437_143763


namespace NUMINAMATH_CALUDE_shortest_chord_parallel_and_separate_l1437_143732

/-- Circle with center at origin and radius r -/
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

/-- Point P inside the circle -/
structure PointInCircle (r : ℝ) where
  a : ℝ
  b : ℝ
  h1 : a ≠ 0
  h2 : b ≠ 0
  h3 : a^2 + b^2 < r^2

/-- Line l1 containing the shortest chord through P -/
def ShortestChordLine (r : ℝ) (p : PointInCircle r) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | p.a * q.1 + p.b * q.2 = p.a^2 + p.b^2}

/-- Line l2 -/
def Line_l2 (r : ℝ) (p : PointInCircle r) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | p.b * q.1 - p.a * q.2 + r^2 = 0}

/-- Two lines are parallel -/
def Parallel (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l1 ↔ (x, y) ∈ l2 ∨ (x + k, y) ∈ l2

/-- A line is separate from a circle -/
def Separate (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∀ (p : ℝ × ℝ), p ∈ l → p ∉ c

theorem shortest_chord_parallel_and_separate (r : ℝ) (p : PointInCircle r) :
  Parallel (ShortestChordLine r p) (Line_l2 r p) ∧
  Separate (Line_l2 r p) (Circle r) := by
  sorry

end NUMINAMATH_CALUDE_shortest_chord_parallel_and_separate_l1437_143732


namespace NUMINAMATH_CALUDE_min_value_a_over_x_l1437_143733

theorem min_value_a_over_x (a x y : ℕ) 
  (ha : a > 100) 
  (hx : x > 100) 
  (hy : y > 100) 
  (h_eq : y^2 - 1 = a^2 * (x^2 - 1)) : 
  (∀ k : ℚ, k > 0 → (a : ℚ) / x ≥ k) ↔ k ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_over_x_l1437_143733


namespace NUMINAMATH_CALUDE_total_interest_earned_l1437_143729

/-- Calculates the total interest earned from two investments --/
theorem total_interest_earned
  (amount1 : ℝ)  -- Amount invested in the first account
  (amount2 : ℝ)  -- Amount invested in the second account
  (rate1 : ℝ)    -- Interest rate for the first account
  (rate2 : ℝ)    -- Interest rate for the second account
  (h1 : amount2 = amount1 + 800)  -- Second account has $800 more
  (h2 : amount1 + amount2 = 2000) -- Total investment is $2000
  (h3 : rate1 = 0.02)  -- 2% interest rate for first account
  (h4 : rate2 = 0.04)  -- 4% interest rate for second account
  : amount1 * rate1 + amount2 * rate2 = 68 := by
  sorry


end NUMINAMATH_CALUDE_total_interest_earned_l1437_143729


namespace NUMINAMATH_CALUDE_nested_fraction_equation_l1437_143752

theorem nested_fraction_equation (x : ℚ) : 
  3 + 1 / (2 + 1 / (3 + 3 / (4 + x))) = 225 / 68 → x = -102 / 19 := by
sorry

end NUMINAMATH_CALUDE_nested_fraction_equation_l1437_143752


namespace NUMINAMATH_CALUDE_shaded_area_equals_unshaded_triangle_area_l1437_143799

/-- The area of the shaded region in a rectangular grid with an unshaded right triangle -/
theorem shaded_area_equals_unshaded_triangle_area (width height : ℝ) :
  width = 14 ∧ height = 5 →
  let grid_area := width * height
  let triangle_area := (1 / 2) * width * height
  let shaded_area := grid_area - triangle_area
  shaded_area = triangle_area := by sorry

end NUMINAMATH_CALUDE_shaded_area_equals_unshaded_triangle_area_l1437_143799


namespace NUMINAMATH_CALUDE_triangle_cos_2C_l1437_143703

theorem triangle_cos_2C (a b : ℝ) (S_ABC : ℝ) (C : ℝ) :
  a = 8 →
  b = 5 →
  S_ABC = 12 →
  S_ABC = (1/2) * a * b * Real.sin C →
  Real.cos (2 * C) = 7/25 := by
sorry

end NUMINAMATH_CALUDE_triangle_cos_2C_l1437_143703


namespace NUMINAMATH_CALUDE_village_population_problem_l1437_143759

theorem village_population_problem (P : ℝ) : 
  (P * 1.3 * 0.7 = 13650) → P = 15000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l1437_143759


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l1437_143739

theorem factorial_fraction_simplification (N : ℕ) :
  (Nat.factorial (N - 1) * N * (N + 1)) / Nat.factorial (N + 2) = 1 / (N + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l1437_143739


namespace NUMINAMATH_CALUDE_bird_watching_average_l1437_143706

theorem bird_watching_average : 
  let marcus_birds : ℕ := 7
  let humphrey_birds : ℕ := 11
  let darrel_birds : ℕ := 9
  let isabella_birds : ℕ := 15
  let total_birds : ℕ := marcus_birds + humphrey_birds + darrel_birds + isabella_birds
  let num_watchers : ℕ := 4
  (total_birds : ℚ) / num_watchers = 10.5 := by sorry

end NUMINAMATH_CALUDE_bird_watching_average_l1437_143706


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l1437_143786

/-- The volume of a cylinder formed by rotating a rectangle about its lengthwise axis -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (length_positive : 0 < length) (width_positive : 0 < width) :
  let radius : ℝ := width / 2
  let height : ℝ := length
  let volume : ℝ := π * radius^2 * height
  (length = 16 ∧ width = 8) → volume = 256 * π := by
  sorry

#check cylinder_volume_from_rectangle

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l1437_143786


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1437_143730

def a : Fin 2 → ℝ := ![4, 3]
def b : Fin 2 → ℝ := ![-1, 2]

theorem perpendicular_vectors_k_value :
  ∃ k : ℝ, (∀ i : Fin 2, (a + k • b) i * (a - b) i = 0) → k = 23/3 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1437_143730


namespace NUMINAMATH_CALUDE_rook_placements_count_l1437_143722

/-- Represents a special chessboard with a long horizontal row at the bottom -/
structure SpecialChessboard where
  rows : Nat
  columns : Nat
  longRowLength : Nat

/-- Represents a rook placement on the special chessboard -/
structure RookPlacement where
  row : Nat
  column : Nat

/-- Checks if two rook placements attack each other on the special chessboard -/
def attacks (board : SpecialChessboard) (r1 r2 : RookPlacement) : Prop :=
  r1.row = r2.row ∨ r1.column = r2.column

/-- Counts the number of valid ways to place 3 rooks on the special chessboard -/
def countValidPlacements (board : SpecialChessboard) : Nat :=
  sorry

/-- The main theorem stating that there are 168 ways to place 3 rooks on the special chessboard -/
theorem rook_placements_count (board : SpecialChessboard) 
  (h1 : board.rows = 4) 
  (h2 : board.columns = 8) 
  (h3 : board.longRowLength = 8) : 
  countValidPlacements board = 168 := by
  sorry

end NUMINAMATH_CALUDE_rook_placements_count_l1437_143722


namespace NUMINAMATH_CALUDE_may_profit_max_profit_l1437_143745

-- Define the profit function
def profit (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 6 then 12 * x + 28
  else if 6 < x ∧ x ≤ 12 then 200 - 14 * x
  else 0

-- Theorem for May's profit
theorem may_profit : profit 5 = 88 := by sorry

-- Theorem for maximum profit
theorem max_profit :
  ∀ x : ℕ, 1 ≤ x ∧ x ≤ 12 → profit x ≤ profit 7 ∧ profit 7 = 102 := by sorry

end NUMINAMATH_CALUDE_may_profit_max_profit_l1437_143745


namespace NUMINAMATH_CALUDE_henry_skittles_l1437_143772

theorem henry_skittles (bridget_initial : ℕ) (bridget_final : ℕ) (henry : ℕ) : 
  bridget_initial = 4 → 
  bridget_final = 8 → 
  bridget_final = bridget_initial + henry → 
  henry = 4 := by
sorry

end NUMINAMATH_CALUDE_henry_skittles_l1437_143772


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1437_143707

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, -x^2 + c*x + 10 < 0 ↔ x < 2 ∨ x > 8) → c = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1437_143707


namespace NUMINAMATH_CALUDE_expression_evaluation_l1437_143754

theorem expression_evaluation : (5 ^ 2 : ℤ) + 15 / 3 - (3 * 2) ^ 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1437_143754


namespace NUMINAMATH_CALUDE_students_absent_correct_absent_count_l1437_143756

theorem students_absent (students_yesterday : ℕ) (students_registered : ℕ) : ℕ :=
  let students_today := (2 * students_yesterday) - (2 * students_yesterday / 10)
  students_registered - students_today

theorem correct_absent_count : students_absent 70 156 = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_absent_correct_absent_count_l1437_143756


namespace NUMINAMATH_CALUDE_roots_of_equation_l1437_143765

theorem roots_of_equation : ∃ x₁ x₂ : ℝ,
  (88 * (x₁ - 2)^2 = 95) ∧
  (88 * (x₂ - 2)^2 = 95) ∧
  (x₁ < 1) ∧
  (x₂ > 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1437_143765


namespace NUMINAMATH_CALUDE_inequality_solution_l1437_143791

theorem inequality_solution (x : ℝ) : 
  (1 / (x^2 + 1) > 4/x + 21/10) ↔ (-2 < x ∧ x < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1437_143791


namespace NUMINAMATH_CALUDE_painter_workdays_l1437_143780

theorem painter_workdays (job_size : ℝ) (rate : ℝ) (h : job_size = 6 * 1.5 * rate) :
  job_size = 4 * 2.25 * rate := by
sorry

end NUMINAMATH_CALUDE_painter_workdays_l1437_143780


namespace NUMINAMATH_CALUDE_fraction_equality_l1437_143760

theorem fraction_equality (a b c : ℝ) (h1 : a = b) (h2 : c ≠ 0) : a / c = b / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1437_143760


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_110_l1437_143721

/-- Given an arithmetic progression with first term a and common difference d -/
def arithmetic_progression (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1) * d

/-- Sum of first n terms of an arithmetic progression -/
def sum_arithmetic_progression (a d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a + (n - 1) * d)

theorem arithmetic_progression_sum_110 
  (a d : ℚ) 
  (h1 : sum_arithmetic_progression a d 10 = 100)
  (h2 : sum_arithmetic_progression a d 100 = 10) :
  sum_arithmetic_progression a d 110 = -110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_110_l1437_143721


namespace NUMINAMATH_CALUDE_right_pyramid_base_side_l1437_143712

-- Define the pyramid structure
structure RightPyramid :=
  (base_side : ℝ)
  (slant_height : ℝ)
  (lateral_face_area : ℝ)

-- Theorem statement
theorem right_pyramid_base_side 
  (p : RightPyramid) 
  (h1 : p.lateral_face_area = 120) 
  (h2 : p.slant_height = 40) : 
  p.base_side = 6 := by
  sorry


end NUMINAMATH_CALUDE_right_pyramid_base_side_l1437_143712


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1437_143782

-- Define the sets M and N
def M : Set ℝ := {x | x - 2 > 0}
def N : Set ℝ := {x | (x - 3) * (x - 1) < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1437_143782


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_135_l1437_143708

/-- The coefficient of x^2 in the expansion of (3x-1)^6 -/
def coefficient_x_squared : ℕ :=
  let n : ℕ := 6
  let k : ℕ := 4
  let binomial_coefficient : ℕ := n.choose k
  let power_of_three : ℕ := 3^(n - k)
  binomial_coefficient * power_of_three

/-- Theorem stating that the coefficient of x^2 in (3x-1)^6 is 135 -/
theorem coefficient_x_squared_is_135 : coefficient_x_squared = 135 := by
  sorry

#eval coefficient_x_squared

end NUMINAMATH_CALUDE_coefficient_x_squared_is_135_l1437_143708


namespace NUMINAMATH_CALUDE_stuffed_animal_sales_difference_l1437_143713

/-- Given the sales of stuffed animals by Quincy, Thor, and Jake, prove the difference between Quincy's and Jake's sales. -/
theorem stuffed_animal_sales_difference 
  (quincy thor jake : ℕ) 
  (h1 : quincy = 100 * thor) 
  (h2 : jake = thor + 15) 
  (h3 : quincy = 2000) : 
  quincy - jake = 1965 := by
sorry

end NUMINAMATH_CALUDE_stuffed_animal_sales_difference_l1437_143713


namespace NUMINAMATH_CALUDE_largest_prime_1005_digits_squared_minus_one_div_24_l1437_143741

-- Define q as the largest prime with 1005 digits
def q : ℕ := sorry

-- Axiom: q is prime
axiom q_prime : Nat.Prime q

-- Axiom: q has 1005 digits
axiom q_digits : 10^1004 ≤ q ∧ q < 10^1005

-- Theorem to prove
theorem largest_prime_1005_digits_squared_minus_one_div_24 :
  24 ∣ (q^2 - 1) := by sorry

end NUMINAMATH_CALUDE_largest_prime_1005_digits_squared_minus_one_div_24_l1437_143741


namespace NUMINAMATH_CALUDE_regular_soda_bottles_count_l1437_143747

/-- The number of regular soda bottles in a grocery store -/
def regular_soda_bottles : ℕ := 30

/-- The total number of bottles in the store -/
def total_bottles : ℕ := 38

/-- The number of diet soda bottles in the store -/
def diet_soda_bottles : ℕ := 8

/-- Theorem stating that the number of regular soda bottles is correct -/
theorem regular_soda_bottles_count : 
  regular_soda_bottles = total_bottles - diet_soda_bottles :=
by sorry

end NUMINAMATH_CALUDE_regular_soda_bottles_count_l1437_143747


namespace NUMINAMATH_CALUDE_power_of_two_start_with_any_digits_l1437_143769

theorem power_of_two_start_with_any_digits :
  ∀ A : ℕ, ∃ n m : ℕ+, (10 ^ m.val : ℝ) * A < (2 ^ n.val : ℝ) ∧ (2 ^ n.val : ℝ) < (10 ^ m.val : ℝ) * (A + 1) := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_start_with_any_digits_l1437_143769


namespace NUMINAMATH_CALUDE_eleventh_sum_14_l1437_143789

/-- Given a natural number, returns the sum of its digits -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if the given natural number has digits that sum to 14 -/
def sum_to_14 (n : ℕ) : Prop := digit_sum n = 14

/-- Returns the nth positive integer whose digits sum to 14 -/
def nth_sum_14 (n : ℕ) : ℕ := sorry

theorem eleventh_sum_14 : nth_sum_14 11 = 149 := by sorry

end NUMINAMATH_CALUDE_eleventh_sum_14_l1437_143789


namespace NUMINAMATH_CALUDE_emily_small_gardens_l1437_143749

/-- The number of small gardens Emily had -/
def num_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

/-- Proof that Emily had 3 small gardens -/
theorem emily_small_gardens :
  num_small_gardens 41 29 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_emily_small_gardens_l1437_143749


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1437_143781

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1437_143781


namespace NUMINAMATH_CALUDE_joan_flour_cups_l1437_143779

theorem joan_flour_cups (total : ℕ) (remaining : ℕ) (already_added : ℕ) : 
  total = 7 → remaining = 4 → already_added = total - remaining → already_added = 3 := by
  sorry

end NUMINAMATH_CALUDE_joan_flour_cups_l1437_143779


namespace NUMINAMATH_CALUDE_iron_cotton_mass_equality_l1437_143728

-- Define the conversion factor from kilograms to grams
def kgToGrams : ℝ → ℝ := (· * 1000)

-- Define the masses in their given units
def ironMassKg : ℝ := 5
def cottonMassG : ℝ := 5000

-- Theorem stating that the masses are equal
theorem iron_cotton_mass_equality :
  kgToGrams ironMassKg = cottonMassG := by sorry

end NUMINAMATH_CALUDE_iron_cotton_mass_equality_l1437_143728


namespace NUMINAMATH_CALUDE_girls_in_classroom_l1437_143797

theorem girls_in_classroom (boys : ℕ) (ratio : ℚ) (girls : ℕ) : 
  boys = 20 → ratio = 1/2 → (girls : ℚ) / boys = ratio → girls = 10 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_classroom_l1437_143797


namespace NUMINAMATH_CALUDE_increasing_condition_l1437_143761

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x + 5

theorem increasing_condition (m : ℝ) :
  (∀ x ∈ Set.Ici (-2 : ℝ), Monotone (fun x => f m x)) ↔ m ∈ Set.Icc 0 (1/4) :=
sorry

end NUMINAMATH_CALUDE_increasing_condition_l1437_143761


namespace NUMINAMATH_CALUDE_banana_count_l1437_143753

theorem banana_count (bananas apples oranges : ℕ) : 
  apples = 2 * bananas →
  oranges = 6 →
  bananas + apples + oranges = 12 →
  bananas = 2 := by
sorry

end NUMINAMATH_CALUDE_banana_count_l1437_143753


namespace NUMINAMATH_CALUDE_reconstructed_text_is_correct_l1437_143785

-- Define the set of original characters
def OriginalChars : Set Char := {'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я'}

-- Define a mapping from distorted characters to original characters
def DistortedToOriginal : Char → Char := sorry

-- Define the reconstructed text
def ReconstructedText : String := "глобальное потепление"

-- Theorem stating that the reconstructed text is correct
theorem reconstructed_text_is_correct :
  ∀ c ∈ ReconstructedText.data, DistortedToOriginal c ∈ OriginalChars :=
sorry

#check reconstructed_text_is_correct

end NUMINAMATH_CALUDE_reconstructed_text_is_correct_l1437_143785


namespace NUMINAMATH_CALUDE_car_speed_problem_l1437_143711

/-- Proves that the speed at which a car takes 15 seconds less to travel 1 kilometer
    compared to traveling at 48 km/h is 60 km/h. -/
theorem car_speed_problem (v : ℝ) : v > 0 →
  (1 / (48 / 3600) = 1 / (v / 3600) + 15) → v = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1437_143711


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l1437_143766

/-- Given a line with equation y = 3x + 1, this theorem states that its symmetric line
    with respect to the y-axis has the equation y = -3x + 1 -/
theorem symmetric_line_wrt_y_axis :
  ∀ (x y : ℝ), (∃ (m n : ℝ), n = 3 * m + 1 ∧ x + m = 0 ∧ y = n) →
  y = -3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_y_axis_l1437_143766


namespace NUMINAMATH_CALUDE_function_range_l1437_143787

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem function_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 a, f x ≥ 2) ∧
  (∃ x ∈ Set.Icc 0 a, f x = 2) →
  a ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_function_range_l1437_143787


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l1437_143783

theorem solve_system_of_equations (s t : ℚ) 
  (eq1 : 15 * s + 7 * t = 210)
  (eq2 : t = 3 * s) : 
  s = 35 / 6 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l1437_143783


namespace NUMINAMATH_CALUDE_probability_no_adjacent_red_in_ring_l1437_143736

/-- The number of red marbles -/
def num_red : ℕ := 4

/-- The number of blue marbles -/
def num_blue : ℕ := 8

/-- The total number of marbles -/
def total_marbles : ℕ := num_red + num_blue

/-- The probability of no two red marbles being adjacent when arranged in a ring -/
def probability_no_adjacent_red : ℚ := 7 / 33

/-- Theorem: The probability of no two red marbles being adjacent when 4 red marbles
    and 8 blue marbles are randomly arranged in a ring is 7/33 -/
theorem probability_no_adjacent_red_in_ring :
  probability_no_adjacent_red = 7 / 33 :=
by sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_red_in_ring_l1437_143736


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1437_143735

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n - 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (a 1 + a n) / 2

-- Define the sum of the first n terms of b_n
def T (n : ℕ) : ℚ := n / (2 * n + 1)

theorem arithmetic_sequence_problem :
  S 9 = 81 ∧ a 3 + a 5 = 14 →
  (∀ n : ℕ, a n = 2 * n - 1) ∧
  (∀ n : ℕ, T n = n / (2 * n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1437_143735


namespace NUMINAMATH_CALUDE_toothfairy_money_is_11_90_l1437_143748

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The number of each type of coin Joan received -/
def coin_count : ℕ := 14

/-- The total value of coins Joan received from the toothfairy -/
def toothfairy_money : ℚ := coin_count * quarter_value + coin_count * half_dollar_value + coin_count * dime_value

theorem toothfairy_money_is_11_90 : toothfairy_money = 11.90 := by
  sorry

end NUMINAMATH_CALUDE_toothfairy_money_is_11_90_l1437_143748


namespace NUMINAMATH_CALUDE_fold_and_cut_square_l1437_143710

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a straight line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Represents a folding operation -/
def Fold := Point → Point

/-- Checks if a line intersects all four 1x1 squares in a 2x2 square -/
def intersectsAllSquares (l : Line) : Prop :=
  ∃ (p1 p2 p3 p4 : Point),
    p1.x + p1.y = 1 ∧ 
    p2.x + p2.y = 3 ∧ 
    p3.x - p3.y = 1 ∧ 
    p4.x - p4.y = -1 ∧
    l.a * p1.x + l.b * p1.y + l.c = 0 ∧
    l.a * p2.x + l.b * p2.y + l.c = 0 ∧
    l.a * p3.x + l.b * p3.y + l.c = 0 ∧
    l.a * p4.x + l.b * p4.y + l.c = 0

/-- The main theorem stating that it's possible to fold and cut a 2x2 square into four 1x1 squares -/
theorem fold_and_cut_square : 
  ∃ (f1 f2 : Fold) (l : Line),
    intersectsAllSquares l :=
sorry

end NUMINAMATH_CALUDE_fold_and_cut_square_l1437_143710


namespace NUMINAMATH_CALUDE_jim_car_efficiency_l1437_143767

/-- Calculates the fuel efficiency of a car given its tank capacity, remaining fuel ratio, and trip distance. -/
def fuel_efficiency (tank_capacity : ℚ) (remaining_ratio : ℚ) (trip_distance : ℚ) : ℚ :=
  trip_distance / (tank_capacity * (1 - remaining_ratio))

/-- Theorem stating that under the given conditions, the fuel efficiency is 5 miles per gallon. -/
theorem jim_car_efficiency :
  let tank_capacity : ℚ := 12
  let remaining_ratio : ℚ := 2/3
  let trip_distance : ℚ := 20
  fuel_efficiency tank_capacity remaining_ratio trip_distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_jim_car_efficiency_l1437_143767


namespace NUMINAMATH_CALUDE_correct_average_l1437_143727

theorem correct_average (n : ℕ) (initial_avg : ℚ) (error1 : ℚ) (error2 : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  error1 = 16 →
  error2 = 18 →
  (n : ℚ) * initial_avg - error1 + error2 = n * 40.4 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l1437_143727


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_not_five_l1437_143731

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of digits of a perfect square cannot be 5 -/
theorem sum_of_digits_of_square_not_five (n : ℕ) : 
  ∃ m : ℕ, n = m^2 → sumOfDigits n ≠ 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_not_five_l1437_143731


namespace NUMINAMATH_CALUDE_min_value_a_l1437_143794

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 → x^2 + 2*x*y ≤ a*(x^2 + y^2)) →
  a ≥ (Real.sqrt 5 + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l1437_143794


namespace NUMINAMATH_CALUDE_right_triangle_trig_l1437_143796

theorem right_triangle_trig (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : C = Real.pi / 2) (h3 : Real.sin A = 2 / 3) : Real.cos B = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_l1437_143796


namespace NUMINAMATH_CALUDE_marks_lost_per_wrong_answer_l1437_143700

/-- Prove that the number of marks lost for each wrong answer is 1 -/
theorem marks_lost_per_wrong_answer
  (total_questions : ℕ)
  (marks_per_correct : ℕ)
  (total_marks : ℕ)
  (correct_answers : ℕ)
  (h1 : total_questions = 60)
  (h2 : marks_per_correct = 4)
  (h3 : total_marks = 130)
  (h4 : correct_answers = 38) :
  ∃ (marks_lost : ℕ), 
    marks_lost = 1 ∧ 
    total_marks = correct_answers * marks_per_correct - (total_questions - correct_answers) * marks_lost :=
by
  sorry

end NUMINAMATH_CALUDE_marks_lost_per_wrong_answer_l1437_143700


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l1437_143701

/-- Represents the cost function for a caterer -/
structure CatererCost where
  basicFee : ℕ
  perPersonCost : ℕ

/-- Calculates the total cost for a caterer given the number of people -/
def totalCost (c : CatererCost) (people : ℕ) : ℕ :=
  c.basicFee + c.perPersonCost * people

/-- The first caterer's cost structure -/
def caterer1 : CatererCost := { basicFee := 150, perPersonCost := 18 }

/-- The second caterer's cost structure -/
def caterer2 : CatererCost := { basicFee := 250, perPersonCost := 15 }

/-- Theorem stating that 34 is the least number of people for which the second caterer is less expensive -/
theorem second_caterer_cheaper_at_34 :
  (∀ n : ℕ, n < 34 → totalCost caterer1 n ≤ totalCost caterer2 n) ∧
  (totalCost caterer2 34 < totalCost caterer1 34) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l1437_143701


namespace NUMINAMATH_CALUDE_four_isosceles_triangles_l1437_143784

-- Define a point in 2D space
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a triangle by its three vertices
structure Triangle :=
  (v1 : Point)
  (v2 : Point)
  (v3 : Point)

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : Point) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := squaredDistance t.v1 t.v2
  let d2 := squaredDistance t.v2 t.v3
  let d3 := squaredDistance t.v3 t.v1
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the 5 triangles
def triangle1 : Triangle := ⟨⟨1, 5⟩, ⟨3, 5⟩, ⟨2, 3⟩⟩
def triangle2 : Triangle := ⟨⟨4, 3⟩, ⟨4, 5⟩, ⟨6, 3⟩⟩
def triangle3 : Triangle := ⟨⟨1, 2⟩, ⟨4, 3⟩, ⟨7, 2⟩⟩
def triangle4 : Triangle := ⟨⟨5, 1⟩, ⟨4, 3⟩, ⟨6, 1⟩⟩
def triangle5 : Triangle := ⟨⟨3, 1⟩, ⟨4, 3⟩, ⟨5, 1⟩⟩

-- Theorem to prove
theorem four_isosceles_triangles :
  (isIsosceles triangle1) ∧
  (isIsosceles triangle2) ∧
  (isIsosceles triangle3) ∧
  (¬ isIsosceles triangle4) ∧
  (isIsosceles triangle5) :=
sorry

end NUMINAMATH_CALUDE_four_isosceles_triangles_l1437_143784
