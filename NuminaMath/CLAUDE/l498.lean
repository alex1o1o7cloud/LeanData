import Mathlib

namespace cubic_factorization_l498_49896

theorem cubic_factorization (x : ℝ) : x^3 - 8*x^2 + 16*x = x*(x-4)^2 := by
  sorry

end cubic_factorization_l498_49896


namespace nested_function_evaluation_l498_49868

noncomputable def N (x : ℝ) : ℝ := 2 * Real.sqrt x

def O (x : ℝ) : ℝ := x^3

theorem nested_function_evaluation :
  N (O (N (O (N (O 2))))) = 724 * Real.sqrt 2 := by sorry

end nested_function_evaluation_l498_49868


namespace pascal_triangle_interior_sum_l498_49831

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def sumInteriorNumbers (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_triangle_interior_sum :
  sumInteriorNumbers 4 = 6 ∧
  sumInteriorNumbers 5 = 14 →
  sumInteriorNumbers 7 = 62 := by
  sorry

end pascal_triangle_interior_sum_l498_49831


namespace deck_size_l498_49827

theorem deck_size (r b : ℕ) : 
  (r : ℚ) / (r + b) = 2 / 5 →
  (r : ℚ) / (r + b + 6) = 1 / 3 →
  r + b = 30 := by
  sorry

end deck_size_l498_49827


namespace inscribed_circle_arithmetic_progression_l498_49829

theorem inscribed_circle_arithmetic_progression (a b c r : ℝ) :
  (0 < r) →
  (0 < a) →
  (0 < b) →
  (0 < c) →
  (a + b > c) →
  (b + c > a) →
  (c + a > b) →
  (∃ d : ℝ, d > 0 ∧ a = 2*r + d ∧ b = 2*r + 2*d ∧ c = 2*r + 3*d) →
  (∃ k : ℝ, k > 0 ∧ a = 3*k ∧ b = 4*k ∧ c = 5*k) :=
by sorry

end inscribed_circle_arithmetic_progression_l498_49829


namespace interest_rate_calculation_l498_49834

theorem interest_rate_calculation (total_investment : ℝ) (first_part : ℝ) (second_part_rate : ℝ) (total_interest : ℝ) : 
  total_investment = 3600 →
  first_part = 1800 →
  second_part_rate = 5 →
  total_interest = 144 →
  (first_part * (3 / 100)) + ((total_investment - first_part) * (second_part_rate / 100)) = total_interest :=
by sorry

end interest_rate_calculation_l498_49834


namespace ratio_of_trigonometric_equation_l498_49872

theorem ratio_of_trigonometric_equation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (a * Real.sin (π / 5) + b * Real.cos (π / 5)) / (a * Real.cos (π / 5) - b * Real.sin (π / 5)) = Real.tan (8 * π / 15)) : 
  b / a = Real.sqrt 3 := by
sorry

end ratio_of_trigonometric_equation_l498_49872


namespace partition_inequality_l498_49885

def f (n : ℕ) : ℕ := sorry

theorem partition_inequality (n : ℕ) (h : n ≥ 1) :
  f (n + 1) ≤ (f n + f (n + 2)) / 2 := by
  sorry

end partition_inequality_l498_49885


namespace common_point_on_intersection_circle_l498_49897

/-- A parabola that intersects the coordinate axes at three distinct points -/
structure TripleIntersectingParabola where
  p : ℝ
  q : ℝ
  x₁ : ℝ
  x₂ : ℝ
  h₁ : x₁ ≠ 0
  h₂ : x₂ ≠ 0
  h₃ : q ≠ 0
  h₄ : x₁ ≠ x₂
  h₅ : x₁^2 + p*x₁ + q = 0  -- x₁ is a root
  h₆ : x₂^2 + p*x₂ + q = 0  -- x₂ is a root

/-- The circle passing through the three intersection points of the parabola with the coordinate axes -/
def intersectionCircle (para : TripleIntersectingParabola) : Set (ℝ × ℝ) :=
  {pt : ℝ × ℝ | ∃ (r : ℝ), (pt.1 - 0)^2 + (pt.2 - 0)^2 = r^2 ∧
                           (pt.1 - para.x₁)^2 + pt.2^2 = r^2 ∧
                           (pt.1 - para.x₂)^2 + pt.2^2 = r^2 ∧
                           (pt.1 - 0)^2 + (pt.2 - para.q)^2 = r^2}

/-- The theorem stating that R(0, 1) lies on the intersection circle for all valid parabolas -/
theorem common_point_on_intersection_circle (para : TripleIntersectingParabola) :
  (0, 1) ∈ intersectionCircle para := by
  sorry

end common_point_on_intersection_circle_l498_49897


namespace maddy_chocolate_eggs_l498_49840

/-- The number of chocolate eggs Maddy eats per day -/
def eggs_per_day : ℕ := 2

/-- The number of weeks the chocolate eggs last -/
def weeks_lasting : ℕ := 4

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Maddy was given 56 chocolate eggs -/
theorem maddy_chocolate_eggs :
  eggs_per_day * weeks_lasting * days_in_week = 56 := by
  sorry

end maddy_chocolate_eggs_l498_49840


namespace range_of_f_on_interval_range_of_a_l498_49847

-- Define the function f(x) = x^2 - ax + 4
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 4

-- Part 1: Range of f(x) on [1, 3] when a = 3
theorem range_of_f_on_interval (x : ℝ) (h : x ∈ Set.Icc 1 3) :
  ∃ y ∈ Set.Icc (7/4) 4, y = f 3 x :=
sorry

-- Part 2: Range of values for a
theorem range_of_a (a : ℝ) (h : ∀ x ∈ Set.Icc 0 2, f a x ≤ 4) :
  a ≥ 2 :=
sorry

end range_of_f_on_interval_range_of_a_l498_49847


namespace martha_blocks_l498_49869

/-- Given Martha's initial and found blocks, prove the total number of blocks she ends with. -/
theorem martha_blocks (initial_blocks found_blocks : ℕ) 
  (h1 : initial_blocks = 4)
  (h2 : found_blocks = 80) :
  initial_blocks + found_blocks = 84 := by
  sorry

#check martha_blocks

end martha_blocks_l498_49869


namespace jack_house_height_correct_l498_49878

/-- The height of Jack's house -/
def jackHouseHeight : ℝ := 49

/-- The length of the shadow cast by Jack's house -/
def jackHouseShadow : ℝ := 56

/-- The height of the tree -/
def treeHeight : ℝ := 21

/-- The length of the shadow cast by the tree -/
def treeShadow : ℝ := 24

/-- Theorem stating that the calculated height of Jack's house is correct -/
theorem jack_house_height_correct :
  jackHouseHeight = (jackHouseShadow * treeHeight) / treeShadow :=
by sorry

end jack_house_height_correct_l498_49878


namespace problem_statement_l498_49852

def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {x | x < a}
def M (m : ℝ) : Set ℝ := {x | x^2 - (1+m)*x + m = 0}

theorem problem_statement (a m : ℝ) (h : m > 1) :
  (A ∩ B a = A → a > 2) ∧
  (m ≠ 2 → A ∪ M m = {1, 2, m}) ∧
  (m = 2 → A ∪ M m = {1, 2}) := by
  sorry

end problem_statement_l498_49852


namespace quadratic_real_roots_l498_49806

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + (k + 1) * x + (k^2 - 3) = 0) ↔ 
  ((1 - 2 * Real.sqrt 10) / 3 ≤ k ∧ k ≤ (1 + 2 * Real.sqrt 10) / 3) := by
sorry

end quadratic_real_roots_l498_49806


namespace train_passenger_count_l498_49853

theorem train_passenger_count (round_trips : ℕ) (return_passengers : ℕ) (total_passengers : ℕ) :
  round_trips = 4 →
  return_passengers = 60 →
  total_passengers = 640 →
  ∃ (one_way_passengers : ℕ),
    one_way_passengers = 100 ∧
    total_passengers = round_trips * (one_way_passengers + return_passengers) :=
by sorry

end train_passenger_count_l498_49853


namespace inequality_preservation_l498_49880

theorem inequality_preservation (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end inequality_preservation_l498_49880


namespace picture_fit_count_l498_49860

-- Define the number of pictures for each category for Ralph and Derrick
def ralph_wild_animals : ℕ := 75
def ralph_landscapes : ℕ := 36
def ralph_family_events : ℕ := 45
def ralph_cars : ℕ := 20

def derrick_wild_animals : ℕ := 95
def derrick_landscapes : ℕ := 42
def derrick_family_events : ℕ := 55
def derrick_cars : ℕ := 25
def derrick_airplanes : ℕ := 10

-- Calculate total pictures for Ralph and Derrick
def ralph_total : ℕ := ralph_wild_animals + ralph_landscapes + ralph_family_events + ralph_cars
def derrick_total : ℕ := derrick_wild_animals + derrick_landscapes + derrick_family_events + derrick_cars + derrick_airplanes

-- Calculate the combined total of pictures
def combined_total : ℕ := ralph_total + derrick_total

-- Calculate the difference in wild animal pictures
def wild_animals_difference : ℕ := derrick_wild_animals - ralph_wild_animals

-- Theorem to prove
theorem picture_fit_count : (combined_total / wild_animals_difference : ℕ) = 20 := by
  sorry

end picture_fit_count_l498_49860


namespace finite_perfect_squares_l498_49883

/-- For positive integers a and b, the set of integers n for which both an^2 + b and a(n+1)^2 + b are perfect squares is finite -/
theorem finite_perfect_squares (a b : ℕ+) :
  {n : ℤ | ∃ x y : ℤ, (a : ℤ) * n^2 + (b : ℤ) = x^2 ∧ (a : ℤ) * (n + 1)^2 + (b : ℤ) = y^2}.Finite :=
by sorry

end finite_perfect_squares_l498_49883


namespace sin_15_cos_15_l498_49826

theorem sin_15_cos_15 : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end sin_15_cos_15_l498_49826


namespace parallel_transitive_infinite_perpendicular_to_skew_l498_49804

/-- Represents a line in 3D space -/
structure Line3D where
  -- We don't need to define the internals of the line structure
  -- as we're only interested in the relationships between lines

/-- Parallel relation between two lines -/
def parallel (l1 l2 : Line3D) : Prop := sorry

/-- Perpendicular relation between two lines -/
def perpendicular (l1 l2 : Line3D) : Prop := sorry

/-- Skew relation between two lines -/
def skew (l1 l2 : Line3D) : Prop := sorry

/-- The set of all lines perpendicular to two given lines -/
def perpendicularLines (l1 l2 : Line3D) : Set Line3D := sorry

theorem parallel_transitive (a b c : Line3D) :
  parallel a b → parallel b c → parallel a c := by sorry

theorem infinite_perpendicular_to_skew (a b : Line3D) :
  skew a b → Set.Infinite (perpendicularLines a b) := by sorry

end parallel_transitive_infinite_perpendicular_to_skew_l498_49804


namespace f_theorem_l498_49861

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂) ∧
  (∀ x₁ x₂, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₁ < f x₂) ∧
  (f 1 = 2)

theorem f_theorem (f : ℝ → ℝ) (h : f_properties f) :
  (∀ x₁ x₂, 0 ≤ x₁ → x₁ < x₂ → -(f x₁)^2 > -(f x₂)^2) ∧
  (∀ x₁ x₂, x₁ < x₂ → x₂ ≤ 0 → -(f x₁)^2 < -(f x₂)^2) ∧
  (∀ a, f (2 * a^2 - 1) + 2 * f a - 6 < 0 ↔ -2 < a ∧ a < 1) :=
by sorry

end f_theorem_l498_49861


namespace square_construction_possible_l498_49854

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represent a compass operation -/
inductive CompassOp
  | drawCircle (center : Point) (radius : ℝ)
  | findIntersection (c1 : Circle) (c2 : Circle)

/-- Represent a sequence of compass operations -/
def CompassConstruction := List CompassOp

/-- The center of the square -/
def O : Point := sorry

/-- One vertex of the square -/
def A : Point := sorry

/-- The radius of the circumcircle -/
def r : ℝ := sorry

/-- Check if a point is a vertex of the square -/
def isSquareVertex (p : Point) : Prop := sorry

/-- Check if a construction is valid (uses only compass operations) -/
def isValidConstruction (c : CompassConstruction) : Prop := sorry

/-- The main theorem: it's possible to construct the other vertices using only a compass -/
theorem square_construction_possible :
  ∃ (B C D : Point) (construction : CompassConstruction),
    isValidConstruction construction ∧
    isSquareVertex B ∧
    isSquareVertex C ∧
    isSquareVertex D :=
  sorry

end square_construction_possible_l498_49854


namespace chloes_candies_l498_49865

/-- Given that Linda has 34 candies and the total number of candies is 62,
    prove that Chloe has 28 candies. -/
theorem chloes_candies (linda_candies : ℕ) (total_candies : ℕ) (chloe_candies : ℕ) : 
  linda_candies = 34 → total_candies = 62 → chloe_candies = total_candies - linda_candies →
  chloe_candies = 28 := by
  sorry

end chloes_candies_l498_49865


namespace line_circle_intersection_and_dot_product_range_l498_49851

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  m * x - y + 1 - m = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + (y-1)^2 = 5

-- Define points D and E
def point_D : ℝ × ℝ := (-2, 0)
def point_E : ℝ × ℝ := (2, 0)

-- Define the condition for P being inside C
def inside_circle (x y : ℝ) : Prop :=
  x^2 + (y-1)^2 < 5

-- Define the geometric sequence condition
def geometric_sequence (x y : ℝ) : Prop :=
  ((x+2)^2 + y^2) * ((x-2)^2 + y^2) = (x^2 + y^2)^2

-- Theorem statement
theorem line_circle_intersection_and_dot_product_range :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), line_l m x y ∧ circle_C x y) ∧
  (∀ (x y : ℝ), 
    inside_circle x y → 
    geometric_sequence x y → 
    -2 ≤ ((x+2)*(-x+2) + y*(-y)) ∧ 
    ((x+2)*(-x+2) + y*(-y)) < 1 + Real.sqrt 5) :=
by sorry

end line_circle_intersection_and_dot_product_range_l498_49851


namespace function_bound_l498_49802

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, |x₁ - x₂| ≤ 1 → |f x₂ - f x₁| ≤ 1) ∧ f 0 = 1

/-- The main theorem -/
theorem function_bound (f : ℝ → ℝ) (h : SatisfiesConditions f) :
  ∀ x : ℝ, -|x| ≤ f x ∧ f x ≤ |x| + 2 := by
  sorry

end function_bound_l498_49802


namespace triangle_area_inequality_l498_49811

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle : a < b + c ∧ b < a + c ∧ c < a + b

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_area_inequality (t : Triangle) :
  area t / (t.a * t.b + t.b * t.c + t.c * t.a) ≤ 1 / (4 * Real.sqrt 3) := by sorry

end triangle_area_inequality_l498_49811


namespace average_of_reeyas_scores_l498_49873

def reeyas_scores : List ℕ := [55, 67, 76, 82, 85]

theorem average_of_reeyas_scores :
  (List.sum reeyas_scores) / (List.length reeyas_scores) = 73 := by
  sorry

end average_of_reeyas_scores_l498_49873


namespace amy_work_schedule_l498_49817

/-- Amy's work schedule and earnings problem -/
theorem amy_work_schedule (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℕ)
  (school_weeks : ℕ) (school_earnings : ℕ) :
  summer_weeks = 12 →
  summer_hours_per_week = 40 →
  summer_earnings = 4800 →
  school_weeks = 36 →
  school_earnings = 7200 →
  (school_earnings / (summer_earnings / (summer_weeks * summer_hours_per_week))) / school_weeks = 20 := by
  sorry

#check amy_work_schedule

end amy_work_schedule_l498_49817


namespace clay_transformation_in_two_operations_l498_49864

/-- Represents a collection of clay pieces -/
structure ClayCollection where
  pieces : List Nat
  deriving Repr

/-- Represents an operation on clay pieces -/
def combine_operation (c : ClayCollection) (group_size : Nat) : ClayCollection :=
  sorry

/-- The initial state of clay pieces -/
def initial_state : ClayCollection :=
  { pieces := List.replicate 111 1 }

/-- The desired final state of clay pieces -/
def final_state : ClayCollection :=
  { pieces := [1, 2, 8, 9, 10, 11, 12, 13, 14, 15, 16] }

/-- Theorem stating that the transformation is possible in 2 operations -/
theorem clay_transformation_in_two_operations :
  ∃ (op1 op2 : Nat),
    (combine_operation (combine_operation initial_state op1) op2) = final_state :=
  sorry

end clay_transformation_in_two_operations_l498_49864


namespace solve_system_l498_49881

theorem solve_system (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 36) : x = 28 := by
  sorry

end solve_system_l498_49881


namespace quadratic_inequality_solution_l498_49836

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  (∀ x, 2*x^2 + b*x + a < 0 ↔ -2 < x ∧ x < 3) :=
by sorry

end quadratic_inequality_solution_l498_49836


namespace prime_divisor_ge_11_l498_49825

def is_valid_digit (d : Nat) : Prop := d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9

def all_digits_valid (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_valid_digit d

theorem prime_divisor_ge_11 (B : Nat) (h1 : B > 10) (h2 : all_digits_valid B) :
  ∃ p : Nat, p.Prime ∧ p ≥ 11 ∧ p ∣ B :=
sorry

end prime_divisor_ge_11_l498_49825


namespace volleyball_team_lineup_l498_49824

/-- The number of players in the volleyball team -/
def total_players : ℕ := 16

/-- The number of triplets -/
def num_triplets : ℕ := 3

/-- The number of twins -/
def num_twins : ℕ := 2

/-- The number of starters to be chosen -/
def num_starters : ℕ := 7

/-- The number of valid starting lineups -/
def valid_lineups : ℕ := 9778

theorem volleyball_team_lineup :
  (Nat.choose total_players num_starters) -
  (Nat.choose (total_players - num_triplets) (num_starters - num_triplets)) -
  (Nat.choose (total_players - num_twins) (num_starters - num_twins)) +
  (Nat.choose (total_players - num_triplets - num_twins) (num_starters - num_triplets - num_twins)) =
  valid_lineups :=
sorry

end volleyball_team_lineup_l498_49824


namespace subtract_fractions_l498_49874

theorem subtract_fractions : (7/3 * 12/5) - 3/5 = 5 := by
  sorry

end subtract_fractions_l498_49874


namespace apple_count_l498_49833

/-- Given a box of fruit with apples and oranges, prove that the number of apples is 14 -/
theorem apple_count (total_oranges : ℕ) (removed_oranges : ℕ) (apple_percentage : ℚ) : 
  total_oranges = 26 →
  removed_oranges = 20 →
  apple_percentage = 70 / 100 →
  (∃ (apples : ℕ), 
    (apples : ℚ) / ((apples : ℚ) + (total_oranges - removed_oranges : ℚ)) = apple_percentage ∧
    apples = 14) :=
by sorry

end apple_count_l498_49833


namespace paths_4x3_grid_l498_49884

/-- The number of unique paths in a grid -/
def grid_paths (m n : ℕ) : ℕ := (m + n).choose m

/-- Theorem: The number of unique paths in a 4x3 grid is 35 -/
theorem paths_4x3_grid : grid_paths 4 3 = 35 := by
  sorry

end paths_4x3_grid_l498_49884


namespace triangle_angle_B_l498_49822

/-- Given a triangle ABC with side lengths a and c, and angle A, proves that angle B has two possible values. -/
theorem triangle_angle_B (a c : ℝ) (A : ℝ) (h1 : a = 5 * Real.sqrt 2) (h2 : c = 10) (h3 : A = π / 6) :
  ∃ (B : ℝ), (B = π * 7 / 12 ∨ B = π / 12) := by
  sorry


end triangle_angle_B_l498_49822


namespace chairlift_halfway_l498_49844

def total_chairs : ℕ := 96
def current_chair : ℕ := 66

def halfway_chair (total : ℕ) (current : ℕ) : ℕ :=
  (current - total / 2 + total) % total

theorem chairlift_halfway :
  halfway_chair total_chairs current_chair = 18 := by
sorry

end chairlift_halfway_l498_49844


namespace complex_number_with_given_real_part_and_magnitude_l498_49898

theorem complex_number_with_given_real_part_and_magnitude (z : ℂ) : 
  (z.re = 5) → (Complex.abs z = Complex.abs (4 - 3*I)) → (z.im = 4) := by
  sorry

end complex_number_with_given_real_part_and_magnitude_l498_49898


namespace second_largest_power_of_ten_in_170_factorial_l498_49805

/-- The number of factors of 5 in the prime factorization of n! -/
def count_factors_of_five (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem second_largest_power_of_ten_in_170_factorial : 
  40 = (count_factors_of_five 170) - 1 :=
sorry

end second_largest_power_of_ten_in_170_factorial_l498_49805


namespace complex_division_l498_49875

theorem complex_division (z₁ z₂ : ℂ) : z₁ = 1 + I → z₂ = 1 - I → z₁ / z₂ = I := by
  sorry

end complex_division_l498_49875


namespace average_weight_increase_l498_49842

theorem average_weight_increase (initial_count : ℕ) (replaced_weight new_weight : ℝ) : 
  initial_count = 8 →
  replaced_weight = 65 →
  new_weight = 93 →
  (new_weight - replaced_weight) / initial_count = 3.5 :=
by sorry

end average_weight_increase_l498_49842


namespace thief_speed_l498_49891

/-- The speed of a thief given chase conditions -/
theorem thief_speed (initial_distance : ℝ) (policeman_speed : ℝ) (thief_distance : ℝ) : 
  initial_distance = 0.2 →
  policeman_speed = 10 →
  thief_distance = 0.8 →
  ∃ (thief_speed : ℝ), 
    thief_speed = 8 ∧ 
    (initial_distance + thief_distance) / policeman_speed = thief_distance / thief_speed :=
by sorry

end thief_speed_l498_49891


namespace area_equality_l498_49857

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if a quadrilateral is a parallelogram -/
def isParallelogram (q : Quadrilateral) : Prop := sorry

/-- Calculate the area of a quadrilateral -/
def areaQuadrilateral (q : Quadrilateral) : ℝ := sorry

/-- Check if a point is the midpoint of a line segment -/
def isMidpoint (M : Point) (A : Point) (B : Point) : Prop := sorry

/-- Calculate the area of a triangle -/
def areaTriangle (A : Point) (B : Point) (C : Point) : ℝ := sorry

/-- Main theorem -/
theorem area_equality 
  (C D E F G H J : Point)
  (CDEF : Quadrilateral)
  (h1 : isParallelogram CDEF)
  (h2 : areaQuadrilateral CDEF = 36)
  (h3 : isMidpoint G C D)
  (h4 : isMidpoint H E F) :
  areaTriangle C D J = areaQuadrilateral CDEF :=
sorry

end area_equality_l498_49857


namespace symmetric_complex_number_l498_49876

theorem symmetric_complex_number : ∀ z : ℂ, 
  (z.re = (-1 : ℝ) ∧ z.im = (1 : ℝ)) ↔ 
  (z.re = (2 / (Complex.I - 1)).re ∧ z.im = -(2 / (Complex.I - 1)).im) :=
by sorry

end symmetric_complex_number_l498_49876


namespace probability_not_math_and_physics_is_four_fifths_l498_49894

def subjects := 6
def selected := 3

def probability_not_math_and_physics : ℚ :=
  1 - (Nat.choose 4 1 : ℚ) / (Nat.choose subjects selected : ℚ)

theorem probability_not_math_and_physics_is_four_fifths :
  probability_not_math_and_physics = 4 / 5 := by
  sorry

end probability_not_math_and_physics_is_four_fifths_l498_49894


namespace consecutive_integers_around_sqrt3_l498_49849

theorem consecutive_integers_around_sqrt3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
sorry

end consecutive_integers_around_sqrt3_l498_49849


namespace morgans_blue_pens_l498_49886

theorem morgans_blue_pens (red_pens black_pens total_pens : ℕ) 
  (h1 : red_pens = 65)
  (h2 : black_pens = 58)
  (h3 : total_pens = 168)
  : total_pens - (red_pens + black_pens) = 45 := by
  sorry

end morgans_blue_pens_l498_49886


namespace tower_of_two_divisibility_l498_49807

def f : ℕ → ℕ
| 0 => 2
| (n + 1) => 2^(f n)

theorem tower_of_two_divisibility (n : ℕ) (h : n ≥ 2) :
  n ∣ (f n - f (n - 1)) :=
sorry

end tower_of_two_divisibility_l498_49807


namespace add_9999_seconds_to_5_45_00_l498_49839

/-- Represents a time of day in hours, minutes, and seconds -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time and returns the new time -/
def addSeconds (time : TimeOfDay) (seconds : Nat) : TimeOfDay :=
  sorry

/-- Converts a TimeOfDay to a string in the format "HH:MM:SS" -/
def TimeOfDay.toString (time : TimeOfDay) : String :=
  sorry

theorem add_9999_seconds_to_5_45_00 :
  let initialTime : TimeOfDay := ⟨17, 45, 0⟩
  let secondsToAdd : Nat := 9999
  let finalTime := addSeconds initialTime secondsToAdd
  finalTime.toString = "20:31:39" :=
sorry

end add_9999_seconds_to_5_45_00_l498_49839


namespace arithmetic_sequence_8th_term_l498_49810

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_8th_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_4th : a 4 = 23)
  (h_6th : a 6 = 47) :
  a 8 = 71 := by
sorry

end arithmetic_sequence_8th_term_l498_49810


namespace right_triangle_hypotenuse_l498_49866

/-- Proves that in a right triangle with non-hypotenuse side lengths of 5 and 12, the hypotenuse length is 13 -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ), 
  a = 5 → b = 12 → c^2 = a^2 + b^2 → c = 13 := by
  sorry

end right_triangle_hypotenuse_l498_49866


namespace unique_orthogonal_chord_l498_49820

-- Define the quadratic function
def f (p q x : ℝ) : ℝ := x^2 - 2*p*x + q

-- State the theorem
theorem unique_orthogonal_chord (p q : ℝ) :
  p > 0 ∧ q > 0 ∧  -- p and q are positive
  (∀ x, f p q x ≠ 0) ∧  -- graph doesn't intersect x-axis
  (∃! a, a > 0 ∧
    f p q (p - a) = f p q (p + a) ∧  -- AB parallel to x-axis
    (p - a) * (p + a) + (f p q (p - a))^2 = 0)  -- angle AOB = π/2
  → q = 1/4 := by
sorry

end unique_orthogonal_chord_l498_49820


namespace square_area_increase_l498_49887

theorem square_area_increase (a : ℝ) (ha : a > 0) : 
  let side_b := 2 * a
  let side_c := side_b * 1.8
  let area_a := a ^ 2
  let area_b := side_b ^ 2
  let area_c := side_c ^ 2
  (area_c - (area_a + area_b)) / (area_a + area_b) = 1.592 := by
  sorry

end square_area_increase_l498_49887


namespace harriets_age_l498_49892

theorem harriets_age (mother_age : ℕ) (peter_age : ℕ) (harriet_age : ℕ) : 
  mother_age = 60 →
  peter_age = mother_age / 2 →
  peter_age + 4 = 2 * (harriet_age + 4) →
  harriet_age = 13 := by
sorry

end harriets_age_l498_49892


namespace value_after_percentage_increase_l498_49879

theorem value_after_percentage_increase 
  (x : ℝ) (p : ℝ) (y : ℝ) 
  (h1 : x = 400) 
  (h2 : p = 20) :
  y = x * (1 + p / 100) → y = 480 := by
  sorry

end value_after_percentage_increase_l498_49879


namespace dog_weight_difference_l498_49895

theorem dog_weight_difference (labrador_initial : ℝ) (dachshund_initial : ℝ) 
  (growth_rate : ℝ) (h1 : labrador_initial = 40) (h2 : dachshund_initial = 12) 
  (h3 : growth_rate = 0.25) : 
  labrador_initial * (1 + growth_rate) - dachshund_initial * (1 + growth_rate) = 35 := by
  sorry

end dog_weight_difference_l498_49895


namespace front_top_area_ratio_l498_49830

/-- A rectangular box with given properties -/
structure Box where
  volume : ℝ
  side_area : ℝ
  top_area : ℝ
  front_area : ℝ
  top_side_ratio : ℝ

/-- The theorem stating the ratio of front face area to top face area -/
theorem front_top_area_ratio (b : Box) 
  (h_volume : b.volume = 5184)
  (h_side_area : b.side_area = 288)
  (h_top_side_ratio : b.top_area = 1.5 * b.side_area) :
  b.front_area / b.top_area = 1 / 2 := by
  sorry

#check front_top_area_ratio

end front_top_area_ratio_l498_49830


namespace two_number_difference_l498_49832

theorem two_number_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) :
  y - x = 80 / 7 := by
  sorry

end two_number_difference_l498_49832


namespace max_candy_leftover_l498_49846

theorem max_candy_leftover (x : ℕ) (h : x > 0) : 
  ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
by sorry

end max_candy_leftover_l498_49846


namespace janet_investment_interest_l498_49843

/-- Calculates the total interest earned from an investment --/
def total_interest (total_investment : ℝ) (high_rate_investment : ℝ) (high_rate : ℝ) (low_rate : ℝ) : ℝ :=
  let low_rate_investment := total_investment - high_rate_investment
  let high_rate_interest := high_rate_investment * high_rate
  let low_rate_interest := low_rate_investment * low_rate
  high_rate_interest + low_rate_interest

/-- Proves that Janet's investment yields $1,390 in interest --/
theorem janet_investment_interest :
  total_interest 31000 12000 0.10 0.01 = 1390 := by
  sorry

end janet_investment_interest_l498_49843


namespace glove_selection_theorem_l498_49867

theorem glove_selection_theorem :
  let total_gloves : ℕ := 8
  let gloves_to_select : ℕ := 4
  let num_pairs : ℕ := 4
  let total_selections : ℕ := Nat.choose total_gloves gloves_to_select
  let no_pair_selections : ℕ := 2^num_pairs
  total_selections - no_pair_selections = 54 :=
by sorry

end glove_selection_theorem_l498_49867


namespace product_remainder_theorem_l498_49871

theorem product_remainder_theorem (x : ℤ) : 
  (37 * x) % 31 = 15 ↔ ∃ k : ℤ, x = 18 + 31 * k := by sorry

end product_remainder_theorem_l498_49871


namespace incorrect_average_theorem_l498_49863

def incorrect_average (n : ℕ) (correct_avg : ℚ) (correct_num wrong_num : ℚ) : ℚ :=
  (n * correct_avg - correct_num + wrong_num) / n

theorem incorrect_average_theorem :
  incorrect_average 10 24 76 26 = 19 := by
  sorry

end incorrect_average_theorem_l498_49863


namespace current_speed_l498_49816

/-- Given a man's speed with and against a current, calculate the speed of the current. -/
theorem current_speed (speed_with_current speed_against_current : ℝ) 
  (h1 : speed_with_current = 16)
  (h2 : speed_against_current = 9.6) :
  ∃ (current_speed : ℝ), current_speed = 3.2 ∧ 
    speed_with_current = speed_against_current + 2 * current_speed :=
by
  sorry

#check current_speed

end current_speed_l498_49816


namespace digits_zeros_equality_l498_49889

/-- 
Given a positive integer n, count_digits n returns the sum of all digits in n.
-/
def count_digits (n : ℕ) : ℕ := sorry

/-- 
Given a positive integer n, count_zeros n returns the number of zeros in n.
-/
def count_zeros (n : ℕ) : ℕ := sorry

/-- 
sum_digits_to_n n returns the sum of digits of all numbers from 1 to n.
-/
def sum_digits_to_n (n : ℕ) : ℕ := sorry

/-- 
sum_zeros_to_n n returns the count of zeros in all numbers from 1 to n.
-/
def sum_zeros_to_n (n : ℕ) : ℕ := sorry

/-- 
For any positive integer k, the sum of digits in all numbers from 1 to 10^k
is equal to the count of zeros in all numbers from 1 to 10^(k+1).
-/
theorem digits_zeros_equality (k : ℕ) (h : k > 0) : 
  sum_digits_to_n (10^k) = sum_zeros_to_n (10^(k+1)) := by sorry

end digits_zeros_equality_l498_49889


namespace arctan_arcsin_sum_equals_pi_l498_49856

theorem arctan_arcsin_sum_equals_pi (x : ℝ) (h : x > 1) :
  2 * Real.arctan x + Real.arcsin (2 * x / (1 + x^2)) = π := by
sorry

end arctan_arcsin_sum_equals_pi_l498_49856


namespace min_dot_product_of_vectors_l498_49803

/-- Given plane vectors AC and BD, prove the minimum value of AB · CD -/
theorem min_dot_product_of_vectors (A B C D : ℝ × ℝ) : 
  (C.1 - A.1 = 1 ∧ C.2 - A.2 = 2) →  -- AC = (1, 2)
  (D.1 - B.1 = -2 ∧ D.2 - B.2 = 2) →  -- BD = (-2, 2)
  ∃ (min : ℝ), min = -9/4 ∧ 
    ∀ (AB CD : ℝ × ℝ), 
      AB.1 = B.1 - A.1 ∧ AB.2 = B.2 - A.2 →
      CD.1 = D.1 - C.1 ∧ CD.2 = D.2 - C.2 →
      AB.1 * CD.1 + AB.2 * CD.2 ≥ min :=
by sorry

end min_dot_product_of_vectors_l498_49803


namespace innings_played_l498_49815

/-- Represents the number of innings played by a cricket player. -/
def innings : ℕ := sorry

/-- Represents the current average runs of the player. -/
def currentAverage : ℕ := 24

/-- Represents the runs needed in the next innings. -/
def nextInningsRuns : ℕ := 96

/-- Represents the increase in average after the next innings. -/
def averageIncrease : ℕ := 8

/-- Theorem stating that the number of innings played is 8. -/
theorem innings_played : innings = 8 := by sorry

end innings_played_l498_49815


namespace union_S_T_l498_49837

def U : Finset Nat := {1,2,3,4,5,6}
def S : Finset Nat := {1,3,5}
def T : Finset Nat := {2,3,4,5}

theorem union_S_T : S ∪ T = {1,2,3,4,5} := by sorry

end union_S_T_l498_49837


namespace angle_QNR_is_165_l498_49850

/-- An isosceles triangle PQR with a point N inside -/
structure IsoscelesTriangleWithPoint where
  /-- The measure of angle PRQ in degrees -/
  angle_PRQ : ℝ
  /-- The measure of angle PNR in degrees -/
  angle_PNR : ℝ
  /-- The measure of angle PRN in degrees -/
  angle_PRN : ℝ
  /-- PR = QR (isosceles condition) -/
  isosceles : True
  /-- N is in the interior of the triangle -/
  N_interior : True
  /-- Angle PRQ is 108 degrees -/
  h_PRQ : angle_PRQ = 108
  /-- Angle PNR is 9 degrees -/
  h_PNR : angle_PNR = 9
  /-- Angle PRN is 21 degrees -/
  h_PRN : angle_PRN = 21

/-- Theorem: In the given isosceles triangle with point N, angle QNR is 165 degrees -/
theorem angle_QNR_is_165 (t : IsoscelesTriangleWithPoint) : ∃ angle_QNR : ℝ, angle_QNR = 165 := by
  sorry

end angle_QNR_is_165_l498_49850


namespace inverse_of_A_l498_49828

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 1/23, 2/23]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end inverse_of_A_l498_49828


namespace expression_factorization_l498_49808

theorem expression_factorization (x y : ℝ) :
  (3 * x^3 + 28 * x^2 * y + 4 * x) - (-4 * x^3 + 5 * x^2 * y - 4 * x) = x * (x + 8) * (7 * x + 1) := by
  sorry

end expression_factorization_l498_49808


namespace quadratic_equation_roots_l498_49870

theorem quadratic_equation_roots (c : ℝ) : 
  c = 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0 :=
by sorry

end quadratic_equation_roots_l498_49870


namespace quadratic_equations_roots_l498_49845

theorem quadratic_equations_roots :
  (∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 3 ∧ ∀ x : ℝ, x^2 - 3*x = 0 ↔ x = x₁ ∨ x = x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 5/4 ∧ x₂ = -1 ∧ ∀ x : ℝ, 4*x^2 - x - 5 = 0 ↔ x = x₁ ∨ x = x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -2/3 ∧ ∀ x : ℝ, 3*x*(x-1) = 2-2*x ↔ x = x₁ ∨ x = x₂) :=
by sorry

end quadratic_equations_roots_l498_49845


namespace athlete_heartbeats_l498_49814

/-- Calculates the total number of heartbeats during a race. -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  heart_rate * pace * race_distance

/-- Proves that the athlete's heart beats 21600 times during the 30-mile race. -/
theorem athlete_heartbeats :
  let heart_rate : ℕ := 120  -- heartbeats per minute
  let pace : ℕ := 6          -- minutes per mile
  let race_distance : ℕ := 30 -- miles
  total_heartbeats heart_rate pace race_distance = 21600 := by
  sorry

#eval total_heartbeats 120 6 30

end athlete_heartbeats_l498_49814


namespace jellybean_ratio_l498_49813

/-- Proves that the ratio of jellybeans Shannon refilled to the total taken out
    by Samantha and Shelby is 1/2, given the initial count, the amounts taken
    by Samantha and Shelby, and the final count. -/
theorem jellybean_ratio (initial : ℕ) (samantha_taken : ℕ) (shelby_taken : ℕ) (final : ℕ)
  (h1 : initial = 90)
  (h2 : samantha_taken = 24)
  (h3 : shelby_taken = 12)
  (h4 : final = 72) :
  (final - (initial - (samantha_taken + shelby_taken))) / (samantha_taken + shelby_taken) = 1 / 2 := by
  sorry

end jellybean_ratio_l498_49813


namespace rectangle_length_proof_l498_49877

theorem rectangle_length_proof (area_single : ℝ) (area_overlap : ℝ) (diagonal : ℝ) :
  area_single = 48 →
  area_overlap = 72 →
  diagonal = 6 →
  ∃ (length width : ℝ),
    length * width = area_single ∧
    length = 10 :=
by sorry

end rectangle_length_proof_l498_49877


namespace divisibility_of_expression_l498_49848

theorem divisibility_of_expression : ∃ k : ℤ, 27195^8 - 10887^8 + 10152^8 = 26460 * k := by
  sorry

end divisibility_of_expression_l498_49848


namespace sin_2alpha_value_l498_49858

theorem sin_2alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.sqrt 2 * Real.cos (2 * α) = Real.sin (α + π/4)) : 
  Real.sin (2 * α) = 3/4 := by
  sorry

end sin_2alpha_value_l498_49858


namespace sunzi_wood_measurement_problem_l498_49862

/-- The wood measurement problem from "The Mathematical Classic of Sunzi" -/
theorem sunzi_wood_measurement_problem (x y : ℝ) : 
  (y - x = 4.5 ∧ y / 2 = x - 1) ↔ 
  (∃ (rope_length wood_length : ℝ),
    rope_length > wood_length ∧
    rope_length - wood_length = 4.5 ∧
    rope_length / 2 > wood_length - 1 ∧
    rope_length / 2 < wood_length) :=
by sorry

end sunzi_wood_measurement_problem_l498_49862


namespace diamonds_G15_l498_49841

/-- The number of diamonds in the nth figure of the sequence -/
def diamonds (n : ℕ) : ℕ :=
  3 * n^2 - 3 * n + 1

/-- The sequence G is constructed such that for n ≥ 3, 
    Gₙ is surrounded by a hexagon with n-1 diamonds on each of its 6 sides -/
axiom sequence_construction (n : ℕ) (h : n ≥ 3) :
  diamonds n = diamonds (n-1) + 6 * (n-1)

/-- G₁ has 1 diamond -/
axiom G1_diamonds : diamonds 1 = 1

/-- The number of diamonds in G₁₅ is 631 -/
theorem diamonds_G15 : diamonds 15 = 631 := by
  sorry

end diamonds_G15_l498_49841


namespace apple_basket_problem_l498_49890

theorem apple_basket_problem :
  ∃ (a b : ℕ), 4 * a + 3 * a + 3 * b + 2 * b = 31 ∧ 3 * a + 2 * b = 13 :=
by sorry

end apple_basket_problem_l498_49890


namespace carly_butterfly_practice_l498_49821

/-- The number of hours Carly practices butterfly stroke per day -/
def butterfly_hours : ℝ := 3

/-- The number of days per week Carly practices butterfly stroke -/
def butterfly_days_per_week : ℕ := 4

/-- The number of hours Carly practices backstroke per day -/
def backstroke_hours : ℝ := 2

/-- The number of days per week Carly practices backstroke -/
def backstroke_days_per_week : ℕ := 6

/-- The total number of hours Carly practices swimming in a month -/
def total_hours_per_month : ℝ := 96

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

theorem carly_butterfly_practice :
  butterfly_hours * (butterfly_days_per_week * weeks_per_month) +
  backstroke_hours * (backstroke_days_per_week * weeks_per_month) =
  total_hours_per_month := by sorry

end carly_butterfly_practice_l498_49821


namespace pomelos_last_week_l498_49800

/-- Represents the number of pomelos in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of boxes shipped last week -/
def boxes_last_week : ℕ := 10

/-- Represents the number of boxes shipped this week -/
def boxes_this_week : ℕ := 20

/-- Represents the total number of dozens of pomelos shipped -/
def total_dozens : ℕ := 60

/-- Theorem stating that the number of pomelos shipped last week is 240 -/
theorem pomelos_last_week :
  (total_dozens * dozen) / (boxes_last_week + boxes_this_week) * boxes_last_week = 240 := by
  sorry


end pomelos_last_week_l498_49800


namespace greatest_prime_factor_of_154_l498_49812

theorem greatest_prime_factor_of_154 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 154 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 154 → q ≤ p :=
by sorry

end greatest_prime_factor_of_154_l498_49812


namespace figures_per_shelf_l498_49899

theorem figures_per_shelf (total_figures : ℕ) (num_shelves : ℕ) 
  (h1 : total_figures = 64) (h2 : num_shelves = 8) :
  total_figures / num_shelves = 8 := by
  sorry

end figures_per_shelf_l498_49899


namespace pool_wall_area_ratio_l498_49893

theorem pool_wall_area_ratio : 
  let pool_radius : ℝ := 20
  let wall_width : ℝ := 4
  let pool_area := π * pool_radius^2
  let total_area := π * (pool_radius + wall_width)^2
  let wall_area := total_area - pool_area
  wall_area / pool_area = 11 / 25 := by
sorry

end pool_wall_area_ratio_l498_49893


namespace blake_purchase_change_l498_49835

/-- The amount of change Blake will receive after his purchase -/
def blakes_change (lollipop_count : ℕ) (chocolate_pack_count : ℕ) (lollipop_price : ℕ) (bill_count : ℕ) (bill_value : ℕ) : ℕ :=
  let chocolate_pack_price := 4 * lollipop_price
  let total_cost := lollipop_count * lollipop_price + chocolate_pack_count * chocolate_pack_price
  let amount_paid := bill_count * bill_value
  amount_paid - total_cost

theorem blake_purchase_change :
  blakes_change 4 6 2 6 10 = 4 := by
  sorry

end blake_purchase_change_l498_49835


namespace solution_approximation_l498_49838

-- Define the equation
def equation (x y : ℝ) : Prop :=
  ((3 * x^2 - 7)^2 / 9) + 5 * y = x^3 - 2 * x

-- State the theorem
theorem solution_approximation :
  ∃ y : ℝ, equation 4 y ∧ abs (y + 26.155) < 0.001 := by
  sorry

end solution_approximation_l498_49838


namespace smallest_divisible_number_l498_49818

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 1) % 45 = 0 ∧
  (n + 1) % 60 = 0 ∧
  (n + 1) % 72 = 0 ∧
  (n + 1) % 81 = 0 ∧
  (n + 1) % 100 = 0 ∧
  (n + 1) % 120 = 0

theorem smallest_divisible_number :
  is_divisible_by_all 16199 ∧
  ∀ m : ℕ, m < 16199 → ¬is_divisible_by_all m :=
by sorry

end smallest_divisible_number_l498_49818


namespace julia_played_with_34_kids_l498_49801

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 17

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 15

/-- The number of kids Julia played with on Wednesday -/
def wednesday_kids : ℕ := 2

/-- The total number of kids Julia played with -/
def total_kids : ℕ := monday_kids + tuesday_kids + wednesday_kids

theorem julia_played_with_34_kids : total_kids = 34 := by
  sorry

end julia_played_with_34_kids_l498_49801


namespace point_on_graph_l498_49819

/-- The function f(x) = -2x + 3 --/
def f (x : ℝ) : ℝ := -2 * x + 3

/-- The point (1, 1) --/
def point : ℝ × ℝ := (1, 1)

/-- Theorem: The point (1, 1) lies on the graph of f(x) = -2x + 3 --/
theorem point_on_graph : f point.1 = point.2 := by
  sorry

end point_on_graph_l498_49819


namespace range_of_cosine_composition_l498_49888

theorem range_of_cosine_composition (x : ℝ) :
  0.5 ≤ Real.cos ((π / 9) * (Real.cos (2 * x) - 2 * Real.sin x)) ∧
  Real.cos ((π / 9) * (Real.cos (2 * x) - 2 * Real.sin x)) ≤ 1 := by
  sorry

end range_of_cosine_composition_l498_49888


namespace quadratic_function_range_l498_49882

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x

-- State the theorem
theorem quadratic_function_range :
  ∃ (a b : ℝ), a = -4 ∧ b = 5 ∧
  (∀ x, x ∈ Set.Icc 0 5 → f x ∈ Set.Icc a b) ∧
  (∀ y, y ∈ Set.Icc a b → ∃ x, x ∈ Set.Icc 0 5 ∧ f x = y) :=
by sorry

end quadratic_function_range_l498_49882


namespace salon_average_customers_l498_49809

def customers_per_day : List ℕ := [10, 12, 15, 13, 18, 16, 11]

def days_per_week : ℕ := 7

def average_daily_customers : ℚ :=
  (customers_per_day.sum : ℚ) / days_per_week

theorem salon_average_customers :
  average_daily_customers = 13.57 := by
  sorry

end salon_average_customers_l498_49809


namespace spring_sales_calculation_l498_49859

-- Define the total annual sandwich sales
def total_sales : ℝ := 15

-- Define the seasonal sales
def winter_sales : ℝ := 3
def summer_sales : ℝ := 4
def fall_sales : ℝ := 5

-- Define the winter sales percentage
def winter_percentage : ℝ := 0.2

-- Theorem to prove
theorem spring_sales_calculation :
  ∃ (spring_sales : ℝ),
    winter_percentage * total_sales = winter_sales ∧
    spring_sales + summer_sales + fall_sales + winter_sales = total_sales ∧
    spring_sales = 3 := by
  sorry


end spring_sales_calculation_l498_49859


namespace sum_a_b_c_value_l498_49855

theorem sum_a_b_c_value :
  ∀ (a b c : ℤ),
  (∀ x : ℤ, x < 0 → x ≤ a) →  -- a is the largest negative integer
  (abs b = 6) →               -- |b| = 6
  (c = -c) →                  -- c is equal to its opposite
  (a + b + c = -7 ∨ a + b + c = 5) := by
sorry

end sum_a_b_c_value_l498_49855


namespace polynomial_equation_sum_l498_49823

theorem polynomial_equation_sum (a b c d : ℤ) : 
  (∀ x, (x^2 + a*x + b) * (x + c) = x^3 + 5*x^2 - 6*x - 4) → 
  a + b + c + d = 11 := by
sorry

end polynomial_equation_sum_l498_49823
