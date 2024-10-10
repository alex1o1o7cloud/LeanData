import Mathlib

namespace roy_has_114_pens_l628_62843

/-- The number of pens Roy has -/
structure PenCounts where
  blue : ℕ
  black : ℕ
  red : ℕ
  green : ℕ
  purple : ℕ

/-- Roy's pen collection satisfies the given conditions -/
def satisfiesConditions (p : PenCounts) : Prop :=
  p.blue = 8 ∧
  p.black = 4 * p.blue ∧
  p.red = p.blue + p.black - 5 ∧
  p.green = p.red / 2 ∧
  p.purple = p.blue + p.green - 3

/-- The total number of pens Roy has -/
def totalPens (p : PenCounts) : ℕ :=
  p.blue + p.black + p.red + p.green + p.purple

/-- Theorem: Roy has 114 pens in total -/
theorem roy_has_114_pens :
  ∃ p : PenCounts, satisfiesConditions p ∧ totalPens p = 114 := by
  sorry

end roy_has_114_pens_l628_62843


namespace circle_diameter_l628_62823

theorem circle_diameter (x y : ℝ) (h : x + y = 100 * Real.pi) : ∃ (r : ℝ), 
  x = Real.pi * r^2 ∧ y = 2 * Real.pi * r ∧ 2 * r = 16 := by
  sorry

end circle_diameter_l628_62823


namespace function_value_ordering_l628_62876

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom even : ∀ x, f (-x) = f x
axiom periodic : ∀ x, f (x + 1) = f (x - 1)
axiom monotonic : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y

-- State the theorem
theorem function_value_ordering : f (-3/2) < f (4/3) ∧ f (4/3) < f 1 := by
  sorry

end function_value_ordering_l628_62876


namespace ellipse_line_slope_l628_62846

/-- Given an ellipse and a line intersecting it, prove the slope of the line --/
theorem ellipse_line_slope (A B : ℝ × ℝ) : 
  (∀ (x y : ℝ), x^2 / 4 + y^2 / 2 = 1 → (x, y) = A ∨ (x, y) = B) →  -- A and B are on the ellipse
  (1, 1) = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →                    -- (1, 1) is the midpoint of AB
  (B.2 - A.2) / (B.1 - A.1) = -1/2 :=                              -- The slope of AB is -1/2
by sorry

end ellipse_line_slope_l628_62846


namespace max_intersection_points_l628_62889

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 12

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 6

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := 990

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersection_points :
  (num_x_points.choose 2) * (num_y_points.choose 2) = max_intersections := by
  sorry

end max_intersection_points_l628_62889


namespace green_packs_count_l628_62893

-- Define the number of balls per pack
def balls_per_pack : ℕ := 10

-- Define the number of packs for red and yellow balls
def red_packs : ℕ := 4
def yellow_packs : ℕ := 8

-- Define the total number of balls
def total_balls : ℕ := 160

-- Define the number of packs of green balls
def green_packs : ℕ := (total_balls - (red_packs + yellow_packs) * balls_per_pack) / balls_per_pack

-- Theorem statement
theorem green_packs_count : green_packs = 4 := by
  sorry

end green_packs_count_l628_62893


namespace complex_expression_equality_l628_62801

theorem complex_expression_equality : 
  (2 + 7/9)^(1/2 : ℝ) + (1/10)^(-2 : ℝ) + (2 + 10/27)^(-(2/3) : ℝ) - Real.pi^(0 : ℝ) + 37/48 = 807/8 := by
  sorry

end complex_expression_equality_l628_62801


namespace extreme_value_and_inequality_l628_62867

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x + 1)

theorem extreme_value_and_inequality (a : ℝ) :
  (∃ x, f x = -1 ∧ ∀ y, f y ≥ f x) ∧
  (∀ x > 0, f x ≥ x + Real.log x + a + 1) ↔ a ≤ 1 :=
by sorry

end extreme_value_and_inequality_l628_62867


namespace percentage_of_male_students_l628_62894

theorem percentage_of_male_students (M F : ℝ) : 
  M + F = 100 →
  0.60 * M + 0.70 * F = 66 →
  M = 40 := by
sorry

end percentage_of_male_students_l628_62894


namespace water_ratio_horse_to_pig_l628_62882

/-- Proves that the ratio of water needed by a horse to a pig is 2:1 given the specified conditions -/
theorem water_ratio_horse_to_pig :
  let num_pigs : ℕ := 8
  let num_horses : ℕ := 10
  let water_per_pig : ℕ := 3
  let water_for_chickens : ℕ := 30
  let total_water : ℕ := 114
  let water_for_horses : ℕ := total_water - (num_pigs * water_per_pig) - water_for_chickens
  let water_per_horse : ℚ := water_for_horses / num_horses
  water_per_horse / water_per_pig = 2 / 1 :=
by
  sorry

end water_ratio_horse_to_pig_l628_62882


namespace derivative_y_wrt_x_at_zero_l628_62861

noncomputable def x (t : ℝ) : ℝ := Real.exp t * Real.cos t

noncomputable def y (t : ℝ) : ℝ := Real.exp t * Real.sin t

theorem derivative_y_wrt_x_at_zero :
  deriv (fun t => y t) 0 / deriv (fun t => x t) 0 = 1 := by sorry

end derivative_y_wrt_x_at_zero_l628_62861


namespace faster_car_distance_l628_62836

/-- Two cars driving towards each other, with one twice as fast as the other and initial distance of 4 miles -/
structure TwoCars where
  slow_speed : ℝ
  fast_speed : ℝ
  initial_distance : ℝ
  slow_distance : ℝ
  fast_distance : ℝ
  meeting_condition : slow_distance + fast_distance = initial_distance
  speed_relation : fast_speed = 2 * slow_speed
  distance_relation : fast_distance = 2 * slow_distance

/-- The theorem stating that the faster car travels 8/3 miles when they meet -/
theorem faster_car_distance (cars : TwoCars) (h : cars.initial_distance = 4) :
  cars.fast_distance = 8/3 := by
  sorry

#check faster_car_distance

end faster_car_distance_l628_62836


namespace min_score_for_higher_average_l628_62813

/-- Represents the scores of a student in four tests -/
structure Scores :=
  (test1 : ℕ) (test2 : ℕ) (test3 : ℕ) (test4 : ℕ)

/-- A-Long's scores -/
def aLong : Scores :=
  { test1 := 81, test2 := 81, test3 := 81, test4 := 81 }

/-- A-Hai's scores -/
def aHai : Scores :=
  { test1 := aLong.test1 + 1,
    test2 := aLong.test2 + 2,
    test3 := aLong.test3 + 3,
    test4 := 99 }

/-- The average score of a student -/
def average (s : Scores) : ℚ :=
  (s.test1 + s.test2 + s.test3 + s.test4) / 4

theorem min_score_for_higher_average :
  average aHai ≥ average aLong + 4 :=
by sorry

end min_score_for_higher_average_l628_62813


namespace probability_different_colors_is_83_128_l628_62857

/-- Represents the number of chips of each color in the bag -/
structure ChipCounts where
  blue : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the probability of drawing two chips of different colors -/
def probabilityDifferentColors (counts : ChipCounts) : ℚ :=
  let total := counts.blue + counts.yellow + counts.red
  let pBlue := counts.blue / total
  let pYellow := counts.yellow / total
  let pRed := counts.red / total
  pBlue * (pYellow + pRed) + pYellow * (pBlue + pRed) + pRed * (pBlue + pYellow)

/-- The main theorem stating the probability of drawing two chips of different colors -/
theorem probability_different_colors_is_83_128 :
  probabilityDifferentColors ⟨7, 5, 4⟩ = 83 / 128 := by
  sorry

end probability_different_colors_is_83_128_l628_62857


namespace sphere_polyhedra_radii_ratio_l628_62827

/-- The ratio of radii for a sequence of spheres inscribed in and circumscribed around
    regular polyhedra (octahedron, icosahedron, dodecahedron, tetrahedron, hexahedron) -/
theorem sphere_polyhedra_radii_ratio :
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ),
    r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧ r₄ > 0 ∧ r₅ > 0 ∧ r₆ > 0 ∧
    (r₂ / r₁ = Real.sqrt (9 + 4 * Real.sqrt 5)) ∧
    (r₃ / r₂ = Real.sqrt (27 + 12 * Real.sqrt 5)) ∧
    (r₄ / r₃ = 3 * Real.sqrt (5 + 2 * Real.sqrt 5)) ∧
    (r₅ / r₄ = 3 * Real.sqrt 15) ∧
    (r₆ / r₅ = 3 * Real.sqrt 5) :=
by sorry

end sphere_polyhedra_radii_ratio_l628_62827


namespace largest_n_binomial_equality_l628_62815

theorem largest_n_binomial_equality : ∃ n : ℕ, n = 6 ∧ 
  (∀ m : ℕ, (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m) → m ≤ n) := by
  sorry

end largest_n_binomial_equality_l628_62815


namespace num_true_propositions_l628_62840

-- Define the original proposition
def original_proposition (x y : ℝ) : Prop := x^2 > y^2 → x > y

-- Define the converse
def converse (x y : ℝ) : Prop := x > y → x^2 > y^2

-- Define the inverse
def inverse (x y : ℝ) : Prop := ¬(x^2 > y^2) → ¬(x > y)

-- Define the contrapositive
def contrapositive (x y : ℝ) : Prop := ¬(x > y) → ¬(x^2 > y^2)

-- Theorem statement
theorem num_true_propositions : 
  (∃ x y : ℝ, ¬(original_proposition x y)) ∧ 
  (∃ x y : ℝ, ¬(converse x y)) ∧ 
  (∃ x y : ℝ, ¬(inverse x y)) ∧ 
  (∃ x y : ℝ, ¬(contrapositive x y)) := by
  sorry

end num_true_propositions_l628_62840


namespace hash_sum_plus_five_l628_62833

-- Define the operation #
def hash (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

-- Theorem statement
theorem hash_sum_plus_five (a b : ℕ) : hash a b = 100 → (a + b) + 5 = 10 := by
  sorry

end hash_sum_plus_five_l628_62833


namespace cost_of_type_B_books_cost_equals_formula_l628_62831

/-- Given a total of 100 books to be purchased, with x books of type A,
    prove that the cost of purchasing type B books is 8(100-x) yuan,
    where the unit price of type B book is 8. -/
theorem cost_of_type_B_books (x : ℕ) : ℕ :=
  let total_books : ℕ := 100
  let unit_price_B : ℕ := 8
  let num_type_B : ℕ := total_books - x
  unit_price_B * num_type_B

#check cost_of_type_B_books

/-- Proof that the cost of type B books is 8(100-x) -/
theorem cost_equals_formula (x : ℕ) :
  cost_of_type_B_books x = 8 * (100 - x) :=
by sorry

#check cost_equals_formula

end cost_of_type_B_books_cost_equals_formula_l628_62831


namespace intersection_line_of_circles_l628_62873

/-- Given two circles in the xy-plane:
    Circle1: x^2 + y^2 - x + y - 2 = 0
    Circle2: x^2 + y^2 = 5
    This theorem states that the line x - y - 3 = 0 passes through their intersection points. -/
theorem intersection_line_of_circles (x y : ℝ) :
  (x^2 + y^2 - x + y - 2 = 0 ∧ x^2 + y^2 = 5) → (x - y - 3 = 0) := by
  sorry

end intersection_line_of_circles_l628_62873


namespace expected_scurries_eq_37_div_7_l628_62855

/-- Represents the number of people and horses -/
def n : ℕ := 8

/-- The probability that the i-th person scurries home -/
def scurry_prob (i : ℕ) : ℚ :=
  if i ≤ 1 then 0 else (i - 1 : ℚ) / i

/-- The expected number of people who scurry home -/
def expected_scurries : ℚ :=
  (Finset.range n).sum (λ i => scurry_prob (i + 1))

/-- Theorem stating that the expected number of people who scurry home is 37/7 -/
theorem expected_scurries_eq_37_div_7 :
  expected_scurries = 37 / 7 := by sorry

end expected_scurries_eq_37_div_7_l628_62855


namespace hcf_36_84_l628_62875

theorem hcf_36_84 : Nat.gcd 36 84 = 12 := by
  sorry

end hcf_36_84_l628_62875


namespace billy_reading_speed_l628_62807

/-- Represents Billy's reading speed in pages per hour -/
def reading_speed (
  free_time_per_day : ℕ)  -- Free time per day in hours
  (weekend_days : ℕ)      -- Number of weekend days
  (gaming_percentage : ℚ) -- Percentage of time spent gaming
  (pages_per_book : ℕ)    -- Number of pages in each book
  (books_read : ℕ)        -- Number of books read
  : ℚ :=
  let total_free_time := free_time_per_day * weekend_days
  let reading_time := total_free_time * (1 - gaming_percentage)
  let total_pages := pages_per_book * books_read
  total_pages / reading_time

theorem billy_reading_speed :
  reading_speed 8 2 (3/4) 80 3 = 60 := by
  sorry

end billy_reading_speed_l628_62807


namespace quadratic_roots_transformation_l628_62845

theorem quadratic_roots_transformation (K : ℝ) (α β : ℝ) : 
  (3 * α^2 + 7 * α + K = 0) →
  (3 * β^2 + 7 * β + K = 0) →
  (∃ m : ℝ, (α^2 - α)^2 + p * (α^2 - α) + m = 0 ∧ (β^2 - β)^2 + p * (β^2 - β) + m = 0) →
  p = -70/9 + 2*K/3 := by
sorry

end quadratic_roots_transformation_l628_62845


namespace original_average_proof_l628_62899

theorem original_average_proof (n : ℕ) (original_avg new_avg : ℚ) : 
  n = 12 → 
  new_avg = 72 → 
  new_avg = 2 * original_avg → 
  original_avg = 36 := by
sorry

end original_average_proof_l628_62899


namespace hyperbola_equation_l628_62825

/-- Given a hyperbola with equation x²/a² - y²/4 = 1 and an asymptote y = x/2,
    prove that the equation of the hyperbola is x²/16 - y²/4 = 1 -/
theorem hyperbola_equation (a : ℝ) :
  (∃ x y, x^2 / a^2 - y^2 / 4 = 1) →
  (∃ x, x / 2 = x / 2) →
  (∃ x y, x^2 / 16 - y^2 / 4 = 1) :=
by sorry

end hyperbola_equation_l628_62825


namespace zero_not_in_range_of_g_l628_62858

-- Define the function g
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (1 / (x + 3))
  else if x < -3 then Int.floor (1 / (x + 3))
  else 0  -- This value doesn't matter as g is undefined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g :
  ¬ ∃ (x : ℝ), g x = 0 :=
sorry

end zero_not_in_range_of_g_l628_62858


namespace remainder_101_37_mod_100_l628_62866

theorem remainder_101_37_mod_100 : 101^37 ≡ 1 [ZMOD 100] := by
  sorry

end remainder_101_37_mod_100_l628_62866


namespace class_average_weight_l628_62812

theorem class_average_weight (students_A : ℕ) (students_B : ℕ) (avg_weight_A : ℝ) (avg_weight_B : ℝ)
  (h1 : students_A = 24)
  (h2 : students_B = 16)
  (h3 : avg_weight_A = 40)
  (h4 : avg_weight_B = 35) :
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B : ℝ) = 38 := by
  sorry

end class_average_weight_l628_62812


namespace sin_15_sin_105_equals_1_l628_62897

theorem sin_15_sin_105_equals_1 : 4 * Real.sin (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end sin_15_sin_105_equals_1_l628_62897


namespace complex_multiplication_l628_62830

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (3 + 2*i)*i = -2 + 3*i := by
  sorry

end complex_multiplication_l628_62830


namespace square_difference_division_equals_318_l628_62862

theorem square_difference_division_equals_318 : (165^2 - 153^2) / 12 = 318 := by
  sorry

end square_difference_division_equals_318_l628_62862


namespace special_sequence_first_term_l628_62886

/-- An arithmetic sequence with common difference 2 where a₁, a₂, and a₄ form a geometric sequence -/
def special_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + 2) ∧  -- arithmetic sequence with difference 2
  ∃ r, a 2 = a 1 * r ∧ a 4 = a 2 * r  -- a₁, a₂, a₄ form geometric sequence

/-- The first term of the special sequence is 2 -/
theorem special_sequence_first_term (a : ℕ → ℝ) (h : special_sequence a) : a 1 = 2 :=
sorry

end special_sequence_first_term_l628_62886


namespace quadratic_equation_from_means_l628_62896

theorem quadratic_equation_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 6)
  (h_geometric : Real.sqrt (a * b) = 5) :
  ∀ x, x^2 - 12*x + 25 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end quadratic_equation_from_means_l628_62896


namespace intersection_circle_equation_l628_62883

-- Define the curves C₁ and C₂
def C₁ (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 2 + t * Real.cos a ∧ p.2 = 1 + t * Real.sin a}

def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2/2 = 1}

-- Define the intersection points M and N
def intersection_points : Set (ℝ × ℝ) :=
  C₁ (Real.pi/4) ∩ C₂

-- State the theorem
theorem intersection_circle_equation :
  ∀ M N : ℝ × ℝ,
  M ∈ intersection_points → N ∈ intersection_points → M ≠ N →
  ∀ P : ℝ × ℝ,
  P ∈ {P : ℝ × ℝ | (P.1 - 1/3)^2 + (P.2 + 2/3)^2 = 8/9} ↔
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t • M + (1 - t) • N) :=
by sorry


end intersection_circle_equation_l628_62883


namespace smallest_of_four_consecutive_even_numbers_l628_62878

theorem smallest_of_four_consecutive_even_numbers (x : ℤ) : 
  (∃ y z w : ℤ, y = x + 2 ∧ z = x + 4 ∧ w = x + 6 ∧ 
   x % 2 = 0 ∧ x + y + z + w = 140) → x = 32 := by
sorry

end smallest_of_four_consecutive_even_numbers_l628_62878


namespace wood_frog_count_l628_62870

theorem wood_frog_count (total : ℕ) (tree : ℕ) (poison : ℕ) (wood : ℕ) 
  (h1 : total = 78)
  (h2 : tree = 55)
  (h3 : poison = 10)
  (h4 : total = tree + poison + wood) : wood = 13 := by
  sorry

end wood_frog_count_l628_62870


namespace equivalent_representations_l628_62819

theorem equivalent_representations : 
  (16 : ℚ) / 20 = 24 / 30 ∧ 
  (16 : ℚ) / 20 = 80 / 100 ∧ 
  (16 : ℚ) / 20 = 4 / 5 ∧ 
  (16 : ℚ) / 20 = 0.8 := by
  sorry

end equivalent_representations_l628_62819


namespace four_weighings_sufficient_three_weighings_insufficient_l628_62879

/-- Represents the result of a weighing: lighter, equal, or heavier -/
inductive WeighingResult
  | Lighter
  | Equal
  | Heavier

/-- Represents a sequence of weighing results -/
def WeighingSequence := List WeighingResult

/-- The number of cans in the problem -/
def numCans : Nat := 80

/-- A function that simulates a weighing, returning a WeighingResult -/
def weighing (a b : Nat) : WeighingResult :=
  sorry

theorem four_weighings_sufficient :
  ∃ (f : Fin numCans → WeighingSequence),
    (∀ (i j : Fin numCans), i ≠ j → f i ≠ f j) ∧
    (∀ (s : WeighingSequence), s.length = 4) :=
  sorry

theorem three_weighings_insufficient :
  ¬∃ (f : Fin numCans → WeighingSequence),
    (∀ (i j : Fin numCans), i ≠ j → f i ≠ f j) ∧
    (∀ (s : WeighingSequence), s.length = 3) :=
  sorry

end four_weighings_sufficient_three_weighings_insufficient_l628_62879


namespace total_scoops_needed_l628_62816

def flour_cups : ℚ := 3
def sugar_cups : ℚ := 2
def scoop_size : ℚ := 1/3

theorem total_scoops_needed : 
  (flour_cups / scoop_size + sugar_cups / scoop_size : ℚ) = 15 := by
  sorry

end total_scoops_needed_l628_62816


namespace rectangle_tiles_l628_62849

theorem rectangle_tiles (length width : ℕ) : 
  width = 2 * length →
  (length * length + width * width : ℚ).sqrt = 45 →
  length * width = 810 :=
by
  sorry

end rectangle_tiles_l628_62849


namespace bacteria_growth_l628_62885

def b (t : ℝ) : ℝ := 105 + 104 * t - 1000 * t^2

theorem bacteria_growth (t : ℝ) :
  (deriv b 5 = 0) ∧
  (deriv b 10 = -10000) ∧
  (∀ t ∈ Set.Ioo 0 5, deriv b t > 0) ∧
  (∀ t ∈ Set.Ioi 5, deriv b t < 0) := by
  sorry

end bacteria_growth_l628_62885


namespace angle_equality_l628_62824

/-- Given a straight line split into two angles and a triangle with specific properties,
    prove that one of the angles equals 60 degrees. -/
theorem angle_equality (angle1 angle2 angle3 angle4 : ℝ) : 
  angle1 + angle2 = 180 →  -- Straight line condition
  angle1 + angle3 + 60 = 180 →  -- Triangle angle sum
  angle3 = angle4 →  -- Given equality
  angle4 = 60 := by
  sorry

end angle_equality_l628_62824


namespace chemistry_students_l628_62850

def basketball_team : ℕ := 18
def math_students : ℕ := 10
def physics_students : ℕ := 6
def math_and_physics : ℕ := 3
def all_three : ℕ := 2

theorem chemistry_students : ℕ := by
  -- The number of students studying chemistry is 7
  have h : basketball_team = math_students + physics_students - math_and_physics + (basketball_team - (math_students + physics_students - math_and_physics)) := by sorry
  -- Proof goes here
  sorry

#check chemistry_students -- Should evaluate to 7

end chemistry_students_l628_62850


namespace thirteen_ts_possible_l628_62822

/-- Represents a grid with horizontal and vertical lines -/
structure Grid :=
  (horizontal_lines : ℕ)
  (vertical_lines : ℕ)

/-- Represents a T shape on the grid -/
structure TShape :=
  (intersections : ℕ)

/-- The problem setup -/
def problem_setup : Prop :=
  ∃ (g : Grid) (t : TShape),
    g.horizontal_lines = 9 ∧
    g.vertical_lines = 9 ∧
    t.intersections = 5

/-- The theorem to be proved -/
theorem thirteen_ts_possible (h : problem_setup) : 
  ∃ (n : ℕ), n = 13 ∧ n * 5 ≤ 9 * 9 :=
sorry

end thirteen_ts_possible_l628_62822


namespace max_ratio_system_l628_62887

theorem max_ratio_system (x y z u : ℕ+) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) (h3 : x ≥ y) :
  (x : ℝ) / y ≤ 3 + 2 * Real.sqrt 2 ∧ ∀ ε > 0, ∃ x' y' z' u' : ℕ+,
    x' + y' = z' + u' ∧
    2 * x' * y' = z' * u' ∧
    x' ≥ y' ∧
    (x' : ℝ) / y' > 3 + 2 * Real.sqrt 2 - ε :=
sorry

end max_ratio_system_l628_62887


namespace A_time_to_complete_l628_62880

-- Define the rates of work for A, B, and C
variable (rA rB rC : ℝ)

-- Define the conditions
axiom AB_time : rA + rB = 1 / 2
axiom BC_time : rB + rC = 1 / 4
axiom AC_time : rA + rC = 5 / 12

-- Define the theorem
theorem A_time_to_complete : 1 / rA = 3 := by
  sorry

end A_time_to_complete_l628_62880


namespace jons_laundry_capacity_l628_62851

/-- Given information about Jon's laundry and machine capacity -/
structure LaundryInfo where
  shirts_per_pound : ℕ  -- Number of shirts that weigh 1 pound
  pants_per_pound : ℕ   -- Number of pairs of pants that weigh 1 pound
  total_shirts : ℕ      -- Total number of shirts to wash
  total_pants : ℕ       -- Total number of pants to wash
  loads : ℕ             -- Number of loads Jon has to do

/-- Calculate the machine capacity given laundry information -/
def machine_capacity (info : LaundryInfo) : ℚ :=
  let shirt_weight := info.total_shirts / info.shirts_per_pound
  let pants_weight := info.total_pants / info.pants_per_pound
  let total_weight := shirt_weight + pants_weight
  total_weight / info.loads

/-- Theorem stating Jon's laundry machine capacity -/
theorem jons_laundry_capacity :
  let info : LaundryInfo := {
    shirts_per_pound := 4,
    pants_per_pound := 2,
    total_shirts := 20,
    total_pants := 20,
    loads := 3
  }
  machine_capacity info = 5 := by sorry

end jons_laundry_capacity_l628_62851


namespace product_mod_seventeen_l628_62863

theorem product_mod_seventeen : (2024 * 2025 * 2026 * 2027 * 2028) % 17 = 6 := by
  sorry

end product_mod_seventeen_l628_62863


namespace money_redistribution_l628_62852

def initial_amount (i : Nat) : Nat :=
  2^(i-1) - 1

def final_amount (n : Nat) : Nat :=
  8 * (List.sum (List.map initial_amount (List.range n)))

theorem money_redistribution (n : Nat) :
  n = 9 → final_amount n = 512 := by sorry

end money_redistribution_l628_62852


namespace strip_to_upper_half_plane_l628_62837

-- Define the complex exponential function
noncomputable def complex_exp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- Define the mapping function
noncomputable def w (z : ℂ) (h : ℝ) : ℂ := complex_exp ((Real.pi * z) / h)

-- State the theorem
theorem strip_to_upper_half_plane (z : ℂ) (h : ℝ) (h_pos : h > 0) (z_in_strip : 0 < z.im ∧ z.im < h) :
  (w z h).im > 0 := by sorry

end strip_to_upper_half_plane_l628_62837


namespace somu_father_age_ratio_l628_62842

/-- Proves the ratio of Somu's age to his father's age -/
theorem somu_father_age_ratio :
  ∀ (S F : ℕ),
  S = 12 →
  S - 6 = (F - 6) / 5 →
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ S * b = F * a ∧ a = 1 ∧ b = 3 :=
by sorry

end somu_father_age_ratio_l628_62842


namespace shortest_path_equals_two_R_l628_62814

/-- A truncated cone with a specific angle between generatrix and larger base -/
structure TruncatedCone where
  R : ℝ  -- Radius of the larger base
  r : ℝ  -- Radius of the smaller base
  h : ℝ  -- Height of the truncated cone
  angle : ℝ  -- Angle between generatrix and larger base in radians

/-- The shortest path on the surface of a truncated cone -/
def shortestPath (cone : TruncatedCone) : ℝ := sorry

/-- Theorem stating that the shortest path is twice the radius of the larger base -/
theorem shortest_path_equals_two_R (cone : TruncatedCone) 
  (h₁ : cone.angle = π / 3)  -- 60 degrees in radians
  : shortestPath cone = 2 * cone.R := by
  sorry

end shortest_path_equals_two_R_l628_62814


namespace arithmetic_sequence_15th_term_l628_62841

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_15th_term 
  (a : ℕ → ℕ) 
  (h_arith : is_arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 12)
  (h_third : a 3 = 21) :
  a 15 = 129 :=
sorry

end arithmetic_sequence_15th_term_l628_62841


namespace jellybean_box_capacity_l628_62839

/-- Represents a rectangular box with a certain capacity of jellybeans -/
structure JellyBean_Box where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℕ

/-- Calculates the volume of a JellyBean_Box -/
def box_volume (box : JellyBean_Box) : ℝ :=
  box.height * box.width * box.length

/-- Theorem stating the relationship between box sizes and jellybean capacities -/
theorem jellybean_box_capacity 
  (box_b box_c : JellyBean_Box)
  (h_capacity_b : box_b.capacity = 125)
  (h_height : box_c.height = 2 * box_b.height)
  (h_width : box_c.width = 2 * box_b.width)
  (h_length : box_c.length = 2 * box_b.length) :
  box_c.capacity = 1000 :=
by sorry


end jellybean_box_capacity_l628_62839


namespace seven_thirteenths_repeating_block_l628_62874

/-- The least number of digits in the repeating block of the decimal expansion of 7/13 -/
def repeating_block_length : ℕ := 6

/-- Theorem stating that the least number of digits in the repeating block 
    of the decimal expansion of 7/13 is equal to repeating_block_length -/
theorem seven_thirteenths_repeating_block : 
  (Nat.lcm 13 10 : ℕ).factorization 2 + (Nat.lcm 13 10 : ℕ).factorization 5 = repeating_block_length :=
sorry

end seven_thirteenths_repeating_block_l628_62874


namespace students_per_normal_class_l628_62871

theorem students_per_normal_class
  (total_students : ℕ)
  (moving_percentage : ℚ)
  (grade_levels : ℕ)
  (advanced_class_size : ℕ)
  (normal_classes_per_grade : ℕ)
  (h1 : total_students = 1590)
  (h2 : moving_percentage = 40 / 100)
  (h3 : grade_levels = 3)
  (h4 : advanced_class_size = 20)
  (h5 : normal_classes_per_grade = 6)
  : ℕ :=
by
  -- Proof goes here
  sorry

#check @students_per_normal_class

end students_per_normal_class_l628_62871


namespace octagon_area_l628_62853

/-- The area of a regular octagon formed by cutting corners from a square --/
theorem octagon_area (m : ℝ) : 
  let square_side : ℝ := 2 * m
  let octagon_area : ℝ := 4 * (Real.sqrt 2 - 1) * m^2
  octagon_area = 
    square_side^2 - 4 * (1/2 * (m * (2 - Real.sqrt 2))^2) :=
by sorry

end octagon_area_l628_62853


namespace smallest_integer_with_consecutive_sums_l628_62811

theorem smallest_integer_with_consecutive_sums : ∃ n : ℕ, 
  (∃ a : ℤ, n = (9 * a + 36)) ∧ 
  (∃ b : ℤ, n = (10 * b + 45)) ∧ 
  (∃ c : ℤ, n = (11 * c + 55)) ∧ 
  (∀ m : ℕ, m < n → 
    (¬∃ x : ℤ, m = (9 * x + 36)) ∨ 
    (¬∃ y : ℤ, m = (10 * y + 45)) ∨ 
    (¬∃ z : ℤ, m = (11 * z + 55))) ∧ 
  n = 495 :=
by sorry

end smallest_integer_with_consecutive_sums_l628_62811


namespace election_votes_l628_62832

theorem election_votes (V : ℝ) 
  (h1 : V > 0) -- Ensure total votes is positive
  (h2 : ∃ (x : ℝ), x = 0.25 * V ∧ x + 4000 = V - x) : V = 8000 := by
  sorry

end election_votes_l628_62832


namespace chloe_age_sum_of_digits_l628_62865

/-- Represents a person's age -/
structure Age :=
  (value : ℕ)

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- Represents the family's ages and their properties -/
structure FamilyAges :=
  (joey : Age)
  (chloe : Age)
  (max : Age)
  (joey_chloe_diff : joey.value = chloe.value + 2)
  (max_age : max.value = 2)
  (joey_multiple_of_max : ∃ k : ℕ, joey.value = k * max.value)
  (future_multiples : ∃ n₁ n₂ n₃ n₄ n₅ : ℕ, 
    (joey.value + n₁) % (max.value + n₁) = 0 ∧
    (joey.value + n₂) % (max.value + n₂) = 0 ∧
    (joey.value + n₃) % (max.value + n₃) = 0 ∧
    (joey.value + n₄) % (max.value + n₄) = 0 ∧
    (joey.value + n₅) % (max.value + n₅) = 0)

theorem chloe_age_sum_of_digits (family : FamilyAges) :
  ∃ n : ℕ, n > 0 ∧ 
    (family.chloe.value + n) % (family.max.value + n) = 0 ∧
    sumOfDigits (family.chloe.value + n) = 10 :=
  sorry

end chloe_age_sum_of_digits_l628_62865


namespace problem_statement_l628_62844

theorem problem_statement (x y : ℝ) (h : -y + 3*x = 3) : 
  2*(y - 3*x) - (3*x - y)^2 + 1 = -14 := by sorry

end problem_statement_l628_62844


namespace science_club_neither_math_nor_physics_l628_62891

theorem science_club_neither_math_nor_physics 
  (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) 
  (h1 : total = 120)
  (h2 : math = 75)
  (h3 : physics = 50)
  (h4 : both = 15) :
  total - (math + physics - both) = 10 :=
by sorry

end science_club_neither_math_nor_physics_l628_62891


namespace probability_of_white_ball_l628_62829

def total_balls : ℕ := 4 + 6
def white_balls : ℕ := 4
def yellow_balls : ℕ := 6

theorem probability_of_white_ball :
  (white_balls : ℚ) / total_balls = 2 / 5 :=
sorry

end probability_of_white_ball_l628_62829


namespace factor_expression_l628_62817

theorem factor_expression (x : ℝ) : 18 * x^2 + 9 * x - 3 = 3 * (6 * x^2 + 3 * x - 1) := by
  sorry

end factor_expression_l628_62817


namespace train_passengers_l628_62800

theorem train_passengers (initial_passengers : ℕ) (stops : ℕ) : 
  initial_passengers = 64 → stops = 4 → 
  (initial_passengers : ℚ) * ((2 : ℚ) / 3) ^ stops = 1024 / 81 := by
  sorry

end train_passengers_l628_62800


namespace triangle_angle_problem_l628_62810

/-- A prime number greater than 2 -/
def OddPrime (n : ℕ) : Prop := Nat.Prime n ∧ n > 2

theorem triangle_angle_problem :
  ∀ y z w : ℕ,
    OddPrime y →
    OddPrime z →
    OddPrime w →
    y + z + w = 90 →
    (∀ w' : ℕ, OddPrime w' → y + z + w' = 90 → w ≤ w') →
    w = 83 := by
  sorry

end triangle_angle_problem_l628_62810


namespace sarah_time_hours_l628_62828

-- Define the time Samuel took in minutes
def samuel_time : ℕ := 30

-- Define the time difference between Sarah and Samuel in minutes
def time_difference : ℕ := 48

-- Define Sarah's time in minutes
def sarah_time_minutes : ℕ := samuel_time + time_difference

-- Define the conversion factor from minutes to hours
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem sarah_time_hours : 
  (sarah_time_minutes : ℚ) / minutes_per_hour = 1.3 := by
  sorry

end sarah_time_hours_l628_62828


namespace bug_path_tiles_l628_62872

def floor_width : ℕ := 10
def floor_length : ℕ := 17

theorem bug_path_tiles : 
  floor_width + floor_length - Nat.gcd floor_width floor_length = 26 := by
  sorry

end bug_path_tiles_l628_62872


namespace happy_snakes_not_purple_l628_62859

structure Snake where
  happy : Bool
  purple : Bool
  canAdd : Bool
  canSubtract : Bool

def TomSnakes : Set Snake := sorry

theorem happy_snakes_not_purple :
  ∀ s ∈ TomSnakes,
  (s.happy → s.canAdd) ∧
  (s.purple → ¬s.canSubtract) ∧
  (¬s.canSubtract → ¬s.canAdd) →
  (s.happy → ¬s.purple) := by
  sorry

#check happy_snakes_not_purple

end happy_snakes_not_purple_l628_62859


namespace fraction_simplification_l628_62835

theorem fraction_simplification (m n : ℚ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (24 * m^3 * n^4) / (32 * m^4 * n^2) = (3 * n^2) / (4 * m) := by
  sorry

end fraction_simplification_l628_62835


namespace halfway_point_between_fractions_l628_62895

theorem halfway_point_between_fractions :
  (1 / 12 + 1 / 14) / 2 = 13 / 168 := by
  sorry

end halfway_point_between_fractions_l628_62895


namespace mean_of_remaining_numbers_l628_62864

theorem mean_of_remaining_numbers : 
  let numbers : List ℕ := [1867, 1993, 2019, 2025, 2109, 2121]
  let total_sum : ℕ := numbers.sum
  let mean_of_four : ℕ := 2008
  let sum_of_four : ℕ := 4 * mean_of_four
  let sum_of_two : ℕ := total_sum - sum_of_four
  sum_of_two / 2 = 2051 := by sorry

end mean_of_remaining_numbers_l628_62864


namespace solution_implies_m_minus_n_abs_l628_62838

/-- Given a system of equations 2x - y = m and x + my = n with solution x = 2 and y = 1, 
    prove that |m - n| = 2 -/
theorem solution_implies_m_minus_n_abs (m n : ℝ) 
  (h1 : 2 * 2 - 1 = m) 
  (h2 : 2 + m * 1 = n) : 
  |m - n| = 2 := by
  sorry

end solution_implies_m_minus_n_abs_l628_62838


namespace candidate_x_win_percentage_l628_62877

theorem candidate_x_win_percentage :
  ∀ (total_voters : ℕ) (republican_ratio democrat_ratio : ℚ) 
    (republican_for_x democrat_for_x : ℚ),
  republican_ratio / democrat_ratio = 3 / 2 →
  republican_for_x = 70 / 100 →
  democrat_for_x = 25 / 100 →
  let republicans := (republican_ratio / (republican_ratio + democrat_ratio)) * total_voters
  let democrats := (democrat_ratio / (republican_ratio + democrat_ratio)) * total_voters
  let votes_for_x := republican_for_x * republicans + democrat_for_x * democrats
  let votes_for_y := total_voters - votes_for_x
  (votes_for_x - votes_for_y) / total_voters = 4 / 100 :=
by sorry

end candidate_x_win_percentage_l628_62877


namespace roger_bike_distance_l628_62803

/-- Calculates the total distance Roger rode his bike over three sessions -/
theorem roger_bike_distance (morning_distance : ℝ) (evening_multiplier : ℝ) (km_per_mile : ℝ) : 
  morning_distance = 2 →
  evening_multiplier = 5 →
  km_per_mile = 1.6 →
  morning_distance + (evening_multiplier * morning_distance) + 
    (2 * morning_distance * km_per_mile / km_per_mile) = 16 := by
  sorry


end roger_bike_distance_l628_62803


namespace right_triangle_circumscribed_circle_radius_l628_62805

theorem right_triangle_circumscribed_circle_radius 
  (a b c R : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_a : a = 5) 
  (h_b : b = 12) 
  (h_R : R = c / 2) : R = 13 / 2 := by
  sorry

end right_triangle_circumscribed_circle_radius_l628_62805


namespace prob_different_numbers_is_three_fourths_l628_62804

/-- Men's team has 3 players -/
def num_men : ℕ := 3

/-- Women's team has 4 players -/
def num_women : ℕ := 4

/-- Total number of possible outcomes when selecting one player from each team -/
def total_outcomes : ℕ := num_men * num_women

/-- Number of outcomes where players have the same number -/
def same_number_outcomes : ℕ := min num_men num_women

/-- Probability of selecting players with different numbers -/
def prob_different_numbers : ℚ := 1 - (same_number_outcomes : ℚ) / total_outcomes

theorem prob_different_numbers_is_three_fourths : 
  prob_different_numbers = 3/4 := by sorry

end prob_different_numbers_is_three_fourths_l628_62804


namespace linear_equations_l628_62826

-- Define what a linear equation is
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- Define the equations
def eq1 : ℝ → ℝ := λ _ => 12
def eq2 : ℝ → ℝ := λ x => 5 * x + 3
def eq3 : ℝ → ℝ → ℝ := λ x y => 2 * x + 3 * y
def eq4 : ℝ → ℝ := λ a => 2 * a - 1
def eq5 : ℝ → ℝ := λ x => 2 * x^2 + x

-- Theorem statement
theorem linear_equations :
  (¬ is_linear_equation eq1) ∧
  (is_linear_equation eq2) ∧
  (¬ is_linear_equation (λ x => eq3 x 0)) ∧
  (is_linear_equation eq4) ∧
  (¬ is_linear_equation eq5) :=
sorry

end linear_equations_l628_62826


namespace ratio_equality_l628_62869

theorem ratio_equality (x y z : ℝ) (h : x / 3 = y / 4 ∧ y / 4 = z / 5) :
  (x + y - z) / (2 * x - y + z) = 2 / 7 := by
  sorry

end ratio_equality_l628_62869


namespace gcd_90_270_l628_62892

theorem gcd_90_270 : Nat.gcd 90 270 = 90 := by
  sorry

end gcd_90_270_l628_62892


namespace heat_engine_efficiency_l628_62834

theorem heat_engine_efficiency
  (η₀ η₁ η₂ α : ℝ)
  (h1 : η₁ < η₀)
  (h2 : η₂ < η₀)
  (h3 : η₀ < 1)
  (h4 : η₁ < 1)
  (h5 : η₂ = (η₀ - η₁) / (1 - η₁))
  (h6 : η₁ = (1 - 0.01 * α) * η₀) :
  η₂ = α / (100 - (100 - α) * η₀) := by
sorry

end heat_engine_efficiency_l628_62834


namespace intersection_condition_subset_condition_l628_62847

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}

-- Define set B as a function of m
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m-3)*x + m^2 - 3*m ≤ 0}

-- Part 1: Intersection condition
theorem intersection_condition (m : ℝ) : A ∩ B m = Set.Icc 2 4 → m = 5 := by sorry

-- Part 2: Subset condition
theorem subset_condition (m : ℝ) : A ⊆ (Set.univ \ B m) → m < -2 ∨ m > 7 := by sorry

end intersection_condition_subset_condition_l628_62847


namespace hospital_staff_count_l628_62808

theorem hospital_staff_count (total : ℕ) (doctor_ratio nurse_ratio : ℕ) 
  (h1 : total = 500)
  (h2 : doctor_ratio = 7)
  (h3 : nurse_ratio = 8) : 
  ∃ (nurses : ℕ), nurses = 264 ∧ 
    ∃ (doctors : ℕ), doctors + nurses = total ∧ 
      doctor_ratio * nurses = nurse_ratio * doctors :=
sorry

end hospital_staff_count_l628_62808


namespace ratio_sum_squares_theorem_l628_62818

theorem ratio_sum_squares_theorem (x y z : ℝ) : 
  y = 2 * x ∧ z = 3 * x ∧ x^2 + y^2 + z^2 = 2744 → x + y + z = 84 := by
  sorry

end ratio_sum_squares_theorem_l628_62818


namespace courtney_marble_count_l628_62821

/-- The number of marbles in Courtney's collection -/
def total_marbles (jar1 jar2 jar3 : ℕ) : ℕ := jar1 + jar2 + jar3

/-- Theorem: Courtney's total marble count -/
theorem courtney_marble_count :
  ∀ (jar1 jar2 jar3 : ℕ),
    jar1 = 80 →
    jar2 = 2 * jar1 →
    jar3 = jar1 / 4 →
    total_marbles jar1 jar2 jar3 = 260 := by
  sorry

#check courtney_marble_count

end courtney_marble_count_l628_62821


namespace value_of_2a_plus_b_l628_62820

-- Define the functions f, g, and h
def f (a b : ℝ) (x : ℝ) : ℝ := a * x - b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem value_of_2a_plus_b (a b : ℝ) :
  (∀ x, h a b x = x - 9) →
  2 * a + b = 7 := by
sorry

end value_of_2a_plus_b_l628_62820


namespace toddler_difference_l628_62898

/-- Represents the group of toddlers playing in the sandbox. -/
structure ToddlerGroup where
  total : ℕ
  forgot_bucket : ℕ
  forgot_shovel : ℕ
  bucket_implies_shovel : Bool

/-- The difference between toddlers with shovel but no bucket and toddlers with bucket -/
def shovel_no_bucket_minus_bucket (group : ToddlerGroup) : ℕ :=
  (group.total - group.forgot_shovel) - (group.total - group.forgot_bucket) - (group.total - group.forgot_bucket)

/-- The main theorem stating the difference is 4 -/
theorem toddler_difference (group : ToddlerGroup) 
  (h1 : group.total = 12)
  (h2 : group.forgot_bucket = 9)
  (h3 : group.forgot_shovel = 2)
  (h4 : group.bucket_implies_shovel = true) :
  shovel_no_bucket_minus_bucket group = 4 := by
  sorry

end toddler_difference_l628_62898


namespace division_relation_l628_62802

theorem division_relation (h : 2994 / 14.5 = 171) : 29.94 / 1.75 = 17.1 := by
  sorry

end division_relation_l628_62802


namespace squares_theorem_l628_62809

-- Define the points and lengths
variable (A B C O : ℝ × ℝ)
variable (a b c : ℝ)

-- Define the conditions
def squares_condition (A B C O : ℝ × ℝ) (a b c : ℝ) : Prop :=
  A = (a, a) ∧
  B = (b, 2*a + b) ∧
  C = (-c, c) ∧
  O = (0, 0) ∧
  c = a + b

-- Define the equality of line segments
def line_segments_equal (P Q R S : ℝ × ℝ) : Prop :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (R.1 - S.1)^2 + (R.2 - S.2)^2

-- Define perpendicularity of line segments
def perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (Q.1 - P.1) * (S.1 - R.1) + (Q.2 - P.2) * (S.2 - R.2) = 0

-- State the theorem
theorem squares_theorem (A B C O : ℝ × ℝ) (a b c : ℝ) 
  (h : squares_condition A B C O a b c) : 
  line_segments_equal O B A C ∧ perpendicular O B A C := by
  sorry

end squares_theorem_l628_62809


namespace common_chord_intersection_l628_62884

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point where two circles intersect -/
def IntersectionPoint (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
       (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2}

/-- The common chord of two intersecting circles -/
def CommonChord (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  IntersectionPoint c1 c2

/-- Theorem: For any three circles in a plane that intersect pairwise, 
    the common chords of these pairs of circles intersect at a single point -/
theorem common_chord_intersection (c1 c2 c3 : Circle) 
  (h12 : (CommonChord c1 c2).Nonempty)
  (h23 : (CommonChord c2 c3).Nonempty)
  (h31 : (CommonChord c3 c1).Nonempty) :
  ∃ p, p ∈ CommonChord c1 c2 ∧ p ∈ CommonChord c2 c3 ∧ p ∈ CommonChord c3 c1 :=
sorry

end common_chord_intersection_l628_62884


namespace new_boarders_count_l628_62854

/-- Represents the number of boarders and day students at a school -/
structure SchoolPopulation where
  boarders : ℕ
  dayStudents : ℕ

/-- Represents the ratio of boarders to day students -/
structure Ratio where
  boarders : ℕ
  dayStudents : ℕ

def initialPopulation : SchoolPopulation :=
  { boarders := 150, dayStudents := 360 }

def initialRatio : Ratio :=
  { boarders := 5, dayStudents := 12 }

def finalRatio : Ratio :=
  { boarders := 1, dayStudents := 2 }

/-- The theorem to be proved -/
theorem new_boarders_count (newBoarders : ℕ) :
  (initialPopulation.boarders + newBoarders) / initialPopulation.dayStudents = 
    finalRatio.boarders / finalRatio.dayStudents ∧
  initialPopulation.boarders / initialPopulation.dayStudents = 
    initialRatio.boarders / initialRatio.dayStudents →
  newBoarders = 30 := by
  sorry

end new_boarders_count_l628_62854


namespace shipping_cost_formula_l628_62881

/-- The shipping cost function for a parcel and flat-rate envelope -/
def shippingCost (P : ℝ) : ℝ :=
  let firstPoundFee : ℝ := 12
  let additionalPoundFee : ℝ := 5
  let flatRateEnvelopeFee : ℝ := 20
  firstPoundFee + additionalPoundFee * (P - 1) + flatRateEnvelopeFee

theorem shipping_cost_formula (P : ℝ) :
  shippingCost P = 5 * P + 27 := by
  sorry

end shipping_cost_formula_l628_62881


namespace cinnamon_swirl_sharing_l628_62806

theorem cinnamon_swirl_sharing (total_swirls : ℕ) (jane_pieces : ℕ) (people : ℕ) : 
  total_swirls = 12 →
  jane_pieces = 4 →
  total_swirls % jane_pieces = 0 →
  total_swirls / jane_pieces = people →
  people = 3 := by
  sorry

end cinnamon_swirl_sharing_l628_62806


namespace residue_sum_mod_19_l628_62848

theorem residue_sum_mod_19 : (8^1356 + 7^1200) % 19 = 10 := by
  sorry

end residue_sum_mod_19_l628_62848


namespace tangent_line_at_one_l628_62890

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - 4 * (x - 1)

noncomputable def f' (x : ℝ) : ℝ := Real.log x + (x + 1) / x - 4

theorem tangent_line_at_one (x y : ℝ) :
  (f' 1 = -2) →
  (f 1 = 0) →
  (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → |f (1 + h) - (f 1 + f' 1 * h)| ≤ ε * |h|) →
  (2 * x + y - 2 = 0 ↔ y = f' 1 * (x - 1) + f 1) :=
by sorry

end tangent_line_at_one_l628_62890


namespace dealer_profit_percentage_l628_62856

/-- Calculates the profit percentage for a dealer's transaction -/
def profit_percentage (purchase_quantity : ℕ) (purchase_price : ℚ) 
                      (sale_quantity : ℕ) (sale_price : ℚ) : ℚ :=
  let cost_per_article := purchase_price / purchase_quantity
  let sale_per_article := sale_price / sale_quantity
  let profit_per_article := sale_per_article - cost_per_article
  (profit_per_article / cost_per_article) * 100

/-- The profit percentage for the given dealer transaction is approximately 89.99% -/
theorem dealer_profit_percentage :
  let result := profit_percentage 15 25 12 38
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 / 100) ∧ |result - 8999 / 100| < ε :=
sorry

end dealer_profit_percentage_l628_62856


namespace second_train_length_l628_62888

-- Define constants
def train1_speed : Real := 60  -- km/hr
def train2_speed : Real := 40  -- km/hr
def crossing_time : Real := 11.159107271418288  -- seconds
def train1_length : Real := 140  -- meters

-- Define the theorem
theorem second_train_length :
  let relative_speed := (train1_speed + train2_speed) * (5/18)  -- Convert km/hr to m/s
  let total_distance := relative_speed * crossing_time
  let train2_length := total_distance - train1_length
  train2_length = 170 := by
  sorry

end second_train_length_l628_62888


namespace cos_increasing_interval_l628_62868

theorem cos_increasing_interval (a : Real) : 
  (∀ x₁ x₂, -Real.pi ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a → Real.cos x₁ < Real.cos x₂) → 
  a ≤ 0 := by
  sorry

end cos_increasing_interval_l628_62868


namespace cubic_sum_minus_product_l628_62860

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_products_eq : a*b + a*c + b*c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 1008 := by sorry

end cubic_sum_minus_product_l628_62860
