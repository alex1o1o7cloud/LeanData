import Mathlib

namespace NUMINAMATH_CALUDE_triangle_point_coordinates_l3481_348195

/-- Given a triangle ABC with median CM and angle bisector BL, prove that the coordinates of C are (14, 2) -/
theorem triangle_point_coordinates (A M L : ℝ × ℝ) :
  A = (2, 8) →
  M = (4, 11) →
  L = (6, 6) →
  ∃ (B C : ℝ × ℝ),
    (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) ∧  -- M is the midpoint of AB
    (∃ (t : ℝ), L = (1 - t) • B + t • C) ∧      -- L lies on BC
    (∃ (s : ℝ), C = (1 - s) • A + s • B) ∧     -- C lies on AB
    (C.1 = 14 ∧ C.2 = 2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_point_coordinates_l3481_348195


namespace NUMINAMATH_CALUDE_probability_at_least_one_red_l3481_348193

def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def selected_balls : ℕ := 2

theorem probability_at_least_one_red :
  (1 : ℚ) - (Nat.choose white_balls selected_balls : ℚ) / (Nat.choose total_balls selected_balls : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_red_l3481_348193


namespace NUMINAMATH_CALUDE_larger_number_proof_l3481_348159

theorem larger_number_proof (A B : ℕ+) (h1 : Nat.gcd A B = 28) 
  (h2 : Nat.lcm A B = 28 * 12 * 15) : max A B = 420 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3481_348159


namespace NUMINAMATH_CALUDE_roots_distribution_l3481_348157

/-- The polynomial p(z) = z^6 + 6z + 10 -/
def p (z : ℂ) : ℂ := z^6 + 6*z + 10

/-- The number of roots of p(z) in the first quadrant -/
def roots_first_quadrant : ℕ := 1

/-- The number of roots of p(z) in the second quadrant -/
def roots_second_quadrant : ℕ := 2

/-- The number of roots of p(z) in the third quadrant -/
def roots_third_quadrant : ℕ := 2

/-- The number of roots of p(z) in the fourth quadrant -/
def roots_fourth_quadrant : ℕ := 1

theorem roots_distribution :
  (∃ (z : ℂ), z.re > 0 ∧ z.im > 0 ∧ p z = 0) ∧
  (∃ (z1 z2 : ℂ), z1 ≠ z2 ∧ z1.re < 0 ∧ z1.im > 0 ∧ z2.re < 0 ∧ z2.im > 0 ∧ p z1 = 0 ∧ p z2 = 0) ∧
  (∃ (z1 z2 : ℂ), z1 ≠ z2 ∧ z1.re < 0 ∧ z1.im < 0 ∧ z2.re < 0 ∧ z2.im < 0 ∧ p z1 = 0 ∧ p z2 = 0) ∧
  (∃ (z : ℂ), z.re > 0 ∧ z.im < 0 ∧ p z = 0) :=
sorry

end NUMINAMATH_CALUDE_roots_distribution_l3481_348157


namespace NUMINAMATH_CALUDE_budget_allocation_l3481_348181

theorem budget_allocation (microphotonics home_electronics food_additives industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ) :
  microphotonics = 14 ∧
  home_electronics = 19 ∧
  food_additives = 10 ∧
  industrial_lubricants = 8 ∧
  basic_astrophysics_degrees = 90 →
  ∃ (genetically_modified_microorganisms : ℝ),
    genetically_modified_microorganisms = 24 ∧
    microphotonics + home_electronics + food_additives + industrial_lubricants +
    (basic_astrophysics_degrees / 360 * 100) + genetically_modified_microorganisms = 100 :=
by sorry

end NUMINAMATH_CALUDE_budget_allocation_l3481_348181


namespace NUMINAMATH_CALUDE_unique_prime_cube_sum_squares_l3481_348188

theorem unique_prime_cube_sum_squares :
  ∀ p q r : ℕ,
    Prime p → Prime q → Prime r →
    p^3 = p^2 + q^2 + r^2 →
    p = 3 ∧ q = 3 ∧ r = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_cube_sum_squares_l3481_348188


namespace NUMINAMATH_CALUDE_four_numbers_average_l3481_348177

theorem four_numbers_average (a b c d : ℕ) : 
  a < b ∧ b < c ∧ c < d →  -- Four different positive integers
  a = 3 →                  -- Smallest number is 3
  (a + b + c + d) / 4 = 6 →  -- Average is 6
  d - a = 9 →              -- Difference between largest and smallest is maximized
  (b + c) / 2 = (9 : ℚ) / 2 := by  -- Average of middle two numbers is 4.5
sorry

end NUMINAMATH_CALUDE_four_numbers_average_l3481_348177


namespace NUMINAMATH_CALUDE_xiao_ming_running_time_l3481_348113

theorem xiao_ming_running_time (track_length : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : track_length = 360)
  (h2 : speed1 = 5)
  (h3 : speed2 = 4) :
  let avg_speed := (speed1 + speed2) / 2
  let total_time := track_length / avg_speed
  let half_distance := track_length / 2
  let second_half_time := half_distance / speed2
  second_half_time = 44 := by
sorry

end NUMINAMATH_CALUDE_xiao_ming_running_time_l3481_348113


namespace NUMINAMATH_CALUDE_sum_first_11_even_numbers_eq_132_l3481_348165

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * (i + 1))

def sum_first_n_even_numbers (n : ℕ) : ℕ :=
  (first_n_even_numbers n).sum

theorem sum_first_11_even_numbers_eq_132 : sum_first_n_even_numbers 11 = 132 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_11_even_numbers_eq_132_l3481_348165


namespace NUMINAMATH_CALUDE_douglas_fir_count_l3481_348176

/-- The number of Douglas fir trees in a forest -/
def douglas_fir : ℕ := sorry

/-- The number of ponderosa pine trees in a forest -/
def ponderosa_pine : ℕ := sorry

/-- The total number of trees in the forest -/
def total_trees : ℕ := 850

/-- The cost of a single Douglas fir tree -/
def douglas_fir_cost : ℕ := 300

/-- The cost of a single ponderosa pine tree -/
def ponderosa_pine_cost : ℕ := 225

/-- The total amount paid for all trees -/
def total_cost : ℕ := 217500

theorem douglas_fir_count : 
  douglas_fir = 350 ∧
  douglas_fir + ponderosa_pine = total_trees ∧
  douglas_fir * douglas_fir_cost + ponderosa_pine * ponderosa_pine_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_douglas_fir_count_l3481_348176


namespace NUMINAMATH_CALUDE_westward_movement_l3481_348189

-- Define a type for directions
inductive Direction
  | East
  | West

-- Define a function to represent movement
def represent_movement (dist : ℝ) (dir : Direction) : ℝ :=
  match dir with
  | Direction.East => dist
  | Direction.West => -dist

-- State the theorem
theorem westward_movement :
  (Direction.East ≠ Direction.West) →  -- East and west are opposite
  (represent_movement 2 Direction.East = 2) →  -- +2 meters represents 2 meters eastward
  (represent_movement 7 Direction.West = -7)  -- 7 meters westward is represented by -7 meters
:= by sorry

end NUMINAMATH_CALUDE_westward_movement_l3481_348189


namespace NUMINAMATH_CALUDE_sally_quarters_l3481_348180

def quarters_problem (initial received spent : ℕ) : ℕ :=
  initial + received - spent

theorem sally_quarters : quarters_problem 760 418 152 = 1026 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_l3481_348180


namespace NUMINAMATH_CALUDE_sum_remainder_by_eight_l3481_348191

theorem sum_remainder_by_eight (n : ℤ) : (9 - n + (n + 5)) % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_by_eight_l3481_348191


namespace NUMINAMATH_CALUDE_intersection_point_sum_l3481_348175

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (4, 4)
def D : ℝ × ℝ := (5, 2)

-- Define the quadrilateral
def ABCD : Set (ℝ × ℝ) := {A, B, C, D}

-- Define a function to calculate the area of a quadrilateral
def quadrilateralArea (q : Set (ℝ × ℝ)) : ℝ := sorry

-- Define a function to check if a point is on line CD
def onLineCD (p : ℝ × ℝ) : Prop := sorry

-- Define a function to check if a line through A and a point divides ABCD into equal areas
def dividesEqually (p : ℝ × ℝ) : Prop := sorry

-- Define a function to check if a fraction is in lowest terms
def lowestTerms (p q : ℤ) : Prop := sorry

theorem intersection_point_sum :
  ∀ p q r s : ℤ,
  onLineCD (p/q, r/s) →
  dividesEqually (p/q, r/s) →
  lowestTerms p q →
  lowestTerms r s →
  p + q + r + s = 60 := by sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l3481_348175


namespace NUMINAMATH_CALUDE_square_difference_divisible_by_13_l3481_348166

theorem square_difference_divisible_by_13 (a b : ℕ) 
  (h1 : 1 ≤ a ∧ a ≤ 1000) 
  (h2 : 1 ≤ b ∧ b ≤ 1000) 
  (h3 : a + b = 1001) : 
  13 ∣ (a^2 - b^2) :=
sorry

end NUMINAMATH_CALUDE_square_difference_divisible_by_13_l3481_348166


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3481_348116

def A : Set ℤ := {1, 3}

def B : Set ℤ := {x | 0 < Real.log (x + 1) ∧ Real.log (x + 1) < 1/2}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3481_348116


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l3481_348141

theorem necessary_not_sufficient (a b : ℝ) :
  (∀ x y : ℝ, x > y + 1 → x > y) ∧
  (∃ x y : ℝ, x > y ∧ ¬(x > y + 1)) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l3481_348141


namespace NUMINAMATH_CALUDE_distance_A_to_B_l3481_348117

/-- Prove that the distance from A to B is 510 km given the travel conditions -/
theorem distance_A_to_B : 
  ∀ (d_AB : ℝ) (d_AC : ℝ) (t_E t_F : ℝ) (speed_ratio : ℝ),
  d_AC = 300 →
  t_E = 3 →
  t_F = 4 →
  speed_ratio = 2.2666666666666666 →
  (d_AB / t_E) / (d_AC / t_F) = speed_ratio →
  d_AB = 510 := by
sorry

end NUMINAMATH_CALUDE_distance_A_to_B_l3481_348117


namespace NUMINAMATH_CALUDE_bus_travel_fraction_l3481_348129

/-- Proves that given a total distance of 24 kilometers, where half is traveled by foot
    and 6 kilometers by car, the fraction of the distance traveled by bus is 1/4. -/
theorem bus_travel_fraction (total_distance : ℝ) (foot_distance : ℝ) (car_distance : ℝ) :
  total_distance = 24 →
  foot_distance = total_distance / 2 →
  car_distance = 6 →
  (total_distance - (foot_distance + car_distance)) / total_distance = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_bus_travel_fraction_l3481_348129


namespace NUMINAMATH_CALUDE_common_chord_length_is_2_sqrt_5_l3481_348194

/-- Two circles in a 2D plane -/
structure TwoCircles where
  /-- Equation of the first circle: x^2 + y^2 - 2x + 10y - 24 = 0 -/
  circle1 : ℝ → ℝ → Prop
  /-- Equation of the second circle: x^2 + y^2 + 2x + 2y - 8 = 0 -/
  circle2 : ℝ → ℝ → Prop
  /-- The circles intersect -/
  intersect : ∃ x y, circle1 x y ∧ circle2 x y

/-- Definition of the specific two circles from the problem -/
def specificCircles : TwoCircles where
  circle1 := fun x y => x^2 + y^2 - 2*x + 10*y - 24 = 0
  circle2 := fun x y => x^2 + y^2 + 2*x + 2*y - 8 = 0
  intersect := sorry -- We assume the circles intersect as given in the problem

/-- The length of the common chord of two intersecting circles -/
def commonChordLength (c : TwoCircles) : ℝ := sorry

/-- Theorem stating that the length of the common chord is 2√5 -/
theorem common_chord_length_is_2_sqrt_5 :
  commonChordLength specificCircles = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_is_2_sqrt_5_l3481_348194


namespace NUMINAMATH_CALUDE_tax_rate_change_income_l3481_348143

/-- Proves that given the conditions of the tax rate change and differential savings, 
    the taxpayer's annual income before tax is $45,000 -/
theorem tax_rate_change_income (I : ℝ) 
  (h1 : 0.40 * I - 0.33 * I = 3150) : I = 45000 := by
  sorry

end NUMINAMATH_CALUDE_tax_rate_change_income_l3481_348143


namespace NUMINAMATH_CALUDE_distinct_subsets_removal_l3481_348122

theorem distinct_subsets_removal (n : ℕ) (X : Finset ℕ) (A : Fin n → Finset ℕ) 
  (h1 : n ≥ 2) 
  (h2 : X.card = n) 
  (h3 : ∀ i : Fin n, A i ⊆ X) 
  (h4 : ∀ i j : Fin n, i ≠ j → A i ≠ A j) :
  ∃ x ∈ X, ∀ i j : Fin n, i ≠ j → A i \ {x} ≠ A j \ {x} := by
  sorry

end NUMINAMATH_CALUDE_distinct_subsets_removal_l3481_348122


namespace NUMINAMATH_CALUDE_delta_value_l3481_348186

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ + 3 → Δ = -15 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l3481_348186


namespace NUMINAMATH_CALUDE_remainder_7835_mod_11_l3481_348123

theorem remainder_7835_mod_11 : 7835 % 11 = (7 + 8 + 3 + 5) % 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7835_mod_11_l3481_348123


namespace NUMINAMATH_CALUDE_chicken_eggs_per_chicken_l3481_348136

theorem chicken_eggs_per_chicken 
  (num_chickens : ℕ) 
  (num_cartons : ℕ) 
  (eggs_per_carton : ℕ) 
  (h1 : num_chickens = 20)
  (h2 : num_cartons = 10)
  (h3 : eggs_per_carton = 12) :
  (num_cartons * eggs_per_carton) / num_chickens = 6 :=
by sorry

end NUMINAMATH_CALUDE_chicken_eggs_per_chicken_l3481_348136


namespace NUMINAMATH_CALUDE_abs_neg_two_equals_two_l3481_348132

theorem abs_neg_two_equals_two :
  abs (-2) = 2 := by
sorry

end NUMINAMATH_CALUDE_abs_neg_two_equals_two_l3481_348132


namespace NUMINAMATH_CALUDE_joan_seashells_l3481_348137

/-- Calculates the number of seashells Joan has after giving some away -/
def remaining_seashells (found : ℕ) (given_away : ℕ) : ℕ :=
  found - given_away

/-- Proves that Joan has 16 seashells after finding 79 and giving away 63 -/
theorem joan_seashells : remaining_seashells 79 63 = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l3481_348137


namespace NUMINAMATH_CALUDE_frank_final_position_l3481_348156

def dance_sequence (start : Int) : Int :=
  let step1 := start - 5
  let step2 := step1 + 10
  let step3 := step2 - 2
  let step4 := step3 + (2 * 2)
  step4

theorem frank_final_position :
  dance_sequence 0 = 7 := by
  sorry

end NUMINAMATH_CALUDE_frank_final_position_l3481_348156


namespace NUMINAMATH_CALUDE_triangle_longest_side_l3481_348170

theorem triangle_longest_side (x : ℕ) (h1 : x > 0) 
  (h2 : 5 * x + 6 * x + 7 * x = 720) 
  (h3 : 5 * x + 6 * x > 7 * x) 
  (h4 : 5 * x + 7 * x > 6 * x) 
  (h5 : 6 * x + 7 * x > 5 * x) :
  7 * x = 280 := by
  sorry

#check triangle_longest_side

end NUMINAMATH_CALUDE_triangle_longest_side_l3481_348170


namespace NUMINAMATH_CALUDE_average_speed_problem_l3481_348174

theorem average_speed_problem (D : ℝ) (S : ℝ) (h1 : D > 0) :
  (0.4 * D / S + 0.6 * D / 60) / D = 1 / 50 →
  S = 40 := by
sorry

end NUMINAMATH_CALUDE_average_speed_problem_l3481_348174


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l3481_348184

theorem arctan_equation_solution (y : ℝ) :
  2 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/3 →
  y = 1005/97 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l3481_348184


namespace NUMINAMATH_CALUDE_jane_rounds_played_l3481_348121

-- Define the parameters of the game
def points_per_round : ℕ := 10
def final_points : ℕ := 60
def lost_points : ℕ := 20

-- Define the theorem
theorem jane_rounds_played :
  (final_points + lost_points) / points_per_round = 8 :=
by sorry

end NUMINAMATH_CALUDE_jane_rounds_played_l3481_348121


namespace NUMINAMATH_CALUDE_incorrect_permutations_hello_l3481_348120

def word_length : ℕ := 5
def repeated_letter_count : ℕ := 2

theorem incorrect_permutations_hello :
  (word_length.factorial / repeated_letter_count.factorial) - 1 = 119 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_permutations_hello_l3481_348120


namespace NUMINAMATH_CALUDE_quadratic_extrema_l3481_348168

-- Define the quadratic equation
def quadratic_equation (a b x : ℝ) : Prop :=
  x^2 - (a^2 + b^2 - 6*b)*x + a^2 + b^2 + 2*a - 4*b + 1 = 0

-- Define the condition for the roots
def root_condition (x₁ x₂ : ℝ) : Prop :=
  x₁ ≤ 0 ∧ 0 ≤ x₂ ∧ x₂ ≤ 1

-- Theorem statement
theorem quadratic_extrema (a b x₁ x₂ : ℝ) 
  (h₁ : quadratic_equation a b x₁)
  (h₂ : quadratic_equation a b x₂)
  (h₃ : root_condition x₁ x₂) :
  (∃ (a b : ℝ), a^2 + b^2 + 4*a = -7/2) ∧ 
  (∃ (a b : ℝ), a^2 + b^2 + 4*a = 5 + 4*Real.sqrt 5) ∧
  (∀ (a b : ℝ), -7/2 ≤ a^2 + b^2 + 4*a ∧ a^2 + b^2 + 4*a ≤ 5 + 4*Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_quadratic_extrema_l3481_348168


namespace NUMINAMATH_CALUDE_percent_of_number_hundred_fifty_percent_of_eighty_l3481_348150

theorem percent_of_number (percent : ℝ) (number : ℝ) : 
  (percent / 100) * number = (percent * number) / 100 := by sorry

theorem hundred_fifty_percent_of_eighty : 
  (150 : ℝ) / 100 * 80 = 120 := by sorry

end NUMINAMATH_CALUDE_percent_of_number_hundred_fifty_percent_of_eighty_l3481_348150


namespace NUMINAMATH_CALUDE_david_twice_rosy_age_l3481_348127

/-- The number of years it will take for David to be twice as old as Rosy -/
def years_until_twice_age : ℕ :=
  sorry

/-- David's current age -/
def david_age : ℕ :=
  sorry

/-- Rosy's current age -/
def rosy_age : ℕ :=
  12

theorem david_twice_rosy_age :
  (david_age = rosy_age + 18) →
  (david_age + years_until_twice_age = 2 * (rosy_age + years_until_twice_age)) →
  years_until_twice_age = 6 :=
by sorry

end NUMINAMATH_CALUDE_david_twice_rosy_age_l3481_348127


namespace NUMINAMATH_CALUDE_line_segment_can_have_specific_length_l3481_348115

/-- A line segment is a geometric object with a measurable, finite length. -/
structure LineSegment where
  length : ℝ
  length_positive : length > 0

/-- Theorem: A line segment can have a specific, finite length (e.g., 0.7 meters). -/
theorem line_segment_can_have_specific_length : ∃ (s : LineSegment), s.length = 0.7 :=
sorry

end NUMINAMATH_CALUDE_line_segment_can_have_specific_length_l3481_348115


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3481_348151

-- Problem 1
theorem problem_1 : -8 - 6 + 24 = 10 := by sorry

-- Problem 2
theorem problem_2 : (-48) / 6 + (-21) * (-1/3) = -1 := by sorry

-- Problem 3
theorem problem_3 : (1/8 - 1/3 + 1/4) * (-24) = -1 := by sorry

-- Problem 4
theorem problem_4 : -1^4 - (1 + 0.5) * (1/3) * (1 - (-2)^2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3481_348151


namespace NUMINAMATH_CALUDE_square_equation_solution_l3481_348196

theorem square_equation_solution (a : ℝ) (h : a^2 + a^2/4 = 5) : a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3481_348196


namespace NUMINAMATH_CALUDE_inequality_proof_l3481_348183

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : 1/x + 1/y + 1/z = 2) :
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3481_348183


namespace NUMINAMATH_CALUDE_inequality_range_l3481_348131

theorem inequality_range (b : ℝ) : 
  (∃ x : ℝ, |x - 2| + |x - 5| < b) ↔ b > 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l3481_348131


namespace NUMINAMATH_CALUDE_infinite_triples_existence_l3481_348144

theorem infinite_triples_existence :
  ∀ m : ℕ, ∃ p : ℕ, ∃ q₁ q₂ : ℤ,
    p > m ∧ 
    |Real.sqrt 2 - (q₁ : ℝ) / p| * |Real.sqrt 3 - (q₂ : ℝ) / p| ≤ 1 / (2 * (p : ℝ) ^ 3) :=
by sorry

end NUMINAMATH_CALUDE_infinite_triples_existence_l3481_348144


namespace NUMINAMATH_CALUDE_three_equidistant_lines_l3481_348108

/-- A point in a plane represented by its coordinates -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in a plane represented by its equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if three points are not collinear -/
def nonCollinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) ≠ (p3.x - p1.x) * (p2.y - p1.y)

/-- Returns true if a line is equidistant from three points -/
def equidistantLine (l : Line2D) (p1 p2 p3 : Point2D) : Prop :=
  let d1 := |l.a * p1.x + l.b * p1.y + l.c| / Real.sqrt (l.a^2 + l.b^2)
  let d2 := |l.a * p2.x + l.b * p2.y + l.c| / Real.sqrt (l.a^2 + l.b^2)
  let d3 := |l.a * p3.x + l.b * p3.y + l.c| / Real.sqrt (l.a^2 + l.b^2)
  d1 = d2 ∧ d2 = d3

/-- Main theorem: There are exactly three lines equidistant from three non-collinear points -/
theorem three_equidistant_lines (p1 p2 p3 : Point2D) 
  (h : nonCollinear p1 p2 p3) : 
  ∃! (s : Finset Line2D), s.card = 3 ∧ ∀ l ∈ s, equidistantLine l p1 p2 p3 :=
sorry

end NUMINAMATH_CALUDE_three_equidistant_lines_l3481_348108


namespace NUMINAMATH_CALUDE_sum_of_xyz_l3481_348185

theorem sum_of_xyz (x y z : ℕ+) 
  (h1 : (x * y * z : ℕ) = 240)
  (h2 : (x * y + z : ℕ) = 46)
  (h3 : (x + y * z : ℕ) = 64) :
  (x + y + z : ℕ) = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l3481_348185


namespace NUMINAMATH_CALUDE_ladder_problem_l3481_348109

theorem ladder_problem (ladder_length base_distance height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : base_distance = 5)
  (h3 : ladder_length ^ 2 = base_distance ^ 2 + height ^ 2) :
  height = 12 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l3481_348109


namespace NUMINAMATH_CALUDE_factorization_equality_l3481_348128

theorem factorization_equality (m n : ℝ) : 2 * m * n^2 - 4 * m * n + 2 * m = 2 * m * (n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3481_348128


namespace NUMINAMATH_CALUDE_ping_pong_balls_count_l3481_348192

/-- The number of ping-pong balls bought with tax -/
def B : ℕ := 60

/-- The sales tax rate -/
def tax_rate : ℚ := 16 / 100

theorem ping_pong_balls_count :
  (B : ℚ) * (1 + tax_rate) = (B + 3 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_ping_pong_balls_count_l3481_348192


namespace NUMINAMATH_CALUDE_tan_theta_value_l3481_348112

theorem tan_theta_value (θ : ℝ) :
  let z : ℂ := Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5)
  (z.re = 0 ∧ z.im ≠ 0) → Real.tan θ = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l3481_348112


namespace NUMINAMATH_CALUDE_river_road_cars_l3481_348148

theorem river_road_cars (buses cars : ℕ) : 
  (buses : ℚ) / cars = 1 / 3 →  -- ratio of buses to cars is 1:3
  cars = buses + 40 →           -- 40 fewer buses than cars
  cars = 60 :=                  -- prove that the number of cars is 60
by
  sorry

end NUMINAMATH_CALUDE_river_road_cars_l3481_348148


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l3481_348153

theorem smallest_square_containing_circle (r : ℝ) (h : r = 6) : 
  (2 * r) ^ 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l3481_348153


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3481_348130

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x + 1) / (x - 2)) ↔ x ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3481_348130


namespace NUMINAMATH_CALUDE_product_equals_negative_six_l3481_348105

/-- Given eight real numbers satisfying certain conditions, prove that their product equals -6 -/
theorem product_equals_negative_six
  (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ)
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_negative_six_l3481_348105


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3481_348119

/-- Given a right triangle with sides a, b, and hypotenuse c, and a point M(m, n) on the line ax+by+3c=0,
    the minimum value of m^2+n^2 is 9. -/
theorem min_distance_to_line (a b c : ℝ) (m n : ℝ → ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  (∀ t, a * (m t) + b * (n t) + 3 * c = 0) →
  (∃ t₀, ∀ t, (m t)^2 + (n t)^2 ≥ (m t₀)^2 + (n t₀)^2) →
  ∃ t₀, (m t₀)^2 + (n t₀)^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3481_348119


namespace NUMINAMATH_CALUDE_solution_is_three_l3481_348110

/-- A linear function passing through (-2, 0) with y-intercept 3 -/
structure LinearFunction where
  k : ℝ
  k_nonzero : k ≠ 0
  passes_through : k * (-2) + 3 = 0

/-- The solution to k(x-5)+3=0 is x=3 -/
theorem solution_is_three (f : LinearFunction) : 
  ∃ x : ℝ, f.k * (x - 5) + 3 = 0 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_three_l3481_348110


namespace NUMINAMATH_CALUDE_sample_size_is_sixty_verify_conditions_l3481_348182

/-- Represents the total number of students in the population -/
def total_students : ℕ := 600

/-- Represents the number of male students in the population -/
def male_students : ℕ := 310

/-- Represents the number of female students in the population -/
def female_students : ℕ := 290

/-- Represents the number of male students in the sample -/
def sample_males : ℕ := 31

/-- Calculates the sample size based on stratified random sampling by gender -/
def calculate_sample_size (total : ℕ) (males : ℕ) (sample_males : ℕ) : ℕ :=
  (sample_males * total) / males

/-- Theorem stating that the calculated sample size is 60 -/
theorem sample_size_is_sixty :
  calculate_sample_size total_students male_students sample_males = 60 := by
  sorry

/-- Theorem verifying the given conditions -/
theorem verify_conditions :
  total_students = male_students + female_students ∧
  male_students = 310 ∧
  female_students = 290 ∧
  sample_males = 31 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_sixty_verify_conditions_l3481_348182


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l3481_348160

theorem smallest_integer_solution : ∃ x : ℤ, 
  (∀ y : ℤ, 10 * y^2 - 40 * y + 36 = 0 → x ≤ y) ∧ 
  (10 * x^2 - 40 * x + 36 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l3481_348160


namespace NUMINAMATH_CALUDE_third_even_integer_l3481_348178

/-- Given four consecutive even integers where the sum of the second and fourth is 156,
    prove that the third integer is 78. -/
theorem third_even_integer (n : ℤ) : 
  (n + 2) + (n + 6) = 156 → n + 4 = 78 := by
  sorry

end NUMINAMATH_CALUDE_third_even_integer_l3481_348178


namespace NUMINAMATH_CALUDE_least_value_theorem_l3481_348171

theorem least_value_theorem (x y z w : ℕ+) 
  (h : (5 : ℕ) * w.val = (3 : ℕ) * x.val ∧ 
       (3 : ℕ) * x.val = (4 : ℕ) * y.val ∧ 
       (4 : ℕ) * y.val = (7 : ℕ) * z.val) : 
  (∀ a b c d : ℕ+, 
    ((5 : ℕ) * d.val = (3 : ℕ) * a.val ∧ 
     (3 : ℕ) * a.val = (4 : ℕ) * b.val ∧ 
     (4 : ℕ) * b.val = (7 : ℕ) * c.val) → 
    (x.val - y.val + z.val - w.val : ℤ) ≤ (a.val - b.val + c.val - d.val : ℤ)) ∧
  (x.val - y.val + z.val - w.val : ℤ) = 11 := by
sorry

end NUMINAMATH_CALUDE_least_value_theorem_l3481_348171


namespace NUMINAMATH_CALUDE_relay_arrangements_verify_arrangements_l3481_348140

def total_athletes : ℕ := 8
def relay_positions : ℕ := 4

def arrangements_condition1 : ℕ := 60
def arrangements_condition2 : ℕ := 480
def arrangements_condition3 : ℕ := 180

/-- Theorem stating the number of arrangements for each condition -/
theorem relay_arrangements :
  (arrangements_condition1 = 60) ∧
  (arrangements_condition2 = 480) ∧
  (arrangements_condition3 = 180) := by
  sorry

/-- Function to calculate the number of arrangements for condition 1 -/
def calc_arrangements_condition1 : ℕ :=
  2 * 1 * 6 * 5

/-- Function to calculate the number of arrangements for condition 2 -/
def calc_arrangements_condition2 : ℕ :=
  2 * 2 * 6 * 5 * 4

/-- Function to calculate the number of arrangements for condition 3 -/
def calc_arrangements_condition3 : ℕ :=
  2 * 1 * (6 * 5 / (2 * 1)) * 3 * 2 * 1

/-- Theorem proving that the calculated arrangements match the given ones -/
theorem verify_arrangements :
  (calc_arrangements_condition1 = arrangements_condition1) ∧
  (calc_arrangements_condition2 = arrangements_condition2) ∧
  (calc_arrangements_condition3 = arrangements_condition3) := by
  sorry

end NUMINAMATH_CALUDE_relay_arrangements_verify_arrangements_l3481_348140


namespace NUMINAMATH_CALUDE_union_M_complement_N_equals_real_l3481_348134

open Set

def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem union_M_complement_N_equals_real : M ∪ Nᶜ = univ :=
sorry

end NUMINAMATH_CALUDE_union_M_complement_N_equals_real_l3481_348134


namespace NUMINAMATH_CALUDE_stool_height_correct_l3481_348139

/-- The height of the ceiling in meters -/
def ceiling_height : ℝ := 2.4

/-- The height of Alice in meters -/
def alice_height : ℝ := 1.5

/-- The additional reach of Alice above her head in meters -/
def alice_reach : ℝ := 0.5

/-- The distance of the light bulb from the ceiling in meters -/
def bulb_distance : ℝ := 0.2

/-- The height of the stool in meters -/
def stool_height : ℝ := 0.2

theorem stool_height_correct : 
  alice_height + alice_reach + stool_height = ceiling_height - bulb_distance := by
  sorry

end NUMINAMATH_CALUDE_stool_height_correct_l3481_348139


namespace NUMINAMATH_CALUDE_same_temperature_exists_l3481_348173

/-- Conversion function from Celsius to Fahrenheit -/
def celsius_to_fahrenheit (c : ℝ) : ℝ := 1.8 * c + 32

/-- Theorem stating that there exists a temperature that is the same in both Celsius and Fahrenheit scales -/
theorem same_temperature_exists : ∃ t : ℝ, t = celsius_to_fahrenheit t := by
  sorry

end NUMINAMATH_CALUDE_same_temperature_exists_l3481_348173


namespace NUMINAMATH_CALUDE_maggies_earnings_l3481_348111

/-- Calculates the total earnings for Maggie's magazine subscription sales --/
theorem maggies_earnings 
  (family_commission : ℕ) 
  (neighbor_commission : ℕ)
  (bonus_threshold : ℕ)
  (bonus_base : ℕ)
  (bonus_per_extra : ℕ)
  (family_subscriptions : ℕ)
  (neighbor_subscriptions : ℕ)
  (h1 : family_commission = 7)
  (h2 : neighbor_commission = 6)
  (h3 : bonus_threshold = 10)
  (h4 : bonus_base = 10)
  (h5 : bonus_per_extra = 1)
  (h6 : family_subscriptions = 9)
  (h7 : neighbor_subscriptions = 6) :
  family_commission * family_subscriptions +
  neighbor_commission * neighbor_subscriptions +
  bonus_base +
  (if family_subscriptions + neighbor_subscriptions > bonus_threshold
   then (family_subscriptions + neighbor_subscriptions - bonus_threshold) * bonus_per_extra
   else 0) = 114 := by
  sorry


end NUMINAMATH_CALUDE_maggies_earnings_l3481_348111


namespace NUMINAMATH_CALUDE_floor_neg_seven_thirds_l3481_348198

theorem floor_neg_seven_thirds : ⌊(-7 : ℝ) / 3⌋ = -3 := by sorry

end NUMINAMATH_CALUDE_floor_neg_seven_thirds_l3481_348198


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3481_348102

-- Define the inequality
def inequality (x : ℝ) : Prop := |x^2 - 5*x + 6| < x^2 - 4

-- Define the solution set
def solution_set : Set ℝ := {x | x > 2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3481_348102


namespace NUMINAMATH_CALUDE_oliver_used_30_tickets_l3481_348124

/-- The number of tickets Oliver used at the town carnival -/
def olivers_tickets (ferris_wheel_rides bumper_car_rides tickets_per_ride : ℕ) : ℕ :=
  (ferris_wheel_rides + bumper_car_rides) * tickets_per_ride

/-- Theorem: Oliver used 30 tickets at the town carnival -/
theorem oliver_used_30_tickets :
  olivers_tickets 7 3 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_oliver_used_30_tickets_l3481_348124


namespace NUMINAMATH_CALUDE_school_capacity_l3481_348133

/-- The total number of students that can be taught at a time by four primary schools -/
def total_students (capacity1 capacity2 : ℕ) : ℕ :=
  2 * capacity1 + 2 * capacity2

/-- Theorem stating that the total number of students is 1480 -/
theorem school_capacity : total_students 400 340 = 1480 := by
  sorry

end NUMINAMATH_CALUDE_school_capacity_l3481_348133


namespace NUMINAMATH_CALUDE_negative_decimal_greater_than_negative_fraction_l3481_348164

theorem negative_decimal_greater_than_negative_fraction : -0.6 > -(2/3) := by
  sorry

end NUMINAMATH_CALUDE_negative_decimal_greater_than_negative_fraction_l3481_348164


namespace NUMINAMATH_CALUDE_initial_garrison_size_l3481_348152

/-- 
Given a garrison with provisions for a certain number of days, and information about
reinforcements and remaining provisions, this theorem proves the initial number of men.
-/
theorem initial_garrison_size 
  (initial_provision_days : ℕ) 
  (days_before_reinforcement : ℕ) 
  (reinforcement_size : ℕ) 
  (remaining_provision_days : ℕ) 
  (h1 : initial_provision_days = 54)
  (h2 : days_before_reinforcement = 15)
  (h3 : reinforcement_size = 600)
  (h4 : remaining_provision_days = 30)
  : ∃ (initial_men : ℕ), 
    initial_men * (initial_provision_days - days_before_reinforcement) = 
    (initial_men + reinforcement_size) * remaining_provision_days ∧ 
    initial_men = 2000 :=
by sorry

end NUMINAMATH_CALUDE_initial_garrison_size_l3481_348152


namespace NUMINAMATH_CALUDE_not_prime_n_pow_n_minus_4n_plus_3_l3481_348158

theorem not_prime_n_pow_n_minus_4n_plus_3 (n : ℕ) : ¬ Nat.Prime (n^n - 4*n + 3) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_n_pow_n_minus_4n_plus_3_l3481_348158


namespace NUMINAMATH_CALUDE_solution_set_problem1_solution_set_problem2_l3481_348187

-- Problem 1
theorem solution_set_problem1 :
  {x : ℝ | x * (x + 2) > x * (3 - x) + 1} = {x : ℝ | x < -1/2 ∨ x > 1} := by sorry

-- Problem 2
theorem solution_set_problem2 (a : ℝ) :
  {x : ℝ | x^2 - 2*a*x - 8*a^2 ≤ 0} = 
    if a > 0 then
      {x : ℝ | -2*a ≤ x ∧ x ≤ 4*a}
    else if a = 0 then
      {0}
    else
      {x : ℝ | 4*a ≤ x ∧ x ≤ -2*a} := by sorry

end NUMINAMATH_CALUDE_solution_set_problem1_solution_set_problem2_l3481_348187


namespace NUMINAMATH_CALUDE_power_of_two_expression_l3481_348147

theorem power_of_two_expression : (2^2)^(2^(2^2)) = 4294967296 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_expression_l3481_348147


namespace NUMINAMATH_CALUDE_emir_book_purchase_l3481_348163

/-- The cost of the dictionary in dollars -/
def dictionary_cost : ℕ := 5

/-- The cost of the dinosaur book in dollars -/
def dinosaur_book_cost : ℕ := 11

/-- The cost of the children's cookbook in dollars -/
def cookbook_cost : ℕ := 5

/-- The amount Emir has saved in dollars -/
def saved_amount : ℕ := 19

/-- The additional money Emir needs to buy all three books -/
def additional_money_needed : ℕ := 2

theorem emir_book_purchase :
  dictionary_cost + dinosaur_book_cost + cookbook_cost - saved_amount = additional_money_needed :=
by sorry

end NUMINAMATH_CALUDE_emir_book_purchase_l3481_348163


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_one_l3481_348125

theorem sqrt_expression_equals_one :
  (Real.sqrt 24 - Real.sqrt 216) / Real.sqrt 6 + 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_one_l3481_348125


namespace NUMINAMATH_CALUDE_problem_solution_l3481_348154

theorem problem_solution : 
  (-(3^2) / 3 + |(-7)| + 3 * (-1/3) = 3) ∧
  ((-1)^2022 - (-1/4 - (-1/3)) / (-1/12) = 2) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3481_348154


namespace NUMINAMATH_CALUDE_third_group_first_student_l3481_348101

/-- Systematic sampling function that returns the number of the first student in a given group -/
def systematic_sample (total_students : ℕ) (sample_size : ℕ) (group : ℕ) : ℕ :=
  let interval := total_students / sample_size
  (group - 1) * interval

theorem third_group_first_student :
  systematic_sample 800 40 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_third_group_first_student_l3481_348101


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3481_348145

theorem quadratic_coefficient (a b c y₁ y₂ : ℝ) : 
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = -16 →
  b = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3481_348145


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3481_348106

theorem polynomial_simplification (y : ℝ) :
  (2 * y^6 + 3 * y^5 + y^3 + 15) - (y^6 + 4 * y^5 - 2 * y^4 + 17) =
  y^6 - y^5 + 2 * y^4 + y^3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3481_348106


namespace NUMINAMATH_CALUDE_honda_cars_count_l3481_348107

-- Define the total number of cars
def total_cars : ℕ := 9000

-- Define the percentage of red Honda cars
def red_honda_percentage : ℚ := 90 / 100

-- Define the percentage of total red cars
def total_red_percentage : ℚ := 60 / 100

-- Define the percentage of red non-Honda cars
def red_non_honda_percentage : ℚ := 225 / 1000

-- Theorem statement
theorem honda_cars_count (honda_cars : ℕ) :
  (honda_cars : ℚ) * red_honda_percentage + 
  ((total_cars - honda_cars) : ℚ) * red_non_honda_percentage = 
  (total_cars : ℚ) * total_red_percentage →
  honda_cars = 5000 := by
sorry

end NUMINAMATH_CALUDE_honda_cars_count_l3481_348107


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l3481_348149

/-- Given that θ is an angle in the second quadrant, prove that θ/2 lies in the first or third quadrant. -/
theorem half_angle_quadrant (θ : Real) (h : ∃ k : ℤ, 2 * k * π + π / 2 < θ ∧ θ < 2 * k * π + π) :
  ∃ k : ℤ, (k * π < θ / 2 ∧ θ / 2 < k * π + π / 2) ∨ 
           (k * π + π < θ / 2 ∧ θ / 2 < k * π + 3 * π / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l3481_348149


namespace NUMINAMATH_CALUDE_floor_equality_implies_abs_diff_less_than_one_l3481_348142

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_equality_implies_abs_diff_less_than_one (x y : ℝ) :
  floor x = floor y → |x - y| < 1 := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_implies_abs_diff_less_than_one_l3481_348142


namespace NUMINAMATH_CALUDE_two_axisymmetric_additions_l3481_348179

/-- Represents a position on a 4x4 grid --/
structure Position where
  x : Fin 4
  y : Fin 4

/-- Represents a configuration of shaded squares on a 4x4 grid --/
def Configuration := List Position

/-- Checks if a configuration is axisymmetric --/
def isAxisymmetric (config : Configuration) : Bool :=
  sorry

/-- Counts the number of ways to add one square to make the configuration axisymmetric --/
def countAxisymmetricAdditions (initialConfig : Configuration) : Nat :=
  sorry

/-- The initial configuration with 3 shaded squares --/
def initialConfig : Configuration :=
  sorry

theorem two_axisymmetric_additions :
  countAxisymmetricAdditions initialConfig = 2 :=
sorry

end NUMINAMATH_CALUDE_two_axisymmetric_additions_l3481_348179


namespace NUMINAMATH_CALUDE_candidates_count_l3481_348199

theorem candidates_count (x : ℝ) : 
  (x > 0) →  -- number of candidates is positive
  (0.07 * x = 0.06 * x + 80) →  -- State B had 80 more selected candidates
  (x = 8000) := by
sorry

end NUMINAMATH_CALUDE_candidates_count_l3481_348199


namespace NUMINAMATH_CALUDE_product_three_consecutive_divisible_by_six_l3481_348167

theorem product_three_consecutive_divisible_by_six (n : ℕ) : 
  6 ∣ (n * (n + 1) * (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_product_three_consecutive_divisible_by_six_l3481_348167


namespace NUMINAMATH_CALUDE_trip_distance_l3481_348190

/-- The total distance of a trip between three cities forming a right-angled triangle -/
theorem trip_distance (DE EF FD : ℝ) (h1 : DE = 4500) (h2 : FD = 4000) 
  (h3 : DE^2 = EF^2 + FD^2) : DE + EF + FD = 10562 := by
  sorry

end NUMINAMATH_CALUDE_trip_distance_l3481_348190


namespace NUMINAMATH_CALUDE_williams_land_percentage_l3481_348118

/-- Given a village with farm tax and Mr. William's tax payment, calculate the percentage of
    Mr. William's taxable land over the total taxable land of the village. -/
theorem williams_land_percentage 
  (total_tax : ℝ) 
  (williams_tax : ℝ) 
  (h1 : total_tax = 3840) 
  (h2 : williams_tax = 480) :
  williams_tax / total_tax = 0.125 := by
  sorry

#check williams_land_percentage

end NUMINAMATH_CALUDE_williams_land_percentage_l3481_348118


namespace NUMINAMATH_CALUDE_lyft_taxi_cost_difference_l3481_348172

def uber_cost : ℝ := 22
def lyft_cost : ℝ := uber_cost - 3
def taxi_cost_with_tip : ℝ := 18
def tip_percentage : ℝ := 0.2

theorem lyft_taxi_cost_difference : 
  lyft_cost - (taxi_cost_with_tip / (1 + tip_percentage)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_lyft_taxi_cost_difference_l3481_348172


namespace NUMINAMATH_CALUDE_A_subset_B_l3481_348135

def A : Set ℕ := {x | ∃ a : ℕ, a > 0 ∧ x = a^2 + 1}
def B : Set ℕ := {y | ∃ b : ℕ, b > 0 ∧ y = b^2 - 4*b + 5}

theorem A_subset_B : A ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_A_subset_B_l3481_348135


namespace NUMINAMATH_CALUDE_triangle_rational_area_l3481_348103

/-- Triangle with rational side lengths and angle bisectors has rational area -/
theorem triangle_rational_area (a b c fa fb fc : ℚ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  fa > 0 ∧ fb > 0 ∧ fc > 0 →  -- positive angle bisector lengths
  ∃ (area : ℚ), area > 0 ∧ area^2 = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c) / 16 :=
sorry

end NUMINAMATH_CALUDE_triangle_rational_area_l3481_348103


namespace NUMINAMATH_CALUDE_power_function_decreasing_condition_l3481_348114

/-- A function f is a power function if it's of the form f(x) = x^a for some real a -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x > 0, f x = x^a

/-- A function f is decreasing on (0, +∞) if for all x, y in (0, +∞), x < y implies f(x) > f(y) -/
def is_decreasing_on_positive_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x > f y

/-- The main theorem -/
theorem power_function_decreasing_condition (m : ℝ) : 
  (is_power_function (fun x => (m^2 - m - 1) * x^(m^2 - 2*m - 1)) ∧ 
   is_decreasing_on_positive_reals (fun x => (m^2 - m - 1) * x^(m^2 - 2*m - 1))) → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_decreasing_condition_l3481_348114


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3481_348155

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 2 / (x - 5)) ↔ x ≠ 5 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3481_348155


namespace NUMINAMATH_CALUDE_odd_products_fraction_l3481_348197

def multiplication_table_size : ℕ := 11

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd_numbers (n : ℕ) : ℕ := (n + 1) / 2

theorem odd_products_fraction :
  (count_odd_numbers multiplication_table_size)^2 / multiplication_table_size^2 = 25 / 121 := by
  sorry

end NUMINAMATH_CALUDE_odd_products_fraction_l3481_348197


namespace NUMINAMATH_CALUDE_binary_division_theorem_l3481_348126

/-- Convert a binary number (represented as a list of 0s and 1s) to a natural number. -/
def binary_to_nat (binary : List Nat) : Nat :=
  binary.foldr (fun bit acc => 2 * acc + bit) 0

/-- The binary representation of 10101₂ -/
def binary_10101 : List Nat := [1, 0, 1, 0, 1]

/-- The binary representation of 11₂ -/
def binary_11 : List Nat := [1, 1]

/-- The binary representation of 111₂ -/
def binary_111 : List Nat := [1, 1, 1]

/-- Theorem stating that the quotient of 10101₂ divided by 11₂ is equal to 111₂ -/
theorem binary_division_theorem :
  (binary_to_nat binary_10101) / (binary_to_nat binary_11) = binary_to_nat binary_111 := by
  sorry

end NUMINAMATH_CALUDE_binary_division_theorem_l3481_348126


namespace NUMINAMATH_CALUDE_lucky_years_2010_to_2014_l3481_348104

/-- A year is lucky if there exists a date in that year where the product of the month and day
    equals the last two digits of the year. -/
def is_lucky_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), 1 ≤ month ∧ month ≤ 12 ∧ 1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

/-- 2013 is not a lucky year, while 2010, 2011, 2012, and 2014 are lucky years. -/
theorem lucky_years_2010_to_2014 :
  is_lucky_year 2010 ∧ is_lucky_year 2011 ∧ is_lucky_year 2012 ∧
  ¬is_lucky_year 2013 ∧ is_lucky_year 2014 := by
  sorry

#check lucky_years_2010_to_2014

end NUMINAMATH_CALUDE_lucky_years_2010_to_2014_l3481_348104


namespace NUMINAMATH_CALUDE_twenty_point_circle_special_chords_l3481_348138

/-- A circle with equally spaced points on its circumference -/
structure PointedCircle where
  n : ℕ  -- number of points
  (n_pos : n > 0)

/-- Counts chords in a PointedCircle satisfying certain length conditions -/
def count_special_chords (c : PointedCircle) : ℕ :=
  sorry

/-- Theorem statement -/
theorem twenty_point_circle_special_chords :
  ∃ (c : PointedCircle), c.n = 20 ∧ count_special_chords c = 120 :=
sorry

end NUMINAMATH_CALUDE_twenty_point_circle_special_chords_l3481_348138


namespace NUMINAMATH_CALUDE_disjoint_subset_union_equality_l3481_348146

/-- Given n+1 non-empty subsets of {1, 2, ..., n}, there exist two disjoint non-empty subsets
    of {1, 2, ..., n+1} such that the union of A_i for one subset equals the union of A_j
    for the other subset. -/
theorem disjoint_subset_union_equality (n : ℕ) (A : Fin (n + 1) → Set (Fin n)) 
    (h : ∀ i, Set.Nonempty (A i)) :
  ∃ (I J : Set (Fin (n + 1))), 
    I.Nonempty ∧ J.Nonempty ∧ Disjoint I J ∧
    (⋃ i ∈ I, A i) = (⋃ j ∈ J, A j) := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subset_union_equality_l3481_348146


namespace NUMINAMATH_CALUDE_largest_certain_divisor_l3481_348161

-- Define the set of numbers on the die
def dieNumbers : Finset ℕ := Finset.range 8

-- Define the type for products of seven numbers from the die
def ProductOfSeven : Type :=
  {s : Finset ℕ // s ⊆ dieNumbers ∧ s.card = 7}

-- Define the product function
def product (s : ProductOfSeven) : ℕ :=
  s.val.prod id

-- Theorem statement
theorem largest_certain_divisor :
  (∀ s : ProductOfSeven, 192 ∣ product s) ∧
  (∀ n : ℕ, n > 192 → ∃ s : ProductOfSeven, ¬(n ∣ product s)) :=
sorry

end NUMINAMATH_CALUDE_largest_certain_divisor_l3481_348161


namespace NUMINAMATH_CALUDE_intersection_point_equality_l3481_348169

-- Define the functions
def f (x : ℝ) : ℝ := 20 * x^3 + 19 * x^2
def g (x : ℝ) : ℝ := 20 * x^2 + 19 * x
def h (x : ℝ) : ℝ := 20 * x + 19

-- Theorem statement
theorem intersection_point_equality :
  ∀ x : ℝ, g x = h x → f x = g x :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_point_equality_l3481_348169


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l3481_348162

/-- The gain percentage of a trader selling pens -/
def gain_percentage (num_sold : ℕ) (num_gained : ℕ) : ℚ :=
  (num_gained : ℚ) / (num_sold : ℚ) * 100

/-- Theorem: The gain percentage is 25% when selling 80 pens and gaining the cost of 20 pens -/
theorem trader_gain_percentage : gain_percentage 80 20 = 25 := by
  sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l3481_348162


namespace NUMINAMATH_CALUDE_sqrt_sum_implies_product_l3481_348100

theorem sqrt_sum_implies_product (x : ℝ) :
  Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8 →
  (10 + x) * (30 - x) = 144 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_implies_product_l3481_348100
