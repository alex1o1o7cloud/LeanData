import Mathlib

namespace probability_at_least_one_red_l2066_206617

def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def selected_balls : ℕ := 2

theorem probability_at_least_one_red :
  (1 : ℚ) - (Nat.choose white_balls selected_balls : ℚ) / (Nat.choose total_balls selected_balls : ℚ) = 7 / 10 := by
  sorry

end probability_at_least_one_red_l2066_206617


namespace half_angle_quadrant_l2066_206605

/-- Given that θ is an angle in the second quadrant, prove that θ/2 lies in the first or third quadrant. -/
theorem half_angle_quadrant (θ : Real) (h : ∃ k : ℤ, 2 * k * π + π / 2 < θ ∧ θ < 2 * k * π + π) :
  ∃ k : ℤ, (k * π < θ / 2 ∧ θ / 2 < k * π + π / 2) ∨ 
           (k * π + π < θ / 2 ∧ θ / 2 < k * π + 3 * π / 2) :=
by sorry

end half_angle_quadrant_l2066_206605


namespace congruence_modulo_ten_l2066_206675

def a : ℤ := 1 + (Finset.sum (Finset.range 20) (fun k => Nat.choose 20 (k + 1) * 2^k))

theorem congruence_modulo_ten (b : ℤ) (h : b ≡ a [ZMOD 10]) : b = 2011 := by
  sorry

end congruence_modulo_ten_l2066_206675


namespace arctan_equation_solution_l2066_206623

theorem arctan_equation_solution (y : ℝ) :
  2 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/3 →
  y = 1005/97 := by
  sorry

end arctan_equation_solution_l2066_206623


namespace complex_function_chain_l2066_206678

theorem complex_function_chain (x y u : ℂ) : 
  u = 2 * x - 5 → (y = (2 * x - 5)^10 ↔ y = u^10) := by
  sorry

end complex_function_chain_l2066_206678


namespace quadratic_max_l2066_206622

theorem quadratic_max (a b c : ℝ) (x₀ : ℝ) (h1 : a < 0) (h2 : 2 * a * x₀ + b = 0) :
  let f := λ x : ℝ => a * x^2 + b * x + c
  ∀ x : ℝ, f x ≤ f x₀ := by
  sorry

end quadratic_max_l2066_206622


namespace orange_stack_count_l2066_206689

/-- Calculates the number of oranges in a single layer of the pyramid -/
def layerOranges (baseWidth : ℕ) (baseLength : ℕ) (layer : ℕ) : ℕ :=
  (baseWidth - layer + 1) * (baseLength - layer + 1)

/-- Calculates the total number of oranges in the pyramid stack -/
def totalOranges (baseWidth : ℕ) (baseLength : ℕ) : ℕ :=
  let numLayers := min baseWidth baseLength
  (List.range numLayers).foldl (fun acc i => acc + layerOranges baseWidth baseLength i) 0

/-- Theorem stating that a pyramid-like stack of oranges with a 6x9 base contains 154 oranges -/
theorem orange_stack_count : totalOranges 6 9 = 154 := by
  sorry

end orange_stack_count_l2066_206689


namespace quadratic_coefficient_l2066_206607

theorem quadratic_coefficient (a b c y₁ y₂ : ℝ) : 
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = -16 →
  b = -4 := by
sorry

end quadratic_coefficient_l2066_206607


namespace quadratic_two_real_roots_l2066_206693

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 4*x + k = 0 ∧ y^2 + 4*y + k = 0) → k ≤ 4 := by
  sorry

end quadratic_two_real_roots_l2066_206693


namespace sum_of_xyz_l2066_206624

theorem sum_of_xyz (x y z : ℕ+) 
  (h1 : (x * y * z : ℕ) = 240)
  (h2 : (x * y + z : ℕ) = 46)
  (h3 : (x + y * z : ℕ) = 64) :
  (x + y + z : ℕ) = 20 := by
sorry

end sum_of_xyz_l2066_206624


namespace disjoint_subset_union_equality_l2066_206628

/-- Given n+1 non-empty subsets of {1, 2, ..., n}, there exist two disjoint non-empty subsets
    of {1, 2, ..., n+1} such that the union of A_i for one subset equals the union of A_j
    for the other subset. -/
theorem disjoint_subset_union_equality (n : ℕ) (A : Fin (n + 1) → Set (Fin n)) 
    (h : ∀ i, Set.Nonempty (A i)) :
  ∃ (I J : Set (Fin (n + 1))), 
    I.Nonempty ∧ J.Nonempty ∧ Disjoint I J ∧
    (⋃ i ∈ I, A i) = (⋃ j ∈ J, A j) := by
  sorry

end disjoint_subset_union_equality_l2066_206628


namespace percent_of_number_hundred_fifty_percent_of_eighty_l2066_206606

theorem percent_of_number (percent : ℝ) (number : ℝ) : 
  (percent / 100) * number = (percent * number) / 100 := by sorry

theorem hundred_fifty_percent_of_eighty : 
  (150 : ℝ) / 100 * 80 = 120 := by sorry

end percent_of_number_hundred_fifty_percent_of_eighty_l2066_206606


namespace polynomial_property_l2066_206644

-- Define the polynomial Q(x)
def Q (d e f : ℝ) (x : ℝ) : ℝ := x^3 + d*x^2 + e*x + f

-- Define the properties of the polynomial
theorem polynomial_property (d e f : ℝ) :
  -- The y-intercept is 5
  Q d e f 0 = 5 →
  -- The mean of zeros equals the product of zeros
  -d/3 = -f →
  -- The mean of zeros equals the sum of coefficients
  -d/3 = 1 + d + e + f →
  -- Conclusion: e = -26
  e = -26 := by sorry

end polynomial_property_l2066_206644


namespace triangle_existence_l2066_206699

theorem triangle_existence (k : ℕ) (a b c : ℝ) 
  (h_k : k ≥ 10) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : 
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ a = y + z ∧ b = z + x ∧ c = x + y :=
sorry

end triangle_existence_l2066_206699


namespace infinite_triples_existence_l2066_206631

theorem infinite_triples_existence :
  ∀ m : ℕ, ∃ p : ℕ, ∃ q₁ q₂ : ℤ,
    p > m ∧ 
    |Real.sqrt 2 - (q₁ : ℝ) / p| * |Real.sqrt 3 - (q₂ : ℝ) / p| ≤ 1 / (2 * (p : ℝ) ^ 3) :=
by sorry

end infinite_triples_existence_l2066_206631


namespace solution_set_problem1_solution_set_problem2_l2066_206662

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

end solution_set_problem1_solution_set_problem2_l2066_206662


namespace max_value_sum_sqrt_l2066_206651

theorem max_value_sum_sqrt (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 6) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) ≤ Real.sqrt 63 ∧
  (Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) = Real.sqrt 63 ↔ x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

end max_value_sum_sqrt_l2066_206651


namespace regular_scoop_cost_l2066_206645

/-- The cost of ice cream scoops for the Martin family --/
def ice_cream_cost (regular_scoop : ℚ) : Prop :=
  let kiddie_scoop : ℚ := 3
  let double_scoop : ℚ := 6
  let total_cost : ℚ := 32
  let num_regular : ℕ := 2  -- Mr. and Mrs. Martin
  let num_kiddie : ℕ := 2   -- Two children
  let num_double : ℕ := 3   -- Three teenagers
  (num_regular * regular_scoop + 
   num_kiddie * kiddie_scoop + 
   num_double * double_scoop) = total_cost

theorem regular_scoop_cost : 
  ∃ (regular_scoop : ℚ), ice_cream_cost regular_scoop ∧ regular_scoop = 4 :=
by
  sorry

end regular_scoop_cost_l2066_206645


namespace bus_travel_fraction_l2066_206614

/-- Proves that given a total distance of 24 kilometers, where half is traveled by foot
    and 6 kilometers by car, the fraction of the distance traveled by bus is 1/4. -/
theorem bus_travel_fraction (total_distance : ℝ) (foot_distance : ℝ) (car_distance : ℝ) :
  total_distance = 24 →
  foot_distance = total_distance / 2 →
  car_distance = 6 →
  (total_distance - (foot_distance + car_distance)) / total_distance = 1 / 4 := by
  sorry


end bus_travel_fraction_l2066_206614


namespace inequality_range_l2066_206613

theorem inequality_range (b : ℝ) : 
  (∃ x : ℝ, |x - 2| + |x - 5| < b) ↔ b > 3 :=
by sorry

end inequality_range_l2066_206613


namespace different_testing_methods_part1_different_testing_methods_part2_l2066_206690

/-- The number of products -/
def n : ℕ := 10

/-- The number of defective products -/
def d : ℕ := 4

/-- The position of the first defective product in part 1 -/
def first_defective : ℕ := 5

/-- The position of the last defective product in part 1 -/
def last_defective : ℕ := 10

/-- The number of different testing methods in part 1 -/
def methods_part1 : ℕ := 103680

/-- The number of different testing methods in part 2 -/
def methods_part2 : ℕ := 576

/-- Theorem for part 1 -/
theorem different_testing_methods_part1 :
  (n = 10) → (d = 4) → (first_defective = 5) → (last_defective = 10) →
  methods_part1 = 103680 := by sorry

/-- Theorem for part 2 -/
theorem different_testing_methods_part2 :
  (n = 10) → (d = 4) → methods_part2 = 576 := by sorry

end different_testing_methods_part1_different_testing_methods_part2_l2066_206690


namespace complex_equation_sum_l2066_206676

theorem complex_equation_sum (x y : ℝ) :
  (x + (y - 2) * Complex.I = 2 / (1 + Complex.I)) → x + y = 2 := by
  sorry

end complex_equation_sum_l2066_206676


namespace trip_distance_l2066_206632

/-- The total distance of a trip between three cities forming a right-angled triangle -/
theorem trip_distance (DE EF FD : ℝ) (h1 : DE = 4500) (h2 : FD = 4000) 
  (h3 : DE^2 = EF^2 + FD^2) : DE + EF + FD = 10562 := by
  sorry

end trip_distance_l2066_206632


namespace odd_products_fraction_l2066_206669

def multiplication_table_size : ℕ := 11

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd_numbers (n : ℕ) : ℕ := (n + 1) / 2

theorem odd_products_fraction :
  (count_odd_numbers multiplication_table_size)^2 / multiplication_table_size^2 = 25 / 121 := by
  sorry

end odd_products_fraction_l2066_206669


namespace smallest_integer_with_remainder_one_smallest_integer_is_2395_l2066_206637

theorem smallest_integer_with_remainder_one (k : ℕ) : 
  (k > 1) ∧ 
  (k % 19 = 1) ∧ 
  (k % 14 = 1) ∧ 
  (k % 9 = 1) → 
  k ≥ 2395 :=
by sorry

theorem smallest_integer_is_2395 : 
  (2395 > 1) ∧ 
  (2395 % 19 = 1) ∧ 
  (2395 % 14 = 1) ∧ 
  (2395 % 9 = 1) :=
by sorry

end smallest_integer_with_remainder_one_smallest_integer_is_2395_l2066_206637


namespace projection_magnitude_l2066_206685

def vector_a : ℝ × ℝ := (7, -4)
def vector_b : ℝ × ℝ := (-8, 6)

theorem projection_magnitude :
  let dot_product := vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2
  let magnitude_b := Real.sqrt (vector_b.1^2 + vector_b.2^2)
  let projection := dot_product / magnitude_b
  |projection| = 8 := by sorry

end projection_magnitude_l2066_206685


namespace intersection_point_sum_l2066_206647

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

end intersection_point_sum_l2066_206647


namespace school_capacity_l2066_206672

/-- The total number of students that can be taught at a time by four primary schools -/
def total_students (capacity1 capacity2 : ℕ) : ℕ :=
  2 * capacity1 + 2 * capacity2

/-- Theorem stating that the total number of students is 1480 -/
theorem school_capacity : total_students 400 340 = 1480 := by
  sorry

end school_capacity_l2066_206672


namespace trig_sum_equals_one_l2066_206677

theorem trig_sum_equals_one : 
  Real.sin (300 * Real.pi / 180) + Real.cos (390 * Real.pi / 180) + Real.tan (-135 * Real.pi / 180) = 1 := by
  sorry

end trig_sum_equals_one_l2066_206677


namespace circle_angle_equality_l2066_206696

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the angle between two vectors
def angle (v1 v2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem circle_angle_equality (Γ : Circle) (O A B M N : ℝ × ℝ) 
  (hO : O = Γ.center)
  (hA : PointOnCircle Γ A)
  (hB : PointOnCircle Γ B)
  (hM : PointOnCircle Γ M)
  (hN : PointOnCircle Γ N) :
  angle (M.1 - A.1, M.2 - A.2) (M.1 - B.1, M.2 - B.2) = 
  (angle (O.1 - A.1, O.2 - A.2) (O.1 - B.1, O.2 - B.2)) / 2 ∧
  angle (N.1 - A.1, N.2 - A.2) (N.1 - B.1, N.2 - B.2) = 
  (angle (O.1 - A.1, O.2 - A.2) (O.1 - B.1, O.2 - B.2)) / 2 :=
sorry

end circle_angle_equality_l2066_206696


namespace westward_movement_l2066_206609

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

end westward_movement_l2066_206609


namespace larger_number_proof_l2066_206640

theorem larger_number_proof (A B : ℕ+) (h1 : Nat.gcd A B = 28) 
  (h2 : Nat.lcm A B = 28 * 12 * 15) : max A B = 420 := by
  sorry

end larger_number_proof_l2066_206640


namespace douglas_fir_count_l2066_206648

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

end douglas_fir_count_l2066_206648


namespace sally_quarters_l2066_206610

def quarters_problem (initial received spent : ℕ) : ℕ :=
  initial + received - spent

theorem sally_quarters : quarters_problem 760 418 152 = 1026 := by
  sorry

end sally_quarters_l2066_206610


namespace quadratic_extrema_l2066_206667

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

end quadratic_extrema_l2066_206667


namespace inscribed_hexagon_area_l2066_206684

theorem inscribed_hexagon_area (circle_area : ℝ) (hexagon_area : ℝ) :
  circle_area = 100 * Real.pi →
  hexagon_area = 6 * (((Real.sqrt (circle_area / Real.pi))^2 * Real.sqrt 3) / 4) →
  hexagon_area = 150 * Real.sqrt 3 := by
  sorry

end inscribed_hexagon_area_l2066_206684


namespace container_volume_ratio_l2066_206697

theorem container_volume_ratio :
  ∀ (A B : ℚ),
  A > 0 → B > 0 →
  (3/4 : ℚ) * A + (1/4 : ℚ) * B = (7/8 : ℚ) * B →
  A / B = (5/6 : ℚ) := by
  sorry

end container_volume_ratio_l2066_206697


namespace book_arrangement_count_l2066_206698

/-- The number of ways to arrange books on a shelf -/
def arrange_books : ℕ := 48

/-- The number of math books -/
def num_math_books : ℕ := 4

/-- The number of English books -/
def num_english_books : ℕ := 5

/-- Theorem stating the number of ways to arrange books on a shelf -/
theorem book_arrangement_count :
  arrange_books = 
    (Nat.factorial 2) * (Nat.factorial num_math_books) * 1 :=
by sorry

end book_arrangement_count_l2066_206698


namespace sample_size_is_sixty_verify_conditions_l2066_206655

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

end sample_size_is_sixty_verify_conditions_l2066_206655


namespace smallest_integer_solution_l2066_206641

theorem smallest_integer_solution : ∃ x : ℤ, 
  (∀ y : ℤ, 10 * y^2 - 40 * y + 36 = 0 → x ≤ y) ∧ 
  (10 * x^2 - 40 * x + 36 = 0) :=
by sorry

end smallest_integer_solution_l2066_206641


namespace inverse_proportional_solution_l2066_206679

theorem inverse_proportional_solution (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 30) (h3 : x - y = 6) :
  x = 6 → y = 36 := by
sorry

end inverse_proportional_solution_l2066_206679


namespace common_chord_length_is_2_sqrt_5_l2066_206618

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

end common_chord_length_is_2_sqrt_5_l2066_206618


namespace correct_propositions_l2066_206687

-- Define the propositions
def vertical_angles_equal : Prop := True
def complementary_angles_of_equal_angles_equal : Prop := True
def corresponding_angles_equal : Prop := False
def parallel_transitivity : Prop := True
def parallel_sides_equal_or_supplementary : Prop := True
def inverse_proportion_inequality : Prop := False
def inequality_squared : Prop := False
def irrational_numbers_not_representable : Prop := False

-- Theorem statement
theorem correct_propositions :
  vertical_angles_equal ∧
  complementary_angles_of_equal_angles_equal ∧
  parallel_transitivity ∧
  parallel_sides_equal_or_supplementary ∧
  ¬corresponding_angles_equal ∧
  ¬inverse_proportion_inequality ∧
  ¬inequality_squared ∧
  ¬irrational_numbers_not_representable :=
by sorry

end correct_propositions_l2066_206687


namespace floor_neg_seven_thirds_l2066_206670

theorem floor_neg_seven_thirds : ⌊(-7 : ℝ) / 3⌋ = -3 := by sorry

end floor_neg_seven_thirds_l2066_206670


namespace intersection_point_equality_l2066_206621

-- Define the functions
def f (x : ℝ) : ℝ := 20 * x^3 + 19 * x^2
def g (x : ℝ) : ℝ := 20 * x^2 + 19 * x
def h (x : ℝ) : ℝ := 20 * x + 19

-- Theorem statement
theorem intersection_point_equality :
  ∀ x : ℝ, g x = h x → f x = g x :=
by
  sorry

end intersection_point_equality_l2066_206621


namespace floor_equality_implies_abs_diff_less_than_one_l2066_206654

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_equality_implies_abs_diff_less_than_one (x y : ℝ) :
  floor x = floor y → |x - y| < 1 := by
  sorry

end floor_equality_implies_abs_diff_less_than_one_l2066_206654


namespace tax_rate_change_income_l2066_206630

/-- Proves that given the conditions of the tax rate change and differential savings, 
    the taxpayer's annual income before tax is $45,000 -/
theorem tax_rate_change_income (I : ℝ) 
  (h1 : 0.40 * I - 0.33 * I = 3150) : I = 45000 := by
  sorry

end tax_rate_change_income_l2066_206630


namespace emir_book_purchase_l2066_206658

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

end emir_book_purchase_l2066_206658


namespace problem_solution_l2066_206612

theorem problem_solution : 
  (-(3^2) / 3 + |(-7)| + 3 * (-1/3) = 3) ∧
  ((-1)^2022 - (-1/4 - (-1/3)) / (-1/12) = 2) := by
sorry

end problem_solution_l2066_206612


namespace factorization_equality_l2066_206652

theorem factorization_equality (m n : ℝ) : 2 * m * n^2 - 4 * m * n + 2 * m = 2 * m * (n - 1)^2 := by
  sorry

end factorization_equality_l2066_206652


namespace centroid_maximizes_min_area_l2066_206682

/-- Represents a triangle in 2D space -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Calculates the area of a triangle -/
def Triangle.area (t : Triangle) : ℝ := sorry

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Calculates the centroid of a triangle -/
def Triangle.centroid (t : Triangle) : Point := sorry

/-- Calculates the minimum area of a piece resulting from dividing a triangle with a line through a given point -/
def Triangle.minAreaThroughPoint (t : Triangle) (p : Point) : ℝ := sorry

/-- Theorem: The minimum area through any point is maximized when the point is the centroid -/
theorem centroid_maximizes_min_area (t : Triangle) :
  ∀ p : Point, Triangle.minAreaThroughPoint t (Triangle.centroid t) ≥ Triangle.minAreaThroughPoint t p := by
  sorry

end centroid_maximizes_min_area_l2066_206682


namespace shaded_area_between_circles_l2066_206600

theorem shaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 10) (h₂ : r₂ = 4) :
  π * r₁^2 - π * r₂^2 = 84 * π :=
by sorry

end shaded_area_between_circles_l2066_206600


namespace river_road_cars_l2066_206604

theorem river_road_cars (buses cars : ℕ) : 
  (buses : ℚ) / cars = 1 / 3 →  -- ratio of buses to cars is 1:3
  cars = buses + 40 →           -- 40 fewer buses than cars
  cars = 60 :=                  -- prove that the number of cars is 60
by
  sorry

end river_road_cars_l2066_206604


namespace power_of_two_expression_l2066_206629

theorem power_of_two_expression : (2^2)^(2^(2^2)) = 4294967296 := by
  sorry

end power_of_two_expression_l2066_206629


namespace least_value_theorem_l2066_206619

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

end least_value_theorem_l2066_206619


namespace range_of_a_l2066_206664

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 2 4, x^2 - a*x - 8 > 0) ∧ 
  (∃ θ : ℝ, a - 1 ≤ Real.sin θ - 2) → 
  a < -2 := by sorry

end range_of_a_l2066_206664


namespace quadruple_base_triple_exponent_l2066_206649

theorem quadruple_base_triple_exponent (a b : ℝ) (x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0) :
  (4 * a) ^ (3 * b) = a ^ b * x ^ b → x = 64 * a ^ 2 := by
  sorry

end quadruple_base_triple_exponent_l2066_206649


namespace tetrahedron_vertex_equality_l2066_206683

theorem tetrahedron_vertex_equality 
  (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) 
  (h_pos_a : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₄ > 0)
  (h_pos_b : b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0 ∧ b₄ > 0)
  (h_face1 : a₁*a₂ + a₂*a₃ + a₃*a₁ = b₁*b₂ + b₂*b₃ + b₃*b₁)
  (h_face2 : a₁*a₂ + a₂*a₄ + a₄*a₁ = b₁*b₂ + b₂*b₄ + b₄*b₁)
  (h_face3 : a₁*a₃ + a₃*a₄ + a₄*a₁ = b₁*b₃ + b₃*b₄ + b₄*b₁)
  (h_face4 : a₂*a₃ + a₃*a₄ + a₄*a₂ = b₂*b₃ + b₃*b₄ + b₄*b₂) :
  (a₁ = b₁ ∧ a₂ = b₂ ∧ a₃ = b₃ ∧ a₄ = b₄) ∨ 
  (a₁ = b₂ ∧ a₂ = b₁ ∧ a₃ = b₃ ∧ a₄ = b₄) ∨
  (a₁ = b₁ ∧ a₂ = b₃ ∧ a₃ = b₂ ∧ a₄ = b₄) ∨
  (a₁ = b₁ ∧ a₂ = b₂ ∧ a₃ = b₄ ∧ a₄ = b₃) :=
by sorry

end tetrahedron_vertex_equality_l2066_206683


namespace unique_prime_cube_sum_squares_l2066_206633

theorem unique_prime_cube_sum_squares :
  ∀ p q r : ℕ,
    Prime p → Prime q → Prime r →
    p^3 = p^2 + q^2 + r^2 →
    p = 3 ∧ q = 3 ∧ r = 3 :=
by sorry

end unique_prime_cube_sum_squares_l2066_206633


namespace problem_1_problem_2_problem_3_problem_4_l2066_206659

-- Problem 1
theorem problem_1 : -8 - 6 + 24 = 10 := by sorry

-- Problem 2
theorem problem_2 : (-48) / 6 + (-21) * (-1/3) = -1 := by sorry

-- Problem 3
theorem problem_3 : (1/8 - 1/3 + 1/4) * (-24) = -1 := by sorry

-- Problem 4
theorem problem_4 : -1^4 - (1 + 0.5) * (1/3) * (1 - (-2)^2) = 1/2 := by sorry

end problem_1_problem_2_problem_3_problem_4_l2066_206659


namespace parabola_transformation_l2066_206674

/-- Given two parabolas, prove that one is a transformation of the other -/
theorem parabola_transformation (x y : ℝ) : 
  (y = 2 * x^2) →
  (y = 2 * (x - 4)^2 + 1) ↔ 
  (∃ (x' y' : ℝ), x' = x - 4 ∧ y' = y - 1 ∧ y' = 2 * x'^2) :=
by sorry

end parabola_transformation_l2066_206674


namespace koi_fish_multiple_l2066_206695

theorem koi_fish_multiple (num_koi : ℕ) (target : ℕ) : 
  num_koi = 39 → target = 64 → 
  ∃ m : ℕ, m * num_koi > target ∧ 
           ∀ k : ℕ, k * num_koi > target → k ≥ m ∧
           m * num_koi = 78 := by
  sorry

end koi_fish_multiple_l2066_206695


namespace division_remainder_l2066_206616

theorem division_remainder (x : ℕ) (h : 23 / x = 7) : 23 % x = 2 := by
  sorry

end division_remainder_l2066_206616


namespace budget_allocation_l2066_206611

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

end budget_allocation_l2066_206611


namespace departure_interval_is_six_l2066_206663

/-- Represents the tram system with a person riding along the route -/
structure TramSystem where
  tram_speed : ℝ
  person_speed : ℝ
  overtake_time : ℝ
  approach_time : ℝ

/-- The interval between tram departures from the station -/
def departure_interval (sys : TramSystem) : ℝ :=
  6

/-- Theorem stating that the departure interval is 6 minutes -/
theorem departure_interval_is_six (sys : TramSystem) 
  (h1 : sys.tram_speed > sys.person_speed) 
  (h2 : sys.overtake_time = 12)
  (h3 : sys.approach_time = 4) :
  departure_interval sys = 6 := by
  sorry

end departure_interval_is_six_l2066_206663


namespace average_speed_problem_l2066_206646

theorem average_speed_problem (D : ℝ) (S : ℝ) (h1 : D > 0) :
  (0.4 * D / S + 0.6 * D / 60) / D = 1 / 50 →
  S = 40 := by
sorry

end average_speed_problem_l2066_206646


namespace inequality_proof_l2066_206656

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : 1/x + 1/y + 1/z = 2) :
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) := by
  sorry

end inequality_proof_l2066_206656


namespace class_average_l2066_206625

theorem class_average (total_students : ℕ) 
                      (top_scorers : ℕ) 
                      (zero_scorers : ℕ) 
                      (top_score : ℕ) 
                      (rest_average : ℕ) : 
  total_students = 25 →
  top_scorers = 3 →
  zero_scorers = 5 →
  top_score = 95 →
  rest_average = 45 →
  (top_scorers * top_score + 
   (total_students - top_scorers - zero_scorers) * rest_average) / total_students = 42 := by
  sorry

end class_average_l2066_206625


namespace triangle_point_coordinates_l2066_206666

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


end triangle_point_coordinates_l2066_206666


namespace lyft_taxi_cost_difference_l2066_206620

def uber_cost : ℝ := 22
def lyft_cost : ℝ := uber_cost - 3
def taxi_cost_with_tip : ℝ := 18
def tip_percentage : ℝ := 0.2

theorem lyft_taxi_cost_difference : 
  lyft_cost - (taxi_cost_with_tip / (1 + tip_percentage)) = 4 := by
  sorry

end lyft_taxi_cost_difference_l2066_206620


namespace product_three_consecutive_divisible_by_six_l2066_206627

theorem product_three_consecutive_divisible_by_six (n : ℕ) : 
  6 ∣ (n * (n + 1) * (n + 2)) := by
  sorry

end product_three_consecutive_divisible_by_six_l2066_206627


namespace soccer_team_games_l2066_206692

theorem soccer_team_games (win lose tie rain higher : ℚ) 
  (ratio : win = 5.5 ∧ lose = 4.5 ∧ tie = 2.5 ∧ rain = 1 ∧ higher = 3.5)
  (lost_games : ℚ) (h_lost : lost_games = 13.5) :
  (win + lose + tie + rain + higher) * (lost_games / lose) = 51 := by
  sorry

end soccer_team_games_l2066_206692


namespace not_prime_n_pow_n_minus_4n_plus_3_l2066_206639

theorem not_prime_n_pow_n_minus_4n_plus_3 (n : ℕ) : ¬ Nat.Prime (n^n - 4*n + 3) := by
  sorry

end not_prime_n_pow_n_minus_4n_plus_3_l2066_206639


namespace delta_value_l2066_206653

theorem delta_value : ∀ Δ : ℤ, 4 * (-3) = Δ + 3 → Δ = -15 := by
  sorry

end delta_value_l2066_206653


namespace binary_division_theorem_l2066_206626

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

end binary_division_theorem_l2066_206626


namespace same_temperature_exists_l2066_206650

/-- Conversion function from Celsius to Fahrenheit -/
def celsius_to_fahrenheit (c : ℝ) : ℝ := 1.8 * c + 32

/-- Theorem stating that there exists a temperature that is the same in both Celsius and Fahrenheit scales -/
theorem same_temperature_exists : ∃ t : ℝ, t = celsius_to_fahrenheit t := by
  sorry

end same_temperature_exists_l2066_206650


namespace chocolate_manufacturer_cost_l2066_206691

/-- Proves that the cost per unit must be ≤ £340 given the problem conditions -/
theorem chocolate_manufacturer_cost (
  monthly_production : ℕ)
  (selling_price : ℝ)
  (minimum_profit : ℝ)
  (cost_per_unit : ℝ)
  (h1 : monthly_production = 400)
  (h2 : selling_price = 440)
  (h3 : minimum_profit = 40000)
  (h4 : monthly_production * selling_price - monthly_production * cost_per_unit ≥ minimum_profit) :
  cost_per_unit ≤ 340 := by
  sorry

end chocolate_manufacturer_cost_l2066_206691


namespace abs_neg_two_equals_two_l2066_206665

theorem abs_neg_two_equals_two :
  abs (-2) = 2 := by
sorry

end abs_neg_two_equals_two_l2066_206665


namespace union_M_complement_N_equals_real_l2066_206673

open Set

def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem union_M_complement_N_equals_real : M ∪ Nᶜ = univ :=
sorry

end union_M_complement_N_equals_real_l2066_206673


namespace circle_radius_reduction_l2066_206686

theorem circle_radius_reduction (r : ℝ) (h : r > 0) :
  let new_area_ratio := 1 - 0.18999999999999993
  let new_radius_ratio := 1 - 0.1
  (new_radius_ratio * r) ^ 2 = new_area_ratio * r ^ 2 := by
sorry

end circle_radius_reduction_l2066_206686


namespace proposition_falsity_l2066_206688

theorem proposition_falsity (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 5) : 
  ¬ P 4 := by
sorry

end proposition_falsity_l2066_206688


namespace sum_first_11_even_numbers_eq_132_l2066_206643

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * (i + 1))

def sum_first_n_even_numbers (n : ℕ) : ℕ :=
  (first_n_even_numbers n).sum

theorem sum_first_11_even_numbers_eq_132 : sum_first_n_even_numbers 11 = 132 := by
  sorry

end sum_first_11_even_numbers_eq_132_l2066_206643


namespace square_equation_solution_l2066_206668

theorem square_equation_solution (a : ℝ) (h : a^2 + a^2/4 = 5) : a = 2 ∨ a = -2 := by
  sorry

end square_equation_solution_l2066_206668


namespace public_swimming_pool_attendance_l2066_206602

/-- Proves the total number of people who used the public swimming pool -/
theorem public_swimming_pool_attendance 
  (child_price : ℚ) 
  (adult_price : ℚ) 
  (total_receipts : ℚ) 
  (num_children : ℕ) : 
  child_price = 3/2 →
  adult_price = 9/4 →
  total_receipts = 1422 →
  num_children = 388 →
  ∃ (num_adults : ℕ), 
    num_adults * adult_price + num_children * child_price = total_receipts ∧
    num_adults + num_children = 761 := by
  sorry

end public_swimming_pool_attendance_l2066_206602


namespace x_lower_bound_l2066_206638

def x : ℕ → ℝ
  | 0 => 1
  | 1 => 1
  | 2 => 3
  | (n + 3) => 4 * x (n + 2) - 2 * x (n + 1) - 3 * x n

theorem x_lower_bound : ∀ n : ℕ, n ≥ 3 → x n > (3/2) * (1 + 3^(n-2)) := by
  sorry

end x_lower_bound_l2066_206638


namespace carson_gold_stars_l2066_206680

/-- Represents the number of gold stars Carson earned today -/
def gold_stars_today (yesterday : ℕ) (total : ℕ) : ℕ :=
  total - yesterday

theorem carson_gold_stars : gold_stars_today 6 15 = 9 := by
  sorry

end carson_gold_stars_l2066_206680


namespace inequality_system_solution_range_l2066_206681

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! x : ℤ, (x + 2) / (4 - x) < 0 ∧ 2*x^2 + (2*a + 7)*x + 7*a < 0) →
  ((-5 ≤ a ∧ a < 3) ∨ (4 < a ∧ a ≤ 5)) :=
by sorry

end inequality_system_solution_range_l2066_206681


namespace square_difference_divisible_by_13_l2066_206671

theorem square_difference_divisible_by_13 (a b : ℕ) 
  (h1 : 1 ≤ a ∧ a ≤ 1000) 
  (h2 : 1 ≤ b ∧ b ≤ 1000) 
  (h3 : a + b = 1001) : 
  13 ∣ (a^2 - b^2) :=
sorry

end square_difference_divisible_by_13_l2066_206671


namespace sum_remainder_by_eight_l2066_206635

theorem sum_remainder_by_eight (n : ℤ) : (9 - n + (n + 5)) % 8 = 6 := by
  sorry

end sum_remainder_by_eight_l2066_206635


namespace ping_pong_balls_count_l2066_206636

/-- The number of ping-pong balls bought with tax -/
def B : ℕ := 60

/-- The sales tax rate -/
def tax_rate : ℚ := 16 / 100

theorem ping_pong_balls_count :
  (B : ℚ) * (1 + tax_rate) = (B + 3 : ℚ) :=
sorry

end ping_pong_balls_count_l2066_206636


namespace area_ratio_hexagon_triangle_l2066_206694

/-- Regular hexagon with vertices ABCDEF -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- Triangle ACE within the regular hexagon -/
def TriangleACE (h : RegularHexagon) : Set (ℝ × ℝ) :=
  {p | ∃ (i : Fin 3), p = h.vertices (2 * i)}

/-- Area of a regular hexagon -/
def area_hexagon (h : RegularHexagon) : ℝ := sorry

/-- Area of triangle ACE -/
def area_triangle (h : RegularHexagon) : ℝ := sorry

/-- The ratio of the area of triangle ACE to the area of the regular hexagon is 1/6 -/
theorem area_ratio_hexagon_triangle (h : RegularHexagon) :
  area_triangle h / area_hexagon h = 1 / 6 := by sorry

end area_ratio_hexagon_triangle_l2066_206694


namespace largest_certain_divisor_l2066_206657

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

end largest_certain_divisor_l2066_206657


namespace initial_garrison_size_l2066_206660

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

end initial_garrison_size_l2066_206660


namespace fraction_meaningful_l2066_206615

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x + 1) / (x - 2)) ↔ x ≠ 2 := by
sorry

end fraction_meaningful_l2066_206615


namespace smallest_square_containing_circle_l2066_206661

theorem smallest_square_containing_circle (r : ℝ) (h : r = 6) : 
  (2 * r) ^ 2 = 144 := by
  sorry

end smallest_square_containing_circle_l2066_206661


namespace trader_gain_percentage_l2066_206634

/-- The gain percentage of a trader selling pens -/
def gain_percentage (num_sold : ℕ) (num_gained : ℕ) : ℚ :=
  (num_gained : ℚ) / (num_sold : ℚ) * 100

/-- Theorem: The gain percentage is 25% when selling 80 pens and gaining the cost of 20 pens -/
theorem trader_gain_percentage : gain_percentage 80 20 = 25 := by
  sorry

end trader_gain_percentage_l2066_206634


namespace negative_decimal_greater_than_negative_fraction_l2066_206642

theorem negative_decimal_greater_than_negative_fraction : -0.6 > -(2/3) := by
  sorry

end negative_decimal_greater_than_negative_fraction_l2066_206642


namespace burgers_remaining_l2066_206603

theorem burgers_remaining (total_burgers : ℕ) (slices_per_burger : ℕ) 
  (friend1 friend2 friend3 friend4 friend5 : ℚ) : 
  total_burgers = 5 →
  slices_per_burger = 8 →
  friend1 = 3 / 8 →
  friend2 = 8 / 8 →
  friend3 = 5 / 8 →
  friend4 = 11 / 8 →
  friend5 = 6 / 8 →
  (total_burgers * slices_per_burger : ℚ) - (friend1 + friend2 + friend3 + friend4 + friend5) * slices_per_burger = 7 := by
  sorry

end burgers_remaining_l2066_206603


namespace no_solution_floor_plus_x_l2066_206601

theorem no_solution_floor_plus_x :
  ¬ ∃ x : ℝ, ⌊x⌋ + x = 15.3 := by sorry

end no_solution_floor_plus_x_l2066_206601


namespace stool_height_correct_l2066_206608

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

end stool_height_correct_l2066_206608
