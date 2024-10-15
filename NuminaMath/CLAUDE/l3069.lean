import Mathlib

namespace NUMINAMATH_CALUDE_standard_deviation_constant_addition_original_sd_equals_new_sd_l3069_306964

/-- The standard deviation of a list of real numbers -/
noncomputable def standardDeviation (l : List ℝ) : ℝ := sorry

/-- Adding a constant to each element in a list -/
def addConstant (l : List ℝ) (c : ℝ) : List ℝ := sorry

theorem standard_deviation_constant_addition 
  (original : List ℝ) (c : ℝ) :
  standardDeviation original = standardDeviation (addConstant original c) :=
sorry

theorem original_sd_equals_new_sd 
  (original : List ℝ) (c : ℝ) :
  standardDeviation original = 2 → 
  standardDeviation (addConstant original c) = 2 :=
sorry

end NUMINAMATH_CALUDE_standard_deviation_constant_addition_original_sd_equals_new_sd_l3069_306964


namespace NUMINAMATH_CALUDE_s_value_l3069_306928

theorem s_value (n : ℝ) (s : ℝ) (h1 : n ≠ 0) 
  (h2 : s = (20 / (2^(2*n+4) + 2^(2*n+2)))^(1/n)) : s = 1/4 := by
sorry

end NUMINAMATH_CALUDE_s_value_l3069_306928


namespace NUMINAMATH_CALUDE_product_first_two_terms_l3069_306970

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem product_first_two_terms (a₁ : ℝ) (d : ℝ) :
  arithmetic_sequence a₁ d 7 = 25 ∧ d = 3 →
  a₁ * (a₁ + d) = 70 := by
  sorry

end NUMINAMATH_CALUDE_product_first_two_terms_l3069_306970


namespace NUMINAMATH_CALUDE_circle_trajectory_and_tangent_line_l3069_306926

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the moving circle P
def circle_P (x y r : ℝ) : Prop := (x - 2)^2 + y^2 = r^2

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the tangent line l
def line_l (x y k : ℝ) : Prop := y = k * (x + 4)

-- Theorem statement
theorem circle_trajectory_and_tangent_line :
  ∀ (x y r k : ℝ),
  (∃ (x₁ y₁ : ℝ), circle_M x₁ y₁ ∧ circle_P (x₁ - 1) y₁ r) →
  (∃ (x₂ y₂ : ℝ), circle_N x₂ y₂ ∧ circle_P (x₂ + 1) y₂ (3 - r)) →
  curve_C x y →
  line_l x y k →
  (∃ (x₃ y₃ : ℝ), circle_M x₃ y₃ ∧ line_l x₃ y₃ k) →
  (∃ (x₄ y₄ : ℝ), circle_P x₄ y₄ 2 ∧ line_l x₄ y₄ k) →
  (∀ (x₅ y₅ : ℝ), curve_C x₅ y₅ → line_l x₅ y₅ k → 
    (x₅ - x)^2 + (y₅ - y)^2 ≤ (18/7)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_trajectory_and_tangent_line_l3069_306926


namespace NUMINAMATH_CALUDE_union_A_B_intersection_complement_A_B_l3069_306944

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 4 < x ∧ x < 10}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | 3 ≤ x ∧ x < 10} := by sorry

-- Theorem for the intersection of complement of A and B
theorem intersection_complement_A_B : (Set.univ \ A) ∩ B = {x | 7 ≤ x ∧ x < 10} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_complement_A_B_l3069_306944


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3069_306916

theorem geometric_sequence_sum (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  (∀ k, k ≥ 1 → S k = 3^k + r) →
  (∀ k, k ≥ 2 → a k = S k - S (k-1)) →
  (∀ k, k ≥ 2 → a k = 2 * 3^(k-1)) →
  a 1 = S 1 →
  r = -1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3069_306916


namespace NUMINAMATH_CALUDE_line_segment_representation_l3069_306930

/-- Represents the scale factor of the drawing -/
def scale_factor : ℝ := 800

/-- Represents the length of the line segment in the drawing (in inches) -/
def line_segment_length : ℝ := 4.75

/-- Calculates the actual length in feet represented by a given length in the drawing -/
def actual_length (drawing_length : ℝ) : ℝ := drawing_length * scale_factor

/-- Theorem stating that a 4.75-inch line segment on the scale drawing represents 3800 feet -/
theorem line_segment_representation : 
  actual_length line_segment_length = 3800 := by sorry

end NUMINAMATH_CALUDE_line_segment_representation_l3069_306930


namespace NUMINAMATH_CALUDE_restaurant_bill_with_discounts_l3069_306920

theorem restaurant_bill_with_discounts
  (bob_bill : ℝ) (kate_bill : ℝ) (bob_discount_rate : ℝ) (kate_discount_rate : ℝ)
  (h_bob_bill : bob_bill = 30)
  (h_kate_bill : kate_bill = 25)
  (h_bob_discount : bob_discount_rate = 0.05)
  (h_kate_discount : kate_discount_rate = 0.02) :
  bob_bill * (1 - bob_discount_rate) + kate_bill * (1 - kate_discount_rate) = 53 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_with_discounts_l3069_306920


namespace NUMINAMATH_CALUDE_gcd_490_910_l3069_306909

theorem gcd_490_910 : Nat.gcd 490 910 = 70 := by
  sorry

end NUMINAMATH_CALUDE_gcd_490_910_l3069_306909


namespace NUMINAMATH_CALUDE_route_down_length_l3069_306906

/-- Represents the hiking trip up and down a mountain -/
structure MountainHike where
  rate_up : ℝ
  time : ℝ
  rate_down_factor : ℝ

/-- Calculates the distance of a route given rate and time -/
def route_distance (rate : ℝ) (time : ℝ) : ℝ := rate * time

/-- Theorem: The route down the mountain is 18 miles long -/
theorem route_down_length (hike : MountainHike)
  (h1 : hike.rate_up = 6)
  (h2 : hike.time = 2)
  (h3 : hike.rate_down_factor = 1.5) :
  route_distance (hike.rate_up * hike.rate_down_factor) hike.time = 18 := by
  sorry

#check route_down_length

end NUMINAMATH_CALUDE_route_down_length_l3069_306906


namespace NUMINAMATH_CALUDE_singer_hire_duration_l3069_306919

theorem singer_hire_duration (hourly_rate : ℝ) (tip_percentage : ℝ) (total_paid : ℝ) 
  (h_rate : hourly_rate = 15)
  (h_tip : tip_percentage = 0.20)
  (h_total : total_paid = 54) :
  ∃ (hours : ℝ), hours = 3 ∧ 
    hourly_rate * hours * (1 + tip_percentage) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_singer_hire_duration_l3069_306919


namespace NUMINAMATH_CALUDE_tiling_condition_l3069_306966

/-- Represents a tile type with its dimensions -/
inductive TileType
  | square : TileType  -- 2 × 2 tile
  | rectangle : TileType  -- 3 × 1 tile

/-- Calculates the area of a tile -/
def tileArea (t : TileType) : ℕ :=
  match t with
  | TileType.square => 4
  | TileType.rectangle => 3

/-- Represents a floor tiling with square and rectangle tiles -/
structure Tiling (n : ℕ) where
  numTiles : ℕ
  complete : n * n = numTiles * (tileArea TileType.square + tileArea TileType.rectangle)

/-- Theorem: A square floor of size n × n can be tiled with an equal number of 2 × 2 and 3 × 1 tiles
    if and only if n is divisible by 7 -/
theorem tiling_condition (n : ℕ) :
  (∃ t : Tiling n, True) ↔ ∃ k : ℕ, n = 7 * k :=
by sorry

end NUMINAMATH_CALUDE_tiling_condition_l3069_306966


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3069_306935

theorem sqrt_fraction_simplification (x : ℝ) (h : x < -1) :
  Real.sqrt ((x + 1) / (2 - (x + 2) / x)) = Real.sqrt (|x^2 + x| / |x - 2|) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3069_306935


namespace NUMINAMATH_CALUDE_monotonicity_of_f_range_of_b_l3069_306911

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.exp x / (a * x^2 + b * x + 1)

theorem monotonicity_of_f :
  let f := f 1 1
  ∀ x₁ x₂, (x₁ < 0 ∧ x₂ < 0 ∧ x₁ < x₂) → f x₁ < f x₂ ∧
           (0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1) → f x₁ > f x₂ ∧
           (1 < x₁ ∧ x₁ < x₂) → f x₁ < f x₂ :=
sorry

theorem range_of_b :
  ∀ b : ℝ, (∀ x : ℝ, x ≥ 1 → f 0 b x ≥ 1) ↔ (0 ≤ b ∧ b ≤ Real.exp 1 - 1) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_of_f_range_of_b_l3069_306911


namespace NUMINAMATH_CALUDE_dinner_seating_arrangements_l3069_306900

theorem dinner_seating_arrangements (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 6) :
  (Nat.choose n k) * (Nat.factorial (k - 1)) = 3360 := by
  sorry

end NUMINAMATH_CALUDE_dinner_seating_arrangements_l3069_306900


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l3069_306913

theorem polynomial_expansion_equality (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  10 * p^9 * q = 120 * p^7 * q^3 → 
  p = Real.sqrt (12/13) := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l3069_306913


namespace NUMINAMATH_CALUDE_jiahao_estimate_l3069_306980

theorem jiahao_estimate (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x + 2) - (y - 1) > x - y := by
  sorry

end NUMINAMATH_CALUDE_jiahao_estimate_l3069_306980


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l3069_306992

def binomial_coefficient (n k : ℕ) : ℕ := sorry

def binomial_expansion_coefficient (r : ℕ) : ℚ :=
  (-3)^r * binomial_coefficient 5 r

theorem coefficient_of_x_squared (expansion : ℕ → ℚ) :
  expansion = binomial_expansion_coefficient →
  (∃ r : ℕ, (10 - 3 * r) / 2 = 2 ∧ expansion r = 90) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l3069_306992


namespace NUMINAMATH_CALUDE_rhombus_side_length_l3069_306982

theorem rhombus_side_length 
  (d1 d2 : ℝ) 
  (h1 : d1 * d2 = 22) 
  (h2 : d1 + d2 = 10) 
  (h3 : (1/2) * d1 * d2 = 11) : 
  ∃ (side : ℝ), side = Real.sqrt 14 ∧ side^2 = (1/4) * (d1^2 + d2^2) := by
sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l3069_306982


namespace NUMINAMATH_CALUDE_n_to_b_equals_sixteen_l3069_306918

-- Define n and b
def n : ℝ := 2 ^ (1 / 4)
def b : ℝ := 16.000000000000004

-- Theorem statement
theorem n_to_b_equals_sixteen : n ^ b = 16 := by
  sorry

end NUMINAMATH_CALUDE_n_to_b_equals_sixteen_l3069_306918


namespace NUMINAMATH_CALUDE_sams_morning_run_distance_l3069_306977

/-- Represents the distances traveled by Sam during different activities --/
structure SamDistances where
  morning_run : ℝ
  afternoon_walk : ℝ
  evening_bike : ℝ

/-- Theorem stating that given the conditions, Sam's morning run was 2 miles --/
theorem sams_morning_run_distance 
  (total_distance : ℝ) 
  (h1 : total_distance = 18) 
  (h2 : SamDistances → ℝ) 
  (h3 : ∀ d : SamDistances, h2 d = d.morning_run + d.afternoon_walk + d.evening_bike) 
  (h4 : ∀ d : SamDistances, d.afternoon_walk = 2 * d.morning_run) 
  (h5 : ∀ d : SamDistances, d.evening_bike = 12) :
  ∃ d : SamDistances, d.morning_run = 2 ∧ h2 d = total_distance := by
  sorry


end NUMINAMATH_CALUDE_sams_morning_run_distance_l3069_306977


namespace NUMINAMATH_CALUDE_calculation_proof_l3069_306989

theorem calculation_proof : (π - 3.15) ^ 0 * (-1) ^ 2023 - (-1/3) ^ (-2) = -10 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3069_306989


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l3069_306925

/-- The number of meals Joe has in a day. -/
def num_meals : ℕ := 3

/-- The number of fruit options Joe has for each meal. -/
def num_fruits : ℕ := 4

/-- The probability of choosing a specific fruit for a meal. -/
def prob_single_fruit : ℚ := 1 / num_fruits

/-- The probability of eating the same fruit for all meals. -/
def prob_same_fruit : ℚ := prob_single_fruit ^ num_meals

/-- The probability of eating at least two different kinds of fruit in a day. -/
def prob_different_fruits : ℚ := 1 - (num_fruits * prob_same_fruit)

theorem joe_fruit_probability : prob_different_fruits = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l3069_306925


namespace NUMINAMATH_CALUDE_second_number_value_l3069_306951

theorem second_number_value (a b c : ℝ) : 
  a + b + c = 110 ∧ 
  a = 2 * b ∧ 
  c = (1/3) * a → 
  b = 30 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l3069_306951


namespace NUMINAMATH_CALUDE_mnp_product_l3069_306971

theorem mnp_product (a b x y : ℝ) (m n p : ℤ) : 
  (a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1)) ↔ 
  ((a^m*x - a^n) * (a^p*y - a^3) = a^5*b^5) → 
  m * n * p = 2 := by sorry

end NUMINAMATH_CALUDE_mnp_product_l3069_306971


namespace NUMINAMATH_CALUDE_unique_solution_l3069_306993

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x^2 * y - x * y^2 - 5*x + 5*y + 3 = 0 ∧
  x^3 * y - x * y^3 - 5*x^2 + 5*y^2 + 15 = 0

/-- The solution to the system of equations is unique and equal to (4, 1) -/
theorem unique_solution : ∃! p : ℝ × ℝ, system p.1 p.2 ∧ p = (4, 1) := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3069_306993


namespace NUMINAMATH_CALUDE_colored_paper_count_l3069_306956

theorem colored_paper_count (people : ℕ) (pieces_per_person : ℕ) (leftover : ℕ) : 
  people = 6 → pieces_per_person = 7 → leftover = 3 → 
  people * pieces_per_person + leftover = 45 := by
  sorry

end NUMINAMATH_CALUDE_colored_paper_count_l3069_306956


namespace NUMINAMATH_CALUDE_triangle_side_length_l3069_306939

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area of triangle is √3
  (B = Real.pi / 3) →  -- B = 60°
  (a^2 + c^2 = 3 * a * c) →  -- Given condition
  (b^2 = 8) →  -- Equivalent to b = 2√2
  b = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3069_306939


namespace NUMINAMATH_CALUDE_max_projection_equals_face_area_l3069_306945

/-- A tetrahedron with two adjacent isosceles right triangle faces -/
structure Tetrahedron where
  /-- Length of the hypotenuse of the isosceles right triangle faces -/
  hypotenuse_length : ℝ
  /-- Dihedral angle between the two adjacent isosceles right triangle faces -/
  dihedral_angle : ℝ
  /-- Assumption that the hypotenuse length is 2 -/
  hypotenuse_is_two : hypotenuse_length = 2
  /-- Assumption that the dihedral angle is 60 degrees (π/3 radians) -/
  angle_is_sixty_degrees : dihedral_angle = π / 3

/-- The area of one isosceles right triangle face of the tetrahedron -/
def face_area (t : Tetrahedron) : ℝ := 1

/-- The maximum area of the projection of the rotating tetrahedron -/
def max_projection_area (t : Tetrahedron) : ℝ := 1

/-- Theorem stating that the maximum projection area equals the face area -/
theorem max_projection_equals_face_area (t : Tetrahedron) :
  max_projection_area t = face_area t := by sorry

end NUMINAMATH_CALUDE_max_projection_equals_face_area_l3069_306945


namespace NUMINAMATH_CALUDE_james_and_louise_ages_l3069_306949

/-- Proves that given the conditions about James and Louise's ages, the sum of their current ages is 25. -/
theorem james_and_louise_ages (J L : ℕ) : 
  J = L + 5 ∧ 
  J + 6 = 3 * (L - 3) →
  J + L = 25 := by
  sorry

end NUMINAMATH_CALUDE_james_and_louise_ages_l3069_306949


namespace NUMINAMATH_CALUDE_passengers_taken_proof_l3069_306984

/-- The number of trains per hour -/
def trains_per_hour : ℕ := 12

/-- The number of passengers each train leaves at the station -/
def passengers_left_per_train : ℕ := 200

/-- The total number of passengers stepping on and off in an hour -/
def total_passengers_per_hour : ℕ := 6240

/-- The number of passengers each train takes from the station -/
def passengers_taken_per_train : ℕ := 320

theorem passengers_taken_proof :
  passengers_taken_per_train * trains_per_hour + 
  passengers_left_per_train * trains_per_hour = 
  total_passengers_per_hour :=
by sorry

end NUMINAMATH_CALUDE_passengers_taken_proof_l3069_306984


namespace NUMINAMATH_CALUDE_car_meeting_speed_l3069_306995

/-- Proves that given the conditions of the problem, the speed of the second car must be 60 mph -/
theorem car_meeting_speed (total_distance : ℝ) (speed1 : ℝ) (start_time1 start_time2 : ℝ) (x : ℝ) : 
  total_distance = 600 →
  speed1 = 50 →
  start_time1 = 7 →
  start_time2 = 8 →
  (total_distance / 2) / speed1 + start_time1 = (total_distance / 2) / x + start_time2 →
  x = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_car_meeting_speed_l3069_306995


namespace NUMINAMATH_CALUDE_hot_dog_cost_l3069_306994

/-- Given that 6 hot dogs cost 300 cents in total, and each hot dog costs the same amount,
    prove that each hot dog costs 50 cents. -/
theorem hot_dog_cost (total_cost : ℕ) (num_hot_dogs : ℕ) (cost_per_hot_dog : ℕ) 
    (h1 : total_cost = 300)
    (h2 : num_hot_dogs = 6)
    (h3 : total_cost = num_hot_dogs * cost_per_hot_dog) : 
  cost_per_hot_dog = 50 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_cost_l3069_306994


namespace NUMINAMATH_CALUDE_no_non_zero_solutions_l3069_306950

theorem no_non_zero_solutions (a b : ℝ) : 
  (Real.sqrt (a^2 + b^2) = a^2 - b^2 → a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = |a - b| → a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = (a + b) / 2 → a = 0 ∧ b = 0) ∧
  (Real.sqrt (a^2 + b^2) = a^3 - b^3 → a = 0 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_non_zero_solutions_l3069_306950


namespace NUMINAMATH_CALUDE_expression_equality_l3069_306955

theorem expression_equality : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3069_306955


namespace NUMINAMATH_CALUDE_fair_coin_probability_difference_l3069_306910

def probability_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

theorem fair_coin_probability_difference :
  let p3 := probability_k_heads 5 3
  let p4 := probability_k_heads 5 4
  |p3 - p4| = 5 / 32 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_probability_difference_l3069_306910


namespace NUMINAMATH_CALUDE_max_value_cube_ratio_l3069_306931

theorem max_value_cube_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y)^3 / (x^3 + y^3) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cube_ratio_l3069_306931


namespace NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l3069_306988

theorem geometric_sequence_eighth_term
  (a₁ a₅ : ℚ)
  (h₁ : a₁ = 2187)
  (h₅ : a₅ = 960)
  (h_geom : ∃ r : ℚ, r ≠ 0 ∧ a₅ = a₁ * r^4) :
  ∃ a₈ : ℚ, a₈ = 35651584 / 4782969 ∧ (∃ r : ℚ, r ≠ 0 ∧ a₈ = a₁ * r^7) :=
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l3069_306988


namespace NUMINAMATH_CALUDE_angle_bisector_sum_l3069_306922

/-- A triangle with vertices P, Q, and R -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The angle bisector equation of the form ax + 2y + c = 0 -/
structure AngleBisectorEq where
  a : ℝ
  c : ℝ

/-- Given a triangle PQR, returns the angle bisector equation of ∠P -/
def angleBisectorP (t : Triangle) : AngleBisectorEq := sorry

theorem angle_bisector_sum (t : Triangle) 
  (h : t.P = (-7, 4) ∧ t.Q = (-14, -20) ∧ t.R = (2, -8)) : 
  let eq := angleBisectorP t
  eq.a + eq.c = 40 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_sum_l3069_306922


namespace NUMINAMATH_CALUDE_triangle_max_area_l3069_306961

/-- Given a triangle ABC with AB = 10 and BC:AC = 35:36, its maximum area is 1260 -/
theorem triangle_max_area (A B C : ℝ × ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (AB + BC + AC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  AB = 10 ∧ BC / AC = 35 / 36 → area ≤ 1260 := by
  sorry

#check triangle_max_area

end NUMINAMATH_CALUDE_triangle_max_area_l3069_306961


namespace NUMINAMATH_CALUDE_gcd_35_and_number_between_65_and_75_l3069_306999

theorem gcd_35_and_number_between_65_and_75 :
  ∃! n : ℕ, 65 < n ∧ n < 75 ∧ Nat.gcd 35 n = 7 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_gcd_35_and_number_between_65_and_75_l3069_306999


namespace NUMINAMATH_CALUDE_tissues_left_l3069_306974

/-- The number of tissues in one box -/
def tissues_per_box : ℕ := 160

/-- The number of boxes bought -/
def boxes_bought : ℕ := 3

/-- The number of tissues used -/
def tissues_used : ℕ := 210

/-- Theorem: Given the conditions, prove that the number of tissues left is 270 -/
theorem tissues_left : 
  tissues_per_box * boxes_bought - tissues_used = 270 := by
  sorry

end NUMINAMATH_CALUDE_tissues_left_l3069_306974


namespace NUMINAMATH_CALUDE_complex_equation_difference_l3069_306941

theorem complex_equation_difference (m n : ℝ) (i : ℂ) (h : i * i = -1) :
  (m + 2 * i) / i = n + i → n - m = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_difference_l3069_306941


namespace NUMINAMATH_CALUDE_imaginary_part_of_iz_l3069_306938

theorem imaginary_part_of_iz (z : ℂ) (h : z^2 - 4*z + 5 = 0) : 
  Complex.im (Complex.I * z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_iz_l3069_306938


namespace NUMINAMATH_CALUDE_condition_iff_in_solution_set_l3069_306901

/-- A pair of positive integers (x, y) satisfies the given condition -/
def satisfies_condition (x y : ℕ+) : Prop :=
  ∃ k : ℕ, x^2 * y + x = k * (x * y^2 + 7)

/-- The set of all pairs (x, y) that satisfy the condition -/
def solution_set : Set (ℕ+ × ℕ+) :=
  {p | p = (7, 1) ∨ p = (14, 1) ∨ p = (35, 1) ∨ p = (7, 2) ∨
       ∃ k : ℕ+, p = (7 * k, 7)}

/-- The main theorem stating the equivalence between the condition and the solution set -/
theorem condition_iff_in_solution_set (x y : ℕ+) :
  satisfies_condition x y ↔ (x, y) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_condition_iff_in_solution_set_l3069_306901


namespace NUMINAMATH_CALUDE_weekly_distance_is_1760_l3069_306933

/-- Calculates the total distance traveled by a driver in a week -/
def weekly_distance : ℕ :=
  let weekday_speed1 : ℕ := 30
  let weekday_time1 : ℕ := 3
  let weekday_speed2 : ℕ := 25
  let weekday_time2 : ℕ := 4
  let weekday_speed3 : ℕ := 40
  let weekday_time3 : ℕ := 2
  let weekday_days : ℕ := 6
  let sunday_speed : ℕ := 35
  let sunday_time : ℕ := 5
  let sunday_breaks : ℕ := 2
  let break_duration : ℕ := 30

  let weekday_distance := (weekday_speed1 * weekday_time1 + 
                           weekday_speed2 * weekday_time2 + 
                           weekday_speed3 * weekday_time3) * weekday_days
  let sunday_distance := sunday_speed * (sunday_time - sunday_breaks * break_duration / 60)
  
  weekday_distance + sunday_distance

theorem weekly_distance_is_1760 : weekly_distance = 1760 := by
  sorry

end NUMINAMATH_CALUDE_weekly_distance_is_1760_l3069_306933


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3069_306942

theorem tan_alpha_value (α : Real) :
  (∃ x y : Real, x = -1 ∧ y = Real.sqrt 3 ∧ 
   (Real.cos α * x - Real.sin α * y = 0)) →
  Real.tan α = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3069_306942


namespace NUMINAMATH_CALUDE_binary_linear_equation_ab_eq_one_l3069_306981

/-- A binary linear equation is an equation of the form ax + by = c, where a, b, and c are constants and x and y are variables. -/
def IsBinaryLinearEquation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y + c

theorem binary_linear_equation_ab_eq_one (a b : ℝ) :
  IsBinaryLinearEquation (fun x y => x^(2*a) + y^(b-1)) →
  a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_binary_linear_equation_ab_eq_one_l3069_306981


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3069_306996

theorem complex_sum_theorem (a b : ℂ) (h1 : a = 3 + 2*I) (h2 : b = 2 - I) :
  3*a + 4*b = 17 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3069_306996


namespace NUMINAMATH_CALUDE_not_always_meaningful_regression_l3069_306990

-- Define the variables and their properties
variable (x y : ℝ)
variable (scatter_points : Set (ℝ × ℝ))

-- Define the conditions
def are_correlated (x y : ℝ) : Prop := sorry

def roughly_linear_distribution (points : Set (ℝ × ℝ)) : Prop := sorry

def regression_equation_meaningful (points : Set (ℝ × ℝ)) : Prop := sorry

-- State the theorem
theorem not_always_meaningful_regression 
  (h1 : are_correlated x y)
  (h2 : roughly_linear_distribution scatter_points) :
  ¬ ∀ (data : Set (ℝ × ℝ)), regression_equation_meaningful data :=
sorry

end NUMINAMATH_CALUDE_not_always_meaningful_regression_l3069_306990


namespace NUMINAMATH_CALUDE_plant_growth_mean_l3069_306934

theorem plant_growth_mean (measurements : List ℝ) 
  (h1 : measurements.length = 15)
  (h2 : (measurements.filter (λ x => 10 ≤ x ∧ x < 20)).length = 3)
  (h3 : (measurements.filter (λ x => 20 ≤ x ∧ x < 30)).length = 7)
  (h4 : (measurements.filter (λ x => 30 ≤ x ∧ x < 40)).length = 5)
  (h5 : measurements.sum = 401) :
  measurements.sum / measurements.length = 401 / 15 := by
sorry

end NUMINAMATH_CALUDE_plant_growth_mean_l3069_306934


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l3069_306998

theorem completing_square_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) ↔ ((x - 3)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l3069_306998


namespace NUMINAMATH_CALUDE_charts_brought_by_associate_prof_l3069_306952

/-- Represents the number of charts brought by each associate professor -/
def charts_per_associate_prof : ℕ := sorry

/-- Represents the number of associate professors -/
def num_associate_profs : ℕ := sorry

/-- Represents the number of assistant professors -/
def num_assistant_profs : ℕ := sorry

theorem charts_brought_by_associate_prof :
  (2 * num_associate_profs + num_assistant_profs = 7) →
  (charts_per_associate_prof * num_associate_profs + 2 * num_assistant_profs = 11) →
  (num_associate_profs + num_assistant_profs = 6) →
  charts_per_associate_prof = 1 := by
    sorry

#check charts_brought_by_associate_prof

end NUMINAMATH_CALUDE_charts_brought_by_associate_prof_l3069_306952


namespace NUMINAMATH_CALUDE_system_solution_iff_b_in_range_l3069_306954

/-- The system of equations has a solution for any real a if and only if 0 ≤ b ≤ 2 -/
theorem system_solution_iff_b_in_range (b : ℝ) :
  (∀ a : ℝ, ∃ x y : ℝ, x^2 - 2*x + y^2 = 0 ∧ a*x + y = a*b) ↔ 0 ≤ b ∧ b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_iff_b_in_range_l3069_306954


namespace NUMINAMATH_CALUDE_inequality_proof_l3069_306967

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3069_306967


namespace NUMINAMATH_CALUDE_negative_integer_solutions_l3069_306978

def satisfies_inequalities (x : ℤ) : Prop :=
  3 * x - 2 ≥ 2 * x - 5 ∧ x / 2 - (x - 2) / 3 < 1 / 2

theorem negative_integer_solutions :
  {x : ℤ | x < 0 ∧ satisfies_inequalities x} = {-3, -2} :=
by sorry

end NUMINAMATH_CALUDE_negative_integer_solutions_l3069_306978


namespace NUMINAMATH_CALUDE_horner_operations_for_f_l3069_306968

/-- The number of operations for Horner's method on a polynomial of degree n -/
def horner_operations (n : ℕ) : ℕ := 2 * n

/-- The polynomial f(x) = 3x^6 + 4x^5 + 5x^4 + 6x^3 + 7x^2 + 8x + 1 -/
def f (x : ℝ) : ℝ := 3*x^6 + 4*x^5 + 5*x^4 + 6*x^3 + 7*x^2 + 8*x + 1

/-- The degree of the polynomial f -/
def degree_f : ℕ := 6

theorem horner_operations_for_f :
  horner_operations degree_f = 12 :=
sorry

end NUMINAMATH_CALUDE_horner_operations_for_f_l3069_306968


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l3069_306979

-- Define the edge lengths
def edge_length_1 : ℚ := 5
def edge_length_2 : ℚ := 24  -- 2 feet = 24 inches

-- Define the volumes
def volume_1 : ℚ := edge_length_1^3
def volume_2 : ℚ := edge_length_2^3

-- Theorem statement
theorem cube_volume_ratio :
  volume_1 / volume_2 = 125 / 13824 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l3069_306979


namespace NUMINAMATH_CALUDE_hundredth_ring_squares_l3069_306973

/-- The number of unit squares in the nth ring around a 2x2 central square -/
def ring_squares (n : ℕ) : ℕ := 8 * n + 8

/-- The 100th ring contains 808 unit squares -/
theorem hundredth_ring_squares : ring_squares 100 = 808 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_ring_squares_l3069_306973


namespace NUMINAMATH_CALUDE_point_movement_l3069_306912

def initial_point : ℝ × ℝ := (-2, -3)

def move_left (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1 - units, p.2)

def move_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

theorem point_movement :
  let p := initial_point
  let p' := move_left p 1
  let p'' := move_up p' 3
  p'' = (-3, 0) := by sorry

end NUMINAMATH_CALUDE_point_movement_l3069_306912


namespace NUMINAMATH_CALUDE_difference_of_squares_l3069_306904

theorem difference_of_squares (n : ℝ) : n^2 - 9 = (n + 3) * (n - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3069_306904


namespace NUMINAMATH_CALUDE_workshop_allocation_valid_l3069_306986

/-- Represents the allocation of workers in a workshop producing bolts and nuts. -/
structure WorkerAllocation where
  bolt_workers : ℕ
  nut_workers : ℕ

/-- Checks if a given worker allocation satisfies the workshop conditions. -/
def is_valid_allocation (total_workers : ℕ) (bolts_per_worker : ℕ) (nuts_per_worker : ℕ) 
    (nuts_per_bolt : ℕ) (allocation : WorkerAllocation) : Prop :=
  allocation.bolt_workers + allocation.nut_workers = total_workers ∧
  2 * (bolts_per_worker * allocation.bolt_workers) = nuts_per_worker * allocation.nut_workers

/-- Theorem stating that the specific allocation of 40 bolt workers and 50 nut workers
    is a valid solution to the workshop problem. -/
theorem workshop_allocation_valid : 
  is_valid_allocation 90 15 24 2 ⟨40, 50⟩ := by
  sorry


end NUMINAMATH_CALUDE_workshop_allocation_valid_l3069_306986


namespace NUMINAMATH_CALUDE_ones_digit_of_7_to_53_l3069_306953

theorem ones_digit_of_7_to_53 : (7^53 : ℕ) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_7_to_53_l3069_306953


namespace NUMINAMATH_CALUDE_triangle_equilateral_condition_l3069_306915

/-- If in a triangle ABC, a/cos(A) = b/cos(B) = c/cos(C), then the triangle is equilateral -/
theorem triangle_equilateral_condition (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_condition : a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.cos C) :
  A = B ∧ B = C := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_condition_l3069_306915


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_cube_l3069_306983

def is_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_multiplier_for_cube (a : ℕ) : 
  (a > 0 ∧ is_cube (5880 * a) ∧ ∀ b : ℕ, 0 < b ∧ b < a → ¬is_cube (5880 * b)) ↔ a = 1575 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_cube_l3069_306983


namespace NUMINAMATH_CALUDE_b_payment_is_360_l3069_306943

/-- Represents the payment for a group of horses in a pasture -/
structure Payment where
  horses : ℕ
  months : ℕ
  amount : ℚ

/-- Calculates the total horse-months for a payment -/
def horse_months (p : Payment) : ℕ := p.horses * p.months

/-- Theorem: Given the conditions of the pasture rental, B's payment is Rs. 360 -/
theorem b_payment_is_360 
  (total_rent : ℚ)
  (a_payment : Payment)
  (b_payment : Payment)
  (c_payment : Payment)
  (h1 : total_rent = 870)
  (h2 : a_payment.horses = 12 ∧ a_payment.months = 8)
  (h3 : b_payment.horses = 16 ∧ b_payment.months = 9)
  (h4 : c_payment.horses = 18 ∧ c_payment.months = 6) :
  b_payment.amount = 360 := by
  sorry

end NUMINAMATH_CALUDE_b_payment_is_360_l3069_306943


namespace NUMINAMATH_CALUDE_abs_sum_min_value_l3069_306902

theorem abs_sum_min_value :
  (∀ x : ℝ, |x + 1| + |2 - x| ≥ 3) ∧
  (∃ x : ℝ, |x + 1| + |2 - x| = 3) :=
by sorry

end NUMINAMATH_CALUDE_abs_sum_min_value_l3069_306902


namespace NUMINAMATH_CALUDE_banana_arrangements_l3069_306972

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem banana_arrangements :
  let total_letters : ℕ := 6
  let repeated_letter1 : ℕ := 3  -- 'a' appears 3 times
  let repeated_letter2 : ℕ := 2  -- 'n' appears 2 times
  factorial total_letters / (factorial repeated_letter1 * factorial repeated_letter2) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l3069_306972


namespace NUMINAMATH_CALUDE_equation_solution_l3069_306962

theorem equation_solution :
  ∃! x : ℝ, (1 : ℝ) / (x - 2) = (3 : ℝ) / (x - 5) ∧ x = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3069_306962


namespace NUMINAMATH_CALUDE_optimal_oil_storage_l3069_306963

/-- Represents the optimal solution for storing oil in barrels -/
structure OilStorage where
  small_barrels : ℕ
  large_barrels : ℕ

/-- Checks if a given oil storage solution is valid -/
def is_valid_solution (total_oil : ℕ) (small_capacity : ℕ) (large_capacity : ℕ) (solution : OilStorage) : Prop :=
  solution.small_barrels * small_capacity + solution.large_barrels * large_capacity = total_oil

/-- Checks if a given oil storage solution is optimal -/
def is_optimal_solution (total_oil : ℕ) (small_capacity : ℕ) (large_capacity : ℕ) (solution : OilStorage) : Prop :=
  is_valid_solution total_oil small_capacity large_capacity solution ∧
  ∀ (other : OilStorage), 
    is_valid_solution total_oil small_capacity large_capacity other → 
    solution.small_barrels + solution.large_barrels ≤ other.small_barrels + other.large_barrels

/-- Theorem stating that the given solution is optimal for the oil storage problem -/
theorem optimal_oil_storage :
  is_optimal_solution 95 5 6 ⟨1, 15⟩ := by sorry

end NUMINAMATH_CALUDE_optimal_oil_storage_l3069_306963


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l3069_306923

theorem inequality_system_solutions : 
  ∃! (s : Finset ℕ), 
    (∀ x ∈ s, (3 - 2*x > 0 ∧ 2*x - 7 ≤ 4*x + 7)) ∧ 
    (∀ x : ℕ, (3 - 2*x > 0 ∧ 2*x - 7 ≤ 4*x + 7) → x ∈ s) ∧
    Finset.card s = 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l3069_306923


namespace NUMINAMATH_CALUDE_container_volume_increase_l3069_306946

/-- Given a container with an initial volume and a volume multiplier, 
    calculate the new volume after applying the multiplier. -/
def new_volume (initial_volume : ℝ) (volume_multiplier : ℝ) : ℝ :=
  initial_volume * volume_multiplier

/-- Theorem: If a container's volume is multiplied by 16, and its original volume was 5 gallons,
    then the new volume is 80 gallons. -/
theorem container_volume_increase :
  let initial_volume : ℝ := 5
  let volume_multiplier : ℝ := 16
  new_volume initial_volume volume_multiplier = 80 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_increase_l3069_306946


namespace NUMINAMATH_CALUDE_journey_speed_problem_l3069_306937

theorem journey_speed_problem (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 448 →
  total_time = 20 →
  second_half_speed = 24 →
  ∃ first_half_speed : ℝ,
    first_half_speed * (total_time / 2) = total_distance / 2 ∧
    second_half_speed * (total_time / 2) = total_distance / 2 ∧
    first_half_speed = 21 := by
  sorry


end NUMINAMATH_CALUDE_journey_speed_problem_l3069_306937


namespace NUMINAMATH_CALUDE_perfect_square_from_fraction_pairs_l3069_306907

theorem perfect_square_from_fraction_pairs (N : ℕ+) 
  (h : ∃! (pairs : Finset (ℕ+ × ℕ+)), pairs.card = 2005 ∧ 
    ∀ (x y : ℕ+), (x, y) ∈ pairs ↔ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / N) : 
  ∃ (k : ℕ+), N = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_from_fraction_pairs_l3069_306907


namespace NUMINAMATH_CALUDE_square_side_length_l3069_306959

theorem square_side_length (area : Real) (side : Real) : 
  area = 25 → side * side = area → side = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3069_306959


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3069_306927

/-- An isosceles triangle with perimeter 3.74 and leg length 1.5 has a base length of 0.74 -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let perimeter : ℝ := 3.74
    let leg : ℝ := 1.5
    (2 * leg + base = perimeter) → (base = 0.74)

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 0.74 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3069_306927


namespace NUMINAMATH_CALUDE_min_additional_marbles_for_john_l3069_306991

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed -/
theorem min_additional_marbles_for_john : 
  min_additional_marbles 15 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_additional_marbles_for_john_l3069_306991


namespace NUMINAMATH_CALUDE_football_team_selection_l3069_306960

theorem football_team_selection (n : ℕ) (k : ℕ) :
  let total_students : ℕ := 31
  let team_size : ℕ := 11
  let remaining_students : ℕ := total_students - 2
  (Nat.choose total_students team_size) - (Nat.choose remaining_students team_size) =
    2 * (Nat.choose remaining_students (team_size - 1)) + (Nat.choose remaining_students (team_size - 2)) :=
by sorry

end NUMINAMATH_CALUDE_football_team_selection_l3069_306960


namespace NUMINAMATH_CALUDE_function_values_l3069_306969

/-- A function from ℝ² to ℝ² defined by f(x, y) = (kx, y + b) -/
def f (k b : ℝ) : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (k * x, y + b)

/-- Theorem stating that if f(3, 1) = (6, 2), then k = 2 and b = 1 -/
theorem function_values (k b : ℝ) : f k b (3, 1) = (6, 2) → k = 2 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_values_l3069_306969


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_times_three_l3069_306948

theorem arithmetic_sequence_sum_times_three : 
  ∀ (a l d n : ℕ), 
    a = 50 → 
    l = 95 → 
    d = 3 → 
    n * d = l - a + d → 
    3 * (n / 2 * (a + l)) = 3480 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_times_three_l3069_306948


namespace NUMINAMATH_CALUDE_sine_amplitude_l3069_306940

theorem sine_amplitude (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : ∀ x, -3 ≤ a * Real.sin (b * x + c) + d) 
  (h6 : ∀ x, a * Real.sin (b * x + c) + d ≤ 5) : a = 4 := by
sorry

end NUMINAMATH_CALUDE_sine_amplitude_l3069_306940


namespace NUMINAMATH_CALUDE_headphone_cost_l3069_306947

/-- The cost of the headphone set given Amanda's shopping scenario -/
theorem headphone_cost (initial_amount : ℕ) (cassette_cost : ℕ) (num_cassettes : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 50 →
  cassette_cost = 9 →
  num_cassettes = 2 →
  remaining_amount = 7 →
  initial_amount - (num_cassettes * cassette_cost) - remaining_amount = 25 := by
sorry

end NUMINAMATH_CALUDE_headphone_cost_l3069_306947


namespace NUMINAMATH_CALUDE_triangle_with_sides_4_6_5_l3069_306975

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_with_sides_4_6_5 :
  can_form_triangle 4 6 5 ∧
  ¬can_form_triangle 4 6 2 ∧
  ¬can_form_triangle 4 6 10 ∧
  ¬can_form_triangle 4 6 11 :=
sorry

end NUMINAMATH_CALUDE_triangle_with_sides_4_6_5_l3069_306975


namespace NUMINAMATH_CALUDE_jerky_order_theorem_l3069_306929

/-- Calculates the total number of jerky bags for a customer order -/
def customer_order_bags (production_rate : ℕ) (initial_inventory : ℕ) (production_days : ℕ) : ℕ :=
  production_rate * production_days + initial_inventory

/-- Theorem stating that given the specific conditions, the customer order is 60 bags -/
theorem jerky_order_theorem :
  let production_rate := 10
  let initial_inventory := 20
  let production_days := 4
  customer_order_bags production_rate initial_inventory production_days = 60 := by
  sorry

#eval customer_order_bags 10 20 4  -- Should output 60

end NUMINAMATH_CALUDE_jerky_order_theorem_l3069_306929


namespace NUMINAMATH_CALUDE_average_lunchmeat_price_l3069_306957

def joan_bologna_weight : ℝ := 3
def joan_bologna_price : ℝ := 2.80
def grant_pastrami_weight : ℝ := 2
def grant_pastrami_price : ℝ := 1.80

theorem average_lunchmeat_price :
  let total_weight := joan_bologna_weight + grant_pastrami_weight
  let total_cost := joan_bologna_weight * joan_bologna_price + grant_pastrami_weight * grant_pastrami_price
  total_cost / total_weight = 2.40 := by
sorry

end NUMINAMATH_CALUDE_average_lunchmeat_price_l3069_306957


namespace NUMINAMATH_CALUDE_charles_whistle_count_l3069_306921

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The difference between Sean's and Charles' whistles -/
def whistle_difference : ℕ := 32

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := sean_whistles - whistle_difference

theorem charles_whistle_count : charles_whistles = 13 := by
  sorry

end NUMINAMATH_CALUDE_charles_whistle_count_l3069_306921


namespace NUMINAMATH_CALUDE_sector_area_l3069_306924

/-- Given a circular sector with central angle 2 radians and circumference 4 cm, its area is 1 cm² -/
theorem sector_area (θ : ℝ) (c : ℝ) (A : ℝ) : 
  θ = 2 → c = 4 → A = 1 → A = (θ * c^2) / (8 * π) := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3069_306924


namespace NUMINAMATH_CALUDE_sqrt_seven_to_six_l3069_306908

theorem sqrt_seven_to_six : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_six_l3069_306908


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3069_306917

theorem fraction_decomposition (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  (x^2 - 2*x + 5) / (x^3 - x) = (-5)/x + (6*x - 2) / (x^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3069_306917


namespace NUMINAMATH_CALUDE_sphere_volume_larger_than_cube_l3069_306965

/-- Given a sphere and a cube with equal surface areas, the volume of the sphere is larger than the volume of the cube. -/
theorem sphere_volume_larger_than_cube (r : ℝ) (s : ℝ) (h : 4 * Real.pi * r^2 = 6 * s^2) :
  (4/3) * Real.pi * r^3 > s^3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_larger_than_cube_l3069_306965


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3069_306958

theorem sufficient_not_necessary (x y : ℝ) :
  (x < y ∧ y < 0 → x^2 > y^2) ∧
  ∃ x y, x^2 > y^2 ∧ ¬(x < y ∧ y < 0) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3069_306958


namespace NUMINAMATH_CALUDE_six_people_non_adjacent_seating_l3069_306997

/-- The number of ways to seat n people around a round table. -/
def roundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to seat n people around a round table
    where two specific individuals are adjacent. -/
def adjacentArrangements (n : ℕ) : ℕ := (n - 1) * Nat.factorial (n - 2)

/-- The number of ways to seat 6 people around a round table
    where two specific individuals are not adjacent. -/
theorem six_people_non_adjacent_seating :
  roundTableArrangements 6 - adjacentArrangements 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_six_people_non_adjacent_seating_l3069_306997


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l3069_306936

theorem least_integer_greater_than_sqrt_500 : ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
  sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l3069_306936


namespace NUMINAMATH_CALUDE_evaluate_expression_l3069_306903

theorem evaluate_expression : (3^2)^2 - (2^3)^3 = -431 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3069_306903


namespace NUMINAMATH_CALUDE_william_road_time_l3069_306985

def departure_time : Nat := 7 * 60  -- 7:00 AM in minutes
def arrival_time : Nat := 20 * 60  -- 8:00 PM in minutes
def stop_durations : List Nat := [25, 10, 25]

def total_journey_time : Nat := arrival_time - departure_time
def total_stop_time : Nat := stop_durations.sum

theorem william_road_time :
  (total_journey_time - total_stop_time) / 60 = 12 := by sorry

end NUMINAMATH_CALUDE_william_road_time_l3069_306985


namespace NUMINAMATH_CALUDE_marlon_gift_card_balance_l3069_306914

def gift_card_balance (initial_balance : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ) : ℝ :=
  let remaining_after_monday := initial_balance * (1 - monday_fraction)
  remaining_after_monday * (1 - tuesday_fraction)

theorem marlon_gift_card_balance :
  gift_card_balance 200 (1/2) (1/4) = 75 := by
  sorry

end NUMINAMATH_CALUDE_marlon_gift_card_balance_l3069_306914


namespace NUMINAMATH_CALUDE_birds_storks_difference_l3069_306987

theorem birds_storks_difference (initial_storks initial_birds additional_birds : ℕ) : 
  initial_storks = 5 →
  initial_birds = 3 →
  additional_birds = 4 →
  (initial_birds + additional_birds) - initial_storks = 2 :=
by sorry

end NUMINAMATH_CALUDE_birds_storks_difference_l3069_306987


namespace NUMINAMATH_CALUDE_power_fraction_equality_l3069_306976

theorem power_fraction_equality : (16^6 * 8^3) / 4^10 = 2^13 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l3069_306976


namespace NUMINAMATH_CALUDE_inez_remaining_money_l3069_306932

def initial_amount : ℕ := 150
def pad_cost : ℕ := 50

theorem inez_remaining_money :
  let skate_cost : ℕ := initial_amount / 2
  let after_skates : ℕ := initial_amount - skate_cost
  let remaining : ℕ := after_skates - pad_cost
  remaining = 25 := by sorry

end NUMINAMATH_CALUDE_inez_remaining_money_l3069_306932


namespace NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3069_306905

theorem quadratic_root_in_unit_interval (a b : ℝ) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ 3 * a * x^2 + 2 * b * x - (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_in_unit_interval_l3069_306905
