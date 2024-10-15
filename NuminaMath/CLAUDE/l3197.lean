import Mathlib

namespace NUMINAMATH_CALUDE_assistant_professor_pencils_l3197_319769

theorem assistant_professor_pencils :
  ∀ (A B P : ℕ),
    A + B = 7 →
    2 * A + P * B = 10 →
    A + 2 * B = 11 →
    P = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_assistant_professor_pencils_l3197_319769


namespace NUMINAMATH_CALUDE_complex_roots_equilateral_triangle_l3197_319721

theorem complex_roots_equilateral_triangle (p q z₁ z₂ : ℂ) :
  z₂^2 + p*z₂ + q = 0 →
  z₁^2 + p*z₁ + q = 0 →
  z₂ = Complex.exp (2*Real.pi*Complex.I/3) * z₁ →
  p^2 / q = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_equilateral_triangle_l3197_319721


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3197_319779

theorem quadratic_minimum (k : ℝ) : 
  (∀ x : ℝ, 3 ≤ x ∧ x ≤ 5 → (1/2) * (x - 1)^2 + k ≥ 3) ∧
  (∃ x : ℝ, 3 ≤ x ∧ x ≤ 5 ∧ (1/2) * (x - 1)^2 + k = 3) →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3197_319779


namespace NUMINAMATH_CALUDE_stereographic_projection_is_inversion_l3197_319722

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  point : Point3D

/-- Represents a sphere (Earth) -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Stereographic projection from a pole onto a plane -/
def stereographicProjection (sphere : Sphere) (pole : Point3D) (plane : Plane) (point : Point3D) : Point3D :=
  sorry

/-- The mapping between corresponding points on two planes -/
def planeMapping (sphere : Sphere) (plane1 : Plane) (plane2 : Plane) (point : Point3D) : Point3D :=
  sorry

/-- Definition of inversion -/
def isInversion (f : Point3D → Point3D) (center : Point3D) (radius : ℝ) : Prop :=
  sorry

theorem stereographic_projection_is_inversion
  (sphere : Sphere)
  (northPole : Point3D)
  (southPole : Point3D)
  (plane1 : Plane)
  (plane2 : Plane)
  (h1 : plane1.point = northPole)
  (h2 : plane2.point = southPole)
  (h3 : northPole.z = sphere.radius)
  (h4 : southPole.z = -sphere.radius) :
  ∃ (center : Point3D) (radius : ℝ),
    isInversion (planeMapping sphere plane1 plane2) center radius :=
  sorry

end NUMINAMATH_CALUDE_stereographic_projection_is_inversion_l3197_319722


namespace NUMINAMATH_CALUDE_petya_has_higher_chance_of_winning_l3197_319704

structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

def vasya_wins (game : CandyGame) : ℝ :=
  1 - game.prob_two_caramels

def petya_wins (game : CandyGame) : ℝ :=
  game.prob_two_caramels

theorem petya_has_higher_chance_of_winning (game : CandyGame)
  (h1 : game.total_candies = 25)
  (h2 : game.prob_two_caramels = 0.54)
  : petya_wins game > vasya_wins game := by
  sorry

end NUMINAMATH_CALUDE_petya_has_higher_chance_of_winning_l3197_319704


namespace NUMINAMATH_CALUDE_function_identity_l3197_319780

theorem function_identity (f : ℕ → ℕ) :
  (∀ n : ℕ, f n + f (f n) + f (f (f n)) = 3 * n) →
  (∀ n : ℕ, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l3197_319780


namespace NUMINAMATH_CALUDE_subtraction_result_l3197_319738

theorem subtraction_result : 34.256 - 12.932 - 1.324 = 20 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l3197_319738


namespace NUMINAMATH_CALUDE_plane_through_origin_l3197_319719

/-- A plane in 3D Cartesian coordinates represented by the equation Ax + By + Cz = 0 -/
structure Plane3D where
  A : ℝ
  B : ℝ
  C : ℝ
  not_all_zero : A ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0

/-- A point in 3D Cartesian coordinates -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin in 3D Cartesian coordinates -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- A point lies on a plane if it satisfies the plane's equation -/
def lies_on (p : Point3D) (plane : Plane3D) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z = 0

/-- A plane passes through the origin if the origin lies on the plane -/
def passes_through_origin (plane : Plane3D) : Prop :=
  lies_on origin plane

theorem plane_through_origin (plane : Plane3D) : 
  passes_through_origin plane :=
sorry

end NUMINAMATH_CALUDE_plane_through_origin_l3197_319719


namespace NUMINAMATH_CALUDE_profit_doubling_l3197_319710

theorem profit_doubling (cost : ℝ) (price : ℝ) (h1 : price = 1.5 * cost) :
  let double_price := 2 * price
  (double_price - cost) / cost * 100 = 200 :=
by sorry

end NUMINAMATH_CALUDE_profit_doubling_l3197_319710


namespace NUMINAMATH_CALUDE_paradise_park_ferris_wheel_small_seat_capacity_l3197_319749

/-- Represents the Ferris wheel in paradise park -/
structure FerrisWheel where
  small_seats : Nat
  large_seats : Nat
  small_seat_capacity : Nat

/-- Calculates the total capacity of small seats on the Ferris wheel -/
def total_small_seat_capacity (fw : FerrisWheel) : Nat :=
  fw.small_seats * fw.small_seat_capacity

theorem paradise_park_ferris_wheel_small_seat_capacity :
  ∃ (fw : FerrisWheel), 
    fw.small_seats = 2 ∧ 
    fw.large_seats = 23 ∧ 
    fw.small_seat_capacity = 14 ∧ 
    total_small_seat_capacity fw = 28 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_ferris_wheel_small_seat_capacity_l3197_319749


namespace NUMINAMATH_CALUDE_triangle_side_length_l3197_319744

theorem triangle_side_length (perimeter side2 side3 : ℝ) 
  (h_perimeter : perimeter = 160)
  (h_side2 : side2 = 50)
  (h_side3 : side3 = 70) :
  perimeter - side2 - side3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3197_319744


namespace NUMINAMATH_CALUDE_circle_and_m_value_l3197_319740

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem conditions
def problem_conditions (C : Circle) (m : ℝ) : Prop :=
  -- Center C is on the line 2x-y-7=0
  2 * C.center.1 - C.center.2 - 7 = 0 ∧
  -- Circle intersects y-axis at (0, -4) and (0, -2)
  (0 - C.center.1)^2 + (-4 - C.center.2)^2 = C.radius^2 ∧
  (0 - C.center.1)^2 + (-2 - C.center.2)^2 = C.radius^2 ∧
  -- Line x+2y+m=0 intersects circle C
  ∃ (A B : ℝ × ℝ), 
    (A.1 + 2*A.2 + m = 0) ∧
    (B.1 + 2*B.2 + m = 0) ∧
    (A.1 - C.center.1)^2 + (A.2 - C.center.2)^2 = C.radius^2 ∧
    (B.1 - C.center.1)^2 + (B.2 - C.center.2)^2 = C.radius^2 ∧
  -- Parallelogram ACBD with CA and CB as adjacent sides, D on circle C
  ∃ (D : ℝ × ℝ),
    (D.1 - C.center.1)^2 + (D.2 - C.center.2)^2 = C.radius^2

-- Theorem statement
theorem circle_and_m_value (C : Circle) (m : ℝ) :
  problem_conditions C m →
  (C.center = (2, -3) ∧ C.radius^2 = 5) ∧ (m = 3/2 ∨ m = 13/2) :=
sorry

end NUMINAMATH_CALUDE_circle_and_m_value_l3197_319740


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3197_319766

theorem arithmetic_sequence_terms (a₁ aₙ : ℤ) (n : ℕ) : 
  a₁ = -1 → aₙ = 89 → aₙ = a₁ + (n - 1) * ((aₙ - a₁) / (n - 1)) → n = 46 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l3197_319766


namespace NUMINAMATH_CALUDE_negation_of_existence_squared_positive_l3197_319724

theorem negation_of_existence_squared_positive :
  (¬ ∃ x : ℝ, x^2 > 0) ↔ (∀ x : ℝ, x^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_squared_positive_l3197_319724


namespace NUMINAMATH_CALUDE_symmetry_y_axis_l3197_319795

/-- Given a point (x, y) in the plane, its reflection across the y-axis is the point (-x, y) -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- A point q is symmetric to p with respect to the y-axis if q is the reflection of p across the y-axis -/
def symmetric_y_axis (p q : ℝ × ℝ) : Prop :=
  q = reflect_y_axis p

theorem symmetry_y_axis :
  symmetric_y_axis (-2, 3) (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_y_axis_l3197_319795


namespace NUMINAMATH_CALUDE_house_painting_time_l3197_319742

/-- Represents the time taken to paint a house given individual rates and a break -/
theorem house_painting_time
  (alice_rate : ℝ) (bob_rate : ℝ) (carlos_rate : ℝ) (break_time : ℝ) (total_time : ℝ)
  (h_alice : alice_rate = 1 / 4)
  (h_bob : bob_rate = 1 / 6)
  (h_carlos : carlos_rate = 1 / 12)
  (h_break : break_time = 2)
  (h_equation : (alice_rate + bob_rate + carlos_rate) * (total_time - break_time) = 1) :
  (1 / 4 + 1 / 6 + 1 / 12) * (total_time - 2) = 1 := by
sorry


end NUMINAMATH_CALUDE_house_painting_time_l3197_319742


namespace NUMINAMATH_CALUDE_smallest_n_for_divisible_sum_l3197_319790

theorem smallest_n_for_divisible_sum (n : ℕ) : n ≥ 4 → (
  (∀ S : Finset ℤ, S.card = n → ∃ a b c d : ℤ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 20 ∣ (a + b - c - d))
  ↔ n ≥ 9
) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisible_sum_l3197_319790


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_A_subset_C_iff_l3197_319716

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Statement 1: Prove that (ℝ\A) ∩ B = {x | 2 < x < 3 or 7 ≤ x < 10}
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Statement 2: Prove that A ⊆ C(a) if and only if a ≥ 7
theorem A_subset_C_iff (a : ℝ) :
  A ⊆ C a ↔ a ≥ 7 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_A_subset_C_iff_l3197_319716


namespace NUMINAMATH_CALUDE_diminished_value_proof_l3197_319717

theorem diminished_value_proof : 
  let numbers := [12, 16, 18, 21, 28]
  let smallest_number := 1015
  let diminished_value := 7
  (∀ n ∈ numbers, (smallest_number - diminished_value) % n = 0) ∧
  (∀ m < smallest_number, ∃ n ∈ numbers, ∀ k : ℕ, m - k ≠ 0 ∨ (m - k) % n ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_diminished_value_proof_l3197_319717


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l3197_319755

/-- Given an arithmetic sequence where the 5th term is 25 and the 8th term is 43,
    prove that the 10th term is 55. -/
theorem arithmetic_sequence_10th_term
  (a : ℕ → ℕ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic property
  (h_5th : a 5 = 25)  -- 5th term is 25
  (h_8th : a 8 = 43)  -- 8th term is 43
  : a 10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l3197_319755


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_1023_l3197_319727

theorem largest_gcd_of_sum_1023 :
  ∃ (a b : ℕ+), a + b = 1023 ∧
  ∀ (c d : ℕ+), c + d = 1023 → Nat.gcd c.val d.val ≤ Nat.gcd a.val b.val ∧
  Nat.gcd a.val b.val = 341 :=
by sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_1023_l3197_319727


namespace NUMINAMATH_CALUDE_family_juice_consumption_l3197_319792

/-- The amount of juice consumed by a family in a week -/
def juice_consumption_per_week (juice_per_serving : ℝ) (servings_per_day : ℕ) (days_per_week : ℕ) : ℝ :=
  juice_per_serving * (servings_per_day : ℝ) * (days_per_week : ℝ)

/-- Theorem stating that a family drinking 0.2 liters of juice three times a day consumes 4.2 liters in a week -/
theorem family_juice_consumption :
  juice_consumption_per_week 0.2 3 7 = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_family_juice_consumption_l3197_319792


namespace NUMINAMATH_CALUDE_x_value_approximation_l3197_319782

/-- The value of x in the given equation is approximately 179692.08 -/
theorem x_value_approximation : 
  let x := 3.5 * ((3.6 * 0.48 * 2.50)^2 / (0.12 * 0.09 * 0.5)) * Real.log (2.5 * 4.3)
  ∃ ε > 0, |x - 179692.08| < ε :=
by sorry

end NUMINAMATH_CALUDE_x_value_approximation_l3197_319782


namespace NUMINAMATH_CALUDE_no_real_solutions_l3197_319751

theorem no_real_solutions (a b c : ℝ) : ¬ ∃ x y z : ℝ, 
  (a^2 + b^2 + c^2 + 3*(x^2 + y^2 + z^2) = 6) ∧ (a*x + b*y + c*z = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3197_319751


namespace NUMINAMATH_CALUDE_onion_root_tip_no_tetrads_l3197_319770

/-- Represents the type of cell division a plant tissue undergoes -/
inductive CellDivisionType
  | Mitosis
  | Meiosis

/-- Represents whether tetrads can be observed in a given tissue -/
def can_observe_tetrads (division_type : CellDivisionType) : Prop :=
  match division_type with
  | CellDivisionType.Meiosis => true
  | CellDivisionType.Mitosis => false

/-- The cell division type of onion root tips -/
def onion_root_tip_division : CellDivisionType := CellDivisionType.Mitosis

theorem onion_root_tip_no_tetrads :
  ¬(can_observe_tetrads onion_root_tip_division) :=
by sorry

end NUMINAMATH_CALUDE_onion_root_tip_no_tetrads_l3197_319770


namespace NUMINAMATH_CALUDE_function_property_proof_l3197_319733

theorem function_property_proof (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x = f (4 - x))
  (h2 : ∀ x : ℝ, f (x + 1) = -f (x + 3))
  (h3 : ∃ a b : ℝ, ∀ x ∈ Set.Icc 0 4, f x = |x - a| + b) :
  ∃ a b : ℝ, (∀ x ∈ Set.Icc 0 4, f x = |x - a| + b) ∧ a + b = 1 := by
  sorry


end NUMINAMATH_CALUDE_function_property_proof_l3197_319733


namespace NUMINAMATH_CALUDE_candy_difference_l3197_319753

/- Define the number of candies each person can eat -/
def nellie_candies : ℕ := 12
def jacob_candies : ℕ := nellie_candies / 2
def lana_candies : ℕ := jacob_candies - 3

/- Define the total number of candies in the bucket -/
def total_candies : ℕ := 30

/- Define the number of remaining candies after they ate -/
def remaining_candies : ℕ := 9

/- Theorem statement -/
theorem candy_difference :
  jacob_candies - lana_candies = 3 :=
by sorry

end NUMINAMATH_CALUDE_candy_difference_l3197_319753


namespace NUMINAMATH_CALUDE_intersecting_chords_length_l3197_319746

/-- Power of a Point theorem for intersecting chords --/
axiom power_of_point (AP BP CP DP : ℝ) : AP * BP = CP * DP

/-- Proof that DP = 8/3 given the conditions --/
theorem intersecting_chords_length (AP BP CP DP : ℝ) 
  (h1 : AP = 4) 
  (h2 : CP = 9) 
  (h3 : BP = 6) : 
  DP = 8/3 := by
  sorry


end NUMINAMATH_CALUDE_intersecting_chords_length_l3197_319746


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_one_l3197_319768

theorem sqrt_difference_equals_one : Real.sqrt 25 - Real.sqrt 16 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_one_l3197_319768


namespace NUMINAMATH_CALUDE_inequality_proof_l3197_319794

theorem inequality_proof (a b : ℝ) (h1 : b < 0) (h2 : 0 < a) : a - b > a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3197_319794


namespace NUMINAMATH_CALUDE_square_perimeters_product_l3197_319784

theorem square_perimeters_product (x y : ℝ) 
  (h1 : x ^ 2 + y ^ 2 = 85)
  (h2 : x ^ 2 - y ^ 2 = 45)
  : (4 * x) * (4 * y) = 32 * Real.sqrt 325 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_product_l3197_319784


namespace NUMINAMATH_CALUDE_all_multiples_of_three_after_four_iterations_no_2020_on_tenth_page_l3197_319754

/-- Represents the numbers written by three schoolchildren on their notebooks. -/
structure SchoolchildrenNumbers where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Performs one iteration of the number writing process. -/
def iterate (nums : SchoolchildrenNumbers) : SchoolchildrenNumbers :=
  { a := nums.c - nums.b
  , b := nums.a - nums.c
  , c := nums.b - nums.a }

/-- Performs n iterations of the number writing process. -/
def iterateN (n : ℕ) (nums : SchoolchildrenNumbers) : SchoolchildrenNumbers :=
  match n with
  | 0 => nums
  | n + 1 => iterate (iterateN n nums)

/-- Theorem stating that after 4 iterations, all numbers are multiples of 3. -/
theorem all_multiples_of_three_after_four_iterations (initial : SchoolchildrenNumbers) :
  ∃ k l m : ℤ, 
    let result := iterateN 4 initial
    result.a = 3 * k ∧ result.b = 3 * l ∧ result.c = 3 * m :=
  sorry

/-- Theorem stating that 2020 cannot appear on the 10th page. -/
theorem no_2020_on_tenth_page (initial : SchoolchildrenNumbers) :
  let result := iterateN 9 initial
  result.a ≠ 2020 ∧ result.b ≠ 2020 ∧ result.c ≠ 2020 :=
  sorry

end NUMINAMATH_CALUDE_all_multiples_of_three_after_four_iterations_no_2020_on_tenth_page_l3197_319754


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_1234567890_div_17_l3197_319787

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem least_subtraction_1234567890_div_17 :
  ∃ (k : Nat), k < 17 ∧ (1234567890 - k) % 17 = 0 ∧
  ∀ (m : Nat), m < k → (1234567890 - m) % 17 ≠ 0 ∧ k = 5 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_1234567890_div_17_l3197_319787


namespace NUMINAMATH_CALUDE_cos_negative_nineteen_pi_sixths_l3197_319799

theorem cos_negative_nineteen_pi_sixths :
  Real.cos (-19 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_nineteen_pi_sixths_l3197_319799


namespace NUMINAMATH_CALUDE_box_height_l3197_319725

theorem box_height (long_width short_width top_area total_area : ℝ) 
  (h_long : long_width = 8)
  (h_short : short_width = 5)
  (h_top : top_area = 40)
  (h_total : total_area = 236) : 
  ∃ height : ℝ, 
    2 * long_width * height + 2 * short_width * height + 2 * top_area = total_area ∧ 
    height = 6 := by
  sorry

end NUMINAMATH_CALUDE_box_height_l3197_319725


namespace NUMINAMATH_CALUDE_square_area_from_rectangles_l3197_319705

/-- The area of a square composed of four identical rectangles and a smaller square, 
    where the perimeter of each rectangle is 28. -/
theorem square_area_from_rectangles (l w : ℝ) : 
  (l + w ≥ 0) →  -- Ensure non-negative side length
  (2 * (l + w) = 28) →  -- Perimeter of rectangle
  (l + w) * (l + w) = 196 := by
  sorry

#check square_area_from_rectangles

end NUMINAMATH_CALUDE_square_area_from_rectangles_l3197_319705


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_l3197_319785

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

-- State the theorem
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a ∈ Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_in_range_l3197_319785


namespace NUMINAMATH_CALUDE_sons_age_l3197_319757

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 29 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 27 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3197_319757


namespace NUMINAMATH_CALUDE_second_reduction_percentage_store_price_reduction_l3197_319758

theorem second_reduction_percentage 
  (initial_reduction : Real) 
  (final_price_percentage : Real) : Real :=
  let price_after_first_reduction := 1 - initial_reduction
  let second_reduction := (price_after_first_reduction - final_price_percentage) / price_after_first_reduction
  14 / 100

-- The main theorem
theorem store_price_reduction 
  (initial_reduction : Real)
  (final_price_percentage : Real)
  (h1 : initial_reduction = 10 / 100)
  (h2 : final_price_percentage = 77.4 / 100) :
  second_reduction_percentage initial_reduction final_price_percentage = 14 / 100 := by
  sorry

end NUMINAMATH_CALUDE_second_reduction_percentage_store_price_reduction_l3197_319758


namespace NUMINAMATH_CALUDE_factors_of_135_l3197_319776

theorem factors_of_135 : Nat.card (Nat.divisors 135) = 8 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_135_l3197_319776


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3197_319748

theorem algebraic_expression_equality (x : ℝ) : 
  x^2 + 3*x + 5 = 7 → 3*x^2 + 9*x - 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3197_319748


namespace NUMINAMATH_CALUDE_power_of_negative_cube_l3197_319771

theorem power_of_negative_cube (x : ℝ) : (-4 * x^3)^2 = 16 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_cube_l3197_319771


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_l3197_319734

-- Define the centers of the circles
def A : ℝ × ℝ := (-2, 2)
def B : ℝ × ℝ := (3, 3)
def C : ℝ × ℝ := (10, 4)

-- Define the radii of the circles
def r_A : ℝ := 2
def r_B : ℝ := 3
def r_C : ℝ := 4

-- Define the distance between centers
def dist_AB : ℝ := r_A + r_B
def dist_BC : ℝ := r_B + r_C

-- Theorem statement
theorem area_of_triangle_ABC :
  let triangle_area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  triangle_area = 1 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_l3197_319734


namespace NUMINAMATH_CALUDE_system_solution_l3197_319709

theorem system_solution (x y : ℚ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : 
  (x + y) / 3 = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3197_319709


namespace NUMINAMATH_CALUDE_probability_threshold_min_probability_value_l3197_319726

/-- The probability that Alex and Dylan are on the same team given their card picks -/
def probability_same_team (a : ℕ) : ℚ :=
  let total_outcomes := (50 : ℚ) * 49 / 2
  let favorable_outcomes := ((a - 1 : ℚ) * (a - 2) / 2) + ((43 - a : ℚ) * (42 - a) / 2)
  favorable_outcomes / total_outcomes

/-- The minimum value of a for which the probability is at least 1/2 -/
def min_a : ℕ := 8

theorem probability_threshold :
  probability_same_team min_a ≥ 1/2 ∧
  ∀ a < min_a, probability_same_team a < 1/2 :=
sorry

theorem min_probability_value :
  probability_same_team min_a = 88/175 :=
sorry

end NUMINAMATH_CALUDE_probability_threshold_min_probability_value_l3197_319726


namespace NUMINAMATH_CALUDE_reciprocal_sum_range_l3197_319718

theorem reciprocal_sum_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 1 / y ≥ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 1 / a + 1 / b = 4 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_range_l3197_319718


namespace NUMINAMATH_CALUDE_yellow_raisins_amount_l3197_319760

theorem yellow_raisins_amount (yellow_raisins black_raisins total_raisins : ℝ) 
  (h1 : black_raisins = 0.4)
  (h2 : total_raisins = 0.7)
  (h3 : yellow_raisins + black_raisins = total_raisins) : 
  yellow_raisins = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_raisins_amount_l3197_319760


namespace NUMINAMATH_CALUDE_average_of_combined_results_l3197_319772

theorem average_of_combined_results (n₁ n₂ : ℕ) (avg₁ avg₂ : ℝ) 
  (h₁ : n₁ = 55) (h₂ : n₂ = 28) (h₃ : avg₁ = 28) (h₄ : avg₂ = 55) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = (55 * 28 + 28 * 55) / (55 + 28) := by
sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l3197_319772


namespace NUMINAMATH_CALUDE_reachable_cells_after_ten_moves_l3197_319731

-- Define the board size
def boardSize : ℕ := 21

-- Define the number of moves
def numMoves : ℕ := 10

-- Define a function to calculate the number of reachable cells
def reachableCells (moves : ℕ) : ℕ :=
  if moves % 2 = 0 then
    1 + 2 * moves * (moves + 1)
  else
    (moves + 1) ^ 2

-- Theorem statement
theorem reachable_cells_after_ten_moves :
  reachableCells numMoves = 121 := by
  sorry

end NUMINAMATH_CALUDE_reachable_cells_after_ten_moves_l3197_319731


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_4_l3197_319702

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The number of outcomes where the sum is 4 or less -/
def outcomes_sum_4_or_less : ℕ := 6

/-- The probability that the sum of two dice is greater than 4 -/
theorem prob_sum_greater_than_4 : 
  (total_outcomes - outcomes_sum_4_or_less : ℚ) / total_outcomes = 5 / 6 :=
sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_4_l3197_319702


namespace NUMINAMATH_CALUDE_square_area_l3197_319773

/-- Square in a coordinate plane --/
structure Square where
  B : ℝ × ℝ
  C : ℝ × ℝ
  E : ℝ × ℝ
  BC_is_side : True  -- Represents that BC is a side of the square
  E_on_line : True   -- Represents that E is on a line intersecting another vertex

/-- The area of the square ABCD is 4 --/
theorem square_area (s : Square) (h1 : s.B = (0, 0)) (h2 : s.C = (2, 0)) (h3 : s.E = (2, 1)) : 
  (s.C.1 - s.B.1) ^ 2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_square_area_l3197_319773


namespace NUMINAMATH_CALUDE_not_divisible_by_8_main_result_l3197_319729

def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

theorem not_divisible_by_8 (n : ℕ) (h : n = 456294604884) :
  ¬(8 ∣ n) ↔ ¬(8 ∣ last_three_digits n) :=
by
  sorry

theorem main_result : ¬(8 ∣ 456294604884) :=
by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_8_main_result_l3197_319729


namespace NUMINAMATH_CALUDE_second_player_wins_l3197_319774

/-- Represents the possible moves in the game -/
inductive Move where
  | two : Move
  | four : Move
  | five : Move

/-- Defines the game state -/
structure GameState where
  chips : Nat
  player_turn : Bool  -- True for first player, False for second player

/-- Determines if a position is winning for the current player -/
def is_winning_position (state : GameState) : Bool :=
  match state.chips % 7 with
  | 0 | 1 | 3 => false
  | _ => true

/-- Theorem stating that the second player has a winning strategy when starting with 2016 chips -/
theorem second_player_wins :
  let initial_state : GameState := { chips := 2016, player_turn := true }
  ¬(is_winning_position initial_state) := by
  sorry

end NUMINAMATH_CALUDE_second_player_wins_l3197_319774


namespace NUMINAMATH_CALUDE_track_circumference_is_720_l3197_319789

/-- Represents the circular track and the movement of A and B -/
structure CircularTrack where
  circumference : ℝ
  speed_A : ℝ
  speed_B : ℝ

/-- The conditions of the problem -/
def problem_conditions (track : CircularTrack) : Prop :=
  ∃ (t₁ t₂ : ℝ),
    t₁ > 0 ∧ t₂ > t₁ ∧
    track.speed_B * t₁ = 150 ∧
    track.speed_A * t₁ = track.circumference / 2 - 150 ∧
    track.speed_A * t₂ = track.circumference - 90 ∧
    track.speed_B * t₂ = track.circumference / 2 + 90

/-- The theorem stating that the track circumference is 720 yards -/
theorem track_circumference_is_720 (track : CircularTrack) :
  problem_conditions track → track.circumference = 720 := by
  sorry


end NUMINAMATH_CALUDE_track_circumference_is_720_l3197_319789


namespace NUMINAMATH_CALUDE_max_value_of_product_l3197_319778

theorem max_value_of_product (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a + b = 4) :
  (∀ x y : ℝ, x > 1 → y > 1 → x + y = 4 → (x - 1) * (y - 1) ≤ (a - 1) * (b - 1)) →
  (a - 1) * (b - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_product_l3197_319778


namespace NUMINAMATH_CALUDE_prism_configuration_impossible_l3197_319788

/-- A rectangular prism in 3D space -/
structure RectangularPrism where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  z_min : ℝ
  z_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max
  h_z : z_min < z_max

/-- Two prisms intersect if their projections overlap on all axes -/
def intersects (p q : RectangularPrism) : Prop :=
  (p.x_min < q.x_max ∧ q.x_min < p.x_max) ∧
  (p.y_min < q.y_max ∧ q.y_min < p.y_max) ∧
  (p.z_min < q.z_max ∧ q.z_min < p.z_max)

/-- A configuration of 12 prisms satisfying the problem conditions -/
structure PrismConfiguration where
  prisms : Fin 12 → RectangularPrism
  h_intersects : ∀ i j : Fin 12, i ≠ j → 
    (i.val + 1) % 12 ≠ j.val ∧ (i.val + 11) % 12 ≠ j.val → 
    intersects (prisms i) (prisms j)
  h_non_intersects : ∀ i : Fin 12, 
    ¬intersects (prisms i) (prisms ⟨(i.val + 1) % 12, sorry⟩) ∧
    ¬intersects (prisms i) (prisms ⟨(i.val + 11) % 12, sorry⟩)

/-- The main theorem stating the impossibility of such a configuration -/
theorem prism_configuration_impossible : ¬∃ (config : PrismConfiguration), True :=
  sorry

end NUMINAMATH_CALUDE_prism_configuration_impossible_l3197_319788


namespace NUMINAMATH_CALUDE_intersection_nonempty_condition_l3197_319715

theorem intersection_nonempty_condition (k : ℝ) : 
  let A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
  let B : Set ℝ := {x | x - k ≥ 0}
  (A ∩ B).Nonempty → k ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_condition_l3197_319715


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l3197_319798

theorem solution_set_abs_inequality :
  {x : ℝ | 1 < |x + 2| ∧ |x + 2| < 5} = {x : ℝ | -7 < x ∧ x < -3} ∪ {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l3197_319798


namespace NUMINAMATH_CALUDE_age_difference_l3197_319703

/-- Given that Sachin is 14 years old and the ratio of Sachin's age to Rahul's age is 7:9,
    prove that the difference between Rahul's age and Sachin's age is 4 years. -/
theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 14 → 
  sachin_age * 9 = rahul_age * 7 →
  rahul_age - sachin_age = 4 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l3197_319703


namespace NUMINAMATH_CALUDE_intercepts_sum_l3197_319712

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the specific parabola y = 3x^2 - 9x + 4 -/
def parabola : QuadraticFunction :=
  { a := 3, b := -9, c := 4 }

/-- The y-intercept of the parabola -/
def y_intercept : Point :=
  { x := 0, y := parabola.c }

/-- Theorem stating that the sum of the y-intercept's y-coordinate and the x-coordinates of the two x-intercepts equals 19/3 -/
theorem intercepts_sum (e f : ℝ) 
  (h1 : parabola.a * e^2 + parabola.b * e + parabola.c = 0)
  (h2 : parabola.a * f^2 + parabola.b * f + parabola.c = 0)
  (h3 : e ≠ f) : 
  y_intercept.y + e + f = 19/3 := by
  sorry

end NUMINAMATH_CALUDE_intercepts_sum_l3197_319712


namespace NUMINAMATH_CALUDE_shaded_to_white_area_ratio_l3197_319764

theorem shaded_to_white_area_ratio : 
  ∀ (quarter_shaded_triangles quarter_white_triangles : ℕ) 
    (total_quarters : ℕ) 
    (shaded_area white_area : ℝ),
  quarter_shaded_triangles = 5 →
  quarter_white_triangles = 3 →
  total_quarters = 4 →
  shaded_area = (quarter_shaded_triangles * total_quarters : ℝ) →
  white_area = (quarter_white_triangles * total_quarters : ℝ) →
  shaded_area / white_area = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_shaded_to_white_area_ratio_l3197_319764


namespace NUMINAMATH_CALUDE_prism_division_theorem_l3197_319714

/-- Represents a rectangular prism -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c

/-- Represents the division of a rectangular prism by three planes -/
structure PrismDivision (T : RectangularPrism) where
  x : ℝ
  y : ℝ
  z : ℝ
  x_bounds : 0 < x ∧ x < T.a
  y_bounds : 0 < y ∧ y < T.b
  z_bounds : 0 < z ∧ z < T.c

/-- The theorem to be proved -/
theorem prism_division_theorem (T : RectangularPrism) (div : PrismDivision T) :
  let vol_black := div.x * div.y * div.z + 
                   div.x * (T.b - div.y) * (T.c - div.z) + 
                   (T.a - div.x) * div.y * (T.c - div.z) + 
                   (T.a - div.x) * (T.b - div.y) * div.z
  let vol_white := (T.a - div.x) * (T.b - div.y) * (T.c - div.z) + 
                   (T.a - div.x) * div.y * div.z + 
                   div.x * (T.b - div.y) * div.z + 
                   div.x * div.y * (T.c - div.z)
  vol_black = vol_white → 
  div.x = T.a / 2 ∨ div.y = T.b / 2 ∨ div.z = T.c / 2 := by
  sorry

end NUMINAMATH_CALUDE_prism_division_theorem_l3197_319714


namespace NUMINAMATH_CALUDE_inscribed_circle_tangent_difference_l3197_319713

/-- A cyclic quadrilateral with an inscribed circle -/
structure CyclicQuadrilateralWithInscribedCircle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Tangent points divide sides
  t_a : ℝ
  t_b : ℝ
  t_c : ℝ
  t_d : ℝ
  -- Conditions
  side_sum : a + b = t_a + t_b
  side_sum' : b + c = t_b + t_c
  side_sum'' : c + d = t_c + t_d
  side_sum''' : d + a = t_d + t_a

/-- The main theorem -/
theorem inscribed_circle_tangent_difference 
  (q : CyclicQuadrilateralWithInscribedCircle)
  (h1 : q.a = 70)
  (h2 : q.b = 90)
  (h3 : q.c = 130)
  (h4 : q.d = 110) :
  |q.t_c - (q.c - q.t_c)| = 13 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_tangent_difference_l3197_319713


namespace NUMINAMATH_CALUDE_periodic_trig_function_l3197_319711

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx - β), where α, β, a, and b are non-zero real numbers,
    if f(2016) = -1, then f(2017) = 1 -/
theorem periodic_trig_function (α β a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x - β)
  f 2016 = -1 → f 2017 = 1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_trig_function_l3197_319711


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l3197_319793

/-- Represents the dimensions of a rectangular parallelepiped -/
structure Cuboid where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular parallelepiped -/
def surface_area (c : Cuboid) : ℝ :=
  2 * (c.width * c.length + c.width * c.height + c.length * c.height)

/-- Theorem stating that the surface area of a cuboid with given dimensions is 340 cm² -/
theorem cuboid_surface_area :
  let c : Cuboid := ⟨8, 5, 10⟩
  surface_area c = 340 := by sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l3197_319793


namespace NUMINAMATH_CALUDE_expression_is_perfect_square_l3197_319737

/-- 
Given real numbers a and b, prove that the expression 
x^2 - 4bx + 4ab + p^2 - 2px is a perfect square when p = a - b
-/
theorem expression_is_perfect_square (a b x : ℝ) : 
  ∃ k : ℝ, x^2 - 4*b*x + 4*a*b + (a - b)^2 - 2*(a - b)*x = k^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_is_perfect_square_l3197_319737


namespace NUMINAMATH_CALUDE_brothers_ages_l3197_319730

theorem brothers_ages (a b c : ℕ+) :
  a * b * c = 36 ∧ 
  a + b + c = 13 ∧ 
  (a ≤ b ∧ b ≤ c) ∧
  (b < c ∨ a < b) →
  a = 2 ∧ b = 2 ∧ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_brothers_ages_l3197_319730


namespace NUMINAMATH_CALUDE_largest_value_l3197_319720

theorem largest_value (a b : ℝ) 
  (ha : 0 < a) (ha1 : a < 1) 
  (hb : 0 < b) (hb1 : b < 1) 
  (hab : a ≠ b) : 
  a + b ≥ 2 * Real.sqrt (a * b) ∧ a + b ≥ (a^2 + b^2) / (2 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l3197_319720


namespace NUMINAMATH_CALUDE_sphere_volume_constant_l3197_319765

theorem sphere_volume_constant (cube_side : Real) (K : Real) : 
  cube_side = 3 →
  (4 / 3 * Real.pi * (((6 * cube_side^2) / (4 * Real.pi))^(3/2))) = K * Real.sqrt 6 / Real.sqrt Real.pi →
  K = 54 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_constant_l3197_319765


namespace NUMINAMATH_CALUDE_andrews_cookie_expenditure_l3197_319739

/-- The number of days in May -/
def days_in_may : ℕ := 31

/-- The number of cookies Andrew buys each day -/
def cookies_per_day : ℕ := 3

/-- The cost of each cookie in dollars -/
def cost_per_cookie : ℕ := 15

/-- The total amount Andrew spent on cookies in May -/
def total_spent : ℕ := days_in_may * cookies_per_day * cost_per_cookie

/-- Theorem stating that Andrew spent 1395 dollars on cookies in May -/
theorem andrews_cookie_expenditure : total_spent = 1395 := by
  sorry

end NUMINAMATH_CALUDE_andrews_cookie_expenditure_l3197_319739


namespace NUMINAMATH_CALUDE_roses_in_vase_l3197_319777

/-- The number of roses in a vase after adding more roses -/
def total_roses (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: Given 10 initial roses and 8 added roses, the total is 18 -/
theorem roses_in_vase : total_roses 10 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_roses_in_vase_l3197_319777


namespace NUMINAMATH_CALUDE_prism_height_l3197_319796

theorem prism_height (ab ac : ℝ) (volume : ℝ) (h1 : ab = ac) (h2 : ab = Real.sqrt 2) (h3 : volume = 3.0000000000000004) :
  let base_area := (1 / 2) * ab * ac
  let height := volume / base_area
  height = 3.0000000000000004 := by
sorry

end NUMINAMATH_CALUDE_prism_height_l3197_319796


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3197_319756

theorem arithmetic_sequence_length (a d last : ℕ) (h : last = a + (n - 1) * d) : 
  a = 2 → d = 5 → last = 2507 → n = 502 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3197_319756


namespace NUMINAMATH_CALUDE_sum_of_binary_numbers_l3197_319759

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 101₂ -/
def binary1 : List Bool := [true, false, true]

/-- The second binary number 110₂ -/
def binary2 : List Bool := [false, true, true]

theorem sum_of_binary_numbers :
  binary_to_decimal binary1 + binary_to_decimal binary2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_binary_numbers_l3197_319759


namespace NUMINAMATH_CALUDE_decreasing_function_implies_b_geq_4_l3197_319783

-- Define the function y
def y (x b : ℝ) : ℝ := x^3 - 3*b*x + 1

-- State the theorem
theorem decreasing_function_implies_b_geq_4 :
  ∀ b : ℝ, (∀ x ∈ Set.Ioo 1 2, ∀ h > 0, x + h ∈ Set.Ioo 1 2 → y (x + h) b < y x b) →
  b ≥ 4 := by sorry

end NUMINAMATH_CALUDE_decreasing_function_implies_b_geq_4_l3197_319783


namespace NUMINAMATH_CALUDE_hyperbola_quadrilateral_area_ratio_max_l3197_319763

theorem hyperbola_quadrilateral_area_ratio_max
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (S₁ : ℝ) (hS₁ : S₁ = 2 * a * b)
  (S₂ : ℝ) (hS₂ : S₂ = 2 * (a^2 + b^2)) :
  (S₁ / S₂) ≤ (1 / 2) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_quadrilateral_area_ratio_max_l3197_319763


namespace NUMINAMATH_CALUDE_jill_clothing_expenditure_l3197_319747

theorem jill_clothing_expenditure 
  (total : ℝ) 
  (food_percent : ℝ) 
  (other_percent : ℝ) 
  (clothing_tax_rate : ℝ) 
  (other_tax_rate : ℝ) 
  (total_tax_rate : ℝ) 
  (h1 : food_percent = 0.2)
  (h2 : other_percent = 0.3)
  (h3 : clothing_tax_rate = 0.04)
  (h4 : other_tax_rate = 0.1)
  (h5 : total_tax_rate = 0.05)
  (h6 : clothing_tax_rate * (1 - food_percent - other_percent) * total + 
        other_tax_rate * other_percent * total = total_tax_rate * total) :
  1 - food_percent - other_percent = 0.5 := by
sorry

end NUMINAMATH_CALUDE_jill_clothing_expenditure_l3197_319747


namespace NUMINAMATH_CALUDE_man_wage_is_350_l3197_319728

/-- The daily wage of a man -/
def man_wage : ℝ := 350

/-- The daily wage of a woman -/
def woman_wage : ℝ := 200

/-- The total number of men -/
def num_men : ℕ := 24

/-- The total number of women -/
def num_women : ℕ := 16

/-- The total daily wages -/
def total_wages : ℝ := 11600

theorem man_wage_is_350 :
  (num_men * man_wage + num_women * woman_wage = total_wages) ∧
  ((num_men / 2) * man_wage + 37 * woman_wage = total_wages) →
  man_wage = 350 := by
  sorry

end NUMINAMATH_CALUDE_man_wage_is_350_l3197_319728


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_and_product_l3197_319750

theorem largest_gcd_of_sum_and_product (x y : ℕ+) 
  (sum_eq : x + y = 1130)
  (prod_eq : x * y = 100000) :
  ∃ (a b : ℕ+), a + b = 1130 ∧ a * b = 100000 ∧ 
    ∀ (c d : ℕ+), c + d = 1130 → c * d = 100000 → Nat.gcd c d ≤ Nat.gcd a b ∧ Nat.gcd a b = 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_and_product_l3197_319750


namespace NUMINAMATH_CALUDE_unique_prime_triple_l3197_319743

theorem unique_prime_triple : ∃! (p q r : ℕ), 
  Prime p ∧ Prime q ∧ Prime r ∧
  p > q ∧ q > r ∧
  Prime (p - q) ∧ Prime (p - r) ∧ Prime (q - r) ∧
  p = 7 ∧ q = 5 ∧ r = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l3197_319743


namespace NUMINAMATH_CALUDE_concentric_circles_radii_inequality_l3197_319786

theorem concentric_circles_radii_inequality (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a < b) (h5 : b < c) : 
  b + a ≠ c + b := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_inequality_l3197_319786


namespace NUMINAMATH_CALUDE_adjacent_supplementary_angles_l3197_319732

theorem adjacent_supplementary_angles (angle_AOB angle_BOC : ℝ) : 
  angle_AOB + angle_BOC = 180 →
  angle_AOB = angle_BOC + 18 →
  angle_AOB = 99 := by
sorry

end NUMINAMATH_CALUDE_adjacent_supplementary_angles_l3197_319732


namespace NUMINAMATH_CALUDE_boys_running_speed_l3197_319752

/-- Given a square field with side length 60 meters and a boy who runs around it in 96 seconds,
    prove that the boy's speed is 9 km/hr. -/
theorem boys_running_speed (side_length : ℝ) (time : ℝ) (speed : ℝ) : 
  side_length = 60 →
  time = 96 →
  speed = (4 * side_length) / time * 3.6 →
  speed = 9 := by sorry

end NUMINAMATH_CALUDE_boys_running_speed_l3197_319752


namespace NUMINAMATH_CALUDE_domain_shift_l3197_319791

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 0 1

-- State the theorem
theorem domain_shift :
  (∀ x, f x ≠ 0 → x ∈ domain_f) →
  (∀ x, f (x + 2) ≠ 0 → x ∈ Set.Icc (-2) (-1)) :=
sorry

end NUMINAMATH_CALUDE_domain_shift_l3197_319791


namespace NUMINAMATH_CALUDE_complex_exponential_form_l3197_319736

/-- For the complex number z = 1 + i√3, when expressed in the form re^(iθ), θ = π/3 -/
theorem complex_exponential_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_form_l3197_319736


namespace NUMINAMATH_CALUDE_geometric_sum_proof_l3197_319723

theorem geometric_sum_proof : 
  let a₁ : ℚ := 3/4
  let r : ℚ := 3/4
  let n : ℕ := 10
  let S := a₁ * (1 - r^n) / (1 - r)
  S = 2968581/1048576 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_proof_l3197_319723


namespace NUMINAMATH_CALUDE_unique_prime_sum_difference_l3197_319745

theorem unique_prime_sum_difference : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  (∃ a b : ℕ, Nat.Prime a ∧ Nat.Prime b ∧ p = a + b) ∧
  (∃ c d : ℕ, Nat.Prime c ∧ Nat.Prime d ∧ p = c - d) ∧
  p = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_sum_difference_l3197_319745


namespace NUMINAMATH_CALUDE_license_plate_difference_l3197_319735

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits -/
def digit_size : ℕ := 10

/-- The number of letters in a California license plate -/
def california_letters : ℕ := 4

/-- The number of digits in a California license plate -/
def california_digits : ℕ := 3

/-- The number of letters in a Texas license plate -/
def texas_letters : ℕ := 3

/-- The number of digits in a Texas license plate -/
def texas_digits : ℕ := 4

/-- The number of possible California license plates -/
def california_plates : ℕ := alphabet_size ^ california_letters * digit_size ^ california_digits

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := alphabet_size ^ texas_letters * digit_size ^ texas_digits

/-- The difference in the number of possible license plates between California and Texas -/
theorem license_plate_difference : california_plates - texas_plates = 281216000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l3197_319735


namespace NUMINAMATH_CALUDE_evaluate_expression_l3197_319741

-- Define x in terms of b
def x (b : ℝ) : ℝ := b + 9

-- Theorem to prove
theorem evaluate_expression (b : ℝ) : x b - b + 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3197_319741


namespace NUMINAMATH_CALUDE_fraction_division_specific_fraction_division_l3197_319762

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem specific_fraction_division :
  (3 : ℚ) / 7 / ((4 : ℚ) / 5) = 15 / 28 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_specific_fraction_division_l3197_319762


namespace NUMINAMATH_CALUDE_average_thirteen_l3197_319701

theorem average_thirteen (x : ℝ) : 
  (6 + 16 + 8 + x) / 4 = 13 → x = 22 := by
sorry

end NUMINAMATH_CALUDE_average_thirteen_l3197_319701


namespace NUMINAMATH_CALUDE_AAA_not_congruence_l3197_319707

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles in radians

-- Define triangle congruence
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define AAA condition
def AAA (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem: AAA does not imply congruence
theorem AAA_not_congruence :
  ∃ t1 t2 : Triangle, AAA t1 t2 ∧ ¬(congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_AAA_not_congruence_l3197_319707


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3197_319761

theorem inequality_equivalence (x : ℝ) : 
  (-1/3 : ℝ) ≤ (5-x)/2 ∧ (5-x)/2 < (1/3 : ℝ) ↔ (13/3 : ℝ) < x ∧ x ≤ (17/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3197_319761


namespace NUMINAMATH_CALUDE_pen_collection_l3197_319767

theorem pen_collection (initial_pens : ℕ) (received_pens : ℕ) (given_away : ℕ) : 
  initial_pens = 5 → received_pens = 20 → given_away = 10 → 
  ((initial_pens + received_pens) * 2 - given_away) = 40 := by
  sorry

end NUMINAMATH_CALUDE_pen_collection_l3197_319767


namespace NUMINAMATH_CALUDE_complex_fourth_power_l3197_319781

theorem complex_fourth_power (z : ℂ) : z = Complex.I * Real.sqrt 2 → z^4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l3197_319781


namespace NUMINAMATH_CALUDE_toms_shirt_purchase_cost_l3197_319797

/-- The total cost of Tom's shirt purchase --/
def totalCost (numFandoms : ℕ) (shirtsPerFandom : ℕ) (originalPrice : ℚ) (discountPercentage : ℚ) (taxRate : ℚ) : ℚ :=
  let totalShirts := numFandoms * shirtsPerFandom
  let discountAmount := originalPrice * discountPercentage
  let discountedPrice := originalPrice - discountAmount
  let subtotal := totalShirts * discountedPrice
  let taxAmount := subtotal * taxRate
  subtotal + taxAmount

/-- Theorem stating that Tom's total cost is $264 --/
theorem toms_shirt_purchase_cost :
  totalCost 4 5 15 0.2 0.1 = 264 := by
  sorry

end NUMINAMATH_CALUDE_toms_shirt_purchase_cost_l3197_319797


namespace NUMINAMATH_CALUDE_angle_bisector_property_l3197_319706

theorem angle_bisector_property (x : ℝ) : 
  x > 0 ∧ x < 180 →
  x / 2 = (180 - x) / 3 →
  x = 72 := by
sorry

end NUMINAMATH_CALUDE_angle_bisector_property_l3197_319706


namespace NUMINAMATH_CALUDE_estimate_shaded_area_l3197_319700

/-- Estimates the area of a shaded region within a square using Monte Carlo method. -/
theorem estimate_shaded_area (side_length : ℝ) (total_points : ℕ) (shaded_points : ℕ) : 
  side_length = 6 →
  total_points = 800 →
  shaded_points = 200 →
  (shaded_points : ℝ) / (total_points : ℝ) * side_length^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_estimate_shaded_area_l3197_319700


namespace NUMINAMATH_CALUDE_large_cube_probabilities_l3197_319708

/-- Represents a large cube composed of 27 smaller dice -/
structure LargeCube where
  dice : Fin 27 → Die

/-- Represents a single die -/
structure Die where
  faces : Fin 6 → Nat

/-- Represents the position of a die in the large cube -/
inductive Position
  | FaceCenter
  | Edge
  | Corner

/-- Returns the position of a die given its index in the large cube -/
def diePosition (i : Fin 27) : Position := sorry

/-- Returns the probability of a specific face showing based on the die's position -/
def faceProbability (p : Position) (face : Nat) : ℚ := sorry

/-- Calculates the probability of exactly 25 sixes showing on the surface -/
def probExactly25Sixes (c : LargeCube) : ℚ := sorry

/-- Calculates the probability of at least one 'one' showing on the surface -/
def probAtLeastOne1 (c : LargeCube) : ℚ := sorry

/-- Calculates the expected number of sixes showing on the surface -/
def expectedSixes (c : LargeCube) : ℚ := sorry

/-- Calculates the expected sum of the numbers showing on the surface -/
def expectedSum (c : LargeCube) : ℚ := sorry

/-- Calculates the expected number of distinct digits appearing on the surface -/
def expectedDistinctDigits (c : LargeCube) : ℚ := sorry

theorem large_cube_probabilities (c : LargeCube) :
  probExactly25Sixes c = 31 / (2^13 * 3^18) ∧
  probAtLeastOne1 c = 1 - (5^6 / (2^2 * 3^18)) ∧
  expectedSixes c = 9 ∧
  expectedSum c = 189 ∧
  expectedDistinctDigits c = 6 * (1 - (5^6 / (2^2 * 3^18))) := by
  sorry

end NUMINAMATH_CALUDE_large_cube_probabilities_l3197_319708


namespace NUMINAMATH_CALUDE_max_value_theorem_l3197_319775

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  a + b^3 + c^4 ≤ 2 ∧ ∃ (a' b' c' : ℝ), a' + b'^3 + c'^4 = 2 ∧ 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3197_319775
