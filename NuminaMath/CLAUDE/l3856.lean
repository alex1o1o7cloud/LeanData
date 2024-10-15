import Mathlib

namespace NUMINAMATH_CALUDE_a_10_equals_1000_l3856_385665

def a (n : ℕ) : ℕ :=
  let first_odd := 2 * n - 1
  let last_odd := first_odd + 2 * (n - 1)
  n * (first_odd + last_odd) / 2

theorem a_10_equals_1000 : a 10 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_a_10_equals_1000_l3856_385665


namespace NUMINAMATH_CALUDE_characterize_positive_product_set_l3856_385603

def positive_product_set : Set ℤ :=
  {a : ℤ | (5 + a) * (3 - a) > 0}

theorem characterize_positive_product_set :
  positive_product_set = {-4, -3, -2, -1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_characterize_positive_product_set_l3856_385603


namespace NUMINAMATH_CALUDE_balls_distribution_proof_l3856_385696

def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

theorem balls_distribution_proof :
  distribute_balls 10 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_balls_distribution_proof_l3856_385696


namespace NUMINAMATH_CALUDE_bankers_gain_calculation_l3856_385630

/-- Banker's gain calculation -/
theorem bankers_gain_calculation (P TD : ℚ) (h1 : P = 576) (h2 : TD = 96) :
  TD^2 / P = 16 := by sorry

end NUMINAMATH_CALUDE_bankers_gain_calculation_l3856_385630


namespace NUMINAMATH_CALUDE_triangle_is_isosceles_triangle_area_l3856_385656

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isIsosceles (t : Triangle) : Prop :=
  t.b * Real.cos t.C = t.a * (Real.cos t.B)^2 + t.b * Real.cos t.A * Real.cos t.B

def hasSpecificProperties (t : Triangle) : Prop :=
  isIsosceles t ∧ Real.cos t.A = 7/8 ∧ t.a + t.b + t.c = 5

-- State the theorems
theorem triangle_is_isosceles (t : Triangle) (h : isIsosceles t) : 
  t.B = t.C := by sorry

theorem triangle_area (t : Triangle) (h : hasSpecificProperties t) :
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 15 / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_is_isosceles_triangle_area_l3856_385656


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l3856_385673

theorem cube_root_unity_sum (ω : ℂ) : 
  ω^3 = 1 → ((-1 + Complex.I * Real.sqrt 3) / 2)^8 + ((-1 - Complex.I * Real.sqrt 3) / 2)^8 = -1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l3856_385673


namespace NUMINAMATH_CALUDE_BC_length_l3856_385682

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A, B, C, B', C', and D
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def C : ℝ × ℝ := sorry
def B' : ℝ × ℝ := sorry
def C' : ℝ × ℝ := sorry
def D : ℝ × ℝ := sorry

-- Define the conditions
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω
axiom C_on_ω : C ∈ ω
axiom BC_is_diameter : sorry -- BC is a diameter of ω
axiom B'C'_parallel_BC : sorry -- B'C' is parallel to BC
axiom B'C'_tangent_ω : sorry -- B'C' is tangent to ω at D
axiom B'D_length : dist B' D = 4
axiom C'D_length : dist C' D = 6

-- Define the theorem
theorem BC_length : dist B C = 24/5 := by sorry

end NUMINAMATH_CALUDE_BC_length_l3856_385682


namespace NUMINAMATH_CALUDE_rival_awards_l3856_385618

/-- Given Scott won 4 awards, Jessie won 3 times as many awards as Scott,
    and the rival won twice as many awards as Jessie,
    prove that the rival won 24 awards. -/
theorem rival_awards (scott_awards : ℕ) (jessie_awards : ℕ) (rival_awards : ℕ)
    (h1 : scott_awards = 4)
    (h2 : jessie_awards = 3 * scott_awards)
    (h3 : rival_awards = 2 * jessie_awards) :
  rival_awards = 24 := by
  sorry

end NUMINAMATH_CALUDE_rival_awards_l3856_385618


namespace NUMINAMATH_CALUDE_expression_value_l3856_385609

theorem expression_value : 
  let a := 2015
  let b := 2016
  (a^3 - 3*a^2*b + 5*a*b^2 - b^3 + 4) / (a*b) = 4032 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3856_385609


namespace NUMINAMATH_CALUDE_larger_number_is_23_l3856_385649

theorem larger_number_is_23 (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 40) :
  x = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_23_l3856_385649


namespace NUMINAMATH_CALUDE_cubic_inequality_false_l3856_385687

theorem cubic_inequality_false (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  ¬(a^3 > b^3) := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_false_l3856_385687


namespace NUMINAMATH_CALUDE_sixDigitPermutationsCount_l3856_385628

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The factorial of a natural number -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of permutations of n elements -/
def permutations (n : ℕ) : ℕ := factorial n

/-- The number of ways to arrange k objects in n positions -/
def arrangements (n k : ℕ) : ℕ := (permutations n) / (factorial (n - k))

/-- The number of 6-digit permutations using x, y, and z with given conditions -/
def sixDigitPermutations : ℕ :=
  let xTwice := choose 6 2 * arrangements 4 2  -- x appears twice
  let xThrice := choose 6 3 * arrangements 3 1 -- x appears thrice
  let yOnce := choose 4 1                      -- y appears once
  let yThrice := choose 4 3                    -- y appears thrice
  let zTwice := 1                              -- z appears twice (only one way)
  (xTwice + xThrice) * (yOnce + yThrice) * zTwice

theorem sixDigitPermutationsCount : sixDigitPermutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_sixDigitPermutationsCount_l3856_385628


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l3856_385689

theorem manuscript_typing_cost : 
  let total_pages : ℕ := 100
  let pages_revised_once : ℕ := 30
  let pages_revised_twice : ℕ := 20
  let pages_not_revised : ℕ := total_pages - pages_revised_once - pages_revised_twice
  let initial_typing_cost : ℕ := 5
  let revision_cost : ℕ := 3
  let total_cost : ℕ := 
    total_pages * initial_typing_cost + 
    pages_revised_once * revision_cost + 
    pages_revised_twice * revision_cost * 2
  total_cost = 710 := by
sorry

end NUMINAMATH_CALUDE_manuscript_typing_cost_l3856_385689


namespace NUMINAMATH_CALUDE_intersection_dot_product_l3856_385666

/-- Given a line Ax + By + C = 0 intersecting the circle x^2 + y^2 = 9 at points P and Q,
    where A^2, C^2, and B^2 form an arithmetic sequence, prove that OP · PQ = -1 -/
theorem intersection_dot_product 
  (A B C : ℝ) 
  (P Q : ℝ × ℝ) 
  (h_line : ∀ x y, A * x + B * y + C = 0 ↔ (x, y) = P ∨ (x, y) = Q)
  (h_circle : P.1^2 + P.2^2 = 9 ∧ Q.1^2 + Q.2^2 = 9)
  (h_arithmetic : 2 * C^2 = A^2 + B^2)
  (h_distinct : P ≠ Q) :
  (P.1 * (Q.1 - P.1) + P.2 * (Q.2 - P.2) : ℝ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l3856_385666


namespace NUMINAMATH_CALUDE_last_three_digits_of_3_to_1000_l3856_385676

theorem last_three_digits_of_3_to_1000 (h : 3^200 ≡ 1 [ZMOD 500]) :
  3^1000 ≡ 1 [ZMOD 1000] :=
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_3_to_1000_l3856_385676


namespace NUMINAMATH_CALUDE_commute_time_sum_of_squares_l3856_385605

theorem commute_time_sum_of_squares 
  (x y : ℝ) 
  (avg_eq : (x + y + 10 + 11 + 9) / 5 = 10) 
  (var_eq : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) : 
  x^2 + y^2 = 208 := by
sorry

end NUMINAMATH_CALUDE_commute_time_sum_of_squares_l3856_385605


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3856_385691

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3856_385691


namespace NUMINAMATH_CALUDE_crowdfunding_highest_level_l3856_385677

/-- Represents the financial backing levels and backers for a crowdfunding campaign -/
structure CrowdfundingCampaign where
  lowest_level : ℕ
  second_level : ℕ
  highest_level : ℕ
  lowest_backers : ℕ
  second_backers : ℕ
  highest_backers : ℕ

/-- Theorem stating the conditions and the result to be proven -/
theorem crowdfunding_highest_level 
  (campaign : CrowdfundingCampaign)
  (level_relation : campaign.second_level = 10 * campaign.lowest_level ∧ 
                    campaign.highest_level = 10 * campaign.second_level)
  (backers : campaign.lowest_backers = 10 ∧ 
             campaign.second_backers = 3 ∧ 
             campaign.highest_backers = 2)
  (total_raised : campaign.lowest_backers * campaign.lowest_level + 
                  campaign.second_backers * campaign.second_level + 
                  campaign.highest_backers * campaign.highest_level = 12000) :
  campaign.highest_level = 5000 := by
  sorry


end NUMINAMATH_CALUDE_crowdfunding_highest_level_l3856_385677


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3856_385642

theorem solve_linear_equation :
  ∀ x : ℝ, 7 - 2 * x = 15 → x = -4 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3856_385642


namespace NUMINAMATH_CALUDE_f_derivative_at_negative_one_l3856_385637

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + 6

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

theorem f_derivative_at_negative_one (a b : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_negative_one_l3856_385637


namespace NUMINAMATH_CALUDE_quadratic_less_than_sqrt_l3856_385622

theorem quadratic_less_than_sqrt (x : ℝ) :
  x^2 - 3*x + 2 < Real.sqrt (x + 4) ↔ 1 < x ∧ x < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_less_than_sqrt_l3856_385622


namespace NUMINAMATH_CALUDE_alfonso_work_weeks_l3856_385694

def hourly_rate : ℝ := 6
def monday_hours : ℝ := 2
def tuesday_hours : ℝ := 3
def wednesday_hours : ℝ := 4
def thursday_hours : ℝ := 2
def friday_hours : ℝ := 3
def helmet_cost : ℝ := 340
def gloves_cost : ℝ := 45
def current_savings : ℝ := 40
def miscellaneous_expenses : ℝ := 20

def weekly_hours : ℝ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
def weekly_earnings : ℝ := weekly_hours * hourly_rate
def total_cost : ℝ := helmet_cost + gloves_cost + miscellaneous_expenses
def additional_earnings_needed : ℝ := total_cost - current_savings

theorem alfonso_work_weeks : 
  ∃ n : ℕ, n * weekly_earnings ≥ additional_earnings_needed ∧ 
           (n - 1) * weekly_earnings < additional_earnings_needed ∧
           n = 5 :=
sorry

end NUMINAMATH_CALUDE_alfonso_work_weeks_l3856_385694


namespace NUMINAMATH_CALUDE_range_of_a_l3856_385613

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0 ∧ x^2 - 2*(a + 7)*x + 5 ≤ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a ↔ -4 ≤ a ∧ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3856_385613


namespace NUMINAMATH_CALUDE_max_points_in_plane_max_points_in_space_l3856_385664

/-- A point in a Euclidean space -/
structure Point (n : Nat) where
  coords : Fin n → ℝ

/-- Checks if three points form an obtuse angle -/
def is_obtuse_angle (n : Nat) (p1 p2 p3 : Point n) : Prop :=
  sorry -- Definition of obtuse angle check

/-- A configuration of points in a Euclidean space -/
structure PointConfiguration (n : Nat) where
  dim : Nat -- dimension of the space (2 for plane, 3 for space)
  points : Fin n → Point dim
  no_obtuse_angles : ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    ¬ is_obtuse_angle dim (points i) (points j) (points k)

/-- The maximum number of points in a plane configuration without obtuse angles -/
theorem max_points_in_plane :
  (∃ (c : PointConfiguration 4), c.dim = 2) ∧
  (∀ (n : Nat), n > 4 → ¬ ∃ (c : PointConfiguration n), c.dim = 2) :=
sorry

/-- The maximum number of points in a space configuration without obtuse angles -/
theorem max_points_in_space :
  (∃ (c : PointConfiguration 8), c.dim = 3) ∧
  (∀ (n : Nat), n > 8 → ¬ ∃ (c : PointConfiguration n), c.dim = 3) :=
sorry

end NUMINAMATH_CALUDE_max_points_in_plane_max_points_in_space_l3856_385664


namespace NUMINAMATH_CALUDE_tailor_buttons_count_l3856_385667

/-- The number of buttons purchased by a tailor -/
theorem tailor_buttons_count : 
  let green : ℕ := 90
  let yellow : ℕ := green + 10
  let blue : ℕ := green - 5
  let red : ℕ := 2 * (yellow + blue)
  green + yellow + blue + red = 645 := by sorry

end NUMINAMATH_CALUDE_tailor_buttons_count_l3856_385667


namespace NUMINAMATH_CALUDE_largest_area_triangle_l3856_385695

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Checks if a point is an internal point of a line segment -/
def isInternalPoint (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := sorry

/-- Checks if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop := sorry

/-- Calculates the area of a triangle -/
def triangleArea (T : Triangle) : ℝ := sorry

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Represents an arc of a circle -/
structure Arc :=
  (circle : Circle)
  (startAngle endAngle : ℝ)

/-- Finds the intersection point of two circles -/
def circleIntersection (c1 c2 : Circle) : Option (ℝ × ℝ) := sorry

/-- Calculates the distance between two points -/
def distance (P1 P2 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem largest_area_triangle
  (A₀B₀C₀ : Triangle)
  (A'B'C' : Triangle)
  (k_a k_c : Circle)
  (i_a i_c : Arc) :
  ∀ (ABC : Triangle),
    isInternalPoint A₀B₀C₀.C ABC.A ABC.B →
    isInternalPoint A₀B₀C₀.A ABC.B ABC.C →
    isInternalPoint A₀B₀C₀.B ABC.C ABC.A →
    areSimilar ABC A'B'C' →
    (∃ (M : ℝ × ℝ), circleIntersection k_a k_c = some M) →
    (∀ (ABC' : Triangle),
      isInternalPoint A₀B₀C₀.C ABC'.A ABC'.B →
      isInternalPoint A₀B₀C₀.A ABC'.B ABC'.C →
      isInternalPoint A₀B₀C₀.B ABC'.C ABC'.A →
      areSimilar ABC' A'B'C' →
      (∃ (M' : ℝ × ℝ), circleIntersection k_a k_c = some M') →
      distance M ABC.C + distance M ABC.A ≥ distance M' ABC'.C + distance M' ABC'.A) →
    ∀ (ABC' : Triangle),
      isInternalPoint A₀B₀C₀.C ABC'.A ABC'.B →
      isInternalPoint A₀B₀C₀.A ABC'.B ABC'.C →
      isInternalPoint A₀B₀C₀.B ABC'.C ABC'.A →
      areSimilar ABC' A'B'C' →
      triangleArea ABC ≥ triangleArea ABC' :=
by
  sorry

end NUMINAMATH_CALUDE_largest_area_triangle_l3856_385695


namespace NUMINAMATH_CALUDE_largest_power_of_five_dividing_factorial_sum_l3856_385641

def factorial (n : ℕ) : ℕ := Nat.factorial n

def divides_exactly (x n y : ℕ) : Prop :=
  (x^n ∣ y) ∧ ¬(x^(n+1) ∣ y)

theorem largest_power_of_five_dividing_factorial_sum :
  ∃ (n : ℕ), n = 26 ∧ divides_exactly 5 n (factorial 98 + factorial 99 + factorial 100) ∧
  ∀ (m : ℕ), m > n → ¬(divides_exactly 5 m (factorial 98 + factorial 99 + factorial 100)) :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_five_dividing_factorial_sum_l3856_385641


namespace NUMINAMATH_CALUDE_sequence_equivalence_l3856_385610

theorem sequence_equivalence (n : ℕ+) : (2*n - 1)^2 - 1 = 4*n*(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_equivalence_l3856_385610


namespace NUMINAMATH_CALUDE_camel_traveler_water_ratio_l3856_385659

/-- The amount of water the traveler drank in ounces -/
def traveler_amount : ℕ := 32

/-- The number of ounces in a gallon -/
def ounces_per_gallon : ℕ := 128

/-- The total number of gallons they drank -/
def total_gallons : ℕ := 2

/-- The ratio of the amount of water the camel drank to the amount the traveler drank -/
def camel_to_traveler_ratio : ℚ := 7

theorem camel_traveler_water_ratio :
  (total_gallons * ounces_per_gallon - traveler_amount) / traveler_amount = camel_to_traveler_ratio :=
by sorry

end NUMINAMATH_CALUDE_camel_traveler_water_ratio_l3856_385659


namespace NUMINAMATH_CALUDE_expression_evaluation_l3856_385680

theorem expression_evaluation :
  let x : ℚ := 1/25
  let y : ℚ := -25
  x * (x + 2*y) - (x + 1)^2 + 2*x = -3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3856_385680


namespace NUMINAMATH_CALUDE_shape_arrangement_possible_l3856_385672

-- Define a type for geometric shapes
structure Shape :=
  (area : ℕ)

-- Define a type for arrangements of shapes
structure Arrangement :=
  (shapes : List Shape)
  (width : ℕ)
  (height : ℕ)

-- Define the properties of the desired arrangements
def is_square_with_cutout (arr : Arrangement) : Prop :=
  arr.width = 9 ∧ arr.height = 9 ∧
  ∃ (center : Shape), center ∈ arr.shapes ∧ center.area = 9

def is_rectangle (arr : Arrangement) : Prop :=
  arr.width = 9 ∧ arr.height = 12

-- Define the given set of shapes
def given_shapes : List Shape := sorry

-- State the theorem
theorem shape_arrangement_possible :
  ∃ (arr1 arr2 : Arrangement),
    (∀ s ∈ arr1.shapes, s ∈ given_shapes) ∧
    (∀ s ∈ arr2.shapes, s ∈ given_shapes) ∧
    is_square_with_cutout arr1 ∧
    is_rectangle arr2 :=
  sorry

end NUMINAMATH_CALUDE_shape_arrangement_possible_l3856_385672


namespace NUMINAMATH_CALUDE_multiply_negative_with_absolute_value_l3856_385685

theorem multiply_negative_with_absolute_value : (-3.6 : ℝ) * |(-2 : ℝ)| = -7.2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_negative_with_absolute_value_l3856_385685


namespace NUMINAMATH_CALUDE_root_transformation_l3856_385646

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 5*r₁^2 + 10 = 0) ∧ 
  (r₂^3 - 5*r₂^2 + 10 = 0) ∧ 
  (r₃^3 - 5*r₃^2 + 10 = 0) → 
  ∀ x : ℂ, x^3 - 15*x^2 + 270 = (x - 3*r₁) * (x - 3*r₂) * (x - 3*r₃) := by
sorry

end NUMINAMATH_CALUDE_root_transformation_l3856_385646


namespace NUMINAMATH_CALUDE_endpoint_sum_is_twelve_l3856_385662

/-- Given a line segment with one endpoint (6, -2) and midpoint (3, 5),
    the sum of the coordinates of the other endpoint is 12. -/
theorem endpoint_sum_is_twelve (x y : ℝ) : 
  (6 + x) / 2 = 3 → (-2 + y) / 2 = 5 → x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_is_twelve_l3856_385662


namespace NUMINAMATH_CALUDE_wendy_camera_pictures_l3856_385690

/-- Represents the number of pictures in Wendy's photo upload scenario -/
structure WendyPictures where
  phone : ℕ
  albums : ℕ
  per_album : ℕ

/-- The number of pictures Wendy uploaded from her camera -/
def camera_pictures (w : WendyPictures) : ℕ :=
  w.albums * w.per_album - w.phone

/-- Theorem stating the number of pictures Wendy uploaded from her camera -/
theorem wendy_camera_pictures :
  ∀ (w : WendyPictures),
    w.phone = 22 →
    w.albums = 4 →
    w.per_album = 6 →
    camera_pictures w = 2 := by
  sorry

end NUMINAMATH_CALUDE_wendy_camera_pictures_l3856_385690


namespace NUMINAMATH_CALUDE_expression_meaningful_iff_l3856_385698

def meaningful_expression (x : ℝ) : Prop :=
  x ≠ -5

theorem expression_meaningful_iff (x : ℝ) :
  meaningful_expression x ↔ x ≠ -5 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_meaningful_iff_l3856_385698


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3856_385639

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3856_385639


namespace NUMINAMATH_CALUDE_whiteboard_count_per_class_l3856_385608

/-- Given:
  - There are 5 classes in a building block at Oakland High.
  - Each whiteboard needs 20ml of ink for a day's use.
  - Ink costs 50 cents per ml.
  - It costs $100 to use the boards for one day.
Prove that each class uses 10 whiteboards. -/
theorem whiteboard_count_per_class (
  num_classes : ℕ)
  (ink_per_board : ℝ)
  (ink_cost_per_ml : ℝ)
  (total_daily_cost : ℝ)
  (h1 : num_classes = 5)
  (h2 : ink_per_board = 20)
  (h3 : ink_cost_per_ml = 0.5)
  (h4 : total_daily_cost = 100) :
  (total_daily_cost * num_classes) / (ink_per_board * ink_cost_per_ml) / num_classes = 10 := by
  sorry

end NUMINAMATH_CALUDE_whiteboard_count_per_class_l3856_385608


namespace NUMINAMATH_CALUDE_g_is_even_f_periodic_4_l3856_385653

-- Define the real-valued function f
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem 1: g is an even function
theorem g_is_even : ∀ x : ℝ, g f x = g f (-x) := by sorry

-- Theorem 2: f is periodic with period 4 if it's odd and f(x+2) is odd
theorem f_periodic_4 (h1 : ∀ x : ℝ, f (-x) = -f x) 
                     (h2 : ∀ x : ℝ, f (-(x+2)) = -f (x+2)) : 
  ∀ x : ℝ, f (x + 4) = f x := by sorry

end NUMINAMATH_CALUDE_g_is_even_f_periodic_4_l3856_385653


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_sequence_l3856_385651

theorem arithmetic_mean_of_sequence : 
  let start : ℕ := 3
  let count : ℕ := 60
  let sequence := fun (n : ℕ) => start + n - 1
  let sum := (count * (sequence 1 + sequence count)) / 2
  (sum : ℚ) / count = 32.5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_sequence_l3856_385651


namespace NUMINAMATH_CALUDE_replace_section_breaks_loop_l3856_385657

/-- Represents a railway section type -/
inductive SectionType
| Type1
| Type2

/-- Represents a railway configuration -/
structure RailwayConfig where
  type1Count : ℕ
  type2Count : ℕ

/-- Checks if a railway configuration forms a valid closed loop -/
def isValidClosedLoop (config : RailwayConfig) : Prop :=
  config.type1Count = config.type2Count

/-- Represents the operation of replacing a type 1 section with a type 2 section -/
def replaceSection (config : RailwayConfig) : RailwayConfig :=
  { type1Count := config.type1Count - 1,
    type2Count := config.type2Count + 1 }

/-- Main theorem: If a configuration forms a valid closed loop, 
    replacing a type 1 section with a type 2 section makes it impossible to form a closed loop -/
theorem replace_section_breaks_loop (config : RailwayConfig) :
  isValidClosedLoop config → ¬isValidClosedLoop (replaceSection config) := by
  sorry

end NUMINAMATH_CALUDE_replace_section_breaks_loop_l3856_385657


namespace NUMINAMATH_CALUDE_real_equal_roots_l3856_385697

theorem real_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - k * x + x + 8 = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - k * y + y + 8 = 0 → y = x) ↔ 
  (k = 9 ∨ k = -7) :=
sorry

end NUMINAMATH_CALUDE_real_equal_roots_l3856_385697


namespace NUMINAMATH_CALUDE_great_dane_weight_l3856_385626

theorem great_dane_weight (chihuahua pitbull great_dane : ℕ) : 
  chihuahua + pitbull + great_dane = 439 →
  pitbull = 3 * chihuahua →
  great_dane = 10 + 3 * pitbull →
  great_dane = 307 := by
  sorry

end NUMINAMATH_CALUDE_great_dane_weight_l3856_385626


namespace NUMINAMATH_CALUDE_coffee_order_total_cost_l3856_385678

def drip_coffee_price : ℝ := 2.25
def drip_coffee_quantity : ℕ := 2

def espresso_price : ℝ := 3.50
def espresso_quantity : ℕ := 1

def latte_price : ℝ := 4.00
def latte_quantity : ℕ := 2

def vanilla_syrup_price : ℝ := 0.50
def vanilla_syrup_quantity : ℕ := 1

def cold_brew_price : ℝ := 2.50
def cold_brew_quantity : ℕ := 2

def cappuccino_price : ℝ := 3.50
def cappuccino_quantity : ℕ := 1

theorem coffee_order_total_cost :
  drip_coffee_price * drip_coffee_quantity +
  espresso_price * espresso_quantity +
  latte_price * latte_quantity +
  vanilla_syrup_price * vanilla_syrup_quantity +
  cold_brew_price * cold_brew_quantity +
  cappuccino_price * cappuccino_quantity = 25.00 := by
  sorry

end NUMINAMATH_CALUDE_coffee_order_total_cost_l3856_385678


namespace NUMINAMATH_CALUDE_championship_outcomes_l3856_385615

/-- The number of possible outcomes for awarding n championship titles to m students. -/
def numberOfOutcomes (m n : ℕ) : ℕ := m^n

/-- Theorem: Given 8 students competing for 3 championship titles, 
    the number of possible outcomes for the champions is equal to 8^3. -/
theorem championship_outcomes : numberOfOutcomes 8 3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_championship_outcomes_l3856_385615


namespace NUMINAMATH_CALUDE_smallest_z_for_cube_equation_l3856_385636

theorem smallest_z_for_cube_equation : 
  (∃ (w x y z : ℕ), 
    w < x ∧ x < y ∧ y < z ∧
    w + 1 = x ∧ x + 1 = y ∧ y + 1 = z ∧
    w^3 + x^3 + y^3 = 2 * z^3) ∧
  (∀ (w x y z : ℕ),
    w < x → x < y → y < z →
    w + 1 = x → x + 1 = y → y + 1 = z →
    w^3 + x^3 + y^3 = 2 * z^3 →
    z ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_for_cube_equation_l3856_385636


namespace NUMINAMATH_CALUDE_digit_product_of_24_l3856_385688

theorem digit_product_of_24 :
  ∀ x y : ℕ,
  x < 10 ∧ y < 10 →  -- Ensures x and y are single digits
  10 * x + y = 24 →  -- The number is 24
  10 * x + y + 18 = 10 * y + x →  -- When 18 is added, digits are reversed
  x * y = 8 :=  -- Product of digits is 8
by
  sorry

end NUMINAMATH_CALUDE_digit_product_of_24_l3856_385688


namespace NUMINAMATH_CALUDE_some_number_value_l3856_385611

theorem some_number_value (some_number : ℝ) : 
  (3.242 * some_number) / 100 = 0.032420000000000004 → some_number = 1 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3856_385611


namespace NUMINAMATH_CALUDE_simplify_expression_l3856_385623

theorem simplify_expression (x : ℝ) : (3 * x + 20) + (50 * x + 25) = 53 * x + 45 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3856_385623


namespace NUMINAMATH_CALUDE_line_circle_intersections_l3856_385675

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 4 * x + 9 * y = 7

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the number of intersection points
def num_intersections : ℕ := 2

-- Theorem statement
theorem line_circle_intersections :
  ∃ (p q : ℝ × ℝ), 
    p ≠ q ∧ 
    line_eq p.1 p.2 ∧ circle_eq p.1 p.2 ∧
    line_eq q.1 q.2 ∧ circle_eq q.1 q.2 ∧
    (∀ (r : ℝ × ℝ), line_eq r.1 r.2 ∧ circle_eq r.1 r.2 → r = p ∨ r = q) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersections_l3856_385675


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l3856_385661

theorem negative_sixty_four_to_four_thirds (x : ℝ) : x = (-64)^(4/3) → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l3856_385661


namespace NUMINAMATH_CALUDE_pet_shop_dogs_l3856_385614

theorem pet_shop_dogs (total_dogs_bunnies : ℕ) (ratio_dogs : ℕ) (ratio_cats : ℕ) (ratio_bunnies : ℕ) 
  (h1 : total_dogs_bunnies = 330)
  (h2 : ratio_dogs = 7)
  (h3 : ratio_cats = 7)
  (h4 : ratio_bunnies = 8) :
  (ratio_dogs * total_dogs_bunnies) / (ratio_dogs + ratio_bunnies) = 154 := by
  sorry

#check pet_shop_dogs

end NUMINAMATH_CALUDE_pet_shop_dogs_l3856_385614


namespace NUMINAMATH_CALUDE_sin_315_degrees_l3856_385616

theorem sin_315_degrees : 
  Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l3856_385616


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3856_385621

theorem sum_of_coefficients (b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^5 = b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 3125 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3856_385621


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_range_l3856_385669

theorem right_triangle_leg_sum_range (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  x^2 + y^2 = 5 → Real.sqrt 5 < x + y ∧ x + y ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_range_l3856_385669


namespace NUMINAMATH_CALUDE_no_solution_gcd_lcm_sum_l3856_385650

theorem no_solution_gcd_lcm_sum (x y : ℕ) : 
  Nat.gcd x y + Nat.lcm x y + x + y ≠ 2019 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_gcd_lcm_sum_l3856_385650


namespace NUMINAMATH_CALUDE_inscribed_polygon_sides_l3856_385671

theorem inscribed_polygon_sides (r : ℝ) (n : ℕ) (a : ℝ) : 
  r = 1 → 
  a = 2 * Real.sin (π / n) → 
  1 < a → 
  a < Real.sqrt 2 → 
  n = 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_polygon_sides_l3856_385671


namespace NUMINAMATH_CALUDE_min_additional_coins_l3856_385617

/-- The number of friends Alex has -/
def num_friends : ℕ := 15

/-- The initial number of coins Alex has -/
def initial_coins : ℕ := 85

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating the minimum number of additional coins needed -/
theorem min_additional_coins : 
  sum_first_n num_friends - initial_coins = 35 := by sorry

end NUMINAMATH_CALUDE_min_additional_coins_l3856_385617


namespace NUMINAMATH_CALUDE_money_left_after_trip_l3856_385692

def initial_savings : ℕ := 6000
def flight_cost : ℕ := 1200
def hotel_cost : ℕ := 800
def food_cost : ℕ := 3000

theorem money_left_after_trip :
  initial_savings - (flight_cost + hotel_cost + food_cost) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_trip_l3856_385692


namespace NUMINAMATH_CALUDE_envelope_counting_time_l3856_385663

/-- Represents the time in seconds to count a given number of envelopes -/
def count_time (envelopes : ℕ) : ℕ :=
  10 * ((100 - envelopes) / 10)

theorem envelope_counting_time :
  (count_time 60 = 40) ∧ (count_time 90 = 10) :=
sorry

end NUMINAMATH_CALUDE_envelope_counting_time_l3856_385663


namespace NUMINAMATH_CALUDE_green_peaches_per_basket_l3856_385674

theorem green_peaches_per_basket (num_baskets : ℕ) (red_per_basket : ℕ) (total_peaches : ℕ) :
  num_baskets = 11 →
  red_per_basket = 10 →
  total_peaches = 308 →
  ∃ green_per_basket : ℕ, 
    green_per_basket * num_baskets + red_per_basket * num_baskets = total_peaches ∧
    green_per_basket = 18 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_per_basket_l3856_385674


namespace NUMINAMATH_CALUDE_correct_answers_range_l3856_385640

/-- Represents the scoring system and conditions of the test --/
structure TestScoring where
  total_questions : Nat
  correct_points : Int
  wrong_points : Int
  min_score : Int

/-- Represents Xiaoyu's test result --/
structure TestResult (scoring : TestScoring) where
  correct_answers : Nat
  no_missed_questions : correct_answers ≤ scoring.total_questions

/-- Calculates the total score based on the number of correct answers --/
def calculate_score (scoring : TestScoring) (result : TestResult scoring) : Int :=
  result.correct_answers * scoring.correct_points + 
  (scoring.total_questions - result.correct_answers) * scoring.wrong_points

/-- Theorem stating the range of possible values for correct answers --/
theorem correct_answers_range (scoring : TestScoring) 
  (h_total : scoring.total_questions = 25)
  (h_correct : scoring.correct_points = 4)
  (h_wrong : scoring.wrong_points = -2)
  (h_min_score : scoring.min_score = 70) :
  ∀ (result : TestResult scoring), 
    calculate_score scoring result ≥ scoring.min_score →
    (20 : Nat) ≤ result.correct_answers ∧ result.correct_answers ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_answers_range_l3856_385640


namespace NUMINAMATH_CALUDE_x_y_power_product_l3856_385645

theorem x_y_power_product (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) :
  (1/3) * x^8 * y^9 = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_x_y_power_product_l3856_385645


namespace NUMINAMATH_CALUDE_runner_lead_l3856_385607

/-- Represents the relative speeds of runners in a race. -/
structure RunnerSpeeds where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The race setup with given conditions. -/
def raceSetup (s : RunnerSpeeds) : Prop :=
  s.b = (5/6) * s.a ∧ s.c = (3/4) * s.a

/-- The theorem statement. -/
theorem runner_lead (s : RunnerSpeeds) (h : raceSetup s) :
  150 - (s.c * (150 / s.a)) = 37.5 := by
  sorry

#check runner_lead

end NUMINAMATH_CALUDE_runner_lead_l3856_385607


namespace NUMINAMATH_CALUDE_book_reading_ratio_l3856_385647

/-- Given the number of books read by Candice, Amanda, Kara, and Patricia in a Book Tournament, 
    prove the ratio of books read by Kara to Amanda. -/
theorem book_reading_ratio 
  (candice amanda kara patricia : ℕ) 
  (x : ℚ) 
  (h1 : candice = 3 * amanda) 
  (h2 : candice = 18) 
  (h3 : kara = x * amanda) 
  (h4 : patricia = 7 * kara) : 
  (kara : ℚ) / amanda = x := by
  sorry

end NUMINAMATH_CALUDE_book_reading_ratio_l3856_385647


namespace NUMINAMATH_CALUDE_expand_product_l3856_385633

theorem expand_product (x : ℝ) : (5*x + 3) * (2*x^2 + 4) = 10*x^3 + 6*x^2 + 20*x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3856_385633


namespace NUMINAMATH_CALUDE_sum_2018_is_1009_l3856_385624

/-- An arithmetic sequence with first term 1 and common difference -1/2017 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  1 - (n - 1 : ℚ) / 2017

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℚ :=
  n * (arithmetic_sequence 1 + arithmetic_sequence n) / 2

/-- Theorem stating that the sum of the first 2018 terms is 1009 -/
theorem sum_2018_is_1009 : S 2018 = 1009 := by
  sorry

end NUMINAMATH_CALUDE_sum_2018_is_1009_l3856_385624


namespace NUMINAMATH_CALUDE_smallest_solution_floor_square_diff_l3856_385620

theorem smallest_solution_floor_square_diff (x : ℝ) :
  (∀ y : ℝ, y < x → ⌊y^2⌋ - ⌊y⌋^2 ≠ 19) ∧ ⌊x^2⌋ - ⌊x⌋^2 = 19 ↔ x = Real.sqrt 104 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_square_diff_l3856_385620


namespace NUMINAMATH_CALUDE_lidia_remaining_money_l3856_385693

/-- Calculates the remaining money after Lidia's app purchase --/
def remaining_money (productivity_apps : ℕ) (productivity_cost : ℚ)
                    (gaming_apps : ℕ) (gaming_cost : ℚ)
                    (lifestyle_apps : ℕ) (lifestyle_cost : ℚ)
                    (initial_money : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_cost := productivity_apps * productivity_cost +
                    gaming_apps * gaming_cost +
                    lifestyle_apps * lifestyle_cost
  let discounted_cost := total_cost * (1 - discount_rate)
  let final_cost := discounted_cost * (1 + tax_rate)
  initial_money - final_cost

/-- Theorem stating that Lidia will be left with $6.16 after her app purchase --/
theorem lidia_remaining_money :
  remaining_money 5 4 7 5 3 3 66 (15/100) (10/100) = (616/100) :=
sorry


end NUMINAMATH_CALUDE_lidia_remaining_money_l3856_385693


namespace NUMINAMATH_CALUDE_edward_booth_tickets_l3856_385638

/-- The number of tickets Edward spent at the 'dunk a clown' booth -/
def tickets_spent_at_booth (total_tickets : ℕ) (cost_per_ride : ℕ) (possible_rides : ℕ) : ℕ :=
  total_tickets - (cost_per_ride * possible_rides)

/-- Proof that Edward spent 23 tickets at the 'dunk a clown' booth -/
theorem edward_booth_tickets : 
  tickets_spent_at_booth 79 7 8 = 23 := by
  sorry

end NUMINAMATH_CALUDE_edward_booth_tickets_l3856_385638


namespace NUMINAMATH_CALUDE_fayes_age_l3856_385629

/-- Represents the ages of Chad, Diana, Eduardo, and Faye --/
structure Ages where
  chad : ℕ
  diana : ℕ
  eduardo : ℕ
  faye : ℕ

/-- The age relationships between Chad, Diana, Eduardo, and Faye --/
def age_relationships (ages : Ages) : Prop :=
  ages.diana = ages.eduardo - 4 ∧
  ages.eduardo = ages.chad + 5 ∧
  ages.faye = ages.chad + 2 ∧
  ages.diana = 16

/-- Theorem stating that given the age relationships, Faye's age is 17 --/
theorem fayes_age (ages : Ages) : age_relationships ages → ages.faye = 17 := by
  sorry

end NUMINAMATH_CALUDE_fayes_age_l3856_385629


namespace NUMINAMATH_CALUDE_elise_initial_dog_food_l3856_385604

/-- The amount of dog food Elise already had -/
def initial_amount : ℕ := sorry

/-- The amount of dog food in the first bag Elise bought -/
def first_bag : ℕ := 15

/-- The amount of dog food in the second bag Elise bought -/
def second_bag : ℕ := 10

/-- The total amount of dog food Elise has after buying -/
def total_amount : ℕ := 40

theorem elise_initial_dog_food : initial_amount = 15 :=
  sorry

end NUMINAMATH_CALUDE_elise_initial_dog_food_l3856_385604


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3856_385681

theorem quadratic_rewrite :
  ∃ (p q r : ℤ), 
    (∀ x, 8 * x^2 - 24 * x - 56 = (p * x + q)^2 + r) ∧
    p * q = -12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3856_385681


namespace NUMINAMATH_CALUDE_inequality_proof_l3856_385686

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  a + b + c ≤ (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ∧
  (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ≤ a^3/(b*c) + b^3/(c*a) + c^3/(a*b) :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3856_385686


namespace NUMINAMATH_CALUDE_unique_divisible_by_19_l3856_385660

/-- Converts a base 7 number of the form 52x3 to its decimal equivalent --/
def base7ToDecimal (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 3

/-- Checks if a number is a valid base 7 digit --/
def isBase7Digit (x : ℕ) : Prop := x ≤ 6

theorem unique_divisible_by_19 :
  ∃! x : ℕ, isBase7Digit x ∧ (base7ToDecimal x) % 19 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_by_19_l3856_385660


namespace NUMINAMATH_CALUDE_red_balls_estimate_l3856_385635

/-- Represents a bag of balls -/
structure Bag where
  total : ℕ
  redProb : ℝ

/-- Calculates the expected number of red balls in the bag -/
def expectedRedBalls (b : Bag) : ℝ :=
  b.total * b.redProb

theorem red_balls_estimate (b : Bag) 
  (h1 : b.total = 20)
  (h2 : b.redProb = 0.25) : 
  expectedRedBalls b = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_estimate_l3856_385635


namespace NUMINAMATH_CALUDE_prob_other_side_red_is_two_thirds_l3856_385631

/-- Represents a card with two sides --/
structure Card where
  side1 : Bool  -- True for red, False for black
  side2 : Bool

/-- The set of all cards in the box --/
def box : Finset Card := sorry

/-- The total number of cards in the box --/
def total_cards : Nat := 8

/-- The number of cards that are black on both sides --/
def black_both_sides : Nat := 4

/-- The number of cards that are black on one side and red on the other --/
def black_red : Nat := 2

/-- The number of cards that are red on both sides --/
def red_both_sides : Nat := 2

/-- Axiom: The box contains the correct number of each type of card --/
axiom box_composition :
  (box.filter (fun c => !c.side1 ∧ !c.side2)).card = black_both_sides ∧
  (box.filter (fun c => (c.side1 ∧ !c.side2) ∨ (!c.side1 ∧ c.side2))).card = black_red ∧
  (box.filter (fun c => c.side1 ∧ c.side2)).card = red_both_sides

/-- Axiom: The total number of cards is correct --/
axiom total_cards_correct : box.card = total_cards

/-- The probability of selecting a card with a red side, given that one side is observed to be red --/
def prob_other_side_red (observed_red : Bool) : ℚ := sorry

/-- Theorem: The probability that the other side is red, given that the observed side is red, is 2/3 --/
theorem prob_other_side_red_is_two_thirds (observed_red : Bool) :
  observed_red → prob_other_side_red observed_red = 2/3 := by sorry

end NUMINAMATH_CALUDE_prob_other_side_red_is_two_thirds_l3856_385631


namespace NUMINAMATH_CALUDE_four_digit_number_with_sum_14_divisible_by_14_l3856_385699

theorem four_digit_number_with_sum_14_divisible_by_14 :
  ∃ n : ℕ,
    1000 ≤ n ∧ n ≤ 9999 ∧
    (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10) = 14) ∧
    n % 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_with_sum_14_divisible_by_14_l3856_385699


namespace NUMINAMATH_CALUDE_pet_food_inventory_l3856_385683

theorem pet_food_inventory (dog_food : ℕ) (difference : ℕ) (cat_food : ℕ) : 
  dog_food = 600 → 
  dog_food = cat_food + difference → 
  difference = 273 →
  cat_food = 327 := by
  sorry

end NUMINAMATH_CALUDE_pet_food_inventory_l3856_385683


namespace NUMINAMATH_CALUDE_probability_age_20_to_40_l3856_385684

theorem probability_age_20_to_40 (total : ℕ) (below_20 : ℕ) (between_20_30 : ℕ) (between_30_40 : ℕ) :
  total = 350 →
  below_20 = 120 →
  between_20_30 = 105 →
  between_30_40 = 85 →
  (between_20_30 + between_30_40 : ℚ) / total = 19 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_age_20_to_40_l3856_385684


namespace NUMINAMATH_CALUDE_probability_one_correct_l3856_385644

/-- The number of options for each multiple-choice question -/
def num_options : ℕ := 4

/-- The number of questions -/
def num_questions : ℕ := 2

/-- The number of correct answers needed -/
def correct_answers : ℕ := 1

/-- The probability of getting exactly one answer correct out of two multiple-choice questions,
    each with 4 options and only one correct answer, when answers are randomly selected -/
theorem probability_one_correct :
  (num_options - 1) * num_questions / (num_options ^ num_questions) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_correct_l3856_385644


namespace NUMINAMATH_CALUDE_x_fourth_coefficient_l3856_385643

def binomial_coefficient (n k : ℕ) : ℤ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def expansion_coefficient (n r : ℕ) : ℤ :=
  (-1)^r * binomial_coefficient n r

theorem x_fourth_coefficient :
  expansion_coefficient 8 3 = -56 :=
by sorry

end NUMINAMATH_CALUDE_x_fourth_coefficient_l3856_385643


namespace NUMINAMATH_CALUDE_sum_F_equals_250_l3856_385602

-- Define the function F
def F (n : ℕ) : ℕ := sorry

-- Define the sum of F from 1 to 50
def sum_F : ℕ := (List.range 50).map (fun i => F (i + 1)) |>.sum

-- Theorem statement
theorem sum_F_equals_250 : sum_F = 250 := by sorry

end NUMINAMATH_CALUDE_sum_F_equals_250_l3856_385602


namespace NUMINAMATH_CALUDE_martha_initial_pantry_bottles_l3856_385654

/-- The number of bottles of juice Martha initially had in the pantry -/
def initial_pantry_bottles : ℕ := sorry

/-- The number of bottles of juice Martha initially had in the refrigerator -/
def initial_fridge_bottles : ℕ := 4

/-- The number of bottles of juice Martha bought during the week -/
def bought_bottles : ℕ := 5

/-- The number of bottles of juice Martha and her family drank during the week -/
def drunk_bottles : ℕ := 3

/-- The number of bottles of juice left at the end of the week -/
def remaining_bottles : ℕ := 10

theorem martha_initial_pantry_bottles :
  initial_pantry_bottles = 4 :=
by sorry

end NUMINAMATH_CALUDE_martha_initial_pantry_bottles_l3856_385654


namespace NUMINAMATH_CALUDE_inequality_proof_l3856_385652

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c ≥ a * b * c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3856_385652


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l3856_385625

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The ratio of a_n to a_(2n) is constant -/
def ConstantRatio (a : ℕ → ℝ) : Prop :=
  ∃ c, ∀ n, a n ≠ 0 → a (2*n) ≠ 0 → a n / a (2*n) = c

theorem arithmetic_sequence_constant_ratio (a : ℕ → ℝ) 
    (h1 : ArithmeticSequence a) (h2 : ConstantRatio a) :
    ∃ c, (c = 1 ∨ c = 1/2) ∧ ∀ n, a n ≠ 0 → a (2*n) ≠ 0 → a n / a (2*n) = c :=
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l3856_385625


namespace NUMINAMATH_CALUDE_min_distance_point_to_tangent_l3856_385600

/-- The minimum distance between a point on the line x - y - 6 = 0 and 
    its tangent point on the circle (x-1)^2 + (y-1)^2 = 4 is √14 -/
theorem min_distance_point_to_tangent (x y : ℝ) : 
  let circle := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 4}
  let line := {p : ℝ × ℝ | p.1 - p.2 - 6 = 0}
  ∃ (M N : ℝ × ℝ), M ∈ line ∧ N ∈ circle ∧ 
    (∀ (M' N' : ℝ × ℝ), M' ∈ line → N' ∈ circle → 
      (M'.1 - N'.1)^2 + (M'.2 - N'.2)^2 ≥ (M.1 - N.1)^2 + (M.2 - N.2)^2) ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_point_to_tangent_l3856_385600


namespace NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_proof_l3856_385655

theorem father_son_age_sum : ℕ → ℕ → ℕ
  | father_age, son_age =>
    father_age + 2 * son_age

theorem father_son_age_proof (father_age son_age : ℕ) 
  (h1 : father_age = 40) 
  (h2 : son_age = 15) : 
  father_son_age_sum father_age son_age = 70 := by
  sorry

end NUMINAMATH_CALUDE_father_son_age_sum_father_son_age_proof_l3856_385655


namespace NUMINAMATH_CALUDE_stamps_from_other_countries_l3856_385670

def total_stamps : ℕ := 500
def chinese_percent : ℚ := 40 / 100
def us_percent : ℚ := 25 / 100
def japanese_percent : ℚ := 15 / 100
def british_percent : ℚ := 10 / 100

theorem stamps_from_other_countries :
  total_stamps * (1 - (chinese_percent + us_percent + japanese_percent + british_percent)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_stamps_from_other_countries_l3856_385670


namespace NUMINAMATH_CALUDE_total_rainfall_sum_l3856_385634

/-- The rainfall recorded on Monday in centimeters -/
def monday_rainfall : ℚ := 0.16666666666666666

/-- The rainfall recorded on Tuesday in centimeters -/
def tuesday_rainfall : ℚ := 0.4166666666666667

/-- The rainfall recorded on Wednesday in centimeters -/
def wednesday_rainfall : ℚ := 0.08333333333333333

/-- The total rainfall recorded over the three days -/
def total_rainfall : ℚ := monday_rainfall + tuesday_rainfall + wednesday_rainfall

/-- Theorem stating that the total rainfall equals 0.6666666666666667 cm -/
theorem total_rainfall_sum :
  total_rainfall = 0.6666666666666667 := by sorry

end NUMINAMATH_CALUDE_total_rainfall_sum_l3856_385634


namespace NUMINAMATH_CALUDE_bobby_total_pieces_l3856_385612

def total_pieces_eaten (initial_candy : ℕ) (initial_chocolate : ℕ) (initial_licorice : ℕ) 
                       (additional_candy : ℕ) (additional_chocolate : ℕ) : ℕ :=
  (initial_candy + additional_candy) + (initial_chocolate + additional_chocolate) + initial_licorice

theorem bobby_total_pieces : 
  total_pieces_eaten 33 14 7 4 5 = 63 := by
  sorry

end NUMINAMATH_CALUDE_bobby_total_pieces_l3856_385612


namespace NUMINAMATH_CALUDE_intersection_point_correct_l3856_385601

/-- Represents a 2D vector --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a parametric line in 2D --/
structure ParametricLine where
  origin : Vector2D
  direction : Vector2D

/-- The first line --/
def line1 : ParametricLine :=
  { origin := { x := 1, y := 2 },
    direction := { x := -2, y := 4 } }

/-- The second line --/
def line2 : ParametricLine :=
  { origin := { x := 3, y := 5 },
    direction := { x := 1, y := 3 } }

/-- Calculates a point on a parametric line given a parameter t --/
def pointOnLine (line : ParametricLine) (t : ℝ) : Vector2D :=
  { x := line.origin.x + t * line.direction.x,
    y := line.origin.y + t * line.direction.y }

/-- The intersection point of the two lines --/
def intersectionPoint : Vector2D :=
  { x := 1.2, y := 1.6 }

/-- Theorem stating that the calculated intersection point is correct --/
theorem intersection_point_correct :
  ∃ t u : ℝ, pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint :=
sorry


end NUMINAMATH_CALUDE_intersection_point_correct_l3856_385601


namespace NUMINAMATH_CALUDE_negative_division_l3856_385619

theorem negative_division (a b : ℤ) (ha : a = -300) (hb : b = -50) :
  a / b = 6 := by
  sorry

end NUMINAMATH_CALUDE_negative_division_l3856_385619


namespace NUMINAMATH_CALUDE_triangle_sides_l3856_385679

theorem triangle_sides (a b c : ℚ) : 
  a + b + c = 24 →
  a + 2*b = 2*c →
  a = (1/2) * b →
  a = 16/3 ∧ b = 32/3 ∧ c = 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_sides_l3856_385679


namespace NUMINAMATH_CALUDE_triangle_third_side_l3856_385658

/-- Given a triangle with sides b, c, and x, where the area S = 0.4bc, 
    prove that the third side x satisfies the equation: x² = b² + c² ± 1.2bc -/
theorem triangle_third_side (b c x : ℝ) (h : b > 0 ∧ c > 0 ∧ x > 0) :
  (0.4 * b * c)^2 = (1/16) * (4 * b^2 * c^2 - (b^2 + c^2 - x^2)^2) →
  x^2 = b^2 + c^2 + 1.2 * b * c ∨ x^2 = b^2 + c^2 - 1.2 * b * c :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l3856_385658


namespace NUMINAMATH_CALUDE_car_down_payment_sharing_l3856_385668

def down_payment : ℕ := 3500
def individual_payment : ℕ := 1167

theorem car_down_payment_sharing :
  (down_payment + 2) / individual_payment = 3 :=
sorry

end NUMINAMATH_CALUDE_car_down_payment_sharing_l3856_385668


namespace NUMINAMATH_CALUDE_circular_pond_area_l3856_385627

theorem circular_pond_area (AB CD : ℝ) (h1 : AB = 20) (h2 : CD = 12) : 
  let R := CD
  let A := π * R^2
  A = 244 * π := by sorry

end NUMINAMATH_CALUDE_circular_pond_area_l3856_385627


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3856_385632

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + m * x + 36 = 0) ↔ m = 12 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3856_385632


namespace NUMINAMATH_CALUDE_equilateral_triangle_to_square_l3856_385648

/-- Given an equilateral triangle with area 121√3 cm², prove that decreasing each side by 6 cm
    and transforming it into a square results in a square with area 256 cm². -/
theorem equilateral_triangle_to_square (s : ℝ) : 
  (s^2 * Real.sqrt 3 / 4 = 121 * Real.sqrt 3) →
  ((s - 6)^2 = 256) := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_to_square_l3856_385648


namespace NUMINAMATH_CALUDE_sunset_time_calculation_l3856_385606

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Adds two Times together -/
def addTime (t1 t2 : Time) : Time :=
  let totalMinutes := t1.hours * 60 + t1.minutes + t2.hours * 60 + t2.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

/-- Converts 24-hour format to 12-hour format -/
def to12HourFormat (t : Time) : Time :=
  if t.hours ≥ 12 then
    { hours := if t.hours = 12 then 12 else t.hours - 12, minutes := t.minutes }
  else
    { hours := if t.hours = 0 then 12 else t.hours, minutes := t.minutes }

theorem sunset_time_calculation (sunrise : Time) (daylight : Time) : 
  to12HourFormat (addTime sunrise daylight) = { hours := 7, minutes := 40 } :=
  sorry

end NUMINAMATH_CALUDE_sunset_time_calculation_l3856_385606
