import Mathlib

namespace NUMINAMATH_CALUDE_function_properties_l87_8724

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

theorem function_properties 
  (ω φ : ℝ) 
  (hω : ω > 0) 
  (hφ : 0 < φ ∧ φ < Real.pi / 2) 
  (hperiod : ∀ x, f ω φ (x + Real.pi) = f ω φ x)
  (hsymmetry : ∀ x, f ω φ (-Real.pi/24 + x) = f ω φ (-Real.pi/24 - x))
  (A B C : ℝ)
  (ha : ∀ a b c : ℝ, a = 3 → b + c = 6 → a^2 = b^2 + c^2 - 2*b*c*Real.cos A)
  (hf : f ω φ (-A/2) = Real.sqrt 2) :
  ω = 2 ∧ φ = Real.pi/12 ∧ ∃ (b c : ℝ), b = 3 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l87_8724


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l87_8736

theorem no_prime_roots_for_quadratic : ¬∃ (k : ℤ), ∃ (p q : ℕ), 
  Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (p : ℤ) + q = 57 ∧ (p : ℤ) * q = k ∧
  ∀ (x : ℤ), x^2 - 57*x + k = 0 ↔ x = p ∨ x = q := by
  sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l87_8736


namespace NUMINAMATH_CALUDE_equality_of_ordered_triples_l87_8735

theorem equality_of_ordered_triples
  (a b c x y z : ℝ)
  (positive_a : 0 < a) (positive_b : 0 < b) (positive_c : 0 < c)
  (positive_x : 0 < x) (positive_y : 0 < y) (positive_z : 0 < z)
  (sum_equality : x + y + z = a + b + c)
  (product_equality : x * y * z = a * b * c)
  (ordering_xyz : a ≤ x ∧ x < y ∧ y < z ∧ z ≤ c)
  (ordering_abc : a < b ∧ b < c) :
  a = x ∧ b = y ∧ c = z := by
  sorry

end NUMINAMATH_CALUDE_equality_of_ordered_triples_l87_8735


namespace NUMINAMATH_CALUDE_store_inventory_count_l87_8738

theorem store_inventory_count : 
  ∀ (original_price : ℝ) (discount_rate : ℝ) (sold_percentage : ℝ) 
    (debt : ℝ) (remaining : ℝ),
  original_price = 50 →
  discount_rate = 0.8 →
  sold_percentage = 0.9 →
  debt = 15000 →
  remaining = 3000 →
  (((1 - discount_rate) * original_price * sold_percentage) * 
    (debt + remaining) / ((1 - discount_rate) * original_price * sold_percentage)) = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_store_inventory_count_l87_8738


namespace NUMINAMATH_CALUDE_z_change_l87_8775

theorem z_change (w h z : ℝ) (z' : ℝ) : 
  let q := 5 * w / (4 * h * z^2)
  let q' := 5 * (4 * w) / (4 * (2 * h) * z'^2)
  q' / q = 2 / 9 →
  z' / z = 3 * Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_z_change_l87_8775


namespace NUMINAMATH_CALUDE_polynomial_factorization_l87_8747

theorem polynomial_factorization (a : ℝ) : a^3 + 2*a^2 + a = a*(a+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l87_8747


namespace NUMINAMATH_CALUDE_translated_line_point_l87_8797

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given amount -/
def translateLine (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + amount }

/-- Checks if a point lies on a line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem translated_line_point (m : ℝ) : 
  let original_line : Line := { slope := 1, intercept := 0 }
  let translated_line := translateLine original_line 3
  pointOnLine translated_line 2 m → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_translated_line_point_l87_8797


namespace NUMINAMATH_CALUDE_cube_difference_l87_8757

theorem cube_difference (x y : ℚ) (h1 : x + y = 10) (h2 : 2 * x - y = 16) :
  x^3 - y^3 = 17512 / 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_l87_8757


namespace NUMINAMATH_CALUDE_set_union_complement_and_subset_necessary_not_sufficient_condition_l87_8721

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}

def B (a : ℝ) : Set ℝ := {x | -a - 1 < x ∧ x < -a + 1}

theorem set_union_complement_and_subset (a : ℝ) :
  a = 3 → (Set.univ \ A) ∪ B a = {x | x < -2 ∨ x ≥ 1} :=
sorry

theorem necessary_not_sufficient_condition (a : ℝ) :
  (∀ x, x ∈ B a → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B a) ↔ 0 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_set_union_complement_and_subset_necessary_not_sufficient_condition_l87_8721


namespace NUMINAMATH_CALUDE_q_polynomial_form_l87_8700

-- Define q as a function from ℝ to ℝ
variable (q : ℝ → ℝ)

-- Define the theorem
theorem q_polynomial_form :
  (∀ x, q x + (x^6 + 4*x^4 + 8*x^2 + 7*x) = (12*x^4 + 30*x^3 + 40*x^2 + 10*x + 2)) →
  (∀ x, q x = -x^6 + 8*x^4 + 30*x^3 + 32*x^2 + 3*x + 2) :=
by
  sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l87_8700


namespace NUMINAMATH_CALUDE_charles_earnings_l87_8749

def housesitting_rate : ℕ := 15
def dog_walking_rate : ℕ := 22
def housesitting_hours : ℕ := 10
def dogs_walked : ℕ := 3
def hours_per_dog : ℕ := 1

def total_earnings : ℕ := housesitting_rate * housesitting_hours + dog_walking_rate * dogs_walked * hours_per_dog

theorem charles_earnings : total_earnings = 216 := by
  sorry

end NUMINAMATH_CALUDE_charles_earnings_l87_8749


namespace NUMINAMATH_CALUDE_football_throw_distance_l87_8713

/-- Proves that Kyle threw the ball 24 yards farther than Parker -/
theorem football_throw_distance (parker_distance : ℝ) (grant_distance : ℝ) (kyle_distance : ℝ) :
  parker_distance = 16 ∧
  grant_distance = parker_distance * 1.25 ∧
  kyle_distance = grant_distance * 2 →
  kyle_distance - parker_distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_football_throw_distance_l87_8713


namespace NUMINAMATH_CALUDE_correct_operation_l87_8781

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - 3 * b * a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l87_8781


namespace NUMINAMATH_CALUDE_money_ratio_l87_8737

theorem money_ratio (j : ℝ) (k : ℝ) : 
  (j + (2 * j - 7) + 60 = 113) →  -- Sum of all money
  (60 = k * j) →                  -- Patricia's money is a multiple of Jethro's
  (60 : ℝ) / j = 3 :=             -- Ratio of Patricia's to Jethro's money
by
  sorry

end NUMINAMATH_CALUDE_money_ratio_l87_8737


namespace NUMINAMATH_CALUDE_exponent_multiplication_l87_8716

theorem exponent_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l87_8716


namespace NUMINAMATH_CALUDE_triangle_ad_length_l87_8707

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the perpendicular foot
def perpendicularFoot (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the ratio of line segments
def ratio (p q r s : ℝ × ℝ) : ℚ := sorry

theorem triangle_ad_length (abc : Triangle) :
  let A := abc.A
  let B := abc.B
  let C := abc.C
  let D := perpendicularFoot A B C
  length A B = 13 →
  length A C = 20 →
  ratio B D C D = 3/4 →
  length A D = 8 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_ad_length_l87_8707


namespace NUMINAMATH_CALUDE_fountain_distance_is_30_l87_8791

/-- The distance from Mrs. Hilt's desk to the water fountain -/
def fountain_distance (total_distance : ℕ) (num_trips : ℕ) : ℕ :=
  total_distance / num_trips

/-- Theorem stating that the distance to the water fountain is 30 feet -/
theorem fountain_distance_is_30 :
  fountain_distance 120 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fountain_distance_is_30_l87_8791


namespace NUMINAMATH_CALUDE_friend_team_assignment_count_l87_8727

theorem friend_team_assignment_count : 
  let n_friends : ℕ := 8
  let n_teams : ℕ := 4
  n_teams ^ n_friends = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friend_team_assignment_count_l87_8727


namespace NUMINAMATH_CALUDE_sum_lowest_two_scores_l87_8762

/-- Represents a set of math test scores -/
structure MathTests where
  scores : Finset ℕ
  count : Nat
  average : ℕ
  median : ℕ
  mode : ℕ

/-- The sum of the lowest two scores in a set of math tests -/
def sumLowestTwo (tests : MathTests) : ℕ :=
  sorry

/-- Theorem: Given 5 math test scores with an average of 90, a median of 91, 
    and a mode of 93, the sum of the lowest two scores is 173 -/
theorem sum_lowest_two_scores (tests : MathTests) 
  (h_count : tests.count = 5)
  (h_avg : tests.average = 90)
  (h_median : tests.median = 91)
  (h_mode : tests.mode = 93) :
  sumLowestTwo tests = 173 := by
  sorry

end NUMINAMATH_CALUDE_sum_lowest_two_scores_l87_8762


namespace NUMINAMATH_CALUDE_equation_solution_l87_8708

/-- Given positive real numbers a, b, c ≤ 1, the equation 
    min{√((ab+1)/(abc)), √((bc+1)/(abc)), √((ac+1)/(abc))} = √((1-a)/a) + √((1-b)/b) + √((1-c)/c)
    is satisfied if and only if (a, b, c) = (1/(-t^2 + t + 1), t, 1 - t) for 1/2 ≤ t < 1 or its permutations. -/
theorem equation_solution (a b c : ℝ) (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (min (Real.sqrt ((a*b+1)/(a*b*c))) (min (Real.sqrt ((b*c+1)/(a*b*c))) (Real.sqrt ((a*c+1)/(a*b*c)))) =
   Real.sqrt ((1-a)/a) + Real.sqrt ((1-b)/b) + Real.sqrt ((1-c)/c)) ↔
  (∃ t : ℝ, (1/2 ≤ t ∧ t < 1) ∧
   ((a = 1/(-t^2 + t + 1) ∧ b = t ∧ c = 1 - t) ∨
    (a = t ∧ b = 1 - t ∧ c = 1/(-t^2 + t + 1)) ∨
    (a = 1 - t ∧ b = 1/(-t^2 + t + 1) ∧ c = t))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l87_8708


namespace NUMINAMATH_CALUDE_race_completion_time_l87_8740

theorem race_completion_time (walking_time jogging_time total_time : ℕ) : 
  walking_time = 9 →
  jogging_time * 3 = walking_time * 4 →
  total_time = walking_time + jogging_time →
  total_time = 21 := by
sorry

end NUMINAMATH_CALUDE_race_completion_time_l87_8740


namespace NUMINAMATH_CALUDE_remaining_painting_time_l87_8782

/-- Calculates the remaining painting time for a building -/
def remaining_time (total_rooms : ℕ) (hours_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * hours_per_room

/-- Theorem: The remaining time to finish all painting work is 155 hours -/
theorem remaining_painting_time : 
  let building1 := remaining_time 12 7 5
  let building2 := remaining_time 15 6 4
  let building3 := remaining_time 10 5 2
  building1 + building2 + building3 = 155 := by
  sorry

end NUMINAMATH_CALUDE_remaining_painting_time_l87_8782


namespace NUMINAMATH_CALUDE_distinct_colorings_count_l87_8788

/-- Represents the symmetries of a regular octagon -/
inductive OctagonSymmetry
| Identity
| Reflection (n : Fin 8)
| Rotation (n : Fin 4)

/-- Represents a coloring of 8 disks -/
def Coloring := Fin 8 → Fin 3

/-- The number of disks -/
def n : ℕ := 8

/-- The number of colors -/
def k : ℕ := 3

/-- The number of each color -/
def colorCounts : Fin 3 → ℕ
| 0 => 4  -- blue
| 1 => 3  -- red
| 2 => 1  -- green
| _ => 0  -- unreachable

/-- The set of all possible colorings -/
def allColorings : Finset Coloring := sorry

/-- Whether a coloring is fixed by a given symmetry -/
def isFixed (c : Coloring) (s : OctagonSymmetry) : Prop := sorry

/-- The number of colorings fixed by each symmetry -/
def fixedColorings (s : OctagonSymmetry) : ℕ := sorry

/-- The set of all symmetries -/
def symmetries : Finset OctagonSymmetry := sorry

/-- The main theorem: the number of distinct colorings is 21 -/
theorem distinct_colorings_count :
  (Finset.sum symmetries fixedColorings) / Finset.card symmetries = 21 := sorry

end NUMINAMATH_CALUDE_distinct_colorings_count_l87_8788


namespace NUMINAMATH_CALUDE_sheet_area_calculation_l87_8744

/-- Represents a rectangular sheet of paper. -/
structure Sheet where
  length : ℝ
  width : ℝ

/-- Represents the perimeters of the three rectangles after folding. -/
structure Perimeters where
  p1 : ℝ
  p2 : ℝ
  p3 : ℝ

/-- Calculates the perimeters of the three rectangles after folding. -/
def calculatePerimeters (s : Sheet) : Perimeters :=
  { p1 := 2 * s.length,
    p2 := 2 * s.width,
    p3 := 2 * (s.length - s.width) }

/-- The main theorem stating the conditions and the result to be proved. -/
theorem sheet_area_calculation (s : Sheet) :
  let p := calculatePerimeters s
  p.p1 = p.p2 + 20 ∧ p.p2 = p.p3 + 16 →
  s.length * s.width = 504 := by
  sorry


end NUMINAMATH_CALUDE_sheet_area_calculation_l87_8744


namespace NUMINAMATH_CALUDE_male_average_is_100_l87_8723

/-- Represents the average number of tickets sold by a group of members -/
structure GroupAverage where
  count : ℕ  -- Number of members in the group
  average : ℝ  -- Average number of tickets sold by the group

/-- Represents the charitable association -/
structure Association where
  male : GroupAverage
  female : GroupAverage
  nonBinary : GroupAverage

/-- The ratio of male to female to non-binary members is 2:3:5 -/
def memberRatio (a : Association) : Prop :=
  a.male.count = 2 * a.female.count / 3 ∧
  a.nonBinary.count = 5 * a.female.count / 3

/-- The average number of tickets sold by all members is 66 -/
def totalAverage (a : Association) : Prop :=
  (a.male.count * a.male.average + a.female.count * a.female.average + a.nonBinary.count * a.nonBinary.average) /
  (a.male.count + a.female.count + a.nonBinary.count) = 66

/-- Main theorem: Given the conditions, prove that the average number of tickets sold by male members is 100 -/
theorem male_average_is_100 (a : Association)
  (h_ratio : memberRatio a)
  (h_total_avg : totalAverage a)
  (h_female_avg : a.female.average = 70)
  (h_nonbinary_avg : a.nonBinary.average = 50) :
  a.male.average = 100 := by
  sorry

end NUMINAMATH_CALUDE_male_average_is_100_l87_8723


namespace NUMINAMATH_CALUDE_contrapositive_real_roots_l87_8789

theorem contrapositive_real_roots (m : ℝ) :
  (¬(∃ x : ℝ, x^2 = m) → m < 0) ↔
  (m ≥ 0 → ∃ x : ℝ, x^2 = m) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_real_roots_l87_8789


namespace NUMINAMATH_CALUDE_reciprocal_sum_one_l87_8755

theorem reciprocal_sum_one (x y z : ℕ+) (h_sum : (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = 1) 
  (h_order : x ≤ y ∧ y ≤ z) : 
  (x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 3 ∧ y = 3 ∧ z = 3) := by
  sorry

#check reciprocal_sum_one

end NUMINAMATH_CALUDE_reciprocal_sum_one_l87_8755


namespace NUMINAMATH_CALUDE_shaded_area_square_configuration_l87_8769

/-- The area of the shaded region in a geometric configuration where a 4-inch square adjoins a 12-inch square -/
theorem shaded_area_square_configuration : 
  -- Large square side length
  ∀ (large_side : ℝ) 
  -- Small square side length
  (small_side : ℝ),
  -- Conditions
  large_side = 12 →
  small_side = 4 →
  -- The shaded area is the difference between the small square's area and the area of a triangle
  let shaded_area := small_side^2 - (1/2 * (3/4 * small_side) * small_side)
  -- Theorem statement
  shaded_area = 10 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_square_configuration_l87_8769


namespace NUMINAMATH_CALUDE_cube_root_of_product_with_nested_roots_l87_8777

theorem cube_root_of_product_with_nested_roots (N : ℝ) (h : N > 1) :
  (N * (N * N^(1/3))^(1/2))^(1/3) = N^(5/9) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_with_nested_roots_l87_8777


namespace NUMINAMATH_CALUDE_skateboarder_distance_is_3720_l87_8703

/-- Represents the skateboarder's journey -/
structure SkateboarderJourney where
  initial_distance : ℕ  -- Distance covered in the first second
  distance_increase : ℕ  -- Increase in distance each second on the ramp
  ramp_time : ℕ  -- Time spent on the ramp
  flat_time : ℕ  -- Time spent on the flat stretch

/-- Calculates the total distance traveled by the skateboarder -/
def total_distance (journey : SkateboarderJourney) : ℕ :=
  let ramp_distance := journey.ramp_time * (journey.initial_distance + (journey.ramp_time - 1) * journey.distance_increase / 2)
  let final_speed := journey.initial_distance + (journey.ramp_time - 1) * journey.distance_increase
  let flat_distance := final_speed * journey.flat_time
  ramp_distance + flat_distance

/-- Theorem stating that the total distance traveled is 3720 meters -/
theorem skateboarder_distance_is_3720 (journey : SkateboarderJourney) 
  (h1 : journey.initial_distance = 10)
  (h2 : journey.distance_increase = 9)
  (h3 : journey.ramp_time = 20)
  (h4 : journey.flat_time = 10) : 
  total_distance journey = 3720 := by
  sorry

end NUMINAMATH_CALUDE_skateboarder_distance_is_3720_l87_8703


namespace NUMINAMATH_CALUDE_perpendicular_lines_angle_relation_l87_8743

-- Define a dihedral angle
structure DihedralAngle where
  plane_angle : ℝ
  -- Add other necessary properties

-- Define a point inside a dihedral angle
structure PointInDihedralAngle where
  dihedral : DihedralAngle
  -- Add other necessary properties

-- Define the angle formed by perpendicular lines
def perpendicularLinesAngle (p : PointInDihedralAngle) : ℝ := sorry

-- Define the relationship between angles
def isEqualOrComplementary (a b : ℝ) : Prop :=
  a = b ∨ a + b = Real.pi / 2

-- Theorem statement
theorem perpendicular_lines_angle_relation (p : PointInDihedralAngle) :
  isEqualOrComplementary (perpendicularLinesAngle p) p.dihedral.plane_angle := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_angle_relation_l87_8743


namespace NUMINAMATH_CALUDE_xyz_expression_bounds_l87_8739

theorem xyz_expression_bounds (x y z : ℝ) 
  (non_neg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (sum_one : x + y + z = 1) : 
  0 ≤ x*y + y*z + z*x - 3*x*y*z ∧ x*y + y*z + z*x - 3*x*y*z ≤ 1/4 := by
sorry

end NUMINAMATH_CALUDE_xyz_expression_bounds_l87_8739


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l87_8741

theorem magic_8_ball_probability : 
  let n : ℕ := 7  -- total number of questions
  let k : ℕ := 4  -- number of positive answers we're interested in
  let p : ℚ := 1/3  -- probability of a positive answer for each question
  Nat.choose n k * p^k * (1-p)^(n-k) = 280/2187 := by sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l87_8741


namespace NUMINAMATH_CALUDE_problem_statement_l87_8751

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 4) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 74.0625 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l87_8751


namespace NUMINAMATH_CALUDE_donut_combinations_l87_8778

/-- The number of ways to distribute n identical objects into k distinct boxes -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The total number of donuts Josh needs to buy -/
def total_donuts : ℕ := 8

/-- The number of different types of donuts -/
def donut_types : ℕ := 5

/-- The number of donuts Josh must buy of the first type -/
def first_type_min : ℕ := 2

/-- The number of donuts Josh must buy of each other type -/
def other_types_min : ℕ := 1

/-- The number of remaining donuts to distribute after meeting minimum requirements -/
def remaining_donuts : ℕ := total_donuts - (first_type_min + (donut_types - 1) * other_types_min)

theorem donut_combinations : stars_and_bars remaining_donuts donut_types = 15 := by
  sorry

end NUMINAMATH_CALUDE_donut_combinations_l87_8778


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l87_8722

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * log x - x^2

-- State the theorem
theorem monotone_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (exp 1), MonotoneOn (f a) (Set.Icc 1 (exp 1))) ↔ a ≥ exp 1 :=
by sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l87_8722


namespace NUMINAMATH_CALUDE_daps_to_dips_l87_8763

/-- The number of daps equivalent to one dop -/
def daps_per_dop : ℚ := 5 / 4

/-- The number of dops equivalent to one dip -/
def dops_per_dip : ℚ := 3 / 11

/-- The number of dips we want to convert to daps -/
def target_dips : ℚ := 66

/-- Theorem stating the equivalence between daps and dips -/
theorem daps_to_dips : daps_per_dop * dops_per_dip⁻¹ * target_dips = 45 / 2 :=
by sorry

end NUMINAMATH_CALUDE_daps_to_dips_l87_8763


namespace NUMINAMATH_CALUDE_complex_fraction_real_l87_8715

theorem complex_fraction_real (m : ℝ) : 
  (((1 : ℂ) + m * Complex.I) / ((1 : ℂ) + Complex.I)).im = 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l87_8715


namespace NUMINAMATH_CALUDE_monotonic_function_k_range_l87_8733

theorem monotonic_function_k_range (k : ℝ) :
  (∀ x ≥ 1, Monotone (fun x : ℝ ↦ 4 * x^2 - k * x - 8)) →
  k ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_k_range_l87_8733


namespace NUMINAMATH_CALUDE_largest_even_number_less_than_150_div_9_l87_8714

theorem largest_even_number_less_than_150_div_9 :
  ∃ (x : ℕ), 
    x % 2 = 0 ∧ 
    9 * x < 150 ∧ 
    ∀ (y : ℕ), y % 2 = 0 → 9 * y < 150 → y ≤ x ∧
    x = 16 :=
by sorry

end NUMINAMATH_CALUDE_largest_even_number_less_than_150_div_9_l87_8714


namespace NUMINAMATH_CALUDE_teacher_selection_theorem_l87_8764

/-- The number of male teachers -/
def num_male_teachers : ℕ := 4

/-- The number of female teachers -/
def num_female_teachers : ℕ := 3

/-- The total number of teachers to be selected -/
def num_selected : ℕ := 3

/-- The number of ways to select teachers with both genders represented -/
def num_ways_to_select : ℕ := 30

theorem teacher_selection_theorem :
  (num_ways_to_select = (Nat.choose num_male_teachers 2 * Nat.choose num_female_teachers 1) +
                        (Nat.choose num_male_teachers 1 * Nat.choose num_female_teachers 2)) ∧
  (num_ways_to_select = Nat.choose (num_male_teachers + num_female_teachers) num_selected -
                        Nat.choose num_male_teachers num_selected -
                        Nat.choose num_female_teachers num_selected) := by
  sorry

end NUMINAMATH_CALUDE_teacher_selection_theorem_l87_8764


namespace NUMINAMATH_CALUDE_largest_square_multiple_18_under_500_l87_8706

theorem largest_square_multiple_18_under_500 : ∃ n : ℕ, 
  n^2 = 324 ∧ 
  18 ∣ n^2 ∧ 
  n^2 < 500 ∧ 
  ∀ m : ℕ, (m^2 > n^2 ∧ 18 ∣ m^2) → m^2 ≥ 500 :=
by sorry

end NUMINAMATH_CALUDE_largest_square_multiple_18_under_500_l87_8706


namespace NUMINAMATH_CALUDE_hen_count_l87_8761

theorem hen_count (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) 
  (h1 : total_animals = 50)
  (h2 : total_feet = 144)
  (h3 : hen_feet = 2)
  (h4 : cow_feet = 4) :
  ∃ (hens : ℕ) (cows : ℕ),
    hens + cows = total_animals ∧
    hens * hen_feet + cows * cow_feet = total_feet ∧
    hens = 28 :=
by sorry

end NUMINAMATH_CALUDE_hen_count_l87_8761


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l87_8754

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) (h2 : α ≠ β) 
  (h3 : parallel m n) (h4 : perpendicular m α) : 
  perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l87_8754


namespace NUMINAMATH_CALUDE_nested_sqrt_solution_l87_8726

/-- The positive solution to the nested square root equation -/
theorem nested_sqrt_solution : 
  ∃! (x : ℝ), x > 0 ∧ 
  (∃ (z : ℝ), z > 0 ∧ z = Real.sqrt (x + z)) ∧
  (∃ (y : ℝ), y > 0 ∧ y = Real.sqrt (x * y)) ∧
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_nested_sqrt_solution_l87_8726


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l87_8712

theorem complex_fraction_simplification :
  (Complex.I + 3) / (Complex.I + 1) = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l87_8712


namespace NUMINAMATH_CALUDE_sugar_solution_mixing_l87_8750

/-- Calculates the percentage of sugar in the resulting solution after replacing
    a portion of an initial sugar solution with another sugar solution. -/
theorem sugar_solution_mixing (initial_sugar_percentage : ℝ)
                               (replacement_portion : ℝ)
                               (replacement_sugar_percentage : ℝ) :
  initial_sugar_percentage = 8 →
  replacement_portion = 1/4 →
  replacement_sugar_percentage = 40 →
  let remaining_portion := 1 - replacement_portion
  let initial_sugar := initial_sugar_percentage * remaining_portion
  let replacement_sugar := replacement_sugar_percentage * replacement_portion
  let final_sugar_percentage := initial_sugar + replacement_sugar
  final_sugar_percentage = 16 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_mixing_l87_8750


namespace NUMINAMATH_CALUDE_roots_modulus_one_preserved_l87_8756

theorem roots_modulus_one_preserved (a b c : ℂ) :
  (∃ α β γ : ℂ, (α^3 + a*α^2 + b*α + c = 0) ∧ 
                (β^3 + a*β^2 + b*β + c = 0) ∧ 
                (γ^3 + a*γ^2 + b*γ + c = 0) ∧
                (Complex.abs α = 1) ∧ (Complex.abs β = 1) ∧ (Complex.abs γ = 1)) →
  (∃ x y z : ℂ, (x^3 + Complex.abs a*x^2 + Complex.abs b*x + Complex.abs c = 0) ∧ 
                (y^3 + Complex.abs a*y^2 + Complex.abs b*y + Complex.abs c = 0) ∧ 
                (z^3 + Complex.abs a*z^2 + Complex.abs b*z + Complex.abs c = 0) ∧
                (Complex.abs x = 1) ∧ (Complex.abs y = 1) ∧ (Complex.abs z = 1)) :=
by sorry

end NUMINAMATH_CALUDE_roots_modulus_one_preserved_l87_8756


namespace NUMINAMATH_CALUDE_remainder_theorem_l87_8758

theorem remainder_theorem : ∃ q : ℕ, 2^300 + 300 = (2^150 + 2^75 + 1) * q + 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l87_8758


namespace NUMINAMATH_CALUDE_no_valid_integers_l87_8799

theorem no_valid_integers : ¬∃ (n : ℤ), ∃ (y : ℤ), 
  (n^2 - 21*n + 110 = y^2) ∧ (∃ (k : ℤ), n = 4*k) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_integers_l87_8799


namespace NUMINAMATH_CALUDE_double_elimination_64_teams_games_range_l87_8787

/-- Represents a double-elimination tournament --/
structure DoubleEliminationTournament where
  num_teams : ℕ
  no_ties : Bool

/-- The minimum number of games required to determine a champion in a double-elimination tournament --/
def min_games (t : DoubleEliminationTournament) : ℕ := sorry

/-- The maximum number of games required to determine a champion in a double-elimination tournament --/
def max_games (t : DoubleEliminationTournament) : ℕ := sorry

/-- Theorem stating the range of games required for a 64-team double-elimination tournament --/
theorem double_elimination_64_teams_games_range (t : DoubleEliminationTournament) 
  (h1 : t.num_teams = 64) (h2 : t.no_ties = true) : 
  min_games t = 96 ∧ max_games t = 97 := by sorry

end NUMINAMATH_CALUDE_double_elimination_64_teams_games_range_l87_8787


namespace NUMINAMATH_CALUDE_largest_integer_less_than_x_l87_8745

theorem largest_integer_less_than_x (x : ℤ) 
  (h1 : 5 < x ∧ x < 21)
  (h2 : 7 < x ∧ x < 18)
  (h3 : x < 13)
  (h4 : 12 > x ∧ x > 9)
  (h5 : x + 1 < 13) :
  ∃ (y : ℤ), x > y ∧ ∀ (z : ℤ), x > z → z ≤ y ∧ y = 9 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_x_l87_8745


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l87_8794

theorem arithmetic_square_root_of_nine (x : ℝ) :
  (x ≥ 0 ∧ x ^ 2 = 9) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l87_8794


namespace NUMINAMATH_CALUDE_b_investment_is_8000_l87_8702

/-- Represents a partnership with three partners -/
structure Partnership where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  a_profit : ℝ
  b_profit : ℝ

/-- The profit share is proportional to the investment -/
def profit_proportional (p : Partnership) : Prop :=
  p.a_profit / p.a_investment = p.b_profit / p.b_investment

/-- Theorem stating that given the conditions, b's investment is $8000 -/
theorem b_investment_is_8000 (p : Partnership) 
  (h1 : p.a_investment = 7000)
  (h2 : p.c_investment = 18000)
  (h3 : p.a_profit = 560)
  (h4 : p.b_profit = 880)
  (h5 : profit_proportional p) : 
  p.b_investment = 8000 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_is_8000_l87_8702


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l87_8704

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l87_8704


namespace NUMINAMATH_CALUDE_matrix_commutator_similarity_l87_8719

/-- Given n×n complex matrices A and B where A^2 = B^2, there exists an invertible n×n complex matrix S such that S(AB - BA) = (BA - AB)S. -/
theorem matrix_commutator_similarity {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℂ) 
  (h : A ^ 2 = B ^ 2) : 
  ∃ S : Matrix (Fin n) (Fin n) ℂ, IsUnit S ∧ S * (A * B - B * A) = (B * A - A * B) * S := by
  sorry

end NUMINAMATH_CALUDE_matrix_commutator_similarity_l87_8719


namespace NUMINAMATH_CALUDE_circumscribed_odd_equal_sides_is_regular_l87_8793

/-- A polygon with an odd number of sides -/
structure OddPolygon where
  n : ℕ
  vertices : Fin (2 * n + 1) → ℝ × ℝ

/-- A circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A polygon is circumscribed around a circle if all its sides are tangent to the circle -/
def isCircumscribed (p : OddPolygon) (c : Circle) : Prop := sorry

/-- All sides of a polygon have equal length -/
def hasEqualSides (p : OddPolygon) : Prop := sorry

/-- A polygon is regular if all its sides have equal length and all its angles are equal -/
def isRegular (p : OddPolygon) : Prop := sorry

/-- Main theorem: A circumscribed polygon with an odd number of sides and all sides of equal length is regular -/
theorem circumscribed_odd_equal_sides_is_regular 
  (p : OddPolygon) (c : Circle) 
  (h1 : isCircumscribed p c) 
  (h2 : hasEqualSides p) : 
  isRegular p := by sorry

end NUMINAMATH_CALUDE_circumscribed_odd_equal_sides_is_regular_l87_8793


namespace NUMINAMATH_CALUDE_simplify_fraction_division_l87_8785

theorem simplify_fraction_division (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 3) * (x - 4) * (x - 2)) := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_division_l87_8785


namespace NUMINAMATH_CALUDE_largest_integer_solution_l87_8710

theorem largest_integer_solution (x : ℤ) : x ≤ 2 ↔ x / 3 + 4 / 5 < 5 / 3 := by sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l87_8710


namespace NUMINAMATH_CALUDE_parabola_circle_area_ratio_l87_8729

/-- The ratio of areas S1 to S2 for a parabola and tangent circle -/
theorem parabola_circle_area_ratio 
  (d : ℝ) 
  (hd : d > 0) : 
  let K : ℝ → ℝ := fun x ↦ (1/d) * x^2
  let P : ℝ × ℝ := (d, d)
  let Q : ℝ × ℝ := (0, d)
  let S1 : ℝ := ∫ x in (0)..(d), (d - K x)
  let S2 : ℝ := ∫ x in (0)..(d), (d - K x)
  S1 / S2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_circle_area_ratio_l87_8729


namespace NUMINAMATH_CALUDE_cheetah_speed_calculation_l87_8771

/-- The top speed of a cheetah in miles per hour -/
def cheetah_speed : ℝ := 60

/-- The top speed of a gazelle in miles per hour -/
def gazelle_speed : ℝ := 40

/-- Conversion factor from miles per hour to feet per second -/
def mph_to_fps : ℝ := 1.5

/-- Time taken for the cheetah to catch up to the gazelle in seconds -/
def catch_up_time : ℝ := 7

/-- Initial distance between the cheetah and the gazelle in feet -/
def initial_distance : ℝ := 210

theorem cheetah_speed_calculation :
  cheetah_speed * mph_to_fps - gazelle_speed * mph_to_fps = initial_distance / catch_up_time :=
by sorry

end NUMINAMATH_CALUDE_cheetah_speed_calculation_l87_8771


namespace NUMINAMATH_CALUDE_equation_solution_l87_8795

theorem equation_solution : ∃ (z₁ z₂ : ℂ), 
  z₁ = (-1 + Complex.I * Real.sqrt 21) / 2 ∧
  z₂ = (-1 - Complex.I * Real.sqrt 21) / 2 ∧
  ∀ x : ℂ, (4 * x^2 + 3 * x + 1) / (x - 2) = 2 * x + 5 ↔ x = z₁ ∨ x = z₂ := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l87_8795


namespace NUMINAMATH_CALUDE_total_feed_amount_l87_8767

/-- Represents the total amount of dog feed mixed -/
def total_feed (cheap_feed expensive_feed : ℝ) : ℝ := cheap_feed + expensive_feed

/-- Represents the total cost of the mixed feed -/
def total_cost (cheap_feed expensive_feed : ℝ) : ℝ :=
  0.18 * cheap_feed + 0.53 * expensive_feed

/-- The theorem stating the total amount of feed mixed -/
theorem total_feed_amount :
  ∃ (expensive_feed : ℝ),
    total_feed 17 expensive_feed = 35 ∧
    total_cost 17 expensive_feed = 0.36 * total_feed 17 expensive_feed :=
sorry

end NUMINAMATH_CALUDE_total_feed_amount_l87_8767


namespace NUMINAMATH_CALUDE_distance_between_runners_l87_8772

/-- The distance between two runners at the end of a 1 km race -/
theorem distance_between_runners (H J : ℝ) (t : ℝ) 
  (h_distance : 1000 = H * t) 
  (j_distance : 152 = J * t) : 
  1000 - 152 = 848 := by sorry

end NUMINAMATH_CALUDE_distance_between_runners_l87_8772


namespace NUMINAMATH_CALUDE_bakery_sugar_amount_l87_8734

/-- Given the ratios of ingredients in a bakery storage room, prove the amount of sugar. -/
theorem bakery_sugar_amount 
  (sugar flour baking_soda : ℚ) 
  (h1 : sugar / flour = 5 / 6)
  (h2 : flour / baking_soda = 10 / 1)
  (h3 : flour / (baking_soda + 60) = 8 / 1) :
  sugar = 2000 := by
  sorry

#check bakery_sugar_amount

end NUMINAMATH_CALUDE_bakery_sugar_amount_l87_8734


namespace NUMINAMATH_CALUDE_ellipse_a_range_l87_8732

/-- Represents an ellipse with the given equation and foci on the x-axis -/
structure Ellipse (a : ℝ) :=
  (eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / (a + 6) = 1)
  (foci_on_x : True)  -- We don't need to formalize this condition for the proof

/-- The range of a for which the given equation represents an ellipse with foci on the x-axis -/
theorem ellipse_a_range (a : ℝ) (e : Ellipse a) : a > 3 ∨ (-6 < a ∧ a < -2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_a_range_l87_8732


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_l87_8717

theorem log_equality_implies_ratio (p q : ℝ) (hp : 0 < p) (hq : 0 < q) :
  (Real.log p / Real.log 4 = Real.log q / Real.log 8) ∧
  (Real.log p / Real.log 4 = Real.log (p + q) / Real.log 18) →
  q / p = Real.sqrt p :=
by sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_l87_8717


namespace NUMINAMATH_CALUDE_cos_x_plus_pi_sixth_l87_8718

theorem cos_x_plus_pi_sixth (x : ℝ) (h : Real.sin (π / 3 - x) = -3 / 5) : 
  Real.cos (x + π / 6) = -3 / 5 := by
sorry

end NUMINAMATH_CALUDE_cos_x_plus_pi_sixth_l87_8718


namespace NUMINAMATH_CALUDE_seed_germination_probabilities_l87_8711

/-- The number of seeds in each pit -/
def seeds_per_pit : ℕ := 3

/-- The probability of a single seed germinating -/
def germination_prob : ℝ := 0.5

/-- The number of pits -/
def num_pits : ℕ := 3

/-- The probability that at least one seed germinates in a pit -/
def prob_at_least_one_germinates : ℝ := 1 - (1 - germination_prob) ^ seeds_per_pit

/-- The probability that exactly two pits need replanting -/
def prob_exactly_two_need_replanting : ℝ := 
  (num_pits.choose 2) * (1 - prob_at_least_one_germinates) ^ 2 * prob_at_least_one_germinates

/-- The probability that at least one pit needs replanting -/
def prob_at_least_one_needs_replanting : ℝ := 1 - prob_at_least_one_germinates ^ num_pits

theorem seed_germination_probabilities :
  (prob_at_least_one_germinates = 0.875) ∧
  (prob_exactly_two_need_replanting = 0.713) ∧
  (prob_at_least_one_needs_replanting = 0.330) := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_probabilities_l87_8711


namespace NUMINAMATH_CALUDE_worker_earnings_l87_8792

theorem worker_earnings
  (regular_rate : ℝ)
  (total_surveys : ℕ)
  (cellphone_rate_increase : ℝ)
  (cellphone_surveys : ℕ)
  (h1 : regular_rate = 30)
  (h2 : total_surveys = 100)
  (h3 : cellphone_rate_increase = 0.2)
  (h4 : cellphone_surveys = 50) :
  let cellphone_rate := regular_rate * (1 + cellphone_rate_increase)
  let regular_surveys := total_surveys - cellphone_surveys
  let total_earnings := regular_rate * regular_surveys + cellphone_rate * cellphone_surveys
  total_earnings = 3300 :=
by sorry

end NUMINAMATH_CALUDE_worker_earnings_l87_8792


namespace NUMINAMATH_CALUDE_total_carrots_is_40_l87_8728

/-- The number of carrots grown by Joan -/
def joan_carrots : ℕ := 29

/-- The number of carrots grown by Jessica -/
def jessica_carrots : ℕ := 11

/-- The total number of carrots grown by Joan and Jessica -/
def total_carrots : ℕ := joan_carrots + jessica_carrots

/-- Theorem stating that the total number of carrots grown is 40 -/
theorem total_carrots_is_40 : total_carrots = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_carrots_is_40_l87_8728


namespace NUMINAMATH_CALUDE_negation_of_p_l87_8796

variable (I : Set ℝ)

def p : Prop := ∀ x ∈ I, x^3 - x^2 + 1 ≤ 0

theorem negation_of_p : 
  ¬p I ↔ ∃ x ∈ I, x^3 - x^2 + 1 > 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l87_8796


namespace NUMINAMATH_CALUDE_x_minus_q_in_terms_of_q_l87_8770

theorem x_minus_q_in_terms_of_q (x q : ℝ) (h1 : |x - 3| = q) (h2 : x < 3) : x - q = 3 - 2*q := by
  sorry

end NUMINAMATH_CALUDE_x_minus_q_in_terms_of_q_l87_8770


namespace NUMINAMATH_CALUDE_camden_rico_dog_fraction_l87_8776

/-- Proves that the fraction of dogs Camden bought compared to Rico is 3/4 -/
theorem camden_rico_dog_fraction :
  let justin_dogs : ℕ := 14
  let rico_dogs : ℕ := justin_dogs + 10
  let camden_dog_legs : ℕ := 72
  let legs_per_dog : ℕ := 4
  let camden_dogs : ℕ := camden_dog_legs / legs_per_dog
  (camden_dogs : ℚ) / rico_dogs = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_camden_rico_dog_fraction_l87_8776


namespace NUMINAMATH_CALUDE_horseshoe_division_l87_8779

/-- Represents a paper horseshoe with holes -/
structure Horseshoe where
  holes : ℕ

/-- Represents a cut on the horseshoe -/
inductive Cut
| straight : Cut

/-- Represents the state of the horseshoe after cuts -/
structure HorseshoeState where
  pieces : ℕ
  holesPerPiece : ℕ

/-- Function to apply a cut to the horseshoe -/
def applyCut (h : Horseshoe) (c : Cut) (s : HorseshoeState) : HorseshoeState :=
  sorry

/-- Function to rearrange pieces -/
def rearrange (s : HorseshoeState) : HorseshoeState :=
  sorry

/-- Theorem stating that a horseshoe can be divided into n parts with n holes using two straight cuts -/
theorem horseshoe_division (h : Horseshoe) :
  ∃ (c1 c2 : Cut), ∃ (s1 s2 s3 : HorseshoeState),
    s1 = applyCut h c1 {pieces := 1, holesPerPiece := h.holes} ∧
    s2 = rearrange s1 ∧
    s3 = applyCut h c2 s2 ∧
    s3.pieces = h.holes ∧
    s3.holesPerPiece = 1 :=
  sorry

end NUMINAMATH_CALUDE_horseshoe_division_l87_8779


namespace NUMINAMATH_CALUDE_equation_solution_l87_8731

theorem equation_solution (a : ℝ) : 
  (∀ x, 2*(x+1) = 3*(x-1) ↔ x = a+2) →
  (∃! x, 2*(2*(x+3) - 3*(x-a)) = 3*a ∧ x = 10) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l87_8731


namespace NUMINAMATH_CALUDE_vector_problem_l87_8759

/-- Given points A, B, C in ℝ², and vectors a, b, c, prove the following statements. -/
theorem vector_problem (A B C M N : ℝ × ℝ) (a b c : ℝ × ℝ) :
  A = (-2, 4) →
  B = (3, -1) →
  C = (-3, -4) →
  a = B - A →
  b = C - B →
  c = A - C →
  M - C = 3 • c →
  N - C = -2 • b →
  (3 • a + b - 3 • c = (6, -42)) ∧
  (a = -b - c) ∧
  (M = (0, 20) ∧ N = (9, 2) ∧ N - M = (9, -18)) := by
sorry


end NUMINAMATH_CALUDE_vector_problem_l87_8759


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l87_8768

theorem smaller_root_of_equation (x : ℝ) : 
  (x - 5/8) * (x - 5/8) + (x - 5/8) * (x - 2/3) = 0 → 
  (∃ y : ℝ, (y - 5/8) * (y - 5/8) + (y - 5/8) * (y - 2/3) = 0 ∧ y ≤ x) → 
  x = 29/48 := by
sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l87_8768


namespace NUMINAMATH_CALUDE_number_problem_l87_8725

theorem number_problem : ∃ n : ℝ, n - (1002 / 20.04) = 2984 ∧ n = 3034 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l87_8725


namespace NUMINAMATH_CALUDE_cryptarithm_solution_exists_and_unique_l87_8720

/-- Represents a cryptarithm solution -/
structure CryptarithmSolution where
  A : Nat
  H : Nat
  J : Nat
  O : Nat
  K : Nat
  E : Nat

/-- Checks if all digits in the solution are unique -/
def uniqueDigits (sol : CryptarithmSolution) : Prop :=
  sol.A ≠ sol.H ∧ sol.A ≠ sol.J ∧ sol.A ≠ sol.O ∧ sol.A ≠ sol.K ∧ sol.A ≠ sol.E ∧
  sol.H ≠ sol.J ∧ sol.H ≠ sol.O ∧ sol.H ≠ sol.K ∧ sol.H ≠ sol.E ∧
  sol.J ≠ sol.O ∧ sol.J ≠ sol.K ∧ sol.J ≠ sol.E ∧
  sol.O ≠ sol.K ∧ sol.O ≠ sol.E ∧
  sol.K ≠ sol.E

/-- Checks if the solution satisfies the cryptarithm equation -/
def satisfiesCryptarithm (sol : CryptarithmSolution) : Prop :=
  (100001 * sol.A + 11010 * sol.H) / (10 * sol.H + sol.A) = 
  1000 * sol.J + 100 * sol.O + 10 * sol.K + sol.E

/-- The main theorem stating that there exists a unique solution to the cryptarithm -/
theorem cryptarithm_solution_exists_and_unique :
  ∃! sol : CryptarithmSolution,
    uniqueDigits sol ∧
    satisfiesCryptarithm sol ∧
    sol.A = 3 ∧ sol.H = 7 ∧ sol.J = 5 ∧ sol.O = 1 ∧ sol.K = 6 ∧ sol.E = 9 :=
sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_exists_and_unique_l87_8720


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l87_8798

theorem complex_fraction_equality : (Complex.I : ℂ) ^ 2 = -1 → (2 + 2 * Complex.I) / (1 - Complex.I) = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l87_8798


namespace NUMINAMATH_CALUDE_expectation_of_specific_distribution_l87_8701

/-- The expected value of a random variable with a specific probability distribution -/
theorem expectation_of_specific_distribution (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h_sum : x + y + x = 1) :
  let ξ : ℝ → ℝ := fun ω => 
    if ω < x then 1
    else if ω < x + y then 2
    else 3
  2 = ∫ ω in Set.Icc 0 1, ξ ω ∂volume :=
by sorry

end NUMINAMATH_CALUDE_expectation_of_specific_distribution_l87_8701


namespace NUMINAMATH_CALUDE_shared_course_count_is_24_l87_8705

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of courses available -/
def total_courses : ℕ := 4

/-- The number of courses each person chooses -/
def courses_per_person : ℕ := 2

/-- The number of ways in which exactly one course is chosen by both people -/
def shared_course_count : ℕ := 
  choose total_courses courses_per_person * choose total_courses courses_per_person -
  choose total_courses courses_per_person -
  choose total_courses courses_per_person

theorem shared_course_count_is_24 : shared_course_count = 24 := by sorry

end NUMINAMATH_CALUDE_shared_course_count_is_24_l87_8705


namespace NUMINAMATH_CALUDE_probability_club_heart_king_l87_8746

theorem probability_club_heart_king (total_cards : ℕ) (clubs : ℕ) (hearts : ℕ) (kings : ℕ) :
  total_cards = 52 →
  clubs = 13 →
  hearts = 13 →
  kings = 4 →
  (clubs / total_cards) * (hearts / (total_cards - 1)) * (kings / (total_cards - 2)) = 13 / 2550 := by
  sorry

end NUMINAMATH_CALUDE_probability_club_heart_king_l87_8746


namespace NUMINAMATH_CALUDE_a_is_perfect_square_l87_8773

/-- Sequence c_n defined recursively -/
def c : ℕ → ℤ
  | 0 => 1
  | 1 => 0
  | 2 => 2005
  | (n + 3) => -3 * c (n + 1) - 4 * c n + 2008

/-- Sequence a_n defined in terms of c_n -/
def a (n : ℕ) : ℤ :=
  5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4^n * 2004 * 501

/-- Theorem stating that a_n is a perfect square for n > 2 -/
theorem a_is_perfect_square (n : ℕ) (h : n > 2) : ∃ k : ℤ, a n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_a_is_perfect_square_l87_8773


namespace NUMINAMATH_CALUDE_roots_sum_logarithmic_equation_l87_8780

theorem roots_sum_logarithmic_equation (m : ℝ) :
  ∃ x₁ x₂ : ℝ, (Real.log (|x₁ - 2|) = m ∧ Real.log (|x₂ - 2|) = m) → x₁ + x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_logarithmic_equation_l87_8780


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l87_8784

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l87_8784


namespace NUMINAMATH_CALUDE_division_problem_l87_8766

theorem division_problem : (88 : ℚ) / ((4 : ℚ) / 2) = 44 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l87_8766


namespace NUMINAMATH_CALUDE_wedding_guests_l87_8748

theorem wedding_guests (total : ℕ) 
  (h1 : (83 : ℚ) / 100 * total + (9 : ℚ) / 100 * total + 16 = total) : 
  total = 200 := by
  sorry

end NUMINAMATH_CALUDE_wedding_guests_l87_8748


namespace NUMINAMATH_CALUDE_total_stamps_l87_8783

theorem total_stamps (harry_stamps : ℕ) (sister_stamps : ℕ) : 
  harry_stamps = 180 → sister_stamps = 60 → harry_stamps + sister_stamps = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_total_stamps_l87_8783


namespace NUMINAMATH_CALUDE_largest_expression_l87_8774

theorem largest_expression : 
  let e1 := 992 * 999 + 999
  let e2 := 993 * 998 + 998
  let e3 := 994 * 997 + 997
  let e4 := 995 * 996 + 996
  (e4 > e1) ∧ (e4 > e2) ∧ (e4 > e3) := by
sorry

end NUMINAMATH_CALUDE_largest_expression_l87_8774


namespace NUMINAMATH_CALUDE_circle_equation_l87_8730

theorem circle_equation (x y : ℝ) :
  (x + 2)^2 + (y - 2)^2 = 25 ↔ 
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-2, 2) ∧ 
    radius = 5 ∧ 
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l87_8730


namespace NUMINAMATH_CALUDE_median_inequality_l87_8709

-- Define a right triangle with medians
structure RightTriangle where
  c : ℝ  -- length of hypotenuse
  sa : ℝ  -- length of median to one leg
  sb : ℝ  -- length of median to the other leg
  c_pos : c > 0  -- hypotenuse length is positive

-- State the theorem
theorem median_inequality (t : RightTriangle) : 
  (3/2) * t.c < t.sa + t.sb ∧ t.sa + t.sb ≤ (Real.sqrt 10 / 2) * t.c := by
  sorry

end NUMINAMATH_CALUDE_median_inequality_l87_8709


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_nine_l87_8742

/-- A quadratic function with roots satisfying specific conditions -/
structure QuadraticWithSpecialRoots where
  a : ℝ
  b : ℝ
  m : ℝ
  n : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_roots : m ≠ n ∧ m > 0 ∧ n > 0
  h_vieta : m + n = a ∧ m * n = b
  h_arithmetic : (m - n = n - (-2)) ∨ (n - m = m - (-2))
  h_geometric : (m / n = n / (-2)) ∨ (n / m = m / (-2))

/-- The sum of coefficients a and b equals 9 -/
theorem sum_of_coefficients_is_nine (q : QuadraticWithSpecialRoots) : q.a + q.b = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_nine_l87_8742


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_A_complement_BC_l87_8786

/-- The universal set U is ℝ -/
def U : Set ℝ := Set.univ

/-- Set A: { x | y = ln(x² - 9) } -/
def A : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 9)}

/-- Set B: { x | (x - 7)/(x + 1) > 0 } -/
def B : Set ℝ := {x | (x - 7) / (x + 1) > 0}

/-- Set C: { x | |x - 2| < 4 } -/
def C : Set ℝ := {x | |x - 2| < 4}

/-- Theorem 1: A ∩ B = { x | x < -3 or x > 7 } -/
theorem intersection_A_B : A ∩ B = {x | x < -3 ∨ x > 7} := by sorry

/-- Theorem 2: A ∩ (U \ (B ∩ C)) = { x | x < -3 or x > 3 } -/
theorem intersection_A_complement_BC : A ∩ (U \ (B ∩ C)) = {x | x < -3 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_A_complement_BC_l87_8786


namespace NUMINAMATH_CALUDE_a_investment_value_l87_8765

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  total_profit : ℝ
  c_profit_share : ℝ

/-- Theorem stating that given the conditions of the partnership,
    a's investment is $45,000 --/
theorem a_investment_value (p : Partnership)
  (hb : p.b_investment = 63000)
  (hc : p.c_investment = 72000)
  (hp : p.total_profit = 60000)
  (hcs : p.c_profit_share = 24000) :
  p.a_investment = 45000 :=
sorry

end NUMINAMATH_CALUDE_a_investment_value_l87_8765


namespace NUMINAMATH_CALUDE_ball_probabilities_solution_l87_8790

/-- Represents the color of a ball -/
inductive Color
  | Red
  | Black
  | Yellow
  | Green

/-- Represents the probabilities of drawing balls of different colors -/
structure BallProbabilities where
  red : ℚ
  black : ℚ
  yellow : ℚ
  green : ℚ

/-- The conditions of the problem -/
def problem_conditions (p : BallProbabilities) : Prop :=
  p.red + p.black + p.yellow + p.green = 1 ∧
  p.red = 1/3 ∧
  p.black + p.yellow = 5/12 ∧
  p.yellow + p.green = 5/12

/-- The theorem stating the solution -/
theorem ball_probabilities_solution :
  ∃ (p : BallProbabilities), problem_conditions p ∧ 
    p.black = 1/4 ∧ p.yellow = 1/6 ∧ p.green = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_solution_l87_8790


namespace NUMINAMATH_CALUDE_error_clock_correct_time_l87_8752

/-- Represents a 12-hour digital clock with a display error -/
structure ErrorClock where
  /-- The number of hours in the clock cycle -/
  total_hours : Nat
  /-- The number of minutes in an hour -/
  minutes_per_hour : Nat
  /-- The number of hours affected by the display error -/
  incorrect_hours : Nat
  /-- The number of minutes per hour affected by the display error -/
  incorrect_minutes : Nat

/-- The fraction of the day when the ErrorClock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : Rat :=
  ((clock.total_hours - clock.incorrect_hours) * (clock.minutes_per_hour - clock.incorrect_minutes)) / 
  (clock.total_hours * clock.minutes_per_hour)

/-- The specific ErrorClock instance for the problem -/
def problem_clock : ErrorClock :=
  { total_hours := 12
  , minutes_per_hour := 60
  , incorrect_hours := 4
  , incorrect_minutes := 15 }

theorem error_clock_correct_time :
  correct_time_fraction problem_clock = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_error_clock_correct_time_l87_8752


namespace NUMINAMATH_CALUDE_sqrt_45_minus_sqrt_20_equals_sqrt_5_l87_8753

theorem sqrt_45_minus_sqrt_20_equals_sqrt_5 : 
  Real.sqrt 45 - Real.sqrt 20 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_45_minus_sqrt_20_equals_sqrt_5_l87_8753


namespace NUMINAMATH_CALUDE_sum_of_penultimate_terms_l87_8760

def arithmetic_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ d : ℕ, ∀ i : ℕ, i < n → a (i + 1) = a i + d

theorem sum_of_penultimate_terms (a : ℕ → ℕ) :
  arithmetic_sequence a 7 →
  a 0 = 3 →
  a 6 = 33 →
  a 4 + a 5 = 48 := by
sorry

end NUMINAMATH_CALUDE_sum_of_penultimate_terms_l87_8760
