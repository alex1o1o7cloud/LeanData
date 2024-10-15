import Mathlib

namespace NUMINAMATH_CALUDE_expression_value_l3006_300636

theorem expression_value : 
  (3^2016 + 3^2014 + 3^2012) / (3^2016 - 3^2014 + 3^2012) = 91/73 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3006_300636


namespace NUMINAMATH_CALUDE_total_canoes_by_april_l3006_300658

def canoes_in_month (n : ℕ) : ℕ :=
  2 * (3 ^ (n - 1))

theorem total_canoes_by_april : 
  canoes_in_month 1 + canoes_in_month 2 + canoes_in_month 3 + canoes_in_month 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_canoes_by_april_l3006_300658


namespace NUMINAMATH_CALUDE_rectangular_box_dimensions_l3006_300607

theorem rectangular_box_dimensions (X Y Z : ℝ) 
  (h1 : X * Y = 40)
  (h2 : X * Z = 72)
  (h3 : Y * Z = 90) :
  X + Y + Z = 18 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_dimensions_l3006_300607


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3006_300654

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 < x + 6} = {x : ℝ | -2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3006_300654


namespace NUMINAMATH_CALUDE_sqrt_of_nine_l3006_300685

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem statement
theorem sqrt_of_nine : sqrt 9 = {3, -3} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_nine_l3006_300685


namespace NUMINAMATH_CALUDE_dalmatians_right_spot_count_l3006_300639

/-- The number of Dalmatians with a spot on the right ear -/
def dalmatians_with_right_spot (total : ℕ) (left_only : ℕ) (right_only : ℕ) (no_spots : ℕ) : ℕ :=
  total - left_only - no_spots

/-- Theorem stating the number of Dalmatians with a spot on the right ear -/
theorem dalmatians_right_spot_count :
  dalmatians_with_right_spot 101 29 17 22 = 50 := by
  sorry

#eval dalmatians_with_right_spot 101 29 17 22

end NUMINAMATH_CALUDE_dalmatians_right_spot_count_l3006_300639


namespace NUMINAMATH_CALUDE_walking_jogging_time_difference_l3006_300644

/-- 
Given:
- Linda walks at 4 miles per hour
- Tom jogs at 9 miles per hour
- Linda starts walking 1 hour before Tom
- They walk in opposite directions

Prove that the difference in time (in minutes) for Tom to cover half and twice Linda's distance is 40 minutes.
-/
theorem walking_jogging_time_difference 
  (linda_speed : ℝ) 
  (tom_speed : ℝ) 
  (head_start : ℝ) :
  linda_speed = 4 →
  tom_speed = 9 →
  head_start = 1 →
  let linda_distance := linda_speed * head_start
  let half_distance := linda_distance / 2
  let double_distance := linda_distance * 2
  let time_half := (half_distance / tom_speed) * 60
  let time_double := (double_distance / tom_speed) * 60
  time_double - time_half = 40 := by
  sorry

end NUMINAMATH_CALUDE_walking_jogging_time_difference_l3006_300644


namespace NUMINAMATH_CALUDE_actual_average_height_l3006_300622

-- Define the problem parameters
def totalStudents : ℕ := 50
def initialAverage : ℚ := 175
def incorrectHeights : List ℚ := [162, 150, 155]
def actualHeights : List ℚ := [142, 135, 145]

-- Define the theorem
theorem actual_average_height :
  let totalInitialHeight : ℚ := initialAverage * totalStudents
  let heightDifference : ℚ := (List.sum incorrectHeights) - (List.sum actualHeights)
  let correctedTotalHeight : ℚ := totalInitialHeight - heightDifference
  let actualAverage : ℚ := correctedTotalHeight / totalStudents
  actualAverage = 174.1 := by
  sorry

end NUMINAMATH_CALUDE_actual_average_height_l3006_300622


namespace NUMINAMATH_CALUDE_distance_to_origin_l3006_300630

open Complex

theorem distance_to_origin : let z : ℂ := (1 - I) * (1 + I) / I
  abs z = 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_origin_l3006_300630


namespace NUMINAMATH_CALUDE_factor_problem_l3006_300645

theorem factor_problem (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 6 → 
  (2 * initial_number + 9) * factor = 63 → 
  factor = 3 := by
sorry

end NUMINAMATH_CALUDE_factor_problem_l3006_300645


namespace NUMINAMATH_CALUDE_roses_in_vase_l3006_300691

theorem roses_in_vase (initial_roses : ℕ) (initial_orchids : ℕ) (final_orchids : ℕ) (cut_orchids : ℕ) :
  initial_roses = 16 →
  initial_orchids = 3 →
  final_orchids = 7 →
  cut_orchids = 4 →
  ∃ (cut_roses : ℕ), cut_roses = cut_orchids →
  initial_roses + cut_roses = 24 :=
by sorry

end NUMINAMATH_CALUDE_roses_in_vase_l3006_300691


namespace NUMINAMATH_CALUDE_abs_cube_plus_cube_equals_two_cube_l3006_300659

theorem abs_cube_plus_cube_equals_two_cube (x : ℝ) : |x^3| + x^3 = 2*x^3 := by
  sorry

end NUMINAMATH_CALUDE_abs_cube_plus_cube_equals_two_cube_l3006_300659


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3006_300681

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0) ↔ -1 < m ∧ m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3006_300681


namespace NUMINAMATH_CALUDE_policy_effect_l3006_300648

-- Define the labor market for teachers
structure TeacherMarket where
  supply : ℝ → ℝ  -- Supply function
  demand : ℝ → ℝ  -- Demand function
  equilibrium_wage : ℝ  -- Equilibrium wage

-- Define the commercial education market
structure CommercialEducationMarket where
  supply : ℝ → ℝ  -- Supply function
  demand : ℝ → ℝ  -- Demand function
  equilibrium_price : ℝ  -- Equilibrium price

-- Define the government policy
def government_policy (min_years : ℕ) (locality : String) : Prop :=
  ∃ (requirement : Prop), requirement

-- Theorem statement
theorem policy_effect 
  (teacher_market : TeacherMarket)
  (commercial_market : CommercialEducationMarket)
  (min_years : ℕ)
  (locality : String) :
  government_policy min_years locality →
  ∃ (new_teacher_market : TeacherMarket)
    (new_commercial_market : CommercialEducationMarket),
    new_teacher_market.equilibrium_wage > teacher_market.equilibrium_wage ∧
    new_commercial_market.equilibrium_price < commercial_market.equilibrium_price :=
by
  sorry

end NUMINAMATH_CALUDE_policy_effect_l3006_300648


namespace NUMINAMATH_CALUDE_divergent_series_convergent_combination_l3006_300600

/-- Two positive sequences with divergent series but convergent combined series -/
theorem divergent_series_convergent_combination :
  ∃ (a b : ℕ → ℝ),
    (∀ n, a n > 0) ∧
    (∀ n, b n > 0) ∧
    (¬ Summable a) ∧
    (¬ Summable b) ∧
    Summable (λ n ↦ (2 * a n * b n) / (a n + b n)) := by
  sorry

end NUMINAMATH_CALUDE_divergent_series_convergent_combination_l3006_300600


namespace NUMINAMATH_CALUDE_petya_lives_in_sixth_entrance_l3006_300616

/-- Represents the layout of the houses -/
structure HouseLayout where
  num_entrances : ℕ
  petya_entrance : ℕ
  vasya_entrance : ℕ

/-- Calculates the distance between two entrances -/
def distance (layout : HouseLayout) (entrance1 entrance2 : ℕ) : ℝ :=
  sorry

/-- Represents the shortest path around Petya's house -/
def shortest_path (layout : HouseLayout) (side : Bool) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem petya_lives_in_sixth_entrance (layout : HouseLayout) :
  layout.vasya_entrance = 4 →
  shortest_path layout true = shortest_path layout false →
  layout.petya_entrance = 6 :=
sorry

end NUMINAMATH_CALUDE_petya_lives_in_sixth_entrance_l3006_300616


namespace NUMINAMATH_CALUDE_octal_subtraction_3456_1234_l3006_300635

/-- Represents a number in base 8 --/
def OctalNumber := List Nat

/-- Converts an octal number to its decimal representation --/
def octal_to_decimal (n : OctalNumber) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

/-- Subtracts two octal numbers --/
def octal_subtract (a b : OctalNumber) : OctalNumber :=
  sorry -- Actual implementation would go here

theorem octal_subtraction_3456_1234 :
  let a : OctalNumber := [6, 5, 4, 3]  -- 3456 in base 8
  let b : OctalNumber := [4, 3, 2, 1]  -- 1234 in base 8
  let result : OctalNumber := [2, 2, 2, 2]  -- 2222 in base 8
  octal_subtract a b = result := by
  sorry

end NUMINAMATH_CALUDE_octal_subtraction_3456_1234_l3006_300635


namespace NUMINAMATH_CALUDE_strawberry_jelly_sales_l3006_300682

/-- Represents the number of jars sold for each type of jelly -/
structure JellySales where
  grape : ℕ
  strawberry : ℕ
  raspberry : ℕ
  plum : ℕ

/-- Defines the relationships between jelly sales and proves the number of strawberry jelly jars sold -/
theorem strawberry_jelly_sales (sales : JellySales) : 
  sales.grape = 2 * sales.strawberry ∧ 
  sales.raspberry = 2 * sales.plum ∧
  sales.raspberry = sales.grape / 3 ∧
  sales.plum = 6 →
  sales.strawberry = 18 := by
sorry

end NUMINAMATH_CALUDE_strawberry_jelly_sales_l3006_300682


namespace NUMINAMATH_CALUDE_kfc_chicken_legs_l3006_300623

/-- Given the number of thighs, wings, and platters, calculate the number of legs baked. -/
def chicken_legs_baked (thighs wings platters : ℕ) : ℕ :=
  let thighs_per_platter := thighs / platters
  thighs_per_platter * platters

/-- Theorem stating that 144 chicken legs were baked given the problem conditions. -/
theorem kfc_chicken_legs :
  let thighs := 144
  let wings := 224
  let platters := 16
  chicken_legs_baked thighs wings platters = 144 := by
  sorry

#eval chicken_legs_baked 144 224 16

end NUMINAMATH_CALUDE_kfc_chicken_legs_l3006_300623


namespace NUMINAMATH_CALUDE_triangles_in_decagon_count_l3006_300698

/-- The number of triangles that can be formed from the vertices of a regular decagon -/
def trianglesInDecagon : ℕ := 120

/-- Proof that the number of triangles in a regular decagon is correct -/
theorem triangles_in_decagon_count : 
  (Finset.univ.filter (λ s : Finset (Fin 10) => s.card = 3)).card = trianglesInDecagon := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_decagon_count_l3006_300698


namespace NUMINAMATH_CALUDE_equation_solution_l3006_300673

theorem equation_solution :
  ∃! x : ℚ, 6 * (3 * x - 1) + 7 = -3 * (2 - 5 * x) - 4 :=
by
  use -11/3
  constructor
  · -- Proof that -11/3 satisfies the equation
    sorry
  · -- Proof of uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3006_300673


namespace NUMINAMATH_CALUDE_shortest_side_length_l3006_300621

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the side divided by the point of tangency -/
  side : ℝ
  /-- The length of the shorter segment of the divided side -/
  segment1 : ℝ
  /-- The length of the longer segment of the divided side -/
  segment2 : ℝ
  /-- The condition that the segments add up to the side length -/
  side_condition : side = segment1 + segment2

/-- The theorem stating the length of the shortest side -/
theorem shortest_side_length (t : InscribedCircleTriangle)
  (h1 : t.r = 3)
  (h2 : t.segment1 = 5)
  (h3 : t.segment2 = 9) :
  ∃ (shortest_side : ℝ), shortest_side = 12 ∧ 
  (∀ (other_side : ℝ), other_side ≥ shortest_side) :=
sorry

end NUMINAMATH_CALUDE_shortest_side_length_l3006_300621


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l3006_300665

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  first_element : ℕ
  h_population_size_pos : 0 < population_size
  h_sample_size_pos : 0 < sample_size
  h_sample_size_le_population : sample_size ≤ population_size
  h_first_element_in_range : first_element ≤ population_size

/-- Check if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.first_element + k * (s.population_size / s.sample_size) ∧ n ≤ s.population_size

/-- The main theorem to be proved -/
theorem systematic_sample_theorem (s : SystematicSample)
  (h_pop_size : s.population_size = 60)
  (h_sample_size : s.sample_size = 4)
  (h_contains_3 : s.contains 3)
  (h_contains_33 : s.contains 33)
  (h_contains_48 : s.contains 48) :
  s.contains 18 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sample_theorem_l3006_300665


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l3006_300642

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 4)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 4) :
  w / y = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l3006_300642


namespace NUMINAMATH_CALUDE_factorial_equation_sum_of_digits_l3006_300601

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- The theorem statement -/
theorem factorial_equation_sum_of_digits :
  ∃ (n : ℕ), n > 0 ∧ 
  (factorial (n + 1) + 2 * factorial (n + 2) = factorial n * 871) ∧
  (sumOfDigits n = 10) := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_sum_of_digits_l3006_300601


namespace NUMINAMATH_CALUDE_diet_soda_count_l3006_300624

/-- Represents the number of apples in the grocery store -/
def num_apples : ℕ := 36

/-- Represents the number of regular soda bottles in the grocery store -/
def num_regular_soda : ℕ := 80

/-- Represents the number of diet soda bottles in the grocery store -/
def num_diet_soda : ℕ := 54

/-- The total number of bottles is 98 more than the number of apples -/
axiom total_bottles_relation : num_regular_soda + num_diet_soda = num_apples + 98

theorem diet_soda_count : num_diet_soda = 54 := by sorry

end NUMINAMATH_CALUDE_diet_soda_count_l3006_300624


namespace NUMINAMATH_CALUDE_find_b_l3006_300656

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - b else 2^x

theorem find_b : ∃ b : ℝ, f b (f b (5/6)) = 4 ∧ b = 11/8 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l3006_300656


namespace NUMINAMATH_CALUDE_circle_with_diameter_OC_l3006_300683

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 6)^2 + (y - 8)^2 = 4

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the center of circle C
def center_C : ℝ × ℝ := (6, 8)

-- Define the equation of the circle with diameter OC
def circle_OC (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 25

-- Theorem statement
theorem circle_with_diameter_OC :
  ∀ x y : ℝ, circle_C x y → circle_OC x y :=
sorry

end NUMINAMATH_CALUDE_circle_with_diameter_OC_l3006_300683


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l3006_300626

-- Define the quadratic equation
def quadratic_equation (a : ℝ) (x : ℝ) : ℝ := x^2 - 3*x + a

-- Define the discriminant
def discriminant (a : ℝ) : ℝ := 9 - 4*a

-- Theorem statement
theorem a_equals_one_sufficient_not_necessary :
  (∃ x : ℝ, quadratic_equation 1 x = 0) ∧
  (∃ a : ℝ, a ≠ 1 ∧ ∃ x : ℝ, quadratic_equation a x = 0) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l3006_300626


namespace NUMINAMATH_CALUDE_second_smallest_divisible_sum_of_digits_l3006_300615

def isDivisibleByAllLessThan8 (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k ∧ k < 8 → n % k = 0

def isSecondSmallestDivisible (n : ℕ) : Prop :=
  isDivisibleByAllLessThan8 n ∧
  ∃ m : ℕ, m < n ∧ isDivisibleByAllLessThan8 m ∧
  ∀ k : ℕ, k < n ∧ isDivisibleByAllLessThan8 k → k ≤ m

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem second_smallest_divisible_sum_of_digits :
  ∃ N : ℕ, isSecondSmallestDivisible N ∧ sumOfDigits N = 12 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_divisible_sum_of_digits_l3006_300615


namespace NUMINAMATH_CALUDE_vector_equality_transitive_l3006_300611

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equality_transitive (a b c : V) :
  a = b → b = c → a = c := by sorry

end NUMINAMATH_CALUDE_vector_equality_transitive_l3006_300611


namespace NUMINAMATH_CALUDE_solve_equation_l3006_300651

theorem solve_equation (t x : ℝ) : 2*t + 2*x - t - 3*x + 4*x + 2*t = 30 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3006_300651


namespace NUMINAMATH_CALUDE_trig_sum_equality_l3006_300660

theorem trig_sum_equality : 
  Real.sin (2 * π / 3) ^ 2 + Real.cos π + Real.tan (π / 4) - 
  Real.cos (-11 * π / 6) ^ 2 + Real.sin (-7 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equality_l3006_300660


namespace NUMINAMATH_CALUDE_smallest_class_size_l3006_300618

theorem smallest_class_size (N : ℕ) (G : ℕ) : N = 7 ↔ 
  (N > 0 ∧ G > 0 ∧ (25 : ℚ) / 100 < (G : ℚ) / N ∧ (G : ℚ) / N < (30 : ℚ) / 100) ∧
  ∀ (M : ℕ) (H : ℕ), M < N → ¬(M > 0 ∧ H > 0 ∧ (25 : ℚ) / 100 < (H : ℚ) / M ∧ (H : ℚ) / M < (30 : ℚ) / 100) :=
sorry

end NUMINAMATH_CALUDE_smallest_class_size_l3006_300618


namespace NUMINAMATH_CALUDE_expression_simplification_l3006_300680

theorem expression_simplification (x y : ℝ) (h : y ≠ 0) :
  ((x^2 + y^2) - (x - y)^2 + 2*y*(x - y)) / (4*y) = x - (1/2)*y :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3006_300680


namespace NUMINAMATH_CALUDE_lee_earnings_l3006_300672

/-- Represents Lee's lawn care services and earnings --/
structure LawnCareServices where
  mowing_price : ℕ
  trimming_price : ℕ
  weed_removal_price : ℕ
  mowed_lawns : ℕ
  trimmed_lawns : ℕ
  weed_removed_lawns : ℕ
  mowing_tips : ℕ
  trimming_tips : ℕ
  weed_removal_tips : ℕ

/-- Calculates the total earnings from Lee's lawn care services --/
def total_earnings (s : LawnCareServices) : ℕ :=
  s.mowing_price * s.mowed_lawns +
  s.trimming_price * s.trimmed_lawns +
  s.weed_removal_price * s.weed_removed_lawns +
  s.mowing_tips +
  s.trimming_tips +
  s.weed_removal_tips

/-- Theorem stating that Lee's total earnings for the week were $747 --/
theorem lee_earnings : 
  let s : LawnCareServices := {
    mowing_price := 33,
    trimming_price := 15,
    weed_removal_price := 10,
    mowed_lawns := 16,
    trimmed_lawns := 8,
    weed_removed_lawns := 5,
    mowing_tips := 3 * 10,
    trimming_tips := 2 * 7,
    weed_removal_tips := 1 * 5
  }
  total_earnings s = 747 := by
  sorry


end NUMINAMATH_CALUDE_lee_earnings_l3006_300672


namespace NUMINAMATH_CALUDE_solution_equation1_no_solution_equation2_l3006_300633

-- Define the equations
def equation1 (x : ℝ) : Prop := (1 / (x - 1) = 5 / (2 * x + 1))
def equation2 (x : ℝ) : Prop := ((x + 1) / (x - 1) - 4 / (x^2 - 1) = 1)

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 2 := by sorry

-- Theorem for equation 2
theorem no_solution_equation2 : ¬ ∃ x : ℝ, equation2 x := by sorry

end NUMINAMATH_CALUDE_solution_equation1_no_solution_equation2_l3006_300633


namespace NUMINAMATH_CALUDE_center_of_mass_distance_to_line_l3006_300675

/-- Two material points in a plane -/
structure MaterialPoint where
  position : ℝ × ℝ
  mass : ℝ

/-- A line in a plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Distance from a point to a line -/
def distanceToLine (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Center of mass of two material points -/
def centerOfMass (p1 p2 : MaterialPoint) : ℝ × ℝ := sorry

theorem center_of_mass_distance_to_line 
  (P Q : MaterialPoint) (MN : Line) 
  (a b : ℝ) 
  (h1 : distanceToLine P.position MN = a) 
  (h2 : distanceToLine Q.position MN = b) :
  let Z := centerOfMass P Q
  distanceToLine Z MN = (P.mass * a + Q.mass * b) / (P.mass + Q.mass) := by
  sorry

end NUMINAMATH_CALUDE_center_of_mass_distance_to_line_l3006_300675


namespace NUMINAMATH_CALUDE_extremum_and_solutions_l3006_300663

/-- A function with an extremum at x = 0 -/
noncomputable def f (a b x : ℝ) : ℝ := x^2 + x - Real.log (x + a) + 3*b

/-- The statement to be proved -/
theorem extremum_and_solutions (a b : ℝ) :
  (f a b 0 = 0 ∧ ∀ x, f a b x ≥ f a b 0) →
  (a = 1 ∧ b = 0) ∧
  ∀ m : ℝ, (∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, -1/2 ≤ x ∧ x ≤ 2 ∧ f 1 0 x = m) ↔
    (0 < m ∧ m ≤ -1/4 + Real.log 2) :=
by sorry


end NUMINAMATH_CALUDE_extremum_and_solutions_l3006_300663


namespace NUMINAMATH_CALUDE_father_twice_son_age_l3006_300634

/-- Proves that the number of years after which a father aged 42 will be twice as old as his son aged 14 is 14 years. -/
theorem father_twice_son_age (father_age son_age : ℕ) (h1 : father_age = 42) (h2 : son_age = 14) :
  ∃ x : ℕ, father_age + x = 2 * (son_age + x) ∧ x = 14 :=
by sorry

end NUMINAMATH_CALUDE_father_twice_son_age_l3006_300634


namespace NUMINAMATH_CALUDE_cloth_sale_commission_calculation_l3006_300684

/-- Calculates the worth of cloth sold given the commission rate and commission amount. -/
def worth_of_cloth_sold (commission_rate : ℚ) (commission : ℚ) : ℚ :=
  commission * (100 / commission_rate)

/-- Theorem stating that for a 4% commission rate and Rs. 12.50 commission, 
    the worth of cloth sold is Rs. 312.50 -/
theorem cloth_sale_commission_calculation :
  worth_of_cloth_sold (4 : ℚ) (25/2 : ℚ) = (625/2 : ℚ) := by
  sorry

#eval worth_of_cloth_sold (4 : ℚ) (25/2 : ℚ)

end NUMINAMATH_CALUDE_cloth_sale_commission_calculation_l3006_300684


namespace NUMINAMATH_CALUDE_sin_210_plus_cos_60_equals_zero_l3006_300668

theorem sin_210_plus_cos_60_equals_zero :
  Real.sin (210 * π / 180) + Real.cos (60 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_plus_cos_60_equals_zero_l3006_300668


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l3006_300697

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem diagonals_25_sided_polygon : num_diagonals 25 = 275 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l3006_300697


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3006_300696

/-- The remainder when x^3 + 3x^2 is divided by x^2 - 7x + 2 is 68x - 20 -/
theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  x^3 + 3*x^2 = (x^2 - 7*x + 2) * q + (68*x - 20) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3006_300696


namespace NUMINAMATH_CALUDE_sum_terms_increase_l3006_300602

def sum_terms (k : ℕ) : ℕ := 2^(k-1) + 1

theorem sum_terms_increase (k : ℕ) (h : k ≥ 2) : 
  sum_terms (k+1) - sum_terms k = 2^(k-1) := by
  sorry

end NUMINAMATH_CALUDE_sum_terms_increase_l3006_300602


namespace NUMINAMATH_CALUDE_expression_simplification_l3006_300693

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2+x) + 4*(2-x) - 5*(2+3*x) = -20*x - 8 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l3006_300693


namespace NUMINAMATH_CALUDE_sum_2016_l3006_300637

/-- An arithmetic sequence with its first term and sum property -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  s : ℕ → ℤ  -- The sum sequence
  first_term : a 1 = -2016
  sum_property : s 20 / 20 - s 18 / 18 = 2
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n : ℕ, s n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)

/-- The sum of the first 2016 terms is -2016 -/
theorem sum_2016 (seq : ArithmeticSequence) : seq.s 2016 = -2016 := by
  sorry

end NUMINAMATH_CALUDE_sum_2016_l3006_300637


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3006_300631

theorem greatest_divisor_four_consecutive_integers :
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (n : ℕ), n > 0 → (k ∣ n * (n + 1) * (n + 2) * (n + 3))) ∧
  (∀ (m : ℕ), m > k → ∃ (n : ℕ), n > 0 ∧ ¬(m ∣ n * (n + 1) * (n + 2) * (n + 3))) :=
by
  use 24
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l3006_300631


namespace NUMINAMATH_CALUDE_opposite_sign_implications_l3006_300643

theorem opposite_sign_implications (a b : ℝ) 
  (h1 : |2*a + b| * Real.sqrt (3*b + 12) ≤ 0) 
  (h2 : |2*a + b| + Real.sqrt (3*b + 12) > 0) : 
  (Real.sqrt (2*a - 3*b) = 4 ∨ Real.sqrt (2*a - 3*b) = -4) ∧ 
  (∀ x : ℝ, a*x^2 + 4*b - 2 = 0 ↔ x = 3 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_opposite_sign_implications_l3006_300643


namespace NUMINAMATH_CALUDE_pinky_bought_36_apples_l3006_300664

/-- The number of apples Danny the Duck bought -/
def danny_apples : ℕ := 73

/-- The total number of apples Pinky the Pig and Danny the Duck have -/
def total_apples : ℕ := 109

/-- The number of apples Pinky the Pig bought -/
def pinky_apples : ℕ := total_apples - danny_apples

theorem pinky_bought_36_apples : pinky_apples = 36 := by
  sorry

end NUMINAMATH_CALUDE_pinky_bought_36_apples_l3006_300664


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_306_l3006_300699

/-- The function that returns the last two digits of 8^n -/
def lastTwoDigits (n : ℕ) : ℕ := (8^n) % 100

/-- The length of the cycle of last two digits of powers of 8 -/
def cycleLength : ℕ := 6

/-- The function that returns the tens digit of a number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_of_8_pow_306 : tensDigit (lastTwoDigits 306) = 6 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_306_l3006_300699


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2019th_term_l3006_300632

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_2019th_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum_5 : (a 1 + a 2 + a 3 + a 4 + a 5) = 15)
  (h_6th_term : a 6 = 6) :
  a 2019 = 2019 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2019th_term_l3006_300632


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l3006_300687

/-- The area of a square with a perimeter of 40 meters is 100 square meters. -/
theorem square_area_from_perimeter :
  ∀ (side : ℝ), 
  (4 * side = 40) →  -- perimeter is 40 meters
  (side * side = 100) -- area is 100 square meters
:= by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l3006_300687


namespace NUMINAMATH_CALUDE_count_solution_pairs_l3006_300671

/-- The number of distinct ordered pairs of positive integers (x,y) satisfying x^4y^2 - 10x^2y + 9 = 0 -/
def solution_count : ℕ := 3

/-- A predicate that checks if a pair of positive integers satisfies the equation -/
def satisfies_equation (x y : ℕ+) : Prop :=
  (x.val ^ 4) * (y.val ^ 2) - 10 * (x.val ^ 2) * y.val + 9 = 0

theorem count_solution_pairs :
  (∃! (s : Finset (ℕ+ × ℕ+)), 
    (∀ p ∈ s, satisfies_equation p.1 p.2) ∧ 
    s.card = solution_count) := by sorry

end NUMINAMATH_CALUDE_count_solution_pairs_l3006_300671


namespace NUMINAMATH_CALUDE_quartic_arithmetic_sequence_roots_l3006_300652

/-- The coefficients of a quartic equation whose roots form an arithmetic sequence -/
theorem quartic_arithmetic_sequence_roots (C D : ℝ) :
  (∃ (a d : ℝ), {a - 3*d, a - d, a + d, a + 3*d} = 
    {x : ℝ | x^4 + 4*x^3 - 34*x^2 + C*x + D = 0}) →
  C = -76 ∧ D = 105 := by
  sorry


end NUMINAMATH_CALUDE_quartic_arithmetic_sequence_roots_l3006_300652


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3006_300655

/-- Proves that the speed of a boat in still water is 25 km/hr, given the speed of the stream
    and the time and distance traveled downstream. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 3)
  (h3 : downstream_distance = 90) :
  (downstream_distance / downstream_time) - stream_speed = 25 := by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3006_300655


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l3006_300657

-- Define the markup and discount percentages
def markup : ℝ := 0.40
def discount : ℝ := 0.15

-- Theorem statement
theorem merchant_profit_percentage :
  let marked_price := 1 + markup
  let selling_price := marked_price * (1 - discount)
  (selling_price - 1) * 100 = 19 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l3006_300657


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3006_300678

theorem solution_set_inequality (a : ℝ) (h : (4 : ℝ)^a = 2^(a + 2)) :
  {x : ℝ | a^(2*x + 1) > a^(x - 1)} = {x : ℝ | x > -2} := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3006_300678


namespace NUMINAMATH_CALUDE_selina_shirt_cost_l3006_300669

/-- Represents the price and quantity of an item of clothing --/
structure ClothingItem where
  price : ℕ
  quantity : ℕ

/-- Calculates the total money Selina got from selling her clothes --/
def totalSalesMoney (pants shorts shirts : ClothingItem) : ℕ :=
  pants.price * pants.quantity + shorts.price * shorts.quantity + shirts.price * shirts.quantity

/-- Represents the problem of finding the cost of each shirt Selina bought --/
theorem selina_shirt_cost (pants shorts shirts : ClothingItem)
  (bought_shirts : ℕ) (money_left : ℕ)
  (h_pants : pants = ⟨5, 3⟩)
  (h_shorts : shorts = ⟨3, 5⟩)
  (h_shirts : shirts = ⟨4, 5⟩)
  (h_bought_shirts : bought_shirts = 2)
  (h_money_left : money_left = 30) :
  (totalSalesMoney pants shorts shirts - money_left) / bought_shirts = 10 := by
  sorry

#check selina_shirt_cost

end NUMINAMATH_CALUDE_selina_shirt_cost_l3006_300669


namespace NUMINAMATH_CALUDE_robin_gum_packages_l3006_300619

theorem robin_gum_packages (pieces_per_package : ℕ) (total_pieces : ℕ) (h1 : pieces_per_package = 18) (h2 : total_pieces = 486) :
  total_pieces / pieces_per_package = 27 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_packages_l3006_300619


namespace NUMINAMATH_CALUDE_cannot_form_desired_rectangle_l3006_300620

-- Define the tile sizes
def tile_size_1 : ℕ := 3
def tile_size_2 : ℕ := 4

-- Define the initial rectangles
def rect1_width : ℕ := 2
def rect1_height : ℕ := 6
def rect2_width : ℕ := 7
def rect2_height : ℕ := 8

-- Define the desired rectangle
def desired_width : ℕ := 12
def desired_height : ℕ := 5

-- Theorem statement
theorem cannot_form_desired_rectangle :
  ∀ (removed_tile1 removed_tile2 : ℕ),
  (removed_tile1 = tile_size_1 ∨ removed_tile1 = tile_size_2) →
  (removed_tile2 = tile_size_1 ∨ removed_tile2 = tile_size_2) →
  (rect1_width * rect1_height + rect2_width * rect2_height - removed_tile1 - removed_tile2) >
  (desired_width * desired_height) :=
by sorry

end NUMINAMATH_CALUDE_cannot_form_desired_rectangle_l3006_300620


namespace NUMINAMATH_CALUDE_min_type_A_costumes_l3006_300647

-- Define the cost of type B costumes
def cost_B : ℝ := 120

-- Define the cost of type A costumes
def cost_A : ℝ := cost_B + 30

-- Define the total number of costumes
def total_costumes : ℕ := 20

-- Define the minimum total cost
def min_total_cost : ℝ := 2800

-- Theorem statement
theorem min_type_A_costumes :
  ∀ m : ℕ,
  (m : ℝ) * cost_A + (total_costumes - m : ℝ) * cost_B ≥ min_total_cost →
  m ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_min_type_A_costumes_l3006_300647


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l3006_300638

/-- The equation of a circle -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to find the symmetric point of a point with respect to a line -/
def symmetricPoint (p : ℝ × ℝ) (l : Line) : ℝ × ℝ := sorry

/-- Function to find the symmetric circle -/
def symmetricCircle (c : Circle) (l : Line) : Circle := sorry

theorem symmetric_circle_equation (c : Circle) (l : Line) : 
  c.center = (1, 0) ∧ c.radius = Real.sqrt 2 ∧ 
  l = { a := 2, b := -1, c := 3 } →
  let c' := symmetricCircle c l
  c'.center = (-3, 2) ∧ c'.radius = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l3006_300638


namespace NUMINAMATH_CALUDE_last_three_digits_of_5_to_9000_l3006_300603

theorem last_three_digits_of_5_to_9000 (h : 5^300 ≡ 1 [ZMOD 125]) :
  5^9000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_5_to_9000_l3006_300603


namespace NUMINAMATH_CALUDE_number_of_groups_l3006_300690

theorem number_of_groups (max min interval : ℝ) (h1 : max = 140) (h2 : min = 51) (h3 : interval = 10) :
  ⌈(max - min) / interval⌉ = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_groups_l3006_300690


namespace NUMINAMATH_CALUDE_a_equals_zero_l3006_300641

theorem a_equals_zero (a : ℝ) : 
  let A : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}
  1 ∈ A → a = 0 := by
sorry

end NUMINAMATH_CALUDE_a_equals_zero_l3006_300641


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3006_300686

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ (∃ x : ℝ, x < 0 ∧ x^3 + x < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3006_300686


namespace NUMINAMATH_CALUDE_logarithm_inconsistency_l3006_300689

-- Define a custom logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the given logarithmic values
def lg3 : ℝ := 0.47712
def lg1_5 : ℝ := 0.17609
def lg5 : ℝ := 0.69897
def lg2 : ℝ := 0.30103
def lg7_incorrect : ℝ := 0.84519

-- Theorem statement
theorem logarithm_inconsistency :
  lg 3 = lg3 ∧
  lg 1.5 = lg1_5 ∧
  lg 5 = lg5 ∧
  lg 2 = lg2 ∧
  lg 7 ≠ lg7_incorrect :=
by sorry

end NUMINAMATH_CALUDE_logarithm_inconsistency_l3006_300689


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l3006_300604

theorem gcd_from_lcm_and_ratio (C D : ℕ+) : 
  C.lcm D = 180 → C.val * 6 = D.val * 5 → C.gcd D = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l3006_300604


namespace NUMINAMATH_CALUDE_equation_solutions_l3006_300666

theorem equation_solutions : 
  (∃ (x₁ x₂ : ℝ), x₁ = 5 + Real.sqrt 35 ∧ x₂ = 5 - Real.sqrt 35 ∧ 
    x₁^2 - 10*x₁ - 10 = 0 ∧ x₂^2 - 10*x₂ - 10 = 0) ∧
  (∃ (y₁ y₂ : ℝ), y₁ = 5 ∧ y₂ = 13/3 ∧ 
    3*(y₁ - 5)^2 = 2*(5 - y₁) ∧ 3*(y₂ - 5)^2 = 2*(5 - y₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3006_300666


namespace NUMINAMATH_CALUDE_brick_tower_heights_l3006_300614

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of distinct tower heights achievable -/
def distinctTowerHeights (numBricks : ℕ) (dimensions : BrickDimensions) : ℕ :=
  sorry

/-- Theorem stating the number of distinct tower heights for the given problem -/
theorem brick_tower_heights :
  let dimensions : BrickDimensions := ⟨3, 11, 17⟩
  distinctTowerHeights 62 dimensions = 435 := by
  sorry

end NUMINAMATH_CALUDE_brick_tower_heights_l3006_300614


namespace NUMINAMATH_CALUDE_polynomial_identity_l3006_300646

theorem polynomial_identity (a b c : ℝ) :
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - 2 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3006_300646


namespace NUMINAMATH_CALUDE_sin_cos_value_l3006_300610

theorem sin_cos_value (α : Real) (h : 1 / Real.sin α + 1 / Real.cos α = Real.sqrt 3) :
  Real.sin α * Real.cos α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_value_l3006_300610


namespace NUMINAMATH_CALUDE_sports_club_overlap_l3006_300649

theorem sports_club_overlap (total : ℕ) (badminton tennis neither : ℕ) 
  (h_total : total = 30)
  (h_badminton : badminton = 17)
  (h_tennis : tennis = 17)
  (h_neither : neither = 2)
  (h_sum : total = badminton + tennis - (total - neither)) :
  badminton + tennis - (total - neither) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l3006_300649


namespace NUMINAMATH_CALUDE_cube_volume_and_diagonal_l3006_300606

/-- Given a cube with surface area 150 square centimeters, prove its volume and space diagonal. -/
theorem cube_volume_and_diagonal (s : ℝ) (h : 6 * s^2 = 150) : 
  s^3 = 125 ∧ s * Real.sqrt 3 = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_and_diagonal_l3006_300606


namespace NUMINAMATH_CALUDE_triangle_circle_intersection_l3006_300613

theorem triangle_circle_intersection (DE DF EY FY : ℕ) (EF : ℝ) : 
  DE = 65 →
  DF = 104 →
  EY + FY = EF →
  FY * EF = 39 * 169 →
  EF = 169 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circle_intersection_l3006_300613


namespace NUMINAMATH_CALUDE_jack_evening_emails_l3006_300625

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 3

/-- The total number of emails Jack received in the morning and evening combined -/
def morning_evening_total : ℕ := 11

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := morning_evening_total - morning_emails

theorem jack_evening_emails : evening_emails = 8 := by
  sorry

end NUMINAMATH_CALUDE_jack_evening_emails_l3006_300625


namespace NUMINAMATH_CALUDE_fraction_equality_l3006_300662

theorem fraction_equality (w x y : ℚ) 
  (h1 : w / y = 2 / 3)
  (h2 : (x + y) / y = 3) :
  w / x = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3006_300662


namespace NUMINAMATH_CALUDE_ladder_problem_l3006_300674

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l3006_300674


namespace NUMINAMATH_CALUDE_f_negative_pi_third_l3006_300677

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^x / (2^x + 1) + a * x + Real.cos (2 * x)

theorem f_negative_pi_third (a : ℝ) : 
  f a (π / 3) = 2 → f a (-π / 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_pi_third_l3006_300677


namespace NUMINAMATH_CALUDE_blue_balls_count_l3006_300650

theorem blue_balls_count (B : ℕ) : 
  (6 : ℚ) * 5 / ((8 + B) * (7 + B)) = 0.19230769230769232 → B = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_balls_count_l3006_300650


namespace NUMINAMATH_CALUDE_rectangle_to_square_l3006_300692

theorem rectangle_to_square (l w : ℝ) : 
  (2 * (l + w) = 40) →  -- Perimeter of rectangle is 40cm
  (l - 8 = w + 2) →     -- Rectangle becomes square after changes
  (l - 8 = 7) :=        -- Side length of resulting square is 7cm
by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l3006_300692


namespace NUMINAMATH_CALUDE_tan_product_values_l3006_300628

theorem tan_product_values (a b : Real) 
  (h : 7 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b - 1) = 0) : 
  (Real.tan (a/2) * Real.tan (b/2) = Real.sqrt ((-7 + Real.sqrt 133) / 3) ∨
   Real.tan (a/2) * Real.tan (b/2) = -Real.sqrt ((-7 + Real.sqrt 133) / 3) ∨
   Real.tan (a/2) * Real.tan (b/2) = Real.sqrt ((-7 - Real.sqrt 133) / 3) ∨
   Real.tan (a/2) * Real.tan (b/2) = -Real.sqrt ((-7 - Real.sqrt 133) / 3)) := by
  sorry

end NUMINAMATH_CALUDE_tan_product_values_l3006_300628


namespace NUMINAMATH_CALUDE_function_equation_solution_l3006_300608

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 1 → f x + f (1 / (1 - x)) = x) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f x = (1 / 2) * (x + 1 - 1 / x - 1 / (1 - x))) ∧
  f 1 = -f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l3006_300608


namespace NUMINAMATH_CALUDE_baylor_freelance_earnings_l3006_300670

theorem baylor_freelance_earnings (initial_amount : ℝ) : initial_amount = 4000 → 
  let first_payment := initial_amount / 2
  let second_payment := first_payment + (2/5 * first_payment)
  let third_payment := 2 * (first_payment + second_payment)
  initial_amount + first_payment + second_payment + third_payment = 18400 := by
sorry

end NUMINAMATH_CALUDE_baylor_freelance_earnings_l3006_300670


namespace NUMINAMATH_CALUDE_f_2019_is_zero_l3006_300605

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def satisfies_equation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 6) - f x = 2 * f 3

theorem f_2019_is_zero (f : ℝ → ℝ) (h1 : is_even f) (h2 : satisfies_equation f) : f 2019 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2019_is_zero_l3006_300605


namespace NUMINAMATH_CALUDE_complex_magnitude_l3006_300679

theorem complex_magnitude (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3006_300679


namespace NUMINAMATH_CALUDE_function_inequality_l3006_300694

theorem function_inequality (f : ℝ → ℝ) (h1 : ∀ x, f x ≥ |x|) (h2 : ∀ x, f x ≥ 2^x) :
  ∀ a b : ℝ, f a ≤ 2^b → a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3006_300694


namespace NUMINAMATH_CALUDE_multiple_properties_l3006_300629

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 5 * k) 
  (hb : ∃ m : ℤ, b = 10 * m) : 
  (∃ n : ℤ, b = 5 * n) ∧ 
  (∃ p : ℤ, a - b = 5 * p) ∧ 
  (∃ q : ℤ, a + b = 5 * q) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l3006_300629


namespace NUMINAMATH_CALUDE_area_of_region_l3006_300653

-- Define the curve and line
def curve (x : ℝ) : ℝ → Prop := λ y ↦ y^2 = 2*x
def line (x : ℝ) : ℝ → Prop := λ y ↦ y = x - 4

-- Define the region
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ curve x y ∧ line x y}

-- State the theorem
theorem area_of_region : MeasureTheory.volume region = 18 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l3006_300653


namespace NUMINAMATH_CALUDE_cone_base_radius_l3006_300609

/-- Given a cone formed from a circular sector with central angle 120° and radius 4,
    the radius of the base circle of the cone is 4/3. -/
theorem cone_base_radius (θ : Real) (R : Real) (r : Real) : 
  θ = 120 → R = 4 → 2 * π * r = (θ / 360) * 2 * π * R → r = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3006_300609


namespace NUMINAMATH_CALUDE_don_raise_is_880_l3006_300661

/-- Calculates Don's raise given the conditions of the problem -/
def calculate_don_raise (wife_raise : ℚ) (salary_difference : ℚ) : ℚ :=
  let wife_salary := wife_raise / 0.08
  let don_salary := (wife_salary + salary_difference + wife_raise) / 1.08
  0.08 * don_salary

/-- Theorem stating that Don's raise is 880 given the problem conditions -/
theorem don_raise_is_880 :
  calculate_don_raise 840 540 = 880 := by
  sorry

end NUMINAMATH_CALUDE_don_raise_is_880_l3006_300661


namespace NUMINAMATH_CALUDE_water_level_rise_rate_l3006_300667

/-- The rate at which the water level rises in a cylinder when water drains from a cube -/
theorem water_level_rise_rate (cube_side : ℝ) (cylinder_radius : ℝ) (cube_fall_rate : ℝ) :
  cube_side = 100 →
  cylinder_radius = 100 →
  cube_fall_rate = 1 →
  (cylinder_radius ^ 2 * π) * (cube_side ^ 2 * cube_fall_rate) / (cylinder_radius ^ 2 * π) ^ 2 = 1 / π := by
  sorry

end NUMINAMATH_CALUDE_water_level_rise_rate_l3006_300667


namespace NUMINAMATH_CALUDE_nickel_quarter_problem_l3006_300676

theorem nickel_quarter_problem :
  ∀ (n : ℕ),
    (n : ℚ) * 0.05 + (n : ℚ) * 0.25 = 12 →
    n = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_nickel_quarter_problem_l3006_300676


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_product_sum_l3006_300640

theorem consecutive_negative_integers_product_sum (n : ℤ) : 
  n < 0 ∧ n > -50 ∧ n * (n + 1) = 2400 → n + (n + 1) = -97 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_product_sum_l3006_300640


namespace NUMINAMATH_CALUDE_ratio_equivalence_l3006_300627

theorem ratio_equivalence (x : ℚ) : (3 / x = 3 / 16) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalence_l3006_300627


namespace NUMINAMATH_CALUDE_spotlight_detection_l3006_300617

/-- Represents the spotlight's properties -/
structure Spotlight where
  illumination_length : ℝ  -- Length of illuminated segment in km
  rotation_period : ℝ      -- Time for one complete rotation in minutes

/-- Represents the boat's properties -/
structure Boat where
  speed : ℝ  -- Speed in km/min

/-- Determines if a boat can approach undetected given a spotlight -/
def can_approach_undetected (s : Spotlight) (b : Boat) : Prop :=
  b.speed ≥ 48.6 / 60  -- Convert 48.6 km/h to km/min

theorem spotlight_detection (s : Spotlight) (b : Boat) :
  s.illumination_length = 1 ∧ s.rotation_period = 1 →
  (b.speed < 800 / 1000 → ¬can_approach_undetected s b) ∧
  (b.speed ≥ 48.6 / 60 → can_approach_undetected s b) := by
  sorry

#check spotlight_detection

end NUMINAMATH_CALUDE_spotlight_detection_l3006_300617


namespace NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_l3006_300695

/-- A regular tetrahedron inscribed in a unit sphere -/
structure InscribedTetrahedron where
  /-- The tetrahedron is regular -/
  regular : Bool
  /-- All vertices lie on the surface of the sphere -/
  vertices_on_sphere : Bool
  /-- The sphere has radius 1 -/
  sphere_radius : ℝ
  /-- Three vertices of the base are on a great circle of the sphere -/
  base_on_great_circle : Bool

/-- The volume of the inscribed regular tetrahedron -/
def tetrahedron_volume (t : InscribedTetrahedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the inscribed regular tetrahedron -/
theorem inscribed_tetrahedron_volume 
  (t : InscribedTetrahedron) 
  (h1 : t.regular = true) 
  (h2 : t.vertices_on_sphere = true) 
  (h3 : t.sphere_radius = 1) 
  (h4 : t.base_on_great_circle = true) : 
  tetrahedron_volume t = Real.sqrt 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_l3006_300695


namespace NUMINAMATH_CALUDE_quadratic_sum_l3006_300688

/-- A quadratic function with vertex at (2, 8) and passing through (0, 0) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, f a b c x = a * x^2 + b * x + c) →
  f a b c 2 = 8 →
  (∀ x, f a b c x ≤ f a b c 2) →
  f a b c 0 = 0 →
  a + b + 2*c = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3006_300688


namespace NUMINAMATH_CALUDE_measure_17kg_cranberries_l3006_300612

/-- Represents a two-pan scale -/
structure TwoPanScale :=
  (leftPan : ℝ)
  (rightPan : ℝ)

/-- Represents the state of the cranberry measurement process -/
structure CranberryMeasurement :=
  (totalAmount : ℝ)
  (weightAmount : ℝ)
  (scale : TwoPanScale)
  (weighingsUsed : ℕ)

/-- Definition of a valid weighing operation -/
def validWeighing (m : CranberryMeasurement) : Prop :=
  m.scale.leftPan = m.scale.rightPan ∧ m.weighingsUsed ≤ 2

/-- The main theorem to prove -/
theorem measure_17kg_cranberries :
  ∃ (m : CranberryMeasurement),
    m.totalAmount = 22 ∧
    m.weightAmount = 2 ∧
    validWeighing m ∧
    ∃ (amount : ℝ), amount = 17 ∧ amount ≤ m.totalAmount :=
sorry

end NUMINAMATH_CALUDE_measure_17kg_cranberries_l3006_300612
