import Mathlib

namespace NUMINAMATH_CALUDE_solution_of_equation_l2156_215694

theorem solution_of_equation (x : ℚ) : 2/3 - 1/4 = 1/x → x = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l2156_215694


namespace NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_l2156_215698

-- Define Pascal's triangle
def pascal_triangle : Nat → Nat → Nat
  | 0, _ => 1
  | n + 1, 0 => 1
  | n + 1, k + 1 => pascal_triangle n k + pascal_triangle n (k + 1)

-- Define a predicate to check if a number is in Pascal's triangle
def in_pascal_triangle (n : Nat) : Prop :=
  ∃ (row col : Nat), pascal_triangle row col = n

-- Theorem statement
theorem smallest_four_digit_in_pascal :
  (∀ n, 1000 ≤ n → n < 10000 → in_pascal_triangle n) →
  (∀ n, n < 1000 → n < 10000 → in_pascal_triangle n) →
  (∃ n, 1000 ≤ n ∧ n < 10000 ∧ in_pascal_triangle n) →
  (∀ n, 1000 ≤ n → n < 10000 → in_pascal_triangle n → 1000 ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_in_pascal_l2156_215698


namespace NUMINAMATH_CALUDE_divisibility_property_l2156_215691

theorem divisibility_property (a b : ℕ+) : ∃ n : ℕ+, (a : ℕ) ∣ (b : ℕ)^(n : ℕ) - (n : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2156_215691


namespace NUMINAMATH_CALUDE_min_distance_on_hyperbola_l2156_215639

theorem min_distance_on_hyperbola :
  ∀ x y : ℝ, (x^2 / 8 - y^2 / 4 = 1) → (∀ x' y' : ℝ, (x'^2 / 8 - y'^2 / 4 = 1) → |x - y| ≤ |x' - y'|) →
  |x - y| = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_on_hyperbola_l2156_215639


namespace NUMINAMATH_CALUDE_largest_angle_not_less_than_60_degrees_l2156_215621

open Real

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : Point) (b : Point)

/-- Calculates the angle between two lines -/
noncomputable def angle (l1 l2 : Line) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (a b c : Point) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (m : Point) (a b : Point) : Prop := sorry

/-- Main theorem -/
theorem largest_angle_not_less_than_60_degrees 
  (a b c : Point) 
  (h_equilateral : isEquilateral a b c)
  (c₁ : Point) (h_c₁_midpoint : isMidpoint c₁ a b)
  (a₁ : Point) (h_a₁_midpoint : isMidpoint a₁ b c)
  (b₁ : Point) (h_b₁_midpoint : isMidpoint b₁ c a)
  (p : Point) :
  let angle1 := angle (Line.mk a b) (Line.mk p c₁)
  let angle2 := angle (Line.mk b c) (Line.mk p a₁)
  let angle3 := angle (Line.mk c a) (Line.mk p b₁)
  max angle1 (max angle2 angle3) ≥ π/3 := by sorry

end NUMINAMATH_CALUDE_largest_angle_not_less_than_60_degrees_l2156_215621


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l2156_215616

theorem lcm_gcd_problem : (Nat.lcm 12 9 * Nat.gcd 12 9) - Nat.gcd 15 9 = 105 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l2156_215616


namespace NUMINAMATH_CALUDE_seventieth_even_positive_integer_seventieth_even_positive_integer_is_140_l2156_215600

theorem seventieth_even_positive_integer : ℕ → ℕ := 
  fun n => 2 * n

#check seventieth_even_positive_integer 70 = 140

theorem seventieth_even_positive_integer_is_140 : 
  seventieth_even_positive_integer 70 = 140 := by
  sorry

end NUMINAMATH_CALUDE_seventieth_even_positive_integer_seventieth_even_positive_integer_is_140_l2156_215600


namespace NUMINAMATH_CALUDE_mk97_check_one_l2156_215677

theorem mk97_check_one (a : ℝ) : 
  (a = 1) ↔ (a ≠ 2 * a ∧ 
             ∃ x : ℝ, x^2 + 2*a*x + a = 0 ∧ 
             ∀ y : ℝ, y^2 + 2*a*y + a = 0 → y = x) := by
  sorry

end NUMINAMATH_CALUDE_mk97_check_one_l2156_215677


namespace NUMINAMATH_CALUDE_parabola_c_value_l2156_215683

/-- A parabola with equation x = ay^2 + by + c, vertex at (4, 3), and passing through (2, 5) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 4
  vertex_y : ℝ := 3
  point_x : ℝ := 2
  point_y : ℝ := 5
  eq_vertex : 4 = a * 3^2 + b * 3 + c
  eq_point : 2 = a * 5^2 + b * 5 + c

/-- The value of c for the given parabola is -1/2 -/
theorem parabola_c_value (p : Parabola) : p.c = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2156_215683


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2156_215675

theorem inverse_variation_problem (y z : ℝ) (k : ℝ) :
  (∀ y z, y^2 * Real.sqrt z = k) →  -- y² varies inversely with √z
  (3^2 * Real.sqrt 16 = k) →        -- y = 3 when z = 16
  (6^2 * Real.sqrt z = k) →         -- condition for y = 6
  z = 1 :=                          -- prove z = 1 when y = 6
by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2156_215675


namespace NUMINAMATH_CALUDE_novel_writing_speed_l2156_215678

/-- Given a novel with 40,000 words written in 80 hours, 
    the average number of words written per hour is 500. -/
theorem novel_writing_speed (total_words : ℕ) (total_hours : ℕ) 
  (h1 : total_words = 40000) (h2 : total_hours = 80) :
  total_words / total_hours = 500 := by
  sorry

#check novel_writing_speed

end NUMINAMATH_CALUDE_novel_writing_speed_l2156_215678


namespace NUMINAMATH_CALUDE_hilary_corn_shucking_l2156_215603

/-- The number of ears of corn per stalk -/
def ears_per_stalk : ℕ := 4

/-- The number of stalks Hilary has -/
def total_stalks : ℕ := 108

/-- The number of kernels on half of the ears -/
def kernels_first_half : ℕ := 500

/-- The additional number of kernels on the other half of the ears -/
def additional_kernels : ℕ := 100

/-- The total number of kernels Hilary has to shuck -/
def total_kernels : ℕ := 
  let total_ears := ears_per_stalk * total_stalks
  let ears_per_half := total_ears / 2
  let kernels_second_half := kernels_first_half + additional_kernels
  ears_per_half * kernels_first_half + ears_per_half * kernels_second_half

theorem hilary_corn_shucking :
  total_kernels = 237600 := by
  sorry

end NUMINAMATH_CALUDE_hilary_corn_shucking_l2156_215603


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2156_215697

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem states that for a geometric sequence satisfying certain conditions, 
    the sum of its 2nd and 8th terms equals 9. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_geo : IsGeometricSequence a) 
  (h_prod : a 3 * a 7 = 8)
  (h_sum : a 4 + a 6 = 6) : 
  a 2 + a 8 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2156_215697


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l2156_215651

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) :
  angle_sum = 720 → (n - 2) * 180 = angle_sum → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l2156_215651


namespace NUMINAMATH_CALUDE_geometric_sequence_roots_l2156_215638

theorem geometric_sequence_roots (a b : ℝ) : 
  (∃ x₁ x₄ : ℝ, x₁ ≠ x₄ ∧ x₁^2 - 9*x₁ + 2^a = 0 ∧ x₄^2 - 9*x₄ + 2^a = 0) →
  (∃ x₂ x₃ : ℝ, x₂ ≠ x₃ ∧ x₂^2 - 6*x₂ + 2^b = 0 ∧ x₃^2 - 6*x₃ + 2^b = 0) →
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ * 2 = x₂ ∧ x₂ * 2 = x₃ ∧ x₃ * 2 = x₄ ∧
    x₁^2 - 9*x₁ + 2^a = 0 ∧ x₄^2 - 9*x₄ + 2^a = 0 ∧
    x₂^2 - 6*x₂ + 2^b = 0 ∧ x₃^2 - 6*x₃ + 2^b = 0) →
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_roots_l2156_215638


namespace NUMINAMATH_CALUDE_race_finish_l2156_215690

theorem race_finish (john_speed steve_speed : ℝ) (initial_distance time : ℝ) : 
  john_speed = 4.2 →
  steve_speed = 3.7 →
  initial_distance = 16 →
  time = 36 →
  john_speed * time - initial_distance - steve_speed * time = 2 :=
by sorry

end NUMINAMATH_CALUDE_race_finish_l2156_215690


namespace NUMINAMATH_CALUDE_garage_wheels_count_l2156_215620

def total_wheels (cars bicycles : Nat) (lawnmower tricycle unicycle skateboard wheelbarrow wagon : Nat) : Nat :=
  cars * 4 + bicycles * 2 + lawnmower * 4 + tricycle * 3 + unicycle + skateboard * 4 + wheelbarrow + wagon * 4

theorem garage_wheels_count :
  total_wheels 2 3 1 1 1 1 1 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheels_count_l2156_215620


namespace NUMINAMATH_CALUDE_power_five_sum_greater_than_mixed_products_l2156_215664

theorem power_five_sum_greater_than_mixed_products {a b : ℝ} 
  (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  a^5 + b^5 > a^3 * b^2 + a^2 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_five_sum_greater_than_mixed_products_l2156_215664


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l2156_215659

theorem quadratic_root_zero (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 6 * x + k^2 - k = 0) ∧
  ((k - 1) * 0^2 + 6 * 0 + k^2 - k = 0) →
  k = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l2156_215659


namespace NUMINAMATH_CALUDE_inequality_preservation_l2156_215684

theorem inequality_preservation (a b : ℝ) (h : a > b) :
  ∀ x : ℝ, a * (2 : ℝ)^x > b * (2 : ℝ)^x := by
sorry


end NUMINAMATH_CALUDE_inequality_preservation_l2156_215684


namespace NUMINAMATH_CALUDE_earth_surface_utilization_l2156_215628

theorem earth_surface_utilization (
  exposed_land : ℚ)
  (inhabitable_land : ℚ)
  (utilized_land : ℚ)
  (h1 : exposed_land = 1 / 3)
  (h2 : inhabitable_land = 2 / 5 * exposed_land)
  (h3 : utilized_land = 3 / 4 * inhabitable_land) :
  utilized_land = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_earth_surface_utilization_l2156_215628


namespace NUMINAMATH_CALUDE_odd_m_triple_g_eq_5_l2156_215685

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n - 4 else n / 3

theorem odd_m_triple_g_eq_5 (m : ℤ) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 5) : m = 17 := by
  sorry

end NUMINAMATH_CALUDE_odd_m_triple_g_eq_5_l2156_215685


namespace NUMINAMATH_CALUDE_right_triangle_sides_l2156_215602

theorem right_triangle_sides : ∃ (a b c : ℝ), 
  a = Real.sqrt 3 ∧ 
  b = Real.sqrt 13 ∧ 
  c = 4 ∧ 
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l2156_215602


namespace NUMINAMATH_CALUDE_water_bottle_boxes_l2156_215655

theorem water_bottle_boxes (bottles_per_box : ℕ) (bottle_capacity : ℚ) (fill_ratio : ℚ) (total_water : ℚ) 
  (h1 : bottles_per_box = 50)
  (h2 : bottle_capacity = 12)
  (h3 : fill_ratio = 3/4)
  (h4 : total_water = 4500) :
  (total_water / (bottle_capacity * fill_ratio)) / bottles_per_box = 10 := by
sorry

end NUMINAMATH_CALUDE_water_bottle_boxes_l2156_215655


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l2156_215665

theorem quadratic_root_ratio (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ 0 ∧ y = 2 * x ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  2 * b^2 = 9 * a * c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l2156_215665


namespace NUMINAMATH_CALUDE_arccos_sin_three_l2156_215681

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sin_three_l2156_215681


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2156_215624

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {x | ∃ m ∈ A, x = 3 * m - 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2156_215624


namespace NUMINAMATH_CALUDE_vertex_locus_is_circle_l2156_215676

/-- A triangle with a fixed base and a median of constant length --/
structure TriangleWithMedian where
  /-- The length of the fixed base AB --/
  base_length : ℝ
  /-- The length of the median from A to side BC --/
  median_length : ℝ

/-- The locus of vertex C in a triangle with a fixed base and constant median length --/
def vertex_locus (t : TriangleWithMedian) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {p | ∃ (A B : EuclideanSpace ℝ (Fin 2)), 
    ‖B - A‖ = t.base_length ∧ 
    ‖p - A‖ = t.median_length}

/-- The theorem stating that the locus of vertex C is a circle --/
theorem vertex_locus_is_circle (t : TriangleWithMedian) 
  (h : t.base_length = 6 ∧ t.median_length = 3) : 
  ∃ (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ),
    vertex_locus t = {p | ‖p - center‖ = radius} ∧ radius = 3 :=
sorry

end NUMINAMATH_CALUDE_vertex_locus_is_circle_l2156_215676


namespace NUMINAMATH_CALUDE_coefficient_x_10_l2156_215671

/-- The coefficient of x^10 in the expansion of (x^3/3 - 3/x^2)^10 is 17010/729 -/
theorem coefficient_x_10 : 
  let f (x : ℚ) := (x^3 / 3 - 3 / x^2)^10
  ∃ (c : ℚ), c = 17010 / 729 ∧ 
    ∃ (g : ℚ → ℚ), (∀ x, x ≠ 0 → f x = c * x^10 + x * g x) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_10_l2156_215671


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l2156_215682

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  red : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- The probability of drawing a yellow marble given the conditions -/
def yellowProbability (bagA bagB bagC bagD : Bag) : ℚ :=
  let totalA := bagA.white + bagA.black + bagA.red
  let probWhite := bagA.white / totalA
  let probBlack := bagA.black / totalA
  let probRed := bagA.red / totalA
  let probYellowB := bagB.yellow / (bagB.yellow + bagB.blue)
  let probYellowC := bagC.yellow / (bagC.yellow + bagC.blue)
  let probYellowD := bagD.yellow / (bagD.yellow + bagD.blue)
  probWhite * probYellowB + probBlack * probYellowC + probRed * probYellowD

theorem yellow_marble_probability :
  let bagA : Bag := { white := 4, black := 5, red := 2 }
  let bagB : Bag := { yellow := 7, blue := 5 }
  let bagC : Bag := { yellow := 3, blue := 7 }
  let bagD : Bag := { yellow := 8, blue := 2 }
  yellowProbability bagA bagB bagC bagD = 163 / 330 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marble_probability_l2156_215682


namespace NUMINAMATH_CALUDE_gcd_of_45_and_75_l2156_215662

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_and_75_l2156_215662


namespace NUMINAMATH_CALUDE_gcd_45678_12345_l2156_215654

theorem gcd_45678_12345 : Nat.gcd 45678 12345 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45678_12345_l2156_215654


namespace NUMINAMATH_CALUDE_ian_painted_cuboids_l2156_215661

/-- The number of cuboids painted by Ian -/
def num_cuboids : ℕ := 8

/-- The total number of faces painted -/
def total_faces : ℕ := 48

/-- The number of faces on one cuboid -/
def faces_per_cuboid : ℕ := 6

/-- Theorem: The number of cuboids painted is equal to 8 -/
theorem ian_painted_cuboids : 
  num_cuboids = total_faces / faces_per_cuboid :=
sorry

end NUMINAMATH_CALUDE_ian_painted_cuboids_l2156_215661


namespace NUMINAMATH_CALUDE_inequality_proof_l2156_215640

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) :
  a^2 / b + b^2 / c + c^2 / d + d^2 / a ≥ 4 + (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2156_215640


namespace NUMINAMATH_CALUDE_peanut_butter_recipe_l2156_215666

/-- Peanut butter recipe proof -/
theorem peanut_butter_recipe (total_weight oil_to_peanut_ratio honey_weight : ℝ) 
  (h1 : total_weight = 32)
  (h2 : oil_to_peanut_ratio = 3 / 12)
  (h3 : honey_weight = 2) :
  let peanut_weight := total_weight * (1 / (1 + oil_to_peanut_ratio + honey_weight / total_weight))
  let oil_weight := peanut_weight * oil_to_peanut_ratio
  oil_weight + honey_weight = 8 := by
sorry


end NUMINAMATH_CALUDE_peanut_butter_recipe_l2156_215666


namespace NUMINAMATH_CALUDE_product_9_to_11_l2156_215679

theorem product_9_to_11 : (List.range 3).foldl (·*·) 1 * 9 = 990 := by
  sorry

end NUMINAMATH_CALUDE_product_9_to_11_l2156_215679


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2156_215630

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def yIntercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line (b : Line) :
  parallel b { slope := -3, point := (0, 6) } →
  b.point = (3, -2) →
  yIntercept b = 7 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2156_215630


namespace NUMINAMATH_CALUDE_equilateral_triangle_formation_l2156_215606

/-- Function to calculate the sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate to check if a number is divisible by 3 -/
def divisible_by_three (n : ℕ) : Prop := n % 3 = 0

/-- Predicate to check if it's possible to form an equilateral triangle from n sticks -/
def can_form_equilateral_triangle (n : ℕ) : Prop :=
  divisible_by_three (sum_to_n n) ∧ 
  ∃ (partition : ℕ → ℕ → ℕ), 
    (∀ i j, i < j → j ≤ n → partition i j ≤ sum_to_n n / 3) ∧
    (∀ i, i ≤ n → ∃ j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ j ≤ n ∧ k ≤ n ∧
      partition i j + partition j k + partition k i = sum_to_n n / 3)

theorem equilateral_triangle_formation :
  ¬can_form_equilateral_triangle 100 ∧ can_form_equilateral_triangle 99 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_formation_l2156_215606


namespace NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l2156_215695

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (12/11, 14/11)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 2 * y = 6 * x - 4

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_on_both_lines : 
  line1 intersection_point.1 intersection_point.2 ∧ 
  line2 intersection_point.1 intersection_point.2 := by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem unique_intersection_point (x y : ℚ) : 
  line1 x y ∧ line2 x y → (x, y) = intersection_point := by sorry

end NUMINAMATH_CALUDE_intersection_point_on_both_lines_unique_intersection_point_l2156_215695


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l2156_215625

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) 
  (hb : ∃ m : ℤ, b = 9 * m) : 
  ∃ n : ℤ, a + b = 3 * n := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_is_multiple_of_3_l2156_215625


namespace NUMINAMATH_CALUDE_one_true_related_proposition_l2156_215649

theorem one_true_related_proposition :
  let P : (ℝ × ℝ) → Prop := λ (a, b) => (a + b = 1) → (a * b ≤ 1/4)
  let converse : (ℝ × ℝ) → Prop := λ (a, b) => (a * b > 1/4) → (a + b ≠ 1)
  let inverse : (ℝ × ℝ) → Prop := λ (a, b) => (a * b ≤ 1/4) → (a + b = 1)
  let contrapositive : (ℝ × ℝ) → Prop := λ (a, b) => (a * b > 1/4) → (a + b ≠ 1)
  (∀ a b, P (a, b)) ∧ (∀ a b, contrapositive (a, b)) ∧ (∃ a b, ¬inverse (a, b)) ∧ (∃ a b, ¬converse (a, b)) :=
by sorry

end NUMINAMATH_CALUDE_one_true_related_proposition_l2156_215649


namespace NUMINAMATH_CALUDE_sylvia_incorrect_fraction_l2156_215615

/-- Proves that Sylvia's fraction of incorrect answers is 1/5 given the conditions -/
theorem sylvia_incorrect_fraction (total_questions : ℕ) (sergio_incorrect : ℕ) (difference : ℕ) :
  total_questions = 50 →
  sergio_incorrect = 4 →
  difference = 6 →
  (total_questions - (total_questions - sergio_incorrect - difference)) / total_questions = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sylvia_incorrect_fraction_l2156_215615


namespace NUMINAMATH_CALUDE_unique_digit_arrangement_l2156_215643

theorem unique_digit_arrangement : ∃! (a b c d e : ℕ),
  (0 < a ∧ a ≤ 9) ∧
  (0 < b ∧ b ≤ 9) ∧
  (0 < c ∧ c ≤ 9) ∧
  (0 < d ∧ d ≤ 9) ∧
  (0 < e ∧ e ≤ 9) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a + b = (c + d + e) / 7 ∧
  a + c = (b + d + e) / 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_arrangement_l2156_215643


namespace NUMINAMATH_CALUDE_cat_kibble_problem_l2156_215674

/-- Represents the amount of kibble eaten by a cat in a given time -/
def kibble_eaten (eating_rate : ℚ) (time : ℚ) : ℚ :=
  (time / 4) * eating_rate

/-- Represents the amount of kibble left in the bowl after some time -/
def kibble_left (initial_amount : ℚ) (eating_rate : ℚ) (time : ℚ) : ℚ :=
  initial_amount - kibble_eaten eating_rate time

theorem cat_kibble_problem :
  let initial_amount : ℚ := 3
  let eating_rate : ℚ := 1
  let time : ℚ := 8
  kibble_left initial_amount eating_rate time = 1 := by sorry

end NUMINAMATH_CALUDE_cat_kibble_problem_l2156_215674


namespace NUMINAMATH_CALUDE_carton_height_calculation_l2156_215601

/-- Calculates the height of a carton given its base dimensions, soap box dimensions, and maximum capacity -/
theorem carton_height_calculation (carton_length carton_width : ℕ) 
  (box_length box_width box_height : ℕ) (max_boxes : ℕ) : 
  carton_length = 25 ∧ carton_width = 42 ∧ 
  box_length = 7 ∧ box_width = 6 ∧ box_height = 5 ∧
  max_boxes = 300 →
  (max_boxes / ((carton_length / box_length) * (carton_width / box_width))) * box_height = 70 :=
by sorry

end NUMINAMATH_CALUDE_carton_height_calculation_l2156_215601


namespace NUMINAMATH_CALUDE_sock_combination_count_l2156_215656

/-- Represents the color of a sock pair -/
inductive Color
  | Red
  | Blue
  | Green

/-- Represents the pattern of a sock pair -/
inductive Pattern
  | Striped
  | Dotted
  | Checkered
  | Plain

/-- Represents a pair of socks -/
structure SockPair :=
  (color : Color)
  (pattern : Pattern)

def total_pairs : ℕ := 12
def red_pairs : ℕ := 4
def blue_pairs : ℕ := 4
def green_pairs : ℕ := 4

def sock_collection : List SockPair := sorry

/-- Checks if two socks form a valid combination according to the constraints -/
def is_valid_combination (sock1 sock2 : SockPair) : Bool := sorry

/-- Counts the number of valid combinations -/
def count_valid_combinations (socks : List SockPair) : ℕ := sorry

theorem sock_combination_count :
  count_valid_combinations sock_collection = 12 := by sorry

end NUMINAMATH_CALUDE_sock_combination_count_l2156_215656


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l2156_215688

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- all terms are positive
  ∃ d : ℝ, a = b - d ∧ c = b + d →  -- forms an arithmetic sequence
  a * b * c = 125 →  -- product is 125
  b ≥ 5 ∧ (∀ b' : ℝ, b' ≥ 5 → b' = 5) :=  -- b is at least 5, and 5 is the smallest such value
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l2156_215688


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2156_215622

theorem largest_prime_factor_of_expression : 
  (Nat.factors (12^3 + 8^4 - 4^5)).maximum = some 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2156_215622


namespace NUMINAMATH_CALUDE_brian_watching_time_l2156_215605

def cat_video_length : ℕ := 4

def dog_video_length (cat_length : ℕ) : ℕ := 2 * cat_length

def gorilla_video_length (cat_length dog_length : ℕ) : ℕ := 2 * (cat_length + dog_length)

def total_watching_time (cat_length dog_length gorilla_length : ℕ) : ℕ :=
  cat_length + dog_length + gorilla_length

theorem brian_watching_time :
  total_watching_time cat_video_length 
    (dog_video_length cat_video_length) 
    (gorilla_video_length cat_video_length (dog_video_length cat_video_length)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_brian_watching_time_l2156_215605


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2156_215692

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2156_215692


namespace NUMINAMATH_CALUDE_subtract_linear_equations_l2156_215663

theorem subtract_linear_equations :
  let eq1 : ℝ → ℝ → ℝ := λ x y => 2 * x + 3 * y
  let eq2 : ℝ → ℝ → ℝ := λ x y => 5 * x + 3 * y
  let result : ℝ → ℝ := λ x => -3 * x
  (∀ x y, eq1 x y = 11) →
  (∀ x y, eq2 x y = -7) →
  (∀ x, result x = 18) →
  ∀ x y, eq1 x y - eq2 x y = result x :=
by
  sorry

end NUMINAMATH_CALUDE_subtract_linear_equations_l2156_215663


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l2156_215614

theorem distance_to_x_axis (P : ℝ × ℝ) (h : P = (-2, 3)) : 
  abs (P.2) = 3 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l2156_215614


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l2156_215657

theorem square_perimeter_relation (x y : Real) 
  (hx : x > 0) 
  (hy : y > 0) 
  (perimeter_x : 4 * x = 32) 
  (area_relation : y^2 = (1/3) * x^2) : 
  4 * y = (32 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l2156_215657


namespace NUMINAMATH_CALUDE_count_multiples_of_ten_l2156_215668

theorem count_multiples_of_ten : ∃ n : ℕ, n = (Finset.filter (λ x => x % 10 = 0 ∧ x > 9 ∧ x < 101) (Finset.range 101)).card ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_ten_l2156_215668


namespace NUMINAMATH_CALUDE_set_representation_l2156_215673

theorem set_representation :
  {x : ℕ | 8 < x ∧ x < 12} = {9, 10, 11} := by
  sorry

end NUMINAMATH_CALUDE_set_representation_l2156_215673


namespace NUMINAMATH_CALUDE_unique_a10a_divisible_by_12_l2156_215693

def is_form_a10a (n : ℕ) : Prop :=
  ∃ a : ℕ, a < 10 ∧ n = 1000 * a + 100 + 10 + a

theorem unique_a10a_divisible_by_12 :
  ∃! n : ℕ, is_form_a10a n ∧ n % 12 = 0 ∧ n = 4104 := by sorry

end NUMINAMATH_CALUDE_unique_a10a_divisible_by_12_l2156_215693


namespace NUMINAMATH_CALUDE_javier_to_anna_fraction_l2156_215623

/-- Represents the number of stickers each person has -/
structure StickerCount where
  lee : ℕ
  anna : ℕ
  javier : ℕ

/-- Calculates the fraction of stickers Javier should give to Anna -/
def fraction_to_anna (initial : StickerCount) (final : StickerCount) : ℚ :=
  (final.anna - initial.anna : ℚ) / initial.javier

/-- Theorem stating that Javier should give 0 fraction of his stickers to Anna -/
theorem javier_to_anna_fraction (l : ℕ) : 
  let initial := StickerCount.mk l (3 * l) (12 * l)
  let final := StickerCount.mk (2 * l) (3 * l) (6 * l)
  fraction_to_anna initial final = 0 := by
  sorry

#check javier_to_anna_fraction

end NUMINAMATH_CALUDE_javier_to_anna_fraction_l2156_215623


namespace NUMINAMATH_CALUDE_earnings_for_55_hours_l2156_215650

/-- Calculates the earnings for a given number of hours based on the described pay rate pattern -/
def earnings (hours : ℕ) : ℕ :=
  let cycleEarnings := (List.range 10).map (· + 1) |> List.sum
  let completeCycles := hours / 10
  completeCycles * cycleEarnings

/-- Proves that working for 55 hours with the given pay rate results in earning $275 -/
theorem earnings_for_55_hours :
  earnings 55 = 275 := by
  sorry

end NUMINAMATH_CALUDE_earnings_for_55_hours_l2156_215650


namespace NUMINAMATH_CALUDE_butter_cost_l2156_215637

theorem butter_cost (initial_amount spent_on_bread spent_on_juice remaining_amount : ℝ)
  (h1 : initial_amount = 15)
  (h2 : remaining_amount = 6)
  (h3 : spent_on_bread = 2)
  (h4 : spent_on_juice = 2 * spent_on_bread)
  : initial_amount - remaining_amount - spent_on_bread - spent_on_juice = 3 := by
  sorry

end NUMINAMATH_CALUDE_butter_cost_l2156_215637


namespace NUMINAMATH_CALUDE_triangle_inequality_l2156_215608

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = Real.pi

-- Theorem statement
theorem triangle_inequality (t : Triangle) (x y z : Real) :
  x^2 + y^2 + z^2 ≥ 2*x*y*(Real.cos t.C) + 2*y*z*(Real.cos t.A) + 2*z*x*(Real.cos t.B) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2156_215608


namespace NUMINAMATH_CALUDE_student_age_problem_l2156_215680

theorem student_age_problem (total_students : ℕ) (avg_age : ℕ) 
  (group1_size : ℕ) (group1_avg : ℕ) 
  (group2_size : ℕ) (group2_avg : ℕ) 
  (group3_size : ℕ) (group3_avg : ℕ) : 
  total_students = 25 →
  avg_age = 24 →
  group1_size = 8 →
  group1_avg = 22 →
  group2_size = 10 →
  group2_avg = 20 →
  group3_size = 6 →
  group3_avg = 28 →
  group1_size + group2_size + group3_size + 1 = total_students →
  (total_students * avg_age) - 
  (group1_size * group1_avg + group2_size * group2_avg + group3_size * group3_avg) = 56 :=
by sorry

end NUMINAMATH_CALUDE_student_age_problem_l2156_215680


namespace NUMINAMATH_CALUDE_vinces_bus_ride_l2156_215634

theorem vinces_bus_ride (zachary_ride : ℝ) (vince_difference : ℝ) :
  zachary_ride = 0.5 →
  vince_difference = 0.125 →
  zachary_ride + vince_difference = 0.625 :=
by
  sorry

end NUMINAMATH_CALUDE_vinces_bus_ride_l2156_215634


namespace NUMINAMATH_CALUDE_jessica_has_two_balloons_l2156_215689

/-- The number of blue balloons Jessica has -/
def jessicas_balloons (joan_initial : ℕ) (popped : ℕ) (total_now : ℕ) : ℕ :=
  total_now - (joan_initial - popped)

/-- Theorem: Jessica has 2 blue balloons -/
theorem jessica_has_two_balloons :
  jessicas_balloons 9 5 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_has_two_balloons_l2156_215689


namespace NUMINAMATH_CALUDE_ad_length_l2156_215699

-- Define the points
variable (A B C D M : Point)

-- Define the length function
variable (length : Point → Point → ℝ)

-- State the conditions
variable (trisect : length A B = length B C ∧ length B C = length C D)
variable (midpoint : length A M = length M D)
variable (mc_length : length M C = 10)

-- Theorem statement
theorem ad_length : length A D = 60 := by sorry

end NUMINAMATH_CALUDE_ad_length_l2156_215699


namespace NUMINAMATH_CALUDE_student_committee_candidates_l2156_215609

theorem student_committee_candidates :
  ∃ n : ℕ, 
    n > 0 ∧ 
    n * (n - 1) = 132 ∧ 
    (∀ m : ℕ, m > 0 ∧ m * (m - 1) = 132 → m = n) ∧
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_student_committee_candidates_l2156_215609


namespace NUMINAMATH_CALUDE_gcd_10011_15015_l2156_215631

theorem gcd_10011_15015 : Nat.gcd 10011 15015 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_10011_15015_l2156_215631


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2156_215658

def a (x : ℝ) : ℝ × ℝ := (x - 1, x)
def b : ℝ × ℝ := (-1, 2)

theorem parallel_vectors_x_value :
  (∃ (k : ℝ), a x = k • b) → x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2156_215658


namespace NUMINAMATH_CALUDE_theo_eggs_needed_l2156_215648

/-- Represents the number of eggs in an omelette -/
inductive OmeletteType
| Three
| Four

/-- Represents an hour of operation with customer orders -/
structure HourOrder where
  customers : ℕ
  omeletteType : OmeletteType

/-- Calculates the total number of eggs needed for all omelettes -/
def totalEggsNeeded (orders : List HourOrder) : ℕ :=
  orders.foldl (fun acc order =>
    acc + order.customers * match order.omeletteType with
      | OmeletteType.Three => 3
      | OmeletteType.Four => 4
  ) 0

/-- The main theorem: given the specific orders, the total eggs needed is 84 -/
theorem theo_eggs_needed :
  let orders := [
    HourOrder.mk 5 OmeletteType.Three,
    HourOrder.mk 7 OmeletteType.Four,
    HourOrder.mk 3 OmeletteType.Three,
    HourOrder.mk 8 OmeletteType.Four
  ]
  totalEggsNeeded orders = 84 := by
  sorry

#eval totalEggsNeeded [
  HourOrder.mk 5 OmeletteType.Three,
  HourOrder.mk 7 OmeletteType.Four,
  HourOrder.mk 3 OmeletteType.Three,
  HourOrder.mk 8 OmeletteType.Four
]

end NUMINAMATH_CALUDE_theo_eggs_needed_l2156_215648


namespace NUMINAMATH_CALUDE_zunyi_temperature_difference_l2156_215647

/-- The temperature difference between the highest and lowest temperatures in Zunyi City on June 1, 2019 -/
def temperature_difference (highest lowest : ℝ) : ℝ := highest - lowest

/-- Theorem stating that the temperature difference is 10°C given the highest and lowest temperatures -/
theorem zunyi_temperature_difference :
  let highest : ℝ := 25
  let lowest : ℝ := 15
  temperature_difference highest lowest = 10 := by
  sorry

end NUMINAMATH_CALUDE_zunyi_temperature_difference_l2156_215647


namespace NUMINAMATH_CALUDE_smallest_positive_integer_linear_combination_l2156_215604

theorem smallest_positive_integer_linear_combination : 
  ∃ (k : ℕ), k > 0 ∧ (∀ (m n p : ℤ), ∃ (x : ℤ), 1234 * m + 56789 * n + 345 * p = x * k) ∧ 
  (∀ (l : ℕ), l > 0 → (∀ (m n p : ℤ), ∃ (x : ℤ), 1234 * m + 56789 * n + 345 * p = x * l) → l ≥ k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_linear_combination_l2156_215604


namespace NUMINAMATH_CALUDE_xiaoming_win_probability_l2156_215653

/-- The probability of winning a single round for each player -/
def win_prob : ℚ := 1 / 2

/-- The number of rounds Xiaoming needs to win to ultimately win -/
def xiaoming_rounds_needed : ℕ := 2

/-- The number of rounds Xiaojie needs to win to ultimately win -/
def xiaojie_rounds_needed : ℕ := 3

/-- The probability that Xiaoming wins 2 consecutive rounds and ultimately wins -/
def xiaoming_win_prob : ℚ := 7 / 16

theorem xiaoming_win_probability : 
  xiaoming_win_prob = 
    win_prob ^ xiaoming_rounds_needed + 
    xiaoming_rounds_needed * win_prob ^ (xiaoming_rounds_needed + 1) + 
    win_prob ^ (xiaoming_rounds_needed + xiaojie_rounds_needed - 1) :=
by sorry

end NUMINAMATH_CALUDE_xiaoming_win_probability_l2156_215653


namespace NUMINAMATH_CALUDE_election_majority_l2156_215646

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 1400 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ).floor - 
  ((1 - winning_percentage) * total_votes : ℚ).floor = 280 := by
  sorry

end NUMINAMATH_CALUDE_election_majority_l2156_215646


namespace NUMINAMATH_CALUDE_inequality_proof_l2156_215687

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2156_215687


namespace NUMINAMATH_CALUDE_all_numbers_even_l2156_215642

theorem all_numbers_even 
  (A B C D E : ℤ) 
  (h1 : Even (A + B + C))
  (h2 : Even (A + B + D))
  (h3 : Even (A + B + E))
  (h4 : Even (A + C + D))
  (h5 : Even (A + C + E))
  (h6 : Even (A + D + E))
  (h7 : Even (B + C + D))
  (h8 : Even (B + C + E))
  (h9 : Even (B + D + E))
  (h10 : Even (C + D + E)) :
  Even A ∧ Even B ∧ Even C ∧ Even D ∧ Even E := by
  sorry

#check all_numbers_even

end NUMINAMATH_CALUDE_all_numbers_even_l2156_215642


namespace NUMINAMATH_CALUDE_team_selection_count_l2156_215627

/-- Represents the number of ways to select a team under given conditions -/
def selectTeam (totalMale totalFemale teamSize : ℕ) 
               (maleCaptains femaleCaptains : ℕ) : ℕ := 
  Nat.choose (totalMale + totalFemale - 1) (teamSize - 1) + 
  Nat.choose (totalMale + totalFemale - maleCaptains - 1) (teamSize - 1) - 
  Nat.choose (totalMale - maleCaptains) (teamSize - 1)

/-- Theorem stating the number of ways to select a team of 5 from 6 male (1 captain) 
    and 4 female (1 captain) athletes, including at least 1 female and a captain -/
theorem team_selection_count : 
  selectTeam 6 4 5 1 1 = 191 := by sorry

end NUMINAMATH_CALUDE_team_selection_count_l2156_215627


namespace NUMINAMATH_CALUDE_max_cards_48_36_16_12_l2156_215617

/-- The maximum number of rectangular cards that can be cut from a rectangular cardboard --/
def max_cards (cardboard_length cardboard_width card_length card_width : ℕ) : ℕ :=
  max ((cardboard_length / card_length) * (cardboard_width / card_width))
      ((cardboard_length / card_width) * (cardboard_width / card_length))

/-- Theorem: The maximum number of 16 cm x 12 cm cards that can be cut from a 48 cm x 36 cm cardboard is 9 --/
theorem max_cards_48_36_16_12 :
  max_cards 48 36 16 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_cards_48_36_16_12_l2156_215617


namespace NUMINAMATH_CALUDE_distance_between_points_l2156_215611

-- Define the equation of the curve
def on_curve (x y : ℝ) : Prop := y^2 + x^3 = 2*x*y + 4

-- Define the theorem
theorem distance_between_points (e a b : ℝ) 
  (h1 : on_curve e a) 
  (h2 : on_curve e b) 
  (h3 : a ≠ b) : 
  |a - b| = 2 * Real.sqrt (e^2 - e^3 + 4) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2156_215611


namespace NUMINAMATH_CALUDE_oil_distribution_l2156_215635

/-- Represents the problem of minimizing the number of small barrels --/
def MinimizeSmallBarrels (total_oil : ℕ) (large_barrel_capacity : ℕ) (small_barrel_capacity : ℕ) : Prop :=
  ∃ (large_barrels small_barrels : ℕ),
    large_barrel_capacity * large_barrels + small_barrel_capacity * small_barrels = total_oil ∧
    small_barrels = 1 ∧
    ∀ (l s : ℕ), large_barrel_capacity * l + small_barrel_capacity * s = total_oil →
      s ≥ small_barrels

theorem oil_distribution :
  MinimizeSmallBarrels 745 11 7 :=
sorry

end NUMINAMATH_CALUDE_oil_distribution_l2156_215635


namespace NUMINAMATH_CALUDE_number_of_indoor_players_l2156_215686

/-- Given a group of players with outdoor, indoor, and both categories, 
    calculate the number of indoor players. -/
theorem number_of_indoor_players 
  (total : ℕ) 
  (outdoor : ℕ) 
  (both : ℕ) 
  (h1 : total = 400) 
  (h2 : outdoor = 350) 
  (h3 : both = 60) : 
  ∃ indoor : ℕ, indoor = 110 ∧ total = outdoor + indoor - both :=
sorry

end NUMINAMATH_CALUDE_number_of_indoor_players_l2156_215686


namespace NUMINAMATH_CALUDE_functional_equation_1_bijective_functional_equation_2_neither_functional_equation_3_neither_functional_equation_4_neither_l2156_215644

-- 1. f(x+f(y))=2f(x)+y is bijective
theorem functional_equation_1_bijective (f : ℝ → ℝ) :
  (∀ x y, f (x + f y) = 2 * f x + y) → Function.Bijective f :=
sorry

-- 2. f(f(x))=0 is neither injective nor surjective
theorem functional_equation_2_neither (f : ℝ → ℝ) :
  (∀ x, f (f x) = 0) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

-- 3. f(f(x))=sin(x) is neither injective nor surjective
theorem functional_equation_3_neither (f : ℝ → ℝ) :
  (∀ x, f (f x) = Real.sin x) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

-- 4. f(x+y)=f(x)f(y) is neither injective nor surjective
theorem functional_equation_4_neither (f : ℝ → ℝ) :
  (∀ x y, f (x + y) = f x * f y) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

end NUMINAMATH_CALUDE_functional_equation_1_bijective_functional_equation_2_neither_functional_equation_3_neither_functional_equation_4_neither_l2156_215644


namespace NUMINAMATH_CALUDE_digit_count_of_2_15_3_2_5_12_l2156_215672

theorem digit_count_of_2_15_3_2_5_12 : 
  (Nat.digits 10 (2^15 * 3^2 * 5^12)).length = 14 := by
  sorry

end NUMINAMATH_CALUDE_digit_count_of_2_15_3_2_5_12_l2156_215672


namespace NUMINAMATH_CALUDE_necessary_sufficient_condition_l2156_215610

theorem necessary_sufficient_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) ↔
  (a ≤ -2 ∨ a = 1) :=
sorry

end NUMINAMATH_CALUDE_necessary_sufficient_condition_l2156_215610


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l2156_215660

theorem largest_prime_factor_of_1001 : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ 1001 ∧ ∀ (q : ℕ), q.Prime → q ∣ 1001 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l2156_215660


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l2156_215632

theorem max_value_of_trigonometric_expression :
  ∀ α : Real, 0 ≤ α → α ≤ π / 2 →
  (1 / (Real.sin α ^ 4 + Real.cos α ^ 4) ≤ 2) ∧
  (∃ α₀, 0 ≤ α₀ ∧ α₀ ≤ π / 2 ∧ 1 / (Real.sin α₀ ^ 4 + Real.cos α₀ ^ 4) = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l2156_215632


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2156_215633

def diophantine_equation (x y : ℤ) : Prop :=
  2 * x^4 - 4 * y^4 - 7 * x^2 * y^2 - 27 * x^2 + 63 * y^2 + 85 = 0

def solution_set : Set (ℤ × ℤ) :=
  {(3, 1), (3, -1), (-3, 1), (-3, -1), (2, 3), (2, -3), (-2, 3), (-2, -3)}

theorem diophantine_equation_solutions :
  ∀ (x y : ℤ), diophantine_equation x y ↔ (x, y) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2156_215633


namespace NUMINAMATH_CALUDE_percentage_of_french_speakers_l2156_215641

theorem percentage_of_french_speakers (total_employees : ℝ) (h1 : total_employees > 0) :
  let men_percentage : ℝ := 70
  let women_percentage : ℝ := 100 - men_percentage
  let men_french_speakers_percentage : ℝ := 50
  let women_non_french_speakers_percentage : ℝ := 83.33333333333331
  let men : ℝ := (men_percentage / 100) * total_employees
  let women : ℝ := (women_percentage / 100) * total_employees
  let men_french_speakers : ℝ := (men_french_speakers_percentage / 100) * men
  let women_french_speakers : ℝ := (1 - women_non_french_speakers_percentage / 100) * women
  let total_french_speakers : ℝ := men_french_speakers + women_french_speakers
  (total_french_speakers / total_employees) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_french_speakers_l2156_215641


namespace NUMINAMATH_CALUDE_ninas_inheritance_l2156_215626

theorem ninas_inheritance (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧                    -- Both investments are positive
  0.06 * x + 0.08 * y = 860 ∧        -- Total yearly interest
  (x = 5000 ∨ y = 5000) →            -- $5000 invested at one rate
  x + y = 12000 :=                   -- Total inheritance
by sorry

end NUMINAMATH_CALUDE_ninas_inheritance_l2156_215626


namespace NUMINAMATH_CALUDE_prob_at_least_one_success_l2156_215669

/-- The probability of success for a single attempt -/
def p : ℝ := 0.5

/-- The number of attempts for each athlete -/
def attempts_per_athlete : ℕ := 2

/-- The total number of attempts -/
def total_attempts : ℕ := 2 * attempts_per_athlete

/-- The probability of at least one successful attempt out of the total attempts -/
theorem prob_at_least_one_success :
  1 - (1 - p) ^ total_attempts = 0.9375 := by
  sorry


end NUMINAMATH_CALUDE_prob_at_least_one_success_l2156_215669


namespace NUMINAMATH_CALUDE_solution_set_part_I_value_of_a_part_II_l2156_215619

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 1

-- Part I
theorem solution_set_part_I (a : ℝ) (h : a > 1) :
  let f := f a
  a = 2 →
  {x : ℝ | f x ≥ 4 - |x - 4|} = {x : ℝ | x ≥ 11/2 ∨ x ≤ 1/2} :=
sorry

-- Part II
theorem value_of_a_part_II (a : ℝ) (h : a > 1) :
  let f := f a
  ({x : ℝ | |f (2*x + a) - 2*f x| ≤ 1} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 1}) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part_I_value_of_a_part_II_l2156_215619


namespace NUMINAMATH_CALUDE_ellipse_tangent_to_circle_l2156_215696

noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 2, 0)

def ellipse (p : ℝ × ℝ) : Prop :=
  p.1^2 / 4 + p.2^2 / 2 = 1

def on_line_y_eq_neg_2 (p : ℝ × ℝ) : Prop :=
  p.2 = -2

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def tangent_to_circle (l : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ l ∧ p.1^2 + p.2^2 = 2 ∧
    ∀ q, q ∈ l → q.1^2 + q.2^2 ≥ 2

theorem ellipse_tangent_to_circle :
  ∀ E F : ℝ × ℝ,
    ellipse E →
    on_line_y_eq_neg_2 F →
    perpendicular E F →
    tangent_to_circle {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • E + t • F} :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_to_circle_l2156_215696


namespace NUMINAMATH_CALUDE_series_sum_l2156_215629

theorem series_sum : ∑' n, (n : ℝ) / 5^n = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_series_sum_l2156_215629


namespace NUMINAMATH_CALUDE_pen_count_problem_l2156_215667

theorem pen_count_problem :
  ∃! X : ℕ, 1 ≤ X ∧ X < 100 ∧ 
  X % 9 = 1 ∧ X % 5 = 3 ∧ X % 2 = 1 ∧ 
  X = 73 :=
by sorry

end NUMINAMATH_CALUDE_pen_count_problem_l2156_215667


namespace NUMINAMATH_CALUDE_log_relation_l2156_215607

theorem log_relation (c b : ℝ) (hc : c = Real.log 81 / Real.log 4) (hb : b = Real.log 3 / Real.log 2) : 
  c = 2 * b := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l2156_215607


namespace NUMINAMATH_CALUDE_tom_gave_two_seashells_l2156_215618

/-- The number of seashells Tom gave to Jessica -/
def seashells_given (initial_seashells current_seashells : ℕ) : ℕ :=
  initial_seashells - current_seashells

/-- Theorem stating that Tom gave 2 seashells to Jessica -/
theorem tom_gave_two_seashells :
  seashells_given 5 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tom_gave_two_seashells_l2156_215618


namespace NUMINAMATH_CALUDE_reciprocal_sum_fractions_l2156_215645

theorem reciprocal_sum_fractions : 
  (1 / (1/3 + 1/4 - 1/12) : ℚ) = 2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_fractions_l2156_215645


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2156_215636

-- Define the set M as the domain of y = log(1-x)
def M : Set ℝ := {x : ℝ | x < 1}

-- Define the set N = {y | y = e^x, x ∈ ℝ}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.exp x}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2156_215636


namespace NUMINAMATH_CALUDE_tetrahedron_triangle_existence_l2156_215612

/-- Represents a tetrahedron with edge lengths -/
structure Tetrahedron where
  edges : Fin 6 → ℝ
  edge_positive : ∀ i, edges i > 0

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Main theorem: In any tetrahedron, there exists a vertex such that 
    the edges connected to it can form a triangle -/
theorem tetrahedron_triangle_existence (t : Tetrahedron) : 
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    can_form_triangle (t.edges i) (t.edges j) (t.edges k) :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_triangle_existence_l2156_215612


namespace NUMINAMATH_CALUDE_child_ticket_cost_is_4_l2156_215613

/-- The cost of a child's ticket at a ball game -/
def child_ticket_cost (num_adults num_children : ℕ) (adult_ticket_cost total_bill : ℚ) : ℚ :=
  (total_bill - num_adults * adult_ticket_cost) / num_children

theorem child_ticket_cost_is_4 :
  child_ticket_cost 10 11 8 124 = 4 := by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_is_4_l2156_215613


namespace NUMINAMATH_CALUDE_upload_time_calculation_l2156_215670

/-- Represents the time in minutes required to upload a file -/
def uploadTime (fileSize : ℕ) (uploadSpeed : ℕ) : ℕ :=
  fileSize / uploadSpeed

/-- Proves that uploading a 160 MB file at 8 MB/min takes 20 minutes -/
theorem upload_time_calculation :
  uploadTime 160 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_upload_time_calculation_l2156_215670


namespace NUMINAMATH_CALUDE_min_lifts_for_equal_weight_l2156_215652

/-- The minimum number of lifts required to match or exceed the initial total weight -/
def min_lifts (initial_weight : ℕ) (initial_reps : ℕ) (new_weight : ℕ) (new_count : ℕ) : ℕ :=
  ((initial_weight * initial_reps + new_weight - 1) / new_weight : ℕ)

theorem min_lifts_for_equal_weight :
  min_lifts 75 10 80 4 = 10 := by sorry

end NUMINAMATH_CALUDE_min_lifts_for_equal_weight_l2156_215652
