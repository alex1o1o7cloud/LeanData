import Mathlib

namespace NUMINAMATH_CALUDE_sector_central_angle_l3397_339728

/-- Given a sector of a circle with arc length and area both equal to 5,
    prove that its central angle is 2.5 radians. -/
theorem sector_central_angle (r : ℝ) (θ : ℝ) : 
  r > 0 → 
  r * θ = 5 →  -- arc length formula
  1/2 * r^2 * θ = 5 →  -- sector area formula
  θ = 2.5 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3397_339728


namespace NUMINAMATH_CALUDE_one_correct_judgment_l3397_339767

theorem one_correct_judgment :
  let judgment1 := ∀ a b : ℝ, a + b ≠ 6 → a ≠ 3 ∨ b ≠ 3
  let judgment2 := ∀ p q : Prop, (p ∨ q) → (p ∧ q)
  let judgment3 := (¬ ∀ a b : ℝ, a^2 + b^2 ≥ 2*(a - b - 1)) ↔ (∃ a b : ℝ, a^2 + b^2 ≤ 2*(a - b - 1))
  let judgment4 := (judgment1 ∧ ¬judgment2 ∧ ¬judgment3)
  judgment4 := by sorry

end NUMINAMATH_CALUDE_one_correct_judgment_l3397_339767


namespace NUMINAMATH_CALUDE_impossible_arrangement_l3397_339772

/-- Represents the type of student: Knight (always tells the truth) or Liar (always lies) -/
inductive StudentType
| Knight
| Liar

/-- Represents a desk with two students -/
structure Desk where
  student1 : StudentType
  student2 : StudentType

/-- The initial arrangement of students -/
def initial_arrangement (desks : List Desk) : Prop :=
  desks.length = 13 ∧ 
  ∀ d ∈ desks, (d.student1 = StudentType.Knight ∧ d.student2 = StudentType.Liar) ∨
                (d.student1 = StudentType.Liar ∧ d.student2 = StudentType.Knight)

/-- The final arrangement of students -/
def final_arrangement (desks : List Desk) : Prop :=
  desks.length = 13 ∧
  ∀ d ∈ desks, d.student1 = d.student2

/-- Theorem stating the impossibility of the final arrangement -/
theorem impossible_arrangement :
  ∀ (initial_desks final_desks : List Desk),
    initial_arrangement initial_desks →
    ¬(final_arrangement final_desks) :=
by sorry


end NUMINAMATH_CALUDE_impossible_arrangement_l3397_339772


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3397_339712

theorem sphere_surface_area (v : ℝ) (h : v = 72 * Real.pi) :
  ∃ (r : ℝ), v = (4 / 3) * Real.pi * r^3 ∧ 4 * Real.pi * r^2 = 4 * Real.pi * (2916 ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3397_339712


namespace NUMINAMATH_CALUDE_exactly_two_integers_l3397_339734

/-- Define the function that we want to check for integrality --/
def f (n : ℕ) : ℚ :=
  (Nat.factorial (n^3 - 1)) / ((Nat.factorial n)^(n + 2))

/-- Predicate to check if a number is in the range [1, 50] --/
def in_range (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 50

/-- Predicate to check if f(n) is an integer --/
def is_integer (n : ℕ) : Prop :=
  ∃ k : ℤ, f n = k

/-- Main theorem statement --/
theorem exactly_two_integers :
  (∃ (S : Finset ℕ), S.card = 2 ∧ 
    (∀ n, n ∈ S ↔ (in_range n ∧ is_integer n)) ∧
    (∀ n, in_range n → is_integer n → n ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_integers_l3397_339734


namespace NUMINAMATH_CALUDE_cafeteria_green_apples_l3397_339741

/-- Prove that the number of green apples ordered by the cafeteria is 23 -/
theorem cafeteria_green_apples :
  let red_apples : ℕ := 33
  let students_wanting_fruit : ℕ := 21
  let extra_apples : ℕ := 35
  let green_apples : ℕ := 23
  (red_apples + green_apples - students_wanting_fruit = extra_apples) →
  green_apples = 23 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_green_apples_l3397_339741


namespace NUMINAMATH_CALUDE_power_of_three_mod_eight_l3397_339745

theorem power_of_three_mod_eight : 3^2010 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eight_l3397_339745


namespace NUMINAMATH_CALUDE_expression_evaluation_l3397_339701

theorem expression_evaluation :
  let a : ℝ := Real.sqrt 3 - 3
  (3 - a) / (2 * a - 4) / (a + 2 - 5 / (a - 2)) = -Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3397_339701


namespace NUMINAMATH_CALUDE_chess_team_arrangements_l3397_339711

/-- Represents the number of boys in the chess team -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the chess team -/
def num_girls : ℕ := 3

/-- Represents the total number of students in the chess team -/
def total_students : ℕ := num_boys + num_girls

/-- Represents the number of ways to arrange the ends of the row -/
def end_arrangements : ℕ := 2 * num_boys * num_girls

/-- Represents the number of ways to arrange the middle of the row -/
def middle_arrangements : ℕ := 2 * 2

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := end_arrangements * middle_arrangements

theorem chess_team_arrangements :
  total_arrangements = 72 :=
sorry

end NUMINAMATH_CALUDE_chess_team_arrangements_l3397_339711


namespace NUMINAMATH_CALUDE_inequality_solution_l3397_339758

theorem inequality_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3397_339758


namespace NUMINAMATH_CALUDE_square_perimeters_sum_l3397_339773

theorem square_perimeters_sum (a b : ℝ) (h1 : a ^ 2 + b ^ 2 = 145) (h2 : a ^ 2 - b ^ 2 = 25) :
  4 * Real.sqrt a ^ 2 + 4 * Real.sqrt b ^ 2 = 4 * Real.sqrt 85 + 4 * Real.sqrt 60 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeters_sum_l3397_339773


namespace NUMINAMATH_CALUDE_route_down_length_is_18_l3397_339735

/-- A hiking trip up and down a mountain -/
structure HikingTrip where
  rate_up : ℝ
  time_up : ℝ
  rate_down_factor : ℝ
  time_down : ℝ

/-- The length of the route down the mountain -/
def route_down_length (trip : HikingTrip) : ℝ :=
  trip.rate_up * trip.rate_down_factor * trip.time_down

/-- Theorem stating the length of the route down the mountain -/
theorem route_down_length_is_18 (trip : HikingTrip) 
  (h1 : trip.time_up = trip.time_down)
  (h2 : trip.rate_down_factor = 1.5)
  (h3 : trip.rate_up = 6)
  (h4 : trip.time_up = 2) : 
  route_down_length trip = 18 := by
  sorry

#eval route_down_length ⟨6, 2, 1.5, 2⟩

end NUMINAMATH_CALUDE_route_down_length_is_18_l3397_339735


namespace NUMINAMATH_CALUDE_total_onions_is_fifteen_l3397_339718

/-- The number of onions grown by Nancy -/
def nancy_onions : ℕ := 2

/-- The number of onions grown by Dan -/
def dan_onions : ℕ := 9

/-- The number of onions grown by Mike -/
def mike_onions : ℕ := 4

/-- The number of days they worked on the farm -/
def days_worked : ℕ := 6

/-- The total number of onions grown by Nancy, Dan, and Mike -/
def total_onions : ℕ := nancy_onions + dan_onions + mike_onions

theorem total_onions_is_fifteen : total_onions = 15 := by sorry

end NUMINAMATH_CALUDE_total_onions_is_fifteen_l3397_339718


namespace NUMINAMATH_CALUDE_opposite_abs_equal_l3397_339740

theorem opposite_abs_equal (x : ℝ) : |x| = |-x| := by sorry

end NUMINAMATH_CALUDE_opposite_abs_equal_l3397_339740


namespace NUMINAMATH_CALUDE_exactly_one_correct_l3397_339770

/-- The probability that exactly one of three independent events occurs, given their individual probabilities -/
theorem exactly_one_correct (pA pB pC : ℝ) 
  (hA : 0 ≤ pA ∧ pA ≤ 1) 
  (hB : 0 ≤ pB ∧ pB ≤ 1) 
  (hC : 0 ≤ pC ∧ pC ≤ 1) 
  (hpA : pA = 3/4) 
  (hpB : pB = 2/3) 
  (hpC : pC = 2/3) : 
  pA * (1 - pB) * (1 - pC) + (1 - pA) * pB * (1 - pC) + (1 - pA) * (1 - pB) * pC = 7/36 := by
  sorry

#check exactly_one_correct

end NUMINAMATH_CALUDE_exactly_one_correct_l3397_339770


namespace NUMINAMATH_CALUDE_singles_on_itunes_l3397_339797

def total_songs : ℕ := 55
def albums_15_songs : ℕ := 2
def songs_per_album_15 : ℕ := 15
def albums_20_songs : ℕ := 1
def songs_per_album_20 : ℕ := 20

theorem singles_on_itunes : 
  total_songs - (albums_15_songs * songs_per_album_15 + albums_20_songs * songs_per_album_20) = 5 := by
  sorry

end NUMINAMATH_CALUDE_singles_on_itunes_l3397_339797


namespace NUMINAMATH_CALUDE_non_negative_for_all_non_negative_exists_l3397_339791

-- Define the function f
def f (x m : ℝ) : ℝ := x^2 - 2*x + m

-- Theorem for part (1)
theorem non_negative_for_all (m : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f x m ≥ 0) ↔ m ≥ 1 :=
sorry

-- Theorem for part (2)
theorem non_negative_exists (m : ℝ) :
  (∃ x ∈ Set.Icc 0 3, f x m ≥ 0) ↔ m ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_non_negative_for_all_non_negative_exists_l3397_339791


namespace NUMINAMATH_CALUDE_base_eight_47_equals_39_l3397_339793

/-- Converts a two-digit base-eight number to base-ten -/
def base_eight_to_ten (a b : Nat) : Nat :=
  a * 8 + b

/-- The base-eight number 47 is equal to the base-ten number 39 -/
theorem base_eight_47_equals_39 : base_eight_to_ten 4 7 = 39 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_47_equals_39_l3397_339793


namespace NUMINAMATH_CALUDE_triangle_side_length_l3397_339744

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π / 2 →
  C = 4 * A →
  a = 21 →
  c = 54 →
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ 8 * x^2 - 12 * x - 4.5714 = 0 ∧
    b = 21 * (16 * x^2 - 20 * x + 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3397_339744


namespace NUMINAMATH_CALUDE_complex_modulus_l3397_339755

theorem complex_modulus (z : ℂ) (h : z * Complex.I = -2 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3397_339755


namespace NUMINAMATH_CALUDE_odd_function_and_monotonicity_l3397_339749

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 1) / (x + a)

theorem odd_function_and_monotonicity (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 0 ∧ ∀ x y, 0 < x → x < y → f a x < f a y) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_and_monotonicity_l3397_339749


namespace NUMINAMATH_CALUDE_average_student_height_l3397_339725

/-- Calculates the average height of all students given the average heights of males and females and the ratio of males to females. -/
theorem average_student_height
  (avg_female_height : ℝ)
  (avg_male_height : ℝ)
  (male_to_female_ratio : ℝ)
  (h1 : avg_female_height = 170)
  (h2 : avg_male_height = 185)
  (h3 : male_to_female_ratio = 2) :
  (male_to_female_ratio * avg_male_height + avg_female_height) / (male_to_female_ratio + 1) = 180 :=
by sorry

end NUMINAMATH_CALUDE_average_student_height_l3397_339725


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_180_and_n_l3397_339719

theorem greatest_common_divisor_of_180_and_n (n : ℕ) : 
  (∃ (d₁ d₂ d₃ : ℕ), d₁ < d₂ ∧ d₂ < d₃ ∧ 
   {d : ℕ | d ∣ 180 ∧ d ∣ n} = {d₁, d₂, d₃}) →
  (Nat.gcd 180 n = 9) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_180_and_n_l3397_339719


namespace NUMINAMATH_CALUDE_fixed_stable_points_range_l3397_339790

/-- The function f(x) = a x^2 - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

/-- The set of fixed points of f -/
def fixedPoints (a : ℝ) : Set ℝ := {x | f a x = x}

/-- The set of stable points of f -/
def stablePoints (a : ℝ) : Set ℝ := {x | f a (f a x) = x}

/-- Theorem stating the range of a for which the fixed points and stable points are equal and non-empty -/
theorem fixed_stable_points_range (a : ℝ) :
  (fixedPoints a = stablePoints a ∧ (fixedPoints a).Nonempty) ↔ -1/4 ≤ a ∧ a ≤ 3/4 :=
sorry

end NUMINAMATH_CALUDE_fixed_stable_points_range_l3397_339790


namespace NUMINAMATH_CALUDE_infiniteSeriesSum_l3397_339736

/-- The sum of the infinite series Σ(k/3^k) for k from 1 to ∞ -/
noncomputable def infiniteSeries : ℝ := ∑' k, k / (3 ^ k)

/-- Theorem: The sum of the infinite series Σ(k/3^k) for k from 1 to ∞ is equal to 3/4 -/
theorem infiniteSeriesSum : infiniteSeries = 3/4 := by sorry

end NUMINAMATH_CALUDE_infiniteSeriesSum_l3397_339736


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3397_339779

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 25 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define a chord passing through F1
def Chord (A B : ℝ × ℝ) : Prop := sorry

-- Define the perimeter of a triangle
def TrianglePerimeter (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter 
  (A B : ℝ × ℝ) 
  (h_ellipse : Ellipse A.1 A.2 ∧ Ellipse B.1 B.2)
  (h_chord : Chord A B) :
  TrianglePerimeter A B F2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l3397_339779


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3397_339713

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  let angle := 60 * π / 180
  let mag_a := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let mag_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  mag_a = 1 ∧ mag_b = 2 ∧
  a.1 * b.1 + a.2 * b.2 = mag_a * mag_b * Real.cos angle →
  Real.sqrt (((a.1 + b.1) ^ 2) + ((a.2 + b.2) ^ 2)) = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3397_339713


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3397_339706

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let x₁ := (-(-10) + Real.sqrt ((-10)^2 - 4*1*36)) / (2*1)
  let x₂ := (-(-10) - Real.sqrt ((-10)^2 - 4*1*36)) / (2*1)
  x₁ + x₂ = 10 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3397_339706


namespace NUMINAMATH_CALUDE_small_bottle_price_theorem_l3397_339787

/-- The price of a small bottle that results in the given average price -/
def price_small_bottle (large_quantity : ℕ) (small_quantity : ℕ) (large_price : ℚ) (average_price : ℚ) : ℚ :=
  ((large_quantity + small_quantity : ℚ) * average_price - large_quantity * large_price) / small_quantity

theorem small_bottle_price_theorem (large_quantity small_quantity : ℕ) (large_price average_price : ℚ) :
  large_quantity = 1365 →
  small_quantity = 720 →
  large_price = 189/100 →
  average_price = 173/100 →
  ∃ ε > 0, |price_small_bottle large_quantity small_quantity large_price average_price - 142/100| < ε :=
by sorry


end NUMINAMATH_CALUDE_small_bottle_price_theorem_l3397_339787


namespace NUMINAMATH_CALUDE_locus_of_M_l3397_339761

-- Define the points A, B, and M
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the angle function
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define the condition for point M
def satisfies_angle_condition (M : ℝ × ℝ) : Prop :=
  angle M B A = 2 * angle M A B

-- Define the locus conditions
def on_hyperbola (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  3 * x^2 - y^2 = 3 ∧ x > -1

def on_segment (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  y = 0 ∧ -1 < x ∧ x < 2

-- State the theorem
theorem locus_of_M (M : ℝ × ℝ) :
  satisfies_angle_condition M ↔ (on_hyperbola M ∨ on_segment M) :=
sorry

end NUMINAMATH_CALUDE_locus_of_M_l3397_339761


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l3397_339762

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) :
  (2 * r) ^ 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l3397_339762


namespace NUMINAMATH_CALUDE_lcm_of_fractions_l3397_339794

theorem lcm_of_fractions (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  lcm (1 / x) (lcm (1 / (x * y)) (1 / (x * y * z))) = 1 / (x * y * z) := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_fractions_l3397_339794


namespace NUMINAMATH_CALUDE_cafeteria_pies_l3397_339729

def initial_apples : ℕ := 372
def handed_out : ℕ := 135
def apples_per_pie : ℕ := 15

theorem cafeteria_pies : 
  (initial_apples - handed_out) / apples_per_pie = 15 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l3397_339729


namespace NUMINAMATH_CALUDE_pencils_in_pack_judys_pencil_pack_l3397_339777

/-- Calculates the number of pencils in a pack given Judy's pencil usage and spending habits. -/
theorem pencils_in_pack (pencils_per_week : ℕ) (days_per_week : ℕ) (cost_per_pack : ℕ) 
  (total_spent : ℕ) (total_days : ℕ) : ℕ :=
  let pencils_used := pencils_per_week * (total_days / days_per_week)
  let packs_bought := total_spent / cost_per_pack
  pencils_used / packs_bought

/-- Proves that there are 30 pencils in a pack based on Judy's usage and spending. -/
theorem judys_pencil_pack : pencils_in_pack 10 5 4 12 45 = 30 := by
  sorry

end NUMINAMATH_CALUDE_pencils_in_pack_judys_pencil_pack_l3397_339777


namespace NUMINAMATH_CALUDE_compressor_stations_theorem_l3397_339727

/-- Represents the configuration of three compressor stations -/
structure CompressorStations where
  x : ℝ  -- Distance between first and second stations
  y : ℝ  -- Distance between second and third stations
  z : ℝ  -- Distance between first and third stations
  a : ℝ  -- Additional parameter

/-- Conditions for the compressor stations configuration -/
def valid_configuration (c : CompressorStations) : Prop :=
  c.x + c.y = 3 * c.z ∧
  c.z + c.y = c.x + c.a ∧
  c.x + c.z = 60 ∧
  c.x > 0 ∧ c.y > 0 ∧ c.z > 0

/-- Theorem stating the valid range for parameter a and specific values when a = 42 -/
theorem compressor_stations_theorem :
  ∀ c : CompressorStations,
    valid_configuration c →
    (0 < c.a ∧ c.a < 60) ∧
    (c.a = 42 → c.x = 33 ∧ c.y = 48 ∧ c.z = 27) :=
by sorry

end NUMINAMATH_CALUDE_compressor_stations_theorem_l3397_339727


namespace NUMINAMATH_CALUDE_triangle_centroid_coordinates_l3397_339765

/-- The centroid of a triangle with vertices (2, 8), (6, 2), and (0, 4) has coordinates (8/3, 14/3). -/
theorem triangle_centroid_coordinates :
  let A : ℝ × ℝ := (2, 8)
  let B : ℝ × ℝ := (6, 2)
  let C : ℝ × ℝ := (0, 4)
  let centroid : ℝ × ℝ := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  centroid = (8/3, 14/3) := by
sorry

end NUMINAMATH_CALUDE_triangle_centroid_coordinates_l3397_339765


namespace NUMINAMATH_CALUDE_box_volume_conversion_l3397_339726

/-- Proves that a box with a volume of 216 cubic feet has a volume of 8 cubic yards. -/
theorem box_volume_conversion (box_volume_cubic_feet : ℝ) 
  (h1 : box_volume_cubic_feet = 216) : 
  box_volume_cubic_feet / 27 = 8 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_conversion_l3397_339726


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l3397_339788

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  is_arithmetic : ∀ n m : ℕ+, a (n + 1) - a n = a (m + 1) - a m

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_max_sum (seq : ArithmeticSequence) :
  (seq.a 1 + seq.a 4 + seq.a 7 = 99) →
  (seq.a 2 + seq.a 5 + seq.a 8 = 93) →
  (∀ n : ℕ+, S seq n ≤ S seq 20) →
  (∀ k : ℕ+, (∀ n : ℕ+, S seq n ≤ S seq k) → k = 20) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l3397_339788


namespace NUMINAMATH_CALUDE_fraction_simplification_l3397_339786

theorem fraction_simplification (a b : ℝ) (h : b ≠ 0) :
  (20 * a^4 * b) / (120 * a^3 * b^2) = a / (6 * b) ∧
  (20 * 2^4 * 3) / (120 * 2^3 * 3^2) = 1 / 9 := by
  sorry

#check fraction_simplification

end NUMINAMATH_CALUDE_fraction_simplification_l3397_339786


namespace NUMINAMATH_CALUDE_library_book_purchase_ratio_l3397_339756

theorem library_book_purchase_ratio :
  ∀ (initial_books last_year_purchase current_total : ℕ),
  initial_books = 100 →
  last_year_purchase = 50 →
  current_total = 300 →
  ∃ (this_year_purchase : ℕ),
    this_year_purchase = 3 * last_year_purchase ∧
    current_total = initial_books + last_year_purchase + this_year_purchase :=
by
  sorry

end NUMINAMATH_CALUDE_library_book_purchase_ratio_l3397_339756


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3397_339789

-- Problem 1
theorem problem_1 : -3 + (-2) * 5 - (-3) = -10 := by sorry

-- Problem 2
theorem problem_2 : -1^4 + ((-5)^2 - 3) / |(-2)| = 10 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3397_339789


namespace NUMINAMATH_CALUDE_total_dolls_count_l3397_339768

/-- The number of dolls in a big box -/
def dolls_per_big_box : ℕ := 7

/-- The number of dolls in a small box -/
def dolls_per_small_box : ℕ := 4

/-- The number of big boxes -/
def num_big_boxes : ℕ := 5

/-- The number of small boxes -/
def num_small_boxes : ℕ := 9

/-- The total number of dolls in all boxes -/
def total_dolls : ℕ := dolls_per_big_box * num_big_boxes + dolls_per_small_box * num_small_boxes

theorem total_dolls_count : total_dolls = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_count_l3397_339768


namespace NUMINAMATH_CALUDE_smallest_positive_solution_tan_equation_l3397_339752

theorem smallest_positive_solution_tan_equation :
  let x : ℝ := π / 26
  (∀ y : ℝ, y > 0 ∧ y < x → ¬(Real.tan (4 * y) + Real.tan (3 * y) = 1 / Real.cos (3 * y))) ∧
  (Real.tan (4 * x) + Real.tan (3 * x) = 1 / Real.cos (3 * x)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_tan_equation_l3397_339752


namespace NUMINAMATH_CALUDE_greatest_integer_radius_for_circle_l3397_339704

theorem greatest_integer_radius_for_circle (r : ℕ) : r * r ≤ 49 → r ≤ 7 ∧ ∃ (s : ℕ), s = 7 ∧ s * s ≤ 49 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_for_circle_l3397_339704


namespace NUMINAMATH_CALUDE_remainder_481207_div_8_l3397_339705

theorem remainder_481207_div_8 :
  ∃ q : ℕ, 481207 = 8 * q + 7 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_481207_div_8_l3397_339705


namespace NUMINAMATH_CALUDE_set_operations_l3397_339739

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem set_operations :
  (A ∩ B = {1, 2, 3}) ∧
  (A ∩ C = {3, 4, 5, 6}) ∧
  (A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6}) ∧
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8}) := by
sorry

end NUMINAMATH_CALUDE_set_operations_l3397_339739


namespace NUMINAMATH_CALUDE_minimum_jars_needed_spice_jar_problem_l3397_339781

theorem minimum_jars_needed 
  (medium_jar_capacity : ℕ) 
  (large_container_capacity : ℕ) 
  (potential_loss : ℕ) : ℕ :=
  let min_jars := (large_container_capacity + medium_jar_capacity - 1) / medium_jar_capacity
  min_jars + potential_loss

theorem spice_jar_problem : 
  minimum_jars_needed 50 825 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_minimum_jars_needed_spice_jar_problem_l3397_339781


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l3397_339717

/-- A geometric sequence with first term a and common ratio r -/
def geometric_sequence (a : ℝ) (r : ℝ) : ℕ → ℝ :=
  λ n => a * r^(n - 1)

theorem tenth_term_of_specific_geometric_sequence :
  let a := 10
  let second_term := -30
  let r := second_term / a
  let seq := geometric_sequence a r
  seq 10 = -196830 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_geometric_sequence_l3397_339717


namespace NUMINAMATH_CALUDE_points_form_parabola_l3397_339785

-- Define the set of points (x, y) parametrized by t
def S : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p.1 = Real.cos t ^ 2 ∧ p.2 = Real.sin t * Real.cos t}

-- Define the parabola
def P : Set (ℝ × ℝ) := {p | p.2 ^ 2 = p.1 * (1 - p.1)}

-- Theorem stating that S is equal to P
theorem points_form_parabola : S = P := by sorry

end NUMINAMATH_CALUDE_points_form_parabola_l3397_339785


namespace NUMINAMATH_CALUDE_modular_inverse_32_mod_37_l3397_339798

theorem modular_inverse_32_mod_37 :
  ∃ x : ℕ, x ≤ 36 ∧ (32 * x) % 37 = 1 :=
by
  use 15
  sorry

end NUMINAMATH_CALUDE_modular_inverse_32_mod_37_l3397_339798


namespace NUMINAMATH_CALUDE_product_first_fifth_l3397_339730

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  third_term : a 3 = 3
  sum_reciprocals : 1 / a 1 + 1 / a 5 = 6 / 5

/-- The product of the first and fifth terms of the arithmetic sequence is 5 -/
theorem product_first_fifth (seq : ArithmeticSequence) : seq.a 1 * seq.a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_first_fifth_l3397_339730


namespace NUMINAMATH_CALUDE_one_third_blue_faces_iff_three_l3397_339795

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- The number of blue faces after cutting the cube into unit cubes -/
def blue_faces (c : Cube n) : ℕ :=
  6 * n^2

/-- The total number of faces of all unit cubes -/
def total_faces (c : Cube n) : ℕ :=
  6 * n^3

/-- Theorem stating that exactly one-third of the faces are blue iff n = 3 -/
theorem one_third_blue_faces_iff_three {n : ℕ} (c : Cube n) :
  3 * blue_faces c = total_faces c ↔ n = 3 :=
sorry

end NUMINAMATH_CALUDE_one_third_blue_faces_iff_three_l3397_339795


namespace NUMINAMATH_CALUDE_apple_pie_count_l3397_339796

/-- Given a box of apples weighing 120 pounds, using half for applesauce and the rest for pies,
    with each pie requiring 4 pounds of apples, prove that 15 pies can be made. -/
theorem apple_pie_count (total_weight : ℕ) (pie_weight : ℕ) : 
  total_weight = 120 →
  pie_weight = 4 →
  (total_weight / 2) / pie_weight = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_pie_count_l3397_339796


namespace NUMINAMATH_CALUDE_systematic_sampling_fifth_segment_l3397_339763

theorem systematic_sampling_fifth_segment 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (second_segment_student : ℕ) 
  (h1 : total_students = 700) 
  (h2 : selected_students = 50) 
  (h3 : second_segment_student = 20) :
  let interval := total_students / selected_students
  let first_student := second_segment_student - interval
  let fifth_segment_student := first_student + 4 * interval
  fifth_segment_student = 62 := by
sorry


end NUMINAMATH_CALUDE_systematic_sampling_fifth_segment_l3397_339763


namespace NUMINAMATH_CALUDE_correct_matching_probability_l3397_339746

def num_celebrities : ℕ := 3
def num_baby_photos : ℕ := 3

theorem correct_matching_probability :
  let total_arrangements := Nat.factorial num_celebrities
  let correct_arrangements := 1
  (correct_arrangements : ℚ) / total_arrangements = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l3397_339746


namespace NUMINAMATH_CALUDE_point_P_location_l3397_339708

-- Define the points on a line
structure Point :=
  (x : ℝ)

-- Define the distances
def OA (a : ℝ) : ℝ := a
def OB (b : ℝ) : ℝ := b
def OC (c : ℝ) : ℝ := c
def OE (e : ℝ) : ℝ := e

-- Define the condition for P being between B and C
def between (B C P : Point) : Prop :=
  B.x ≤ P.x ∧ P.x ≤ C.x

-- Define the ratio condition
def ratio_condition (A B C E P : Point) : Prop :=
  (A.x - P.x) * (P.x - C.x) = (B.x - P.x) * (P.x - E.x)

-- Theorem statement
theorem point_P_location 
  (O A B C E P : Point) 
  (a b c e : ℝ) 
  (h1 : O.x = 0) 
  (h2 : A.x = a) 
  (h3 : B.x = b) 
  (h4 : C.x = c) 
  (h5 : E.x = e) 
  (h6 : between B C P) 
  (h7 : ratio_condition A B C E P) : 
  P.x = (b * e - a * c) / (a - b + e - c) :=
sorry

end NUMINAMATH_CALUDE_point_P_location_l3397_339708


namespace NUMINAMATH_CALUDE_decagon_diagonals_l3397_339778

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular decagon has 35 diagonals -/
theorem decagon_diagonals : num_diagonals 10 = 35 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l3397_339778


namespace NUMINAMATH_CALUDE_disjoint_sets_imply_a_values_l3397_339715

-- Define the sets A and B
def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = a + 1}

def B (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (a^2 - 1) * p.1 + (a - 1) * p.2 = 15}

-- State the theorem
theorem disjoint_sets_imply_a_values (a : ℝ) :
  A a ∩ B a = ∅ → a = 1 ∨ a = -1 ∨ a = 5/2 ∨ a = -4 :=
by sorry

end NUMINAMATH_CALUDE_disjoint_sets_imply_a_values_l3397_339715


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3397_339760

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∀ z : ℂ, (1 - i) * z = 1 + i → z = i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3397_339760


namespace NUMINAMATH_CALUDE_basketball_team_min_score_l3397_339764

theorem basketball_team_min_score (n : ℕ) (min_score max_score : ℕ) 
  (h1 : n = 12) 
  (h2 : min_score = 7) 
  (h3 : max_score = 23) 
  (h4 : ∀ player_score, min_score ≤ player_score ∧ player_score ≤ max_score) : 
  n * min_score + (max_score - min_score) = 100 := by
sorry

end NUMINAMATH_CALUDE_basketball_team_min_score_l3397_339764


namespace NUMINAMATH_CALUDE_refrigerator_discount_proof_l3397_339722

/-- The original price of the refrigerator -/
def original_price : ℝ := 250.00

/-- The first discount rate -/
def first_discount : ℝ := 0.20

/-- The second discount rate -/
def second_discount : ℝ := 0.15

/-- The final price as a percentage of the original price -/
def final_percentage : ℝ := 0.68

theorem refrigerator_discount_proof :
  original_price * (1 - first_discount) * (1 - second_discount) = original_price * final_percentage :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_discount_proof_l3397_339722


namespace NUMINAMATH_CALUDE_not_monotonic_iff_a_in_range_l3397_339766

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2

theorem not_monotonic_iff_a_in_range (a : ℝ) :
  (∃ x y, 2 ≤ x ∧ x < y ∧ y ≤ 4 ∧ (f a x < f a y ∧ f a y < f a x)) ↔ 3 < a ∧ a < 6 := by
  sorry

end NUMINAMATH_CALUDE_not_monotonic_iff_a_in_range_l3397_339766


namespace NUMINAMATH_CALUDE_vector_angle_theorem_l3397_339710

/-- Given two vectors in 2D space, if the angle between them is 5π/6 and the magnitude of one vector
    equals the magnitude of their sum, then the angle between that vector and their sum is 2π/3. -/
theorem vector_angle_theorem (a b : ℝ × ℝ) :
  let angle_between := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  let magnitude (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
  angle_between = 5 * Real.pi / 6 ∧ magnitude a = magnitude (a.1 + b.1, a.2 + b.2) →
  Real.arccos ((a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2)) /
    (magnitude a * magnitude (a.1 + b.1, a.2 + b.2))) = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_theorem_l3397_339710


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3397_339714

theorem complex_equation_sum (x y : ℝ) : 
  (x : ℂ) + (y - 2) * Complex.I = 2 / (1 + Complex.I) → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3397_339714


namespace NUMINAMATH_CALUDE_lcm_gcd_relation_l3397_339742

theorem lcm_gcd_relation (a b : ℕ) : 
  (Nat.lcm a b = Nat.gcd a b + 19) ↔ 
  ((a = 1 ∧ b = 20) ∨ (a = 20 ∧ b = 1) ∨ 
   (a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4) ∨ 
   (a = 19 ∧ b = 38) ∨ (a = 38 ∧ b = 19)) :=
by sorry

end NUMINAMATH_CALUDE_lcm_gcd_relation_l3397_339742


namespace NUMINAMATH_CALUDE_game_ends_in_53_rounds_l3397_339702

/-- Represents the state of the game at any given round -/
structure GameState :=
  (A B C D : ℕ)

/-- The initial state of the game -/
def initial_state : GameState :=
  ⟨16, 15, 14, 13⟩

/-- Function to update the game state after one round -/
def update_state (state : GameState) : GameState :=
  sorry

/-- Predicate to check if the game has ended -/
def game_ended (state : GameState) : Prop :=
  state.A = 0 ∨ state.B = 0 ∨ state.C = 0 ∨ state.D = 0

/-- The number of rounds the game lasts -/
def game_duration : ℕ := 53

theorem game_ends_in_53_rounds :
  ∃ (final_state : GameState),
    (game_duration.iterate update_state initial_state = final_state) ∧
    game_ended final_state ∧
    ∀ (n : ℕ), n < game_duration →
      ¬game_ended (n.iterate update_state initial_state) :=
  sorry

end NUMINAMATH_CALUDE_game_ends_in_53_rounds_l3397_339702


namespace NUMINAMATH_CALUDE_polynomial_value_l3397_339792

theorem polynomial_value (a : ℝ) (h : a^2 + 3*a = 2) : 2*a^2 + 6*a - 10 = -6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l3397_339792


namespace NUMINAMATH_CALUDE_stratified_sample_intermediate_count_l3397_339750

/-- Represents the composition of teachers in a school -/
structure TeacherPopulation where
  total : Nat
  intermediate : Nat
  
/-- Represents a stratified sample of teachers -/
structure StratifiedSample where
  sampleSize : Nat
  intermediateSample : Nat

/-- Calculates the expected number of teachers with intermediate titles in a stratified sample -/
def expectedIntermediateSample (pop : TeacherPopulation) (sample : StratifiedSample) : Rat :=
  (pop.intermediate : Rat) * sample.sampleSize / pop.total

/-- Theorem stating that the number of teachers with intermediate titles in the sample is 7 -/
theorem stratified_sample_intermediate_count 
  (pop : TeacherPopulation) 
  (sample : StratifiedSample) : 
  pop.total = 160 → 
  pop.intermediate = 56 → 
  sample.sampleSize = 20 → 
  expectedIntermediateSample pop sample = 7 := by
  sorry

#check stratified_sample_intermediate_count

end NUMINAMATH_CALUDE_stratified_sample_intermediate_count_l3397_339750


namespace NUMINAMATH_CALUDE_soap_amount_is_fifteen_l3397_339799

/-- Represents the recipe for bubble mix -/
structure BubbleMixRecipe where
  soap_per_cup : ℚ  -- tablespoons of soap per cup of water
  ounces_per_cup : ℚ  -- ounces in a cup of water

/-- Represents a container for bubble mix -/
structure BubbleMixContainer where
  capacity : ℚ  -- capacity in ounces

/-- Calculates the amount of soap needed for a given container and recipe -/
def soap_needed (recipe : BubbleMixRecipe) (container : BubbleMixContainer) : ℚ :=
  (container.capacity / recipe.ounces_per_cup) * recipe.soap_per_cup

/-- Theorem: The amount of soap needed for the given recipe and container is 15 tablespoons -/
theorem soap_amount_is_fifteen (recipe : BubbleMixRecipe) (container : BubbleMixContainer) 
    (h1 : recipe.soap_per_cup = 3)
    (h2 : recipe.ounces_per_cup = 8)
    (h3 : container.capacity = 40) :
    soap_needed recipe container = 15 := by
  sorry

end NUMINAMATH_CALUDE_soap_amount_is_fifteen_l3397_339799


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_with_quarter_circles_l3397_339780

/-- The area of the shaded region in a rectangle with quarter circles in each corner -/
theorem shaded_area_rectangle_with_quarter_circles
  (length : ℝ) (width : ℝ) (radius : ℝ)
  (h_length : length = 12)
  (h_width : width = 8)
  (h_radius : radius = 4) :
  length * width - π * radius^2 = 96 - 16 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_with_quarter_circles_l3397_339780


namespace NUMINAMATH_CALUDE_fish_per_person_l3397_339723

/-- Represents the number of fish eyes Oomyapeck eats in a day -/
def eyes_eaten : ℕ := 22

/-- Represents the number of fish eyes Oomyapeck gives to his dog -/
def eyes_to_dog : ℕ := 2

/-- Represents the number of eyes each fish has -/
def eyes_per_fish : ℕ := 2

/-- Represents the number of family members -/
def family_members : ℕ := 3

theorem fish_per_person (eyes_eaten : ℕ) (eyes_to_dog : ℕ) (eyes_per_fish : ℕ) (family_members : ℕ) :
  eyes_eaten = 22 →
  eyes_to_dog = 2 →
  eyes_per_fish = 2 →
  family_members = 3 →
  (eyes_eaten - eyes_to_dog) / eyes_per_fish = 10 :=
by sorry

end NUMINAMATH_CALUDE_fish_per_person_l3397_339723


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l3397_339747

theorem roots_quadratic_equation (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) →
  q = -2*p →
  p = 1 ∧ q = -2 := by
sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l3397_339747


namespace NUMINAMATH_CALUDE_longest_tape_measure_l3397_339716

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 600) (hb : b = 500) (hc : c = 1200) : 
  Nat.gcd a (Nat.gcd b c) = 100 := by
  sorry

end NUMINAMATH_CALUDE_longest_tape_measure_l3397_339716


namespace NUMINAMATH_CALUDE_fraction_subtraction_equality_l3397_339703

theorem fraction_subtraction_equality : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_equality_l3397_339703


namespace NUMINAMATH_CALUDE_condition_property_l3397_339738

theorem condition_property (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a + a^2 > b + b^2) ∧
  (∃ a b, a + a^2 > b + b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_property_l3397_339738


namespace NUMINAMATH_CALUDE_stock_value_change_l3397_339709

theorem stock_value_change (initial_value : ℝ) (h : initial_value > 0) :
  let day1_value := initial_value * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.40)
  day2_value = initial_value * 1.05 := by
    sorry

end NUMINAMATH_CALUDE_stock_value_change_l3397_339709


namespace NUMINAMATH_CALUDE_parabola_equation_l3397_339743

/-- The standard equation of a parabola with focus (3, 0) and vertex (0, 0) is y² = 12x -/
theorem parabola_equation (x y : ℝ) :
  let focus : ℝ × ℝ := (3, 0)
  let vertex : ℝ × ℝ := (0, 0)
  (x - vertex.1) ^ 2 + (y - vertex.2) ^ 2 = (x - focus.1) ^ 2 + (y - focus.2) ^ 2 →
  y ^ 2 = 12 * x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3397_339743


namespace NUMINAMATH_CALUDE_min_value_expression_l3397_339784

theorem min_value_expression (x y : ℝ) 
  (h : 4 - 16 * x^2 - 8 * x * y - y^2 > 0) : 
  (13 * x^2 + 24 * x * y + 13 * y^2 - 14 * x - 16 * y + 61) / 
  (4 - 16 * x^2 - 8 * x * y - y^2)^(7/2) ≥ 7/16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3397_339784


namespace NUMINAMATH_CALUDE_cone_sin_theta_l3397_339707

/-- Theorem: For a cone with base radius 5 and lateral area 65π, 
    if θ is the angle between the slant height and the height of the cone, 
    then sinθ = 5/13 -/
theorem cone_sin_theta (r : ℝ) (lat_area : ℝ) (θ : ℝ) 
    (h1 : r = 5) 
    (h2 : lat_area = 65 * Real.pi) 
    (h3 : θ = Real.arcsin (r / (lat_area / (2 * Real.pi * r)))) : 
  Real.sin θ = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_cone_sin_theta_l3397_339707


namespace NUMINAMATH_CALUDE_store_purchase_combinations_l3397_339700

theorem store_purchase_combinations (headphones : ℕ) (mice : ℕ) (keyboards : ℕ) 
  (keyboard_mouse_sets : ℕ) (headphones_mouse_sets : ℕ) : 
  headphones = 9 → mice = 13 → keyboards = 5 → 
  keyboard_mouse_sets = 4 → headphones_mouse_sets = 5 → 
  keyboard_mouse_sets * headphones + 
  headphones_mouse_sets * keyboards + 
  headphones * mice * keyboards = 646 := by
  sorry

end NUMINAMATH_CALUDE_store_purchase_combinations_l3397_339700


namespace NUMINAMATH_CALUDE_lucy_had_twenty_l3397_339771

/-- The amount of money Lucy originally had -/
def lucy_original : ℕ := sorry

/-- The amount of money Linda originally had -/
def linda_original : ℕ := 10

/-- Proposition that if Lucy gives Linda $5, they would have the same amount of money -/
def equal_after_transfer : Prop :=
  lucy_original - 5 = linda_original + 5

theorem lucy_had_twenty :
  lucy_original = 20 :=
by sorry

end NUMINAMATH_CALUDE_lucy_had_twenty_l3397_339771


namespace NUMINAMATH_CALUDE_jerry_speed_is_30_l3397_339737

/-- Jerry's average speed in miles per hour -/
def jerry_speed : ℝ := 30

/-- Carla's average speed in miles per hour -/
def carla_speed : ℝ := 35

/-- Time difference between Jerry and Carla's departure in hours -/
def time_difference : ℝ := 0.5

/-- Time it takes Carla to catch up to Jerry in hours -/
def catch_up_time : ℝ := 3

/-- Theorem stating that Jerry's speed is 30 miles per hour -/
theorem jerry_speed_is_30 :
  jerry_speed = 30 ∧
  carla_speed * catch_up_time = jerry_speed * (catch_up_time + time_difference) :=
by sorry

end NUMINAMATH_CALUDE_jerry_speed_is_30_l3397_339737


namespace NUMINAMATH_CALUDE_expression_values_l3397_339731

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d > 0) :
  let expr := a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| + d / |d|
  expr = 5 ∨ expr = 1 ∨ expr = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l3397_339731


namespace NUMINAMATH_CALUDE_number_difference_l3397_339732

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 17402)
  (b_div_10 : 10 ∣ b)
  (a_eq_b_div_10 : a = b / 10) : 
  b - a = 14238 := by sorry

end NUMINAMATH_CALUDE_number_difference_l3397_339732


namespace NUMINAMATH_CALUDE_min_c_value_l3397_339783

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! (x y : ℝ), 2*x + y = 2033 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1017 :=
sorry

end NUMINAMATH_CALUDE_min_c_value_l3397_339783


namespace NUMINAMATH_CALUDE_decimal_comparisons_l3397_339774

theorem decimal_comparisons : 
  (0.839 < 0.9) ∧ (6.7 > 6.07) ∧ (5.45 = 5.450) := by
  sorry

end NUMINAMATH_CALUDE_decimal_comparisons_l3397_339774


namespace NUMINAMATH_CALUDE_f_positive_implies_a_bounded_l3397_339720

/-- The function f(x) defined as x^2 - ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 2

/-- Theorem stating that if f(x) > 0 for all x > 2, then a ≤ 3 -/
theorem f_positive_implies_a_bounded (a : ℝ) : 
  (∀ x > 2, f a x > 0) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_f_positive_implies_a_bounded_l3397_339720


namespace NUMINAMATH_CALUDE_max_volume_cross_section_area_l3397_339721

/-- Sphere with radius 2 -/
def Sphere : Type := Unit

/-- Points on the surface of the sphere -/
def Point : Type := Unit

/-- Angle between two points and the center of the sphere -/
def angle (p q : Point) : ℝ := sorry

/-- Volume of the triangular pyramid formed by three points and the center of the sphere -/
def pyramidVolume (a b c : Point) : ℝ := sorry

/-- Area of the circular cross-section formed by a plane through three points on the sphere -/
def crossSectionArea (a b c : Point) : ℝ := sorry

/-- The theorem statement -/
theorem max_volume_cross_section_area (o : Sphere) (a b c : Point) :
  (∀ (p q : Point), angle p q = angle a b) →
  (∀ (p q r : Point), pyramidVolume p q r ≤ pyramidVolume a b c) →
  crossSectionArea a b c = 8 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_max_volume_cross_section_area_l3397_339721


namespace NUMINAMATH_CALUDE_exponential_function_problem_l3397_339757

theorem exponential_function_problem (a : ℝ) (f : ℝ → ℝ) :
  a > 0 ∧ a ≠ 1 ∧ (∀ x, f x = a^x) ∧ f 3 = 8 →
  f (-1) = (1/2) := by
sorry

end NUMINAMATH_CALUDE_exponential_function_problem_l3397_339757


namespace NUMINAMATH_CALUDE_lizette_minerva_stamp_difference_l3397_339751

theorem lizette_minerva_stamp_difference :
  let lizette_stamps : ℕ := 813
  let minerva_stamps : ℕ := 688
  lizette_stamps > minerva_stamps →
  lizette_stamps - minerva_stamps = 125 := by
sorry

end NUMINAMATH_CALUDE_lizette_minerva_stamp_difference_l3397_339751


namespace NUMINAMATH_CALUDE_arrangement_exists_for_P_23_l3397_339753

/-- Fibonacci-like sequence defined by F_0 = 0, F_1 = 1, F_i = 3F_{i-1} - F_{i-2} for i ≥ 2 -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of an arrangement satisfying the given conditions for P = 23 -/
theorem arrangement_exists_for_P_23 : F 12 % 23 = 0 := by sorry

end NUMINAMATH_CALUDE_arrangement_exists_for_P_23_l3397_339753


namespace NUMINAMATH_CALUDE_cost_per_serving_soup_l3397_339748

/-- Calculates the cost per serving of soup given ingredient quantities and prices -/
theorem cost_per_serving_soup (beef_quantity beef_price chicken_quantity chicken_price
                               carrot_quantity carrot_price potato_quantity potato_price
                               onion_quantity onion_price servings : ℚ) :
  beef_quantity = 4 →
  beef_price = 6 →
  chicken_quantity = 3 →
  chicken_price = 4 →
  carrot_quantity = 2 →
  carrot_price = (3/2) →
  potato_quantity = 3 →
  potato_price = 2 →
  onion_quantity = 1 →
  onion_price = 3 →
  servings = 12 →
  (beef_quantity * beef_price +
   chicken_quantity * chicken_price +
   carrot_quantity * carrot_price +
   potato_quantity * potato_price +
   onion_quantity * onion_price) / servings = 4 :=
by sorry

end NUMINAMATH_CALUDE_cost_per_serving_soup_l3397_339748


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l3397_339782

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
def calculate_compensation (regular_rate : ℚ) (overtime_multiplier : ℚ) (total_hours : ℕ) (regular_hours : ℕ) : ℚ :=
  let overtime_rate := regular_rate * (1 + overtime_multiplier)
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * (total_hours - regular_hours)
  regular_pay + overtime_pay

/-- Theorem stating that the bus driver's compensation for 60 hours of work is $1200. -/
theorem bus_driver_compensation :
  calculate_compensation 16 0.75 60 40 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_bus_driver_compensation_l3397_339782


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3397_339759

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 9 ∧ x ≠ -7 →
  (5 * x - 3) / (x^2 - 2*x - 63) = (21/8) / (x - 9) + (19/8) / (x + 7) :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3397_339759


namespace NUMINAMATH_CALUDE_choose_officers_specific_club_l3397_339775

/-- Represents a club with boys and girls -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ

/-- Calculates the number of ways to choose officers in a club -/
def choose_officers (c : Club) : ℕ :=
  c.total_members * (c.boys - 1 + c.girls - 1) * (c.total_members - 2)

/-- Theorem: The number of ways to choose officers in a specific club configuration -/
theorem choose_officers_specific_club :
  let c : Club := { total_members := 30, boys := 15, girls := 15 }
  choose_officers c = 11760 := by
  sorry

#eval choose_officers { total_members := 30, boys := 15, girls := 15 }

end NUMINAMATH_CALUDE_choose_officers_specific_club_l3397_339775


namespace NUMINAMATH_CALUDE_ab_not_always_negative_l3397_339776

theorem ab_not_always_negative (a b : ℚ) 
  (h1 : (a - b)^2 + (b - a) * |a - b| = a * b) 
  (h2 : a * b ≠ 0) : 
  ¬(∀ a b : ℚ, (a - b)^2 + (b - a) * |a - b| = a * b → a * b < 0) := by
sorry

end NUMINAMATH_CALUDE_ab_not_always_negative_l3397_339776


namespace NUMINAMATH_CALUDE_thirty_minus_twelve_base5_l3397_339769

/-- Converts a natural number to its base 5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Theorem: 30 in base 10 minus 12 in base 10 equals 33 in base 5 --/
theorem thirty_minus_twelve_base5 : toBase5 (30 - 12) = [3, 3] := by
  sorry

end NUMINAMATH_CALUDE_thirty_minus_twelve_base5_l3397_339769


namespace NUMINAMATH_CALUDE_not_power_function_l3397_339724

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x^α

-- Define the specific function
def f (x : ℝ) : ℝ := 2 * x^(1/2)

-- Theorem statement
theorem not_power_function : ¬ isPowerFunction f := by
  sorry

end NUMINAMATH_CALUDE_not_power_function_l3397_339724


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l3397_339754

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l3397_339754


namespace NUMINAMATH_CALUDE_cat_ratio_l3397_339733

theorem cat_ratio (jacob_cats : ℕ) (melanie_cats : ℕ) :
  jacob_cats = 90 →
  melanie_cats = 60 →
  ∃ (annie_cats : ℕ),
    annie_cats = jacob_cats / 3 ∧
    melanie_cats = annie_cats ∧
    melanie_cats / annie_cats = 2 :=
by sorry

end NUMINAMATH_CALUDE_cat_ratio_l3397_339733
