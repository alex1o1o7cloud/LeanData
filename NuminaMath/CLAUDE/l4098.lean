import Mathlib

namespace NUMINAMATH_CALUDE_solve_for_y_l4098_409868

theorem solve_for_y (x y : ℝ) (h1 : x^(2*y) = 64) (h2 : x = 8) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l4098_409868


namespace NUMINAMATH_CALUDE_chocolate_manufacturer_cost_l4098_409858

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

end NUMINAMATH_CALUDE_chocolate_manufacturer_cost_l4098_409858


namespace NUMINAMATH_CALUDE_stratified_sampling_best_l4098_409851

/-- Represents different sampling methods -/
inductive SamplingMethod
| Lottery
| RandomNumberTable
| Systematic
| Stratified

/-- Represents product quality classes -/
inductive ProductClass
| FirstClass
| SecondClass
| Defective

/-- Represents a collection of products with their quantities -/
structure ProductCollection :=
  (total : ℕ)
  (firstClass : ℕ)
  (secondClass : ℕ)
  (defective : ℕ)

/-- Determines the most appropriate sampling method for quality analysis -/
def bestSamplingMethod (products : ProductCollection) (sampleSize : ℕ) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is the best method for the given conditions -/
theorem stratified_sampling_best :
  let products : ProductCollection := {
    total := 40,
    firstClass := 10,
    secondClass := 25,
    defective := 5
  }
  let sampleSize := 8
  bestSamplingMethod products sampleSize = SamplingMethod.Stratified :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_best_l4098_409851


namespace NUMINAMATH_CALUDE_even_pairs_ge_odd_pairs_l4098_409859

/-- A sequence of zeros and ones -/
def BinarySequence := List Bool

/-- Count the number of (1,0) pairs with even number of digits between them -/
def countEvenPairs (seq : BinarySequence) : ℕ := sorry

/-- Count the number of (1,0) pairs with odd number of digits between them -/
def countOddPairs (seq : BinarySequence) : ℕ := sorry

/-- Theorem: In any binary sequence, the number of (1,0) pairs with even number
    of digits between them is greater than or equal to the number of (1,0) pairs
    with odd number of digits between them -/
theorem even_pairs_ge_odd_pairs (seq : BinarySequence) :
  countEvenPairs seq ≥ countOddPairs seq := by sorry

end NUMINAMATH_CALUDE_even_pairs_ge_odd_pairs_l4098_409859


namespace NUMINAMATH_CALUDE_A_run_time_l4098_409808

/-- The time it takes for A to run 160 meters -/
def time_A : ℝ := 28

/-- The time it takes for B to run 160 meters -/
def time_B : ℝ := 32

/-- The distance A runs -/
def distance_A : ℝ := 160

/-- The distance B runs when A finishes -/
def distance_B : ℝ := 140

theorem A_run_time :
  (distance_A / time_A = distance_B / time_B) ∧
  (distance_A - distance_B = 20) →
  time_A = 28 := by sorry

end NUMINAMATH_CALUDE_A_run_time_l4098_409808


namespace NUMINAMATH_CALUDE_probability_good_not_less_than_defective_expected_value_defective_l4098_409829

/-- The total number of items -/
def total_items : ℕ := 7

/-- The number of good items -/
def good_items : ℕ := 4

/-- The number of defective items -/
def defective_items : ℕ := 3

/-- The number of items selected in the first scenario -/
def selected_items_1 : ℕ := 3

/-- The number of items selected in the second scenario -/
def selected_items_2 : ℕ := 5

/-- Probability of selecting at least as many good items as defective items -/
theorem probability_good_not_less_than_defective :
  (Nat.choose good_items 2 * Nat.choose defective_items 1 + Nat.choose good_items 3) / 
  Nat.choose total_items selected_items_1 = 22 / 35 := by sorry

/-- Expected value of defective items when selecting 5 out of 7 -/
theorem expected_value_defective :
  (1 * Nat.choose good_items 4 * Nat.choose defective_items 1 +
   2 * Nat.choose good_items 3 * Nat.choose defective_items 2 +
   3 * Nat.choose good_items 2 * Nat.choose defective_items 3) /
  Nat.choose total_items selected_items_2 = 15 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_good_not_less_than_defective_expected_value_defective_l4098_409829


namespace NUMINAMATH_CALUDE_sqrt_one_minus_sqrt_three_squared_l4098_409813

theorem sqrt_one_minus_sqrt_three_squared : 
  Real.sqrt ((1 - Real.sqrt 3) ^ 2) = Real.sqrt 3 - 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_one_minus_sqrt_three_squared_l4098_409813


namespace NUMINAMATH_CALUDE_carson_gold_stars_l4098_409891

/-- Represents the number of gold stars Carson earned today -/
def gold_stars_today (yesterday : ℕ) (total : ℕ) : ℕ :=
  total - yesterday

theorem carson_gold_stars : gold_stars_today 6 15 = 9 := by
  sorry

end NUMINAMATH_CALUDE_carson_gold_stars_l4098_409891


namespace NUMINAMATH_CALUDE_album_distribution_ways_l4098_409889

/-- The number of ways to distribute albums to friends -/
def distribute_albums (photo_albums : ℕ) (stamp_albums : ℕ) (friends : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of ways to distribute the albums -/
theorem album_distribution_ways :
  distribute_albums 2 3 4 = 10 := by sorry

end NUMINAMATH_CALUDE_album_distribution_ways_l4098_409889


namespace NUMINAMATH_CALUDE_remainder_problem_l4098_409827

theorem remainder_problem (N : ℤ) (h : ∃ k : ℤ, N = 39 * k + 18) : 
  ∃ m : ℤ, N = 13 * m + 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l4098_409827


namespace NUMINAMATH_CALUDE_root_intervals_l4098_409836

noncomputable def f (x : ℝ) : ℝ :=
  if x > -2 then Real.exp (x + 1) - 2
  else Real.exp (-x - 3) - 2

theorem root_intervals (e : ℝ) (h_e : e = Real.exp 1) :
  {k : ℤ | ∃ x : ℝ, f x = 0 ∧ k - 1 < x ∧ x < k} = {-4, 0} := by
  sorry

end NUMINAMATH_CALUDE_root_intervals_l4098_409836


namespace NUMINAMATH_CALUDE_inverse_proportional_solution_l4098_409849

theorem inverse_proportional_solution (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 30) (h3 : x - y = 6) :
  x = 6 → y = 36 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportional_solution_l4098_409849


namespace NUMINAMATH_CALUDE_average_scores_is_68_l4098_409879

def scores : List ℝ := [50, 60, 70, 80, 80]

theorem average_scores_is_68 : (scores.sum / scores.length) = 68 := by
  sorry

end NUMINAMATH_CALUDE_average_scores_is_68_l4098_409879


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l4098_409805

/-- The area of a right triangle with side lengths in the ratio 5:12:13, 
    inscribed in a circle of radius 5, is equal to 3000/169. -/
theorem triangle_area_in_circle (r : ℝ) (h : r = 5) : 
  let s := r * 2 / 13  -- Scale factor
  let a := 5 * s       -- First side
  let b := 12 * s      -- Second side
  let c := 13 * s      -- Third side (hypotenuse)
  (a^2 + b^2 = c^2) ∧  -- Pythagorean theorem
  (c = 2 * r) →        -- Diameter equals hypotenuse
  (1/2 * a * b = 3000/169) := by
sorry

end NUMINAMATH_CALUDE_triangle_area_in_circle_l4098_409805


namespace NUMINAMATH_CALUDE_smallest_number_in_specific_integer_set_l4098_409897

theorem smallest_number_in_specific_integer_set :
  ∀ (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    (a + b + c : ℚ) / 3 = 30 →
    b = 29 →
    max a (max b c) = b + 4 →
    min a (min b c) = 28 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_in_specific_integer_set_l4098_409897


namespace NUMINAMATH_CALUDE_factorization_identities_l4098_409821

theorem factorization_identities (a b x m : ℝ) :
  (2 * a^2 - 2*a*b = 2*a*(a-b)) ∧
  (2 * x^2 - 18 = 2*(x+3)*(x-3)) ∧
  (-3*m*a^3 + 6*m*a^2 - 3*m*a = -3*m*a*(a-1)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identities_l4098_409821


namespace NUMINAMATH_CALUDE_container_volume_ratio_l4098_409850

theorem container_volume_ratio :
  ∀ (A B : ℚ),
  A > 0 → B > 0 →
  (3/4 : ℚ) * A + (1/4 : ℚ) * B = (7/8 : ℚ) * B →
  A / B = (5/6 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l4098_409850


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l4098_409872

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! x : ℤ, (x + 2) / (4 - x) < 0 ∧ 2*x^2 + (2*a + 7)*x + 7*a < 0) →
  ((-5 ≤ a ∧ a < 3) ∨ (4 < a ∧ a ≤ 5)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l4098_409872


namespace NUMINAMATH_CALUDE_total_carrot_sticks_l4098_409820

def before_dinner : ℕ := 22
def after_dinner : ℕ := 15

theorem total_carrot_sticks : before_dinner + after_dinner = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_carrot_sticks_l4098_409820


namespace NUMINAMATH_CALUDE_parabola_focus_on_line_l4098_409890

/-- The value of p for a parabola y^2 = 2px whose focus lies on 2x + y - 2 = 0 -/
theorem parabola_focus_on_line : ∃ (p : ℝ), 
  p > 0 ∧ 
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ 2*x + y - 2 = 0 ∧ x = p/2 ∧ y = 0) →
  p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_on_line_l4098_409890


namespace NUMINAMATH_CALUDE_tetrahedron_vertex_equality_l4098_409861

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

end NUMINAMATH_CALUDE_tetrahedron_vertex_equality_l4098_409861


namespace NUMINAMATH_CALUDE_centroid_maximizes_min_area_l4098_409860

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

end NUMINAMATH_CALUDE_centroid_maximizes_min_area_l4098_409860


namespace NUMINAMATH_CALUDE_complex_equation_sum_l4098_409865

theorem complex_equation_sum (x y : ℝ) :
  (x + (y - 2) * Complex.I = 2 / (1 + Complex.I)) → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l4098_409865


namespace NUMINAMATH_CALUDE_dog_cost_l4098_409878

/-- The cost of a dog given the current money and additional money needed -/
theorem dog_cost (current_money additional_money : ℕ) :
  current_money = 34 →
  additional_money = 13 →
  current_money + additional_money = 47 :=
by sorry

end NUMINAMATH_CALUDE_dog_cost_l4098_409878


namespace NUMINAMATH_CALUDE_congruence_modulo_ten_l4098_409864

def a : ℤ := 1 + (Finset.sum (Finset.range 20) (fun k => Nat.choose 20 (k + 1) * 2^k))

theorem congruence_modulo_ten (b : ℤ) (h : b ≡ a [ZMOD 10]) : b = 2011 := by
  sorry

end NUMINAMATH_CALUDE_congruence_modulo_ten_l4098_409864


namespace NUMINAMATH_CALUDE_range_of_a_l4098_409824

-- Define the two curves
def curve1 (x y a : ℝ) : Prop := x^2 + 4*(y - a)^2 = 4
def curve2 (x y : ℝ) : Prop := x^2 = 4*y

-- Define the intersection of the curves
def curves_intersect (a : ℝ) : Prop :=
  ∃ x y, curve1 x y a ∧ curve2 x y

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, curves_intersect a ↔ a ∈ Set.Icc (-1) (5/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4098_409824


namespace NUMINAMATH_CALUDE_mouse_jump_distance_l4098_409892

theorem mouse_jump_distance (grasshopper_jump frog_jump mouse_jump : ℕ) :
  grasshopper_jump = 25 →
  frog_jump = grasshopper_jump + 32 →
  mouse_jump = frog_jump - 26 →
  mouse_jump = 31 := by sorry

end NUMINAMATH_CALUDE_mouse_jump_distance_l4098_409892


namespace NUMINAMATH_CALUDE_queen_placement_probability_l4098_409809

/-- The number of squares on a chessboard -/
def chessboardSize : ℕ := 64

/-- The number of trials in the experiment -/
def numberOfTrials : ℕ := 3

/-- The probability that two randomly placed queens can attack each other -/
def attackingProbability : ℚ := 13 / 36

/-- The probability of at least one non-attacking configuration in 3 trials -/
def nonAttackingProbability : ℚ := 1 - attackingProbability ^ numberOfTrials

theorem queen_placement_probability :
  nonAttackingProbability = 1 - (13 / 36) ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_queen_placement_probability_l4098_409809


namespace NUMINAMATH_CALUDE_abc_def_ratio_l4098_409830

theorem abc_def_ratio (a b c d e f : ℚ)
  (h1 : a / b = 5 / 2)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 1)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) :
  a * b * c / (d * e * f) = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_def_ratio_l4098_409830


namespace NUMINAMATH_CALUDE_basic_computer_price_l4098_409803

/-- The price of a basic computer and printer totaling $2,500, 
    where an enhanced computer costing $500 more would make the printer 1/8 of the new total. -/
theorem basic_computer_price : 
  ∀ (basic_price printer_price enhanced_price : ℝ),
  basic_price + printer_price = 2500 →
  enhanced_price = basic_price + 500 →
  printer_price = (1/8) * (enhanced_price + printer_price) →
  basic_price = 2125 := by
sorry

end NUMINAMATH_CALUDE_basic_computer_price_l4098_409803


namespace NUMINAMATH_CALUDE_difference_of_fractions_of_6000_l4098_409886

theorem difference_of_fractions_of_6000 : 
  (1 / 10 : ℚ) * 6000 - (1 / 1000 : ℚ) * 6000 = 594 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_fractions_of_6000_l4098_409886


namespace NUMINAMATH_CALUDE_black_lambs_count_all_lambs_accounted_l4098_409823

/-- The number of black lambs in Farmer Cunningham's flock -/
def black_lambs : ℕ := 5855

/-- The total number of lambs in Farmer Cunningham's flock -/
def total_lambs : ℕ := 6048

/-- The number of white lambs in Farmer Cunningham's flock -/
def white_lambs : ℕ := 193

/-- Theorem stating that the number of black lambs is correct -/
theorem black_lambs_count : black_lambs = total_lambs - white_lambs := by
  sorry

/-- Theorem stating that all lambs are accounted for -/
theorem all_lambs_accounted : total_lambs = black_lambs + white_lambs := by
  sorry

end NUMINAMATH_CALUDE_black_lambs_count_all_lambs_accounted_l4098_409823


namespace NUMINAMATH_CALUDE_correct_propositions_l4098_409846

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

end NUMINAMATH_CALUDE_correct_propositions_l4098_409846


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l4098_409866

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 4*x + k = 0 ∧ y^2 + 4*y + k = 0) → k ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l4098_409866


namespace NUMINAMATH_CALUDE_book_arrangement_count_l4098_409875

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

end NUMINAMATH_CALUDE_book_arrangement_count_l4098_409875


namespace NUMINAMATH_CALUDE_area_ratio_hexagon_triangle_l4098_409867

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

end NUMINAMATH_CALUDE_area_ratio_hexagon_triangle_l4098_409867


namespace NUMINAMATH_CALUDE_equation_solution_l4098_409874

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 3

-- Theorem statement
theorem equation_solution :
  ∃ x : ℝ, 2 * (f x) - 11 = f (x - 2) ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4098_409874


namespace NUMINAMATH_CALUDE_complex_bound_l4098_409885

theorem complex_bound (z : ℂ) (h : Complex.abs (z + z⁻¹) = 1) :
  (Real.sqrt 5 - 1) / 2 ≤ Complex.abs z ∧ Complex.abs z ≤ (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_bound_l4098_409885


namespace NUMINAMATH_CALUDE_trig_sum_equals_one_l4098_409898

theorem trig_sum_equals_one : 
  Real.sin (300 * Real.pi / 180) + Real.cos (390 * Real.pi / 180) + Real.tan (-135 * Real.pi / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equals_one_l4098_409898


namespace NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l4098_409802

-- Part 1: Non-existence of an infinite sequence of positive integers
theorem no_positive_integer_sequence :
  ¬ (∃ (a : ℕ → ℕ+), ∀ (n : ℕ), (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) :=
sorry

-- Part 2: Existence of an infinite sequence of positive irrational numbers
theorem exists_positive_irrational_sequence :
  ∃ (a : ℕ → ℝ), (∀ (n : ℕ), Irrational (a n) ∧ a n > 0) ∧
    (∀ (n : ℕ), (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) :=
sorry

end NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l4098_409802


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l4098_409807

theorem quadratic_always_positive (m : ℝ) :
  (∀ x : ℝ, (4 - m) * x^2 - 3 * x + (4 + m) > 0) ↔ 
  (-Real.sqrt 55 / 2 < m ∧ m < Real.sqrt 55 / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l4098_409807


namespace NUMINAMATH_CALUDE_divisible_by_64_l4098_409871

theorem divisible_by_64 (n : ℕ) (h : n > 0) : ∃ k : ℤ, 3^(2*n + 2) - 8*n - 9 = 64*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_64_l4098_409871


namespace NUMINAMATH_CALUDE_choose_leaders_count_l4098_409882

/-- Represents the number of members in each category -/
structure ClubMembers where
  senior_boys : Nat
  junior_boys : Nat
  senior_girls : Nat
  junior_girls : Nat

/-- Calculates the number of ways to choose a president and vice-president -/
def choose_leaders (members : ClubMembers) : Nat :=
  let boys_combinations := members.senior_boys * members.junior_boys * 2
  let girls_combinations := members.senior_girls * members.junior_girls * 2
  boys_combinations + girls_combinations

/-- Theorem stating the number of ways to choose leaders under given conditions -/
theorem choose_leaders_count (members : ClubMembers) 
  (h1 : members.senior_boys = 6)
  (h2 : members.junior_boys = 6)
  (h3 : members.senior_girls = 6)
  (h4 : members.junior_girls = 6) :
  choose_leaders members = 144 := by
  sorry

#eval choose_leaders ⟨6, 6, 6, 6⟩

end NUMINAMATH_CALUDE_choose_leaders_count_l4098_409882


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l4098_409862

theorem simplify_nested_roots (x : ℝ) :
  (((x ^ 16) ^ (1 / 8)) ^ (1 / 4)) ^ 2 + (((x ^ 16) ^ (1 / 4)) ^ (1 / 8)) ^ 2 = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l4098_409862


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomials_l4098_409838

def poly1 (x : ℤ) : ℤ := 2 * x^3 - 3 * x^2 - 11 * x + 6
def poly2 (x : ℤ) : ℤ := x^4 + 4 * x^3 - 9 * x^2 - 16 * x + 20

theorem integer_roots_of_polynomials :
  (∀ x : ℤ, poly1 x = 0 ↔ x = -2 ∨ x = 3) ∧
  (∀ x : ℤ, poly2 x = 0 ↔ x = 1 ∨ x = 2 ∨ x = -2 ∨ x = -5) :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomials_l4098_409838


namespace NUMINAMATH_CALUDE_complex_function_chain_l4098_409856

theorem complex_function_chain (x y u : ℂ) : 
  u = 2 * x - 5 → (y = (2 * x - 5)^10 ↔ y = u^10) := by
  sorry

end NUMINAMATH_CALUDE_complex_function_chain_l4098_409856


namespace NUMINAMATH_CALUDE_smallest_quadratic_root_l4098_409877

theorem smallest_quadratic_root : 
  let f : ℝ → ℝ := λ y => 4 * y^2 - 7 * y + 3
  ∃ y : ℝ, f y = 0 ∧ ∀ z : ℝ, f z = 0 → y ≤ z ∧ y = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_quadratic_root_l4098_409877


namespace NUMINAMATH_CALUDE_circle_radius_reduction_l4098_409899

theorem circle_radius_reduction (r : ℝ) (h : r > 0) :
  let new_area_ratio := 1 - 0.18999999999999993
  let new_radius_ratio := 1 - 0.1
  (new_radius_ratio * r) ^ 2 = new_area_ratio * r ^ 2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_reduction_l4098_409899


namespace NUMINAMATH_CALUDE_counterexample_exists_l4098_409883

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l4098_409883


namespace NUMINAMATH_CALUDE_hotel_rooms_l4098_409826

theorem hotel_rooms (total_rooms : ℕ) (single_cost double_cost total_revenue : ℚ) 
  (h1 : total_rooms = 260)
  (h2 : single_cost = 35)
  (h3 : double_cost = 60)
  (h4 : total_revenue = 14000) :
  ∃ (single_rooms double_rooms : ℕ),
    single_rooms + double_rooms = total_rooms ∧
    single_cost * single_rooms + double_cost * double_rooms = total_revenue ∧
    double_rooms = 196 :=
by sorry

end NUMINAMATH_CALUDE_hotel_rooms_l4098_409826


namespace NUMINAMATH_CALUDE_juan_lunch_time_l4098_409840

/-- The number of pages in Juan's book -/
def book_pages : ℕ := 4000

/-- The number of pages Juan reads per hour -/
def pages_per_hour : ℕ := 250

/-- The time it takes Juan to read the entire book, in hours -/
def reading_time : ℚ := book_pages / pages_per_hour

/-- The time it takes Juan to grab lunch from his office and back, in hours -/
def lunch_time : ℚ := reading_time / 2

theorem juan_lunch_time : lunch_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_juan_lunch_time_l4098_409840


namespace NUMINAMATH_CALUDE_sad_girls_l4098_409804

/-- Given information about children's emotions and genders -/
structure ChildrenInfo where
  total : ℕ
  happy : ℕ
  sad : ℕ
  neither : ℕ
  boys : ℕ
  girls : ℕ
  happyBoys : ℕ
  neitherBoys : ℕ

/-- Theorem stating the number of sad girls -/
theorem sad_girls (info : ChildrenInfo)
  (h1 : info.total = 60)
  (h2 : info.happy = 30)
  (h3 : info.sad = 10)
  (h4 : info.neither = 20)
  (h5 : info.boys = 17)
  (h6 : info.girls = 43)
  (h7 : info.happyBoys = 6)
  (h8 : info.neitherBoys = 5)
  (h9 : info.total = info.happy + info.sad + info.neither)
  (h10 : info.total = info.boys + info.girls)
  : info.sad - (info.boys - info.happyBoys - info.neitherBoys) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sad_girls_l4098_409804


namespace NUMINAMATH_CALUDE_proposition_falsity_l4098_409847

theorem proposition_falsity (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 5) : 
  ¬ P 4 := by
sorry

end NUMINAMATH_CALUDE_proposition_falsity_l4098_409847


namespace NUMINAMATH_CALUDE_quotient_of_arctangents_eq_one_l4098_409814

theorem quotient_of_arctangents_eq_one :
  (π - Real.arctan (8/15)) / (2 * Real.arctan 4) = 1 := by sorry

end NUMINAMATH_CALUDE_quotient_of_arctangents_eq_one_l4098_409814


namespace NUMINAMATH_CALUDE_seating_arrangements_l4098_409870

def number_of_people : ℕ := 10
def table_seats : ℕ := 8

def alice_bob_block : ℕ := 1
def other_individuals : ℕ := table_seats - 2

def ways_to_choose : ℕ := Nat.choose number_of_people table_seats
def ways_to_arrange_units : ℕ := Nat.factorial (other_individuals + alice_bob_block - 1)
def ways_to_arrange_alice_bob : ℕ := 2

theorem seating_arrangements :
  ways_to_choose * ways_to_arrange_units * ways_to_arrange_alice_bob = 64800 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangements_l4098_409870


namespace NUMINAMATH_CALUDE_ten_steps_climb_l4098_409837

/-- Number of ways to climb n steps when allowed to take 1, 2, or 3 steps at a time -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | k + 3 => climbStairs (k + 2) + climbStairs (k + 1) + climbStairs k

/-- Theorem stating that there are 274 ways to climb 10 steps -/
theorem ten_steps_climb : climbStairs 10 = 274 := by
  sorry


end NUMINAMATH_CALUDE_ten_steps_climb_l4098_409837


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l4098_409852

theorem absolute_value_inequality (x : ℝ) : 
  |x^2 - 3*x| > 4 ↔ x < -1 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l4098_409852


namespace NUMINAMATH_CALUDE_sum_of_roots_l4098_409848

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l4098_409848


namespace NUMINAMATH_CALUDE_power_of_one_sixth_l4098_409831

def is_greatest_power_of_2_dividing_180 (x : ℕ) : Prop :=
  2^x ∣ 180 ∧ ∀ k > x, ¬(2^k ∣ 180)

def is_greatest_power_of_3_dividing_180 (y : ℕ) : Prop :=
  3^y ∣ 180 ∧ ∀ k > y, ¬(3^k ∣ 180)

theorem power_of_one_sixth (x y : ℕ) 
  (h1 : is_greatest_power_of_2_dividing_180 x) 
  (h2 : is_greatest_power_of_3_dividing_180 y) : 
  (1/6 : ℚ)^(y - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_one_sixth_l4098_409831


namespace NUMINAMATH_CALUDE_max_xy_value_l4098_409810

theorem max_xy_value (x y : ℝ) (hx : x < 0) (hy : y < 0) (heq : 3*x + y = -2) :
  (∀ z : ℝ, z = x*y → z ≤ 1/3) ∧ ∃ z : ℝ, z = x*y ∧ z = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l4098_409810


namespace NUMINAMATH_CALUDE_second_scenario_pipes_l4098_409825

-- Define the capacity of a single pipe
def pipe_capacity : ℝ := 1

-- Define the total capacity of the tank
def tank_capacity : ℝ := 3 * pipe_capacity * 8

-- Define the time taken in the first scenario
def time1 : ℝ := 8

-- Define the time taken in the second scenario
def time2 : ℝ := 12

-- Define the number of pipes in the first scenario
def pipes1 : ℕ := 3

-- Theorem to prove
theorem second_scenario_pipes :
  ∃ (pipes2 : ℕ), 
    (pipes1 : ℝ) * pipe_capacity * time1 = (pipes2 : ℝ) * pipe_capacity * time2 ∧ 
    pipes2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_scenario_pipes_l4098_409825


namespace NUMINAMATH_CALUDE_eighteen_mangoes_yield_fortyeight_lassis_l4098_409812

/-- Given that 3 mangoes make 8 lassis, this function calculates
    the number of lassis that can be made from a given number of mangoes. -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  (mangoes * 8) / 3

/-- Theorem stating that 18 mangoes will yield 48 lassis, 
    given the ratio of 8 lassis to 3 mangoes. -/
theorem eighteen_mangoes_yield_fortyeight_lassis :
  lassis_from_mangoes 18 = 48 := by
  sorry

#eval lassis_from_mangoes 18

end NUMINAMATH_CALUDE_eighteen_mangoes_yield_fortyeight_lassis_l4098_409812


namespace NUMINAMATH_CALUDE_tangent_slope_implies_function_value_l4098_409839

open Real

theorem tangent_slope_implies_function_value (x₀ : ℝ) (h : x₀ > 0) : 
  let f : ℝ → ℝ := λ x ↦ log x + 2 * x
  (deriv f x₀ = 3) → f x₀ = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_function_value_l4098_409839


namespace NUMINAMATH_CALUDE_fraction_division_l4098_409869

theorem fraction_division (x : ℝ) (hx : x ≠ 0) :
  (3 / 8) / (5 * x / 12) = 9 / (10 * x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l4098_409869


namespace NUMINAMATH_CALUDE_pentagonal_field_fencing_cost_l4098_409816

/-- Represents the cost of fencing an irregular pentagonal field --/
def fencing_cost (side_a side_b side_c side_d side_e : ℝ) (cost_per_meter : ℝ) : ℝ :=
  (side_a + side_b + side_c + side_d + side_e) * cost_per_meter

/-- Theorem stating the total cost of fencing the given irregular pentagonal field --/
theorem pentagonal_field_fencing_cost :
  fencing_cost 42 35 52 66 40 3 = 705 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_field_fencing_cost_l4098_409816


namespace NUMINAMATH_CALUDE_greatest_second_term_arithmetic_sequence_l4098_409841

theorem greatest_second_term_arithmetic_sequence :
  ∀ (a d : ℕ),
    a > 0 →
    d > 0 →
    a + (a + d) + (a + 2*d) + (a + 3*d) = 80 →
    ∀ (b e : ℕ),
      b > 0 →
      e > 0 →
      b + (b + e) + (b + 2*e) + (b + 3*e) = 80 →
      a + d ≤ 19 :=
by sorry

end NUMINAMATH_CALUDE_greatest_second_term_arithmetic_sequence_l4098_409841


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l4098_409828

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 2*x - 2 = 0) :
  3*x^2 - 6*x + 9 = 15 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l4098_409828


namespace NUMINAMATH_CALUDE_equation_solutions_l4098_409842

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, (2*x - 1)^2 = (3 - x)^2 ↔ x = x₁ ∨ x = x₂) ∧ x₁ = -2 ∧ x₂ = 4/3) ∧
  (∃ y₁ y₂ : ℝ, (∀ x : ℝ, x^2 - Real.sqrt 3 * x - 1/4 = 0 ↔ x = y₁ ∨ x = y₂) ∧ 
    y₁ = (Real.sqrt 3 + 2)/2 ∧ y₂ = (Real.sqrt 3 - 2)/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4098_409842


namespace NUMINAMATH_CALUDE_tank_capacity_theorem_l4098_409843

/-- Represents a tank with a leak and an inlet pipe. -/
structure Tank where
  capacity : ℝ
  leak_empty_time : ℝ
  inlet_rate : ℝ
  combined_empty_time : ℝ

/-- Theorem stating the relationship between tank properties and its capacity. -/
theorem tank_capacity_theorem (t : Tank) 
  (h1 : t.leak_empty_time = 6)
  (h2 : t.inlet_rate = 2.5 * 60)
  (h3 : t.combined_empty_time = 8) :
  t.capacity = 3600 / 7 := by
  sorry

#check tank_capacity_theorem

end NUMINAMATH_CALUDE_tank_capacity_theorem_l4098_409843


namespace NUMINAMATH_CALUDE_range_of_a_l4098_409844

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + a| < 3) → a ∈ Set.Ioo (-4) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4098_409844


namespace NUMINAMATH_CALUDE_mike_plant_cost_l4098_409817

theorem mike_plant_cost (rose_price : ℝ) (rose_quantity : ℕ) 
  (rose_discount : ℝ) (rose_tax : ℝ) (aloe_price : ℝ) 
  (aloe_quantity : ℕ) (aloe_tax : ℝ) (friend_roses : ℕ) :
  rose_price = 75 ∧ 
  rose_quantity = 6 ∧ 
  rose_discount = 0.1 ∧ 
  rose_tax = 0.05 ∧ 
  aloe_price = 100 ∧ 
  aloe_quantity = 2 ∧ 
  aloe_tax = 0.07 ∧ 
  friend_roses = 2 →
  let total_rose_cost := rose_price * rose_quantity * (1 - rose_discount) * (1 + rose_tax)
  let friend_rose_cost := rose_price * friend_roses * (1 - rose_discount) * (1 + rose_tax)
  let aloe_cost := aloe_price * aloe_quantity * (1 + aloe_tax)
  total_rose_cost - friend_rose_cost + aloe_cost = 497.50 := by
sorry


end NUMINAMATH_CALUDE_mike_plant_cost_l4098_409817


namespace NUMINAMATH_CALUDE_not_divisible_3n_minus_1_by_2n_minus_1_l4098_409801

theorem not_divisible_3n_minus_1_by_2n_minus_1 (n : ℕ) (h : n > 1) :
  ¬(2^n - 1 ∣ 3^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_3n_minus_1_by_2n_minus_1_l4098_409801


namespace NUMINAMATH_CALUDE_different_testing_methods_part1_different_testing_methods_part2_l4098_409857

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

end NUMINAMATH_CALUDE_different_testing_methods_part1_different_testing_methods_part2_l4098_409857


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l4098_409800

/-- Calculates the final amount after compound interest for two years with different rates -/
def final_amount (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount1 := initial * (1 + rate1)
  amount1 * (1 + rate2)

/-- Theorem stating that given the initial amount and interest rates, 
    the final amount after 2 years is as calculated -/
theorem compound_interest_calculation :
  final_amount 4368 0.04 0.05 = 4769.856 := by
  sorry

#eval final_amount 4368 0.04 0.05

end NUMINAMATH_CALUDE_compound_interest_calculation_l4098_409800


namespace NUMINAMATH_CALUDE_average_age_is_35_l4098_409873

/-- Represents the ages of John, Mary, and Tonya -/
structure Ages where
  john : ℕ
  mary : ℕ
  tonya : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.john = 2 * ages.mary ∧
  2 * ages.john = ages.tonya ∧
  ages.tonya = 60

/-- The average age of John, Mary, and Tonya -/
def average_age (ages : Ages) : ℚ :=
  (ages.john + ages.mary + ages.tonya : ℚ) / 3

/-- Theorem stating that the average age is 35 given the conditions -/
theorem average_age_is_35 (ages : Ages) (h : satisfies_conditions ages) :
  average_age ages = 35 := by
  sorry

end NUMINAMATH_CALUDE_average_age_is_35_l4098_409873


namespace NUMINAMATH_CALUDE_grandfather_age_relationship_l4098_409834

/-- Represents the ages and relationships in the family problem -/
structure FamilyAges where
  fatherCurrentAge : ℕ
  sonCurrentAge : ℕ
  grandfatherAgeFiveYearsAgo : ℕ
  fatherAgeSameAsSonAtBirth : fatherCurrentAge = sonCurrentAge + sonCurrentAge
  fatherCurrentAge58 : fatherCurrentAge = 58
  sonAgeFiveYearsAgoHalfGrandfather : sonCurrentAge - 5 = (grandfatherAgeFiveYearsAgo - 5) / 2

/-- Theorem stating the relationship between the grandfather's age 5 years ago and the son's current age -/
theorem grandfather_age_relationship (f : FamilyAges) : 
  f.grandfatherAgeFiveYearsAgo = 2 * f.sonCurrentAge - 5 := by
  sorry

#check grandfather_age_relationship

end NUMINAMATH_CALUDE_grandfather_age_relationship_l4098_409834


namespace NUMINAMATH_CALUDE_area_of_triangle_FOH_l4098_409887

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- Theorem about the area of triangle FOH in a trapezoid -/
theorem area_of_triangle_FOH (t : Trapezoid) 
  (h1 : t.base1 = 40)
  (h2 : t.base2 = 50)
  (h3 : t.area = 900) : 
  ∃ (area_FOH : ℝ), abs (area_FOH - 400/9) < 0.01 := by
  sorry

#check area_of_triangle_FOH

end NUMINAMATH_CALUDE_area_of_triangle_FOH_l4098_409887


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l4098_409896

theorem min_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (sum_eq : x + y + z = 3) (z_eq : z = 1) :
  1/x + 1/y + 1/z ≥ 3 ∧ (1/x + 1/y + 1/z = 3 ↔ x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l4098_409896


namespace NUMINAMATH_CALUDE_shorter_can_radius_l4098_409815

/-- Represents a cylindrical can with radius and height -/
structure Can where
  radius : ℝ
  height : ℝ

/-- Given two cans with equal volume, one with 4 times the height of the other,
    and the taller can having a radius of 5, prove the radius of the shorter can is 10 -/
theorem shorter_can_radius (can1 can2 : Can) 
  (h_volume : π * can1.radius^2 * can1.height = π * can2.radius^2 * can2.height)
  (h_height : can2.height = 4 * can1.height)
  (h_taller_radius : can2.radius = 5) :
  can1.radius = 10 := by
  sorry

end NUMINAMATH_CALUDE_shorter_can_radius_l4098_409815


namespace NUMINAMATH_CALUDE_factor_expression_l4098_409833

theorem factor_expression (x : ℝ) : 54 * x^6 - 231 * x^13 = 3 * x^6 * (18 - 77 * x^7) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4098_409833


namespace NUMINAMATH_CALUDE_lcm_of_three_numbers_specific_lcm_l4098_409845

theorem lcm_of_three_numbers (a b c : ℕ) (hcf : ℕ) (h_hcf : Nat.gcd a (Nat.gcd b c) = hcf) :
  Nat.lcm a (Nat.lcm b c) = a * b * c / hcf :=
by sorry

theorem specific_lcm :
  Nat.lcm 136 (Nat.lcm 144 168) = 411264 :=
by
  have h_hcf : Nat.gcd 136 (Nat.gcd 144 168) = 8 := by sorry
  exact lcm_of_three_numbers 136 144 168 8 h_hcf

end NUMINAMATH_CALUDE_lcm_of_three_numbers_specific_lcm_l4098_409845


namespace NUMINAMATH_CALUDE_dog_walker_base_charge_l4098_409888

/-- Represents the earnings of a dog walker given their base charge per dog and walking durations. -/
def dog_walker_earnings (base_charge : ℝ) : ℝ :=
  (base_charge + 10 * 1) +  -- One dog for 10 minutes
  (2 * base_charge + 2 * 7 * 1) +  -- Two dogs for 7 minutes each
  (3 * base_charge + 3 * 9 * 1)  -- Three dogs for 9 minutes each

/-- Theorem stating that if a dog walker earns $171 with the given walking schedule, 
    their base charge per dog must be $20. -/
theorem dog_walker_base_charge : 
  ∃ (x : ℝ), dog_walker_earnings x = 171 → x = 20 :=
sorry

end NUMINAMATH_CALUDE_dog_walker_base_charge_l4098_409888


namespace NUMINAMATH_CALUDE_players_who_quit_l4098_409855

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) 
  (h1 : initial_players = 8)
  (h2 : lives_per_player = 3)
  (h3 : total_lives = 15) :
  initial_players - (total_lives / lives_per_player) = 3 :=
by sorry

end NUMINAMATH_CALUDE_players_who_quit_l4098_409855


namespace NUMINAMATH_CALUDE_sunny_lead_in_new_race_l4098_409895

/-- Represents the race conditions and results -/
structure RaceData where
  initial_race_length : ℝ
  initial_sunny_lead : ℝ
  new_race_length : ℝ
  sunny_speed_increase : ℝ
  windy_speed_decrease : ℝ
  sunny_initial_lag : ℝ

/-- Calculates Sunny's lead at the end of the new race -/
def calculate_sunny_lead (data : RaceData) : ℝ :=
  sorry

/-- Theorem stating that given the race conditions, Sunny's lead at the end of the new race is 106.25 meters -/
theorem sunny_lead_in_new_race (data : RaceData) 
  (h1 : data.initial_race_length = 400)
  (h2 : data.initial_sunny_lead = 50)
  (h3 : data.new_race_length = 500)
  (h4 : data.sunny_speed_increase = 0.1)
  (h5 : data.windy_speed_decrease = 0.1)
  (h6 : data.sunny_initial_lag = 50) :
  calculate_sunny_lead data = 106.25 :=
sorry

end NUMINAMATH_CALUDE_sunny_lead_in_new_race_l4098_409895


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l4098_409884

theorem binomial_expansion_coefficient (x : ℝ) : 
  let expansion := (x - 2 / Real.sqrt x) ^ 7
  ∃ (a b c : ℝ), expansion = a*x + 560*x + b*x^2 + c :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l4098_409884


namespace NUMINAMATH_CALUDE_vector_magnitude_l4098_409880

theorem vector_magnitude (a b : ℝ × ℝ) :
  let angle := 60 * π / 180
  (a.1^2 + a.2^2 = 4) →
  (b.1^2 + b.2^2 = 1) →
  (a.1 * b.1 + a.2 * b.2 = 2 * Real.cos angle) →
  ((a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2 = 4) :=
by
  sorry

#check vector_magnitude

end NUMINAMATH_CALUDE_vector_magnitude_l4098_409880


namespace NUMINAMATH_CALUDE_special_rectangle_side_gt_12_l4098_409832

/-- A rectangle with sides a and b, where the area is three times the perimeter --/
structure SpecialRectangle where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : a ≠ b
  h4 : a * b = 3 * (2 * (a + b))

/-- Theorem: For a SpecialRectangle, one of its sides is greater than 12 --/
theorem special_rectangle_side_gt_12 (rect : SpecialRectangle) : max rect.a rect.b > 12 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_side_gt_12_l4098_409832


namespace NUMINAMATH_CALUDE_three_black_reachable_l4098_409881

structure UrnState :=
  (black : ℕ)
  (white : ℕ)

def initial_state : UrnState :=
  ⟨100, 120⟩

inductive Operation
  | replace_3b_with_2b
  | replace_2b1w_with_1b1w
  | replace_1b2w_with_2w
  | replace_3w_with_1b1w

def apply_operation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.replace_3b_with_2b => ⟨state.black - 1, state.white⟩
  | Operation.replace_2b1w_with_1b1w => ⟨state.black - 1, state.white⟩
  | Operation.replace_1b2w_with_2w => ⟨state.black - 1, state.white⟩
  | Operation.replace_3w_with_1b1w => ⟨state.black + 1, state.white - 2⟩

def reachable (target : UrnState) : Prop :=
  ∃ (n : ℕ) (ops : Fin n → Operation),
    (List.foldl apply_operation initial_state (List.ofFn ops)) = target

theorem three_black_reachable :
  reachable ⟨3, 120⟩ :=
sorry

end NUMINAMATH_CALUDE_three_black_reachable_l4098_409881


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l4098_409811

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + Nat.factorial 6 = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l4098_409811


namespace NUMINAMATH_CALUDE_correct_product_l4098_409853

/-- Given two positive integers a and b, where a is a two-digit number,
    if reversing the digits of a and multiplying by b results in 172,
    then the correct product of a and b is 136. -/
theorem correct_product (a b : ℕ) : 
  (a ≥ 10 ∧ a ≤ 99) →  -- a is a two-digit number
  (b > 0) →  -- b is positive
  (((a % 10) * 10 + (a / 10)) * b = 172) →  -- reversing digits of a and multiplying by b gives 172
  (a * b = 136) :=
by sorry

end NUMINAMATH_CALUDE_correct_product_l4098_409853


namespace NUMINAMATH_CALUDE_right_triangle_and_symmetric_circle_l4098_409876

/-- Given a right triangle OAB in a rectangular coordinate system where:
  - O is the origin (0, 0)
  - A is the right-angle vertex at (4, -3)
  - |AB| = 2|OA|
  - The y-coordinate of B is positive
This theorem proves the coordinates of B and the equation of a symmetric circle. -/
theorem right_triangle_and_symmetric_circle :
  ∃ (B : ℝ × ℝ),
    let O : ℝ × ℝ := (0, 0)
    let A : ℝ × ℝ := (4, -3)
    -- B is in the first quadrant
    B.1 > 0 ∧ B.2 > 0 ∧
    -- OA ⟂ AB (right angle at A)
    (B.1 - A.1) * A.1 + (B.2 - A.2) * A.2 = 0 ∧
    -- |AB| = 2|OA|
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = 4 * (A.1^2 + A.2^2) ∧
    -- B has coordinates (10, 5)
    B = (10, 5) ∧
    -- The equation of the symmetric circle
    ∀ (x y : ℝ),
      (x^2 - 6*x + y^2 + 2*y = 0) ↔
      ((x - 1)^2 + (y - 3)^2 = 10) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_and_symmetric_circle_l4098_409876


namespace NUMINAMATH_CALUDE_sum_of_digits_7_pow_23_l4098_409854

/-- Returns the ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- Returns the tens digit of a natural number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- The sum of the tens digit and the ones digit of 7^23 is 7 -/
theorem sum_of_digits_7_pow_23 :
  tensDigit (7^23) + onesDigit (7^23) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_7_pow_23_l4098_409854


namespace NUMINAMATH_CALUDE_parabola_transformation_l4098_409863

/-- Given two parabolas, prove that one is a transformation of the other -/
theorem parabola_transformation (x y : ℝ) : 
  (y = 2 * x^2) →
  (y = 2 * (x - 4)^2 + 1) ↔ 
  (∃ (x' y' : ℝ), x' = x - 4 ∧ y' = y - 1 ∧ y' = 2 * x'^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l4098_409863


namespace NUMINAMATH_CALUDE_c_value_l4098_409806

def f (a c x : ℝ) : ℝ := a * x^3 + c

theorem c_value (a c : ℝ) :
  (∃ x, f a c x = 20 ∧ x ∈ Set.Icc 1 2) ∧
  (∀ x, x ∈ Set.Icc 1 2 → f a c x ≤ 20) ∧
  (deriv (f a c) 1 = 6) →
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_c_value_l4098_409806


namespace NUMINAMATH_CALUDE_sin_18_deg_identity_l4098_409822

theorem sin_18_deg_identity :
  let x : ℝ := Real.sin (18 * π / 180)
  4 * x^2 + 2 * x = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_deg_identity_l4098_409822


namespace NUMINAMATH_CALUDE_safe_combinations_l4098_409894

def digits : Finset Nat := {1, 3, 5}

theorem safe_combinations : Fintype.card (Equiv.Perm digits) = 6 := by
  sorry

end NUMINAMATH_CALUDE_safe_combinations_l4098_409894


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l4098_409893

/-- The amount of flour Mary put in -/
def flour_added : ℝ := 7.5

/-- The amount of excess flour added -/
def excess_flour : ℝ := 0.8

/-- The amount of flour the recipe wants -/
def recipe_flour : ℝ := flour_added - excess_flour

theorem recipe_flour_amount : recipe_flour = 6.7 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l4098_409893


namespace NUMINAMATH_CALUDE_cloth_trimming_l4098_409818

/-- The number of feet trimmed from opposite edges of a square cloth -/
def x : ℝ := 15

/-- The original side length of the square cloth in feet -/
def original_side : ℝ := 22

/-- The remaining area of cloth after trimming in square feet -/
def remaining_area : ℝ := 120

/-- The number of feet trimmed from the other two edges -/
def other_edge_trim : ℝ := 5

theorem cloth_trimming :
  round ((original_side - x) * (original_side - other_edge_trim) - remaining_area) = 0 :=
sorry

end NUMINAMATH_CALUDE_cloth_trimming_l4098_409818


namespace NUMINAMATH_CALUDE_parabola_theorem_l4098_409819

/-- Parabola with parameter p and a tangent line -/
structure Parabola where
  p : ℝ
  tangent_x_intercept : ℝ
  tangent_y_intercept : ℝ

/-- Properties of the parabola -/
def parabola_properties (para : Parabola) : Prop :=
  -- Tangent line equation matches the given form
  para.tangent_x_intercept = -75 ∧ para.tangent_y_intercept = 15 ∧
  -- Parameter p is 6
  para.p = 6 ∧
  -- Focus coordinates are (3, 0)
  (3 : ℝ) = para.p / 2 ∧
  -- Directrix equation is x = -3
  (-3 : ℝ) = -para.p / 2

/-- Theorem stating the properties of the parabola -/
theorem parabola_theorem (para : Parabola) :
  parabola_properties para :=
sorry

end NUMINAMATH_CALUDE_parabola_theorem_l4098_409819


namespace NUMINAMATH_CALUDE_pen_ratio_theorem_l4098_409835

/-- Represents the number of pens bought by each person -/
structure PenPurchase where
  julia : ℕ
  dorothy : ℕ
  robert : ℕ

/-- Represents the given conditions of the problem -/
def ProblemConditions (p : PenPurchase) : Prop :=
  p.dorothy = p.julia / 2 ∧
  p.robert = 4 ∧
  p.julia + p.dorothy + p.robert = 22

theorem pen_ratio_theorem (p : PenPurchase) :
  ProblemConditions p → p.julia / p.robert = 3 := by
  sorry

#check pen_ratio_theorem

end NUMINAMATH_CALUDE_pen_ratio_theorem_l4098_409835
