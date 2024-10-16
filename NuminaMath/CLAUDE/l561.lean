import Mathlib

namespace NUMINAMATH_CALUDE_function_transformation_l561_56157

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_transformation (x : ℝ) :
  (∀ y : ℝ, f (y + 1) = 3 * y + 4) →
  f x = 3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l561_56157


namespace NUMINAMATH_CALUDE_hotel_bubble_bath_l561_56119

/-- The amount of bubble bath needed for a hotel with couples and single rooms -/
def bubble_bath_needed (couple_rooms single_rooms : ℕ) (bath_per_person : ℕ) : ℕ :=
  (2 * couple_rooms + single_rooms) * bath_per_person

/-- Theorem: The amount of bubble bath needed for 13 couple rooms and 14 single rooms is 400ml -/
theorem hotel_bubble_bath :
  bubble_bath_needed 13 14 10 = 400 := by
  sorry

end NUMINAMATH_CALUDE_hotel_bubble_bath_l561_56119


namespace NUMINAMATH_CALUDE_college_students_count_l561_56190

/-- Calculates the total number of students in a college given the ratio of boys to girls and the number of girls. -/
def totalStudents (boyRatio girlRatio numGirls : ℕ) : ℕ :=
  let numBoys := boyRatio * numGirls / girlRatio
  numBoys + numGirls

/-- Theorem: In a college where the ratio of boys to girls is 8:5 and there are 190 girls, the total number of students is 494. -/
theorem college_students_count :
  totalStudents 8 5 190 = 494 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l561_56190


namespace NUMINAMATH_CALUDE_at_least_one_passes_l561_56162

theorem at_least_one_passes (p : ℝ) (h : p = 1/3) :
  let q := 1 - p
  1 - q^3 = 19/27 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_passes_l561_56162


namespace NUMINAMATH_CALUDE_total_sum_calculation_l561_56121

/-- 
Given that Maggie's share is 75% of the total sum and equals $4,500, 
prove that the total sum is $6,000.
-/
theorem total_sum_calculation (maggies_share : ℝ) (total_sum : ℝ) : 
  maggies_share = 4500 ∧ 
  maggies_share = 0.75 * total_sum →
  total_sum = 6000 := by
  sorry

end NUMINAMATH_CALUDE_total_sum_calculation_l561_56121


namespace NUMINAMATH_CALUDE_triangle_angle_C_l561_56113

open Real

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_angle_C (t : Triangle) 
  (h : t.b * (2 * sin t.B + sin t.A) + (2 * t.a + t.b) * sin t.A = 2 * t.c * sin t.C) :
  t.C = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l561_56113


namespace NUMINAMATH_CALUDE_zachary_exercise_difference_l561_56139

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 46

/-- The number of crunches Zachary did -/
def zachary_crunches : ℕ := 58

/-- The difference between Zachary's crunches and push-ups -/
def zachary_difference : ℤ := zachary_crunches - zachary_pushups

theorem zachary_exercise_difference : zachary_difference = 12 := by
  sorry

end NUMINAMATH_CALUDE_zachary_exercise_difference_l561_56139


namespace NUMINAMATH_CALUDE_smallest_b_value_l561_56137

theorem smallest_b_value (a b : ℕ) : 
  (a ≥ 1000) → (a ≤ 9999) → (b ≥ 100000) → (b ≤ 999999) → 
  (1 : ℚ) / 2006 = 1 / a + 1 / b → 
  ∀ b' ≥ 100000, b' ≤ 999999 → 
    ∃ a' ≥ 1000, a' ≤ 9999 → (1 : ℚ) / 2006 = 1 / a' + 1 / b' → 
      b ≤ b' → b = 120360 := by
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l561_56137


namespace NUMINAMATH_CALUDE_mike_taller_than_mark_l561_56155

/-- Converts feet and inches to total inches -/
def heightToInches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- The height difference between two people in inches -/
def heightDifference (height1 : ℕ) (height2 : ℕ) : ℕ :=
  max height1 height2 - min height1 height2

theorem mike_taller_than_mark : 
  let markHeight := heightToInches 5 3
  let mikeHeight := heightToInches 6 1
  heightDifference markHeight mikeHeight = 10 := by
sorry

end NUMINAMATH_CALUDE_mike_taller_than_mark_l561_56155


namespace NUMINAMATH_CALUDE_construct_quadrilateral_l561_56111

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Checks if three sides of a quadrilateral are equal -/
def hasThreeEqualSides (q : Quadrilateral) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (M : Point) (A B : Point) : Prop := sorry

/-- Theorem: Given three points that are midpoints of three equal sides of a convex quadrilateral,
    a unique quadrilateral can be constructed -/
theorem construct_quadrilateral 
  (P Q R : Point) 
  (h_exists : ∃ (q : Quadrilateral), 
    isConvex q ∧ 
    hasThreeEqualSides q ∧
    isMidpoint P q.A q.B ∧
    isMidpoint Q q.B q.C ∧
    isMidpoint R q.C q.D) :
  ∃! (q : Quadrilateral), 
    isConvex q ∧ 
    hasThreeEqualSides q ∧
    isMidpoint P q.A q.B ∧
    isMidpoint Q q.B q.C ∧
    isMidpoint R q.C q.D :=
sorry

end NUMINAMATH_CALUDE_construct_quadrilateral_l561_56111


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l561_56102

theorem arithmetic_square_root_of_nine :
  Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l561_56102


namespace NUMINAMATH_CALUDE_inverse_of_21_mod_47_l561_56129

theorem inverse_of_21_mod_47 (h : (8⁻¹ : ZMod 47) = 6) : (21⁻¹ : ZMod 47) = 38 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_21_mod_47_l561_56129


namespace NUMINAMATH_CALUDE_larger_number_proof_l561_56197

theorem larger_number_proof (L S : ℕ) (hL : L > S) (h1 : L - S = 2500) (h2 : L = 6 * S + 15) : L = 2997 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l561_56197


namespace NUMINAMATH_CALUDE_gcd_51_119_l561_56185

theorem gcd_51_119 : Nat.gcd 51 119 = 17 := by sorry

end NUMINAMATH_CALUDE_gcd_51_119_l561_56185


namespace NUMINAMATH_CALUDE_polynomial_division_result_l561_56127

theorem polynomial_division_result (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (25 * x^2 * y - 5 * x * y^2) / (5 * x * y) = 5 * x - y := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_result_l561_56127


namespace NUMINAMATH_CALUDE_percentage_problem_l561_56188

theorem percentage_problem (y : ℝ) (h1 : y > 0) (h2 : (y / 100) * y = 9) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l561_56188


namespace NUMINAMATH_CALUDE_triangle_ad_length_l561_56175

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

end NUMINAMATH_CALUDE_triangle_ad_length_l561_56175


namespace NUMINAMATH_CALUDE_line_properties_l561_56194

/-- The line l₁ with equation (m + 1)x - (m - 3)y - 8 = 0 where m ∈ ℝ --/
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 1) * x - (m - 3) * y - 8 = 0

/-- The line l₂ parallel to l₁ passing through the origin --/
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m + 1) * x - (m - 3) * y = 0

theorem line_properties :
  (∀ m : ℝ, l₁ m 2 2) ∧ 
  (∀ x y : ℝ, x + y = 0 → (∀ m : ℝ, l₂ m x y) ∧ 
    ∀ a b : ℝ, l₂ m a b → (a^2 + b^2 ≤ x^2 + y^2)) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l561_56194


namespace NUMINAMATH_CALUDE_product_of_primes_summing_to_17_l561_56191

theorem product_of_primes_summing_to_17 (p₁ p₂ p₃ p₄ : ℕ) : 
  p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧ p₄.Prime ∧ 
  p₁ + p₂ + p₃ + p₄ = 17 → 
  p₁ * p₂ * p₃ * p₄ = 210 := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_summing_to_17_l561_56191


namespace NUMINAMATH_CALUDE_transformed_quadratic_equation_l561_56184

theorem transformed_quadratic_equation 
  (p q : ℝ) 
  (x₁ x₂ : ℝ) 
  (h1 : x₁^2 + p*x₁ + q = 0) 
  (h2 : x₂^2 + p*x₂ + q = 0) 
  (h3 : x₁ ≠ x₂) :
  ∃ (t : ℝ), q*t^2 + (q+1)*t + 1 = 0 ↔ 
    (t = x₁ + 1/x₁ ∨ t = x₂ + 1/x₂) :=
by sorry

end NUMINAMATH_CALUDE_transformed_quadratic_equation_l561_56184


namespace NUMINAMATH_CALUDE_probability_four_old_balls_value_l561_56132

def total_balls : ℕ := 12
def new_balls : ℕ := 9
def old_balls : ℕ := 3
def drawn_balls : ℕ := 3

def probability_four_old_balls : ℚ :=
  (Nat.choose old_balls 2 * Nat.choose new_balls 1) / Nat.choose total_balls drawn_balls

theorem probability_four_old_balls_value :
  probability_four_old_balls = 27 / 220 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_old_balls_value_l561_56132


namespace NUMINAMATH_CALUDE_cats_sold_proof_l561_56115

/-- Calculates the number of cats sold during a sale at a pet store. -/
def cats_sold (siamese : ℕ) (house : ℕ) (left : ℕ) : ℕ :=
  siamese + house - left

/-- Proves that the number of cats sold during the sale is 45. -/
theorem cats_sold_proof :
  cats_sold 38 25 18 = 45 := by
  sorry

end NUMINAMATH_CALUDE_cats_sold_proof_l561_56115


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l561_56178

/-- Represents a point in 2D plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semimajor : ℚ
  semiminor : ℚ

/-- Check if a point lies on the ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.semimajor^2 + (p.y - e.center.y)^2 / e.semiminor^2 = 1

/-- The six given points -/
def points : List Point := [
  ⟨-5/2, 2⟩, ⟨0, 0⟩, ⟨0, 3⟩, ⟨4, 0⟩, ⟨4, 3⟩, ⟨2, 4⟩
]

/-- The ellipse passing through the points -/
def ellipse : Ellipse := ⟨⟨2, 3/2⟩, 2, 5/2⟩

theorem ellipse_minor_axis_length :
  (∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬ (∃ (m b : ℚ), p1.y = m * p1.x + b ∧ p2.y = m * p2.x + b ∧ p3.y = m * p3.x + b)) →
  (∀ p : Point, p ∈ points → pointOnEllipse p ellipse) →
  ellipse.semiminor * 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l561_56178


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l561_56108

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Predicate for a sequence being arithmetic -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Predicate for a point being on the line y = 2x + 1 -/
def on_line (n : ℕ) (a : Sequence) : Prop :=
  a n = 2 * n + 1

theorem sufficient_not_necessary :
  (∀ a : Sequence, (∀ n : ℕ, n > 0 → on_line n a) → is_arithmetic a) ∧
  (∃ a : Sequence, is_arithmetic a ∧ ∃ n : ℕ, n > 0 ∧ ¬on_line n a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l561_56108


namespace NUMINAMATH_CALUDE_multiple_sum_properties_l561_56141

theorem multiple_sum_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, a + b = 2 * n) ∧ 
  (¬ ∀ p : ℤ, a + b = 6 * p) ∧
  (¬ ∀ q : ℤ, a + b = 8 * q) ∧
  (∃ r s : ℤ, a = 6 * r ∧ b = 8 * s ∧ ¬ ∃ t : ℤ, a + b = 8 * t) :=
by sorry

end NUMINAMATH_CALUDE_multiple_sum_properties_l561_56141


namespace NUMINAMATH_CALUDE_inverse_of_BP_squared_l561_56192

/-- Given a 2x2 matrix B and a diagonal matrix P, prove that the inverse of (BP)² has a specific form. -/
theorem inverse_of_BP_squared (B P : Matrix (Fin 2) (Fin 2) ℚ) : 
  B⁻¹ = ![![3, 7], ![-2, -4]] →
  P = ![![1, 0], ![0, 2]] →
  ((B * P)^2)⁻¹ = ![![8, 28], ![-4, -12]] := by sorry

end NUMINAMATH_CALUDE_inverse_of_BP_squared_l561_56192


namespace NUMINAMATH_CALUDE_linda_basketball_scores_l561_56156

theorem linda_basketball_scores (first_seven : List Nat) 
  (h1 : first_seven = [5, 6, 4, 7, 3, 2, 6])
  (h2 : first_seven.length = 7)
  (eighth_game : Nat) (ninth_game : Nat)
  (h3 : eighth_game < 10)
  (h4 : ninth_game < 10)
  (h5 : (first_seven.sum + eighth_game) % 8 = 0)
  (h6 : (first_seven.sum + eighth_game + ninth_game) % 9 = 0) :
  eighth_game * ninth_game = 35 := by
sorry

end NUMINAMATH_CALUDE_linda_basketball_scores_l561_56156


namespace NUMINAMATH_CALUDE_batsman_overall_average_l561_56193

def total_matches : ℕ := 30
def first_set_matches : ℕ := 20
def second_set_matches : ℕ := 10
def first_set_average : ℕ := 30
def second_set_average : ℕ := 15

theorem batsman_overall_average :
  let first_set_total := first_set_matches * first_set_average
  let second_set_total := second_set_matches * second_set_average
  let total_runs := first_set_total + second_set_total
  total_runs / total_matches = 25 := by sorry

end NUMINAMATH_CALUDE_batsman_overall_average_l561_56193


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_property_l561_56182

theorem quadratic_equation_roots_property : ∃ (p q : ℝ),
  p + q = 7 ∧
  |p - q| = 9 ∧
  ∀ x, x^2 - 7*x - 8 = 0 ↔ (x = p ∨ x = q) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_property_l561_56182


namespace NUMINAMATH_CALUDE_inequality_solution_set_l561_56152

-- Define the function f
def f (x : ℝ) : ℝ := |x| - x + 1

-- State the theorem
theorem inequality_solution_set (x : ℝ) : 
  f (1 - x^2) > f (1 - 2*x) ↔ x > 2 ∨ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l561_56152


namespace NUMINAMATH_CALUDE_polygon_sides_l561_56179

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 540 → ∃ n : ℕ, n = 5 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l561_56179


namespace NUMINAMATH_CALUDE_square_completion_l561_56105

theorem square_completion (a h k : ℝ) : 
  (∀ x, x^2 - 6*x = a*(x - h)^2 + k) → k = -9 := by
  sorry

end NUMINAMATH_CALUDE_square_completion_l561_56105


namespace NUMINAMATH_CALUDE_f_upper_bound_l561_56165

-- Define the set A
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Define the properties of function f
def is_valid_f (f : ℝ → ℝ) : Prop :=
  f 1 = 1 ∧
  (∀ x ∈ A, f x ≥ 0) ∧
  (∀ x y, x ∈ A → y ∈ A → x + y ∈ A → f (x + y) ≥ f x + f y)

-- Theorem statement
theorem f_upper_bound 
  (f : ℝ → ℝ) 
  (hf : is_valid_f f) :
  ∀ x ∈ A, f x ≤ 2 * x :=
sorry

end NUMINAMATH_CALUDE_f_upper_bound_l561_56165


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l561_56101

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 > 4}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- State the theorem
theorem intersection_complement_theorem :
  N ∩ (Set.univ \ M) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l561_56101


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l561_56153

theorem smallest_n_congruence : ∃! n : ℕ, (∀ a ∈ Finset.range 9, n % (a + 2) = (a + 1)) ∧ 
  (∀ m : ℕ, m < n → ∃ a ∈ Finset.range 9, m % (a + 2) ≠ (a + 1)) ∧ n = 2519 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l561_56153


namespace NUMINAMATH_CALUDE_M_subset_N_l561_56163

def M : Set ℝ := {-1, 1}

def N : Set ℝ := {x | (1 : ℝ) / x < 2}

theorem M_subset_N : M ⊆ N := by sorry

end NUMINAMATH_CALUDE_M_subset_N_l561_56163


namespace NUMINAMATH_CALUDE_can_meet_in_three_jumps_l561_56112

-- Define the grid
def Grid := ℤ × ℤ

-- Define the color of a square
inductive Color
| Red
| White

-- Define the coloring function
def coloring : Grid → Color := sorry

-- Define the grasshopper's position
def grasshopper : Grid := sorry

-- Define the flea's position
def flea : Grid := sorry

-- Define a valid jump
def valid_jump (start finish : Grid) (c : Color) : Prop :=
  (coloring start = c ∧ coloring finish = c) ∧
  (start.1 = finish.1 ∨ start.2 = finish.2)

-- Define adjacent squares
def adjacent (a b : Grid) : Prop :=
  (abs (a.1 - b.1) + abs (a.2 - b.2) = 1)

-- Main theorem
theorem can_meet_in_three_jumps :
  ∃ (g1 g2 g3 f1 f2 f3 : Grid),
    (valid_jump grasshopper g1 Color.Red ∨ g1 = grasshopper) ∧
    (valid_jump g1 g2 Color.Red ∨ g2 = g1) ∧
    (valid_jump g2 g3 Color.Red ∨ g3 = g2) ∧
    (valid_jump flea f1 Color.White ∨ f1 = flea) ∧
    (valid_jump f1 f2 Color.White ∨ f2 = f1) ∧
    (valid_jump f2 f3 Color.White ∨ f3 = f2) ∧
    adjacent g3 f3 :=
  sorry


end NUMINAMATH_CALUDE_can_meet_in_three_jumps_l561_56112


namespace NUMINAMATH_CALUDE_additional_marbles_for_lisa_l561_56106

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed -/
theorem additional_marbles_for_lisa : 
  min_additional_marbles 12 40 = 38 := by
  sorry

end NUMINAMATH_CALUDE_additional_marbles_for_lisa_l561_56106


namespace NUMINAMATH_CALUDE_binomial_product_simplification_l561_56144

theorem binomial_product_simplification (x : ℝ) :
  (4 * x + 3) * (x - 6) = 4 * x^2 - 21 * x - 18 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_simplification_l561_56144


namespace NUMINAMATH_CALUDE_modulus_of_complex_power_l561_56198

theorem modulus_of_complex_power : Complex.abs ((2 : ℂ) + Complex.I) ^ 8 = 625 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_power_l561_56198


namespace NUMINAMATH_CALUDE_decreasing_quadratic_function_parameter_range_l561_56169

/-- If f(x) = x^2 - 2(1-a)x + 2 is a decreasing function on (-∞, 4], then a ∈ (-∞, -3] -/
theorem decreasing_quadratic_function_parameter_range (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → (x^2 - 2*(1-a)*x + 2) > (y^2 - 2*(1-a)*y + 2)) →
  a ∈ Set.Iic (-3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_function_parameter_range_l561_56169


namespace NUMINAMATH_CALUDE_equation_two_complex_roots_l561_56104

/-- The equation under consideration -/
def equation (x k : ℂ) : Prop :=
  x / (x + 2) + x / (x + 3) = k * x

/-- The equation has exactly two complex roots -/
def has_two_complex_roots (k : ℂ) : Prop :=
  ∃! (r₁ r₂ : ℂ), r₁ ≠ r₂ ∧ ∀ x, equation x k ↔ x = 0 ∨ x = r₁ ∨ x = r₂

/-- The main theorem stating the condition for the equation to have exactly two complex roots -/
theorem equation_two_complex_roots :
  ∀ k : ℂ, has_two_complex_roots k ↔ k = 2*I ∨ k = -2*I :=
sorry

end NUMINAMATH_CALUDE_equation_two_complex_roots_l561_56104


namespace NUMINAMATH_CALUDE_smallest_divisor_with_remainder_l561_56126

theorem smallest_divisor_with_remainder (d : ℕ) : d = 6 ↔ 
  d > 1 ∧ 
  (∀ n : ℤ, n % d = 1 → (5 * n) % d = 5) ∧
  (∀ d' : ℕ, d' < d → d' > 1 → ∃ n : ℤ, n % d' = 1 ∧ (5 * n) % d' ≠ 5) :=
by sorry

#check smallest_divisor_with_remainder

end NUMINAMATH_CALUDE_smallest_divisor_with_remainder_l561_56126


namespace NUMINAMATH_CALUDE_delta_donuts_calculation_l561_56107

def total_donuts : ℕ := 40
def gamma_donuts : ℕ := 8
def beta_donuts : ℕ := 3 * gamma_donuts

theorem delta_donuts_calculation :
  total_donuts - (beta_donuts + gamma_donuts) = 8 :=
by sorry

end NUMINAMATH_CALUDE_delta_donuts_calculation_l561_56107


namespace NUMINAMATH_CALUDE_line_equation_proof_line_parameters_l561_56186

/-- Given a line defined by (3, -4) · ((x, y) - (-2, 8)) = 0, prove that it can be expressed as y = (3/4)x + 9.5 with m = 3/4 and b = 9.5 -/
theorem line_equation_proof (x y : ℝ) :
  (3 * (x + 2) + (-4) * (y - 8) = 0) ↔ (y = (3 / 4) * x + (19 / 2)) :=
by sorry

/-- Prove that for the given line, m = 3/4 and b = 9.5 -/
theorem line_parameters :
  ∃ (m b : ℝ), m = 3 / 4 ∧ b = 19 / 2 ∧
  ∀ (x y : ℝ), (3 * (x + 2) + (-4) * (y - 8) = 0) ↔ (y = m * x + b) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_line_parameters_l561_56186


namespace NUMINAMATH_CALUDE_ceiling_times_x_equals_156_l561_56135

theorem ceiling_times_x_equals_156 :
  ∃ x : ℝ, x > 0 ∧ ⌈x⌉ = 13 ∧ ⌈x⌉ * x = 156 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_times_x_equals_156_l561_56135


namespace NUMINAMATH_CALUDE_sum_of_decimals_l561_56166

theorem sum_of_decimals : 5.47 + 2.58 + 1.95 = 10.00 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l561_56166


namespace NUMINAMATH_CALUDE_homework_duration_decrease_l561_56167

/-- Represents the decrease in homework duration over two adjustments --/
theorem homework_duration_decrease (initial_duration final_duration : ℝ) (x : ℝ) :
  initial_duration = 120 →
  final_duration = 60 →
  initial_duration * (1 - x)^2 = final_duration :=
by sorry

end NUMINAMATH_CALUDE_homework_duration_decrease_l561_56167


namespace NUMINAMATH_CALUDE_contrapositive_example_l561_56145

theorem contrapositive_example : 
  (∀ x : ℝ, x > 2 → x > 1) ↔ (∀ x : ℝ, x ≤ 1 → x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_example_l561_56145


namespace NUMINAMATH_CALUDE_ned_remaining_lives_l561_56177

/-- Given that Ned started with 83 lives and lost 13 lives, prove that he now has 70 lives. -/
theorem ned_remaining_lives (initial_lives : ℕ) (lost_lives : ℕ) (remaining_lives : ℕ) : 
  initial_lives = 83 → lost_lives = 13 → remaining_lives = initial_lives - lost_lives → remaining_lives = 70 := by
  sorry

end NUMINAMATH_CALUDE_ned_remaining_lives_l561_56177


namespace NUMINAMATH_CALUDE_relationship_abc_l561_56160

theorem relationship_abc (a b c : ℕ) : 
  a = 2^555 → b = 3^444 → c = 6^222 → a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l561_56160


namespace NUMINAMATH_CALUDE_negation_equivalence_no_real_solutions_range_sufficient_not_necessary_condition_l561_56180

-- Statement 1
theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x - 2 = 0) ↔ (∀ x : ℝ, x^2 - x - 2 ≠ 0) :=
sorry

-- Statement 2
theorem no_real_solutions_range (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 4*x + m = 0) → m > 4 :=
sorry

-- Statement 3
theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > 1 → 1/a < 1) ∧ (∃ a : ℝ, 1/a < 1 ∧ ¬(a > 1)) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_no_real_solutions_range_sufficient_not_necessary_condition_l561_56180


namespace NUMINAMATH_CALUDE_eraser_difference_l561_56110

theorem eraser_difference (andrea_erasers : ℕ) (anya_multiplier : ℕ) : 
  andrea_erasers = 4 →
  anya_multiplier = 4 →
  anya_multiplier * andrea_erasers - andrea_erasers = 12 := by
  sorry

end NUMINAMATH_CALUDE_eraser_difference_l561_56110


namespace NUMINAMATH_CALUDE_interval_eq_set_representation_l561_56147

-- Define the interval (-3, 2]
def interval : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 2}

-- Define the set representation
def set_representation : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 2}

-- Theorem stating that the interval and set representation are equal
theorem interval_eq_set_representation : interval = set_representation := by
  sorry

end NUMINAMATH_CALUDE_interval_eq_set_representation_l561_56147


namespace NUMINAMATH_CALUDE_min_occupied_seats_180_l561_56143

/-- Given a row of seats, returns the minimum number of occupied seats required
    to ensure the next person must sit next to someone. -/
def minOccupiedSeats (totalSeats : ℕ) : ℕ :=
  (totalSeats + 3) / 4

/-- Theorem stating that for 180 seats, the minimum number of occupied seats
    required to ensure the next person must sit next to someone is 45. -/
theorem min_occupied_seats_180 :
  minOccupiedSeats 180 = 45 := by sorry

end NUMINAMATH_CALUDE_min_occupied_seats_180_l561_56143


namespace NUMINAMATH_CALUDE_pipe_B_fill_time_l561_56159

/-- The time it takes for pipe A to fill the cistern (in minutes) -/
def time_A : ℝ := 45

/-- The time it takes for the third pipe to empty the cistern (in minutes) -/
def time_empty : ℝ := 72

/-- The time it takes to fill the cistern when all three pipes are open (in minutes) -/
def time_all : ℝ := 40

/-- The time it takes for pipe B to fill the cistern (in minutes) -/
def time_B : ℝ := 60

theorem pipe_B_fill_time :
  ∃ (t : ℝ), t > 0 ∧ 1 / time_A + 1 / t - 1 / time_empty = 1 / time_all ∧ t = time_B := by
  sorry

end NUMINAMATH_CALUDE_pipe_B_fill_time_l561_56159


namespace NUMINAMATH_CALUDE_negation_equivalence_l561_56189

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Doctor : U → Prop)
variable (GoodAtMath : U → Prop)

-- Define the statements
def AllDoctorsGoodAtMath : Prop := ∀ x, Doctor x → GoodAtMath x
def AtLeastOneDoctorBadAtMath : Prop := ∃ x, Doctor x ∧ ¬GoodAtMath x

-- Theorem to prove
theorem negation_equivalence :
  AtLeastOneDoctorBadAtMath U Doctor GoodAtMath ↔ ¬(AllDoctorsGoodAtMath U Doctor GoodAtMath) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l561_56189


namespace NUMINAMATH_CALUDE_nabla_sum_equals_32_l561_56138

-- Define the ∇ operation
def nabla (k m : ℕ) : ℕ := k * (k - m)

-- State the theorem
theorem nabla_sum_equals_32 : nabla 5 1 + nabla 4 1 = 32 := by
  sorry

end NUMINAMATH_CALUDE_nabla_sum_equals_32_l561_56138


namespace NUMINAMATH_CALUDE_not_increasing_on_interval_l561_56100

-- Define the function f(x) = -x²
def f (x : ℝ) : ℝ := -x^2

-- Define what it means for a function to be increasing on an interval
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem not_increasing_on_interval : ¬ IsIncreasing f 0 2 := by
  sorry

end NUMINAMATH_CALUDE_not_increasing_on_interval_l561_56100


namespace NUMINAMATH_CALUDE_remainder_problem_l561_56103

theorem remainder_problem : (11^7 + 9^8 + 7^9) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l561_56103


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l561_56196

theorem fraction_to_decimal : (45 : ℚ) / (2^3 * 5^4) = (9 : ℚ) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l561_56196


namespace NUMINAMATH_CALUDE_no_fishes_brought_home_l561_56134

/-- Represents the number of fishes caught from a lake -/
def FishesCaught : Type := ℕ

/-- Represents whether all youngling fishes are returned -/
def ReturnedYounglings : Type := Bool

/-- Calculates the number of fishes brought home -/
def fishesBroughtHome (caught : List FishesCaught) (returned : ReturnedYounglings) : ℕ :=
  sorry

/-- Theorem: If all youngling fishes are returned, no fishes are brought home -/
theorem no_fishes_brought_home (caught : List FishesCaught) :
  fishesBroughtHome caught true = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_fishes_brought_home_l561_56134


namespace NUMINAMATH_CALUDE_starting_lineup_count_l561_56170

/-- Represents a football team with its composition and eligibility rules. -/
structure FootballTeam where
  totalMembers : ℕ
  offensiveLinemenEligible : ℕ
  tightEndEligible : ℕ
  
/-- Calculates the number of ways to choose a starting lineup for a given football team. -/
def chooseStartingLineup (team : FootballTeam) : ℕ :=
  team.offensiveLinemenEligible * 
  team.tightEndEligible * 
  (team.totalMembers - 2) * 
  (team.totalMembers - 3) * 
  (team.totalMembers - 4)

/-- Theorem stating that for the given team composition, there are 5760 ways to choose a starting lineup. -/
theorem starting_lineup_count : 
  chooseStartingLineup ⟨12, 4, 2⟩ = 5760 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l561_56170


namespace NUMINAMATH_CALUDE_set_union_problem_l561_56164

theorem set_union_problem (A B : Set ℕ) (a : ℕ) :
  A = {1, 2, 3} →
  B = {2, a} →
  A ∪ B = {0, 1, 2, 3} →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l561_56164


namespace NUMINAMATH_CALUDE_polynomial_equality_l561_56118

theorem polynomial_equality : 
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 10406040101 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l561_56118


namespace NUMINAMATH_CALUDE_optimal_plan_l561_56122

/-- Represents the cost and quantity of new energy vehicles --/
structure VehiclePlan where
  costA : ℝ  -- Cost of A-type car in million yuan
  costB : ℝ  -- Cost of B-type car in million yuan
  quantA : ℕ -- Quantity of A-type cars
  quantB : ℕ -- Quantity of B-type cars

/-- Conditions for the vehicle purchase plan --/
def satisfiesConditions (plan : VehiclePlan) : Prop :=
  3 * plan.costA + plan.costB = 85 ∧
  2 * plan.costA + 4 * plan.costB = 140 ∧
  plan.quantA + plan.quantB = 15 ∧
  plan.quantA ≤ 2 * plan.quantB

/-- Total cost of the vehicle purchase plan --/
def totalCost (plan : VehiclePlan) : ℝ :=
  plan.costA * plan.quantA + plan.costB * plan.quantB

/-- Theorem stating the most cost-effective plan --/
theorem optimal_plan :
  ∃ (plan : VehiclePlan),
    satisfiesConditions plan ∧
    plan.costA = 20 ∧
    plan.costB = 25 ∧
    plan.quantA = 10 ∧
    plan.quantB = 5 ∧
    totalCost plan = 325 ∧
    (∀ (otherPlan : VehiclePlan),
      satisfiesConditions otherPlan →
      totalCost otherPlan ≥ totalCost plan) :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_plan_l561_56122


namespace NUMINAMATH_CALUDE_net_population_increase_l561_56199

/-- The net population increase in one day given specific birth and death rates -/
theorem net_population_increase (birth_rate : ℕ) (death_rate : ℕ) (seconds_per_interval : ℕ) (seconds_per_day : ℕ) :
  birth_rate = 4 →
  death_rate = 2 →
  seconds_per_interval = 2 →
  seconds_per_day = 86400 →
  (birth_rate - death_rate) * (seconds_per_day / seconds_per_interval) = 86400 := by
  sorry

#check net_population_increase

end NUMINAMATH_CALUDE_net_population_increase_l561_56199


namespace NUMINAMATH_CALUDE_meaningful_expression_l561_56124

theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x = (Real.sqrt (a + 3)) / (a - 1)) ↔ (a ≥ -3 ∧ a ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l561_56124


namespace NUMINAMATH_CALUDE_not_yellow_houses_l561_56142

/-- Represents the number of houses Isabella has of each color --/
structure Houses where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Conditions for Isabella's houses --/
def isabellas_houses (h : Houses) : Prop :=
  h.green = 3 * h.yellow ∧
  h.yellow = h.red - 40 ∧
  h.green = 90

/-- Theorem stating the number of houses that are not yellow --/
theorem not_yellow_houses (h : Houses) (hcond : isabellas_houses h) :
  h.green + h.red = 160 :=
sorry

end NUMINAMATH_CALUDE_not_yellow_houses_l561_56142


namespace NUMINAMATH_CALUDE_proposition_logic_l561_56195

theorem proposition_logic (p q : Prop) (hp : p = (2 + 2 = 5)) (hq : q = (3 > 2)) :
  (p ∨ q) ∧ ¬(¬q) := by sorry

end NUMINAMATH_CALUDE_proposition_logic_l561_56195


namespace NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l561_56168

theorem cubic_polynomial_uniqueness (p q r : ℝ) (Q : ℝ → ℝ) :
  (p^3 + 4*p^2 + 6*p + 8 = 0) →
  (q^3 + 4*q^2 + 6*q + 8 = 0) →
  (r^3 + 4*r^2 + 6*r + 8 = 0) →
  (∃ a b c d : ℝ, ∀ x, Q x = a*x^3 + b*x^2 + c*x + d) →
  (Q p = q + r) →
  (Q q = p + r) →
  (Q r = p + q) →
  (Q (p + q + r) = -20) →
  (∀ x, Q x = 5/4*x^3 + 4*x^2 + 23/4*x + 6) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l561_56168


namespace NUMINAMATH_CALUDE_first_five_pages_drawings_l561_56176

def drawings_on_page (page : Nat) : Nat :=
  5 * 2^(page - 1)

def total_drawings (n : Nat) : Nat :=
  (List.range n).map drawings_on_page |>.sum

theorem first_five_pages_drawings : total_drawings 5 = 155 := by
  sorry

end NUMINAMATH_CALUDE_first_five_pages_drawings_l561_56176


namespace NUMINAMATH_CALUDE_sum_even_product_not_necessarily_odd_l561_56149

theorem sum_even_product_not_necessarily_odd :
  ∃ (a b : ℤ), Even (a + b) ∧ ¬Odd (a * b) := by sorry

end NUMINAMATH_CALUDE_sum_even_product_not_necessarily_odd_l561_56149


namespace NUMINAMATH_CALUDE_matches_needed_for_new_win_rate_l561_56131

/-- Given a player who has won 19 out of 20 matches, prove that they need to win 5 more matches
    without any losses to achieve a 96% winning rate. -/
theorem matches_needed_for_new_win_rate
  (initial_matches : Nat)
  (initial_wins : Nat)
  (target_win_rate : Rat)
  (h1 : initial_matches = 20)
  (h2 : initial_wins = 19)
  (h3 : target_win_rate = 24/25) :
  ∃ (additional_wins : Nat),
    additional_wins = 5 ∧
    (initial_wins + additional_wins : Rat) / (initial_matches + additional_wins) = target_win_rate :=
by sorry

end NUMINAMATH_CALUDE_matches_needed_for_new_win_rate_l561_56131


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l561_56140

theorem quadratic_form_k_value (a h k : ℚ) : 
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l561_56140


namespace NUMINAMATH_CALUDE_frank_can_buy_seven_candies_l561_56125

-- Define the given conditions
def whack_a_mole_tickets : ℕ := 33
def skee_ball_tickets : ℕ := 9
def candy_cost : ℕ := 6

-- Define the total number of tickets
def total_tickets : ℕ := whack_a_mole_tickets + skee_ball_tickets

-- Define the number of candies Frank can buy
def candies_bought : ℕ := total_tickets / candy_cost

-- Theorem statement
theorem frank_can_buy_seven_candies : candies_bought = 7 := by
  sorry

end NUMINAMATH_CALUDE_frank_can_buy_seven_candies_l561_56125


namespace NUMINAMATH_CALUDE_geometric_harmonic_mean_inequality_l561_56116

theorem geometric_harmonic_mean_inequality {a b : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt (a * b) ≥ 2 / (1 / a + 1 / b) ∧
  (Real.sqrt (a * b) = 2 / (1 / a + 1 / b) ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_geometric_harmonic_mean_inequality_l561_56116


namespace NUMINAMATH_CALUDE_crayon_difference_l561_56130

def karen_crayons : ℕ := 639
def cindy_crayons : ℕ := 504
def peter_crayons : ℕ := 752
def rachel_crayons : ℕ := 315

theorem crayon_difference :
  (max karen_crayons (max cindy_crayons (max peter_crayons rachel_crayons))) -
  (min karen_crayons (min cindy_crayons (min peter_crayons rachel_crayons))) = 437 := by
  sorry

end NUMINAMATH_CALUDE_crayon_difference_l561_56130


namespace NUMINAMATH_CALUDE_wedge_volume_l561_56123

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d : ℝ) (θ : ℝ) (h : d = 16 ∧ θ = 30 * π / 180) : 
  let r := d / 2
  let v := (r^2 * d * π) / 4
  v = 256 * π := by sorry

end NUMINAMATH_CALUDE_wedge_volume_l561_56123


namespace NUMINAMATH_CALUDE_profit_per_meter_l561_56183

/-- Profit per meter calculation -/
theorem profit_per_meter
  (length : ℕ)
  (selling_price : ℕ)
  (total_profit : ℕ)
  (h1 : length = 40)
  (h2 : selling_price = 8200)
  (h3 : total_profit = 1000) :
  total_profit / length = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_profit_per_meter_l561_56183


namespace NUMINAMATH_CALUDE_count_integer_solutions_l561_56114

theorem count_integer_solutions : ∃! (s : Finset (ℕ × ℕ)), 
  (∀ (p : ℕ × ℕ), p ∈ s ↔ p.1 > 0 ∧ p.2 > 0 ∧ 8 / p.1 + 6 / p.2 = 1) ∧ 
  s.card = 5 := by
sorry

end NUMINAMATH_CALUDE_count_integer_solutions_l561_56114


namespace NUMINAMATH_CALUDE_smallest_perfect_square_multiplier_l561_56128

theorem smallest_perfect_square_multiplier : ∃ (n : ℕ), 
  (7 * n = 7 * 7) ∧ 
  (∃ (m : ℕ), m * m = 7 * n) ∧
  (∀ (k : ℕ), k < 7 → ¬∃ (m : ℕ), m * m = k * n) := by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_multiplier_l561_56128


namespace NUMINAMATH_CALUDE_days_to_read_book_290_l561_56120

/-- Calculates the number of days required to read a book with the given reading pattern -/
def daysToReadBook (totalPages : ℕ) (sundayPages : ℕ) (otherDayPages : ℕ) : ℕ :=
  let pagesPerWeek := sundayPages + 6 * otherDayPages
  let completeWeeks := totalPages / pagesPerWeek
  let remainingPages := totalPages % pagesPerWeek
  let additionalDays := 
    if remainingPages ≤ sundayPages 
    then 1
    else 1 + ((remainingPages - sundayPages) + (otherDayPages - 1)) / otherDayPages
  7 * completeWeeks + additionalDays

/-- Theorem stating that it takes 41 days to read a 290-page book with the given reading pattern -/
theorem days_to_read_book_290 : 
  daysToReadBook 290 25 4 = 41 := by
  sorry

end NUMINAMATH_CALUDE_days_to_read_book_290_l561_56120


namespace NUMINAMATH_CALUDE_twenty_team_tournament_games_l561_56117

/-- Calculates the number of games in a single-elimination tournament. -/
def gamesInTournament (n : ℕ) : ℕ := n - 1

/-- Theorem: A single-elimination tournament with 20 teams requires 19 games to determine the winner. -/
theorem twenty_team_tournament_games :
  gamesInTournament 20 = 19 := by
  sorry

end NUMINAMATH_CALUDE_twenty_team_tournament_games_l561_56117


namespace NUMINAMATH_CALUDE_motorboat_travel_time_l561_56136

/-- The time taken for a motorboat to travel from pier X to pier Y downstream,
    given the conditions of the river journey problem. -/
theorem motorboat_travel_time (s r : ℝ) (h₁ : s > 0) (h₂ : r > 0) (h₃ : s > r) : 
  ∃ t : ℝ, t = (12 * (s - r)) / (s + r) ∧ 
    (s + r) * t + (s - r) * (12 - t) = 12 * r := by
  sorry

end NUMINAMATH_CALUDE_motorboat_travel_time_l561_56136


namespace NUMINAMATH_CALUDE_vector_simplification_l561_56154

variable (V : Type*) [AddCommGroup V]
variable (A B C D : V)

theorem vector_simplification (h : A + (B - A) + (C - B) = C) :
  (B - A) + (C - B) - (C - A) - (D - C) = C - D :=
sorry

end NUMINAMATH_CALUDE_vector_simplification_l561_56154


namespace NUMINAMATH_CALUDE_distance_PQ_is_25_l561_56133

/-- The distance between point P and the intersection point Q of lines l₁ and l₂ is 25. -/
theorem distance_PQ_is_25 
  (P : ℝ × ℝ)
  (l₁ : Set (ℝ × ℝ))
  (l₂ : Set (ℝ × ℝ))
  (Q : ℝ × ℝ)
  (h₁ : P = (3, 2))
  (h₂ : ∀ (x y : ℝ), (x, y) ∈ l₁ ↔ ∃ t, x = 3 + 4/5 * t ∧ y = 2 + 3/5 * t)
  (h₃ : ∀ (x y : ℝ), (x, y) ∈ l₂ ↔ x - 2*y + 11 = 0)
  (h₄ : Q ∈ l₁ ∧ Q ∈ l₂) :
  dist P Q = 25 := by
  sorry

#check distance_PQ_is_25

end NUMINAMATH_CALUDE_distance_PQ_is_25_l561_56133


namespace NUMINAMATH_CALUDE_min_value_inequality_l561_56148

def f (k : ℝ) (x : ℝ) : ℝ := |x - 1| + |x + k|

theorem min_value_inequality (k : ℝ) (a b c : ℝ) 
  (h1 : k > 0)
  (h2 : ∀ x, f k x ≥ 3)
  (h3 : ∃ x, f k x = 3)
  (h4 : a + b + c = k) :
  a^2 + b^2 + c^2 ≥ 4/3 := by
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l561_56148


namespace NUMINAMATH_CALUDE_investment_ratio_l561_56173

/-- Given two investors P and Q, with P investing 50000 and profits divided in ratio 3:4, 
    prove that Q's investment is 66666.67 -/
theorem investment_ratio (p q : ℝ) (h1 : p = 50000) (h2 : p / q = 3 / 4) : 
  q = 66666.67 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l561_56173


namespace NUMINAMATH_CALUDE_square_problem_l561_56161

/-- Square with side length 1200 -/
structure Square :=
  (side : ℝ)
  (is_1200 : side = 1200)

/-- Point on the side AB of the square -/
structure PointOnAB (S : Square) :=
  (x : ℝ)
  (on_side : 0 ≤ x ∧ x ≤ S.side)

theorem square_problem (S : Square) (G H : PointOnAB S)
  (h_order : G.x < H.x)
  (h_angle : Real.cos (Real.pi / 3) = (H.x - G.x) / 600)
  (h_dist : H.x - G.x = 600) :
  S.side - H.x = 300 + 100 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_square_problem_l561_56161


namespace NUMINAMATH_CALUDE_saline_solution_water_calculation_l561_56109

/-- Given a saline solution mixture, calculate the amount of water needed for a larger volume -/
theorem saline_solution_water_calculation 
  (salt_solution : ℝ) 
  (initial_water : ℝ) 
  (initial_total : ℝ) 
  (final_volume : ℝ) 
  (h1 : salt_solution = 0.05)
  (h2 : initial_water = 0.03)
  (h3 : initial_total = salt_solution + initial_water)
  (h4 : final_volume = 0.64) :
  final_volume * (initial_water / initial_total) = 0.24 := by
sorry

end NUMINAMATH_CALUDE_saline_solution_water_calculation_l561_56109


namespace NUMINAMATH_CALUDE_car_journey_distance_l561_56146

/-- Proves that given a car that travels a certain distance in 9 hours for the forward journey,
    and returns with a speed increased by 20 km/hr in 6 hours, the distance traveled is 360 km. -/
theorem car_journey_distance : ∀ (v : ℝ),
  v * 9 = (v + 20) * 6 →
  v * 9 = 360 :=
by
  sorry

end NUMINAMATH_CALUDE_car_journey_distance_l561_56146


namespace NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l561_56158

theorem floor_plus_self_unique_solution :
  ∃! r : ℝ, ⌊r⌋ + r = 14.5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_unique_solution_l561_56158


namespace NUMINAMATH_CALUDE_equation_solution_l561_56151

theorem equation_solution (x : ℝ) : 
  (8 / (Real.sqrt (x - 10) - 10) + 2 / (Real.sqrt (x - 10) - 5) + 
   10 / (Real.sqrt (x - 10) + 5) + 16 / (Real.sqrt (x - 10) + 10) = 0) ↔ 
  x = 60 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l561_56151


namespace NUMINAMATH_CALUDE_union_complement_problem_l561_56150

universe u

def I : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem union_complement_problem : A ∪ (I \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_complement_problem_l561_56150


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l561_56181

def p (x : ℝ) : ℝ := 5 * x^3 - 3 * x^2 + 9 * x - 2
def q (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

theorem coefficient_of_x_squared :
  ∃ (a b c d : ℝ), p x * q x = a * x^4 + b * x^3 - 48 * x^2 + c * x + d :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l561_56181


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l561_56187

/-- Given a geometric sequence {a_n} with common ratio q = 1/2 and sum of first n terms S_n, 
    prove that S_3 / a_3 = 7 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = (1 / 2) * a n) →  -- Geometric sequence with common ratio 1/2
  (∀ n, S n = a 1 * (1 - (1 / 2)^n) / (1 - (1 / 2))) →  -- Sum formula
  S 3 / a 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l561_56187


namespace NUMINAMATH_CALUDE_largest_square_multiple_18_under_500_l561_56174

theorem largest_square_multiple_18_under_500 : ∃ n : ℕ, 
  n^2 = 324 ∧ 
  18 ∣ n^2 ∧ 
  n^2 < 500 ∧ 
  ∀ m : ℕ, (m^2 > n^2 ∧ 18 ∣ m^2) → m^2 ≥ 500 :=
by sorry

end NUMINAMATH_CALUDE_largest_square_multiple_18_under_500_l561_56174


namespace NUMINAMATH_CALUDE_min_value_inverse_sum_l561_56171

theorem min_value_inverse_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 3 * b = 1) :
  1 / a + 3 / b ≥ 16 ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ 1 / a + 3 / b = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inverse_sum_l561_56171


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l561_56172

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  equation : (ℝ × ℝ) → Prop

-- Define our specific circle
def myCircle : Circle :=
  { center := (2, -1),
    equation := fun (x, y) => (x - 2)^2 + (y + 1)^2 = 3 }

-- Theorem statement
theorem circle_center_coordinates :
  myCircle.center = (2, -1) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l561_56172
