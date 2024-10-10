import Mathlib

namespace no_factors_l50_5015

/-- The main polynomial -/
def p (x : ℝ) : ℝ := x^4 - 4*x^2 + 16

/-- Potential factor 1 -/
def f1 (x : ℝ) : ℝ := x^2 + 4

/-- Potential factor 2 -/
def f2 (x : ℝ) : ℝ := x - 2

/-- Potential factor 3 -/
def f3 (x : ℝ) : ℝ := x^2 - 4

/-- Potential factor 4 -/
def f4 (x : ℝ) : ℝ := x^2 + 2*x + 4

/-- Theorem stating that none of the given polynomials are factors of p -/
theorem no_factors : 
  (∀ x, p x ≠ 0 → f1 x ≠ 0) ∧ 
  (∀ x, p x ≠ 0 → f2 x ≠ 0) ∧ 
  (∀ x, p x ≠ 0 → f3 x ≠ 0) ∧ 
  (∀ x, p x ≠ 0 → f4 x ≠ 0) := by
  sorry

end no_factors_l50_5015


namespace last_three_digits_of_7_to_103_l50_5065

theorem last_three_digits_of_7_to_103 : 7^103 % 1000 = 60 := by
  sorry

end last_three_digits_of_7_to_103_l50_5065


namespace parabola_directrix_l50_5003

/-- The equation of the directrix of the parabola y = x^2 is y = -1/4 -/
theorem parabola_directrix : ∃ (k : ℝ), k = -1/4 ∧
  ∀ (x y : ℝ), y = x^2 → (x = 0 ∨ (x^2 + (y - k)^2) / (2 * (y - k)) = k) :=
by sorry

end parabola_directrix_l50_5003


namespace horner_v₁_value_l50_5030

def horner_polynomial (x : ℝ) : ℝ := 12 + 3*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def v₀ : ℝ := 3

def a_n_minus_1 : ℝ := 5

def x : ℝ := -4

def v₁ : ℝ := v₀ * x + a_n_minus_1

theorem horner_v₁_value : v₁ = -7 := by sorry

end horner_v₁_value_l50_5030


namespace max_sin_A_in_triangle_l50_5090

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi

-- Define the theorem
theorem max_sin_A_in_triangle (t : Triangle) 
  (h : Real.tan t.A / Real.tan t.B + Real.tan t.A / Real.tan t.C = 3) :
  Real.sin t.A ≤ Real.sqrt 21 / 5 := by
  sorry

end max_sin_A_in_triangle_l50_5090


namespace smallest_even_triangle_perimeter_l50_5074

/-- Represents a triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  x : ℕ
  is_even : Even x

/-- The perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ := t.x + (t.x + 2) + (t.x + 4)

/-- Checks if the triangle inequality holds for an EvenTriangle -/
def satisfies_triangle_inequality (t : EvenTriangle) : Prop :=
  t.x + (t.x + 2) > t.x + 4 ∧
  t.x + (t.x + 4) > t.x + 2 ∧
  (t.x + 2) + (t.x + 4) > t.x

/-- The smallest possible perimeter of a valid EvenTriangle is 18 -/
theorem smallest_even_triangle_perimeter :
  ∃ (t : EvenTriangle), satisfies_triangle_inequality t ∧
    perimeter t = 18 ∧
    ∀ (t' : EvenTriangle), satisfies_triangle_inequality t' → perimeter t' ≥ 18 := by
  sorry

end smallest_even_triangle_perimeter_l50_5074


namespace ribbon_cutting_theorem_l50_5025

/-- Represents the cutting time for a pair of centimeters -/
structure CutTimePair :=
  (first : Nat)
  (second : Nat)

/-- Calculates the total cutting time for the ribbon -/
def totalCutTime (ribbonLength : Nat) (cutTimePair : CutTimePair) : Nat :=
  (ribbonLength / 2) * (cutTimePair.first + cutTimePair.second)

/-- Calculates the length of ribbon cut in half the total time -/
def ribbonCutInHalfTime (ribbonLength : Nat) (cutTimePair : CutTimePair) : Nat :=
  ((totalCutTime ribbonLength cutTimePair) / 2) / (cutTimePair.first + cutTimePair.second) * 2

theorem ribbon_cutting_theorem (ribbonLength : Nat) (cutTimePair : CutTimePair) :
  ribbonLength = 200 →
  cutTimePair = { first := 35, second := 40 } →
  totalCutTime ribbonLength cutTimePair = 3750 ∧
  ribbonLength - ribbonCutInHalfTime ribbonLength cutTimePair = 150 :=
by sorry

#eval totalCutTime 200 { first := 35, second := 40 }
#eval 200 - ribbonCutInHalfTime 200 { first := 35, second := 40 }

end ribbon_cutting_theorem_l50_5025


namespace inverse_proportion_problem_l50_5016

/-- Given inversely proportional variables x and y, if x + y = 30 and x - y = 10,
    then y = 200/7 when x = 7. -/
theorem inverse_proportion_problem (x y : ℝ) (D : ℝ) (h1 : x * y = D)
    (h2 : x + y = 30) (h3 : x - y = 10) :
  (x = 7) → (y = 200 / 7) := by
  sorry

end inverse_proportion_problem_l50_5016


namespace function_properties_l50_5064

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivatives of f and g
variable (f' g' : ℝ → ℝ)

-- State the given conditions
variable (h1 : ∀ x, f (x + 3) = g (-x) + 2)
variable (h2 : ∀ x, f' (x - 1) = g' x)
variable (h3 : ∀ x, g (-x + 1) = -g (x + 1))

-- State the properties to be proved
theorem function_properties :
  (g 1 = 0) ∧
  (∀ x, g' (x + 1) = -g' (3 - x)) ∧
  (∀ x, g (x + 1) = g (3 - x)) ∧
  (∀ x, g (x + 4) = g x) :=
sorry

end function_properties_l50_5064


namespace red_to_blue_bead_ratio_l50_5051

theorem red_to_blue_bead_ratio :
  let red_beads : ℕ := 30
  let blue_beads : ℕ := 20
  (red_beads : ℚ) / blue_beads = 3 / 2 := by
  sorry

end red_to_blue_bead_ratio_l50_5051


namespace proposition_logic_l50_5001

theorem proposition_logic (p q : Prop) 
  (h_p_false : ¬p) 
  (h_q_true : q) : 
  (¬(p ∧ q)) ∧ 
  (p ∨ q) ∧ 
  (¬p) ∧ 
  (¬(¬q)) := by
  sorry

end proposition_logic_l50_5001


namespace utensils_per_pack_l50_5069

/-- Given that packs have an equal number of knives, forks, and spoons,
    and 5 packs contain 50 spoons, prove that each pack contains 30 utensils. -/
theorem utensils_per_pack (total_packs : ℕ) (total_spoons : ℕ) 
  (h1 : total_packs = 5)
  (h2 : total_spoons = 50) :
  let spoons_per_pack := total_spoons / total_packs
  let utensils_per_pack := 3 * spoons_per_pack
  utensils_per_pack = 30 := by
sorry

end utensils_per_pack_l50_5069


namespace smallest_sum_consecutive_integers_l50_5096

theorem smallest_sum_consecutive_integers : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (l : ℤ), n = (9 : ℤ) * (l + 4)) ∧ 
  (∃ (m : ℤ), n = (5 : ℤ) * (2 * m + 9)) ∧ 
  (∃ (k : ℤ), n = (11 : ℤ) * (k + 5)) ∧ 
  (∀ (n' : ℕ), n' > 0 → 
    (∃ (l : ℤ), n' = (9 : ℤ) * (l + 4)) → 
    (∃ (m : ℤ), n' = (5 : ℤ) * (2 * m + 9)) → 
    (∃ (k : ℤ), n' = (11 : ℤ) * (k + 5)) → 
    n ≤ n') ∧ 
  n = 495 :=
sorry

end smallest_sum_consecutive_integers_l50_5096


namespace impurity_reduction_proof_l50_5080

/-- Represents the reduction factor of impurities after each filtration -/
def reduction_factor : ℝ := 0.8

/-- Represents the target impurity level as a fraction of the original -/
def target_impurity : ℝ := 0.05

/-- The minimum number of filtrations required to reduce impurities below the target level -/
def min_filtrations : ℕ := 14

theorem impurity_reduction_proof :
  (reduction_factor ^ min_filtrations : ℝ) < target_impurity ∧
  ∀ n : ℕ, n < min_filtrations → (reduction_factor ^ n : ℝ) ≥ target_impurity :=
sorry

end impurity_reduction_proof_l50_5080


namespace trajectory_and_intersection_l50_5095

/-- The trajectory C of point P in the Cartesian coordinate system xOy,
    where the sum of distances from P to (0, -√3) and (0, √3) equals 4 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 / 4 + p.1^2 = 1}

/-- The line that intersects C -/
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 1}

/-- Theorem stating the properties of the trajectory C and its intersection with the line -/
theorem trajectory_and_intersection :
  ∀ k : ℝ,
  (∀ p : ℝ × ℝ, p ∈ C → (Real.sqrt ((p.1)^2 + (p.2 + Real.sqrt 3)^2) +
                         Real.sqrt ((p.1)^2 + (p.2 - Real.sqrt 3)^2) = 4)) ∧
  (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ A ∈ line k ∧ B ∈ line k ∧
    (k = 1/2 ∨ k = -1/2) ↔ (A.1 * B.1 + A.2 * B.2 = 0)) := by
  sorry

end trajectory_and_intersection_l50_5095


namespace handshakes_eight_people_l50_5022

/-- The number of handshakes in a group where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 8 people, where each person shakes hands with every other person exactly once, the total number of handshakes is 28. -/
theorem handshakes_eight_people : handshakes 8 = 28 := by
  sorry

end handshakes_eight_people_l50_5022


namespace day_1500_is_sunday_l50_5004

/-- Given that the first day is a Friday, prove that the 1500th day is a Sunday -/
theorem day_1500_is_sunday (first_day : Nat) (h : first_day % 7 = 5) : 
  (first_day + 1499) % 7 = 0 := by
  sorry

#check day_1500_is_sunday

end day_1500_is_sunday_l50_5004


namespace workout_ratio_theorem_l50_5092

/-- Represents the workout schedule for Rayman, Junior, and Wolverine -/
structure WorkoutSchedule where
  junior_hours : ℝ
  rayman_hours : ℝ
  wolverine_hours : ℝ
  ratio : ℝ

/-- Theorem stating the relationship between workout hours -/
theorem workout_ratio_theorem (w : WorkoutSchedule) 
  (h1 : w.rayman_hours = w.junior_hours / 2)
  (h2 : w.wolverine_hours = 60)
  (h3 : w.wolverine_hours = w.ratio * (w.rayman_hours + w.junior_hours)) :
  w.ratio = 40 / w.junior_hours :=
sorry

end workout_ratio_theorem_l50_5092


namespace sqrt_neg_four_squared_l50_5029

theorem sqrt_neg_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end sqrt_neg_four_squared_l50_5029


namespace hexagon_vertex_recovery_erased_vertex_recoverable_l50_5077

/-- Represents a hexagon with numbers on its vertices -/
structure Hexagon where
  -- Vertex numbers
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Theorem: Any vertex number in a hexagon can be determined from the other five -/
theorem hexagon_vertex_recovery (h : Hexagon) :
  h.a = h.b + h.d + h.f - h.c - h.e :=
by sorry

/-- Corollary: It's possible to recover an erased vertex number in the hexagon -/
theorem erased_vertex_recoverable (h : Hexagon) :
  ∃ (x : ℝ), x = h.b + h.d + h.f - h.c - h.e :=
by sorry

end hexagon_vertex_recovery_erased_vertex_recoverable_l50_5077


namespace smallest_solution_abs_equation_l50_5033

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x^2 * |x| = 3*x + 4 ∧ 
  ∀ (y : ℝ), y^2 * |y| = 3*y + 4 → x ≤ y :=
by sorry

end smallest_solution_abs_equation_l50_5033


namespace same_color_probability_value_l50_5037

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 5

def same_color_probability : ℚ :=
  (Nat.choose white_balls drawn_balls + Nat.choose black_balls drawn_balls) /
  Nat.choose total_balls drawn_balls

theorem same_color_probability_value :
  same_color_probability = 77 / 3003 := by sorry

end same_color_probability_value_l50_5037


namespace order_of_expressions_l50_5026

theorem order_of_expressions :
  let a : ℝ := (1/2)^(1/3)
  let b : ℝ := (1/3)^(1/2)
  let c : ℝ := Real.log (3/Real.pi)
  c < b ∧ b < a := by sorry

end order_of_expressions_l50_5026


namespace dot_product_equals_three_l50_5076

def vector_a : ℝ × ℝ := (2, -1)
def vector_b (x : ℝ) : ℝ × ℝ := (3, x)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem dot_product_equals_three (x : ℝ) :
  dot_product vector_a (vector_b x) = 3 → x = 3 := by
sorry

end dot_product_equals_three_l50_5076


namespace product_of_second_and_third_smallest_l50_5012

theorem product_of_second_and_third_smallest (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  (max a (max b c)) * (max (min a b) (min (max a b) c)) = 132 := by
  sorry

end product_of_second_and_third_smallest_l50_5012


namespace coupon_value_l50_5089

def total_price : ℕ := 67
def num_people : ℕ := 3
def individual_contribution : ℕ := 21

theorem coupon_value :
  total_price - (num_people * individual_contribution) = 4 := by
  sorry

end coupon_value_l50_5089


namespace jelly_bean_division_l50_5059

theorem jelly_bean_division (initial_amount : ℕ) (eaten_amount : ℕ) (num_piles : ℕ) :
  initial_amount = 72 →
  eaten_amount = 12 →
  num_piles = 5 →
  (initial_amount - eaten_amount) / num_piles = 12 :=
by
  sorry

end jelly_bean_division_l50_5059


namespace distinct_pairs_count_l50_5067

/-- Represents the colors of marbles --/
inductive Color
  | Red
  | Green
  | Blue
  | Yellow

/-- Represents a marble with a color and quantity --/
structure Marble where
  color : Color
  quantity : Nat

/-- Calculates the number of distinct pairs of marbles that can be chosen --/
def countDistinctPairs (marbles : List Marble) : Nat :=
  sorry

/-- Theorem: Given the specific set of marbles, the number of distinct pairs is 7 --/
theorem distinct_pairs_count :
  let marbles : List Marble := [
    ⟨Color.Red, 1⟩,
    ⟨Color.Green, 1⟩,
    ⟨Color.Blue, 2⟩,
    ⟨Color.Yellow, 2⟩
  ]
  countDistinctPairs marbles = 7 := by
  sorry

end distinct_pairs_count_l50_5067


namespace exists_unreachable_number_l50_5014

/-- A function that returns true if a number is a 4-digit integer -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that counts the number of differing digits between two numbers -/
def digit_difference (n m : ℕ) : ℕ := sorry

/-- The theorem stating that there exists a 4-digit number that cannot be changed 
    into a multiple of 1992 by changing 3 of its digits -/
theorem exists_unreachable_number : 
  ∃ n : ℕ, is_four_digit n ∧ 
    ∀ m : ℕ, is_four_digit m → m % 1992 = 0 → digit_difference n m > 3 := by
  sorry

end exists_unreachable_number_l50_5014


namespace volume_surface_area_radius_relation_l50_5028

/-- A convex polyhedron with an inscribed sphere -/
class ConvexPolyhedron where
  /-- The volume of the polyhedron -/
  volume : ℝ
  /-- The surface area of the polyhedron -/
  surface_area : ℝ
  /-- The radius of the inscribed sphere -/
  inscribed_radius : ℝ

/-- The theorem stating the relationship between volume, surface area, and inscribed sphere radius -/
theorem volume_surface_area_radius_relation (P : ConvexPolyhedron) : 
  P.volume = (1 / 3) * P.surface_area * P.inscribed_radius :=
sorry

end volume_surface_area_radius_relation_l50_5028


namespace rational_equation_system_l50_5053

theorem rational_equation_system (x y z : ℚ) 
  (eq1 : x - y + 2 * z = 1)
  (eq2 : x + y + 4 * z = 3) : 
  x + 2 * y + 5 * z = 4 := by
sorry

end rational_equation_system_l50_5053


namespace circle_equation_proof_l50_5020

/-- A circle with center (h, k) and radius r is represented by the equation (x - h)² + (y - k)² = r² --/
def is_circle (h k r : ℝ) (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ (x - h)^2 + (y - k)^2 = r^2

/-- A circle is tangent to the x-axis if its distance from the x-axis equals its radius --/
def tangent_to_x_axis (h k r : ℝ) : Prop := k = r

theorem circle_equation_proof (x y : ℝ) :
  let h : ℝ := 2
  let k : ℝ := 1
  let f : ℝ → ℝ → Prop := λ x y ↦ (x - 2)^2 + (y - 1)^2 = 1
  is_circle h k 1 f ∧ tangent_to_x_axis h k 1 := by sorry

end circle_equation_proof_l50_5020


namespace log_properties_l50_5098

-- Define approximate values for log₁₀ 2 and log₁₀ 3
def log10_2 : ℝ := 0.3010
def log10_3 : ℝ := 0.4771

-- Define the properties to be proved
theorem log_properties :
  let log10_27 := 3 * log10_3
  let log10_100_div_9 := 2 - 2 * log10_3
  let log10_sqrt_10 := (1 : ℝ) / 2
  (log10_27 = 3 * log10_3) ∧
  (log10_100_div_9 = 2 - 2 * log10_3) ∧
  (log10_sqrt_10 = (1 : ℝ) / 2) := by
  sorry


end log_properties_l50_5098


namespace painting_wall_coverage_percentage_l50_5040

/-- Represents the dimensions of a rectangular painting -/
structure PaintingDimensions where
  length : ℚ
  width : ℚ

/-- Represents the dimensions of an irregular pentagonal wall -/
structure WallDimensions where
  side1 : ℚ
  side2 : ℚ
  side3 : ℚ
  side4 : ℚ
  side5 : ℚ
  height : ℚ

/-- Calculates the area of a rectangular painting -/
def paintingArea (p : PaintingDimensions) : ℚ :=
  p.length * p.width

/-- Calculates the approximate area of the irregular pentagonal wall -/
def wallArea (w : WallDimensions) : ℚ :=
  (w.side3 * w.height) / 2

/-- Calculates the percentage of the wall covered by the painting -/
def coveragePercentage (p : PaintingDimensions) (w : WallDimensions) : ℚ :=
  (paintingArea p / wallArea w) * 100

/-- Theorem stating that the painting covers approximately 39.21% of the wall -/
theorem painting_wall_coverage_percentage :
  let painting := PaintingDimensions.mk (13/4) (38/5)
  let wall := WallDimensions.mk 4 12 14 10 8 9
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ 
    abs (coveragePercentage painting wall - 39.21) < ε :=
sorry

end painting_wall_coverage_percentage_l50_5040


namespace triangular_number_difference_l50_5039

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The difference between the 2010th and 2008th triangular numbers is 4019 -/
theorem triangular_number_difference : 
  triangular_number 2010 - triangular_number 2008 = 4019 := by
  sorry

end triangular_number_difference_l50_5039


namespace sum_of_x_and_y_on_circle_l50_5094

theorem sum_of_x_and_y_on_circle (x y : ℝ) 
  (h : 4 * x^2 + 4 * y^2 = 40 * x - 24 * y + 64) : x + y = 2 := by
  sorry

end sum_of_x_and_y_on_circle_l50_5094


namespace total_cost_proof_l50_5075

def flower_cost : ℕ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2

theorem total_cost_proof :
  (roses_bought + daisies_bought) * flower_cost = 12 := by
  sorry

end total_cost_proof_l50_5075


namespace girls_in_class_l50_5046

theorem girls_in_class (boys girls : ℕ) : 
  girls = boys + 3 →
  girls + boys = 41 →
  girls = 22 := by
sorry

end girls_in_class_l50_5046


namespace tangent_line_triangle_area_l50_5062

/-- The area of the triangle formed by the tangent line to y = e^x at (2, e^2) and the coordinate axes -/
theorem tangent_line_triangle_area : 
  let f (x : ℝ) := Real.exp x
  let x₀ : ℝ := 2
  let y₀ : ℝ := Real.exp x₀
  let m : ℝ := Real.exp x₀  -- slope of the tangent line
  let b : ℝ := y₀ - m * x₀  -- y-intercept of the tangent line
  let x_intercept : ℝ := -b / m  -- x-intercept of the tangent line
  Real.exp 2 / 2 = (x_intercept * y₀) / 2 := by sorry

end tangent_line_triangle_area_l50_5062


namespace ruler_cost_l50_5031

theorem ruler_cost (total_spent : ℕ) (notebook_cost : ℕ) (num_pencils : ℕ) (pencil_cost : ℕ) 
  (h1 : total_spent = 74)
  (h2 : notebook_cost = 35)
  (h3 : num_pencils = 3)
  (h4 : pencil_cost = 7) :
  total_spent - (notebook_cost + num_pencils * pencil_cost) = 18 := by
  sorry

end ruler_cost_l50_5031


namespace meaningful_sqrt_fraction_range_l50_5086

theorem meaningful_sqrt_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = (Real.sqrt (4 - x)) / (Real.sqrt (x - 1))) ↔ (1 < x ∧ x ≤ 4) :=
sorry

end meaningful_sqrt_fraction_range_l50_5086


namespace track_length_l50_5060

theorem track_length : ∀ (x : ℝ), 
  (x > 0) →  -- track length is positive
  (120 / (x/2 - 120) = (x/2 + 50) / (3*x/2 - 170)) →  -- ratio of distances is constant
  x = 418 := by
sorry

end track_length_l50_5060


namespace parallel_vectors_magnitude_l50_5011

/-- Given two vectors a and b in ℝ², if a is parallel to b, then the magnitude of b is 2√5. -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, 2) → 
  b.1 = -2 → 
  ∃ (t : ℝ), a = t • b → 
  ‖b‖ = 2 * Real.sqrt 5 := by
sorry

end parallel_vectors_magnitude_l50_5011


namespace combined_figure_area_l50_5044

/-- The area of a figure consisting of a twelve-sided polygon and a rhombus -/
theorem combined_figure_area (polygon_area : ℝ) (rhombus_diagonal1 : ℝ) (rhombus_diagonal2 : ℝ) :
  polygon_area = 13 →
  rhombus_diagonal1 = 2 →
  rhombus_diagonal2 = 1 →
  polygon_area + (rhombus_diagonal1 * rhombus_diagonal2) / 2 = 14 :=
by sorry

end combined_figure_area_l50_5044


namespace cube_root_of_quarter_l50_5058

theorem cube_root_of_quarter (t s : ℝ) : t = 15 * s^3 ∧ t = 3.75 → s = 0.5 := by
  sorry

end cube_root_of_quarter_l50_5058


namespace sqrt_132_plus_46_sqrt_11_l50_5048

theorem sqrt_132_plus_46_sqrt_11 :
  ∃ (a b c : ℤ), 
    (132 + 46 * Real.sqrt 11 : ℝ).sqrt = a + b * Real.sqrt c ∧
    ¬ ∃ (d : ℤ), c = d * d ∧
    ∃ (e f : ℤ), c = e * f ∧ (∀ (g : ℤ), g * g ∣ e → g = 1 ∨ g = -1) ∧
                             (∀ (h : ℤ), h * h ∣ f → h = 1 ∨ h = -1) :=
sorry

end sqrt_132_plus_46_sqrt_11_l50_5048


namespace sqrt_expression_equals_one_l50_5041

theorem sqrt_expression_equals_one :
  (Real.sqrt 6 - Real.sqrt 2) / Real.sqrt 2 + |Real.sqrt 3 - 2| = 1 := by
  sorry

end sqrt_expression_equals_one_l50_5041


namespace smallest_w_l50_5083

theorem smallest_w (w : ℕ+) 
  (h1 : (2^5 : ℕ) ∣ (936 * w))
  (h2 : (3^3 : ℕ) ∣ (936 * w))
  (h3 : (10^2 : ℕ) ∣ (936 * w)) :
  w ≥ 900 ∧ ∃ (v : ℕ+), v = 900 ∧ 
    (2^5 : ℕ) ∣ (936 * v) ∧ 
    (3^3 : ℕ) ∣ (936 * v) ∧ 
    (10^2 : ℕ) ∣ (936 * v) :=
by sorry

end smallest_w_l50_5083


namespace sqrt_expression_equality_l50_5006

theorem sqrt_expression_equality : 
  Real.sqrt 8 - (1/3)⁻¹ / Real.sqrt 3 + (1 - Real.sqrt 2)^2 = 3 - Real.sqrt 3 := by
  sorry

end sqrt_expression_equality_l50_5006


namespace carols_peanuts_l50_5061

/-- Represents the number of peanuts Carol's father gave her -/
def peanuts_from_father (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem carols_peanuts : peanuts_from_father 2 7 = 5 := by
  sorry

end carols_peanuts_l50_5061


namespace water_bottle_drinking_time_l50_5049

/-- Proves that drinking a 2-liter bottle of water with 40 ml sips every 5 minutes takes 250 minutes -/
theorem water_bottle_drinking_time :
  let bottle_capacity_liters : ℝ := 2
  let ml_per_liter : ℝ := 1000
  let sip_volume_ml : ℝ := 40
  let minutes_per_sip : ℝ := 5
  
  bottle_capacity_liters * ml_per_liter / sip_volume_ml * minutes_per_sip = 250 := by
  sorry


end water_bottle_drinking_time_l50_5049


namespace function_transformation_l50_5052

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_transformation (x : ℝ) : 
  (∀ y, f (y + 1) = 3 * y + 2) → f x = 3 * x - 1 := by
  sorry

end function_transformation_l50_5052


namespace divisors_18_product_and_sum_l50_5005

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0)

theorem divisors_18_product_and_sum :
  (divisors 18).prod = 5832 ∧ (divisors 18).sum = 39 := by
  sorry

end divisors_18_product_and_sum_l50_5005


namespace job_completion_time_l50_5079

/-- The time taken for three workers to complete a job together, given their individual completion times -/
theorem job_completion_time (time_A time_B time_C : ℝ) 
  (hA : time_A = 7) 
  (hB : time_B = 10) 
  (hC : time_C = 12) : 
  1 / (1 / time_A + 1 / time_B + 1 / time_C) = 420 / 137 := by
  sorry

end job_completion_time_l50_5079


namespace cuboid_volume_l50_5002

/-- The volume of a cuboid with edges 2 cm, 5 cm, and 3 cm is 30 cubic centimeters. -/
theorem cuboid_volume : 
  ∀ (length width height : ℝ), 
    length = 2 → width = 5 → height = 3 → 
    length * width * height = 30 := by
  sorry

end cuboid_volume_l50_5002


namespace largest_n_is_correct_l50_5073

/-- The largest positive integer n for which the system of equations has integer solutions -/
def largest_n : ℕ := 3

/-- Predicate to check if a given n has integer solutions for the system of equations -/
def has_integer_solution (n : ℕ) : Prop :=
  ∃ x : ℤ, ∃ y : Fin n → ℤ,
    ∀ i j : Fin n, (x + i.val + 1)^2 + y i^2 = (x + j.val + 1)^2 + y j^2

/-- Theorem stating that largest_n is indeed the largest n with integer solutions -/
theorem largest_n_is_correct :
  (has_integer_solution largest_n) ∧
  (∀ m : ℕ, m > largest_n → ¬(has_integer_solution m)) :=
by sorry

end largest_n_is_correct_l50_5073


namespace intersection_M_N_l50_5010

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by sorry

end intersection_M_N_l50_5010


namespace no_eulerian_or_hamiltonian_path_l50_5038

/-- A graph representing the science museum layout. -/
structure MuseumGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  bipartite : Finset Nat × Finset Nat
  degree_three : Finset Nat

/-- Predicate for the existence of an Eulerian path in the graph. -/
def has_eulerian_path (g : MuseumGraph) : Prop :=
  ∃ (path : List (Nat × Nat)), path.Nodup ∧ path.length = g.edges.card

/-- Predicate for the existence of a Hamiltonian path in the graph. -/
def has_hamiltonian_path (g : MuseumGraph) : Prop :=
  ∃ (path : List Nat), path.Nodup ∧ path.length = g.vertices.card

/-- The main theorem stating the non-existence of Eulerian and Hamiltonian paths. -/
theorem no_eulerian_or_hamiltonian_path (g : MuseumGraph)
  (h1 : g.vertices.card = 19)
  (h2 : g.edges.card = 30)
  (h3 : g.bipartite.1.card = 7 ∧ g.bipartite.2.card = 12)
  (h4 : g.degree_three.card ≥ 6) :
  ¬(has_eulerian_path g) ∧ ¬(has_hamiltonian_path g) := by
  sorry

#check no_eulerian_or_hamiltonian_path

end no_eulerian_or_hamiltonian_path_l50_5038


namespace smallest_shift_l50_5085

open Real

theorem smallest_shift (n : ℝ) : n > 0 ∧ 
  (∀ x, cos (2 * π * x - π / 3) = sin (2 * π * (x - n) + π / 3)) → 
  n ≥ 1 / 12 :=
sorry

end smallest_shift_l50_5085


namespace numeric_methods_count_l50_5017

/-- The number of second-year students studying numeric methods -/
def numeric_methods_students : ℕ := 225

/-- The number of second-year students studying automatic control of airborne vehicles -/
def automatic_control_students : ℕ := 450

/-- The number of second-year students studying both subjects -/
def both_subjects_students : ℕ := 134

/-- The total number of students in the faculty -/
def total_students : ℕ := 676

/-- The approximate percentage of second-year students -/
def second_year_percentage : ℚ := 80 / 100

/-- The total number of second-year students -/
def total_second_year_students : ℕ := 541

theorem numeric_methods_count : 
  numeric_methods_students = 
    total_second_year_students + both_subjects_students - automatic_control_students :=
by sorry

end numeric_methods_count_l50_5017


namespace line_segment_length_l50_5019

theorem line_segment_length (volume : ℝ) (radius : ℝ) (length : ℝ) : 
  volume = 432 * Real.pi →
  radius = 4 →
  volume = (Real.pi * radius^2 * length) + (2 * (2/3) * Real.pi * radius^3) →
  length = 50/3 := by
sorry

end line_segment_length_l50_5019


namespace fraction_equality_l50_5072

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (a + 10 * b) / (b + 10 * a) = 2) : 
  a / b = 0.8 := by sorry

end fraction_equality_l50_5072


namespace s_of_one_eq_394_div_25_l50_5050

/-- Given functions t and s, prove that s(1) = 394/25 -/
theorem s_of_one_eq_394_div_25 
  (t : ℝ → ℝ)
  (s : ℝ → ℝ)
  (h1 : ∀ x, t x = 5 * x - 12)
  (h2 : ∀ x, s (t x) = x^2 + 5 * x - 4) :
  s 1 = 394 / 25 := by
  sorry

end s_of_one_eq_394_div_25_l50_5050


namespace quartic_equation_solution_l50_5091

theorem quartic_equation_solution :
  ∀ x : ℂ, x^4 - 16*x^2 + 256 = 0 ↔ x = 4 ∨ x = -4 :=
by sorry

end quartic_equation_solution_l50_5091


namespace optimal_pricing_and_profit_daily_profit_function_l50_5018

/-- Represents the daily profit function for a product --/
def daily_profit (x : ℝ) : ℝ := -3 * x^2 + 252 * x - 4860

/-- Represents the constraint on the selling price --/
def price_constraint (x : ℝ) : Prop := 30 ≤ x ∧ x ≤ 54

/-- The theorem stating the optimal selling price and maximum profit --/
theorem optimal_pricing_and_profit :
  ∃ (x : ℝ), price_constraint x ∧ 
    (∀ y, price_constraint y → daily_profit y ≤ daily_profit x) ∧
    x = 42 ∧ daily_profit x = 432 := by
  sorry

/-- The theorem stating the form of the daily profit function --/
theorem daily_profit_function (x : ℝ) :
  daily_profit x = (x - 30) * (162 - 3 * x) := by
  sorry

end optimal_pricing_and_profit_daily_profit_function_l50_5018


namespace manuscript_revisions_l50_5000

/-- The number of pages revised twice in a manuscript -/
def pages_revised_twice (total_pages : ℕ) (pages_revised_once : ℕ) (cost_first_typing : ℕ) (cost_revision : ℕ) (total_cost : ℕ) : ℕ :=
  let cost_all_first_typing := total_pages * cost_first_typing
  let cost_revisions_once := pages_revised_once * cost_revision
  let remaining_cost := total_cost - cost_all_first_typing - cost_revisions_once
  remaining_cost / (2 * cost_revision)

theorem manuscript_revisions (total_pages : ℕ) (pages_revised_once : ℕ) (cost_first_typing : ℕ) (cost_revision : ℕ) (total_cost : ℕ)
  (h1 : total_pages = 100)
  (h2 : pages_revised_once = 30)
  (h3 : cost_first_typing = 10)
  (h4 : cost_revision = 5)
  (h5 : total_cost = 1350) :
  pages_revised_twice total_pages pages_revised_once cost_first_typing cost_revision total_cost = 20 := by
  sorry

end manuscript_revisions_l50_5000


namespace volume_ratio_cubes_l50_5035

/-- Given two cubes with edge lengths in the ratio 3:1, if the volume of the smaller cube is 1 unit,
    then the volume of the larger cube is 27 units. -/
theorem volume_ratio_cubes (e : ℝ) (h1 : e > 0) (h2 : e^3 = 1) :
  (3*e)^3 = 27 := by
  sorry

end volume_ratio_cubes_l50_5035


namespace inscribed_sphere_radius_to_height_ratio_l50_5055

/-- A regular tetrahedron with an inscribed sphere -/
structure RegularTetrahedronWithInscribedSphere where
  /-- The height of the regular tetrahedron -/
  height : ℝ
  /-- The radius of the inscribed sphere -/
  sphereRadius : ℝ
  /-- The area of one face of the regular tetrahedron -/
  faceArea : ℝ
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The sphere radius is positive -/
  sphereRadius_pos : 0 < sphereRadius
  /-- The face area is positive -/
  faceArea_pos : 0 < faceArea
  /-- Volume relation between the tetrahedron and the four pyramids formed by the inscribed sphere -/
  volume_relation : 4 * (1/3 * faceArea * sphereRadius) = 1/3 * faceArea * height

/-- The ratio of the radius of the inscribed sphere to the height of the regular tetrahedron is 1/4 -/
theorem inscribed_sphere_radius_to_height_ratio 
  (t : RegularTetrahedronWithInscribedSphere) : t.sphereRadius = 1/4 * t.height := by
  sorry

end inscribed_sphere_radius_to_height_ratio_l50_5055


namespace glass_to_sand_ratio_l50_5082

/-- Represents the number of items in each container --/
structure BeachTreasures where
  bucket : ℕ  -- number of seashells in the bucket
  jar : ℕ     -- number of glass pieces in the jar
  bag : ℕ     -- number of sand dollars in the bag

/-- The conditions of Simon's beach treasure collection --/
def simons_treasures : BeachTreasures → Prop
  | t => t.bucket = 5 * t.jar ∧ 
         t.jar = t.bag ∧ 
         t.bag = 10 ∧ 
         t.bucket + t.jar + t.bag = 190

/-- The theorem stating the ratio of glass pieces to sand dollars --/
theorem glass_to_sand_ratio (t : BeachTreasures) 
  (h : simons_treasures t) : t.jar / t.bag = 3 := by
  sorry

end glass_to_sand_ratio_l50_5082


namespace perpendicular_vectors_l50_5023

/-- Given two vectors OA and OB in 2D space, where OA is perpendicular to AB, prove that m = 4 -/
theorem perpendicular_vectors (OA OB : ℝ × ℝ) (m : ℝ) : 
  OA = (-1, 2) → 
  OB = (3, m) → 
  OA.1 * (OB.1 - OA.1) + OA.2 * (OB.2 - OA.2) = 0 → 
  m = 4 := by
sorry

end perpendicular_vectors_l50_5023


namespace class_average_height_l50_5088

theorem class_average_height (total_girls : ℕ) (group1_girls : ℕ) (group2_girls : ℕ) 
  (group1_avg_height : ℝ) (group2_avg_height : ℝ) :
  total_girls = group1_girls + group2_girls →
  group1_girls = 30 →
  group2_girls = 10 →
  group1_avg_height = 160 →
  group2_avg_height = 156 →
  (group1_girls * group1_avg_height + group2_girls * group2_avg_height) / total_girls = 159 := by
sorry

end class_average_height_l50_5088


namespace arithmetic_sequence_sum_l50_5047

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 2 + a 5 + a 8 = 39) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 117 :=
by
  sorry

end arithmetic_sequence_sum_l50_5047


namespace exists_natural_not_in_five_gp_l50_5099

/-- A geometric progression with integer terms -/
structure GeometricProgression where
  first_term : ℤ
  common_ratio : ℤ
  common_ratio_nonzero : common_ratio ≠ 0

/-- The nth term of a geometric progression -/
def GeometricProgression.nth_term (gp : GeometricProgression) (n : ℕ) : ℤ :=
  gp.first_term * gp.common_ratio ^ n

/-- Theorem: There exists a natural number not in any of five given geometric progressions -/
theorem exists_natural_not_in_five_gp (gp1 gp2 gp3 gp4 gp5 : GeometricProgression) :
  ∃ (k : ℕ), (∀ n : ℕ, gp1.nth_term n ≠ k) ∧
             (∀ n : ℕ, gp2.nth_term n ≠ k) ∧
             (∀ n : ℕ, gp3.nth_term n ≠ k) ∧
             (∀ n : ℕ, gp4.nth_term n ≠ k) ∧
             (∀ n : ℕ, gp5.nth_term n ≠ k) :=
  sorry

end exists_natural_not_in_five_gp_l50_5099


namespace max_value_of_expression_l50_5007

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5*x + 6*y < 90) :
  ∃ (M : ℝ), M = 900 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → 5*a + 6*b < 90 → a*b*(90 - 5*a - 6*b) ≤ M :=
by sorry

end max_value_of_expression_l50_5007


namespace perfect_square_coefficient_l50_5087

theorem perfect_square_coefficient (x : ℝ) : ∃ (r s : ℝ), 
  (81/16) * x^2 + 18 * x + 16 = (r * x + s)^2 := by
  sorry

end perfect_square_coefficient_l50_5087


namespace sin_two_theta_value_l50_5009

/-- If e^(2iθ) = (2 + i√5) / 3, then sin 2θ = √3 / 3 -/
theorem sin_two_theta_value (θ : ℝ) (h : Complex.exp (2 * θ * Complex.I) = (2 + Complex.I * Real.sqrt 5) / 3) : 
  Real.sin (2 * θ) = Real.sqrt 3 / 3 := by
sorry

end sin_two_theta_value_l50_5009


namespace profit_percentage_l50_5042

theorem profit_percentage (C S : ℝ) (h : 315 * C = 250 * S) : 
  (S - C) / C * 100 = 26 := by
sorry

end profit_percentage_l50_5042


namespace area_at_stage_6_l50_5068

/-- The side length of each square in inches -/
def square_side : ℕ := 4

/-- The number of squares at a given stage -/
def num_squares (stage : ℕ) : ℕ := stage

/-- The area of the rectangle at a given stage in square inches -/
def rectangle_area (stage : ℕ) : ℕ :=
  (num_squares stage) * (square_side * square_side)

/-- Theorem: The area of the rectangle at Stage 6 is 96 square inches -/
theorem area_at_stage_6 : rectangle_area 6 = 96 := by
  sorry

end area_at_stage_6_l50_5068


namespace inequality_solutions_range_l50_5027

theorem inequality_solutions_range (a : ℝ) : 
  (∀ x : ℕ+, x < a ↔ x ≤ 5) → (5 < a ∧ a < 6) :=
by sorry

end inequality_solutions_range_l50_5027


namespace rectangle_areas_sum_l50_5054

theorem rectangle_areas_sum : 
  let width : ℝ := 2
  let lengths : List ℝ := [1, 8, 27]
  let areas : List ℝ := lengths.map (λ l => width * l)
  areas.sum = 72 := by
  sorry

end rectangle_areas_sum_l50_5054


namespace age_ratio_problem_l50_5034

/-- Given that Rahul's age after 6 years will be 26 and Deepak's current age is 15,
    prove that the ratio of their current ages is 4:3. -/
theorem age_ratio_problem (rahul_future_age : ℕ) (deepak_age : ℕ) : 
  rahul_future_age = 26 → 
  deepak_age = 15 → 
  (rahul_future_age - 6) / deepak_age = 4 / 3 := by
sorry

end age_ratio_problem_l50_5034


namespace range_of_p_l50_5078

/-- The function p(x) = x^4 + 6x^2 + 9 -/
def p (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

/-- The domain of the function -/
def domain : Set ℝ := { x | x ≥ 0 }

/-- The range of the function -/
def range : Set ℝ := { y | ∃ x ∈ domain, p x = y }

theorem range_of_p : range = { y | y ≥ 9 } := by sorry

end range_of_p_l50_5078


namespace missing_number_is_five_l50_5056

/-- Represents the sum of two adjacent children's favorite numbers -/
structure AdjacentSum :=
  (value : ℕ)

/-- Represents a circle of children with their favorite numbers -/
structure ChildrenCircle :=
  (size : ℕ)
  (sums : List AdjacentSum)

/-- Calculates the missing number in the circle -/
def calculateMissingNumber (circle : ChildrenCircle) : ℕ :=
  sorry

/-- Theorem stating that the missing number is 5 -/
theorem missing_number_is_five (circle : ChildrenCircle) 
  (h1 : circle.size = 6)
  (h2 : circle.sums = [⟨8⟩, ⟨14⟩, ⟨12⟩])
  : calculateMissingNumber circle = 5 := by
  sorry

end missing_number_is_five_l50_5056


namespace painter_problem_l50_5070

/-- Given a painting job with a total number of rooms, time per room, and rooms already painted,
    calculates the time needed to paint the remaining rooms. -/
def time_to_paint_remaining (total_rooms : ℕ) (time_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * time_per_room

/-- Proves that for the given scenario, the time to paint the remaining rooms is 49 hours. -/
theorem painter_problem :
  let total_rooms : ℕ := 12
  let time_per_room : ℕ := 7
  let painted_rooms : ℕ := 5
  time_to_paint_remaining total_rooms time_per_room painted_rooms = 49 := by
  sorry


end painter_problem_l50_5070


namespace largest_mu_inequality_l50_5013

theorem largest_mu_inequality :
  ∃ (μ : ℝ), μ = 3/4 ∧ 
  (∀ (a b c d : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
    a^2 + b^2 + c^2 + d^2 ≥ a*b + μ*(b*c + d*a) + c*d) ∧
  (∀ (μ' : ℝ), μ' > μ →
    ∃ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
      a^2 + b^2 + c^2 + d^2 < a*b + μ'*(b*c + d*a) + c*d) :=
by sorry

end largest_mu_inequality_l50_5013


namespace rearrangement_time_proof_l50_5071

/-- The number of hours required to write all rearrangements of a 12-letter name -/
def rearrangement_hours : ℕ := 798336

/-- The number of letters in the name -/
def name_length : ℕ := 12

/-- The number of arrangements written per minute -/
def arrangements_per_minute : ℕ := 10

/-- Theorem stating the time required to write all rearrangements -/
theorem rearrangement_time_proof :
  rearrangement_hours = (name_length.factorial / arrangements_per_minute) / 60 := by
  sorry


end rearrangement_time_proof_l50_5071


namespace subtraction_multiplication_problem_l50_5021

theorem subtraction_multiplication_problem : 
  let initial_value : ℚ := 555.55
  let subtracted_value : ℚ := 111.11
  let multiplier : ℚ := 2
  let result : ℚ := (initial_value - subtracted_value) * multiplier
  result = 888.88 := by sorry

end subtraction_multiplication_problem_l50_5021


namespace balloon_final_height_l50_5024

/-- Represents the sequence of balloon movements -/
def BalloonMovements : List Int := [6, -2, 3, -2]

/-- Calculates the final height of the balloon after a sequence of movements -/
def finalHeight (movements : List Int) : Int :=
  movements.foldl (· + ·) 0

/-- Theorem stating that the final height of the balloon is 5 meters -/
theorem balloon_final_height :
  finalHeight BalloonMovements = 5 := by
  sorry

end balloon_final_height_l50_5024


namespace cube_root_of_1331_l50_5084

theorem cube_root_of_1331 (y : ℝ) (h1 : y > 0) (h2 : y^3 = 1331) : y = 11 := by
  sorry

end cube_root_of_1331_l50_5084


namespace smallest_c_for_inverse_l50_5063

-- Define the function f
def f (x : ℝ) : ℝ := (x - 3)^2 + 4

-- State the theorem
theorem smallest_c_for_inverse : 
  ∀ c : ℝ, (∀ x y, x ≥ c → y ≥ c → f x = f y → x = y) ↔ c ≥ 3 :=
by sorry

end smallest_c_for_inverse_l50_5063


namespace system_solution_l50_5057

theorem system_solution :
  ∃ (x y z : ℝ), 
    (x + 2*y = 4) ∧ 
    (2*x + 5*y - 2*z = 11) ∧ 
    (3*x - 5*y + 2*z = -1) ∧
    (x = 2) ∧ (y = 1) ∧ (z = -1) := by
  sorry

end system_solution_l50_5057


namespace odd_function_composition_even_l50_5036

-- Define an odd function
def OddFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem statement
theorem odd_function_composition_even
  (g : ℝ → ℝ)
  (h : OddFunction g) :
  EvenFunction (fun x ↦ g (g (g (g x)))) :=
sorry

end odd_function_composition_even_l50_5036


namespace eliminate_denominators_l50_5097

theorem eliminate_denominators (x : ℝ) :
  (2*x - 1) / 3 - (3*x - 4) / 4 = 1 ↔ 4*(2*x - 1) - 3*(3*x - 4) = 12 :=
by sorry

end eliminate_denominators_l50_5097


namespace inequality_proof_l50_5093

theorem inequality_proof (a b c A B C u v : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hu : 0 < u) (hv : 0 < v)
  (h1 : a * u^2 - b * u + c ≤ 0)
  (h2 : A * v^2 - B * v + C ≤ 0) :
  (a * u + A * v) * (c / u + C / v) ≤ ((b + B) / 2)^2 := by
sorry

end inequality_proof_l50_5093


namespace isosceles_triangle_l50_5032

theorem isosceles_triangle (A B C : Real) (h : 2 * Real.sin A * Real.cos B = Real.sin C) :
  ∃ (s : Real), s > 0 ∧ Real.sin A = s ∧ Real.sin B = s := by
  sorry

end isosceles_triangle_l50_5032


namespace x_squared_less_than_abs_x_plus_two_l50_5043

theorem x_squared_less_than_abs_x_plus_two (x : ℝ) :
  x^2 < |x| + 2 ↔ -2 < x ∧ x < 2 := by sorry

end x_squared_less_than_abs_x_plus_two_l50_5043


namespace fitted_bowling_ball_volume_l50_5066

/-- The volume of a fitted bowling ball -/
theorem fitted_bowling_ball_volume :
  let sphere_diameter : ℝ := 24
  let hole_depth : ℝ := 6
  let small_hole_diameter : ℝ := 1.5
  let large_hole_diameter : ℝ := 2.5
  let sphere_volume := (4 / 3) * π * (sphere_diameter / 2)^3
  let small_hole_volume := π * (small_hole_diameter / 2)^2 * hole_depth
  let large_hole_volume := π * (large_hole_diameter / 2)^2 * hole_depth
  sphere_volume - 2 * small_hole_volume - large_hole_volume = 2287.875 * π :=
by sorry

end fitted_bowling_ball_volume_l50_5066


namespace fiona_owns_three_hoodies_l50_5045

/-- The number of hoodies Fiona owns -/
def fiona_hoodies : ℕ := sorry

/-- The number of hoodies Casey owns -/
def casey_hoodies : ℕ := sorry

/-- The total number of hoodies Fiona and Casey own -/
def total_hoodies : ℕ := 8

theorem fiona_owns_three_hoodies :
  fiona_hoodies = 3 ∧ casey_hoodies = fiona_hoodies + 2 ∧ fiona_hoodies + casey_hoodies = total_hoodies :=
sorry

end fiona_owns_three_hoodies_l50_5045


namespace complex_cube_eq_negative_eight_l50_5008

theorem complex_cube_eq_negative_eight :
  (1 + Complex.I * Real.sqrt 3) ^ 3 = -8 := by
  sorry

end complex_cube_eq_negative_eight_l50_5008


namespace two_pump_filling_time_l50_5081

/-- Given two pumps with filling rates of 1/3 tank per hour and 4 tanks per hour respectively,
    the time taken to fill a tank when both pumps work together is 3/13 hours. -/
theorem two_pump_filling_time :
  let small_pump_rate : ℚ := 1/3  -- Rate of small pump in tanks per hour
  let large_pump_rate : ℚ := 4    -- Rate of large pump in tanks per hour
  let combined_rate : ℚ := small_pump_rate + large_pump_rate
  let filling_time : ℚ := 1 / combined_rate
  filling_time = 3/13 := by
  sorry

end two_pump_filling_time_l50_5081
