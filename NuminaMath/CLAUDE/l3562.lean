import Mathlib

namespace NUMINAMATH_CALUDE_line_segments_in_proportion_l3562_356230

theorem line_segments_in_proportion :
  let a : ℝ := 2
  let b : ℝ := Real.sqrt 5
  let c : ℝ := 2 * Real.sqrt 3
  let d : ℝ := Real.sqrt 15
  a * d = b * c := by sorry

end NUMINAMATH_CALUDE_line_segments_in_proportion_l3562_356230


namespace NUMINAMATH_CALUDE_min_angular_frequency_l3562_356232

/-- Given a cosine function with specific properties, prove that the minimum angular frequency is 2 -/
theorem min_angular_frequency (ω φ : ℝ) : 
  ω > 0 → 
  (∃ k : ℤ, ω * (π / 3) + φ = k * π) →
  1/2 * Real.cos (ω * (π / 12) + φ) + 1 = 1 →
  (∀ ω' > 0, 
    (∃ k : ℤ, ω' * (π / 3) + φ = k * π) →
    1/2 * Real.cos (ω' * (π / 12) + φ) + 1 = 1 →
    ω' ≥ ω) →
  ω = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_angular_frequency_l3562_356232


namespace NUMINAMATH_CALUDE_dealer_profit_theorem_l3562_356284

/-- Represents the pricing and discount strategy of a dealer -/
structure DealerStrategy where
  markup_percentage : ℝ
  discount_percentage : ℝ
  bulk_deal_articles_sold : ℕ
  bulk_deal_articles_cost : ℕ

/-- Calculates the profit percentage for a dealer given their strategy -/
def calculate_profit_percentage (strategy : DealerStrategy) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the dealer's profit percentage is 80% under the given conditions -/
theorem dealer_profit_theorem (strategy : DealerStrategy) 
  (h1 : strategy.markup_percentage = 100)
  (h2 : strategy.discount_percentage = 10)
  (h3 : strategy.bulk_deal_articles_sold = 20)
  (h4 : strategy.bulk_deal_articles_cost = 15) :
  calculate_profit_percentage strategy = 80 := by
  sorry

end NUMINAMATH_CALUDE_dealer_profit_theorem_l3562_356284


namespace NUMINAMATH_CALUDE_max_area_of_divided_rectangle_l3562_356248

/-- Given a large rectangle divided into 8 smaller rectangles with specific perimeters,
    prove that its maximum area is 512 square centimeters. -/
theorem max_area_of_divided_rectangle :
  ∀ (pA pB pC pD pE : ℝ) (area : ℝ → ℝ),
  pA = 26 →
  pB = 28 →
  pC = 30 →
  pD = 32 →
  pE = 34 →
  (∀ x, area x ≤ 512) →
  (∃ x, area x = 512) :=
by sorry

end NUMINAMATH_CALUDE_max_area_of_divided_rectangle_l3562_356248


namespace NUMINAMATH_CALUDE_sum_of_number_and_its_square_l3562_356282

theorem sum_of_number_and_its_square (n : ℕ) : n = 11 → n + n^2 = 132 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_its_square_l3562_356282


namespace NUMINAMATH_CALUDE_bisecting_line_slope_intercept_sum_l3562_356227

/-- Triangle XYZ with vertices X(1, 9), Y(3, 1), and Z(9, 1) -/
structure Triangle where
  X : ℝ × ℝ := (1, 9)
  Y : ℝ × ℝ := (3, 1)
  Z : ℝ × ℝ := (9, 1)

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The line that bisects the area of the triangle -/
def bisectingLine (t : Triangle) : Line :=
  sorry

theorem bisecting_line_slope_intercept_sum (t : Triangle) :
  (bisectingLine t).slope + (bisectingLine t).yIntercept = -3 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_slope_intercept_sum_l3562_356227


namespace NUMINAMATH_CALUDE_license_plate_combinations_l3562_356274

/-- The number of possible characters for each position in the license plate -/
def numCharOptions : ℕ := 26 + 10

/-- The length of the license plate -/
def plateLength : ℕ := 4

/-- The number of ways to position two identical characters in non-adjacent positions in a 4-character plate -/
def numPairPositions : ℕ := 3

/-- The number of ways to choose characters for the non-duplicate positions -/
def numNonDuplicateChoices : ℕ := numCharOptions * (numCharOptions - 1)

theorem license_plate_combinations :
  numPairPositions * numCharOptions * numNonDuplicateChoices = 136080 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l3562_356274


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l3562_356237

theorem sqrt_sum_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  Real.sqrt (x / 2) + Real.sqrt (y / 2) ≤ Real.sqrt (x + y) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l3562_356237


namespace NUMINAMATH_CALUDE_angle_bisector_ratio_not_determine_shape_l3562_356238

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The ratio of an angle bisector to its corresponding side length --/
def angleBisectorToSideRatio (t : Triangle) : ℝ := sorry

/-- Two triangles are similar if they have the same shape --/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Theorem: The ratio of an angle bisector to the corresponding side's length
    does not uniquely determine the shape of a triangle --/
theorem angle_bisector_ratio_not_determine_shape :
  ∃ (t1 t2 : Triangle), angleBisectorToSideRatio t1 = angleBisectorToSideRatio t2 ∧ ¬ areSimilar t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_ratio_not_determine_shape_l3562_356238


namespace NUMINAMATH_CALUDE_remainder_theorem_l3562_356257

theorem remainder_theorem (P K Q R K' Q' S' T : ℕ) 
  (h1 : P = K * Q + R)
  (h2 : Q = K' * Q' + S')
  (h3 : R * Q' = T)
  (h4 : Q' ≠ 0) :
  P % (K * K') = K * S' + T / Q' :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3562_356257


namespace NUMINAMATH_CALUDE_base_2_representation_of_125_l3562_356250

theorem base_2_representation_of_125 :
  ∃ (a b c d e f g : ℕ),
    (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 0 ∧ g = 1) ∧
    125 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_125_l3562_356250


namespace NUMINAMATH_CALUDE_simplify_fraction_l3562_356200

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) :
  (x + 2) / (x^2 - 2*x) / ((8*x / (x - 2)) + x - 2) = 1 / (x * (x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3562_356200


namespace NUMINAMATH_CALUDE_payment_difference_l3562_356273

-- Define the pizza parameters
def total_slices : ℕ := 10
def plain_pizza_cost : ℚ := 12
def double_cheese_cost : ℚ := 4

-- Define the number of slices each person ate
def bob_double_cheese_slices : ℕ := 5
def bob_plain_slices : ℕ := 2
def cindy_plain_slices : ℕ := 3

-- Calculate the total cost of the pizza
def total_pizza_cost : ℚ := plain_pizza_cost + double_cheese_cost

-- Calculate the cost per slice
def cost_per_slice : ℚ := total_pizza_cost / total_slices

-- Calculate Bob's payment
def bob_payment : ℚ := cost_per_slice * (bob_double_cheese_slices + bob_plain_slices)

-- Calculate Cindy's payment
def cindy_payment : ℚ := cost_per_slice * cindy_plain_slices

-- State the theorem
theorem payment_difference : bob_payment - cindy_payment = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_payment_difference_l3562_356273


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_contained_l3562_356235

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ
  h_pos_r : 0 < r

/-- Check if a point (x, y) is on the ellipse -/
def Ellipse.contains (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Check if a point (x, y) is on or inside the circle -/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  (x - c.h)^2 + (y - c.k)^2 ≤ c.r^2

/-- Check if the circle is tangent to the ellipse -/
def is_tangent (e : Ellipse) (c : Circle) : Prop :=
  ∃ x y : ℝ, e.contains x y ∧ (x - c.h)^2 + (y - c.k)^2 = c.r^2 ∧
    ∀ x' y' : ℝ, e.contains x' y' → (x' - c.h)^2 + (y' - c.k)^2 ≥ c.r^2

/-- Check if the circle is entirely contained within the ellipse -/
def is_contained (e : Ellipse) (c : Circle) : Prop :=
  ∀ x y : ℝ, c.contains x y → e.contains x y

/-- Main theorem: The circle with radius 2 centered at a focus of the ellipse
    is tangent to the ellipse and contained within it -/
theorem ellipse_circle_tangent_contained (e : Ellipse) (c : Circle)
    (h_e : e.a = 6 ∧ e.b = 5)
    (h_c : c.h = Real.sqrt 11 ∧ c.k = 0 ∧ c.r = 2) :
    is_tangent e c ∧ is_contained e c := by
  sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_contained_l3562_356235


namespace NUMINAMATH_CALUDE_marble_difference_l3562_356252

theorem marble_difference (e d : ℕ) (h1 : e > d) (h2 : e = (d - 8) + 30) : e - d = 22 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l3562_356252


namespace NUMINAMATH_CALUDE_sin_cos_alpha_abs_value_l3562_356267

theorem sin_cos_alpha_abs_value (α : Real) 
  (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) : 
  |Real.sin α * Real.cos α| = 2/5 := by sorry

end NUMINAMATH_CALUDE_sin_cos_alpha_abs_value_l3562_356267


namespace NUMINAMATH_CALUDE_sandy_marbles_l3562_356255

/-- The number of marbles in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of red marbles Jessica has -/
def jessica_dozens : ℕ := 3

/-- The number of times more red marbles Sandy has compared to Jessica -/
def sandy_multiplier : ℕ := 4

/-- Theorem stating the number of red marbles Sandy has -/
theorem sandy_marbles : jessica_dozens * dozen * sandy_multiplier = 144 := by
  sorry

end NUMINAMATH_CALUDE_sandy_marbles_l3562_356255


namespace NUMINAMATH_CALUDE_charlie_feather_collection_l3562_356243

/-- The number of sets of wings Charlie needs to make -/
def num_sets : ℕ := 2

/-- The number of feathers required for each set of wings -/
def feathers_per_set : ℕ := 900

/-- The number of feathers Charlie already has -/
def feathers_collected : ℕ := 387

/-- The total number of additional feathers Charlie needs to collect -/
def additional_feathers_needed : ℕ := num_sets * feathers_per_set - feathers_collected

theorem charlie_feather_collection :
  additional_feathers_needed = 1413 := by sorry

end NUMINAMATH_CALUDE_charlie_feather_collection_l3562_356243


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l3562_356278

theorem quadratic_minimum_value (c : ℝ) : 
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, -x^2 - 2*x + c ≥ -5) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, -x^2 - 2*x + c = -5) → 
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l3562_356278


namespace NUMINAMATH_CALUDE_min_prime_angle_in_linear_pair_l3562_356224

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem min_prime_angle_in_linear_pair (a b : ℕ) :
  a + b = 180 →
  is_prime a →
  is_prime b →
  a > b →
  b ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_min_prime_angle_in_linear_pair_l3562_356224


namespace NUMINAMATH_CALUDE_total_length_of_remaining_segments_l3562_356296

/-- A figure with perpendicular adjacent sides -/
structure PerpendicularFigure where
  top_segments : List ℝ
  bottom_segment : ℝ
  left_segment : ℝ
  right_segment : ℝ

/-- The remaining figure after removing six sides -/
def RemainingFigure (f : PerpendicularFigure) : PerpendicularFigure :=
  { top_segments := [1],
    bottom_segment := f.bottom_segment,
    left_segment := f.left_segment,
    right_segment := 9 }

theorem total_length_of_remaining_segments (f : PerpendicularFigure)
  (h1 : f.top_segments = [3, 1, 1])
  (h2 : f.left_segment = 10)
  (h3 : f.bottom_segment = f.top_segments.sum)
  : (RemainingFigure f).top_segments.sum + 
    (RemainingFigure f).bottom_segment + 
    (RemainingFigure f).left_segment + 
    (RemainingFigure f).right_segment = 25 := by
  sorry


end NUMINAMATH_CALUDE_total_length_of_remaining_segments_l3562_356296


namespace NUMINAMATH_CALUDE_solution_set_implies_a_and_b_solution_set_when_a_negative_l3562_356299

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - (a - 2) * x - 2

-- Part 1
theorem solution_set_implies_a_and_b :
  ∀ a b : ℝ, (∀ x : ℝ, f a x ≤ b ↔ -2 ≤ x ∧ x ≤ 1) → a = 1 ∧ b = 0 := by sorry

-- Part 2
theorem solution_set_when_a_negative :
  ∀ a : ℝ, a < 0 →
    (∀ x : ℝ, f a x ≥ 0 ↔
      ((-2 < a ∧ a < 0 ∧ 1 ≤ x ∧ x ≤ -2/a) ∨
       (a = -2 ∧ x = 1) ∨
       (a < -2 ∧ -2/a ≤ x ∧ x ≤ 1))) := by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_and_b_solution_set_when_a_negative_l3562_356299


namespace NUMINAMATH_CALUDE_solution_in_interval_implies_a_range_l3562_356290

theorem solution_in_interval_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Icc 1 5, x^2 + a*x - 2 = 0) →
  a ∈ Set.Icc (-23/5) 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_in_interval_implies_a_range_l3562_356290


namespace NUMINAMATH_CALUDE_cube_edge_length_is_5_l3562_356264

/-- The edge length of a cube immersed in water --/
def cube_edge_length (base_length base_width water_rise : ℝ) : ℝ :=
  (base_length * base_width * water_rise) ^ (1/3)

/-- Theorem stating that the edge length of the cube is 5 cm --/
theorem cube_edge_length_is_5 :
  cube_edge_length 10 5 2.5 = 5 := by sorry

end NUMINAMATH_CALUDE_cube_edge_length_is_5_l3562_356264


namespace NUMINAMATH_CALUDE_f_2013_equals_2_l3562_356236

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def satisfies_recurrence (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) = f x + 2 * f 2

theorem f_2013_equals_2 (f : ℝ → ℝ) 
  (h1 : is_even_function f)
  (h2 : satisfies_recurrence f)
  (h3 : f (-1) = 2) :
  f 2013 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_2013_equals_2_l3562_356236


namespace NUMINAMATH_CALUDE_probability_of_selection_l3562_356229

def total_students : ℕ := 10
def students_per_teacher : ℕ := 4

theorem probability_of_selection (total_students : ℕ) (students_per_teacher : ℕ) :
  total_students = 10 → students_per_teacher = 4 →
  (1 : ℚ) - (1 - students_per_teacher / total_students) ^ 2 = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_l3562_356229


namespace NUMINAMATH_CALUDE_circle_area_decrease_l3562_356234

/-- Given three circles with radii r1, r2, and r3, prove that the decrease in their combined area
    when each radius is reduced by 50% is equal to 75% of their original combined area. -/
theorem circle_area_decrease (r1 r2 r3 : ℝ) (hr1 : r1 > 0) (hr2 : r2 > 0) (hr3 : r3 > 0) :
  let original_area := π * (r1^2 + r2^2 + r3^2)
  let new_area := π * ((r1/2)^2 + (r2/2)^2 + (r3/2)^2)
  original_area - new_area = (3/4) * original_area :=
by sorry

end NUMINAMATH_CALUDE_circle_area_decrease_l3562_356234


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l3562_356283

/-- Converts a list of bits (represented as Bools) to a natural number. -/
def bitsToNat (bits : List Bool) : Nat :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- The binary number 10101₂ -/
def num1 : List Bool := [true, false, true, false, true]

/-- The binary number 1011₂ -/
def num2 : List Bool := [true, false, true, true]

/-- The binary number 1110₂ -/
def num3 : List Bool := [true, true, true, false]

/-- The binary number 110001₂ -/
def num4 : List Bool := [true, true, false, false, false, true]

/-- The binary number 1101₂ -/
def num5 : List Bool := [true, true, false, true]

/-- The binary number 101100₂ (the expected result) -/
def result : List Bool := [true, false, true, true, false, false]

theorem binary_addition_subtraction :
  bitsToNat num1 + bitsToNat num2 + bitsToNat num3 + bitsToNat num4 - bitsToNat num5 = bitsToNat result := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l3562_356283


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_t_values_l3562_356211

theorem sqrt_equality_implies_t_values (t : ℝ) :
  (Real.sqrt (5 * Real.sqrt (t - 5)) = (10 - t + t^2)^(1/4)) →
  (t = 13 + Real.sqrt 34 ∨ t = 13 - Real.sqrt 34) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_t_values_l3562_356211


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l3562_356240

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for z
def z_condition (z : ℂ) : Prop := z * (1 + i) = 2 * i + 1

-- Theorem statement
theorem z_in_first_quadrant (z : ℂ) (h : z_condition z) : 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l3562_356240


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3562_356288

theorem smallest_multiplier_for_perfect_square : ∃ (k : ℕ+), 
  (∀ (m : ℕ+), (∃ (n : ℕ), 2010 * m = n * n) → k ≤ m) ∧ 
  (∃ (n : ℕ), 2010 * k = n * n) ∧
  k = 2010 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l3562_356288


namespace NUMINAMATH_CALUDE_pentagon_tiles_18gon_l3562_356210

-- Define the pentagon
structure Pentagon where
  side_length : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ
  sum_of_angles : angle1 + angle2 + angle3 + angle4 + angle5 = 540

-- Define the regular 18-gon
structure Regular18Gon where
  side_length : ℝ
  interior_angle : ℝ
  interior_angle_eq : interior_angle = 160

-- Theorem statement
theorem pentagon_tiles_18gon (c : ℝ) (h : c > 0) :
  ∃ (p : Pentagon) (g : Regular18Gon),
    p.side_length = c ∧
    g.side_length = c ∧
    p.angle1 = 60 ∧
    p.angle2 = 160 ∧
    p.angle3 = 80 ∧
    p.angle4 = 100 ∧
    p.angle5 = 140 ∧
    (∃ (n : ℕ), n = 18 ∧ n * p.side_length = 18 * g.side_length) :=
  sorry

end NUMINAMATH_CALUDE_pentagon_tiles_18gon_l3562_356210


namespace NUMINAMATH_CALUDE_circle_equation_correct_l3562_356258

-- Define the center and radius of the circle
def center : ℝ × ℝ := (-2, 3)
def radius : ℝ := 2

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 3)^2 = 4

-- Theorem stating that the given equation represents the circle with the specified center and radius
theorem circle_equation_correct :
  ∀ x y : ℝ, circle_equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l3562_356258


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l3562_356226

theorem min_value_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : x - 2*y + 3*z = 0) :
  y^2 / (x*z) ≥ 3 := by
  sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x - 2*y + 3*z = 0 ∧ y^2 / (x*z) < 3 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l3562_356226


namespace NUMINAMATH_CALUDE_shoes_to_sell_l3562_356247

def monthly_goal : ℕ := 80
def sold_last_week : ℕ := 27
def sold_this_week : ℕ := 12

theorem shoes_to_sell : monthly_goal - (sold_last_week + sold_this_week) = 41 := by
  sorry

end NUMINAMATH_CALUDE_shoes_to_sell_l3562_356247


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l3562_356225

/-- Calculates the final weight after a two-week diet plan -/
def final_weight (initial_weight : ℝ) (first_week_loss : ℝ) (second_week_rate : ℝ) : ℝ :=
  initial_weight - (first_week_loss + second_week_rate * first_week_loss)

/-- Jessie's weight loss problem -/
theorem jessie_weight_loss :
  let initial_weight : ℝ := 92
  let first_week_loss : ℝ := 5
  let second_week_rate : ℝ := 1.3
  final_weight initial_weight first_week_loss second_week_rate = 80.5 := by
  sorry

#eval final_weight 92 5 1.3

end NUMINAMATH_CALUDE_jessie_weight_loss_l3562_356225


namespace NUMINAMATH_CALUDE_pauls_books_l3562_356292

theorem pauls_books (books_sold : ℕ) (books_left : ℕ) : 
  books_sold = 137 → books_left = 105 → books_sold + books_left = 242 :=
by sorry

end NUMINAMATH_CALUDE_pauls_books_l3562_356292


namespace NUMINAMATH_CALUDE_inequality_counterexample_l3562_356251

theorem inequality_counterexample : 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b ≥ 2 * Real.sqrt (a * b)) → 
  ¬(∀ x : ℝ, x + 1/x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_counterexample_l3562_356251


namespace NUMINAMATH_CALUDE_notebook_cost_is_50_l3562_356201

def mean_expenditure : ℝ := 500
def num_days : ℕ := 7
def other_days_expenditure : List ℝ := [450, 600, 400, 500, 550, 300]
def pen_cost : ℝ := 30
def earphone_cost : ℝ := 620

def total_week_expenditure : ℝ := mean_expenditure * num_days
def other_days_total : ℝ := other_days_expenditure.sum
def friday_expenditure : ℝ := total_week_expenditure - other_days_total

theorem notebook_cost_is_50 :
  friday_expenditure - (pen_cost + earphone_cost) = 50 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_is_50_l3562_356201


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l3562_356214

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l3562_356214


namespace NUMINAMATH_CALUDE_adjacent_sum_theorem_l3562_356256

/-- Represents a 3x3 table with numbers from 1 to 9 -/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Checks if a table contains each number from 1 to 9 exactly once -/
def is_valid (t : Table) : Prop :=
  ∀ n : Fin 9, ∃! (i j : Fin 3), t i j = n

/-- Checks if the table has 1, 2, 3, and 4 in the correct positions -/
def correct_positions (t : Table) : Prop :=
  t 0 0 = 0 ∧ t 2 0 = 1 ∧ t 0 2 = 2 ∧ t 2 2 = 3

/-- Returns the sum of adjacent numbers to the given position -/
def adjacent_sum (t : Table) (i j : Fin 3) : ℕ :=
  (if i > 0 then (t (i-1) j).val + 1 else 0) +
  (if i < 2 then (t (i+1) j).val + 1 else 0) +
  (if j > 0 then (t i (j-1)).val + 1 else 0) +
  (if j < 2 then (t i (j+1)).val + 1 else 0)

/-- The main theorem -/
theorem adjacent_sum_theorem (t : Table) :
  is_valid t →
  correct_positions t →
  (∃ i j : Fin 3, t i j = 4 ∧ adjacent_sum t i j = 9) →
  (∃ i j : Fin 3, t i j = 5 ∧ adjacent_sum t i j = 29) :=
by sorry

end NUMINAMATH_CALUDE_adjacent_sum_theorem_l3562_356256


namespace NUMINAMATH_CALUDE_max_rabbits_with_traits_l3562_356287

theorem max_rabbits_with_traits (N : ℕ) : 
  (∃ (long_ears jump_far both : Finset (Fin N)),
    long_ears.card = 13 ∧ 
    jump_far.card = 17 ∧ 
    both ⊆ long_ears ∧ 
    both ⊆ jump_far ∧ 
    both.card ≥ 3) →
  N ≤ 27 := by
sorry

end NUMINAMATH_CALUDE_max_rabbits_with_traits_l3562_356287


namespace NUMINAMATH_CALUDE_batsman_average_l3562_356239

/-- 
Given a batsman who has played 16 innings, prove that if he scores 87 runs 
in the 17th inning and this increases his average by 4 runs, 
then his new average after the 17th inning is 23 runs.
-/
theorem batsman_average (prev_average : ℝ) : 
  (16 * prev_average + 87) / 17 = prev_average + 4 → 
  prev_average + 4 = 23 := by sorry

end NUMINAMATH_CALUDE_batsman_average_l3562_356239


namespace NUMINAMATH_CALUDE_xy_sum_theorem_l3562_356298

theorem xy_sum_theorem (x y : ℕ) (hx : x > 0) (hy : y > 0) (hx_lt_20 : x < 20) (hy_lt_20 : y < 20) 
  (h_eq : x + y + x * y = 99) : x + y = 23 ∨ x + y = 18 :=
sorry

end NUMINAMATH_CALUDE_xy_sum_theorem_l3562_356298


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_25_l3562_356263

theorem cubic_fraction_equals_25 (a b : ℝ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b^3)^2 / (a^2 - a*b + b^2)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_25_l3562_356263


namespace NUMINAMATH_CALUDE_sqrt_identity_l3562_356205

theorem sqrt_identity (a b : ℝ) (h : a^2 ≥ b ∧ a ≥ 0 ∧ b ≥ 0) :
  (∀ (s : Bool), Real.sqrt (a + (if s then 1 else -1) * Real.sqrt b) = 
    Real.sqrt ((a + Real.sqrt (a^2 - b)) / 2) + 
    (if s then 1 else -1) * Real.sqrt ((a - Real.sqrt (a^2 - b)) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_identity_l3562_356205


namespace NUMINAMATH_CALUDE_log_inequality_l3562_356295

theorem log_inequality : ∃ (a b c : ℝ), 
  a = Real.log 2 / Real.log 5 ∧ 
  b = Real.log 3 / Real.log 8 ∧ 
  c = (1 : ℝ) / 2 ∧ 
  a < c ∧ c < b :=
sorry

end NUMINAMATH_CALUDE_log_inequality_l3562_356295


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l3562_356271

theorem compare_negative_fractions : -10/11 > -11/12 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l3562_356271


namespace NUMINAMATH_CALUDE_power_four_remainder_l3562_356286

theorem power_four_remainder (a : ℕ) (h1 : a > 0) (h2 : 2 ∣ a) : 4^a % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_four_remainder_l3562_356286


namespace NUMINAMATH_CALUDE_total_birds_count_l3562_356260

/-- Proves that given the specified conditions, the total number of birds is 185 -/
theorem total_birds_count (chickens ducks : ℕ) 
  (h1 : ducks = 4 * chickens + 10) 
  (h2 : ducks = 150) : 
  chickens + ducks = 185 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_count_l3562_356260


namespace NUMINAMATH_CALUDE_greatest_a_no_integral_solution_l3562_356246

theorem greatest_a_no_integral_solution :
  (∀ a : ℤ, (∀ x : ℤ, ¬(|x + 1| < a - (3/2))) → a ≤ 1) ∧
  (∃ x : ℤ, |x + 1| < 2 - (3/2)) :=
sorry

end NUMINAMATH_CALUDE_greatest_a_no_integral_solution_l3562_356246


namespace NUMINAMATH_CALUDE_range_of_a_l3562_356231

/-- Given sets A and B, where "x ∈ B" is a sufficient but not necessary condition for "x ∈ A",
    this theorem proves that the range of values for a is [0, 1]. -/
theorem range_of_a (A B : Set ℝ) (a : ℝ) : 
  A = {x : ℝ | x^2 - x - 2 ≤ 0} →
  B = {x : ℝ | |x - a| ≤ 1} →
  (∀ x, x ∈ B → x ∈ A) →
  ¬(∀ x, x ∈ A → x ∈ B) →
  0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3562_356231


namespace NUMINAMATH_CALUDE_sequence_type_l3562_356228

theorem sequence_type (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = 2 * n^2 - 2 * n) → 
  (∃ d : ℝ, d = 4 ∧ ∀ n, a (n + 1) - a n = d) := by
sorry

end NUMINAMATH_CALUDE_sequence_type_l3562_356228


namespace NUMINAMATH_CALUDE_marbles_remaining_l3562_356216

theorem marbles_remaining (initial : ℝ) (lost : ℝ) (given_away : ℝ) (found : ℝ) : 
  initial = 150 → 
  lost = 58.5 → 
  given_away = 37.2 → 
  found = 10.8 → 
  initial - lost - given_away + found = 65.1 := by
sorry

end NUMINAMATH_CALUDE_marbles_remaining_l3562_356216


namespace NUMINAMATH_CALUDE_alpha_beta_cosine_l3562_356212

theorem alpha_beta_cosine (α β : Real)
  (h_α : α ∈ Set.Ioo 0 (π / 3))
  (h_β : β ∈ Set.Ioo (π / 6) (π / 2))
  (eq_α : 5 * Real.sqrt 3 * Real.sin α + 5 * Real.cos α = 8)
  (eq_β : Real.sqrt 2 * Real.sin β + Real.sqrt 6 * Real.cos β = 2) :
  Real.cos (α + π / 6) = 3 / 5 ∧ Real.cos (α + β) = - Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_cosine_l3562_356212


namespace NUMINAMATH_CALUDE_h_definition_l3562_356269

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x - 1
def g (x : ℝ) : ℝ := 2 * x + 3

-- Define h as a function that satisfies f(h(x)) = g(x)
noncomputable def h (x : ℝ) : ℝ := sorry

-- State the theorem
theorem h_definition (x : ℝ) : f (h x) = g x → h x = (2 * x + 4) / 3 := by
  sorry

end NUMINAMATH_CALUDE_h_definition_l3562_356269


namespace NUMINAMATH_CALUDE_basketball_shot_probability_l3562_356244

theorem basketball_shot_probability :
  let p_at_least_one : ℝ := 0.9333333333333333
  let p_free_throw : ℝ := 4/5
  let p_high_school : ℝ := 1/2
  let p_pro : ℝ := 1/3
  (1 - (1 - p_free_throw) * (1 - p_high_school) * (1 - p_pro) = p_at_least_one) :=
by sorry

end NUMINAMATH_CALUDE_basketball_shot_probability_l3562_356244


namespace NUMINAMATH_CALUDE_unique_prime_evaluation_l3562_356262

theorem unique_prime_evaluation (T : ℕ) (h : T = 2161) :
  ∃! p : ℕ, Prime p ∧ ∃ n : ℤ, n^4 - 898*n^2 + T - 2160 = p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_evaluation_l3562_356262


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_l3562_356285

theorem no_solution_to_inequality : ¬ ∃ x : ℝ, |x - 3| + |x + 4| < 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_l3562_356285


namespace NUMINAMATH_CALUDE_edric_hourly_rate_l3562_356293

/-- Edric's salary calculation --/
def salary_calculation (B C S P D : ℚ) (H W : ℕ) : ℚ :=
  let E := B + (C * S) + P - D
  let T := (H * W * 4 : ℚ)
  E / T

/-- Edric's hourly rate is approximately $3.86 --/
theorem edric_hourly_rate :
  let B := 576
  let C := 3 / 100
  let S := 4000
  let P := 75
  let D := 30
  let H := 8
  let W := 6
  abs (salary_calculation B C S P D H W - 386 / 100) < 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_edric_hourly_rate_l3562_356293


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3562_356253

theorem fraction_evaluation : (3020 - 2890)^2 / 196 = 86 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3562_356253


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_greater_than_one_l3562_356281

/-- If the equation 2ax^2 - x - 1 = 0 has exactly one solution in the interval (0,1), then a > 1 -/
theorem unique_solution_implies_a_greater_than_one (a : ℝ) : 
  (∃! x : ℝ, x ∈ Set.Ioo 0 1 ∧ 2*a*x^2 - x - 1 = 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_greater_than_one_l3562_356281


namespace NUMINAMATH_CALUDE_square_property_l3562_356280

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def remove_last_two_digits (n : ℕ) : ℕ := n / 100

theorem square_property (n : ℕ) :
  (n > 0 ∧ is_perfect_square (remove_last_two_digits (n^2))) ↔
  (∃ k : ℕ, k > 0 ∧ n = 10 * k) ∨
  (n ∈ ({11,12,13,14,21,22,31,41,1,2,3,4,5,6,7,8,9} : Finset ℕ)) :=
sorry

end NUMINAMATH_CALUDE_square_property_l3562_356280


namespace NUMINAMATH_CALUDE_rectangle_placement_l3562_356217

theorem rectangle_placement (a b c d : ℝ) 
  (h1 : a < c) (h2 : c < d) (h3 : d < b) (h4 : a * b < c * d) :
  (∃ (θ : ℝ), 0 < θ ∧ θ < π / 2 ∧ 
    b * Real.cos θ + a * Real.sin θ ≤ c ∧
    b * Real.sin θ + a * Real.cos θ ≤ d) ↔ 
  (b^2 - a^2)^2 ≤ (b*d - a*c)^2 + (b*c - a*d)^2 := by sorry

end NUMINAMATH_CALUDE_rectangle_placement_l3562_356217


namespace NUMINAMATH_CALUDE_last_digit_sum_powers_l3562_356275

theorem last_digit_sum_powers : (2^2011 + 3^2011) % 10 = 5 := by sorry

end NUMINAMATH_CALUDE_last_digit_sum_powers_l3562_356275


namespace NUMINAMATH_CALUDE_circle_properties_l3562_356268

/-- Given a circle with diameter endpoints (2, 1) and (8, 7), prove its center and diameter length -/
theorem circle_properties :
  let p1 : ℝ × ℝ := (2, 1)
  let p2 : ℝ × ℝ := (8, 7)
  let center := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let diameter_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  center = (5, 4) ∧ diameter_length = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l3562_356268


namespace NUMINAMATH_CALUDE_complex_multiplication_l3562_356266

theorem complex_multiplication (i : ℂ) (h : i * i = -1) : i * (2 * i + 1) = -2 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3562_356266


namespace NUMINAMATH_CALUDE_length_of_AB_is_seven_l3562_356289

-- Define the triangle ABC
structure TriangleABC where
  A : Point
  B : Point
  C : Point

-- Define the triangle CBD
structure TriangleCBD where
  C : Point
  B : Point
  D : Point

-- Define the properties of the triangles
def isIsosceles (t : TriangleABC) : Prop := sorry
def isEquilateral (t : TriangleABC) : Prop := sorry
def isIsoscelesCBD (t : TriangleCBD) : Prop := sorry
def perimeterCBD (t : TriangleCBD) : ℝ := sorry
def perimeterABC (t : TriangleABC) : ℝ := sorry
def lengthBD (t : TriangleCBD) : ℝ := sorry
def lengthAB (t : TriangleABC) : ℝ := sorry

theorem length_of_AB_is_seven 
  (abc : TriangleABC) 
  (cbd : TriangleCBD) 
  (h1 : isIsosceles abc)
  (h2 : isEquilateral abc)
  (h3 : isIsoscelesCBD cbd)
  (h4 : perimeterCBD cbd = 24)
  (h5 : perimeterABC abc = 21)
  (h6 : lengthBD cbd = 10) :
  lengthAB abc = 7 := by sorry

end NUMINAMATH_CALUDE_length_of_AB_is_seven_l3562_356289


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l3562_356272

theorem ceiling_floor_product_range (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 210 → -15 < y ∧ y < -14 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l3562_356272


namespace NUMINAMATH_CALUDE_gcd_1189_264_l3562_356245

theorem gcd_1189_264 : Nat.gcd 1189 264 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1189_264_l3562_356245


namespace NUMINAMATH_CALUDE_cinema_seats_l3562_356203

theorem cinema_seats (rows : ℕ) (seats_per_row : ℕ) (h1 : rows = 21) (h2 : seats_per_row = 26) :
  rows * seats_per_row = 546 := by
  sorry

end NUMINAMATH_CALUDE_cinema_seats_l3562_356203


namespace NUMINAMATH_CALUDE_jakes_weight_l3562_356270

theorem jakes_weight (jake_weight sister_weight : ℝ) 
  (h1 : jake_weight - 8 = 2 * sister_weight)
  (h2 : jake_weight + sister_weight = 278) :
  jake_weight = 188 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l3562_356270


namespace NUMINAMATH_CALUDE_volume_common_tetrahedra_l3562_356265

/-- Given a parallelepiped ABCDA₁B₁C₁D₁ with volume V, the volume of the common part
    of tetrahedra AB₁CD₁ and A₁BC₁D is V/12 -/
theorem volume_common_tetrahedra (V : ℝ) : ℝ :=
  let parallelepiped_volume := V
  let common_volume := V / 12
  common_volume

#check volume_common_tetrahedra

end NUMINAMATH_CALUDE_volume_common_tetrahedra_l3562_356265


namespace NUMINAMATH_CALUDE_tylenol_dosage_l3562_356209

/-- Calculates the mg per pill given the total dosage and number of pills -/
def mg_per_pill (dosage_mg : ℕ) (dosage_interval_hours : ℕ) (duration_days : ℕ) (total_pills : ℕ) : ℚ :=
  let doses_per_day := 24 / dosage_interval_hours
  let total_doses := doses_per_day * duration_days
  let total_mg := dosage_mg * total_doses
  (total_mg : ℚ) / total_pills

theorem tylenol_dosage :
  mg_per_pill 1000 6 14 112 = 500 := by
  sorry

end NUMINAMATH_CALUDE_tylenol_dosage_l3562_356209


namespace NUMINAMATH_CALUDE_product_of_integers_l3562_356206

theorem product_of_integers (p q r : ℕ+) : 
  p + q + r = 30 → 
  (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 420 / (p * q * r) = 1 → 
  p * q * r = 1800 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l3562_356206


namespace NUMINAMATH_CALUDE_discriminant_of_2x2_minus_5x_plus_6_l3562_356276

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 2x^2 - 5x + 6 is -23 -/
theorem discriminant_of_2x2_minus_5x_plus_6 :
  discriminant 2 (-5) 6 = -23 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_2x2_minus_5x_plus_6_l3562_356276


namespace NUMINAMATH_CALUDE_solve_equation_l3562_356215

theorem solve_equation (x : ℝ) : (1 / 2) * (1 / 7) * x = 14 → x = 196 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3562_356215


namespace NUMINAMATH_CALUDE_verbal_equals_algebraic_l3562_356233

/-- The verbal description of the algebraic expression "5-4a" -/
def verbal_description : String := "the difference of 5 and 4 times a"

/-- The algebraic expression -/
def algebraic_expression (a : ℝ) : ℝ := 5 - 4 * a

theorem verbal_equals_algebraic :
  ∀ a : ℝ, verbal_description = "the difference of 5 and 4 times a" ↔ 
  algebraic_expression a = 5 - 4 * a :=
by sorry

end NUMINAMATH_CALUDE_verbal_equals_algebraic_l3562_356233


namespace NUMINAMATH_CALUDE_sum_two_digit_integers_mod_1000_l3562_356213

/-- The sum of all four-digit integers formed using exactly two different digits -/
def S : ℕ := sorry

/-- Theorem stating that S mod 1000 = 370 -/
theorem sum_two_digit_integers_mod_1000 : S % 1000 = 370 := by sorry

end NUMINAMATH_CALUDE_sum_two_digit_integers_mod_1000_l3562_356213


namespace NUMINAMATH_CALUDE_inverse_arcsin_function_l3562_356242

theorem inverse_arcsin_function (f : ℝ → ℝ) (h : ∀ x, f x = Real.arcsin (2 * x + 1)) :
  f⁻¹ (π / 6) = -1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_arcsin_function_l3562_356242


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3562_356221

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_pos : q > 0) 
  (h_geom : ∀ n, a (n + 1) = q * a n) 
  (h_eq : a 3 * a 9 = 2 * (a 5)^2) : 
  q = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3562_356221


namespace NUMINAMATH_CALUDE_product_of_fractions_l3562_356204

theorem product_of_fractions : (1 : ℚ) / 3 * 3 / 5 * 5 / 7 * 7 / 9 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3562_356204


namespace NUMINAMATH_CALUDE_minimum_cost_theorem_l3562_356241

/-- Represents the cost and quantity of prizes A and B --/
structure PrizePurchase where
  costA : ℕ  -- Cost of prize A
  costB : ℕ  -- Cost of prize B
  quantityA : ℕ  -- Quantity of prize A
  quantityB : ℕ  -- Quantity of prize B

/-- Conditions for the prize purchase problem --/
def PrizePurchaseConditions (p : PrizePurchase) : Prop :=
  3 * p.costA + 2 * p.costB = 390 ∧  -- Condition 1
  4 * p.costA = 5 * p.costB + 60 ∧  -- Condition 2
  p.quantityA + p.quantityB = 30 ∧  -- Condition 3
  p.quantityA ≥ p.quantityB / 2 ∧  -- Condition 4
  p.costA * p.quantityA + p.costB * p.quantityB ≤ 2170  -- Condition 5

/-- The theorem to be proved --/
theorem minimum_cost_theorem (p : PrizePurchase) 
  (h : PrizePurchaseConditions p) : 
  p.costA = 90 ∧ p.costB = 60 ∧ 
  p.quantityA * p.costA + p.quantityB * p.costB ≥ 2100 :=
sorry

end NUMINAMATH_CALUDE_minimum_cost_theorem_l3562_356241


namespace NUMINAMATH_CALUDE_reading_pattern_l3562_356207

theorem reading_pattern (x y : ℝ) : 
  (∀ (days_xiaoming days_xiaoying : ℕ), 
    days_xiaoming = 3 ∧ days_xiaoying = 5 → 
    days_xiaoming * x + 6 = days_xiaoying * y) ∧
  (y = x - 10) →
  3 * x = 5 * y - 6 ∧ y = 2 * x - 10 := by
sorry

end NUMINAMATH_CALUDE_reading_pattern_l3562_356207


namespace NUMINAMATH_CALUDE_tangerine_orange_difference_l3562_356254

def initial_oranges : ℕ := 5
def initial_tangerines : ℕ := 17
def removed_oranges : ℕ := 2
def removed_tangerines : ℕ := 10
def added_oranges : ℕ := 3
def added_tangerines : ℕ := 6

theorem tangerine_orange_difference :
  (initial_tangerines - removed_tangerines + added_tangerines) -
  (initial_oranges - removed_oranges + added_oranges) = 7 := by
sorry

end NUMINAMATH_CALUDE_tangerine_orange_difference_l3562_356254


namespace NUMINAMATH_CALUDE_positive_real_inequality_l3562_356202

theorem positive_real_inequality (x : ℝ) (hx : x > 0) :
  x + 1 / x ≥ 2 ∧ (x + 1 / x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l3562_356202


namespace NUMINAMATH_CALUDE_students_like_both_desserts_l3562_356208

/-- Proves the number of students who like both apple pie and chocolate cake -/
theorem students_like_both_desserts 
  (total : ℕ) 
  (like_apple : ℕ) 
  (like_chocolate : ℕ) 
  (like_neither : ℕ) 
  (h1 : total = 50)
  (h2 : like_apple = 22)
  (h3 : like_chocolate = 20)
  (h4 : like_neither = 15) :
  total - like_neither - (like_apple + like_chocolate - (total - like_neither)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_like_both_desserts_l3562_356208


namespace NUMINAMATH_CALUDE_concrete_blocks_theorem_l3562_356261

/-- Calculates the number of concrete blocks per section in a hedge. -/
def concrete_blocks_per_section (total_sections : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) : ℕ :=
  (total_cost / cost_per_piece) / total_sections

/-- Proves that the number of concrete blocks per section is 30 given the specified conditions. -/
theorem concrete_blocks_theorem :
  concrete_blocks_per_section 8 480 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_concrete_blocks_theorem_l3562_356261


namespace NUMINAMATH_CALUDE_diamonds_in_F_10_l3562_356220

/-- Number of diamonds in figure F_n -/
def diamonds (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 1 + 3 * (n * (n - 1) / 2)

/-- Theorem stating that F_10 contains 136 diamonds -/
theorem diamonds_in_F_10 : diamonds 10 = 136 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_in_F_10_l3562_356220


namespace NUMINAMATH_CALUDE_farm_tax_calculation_l3562_356249

/-- The farm tax calculation problem -/
theorem farm_tax_calculation 
  (tax_percentage : Real) 
  (total_tax_collected : Real) 
  (willam_land_percentage : Real) : 
  tax_percentage = 0.4 →
  total_tax_collected = 3840 →
  willam_land_percentage = 0.3125 →
  willam_land_percentage * (total_tax_collected / tax_percentage) = 3000 := by
  sorry

#check farm_tax_calculation

end NUMINAMATH_CALUDE_farm_tax_calculation_l3562_356249


namespace NUMINAMATH_CALUDE_not_divisible_by_five_l3562_356222

theorem not_divisible_by_five (n : ℤ) : ¬ (5 ∣ (n^2 + n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_five_l3562_356222


namespace NUMINAMATH_CALUDE_avg_sq_feet_per_person_approx_l3562_356218

/-- The population of the United States -/
def us_population : ℕ := 226504825

/-- The area of the United States in square miles -/
def us_area_sq_miles : ℕ := 3615122

/-- The number of square feet in a square mile -/
def sq_feet_per_sq_mile : ℕ := 5280 * 5280

/-- The average square feet per person in the United States -/
def avg_sq_feet_per_person : ℚ :=
  (us_area_sq_miles * sq_feet_per_sq_mile : ℚ) / us_population

/-- Theorem stating that the average square feet per person is approximately 500000 -/
theorem avg_sq_feet_per_person_approx :
  ∃ ε > 0, abs (avg_sq_feet_per_person - 500000) < ε := by
  sorry

end NUMINAMATH_CALUDE_avg_sq_feet_per_person_approx_l3562_356218


namespace NUMINAMATH_CALUDE_haley_jason_difference_l3562_356297

/-- The number of necklaces Haley has -/
def haley_necklaces : ℕ := 25

/-- The difference between Haley's and Josh's necklaces -/
def haley_josh_diff : ℕ := 15

/-- Represents the relationship between Josh's and Jason's necklaces -/
def josh_jason_ratio : ℚ := 1/2

/-- The number of necklaces Josh has -/
def josh_necklaces : ℕ := haley_necklaces - haley_josh_diff

/-- The number of necklaces Jason has -/
def jason_necklaces : ℕ := (2 * josh_necklaces)

theorem haley_jason_difference : haley_necklaces - jason_necklaces = 5 := by
  sorry

end NUMINAMATH_CALUDE_haley_jason_difference_l3562_356297


namespace NUMINAMATH_CALUDE_last_two_digits_of_floor_fraction_l3562_356277

theorem last_two_digits_of_floor_fraction : ∃ n : ℕ, 
  n ≥ 10^62 - 3 * 10^31 + 8 ∧ 
  n < 10^62 - 3 * 10^31 + 9 ∧ 
  n % 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_floor_fraction_l3562_356277


namespace NUMINAMATH_CALUDE_exists_geometric_subsequence_l3562_356291

/-- A strictly increasing sequence of positive integers in arithmetic progression -/
def ArithmeticSequence : ℕ → ℕ := λ n => sorry

/-- The first term of the arithmetic progression -/
def a : ℕ := sorry

/-- The common difference of the arithmetic progression -/
def d : ℕ := sorry

/-- Condition: ArithmeticSequence is strictly increasing -/
axiom strictly_increasing : ∀ n : ℕ, ArithmeticSequence n < ArithmeticSequence (n + 1)

/-- Condition: ArithmeticSequence is an arithmetic progression -/
axiom is_arithmetic_progression : ∀ n : ℕ, ArithmeticSequence n = a + (n - 1) * d

/-- The existence of an infinite geometric sub-sequence -/
theorem exists_geometric_subsequence :
  ∃ (SubSeq : ℕ → ℕ) (r : ℚ),
    (∀ n : ℕ, ∃ k : ℕ, ArithmeticSequence k = SubSeq n) ∧
    (∀ n : ℕ, SubSeq (n + 1) = r * SubSeq n) :=
sorry

end NUMINAMATH_CALUDE_exists_geometric_subsequence_l3562_356291


namespace NUMINAMATH_CALUDE_min_orders_for_given_conditions_l3562_356279

/-- The minimum number of orders required to purchase a given number of items
    while minimizing the total cost under specific discount conditions. -/
def min_orders (original_price : ℚ) (total_items : ℕ) (discount_percent : ℚ) 
                (additional_discount_threshold : ℚ) (additional_discount : ℚ) : ℕ :=
  sorry

/-- The theorem stating that the minimum number of orders is 4 under the given conditions. -/
theorem min_orders_for_given_conditions : 
  min_orders 48 42 0.6 300 100 = 4 := by sorry

end NUMINAMATH_CALUDE_min_orders_for_given_conditions_l3562_356279


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3562_356223

theorem expand_and_simplify (a : ℝ) : 3*a*(2*a^2 - 4*a) - 2*a^2*(3*a + 4) = -20*a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3562_356223


namespace NUMINAMATH_CALUDE_louisa_average_speed_l3562_356259

/-- Proves that given the conditions of Louisa's travel, her average speed was 50 miles per hour -/
theorem louisa_average_speed :
  ∀ (v : ℝ), 
    v > 0 →
    350 / v - 200 / v = 3 →
    v = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_louisa_average_speed_l3562_356259


namespace NUMINAMATH_CALUDE_rectangle_area_l3562_356294

theorem rectangle_area (perimeter : ℝ) (length_ratio width_ratio : ℕ) : 
  perimeter = 280 →
  length_ratio = 5 →
  width_ratio = 2 →
  ∃ (length width : ℝ),
    length / width = length_ratio / width_ratio ∧
    2 * (length + width) = perimeter ∧
    length * width = 4000 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3562_356294


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_squared_l3562_356219

/-- Given complex numbers p, q, and r that are zeros of a cubic polynomial
    and form a right triangle in the complex plane, if the sum of their
    squared magnitudes is 360, then the square of the hypotenuse of the
    triangle is 540. -/
theorem right_triangle_hypotenuse_squared 
  (p q r : ℂ) 
  (h_zeros : ∃ (s t u : ℂ), p^3 + s*p^2 + t*p + u = 0 ∧ 
                             q^3 + s*q^2 + t*q + u = 0 ∧ 
                             r^3 + s*r^2 + t*r + u = 0)
  (h_right_triangle : ∃ (k : ℝ), (Complex.abs (p - q))^2 + (Complex.abs (q - r))^2 = k^2 ∨
                                 (Complex.abs (q - r))^2 + (Complex.abs (r - p))^2 = k^2 ∨
                                 (Complex.abs (r - p))^2 + (Complex.abs (p - q))^2 = k^2)
  (h_sum_squares : Complex.abs p^2 + Complex.abs q^2 + Complex.abs r^2 = 360) :
  ∃ (k : ℝ), k^2 = 540 ∧ 
    ((Complex.abs (p - q))^2 + (Complex.abs (q - r))^2 = k^2 ∨
     (Complex.abs (q - r))^2 + (Complex.abs (r - p))^2 = k^2 ∨
     (Complex.abs (r - p))^2 + (Complex.abs (p - q))^2 = k^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_squared_l3562_356219
