import Mathlib

namespace NUMINAMATH_CALUDE_z_properties_l2921_292149

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := 2 * m + (4 - m^2) * Complex.I

/-- z lies on the imaginary axis -/
def on_imaginary_axis (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- z lies in the first or third quadrant -/
def in_first_or_third_quadrant (z : ℂ) : Prop := z.re * z.im < 0

theorem z_properties (m : ℝ) :
  (on_imaginary_axis (z m) ↔ m = 0) ∧
  (in_first_or_third_quadrant (z m) ↔ m > 2 ∨ (-2 < m ∧ m < 0)) := by
  sorry

end NUMINAMATH_CALUDE_z_properties_l2921_292149


namespace NUMINAMATH_CALUDE_percentage_studying_both_languages_l2921_292194

def english_percentage : ℝ := 90
def german_percentage : ℝ := 80

theorem percentage_studying_both_languages :
  let both_percentage := english_percentage + german_percentage - 100
  both_percentage = 70 := by sorry

end NUMINAMATH_CALUDE_percentage_studying_both_languages_l2921_292194


namespace NUMINAMATH_CALUDE_paint_cost_per_kg_paint_cost_is_36_5_l2921_292160

/-- The cost of paint per kg, given the coverage rate and the cost to paint a cube. -/
theorem paint_cost_per_kg (coverage : ℝ) (cube_side : ℝ) (total_cost : ℝ) : ℝ :=
  let surface_area := 6 * cube_side^2
  let paint_needed := surface_area / coverage
  let cost_per_kg := total_cost / paint_needed
  by
    -- Proof goes here
    sorry

/-- The cost of paint is Rs. 36.5 per kg -/
theorem paint_cost_is_36_5 :
  paint_cost_per_kg 16 8 876 = 36.5 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_paint_cost_per_kg_paint_cost_is_36_5_l2921_292160


namespace NUMINAMATH_CALUDE_park_conditions_l2921_292141

-- Define the conditions for the park
structure ParkConditions where
  temperature : ℝ
  windy : Prop

-- Define when the park is ideal for picnicking
def isIdealForPicnicking (conditions : ParkConditions) : Prop :=
  conditions.temperature ≥ 70 ∧ ¬conditions.windy

-- Theorem statement
theorem park_conditions (conditions : ParkConditions) :
  ¬(isIdealForPicnicking conditions) →
  (conditions.temperature < 70 ∨ conditions.windy) := by
  sorry

end NUMINAMATH_CALUDE_park_conditions_l2921_292141


namespace NUMINAMATH_CALUDE_square_divisibility_l2921_292173

-- Define the divisibility relation
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem square_divisibility (x y : ℕ) : 
  x > 0 → y > 0 → x > y → divides (x * y) (x^2022 + x + y^2) → ∃ n : ℕ, x = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l2921_292173


namespace NUMINAMATH_CALUDE_vegetable_bins_l2921_292126

theorem vegetable_bins (soup_bins pasta_bins total_bins : Real) 
  (h1 : soup_bins = 0.125)
  (h2 : pasta_bins = 0.5)
  (h3 : total_bins = 0.75) :
  total_bins - soup_bins - pasta_bins = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_bins_l2921_292126


namespace NUMINAMATH_CALUDE_ice_melting_problem_l2921_292118

theorem ice_melting_problem (V : ℝ) : 
  V > 0 → 
  ((1 - 3/4) * (1 - 3/4) * V = 0.75) → 
  V = 12 := by
  sorry

end NUMINAMATH_CALUDE_ice_melting_problem_l2921_292118


namespace NUMINAMATH_CALUDE_prob_ace_ten_king_standard_deck_l2921_292170

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (aces : ℕ)
  (tens : ℕ)
  (kings : ℕ)

/-- The probability of drawing an Ace, then a Ten, then a King without replacement -/
def prob_ace_ten_king (d : Deck) : ℚ :=
  (d.aces : ℚ) / d.total_cards *
  (d.tens : ℚ) / (d.total_cards - 1) *
  (d.kings : ℚ) / (d.total_cards - 2)

/-- Theorem stating the probability of drawing an Ace, then a Ten, then a King from a standard deck -/
theorem prob_ace_ten_king_standard_deck : 
  prob_ace_ten_king {total_cards := 52, aces := 4, tens := 4, kings := 4} = 2 / 16575 := by
  sorry


end NUMINAMATH_CALUDE_prob_ace_ten_king_standard_deck_l2921_292170


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2921_292138

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E F : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l2921_292138


namespace NUMINAMATH_CALUDE_min_points_for_isosceles_l2921_292114

/-- Represents a point in the lattice of the triangle --/
structure LatticePoint where
  x : ℝ
  y : ℝ

/-- Represents the regular triangle with its lattice points --/
structure RegularTriangleLattice where
  points : List LatticePoint
  -- Ensure there are exactly 15 lattice points
  point_count : points.length = 15

/-- Checks if three points form an isosceles triangle --/
def isIsosceles (p1 p2 p3 : LatticePoint) : Prop := sorry

/-- The main theorem to be proved --/
theorem min_points_for_isosceles (t : RegularTriangleLattice) :
  ∀ (chosen : List LatticePoint),
    chosen.length ≥ 6 →
    (∀ p, p ∈ chosen → p ∈ t.points) →
    ∃ p1 p2 p3, p1 ∈ chosen ∧ p2 ∈ chosen ∧ p3 ∈ chosen ∧ isIsosceles p1 p2 p3 :=
by sorry

end NUMINAMATH_CALUDE_min_points_for_isosceles_l2921_292114


namespace NUMINAMATH_CALUDE_no_21_length2_segments_in_10x10_grid_l2921_292144

/-- Represents a grid skeleton -/
structure GridSkeleton :=
  (size : ℕ)

/-- Represents the division of a grid skeleton into angle pieces and segments of length 2 -/
structure GridDivision :=
  (grid : GridSkeleton)
  (length2_segments : ℕ)

/-- Theorem stating that a 10x10 grid skeleton cannot have exactly 21 segments of length 2 -/
theorem no_21_length2_segments_in_10x10_grid :
  ∀ (d : GridDivision), d.grid.size = 10 → d.length2_segments ≠ 21 := by
  sorry

end NUMINAMATH_CALUDE_no_21_length2_segments_in_10x10_grid_l2921_292144


namespace NUMINAMATH_CALUDE_boat_speed_difference_l2921_292180

/-- Proves that the boat's speed is 1 km/h greater than the stream current speed --/
theorem boat_speed_difference (V : ℝ) : 
  let S := 1 -- distance in km
  let V₁ := 2*V + 1 -- river current speed in km/h
  let T := 1 -- total time in hours
  ∃ (U : ℝ), -- boat's speed
    U > V ∧ -- boat is faster than stream current
    S / (U - V) - S / (U + V) + S / V₁ = T ∧ -- time equation
    U - V = 1 -- difference in speeds
  := by sorry

end NUMINAMATH_CALUDE_boat_speed_difference_l2921_292180


namespace NUMINAMATH_CALUDE_f_properties_l2921_292102

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 
  Real.sqrt 3 * Real.sin (ω * x + φ) - Real.cos (ω * x + φ)

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def adjacentSymmetryDistance (f : ℝ → ℝ) (d : ℝ) : Prop :=
  ∀ x, f (x + d) = f x

theorem f_properties (ω φ : ℝ) 
  (h_φ : -π/2 < φ ∧ φ < 0) 
  (h_ω : ω > 0) 
  (h_even : isEven (f ω φ))
  (h_symmetry : adjacentSymmetryDistance (f ω φ) (π/2)) :
  f ω φ (π/24) = -(Real.sqrt 6 + Real.sqrt 2)/2 ∧
  ∃ g : ℝ → ℝ, g = fun x ↦ -2 * Real.cos (x/2 - π/3) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l2921_292102


namespace NUMINAMATH_CALUDE_second_number_problem_l2921_292198

theorem second_number_problem (A B C : ℝ) (h_sum : A + B + C = 98) 
  (h_ratio1 : A / B = 2 / 3) (h_ratio2 : B / C = 5 / 8) (h_positive : A > 0 ∧ B > 0 ∧ C > 0) : 
  B = 30 := by
sorry

end NUMINAMATH_CALUDE_second_number_problem_l2921_292198


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2921_292181

theorem contrapositive_equivalence (a b : ℝ) :
  ((a^2 + b^2 = 0) → (a = 0 ∧ b = 0)) ↔ ((a ≠ 0 ∨ b ≠ 0) → (a^2 + b^2 ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2921_292181


namespace NUMINAMATH_CALUDE_selection_theorem_l2921_292100

/-- The number of athletes who can play both basketball and soccer -/
def both_sports (total : ℕ) (basketball : ℕ) (soccer : ℕ) : ℕ :=
  basketball + soccer - total

/-- The number of athletes who can only play basketball -/
def only_basketball (total : ℕ) (basketball : ℕ) (soccer : ℕ) : ℕ :=
  basketball - both_sports total basketball soccer

/-- The number of athletes who can only play soccer -/
def only_soccer (total : ℕ) (basketball : ℕ) (soccer : ℕ) : ℕ :=
  soccer - both_sports total basketball soccer

/-- The number of ways to select two athletes for basketball and soccer -/
def selection_ways (total : ℕ) (basketball : ℕ) (soccer : ℕ) : ℕ :=
  let b := both_sports total basketball soccer
  let ob := only_basketball total basketball soccer
  let os := only_soccer total basketball soccer
  Nat.choose b 2 + b * ob + b * os + ob * os

theorem selection_theorem (total basketball soccer : ℕ) 
  (h1 : total = 9) (h2 : basketball = 5) (h3 : soccer = 6) :
  selection_ways total basketball soccer = 28 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l2921_292100


namespace NUMINAMATH_CALUDE_max_leftover_grapes_l2921_292155

theorem max_leftover_grapes (n : ℕ) : 
  ∃ (q r : ℕ), n = 5 * q + r ∧ r < 5 ∧ r ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_grapes_l2921_292155


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2921_292116

theorem largest_prime_divisor_to_test (n : ℕ) (h : 1900 ≤ n ∧ n ≤ 1950) :
  (∀ p : ℕ, p.Prime → p ≤ 43 → ¬(p ∣ n)) → n.Prime :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2921_292116


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2921_292125

theorem inequality_solution_set (x : ℝ) : 
  (x * (1 - 3 * x) > 0) ↔ (x > 0 ∧ x < 1/3) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2921_292125


namespace NUMINAMATH_CALUDE_cost_of_three_pencils_four_pens_l2921_292190

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a pen -/
def pen_cost : ℝ := sorry

/-- The cost of 8 pencils and 3 pens is $5.60 -/
axiom first_equation : 8 * pencil_cost + 3 * pen_cost = 5.60

/-- The cost of 2 pencils and 5 pens is $4.25 -/
axiom second_equation : 2 * pencil_cost + 5 * pen_cost = 4.25

/-- The cost of 3 pencils and 4 pens is approximately $9.68 -/
theorem cost_of_three_pencils_four_pens :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |3 * pencil_cost + 4 * pen_cost - 9.68| < ε :=
sorry

end NUMINAMATH_CALUDE_cost_of_three_pencils_four_pens_l2921_292190


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2921_292163

/-- Given a real number a and the imaginary unit i, if (2+ai)/(1+i) = 3+i, then a = 4 -/
theorem complex_equation_solution (a : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (2 + a * Complex.I) / (1 + Complex.I) = (3 : ℂ) + Complex.I →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2921_292163


namespace NUMINAMATH_CALUDE_unique_m_satisfying_lcm_conditions_l2921_292165

theorem unique_m_satisfying_lcm_conditions (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 60 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_satisfying_lcm_conditions_l2921_292165


namespace NUMINAMATH_CALUDE_product_inequality_l2921_292188

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  let M := (1 - 1/a) * (1 - 1/b) * (1 - 1/c)
  M ≤ -8 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l2921_292188


namespace NUMINAMATH_CALUDE_students_walking_home_l2921_292130

theorem students_walking_home (bus car bike scooter : ℚ) : 
  bus = 1/2 → car = 1/4 → bike = 1/10 → scooter = 1/8 → 
  1 - (bus + car + bike + scooter) = 1/40 := by
sorry

end NUMINAMATH_CALUDE_students_walking_home_l2921_292130


namespace NUMINAMATH_CALUDE_hall_volume_theorem_l2921_292153

/-- Represents the dimensions of a rectangular hall. -/
structure HallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular hall given its dimensions. -/
def hallVolume (d : HallDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the sum of the areas of the floor and ceiling of a rectangular hall. -/
def floorCeilingArea (d : HallDimensions) : ℝ :=
  2 * d.length * d.width

/-- Calculates the sum of the areas of the four walls of a rectangular hall. -/
def wallsArea (d : HallDimensions) : ℝ :=
  2 * d.height * (d.length + d.width)

/-- Theorem stating the volume of a specific rectangular hall with given conditions. -/
theorem hall_volume_theorem (d : HallDimensions) 
    (h_length : d.length = 15)
    (h_width : d.width = 12)
    (h_area_equality : floorCeilingArea d = wallsArea d) :
    ∃ ε > 0, |hallVolume d - 1201.8| < ε := by
  sorry

end NUMINAMATH_CALUDE_hall_volume_theorem_l2921_292153


namespace NUMINAMATH_CALUDE_profit_loss_ratio_l2921_292164

theorem profit_loss_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_profit_loss_ratio_l2921_292164


namespace NUMINAMATH_CALUDE_will_chocolate_boxes_l2921_292177

theorem will_chocolate_boxes :
  ∀ (boxes_given : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ),
    boxes_given = 3 →
    pieces_per_box = 4 →
    pieces_left = 16 →
    (boxes_given * pieces_per_box + pieces_left) / pieces_per_box = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_will_chocolate_boxes_l2921_292177


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_l2921_292137

noncomputable section

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define point S on PR
def S (t : Triangle) : ℝ × ℝ := sorry

-- Define the lengths
def PR (t : Triangle) : ℝ := sorry
def PQ (t : Triangle) : ℝ := sorry
def QR (t : Triangle) : ℝ := sorry
def PS (t : Triangle) : ℝ := sorry

-- Define the angle bisector property
def bisects_angle_Q (t : Triangle) : Prop := sorry

-- Theorem statement
theorem angle_bisector_theorem (t : Triangle) 
  (h1 : PR t = 72)
  (h2 : PQ t = 32)
  (h3 : QR t = 64)
  (h4 : bisects_angle_Q t) :
  PS t = 24 := by sorry

end

end NUMINAMATH_CALUDE_angle_bisector_theorem_l2921_292137


namespace NUMINAMATH_CALUDE_difference_of_squares_l2921_292111

theorem difference_of_squares (a : ℝ) : (a + 3) * (a - 3) = a^2 - 9 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2921_292111


namespace NUMINAMATH_CALUDE_triangle_area_l2921_292168

/-- A triangle with sides in ratio 3:4:5 and perimeter 60 has area 150 -/
theorem triangle_area (a b c : ℝ) (h_ratio : (a, b, c) = (3, 4, 5)) 
  (h_perimeter : a + b + c = 60) : 
  (1/2) * a * b = 150 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2921_292168


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l2921_292157

/-- Given a total purchase amount, sales tax paid, and tax rate,
    calculate the cost of tax-free items. -/
def cost_of_tax_free_items (total_purchase : ℚ) (sales_tax : ℚ) (tax_rate : ℚ) : ℚ :=
  total_purchase - sales_tax / tax_rate

/-- Theorem stating that given the specific conditions in the problem,
    the cost of tax-free items is 22. -/
theorem tax_free_items_cost :
  let total_purchase : ℚ := 25
  let sales_tax : ℚ := 30 / 100  -- 30 paise = 0.30 rupees
  let tax_rate : ℚ := 10 / 100   -- 10% = 0.10
  cost_of_tax_free_items total_purchase sales_tax tax_rate = 22 := by
  sorry


end NUMINAMATH_CALUDE_tax_free_items_cost_l2921_292157


namespace NUMINAMATH_CALUDE_exponent_rules_l2921_292115

theorem exponent_rules (a : ℝ) : (a^3 * a^2 = a^5) ∧ (a^6 / a^2 = a^4) := by
  sorry

end NUMINAMATH_CALUDE_exponent_rules_l2921_292115


namespace NUMINAMATH_CALUDE_square_sum_power_of_two_l2921_292169

theorem square_sum_power_of_two (x y z : ℕ) (h : x^2 + y^2 = 2^z) :
  ∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2*n + 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_power_of_two_l2921_292169


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2921_292162

theorem quadratic_equation_solution : 
  let x₁ : ℝ := (-1 + Real.sqrt 5) / 2
  let x₂ : ℝ := (-1 - Real.sqrt 5) / 2
  ∀ x : ℝ, x^2 + x - 1 = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2921_292162


namespace NUMINAMATH_CALUDE_ellipse_equation_l2921_292139

/-- An ellipse with parametric equations x = 5cos(α) and y = 3sin(α) has the general equation x²/25 + y²/9 = 1 -/
theorem ellipse_equation (α : ℝ) (x y : ℝ) (h1 : x = 5 * Real.cos α) (h2 : y = 3 * Real.sin α) : 
  x^2 / 25 + y^2 / 9 = 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2921_292139


namespace NUMINAMATH_CALUDE_f_extrema_l2921_292193

def f (x : ℝ) : ℝ := -x^3 + 3*x - 1

theorem f_extrema :
  (∃ x : ℝ, f x = 1 ∧ ∀ y : ℝ, f y ≤ f x) ∧
  (∃ x : ℝ, f x = -3 ∧ ∀ y : ℝ, f y ≥ f x) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l2921_292193


namespace NUMINAMATH_CALUDE_coffee_consumption_theorem_l2921_292117

/-- Represents the relationship between coffee consumption, sleep, and work intensity -/
def coffee_relation (sleep : ℝ) (work : ℝ) (coffee : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ coffee * sleep * work = k

/-- Theorem stating the relationship between coffee consumption on two different days -/
theorem coffee_consumption_theorem (sleep_mon sleep_tue work_mon work_tue coffee_mon : ℝ) :
  sleep_mon = 8 →
  work_mon = 4 →
  coffee_mon = 1 →
  sleep_tue = 5 →
  work_tue = 7 →
  coffee_relation sleep_mon work_mon coffee_mon →
  coffee_relation sleep_tue work_tue ((32 : ℝ) / 35) :=
by sorry

end NUMINAMATH_CALUDE_coffee_consumption_theorem_l2921_292117


namespace NUMINAMATH_CALUDE_assembly_line_arrangements_l2921_292175

theorem assembly_line_arrangements (n : ℕ) (arrangements : ℕ) 
  (h1 : n = 6) 
  (h2 : arrangements = 360) :
  arrangements = n.factorial / 2 := by
sorry

end NUMINAMATH_CALUDE_assembly_line_arrangements_l2921_292175


namespace NUMINAMATH_CALUDE_floor_plus_self_equal_five_l2921_292142

theorem floor_plus_self_equal_five (y : ℝ) : ⌊y⌋ + y = 5 → y = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_equal_five_l2921_292142


namespace NUMINAMATH_CALUDE_coefficient_a2_l2921_292158

theorem coefficient_a2 (x a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (x - 1)^5 + (x - 1)^3 + (x - 1) = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a →
  a₂ = -13 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a2_l2921_292158


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l2921_292112

/-- A linear function y = (m-2)x + m + 1 passes through the first, second, and fourth quadrants
    if and only if -1 < m < m < 2 -/
theorem linear_function_quadrants (m : ℝ) :
  (∀ x y : ℝ, y = (m - 2) * x + m + 1 →
    (y > 0 ∧ x > 0) ∨ (y < 0 ∧ x < 0) ∨ (y < 0 ∧ x > 0)) ↔
  (-1 < m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l2921_292112


namespace NUMINAMATH_CALUDE_correct_verb_form_surround_is_correct_verb_l2921_292156

/-- Represents the grammatical form of a verb --/
inductive VerbForm
| Base
| PresentParticiple
| PastParticiple
| PresentPerfect

/-- Represents the structure of a sentence --/
structure Sentence :=
  (verb : VerbForm)
  (isImperative : Bool)
  (hasFutureTense : Bool)

/-- Determines if a given sentence structure is correct --/
def isCorrectSentenceStructure (s : Sentence) : Prop :=
  s.isImperative ∧ s.hasFutureTense ∧ s.verb = VerbForm.Base

/-- The specific sentence structure in the problem --/
def givenSentence : Sentence :=
  { verb := VerbForm.Base,
    isImperative := true,
    hasFutureTense := true }

/-- Theorem stating that the given sentence structure is correct --/
theorem correct_verb_form :
  isCorrectSentenceStructure givenSentence :=
sorry

/-- Theorem stating that "Surround" is the correct verb to use --/
theorem surround_is_correct_verb :
  givenSentence.verb = VerbForm.Base →
  isCorrectSentenceStructure givenSentence →
  "Surround" = "Surround" :=
sorry

end NUMINAMATH_CALUDE_correct_verb_form_surround_is_correct_verb_l2921_292156


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l2921_292176

theorem sum_of_square_areas : 
  let square1_side : ℝ := 8
  let square2_side : ℝ := 10
  let square1_area : ℝ := square1_side * square1_side
  let square2_area : ℝ := square2_side * square2_side
  square1_area + square2_area = 164 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l2921_292176


namespace NUMINAMATH_CALUDE_equation_solution_l2921_292127

theorem equation_solution (x : ℝ) (hx : x ≠ 0) : 
  (9 * x)^18 = (27 * x)^9 + 81 * x ↔ x = 1/3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2921_292127


namespace NUMINAMATH_CALUDE_shelf_capacity_l2921_292106

/-- The total capacity of jars on a shelf. -/
def total_capacity (total_jars small_jars : ℕ) (small_capacity large_capacity : ℕ) : ℕ :=
  small_jars * small_capacity + (total_jars - small_jars) * large_capacity

/-- Theorem stating the total capacity of jars on the shelf. -/
theorem shelf_capacity : total_capacity 100 62 3 5 = 376 := by
  sorry

end NUMINAMATH_CALUDE_shelf_capacity_l2921_292106


namespace NUMINAMATH_CALUDE_walmart_sales_l2921_292131

theorem walmart_sales (thermometer_price hot_water_bottle_price total_sales : ℕ)
  (thermometer_ratio : ℕ) (h1 : thermometer_price = 2)
  (h2 : hot_water_bottle_price = 6) (h3 : total_sales = 1200)
  (h4 : thermometer_ratio = 7) :
  ∃ (thermometers hot_water_bottles : ℕ),
    thermometer_price * thermometers + hot_water_bottle_price * hot_water_bottles = total_sales ∧
    thermometers = thermometer_ratio * hot_water_bottles ∧
    hot_water_bottles = 60 := by
  sorry

end NUMINAMATH_CALUDE_walmart_sales_l2921_292131


namespace NUMINAMATH_CALUDE_equation_solution_l2921_292129

theorem equation_solution :
  ∃ x : ℝ, (3 / x - 2 / (x - 2) = 0) ∧ (x = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2921_292129


namespace NUMINAMATH_CALUDE_right_triangle_existence_l2921_292191

theorem right_triangle_existence (c h : ℝ) (hc : c > 0) (hh : h > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2 ∧ (a * b) / c = h ↔ h ≤ c / 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l2921_292191


namespace NUMINAMATH_CALUDE_function_composition_l2921_292122

theorem function_composition (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = x^2 + 2*x) →
  f (2*x + 1) = 4*x^2 + 8*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l2921_292122


namespace NUMINAMATH_CALUDE_largest_n_for_arithmetic_sequences_l2921_292143

theorem largest_n_for_arithmetic_sequences (a b : ℕ → ℕ) : 
  (∀ n, ∃ x y : ℤ, a n = 1 + (n - 1) * x ∧ b n = 1 + (n - 1) * y) →  -- arithmetic sequences
  (a 1 = 1 ∧ b 1 = 1) →  -- first terms are 1
  (a 2 ≤ b 2) →  -- a_2 ≤ b_2
  (∃ n, a n * b n = 1540) →  -- product condition
  (∀ n, a n * b n = 1540 → n ≤ 512) ∧  -- 512 is an upper bound
  (∃ n, a n * b n = 1540 ∧ n = 512) -- 512 is achievable
  := by sorry

end NUMINAMATH_CALUDE_largest_n_for_arithmetic_sequences_l2921_292143


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l2921_292178

/-- Given a tetrahedron with two faces of areas S₁ and S₂, sharing a common edge of length a,
    and with a dihedral angle α between these faces, the volume V of the tetrahedron is
    (2 * S₁ * S₂ * sin α) / (3 * a) -/
theorem tetrahedron_volume
  (S₁ S₂ a : ℝ)
  (α : ℝ)
  (h₁ : S₁ > 0)
  (h₂ : S₂ > 0)
  (h₃ : a > 0)
  (h₄ : 0 < α ∧ α < π) :
  ∃ V : ℝ, V = (2 * S₁ * S₂ * Real.sin α) / (3 * a) ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l2921_292178


namespace NUMINAMATH_CALUDE_lines_parallel_perpendicular_l2921_292113

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  l1 : ℝ → ℝ → Prop := λ x y => x + a * y - 2 * a - 2 = 0
  l2 : ℝ → ℝ → Prop := λ x y => a * x + y - 1 - a = 0

/-- Definition of parallel lines -/
def parallel (tl : TwoLines) : Prop :=
  ∃ k : ℝ, ∀ x y, tl.l1 x y ↔ tl.l2 (x + k) y

/-- Definition of perpendicular lines -/
def perpendicular (tl : TwoLines) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ,
    tl.l1 x1 y1 ∧ tl.l2 x2 y2 ∧
    (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) ≠ 0 ∧
    (x2 - x1) * (y2 - y1) = 0

/-- Main theorem -/
theorem lines_parallel_perpendicular (tl : TwoLines) :
  (parallel tl ↔ tl.a = 1) ∧ (perpendicular tl ↔ tl.a = 0) := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_perpendicular_l2921_292113


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2921_292166

theorem cube_volume_problem (a : ℝ) : 
  (a > 0) →
  ((a + 2) * (a - 2) * a = a^3 - 12) →
  (a^3 = 27) := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2921_292166


namespace NUMINAMATH_CALUDE_square_sum_inequality_l2921_292185

theorem square_sum_inequality (x y : ℝ) : x^2 + y^2 + 1 ≥ x + y + x*y := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l2921_292185


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_l2921_292105

theorem brown_eyed_brunettes (total : ℕ) (blue_eyed_blondes : ℕ) (brunettes : ℕ) (brown_eyed : ℕ) :
  total = 60 →
  blue_eyed_blondes = 16 →
  brunettes = 36 →
  brown_eyed = 25 →
  (total - brunettes) - blue_eyed_blondes + brown_eyed = total →
  brown_eyed - ((total - brunettes) - blue_eyed_blondes) = 17 := by
  sorry

#check brown_eyed_brunettes

end NUMINAMATH_CALUDE_brown_eyed_brunettes_l2921_292105


namespace NUMINAMATH_CALUDE_convenience_store_soda_sales_l2921_292146

/-- Represents the weekly soda sales of a convenience store -/
structure SodaSales where
  gallons_per_box : ℕ
  cost_per_box : ℕ
  weekly_syrup_cost : ℕ

/-- Calculates the number of gallons of soda sold per week -/
def gallons_sold_per_week (s : SodaSales) : ℕ :=
  (s.weekly_syrup_cost / s.cost_per_box) * s.gallons_per_box

/-- Theorem: Given the conditions, the store sells 180 gallons of soda per week -/
theorem convenience_store_soda_sales :
  ∀ (s : SodaSales),
    s.gallons_per_box = 30 →
    s.cost_per_box = 40 →
    s.weekly_syrup_cost = 240 →
    gallons_sold_per_week s = 180 := by
  sorry

end NUMINAMATH_CALUDE_convenience_store_soda_sales_l2921_292146


namespace NUMINAMATH_CALUDE_midpoint_trajectory_equation_l2921_292123

/-- The equation of the trajectory of the midpoint M of line segment PQ, where P is on the parabola y = x^2 + 1 and Q is (0, 1) -/
theorem midpoint_trajectory_equation :
  ∀ (x y a b : ℝ),
  y = x^2 + 1 →  -- P (x, y) is on the parabola y = x^2 + 1
  a = x / 2 →    -- M (a, b) is the midpoint of PQ, so a = x/2
  b = (y + 1) / 2 →  -- and b = (y + 1)/2
  b = 2 * a^2 + 1 :=  -- The equation of the trajectory of M is y = 2x^2 + 1
by
  sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_equation_l2921_292123


namespace NUMINAMATH_CALUDE_batsman_average_increase_l2921_292124

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored : ℚ) / (stats.innings + 1)

/-- Theorem: If a batsman's average increases by 3 after scoring 84 in the 17th inning, the new average is 36 -/
theorem batsman_average_increase (stats : BatsmanStats) 
    (h1 : stats.innings = 16)
    (h2 : newAverage stats 84 = stats.average + 3) :
    newAverage stats 84 = 36 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_increase_l2921_292124


namespace NUMINAMATH_CALUDE_characterization_of_finite_sets_l2921_292136

def ClosedUnderAbsoluteSum (X : Set ℝ) : Prop :=
  ∀ x ∈ X, x + |x| ∈ X

theorem characterization_of_finite_sets (X : Set ℝ) 
  (h_nonempty : X.Nonempty) (h_finite : X.Finite) (h_closed : ClosedUnderAbsoluteSum X) :
  ∃ F : Set ℝ, F.Finite ∧ (∀ x ∈ F, x < 0) ∧ X = F ∪ {0} :=
sorry

end NUMINAMATH_CALUDE_characterization_of_finite_sets_l2921_292136


namespace NUMINAMATH_CALUDE_tina_career_difference_l2921_292183

def boxing_career (initial_wins : ℕ) (second_wins : ℕ) (third_wins : ℕ) (fourth_wins : ℕ) : ℕ := 
  let wins1 := initial_wins + second_wins
  let wins2 := wins1 * 3
  let wins3 := wins2 + third_wins
  let wins4 := wins3 * 2
  let wins5 := wins4 + fourth_wins
  wins5 * wins5

theorem tina_career_difference : 
  boxing_career 10 5 7 11 - 4 = 13221 := by sorry

end NUMINAMATH_CALUDE_tina_career_difference_l2921_292183


namespace NUMINAMATH_CALUDE_union_and_intersection_when_a_eq_4_subset_condition_l2921_292140

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x ≤ 4}

-- Define set B with parameter a
def B (a : ℝ) : Set ℝ := {x | 6 - a < x ∧ x < 2*a - 1}

-- Theorem for question 1
theorem union_and_intersection_when_a_eq_4 :
  (A ∪ B 4) = {x | 1 < x ∧ x < 7} ∧
  (B 4 ∩ (U \ A)) = {x | 4 < x ∧ x < 7} :=
sorry

-- Theorem for question 2
theorem subset_condition :
  ∀ a : ℝ, A ⊆ B a ↔ a ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_union_and_intersection_when_a_eq_4_subset_condition_l2921_292140


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2921_292179

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2921_292179


namespace NUMINAMATH_CALUDE_defective_probability_l2921_292186

theorem defective_probability (total_output : ℝ) : 
  let machine_a_output := 0.40 * total_output
  let machine_b_output := 0.35 * total_output
  let machine_c_output := 0.25 * total_output
  let machine_a_defective_rate := 14 / 2000
  let machine_b_defective_rate := 9 / 1500
  let machine_c_defective_rate := 7 / 1000
  let total_defective := 
    machine_a_defective_rate * machine_a_output +
    machine_b_defective_rate * machine_b_output +
    machine_c_defective_rate * machine_c_output
  total_defective / total_output = 0.00665 := by
sorry

end NUMINAMATH_CALUDE_defective_probability_l2921_292186


namespace NUMINAMATH_CALUDE_exact_three_ones_between_zeros_l2921_292154

/-- A sequence of 10 elements consisting of 8 ones and 2 zeros -/
def Sequence := Fin 10 → Fin 2

/-- The number of sequences with exactly three ones between two zeros -/
def favorable_sequences : ℕ := 12

/-- The total number of possible sequences -/
def total_sequences : ℕ := Nat.choose 10 2

/-- The probability of having exactly three ones between two zeros -/
def probability : ℚ := favorable_sequences / total_sequences

theorem exact_three_ones_between_zeros :
  probability = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_exact_three_ones_between_zeros_l2921_292154


namespace NUMINAMATH_CALUDE_probability_second_black_given_first_black_l2921_292192

/-- A bag of balls with white and black colors -/
structure BallBag where
  white : ℕ
  black : ℕ

/-- The probability of drawing a specific color ball given the current state of the bag -/
def drawProbability (bag : BallBag) (isBlack : Bool) : ℚ :=
  if isBlack then
    bag.black / (bag.white + bag.black)
  else
    bag.white / (bag.white + bag.black)

/-- The probability of drawing a black ball in the second draw given a black ball was drawn first -/
def secondBlackGivenFirstBlack (initialBag : BallBag) : ℚ :=
  let bagAfterFirstDraw := BallBag.mk initialBag.white (initialBag.black - 1)
  drawProbability bagAfterFirstDraw true

theorem probability_second_black_given_first_black :
  let initialBag := BallBag.mk 3 2
  secondBlackGivenFirstBlack initialBag = 1/4 := by
  sorry

#eval secondBlackGivenFirstBlack (BallBag.mk 3 2)

end NUMINAMATH_CALUDE_probability_second_black_given_first_black_l2921_292192


namespace NUMINAMATH_CALUDE_residue_of_7_500_mod_19_l2921_292174

theorem residue_of_7_500_mod_19 : 7^500 % 19 = 15 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_7_500_mod_19_l2921_292174


namespace NUMINAMATH_CALUDE_price_two_birdhouses_is_32_l2921_292107

/-- The price Denver charges for two birdhouses -/
def price_two_birdhouses : ℚ :=
  let pieces_per_birdhouse : ℕ := 7
  let price_per_piece : ℚ := 3/2  -- $1.50 as a rational number
  let profit_per_birdhouse : ℚ := 11/2  -- $5.50 as a rational number
  let cost_per_birdhouse : ℚ := pieces_per_birdhouse * price_per_piece
  let price_per_birdhouse : ℚ := cost_per_birdhouse + profit_per_birdhouse
  2 * price_per_birdhouse

/-- Theorem stating that the price for two birdhouses is $32.00 -/
theorem price_two_birdhouses_is_32 : price_two_birdhouses = 32 := by
  sorry

end NUMINAMATH_CALUDE_price_two_birdhouses_is_32_l2921_292107


namespace NUMINAMATH_CALUDE_root_difference_l2921_292199

-- Define the equations
def equation1 (x : ℝ) : Prop := 2002^2 * x^2 - 2003 * 2001 * x - 1 = 0
def equation2 (x : ℝ) : Prop := 2001 * x^2 - 2002 * x + 1 = 0

-- Define r and s
def r : ℝ := sorry
def s : ℝ := sorry

-- State the theorem
theorem root_difference : 
  (equation1 r ∧ ∀ x, equation1 x → x ≤ r) ∧ 
  (equation2 s ∧ ∀ x, equation2 x → x ≥ s) → 
  r - s = 2000 / 2001 := by sorry

end NUMINAMATH_CALUDE_root_difference_l2921_292199


namespace NUMINAMATH_CALUDE_otimes_example_otimes_sum_property_l2921_292171

-- Define the custom operation
def otimes (a b : ℝ) : ℝ := a * (1 - b)

-- Theorem 1
theorem otimes_example : otimes 2 (-2) = 6 := by sorry

-- Theorem 2
theorem otimes_sum_property (a b : ℝ) (h : a + b = 0) : 
  (otimes a a) + (otimes b b) = 2 * a * b := by sorry

end NUMINAMATH_CALUDE_otimes_example_otimes_sum_property_l2921_292171


namespace NUMINAMATH_CALUDE_circle_xy_bounds_l2921_292128

/-- The circle defined by x² + y² - 4x - 4y + 6 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 4*p.2 + 6 = 0}

/-- The product function xy for points on the circle -/
def xy_product (p : ℝ × ℝ) : ℝ := p.1 * p.2

theorem circle_xy_bounds :
  (∃ p ∈ Circle, ∀ q ∈ Circle, xy_product q ≤ xy_product p) ∧
  (∃ p ∈ Circle, ∀ q ∈ Circle, xy_product p ≤ xy_product q) ∧
  (∃ p ∈ Circle, xy_product p = 9) ∧
  (∃ p ∈ Circle, xy_product p = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_xy_bounds_l2921_292128


namespace NUMINAMATH_CALUDE_point_comparison_l2921_292196

/-- Given that points (-2, y₁) and (-1, y₂) lie on the line y = -3x + b, prove that y₁ > y₂ -/
theorem point_comparison (b : ℝ) (y₁ y₂ : ℝ) 
  (h₁ : y₁ = -3 * (-2) + b) 
  (h₂ : y₂ = -3 * (-1) + b) : 
  y₁ > y₂ := by
  sorry


end NUMINAMATH_CALUDE_point_comparison_l2921_292196


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2921_292159

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 square units -/
theorem isosceles_right_triangle_area (h : ℝ) (a : ℝ) 
  (hyp_length : h = 6 * Real.sqrt 2)
  (isosceles_right : a = h / Real.sqrt 2) : a * a / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2921_292159


namespace NUMINAMATH_CALUDE_geometric_series_remainder_remainder_of_series_l2921_292195

theorem geometric_series_remainder (n : ℕ) (a r : ℤ) (m : ℕ) (h : m > 0) :
  (a * (r^n - 1) / (r - 1)) % m = 
  ((a * (r^n % (m * (r - 1))) - a) / (r - 1)) % m :=
sorry

theorem remainder_of_series : 
  (((3^1002 - 1) / 2) % 500) = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_remainder_remainder_of_series_l2921_292195


namespace NUMINAMATH_CALUDE_tempo_premium_calculation_l2921_292152

/-- Calculate the premium amount for an insured tempo --/
theorem tempo_premium_calculation (original_value : ℝ) (insurance_ratio : ℝ) (premium_rate : ℝ) :
  original_value = 87500 →
  insurance_ratio = 4/5 →
  premium_rate = 0.013 →
  (original_value * insurance_ratio * premium_rate : ℝ) = 910 := by
  sorry

end NUMINAMATH_CALUDE_tempo_premium_calculation_l2921_292152


namespace NUMINAMATH_CALUDE_diamond_3_7_l2921_292189

-- Define the star operation
def star (a b : ℕ) : ℕ := a^2 + 2*a*b + b^2

-- Define the diamond operation
def diamond (a b : ℕ) : ℕ := star a b - a*b

-- Theorem to prove
theorem diamond_3_7 : diamond 3 7 = 79 := by
  sorry

end NUMINAMATH_CALUDE_diamond_3_7_l2921_292189


namespace NUMINAMATH_CALUDE_number_of_possible_D_values_l2921_292132

-- Define the type for digits (0-9)
def Digit := Fin 10

-- Define the addition operation
def add (a b : Digit) : ℕ := a.val + b.val

-- Define the property of being distinct
def distinct (a b c d : Digit) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Define the main theorem
theorem number_of_possible_D_values :
  ∃ (s : Finset Digit),
    (∀ d ∈ s, ∃ (a b c e : Digit),
      distinct a b c d ∧
      add a b = d.val ∧
      add c e = d.val) ∧
    s.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_possible_D_values_l2921_292132


namespace NUMINAMATH_CALUDE_triangle_angle_and_area_l2921_292121

/-- Given a triangle ABC with angle A and vectors m and n, prove the measure of A and the area of the triangle -/
theorem triangle_angle_and_area 
  (A B C : ℝ) 
  (m n : ℝ × ℝ) 
  (h1 : m = (Real.sin (A/2), Real.cos (A/2)))
  (h2 : n = (Real.cos (A/2), -Real.cos (A/2)))
  (h3 : 2 * (m.1 * n.1 + m.2 * n.2) + Real.sqrt (m.1^2 + m.2^2) = Real.sqrt 2 / 2)
  (h4 : Real.cos A = 1 / (Real.sin A)) :
  A = 5 * Real.pi / 12 ∧ 
  (Real.sin A) / 2 = (2 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_and_area_l2921_292121


namespace NUMINAMATH_CALUDE_third_power_four_five_l2921_292148

theorem third_power_four_five (x y : ℚ) : 
  x = 5/6 → y = 6/5 → (1/3) * x^4 * y^5 = 44/111 := by
  sorry

end NUMINAMATH_CALUDE_third_power_four_five_l2921_292148


namespace NUMINAMATH_CALUDE_max_matches_theorem_l2921_292108

/-- The maximum number of matches in a table tennis tournament -/
def max_matches : ℕ := 120

/-- Represents the number of players in each team -/
structure TeamSizes where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the total number of matches given team sizes -/
def calculate_matches (teams : TeamSizes) : ℕ :=
  teams.x * teams.y + teams.y * teams.z + teams.x * teams.z

/-- Theorem stating the maximum number of matches -/
theorem max_matches_theorem :
  ∀ (teams : TeamSizes),
  teams.x + teams.y + teams.z = 19 →
  calculate_matches teams ≤ max_matches :=
by sorry

end NUMINAMATH_CALUDE_max_matches_theorem_l2921_292108


namespace NUMINAMATH_CALUDE_only_prime_with_alternating_base14_rep_l2921_292161

/-- Represents a number in base-14 with alternating 1s and 0s -/
def alternatingBaseRepresentation (n : ℕ) : ℕ :=
  (14^(2*n+2) - 1) / (14^2 - 1)

/-- Checks if a number is prime -/
def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

theorem only_prime_with_alternating_base14_rep :
  ∃! p : ℕ, isPrime p ∧ ∃ n : ℕ, alternatingBaseRepresentation n = p :=
by
  -- The unique prime is 197
  use 197
  sorry -- Proof omitted

#eval alternatingBaseRepresentation 1  -- Should evaluate to 197

end NUMINAMATH_CALUDE_only_prime_with_alternating_base14_rep_l2921_292161


namespace NUMINAMATH_CALUDE_sprinkles_problem_l2921_292103

theorem sprinkles_problem (initial_cans remaining_cans subtracted_number : ℕ) : 
  initial_cans = 12 →
  remaining_cans = 3 →
  remaining_cans = initial_cans / 2 - subtracted_number →
  subtracted_number = 3 := by
  sorry

end NUMINAMATH_CALUDE_sprinkles_problem_l2921_292103


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2921_292172

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  a / (Real.sin A) = c / (Real.sin C) ∧
  a = 3 ∧ b = Real.sqrt 3 ∧ A = π / 3 →
  B = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2921_292172


namespace NUMINAMATH_CALUDE_complex_product_modulus_l2921_292104

theorem complex_product_modulus : Complex.abs (4 - 5 * Complex.I) * Complex.abs (4 + 5 * Complex.I) = 41 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_modulus_l2921_292104


namespace NUMINAMATH_CALUDE_sequence_inequality_l2921_292187

def is_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

def not_in_sequence (a : ℕ → ℕ) (x : ℕ) : Prop :=
  ∀ n : ℕ, a n ≠ x

def representable (a : ℕ → ℕ) : Prop :=
  ∀ x : ℕ, not_in_sequence a x → ∃ k : ℕ, x = a k + 2 * k

theorem sequence_inequality (a : ℕ → ℕ) 
  (h1 : is_increasing a) 
  (h2 : representable a) : 
  ∀ k : ℕ, (a k : ℝ) < Real.sqrt (2 * k) := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2921_292187


namespace NUMINAMATH_CALUDE_base5_to_base7_conversion_l2921_292184

/-- Converts a number from base 5 to decimal --/
def base5ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from decimal to base 7 --/
def decimalToBase7 (n : ℕ) : ℕ := sorry

theorem base5_to_base7_conversion :
  decimalToBase7 (base5ToDecimal 412) = 212 := by sorry

end NUMINAMATH_CALUDE_base5_to_base7_conversion_l2921_292184


namespace NUMINAMATH_CALUDE_equal_roots_condition_l2921_292101

theorem equal_roots_condition (m : ℝ) : 
  (∃ (x : ℝ), (x * (x - 1) - (m + 3)) / ((x - 1) * (m - 1)) = x / m ∧ 
   (∀ (y : ℝ), (y * (y - 1) - (m + 3)) / ((y - 1) * (m - 1)) = y / m → y = x)) ↔ 
  (m = -1.5 + Real.sqrt 2 ∨ m = -1.5 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l2921_292101


namespace NUMINAMATH_CALUDE_distance_ratio_of_cars_l2921_292151

-- Define the speeds and travel times for both cars
def speed_A : ℝ := 50
def time_A : ℝ := 8
def speed_B : ℝ := 25
def time_B : ℝ := 4

-- Define a function to calculate distance
def distance (speed time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem distance_ratio_of_cars :
  (distance speed_A time_A) / (distance speed_B time_B) = 4 := by
  sorry


end NUMINAMATH_CALUDE_distance_ratio_of_cars_l2921_292151


namespace NUMINAMATH_CALUDE_cookie_jar_problem_l2921_292110

/-- Represents the number of raisins in the larger cookie -/
def larger_cookie_raisins : ℕ := 12

/-- Represents the total number of raisins in the jar -/
def total_raisins : ℕ := 100

/-- Represents the range of cookies in the jar -/
def cookie_range : Set ℕ := {n | 5 ≤ n ∧ n ≤ 10}

theorem cookie_jar_problem (n : ℕ) (h_n : n ∈ cookie_range) :
  ∃ (r : ℕ),
    r + 1 = larger_cookie_raisins ∧
    (n - 1) * r + (r + 1) = total_raisins :=
sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l2921_292110


namespace NUMINAMATH_CALUDE_missing_number_implies_next_prime_l2921_292150

/-- Definition of the table entry function -/
def table_entry (r s : ℕ) : ℕ := r * s - (r + s)

/-- Theorem: If n > 3 is not in the table, then n + 1 is prime -/
theorem missing_number_implies_next_prime (n : ℕ) (h1 : n > 3) 
  (h2 : ∀ r s, r ≥ 3 → s ≥ 3 → table_entry r s ≠ n) : 
  Nat.Prime (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_missing_number_implies_next_prime_l2921_292150


namespace NUMINAMATH_CALUDE_calculation_proof_l2921_292135

theorem calculation_proof : Real.sqrt 2 * Real.sqrt 2 - 4 * Real.sin (π / 6) + (1 / 2)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2921_292135


namespace NUMINAMATH_CALUDE_notebook_cost_l2921_292119

theorem notebook_cost (total_students : Nat) (buyers : Nat) (total_cost : ℚ) 
  (h1 : total_students = 36)
  (h2 : buyers > total_students / 2)
  (h3 : ∃ (notebooks_per_student : Nat) (cost_per_notebook : ℚ),
    notebooks_per_student > 0 ∧
    cost_per_notebook > notebooks_per_student ∧
    buyers * notebooks_per_student * cost_per_notebook = total_cost)
  (h4 : total_cost = 2664 / 100) :
  ∃ (notebooks_per_student : Nat) (cost_per_notebook : ℚ),
    notebooks_per_student > 0 ∧
    cost_per_notebook > notebooks_per_student ∧
    buyers * notebooks_per_student * cost_per_notebook = total_cost ∧
    cost_per_notebook = 37 / 100 :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l2921_292119


namespace NUMINAMATH_CALUDE_sum_two_angles_gt_90_implies_acute_l2921_292133

-- Define a triangle type
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_180 : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the property of sum of any two angles greater than 90°
def sum_two_angles_gt_90 (t : Triangle) : Prop :=
  t.A + t.B > 90 ∧ t.B + t.C > 90 ∧ t.C + t.A > 90

-- Define an acute triangle
def is_acute_triangle (t : Triangle) : Prop :=
  t.A < 90 ∧ t.B < 90 ∧ t.C < 90

-- Theorem statement
theorem sum_two_angles_gt_90_implies_acute (t : Triangle) :
  sum_two_angles_gt_90 t → is_acute_triangle t :=
by sorry

end NUMINAMATH_CALUDE_sum_two_angles_gt_90_implies_acute_l2921_292133


namespace NUMINAMATH_CALUDE_mint_problem_solvable_l2921_292109

/-- Represents a set of coin denominations. -/
def CoinSet := Finset ℕ

/-- Checks if a given amount can be represented using at most 8 coins from the set. -/
def canRepresent (coins : CoinSet) (amount : ℕ) : Prop :=
  ∃ (representation : Finset ℕ), 
    representation.card ≤ 8 ∧ 
    (representation.sum (λ x => x * (coins.filter (λ c => c = x)).card)) = amount

/-- The main theorem stating that there exists a set of 12 coin denominations
    that can represent all amounts from 1 to 6543 using at most 8 coins. -/
theorem mint_problem_solvable : 
  ∃ (coins : CoinSet), 
    coins.card = 12 ∧ 
    ∀ (amount : ℕ), 1 ≤ amount ∧ amount ≤ 6543 → canRepresent coins amount :=
by
  sorry


end NUMINAMATH_CALUDE_mint_problem_solvable_l2921_292109


namespace NUMINAMATH_CALUDE_marie_erasers_l2921_292145

theorem marie_erasers (initial final lost : ℕ) 
  (h1 : lost = 42)
  (h2 : final = 53)
  (h3 : initial = final + lost) : initial = 95 := by
  sorry

end NUMINAMATH_CALUDE_marie_erasers_l2921_292145


namespace NUMINAMATH_CALUDE_tree_planting_correct_l2921_292134

/-- Represents the number of trees each person should plant in different scenarios -/
structure TreePlanting where
  average : ℝ  -- Average number of trees per person for the whole class
  female : ℝ   -- Number of trees per person if only females plant
  male : ℝ     -- Number of trees per person if only males plant

/-- The tree planting scenario for the ninth-grade class -/
def class_planting : TreePlanting :=
  { average := 6
  , female := 15
  , male := 10 }

/-- Theorem stating that the given values satisfy the tree planting scenario -/
theorem tree_planting_correct (tp : TreePlanting) (h : tp = class_planting) :
  1 / tp.male + 1 / tp.female = 1 / tp.average :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_correct_l2921_292134


namespace NUMINAMATH_CALUDE_probability_twelve_rolls_last_l2921_292197

/-- The probability of getting the same number on the 12th roll as on the 11th roll,
    given that all previous pairs of consecutive rolls were different. -/
theorem probability_twelve_rolls_last (d : ℕ) (h : d = 6) : 
  (((d - 1) / d) ^ 10 * (1 / d) : ℚ) = 9765625 / 362797056 := by
  sorry

end NUMINAMATH_CALUDE_probability_twelve_rolls_last_l2921_292197


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2921_292167

theorem fraction_to_decimal : (58 : ℚ) / 160 = (3625 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2921_292167


namespace NUMINAMATH_CALUDE_right_triangle_area_l2921_292182

/-- A right triangle with vertices A(0, 0), B(0, 5), and C(3, 0) has an area of 7.5 square units. -/
theorem right_triangle_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 5)
  let C : ℝ × ℝ := (3, 0)
  let triangle_area := (1/2) * 3 * 5
  triangle_area = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2921_292182


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_x_l2921_292120

theorem gcd_polynomial_and_x (x : ℤ) (h : ∃ k : ℤ, x = 23478 * k) :
  Int.gcd ((2*x+3)*(7*x+2)*(13*x+7)*(x+13)) x = 546 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_x_l2921_292120


namespace NUMINAMATH_CALUDE_derivative_of_f_l2921_292147

-- Define the function f(x) = (5x - 4)^3
def f (x : ℝ) : ℝ := (5 * x - 4) ^ 3

-- State the theorem that the derivative of f(x) is 15(5x - 4)^2
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 15 * (5 * x - 4) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l2921_292147
