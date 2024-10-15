import Mathlib

namespace NUMINAMATH_CALUDE_arrangement_counts_l595_59531

/-- The number of boys in the arrangement -/
def num_boys : ℕ := 3

/-- The number of girls in the arrangement -/
def num_girls : ℕ := 4

/-- The total number of people to be arranged -/
def total_people : ℕ := num_boys + num_girls

/-- Calculates the number of arrangements with Person A and Person B at the ends -/
def arrangements_ends : ℕ := sorry

/-- Calculates the number of arrangements with all boys standing together -/
def arrangements_boys_together : ℕ := sorry

/-- Calculates the number of arrangements with no two boys standing next to each other -/
def arrangements_boys_separated : ℕ := sorry

/-- Calculates the number of arrangements with exactly one person between Person A and Person B -/
def arrangements_one_between : ℕ := sorry

theorem arrangement_counts :
  arrangements_ends = 240 ∧
  arrangements_boys_together = 720 ∧
  arrangements_boys_separated = 1440 ∧
  arrangements_one_between = 1200 := by sorry

end NUMINAMATH_CALUDE_arrangement_counts_l595_59531


namespace NUMINAMATH_CALUDE_fewer_onions_l595_59508

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

theorem fewer_onions : (tomatoes + corn) - onions = 5200 := by
  sorry

end NUMINAMATH_CALUDE_fewer_onions_l595_59508


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l595_59515

theorem complex_fraction_calculation (z : ℂ) (h : z = 1 - I) : z^2 / (z - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l595_59515


namespace NUMINAMATH_CALUDE_max_rectangular_pen_area_l595_59530

/-- Given 60 feet of fencing, the maximum area of a rectangular pen is 225 square feet. -/
theorem max_rectangular_pen_area :
  ∀ w h : ℝ,
  w > 0 → h > 0 →
  2 * w + 2 * h = 60 →
  w * h ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangular_pen_area_l595_59530


namespace NUMINAMATH_CALUDE_original_selling_price_l595_59562

/-- Given an article with a cost price of $15000, prove that the original selling price
    that would result in an 8% profit if discounted by 10% is $18000. -/
theorem original_selling_price (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  cost_price = 15000 →
  discount_rate = 0.1 →
  profit_rate = 0.08 →
  ∃ (selling_price : ℝ),
    selling_price * (1 - discount_rate) = cost_price * (1 + profit_rate) ∧
    selling_price = 18000 := by
  sorry

end NUMINAMATH_CALUDE_original_selling_price_l595_59562


namespace NUMINAMATH_CALUDE_parents_age_at_marks_birth_l595_59549

/-- The age of Mark and John's parents when Mark was born, given their current ages and age differences. -/
theorem parents_age_at_marks_birth (mark_age john_age_diff parents_age_multiplier : ℕ) : 
  mark_age = 18 → 
  john_age_diff = 10 → 
  parents_age_multiplier = 5 → 
  (mark_age - john_age_diff) * parents_age_multiplier - mark_age = 22 := by
sorry

end NUMINAMATH_CALUDE_parents_age_at_marks_birth_l595_59549


namespace NUMINAMATH_CALUDE_zero_exponent_l595_59567

theorem zero_exponent (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry

end NUMINAMATH_CALUDE_zero_exponent_l595_59567


namespace NUMINAMATH_CALUDE_ketchup_mustard_arrangement_l595_59544

/-- Represents the number of ways to arrange ketchup and mustard bottles. -/
def arrange_bottles (k m : ℕ) : ℕ := sorry

/-- The property that no ketchup bottle is between two mustard bottles. -/
def valid_arrangement (k m : ℕ) (arrangement : List Bool) : Prop := sorry

/-- The main theorem stating the number of valid arrangements. -/
theorem ketchup_mustard_arrangement :
  ∃ (n : ℕ), 
    (n = arrange_bottles 3 7) ∧ 
    (∀ arrangement : List Bool, 
      (arrangement.length = 10) →
      (arrangement.count true = 3) →
      (arrangement.count false = 7) →
      valid_arrangement 3 7 arrangement) ∧
    n = 22 := by sorry

end NUMINAMATH_CALUDE_ketchup_mustard_arrangement_l595_59544


namespace NUMINAMATH_CALUDE_rohan_investment_is_8040_l595_59529

/-- Represents the investment scenario described in the problem -/
structure InvestmentScenario where
  suresh_investment : ℕ
  suresh_months : ℕ
  rohan_months : ℕ
  sudhir_investment : ℕ
  sudhir_months : ℕ
  total_profit : ℕ
  rohan_sudhir_diff : ℕ

/-- Calculates Rohan's investment based on the given scenario -/
def calculate_rohan_investment (scenario : InvestmentScenario) : ℕ :=
  sorry

/-- Theorem stating that Rohan's investment is 8040 given the specific scenario -/
theorem rohan_investment_is_8040 : 
  let scenario : InvestmentScenario := {
    suresh_investment := 18000,
    suresh_months := 12,
    rohan_months := 9,
    sudhir_investment := 9000,
    sudhir_months := 8,
    total_profit := 3872,
    rohan_sudhir_diff := 352
  }
  calculate_rohan_investment scenario = 8040 := by
  sorry

end NUMINAMATH_CALUDE_rohan_investment_is_8040_l595_59529


namespace NUMINAMATH_CALUDE_tricycle_wheel_revolutions_l595_59527

/-- Calculates the number of revolutions of the back wheel of a tricycle -/
theorem tricycle_wheel_revolutions (front_radius back_radius : ℝ) (front_revolutions : ℕ) : 
  front_radius = 3 →
  back_radius = 1/2 →
  front_revolutions = 50 →
  (2 * π * front_radius * front_revolutions) / (2 * π * back_radius) = 300 := by
  sorry

#check tricycle_wheel_revolutions

end NUMINAMATH_CALUDE_tricycle_wheel_revolutions_l595_59527


namespace NUMINAMATH_CALUDE_volume_removed_tetrahedra_l595_59599

/-- The volume of tetrahedra removed from a cube with specified edge divisions -/
theorem volume_removed_tetrahedra (edge_length : ℝ) (h_edge : edge_length = 2) :
  let central_segment : ℝ := 1
  let slanted_segment : ℝ := 1 / 2
  let height : ℝ := edge_length - central_segment / Real.sqrt 2
  let base_area : ℝ := 1 / 8
  let tetrahedron_volume : ℝ := 1 / 3 * base_area * height
  let total_volume : ℝ := 8 * tetrahedron_volume
  total_volume = 4 / 3 - Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_removed_tetrahedra_l595_59599


namespace NUMINAMATH_CALUDE_total_athletes_count_l595_59509

/-- Represents the number of athletes in the sports meeting -/
structure AthleteCount where
  male : ℕ
  female : ℕ

/-- The ratio of male to female athletes at different stages -/
def initial_ratio : Rat := 19 / 12
def after_gymnastics_ratio : Rat := 20 / 13
def final_ratio : Rat := 30 / 19

/-- The difference between added male chess players and female gymnasts -/
def extra_male_players : ℕ := 30

theorem total_athletes_count (initial : AthleteCount) 
  (h1 : initial.male / initial.female = initial_ratio)
  (h2 : (initial.male / (initial.female + extra_male_players)) = after_gymnastics_ratio)
  (h3 : ((initial.male + extra_male_players) / (initial.female + extra_male_players)) = final_ratio)
  : initial.male + initial.female + 2 * extra_male_players = 6370 := by
  sorry

#check total_athletes_count

end NUMINAMATH_CALUDE_total_athletes_count_l595_59509


namespace NUMINAMATH_CALUDE_missing_element_is_five_l595_59581

/-- Represents a 2x2 matrix --/
structure Matrix2x2 where
  a11 : ℤ
  a12 : ℤ
  a21 : ℤ
  a22 : ℤ

/-- Calculates the sum of diagonal products for a 2x2 matrix --/
def diagonalProductSum (m : Matrix2x2) : ℤ :=
  m.a11 * m.a22 + m.a12 * m.a21

/-- Theorem stating that for a matrix with given conditions, the missing element must be 5 --/
theorem missing_element_is_five (m : Matrix2x2) 
  (h1 : m.a11 = 2)
  (h2 : m.a12 = 6)
  (h3 : m.a21 = 1)
  (h4 : diagonalProductSum m = 16) :
  m.a22 = 5 := by
  sorry


end NUMINAMATH_CALUDE_missing_element_is_five_l595_59581


namespace NUMINAMATH_CALUDE_yellow_balls_count_l595_59580

theorem yellow_balls_count (total : ℕ) (red blue green yellow : ℕ) : 
  total = 500 ∧ 
  red = (total / 3 : ℕ) ∧ 
  blue = ((total - red) / 5 : ℕ) ∧ 
  green = ((total - red - blue) / 4 : ℕ) ∧ 
  yellow = total - red - blue - green →
  yellow = 201 := by
sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l595_59580


namespace NUMINAMATH_CALUDE_division_problem_l595_59558

theorem division_problem (dividend quotient remainder : ℕ) (h1 : dividend = 131) (h2 : quotient = 9) (h3 : remainder = 5) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 14 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l595_59558


namespace NUMINAMATH_CALUDE_a_profit_share_l595_59573

/-- Calculates the share of profit for partner A in a business partnership --/
def calculate_profit_share (initial_a initial_b : ℕ) (withdraw_a advance_b : ℕ) (months : ℕ) (total_profit : ℕ) : ℕ :=
  let investment_months_a := initial_a * months + (initial_a - withdraw_a) * (12 - months)
  let investment_months_b := initial_b * months + (initial_b + advance_b) * (12 - months)
  let total_investment_months := investment_months_a + investment_months_b
  (investment_months_a * total_profit) / total_investment_months

/-- Theorem stating that A's share of the profit is 357 given the problem conditions --/
theorem a_profit_share :
  calculate_profit_share 6000 4000 1000 1000 8 630 = 357 :=
by
  sorry

#eval calculate_profit_share 6000 4000 1000 1000 8 630

end NUMINAMATH_CALUDE_a_profit_share_l595_59573


namespace NUMINAMATH_CALUDE_tire_sample_size_l595_59593

theorem tire_sample_size (p : ℝ) (n : ℕ) (h1 : p = 0.015) (h2 : n = 168) :
  1 - (1 - p) ^ n > 0.92 := by
  sorry

end NUMINAMATH_CALUDE_tire_sample_size_l595_59593


namespace NUMINAMATH_CALUDE_base_edge_length_is_six_l595_59559

/-- A square pyramid with a hemisphere resting on its base -/
structure PyramidWithHemisphere where
  /-- The height of the pyramid -/
  height : ℝ
  /-- The radius of the hemisphere -/
  radius : ℝ
  /-- The hemisphere is tangent to the other four faces of the pyramid -/
  is_tangent : Bool

/-- The edge length of the base of the pyramid -/
def base_edge_length (p : PyramidWithHemisphere) : ℝ :=
  sorry

/-- Theorem stating the edge length of the base of the pyramid is 6 -/
theorem base_edge_length_is_six (p : PyramidWithHemisphere) 
  (h1 : p.height = 12)
  (h2 : p.radius = 4)
  (h3 : p.is_tangent = true) : 
  base_edge_length p = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_edge_length_is_six_l595_59559


namespace NUMINAMATH_CALUDE_unique_stickers_count_l595_59539

def emily_stickers : ℕ := 22
def mia_unique_stickers : ℕ := 10
def shared_stickers : ℕ := 12

theorem unique_stickers_count :
  (emily_stickers - shared_stickers) + mia_unique_stickers = 20 :=
by sorry

end NUMINAMATH_CALUDE_unique_stickers_count_l595_59539


namespace NUMINAMATH_CALUDE_point_classification_l595_59591

-- Define the plane region
def in_region (x y : ℝ) : Prop := x + y - 1 ≤ 0

-- Theorem stating that (-1,3) is not in the region, while the other points are
theorem point_classification :
  ¬(in_region (-1) 3) ∧ 
  in_region 0 0 ∧ 
  in_region (-1) 1 ∧ 
  in_region 2 (-3) := by
  sorry

end NUMINAMATH_CALUDE_point_classification_l595_59591


namespace NUMINAMATH_CALUDE_point_on_lines_abs_diff_zero_l595_59521

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line passing through the origin
structure Line where
  slope : ℝ

-- Define the two lines l₁ and l₂
def l₁ : Line := { slope := 1 }
def l₂ : Line := { slope := -1 }

-- A point is on a line if it satisfies the line's equation
def pointOnLine (p : Point2D) (l : Line) : Prop :=
  p.y = l.slope * p.x

-- The lines are symmetric about the y-axis
axiom line_symmetry : l₁.slope = -l₂.slope

-- Theorem stating that for any point on either line, |x| - |y| = 0
theorem point_on_lines_abs_diff_zero (p : Point2D) :
  (pointOnLine p l₁ ∨ pointOnLine p l₂) → |p.x| - |p.y| = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_lines_abs_diff_zero_l595_59521


namespace NUMINAMATH_CALUDE_certain_amount_problem_l595_59572

theorem certain_amount_problem (x y : ℝ) : 
  x = 7 → x + y = 15 → 5 * y - 3 * x = 19 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_problem_l595_59572


namespace NUMINAMATH_CALUDE_green_turtles_count_l595_59589

theorem green_turtles_count (total : ℕ) (h : total = 3200) :
  ∃ (green hawksbill : ℕ),
    green + hawksbill = total ∧
    hawksbill = 2 * green ∧
    green = 1066 :=
by
  sorry

end NUMINAMATH_CALUDE_green_turtles_count_l595_59589


namespace NUMINAMATH_CALUDE_cos_alpha_plus_5pi_12_l595_59557

theorem cos_alpha_plus_5pi_12 (α : Real) (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 5 * π / 12) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_5pi_12_l595_59557


namespace NUMINAMATH_CALUDE_scientific_notation_of_280_million_l595_59501

theorem scientific_notation_of_280_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 280000000 = a * (10 : ℝ) ^ n ∧ a = 2.8 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_280_million_l595_59501


namespace NUMINAMATH_CALUDE_median_mode_of_scores_l595_59565

def scores : List ℕ := [7, 10, 9, 8, 9, 9, 8]

def median (l : List ℕ) : ℚ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem median_mode_of_scores :
  median scores = 9 ∧ mode scores = 9 := by sorry

end NUMINAMATH_CALUDE_median_mode_of_scores_l595_59565


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l595_59576

def is_valid_digit (d : Nat) : Prop :=
  d = 4 ∨ d = 6 ∨ d = 9

def contains_all_required_digits (n : Nat) : Prop :=
  ∃ (d1 d2 d3 : Nat), d1 ∈ n.digits 10 ∧ d2 ∈ n.digits 10 ∧ d3 ∈ n.digits 10 ∧
  d1 = 4 ∧ d2 = 6 ∧ d3 = 9

def all_digits_valid (n : Nat) : Prop :=
  ∀ d ∈ n.digits 10, is_valid_digit d

def last_four_digits (n : Nat) : Nat :=
  n % 10000

theorem smallest_valid_number_last_four_digits :
  ∃ (n : Nat), 
    (∀ m : Nat, m < n → ¬(m % 6 = 0 ∧ m % 9 = 0 ∧ all_digits_valid m ∧ contains_all_required_digits m)) ∧
    n % 6 = 0 ∧
    n % 9 = 0 ∧
    all_digits_valid n ∧
    contains_all_required_digits n ∧
    last_four_digits n = 4699 :=
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l595_59576


namespace NUMINAMATH_CALUDE_dihedral_angle_adjacent_faces_l595_59578

/-- Given a regular n-sided pyramid with base dihedral angle α, 
    the dihedral angle φ between adjacent lateral faces satisfies:
    cos(φ/2) = sin(α) * sin(π/n) -/
theorem dihedral_angle_adjacent_faces 
  (n : ℕ) 
  (α : ℝ) 
  (h_n : n ≥ 3) 
  (h_α : 0 < α ∧ α < π / 2) : 
  ∃ φ : ℝ, 
    0 < φ ∧ 
    φ < π ∧ 
    Real.cos (φ / 2) = Real.sin α * Real.sin (π / n) := by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_adjacent_faces_l595_59578


namespace NUMINAMATH_CALUDE_max_value_bound_max_value_achievable_l595_59524

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def max_value (u v w : E) : ℝ :=
  ‖u - 3 • v‖^2 + ‖v - 3 • w‖^2 + ‖w - 3 • u‖^2

theorem max_value_bound (u v w : E) 
  (hu : ‖u‖ = 2) (hv : ‖v‖ = 3) (hw : ‖w‖ = 4) :
  max_value u v w ≤ 377 :=
by sorry

theorem max_value_achievable :
  ∃ (u v w : E), ‖u‖ = 2 ∧ ‖v‖ = 3 ∧ ‖w‖ = 4 ∧ max_value u v w = 377 :=
by sorry

end NUMINAMATH_CALUDE_max_value_bound_max_value_achievable_l595_59524


namespace NUMINAMATH_CALUDE_tan_negative_seven_pi_sixths_l595_59538

theorem tan_negative_seven_pi_sixths : 
  Real.tan (-7 * π / 6) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_seven_pi_sixths_l595_59538


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l595_59552

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^2 - 16 * X + 38 = (X - 4) * q + 22 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l595_59552


namespace NUMINAMATH_CALUDE_compound_weight_is_334_13_l595_59588

/-- Atomic weight of Aluminium in g/mol -/
def Al_weight : ℝ := 26.98

/-- Atomic weight of Bromine in g/mol -/
def Br_weight : ℝ := 79.90

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Chlorine in g/mol -/
def Cl_weight : ℝ := 35.45

/-- The molecular weight of the compound in g/mol -/
def compound_weight : ℝ := Al_weight + 3 * Br_weight + 2 * O_weight + Cl_weight

/-- Theorem stating that the molecular weight of the compound is 334.13 g/mol -/
theorem compound_weight_is_334_13 : compound_weight = 334.13 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_is_334_13_l595_59588


namespace NUMINAMATH_CALUDE_locus_of_M_l595_59569

/-- Circle with center at the origin and radius r -/
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

/-- Line parallel to y-axis at distance a from origin -/
def Line (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = a}

/-- Polar of point A with respect to circle -/
def Polar (r a β : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + β * p.2 = r^2}

/-- Perpendicular line from A to e -/
def Perpendicular (β : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = β}

/-- Locus of point M -/
def Locus (r a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = r^2 - a * p.1}

theorem locus_of_M (r a : ℝ) (hr : r > 0) (ha : a ≠ 0) :
  ∀ β : ℝ, ∃ M : ℝ × ℝ,
    M ∈ Polar r a β ∩ Perpendicular β →
    M ∈ Locus r a :=
  sorry

end NUMINAMATH_CALUDE_locus_of_M_l595_59569


namespace NUMINAMATH_CALUDE_hyperbola_angle_in_fourth_quadrant_l595_59517

/-- Represents a hyperbola equation with angle α -/
def hyperbola_equation (x y α : ℝ) : Prop :=
  x^2 * Real.sin α + y^2 * Real.cos α = 1

/-- Indicates that the foci of the hyperbola are on the y-axis -/
def foci_on_y_axis (α : ℝ) : Prop :=
  Real.cos α > 0 ∧ Real.sin α < 0

/-- Indicates that an angle is in the fourth quadrant -/
def fourth_quadrant (α : ℝ) : Prop :=
  Real.cos α > 0 ∧ Real.sin α < 0

theorem hyperbola_angle_in_fourth_quadrant (α : ℝ) 
  (h1 : ∃ x y : ℝ, hyperbola_equation x y α)
  (h2 : foci_on_y_axis α) : 
  fourth_quadrant α :=
sorry

end NUMINAMATH_CALUDE_hyperbola_angle_in_fourth_quadrant_l595_59517


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l595_59505

theorem unique_solution_to_equation : 
  ∀ a b : ℝ, 2 * (a^2 + 1) * (b^2 + 1) = (a + 1) * (b + 1) * (a * b + 1) → a = 1 ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l595_59505


namespace NUMINAMATH_CALUDE_all_points_on_same_circle_l595_59518

-- Define a type for points in the plane
variable (Point : Type)

-- Define a type for circles in the plane
variable (Circle : Type)

-- Define a function to check if a point lies on a circle
variable (lies_on : Point → Circle → Prop)

-- Define a function to create a circle from four points
variable (circle_from_four_points : Point → Point → Point → Point → Circle)

theorem all_points_on_same_circle 
  (P : Set Point) 
  (h : ∀ (a b c d : Point), a ∈ P → b ∈ P → c ∈ P → d ∈ P → 
    ∃ (C : Circle), lies_on a C ∧ lies_on b C ∧ lies_on c C ∧ lies_on d C) :
  ∃ (C : Circle), ∀ (p : Point), p ∈ P → lies_on p C :=
sorry

end NUMINAMATH_CALUDE_all_points_on_same_circle_l595_59518


namespace NUMINAMATH_CALUDE_tan_sin_cos_relation_l595_59537

theorem tan_sin_cos_relation (α : Real) (h : Real.tan α = -3) :
  (Real.sin α = 3 * Real.sqrt 10 / 10 ∨ Real.sin α = -3 * Real.sqrt 10 / 10) ∧
  (Real.cos α = Real.sqrt 10 / 10 ∨ Real.cos α = -Real.sqrt 10 / 10) :=
by sorry

end NUMINAMATH_CALUDE_tan_sin_cos_relation_l595_59537


namespace NUMINAMATH_CALUDE_complement_of_A_l595_59516

def A : Set ℝ := {x | Real.log x > 0}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l595_59516


namespace NUMINAMATH_CALUDE_tennis_balls_order_l595_59579

theorem tennis_balls_order (white yellow : ℕ) (h1 : white = yellow)
  (h2 : (white : ℚ) / (yellow + 70 : ℚ) = 8 / 13) :
  white + yellow = 224 := by
  sorry

end NUMINAMATH_CALUDE_tennis_balls_order_l595_59579


namespace NUMINAMATH_CALUDE_remaining_onions_l595_59584

/-- Given that Sally grew 5 onions, Fred grew 9 onions, and they gave Sara 4 onions,
    prove that Sally and Fred have 10 onions now. -/
theorem remaining_onions (sally_onions fred_onions given_onions : ℕ)
    (h1 : sally_onions = 5)
    (h2 : fred_onions = 9)
    (h3 : given_onions = 4) :
  sally_onions + fred_onions - given_onions = 10 := by
  sorry

end NUMINAMATH_CALUDE_remaining_onions_l595_59584


namespace NUMINAMATH_CALUDE_walkway_area_is_296_l595_59575

-- Define the garden layout
def num_rows : ℕ := 4
def num_cols : ℕ := 3
def bed_width : ℕ := 4
def bed_height : ℕ := 3
def walkway_width : ℕ := 2

-- Define the total garden dimensions
def garden_width : ℕ := num_cols * bed_width + (num_cols + 1) * walkway_width
def garden_height : ℕ := num_rows * bed_height + (num_rows + 1) * walkway_width

-- Define the total garden area
def total_garden_area : ℕ := garden_width * garden_height

-- Define the total flower bed area
def total_bed_area : ℕ := num_rows * num_cols * bed_width * bed_height

-- Theorem to prove
theorem walkway_area_is_296 : total_garden_area - total_bed_area = 296 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_is_296_l595_59575


namespace NUMINAMATH_CALUDE_barbed_wire_rate_proof_l595_59596

-- Define the given constants
def field_area : ℝ := 3136
def gate_width : ℝ := 1
def num_gates : ℕ := 2
def total_cost : ℝ := 799.20

-- Define the theorem
theorem barbed_wire_rate_proof :
  let side_length : ℝ := Real.sqrt field_area
  let perimeter : ℝ := 4 * side_length
  let wire_length : ℝ := perimeter - (↑num_gates * gate_width)
  let rate_per_meter : ℝ := total_cost / wire_length
  rate_per_meter = 3.60 := by sorry

end NUMINAMATH_CALUDE_barbed_wire_rate_proof_l595_59596


namespace NUMINAMATH_CALUDE_difference_of_squares_problem_1_problem_2_l595_59568

-- Difference of squares formula
theorem difference_of_squares (a b : ℤ) : (a + b) * (a - b) = a^2 - b^2 := by sorry

-- Problem 1
theorem problem_1 : 3001 * 2999 = 3000^2 - 1^2 := by sorry

-- Problem 2
theorem problem_2 : (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1) = 2^64 - 1 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_problem_1_problem_2_l595_59568


namespace NUMINAMATH_CALUDE_a_gt_2_necessary_not_sufficient_for_a_gt_5_l595_59577

theorem a_gt_2_necessary_not_sufficient_for_a_gt_5 :
  (∀ a : ℝ, a > 5 → a > 2) ∧ 
  (∃ a : ℝ, a > 2 ∧ ¬(a > 5)) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_2_necessary_not_sufficient_for_a_gt_5_l595_59577


namespace NUMINAMATH_CALUDE_solution_set_of_even_increasing_function_l595_59592

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem solution_set_of_even_increasing_function
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_increasing : is_increasing_on_nonneg f) :
  {x : ℝ | f x > f 1} = {x : ℝ | x > 1 ∨ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_even_increasing_function_l595_59592


namespace NUMINAMATH_CALUDE_max_value_theorem_l595_59503

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^2 + y^2 + 4 * Real.sqrt (x * y) ≤ 6 ∧
  (x^2 + y^2 + 4 * Real.sqrt (x * y) = 6 ↔ x = 1 ∧ y = 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l595_59503


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l595_59563

/-- Given two points A(a, 3) and A'(2, b) that are symmetric with respect to the x-axis,
    prove that (a + b)^2023 = -1 -/
theorem symmetric_points_sum_power (a b : ℝ) : 
  (∃ A A' : ℝ × ℝ, A = (a, 3) ∧ A' = (2, b) ∧ 
   (A.1 = A'.1 ∧ A.2 = -A'.2)) → (a + b)^2023 = -1 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l595_59563


namespace NUMINAMATH_CALUDE_equation_solution_l595_59513

theorem equation_solution : ∃! x : ℚ, (x - 60) / 3 = (4 - 3 * x) / 6 :=
  by
    use 124 / 5
    sorry

end NUMINAMATH_CALUDE_equation_solution_l595_59513


namespace NUMINAMATH_CALUDE_ab_bounds_l595_59545

theorem ab_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^2 + b^2 - a - b + a*b = 1) : 
  (∀ x y, x > 0 → y > 0 → x^2 + y^2 - x - y + x*y = 1 → a + b ≥ x + y) ∧
  (∀ x y, x > 0 → y > 0 → x^2 + y^2 - x - y + x*y = 1 → a^2 + b^2 ≤ x^2 + y^2) ∧
  (a + b ≤ 2) ∧
  (a^2 + b^2 ≥ 2) := by
sorry

end NUMINAMATH_CALUDE_ab_bounds_l595_59545


namespace NUMINAMATH_CALUDE_unique_pair_no_real_solutions_l595_59546

theorem unique_pair_no_real_solutions : 
  ∃! (b c : ℕ+), 
    (∀ x : ℝ, x^2 + 2*(b:ℝ)*x + (c:ℝ) ≠ 0) ∧ 
    (∀ x : ℝ, x^2 + 2*(c:ℝ)*x + (b:ℝ) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_no_real_solutions_l595_59546


namespace NUMINAMATH_CALUDE_garbage_collection_difference_l595_59582

/-- Given that Lizzie's group collected 387 pounds of garbage and the total amount
    collected by both groups is 735 pounds, prove that the other group collected
    348 pounds less than Lizzie's group. -/
theorem garbage_collection_difference (lizzie_group : ℕ) (total : ℕ) 
  (h1 : lizzie_group = 387)
  (h2 : total = 735) :
  total - lizzie_group = 348 := by
sorry

end NUMINAMATH_CALUDE_garbage_collection_difference_l595_59582


namespace NUMINAMATH_CALUDE_aquarium_count_l595_59536

/-- Given a total number of saltwater animals and the number of animals per aquarium,
    calculate the number of aquariums. -/
def calculate_aquariums (total_animals : ℕ) (animals_per_aquarium : ℕ) : ℕ :=
  total_animals / animals_per_aquarium

theorem aquarium_count :
  let total_animals : ℕ := 52
  let animals_per_aquarium : ℕ := 2
  calculate_aquariums total_animals animals_per_aquarium = 26 := by
  sorry

#eval calculate_aquariums 52 2

end NUMINAMATH_CALUDE_aquarium_count_l595_59536


namespace NUMINAMATH_CALUDE_school_students_count_l595_59511

theorem school_students_count : ℕ :=
  let initial_bananas_per_student : ℕ := 2
  let initial_apples_per_student : ℕ := 1
  let initial_oranges_per_student : ℕ := 1
  let absent_students : ℕ := 420
  let final_bananas_per_student : ℕ := 6
  let final_apples_per_student : ℕ := 3
  let final_oranges_per_student : ℕ := 2

  have h1 : ∀ (S : ℕ), S * initial_bananas_per_student = (S - absent_students) * final_bananas_per_student →
    S = 840 :=
    sorry

  840

/- Proof omitted -/

end NUMINAMATH_CALUDE_school_students_count_l595_59511


namespace NUMINAMATH_CALUDE_equation_solution_l595_59587

theorem equation_solution (r : ℚ) (h1 : r ≠ 2) (h2 : r ≠ -1) :
  (r + 3) / (r - 2) = (r - 1) / (r + 1) ↔ r = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l595_59587


namespace NUMINAMATH_CALUDE_probability_three_common_books_l595_59597

def total_books : ℕ := 12
def books_to_select : ℕ := 4

theorem probability_three_common_books :
  (Nat.choose total_books 3 * Nat.choose (total_books - 3) 1 * Nat.choose (total_books - 4) 1) /
  (Nat.choose total_books books_to_select * Nat.choose total_books books_to_select) =
  32 / 495 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_common_books_l595_59597


namespace NUMINAMATH_CALUDE_matrix_multiplication_l595_59556

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 0, 2]
def C : Matrix (Fin 2) (Fin 2) ℤ := !![15, -7; 20, -16]

theorem matrix_multiplication :
  A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_l595_59556


namespace NUMINAMATH_CALUDE_sundae_price_l595_59507

theorem sundae_price (ice_cream_bars sundaes : ℕ) (total_price ice_cream_price : ℚ) :
  ice_cream_bars = 200 →
  sundaes = 200 →
  total_price = 200 →
  ice_cream_price = 0.4 →
  (total_price - (ice_cream_bars : ℚ) * ice_cream_price) / (sundaes : ℚ) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_sundae_price_l595_59507


namespace NUMINAMATH_CALUDE_final_sparrow_count_l595_59574

/-- Given the initial number of sparrows, the number of sparrows that joined,
    and the number of sparrows that flew away, prove that the final number
    of sparrows on the fence is 3. -/
theorem final_sparrow_count
  (initial_sparrows : ℕ)
  (joined_sparrows : ℕ)
  (flew_away_sparrows : ℕ)
  (h1 : initial_sparrows = 2)
  (h2 : joined_sparrows = 4)
  (h3 : flew_away_sparrows = 3) :
  initial_sparrows + joined_sparrows - flew_away_sparrows = 3 :=
by sorry

end NUMINAMATH_CALUDE_final_sparrow_count_l595_59574


namespace NUMINAMATH_CALUDE_opposite_reciprocal_sum_l595_59540

theorem opposite_reciprocal_sum (a b m n : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : m * n = 1)  -- m and n are reciprocals
  : 5*a + 5*b - m*n = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_sum_l595_59540


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l595_59554

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 23 ∧ 
  correct_value = 48 ∧ 
  corrected_mean = 32.5 →
  ∃ initial_mean : ℝ, 
    initial_mean = 32 ∧ 
    n * corrected_mean = n * initial_mean + (correct_value - wrong_value) :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l595_59554


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l595_59510

theorem cube_plus_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^3 + 1/r^3 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l595_59510


namespace NUMINAMATH_CALUDE_cube_edge_length_from_circumscribed_sphere_volume_l595_59526

/-- Given a sphere circumscribed around a cube, if the volume of the sphere is 32π/3,
    then the edge length of the cube is 4√3/3. -/
theorem cube_edge_length_from_circumscribed_sphere_volume :
  ∀ (r : ℝ) (edge : ℝ),
  r > 0 →
  edge > 0 →
  (4 / 3) * π * r^3 = 32 * π / 3 →
  edge = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_from_circumscribed_sphere_volume_l595_59526


namespace NUMINAMATH_CALUDE_number_of_roses_rose_count_proof_l595_59570

theorem number_of_roses (vase_capacity : ℕ) (num_carnations : ℕ) (num_vases : ℕ) : ℕ :=
  let total_flowers := vase_capacity * num_vases
  total_flowers - num_carnations

theorem rose_count_proof 
  (vase_capacity : ℕ) 
  (num_carnations : ℕ) 
  (num_vases : ℕ) 
  (h1 : vase_capacity = 6) 
  (h2 : num_carnations = 7) 
  (h3 : num_vases = 9) : 
  number_of_roses vase_capacity num_carnations num_vases = 47 := by
  sorry

end NUMINAMATH_CALUDE_number_of_roses_rose_count_proof_l595_59570


namespace NUMINAMATH_CALUDE_average_weight_increase_l595_59533

theorem average_weight_increase (initial_count : ℕ) (initial_weight : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  old_weight = 60 →
  new_weight = 80 →
  (new_weight - old_weight) / initial_count = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l595_59533


namespace NUMINAMATH_CALUDE_unique_p_value_l595_59522

-- Define the properties of p, q, and s
def is_valid_triple (p q s : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime s ∧ p * q = s + 6 ∧ 3 < p ∧ p < q

-- Theorem statement
theorem unique_p_value :
  ∃! p, ∃ q s, is_valid_triple p q s ∧ p = 5 :=
sorry

end NUMINAMATH_CALUDE_unique_p_value_l595_59522


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l595_59548

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 1000 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 92 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l595_59548


namespace NUMINAMATH_CALUDE_find_y_l595_59566

theorem find_y (x : ℝ) (y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 20) : y = 100 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l595_59566


namespace NUMINAMATH_CALUDE_no_real_solutions_l595_59586

theorem no_real_solutions : ¬∃ (x : ℝ), (x^3 - x^2 - 4*x)/(x^2 + 5*x + 6) + 2*x = -6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l595_59586


namespace NUMINAMATH_CALUDE_boat_half_speed_time_and_distance_l595_59547

/-- Represents the motion of a boat experiencing water resistance -/
noncomputable def boat_motion (v₀ : ℝ) (m : ℝ) : ℝ → ℝ × ℝ := fun t =>
  let v := 50 / (t + 10)
  let s := 50 * Real.log ((t + 10) / 10)
  (v, s)

/-- Theorem stating the time and distance for the boat's speed to halve -/
theorem boat_half_speed_time_and_distance :
  let v₀ : ℝ := 5
  let m : ℝ := 1  -- We don't need a specific mass value for this problem
  let (v, s) := boat_motion v₀ m 10
  v = v₀ / 2 ∧ s = 50 * Real.log 2 := by sorry


end NUMINAMATH_CALUDE_boat_half_speed_time_and_distance_l595_59547


namespace NUMINAMATH_CALUDE_one_true_proposition_l595_59525

/-- A parabola y = ax^2 + bx + c opens downwards if a < 0 -/
def opens_downwards (a b c : ℝ) : Prop := a < 0

/-- The set of x where y < 0 for the parabola y = ax^2 + bx + c -/
def negative_y_set (a b c : ℝ) : Set ℝ := {x | a * x^2 + b * x + c < 0}

/-- The original proposition -/
def original_prop (a b c : ℝ) : Prop :=
  opens_downwards a b c → negative_y_set a b c ≠ ∅

/-- The converse of the original proposition -/
def converse_prop (a b c : ℝ) : Prop :=
  negative_y_set a b c ≠ ∅ → opens_downwards a b c

/-- The inverse of the original proposition -/
def inverse_prop (a b c : ℝ) : Prop :=
  ¬(opens_downwards a b c) → negative_y_set a b c = ∅

/-- The contrapositive of the original proposition -/
def contrapositive_prop (a b c : ℝ) : Prop :=
  negative_y_set a b c = ∅ → ¬(opens_downwards a b c)

/-- The main theorem: exactly one of the converse, inverse, and contrapositive is true -/
theorem one_true_proposition :
  ∃! p : Prop, p = ∀ a b c : ℝ, converse_prop a b c ∨
                                p = ∀ a b c : ℝ, inverse_prop a b c ∨
                                p = ∀ a b c : ℝ, contrapositive_prop a b c :=
sorry

end NUMINAMATH_CALUDE_one_true_proposition_l595_59525


namespace NUMINAMATH_CALUDE_exists_divisible_by_three_l595_59535

/-- A circular sequence of natural numbers satisfying specific neighboring conditions -/
structure CircularSequence where
  nums : Fin 99 → ℕ
  neighbor_condition : ∀ i : Fin 99, 
    (nums i.succ = 2 * nums i) ∨ 
    (nums i.succ = nums i + 1) ∨ 
    (nums i.succ = nums i + 2)

/-- Theorem: In any CircularSequence, there exists a number divisible by 3 -/
theorem exists_divisible_by_three (seq : CircularSequence) :
  ∃ i : Fin 99, 3 ∣ seq.nums i := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_three_l595_59535


namespace NUMINAMATH_CALUDE_opposite_numbers_l595_59528

-- Definition of opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Theorem to prove
theorem opposite_numbers : are_opposite (-|(-6)|) (-(-6)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_l595_59528


namespace NUMINAMATH_CALUDE_complex_on_real_axis_l595_59598

theorem complex_on_real_axis (a b : ℝ) : 
  (Complex.ofReal a + Complex.I * b).im = 0 → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_real_axis_l595_59598


namespace NUMINAMATH_CALUDE_bill_share_calculation_l595_59512

/-- Represents the profit share of a partner -/
structure ProfitShare where
  amount : ℕ

/-- Represents the profit-sharing ratio for a partnership -/
structure ProfitRatio where
  bess : ℕ
  bill : ℕ
  bob : ℕ

/-- Calculates a partner's share based on the total profit and their ratio -/
def calculateShare (totalProfit : ℕ) (partnerRatio : ℕ) (totalRatio : ℕ) : ProfitShare :=
  { amount := totalProfit * partnerRatio / totalRatio }

theorem bill_share_calculation (ratio : ProfitRatio) (bobShare : ProfitShare) :
  ratio.bess = 1 → ratio.bill = 2 → ratio.bob = 3 → bobShare.amount = 900 →
  (calculateShare bobShare.amount ratio.bill (ratio.bess + ratio.bill + ratio.bob)).amount = 600 := by
  sorry

end NUMINAMATH_CALUDE_bill_share_calculation_l595_59512


namespace NUMINAMATH_CALUDE_functional_equation_solution_l595_59520

theorem functional_equation_solution (f : ℤ → ℤ) 
  (h : ∀ x y : ℤ, f (x - f y) = f (f x) - f y - 1) : 
  (∀ x : ℤ, f x = -1) ∨ (∀ x : ℤ, f x = x + 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l595_59520


namespace NUMINAMATH_CALUDE_split_eggs_into_groups_l595_59550

/-- The number of groups created when splitting eggs -/
def number_of_groups (total_eggs : ℕ) (eggs_per_group : ℕ) : ℕ :=
  total_eggs / eggs_per_group

/-- Theorem: Splitting 9 eggs into groups of 3 creates 3 groups -/
theorem split_eggs_into_groups :
  number_of_groups 9 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_split_eggs_into_groups_l595_59550


namespace NUMINAMATH_CALUDE_tom_final_book_count_l595_59500

/-- The number of books Tom has after selling some and buying new ones -/
def final_book_count (initial_books sold_books new_books : ℕ) : ℕ :=
  initial_books - sold_books + new_books

/-- Theorem stating that Tom ends up with 39 books -/
theorem tom_final_book_count :
  final_book_count 5 4 38 = 39 := by
  sorry

end NUMINAMATH_CALUDE_tom_final_book_count_l595_59500


namespace NUMINAMATH_CALUDE_expression_value_l595_59553

theorem expression_value (p q r : ℝ) (hp : p ≠ 2) (hq : q ≠ 3) (hr : r ≠ 4) :
  (p^2 - 4) / (4 - r^2) * (q^2 - 9) / (2 - p^2) * (r^2 - 16) / (3 - q^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l595_59553


namespace NUMINAMATH_CALUDE_relationship_abc_l595_59561

theorem relationship_abc (a b c : ℝ) 
  (ha : a = 1 / 2022)
  (hb : b = Real.exp (-2021 / 2022))
  (hc : c = Real.log (2023 / 2022)) :
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l595_59561


namespace NUMINAMATH_CALUDE_line_circle_intersection_l595_59541

/-- The line l: mx - y + 3 - m = 0 and the circle C: x^2 + (y-1)^2 = 5 have at least one common point. -/
theorem line_circle_intersection (m : ℝ) : 
  ∃ (x y : ℝ), (m * x - y + 3 - m = 0) ∧ (x^2 + (y-1)^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l595_59541


namespace NUMINAMATH_CALUDE_undefined_fraction_l595_59583

/-- The expression (2x-6)/(5x-15) is undefined when x = 3 -/
theorem undefined_fraction (x : ℝ) : 
  x = 3 → (2 * x - 6) / (5 * x - 15) = 0 / 0 := by
  sorry

end NUMINAMATH_CALUDE_undefined_fraction_l595_59583


namespace NUMINAMATH_CALUDE_jihye_marbles_l595_59555

/-- Given a total number of marbles and the difference between two people's marbles,
    calculate the number of marbles the person with more marbles has. -/
def marblesWithMore (total : ℕ) (difference : ℕ) : ℕ :=
  (total + difference) / 2

/-- Theorem stating that given 85 total marbles and a difference of 11,
    the person with more marbles has 48 marbles. -/
theorem jihye_marbles : marblesWithMore 85 11 = 48 := by
  sorry

end NUMINAMATH_CALUDE_jihye_marbles_l595_59555


namespace NUMINAMATH_CALUDE_inverse_difference_theorem_l595_59532

theorem inverse_difference_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ((3 * a)⁻¹ - (b / 3)⁻¹) = -(a⁻¹ * b⁻¹) := by
  sorry

end NUMINAMATH_CALUDE_inverse_difference_theorem_l595_59532


namespace NUMINAMATH_CALUDE_second_number_equation_l595_59514

/-- Given three real numbers x, y, and z satisfying the equation 3x + 3y + 3z + 11 = 143,
    prove that y = 44 - x - z. -/
theorem second_number_equation (x y z : ℝ) (h : 3*x + 3*y + 3*z + 11 = 143) :
  y = 44 - x - z := by
  sorry

end NUMINAMATH_CALUDE_second_number_equation_l595_59514


namespace NUMINAMATH_CALUDE_airline_wireless_internet_percentage_l595_59571

theorem airline_wireless_internet_percentage
  (snack_percentage : ℝ)
  (both_services_percentage : ℝ)
  (h1 : snack_percentage = 70)
  (h2 : both_services_percentage = 35)
  (h3 : both_services_percentage ≤ snack_percentage) :
  ∃ (wireless_percentage : ℝ),
    wireless_percentage = both_services_percentage ∧
    wireless_percentage ≤ 100 ∧
    wireless_percentage ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_airline_wireless_internet_percentage_l595_59571


namespace NUMINAMATH_CALUDE_curve_and_line_properties_l595_59551

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 1 = 0

-- Define the distance ratio condition
def distance_ratio (x y : ℝ) : Prop :=
  ((x + 1)^2 + y^2) = 2 * ((x - 1)^2 + y^2)

-- Define the line l
def line_l (x y : ℝ) : Prop := x = 1 ∨ y = 2

-- Main theorem
theorem curve_and_line_properties :
  ∀ x y : ℝ,
  distance_ratio x y →
  (C x y ∧
   (∃ x' y' : ℝ, C x' y' ∧ line_l x' y' ∧
    (x' - 1)^2 + (y' - 2)^2 = 4)) →
  line_l x y :=
sorry

end NUMINAMATH_CALUDE_curve_and_line_properties_l595_59551


namespace NUMINAMATH_CALUDE_k_value_l595_59523

theorem k_value (a b k : ℝ) 
  (h1 : 2 * a = k) 
  (h2 : 3 * b = k) 
  (h3 : k ≠ 1) 
  (h4 : 2 * a + b = a * b) : 
  k = 8 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l595_59523


namespace NUMINAMATH_CALUDE_perfect_square_condition_l595_59542

theorem perfect_square_condition (a b : ℕ+) :
  (∃ k : ℕ, (Nat.gcd a.val b.val + Nat.lcm a.val b.val) = k * (a.val + 1)) →
  b.val ≤ a.val →
  ∃ m : ℕ, b.val = m^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l595_59542


namespace NUMINAMATH_CALUDE_gcd_370_1332_l595_59560

theorem gcd_370_1332 : Nat.gcd 370 1332 = 74 := by
  sorry

end NUMINAMATH_CALUDE_gcd_370_1332_l595_59560


namespace NUMINAMATH_CALUDE_cos_angle_BHD_value_l595_59506

structure RectangularSolid where
  angle_DHG : ℝ
  angle_FHB : ℝ

def cos_angle_BHD (solid : RectangularSolid) : ℝ := sorry

theorem cos_angle_BHD_value (solid : RectangularSolid) 
  (h1 : solid.angle_DHG = π/3)  -- 60 degrees in radians
  (h2 : solid.angle_FHB = π/4)  -- 45 degrees in radians
  : cos_angle_BHD solid = -Real.sqrt 30 / 12 := by sorry

end NUMINAMATH_CALUDE_cos_angle_BHD_value_l595_59506


namespace NUMINAMATH_CALUDE_average_of_averages_l595_59502

theorem average_of_averages (x y : ℝ) (x_positive : 0 < x) (y_positive : 0 < y) : 
  let total_sum := x * y + y * x
  let total_count := x + y
  (x * y) / x = y ∧ (y * x) / y = x → total_sum / total_count = (2 * x * y) / (x + y) := by
sorry

end NUMINAMATH_CALUDE_average_of_averages_l595_59502


namespace NUMINAMATH_CALUDE_chess_matches_to_reach_target_percentage_l595_59590

theorem chess_matches_to_reach_target_percentage 
  (initial_matches : ℕ) 
  (initial_wins : ℕ) 
  (target_percentage : ℚ) : 
  initial_matches = 20 → 
  initial_wins = 19 → 
  target_percentage = 96/100 → 
  ∃ (additional_matches : ℕ), 
    additional_matches = 5 ∧ 
    (initial_wins + additional_matches) / (initial_matches + additional_matches) = target_percentage :=
by sorry

end NUMINAMATH_CALUDE_chess_matches_to_reach_target_percentage_l595_59590


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l595_59534

/-- The function we're considering -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 5

/-- The derivative of our function -/
def f' (x : ℝ) : ℝ := 2*x - 3

theorem function_satisfies_conditions : 
  (f 1 = 3) ∧ (∀ x, (deriv f) x = f' x) := by sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l595_59534


namespace NUMINAMATH_CALUDE_walking_running_distance_ratio_l595_59564

/-- Proves that the ratio of distance walked to distance run is 1:1 given the specified conditions --/
theorem walking_running_distance_ratio
  (walking_speed : ℝ)
  (running_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (h1 : walking_speed = 4)
  (h2 : running_speed = 8)
  (h3 : total_time = 3)
  (h4 : total_distance = 16)
  : ∃ (distance_walked distance_run : ℝ),
    distance_walked / walking_speed + distance_run / running_speed = total_time ∧
    distance_walked + distance_run = total_distance ∧
    distance_walked / distance_run = 1 := by
  sorry

end NUMINAMATH_CALUDE_walking_running_distance_ratio_l595_59564


namespace NUMINAMATH_CALUDE_tyson_jenna_meeting_l595_59543

/-- The distance between points A and B in miles -/
def total_distance : ℝ := 80

/-- Jenna's head start in hours -/
def jenna_head_start : ℝ := 1.5

/-- Jenna's walking speed in miles per hour -/
def jenna_speed : ℝ := 3.5

/-- Tyson's walking speed in miles per hour -/
def tyson_speed : ℝ := 2.8

/-- The distance Tyson walked when he met Jenna -/
def tyson_distance : ℝ := 33.25

theorem tyson_jenna_meeting :
  ∃ t : ℝ, t > 0 ∧
  jenna_speed * (t + jenna_head_start) + tyson_speed * t = total_distance ∧
  tyson_speed * t = tyson_distance :=
sorry

end NUMINAMATH_CALUDE_tyson_jenna_meeting_l595_59543


namespace NUMINAMATH_CALUDE_joan_money_proof_l595_59585

def dimes_to_dollars (jacket_dimes shorts_dimes : ℕ) : ℚ :=
  (jacket_dimes + shorts_dimes) * (10 : ℚ) / 100

theorem joan_money_proof (jacket_dimes shorts_dimes : ℕ) 
  (h1 : jacket_dimes = 15) (h2 : shorts_dimes = 4) : 
  dimes_to_dollars jacket_dimes shorts_dimes = 1.90 := by
  sorry

end NUMINAMATH_CALUDE_joan_money_proof_l595_59585


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l595_59594

theorem algebraic_expression_equality (x y : ℝ) (h : x + 2*y = 2) : 2*x + 4*y - 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l595_59594


namespace NUMINAMATH_CALUDE_function_pair_properties_l595_59519

/-- Real functions c and s defined on ℝ\{0} satisfying certain properties -/
def FunctionPair (c s : ℝ → ℝ) : Prop :=
  (∀ x, x ≠ 0 → c x ≠ 0 ∧ s x ≠ 0) ∧
  (∀ x y, x ≠ 0 → y ≠ 0 → c (x / y) = c x * c y - s x * s y)

/-- Properties of the function pair c and s -/
theorem function_pair_properties (c s : ℝ → ℝ) (h : FunctionPair c s) :
  (∀ x, x ≠ 0 → c (1 / x) = c x) ∧
  (∀ x, x ≠ 0 → s (1 / x) = -s x) ∧
  (c 1 = 1) ∧
  (s 1 = 0) ∧
  (s (-1) = 0) ∧
  ((∀ x, c (-x) = c x ∧ s (-x) = s x) ∨ (∀ x, c (-x) = -c x ∧ s (-x) = -s x)) :=
sorry


end NUMINAMATH_CALUDE_function_pair_properties_l595_59519


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_open_interval_l595_59504

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}

-- State the theorem
theorem M_intersect_N_equals_open_interval : M ∩ N = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_open_interval_l595_59504


namespace NUMINAMATH_CALUDE_game_result_l595_59595

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 10
  else if n % 2 = 0 then 5
  else 0

def allieRolls : List ℕ := [2, 3, 6, 4]
def bettyRolls : List ℕ := [2, 1, 5, 6]

def totalPoints (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result : totalPoints allieRolls + totalPoints bettyRolls = 45 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l595_59595
