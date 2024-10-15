import Mathlib

namespace NUMINAMATH_CALUDE_square_diagonals_properties_l582_58285

structure Square where
  diagonals_perpendicular : Prop
  diagonals_equal : Prop

theorem square_diagonals_properties (s : Square) :
  (s.diagonals_perpendicular ∨ s.diagonals_equal) ∧
  (s.diagonals_perpendicular ∧ s.diagonals_equal) ∧
  ¬(¬s.diagonals_perpendicular) := by
  sorry

end NUMINAMATH_CALUDE_square_diagonals_properties_l582_58285


namespace NUMINAMATH_CALUDE_female_students_count_l582_58269

/-- Given a school with stratified sampling, prove the number of female students -/
theorem female_students_count (total_students sample_size : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : ∃ (girls boys : ℕ), girls + boys = sample_size ∧ boys = girls + 10) :
  (760 : ℝ) = (total_students : ℝ) * (95 : ℝ) / (sample_size : ℝ) :=
sorry

end NUMINAMATH_CALUDE_female_students_count_l582_58269


namespace NUMINAMATH_CALUDE_cylinder_volume_l582_58255

theorem cylinder_volume (r_cylinder r_cone h_cylinder h_cone v_cone : ℝ) :
  r_cylinder / r_cone = 2 / 3 →
  h_cylinder / h_cone = 4 / 3 →
  v_cone = 5.4 →
  (π * r_cylinder^2 * h_cylinder) = 3.2 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l582_58255


namespace NUMINAMATH_CALUDE_two_digit_number_property_l582_58276

theorem two_digit_number_property : 
  let n : ℕ := 27
  let tens_digit : ℕ := n / 10
  let units_digit : ℕ := n % 10
  (units_digit = tens_digit + 5) →
  (n * (tens_digit + units_digit) = 243) :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l582_58276


namespace NUMINAMATH_CALUDE_tommy_balloons_l582_58236

/-- Prove that Tommy started with 26 balloons given the conditions of the problem -/
theorem tommy_balloons (initial : ℕ) (from_mom : ℕ) (after_mom : ℕ) (total : ℕ) : 
  after_mom = 26 → total = 60 → initial + from_mom = total → initial = 26 := by
  sorry

end NUMINAMATH_CALUDE_tommy_balloons_l582_58236


namespace NUMINAMATH_CALUDE_cube_volume_problem_l582_58284

theorem cube_volume_problem (a : ℝ) : 
  a > 0 → 
  (a - 2) * a * (a + 2) = a^3 - 8 → 
  a^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l582_58284


namespace NUMINAMATH_CALUDE_simple_interest_principal_l582_58272

/-- Simple interest calculation -/
theorem simple_interest_principal (interest : ℝ) (rate_paise : ℝ) (time_months : ℝ) :
  interest = 23 * (rate_paise / 100) * time_months →
  interest = 3.45 ∧ rate_paise = 5 ∧ time_months = 3 →
  23 = interest / ((rate_paise / 100) * time_months) :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l582_58272


namespace NUMINAMATH_CALUDE_complex_magnitude_l582_58222

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 9)
  (h3 : Complex.abs (z + w) = 2) :
  Complex.abs z = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l582_58222


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l582_58219

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l582_58219


namespace NUMINAMATH_CALUDE_parabola_focus_and_directrix_l582_58209

/-- Represents a parabola in the form y² = -4px where p is the focal length -/
structure Parabola where
  p : ℝ

/-- The focus of a parabola -/
def Parabola.focus (par : Parabola) : ℝ × ℝ := (-par.p, 0)

/-- The x-coordinate of the directrix of a parabola -/
def Parabola.directrix (par : Parabola) : ℝ := par.p

theorem parabola_focus_and_directrix (par : Parabola) 
  (h : par.p = 2) : 
  (par.focus = (-2, 0)) ∧ (par.directrix = 2) := by
  sorry

#check parabola_focus_and_directrix

end NUMINAMATH_CALUDE_parabola_focus_and_directrix_l582_58209


namespace NUMINAMATH_CALUDE_chameleons_changed_count_l582_58200

/-- Represents the number of chameleons that changed color --/
def chameleons_changed (total : ℕ) (blue_factor : ℕ) (red_factor : ℕ) : ℕ :=
  let initial_blue := blue_factor * (total / (blue_factor + 1))
  total - initial_blue - (total - initial_blue) / red_factor

/-- Theorem stating that 80 chameleons changed color under the given conditions --/
theorem chameleons_changed_count :
  chameleons_changed 140 5 3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_chameleons_changed_count_l582_58200


namespace NUMINAMATH_CALUDE_four_digit_count_l582_58267

/-- The count of four-digit numbers -/
def count_four_digit_numbers : ℕ := 9999 - 1000 + 1

/-- The first four-digit number -/
def first_four_digit : ℕ := 1000

/-- The last four-digit number -/
def last_four_digit : ℕ := 9999

theorem four_digit_count :
  count_four_digit_numbers = 9000 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_count_l582_58267


namespace NUMINAMATH_CALUDE_r_value_l582_58202

/-- The polynomial 8x^3 - 4x^2 - 42x + 45 -/
def P (x : ℝ) : ℝ := 8 * x^3 - 4 * x^2 - 42 * x + 45

/-- (x - r)^2 divides P(x) -/
def divides (r : ℝ) : Prop := ∃ Q : ℝ → ℝ, ∀ x, P x = (x - r)^2 * Q x

theorem r_value : ∃ r : ℝ, divides r ∧ r = 3/2 := by sorry

end NUMINAMATH_CALUDE_r_value_l582_58202


namespace NUMINAMATH_CALUDE_inscribed_box_radius_l582_58249

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  a : ℝ
  b : ℝ
  c : ℝ
  s : ℝ

/-- Properties of the inscribed box -/
def InscribedBoxProperties (box : InscribedBox) : Prop :=
  (box.a + box.b + box.c = 40) ∧
  (box.a * box.b + box.b * box.c + box.c * box.a = 432) ∧
  (4 * box.s^2 = box.a^2 + box.b^2 + box.c^2)

theorem inscribed_box_radius (box : InscribedBox) 
  (h : InscribedBoxProperties box) : box.s = 2 * Real.sqrt 46 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_radius_l582_58249


namespace NUMINAMATH_CALUDE_circle_area_through_isosceles_triangle_vertices_l582_58237

/-- The area of a circle passing through the vertices of an isosceles triangle -/
theorem circle_area_through_isosceles_triangle_vertices (a b c : ℝ) :
  a = 4 →  -- Two sides of the triangle are 4 units long
  b = 4 →  -- Two sides of the triangle are 4 units long
  c = 3 →  -- The base of the triangle is 3 units long
  a = b →  -- The triangle is isosceles
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = (256/55) * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_through_isosceles_triangle_vertices_l582_58237


namespace NUMINAMATH_CALUDE_left_handed_rock_lovers_l582_58206

theorem left_handed_rock_lovers (total : ℕ) (left_handed : ℕ) (rock_lovers : ℕ) (right_handed_non_rock : ℕ) :
  total = 25 →
  left_handed = 10 →
  rock_lovers = 18 →
  right_handed_non_rock = 3 →
  left_handed + (total - left_handed) = total →
  ∃ (left_handed_rock : ℕ),
    left_handed_rock + (left_handed - left_handed_rock) + (rock_lovers - left_handed_rock) + right_handed_non_rock = total ∧
    left_handed_rock = 6 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_rock_lovers_l582_58206


namespace NUMINAMATH_CALUDE_f_max_min_difference_l582_58275

noncomputable def f (x : ℝ) : ℝ := 4 * Real.pi * Real.arcsin x - (Real.arccos (-x))^2

theorem f_max_min_difference :
  ∃ (M m : ℝ), (∀ x : ℝ, f x ≤ M ∧ f x ≥ m) ∧ M - m = 3 * Real.pi^2 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_difference_l582_58275


namespace NUMINAMATH_CALUDE_square_side_length_range_l582_58273

theorem square_side_length_range (a : ℝ) : a^2 = 30 → 5.4 < a ∧ a < 5.5 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_range_l582_58273


namespace NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l582_58228

def is_abcba (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

theorem greatest_abcba_divisible_by_13 :
  ∀ n : ℕ,
    10000 ≤ n ∧ n < 100000 ∧
    is_abcba n ∧
    n % 13 = 0 →
    n ≤ 96769 :=
by sorry

end NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l582_58228


namespace NUMINAMATH_CALUDE_cubic_expansion_sum_difference_l582_58274

theorem cubic_expansion_sum_difference (a a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5*x + 4)^3 = a + a₁*x + a₂*x^2 + a₃*x^3) →
  (a + a₂) - (a₁ + a₃) = -1 :=
by sorry

end NUMINAMATH_CALUDE_cubic_expansion_sum_difference_l582_58274


namespace NUMINAMATH_CALUDE_road_repair_theorem_l582_58230

/-- The number of persons in the first group -/
def first_group : ℕ := 39

/-- The number of days for the first group to complete the work -/
def days_first : ℕ := 24

/-- The number of hours per day for the first group -/
def hours_first : ℕ := 5

/-- The number of days for the second group to complete the work -/
def days_second : ℕ := 26

/-- The number of hours per day for the second group -/
def hours_second : ℕ := 6

/-- The total man-hours required to complete the work -/
def total_man_hours : ℕ := first_group * days_first * hours_first

/-- The number of persons in the second group -/
def second_group : ℕ := total_man_hours / (days_second * hours_second)

theorem road_repair_theorem : second_group = 30 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_theorem_l582_58230


namespace NUMINAMATH_CALUDE_six_by_six_grid_shaded_percentage_l582_58281

/-- Represents a square grid --/
structure SquareGrid :=
  (side : ℕ)
  (total_squares : ℕ)
  (shaded_squares : ℕ)

/-- Calculates the percentage of shaded area in a square grid --/
def shaded_percentage (grid : SquareGrid) : ℚ :=
  (grid.shaded_squares : ℚ) / (grid.total_squares : ℚ)

theorem six_by_six_grid_shaded_percentage :
  let grid : SquareGrid := ⟨6, 36, 21⟩
  shaded_percentage grid = 7 / 12 := by
  sorry

#eval (7 : ℚ) / 12 * 100  -- To show the decimal representation

end NUMINAMATH_CALUDE_six_by_six_grid_shaded_percentage_l582_58281


namespace NUMINAMATH_CALUDE_product_equals_24255_l582_58204

theorem product_equals_24255 : 3^2 * 5 * 7^2 * 11 = 24255 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_24255_l582_58204


namespace NUMINAMATH_CALUDE_sector_to_cone_l582_58291

/-- Given a 300° sector of a circle with radius 12, prove it forms a cone with base radius 10 and slant height 12 -/
theorem sector_to_cone (r : ℝ) (angle : ℝ) :
  r = 12 →
  angle = 300 * (π / 180) →
  ∃ (base_radius slant_height : ℝ),
    base_radius = 10 ∧
    slant_height = r ∧
    2 * π * base_radius = angle * r :=
by sorry

end NUMINAMATH_CALUDE_sector_to_cone_l582_58291


namespace NUMINAMATH_CALUDE_negative_x_implies_positive_expression_l582_58287

theorem negative_x_implies_positive_expression (x : ℝ) (h : x < 0) : -3 * x⁻¹ > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_implies_positive_expression_l582_58287


namespace NUMINAMATH_CALUDE_scenic_spot_selections_l582_58252

theorem scenic_spot_selections (num_classes : ℕ) (num_spots : ℕ) : 
  num_classes = 3 → num_spots = 5 → (num_spots ^ num_classes) = 125 := by
  sorry

end NUMINAMATH_CALUDE_scenic_spot_selections_l582_58252


namespace NUMINAMATH_CALUDE_circle_a_range_l582_58280

-- Define the equation of the circle
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + a = 0

-- Define what it means for an equation to represent a circle
def is_circle (a : ℝ) : Prop :=
  ∃ h k r, ∀ x y, circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0

-- Theorem statement
theorem circle_a_range (a : ℝ) :
  is_circle a → a < 5 :=
sorry

end NUMINAMATH_CALUDE_circle_a_range_l582_58280


namespace NUMINAMATH_CALUDE_unique_intersection_l582_58235

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - x + 1

/-- The line equation -/
def line (k : ℝ) (x : ℝ) : ℝ := 4*x + k

/-- Theorem stating the condition for exactly one intersection point -/
theorem unique_intersection (k : ℝ) : 
  (∃! x, parabola x = line k x) ↔ k = -21/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_l582_58235


namespace NUMINAMATH_CALUDE_max_value_sum_l582_58233

theorem max_value_sum (a b c : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 6) :
  ∃ (M : ℝ), M = Real.sqrt 11 ∧ a + b + c ≤ M ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = M := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_l582_58233


namespace NUMINAMATH_CALUDE_tonya_large_lemonade_sales_l582_58257

/-- Represents the sales data for Tonya's lemonade stand. -/
structure LemonadeSales where
  small_price : ℕ
  medium_price : ℕ
  large_price : ℕ
  total_revenue : ℕ
  small_revenue : ℕ
  medium_revenue : ℕ

/-- Calculates the number of large lemonade cups sold. -/
def large_cups_sold (sales : LemonadeSales) : ℕ :=
  (sales.total_revenue - sales.small_revenue - sales.medium_revenue) / sales.large_price

/-- Theorem stating that Tonya sold 5 cups of large lemonade. -/
theorem tonya_large_lemonade_sales :
  let sales : LemonadeSales := {
    small_price := 1,
    medium_price := 2,
    large_price := 3,
    total_revenue := 50,
    small_revenue := 11,
    medium_revenue := 24
  }
  large_cups_sold sales = 5 := by sorry

end NUMINAMATH_CALUDE_tonya_large_lemonade_sales_l582_58257


namespace NUMINAMATH_CALUDE_collinear_vectors_y_value_l582_58203

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- The problem statement -/
theorem collinear_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-6, y)
  collinear a b → y = -9 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_y_value_l582_58203


namespace NUMINAMATH_CALUDE_squirrel_count_ratio_l582_58250

theorem squirrel_count_ratio :
  ∀ (first_count second_count : ℕ),
  first_count = 12 →
  first_count + second_count = 28 →
  second_count > first_count →
  (second_count : ℚ) / first_count = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_squirrel_count_ratio_l582_58250


namespace NUMINAMATH_CALUDE_min_value_theorem_l582_58213

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 4 * Real.sqrt 5 + 8 ∧
  ((x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) = 4 * Real.sqrt 5 + 8 ↔ x = Real.sqrt 5 + 2 ∧ y = Real.sqrt 5 + 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l582_58213


namespace NUMINAMATH_CALUDE_person_c_start_time_l582_58288

/-- Represents a point on the line AB -/
inductive Point : Type
| A : Point
| C : Point
| D : Point
| B : Point

/-- Represents a person walking on the line AB -/
structure Person where
  name : String
  startTime : Nat
  startPoint : Point
  endPoint : Point
  speed : Nat

/-- Represents the problem setup -/
structure ProblemSetup where
  personA : Person
  personB : Person
  personC : Person
  meetingTimeAB : Nat
  meetingTimeAC : Nat

/-- The theorem to prove -/
theorem person_c_start_time (setup : ProblemSetup) : setup.personC.startTime = 16 :=
  by sorry

end NUMINAMATH_CALUDE_person_c_start_time_l582_58288


namespace NUMINAMATH_CALUDE_lines_intersect_at_point_l582_58201

/-- Represents a 2D point --/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line parameterized by a point and a direction vector --/
structure Line where
  point : Point
  direction : Point

def line1 : Line :=
  { point := { x := 2, y := 3 },
    direction := { x := 3, y := -4 } }

def line2 : Line :=
  { point := { x := 4, y := 1 },
    direction := { x := 5, y := 3 } }

def intersection : Point :=
  { x := 26/11, y := 19/11 }

/-- Returns a point on the line for a given parameter value --/
def pointOnLine (l : Line) (t : ℚ) : Point :=
  { x := l.point.x + t * l.direction.x,
    y := l.point.y + t * l.direction.y }

theorem lines_intersect_at_point :
  ∃ (t u : ℚ), pointOnLine line1 t = intersection ∧ pointOnLine line2 u = intersection ∧
  ∀ (p : Point), (∃ (t' : ℚ), pointOnLine line1 t' = p) ∧ (∃ (u' : ℚ), pointOnLine line2 u' = p) →
  p = intersection := by
  sorry

end NUMINAMATH_CALUDE_lines_intersect_at_point_l582_58201


namespace NUMINAMATH_CALUDE_steps_correct_l582_58231

/-- The number of steps Xiao Gang takes from his house to his school -/
def steps : ℕ := 2000

/-- The distance from Xiao Gang's house to his school in meters -/
def distance : ℝ := 900

/-- Xiao Gang's step length in meters -/
def step_length : ℝ := 0.45

/-- Theorem stating that the number of steps multiplied by the step length equals the distance -/
theorem steps_correct : (steps : ℝ) * step_length = distance := by sorry

end NUMINAMATH_CALUDE_steps_correct_l582_58231


namespace NUMINAMATH_CALUDE_circle_center_point_is_center_l582_58282

/-- The center of a circle given by the equation x^2 + 4x + y^2 - 6y = 24 is (-2, 3) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + 4*x + y^2 - 6*y = 24) ↔ ((x + 2)^2 + (y - 3)^2 = 37) :=
by sorry

/-- The point (-2, 3) is the center of the circle -/
theorem point_is_center : 
  ∃! (a b : ℝ), ∀ (x y : ℝ), (x^2 + 4*x + y^2 - 6*y = 24) ↔ ((x - a)^2 + (y - b)^2 = 37) ∧ 
  a = -2 ∧ b = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_point_is_center_l582_58282


namespace NUMINAMATH_CALUDE_meaningful_reciprocal_l582_58293

theorem meaningful_reciprocal (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_reciprocal_l582_58293


namespace NUMINAMATH_CALUDE_product_evaluation_l582_58220

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l582_58220


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l582_58295

def repeating_decimal_4 : ℚ := 4/9
def repeating_decimal_7 : ℚ := 7/9

theorem product_of_repeating_decimals :
  repeating_decimal_4 * repeating_decimal_7 = 28/81 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l582_58295


namespace NUMINAMATH_CALUDE_five_black_cards_taken_out_l582_58262

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (original_black_cards : ℕ)
  (remaining_black_cards : ℕ)

/-- Defines a standard deck with 52 total cards and 26 black cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    original_black_cards := 26,
    remaining_black_cards := 21 }

/-- Calculates the number of black cards taken out from a deck -/
def black_cards_taken_out (d : Deck) : ℕ :=
  d.original_black_cards - d.remaining_black_cards

/-- Theorem stating that 5 black cards were taken out from the standard deck -/
theorem five_black_cards_taken_out :
  black_cards_taken_out standard_deck = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_black_cards_taken_out_l582_58262


namespace NUMINAMATH_CALUDE_least_cans_required_l582_58232

theorem least_cans_required (maaza pepsi sprite cola fanta : ℕ) 
  (h_maaza : maaza = 200)
  (h_pepsi : pepsi = 288)
  (h_sprite : sprite = 736)
  (h_cola : cola = 450)
  (h_fanta : fanta = 625) :
  let gcd := Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd maaza pepsi) sprite) cola) fanta
  gcd = 1 ∧ maaza / gcd + pepsi / gcd + sprite / gcd + cola / gcd + fanta / gcd = 2299 :=
by sorry

end NUMINAMATH_CALUDE_least_cans_required_l582_58232


namespace NUMINAMATH_CALUDE_bookstore_sales_l582_58294

/-- Given a store that sold 72 books and has a ratio of books to bookmarks sold of 9:2,
    prove that the number of bookmarks sold is 16. -/
theorem bookstore_sales (books_sold : ℕ) (book_ratio : ℕ) (bookmark_ratio : ℕ) 
    (h1 : books_sold = 72)
    (h2 : book_ratio = 9)
    (h3 : bookmark_ratio = 2) :
    (books_sold * bookmark_ratio) / book_ratio = 16 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_sales_l582_58294


namespace NUMINAMATH_CALUDE_min_participants_in_race_l582_58229

/-- Represents a participant in the race -/
structure Participant where
  name : String
  position : Nat

/-- Represents the race with its participants -/
structure Race where
  participants : List Participant

/-- Checks if the given race satisfies the conditions for Andrei -/
def satisfiesAndreiCondition (race : Race) : Prop :=
  ∃ (x : Nat), 3 * x + 1 = race.participants.length

/-- Checks if the given race satisfies the conditions for Dima -/
def satisfiesDimaCondition (race : Race) : Prop :=
  ∃ (y : Nat), 4 * y + 1 = race.participants.length

/-- Checks if the given race satisfies the conditions for Lenya -/
def satisfiesLenyaCondition (race : Race) : Prop :=
  ∃ (z : Nat), 5 * z + 1 = race.participants.length

/-- Checks if all participants have unique finishing positions -/
def uniqueFinishingPositions (race : Race) : Prop :=
  ∀ p1 p2 : Participant, p1 ∈ race.participants → p2 ∈ race.participants → 
    p1 ≠ p2 → p1.position ≠ p2.position

/-- The main theorem stating the minimum number of participants -/
theorem min_participants_in_race : 
  ∀ race : Race, 
    uniqueFinishingPositions race →
    satisfiesAndreiCondition race →
    satisfiesDimaCondition race →
    satisfiesLenyaCondition race →
    race.participants.length ≥ 61 :=
by
  sorry

end NUMINAMATH_CALUDE_min_participants_in_race_l582_58229


namespace NUMINAMATH_CALUDE_product_equals_zero_l582_58227

theorem product_equals_zero : (3 - 5) * (3 - 4) * (3 - 3) * (3 - 2) * (3 - 1) * 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l582_58227


namespace NUMINAMATH_CALUDE_cd_cost_l582_58277

theorem cd_cost (two_cd_cost : ℝ) (h : two_cd_cost = 36) :
  8 * (two_cd_cost / 2) = 144 := by
sorry

end NUMINAMATH_CALUDE_cd_cost_l582_58277


namespace NUMINAMATH_CALUDE_trick_decks_cost_l582_58270

def price_per_deck (quantity : ℕ) : ℕ :=
  if quantity ≤ 3 then 8
  else if quantity ≤ 6 then 7
  else 6

def total_cost (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  victor_decks * price_per_deck victor_decks + friend_decks * price_per_deck friend_decks

theorem trick_decks_cost (victor_decks friend_decks : ℕ) 
  (h1 : victor_decks = 6) (h2 : friend_decks = 2) : 
  total_cost victor_decks friend_decks = 58 := by
  sorry

end NUMINAMATH_CALUDE_trick_decks_cost_l582_58270


namespace NUMINAMATH_CALUDE_sum_of_max_min_f_l582_58212

/-- Given a > 0, prove that the sum of the maximum and minimum values of the function
f(x) = (2009^(x+1) + 2007) / (2009^x + 1) + sin x on the interval [-a, a] is equal to 4016. -/
theorem sum_of_max_min_f (a : ℝ) (h : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ (2009^(x+1) + 2007) / (2009^x + 1) + Real.sin x
  (⨆ x ∈ Set.Icc (-a) a, f x) + (⨅ x ∈ Set.Icc (-a) a, f x) = 4016 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_f_l582_58212


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l582_58208

/-- Represents the four grade levels --/
inductive Grade
| Freshman
| Sophomore
| Junior
| Senior

/-- Represents a student --/
structure Student where
  grade : Grade
  isTwin : Bool

/-- Represents the arrangement of students in a car --/
def CarArrangement := List Student

/-- Total number of students --/
def totalStudents : Nat := 8

/-- Number of students per grade --/
def studentsPerGrade : Nat := 2

/-- Number of students per car --/
def studentsPerCar : Nat := 4

/-- The twin sisters are freshmen --/
def twinSisters : List Student := [
  { grade := Grade.Freshman, isTwin := true },
  { grade := Grade.Freshman, isTwin := true }
]

/-- Checks if an arrangement has exactly two students from the same grade --/
def hasTwoSameGrade (arrangement : CarArrangement) : Bool :=
  sorry

/-- Counts the number of valid arrangements for Car A --/
def countValidArrangements : Nat :=
  sorry

theorem valid_arrangements_count :
  countValidArrangements = 24 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l582_58208


namespace NUMINAMATH_CALUDE_linear_function_point_relation_l582_58299

/-- Given a linear function f(x) = -x + b, prove that if P₁(-1, y₁) and P₂(2, y₂) 
    are points on the graph of f, then y₁ > y₂ -/
theorem linear_function_point_relation (b : ℝ) (y₁ y₂ : ℝ) 
    (h₁ : y₁ = -(-1) + b) 
    (h₂ : y₂ = -(2) + b) : 
  y₁ > y₂ := by
  sorry

#check linear_function_point_relation

end NUMINAMATH_CALUDE_linear_function_point_relation_l582_58299


namespace NUMINAMATH_CALUDE_total_people_waiting_l582_58241

/-- The number of people waiting at each entrance -/
def people_per_entrance : ℕ := 283

/-- The number of entrances -/
def num_entrances : ℕ := 5

/-- The total number of people waiting to get in -/
def total_people : ℕ := people_per_entrance * num_entrances

theorem total_people_waiting :
  total_people = 1415 := by sorry

end NUMINAMATH_CALUDE_total_people_waiting_l582_58241


namespace NUMINAMATH_CALUDE_partition_sum_exists_l582_58218

theorem partition_sum_exists : ∃ (A B : Finset ℕ),
  A ∪ B = Finset.range 14 \ {0} ∧
  A ∩ B = ∅ ∧
  A.card = 5 ∧
  B.card = 8 ∧
  3 * (A.sum id) + 7 * (B.sum id) = 433 := by
  sorry

end NUMINAMATH_CALUDE_partition_sum_exists_l582_58218


namespace NUMINAMATH_CALUDE_q_undetermined_l582_58254

theorem q_undetermined (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (q ∨ ¬q) := by sorry

end NUMINAMATH_CALUDE_q_undetermined_l582_58254


namespace NUMINAMATH_CALUDE_remaining_work_days_for_z_l582_58271

-- Define work rates for each person
def work_rate_x : ℚ := 1 / 5
def work_rate_y : ℚ := 1 / 20
def work_rate_z : ℚ := 1 / 30

-- Define the total work as 1 (100%)
def total_work : ℚ := 1

-- Define the number of days all three work together
def days_together : ℚ := 2

-- Theorem statement
theorem remaining_work_days_for_z :
  let combined_rate := work_rate_x + work_rate_y + work_rate_z
  let work_done_together := combined_rate * days_together
  let remaining_work := total_work - work_done_together
  (remaining_work / work_rate_z : ℚ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_remaining_work_days_for_z_l582_58271


namespace NUMINAMATH_CALUDE_function_identity_l582_58210

def IsNonDegenerateTriangle (a b c : ℕ+) : Prop :=
  a.val + b.val > c.val ∧ b.val + c.val > a.val ∧ c.val + a.val > b.val

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ (a b : ℕ+), IsNonDegenerateTriangle a (f b) (f (b + f a - 1))) :
  ∀ (a : ℕ+), f a = a := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l582_58210


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l582_58224

theorem unique_solution_for_equation :
  ∀ x y : ℝ,
    (Real.sqrt (1 / (4 - x^2)) + Real.sqrt (y^2 / (y - 1)) = 5/2) →
    (x = 0 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l582_58224


namespace NUMINAMATH_CALUDE_valid_combinations_l582_58221

/-- A combination is valid if it satisfies the given equation and range constraints. -/
def is_valid_combination (x y z : ℕ) : Prop :=
  10 ≤ x ∧ x ≤ 20 ∧
  10 ≤ y ∧ y ≤ 20 ∧
  10 ≤ z ∧ z ≤ 20 ∧
  3 * x^2 - y^2 - 7 * z = 99

/-- The theorem states that there are exactly three valid combinations. -/
theorem valid_combinations :
  (∀ x y z : ℕ, is_valid_combination x y z ↔ 
    ((x = 15 ∧ y = 10 ∧ z = 68) ∨ 
     (x = 16 ∧ y = 12 ∧ z = 75) ∨ 
     (x = 18 ∧ y = 15 ∧ z = 78))) :=
by sorry

end NUMINAMATH_CALUDE_valid_combinations_l582_58221


namespace NUMINAMATH_CALUDE_correct_average_l582_58217

theorem correct_average (n : ℕ) (initial_avg : ℚ) (correction1 correction2 : ℚ) :
  n = 10 ∧ 
  initial_avg = 40.2 ∧ 
  correction1 = -19 ∧ 
  correction2 = 18 →
  (n * initial_avg + correction1 + correction2) / n = 40.1 := by
sorry

end NUMINAMATH_CALUDE_correct_average_l582_58217


namespace NUMINAMATH_CALUDE_q_function_determination_l582_58289

theorem q_function_determination (q : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c) →  -- q is quadratic
  q 3 = 0 →                                       -- vertical asymptote at x = 3
  q (-3) = 0 →                                    -- vertical asymptote at x = -3
  q 2 = 18 →                                      -- given condition
  ∀ x, q x = -((18 : ℝ) / 5) * x^2 + (162 : ℝ) / 5 :=
by sorry

end NUMINAMATH_CALUDE_q_function_determination_l582_58289


namespace NUMINAMATH_CALUDE_lcm_of_12_18_30_l582_58296

theorem lcm_of_12_18_30 : Nat.lcm (Nat.lcm 12 18) 30 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_18_30_l582_58296


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l582_58226

-- Problem 1
theorem problem_1 : -1^2023 * ((-8) + 2 / (1/2)) - |(-3)| = 1 := by sorry

-- Problem 2
theorem problem_2 : ∃ x : ℚ, (x + 2) / 3 - (x - 1) / 2 = x + 2 ∧ x = -5/7 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l582_58226


namespace NUMINAMATH_CALUDE_subway_bike_speed_ratio_l582_58292

/-- The speed of the mountain bike in km/h -/
def bike_speed : ℝ := sorry

/-- The speed of the subway in km/h -/
def subway_speed : ℝ := sorry

/-- The time taken to ride the bike initially in minutes -/
def initial_bike_time : ℝ := 10

/-- The time taken by subway in minutes -/
def subway_time : ℝ := 40

/-- The total time taken when riding the bike for the entire journey in hours -/
def total_bike_time : ℝ := 3.5

theorem subway_bike_speed_ratio : 
  subway_speed = 5 * bike_speed :=
sorry

end NUMINAMATH_CALUDE_subway_bike_speed_ratio_l582_58292


namespace NUMINAMATH_CALUDE_area_of_triangle_A_l582_58242

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Represents the folded state of the parallelogram -/
structure FoldedParallelogram :=
  (original : Parallelogram)
  (A' : Point)
  (K : Point)

/-- The area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := 27

/-- The ratio of BK to KC -/
def BK_KC_ratio : ℝ × ℝ := (3, 2)

/-- The area of triangle A'KC -/
def triangleA'KC_area (fp : FoldedParallelogram) : ℝ := sorry

theorem area_of_triangle_A'KC 
  (p : Parallelogram)
  (fp : FoldedParallelogram)
  (h1 : fp.original = p)
  (h2 : parallelogramArea p = 27)
  (h3 : BK_KC_ratio = (3, 2)) :
  triangleA'KC_area fp = 3.6 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_A_l582_58242


namespace NUMINAMATH_CALUDE_system_and_linear_equation_solution_l582_58290

theorem system_and_linear_equation_solution (a : ℝ) :
  (∃ x y : ℝ, x + y = a ∧ x - y = 4*a ∧ 3*x - 5*y - 90 = 0) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_system_and_linear_equation_solution_l582_58290


namespace NUMINAMATH_CALUDE_phil_quarters_left_l582_58246

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.50
def jeans_cost : ℚ := 11.50
def quarter_value : ℚ := 0.25

theorem phil_quarters_left : 
  let total_spent := pizza_cost + soda_cost + jeans_cost
  let remaining_amount := initial_amount - total_spent
  (remaining_amount / quarter_value).floor = 97 := by sorry

end NUMINAMATH_CALUDE_phil_quarters_left_l582_58246


namespace NUMINAMATH_CALUDE_disjunction_truth_implication_false_l582_58258

theorem disjunction_truth_implication_false : 
  ¬(∀ (p q : Prop), (p ∨ q) → (p ∧ q)) := by sorry

end NUMINAMATH_CALUDE_disjunction_truth_implication_false_l582_58258


namespace NUMINAMATH_CALUDE_expression_equals_zero_l582_58278

theorem expression_equals_zero : 
  |Real.sqrt 3 - 1| + (Real.pi - 3)^0 - Real.tan (Real.pi / 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l582_58278


namespace NUMINAMATH_CALUDE_percentage_problem_l582_58244

-- Define the percentage P
def P : ℝ := sorry

-- Theorem to prove
theorem percentage_problem : P = 45 := by
  -- Define the conditions
  have h1 : P / 100 * 60 = 35 / 100 * 40 + 13 := sorry
  
  -- Prove that P equals 45
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l582_58244


namespace NUMINAMATH_CALUDE_friends_not_going_to_movies_l582_58214

theorem friends_not_going_to_movies (total_friends : ℕ) (friends_going : ℕ) : 
  total_friends = 15 → friends_going = 8 → total_friends - friends_going = 7 := by
  sorry

end NUMINAMATH_CALUDE_friends_not_going_to_movies_l582_58214


namespace NUMINAMATH_CALUDE_f_max_min_implies_a_range_l582_58223

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + a + 6

/-- Theorem: If f has both a maximum and a minimum, then a < -3 or a > 6 -/
theorem f_max_min_implies_a_range (a : ℝ) : 
  (∃ (x_max x_min : ℝ), ∀ x, f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_implies_a_range_l582_58223


namespace NUMINAMATH_CALUDE_fastest_growing_function_l582_58260

/-- Proves that 0.001e^x grows faster than 1000ln(x), x^1000, and 1000⋅2^x as x approaches infinity -/
theorem fastest_growing_function :
  ∀ (ε : ℝ), ε > 0 → ∃ (X : ℝ), ∀ (x : ℝ), x > X →
    (0.001 * Real.exp x > 1000 * Real.log x) ∧
    (0.001 * Real.exp x > x^1000) ∧
    (0.001 * Real.exp x > 1000 * 2^x) :=
sorry

end NUMINAMATH_CALUDE_fastest_growing_function_l582_58260


namespace NUMINAMATH_CALUDE_log_comparison_l582_58259

theorem log_comparison (a : ℝ) (h : a > 1) : Real.log a / Real.log (a - 1) > Real.log (a + 1) / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_log_comparison_l582_58259


namespace NUMINAMATH_CALUDE_scientific_notation_56_9_billion_l582_58253

def billion : ℝ := 1000000000

theorem scientific_notation_56_9_billion :
  56.9 * billion = 5.69 * (10 : ℝ) ^ 9 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_56_9_billion_l582_58253


namespace NUMINAMATH_CALUDE_brainiac_teaser_ratio_l582_58216

/-- Represents the number of brainiacs who like rebus teasers -/
def R : ℕ := 58

/-- Represents the number of brainiacs who like math teasers -/
def M : ℕ := 38

/-- The total number of brainiacs surveyed -/
def total : ℕ := 100

/-- The number of brainiacs who like both rebus and math teasers -/
def both : ℕ := 18

/-- The number of brainiacs who like neither rebus nor math teasers -/
def neither : ℕ := 4

/-- The number of brainiacs who like math teasers but not rebus teasers -/
def mathOnly : ℕ := 20

theorem brainiac_teaser_ratio :
  R = 58 ∧ M = 38 ∧ 
  total = 100 ∧
  both = 18 ∧
  neither = 4 ∧
  mathOnly = 20 →
  R * 19 = M * 29 := by
  sorry

end NUMINAMATH_CALUDE_brainiac_teaser_ratio_l582_58216


namespace NUMINAMATH_CALUDE_hotel_arrangement_count_l582_58238

/-- Represents the number of ways to arrange people in rooms -/
def arrangement_count (n : ℕ) (r : ℕ) (m : ℕ) : ℕ := sorry

/-- The number of people -/
def total_people : ℕ := 5

/-- The number of rooms -/
def total_rooms : ℕ := 3

/-- The number of people who cannot be in the same room -/
def restricted_people : ℕ := 2

/-- Theorem stating the number of possible arrangements -/
theorem hotel_arrangement_count :
  arrangement_count total_people total_rooms restricted_people = 114 := by
  sorry

end NUMINAMATH_CALUDE_hotel_arrangement_count_l582_58238


namespace NUMINAMATH_CALUDE_opera_selection_probability_l582_58234

theorem opera_selection_probability :
  let total_operas : ℕ := 5
  let distinguished_operas : ℕ := 2
  let selection_size : ℕ := 2

  let total_combinations : ℕ := Nat.choose total_operas selection_size
  let favorable_combinations : ℕ := distinguished_operas * (total_operas - distinguished_operas)

  (favorable_combinations : ℚ) / total_combinations = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_opera_selection_probability_l582_58234


namespace NUMINAMATH_CALUDE_min_value_sum_l582_58239

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 3 / Real.rpow 162 (1/3) ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' / (3 * b') + b' / (6 * c') + c' / (9 * a') = 3 / Real.rpow 162 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l582_58239


namespace NUMINAMATH_CALUDE_most_cost_effective_plan_optimal_plan_is_valid_l582_58263

/-- Represents the rental plan for buses -/
structure RentalPlan where
  large_buses : ℕ
  small_buses : ℕ

/-- Calculates the total number of seats for a given rental plan -/
def total_seats (plan : RentalPlan) : ℕ :=
  plan.large_buses * 45 + plan.small_buses * 30

/-- Calculates the total cost for a given rental plan -/
def total_cost (plan : RentalPlan) : ℕ :=
  plan.large_buses * 400 + plan.small_buses * 300

/-- Checks if a rental plan is valid according to the given conditions -/
def is_valid_plan (plan : RentalPlan) : Prop :=
  total_seats plan ≥ 240 ∧ 
  plan.large_buses + plan.small_buses ≤ 6 ∧
  total_cost plan ≤ 2300

/-- The theorem stating that the most cost-effective valid plan is 4 large buses and 2 small buses -/
theorem most_cost_effective_plan :
  ∀ (plan : RentalPlan),
    is_valid_plan plan →
    total_cost plan ≥ total_cost { large_buses := 4, small_buses := 2 } :=
by sorry

/-- The theorem stating that the plan with 4 large buses and 2 small buses is valid -/
theorem optimal_plan_is_valid :
  is_valid_plan { large_buses := 4, small_buses := 2 } :=
by sorry

end NUMINAMATH_CALUDE_most_cost_effective_plan_optimal_plan_is_valid_l582_58263


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l582_58215

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ) (ha : a ≠ 0), ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 2

/-- Theorem: The given equation is a quadratic equation -/
theorem equation_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_equation_is_quadratic_l582_58215


namespace NUMINAMATH_CALUDE_average_mark_five_subjects_l582_58207

/-- Given a student's marks in six subjects, prove that the average mark for five subjects
    (excluding physics) is 70, when the total marks are 350 more than the physics marks. -/
theorem average_mark_five_subjects (physics_mark : ℕ) : 
  let total_marks : ℕ := physics_mark + 350
  let remaining_marks : ℕ := total_marks - physics_mark
  let num_subjects : ℕ := 5
  (remaining_marks : ℚ) / num_subjects = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_mark_five_subjects_l582_58207


namespace NUMINAMATH_CALUDE_survey_sample_size_l582_58240

/-- Represents a survey conducted in an urban area -/
structure UrbanSurvey where
  year : Nat
  month : Nat
  investigators : Nat
  households : Nat
  questionnaires : Nat

/-- Definition of sample size for an urban survey -/
def sampleSize (survey : UrbanSurvey) : Nat :=
  survey.questionnaires

/-- Theorem stating that the sample size of the given survey is 30,000 -/
theorem survey_sample_size :
  let survey : UrbanSurvey := {
    year := 2010
    month := 5  -- May
    investigators := 400
    households := 10000
    questionnaires := 30000
  }
  sampleSize survey = 30000 := by
  sorry


end NUMINAMATH_CALUDE_survey_sample_size_l582_58240


namespace NUMINAMATH_CALUDE_smallest_reciprocal_sum_l582_58268

/-- Given a quadratic equation x^2 - s*x + p with roots r₁ and r₂ -/
def quadratic_equation (s p : ℝ) (x : ℝ) : ℝ := x^2 - s*x + p

/-- The sum of powers of roots is constant for powers 1 to 1004 -/
def sum_powers_constant (r₁ r₂ : ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1004 → r₁^n + r₂^n = r₁ + r₂

/-- The theorem stating the smallest possible value of 1/r₁^1005 + 1/r₂^1005 -/
theorem smallest_reciprocal_sum (s p r₁ r₂ : ℝ) :
  (∀ x : ℝ, quadratic_equation s p x = 0 ↔ x = r₁ ∨ x = r₂) →
  sum_powers_constant r₁ r₂ →
  (∃ v : ℝ, v = 1/r₁^1005 + 1/r₂^1005 ∧ ∀ w : ℝ, w = 1/r₁^1005 + 1/r₂^1005 → v ≤ w) →
  ∃ v : ℝ, v = 2 ∧ v = 1/r₁^1005 + 1/r₂^1005 := by
  sorry

end NUMINAMATH_CALUDE_smallest_reciprocal_sum_l582_58268


namespace NUMINAMATH_CALUDE_ariels_age_ariels_current_age_l582_58264

theorem ariels_age (birth_year : Nat) (fencing_start_year : Nat) (years_fencing : Nat) : Nat :=
  let current_year := fencing_start_year + years_fencing
  let age := current_year - birth_year
  age

theorem ariels_current_age : 
  ariels_age 1992 2006 16 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ariels_age_ariels_current_age_l582_58264


namespace NUMINAMATH_CALUDE_two_tangents_iff_a_in_range_l582_58261

/-- Definition of the circle C -/
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + a*x + 2*y + a^2 = 0

/-- Point A -/
def point_A : ℝ × ℝ := (1, 2)

/-- Condition for exactly two tangents -/
def has_two_tangents (a : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (circle_C a (center.1) (center.2)) ∧
    ((point_A.1 - center.1)^2 + (point_A.2 - center.2)^2 > radius^2) ∧
    (radius^2 > 0)

/-- Main theorem -/
theorem two_tangents_iff_a_in_range :
  ∀ a : ℝ, has_two_tangents a ↔ -2*(3:ℝ).sqrt/3 < a ∧ a < 2*(3:ℝ).sqrt/3 :=
sorry

end NUMINAMATH_CALUDE_two_tangents_iff_a_in_range_l582_58261


namespace NUMINAMATH_CALUDE_company_gender_ratio_l582_58245

/-- Represents the number of employees of each gender in a company -/
structure Company where
  male : ℕ
  female : ℕ

/-- The ratio of male to female employees -/
def genderRatio (c : Company) : ℚ :=
  c.male / c.female

theorem company_gender_ratio (c : Company) :
  c.male = 189 ∧ 
  genderRatio {male := c.male + 3, female := c.female} = 8 / 9 →
  genderRatio c = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_company_gender_ratio_l582_58245


namespace NUMINAMATH_CALUDE_garrison_provisions_theorem_l582_58286

/-- Calculates the number of days provisions will last after reinforcement arrives -/
def daysProvisionsLast (initialMen : ℕ) (initialDays : ℕ) (reinforcementMen : ℕ) (daysPassed : ℕ) : ℕ :=
  let totalProvisions := initialMen * initialDays
  let remainingProvisions := totalProvisions - (initialMen * daysPassed)
  let totalMenAfterReinforcement := initialMen + reinforcementMen
  remainingProvisions / totalMenAfterReinforcement

/-- Theorem stating that given the specific conditions, provisions will last 10 more days -/
theorem garrison_provisions_theorem :
  daysProvisionsLast 2000 40 2000 20 = 10 := by
  sorry

#eval daysProvisionsLast 2000 40 2000 20

end NUMINAMATH_CALUDE_garrison_provisions_theorem_l582_58286


namespace NUMINAMATH_CALUDE_alpha_value_proof_l582_58256

theorem alpha_value_proof (α : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^α) 
  (h2 : (deriv f) (-1) = -4) : α = 4 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_proof_l582_58256


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l582_58265

theorem complex_arithmetic_equality : 
  4 * (7 * 24) / 3 + 5 * (13 * 15) - 2 * (6 * 28) + 7 * (3 * 19) / 2 = 1062.5 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l582_58265


namespace NUMINAMATH_CALUDE_unique_root_P_l582_58298

-- Define the polynomial sequence
def P : ℕ → ℝ → ℝ
  | 0, x => 0
  | 1, x => x
  | (n+2), x => x * P (n+1) x + (1 - x) * P n x

-- State the theorem
theorem unique_root_P (n : ℕ) (hn : n ≥ 1) : 
  ∀ x : ℝ, P n x = 0 ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_unique_root_P_l582_58298


namespace NUMINAMATH_CALUDE_cube_volume_in_pyramid_l582_58243

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (has_square_base : base_side > 0)
  (has_equilateral_lateral_faces : True)

/-- A cube placed inside the pyramid -/
structure InsideCube :=
  (side_length : ℝ)
  (base_on_pyramid_base : True)
  (top_vertices_touch_midpoints : True)

/-- The theorem stating the volume of the cube inside the pyramid -/
theorem cube_volume_in_pyramid (p : Pyramid) (c : InsideCube) 
  (h1 : p.base_side = 2) : c.side_length ^ 3 = 3 * Real.sqrt 6 / 4 := by
  sorry

#check cube_volume_in_pyramid

end NUMINAMATH_CALUDE_cube_volume_in_pyramid_l582_58243


namespace NUMINAMATH_CALUDE_min_lcm_a_c_l582_58251

theorem min_lcm_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) :
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 30 ∧ 
  (∀ (x y : ℕ), Nat.lcm x b = 20 → Nat.lcm b y = 24 → Nat.lcm a' c' ≤ Nat.lcm x y) :=
sorry

end NUMINAMATH_CALUDE_min_lcm_a_c_l582_58251


namespace NUMINAMATH_CALUDE_unique_intersection_point_l582_58248

/-- The function g(x) = x^3 - 9x^2 + 27x - 29 -/
def g (x : ℝ) : ℝ := x^3 - 9*x^2 + 27*x - 29

/-- The point (1, 1) is the unique intersection of y = g(x) and y = g^(-1)(x) -/
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (1, 1) := by sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l582_58248


namespace NUMINAMATH_CALUDE_hit_target_probability_l582_58205

/-- The probability of hitting a target at least 2 times out of 3 independent shots,
    given that the probability of hitting the target in a single shot is 0.6 -/
theorem hit_target_probability :
  let p : ℝ := 0.6  -- Probability of hitting the target in a single shot
  let n : ℕ := 3    -- Total number of shots
  let k : ℕ := 2    -- Minimum number of successful hits

  -- Probability of hitting the target at least k times out of n shots
  (Finset.sum (Finset.range (n - k + 1)) (fun i =>
    (n.choose (k + i)) * p^(k + i) * (1 - p)^(n - k - i))) = 81 / 125 :=
by sorry

end NUMINAMATH_CALUDE_hit_target_probability_l582_58205


namespace NUMINAMATH_CALUDE_clothing_color_theorem_l582_58266

-- Define the colors
inductive Color
| Red
| Blue

-- Define a structure for clothing
structure Clothing :=
  (tshirt : Color)
  (shorts : Color)

-- Define a function to check if two colors are different
def different_colors (c1 c2 : Color) : Prop :=
  c1 ≠ c2

-- Define the problem statement
theorem clothing_color_theorem 
  (alyna bohdan vika grysha : Clothing) : 
  (alyna.tshirt = Color.Red) →
  (bohdan.tshirt = Color.Red) →
  (different_colors alyna.shorts bohdan.shorts) →
  (different_colors vika.tshirt grysha.tshirt) →
  (vika.shorts = Color.Blue) →
  (grysha.shorts = Color.Blue) →
  (different_colors alyna.tshirt vika.tshirt) →
  (different_colors alyna.shorts vika.shorts) →
  (alyna = ⟨Color.Red, Color.Red⟩ ∧
   bohdan = ⟨Color.Red, Color.Blue⟩ ∧
   vika = ⟨Color.Blue, Color.Blue⟩ ∧
   grysha = ⟨Color.Red, Color.Blue⟩) :=
by sorry


end NUMINAMATH_CALUDE_clothing_color_theorem_l582_58266


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l582_58211

theorem inverse_variation_problem (x w : ℝ) (h : ∃ (c : ℝ), ∀ (x w : ℝ), x^4 * w^(1/4) = c) :
  (x = 3 ∧ w = 16) → (x = 6 → w = 1/4096) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l582_58211


namespace NUMINAMATH_CALUDE_rectangle_hall_length_l582_58283

theorem rectangle_hall_length :
  ∀ (length breadth : ℝ),
    length = breadth + 5 →
    length * breadth = 750 →
    length = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_hall_length_l582_58283


namespace NUMINAMATH_CALUDE_f_min_value_iff_a_range_l582_58225

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 - 2*a*x - 2 else x + 36/x - 6*a

-- State the theorem
theorem f_min_value_iff_a_range (a : ℝ) :
  (∀ x : ℝ, f a 2 ≤ f a x) ↔ 2 ≤ a ∧ a ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_iff_a_range_l582_58225


namespace NUMINAMATH_CALUDE_right_triangle_sin_identity_l582_58279

theorem right_triangle_sin_identity (A B C : Real) (h1 : C = Real.pi / 2) (h2 : A + B = Real.pi / 2) :
  Real.sin A * Real.sin B * Real.sin (A - B) + 
  Real.sin B * Real.sin C * Real.sin (B - C) + 
  Real.sin C * Real.sin A * Real.sin (C - A) + 
  Real.sin (A - B) * Real.sin (B - C) * Real.sin (C - A) = 0 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_identity_l582_58279


namespace NUMINAMATH_CALUDE_most_reasonable_sampling_methods_l582_58297

/-- Represents different sampling methods --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents a sampling survey --/
structure Survey where
  totalItems : ℕ
  sampleSize : ℕ
  hasStrata : Bool
  hasStructure : Bool

/-- Determines the most reasonable sampling method for a given survey --/
def mostReasonableSamplingMethod (s : Survey) : SamplingMethod :=
  if s.hasStrata then SamplingMethod.Stratified
  else if s.hasStructure then SamplingMethod.Systematic
  else SamplingMethod.SimpleRandom

/-- The three surveys described in the problem --/
def survey1 : Survey := { totalItems := 15, sampleSize := 5, hasStrata := false, hasStructure := false }
def survey2 : Survey := { totalItems := 240, sampleSize := 20, hasStrata := true, hasStructure := false }
def survey3 : Survey := { totalItems := 25 * 38, sampleSize := 25, hasStrata := false, hasStructure := true }

/-- Theorem stating the most reasonable sampling methods for the given surveys --/
theorem most_reasonable_sampling_methods :
  (mostReasonableSamplingMethod survey1 = SamplingMethod.SimpleRandom) ∧
  (mostReasonableSamplingMethod survey2 = SamplingMethod.Stratified) ∧
  (mostReasonableSamplingMethod survey3 = SamplingMethod.Systematic) :=
sorry


end NUMINAMATH_CALUDE_most_reasonable_sampling_methods_l582_58297


namespace NUMINAMATH_CALUDE_log_equation_holds_l582_58247

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 4 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_holds_l582_58247
