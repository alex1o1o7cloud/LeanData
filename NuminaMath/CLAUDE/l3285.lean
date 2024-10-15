import Mathlib

namespace NUMINAMATH_CALUDE_total_squares_6x6_grid_l3285_328570

/-- The number of squares of a given size in a grid --/
def count_squares (grid_size : ℕ) (square_size : ℕ) : ℕ :=
  (grid_size - square_size + 1) ^ 2

/-- The total number of squares in a 6x6 grid --/
theorem total_squares_6x6_grid :
  let grid_size := 6
  let square_sizes := [1, 2, 3, 4]
  (square_sizes.map (count_squares grid_size)).sum = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_squares_6x6_grid_l3285_328570


namespace NUMINAMATH_CALUDE_perfect_line_statistics_l3285_328571

/-- A scatter plot represents a set of points in a 2D plane. -/
structure ScatterPlot where
  points : Set (ℝ × ℝ)

/-- A straight line in 2D space. -/
structure StraightLine where
  slope : ℝ
  intercept : ℝ

/-- The sum of squared residuals for a scatter plot and a fitted line. -/
def sumSquaredResiduals (plot : ScatterPlot) (line : StraightLine) : ℝ := sorry

/-- The correlation coefficient for a scatter plot. -/
def correlationCoefficient (plot : ScatterPlot) : ℝ := sorry

/-- Predicate to check if all points in a scatter plot lie on a given straight line. -/
def allPointsOnLine (plot : ScatterPlot) (line : StraightLine) : Prop := sorry

theorem perfect_line_statistics (plot : ScatterPlot) (line : StraightLine) :
  allPointsOnLine plot line →
  sumSquaredResiduals plot line = 0 ∧ correlationCoefficient plot = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_line_statistics_l3285_328571


namespace NUMINAMATH_CALUDE_bryson_new_shoes_l3285_328552

/-- Proves that buying 2 pairs of shoes results in 4 new shoes -/
theorem bryson_new_shoes : 
  ∀ (pairs_bought : ℕ) (shoes_per_pair : ℕ),
  pairs_bought = 2 → shoes_per_pair = 2 →
  pairs_bought * shoes_per_pair = 4 := by
  sorry

end NUMINAMATH_CALUDE_bryson_new_shoes_l3285_328552


namespace NUMINAMATH_CALUDE_probability_is_correct_l3285_328545

/-- Represents the total number of cards -/
def t : ℕ := 93

/-- Represents the number of cards with blue dinosaurs -/
def blue_dinosaurs : ℕ := 16

/-- Represents the number of cards with green robots -/
def green_robots : ℕ := 14

/-- Represents the number of cards with blue robots -/
def blue_robots : ℕ := 36

/-- Represents the number of cards with green dinosaurs -/
def green_dinosaurs : ℕ := t - (blue_dinosaurs + green_robots + blue_robots)

/-- The probability of choosing a card with either a green dinosaur or a blue robot -/
def probability : ℚ := (green_dinosaurs + blue_robots : ℚ) / t

theorem probability_is_correct : probability = 21 / 31 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_correct_l3285_328545


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l3285_328544

theorem quadratic_equivalence : 
  ∀ x y : ℝ, y = x^2 - 2*x + 3 ↔ y = (x - 1)^2 + 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l3285_328544


namespace NUMINAMATH_CALUDE_no_xy_term_implies_m_eq_6_l3285_328578

/-- The polynomial that does not contain the xy term -/
def polynomial (x y m : ℝ) : ℝ := x^2 - m*x*y - y^2 + 6*x*y - 1

/-- The coefficient of xy in the polynomial -/
def xy_coefficient (m : ℝ) : ℝ := -m + 6

theorem no_xy_term_implies_m_eq_6 (m : ℝ) :
  (∀ x y : ℝ, polynomial x y m = x^2 - y^2 - 1) → m = 6 :=
by sorry

end NUMINAMATH_CALUDE_no_xy_term_implies_m_eq_6_l3285_328578


namespace NUMINAMATH_CALUDE_trader_cloth_sale_l3285_328577

/-- The number of meters of cloth sold by a trader -/
def meters_sold (total_price profit_per_meter cost_per_meter : ℚ) : ℚ :=
  total_price / (cost_per_meter + profit_per_meter)

/-- Theorem: The trader sold 85 meters of cloth -/
theorem trader_cloth_sale : meters_sold 8925 5 100 = 85 := by
  sorry

end NUMINAMATH_CALUDE_trader_cloth_sale_l3285_328577


namespace NUMINAMATH_CALUDE_parents_in_program_l3285_328572

theorem parents_in_program (total_people : ℕ) (pupils : ℕ) (h1 : total_people = 803) (h2 : pupils = 698) :
  total_people - pupils = 105 := by
  sorry

end NUMINAMATH_CALUDE_parents_in_program_l3285_328572


namespace NUMINAMATH_CALUDE_function_and_range_l3285_328591

def f (a c : ℝ) (x : ℝ) : ℝ := a * x^3 + c * x

theorem function_and_range (a c : ℝ) (h1 : a > 0) :
  (∃ k, (3 * a + c) * k = -1 ∧ k ≠ 0) →
  (∀ x, 3 * a * x^2 + c ≥ -12) →
  (∃ x, 3 * a * x^2 + c = -12) →
  (f a c = fun x ↦ 2 * x^3 - 12 * x) ∧
  (∀ y ∈ Set.Icc (-8 * Real.sqrt 2) (8 * Real.sqrt 2), 
    ∃ x ∈ Set.Icc (-2) 2, f a c x = y) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a c x ∈ Set.Icc (-8 * Real.sqrt 2) (8 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_function_and_range_l3285_328591


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3285_328547

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3285_328547


namespace NUMINAMATH_CALUDE_fishing_line_section_length_l3285_328592

theorem fishing_line_section_length 
  (num_reels : ℕ) 
  (reel_length : ℝ) 
  (num_sections : ℕ) 
  (h1 : num_reels = 3) 
  (h2 : reel_length = 100) 
  (h3 : num_sections = 30) : 
  (num_reels * reel_length) / num_sections = 10 := by
  sorry

end NUMINAMATH_CALUDE_fishing_line_section_length_l3285_328592


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3285_328581

/-- The surface area of a cylinder with diameter and height both equal to 4 is 24π. -/
theorem cylinder_surface_area : 
  ∀ (d h : ℝ), d = 4 → h = 4 → 2 * π * (d / 2) * (d / 2 + h) = 24 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3285_328581


namespace NUMINAMATH_CALUDE_cube_root_of_2197_l3285_328566

theorem cube_root_of_2197 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 2197) : x = 13 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_2197_l3285_328566


namespace NUMINAMATH_CALUDE_simplify_absolute_expression_l3285_328511

theorem simplify_absolute_expression : abs (-4^2 - 3 + 6) = 13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_expression_l3285_328511


namespace NUMINAMATH_CALUDE_rope_cutting_l3285_328555

theorem rope_cutting (l : ℚ) : 
  l > 0 ∧ (1 / l).isInt ∧ (2 / l).isInt → (3 / l) ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_l3285_328555


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3285_328522

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (π/6 + α) = 3/5)
  (h2 : π/3 < α ∧ α < 5*π/6) : 
  Real.cos α = (3 - 4 * Real.sqrt 3) / 10 := by sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3285_328522


namespace NUMINAMATH_CALUDE_total_books_l3285_328519

theorem total_books (joan_books tom_books : ℕ) 
  (h1 : joan_books = 10) 
  (h2 : tom_books = 38) : 
  joan_books + tom_books = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l3285_328519


namespace NUMINAMATH_CALUDE_number_of_rattlesnakes_rattlesnakes_count_l3285_328535

/-- The number of rattlesnakes in a park with given conditions -/
theorem number_of_rattlesnakes (total_snakes : ℕ) (boa_constrictors : ℕ) : ℕ :=
  let pythons := 3 * boa_constrictors
  let rattlesnakes := total_snakes - (boa_constrictors + pythons)
  rattlesnakes

/-- Proof that the number of rattlesnakes is 40 given the conditions -/
theorem rattlesnakes_count :
  number_of_rattlesnakes 200 40 = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_of_rattlesnakes_rattlesnakes_count_l3285_328535


namespace NUMINAMATH_CALUDE_fastest_reaction_rate_C_l3285_328586

-- Define the reaction rates
def rate_A : ℝ := 0.15
def rate_B : ℝ := 0.6
def rate_C : ℝ := 0.5
def rate_D : ℝ := 0.4

-- Define the stoichiometric coefficients
def coeff_A : ℝ := 1
def coeff_B : ℝ := 3
def coeff_C : ℝ := 2
def coeff_D : ℝ := 2

-- Theorem: The reaction rate of C is the fastest
theorem fastest_reaction_rate_C :
  rate_C / coeff_C > rate_A / coeff_A ∧
  rate_C / coeff_C > rate_B / coeff_B ∧
  rate_C / coeff_C > rate_D / coeff_D :=
by sorry

end NUMINAMATH_CALUDE_fastest_reaction_rate_C_l3285_328586


namespace NUMINAMATH_CALUDE_students_taking_both_courses_l3285_328502

/-- Given a class with the following properties:
  * total_students: The total number of students in the class
  * french_students: The number of students taking French
  * german_students: The number of students taking German
  * neither_students: The number of students taking neither French nor German
  
  Prove that the number of students taking both French and German is equal to
  french_students + german_students - (total_students - neither_students) -/
theorem students_taking_both_courses
  (total_students : ℕ)
  (french_students : ℕ)
  (german_students : ℕ)
  (neither_students : ℕ)
  (h1 : total_students = 87)
  (h2 : french_students = 41)
  (h3 : german_students = 22)
  (h4 : neither_students = 33) :
  french_students + german_students - (total_students - neither_students) = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_both_courses_l3285_328502


namespace NUMINAMATH_CALUDE_trig_sum_zero_l3285_328512

theorem trig_sum_zero (θ : ℝ) (a : ℝ) (h : Real.cos (π / 6 - θ) = a) :
  Real.cos (5 * π / 6 + θ) + Real.sin (2 * π / 3 - θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_zero_l3285_328512


namespace NUMINAMATH_CALUDE_point_translation_second_quadrant_l3285_328565

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point by a given vector -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

/-- Check if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The main theorem -/
theorem point_translation_second_quadrant (m n : ℝ) :
  let A : Point := { x := m, y := n }
  let A' : Point := translate A 2 3
  isInSecondQuadrant A' → m < -2 ∧ n > -3 := by
  sorry

end NUMINAMATH_CALUDE_point_translation_second_quadrant_l3285_328565


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_minimum_l3285_328568

theorem hyperbola_focal_length_minimum (a b c : ℝ) : 
  a > 0 → b > 0 → c^2 = a^2 + b^2 → a + b - c = 2 → 2*c ≥ 4 + 4*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_minimum_l3285_328568


namespace NUMINAMATH_CALUDE_robin_gum_total_l3285_328506

theorem robin_gum_total (packages : ℕ) (pieces_per_package : ℕ) (extra_pieces : ℕ) : 
  packages = 5 → pieces_per_package = 7 → extra_pieces = 6 →
  packages * pieces_per_package + extra_pieces = 41 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_total_l3285_328506


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3285_328598

theorem simplify_polynomial (y : ℝ) : 
  3 * y^3 - 7 * y^2 + 12 * y + 5 - (2 * y^3 - 4 + 3 * y^2 - 9 * y) = y^3 - 10 * y^2 + 21 * y + 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3285_328598


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l3285_328560

theorem unique_root_quadratic (b c : ℝ) : 
  (∃! x : ℝ, x^2 + b*x + c = 0) → 
  (b = c + 1) → 
  c = 1 :=
by
  sorry

#check unique_root_quadratic

end NUMINAMATH_CALUDE_unique_root_quadratic_l3285_328560


namespace NUMINAMATH_CALUDE_plot_area_in_acres_l3285_328584

-- Define the triangle dimensions
def leg1 : ℝ := 8
def leg2 : ℝ := 6

-- Define scale and conversion factors
def scale : ℝ := 3  -- 1 cm = 3 miles
def acres_per_square_mile : ℝ := 640

-- Define the theorem
theorem plot_area_in_acres :
  let triangle_area := (1/2) * leg1 * leg2
  let scaled_area := triangle_area * scale * scale
  let area_in_acres := scaled_area * acres_per_square_mile
  area_in_acres = 138240 := by sorry

end NUMINAMATH_CALUDE_plot_area_in_acres_l3285_328584


namespace NUMINAMATH_CALUDE_unique_divisor_product_100_l3285_328583

/-- Product of all divisors of a natural number -/
def divisor_product (n : ℕ) : ℕ := sorry

/-- Theorem stating that 100 is the only natural number whose divisor product is 10^9 -/
theorem unique_divisor_product_100 :
  ∀ n : ℕ, divisor_product n = 10^9 ↔ n = 100 := by sorry

end NUMINAMATH_CALUDE_unique_divisor_product_100_l3285_328583


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3285_328564

theorem vector_equation_solution (α β : Real) :
  let A : Real × Real := (Real.cos α, Real.sin α)
  let B : Real × Real := (Real.cos β, Real.sin β)
  let C : Real × Real := (1/2, Real.sqrt 3/2)
  (C.1 = B.1 - A.1 ∧ C.2 = B.2 - A.2) → β = 2*Real.pi/3 ∨ β = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3285_328564


namespace NUMINAMATH_CALUDE_sum_of_consecutive_integers_l3285_328520

theorem sum_of_consecutive_integers (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 22 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_integers_l3285_328520


namespace NUMINAMATH_CALUDE_prove_movie_theatre_seats_l3285_328553

def movie_theatre_seats (adult_price child_price : ℕ) (num_children total_revenue : ℕ) : Prop :=
  let total_seats := num_children + (total_revenue - num_children * child_price) / adult_price
  total_seats = 250 ∧
  adult_price * (total_seats - num_children) + child_price * num_children = total_revenue

theorem prove_movie_theatre_seats :
  movie_theatre_seats 6 4 188 1124 := by
  sorry

end NUMINAMATH_CALUDE_prove_movie_theatre_seats_l3285_328553


namespace NUMINAMATH_CALUDE_brianna_books_to_reread_l3285_328557

/-- The number of books Brianna reads per month -/
def books_per_month : ℕ := 2

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of new books Brianna was given as a gift -/
def books_gifted : ℕ := 6

/-- The number of new books Brianna bought -/
def books_bought : ℕ := 8

/-- The number of new books Brianna plans to borrow from the library -/
def books_borrowed : ℕ := books_bought - 2

/-- The total number of books Brianna needs for the year -/
def total_books_needed : ℕ := books_per_month * months_in_year

/-- The total number of new books Brianna will have -/
def total_new_books : ℕ := books_gifted + books_bought + books_borrowed

/-- The number of old books Brianna needs to reread -/
def old_books_to_reread : ℕ := total_books_needed - total_new_books

theorem brianna_books_to_reread : old_books_to_reread = 4 := by
  sorry

end NUMINAMATH_CALUDE_brianna_books_to_reread_l3285_328557


namespace NUMINAMATH_CALUDE_product_of_divisors_2022_no_prime_power_2022_l3285_328529

def T (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem product_of_divisors_2022 : T 2022 = 2022^4 := by sorry

theorem no_prime_power_2022 : ∀ (n : ℕ) (p : ℕ), Nat.Prime p → T n ≠ p^2022 := by sorry

end NUMINAMATH_CALUDE_product_of_divisors_2022_no_prime_power_2022_l3285_328529


namespace NUMINAMATH_CALUDE_reporter_earnings_per_hour_l3285_328551

/-- Calculate reporter's earnings per hour given their pay rate and work conditions --/
theorem reporter_earnings_per_hour 
  (words_per_minute : ℕ)
  (pay_per_word : ℚ)
  (pay_per_article : ℕ)
  (num_articles : ℕ)
  (total_hours : ℕ)
  (h1 : words_per_minute = 10)
  (h2 : pay_per_word = 1/10)
  (h3 : pay_per_article = 60)
  (h4 : num_articles = 3)
  (h5 : total_hours = 4) :
  (words_per_minute * 60 * total_hours : ℚ) * pay_per_word + 
  (num_articles * pay_per_article : ℚ) / total_hours = 105 := by
  sorry

#eval (10 * 60 * 4 : ℚ) * (1/10) + (3 * 60 : ℚ) / 4

end NUMINAMATH_CALUDE_reporter_earnings_per_hour_l3285_328551


namespace NUMINAMATH_CALUDE_triangle_properties_l3285_328596

-- Define the triangle ABC
def A : ℝ × ℝ := (-1, 4)
def B : ℝ × ℝ := (-2, -1)
def C : ℝ × ℝ := (2, 3)

-- Define the height line from B to AC
def height_line (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the area of the triangle
def triangle_area : ℝ := 8

-- Theorem statement
theorem triangle_properties :
  (∀ x y : ℝ, height_line x y ↔ (x + y - 3 = 0)) ∧
  triangle_area = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3285_328596


namespace NUMINAMATH_CALUDE_total_notes_count_l3285_328587

theorem total_notes_count (total_amount : ℕ) (note_50_count : ℕ) (note_50_value : ℕ) (note_500_value : ℕ) 
  (h1 : total_amount = 10350)
  (h2 : note_50_count = 97)
  (h3 : note_50_value = 50)
  (h4 : note_500_value = 500)
  (h5 : ∃ (note_500_count : ℕ), total_amount = note_50_count * note_50_value + note_500_count * note_500_value) :
  ∃ (total_notes : ℕ), total_notes = note_50_count + (total_amount - note_50_count * note_50_value) / note_500_value ∧ total_notes = 108 := by
sorry

end NUMINAMATH_CALUDE_total_notes_count_l3285_328587


namespace NUMINAMATH_CALUDE_jane_hector_meeting_l3285_328590

/-- Represents a point on the circular path --/
inductive Point := | A | B | C | D | E

/-- The circular path with its length in blocks --/
def CircularPath := 24

/-- Hector's walking speed (arbitrary units) --/
def HectorSpeed : ℝ := 1

/-- Jane's walking speed in terms of Hector's --/
def JaneSpeed : ℝ := 3 * HectorSpeed

/-- The meeting point of Jane and Hector --/
def MeetingPoint : Point := Point.B

theorem jane_hector_meeting :
  ∀ (t : ℝ),
  t > 0 →
  t * HectorSpeed + t * JaneSpeed = CircularPath →
  MeetingPoint = Point.B :=
sorry

end NUMINAMATH_CALUDE_jane_hector_meeting_l3285_328590


namespace NUMINAMATH_CALUDE_simplest_fraction_of_0_63575_l3285_328563

theorem simplest_fraction_of_0_63575 :
  ∃ (a b : ℕ+), (a.val : ℚ) / b.val = 63575 / 100000 ∧
  ∀ (c d : ℕ+), (c.val : ℚ) / d.val = 63575 / 100000 → a.val ≤ c.val ∧ b.val ≤ d.val →
  a = 2543 ∧ b = 4000 := by
sorry

end NUMINAMATH_CALUDE_simplest_fraction_of_0_63575_l3285_328563


namespace NUMINAMATH_CALUDE_student_competition_numbers_l3285_328589

theorem student_competition_numbers (n : ℕ) : 
  (100 < n ∧ n < 200) ∧ 
  (∃ k : ℕ, n = 4 * k - 2) ∧
  (∃ l : ℕ, n = 5 * l - 3) ∧
  (∃ m : ℕ, n = 6 * m - 4) →
  n = 122 ∨ n = 182 := by
sorry

end NUMINAMATH_CALUDE_student_competition_numbers_l3285_328589


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3285_328546

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def B : Set ℕ := {x | 2 ≤ x ∧ x < 6}

theorem intersection_of_A_and_B : A ∩ B = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3285_328546


namespace NUMINAMATH_CALUDE_focal_chord_length_l3285_328534

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * Real.sqrt 3 * x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a chord passing through the focus
structure FocalChord where
  a : PointOnParabola
  b : PointOnParabola
  passes_through_focus : True  -- We assume this property without specifying the focus

-- Theorem statement
theorem focal_chord_length 
  (ab : FocalChord) 
  (midpoint_x : ab.a.x + ab.b.x = 4) : 
  Real.sqrt ((ab.a.x - ab.b.x)^2 + (ab.a.y - ab.b.y)^2) = 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_focal_chord_length_l3285_328534


namespace NUMINAMATH_CALUDE_orthocenter_circumcircle_property_l3285_328505

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric operations
variable (orthocenter : Point → Point → Point → Point)
variable (circumcircle : Point → Point → Point → Circle)
variable (circumcenter : Point → Point → Point → Point)
variable (line_intersection : Point → Point → Point → Point → Point)
variable (circle_line_intersection : Circle → Point → Point → Point)
variable (on_circle : Point → Circle → Prop)

-- State the theorem
theorem orthocenter_circumcircle_property
  (A B C H D P Q : Point)
  (ω : Circle)
  (h1 : H = orthocenter A B C)
  (h2 : ω = circumcircle H A B)
  (h3 : D = circle_line_intersection ω B C)
  (h4 : D ≠ B)
  (h5 : P = line_intersection D H A C)
  (h6 : Q = circumcenter A D P) :
  on_circle (circumcenter H A B) (circumcircle B D Q) :=
sorry

end NUMINAMATH_CALUDE_orthocenter_circumcircle_property_l3285_328505


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3285_328526

theorem tan_alpha_value (α : Real) (h1 : π/2 < α ∧ α < π) 
  (h2 : Real.sin (α + π/4) = Real.sqrt 2 / 10) : Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3285_328526


namespace NUMINAMATH_CALUDE_equilateral_iff_sum_zero_l3285_328507

-- Define j as a complex number representing a rotation by 120°
noncomputable def j : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

-- Define the property of j
axiom j_cube : j ^ 3 = 1
axiom j_sum : 1 + j + j^2 = 0

-- Define a triangle in the complex plane
structure Triangle :=
  (A B C : ℂ)

-- Define the property of being equilateral
def is_equilateral (t : Triangle) : Prop :=
  Complex.abs (t.B - t.A) = Complex.abs (t.C - t.B) ∧
  Complex.abs (t.C - t.B) = Complex.abs (t.A - t.C)

-- State the theorem
theorem equilateral_iff_sum_zero (t : Triangle) :
  is_equilateral t ↔ t.A + j * t.B + j^2 * t.C = 0 :=
sorry

end NUMINAMATH_CALUDE_equilateral_iff_sum_zero_l3285_328507


namespace NUMINAMATH_CALUDE_largest_positive_integer_satisfying_condition_l3285_328528

theorem largest_positive_integer_satisfying_condition : 
  ∀ x : ℕ+, x + 1000 > 1000 * x.val → x.val ≤ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_positive_integer_satisfying_condition_l3285_328528


namespace NUMINAMATH_CALUDE_video_game_spending_l3285_328562

theorem video_game_spending (weekly_allowance : ℝ) (weeks : ℕ) 
  (video_game_cost : ℝ) (book_fraction : ℝ) (remaining : ℝ) :
  weekly_allowance = 10 →
  weeks = 4 →
  book_fraction = 1/4 →
  remaining = 15 →
  video_game_cost > 0 →
  video_game_cost < weekly_allowance * weeks →
  remaining = weekly_allowance * weeks - video_game_cost - 
    (weekly_allowance * weeks - video_game_cost) * book_fraction →
  video_game_cost / (weekly_allowance * weeks) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_video_game_spending_l3285_328562


namespace NUMINAMATH_CALUDE_malt_shop_syrup_usage_l3285_328549

/-- Calculates the total syrup used in a malt shop given specific conditions -/
theorem malt_shop_syrup_usage
  (syrup_per_shake : ℝ)
  (syrup_per_cone : ℝ)
  (syrup_per_sundae : ℝ)
  (extra_syrup : ℝ)
  (extra_syrup_percentage : ℝ)
  (num_shakes : ℕ)
  (num_cones : ℕ)
  (num_sundaes : ℕ)
  (h1 : syrup_per_shake = 5.5)
  (h2 : syrup_per_cone = 8)
  (h3 : syrup_per_sundae = 4.2)
  (h4 : extra_syrup = 0.3)
  (h5 : extra_syrup_percentage = 0.1)
  (h6 : num_shakes = 5)
  (h7 : num_cones = 4)
  (h8 : num_sundaes = 3) :
  ∃ total_syrup : ℝ,
    total_syrup = num_shakes * syrup_per_shake +
                  num_cones * syrup_per_cone +
                  num_sundaes * syrup_per_sundae +
                  (↑(round ((num_shakes + num_cones) * extra_syrup_percentage)) * extra_syrup) ∧
    total_syrup = 72.4 := by
  sorry

end NUMINAMATH_CALUDE_malt_shop_syrup_usage_l3285_328549


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3285_328569

theorem profit_percentage_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.96 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100 / 96 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3285_328569


namespace NUMINAMATH_CALUDE_sum_xyz_equals_zero_l3285_328538

theorem sum_xyz_equals_zero 
  (x y z a b c : ℝ) 
  (eq1 : x + y - z = a - b)
  (eq2 : x - y + z = b - c)
  (eq3 : -x + y + z = c - a) : 
  x + y + z = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_xyz_equals_zero_l3285_328538


namespace NUMINAMATH_CALUDE_solve_x_equation_l3285_328537

theorem solve_x_equation (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^2 + 1) (h2 : x / 5 = 5 * y) :
  x = (625 + 25 * Real.sqrt 589) / 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_x_equation_l3285_328537


namespace NUMINAMATH_CALUDE_uniform_transform_l3285_328515

-- Define a uniform random variable on an interval
def UniformRandom (a b : ℝ) := {X : ℝ → ℝ | ∀ x, a ≤ x ∧ x ≤ b → X x = (b - a)⁻¹}

theorem uniform_transform (b₁ b : ℝ → ℝ) :
  UniformRandom 0 1 b₁ →
  (∀ x, b x = 3 * (b₁ x - 2)) →
  UniformRandom (-6) (-3) b := by
sorry

end NUMINAMATH_CALUDE_uniform_transform_l3285_328515


namespace NUMINAMATH_CALUDE_equation_holds_l3285_328575

theorem equation_holds (a b c : ℤ) (h1 : a = c + 1) (h2 : b = a - 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l3285_328575


namespace NUMINAMATH_CALUDE_diagonal_segments_100x101_l3285_328500

/-- The number of segments in the diagonal of a rectangle divided by grid lines -/
def diagonal_segments (width : ℕ) (height : ℕ) : ℕ :=
  width + height - 1

/-- The width of the rectangle -/
def rectangle_width : ℕ := 100

/-- The height of the rectangle -/
def rectangle_height : ℕ := 101

theorem diagonal_segments_100x101 :
  diagonal_segments rectangle_width rectangle_height = 200 := by
  sorry

#eval diagonal_segments rectangle_width rectangle_height

end NUMINAMATH_CALUDE_diagonal_segments_100x101_l3285_328500


namespace NUMINAMATH_CALUDE_pants_price_calculation_l3285_328510

/-- The price of a T-shirt in dollars -/
def tshirt_price : ℚ := 5

/-- The price of a skirt in dollars -/
def skirt_price : ℚ := 6

/-- The price of a refurbished T-shirt in dollars -/
def refurbished_tshirt_price : ℚ := tshirt_price / 2

/-- The total income from the sales in dollars -/
def total_income : ℚ := 53

/-- The number of T-shirts sold -/
def tshirts_sold : ℕ := 2

/-- The number of pants sold -/
def pants_sold : ℕ := 1

/-- The number of skirts sold -/
def skirts_sold : ℕ := 4

/-- The number of refurbished T-shirts sold -/
def refurbished_tshirts_sold : ℕ := 6

/-- The price of a pair of pants in dollars -/
def pants_price : ℚ := 4

theorem pants_price_calculation :
  pants_price = total_income - 
    (tshirts_sold * tshirt_price + 
     skirts_sold * skirt_price + 
     refurbished_tshirts_sold * refurbished_tshirt_price) :=
by sorry

end NUMINAMATH_CALUDE_pants_price_calculation_l3285_328510


namespace NUMINAMATH_CALUDE_yellow_balls_unchanged_yellow_balls_count_l3285_328543

/-- Represents the contents of a box with colored balls -/
structure BoxContents where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Removes one blue ball from the box -/
def removeOneBlueBall (box : BoxContents) : BoxContents :=
  { box with blue := box.blue - 1 }

/-- Theorem stating that the number of yellow balls remains unchanged after removing a blue ball -/
theorem yellow_balls_unchanged (initialBox : BoxContents) :
  (removeOneBlueBall initialBox).yellow = initialBox.yellow :=
by
  sorry

/-- The main theorem proving that the number of yellow balls remains 5 after removing a blue ball -/
theorem yellow_balls_count (initialBox : BoxContents)
  (h1 : initialBox.red = 3)
  (h2 : initialBox.blue = 2)
  (h3 : initialBox.yellow = 5) :
  (removeOneBlueBall initialBox).yellow = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_balls_unchanged_yellow_balls_count_l3285_328543


namespace NUMINAMATH_CALUDE_real_part_of_complex_fraction_l3285_328542

theorem real_part_of_complex_fraction (i : ℂ) : 
  i * i = -1 → Complex.re ((1 + i) / (1 - i)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_fraction_l3285_328542


namespace NUMINAMATH_CALUDE_number42_does_not_contain_5_l3285_328594

/-- Represents a five-digit rising number -/
structure RisingNumber :=
  (d1 d2 d3 d4 d5 : Nat)
  (h1 : d1 < d2)
  (h2 : d2 < d3)
  (h3 : d3 < d4)
  (h4 : d4 < d5)
  (h5 : 1 ≤ d1 ∧ d5 ≤ 8)

/-- The list of all valid rising numbers -/
def risingNumbers : List RisingNumber := sorry

/-- The 42nd number in the sorted list of rising numbers -/
def number42 : RisingNumber := sorry

/-- Theorem stating that the 42nd rising number does not contain 5 -/
theorem number42_does_not_contain_5 : 
  number42.d1 ≠ 5 ∧ number42.d2 ≠ 5 ∧ number42.d3 ≠ 5 ∧ number42.d4 ≠ 5 ∧ number42.d5 ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_number42_does_not_contain_5_l3285_328594


namespace NUMINAMATH_CALUDE_min_value_on_line_l3285_328573

theorem min_value_on_line (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_line : 2 * a + b = 1) :
  1 / a + 2 / b ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ 1 / a₀ + 2 / b₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_line_l3285_328573


namespace NUMINAMATH_CALUDE_ratio_to_eleven_l3285_328574

theorem ratio_to_eleven : ∃ x : ℚ, (5 : ℚ) / 1 = x / 11 ∧ x = 55 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_eleven_l3285_328574


namespace NUMINAMATH_CALUDE_contradiction_assumption_for_no_real_roots_l3285_328576

theorem contradiction_assumption_for_no_real_roots (a b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + a = 0) ↔ 
  ¬(∀ x : ℝ, x^2 + b*x + a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_contradiction_assumption_for_no_real_roots_l3285_328576


namespace NUMINAMATH_CALUDE_product_digit_sum_l3285_328536

/-- The number of digits in the second factor of the product (9)(999...9) -/
def k : ℕ := sorry

/-- The sum of digits in the resulting integer -/
def digit_sum : ℕ := 1009

/-- The resulting integer from the product (9)(999...9) -/
def result : ℕ := 10^k - 1

theorem product_digit_sum : 
  (∀ n : ℕ, n ≤ k → (result / 10^n) % 10 = 9) ∧ 
  (result % 10 = 9) ∧
  (digit_sum = 9 * k) ∧
  (k = 112) :=
sorry

end NUMINAMATH_CALUDE_product_digit_sum_l3285_328536


namespace NUMINAMATH_CALUDE_simplify_expression_l3285_328593

theorem simplify_expression (a b x y : ℝ) (h : b*x + a*y ≠ 0) :
  (b*x*(a^2*x^2 + 2*a^2*y^2 + b^2*y^2)) / (b*x + a*y) +
  (a*y*(a^2*x^2 + 2*b^2*x^2 + b^2*y^2)) / (b*x + a*y) =
  (a*x + b*y)^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3285_328593


namespace NUMINAMATH_CALUDE_f_g_3_equals_95_l3285_328541

def f (x : ℝ) : ℝ := 4 * x - 5

def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem f_g_3_equals_95 : f (g 3) = 95 := by
  sorry

end NUMINAMATH_CALUDE_f_g_3_equals_95_l3285_328541


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3285_328527

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_sum : a 3 + a 8 = 10) : 
  3 * a 5 + a 7 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3285_328527


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l3285_328599

theorem average_of_three_numbers (N : ℝ) : 
  9 ≤ N ∧ N ≤ 17 →
  ∃ k : ℕ, (6 + 10 + N) / 3 = 2 * k →
  (6 + 10 + N) / 3 = 10 := by
sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l3285_328599


namespace NUMINAMATH_CALUDE_radio_survey_males_not_listening_l3285_328521

theorem radio_survey_males_not_listening (males_listening : ℕ) 
  (females_not_listening : ℕ) (total_listening : ℕ) (total_not_listening : ℕ) 
  (h1 : males_listening = 70)
  (h2 : females_not_listening = 110)
  (h3 : total_listening = 145)
  (h4 : total_not_listening = 160) :
  total_not_listening - females_not_listening = 50 :=
by
  sorry

#check radio_survey_males_not_listening

end NUMINAMATH_CALUDE_radio_survey_males_not_listening_l3285_328521


namespace NUMINAMATH_CALUDE_orchid_planting_problem_l3285_328501

/-- Calculates the number of orchid bushes to be planted -/
def orchids_to_plant (current : ℕ) (after : ℕ) : ℕ :=
  after - current

theorem orchid_planting_problem :
  let current_orchids : ℕ := 22
  let total_after_planting : ℕ := 35
  orchids_to_plant current_orchids total_after_planting = 13 := by
  sorry

end NUMINAMATH_CALUDE_orchid_planting_problem_l3285_328501


namespace NUMINAMATH_CALUDE_labourer_salary_proof_l3285_328525

/-- The salary increase rate per year -/
def increase_rate : ℝ := 1.4

/-- The number of years -/
def years : ℕ := 3

/-- The final salary after 3 years -/
def final_salary : ℝ := 8232

/-- The present salary of the labourer -/
def present_salary : ℝ := 3000

theorem labourer_salary_proof :
  (present_salary * increase_rate ^ years) = final_salary :=
sorry

end NUMINAMATH_CALUDE_labourer_salary_proof_l3285_328525


namespace NUMINAMATH_CALUDE_vector_dot_product_l3285_328561

theorem vector_dot_product (a b : ℝ × ℝ) : 
  (Real.sqrt 2 : ℝ) = Real.sqrt (a.1 ^ 2 + a.2 ^ 2) →
  2 = Real.sqrt (b.1 ^ 2 + b.2 ^ 2) →
  (3 * Real.pi / 4 : ℝ) = Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1 ^ 2 + a.2 ^ 2) * Real.sqrt (b.1 ^ 2 + b.2 ^ 2))) →
  (a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) : ℝ) = 6 := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_l3285_328561


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l3285_328588

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate_in_still_water
  (speed_with_stream : ℝ)
  (speed_against_stream : ℝ)
  (h1 : speed_with_stream = 26)
  (h2 : speed_against_stream = 4) :
  (speed_with_stream + speed_against_stream) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l3285_328588


namespace NUMINAMATH_CALUDE_athletes_meeting_distance_l3285_328558

theorem athletes_meeting_distance (v₁ v₂ : ℝ) (h₁ : v₁ > 0) (h₂ : v₂ > 0) : 
  (∃ x : ℝ, x > 0 ∧ 
    300 / v₁ = (x - 300) / v₂ ∧ 
    (x + 100) / v₁ = (x - 100) / v₂) → 
  (∃ x : ℝ, x = 500) :=
by sorry

end NUMINAMATH_CALUDE_athletes_meeting_distance_l3285_328558


namespace NUMINAMATH_CALUDE_solution_set_correct_l3285_328509

/-- The solution set of the inequality a*x^2 - (a+2)*x + 2 < 0 for x, where a ∈ ℝ -/
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x > 1 }
  else if 0 < a ∧ a < 2 then { x | 1 < x ∧ x < 2/a }
  else if a = 2 then ∅
  else if a > 2 then { x | 2/a < x ∧ x < 1 }
  else { x | x < 2/a ∨ x > 1 }

/-- Theorem stating that the solution_set function correctly describes the solutions of the inequality -/
theorem solution_set_correct (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a*x^2 - (a+2)*x + 2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l3285_328509


namespace NUMINAMATH_CALUDE_min_max_theorem_l3285_328531

theorem min_max_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1/x + 1/y ≥ 2) ∧ (x * (y + 1) ≤ 9/4) := by
  sorry

end NUMINAMATH_CALUDE_min_max_theorem_l3285_328531


namespace NUMINAMATH_CALUDE_roots_when_p_is_8_p_value_when_root_is_3_plus_4i_l3285_328524

-- Define the complex quadratic equation
def complex_quadratic (p : ℝ) (x : ℂ) : ℂ := x^2 - p*x + 25

-- Part 1: Prove that when p = 8, the roots are 4 + 3i and 4 - 3i
theorem roots_when_p_is_8 :
  let p : ℝ := 8
  let x₁ : ℂ := 4 + 3*I
  let x₂ : ℂ := 4 - 3*I
  complex_quadratic p x₁ = 0 ∧ complex_quadratic p x₂ = 0 :=
sorry

-- Part 2: Prove that when one root is 3 + 4i, p = 6
theorem p_value_when_root_is_3_plus_4i :
  let x₁ : ℂ := 3 + 4*I
  ∃ p : ℝ, complex_quadratic p x₁ = 0 ∧ p = 6 :=
sorry

end NUMINAMATH_CALUDE_roots_when_p_is_8_p_value_when_root_is_3_plus_4i_l3285_328524


namespace NUMINAMATH_CALUDE_joe_trip_theorem_l3285_328513

/-- Represents Joe's trip expenses and calculations -/
def joe_trip (exchange_rate : ℝ) (initial_savings flight hotel food transportation entertainment miscellaneous : ℝ) : Prop :=
  let total_savings_aud := initial_savings * exchange_rate
  let total_expenses_usd := flight + hotel + food + transportation + entertainment + miscellaneous
  let total_expenses_aud := total_expenses_usd * exchange_rate
  let amount_left := total_savings_aud - total_expenses_aud
  (total_expenses_aud = 9045) ∧ (amount_left = -945)

/-- Theorem stating the correctness of Joe's trip calculations -/
theorem joe_trip_theorem : 
  joe_trip 1.35 6000 1200 800 3000 500 850 350 := by
  sorry

end NUMINAMATH_CALUDE_joe_trip_theorem_l3285_328513


namespace NUMINAMATH_CALUDE_product_distribution_l3285_328548

theorem product_distribution (n : ℕ) (h : n = 6) :
  (Nat.choose n 1) * (Nat.choose (n - 1) 2) * (Nat.choose (n - 3) 3) =
  (Nat.choose n 1) * (Nat.choose (n - 1) 2) * (Nat.choose (n - 3) 3) :=
by sorry

end NUMINAMATH_CALUDE_product_distribution_l3285_328548


namespace NUMINAMATH_CALUDE_function_inequality_l3285_328517

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, f x > (deriv f) x) : 
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3285_328517


namespace NUMINAMATH_CALUDE_expression_simplification_l3285_328508

theorem expression_simplification (x : ℝ) : 
  3*x*(3*x^2 - 2*x + 1) - 2*x^2 + x = 9*x^3 - 8*x^2 + 4*x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3285_328508


namespace NUMINAMATH_CALUDE_pizza_piece_cost_l3285_328504

/-- Given that 4 pizzas cost $80 in total, and each pizza is cut into 5 pieces,
    prove that the cost of each piece of pizza is $4. -/
theorem pizza_piece_cost : 
  (total_cost : ℝ) →
  (num_pizzas : ℕ) →
  (pieces_per_pizza : ℕ) →
  total_cost = 80 →
  num_pizzas = 4 →
  pieces_per_pizza = 5 →
  (total_cost / (num_pizzas * pieces_per_pizza : ℝ)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_pizza_piece_cost_l3285_328504


namespace NUMINAMATH_CALUDE_sufficient_condition_product_greater_than_one_l3285_328582

theorem sufficient_condition_product_greater_than_one :
  ∀ a b : ℝ, a > 1 → b > 1 → a * b > 1 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_product_greater_than_one_l3285_328582


namespace NUMINAMATH_CALUDE_weight_replacement_l3285_328554

theorem weight_replacement (initial_count : ℕ) (weight_increase : ℚ) (new_weight : ℚ) :
  initial_count = 8 →
  weight_increase = 5/2 →
  new_weight = 40 →
  ∃ (old_weight : ℚ),
    old_weight = new_weight - (initial_count * weight_increase) ∧
    old_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l3285_328554


namespace NUMINAMATH_CALUDE_muffin_to_banana_ratio_l3285_328530

/-- The cost of a muffin -/
def muffin_cost : ℝ := sorry

/-- The cost of a banana -/
def banana_cost : ℝ := sorry

/-- Kristy's total cost -/
def kristy_cost : ℝ := 5 * muffin_cost + 4 * banana_cost

/-- Tim's total cost -/
def tim_cost : ℝ := 3 * muffin_cost + 20 * banana_cost

/-- The theorem stating the ratio of muffin cost to banana cost -/
theorem muffin_to_banana_ratio :
  tim_cost = 3 * kristy_cost →
  muffin_cost = (2/3) * banana_cost :=
by sorry

end NUMINAMATH_CALUDE_muffin_to_banana_ratio_l3285_328530


namespace NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l3285_328503

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fourth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 0 = 23)
  (h_last : a 5 = 59) :
  a 3 = 41 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l3285_328503


namespace NUMINAMATH_CALUDE_smallest_integer_l3285_328533

theorem smallest_integer (a b : ℕ+) (h1 : a = 60) (h2 : (Nat.lcm a b) / (Nat.gcd a b) = 44) : 
  b ≥ 165 ∧ ∃ (b' : ℕ+), b' = 165 ∧ (Nat.lcm a b') / (Nat.gcd a b') = 44 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_l3285_328533


namespace NUMINAMATH_CALUDE_equilibrium_shift_without_K_change_l3285_328580

-- Define the type for factors that can influence chemical equilibrium
inductive EquilibriumFactor
  | Temperature
  | Concentration
  | Pressure
  | Catalyst

-- Define a function to represent if a factor changes the equilibrium constant
def changesK (factor : EquilibriumFactor) : Prop :=
  match factor with
  | EquilibriumFactor.Temperature => True
  | _ => False

-- Define a function to represent if a factor can shift the equilibrium
def canShiftEquilibrium (factor : EquilibriumFactor) : Prop :=
  match factor with
  | EquilibriumFactor.Temperature => True
  | EquilibriumFactor.Concentration => True
  | EquilibriumFactor.Pressure => True
  | EquilibriumFactor.Catalyst => True

-- Theorem stating that there exists a factor that can shift equilibrium without changing K
theorem equilibrium_shift_without_K_change :
  ∃ (factor : EquilibriumFactor), canShiftEquilibrium factor ∧ ¬changesK factor :=
by
  sorry


end NUMINAMATH_CALUDE_equilibrium_shift_without_K_change_l3285_328580


namespace NUMINAMATH_CALUDE_robert_coin_arrangements_l3285_328518

/-- Represents the number of distinguishable arrangements of coins -/
def coin_arrangements (gold_coins silver_coins : Nat) : Nat :=
  let total_coins := gold_coins + silver_coins
  let positions := Nat.choose total_coins gold_coins
  let orientations := 30  -- Simplified representation of valid orientations
  positions * orientations

/-- Theorem stating the number of distinguishable arrangements for the given problem -/
theorem robert_coin_arrangements :
  coin_arrangements 5 3 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_robert_coin_arrangements_l3285_328518


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l3285_328559

theorem power_fraction_simplification :
  (3^2023 + 3^2021) / (3^2023 - 3^2021) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l3285_328559


namespace NUMINAMATH_CALUDE_factor_tree_problem_l3285_328532

theorem factor_tree_problem (X Y Z F G : ℕ) : 
  X = Y * Z ∧ 
  Y = 7 * F ∧ 
  Z = 11 * G ∧ 
  F = 2 * 5 ∧ 
  G = 7 * 3 → 
  X = 16170 := by
sorry


end NUMINAMATH_CALUDE_factor_tree_problem_l3285_328532


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3285_328595

/-- An isosceles triangle with side lengths 3, 6, and 6 has a perimeter of 15. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 6 ∧ c = 6 →  -- Two sides are 6, one side is 3
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a + b + c = 15  -- Perimeter is 15
:= by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3285_328595


namespace NUMINAMATH_CALUDE_parabola_chord_intersection_l3285_328550

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16*x

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop := parabola p.1 p.2

-- Define perpendicular vectors
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Theorem statement
theorem parabola_chord_intersection :
  ∀ (A B : ℝ × ℝ),
  point_on_parabola A →
  point_on_parabola B →
  perpendicular A B →
  ∃ (t : ℝ), A.1 = t * A.2 + 16 ∧ B.1 = t * B.2 + 16 :=
sorry

end NUMINAMATH_CALUDE_parabola_chord_intersection_l3285_328550


namespace NUMINAMATH_CALUDE_computer_lab_setup_l3285_328579

-- Define the cost of computers and investment range
def standard_teacher_cost : ℕ := 8000
def standard_student_cost : ℕ := 3500
def advanced_teacher_cost : ℕ := 11500
def advanced_student_cost : ℕ := 7000
def min_investment : ℕ := 200000
def max_investment : ℕ := 210000

-- Define the number of student computers in each lab
def standard_students : ℕ := 55
def advanced_students : ℕ := 27

-- Theorem stating the problem
theorem computer_lab_setup :
  (standard_teacher_cost + standard_student_cost * standard_students = 
   advanced_teacher_cost + advanced_student_cost * advanced_students) ∧
  (min_investment < standard_teacher_cost + standard_student_cost * standard_students) ∧
  (standard_teacher_cost + standard_student_cost * standard_students < max_investment) ∧
  (min_investment < advanced_teacher_cost + advanced_student_cost * advanced_students) ∧
  (advanced_teacher_cost + advanced_student_cost * advanced_students < max_investment) := by
  sorry

end NUMINAMATH_CALUDE_computer_lab_setup_l3285_328579


namespace NUMINAMATH_CALUDE_rachel_furniture_assembly_time_l3285_328539

/-- Calculates the total assembly time for furniture --/
def total_assembly_time (chairs tables bookshelves : ℕ) 
  (chair_time table_time bookshelf_time : ℕ) : ℕ :=
  chairs * chair_time + tables * table_time + bookshelves * bookshelf_time

/-- Proves that the total assembly time for Rachel's furniture is 244 minutes --/
theorem rachel_furniture_assembly_time :
  total_assembly_time 20 8 5 6 8 12 = 244 := by
  sorry

end NUMINAMATH_CALUDE_rachel_furniture_assembly_time_l3285_328539


namespace NUMINAMATH_CALUDE_lauras_average_speed_l3285_328597

def first_distance : ℝ := 420
def first_time : ℝ := 6.5
def second_distance : ℝ := 480
def second_time : ℝ := 8.25

def total_distance : ℝ := first_distance + second_distance
def total_time : ℝ := first_time + second_time

theorem lauras_average_speed :
  total_distance / total_time = 900 / 14.75 := by sorry

end NUMINAMATH_CALUDE_lauras_average_speed_l3285_328597


namespace NUMINAMATH_CALUDE_equation_solution_range_l3285_328567

theorem equation_solution_range (a : ℝ) : 
  (∃ x : ℝ, 9^x + (a+4)*3^x + 4 = 0) ↔ a ≤ -8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3285_328567


namespace NUMINAMATH_CALUDE_problem_pyramid_rows_l3285_328523

/-- Represents a pyramid display of cans -/
structure CanPyramid where
  topRowCans : ℕ
  rowIncrement : ℕ
  totalCans : ℕ

/-- Calculates the number of rows in a can pyramid -/
def numberOfRows (p : CanPyramid) : ℕ :=
  sorry

/-- The specific can pyramid from the problem -/
def problemPyramid : CanPyramid :=
  { topRowCans := 3
  , rowIncrement := 3
  , totalCans := 225 }

/-- Theorem stating that the number of rows in the problem pyramid is 12 -/
theorem problem_pyramid_rows :
  numberOfRows problemPyramid = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_pyramid_rows_l3285_328523


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3285_328540

theorem scientific_notation_equivalence : ∃ (a : ℝ) (n : ℤ), 
  0.00000065 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 6.5 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3285_328540


namespace NUMINAMATH_CALUDE_no_right_triangle_with_given_conditions_l3285_328514

theorem no_right_triangle_with_given_conditions :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b = 8 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_given_conditions_l3285_328514


namespace NUMINAMATH_CALUDE_no_valid_cd_l3285_328556

theorem no_valid_cd : ¬ ∃ (C D : ℕ+), 
  (Nat.lcm C D = 210) ∧ 
  (C : ℚ) / (D : ℚ) = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_no_valid_cd_l3285_328556


namespace NUMINAMATH_CALUDE_comic_book_ratio_l3285_328516

def initial_books : ℕ := 22
def final_books : ℕ := 17
def bought_books : ℕ := 6

theorem comic_book_ratio : 
  ∃ (sold_books : ℕ), 
    initial_books - sold_books + bought_books = final_books ∧
    sold_books * 2 = initial_books := by
  sorry

end NUMINAMATH_CALUDE_comic_book_ratio_l3285_328516


namespace NUMINAMATH_CALUDE_qin_jiushao_correct_f_3_equals_22542_l3285_328585

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qin_jiushao (f : ℝ → ℝ) (x : ℝ) : ℝ := 
  let v0 := 1
  let v1 := x * v0 + 2
  let v2 := x * v1 + 0
  let v3 := x * v2 + 4
  let v4 := x * v3 + 5
  let v5 := x * v4 + 6
  x * v5 + 12

/-- The polynomial f(x) = x^6 + 2x^5 + 4x^3 + 5x^2 + 6x + 12 -/
def f (x : ℝ) : ℝ := x^6 + 2*x^5 + 4*x^3 + 5*x^2 + 6*x + 12

/-- Theorem: Qin Jiushao's algorithm correctly evaluates f(3) -/
theorem qin_jiushao_correct : qin_jiushao f 3 = 22542 := by
  sorry

/-- Theorem: f(3) equals 22542 -/
theorem f_3_equals_22542 : f 3 = 22542 := by
  sorry

end NUMINAMATH_CALUDE_qin_jiushao_correct_f_3_equals_22542_l3285_328585
