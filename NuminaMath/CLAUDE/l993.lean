import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l993_99382

theorem sum_of_roots_zero (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 6*x^2 - x + 6
  ∃ a b c d : ℝ, (∀ x, f x = (x^2 + a*x + b) * (x^2 + c*x + d)) →
  (a + b + c + d = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l993_99382


namespace NUMINAMATH_CALUDE_parabola_vertex_and_a_range_l993_99300

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a^2 + 2*a

-- Define the line
def line (x : ℝ) : ℝ := 2*x - 2

-- Define the length of PQ
def PQ_length (a : ℝ) (m : ℝ) : ℝ := (m - (a + 1))^2 + 1

theorem parabola_vertex_and_a_range :
  (∀ x : ℝ, parabola 1 x ≥ 2) ∧
  (parabola 1 1 = 2) ∧
  (∀ a : ℝ, a > 0 → 
    (∀ m : ℝ, m < 3 → 
      (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < h → h < δ → 
        PQ_length a (m + h) < PQ_length a m
      ) → a ≥ 2
    )
  ) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_and_a_range_l993_99300


namespace NUMINAMATH_CALUDE_inscribed_rectangle_exists_l993_99390

/-- Represents a right triangle with sides 3, 4, and 5 -/
structure EgyptianTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 3
  hb : b = 4
  hc : c = 5
  right_angle : a^2 + b^2 = c^2

/-- Represents a rectangle inscribed in the Egyptian triangle -/
structure InscribedRectangle (t : EgyptianTriangle) where
  width : ℝ
  height : ℝ
  ratio : width * 3 = height
  fits_in_triangle : width ≤ t.a ∧ width ≤ t.b ∧ height ≤ t.b ∧ height ≤ t.c

/-- The theorem stating the existence and dimensions of the inscribed rectangle -/
theorem inscribed_rectangle_exists (t : EgyptianTriangle) :
  ∃ (r : InscribedRectangle t), r.width = 20/29 ∧ r.height = 60/29 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_exists_l993_99390


namespace NUMINAMATH_CALUDE_larger_number_is_322_l993_99396

def is_hcf (a b h : ℕ) : Prop := h ∣ a ∧ h ∣ b ∧ ∀ d : ℕ, d ∣ a → d ∣ b → d ≤ h

def is_lcm (a b l : ℕ) : Prop := a ∣ l ∧ b ∣ l ∧ ∀ m : ℕ, a ∣ m → b ∣ m → l ∣ m

theorem larger_number_is_322 (a b : ℕ) (h : a > 0 ∧ b > 0) :
  is_hcf a b 23 → is_lcm a b (23 * 13 * 14) → max a b = 322 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_322_l993_99396


namespace NUMINAMATH_CALUDE_anya_additional_biscuits_l993_99369

/-- Represents the distribution of biscuits and payments among three sisters. -/
structure BiscuitDistribution where
  total_biscuits : ℕ
  total_payment : ℕ
  anya_payment : ℕ
  berini_payment : ℕ
  carla_payment : ℕ

/-- Calculates the number of additional biscuits Anya would receive if distributed proportionally to payments. -/
def additional_biscuits_for_anya (bd : BiscuitDistribution) : ℕ :=
  let equal_share := bd.total_biscuits / 3
  let proportional_share := (bd.anya_payment * bd.total_biscuits) / bd.total_payment
  proportional_share - equal_share

/-- Theorem stating that Anya would receive 6 more biscuits in a proportional distribution. -/
theorem anya_additional_biscuits :
  ∀ (bd : BiscuitDistribution),
  bd.total_biscuits = 30 ∧
  bd.total_payment = 150 ∧
  bd.anya_payment = 80 ∧
  bd.berini_payment = 50 ∧
  bd.carla_payment = 20 →
  additional_biscuits_for_anya bd = 6 := by
  sorry

end NUMINAMATH_CALUDE_anya_additional_biscuits_l993_99369


namespace NUMINAMATH_CALUDE_nineteen_percent_female_officers_on_duty_l993_99378

/-- Calculates the percentage of female officers on duty given the total officers on duty,
    the fraction of officers on duty who are female, and the total number of female officers. -/
def percentage_female_officers_on_duty (total_on_duty : ℕ) (fraction_female : ℚ) (total_female : ℕ) : ℚ :=
  (fraction_female * total_on_duty : ℚ) / total_female * 100

/-- Theorem stating that 19% of female officers were on duty that night. -/
theorem nineteen_percent_female_officers_on_duty :
  percentage_female_officers_on_duty 152 (1/2) 400 = 19 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_percent_female_officers_on_duty_l993_99378


namespace NUMINAMATH_CALUDE_seventh_person_age_l993_99313

theorem seventh_person_age
  (n : ℕ)
  (initial_people : ℕ)
  (future_average : ℕ)
  (new_average : ℕ)
  (years_passed : ℕ)
  (h1 : initial_people = 6)
  (h2 : future_average = 43)
  (h3 : new_average = 45)
  (h4 : years_passed = 2)
  (h5 : n = initial_people + 1) :
  (n * new_average) - (initial_people * (future_average + years_passed)) = 69 := by
  sorry

end NUMINAMATH_CALUDE_seventh_person_age_l993_99313


namespace NUMINAMATH_CALUDE_right_triangle_5_12_13_l993_99387

/-- A right triangle with sides a, b, and c satisfies the Pythagorean theorem --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The triple (5, 12, 13) forms a right triangle --/
theorem right_triangle_5_12_13 :
  is_right_triangle 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_5_12_13_l993_99387


namespace NUMINAMATH_CALUDE_karen_walnuts_l993_99394

/-- The amount of nuts in cups added to the trail mix -/
def total_nuts : ℝ := 0.5

/-- The amount of almonds in cups added to the trail mix -/
def almonds : ℝ := 0.25

/-- The amount of walnuts in cups added to the trail mix -/
def walnuts : ℝ := total_nuts - almonds

theorem karen_walnuts : walnuts = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_karen_walnuts_l993_99394


namespace NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l993_99365

/-- Represents a number in the sequence -/
def SequenceNumber (n : ℕ) : ℕ := 20142015 + n * 10^6

/-- The sum of digits for any number in the sequence -/
def DigitSum : ℕ := 15

/-- A number is a candidate for being a perfect square if its digit sum is 0, 1, 4, 7, or 9 mod 9 -/
def IsPerfectSquareCandidate (n : ℕ) : Prop :=
  n % 9 = 0 ∨ n % 9 = 1 ∨ n % 9 = 4 ∨ n % 9 = 7 ∨ n % 9 = 9

theorem no_perfect_squares_in_sequence :
  ∀ n : ℕ, ¬ ∃ m : ℕ, (SequenceNumber n) = m^2 :=
sorry

end NUMINAMATH_CALUDE_no_perfect_squares_in_sequence_l993_99365


namespace NUMINAMATH_CALUDE_moores_law_transistor_count_l993_99397

/-- Moore's law calculation for transistor count --/
theorem moores_law_transistor_count 
  (initial_year : Nat) 
  (final_year : Nat) 
  (initial_transistors : Nat) 
  (doubling_period : Nat) 
  (h1 : initial_year = 1985)
  (h2 : final_year = 2010)
  (h3 : initial_transistors = 300000)
  (h4 : doubling_period = 2) :
  let years_passed := final_year - initial_year
  let doublings := years_passed / doubling_period
  initial_transistors * (2 ^ doublings) = 1228800000 :=
by sorry

end NUMINAMATH_CALUDE_moores_law_transistor_count_l993_99397


namespace NUMINAMATH_CALUDE_square_roots_of_sqrt_256_is_correct_l993_99328

-- Define the set of square roots of √256
def square_roots_of_sqrt_256 : Set ℝ :=
  {x : ℝ | x ^ 2 = Real.sqrt 256}

-- Theorem statement
theorem square_roots_of_sqrt_256_is_correct :
  square_roots_of_sqrt_256 = {-4, 4} := by
sorry

end NUMINAMATH_CALUDE_square_roots_of_sqrt_256_is_correct_l993_99328


namespace NUMINAMATH_CALUDE_floor_painting_cost_l993_99310

theorem floor_painting_cost 
  (length : ℝ) 
  (paint_rate : ℝ) 
  (length_ratio : ℝ) : 
  length = 12.24744871391589 →
  paint_rate = 2 →
  length_ratio = 3 →
  (length * (length / length_ratio)) * paint_rate = 100 := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_cost_l993_99310


namespace NUMINAMATH_CALUDE_ball_reaches_top_left_pocket_l993_99399

/-- Represents a point on the billiard table or its reflections -/
structure TablePoint where
  x : Int
  y : Int

/-- Represents the dimensions of the billiard table -/
structure TableDimensions where
  width : Nat
  height : Nat

/-- Checks if a point is a top-left pocket in the reflected grid -/
def isTopLeftPocket (p : TablePoint) (dim : TableDimensions) : Prop :=
  ∃ (m n : Int), p.x = dim.width * m ∧ p.y = dim.height * n ∧ m % 2 = 0 ∧ n % 2 = 1

/-- The theorem stating that the ball will reach the top-left pocket -/
theorem ball_reaches_top_left_pocket (dim : TableDimensions) 
  (h_dim : dim.width = 1965 ∧ dim.height = 26) :
  ∃ (p : TablePoint), p.y = p.x ∧ isTopLeftPocket p dim := by
  sorry

end NUMINAMATH_CALUDE_ball_reaches_top_left_pocket_l993_99399


namespace NUMINAMATH_CALUDE_remainder_3_pow_2024_mod_17_l993_99340

theorem remainder_3_pow_2024_mod_17 : 3^2024 % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2024_mod_17_l993_99340


namespace NUMINAMATH_CALUDE_ball_bounce_height_l993_99305

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (h₁ : h₀ = 1000) (h₂ : r = 1/2) :
  ∃ k : ℕ, k > 0 ∧ h₀ * r^k < 1 ∧ ∀ j : ℕ, 0 < j → j < k → h₀ * r^j ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_ball_bounce_height_l993_99305


namespace NUMINAMATH_CALUDE_quadratic_symmetry_inequality_l993_99325

/-- Given real numbers a, b, c, and a quadratic function f(x) = ax^2 + bx + c
    that is symmetric about x = 1, prove that f(1-a) < f(1-2a) < f(1) is impossible. -/
theorem quadratic_symmetry_inequality (a b c : ℝ) 
    (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x^2 + b * x + c) 
    (h_sym : ∀ x, f x = f (2 - x)) : 
  ¬(f (1 - a) < f (1 - 2*a) ∧ f (1 - 2*a) < f 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_inequality_l993_99325


namespace NUMINAMATH_CALUDE_train_speed_problem_l993_99345

/-- Proves that the original speed of a train is 60 km/h given the specified conditions -/
theorem train_speed_problem (delay : Real) (distance : Real) (speed_increase : Real) :
  delay = 0.2 ∧ distance = 60 ∧ speed_increase = 15 →
  ∃ original_speed : Real,
    original_speed > 0 ∧
    distance / original_speed - distance / (original_speed + speed_increase) = delay ∧
    original_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l993_99345


namespace NUMINAMATH_CALUDE_complex_square_plus_self_l993_99338

theorem complex_square_plus_self : 
  let z : ℂ := 1 + Complex.I
  z^2 + z = 1 + 3 * Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_square_plus_self_l993_99338


namespace NUMINAMATH_CALUDE_students_not_enrolled_l993_99366

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 6 := by
sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l993_99366


namespace NUMINAMATH_CALUDE_a_in_S_l993_99367

def S : Set ℤ := {n | ∃ x y : ℤ, n = x^2 + 2*y^2}

theorem a_in_S (a : ℤ) (h : 3*a ∈ S) : a ∈ S := by
  sorry

end NUMINAMATH_CALUDE_a_in_S_l993_99367


namespace NUMINAMATH_CALUDE_at_most_two_solutions_l993_99373

theorem at_most_two_solutions (m : ℕ) : 
  ∃ (a₁ a₂ : ℤ), ∀ (a : ℤ), 
    (⌊(a : ℝ) - Real.sqrt (a : ℝ)⌋ = m) → (a = a₁ ∨ a = a₂) :=
sorry

end NUMINAMATH_CALUDE_at_most_two_solutions_l993_99373


namespace NUMINAMATH_CALUDE_brochure_printing_problem_l993_99377

/-- Represents the number of pages printed for the spreads for which the press prints a block of 4 ads -/
def pages_per_ad_block : ℕ := by sorry

theorem brochure_printing_problem :
  let single_page_spreads : ℕ := 20
  let double_page_spreads : ℕ := 2 * single_page_spreads
  let pages_per_brochure : ℕ := 5
  let total_brochures : ℕ := 25
  let total_pages : ℕ := total_brochures * pages_per_brochure
  let pages_from_double_spreads : ℕ := double_page_spreads * 2
  let remaining_pages : ℕ := total_pages - pages_from_double_spreads
  let unused_single_spreads : ℕ := single_page_spreads - remaining_pages
  pages_per_ad_block = unused_single_spreads := by sorry

end NUMINAMATH_CALUDE_brochure_printing_problem_l993_99377


namespace NUMINAMATH_CALUDE_cafe_order_combinations_l993_99315

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- The number of distinct meal combinations for two people choosing from a menu with a given number of items, where order matters and repetition is allowed -/
def meal_combinations (items : ℕ) : ℕ := items ^ num_people

theorem cafe_order_combinations :
  meal_combinations menu_items = 225 := by
  sorry

end NUMINAMATH_CALUDE_cafe_order_combinations_l993_99315


namespace NUMINAMATH_CALUDE_flag_combinations_l993_99363

def num_colors : ℕ := 2
def num_stripes : ℕ := 3

theorem flag_combinations : (num_colors ^ num_stripes : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_flag_combinations_l993_99363


namespace NUMINAMATH_CALUDE_max_sum_of_diagonals_l993_99371

/-- A rhombus with side length 5 and diagonals d1 and d2 where d1 ≤ 6 and d2 ≥ 6 -/
structure Rhombus where
  side_length : ℝ
  d1 : ℝ
  d2 : ℝ
  side_is_5 : side_length = 5
  d1_le_6 : d1 ≤ 6
  d2_ge_6 : d2 ≥ 6

/-- The maximum sum of diagonals in the given rhombus is 14 -/
theorem max_sum_of_diagonals (r : Rhombus) : (r.d1 + r.d2 ≤ 14) ∧ (∃ (s : Rhombus), s.d1 + s.d2 = 14) := by
  sorry


end NUMINAMATH_CALUDE_max_sum_of_diagonals_l993_99371


namespace NUMINAMATH_CALUDE_gcd_111_1850_l993_99331

theorem gcd_111_1850 : Nat.gcd 111 1850 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_111_1850_l993_99331


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l993_99388

def numbers : List ℝ := [15, 23, 37, 45]

theorem arithmetic_mean_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 30 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l993_99388


namespace NUMINAMATH_CALUDE_regular_polygon_center_containment_l993_99347

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) where
  sideLength : ℝ
  center : ℝ × ℝ

/-- M1 is situated inside M2 -/
def isInside (M1 M2 : RegularPolygon n) : Prop :=
  sorry

/-- The center of a polygon -/
def centerOf (M : RegularPolygon n) : ℝ × ℝ :=
  M.center

/-- A point is contained in a polygon -/
def contains (M : RegularPolygon n) (p : ℝ × ℝ) : Prop :=
  sorry

theorem regular_polygon_center_containment (n : ℕ) (M1 M2 : RegularPolygon n) 
  (h1 : M1.sideLength = a)
  (h2 : M2.sideLength = 2 * a)
  (h3 : isInside M1 M2)
  : contains M1 (centerOf M2) :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_center_containment_l993_99347


namespace NUMINAMATH_CALUDE_unique_rectangle_pieces_l993_99309

theorem unique_rectangle_pieces :
  ∀ (a b : ℕ),
    a < b →
    (49 * 51) % (a * b) = 0 →
    (99 * 101) % (a * b) = 0 →
    a = 1 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_rectangle_pieces_l993_99309


namespace NUMINAMATH_CALUDE_evaluate_expression_l993_99324

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 3/4) (hz : z = 8) :
  x^2 * y^3 * z = 27/128 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l993_99324


namespace NUMINAMATH_CALUDE_certain_number_sum_l993_99334

theorem certain_number_sum (x : ℤ) : 47 + x = 30 → x = -17 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_sum_l993_99334


namespace NUMINAMATH_CALUDE_watermelon_seeds_count_l993_99358

/-- The number of seeds in each watermelon -/
def seeds_per_watermelon : ℕ := 345

/-- The number of watermelons -/
def number_of_watermelons : ℕ := 27

/-- The total number of seeds in all watermelons -/
def total_seeds : ℕ := seeds_per_watermelon * number_of_watermelons

theorem watermelon_seeds_count : total_seeds = 9315 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_seeds_count_l993_99358


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l993_99319

/-- Prove that arctan(tan 75° - 2 tan 30°) = 75° --/
theorem arctan_tan_difference (π : Real) : 
  let deg_to_rad : Real → Real := (· * π / 180)
  Real.arctan (Real.tan (deg_to_rad 75) - 2 * Real.tan (deg_to_rad 30)) = deg_to_rad 75 := by
  sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l993_99319


namespace NUMINAMATH_CALUDE_system_solution_l993_99386

theorem system_solution (x y : ℝ) : 
  (4 * (x - y) = 8 - 3 * y) ∧ 
  (x / 2 + y / 3 = 1) ↔ 
  (x = 2 ∧ y = 0) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l993_99386


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l993_99356

theorem complex_ratio_theorem (a b : ℝ) (z : ℂ) (h1 : z = Complex.mk a b) 
  (h2 : ∃ (k : ℝ), z / Complex.mk 2 1 = Complex.mk 0 k) : b / a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l993_99356


namespace NUMINAMATH_CALUDE_train_length_calculation_train_length_proof_l993_99391

theorem train_length_calculation (jogger_speed : ℝ) (train_speed : ℝ) 
  (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed - jogger_speed
  let distance_traveled := relative_speed * passing_time
  let train_length := distance_traveled - initial_distance
  train_length

theorem train_length_proof :
  train_length_calculation 2.5 12.5 240 36 = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_train_length_proof_l993_99391


namespace NUMINAMATH_CALUDE_water_left_in_cooler_l993_99370

/-- Calculates the remaining water in a cooler after filling Dixie cups for a meeting --/
theorem water_left_in_cooler 
  (initial_gallons : ℕ) 
  (ounces_per_cup : ℕ) 
  (rows : ℕ) 
  (chairs_per_row : ℕ) 
  (ounces_per_gallon : ℕ) 
  (h1 : initial_gallons = 3)
  (h2 : ounces_per_cup = 6)
  (h3 : rows = 5)
  (h4 : chairs_per_row = 10)
  (h5 : ounces_per_gallon = 128) : 
  initial_gallons * ounces_per_gallon - rows * chairs_per_row * ounces_per_cup = 84 := by
  sorry

end NUMINAMATH_CALUDE_water_left_in_cooler_l993_99370


namespace NUMINAMATH_CALUDE_soccer_lineup_combinations_l993_99357

def team_size : ℕ := 16
def non_goalkeeper : ℕ := 1
def lineup_positions : ℕ := 4

theorem soccer_lineup_combinations :
  (team_size - non_goalkeeper) *
  (team_size - 1) *
  (team_size - 2) *
  (team_size - 3) = 42210 :=
by sorry

end NUMINAMATH_CALUDE_soccer_lineup_combinations_l993_99357


namespace NUMINAMATH_CALUDE_total_class_time_l993_99302

def hours_per_day : ℝ := 8
def days_per_week : ℝ := 5
def math_percentage : ℝ := 0.25
def language_percentage : ℝ := 0.30
def science_percentage : ℝ := 0.20
def history_percentage : ℝ := 0.10

theorem total_class_time :
  let total_hours := hours_per_day * days_per_week
  let math_hours := total_hours * math_percentage
  let language_hours := total_hours * language_percentage
  let science_hours := total_hours * science_percentage
  let history_hours := total_hours * history_percentage
  math_hours + language_hours + science_hours + history_hours = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_class_time_l993_99302


namespace NUMINAMATH_CALUDE_train_passing_pole_time_l993_99327

/-- Proves that a train 150 meters long running at 90 km/hr takes 6 seconds to pass a pole. -/
theorem train_passing_pole_time (train_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 150 ∧ train_speed_kmh = 90 →
  (train_length / (train_speed_kmh * (1000 / 3600))) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_pole_time_l993_99327


namespace NUMINAMATH_CALUDE_obtuse_triangle_from_altitudes_l993_99333

theorem obtuse_triangle_from_altitudes (h₁ h₂ h₃ : ℝ) 
  (h_pos : h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0) 
  (h_ineq : 1/h₁ + 1/h₂ > 1/h₃ ∧ 1/h₂ + 1/h₃ > 1/h₁ ∧ 1/h₃ + 1/h₁ > 1/h₂) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    h₁ = (2 * (a * b * c / (a + b + c))) / a ∧
    h₂ = (2 * (a * b * c / (a + b + c))) / b ∧
    h₃ = (2 * (a * b * c / (a + b + c))) / c ∧
    a^2 + b^2 < c^2 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_from_altitudes_l993_99333


namespace NUMINAMATH_CALUDE_trig_expression_equals_negative_one_l993_99303

theorem trig_expression_equals_negative_one :
  (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) - Real.sin (6 * π / 180) * Real.sin (66 * π / 180)) /
  (Real.sin (21 * π / 180) * Real.cos (39 * π / 180) - Real.sin (39 * π / 180) * Real.cos (21 * π / 180)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_negative_one_l993_99303


namespace NUMINAMATH_CALUDE_missing_number_proof_l993_99385

theorem missing_number_proof :
  ∃ x : ℝ, 0.72 * 0.43 + x * 0.34 = 0.3504 ∧ abs (x - 0.12) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l993_99385


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l993_99393

theorem equilateral_triangle_area_decrease :
  ∀ s : ℝ,
  s > 0 →
  (s^2 * Real.sqrt 3) / 4 = 81 * Real.sqrt 3 →
  let s' := s - 3
  let new_area := (s'^2 * Real.sqrt 3) / 4
  let area_decrease := 81 * Real.sqrt 3 - new_area
  area_decrease = 24.75 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_decrease_l993_99393


namespace NUMINAMATH_CALUDE_basketball_game_points_l993_99344

/-- The total points scored by three players in a basketball game. -/
def total_points (jon_points jack_points tom_points : ℕ) : ℕ :=
  jon_points + jack_points + tom_points

/-- Theorem stating the total points scored by Jon, Jack, and Tom. -/
theorem basketball_game_points : ∃ (jack_points tom_points : ℕ),
  let jon_points := 3
  jack_points = jon_points + 5 ∧
  tom_points = (jon_points + jack_points) - 4 ∧
  total_points jon_points jack_points tom_points = 18 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_points_l993_99344


namespace NUMINAMATH_CALUDE_robert_birth_year_l993_99316

def first_amc8_year : ℕ := 1985

def amc8_year (n : ℕ) : ℕ := first_amc8_year + n - 1

def robert_age_at_tenth_amc8 : ℕ := 15

theorem robert_birth_year :
  ∃ (birth_year : ℕ),
    birth_year = amc8_year 10 - robert_age_at_tenth_amc8 ∧
    birth_year = 1979 :=
by sorry

end NUMINAMATH_CALUDE_robert_birth_year_l993_99316


namespace NUMINAMATH_CALUDE_nancy_homework_time_l993_99320

/-- The time required to finish all problems -/
def time_to_finish (math_problems : Float) (spelling_problems : Float) (problems_per_hour : Float) : Float :=
  (math_problems + spelling_problems) / problems_per_hour

/-- Proof that Nancy will take 4.0 hours to finish all problems -/
theorem nancy_homework_time : 
  time_to_finish 17.0 15.0 8.0 = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_nancy_homework_time_l993_99320


namespace NUMINAMATH_CALUDE_num_valid_selections_eq_twenty_l993_99336

/-- Represents a volunteer --/
inductive Volunteer : Type
  | A | B | C | D | E

/-- Represents a role --/
inductive Role : Type
  | Translator | TourGuide | Etiquette | Driver

/-- Predicate to check if a volunteer can perform a given role --/
def can_perform (v : Volunteer) (r : Role) : Prop :=
  match v, r with
  | Volunteer.A, Role.Translator => True
  | Volunteer.A, Role.TourGuide => True
  | Volunteer.B, Role.Translator => True
  | Volunteer.B, Role.TourGuide => True
  | Volunteer.C, _ => True
  | Volunteer.D, _ => True
  | Volunteer.E, _ => True
  | _, _ => False

/-- A valid selection is a function from Role to Volunteer satisfying the constraints --/
def ValidSelection : Type :=
  { f : Role → Volunteer // ∀ r, can_perform (f r) r ∧ ∀ r' ≠ r, f r ≠ f r' }

/-- The number of valid selections --/
def num_valid_selections : ℕ := sorry

/-- Theorem stating that the number of valid selections is 20 --/
theorem num_valid_selections_eq_twenty : num_valid_selections = 20 := by sorry

end NUMINAMATH_CALUDE_num_valid_selections_eq_twenty_l993_99336


namespace NUMINAMATH_CALUDE_probability_diamond_spade_heart_l993_99392

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Nat)
  (diamonds : Nat)
  (spades : Nat)
  (hearts : Nat)

/-- Calculates the probability of drawing a specific sequence of cards -/
def probability_specific_sequence (d : Deck) : ℚ :=
  (d.diamonds : ℚ) / d.cards *
  (d.spades : ℚ) / (d.cards - 1) *
  (d.hearts : ℚ) / (d.cards - 2)

/-- A standard deck of 52 cards with 13 cards of each suit -/
def standard_deck : Deck :=
  { cards := 52,
    diamonds := 13,
    spades := 13,
    hearts := 13 }

theorem probability_diamond_spade_heart :
  probability_specific_sequence standard_deck = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_probability_diamond_spade_heart_l993_99392


namespace NUMINAMATH_CALUDE_watermelon_pricing_l993_99383

/-- Represents the number of watermelons sold by each student in the morning -/
structure MorningSales where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the prices of watermelons -/
structure Prices where
  morning : ℚ
  afternoon : ℚ

/-- Theorem statement for the watermelon pricing problem -/
theorem watermelon_pricing
  (sales : MorningSales)
  (prices : Prices)
  (h1 : prices.morning > prices.afternoon)
  (h2 : prices.afternoon > 0)
  (h3 : sales.first < 10)
  (h4 : sales.second < 16)
  (h5 : sales.third < 26)
  (h6 : prices.morning * sales.first + prices.afternoon * (10 - sales.first) = 42)
  (h7 : prices.morning * sales.second + prices.afternoon * (16 - sales.second) = 42)
  (h8 : prices.morning * sales.third + prices.afternoon * (26 - sales.third) = 42)
  : prices.morning = 4.5 ∧ prices.afternoon = 1.5 := by
  sorry

#check watermelon_pricing

end NUMINAMATH_CALUDE_watermelon_pricing_l993_99383


namespace NUMINAMATH_CALUDE_shirts_bought_l993_99350

/-- Given John's initial and final shirt counts, prove the number of shirts bought. -/
theorem shirts_bought (initial_shirts final_shirts : ℕ) 
  (h1 : initial_shirts = 12)
  (h2 : final_shirts = 16)
  : final_shirts - initial_shirts = 4 := by
  sorry

end NUMINAMATH_CALUDE_shirts_bought_l993_99350


namespace NUMINAMATH_CALUDE_pencil_eraser_combinations_l993_99398

/-- The number of possible combinations when choosing one item from each of two sets -/
def combinations (set1 : ℕ) (set2 : ℕ) : ℕ := set1 * set2

/-- Theorem: The number of combinations when choosing one pencil from 2 types
    and one eraser from 3 types is equal to 6 -/
theorem pencil_eraser_combinations :
  combinations 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencil_eraser_combinations_l993_99398


namespace NUMINAMATH_CALUDE_no_solution_condition_l993_99364

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (3 - 2*x)/(x - 3) - (m*x - 2)/(3 - x) ≠ -1) ↔ 
  (m = 5/3 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_condition_l993_99364


namespace NUMINAMATH_CALUDE_condition_relationship_l993_99352

theorem condition_relationship (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (p → ¬q) ∧ ¬(¬q → p) := by
sorry

end NUMINAMATH_CALUDE_condition_relationship_l993_99352


namespace NUMINAMATH_CALUDE_factors_of_2520_l993_99337

theorem factors_of_2520 : Nat.card (Nat.divisors 2520) = 48 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_2520_l993_99337


namespace NUMINAMATH_CALUDE_problem_solution_l993_99312

theorem problem_solution (A B C D : ℕ+) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A * B = 72 →
  C * D = 72 →
  A - B = C * D →
  A = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l993_99312


namespace NUMINAMATH_CALUDE_melanie_caught_ten_l993_99322

/-- The number of trout Sara caught -/
def sara_trout : ℕ := 5

/-- The factor by which Melanie's catch exceeds Sara's -/
def melanie_factor : ℕ := 2

/-- The number of trout Melanie caught -/
def melanie_trout : ℕ := melanie_factor * sara_trout

theorem melanie_caught_ten : melanie_trout = 10 := by
  sorry

end NUMINAMATH_CALUDE_melanie_caught_ten_l993_99322


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l993_99314

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- a_n is a geometric sequence
  a 1 + a 2 = 3 →                           -- a_1 + a_2 = 3
  a 2 + a 3 = 6 →                           -- a_2 + a_3 = 6
  a 4 + a 5 = 24 :=                         -- a_4 + a_5 = 24
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l993_99314


namespace NUMINAMATH_CALUDE_factorization_theorem_l993_99308

theorem factorization_theorem (x y : ℝ) :
  4 * (x + y)^2 - (x^2 - y^2)^2 = (x + y)^2 * (2 + x - y) * (2 - x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l993_99308


namespace NUMINAMATH_CALUDE_hyperbola_focus_l993_99318

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the focus coordinates
def focus_coordinates : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola_equation x y ∧ focus_coordinates = (x, y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l993_99318


namespace NUMINAMATH_CALUDE_circles_are_tangent_l993_99362

/-- Represents a circle in the 2D plane -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Checks if two circles are tangent to each other -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let x1 := -c1.b / 2
  let y1 := -c1.c / 2
  let r1 := Real.sqrt (x1^2 + y1^2 - c1.e)
  let x2 := -c2.b / 2
  let y2 := -c2.c / 2
  let r2 := Real.sqrt (x2^2 + y2^2 - c2.e)
  let d := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)
  d = r1 + r2 ∨ d = abs (r1 - r2)

theorem circles_are_tangent : 
  let c1 : Circle := ⟨1, -6, 4, 1, 12⟩
  let c2 : Circle := ⟨1, -14, -2, 1, 14⟩
  are_tangent c1 c2 := by
  sorry

end NUMINAMATH_CALUDE_circles_are_tangent_l993_99362


namespace NUMINAMATH_CALUDE_sheet_reduction_percentage_l993_99335

def original_sheets : ℕ := 20
def original_lines_per_sheet : ℕ := 55
def original_chars_per_line : ℕ := 65

def retyped_lines_per_sheet : ℕ := 65
def retyped_chars_per_line : ℕ := 70

def total_chars : ℕ := original_sheets * original_lines_per_sheet * original_chars_per_line

def chars_per_retyped_sheet : ℕ := retyped_lines_per_sheet * retyped_chars_per_line

def retyped_sheets : ℕ := (total_chars + chars_per_retyped_sheet - 1) / chars_per_retyped_sheet

theorem sheet_reduction_percentage : 
  (original_sheets - retyped_sheets) * 100 / original_sheets = 20 := by
  sorry

end NUMINAMATH_CALUDE_sheet_reduction_percentage_l993_99335


namespace NUMINAMATH_CALUDE_solution_set_characterization_range_of_a_characterization_l993_99341

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - 2)

-- Part 1: Characterize the solution set of f(x) ≥ 4
theorem solution_set_characterization :
  {x : ℝ | f x ≥ 4} = {x : ℝ | x ≤ -4/3 ∨ x ≥ 0} :=
sorry

-- Part 2: Characterize the range of a
theorem range_of_a_characterization :
  {a : ℝ | ∀ x > 0, f x + a * x - 1 > 0} = {a : ℝ | a > -5/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_characterization_range_of_a_characterization_l993_99341


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l993_99317

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 486 → volume = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l993_99317


namespace NUMINAMATH_CALUDE_a_minus_c_value_l993_99389

theorem a_minus_c_value (a b c d : ℝ) 
  (h1 : (a + d + b + d) / 2 = 80)
  (h2 : (b + d + c + d) / 2 = 180)
  (h3 : d = 2 * (a - b)) : 
  a - c = -200 := by
sorry

end NUMINAMATH_CALUDE_a_minus_c_value_l993_99389


namespace NUMINAMATH_CALUDE_no_integer_solutions_l993_99351

theorem no_integer_solutions : ¬∃ (x y : ℤ), 3 * x^2 = 16 * y^2 + 8 * y + 5 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l993_99351


namespace NUMINAMATH_CALUDE_at_least_one_fraction_less_than_two_l993_99332

theorem at_least_one_fraction_less_than_two (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_fraction_less_than_two_l993_99332


namespace NUMINAMATH_CALUDE_exponent_sum_theorem_l993_99360

theorem exponent_sum_theorem : (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_exponent_sum_theorem_l993_99360


namespace NUMINAMATH_CALUDE_first_term_of_sequence_l993_99361

theorem first_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = n^2 + 1) : a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_sequence_l993_99361


namespace NUMINAMATH_CALUDE_sticker_distribution_l993_99376

theorem sticker_distribution (total : ℕ) (andrew_kept : ℕ) (daniel_received : ℕ) 
  (h1 : total = 750)
  (h2 : andrew_kept = 130)
  (h3 : daniel_received = 250) :
  total - andrew_kept - daniel_received - daniel_received = 120 :=
by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l993_99376


namespace NUMINAMATH_CALUDE_westward_plane_speed_l993_99372

/-- Given two planes traveling in opposite directions, this theorem calculates
    the speed of the westward-traveling plane. -/
theorem westward_plane_speed
  (east_speed : ℝ)
  (time : ℝ)
  (total_distance : ℝ)
  (h1 : east_speed = 325)
  (h2 : time = 3.5)
  (h3 : total_distance = 2100)
  : ∃ (west_speed : ℝ),
    west_speed = 275 ∧
    total_distance = (east_speed + west_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_westward_plane_speed_l993_99372


namespace NUMINAMATH_CALUDE_adventure_team_probabilities_l993_99342

def team_size : ℕ := 8
def medical_staff : ℕ := 3
def group_size : ℕ := 4

def probability_one_medical_in_one_group : ℚ := 6/7
def probability_at_least_two_medical_in_group : ℚ := 1/2
def expected_medical_in_group : ℚ := 3/2

theorem adventure_team_probabilities :
  (team_size = 8) →
  (medical_staff = 3) →
  (group_size = 4) →
  (probability_one_medical_in_one_group = 6/7) ∧
  (probability_at_least_two_medical_in_group = 1/2) ∧
  (expected_medical_in_group = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_adventure_team_probabilities_l993_99342


namespace NUMINAMATH_CALUDE_total_fruits_eq_137_l993_99395

/-- The number of fruits picked by George, Amelia, and Olivia --/
def total_fruits (george_oranges amelia_apples olivia_time olivia_rate_time olivia_rate_oranges olivia_rate_apples : ℕ) : ℕ :=
  let george_apples := amelia_apples + 5
  let amelia_oranges := george_oranges - 18
  let olivia_sets := olivia_time / olivia_rate_time
  let olivia_oranges := olivia_sets * olivia_rate_oranges
  let olivia_apples := olivia_sets * olivia_rate_apples
  (george_oranges + george_apples) + (amelia_oranges + amelia_apples) + (olivia_oranges + olivia_apples)

/-- Theorem stating the total number of fruits picked --/
theorem total_fruits_eq_137 :
  total_fruits 45 15 30 5 3 2 = 137 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_eq_137_l993_99395


namespace NUMINAMATH_CALUDE_product_OA_OC_constant_C_trajectory_l993_99381

-- Define the rhombus ABCD
structure Rhombus :=
  (A B C D : ℝ × ℝ)
  (side_length : ℝ)
  (is_rhombus : side_length = 4)
  (OB_length : ℝ)
  (OD_length : ℝ)
  (OB_OD_equal : OB_length = 6 ∧ OD_length = 6)

-- Define the function for |OA| * |OC|
def product_OA_OC (r : Rhombus) : ℝ := sorry

-- Define the function for the coordinates of C
def C_coordinates (r : Rhombus) (A_x A_y : ℝ) : ℝ × ℝ := sorry

-- Theorem 1: |OA| * |OC| is constant
theorem product_OA_OC_constant (r : Rhombus) : 
  product_OA_OC r = 20 := by sorry

-- Theorem 2: Trajectory of C
theorem C_trajectory (r : Rhombus) (A_x A_y : ℝ) 
  (h1 : (A_x - 2)^2 + A_y^2 = 4) (h2 : 2 ≤ A_x ∧ A_x ≤ 4) :
  ∃ (y : ℝ), C_coordinates r A_x A_y = (5, y) ∧ -5 ≤ y ∧ y ≤ 5 := by sorry

end NUMINAMATH_CALUDE_product_OA_OC_constant_C_trajectory_l993_99381


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l993_99349

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a →
  (a 1)^2 - 10*(a 1) + 16 = 0 →
  (a 2015)^2 - 10*(a 2015) + 16 = 0 →
  a 2 + a 1008 + a 2014 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l993_99349


namespace NUMINAMATH_CALUDE_teenas_speed_l993_99348

theorem teenas_speed (initial_distance : ℝ) (poes_speed : ℝ) (time : ℝ) (final_distance : ℝ) :
  initial_distance = 7.5 →
  poes_speed = 40 →
  time = 1.5 →
  final_distance = 15 →
  (initial_distance + poes_speed * time + final_distance) / time = 55 := by
sorry

end NUMINAMATH_CALUDE_teenas_speed_l993_99348


namespace NUMINAMATH_CALUDE_missing_number_proof_l993_99321

theorem missing_number_proof (x : ℝ) : 
  11 + Real.sqrt (-4 + 6 * 4 / x) = 13 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l993_99321


namespace NUMINAMATH_CALUDE_smallest_natural_number_square_cube_seventy_two_satisfies_conditions_smallest_natural_number_is_72_l993_99326

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_natural_number_square_cube : 
  ∀ x : ℕ, (is_perfect_square (2 * x) ∧ is_perfect_cube (3 * x)) → x ≥ 72 :=
by sorry

theorem seventy_two_satisfies_conditions : 
  is_perfect_square (2 * 72) ∧ is_perfect_cube (3 * 72) :=
by sorry

theorem smallest_natural_number_is_72 : 
  ∃! x : ℕ, x = 72 ∧ 
    (∀ y : ℕ, (is_perfect_square (2 * y) ∧ is_perfect_cube (3 * y)) → y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_natural_number_square_cube_seventy_two_satisfies_conditions_smallest_natural_number_is_72_l993_99326


namespace NUMINAMATH_CALUDE_f_inequality_l993_99343

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * exp x - k * x^2

theorem f_inequality (k : ℝ) (h1 : k > 1/2) 
  (h2 : ∀ x > 0, f k x + (log (2*k))^2 + 2*k * log (exp 1 / (2*k)) > 0) :
  f k (k - 1 + log 2) < f k k := by
sorry

end NUMINAMATH_CALUDE_f_inequality_l993_99343


namespace NUMINAMATH_CALUDE_discretionary_income_ratio_l993_99375

/-- Represents Jill's financial situation --/
structure JillFinances where
  netSalary : ℝ
  discretionaryIncome : ℝ
  vacationFundPercent : ℝ
  savingsPercent : ℝ
  socializingPercent : ℝ
  giftsAmount : ℝ

/-- The conditions of Jill's finances --/
def jillFinancesConditions (j : JillFinances) : Prop :=
  j.netSalary = 3300 ∧
  j.vacationFundPercent = 0.3 ∧
  j.savingsPercent = 0.2 ∧
  j.socializingPercent = 0.35 ∧
  j.giftsAmount = 99 ∧
  j.giftsAmount = (1 - (j.vacationFundPercent + j.savingsPercent + j.socializingPercent)) * j.discretionaryIncome

/-- The theorem stating the ratio of discretionary income to net salary --/
theorem discretionary_income_ratio (j : JillFinances) 
  (h : jillFinancesConditions j) : 
  j.discretionaryIncome / j.netSalary = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_discretionary_income_ratio_l993_99375


namespace NUMINAMATH_CALUDE_black_car_overtake_time_l993_99301

/-- Proves that the time for the black car to overtake the red car is 3 hours. -/
theorem black_car_overtake_time (red_speed black_speed initial_distance : ℝ) 
  (h1 : red_speed = 40)
  (h2 : black_speed = 50)
  (h3 : initial_distance = 30)
  (h4 : red_speed > 0)
  (h5 : black_speed > red_speed) :
  (initial_distance / (black_speed - red_speed)) = 3 := by
  sorry

#check black_car_overtake_time

end NUMINAMATH_CALUDE_black_car_overtake_time_l993_99301


namespace NUMINAMATH_CALUDE_least_acute_triangle_side_l993_99374

/-- A function that checks if three side lengths form an acute triangle -/
def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2

/-- The least positive integer A such that an acute triangle with side lengths 5, A, and 8 exists -/
theorem least_acute_triangle_side : ∃ (A : ℕ), 
  (∀ (k : ℕ), k < A → ¬is_acute_triangle 5 (k : ℝ) 8) ∧ 
  is_acute_triangle 5 A 8 ∧
  A = 7 := by
  sorry

end NUMINAMATH_CALUDE_least_acute_triangle_side_l993_99374


namespace NUMINAMATH_CALUDE_pebble_collection_sum_l993_99329

theorem pebble_collection_sum (n : ℕ) (h : n = 20) : 
  (List.range n).sum = 210 := by
  sorry

end NUMINAMATH_CALUDE_pebble_collection_sum_l993_99329


namespace NUMINAMATH_CALUDE_tim_cantaloupes_count_l993_99306

/-- The number of cantaloupes Fred grew -/
def fred_cantaloupes : ℕ := 38

/-- The total number of cantaloupes grown by Fred and Tim -/
def total_cantaloupes : ℕ := 82

/-- The number of cantaloupes Tim grew -/
def tim_cantaloupes : ℕ := total_cantaloupes - fred_cantaloupes

theorem tim_cantaloupes_count : tim_cantaloupes = 44 := by
  sorry

end NUMINAMATH_CALUDE_tim_cantaloupes_count_l993_99306


namespace NUMINAMATH_CALUDE_max_sum_with_condition_l993_99307

/-- Given positive integers a and b not exceeding 100 satisfying the condition,
    the maximum value of a + b is 78. -/
theorem max_sum_with_condition (a b : ℕ) : 
  0 < a ∧ 0 < b ∧ a ≤ 100 ∧ b ≤ 100 →
  a * b = (Nat.lcm a b / Nat.gcd a b) ^ 2 →
  ∀ (x y : ℕ), 0 < x ∧ 0 < y ∧ x ≤ 100 ∧ y ≤ 100 →
    x * y = (Nat.lcm x y / Nat.gcd x y) ^ 2 →
    a + b ≤ 78 ∧ (∃ (a' b' : ℕ), a' + b' = 78 ∧ 
      0 < a' ∧ 0 < b' ∧ a' ≤ 100 ∧ b' ≤ 100 ∧
      a' * b' = (Nat.lcm a' b' / Nat.gcd a' b') ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_condition_l993_99307


namespace NUMINAMATH_CALUDE_perpendicular_bisectors_intersection_l993_99359

-- Define a triangle as a structure with three points in a 2D plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem perpendicular_bisectors_intersection (t : Triangle) :
  ∃! O : ℝ × ℝ, distance O t.A = distance O t.B ∧ distance O t.A = distance O t.C := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisectors_intersection_l993_99359


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l993_99353

/-- Given a triangle ABC where:
  * The side opposite to angle A is 2
  * The side opposite to angle B is √2
  * Angle A measures 45°
Prove that angle B measures 30° -/
theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  b = Real.sqrt 2 →
  A = π / 4 →
  B = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l993_99353


namespace NUMINAMATH_CALUDE_num_cars_in_parking_lot_l993_99379

def num_bikes : ℕ := 10
def total_wheels : ℕ := 76
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

theorem num_cars_in_parking_lot : 
  (total_wheels - num_bikes * wheels_per_bike) / wheels_per_car = 14 := by
  sorry

end NUMINAMATH_CALUDE_num_cars_in_parking_lot_l993_99379


namespace NUMINAMATH_CALUDE_solution_existence_l993_99355

theorem solution_existence (k : ℝ) : 
  (∃ x ∈ Set.Icc 0 2, k * 9^x - k * 3^(x + 1) + 6 * (k - 5) = 0) ↔ k ∈ Set.Icc (1/2) 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_existence_l993_99355


namespace NUMINAMATH_CALUDE_group_size_l993_99311

theorem group_size (iceland : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : iceland = 35)
  (h2 : norway = 23)
  (h3 : both = 31)
  (h4 : neither = 33) :
  iceland + norway - both + neither = 60 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l993_99311


namespace NUMINAMATH_CALUDE_not_always_y_equals_a_when_x_zero_l993_99354

/-- Linear regression model -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope
  x_bar : ℝ  -- mean of x
  y_bar : ℝ  -- mean of y

/-- Predicted y value for a given x -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.b * x + model.a

/-- The regression line passes through the point (x_bar, y_bar) -/
axiom passes_through_mean (model : LinearRegression) :
  predict model model.x_bar = model.y_bar

/-- b represents the average change in y for a unit increase in x -/
axiom slope_interpretation (model : LinearRegression) (x₁ x₂ : ℝ) :
  predict model x₂ - predict model x₁ = model.b * (x₂ - x₁)

/-- Sample data point -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Theorem: It is not necessarily true that y = a when x = 0 in the sample data -/
theorem not_always_y_equals_a_when_x_zero (model : LinearRegression) :
  ∃ (data : DataPoint), data.x = 0 ∧ data.y ≠ model.a :=
sorry

end NUMINAMATH_CALUDE_not_always_y_equals_a_when_x_zero_l993_99354


namespace NUMINAMATH_CALUDE_escalator_length_l993_99339

/-- The length of an escalator given specific conditions -/
theorem escalator_length : 
  ∀ (escalator_speed person_speed time length : ℝ),
  escalator_speed = 12 →
  person_speed = 3 →
  time = 10 →
  length = (escalator_speed + person_speed) * time →
  length = 150 := by
sorry

end NUMINAMATH_CALUDE_escalator_length_l993_99339


namespace NUMINAMATH_CALUDE_tan_beta_value_l993_99368

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/7) 
  (h2 : Real.tan (α + β) = 1/3) : 
  Real.tan β = 2/11 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l993_99368


namespace NUMINAMATH_CALUDE_max_diagonal_value_l993_99330

/-- Represents a table with n rows and columns where the first column contains 1s
    and each row k is an arithmetic sequence with common difference k -/
def specialTable (n : ℕ) : ℕ → ℕ → ℕ :=
  fun k j => if j = 1 then 1 else 1 + (j - 1) * k

/-- The value on the diagonal from bottom-left to top-right at row k -/
def diagonalValue (n : ℕ) (k : ℕ) : ℕ :=
  specialTable n k (n - k + 1)

theorem max_diagonal_value :
  ∃ k, k ≤ 100 ∧ diagonalValue 100 k = 2501 ∧
  ∀ m, m ≤ 100 → diagonalValue 100 m ≤ 2501 := by
  sorry

end NUMINAMATH_CALUDE_max_diagonal_value_l993_99330


namespace NUMINAMATH_CALUDE_quadratic_equation_magnitude_l993_99346

theorem quadratic_equation_magnitude (z : ℂ) : 
  z^2 - 10*z + 28 = 0 → ∃! m : ℝ, ∃ z : ℂ, z^2 - 10*z + 28 = 0 ∧ Complex.abs z = m :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_magnitude_l993_99346


namespace NUMINAMATH_CALUDE_equation_solution_l993_99384

theorem equation_solution :
  ∃ x : ℝ, (x^2 + x ≠ 0) ∧ (x^2 - x ≠ 0) ∧
  (4 / (x^2 + x) - 3 / (x^2 - x) = 0) ∧ 
  (x = 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l993_99384


namespace NUMINAMATH_CALUDE_pure_imaginary_value_l993_99323

def complex_number (a : ℝ) : ℂ := (a^2 + 2*a - 3 : ℝ) + (a^2 - 4*a + 3 : ℝ) * Complex.I

theorem pure_imaginary_value (a : ℝ) :
  (complex_number a).re = 0 ∧ (complex_number a).im ≠ 0 → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_value_l993_99323


namespace NUMINAMATH_CALUDE_smallest_square_containing_rectangles_l993_99380

/-- The smallest square containing two non-overlapping rectangles -/
theorem smallest_square_containing_rectangles :
  ∀ (w₁ h₁ w₂ h₂ : ℕ),
  w₁ = 3 ∧ h₁ = 5 ∧ w₂ = 4 ∧ h₂ = 6 →
  ∃ (s : ℕ),
    s ≥ w₁ ∧ s ≥ h₁ ∧ s ≥ w₂ ∧ s ≥ h₂ ∧
    s ≥ w₁ + w₂ ∧ s ≥ h₁ ∧ s ≥ h₂ ∧
    (∀ (t : ℕ),
      t ≥ w₁ ∧ t ≥ h₁ ∧ t ≥ w₂ ∧ t ≥ h₂ ∧
      t ≥ w₁ + w₂ ∧ t ≥ h₁ ∧ t ≥ h₂ →
      t ≥ s) ∧
    s^2 = 49 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_containing_rectangles_l993_99380


namespace NUMINAMATH_CALUDE_angles_around_point_l993_99304

theorem angles_around_point (a b c : ℝ) : 
  a + b + c = 360 →  -- sum of angles around a point is 360°
  c = 120 →          -- one angle is 120°
  a = b →            -- the other two angles are equal
  a = 120 :=         -- prove that each of the equal angles is 120°
by sorry

end NUMINAMATH_CALUDE_angles_around_point_l993_99304
