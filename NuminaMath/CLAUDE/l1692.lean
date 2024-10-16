import Mathlib

namespace NUMINAMATH_CALUDE_find_x_l1692_169240

-- Define the binary operation ★
def star (a b c d : ℤ) : ℤ × ℤ := (a + c, b - 2*d)

-- Theorem statement
theorem find_x : ∀ x y : ℤ, star (x + 1) (y - 1) 1 3 = (2, -4) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1692_169240


namespace NUMINAMATH_CALUDE_f_equals_n_plus_one_f_1993_l1692_169215

def N₀ : Set ℕ := {n : ℕ | True}

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ n, f (f n) + f n = 2 * n + 3

theorem f_equals_n_plus_one (f : ℕ → ℕ) (h : is_valid_f f) :
  ∀ n, f n = n + 1 :=
by
  sorry

-- The original question can be answered as a corollary
theorem f_1993 (f : ℕ → ℕ) (h : is_valid_f f) :
  f 1993 = 1994 :=
by
  sorry

end NUMINAMATH_CALUDE_f_equals_n_plus_one_f_1993_l1692_169215


namespace NUMINAMATH_CALUDE_max_value_condition_l1692_169275

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + (2*a - 1) * x

theorem max_value_condition (a : ℝ) :
  (∀ x > 0, f a x ≤ f a 1) → a > 1/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_condition_l1692_169275


namespace NUMINAMATH_CALUDE_book_ratio_problem_l1692_169287

theorem book_ratio_problem (darla_books katie_books gary_books : ℕ) : 
  darla_books = 6 →
  gary_books = 5 * (darla_books + katie_books) →
  darla_books + katie_books + gary_books = 54 →
  katie_books = darla_books / 2 := by
  sorry

end NUMINAMATH_CALUDE_book_ratio_problem_l1692_169287


namespace NUMINAMATH_CALUDE_product_of_numbers_with_sum_and_difference_l1692_169272

theorem product_of_numbers_with_sum_and_difference 
  (x y : ℝ) (h_sum : x + y = 50) (h_diff : x - y = 6) : x * y = 616 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_sum_and_difference_l1692_169272


namespace NUMINAMATH_CALUDE_correct_calculation_l1692_169257

theorem correct_calculation (x : ℚ) (h : 15 / x = 5) : 21 / x = 7 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1692_169257


namespace NUMINAMATH_CALUDE_move_right_four_units_l1692_169203

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point horizontally in a Cartesian coordinate system -/
def moveRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem move_right_four_units :
  let p := Point.mk (-2) 3
  moveRight p 4 = Point.mk 2 3 := by
  sorry

end NUMINAMATH_CALUDE_move_right_four_units_l1692_169203


namespace NUMINAMATH_CALUDE_salon_customers_l1692_169254

/-- Represents the daily operations of a hair salon -/
structure Salon where
  total_cans : ℕ
  extra_cans : ℕ
  cans_per_customer : ℕ

/-- Calculates the number of customers given a salon's daily operations -/
def customers (s : Salon) : ℕ :=
  (s.total_cans - s.extra_cans) / s.cans_per_customer

/-- Theorem stating that a salon with the given parameters has 14 customers per day -/
theorem salon_customers :
  let s : Salon := {
    total_cans := 33,
    extra_cans := 5,
    cans_per_customer := 2
  }
  customers s = 14 := by
  sorry

end NUMINAMATH_CALUDE_salon_customers_l1692_169254


namespace NUMINAMATH_CALUDE_unique_divisibility_by_99_l1692_169201

def number (x y : ℕ) : ℕ := 141 * 10000 + x * 1000 + 28 * 100 + y * 10 + 3

theorem unique_divisibility_by_99 (x y : ℕ) 
  (h_x : x ≤ 9) (h_y : y ≤ 9) (h_div : (number x y) % 99 = 0) : 
  x = 4 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisibility_by_99_l1692_169201


namespace NUMINAMATH_CALUDE_friend_reading_time_l1692_169267

/-- Proves that given the conditions on reading speeds and time, 
    the friend's reading time for one volume is 0.3 hours -/
theorem friend_reading_time 
  (my_speed : ℝ) 
  (friend_speed : ℝ) 
  (my_time_two_volumes : ℝ) 
  (h1 : my_speed = (1 / 5) * friend_speed) 
  (h2 : my_time_two_volumes = 3) : 
  (my_time_two_volumes / 2) / 5 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_friend_reading_time_l1692_169267


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l1692_169223

theorem framed_painting_ratio : 
  let painting_width : ℝ := 18
  let painting_height : ℝ := 24
  let frame_side_width : ℝ := 3  -- This is derived from solving the equation in the solution
  let frame_top_bottom_width : ℝ := 2 * frame_side_width
  let framed_width : ℝ := painting_width + 2 * frame_side_width
  let framed_height : ℝ := painting_height + 2 * frame_top_bottom_width
  let frame_area : ℝ := framed_width * framed_height - painting_width * painting_height
  frame_area = painting_width * painting_height →
  (min framed_width framed_height) / (max framed_width framed_height) = 2 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l1692_169223


namespace NUMINAMATH_CALUDE_smallest_n_with_seven_l1692_169265

/-- Check if a natural number contains the digit 7 -/
def containsSeven (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + 7 + 10 * m

/-- The smallest natural number n such that both n^2 and (n+1)^2 contain the digit 7 -/
theorem smallest_n_with_seven : ∀ n : ℕ, n < 26 →
  ¬(containsSeven (n^2) ∧ containsSeven ((n+1)^2)) ∧
  (containsSeven (26^2) ∧ containsSeven (27^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_seven_l1692_169265


namespace NUMINAMATH_CALUDE_binomial_7_choose_4_l1692_169252

theorem binomial_7_choose_4 : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_7_choose_4_l1692_169252


namespace NUMINAMATH_CALUDE_probability_square_or_circle_l1692_169269

theorem probability_square_or_circle (total : ℕ) (triangles squares circles : ℕ) : 
  total = triangles + squares + circles →
  triangles = 4 →
  squares = 3 →
  circles = 5 →
  (squares + circles : ℚ) / total = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_square_or_circle_l1692_169269


namespace NUMINAMATH_CALUDE_draw_specific_sequence_l1692_169217

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the number of cards of each rank (Ace, King, Queen, Jack) -/
def rank_count : ℕ := 4

/-- Represents the number of cards in the hearts suit -/
def hearts_count : ℕ := 13

/-- Calculates the probability of drawing the specified sequence of cards -/
def draw_probability (d : Deck) : ℚ :=
  (rank_count : ℚ) / 52 *
  (rank_count : ℚ) / 51 *
  (rank_count : ℚ) / 50 *
  (rank_count : ℚ) / 49 *
  ((hearts_count - rank_count) : ℚ) / 48

/-- The theorem stating the probability of drawing the specified sequence of cards -/
theorem draw_specific_sequence (d : Deck) :
  draw_probability d = 2304 / 31187500 := by
  sorry

end NUMINAMATH_CALUDE_draw_specific_sequence_l1692_169217


namespace NUMINAMATH_CALUDE_vector_equation_l1692_169294

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem vector_equation : 4 • a - 3 • (a + b) = a - 3 • b := by sorry

end NUMINAMATH_CALUDE_vector_equation_l1692_169294


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l1692_169247

def team_size : ℕ := 18
def lineup_size : ℕ := 8
def non_pg_players : ℕ := lineup_size - 1

theorem basketball_lineup_combinations :
  (team_size : ℕ) * (Nat.choose (team_size - 1) non_pg_players) = 349864 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l1692_169247


namespace NUMINAMATH_CALUDE_ball_probability_l1692_169248

/-- The probability of choosing a ball that is neither red nor purple from a bag -/
theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 10)
  (h_green : green = 30)
  (h_yellow : yellow = 10)
  (h_red : red = 47)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ball_probability_l1692_169248


namespace NUMINAMATH_CALUDE_homework_question_count_l1692_169241

/-- Calculates the number of true/false questions in a homework assignment -/
theorem homework_question_count (total : ℕ) (mc_ratio : ℕ) (fr_diff : ℕ) (h1 : total = 45) (h2 : mc_ratio = 2) (h3 : fr_diff = 7) : 
  ∃ (tf : ℕ) (fr : ℕ) (mc : ℕ), 
    tf + fr + mc = total ∧ 
    mc = mc_ratio * fr ∧ 
    fr = tf + fr_diff ∧ 
    tf = 6 := by
  sorry

end NUMINAMATH_CALUDE_homework_question_count_l1692_169241


namespace NUMINAMATH_CALUDE_time_to_see_again_l1692_169214

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a person walking -/
structure Walker where
  initialPosition : Point
  speed : ℝ

/-- The setup of the problem -/
def problemSetup : Prop := ∃ (sam kim : Walker) (tower : Point) (r : ℝ),
  -- Sam's initial position and speed
  sam.initialPosition = ⟨-100, -150⟩ ∧ sam.speed = 4 ∧
  -- Kim's initial position and speed
  kim.initialPosition = ⟨-100, 150⟩ ∧ kim.speed = 2 ∧
  -- Tower's position and radius
  tower = ⟨0, 0⟩ ∧ r = 100 ∧
  -- Initial distance between Sam and Kim
  (sam.initialPosition.x - kim.initialPosition.x)^2 + (sam.initialPosition.y - kim.initialPosition.y)^2 = 300^2

/-- The theorem to be proved -/
theorem time_to_see_again (setup : problemSetup) : 
  ∃ (t : ℝ), t = 240 ∧ 
  (∀ (t' : ℝ), t' < t → ∃ (x y : ℝ), 
    x^2 + y^2 = 100^2 ∧ 
    (y - (-150)) / (x - (-100 + 4 * t')) = (150 - (-150)) / (((-100 + 2 * t') - (-100 + 4 * t'))) ∧
    x * (150 - (-150)) = y * (((-100 + 2 * t') - (-100 + 4 * t'))))
  ∧ 
  (∃ (x y : ℝ), 
    x^2 + y^2 = 100^2 ∧ 
    (y - (-150)) / (x - (-100 + 4 * t)) = (150 - (-150)) / (((-100 + 2 * t) - (-100 + 4 * t))) ∧
    x * (150 - (-150)) = y * (((-100 + 2 * t) - (-100 + 4 * t)))) :=
by
  sorry


end NUMINAMATH_CALUDE_time_to_see_again_l1692_169214


namespace NUMINAMATH_CALUDE_work_completion_theorem_l1692_169299

/-- Represents the work completion scenario -/
structure WorkCompletion where
  initial_men : ℕ
  initial_hours_per_day : ℕ
  initial_days : ℕ
  new_men : ℕ
  new_days : ℕ

/-- Calculates the hours per day for the new workforce -/
def hours_per_day (w : WorkCompletion) : ℚ :=
  (w.initial_men * w.initial_hours_per_day * w.initial_days : ℚ) / (w.new_men * w.new_days)

theorem work_completion_theorem (w : WorkCompletion) 
    (h1 : w.initial_men = 10)
    (h2 : w.initial_hours_per_day = 7)
    (h3 : w.initial_days = 18)
    (h4 : w.new_days = 12)
    (h5 : w.new_men > 10) :
    hours_per_day w = 1260 / (12 * w.new_men) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l1692_169299


namespace NUMINAMATH_CALUDE_sin_140_cos_50_plus_sin_130_cos_40_eq_1_l1692_169226

theorem sin_140_cos_50_plus_sin_130_cos_40_eq_1 :
  Real.sin (140 * π / 180) * Real.cos (50 * π / 180) +
  Real.sin (130 * π / 180) * Real.cos (40 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_140_cos_50_plus_sin_130_cos_40_eq_1_l1692_169226


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1692_169284

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n + q

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 4 + a 8 = π →
  a 6 * (a 2 + 2 * a 6 + a 10) = π^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1692_169284


namespace NUMINAMATH_CALUDE_jane_final_crayons_l1692_169277

/-- The number of crayons Jane ends up with after the hippopotamus incident and finding additional crayons -/
def final_crayon_count (x y : ℕ) : ℕ :=
  y - x + 15

/-- Theorem stating that given the conditions, Jane ends up with 95 crayons -/
theorem jane_final_crayons :
  let x : ℕ := 7  -- number of crayons eaten by the hippopotamus
  let y : ℕ := 87 -- number of crayons Jane had initially
  final_crayon_count x y = 95 := by
  sorry

end NUMINAMATH_CALUDE_jane_final_crayons_l1692_169277


namespace NUMINAMATH_CALUDE_wheel_turns_l1692_169278

/-- A wheel makes 6 turns every 30 seconds. This theorem proves that it makes 1440 turns in 2 hours. -/
theorem wheel_turns (turns_per_30_sec : ℕ) (hours : ℕ) : 
  turns_per_30_sec = 6 → hours = 2 → turns_per_30_sec * 240 * hours = 1440 := by
  sorry

end NUMINAMATH_CALUDE_wheel_turns_l1692_169278


namespace NUMINAMATH_CALUDE_total_markers_l1692_169221

theorem total_markers :
  let red_markers : ℕ := 41
  let blue_markers : ℕ := 64
  let green_markers : ℕ := 35
  let black_markers : ℕ := 78
  let yellow_markers : ℕ := 102
  red_markers + blue_markers + green_markers + black_markers + yellow_markers = 320 :=
by
  sorry

end NUMINAMATH_CALUDE_total_markers_l1692_169221


namespace NUMINAMATH_CALUDE_area_ratio_correct_l1692_169279

/-- Represents a rectangle inscribed in a circle with a smaller rectangle inside it. -/
structure InscribedRectangles where
  /-- The ratio of the smaller rectangle's width to the larger rectangle's width -/
  x : ℝ
  /-- The ratio of the smaller rectangle's height to the larger rectangle's height -/
  y : ℝ
  /-- Constraint ensuring the smaller rectangle's vertices lie on the circle -/
  h_circle : 4 * y^2 + 4 * y + x^2 = 1

/-- The area ratio of the smaller rectangle to the larger rectangle -/
def areaRatio (r : InscribedRectangles) : ℝ := r.x * r.y

theorem area_ratio_correct (r : InscribedRectangles) : 
  areaRatio r = r.x * r.y := by sorry

end NUMINAMATH_CALUDE_area_ratio_correct_l1692_169279


namespace NUMINAMATH_CALUDE_supplement_of_angle_with_complement_50_l1692_169238

def angle_with_complement_50 (θ : ℝ) : Prop :=
  90 - θ = 50

theorem supplement_of_angle_with_complement_50 (θ : ℝ) 
  (h : angle_with_complement_50 θ) : 180 - θ = 140 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_angle_with_complement_50_l1692_169238


namespace NUMINAMATH_CALUDE_inequality_proof_l1692_169295

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y)) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1692_169295


namespace NUMINAMATH_CALUDE_additional_volunteers_needed_l1692_169288

def volunteers_needed : ℕ := 50
def math_classes : ℕ := 6
def students_per_class : ℕ := 5
def teachers_volunteered : ℕ := 13

theorem additional_volunteers_needed :
  volunteers_needed - (math_classes * students_per_class + teachers_volunteered) = 7 := by
  sorry

end NUMINAMATH_CALUDE_additional_volunteers_needed_l1692_169288


namespace NUMINAMATH_CALUDE_functional_equation_implies_linear_scaling_l1692_169200

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x^3 + y^3) = (x + y) * (f x^2 - f x * f y + f y^2)

/-- The main theorem to be proved -/
theorem functional_equation_implies_linear_scaling
  (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x, f (1996 * x) = 1996 * f x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_linear_scaling_l1692_169200


namespace NUMINAMATH_CALUDE_third_circle_radius_l1692_169274

theorem third_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 34) (h₂ : r₂ = 14) : 
  π * r₃^2 = π * (r₁^2 - r₂^2) → r₃ = 8 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_third_circle_radius_l1692_169274


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1692_169225

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 2 = 1 →
  a 3 + a 4 = 9 →
  a 5 + a 6 = 81 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1692_169225


namespace NUMINAMATH_CALUDE_common_ratio_of_sequence_l1692_169228

def geometric_sequence (a : ℤ → ℤ) (r : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem common_ratio_of_sequence (a : ℤ → ℤ) :
  a 0 = 25 ∧ a 1 = -50 ∧ a 2 = 100 ∧ a 3 = -200 →
  ∃ r : ℤ, geometric_sequence a r ∧ r = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_common_ratio_of_sequence_l1692_169228


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1692_169286

theorem cubic_equation_roots (a b : ℝ) :
  (∃ x y z : ℕ+, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    (∀ t : ℝ, t^3 - 8*t^2 + a*t - b = 0 ↔ (t = x ∨ t = y ∨ t = z))) →
  a + b = 31 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1692_169286


namespace NUMINAMATH_CALUDE_ratio_of_roots_quadratic_l1692_169280

theorem ratio_of_roots_quadratic (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ - 16 = 0 ∧ 
    x₂^2 + p*x₂ - 16 = 0 ∧ 
    x₁/x₂ = -4) → 
  p = 6 ∨ p = -6 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_roots_quadratic_l1692_169280


namespace NUMINAMATH_CALUDE_average_speed_first_half_l1692_169291

theorem average_speed_first_half (total_distance : ℝ) (total_avg_speed : ℝ) : 
  total_distance = 640 →
  total_avg_speed = 40 →
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let first_half_time := first_half_distance / (first_half_distance / (total_distance / (4 * total_avg_speed)))
  let second_half_time := 3 * first_half_time
  first_half_distance / first_half_time = 80 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_first_half_l1692_169291


namespace NUMINAMATH_CALUDE_danny_chemistry_marks_l1692_169212

theorem danny_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (h1 : english = 76) 
  (h2 : mathematics = 65) 
  (h3 : physics = 82) 
  (h4 : biology = 75) 
  (h5 : average = 73) 
  (h6 : (english + mathematics + physics + biology + chemistry) / 5 = average) :
  chemistry = 67 :=
by
  sorry

end NUMINAMATH_CALUDE_danny_chemistry_marks_l1692_169212


namespace NUMINAMATH_CALUDE_root_comparison_l1692_169260

theorem root_comparison (m n : ℕ) : 
  min ((n : ℝ) ^ (1 / m : ℝ)) ((m : ℝ) ^ (1 / n : ℝ)) ≤ (3 : ℝ) ^ (1 / 3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_root_comparison_l1692_169260


namespace NUMINAMATH_CALUDE_problem_statement_l1692_169227

theorem problem_statement (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^2 + 4*b^2 = 1/(a*b) + 3) : 
  (a * b ≤ 1) ∧ 
  (b > a → 1/a^3 - 1/b^3 ≥ 3*(1/a - 1/b)) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1692_169227


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1692_169220

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x < 1 → x^2 - 4*x + 3 > 0) ∧ 
  (∃ x : ℝ, x^2 - 4*x + 3 > 0 ∧ x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1692_169220


namespace NUMINAMATH_CALUDE_farm_ploughing_problem_l1692_169211

/-- Calculates the remaining area to be ploughed given the total area, planned and actual ploughing rates, and additional days worked. -/
def remaining_area (total_area planned_rate actual_rate extra_days : ℕ) : ℕ :=
  let planned_days := total_area / planned_rate
  let actual_days := planned_days + extra_days
  let ploughed_area := actual_rate * actual_days
  total_area - ploughed_area

/-- Theorem stating that given the specific conditions of the farm problem, the remaining area to be ploughed is 40 hectares. -/
theorem farm_ploughing_problem :
  remaining_area 720 120 85 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_farm_ploughing_problem_l1692_169211


namespace NUMINAMATH_CALUDE_third_shot_probability_at_least_one_hit_probability_l1692_169209

/-- A marksman shoots four times independently with a probability of hitting the target of 0.9 each time. -/
structure Marksman where
  shots : Fin 4 → ℝ
  prob_hit : ∀ i, shots i = 0.9
  independent : ∀ i j, i ≠ j → shots i = shots j

/-- The probability of hitting the target on the third shot is 0.9. -/
theorem third_shot_probability (m : Marksman) : m.shots 2 = 0.9 := by sorry

/-- The probability of hitting the target at least once is 1 - 0.1^4. -/
theorem at_least_one_hit_probability (m : Marksman) : 
  1 - (1 - m.shots 0) * (1 - m.shots 1) * (1 - m.shots 2) * (1 - m.shots 3) = 1 - 0.1^4 := by sorry

end NUMINAMATH_CALUDE_third_shot_probability_at_least_one_hit_probability_l1692_169209


namespace NUMINAMATH_CALUDE_exactly_three_false_l1692_169242

-- Define the type for statements
inductive Statement
| one : Statement
| two : Statement
| three : Statement
| four : Statement

-- Define a function to evaluate the truth value of a statement
def evaluate : Statement → (Statement → Bool) → Bool
| Statement.one, f => (f Statement.one && ¬f Statement.two && ¬f Statement.three && ¬f Statement.four) ||
                      (¬f Statement.one && f Statement.two && ¬f Statement.three && ¬f Statement.four) ||
                      (¬f Statement.one && ¬f Statement.two && f Statement.three && ¬f Statement.four) ||
                      (¬f Statement.one && ¬f Statement.two && ¬f Statement.three && f Statement.four)
| Statement.two, f => (¬f Statement.one && f Statement.two && f Statement.three && ¬f Statement.four) ||
                      (¬f Statement.one && f Statement.two && ¬f Statement.three && f Statement.four) ||
                      (f Statement.one && ¬f Statement.two && f Statement.three && ¬f Statement.four) ||
                      (f Statement.one && ¬f Statement.two && ¬f Statement.three && f Statement.four) ||
                      (¬f Statement.one && f Statement.two && f Statement.three && ¬f Statement.four) ||
                      (f Statement.one && f Statement.two && ¬f Statement.three && ¬f Statement.four)
| Statement.three, f => (¬f Statement.one && ¬f Statement.two && f Statement.three && ¬f Statement.four)
| Statement.four, f => (f Statement.one && f Statement.two && f Statement.three && f Statement.four)

-- Theorem statement
theorem exactly_three_false :
  ∃ (f : Statement → Bool),
    (∀ s, evaluate s f = f s) ∧
    (f Statement.one = false ∧
     f Statement.two = false ∧
     f Statement.three = true ∧
     f Statement.four = false) :=
by sorry

end NUMINAMATH_CALUDE_exactly_three_false_l1692_169242


namespace NUMINAMATH_CALUDE_triangle_properties_triangle_max_area_l1692_169281

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The sides form an arithmetic sequence -/
def isArithmeticSequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

/-- Vectors (3, sin B) and (2, sin C) are collinear -/
def areVectorsCollinear (t : Triangle) : Prop :=
  3 * Real.sin t.C = 2 * Real.sin t.B

/-- The product of sides a and c is 8 -/
def hasSideProduct8 (t : Triangle) : Prop :=
  t.a * t.c = 8

theorem triangle_properties (t : Triangle) 
  (h1 : isArithmeticSequence t) 
  (h2 : areVectorsCollinear t) :
  Real.cos t.A = -1/4 := by sorry

theorem triangle_max_area (t : Triangle) 
  (h1 : isArithmeticSequence t)
  (h2 : hasSideProduct8 t) :
  ∃ (S : ℝ), S = 2 * Real.sqrt 3 ∧ 
  ∀ (area : ℝ), area ≤ S := by sorry

end NUMINAMATH_CALUDE_triangle_properties_triangle_max_area_l1692_169281


namespace NUMINAMATH_CALUDE_salt_solution_percentage_l1692_169236

theorem salt_solution_percentage (S : ℝ) : 
  S ≥ 0 ∧ S ≤ 100 →  -- Ensure S is a valid percentage
  (3/4 * S + 1/4 * 28 = 16) →  -- Equation representing the mixing of solutions
  S = 12 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_percentage_l1692_169236


namespace NUMINAMATH_CALUDE_rals_age_l1692_169206

theorem rals_age (suri_age suri_age_in_3_years ral_age : ℕ) :
  suri_age_in_3_years = suri_age + 3 →
  suri_age_in_3_years = 16 →
  ral_age = 2 * suri_age →
  ral_age = 26 := by
  sorry

end NUMINAMATH_CALUDE_rals_age_l1692_169206


namespace NUMINAMATH_CALUDE_philip_banana_count_l1692_169239

/-- The number of banana groups in Philip's collection -/
def banana_groups : ℕ := 7

/-- The number of bananas in each group -/
def bananas_per_group : ℕ := 29

/-- The total number of bananas in Philip's collection -/
def total_bananas : ℕ := banana_groups * bananas_per_group

/-- Theorem stating that the total number of bananas is 203 -/
theorem philip_banana_count : total_bananas = 203 := by
  sorry

end NUMINAMATH_CALUDE_philip_banana_count_l1692_169239


namespace NUMINAMATH_CALUDE_fraction_equals_seven_l1692_169261

theorem fraction_equals_seven (x : ℝ) (h : x = 2) : (x^4 + 6*x^2 + 9) / (x^2 + 3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_seven_l1692_169261


namespace NUMINAMATH_CALUDE_largest_number_l1692_169290

theorem largest_number (a b c d : ℝ) (h1 : a = Real.sqrt 5) (h2 : b = -1.6) (h3 : c = 0) (h4 : d = 2) :
  max a (max b (max c d)) = a :=
sorry

end NUMINAMATH_CALUDE_largest_number_l1692_169290


namespace NUMINAMATH_CALUDE_function_extremum_l1692_169222

/-- The function f(x) with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem function_extremum (a b : ℝ) :
  f a b 1 = 10 ∧ f_deriv a b 1 = 0 →
  ∀ x, f a b x = x^3 + 4*x^2 - 11*x + 16 :=
by sorry

end NUMINAMATH_CALUDE_function_extremum_l1692_169222


namespace NUMINAMATH_CALUDE_sin_315_degrees_l1692_169270

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l1692_169270


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l1692_169285

def circle_radius : ℝ := 2
def inner_circle_radius : ℝ := 1
def num_points : ℕ := 6
def num_symmetrical_parts : ℕ := 3

theorem shaded_area_theorem :
  let sector_angle : ℝ := 2 * Real.pi / num_points
  let sector_area : ℝ := (1 / 2) * circle_radius^2 * sector_angle
  let triangle_area : ℝ := (1 / 2) * circle_radius * inner_circle_radius * Real.sin (sector_angle / 2)
  let quadrilateral_area : ℝ := 2 * triangle_area
  let part_area : ℝ := sector_area + quadrilateral_area
  num_symmetrical_parts * part_area = 2 * Real.pi + 3 := by sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l1692_169285


namespace NUMINAMATH_CALUDE_cone_in_cylinder_volume_ratio_l1692_169204

noncomputable def cone_volume (base_area : ℝ) (height : ℝ) : ℝ := 
  (1 / 3) * base_area * height

noncomputable def cylinder_volume (base_area : ℝ) (height : ℝ) : ℝ := 
  base_area * height

theorem cone_in_cylinder_volume_ratio 
  (base_area : ℝ) (height : ℝ) (h_pos : base_area > 0 ∧ height > 0) :
  let v_cone := cone_volume base_area height
  let v_cylinder := cylinder_volume base_area height
  (v_cylinder - v_cone) / v_cone = 2 := by
sorry

end NUMINAMATH_CALUDE_cone_in_cylinder_volume_ratio_l1692_169204


namespace NUMINAMATH_CALUDE_expression_evaluation_l1692_169276

theorem expression_evaluation :
  let a : ℚ := -1/6
  2 * (a + 1) * (a - 1) - a * (2 * a - 3) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1692_169276


namespace NUMINAMATH_CALUDE_crow_speed_l1692_169271

/-- Calculates the speed of a crow flying between its nest and a ditch -/
theorem crow_speed (distance : ℝ) (trips : ℕ) (time : ℝ) : 
  distance = 200 → 
  trips = 15 → 
  time = 1.5 → 
  (2 * distance * trips) / (time * 1000) = 4 :=
by sorry

end NUMINAMATH_CALUDE_crow_speed_l1692_169271


namespace NUMINAMATH_CALUDE_inequality_proof_l1692_169219

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≤ a*c) :
  (a*f - c*d)^2 ≥ (a*e - b*d)*(b*f - c*e) := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1692_169219


namespace NUMINAMATH_CALUDE_fraction_irreducible_l1692_169258

theorem fraction_irreducible (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  ¬∃ (f g : ℝ → ℝ → ℝ → ℝ), (∀ x y z, f x y z ≠ 0 ∧ g x y z ≠ 0) ∧ 
    (∀ x y z, (x^2 + y^2 - z^2 + x*y) / (x^2 + z^2 - y^2 + y*z) = f x y z / g x y z) ∧
    (f a b c / g a b c ≠ (a^2 + b^2 - c^2 + a*b) / (a^2 + c^2 - b^2 + b*c)) :=
sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l1692_169258


namespace NUMINAMATH_CALUDE_circle_motion_problem_l1692_169230

/-- Given a circle with two points A and B moving along its circumference, 
    this theorem proves the speeds of the points and the circumference of the circle. -/
theorem circle_motion_problem 
  (smaller_arc : ℝ) 
  (smaller_arc_time : ℝ) 
  (larger_arc_time : ℝ) 
  (b_distance : ℝ) 
  (h1 : smaller_arc = 150)
  (h2 : smaller_arc_time = 10)
  (h3 : larger_arc_time = 14)
  (h4 : b_distance = 90) :
  ∃ (va vb l : ℝ),
    va = 12 ∧ 
    vb = 3 ∧ 
    l = 360 ∧
    smaller_arc_time * (va + vb) = smaller_arc ∧
    larger_arc_time * (va + vb) = l - smaller_arc ∧
    l / va = b_distance / vb :=
by sorry

end NUMINAMATH_CALUDE_circle_motion_problem_l1692_169230


namespace NUMINAMATH_CALUDE_candy_distribution_l1692_169264

theorem candy_distribution (initial_candy : ℕ) (eaten : ℕ) (bowls : ℕ) (taken : ℕ) : 
  initial_candy = 100 →
  eaten = 8 →
  bowls = 4 →
  taken = 3 →
  (initial_candy - eaten) / bowls - taken = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1692_169264


namespace NUMINAMATH_CALUDE_theresas_work_hours_l1692_169283

theorem theresas_work_hours (total_weeks : ℕ) (target_average : ℕ) 
  (week1 week2 week3 week4 : ℕ) (additional_task : ℕ) :
  total_weeks = 5 →
  target_average = 12 →
  week1 = 10 →
  week2 = 14 →
  week3 = 11 →
  week4 = 9 →
  additional_task = 1 →
  ∃ (week5 : ℕ), 
    (week1 + week2 + week3 + week4 + week5 + additional_task) / total_weeks = target_average ∧
    week5 = 15 :=
by sorry

end NUMINAMATH_CALUDE_theresas_work_hours_l1692_169283


namespace NUMINAMATH_CALUDE_xy_z_squared_plus_one_representation_l1692_169202

theorem xy_z_squared_plus_one_representation (x y z : ℕ+) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), (x : ℤ) = a^2 + b^2 ∧ (y : ℤ) = c^2 + d^2 ∧ (z : ℤ) = a * c + b * d := by
  sorry

end NUMINAMATH_CALUDE_xy_z_squared_plus_one_representation_l1692_169202


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l1692_169237

theorem arithmetic_mean_of_first_four_primes_reciprocals :
  let first_four_primes := [2, 3, 5, 7]
  let reciprocals := first_four_primes.map (λ x => 1 / x)
  (reciprocals.sum / reciprocals.length : ℚ) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l1692_169237


namespace NUMINAMATH_CALUDE_distribution_theorem_l1692_169249

-- Define the total number of employees
def total_employees : ℕ := 8

-- Define the number of departments
def num_departments : ℕ := 2

-- Define the number of English translators
def num_translators : ℕ := 2

-- Define the function to calculate the number of distribution schemes
def distribution_schemes (n : ℕ) (k : ℕ) (t : ℕ) : ℕ := 
  (Nat.choose (n - t) ((n - t) / 2)) * 2

-- Theorem statement
theorem distribution_theorem : 
  distribution_schemes total_employees num_departments num_translators = 40 := by
  sorry

end NUMINAMATH_CALUDE_distribution_theorem_l1692_169249


namespace NUMINAMATH_CALUDE_factorize_quadratic_factorize_cubic_factorize_quartic_l1692_169224

-- Problem 1
theorem factorize_quadratic (m : ℝ) : m^2 + 4*m + 4 = (m + 2)^2 := by sorry

-- Problem 2
theorem factorize_cubic (a b : ℝ) : a^2*b - 4*a*b^2 + 3*b^3 = b*(a-b)*(a-3*b) := by sorry

-- Problem 3
theorem factorize_quartic (x y : ℝ) : (x^2 + y^2)^2 - 4*x^2*y^2 = (x + y)^2 * (x - y)^2 := by sorry

end NUMINAMATH_CALUDE_factorize_quadratic_factorize_cubic_factorize_quartic_l1692_169224


namespace NUMINAMATH_CALUDE_quadratic_root_property_l1692_169256

theorem quadratic_root_property (a : ℝ) (h : a^2 - 2*a - 3 = 0) : a^2 - 2*a + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l1692_169256


namespace NUMINAMATH_CALUDE_collinear_points_d_values_l1692_169250

-- Define the points
def point_a (a : ℝ) : ℝ × ℝ × ℝ := (1, 0, a)
def point_b (b : ℝ) : ℝ × ℝ × ℝ := (b, 1, 0)
def point_c (c : ℝ) : ℝ × ℝ × ℝ := (0, c, 1)
def point_d (d : ℝ) : ℝ × ℝ × ℝ := (4*d, 4*d, -2*d)

-- Define collinearity
def collinear (p q r : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), q - p = t • (r - p)

theorem collinear_points_d_values (a b c d : ℝ) :
  collinear (point_a a) (point_b b) (point_c c) ∧
  collinear (point_a a) (point_b b) (point_d d) →
  d = 1 ∨ d = 1/4 :=
sorry

end NUMINAMATH_CALUDE_collinear_points_d_values_l1692_169250


namespace NUMINAMATH_CALUDE_min_longest_palindrome_length_l1692_169259

/-- A string consisting only of characters 'A' and 'B' -/
def ABString : Type := List Char

/-- Check if a string is a palindrome -/
def isPalindrome (s : ABString) : Prop :=
  s = s.reverse

/-- The length of the longest palindromic substring in an ABString -/
def longestPalindromeLength (s : ABString) : ℕ :=
  sorry

theorem min_longest_palindrome_length :
  (∀ s : ABString, s.length = 2021 → longestPalindromeLength s ≥ 4) ∧
  (∃ s : ABString, s.length = 2021 ∧ longestPalindromeLength s = 4) :=
sorry

end NUMINAMATH_CALUDE_min_longest_palindrome_length_l1692_169259


namespace NUMINAMATH_CALUDE_union_necessary_not_sufficient_for_intersection_l1692_169266

theorem union_necessary_not_sufficient_for_intersection (A B : Set α) :
  (∀ x, x ∈ A ∩ B → x ∈ A ∪ B) ∧
  (∃ x, x ∈ A ∪ B ∧ x ∉ A ∩ B) :=
sorry

end NUMINAMATH_CALUDE_union_necessary_not_sufficient_for_intersection_l1692_169266


namespace NUMINAMATH_CALUDE_child_height_at_last_visit_l1692_169253

/-- Given a child's current height and growth since last visit, 
    prove the height at the last visit. -/
theorem child_height_at_last_visit 
  (current_height : ℝ) 
  (growth_since_last_visit : ℝ) 
  (h1 : current_height = 41.5) 
  (h2 : growth_since_last_visit = 3.0) : 
  current_height - growth_since_last_visit = 38.5 := by
sorry

end NUMINAMATH_CALUDE_child_height_at_last_visit_l1692_169253


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1692_169213

theorem polynomial_expansion (x : ℝ) : 
  (5 * x + 3) * (7 * x^2 + 2 * x + 4) = 35 * x^3 + 31 * x^2 + 26 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1692_169213


namespace NUMINAMATH_CALUDE_c_profit_is_3600_l1692_169293

def initial_home_value : ℝ := 20000
def profit_percentage : ℝ := 0.20
def loss_percentage : ℝ := 0.15

def sale_price : ℝ := initial_home_value * (1 + profit_percentage)
def repurchase_price : ℝ := sale_price * (1 - loss_percentage)

theorem c_profit_is_3600 : sale_price - repurchase_price = 3600 := by sorry

end NUMINAMATH_CALUDE_c_profit_is_3600_l1692_169293


namespace NUMINAMATH_CALUDE_max_value_of_largest_integer_l1692_169210

theorem max_value_of_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℝ) / 5 = 45 →
  (max a (max b (max c (max d e)))) - (min a (min b (min c (min d e)))) = 10 →
  max a (max b (max c (max d e))) ≤ 215 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_largest_integer_l1692_169210


namespace NUMINAMATH_CALUDE_farmer_weekly_milk_production_l1692_169273

/-- Given a number of cows and daily milk production per cow, calculates the total milk production for a week. -/
def weekly_milk_production (num_cows : ℕ) (daily_milk_per_cow : ℕ) : ℕ :=
  num_cows * daily_milk_per_cow * 7

/-- Proves that 52 cows producing 5 liters each per day will produce 1820 liters in a week. -/
theorem farmer_weekly_milk_production :
  weekly_milk_production 52 5 = 1820 := by
  sorry

end NUMINAMATH_CALUDE_farmer_weekly_milk_production_l1692_169273


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1692_169282

theorem inequality_solution_set (x : ℝ) : 
  (3 * x^2) / (1 - (3*x + 1)^(1/3))^2 ≤ x + 2 + (3*x + 1)^(1/3) → 
  -2/3 ≤ x ∧ x < 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1692_169282


namespace NUMINAMATH_CALUDE_unique_solution_system_l1692_169245

/-- The system of equations:
    3^y * 81 = 9^(x^2)
    lg y = lg x - lg 0.5
    has only one positive real solution (x, y) = (2, 4) -/
theorem unique_solution_system (x y : ℝ) 
  (h1 : (3 : ℝ)^y * 81 = 9^(x^2))
  (h2 : Real.log y / Real.log 10 = Real.log x / Real.log 10 - Real.log 0.5 / Real.log 10)
  (h3 : x > 0)
  (h4 : y > 0) : 
  x = 2 ∧ y = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1692_169245


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1692_169298

theorem simplify_trig_expression :
  Real.sin (50 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 
    (1 / 2) * Real.cos (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1692_169298


namespace NUMINAMATH_CALUDE_wendys_cookies_l1692_169216

theorem wendys_cookies (pastries_left pastries_sold num_cupcakes : ℕ) 
  (h1 : pastries_left = 24)
  (h2 : pastries_sold = 9)
  (h3 : num_cupcakes = 4) : 
  (pastries_left + pastries_sold) - num_cupcakes = 29 := by
  sorry

#check wendys_cookies

end NUMINAMATH_CALUDE_wendys_cookies_l1692_169216


namespace NUMINAMATH_CALUDE_t_value_on_line_l1692_169255

/-- A straight line passing through points (1, 7), (3, 13), (5, 19), and (28, t) -/
def straightLine (t : ℝ) : Prop :=
  ∃ (m c : ℝ),
    (7 = m * 1 + c) ∧
    (13 = m * 3 + c) ∧
    (19 = m * 5 + c) ∧
    (t = m * 28 + c)

/-- Theorem stating that t = 88 for the given straight line -/
theorem t_value_on_line : straightLine 88 := by
  sorry

end NUMINAMATH_CALUDE_t_value_on_line_l1692_169255


namespace NUMINAMATH_CALUDE_children_distribution_l1692_169296

theorem children_distribution (n : ℕ) : 
  (6 : ℝ) / n - (6 : ℝ) / (n + 2) = (1 : ℝ) / 4 → n + 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_children_distribution_l1692_169296


namespace NUMINAMATH_CALUDE_average_chocolate_pieces_per_cookie_l1692_169289

theorem average_chocolate_pieces_per_cookie 
  (num_cookies : ℕ) 
  (num_choc_chips : ℕ) 
  (num_mms : ℕ) 
  (h1 : num_cookies = 48) 
  (h2 : num_choc_chips = 108) 
  (h3 : num_mms = num_choc_chips / 3) : 
  (num_choc_chips + num_mms) / num_cookies = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_chocolate_pieces_per_cookie_l1692_169289


namespace NUMINAMATH_CALUDE_factors_of_M_l1692_169233

/-- The number of natural-number factors of M, where M = 2^4 · 3^3 · 5^2 · 7^1 -/
def number_of_factors (M : ℕ) : ℕ :=
  (4 + 1) * (3 + 1) * (2 + 1) * (1 + 1)

/-- Theorem stating that the number of natural-number factors of M is 120 -/
theorem factors_of_M :
  let M : ℕ := 2^4 * 3^3 * 5^2 * 7^1
  number_of_factors M = 120 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_M_l1692_169233


namespace NUMINAMATH_CALUDE_purchase_price_l1692_169235

/-- The total price of a purchase of shirts and a tie -/
def total_price (shirt_price : ℝ) (tie_price : ℝ) (discount : ℝ) : ℝ :=
  2 * shirt_price + tie_price + shirt_price * (1 - discount)

/-- The proposition that the total price is 3500 rubles -/
theorem purchase_price :
  ∃ (shirt_price tie_price : ℝ),
    2 * shirt_price + tie_price = 2600 ∧
    total_price shirt_price tie_price 0.25 = 3500 ∧
    shirt_price = 1200 := by
  sorry

end NUMINAMATH_CALUDE_purchase_price_l1692_169235


namespace NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l1692_169297

theorem sqrt_eight_equals_two_sqrt_two : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_equals_two_sqrt_two_l1692_169297


namespace NUMINAMATH_CALUDE_martha_total_time_l1692_169244

def router_time : ℕ := 10

def on_hold_time : ℕ := 6 * router_time

def yelling_time : ℕ := on_hold_time / 2

def total_time : ℕ := router_time + on_hold_time + yelling_time

theorem martha_total_time : total_time = 100 := by
  sorry

end NUMINAMATH_CALUDE_martha_total_time_l1692_169244


namespace NUMINAMATH_CALUDE_second_number_less_than_twice_first_l1692_169232

theorem second_number_less_than_twice_first (x y : ℤ) : 
  x + y = 57 → y = 37 → 2 * x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_number_less_than_twice_first_l1692_169232


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1692_169207

/-- For an arithmetic sequence {a_n} with S_n as the sum of its first n terms, 
    if S_9 = 27, then a_4 + a_6 = 6 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_S9 : S 9 = 27) : 
  a 4 + a 6 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1692_169207


namespace NUMINAMATH_CALUDE_roots_product_equality_l1692_169234

theorem roots_product_equality (p q : ℝ) (α β γ δ : ℝ) 
  (h1 : α^2 + p*α - 2 = 0)
  (h2 : β^2 + p*β - 2 = 0)
  (h3 : γ^2 + q*γ - 2 = 0)
  (h4 : δ^2 + q*δ - 2 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -(p^2 - q^2) := by
  sorry

end NUMINAMATH_CALUDE_roots_product_equality_l1692_169234


namespace NUMINAMATH_CALUDE_expression_factorization_l1692_169262

theorem expression_factorization (a b c : ℝ) :
  a^3 * (b^2 - c^2) + b^3 * (c^2 - b^2) + c^3 * (a^2 - b^2) =
  (a - b) * (b - c) * (c - a) * (a*b + a*c + b*c) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l1692_169262


namespace NUMINAMATH_CALUDE_simplify_expression_l1692_169251

theorem simplify_expression : 
  ((9 * 10^8) * 2^2) / (3 * 2^3 * 10^3) = 150000 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l1692_169251


namespace NUMINAMATH_CALUDE_oil_price_reduction_l1692_169205

/-- Represents the price reduction problem for oil -/
theorem oil_price_reduction (original_price : ℝ) (original_quantity : ℝ) : 
  (original_price * original_quantity = 684) →
  (0.8 * original_price * (original_quantity + 4) = 684) →
  (0.8 * original_price = 34.20) :=
by sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l1692_169205


namespace NUMINAMATH_CALUDE_garden_perimeter_l1692_169231

/-- The perimeter of a rectangular garden with width 24 meters and the same area as a rectangular playground of length 16 meters and width 12 meters is 64 meters. -/
theorem garden_perimeter : 
  ∀ (garden_length : ℝ),
  garden_length > 0 →
  24 * garden_length = 16 * 12 →
  2 * garden_length + 2 * 24 = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1692_169231


namespace NUMINAMATH_CALUDE_marathon_water_bottles_l1692_169263

theorem marathon_water_bottles (runners : ℕ) (bottles_per_runner : ℕ) (available_bottles : ℕ) : 
  runners = 14 → bottles_per_runner = 5 → available_bottles = 68 → 
  (runners * bottles_per_runner - available_bottles) = 2 := by
sorry

end NUMINAMATH_CALUDE_marathon_water_bottles_l1692_169263


namespace NUMINAMATH_CALUDE_power_difference_value_l1692_169243

theorem power_difference_value (x m n : ℝ) (hm : x^m = 6) (hn : x^n = 3) : x^(m-n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_value_l1692_169243


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1692_169246

/-- The quadratic equation (m-1)x^2 - 2x + 1 = 0 has real roots if and only if m ≤ 2 and m ≠ 1 -/
theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 2 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1692_169246


namespace NUMINAMATH_CALUDE_infinitely_many_m_minus_f_eq_1989_l1692_169218

/-- The number of factors of 2 in m! -/
def f (m : ℕ) : ℕ := sorry

/-- Condition that 11 · 15m is a positive integer -/
def is_valid (m : ℕ) : Prop := 0 < 11 * 15 * m

/-- The main theorem -/
theorem infinitely_many_m_minus_f_eq_1989 :
  ∀ n : ℕ, ∃ m > n, is_valid m ∧ m - f m = 1989 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_m_minus_f_eq_1989_l1692_169218


namespace NUMINAMATH_CALUDE_window_dimensions_l1692_169208

/-- Represents the dimensions of a rectangular glass pane -/
structure PaneDimensions where
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a rectangular window -/
structure WindowDimensions where
  width : ℝ
  height : ℝ

/-- Calculates the dimensions of a window given the pane dimensions and border widths -/
def calculateWindowDimensions (pane : PaneDimensions) (topBorderWidth sideBorderWidth : ℝ) : WindowDimensions :=
  { width := 3 * pane.width + 4 * sideBorderWidth,
    height := 2 * pane.height + 2 * topBorderWidth + sideBorderWidth }

theorem window_dimensions (y : ℝ) :
  let pane : PaneDimensions := { width := 4 * y, height := 3 * y }
  let window := calculateWindowDimensions pane 3 1
  window.width = 12 * y + 4 ∧ window.height = 6 * y + 7 := by
  sorry

#check window_dimensions

end NUMINAMATH_CALUDE_window_dimensions_l1692_169208


namespace NUMINAMATH_CALUDE_f_prime_zero_l1692_169229

theorem f_prime_zero (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x, f x = x^2 + 2 * (deriv f 2) * x + 3) : 
  deriv f 0 = -8 := by sorry

end NUMINAMATH_CALUDE_f_prime_zero_l1692_169229


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1692_169292

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1692_169292


namespace NUMINAMATH_CALUDE_chrysler_has_23_floors_l1692_169268

/-- The number of floors in the Leeward Center -/
def leeward_floors : ℕ := sorry

/-- The number of floors in the Chrysler Building -/
def chrysler_floors : ℕ := sorry

/-- The Chrysler Building has 11 more floors than the Leeward Center -/
axiom chrysler_leeward_difference : chrysler_floors = leeward_floors + 11

/-- The total number of floors in both buildings is 35 -/
axiom total_floors : leeward_floors + chrysler_floors = 35

/-- Theorem: The Chrysler Building has 23 floors -/
theorem chrysler_has_23_floors : chrysler_floors = 23 := by sorry

end NUMINAMATH_CALUDE_chrysler_has_23_floors_l1692_169268
