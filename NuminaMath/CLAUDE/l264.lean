import Mathlib

namespace NUMINAMATH_CALUDE_andrews_age_l264_26473

theorem andrews_age (grandfather_age andrew_age : ℝ) 
  (h1 : grandfather_age = 9 * andrew_age)
  (h2 : grandfather_age - andrew_age = 63) : 
  andrew_age = 7.875 := by
sorry

end NUMINAMATH_CALUDE_andrews_age_l264_26473


namespace NUMINAMATH_CALUDE_kim_cousins_count_l264_26411

theorem kim_cousins_count (total_gum : ℕ) (gum_per_cousin : ℕ) (cousin_count : ℕ) : 
  total_gum = 20 → gum_per_cousin = 5 → total_gum = gum_per_cousin * cousin_count → cousin_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_kim_cousins_count_l264_26411


namespace NUMINAMATH_CALUDE_tan_difference_pi_12_5pi_12_l264_26425

theorem tan_difference_pi_12_5pi_12 : 
  Real.tan (π / 12) - Real.tan (5 * π / 12) = -2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_pi_12_5pi_12_l264_26425


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l264_26471

/-- A quadratic function f(x) = ax^2 + bx + 6 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 6

theorem quadratic_function_properties (a b : ℝ) :
  f a b 1 = 8 ∧ f a b (-1) = f a b 3 →
  (a + b = 2 ∧ f a b 2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l264_26471


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l264_26413

theorem quadratic_inequality_solution (n : ℕ) (x : ℝ) :
  (∀ n : ℕ, n^2 * x^2 - (2*n^2 + n) * x + n^2 + n - 6 ≤ 0) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l264_26413


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_l264_26467

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

/-- The fixed point through which the line passes -/
def fixedPoint : ℝ × ℝ := (0, 1)

/-- The equation of the trajectory of the midpoint of the chord -/
def trajectoryEquation (x y : ℝ) : Prop := 4*x^2 - y^2 + y = 0

/-- Theorem stating that the trajectory equation is correct for the given conditions -/
theorem trajectory_of_midpoint (x y : ℝ) :
  (∃ (k : ℝ), ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧
    y₁ = k*x₁ + fixedPoint.2 ∧ y₂ = k*x₂ + fixedPoint.2 ∧
    x = (x₁ + x₂)/2 ∧ y = (y₁ + y₂)/2) →
  trajectoryEquation x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_l264_26467


namespace NUMINAMATH_CALUDE_deck_width_is_four_feet_l264_26496

/-- Given a rectangular pool and a surrounding deck, this theorem proves
    that the deck width is 4 feet under specific conditions. -/
theorem deck_width_is_four_feet 
  (pool_length : ℝ) 
  (pool_width : ℝ) 
  (total_area : ℝ) 
  (h1 : pool_length = 10)
  (h2 : pool_width = 12)
  (h3 : total_area = 360)
  (w : ℝ) -- deck width
  (h4 : (pool_length + 2 * w) * (pool_width + 2 * w) = total_area) :
  w = 4 := by
  sorry

#check deck_width_is_four_feet

end NUMINAMATH_CALUDE_deck_width_is_four_feet_l264_26496


namespace NUMINAMATH_CALUDE_triangle_circumcircle_l264_26410

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 3)

-- Define the line BC
def line_BC (x y : ℝ) : Prop := y - 1 = 0

-- Define the median from A to BC
def median_A (x y : ℝ) : Prop := x - 3*y + 4 = 0

-- Define the circumcircle equation
def circumcircle (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Theorem statement
theorem triangle_circumcircle : 
  ∀ (B C : ℝ × ℝ),
  line_BC B.1 B.2 ∧ line_BC C.1 C.2 ∧
  median_A ((B.1 + C.1)/2) ((B.2 + C.2)/2) →
  circumcircle B.1 B.2 ∧ circumcircle C.1 C.2 ∧ circumcircle A.1 A.2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_l264_26410


namespace NUMINAMATH_CALUDE_inequality_proof_l264_26417

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y^2016 ≥ 1) :
  x^2016 + y > 1 - 1/100 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l264_26417


namespace NUMINAMATH_CALUDE_either_or_implies_at_least_one_l264_26423

theorem either_or_implies_at_least_one (p q : Prop) : 
  (p ∨ q) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_either_or_implies_at_least_one_l264_26423


namespace NUMINAMATH_CALUDE_inequality_proof_l264_26432

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b) + (b / Real.sqrt a) ≥ Real.sqrt a + Real.sqrt b ∧
  (a + b = 1 → (1/a) + (1/b) + (1/(a*b)) ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l264_26432


namespace NUMINAMATH_CALUDE_percentage_problem_l264_26449

theorem percentage_problem (P : ℝ) : 
  0.15 * 0.30 * (P / 100) * 4000 = 90 → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l264_26449


namespace NUMINAMATH_CALUDE_basketball_team_selection_l264_26416

theorem basketball_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 6 →
  (Nat.choose quadruplets 3) * (Nat.choose (total_players - quadruplets) (starters - 3)) = 880 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l264_26416


namespace NUMINAMATH_CALUDE_judys_score_is_25_l264_26488

/-- Represents the scoring system for a math competition -/
structure ScoringSystem where
  correctPoints : Int
  incorrectPoints : Int

/-- Represents a participant's answers in the competition -/
structure Answers where
  total : Nat
  correct : Nat
  incorrect : Nat
  unanswered : Nat

/-- Calculates the score based on the scoring system and answers -/
def calculateScore (system : ScoringSystem) (answers : Answers) : Int :=
  system.correctPoints * answers.correct + system.incorrectPoints * answers.incorrect

/-- Theorem: Judy's score in the math competition is 25 points -/
theorem judys_score_is_25 (system : ScoringSystem) (answers : Answers) :
  system.correctPoints = 2 →
  system.incorrectPoints = -1 →
  answers.total = 30 →
  answers.correct = 15 →
  answers.incorrect = 5 →
  answers.unanswered = 10 →
  calculateScore system answers = 25 := by
  sorry

#eval calculateScore { correctPoints := 2, incorrectPoints := -1 }
                     { total := 30, correct := 15, incorrect := 5, unanswered := 10 }

end NUMINAMATH_CALUDE_judys_score_is_25_l264_26488


namespace NUMINAMATH_CALUDE_john_chess_probability_l264_26408

theorem john_chess_probability (p_win : ℚ) (h : p_win = 2 / 5) : 1 - p_win = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_john_chess_probability_l264_26408


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_five_thirds_l264_26490

theorem fraction_of_powers_equals_five_thirds :
  (2^2014 + 2^2012) / (2^2014 - 2^2012) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_five_thirds_l264_26490


namespace NUMINAMATH_CALUDE_tangent_line_curve_range_l264_26419

theorem tangent_line_curve_range (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x y : ℝ, y = x - a ∧ y = Real.log (x + b) ∧ 
    (∀ x' y' : ℝ, y' = x' - a → y' ≤ Real.log (x' + b))) →
  (∀ z : ℝ, z ∈ Set.Ioo 0 (1/2) ↔ ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 
    (∃ x y : ℝ, y = x - a' ∧ y = Real.log (x + b') ∧ 
      (∀ x' y' : ℝ, y' = x' - a' → y' ≤ Real.log (x' + b'))) ∧
    z = a'^2 / (2 + b')) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_curve_range_l264_26419


namespace NUMINAMATH_CALUDE_nigella_base_salary_l264_26457

def house_sale_income (base_salary : ℝ) (commission_rate : ℝ) (house_prices : List ℝ) : ℝ :=
  base_salary + (commission_rate * (house_prices.sum))

theorem nigella_base_salary :
  let commission_rate : ℝ := 0.02
  let house_a_price : ℝ := 60000
  let house_b_price : ℝ := 3 * house_a_price
  let house_c_price : ℝ := 2 * house_a_price - 110000
  let house_prices : List ℝ := [house_a_price, house_b_price, house_c_price]
  let total_income : ℝ := 8000
  ∃ (base_salary : ℝ), 
    house_sale_income base_salary commission_rate house_prices = total_income ∧
    base_salary = 3000 :=
by sorry

end NUMINAMATH_CALUDE_nigella_base_salary_l264_26457


namespace NUMINAMATH_CALUDE_f_has_zero_in_interval_l264_26483

-- Define the function f(x) = x³ + 3x - 3
def f (x : ℝ) : ℝ := x^3 + 3*x - 3

-- Theorem statement
theorem f_has_zero_in_interval :
  ∃ c ∈ Set.Icc 0 1, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_zero_in_interval_l264_26483


namespace NUMINAMATH_CALUDE_sparrow_percentage_among_non_eagles_l264_26430

theorem sparrow_percentage_among_non_eagles (total percentage : ℝ)
  (robins eagles falcons sparrows : ℝ)
  (h1 : total = 100)
  (h2 : robins = 20)
  (h3 : eagles = 30)
  (h4 : falcons = 15)
  (h5 : sparrows = total - (robins + eagles + falcons))
  (h6 : percentage = (sparrows / (total - eagles)) * 100) :
  percentage = 50 := by
sorry

end NUMINAMATH_CALUDE_sparrow_percentage_among_non_eagles_l264_26430


namespace NUMINAMATH_CALUDE_mitchell_gum_chewing_l264_26431

theorem mitchell_gum_chewing (packets : ℕ) (pieces_per_packet : ℕ) (not_chewed : ℕ) :
  packets = 8 →
  pieces_per_packet = 7 →
  not_chewed = 2 →
  packets * pieces_per_packet - not_chewed = 54 := by
  sorry

end NUMINAMATH_CALUDE_mitchell_gum_chewing_l264_26431


namespace NUMINAMATH_CALUDE_quadratic_always_positive_iff_a_in_range_l264_26478

theorem quadratic_always_positive_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + 1 > 0) ↔ -2 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_iff_a_in_range_l264_26478


namespace NUMINAMATH_CALUDE_parallelogram_area_24_16_l264_26470

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_24_16 :
  parallelogramArea 24 16 = 384 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_24_16_l264_26470


namespace NUMINAMATH_CALUDE_count_valid_voucher_codes_l264_26429

/-- Represents a voucher code -/
structure VoucherCode where
  first : Char
  second : Nat
  third : Nat
  fourth : Nat

/-- Checks if a character is a valid first character -/
def isValidFirstChar (c : Char) : Bool :=
  c = 'V' || c = 'X' || c = 'P'

/-- Checks if a voucher code is valid -/
def isValidVoucherCode (code : VoucherCode) : Bool :=
  isValidFirstChar code.first &&
  code.second < 10 &&
  code.third < 10 &&
  code.second ≠ code.third &&
  code.fourth = (code.second + code.third) % 10

/-- The set of all valid voucher codes -/
def validVoucherCodes : Finset VoucherCode :=
  sorry

/-- The number of valid voucher codes is 270 -/
theorem count_valid_voucher_codes :
  Finset.card validVoucherCodes = 270 :=
sorry

end NUMINAMATH_CALUDE_count_valid_voucher_codes_l264_26429


namespace NUMINAMATH_CALUDE_janet_fertilizer_time_l264_26472

-- Define the constants from the problem
def gallons_per_horse_per_day : ℕ := 5
def number_of_horses : ℕ := 80
def total_acres : ℕ := 20
def gallons_per_acre : ℕ := 400
def acres_spread_per_day : ℕ := 4

-- Define the theorem
theorem janet_fertilizer_time : 
  (total_acres * gallons_per_acre) / (number_of_horses * gallons_per_horse_per_day) +
  total_acres / acres_spread_per_day = 25 := by
  sorry

end NUMINAMATH_CALUDE_janet_fertilizer_time_l264_26472


namespace NUMINAMATH_CALUDE_base7_perfect_square_last_digit_l264_26442

def is_base7_perfect_square (x y z : ℕ) : Prop :=
  x ≠ 0 ∧ z < 7 ∧ ∃ k : ℕ, k^2 = x * 7^3 + y * 7^2 + 5 * 7 + z

theorem base7_perfect_square_last_digit 
  (x y z : ℕ) (h : is_base7_perfect_square x y z) : z = 1 ∨ z = 6 := by
  sorry

end NUMINAMATH_CALUDE_base7_perfect_square_last_digit_l264_26442


namespace NUMINAMATH_CALUDE_pages_per_book_l264_26499

/-- Given that Frank took 12 days to finish each book and 492 days to finish all 41 books,
    prove that each book had 492 pages. -/
theorem pages_per_book (days_per_book : ℕ) (total_days : ℕ) (total_books : ℕ) :
  days_per_book = 12 →
  total_days = 492 →
  total_books = 41 →
  (total_days / days_per_book) * days_per_book = 492 := by
  sorry

#check pages_per_book

end NUMINAMATH_CALUDE_pages_per_book_l264_26499


namespace NUMINAMATH_CALUDE_A_intersect_B_is_empty_l264_26426

-- Define set A
def A : Set ℝ := {x : ℝ | |x| ≥ 2}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - x - 2 < 0}

-- Theorem statement
theorem A_intersect_B_is_empty : A ∩ B = ∅ := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_is_empty_l264_26426


namespace NUMINAMATH_CALUDE_quadratic_inequality_l264_26486

theorem quadratic_inequality (x : ℝ) : (x - 2) * (x + 2) > 0 ↔ x > 2 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l264_26486


namespace NUMINAMATH_CALUDE_crayon_selection_theorem_l264_26421

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of red crayons in the box -/
def red_crayons : ℕ := 3

/-- The number of crayons to be selected -/
def selected_crayons : ℕ := 5

/-- The number of red crayons that must be selected -/
def selected_red : ℕ := 2

/-- The number of ways to select crayons under the given conditions -/
def ways_to_select : ℕ := choose red_crayons selected_red * choose (total_crayons - red_crayons) (selected_crayons - selected_red)

theorem crayon_selection_theorem : ways_to_select = 660 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_theorem_l264_26421


namespace NUMINAMATH_CALUDE_circle_area_increase_l264_26479

theorem circle_area_increase (r : ℝ) : 
  let initial_area := π * r^2
  let final_area := π * (r + 3)^2
  final_area - initial_area = 6 * π * r + 9 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_increase_l264_26479


namespace NUMINAMATH_CALUDE_greater_than_negative_five_by_negative_six_l264_26453

theorem greater_than_negative_five_by_negative_six :
  ((-5) + (-6) : ℤ) = -11 :=
by sorry

end NUMINAMATH_CALUDE_greater_than_negative_five_by_negative_six_l264_26453


namespace NUMINAMATH_CALUDE_triangle_ABC_coordinates_l264_26428

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (a b c : Point) : ℝ := sorry

/-- Checks if a point is on a coordinate axis -/
def isOnAxis (p : Point) : Prop :=
  p.x = 0 ∨ p.y = 0

theorem triangle_ABC_coordinates :
  let a : Point := ⟨2, 0⟩
  let b : Point := ⟨0, 3⟩
  ∀ c : Point,
    triangleArea a b c = 6 ∧ isOnAxis c →
    c = ⟨0, 9⟩ ∨ c = ⟨0, -3⟩ ∨ c = ⟨-2, 0⟩ ∨ c = ⟨6, 0⟩ :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_coordinates_l264_26428


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l264_26444

theorem rectangle_area_perimeter_sum (a b : ℕ+) : 
  let A := (a : ℝ) * (b : ℝ)
  let P := 2 * (a : ℝ) + 2 * (b : ℝ) + 2
  A + P ≠ 114 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l264_26444


namespace NUMINAMATH_CALUDE_real_roots_of_p_l264_26462

def p (x : ℝ) : ℝ := x^4 - 3*x^3 + 3*x^2 - x - 6

theorem real_roots_of_p :
  ∃ (a b c d : ℝ), (∀ x, p x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) ∧
                   (a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 1) := by
  sorry

end NUMINAMATH_CALUDE_real_roots_of_p_l264_26462


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l264_26466

theorem no_prime_roots_for_quadratic : 
  ¬ ∃ (k : ℤ), ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    (p : ℤ) + q = 59 ∧
    (p : ℤ) * q = k ∧
    ∀ (x : ℤ), x^2 - 59*x + k = 0 ↔ x = p ∨ x = q :=
by sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l264_26466


namespace NUMINAMATH_CALUDE_lattice_points_limit_l264_26491

/-- The number of lattice points inside a circle of radius r centered at the origin -/
noncomputable def f (r : ℝ) : ℝ := sorry

/-- The difference between f(r) and πr^2 -/
noncomputable def g (r : ℝ) : ℝ := f r - Real.pi * r^2

theorem lattice_points_limit :
  (∀ ε > 0, ∃ R, ∀ r ≥ R, |f r / r^2 - Real.pi| < ε) ∧
  (∀ h < 2, ∀ ε > 0, ∃ R, ∀ r ≥ R, |g r / r^h| < ε) := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_limit_l264_26491


namespace NUMINAMATH_CALUDE_intersection_volume_of_reflected_tetrahedron_l264_26440

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  volume : ℝ
  is_regular : Bool

/-- The intersection of two regular tetrahedra -/
def tetrahedra_intersection (t1 t2 : RegularTetrahedron) : ℝ := sorry

/-- Reflection of a regular tetrahedron through its center -/
def reflect_through_center (t : RegularTetrahedron) : RegularTetrahedron := sorry

theorem intersection_volume_of_reflected_tetrahedron (t : RegularTetrahedron) 
  (h1 : t.volume = 1)
  (h2 : t.is_regular = true) :
  tetrahedra_intersection t (reflect_through_center t) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_volume_of_reflected_tetrahedron_l264_26440


namespace NUMINAMATH_CALUDE_trapezoid_height_is_four_l264_26437

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  -- The length of the midline
  midline : ℝ
  -- The lengths of the bases
  base1 : ℝ
  base2 : ℝ
  -- The height of the trapezoid
  height : ℝ
  -- Condition: The trapezoid has an inscribed circle
  has_inscribed_circle : Prop
  -- Condition: The midline is the average of the bases
  midline_avg : midline = (base1 + base2) / 2
  -- Condition: The area ratio of the parts divided by the midline
  area_ratio : (base1 - midline) / (base2 - midline) = 7 / 13

/-- The main theorem about the height of the trapezoid -/
theorem trapezoid_height_is_four (t : IsoscelesTrapezoid) 
  (h_midline : t.midline = 5) : t.height = 4 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_height_is_four_l264_26437


namespace NUMINAMATH_CALUDE_feline_sanctuary_count_l264_26493

theorem feline_sanctuary_count :
  let lions : ℕ := 12
  let tigers : ℕ := 14
  let cougars : ℕ := (lions + tigers) / 3
  lions + tigers + cougars = 34 := by
sorry

end NUMINAMATH_CALUDE_feline_sanctuary_count_l264_26493


namespace NUMINAMATH_CALUDE_sphere_radius_l264_26492

theorem sphere_radius (V : Real) (r : Real) : V = (4 / 3) * Real.pi * r^3 → V = 36 * Real.pi → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_l264_26492


namespace NUMINAMATH_CALUDE_P_root_characteristics_l264_26489

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^7 - 4*x^5 - 8*x^3 - x + 12

-- Theorem statement
theorem P_root_characteristics :
  (∀ x < 0, P x ≠ 0) ∧ (∃ x > 0, P x = 0) := by sorry

end NUMINAMATH_CALUDE_P_root_characteristics_l264_26489


namespace NUMINAMATH_CALUDE_man_downstream_speed_l264_26494

/-- Given a man's upstream speed and the stream speed, calculates his downstream speed. -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  upstream_speed + 2 * stream_speed

/-- Theorem stating that given the specific upstream speed and stream speed, 
    the downstream speed is 15 kmph. -/
theorem man_downstream_speed :
  downstream_speed 8 3.5 = 15 := by
  sorry

#eval downstream_speed 8 3.5

end NUMINAMATH_CALUDE_man_downstream_speed_l264_26494


namespace NUMINAMATH_CALUDE_red_shirt_pairs_l264_26446

theorem red_shirt_pairs (total_students : ℕ) (green_students : ℕ) (red_students : ℕ) 
  (total_pairs : ℕ) (green_green_pairs : ℕ) 
  (h1 : total_students = 144)
  (h2 : green_students = 65)
  (h3 : red_students = 79)
  (h4 : total_pairs = 72)
  (h5 : green_green_pairs = 27)
  (h6 : total_students = green_students + red_students)
  (h7 : total_pairs * 2 = total_students) :
  ∃ red_red_pairs : ℕ, red_red_pairs = 34 ∧ 
    red_red_pairs + green_green_pairs + (green_students - 2 * green_green_pairs) = total_pairs :=
by
  sorry


end NUMINAMATH_CALUDE_red_shirt_pairs_l264_26446


namespace NUMINAMATH_CALUDE_problem_solution_l264_26403

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

theorem problem_solution :
  (f (25 * Real.pi / 6) = 0) ∧
  (∀ α : ℝ, 0 < α ∧ α < Real.pi →
    f (α / 2) = 1 / 4 - Real.sqrt 3 / 2 →
    Real.sin α = (1 + 3 * Real.sqrt 5) / 8) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l264_26403


namespace NUMINAMATH_CALUDE_integer_count_in_sequence_l264_26406

def arithmeticSequence (n : ℕ) : ℚ :=
  8505 / (5 ^ n)

def isInteger (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem integer_count_in_sequence :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → isInteger (arithmeticSequence n)) ∧
    ¬isInteger (arithmeticSequence k)) →
  (∃! (k : ℕ), k = 3 ∧
    (∀ (n : ℕ), n < k → isInteger (arithmeticSequence n)) ∧
    ¬isInteger (arithmeticSequence k)) :=
by sorry

end NUMINAMATH_CALUDE_integer_count_in_sequence_l264_26406


namespace NUMINAMATH_CALUDE_quadratic_minimum_l264_26439

theorem quadratic_minimum (x : ℝ) : ∃ (min : ℝ), min = -29 ∧ ∀ y : ℝ, x^2 + 14*x + 20 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l264_26439


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l264_26497

theorem rationalize_and_simplify :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l264_26497


namespace NUMINAMATH_CALUDE_people_who_left_line_l264_26454

theorem people_who_left_line (initial_people : ℕ) (joined_people : ℕ) (people_who_left : ℕ) : 
  initial_people = 31 → 
  joined_people = 25 → 
  initial_people = (initial_people - people_who_left) + joined_people →
  people_who_left = 25 := by
sorry

end NUMINAMATH_CALUDE_people_who_left_line_l264_26454


namespace NUMINAMATH_CALUDE_positive_real_equivalence_l264_26424

theorem positive_real_equivalence (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_equivalence_l264_26424


namespace NUMINAMATH_CALUDE_probability_total_gt_seven_is_five_twelfths_l264_26450

/-- The number of faces on each die -/
def faces : ℕ := 6

/-- The total number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := faces * faces

/-- The number of outcomes that result in a total greater than 7 -/
def favorable_outcomes : ℕ := 15

/-- The probability of getting a total more than 7 when throwing two 6-sided dice -/
def probability_total_gt_seven : ℚ := favorable_outcomes / total_outcomes

theorem probability_total_gt_seven_is_five_twelfths :
  probability_total_gt_seven = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_total_gt_seven_is_five_twelfths_l264_26450


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l264_26407

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  eq : ℝ → ℝ → Prop := λ x y => x^2 / a^2 + y^2 / b^2 = 1
  vertex : ℝ × ℝ := (0, -1)
  focus_distance : ℝ := 3

/-- The line that intersects the ellipse -/
structure IntersectingLine where
  k : ℝ
  m : ℝ
  h_k : k ≠ 0
  eq : ℝ → ℝ → Prop := λ x y => y = k * x + m

/-- Main theorem about the ellipse and intersecting line -/
theorem ellipse_and_line_properties (e : Ellipse) (l : IntersectingLine) :
  (e.eq = λ x y => x^2 / 3 + y^2 = 1) ∧
  (∀ M N : ℝ × ℝ, e.eq M.1 M.2 → e.eq N.1 N.2 → l.eq M.1 M.2 → l.eq N.1 N.2 → M ≠ N →
    (dist M e.vertex = dist N e.vertex) → (1/2 < l.m ∧ l.m < 2)) := by
  sorry


end NUMINAMATH_CALUDE_ellipse_and_line_properties_l264_26407


namespace NUMINAMATH_CALUDE_asparagus_cost_l264_26420

def initial_amount : ℕ := 55
def banana_pack_cost : ℕ := 4
def banana_packs : ℕ := 2
def pear_cost : ℕ := 2
def chicken_cost : ℕ := 11
def remaining_amount : ℕ := 28

theorem asparagus_cost :
  ∃ (asparagus_cost : ℕ),
    initial_amount - (banana_pack_cost * banana_packs + pear_cost + chicken_cost + asparagus_cost) = remaining_amount ∧
    asparagus_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_asparagus_cost_l264_26420


namespace NUMINAMATH_CALUDE_coefficient_of_x_term_l264_26474

theorem coefficient_of_x_term (x : ℝ) : 
  let expansion := (x - x + 1)^3
  ∃ a b c d : ℝ, expansion = a*x^3 + b*x^2 + c*x + d ∧ c = -3 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_term_l264_26474


namespace NUMINAMATH_CALUDE_two_prime_pairs_sum_to_100_l264_26480

def isPrime (n : ℕ) : Prop := sorry

theorem two_prime_pairs_sum_to_100 : 
  ∃! (count : ℕ), ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (p q : ℕ), (p, q) ∈ S ↔ isPrime p ∧ isPrime q ∧ p + q = 100 ∧ p ≤ q) ∧
    S.card = count ∧
    count = 2 :=
sorry

end NUMINAMATH_CALUDE_two_prime_pairs_sum_to_100_l264_26480


namespace NUMINAMATH_CALUDE_marble_difference_l264_26441

theorem marble_difference (jar1_blue jar1_red jar2_blue jar2_red : ℕ) :
  jar1_blue + jar1_red = jar2_blue + jar2_red →
  7 * jar1_red = 3 * jar1_blue →
  3 * jar2_red = 2 * jar2_blue →
  jar1_red + jar2_red = 80 →
  jar1_blue - jar2_blue = 80 / 7 := by
sorry

end NUMINAMATH_CALUDE_marble_difference_l264_26441


namespace NUMINAMATH_CALUDE_table_tennis_probabilities_l264_26465

def num_players : ℕ := 6
def num_players_A : ℕ := 3
def num_players_B : ℕ := 1
def num_players_C : ℕ := 2

def probability_at_least_one_C : ℚ := 3/5
def probability_same_association : ℚ := 4/15

theorem table_tennis_probabilities :
  (num_players = num_players_A + num_players_B + num_players_C) →
  (probability_at_least_one_C = 3/5) ∧
  (probability_same_association = 4/15) :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_probabilities_l264_26465


namespace NUMINAMATH_CALUDE_translation_of_complex_plane_l264_26468

open Complex

theorem translation_of_complex_plane (t : ℂ → ℂ) :
  (t (-3 + 3*I) = -8 - 2*I) →
  (∃ w : ℂ, ∀ z : ℂ, t z = z + w) →
  (t (-2 + 6*I) = -7 + I) :=
by sorry

end NUMINAMATH_CALUDE_translation_of_complex_plane_l264_26468


namespace NUMINAMATH_CALUDE_claire_gift_card_balance_l264_26464

/-- Calculates the remaining balance on Claire's gift card after a week of purchases. -/
def remaining_balance (gift_card_value : ℚ) (latte_cost : ℚ) (croissant_cost : ℚ) 
  (days : ℕ) (cookie_cost : ℚ) (num_cookies : ℕ) : ℚ :=
  gift_card_value - 
  ((latte_cost + croissant_cost) * days + cookie_cost * num_cookies)

/-- Proves that Claire will have $43.00 left on her gift card after a week of purchases. -/
theorem claire_gift_card_balance : 
  remaining_balance 100 3.75 3.50 7 1.25 5 = 43 := by
  sorry

end NUMINAMATH_CALUDE_claire_gift_card_balance_l264_26464


namespace NUMINAMATH_CALUDE_license_plate_count_l264_26469

def letter_choices : ℕ := 26
def odd_digits : ℕ := 5
def all_digits : ℕ := 10
def even_digits : ℕ := 4

theorem license_plate_count : 
  letter_choices^3 * odd_digits * all_digits * even_digits = 3514400 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l264_26469


namespace NUMINAMATH_CALUDE_simplest_fraction_l264_26477

variable (x : ℝ)

-- Define the fractions
def f1 : ℚ → ℚ := λ x => 4 / (2 * x)
def f2 : ℚ → ℚ := λ x => (x - 1) / (x^2 - 1)
def f3 : ℚ → ℚ := λ x => 1 / (x + 1)
def f4 : ℚ → ℚ := λ x => (1 - x) / (x - 1)

-- Define what it means for a fraction to be simplest
def is_simplest (f : ℚ → ℚ) : Prop :=
  ∀ g : ℚ → ℚ, (∀ x, f x = g x) → f = g

-- Theorem statement
theorem simplest_fraction :
  is_simplest f3 ∧ ¬is_simplest f1 ∧ ¬is_simplest f2 ∧ ¬is_simplest f4 := by
  sorry

end NUMINAMATH_CALUDE_simplest_fraction_l264_26477


namespace NUMINAMATH_CALUDE_inequalities_hold_l264_26436

theorem inequalities_hold (a b : ℝ) (h : a ≠ b) : 
  (a^2 - 4*a + 5 > 0) ∧ (a^2 + b^2 ≥ 2*(a - b - 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l264_26436


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l264_26460

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 1 < 0) ↔ -4 < a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l264_26460


namespace NUMINAMATH_CALUDE_polynomial_factorization_l264_26435

theorem polynomial_factorization (x : ℝ) : 
  x^2 + 4*x + 4 - 81*x^4 = (-9*x^2 + x + 2) * (9*x^2 + x + 2) := by
  sorry

#check polynomial_factorization

end NUMINAMATH_CALUDE_polynomial_factorization_l264_26435


namespace NUMINAMATH_CALUDE_certain_number_proof_l264_26405

theorem certain_number_proof (x : ℝ) (n : ℝ) (h1 : x^2 - 3*x = n) (h2 : x - 4 = 2) : n = 18 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l264_26405


namespace NUMINAMATH_CALUDE_inequality_proof_l264_26451

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*a*c)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l264_26451


namespace NUMINAMATH_CALUDE_system_solution_l264_26412

theorem system_solution (x y z u : ℝ) 
  (eq1 : x + y = 4)
  (eq2 : x * z + y * u = 7)
  (eq3 : x * z^2 + y * u^2 = 12)
  (eq4 : x * z^3 + y * u^3 = 21) :
  z = 7/3 ∧ y = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l264_26412


namespace NUMINAMATH_CALUDE_smallest_multiple_forty_satisfies_forty_is_smallest_l264_26495

theorem smallest_multiple (y : ℕ) : y > 0 ∧ 800 ∣ (540 * y) → y ≥ 40 :=
sorry

theorem forty_satisfies : 800 ∣ (540 * 40) :=
sorry

theorem forty_is_smallest : ∀ y : ℕ, y > 0 ∧ 800 ∣ (540 * y) → y ≥ 40 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_forty_satisfies_forty_is_smallest_l264_26495


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_ellipse_chord_slope_at_4_2_l264_26481

/-- The slope of a chord in an ellipse given its midpoint -/
theorem ellipse_chord_slope (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 / 36 + y₁^2 / 9 = 1) →  -- Point (x₁, y₁) is on the ellipse
  (x₂^2 / 36 + y₂^2 / 9 = 1) →  -- Point (x₂, y₂) is on the ellipse
  ((x₁ + x₂) / 2 = 4) →         -- Midpoint x-coordinate is 4
  ((y₁ + y₂) / 2 = 2) →         -- Midpoint y-coordinate is 2
  (y₂ - y₁) / (x₂ - x₁) = -1/2  -- Slope of the chord
:= by sorry

/-- The main theorem stating the slope of the chord with midpoint (4, 2) -/
theorem ellipse_chord_slope_at_4_2 : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / 36 + y₁^2 / 9 = 1) ∧ 
    (x₂^2 / 36 + y₂^2 / 9 = 1) ∧ 
    ((x₁ + x₂) / 2 = 4) ∧ 
    ((y₁ + y₂) / 2 = 2) ∧ 
    (y₂ - y₁) / (x₂ - x₁) = -1/2
:= by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_ellipse_chord_slope_at_4_2_l264_26481


namespace NUMINAMATH_CALUDE_distance_A_to_C_distance_A_to_C_is_300_l264_26487

/-- The distance between city A and city C given the travel times and speeds of Eddy and Freddy -/
theorem distance_A_to_C (eddy_time : ℝ) (freddy_time : ℝ) (distance_A_to_B : ℝ) (speed_ratio : ℝ) : ℝ :=
  let eddy_speed := distance_A_to_B / eddy_time
  let freddy_speed := eddy_speed / speed_ratio
  freddy_speed * freddy_time

/-- The actual distance between city A and city C is 300 km -/
theorem distance_A_to_C_is_300 : distance_A_to_C 3 4 450 2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_to_C_distance_A_to_C_is_300_l264_26487


namespace NUMINAMATH_CALUDE_tile_count_l264_26476

def room_length : ℕ := 18
def room_width : ℕ := 24
def border_width : ℕ := 2
def small_tile_size : ℕ := 1
def large_tile_size : ℕ := 3

def border_tiles : ℕ := 
  2 * (room_length + room_width - 2 * border_width) * border_width

def interior_length : ℕ := room_length - 2 * border_width
def interior_width : ℕ := room_width - 2 * border_width

def interior_tiles : ℕ := 
  (interior_length * interior_width) / (large_tile_size * large_tile_size)

def total_tiles : ℕ := border_tiles + interior_tiles

theorem tile_count : total_tiles = 167 := by
  sorry

end NUMINAMATH_CALUDE_tile_count_l264_26476


namespace NUMINAMATH_CALUDE_least_tablets_for_given_box_l264_26459

/-- The least number of tablets to extract from a box containing two types of medicine
    to ensure at least two tablets of each kind are among the extracted. -/
def least_tablets_to_extract (tablets_a tablets_b : ℕ) : ℕ :=
  max ((tablets_a - 1) + 2) ((tablets_b - 1) + 2)

/-- Theorem: Given a box with 10 tablets of medicine A and 13 tablets of medicine B,
    the least number of tablets that should be taken to ensure at least two tablets
    of each kind are among the extracted is 12. -/
theorem least_tablets_for_given_box :
  least_tablets_to_extract 10 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_least_tablets_for_given_box_l264_26459


namespace NUMINAMATH_CALUDE_smallest_n_for_2007n_mod_1000_l264_26401

theorem smallest_n_for_2007n_mod_1000 : 
  ∀ n : ℕ+, n < 691 → (2007 * n.val) % 1000 ≠ 837 ∧ (2007 * 691) % 1000 = 837 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_2007n_mod_1000_l264_26401


namespace NUMINAMATH_CALUDE_constant_function_invariant_l264_26400

/-- Given a function f that is constant 3 for all real inputs, 
    prove that f(x + 5) = 3 for any real x -/
theorem constant_function_invariant (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = 3) :
  ∀ x : ℝ, f (x + 5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_invariant_l264_26400


namespace NUMINAMATH_CALUDE_xyz_value_l264_26455

theorem xyz_value (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 100 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l264_26455


namespace NUMINAMATH_CALUDE_comparison_abc_l264_26452

theorem comparison_abc (a b c : ℝ) 
  (ha : a = Real.rpow 0.7 0.6)
  (hb : b = Real.rpow 0.6 (-0.6))
  (hc : c = Real.rpow 0.6 0.7) :
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_comparison_abc_l264_26452


namespace NUMINAMATH_CALUDE_composition_difference_l264_26485

/-- Given two functions f and g, prove that their composition difference
    equals a specific polynomial. -/
theorem composition_difference (x : ℝ) : 
  let f (x : ℝ) := 3 * x^2 + 4 * x - 5
  let g (x : ℝ) := 2 * x + 1
  (f (g x) - g (f x)) = 6 * x^2 + 12 * x + 11 := by
  sorry

end NUMINAMATH_CALUDE_composition_difference_l264_26485


namespace NUMINAMATH_CALUDE_triangle_third_side_l264_26445

theorem triangle_third_side (a b c : ℕ) : 
  a = 3 → b = 8 → c % 2 = 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) → 
  c ≠ 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_l264_26445


namespace NUMINAMATH_CALUDE_butter_mixture_profit_percentage_l264_26456

/-- Calculates the profit percentage for a butter mixture sale --/
theorem butter_mixture_profit_percentage 
  (butter1_weight : ℝ) 
  (butter1_price : ℝ) 
  (butter2_weight : ℝ) 
  (butter2_price : ℝ) 
  (selling_price : ℝ) :
  butter1_weight = 54 →
  butter1_price = 150 →
  butter2_weight = 36 →
  butter2_price = 125 →
  selling_price = 196 →
  let total_cost := butter1_weight * butter1_price + butter2_weight * butter2_price
  let total_weight := butter1_weight + butter2_weight
  let selling_amount := selling_price * total_weight
  let profit := selling_amount - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage = 40 := by
sorry

end NUMINAMATH_CALUDE_butter_mixture_profit_percentage_l264_26456


namespace NUMINAMATH_CALUDE_inequality_equivalence_l264_26434

theorem inequality_equivalence (x : ℝ) : 
  (x - 3) / 3 < (2 * x + 1) / 2 - 1 ↔ 2 * (x - 3) < 3 * (2 * x + 1) - 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l264_26434


namespace NUMINAMATH_CALUDE_mistaken_subtraction_l264_26409

theorem mistaken_subtraction (x : ℤ) : x - 59 = 43 → x - 46 = 56 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_subtraction_l264_26409


namespace NUMINAMATH_CALUDE_greatest_possible_average_speed_l264_26414

/-- Represents a palindromic number --/
def IsPalindrome (n : ℕ) : Prop := sorry

/-- Calculates the next palindrome after a given number --/
def NextPalindrome (n : ℕ) : ℕ := sorry

/-- Represents the maximum speed limit in miles per hour --/
def MaxSpeedLimit : ℕ := 80

/-- Represents the trip duration in hours --/
def TripDuration : ℕ := 4

/-- Represents the initial odometer reading --/
def InitialReading : ℕ := 12321

theorem greatest_possible_average_speed :
  ∃ (finalReading : ℕ),
    IsPalindrome InitialReading ∧
    IsPalindrome finalReading ∧
    finalReading > InitialReading ∧
    finalReading ≤ InitialReading + MaxSpeedLimit * TripDuration ∧
    (∀ (n : ℕ),
      IsPalindrome n ∧
      n > InitialReading ∧
      n ≤ InitialReading + MaxSpeedLimit * TripDuration →
      n ≤ finalReading) ∧
    (finalReading - InitialReading) / TripDuration = 75 :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_average_speed_l264_26414


namespace NUMINAMATH_CALUDE_smallest_d_is_four_l264_26458

def is_valid_pair (c d : ℕ+) : Prop :=
  (c : ℤ) - (d : ℤ) = 8 ∧ 
  Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16

theorem smallest_d_is_four :
  ∀ c d : ℕ+, is_valid_pair c d → d ≥ 4 ∧ ∃ c' : ℕ+, is_valid_pair c' 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_d_is_four_l264_26458


namespace NUMINAMATH_CALUDE_digit_product_sum_l264_26415

/-- A function that converts a pair of digits to a two-digit integer -/
def twoDigitInt (tens ones : Nat) : Nat :=
  10 * tens + ones

/-- A predicate that checks if a number is a positive digit (1-9) -/
def isPositiveDigit (n : Nat) : Prop :=
  0 < n ∧ n ≤ 9

theorem digit_product_sum (p q r : Nat) : 
  isPositiveDigit p ∧ isPositiveDigit q ∧ isPositiveDigit r →
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  (twoDigitInt p q) * (twoDigitInt p r) = 221 →
  p + q + r = 11 →
  q = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_product_sum_l264_26415


namespace NUMINAMATH_CALUDE_clock_strike_times_l264_26484

/-- Represents the time taken for a given number of clock strikes -/
def strike_time (n : ℕ) : ℚ :=
  (n - 1) * (10 : ℚ) / 9

/-- The clock takes 10 seconds to strike 10 times at 10:00 o'clock -/
axiom ten_strikes_time : strike_time 10 = 10

/-- The strikes are uniformly spaced -/
axiom uniform_strikes : ∀ (n m : ℕ), n > 0 → m > 0 → 
  strike_time n / (n - 1) = strike_time m / (m - 1)

theorem clock_strike_times :
  strike_time 8 = 70 / 9 ∧ strike_time 15 = 140 / 9 := by
  sorry

end NUMINAMATH_CALUDE_clock_strike_times_l264_26484


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l264_26447

theorem inequality_holds_iff (a : ℝ) : 
  (∀ x : ℝ, (4:ℝ)^(x^2) + 2*(2*a+1) * (2:ℝ)^(x^2) + 4*a^2 - 3 > 0) ↔ 
  (a < -1 ∨ a ≥ Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l264_26447


namespace NUMINAMATH_CALUDE_square_approximation_l264_26443

theorem square_approximation (x : ℝ) (h : x ≥ 1/2) :
  ∃ n : ℤ, |x - (n : ℝ)^2| ≤ Real.sqrt (x - 1/4) := by
  sorry

end NUMINAMATH_CALUDE_square_approximation_l264_26443


namespace NUMINAMATH_CALUDE_larger_rectangle_area_larger_rectangle_area_proof_l264_26418

theorem larger_rectangle_area : ℝ → ℝ → ℝ → Prop :=
  fun (small_square_area : ℝ) (small_rect_length : ℝ) (small_rect_width : ℝ) =>
    small_square_area = 25 ∧
    small_rect_length = 3 * Real.sqrt small_square_area ∧
    small_rect_width = Real.sqrt small_square_area ∧
    2 * small_rect_width = small_rect_length →
    small_rect_length * (2 * small_rect_width) = 150

-- The proof goes here
theorem larger_rectangle_area_proof :
  ∃ (small_square_area small_rect_length small_rect_width : ℝ),
    larger_rectangle_area small_square_area small_rect_length small_rect_width :=
by
  sorry

end NUMINAMATH_CALUDE_larger_rectangle_area_larger_rectangle_area_proof_l264_26418


namespace NUMINAMATH_CALUDE_gcd_18_30_l264_26475

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l264_26475


namespace NUMINAMATH_CALUDE_linear_regression_passes_through_mean_point_l264_26448

/-- Linear regression equation passes through the mean point -/
theorem linear_regression_passes_through_mean_point 
  (b a x_bar y_bar : ℝ) : 
  y_bar = b * x_bar + a :=
sorry

end NUMINAMATH_CALUDE_linear_regression_passes_through_mean_point_l264_26448


namespace NUMINAMATH_CALUDE_terminal_side_in_fourth_quadrant_l264_26482

def angle_in_fourth_quadrant (α : Real) : Prop :=
  -2 * Real.pi < α ∧ α < -3 * Real.pi / 2

theorem terminal_side_in_fourth_quadrant :
  angle_in_fourth_quadrant (-5) :=
sorry

end NUMINAMATH_CALUDE_terminal_side_in_fourth_quadrant_l264_26482


namespace NUMINAMATH_CALUDE_rectangle_breadth_l264_26427

theorem rectangle_breadth (area : ℝ) (length_ratio : ℝ) (breadth : ℝ) : 
  area = 460 →
  length_ratio = 1.15 →
  area = (length_ratio * breadth) * breadth →
  breadth = 20 := by
sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l264_26427


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_values_l264_26433

def A (a : ℝ) : Set ℝ := {x | (x - 6) * (x - a) = 0}

def B : Set ℝ := {x | (x - 2) * (x - 3) = 0}

theorem intersection_empty_iff_a_values (a : ℝ) :
  A a ∩ B = ∅ ↔ a = 1 ∨ a = 4 ∨ a = 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_values_l264_26433


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l264_26498

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2*x + a ≤ 0}

-- State the theorem
theorem intersection_implies_a_value :
  ∀ a : ℝ, (A ∩ B a = {x | -2 ≤ x ∧ x ≤ 1}) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l264_26498


namespace NUMINAMATH_CALUDE_log_negative_undefined_l264_26402

-- Define the logarithm function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_negative_undefined (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 3) :
  ¬∃ y, f a (-2) = y := by
  sorry


end NUMINAMATH_CALUDE_log_negative_undefined_l264_26402


namespace NUMINAMATH_CALUDE_bike_price_proof_l264_26422

theorem bike_price_proof (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 150 ∧ 
  upfront_percentage = 0.1 ∧ 
  upfront_payment = upfront_percentage * total_price →
  total_price = 1500 :=
by sorry

end NUMINAMATH_CALUDE_bike_price_proof_l264_26422


namespace NUMINAMATH_CALUDE_zaras_goats_l264_26404

theorem zaras_goats (cows sheep : ℕ) (groups : ℕ) (animals_per_group : ℕ) : 
  cows = 24 → 
  sheep = 7 → 
  groups = 3 → 
  animals_per_group = 48 → 
  groups * animals_per_group - cows - sheep = 113 :=
by sorry

end NUMINAMATH_CALUDE_zaras_goats_l264_26404


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l264_26463

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : perp m α) 
  (h3 : perp n α) : 
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l264_26463


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l264_26438

theorem complex_expression_evaluation :
  let a : ℂ := 1 + 2*I
  let b : ℂ := 2 + I
  a * b - 2 * b^2 = -6 - 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l264_26438


namespace NUMINAMATH_CALUDE_absolute_value_comparison_l264_26461

theorem absolute_value_comparison (m n : ℝ) : m < n → n < 0 → abs m > abs n := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_comparison_l264_26461
