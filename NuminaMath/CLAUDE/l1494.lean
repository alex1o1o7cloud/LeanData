import Mathlib

namespace NUMINAMATH_CALUDE_gcd_of_squares_l1494_149409

theorem gcd_of_squares : Nat.gcd (101^2 + 202^2 + 303^2) (100^2 + 201^2 + 304^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_l1494_149409


namespace NUMINAMATH_CALUDE_square_area_calculation_l1494_149479

theorem square_area_calculation (side_length : ℝ) (h : side_length = 28) :
  side_length ^ 2 = 784 := by
  sorry

#check square_area_calculation

end NUMINAMATH_CALUDE_square_area_calculation_l1494_149479


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1494_149466

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8*x + 20) / (m*x^2 + 2*(m+1)*x + 9*m + 4) < 0) ↔ m < -1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1494_149466


namespace NUMINAMATH_CALUDE_bridge_length_l1494_149477

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 130 ∧ 
  train_speed_kmh = 54 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 320 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l1494_149477


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l1494_149448

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

/-- Given two points M(a,3) and N(-4,b) symmetric with respect to the origin, prove that a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin a 3 (-4) b) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l1494_149448


namespace NUMINAMATH_CALUDE_inequality_proof_l1494_149435

theorem inequality_proof (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : 0 ≤ x ∧ x ≤ π / 2) :
  a^(Real.sin x) * (a + 1)^(Real.cos x) ≥ a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1494_149435


namespace NUMINAMATH_CALUDE_det_A_l1494_149488

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 0, -2; 8, 5, -4; 3, 3, 6]

theorem det_A : Matrix.det A = 108 := by sorry

end NUMINAMATH_CALUDE_det_A_l1494_149488


namespace NUMINAMATH_CALUDE_rose_ratio_l1494_149418

theorem rose_ratio (total : ℕ) (tulips : ℕ) (carnations : ℕ) :
  total = 40 ∧ tulips = 10 ∧ carnations = 14 →
  (total - tulips - carnations : ℚ) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rose_ratio_l1494_149418


namespace NUMINAMATH_CALUDE_max_value_2q_minus_r_l1494_149478

theorem max_value_2q_minus_r :
  ∀ q r : ℕ+, 
  965 = 22 * q + r → 
  ∀ q' r' : ℕ+, 
  965 = 22 * q' + r' → 
  2 * q - r ≤ 67 :=
by sorry

end NUMINAMATH_CALUDE_max_value_2q_minus_r_l1494_149478


namespace NUMINAMATH_CALUDE_sum_of_tenth_powers_l1494_149487

theorem sum_of_tenth_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tenth_powers_l1494_149487


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1494_149425

theorem quadratic_always_positive (m : ℝ) :
  (∀ x : ℝ, (4 - m) * x^2 - 3 * x + (4 + m) > 0) ↔ 
  (-Real.sqrt 55 / 2 < m ∧ m < Real.sqrt 55 / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1494_149425


namespace NUMINAMATH_CALUDE_christmas_gifts_left_l1494_149459

/-- The number of gifts left under the Christmas tree -/
def gifts_left (initial : ℕ) (sent : ℕ) : ℕ := initial - sent

/-- Theorem stating that given 77 initial gifts and 66 sent gifts, 11 gifts are left -/
theorem christmas_gifts_left : gifts_left 77 66 = 11 := by
  sorry

end NUMINAMATH_CALUDE_christmas_gifts_left_l1494_149459


namespace NUMINAMATH_CALUDE_marilyn_bottle_caps_l1494_149440

/-- The number of bottle caps Marilyn has after receiving some from Nancy -/
def total_bottle_caps (initial : Real) (received : Real) : Real :=
  initial + received

/-- Theorem: Marilyn's total bottle caps is the sum of her initial count and what she received -/
theorem marilyn_bottle_caps (initial : Real) (received : Real) :
  total_bottle_caps initial received = initial + received := by
  sorry

end NUMINAMATH_CALUDE_marilyn_bottle_caps_l1494_149440


namespace NUMINAMATH_CALUDE_max_students_is_nine_l1494_149411

/-- Represents the answer choices for each question -/
inductive Choice
| A
| B
| C

/-- Represents a student's answers to all questions -/
def StudentAnswers := Fin 4 → Choice

/-- The property that for any 3 students, there is at least one question where their answers differ -/
def DifferentAnswersExist (answers : Finset StudentAnswers) : Prop :=
  ∀ s1 s2 s3 : StudentAnswers, s1 ∈ answers → s2 ∈ answers → s3 ∈ answers →
    s1 ≠ s2 → s2 ≠ s3 → s1 ≠ s3 →
    ∃ q : Fin 4, s1 q ≠ s2 q ∧ s2 q ≠ s3 q ∧ s1 q ≠ s3 q

/-- The main theorem stating that the maximum number of students is 9 -/
theorem max_students_is_nine :
  ∃ (answers : Finset StudentAnswers),
    DifferentAnswersExist answers ∧
    answers.card = 9 ∧
    ∀ (larger_set : Finset StudentAnswers),
      larger_set.card > 9 →
      ¬DifferentAnswersExist larger_set :=
sorry

end NUMINAMATH_CALUDE_max_students_is_nine_l1494_149411


namespace NUMINAMATH_CALUDE_expression_simplification_l1494_149474

theorem expression_simplification (a : ℝ) (h1 : a^2 - 4*a + 3 = 0) (h2 : a ≠ 3) :
  (a^2 - 9) / (a^2 - 3*a) / ((a^2 + 9) / a + 6) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1494_149474


namespace NUMINAMATH_CALUDE_lana_extra_flowers_l1494_149417

/-- The number of extra flowers Lana picked -/
def extra_flowers (tulips roses used : ℕ) : ℕ :=
  tulips + roses - used

/-- Theorem: Lana picked 3 extra flowers -/
theorem lana_extra_flowers :
  extra_flowers 36 37 70 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lana_extra_flowers_l1494_149417


namespace NUMINAMATH_CALUDE_tunnel_length_is_900_l1494_149499

/-- Calculates the length of a tunnel given train parameters -/
def tunnel_length (train_length : ℝ) (total_time : ℝ) (inside_time : ℝ) : ℝ :=
  -- Define the tunnel length calculation here
  0 -- Placeholder, replace with actual calculation

/-- Theorem stating that given the specified conditions, the tunnel length is 900 meters -/
theorem tunnel_length_is_900 :
  tunnel_length 300 60 30 = 900 := by
  sorry

end NUMINAMATH_CALUDE_tunnel_length_is_900_l1494_149499


namespace NUMINAMATH_CALUDE_product_is_even_l1494_149424

theorem product_is_even (a b c : ℤ) : 
  ∃ k : ℤ, (7 * a + b - 2 * c + 1) * (3 * a - 5 * b + 4 * c + 10) = 2 * k := by
sorry

end NUMINAMATH_CALUDE_product_is_even_l1494_149424


namespace NUMINAMATH_CALUDE_team_selection_ways_l1494_149465

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the required team size
def team_size : ℕ := 8

-- Define the number of quadruplets that must be in the team
def required_quadruplets : ℕ := 2

-- Define the number of non-quadruplet players
def non_quadruplet_players : ℕ := total_players - num_quadruplets

-- Define the number of additional players needed after selecting quadruplets
def additional_players : ℕ := team_size - required_quadruplets

-- Theorem statement
theorem team_selection_ways : 
  (Nat.choose num_quadruplets required_quadruplets) * 
  (Nat.choose non_quadruplet_players additional_players) = 18018 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_ways_l1494_149465


namespace NUMINAMATH_CALUDE_rectangle_with_cut_corners_l1494_149445

/-- Given a rectangle ABCD with identical isosceles right triangles cut off from its corners,
    each having a leg of length a, and the total area cut off is 160 cm²,
    if the longer side of ABCD is 32√2 cm, then the length of PQ is 16√2 cm. -/
theorem rectangle_with_cut_corners (a : ℝ) (l : ℝ) (PQ : ℝ) :
  (4 * (1/2 * a^2) = 160) →  -- Total area cut off
  (l = 32 * Real.sqrt 2) →   -- Longer side of ABCD
  (PQ = l - 2*a) →           -- Definition of PQ
  (PQ = 16 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_with_cut_corners_l1494_149445


namespace NUMINAMATH_CALUDE_set_A_properties_l1494_149410

def A : Set ℝ := {x | x^2 - 4 = 0}

theorem set_A_properties :
  (2 ∈ A) ∧
  (-2 ∈ A) ∧
  (A = {-2, 2}) ∧
  (∅ ⊆ A) := by
sorry

end NUMINAMATH_CALUDE_set_A_properties_l1494_149410


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l1494_149454

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 := by
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l1494_149454


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l1494_149493

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_percentage : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 14)
  (h2 : regular_hours = 40)
  (h3 : overtime_percentage = 0.75)
  (h4 : total_hours = 57.88) :
  ∃ (total_compensation : ℝ), 
    abs (total_compensation - 998.06) < 0.01 ∧
    total_compensation = 
      regular_rate * regular_hours + 
      (regular_rate * (1 + overtime_percentage)) * (total_hours - regular_hours) :=
by sorry


end NUMINAMATH_CALUDE_bus_driver_compensation_l1494_149493


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l1494_149423

open Complex

theorem max_imaginary_part_of_roots (z : ℂ) :
  z^6 - z^5 + z^4 - z^3 + z^2 - z + 1 = 0 →
  ∃ (φ : ℝ), -π/2 ≤ φ ∧ φ ≤ π/2 ∧
  (∀ (w : ℂ), w^6 - w^5 + w^4 - w^3 + w^2 - w + 1 = 0 →
    w.im ≤ Real.sin φ) ∧
  φ = (900 * π) / (7 * 180) :=
sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l1494_149423


namespace NUMINAMATH_CALUDE_triangle_properties_l1494_149443

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.C + Real.sqrt 3 * Real.sin t.A = t.b + t.c ∧
  t.a = 6 ∧
  1/2 * t.b * t.c * Real.sin t.A = 9 * Real.sqrt 3

-- State the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.A = π/3 ∧ t.a + t.b + t.c = 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1494_149443


namespace NUMINAMATH_CALUDE_product_of_numbers_l1494_149432

theorem product_of_numbers (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 31) : a * b = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1494_149432


namespace NUMINAMATH_CALUDE_count_male_students_l1494_149464

theorem count_male_students (total : ℕ) (girls : ℕ) (h1 : total = 13) (h2 : girls = 6) :
  total - girls = 7 := by
  sorry

end NUMINAMATH_CALUDE_count_male_students_l1494_149464


namespace NUMINAMATH_CALUDE_no_real_roots_of_composition_l1494_149412

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_real_roots_of_composition 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, quadratic a b c x ≠ x) :
  ∀ x : ℝ, quadratic a b c (quadratic a b c x) ≠ x := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_of_composition_l1494_149412


namespace NUMINAMATH_CALUDE_sravans_journey_l1494_149438

/-- Calculates the total distance traveled given the conditions of Sravan's journey -/
theorem sravans_journey (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 15 ∧ speed1 = 45 ∧ speed2 = 30 → 
  ∃ (distance : ℝ), 
    distance / (2 * speed1) + distance / (2 * speed2) = total_time ∧
    distance = 540 := by
  sorry


end NUMINAMATH_CALUDE_sravans_journey_l1494_149438


namespace NUMINAMATH_CALUDE_existence_of_non_triangle_forming_numbers_l1494_149416

theorem existence_of_non_triangle_forming_numbers : 
  ∃ (a b : ℕ), a > 1000 ∧ b > 1000 ∧ 
  (∀ (c : ℕ), ∃ (k : ℕ), c = k^2 → 
    ¬(a + b > c ∧ a + c > b ∧ b + c > a)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_non_triangle_forming_numbers_l1494_149416


namespace NUMINAMATH_CALUDE_equilateral_triangle_division_l1494_149457

/-- A type representing a polygon with a given number of sides -/
def Polygon (n : ℕ) := Unit

/-- A type representing an equilateral triangle -/
def EquilateralTriangle := Unit

/-- A function that divides an equilateral triangle into two polygons -/
def divide (t : EquilateralTriangle) : Polygon 2020 × Polygon 2021 := sorry

/-- Theorem stating that an equilateral triangle can be divided into a 2020-gon and a 2021-gon -/
theorem equilateral_triangle_division :
  ∃ (t : EquilateralTriangle), ∃ (p : Polygon 2020 × Polygon 2021), divide t = p := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_division_l1494_149457


namespace NUMINAMATH_CALUDE_largest_divisible_n_l1494_149421

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > n → ¬((m + 8) ∣ (m^3 + 64))) ∧ 
  ((n + 8) ∣ (n^3 + 64)) ∧ 
  n = 440 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l1494_149421


namespace NUMINAMATH_CALUDE_abs_value_sum_diff_l1494_149486

theorem abs_value_sum_diff (a b c : ℝ) : 
  (|a| = 1) → (|b| = 2) → (|c| = 3) → (a > b) → (b > c) → 
  (a + b - c = 2 ∨ a + b - c = 0) := by
sorry

end NUMINAMATH_CALUDE_abs_value_sum_diff_l1494_149486


namespace NUMINAMATH_CALUDE_alphabet_letters_l1494_149431

theorem alphabet_letters (total : ℕ) (both : ℕ) (line_only : ℕ) (h1 : total = 60) (h2 : both = 20) (h3 : line_only = 36) :
  total = both + line_only + (total - (both + line_only)) →
  total - (both + line_only) = 24 :=
by sorry

end NUMINAMATH_CALUDE_alphabet_letters_l1494_149431


namespace NUMINAMATH_CALUDE_log_product_equality_l1494_149483

theorem log_product_equality : 
  ∀ (x : ℝ), x > 0 → 
  (Real.log 4 / Real.log 3) * (Real.log 5 / Real.log 4) * 
  (Real.log 6 / Real.log 5) * (Real.log 7 / Real.log 6) * 
  (Real.log 8 / Real.log 7) * (Real.log 9 / Real.log 8) * 
  (Real.log 10 / Real.log 9) * (Real.log 11 / Real.log 10) * 
  (Real.log 12 / Real.log 11) * (Real.log 13 / Real.log 12) * 
  (Real.log 14 / Real.log 13) * (Real.log 15 / Real.log 14) = 
  1 + Real.log 5 / Real.log 3 := by
sorry

end NUMINAMATH_CALUDE_log_product_equality_l1494_149483


namespace NUMINAMATH_CALUDE_concert_ticket_problem_l1494_149461

def ticket_price_possibilities (seventh_grade_total eighth_grade_total : ℕ) : ℕ :=
  (Finset.filter (fun x => seventh_grade_total % x = 0 ∧ eighth_grade_total % x = 0)
    (Finset.range (min seventh_grade_total eighth_grade_total + 1))).card

theorem concert_ticket_problem : ticket_price_possibilities 36 90 = 6 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_problem_l1494_149461


namespace NUMINAMATH_CALUDE_first_player_always_wins_l1494_149472

/-- Represents a cubic polynomial of the form x^3 + ax^2 + bx + c -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determines if a cubic polynomial has exactly one real root -/
def has_exactly_one_real_root (p : CubicPolynomial) : Prop :=
  ∃! x : ℝ, x^3 + p.a * x^2 + p.b * x + p.c = 0

/-- Represents a strategy for the first player -/
def first_player_strategy : CubicPolynomial → CubicPolynomial → CubicPolynomial :=
  sorry

/-- Represents a strategy for the second player -/
def second_player_strategy : CubicPolynomial → CubicPolynomial :=
  sorry

/-- The main theorem stating that the first player can always win -/
theorem first_player_always_wins :
  ∀ (second_strategy : CubicPolynomial → CubicPolynomial),
    ∃ (first_strategy : CubicPolynomial → CubicPolynomial → CubicPolynomial),
      ∀ (initial : CubicPolynomial),
        has_exactly_one_real_root (first_strategy initial (second_strategy initial)) :=
sorry

end NUMINAMATH_CALUDE_first_player_always_wins_l1494_149472


namespace NUMINAMATH_CALUDE_marbles_in_jar_l1494_149429

/-- The number of marbles in a jar when two boys combine their collections -/
theorem marbles_in_jar (ben_marbles : ℕ) (leo_extra_marbles : ℕ) : 
  ben_marbles = 56 → leo_extra_marbles = 20 → 
  ben_marbles + (ben_marbles + leo_extra_marbles) = 132 := by
  sorry

#check marbles_in_jar

end NUMINAMATH_CALUDE_marbles_in_jar_l1494_149429


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l1494_149414

theorem subtraction_of_decimals : (3.75 : ℝ) - (1.46 : ℝ) = 2.29 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l1494_149414


namespace NUMINAMATH_CALUDE_unique_single_solution_quadratic_l1494_149462

theorem unique_single_solution_quadratic :
  ∃! (p : ℝ), p ≠ 0 ∧ (∃! x : ℝ, p * x^2 - 12 * x + 4 = 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_single_solution_quadratic_l1494_149462


namespace NUMINAMATH_CALUDE_abs_inequality_l1494_149447

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l1494_149447


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l1494_149463

theorem quadratic_roots_condition (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → 
  d = 9.8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l1494_149463


namespace NUMINAMATH_CALUDE_function_form_proof_l1494_149469

theorem function_form_proof (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x + Real.cos x ^ 2| ≤ 3/4)
  (h2 : ∀ x, |f x - Real.sin x ^ 2| ≤ 1/4) :
  ∀ x, f x = 3/4 - Real.cos x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_form_proof_l1494_149469


namespace NUMINAMATH_CALUDE_square_grid_division_l1494_149434

theorem square_grid_division (m n k : ℕ) (h : m * m = n * k) :
  ∃ (d : ℕ), d ∣ m ∧ d ∣ n ∧ (m / d) * d = k ∧ (n / d) * d = n :=
sorry

end NUMINAMATH_CALUDE_square_grid_division_l1494_149434


namespace NUMINAMATH_CALUDE_alternating_ball_probability_l1494_149498

def num_black_balls : ℕ := 5
def num_white_balls : ℕ := 4
def total_balls : ℕ := num_black_balls + num_white_balls

def alternating_sequence (n : ℕ) : List Bool :=
  List.map (fun i => i % 2 = 0) (List.range n)

def is_valid_sequence (seq : List Bool) : Prop :=
  seq.length = total_balls ∧
  seq.head? = some true ∧
  seq = alternating_sequence total_balls

def num_valid_sequences : ℕ := 1

def total_outcomes : ℕ := Nat.choose total_balls num_black_balls

theorem alternating_ball_probability :
  (num_valid_sequences : ℚ) / total_outcomes = 1 / 126 :=
sorry

end NUMINAMATH_CALUDE_alternating_ball_probability_l1494_149498


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1494_149489

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x : ℕ | ∃ k ∈ A, x = 2 * k}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1494_149489


namespace NUMINAMATH_CALUDE_negation_equivalence_l1494_149495

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ = x₀ - 1) ↔ 
  (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1494_149495


namespace NUMINAMATH_CALUDE_color_coat_drying_time_l1494_149446

/-- Represents the drying time for nail polish coats -/
structure NailPolishDryingTime where
  base_coat : ℕ
  color_coat : ℕ
  top_coat : ℕ
  total_time : ℕ

/-- Theorem: Given the conditions of Jane's nail polish application,
    prove that each color coat takes 3 minutes to dry -/
theorem color_coat_drying_time (t : NailPolishDryingTime)
  (h1 : t.base_coat = 2)
  (h2 : t.top_coat = 5)
  (h3 : t.total_time = 13)
  (h4 : t.total_time = t.base_coat + 2 * t.color_coat + t.top_coat) :
  t.color_coat = 3 := by
  sorry

end NUMINAMATH_CALUDE_color_coat_drying_time_l1494_149446


namespace NUMINAMATH_CALUDE_blue_paper_side_length_l1494_149451

theorem blue_paper_side_length (red_side : ℝ) (blue_side1 : ℝ) (blue_side2 : ℝ) : 
  red_side = 5 →
  blue_side1 = 4 →
  red_side * red_side = blue_side1 * blue_side2 →
  blue_side2 = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_blue_paper_side_length_l1494_149451


namespace NUMINAMATH_CALUDE_five_fourths_of_sum_l1494_149402

theorem five_fourths_of_sum : ∀ (a b c d : ℚ),
  a = 6 / 3 ∧ b = 8 / 4 → (5 / 4) * (a + b) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_sum_l1494_149402


namespace NUMINAMATH_CALUDE_range_of_x₁_l1494_149482

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the condition given in the problem
def Condition (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ + x₂ = 1 → f x₁ + f 0 > f x₂ + f 1

-- Theorem statement
theorem range_of_x₁ (h_increasing : IsIncreasing f) (h_condition : Condition f) :
  ∀ x₁, (∃ x₂, x₁ + x₂ = 1 ∧ f x₁ + f 0 > f x₂ + f 1) ↔ x₁ > 1 :=
by sorry


end NUMINAMATH_CALUDE_range_of_x₁_l1494_149482


namespace NUMINAMATH_CALUDE_cosine_equation_roots_l1494_149481

theorem cosine_equation_roots :
  ∃ (a b c : ℝ), (∀ x : ℝ, 4 * Real.cos (2007 * x) = 2007 * x ↔ x = a ∨ x = b ∨ x = c) ∧
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
sorry

end NUMINAMATH_CALUDE_cosine_equation_roots_l1494_149481


namespace NUMINAMATH_CALUDE_second_brand_growth_rate_l1494_149420

/-- Proves that the growth rate of the second brand of computers is approximately 0.7 million households per year -/
theorem second_brand_growth_rate (initial_first : ℝ) (growth_first : ℝ) (initial_second : ℝ) (time_to_equal : ℝ)
  (h1 : initial_first = 4.9)
  (h2 : growth_first = 0.275)
  (h3 : initial_second = 2.5)
  (h4 : time_to_equal = 5.647)
  (h5 : initial_first + growth_first * time_to_equal = initial_second + growth_second * time_to_equal) :
  ∃ growth_second : ℝ, abs (growth_second - 0.7) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_second_brand_growth_rate_l1494_149420


namespace NUMINAMATH_CALUDE_mary_initial_marbles_l1494_149484

/-- The number of yellow marbles Mary gave to Joan -/
def marbles_given : ℕ := 3

/-- The number of yellow marbles Mary has left -/
def marbles_left : ℕ := 6

/-- The initial number of yellow marbles Mary had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem mary_initial_marbles : initial_marbles = 9 := by
  sorry

end NUMINAMATH_CALUDE_mary_initial_marbles_l1494_149484


namespace NUMINAMATH_CALUDE_jeds_cards_after_four_weeks_l1494_149404

/-- Calculates the number of cards Jed has after a given number of weeks -/
def cards_after_weeks (initial_cards : ℕ) (cards_per_week : ℕ) (cards_given_away : ℕ) (weeks : ℕ) : ℕ :=
  initial_cards + cards_per_week * weeks - cards_given_away * (weeks / 2)

/-- Proves that Jed will have 40 cards after 4 weeks -/
theorem jeds_cards_after_four_weeks :
  ∃ (weeks : ℕ), cards_after_weeks 20 6 2 weeks = 40 ∧ weeks = 4 :=
by
  sorry

#check jeds_cards_after_four_weeks

end NUMINAMATH_CALUDE_jeds_cards_after_four_weeks_l1494_149404


namespace NUMINAMATH_CALUDE_least_number_divisible_by_seven_with_remainder_one_l1494_149437

theorem least_number_divisible_by_seven_with_remainder_one : ∃ n : ℕ, 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 6 → n % k = 1) ∧ 
  n % 7 = 0 ∧
  (∀ m : ℕ, m < n → ¬(∀ k : ℕ, 2 ≤ k ∧ k ≤ 6 → m % k = 1) ∨ m % 7 ≠ 0) ∧
  n = 301 :=
by
  sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_seven_with_remainder_one_l1494_149437


namespace NUMINAMATH_CALUDE_M_inter_compl_N_l1494_149406

/-- The set M defined by the square root function -/
def M : Set ℝ := {x | ∃ y, y = Real.sqrt x}

/-- The set N defined by a quadratic inequality -/
def N : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}

/-- The theorem stating the intersection of M and the complement of N -/
theorem M_inter_compl_N : M ∩ (Set.univ \ N) = {x | 0 ≤ x ∧ x < 2 ∨ x > 4} := by sorry

end NUMINAMATH_CALUDE_M_inter_compl_N_l1494_149406


namespace NUMINAMATH_CALUDE_one_more_square_possible_l1494_149428

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a square that can be cut out from the grid -/
structure Square :=
  (size : ℕ)

/-- The number of squares already cut out -/
def cut_squares : ℕ := 15

/-- The function that determines if it's possible to cut out one more square -/
def can_cut_one_more (g : Grid) (s : Square) (n : ℕ) : Prop :=
  ∃ (remaining : ℕ), remaining > 0

/-- The theorem statement -/
theorem one_more_square_possible (g : Grid) (s : Square) :
  g.size = 11 → s.size = 2 → can_cut_one_more g s cut_squares :=
sorry

end NUMINAMATH_CALUDE_one_more_square_possible_l1494_149428


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1494_149407

theorem pure_imaginary_complex_number (x : ℝ) : 
  (((x^2 - 4) : ℂ) + (x^2 + 3*x + 2)*I = (0 : ℂ) + y*I ∧ y ≠ 0) → x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1494_149407


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l1494_149439

theorem largest_solution_of_equation (x : ℝ) :
  (x / 3 + 1 / (3 * x) = 1 / 2) → x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l1494_149439


namespace NUMINAMATH_CALUDE_cubic_and_sixth_degree_polynomial_roots_l1494_149458

theorem cubic_and_sixth_degree_polynomial_roots : ∀ s : ℂ,
  s^3 - 2*s^2 + s - 1 = 0 → s^6 - 16*s - 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_and_sixth_degree_polynomial_roots_l1494_149458


namespace NUMINAMATH_CALUDE_tan_half_sum_of_angles_l1494_149450

theorem tan_half_sum_of_angles (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 3/5)
  (h2 : Real.sin a + Real.sin b = 5/13) : 
  Real.tan ((a + b)/2) = 25/39 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_sum_of_angles_l1494_149450


namespace NUMINAMATH_CALUDE_complex_sum_difference_l1494_149460

theorem complex_sum_difference (A M S : ℂ) (P : ℝ) 
  (hA : A = 3 - 2*I) 
  (hM : M = -5 + 3*I) 
  (hS : S = -2*I) 
  (hP : P = 3) : 
  A + M + S - P = -5 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_difference_l1494_149460


namespace NUMINAMATH_CALUDE_A_run_time_l1494_149426

/-- The time it takes for A to run 160 meters -/
def time_A : ℝ := 28

/-- The time it takes for B to run 160 meters -/
def time_B : ℝ := 32

/-- The distance A runs -/
def distance_A : ℝ := 160

/-- The distance B runs when A finishes -/
def distance_B : ℝ := 140

theorem A_run_time :
  (distance_A / time_A = distance_B / time_B) ∧
  (distance_A - distance_B = 20) →
  time_A = 28 := by sorry

end NUMINAMATH_CALUDE_A_run_time_l1494_149426


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l1494_149485

theorem rectangle_dimension_change (L B : ℝ) (x : ℝ) (h_positive : L > 0 ∧ B > 0) :
  (1.20 * L) * (B * (1 - x / 100)) = 1.04 * (L * B) → x = 40 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l1494_149485


namespace NUMINAMATH_CALUDE_common_factor_polynomial_l1494_149436

theorem common_factor_polynomial (a b c : ℤ) :
  ∃ (k : ℤ), (12 * a * b^3 * c + 8 * a^3 * b) = k * (4 * a * b) ∧ k ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_common_factor_polynomial_l1494_149436


namespace NUMINAMATH_CALUDE_a_in_S_l1494_149475

theorem a_in_S (S T : Set ℕ) (a : ℕ) 
  (h1 : S = {1, 2})
  (h2 : T = {a})
  (h3 : S ∪ T = S) :
  a ∈ S := by
  sorry

end NUMINAMATH_CALUDE_a_in_S_l1494_149475


namespace NUMINAMATH_CALUDE_fraction_sum_and_divide_l1494_149415

theorem fraction_sum_and_divide : (3/20 + 5/200 + 7/2000) / 2 = 0.08925 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_and_divide_l1494_149415


namespace NUMINAMATH_CALUDE_cubic_expression_equality_l1494_149467

theorem cubic_expression_equality : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by sorry

end NUMINAMATH_CALUDE_cubic_expression_equality_l1494_149467


namespace NUMINAMATH_CALUDE_problem_solution_l1494_149492

theorem problem_solution (a b c d m n : ℕ) 
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 1989)
  (sum_linear : a + b + c + d = m^2)
  (max_value : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1494_149492


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1494_149456

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, -x^2 + 2*x + 3 ≤ a^2 - 3*a) ↔ 
  (a ≤ -1 ∨ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1494_149456


namespace NUMINAMATH_CALUDE_fourth_person_height_l1494_149400

/-- Proves that the height of the 4th person is 85 inches given the conditions of the problem -/
theorem fourth_person_height :
  ∀ (h₁ h₂ h₃ h₄ : ℝ),
    h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- Heights are in increasing order
    h₂ - h₁ = 2 →  -- Difference between 1st and 2nd person
    h₃ - h₂ = 2 →  -- Difference between 2nd and 3rd person
    h₄ - h₃ = 6 →  -- Difference between 3rd and 4th person
    (h₁ + h₂ + h₃ + h₄) / 4 = 79 →  -- Average height
    h₄ = 85 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l1494_149400


namespace NUMINAMATH_CALUDE_cube_face_area_l1494_149405

/-- Given a cube with surface area 36 square centimeters, 
    prove that the area of one face is 6 square centimeters. -/
theorem cube_face_area (surface_area : ℝ) (h : surface_area = 36) :
  surface_area / 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_area_l1494_149405


namespace NUMINAMATH_CALUDE_square_side_sum_l1494_149455

theorem square_side_sum (b d : ℕ) : 
  15^2 = b^2 + 10^2 + d^2 → (b + d = 13 ∨ b + d = 15) :=
by sorry

end NUMINAMATH_CALUDE_square_side_sum_l1494_149455


namespace NUMINAMATH_CALUDE_cost_price_of_article_l1494_149444

/-- Proves that the cost price of an article is 800, given the conditions from the problem. -/
theorem cost_price_of_article : ∃ (C : ℝ), 
  (C = 800) ∧ 
  (1.05 * C - (0.95 * C + 0.1 * (0.95 * C)) = 4) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_article_l1494_149444


namespace NUMINAMATH_CALUDE_diamond_operation_result_l1494_149496

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the diamond operation
def diamond : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.three
  | Element.one, Element.four => Element.two
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.two
  | Element.two, Element.four => Element.four
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.two
  | Element.three, Element.three => Element.four
  | Element.three, Element.four => Element.one
  | Element.four, Element.one => Element.two
  | Element.four, Element.two => Element.four
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.three

theorem diamond_operation_result :
  diamond (diamond Element.three Element.one) (diamond Element.four Element.two) = Element.one := by
  sorry

end NUMINAMATH_CALUDE_diamond_operation_result_l1494_149496


namespace NUMINAMATH_CALUDE_f_divides_implies_f_divides_for_all_l1494_149491

/-- Definition of c(n, i) -/
def c (n i : ℕ) : Fin 2 :=
  if (Nat.choose n i) % 2 = 0 then 0 else 1

/-- Definition of f(n, q) -/
def f (n q : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (fun i => (c n i : ℕ) * q ^ i)

/-- Main theorem -/
theorem f_divides_implies_f_divides_for_all (m n q : ℕ+) 
  (h_not_power : ¬∃ k : ℕ, q + 1 = 2 ^ k) :
  (f m q ∣ f n q) → ∀ r : ℕ+, (f m r ∣ f n r) := by
  sorry

end NUMINAMATH_CALUDE_f_divides_implies_f_divides_for_all_l1494_149491


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1494_149470

theorem negation_of_proposition (p : (x : ℝ) → x > 1 → x^3 + 1 > 8*x) :
  (¬ ∀ (x : ℝ), x > 1 → x^3 + 1 > 8*x) ↔ 
  (∃ (x : ℝ), x > 1 ∧ x^3 + 1 ≤ 8*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1494_149470


namespace NUMINAMATH_CALUDE_minimum_greenhouse_dimensions_l1494_149427

/-- Represents the dimensions of a rectangular greenhouse. -/
structure Greenhouse where
  height : ℝ
  width : ℝ

/-- Checks if the greenhouse satisfies the given conditions. -/
def satisfiesConditions (g : Greenhouse) : Prop :=
  g.width = 2 * g.height ∧ g.height * g.width ≥ 800

/-- Theorem stating the minimum dimensions of the greenhouse. -/
theorem minimum_greenhouse_dimensions :
  ∃ (g : Greenhouse), satisfiesConditions g ∧
    ∀ (g' : Greenhouse), satisfiesConditions g' → g.height ≤ g'.height ∧ g.width ≤ g'.width :=
  sorry

end NUMINAMATH_CALUDE_minimum_greenhouse_dimensions_l1494_149427


namespace NUMINAMATH_CALUDE_larger_number_is_sixteen_l1494_149441

theorem larger_number_is_sixteen (a b : ℝ) (h1 : a - b = 5) (h2 : a + b = 27) :
  max a b = 16 := by sorry

end NUMINAMATH_CALUDE_larger_number_is_sixteen_l1494_149441


namespace NUMINAMATH_CALUDE_circular_matrix_determinant_properties_l1494_149430

def circularMatrix (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a, b, c],
    ![c, a, b],
    ![b, c, a]]

theorem circular_matrix_determinant_properties :
  (∃ (S : Set (ℚ × ℚ × ℚ)), Set.Infinite S ∧
    ∀ (abc : ℚ × ℚ × ℚ), abc ∈ S →
      Matrix.det (circularMatrix abc.1 abc.2.1 abc.2.2) = 1) ∧
  (∃ (T : Set (ℤ × ℤ × ℤ)), Set.Finite T ∧
    ∀ (abc : ℤ × ℤ × ℤ), Matrix.det (circularMatrix ↑abc.1 ↑abc.2.1 ↑abc.2.2) = 1 →
      abc ∈ T) := by
  sorry

end NUMINAMATH_CALUDE_circular_matrix_determinant_properties_l1494_149430


namespace NUMINAMATH_CALUDE_solve_problem_l1494_149442

-- Define the type for gender
inductive Gender
| Boy
| Girl

-- Define the type for a child
structure Child :=
  (name : String)
  (gender : Gender)
  (statement : Gender)

-- Define the problem setup
def problem_setup (sasha zhenya : Child) : Prop :=
  (sasha.name = "Sasha" ∧ zhenya.name = "Zhenya") ∧
  (sasha.gender ≠ zhenya.gender) ∧
  (sasha.statement = Gender.Boy) ∧
  (zhenya.statement = Gender.Girl) ∧
  (sasha.statement ≠ sasha.gender ∨ zhenya.statement ≠ zhenya.gender)

-- Theorem to prove
theorem solve_problem (sasha zhenya : Child) :
  problem_setup sasha zhenya →
  sasha.gender = Gender.Girl ∧ zhenya.gender = Gender.Boy :=
by
  sorry

end NUMINAMATH_CALUDE_solve_problem_l1494_149442


namespace NUMINAMATH_CALUDE_det_matrix_2x2_l1494_149497

theorem det_matrix_2x2 (A : Matrix (Fin 2) (Fin 2) ℝ) : 
  A = ![![6, -2], ![-5, 3]] → Matrix.det A = 8 := by
  sorry

end NUMINAMATH_CALUDE_det_matrix_2x2_l1494_149497


namespace NUMINAMATH_CALUDE_unique_solution_square_sum_l1494_149490

theorem unique_solution_square_sum (x y : ℝ) : 
  (x - 2*y)^2 + (y - 1)^2 = 0 ↔ x = 2 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_square_sum_l1494_149490


namespace NUMINAMATH_CALUDE_simple_interest_rate_l1494_149449

/-- Given a principal amount that grows to 7/6 of itself in 7 years under simple interest, 
    the annual interest rate is 1/42. -/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) : 
  ∃ R : ℝ, R > 0 ∧ P * (1 + 7 * R) = 7/6 * P ∧ R = 1/42 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l1494_149449


namespace NUMINAMATH_CALUDE_table_price_is_56_l1494_149471

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The condition that the price of 2 chairs and 1 table is 60% of the price of 1 chair and 2 tables -/
axiom price_ratio : 2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

/-- The condition that the price of 1 table and 1 chair is $64 -/
axiom total_price : chair_price + table_price = 64

/-- Theorem stating that the price of 1 table is $56 -/
theorem table_price_is_56 : table_price = 56 := by sorry

end NUMINAMATH_CALUDE_table_price_is_56_l1494_149471


namespace NUMINAMATH_CALUDE_age_difference_l1494_149433

/-- Given three people A, B, and C, where C is 14 years younger than A,
    prove that the total age of A and B is 14 years more than the total age of B and C. -/
theorem age_difference (A B C : ℕ) (h : C = A - 14) :
  (A + B) - (B + C) = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1494_149433


namespace NUMINAMATH_CALUDE_matrix_power_sum_l1494_149452

def A (a : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1, 3, a],
    ![0, 1, 5],
    ![0, 0, 1]]

theorem matrix_power_sum (a : ℝ) (n : ℕ) :
  (A a)^n = ![![1, 27, 2883],
              ![0,  1,   45],
              ![0,  0,    1]] →
  a + n = 264 := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_sum_l1494_149452


namespace NUMINAMATH_CALUDE_leftover_snacks_problem_l1494_149476

/-- Calculates the number of leftover snacks when feeding goats with dietary restrictions --/
def leftover_snacks (total_goats : ℕ) (restricted_goats : ℕ) (baby_carrots : ℕ) (cherry_tomatoes : ℕ) : ℕ :=
  let unrestricted_goats := total_goats - restricted_goats
  let tomatoes_per_restricted_goat := cherry_tomatoes / restricted_goats
  let leftover_tomatoes := cherry_tomatoes % restricted_goats
  let carrots_per_unrestricted_goat := baby_carrots / unrestricted_goats
  let leftover_carrots := baby_carrots % unrestricted_goats
  leftover_tomatoes + leftover_carrots

/-- Theorem stating that given the problem conditions, 6 snacks will be left over --/
theorem leftover_snacks_problem :
  leftover_snacks 9 3 124 56 = 6 := by
  sorry

end NUMINAMATH_CALUDE_leftover_snacks_problem_l1494_149476


namespace NUMINAMATH_CALUDE_free_square_positions_l1494_149419

-- Define the chessboard
def Chessboard := Fin 8 × Fin 8

-- Define the rectangle size
def RectangleSize := (3, 1)

-- Define the number of rectangles
def NumRectangles := 21

-- Define the possible free square positions
def FreePosns : List (Fin 8 × Fin 8) := [(3, 3), (3, 6), (6, 3), (6, 6)]

-- Theorem statement
theorem free_square_positions (board : Chessboard) (rectangles : Fin NumRectangles → Chessboard) :
  (∃! pos : Chessboard, pos ∉ (rectangles '' univ)) →
  (∃ pos ∈ FreePosns, pos ∉ (rectangles '' univ)) :=
sorry

end NUMINAMATH_CALUDE_free_square_positions_l1494_149419


namespace NUMINAMATH_CALUDE_set_operations_l1494_149480

-- Define the sets A and B
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

-- State the theorem
theorem set_operations :
  (A ∪ B = {0, 1, 2, 3}) ∧ (A ∩ B = {1, 2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1494_149480


namespace NUMINAMATH_CALUDE_amount_paid_l1494_149468

def lemonade_cups : ℕ := 2
def lemonade_price : ℚ := 2
def sandwich_count : ℕ := 2
def sandwich_price : ℚ := 2.5
def change_received : ℚ := 11

def total_cost : ℚ := lemonade_cups * lemonade_price + sandwich_count * sandwich_price

theorem amount_paid (paid : ℚ) : paid = 20 ↔ paid = total_cost + change_received := by
  sorry

end NUMINAMATH_CALUDE_amount_paid_l1494_149468


namespace NUMINAMATH_CALUDE_min_point_of_translated_abs_value_l1494_149408

-- Define the function representing the translated graph
def f (x : ℝ) : ℝ := |x - 3| - 1

-- Theorem stating that the minimum point of the graph is (3, -1)
theorem min_point_of_translated_abs_value :
  ∀ x : ℝ, f x ≥ f 3 ∧ f 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_min_point_of_translated_abs_value_l1494_149408


namespace NUMINAMATH_CALUDE_largest_prime_factor_93_l1494_149401

def numbers : List Nat := [57, 63, 85, 93, 133]

def largest_prime_factor (n : Nat) : Nat :=
  Nat.factors n |>.maximum?
    |>.getD 1  -- Default to 1 if the list is empty

theorem largest_prime_factor_93 :
  ∀ n ∈ numbers, n ≠ 93 → largest_prime_factor n ≤ largest_prime_factor 93 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_93_l1494_149401


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1494_149473

-- Define the sets U, A, and B
def U : Set ℝ := {x | x ≤ -1 ∨ x ≥ 0}
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 > 1}

-- State the theorem
theorem intersection_complement_equality :
  A ∩ (U \ B) = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1494_149473


namespace NUMINAMATH_CALUDE_shed_width_calculation_l1494_149453

theorem shed_width_calculation (backyard_length backyard_width shed_length sod_area : ℝ)
  (h1 : backyard_length = 20)
  (h2 : backyard_width = 13)
  (h3 : shed_length = 3)
  (h4 : sod_area = 245)
  (h5 : backyard_length * backyard_width - sod_area = shed_length * shed_width) :
  shed_width = 5 := by
  sorry

end NUMINAMATH_CALUDE_shed_width_calculation_l1494_149453


namespace NUMINAMATH_CALUDE_like_terms_exponent_l1494_149494

theorem like_terms_exponent (a : ℝ) : (∃ x : ℝ, x ≠ 0 ∧ ∃ k : ℝ, k ≠ 0 ∧ k * x^(2*a) = 5 * x^(a+3)) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_l1494_149494


namespace NUMINAMATH_CALUDE_polynomial_value_at_negative_one_l1494_149422

theorem polynomial_value_at_negative_one (r : ℝ) : 
  (fun x : ℝ => 3 * x^4 - 2 * x^3 + x^2 + 4 * x + r) (-1) = 0 → r = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_negative_one_l1494_149422


namespace NUMINAMATH_CALUDE_multiple_value_l1494_149413

-- Define the variables
variable (x : ℝ)
variable (m : ℝ)

-- State the theorem
theorem multiple_value (h1 : m * x + 36 = 48) (h2 : x = 4) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiple_value_l1494_149413


namespace NUMINAMATH_CALUDE_rounding_estimate_larger_l1494_149403

theorem rounding_estimate_larger (a b c d a' b' c' d' : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a' ≥ a → b' ≤ b → c' ≤ c → d' ≤ d →
  (a' : ℚ) / b' - c' - d' > (a : ℚ) / b - c - d :=
by sorry

end NUMINAMATH_CALUDE_rounding_estimate_larger_l1494_149403
