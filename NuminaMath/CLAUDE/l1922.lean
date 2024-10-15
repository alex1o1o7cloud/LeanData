import Mathlib

namespace NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l1922_192253

theorem half_abs_diff_squares_20_15 : 
  (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l1922_192253


namespace NUMINAMATH_CALUDE_milo_running_distance_l1922_192237

def cory_speed : ℝ := 12

theorem milo_running_distance
  (h1 : cory_speed = 12)
  (h2 : ∃ milo_skateboard_speed : ℝ, cory_speed = 2 * milo_skateboard_speed)
  (h3 : ∃ milo_running_speed : ℝ, milo_skateboard_speed = 2 * milo_running_speed)
  : ∃ distance : ℝ, distance = 2 * milo_running_speed ∧ distance = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_milo_running_distance_l1922_192237


namespace NUMINAMATH_CALUDE_smallest_c_and_b_for_real_roots_l1922_192215

theorem smallest_c_and_b_for_real_roots (c b : ℝ) : 
  (∀ x : ℝ, x^4 - c*x^3 + b*x^2 - c*x + 1 = 0 → x > 0) →
  (c > 0) →
  (b > 0) →
  (∀ c' b' : ℝ, c' > 0 → b' > 0 → 
    (∀ x : ℝ, x^4 - c'*x^3 + b'*x^2 - c'*x + 1 = 0 → x > 0) →
    c ≤ c') →
  c = 4 ∧ b = 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_c_and_b_for_real_roots_l1922_192215


namespace NUMINAMATH_CALUDE_fraction_product_l1922_192280

theorem fraction_product : (2 : ℚ) / 3 * (4 : ℚ) / 7 * (9 : ℚ) / 13 = (24 : ℚ) / 91 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l1922_192280


namespace NUMINAMATH_CALUDE_triangle_classification_l1922_192297

/-- Triangle classification based on side lengths --/
def TriangleType (a b c : ℝ) : Type :=
  { type : String // 
    type = "acute" ∧ a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2 ∨
    type = "right" ∧ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) ∨
    type = "obtuse" ∧ (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2) }

theorem triangle_classification :
  ∃ (t1 : TriangleType 4 6 8) (t2 : TriangleType 10 24 26) (t3 : TriangleType 10 12 14),
    t1.val = "obtuse" ∧ t2.val = "right" ∧ t3.val = "acute" := by
  sorry


end NUMINAMATH_CALUDE_triangle_classification_l1922_192297


namespace NUMINAMATH_CALUDE_triangle_area_half_parallelogram_area_l1922_192246

/-- The area of a triangle with equal base and height is half the area of a parallelogram with the same base and height. -/
theorem triangle_area_half_parallelogram_area (b h : ℝ) (b_pos : 0 < b) (h_pos : 0 < h) :
  (1 / 2 * b * h) = (1 / 2) * (b * h) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_half_parallelogram_area_l1922_192246


namespace NUMINAMATH_CALUDE_product_of_conjugates_equals_one_l1922_192257

theorem product_of_conjugates_equals_one :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_of_conjugates_equals_one_l1922_192257


namespace NUMINAMATH_CALUDE_inequality_proof_l1922_192265

theorem inequality_proof (x y : ℝ) (h1 : x > -1) (h2 : y > -1) (h3 : x + y = 1) :
  x / (y + 1) + y / (x + 1) ≥ 2/3 ∧ 
  (x / (y + 1) + y / (x + 1) = 2/3 ↔ x = 1/2 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1922_192265


namespace NUMINAMATH_CALUDE_collinear_vectors_l1922_192239

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-2, -1]
def c : Fin 2 → ℝ := ![1, 2]

def is_collinear (v w : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, v 0 * w 1 = t * v 1 * w 0

theorem collinear_vectors (k : ℝ) :
  is_collinear (fun i => a i + k * b i) c ↔ k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l1922_192239


namespace NUMINAMATH_CALUDE_logo_area_difference_l1922_192250

/-- The logo problem -/
theorem logo_area_difference :
  let triangle_side : ℝ := 12
  let square_side : ℝ := 2 * (9 - 3 * Real.sqrt 3)
  let overlapped_area : ℝ := square_side^2 - (square_side / 2) * (triangle_side - square_side / 2)
  let non_overlapping_area : ℝ := 2 * (square_side / 2) * (triangle_side - square_side / 2) / Real.sqrt 3
  overlapped_area - non_overlapping_area = 102.6 - 57.6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_logo_area_difference_l1922_192250


namespace NUMINAMATH_CALUDE_minimum_third_term_l1922_192275

def SallySequence (a : ℕ → ℕ) : Prop :=
  (∀ n ≥ 3, a n = a (n - 1) + a (n - 2)) ∧
  (a 8 = 400)

theorem minimum_third_term (a : ℕ → ℕ) (h : SallySequence a) :
  ∃ (m : ℕ), (∀ (b : ℕ → ℕ), SallySequence b → a 3 ≤ b 3) ∧ (a 3 = m) ∧ (m = 35) := by
  sorry

end NUMINAMATH_CALUDE_minimum_third_term_l1922_192275


namespace NUMINAMATH_CALUDE_area_of_similar_rectangle_l1922_192207

-- Define the properties of rectangle R1
def side_R1 : ℝ := 3
def area_R1 : ℝ := 18

-- Define the diagonal of rectangle R2
def diagonal_R2 : ℝ := 20

-- Theorem statement
theorem area_of_similar_rectangle (side_R1 area_R1 diagonal_R2 : ℝ) 
  (h1 : side_R1 > 0)
  (h2 : area_R1 > 0)
  (h3 : diagonal_R2 > 0) :
  let other_side_R1 := area_R1 / side_R1
  let ratio := other_side_R1 / side_R1
  let side_R2 := (diagonal_R2^2 / (1 + ratio^2))^(1/2)
  side_R2 * (ratio * side_R2) = 160 := by
sorry

end NUMINAMATH_CALUDE_area_of_similar_rectangle_l1922_192207


namespace NUMINAMATH_CALUDE_sample_size_is_450_l1922_192266

/-- Represents a population of students -/
structure Population where
  size : ℕ

/-- Represents a sample of students -/
structure Sample where
  size : ℕ

/-- Theorem: Given a population of 5000 students and a sample of 450 students,
    the sample size is 450. -/
theorem sample_size_is_450 (pop : Population) (sample : Sample) 
    (h1 : pop.size = 5000) (h2 : sample.size = 450) : 
  sample.size = 450 := by
  sorry

#check sample_size_is_450

end NUMINAMATH_CALUDE_sample_size_is_450_l1922_192266


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1922_192228

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, b > a ∧ a > 0 → 1 / a^2 > 1 / b^2) ∧
  (∃ a b : ℝ, 1 / a^2 > 1 / b^2 ∧ ¬(b > a ∧ a > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1922_192228


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2017th_term_l1922_192248

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum of the first n terms
  sum_property : ∀ n : ℕ, S n = n * (a 1 + a n) / 2
  arithmetic_property : ∀ n m : ℕ, a (n + m) - a n = m * (a 2 - a 1)

/-- Theorem stating the property of the 2017th term of the arithmetic sequence -/
theorem arithmetic_sequence_2017th_term 
  (seq : ArithmeticSequence)
  (h1 : seq.a 1 = -2017)
  (h2 : seq.S 2007 / 2007 - seq.S 2005 / 2005 = 2) :
  seq.a 2017 = 2015 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2017th_term_l1922_192248


namespace NUMINAMATH_CALUDE_final_cow_count_l1922_192272

def cow_count (initial : ℕ) (died : ℕ) (sold : ℕ) (increase : ℕ) (bought : ℕ) (gift : ℕ) : ℕ :=
  initial - died - sold + increase + bought + gift

theorem final_cow_count :
  cow_count 39 25 6 24 43 8 = 83 := by
  sorry

end NUMINAMATH_CALUDE_final_cow_count_l1922_192272


namespace NUMINAMATH_CALUDE_last_digit_is_four_l1922_192230

/-- Represents the process of repeatedly removing digits in odd positions -/
def remove_odd_positions (n : ℕ) : ℕ → ℕ
| 0 => 0
| 1 => n % 10
| m + 2 => remove_odd_positions (n / 100) m

/-- The initial 100-digit number -/
def initial_number : ℕ := 1234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890

/-- The theorem stating that the last remaining digit is 4 -/
theorem last_digit_is_four :
  ∃ k, remove_odd_positions initial_number k = 4 ∧ 
       ∀ m > k, remove_odd_positions initial_number m = 0 :=
sorry

end NUMINAMATH_CALUDE_last_digit_is_four_l1922_192230


namespace NUMINAMATH_CALUDE_airplane_distance_theorem_l1922_192209

/-- Calculates the distance traveled by an airplane given its speed and time. -/
def airplane_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that an airplane flying for 38 hours at 30 miles per hour travels 1140 miles. -/
theorem airplane_distance_theorem :
  let speed : ℝ := 30
  let time : ℝ := 38
  airplane_distance speed time = 1140 := by
  sorry

end NUMINAMATH_CALUDE_airplane_distance_theorem_l1922_192209


namespace NUMINAMATH_CALUDE_apple_ratio_proof_l1922_192255

theorem apple_ratio_proof (red_apples green_apples : ℕ) : 
  red_apples = 32 →
  red_apples + green_apples = 44 →
  (red_apples : ℚ) / green_apples = 8 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_ratio_proof_l1922_192255


namespace NUMINAMATH_CALUDE_largest_integer_divisibility_l1922_192252

theorem largest_integer_divisibility : ∃ (n : ℕ), n = 1956 ∧ 
  (∀ m : ℕ, m > n → ¬(∃ k : ℤ, (m^2 - 2012 : ℤ) = k * (m + 7))) ∧
  (∃ k : ℤ, (n^2 - 2012 : ℤ) = k * (n + 7)) :=
sorry

end NUMINAMATH_CALUDE_largest_integer_divisibility_l1922_192252


namespace NUMINAMATH_CALUDE_parabola_with_vertex_and_focus_parabola_through_point_l1922_192267

-- Define a parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Theorem 1
theorem parabola_with_vertex_and_focus
  (p : Parabola)
  (vertex : Point)
  (focus : Point)
  (h1 : vertex.x = 0 ∧ vertex.y = 0)
  (h2 : focus.x = 6 ∧ focus.y = 0) :
  p.equation = fun x y ↦ y^2 = 24*x :=
sorry

-- Theorem 2
theorem parabola_through_point
  (p : Parabola)
  (point : Point)
  (h : point.x = 1 ∧ point.y = 2) :
  (p.equation = fun x y ↦ x^2 = (1/2)*y) ∨
  (p.equation = fun x y ↦ y^2 = 4*x) :=
sorry

end NUMINAMATH_CALUDE_parabola_with_vertex_and_focus_parabola_through_point_l1922_192267


namespace NUMINAMATH_CALUDE_min_value_x_plus_81_over_x_l1922_192286

theorem min_value_x_plus_81_over_x (x : ℝ) (h : x > 0) : 
  x + 81 / x ≥ 18 ∧ ∃ y > 0, y + 81 / y = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_81_over_x_l1922_192286


namespace NUMINAMATH_CALUDE_subcommittee_formation_count_l1922_192251

def number_of_ways_to_form_subcommittee (total_republicans : ℕ) (total_democrats : ℕ) (subcommittee_republicans : ℕ) (subcommittee_democrats : ℕ) : ℕ :=
  Nat.choose total_republicans subcommittee_republicans * Nat.choose total_democrats subcommittee_democrats

theorem subcommittee_formation_count :
  number_of_ways_to_form_subcommittee 10 7 4 3 = 7350 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_formation_count_l1922_192251


namespace NUMINAMATH_CALUDE_average_of_solutions_is_zero_l1922_192295

theorem average_of_solutions_is_zero :
  let f : ℝ → ℝ := fun x => Real.sqrt (3 * x^2 + 4)
  let solutions := {x : ℝ | f x = Real.sqrt 28}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ (x : ℝ), x ∈ solutions → (x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_average_of_solutions_is_zero_l1922_192295


namespace NUMINAMATH_CALUDE_series_solution_l1922_192276

/-- The sum of the infinite geometric series with first term a and common ratio r -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The given series as a function of k -/
noncomputable def givenSeries (k : ℝ) : ℝ :=
  4 + geometricSum ((4 + k) / 5) (1 / 5)

theorem series_solution :
  ∃ k : ℝ, givenSeries k = 10 ∧ k = 16 := by sorry

end NUMINAMATH_CALUDE_series_solution_l1922_192276


namespace NUMINAMATH_CALUDE_locus_is_circle_l1922_192221

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
structure PointOnCircle (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

-- Define the locus of points
def Locus (c : Circle) (B C : PointOnCircle c) : Set (ℝ × ℝ) :=
  { M | ∃ (A : PointOnCircle c),
    let K := ((A.point.1 + B.point.1) / 2, (A.point.2 + B.point.2) / 2)
    M ∈ { P | (P.1 - A.point.1) * (C.point.1 - A.point.1) + (P.2 - A.point.2) * (C.point.2 - A.point.2) = 0 } ∧
    (K.1 - M.1) * (C.point.1 - A.point.1) + (K.2 - M.2) * (C.point.2 - A.point.2) = 0 }

-- Theorem statement
theorem locus_is_circle (c : Circle) (B C : PointOnCircle c) :
  ∃ (c' : Circle), Locus c B C = { P | (P.1 - c'.center.1)^2 + (P.2 - c'.center.2)^2 = c'.radius^2 } ∧
  B.point ∈ Locus c B C ∧ C.point ∈ Locus c B C :=
sorry

end NUMINAMATH_CALUDE_locus_is_circle_l1922_192221


namespace NUMINAMATH_CALUDE_total_grading_time_l1922_192231

def math_worksheets : ℕ := 45
def science_worksheets : ℕ := 37
def history_worksheets : ℕ := 32

def math_grading_time : ℕ := 15
def science_grading_time : ℕ := 20
def history_grading_time : ℕ := 25

theorem total_grading_time :
  math_worksheets * math_grading_time +
  science_worksheets * science_grading_time +
  history_worksheets * history_grading_time = 2215 := by
sorry

end NUMINAMATH_CALUDE_total_grading_time_l1922_192231


namespace NUMINAMATH_CALUDE_carol_wins_probability_l1922_192202

/-- The probability of getting a six on a single die toss -/
def prob_six : ℚ := 1 / 6

/-- The probability of not getting a six on a single die toss -/
def prob_not_six : ℚ := 1 - prob_six

/-- The number of players before Carol in the sequence -/
def players_before_carol : ℕ := 2

/-- The total number of players in the sequence -/
def total_players : ℕ := 4

/-- The probability that Carol wins on her first turn in any cycle -/
def prob_carol_wins_first_turn : ℚ := prob_not_six ^ players_before_carol * prob_six

/-- The probability that no one wins in a full cycle -/
def prob_no_win_cycle : ℚ := prob_not_six ^ total_players

/-- Theorem: The probability that Carol is the first to toss a six is 25/91 -/
theorem carol_wins_probability :
  prob_carol_wins_first_turn / (1 - prob_no_win_cycle) = 25 / 91 := by
  sorry

end NUMINAMATH_CALUDE_carol_wins_probability_l1922_192202


namespace NUMINAMATH_CALUDE_quadratic_roots_irrational_l1922_192226

theorem quadratic_roots_irrational (k : ℝ) (h1 : k^2 = 16/3) (h2 : ∀ x, x^2 - 5*k*x + 3*k^2 = 0 → ∃ y, x^2 - 5*k*x + 3*k^2 = 0 ∧ x * y = 16) :
  ∃ x y : ℝ, x^2 - 5*k*x + 3*k^2 = 0 ∧ y^2 - 5*k*y + 3*k^2 = 0 ∧ x * y = 16 ∧ (¬ ∃ m n : ℤ, x = m / n ∨ y = m / n) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_irrational_l1922_192226


namespace NUMINAMATH_CALUDE_parallelogram_area_l1922_192284

def v : Fin 2 → ℝ := ![6, -4]
def w : Fin 2 → ℝ := ![13, -1]

theorem parallelogram_area : 
  abs (Matrix.det !![v 0, v 1; w 0, w 1]) = 46 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1922_192284


namespace NUMINAMATH_CALUDE_product_of_powers_of_ten_l1922_192294

theorem product_of_powers_of_ten : (10^0.4) * (10^0.6) * (10^0.3) * (10^0.2) * (10^0.5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_of_ten_l1922_192294


namespace NUMINAMATH_CALUDE_range_sin_plus_cos_range_sin_plus_cos_minus_sin_2x_l1922_192204

-- Part 1
theorem range_sin_plus_cos :
  Set.range (fun x : ℝ => Real.sin x + Real.cos x) = Set.Icc (-Real.sqrt 2) (Real.sqrt 2) := by
sorry

-- Part 2
theorem range_sin_plus_cos_minus_sin_2x :
  Set.range (fun x : ℝ => Real.sin x + Real.cos x - Real.sin (2 * x)) = Set.Icc (-1 - Real.sqrt 2) (5/4) := by
sorry

end NUMINAMATH_CALUDE_range_sin_plus_cos_range_sin_plus_cos_minus_sin_2x_l1922_192204


namespace NUMINAMATH_CALUDE_min_both_mozart_bach_l1922_192244

theorem min_both_mozart_bach (total : ℕ) (mozart_fans : ℕ) (bach_fans : ℕ)
  (h1 : total = 150)
  (h2 : mozart_fans = 120)
  (h3 : bach_fans = 110)
  : ∃ (both : ℕ), both ≥ 80 ∧ 
    both ≤ mozart_fans ∧ 
    both ≤ bach_fans ∧ 
    ∀ (x : ℕ), x < both → 
      (mozart_fans - x) + (bach_fans - x) > total := by
  sorry

end NUMINAMATH_CALUDE_min_both_mozart_bach_l1922_192244


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1922_192282

theorem purely_imaginary_complex_number (i : ℂ) (a : ℝ) : 
  i * i = -1 → 
  (∃ (k : ℝ), (1 + a * i) / (2 - i) = k * i) → 
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1922_192282


namespace NUMINAMATH_CALUDE_xy_value_from_absolute_sum_l1922_192298

theorem xy_value_from_absolute_sum (x y : ℝ) :
  |x - 5| + |y + 3| = 0 → x * y = -15 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_from_absolute_sum_l1922_192298


namespace NUMINAMATH_CALUDE_range_of_f_l1922_192229

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 4*x + 6

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = { y | y ≥ 2 } := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l1922_192229


namespace NUMINAMATH_CALUDE_even_composite_ratio_l1922_192273

def first_five_even_composites : List Nat := [4, 6, 8, 10, 12]
def next_five_even_composites : List Nat := [14, 16, 18, 20, 22]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem even_composite_ratio :
  (product_of_list first_five_even_composites) / 
  (product_of_list next_five_even_composites) = 1 / 42 := by
  sorry

end NUMINAMATH_CALUDE_even_composite_ratio_l1922_192273


namespace NUMINAMATH_CALUDE_min_total_books_l1922_192296

/-- Represents the number of books for each subject in the library. -/
structure LibraryBooks where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  mathematics : ℕ
  history : ℕ

/-- Defines the conditions for the library books problem. -/
def LibraryBooksProblem (books : LibraryBooks) : Prop :=
  books.physics * 2 = books.chemistry * 3 ∧
  books.chemistry * 3 = books.biology * 4 ∧
  books.biology * 6 = books.mathematics * 5 ∧
  books.mathematics * 8 = books.history * 7 ∧
  books.mathematics ≥ 1000 ∧
  books.physics + books.chemistry + books.biology + books.mathematics + books.history > 10000

/-- Theorem stating the minimum possible total number of books in the library. -/
theorem min_total_books (books : LibraryBooks) (h : LibraryBooksProblem books) :
  books.physics + books.chemistry + books.biology + books.mathematics + books.history = 10050 :=
by
  sorry


end NUMINAMATH_CALUDE_min_total_books_l1922_192296


namespace NUMINAMATH_CALUDE_circle_tangent_perpendicular_l1922_192260

-- Define the types for our geometric objects
variable (Point Circle Line : Type)

-- Define the necessary operations and relations
variable (radius : Circle → ℝ)
variable (intersect : Circle → Circle → Set Point)
variable (tangent_point : Circle → Line → Point)
variable (line_through : Point → Point → Line)
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem circle_tangent_perpendicular 
  (Γ Γ' : Circle) 
  (A B C D : Point) 
  (t : Line) :
  radius Γ = radius Γ' →
  A ∈ intersect Γ Γ' →
  B ∈ intersect Γ Γ' →
  C = tangent_point Γ t →
  D = tangent_point Γ' t →
  perpendicular (line_through A C) (line_through B D) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_perpendicular_l1922_192260


namespace NUMINAMATH_CALUDE_overlap_area_is_one_l1922_192278

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by its vertices -/
structure Square where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- A triangle defined by its vertices -/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Calculate the area of overlap between a square and a triangle -/
def areaOfOverlap (s : Square) (t : Triangle) : ℝ := sorry

/-- The main theorem stating that the area of overlap is 1 square unit -/
theorem overlap_area_is_one :
  let s := Square.mk
    (Point.mk 0 0)
    (Point.mk 0 2)
    (Point.mk 2 2)
    (Point.mk 2 0)
  let t := Triangle.mk
    (Point.mk 2 2)
    (Point.mk 0 1)
    (Point.mk 1 0)
  areaOfOverlap s t = 1 := by sorry

end NUMINAMATH_CALUDE_overlap_area_is_one_l1922_192278


namespace NUMINAMATH_CALUDE_minimal_area_parallelepiped_l1922_192201

/-- A right parallelepiped with integer side lengths -/
structure RightParallelepiped where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- The volume of a right parallelepiped -/
def volume (p : RightParallelepiped) : ℕ :=
  p.a * p.b * p.c

/-- The surface area of a right parallelepiped -/
def surfaceArea (p : RightParallelepiped) : ℕ :=
  2 * (p.a * p.b + p.b * p.c + p.c * p.a)

/-- The set of all right parallelepipeds with volume > 1000 -/
def validParallelepipeds : Set RightParallelepiped :=
  {p : RightParallelepiped | volume p > 1000}

theorem minimal_area_parallelepiped :
  ∃ (p : RightParallelepiped),
    p ∈ validParallelepipeds ∧
    p.a = 7 ∧ p.b = 12 ∧ p.c = 12 ∧
    ∀ (q : RightParallelepiped),
      q ∈ validParallelepipeds →
      surfaceArea p ≤ surfaceArea q :=
sorry

end NUMINAMATH_CALUDE_minimal_area_parallelepiped_l1922_192201


namespace NUMINAMATH_CALUDE_piggy_bank_savings_l1922_192233

-- Define the initial amount in the piggy bank
def initial_amount : ℕ := 200

-- Define the cost per store trip
def cost_per_trip : ℕ := 2

-- Define the number of trips per month
def trips_per_month : ℕ := 4

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Define the function to calculate the remaining amount
def remaining_amount : ℕ :=
  initial_amount - (cost_per_trip * trips_per_month * months_in_year)

-- Theorem to prove
theorem piggy_bank_savings : remaining_amount = 104 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_savings_l1922_192233


namespace NUMINAMATH_CALUDE_quadratic_sum_and_reciprocal_l1922_192247

theorem quadratic_sum_and_reciprocal (t : ℝ) (h : t^2 - 3*t + 1 = 0) : t + 1/t = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_and_reciprocal_l1922_192247


namespace NUMINAMATH_CALUDE_triangle_area_inequality_l1922_192274

/-- Given two triangles with sides a₁ ≤ b₁ ≤ c and a₂ ≤ b₂ ≤ c, and equal smallest angles α,
    the area of a triangle with sides (a₁ + a₂), (b₁ + b₂), and (c + c) is no less than
    twice the sum of the areas of the original triangles. -/
theorem triangle_area_inequality
  (a₁ b₁ c a₂ b₂ : ℝ) (α : ℝ)
  (h₁ : 0 < a₁ ∧ 0 < b₁ ∧ 0 < c)
  (h₂ : 0 < a₂ ∧ 0 < b₂)
  (h₃ : a₁ ≤ b₁ ∧ b₁ ≤ c)
  (h₄ : a₂ ≤ b₂ ∧ b₂ ≤ c)
  (h₅ : 0 < α ∧ α < π)
  (area₁ : ℝ := (1/2) * b₁ * c * Real.sin α)
  (area₂ : ℝ := (1/2) * b₂ * c * Real.sin α)
  (new_area : ℝ := (1/2) * (b₁ + b₂) * (2*c) * Real.sin (min α π/2)) :
  new_area ≥ 2 * (area₁ + area₂) :=
by sorry


end NUMINAMATH_CALUDE_triangle_area_inequality_l1922_192274


namespace NUMINAMATH_CALUDE_stating_min_pieces_for_equal_division_l1922_192213

/-- Represents the number of pieces a pie is cut into -/
def NumPieces : ℕ := 11

/-- Represents the first group size -/
def GroupSize1 : ℕ := 5

/-- Represents the second group size -/
def GroupSize2 : ℕ := 7

/-- 
Theorem stating that NumPieces is the minimum number of pieces 
that allows equal division among GroupSize1 or GroupSize2 people 
-/
theorem min_pieces_for_equal_division :
  (∃ (k : ℕ), k * GroupSize1 = NumPieces) ∧ 
  (∃ (k : ℕ), k * GroupSize2 = NumPieces) ∧
  (∀ (n : ℕ), n < NumPieces → 
    (¬∃ (k : ℕ), k * GroupSize1 = n) ∨ 
    (¬∃ (k : ℕ), k * GroupSize2 = n)) :=
sorry

end NUMINAMATH_CALUDE_stating_min_pieces_for_equal_division_l1922_192213


namespace NUMINAMATH_CALUDE_divided_square_longer_side_l1922_192299

/-- Represents a square divided into a trapezoid and hexagon -/
structure DividedSquare where
  side_length : ℝ
  trapezoid_area : ℝ
  hexagon_area : ℝ
  longer_parallel_side : ℝ

/-- The properties of our specific divided square -/
def my_square : DividedSquare where
  side_length := 2
  trapezoid_area := 2
  hexagon_area := 2
  longer_parallel_side := 2  -- This is what we want to prove

theorem divided_square_longer_side (s : DividedSquare) 
  (h1 : s.side_length = 2)
  (h2 : s.trapezoid_area = s.hexagon_area)
  (h3 : s.trapezoid_area + s.hexagon_area = s.side_length ^ 2)
  (h4 : s.trapezoid_area = (s.longer_parallel_side + s.side_length) * (s.side_length / 2) / 2) :
  s.longer_parallel_side = 2 := by
  sorry

#check divided_square_longer_side my_square

end NUMINAMATH_CALUDE_divided_square_longer_side_l1922_192299


namespace NUMINAMATH_CALUDE_rectangle_perimeter_problem_l1922_192262

/-- Given a rectangle A with perimeter 32 cm and length twice its width,
    and a square B with area one-third of rectangle A's area,
    prove that the perimeter of square B is 64√3/9 cm. -/
theorem rectangle_perimeter_problem (width_A : ℝ) (length_A : ℝ) (side_B : ℝ) :
  width_A > 0 →
  length_A = 2 * width_A →
  2 * (length_A + width_A) = 32 →
  side_B^2 = (1/3) * (length_A * width_A) →
  4 * side_B = (64 * Real.sqrt 3) / 9 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_problem_l1922_192262


namespace NUMINAMATH_CALUDE_max_profit_price_l1922_192220

/-- Represents the sales volume as a function of unit price -/
def sales_volume (x : ℝ) : ℝ := -2 * x + 100

/-- Represents the profit as a function of unit price -/
def profit (x : ℝ) : ℝ := (x - 20) * (sales_volume x)

/-- Theorem: The unit price that maximizes profit is 35 yuan -/
theorem max_profit_price : 
  ∃ (x : ℝ), x = 35 ∧ ∀ (y : ℝ), profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_price_l1922_192220


namespace NUMINAMATH_CALUDE_students_in_same_group_l1922_192288

/-- The number of interest groups -/
def num_groups : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 2

/-- The probability of a student joining any specific group -/
def prob_join_group : ℚ := 1 / num_groups

/-- The probability of both students being in the same group -/
def prob_same_group : ℚ := 1 / num_groups

theorem students_in_same_group : 
  prob_same_group = 1 / num_groups :=
sorry

end NUMINAMATH_CALUDE_students_in_same_group_l1922_192288


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l1922_192216

-- Define the function
def f (x : ℝ) : ℝ := 2 * x + 3

-- State the theorem
theorem f_satisfies_conditions :
  (∃ x y, x > 0 ∧ y > 0 ∧ f x = y) ∧  -- Passes through first quadrant
  (∃ x y, x < 0 ∧ y > 0 ∧ f x = y) ∧  -- Passes through second quadrant
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂)  -- Increasing in first quadrant
  := by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l1922_192216


namespace NUMINAMATH_CALUDE_max_quantity_a_theorem_l1922_192214

/-- Represents the prices and quantities of fertilizers A and B -/
structure Fertilizers where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℝ
  quantity_b : ℝ

/-- Conditions for the fertilizer problem -/
def fertilizer_conditions (f : Fertilizers) : Prop :=
  f.price_a = f.price_b + 100 ∧
  2 * f.price_a + f.price_b = 1700 ∧
  f.quantity_a + f.quantity_b = 10 ∧
  f.quantity_a * f.price_a + f.quantity_b * f.price_b ≤ 5600

/-- The maximum quantity of fertilizer A that can be purchased -/
def max_quantity_a (f : Fertilizers) : ℝ := 6

/-- Theorem stating the maximum quantity of fertilizer A that can be purchased -/
theorem max_quantity_a_theorem (f : Fertilizers) :
  fertilizer_conditions f → f.quantity_a ≤ max_quantity_a f := by
  sorry

end NUMINAMATH_CALUDE_max_quantity_a_theorem_l1922_192214


namespace NUMINAMATH_CALUDE_ratio_twenty_ten_l1922_192234

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  prop1 : a 2 * a 6 = 16
  prop2 : a 4 + a 8 = 8

/-- The ratio of the 20th term to the 10th term is 1 -/
theorem ratio_twenty_ten (seq : GeometricSequence) : seq.a 20 / seq.a 10 = 1 := by
  sorry

#check ratio_twenty_ten

end NUMINAMATH_CALUDE_ratio_twenty_ten_l1922_192234


namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_thrice_exterior_polygon_sides_when_interior_thrice_exterior_proof_l1922_192261

theorem polygon_sides_when_interior_thrice_exterior : ℕ → Prop :=
  fun n =>
    (180 * (n - 2) = 3 * 360) →
    n = 8

-- The proof is omitted
theorem polygon_sides_when_interior_thrice_exterior_proof :
  polygon_sides_when_interior_thrice_exterior 8 :=
sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_thrice_exterior_polygon_sides_when_interior_thrice_exterior_proof_l1922_192261


namespace NUMINAMATH_CALUDE_expression_equality_l1922_192270

theorem expression_equality : 3 * Real.sqrt 2 + |1 - Real.sqrt 2| + (8 : ℝ) ^ (1/3) = 4 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1922_192270


namespace NUMINAMATH_CALUDE_probability_spade_face_diamond_l1922_192206

/-- Represents a standard 52-card deck --/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (h : cards.card = 52)

/-- Represents a suit in a deck of cards --/
inductive Suit
| Spade | Heart | Diamond | Club

/-- Represents a rank in a deck of cards --/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Checks if a rank is a face card --/
def is_face_card (r : Rank) : Bool :=
  match r with
  | Rank.Jack | Rank.Queen | Rank.King => true
  | _ => false

/-- Calculates the probability of drawing three specific cards --/
def probability_three_cards (d : Deck) (first : Suit) (second : Rank → Bool) (third : Suit) : ℚ :=
  sorry

/-- Theorem stating the probability of drawing a spade, then a face card, then a diamond --/
theorem probability_spade_face_diamond (d : Deck) :
  probability_three_cards d Suit.Spade is_face_card Suit.Diamond = 1911 / 132600 :=
sorry

end NUMINAMATH_CALUDE_probability_spade_face_diamond_l1922_192206


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l1922_192287

theorem complex_product_magnitude (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 3 * Complex.I →
  x = 6 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l1922_192287


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_rope_l1922_192232

/-- Represents the sides of an isosceles triangle --/
structure IsoscelesTriangle where
  short : ℝ
  long : ℝ
  isIsosceles : long = 2 * short

/-- Checks if the given sides form a valid triangle --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem to be proved --/
theorem isosceles_triangle_from_rope (t : IsoscelesTriangle) :
  t.short + t.long + t.long = 20 →
  is_valid_triangle t.short t.long t.long →
  t.short = 4 ∧ t.long = 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_rope_l1922_192232


namespace NUMINAMATH_CALUDE_min_additional_cells_for_symmetry_l1922_192238

/-- Represents a cell in the rectangle --/
structure Cell where
  x : ℤ
  y : ℤ

/-- Represents the rectangle --/
structure Rectangle where
  width : ℕ
  height : ℕ
  center : Cell

/-- The set of initially colored cells --/
def initialColoredCells : Finset Cell := sorry

/-- Function to determine if two cells are symmetric about the center --/
def isSymmetric (c1 c2 : Cell) (center : Cell) : Prop := sorry

/-- Function to count the number of additional cells needed for symmetry --/
def additionalCellsForSymmetry (rect : Rectangle) (initial : Finset Cell) : ℕ := sorry

/-- Theorem stating that the minimum number of additional cells to color is 7 --/
theorem min_additional_cells_for_symmetry (rect : Rectangle) : 
  additionalCellsForSymmetry rect initialColoredCells = 7 := by sorry

end NUMINAMATH_CALUDE_min_additional_cells_for_symmetry_l1922_192238


namespace NUMINAMATH_CALUDE_equation_proof_l1922_192224

theorem equation_proof : 300 * 2 + (12 + 4) * 1 / 8 = 602 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1922_192224


namespace NUMINAMATH_CALUDE_inequality_proof_l1922_192211

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/3) :
  (1 - a) * (1 - b) ≤ 25/36 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1922_192211


namespace NUMINAMATH_CALUDE_find_M_l1922_192290

theorem find_M (x y z M : ℚ) 
  (sum_eq : x + y + z = 120)
  (x_dec : x - 10 = M)
  (y_inc : y + 10 = M)
  (z_mul : 10 * z = M) :
  M = 400 / 7 := by
  sorry

end NUMINAMATH_CALUDE_find_M_l1922_192290


namespace NUMINAMATH_CALUDE_surface_area_of_T_l1922_192242

-- Define the cube
structure Cube where
  edge_length : ℝ
  vertex_A : ℝ × ℝ × ℝ

-- Define points on the cube
def L (c : Cube) : ℝ × ℝ × ℝ := (3, 0, 0)
def M (c : Cube) : ℝ × ℝ × ℝ := (0, 3, 0)
def N (c : Cube) : ℝ × ℝ × ℝ := (0, 0, 3)
def P (c : Cube) : ℝ × ℝ × ℝ := (c.edge_length, c.edge_length, c.edge_length)

-- Define the solid T
structure SolidT (c : Cube) where
  tunnel_sides : Set (ℝ × ℝ × ℝ)

-- Define the surface area of T
def surface_area (t : SolidT c) : ℝ := sorry

-- Theorem statement
theorem surface_area_of_T (c : Cube) (t : SolidT c) :
  c.edge_length = 10 →
  surface_area t = 582 + 9 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_T_l1922_192242


namespace NUMINAMATH_CALUDE_garden_volume_l1922_192222

/-- Calculates the volume of a rectangular prism -/
def rectangularPrismVolume (length width height : ℝ) : ℝ :=
  length * width * height

/-- Theorem: The volume of a rectangular prism garden with dimensions 12 m, 5 m, and 3 m is 180 cubic meters -/
theorem garden_volume :
  rectangularPrismVolume 12 5 3 = 180 := by
  sorry

end NUMINAMATH_CALUDE_garden_volume_l1922_192222


namespace NUMINAMATH_CALUDE_ratio_of_numbers_with_special_average_l1922_192279

theorem ratio_of_numbers_with_special_average (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (h4 : (a + b) / 2 = a - b) : a / b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_with_special_average_l1922_192279


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1922_192263

theorem smallest_solution_of_equation : 
  ∃ x : ℝ, x^4 - 50*x^2 + 625 = 0 ∧ 
  (∀ y : ℝ, y^4 - 50*y^2 + 625 = 0 → x ≤ y) ∧ 
  x = -5 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1922_192263


namespace NUMINAMATH_CALUDE_expression_value_l1922_192217

theorem expression_value : (3^4 * 5^2 * 7^3 * 11) / (7 * 11^2) = 9025 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1922_192217


namespace NUMINAMATH_CALUDE_sale_price_increase_l1922_192254

theorem sale_price_increase (regular_price : ℝ) (regular_price_positive : regular_price > 0) : 
  let sale_price := regular_price * (1 - 0.2)
  let price_increase := regular_price - sale_price
  let percent_increase := (price_increase / sale_price) * 100
  percent_increase = 25 := by
sorry

end NUMINAMATH_CALUDE_sale_price_increase_l1922_192254


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l1922_192227

theorem polynomial_root_implies_coefficients : ∀ (c d : ℝ),
  (∃ (x : ℂ), x^3 + c*x^2 + 2*x + d = 0 ∧ x = Complex.mk 2 (-3)) →
  c = 5/4 ∧ d = -143/4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l1922_192227


namespace NUMINAMATH_CALUDE_inequalities_count_l1922_192240

theorem inequalities_count (a c : ℝ) (h : a * c < 0) :
  ∃! n : ℕ, n = (Bool.toNat (a / c < 0) +
                 Bool.toNat (a * c^2 < 0) +
                 Bool.toNat (a^2 * c < 0) +
                 Bool.toNat (c^3 * a < 0) +
                 Bool.toNat (c * a^3 < 0)) ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_count_l1922_192240


namespace NUMINAMATH_CALUDE_square_sum_geq_product_sum_l1922_192203

theorem square_sum_geq_product_sum (a b : ℝ) : a^2 + b^2 ≥ a*b + a + b - 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_product_sum_l1922_192203


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_all_nines_l1922_192236

def is_all_nines (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10^k - 1

theorem smallest_multiple_of_seven_all_nines :
  ∃ N : ℕ, (N = 142857 ∧
            is_all_nines (7 * N) ∧
            ∀ m : ℕ, m < N → ¬is_all_nines (7 * m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_all_nines_l1922_192236


namespace NUMINAMATH_CALUDE_not_necessarily_true_squared_l1922_192256

theorem not_necessarily_true_squared (a b : ℝ) (h : a < b) : 
  ∃ (a b : ℝ), a < b ∧ a^2 ≥ b^2 := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_true_squared_l1922_192256


namespace NUMINAMATH_CALUDE_remainder_problem_l1922_192289

theorem remainder_problem (k : ℕ+) (h : ∃ a : ℕ, 120 = a * k ^ 2 + 12) :
  ∃ b : ℕ, 144 = b * k + 0 :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l1922_192289


namespace NUMINAMATH_CALUDE_train_passing_time_l1922_192245

/-- Calculates the time taken for a train to pass a man moving in the opposite direction. -/
theorem train_passing_time (train_length : Real) (train_speed : Real) (man_speed : Real) :
  train_length = 200 →
  train_speed = 80 →
  man_speed = 10 →
  (train_length / ((train_speed + man_speed) * 1000 / 3600)) = 8 := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l1922_192245


namespace NUMINAMATH_CALUDE_boys_count_in_class_l1922_192264

theorem boys_count_in_class (total : ℕ) (boy_ratio girl_ratio : ℕ) (h1 : total = 49) (h2 : boy_ratio = 4) (h3 : girl_ratio = 3) :
  (total * boy_ratio) / (boy_ratio + girl_ratio) = 28 := by
  sorry

end NUMINAMATH_CALUDE_boys_count_in_class_l1922_192264


namespace NUMINAMATH_CALUDE_sum_of_consecutive_integers_l1922_192283

theorem sum_of_consecutive_integers :
  let start : Int := -9
  let count : Nat := 20
  let sequence := List.range count |>.map (λ i => start + i)
  sequence.sum = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_integers_l1922_192283


namespace NUMINAMATH_CALUDE_triangle_area_l1922_192243

theorem triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) : 
  let S := (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)
  S = 84 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1922_192243


namespace NUMINAMATH_CALUDE_tree_planting_event_percentage_l1922_192285

theorem tree_planting_event_percentage (boys : ℕ) (girls : ℕ) : 
  boys = 600 →
  girls = boys + 400 →
  (960 : ℚ) / (boys + girls : ℚ) = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_event_percentage_l1922_192285


namespace NUMINAMATH_CALUDE_final_center_coordinates_l1922_192218

-- Define the initial center coordinates
def initial_center : ℝ × ℝ := (6, -5)

-- Define the reflection about y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Define the reflection over y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Define the composition of the two reflections
def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_axis (reflect_y_eq_x p)

-- Theorem statement
theorem final_center_coordinates :
  double_reflection initial_center = (5, 6) := by sorry

end NUMINAMATH_CALUDE_final_center_coordinates_l1922_192218


namespace NUMINAMATH_CALUDE_sixteen_greater_than_thirtytwo_l1922_192225

/-- Represents a domino placement on a board -/
structure DominoPlacement (n : ℕ) where
  placements : Fin n → Fin 8 × Fin 8 × Bool
  no_overlap : ∀ i j, i ≠ j → placements i ≠ placements j

/-- The number of ways to place n dominoes on an 8x8 board -/
def num_placements (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of 16-domino placements is greater than 32-domino placements -/
theorem sixteen_greater_than_thirtytwo :
  num_placements 16 > num_placements 32 := by sorry

end NUMINAMATH_CALUDE_sixteen_greater_than_thirtytwo_l1922_192225


namespace NUMINAMATH_CALUDE_specific_shiny_penny_last_probability_l1922_192293

/-- The number of shiny pennies in the box -/
def shiny_pennies : ℕ := 4

/-- The number of dull pennies in the box -/
def dull_pennies : ℕ := 4

/-- The total number of pennies in the box -/
def total_pennies : ℕ := shiny_pennies + dull_pennies

/-- The probability of drawing a specific shiny penny last -/
def prob_specific_shiny_last : ℚ := 1 / 2

theorem specific_shiny_penny_last_probability :
  prob_specific_shiny_last = (Nat.choose (total_pennies - 1) (shiny_pennies - 1)) / (Nat.choose total_pennies shiny_pennies) :=
by sorry

end NUMINAMATH_CALUDE_specific_shiny_penny_last_probability_l1922_192293


namespace NUMINAMATH_CALUDE_smallest_multiple_twenty_five_satisfies_smallest_x_is_25_l1922_192223

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 625 ∣ (450 * x) → x ≥ 25 := by
  sorry

theorem twenty_five_satisfies : 625 ∣ (450 * 25) := by
  sorry

theorem smallest_x_is_25 : ∃ x : ℕ, x > 0 ∧ 625 ∣ (450 * x) ∧ ∀ y : ℕ, (y > 0 ∧ 625 ∣ (450 * y)) → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_twenty_five_satisfies_smallest_x_is_25_l1922_192223


namespace NUMINAMATH_CALUDE_mr_green_garden_yield_l1922_192268

/-- Calculates the total expected yield from a rectangular garden -/
def gardenYield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) 
                (potato_yield : ℝ) (carrot_yield : ℝ) : ℝ :=
  let area := (length_steps : ℝ) * step_length * (width_steps : ℝ) * step_length
  area * (potato_yield + carrot_yield)

/-- Theorem stating the expected yield from Mr. Green's garden -/
theorem mr_green_garden_yield :
  gardenYield 20 25 2.5 0.5 0.25 = 2343.75 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_garden_yield_l1922_192268


namespace NUMINAMATH_CALUDE_compound_propositions_true_l1922_192292

-- Define proposition P
def P : Prop := ∀ x y : ℝ, x > y → -x > -y

-- Define proposition Q
def Q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Theorem to prove
theorem compound_propositions_true : (¬P ∨ ¬Q) ∧ ((¬P) ∨ Q) := by
  sorry

end NUMINAMATH_CALUDE_compound_propositions_true_l1922_192292


namespace NUMINAMATH_CALUDE_chord_equation_through_bisection_point_l1922_192291

/-- Given a parabola y² = 6x and a chord passing through point P(4, 1) that is bisected at P,
    prove that the equation of the line l on which this chord lies is 3x - y - 11 = 0. -/
theorem chord_equation_through_bisection_point (x y : ℝ) :
  (∀ x y, y^2 = 6*x) →  -- Parabola equation
  (∃ x₁ y₁ x₂ y₂ : ℝ,   -- Existence of two points on the parabola
    y₁^2 = 6*x₁ ∧ y₂^2 = 6*x₂ ∧
    (4 = (x₁ + x₂) / 2) ∧ (1 = (y₁ + y₂) / 2)) →  -- P(4,1) is midpoint
  (3*x - y - 11 = 0) :=  -- Equation of the line
by sorry

end NUMINAMATH_CALUDE_chord_equation_through_bisection_point_l1922_192291


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l1922_192205

theorem divisibility_of_sum_of_powers (n : ℕ) : 13 ∣ (3^1974 + 2^1974) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l1922_192205


namespace NUMINAMATH_CALUDE_fundraiser_theorem_l1922_192210

def fundraiser (num_students : ℕ) (individual_cost : ℕ) (collective_cost : ℕ) 
                (day1_raised : ℕ) (day2_raised : ℕ) (day3_raised : ℕ) : ℕ :=
  let total_needed := num_students * individual_cost + collective_cost
  let first3days_raised := day1_raised + day2_raised + day3_raised
  let next4days_raised := first3days_raised / 2
  let total_raised := first3days_raised + next4days_raised
  let remaining := total_needed - total_raised
  remaining / num_students

theorem fundraiser_theorem : 
  fundraiser 6 450 3000 600 900 400 = 475 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_theorem_l1922_192210


namespace NUMINAMATH_CALUDE_min_value_theorem_l1922_192271

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (a + b) * (a + c) = 4) : 
  2 * a + b + c ≥ 4 ∧ (2 * a + b + c = 4 ↔ b = c) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1922_192271


namespace NUMINAMATH_CALUDE_min_cost_grass_seed_l1922_192277

/-- Represents a bag of grass seed -/
structure GrassSeedBag where
  weight : Nat
  price : Rat
  deriving Repr

/-- Calculates the total weight of a list of bags -/
def totalWeight (bags : List GrassSeedBag) : Nat :=
  bags.foldl (fun acc bag => acc + bag.weight) 0

/-- Calculates the total cost of a list of bags -/
def totalCost (bags : List GrassSeedBag) : Rat :=
  bags.foldl (fun acc bag => acc + bag.price) 0

/-- Checks if a list of bags satisfies the purchase conditions -/
def isValidPurchase (bags : List GrassSeedBag) : Prop :=
  totalWeight bags ≥ 65 ∧
  totalWeight bags ≤ 80 ∧
  bags.length ≤ 5 ∧
  bags.length ≥ 4 ∧
  (∃ b ∈ bags, b.weight = 5) ∧
  (∃ b ∈ bags, b.weight = 10) ∧
  (∃ b ∈ bags, b.weight = 25) ∧
  (∃ b ∈ bags, b.weight = 40)

theorem min_cost_grass_seed :
  let bags := [
    GrassSeedBag.mk 5 (13.85),
    GrassSeedBag.mk 10 (20.43),
    GrassSeedBag.mk 25 (32.20),
    GrassSeedBag.mk 40 (54.30)
  ]
  ∀ purchase : List GrassSeedBag,
    isValidPurchase purchase →
    totalCost purchase ≥ 120.78 :=
by sorry

end NUMINAMATH_CALUDE_min_cost_grass_seed_l1922_192277


namespace NUMINAMATH_CALUDE_square_of_105_l1922_192241

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by sorry

end NUMINAMATH_CALUDE_square_of_105_l1922_192241


namespace NUMINAMATH_CALUDE_solution_set_part1_a_upper_bound_l1922_192281

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 2 x ≥ 3} = {x : ℝ | x ≤ -1 ∨ x ≥ 3} :=
sorry

-- Part 2
theorem a_upper_bound (a : ℝ) :
  (∀ x ∈ Set.Ici 1, f a x ≥ -x^2 - 2) → a ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_a_upper_bound_l1922_192281


namespace NUMINAMATH_CALUDE_pushup_difference_l1922_192269

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 59

/-- The number of crunches Zachary did -/
def zachary_crunches : ℕ := 44

/-- The number of push-ups David did -/
def david_pushups : ℕ := 78

/-- The difference in crunches between Zachary and David -/
def crunch_difference : ℕ := 27

/-- Theorem stating the difference in push-ups between David and Zachary -/
theorem pushup_difference : david_pushups - zachary_pushups = 19 := by
  sorry

end NUMINAMATH_CALUDE_pushup_difference_l1922_192269


namespace NUMINAMATH_CALUDE_wage_decrease_compensation_l1922_192235

/-- Proves that a 25% increase in working hours maintains the same income after a 20% wage decrease --/
theorem wage_decrease_compensation (W H S : ℝ) (C : ℝ) (H_pos : H > 0) (W_pos : W > 0) :
  let original_income := W * H + C * S
  let new_wage := W * 0.8
  let new_hours := H * 1.25
  new_wage * new_hours + C * S = original_income := by
  sorry

end NUMINAMATH_CALUDE_wage_decrease_compensation_l1922_192235


namespace NUMINAMATH_CALUDE_painted_cubes_theorem_l1922_192212

/-- Represents the dimensions of a parallelepiped -/
structure Parallelepiped where
  m : ℕ
  n : ℕ
  k : ℕ
  h1 : 0 < k
  h2 : k ≤ n
  h3 : n ≤ m

/-- The set of possible numbers of painted cubes -/
def PaintedCubesCounts : Set ℕ := {60, 72, 84, 90, 120}

/-- 
  Given a parallelepiped where three faces sharing a common vertex are painted,
  if half of all cubes have at least one painted face, then the number of
  painted cubes is in the set PaintedCubesCounts
-/
theorem painted_cubes_theorem (p : Parallelepiped) :
  (p.m - 1) * (p.n - 1) * (p.k - 1) = p.m * p.n * p.k / 2 →
  (p.m * p.n * p.k - (p.m - 1) * (p.n - 1) * (p.k - 1)) ∈ PaintedCubesCounts := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_theorem_l1922_192212


namespace NUMINAMATH_CALUDE_triangle_segment_equality_l1922_192208

theorem triangle_segment_equality (AB AC : ℝ) (m n : ℕ+) :
  AB = 33 →
  AC = 21 →
  ∃ (BC : ℝ), BC = m →
  ∃ (D E : ℝ × ℝ),
    D.1 + D.2 = AB ∧
    E.1 + E.2 = AC ∧
    D.1 = n ∧
    Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = n ∧
    E.2 = n →
  n = 11 ∨ n = 21 := by
sorry


end NUMINAMATH_CALUDE_triangle_segment_equality_l1922_192208


namespace NUMINAMATH_CALUDE_special_line_properties_l1922_192219

/-- A line passing through (2,3) with x-intercept twice the y-intercept -/
def special_line (x y : ℝ) : Prop := x + 2*y - 8 = 0

theorem special_line_properties :
  (special_line 2 3) ∧ 
  (∃ (a : ℝ), a ≠ 0 ∧ special_line (2*a) 0 ∧ special_line 0 a) :=
by sorry

end NUMINAMATH_CALUDE_special_line_properties_l1922_192219


namespace NUMINAMATH_CALUDE_five_digit_division_l1922_192259

/-- A five-digit number -/
def FiveDigitNumber (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A four-digit number -/
def FourDigitNumber (m : ℕ) : Prop :=
  1000 ≤ m ∧ m ≤ 9999

/-- m is formed by removing the middle digit of n -/
def MiddleDigitRemoved (n m : ℕ) : Prop :=
  FiveDigitNumber n ∧ FourDigitNumber m ∧
  ∃ (a b c d e : ℕ), n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
                     m = a * 1000 + b * 100 + d * 10 + e

theorem five_digit_division (n m : ℕ) :
  FiveDigitNumber n → MiddleDigitRemoved n m →
  (∃ k : ℕ, n = k * m) ↔ ∃ a : ℕ, 10 ≤ a ∧ a ≤ 99 ∧ n = a * 1000 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_division_l1922_192259


namespace NUMINAMATH_CALUDE_horse_and_saddle_value_l1922_192249

/-- The total value of a horse and saddle is $100, given that the horse is worth 7 times as much as the saddle, and the saddle is worth $12.5. -/
theorem horse_and_saddle_value :
  let saddle_value : ℝ := 12.5
  let horse_value : ℝ := 7 * saddle_value
  horse_value + saddle_value = 100 := by
  sorry

end NUMINAMATH_CALUDE_horse_and_saddle_value_l1922_192249


namespace NUMINAMATH_CALUDE_role_assignment_count_l1922_192258

/-- The number of ways to assign roles in a play. -/
def assign_roles (num_men num_women : ℕ) : ℕ :=
  let male_role_assignments := num_men
  let female_role_assignments := num_women * (num_women - 1)
  let specific_role_assignment := 1
  let remaining_actors := (num_men - 1) + num_women
  let remaining_role_assignments := remaining_actors * (remaining_actors - 1)
  male_role_assignments * female_role_assignments * specific_role_assignment * remaining_role_assignments

/-- Theorem stating the number of ways to assign roles in the given scenario. -/
theorem role_assignment_count :
  assign_roles 6 7 = 27720 :=
by sorry

end NUMINAMATH_CALUDE_role_assignment_count_l1922_192258


namespace NUMINAMATH_CALUDE_set_formation_criterion_l1922_192200

-- Define a type for objects
variable {α : Type}

-- Define a predicate for well-defined and specific objects
variable (is_well_defined : α → Prop)

-- Define a predicate for collections that can form sets
def can_form_set (collection : Set α) : Prop :=
  ∀ x ∈ collection, is_well_defined x

-- Theorem statement
theorem set_formation_criterion (collection : Set α) :
  can_form_set is_well_defined collection ↔ ∀ x ∈ collection, is_well_defined x :=
sorry

end NUMINAMATH_CALUDE_set_formation_criterion_l1922_192200
