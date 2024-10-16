import Mathlib

namespace NUMINAMATH_CALUDE_stick_pieces_l988_98854

theorem stick_pieces (n₁ n₂ : ℕ) (h₁ : n₁ = 12) (h₂ : n₂ = 18) : 
  (n₁ - 1) + (n₂ - 1) - (n₁.lcm n₂ / n₁.gcd n₂ - 1) + 1 = 24 := by sorry

end NUMINAMATH_CALUDE_stick_pieces_l988_98854


namespace NUMINAMATH_CALUDE_max_m_value_inequality_proof_l988_98800

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x + 2|

-- Theorem for part (1)
theorem max_m_value (M : ℝ) : (∀ x, f x ≥ |M + 1|) → M ≤ 4 :=
sorry

-- Theorem for part (2)
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + 2*b + c = 4) : 1 / (a + b) + 1 / (b + c) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_inequality_proof_l988_98800


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l988_98820

theorem rational_inequality_solution (x : ℝ) :
  (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6)) > 0 ↔ 
  x < 2 ∨ (4 < x ∧ x < 5) ∨ 6 < x :=
by sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l988_98820


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l988_98830

theorem partial_fraction_decomposition :
  ∃ (A B : ℝ),
    (∀ x : ℝ, x ≠ 12 ∧ x ≠ -3 →
      (6 * x + 3) / (x^2 - 9 * x - 36) = A / (x - 12) + B / (x + 3)) ∧
    A = 5 ∧
    B = 1 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l988_98830


namespace NUMINAMATH_CALUDE_histogram_total_area_is_one_l988_98818

/-- A histogram representing a data distribution -/
structure Histogram where
  -- We don't need to define the internal structure of the histogram
  -- as we're only concerned with its total area property

/-- The total area of a histogram -/
def total_area (h : Histogram) : ℝ := sorry

/-- Theorem: The total area of a histogram representing a data distribution is equal to 1 -/
theorem histogram_total_area_is_one (h : Histogram) : total_area h = 1 := by
  sorry

end NUMINAMATH_CALUDE_histogram_total_area_is_one_l988_98818


namespace NUMINAMATH_CALUDE_factorization_theorem_l988_98869

theorem factorization_theorem (a : ℝ) : (a^2 + 4)^2 - 16*a^2 = (a + 2)^2 * (a - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l988_98869


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l988_98868

/-- An equilateral triangle with one vertex at the origin and the other two on the parabola x^2 = 2y has side length 4√3. -/
theorem equilateral_triangle_on_parabola :
  ∃ (a : ℝ) (v1 v2 : ℝ × ℝ),
    a > 0 ∧
    v1.1^2 = 2 * v1.2 ∧
    v2.1^2 = 2 * v2.2 ∧
    (v1.1 - 0)^2 + (v1.2 - 0)^2 = a^2 ∧
    (v2.1 - 0)^2 + (v2.2 - 0)^2 = a^2 ∧
    (v2.1 - v1.1)^2 + (v2.2 - v1.2)^2 = a^2 ∧
    a = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_parabola_l988_98868


namespace NUMINAMATH_CALUDE_jack_needs_more_money_l988_98896

def sock_price : ℝ := 9.50
def shoe_price : ℝ := 92
def jack_money : ℝ := 40
def num_socks : ℕ := 2

theorem jack_needs_more_money :
  let total_cost := num_socks * sock_price + shoe_price
  total_cost - jack_money = 71 := by sorry

end NUMINAMATH_CALUDE_jack_needs_more_money_l988_98896


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l988_98837

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (carbon_atoms hydrogen_atoms oxygen_atoms : ℕ) 
  (carbon_weight hydrogen_weight oxygen_weight : ℝ) : ℝ :=
  (carbon_atoms : ℝ) * carbon_weight + 
  (hydrogen_atoms : ℝ) * hydrogen_weight + 
  (oxygen_atoms : ℝ) * oxygen_weight

/-- The molecular weight of a compound with 4 Carbon atoms, 1 Hydrogen atom, and 1 Oxygen atom 
    is equal to 65.048 g/mol -/
theorem compound_molecular_weight : 
  molecular_weight 4 1 1 12.01 1.008 16.00 = 65.048 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l988_98837


namespace NUMINAMATH_CALUDE_sci_fi_section_pages_per_book_l988_98826

/-- Given a library section with a number of books and a total number of pages,
    calculate the number of pages per book. -/
def pages_per_book (num_books : ℕ) (total_pages : ℕ) : ℕ :=
  total_pages / num_books

/-- Theorem stating that in a library section with 8 books and 3824 total pages,
    each book has 478 pages. -/
theorem sci_fi_section_pages_per_book :
  pages_per_book 8 3824 = 478 := by
  sorry

end NUMINAMATH_CALUDE_sci_fi_section_pages_per_book_l988_98826


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l988_98853

theorem difference_of_squares_example (x y : ℤ) (hx : x = 12) (hy : y = 5) :
  (x - y) * (x + y) = 119 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l988_98853


namespace NUMINAMATH_CALUDE_negation_equivalence_l988_98887

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l988_98887


namespace NUMINAMATH_CALUDE_other_number_proof_l988_98802

theorem other_number_proof (a b : ℕ+) (h1 : Nat.lcm a b = 56) (h2 : Nat.gcd a b = 10) (h3 : a = 14) : b = 40 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l988_98802


namespace NUMINAMATH_CALUDE_trapezoid_area_l988_98893

/-- The area of a trapezoid given its median line and height -/
theorem trapezoid_area (median_line height : ℝ) (h1 : median_line = 8) (h2 : height = 12) :
  median_line * height = 96 :=
sorry

end NUMINAMATH_CALUDE_trapezoid_area_l988_98893


namespace NUMINAMATH_CALUDE_roots_reciprocal_sum_squared_l988_98889

theorem roots_reciprocal_sum_squared (a b c r s : ℝ) (ha : a ≠ 0) (hc : c ≠ 0)
  (hr : a * r^2 + b * r + c = 0) (hs : a * s^2 + b * s + c = 0) :
  1 / r^2 + 1 / s^2 = (b^2 - 2*a*c) / c^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_reciprocal_sum_squared_l988_98889


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l988_98832

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n < 1000 → n % 17 = 0 → n ≤ 986 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l988_98832


namespace NUMINAMATH_CALUDE_desired_depth_is_50_l988_98850

/-- Represents the digging scenario with initial and new conditions -/
structure DiggingScenario where
  initial_men : ℕ
  initial_hours : ℕ
  initial_depth : ℕ
  new_hours : ℕ
  extra_men : ℕ

/-- Calculates the desired depth given a digging scenario -/
def desired_depth (scenario : DiggingScenario) : ℕ :=
  let initial_work := scenario.initial_men * scenario.initial_hours
  let new_men := scenario.initial_men + scenario.extra_men
  let new_work := new_men * scenario.new_hours
  (new_work * scenario.initial_depth) / initial_work

/-- The main theorem stating that the desired depth is 50 meters -/
theorem desired_depth_is_50 (scenario : DiggingScenario)
  (h1 : scenario.initial_men = 18)
  (h2 : scenario.initial_hours = 8)
  (h3 : scenario.initial_depth = 30)
  (h4 : scenario.new_hours = 6)
  (h5 : scenario.extra_men = 22) :
  desired_depth scenario = 50 := by
  sorry

end NUMINAMATH_CALUDE_desired_depth_is_50_l988_98850


namespace NUMINAMATH_CALUDE_exclusive_movies_count_l988_98885

/-- Given two movie collections belonging to Andrew and John, this theorem proves
    the number of movies that are in either collection but not both. -/
theorem exclusive_movies_count
  (total_andrew : ℕ)
  (shared : ℕ)
  (john_exclusive : ℕ)
  (h1 : total_andrew = 25)
  (h2 : shared = 15)
  (h3 : john_exclusive = 8) :
  total_andrew - shared + john_exclusive = 18 :=
by sorry

end NUMINAMATH_CALUDE_exclusive_movies_count_l988_98885


namespace NUMINAMATH_CALUDE_square_sum_is_one_l988_98828

/-- Given two real numbers A and B, we define two functions f and g. -/
def f (A B x : ℝ) : ℝ := A * x^2 + B

def g (A B x : ℝ) : ℝ := B * x^2 + A

/-- The main theorem stating that under certain conditions, A^2 + B^2 = 1 -/
theorem square_sum_is_one (A B : ℝ) (h1 : A ≠ B) 
    (h2 : ∀ x, f A B (g A B x) - g A B (f A B x) = B^2 - A^2) : 
  A^2 + B^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_square_sum_is_one_l988_98828


namespace NUMINAMATH_CALUDE_no_solution_PP_QQ_l988_98823

-- Define the type of polynomials over ℝ
variable (P Q : ℝ → ℝ)

-- Hypothesis: P and Q are polynomials
axiom P_polynomial : Polynomial ℝ
axiom Q_polynomial : Polynomial ℝ

-- Hypothesis: ∀x ∈ ℝ, P(Q(x)) = Q(P(x))
axiom functional_equality : ∀ x : ℝ, P (Q x) = Q (P x)

-- Hypothesis: P(x) = Q(x) has no solutions
axiom no_solution_PQ : ∀ x : ℝ, P x ≠ Q x

-- Theorem: P(P(x)) = Q(Q(x)) has no solutions
theorem no_solution_PP_QQ : ∀ x : ℝ, P (P x) ≠ Q (Q x) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_PP_QQ_l988_98823


namespace NUMINAMATH_CALUDE_expression_evaluation_l988_98831

theorem expression_evaluation (a b : ℝ) (h1 : a = 2) (h2 : b = -1) :
  ((2*a + 3*b) * (2*a - 3*b) - (2*a - b)^2 - 2*a*b) / (-2*b) = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l988_98831


namespace NUMINAMATH_CALUDE_age_difference_l988_98821

/-- Given the ages of three people a, b, and c, prove that a is 2 years older than b -/
theorem age_difference (a b c : ℕ) : 
  b = 2 * c →
  a + b + c = 12 →
  b = 4 →
  a = b + 2 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l988_98821


namespace NUMINAMATH_CALUDE_marble_count_l988_98879

theorem marble_count (blue : ℕ) (yellow : ℕ) (p_yellow : ℚ) (red : ℕ) :
  blue = 7 →
  yellow = 6 →
  p_yellow = 1/4 →
  red = blue + yellow + red →
  yellow = p_yellow * (blue + yellow + red) →
  red = 11 := by sorry

end NUMINAMATH_CALUDE_marble_count_l988_98879


namespace NUMINAMATH_CALUDE_min_value_quadratic_l988_98815

theorem min_value_quadratic :
  let f (x : ℝ) := x^2 + 14*x + 10
  ∃ (y_min : ℝ), (∀ (x : ℝ), f x ≥ y_min) ∧ (∃ (x : ℝ), f x = y_min) ∧ y_min = -39 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l988_98815


namespace NUMINAMATH_CALUDE_consecutive_odd_squares_difference_l988_98880

theorem consecutive_odd_squares_difference (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1)^2 - (2*n - 1)^2 = 8*k := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_difference_l988_98880


namespace NUMINAMATH_CALUDE_bug_on_square_probability_l988_98864

/-- Represents the probability of the bug being at its starting vertex after n moves -/
def Q (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => 1 - Q n

/-- The bug's movement on a square -/
theorem bug_on_square_probability : Q 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_bug_on_square_probability_l988_98864


namespace NUMINAMATH_CALUDE_set_equality_implies_value_l988_98856

theorem set_equality_implies_value (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2012 + b^2012 = 1 :=
by sorry

end NUMINAMATH_CALUDE_set_equality_implies_value_l988_98856


namespace NUMINAMATH_CALUDE_tetrahedron_passage_l988_98895

/-- The minimal radius through which a regular tetrahedron with edge length 1 can pass -/
def min_radius : ℝ := 0.4478

/-- A regular tetrahedron with edge length 1 -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_regular : edge_length = 1

/-- A circular hole -/
structure CircularHole where
  radius : ℝ

/-- Predicate for whether a tetrahedron can pass through a hole -/
def can_pass_through (t : RegularTetrahedron) (h : CircularHole) : Prop :=
  h.radius ≥ min_radius

/-- Theorem stating the condition for a regular tetrahedron to pass through a circular hole -/
theorem tetrahedron_passage (t : RegularTetrahedron) (h : CircularHole) :
  can_pass_through t h ↔ h.radius ≥ min_radius :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_passage_l988_98895


namespace NUMINAMATH_CALUDE_total_crayons_l988_98866

def initial_crayons : ℕ := 7
def added_crayons : ℕ := 3

theorem total_crayons : 
  initial_crayons + added_crayons = 10 := by sorry

end NUMINAMATH_CALUDE_total_crayons_l988_98866


namespace NUMINAMATH_CALUDE_zig_book_count_l988_98829

/-- Given that Zig wrote four times as many books as Flo and they wrote 75 books in total,
    prove that Zig wrote 60 books. -/
theorem zig_book_count (flo_books : ℕ) (zig_books : ℕ) : 
  zig_books = 4 * flo_books →  -- Zig wrote four times as many books as Flo
  zig_books + flo_books = 75 →  -- They wrote 75 books altogether
  zig_books = 60 :=  -- Prove that Zig wrote 60 books
by sorry

end NUMINAMATH_CALUDE_zig_book_count_l988_98829


namespace NUMINAMATH_CALUDE_decorative_band_length_l988_98886

/-- The length of a decorative band for a circular sign -/
theorem decorative_band_length :
  let π : ℚ := 22 / 7
  let area : ℚ := 616
  let extra_length : ℚ := 5
  let radius : ℚ := (area / π).sqrt
  let circumference : ℚ := 2 * π * radius
  let band_length : ℚ := circumference + extra_length
  band_length = 93 := by sorry

end NUMINAMATH_CALUDE_decorative_band_length_l988_98886


namespace NUMINAMATH_CALUDE_prob_black_then_red_is_15_59_l988_98839

/-- A deck of cards with specific properties -/
structure Deck :=
  (total : ℕ)
  (black : ℕ)
  (red : ℕ)
  (h_total : total = 60)
  (h_black : black = 30)
  (h_red : red = 30)
  (h_sum : black + red = total)

/-- The probability of drawing a black card first and a red card second -/
def prob_black_then_red (d : Deck) : ℚ :=
  (d.black : ℚ) / d.total * d.red / (d.total - 1)

/-- Theorem stating the probability is equal to 15/59 -/
theorem prob_black_then_red_is_15_59 (d : Deck) :
  prob_black_then_red d = 15 / 59 := by
  sorry


end NUMINAMATH_CALUDE_prob_black_then_red_is_15_59_l988_98839


namespace NUMINAMATH_CALUDE_highest_score_is_174_l988_98801

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  total_innings : ℕ
  total_runs : ℕ
  highest_score : ℕ
  lowest_score : ℕ

/-- Calculates the average score for a batsman -/
def average_score (stats : BatsmanStats) : ℚ :=
  stats.total_runs / stats.total_innings

/-- Calculates the average score excluding highest and lowest scores -/
def average_score_excluding_extremes (stats : BatsmanStats) : ℚ :=
  (stats.total_runs - stats.highest_score - stats.lowest_score) / (stats.total_innings - 2)

/-- Theorem: Given the conditions, the batsman's highest score is 174 runs -/
theorem highest_score_is_174 (stats : BatsmanStats) :
  stats.total_innings = 40 ∧
  average_score stats = 50 ∧
  stats.highest_score = stats.lowest_score + 172 ∧
  average_score_excluding_extremes stats = 48 →
  stats.highest_score = 174 := by
  sorry

#check highest_score_is_174

end NUMINAMATH_CALUDE_highest_score_is_174_l988_98801


namespace NUMINAMATH_CALUDE_function_value_theorem_l988_98877

theorem function_value_theorem (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f ((1/2) * x - 1) = 2 * x + 3) →
  f m = 6 →
  m = -(1/4) := by
sorry

end NUMINAMATH_CALUDE_function_value_theorem_l988_98877


namespace NUMINAMATH_CALUDE_car_wash_earnings_difference_l988_98857

theorem car_wash_earnings_difference :
  ∀ (total : ℝ) (lisa_earnings : ℝ) (tommy_earnings : ℝ),
  total = 60 →
  lisa_earnings = total / 2 →
  tommy_earnings = lisa_earnings / 2 →
  lisa_earnings - tommy_earnings = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_car_wash_earnings_difference_l988_98857


namespace NUMINAMATH_CALUDE_prism_18_edges_has_8_faces_l988_98870

/-- A prism is a polyhedron with two congruent parallel faces (bases) and lateral faces that are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given the number of edges. -/
def num_faces (p : Prism) : ℕ :=
  2 + (p.edges / 3)  -- 2 bases + lateral faces

/-- Theorem: A prism with 18 edges has 8 faces. -/
theorem prism_18_edges_has_8_faces (p : Prism) (h : p.edges = 18) : num_faces p = 8 := by
  sorry


end NUMINAMATH_CALUDE_prism_18_edges_has_8_faces_l988_98870


namespace NUMINAMATH_CALUDE_z3_magnitude_range_l988_98807

open Complex

theorem z3_magnitude_range (z₁ z₂ z₃ : ℂ) 
  (h1 : abs z₁ = Real.sqrt 2)
  (h2 : abs z₂ = Real.sqrt 2)
  (h3 : (z₁.re * z₂.re + z₁.im * z₂.im) = 0)
  (h4 : abs (z₁ + z₂ - z₃) = 2) :
  ∃ (r : ℝ), r ∈ Set.Icc 0 4 ∧ abs z₃ = r :=
by sorry

end NUMINAMATH_CALUDE_z3_magnitude_range_l988_98807


namespace NUMINAMATH_CALUDE_no_leftover_eggs_l988_98892

/-- The number of eggs Abigail has -/
def abigail_eggs : ℕ := 58

/-- The number of eggs Beatrice has -/
def beatrice_eggs : ℕ := 35

/-- The number of eggs Carson has -/
def carson_eggs : ℕ := 27

/-- The size of each egg carton -/
def carton_size : ℕ := 10

/-- The theorem stating that there are no leftover eggs -/
theorem no_leftover_eggs : (abigail_eggs + beatrice_eggs + carson_eggs) % carton_size = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_leftover_eggs_l988_98892


namespace NUMINAMATH_CALUDE_equation_solution_l988_98852

theorem equation_solution : 
  ∃ x : ℝ, (-3 * x - 9 = 6 * x + 18) ∧ (x = -3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l988_98852


namespace NUMINAMATH_CALUDE_one_rhythm_for_specific_phrase_l988_98809

/-- Represents the duration of a note in terms of fractions of a measure -/
structure NoteDuration where
  numerator : ℕ
  denominator : ℕ+

/-- Represents a musical phrase -/
structure MusicalPhrase where
  measures : ℕ
  note_duration : NoteDuration
  no_rests : Bool

/-- Counts the number of different rhythms possible for a given musical phrase -/
def count_rhythms (phrase : MusicalPhrase) : ℕ :=
  sorry

/-- Theorem stating that a 2-measure phrase with notes lasting 1/8 of 1/4 of a measure and no rests has only one possible rhythm -/
theorem one_rhythm_for_specific_phrase :
  ∀ (phrase : MusicalPhrase),
    phrase.measures = 2 ∧
    phrase.note_duration = { numerator := 1, denominator := 32 } ∧
    phrase.no_rests = true →
    count_rhythms phrase = 1 :=
  sorry

end NUMINAMATH_CALUDE_one_rhythm_for_specific_phrase_l988_98809


namespace NUMINAMATH_CALUDE_square_calculation_l988_98883

theorem square_calculation :
  (41 ^ 2 = 40 ^ 2 + 81) ∧ (39 ^ 2 = 40 ^ 2 - 79) := by
  sorry

end NUMINAMATH_CALUDE_square_calculation_l988_98883


namespace NUMINAMATH_CALUDE_no_nonzero_solution_l988_98894

theorem no_nonzero_solution :
  ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ (1 / x + 2 / y = 1 / (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_solution_l988_98894


namespace NUMINAMATH_CALUDE_solution_set_inequality_l988_98844

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) / (2 * x + 1) ≤ 0 ↔ -1/2 < x ∧ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l988_98844


namespace NUMINAMATH_CALUDE_range_of_expressions_l988_98843

-- Define variables a and b with given constraints
theorem range_of_expressions (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : 2 < b ∧ b < 8) :
  -- (1) Range of a/b
  (1/8 : ℝ) < a/b ∧ a/b < 2 ∧
  -- (2) Range of 2a + 3b
  8 < 2*a + 3*b ∧ 2*a + 3*b < 32 ∧
  -- (3) Range of a - b
  -7 < a - b ∧ a - b < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expressions_l988_98843


namespace NUMINAMATH_CALUDE_min_sum_box_dimensions_l988_98833

theorem min_sum_box_dimensions : 
  ∀ (l w h : ℕ+), 
  l * w * h = 3003 → 
  ∀ (a b c : ℕ+), 
  a * b * c = 3003 → 
  l + w + h ≤ a + b + c ∧
  ∃ (x y z : ℕ+), x * y * z = 3003 ∧ x + y + z = 45 := by
sorry

end NUMINAMATH_CALUDE_min_sum_box_dimensions_l988_98833


namespace NUMINAMATH_CALUDE_intersection_distance_is_2b_l988_98825

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < a

/-- Represents a parabola with focus (p, 0) and directrix x = -p -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- The distance between intersection points of an ellipse and a parabola -/
def intersection_distance (e : Ellipse) (p : Parabola) : ℝ :=
  sorry

/-- Theorem stating the distance between intersection points -/
theorem intersection_distance_is_2b 
  (e : Ellipse) 
  (p : Parabola) 
  (h1 : e.a = 5 ∧ e.b = 4)  -- Ellipse equation condition
  (h2 : p.p = 3)  -- Shared focus condition
  (h3 : ∃ b : ℝ, b > 0 ∧ 
    (b^2 / 6 + 1.5)^2 / 25 + b^2 / 16 = 1)  -- Intersection condition
  : 
  ∃ b : ℝ, intersection_distance e p = 2 * b ∧ 
    b > 0 ∧ 
    (b^2 / 6 + 1.5)^2 / 25 + b^2 / 16 = 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_is_2b_l988_98825


namespace NUMINAMATH_CALUDE_g_is_linear_l988_98824

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom integral_condition : ∀ x : ℝ, f x + g x = ∫ t in x..(x+1), 2*t

axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem g_is_linear : (∀ x : ℝ, f x + g x = ∫ t in x..(x+1), 2*t) → 
                      (∀ x : ℝ, f (-x) = -f x) → 
                      (∀ x : ℝ, g x = 1 + x) :=
sorry

end NUMINAMATH_CALUDE_g_is_linear_l988_98824


namespace NUMINAMATH_CALUDE_perpendicular_lines_l988_98822

-- Define the slopes of two lines
def slope1 (a : ℝ) : ℝ := a
def slope2 (a : ℝ) : ℝ := a + 2

-- Define the condition for perpendicular lines
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_lines (a : ℝ) :
  perpendicular (slope1 a) (slope2 a) → a = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l988_98822


namespace NUMINAMATH_CALUDE_part_one_part_two_l988_98876

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part_one :
  let f := f 2
  {x : ℝ | f x ≥ 3 - |x - 1|} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} := by sorry

-- Part 2
theorem part_two :
  ∀ a m n : ℝ,
  m > 0 → n > 0 →
  m + 2*n = a →
  ({x : ℝ | f a x ≤ 1} = {x : ℝ | 2 ≤ x ∧ x ≤ 4}) →
  ∃ (min : ℝ), min = 9/2 ∧ ∀ m' n' : ℝ, m' > 0 → n' > 0 → m' + 2*n' = a → m'^2 + 4*n'^2 ≥ min := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l988_98876


namespace NUMINAMATH_CALUDE_units_digit_17_2025_l988_98805

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The property that 17^n and 7^n have the same units digit for all n -/
axiom units_digit_17_7 (n : ℕ) : unitsDigit (17^n) = unitsDigit (7^n)

/-- The main theorem: the units digit of 17^2025 is 7 -/
theorem units_digit_17_2025 : unitsDigit (17^2025) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_2025_l988_98805


namespace NUMINAMATH_CALUDE_inequality_proof_l988_98858

theorem inequality_proof (a b : ℝ) (n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l988_98858


namespace NUMINAMATH_CALUDE_square_and_sqrt_problem_l988_98872

theorem square_and_sqrt_problem :
  let a : ℕ := 101
  let b : ℕ := 10101
  let c : ℕ := 102030405060504030201
  (a ^ 2 = 10201) ∧
  (b ^ 2 = 102030201) ∧
  (c = 10101010101 ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_square_and_sqrt_problem_l988_98872


namespace NUMINAMATH_CALUDE_distinct_values_of_d_l988_98810

theorem distinct_values_of_d (d : ℂ) (u v w x : ℂ) 
  (h_distinct : u ≠ v ∧ u ≠ w ∧ u ≠ x ∧ v ≠ w ∧ v ≠ x ∧ w ≠ x)
  (h_eq : ∀ z : ℂ, (z - u) * (z - v) * (z - w) * (z - x) = 
                   (z - d*u) * (z - d*v) * (z - d*w) * (z - d*x)) :
  ∃! (S : Finset ℂ), S.card = 4 ∧ ∀ d' : ℂ, d' ∈ S ↔ 
    (∀ z : ℂ, (z - u) * (z - v) * (z - w) * (z - x) = 
              (z - d'*u) * (z - d'*v) * (z - d'*w) * (z - d'*x)) :=
by sorry

end NUMINAMATH_CALUDE_distinct_values_of_d_l988_98810


namespace NUMINAMATH_CALUDE_lcm_problem_l988_98891

theorem lcm_problem (n : ℕ) (h1 : n > 0) (h2 : Nat.lcm 24 n = 72) (h3 : Nat.lcm n 27 = 108) :
  n = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l988_98891


namespace NUMINAMATH_CALUDE_tailor_time_ratio_l988_98897

theorem tailor_time_ratio (num_shirts : ℕ) (num_pants : ℕ) (shirt_time : ℚ) 
  (hourly_rate : ℚ) (total_cost : ℚ) :
  num_shirts = 10 →
  num_pants = 12 →
  shirt_time = 3/2 →
  hourly_rate = 30 →
  total_cost = 1530 →
  ∃ (pants_time : ℚ), 
    pants_time / shirt_time = 2 ∧
    total_cost = hourly_rate * (num_shirts * shirt_time + num_pants * pants_time) :=
by sorry

end NUMINAMATH_CALUDE_tailor_time_ratio_l988_98897


namespace NUMINAMATH_CALUDE_exists_favorable_config_for_second_player_l988_98840

/-- Represents a square on the game board -/
structure Square :=
  (hasArrow : Bool)

/-- Represents the game board -/
def Board := List Square

/-- Calculates the probability of the second player winning given a board configuration and game parameters -/
def secondPlayerWinProbability (board : Board) (s₁ : ℕ) (s₂ : ℕ) : ℝ :=
  sorry -- Implementation details omitted

/-- Theorem stating that there exists a board configuration where the second player has a winning probability greater than 1/2, even when s₁ > s₂ -/
theorem exists_favorable_config_for_second_player :
  ∃ (board : Board) (s₁ s₂ : ℕ), s₁ > s₂ ∧ secondPlayerWinProbability board s₁ s₂ > (1/2 : ℝ) :=
sorry


end NUMINAMATH_CALUDE_exists_favorable_config_for_second_player_l988_98840


namespace NUMINAMATH_CALUDE_linear_inequality_solution_set_l988_98848

theorem linear_inequality_solution_set 
  (a b : ℝ) 
  (h1 : a = -1) 
  (h2 : b = 1) : 
  {x : ℝ | a * x + b < 0} = {x : ℝ | x > 1} := by
sorry

end NUMINAMATH_CALUDE_linear_inequality_solution_set_l988_98848


namespace NUMINAMATH_CALUDE_larry_initial_amount_l988_98882

def larry_problem (initial_amount lunch_cost brother_gift current_amount : ℕ) : Prop :=
  initial_amount = lunch_cost + brother_gift + current_amount

theorem larry_initial_amount :
  ∃ (initial_amount : ℕ), larry_problem initial_amount 5 2 15 ∧ initial_amount = 22 := by
  sorry

end NUMINAMATH_CALUDE_larry_initial_amount_l988_98882


namespace NUMINAMATH_CALUDE_rotation_90_degrees_l988_98842

def rotate90 (z : ℂ) : ℂ := Complex.I * z

theorem rotation_90_degrees : 
  rotate90 (-8 - 4 * Complex.I) = (4 : ℂ) - 8 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_rotation_90_degrees_l988_98842


namespace NUMINAMATH_CALUDE_sum_four_pentagons_l988_98808

/-- The value of a square -/
def square : ℚ := sorry

/-- The value of a pentagon -/
def pentagon : ℚ := sorry

/-- First equation: square + 3*pentagon + square + pentagon = 25 -/
axiom eq1 : square + 3*pentagon + square + pentagon = 25

/-- Second equation: pentagon + 2*square + pentagon + square + pentagon = 22 -/
axiom eq2 : pentagon + 2*square + pentagon + square + pentagon = 22

/-- The sum of four pentagons is equal to 62/3 -/
theorem sum_four_pentagons : 4 * pentagon = 62/3 := by sorry

end NUMINAMATH_CALUDE_sum_four_pentagons_l988_98808


namespace NUMINAMATH_CALUDE_cube_root_squared_times_fifth_root_l988_98890

theorem cube_root_squared_times_fifth_root (x : ℝ) (h : x > 0) :
  (x^(1/3))^2 * x^(1/5) = x^(13/15) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_squared_times_fifth_root_l988_98890


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l988_98863

theorem no_solutions_for_equation : ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (2 / a + 2 / b = 1 / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l988_98863


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_half_l988_98813

theorem reciprocal_of_negative_half :
  (1 : ℚ) / (-1/2 : ℚ) = -2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_half_l988_98813


namespace NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_leq_neg_seven_l988_98804

/-- A quadratic function f(x) = x^2 + (a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-1)*x + 2

/-- The property of f being monotonically decreasing on (-∞, 4] -/
def monotonically_decreasing (a : ℝ) : Prop :=
  ∀ x y, x < y → x ≤ 4 → f a x ≥ f a y

/-- Theorem: If f is monotonically decreasing on (-∞, 4], then a ≤ -7 -/
theorem monotonic_decreasing_implies_a_leq_neg_seven (a : ℝ) :
  monotonically_decreasing a → a ≤ -7 := by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_implies_a_leq_neg_seven_l988_98804


namespace NUMINAMATH_CALUDE_sum_of_roots_l988_98875

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 12*a^2 + 9*a - 18 = 0)
  (hb : 9*b^3 - 135*b^2 + 450*b - 1650 = 0) :
  a + b = 6 ∨ a + b = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l988_98875


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_l988_98836

theorem pencil_eraser_cost : ∃ (p e : ℕ), 
  15 * p + 5 * e = 200 ∧ 
  p ≥ 2 * e ∧ 
  p + e = 14 := by
sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_l988_98836


namespace NUMINAMATH_CALUDE_marble_remainder_l988_98855

theorem marble_remainder (n m : ℤ) : ∃ k : ℤ, (8*n + 5) + (8*m + 7) + 6 = 8*k + 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_remainder_l988_98855


namespace NUMINAMATH_CALUDE_double_iced_cubes_count_l988_98888

/-- Represents a cube cake with icing -/
structure IcedCake where
  size : Nat
  has_top_icing : Bool
  has_side_icing : Bool
  middle_icing_height : Rat

/-- Counts cubes with exactly two iced sides in an iced cake -/
def count_double_iced_cubes (cake : IcedCake) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem double_iced_cubes_count (cake : IcedCake) : 
  cake.size = 5 ∧ 
  cake.has_top_icing = true ∧ 
  cake.has_side_icing = true ∧ 
  cake.middle_icing_height = 5/2 →
  count_double_iced_cubes cake = 72 :=
by sorry

end NUMINAMATH_CALUDE_double_iced_cubes_count_l988_98888


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l988_98873

theorem complex_modulus_problem (z : ℂ) (a : ℝ) : 
  z = a * Complex.I → 
  (Complex.re ((1 + z) * (1 + Complex.I)) = (1 + z) * (1 + Complex.I)) → 
  Complex.abs (z + 2) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l988_98873


namespace NUMINAMATH_CALUDE_new_ratio_is_25_to_1_l988_98816

/-- Represents the ratio of students to teachers -/
structure Ratio where
  students : ℕ
  teachers : ℕ

def initial_ratio : Ratio := { students := 50, teachers := 1 }
def initial_teachers : ℕ := 3
def student_increase : ℕ := 50
def teacher_increase : ℕ := 5

def new_ratio : Ratio :=
  { students := initial_ratio.students * initial_teachers + student_increase,
    teachers := initial_teachers + teacher_increase }

theorem new_ratio_is_25_to_1 : new_ratio = { students := 25, teachers := 1 } := by
  sorry

end NUMINAMATH_CALUDE_new_ratio_is_25_to_1_l988_98816


namespace NUMINAMATH_CALUDE_function_inequality_l988_98838

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h1 : deriv f 1 = 0) (h2 : ∀ x ≠ 1, (x - 1) * (deriv f x) > 0) : 
  f 0 + f 2 > 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l988_98838


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l988_98867

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 512 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l988_98867


namespace NUMINAMATH_CALUDE_steve_pencil_theorem_l988_98814

def steve_pencil_problem (boxes : ℕ) (pencils_per_box : ℕ) (lauren_pencils : ℕ) (matt_extra_pencils : ℕ) : Prop :=
  let total_pencils := boxes * pencils_per_box
  let matt_pencils := lauren_pencils + matt_extra_pencils
  let given_away_pencils := lauren_pencils + matt_pencils
  let remaining_pencils := total_pencils - given_away_pencils
  remaining_pencils = 9

theorem steve_pencil_theorem :
  steve_pencil_problem 2 12 6 3 :=
by
  sorry

end NUMINAMATH_CALUDE_steve_pencil_theorem_l988_98814


namespace NUMINAMATH_CALUDE_derivative_of_y_l988_98847

-- Define the function y
def y (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

-- State the theorem
theorem derivative_of_y (x : ℝ) : 
  deriv y x = 4 * x - 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_y_l988_98847


namespace NUMINAMATH_CALUDE_pear_apple_equivalence_l988_98811

/-- The cost of fruits at Joe's Fruit Stand -/
structure FruitCost where
  pear : ℕ
  grape : ℕ
  apple : ℕ

/-- The relation between pears and grapes -/
def pear_grape_relation (c : FruitCost) : Prop :=
  4 * c.pear = 3 * c.grape

/-- The relation between grapes and apples -/
def grape_apple_relation (c : FruitCost) : Prop :=
  9 * c.grape = 6 * c.apple

/-- Theorem stating the cost equivalence of 24 pears and 12 apples -/
theorem pear_apple_equivalence (c : FruitCost) 
  (h1 : pear_grape_relation c) 
  (h2 : grape_apple_relation c) : 
  24 * c.pear = 12 * c.apple :=
by
  sorry

end NUMINAMATH_CALUDE_pear_apple_equivalence_l988_98811


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_pi_6_l988_98851

theorem sin_2alpha_minus_pi_6 (α : Real) :
  (∃ P : Real × Real, P.1 = -3/5 ∧ P.2 = 4/5 ∧ P.1^2 + P.2^2 = 1 ∧
    P.1 = Real.cos (α + π/6) ∧ P.2 = Real.sin (α + π/6)) →
  Real.sin (2*α - π/6) = 7/25 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_pi_6_l988_98851


namespace NUMINAMATH_CALUDE_sin_210_degrees_l988_98841

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l988_98841


namespace NUMINAMATH_CALUDE_max_b_value_l988_98819

theorem max_b_value (x b : ℤ) : 
  x^2 + b*x = -21 → 
  b > 0 → 
  (∃ (max_b : ℤ), max_b = 22 ∧ ∀ (b' : ℤ), b' > 0 → (∃ (x' : ℤ), x'^2 + b'*x' = -21) → b' ≤ max_b) :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l988_98819


namespace NUMINAMATH_CALUDE_factor_expression_l988_98834

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l988_98834


namespace NUMINAMATH_CALUDE_unique_number_with_properties_l988_98817

def has_two_prime_factors (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = p^a * q^b

def count_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem unique_number_with_properties :
  ∃! n : ℕ, n > 0 ∧ has_two_prime_factors n ∧ count_divisors n = 6 ∧ sum_of_divisors n = 28 ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_properties_l988_98817


namespace NUMINAMATH_CALUDE_second_divisor_problem_l988_98865

theorem second_divisor_problem (N : ℕ) (D : ℕ) : 
  N % 35 = 25 → N % D = 4 → D = 31 := by
sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l988_98865


namespace NUMINAMATH_CALUDE_rainfall_difference_l988_98827

/-- The number of Mondays -/
def num_mondays : ℕ := 13

/-- The rainfall on each Monday in centimeters -/
def rain_per_monday : ℝ := 1.75

/-- The number of Tuesdays -/
def num_tuesdays : ℕ := 16

/-- The rainfall on each Tuesday in centimeters -/
def rain_per_tuesday : ℝ := 2.65

/-- The difference in total rainfall between Tuesdays and Mondays -/
theorem rainfall_difference : 
  (num_tuesdays : ℝ) * rain_per_tuesday - (num_mondays : ℝ) * rain_per_monday = 19.65 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_difference_l988_98827


namespace NUMINAMATH_CALUDE_particular_number_solution_l988_98849

theorem particular_number_solution (A B : ℤ) (h1 : A = 14) (h2 : B = 24) :
  ∃ x : ℚ, ((A + x) * A - B) / B = 13 ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_solution_l988_98849


namespace NUMINAMATH_CALUDE_solution_difference_l988_98861

theorem solution_difference (p q : ℝ) : 
  ((3 * p - 9) / (p^2 + 3*p - 18) = p + 3) →
  ((3 * q - 9) / (q^2 + 3*q - 18) = q + 3) →
  p ≠ q →
  p > q →
  p - q = 2 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l988_98861


namespace NUMINAMATH_CALUDE_fourth_competitor_jump_distance_l988_98899

/-- Long jump competition with four competitors -/
structure LongJumpCompetition where
  first_jump : ℕ
  second_jump : ℕ
  third_jump : ℕ
  fourth_jump : ℕ

/-- The long jump competition satisfying the given conditions -/
def competition : LongJumpCompetition where
  first_jump := 22
  second_jump := 23
  third_jump := 21
  fourth_jump := 24

/-- Theorem stating the conditions and the result to be proved -/
theorem fourth_competitor_jump_distance :
  let c := competition
  c.first_jump = 22 ∧
  c.second_jump = c.first_jump + 1 ∧
  c.third_jump = c.second_jump - 2 ∧
  c.fourth_jump = c.third_jump + 3 →
  c.fourth_jump = 24 := by
  sorry


end NUMINAMATH_CALUDE_fourth_competitor_jump_distance_l988_98899


namespace NUMINAMATH_CALUDE_inscribed_square_area_l988_98845

/-- A square inscribed in a right triangle -/
structure InscribedSquare where
  /-- The side length of the inscribed square -/
  side : ℝ
  /-- The distance from one vertex of the right triangle to where the square touches the hypotenuse -/
  dist1 : ℝ
  /-- The distance from the other vertex of the right triangle to where the square touches the hypotenuse -/
  dist2 : ℝ
  /-- The constraint that the square is properly inscribed in the right triangle -/
  inscribed : side * side = dist1 * dist2

/-- The theorem stating that a square inscribed in a right triangle with specific measurements has an area of 975 -/
theorem inscribed_square_area (s : InscribedSquare) 
    (h1 : s.dist1 = 15) 
    (h2 : s.dist2 = 65) : 
  s.side * s.side = 975 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l988_98845


namespace NUMINAMATH_CALUDE_problem_solution_l988_98859

/-- The number of people initially working on the problem -/
def initial_people : ℕ := 1

/-- The initial working time in hours -/
def initial_time : ℕ := 10

/-- The working time after adding one person, in hours -/
def reduced_time : ℕ := 5

theorem problem_solution :
  initial_people * initial_time = (initial_people + 1) * reduced_time :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l988_98859


namespace NUMINAMATH_CALUDE_angle_tangent_sum_zero_l988_98878

theorem angle_tangent_sum_zero :
  ∃ θ : Real,
    0 < θ ∧ θ < π / 6 ∧
    Real.tan θ + Real.tan (2 * θ) + Real.tan (4 * θ) + Real.tan (5 * θ) = 0 ∧
    Real.tan θ = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_tangent_sum_zero_l988_98878


namespace NUMINAMATH_CALUDE_shooting_competition_l988_98884

theorem shooting_competition (hit_rate_A hit_rate_B prob_total_2 : ℚ) : 
  hit_rate_A = 3/5 →
  prob_total_2 = 9/20 →
  hit_rate_A * (1 - hit_rate_B) + (1 - hit_rate_A) * hit_rate_B = prob_total_2 →
  hit_rate_B = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_shooting_competition_l988_98884


namespace NUMINAMATH_CALUDE_bus_driver_hours_l988_98862

/-- Represents the compensation structure and work hours of a bus driver --/
structure BusDriver where
  regularRate : ℝ
  regularHours : ℝ
  overtimeRate : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Calculates the total hours worked by a bus driver --/
def totalHours (driver : BusDriver) : ℝ :=
  driver.regularHours + driver.overtimeHours

/-- Theorem stating the conditions and the result to be proved --/
theorem bus_driver_hours (driver : BusDriver) 
  (h1 : driver.regularRate = 16)
  (h2 : driver.regularHours = 40)
  (h3 : driver.overtimeRate = driver.regularRate * 1.75)
  (h4 : driver.totalCompensation = 976)
  (h5 : driver.totalCompensation = driver.regularRate * driver.regularHours + 
                                   driver.overtimeRate * driver.overtimeHours) :
  totalHours driver = 52 := by
  sorry

#eval 40 + 12 -- Expected output: 52

end NUMINAMATH_CALUDE_bus_driver_hours_l988_98862


namespace NUMINAMATH_CALUDE_sum_geq_three_over_product_l988_98806

theorem sum_geq_three_over_product {a b c : ℝ} 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : a + b + c > 1/a + 1/b + 1/c) : 
  a + b + c ≥ 3/(a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_three_over_product_l988_98806


namespace NUMINAMATH_CALUDE_symmetric_monotonic_sum_property_l988_98803

/-- A function that is monotonically increasing on R and symmetric with respect to (a,b) -/
class SymmetricMonotonicFunction (f : ℝ → ℝ) (a b : ℝ) :=
  (monotone : Monotone f)
  (symmetric : ∀ x, f (a - x) + f (a + x) = 2 * b)

/-- Theorem: If f(x₁) + f(x₂) > 2b, then x₁ + x₂ > 2a -/
theorem symmetric_monotonic_sum_property
  {f : ℝ → ℝ} {a b : ℝ} [SymmetricMonotonicFunction f a b]
  {x₁ x₂ : ℝ} (h : f x₁ + f x₂ > 2 * b) :
  x₁ + x₂ > 2 * a :=
by sorry

end NUMINAMATH_CALUDE_symmetric_monotonic_sum_property_l988_98803


namespace NUMINAMATH_CALUDE_max_surface_area_after_cut_l988_98881

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cuboid -/
def Cuboid.surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.width * c.height + c.length * c.height)

/-- Represents the result of cutting a cuboid into two triangular prisms -/
structure CutResult where
  prism1_surface_area : ℝ
  prism2_surface_area : ℝ

/-- Calculates the sum of surface areas after cutting -/
def CutResult.totalSurfaceArea (cr : CutResult) : ℝ :=
  cr.prism1_surface_area + cr.prism2_surface_area

/-- The main theorem stating the maximum sum of surface areas after cutting -/
theorem max_surface_area_after_cut (c : Cuboid) 
  (h1 : c.length = 5) 
  (h2 : c.width = 4) 
  (h3 : c.height = 3) : 
  (∃ (cr : CutResult), ∀ (cr' : CutResult), cr.totalSurfaceArea ≥ cr'.totalSurfaceArea) → 
  (∃ (cr : CutResult), cr.totalSurfaceArea = 144) :=
sorry

end NUMINAMATH_CALUDE_max_surface_area_after_cut_l988_98881


namespace NUMINAMATH_CALUDE_even_mono_increasing_inequality_l988_98835

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function is monotonically increasing on [0, +∞) if f(x) ≤ f(y) for 0 ≤ x ≤ y -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem even_mono_increasing_inequality (f : ℝ → ℝ) 
    (h_even : IsEven f) (h_mono : MonoIncreasing f) : 
    f (-2) > f 1 ∧ f 1 > f 0 := by
  sorry

end NUMINAMATH_CALUDE_even_mono_increasing_inequality_l988_98835


namespace NUMINAMATH_CALUDE_distance_between_homes_is_40_l988_98874

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 40

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 3

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 5

/-- The distance Maxwell travels before they meet -/
def maxwell_distance : ℝ := 15

/-- Theorem stating that the distance between homes is 40 km -/
theorem distance_between_homes_is_40 :
  distance_between_homes = maxwell_distance * (maxwell_speed + brad_speed) / maxwell_speed :=
by sorry

end NUMINAMATH_CALUDE_distance_between_homes_is_40_l988_98874


namespace NUMINAMATH_CALUDE_percentage_calculation_l988_98812

theorem percentage_calculation (P : ℝ) : 
  (P / 100) * 200 - 30 = 50 → P = 40 :=
by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l988_98812


namespace NUMINAMATH_CALUDE_box_of_balls_theorem_l988_98898

theorem box_of_balls_theorem :
  ∃ (B X Y : ℝ),
    40 < X ∧ X < 50 ∧
    60 < Y ∧ Y < 70 ∧
    B - X = Y - B ∧
    B = 55 := by sorry

end NUMINAMATH_CALUDE_box_of_balls_theorem_l988_98898


namespace NUMINAMATH_CALUDE_largest_constructible_cube_l988_98871

/-- Represents the dimensions of the cardboard sheet -/
def sheet_length : ℕ := 60
def sheet_width : ℕ := 25

/-- Checks if a cube with given edge length can be constructed from the sheet -/
def can_construct_cube (edge_length : ℕ) : Prop :=
  6 * edge_length^2 ≤ sheet_length * sheet_width ∧ 
  edge_length ≤ sheet_length ∧ 
  edge_length ≤ sheet_width

/-- The largest cube edge length that can be constructed -/
def max_cube_edge : ℕ := 15

/-- Theorem stating that the largest constructible cube has edge length of 15 cm -/
theorem largest_constructible_cube :
  can_construct_cube max_cube_edge ∧
  ∀ (n : ℕ), n > max_cube_edge → ¬(can_construct_cube n) :=
by sorry

#check largest_constructible_cube

end NUMINAMATH_CALUDE_largest_constructible_cube_l988_98871


namespace NUMINAMATH_CALUDE_x_14_plus_inverse_l988_98860

theorem x_14_plus_inverse (x : ℂ) (h : x^2 + x + 1 = 0) : x^14 + 1/x^14 = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_14_plus_inverse_l988_98860


namespace NUMINAMATH_CALUDE_problem1_l988_98846

theorem problem1 (a b : ℝ) (h1 : a = 1) (h2 : b = -3) :
  (a - b)^2 - 2*a*(a + 3*b) + (a + 2*b)*(a - 2*b) = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l988_98846
