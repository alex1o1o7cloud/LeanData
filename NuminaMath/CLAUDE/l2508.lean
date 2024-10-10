import Mathlib

namespace cone_base_radius_l2508_250868

/-- Given a cone whose lateral surface unfolds into a semicircle with radius 4,
    prove that the radius of the base of the cone is also 4. -/
theorem cone_base_radius (r : ℝ) (h : r = 4) : r = 4 := by
  sorry

end cone_base_radius_l2508_250868


namespace complex_exp_thirteen_pi_over_three_l2508_250807

theorem complex_exp_thirteen_pi_over_three : 
  Complex.exp (Complex.I * (13 * Real.pi / 3)) = Complex.ofReal (1 / 2) + Complex.I * Complex.ofReal (Real.sqrt 3 / 2) := by
  sorry

end complex_exp_thirteen_pi_over_three_l2508_250807


namespace two_points_same_color_distance_l2508_250827

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Theorem statement
theorem two_points_same_color_distance (x : ℝ) (h : x > 0) (coloring : Coloring) :
  ∃ (p q : Point) (c : Color), coloring p = c ∧ coloring q = c ∧ 
    Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2) = x := by
  sorry

end two_points_same_color_distance_l2508_250827


namespace seagull_fraction_l2508_250821

theorem seagull_fraction (initial_seagulls : ℕ) (scared_fraction : ℚ) (remaining_seagulls : ℕ) :
  initial_seagulls = 36 →
  scared_fraction = 1/4 →
  remaining_seagulls = 18 →
  (initial_seagulls - initial_seagulls * scared_fraction : ℚ) - remaining_seagulls = 
  (1/3) * (initial_seagulls - initial_seagulls * scared_fraction) :=
by
  sorry

end seagull_fraction_l2508_250821


namespace feed_supply_ducks_l2508_250809

/-- A batch of feed can supply a certain number of ducks for a given number of days. -/
def FeedSupply (ducks chickens days : ℕ) : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (ducks * x + chickens * y) * days = 210 * y

theorem feed_supply_ducks :
  FeedSupply 10 15 6 →
  FeedSupply 12 6 7 →
  FeedSupply 5 0 21 :=
by sorry

end feed_supply_ducks_l2508_250809


namespace coefficient_x_fourth_l2508_250876

def binomial_coeff (n k : ℕ) : ℕ := sorry

def binomial_expansion_term (n r : ℕ) (a b : ℚ) : ℚ := sorry

theorem coefficient_x_fourth (n : ℕ) (h : n = 5) :
  ∃ (k : ℕ), binomial_coeff n k * (2 * k - 5) = 10 ∧
             binomial_expansion_term n k 1 1 = binomial_coeff n k * (2 * k - 5) := by
  sorry

end coefficient_x_fourth_l2508_250876


namespace sqrt_meaningful_range_l2508_250850

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_meaningful_range_l2508_250850


namespace pasta_preference_ratio_l2508_250862

/-- Given a survey of students' pasta preferences, prove the ratio of spaghetti to tortellini preference -/
theorem pasta_preference_ratio 
  (total_students : ℕ) 
  (spaghetti_preference : ℕ) 
  (tortellini_preference : ℕ) 
  (h1 : total_students = 850)
  (h2 : spaghetti_preference = 300)
  (h3 : tortellini_preference = 200) :
  (spaghetti_preference : ℚ) / tortellini_preference = 3 / 2 :=
by sorry

end pasta_preference_ratio_l2508_250862


namespace manuscript_year_count_l2508_250842

/-- The number of possible 6-digit years formed from the digits 2, 2, 2, 2, 3, and 9,
    where the year must begin with an odd digit -/
def manuscript_year_possibilities : ℕ :=
  let total_digits : ℕ := 6
  let repeated_digit_count : ℕ := 4
  let odd_digit_choices : ℕ := 2
  odd_digit_choices * (Nat.factorial total_digits) / (Nat.factorial repeated_digit_count)

theorem manuscript_year_count : manuscript_year_possibilities = 60 := by
  sorry

end manuscript_year_count_l2508_250842


namespace no_integer_arithmetic_progression_l2508_250823

theorem no_integer_arithmetic_progression : 
  ¬ ∃ (a b : ℤ), (b - a = a - 6) ∧ (ab + 3 - b = b - a) := by sorry

end no_integer_arithmetic_progression_l2508_250823


namespace intersection_point_on_x_equals_4_l2508_250866

/-- An ellipse with center at origin and foci on coordinate axes -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  k : ℝ
  m : ℝ

def Ellipse.equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def Line.equation (l : Line) (p : Point) : Prop :=
  p.y = l.k * (p.x - l.m)

/-- The main theorem -/
theorem intersection_point_on_x_equals_4 
  (e : Ellipse)
  (h_passes_through_A : e.equation ⟨-2, 0⟩)
  (h_passes_through_B : e.equation ⟨2, 0⟩)
  (h_passes_through_C : e.equation ⟨1, 3/2⟩)
  (l : Line)
  (h_k_nonzero : l.k ≠ 0)
  (M N : Point)
  (h_M_on_E : e.equation M)
  (h_N_on_E : e.equation N)
  (h_M_on_l : l.equation M)
  (h_N_on_l : l.equation N) :
  ∃ (P : Point), P.x = 4 ∧ 
    (∃ (t : ℝ), P = ⟨4, t * (M.y + 2) + (1 - t) * M.y⟩) ∧
    (∃ (s : ℝ), P = ⟨4, s * (N.y - 2) + (1 - s) * N.y⟩) :=
sorry

end intersection_point_on_x_equals_4_l2508_250866


namespace max_correct_is_42_l2508_250853

/-- Represents the exam scoring system and Xiaolong's result -/
structure ExamResult where
  total_questions : Nat
  correct_points : Int
  incorrect_points : Int
  no_answer_points : Int
  total_score : Int

/-- Calculates the maximum number of correctly answered questions -/
def max_correct_answers (exam : ExamResult) : Nat :=
  sorry

/-- Theorem stating that the maximum number of correct answers is 42 -/
theorem max_correct_is_42 (exam : ExamResult) 
  (h1 : exam.total_questions = 50)
  (h2 : exam.correct_points = 3)
  (h3 : exam.incorrect_points = -1)
  (h4 : exam.no_answer_points = 0)
  (h5 : exam.total_score = 120) :
  max_correct_answers exam = 42 :=
  sorry

end max_correct_is_42_l2508_250853


namespace great_wall_precision_l2508_250829

/-- The precision of a number in scientific notation is determined by the place value of its last significant digit. -/
def precision_scientific_notation (mantissa : ℝ) (exponent : ℤ) : ℕ :=
  sorry

/-- The Great Wall's length in scientific notation -/
def great_wall_length : ℝ := 6.7

/-- The exponent in the scientific notation of the Great Wall's length -/
def great_wall_exponent : ℤ := 6

/-- Hundred thousands place value -/
def hundred_thousands : ℕ := 100000

theorem great_wall_precision :
  precision_scientific_notation great_wall_length great_wall_exponent = hundred_thousands :=
sorry

end great_wall_precision_l2508_250829


namespace intersecting_line_parameter_range_l2508_250800

/-- Given a line segment PQ and a line l, this theorem proves the range of m for which l intersects the extension of PQ. -/
theorem intersecting_line_parameter_range 
  (P : ℝ × ℝ) 
  (Q : ℝ × ℝ) 
  (l : ℝ → ℝ → Prop) 
  (h_P : P = (-1, 1)) 
  (h_Q : Q = (2, 2)) 
  (h_l : ∀ x y, l x y ↔ x + m * y + m = 0) 
  (h_intersect : ∃ x y, l x y ∧ (∃ t : ℝ, (x, y) = (1 - t) • P + t • Q ∧ t ∉ [0, 1])) :
  m ∈ Set.Ioo (-3 : ℝ) (-2/3) :=
sorry

end intersecting_line_parameter_range_l2508_250800


namespace simplify_expressions_l2508_250804

theorem simplify_expressions :
  (99^2 = 9801) ∧ (2000^2 - 1999 * 2001 = 1) := by
  sorry

end simplify_expressions_l2508_250804


namespace two_tangent_lines_l2508_250830

-- Define the point P
def P : ℝ × ℝ := (-4, 1)

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2 = 1

-- Define a line passing through P
def line_through_P (m : ℝ) (x y : ℝ) : Prop :=
  y - P.2 = m * (x - P.1)

-- Define the condition for a line to intersect the hyperbola at only one point
def intersects_at_one_point (m : ℝ) : Prop :=
  ∃! (x y : ℝ), hyperbola x y ∧ line_through_P m x y

-- The main theorem
theorem two_tangent_lines :
  ∃! (m₁ m₂ : ℝ), m₁ ≠ m₂ ∧ 
    intersects_at_one_point m₁ ∧ 
    intersects_at_one_point m₂ ∧
    ∀ m, intersects_at_one_point m → m = m₁ ∨ m = m₂ :=
  sorry

end two_tangent_lines_l2508_250830


namespace monotonic_decreasing_range_l2508_250805

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- State the theorem
theorem monotonic_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) → a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) := by
  sorry


end monotonic_decreasing_range_l2508_250805


namespace snowflake_four_two_l2508_250817

-- Define the snowflake operation
def snowflake (a b : ℕ) : ℕ := a * (b - 1) + a * b

-- Theorem statement
theorem snowflake_four_two : snowflake 4 2 = 12 := by
  sorry

end snowflake_four_two_l2508_250817


namespace simplify_fraction_l2508_250873

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 1625 / 12 := by
  sorry

end simplify_fraction_l2508_250873


namespace kats_training_hours_l2508_250857

/-- The number of hours Kat trains per week -/
def total_training_hours (strength_sessions : ℕ) (strength_hours : ℝ) 
  (boxing_sessions : ℕ) (boxing_hours : ℝ) : ℝ :=
  (strength_sessions : ℝ) * strength_hours + (boxing_sessions : ℝ) * boxing_hours

/-- Theorem stating that Kat's total training hours per week is 9 -/
theorem kats_training_hours :
  total_training_hours 3 1 4 1.5 = 9 := by
  sorry

end kats_training_hours_l2508_250857


namespace complex_equation_solution_l2508_250869

theorem complex_equation_solution (z : ℂ) :
  (1 + 2 * z) / (1 - z) = Complex.I → z = -1/5 + 3/5 * Complex.I :=
by
  sorry

end complex_equation_solution_l2508_250869


namespace rain_probability_l2508_250889

theorem rain_probability (p_friday p_monday : ℝ) 
  (h1 : p_friday = 0.3)
  (h2 : p_monday = 0.6)
  (h3 : 0 ≤ p_friday ∧ p_friday ≤ 1)
  (h4 : 0 ≤ p_monday ∧ p_monday ≤ 1) :
  1 - (1 - p_friday) * (1 - p_monday) = 0.72 := by
  sorry

end rain_probability_l2508_250889


namespace floor_sum_equals_n_l2508_250812

theorem floor_sum_equals_n (n : ℤ) : 
  ⌊n / 2⌋ + ⌊(n + 1) / 2⌋ = n := by sorry

end floor_sum_equals_n_l2508_250812


namespace sparrow_distribution_l2508_250822

theorem sparrow_distribution (a b c : ℕ) : 
  a + b + c = 24 →
  a - 4 = b + 1 →
  b + 1 = c + 3 →
  (a, b, c) = (12, 7, 5) := by
sorry

end sparrow_distribution_l2508_250822


namespace divisor_sum_inequality_equality_condition_l2508_250895

theorem divisor_sum_inequality (n : ℕ) (hn : n ≥ 2) :
  let divisors := (Finset.range (n + 1)).filter (λ d => n % d = 0)
  (divisors.sum id) / divisors.card ≥ Real.sqrt (n + 1/4) :=
sorry

theorem equality_condition (n : ℕ) (hn : n ≥ 2) :
  let divisors := (Finset.range (n + 1)).filter (λ d => n % d = 0)
  (divisors.sum id) / divisors.card = Real.sqrt (n + 1/4) ↔ n = 2 :=
sorry

end divisor_sum_inequality_equality_condition_l2508_250895


namespace hyperbola_in_trilinear_coordinates_l2508_250834

/-- Trilinear coordinates -/
structure TrilinearCoord where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Triangle with angles A, B, C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Hyperbola equation in trilinear coordinates -/
def hyperbola_equation (t : Triangle) (p : TrilinearCoord) : Prop :=
  (Real.sin (2 * t.A) * Real.cos (t.B - t.C)) / p.x +
  (Real.sin (2 * t.B) * Real.cos (t.C - t.A)) / p.y +
  (Real.sin (2 * t.C) * Real.cos (t.A - t.B)) / p.z = 0

/-- Theorem: The equation of the hyperbola in trilinear coordinates -/
theorem hyperbola_in_trilinear_coordinates (t : Triangle) (p : TrilinearCoord) :
  hyperbola_equation t p := by
  sorry

end hyperbola_in_trilinear_coordinates_l2508_250834


namespace remainder_theorem_l2508_250885

def P (x : ℝ) : ℝ := x^100 - x^99 + x^98 - x^97 + x^96 - x^95 + x^94 - x^93 + x^92 - x^91 + x^90 - x^89 + x^88 - x^87 + x^86 - x^85 + x^84 - x^83 + x^82 - x^81 + x^80 - x^79 + x^78 - x^77 + x^76 - x^75 + x^74 - x^73 + x^72 - x^71 + x^70 - x^69 + x^68 - x^67 + x^66 - x^65 + x^64 - x^63 + x^62 - x^61 + x^60 - x^59 + x^58 - x^57 + x^56 - x^55 + x^54 - x^53 + x^52 - x^51 + x^50 - x^49 + x^48 - x^47 + x^46 - x^45 + x^44 - x^43 + x^42 - x^41 + x^40 - x^39 + x^38 - x^37 + x^36 - x^35 + x^34 - x^33 + x^32 - x^31 + x^30 - x^29 + x^28 - x^27 + x^26 - x^25 + x^24 - x^23 + x^22 - x^21 + x^20 - x^19 + x^18 - x^17 + x^16 - x^15 + x^14 - x^13 + x^12 - x^11 + x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1

theorem remainder_theorem (a b : ℝ) : 
  (∃ Q : ℝ → ℝ, ∀ x, P x = Q x * (x^2 - 1) + a * x + b) → 
  2 * a + b = -49 := by
sorry

end remainder_theorem_l2508_250885


namespace inequality_solution_set_l2508_250846

theorem inequality_solution_set (m : ℝ) :
  {x : ℝ | x^2 - (2*m + 1)*x + m^2 + m < 0} = {x : ℝ | m < x ∧ x < m + 1} := by
  sorry

end inequality_solution_set_l2508_250846


namespace robotics_club_enrollment_l2508_250803

theorem robotics_club_enrollment (total : ℕ) (cs : ℕ) (elec : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : cs = 52)
  (h3 : elec = 45)
  (h4 : both = 32) :
  total - cs - elec + both = 15 := by
  sorry

end robotics_club_enrollment_l2508_250803


namespace classroom_weight_distribution_exists_l2508_250852

theorem classroom_weight_distribution_exists :
  ∃ (n : ℕ) (b g : ℕ) (boys_weights girls_weights : List ℝ),
    n < 35 ∧
    n = b + g ∧
    b > 0 ∧
    g > 0 ∧
    boys_weights.length = b ∧
    girls_weights.length = g ∧
    (boys_weights.sum + girls_weights.sum) / n = 53.5 ∧
    boys_weights.sum / b = 60 ∧
    girls_weights.sum / g = 47 ∧
    (∃ (min_boy : ℝ) (max_girl : ℝ),
      min_boy ∈ boys_weights ∧
      max_girl ∈ girls_weights ∧
      (∀ w ∈ boys_weights, min_boy ≤ w) ∧
      (∀ w ∈ girls_weights, w ≤ max_girl) ∧
      min_boy < max_girl) :=
by sorry

end classroom_weight_distribution_exists_l2508_250852


namespace negation_of_universal_proposition_l2508_250801

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x < 0) :=
by sorry

end negation_of_universal_proposition_l2508_250801


namespace continuous_piecewise_function_l2508_250855

-- Define the piecewise function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then x + 2 else 2 * x + a

-- State the theorem
theorem continuous_piecewise_function (a : ℝ) :
  Continuous (f a) ↔ a = -1 := by sorry

end continuous_piecewise_function_l2508_250855


namespace diagonal_length_from_area_and_offsets_l2508_250825

/-- The length of a quadrilateral's diagonal given its area and offsets -/
theorem diagonal_length_from_area_and_offsets (area : ℝ) (offset1 : ℝ) (offset2 : ℝ) :
  area = 90 ∧ offset1 = 5 ∧ offset2 = 4 →
  ∃ (diagonal : ℝ), diagonal = 20 ∧ area = (offset1 + offset2) * diagonal / 2 := by
  sorry

end diagonal_length_from_area_and_offsets_l2508_250825


namespace no_positive_integer_perfect_squares_l2508_250883

theorem no_positive_integer_perfect_squares :
  ¬ ∃ (n : ℕ), n > 0 ∧ ∃ (a b : ℕ), (n + 1) * 2^n = a^2 ∧ (n + 3) * 2^(n + 2) = b^2 := by
  sorry

end no_positive_integer_perfect_squares_l2508_250883


namespace square_plus_reciprocal_square_l2508_250887

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 2.5) : 
  x^2 + 1/x^2 = 4.25 := by
sorry

end square_plus_reciprocal_square_l2508_250887


namespace unique_special_sequence_l2508_250845

-- Define the sequence type
def SpecialSequence := ℕ → ℕ

-- Define the property of the sequence
def HasUniqueRepresentation (a : SpecialSequence) : Prop :=
  ∀ n : ℕ, ∃! (i j k : ℕ), n = a i + 2 * a j + 4 * a k

-- Define the strictly increasing property
def StrictlyIncreasing (a : SpecialSequence) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

-- Main theorem
theorem unique_special_sequence :
  ∃! a : SpecialSequence,
    StrictlyIncreasing a ∧
    HasUniqueRepresentation a ∧
    a 2002 = 1227132168 := by
  sorry


end unique_special_sequence_l2508_250845


namespace almost_perfect_numbers_l2508_250819

def d (n : ℕ) : ℕ := (Nat.divisors n).card

def f (n : ℕ) : ℕ := (Nat.divisors n).sum d

def is_almost_perfect (n : ℕ) : Prop := n > 1 ∧ f n = n

theorem almost_perfect_numbers :
  ∀ n : ℕ, is_almost_perfect n ↔ n = 3 ∨ n = 18 ∨ n = 36 := by sorry

end almost_perfect_numbers_l2508_250819


namespace ed_pets_problem_l2508_250826

theorem ed_pets_problem (dogs : ℕ) (cats : ℕ) (fish : ℕ) : 
  cats = 3 → 
  fish = 2 * (dogs + cats) → 
  dogs + cats + fish = 15 → 
  dogs = 2 := by
sorry

end ed_pets_problem_l2508_250826


namespace product_of_points_on_line_l2508_250814

/-- A line passing through the origin with slope 1/4 -/
def line_k (x y : ℝ) : Prop := y = (1/4) * x

theorem product_of_points_on_line (x y : ℝ) :
  line_k x 8 → line_k 20 y → x * y = 160 := by
  sorry

end product_of_points_on_line_l2508_250814


namespace circle_area_theorem_l2508_250838

theorem circle_area_theorem (c : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ π * r^2 = (c + 4 * Real.sqrt 3) * π / 3) → c = 7 := by
  sorry

end circle_area_theorem_l2508_250838


namespace parabola_point_distance_l2508_250837

theorem parabola_point_distance (x y : ℝ) :
  x^2 = 4*y →  -- Point (x, y) is on the parabola
  (x^2 + (y - 1)^2 = 9) →  -- Distance from (x, y) to focus (0, 1) is 3
  y = 2 := by  -- The y-coordinate of the point is 2
sorry

end parabola_point_distance_l2508_250837


namespace rope_length_problem_l2508_250894

theorem rope_length_problem (short_rope : ℝ) (long_rope : ℝ) : 
  short_rope = 150 →
  short_rope = long_rope * (1 - 1/8) →
  long_rope = 1200/7 := by
sorry

end rope_length_problem_l2508_250894


namespace polar_to_rectangular_on_circle_l2508_250880

/-- Proves that the point (5, 3π/4) in polar coordinates, when converted to rectangular coordinates, lies on the circle x^2 + y^2 = 25. -/
theorem polar_to_rectangular_on_circle :
  let r : ℝ := 5
  let θ : ℝ := 3 * π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  x^2 + y^2 = 25 := by sorry

end polar_to_rectangular_on_circle_l2508_250880


namespace mandy_quarters_l2508_250835

theorem mandy_quarters : 
  ∃ q : ℕ, (40 < q ∧ q < 400) ∧ 
           (q % 6 = 2) ∧ 
           (q % 7 = 2) ∧ 
           (q % 8 = 2) ∧ 
           (q = 170 ∨ q = 338) := by
  sorry

end mandy_quarters_l2508_250835


namespace smallest_a_minus_b_l2508_250865

theorem smallest_a_minus_b (a b n : ℤ) : 
  (a + b < 11) →
  (a > n) →
  (∀ (c d : ℤ), c + d < 11 → c - d ≥ 4) →
  (a - b = 4) →
  (∀ m : ℤ, a > m → m ≤ 6) :=
by sorry

end smallest_a_minus_b_l2508_250865


namespace average_age_of_five_students_l2508_250843

/-- Given a class of 17 students with an average age of 17 years,
    where 9 students have an average age of 16 years,
    and one student is 75 years old,
    prove that the average age of the remaining 5 students is 14 years. -/
theorem average_age_of_five_students
  (total_students : Nat)
  (total_average : ℝ)
  (nine_students : Nat)
  (nine_average : ℝ)
  (old_student_age : ℝ)
  (h1 : total_students = 17)
  (h2 : total_average = 17)
  (h3 : nine_students = 9)
  (h4 : nine_average = 16)
  (h5 : old_student_age = 75)
  : (total_students * total_average - nine_students * nine_average - old_student_age) / (total_students - nine_students - 1) = 14 :=
by sorry

end average_age_of_five_students_l2508_250843


namespace simplify_expression_l2508_250875

theorem simplify_expression (x : ℝ) : 8*x - 3 + 2*x - 7 + 4*x + 15 = 14*x + 5 := by
  sorry

end simplify_expression_l2508_250875


namespace NaClO_molecular_weight_l2508_250897

/-- The atomic weight of sodium in g/mol -/
def sodium_weight : ℝ := 22.99

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The molecular weight of NaClO in g/mol -/
def NaClO_weight : ℝ := sodium_weight + chlorine_weight + oxygen_weight

/-- Theorem stating that the molecular weight of NaClO is approximately 74.44 g/mol -/
theorem NaClO_molecular_weight : 
  ‖NaClO_weight - 74.44‖ < 0.01 := by sorry

end NaClO_molecular_weight_l2508_250897


namespace simplify_power_sum_l2508_250824

theorem simplify_power_sum : 
  -(2^2004) + (-2)^2005 + 2^2006 - 2^2007 = -(2^2004) - 2^2005 + 2^2006 - 2^2007 := by
  sorry

end simplify_power_sum_l2508_250824


namespace equilateral_triangle_area_l2508_250863

/-- The area of an equilateral triangle with base 10 and height 5√3 is 25√3 -/
theorem equilateral_triangle_area : 
  ∀ (base height area : ℝ),
  base = 10 →
  height = 5 * Real.sqrt 3 →
  area = (1 / 2) * base * height →
  area = 25 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l2508_250863


namespace stationary_points_of_f_l2508_250872

def f (x : ℝ) : ℝ := x^3 - 3*x + 2

theorem stationary_points_of_f :
  ∀ x : ℝ, (∃ y : ℝ, y ≠ x ∧ (∀ z : ℝ, z ≠ x → |z - x| < |y - x| → |f z - f x| ≤ |f y - f x|)) ↔ x = 1 ∨ x = -1 := by
  sorry

end stationary_points_of_f_l2508_250872


namespace tank_capacity_l2508_250802

/-- Represents the capacity of a tank and its inlet/outlet properties. -/
structure Tank where
  capacity : ℝ
  outlet_time : ℝ
  inlet_rate : ℝ
  combined_time : ℝ

/-- The tank satisfies the given conditions. -/
def satisfies_conditions (t : Tank) : Prop :=
  t.outlet_time = 5 ∧
  t.inlet_rate = 8 * 60 ∧
  t.combined_time = t.outlet_time + 3

/-- The theorem stating that a tank satisfying the given conditions has a capacity of 6400 litres. -/
theorem tank_capacity (t : Tank) (h : satisfies_conditions t) : t.capacity = 6400 := by
  sorry

end tank_capacity_l2508_250802


namespace polynomial_factor_l2508_250808

-- Define the polynomials
def p (c : ℝ) (x : ℝ) : ℝ := 3 * x^3 + c * x + 9
def f (q : ℝ) (x : ℝ) : ℝ := x^2 + q * x + 3

-- Theorem statement
theorem polynomial_factor (c : ℝ) : 
  (∃ q : ℝ, ∃ r : ℝ → ℝ, ∀ x, p c x = f q x * r x) → c = 0 := by
  sorry

end polynomial_factor_l2508_250808


namespace existence_of_numbers_l2508_250890

theorem existence_of_numbers : ∃ n : ℕ, 
  70 ≤ n ∧ n ≤ 80 ∧ 
  Nat.gcd 30 n = 10 ∧ 
  200 < Nat.lcm 30 n ∧ Nat.lcm 30 n < 300 := by
  sorry

end existence_of_numbers_l2508_250890


namespace number_of_bowls_l2508_250847

/-- The number of bowls on the table -/
def num_bowls : ℕ := sorry

/-- The initial number of grapes in each bowl -/
def initial_grapes : ℕ → ℕ := sorry

/-- The total number of grapes initially -/
def total_initial_grapes : ℕ := sorry

/-- The number of bowls that receive additional grapes -/
def bowls_with_added_grapes : ℕ := 12

/-- The number of grapes added to each of the specified bowls -/
def grapes_added_per_bowl : ℕ := 8

/-- The increase in the average number of grapes across all bowls -/
def average_increase : ℕ := 6

theorem number_of_bowls :
  (total_initial_grapes + bowls_with_added_grapes * grapes_added_per_bowl) / num_bowls =
  total_initial_grapes / num_bowls + average_increase →
  num_bowls = 16 := by sorry

end number_of_bowls_l2508_250847


namespace total_spending_is_48_l2508_250815

/-- Represents the savings and spending pattern for a week -/
structure SavingsPattern where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  friday : ℝ
  thursday_spend_ratio : ℝ
  saturday_spend_ratio : ℝ

/-- Calculates the total spending on Thursday and Saturday -/
def total_spending (pattern : SavingsPattern) : ℝ :=
  let initial_savings := pattern.monday + pattern.tuesday + pattern.wednesday
  let thursday_spending := initial_savings * pattern.thursday_spend_ratio
  let friday_total := initial_savings - thursday_spending + pattern.friday
  let saturday_spending := friday_total * pattern.saturday_spend_ratio
  thursday_spending + saturday_spending

/-- Theorem stating that the total spending on Thursday and Saturday is $48 -/
theorem total_spending_is_48 (pattern : SavingsPattern) 
  (h1 : pattern.monday = 15)
  (h2 : pattern.tuesday = 28)
  (h3 : pattern.wednesday = 13)
  (h4 : pattern.friday = 22)
  (h5 : pattern.thursday_spend_ratio = 0.5)
  (h6 : pattern.saturday_spend_ratio = 0.4) :
  total_spending pattern = 48 := by
  sorry


end total_spending_is_48_l2508_250815


namespace equation_solution_l2508_250813

theorem equation_solution :
  ∃ x : ℝ, x ≠ 0 ∧ (2 / x + (3 / x) / (6 / x) + 2 = 4) ∧ x = 4 / 3 := by
  sorry

end equation_solution_l2508_250813


namespace return_trip_time_l2508_250864

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  p : ℝ  -- Speed of the plane in still air
  w : ℝ  -- Speed of the wind
  d : ℝ  -- Distance between the cities

/-- The conditions of the flight scenario -/
def validFlightScenario (f : FlightScenario) : Prop :=
  f.p > 0 ∧ f.w > 0 ∧ f.d > 0 ∧
  f.d / (f.p - f.w) = 90 ∧
  f.d / (f.p + f.w) = f.d / f.p - 15

/-- The theorem stating that the return trip takes 64 minutes -/
theorem return_trip_time (f : FlightScenario) 
  (h : validFlightScenario f) : 
  f.d / (f.p + f.w) = 64 := by
  sorry

end return_trip_time_l2508_250864


namespace base_five_equals_base_b_l2508_250882

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => digit + base * acc) 0

theorem base_five_equals_base_b (b : Nat) : b > 0 → 
  (base_to_decimal [3, 2] 5 = base_to_decimal [1, 2, 1] b) → b = 4 := by
  sorry

end base_five_equals_base_b_l2508_250882


namespace arithmetic_mean_problem_l2508_250836

theorem arithmetic_mean_problem : 
  let a := 3 / 4
  let b := 5 / 8
  let mean := (a + b) / 2
  3 * mean = 33 / 16 := by
  sorry

end arithmetic_mean_problem_l2508_250836


namespace g_100_zeros_l2508_250854

-- Define g₀(x)
def g₀ (x : ℝ) : ℝ := x + |x - 50| - |x + 50|

-- Define gₙ(x) recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

-- Theorem statement
theorem g_100_zeros :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x : ℝ, x ∈ s ↔ g 100 x = 0 := by
  sorry

end g_100_zeros_l2508_250854


namespace train_length_l2508_250871

/-- Given a train passing a bridge, calculate its length. -/
theorem train_length
  (train_speed : Real) -- Speed of the train in km/hour
  (bridge_length : Real) -- Length of the bridge in meters
  (passing_time : Real) -- Time to pass the bridge in seconds
  (h1 : train_speed = 45) -- Train speed is 45 km/hour
  (h2 : bridge_length = 160) -- Bridge length is 160 meters
  (h3 : passing_time = 41.6) -- Time to pass the bridge is 41.6 seconds
  : Real := by
  sorry

#check train_length

end train_length_l2508_250871


namespace smallest_with_ten_factors_l2508_250861

/-- The number of distinct positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer has exactly ten distinct positive factors -/
def has_ten_factors (n : ℕ+) : Prop := num_factors n = 10

theorem smallest_with_ten_factors :
  ∃ (m : ℕ+), has_ten_factors m ∧ ∀ (k : ℕ+), has_ten_factors k → m ≤ k :=
sorry

end smallest_with_ten_factors_l2508_250861


namespace decimal_21_equals_binary_10101_l2508_250896

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem decimal_21_equals_binary_10101 : 
  to_binary 21 = [true, false, true, false, true] ∧ from_binary [true, false, true, false, true] = 21 := by
  sorry

end decimal_21_equals_binary_10101_l2508_250896


namespace brownie_pieces_l2508_250893

/-- Proves that a 24-inch by 15-inch pan can be divided into exactly 40 pieces of 3-inch by 3-inch brownies. -/
theorem brownie_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_size : ℕ) : 
  pan_length = 24 → pan_width = 15 → piece_size = 3 → 
  (pan_length * pan_width) / (piece_size * piece_size) = 40 := by
  sorry

#check brownie_pieces

end brownie_pieces_l2508_250893


namespace inconsistent_extension_system_l2508_250892

/-- Represents a 4-digit extension number -/
structure Extension :=
  (digits : Fin 4 → Nat)
  (valid : ∀ i, digits i < 10)
  (even : digits 3 % 2 = 0)

/-- The set of 4 specific digits used for extensions -/
def SpecificDigits : Finset Nat := sorry

/-- The set of all valid extensions -/
def AllExtensions : Finset Extension :=
  sorry

theorem inconsistent_extension_system :
  (∀ e ∈ AllExtensions, (∀ i, e.digits i ∈ SpecificDigits)) →
  (Finset.card AllExtensions = 12) →
  False :=
sorry

end inconsistent_extension_system_l2508_250892


namespace constant_speed_distance_time_not_correlation_l2508_250881

/-- A relationship between two variables -/
inductive Relationship
  | Correlation
  | Functional

/-- Represents the relationship between distance, speed, and time for a vehicle moving at constant speed -/
def constant_speed_distance_time_relationship : Relationship :=
  Relationship.Functional

/-- Theorem: The relationship between distance, speed, and time for a vehicle moving at constant speed is not a correlation -/
theorem constant_speed_distance_time_not_correlation :
  constant_speed_distance_time_relationship ≠ Relationship.Correlation :=
by sorry

end constant_speed_distance_time_not_correlation_l2508_250881


namespace sequence_property_characterization_l2508_250884

/-- A sequence satisfies the required property if for any k = 1, ..., n, 
    it contains two numbers equal to k with exactly k numbers between them. -/
def satisfies_property (seq : List ℕ) (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range n, ∃ i j, i < j ∧ j - i = k + 1 ∧ 
    seq.nthLe i (by sorry) = k ∧ seq.nthLe j (by sorry) = k

/-- The main theorem stating the necessary and sufficient condition for n -/
theorem sequence_property_characterization (n : ℕ) :
  (∃ seq : List ℕ, seq.length = 2 * n ∧ satisfies_property seq n) ↔ 
  (∃ l : ℕ, n = 4 * l ∨ n = 4 * l - 1) :=
sorry

end sequence_property_characterization_l2508_250884


namespace song_book_cost_l2508_250839

theorem song_book_cost (total_spent : ℝ) (trumpet_cost : ℝ) (song_book_cost : ℝ) :
  total_spent = 151 →
  trumpet_cost = 145.16 →
  total_spent = trumpet_cost + song_book_cost →
  song_book_cost = 5.84 := by
sorry

end song_book_cost_l2508_250839


namespace rectangular_to_cylindrical_l2508_250860

theorem rectangular_to_cylindrical :
  let x : ℝ := 3
  let y : ℝ := -3 * Real.sqrt 3
  let z : ℝ := 2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 5 * π / 3
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 6 ∧
  θ = 5 * π / 3 ∧
  z = 2 ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ := by
sorry

end rectangular_to_cylindrical_l2508_250860


namespace complex_equation_solution_l2508_250841

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution (x : ℝ) : 
  (1 - i) * (x + i) = 1 + i → x = 0 := by
sorry

end complex_equation_solution_l2508_250841


namespace johns_allowance_theorem_l2508_250811

/-- The fraction of John's remaining allowance spent at the toy store -/
def toy_store_fraction (total_allowance : ℚ) (arcade_fraction : ℚ) (candy_amount : ℚ) : ℚ :=
  let remaining_after_arcade := total_allowance * (1 - arcade_fraction)
  let toy_store_amount := remaining_after_arcade - candy_amount
  toy_store_amount / remaining_after_arcade

/-- Proof that John spent 1/3 of his remaining allowance at the toy store -/
theorem johns_allowance_theorem :
  toy_store_fraction 3.60 (3/5) 0.96 = 1/3 := by
  sorry

end johns_allowance_theorem_l2508_250811


namespace quadratic_properties_l2508_250859

def f (x : ℝ) := x^2 - 4*x + 6

theorem quadratic_properties :
  (∀ x : ℝ, f x = 2 ↔ x = 2) ∧
  (∀ x y : ℝ, x > 2 ∧ y > x → f y > f x) ∧
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m) ∧
  (∀ x : ℝ, f x ≠ 0) :=
by sorry

end quadratic_properties_l2508_250859


namespace election_votes_total_l2508_250874

theorem election_votes_total (votes_A : ℝ) (votes_B : ℝ) (votes_C : ℝ) (votes_D : ℝ) 
  (total_votes : ℝ) :
  votes_A = 0.45 * total_votes →
  votes_B = 0.25 * total_votes →
  votes_C = 0.15 * total_votes →
  votes_D = total_votes - (votes_A + votes_B + votes_C) →
  votes_A - votes_B = 800 →
  total_votes = 4000 := by
  sorry

#check election_votes_total

end election_votes_total_l2508_250874


namespace animal_arrangement_count_l2508_250878

def num_rabbits : ℕ := 5
def num_dogs : ℕ := 3
def num_goats : ℕ := 4
def num_parrots : ℕ := 2
def num_species : ℕ := 4

def total_arrangements : ℕ := Nat.factorial num_species * 
                               Nat.factorial num_rabbits * 
                               Nat.factorial num_dogs * 
                               Nat.factorial num_goats * 
                               Nat.factorial num_parrots

theorem animal_arrangement_count : total_arrangements = 414720 := by
  sorry

end animal_arrangement_count_l2508_250878


namespace no_real_solutions_l2508_250888

theorem no_real_solutions :
  ¬ ∃ y : ℝ, (y - 3*y + 7)^2 + 2 = -2 * |y| := by
  sorry

end no_real_solutions_l2508_250888


namespace rational_square_sum_difference_l2508_250879

theorem rational_square_sum_difference (m n : ℚ) 
  (h1 : (m + n)^2 = 9) 
  (h2 : (m - n)^2 = 1) : 
  m * n = 2 ∧ m^2 + n^2 - m * n = 3 := by
  sorry

end rational_square_sum_difference_l2508_250879


namespace expression_value_at_three_l2508_250828

theorem expression_value_at_three :
  let f (x : ℝ) := (x^2 - 2*x - 8) / (x - 4)
  f 3 = 5 := by sorry

end expression_value_at_three_l2508_250828


namespace no_solution_for_digit_equation_l2508_250877

theorem no_solution_for_digit_equation : 
  ¬ ∃ (x : ℕ), x ≤ 9 ∧ ((x : ℤ) - (10 * x + x) = 801 ∨ (x : ℤ) - (10 * x + x) = 812) := by
  sorry

end no_solution_for_digit_equation_l2508_250877


namespace divideAthletes_eq_56_l2508_250886

/-- The number of ways to divide 10 athletes into two teams of 5 people each,
    given that two specific athletes must be on the same team -/
def divideAthletes : ℕ :=
  Nat.choose 8 3

theorem divideAthletes_eq_56 : divideAthletes = 56 := by
  sorry

end divideAthletes_eq_56_l2508_250886


namespace tadpoles_kept_l2508_250891

theorem tadpoles_kept (total : ℕ) (released_percent : ℚ) (kept : ℕ) : 
  total = 180 → 
  released_percent = 75 / 100 → 
  kept = total - (total * released_percent).floor → 
  kept = 45 := by
sorry

end tadpoles_kept_l2508_250891


namespace bills_final_money_l2508_250844

/-- Calculates Bill's final amount of money after Frank buys pizzas and gives him the rest. -/
theorem bills_final_money (total_initial : ℕ) (pizza_cost : ℕ) (num_pizzas : ℕ) (bills_initial : ℕ) : 
  total_initial = 42 →
  pizza_cost = 11 →
  num_pizzas = 3 →
  bills_initial = 30 →
  bills_initial + (total_initial - (pizza_cost * num_pizzas)) = 39 := by
  sorry

end bills_final_money_l2508_250844


namespace coin_value_difference_l2508_250806

theorem coin_value_difference :
  ∀ (x : ℕ),
  1 ≤ x ∧ x ≤ 3029 →
  (30300 - 9 * 1) - (30300 - 9 * 3029) = 27252 :=
by
  sorry

end coin_value_difference_l2508_250806


namespace building_heights_sum_l2508_250899

/-- The sum of heights of four buildings with specific height relationships -/
theorem building_heights_sum : 
  let tallest : ℝ := 100
  let second : ℝ := tallest / 2
  let third : ℝ := second / 2
  let fourth : ℝ := third / 5
  tallest + second + third + fourth = 180 := by sorry

end building_heights_sum_l2508_250899


namespace parabola_intersection_fixed_point_l2508_250820

-- Define the parabola E
def E (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the lines l₁ and l₂
def l₁ (k₁ x y : ℝ) : Prop := y = k₁*(x - 1)
def l₂ (k₂ x y : ℝ) : Prop := y = k₂*(x - 1)

-- Define the line l
def l (k k₁ k₂ x y : ℝ) : Prop := k*x - y - k*k₁ - k*k₂ = 0

theorem parabola_intersection_fixed_point 
  (p : ℝ) (k₁ k₂ k : ℝ) :
  E p 4 0 ∧ -- This represents y² = 8x, derived from the minimum value condition
  k₁ * k₂ = -3/2 ∧
  k = (4/k₁ - 4/k₂) / ((k₁^2 + 4)/k₁^2 - (k₂^2 + 4)/k₂^2) →
  l k k₁ k₂ 0 (3/2) :=
sorry

end parabola_intersection_fixed_point_l2508_250820


namespace unique_x_with_three_prime_factors_l2508_250810

theorem unique_x_with_three_prime_factors (x n : ℕ) : 
  x = 9^n - 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 61 ∧ q ≠ 61 ∧ 
   x = 2 * p * q * 61 ∧ 
   ∀ r : ℕ, Prime r → r ∣ x → (r = 2 ∨ r = p ∨ r = q ∨ r = 61)) →
  x = 59048 := by
sorry

end unique_x_with_three_prime_factors_l2508_250810


namespace not_necessarily_right_triangle_l2508_250833

theorem not_necessarily_right_triangle 
  (a b c : ℝ) 
  (ha : a^2 = 5) 
  (hb : b^2 = 12) 
  (hc : c^2 = 13) : 
  ¬ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) := by
sorry

end not_necessarily_right_triangle_l2508_250833


namespace robertson_seymour_grid_minor_theorem_l2508_250816

-- Define a graph type
def Graph := Type

-- Define treewidth for a graph
def treewidth (G : Graph) : ℕ := sorry

-- Define the concept of a minor for graphs
def is_minor (H G : Graph) : Prop := sorry

-- Define a grid graph
def grid_graph (r : ℕ) : Graph := sorry

theorem robertson_seymour_grid_minor_theorem :
  ∀ r : ℕ, ∃ k : ℕ, ∀ G : Graph, treewidth G ≥ k → is_minor (grid_graph r) G := by
  sorry

end robertson_seymour_grid_minor_theorem_l2508_250816


namespace quadratic_equation_rewrite_l2508_250898

theorem quadratic_equation_rewrite :
  ∃ (a b c : ℝ), a = 2 ∧ b = -4 ∧ c = 7 ∧
  ∀ x, 2 * x^2 + 7 = 4 * x ↔ a * x^2 + b * x + c = 0 :=
by sorry

end quadratic_equation_rewrite_l2508_250898


namespace shortest_tangent_is_sqrt_449_l2508_250856

/-- Circle C₁ with center (8, 3) and radius 7 -/
def C₁ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 8)^2 + (p.2 - 3)^2 = 49}

/-- Circle C₂ with center (-12, -4) and radius 5 -/
def C₂ : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + 12)^2 + (p.2 + 4)^2 = 25}

/-- The length of the shortest line segment PQ tangent to C₁ at P and C₂ at Q -/
def shortest_tangent_length (C₁ C₂ : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem stating that the shortest tangent length between C₁ and C₂ is √449 -/
theorem shortest_tangent_is_sqrt_449 : 
  shortest_tangent_length C₁ C₂ = Real.sqrt 449 := by sorry

end shortest_tangent_is_sqrt_449_l2508_250856


namespace train_journey_time_l2508_250831

/-- Calculate the total travel time for a train journey with multiple stops and varying speeds -/
theorem train_journey_time (d1 d2 d3 : ℝ) (v1 v2 v3 : ℝ) (t1 t2 : ℝ) :
  d1 = 30 →
  d2 = 40 →
  d3 = 50 →
  v1 = 60 →
  v2 = 40 →
  v3 = 80 →
  t1 = 10 / 60 →
  t2 = 5 / 60 →
  (d1 / v1 + t1 + d2 / v2 + t2 + d3 / v3) * 60 = 142.5 :=
by sorry

end train_journey_time_l2508_250831


namespace quadratic_minimum_l2508_250818

theorem quadratic_minimum : ∃ (min : ℝ), 
  (∀ x : ℝ, x^2 + 12*x + 18 ≥ min) ∧ 
  (∃ x : ℝ, x^2 + 12*x + 18 = min) ∧
  (min = -18) := by
  sorry

end quadratic_minimum_l2508_250818


namespace no_solution_x4_plus_6_eq_y3_l2508_250858

theorem no_solution_x4_plus_6_eq_y3 :
  ∀ (x y : ℤ), (x^4 + 6) % 13 ≠ y^3 % 13 := by
  sorry

end no_solution_x4_plus_6_eq_y3_l2508_250858


namespace sequence_monotonicity_l2508_250867

/-- A sequence a_n is defined as a_n = -n^2 + tn, where n is a positive natural number and t is a constant real number. The sequence is monotonically decreasing. -/
theorem sequence_monotonicity (t : ℝ) : 
  (∀ n : ℕ+, ∀ m : ℕ+, n < m → (-n^2 + t * n) > (-m^2 + t * m)) → 
  t < 3 := by
sorry

end sequence_monotonicity_l2508_250867


namespace monicas_first_class_size_l2508_250832

/-- Represents the number of students in Monica's classes -/
structure MonicasClasses where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ

/-- Theorem stating the number of students in Monica's first class -/
theorem monicas_first_class_size (c : MonicasClasses) : c.first = 20 :=
  by
  have h1 : c.second = 25 := by sorry
  have h2 : c.third = 25 := by sorry
  have h3 : c.fourth = c.first / 2 := by sorry
  have h4 : c.fifth = 28 := by sorry
  have h5 : c.sixth = 28 := by sorry
  have h6 : c.first + c.second + c.third + c.fourth + c.fifth + c.sixth = 136 := by sorry
  sorry

#check monicas_first_class_size

end monicas_first_class_size_l2508_250832


namespace potato_bag_weight_l2508_250840

def bag_weight : ℝ → Prop := λ w => w = 36 / (w / 2)

theorem potato_bag_weight : ∃ w : ℝ, bag_weight w ∧ w = 36 := by
  sorry

end potato_bag_weight_l2508_250840


namespace largest_non_representable_l2508_250849

def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

def is_representable (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 0 ∧ is_composite m ∧ n = 30 * k + m

theorem largest_non_representable : 
  (∀ n : ℕ, n > 157 → is_representable n) ∧
  ¬is_representable 157 :=
sorry

end largest_non_representable_l2508_250849


namespace unique_divisible_by_1375_l2508_250851

theorem unique_divisible_by_1375 : 
  ∃! n : ℕ, 
    (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ n = 700000 + 10000 * x + 3600 + 10 * y + 5) ∧ 
    n % 1375 = 0 :=
by
  sorry

end unique_divisible_by_1375_l2508_250851


namespace max_value_inequality_l2508_250848

theorem max_value_inequality (x y : ℝ) :
  (x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 26 := by
  sorry

end max_value_inequality_l2508_250848


namespace downstream_speed_is_48_l2508_250870

/-- The speed of a man rowing in a stream -/
structure RowingSpeed :=
  (upstream : ℝ)
  (stillWater : ℝ)

/-- Calculate the downstream speed of a man rowing in a stream -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  s.stillWater + (s.stillWater - s.upstream)

/-- Theorem: Given the upstream and still water speeds, the downstream speed is 48 -/
theorem downstream_speed_is_48 (s : RowingSpeed) 
    (h1 : s.upstream = 34) 
    (h2 : s.stillWater = 41) : 
  downstreamSpeed s = 48 := by
  sorry

end downstream_speed_is_48_l2508_250870
