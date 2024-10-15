import Mathlib

namespace NUMINAMATH_CALUDE_minimum_sum_geometric_mean_l1485_148578

theorem minimum_sum_geometric_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) (hgm : Real.sqrt (a * b) = 1) :
  2 * (a + b) ≥ 4 ∧ (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ Real.sqrt (x * y) = 1 ∧ 2 * (x + y) = 4) := by
  sorry

end NUMINAMATH_CALUDE_minimum_sum_geometric_mean_l1485_148578


namespace NUMINAMATH_CALUDE_time_period_is_12_hours_l1485_148500

/-- The time period in hours for a given population net increase -/
def time_period (birth_rate : ℚ) (death_rate : ℚ) (net_increase : ℕ) : ℚ :=
  let net_rate_per_second : ℚ := (birth_rate - death_rate) / 2
  let seconds : ℚ := net_increase / net_rate_per_second
  seconds / 3600

/-- Theorem stating that given the problem conditions, the time period is 12 hours -/
theorem time_period_is_12_hours :
  time_period 8 6 86400 = 12 := by
  sorry

end NUMINAMATH_CALUDE_time_period_is_12_hours_l1485_148500


namespace NUMINAMATH_CALUDE_cos_a_sin_b_value_l1485_148564

theorem cos_a_sin_b_value (A B : Real) (hA : 0 < A ∧ A < Real.pi / 2) (hB : 0 < B ∧ B < Real.pi / 2)
  (h : (4 + Real.tan A ^ 2) * (5 + Real.tan B ^ 2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
  Real.cos A * Real.sin B = Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_a_sin_b_value_l1485_148564


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1485_148507

theorem angle_measure_proof (x : ℝ) : 
  (x + (3 * x + 10) = 90) → x = 20 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1485_148507


namespace NUMINAMATH_CALUDE_point_transformation_l1485_148535

def rotate_z (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def initial_point : ℝ × ℝ × ℝ := (2, 2, -1)

def final_point : ℝ × ℝ × ℝ := (-2, 2, -1)

theorem point_transformation :
  (reflect_xy ∘ rotate_z ∘ reflect_yz ∘ reflect_xy ∘ rotate_z) initial_point = final_point := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l1485_148535


namespace NUMINAMATH_CALUDE_f_integer_values_l1485_148548

def f (a b : ℕ+) : ℚ :=
  (a.val ^ 2 + b.val ^ 2 + a.val * b.val) / (a.val * b.val - 1)

theorem f_integer_values (a b : ℕ+) (h : a.val * b.val ≠ 1) :
  (∃ n : ℤ, f a b = n) → (f a b = 4 ∨ f a b = 7) := by
  sorry

end NUMINAMATH_CALUDE_f_integer_values_l1485_148548


namespace NUMINAMATH_CALUDE_intersection_distance_two_points_condition_l1485_148503

-- Define the line l in polar coordinates
def line_l (ρ θ : ℝ) : Prop := ρ * (3 * Real.cos θ - 4 * Real.sin θ) = 2

-- Define the curve C in polar coordinates
def curve_C (ρ m : ℝ) : Prop := ρ = m ∧ m > 0

-- Theorem for part (1)
theorem intersection_distance : 
  ∃ (ρ : ℝ), line_l ρ 0 ∧ ρ = 2/3 :=
sorry

-- Theorem for part (2)
theorem two_points_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = m^2 ∧ 
   (|3*x - 4*y - 2| / 5 = 1/5 ∨ |3*x - 4*y - 2| / 5 = 1/5) ∧
   (∀ (x' y' : ℝ), x'^2 + y'^2 = m^2 ∧ |3*x' - 4*y' - 2| / 5 = 1/5 → 
     (x' = x ∧ y' = y) ∨ (x' = -x ∧ y' = -y))) ↔ 
  (1/5 < m ∧ m < 3/5) :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_two_points_condition_l1485_148503


namespace NUMINAMATH_CALUDE_kenneths_earnings_l1485_148582

theorem kenneths_earnings (spent_percentage : ℝ) (remaining_amount : ℝ) (total_earnings : ℝ) : 
  spent_percentage = 10 →
  remaining_amount = 405 →
  (100 - spent_percentage) / 100 * total_earnings = remaining_amount →
  total_earnings = 450 := by
sorry

end NUMINAMATH_CALUDE_kenneths_earnings_l1485_148582


namespace NUMINAMATH_CALUDE_second_year_percentage_approx_l1485_148575

def numeric_methods_students : ℕ := 240
def automatic_control_students : ℕ := 423
def both_subjects_students : ℕ := 134
def total_faculty_students : ℕ := 663

def second_year_students : ℕ := numeric_methods_students + automatic_control_students - both_subjects_students

def percentage_second_year : ℚ := (second_year_students : ℚ) / (total_faculty_students : ℚ) * 100

theorem second_year_percentage_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |percentage_second_year - 79.79| < ε :=
sorry

end NUMINAMATH_CALUDE_second_year_percentage_approx_l1485_148575


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l1485_148545

theorem imaginary_part_of_complex_product (i : ℂ) : i * i = -1 → Complex.im (i * (1 + i) * i) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l1485_148545


namespace NUMINAMATH_CALUDE_remainder_problem_l1485_148566

theorem remainder_problem (t : ℕ) :
  let n : ℤ := 209 * t + 23
  (n % 19 = 4) ∧ (n % 11 = 1) := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1485_148566


namespace NUMINAMATH_CALUDE_number_conditions_l1485_148571

theorem number_conditions (x y : ℝ) : 
  (0.65 * x > 26) → 
  (0.4 * y < -3) → 
  ((x - y)^2 ≥ 100) → 
  (x > 40 ∧ y < -7.5) := by
sorry

end NUMINAMATH_CALUDE_number_conditions_l1485_148571


namespace NUMINAMATH_CALUDE_calculation_proof_l1485_148573

theorem calculation_proof :
  (Real.sqrt 8 - abs (-2) + (1/3)⁻¹ - 4 * Real.cos (45 * π / 180)) = 1 ∧
  ∀ x : ℝ, (x - 2)^2 - x*(x - 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1485_148573


namespace NUMINAMATH_CALUDE_r_2011_equals_2_l1485_148595

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

def r (n : ℕ) : ℕ := fib n % 3

theorem r_2011_equals_2 : r 2011 = 2 := by
  sorry

end NUMINAMATH_CALUDE_r_2011_equals_2_l1485_148595


namespace NUMINAMATH_CALUDE_initial_walnut_count_l1485_148524

/-- The number of walnut trees initially in the park -/
def initial_walnut_trees : ℕ := sorry

/-- The number of walnut trees cut down -/
def cut_trees : ℕ := 13

/-- The number of walnut trees remaining after cutting -/
def remaining_trees : ℕ := 29

/-- The number of orange trees in the park -/
def orange_trees : ℕ := 12

theorem initial_walnut_count :
  initial_walnut_trees = remaining_trees + cut_trees :=
by sorry

end NUMINAMATH_CALUDE_initial_walnut_count_l1485_148524


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_is_135_l1485_148588

/-- The measure of one interior angle of a regular octagon in degrees -/
def regular_octagon_interior_angle : ℝ := 135

/-- Theorem: The measure of one interior angle of a regular octagon is 135 degrees -/
theorem regular_octagon_interior_angle_is_135 :
  regular_octagon_interior_angle = 135 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_is_135_l1485_148588


namespace NUMINAMATH_CALUDE_nancy_carrots_l1485_148515

/-- The total number of carrots Nancy has after two days of picking and throwing out some -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) : ℕ :=
  initial - thrown_out + picked_next_day

/-- Theorem stating that Nancy's total carrots is 31 given the specific numbers in the problem -/
theorem nancy_carrots : total_carrots 12 2 21 = 31 := by
  sorry

end NUMINAMATH_CALUDE_nancy_carrots_l1485_148515


namespace NUMINAMATH_CALUDE_prob_at_most_one_eq_seven_twenty_sevenths_l1485_148590

/-- The probability of making a basket -/
def p : ℚ := 2/3

/-- The number of attempts -/
def n : ℕ := 3

/-- The probability of making exactly k successful shots in n attempts -/
def binomial_prob (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of making at most 1 successful shot in 3 attempts -/
def prob_at_most_one : ℚ :=
  binomial_prob 0 + binomial_prob 1

theorem prob_at_most_one_eq_seven_twenty_sevenths : 
  prob_at_most_one = 7/27 := by sorry

end NUMINAMATH_CALUDE_prob_at_most_one_eq_seven_twenty_sevenths_l1485_148590


namespace NUMINAMATH_CALUDE_probability_not_hearing_favorite_song_l1485_148514

-- Define the number of songs
def num_songs : ℕ := 8

-- Define the length of the shortest song
def shortest_song_length : ℕ := 1

-- Define the length of the favorite song
def favorite_song_length : ℕ := 5

-- Define the duration we're considering
def considered_duration : ℕ := 7

-- Function to calculate song length based on its position
def song_length (position : ℕ) : ℕ :=
  shortest_song_length + position - 1

-- Theorem stating the probability of not hearing every second of the favorite song
theorem probability_not_hearing_favorite_song :
  let total_arrangements := num_songs.factorial
  let favorable_arrangements := (num_songs - 1).factorial + (num_songs - 2).factorial
  (total_arrangements - favorable_arrangements) / total_arrangements = 6 / 7 :=
sorry

end NUMINAMATH_CALUDE_probability_not_hearing_favorite_song_l1485_148514


namespace NUMINAMATH_CALUDE_intersection_M_N_l1485_148532

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1485_148532


namespace NUMINAMATH_CALUDE_tangency_line_parallel_to_common_tangent_l1485_148597

/-- Given three parabolas p₁, p₂, and p₃, where p₁ and p₂ both touch p₃,
    the line connecting the points of tangency of p₁ and p₂ with p₃
    is parallel to the common tangent of p₁ and p₂. -/
theorem tangency_line_parallel_to_common_tangent
  (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) :
  let p₁ := fun x => -x^2 + b₁ * x + c₁
  let p₂ := fun x => -x^2 + b₂ * x + c₂
  let p₃ := fun x => x^2 + b₃ * x + c₃
  let x₁ := (b₁ - b₃) / 4
  let y₁ := p₃ x₁
  let x₂ := (b₂ - b₃) / 4
  let y₂ := p₃ x₂
  let m_tangency := (y₂ - y₁) / (x₂ - x₁)
  let m_common_tangent := (4 * (c₁ - c₂) - 2 * b₃ * (b₁ - b₂)) / (2 * (b₂ - b₁))
  (b₃ - b₁)^2 = 8 * (c₃ - c₁) →
  (b₃ - b₂)^2 = 8 * (c₃ - c₂) →
  m_tangency = m_common_tangent :=
by sorry

end NUMINAMATH_CALUDE_tangency_line_parallel_to_common_tangent_l1485_148597


namespace NUMINAMATH_CALUDE_f_properties_l1485_148577

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem for monotonicity intervals and extreme values
theorem f_properties :
  (∀ x < -1, HasDerivAt f (f x) x ∧ 0 < (deriv f x)) ∧
  (∀ x ∈ Set.Ioo (-1) 1, HasDerivAt f (f x) x ∧ (deriv f x) < 0) ∧
  (∀ x > 1, HasDerivAt f (f x) x ∧ 0 < (deriv f x)) ∧
  (∀ x ∈ Set.Icc (-3) 2, f x ≥ -18) ∧
  (∀ x ∈ Set.Icc (-3) 2, f x ≤ 2) ∧
  (f (-3) = -18) ∧
  (f (-1) = 2 ∨ f 2 = 2) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l1485_148577


namespace NUMINAMATH_CALUDE_expression_simplification_l1485_148530

theorem expression_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 - 11*x + 13*x^2 - 15 + 17*x + 19*x^2 = 25*x^2 + x - 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1485_148530


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1485_148583

theorem equilateral_triangle_perimeter (side_length : ℝ) (h : side_length = 7) :
  3 * side_length = 21 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1485_148583


namespace NUMINAMATH_CALUDE_range_of_a_l1485_148558

-- Define the function f(x)
def f (a x : ℝ) : ℝ := x^2 + abs (x - a)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a (1/2) ≥ f a x → x = 1/2) →
  (∀ x : ℝ, f a (-1/2) ≥ f a x → x = -1/2) →
  a > -1/2 ∧ a < 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1485_148558


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1485_148599

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given condition for the arithmetic sequence -/
def SequenceCondition (a : ℕ → ℝ) : Prop :=
  a 2 + 2 * a 6 + a 10 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SequenceCondition a) : 
  a 3 + a 9 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1485_148599


namespace NUMINAMATH_CALUDE_sin_taylor_expansion_at_3_l1485_148534

open Complex

/-- Taylor series expansion of sine function around z = 3 -/
theorem sin_taylor_expansion_at_3 (z : ℂ) : 
  sin z = (sin 3 * (∑' n, ((-1)^n / (2*n).factorial : ℂ) * (z - 3)^(2*n))) + 
          (cos 3 * (∑' n, ((-1)^n / (2*n + 1).factorial : ℂ) * (z - 3)^(2*n + 1))) := by
  sorry

end NUMINAMATH_CALUDE_sin_taylor_expansion_at_3_l1485_148534


namespace NUMINAMATH_CALUDE_solve_for_a_l1485_148580

theorem solve_for_a (x a : ℝ) (h1 : 2 * x - a + 5 = 0) (h2 : x = -2) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1485_148580


namespace NUMINAMATH_CALUDE_alternating_arithmetic_series_sum_l1485_148553

def alternatingArithmeticSeries (a₁ : ℤ) (d : ℤ) (aₙ : ℤ) : ℤ :=
  let n := (aₙ - a₁) / d + 1
  let pairs := (n - 1) / 2
  let pairSum := -d
  let leftover := if n % 2 = 0 then 0 else aₙ
  pairs * pairSum + leftover

theorem alternating_arithmetic_series_sum :
  alternatingArithmeticSeries 2 3 56 = 29 := by sorry

end NUMINAMATH_CALUDE_alternating_arithmetic_series_sum_l1485_148553


namespace NUMINAMATH_CALUDE_forty_percent_of_sixty_minus_four_fifths_of_twenty_five_l1485_148556

theorem forty_percent_of_sixty_minus_four_fifths_of_twenty_five :
  (40 / 100 * 60) - (4 / 5 * 25) = 4 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_sixty_minus_four_fifths_of_twenty_five_l1485_148556


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1485_148538

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  de : ℝ
  ef : ℝ
  df : ℝ
  de_eq : de = 5
  ef_eq : ef = 12
  df_eq : df = 13
  right_angle : de^2 + ef^2 = df^2

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side_length : ℝ
  on_df : side_length ≤ t.df
  on_de : side_length ≤ t.de
  on_ef : side_length ≤ t.ef

/-- The theorem stating the side length of the inscribed square -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 10140 / 229 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l1485_148538


namespace NUMINAMATH_CALUDE_alphabet_letter_count_l1485_148518

theorem alphabet_letter_count (total : ℕ) (both : ℕ) (line_only : ℕ) :
  total = 50 →
  both = 16 →
  line_only = 30 →
  ∃ (dot_only : ℕ),
    dot_only = total - (both + line_only) ∧
    dot_only = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_alphabet_letter_count_l1485_148518


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l1485_148526

/-- A parabola with vertex at the origin, axis of symmetry along the x-axis,
    and passing through the point (-4, 4) has the standard equation y² = -4x. -/
theorem parabola_standard_equation :
  ∀ (f : ℝ → ℝ),
  (∀ x y : ℝ, f x = y ↔ y^2 = -4*x) →  -- Standard equation of the parabola
  f 0 = 0 →                            -- Vertex at the origin
  (∀ x : ℝ, f x = f (-x)) →            -- Axis of symmetry along x-axis
  f (-4) = 4 →                         -- Passes through (-4, 4)
  ∀ x y : ℝ, f x = y ↔ y^2 = -4*x :=   -- Conclusion: standard equation
by sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l1485_148526


namespace NUMINAMATH_CALUDE_painted_cube_probability_l1485_148584

/-- Represents a 5x5x5 cube with three adjacent faces painted -/
structure PaintedCube :=
  (size : ℕ)
  (total_cubes : ℕ)
  (painted_faces : ℕ)

/-- Calculates the number of unit cubes with exactly three painted faces -/
def three_painted_faces (cube : PaintedCube) : ℕ := sorry

/-- Calculates the number of unit cubes with exactly one painted face -/
def one_painted_face (cube : PaintedCube) : ℕ := sorry

/-- Calculates the total number of ways to choose two unit cubes -/
def total_choices (cube : PaintedCube) : ℕ := sorry

/-- Calculates the number of ways to choose one cube with three painted faces and one with one painted face -/
def favorable_choices (cube : PaintedCube) : ℕ := sorry

/-- The main theorem stating the probability -/
theorem painted_cube_probability (cube : PaintedCube) 
  (h1 : cube.size = 5)
  (h2 : cube.total_cubes = 125)
  (h3 : cube.painted_faces = 3) :
  (favorable_choices cube : ℚ) / (total_choices cube : ℚ) = 8 / 235 := by sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l1485_148584


namespace NUMINAMATH_CALUDE_simplify_fraction_l1485_148551

theorem simplify_fraction : (120 : ℚ) / 2160 = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1485_148551


namespace NUMINAMATH_CALUDE_circle_equation_solution_l1485_148579

theorem circle_equation_solution :
  ∃! (x y : ℝ), (x - 12)^2 + (y - 13)^2 + (x - y)^2 = 1/3 ∧ 
  x = 37/3 ∧ y = 38/3 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_solution_l1485_148579


namespace NUMINAMATH_CALUDE_race_heartbeats_l1485_148586

/-- Calculates the total number of heartbeats during a race. -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  (race_distance * pace * heart_rate)

/-- Proves that the total number of heartbeats during a 30-mile race is 28800,
    given the specified heart rate and pace. -/
theorem race_heartbeats :
  total_heartbeats 160 6 30 = 28800 := by
  sorry

end NUMINAMATH_CALUDE_race_heartbeats_l1485_148586


namespace NUMINAMATH_CALUDE_farm_distance_is_six_l1485_148505

/-- Represents the distance to the farm given the conditions of Bobby's trips -/
def distance_to_farm (initial_gas : ℝ) (supermarket_distance : ℝ) (partial_farm_trip : ℝ) 
  (final_gas : ℝ) (miles_per_gallon : ℝ) : ℝ :=
  let total_miles_driven := (initial_gas - final_gas) * miles_per_gallon
  let known_miles := 2 * supermarket_distance + 2 * partial_farm_trip
  total_miles_driven - known_miles

/-- Theorem stating that the distance to the farm is 6 miles -/
theorem farm_distance_is_six :
  distance_to_farm 12 5 2 2 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_farm_distance_is_six_l1485_148505


namespace NUMINAMATH_CALUDE_min_m_value_l1485_148552

theorem min_m_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 2*y + 2/x + 1/y = 6) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y + 2/x + 1/y = 6 → m ≥ x + 2*y) ∧ 
  (∀ (m' : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y + 2/x + 1/y = 6 → m' ≥ x + 2*y) → m' ≥ m) ∧
  m = 4 :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l1485_148552


namespace NUMINAMATH_CALUDE_max_abs_z_on_circle_l1485_148567

open Complex

theorem max_abs_z_on_circle (z : ℂ) : 
  (abs (z - I) = abs (3 - 4*I)) → (abs z ≤ 6) ∧ (∃ w : ℂ, abs (w - I) = abs (3 - 4*I) ∧ abs w = 6) := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_on_circle_l1485_148567


namespace NUMINAMATH_CALUDE_sum_even_factors_630_eq_1248_l1485_148554

/-- The sum of all positive even factors of 630 -/
def sum_even_factors_630 : ℕ := sorry

/-- 630 is the number we're examining -/
def n : ℕ := 630

/-- Theorem stating that the sum of all positive even factors of 630 is 1248 -/
theorem sum_even_factors_630_eq_1248 : sum_even_factors_630 = 1248 := by sorry

end NUMINAMATH_CALUDE_sum_even_factors_630_eq_1248_l1485_148554


namespace NUMINAMATH_CALUDE_largest_base5_five_digit_in_base10_l1485_148531

def largest_base5_five_digit : ℕ := 4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_five_digit_in_base10 : 
  largest_base5_five_digit = 3124 := by sorry

end NUMINAMATH_CALUDE_largest_base5_five_digit_in_base10_l1485_148531


namespace NUMINAMATH_CALUDE_equation_has_real_root_l1485_148572

-- Define the polynomial function
def f (K x : ℝ) : ℝ := K^2 * (x - 1) * (x - 2) * (x - 3) - x

-- Theorem statement
theorem equation_has_real_root :
  ∀ K : ℝ, ∃ x : ℝ, f K x = 0 :=
sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l1485_148572


namespace NUMINAMATH_CALUDE_perpendicular_vectors_trig_equality_l1485_148513

/-- Given two perpendicular vectors a and b, prove that 
    (sin³α + cos³α) / (sinα - cosα) = 9/5 -/
theorem perpendicular_vectors_trig_equality 
  (a b : ℝ × ℝ) 
  (h1 : a = (4, -2)) 
  (h2 : ∃ α : ℝ, b = (Real.cos α, Real.sin α)) 
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  ∃ α : ℝ, (Real.sin α)^3 + (Real.cos α)^3 = 9/5 * ((Real.sin α) - (Real.cos α)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_trig_equality_l1485_148513


namespace NUMINAMATH_CALUDE_minimum_score_for_english_l1485_148542

/-- Given the average score of two subjects and a desired average for three subjects,
    calculate the minimum score needed for the third subject. -/
def minimum_third_score (avg_two : ℝ) (desired_avg : ℝ) : ℝ :=
  3 * desired_avg - 2 * avg_two

theorem minimum_score_for_english (avg_two : ℝ) (desired_avg : ℝ)
  (h1 : avg_two = 90)
  (h2 : desired_avg ≥ 92) :
  minimum_third_score avg_two desired_avg ≥ 96 :=
sorry

end NUMINAMATH_CALUDE_minimum_score_for_english_l1485_148542


namespace NUMINAMATH_CALUDE_pet_store_cats_l1485_148501

theorem pet_store_cats (initial_siamese : ℕ) (cats_sold : ℕ) (cats_remaining : ℕ) 
  (h1 : initial_siamese = 13)
  (h2 : cats_sold = 10)
  (h3 : cats_remaining = 8) :
  ∃ initial_house : ℕ, 
    initial_house = 5 ∧ 
    initial_siamese + initial_house - cats_sold = cats_remaining :=
  sorry

end NUMINAMATH_CALUDE_pet_store_cats_l1485_148501


namespace NUMINAMATH_CALUDE_quadratic_solution_l1485_148508

theorem quadratic_solution (c : ℝ) : 
  (18^2 + 12*18 + c = 0) → 
  (∃ x : ℝ, x^2 + 12*x + c = 0 ∧ x ≠ 18) → 
  ((-30)^2 + 12*(-30) + c = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1485_148508


namespace NUMINAMATH_CALUDE_range_of_m_l1485_148509

def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m :
  (∀ m : ℝ, ¬(p m ∧ q m)) →
  (∀ m : ℝ, q m) →
  ∀ m : ℝ, 1 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1485_148509


namespace NUMINAMATH_CALUDE_probability_y_leq_x_pow_five_l1485_148549

/-- The probability that y ≤ x^5 when x and y are uniformly distributed over [0,1] -/
theorem probability_y_leq_x_pow_five : Real := by
  -- Define x and y as random variables uniformly distributed over [0,1]
  -- Calculate the probability that y ≤ x^5
  -- Prove that this probability is equal to 1/6
  sorry

#check probability_y_leq_x_pow_five

end NUMINAMATH_CALUDE_probability_y_leq_x_pow_five_l1485_148549


namespace NUMINAMATH_CALUDE_wine_without_cork_cost_l1485_148511

def bottle_with_cork : ℝ := 2.10
def cork : ℝ := 2.05

theorem wine_without_cork_cost (bottle_without_cork : ℝ) 
  (h1 : bottle_without_cork > cork) : 
  bottle_without_cork - cork > 0.05 := by
  sorry

end NUMINAMATH_CALUDE_wine_without_cork_cost_l1485_148511


namespace NUMINAMATH_CALUDE_fencing_calculation_l1485_148559

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_calculation (area : ℝ) (uncovered_side : ℝ) : 
  area = 50 → uncovered_side = 20 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    2 * width + uncovered_side = 25 := by
  sorry

end NUMINAMATH_CALUDE_fencing_calculation_l1485_148559


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1485_148529

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  (b/a = Real.sqrt 2) →                   -- Slope of asymptote
  (a^2 + b^2 = 3) →                       -- Right focus coincides with parabola focus
  (∀ (x y : ℝ), x^2 - y^2/2 = 1 ↔ x^2/a^2 - y^2/b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1485_148529


namespace NUMINAMATH_CALUDE_probability_of_mixed_selection_l1485_148589

theorem probability_of_mixed_selection (n_boys n_girls n_select : ℕ) :
  n_boys = 5 →
  n_girls = 2 →
  n_select = 3 →
  (Nat.choose (n_boys + n_girls) n_select - Nat.choose n_boys n_select - Nat.choose n_girls n_select) / Nat.choose (n_boys + n_girls) n_select = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_mixed_selection_l1485_148589


namespace NUMINAMATH_CALUDE_quadratic_roots_l1485_148546

theorem quadratic_roots (a b c : ℝ) (ha : a ≠ 0) 
  (h1 : a + b + c = 0) (h2 : a - b + c = 0) :
  ∃ (x y : ℝ), x = 1 ∧ y = -1 ∧ 
  (∀ z : ℝ, a * z^2 + b * z + c = 0 ↔ z = x ∨ z = y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1485_148546


namespace NUMINAMATH_CALUDE_first_part_games_l1485_148506

/-- Prove that the number of games in the first part of the season is 100 -/
theorem first_part_games (total_games : ℕ) (first_win_rate remaining_win_rate overall_win_rate : ℚ) : 
  total_games = 175 →
  first_win_rate = 85/100 →
  remaining_win_rate = 1/2 →
  overall_win_rate = 7/10 →
  ∃ (x : ℕ), x = 100 ∧ 
    first_win_rate * x + remaining_win_rate * (total_games - x) = overall_win_rate * total_games :=
by sorry

end NUMINAMATH_CALUDE_first_part_games_l1485_148506


namespace NUMINAMATH_CALUDE_ellipse_m_range_l1485_148510

theorem ellipse_m_range (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (2 + m)) - (y^2 / (m + 1)) = 1 ∧ 
   ((2 + m > 0 ∧ -(m + 1) > 0) ∨ (-(m + 1) > 0 ∧ 2 + m > 0))) ↔ 
  (m ∈ Set.Ioo (-2 : ℝ) (-3/2) ∪ Set.Ioo (-3/2 : ℝ) (-1)) := by
sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l1485_148510


namespace NUMINAMATH_CALUDE_banana_arrangements_l1485_148591

theorem banana_arrangements : 
  let total_letters : ℕ := 6
  let freq_b : ℕ := 1
  let freq_n : ℕ := 2
  let freq_a : ℕ := 3
  (total_letters = freq_b + freq_n + freq_a) →
  (Nat.factorial total_letters / (Nat.factorial freq_b * Nat.factorial freq_n * Nat.factorial freq_a) = 60) := by
sorry

end NUMINAMATH_CALUDE_banana_arrangements_l1485_148591


namespace NUMINAMATH_CALUDE_nested_fourth_root_l1485_148563

theorem nested_fourth_root (M : ℝ) (h : M > 1) :
  (M * (M * (M * M^(1/4))^(1/4))^(1/4))^(1/4) = M^(21/64) := by
  sorry

end NUMINAMATH_CALUDE_nested_fourth_root_l1485_148563


namespace NUMINAMATH_CALUDE_distinct_sequences_count_l1485_148536

/-- The number of sides on the die -/
def die_sides : ℕ := 6

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 10

/-- The number of distinct sequences when rolling a die -/
def num_sequences : ℕ := die_sides ^ num_rolls

theorem distinct_sequences_count : num_sequences = 60466176 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sequences_count_l1485_148536


namespace NUMINAMATH_CALUDE_first_hour_distance_correct_l1485_148550

/-- The distance traveled by a car in the first hour, given that its speed increases by 2 km/h every hour and it travels 492 km in 12 hours -/
def first_hour_distance : ℝ :=
  let speed_increase : ℝ := 2
  let total_hours : ℕ := 12
  let total_distance : ℝ := 492
  30

/-- Theorem stating that the first hour distance is correct -/
theorem first_hour_distance_correct :
  let speed_increase : ℝ := 2
  let total_hours : ℕ := 12
  let total_distance : ℝ := 492
  (first_hour_distance + total_hours * (total_hours - 1) / 2 * speed_increase) * total_hours / 2 = total_distance :=
by
  sorry

#eval first_hour_distance

end NUMINAMATH_CALUDE_first_hour_distance_correct_l1485_148550


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1485_148547

theorem constant_term_expansion (x : ℝ) : 
  let expression := (x - 4 + 4 / x)^3
  ∃ (a b c : ℝ), expression = a * x^3 + b * x^2 + c * x - 160
  := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1485_148547


namespace NUMINAMATH_CALUDE_covered_boards_l1485_148592

/-- Represents a modified checkerboard with one corner removed. -/
structure ModifiedBoard :=
  (rows : Nat)
  (cols : Nat)

/-- Checks if a modified board can be completely covered by dominoes. -/
def can_be_covered (board : ModifiedBoard) : Prop :=
  let total_squares := board.rows * board.cols - 1
  (total_squares % 2 = 0) ∧ 
  (board.rows ≥ 2) ∧ 
  (board.cols ≥ 2)

/-- Theorem stating which modified boards can be covered. -/
theorem covered_boards :
  (can_be_covered ⟨5, 5⟩) ∧
  (can_be_covered ⟨7, 3⟩) ∧
  ¬(can_be_covered ⟨4, 5⟩) ∧
  ¬(can_be_covered ⟨6, 5⟩) ∧
  ¬(can_be_covered ⟨5, 4⟩) :=
sorry

end NUMINAMATH_CALUDE_covered_boards_l1485_148592


namespace NUMINAMATH_CALUDE_tea_bags_count_l1485_148576

/-- The number of tea bags in a box -/
def n : ℕ+ := sorry

/-- The number of cups of tea made from Natasha's box -/
def natasha_cups : ℕ := 41

/-- The number of cups of tea made from Inna's box -/
def inna_cups : ℕ := 58

/-- Theorem stating that the number of tea bags in the box is 20 -/
theorem tea_bags_count :
  (2 * n ≤ natasha_cups ∧ natasha_cups ≤ 3 * n) ∧
  (2 * n ≤ inna_cups ∧ inna_cups ≤ 3 * n) →
  n = 20 := by sorry

end NUMINAMATH_CALUDE_tea_bags_count_l1485_148576


namespace NUMINAMATH_CALUDE_problem_solution_l1485_148585

theorem problem_solution :
  ∃ (a b : ℤ) (c d : ℚ),
    (∀ n : ℤ, n > 0 → a ≤ n) ∧
    (∀ n : ℤ, n < 0 → n ≤ b) ∧
    (∀ q : ℚ, q ≠ 0 → |c| ≤ |q|) ∧
    (d⁻¹ = d) ∧
    ((a^2 : ℚ) - (b^2 : ℚ) + 2*d - c = 2 ∨ (a^2 : ℚ) - (b^2 : ℚ) + 2*d - c = -2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1485_148585


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1485_148562

-- Problem 1
theorem problem_1 (a b c : ℝ) : 
  (-8 * a^4 * b^5 * c / (4 * a * b^5)) * (3 * a^3 * b^2) = -6 * a^6 * b^2 * c :=
sorry

-- Problem 2
theorem problem_2 (a : ℝ) :
  (2*a + 1)^2 - (2*a + 1)*(2*a - 1) = 4*a + 2 :=
sorry

-- Problem 3
theorem problem_3 (x y : ℝ) :
  (x - y - 2) * (x - y + 2) - (x + 2*y) * (x - 3*y) = 7*y^2 - x*y - 4 :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1485_148562


namespace NUMINAMATH_CALUDE_solve_equation_solve_system_l1485_148544

-- Problem 1
theorem solve_equation (x : ℝ) : (x + 2) / 3 - 1 = (1 - x) / 2 ↔ x = 1 := by sorry

-- Problem 2
theorem solve_system (x y : ℝ) : x + 2*y = 8 ∧ 3*x - 4*y = 4 ↔ x = 4 ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_solve_equation_solve_system_l1485_148544


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l1485_148543

theorem cubic_function_extrema (a b : ℝ) (h_a : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - 3 * a * x^2 + b
  (∀ x ∈ Set.Icc (-1) 2, f x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = 3) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ -21) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = -21) →
  a = 6 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l1485_148543


namespace NUMINAMATH_CALUDE_range_of_a_l1485_148587

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log (x / 2) - (3 * x - 6) / (x + 1)

noncomputable def g (x t a : ℝ) : ℝ := (x - t)^2 + (Real.log x - a * t)^2

theorem range_of_a :
  ∀ a : ℝ,
  (∀ x₁ : ℝ, x₁ > 1 → ∃ t x₂ : ℝ, x₂ > 0 ∧ f x₁ ≥ g x₂ t a) ↔
  a ≤ 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1485_148587


namespace NUMINAMATH_CALUDE_trishul_investment_percentage_l1485_148525

/-- Represents the investment amounts in Rupees -/
structure Investment where
  vishal : ℝ
  trishul : ℝ
  raghu : ℝ

/-- The given conditions of the investment problem -/
def investment_conditions (i : Investment) : Prop :=
  i.vishal = 1.1 * i.trishul ∧
  i.vishal + i.trishul + i.raghu = 6936 ∧
  i.raghu = 2400

/-- The theorem stating that Trishul invested 10% less than Raghu -/
theorem trishul_investment_percentage (i : Investment) 
  (h : investment_conditions i) : 
  (i.raghu - i.trishul) / i.raghu = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_trishul_investment_percentage_l1485_148525


namespace NUMINAMATH_CALUDE_octahedron_intersection_area_l1485_148540

/-- Represents a regular octahedron -/
structure RegularOctahedron where
  side_length : ℝ

/-- Represents the hexagonal intersection formed by a plane cutting the octahedron -/
structure HexagonalIntersection where
  octahedron : RegularOctahedron

/-- The area of the hexagonal intersection -/
def intersection_area (h : HexagonalIntersection) : ℝ := sorry

theorem octahedron_intersection_area 
  (o : RegularOctahedron)
  (h : HexagonalIntersection)
  (h_octahedron : h.octahedron = o)
  (side_length_eq : o.side_length = 2) :
  intersection_area h = 9 * Real.sqrt 3 / 8 := by sorry

end NUMINAMATH_CALUDE_octahedron_intersection_area_l1485_148540


namespace NUMINAMATH_CALUDE_orange_ribbons_l1485_148523

theorem orange_ribbons (total : ℕ) (yellow purple orange black : ℕ) : 
  yellow + purple + orange + black = total →
  4 * yellow = total →
  3 * purple = total →
  6 * orange = total →
  black = 40 →
  orange = 27 :=
by sorry

end NUMINAMATH_CALUDE_orange_ribbons_l1485_148523


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l1485_148565

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem stating that the smallest number of cubes to fill the given box is 90 -/
theorem smallest_number_of_cubes_for_given_box :
  smallestNumberOfCubes ⟨27, 15, 6⟩ = 90 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l1485_148565


namespace NUMINAMATH_CALUDE_first_group_size_is_correct_l1485_148541

/-- The number of men in the first group -/
def first_group_size : ℕ := 20

/-- The length of the fountain built by the first group -/
def first_fountain_length : ℕ := 56

/-- The number of days taken by the first group to build their fountain -/
def first_group_days : ℕ := 14

/-- The number of men in the second group -/
def second_group_size : ℕ := 35

/-- The length of the fountain built by the second group -/
def second_fountain_length : ℕ := 21

/-- The number of days taken by the second group to build their fountain -/
def second_group_days : ℕ := 3

/-- Theorem stating that the first group size is correct given the conditions -/
theorem first_group_size_is_correct :
  (first_group_size : ℚ) * second_fountain_length * second_group_days =
  second_group_size * first_fountain_length * first_group_days :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_is_correct_l1485_148541


namespace NUMINAMATH_CALUDE_imaginary_part_of_product_l1485_148517

theorem imaginary_part_of_product : Complex.im ((1 - 3*Complex.I) * (1 - Complex.I)) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_product_l1485_148517


namespace NUMINAMATH_CALUDE_infinitely_many_not_n_attainable_all_except_seven_3_attainable_l1485_148581

/-- Definition of an n-admissible sequence -/
def IsNAdmissibleSequence (n : ℕ) (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  (∀ k, k > 0 →
    ((a (2*k) = a (2*k-1) + 2 ∨ a (2*k) = a (2*k-1) + n) ∧
     (a (2*k+1) = 2 * a (2*k) ∨ a (2*k+1) = n * a (2*k))) ∨
    ((a (2*k) = 2 * a (2*k-1) ∨ a (2*k) = n * a (2*k-1)) ∧
     (a (2*k+1) = a (2*k) + 2 ∨ a (2*k+1) = a (2*k) + n)))

/-- Definition of n-attainable number -/
def IsNAttainable (n : ℕ) (m : ℕ) : Prop :=
  m > 1 ∧ ∃ a, IsNAdmissibleSequence n a ∧ ∃ k, a k = m

/-- There are infinitely many positive integers not n-attainable for n > 8 -/
theorem infinitely_many_not_n_attainable (n : ℕ) (hn : n > 8) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ m ∈ S, ¬IsNAttainable n m :=
sorry

/-- All positive integers except 7 are 3-attainable -/
theorem all_except_seven_3_attainable :
  ∀ m : ℕ, m > 0 ∧ m ≠ 7 → IsNAttainable 3 m :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_not_n_attainable_all_except_seven_3_attainable_l1485_148581


namespace NUMINAMATH_CALUDE_divisibility_properties_l1485_148537

theorem divisibility_properties (n : ℤ) : 
  (3 ∣ (n^3 - n)) ∧ 
  (5 ∣ (n^5 - n)) ∧ 
  (7 ∣ (n^7 - n)) ∧ 
  (11 ∣ (n^11 - n)) ∧ 
  (13 ∣ (n^13 - n)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_properties_l1485_148537


namespace NUMINAMATH_CALUDE_red_star_company_profit_optimization_l1485_148527

/-- Red Star Company's profit optimization problem -/
theorem red_star_company_profit_optimization :
  -- Define the cost per item
  let cost : ℝ := 40
  -- Define the initial sales volume (in thousand items)
  let initial_sales : ℝ := 5
  -- Define the price-sales relationship function
  let sales (x : ℝ) : ℝ :=
    if x ≤ 50 then initial_sales else 10 - 0.1 * x
  -- Define the profit function without donation
  let profit (x : ℝ) : ℝ := (x - cost) * sales x
  -- Define the profit function with donation
  let profit_with_donation (x a : ℝ) : ℝ := (x - cost - a) * sales x
  -- State the conditions and the theorem
  ∀ x a : ℝ,
    cost ≤ x ∧ x ≤ 100 →
    -- Maximum profit occurs at x = 70
    profit 70 = 90 ∧
    -- Maximum profit is 90 million yuan
    (∀ y, cost ≤ y ∧ y ≤ 100 → profit y ≤ 90) ∧
    -- With donation a = 4, maximum profit is 78 million yuan
    (x ≤ 70 → profit_with_donation x 4 ≤ 78) ∧
    profit_with_donation 70 4 = 78 := by
  sorry

end NUMINAMATH_CALUDE_red_star_company_profit_optimization_l1485_148527


namespace NUMINAMATH_CALUDE_browser_usage_inconsistency_l1485_148593

theorem browser_usage_inconsistency (total_A : ℕ) (total_B : ℕ) (both : ℕ) (only_one : ℕ) :
  total_A = 316 →
  total_B = 478 →
  both = 104 →
  only_one = 567 →
  (total_A - both) + (total_B - both) ≠ only_one :=
by
  sorry

end NUMINAMATH_CALUDE_browser_usage_inconsistency_l1485_148593


namespace NUMINAMATH_CALUDE_complete_square_expression_l1485_148504

theorem complete_square_expression (y : ℝ) : 
  ∃ (k : ℤ) (b : ℝ), y^2 + 16*y + 60 = (y + b)^2 + k ∧ k = -4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_expression_l1485_148504


namespace NUMINAMATH_CALUDE_expand_product_l1485_148594

theorem expand_product (x : ℝ) : 3 * (x - 2) * (x^2 + 6) = 3*x^3 - 6*x^2 + 18*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1485_148594


namespace NUMINAMATH_CALUDE_probability_not_above_x_axis_is_half_l1485_148598

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  p : Point
  q : Point
  r : Point
  s : Point

/-- The probability of a point not being above the x-axis in a given parallelogram -/
def probabilityNotAboveXAxis (para : Parallelogram) : ℝ := sorry

/-- The specific parallelogram PQRS from the problem -/
def pqrs : Parallelogram :=
  { p := { x := 4, y := 4 }
  , q := { x := -2, y := -2 }
  , r := { x := -8, y := -2 }
  , s := { x := -2, y := 4 }
  }

theorem probability_not_above_x_axis_is_half :
  probabilityNotAboveXAxis pqrs = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_not_above_x_axis_is_half_l1485_148598


namespace NUMINAMATH_CALUDE_s_5_l1485_148574

/-- s(n) is a function that attaches the first n perfect squares in order -/
def s (n : ℕ) : ℕ := sorry

/-- Examples of s(n) for n = 1, 2, 3, 4 -/
axiom s_examples : s 1 = 1 ∧ s 2 = 14 ∧ s 3 = 149 ∧ s 4 = 14916

/-- Theorem: s(5) equals 1491625 -/
theorem s_5 : s 5 = 1491625 := sorry

end NUMINAMATH_CALUDE_s_5_l1485_148574


namespace NUMINAMATH_CALUDE_average_speed_calculation_l1485_148557

/-- Given a trip with specified distances and speeds, calculate the average speed -/
theorem average_speed_calculation (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ)
  (h1 : total_distance = 350)
  (h2 : distance1 = 200)
  (h3 : speed1 = 20)
  (h4 : distance2 = total_distance - distance1)
  (h5 : speed2 = 15) :
  (total_distance) / ((distance1 / speed1) + (distance2 / speed2)) = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l1485_148557


namespace NUMINAMATH_CALUDE_goat_kangaroo_ratio_l1485_148520

theorem goat_kangaroo_ratio : 
  ∀ (num_goats : ℕ), 
    (2 * 23 + 4 * num_goats = 322) → 
    (num_goats : ℚ) / 23 = 3 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_goat_kangaroo_ratio_l1485_148520


namespace NUMINAMATH_CALUDE_food_drive_problem_l1485_148569

theorem food_drive_problem (total_students : ℕ) (cans_per_first_group : ℕ) (non_collecting_students : ℕ) (last_group_students : ℕ) (total_cans : ℕ) :
  total_students = 30 →
  cans_per_first_group = 12 →
  non_collecting_students = 2 →
  last_group_students = 13 →
  total_cans = 232 →
  (total_students / 2) * cans_per_first_group + 0 * non_collecting_students + last_group_students * ((total_cans - (total_students / 2) * cans_per_first_group) / last_group_students) = total_cans →
  (total_cans - (total_students / 2) * cans_per_first_group) / last_group_students = 4 :=
by sorry

end NUMINAMATH_CALUDE_food_drive_problem_l1485_148569


namespace NUMINAMATH_CALUDE_library_fee_calculation_l1485_148596

/-- Calculates the total amount paid for borrowing books from a library. -/
def calculate_library_fee (daily_rate : ℚ) (book1_days : ℕ) (book2_days : ℕ) (book3_days : ℕ) : ℚ :=
  daily_rate * (book1_days + book2_days + book3_days)

theorem library_fee_calculation :
  let daily_rate : ℚ := 1/2
  let book1_days : ℕ := 20
  let book2_days : ℕ := 31
  let book3_days : ℕ := 31
  calculate_library_fee daily_rate book1_days book2_days book3_days = 41 := by
  sorry

#eval calculate_library_fee (1/2) 20 31 31

end NUMINAMATH_CALUDE_library_fee_calculation_l1485_148596


namespace NUMINAMATH_CALUDE_square_inequality_l1485_148560

theorem square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l1485_148560


namespace NUMINAMATH_CALUDE_weavers_in_first_group_l1485_148561

/-- The number of weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of weavers in the second group -/
def second_group_weavers : ℕ := 12

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 36

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 12

/-- Theorem stating that the number of weavers in the first group is 4 -/
theorem weavers_in_first_group :
  first_group_weavers = 4 :=
by sorry

end NUMINAMATH_CALUDE_weavers_in_first_group_l1485_148561


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_l1485_148519

/-- A structure representing a line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A structure representing a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Parallel relation between two lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Containment relation of a line in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Theorem: If line a is parallel to line b, line a is not contained in plane α,
    and line b is contained in plane α, then line a is parallel to plane α -/
theorem line_parallel_to_plane (a b : Line3D) (α : Plane3D) :
  parallel_lines a b → ¬line_in_plane a α → line_in_plane b α → parallel_line_plane a α :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_l1485_148519


namespace NUMINAMATH_CALUDE_table_tennis_ball_surface_area_l1485_148516

/-- The surface area of a sphere with diameter 40 millimeters is approximately 5026.55 square millimeters. -/
theorem table_tennis_ball_surface_area :
  let diameter : ℝ := 40
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius^2
  ∃ ε > 0, abs (surface_area - 5026.55) < ε :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_ball_surface_area_l1485_148516


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1485_148528

theorem sqrt_equation_solution :
  let x : ℝ := 3721 / 256
  Real.sqrt x + Real.sqrt (x + 3) = 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1485_148528


namespace NUMINAMATH_CALUDE_max_quotient_value_l1485_148570

theorem max_quotient_value (a b : ℝ) (ha : 200 ≤ a ∧ a ≤ 400) (hb : 600 ≤ b ∧ b ≤ 1200) :
  (∀ x y, 200 ≤ x ∧ x ≤ 400 → 600 ≤ y ∧ y ≤ 1200 → y / x ≤ b / a) →
  b / a = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l1485_148570


namespace NUMINAMATH_CALUDE_volleyball_team_median_age_l1485_148512

/-- Represents the age distribution of the volleyball team --/
def AgeDistribution : List (Nat × Nat) :=
  [(18, 3), (19, 5), (20, 2), (21, 1), (22, 1)]

/-- The total number of team members --/
def TotalMembers : Nat := 12

/-- Calculates the median age of the team --/
def medianAge (dist : List (Nat × Nat)) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median age of the team is 19 --/
theorem volleyball_team_median_age :
  medianAge AgeDistribution TotalMembers = 19 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_median_age_l1485_148512


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a10_l1485_148539

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  a3_eq_2 : a 3 = 2
  a5_plus_a8_eq_15 : a 5 + a 8 = 15

/-- The 10th term of the arithmetic sequence is 13 -/
theorem arithmetic_sequence_a10 (seq : ArithmeticSequence) : seq.a 10 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a10_l1485_148539


namespace NUMINAMATH_CALUDE_remainder_5_pow_2023_mod_6_l1485_148533

theorem remainder_5_pow_2023_mod_6 : 5^2023 % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_5_pow_2023_mod_6_l1485_148533


namespace NUMINAMATH_CALUDE_second_section_students_correct_l1485_148555

/-- The number of students in the second section of chemistry class X -/
def students_section2 : ℕ := 35

/-- The total number of students in all four sections -/
def total_students : ℕ := 65 + students_section2 + 45 + 42

/-- The overall average of marks per student -/
def overall_average : ℚ := 5195 / 100

/-- Theorem stating that the number of students in the second section is correct -/
theorem second_section_students_correct :
  (65 * 50 + students_section2 * 60 + 45 * 55 + 42 * 45 : ℚ) / total_students = overall_average :=
sorry

end NUMINAMATH_CALUDE_second_section_students_correct_l1485_148555


namespace NUMINAMATH_CALUDE_sequence_formula_l1485_148522

theorem sequence_formula (n : ℕ) : 
  let a : ℕ → ℕ := λ k => 2^k + 1
  (a 1 = 3) ∧ (a 2 = 5) ∧ (a 3 = 9) ∧ (a 4 = 17) ∧ (a 5 = 33) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_l1485_148522


namespace NUMINAMATH_CALUDE_division_problem_l1485_148521

theorem division_problem :
  ∀ (dividend divisor quotient remainder : ℕ),
    dividend = 171 →
    divisor = 21 →
    remainder = 3 →
    dividend = divisor * quotient + remainder →
    quotient = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1485_148521


namespace NUMINAMATH_CALUDE_multiply_64_56_l1485_148568

theorem multiply_64_56 : 64 * 56 = 3584 := by
  sorry

end NUMINAMATH_CALUDE_multiply_64_56_l1485_148568


namespace NUMINAMATH_CALUDE_red_balls_count_l1485_148502

theorem red_balls_count (total : ℕ) (prob : ℚ) (red : ℕ) : 
  total = 20 → prob = 1/4 → (red : ℚ)/total = prob → red = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l1485_148502
