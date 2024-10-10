import Mathlib

namespace triangle_angle_problem_l1682_168234

theorem triangle_angle_problem (a b c : ℝ) (A : ℝ) :
  b = c →
  a^2 = 2 * b^2 * (1 - Real.sin A) →
  A = π / 4 := by
  sorry

end triangle_angle_problem_l1682_168234


namespace buffy_whiskers_l1682_168225

/-- The number of whiskers for each cat -/
structure CatWhiskers where
  juniper : ℕ
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ

/-- The conditions for the cat whiskers problem -/
def catWhiskersConditions (c : CatWhiskers) : Prop :=
  c.juniper = 12 ∧
  c.puffy = 3 * c.juniper ∧
  c.scruffy = 2 * c.puffy ∧
  c.buffy = (c.juniper + c.puffy + c.scruffy) / 3

/-- Theorem stating that given the conditions, Buffy has 40 whiskers -/
theorem buffy_whiskers (c : CatWhiskers) (h : catWhiskersConditions c) : c.buffy = 40 := by
  sorry

end buffy_whiskers_l1682_168225


namespace card_area_reduction_l1682_168250

theorem card_area_reduction (initial_width initial_height : ℝ) 
  (h1 : initial_width = 10 ∧ initial_height = 8)
  (h2 : ∃ (reduced_side : ℝ), reduced_side = initial_width - 2 ∨ reduced_side = initial_height - 2)
  (h3 : ∃ (unreduced_side : ℝ), (reduced_side = initial_width - 2 → unreduced_side = initial_height) ∧
                                (reduced_side = initial_height - 2 → unreduced_side = initial_width))
  (h4 : reduced_side * unreduced_side = 64) :
  (initial_width - 2) * initial_height = 60 ∨ initial_width * (initial_height - 2) = 60 :=
sorry

end card_area_reduction_l1682_168250


namespace sample_size_is_176_l1682_168216

/-- Represents the number of students in a stratum -/
structure Stratum where
  size : ℕ

/-- Represents a sample taken from a stratum -/
structure Sample where
  size : ℕ

/-- Calculates the total sample size for stratified sampling -/
def stratifiedSampleSize (male : Stratum) (female : Stratum) (femaleSample : Sample) : ℕ :=
  let maleSampleSize := (male.size * femaleSample.size) / female.size
  maleSampleSize + femaleSample.size

/-- Theorem: The total sample size is 176 given the specified conditions -/
theorem sample_size_is_176
  (male : Stratum)
  (female : Stratum)
  (femaleSample : Sample)
  (h1 : male.size = 1200)
  (h2 : female.size = 1000)
  (h3 : femaleSample.size = 80) :
  stratifiedSampleSize male female femaleSample = 176 := by
  sorry

#check sample_size_is_176

end sample_size_is_176_l1682_168216


namespace white_balls_count_l1682_168240

theorem white_balls_count (total : ℕ) (red : ℕ) (white : ℕ) : 
  red = 8 →
  red + white = total →
  (5 : ℚ) / 6 * total = white →
  white = 40 :=
by
  sorry

end white_balls_count_l1682_168240


namespace symmetric_point_and_line_l1682_168299

-- Define the point A
def A : ℝ × ℝ := (0, 1)

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the line l₂
def l₂ (x y : ℝ) : Prop := x - 2*y + 2 = 0

-- Define the symmetric point B
def B : ℝ × ℝ := (2, -1)

-- Define the symmetric line l
def l (x y : ℝ) : Prop := 2*x - y - 5 = 0

-- Theorem statement
theorem symmetric_point_and_line :
  (∀ x y : ℝ, l₁ x y ↔ x - y - 1 = 0) ∧ 
  (∀ x y : ℝ, l₂ x y ↔ x - 2*y + 2 = 0) →
  (B = (2, -1) ∧ (∀ x y : ℝ, l x y ↔ 2*x - y - 5 = 0)) :=
sorry

end symmetric_point_and_line_l1682_168299


namespace exam_pass_percentage_l1682_168247

/-- Proves that the percentage of passed candidates is 32% given the conditions of the examination --/
theorem exam_pass_percentage : 
  ∀ (total_candidates : ℕ) 
    (num_girls : ℕ) 
    (num_boys : ℕ) 
    (fail_percentage : ℝ) 
    (pass_percentage : ℝ),
  total_candidates = 2000 →
  num_girls = 900 →
  num_boys = total_candidates - num_girls →
  fail_percentage = 68 →
  pass_percentage = 100 - fail_percentage →
  pass_percentage = 32 := by
sorry

end exam_pass_percentage_l1682_168247


namespace complex_fraction_simplification_l1682_168213

theorem complex_fraction_simplification (z : ℂ) (h : z = 1 + I) :
  (z - 2) / z = I := by
  sorry

end complex_fraction_simplification_l1682_168213


namespace cubic_root_equation_l1682_168257

theorem cubic_root_equation : 2 / (2 - Real.rpow 3 (1/3)) = 2 * (2 + Real.rpow 3 (1/3)) * (4 + Real.rpow 9 (1/3)) / 10 := by
  sorry

end cubic_root_equation_l1682_168257


namespace multiply_fractions_of_numbers_l1682_168242

theorem multiply_fractions_of_numbers : 
  (1/4 : ℚ) * 15 * ((1/3 : ℚ) * 10) = 25/2 := by sorry

end multiply_fractions_of_numbers_l1682_168242


namespace jamies_flyer_delivery_l1682_168212

/-- Jamie's flyer delivery problem -/
theorem jamies_flyer_delivery 
  (hourly_rate : ℝ) 
  (hours_per_delivery : ℝ) 
  (total_weeks : ℕ) 
  (total_earnings : ℝ) 
  (h1 : hourly_rate = 10)
  (h2 : hours_per_delivery = 3)
  (h3 : total_weeks = 6)
  (h4 : total_earnings = 360) : 
  (total_earnings / hourly_rate / total_weeks / hours_per_delivery : ℝ) = 2 := by
  sorry

end jamies_flyer_delivery_l1682_168212


namespace art_club_committee_probability_l1682_168218

def art_club_size : ℕ := 24
def boys_count : ℕ := 12
def girls_count : ℕ := 12
def committee_size : ℕ := 5

theorem art_club_committee_probability :
  let total_combinations := Nat.choose art_club_size committee_size
  let all_boys_or_all_girls := 2 * Nat.choose boys_count committee_size
  (total_combinations - all_boys_or_all_girls : ℚ) / total_combinations = 3427 / 3542 := by
  sorry

end art_club_committee_probability_l1682_168218


namespace ellipse_tangent_perpendicular_l1682_168210

/-- Two ellipses with equations x²/a² + y²/b² = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

def is_on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def is_tangent_to_ellipse (l : Line) (e : Ellipse) (p : Point) : Prop :=
  is_on_ellipse p e ∧ l.m = -p.x * e.b^2 / (p.y * e.a^2)

def intersect_line_ellipse (l : Line) (e : Ellipse) : Set Point :=
  {p : Point | is_on_ellipse p e ∧ p.y = l.m * p.x + l.c}

def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.m * l2.m = -1

theorem ellipse_tangent_perpendicular 
  (e1 e2 : Ellipse) 
  (p : Point) 
  (l : Line) 
  (a b q : Point) :
  e1.a^2 - e1.b^2 = e2.a^2 - e2.b^2 →  -- shared foci condition
  is_tangent_to_ellipse l e1 p →
  a ∈ intersect_line_ellipse l e2 →
  b ∈ intersect_line_ellipse l e2 →
  is_tangent_to_ellipse (Line.mk ((q.y - a.y) / (q.x - a.x)) (q.y - (q.y - a.y) / (q.x - a.x) * q.x)) e2 a →
  is_tangent_to_ellipse (Line.mk ((q.y - b.y) / (q.x - b.x)) (q.y - (q.y - b.y) / (q.x - b.x) * q.x)) e2 b →
  are_perpendicular 
    (Line.mk ((q.y - p.y) / (q.x - p.x)) (q.y - (q.y - p.y) / (q.x - p.x) * q.x))
    (Line.mk ((b.y - a.y) / (b.x - a.x)) (b.y - (b.y - a.y) / (b.x - a.x) * b.x)) :=
by sorry

end ellipse_tangent_perpendicular_l1682_168210


namespace cans_needed_for_35_rooms_l1682_168269

/-- Represents the number of rooms that can be painted with the available paint -/
def initial_rooms : ℕ := 45

/-- Represents the number of paint cans lost -/
def lost_cans : ℕ := 5

/-- Represents the number of rooms that can be painted after losing some paint cans -/
def remaining_rooms : ℕ := 35

/-- Represents that each can must be used entirely (no partial cans) -/
def whole_cans_only : Prop := True

/-- Theorem stating that 18 cans are needed to paint 35 rooms given the conditions -/
theorem cans_needed_for_35_rooms : 
  ∃ (cans_per_room : ℚ),
    cans_per_room * (initial_rooms - remaining_rooms) = lost_cans ∧
    ∃ (cans_needed : ℕ),
      cans_needed = ⌈(remaining_rooms : ℚ) / cans_per_room⌉ ∧
      cans_needed = 18 :=
sorry

end cans_needed_for_35_rooms_l1682_168269


namespace exactly_one_greater_than_one_l1682_168221

theorem exactly_one_greater_than_one
  (a b c : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (prod_one : a * b * c = 1)
  (ineq : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨
  (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨
  (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end exactly_one_greater_than_one_l1682_168221


namespace function_value_at_three_l1682_168275

/-- Given a continuous and differentiable function f satisfying
    f(2x + 1) = 2f(x) + 1 for all real x, and f(0) = 2,
    prove that f(3) = 11. -/
theorem function_value_at_three
  (f : ℝ → ℝ)
  (hcont : Continuous f)
  (hdiff : Differentiable ℝ f)
  (hfunc : ∀ x : ℝ, f (2 * x + 1) = 2 * f x + 1)
  (hf0 : f 0 = 2) :
  f 3 = 11 := by
  sorry

end function_value_at_three_l1682_168275


namespace min_x_plus_y_l1682_168254

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = x*y) :
  x + y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 4*y₀ = x₀*y₀ ∧ x₀ + y₀ = 9 := by
  sorry

end min_x_plus_y_l1682_168254


namespace count_cubic_functions_l1682_168244

-- Define the structure of our cubic function
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the property we're interested in
def satisfiesProperty (f : CubicFunction) : Prop :=
  ∀ x : ℝ, (f.a * x^3 + f.b * x^2 + f.c * x + f.d) *
            ((-f.a) * x^3 + f.b * x^2 + (-f.c) * x + f.d) =
            f.a * x^6 + f.b * x^4 + f.c * x^2 + f.d

-- State the theorem
theorem count_cubic_functions :
  ∃! (s : Finset CubicFunction),
    (∀ f ∈ s, satisfiesProperty f) ∧ s.card = 16 := by
  sorry

end count_cubic_functions_l1682_168244


namespace custom_operation_equation_solution_l1682_168227

-- Define the custom operation
def star (a b : ℝ) : ℝ := 4 * a * b

-- Theorem statement
theorem custom_operation_equation_solution :
  ∀ x : ℝ, star x x + 2 * (star 1 x) - star 2 2 = 0 → x = 2 ∨ x = -4 := by
  sorry

end custom_operation_equation_solution_l1682_168227


namespace percentage_of_boys_studying_science_l1682_168238

theorem percentage_of_boys_studying_science (total_boys : ℕ) (boys_from_A : ℕ) (boys_A_not_science : ℕ) :
  total_boys = 150 →
  boys_from_A = (20 : ℕ) * total_boys / 100 →
  boys_A_not_science = 21 →
  (boys_from_A - boys_A_not_science) * 100 / boys_from_A = 30 := by
  sorry

end percentage_of_boys_studying_science_l1682_168238


namespace characterize_functions_l1682_168208

def is_valid_function (f : ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, m > 0 ∧ n > 0 → ⌊(f (m * n) : ℚ) / n⌋ = f m

theorem characterize_functions :
  ∀ f : ℕ → ℤ, is_valid_function f →
    ∃ r : ℝ, (∀ n : ℕ, n > 0 → f n = ⌊n * r⌋) ∨
              (∀ n : ℕ, n > 0 → f n = ⌈n * r⌉ - 1) :=
sorry

end characterize_functions_l1682_168208


namespace unique_solutions_l1682_168206

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem unique_solutions (x y n : ℕ) : 
  (factorial x + factorial y) / factorial n = 3^n ↔ 
  ((x = 0 ∧ y = 2 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1)) :=
by sorry

end unique_solutions_l1682_168206


namespace tangent_point_x_coordinate_l1682_168293

theorem tangent_point_x_coordinate
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^2 + 1)
  (h2 : ∃ x, HasDerivAt f 4 x) :
  ∃ x, HasDerivAt f 4 x ∧ x = 2 :=
by sorry

end tangent_point_x_coordinate_l1682_168293


namespace sqrt_product_simplification_l1682_168296

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (2 * q^3) = 3 * q^3 * Real.sqrt 10 :=
by sorry

end sqrt_product_simplification_l1682_168296


namespace constant_regular_cells_problem_solution_l1682_168235

/-- Represents the number of regular cells capable of division after a given number of days -/
def regular_cells (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells

/-- Theorem stating that the number of regular cells remains constant -/
theorem constant_regular_cells (initial_cells : ℕ) (days : ℕ) :
  regular_cells initial_cells days = initial_cells :=
by sorry

/-- The specific case for the problem with 4 initial cells and 10 days -/
theorem problem_solution :
  regular_cells 4 10 = 4 :=
by sorry

end constant_regular_cells_problem_solution_l1682_168235


namespace line_passes_through_fixed_point_l1682_168258

/-- The line kx + 3y + k - 9 = 0 passes through the point (-1, 3) for all values of k -/
theorem line_passes_through_fixed_point (k : ℝ) : k * (-1) + 3 * 3 + k - 9 = 0 := by
  sorry

end line_passes_through_fixed_point_l1682_168258


namespace f_nonnegative_l1682_168274

/-- Definition of the function f --/
def f (A B C a b c : ℝ) : ℝ :=
  A * (a^3 + b^3 + c^3) + B * (a^2*b + b^2*c + c^2*a + a*b^2 + b*c^2 + c*a^2) + C * a * b * c

/-- Triangle inequality --/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Main theorem --/
theorem f_nonnegative (A B C : ℝ) :
  (f A B C 1 1 1 ≥ 0) →
  (f A B C 1 1 0 ≥ 0) →
  (f A B C 2 1 1 ≥ 0) →
  ∀ a b c : ℝ, is_triangle a b c → f A B C a b c ≥ 0 :=
by sorry

end f_nonnegative_l1682_168274


namespace doghouse_accessible_area_l1682_168294

-- Define the doghouse
def doghouse_side_length : ℝ := 2

-- Define the tether length
def tether_length : ℝ := 3

-- Theorem statement
theorem doghouse_accessible_area :
  let total_sector_area := π * tether_length^2 * (240 / 360)
  let small_sector_area := 2 * (π * doghouse_side_length^2 * (60 / 360))
  total_sector_area + small_sector_area = (22 * π) / 3 := by
  sorry

end doghouse_accessible_area_l1682_168294


namespace tim_stored_bales_l1682_168276

theorem tim_stored_bales (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 28) 
  (h2 : final_bales = 54) : 
  final_bales - initial_bales = 26 := by
  sorry

end tim_stored_bales_l1682_168276


namespace fourth_term_is_nine_l1682_168270

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The theorem stating that the 4th term of the arithmetic sequence is 9 -/
theorem fourth_term_is_nine (seq : ArithmeticSequence) 
    (first_term : seq.a 1 = 3)
    (sum_three : seq.S 3 = 15) : 
  seq.a 4 = 9 := by
  sorry

end fourth_term_is_nine_l1682_168270


namespace value_two_std_dev_below_mean_l1682_168245

-- Define the properties of the normal distribution
def mean : ℝ := 16.2
def std_dev : ℝ := 2.3

-- Define the value we're looking for
def value : ℝ := mean - 2 * std_dev

-- Theorem stating that the value is 11.6
theorem value_two_std_dev_below_mean :
  value = 11.6 := by sorry

end value_two_std_dev_below_mean_l1682_168245


namespace square_sum_sqrt_difference_and_sum_l1682_168267

theorem square_sum_sqrt_difference_and_sum (x₁ x₂ : ℝ) :
  x₁ = Real.sqrt 3 - Real.sqrt 2 →
  x₂ = Real.sqrt 3 + Real.sqrt 2 →
  x₁^2 + x₂^2 = 10 := by
sorry

end square_sum_sqrt_difference_and_sum_l1682_168267


namespace product_of_sums_equals_x_l1682_168283

theorem product_of_sums_equals_x : ∃ X : ℕ,
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) * (3 + 2) = X := by
  sorry

end product_of_sums_equals_x_l1682_168283


namespace intersection_point_is_unique_l1682_168256

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (21/16, 9/8)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 2 * y = 4 * x - 3

theorem intersection_point_is_unique :
  ∀ x y : ℚ, line1 x y ∧ line2 x y ↔ (x, y) = intersection_point :=
by sorry

end intersection_point_is_unique_l1682_168256


namespace complex_equation_solution_l1682_168280

theorem complex_equation_solution :
  ∃ (x : ℂ), 5 - 2 * I * x = 4 - 5 * I * x ∧ x = I / 3 :=
by sorry

end complex_equation_solution_l1682_168280


namespace line_passes_through_center_l1682_168281

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 8 = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x + y + 1 = 0

-- Define the center of a circle
def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 2

-- Theorem statement
theorem line_passes_through_center :
  ∃ h k : ℝ, is_center h k ∧ line_equation h k :=
sorry

end line_passes_through_center_l1682_168281


namespace square_of_negative_double_product_l1682_168266

theorem square_of_negative_double_product (x y : ℝ) : (-2 * x * y)^2 = 4 * x^2 * y^2 := by
  sorry

end square_of_negative_double_product_l1682_168266


namespace scientific_notation_proof_l1682_168291

def number_to_convert : ℝ := 280000

theorem scientific_notation_proof :
  number_to_convert = 2.8 * (10 : ℝ)^5 :=
by sorry

end scientific_notation_proof_l1682_168291


namespace tan_difference_alpha_pi_8_l1682_168295

theorem tan_difference_alpha_pi_8 (α : ℝ) (h : 2 * Real.tan α = 3 * Real.tan (π / 8)) :
  Real.tan (α - π / 8) = (1 + 5 * Real.sqrt 2) / 49 := by
  sorry

end tan_difference_alpha_pi_8_l1682_168295


namespace quadratic_root_implies_m_value_l1682_168264

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (81^2 - (m + 3) * 81 + m + 2 = 0) → m = 79 := by
  sorry

end quadratic_root_implies_m_value_l1682_168264


namespace area_of_triangle_ABC_l1682_168252

/-- Calculate the area of a triangle given its vertices' coordinates -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The coordinates of points X, Y, and Z -/
def X : ℝ × ℝ := (6, 0)
def Y : ℝ × ℝ := (8, 4)
def Z : ℝ × ℝ := (10, 0)

/-- The ratio of areas between triangles XYZ and ABC -/
def areaRatio : ℝ := 0.1111111111111111

theorem area_of_triangle_ABC : 
  ∃ (A B C : ℝ × ℝ), 
    triangleArea X.1 X.2 Y.1 Y.2 Z.1 Z.2 / triangleArea A.1 A.2 B.1 B.2 C.1 C.2 = areaRatio ∧ 
    triangleArea A.1 A.2 B.1 B.2 C.1 C.2 = 72 := by
  sorry

end area_of_triangle_ABC_l1682_168252


namespace triangle_abc_proof_l1682_168286

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) (M : ℝ × ℝ) :
  (2 * b - Real.sqrt 3 * c) * Real.cos A = Real.sqrt 3 * a * Real.cos C →
  B = π / 6 →
  Real.sqrt ((M.1 - (b + c) / 2)^2 + (M.2)^2) = Real.sqrt 7 →
  A = π / 6 ∧
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 :=
by sorry

end triangle_abc_proof_l1682_168286


namespace f_properties_l1682_168231

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1 - 2 * (Real.sin x) ^ 2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, (0 < S ∧ S < T) → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x, f x ≤ 2) ∧
  (∀ α, 0 < α ∧ α < Real.pi / 3 → (f α = 2 → α = Real.pi / 6)) :=
sorry

end f_properties_l1682_168231


namespace triangle_abc_proof_l1682_168226

theorem triangle_abc_proof (a b c A B C S_ΔABC : Real) 
  (h1 : a = Real.sqrt 3)
  (h2 : b = Real.sqrt 2)
  (h3 : A = π / 3)
  (h4 : 0 < A ∧ A < π)
  (h5 : 0 < B ∧ B < π)
  (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π)
  (h8 : Real.sin A / a = Real.sin B / b)
  (h9 : S_ΔABC = (1 / 2) * a * b * Real.sin C) :
  B = π / 4 ∧ S_ΔABC = (3 + Real.sqrt 3) / 4 := by
  sorry

end triangle_abc_proof_l1682_168226


namespace equation_solution_l1682_168255

theorem equation_solution :
  let f (x : ℝ) := 5 * (x^2)^2 + 3 * x^2 + 2 - 4 * (4 * x^2 + x^2 + 1)
  ∀ x : ℝ, f x = 0 ↔ x = Real.sqrt ((17 + Real.sqrt 329) / 10) ∨ x = -Real.sqrt ((17 + Real.sqrt 329) / 10) :=
by
  sorry


end equation_solution_l1682_168255


namespace darcy_shirts_count_darcy_shirts_proof_l1682_168278

theorem darcy_shirts_count : ℕ :=
  let total_shorts : ℕ := 8
  let folded_shirts : ℕ := 12
  let folded_shorts : ℕ := 5
  let remaining_to_fold : ℕ := 11

  -- Define a function to calculate the total number of clothing items
  let total_clothing (shirts : ℕ) : ℕ := shirts + total_shorts

  -- Define a function to calculate the number of folded items
  let folded_items : ℕ := folded_shirts + folded_shorts

  -- The number of shirts that satisfies the conditions
  20

theorem darcy_shirts_proof (shirts : ℕ) : 
  let total_shorts : ℕ := 8
  let folded_shirts : ℕ := 12
  let folded_shorts : ℕ := 5
  let remaining_to_fold : ℕ := 11
  let total_clothing := shirts + total_shorts
  let folded_items := folded_shirts + folded_shorts

  shirts = 20 ↔ 
    total_clothing - folded_items = remaining_to_fold :=
by sorry

end darcy_shirts_count_darcy_shirts_proof_l1682_168278


namespace coin_problem_l1682_168259

/-- Given a total sum in paise, the number of 20 paise coins, and that the remaining sum is made up of 25 paise coins, calculate the total number of coins. -/
def total_coins (total_sum : ℕ) (coins_20 : ℕ) : ℕ :=
  let sum_20 := coins_20 * 20
  let sum_25 := total_sum - sum_20
  let coins_25 := sum_25 / 25
  coins_20 + coins_25

/-- Theorem stating that given the specific conditions, the total number of coins is 334. -/
theorem coin_problem : total_coins 7100 250 = 334 := by
  sorry

end coin_problem_l1682_168259


namespace loss_percentage_is_twenty_percent_l1682_168237

-- Define the given conditions
def articles_sold_gain : ℕ := 20
def selling_price_gain : ℚ := 60
def gain_percentage : ℚ := 20 / 100

def articles_sold_loss : ℚ := 24.999996875000388
def selling_price_loss : ℚ := 50

-- Theorem to prove
theorem loss_percentage_is_twenty_percent :
  let cost_price := selling_price_gain / (1 + gain_percentage)
  let cost_per_article := cost_price / articles_sold_gain
  let cost_price_loss := cost_per_article * articles_sold_loss
  let loss := cost_price_loss - selling_price_loss
  loss / cost_price_loss = 1 / 5 := by
  sorry

end loss_percentage_is_twenty_percent_l1682_168237


namespace base_4_last_digit_379_l1682_168260

def base_4_last_digit (n : ℕ) : ℕ :=
  n % 4

theorem base_4_last_digit_379 : base_4_last_digit 379 = 3 := by
  sorry

end base_4_last_digit_379_l1682_168260


namespace beach_house_rental_l1682_168214

theorem beach_house_rental (individual_payment : ℕ) (total_payment : ℕ) 
  (h1 : individual_payment = 70)
  (h2 : total_payment = 490) :
  total_payment / individual_payment = 7 :=
by sorry

end beach_house_rental_l1682_168214


namespace no_real_roots_l1682_168288

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (2 * x + 8) - Real.sqrt (x - 1) + 2 = 0 := by
  sorry

end no_real_roots_l1682_168288


namespace geometric_sequence_property_l1682_168217

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: For a geometric sequence where a_4 + a_8 = -2, 
    the value of a_6(a_2 + 2a_6 + a_10) is equal to 4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : geometric_sequence a) 
    (h_sum : a 4 + a 8 = -2) : 
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 := by
  sorry

end geometric_sequence_property_l1682_168217


namespace quadratic_inequality_sufficient_not_necessary_l1682_168201

theorem quadratic_inequality_sufficient_not_necessary :
  (∃ x : ℝ, 0 < x ∧ x < 4 ∧ ¬(x^2 - 2*x < 0)) ∧
  (∀ x : ℝ, x^2 - 2*x < 0 → 0 < x ∧ x < 4) :=
sorry

end quadratic_inequality_sufficient_not_necessary_l1682_168201


namespace remainder_2015_div_28_l1682_168263

theorem remainder_2015_div_28 : 2015 % 28 = 17 := by
  sorry

end remainder_2015_div_28_l1682_168263


namespace onion_basket_change_l1682_168292

theorem onion_basket_change (x : ℤ) : x + 4 - 5 + 9 = x + 8 := by
  sorry

end onion_basket_change_l1682_168292


namespace tan_alpha_value_l1682_168233

theorem tan_alpha_value (α β : Real) 
  (h1 : Real.tan (α + β) = 3/5) 
  (h2 : Real.tan β = 1/3) : 
  Real.tan α = 2/9 := by
  sorry

end tan_alpha_value_l1682_168233


namespace unique_number_l1682_168262

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  Odd n ∧ 
  n % 9 = 0 ∧ 
  is_perfect_square (digit_product n) :=
by sorry

end unique_number_l1682_168262


namespace calvin_chips_weeks_l1682_168285

/-- Calculates the number of weeks Calvin has been buying chips -/
def weeks_buying_chips (cost_per_pack : ℚ) (days_per_week : ℕ) (total_spent : ℚ) : ℚ :=
  total_spent / (cost_per_pack * days_per_week)

/-- Theorem stating that Calvin has been buying chips for 4 weeks -/
theorem calvin_chips_weeks :
  let cost_per_pack : ℚ := 1/2  -- $0.50 represented as a rational number
  let days_per_week : ℕ := 5
  let total_spent : ℚ := 10
  weeks_buying_chips cost_per_pack days_per_week total_spent = 4 := by
  sorry

end calvin_chips_weeks_l1682_168285


namespace puppy_adoption_ratio_l1682_168246

theorem puppy_adoption_ratio :
  let first_week : ℕ := 20
  let second_week : ℕ := (2 * first_week) / 5
  let fourth_week : ℕ := first_week + 10
  let total_puppies : ℕ := 74
  let third_week : ℕ := total_puppies - (first_week + second_week + fourth_week)
  (third_week : ℚ) / second_week = 2 := by
  sorry

end puppy_adoption_ratio_l1682_168246


namespace yogurt_cost_yogurt_cost_is_one_l1682_168232

/-- The cost of yogurt given Seth's purchase information -/
theorem yogurt_cost (ice_cream_quantity : ℕ) (yogurt_quantity : ℕ) 
  (ice_cream_cost : ℕ) (extra_spent : ℕ) : ℕ :=
  let total_ice_cream_cost := ice_cream_quantity * ice_cream_cost
  let yogurt_cost := (total_ice_cream_cost - extra_spent) / yogurt_quantity
  yogurt_cost

/-- Proof that the cost of each carton of yogurt is $1 -/
theorem yogurt_cost_is_one :
  yogurt_cost 20 2 6 118 = 1 := by
  sorry

end yogurt_cost_yogurt_cost_is_one_l1682_168232


namespace min_value_quadratic_l1682_168251

theorem min_value_quadratic (x y : ℝ) : 
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧ 
  ∃ (a b : ℝ), a^2 + b^2 - 8*a + 6*b + 25 = 0 := by
sorry

end min_value_quadratic_l1682_168251


namespace equation_equivalence_l1682_168230

theorem equation_equivalence (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  1 + 1/x + 2*(x+1)/(x*y) + 3*(x+1)*(y+2)/(x*y*z) + 4*(x+1)*(y+2)*(z+3)/(x*y*z*w) = 0 ↔
  (1 + 1/x) * (1 + 2/y) * (1 + 3/z) * (1 + 4/w) = 0 :=
by sorry

end equation_equivalence_l1682_168230


namespace total_pebbles_is_50_l1682_168261

/-- Represents the number of pebbles of each color and the total --/
structure PebbleCounts where
  white : ℕ
  red : ℕ
  blue : ℕ
  green : ℕ
  total : ℕ

/-- Defines the conditions of the pebble problem --/
def pebble_problem (p : PebbleCounts) : Prop :=
  p.white = 20 ∧
  p.red = p.white / 2 ∧
  p.blue = p.red / 3 ∧
  p.green = p.blue + 5 ∧
  p.red = p.total / 5 ∧
  p.total = p.white + p.red + p.blue + p.green

/-- Theorem stating that the total number of pebbles is 50 --/
theorem total_pebbles_is_50 :
  ∃ p : PebbleCounts, pebble_problem p ∧ p.total = 50 :=
by sorry

end total_pebbles_is_50_l1682_168261


namespace man_speed_opposite_train_man_speed_specific_case_l1682_168289

/-- Calculates the speed of a man running opposite to a train, given the train's length, speed, and time to pass the man. -/
theorem man_speed_opposite_train (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let man_speed_mps := train_length / passing_time - train_speed_mps
  let man_speed_kmph := man_speed_mps * (3600 / 1000)
  man_speed_kmph

/-- The speed of a man running opposite to a train is approximately 5.99 kmph, given:
    - The train is 550 meters long
    - The train's speed is 60 kmph
    - The train passes the man in 30 seconds -/
theorem man_speed_specific_case : 
  abs (man_speed_opposite_train 550 60 30 - 5.99) < 0.01 := by
  sorry

end man_speed_opposite_train_man_speed_specific_case_l1682_168289


namespace pattern_and_application_l1682_168228

theorem pattern_and_application (n : ℕ) (a b : ℝ) :
  n > 1 →
  (n : ℝ) * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) ∧
  (a * Real.sqrt (7 / b) = Real.sqrt (a + 7 / b) → a + b = 55) :=
by sorry

end pattern_and_application_l1682_168228


namespace no_integer_solutions_l1682_168268

theorem no_integer_solutions : ¬ ∃ (x y z : ℤ), 
  (x^2 - 2*x*y + 3*y^2 - 2*z^2 = 25) ∧ 
  (-x^2 + 4*y*z + 3*z^2 = 55) ∧ 
  (x^2 + 3*x*y - y^2 + 7*z^2 = 130) := by
  sorry

end no_integer_solutions_l1682_168268


namespace prime_sum_30_l1682_168202

theorem prime_sum_30 (A B C : ℕ) : 
  Prime A ∧ Prime B ∧ Prime C ∧
  A < 20 ∧ B < 20 ∧ C < 20 ∧
  A + B + C = 30 →
  (A = 2 ∧ B = 11 ∧ C = 17) ∨
  (A = 2 ∧ B = 17 ∧ C = 11) ∨
  (A = 11 ∧ B = 2 ∧ C = 17) ∨
  (A = 11 ∧ B = 17 ∧ C = 2) ∨
  (A = 17 ∧ B = 2 ∧ C = 11) ∨
  (A = 17 ∧ B = 11 ∧ C = 2) := by
sorry

end prime_sum_30_l1682_168202


namespace root_sum_product_l1682_168211

def complex_plane : Type := ℂ

def coordinates (z : ℂ) : ℝ × ℝ := (z.re, z.im)

theorem root_sum_product (z : ℂ) (p q : ℝ) :
  coordinates z = (-1, 3) →
  (z^2 + p*z + q = 0) →
  p + q = 12 := by sorry

end root_sum_product_l1682_168211


namespace parallel_lines_m_value_l1682_168298

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y, m₁ * x - y = b₁ ↔ m₂ * x - y = b₂) ↔ m₁ = m₂

/-- The value of m for which mx - y - 1 = 0 is parallel to x - 2y + 3 = 0 -/
theorem parallel_lines_m_value :
  (∀ x y, m * x - y - 1 = 0 ↔ x - 2 * y + 3 = 0) → m = 1 / 2 :=
by
  sorry


end parallel_lines_m_value_l1682_168298


namespace sin_C_equals_half_l1682_168239

theorem sin_C_equals_half (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- b = 2c * sin(B)
  (b = 2 * c * Real.sin B) →
  -- Then sin(C) = 1/2
  Real.sin C = 1/2 := by
sorry

end sin_C_equals_half_l1682_168239


namespace probability_twelve_rolls_eight_sided_die_l1682_168273

/-- The probability of rolling an eight-sided die 12 times, where the first 11 rolls are all
    different from their immediate predecessors, and the 12th roll matches the 11th roll. -/
def probability_twelve_rolls (n : ℕ) : ℚ :=
  if n = 8 then
    (7 : ℚ)^10 / 8^11
  else
    0

/-- Theorem stating that the probability of the described event with an eight-sided die
    is equal to 7^10 / 8^11. -/
theorem probability_twelve_rolls_eight_sided_die :
  probability_twelve_rolls 8 = (7 : ℚ)^10 / 8^11 :=
by sorry

end probability_twelve_rolls_eight_sided_die_l1682_168273


namespace cannot_generate_AC_l1682_168297

/-- Represents a sequence of letters --/
inductive Sequence
| empty : Sequence
| cons : Char → Sequence → Sequence

/-- Checks if a sequence ends with the letter B --/
def endsWithB : Sequence → Bool := sorry

/-- Checks if a sequence starts with the letter A --/
def startsWithA : Sequence → Bool := sorry

/-- Counts the number of consecutive B's in a sequence --/
def countConsecutiveB : Sequence → Nat := sorry

/-- Counts the number of consecutive C's in a sequence --/
def countConsecutiveC : Sequence → Nat := sorry

/-- Applies Rule I: If a sequence ends with B, append C --/
def applyRuleI : Sequence → Sequence := sorry

/-- Applies Rule II: If a sequence starts with A, double the sequence after A --/
def applyRuleII : Sequence → Sequence := sorry

/-- Applies Rule III: Replace BBB with C anywhere in the sequence --/
def applyRuleIII : Sequence → Sequence := sorry

/-- Applies Rule IV: Remove CC anywhere in the sequence --/
def applyRuleIV : Sequence → Sequence := sorry

/-- Checks if a sequence is equal to "AC" --/
def isAC : Sequence → Bool := sorry

/-- Initial sequence "AB" --/
def initialSequence : Sequence := sorry

/-- Represents all sequences that can be generated from the initial sequence --/
inductive GeneratedSequence : Sequence → Prop
| initial : GeneratedSequence initialSequence
| rule1 {s : Sequence} : GeneratedSequence s → endsWithB s = true → GeneratedSequence (applyRuleI s)
| rule2 {s : Sequence} : GeneratedSequence s → startsWithA s = true → GeneratedSequence (applyRuleII s)
| rule3 {s : Sequence} : GeneratedSequence s → countConsecutiveB s ≥ 3 → GeneratedSequence (applyRuleIII s)
| rule4 {s : Sequence} : GeneratedSequence s → countConsecutiveC s ≥ 2 → GeneratedSequence (applyRuleIV s)

theorem cannot_generate_AC :
  ∀ s, GeneratedSequence s → isAC s = false := by sorry

end cannot_generate_AC_l1682_168297


namespace river_road_cars_l1682_168203

theorem river_road_cars (buses cars : ℕ) : 
  (buses : ℚ) / cars = 1 / 3 →
  buses = cars - 40 →
  cars = 60 := by
sorry

end river_road_cars_l1682_168203


namespace eggs_remaining_l1682_168229

theorem eggs_remaining (initial_eggs : ℕ) (eggs_taken : ℕ) (eggs_left : ℕ) : 
  initial_eggs = 47 → eggs_taken = 5 → eggs_left = initial_eggs - eggs_taken → eggs_left = 42 :=
by sorry

end eggs_remaining_l1682_168229


namespace f_max_value_l1682_168272

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x - 2 * Real.sin (3 * x)

theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = (16 * Real.sqrt 3) / 9 := by
  sorry

end f_max_value_l1682_168272


namespace equation_satisfied_l1682_168223

theorem equation_satisfied (a b c : ℤ) (h1 : a = c - 1) (h2 : b = a - 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 1 := by
  sorry

end equation_satisfied_l1682_168223


namespace symmetry_xoy_plane_l1682_168205

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoy plane in 3D space -/
def xoy_plane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xoy plane -/
def symmetric_xoy (p : Point3D) : Point3D :=
  ⟨p.x, p.y, -p.z⟩

theorem symmetry_xoy_plane :
  let P : Point3D := ⟨1, 3, -5⟩
  symmetric_xoy P = ⟨1, 3, 5⟩ := by
  sorry


end symmetry_xoy_plane_l1682_168205


namespace expression_value_l1682_168282

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  (x^5 + 3*y^3) / 9 = 141 := by
  sorry

end expression_value_l1682_168282


namespace perpendicular_line_equation_l1682_168241

/-- The center of the circle (x-1)^2 + (y+1)^2 = 2 -/
def circle_center : ℝ × ℝ := (1, -1)

/-- The slope of the given line 2x + y = 0 -/
def given_line_slope : ℝ := -2

/-- The perpendicular line passing through the circle center -/
def perpendicular_line (x y : ℝ) : Prop :=
  x - given_line_slope * y - (circle_center.1 - given_line_slope * circle_center.2) = 0

theorem perpendicular_line_equation :
  perpendicular_line = fun x y ↦ x - 2 * y - 3 = 0 := by sorry

end perpendicular_line_equation_l1682_168241


namespace remainder_of_741147_div_6_l1682_168209

theorem remainder_of_741147_div_6 : 741147 % 6 = 3 := by
  sorry

end remainder_of_741147_div_6_l1682_168209


namespace lacy_correct_percentage_l1682_168215

theorem lacy_correct_percentage (x : ℝ) (h : x > 0) :
  let total_problems := 6 * x
  let missed_problems := 2 * x
  let correct_problems := total_problems - missed_problems
  (correct_problems / total_problems) * 100 = 200 / 3 := by
  sorry

end lacy_correct_percentage_l1682_168215


namespace keith_picked_no_pears_l1682_168271

-- Define the number of apples picked by each person
def mike_apples : ℕ := 7
def nancy_apples : ℕ := 3
def keith_apples : ℕ := 6

-- Define the total number of apples picked
def total_apples : ℕ := 16

-- Define Keith's pears as a variable
def keith_pears : ℕ := sorry

-- Theorem to prove
theorem keith_picked_no_pears : keith_pears = 0 := by
  sorry

end keith_picked_no_pears_l1682_168271


namespace construction_cost_equation_l1682_168248

/-- The cost of land per square meter that satisfies the construction cost equation -/
def land_cost_per_sqm : ℝ := 50

/-- The cost of bricks per 1000 bricks -/
def brick_cost_per_1000 : ℝ := 100

/-- The cost of roof tiles per tile -/
def roof_tile_cost : ℝ := 10

/-- The required land area in square meters -/
def required_land_area : ℝ := 2000

/-- The required number of bricks -/
def required_bricks : ℝ := 10000

/-- The required number of roof tiles -/
def required_roof_tiles : ℝ := 500

/-- The total construction cost -/
def total_construction_cost : ℝ := 106000

theorem construction_cost_equation :
  land_cost_per_sqm * required_land_area +
  brick_cost_per_1000 * (required_bricks / 1000) +
  roof_tile_cost * required_roof_tiles =
  total_construction_cost :=
sorry

end construction_cost_equation_l1682_168248


namespace t_leq_s_l1682_168249

theorem t_leq_s (a b t s : ℝ) (ht : t = a + 2*b) (hs : s = a + b^2 + 1) : t ≤ s := by
  sorry

end t_leq_s_l1682_168249


namespace max_2x2_squares_5x7_grid_l1682_168219

/-- Represents the dimensions of the grid -/
structure GridDimensions where
  rows : ℕ
  cols : ℕ

/-- Represents the different types of pieces that can be cut from the grid -/
inductive PieceType
  | Square2x2
  | LShape
  | Strip1x3

/-- Represents a configuration of pieces cut from the grid -/
structure Configuration where
  square2x2Count : ℕ
  lShapeCount : ℕ
  strip1x3Count : ℕ

/-- Checks if a configuration is valid for the given grid dimensions -/
def isValidConfiguration (grid : GridDimensions) (config : Configuration) : Prop :=
  4 * config.square2x2Count + 3 * config.lShapeCount + 3 * config.strip1x3Count = grid.rows * grid.cols

/-- Theorem: The maximum number of 2x2 squares in a valid configuration for a 5x7 grid is 5 -/
theorem max_2x2_squares_5x7_grid :
  ∃ (maxSquares : ℕ),
    maxSquares = 5 ∧
    (∃ (config : Configuration),
      isValidConfiguration ⟨5, 7⟩ config ∧
      config.square2x2Count = maxSquares) ∧
    (∀ (config : Configuration),
      isValidConfiguration ⟨5, 7⟩ config →
      config.square2x2Count ≤ maxSquares) :=
by
  sorry

end max_2x2_squares_5x7_grid_l1682_168219


namespace pies_sold_per_day_l1682_168284

/-- Given a restaurant that sells pies every day for a week and sells 56 pies in total,
    prove that the number of pies sold each day is 8. -/
theorem pies_sold_per_day (total_pies : ℕ) (days_in_week : ℕ) 
  (h1 : total_pies = 56) 
  (h2 : days_in_week = 7) :
  total_pies / days_in_week = 8 := by
  sorry

end pies_sold_per_day_l1682_168284


namespace beach_relaxation_l1682_168290

/-- The number of people left relaxing on the beach -/
def people_left_relaxing (row1_initial : ℕ) (row1_left : ℕ) (row2_initial : ℕ) (row2_left : ℕ) (row3 : ℕ) : ℕ :=
  (row1_initial - row1_left) + (row2_initial - row2_left) + row3

/-- Theorem stating the number of people left relaxing on the beach -/
theorem beach_relaxation : 
  people_left_relaxing 24 3 20 5 18 = 54 := by
  sorry

end beach_relaxation_l1682_168290


namespace dog_tail_length_l1682_168224

/-- Represents a dog with specific proportions and measurements -/
structure Dog where
  body_length : ℝ
  tail_length : ℝ
  head_length : ℝ
  height : ℝ
  width : ℝ
  weight : ℝ

/-- The dog satisfies the given proportions and measurements -/
def is_valid_dog (d : Dog) : Prop :=
  d.tail_length = d.body_length / 2 ∧
  d.head_length = d.body_length / 6 ∧
  d.height = 1.5 * d.width ∧
  d.weight = 36 ∧
  d.body_length + d.tail_length + d.head_length = 30 ∧
  d.width = 12

/-- The theorem stating that a valid dog's tail length is 10 inches -/
theorem dog_tail_length (d : Dog) (h : is_valid_dog d) : d.tail_length = 10 := by
  sorry

end dog_tail_length_l1682_168224


namespace fundraiser_results_l1682_168222

/-- Represents the sales data for Markeesha's fundraiser --/
structure SalesData where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Calculates the total profit given the sales data --/
def totalProfit (data : SalesData) : ℚ :=
  (4 * data.monday + 205) * (4.5 : ℚ)

/-- Calculates the total number of boxes sold given the sales data --/
def totalBoxesSold (data : SalesData) : ℕ :=
  4 * data.monday + 205

/-- Determines the most profitable day given the sales data --/
def mostProfitableDay (data : SalesData) : String :=
  if data.saturday ≥ data.monday ∧ data.saturday ≥ data.tuesday ∧ 
     data.saturday ≥ data.wednesday ∧ data.saturday ≥ data.thursday ∧ 
     data.saturday ≥ data.friday ∧ data.saturday ≥ data.sunday
  then "Saturday"
  else "Other"

theorem fundraiser_results (M : ℕ) :
  let data : SalesData := {
    monday := M,
    tuesday := M + 10,
    wednesday := M + 20,
    thursday := M + 30,
    friday := 30,
    saturday := 60,
    sunday := 45
  }
  totalProfit data = (4 * M + 205 : ℕ) * (4.5 : ℚ) ∧
  totalBoxesSold data = 4 * M + 205 ∧
  mostProfitableDay data = "Saturday" :=
by sorry

#check fundraiser_results

end fundraiser_results_l1682_168222


namespace log_weight_when_cut_l1682_168204

/-- Given a log of length 20 feet that weighs 150 pounds per linear foot,
    prove that when cut in half, each piece weighs 1500 pounds. -/
theorem log_weight_when_cut (log_length : ℝ) (weight_per_foot : ℝ) :
  log_length = 20 →
  weight_per_foot = 150 →
  (log_length / 2) * weight_per_foot = 1500 := by
  sorry

end log_weight_when_cut_l1682_168204


namespace pizzeria_sales_l1682_168287

/-- Calculates the total sales of a pizzeria given the prices and quantities of small and large pizzas sold. -/
theorem pizzeria_sales
  (small_price : ℕ)
  (large_price : ℕ)
  (small_quantity : ℕ)
  (large_quantity : ℕ)
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : small_quantity = 8)
  (h4 : large_quantity = 3) :
  small_price * small_quantity + large_price * large_quantity = 40 :=
by sorry

#check pizzeria_sales

end pizzeria_sales_l1682_168287


namespace earthquake_victims_scientific_notation_l1682_168279

/-- Definition of scientific notation -/
def is_scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  x = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10

/-- The problem statement -/
theorem earthquake_victims_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation 153000 a n ∧ a = 1.53 ∧ n = 5 := by
  sorry

end earthquake_victims_scientific_notation_l1682_168279


namespace largest_number_proof_l1682_168243

theorem largest_number_proof (a b c : ℕ) 
  (h1 : c - a = 6) 
  (h2 : b = (a + c) / 2) 
  (h3 : a * b * c = 46332) : 
  c = 39 := by
  sorry

end largest_number_proof_l1682_168243


namespace integer_part_inequality_l1682_168200

theorem integer_part_inequality (m n : ℕ+) : 
  (∀ (α β : ℝ), ⌊(m+n)*α⌋ + ⌊(m+n)*β⌋ ≥ ⌊m*α⌋ + ⌊m*β⌋ + ⌊n*(α+β)⌋) ↔ m = n :=
sorry

end integer_part_inequality_l1682_168200


namespace stamp_denominations_l1682_168236

/-- Given stamps of denominations 7, n, and n+2 cents, 
    if 120 cents is the greatest postage that cannot be formed, then n = 22 -/
theorem stamp_denominations (n : ℕ) : 
  (∀ k > 120, ∃ a b c : ℕ, k = 7 * a + n * b + (n + 2) * c) ∧
  (¬ ∃ a b c : ℕ, 120 = 7 * a + n * b + (n + 2) * c) →
  n = 22 := by sorry

end stamp_denominations_l1682_168236


namespace intersection_theorem_l1682_168253

open Set Real

-- Define sets A and B
def A : Set ℝ := {x | x^2 + x - 6 < 0}
def B : Set ℝ := {x | x + 1 > 0}

-- Define the intersection of A and B
def A_inter_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_theorem : A_inter_B = Ioo (-1) 2 := by sorry

end intersection_theorem_l1682_168253


namespace rectangle_area_l1682_168265

theorem rectangle_area (r : ℝ) (ratio : ℝ) : 
  r = 7 →
  ratio = 3 →
  (2 * r * ratio) * (2 * r) = 588 := by
  sorry

end rectangle_area_l1682_168265


namespace min_value_expression_min_value_achievable_l1682_168220

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 2 * Real.sqrt 5 - 4 * 5^(1/4) + 4 :=
by sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 5 ∧
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 = 2 * Real.sqrt 5 - 4 * 5^(1/4) + 4 :=
by sorry

end min_value_expression_min_value_achievable_l1682_168220


namespace tan_double_alpha_l1682_168207

theorem tan_double_alpha (α : ℝ) (h : (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3) :
  Real.tan (2 * α) = -8 / 15 := by
  sorry

end tan_double_alpha_l1682_168207


namespace special_matrix_sum_l1682_168277

/-- Represents a 3x3 matrix with the given structure -/
structure SpecialMatrix where
  v : ℝ
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  sum_equality : ℝ
  sum_row_1 : v + 50 + w = sum_equality
  sum_row_2 : 196 + x + y = sum_equality
  sum_row_3 : 269 + z + 123 = sum_equality
  sum_col_1 : v + 196 + 269 = sum_equality
  sum_col_2 : 50 + x + z = sum_equality
  sum_col_3 : w + y + 123 = sum_equality
  sum_diag_1 : v + x + 123 = sum_equality
  sum_diag_2 : w + x + 269 = sum_equality

/-- Theorem: In the SpecialMatrix, y + z = 196 -/
theorem special_matrix_sum (m : SpecialMatrix) : m.y + m.z = 196 := by
  sorry

end special_matrix_sum_l1682_168277
