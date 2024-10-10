import Mathlib

namespace rent_percentage_last_year_l3232_323263

theorem rent_percentage_last_year (E : ℝ) (P : ℝ) : 
  E > 0 → 
  (0.30 * (1.25 * E) = 1.875 * (P / 100) * E) → 
  P = 20 :=
by
  sorry

end rent_percentage_last_year_l3232_323263


namespace lunch_break_duration_l3232_323255

/-- Represents the painting rate of an individual or group in terms of house percentage per hour -/
structure PaintingRate where
  rate : ℝ
  (nonneg : rate ≥ 0)

/-- Represents the duration of work in hours -/
def workDuration (startTime endTime : ℝ) : ℝ := endTime - startTime

/-- Represents the percentage of house painted given a painting rate and work duration -/
def percentPainted (r : PaintingRate) (duration : ℝ) : ℝ := r.rate * duration

theorem lunch_break_duration (paula : PaintingRate) (helpers : PaintingRate) 
  (lunchBreak : ℝ) : 
  -- Monday's condition
  percentPainted (PaintingRate.mk (paula.rate + helpers.rate) (by sorry)) (workDuration 8 16 - lunchBreak) = 0.5 →
  -- Tuesday's condition
  percentPainted helpers (workDuration 8 14.2 - lunchBreak) = 0.24 →
  -- Wednesday's condition
  percentPainted paula (workDuration 8 19.2 - lunchBreak) = 0.26 →
  -- Conclusion
  lunchBreak * 60 = 48 := by sorry

end lunch_break_duration_l3232_323255


namespace distance_between_vertices_l3232_323228

-- Define the equation
def equation (x y : ℝ) : Prop := Real.sqrt (x^2 + y^2) + |y - 2| = 4

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = 3 - (1/12) * x^2
def parabola2 (x y : ℝ) : Prop := y = (1/4) * x^2 - 1

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem distance_between_vertices : 
  ∃ (x1 y1 x2 y2 : ℝ), 
    parabola1 x1 y1 ∧ 
    parabola2 x2 y2 ∧ 
    (x1, y1) = vertex1 ∧ 
    (x2, y2) = vertex2 ∧ 
    Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 :=
by sorry

end distance_between_vertices_l3232_323228


namespace chip_defect_rate_line_A_l3232_323220

theorem chip_defect_rate_line_A :
  let total_chips : ℕ := 20
  let chips_line_A : ℕ := 12
  let chips_line_B : ℕ := 8
  let defect_rate_B : ℚ := 1 / 20
  let overall_defect_rate : ℚ := 8 / 100
  let defect_rate_A : ℚ := 1 / 10
  (chips_line_A : ℚ) * defect_rate_A + (chips_line_B : ℚ) * defect_rate_B = (total_chips : ℚ) * overall_defect_rate :=
by sorry

end chip_defect_rate_line_A_l3232_323220


namespace marble_probability_l3232_323232

theorem marble_probability (total : ℕ) (blue : ℕ) (red_white_prob : ℚ) : 
  total = 30 → blue = 5 → red_white_prob = 5/6 → (total - blue : ℚ) / total = red_white_prob :=
by
  sorry

end marble_probability_l3232_323232


namespace wizard_elixir_combinations_l3232_323290

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of mystical crystals available. -/
def num_crystals : ℕ := 6

/-- The number of incompatible crystals. -/
def num_incompatible_crystals : ℕ := 2

/-- The number of herbs incompatible with some crystals. -/
def num_incompatible_herbs : ℕ := 3

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - num_incompatible_crystals * num_incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 18 :=
sorry

end wizard_elixir_combinations_l3232_323290


namespace normal_distribution_symmetry_l3232_323244

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- Probability of a normal random variable being less than a given value -/
noncomputable def prob_less_than (X : NormalRV) (x : ℝ) : ℝ := sorry

theorem normal_distribution_symmetry 
  (X : NormalRV) 
  (h : X.μ = 2) 
  (h2 : prob_less_than X 4 = 0.8) : 
  prob_less_than X 0 = 0.2 := by sorry

end normal_distribution_symmetry_l3232_323244


namespace lines_intersect_on_ellipse_l3232_323209

/-- Two lines intersect and their intersection point lies on a specific ellipse -/
theorem lines_intersect_on_ellipse (k₁ k₂ : ℝ) (h : k₁ * k₂ + 2 = 0) :
  ∃ (x y : ℝ),
    (y = k₁ * x + 1 ∧ y = k₂ * x - 1) ∧  -- Lines intersect
    2 * x^2 + y^2 = 6 :=                 -- Intersection point on ellipse
by sorry

end lines_intersect_on_ellipse_l3232_323209


namespace hyperbola_eccentricity_l3232_323215

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- The asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola) : (ℝ → ℝ) × (ℝ → ℝ) := sorry

/-- A perpendicular line from a point to a line -/
def perpendicular_line (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ → ℝ) : ℝ × ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  let f := right_focus h
  let (asym1, asym2) := asymptotes h
  let perp := perpendicular_line f asym1
  let a := intersection_point perp asym1
  let b := intersection_point perp asym2
  (b.1 - f.1)^2 + (b.2 - f.2)^2 = 4 * ((a.1 - f.1)^2 + (a.2 - f.2)^2) →
  eccentricity h = 2 := by sorry

end hyperbola_eccentricity_l3232_323215


namespace red_shirt_percentage_l3232_323257

theorem red_shirt_percentage (total_students : ℕ) (blue_percent : ℚ) (green_percent : ℚ) (other_colors : ℕ) 
  (h1 : total_students = 900)
  (h2 : blue_percent = 44 / 100)
  (h3 : green_percent = 10 / 100)
  (h4 : other_colors = 162) :
  (total_students - (blue_percent * total_students + green_percent * total_students + other_colors)) / total_students = 28 / 100 := by
  sorry

end red_shirt_percentage_l3232_323257


namespace chord_slope_l3232_323294

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x^2 = -2*y

/-- Point P on the parabola -/
def P : ℝ × ℝ := (2, -2)

/-- Complementary angles of inclination -/
def complementary_angles (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The theorem statement -/
theorem chord_slope : 
  ∃ (A B : ℝ × ℝ), 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧
    parabola P.1 P.2 ∧
    (∃ (m_PA m_PB : ℝ), complementary_angles m_PA m_PB) ∧
    (B.2 - A.2) / (B.1 - A.1) = 2 := by
  sorry

end chord_slope_l3232_323294


namespace rectangle_cutting_l3232_323256

theorem rectangle_cutting (a b : ℕ) (h_ab : a ≤ b) 
  (h_2 : a * (b - 1) + b * (a - 1) = 940)
  (h_3 : a * (b - 2) + b * (a - 2) = 894) :
  a * (b - 4) + b * (a - 4) = 802 :=
sorry

end rectangle_cutting_l3232_323256


namespace factorial_not_equal_even_factorial_l3232_323276

theorem factorial_not_equal_even_factorial (n m : ℕ) (hn : n > 1) (hm : m > 1) :
  n.factorial ≠ 2^m * m.factorial := by
  sorry

end factorial_not_equal_even_factorial_l3232_323276


namespace original_student_count_l3232_323218

/-- Prove that given the initial average weight, new student's weight, and new average weight,
    the number of original students is 29. -/
theorem original_student_count
  (initial_avg : ℝ)
  (new_student_weight : ℝ)
  (new_avg : ℝ)
  (h1 : initial_avg = 28)
  (h2 : new_student_weight = 22)
  (h3 : new_avg = 27.8)
  : ∃ n : ℕ, n = 29 ∧ 
    (n : ℝ) * initial_avg + new_student_weight = (n + 1 : ℝ) * new_avg :=
by
  sorry

end original_student_count_l3232_323218


namespace tree_planting_ratio_l3232_323231

/-- Represents the number of trees planted by each grade --/
structure TreePlanting where
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ

/-- The conditions of the tree planting activity --/
def treePlantingConditions (t : TreePlanting) : Prop :=
  t.fourth = 30 ∧
  t.sixth = 3 * t.fifth - 30 ∧
  t.fourth + t.fifth + t.sixth = 240

/-- The theorem stating the ratio of trees planted by 5th graders to 4th graders --/
theorem tree_planting_ratio (t : TreePlanting) :
  treePlantingConditions t → (t.fifth : ℚ) / t.fourth = 2 := by
  sorry

end tree_planting_ratio_l3232_323231


namespace binomial_variance_example_l3232_323221

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: The variance of a binomial random variable with n=10 and p=1/4 is 15/8 -/
theorem binomial_variance_example : 
  ∀ ξ : BinomialRV, ξ.n = 10 ∧ ξ.p = 1/4 → variance ξ = 15/8 := by
  sorry

end binomial_variance_example_l3232_323221


namespace polynomial_remainder_theorem_l3232_323248

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^4 - 2*x^3 + 3*x + 1) % (x - 2) = 7 := by
sorry

end polynomial_remainder_theorem_l3232_323248


namespace rose_mother_age_ratio_l3232_323277

/-- Represents the ratio of two ages -/
structure AgeRatio where
  numerator : ℕ
  denominator : ℕ

/-- Rose's age in years -/
def rose_age : ℕ := 25

/-- Rose's mother's age in years -/
def mother_age : ℕ := 75

/-- The ratio of Rose's age to her mother's age -/
def rose_to_mother_ratio : AgeRatio := ⟨1, 3⟩

/-- Theorem stating that the ratio of Rose's age to her mother's age is 1:3 -/
theorem rose_mother_age_ratio : 
  (rose_age : ℚ) / (mother_age : ℚ) = (rose_to_mother_ratio.numerator : ℚ) / (rose_to_mother_ratio.denominator : ℚ) := by
  sorry

end rose_mother_age_ratio_l3232_323277


namespace max_ratio_two_digit_mean60_l3232_323285

-- Define the set of two-digit positive integers
def TwoDigitPositiveInt : Set ℕ := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the mean of x and y
def meanIs60 (x y : ℕ) : Prop := (x + y) / 2 = 60

-- Theorem statement
theorem max_ratio_two_digit_mean60 :
  ∃ (x y : ℕ), x ∈ TwoDigitPositiveInt ∧ y ∈ TwoDigitPositiveInt ∧ meanIs60 x y ∧
  ∀ (a b : ℕ), a ∈ TwoDigitPositiveInt → b ∈ TwoDigitPositiveInt → meanIs60 a b →
  (a : ℚ) / b ≤ 33 / 7 :=
sorry

end max_ratio_two_digit_mean60_l3232_323285


namespace quadratic_quotient_cubic_at_zero_l3232_323281

-- Define the set of integers from 1 to 5
def S : Set ℕ := {1, 2, 3, 4, 5}

-- Define the property that f(n) = n^3 for n in S
def cubic_on_S (f : ℚ → ℚ) : Prop :=
  ∀ n ∈ S, f n = n^3

-- Define the property that f is a quotient of two quadratic polynomials
def is_quadratic_quotient (f : ℚ → ℚ) : Prop :=
  ∃ (p q : ℚ → ℚ),
    (∀ x, ∃ a b c, p x = a*x^2 + b*x + c) ∧
    (∀ x, ∃ d e g, q x = d*x^2 + e*x + g) ∧
    (∀ x, q x ≠ 0) ∧
    (∀ x, f x = p x / q x)

-- The main theorem
theorem quadratic_quotient_cubic_at_zero
  (f : ℚ → ℚ)
  (h1 : is_quadratic_quotient f)
  (h2 : cubic_on_S f) :
  f 0 = 24/17 := by
sorry

end quadratic_quotient_cubic_at_zero_l3232_323281


namespace unique_two_digit_square_l3232_323258

theorem unique_two_digit_square : ∃! n : ℕ,
  10 ≤ n ∧ n < 100 ∧
  1000 ≤ n^2 ∧ n^2 < 10000 ∧
  (∃ a b : ℕ, n^2 = 1100 * a + 11 * b ∧ 0 < a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10) ∧
  n = 88 := by
sorry

end unique_two_digit_square_l3232_323258


namespace prism_sphere_surface_area_l3232_323280

/-- Right triangular prism with specified properties -/
structure RightTriangularPrism where
  -- Base triangle
  AB : ℝ
  AC : ℝ
  angleBAC : ℝ
  -- Prism properties
  volume : ℝ
  -- Ensure all vertices lie on the same spherical surface
  onSphere : Bool

/-- Theorem stating the surface area of the sphere containing the prism -/
theorem prism_sphere_surface_area (p : RightTriangularPrism) 
  (h1 : p.AB = 2)
  (h2 : p.AC = 1)
  (h3 : p.angleBAC = π / 3)  -- 60° in radians
  (h4 : p.volume = Real.sqrt 3)
  (h5 : p.onSphere = true) :
  ∃ (r : ℝ), 4 * π * r^2 = 8 * π := by
    sorry


end prism_sphere_surface_area_l3232_323280


namespace softball_team_ratio_l3232_323235

theorem softball_team_ratio (total_players : ℕ) (more_women : ℕ) : 
  total_players = 15 → more_women = 5 → 
  ∃ (men women : ℕ), 
    men + women = total_players ∧ 
    women = men + more_women ∧ 
    men * 2 = women := by
  sorry

end softball_team_ratio_l3232_323235


namespace no_two_digit_factors_of_1806_l3232_323229

theorem no_two_digit_factors_of_1806 : 
  ¬∃ (a b : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 ∧ a * b = 1806 :=
by sorry

end no_two_digit_factors_of_1806_l3232_323229


namespace spatial_vector_division_not_defined_l3232_323279

-- Define a spatial vector
structure SpatialVector where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define valid operations
def add (v w : SpatialVector) : SpatialVector :=
  { x := v.x + w.x, y := v.y + w.y, z := v.z + w.z }

def sub (v w : SpatialVector) : SpatialVector :=
  { x := v.x - w.x, y := v.y - w.y, z := v.z - w.z }

def scalarProduct (v w : SpatialVector) : ℝ :=
  v.x * w.x + v.y * w.y + v.z * w.z

-- Theorem stating that division is not well-defined for spatial vectors
theorem spatial_vector_division_not_defined :
  ¬ ∃ (f : SpatialVector → SpatialVector → SpatialVector),
    ∀ (v w : SpatialVector), w ≠ { x := 0, y := 0, z := 0 } →
      f v w = { x := v.x / w.x, y := v.y / w.y, z := v.z / w.z } :=
by
  sorry


end spatial_vector_division_not_defined_l3232_323279


namespace megan_folders_l3232_323245

def number_of_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : ℕ :=
  (initial_files - deleted_files) / files_per_folder

theorem megan_folders :
  number_of_folders 93 21 8 = 9 := by
  sorry

end megan_folders_l3232_323245


namespace percentage_of_hindu_boys_l3232_323204

theorem percentage_of_hindu_boys (total_boys : ℕ) (muslim_percent : ℚ) (sikh_percent : ℚ) (other_boys : ℕ) : 
  total_boys = 650 →
  muslim_percent = 44 / 100 →
  sikh_percent = 10 / 100 →
  other_boys = 117 →
  (total_boys - (muslim_percent * total_boys + sikh_percent * total_boys + other_boys)) / total_boys = 28 / 100 := by
sorry

end percentage_of_hindu_boys_l3232_323204


namespace simplify_trig_expression_l3232_323233

theorem simplify_trig_expression : 
  (1 - Real.cos (30 * π / 180)) * (1 + Real.cos (30 * π / 180)) = 1/4 := by sorry

end simplify_trig_expression_l3232_323233


namespace special_ellipse_eccentricity_l3232_323207

/-- An ellipse with the property that the lines connecting the two vertices 
    on the minor axis and one of its foci are perpendicular to each other. -/
structure SpecialEllipse where
  /-- Semi-major axis length -/
  a : ℝ
  /-- Semi-minor axis length -/
  b : ℝ
  /-- Distance from center to focus -/
  c : ℝ
  /-- The ellipse satisfies a² = b² + c² -/
  h1 : a^2 = b^2 + c^2
  /-- The lines connecting the vertices on the minor axis and a focus are perpendicular -/
  h2 : b = c

/-- The eccentricity of a SpecialEllipse is √2/2 -/
theorem special_ellipse_eccentricity (E : SpecialEllipse) : 
  E.c / E.a = Real.sqrt 2 / 2 := by
  sorry

end special_ellipse_eccentricity_l3232_323207


namespace consecutive_discounts_l3232_323287

theorem consecutive_discounts (original_price : ℝ) (h : original_price > 0) :
  let price_after_first_discount := original_price * (1 - 0.3)
  let price_after_second_discount := price_after_first_discount * (1 - 0.2)
  let final_price := price_after_second_discount * (1 - 0.1)
  (original_price - final_price) / original_price = 0.496 := by
sorry

end consecutive_discounts_l3232_323287


namespace system_solution_l3232_323203

theorem system_solution (x y m : ℚ) : 
  (2 * x + 3 * y = 4) → 
  (3 * x + 2 * y = 2 * m - 3) → 
  (x + y = -3/5) → 
  m = -2 := by
sorry

end system_solution_l3232_323203


namespace max_value_sqrt_sum_l3232_323202

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 8) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) ≤ 3 * Real.sqrt 3 :=
by sorry

end max_value_sqrt_sum_l3232_323202


namespace problem_1_problem_2_l3232_323269

/-- Prove that the given expression equals 1/15 -/
theorem problem_1 : 
  (2 * (Nat.factorial 8 / Nat.factorial 3) + 7 * (Nat.factorial 8 / Nat.factorial 4)) / 
  (Nat.factorial 8 - Nat.factorial 9 / Nat.factorial 4) = 1 / 15 := by
  sorry

/-- Prove that the sum of combinations equals C(202, 4) -/
theorem problem_2 : 
  Nat.choose 200 198 + Nat.choose 200 196 + 2 * Nat.choose 200 197 = Nat.choose 202 4 := by
  sorry

end problem_1_problem_2_l3232_323269


namespace power_equality_l3232_323288

theorem power_equality (x y : ℕ) (h1 : 8^x = 2^y) (h2 : x = 3) : y = 9 := by
  sorry

end power_equality_l3232_323288


namespace geometric_sequence_sum_l3232_323286

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry


end geometric_sequence_sum_l3232_323286


namespace math_team_selection_l3232_323249

theorem math_team_selection (boys girls : ℕ) (h1 : boys = 7) (h2 : girls = 10) :
  (boys.choose 4) * (girls.choose 2) = 1575 :=
by sorry

end math_team_selection_l3232_323249


namespace cubic_polynomial_problem_l3232_323237

/-- Given a cubic equation and conditions on a polynomial P,
    prove that P has a specific form. -/
theorem cubic_polynomial_problem (a b c : ℝ) (P : ℝ → ℝ) :
  (a^3 + 5*a^2 + 8*a + 13 = 0) →
  (b^3 + 5*b^2 + 8*b + 13 = 0) →
  (c^3 + 5*c^2 + 8*c + 13 = 0) →
  (∀ x, ∃ p q r s, P x = p*x^3 + q*x^2 + r*x + s) →
  (P a = b + c + 2) →
  (P b = a + c + 2) →
  (P c = a + b + 2) →
  (P (a + b + c) = -22) →
  (∀ x, P x = (19*x^3 + 95*x^2 + 152*x + 247) / 52 - x - 3) :=
by sorry


end cubic_polynomial_problem_l3232_323237


namespace managers_salary_l3232_323208

/-- Given an organization with 20 employees and a manager, prove the manager's salary
    based on the change in average salary. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (avg_increase : ℝ) :
  num_employees = 20 →
  avg_salary = 1600 →
  avg_increase = 100 →
  (num_employees * avg_salary + (avg_salary + avg_increase) * (num_employees + 1)) -
    (num_employees * avg_salary) = 3700 :=
by sorry

end managers_salary_l3232_323208


namespace orange_juice_amount_l3232_323293

theorem orange_juice_amount (total ingredients : ℝ) 
  (strawberries yogurt : ℝ) (h1 : total = 0.5) 
  (h2 : strawberries = 0.2) (h3 : yogurt = 0.1) :
  total - (strawberries + yogurt) = 0.2 := by
  sorry

end orange_juice_amount_l3232_323293


namespace gcd_459_357_l3232_323275

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l3232_323275


namespace cubic_expression_value_l3232_323259

theorem cubic_expression_value (x : ℝ) (h : x^2 + x - 3 = 0) :
  x^3 + 2*x^2 - 2*x + 2 = 5 := by
  sorry

end cubic_expression_value_l3232_323259


namespace tangent_circle_radii_product_l3232_323219

/-- A circle passing through (3,4) and tangent to both axes -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_point : (center.1 - 3)^2 + (center.2 - 4)^2 = radius^2
  tangent_to_x_axis : center.2 = radius
  tangent_to_y_axis : center.1 = radius

/-- The two possible radii of tangent circles -/
def radii : ℝ × ℝ :=
  let a := TangentCircle.radius
  let equation := a^2 - 14*a + 25 = 0
  sorry

theorem tangent_circle_radii_product :
  let (r₁, r₂) := radii
  r₁ * r₂ = 25 := by sorry

end tangent_circle_radii_product_l3232_323219


namespace cube_diagonal_l3232_323270

theorem cube_diagonal (s : ℝ) (h : s > 0) (eq : s^3 + 36*s = 12*s^2) : 
  Real.sqrt (3 * s^2) = 6 * Real.sqrt 3 := by
  sorry

end cube_diagonal_l3232_323270


namespace intersection_complement_theorem_l3232_323210

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {x | x < 0}

theorem intersection_complement_theorem : 
  M ∩ (Set.univ \ N) = {x : ℝ | 0 ≤ x ∧ x < 3} := by sorry

end intersection_complement_theorem_l3232_323210


namespace simplify_cube_roots_l3232_323283

theorem simplify_cube_roots : (512 : ℝ)^(1/3) * (125 : ℝ)^(1/3) = 40 := by sorry

end simplify_cube_roots_l3232_323283


namespace units_digit_of_fraction_l3232_323265

def numerator : ℕ := 30 * 32 * 34 * 36 * 38 * 40
def denominator : ℕ := 2000

theorem units_digit_of_fraction (n d : ℕ) (h : d ≠ 0) :
  (n / d) % 10 = 2 :=
sorry

end units_digit_of_fraction_l3232_323265


namespace smallest_p_satisfying_gcd_conditions_l3232_323264

theorem smallest_p_satisfying_gcd_conditions : 
  ∃ (p : ℕ), 
    p > 1500 ∧ 
    Nat.gcd 90 (p + 150) = 30 ∧ 
    Nat.gcd (p + 90) 150 = 75 ∧ 
    (∀ (q : ℕ), q > 1500 → Nat.gcd 90 (q + 150) = 30 → Nat.gcd (q + 90) 150 = 75 → p ≤ q) ∧
    p = 1560 :=
by sorry

end smallest_p_satisfying_gcd_conditions_l3232_323264


namespace fred_dime_count_l3232_323234

def final_dime_count (initial : ℕ) (borrowed : ℕ) (returned : ℕ) (given : ℕ) : ℕ :=
  initial - borrowed + returned + given

theorem fred_dime_count : final_dime_count 12 4 2 5 = 15 := by
  sorry

end fred_dime_count_l3232_323234


namespace bill_calculation_l3232_323260

def restaurant_bill (num_friends : ℕ) (extra_payment : ℕ) : Prop :=
  num_friends > 0 ∧ 
  ∃ (total_bill : ℕ), 
    total_bill = num_friends * (total_bill / num_friends + extra_payment * (num_friends - 1) / num_friends)

theorem bill_calculation :
  restaurant_bill 6 3 → ∃ (total_bill : ℕ), total_bill = 90 :=
by sorry

end bill_calculation_l3232_323260


namespace frog_jump_expected_time_l3232_323289

/-- A dodecagon with vertices A₁ to A₁₂ -/
structure Dodecagon where
  vertices : Fin 12 → Point

/-- Represents the position of three frogs on the dodecagon -/
structure FrogPositions where
  frog1 : Fin 12
  frog2 : Fin 12
  frog3 : Fin 12

/-- The expected number of minutes until the frogs stop jumping -/
def expected_stop_time (d : Dodecagon) (initial : FrogPositions) : ℚ :=
  16/3

/-- Theorem stating the expected stop time for the given initial configuration -/
theorem frog_jump_expected_time 
  (d : Dodecagon) 
  (initial : FrogPositions) 
  (h1 : initial.frog1 = 4)
  (h2 : initial.frog2 = 8)
  (h3 : initial.frog3 = 12) :
  expected_stop_time d initial = 16/3 :=
sorry

end frog_jump_expected_time_l3232_323289


namespace cyclist_distance_l3232_323253

/-- Represents a cyclist with a given speed -/
structure Cyclist where
  speed : ℝ
  speed_positive : speed > 0

/-- The problem setup -/
def cyclistProblem (c₁ c₂ c₃ : Cyclist) (total_time : ℝ) : Prop :=
  c₁.speed = 12 ∧ c₂.speed = 16 ∧ c₃.speed = 24 ∧
  total_time = 3 ∧
  ∃ (t₁ t₂ t₃ : ℝ),
    t₁ > 0 ∧ t₂ > 0 ∧ t₃ > 0 ∧
    t₁ + t₂ + t₃ = total_time ∧
    c₁.speed * t₁ = c₂.speed * t₂ ∧
    c₂.speed * t₂ = c₃.speed * t₃

theorem cyclist_distance (c₁ c₂ c₃ : Cyclist) (total_time : ℝ) :
  cyclistProblem c₁ c₂ c₃ total_time →
  ∃ (distance : ℝ), distance = 16 ∧
    c₁.speed * (total_time / (1 + c₁.speed / c₂.speed + c₁.speed / c₃.speed)) = distance :=
sorry

end cyclist_distance_l3232_323253


namespace y1_greater_than_y2_l3232_323273

/-- A linear function y = -3x + b -/
def linearFunction (x : ℝ) (b : ℝ) : ℝ := -3 * x + b

/-- Theorem: For a linear function y = -3x + b, if P₁(-3, y₁) and P₂(4, y₂) are points on the graph, then y₁ > y₂ -/
theorem y1_greater_than_y2 (b : ℝ) (y₁ y₂ : ℝ) 
  (h₁ : y₁ = linearFunction (-3) b) 
  (h₂ : y₂ = linearFunction 4 b) : 
  y₁ > y₂ := by
  sorry

end y1_greater_than_y2_l3232_323273


namespace student_A_selection_probability_l3232_323254

/-- The number of students -/
def n : ℕ := 5

/-- The number of students to be selected -/
def k : ℕ := 2

/-- The probability of selecting student A -/
def prob_A : ℚ := 2/5

/-- The combination function -/
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem student_A_selection_probability :
  (combination (n - 1) (k - 1) : ℚ) / (combination n k : ℚ) = prob_A :=
sorry

end student_A_selection_probability_l3232_323254


namespace abs_value_inequality_iff_l3232_323206

theorem abs_value_inequality_iff (a b : ℝ) : a * |a| > b * |b| ↔ a > b := by sorry

end abs_value_inequality_iff_l3232_323206


namespace stack_b_tallest_l3232_323297

/-- Represents the height of a stack of wood blocks -/
def stack_height (num_pieces : ℕ) (block_height : ℝ) : ℝ :=
  (num_pieces : ℝ) * block_height

/-- Proves that stack B is the tallest among the three stacks of wood blocks -/
theorem stack_b_tallest (height_a height_b height_c : ℝ) 
  (h_height_a : height_a = 2)
  (h_height_b : height_b = 1.5)
  (h_height_c : height_c = 2.5) :
  stack_height 11 height_b > stack_height 8 height_a ∧ 
  stack_height 11 height_b > stack_height 6 height_c :=
by
  sorry

#check stack_b_tallest

end stack_b_tallest_l3232_323297


namespace jiwon_walk_distance_l3232_323291

theorem jiwon_walk_distance 
  (sets_of_steps : ℕ) 
  (steps_per_set : ℕ) 
  (distance_per_step : ℝ) : 
  sets_of_steps = 13 → 
  steps_per_set = 90 → 
  distance_per_step = 0.45 → 
  (sets_of_steps * steps_per_set : ℝ) * distance_per_step = 526.5 := by
sorry

end jiwon_walk_distance_l3232_323291


namespace files_remaining_l3232_323261

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 16)
  (h2 : video_files = 48)
  (h3 : deleted_files = 30) :
  music_files + video_files - deleted_files = 34 :=
by sorry

end files_remaining_l3232_323261


namespace tina_total_time_l3232_323223

/-- The time it takes to clean one key, in minutes -/
def time_per_key : ℕ := 3

/-- The number of keys left to clean -/
def keys_to_clean : ℕ := 14

/-- The time it takes to complete the assignment, in minutes -/
def assignment_time : ℕ := 10

/-- The total time it takes for Tina to clean the remaining keys and finish her assignment -/
def total_time : ℕ := time_per_key * keys_to_clean + assignment_time

theorem tina_total_time : total_time = 52 := by
  sorry

end tina_total_time_l3232_323223


namespace green_garden_potato_yield_l3232_323282

/-- Represents Mr. Green's garden and potato yield calculation --/
theorem green_garden_potato_yield :
  let garden_length_steps : ℕ := 25
  let garden_width_steps : ℕ := 30
  let step_length_feet : ℕ := 3
  let non_productive_percentage : ℚ := 1/10
  let yield_per_square_foot : ℚ := 3/4

  let garden_length_feet : ℕ := garden_length_steps * step_length_feet
  let garden_width_feet : ℕ := garden_width_steps * step_length_feet
  let garden_area : ℕ := garden_length_feet * garden_width_feet
  let productive_area : ℚ := garden_area * (1 - non_productive_percentage)
  let total_yield : ℚ := productive_area * yield_per_square_foot

  total_yield = 4556.25 := by sorry

end green_garden_potato_yield_l3232_323282


namespace probability_of_third_six_l3232_323225

theorem probability_of_third_six (p_fair : ℝ) (p_biased : ℝ) (p_other : ℝ) : 
  p_fair = 1/6 →
  p_biased = 2/3 →
  p_other = 1/15 →
  (1/6^2 / (1/6^2 + (2/3)^2)) * (1/6) + ((2/3)^2 / (1/6^2 + (2/3)^2)) * (2/3) = 65/102 := by
  sorry

end probability_of_third_six_l3232_323225


namespace gcd_459_357_l3232_323205

theorem gcd_459_357 : Int.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l3232_323205


namespace mean_temperature_l3232_323250

def temperatures : List ℝ := [-3.5, -2.25, 0, 3.75, 4.5]

theorem mean_temperature : (temperatures.sum / temperatures.length) = 0.5 := by
  sorry

end mean_temperature_l3232_323250


namespace money_distribution_l3232_323239

theorem money_distribution (total : ℕ) (faruk vasim ranjith : ℕ) : 
  faruk + vasim + ranjith = total →
  3 * vasim = 5 * faruk →
  8 * faruk = 3 * ranjith →
  ranjith - faruk = 1500 →
  vasim = 1500 := by
sorry

end money_distribution_l3232_323239


namespace train_bridge_crossing_time_l3232_323200

/-- Proves that a train with given length and speed takes a specific time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) : 
  train_length = 180 ∧ 
  train_speed_kmh = 72 ∧ 
  bridge_length = 270 → 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 22.5 := by
  sorry

end train_bridge_crossing_time_l3232_323200


namespace root_sum_reciprocal_l3232_323247

theorem root_sum_reciprocal (a b c : ℝ) : 
  (a^3 - a - 2 = 0) → 
  (b^3 - b - 2 = 0) → 
  (c^3 - c - 2 = 0) → 
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = -3 / 5) := by
  sorry

end root_sum_reciprocal_l3232_323247


namespace tire_price_proof_l3232_323230

/-- The regular price of a single tire -/
def regular_price : ℚ := 295 / 3

/-- The price of the fourth tire under the offer -/
def fourth_tire_price : ℚ := 5

/-- The total discount applied to the purchase -/
def total_discount : ℚ := 10

/-- The total amount Jane paid for four tires -/
def total_paid : ℚ := 290

/-- Theorem stating that the regular price of a tire is 295/3 given the sale conditions -/
theorem tire_price_proof :
  3 * regular_price + fourth_tire_price - total_discount = total_paid :=
by sorry

end tire_price_proof_l3232_323230


namespace unique_number_with_special_properties_l3232_323241

/-- Returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Returns the product of digits of a natural number -/
def prod_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop := sorry

theorem unique_number_with_special_properties : 
  ∃! x : ℕ, 
    prod_of_digits x = 44 * x - 86868 ∧ 
    is_perfect_cube (sum_of_digits x) ∧
    x = 1989 := by sorry

end unique_number_with_special_properties_l3232_323241


namespace miss_at_least_once_probability_l3232_323238

/-- The probability of missing a target at least once in three shots -/
def miss_at_least_once (P : ℝ) : ℝ :=
  1 - P^3

theorem miss_at_least_once_probability (P : ℝ) 
  (h1 : 0 ≤ P) (h2 : P ≤ 1) : 
  miss_at_least_once P = 1 - P^3 := by
sorry

end miss_at_least_once_probability_l3232_323238


namespace age_difference_l3232_323224

theorem age_difference (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (10 * a + b + 5) = 3 * (10 * b + a + 5)) :
  (10 * a + b) - (10 * b + a) = 45 :=
sorry

end age_difference_l3232_323224


namespace largest_d_for_negative_three_in_range_l3232_323268

/-- The function f(x) = x^2 + 4x + d -/
def f (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- Proposition: The largest value of d such that -3 is in the range of f(x) = x^2 + 4x + d is 1 -/
theorem largest_d_for_negative_three_in_range :
  (∃ (d : ℝ), ∀ (e : ℝ), (∃ (x : ℝ), f d x = -3) → e ≤ d) ∧
  (∃ (x : ℝ), f 1 x = -3) :=
sorry

end largest_d_for_negative_three_in_range_l3232_323268


namespace simplify_expression_l3232_323236

theorem simplify_expression (a b c d x : ℝ) 
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) 
  (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  ((x + a)^4) / ((a - b)*(a - c)*(a - d)) + 
  ((x + b)^4) / ((b - a)*(b - c)*(b - d)) + 
  ((x + c)^4) / ((c - a)*(c - b)*(c - d)) + 
  ((x + d)^4) / ((d - a)*(d - b)*(d - c)) = 
  a + b + c + d + 4*x := by
sorry

end simplify_expression_l3232_323236


namespace pedro_plums_problem_l3232_323267

theorem pedro_plums_problem (total_fruits : ℕ) (total_cost : ℕ) 
  (plum_cost peach_cost : ℕ) (h1 : total_fruits = 32) 
  (h2 : total_cost = 52) (h3 : plum_cost = 2) (h4 : peach_cost = 1) :
  ∃ (plums peaches : ℕ), 
    plums + peaches = total_fruits ∧
    plum_cost * plums + peach_cost * peaches = total_cost ∧
    plums = 20 := by
  sorry

end pedro_plums_problem_l3232_323267


namespace f_increasing_implies_f_one_geq_25_l3232_323217

/-- A function f that is quadratic with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- Theorem stating that if f is increasing on [-2, +∞), then f(1) ≥ 25 -/
theorem f_increasing_implies_f_one_geq_25 (m : ℝ) 
  (h : ∀ x y, -2 ≤ x ∧ x < y → f m x < f m y) : 
  f m 1 ≥ 25 := by
  sorry

end f_increasing_implies_f_one_geq_25_l3232_323217


namespace parallelogram_area_and_perimeter_l3232_323299

/-- Represents a parallelogram EFGH -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  side : ℝ

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Calculate the perimeter of a parallelogram with all sides equal -/
def perimeter (p : Parallelogram) : ℝ := 4 * p.side

/-- Theorem about the area and perimeter of a specific parallelogram -/
theorem parallelogram_area_and_perimeter :
  ∀ (p : Parallelogram),
  p.base = 6 → p.height = 3 → p.side = 5 →
  area p = 18 ∧ perimeter p = 20 := by sorry

end parallelogram_area_and_perimeter_l3232_323299


namespace division_equality_l3232_323274

theorem division_equality (x : ℝ) (h : 2994 / x = 173) : x = 17.3 := by
  sorry

end division_equality_l3232_323274


namespace chemistry_class_average_l3232_323246

theorem chemistry_class_average (n₁ n₂ n₃ n₄ : ℕ) (m₁ m₂ m₃ m₄ : ℚ) :
  let total_students := n₁ + n₂ + n₃ + n₄
  let total_marks := n₁ * m₁ + n₂ * m₂ + n₃ * m₃ + n₄ * m₄
  total_marks / total_students = (n₁ * m₁ + n₂ * m₂ + n₃ * m₃ + n₄ * m₄) / (n₁ + n₂ + n₃ + n₄) :=
by
  sorry

#eval (60 * 50 + 35 * 60 + 45 * 55 + 42 * 45) / (60 + 35 + 45 + 42)

end chemistry_class_average_l3232_323246


namespace f_properties_l3232_323212

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x

theorem f_properties (f : ℝ → ℝ) 
  (h1 : is_odd (λ x => f (x + 1)))
  (h2 : ∀ x, f (x + 4) = f (-x)) :
  is_even f ∧ f 3 = 0 ∧ f 2023 = 0 := by sorry

end f_properties_l3232_323212


namespace inequality_theorem_l3232_323271

/-- A function f: ℝ⁺ → ℝ⁺ such that f(x)/x is increasing on ℝ⁺ -/
def IncreasingRatioFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → x < y → (f x) / x < (f y) / y

theorem inequality_theorem (f : ℝ → ℝ) (h : IncreasingRatioFunction f) 
    (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  2 * ((f a + f b) / (a + b) + (f b + f c) / (b + c) + (f c + f a) / (c + a)) ≥ 
  3 * ((f a + f b + f c) / (a + b + c)) + f a / a + f b / b + f c / c := by
  sorry

end inequality_theorem_l3232_323271


namespace books_per_shelf_l3232_323201

theorem books_per_shelf (mystery_shelves : ℕ) (picture_shelves : ℕ) (total_books : ℕ) :
  mystery_shelves = 6 →
  picture_shelves = 2 →
  total_books = 72 →
  total_books / (mystery_shelves + picture_shelves) = 9 :=
by sorry

end books_per_shelf_l3232_323201


namespace angle_measure_l3232_323214

theorem angle_measure : 
  ∀ x : ℝ, 
  (x + (4 * x + 7) = 90) →  -- Condition 2 (complementary angles)
  x = 83 / 5 := by
  sorry

end angle_measure_l3232_323214


namespace binomial_expansion_coefficient_l3232_323284

theorem binomial_expansion_coefficient (n : ℕ) : 
  (3^2 * (n.choose 2) = 54) → n = 4 := by
  sorry

end binomial_expansion_coefficient_l3232_323284


namespace local_minimum_implies_a_equals_negative_three_l3232_323252

/-- The function f(x) defined as x(x-a)² --/
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - a)^2

/-- The first derivative of f(x) --/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*a*x + a^2

/-- The second derivative of f(x) --/
def f_second_derivative (a : ℝ) (x : ℝ) : ℝ := 6*x - 4*a

theorem local_minimum_implies_a_equals_negative_three (a : ℝ) :
  (f_derivative a (-1) = 0) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - (-1)| < δ → f a x ≥ f a (-1)) →
  a = -3 :=
sorry

end local_minimum_implies_a_equals_negative_three_l3232_323252


namespace geometric_sequence_sum_8_l3232_323227

/-- A geometric sequence with its sum of terms -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_geometric : ∀ n, a (n + 1) = a n * (a 1)⁻¹ * a 2
  sum_formula : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)

/-- The main theorem -/
theorem geometric_sequence_sum_8 (seq : GeometricSequence) 
    (h2 : seq.S 2 = 3)
    (h4 : seq.S 4 = 15) :
  seq.S 8 = 255 := by
  sorry

end geometric_sequence_sum_8_l3232_323227


namespace distance_between_foci_l3232_323262

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 3)^2) + Real.sqrt ((x + 6)^2 + (y - 7)^2) = 26

-- Define the foci
def focus1 : ℝ × ℝ := (4, 3)
def focus2 : ℝ × ℝ := (-6, 7)

-- Theorem statement
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 2 * Real.sqrt 29 := by sorry

end distance_between_foci_l3232_323262


namespace fraction_simplification_l3232_323295

theorem fraction_simplification (m : ℝ) (h : m^2 ≠ 1) :
  (m^2 - m) / (m^2 - 1) = m / (m + 1) := by
  sorry

end fraction_simplification_l3232_323295


namespace sum_of_fractions_in_different_bases_l3232_323243

/-- Converts a number from a given base to base 10 --/
def toBase10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

/-- Rounds a rational number to the nearest integer --/
def roundToNearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem sum_of_fractions_in_different_bases : 
  let a := toBase10 [2, 5, 4] 8
  let b := toBase10 [1, 2] 4
  let c := toBase10 [1, 3, 2] 5
  let d := toBase10 [2, 3] 3
  roundToNearest ((a / b : ℚ) + (c / d : ℚ)) = 33 := by
  sorry

end sum_of_fractions_in_different_bases_l3232_323243


namespace distance_before_stop_correct_concert_drive_distance_l3232_323251

/-- Calculates the distance driven before stopping for gas --/
def distance_before_stop (total_distance : ℕ) (remaining_distance : ℕ) : ℕ :=
  total_distance - remaining_distance

/-- Theorem: The distance driven before stopping for gas is equal to 
    the total distance minus the remaining distance --/
theorem distance_before_stop_correct (total_distance : ℕ) (remaining_distance : ℕ) 
    (h : remaining_distance ≤ total_distance) :
  distance_before_stop total_distance remaining_distance = 
    total_distance - remaining_distance := by
  sorry

/-- Given the total distance and remaining distance, 
    prove that the distance driven before stopping is 32 miles --/
theorem concert_drive_distance :
  distance_before_stop 78 46 = 32 := by
  sorry

end distance_before_stop_correct_concert_drive_distance_l3232_323251


namespace product_abcd_l3232_323272

theorem product_abcd (a b c d : ℚ) : 
  (3 * a + 2 * b + 4 * c + 6 * d = 42) →
  (4 * d + 2 * c = b) →
  (4 * b - 2 * c = a) →
  (d + 2 = c) →
  (a * b * c * d = -(5 * 83 * 46 * 121) / (44 * 44 * 11 * 11)) := by
  sorry

end product_abcd_l3232_323272


namespace interest_difference_implies_principal_l3232_323296

/-- Prove that for a given principal amount, if the difference between compound
    interest (compounded annually) and simple interest over 2 years at 4% per annum
    is 1, then the principal amount is 625. -/
theorem interest_difference_implies_principal (P : ℝ) : 
  P * (1 + 0.04)^2 - P - (P * 0.04 * 2) = 1 → P = 625 := by
  sorry

end interest_difference_implies_principal_l3232_323296


namespace tv_final_price_l3232_323222

/-- Calculates the final price after applying successive discounts -/
def final_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (fun price discount => price * (1 - discount)) original_price

/-- Proves that the final price of a $450 TV after 10%, 20%, and 5% discounts is $307.80 -/
theorem tv_final_price : 
  let original_price : ℝ := 450
  let discounts : List ℝ := [0.1, 0.2, 0.05]
  final_price original_price discounts = 307.80 := by
sorry

#eval final_price 450 [0.1, 0.2, 0.05]

end tv_final_price_l3232_323222


namespace total_cantaloupes_is_65_l3232_323266

/-- The number of cantaloupes grown by Keith -/
def keith_cantaloupes : ℕ := 29

/-- The number of cantaloupes grown by Fred -/
def fred_cantaloupes : ℕ := 16

/-- The number of cantaloupes grown by Jason -/
def jason_cantaloupes : ℕ := 20

/-- The total number of cantaloupes grown by Keith, Fred, and Jason -/
def total_cantaloupes : ℕ := keith_cantaloupes + fred_cantaloupes + jason_cantaloupes

theorem total_cantaloupes_is_65 : total_cantaloupes = 65 := by
  sorry

end total_cantaloupes_is_65_l3232_323266


namespace TI_is_euler_line_l3232_323213

-- Define the basic structures
variable (A B C I T X Y Z : ℝ × ℝ)

-- Define the properties
variable (h1 : is_incenter I A B C)
variable (h2 : is_antigonal_point T I A B C)
variable (h3 : is_antipedal_triangle X Y Z T A B C)

-- Define the Euler line
def euler_line (X Y Z : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define the line TI
def line_TI (T I : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- State the theorem
theorem TI_is_euler_line :
  line_TI T I = euler_line X Y Z :=
sorry

end TI_is_euler_line_l3232_323213


namespace larger_number_proof_l3232_323226

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 1370) (h3 : L = 6 * S + 15) : L = 1641 := by
  sorry

end larger_number_proof_l3232_323226


namespace parking_lot_tires_l3232_323216

/-- Represents the number of tires for a vehicle type -/
structure VehicleTires where
  count : Nat
  wheels : Nat
  spares : Nat

/-- Calculates the total number of tires for a vehicle type -/
def totalTires (v : VehicleTires) : Nat :=
  v.count * (v.wheels + v.spares)

/-- Theorem: The total number of tires in the parking lot is 310 -/
theorem parking_lot_tires :
  let cars := VehicleTires.mk 30 4 1
  let motorcycles := VehicleTires.mk 20 2 2
  let trucks := VehicleTires.mk 10 6 1
  let bicycles := VehicleTires.mk 5 2 0
  totalTires cars + totalTires motorcycles + totalTires trucks + totalTires bicycles = 310 :=
by sorry

end parking_lot_tires_l3232_323216


namespace inscribed_square_area_l3232_323298

/-- Given an equilateral triangle with side length a inscribed in a circle,
    the area of a square inscribed in the same circle is 2a^2/3 -/
theorem inscribed_square_area (a : ℝ) (ha : a > 0) :
  ∃ (R : ℝ), R > 0 ∧
  ∃ (s : ℝ), s > 0 ∧
  (a = R * Real.sqrt 3) ∧ 
  (s = R * Real.sqrt 2) ∧
  (s^2 = 2 * a^2 / 3) :=
sorry

end inscribed_square_area_l3232_323298


namespace smallest_multiple_l3232_323292

theorem smallest_multiple (n : ℕ) : n = 1050 ↔ 
  n > 0 ∧ 
  50 ∣ n ∧ 
  75 ∣ n ∧ 
  ¬(18 ∣ n) ∧ 
  7 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → 50 ∣ m → 75 ∣ m → ¬(18 ∣ m) → 7 ∣ m → m ≥ n :=
by sorry

end smallest_multiple_l3232_323292


namespace inequality_solution_l3232_323242

open Set

def solution_set : Set ℝ :=
  Ioo (-3 : ℝ) (-8/3) ∪ Ioo ((1 - Real.sqrt 89) / 4) ((1 + Real.sqrt 89) / 4)

theorem inequality_solution :
  {x : ℝ | (x - 2) / (x + 3) > (4 * x + 5) / (3 * x + 8) ∧ x ≠ -3 ∧ x ≠ -8/3} = solution_set :=
by sorry

end inequality_solution_l3232_323242


namespace X_4_equivalence_l3232_323240

-- Define the type for a die
def Die : Type := Fin 6

-- Define the type for a pair of dice
def DicePair : Type := Die × Die

-- Define the sum of points on a pair of dice
def sum_points (pair : DicePair) : Nat :=
  pair.1.val + 1 + pair.2.val + 1

-- Define the event X = 4
def X_equals_4 (pair : DicePair) : Prop :=
  sum_points pair = 4

-- Define the event where one die shows 3 and the other shows 1
def one_3_one_1 (pair : DicePair) : Prop :=
  (pair.1.val = 2 ∧ pair.2.val = 0) ∨ (pair.1.val = 0 ∧ pair.2.val = 2)

-- Define the event where both dice show 2
def both_2 (pair : DicePair) : Prop :=
  pair.1.val = 1 ∧ pair.2.val = 1

-- Theorem: X = 4 is equivalent to (one 3 and one 1) or (both 2)
theorem X_4_equivalence (pair : DicePair) :
  X_equals_4 pair ↔ one_3_one_1 pair ∨ both_2 pair :=
sorry

end X_4_equivalence_l3232_323240


namespace popped_kernel_probability_l3232_323278

theorem popped_kernel_probability (white_ratio : ℚ) (yellow_ratio : ℚ) 
  (white_pop_prob : ℚ) (yellow_pop_prob : ℚ) 
  (h1 : white_ratio = 2/3) 
  (h2 : yellow_ratio = 1/3)
  (h3 : white_pop_prob = 1/2)
  (h4 : yellow_pop_prob = 2/3) :
  (white_ratio * white_pop_prob) / (white_ratio * white_pop_prob + yellow_ratio * yellow_pop_prob) = 3/5 := by
  sorry

#check popped_kernel_probability

end popped_kernel_probability_l3232_323278


namespace sin_negative_four_thirds_pi_l3232_323211

theorem sin_negative_four_thirds_pi : 
  Real.sin (-(4/3) * Real.pi) = Real.sqrt 3 / 2 := by
  sorry

end sin_negative_four_thirds_pi_l3232_323211
