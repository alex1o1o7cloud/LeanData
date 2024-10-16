import Mathlib

namespace NUMINAMATH_CALUDE_amp_pamp_theorem_l151_15113

-- Define the & operation
def amp (x : ℝ) : ℝ := 7 - x

-- Define the & prefix operation
def pamp (x : ℝ) : ℝ := x - 7

-- Theorem statement
theorem amp_pamp_theorem : pamp (amp 12) = -12 := by
  sorry

end NUMINAMATH_CALUDE_amp_pamp_theorem_l151_15113


namespace NUMINAMATH_CALUDE_expected_points_bound_l151_15147

/-- A game where players choose an integer between 0 and 10 --/
structure Game where
  /-- The number of players in the game --/
  num_players : ℕ
  /-- The choice of each player --/
  choices : Fin num_players → Fin 11

/-- The points a player receives based on their choice and other players' choices --/
def points (g : Game) (player : Fin g.num_players) : ℕ :=
  if ∃ (other : Fin g.num_players), other ≠ player ∧ g.choices other = g.choices player
  then 0
  else g.choices player

/-- The expected value of points for a player in the game --/
def expected_points (g : Game) (player : Fin g.num_players) : ℚ :=
  (points g player : ℚ) / g.num_players

theorem expected_points_bound (g : Game) (player : Fin g.num_players) :
  expected_points g player ≤ g.choices player := by
  sorry

end NUMINAMATH_CALUDE_expected_points_bound_l151_15147


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l151_15196

theorem cube_sum_given_sum_and_product (x y : ℝ) :
  x + y = 10 → x * y = 15 → x^3 + y^3 = 550 := by sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l151_15196


namespace NUMINAMATH_CALUDE_min_product_of_three_l151_15191

def S : Finset ℤ := {-10, -7, -5, -3, 0, 2, 4, 6, 8}

theorem min_product_of_three (a b c : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a * b * c ≥ -480 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_l151_15191


namespace NUMINAMATH_CALUDE_percentage_calculation_l151_15122

theorem percentage_calculation (x : ℝ) (h : x ≠ 0) : (x + 0.5 * x) / (0.75 * x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l151_15122


namespace NUMINAMATH_CALUDE_problem_part1_problem_part2_l151_15185

-- Part 1
theorem problem_part1 (m n : ℕ) (h1 : m > 0) (h2 : n > 0) 
  (h3 : 3 * m + 2 * n = 225) (h4 : Nat.gcd m n = 15) : 
  m + n = 105 := by
  sorry

-- Part 2
theorem problem_part2 (m n : ℕ) (h1 : m > 0) (h2 : n > 0) 
  (h3 : 3 * m + 2 * n = 225) (h4 : Nat.lcm m n = 45) : 
  m + n = 90 := by
  sorry

end NUMINAMATH_CALUDE_problem_part1_problem_part2_l151_15185


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l151_15159

theorem sufficient_not_necessary (a b : ℝ) :
  (((a - b) * a^2 < 0) → (a < b)) ∧
  (∃ a b : ℝ, (a < b) ∧ ((a - b) * a^2 ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l151_15159


namespace NUMINAMATH_CALUDE_complex_square_eq_abs_square_iff_real_l151_15161

open Complex

theorem complex_square_eq_abs_square_iff_real (z : ℂ) :
  (z - 1)^2 = abs (z - 1)^2 ↔ z.im = 0 :=
sorry

end NUMINAMATH_CALUDE_complex_square_eq_abs_square_iff_real_l151_15161


namespace NUMINAMATH_CALUDE_equation_system_solution_l151_15108

def solution_set (a b c x y z : ℝ) : Prop :=
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = c) ∨
  (x = a ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = b ∧ z = 0)

theorem equation_system_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∀ x y z : ℝ,
    (a * x + b * y = (x - y)^2 ∧
     b * y + c * z = (y - z)^2 ∧
     c * z + a * x = (z - x)^2) ↔
    solution_set a b c x y z :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l151_15108


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l151_15186

theorem complex_modulus_problem (z : ℂ) : z = (2 * I) / (1 - I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l151_15186


namespace NUMINAMATH_CALUDE_equation_holds_l151_15100

theorem equation_holds (x y z : ℝ) (h : (x - z)^2 - 4*(x - y)*(y - z) = 0) :
  z + x - 2*y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l151_15100


namespace NUMINAMATH_CALUDE_women_in_room_l151_15102

theorem women_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (initial_men + 2) = 14 →
  24 = 2 * (initial_women - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_women_in_room_l151_15102


namespace NUMINAMATH_CALUDE_tangent_half_angle_identity_l151_15120

theorem tangent_half_angle_identity (α : Real) (m : Real) 
  (h : Real.tan (α / 2) = m) : 
  (1 - 2 * Real.sin (α / 2) ^ 2) / (1 + Real.sin α) = (1 - m) / (1 + m) := by
  sorry

end NUMINAMATH_CALUDE_tangent_half_angle_identity_l151_15120


namespace NUMINAMATH_CALUDE_work_completion_time_l151_15127

/-- Given a work that can be completed by person A in 20 days, and 0.375 of the work
    can be completed by A and B together in 5 days, prove that person B can complete
    the work alone in 40 days. -/
theorem work_completion_time (work_rate_A work_rate_B : ℝ) : 
  work_rate_A = 1 / 20 →
  5 * (work_rate_A + work_rate_B) = 0.375 →
  1 / work_rate_B = 40 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l151_15127


namespace NUMINAMATH_CALUDE_quadratic_inequality_l151_15162

theorem quadratic_inequality (x : ℝ) : x ^ 2 - 4 * x - 21 ≤ 0 ↔ x ∈ Set.Icc (-3) 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l151_15162


namespace NUMINAMATH_CALUDE_certain_amount_added_l151_15128

theorem certain_amount_added (x y : ℝ) : 
  x = 18 → 3 * (2 * x + y) = 123 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_added_l151_15128


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l151_15192

def A : Set ℕ := {0, 1, 2, 3}

def B : Set ℕ := {x | ∃ a ∈ A, x = 3 * a}

theorem intersection_of_A_and_B : A ∩ B = {0, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l151_15192


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_three_shared_l151_15133

theorem greatest_common_divisor_of_three_shared (n : ℕ) : 
  (∃ (d1 d2 d3 : ℕ), d1 < d2 ∧ d2 < d3 ∧ 
   d1 ∣ 120 ∧ d2 ∣ 120 ∧ d3 ∣ 120 ∧
   d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧
   (∀ (x : ℕ), x ∣ 120 ∧ x ∣ n → x = d1 ∨ x = d2 ∨ x = d3)) →
  (∃ (d : ℕ), d ∣ 120 ∧ d ∣ n ∧ d = 9 ∧ 
   (∀ (x : ℕ), x ∣ 120 ∧ x ∣ n → x ≤ d)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_three_shared_l151_15133


namespace NUMINAMATH_CALUDE_sampling_survey_most_suitable_l151_15140

/-- Represents a survey method -/
inductive SurveyMethod
  | ComprehensiveSurvey
  | SamplingSurvey

/-- Represents the characteristics of a nationwide survey -/
structure NationwideSurvey where
  population : Nat  -- Number of students
  geographical_spread : Nat  -- Measure of how spread out the students are
  resource_constraints : Nat  -- Measure of available resources for the survey

/-- Determines the most suitable survey method for a given nationwide survey -/
def most_suitable_survey_method (survey : NationwideSurvey) : SurveyMethod :=
  sorry

/-- Theorem stating that for a nationwide survey of primary and secondary school students' 
    homework time, the most suitable survey method is a sampling survey -/
theorem sampling_survey_most_suitable (survey : NationwideSurvey) :
  most_suitable_survey_method survey = SurveyMethod.SamplingSurvey :=
  sorry

end NUMINAMATH_CALUDE_sampling_survey_most_suitable_l151_15140


namespace NUMINAMATH_CALUDE_f_is_even_g_is_odd_h_is_neither_l151_15145

-- Define the functions
def f (x : ℝ) : ℝ := 1 + x^2 + x^4
def g (x : ℝ) : ℝ := x + x^3 + x^5
def h (x : ℝ) : ℝ := 1 + x + x^2 + x^3 + x^4

-- Define properties of even and odd functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statements
theorem f_is_even : is_even f := by sorry

theorem g_is_odd : is_odd g := by sorry

theorem h_is_neither : ¬(is_even h) ∧ ¬(is_odd h) := by sorry

end NUMINAMATH_CALUDE_f_is_even_g_is_odd_h_is_neither_l151_15145


namespace NUMINAMATH_CALUDE_coat_drive_l151_15165

theorem coat_drive (total_coats : ℕ) (elementary_coats : ℕ) (high_school_coats : ℕ) :
  total_coats = 9437 →
  elementary_coats = 2515 →
  high_school_coats = total_coats - elementary_coats →
  high_school_coats = 6922 :=
by
  sorry

end NUMINAMATH_CALUDE_coat_drive_l151_15165


namespace NUMINAMATH_CALUDE_distance_to_focus_of_parabola_l151_15168

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus_of_parabola (x y : ℝ) :
  x^2 = 2*y →  -- Parabola equation
  y = 3 →      -- Ordinate of point P
  (y + 1/4) = 7/2  -- Distance to focus
  := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_of_parabola_l151_15168


namespace NUMINAMATH_CALUDE_cake_distribution_l151_15194

theorem cake_distribution (n : ℕ) (initial_cakes : ℕ) : 
  n = 5 →
  initial_cakes = 2 * (n * (n - 1)) →
  initial_cakes = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_cake_distribution_l151_15194


namespace NUMINAMATH_CALUDE_student_arrangement_count_l151_15167

def num_students : Nat := 6

def leftmost_students : Finset Char := {'A', 'B'}

def all_students : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

theorem student_arrangement_count :
  (leftmost_students.card * (num_students - 1).factorial) +
  ((all_students.card - 2) * (num_students - 2).factorial) = 216 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l151_15167


namespace NUMINAMATH_CALUDE_quadratic_inequality_l151_15129

theorem quadratic_inequality (y : ℝ) : y^2 - 9*y + 14 < 0 ↔ 2 < y ∧ y < 7 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l151_15129


namespace NUMINAMATH_CALUDE_unique_divisible_by_7_l151_15119

def is_divisible_by_7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

theorem unique_divisible_by_7 : 
  (is_divisible_by_7 41126) ∧ 
  (∀ B : ℕ, B < 10 → B ≠ 1 → ¬(is_divisible_by_7 (40000 + 2000 * B + 100 * B + 26))) :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_7_l151_15119


namespace NUMINAMATH_CALUDE_one_positive_number_l151_15166

theorem one_positive_number (numbers : List ℝ := [3, -2.1, -1/2, 0, -9]) :
  (numbers.filter (λ x => x > 0)).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_positive_number_l151_15166


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l151_15123

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 2*m*x + 16 = (x - a)^2) → (m = 4 ∨ m = -4) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l151_15123


namespace NUMINAMATH_CALUDE_subset_intersection_count_l151_15173

-- Define the set S with n elements
variable (n : ℕ)
variable (S : Finset (Fin n))

-- Define k subsets of S
variable (k : ℕ)
variable (A : Fin k → Finset (Fin n))

-- Conditions
variable (h1 : ∀ i, A i ⊆ S)
variable (h2 : ∀ i j, i ≠ j → (A i ∩ A j).Nonempty)
variable (h3 : ∀ X, X ⊆ S → (∀ i, (X ∩ A i).Nonempty) → ∃ i, X = A i)

-- Theorem statement
theorem subset_intersection_count : k = 2^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_subset_intersection_count_l151_15173


namespace NUMINAMATH_CALUDE_variance_of_transformed_data_l151_15137

variable (x : Fin 10 → ℝ)

def variance (data : Fin 10 → ℝ) : ℝ := sorry

def transform (data : Fin 10 → ℝ) : Fin 10 → ℝ := 
  fun i => 2 * data i - 1

theorem variance_of_transformed_data 
  (h : variance x = 8) : 
  variance (transform x) = 32 := by sorry

end NUMINAMATH_CALUDE_variance_of_transformed_data_l151_15137


namespace NUMINAMATH_CALUDE_tangent_line_and_intersections_l151_15125

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 1)^2
noncomputable def g (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_line_and_intersections :
  ∃ (x₀ : ℝ),
    (∀ x, g x₀ + (x - x₀) * ((x₀ + 1) * Real.exp x₀) = -Real.exp (-2) * (x + 4)) ∧
    (∀ a : ℝ,
      (a ≥ 0 → ∃! x, f a x = g x) ∧
      (a < 0 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = g x₁ ∧ f a x₂ = g x₂ ∧
        ∀ x, f a x = g x → x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_intersections_l151_15125


namespace NUMINAMATH_CALUDE_cube_properties_l151_15187

/-- A cube is a convex polyhedron with specific properties -/
structure Cube where
  vertices : ℕ
  faces : ℕ
  edges : ℕ

/-- Euler's formula for convex polyhedrons -/
def euler_formula (c : Cube) : Prop :=
  c.vertices - c.edges + c.faces = 2

/-- Theorem stating the properties of a cube -/
theorem cube_properties : ∃ (c : Cube), c.vertices = 8 ∧ c.faces = 6 ∧ c.edges = 12 ∧ euler_formula c := by
  sorry

end NUMINAMATH_CALUDE_cube_properties_l151_15187


namespace NUMINAMATH_CALUDE_remaining_perimeter_l151_15138

/-- The perimeter of the remaining shape after cutting out two squares from a rectangle. -/
theorem remaining_perimeter (rectangle_length rectangle_width square1_side square2_side : ℕ) :
  rectangle_length = 50 ∧ 
  rectangle_width = 20 ∧ 
  square1_side = 12 ∧ 
  square2_side = 4 →
  2 * (rectangle_length + rectangle_width) + 4 * square1_side + 4 * square2_side = 204 := by
  sorry

end NUMINAMATH_CALUDE_remaining_perimeter_l151_15138


namespace NUMINAMATH_CALUDE_negative_roots_existence_l151_15170

theorem negative_roots_existence (p : ℝ) :
  p > 3/5 →
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  x₁^4 + 2*p*x₁^3 + p*x₁^2 + x₁^2 + 2*p*x₁ + 1 = 0 ∧
  x₂^4 + 2*p*x₂^3 + p*x₂^2 + x₂^2 + 2*p*x₂ + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_roots_existence_l151_15170


namespace NUMINAMATH_CALUDE_max_product_of_areas_l151_15164

theorem max_product_of_areas (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a + b + c + d = 1 →
  a * b * c * d ≤ 1 / 256 :=
sorry

end NUMINAMATH_CALUDE_max_product_of_areas_l151_15164


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l151_15110

theorem smallest_k_with_remainder_one : ∃! k : ℕ, 
  k > 1 ∧ 
  k % 13 = 1 ∧ 
  k % 7 = 1 ∧ 
  k % 5 = 1 ∧ 
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 13 = 1 ∧ m % 7 = 1 ∧ m % 5 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l151_15110


namespace NUMINAMATH_CALUDE_smallest_stairs_l151_15157

theorem smallest_stairs (n : ℕ) : 
  (n > 10) ∧ 
  (n % 6 = 4) ∧ 
  (n % 7 = 3) ∧ 
  (∀ m : ℕ, m > 10 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_stairs_l151_15157


namespace NUMINAMATH_CALUDE_pen_price_ratio_l151_15169

/-- The ratio of gel pen price to ballpoint pen price -/
def gel_to_ballpoint_ratio : ℝ := 8

theorem pen_price_ratio :
  ∀ (x y : ℕ) (b g : ℝ),
  x > 0 → y > 0 → b > 0 → g > 0 →
  (x + y : ℝ) * g = 4 * (x * b + y * g) →
  (x + y : ℝ) * b = (1 / 2) * (x * b + y * g) →
  g / b = gel_to_ballpoint_ratio :=
by sorry

end NUMINAMATH_CALUDE_pen_price_ratio_l151_15169


namespace NUMINAMATH_CALUDE_game_points_difference_l151_15106

theorem game_points_difference (layla_points nahima_points total_points : ℕ) : 
  layla_points = 70 → total_points = 112 → layla_points + nahima_points = total_points →
  layla_points - nahima_points = 28 := by
sorry

end NUMINAMATH_CALUDE_game_points_difference_l151_15106


namespace NUMINAMATH_CALUDE_straight_row_not_tetrahedron_l151_15132

/-- A pattern of squares that can be folded -/
structure FoldablePattern :=
  (squares : ℕ)
  (arrangement : String)

/-- Properties of a regular tetrahedron -/
structure RegularTetrahedron :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

/-- Definition of a straight row pattern -/
def straightRowPattern : FoldablePattern :=
  { squares := 4,
    arrangement := "straight row" }

/-- Definition of a regular tetrahedron -/
def regularTetrahedron : RegularTetrahedron :=
  { faces := 4,
    edges := 6,
    vertices := 4 }

/-- Function to check if a pattern can be folded into a regular tetrahedron -/
def canFoldToTetrahedron (pattern : FoldablePattern) : Prop :=
  ∃ (t : RegularTetrahedron), t = regularTetrahedron

/-- Theorem stating that a straight row pattern cannot be folded into a regular tetrahedron -/
theorem straight_row_not_tetrahedron :
  ¬(canFoldToTetrahedron straightRowPattern) :=
sorry

end NUMINAMATH_CALUDE_straight_row_not_tetrahedron_l151_15132


namespace NUMINAMATH_CALUDE_identical_permutations_of_increasing_sum_l151_15107

/-- A strictly increasing finite sequence of real numbers -/
def StrictlyIncreasingSeq (a : Fin n → ℝ) : Prop :=
  ∀ i j : Fin n, i < j → a i < a j

/-- A permutation of indices -/
def IsPermutation (σ : Fin n → Fin n) : Prop :=
  Function.Bijective σ

theorem identical_permutations_of_increasing_sum
  (a : Fin n → ℝ) (σ : Fin n → Fin n)
  (h_inc : StrictlyIncreasingSeq a)
  (h_perm : IsPermutation σ)
  (h_sum_inc : StrictlyIncreasingSeq (fun i => a i + a (σ i))) :
  ∀ i, a i = a (σ i) := by
sorry

end NUMINAMATH_CALUDE_identical_permutations_of_increasing_sum_l151_15107


namespace NUMINAMATH_CALUDE_dice_probability_l151_15190

def standard_die : Finset ℕ := Finset.range 6
def eight_sided_die : Finset ℕ := Finset.range 8

def prob_not_one (die : Finset ℕ) : ℚ :=
  (die.filter (· ≠ 1)).card / die.card

theorem dice_probability : 
  (prob_not_one standard_die)^2 * (prob_not_one eight_sided_die) = 175/288 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l151_15190


namespace NUMINAMATH_CALUDE_onions_sum_to_285_l151_15199

/-- The total number of onions grown by Sara, Sally, Fred, Amy, and Matthew -/
def total_onions (sara sally fred amy matthew : ℕ) : ℕ :=
  sara + sally + fred + amy + matthew

/-- Theorem stating that the total number of onions grown is 285 -/
theorem onions_sum_to_285 :
  total_onions 40 55 90 25 75 = 285 := by
  sorry

end NUMINAMATH_CALUDE_onions_sum_to_285_l151_15199


namespace NUMINAMATH_CALUDE_attendance_difference_l151_15126

/-- Proves that the difference in student attendance between the second and first day is 40 --/
theorem attendance_difference (total_students : ℕ) (total_absent : ℕ) 
  (absent_day1 absent_day2 absent_day3 : ℕ) :
  total_students = 280 →
  total_absent = 240 →
  absent_day1 + absent_day2 + absent_day3 = total_absent →
  absent_day2 = 2 * absent_day3 →
  absent_day3 = total_students / 7 →
  absent_day2 < absent_day1 →
  (total_students - absent_day2) - (total_students - absent_day1) = 40 := by
  sorry

end NUMINAMATH_CALUDE_attendance_difference_l151_15126


namespace NUMINAMATH_CALUDE_test_score_theorem_l151_15184

/-- Proves that given a test with 30 questions, where each correct answer awards 3 points
    and each incorrect answer deducts 1 point, if the total score is 78 points,
    then the number of correctly answered questions is 27. -/
theorem test_score_theorem (total_questions : Nat) (correct_points : Nat) (incorrect_points : Nat) 
    (total_score : Int) (correct_answers : Nat) :
  total_questions = 30 →
  correct_points = 3 →
  incorrect_points = 1 →
  total_score = 78 →
  (correct_answers : Int) * correct_points - 
    (total_questions - correct_answers) * incorrect_points = total_score →
  correct_answers = 27 := by
  sorry

#check test_score_theorem

end NUMINAMATH_CALUDE_test_score_theorem_l151_15184


namespace NUMINAMATH_CALUDE_lucca_bread_problem_l151_15130

/-- The fraction of remaining bread Lucca ate on the second day -/
def second_day_fraction (initial_bread : ℕ) (first_day_fraction : ℚ) (third_day_fraction : ℚ) (remaining_bread : ℕ) : ℚ :=
  let remaining_after_first := initial_bread - initial_bread * first_day_fraction
  2 / 5

/-- Theorem stating the fraction of remaining bread Lucca ate on the second day -/
theorem lucca_bread_problem (initial_bread : ℕ) (first_day_fraction : ℚ) (third_day_fraction : ℚ) (remaining_bread : ℕ)
    (h1 : initial_bread = 200)
    (h2 : first_day_fraction = 1 / 4)
    (h3 : third_day_fraction = 1 / 2)
    (h4 : remaining_bread = 45) :
  second_day_fraction initial_bread first_day_fraction third_day_fraction remaining_bread = 2 / 5 := by
  sorry

#eval second_day_fraction 200 (1/4) (1/2) 45

end NUMINAMATH_CALUDE_lucca_bread_problem_l151_15130


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_Q_perfect_square_l151_15136

/-- The polynomial Q as a function of x -/
def Q (x : ℤ) : ℤ := x^4 + 5*x^3 + 10*x^2 + 5*x + 56

/-- Theorem stating that there are no integer solutions for x such that Q(x) is a perfect square -/
theorem no_integer_solutions_for_Q_perfect_square :
  ∀ x : ℤ, ¬∃ k : ℤ, Q x = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_Q_perfect_square_l151_15136


namespace NUMINAMATH_CALUDE_caramel_distribution_solution_l151_15163

def caramel_distribution (a b c d : ℕ) : Prop :=
  a + b + c + d = 26 ∧
  ∃ (x y : ℕ),
    a = x + y ∧
    b = 2 * x ∧
    c = x + y ∧
    d = x + (2 * y + x) ∧
    x > 0 ∧ y > 0

theorem caramel_distribution_solution :
  caramel_distribution 5 6 5 10 :=
sorry

end NUMINAMATH_CALUDE_caramel_distribution_solution_l151_15163


namespace NUMINAMATH_CALUDE_train_braking_problem_l151_15151

/-- The distance function for the train's motion during braking -/
def S (t : ℝ) : ℝ := 27 * t - 0.45 * t^2

/-- The time when the train stops -/
def stop_time : ℝ := 30

/-- The distance traveled during the braking period -/
def total_distance : ℝ := 405

theorem train_braking_problem :
  (∀ t, t > stop_time → S t < S stop_time) ∧
  S stop_time = total_distance := by
  sorry

end NUMINAMATH_CALUDE_train_braking_problem_l151_15151


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_l151_15144

theorem polygon_interior_angle_sum (n : ℕ) (h1 : n > 2) (h2 : 40 * n = 360) :
  (n - 2) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_l151_15144


namespace NUMINAMATH_CALUDE_seventh_root_unity_sum_l151_15112

theorem seventh_root_unity_sum (q : ℂ) (h : q^7 = 1) :
  q / (1 + q^2) + q^2 / (1 + q^4) + q^3 / (1 + q^6) = 
    if q = 1 then 3/2 else -2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_sum_l151_15112


namespace NUMINAMATH_CALUDE_increasing_function_range_l151_15131

theorem increasing_function_range (f : ℝ → ℝ) (h_increasing : ∀ x y, x < y → x ∈ [-1, 3] → y ∈ [-1, 3] → f x < f y) :
  ∀ a : ℝ, f a > f (1 - 2 * a) → a ∈ Set.Ioo (1/3) 1 := by
sorry

end NUMINAMATH_CALUDE_increasing_function_range_l151_15131


namespace NUMINAMATH_CALUDE_triangle_inequality_relationships_l151_15134

/-- A triangle with perimeter, circumradius, and inradius -/
structure Triangle where
  perimeter : ℝ
  circumradius : ℝ
  inradius : ℝ
  perimeter_pos : 0 < perimeter
  circumradius_pos : 0 < circumradius
  inradius_pos : 0 < inradius

/-- Theorem stating that none of the given relationships hold universally for all triangles -/
theorem triangle_inequality_relationships (t : Triangle) : 
  ¬(∀ t : Triangle, t.perimeter > t.circumradius + t.inradius) ∧ 
  ¬(∀ t : Triangle, t.perimeter ≤ t.circumradius + t.inradius) ∧ 
  ¬(∀ t : Triangle, 1/6 < t.circumradius + t.inradius ∧ t.circumradius + t.inradius < 6*t.perimeter) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_relationships_l151_15134


namespace NUMINAMATH_CALUDE_candy_jar_problem_l151_15158

/-- Represents the number of candies in a jar -/
structure JarContents where
  red : ℕ
  yellow : ℕ

/-- The problem statement -/
theorem candy_jar_problem :
  ∀ (jar1 jar2 : JarContents),
    -- Both jars have the same total number of candies
    (jar1.red + jar1.yellow = jar2.red + jar2.yellow) →
    -- Jar 1 has a red to yellow ratio of 7:3
    (7 * jar1.yellow = 3 * jar1.red) →
    -- Jar 2 has a red to yellow ratio of 5:4
    (5 * jar2.yellow = 4 * jar2.red) →
    -- The total number of yellow candies is 108
    (jar1.yellow + jar2.yellow = 108) →
    -- The difference in red candies between Jar 1 and Jar 2 is 21
    (jar1.red - jar2.red = 21) :=
by sorry

end NUMINAMATH_CALUDE_candy_jar_problem_l151_15158


namespace NUMINAMATH_CALUDE_train_passing_time_l151_15177

/-- The time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 150 →
  train_speed = 62 * (1000 / 3600) →
  man_speed = 8 * (1000 / 3600) →
  (train_length / (train_speed - man_speed)) = 10 :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l151_15177


namespace NUMINAMATH_CALUDE_larry_initial_amount_l151_15104

def initial_amount (lunch_cost brother_gift current_amount : ℕ) : ℕ :=
  lunch_cost + brother_gift + current_amount

theorem larry_initial_amount :
  initial_amount 5 2 15 = 22 :=
by sorry

end NUMINAMATH_CALUDE_larry_initial_amount_l151_15104


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l151_15148

theorem sin_2alpha_value (α : Real) 
  (h : Real.sin (π - α) = -2 * Real.sin (π / 2 + α)) : 
  Real.sin (2 * α) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l151_15148


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l151_15116

theorem complex_fraction_sum : (1 / (1 - Complex.I)) + (Complex.I / (1 + Complex.I)) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l151_15116


namespace NUMINAMATH_CALUDE_m_squared_plus_inverse_squared_plus_six_l151_15154

theorem m_squared_plus_inverse_squared_plus_six (m : ℝ) (h : m + 1/m = 10) : 
  m^2 + 1/m^2 + 6 = 104 := by
  sorry

end NUMINAMATH_CALUDE_m_squared_plus_inverse_squared_plus_six_l151_15154


namespace NUMINAMATH_CALUDE_draw_ball_one_probability_l151_15139

/-- The number of balls in the box -/
def total_balls : ℕ := 5

/-- The number of balls drawn -/
def drawn_balls : ℕ := 2

/-- The number of ways to draw the specific ball (number 1) -/
def favorable_outcomes : ℕ := 4

/-- The total number of ways to draw 2 balls out of 5 -/
def total_outcomes : ℕ := 10

/-- The probability of drawing ball number 1 -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem draw_ball_one_probability :
  probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_draw_ball_one_probability_l151_15139


namespace NUMINAMATH_CALUDE_brad_start_time_l151_15143

/-- Proves that Brad started running 9 hours after Maxwell started walking -/
theorem brad_start_time (maxwell_speed : ℝ) (brad_speed : ℝ) (total_distance : ℝ) (maxwell_time : ℝ) :
  maxwell_speed = 4 →
  brad_speed = 6 →
  total_distance = 94 →
  maxwell_time = 10 →
  total_distance - maxwell_speed * maxwell_time = brad_speed * (maxwell_time - 9) :=
by
  sorry

#check brad_start_time

end NUMINAMATH_CALUDE_brad_start_time_l151_15143


namespace NUMINAMATH_CALUDE_odd_function_implies_k_equals_two_inequality_range_minimum_value_of_g_l151_15181

noncomputable section

variable (a : ℝ) (k : ℝ)

def f (x : ℝ) : ℝ := a^x - (k-1) * a^(-x)

theorem odd_function_implies_k_equals_two
  (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x, f a k x = -f a k (-x)) :
  k = 2 := by sorry

theorem inequality_range
  (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : f a 2 1 < 0) :
  (∀ t, (∀ x, f a 2 (x^2 + t*x) + f a 2 (4-x) < 0) ↔ -3 < t ∧ t < 5) := by sorry

theorem minimum_value_of_g
  (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : f a 2 1 = 3/2) :
  (∃ x_min : ℝ, x_min ∈ Set.Ici 1 ∧
    ∀ x, x ∈ Set.Ici 1 →
      a^(2*x) + a^(-2*x) - 2 * f a 2 x ≥ a^(2*x_min) + a^(-2*x_min) - 2 * f a 2 x_min) ∧
  (∃ x_0 : ℝ, x_0 ∈ Set.Ici 1 ∧ a^(2*x_0) + a^(-2*x_0) - 2 * f a 2 x_0 = 5/4) := by sorry

end NUMINAMATH_CALUDE_odd_function_implies_k_equals_two_inequality_range_minimum_value_of_g_l151_15181


namespace NUMINAMATH_CALUDE_stock_market_investment_l151_15193

theorem stock_market_investment (P : ℝ) (x : ℝ) (h : P > 0) :
  (P + x / 100 * P) * (1 - 30 / 100) = P * (1 + 4.999999999999982 / 100) →
  x = 50 := by
sorry

end NUMINAMATH_CALUDE_stock_market_investment_l151_15193


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l151_15195

theorem greatest_integer_solution (x : ℤ) : (7 - 3 * x > 20) ↔ x ≤ -5 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l151_15195


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l151_15141

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E F : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l151_15141


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l151_15142

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  a = (1, -Real.sqrt 3) →
  Real.sqrt ((a.1 ^ 2 + a.2 ^ 2)) = 2 →
  Real.sqrt ((b.1 ^ 2 + b.2 ^ 2)) = 1 →
  a.1 * b.1 + a.2 * b.2 = -1 →
  Real.sqrt (((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2)) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l151_15142


namespace NUMINAMATH_CALUDE_cistern_problem_solution_l151_15188

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width water_breadth : ℝ) : ℝ :=
  let bottom_area := length * width
  let longer_side_area := 2 * length * water_breadth
  let shorter_side_area := 2 * width * water_breadth
  bottom_area + longer_side_area + shorter_side_area

/-- Theorem stating that the wet surface area of the given cistern is 121.5 m² -/
theorem cistern_problem_solution :
  cistern_wet_surface_area 9 6 2.25 = 121.5 := by
  sorry

#eval cistern_wet_surface_area 9 6 2.25

end NUMINAMATH_CALUDE_cistern_problem_solution_l151_15188


namespace NUMINAMATH_CALUDE_solar_usage_exponential_growth_l151_15172

/-- Represents the percentage of households using solar energy -/
def SolarUsage : ℕ → ℝ
  | 2000 => 6
  | 2010 => 12
  | 2015 => 24
  | 2020 => 48
  | _ => 0  -- For years not specified, we return 0

/-- Checks if the growth is exponential between two time points -/
def IsExponentialGrowth (t₁ t₂ : ℕ) : Prop :=
  ∃ (r : ℝ), r > 1 ∧ SolarUsage t₂ = SolarUsage t₁ * r^(t₂ - t₁)

/-- Theorem stating that the solar usage growth is exponential -/
theorem solar_usage_exponential_growth :
  IsExponentialGrowth 2000 2010 ∧
  IsExponentialGrowth 2010 2015 ∧
  IsExponentialGrowth 2015 2020 :=
sorry

end NUMINAMATH_CALUDE_solar_usage_exponential_growth_l151_15172


namespace NUMINAMATH_CALUDE_marathon_training_percentage_l151_15103

theorem marathon_training_percentage (total_miles : ℝ) (day3_miles : ℝ) 
  (h1 : total_miles = 70)
  (h2 : day3_miles = 28) : 
  ∃ (p : ℝ), 
    p * total_miles + 0.5 * (total_miles - p * total_miles) + day3_miles = total_miles ∧ 
    p = 0.2 := by
sorry

end NUMINAMATH_CALUDE_marathon_training_percentage_l151_15103


namespace NUMINAMATH_CALUDE_train_length_l151_15109

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 12 → 
  ∃ length_m : ℝ, abs (length_m - 200.04) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l151_15109


namespace NUMINAMATH_CALUDE_second_player_wins_l151_15175

-- Define the graph structure
structure GameGraph where
  vertices : Finset Char
  edges : Finset (Char × Char)
  start : Char
  degree : Char → Nat

-- Define the game rules
structure GameRules where
  graph : GameGraph
  current_player : Nat
  used_edges : Finset (Char × Char)
  current_position : Char

-- Define a move
def valid_move (rules : GameRules) (next : Char) : Prop :=
  (rules.current_position, next) ∈ rules.graph.edges ∧
  (rules.current_position, next) ∉ rules.used_edges

-- Define the winning condition
def is_winning_position (rules : GameRules) : Prop :=
  ∀ next, ¬(valid_move rules next)

-- Theorem statement
theorem second_player_wins (g : GameGraph)
  (h1 : g.vertices = {'A', 'B', 'C', 'D', 'E', 'F'})
  (h2 : g.start = 'A')
  (h3 : g.degree 'A' = 4)
  (h4 : g.degree 'B' = 5)
  (h5 : g.degree 'C' = 5)
  (h6 : g.degree 'D' = 3)
  (h7 : g.degree 'E' = 3)
  (h8 : g.degree 'F' = 5)
  : ∃ (strategy : GameRules → Char),
    ∀ (rules : GameRules),
      rules.graph = g →
      rules.current_player = 2 →
      (valid_move rules (strategy rules) ∧
       is_winning_position
         { graph := rules.graph,
           current_player := 1,
           used_edges := insert (rules.current_position, strategy rules) rules.used_edges,
           current_position := strategy rules }) :=
sorry

end NUMINAMATH_CALUDE_second_player_wins_l151_15175


namespace NUMINAMATH_CALUDE_exponent_addition_l151_15111

theorem exponent_addition (a : ℝ) : a^3 + a^3 = 2*a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_addition_l151_15111


namespace NUMINAMATH_CALUDE_max_path_length_rectangular_prism_l151_15153

/-- Represents a rectangular prism with integer dimensions -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a path through all corners of a rectangular prism -/
def CornerPath (p : RectangularPrism) : Type :=
  List (Fin 2 × Fin 2 × Fin 2)

/-- Calculates the length of a given path in a rectangular prism -/
def pathLength (p : RectangularPrism) (path : CornerPath p) : ℝ :=
  sorry

/-- Checks if a path visits all corners exactly once and returns to start -/
def isValidPath (p : RectangularPrism) (path : CornerPath p) : Prop :=
  sorry

/-- The maximum possible path length for a given rectangular prism -/
def maxPathLength (p : RectangularPrism) : ℝ :=
  sorry

theorem max_path_length_rectangular_prism :
  ∃ (k : ℝ),
    maxPathLength ⟨3, 4, 5⟩ = 4 * Real.sqrt 50 + k ∧
    k > 0 ∧ k < 2 * Real.sqrt 50 :=
  sorry

end NUMINAMATH_CALUDE_max_path_length_rectangular_prism_l151_15153


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l151_15178

/-- Given a line segment with midpoint (3, 1) and one endpoint (7, -3), 
    prove that the other endpoint is (-1, 5). -/
theorem line_segment_endpoint (M x₁ y₁ x₂ y₂ : ℝ) : 
  M = 3 ∧ 
  x₁ = 7 ∧ 
  y₁ = -3 ∧ 
  M = (x₁ + x₂) / 2 ∧ 
  1 = (y₁ + y₂) / 2 → 
  x₂ = -1 ∧ y₂ = 5 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l151_15178


namespace NUMINAMATH_CALUDE_expression_change_l151_15179

theorem expression_change (x a : ℝ) (b : ℝ) (h : a > 0) : 
  let f := fun x => x^3 - b
  let δ := fun (ε : ℝ) => f (x + ε) - (b + a^2) - (f x - b)
  (δ a = 3*x^2*a + 3*x*a^2 + a^3 - a^2) ∧ 
  (δ (-a) = -3*x^2*a + 3*x*a^2 - a^3 - a^2) :=
by sorry

end NUMINAMATH_CALUDE_expression_change_l151_15179


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l151_15160

/-- Given a quadratic equation x^2 + mx - 2 = 0 where -1 is a root,
    prove that m = -1 and the other root is 2 -/
theorem quadratic_equation_roots (m : ℝ) : 
  ((-1 : ℝ)^2 + m*(-1) - 2 = 0) → 
  (m = -1 ∧ ∃ r : ℝ, r ≠ -1 ∧ r^2 + m*r - 2 = 0 ∧ r = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l151_15160


namespace NUMINAMATH_CALUDE_tangent_slope_implies_trig_ratio_triangle_perimeter_range_l151_15135

-- Problem 1
theorem tangent_slope_implies_trig_ratio 
  (f : ℝ → ℝ) (α : ℝ) 
  (h1 : ∀ x, f x = 2*x + 2*Real.sin x + Real.cos x) 
  (h2 : HasDerivAt f 2 α) : 
  (Real.sin (π - α) + Real.cos (-α)) / (2 * Real.cos (π/2 - α) + Real.cos (2*π - α)) = 3/5 := 
sorry

-- Problem 2
theorem triangle_perimeter_range 
  (A B C a b c : ℝ) 
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h2 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h3 : a = 1) 
  (h4 : a * Real.cos C + c/2 = b) : 
  ∃ l, l = a + b + c ∧ 2 < l ∧ l ≤ 3 := 
sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_trig_ratio_triangle_perimeter_range_l151_15135


namespace NUMINAMATH_CALUDE_prob_two_red_before_three_green_is_two_sevenths_l151_15114

/-- Represents the outcome of drawing chips from a hat -/
inductive DrawOutcome
| TwoRed
| ThreeGreen

/-- The probability of drawing 2 red chips before 3 green chips -/
def prob_two_red_before_three_green : ℚ :=
  2 / 7

/-- The number of red chips in the hat initially -/
def initial_red_chips : ℕ := 4

/-- The number of green chips in the hat initially -/
def initial_green_chips : ℕ := 3

/-- The total number of chips in the hat initially -/
def total_chips : ℕ := initial_red_chips + initial_green_chips

/-- Theorem stating that the probability of drawing 2 red chips before 3 green chips is 2/7 -/
theorem prob_two_red_before_three_green_is_two_sevenths :
  prob_two_red_before_three_green = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_before_three_green_is_two_sevenths_l151_15114


namespace NUMINAMATH_CALUDE_trig_identity_l151_15150

theorem trig_identity (a b c : ℝ) (θ : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (Real.sin θ)^6 / a + (Real.cos θ)^6 / b + (Real.sin θ)^2 * (Real.cos θ)^2 / c = 1 / (a + b + c) →
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 + ((Real.sin θ)^2 * (Real.cos θ)^2)^3 / c^5 = 
    (a + b + (a*b)^3/c^5) / (a + b + c)^6 :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l151_15150


namespace NUMINAMATH_CALUDE_salary_decrease_percentage_typist_salary_problem_l151_15117

theorem salary_decrease_percentage 
  (original_salary : ℝ) 
  (increase_percentage : ℝ) 
  (final_salary : ℝ) : ℝ :=
  let increased_salary := original_salary * (1 + increase_percentage / 100)
  let decrease_percentage := (increased_salary - final_salary) / increased_salary * 100
  decrease_percentage

theorem typist_salary_problem : 
  salary_decrease_percentage 2000 10 2090 = 5 := by
  sorry

end NUMINAMATH_CALUDE_salary_decrease_percentage_typist_salary_problem_l151_15117


namespace NUMINAMATH_CALUDE_shaltaev_boltaev_inequality_l151_15189

theorem shaltaev_boltaev_inequality (S B : ℝ) 
  (h1 : S > 0) (h2 : B > 0) 
  (h3 : 175 * S > 125 * B) (h4 : 175 * S < 126 * B) : 
  3 * S + B > S := by
sorry

end NUMINAMATH_CALUDE_shaltaev_boltaev_inequality_l151_15189


namespace NUMINAMATH_CALUDE_expression_evaluation_l151_15115

theorem expression_evaluation (a b x y : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a * x + y / b ≠ 0) :
  (a * x + y / b)⁻¹ * ((a * x)⁻¹ + (y / b)⁻¹) = (a * x * y)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l151_15115


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_rate_example_l151_15183

/-- Calculates the net rate of pay for a driver given specific conditions --/
theorem driver_net_pay_rate (hours : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (pay_per_mile : ℝ) 
  (fuel_cost : ℝ) (maintenance_threshold : ℝ) (maintenance_cost : ℝ) : ℝ :=
  let distance := hours * speed
  let fuel_used := distance / fuel_efficiency
  let earnings := distance * pay_per_mile
  let fuel_expense := fuel_used * fuel_cost
  let maintenance_expense := if distance > maintenance_threshold then maintenance_cost else 0
  let total_expense := fuel_expense + maintenance_expense
  let net_earnings := earnings - total_expense
  let net_rate := net_earnings / hours
  net_rate

/-- The driver's net rate of pay is approximately 21.67 dollars per hour --/
theorem driver_net_pay_rate_example : 
  ∃ ε > 0, |driver_net_pay_rate 3 50 25 0.60 2.50 100 10 - 21.67| < ε :=
sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_rate_example_l151_15183


namespace NUMINAMATH_CALUDE_intersection_equality_implies_m_value_l151_15198

theorem intersection_equality_implies_m_value (m : ℝ) : 
  ({3, 4, m^2 - 3*m - 1} ∩ {2*m, -3} : Set ℝ) = {-3} → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_m_value_l151_15198


namespace NUMINAMATH_CALUDE_round_75_36_bar_l151_15146

/-- Represents a number with a repeating decimal part -/
structure RepeatingDecimal where
  wholePart : ℕ
  nonRepeatingPart : ℕ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest hundredth -/
def roundToHundredth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The number 75.363636... -/
def number : RepeatingDecimal :=
  { wholePart := 75,
    nonRepeatingPart := 36,
    repeatingPart := 36 }

theorem round_75_36_bar : roundToHundredth number = 75.37 := by
  sorry

end NUMINAMATH_CALUDE_round_75_36_bar_l151_15146


namespace NUMINAMATH_CALUDE_ancient_tribe_leadership_choices_l151_15101

/-- The number of ways to choose the leadership of an ancient human tribe --/
theorem ancient_tribe_leadership_choices (n : ℕ) : n = 22 →
  (n * (Nat.choose (n - 1) 3) * 6 * (Nat.choose (n - 4) 3) * (Nat.choose (n - 7) 3) * (Nat.choose (n - 10) 3)) = 22308038400 := by
  sorry


end NUMINAMATH_CALUDE_ancient_tribe_leadership_choices_l151_15101


namespace NUMINAMATH_CALUDE_unique_tangent_length_l151_15121

theorem unique_tangent_length (m n t₁ : ℝ) : 
  (30 : ℝ) = m + n →
  t₁^2 = m * n →
  m ∈ Set.Ioo 0 30 →
  ∃ k : ℕ, m = 2 * k →
  ∃! t₁ : ℝ, t₁ > 0 ∧ t₁^2 = m * (30 - m) :=
sorry

end NUMINAMATH_CALUDE_unique_tangent_length_l151_15121


namespace NUMINAMATH_CALUDE_birds_on_fence_l151_15155

/-- The number of birds initially sitting on the fence -/
def initial_birds : ℕ := 4

/-- The initial number of storks -/
def initial_storks : ℕ := 3

/-- The number of additional storks that joined -/
def additional_storks : ℕ := 6

theorem birds_on_fence :
  initial_birds = 4 ∧
  initial_storks = 3 ∧
  additional_storks = 6 ∧
  initial_storks + additional_storks = initial_birds + 5 :=
by sorry

end NUMINAMATH_CALUDE_birds_on_fence_l151_15155


namespace NUMINAMATH_CALUDE_striped_cube_loop_probability_l151_15180

/-- Represents a cube with stripes on its faces -/
structure StripedCube where
  /-- Each face has a stripe from midpoint to midpoint of opposite edges -/
  faces : Fin 6 → Bool
  /-- For any two opposing faces, one stripe must be perpendicular to the other -/
  opposing_perpendicular : ∀ i : Fin 3, faces i ≠ faces (i + 3)

/-- Predicate to check if a given striped cube forms a valid loop -/
def forms_loop (cube : StripedCube) : Prop :=
  ∃ i : Fin 3, cube.faces i = cube.faces (i + 3) ∧
    (cube.faces ((i + 1) % 3) ≠ cube.faces ((i + 4) % 3)) ∧
    (cube.faces ((i + 2) % 3) ≠ cube.faces ((i + 5) % 3))

/-- The total number of valid striped cube configurations -/
def total_configurations : ℕ := 64

/-- The number of striped cube configurations that form a loop -/
def loop_configurations : ℕ := 6

/-- Theorem stating the probability of a striped cube forming a loop -/
theorem striped_cube_loop_probability :
  (loop_configurations : ℚ) / total_configurations = 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_striped_cube_loop_probability_l151_15180


namespace NUMINAMATH_CALUDE_balloon_radius_increase_l151_15152

/-- Proves that when a circular object's circumference increases from 24 inches to 30 inches, 
    its radius increases by 3/π inches. -/
theorem balloon_radius_increase (r₁ r₂ : ℝ) : 
  2 * π * r₁ = 24 → 2 * π * r₂ = 30 → r₂ - r₁ = 3 / π := by sorry

end NUMINAMATH_CALUDE_balloon_radius_increase_l151_15152


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l151_15149

theorem quadratic_form_sum (x : ℝ) : ∃ (a h k : ℝ),
  (6 * x^2 - 12 * x + 4 = a * (x - h)^2 + k) ∧ (a + h + k = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l151_15149


namespace NUMINAMATH_CALUDE_equation_solutions_l151_15182

theorem equation_solutions :
  (∀ x : ℝ, (1/2 * x^2 - 8 = 0) ↔ (x = 4 ∨ x = -4)) ∧
  (∀ x : ℝ, ((x - 5)^3 = -27) ↔ (x = 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l151_15182


namespace NUMINAMATH_CALUDE_power_sine_inequality_l151_15118

theorem power_sine_inequality (α : Real) (x₁ x₂ : Real) 
  (h1 : 0 < α ∧ α < π)
  (h2 : 0 < x₁)
  (h3 : x₁ < x₂) :
  (x₂ / x₁) ^ (Real.sin α) > 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sine_inequality_l151_15118


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l151_15174

/-- A parabola y = ax^2 + bx - 4 is tangent to the line y = 2x + 3 if and only if
    a = -(b-2)^2 / 28 and b ≠ 2 -/
theorem parabola_tangent_to_line (a b : ℝ) :
  (∃ x y : ℝ, y = a * x^2 + b * x - 4 ∧ y = 2 * x + 3 ∧
    ∀ x' : ℝ, x' ≠ x → a * x'^2 + b * x' - 4 ≠ 2 * x' + 3) ↔
  (a = -(b-2)^2 / 28 ∧ b ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l151_15174


namespace NUMINAMATH_CALUDE_problem_solution_l151_15171

def f (x : ℝ) := |x| - |2*x - 1|

def M := {x : ℝ | f x > -1}

theorem problem_solution :
  (M = {x : ℝ | 0 < x ∧ x < 2}) ∧
  (∀ a ∈ M,
    (0 < a ∧ a < 1 → a^2 - a + 1 < 1/a) ∧
    (a = 1 → a^2 - a + 1 = 1/a) ∧
    (1 < a ∧ a < 2 → a^2 - a + 1 > 1/a)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l151_15171


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l151_15197

theorem abs_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (abs (x₁ + 3) = 15 ∧ abs (x₂ + 3) = 15) ∧ 
  x₁ ≠ x₂ ∧
  abs (x₁ - x₂) = 30 := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l151_15197


namespace NUMINAMATH_CALUDE_basic_operation_time_scientific_notation_l151_15124

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  norm : 1 ≤ coefficient ∧ coefficient < 10

/-- The time taken for one basic operation in seconds -/
def basicOperationTime : ℝ := 0.000000001

/-- The scientific notation representation of the basic operation time -/
def basicOperationTimeScientific : ScientificNotation :=
  { coefficient := 1
  , exponent := -9
  , norm := by sorry }

/-- Theorem stating that the basic operation time is correctly represented in scientific notation -/
theorem basic_operation_time_scientific_notation :
  basicOperationTime = basicOperationTimeScientific.coefficient * (10 : ℝ) ^ basicOperationTimeScientific.exponent :=
by sorry

end NUMINAMATH_CALUDE_basic_operation_time_scientific_notation_l151_15124


namespace NUMINAMATH_CALUDE_f_equals_g_l151_15105

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem f_equals_g : f = g := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l151_15105


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_192_l151_15156

/-- The sum of all integer coefficients in the factorization of 216x^9 - 1000y^9 -/
def sum_of_coefficients (x y : ℚ) : ℤ :=
  let expression := 216 * x^9 - 1000 * y^9
  -- The actual computation of the sum is not implemented here
  192

/-- Theorem stating that the sum of all integer coefficients in the factorization of 216x^9 - 1000y^9 is 192 -/
theorem sum_of_coefficients_is_192 (x y : ℚ) : sum_of_coefficients x y = 192 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_192_l151_15156


namespace NUMINAMATH_CALUDE_comparison_of_radicals_and_fractions_l151_15176

theorem comparison_of_radicals_and_fractions : 
  (2 * Real.sqrt 7 < 4 * Real.sqrt 2) ∧ ((Real.sqrt 5 - 1) / 2 > 0.5) := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_radicals_and_fractions_l151_15176
