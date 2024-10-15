import Mathlib

namespace NUMINAMATH_CALUDE_purely_imaginary_implies_m_equals_one_l1980_198098

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number in question -/
def complex_number (m : ℝ) : ℂ :=
  ⟨m^2 - 3*m + 2, m^2 - 2*m⟩

/-- Theorem stating that if the complex number is purely imaginary, then m = 1 -/
theorem purely_imaginary_implies_m_equals_one :
  ∀ m : ℝ, is_purely_imaginary (complex_number m) → m = 1 := by
  sorry

#check purely_imaginary_implies_m_equals_one

end NUMINAMATH_CALUDE_purely_imaginary_implies_m_equals_one_l1980_198098


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1980_198070

/-- An arithmetic sequence with common difference 3 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 3

/-- The first, third, and fourth terms form a geometric sequence -/
def geometric_subsequence (a : ℕ → ℝ) : Prop :=
  (a 3)^2 = a 1 * a 4

/-- Main theorem: If a is an arithmetic sequence with common difference 3
    and its first, third, and fourth terms form a geometric sequence,
    then the second term equals -9 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
    (h1 : arithmetic_sequence a) (h2 : geometric_subsequence a) : 
  a 2 = -9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1980_198070


namespace NUMINAMATH_CALUDE_total_floor_area_l1980_198086

/-- The total floor area covered by square stone slabs -/
theorem total_floor_area (num_slabs : ℕ) (slab_length : ℝ) : 
  num_slabs = 30 → slab_length = 150 → 
  (num_slabs * (slab_length / 100)^2 : ℝ) = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_total_floor_area_l1980_198086


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1980_198057

theorem complex_equation_solution (z : ℂ) (h : z + Complex.abs z = 2 + 8 * I) : z = -15 + 8 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1980_198057


namespace NUMINAMATH_CALUDE_copper_ion_test_l1980_198007

theorem copper_ion_test (total_beakers : ℕ) (copper_beakers : ℕ) (total_drops : ℕ) (non_copper_tested : ℕ) :
  total_beakers = 22 →
  copper_beakers = 8 →
  total_drops = 45 →
  non_copper_tested = 7 →
  (copper_beakers + non_copper_tested) * 3 = total_drops :=
by sorry

end NUMINAMATH_CALUDE_copper_ion_test_l1980_198007


namespace NUMINAMATH_CALUDE_problem_statement_l1980_198043

theorem problem_statement (x : ℝ) :
  x = (Real.sqrt (6 + 2 * Real.sqrt 5) + Real.sqrt (6 - 2 * Real.sqrt 5)) / Real.sqrt 20 →
  (1 + x^5 - x^7)^(2012^(3^11)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1980_198043


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1980_198093

theorem quadratic_equation_solution :
  ∃! x : ℚ, x > 0 ∧ 3 * x^2 + 11 * x - 20 = 0 :=
by
  -- The unique positive solution is x = 4/3
  use 4/3
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1980_198093


namespace NUMINAMATH_CALUDE_twin_running_problem_l1980_198036

theorem twin_running_problem (x : ℝ) :
  (x ≥ 0) →  -- Ensure distance is non-negative
  (2 * x = 25) →  -- Final distance equation
  (x = 12.5) :=
by
  sorry

end NUMINAMATH_CALUDE_twin_running_problem_l1980_198036


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1980_198055

theorem triangle_perimeter (a b c : ℕ) (ha : a = 3) (hb : b = 8) (hc : Odd c) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  a + b + c = 18 ∨ a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1980_198055


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l1980_198012

theorem arithmetic_geometric_inequality (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (arith_prog : ∃ r : ℝ, b = a + r ∧ c = a + 2*r ∧ d = a + 3*r)
  (geom_prog : ∃ q : ℝ, e = a * q ∧ f = a * q^2 ∧ d = a * q^3) :
  b * c ≥ e * f := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l1980_198012


namespace NUMINAMATH_CALUDE_ramon_age_in_twenty_years_ramon_current_age_l1980_198039

/-- Ramon's current age -/
def ramon_age : ℕ := 26

/-- Loui's current age -/
def loui_age : ℕ := 23

/-- In twenty years, Ramon will be twice as old as Loui is today -/
theorem ramon_age_in_twenty_years (ramon_age loui_age : ℕ) :
  ramon_age + 20 = 2 * loui_age := by sorry

theorem ramon_current_age : ramon_age = 26 := by sorry

end NUMINAMATH_CALUDE_ramon_age_in_twenty_years_ramon_current_age_l1980_198039


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l1980_198051

/-- The focal length of a hyperbola with equation y²/4 - x² = 1 is 2√5 -/
theorem hyperbola_focal_length :
  let hyperbola := {(x, y) : ℝ × ℝ | y^2 / 4 - x^2 = 1}
  ∃ (f : ℝ), f = 2 * Real.sqrt 5 ∧ 
    ∀ (p q : ℝ × ℝ), p ∈ hyperbola → q ∈ hyperbola → 
      abs (dist p (0, f) - dist p (0, -f)) = 2 * abs (p.1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l1980_198051


namespace NUMINAMATH_CALUDE_center_on_line_common_chord_condition_external_tangent_length_l1980_198071

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C_k (k x y : ℝ) : Prop := (x - k)^2 + (y - Real.sqrt 3 * k)^2 = 4

-- Define the line y = √3x
def line_sqrt3 (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Theorem 1: The center of circle C_k always lies on the line y = √3x
theorem center_on_line (k : ℝ) : line_sqrt3 k (Real.sqrt 3 * k) := by sorry

-- Theorem 2: If the common chord length is √15/2, then k = ±1 or k = ±3/4
theorem common_chord_condition (k : ℝ) : 
  (∃ x y : ℝ, circle_O x y ∧ circle_C_k k x y ∧ 
   (x^2 + y^2 = 1 - (15/16))) → 
  (k = 1 ∨ k = -1 ∨ k = 3/4 ∨ k = -3/4) := by sorry

-- Theorem 3: When k = ±3/2, the length of the external common tangent is 2√2
theorem external_tangent_length : 
  ∀ k : ℝ, (k = 3/2 ∨ k = -3/2) → 
  (∃ x1 y1 x2 y2 : ℝ, 
    circle_O x1 y1 ∧ circle_C_k k x2 y2 ∧
    ((x2 - x1)^2 + (y2 - y1)^2 = 8)) := by sorry

end NUMINAMATH_CALUDE_center_on_line_common_chord_condition_external_tangent_length_l1980_198071


namespace NUMINAMATH_CALUDE_backyard_area_l1980_198000

/-- A rectangular backyard satisfying certain conditions -/
structure Backyard where
  length : ℝ
  width : ℝ
  length_condition : 25 * length = 1000
  perimeter_condition : 10 * (2 * (length + width)) = 1000

/-- The area of a backyard is 400 square meters -/
theorem backyard_area (b : Backyard) : b.length * b.width = 400 := by
  sorry


end NUMINAMATH_CALUDE_backyard_area_l1980_198000


namespace NUMINAMATH_CALUDE_evaluate_expression_l1980_198072

theorem evaluate_expression : (3 : ℝ)^4 - 4 * (3 : ℝ)^2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1980_198072


namespace NUMINAMATH_CALUDE_b_share_is_3315_l1980_198085

/-- Calculates the share of a partner in a partnership based on investments and known share. -/
def calculate_share (investment_a investment_b investment_c share_a : ℚ) : ℚ :=
  (share_a * investment_b) / investment_a

/-- Theorem stating that given the investments and a's share, b's share is 3315. -/
theorem b_share_is_3315 (investment_a investment_b investment_c share_a : ℚ) 
  (h1 : investment_a = 11000)
  (h2 : investment_b = 15000)
  (h3 : investment_c = 23000)
  (h4 : share_a = 2431) :
  calculate_share investment_a investment_b investment_c share_a = 3315 := by
sorry

#eval calculate_share 11000 15000 23000 2431

end NUMINAMATH_CALUDE_b_share_is_3315_l1980_198085


namespace NUMINAMATH_CALUDE_inequality_proof_l1980_198010

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  a * Real.sqrt b + b * Real.sqrt c + c * Real.sqrt a ≤ 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1980_198010


namespace NUMINAMATH_CALUDE_value_of_d_l1980_198078

theorem value_of_d (a b c d e : ℝ) 
  (h : 3 * (a^2 + b^2 + c^2) + 4 = 2*d + Real.sqrt (a + b + c - d + e)) 
  (he : e = 1) : 
  d = 7/4 := by
sorry

end NUMINAMATH_CALUDE_value_of_d_l1980_198078


namespace NUMINAMATH_CALUDE_joe_paint_usage_l1980_198068

/-- The amount of paint Joe used given the initial amount and usage fractions -/
def paint_used (initial : ℚ) (first_week_fraction : ℚ) (second_week_fraction : ℚ) : ℚ :=
  let first_week := initial * first_week_fraction
  let remaining := initial - first_week
  let second_week := remaining * second_week_fraction
  first_week + second_week

/-- Theorem stating that Joe used 168 gallons of paint -/
theorem joe_paint_usage :
  paint_used 360 (1/3) (1/5) = 168 := by
  sorry

end NUMINAMATH_CALUDE_joe_paint_usage_l1980_198068


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l1980_198001

theorem sum_of_roots_equation (x : ℝ) : 
  let f : ℝ → ℝ := λ x => (3*x + 4)*(x - 5) + (3*x + 4)*(x - 7)
  (∃ a b c : ℝ, ∀ x, f x = a*x^2 + b*x + c) →
  (∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ + r₂ = -2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l1980_198001


namespace NUMINAMATH_CALUDE_contributions_before_johns_l1980_198065

def average_before_johns (n : ℕ) : ℚ := 50

def johns_contribution : ℚ := 150

def new_average (n : ℕ) : ℚ := 75

def total_before_johns (n : ℕ) : ℚ := n * average_before_johns n

def total_after_johns (n : ℕ) : ℚ := total_before_johns n + johns_contribution

theorem contributions_before_johns :
  ∃ n : ℕ, 
    (new_average n = (3/2) * average_before_johns n) ∧
    (new_average n = 75) ∧
    (johns_contribution = 150) ∧
    (new_average n = total_after_johns n / (n + 1)) ∧
    (n = 3) :=
by sorry

end NUMINAMATH_CALUDE_contributions_before_johns_l1980_198065


namespace NUMINAMATH_CALUDE_starters_count_l1980_198073

/-- The number of ways to choose k elements from a set of n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 5 starters from a team of 12 players,
    including a set of twins, with at most one of the twins in the starting lineup -/
def chooseStarters : ℕ :=
  choose 10 5 + 2 * choose 10 4

theorem starters_count : chooseStarters = 672 := by sorry

end NUMINAMATH_CALUDE_starters_count_l1980_198073


namespace NUMINAMATH_CALUDE_smallest_quotient_by_18_l1980_198047

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_quotient_by_18 (U : ℕ) (hU : is_binary_number U) (hDiv : U % 18 = 0) :
  ∃ Y : ℕ, Y = U / 18 ∧ Y ≥ 61728395 ∧ (∀ Z : ℕ, (∃ V : ℕ, is_binary_number V ∧ V % 18 = 0 ∧ Z = V / 18) → Z ≥ Y) :=
sorry

end NUMINAMATH_CALUDE_smallest_quotient_by_18_l1980_198047


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_is_sqrt_6_over_36_l1980_198048

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (lateral_face_equilateral : Bool)

/-- A cube inscribed in a pyramid -/
structure InscribedCube :=
  (pyramid : Pyramid)
  (bottom_face_on_base : Bool)
  (top_face_edges_on_lateral_faces : Bool)

/-- The volume of an inscribed cube in a specific pyramid -/
noncomputable def inscribed_cube_volume (cube : InscribedCube) : ℝ :=
  sorry

/-- Theorem stating the volume of the inscribed cube -/
theorem inscribed_cube_volume_is_sqrt_6_over_36 
  (cube : InscribedCube) 
  (h1 : cube.pyramid.base_side = 1) 
  (h2 : cube.pyramid.lateral_face_equilateral = true)
  (h3 : cube.bottom_face_on_base = true)
  (h4 : cube.top_face_edges_on_lateral_faces = true) : 
  inscribed_cube_volume cube = Real.sqrt 6 / 36 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_is_sqrt_6_over_36_l1980_198048


namespace NUMINAMATH_CALUDE_triangle_base_length_l1980_198014

theorem triangle_base_length (height : ℝ) (area : ℝ) : 
  height = 8 → area = 24 → (1/2) * 6 * height = area :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l1980_198014


namespace NUMINAMATH_CALUDE_ace_ten_king_probability_l1980_198060

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The number of Tens in a standard deck -/
def num_tens : ℕ := 4

/-- The number of Kings in a standard deck -/
def num_kings : ℕ := 4

/-- The probability of drawing an Ace, then a 10, and then a King from a standard deck -/
def prob_ace_ten_king : ℚ := 8 / 16575

theorem ace_ten_king_probability :
  (num_aces : ℚ) / deck_size *
  num_tens / (deck_size - 1) *
  num_kings / (deck_size - 2) = prob_ace_ten_king := by
  sorry

end NUMINAMATH_CALUDE_ace_ten_king_probability_l1980_198060


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l1980_198099

theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of S_n
  (∀ n m, a n - a m = (n - m) * (a 2 - a 1)) →  -- Definition of arithmetic sequence
  a 8 / a 7 = 13 / 5 →  -- Given condition
  S 15 / S 13 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l1980_198099


namespace NUMINAMATH_CALUDE_spinner_probability_l1980_198004

theorem spinner_probability : 
  let spinner_sections : ℕ := 4
  let e_section : ℕ := 1
  let spins : ℕ := 2
  let prob_not_e_single : ℚ := (spinner_sections - e_section) / spinner_sections
  (prob_not_e_single ^ spins) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l1980_198004


namespace NUMINAMATH_CALUDE_base_12_remainder_div_7_l1980_198080

-- Define the base-12 number
def base_12_num : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3

-- Theorem statement
theorem base_12_remainder_div_7 : base_12_num % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_12_remainder_div_7_l1980_198080


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_6342_l1980_198013

/-- Calculates the cost of white washing a room with given dimensions, door, windows, and cost per square foot. -/
def whitewashing_cost (room_length room_width room_height : ℝ)
                      (door_width door_height : ℝ)
                      (window_width window_height : ℝ)
                      (num_windows : ℕ)
                      (cost_per_sqft : ℝ) : ℝ :=
  let total_wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_width * door_height
  let window_area := num_windows * (window_width * window_height)
  let net_area := total_wall_area - door_area - window_area
  net_area * cost_per_sqft

/-- Theorem stating that the cost of white washing the given room is 6342 Rs. -/
theorem whitewashing_cost_is_6342 :
  whitewashing_cost 25 15 12 6 3 4 3 3 7 = 6342 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_is_6342_l1980_198013


namespace NUMINAMATH_CALUDE_max_sum_2_by_1009_l1980_198077

/-- Represents a grid of squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the maximum sum of numbers in white squares for a given grid -/
def maxSumWhiteSquares (g : Grid) : ℕ :=
  if g.rows ≠ 2 ∨ g.cols ≠ 1009 then 0
  else
    let interiorContribution := (g.cols - 2) * 3
    let endpointContribution := 2 * 2
    interiorContribution + endpointContribution

/-- The theorem stating the maximum sum for a 2 by 1009 grid -/
theorem max_sum_2_by_1009 :
  ∀ g : Grid, g.rows = 2 ∧ g.cols = 1009 → maxSumWhiteSquares g = 3025 :=
by
  sorry

#eval maxSumWhiteSquares ⟨2, 1009⟩

end NUMINAMATH_CALUDE_max_sum_2_by_1009_l1980_198077


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1980_198063

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 2) : 
  ∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x^2 + y^2 + z^2 = 2 → 2*a*b + 3*b*c ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1980_198063


namespace NUMINAMATH_CALUDE_student_arrangement_count_l1980_198025

/-- The number of ways to select and arrange students with non-adjacent boys -/
def student_arrangements (num_boys num_girls select_boys select_girls : ℕ) : ℕ :=
  Nat.choose num_boys select_boys *
  Nat.choose num_girls select_girls *
  Nat.factorial select_girls *
  Nat.factorial (select_girls + 1)

/-- Theorem: The number of arrangements of 2 boys from 4 and 3 girls from 6,
    where the boys are not adjacent, is 8640 -/
theorem student_arrangement_count :
  student_arrangements 4 6 2 3 = 8640 := by
sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l1980_198025


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1980_198095

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 (a > 0) and distance between foci equal to 10,
    its eccentricity is 5/4 -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) : 
  (∃ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1) →
  (∃ c : ℝ, 2 * c = 10) →
  (∃ e : ℝ, e = 5/4 ∧ e = c/a) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1980_198095


namespace NUMINAMATH_CALUDE_fold_square_diagonal_l1980_198066

/-- Given a square ABCD with side length 8 cm, where corner C is folded to point E
    (located 1/3 of the way from A to D on AD), prove that the length of FD is 32/9 cm,
    where F is the point where the fold intersects CD. -/
theorem fold_square_diagonal (A B C D E F G : ℝ × ℝ) : 
  let side_length : ℝ := 8
  -- ABCD is a square
  (A.1 = 0 ∧ A.2 = 0) →
  (B.1 = side_length ∧ B.2 = 0) →
  (C.1 = side_length ∧ C.2 = side_length) →
  (D.1 = 0 ∧ D.2 = side_length) →
  -- E is one-third of the way along AD
  (E.1 = 0 ∧ E.2 = side_length / 3) →
  -- F is on CD
  (F.1 = side_length ∧ F.2 ≥ 0 ∧ F.2 ≤ side_length) →
  -- C coincides with E after folding
  (dist C E = dist C F) →
  -- FD length
  dist F D = 32 / 9 := by
sorry

end NUMINAMATH_CALUDE_fold_square_diagonal_l1980_198066


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_shift_l1980_198021

theorem quadratic_equation_solution_shift 
  (m h k : ℝ) 
  (hm : m ≠ 0) 
  (h1 : m * (2 - h)^2 - k = 0) 
  (h2 : m * (5 - h)^2 - k = 0) :
  m * (1 - h + 1)^2 = k ∧ m * (4 - h + 1)^2 = k := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_shift_l1980_198021


namespace NUMINAMATH_CALUDE_senior_mean_score_senior_mean_score_is_88_l1980_198017

/-- The mean score of seniors in a math competition --/
theorem senior_mean_score (total_students : ℕ) (overall_mean : ℝ) 
  (junior_ratio : ℝ) (senior_score_ratio : ℝ) : ℝ :=
  let senior_count := (total_students : ℝ) / (1 + junior_ratio)
  let junior_count := senior_count * junior_ratio
  let junior_mean := overall_mean * (total_students : ℝ) / (senior_count * senior_score_ratio + junior_count)
  junior_mean * senior_score_ratio

/-- The mean score of seniors is approximately 88 --/
theorem senior_mean_score_is_88 : 
  ∃ ε > 0, |senior_mean_score 150 80 1.2 1.2 - 88| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_senior_mean_score_senior_mean_score_is_88_l1980_198017


namespace NUMINAMATH_CALUDE_remainder_sum_l1980_198033

theorem remainder_sum (c d : ℤ) 
  (hc : c % 80 = 74) 
  (hd : d % 120 = 114) : 
  (c + d) % 40 = 28 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l1980_198033


namespace NUMINAMATH_CALUDE_triangle_longest_side_l1980_198082

theorem triangle_longest_side (x y : ℝ) :
  10 + (2 * y + 3) + (3 * x + 2) = 45 →
  (10 > 0) ∧ (2 * y + 3 > 0) ∧ (3 * x + 2 > 0) →
  max 10 (max (2 * y + 3) (3 * x + 2)) ≤ 32 :=
by sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l1980_198082


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1980_198088

-- Define variables
variable (a b x y : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 : 2*a - (4*a + 5*b) + 2*(3*a - 4*b) = 4*a - 13*b := by
  sorry

-- Theorem for the second expression
theorem simplify_expression_2 : 5*x^2 - 2*(3*y^2 - 5*x^2) + (-4*y^2 + 7*x*y) = 15*x^2 - 10*y^2 + 7*x*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1980_198088


namespace NUMINAMATH_CALUDE_matthews_cakes_equal_crackers_l1980_198031

/-- The number of friends Matthew gave crackers and cakes to -/
def num_friends : ℕ := 4

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 32

/-- The number of crackers each person ate -/
def crackers_per_person : ℕ := 8

/-- The number of cakes Matthew had initially -/
def initial_cakes : ℕ := initial_crackers

theorem matthews_cakes_equal_crackers :
  initial_cakes = initial_crackers :=
by sorry

end NUMINAMATH_CALUDE_matthews_cakes_equal_crackers_l1980_198031


namespace NUMINAMATH_CALUDE_equilateral_pyramid_cross_section_l1980_198089

/-- Represents a pyramid with an equilateral triangular base -/
structure EquilateralPyramid where
  /-- Side length of the base triangle -/
  base_side : ℝ
  /-- Height of the pyramid -/
  height : ℝ

/-- Represents a plane that intersects the pyramid -/
structure IntersectingPlane where
  /-- Angle between the plane and the base of the pyramid -/
  angle_with_base : ℝ

/-- Calculates the area of the cross-section of the pyramid -/
noncomputable def cross_section_area (p : EquilateralPyramid) (plane : IntersectingPlane) : ℝ :=
  sorry

theorem equilateral_pyramid_cross_section
  (p : EquilateralPyramid)
  (plane : IntersectingPlane) :
  p.base_side = 3 ∧
  p.height = Real.sqrt 3 ∧
  plane.angle_with_base = π / 3 →
  cross_section_area p plane = 11 * Real.sqrt 3 / 10 := by
    sorry

end NUMINAMATH_CALUDE_equilateral_pyramid_cross_section_l1980_198089


namespace NUMINAMATH_CALUDE_distance_origin_to_point_l1980_198074

theorem distance_origin_to_point :
  let x : ℝ := 20
  let y : ℝ := 21
  Real.sqrt (x^2 + y^2) = 29 :=
by sorry

end NUMINAMATH_CALUDE_distance_origin_to_point_l1980_198074


namespace NUMINAMATH_CALUDE_red_jellybeans_count_l1980_198061

/-- The probability of drawing 3 blue jellybeans in a row without replacement -/
def probability : ℚ := 10526315789473684 / 100000000000000000

/-- The number of blue jellybeans in the bag -/
def blue_jellybeans : ℕ := 10

/-- Calculates the probability of drawing 3 blue jellybeans in a row without replacement -/
def calculate_probability (red : ℕ) : ℚ :=
  (blue_jellybeans : ℚ) / (blue_jellybeans + red) *
  ((blue_jellybeans - 1) : ℚ) / (blue_jellybeans + red - 1) *
  ((blue_jellybeans - 2) : ℚ) / (blue_jellybeans + red - 2)

/-- Theorem stating that the number of red jellybeans is 10 -/
theorem red_jellybeans_count : ∃ (red : ℕ), calculate_probability red = probability ∧ red = 10 := by
  sorry

end NUMINAMATH_CALUDE_red_jellybeans_count_l1980_198061


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1980_198083

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  ((a = 4 ∧ b = 5) ∨ (a = 4 ∧ c = 5) ∨ (b = 4 ∧ c = 5)) →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c = 3 ∨ c = Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1980_198083


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1980_198018

theorem inequality_solution_set :
  {x : ℝ | 3 - 2*x > 7} = {x : ℝ | x < -2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1980_198018


namespace NUMINAMATH_CALUDE_cube_construction_count_l1980_198087

/-- The number of distinguishable ways to construct a cube from colored squares -/
def distinguishable_cube_constructions : ℕ := 1260

/-- The number of faces on a cube -/
def cube_faces : ℕ := 6

/-- The number of colored squares available -/
def colored_squares : ℕ := 8

/-- The number of rotational symmetries when one face is fixed -/
def rotational_symmetries : ℕ := 4

theorem cube_construction_count :
  distinguishable_cube_constructions = (colored_squares - 1).factorial / rotational_symmetries :=
sorry

end NUMINAMATH_CALUDE_cube_construction_count_l1980_198087


namespace NUMINAMATH_CALUDE_fib_sum_eq_49_287_l1980_198045

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of G_n / 7^n from n = 0 to infinity -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / 7^n

theorem fib_sum_eq_49_287 : fibSum = 49 / 287 := by sorry

end NUMINAMATH_CALUDE_fib_sum_eq_49_287_l1980_198045


namespace NUMINAMATH_CALUDE_equal_bills_at_120_minutes_l1980_198056

/-- The base rate for United Telephone service in dollars -/
def united_base_rate : ℝ := 6

/-- The per-minute charge for United Telephone in dollars -/
def united_per_minute : ℝ := 0.25

/-- The base rate for Atlantic Call service in dollars -/
def atlantic_base_rate : ℝ := 12

/-- The per-minute charge for Atlantic Call in dollars -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which the bills are equal -/
def equal_minutes : ℝ := 120

theorem equal_bills_at_120_minutes :
  united_base_rate + united_per_minute * equal_minutes =
  atlantic_base_rate + atlantic_per_minute * equal_minutes :=
by sorry

end NUMINAMATH_CALUDE_equal_bills_at_120_minutes_l1980_198056


namespace NUMINAMATH_CALUDE_pythagorean_triple_even_l1980_198034

theorem pythagorean_triple_even (x y z : ℤ) (h : x^2 + y^2 = z^2) : Even x ∨ Even y := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_even_l1980_198034


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1980_198029

/-- Given a line L1 with equation x - y - 2 = 0 and a point A (2, 6),
    prove that the line L2 with equation x + y - 8 = 0 passes through A
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point 
  (L1 : Set (ℝ × ℝ)) 
  (A : ℝ × ℝ) :
  let L2 := {(x, y) : ℝ × ℝ | x + y - 8 = 0}
  (∀ (x y : ℝ), (x, y) ∈ L1 ↔ x - y - 2 = 0) →
  A = (2, 6) →
  A ∈ L2 ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁, y₁) ∈ L1 → (x₂, y₂) ∈ L1 → x₁ ≠ x₂ →
    (x₁ - x₂) * (2 - 2) + (y₁ - y₂) * (6 - 6) = 0) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1980_198029


namespace NUMINAMATH_CALUDE_work_completed_together_l1980_198091

/-- The amount of work that can be completed by two workers in one day, given their individual work rates. -/
theorem work_completed_together 
  (days_A : ℝ) -- Number of days A takes to complete the work
  (days_B : ℝ) -- Number of days B takes to complete the work
  (h1 : days_A = 10) -- A can finish the work in 10 days
  (h2 : days_B = days_A / 2) -- B can do the same work in half the time taken by A
  : (1 / days_A + 1 / days_B) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_work_completed_together_l1980_198091


namespace NUMINAMATH_CALUDE_trey_bracelet_sales_l1980_198092

/-- The average number of bracelets Trey needs to sell each day -/
def average_bracelets_per_day (total_cost : ℕ) (num_days : ℕ) (bracelet_price : ℕ) : ℚ :=
  (total_cost : ℚ) / (num_days : ℚ)

/-- Theorem stating that Trey needs to sell 8 bracelets per day on average -/
theorem trey_bracelet_sales :
  average_bracelets_per_day 112 14 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_trey_bracelet_sales_l1980_198092


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_pythagorean_triplet_number_l1980_198005

/-- Given a three-digit number abc where a, b, and c are nonzero digits
    satisfying a^2 + b^2 = c^2, the largest possible prime factor of abc is 29. -/
theorem largest_prime_factor_of_pythagorean_triplet_number : ∃ (a b c : ℕ),
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (1 ≤ b ∧ b ≤ 9) ∧ 
  (1 ≤ c ∧ c ≤ 9) ∧ 
  a^2 + b^2 = c^2 ∧
  (∀ p : ℕ, p.Prime → p ∣ (100*a + 10*b + c) → p ≤ 29) ∧
  29 ∣ (100*a + 10*b + c) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_pythagorean_triplet_number_l1980_198005


namespace NUMINAMATH_CALUDE_shirt_sale_price_l1980_198008

theorem shirt_sale_price (original_price : ℝ) (initial_sale_price : ℝ) 
  (h1 : initial_sale_price > 0)
  (h2 : original_price > 0)
  (h3 : initial_sale_price * 0.8 = original_price * 0.64) :
  initial_sale_price / original_price = 0.8 := by
sorry

end NUMINAMATH_CALUDE_shirt_sale_price_l1980_198008


namespace NUMINAMATH_CALUDE_x0_value_l1980_198002

def f (x : ℝ) : ℝ := x^3 + x - 1

theorem x0_value (x₀ : ℝ) (h : (deriv f) x₀ = 4) : x₀ = 1 ∨ x₀ = -1 := by
  sorry

end NUMINAMATH_CALUDE_x0_value_l1980_198002


namespace NUMINAMATH_CALUDE_f_properties_l1980_198059

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + log x

theorem f_properties :
  (∀ m : ℝ, (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (exp 1) ∧ f 1 x₀ ≤ m) ↔ m ∈ Set.Ici (1/2)) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 1 → f a x < 2 * a * x) ↔ a ∈ Set.Icc (-1/2) (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1980_198059


namespace NUMINAMATH_CALUDE_prob_all_even_before_odd_prob_all_even_before_odd_proof_l1980_198019

/-- Represents an 8-sided die with numbers from 1 to 8 -/
inductive Die
| one | two | three | four | five | six | seven | eight

/-- Defines whether a number on the die is even or odd -/
def Die.isEven : Die → Bool
| Die.two => true
| Die.four => true
| Die.six => true
| Die.eight => true
| _ => false

/-- The probability of rolling an even number -/
def probEven : ℚ := 1/2

/-- The probability of rolling an odd number -/
def probOdd : ℚ := 1/2

/-- The set of even numbers on the die -/
def evenNumbers : Set Die := {Die.two, Die.four, Die.six, Die.eight}

/-- Theorem: The probability of rolling every even number at least once
    before rolling any odd number on an 8-sided die is 1/384 -/
theorem prob_all_even_before_odd : ℚ :=
  1/384

/-- Proof of the theorem -/
theorem prob_all_even_before_odd_proof :
  prob_all_even_before_odd = 1/384 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_even_before_odd_prob_all_even_before_odd_proof_l1980_198019


namespace NUMINAMATH_CALUDE_probability_theorem_l1980_198016

def family_A_size : ℕ := 5
def family_B_size : ℕ := 3
def total_girls : ℕ := 5
def total_boys : ℕ := 3

def probability_at_least_one_family_all_girls : ℚ :=
  11 / 56

theorem probability_theorem :
  let total_children := family_A_size + family_B_size
  probability_at_least_one_family_all_girls = 11 / 56 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l1980_198016


namespace NUMINAMATH_CALUDE_cupcake_price_is_two_l1980_198026

/-- Calculates the price per cupcake given the number of trays, cupcakes per tray,
    fraction of cupcakes sold, and total earnings. -/
def price_per_cupcake (num_trays : ℕ) (cupcakes_per_tray : ℕ) 
                      (fraction_sold : ℚ) (total_earnings : ℚ) : ℚ :=
  total_earnings / (fraction_sold * (num_trays * cupcakes_per_tray))

/-- Proves that the price per cupcake is $2 given the specific conditions. -/
theorem cupcake_price_is_two :
  price_per_cupcake 4 20 (3/5) 96 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_price_is_two_l1980_198026


namespace NUMINAMATH_CALUDE_total_miles_driven_l1980_198024

/-- The total miles driven by Darius and Julia -/
def total_miles (darius_miles julia_miles : ℕ) : ℕ :=
  darius_miles + julia_miles

/-- Theorem stating that the total miles driven by Darius and Julia is 1677 -/
theorem total_miles_driven :
  total_miles 679 998 = 1677 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_driven_l1980_198024


namespace NUMINAMATH_CALUDE_visit_probability_l1980_198022

/-- The probability of Jen visiting either Chile or Madagascar, but not both -/
theorem visit_probability (p_chile p_madagascar : ℝ) 
  (h_chile : p_chile = 0.30)
  (h_madagascar : p_madagascar = 0.50) : 
  p_chile + p_madagascar - p_chile * p_madagascar = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_visit_probability_l1980_198022


namespace NUMINAMATH_CALUDE_tan_theta_value_l1980_198032

theorem tan_theta_value (θ : ℝ) (z : ℂ) : 
  z = Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5) → 
  z.re = 0 → 
  z.im ≠ 0 → 
  Real.tan θ = -3/4 :=
sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1980_198032


namespace NUMINAMATH_CALUDE_green_chips_count_l1980_198067

theorem green_chips_count (total : ℕ) (blue white green : ℕ) : 
  blue = 3 →
  blue = total / 10 →
  white = total / 2 →
  green = total - blue - white →
  green = 12 := by
  sorry

end NUMINAMATH_CALUDE_green_chips_count_l1980_198067


namespace NUMINAMATH_CALUDE_oranges_picked_total_l1980_198037

/-- The number of oranges Mary picked -/
def mary_oranges : ℕ := 14

/-- The number of oranges Jason picked -/
def jason_oranges : ℕ := 41

/-- The total number of oranges picked -/
def total_oranges : ℕ := mary_oranges + jason_oranges

theorem oranges_picked_total :
  total_oranges = 55 := by sorry

end NUMINAMATH_CALUDE_oranges_picked_total_l1980_198037


namespace NUMINAMATH_CALUDE_prime_composite_inequality_l1980_198094

theorem prime_composite_inequality (n : ℕ) : 
  (Nat.Prime (2 * n - 1) → 
    ∀ (a : Fin n → ℕ), Function.Injective a → 
      ∃ i j : Fin n, (a i + a j : ℚ) / (Nat.gcd (a i) (a j)) ≥ 2 * n - 1) ∧
  (¬Nat.Prime (2 * n - 1) → 
    ∃ (a : Fin n → ℕ), Function.Injective a ∧
      ∀ i j : Fin n, (a i + a j : ℚ) / (Nat.gcd (a i) (a j)) < 2 * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_composite_inequality_l1980_198094


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1980_198011

theorem necessary_but_not_sufficient 
  (a b : ℝ) : 
  (((b + 2) / (a + 2) > b / a) ↔ (a > b ∧ b > 0)) → False :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1980_198011


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l1980_198006

-- Part 1
theorem factorization_1 (a : ℝ) : (a^2 - 4*a + 4) - 4*(a - 2) + 4 = (a - 4)^2 := by
  sorry

-- Part 2
theorem factorization_2 (x y : ℝ) : 16*x^4 - 81*y^4 = (4*x^2 + 9*y^2)*(2*x + 3*y)*(2*x - 3*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l1980_198006


namespace NUMINAMATH_CALUDE_circle_contains_origin_l1980_198084

theorem circle_contains_origin
  (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ a b c : ℝ)
  (h₁ : x₁ > 0) (h₂ : y₁ > 0)
  (h₃ : x₂ < 0) (h₄ : y₂ > 0)
  (h₅ : x₃ < 0) (h₆ : y₃ < 0)
  (h₇ : x₄ > 0) (h₈ : y₄ < 0)
  (h₉ : (x₁ - a)^2 + (y₁ - b)^2 ≤ c^2)
  (h₁₀ : (x₂ - a)^2 + (y₂ - b)^2 ≤ c^2)
  (h₁₁ : (x₃ - a)^2 + (y₃ - b)^2 ≤ c^2)
  (h₁₂ : (x₄ - a)^2 + (y₄ - b)^2 ≤ c^2)
  (h₁₃ : c > 0) :
  a^2 + b^2 < c^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_contains_origin_l1980_198084


namespace NUMINAMATH_CALUDE_min_distance_sum_l1980_198050

theorem min_distance_sum (x : ℝ) : 
  Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 1)^2) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1980_198050


namespace NUMINAMATH_CALUDE_correlation_coefficient_relationship_l1980_198079

/-- The correlation coefficient type -/
def CorrelationCoefficient := { r : ℝ // -1 ≤ r ∧ r ≤ 1 }

/-- The degree of correlation between two variables -/
noncomputable def degreeOfCorrelation (r : CorrelationCoefficient) : ℝ := sorry

/-- Theorem stating the relationship between |r| and the degree of correlation -/
theorem correlation_coefficient_relationship (r1 r2 : CorrelationCoefficient) :
  (|r1.val| < |r2.val| ∧ |r2.val| ≤ 1) → degreeOfCorrelation r1 < degreeOfCorrelation r2 := by sorry

end NUMINAMATH_CALUDE_correlation_coefficient_relationship_l1980_198079


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1980_198075

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop := x^2 - 2*x + a < 0

-- Define the solution set
def solution_set (t : ℝ) (x : ℝ) : Prop := -2 < x ∧ x < t

-- Define the second inequality
def second_inequality (c a : ℝ) (x : ℝ) : Prop := (c+a)*x^2 + 2*(c+a)*x - 1 < 0

theorem quadratic_inequality_solution :
  ∃ (a t : ℝ),
    (∀ x, quadratic_inequality a x ↔ solution_set t x) ∧
    a = -8 ∧
    t = 4 ∧
    ∀ c, (∀ x, second_inequality c a x) ↔ (7 < c ∧ c ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1980_198075


namespace NUMINAMATH_CALUDE_angle_A_is_pi_third_max_area_is_sqrt_three_max_area_achieved_l1980_198041

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a = 2 ∧ (2 + t.b) * (Real.sin t.A - Real.sin t.B) = (t.c - t.b) * Real.sin t.C

-- Theorem 1: Angle A is π/3
theorem angle_A_is_pi_third (t : Triangle) (h : satisfiesConditions t) : t.A = π / 3 := by
  sorry

-- Theorem 2: Maximum area is √3
theorem max_area_is_sqrt_three (t : Triangle) (h : satisfiesConditions t) : 
  (1/2 * t.b * t.c * Real.sin t.A) ≤ Real.sqrt 3 := by
  sorry

-- Theorem 2 (continued): The maximum area is achieved
theorem max_area_achieved (t : Triangle) : 
  ∃ (t : Triangle), satisfiesConditions t ∧ (1/2 * t.b * t.c * Real.sin t.A) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_third_max_area_is_sqrt_three_max_area_achieved_l1980_198041


namespace NUMINAMATH_CALUDE_max_intersections_six_paths_l1980_198020

/-- The number of intersection points for a given number of paths -/
def intersection_points (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: With 6 paths, where each path intersects with every other path
    exactly once, the maximum number of intersection points is 15 -/
theorem max_intersections_six_paths :
  intersection_points 6 = 15 := by
  sorry

#eval intersection_points 6  -- This will output 15

end NUMINAMATH_CALUDE_max_intersections_six_paths_l1980_198020


namespace NUMINAMATH_CALUDE_line_intersections_l1980_198054

theorem line_intersections : 
  let line1 : ℝ → ℝ := λ x => 5 * x - 20
  let line2 : ℝ → ℝ := λ x => 190 - 3 * x
  let line3 : ℝ → ℝ := λ x => 2 * x + 15
  ∃ (x1 x2 : ℝ), 
    (line1 x1 = line2 x1 ∧ x1 = 105 / 4) ∧
    (line1 x2 = line3 x2 ∧ x2 = 35 / 3) := by
  sorry

end NUMINAMATH_CALUDE_line_intersections_l1980_198054


namespace NUMINAMATH_CALUDE_cuboid_to_cube_surface_area_l1980_198023

/-- Given a cuboid with a square base, if reducing its height by 4 cm results in a cube
    and decreases its volume by 64 cubic centimeters, then the surface area of the
    resulting cube is 96 square centimeters. -/
theorem cuboid_to_cube_surface_area (l w h : ℝ) : 
  l = w → -- The base is square
  (l * w * h) - (l * w * (h - 4)) = 64 → -- Volume decrease
  l * w * 4 = 64 → -- Volume decrease equals base area times height reduction
  6 * (l * l) = 96 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_to_cube_surface_area_l1980_198023


namespace NUMINAMATH_CALUDE_min_perimeter_of_triangle_l1980_198049

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- A point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

/-- The center of the ellipse -/
def center : ℝ × ℝ := (0, 0)

/-- A line passing through the center of the ellipse -/
structure LineThroughCenter where
  slope : ℝ

/-- Intersection points of the line with the ellipse -/
def intersectionPoints (l : LineThroughCenter) : PointOnEllipse × PointOnEllipse := sorry

/-- One of the foci of the ellipse -/
def focus : ℝ × ℝ := (3, 0)

/-- The perimeter of the triangle formed by two points on the ellipse and the focus -/
def trianglePerimeter (p q : PointOnEllipse) : ℝ := sorry

/-- The statement to be proved -/
theorem min_perimeter_of_triangle : 
  ∀ l : LineThroughCenter, 
  let (p, q) := intersectionPoints l
  18 ≤ trianglePerimeter p q ∧ 
  ∃ l₀ : LineThroughCenter, trianglePerimeter (intersectionPoints l₀).1 (intersectionPoints l₀).2 = 18 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_of_triangle_l1980_198049


namespace NUMINAMATH_CALUDE_cross_shape_surface_area_l1980_198009

/-- Represents a 3D shape made of unit cubes -/
structure CubeShape where
  num_cubes : ℕ
  exposed_faces : ℕ

/-- The cross-like shape made of 5 unit cubes -/
def cross_shape : CubeShape :=
  { num_cubes := 5,
    exposed_faces := 22 }

/-- Theorem stating that the surface area of the cross-shape is 22 square units -/
theorem cross_shape_surface_area :
  cross_shape.exposed_faces = 22 := by
  sorry

end NUMINAMATH_CALUDE_cross_shape_surface_area_l1980_198009


namespace NUMINAMATH_CALUDE_a_range_l1980_198090

/-- Given a > 0, if the function y = a^x is not monotonically increasing on ℝ
    or the inequality ax^2 - ax + 1 > 0 does not hold for ∀x ∈ ℝ,
    and at least one of these conditions is true,
    then a ∈ (0,1] ∪ [4,+∞) -/
theorem a_range (a : ℝ) (h_a_pos : a > 0) : 
  (¬∀ x y : ℝ, x < y → a^x < a^y) ∨ 
  (¬∀ x : ℝ, a*x^2 - a*x + 1 > 0) ∧ 
  ((∀ x y : ℝ, x < y → a^x < a^y) ∨ 
   (∀ x : ℝ, a*x^2 - a*x + 1 > 0)) → 
  a ∈ Set.Ioc 0 1 ∪ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_a_range_l1980_198090


namespace NUMINAMATH_CALUDE_partition_theorem_l1980_198096

def is_valid_partition (n : ℕ) : Prop :=
  ∃ (partition : List (Fin n × Fin n × Fin n)),
    (∀ (i j : Fin n), i ≠ j → (∃ (t : Fin n × Fin n × Fin n), t ∈ partition ∧ (i = t.1 ∨ i = t.2.1 ∨ i = t.2.2)) →
                              (∃ (t : Fin n × Fin n × Fin n), t ∈ partition ∧ (j = t.1 ∨ j = t.2.1 ∨ j = t.2.2)) →
                              (∀ (t : Fin n × Fin n × Fin n), t ∈ partition → (i = t.1 ∨ i = t.2.1 ∨ i = t.2.2) →
                                                                              (j ≠ t.1 ∧ j ≠ t.2.1 ∧ j ≠ t.2.2))) ∧
    (∀ (t : Fin n × Fin n × Fin n), t ∈ partition → t.1.val + t.2.1.val = t.2.2.val ∨
                                                    t.1.val + t.2.2.val = t.2.1.val ∨
                                                    t.2.1.val + t.2.2.val = t.1.val)

theorem partition_theorem (n : ℕ) (h : n ∈ Finset.range 10 ∪ {3900}) : 
  is_valid_partition n ↔ n = 3900 ∨ n = 3903 :=
sorry

end NUMINAMATH_CALUDE_partition_theorem_l1980_198096


namespace NUMINAMATH_CALUDE_range_of_3x_plus_y_l1980_198076

theorem range_of_3x_plus_y (x y : ℝ) :
  3 * x^2 + y^2 ≤ 1 →
  ∃ (max min : ℝ), max = 2 ∧ min = -2 ∧
    (3 * x + y ≤ max ∧ 3 * x + y ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_range_of_3x_plus_y_l1980_198076


namespace NUMINAMATH_CALUDE_calculator_purchase_theorem_l1980_198040

/-- Represents the unit price of a type A calculator -/
def price_A : ℝ := 110

/-- Represents the unit price of a type B calculator -/
def price_B : ℝ := 120

/-- Represents the total number of calculators to be purchased -/
def total_calculators : ℕ := 100

/-- Theorem stating the properties of calculator prices and minimum purchase cost -/
theorem calculator_purchase_theorem :
  (price_B = price_A + 10) ∧
  (550 / price_A = 600 / price_B) ∧
  (∀ a b : ℕ, a + b = total_calculators → b ≤ 3 * a →
    price_A * a + price_B * b ≥ 11000) :=
by sorry

end NUMINAMATH_CALUDE_calculator_purchase_theorem_l1980_198040


namespace NUMINAMATH_CALUDE_sue_necklace_beads_l1980_198042

def necklace_beads (purple blue green red : ℕ) : Prop :=
  (blue = 2 * purple) ∧
  (green = blue + 11) ∧
  (red = green / 2) ∧
  (purple + blue + green + red = 58)

theorem sue_necklace_beads :
  ∃ (purple blue green red : ℕ),
    purple = 7 ∧
    necklace_beads purple blue green red :=
by sorry

end NUMINAMATH_CALUDE_sue_necklace_beads_l1980_198042


namespace NUMINAMATH_CALUDE_min_sin6_2cos6_l1980_198097

theorem min_sin6_2cos6 :
  ∀ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_min_sin6_2cos6_l1980_198097


namespace NUMINAMATH_CALUDE_salon_cost_calculation_l1980_198035

def salon_total_cost (manicure_cost pedicure_cost hair_treatment_cost : ℝ)
                     (manicure_tax_rate pedicure_tax_rate hair_treatment_tax_rate : ℝ)
                     (manicure_tip_rate pedicure_tip_rate hair_treatment_tip_rate : ℝ) : ℝ :=
  let manicure_total := manicure_cost * (1 + manicure_tax_rate + manicure_tip_rate)
  let pedicure_total := pedicure_cost * (1 + pedicure_tax_rate + pedicure_tip_rate)
  let hair_treatment_total := hair_treatment_cost * (1 + hair_treatment_tax_rate + hair_treatment_tip_rate)
  manicure_total + pedicure_total + hair_treatment_total

theorem salon_cost_calculation :
  salon_total_cost 30 40 50 0.05 0.07 0.09 0.25 0.20 0.15 = 151.80 := by
  sorry

end NUMINAMATH_CALUDE_salon_cost_calculation_l1980_198035


namespace NUMINAMATH_CALUDE_equation_solution_l1980_198015

theorem equation_solution : ∃ x : ℝ, (((1 + x) / (2 - x)) - 1 = 1 / (x - 2)) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1980_198015


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1980_198044

noncomputable def solution_sum : ℝ → Prop :=
  fun x ↦ (x^2 - 6*x - 3 = 0) ∧ (x ≠ 1) ∧ (x ≠ -1)

theorem sum_of_solutions :
  ∃ (a b : ℝ), solution_sum a ∧ solution_sum b ∧ a + b = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1980_198044


namespace NUMINAMATH_CALUDE_johns_number_l1980_198038

theorem johns_number : ∃! n : ℕ, 1000 < n ∧ n < 3000 ∧ 200 ∣ n ∧ 45 ∣ n ∧ n = 1800 := by
  sorry

end NUMINAMATH_CALUDE_johns_number_l1980_198038


namespace NUMINAMATH_CALUDE_fraction_problem_l1980_198046

theorem fraction_problem : ∃ x : ℚ, (0.60 * 40 : ℚ) = x * 25 + 4 :=
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1980_198046


namespace NUMINAMATH_CALUDE_joes_total_weight_l1980_198003

/-- Proves that the total weight of Joe's two lifts is 1800 pounds given the conditions -/
theorem joes_total_weight (first_lift second_lift : ℕ) : 
  first_lift = 700 ∧ 
  2 * first_lift = second_lift + 300 → 
  first_lift + second_lift = 1800 := by
sorry

end NUMINAMATH_CALUDE_joes_total_weight_l1980_198003


namespace NUMINAMATH_CALUDE_inequality_properties_l1980_198053

theorem inequality_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (-1 / b < -1 / a) ∧ (a^2 * b > a * b^2) ∧ (a / b > b / a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l1980_198053


namespace NUMINAMATH_CALUDE_certain_number_exists_l1980_198069

theorem certain_number_exists : ∃ x : ℝ, 5 * 1.25 * x^(1/4) * 60^(3/4) = 300 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l1980_198069


namespace NUMINAMATH_CALUDE_range_of_f_on_interval_solution_sets_f_positive_l1980_198081

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1) * x + a

-- Part 1: Range of f(x) when a = 3 on [-1, 3]
theorem range_of_f_on_interval :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3, -1 ≤ f 3 x ∧ f 3 x ≤ 8 :=
sorry

-- Part 2: Solution sets for f(x) > 0
theorem solution_sets_f_positive (a : ℝ) :
  (∀ x, f a x > 0 ↔ 
    (a > 1 ∧ (x < 1 ∨ x > a)) ∨
    (a < 1 ∧ (x < a ∨ x > 1)) ∨
    (a = 1 ∧ x ≠ 1)) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_solution_sets_f_positive_l1980_198081


namespace NUMINAMATH_CALUDE_marker_problem_l1980_198028

theorem marker_problem :
  ∃ (n : ℕ) (p : ℝ), 
    p > 0 ∧
    3.51 = p * n ∧
    4.25 = p * (n + 4) ∧
    n > 0 := by
  sorry

end NUMINAMATH_CALUDE_marker_problem_l1980_198028


namespace NUMINAMATH_CALUDE_digits_after_decimal_point_of_fraction_l1980_198030

/-- The number of digits to the right of the decimal point when 5^8 / (10^6 * 16) is expressed as a decimal is 3. -/
theorem digits_after_decimal_point_of_fraction : ∃ (n : ℕ) (d : ℕ+), 
  5^8 / (10^6 * 16) = n / d ∧ 
  (∃ (k : ℕ), 10^3 * (n / d) = k ∧ 10^2 * (n / d) < 1) :=
by sorry

end NUMINAMATH_CALUDE_digits_after_decimal_point_of_fraction_l1980_198030


namespace NUMINAMATH_CALUDE_not_special_2013_l1980_198062

/-- A year is special if there exists a month and day such that their product
    equals the last two digits of the year. -/
def is_special_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), 
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

/-- The last two digits of 2013. -/
def last_two_digits_2013 : ℕ := 13

/-- Theorem stating that 2013 is not a special year. -/
theorem not_special_2013 : ¬(is_special_year 2013) := by
  sorry

end NUMINAMATH_CALUDE_not_special_2013_l1980_198062


namespace NUMINAMATH_CALUDE_intersection_in_sphere_l1980_198052

/-- Given three unit cylinders with pairwise perpendicular axes, 
    their intersection is contained in a sphere of radius √(3/2) --/
theorem intersection_in_sphere (a b c d e f : ℝ) :
  ∀ x y z : ℝ, 
  (x - a)^2 + (y - b)^2 ≤ 1 →
  (y - c)^2 + (z - d)^2 ≤ 1 →
  (z - e)^2 + (x - f)^2 ≤ 1 →
  ∃ center_x center_y center_z : ℝ, 
    (x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2 ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_in_sphere_l1980_198052


namespace NUMINAMATH_CALUDE_dave_spent_on_mom_lunch_l1980_198027

def derek_initial : ℕ := 40
def derek_lunch1 : ℕ := 14
def derek_dad_lunch : ℕ := 11
def derek_lunch2 : ℕ := 5
def dave_initial : ℕ := 50
def difference_left : ℕ := 33

theorem dave_spent_on_mom_lunch :
  dave_initial - (derek_initial - derek_lunch1 - derek_dad_lunch - derek_lunch2 + difference_left) = 7 := by
  sorry

end NUMINAMATH_CALUDE_dave_spent_on_mom_lunch_l1980_198027


namespace NUMINAMATH_CALUDE_parabola_shift_l1980_198064

def original_function (x : ℝ) : ℝ := (x + 1)^2 + 3

def shift_right (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f (x - shift)

def shift_down (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f x - shift

def final_function (x : ℝ) : ℝ := (x - 1)^2 + 2

theorem parabola_shift :
  (shift_down (shift_right original_function 2) 1) = final_function := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l1980_198064


namespace NUMINAMATH_CALUDE_crayons_per_box_l1980_198058

theorem crayons_per_box (total_boxes : ℕ) (crayons_to_mae : ℕ) (extra_crayons_to_lea : ℕ) (crayons_left : ℕ) :
  total_boxes = 4 →
  crayons_to_mae = 5 →
  extra_crayons_to_lea = 7 →
  crayons_left = 15 →
  ∃ (crayons_per_box : ℕ),
    crayons_per_box * total_boxes = crayons_to_mae + (crayons_to_mae + extra_crayons_to_lea) + crayons_left ∧
    crayons_per_box = 8 :=
by sorry

end NUMINAMATH_CALUDE_crayons_per_box_l1980_198058
