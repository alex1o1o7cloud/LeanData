import Mathlib

namespace NUMINAMATH_CALUDE_sign_of_product_l2801_280130

theorem sign_of_product (h1 : 0 < 1 ∧ 1 < Real.pi / 2) 
  (h2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → Real.sin x < Real.sin y)
  (h3 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → Real.cos y < Real.cos x) :
  (Real.cos (Real.cos 1) - Real.cos 1) * (Real.sin (Real.sin 1) - Real.sin 1) < 0 := by
  sorry

end NUMINAMATH_CALUDE_sign_of_product_l2801_280130


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2801_280133

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 5)*x - k + 9 > 1) ↔ k > -1 ∧ k < 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2801_280133


namespace NUMINAMATH_CALUDE_ivan_max_bars_ivan_min_bars_ivan_exact_bars_l2801_280123

/-- Represents the state of the game -/
structure GameState where
  ivan_bars : ℕ
  chest_bars : ℕ

/-- Represents a player's move -/
structure Move where
  bars : ℕ

/-- Defines a valid move in the game -/
def valid_move (state : GameState) (move : Move) : Prop :=
  move.bars > 0 ∧ move.bars ≤ state.chest_bars

/-- Defines the game's rules and outcome -/
def game_outcome (initial_bars : ℕ) : ℕ :=
  sorry

/-- Theorem stating that Ivan can always take at most 13 bars -/
theorem ivan_max_bars (n : ℕ) (h : n ≥ 13) : 
  game_outcome n ≤ 13 :=
sorry

/-- Theorem stating that Ivan can always take at least 13 bars -/
theorem ivan_min_bars (n : ℕ) (h : n ≥ 13) : 
  game_outcome n ≥ 13 :=
sorry

/-- Main theorem proving that Ivan can always take exactly 13 bars -/
theorem ivan_exact_bars (n : ℕ) (h : n ≥ 13) : 
  game_outcome n = 13 :=
sorry

end NUMINAMATH_CALUDE_ivan_max_bars_ivan_min_bars_ivan_exact_bars_l2801_280123


namespace NUMINAMATH_CALUDE_jim_age_proof_l2801_280113

/-- Calculates Jim's age X years from now -/
def jim_future_age (x : ℕ) : ℕ :=
  let tom_age_5_years_ago : ℕ := 32
  let years_since_tom_32 : ℕ := 5
  let years_to_past_reference : ℕ := 7
  let jim_age_difference : ℕ := 5
  27 + x

/-- Proves that Jim's age X years from now is (27 + X) -/
theorem jim_age_proof (x : ℕ) :
  jim_future_age x = 27 + x := by
  sorry

end NUMINAMATH_CALUDE_jim_age_proof_l2801_280113


namespace NUMINAMATH_CALUDE_D_largest_l2801_280109

def D : ℚ := 3006 / 3005 + 3006 / 3007
def E : ℚ := 3006 / 3007 + 3008 / 3007
def F : ℚ := 3007 / 3006 + 3007 / 3008

theorem D_largest : D > E ∧ D > F := by
  sorry

end NUMINAMATH_CALUDE_D_largest_l2801_280109


namespace NUMINAMATH_CALUDE_fishing_theorem_l2801_280188

def fishing_problem (caleb_catch dad_catch : ℕ) : Prop :=
  caleb_catch = 2 ∧ 
  dad_catch = 3 * caleb_catch ∧ 
  dad_catch - caleb_catch = 4

theorem fishing_theorem : ∃ caleb_catch dad_catch, fishing_problem caleb_catch dad_catch :=
  sorry

end NUMINAMATH_CALUDE_fishing_theorem_l2801_280188


namespace NUMINAMATH_CALUDE_vector_triangle_l2801_280185

/-- Given a triangle ABC, a point D on BC such that BD = 3DC, and E the midpoint of AC,
    prove that ED = 1/4 AB + 1/4 AC -/
theorem vector_triangle (A B C D E : ℝ × ℝ) : 
  (∃ (t : ℝ), D = B + t • (C - B) ∧ t = 3/4) →  -- D is on BC with BD = 3DC
  E = A + (1/2 : ℝ) • (C - A) →                 -- E is midpoint of AC
  E - D = (1/4 : ℝ) • (B - A) + (1/4 : ℝ) • (C - A) := by
sorry

end NUMINAMATH_CALUDE_vector_triangle_l2801_280185


namespace NUMINAMATH_CALUDE_problem_statement_l2801_280121

theorem problem_statement (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x*y + x + y = 7) : 
  x^2*y + x*y^2 = 49 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2801_280121


namespace NUMINAMATH_CALUDE_ice_cream_volume_l2801_280168

/-- The volume of ice cream in a cone with hemisphere and cylinder topping -/
theorem ice_cream_volume (cone_height : Real) (cone_radius : Real) (cylinder_height : Real) :
  cone_height = 12 →
  cone_radius = 3 →
  cylinder_height = 2 →
  (1/3 * Real.pi * cone_radius^2 * cone_height) + 
  (2/3 * Real.pi * cone_radius^3) + 
  (Real.pi * cone_radius^2 * cylinder_height) = 72 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l2801_280168


namespace NUMINAMATH_CALUDE_pyramid_face_area_l2801_280187

theorem pyramid_face_area (base_edge : ℝ) (lateral_edge : ℝ) 
  (h_base : base_edge = 8) 
  (h_lateral : lateral_edge = 7) : 
  4 * (1/2 * base_edge * Real.sqrt (lateral_edge^2 - (base_edge/2)^2)) = 16 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_face_area_l2801_280187


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2801_280103

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x + a + 3 ≥ 0) ↔ a ∈ Set.Ici (0 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2801_280103


namespace NUMINAMATH_CALUDE_max_k_value_l2801_280137

/-- The maximum value of k satisfying the given inequality -/
theorem max_k_value (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_sum : a + b + c = 1) :
  (∃ k : ℝ, ∀ a b c : ℝ, 0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 1 →
    (a / (1 + 9*b*c + k*(b-c)^2)) + (b / (1 + 9*c*a + k*(c-a)^2)) + (c / (1 + 9*a*b + k*(a-b)^2)) ≥ 1/2) ∧
  (∀ k' : ℝ, k' > 4 →
    ∃ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 ∧
      (a / (1 + 9*b*c + k'*(b-c)^2)) + (b / (1 + 9*c*a + k'*(c-a)^2)) + (c / (1 + 9*a*b + k'*(a-b)^2)) < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l2801_280137


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l2801_280102

theorem min_perimeter_triangle (a b c : ℕ) : 
  a = 47 → b = 53 → c > 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  a + b + c ≥ 107 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l2801_280102


namespace NUMINAMATH_CALUDE_arccos_sqrt3_div2_l2801_280180

theorem arccos_sqrt3_div2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sqrt3_div2_l2801_280180


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2801_280178

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2801_280178


namespace NUMINAMATH_CALUDE_interest_difference_proof_l2801_280115

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem interest_difference_proof : 
  let principal : ℚ := 2500
  let rate : ℚ := 8
  let time : ℚ := 8
  let interest := simple_interest principal rate time
  principal - interest = 900 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_proof_l2801_280115


namespace NUMINAMATH_CALUDE_gcd_180_480_l2801_280149

theorem gcd_180_480 : Nat.gcd 180 480 = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_480_l2801_280149


namespace NUMINAMATH_CALUDE_angle_XZY_is_50_l2801_280146

/-- Given a diagram where AB and CD are straight lines -/
structure Diagram where
  /-- The angle AXB is 180 degrees -/
  angle_AXB : ℝ
  /-- The angle CYX is 120 degrees -/
  angle_CYX : ℝ
  /-- The angle YXB is 60 degrees -/
  angle_YXB : ℝ
  /-- The angle AXY is 50 degrees -/
  angle_AXY : ℝ
  /-- AB is a straight line -/
  h_AB_straight : angle_AXB = 180
  /-- CD is a straight line (not directly used but implied) -/
  h_CYX : angle_CYX = 120
  h_YXB : angle_YXB = 60
  h_AXY : angle_AXY = 50

/-- The theorem stating that the angle XZY is 50 degrees -/
theorem angle_XZY_is_50 (d : Diagram) : ∃ x, x = 50 ∧ x = d.angle_AXB - d.angle_CYX + d.angle_YXB - d.angle_AXY :=
  sorry

end NUMINAMATH_CALUDE_angle_XZY_is_50_l2801_280146


namespace NUMINAMATH_CALUDE_student_calculation_mistake_l2801_280171

theorem student_calculation_mistake (x y z : ℝ) 
  (h1 : x - (y - z) = 15) 
  (h2 : x - y - z = 7) : 
  x - y = 11 := by
sorry

end NUMINAMATH_CALUDE_student_calculation_mistake_l2801_280171


namespace NUMINAMATH_CALUDE_responses_always_match_l2801_280163

-- Define the types of inhabitants
inductive Inhabitant : Type
| Knight : Inhabitant
| Liar : Inhabitant

-- Define a function to represent an inhabitant's response
def responds (a b : Inhabitant) : Prop :=
  match a, b with
  | Inhabitant.Knight, Inhabitant.Knight => true
  | Inhabitant.Knight, Inhabitant.Liar => false
  | Inhabitant.Liar, Inhabitant.Knight => true
  | Inhabitant.Liar, Inhabitant.Liar => true

-- Theorem: The responses of two inhabitants about each other are always the same
theorem responses_always_match (a b : Inhabitant) :
  responds a b = responds b a :=
sorry

end NUMINAMATH_CALUDE_responses_always_match_l2801_280163


namespace NUMINAMATH_CALUDE_digital_earth_implies_science_technology_expression_l2801_280199

-- Define the concept of Digital Earth
def DigitalEarth : Prop := sorry

-- Define the concept of technological innovation paradigm
def TechnologicalInnovationParadigm : Prop := sorry

-- Define the concept of science and technology as expression of advanced productive forces
def ScienceTechnologyExpression : Prop := sorry

-- Theorem statement
theorem digital_earth_implies_science_technology_expression :
  (DigitalEarth → TechnologicalInnovationParadigm) →
  (TechnologicalInnovationParadigm → ScienceTechnologyExpression) :=
by
  sorry

end NUMINAMATH_CALUDE_digital_earth_implies_science_technology_expression_l2801_280199


namespace NUMINAMATH_CALUDE_new_library_capacity_l2801_280166

theorem new_library_capacity 
  (M : ℚ) -- Millicent's books
  (H : ℚ) -- Harold's books
  (G : ℚ) -- Gertrude's books
  (h1 : H = (1/2) * M) -- Harold has 1/2 as many books as Millicent
  (h2 : G = 3 * H) -- Gertrude has 3 times more books than Harold
  : (1/3) * H + (2/5) * G + (1/2) * M = (29/30) * M := by
  sorry

end NUMINAMATH_CALUDE_new_library_capacity_l2801_280166


namespace NUMINAMATH_CALUDE_chip_cost_is_correct_l2801_280169

/-- The cost of a bag of chips, given Amber's spending scenario -/
def chip_cost (total_money : ℚ) (candy_cost : ℚ) (candy_ounces : ℚ) (chip_ounces : ℚ) (max_ounces : ℚ) : ℚ :=
  total_money / (max_ounces / chip_ounces)

/-- Theorem stating that the cost of a bag of chips is $1.40 in Amber's scenario -/
theorem chip_cost_is_correct :
  chip_cost 7 1 12 17 85 = (14 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_chip_cost_is_correct_l2801_280169


namespace NUMINAMATH_CALUDE_ceremonial_team_arrangements_l2801_280105

def total_boys : ℕ := 48
def total_girls : ℕ := 32

def is_valid_arrangement (n : ℕ) : Prop :=
  n > 1 ∧
  total_boys % n = 0 ∧
  total_girls % n = 0 ∧
  (total_boys / n) = (total_girls / n) * 3 / 2

theorem ceremonial_team_arrangements :
  {n : ℕ | is_valid_arrangement n} = {2, 4, 8, 16} :=
by sorry

end NUMINAMATH_CALUDE_ceremonial_team_arrangements_l2801_280105


namespace NUMINAMATH_CALUDE_rectangle_area_18_pairs_l2801_280193

def rectangle_area_18 : Set (ℕ × ℕ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 18}

theorem rectangle_area_18_pairs : 
  rectangle_area_18 = {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_18_pairs_l2801_280193


namespace NUMINAMATH_CALUDE_solve_system_l2801_280117

theorem solve_system (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) :
  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_solve_system_l2801_280117


namespace NUMINAMATH_CALUDE_binomial_square_constant_l2801_280174

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x, 16 * x^2 + 40 * x + c = (a * x + b)^2) → c = 25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l2801_280174


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l2801_280152

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def drawn_balls : ℕ := 5

def probability_two_red : ℚ := 10 / 21

theorem probability_two_red_balls :
  (Nat.choose red_balls 2 * Nat.choose white_balls 3) / Nat.choose total_balls drawn_balls = probability_two_red :=
sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l2801_280152


namespace NUMINAMATH_CALUDE_circles_intersection_sum_l2801_280161

/-- Given two circles intersecting at points (1,3) and (m,1), with their centers 
    on the line 2x-y+c=0, prove that m + c = 1 -/
theorem circles_intersection_sum (m c : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    -- Centers of circles lie on the line 2x-y+c=0
    (2 * x₁ - y₁ + c = 0) ∧ 
    (2 * x₂ - y₂ + c = 0) ∧ 
    -- Circles intersect at (1,3) and (m,1)
    ((x₁ - 1)^2 + (y₁ - 3)^2 = (x₁ - m)^2 + (y₁ - 1)^2) ∧
    ((x₂ - 1)^2 + (y₂ - 3)^2 = (x₂ - m)^2 + (y₂ - 1)^2)) →
  m + c = 1 := by
sorry

end NUMINAMATH_CALUDE_circles_intersection_sum_l2801_280161


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2801_280140

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∃ (x y : ℝ), 2*x + 2*y = r ∧ x^2 + y^2 = 2*r) →
  (∀ (x y : ℝ), 2*x + 2*y = r → x^2 + y^2 ≥ 2*r) →
  r = 16 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2801_280140


namespace NUMINAMATH_CALUDE_sqrt_nine_subtraction_l2801_280158

theorem sqrt_nine_subtraction : 1 - Real.sqrt 9 = -2 := by sorry

end NUMINAMATH_CALUDE_sqrt_nine_subtraction_l2801_280158


namespace NUMINAMATH_CALUDE_cross_section_area_cross_section_area_is_14_l2801_280181

/-- Regular triangular prism with cross-section --/
structure Prism where
  a : ℝ  -- side length of the base
  S_ABC : ℝ  -- area of the base
  base_area_eq : S_ABC = a^2 * Real.sqrt 3 / 4
  D_midpoint : ℝ  -- D is midpoint of AB
  D_midpoint_eq : D_midpoint = a / 2
  K_on_BC : ℝ  -- distance BK
  K_on_BC_eq : K_on_BC = 3 * a / 4
  M_on_AC1 : ℝ  -- height of the prism
  N_on_A1B1 : ℝ  -- distance BG (projection of N)
  N_on_A1B1_eq : N_on_A1B1 = a / 6

/-- Theorem: The area of the cross-section is 14 --/
theorem cross_section_area (p : Prism) : ℝ :=
  let S_np := p.S_ABC * (3/8 - 1/24)  -- area of projection
  let cos_alpha := 1 / Real.sqrt 3
  S_np / cos_alpha

/-- Main theorem: The area of the cross-section is equal to 14 --/
theorem cross_section_area_is_14 (p : Prism) : cross_section_area p = 14 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_cross_section_area_is_14_l2801_280181


namespace NUMINAMATH_CALUDE_absolute_difference_of_numbers_l2801_280184

theorem absolute_difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 40) 
  (product_eq : x * y = 396) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_of_numbers_l2801_280184


namespace NUMINAMATH_CALUDE_hexagon_shape_partition_ways_l2801_280192

/-- A shape formed by gluing together congruent regular hexagons -/
structure HexagonShape where
  num_hexagons : ℕ
  num_quadrilaterals : ℕ

/-- The number of ways to partition a HexagonShape -/
def partition_ways (shape : HexagonShape) : ℕ :=
  2 ^ shape.num_hexagons

/-- The theorem to prove -/
theorem hexagon_shape_partition_ways :
  ∀ (shape : HexagonShape),
    shape.num_hexagons = 7 →
    shape.num_quadrilaterals = 21 →
    partition_ways shape = 128 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_shape_partition_ways_l2801_280192


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l2801_280186

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m - 2) * x^(|m|) + x - 1 = a * x^2 + b * x + c) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l2801_280186


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l2801_280111

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: For a geometric sequence, a₁ < a₃ if and only if a₅ < a₇ -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 1 < a 3 ↔ a 5 < a 7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l2801_280111


namespace NUMINAMATH_CALUDE_pepik_problem_l2801_280197

def letter_sum (M A T R D E I K U : Nat) : Nat :=
  4*M + 4*A + R + D + 2*T + E + I + K + U

theorem pepik_problem :
  (∀ M A T R D E I K U : Nat,
    M ≤ 9 ∧ A ≤ 9 ∧ T ≤ 9 ∧ R ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧ I ≤ 9 ∧ K ≤ 9 ∧ U ≤ 9 ∧
    M ≠ 0 ∧ A ≠ 0 ∧ T ≠ 0 ∧ R ≠ 0 ∧ D ≠ 0 ∧ E ≠ 0 ∧ I ≠ 0 ∧ K ≠ 0 ∧ U ≠ 0 ∧
    M ≠ A ∧ M ≠ T ∧ M ≠ R ∧ M ≠ D ∧ M ≠ E ∧ M ≠ I ∧ M ≠ K ∧ M ≠ U ∧
    A ≠ T ∧ A ≠ R ∧ A ≠ D ∧ A ≠ E ∧ A ≠ I ∧ A ≠ K ∧ A ≠ U ∧
    T ≠ R ∧ T ≠ D ∧ T ≠ E ∧ T ≠ I ∧ T ≠ K ∧ T ≠ U ∧
    R ≠ D ∧ R ≠ E ∧ R ≠ I ∧ R ≠ K ∧ R ≠ U ∧
    D ≠ E ∧ D ≠ I ∧ D ≠ K ∧ D ≠ U ∧
    E ≠ I ∧ E ≠ K ∧ E ≠ U ∧
    I ≠ K ∧ I ≠ U ∧
    K ≠ U →
    (∀ x : Nat, letter_sum M A T R D E I K U ≤ 103) ∧
    (letter_sum M A T R D E I K U ≠ 50) ∧
    (letter_sum M A T R D E I K U = 59 → (T = 5 ∨ T = 2))) :=
by sorry

end NUMINAMATH_CALUDE_pepik_problem_l2801_280197


namespace NUMINAMATH_CALUDE_sequence_fifth_term_l2801_280134

theorem sequence_fifth_term (a : ℕ → ℤ) :
  (∀ n : ℕ, a n = 4 * n - 3) →
  a 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sequence_fifth_term_l2801_280134


namespace NUMINAMATH_CALUDE_complex_simplification_l2801_280122

theorem complex_simplification :
  (4 - 3 * Complex.I) - (6 - 5 * Complex.I) + (2 + 3 * Complex.I) = 5 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l2801_280122


namespace NUMINAMATH_CALUDE_hall_length_l2801_280194

/-- Given a rectangular hall where the length is 5 meters more than the breadth
    and the area is 750 square meters, prove that the length is 30 meters. -/
theorem hall_length (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = breadth + 5 →
  area = length * breadth →
  area = 750 →
  length = 30 := by
sorry

end NUMINAMATH_CALUDE_hall_length_l2801_280194


namespace NUMINAMATH_CALUDE_parabola_equation_l2801_280190

theorem parabola_equation (a : ℝ) (x₀ : ℝ) : 
  (∃ (x : ℝ → ℝ) (y : ℝ → ℝ), 
    (∀ t, x t ^ 2 = a * y t) ∧ 
    (y x₀ = 2) ∧ 
    ((x x₀ - 0) ^ 2 + (y x₀ - a / 4) ^ 2 = 3 ^ 2)) → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2801_280190


namespace NUMINAMATH_CALUDE_gcd_1239_2829_times_15_l2801_280125

theorem gcd_1239_2829_times_15 : 15 * Int.gcd 1239 2829 = 315 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1239_2829_times_15_l2801_280125


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l2801_280195

/-- The area of the circle represented by the polar equation r = 3 cos θ - 4 sin θ -/
theorem circle_area_from_polar_equation : 
  let r : ℝ → ℝ := λ θ => 3 * Real.cos θ - 4 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (∀ θ, (r θ * Real.cos θ - center.1)^2 + (r θ * Real.sin θ - center.2)^2 = radius^2) ∧
    Real.pi * radius^2 = 25 * Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l2801_280195


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l2801_280138

theorem least_positive_integer_congruence :
  ∃! y : ℕ+, y.val + 3077 ≡ 1456 [ZMOD 15] ∧
  ∀ z : ℕ+, z.val + 3077 ≡ 1456 [ZMOD 15] → y ≤ z ∧ y.val = 14 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l2801_280138


namespace NUMINAMATH_CALUDE_constant_function_theorem_l2801_280154

/-- The set of all points in the plane -/
def S : Type := ℝ × ℝ

/-- A function from the plane to real numbers -/
def PlaneFunction : Type := S → ℝ

/-- Predicate for a nondegenerate triangle -/
def NonDegenerateTriangle (A B C : S) : Prop := sorry

/-- The orthocenter of a triangle -/
def Orthocenter (A B C : S) : S := sorry

/-- The property that the function satisfies for all nondegenerate triangles -/
def SatisfiesTriangleProperty (f : PlaneFunction) : Prop :=
  ∀ A B C : S, NonDegenerateTriangle A B C →
    let H := Orthocenter A B C
    (f A ≤ f B ∧ f B ≤ f C) → f A + f C = f B + f H

/-- The main theorem: if a function satisfies the triangle property, it must be constant -/
theorem constant_function_theorem (f : PlaneFunction) 
  (h : SatisfiesTriangleProperty f) : 
  ∀ x y : S, f x = f y := sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l2801_280154


namespace NUMINAMATH_CALUDE_fraction_value_l2801_280170

theorem fraction_value (x y : ℝ) (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 3) 
  (h3 : ∃ (n : ℤ), x / y = n) : x / y = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l2801_280170


namespace NUMINAMATH_CALUDE_paving_rate_per_square_metre_l2801_280191

/-- Given a room with length 5.5 m and width 3.75 m, and a total paving cost of Rs. 28875,
    the rate of paving per square metre is Rs. 1400. -/
theorem paving_rate_per_square_metre 
  (length : ℝ) 
  (width : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : total_cost = 28875) :
  total_cost / (length * width) = 1400 := by
sorry


end NUMINAMATH_CALUDE_paving_rate_per_square_metre_l2801_280191


namespace NUMINAMATH_CALUDE_dog_food_total_l2801_280198

/-- Theorem: Given an initial amount of dog food and two additional purchases,
    prove the total amount of dog food. -/
theorem dog_food_total (initial : ℕ) (bag1 : ℕ) (bag2 : ℕ) :
  initial = 15 → bag1 = 15 → bag2 = 10 → initial + bag1 + bag2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_total_l2801_280198


namespace NUMINAMATH_CALUDE_quadratic_function_unique_form_l2801_280153

/-- A quadratic function f(x) = x^2 + ax + b that intersects the x-axis at (1,0) 
    and has an axis of symmetry at x = 2 is equal to x^2 - 4x + 3. -/
theorem quadratic_function_unique_form 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = x^2 + a*x + b) 
  (h2 : f 1 = 0) 
  (h3 : ∀ x, f (2 + x) = f (2 - x)) : 
  ∀ x, f x = x^2 - 4*x + 3 := by 
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_form_l2801_280153


namespace NUMINAMATH_CALUDE_son_age_proof_l2801_280110

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 28 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l2801_280110


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2801_280104

theorem simple_interest_problem (P R : ℚ) : 
  P + (P * R * 2) / 100 = 820 → 
  P + (P * R * 6) / 100 = 1020 → 
  P = 720 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2801_280104


namespace NUMINAMATH_CALUDE_point_b_position_l2801_280131

theorem point_b_position (a b : ℤ) : 
  a = -5 → (b = a + 4 ∨ b = a - 4) → (b = -1 ∨ b = -9) := by
  sorry

end NUMINAMATH_CALUDE_point_b_position_l2801_280131


namespace NUMINAMATH_CALUDE_division_remainder_l2801_280112

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 149 →
  divisor = 16 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l2801_280112


namespace NUMINAMATH_CALUDE_fraction_equality_l2801_280151

theorem fraction_equality : (1877^2 - 1862^2) / (1880^2 - 1859^2) = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2801_280151


namespace NUMINAMATH_CALUDE_mollys_present_age_l2801_280118

def mollys_age_equation (x : ℕ) : Prop :=
  x + 18 = 5 * (x - 6)

theorem mollys_present_age : 
  ∃ (x : ℕ), mollys_age_equation x ∧ x = 12 :=
by sorry

end NUMINAMATH_CALUDE_mollys_present_age_l2801_280118


namespace NUMINAMATH_CALUDE_equation_solutions_l2801_280106

theorem equation_solutions :
  (∀ x : ℝ, (x - 4)^2 - 9 = 0 ↔ x = 7 ∨ x = 1) ∧
  (∀ x : ℝ, (x + 1)^3 = -27 ↔ x = -4) := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2801_280106


namespace NUMINAMATH_CALUDE_division_remainder_l2801_280142

theorem division_remainder : 
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := 2 * diff
  let dividend := 220040
  dividend % sum = 40 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_l2801_280142


namespace NUMINAMATH_CALUDE_distance_between_buildings_eight_trees_nine_meters_l2801_280182

/-- Given two buildings with trees planted between them, calculate the distance between the buildings. -/
theorem distance_between_buildings (num_trees : ℕ) (tree_spacing : ℕ) : ℕ :=
  (num_trees + 1) * tree_spacing

/-- Prove that with 8 trees planted 1 meter apart, the distance between buildings is 9 meters. -/
theorem eight_trees_nine_meters :
  distance_between_buildings 8 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_buildings_eight_trees_nine_meters_l2801_280182


namespace NUMINAMATH_CALUDE_definite_integral_f_l2801_280148

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x + 2

-- State the theorem
theorem definite_integral_f : ∫ x in (0:ℝ)..(1:ℝ), f x = 11/6 := by sorry

end NUMINAMATH_CALUDE_definite_integral_f_l2801_280148


namespace NUMINAMATH_CALUDE_digit_206788_is_7_l2801_280179

/-- The sequence of digits formed by concatenating all natural numbers from 1 onwards -/
def digit_sequence : ℕ → ℕ :=
  sorry

/-- The number of digits used to represent all natural numbers up to n -/
def digits_used_up_to (n : ℕ) : ℕ :=
  sorry

/-- The function that returns the digit at a given position in the sequence -/
def digit_at_position (pos : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the 206788th digit in the sequence is 7 -/
theorem digit_206788_is_7 : digit_at_position 206788 = 7 :=
  sorry

end NUMINAMATH_CALUDE_digit_206788_is_7_l2801_280179


namespace NUMINAMATH_CALUDE_intersection_point_l2801_280162

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = x + 3

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Theorem statement
theorem intersection_point :
  ∃ (x y : ℝ), line_equation x y ∧ y_axis x ∧ x = 0 ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l2801_280162


namespace NUMINAMATH_CALUDE_sum_in_base7_l2801_280164

/-- Converts a base 7 number to base 10 --/
def base7_to_base10 (x : ℕ) : ℕ :=
  (x / 10) * 7 + (x % 10)

/-- Converts a base 10 number to base 7 --/
def base10_to_base7 (x : ℕ) : ℕ :=
  if x < 7 then x
  else (base10_to_base7 (x / 7)) * 10 + (x % 7)

theorem sum_in_base7 :
  base10_to_base7 (base7_to_base10 15 + base7_to_base10 26) = 44 :=
by sorry

end NUMINAMATH_CALUDE_sum_in_base7_l2801_280164


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2801_280175

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  a > c →
  (1/2) * a * c * Real.sin B = 3/2 →
  Real.cos B = 4/5 →
  b = 3 * Real.sqrt 2 →
  (a = 5 ∧ c = 1) ∧
  Real.cos (B - C) = (31 * Real.sqrt 2) / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2801_280175


namespace NUMINAMATH_CALUDE_largest_modulus_of_cubic_root_l2801_280144

theorem largest_modulus_of_cubic_root (a b c d z : ℂ) :
  (Complex.abs a = Complex.abs b) →
  (Complex.abs b = Complex.abs c) →
  (Complex.abs c = Complex.abs d) →
  (Complex.abs a > 0) →
  (a * z^3 + b * z^2 + c * z + d = 0) →
  ∃ t : ℝ, t^3 - t^2 - t - 1 = 0 ∧ Complex.abs z ≤ t :=
by sorry

end NUMINAMATH_CALUDE_largest_modulus_of_cubic_root_l2801_280144


namespace NUMINAMATH_CALUDE_divided_square_area_is_eight_l2801_280196

/-- A square with a diagonal divided into three segments -/
structure DividedSquare where
  side : ℝ
  diagonal_length : ℝ
  de : ℝ
  ef : ℝ
  fb : ℝ
  diagonal_sum : de + ef + fb = diagonal_length
  diagonal_pythagoras : 2 * side * side = diagonal_length * diagonal_length

/-- The area of a square with a divided diagonal is 8 -/
theorem divided_square_area_is_eight (s : DividedSquare) 
  (h1 : s.de = 1) (h2 : s.ef = 2) (h3 : s.fb = 1) : s.side * s.side = 8 := by
  sorry

#check divided_square_area_is_eight

end NUMINAMATH_CALUDE_divided_square_area_is_eight_l2801_280196


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l2801_280176

/-- The number of points on the circle -/
def n : ℕ := 7

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The number of ways to select k chords from the total chords -/
def total_selections : ℕ := total_chords.choose k

/-- The number of ways to select k points from n points -/
def convex_quads : ℕ := n.choose k

/-- The probability of forming a convex quadrilateral -/
def probability : ℚ := convex_quads / total_selections

theorem convex_quadrilateral_probability :
  probability = 1 / 171 :=
sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l2801_280176


namespace NUMINAMATH_CALUDE_ed_marbles_l2801_280173

theorem ed_marbles (doug_initial : ℕ) (ed_more : ℕ) (doug_lost : ℕ) : 
  doug_initial = 22 → ed_more = 5 → doug_lost = 3 →
  doug_initial + ed_more = 27 :=
by sorry

end NUMINAMATH_CALUDE_ed_marbles_l2801_280173


namespace NUMINAMATH_CALUDE_race_head_start_l2801_280127

theorem race_head_start (v_a v_b L H : ℝ) : 
  v_a = (32/27) * v_b →
  (L / v_a = (L - H) / v_b) →
  H = (5/32) * L :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l2801_280127


namespace NUMINAMATH_CALUDE_students_surveyed_students_surveyed_proof_l2801_280126

theorem students_surveyed : ℕ :=
  let total_students : ℕ := sorry
  let french_speakers : ℕ := sorry
  let french_english_speakers : ℕ := 10
  let french_only_speakers : ℕ := 40

  have h1 : french_speakers = french_english_speakers + french_only_speakers := by sorry
  have h2 : french_speakers = 50 := by sorry
  have h3 : french_speakers = total_students / 4 := by sorry

  200

theorem students_surveyed_proof : students_surveyed = 200 := by sorry

end NUMINAMATH_CALUDE_students_surveyed_students_surveyed_proof_l2801_280126


namespace NUMINAMATH_CALUDE_hemisphere_intersection_area_l2801_280124

/-- Given two hemispheres A and B, where A has a surface area of 50π, and B has twice the surface area of A,
    if B shares 1/4 of its surface area with A, then the surface area of the remainder of hemisphere B
    after the intersection is 75π. -/
theorem hemisphere_intersection_area (A B : ℝ) : 
  A = 50 * Real.pi →
  B = 2 * A →
  let shared := (1/4) * B
  B - shared = 75 * Real.pi := by sorry

end NUMINAMATH_CALUDE_hemisphere_intersection_area_l2801_280124


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2801_280172

theorem inequality_equivalence (x : ℝ) :
  (3 * x - 4 < 9 - 2 * x + |x - 1|) ↔ (x < 3 ∧ x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2801_280172


namespace NUMINAMATH_CALUDE_sally_lemonade_sales_l2801_280177

/-- Calculates the total number of lemonade cups sold over two weeks -/
def total_lemonade_sales (last_week : ℕ) (increase_percentage : ℕ) : ℕ :=
  let this_week := last_week + last_week * increase_percentage / 100
  last_week + this_week

/-- Proves that given the conditions, Sally sold 46 cups of lemonade in total -/
theorem sally_lemonade_sales : total_lemonade_sales 20 30 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sally_lemonade_sales_l2801_280177


namespace NUMINAMATH_CALUDE_average_cost_28_apples_l2801_280135

/-- Represents the cost and quantity of apples in a bundle --/
structure AppleBundle where
  quantity : ℕ
  cost : ℕ

/-- Calculates the total number of apples received when purchasing a given amount --/
def totalApples (purchased : ℕ) : ℕ :=
  if purchased ≥ 20 then purchased + 5 else purchased

/-- Calculates the total cost of apples purchased --/
def totalCost (purchased : ℕ) : ℕ :=
  let bundle1 : AppleBundle := ⟨4, 15⟩
  let bundle2 : AppleBundle := ⟨7, 25⟩
  (purchased / bundle2.quantity) * bundle2.cost

/-- Theorem stating the average cost per apple when purchasing 28 apples --/
theorem average_cost_28_apples :
  (totalCost 28 : ℚ) / (totalApples 28 : ℚ) = 100 / 33 := by
  sorry

#check average_cost_28_apples

end NUMINAMATH_CALUDE_average_cost_28_apples_l2801_280135


namespace NUMINAMATH_CALUDE_min_tangent_product_right_triangle_l2801_280116

theorem min_tangent_product_right_triangle (A B C : Real) : 
  0 < A → A < π / 2 →
  0 < B → B < π / 2 →
  C ≤ π / 2 →
  A + B + C = π →
  (∀ A' B' C', 0 < A' → A' < π / 2 → 0 < B' → B' < π / 2 → C' ≤ π / 2 → A' + B' + C' = π → 
    Real.tan A * Real.tan B ≤ Real.tan A' * Real.tan B') →
  C = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_tangent_product_right_triangle_l2801_280116


namespace NUMINAMATH_CALUDE_discount_calculation_l2801_280100

-- Define the discount rate
def discount_rate : ℝ := 0.2

-- Define the original price (can be any positive real number)
variable (original_price : ℝ)
variable (original_price_positive : original_price > 0)

-- Define the purchase price
def purchase_price (original_price : ℝ) : ℝ := original_price * (1 - discount_rate)

-- Define the selling price
def selling_price (original_price : ℝ) : ℝ := original_price * 1.24

-- Theorem statement
theorem discount_calculation (original_price : ℝ) (original_price_positive : original_price > 0) :
  selling_price original_price = purchase_price original_price * 1.55 := by
  sorry

#check discount_calculation

end NUMINAMATH_CALUDE_discount_calculation_l2801_280100


namespace NUMINAMATH_CALUDE_age_of_other_man_l2801_280157

theorem age_of_other_man (n : ℕ) (avg_increase : ℝ) (age_one_man : ℕ) (avg_age_women : ℝ) :
  n = 8 ∧ 
  avg_increase = 2 ∧ 
  age_one_man = 20 ∧ 
  avg_age_women = 29 →
  ∃ (original_avg : ℝ) (age_other_man : ℕ),
    n * (original_avg + avg_increase) = n * original_avg + 2 * avg_age_women - (age_one_man + age_other_man) ∧
    age_other_man = 22 :=
by sorry

end NUMINAMATH_CALUDE_age_of_other_man_l2801_280157


namespace NUMINAMATH_CALUDE_trig_identity_l2801_280132

theorem trig_identity : 
  (Real.cos (70 * π / 180) + Real.cos (50 * π / 180)) * 
  (Real.cos (310 * π / 180) + Real.cos (290 * π / 180)) + 
  (Real.cos (40 * π / 180) + Real.cos (160 * π / 180)) * 
  (Real.cos (320 * π / 180) - Real.cos (380 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2801_280132


namespace NUMINAMATH_CALUDE_chord_length_of_intersecting_curves_l2801_280159

/-- The length of the chord formed by the intersection of two curves in polar coordinates -/
theorem chord_length_of_intersecting_curves (C₁ C₂ : ℝ → ℝ → Prop) :
  (∀ ρ θ, C₁ ρ θ ↔ ρ = 2 * Real.sin θ) →
  (∀ ρ θ, C₂ ρ θ ↔ θ = Real.pi / 3) →
  ∃ M N : ℝ × ℝ,
    (∃ ρ₁ θ₁, C₁ ρ₁ θ₁ ∧ M = (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁)) ∧
    (∃ ρ₂ θ₂, C₂ ρ₂ θ₂ ∧ M = (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂)) ∧
    (∃ ρ₃ θ₃, C₁ ρ₃ θ₃ ∧ N = (ρ₃ * Real.cos θ₃, ρ₃ * Real.sin θ₃)) ∧
    (∃ ρ₄ θ₄, C₂ ρ₄ θ₄ ∧ N = (ρ₄ * Real.cos θ₄, ρ₄ * Real.sin θ₄)) ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_of_intersecting_curves_l2801_280159


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l2801_280145

/-- The parabolas y = (x + 1)² and x + 4 = (y - 3)² intersect at four points that lie on a circle --/
theorem intersection_points_on_circle (x y : ℝ) : 
  (y = (x + 1)^2 ∧ x + 4 = (y - 3)^2) →
  ∃ (center : ℝ × ℝ), (x - center.1)^2 + (y - center.2)^2 = 13/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l2801_280145


namespace NUMINAMATH_CALUDE_number_of_big_boxes_l2801_280120

theorem number_of_big_boxes (soaps_per_package : ℕ) (packages_per_box : ℕ) (total_soaps : ℕ) : 
  soaps_per_package = 192 →
  packages_per_box = 6 →
  total_soaps = 2304 →
  total_soaps / (soaps_per_package * packages_per_box) = 2 :=
by
  sorry

#check number_of_big_boxes

end NUMINAMATH_CALUDE_number_of_big_boxes_l2801_280120


namespace NUMINAMATH_CALUDE_result_units_digit_is_seven_l2801_280150

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  hundreds_lt_10 : hundreds < 10
  tens_lt_10 : tens < 10
  units_lt_10 : units < 10
  hundreds_gt_0 : hundreds > 0

/-- The original three-digit number satisfying the condition -/
def original : ThreeDigitNumber := sorry

/-- The condition that the hundreds digit is 3 more than the units digit -/
axiom hundreds_units_relation : original.hundreds = original.units + 3

/-- The reversed number -/
def reversed : ThreeDigitNumber := sorry

/-- The result of subtracting the reversed number from the original number -/
def result : Nat := 
  (100 * original.hundreds + 10 * original.tens + original.units) - 
  (100 * reversed.hundreds + 10 * reversed.tens + reversed.units)

/-- The theorem stating that the units digit of the result is 7 -/
theorem result_units_digit_is_seven : result % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_result_units_digit_is_seven_l2801_280150


namespace NUMINAMATH_CALUDE_jungkook_has_largest_number_l2801_280183

/-- Given the numbers collected by Yoongi, Yuna, and Jungkook, prove that Jungkook has the largest number. -/
theorem jungkook_has_largest_number (yoongi_number yuna_number : ℕ) : 
  yoongi_number = 4 → 
  yuna_number = 5 → 
  6 + 3 > yoongi_number ∧ 6 + 3 > yuna_number := by
  sorry

#check jungkook_has_largest_number

end NUMINAMATH_CALUDE_jungkook_has_largest_number_l2801_280183


namespace NUMINAMATH_CALUDE_vectors_are_parallel_l2801_280101

def a : ℝ × ℝ × ℝ := (1, 2, -2)
def b : ℝ × ℝ × ℝ := (-2, -4, 4)

theorem vectors_are_parallel : ∃ (k : ℝ), b = k • a := by sorry

end NUMINAMATH_CALUDE_vectors_are_parallel_l2801_280101


namespace NUMINAMATH_CALUDE_train_passing_time_symmetry_l2801_280155

theorem train_passing_time_symmetry 
  (fast_train_length slow_train_length : ℝ)
  (time_slow_passes_fast : ℝ)
  (fast_train_length_pos : 0 < fast_train_length)
  (slow_train_length_pos : 0 < slow_train_length)
  (time_slow_passes_fast_pos : 0 < time_slow_passes_fast) :
  let total_length := fast_train_length + slow_train_length
  let relative_speed := total_length / time_slow_passes_fast
  total_length / relative_speed = time_slow_passes_fast :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_symmetry_l2801_280155


namespace NUMINAMATH_CALUDE_region_location_l2801_280136

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Define the region
def region (x y : ℝ) : Prop := x - 2*y + 6 > 0

-- Define what it means to be on the lower right side of the line
def lower_right_side (x y : ℝ) : Prop := 
  x > -6 ∧ y < 3 ∧ region x y

-- Theorem statement
theorem region_location : 
  ∀ x y : ℝ, region x y → lower_right_side x y :=
sorry

end NUMINAMATH_CALUDE_region_location_l2801_280136


namespace NUMINAMATH_CALUDE_simplify_expression_l2801_280128

theorem simplify_expression (y : ℝ) : 
  y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8) = 4 * y^3 - 6 * y^2 + 15 * y - 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2801_280128


namespace NUMINAMATH_CALUDE_root_difference_zero_l2801_280107

theorem root_difference_zero (x : ℝ) : 
  (x ^ 2 + 30 * x + 225 = 0) → (abs (x - x) = 0) := by
  sorry

end NUMINAMATH_CALUDE_root_difference_zero_l2801_280107


namespace NUMINAMATH_CALUDE_circle_max_values_l2801_280129

theorem circle_max_values (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ - x₀ = Real.sqrt 6 - 2) ∧
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀^2 + y₀^2 = 7 + 4*Real.sqrt 3) ∧
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀ ≠ 0 ∧ y₀ / x₀ = Real.sqrt 3) ∧
  (∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀ + y₀ = 2 + Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_max_values_l2801_280129


namespace NUMINAMATH_CALUDE_factor_sum_l2801_280189

theorem factor_sum (a b : ℤ) : 
  (∀ x, 25 * x^2 - 130 * x - 120 = (5 * x + a) * (5 * x + b)) →
  a + 3 * b = -86 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l2801_280189


namespace NUMINAMATH_CALUDE_new_observation_count_l2801_280139

theorem new_observation_count (initial_count : ℕ) (initial_avg : ℚ) (new_obs : ℚ) (avg_decrease : ℚ) : 
  initial_count = 6 → 
  initial_avg = 13 → 
  new_obs = 6 → 
  avg_decrease = 1 → 
  (initial_count * initial_avg + new_obs) / (initial_count + 1) = initial_avg - avg_decrease → 
  initial_count + 1 = 7 := by
sorry

end NUMINAMATH_CALUDE_new_observation_count_l2801_280139


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2801_280141

def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_inequality (b c : ℝ) (h : f b c (-1) = f b c 3) :
  f b c 1 < c ∧ c < f b c (-1) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2801_280141


namespace NUMINAMATH_CALUDE_lice_check_time_is_three_hours_l2801_280143

/-- The total time required to check all students for lice -/
def total_check_time (kindergarteners first_graders second_graders third_graders : ℕ) 
  (check_time_per_student : ℕ) : ℚ :=
  let total_students := kindergarteners + first_graders + second_graders + third_graders
  let total_minutes := total_students * check_time_per_student
  (total_minutes : ℚ) / 60

/-- Theorem stating that the total check time for the given number of students is 3 hours -/
theorem lice_check_time_is_three_hours :
  total_check_time 26 19 20 25 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lice_check_time_is_three_hours_l2801_280143


namespace NUMINAMATH_CALUDE_gain_percent_is_one_percent_l2801_280119

-- Define the gain and cost price
def gain : ℚ := 70 / 100  -- 70 paise = 0.70 Rs
def cost_price : ℚ := 70  -- 70 Rs

-- Define the gain percent formula
def gain_percent (g c : ℚ) : ℚ := (g / c) * 100

-- Theorem statement
theorem gain_percent_is_one_percent :
  gain_percent gain cost_price = 1 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_is_one_percent_l2801_280119


namespace NUMINAMATH_CALUDE_modified_mindmaster_codes_l2801_280165

/-- The number of possible secret codes in a modified Mindmaster game -/
def secret_codes (num_colors : ℕ) (num_slots : ℕ) : ℕ :=
  num_colors ^ num_slots

/-- Theorem: The number of secret codes in a game with 5 colors and 6 slots is 15625 -/
theorem modified_mindmaster_codes :
  secret_codes 5 6 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_modified_mindmaster_codes_l2801_280165


namespace NUMINAMATH_CALUDE_tan_sum_zero_implies_tan_sqrt_three_l2801_280156

open Real

theorem tan_sum_zero_implies_tan_sqrt_three (θ : ℝ) :
  π/4 < θ ∧ θ < π/2 →
  tan θ + tan (2*θ) + tan (3*θ) + tan (4*θ) = 0 →
  tan θ = sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_zero_implies_tan_sqrt_three_l2801_280156


namespace NUMINAMATH_CALUDE_mulch_cost_theorem_l2801_280108

/-- The cost of mulch in dollars per cubic foot -/
def mulch_cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of mulch in cubic yards -/
def mulch_volume_cubic_yards : ℝ := 7

/-- The cost of mulch for a given volume in cubic yards -/
def mulch_cost (volume_cubic_yards : ℝ) : ℝ :=
  volume_cubic_yards * cubic_yards_to_cubic_feet * mulch_cost_per_cubic_foot

theorem mulch_cost_theorem : mulch_cost mulch_volume_cubic_yards = 1512 := by
  sorry

end NUMINAMATH_CALUDE_mulch_cost_theorem_l2801_280108


namespace NUMINAMATH_CALUDE_elder_son_toys_l2801_280167

theorem elder_son_toys (total : ℕ) (younger_ratio : ℕ) : 
  total = 240 → younger_ratio = 3 → 
  ∃ (elder : ℕ), elder * (1 + younger_ratio) = total ∧ elder = 60 := by
sorry

end NUMINAMATH_CALUDE_elder_son_toys_l2801_280167


namespace NUMINAMATH_CALUDE_starting_number_is_24_l2801_280114

/-- Given that there are 35 even integers between a starting number and 95,
    prove that the starting number is 24. -/
theorem starting_number_is_24 (start : ℤ) : 
  (start < 95) →
  (∃ (evens : Finset ℤ), evens.card = 35 ∧ 
    (∀ n ∈ evens, start < n ∧ n < 95 ∧ Even n) ∧
    (∀ n, start < n ∧ n < 95 ∧ Even n → n ∈ evens)) →
  start = 24 := by
sorry

end NUMINAMATH_CALUDE_starting_number_is_24_l2801_280114


namespace NUMINAMATH_CALUDE_triangle_inequality_l2801_280147

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let s := (a + b + c) / 2
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * Real.sqrt (s * (s - a) * (s - b) * (s - c)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2801_280147


namespace NUMINAMATH_CALUDE_female_turtle_percentage_is_60_l2801_280160

/-- Represents the number of turtles in the lake -/
def total_turtles : ℕ := 100

/-- Represents the fraction of male turtles that have stripes -/
def male_stripe_ratio : ℚ := 1 / 4

/-- Represents the number of baby striped male turtles -/
def baby_striped_males : ℕ := 4

/-- Represents the percentage of adult striped male turtles -/
def adult_striped_male_percentage : ℚ := 60 / 100

/-- Calculates the percentage of female turtles in the lake -/
def female_turtle_percentage : ℚ :=
  let total_striped_males : ℚ := baby_striped_males / (1 - adult_striped_male_percentage)
  let total_males : ℚ := total_striped_males / male_stripe_ratio
  let total_females : ℚ := total_turtles - total_males
  (total_females / total_turtles) * 100

theorem female_turtle_percentage_is_60 :
  female_turtle_percentage = 60 := by sorry

end NUMINAMATH_CALUDE_female_turtle_percentage_is_60_l2801_280160
