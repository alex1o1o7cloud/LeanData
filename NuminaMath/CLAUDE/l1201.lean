import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_area_example_l1201_120160

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_example : parallelogram_area 12 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l1201_120160


namespace NUMINAMATH_CALUDE_longest_piece_length_l1201_120148

theorem longest_piece_length (a b c : ℕ) (ha : a = 45) (hb : b = 75) (hc : c = 90) :
  Nat.gcd a (Nat.gcd b c) = 15 := by
  sorry

end NUMINAMATH_CALUDE_longest_piece_length_l1201_120148


namespace NUMINAMATH_CALUDE_multiples_of_six_ending_in_four_l1201_120158

theorem multiples_of_six_ending_in_four (n : ℕ) : 
  (∃ k, k ∈ Finset.range 1000 ∧ k % 10 = 4 ∧ k % 6 = 0) ↔ n = 17 :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_six_ending_in_four_l1201_120158


namespace NUMINAMATH_CALUDE_almond_walnut_ratio_is_five_to_two_l1201_120175

/-- Represents a mixture of almonds and walnuts -/
structure NutMixture where
  total_weight : ℝ
  almond_weight : ℝ
  almond_parts : ℝ
  walnut_parts : ℝ

/-- The ratio of almonds to walnuts in a nut mixture -/
def almond_to_walnut_ratio (mix : NutMixture) : ℝ × ℝ :=
  (mix.almond_parts, mix.walnut_parts)

/-- Theorem stating the ratio of almonds to walnuts in the specific mixture -/
theorem almond_walnut_ratio_is_five_to_two 
  (mix : NutMixture)
  (h1 : mix.total_weight = 210)
  (h2 : mix.almond_weight = 150)
  (h3 : mix.almond_parts = 5)
  (h4 : mix.almond_parts + mix.walnut_parts = mix.total_weight / (mix.almond_weight / mix.almond_parts)) :
  almond_to_walnut_ratio mix = (5, 2) := by
  sorry


end NUMINAMATH_CALUDE_almond_walnut_ratio_is_five_to_two_l1201_120175


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l1201_120172

theorem not_p_necessary_not_sufficient_for_not_q 
  (h1 : p → q) 
  (h2 : ¬(q → p)) : 
  (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_not_q_l1201_120172


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_120_504_l1201_120104

theorem lcm_gcf_ratio_120_504 : 
  (Nat.lcm 120 504) / (Nat.gcd 120 504) = 105 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_120_504_l1201_120104


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1201_120106

theorem solution_set_inequality (x : ℝ) :
  (x^2 - |x| > 0) ↔ (x < -1 ∨ x > 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1201_120106


namespace NUMINAMATH_CALUDE_line_symmetry_l1201_120190

/-- Given two lines in the plane and a point, this theorem states that
    the lines are symmetric about the point. -/
theorem line_symmetry (x y : ℝ) : 
  (2 * x + 3 * y - 6 = 0) → 
  (2 * ((2 : ℝ) - x) + 3 * ((2 : ℝ) - y) - 6 = 0) →
  (2 * x + 3 * y - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l1201_120190


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l1201_120179

theorem min_x_prime_factorization (x y : ℕ+) (h : 5 * x ^ 7 = 13 * y ^ 11) :
  ∃ (a b c d : ℕ),
    x = a ^ c * b ^ d ∧
    a.Prime ∧ b.Prime ∧
    (∀ (a' b' c' d' : ℕ), x = a' ^ c' * b' ^ d' → a' ^ c' * b' ^ d' ≥ a ^ c * b ^ d) ∧
    a + b + c + d = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l1201_120179


namespace NUMINAMATH_CALUDE_car_journey_distance_l1201_120135

/-- Represents the car's journey with given speeds and break times -/
structure CarJourney where
  initial_speed : ℝ
  initial_duration : ℝ
  second_speed : ℝ
  second_duration : ℝ
  final_speed : ℝ
  final_duration : ℝ

/-- Calculates the total distance covered by the car -/
def total_distance (journey : CarJourney) : ℝ :=
  journey.initial_speed * journey.initial_duration +
  journey.second_speed * journey.second_duration +
  journey.final_speed * journey.final_duration

/-- Theorem stating that the car's journey covers 390 miles -/
theorem car_journey_distance :
  let journey : CarJourney := {
    initial_speed := 65,
    initial_duration := 2,
    second_speed := 60,
    second_duration := 2.5,
    final_speed := 55,
    final_duration := 2
  }
  total_distance journey = 390 := by sorry

end NUMINAMATH_CALUDE_car_journey_distance_l1201_120135


namespace NUMINAMATH_CALUDE_james_paper_usage_l1201_120142

/-- The number of books James prints -/
def num_books : ℕ := 2

/-- The number of pages in each book -/
def pages_per_book : ℕ := 600

/-- The number of pages printed on each side of a sheet -/
def pages_per_side : ℕ := 4

/-- Whether the printing is double-sided (true) or single-sided (false) -/
def is_double_sided : Bool := true

/-- Calculates the total number of sheets of paper James uses -/
def sheets_used : ℕ :=
  let total_pages := num_books * pages_per_book
  let pages_per_sheet := pages_per_side * (if is_double_sided then 2 else 1)
  total_pages / pages_per_sheet

theorem james_paper_usage :
  sheets_used = 150 := by sorry

end NUMINAMATH_CALUDE_james_paper_usage_l1201_120142


namespace NUMINAMATH_CALUDE_hidden_block_surface_area_l1201_120183

/-- Represents a block with a surface area -/
structure Block where
  surfaceArea : ℝ

/-- Represents a set of blocks created by cutting a larger block -/
structure CutBlocks where
  blocks : List Block
  numCuts : ℕ

/-- The proposition that the surface area of the hidden block is correct -/
def hiddenBlockSurfaceAreaIsCorrect (cb : CutBlocks) (hiddenSurfaceArea : ℝ) : Prop :=
  cb.numCuts = 3 ∧ 
  cb.blocks.length = 7 ∧ 
  (cb.blocks.map Block.surfaceArea).sum = 566 ∧
  hiddenSurfaceArea = 22

/-- Theorem stating that given the conditions, the hidden block's surface area is 22 -/
theorem hidden_block_surface_area 
  (cb : CutBlocks) (hiddenSurfaceArea : ℝ) : 
  hiddenBlockSurfaceAreaIsCorrect cb hiddenSurfaceArea := by
  sorry

#check hidden_block_surface_area

end NUMINAMATH_CALUDE_hidden_block_surface_area_l1201_120183


namespace NUMINAMATH_CALUDE_correct_ordering_l1201_120176

/-- Represents the labels of the conjectures -/
inductive ConjLabel
  | A | C | G | P | R | E | S

/-- The smallest counterexample for each conjecture -/
def smallest_counterexample : ConjLabel → ℕ
  | ConjLabel.A => 44
  | ConjLabel.C => 105
  | ConjLabel.G => 5777
  | ConjLabel.P => 906150257
  | ConjLabel.R => 23338590792
  | ConjLabel.E => 31858749840007945920321
  | ConjLabel.S => 8424432925592889329288197322308900672459420460792433

/-- Checks if a list of ConjLabels is in ascending order based on their smallest counterexamples -/
def is_ascending (labels : List ConjLabel) : Prop :=
  labels.Pairwise (λ l1 l2 => smallest_counterexample l1 < smallest_counterexample l2)

/-- The theorem to be proved -/
theorem correct_ordering :
  is_ascending [ConjLabel.A, ConjLabel.C, ConjLabel.G, ConjLabel.P, ConjLabel.R, ConjLabel.E, ConjLabel.S] :=
by sorry

end NUMINAMATH_CALUDE_correct_ordering_l1201_120176


namespace NUMINAMATH_CALUDE_business_profit_l1201_120162

theorem business_profit (total_profit : ℝ) : 
  (0.25 * total_profit) + (2 * (0.25 * (0.75 * total_profit))) = 50000 → 
  total_profit = 80000 := by
sorry

end NUMINAMATH_CALUDE_business_profit_l1201_120162


namespace NUMINAMATH_CALUDE_shift_sine_graph_l1201_120163

/-- The problem statement as a theorem -/
theorem shift_sine_graph (φ : ℝ) (h₁ : 0 < φ) (h₂ : φ < π) : 
  let f : ℝ → ℝ := λ x => 2 * Real.sin (2 * x)
  let g : ℝ → ℝ := λ x => 2 * Real.sin (2 * x - 2 * φ)
  (∃ x₁ x₂ : ℝ, |f x₁ - g x₂| = 4 ∧ 
    (∀ y₁ y₂ : ℝ, |f y₁ - g y₂| = 4 → |x₁ - x₂| ≤ |y₁ - y₂|) ∧
    |x₁ - x₂| = π / 6) →
  φ = π / 3 ∨ φ = 2 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_shift_sine_graph_l1201_120163


namespace NUMINAMATH_CALUDE_set1_equality_set2_equality_set3_equality_set4_equality_set5_equality_l1201_120103

-- Set 1
def set1 : Set ℤ := {x | x.natAbs ≤ 2}
def set1_alt : Set ℤ := {-2, -1, 0, 1, 2}

theorem set1_equality : set1 = set1_alt := by sorry

-- Set 2
def set2 : Set ℕ := {x | x > 0 ∧ x % 3 = 0 ∧ x < 10}
def set2_alt : Set ℕ := {3, 6, 9}

theorem set2_equality : set2 = set2_alt := by sorry

-- Set 3
def set3 : Set ℤ := {x | x = Int.natAbs x ∧ x < 5}
def set3_alt : Set ℤ := {0, 1, 2, 3, 4}

theorem set3_equality : set3 = set3_alt := by sorry

-- Set 4
def set4 : Set (ℕ+ × ℕ+) := {p | p.1 + p.2 = 6}
def set4_alt : Set (ℕ+ × ℕ+) := {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)}

theorem set4_equality : set4 = set4_alt := by sorry

-- Set 5
def set5 : Set ℤ := {-3, -1, 1, 3, 5}
def set5_alt : Set ℤ := {x | ∃ k : ℤ, x = 2*k - 1 ∧ -1 ≤ k ∧ k ≤ 3}

theorem set5_equality : set5 = set5_alt := by sorry

end NUMINAMATH_CALUDE_set1_equality_set2_equality_set3_equality_set4_equality_set5_equality_l1201_120103


namespace NUMINAMATH_CALUDE_triangle_area_l1201_120146

/-- The area of a triangle with vertices at (-2,3), (7,-3), and (4,6) is 31.5 square units. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (7, -3)
  let C : ℝ × ℝ := (4, 6)
  let area := (1/2) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|
  area = 31.5 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_l1201_120146


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_eight_satisfies_inequality_eight_is_smallest_integer_l1201_120166

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, y < 3*y - 15 → y ≥ 8 :=
by
  sorry

theorem eight_satisfies_inequality : 
  (8 : ℤ) < 3*8 - 15 :=
by
  sorry

theorem eight_is_smallest_integer :
  ∃ y : ℤ, y < 3*y - 15 ∧ ∀ z : ℤ, z < 3*z - 15 → z ≥ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_eight_satisfies_inequality_eight_is_smallest_integer_l1201_120166


namespace NUMINAMATH_CALUDE_max_value_of_f_l1201_120199

-- Define the function f(x) = x³ - 3x²
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Theorem statement
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 4 ∧
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 4 → f x ≤ f c) ∧
  f c = 16 := by
sorry


end NUMINAMATH_CALUDE_max_value_of_f_l1201_120199


namespace NUMINAMATH_CALUDE_isosceles_triangles_height_ratio_l1201_120125

/-- Two isosceles triangles with equal vertical angles and areas in ratio 16:36 have heights in ratio 2:3 -/
theorem isosceles_triangles_height_ratio (b₁ b₂ h₁ h₂ : ℝ) (area₁ area₂ : ℝ) :
  b₁ > 0 → b₂ > 0 → h₁ > 0 → h₂ > 0 →
  area₁ = (b₁ * h₁) / 2 →
  area₂ = (b₂ * h₂) / 2 →
  area₁ / area₂ = 16 / 36 →
  b₁ / b₂ = h₁ / h₂ →
  h₁ / h₂ = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_height_ratio_l1201_120125


namespace NUMINAMATH_CALUDE_art_club_collection_l1201_120119

/-- The number of artworks collected by the art club in two school years -/
def artworks_collected (num_students : ℕ) (artworks_per_student_per_quarter : ℕ) (quarters_per_year : ℕ) (years : ℕ) : ℕ :=
  num_students * artworks_per_student_per_quarter * quarters_per_year * years

/-- Theorem stating that the art club collects 240 artworks in two school years -/
theorem art_club_collection :
  artworks_collected 15 2 4 2 = 240 := by
  sorry

#eval artworks_collected 15 2 4 2

end NUMINAMATH_CALUDE_art_club_collection_l1201_120119


namespace NUMINAMATH_CALUDE_congruence_problem_l1201_120168

theorem congruence_problem (x : ℤ) :
  (5 * x + 9) % 20 = 3 → (3 * x + 14) % 20 = 12 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1201_120168


namespace NUMINAMATH_CALUDE_olive_flea_fraction_is_half_l1201_120101

/-- The fraction of fleas Olive has compared to Gertrude -/
def olive_flea_fraction (gertrude_fleas maud_fleas olive_fleas total_fleas : ℕ) : ℚ :=
  olive_fleas / gertrude_fleas

theorem olive_flea_fraction_is_half :
  ∀ (gertrude_fleas maud_fleas olive_fleas total_fleas : ℕ),
    gertrude_fleas = 10 →
    maud_fleas = 5 * olive_fleas →
    total_fleas = 40 →
    gertrude_fleas + maud_fleas + olive_fleas = total_fleas →
    olive_flea_fraction gertrude_fleas maud_fleas olive_fleas total_fleas = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_olive_flea_fraction_is_half_l1201_120101


namespace NUMINAMATH_CALUDE_gcf_36_60_l1201_120112

theorem gcf_36_60 : Nat.gcd 36 60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_36_60_l1201_120112


namespace NUMINAMATH_CALUDE_unique_solution_l1201_120105

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that 1958 is the unique solution -/
theorem unique_solution : ∃! n : ℕ, n + S n = 1981 ∧ n = 1958 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1201_120105


namespace NUMINAMATH_CALUDE_julia_math_contest_julia_math_contest_proof_l1201_120191

theorem julia_math_contest (total_problems : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) 
  (julia_score : ℤ) (julia_correct : ℕ) : Prop :=
  total_problems = 12 →
  correct_points = 6 →
  incorrect_points = -3 →
  julia_score = 27 →
  julia_correct = 7 →
  (julia_correct : ℤ) * correct_points + (total_problems - julia_correct : ℤ) * incorrect_points = julia_score

theorem julia_math_contest_proof : 
  ∃ (total_problems : ℕ) (correct_points incorrect_points julia_score : ℤ) (julia_correct : ℕ),
    julia_math_contest total_problems correct_points incorrect_points julia_score julia_correct :=
by
  sorry

end NUMINAMATH_CALUDE_julia_math_contest_julia_math_contest_proof_l1201_120191


namespace NUMINAMATH_CALUDE_trapezium_marked_length_l1201_120187

/-- Represents an isosceles triangle ABC with base AC and equal sides AB and BC -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ

/-- Represents a trapezium AMNC formed from an isosceles triangle ABC -/
structure Trapezium (triangle : IsoscelesTriangle) where
  markedLength : ℝ
  perimeter : ℝ

/-- Theorem: In an isosceles triangle with base 12 and side 18, 
    if a trapezium is formed with perimeter 40, 
    then the marked length on each side is 6 -/
theorem trapezium_marked_length 
  (triangle : IsoscelesTriangle) 
  (trap : Trapezium triangle) : 
  triangle.base = 12 → 
  triangle.side = 18 → 
  trap.perimeter = 40 → 
  trap.markedLength = 6 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_marked_length_l1201_120187


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1201_120151

theorem coin_flip_probability (n : ℕ) : 
  (n.choose 2 : ℚ) / 2^n = 1/8 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1201_120151


namespace NUMINAMATH_CALUDE_unique_triangle_exists_l1201_120174

/-- Triangle ABC with given side lengths and angle -/
structure Triangle where
  a : ℝ
  b : ℝ
  A : ℝ

/-- Predicate for a valid triangle satisfying the given conditions -/
def is_valid_triangle (t : Triangle) : Prop :=
  t.a = Real.sqrt 3 ∧ t.b = 1 ∧ t.A = 130 * (Real.pi / 180)

/-- Theorem stating that there exists exactly one valid triangle -/
theorem unique_triangle_exists : ∃! t : Triangle, is_valid_triangle t :=
sorry

end NUMINAMATH_CALUDE_unique_triangle_exists_l1201_120174


namespace NUMINAMATH_CALUDE_relationship_between_p_and_q_l1201_120136

theorem relationship_between_p_and_q (p q : ℝ) (h : p > 0) (h' : q > 0) (h'' : q ≠ 1) 
  (eq : Real.log p + Real.log q = Real.log (p + q + q^2)) : 
  p = (q + q^2) / (q - 1) := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_p_and_q_l1201_120136


namespace NUMINAMATH_CALUDE_compound_weight_proof_l1201_120108

/-- Molar mass of Nitrogen in g/mol -/
def N_mass : ℝ := 14.01

/-- Molar mass of Hydrogen in g/mol -/
def H_mass : ℝ := 1.01

/-- Molar mass of Iodine in g/mol -/
def I_mass : ℝ := 126.90

/-- Molar mass of Oxygen in g/mol -/
def O_mass : ℝ := 16.00

/-- Molar mass of NH4I in g/mol -/
def NH4I_mass : ℝ := N_mass + 4 * H_mass + I_mass

/-- Molar mass of H2O in g/mol -/
def H2O_mass : ℝ := 2 * H_mass + O_mass

/-- Number of moles of NH4I -/
def NH4I_moles : ℝ := 15

/-- Number of moles of H2O -/
def H2O_moles : ℝ := 7

/-- Total weight of the compound (NH4I·H2O) in grams -/
def total_weight : ℝ := NH4I_moles * NH4I_mass + H2O_moles * H2O_mass

theorem compound_weight_proof : total_weight = 2300.39 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_proof_l1201_120108


namespace NUMINAMATH_CALUDE_quadratic_has_real_roots_k_values_l1201_120141

-- Define the quadratic equation
def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := (x - 1)^2 + k*(x - 1)

-- Theorem 1: The quadratic equation always has real roots
theorem quadratic_has_real_roots (k : ℝ) : 
  ∃ x : ℝ, quadratic_equation k x = 0 :=
sorry

-- Theorem 2: If the roots satisfy the given condition, k is either 4 or -1
theorem k_values (k : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    quadratic_equation k x₁ = 0 ∧ 
    quadratic_equation k x₂ = 0 ∧ 
    x₁^2 + x₂^2 = 7 - x₁*x₂) →
  (k = 4 ∨ k = -1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_real_roots_k_values_l1201_120141


namespace NUMINAMATH_CALUDE_physics_class_size_l1201_120157

theorem physics_class_size (total_students : ℕ) (both_classes : ℕ) :
  total_students = 75 →
  both_classes = 9 →
  ∃ (math_only : ℕ) (phys_only : ℕ),
    total_students = math_only + phys_only + both_classes ∧
    phys_only + both_classes = 2 * (math_only + both_classes) →
  phys_only + both_classes = 56 := by
  sorry

end NUMINAMATH_CALUDE_physics_class_size_l1201_120157


namespace NUMINAMATH_CALUDE_closed_broken_line_length_lower_bound_l1201_120122

/-- A closed broken line on the surface of a unit cube -/
structure ClosedBrokenLine where
  /-- The line passes over the surface of the cube -/
  onSurface : Bool
  /-- The line has common points with all faces of the cube -/
  touchesAllFaces : Bool
  /-- The length of the line -/
  length : ℝ

/-- Theorem: The length of a closed broken line on a unit cube touching all faces is at least 3√2 -/
theorem closed_broken_line_length_lower_bound (line : ClosedBrokenLine) 
    (h1 : line.onSurface = true) 
    (h2 : line.touchesAllFaces = true) : 
  line.length ≥ 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_closed_broken_line_length_lower_bound_l1201_120122


namespace NUMINAMATH_CALUDE_constant_d_value_l1201_120132

theorem constant_d_value (e f d : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 2 * x + 4) * (e * x^2 + d * x + f) = 9 * x^4 - 8 * x^3 + 13 * x^2 + 12 * x - 16) →
  d = -2/3 := by
sorry

end NUMINAMATH_CALUDE_constant_d_value_l1201_120132


namespace NUMINAMATH_CALUDE_canoe_travel_time_l1201_120128

/-- Given two villages A and B connected by a river with current velocity v_r,
    and a canoe with velocity v in still water, prove that if the time to travel
    from A to B is 3 times the time to travel from B to A, and v = 2*v_r, then
    the time to travel from B to A without paddles is 3 times longer than with paddles. -/
theorem canoe_travel_time (v v_r : ℝ) (S : ℝ) (h1 : v > 0) (h2 : v_r > 0) (h3 : S > 0) :
  (S / (v + v_r) = 3 * S / (v - v_r)) → (v = 2 * v_r) → (S / v_r = 3 * S / (v - v_r)) := by
  sorry

end NUMINAMATH_CALUDE_canoe_travel_time_l1201_120128


namespace NUMINAMATH_CALUDE_dans_remaining_marbles_l1201_120159

-- Define the initial number of green marbles Dan has
def initial_green_marbles : ℝ := 32.0

-- Define the number of green marbles Mike took
def marbles_taken : ℝ := 23.0

-- Define the number of green marbles Dan has now
def remaining_green_marbles : ℝ := initial_green_marbles - marbles_taken

-- Theorem to prove
theorem dans_remaining_marbles :
  remaining_green_marbles = 9.0 := by sorry

end NUMINAMATH_CALUDE_dans_remaining_marbles_l1201_120159


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1201_120197

theorem simplify_trig_expression :
  let cos45 := Real.sqrt 2 / 2
  let sin45 := Real.sqrt 2 / 2
  (cos45^3 + sin45^3) / (cos45 + sin45) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1201_120197


namespace NUMINAMATH_CALUDE_triangle_property_l1201_120185

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2a*sin(B) = √3*b, a = 6, and b = 2√3, then angle A = π/3 and the area is 6√3 --/
theorem triangle_property (a b c A B C : Real) : 
  0 < A ∧ A < π/2 →  -- Acute triangle condition
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  2 * a * Real.sin B = Real.sqrt 3 * b →  -- Given condition
  a = 6 →  -- Given condition
  b = 2 * Real.sqrt 3 →  -- Given condition
  A = π/3 ∧ (1/2 * b * c * Real.sin A = 6 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1201_120185


namespace NUMINAMATH_CALUDE_function_translation_l1201_120124

/-- Given a function f(x) = 3 * sin(2x + π/3), prove that translating it right by π/6 units
    and then downwards by 1 unit results in the function g(x) = 3 * sin(2x) - 1 -/
theorem function_translation (x : ℝ) :
  let f := λ x : ℝ => 3 * Real.sin (2 * x + π / 3)
  let g := λ x : ℝ => 3 * Real.sin (2 * x) - 1
  f (x - π / 6) - 1 = g x := by
  sorry

end NUMINAMATH_CALUDE_function_translation_l1201_120124


namespace NUMINAMATH_CALUDE_blue_ridge_elementary_calculation_l1201_120173

theorem blue_ridge_elementary_calculation (num_classrooms : ℕ) 
  (students_per_classroom : ℕ) (turtles_per_classroom : ℕ) (teachers_per_classroom : ℕ) : 
  num_classrooms = 6 →
  students_per_classroom = 22 →
  turtles_per_classroom = 2 →
  teachers_per_classroom = 1 →
  num_classrooms * students_per_classroom - 
  (num_classrooms * turtles_per_classroom + num_classrooms * teachers_per_classroom) = 114 := by
  sorry

#check blue_ridge_elementary_calculation

end NUMINAMATH_CALUDE_blue_ridge_elementary_calculation_l1201_120173


namespace NUMINAMATH_CALUDE_stratified_sampling_second_group_l1201_120184

theorem stratified_sampling_second_group (total_sample : ℕ) 
  (ratio_first ratio_second ratio_third : ℕ) :
  ratio_first > 0 ∧ ratio_second > 0 ∧ ratio_third > 0 →
  total_sample = 240 →
  ratio_first = 5 ∧ ratio_second = 4 ∧ ratio_third = 3 →
  (ratio_second : ℚ) / (ratio_first + ratio_second + ratio_third : ℚ) * total_sample = 80 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_second_group_l1201_120184


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l1201_120143

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n : ℕ, b (n + 1) > b n) →  -- increasing sequence
  (∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d) →  -- arithmetic sequence
  (b 5 * b 6 = 21) →  -- given condition
  (b 4 * b 7 = -779 ∨ b 4 * b 7 = 21) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l1201_120143


namespace NUMINAMATH_CALUDE_linear_function_property_l1201_120138

-- Define a linear function
def LinearFunction (g : ℝ → ℝ) : Prop :=
  ∀ x y t : ℝ, g (x + t * (y - x)) = g x + t * (g y - g x)

-- State the theorem
theorem linear_function_property (g : ℝ → ℝ) 
  (h_linear : LinearFunction g)
  (h1 : g 8 - g 3 = 15)
  (h2 : g 4 - g 1 = 9) :
  g 10 - g 1 = 27 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l1201_120138


namespace NUMINAMATH_CALUDE_cost_of_skirt_l1201_120140

/-- Proves that the cost of each skirt is $15 --/
theorem cost_of_skirt (total_spent art_supplies_cost number_of_skirts : ℕ) 
  (h1 : total_spent = 50)
  (h2 : art_supplies_cost = 20)
  (h3 : number_of_skirts = 2) :
  (total_spent - art_supplies_cost) / number_of_skirts = 15 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_skirt_l1201_120140


namespace NUMINAMATH_CALUDE_dimes_borrowed_l1201_120195

/-- Represents the number of dimes Sam had initially -/
def initial_dimes : ℕ := 8

/-- Represents the number of dimes Sam has now -/
def remaining_dimes : ℕ := 4

/-- Represents the number of dimes Sam's sister borrowed -/
def borrowed_dimes : ℕ := initial_dimes - remaining_dimes

theorem dimes_borrowed :
  borrowed_dimes = initial_dimes - remaining_dimes :=
by sorry

end NUMINAMATH_CALUDE_dimes_borrowed_l1201_120195


namespace NUMINAMATH_CALUDE_min_empty_cells_after_move_l1201_120110

/-- Represents a 3D box with dimensions width, height, and depth -/
structure Box where
  width : Nat
  height : Nat
  depth : Nat

/-- Represents the movement of cockchafers to neighboring cells -/
def move_to_neighbor (box : Box) : Nat :=
  sorry

/-- Theorem: In a 3x5x7 box, after cockchafers move to neighboring cells,
    the minimum number of empty cells is 1 -/
theorem min_empty_cells_after_move (box : Box) 
  (h1 : box.width = 3) 
  (h2 : box.height = 5) 
  (h3 : box.depth = 7) : 
  move_to_neighbor box = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_empty_cells_after_move_l1201_120110


namespace NUMINAMATH_CALUDE_gasoline_price_quantity_adjustment_l1201_120111

theorem gasoline_price_quantity_adjustment 
  (original_price original_quantity : ℝ) 
  (price_increase : ℝ) 
  (spending_increase : ℝ) 
  (h1 : price_increase = 0.25) 
  (h2 : spending_increase = 0.10) : 
  let new_price := original_price * (1 + price_increase)
  let new_total_cost := original_price * original_quantity * (1 + spending_increase)
  let new_quantity := new_total_cost / new_price
  1 - (new_quantity / original_quantity) = 0.12 := by
sorry

end NUMINAMATH_CALUDE_gasoline_price_quantity_adjustment_l1201_120111


namespace NUMINAMATH_CALUDE_negation_at_most_four_l1201_120107

-- Define "at most four" for natural numbers
def at_most_four (n : ℕ) : Prop := n ≤ 4

-- Define "at least five" for natural numbers
def at_least_five (n : ℕ) : Prop := n ≥ 5

-- Theorem stating that the negation of "at most four" is equivalent to "at least five"
theorem negation_at_most_four (n : ℕ) : ¬(at_most_four n) ↔ at_least_five n := by
  sorry

end NUMINAMATH_CALUDE_negation_at_most_four_l1201_120107


namespace NUMINAMATH_CALUDE_total_calculators_l1201_120147

/-- Represents the number of calculators assembled by a person in a unit of time -/
structure AssemblyRate where
  calculators : ℕ
  time_units : ℕ

/-- The problem setup -/
def calculator_problem (erika nick sam : AssemblyRate) : Prop :=
  -- Erika assembles 3 calculators in the same time Nick assembles 2
  erika.calculators * nick.time_units = 3 * nick.calculators * erika.time_units ∧
  -- Nick assembles 1 calculator in the same time Sam assembles 3
  nick.calculators * sam.time_units = sam.calculators * nick.time_units ∧
  -- Erika's rate is 3 calculators per time unit
  erika.calculators = 3 ∧ erika.time_units = 1

/-- The theorem to prove -/
theorem total_calculators (erika nick sam : AssemblyRate) 
  (h : calculator_problem erika nick sam) : 
  9 * erika.time_units / erika.calculators * 
  (erika.calculators + nick.calculators * erika.time_units / nick.time_units + 
   sam.calculators * erika.time_units / sam.time_units) = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_calculators_l1201_120147


namespace NUMINAMATH_CALUDE_max_additional_plates_l1201_120186

/-- Represents the sets of letters for license plates --/
structure LicensePlateSets :=
  (first : Finset Char)
  (second : Finset Char)
  (third : Finset Char)

/-- The initial configuration of license plate sets --/
def initialSets : LicensePlateSets :=
  { first := {'B', 'F', 'G', 'T', 'Y'},
    second := {'E', 'U'},
    third := {'K', 'S', 'W'} }

/-- Calculates the number of possible license plates --/
def numPlates (sets : LicensePlateSets) : Nat :=
  sets.first.card * sets.second.card * sets.third.card

/-- Represents a new configuration after adding letters --/
structure NewConfig :=
  (sets : LicensePlateSets)
  (totalAdded : Nat)

/-- The theorem to be proved --/
theorem max_additional_plates :
  ∃ (newConfig : NewConfig),
    newConfig.totalAdded = 3 ∧
    ∀ (otherConfig : NewConfig),
      otherConfig.totalAdded = 3 →
      numPlates newConfig.sets - numPlates initialSets ≥
      numPlates otherConfig.sets - numPlates initialSets ∧
    numPlates newConfig.sets - numPlates initialSets = 50 :=
  sorry

end NUMINAMATH_CALUDE_max_additional_plates_l1201_120186


namespace NUMINAMATH_CALUDE_thursday_coffee_consumption_l1201_120129

/-- Represents the professor's coffee consumption model -/
structure CoffeeModel where
  k : ℝ
  coffee : ℝ → ℝ → ℝ
  wednesday_meetings : ℝ
  wednesday_sleep : ℝ
  wednesday_coffee : ℝ
  thursday_meetings : ℝ
  thursday_sleep : ℝ

/-- Theorem stating the professor's coffee consumption on Thursday -/
theorem thursday_coffee_consumption (model : CoffeeModel) 
  (h1 : model.coffee m h = model.k * m / h)
  (h2 : model.wednesday_coffee = model.coffee model.wednesday_meetings model.wednesday_sleep)
  (h3 : model.wednesday_meetings = 3)
  (h4 : model.wednesday_sleep = 8)
  (h5 : model.wednesday_coffee = 3)
  (h6 : model.thursday_meetings = 5)
  (h7 : model.thursday_sleep = 10) :
  model.coffee model.thursday_meetings model.thursday_sleep = 4 := by
  sorry

end NUMINAMATH_CALUDE_thursday_coffee_consumption_l1201_120129


namespace NUMINAMATH_CALUDE_debate_team_girls_l1201_120123

/-- The number of girls on a debate team -/
def girls_on_team (total_students : ℕ) (boys : ℕ) : ℕ :=
  total_students - boys

theorem debate_team_girls :
  let total_students := 7 * 9
  let boys := 31
  girls_on_team total_students boys = 32 := by
  sorry

#check debate_team_girls

end NUMINAMATH_CALUDE_debate_team_girls_l1201_120123


namespace NUMINAMATH_CALUDE_chord_equation_l1201_120180

/-- The equation of a line containing a chord of a circle, given specific conditions -/
theorem chord_equation (O : ℝ × ℝ) (r : ℝ) (P : ℝ × ℝ) :
  O = (-1, 0) →
  r = 5 →
  P = (2, -3) →
  (∃ A B : ℝ × ℝ, 
    (A.1 - O.1)^2 + (A.2 - O.2)^2 = r^2 ∧
    (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2 ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) →
  ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ x - y - 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l1201_120180


namespace NUMINAMATH_CALUDE_digit_extraction_l1201_120153

theorem digit_extraction (a b c : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9) :
  let S := 100 * a + 10 * b + c
  (S / 100 = a) ∧ ((S / 10) % 10 = b) ∧ (S % 10 = c) := by
  sorry

end NUMINAMATH_CALUDE_digit_extraction_l1201_120153


namespace NUMINAMATH_CALUDE_mrs_hilt_bugs_l1201_120188

theorem mrs_hilt_bugs (total_flowers : ℝ) (flowers_per_bug : ℝ) (h1 : total_flowers = 3.0) (h2 : flowers_per_bug = 1.5) :
  total_flowers / flowers_per_bug = 2 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_bugs_l1201_120188


namespace NUMINAMATH_CALUDE_distance_point_to_line_l1201_120167

def point : ℝ × ℝ × ℝ := (0, 3, -1)
def linePoint1 : ℝ × ℝ × ℝ := (1, -2, 0)
def linePoint2 : ℝ × ℝ × ℝ := (3, 1, 4)

def distancePointToLine (p : ℝ × ℝ × ℝ) (l1 l2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_point_to_line :
  distancePointToLine point linePoint1 linePoint2 = Real.sqrt 22058 / 29 := by
  sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l1201_120167


namespace NUMINAMATH_CALUDE_car_truck_difference_l1201_120196

theorem car_truck_difference (total_vehicles : ℕ) (trucks : ℕ) 
  (h1 : total_vehicles = 69) 
  (h2 : trucks = 21) : 
  total_vehicles - trucks - trucks = 27 := by
  sorry

end NUMINAMATH_CALUDE_car_truck_difference_l1201_120196


namespace NUMINAMATH_CALUDE_cake_cost_is_correct_l1201_120178

/-- The cost of a piece of cake in dollars -/
def cake_cost : ℚ := 7

/-- The cost of a cup of coffee in dollars -/
def coffee_cost : ℚ := 4

/-- The cost of a bowl of ice cream in dollars -/
def ice_cream_cost : ℚ := 3

/-- The total cost for Mell and her two friends in dollars -/
def total_cost : ℚ := 51

/-- Theorem stating that the cake cost is correct given the conditions -/
theorem cake_cost_is_correct :
  cake_cost = 7 ∧
  coffee_cost = 4 ∧
  ice_cream_cost = 3 ∧
  total_cost = 51 ∧
  (2 * coffee_cost + cake_cost) + 2 * (2 * coffee_cost + cake_cost + ice_cream_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_cake_cost_is_correct_l1201_120178


namespace NUMINAMATH_CALUDE_students_behind_yoongi_l1201_120194

/-- Given a line of students, prove the number standing behind a specific student. -/
theorem students_behind_yoongi (total_students : ℕ) (jungkook_position : ℕ) (yoongi_position : ℕ) :
  total_students = 20 →
  jungkook_position = 3 →
  yoongi_position = jungkook_position - 1 →
  total_students - yoongi_position = 18 := by
  sorry

end NUMINAMATH_CALUDE_students_behind_yoongi_l1201_120194


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l1201_120150

theorem eighth_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, S n = n^2) → a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l1201_120150


namespace NUMINAMATH_CALUDE_frog_jump_coordinates_l1201_120169

def initial_point : ℝ × ℝ := (-1, 0)
def right_jump : ℝ := 2
def up_jump : ℝ := 2

def final_point (p : ℝ × ℝ) (right : ℝ) (up : ℝ) : ℝ × ℝ :=
  (p.1 + right, p.2 + up)

theorem frog_jump_coordinates :
  final_point initial_point right_jump up_jump = (1, 2) := by sorry

end NUMINAMATH_CALUDE_frog_jump_coordinates_l1201_120169


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1201_120149

theorem scientific_notation_equivalence : 
  ∃ (x : ℝ) (n : ℤ), 11580000 = x * (10 : ℝ) ^ n ∧ 1 ≤ x ∧ x < 10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1201_120149


namespace NUMINAMATH_CALUDE_largest_number_of_three_l1201_120154

theorem largest_number_of_three (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (sum_prod_eq : a * b + a * c + b * c = -10)
  (prod_eq : a * b * c = -18) :
  max a (max b c) = -1 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_of_three_l1201_120154


namespace NUMINAMATH_CALUDE_distance_to_CD_l1201_120139

/-- A square with semi-circle arcs -/
structure SquareWithArcs (s : ℝ) where
  -- Square ABCD
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Ensure it's a square with side length s
  square_side : dist A B = s ∧ dist B C = s ∧ dist C D = s ∧ dist D A = s
  -- Semi-circle arcs
  arc_A : Set (ℝ × ℝ)
  arc_B : Set (ℝ × ℝ)
  -- Ensure arcs have correct radii and centers
  arc_A_def : arc_A = {p : ℝ × ℝ | dist p A = s/2 ∧ p.1 ≥ A.1 ∧ p.2 ≤ C.2}
  arc_B_def : arc_B = {p : ℝ × ℝ | dist p B = s/2 ∧ p.1 ≤ B.1 ∧ p.2 ≤ C.2}
  -- Intersection point X
  X : ℝ × ℝ
  X_def : X ∈ arc_A ∧ X ∈ arc_B

/-- The main theorem -/
theorem distance_to_CD (s : ℝ) (h : s > 0) (sq : SquareWithArcs s) :
  dist sq.X (sq.C.1, sq.X.2) = s :=
sorry

end NUMINAMATH_CALUDE_distance_to_CD_l1201_120139


namespace NUMINAMATH_CALUDE_line_intersection_regions_l1201_120115

theorem line_intersection_regions (h s : ℕ+) : 
  (s + 1) * (s + 2 * h) = 3984 ↔ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_regions_l1201_120115


namespace NUMINAMATH_CALUDE_resale_value_drops_below_target_in_four_years_l1201_120192

def initial_price : ℝ := 625000
def first_year_depreciation : ℝ := 0.20
def subsequent_depreciation : ℝ := 0.08
def target_value : ℝ := 400000

def resale_value (n : ℕ) : ℝ :=
  if n = 0 then initial_price
  else if n = 1 then initial_price * (1 - first_year_depreciation)
  else initial_price * (1 - first_year_depreciation) * (1 - subsequent_depreciation) ^ (n - 1)

theorem resale_value_drops_below_target_in_four_years :
  resale_value 4 < target_value ∧ ∀ k : ℕ, k < 4 → resale_value k ≥ target_value :=
sorry

end NUMINAMATH_CALUDE_resale_value_drops_below_target_in_four_years_l1201_120192


namespace NUMINAMATH_CALUDE_max_value_expression_l1201_120113

theorem max_value_expression (a b c d : ℝ) 
  (ha : -7 ≤ a ∧ a ≤ 7) 
  (hb : -7 ≤ b ∧ b ≤ 7) 
  (hc : -7 ≤ c ∧ c ≤ 7) 
  (hd : -7 ≤ d ∧ d ≤ 7) : 
  (∀ a' b' c' d' : ℝ, 
    -7 ≤ a' ∧ a' ≤ 7 → 
    -7 ≤ b' ∧ b' ≤ 7 → 
    -7 ≤ c' ∧ c' ≤ 7 → 
    -7 ≤ d' ∧ d' ≤ 7 → 
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' ≤ 210) ∧
  (∃ a' b' c' d' : ℝ, 
    -7 ≤ a' ∧ a' ≤ 7 ∧
    -7 ≤ b' ∧ b' ≤ 7 ∧
    -7 ≤ c' ∧ c' ≤ 7 ∧
    -7 ≤ d' ∧ d' ≤ 7 ∧
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' = 210) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1201_120113


namespace NUMINAMATH_CALUDE_sin_graph_transformation_l1201_120109

theorem sin_graph_transformation (x : ℝ) :
  let f (x : ℝ) := 2 * Real.sin x
  let g (x : ℝ) := 2 * Real.sin (x / 3 + π / 6)
  let h (x : ℝ) := f (x + π / 6)
  g x = h (x / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_graph_transformation_l1201_120109


namespace NUMINAMATH_CALUDE_base4_calculation_l1201_120161

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ := sorry

/-- Multiplies two base 4 numbers --/
def multiplyBase4 (a b : ℕ) : ℕ := sorry

/-- Divides two base 4 numbers --/
def divideBase4 (a b : ℕ) : ℕ := sorry

/-- Subtracts two base 4 numbers --/
def subtractBase4 (a b : ℕ) : ℕ := sorry

theorem base4_calculation : 
  let a := 230
  let b := 21
  let c := 2
  let d := 12
  let e := 3
  subtractBase4 (divideBase4 (multiplyBase4 a b) c) (multiplyBase4 d e) = 3222 := by
  sorry

end NUMINAMATH_CALUDE_base4_calculation_l1201_120161


namespace NUMINAMATH_CALUDE_hotel_rate_proof_l1201_120137

/-- The flat rate for the first night in a hotel. -/
def flat_rate : ℝ := 80

/-- The additional fee for each subsequent night. -/
def additional_fee : ℝ := 40

/-- The cost for a stay of n nights. -/
def cost (n : ℕ) : ℝ := flat_rate + additional_fee * (n - 1)

theorem hotel_rate_proof :
  (cost 4 = 200) ∧ (cost 7 = 320) → flat_rate = 80 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rate_proof_l1201_120137


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l1201_120130

theorem complex_magnitude_equation (x : ℝ) (h1 : x > 0) :
  Complex.abs (3 + 4 * x * Complex.I) = 5 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l1201_120130


namespace NUMINAMATH_CALUDE_pond_radius_l1201_120182

/-- The radius of a circular pond with a diameter of 14 meters is 7 meters. -/
theorem pond_radius (diameter : ℝ) (h : diameter = 14) : diameter / 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_pond_radius_l1201_120182


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1201_120164

/-- A monotonically decreasing geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 0 < q ∧ q < 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_a3 : a 3 = 1)
  (h_sum : a 2 + a 4 = 5/2) :
  a 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1201_120164


namespace NUMINAMATH_CALUDE_inequality_solution_l1201_120133

theorem inequality_solution (x : ℝ) : 
  -1 < (x^2 - 16*x + 15) / (x^2 - 4*x + 5) ∧ 
  (x^2 - 16*x + 15) / (x^2 - 4*x + 5) < 1 ↔ 
  x < 5/2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1201_120133


namespace NUMINAMATH_CALUDE_palm_tree_count_l1201_120189

theorem palm_tree_count (desert forest : ℕ) 
  (h1 : desert = (2 : ℚ) / 5 * forest)  -- Desert has 2/5 the trees of the forest
  (h2 : desert + forest = 7000)         -- Total trees in both locations
  : forest = 5000 := by
  sorry

end NUMINAMATH_CALUDE_palm_tree_count_l1201_120189


namespace NUMINAMATH_CALUDE_pizza_ratio_proof_l1201_120170

theorem pizza_ratio_proof (total_slices : ℕ) (calories_per_slice : ℕ) (calories_eaten : ℕ) : 
  total_slices = 8 → 
  calories_per_slice = 300 → 
  calories_eaten = 1200 → 
  (calories_eaten / calories_per_slice : ℚ) / total_slices = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_ratio_proof_l1201_120170


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_factors_of_30_l1201_120177

def factors_of_30 : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

theorem sum_of_reciprocals_of_factors_of_30 :
  (factors_of_30.map (λ x => (1 : ℚ) / x)).sum = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_factors_of_30_l1201_120177


namespace NUMINAMATH_CALUDE_calculate_expression_l1201_120116

theorem calculate_expression : 
  (Real.sqrt 2 / 2) * (2 * Real.sqrt 12 / (4 * Real.sqrt (1/8)) - 3 * Real.sqrt 48) = 
  2 * Real.sqrt 3 - 6 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_calculate_expression_l1201_120116


namespace NUMINAMATH_CALUDE_line_slope_through_origin_and_one_neg_one_l1201_120193

/-- The slope of a line passing through points (0,0) and (1,-1) is -1. -/
theorem line_slope_through_origin_and_one_neg_one : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, -1)
  (B.2 - A.2) / (B.1 - A.1) = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_through_origin_and_one_neg_one_l1201_120193


namespace NUMINAMATH_CALUDE_jordan_danielle_roses_l1201_120171

def roses_remaining (initial : ℕ) (additional : ℕ) : ℕ :=
  let total := initial + additional
  let after_first_day := total / 2
  let after_second_day := after_first_day / 2
  after_second_day

theorem jordan_danielle_roses : roses_remaining 24 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jordan_danielle_roses_l1201_120171


namespace NUMINAMATH_CALUDE_coefficient_of_b_l1201_120145

theorem coefficient_of_b (a b : ℝ) (h1 : 7 * a = b) (h2 : b = 15) (h3 : 42 * a * b = 675) :
  42 * a = 45 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_b_l1201_120145


namespace NUMINAMATH_CALUDE_y₁_y₂_friendly_l1201_120144

/-- Two functions are friendly if their difference is between -1 and 1 for all x in (0,1) -/
def friendly (f g : ℝ → ℝ) : Prop :=
  ∀ x, 0 < x → x < 1 → -1 < f x - g x ∧ f x - g x < 1

/-- The function y₁(x) = x² - 1 -/
def y₁ (x : ℝ) : ℝ := x^2 - 1

/-- The function y₂(x) = 2x - 1 -/
def y₂ (x : ℝ) : ℝ := 2*x - 1

/-- Theorem: y₁ and y₂ are friendly functions -/
theorem y₁_y₂_friendly : friendly y₁ y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_y₂_friendly_l1201_120144


namespace NUMINAMATH_CALUDE_book_words_per_page_l1201_120102

theorem book_words_per_page :
  ∀ (words_per_page : ℕ),
    words_per_page ≤ 120 →
    (150 * words_per_page) % 221 = 210 →
    words_per_page = 98 := by
  sorry

end NUMINAMATH_CALUDE_book_words_per_page_l1201_120102


namespace NUMINAMATH_CALUDE_simplify_expression_l1201_120121

theorem simplify_expression (a b c : ℝ) (h : (c - a) / (c - b) = 1) :
  (5 * b - 2 * a) / (c - a) = 3 * a / (c - a) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1201_120121


namespace NUMINAMATH_CALUDE_line_equation_fourth_quadrant_triangle_l1201_120114

/-- Given a line passing through (-b, 0) and cutting a triangle with area T in the fourth quadrant,
    prove that its equation is 2Tx - b²y - 2bT = 0 --/
theorem line_equation_fourth_quadrant_triangle (b T : ℝ) (h₁ : b > 0) (h₂ : T > 0) : 
  ∃ (m c : ℝ), ∀ (x y : ℝ),
    (x = -b ∧ y = 0) ∨ (x ≥ 0 ∧ y ≤ 0 ∧ y = m * x + c) →
    (1/2 * b * (-y)) = T →
    2 * T * x - b^2 * y - 2 * b * T = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_fourth_quadrant_triangle_l1201_120114


namespace NUMINAMATH_CALUDE_road_repaving_l1201_120100

/-- Proves that the number of inches repaved before today is 4133,
    given the total repaved and the amount repaved today. -/
theorem road_repaving (total_repaved : ℕ) (repaved_today : ℕ)
    (h1 : total_repaved = 4938)
    (h2 : repaved_today = 805) :
    total_repaved - repaved_today = 4133 := by
  sorry

end NUMINAMATH_CALUDE_road_repaving_l1201_120100


namespace NUMINAMATH_CALUDE_unique_digit_product_solution_l1201_120152

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem unique_digit_product_solution :
  ∃! n : ℕ, digit_product n = n^2 - 10*n - 22 :=
sorry

end NUMINAMATH_CALUDE_unique_digit_product_solution_l1201_120152


namespace NUMINAMATH_CALUDE_stating_one_empty_neighborhood_probability_l1201_120118

/-- The number of neighborhoods --/
def num_neighborhoods : ℕ := 3

/-- The number of staff members --/
def num_staff : ℕ := 4

/-- The probability of exactly one neighborhood not being assigned any staff members --/
def probability_one_empty : ℚ := 14/27

/-- 
Theorem stating that the probability of exactly one neighborhood out of three 
not being assigned any staff members, when four staff members are independently 
assigned to the neighborhoods, is 14/27.
--/
theorem one_empty_neighborhood_probability : 
  (num_neighborhoods = 3 ∧ num_staff = 4) → 
  probability_one_empty = 14/27 := by
  sorry

end NUMINAMATH_CALUDE_stating_one_empty_neighborhood_probability_l1201_120118


namespace NUMINAMATH_CALUDE_cool_parents_problem_l1201_120120

theorem cool_parents_problem (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ)
  (h1 : total = 40)
  (h2 : cool_dads = 18)
  (h3 : cool_moms = 22)
  (h4 : both_cool = 10) :
  total - (cool_dads + cool_moms - both_cool) = 10 := by
  sorry

end NUMINAMATH_CALUDE_cool_parents_problem_l1201_120120


namespace NUMINAMATH_CALUDE_meet_time_opposite_directions_l1201_120155

/-- Represents an athlete running on a track -/
structure Athlete where
  lap_time : ℝ
  speed : ℝ

/-- Represents a closed track -/
structure Track where
  length : ℝ

/-- The scenario of two athletes running on a track -/
def running_scenario (t : Track) (a1 a2 : Athlete) : Prop :=
  a1.speed = t.length / a1.lap_time ∧
  a2.speed = t.length / a2.lap_time ∧
  a2.lap_time = a1.lap_time + 5 ∧
  30 * a1.speed - 30 * a2.speed = t.length

theorem meet_time_opposite_directions 
  (t : Track) (a1 a2 : Athlete) 
  (h : running_scenario t a1 a2) : 
  t.length / (a1.speed + a2.speed) = 6 := by
  sorry


end NUMINAMATH_CALUDE_meet_time_opposite_directions_l1201_120155


namespace NUMINAMATH_CALUDE_shared_vertex_angle_measure_l1201_120181

/-- The measure of the angle at the common vertex formed by a side of an equilateral triangle
    and a side of a regular pentagon, both inscribed in a circle. -/
def common_vertex_angle : ℝ := 24

/-- A regular pentagon inscribed in a circle -/
structure RegularPentagonInCircle where
  -- Add necessary fields

/-- An equilateral triangle inscribed in a circle -/
structure EquilateralTriangleInCircle where
  -- Add necessary fields

/-- Configuration of a regular pentagon and an equilateral triangle inscribed in a circle
    with a shared vertex -/
structure SharedVertexConfiguration where
  pentagon : RegularPentagonInCircle
  triangle : EquilateralTriangleInCircle
  -- Add field to represent the shared vertex

theorem shared_vertex_angle_measure (config : SharedVertexConfiguration) :
  common_vertex_angle = 24 := by
  sorry

end NUMINAMATH_CALUDE_shared_vertex_angle_measure_l1201_120181


namespace NUMINAMATH_CALUDE_license_plate_combinations_l1201_120117

/-- The number of ways to choose 2 items from n items -/
def choose (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions on the license plate -/
def letter_positions : ℕ := 4

/-- The number of digit positions on the license plate -/
def digit_positions : ℕ := 3

/-- The maximum starting digit to allow for 3 consecutive increasing digits -/
def max_start_digit : ℕ := 7

theorem license_plate_combinations :
  (choose alphabet_size 2) * (choose letter_positions 2) * (max_start_digit + 1) = 15600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l1201_120117


namespace NUMINAMATH_CALUDE_max_value_x_l1201_120134

theorem max_value_x : 
  ∃ (x_max : ℝ), 
    (∀ x : ℝ, ((5*x - 25)/(4*x - 5))^2 + ((5*x - 25)/(4*x - 5)) = 20 → x ≤ x_max) ∧
    ((5*x_max - 25)/(4*x_max - 5))^2 + ((5*x_max - 25)/(4*x_max - 5)) = 20 ∧
    x_max = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_l1201_120134


namespace NUMINAMATH_CALUDE_smallest_number_is_negative_one_l1201_120198

theorem smallest_number_is_negative_one :
  let numbers : Finset ℝ := {0, 1/3, -1, Real.sqrt 2}
  ∀ x ∈ numbers, -1 ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_is_negative_one_l1201_120198


namespace NUMINAMATH_CALUDE_exists_N_average_fifteen_l1201_120156

theorem exists_N_average_fifteen : 
  ∃ N : ℝ, 15 < N ∧ N < 25 ∧ (8 + 14 + N) / 3 = 15 := by
sorry

end NUMINAMATH_CALUDE_exists_N_average_fifteen_l1201_120156


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1201_120165

theorem constant_term_expansion (n : ℕ) : 
  (∃ k : ℕ, k = n / 3 ∧ Nat.choose n k = 15) ↔ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1201_120165


namespace NUMINAMATH_CALUDE_ians_jogging_laps_l1201_120126

/-- Given information about Ian's jogging routine, calculate the number of laps he does every night -/
theorem ians_jogging_laps 
  (lap_length : ℝ)
  (feet_per_calorie : ℝ)
  (total_calories : ℝ)
  (total_days : ℝ)
  (h1 : lap_length = 100)
  (h2 : feet_per_calorie = 25)
  (h3 : total_calories = 100)
  (h4 : total_days = 5)
  : (total_calories * feet_per_calorie / total_days) / lap_length = 5 := by
  sorry

end NUMINAMATH_CALUDE_ians_jogging_laps_l1201_120126


namespace NUMINAMATH_CALUDE_common_terms_theorem_l1201_120131

def a (n : ℕ) : ℤ := 3 * n - 19
def b (n : ℕ) : ℕ := 2^n

-- c_n is the nth common term of sequences a and b in ascending order
def c (n : ℕ) : ℕ := 2^(2*n - 1)

theorem common_terms_theorem (n : ℕ) :
  ∃ (m k : ℕ), a m = b k ∧ c n = b k ∧ 
  (∀ (i j : ℕ), i < m ∧ j < k → a i ≠ b j) ∧
  (∀ (i j : ℕ), a i = b j → i ≥ m ∨ j ≥ k) :=
sorry

end NUMINAMATH_CALUDE_common_terms_theorem_l1201_120131


namespace NUMINAMATH_CALUDE_average_height_l1201_120127

def heights_problem (h₁ h₂ h₃ h₄ : ℝ) : Prop :=
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ ∧
  h₂ - h₁ = 2 ∧
  h₃ - h₂ = 2 ∧
  h₄ - h₃ = 6 ∧
  h₄ = 83

theorem average_height (h₁ h₂ h₃ h₄ : ℝ) 
  (hproblem : heights_problem h₁ h₂ h₃ h₄) : 
  (h₁ + h₂ + h₃ + h₄) / 4 = 77 := by
  sorry

end NUMINAMATH_CALUDE_average_height_l1201_120127
