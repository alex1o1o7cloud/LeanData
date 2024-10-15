import Mathlib

namespace NUMINAMATH_CALUDE_abc_sum_mod_five_l54_5454

theorem abc_sum_mod_five (a b c : ℕ) : 
  0 < a ∧ a < 5 ∧
  0 < b ∧ b < 5 ∧
  0 < c ∧ c < 5 ∧
  (a * b * c) % 5 = 1 ∧
  (4 * c) % 5 = 3 ∧
  (3 * b) % 5 = (2 + b) % 5 →
  (a + b + c) % 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_mod_five_l54_5454


namespace NUMINAMATH_CALUDE_stating_equal_probability_for_all_methods_l54_5492

/-- Represents a sampling method -/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- The total number of components -/
def total_components : ℕ := 100

/-- The number of items to be sampled -/
def sample_size : ℕ := 20

/-- The number of first-grade items -/
def first_grade : ℕ := 20

/-- The number of second-grade items -/
def second_grade : ℕ := 30

/-- The number of third-grade items -/
def third_grade : ℕ := 50

/-- The probability of selecting any individual component -/
def selection_probability : ℚ := 1 / 5

/-- 
  Theorem stating that for all sampling methods, 
  the probability of selecting any individual component is 1/5
-/
theorem equal_probability_for_all_methods (method : SamplingMethod) : 
  (selection_probability : ℚ) = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_stating_equal_probability_for_all_methods_l54_5492


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l54_5406

theorem min_value_of_expression (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 5 * Real.sqrt 6 / 3 :=
by sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) = 5 * Real.sqrt 6 / 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_achievable_l54_5406


namespace NUMINAMATH_CALUDE_brendan_total_wins_l54_5441

/-- Represents the number of matches won in each round of the kickboxing competition -/
structure KickboxingResults where
  round1_wins : Nat
  round2_wins : Nat
  round3_wins : Nat
  round4_wins : Nat

/-- Calculates the total number of matches won across all rounds -/
def total_wins (results : KickboxingResults) : Nat :=
  results.round1_wins + results.round2_wins + results.round3_wins + results.round4_wins

/-- Theorem stating that Brendan's total wins in the kickboxing competition is 18 -/
theorem brendan_total_wins :
  ∃ (results : KickboxingResults),
    results.round1_wins = 6 ∧
    results.round2_wins = 4 ∧
    results.round3_wins = 3 ∧
    results.round4_wins = 5 ∧
    total_wins results = 18 := by
  sorry

end NUMINAMATH_CALUDE_brendan_total_wins_l54_5441


namespace NUMINAMATH_CALUDE_complex_equation_sum_l54_5427

theorem complex_equation_sum (a b : ℝ) :
  (a + 4 * Complex.I) * Complex.I = b + Complex.I →
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l54_5427


namespace NUMINAMATH_CALUDE_necessary_condition_equality_l54_5418

theorem necessary_condition_equality (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_equality_l54_5418


namespace NUMINAMATH_CALUDE_mary_picked_12kg_l54_5457

/-- Given three people picking chestnuts, prove that one person picked 12 kg. -/
theorem mary_picked_12kg (peter lucy mary : ℕ) : 
  mary = 2 * peter →  -- Mary picked twice as much as Peter
  lucy = peter + 2 →  -- Lucy picked 2 kg more than Peter
  peter + mary + lucy = 26 →  -- Total amount picked is 26 kg
  mary = 12 := by
sorry

end NUMINAMATH_CALUDE_mary_picked_12kg_l54_5457


namespace NUMINAMATH_CALUDE_kevins_toad_feeding_l54_5481

/-- Given Kevin's toad feeding scenario, prove the number of worms per toad. -/
theorem kevins_toad_feeding (num_toads : ℕ) (minutes_per_worm : ℕ) (total_hours : ℕ) 
  (h1 : num_toads = 8)
  (h2 : minutes_per_worm = 15)
  (h3 : total_hours = 6) :
  (total_hours * 60) / minutes_per_worm / num_toads = 3 := by
  sorry

#check kevins_toad_feeding

end NUMINAMATH_CALUDE_kevins_toad_feeding_l54_5481


namespace NUMINAMATH_CALUDE_probability_endpoints_of_edge_is_four_fifths_l54_5415

/-- A regular octahedron -/
structure RegularOctahedron where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  vertex_count : vertices.card = 6
  edge_count_per_vertex : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 4

/-- The probability of choosing two vertices that are endpoints of an edge -/
def probability_endpoints_of_edge (o : RegularOctahedron) : ℚ :=
  (4 : ℚ) / 5

/-- Theorem: The probability of randomly choosing two vertices of a regular octahedron 
    that are endpoints of an edge is 4/5 -/
theorem probability_endpoints_of_edge_is_four_fifths (o : RegularOctahedron) :
  probability_endpoints_of_edge o = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_endpoints_of_edge_is_four_fifths_l54_5415


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l54_5433

/-- Represents the stratified sampling scenario -/
structure SamplingScenario where
  total_members : ℕ
  boys : ℕ
  girls : ℕ
  sample_size : ℕ

/-- The specific scenario from the problem -/
def track_team : SamplingScenario :=
  { total_members := 42
  , boys := 28
  , girls := 14
  , sample_size := 6 }

/-- The probability of an individual being selected -/
def selection_probability (s : SamplingScenario) : ℚ :=
  s.sample_size / s.total_members

/-- The number of boys selected in stratified sampling -/
def boys_selected (s : SamplingScenario) : ℕ :=
  (s.sample_size * s.boys) / s.total_members

/-- The number of girls selected in stratified sampling -/
def girls_selected (s : SamplingScenario) : ℕ :=
  (s.sample_size * s.girls) / s.total_members

theorem stratified_sampling_theorem (s : SamplingScenario) :
  s = track_team →
  selection_probability s = 1/7 ∧
  boys_selected s = 4 ∧
  girls_selected s = 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l54_5433


namespace NUMINAMATH_CALUDE_average_not_equal_given_l54_5442

def numbers : List ℝ := [1200, 1300, 1400, 1510, 1530, 1200]
def given_average : ℝ := 1380

theorem average_not_equal_given : (numbers.sum / numbers.length) ≠ given_average := by
  sorry

end NUMINAMATH_CALUDE_average_not_equal_given_l54_5442


namespace NUMINAMATH_CALUDE_largest_possible_s_value_l54_5423

theorem largest_possible_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3) : 
  (((r - 2) * 180 : ℚ) / r) / (((s - 2) * 180 : ℚ) / s) = 101 / 97 → s ≤ 100 := by
  sorry

end NUMINAMATH_CALUDE_largest_possible_s_value_l54_5423


namespace NUMINAMATH_CALUDE_sum_of_999_and_999_l54_5458

theorem sum_of_999_and_999 : 999 + 999 = 1998 := by sorry

end NUMINAMATH_CALUDE_sum_of_999_and_999_l54_5458


namespace NUMINAMATH_CALUDE_sweets_distribution_l54_5483

theorem sweets_distribution (total : ℕ) (june_sweets : ℕ) : 
  total = 90 →
  (3 : ℕ) * june_sweets = 4 * ((2 : ℕ) * june_sweets / 3) → 
  (1 : ℕ) * june_sweets / 2 + (3 : ℕ) * june_sweets / 4 + june_sweets = total →
  june_sweets = 40 := by
sorry

end NUMINAMATH_CALUDE_sweets_distribution_l54_5483


namespace NUMINAMATH_CALUDE_amy_biking_distance_l54_5496

theorem amy_biking_distance (yesterday_distance today_distance : ℝ) : 
  yesterday_distance = 12 →
  yesterday_distance + today_distance = 33 →
  today_distance < 2 * yesterday_distance →
  2 * yesterday_distance - today_distance = 3 :=
by sorry

end NUMINAMATH_CALUDE_amy_biking_distance_l54_5496


namespace NUMINAMATH_CALUDE_poppy_seed_count_l54_5447

def total_slices : ℕ := 58

theorem poppy_seed_count (x : ℕ) 
  (h1 : x ≤ total_slices)
  (h2 : Nat.choose x 3 = Nat.choose (total_slices - x) 2 * x) :
  total_slices - x = 21 := by
  sorry

end NUMINAMATH_CALUDE_poppy_seed_count_l54_5447


namespace NUMINAMATH_CALUDE_point_inside_circle_l54_5408

/-- Given a line ax + by + 1 = 0 and a circle x² + y² = 1 that are separate,
    prove that the point P(a, b) is inside the circle. -/
theorem point_inside_circle (a b : ℝ) 
  (h_separate : (1 : ℝ) / Real.sqrt (a^2 + b^2) > 1) : 
  a^2 + b^2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_l54_5408


namespace NUMINAMATH_CALUDE_infinite_divisible_sequence_l54_5422

theorem infinite_divisible_sequence : 
  ∃ (f : ℕ → ℕ), 
    (∀ k, f k > 0) ∧ 
    (∀ k, k < k.succ → f k < f k.succ) ∧ 
    (∀ k, (2 ^ (f k) + 3 ^ (f k)) % (f k)^2 = 0) :=
sorry

end NUMINAMATH_CALUDE_infinite_divisible_sequence_l54_5422


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l54_5472

/-- Represents a dodecahedron -/
structure Dodecahedron where
  vertices : Finset (Fin 20)
  faces : Finset (Fin 12)
  is_pentagonal : faces → Prop
  vertex_face_incidence : vertices → faces → Prop
  three_faces_per_vertex : ∀ v : vertices, ∃! (f1 f2 f3 : faces), 
    vertex_face_incidence v f1 ∧ vertex_face_incidence v f2 ∧ vertex_face_incidence v f3 ∧ f1 ≠ f2 ∧ f2 ≠ f3 ∧ f1 ≠ f3

/-- An interior diagonal in a dodecahedron -/
def interior_diagonal (d : Dodecahedron) (v1 v2 : d.vertices) : Prop :=
  v1 ≠ v2 ∧ ∀ f : d.faces, ¬(d.vertex_face_incidence v1 f ∧ d.vertex_face_incidence v2 f)

/-- The number of interior diagonals in a dodecahedron -/
def num_interior_diagonals (d : Dodecahedron) : ℕ :=
  (d.vertices.card * (d.vertices.card - 3)) / 2

/-- Theorem stating that a dodecahedron has 170 interior diagonals -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) : 
  num_interior_diagonals d = 170 := by sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l54_5472


namespace NUMINAMATH_CALUDE_inequality_solution_set_l54_5479

theorem inequality_solution_set :
  {x : ℝ | (1/2: ℝ)^x ≤ (1/2 : ℝ)^(x+1) + 1} = {x : ℝ | x ≥ -1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l54_5479


namespace NUMINAMATH_CALUDE_log2_odd_and_increasing_l54_5495

open Real

-- Define the function f(x) = log₂ x
noncomputable def f (x : ℝ) : ℝ := log x / log 2

-- Theorem statement
theorem log2_odd_and_increasing :
  (∀ x > 0, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_log2_odd_and_increasing_l54_5495


namespace NUMINAMATH_CALUDE_total_books_l54_5419

theorem total_books (tim_books mike_books : ℕ) 
  (h1 : tim_books = 22) 
  (h2 : mike_books = 20) : 
  tim_books + mike_books = 42 := by
sorry

end NUMINAMATH_CALUDE_total_books_l54_5419


namespace NUMINAMATH_CALUDE_rhind_papyrus_fraction_decomposition_l54_5439

theorem rhind_papyrus_fraction_decomposition : 
  2 / 73 = 1 / 60 + 1 / 219 + 1 / 292 + 1 / 365 := by
  sorry

end NUMINAMATH_CALUDE_rhind_papyrus_fraction_decomposition_l54_5439


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l54_5469

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_inequality :
  (¬ ∀ x : ℝ, x^2 + 2 > 2*x) ↔ (∃ x : ℝ, x^2 + 2 ≤ 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_inequality_l54_5469


namespace NUMINAMATH_CALUDE_cricket_team_size_l54_5488

-- Define the number of team members
variable (n : ℕ)

-- Define the average age of the team
def team_average : ℝ := 25

-- Define the wicket keeper's age
def wicket_keeper_age : ℝ := team_average + 3

-- Define the average age of remaining players after excluding two members
def remaining_average : ℝ := team_average - 1

-- Define the total age of the team
def total_age : ℝ := n * team_average

-- Define the total age of remaining players
def remaining_total_age : ℝ := (n - 2) * remaining_average

-- Define the total age of the two excluded members
def excluded_total_age : ℝ := wicket_keeper_age + team_average

-- Theorem stating that the number of team members is 5
theorem cricket_team_size : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l54_5488


namespace NUMINAMATH_CALUDE_intersection_points_range_l54_5438

/-- The curve equation x^2 + (y+3)^2 = 4 -/
def curve (x y : ℝ) : Prop := x^2 + (y+3)^2 = 4

/-- The line equation y = k(x-2) -/
def line (k x y : ℝ) : Prop := y = k*(x-2)

/-- The theorem stating the range of k for which the curve and line have two distinct intersection points -/
theorem intersection_points_range :
  ∀ k : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    curve x₁ y₁ ∧ curve x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂ ∧ 
    y₁ ≥ -3 ∧ y₂ ≥ -3) ↔ 
  (5/12 < k ∧ k ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_range_l54_5438


namespace NUMINAMATH_CALUDE_units_digit_of_product_l54_5425

theorem units_digit_of_product (a b c : ℕ) : 
  (4^503 * 3^401 * 15^402) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l54_5425


namespace NUMINAMATH_CALUDE_valid_choices_count_l54_5412

/-- Represents a point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle in the plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a straight line in the plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- The set of 9 points created by the intersection of two lines and two circles -/
def intersection_points : Finset Point := sorry

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if three points lie on the same circle -/
def on_same_circle (p q r : Point) (c1 c2 : Circle) : Prop := sorry

/-- The number of ways to choose 4 points from the intersection points
    such that no 3 of them are collinear or on the same circle -/
def valid_choices : ℕ := sorry

theorem valid_choices_count :
  valid_choices = 114 :=
sorry

end NUMINAMATH_CALUDE_valid_choices_count_l54_5412


namespace NUMINAMATH_CALUDE_min_abs_phi_l54_5431

theorem min_abs_phi (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) : 
  (∀ x, A * Real.sin (ω * x + φ) = A * Real.sin (ω * (x + π) + φ)) →
  (∀ x, A * Real.sin (ω * x + φ) = A * Real.sin (ω * (2 * π / 3 - x) + φ)) →
  ∃ k : ℤ, |φ + k * π| = π / 6 :=
sorry

end NUMINAMATH_CALUDE_min_abs_phi_l54_5431


namespace NUMINAMATH_CALUDE_problem_solution_l54_5486

theorem problem_solution :
  ∀ x y : ℕ,
    x > 0 → y > 0 →
    x < 15 → y < 15 →
    x + y + x * y = 49 →
    x + y = 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l54_5486


namespace NUMINAMATH_CALUDE_negation_existence_proposition_l54_5424

theorem negation_existence_proposition :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existence_proposition_l54_5424


namespace NUMINAMATH_CALUDE_divisibility_proof_l54_5497

theorem divisibility_proof (n : ℕ) : 
  n = 6268440 → n % 8 = 0 ∧ n % 66570 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_proof_l54_5497


namespace NUMINAMATH_CALUDE_total_seedlings_l54_5436

/-- Given that each packet contains 7 seeds and there are 60 packets,
    prove that the total number of seedlings is 420. -/
theorem total_seedlings (seeds_per_packet : ℕ) (num_packets : ℕ) 
  (h1 : seeds_per_packet = 7) 
  (h2 : num_packets = 60) : 
  seeds_per_packet * num_packets = 420 := by
sorry

end NUMINAMATH_CALUDE_total_seedlings_l54_5436


namespace NUMINAMATH_CALUDE_solution_set_inequality_l54_5463

theorem solution_set_inequality (x : ℝ) : 
  (x^2 - |x - 1| - 1 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l54_5463


namespace NUMINAMATH_CALUDE_cube_preserves_order_l54_5448

theorem cube_preserves_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l54_5448


namespace NUMINAMATH_CALUDE_author_writing_speed_l54_5410

/-- Given an author who writes 25,000 words in 50 hours, prove that their average writing speed is 500 words per hour. -/
theorem author_writing_speed :
  let total_words : ℕ := 25000
  let total_hours : ℕ := 50
  let average_speed : ℕ := total_words / total_hours
  average_speed = 500 :=
by sorry

end NUMINAMATH_CALUDE_author_writing_speed_l54_5410


namespace NUMINAMATH_CALUDE_economy_to_luxury_ratio_l54_5494

/-- Represents the ratio between two quantities -/
structure Ratio where
  antecedent : ℕ
  consequent : ℕ

/-- Represents the inventory of a car dealership -/
structure CarInventory where
  economy_to_suv : Ratio
  luxury_to_suv : Ratio

theorem economy_to_luxury_ratio (inventory : CarInventory) 
  (h1 : inventory.economy_to_suv = Ratio.mk 4 1)
  (h2 : inventory.luxury_to_suv = Ratio.mk 8 1) :
  Ratio.mk 1 2 = 
    Ratio.mk 
      (inventory.economy_to_suv.antecedent * inventory.luxury_to_suv.consequent)
      (inventory.economy_to_suv.consequent * inventory.luxury_to_suv.antecedent) :=
by sorry

end NUMINAMATH_CALUDE_economy_to_luxury_ratio_l54_5494


namespace NUMINAMATH_CALUDE_reuschles_theorem_l54_5460

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define points A₁, B₁, C₁ on the sides of triangle ABC
variable (A₁ B₁ C₁ : ℝ × ℝ)

-- Define the condition that AA₁, BB₁, CC₁ intersect at a single point
def lines_concurrent (A B C A₁ B₁ C₁ : ℝ × ℝ) : Prop := sorry

-- Define the circumcircle of triangle A₁B₁C₁
def circumcircle (A₁ B₁ C₁ : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define points A₂, B₂, C₂ as the second intersection points
def second_intersection (A B C A₁ B₁ C₁ : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Theorem statement
theorem reuschles_theorem (A B C A₁ B₁ C₁ : ℝ × ℝ) :
  lines_concurrent A B C A₁ B₁ C₁ →
  let (A₂, B₂, C₂) := second_intersection A B C A₁ B₁ C₁
  lines_concurrent A B C A₂ B₂ C₂ := by sorry

end NUMINAMATH_CALUDE_reuschles_theorem_l54_5460


namespace NUMINAMATH_CALUDE_complex_division_product_l54_5440

/-- Given (2+3i)/i = a+bi, where a and b are real numbers and i is the imaginary unit, prove that ab = 6 -/
theorem complex_division_product (a b : ℝ) : (Complex.I : ℂ)⁻¹ * (2 + 3 * Complex.I) = a + b * Complex.I → a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_division_product_l54_5440


namespace NUMINAMATH_CALUDE_units_digit_of_special_number_l54_5446

def is_product_of_one_digit_numbers (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), (factors.all (λ x => x > 0 ∧ x < 10)) ∧ 
    (factors.prod = n)

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

theorem units_digit_of_special_number (n : ℕ) :
  n > 10 ∧ 
  is_product_of_one_digit_numbers n ∧ 
  Odd (digit_product n) →
  n % 10 = 5 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_special_number_l54_5446


namespace NUMINAMATH_CALUDE_cubic_function_theorem_l54_5467

/-- A cubic function with a parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

/-- The derivative of f -/
def f' (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3

theorem cubic_function_theorem (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f c x₁ = 0 ∧ f c x₂ = 0) →  -- two distinct zeros
  (∃ x₀ : ℝ, f c x₀ = 0 ∧ ∀ x : ℝ, f c x ≤ f c x₀) →  -- one zero is the maximum point
  c = -2 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_theorem_l54_5467


namespace NUMINAMATH_CALUDE_no_one_blue_point_coloring_l54_5404

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a color type
inductive Color
  | Red
  | Blue

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define a circle of radius 1
def unitCircle (center : Point) : Set Point :=
  {p : Point | (p.x - center.x)^2 + (p.y - center.y)^2 = 1}

-- State the theorem
theorem no_one_blue_point_coloring :
  ¬ (∀ (center : Point),
      ∃! (p : Point), p ∈ unitCircle center ∧ coloring p = Color.Blue) ∧
    (∃ (p q : Point), coloring p ≠ coloring q) :=
  sorry

end NUMINAMATH_CALUDE_no_one_blue_point_coloring_l54_5404


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l54_5416

def i : ℂ := Complex.I

theorem sum_of_powers_of_i : i^14760 + i^14761 + i^14762 + i^14763 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l54_5416


namespace NUMINAMATH_CALUDE_minimum_value_and_tangent_line_l54_5449

noncomputable def f (a b x : ℝ) : ℝ := a * Real.exp x + 1 / (a * Real.exp x) + b

theorem minimum_value_and_tangent_line (a b : ℝ) (ha : a > 0) :
  (∀ x ≥ 0, f a b x ≥ (if a ≥ 1 then a + 1/a + b else b + 2)) ∧
  (∃ x ≥ 0, f a b x = (if a ≥ 1 then a + 1/a + b else b + 2)) ∧
  ((f a b 2 = 3 ∧ (deriv (f a b)) 2 = 3/2) → a = 2 / Real.exp 2 ∧ b = 1/2) :=
sorry

end NUMINAMATH_CALUDE_minimum_value_and_tangent_line_l54_5449


namespace NUMINAMATH_CALUDE_reflection_theorem_l54_5493

def P : ℝ × ℝ := (1, 2)

-- Reflection across x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Reflection across origin
def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem reflection_theorem :
  reflect_x P = (1, -2) ∧ reflect_origin P = (-1, -2) := by
  sorry

end NUMINAMATH_CALUDE_reflection_theorem_l54_5493


namespace NUMINAMATH_CALUDE_honey_container_size_l54_5461

/-- The number of ounces in Tabitha's honey container -/
def honey_container_ounces : ℕ :=
  let servings_per_cup : ℕ := 1
  let cups_per_night : ℕ := 2
  let servings_per_ounce : ℕ := 6
  let nights_honey_lasts : ℕ := 48
  (servings_per_cup * cups_per_night * nights_honey_lasts) / servings_per_ounce

theorem honey_container_size :
  honey_container_ounces = 16 := by
  sorry

end NUMINAMATH_CALUDE_honey_container_size_l54_5461


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l54_5405

theorem sqrt_product_simplification : Real.sqrt 18 * Real.sqrt 72 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l54_5405


namespace NUMINAMATH_CALUDE_column_sorting_preserves_row_order_l54_5400

/-- Represents a 10x10 table of natural numbers -/
def Table := Fin 10 → Fin 10 → ℕ

/-- Check if a row is sorted in ascending order -/
def is_row_sorted (t : Table) (row : Fin 10) : Prop :=
  ∀ i j : Fin 10, i < j → t row i ≤ t row j

/-- Check if a column is sorted in ascending order -/
def is_column_sorted (t : Table) (col : Fin 10) : Prop :=
  ∀ i j : Fin 10, i < j → t i col ≤ t j col

/-- Check if all rows are sorted in ascending order -/
def are_all_rows_sorted (t : Table) : Prop :=
  ∀ row : Fin 10, is_row_sorted t row

/-- Check if all columns are sorted in ascending order -/
def are_all_columns_sorted (t : Table) : Prop :=
  ∀ col : Fin 10, is_column_sorted t col

/-- The table contains the first 100 natural numbers -/
def contains_first_100_numbers (t : Table) : Prop :=
  ∀ n : ℕ, n ≤ 100 → ∃ i j : Fin 10, t i j = n

theorem column_sorting_preserves_row_order :
  ∀ t : Table,
  contains_first_100_numbers t →
  are_all_rows_sorted t →
  ∃ t' : Table,
    (∀ i j : Fin 10, t i j ≤ t' i j) ∧
    are_all_columns_sorted t' ∧
    are_all_rows_sorted t' :=
sorry

end NUMINAMATH_CALUDE_column_sorting_preserves_row_order_l54_5400


namespace NUMINAMATH_CALUDE_marnie_bracelets_l54_5480

/-- The number of bags of 50 beads Marnie bought -/
def bags_50 : ℕ := 5

/-- The number of bags of 100 beads Marnie bought -/
def bags_100 : ℕ := 2

/-- The number of beads in each bag of 50 -/
def beads_per_bag_50 : ℕ := 50

/-- The number of beads in each bag of 100 -/
def beads_per_bag_100 : ℕ := 100

/-- The number of beads needed to make one bracelet -/
def beads_per_bracelet : ℕ := 50

/-- The total number of beads Marnie bought -/
def total_beads : ℕ := bags_50 * beads_per_bag_50 + bags_100 * beads_per_bag_100

/-- The number of bracelets Marnie can make -/
def bracelets : ℕ := total_beads / beads_per_bracelet

theorem marnie_bracelets : bracelets = 9 := by sorry

end NUMINAMATH_CALUDE_marnie_bracelets_l54_5480


namespace NUMINAMATH_CALUDE_ring_cost_calculation_l54_5437

/-- The cost of a single ring given the total sales and necklace price -/
def ring_cost (total_sales necklace_price : ℕ) (num_necklaces num_rings : ℕ) : ℕ :=
  (total_sales - necklace_price * num_necklaces) / num_rings

theorem ring_cost_calculation (total_sales necklace_price : ℕ) 
  (h1 : total_sales = 80)
  (h2 : necklace_price = 12)
  (h3 : ring_cost total_sales necklace_price 4 8 = 4) : 
  ∃ (x : ℕ), x = ring_cost 80 12 4 8 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ring_cost_calculation_l54_5437


namespace NUMINAMATH_CALUDE_problem_statement_l54_5498

theorem problem_statement (x y : ℝ) : (x + 1)^2 + |y - 2| = 0 → 2*x + 3*y = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l54_5498


namespace NUMINAMATH_CALUDE_count_sevens_in_range_l54_5409

/-- Count of digit 7 appearances in integers from 1 to 1000 -/
def count_sevens : ℕ := sorry

/-- The range of integers we're considering -/
def range_start : ℕ := 1
def range_end : ℕ := 1000

theorem count_sevens_in_range : count_sevens = 300 := by sorry

end NUMINAMATH_CALUDE_count_sevens_in_range_l54_5409


namespace NUMINAMATH_CALUDE_fedya_deposit_l54_5432

theorem fedya_deposit (k : ℕ) (h1 : k < 30) (h2 : k > 0) : 
  ∃ (n : ℕ), n * (100 - k) = 84700 ∧ n = 1100 := by
sorry

end NUMINAMATH_CALUDE_fedya_deposit_l54_5432


namespace NUMINAMATH_CALUDE_smaller_circle_radius_l54_5407

theorem smaller_circle_radius (r_large : ℝ) (A₁ A₂ : ℝ) : 
  r_large = 4 →
  A₁ + A₂ = π * (2 * r_large)^2 →
  2 * A₂ = A₁ + (A₁ + A₂) →
  A₁ = π * r_small^2 →
  r_small = 4 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_radius_l54_5407


namespace NUMINAMATH_CALUDE_quadratic_zeros_imply_range_bound_l54_5487

def quadratic_function (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_zeros_imply_range_bound (b c : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧
    quadratic_function b c x₁ = 0 ∧ quadratic_function b c x₂ = 0) →
  0 < (1 + b) * c + c^2 ∧ (1 + b) * c + c^2 < 1/16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_zeros_imply_range_bound_l54_5487


namespace NUMINAMATH_CALUDE_third_term_value_l54_5402

def S (n : ℕ) : ℤ := 2 * n^2 - 1

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem third_term_value : a 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_third_term_value_l54_5402


namespace NUMINAMATH_CALUDE_cube_diff_even_iff_sum_even_l54_5434

theorem cube_diff_even_iff_sum_even (p q : ℕ) : 
  Even (p^3 - q^3) ↔ Even (p + q) :=
by sorry

end NUMINAMATH_CALUDE_cube_diff_even_iff_sum_even_l54_5434


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l54_5443

theorem sum_of_reciprocal_roots (a b : ℝ) : 
  a ≠ b → 
  a^2 - 3*a - 1 = 0 → 
  b^2 - 3*b - 1 = 0 → 
  b/a + a/b = -11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l54_5443


namespace NUMINAMATH_CALUDE_root_quadratic_equation_l54_5464

theorem root_quadratic_equation (m : ℝ) : 
  m^2 - 2*m - 3 = 0 → m^2 - 2*m + 2020 = 2023 := by
sorry

end NUMINAMATH_CALUDE_root_quadratic_equation_l54_5464


namespace NUMINAMATH_CALUDE_fewest_students_twenty_two_satisfies_fewest_students_is_22_l54_5490

theorem fewest_students (n : ℕ) : (n % 5 = 2 ∧ n % 6 = 4 ∧ n % 8 = 6) → n ≥ 22 :=
by sorry

theorem twenty_two_satisfies : 22 % 5 = 2 ∧ 22 % 6 = 4 ∧ 22 % 8 = 6 :=
by sorry

theorem fewest_students_is_22 : 
  ∃ n : ℕ, n % 5 = 2 ∧ n % 6 = 4 ∧ n % 8 = 6 ∧ ∀ m : ℕ, (m % 5 = 2 ∧ m % 6 = 4 ∧ m % 8 = 6) → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_fewest_students_twenty_two_satisfies_fewest_students_is_22_l54_5490


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_sum_l54_5484

theorem cubic_polynomial_root_sum (f : ℝ → ℝ) (r₁ r₂ r₃ : ℝ) :
  (∃ a b c d : ℝ, ∀ x, f x = a * x^3 + b * x^2 + c * x + d) →
  (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  ((f (1/2) + f (-1/2)) / f 0 = 1003) →
  (1 / (r₁ * r₂) + 1 / (r₂ * r₃) + 1 / (r₃ * r₁) = 2002) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_sum_l54_5484


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l54_5417

theorem inequality_system_solution_set :
  {x : ℝ | 6 > 2 * (x + 1) ∧ 1 - x < 2} = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l54_5417


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_range_l54_5476

/-- Given f(x) = ln x - a, if f(x) < x^2 holds for all x > 1, then a ≥ -1 -/
theorem function_inequality_implies_parameter_range (a : ℝ) :
  (∀ x > 1, Real.log x - a < x^2) →
  a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_range_l54_5476


namespace NUMINAMATH_CALUDE_binary_263_ones_minus_zeros_l54_5477

def binary_representation (n : Nat) : List Nat :=
  sorry

def count_zeros (l : List Nat) : Nat :=
  sorry

def count_ones (l : List Nat) : Nat :=
  sorry

theorem binary_263_ones_minus_zeros :
  let bin_263 := binary_representation 263
  let x := count_zeros bin_263
  let y := count_ones bin_263
  y - x = 0 := by sorry

end NUMINAMATH_CALUDE_binary_263_ones_minus_zeros_l54_5477


namespace NUMINAMATH_CALUDE_infinitely_many_prime_divisors_l54_5420

theorem infinitely_many_prime_divisors 
  (a b c d : ℕ+) 
  (ha : a ≠ b ∧ a ≠ c ∧ a ≠ d) 
  (hb : b ≠ c ∧ b ≠ d) 
  (hc : c ≠ d) : 
  ∃ (s : Set ℕ), Set.Infinite s ∧ 
  (∀ p ∈ s, Prime p ∧ ∃ n : ℕ, p ∣ (a * c^n + b * d^n)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_prime_divisors_l54_5420


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l54_5411

theorem quadratic_equation_equivalence : ∃ (x : ℝ), 16 * x^2 - 32 * x - 512 = 0 ↔ ∃ (x : ℝ), (x - 1)^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l54_5411


namespace NUMINAMATH_CALUDE_cost_price_calculation_l54_5471

/-- Given an article sold at a 30% profit with a selling price of 364,
    prove that the cost price of the article is 280. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
    (h1 : selling_price = 364)
    (h2 : profit_percentage = 0.30) : 
  ∃ (cost_price : ℝ), cost_price = 280 ∧ 
    selling_price = cost_price * (1 + profit_percentage) := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l54_5471


namespace NUMINAMATH_CALUDE_x_plus_y_value_l54_5451

theorem x_plus_y_value (x y : ℝ) 
  (h1 : (4 : ℝ) ^ x = 16 ^ (y + 2))
  (h2 : (25 : ℝ) ^ y = 5 ^ (x - 7)) : 
  x + y = 8.5 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l54_5451


namespace NUMINAMATH_CALUDE_line_intersection_theorem_l54_5453

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the properties and relations
variable (skew : Line → Line → Prop)
variable (lies_on : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line)
variable (intersects : Line → Line → Prop)

-- State the theorem
theorem line_intersection_theorem 
  (l₁ l₂ l : Line) (α β : Plane)
  (h1 : skew l₁ l₂)
  (h2 : lies_on l₁ α)
  (h3 : lies_on l₂ β)
  (h4 : l = intersection α β) :
  intersects l l₁ ∨ intersects l l₂ :=
sorry

end NUMINAMATH_CALUDE_line_intersection_theorem_l54_5453


namespace NUMINAMATH_CALUDE_expand_product_l54_5429

theorem expand_product (x : ℝ) : (x + 4) * (2 * x - 9) = 2 * x^2 - x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l54_5429


namespace NUMINAMATH_CALUDE_guam_stay_duration_l54_5489

/-- Calculates the number of days spent in Guam given the regular plan cost, international data cost per day, and total charges for the month. -/
def days_in_guam (regular_plan : ℚ) (intl_data_cost : ℚ) (total_charges : ℚ) : ℚ :=
  (total_charges - regular_plan) / intl_data_cost

/-- Theorem stating that given the specific costs in the problem, the number of days in Guam is 10. -/
theorem guam_stay_duration :
  let regular_plan : ℚ := 175
  let intl_data_cost : ℚ := 3.5
  let total_charges : ℚ := 210
  days_in_guam regular_plan intl_data_cost total_charges = 10 := by
  sorry

end NUMINAMATH_CALUDE_guam_stay_duration_l54_5489


namespace NUMINAMATH_CALUDE_marbles_given_correct_l54_5421

/-- The number of marbles Tyrone gave to Eric -/
def marbles_given : ℕ := sorry

/-- Tyrone's initial number of marbles -/
def tyrone_initial : ℕ := 150

/-- Eric's initial number of marbles -/
def eric_initial : ℕ := 18

/-- Theorem stating the number of marbles Tyrone gave to Eric -/
theorem marbles_given_correct : 
  marbles_given = 24 ∧
  tyrone_initial - marbles_given = 3 * (eric_initial + marbles_given) :=
by sorry

end NUMINAMATH_CALUDE_marbles_given_correct_l54_5421


namespace NUMINAMATH_CALUDE_vaccine_waiting_time_l54_5468

/-- 
Given the waiting times for vaccine appointments and the total waiting time,
prove that the time waited after the second appointment is 14 days.
-/
theorem vaccine_waiting_time 
  (first_appointment_wait : ℕ) 
  (second_appointment_wait : ℕ)
  (total_wait : ℕ)
  (h1 : first_appointment_wait = 4)
  (h2 : second_appointment_wait = 20)
  (h3 : total_wait = 38) :
  total_wait - (first_appointment_wait + second_appointment_wait) = 14 := by
  sorry

end NUMINAMATH_CALUDE_vaccine_waiting_time_l54_5468


namespace NUMINAMATH_CALUDE_other_communities_count_l54_5401

/-- The number of boys belonging to other communities in a school with given total and percentages of specific communities -/
theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total = 850 →
  muslim_percent = 44 / 100 →
  hindu_percent = 14 / 100 →
  sikh_percent = 10 / 100 →
  ↑total * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 272 := by
  sorry

#check other_communities_count

end NUMINAMATH_CALUDE_other_communities_count_l54_5401


namespace NUMINAMATH_CALUDE_inequality_proof_l54_5462

theorem inequality_proof (a b c d e f : ℝ) (h : b^2 ≥ a^2 + c^2) :
  (a*f - c*d)^2 ≤ (a*e - b*d)^2 + (b*f - c*e)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l54_5462


namespace NUMINAMATH_CALUDE_neighborhood_cable_cost_l54_5478

/-- Calculates the total cost of power cable for a neighborhood grid --/
theorem neighborhood_cable_cost
  (ew_streets : ℕ) (ew_length : ℕ)
  (ns_streets : ℕ) (ns_length : ℕ)
  (cable_per_street : ℕ) (cable_cost : ℕ) :
  ew_streets = 18 →
  ew_length = 2 →
  ns_streets = 10 →
  ns_length = 4 →
  cable_per_street = 5 →
  cable_cost = 2000 →
  (ew_streets * ew_length + ns_streets * ns_length) * cable_per_street * cable_cost = 760000 :=
by sorry

end NUMINAMATH_CALUDE_neighborhood_cable_cost_l54_5478


namespace NUMINAMATH_CALUDE_test_questions_l54_5485

theorem test_questions (sections : ℕ) (correct_answers : ℕ) (lower_bound : ℚ) (upper_bound : ℚ) :
  sections = 5 →
  correct_answers = 32 →
  lower_bound = 70/100 →
  upper_bound = 77/100 →
  ∃ (total_questions : ℕ),
    (total_questions % sections = 0) ∧
    (lower_bound < (correct_answers : ℚ) / total_questions) ∧
    ((correct_answers : ℚ) / total_questions < upper_bound) ∧
    (total_questions = 45) := by
  sorry

#check test_questions

end NUMINAMATH_CALUDE_test_questions_l54_5485


namespace NUMINAMATH_CALUDE_nancy_crayons_l54_5455

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 15

/-- The number of packs Nancy bought -/
def packs_bought : ℕ := 41

/-- The total number of crayons Nancy bought -/
def total_crayons : ℕ := crayons_per_pack * packs_bought

theorem nancy_crayons : total_crayons = 615 := by
  sorry

end NUMINAMATH_CALUDE_nancy_crayons_l54_5455


namespace NUMINAMATH_CALUDE_johns_age_satisfies_condition_l54_5456

/-- Represents John's current age in years -/
def johnsCurrentAge : ℕ := 18

/-- Represents the condition that five years ago, John's age was half of what it will be in 8 years -/
def ageCondition (age : ℕ) : Prop :=
  age - 5 = (age + 8) / 2

/-- Theorem stating that John's current age satisfies the given condition -/
theorem johns_age_satisfies_condition : ageCondition johnsCurrentAge := by
  sorry

#check johns_age_satisfies_condition

end NUMINAMATH_CALUDE_johns_age_satisfies_condition_l54_5456


namespace NUMINAMATH_CALUDE_range_of_a_l54_5499

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ a ∈ Set.Ioo (-1 : ℝ) 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l54_5499


namespace NUMINAMATH_CALUDE_statements_b_and_c_correct_l54_5475

theorem statements_b_and_c_correct :
  (∀ (a b c : ℝ), a * c^2 > b * c^2 → a > b) ∧
  (∀ (a b : ℝ), a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) :=
by sorry

end NUMINAMATH_CALUDE_statements_b_and_c_correct_l54_5475


namespace NUMINAMATH_CALUDE_midpoint_sum_x_invariant_l54_5452

/-- Represents a polygon in the Cartesian plane -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- Creates a new polygon from the midpoints of the sides of the given polygon -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- Computes the sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

/-- Theorem: The sum of x-coordinates remains constant through midpoint constructions -/
theorem midpoint_sum_x_invariant (Q₁ : Polygon) :
  sumXCoordinates Q₁ = 120 →
  let Q₂ := midpointPolygon Q₁
  let Q₃ := midpointPolygon Q₂
  let Q₄ := midpointPolygon Q₃
  sumXCoordinates Q₄ = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_x_invariant_l54_5452


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_18_20_l54_5473

theorem smallest_divisible_by_15_18_20 : 
  ∃ n : ℕ, n > 0 ∧ 15 ∣ n ∧ 18 ∣ n ∧ 20 ∣ n ∧ ∀ m : ℕ, m > 0 → 15 ∣ m → 18 ∣ m → 20 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_18_20_l54_5473


namespace NUMINAMATH_CALUDE_smallest_multiple_of_three_l54_5435

def cards : List ℕ := [1, 2, 6]

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ ∃ (a b : ℕ), a ∈ cards ∧ b ∈ cards ∧ a ≠ b ∧ n = 10 * a + b

def is_multiple_of_three (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 3 * k

theorem smallest_multiple_of_three :
  ∃ (n : ℕ), is_valid_number n ∧ is_multiple_of_three n ∧
  ∀ (m : ℕ), is_valid_number m → is_multiple_of_three m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_three_l54_5435


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l54_5414

/-- The standard equation of an ellipse given its parametric form -/
theorem ellipse_standard_equation (x y α : ℝ) :
  (x = 5 * Real.cos α) ∧ (y = 3 * Real.sin α) →
  (x^2 / 25 + y^2 / 9 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l54_5414


namespace NUMINAMATH_CALUDE_additional_earnings_is_correct_l54_5445

/-- Represents the company's dividend policy and earnings information -/
structure CompanyData where
  expected_earnings : ℝ
  actual_earnings : ℝ
  base_dividend_ratio : ℝ
  extra_dividend_rate : ℝ
  shares_owned : ℕ
  total_dividend_paid : ℝ

/-- Calculates the additional earnings per share that triggers the extra dividend -/
def additional_earnings (data : CompanyData) : ℝ :=
  data.actual_earnings - data.expected_earnings

/-- Theorem stating that the additional earnings per share is $0.30 -/
theorem additional_earnings_is_correct (data : CompanyData) 
  (h1 : data.expected_earnings = 0.80)
  (h2 : data.actual_earnings = 1.10)
  (h3 : data.base_dividend_ratio = 0.5)
  (h4 : data.extra_dividend_rate = 0.04)
  (h5 : data.shares_owned = 400)
  (h6 : data.total_dividend_paid = 208) :
  additional_earnings data = 0.30 := by
  sorry

#eval additional_earnings {
  expected_earnings := 0.80,
  actual_earnings := 1.10,
  base_dividend_ratio := 0.5,
  extra_dividend_rate := 0.04,
  shares_owned := 400,
  total_dividend_paid := 208
}

end NUMINAMATH_CALUDE_additional_earnings_is_correct_l54_5445


namespace NUMINAMATH_CALUDE_price_decrease_after_increase_l54_5403

theorem price_decrease_after_increase (original_price : ℝ) (original_price_pos : original_price > 0) :
  let increased_price := original_price * 1.3
  let decrease_factor := 1 - (1 / 1.3)
  increased_price * (1 - decrease_factor) = original_price :=
by sorry

end NUMINAMATH_CALUDE_price_decrease_after_increase_l54_5403


namespace NUMINAMATH_CALUDE_large_circle_diameter_is_32_l54_5470

/-- Represents the arrangement of circles as described in the problem -/
structure CircleArrangement where
  small_circle_radius : ℝ
  num_small_circles : ℕ
  num_layers : ℕ

/-- The specific arrangement described in the problem -/
def problem_arrangement : CircleArrangement :=
  { small_circle_radius := 4
  , num_small_circles := 8
  , num_layers := 2 }

/-- The diameter of the large circle in the arrangement -/
def large_circle_diameter (ca : CircleArrangement) : ℝ := 32

/-- Theorem stating that the diameter of the large circle in the problem arrangement is 32 units -/
theorem large_circle_diameter_is_32 :
  large_circle_diameter problem_arrangement = 32 := by
  sorry

end NUMINAMATH_CALUDE_large_circle_diameter_is_32_l54_5470


namespace NUMINAMATH_CALUDE_product_expansion_l54_5465

theorem product_expansion (x y : ℝ) :
  (3 * x + 4) * (2 * x + 6 * y + 7) = 6 * x^2 + 18 * x * y + 29 * x + 24 * y + 28 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l54_5465


namespace NUMINAMATH_CALUDE_cube_root_five_to_seven_sum_l54_5459

theorem cube_root_five_to_seven_sum : 
  (5^7 + 5^7 + 5^7 + 5^7 + 5^7 : ℝ)^(1/3) = 25 * (5^2)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_five_to_seven_sum_l54_5459


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l54_5466

theorem similar_triangles_leg_sum (a b c d e f : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  a^2 + b^2 = c^2 →
  d^2 + e^2 = f^2 →
  (1/2) * a * b = 24 →
  (1/2) * d * e = 600 →
  c = 13 →
  (a / d)^2 = (b / e)^2 →
  d + e = 85 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l54_5466


namespace NUMINAMATH_CALUDE_student_arrangements_l54_5426

/-- The number of students in the row -/
def n : ℕ := 7

/-- The number of arrangements where A and B must stand together -/
def arrangements_together : ℕ := 1440

/-- The number of arrangements where A is not at the head and B is not at the end -/
def arrangements_not_head_end : ℕ := 3720

/-- The number of arrangements where there is exactly one person between A and B -/
def arrangements_one_between : ℕ := 1200

/-- Theorem stating the correct number of arrangements for each situation -/
theorem student_arrangements :
  (arrangements_together = 1440) ∧
  (arrangements_not_head_end = 3720) ∧
  (arrangements_one_between = 1200) := by sorry

end NUMINAMATH_CALUDE_student_arrangements_l54_5426


namespace NUMINAMATH_CALUDE_square_area_proof_l54_5482

theorem square_area_proof (x : ℝ) : 
  (3 * x - 12 = 24 - 2 * x) → 
  ((3 * x - 12) ^ 2 = 92.16) := by
sorry

end NUMINAMATH_CALUDE_square_area_proof_l54_5482


namespace NUMINAMATH_CALUDE_existence_of_special_set_l54_5474

theorem existence_of_special_set :
  ∃ (S : Finset ℕ), 
    Finset.card S = 1998 ∧ 
    ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → (a * b) % ((a - b) ^ 2) = 0 :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_set_l54_5474


namespace NUMINAMATH_CALUDE_probability_marked_vertex_half_l54_5428

/-- Represents a shape with triangles and a marked vertex -/
structure TriangleShape where
  totalTriangles : ℕ
  trianglesWithMarkedVertex : ℕ
  hasProp : trianglesWithMarkedVertex ≤ totalTriangles

/-- The probability of selecting a triangle with the marked vertex -/
def probabilityMarkedVertex (shape : TriangleShape) : ℚ :=
  shape.trianglesWithMarkedVertex / shape.totalTriangles

theorem probability_marked_vertex_half (shape : TriangleShape) 
  (h1 : shape.totalTriangles = 6)
  (h2 : shape.trianglesWithMarkedVertex = 3) :
  probabilityMarkedVertex shape = 1/2 := by
  sorry

#check probability_marked_vertex_half

end NUMINAMATH_CALUDE_probability_marked_vertex_half_l54_5428


namespace NUMINAMATH_CALUDE_expand_expression_1_expand_expression_2_expand_expression_3_simplified_calculation_l54_5491

-- Problem 1
theorem expand_expression_1 (x y : ℝ) :
  -4 * x^2 * y * (x * y - 5 * y^2 - 1) = -4 * x^3 * y^2 + 20 * x^2 * y^3 + 4 * x^2 * y :=
sorry

-- Problem 2
theorem expand_expression_2 (a : ℝ) :
  (-3 * a)^2 - (2 * a + 1) * (a - 2) = 7 * a^2 + 3 * a + 2 :=
sorry

-- Problem 3
theorem expand_expression_3 (x y : ℝ) :
  (-2 * x - 3 * y) * (3 * y - 2 * x) - (2 * x - 3 * y)^2 = 12 * x * y - 18 * y^2 :=
sorry

-- Problem 4
theorem simplified_calculation :
  2010^2 - 2011 * 2009 = 1 :=
sorry

end NUMINAMATH_CALUDE_expand_expression_1_expand_expression_2_expand_expression_3_simplified_calculation_l54_5491


namespace NUMINAMATH_CALUDE_fixed_deposit_equation_l54_5413

theorem fixed_deposit_equation (x : ℝ) : 
  (∀ (interest_rate deposit_tax_rate final_amount : ℝ),
    interest_rate = 0.0198 →
    deposit_tax_rate = 0.20 →
    final_amount = 1300 →
    x + interest_rate * x * (1 - deposit_tax_rate) = final_amount) :=
by sorry

end NUMINAMATH_CALUDE_fixed_deposit_equation_l54_5413


namespace NUMINAMATH_CALUDE_sector_area_l54_5430

/-- The area of a circular sector with central angle 5π/7 and perimeter 5π+14 is 35π/2 -/
theorem sector_area (r : ℝ) (h1 : r > 0) : 
  (5 / 7 * π * r + 2 * r = 5 * π + 14) →
  (1 / 2 * (5 / 7 * π) * r^2 = 35 * π / 2) := by
sorry


end NUMINAMATH_CALUDE_sector_area_l54_5430


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l54_5450

theorem simplify_radical_expression : 
  Real.sqrt 80 - 3 * Real.sqrt 20 + Real.sqrt 500 / Real.sqrt 5 + 2 * Real.sqrt 45 = 4 * Real.sqrt 5 + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l54_5450


namespace NUMINAMATH_CALUDE_cylindrical_block_volume_l54_5444

/-- Represents a cylindrical iron block -/
structure CylindricalBlock where
  height : ℝ
  volume : ℝ

/-- Represents a frustum-shaped iron block -/
structure FrustumBlock where
  height : ℝ
  base_radius : ℝ

/-- Represents a container with a cylindrical and a frustum-shaped block -/
structure Container where
  cylindrical_block : CylindricalBlock
  frustum_block : FrustumBlock

/-- Theorem stating the volume of the cylindrical block in the container -/
theorem cylindrical_block_volume (container : Container) 
  (h1 : container.cylindrical_block.height = 3)
  (h2 : container.frustum_block.height = 3)
  (h3 : container.frustum_block.base_radius = container.frustum_block.base_radius) :
  container.cylindrical_block.volume = 15.42 := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_block_volume_l54_5444
