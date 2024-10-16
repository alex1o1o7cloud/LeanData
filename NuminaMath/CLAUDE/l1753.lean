import Mathlib

namespace NUMINAMATH_CALUDE_max_area_region_S_l1753_175347

/-- A circle in a plane -/
structure Circle where
  radius : ℝ
  center : ℝ × ℝ

/-- The line to which the circles are tangent -/
def TangentLine : Set (ℝ × ℝ) := sorry

/-- The point at which the circles are tangent to the line -/
def TangentPoint : ℝ × ℝ := sorry

/-- The set of four circles with radii 1, 3, 5, and 7 -/
def FourCircles : Set Circle := sorry

/-- The region S composed of all points that lie within one of the four circles -/
def RegionS (circles : Set Circle) : Set (ℝ × ℝ) := sorry

/-- The area of a set in the plane -/
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the maximum area of region S is 65π -/
theorem max_area_region_S :
  ∃ (c : Set Circle), c = FourCircles ∧
    (∀ circle ∈ c, ∃ p ∈ TangentLine, p = TangentPoint ∧
      (circle.center.1 - p.1)^2 + (circle.center.2 - p.2)^2 = circle.radius^2) ∧
    (∀ arrangement : Set Circle, arrangement = FourCircles →
      area (RegionS arrangement) ≤ area (RegionS c)) ∧
    area (RegionS c) = 65 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_max_area_region_S_l1753_175347


namespace NUMINAMATH_CALUDE_complement_union_A_B_l1753_175324

def U : Set Nat := {0, 1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 5}

theorem complement_union_A_B :
  (U \ (A ∪ B)) = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_A_B_l1753_175324


namespace NUMINAMATH_CALUDE_symmetric_probability_l1753_175332

/-- Represents a standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The number of dice rolled -/
def numDice : Nat := 8

/-- The sum we're comparing to -/
def givenSum : Nat := 15

/-- The sum we're proving has the same probability -/
def symmetricSum : Nat := 41

/-- Function to calculate the probability of a specific sum when rolling n dice -/
noncomputable def probability (n : Nat) (sum : Nat) : Real := sorry

theorem symmetric_probability : 
  probability numDice givenSum = probability numDice symmetricSum := by sorry

end NUMINAMATH_CALUDE_symmetric_probability_l1753_175332


namespace NUMINAMATH_CALUDE_sum_of_roots_l1753_175322

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 5*x = 1) 
  (hy : y^3 - 3*y^2 + 5*y = 5) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1753_175322


namespace NUMINAMATH_CALUDE_helga_shoe_shopping_l1753_175344

theorem helga_shoe_shopping (x : ℕ) : 
  (x + (x + 2) + 0 + 2 * (x + (x + 2) + 0) = 48) → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_helga_shoe_shopping_l1753_175344


namespace NUMINAMATH_CALUDE_number_of_book_pairs_l1753_175348

/-- Represents the number of books in each genre --/
structure BookCollection where
  mystery : Nat
  fantasy : Nat
  biography : Nat

/-- Represents the condition that "Mystery Masterpiece" must be included --/
def mustIncludeMysteryMasterpiece : Bool := true

/-- Calculates the number of possible book pairs --/
def calculatePossiblePairs (books : BookCollection) : Nat :=
  books.fantasy + books.biography

/-- Theorem stating the number of possible book pairs --/
theorem number_of_book_pairs :
  let books : BookCollection := ⟨4, 3, 3⟩
  calculatePossiblePairs books = 6 := by sorry

end NUMINAMATH_CALUDE_number_of_book_pairs_l1753_175348


namespace NUMINAMATH_CALUDE_cube_root_floor_product_limit_l1753_175306

def cube_root_floor_product (n : ℕ) : ℚ :=
  (Finset.range n).prod (λ i => ⌊(3 * i + 1 : ℚ)^(1/3)⌋) /
  (Finset.range n).prod (λ i => ⌊(3 * i + 2 : ℚ)^(1/3)⌋)

theorem cube_root_floor_product_limit : 
  cube_root_floor_product 167 = 1/8 := by sorry

end NUMINAMATH_CALUDE_cube_root_floor_product_limit_l1753_175306


namespace NUMINAMATH_CALUDE_cos_negative_thirteen_pi_fourths_l1753_175321

theorem cos_negative_thirteen_pi_fourths : 
  Real.cos (-13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_thirteen_pi_fourths_l1753_175321


namespace NUMINAMATH_CALUDE_parallelogram_area_l1753_175301

theorem parallelogram_area (base height : ℝ) (h1 : base = 12) (h2 : height = 18) :
  base * height = 216 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1753_175301


namespace NUMINAMATH_CALUDE_problem_statement_l1753_175353

theorem problem_statement (a b : ℝ) (ha : a > 0) (heq : Real.exp a + Real.log b = 1) :
  (a + Real.log b < 0) ∧ (Real.exp a + b > 2) ∧ (a + b > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1753_175353


namespace NUMINAMATH_CALUDE_max_value_function_1_inequality_function_2_inequality_function_3_max_value_function_4_l1753_175327

-- Statement 1
theorem max_value_function_1 (x : ℝ) (h : x < 1/2) :
  ∃ (M : ℝ), M = -1 ∧ ∀ y, y < 1/2 → 2*y + 1/(2*y-1) ≤ M :=
sorry

-- Statement 2
theorem inequality_function_2 (x : ℝ) (h : x > -2) :
  (x + 6) / Real.sqrt (x + 2) ≥ 4 :=
sorry

-- Statement 3
theorem inequality_function_3 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + 2*x*y = 8) :
  x + 2*y ≥ 4 :=
sorry

-- Statement 4
theorem max_value_function_4 (x : ℝ) (h : x < 1) :
  ∃ (M : ℝ), M = -5 ∧ ∀ y, y < 1 → (y^2 - y + 9) / (y - 1) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_function_1_inequality_function_2_inequality_function_3_max_value_function_4_l1753_175327


namespace NUMINAMATH_CALUDE_hockey_games_per_month_l1753_175313

/-- Proves that the number of hockey games played each month is 13,
    given that there are 182 hockey games in a 14-month season. -/
theorem hockey_games_per_month :
  let total_games : ℕ := 182
  let season_months : ℕ := 14
  let games_per_month : ℕ := total_games / season_months
  games_per_month = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_hockey_games_per_month_l1753_175313


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1753_175368

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1753_175368


namespace NUMINAMATH_CALUDE_equation_solution_l1753_175384

theorem equation_solution :
  ∃ y : ℚ, (2 * y + 3 * y = 600 - (4 * y + 5 * y + 100)) ∧ y = 250 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1753_175384


namespace NUMINAMATH_CALUDE_root_product_expression_l1753_175361

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 - p*α + 2 = 0) → 
  (β^2 - p*β + 2 = 0) → 
  (γ^2 + q*γ - 2 = 0) → 
  (δ^2 + q*δ - 2 = 0) → 
  (α - γ)*(β - γ)*(α - δ)*(β - δ) = -2*(p-q)^2 - 4*p*q + 4*q^2 + 16 := by
sorry

end NUMINAMATH_CALUDE_root_product_expression_l1753_175361


namespace NUMINAMATH_CALUDE_dog_park_problem_l1753_175377

theorem dog_park_problem (total_dogs : ℕ) (spotted_dogs : ℕ) (pointy_eared_dogs : ℕ) :
  (2 * spotted_dogs = total_dogs) →  -- Half of the dogs have spots
  (5 * pointy_eared_dogs = total_dogs) →  -- 1/5 of the dogs have pointy ears
  (spotted_dogs = 15) →  -- 15 dogs have spots
  pointy_eared_dogs = 6 :=
by sorry

end NUMINAMATH_CALUDE_dog_park_problem_l1753_175377


namespace NUMINAMATH_CALUDE_collinear_points_l1753_175326

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def are_collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem collinear_points :
  let a : ℝ := -1/14
  let b : ℝ := 1
  are_collinear 5 (-3) (-a + 4) b (3*a + 4) (b - 1) := by
sorry


end NUMINAMATH_CALUDE_collinear_points_l1753_175326


namespace NUMINAMATH_CALUDE_fraction_meaningful_range_l1753_175309

theorem fraction_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y = (x - 1) / (x - 2)) ↔ x ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_range_l1753_175309


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_cubic_vs_quadratic_l1753_175330

-- Part 1
theorem sqrt_sum_comparison : Real.sqrt 7 + Real.sqrt 10 > Real.sqrt 3 + Real.sqrt 14 := by
  sorry

-- Part 2
theorem cubic_vs_quadratic (x : ℝ) (h : x > 1) : x^3 > x^2 - x + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_cubic_vs_quadratic_l1753_175330


namespace NUMINAMATH_CALUDE_min_floor_sum_l1753_175308

theorem min_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_floor_sum_l1753_175308


namespace NUMINAMATH_CALUDE_proportion_equality_l1753_175351

theorem proportion_equality (h : (7 : ℚ) / 11 * 66 = 42) : (13 : ℚ) / 17 * 289 = 221 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l1753_175351


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l1753_175399

-- Define the universal set U
def U : Set ℤ := {x | 1 ≤ x ∧ x ≤ 7}

-- Define set A
def A : Set ℤ := {1, 3, 5, 7}

-- Define set B
def B : Set ℤ := {2, 4, 5}

-- Theorem statement
theorem intersection_complement_equals_set : B ∩ (U \ A) = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l1753_175399


namespace NUMINAMATH_CALUDE_min_sum_intercepts_l1753_175334

/-- A line passing through (1, 1) with positive intercepts -/
structure LineThroughOneOne where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : 1 / a + 1 / b = 1

/-- The sum of intercepts of a line -/
def sumOfIntercepts (l : LineThroughOneOne) : ℝ := l.a + l.b

/-- The equation x + y - 2 = 0 minimizes the sum of intercepts -/
theorem min_sum_intercepts :
  ∀ l : LineThroughOneOne, sumOfIntercepts l ≥ 4 ∧
  (sumOfIntercepts l = 4 ↔ l.a = 2 ∧ l.b = 2) :=
sorry

end NUMINAMATH_CALUDE_min_sum_intercepts_l1753_175334


namespace NUMINAMATH_CALUDE_initial_number_proof_l1753_175379

theorem initial_number_proof : ∃ x : ℕ, 
  (↑x + 5.000000000000043 : ℝ) % 23 = 0 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1753_175379


namespace NUMINAMATH_CALUDE_no_triple_prime_l1753_175350

theorem no_triple_prime (p : ℕ) : ¬(Nat.Prime p ∧ Nat.Prime (p^2 + 4) ∧ Nat.Prime (p^2 + 6)) := by
  sorry

end NUMINAMATH_CALUDE_no_triple_prime_l1753_175350


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1753_175356

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (h1 : a 3 = 4) (h2 : a 6 = 1/2) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1753_175356


namespace NUMINAMATH_CALUDE_smallest_in_S_l1753_175388

def S : Set ℕ := {5, 8, 1, 2, 6}

theorem smallest_in_S : ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_in_S_l1753_175388


namespace NUMINAMATH_CALUDE_exactly_three_valid_sequences_l1753_175372

/-- An arithmetic sequence with the given properties -/
structure ValidSequence where
  a₁ : ℕ
  d : ℕ
  h_a₁_single_digit : a₁ < 10
  h_100_in_seq : ∃ n : ℕ, a₁ + (n - 1) * d = 100
  h_3103_in_seq : ∃ m : ℕ, a₁ + (m - 1) * d = 3103
  h_max_terms : ∀ k : ℕ, a₁ + (k - 1) * d ≤ 3103 → k ≤ 240

/-- The set of all valid sequences -/
def validSequences : Set ValidSequence := {s | s.a₁ + 239 * s.d ≥ 3103}

theorem exactly_three_valid_sequences :
  ∃! (s₁ s₂ s₃ : ValidSequence),
    validSequences = {s₁, s₂, s₃} ∧
    s₁.a₁ = 9 ∧ s₁.d = 13 ∧
    s₂.a₁ = 1 ∧ s₂.d = 33 ∧
    s₃.a₁ = 9 ∧ s₃.d = 91 :=
  sorry

end NUMINAMATH_CALUDE_exactly_three_valid_sequences_l1753_175372


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1753_175320

/-- Given a circle (x-a)^2 + y^2 = 4 and a line x - y = 2, 
    if the chord length intercepted by the circle on the line is 2√2, 
    then a = 0 or a = 4 -/
theorem circle_line_intersection (a : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + y^2 = 4 ∧ x - y = 2) →  -- circle and line intersect
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ - a)^2 + y₁^2 = 4 ∧ x₁ - y₁ = 2 ∧  -- first intersection point
    (x₂ - a)^2 + y₂^2 = 4 ∧ x₂ - y₂ = 2 ∧  -- second intersection point
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →      -- chord length is 2√2
  a = 0 ∨ a = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1753_175320


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1753_175363

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im ((1 + i)^2 / (1 - i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1753_175363


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l1753_175394

/-- Given a college with students playing cricket or basketball, 
    this theorem proves the number of students playing both sports. -/
theorem students_playing_both_sports 
  (total : ℕ) 
  (cricket : ℕ) 
  (basketball : ℕ) 
  (h1 : total = 880) 
  (h2 : cricket = 500) 
  (h3 : basketball = 600) : 
  cricket + basketball - total = 220 := by
  sorry


end NUMINAMATH_CALUDE_students_playing_both_sports_l1753_175394


namespace NUMINAMATH_CALUDE_initial_amount_of_liquid_A_l1753_175300

/-- Given a mixture of liquids A and B with an initial ratio of 4:1, prove that the initial amount
of liquid A is 16 liters when 10 L of the mixture is replaced with liquid B, resulting in a new
ratio of 2:3. -/
theorem initial_amount_of_liquid_A (x : ℝ) : 
  (4 * x) / x = 4 / 1 →  -- Initial ratio of A to B is 4:1
  ((4 * x - 8) / (x + 8) = 2 / 3) →  -- New ratio after replacement is 2:3
  4 * x = 16 :=  -- Initial amount of liquid A is 16 liters
by sorry

#check initial_amount_of_liquid_A

end NUMINAMATH_CALUDE_initial_amount_of_liquid_A_l1753_175300


namespace NUMINAMATH_CALUDE_gcd_16_12_l1753_175392

def operation : List (ℕ × ℕ) := [(16, 12), (12, 4), (8, 4), (4, 4)]

theorem gcd_16_12 : Nat.gcd 16 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_16_12_l1753_175392


namespace NUMINAMATH_CALUDE_tournament_claim_inconsistency_l1753_175386

/-- Represents a single-elimination tournament -/
structure Tournament :=
  (participants : ℕ)
  (games_played : ℕ)

/-- Calculates the number of games in a single-elimination tournament -/
def games_in_tournament (t : Tournament) : ℕ := t.participants - 1

/-- Represents the claim made by some players -/
structure Claim :=
  (num_players : ℕ)
  (games_per_player : ℕ)

/-- Calculates the minimum number of games implied by a claim -/
def min_games_from_claim (c : Claim) : ℕ :=
  c.num_players * (c.games_per_player - 1)

/-- The main theorem -/
theorem tournament_claim_inconsistency (t : Tournament) (c : Claim) 
  (h1 : t.participants = 18)
  (h2 : c.num_players = 6)
  (h3 : c.games_per_player = 4) :
  min_games_from_claim c > games_in_tournament t :=
by sorry

end NUMINAMATH_CALUDE_tournament_claim_inconsistency_l1753_175386


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l1753_175387

/-- Represents the total population size -/
def total_population : ℕ := 260

/-- Represents the elderly population size -/
def elderly_population : ℕ := 60

/-- Represents the number of elderly people selected in the sample -/
def elderly_sample : ℕ := 3

/-- Represents the total sample size -/
def total_sample : ℕ := 13

/-- Theorem stating that the total sample size is correct given the stratified sampling conditions -/
theorem stratified_sampling_size :
  (elderly_sample : ℚ) / elderly_population = (total_sample : ℚ) / total_population :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l1753_175387


namespace NUMINAMATH_CALUDE_apples_remaining_after_three_days_l1753_175391

/-- Represents the number of apples picked from a tree on a given day -/
structure Picking where
  treeA : ℕ
  treeB : ℕ
  treeC : ℕ

/-- Calculates the total number of apples remaining after three days of picking -/
def applesRemaining (initialA initialB initialC : ℕ) (day1 day2 day3 : Picking) : ℕ :=
  (initialA - day1.treeA - day2.treeA - day3.treeA) +
  (initialB - day1.treeB - day2.treeB - day3.treeB) +
  (initialC - day1.treeC - day2.treeC - day3.treeC)

theorem apples_remaining_after_three_days :
  let initialA := 200
  let initialB := 250
  let initialC := 300
  let day1 := Picking.mk 40 25 0
  let day2 := Picking.mk 0 80 38
  let day3 := Picking.mk 60 0 40
  applesRemaining initialA initialB initialC day1 day2 day3 = 467 := by
  sorry


end NUMINAMATH_CALUDE_apples_remaining_after_three_days_l1753_175391


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1753_175383

-- Problem 1
theorem problem_1 (a b : ℝ) (h : a ≠ b) : 
  (a^2 / (a - b)) - (b^2 / (a - b)) = a + b :=
sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) (h3 : x ≠ 1) : 
  ((x^2 - 1) / (x^2 + 2*x + 1)) / ((x^2 - x) / (x + 1)) = 1 / x :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1753_175383


namespace NUMINAMATH_CALUDE_unit_vector_of_AB_l1753_175370

/-- Given a plane vector AB = (-1, 2), prove that its unit vector is (-√5/5, 2√5/5) -/
theorem unit_vector_of_AB (AB : ℝ × ℝ) (h : AB = (-1, 2)) :
  let magnitude := Real.sqrt ((AB.1)^2 + (AB.2)^2)
  (AB.1 / magnitude, AB.2 / magnitude) = (-Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_unit_vector_of_AB_l1753_175370


namespace NUMINAMATH_CALUDE_max_volume_l1753_175371

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  -- AB is perpendicular to BC and CD
  ab_perp_bc : True
  ab_perp_cd : True
  -- Length of BC is 2
  bc_length : ℝ
  bc_eq_two : bc_length = 2
  -- Dihedral angle between AB and CD is 60°
  dihedral_angle : ℝ
  dihedral_angle_eq_sixty : dihedral_angle = 60
  -- Circumradius is √5
  circumradius : ℝ
  circumradius_eq_sqrt_five : circumradius = Real.sqrt 5

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- The maximum possible volume of the tetrahedron -/
theorem max_volume (t : Tetrahedron) : volume t ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_volume_l1753_175371


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1753_175304

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)  -- arithmetic sequence
  (S : ℕ → ℚ)  -- sum function
  (h1 : a 1 = 2022)  -- first term
  (h2 : S 20 = 22)  -- sum of first 20 terms
  (h3 : ∀ n : ℕ, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1)))  -- sum formula
  (h4 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- common difference property
  : a 2 - a 1 = -20209 / 95 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1753_175304


namespace NUMINAMATH_CALUDE_abc_inequality_l1753_175374

theorem abc_inequality (a b c : ℝ) (ha : a = Real.rpow 0.8 0.8) 
  (hb : b = Real.rpow 0.8 0.9) (hc : c = Real.rpow 1.2 0.8) : c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1753_175374


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l1753_175373

theorem binomial_expansion_theorem (x a : ℝ) (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ 
    (n.choose 3) * x^(n - 3) * a^3 = 210 * k ∧
    (n.choose 4) * x^(n - 4) * a^4 = 420 * k ∧
    (n.choose 5) * x^(n - 5) * a^5 = 630 * k) →
  n = 19 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l1753_175373


namespace NUMINAMATH_CALUDE_star_pattern_identifiable_and_separable_l1753_175381

/-- Represents a patch in the tablecloth -/
structure Patch :=
  (shape : Type)
  (material : Type)

/-- Represents the tablecloth -/
structure Tablecloth :=
  (patches : Set Patch)
  (isTriangular : ∀ p ∈ patches, p.shape = Triangle)
  (isSilk : ∀ p ∈ patches, p.material = Silk)

/-- Represents a star pattern -/
structure StarPattern :=
  (patches : Set Patch)
  (isSymmetrical : Bool)
  (fitsWithRest : Tablecloth → Bool)

/-- Theorem: If a symmetrical star pattern exists in the tablecloth, it can be identified and separated -/
theorem star_pattern_identifiable_and_separable 
  (tc : Tablecloth) 
  (sp : StarPattern) 
  (h1 : sp.patches ⊆ tc.patches) 
  (h2 : sp.isSymmetrical = true) 
  (h3 : sp.fitsWithRest tc = true) : 
  ∃ (identified_sp : StarPattern), identified_sp = sp ∧ 
  ∃ (separated_tc : Tablecloth), separated_tc.patches = tc.patches \ sp.patches :=
sorry


end NUMINAMATH_CALUDE_star_pattern_identifiable_and_separable_l1753_175381


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l1753_175310

/-- Jessie's weight loss journey -/
theorem jessie_weight_loss (initial_weight lost_weight : ℕ) 
  (h1 : initial_weight = 192)
  (h2 : lost_weight = 126) :
  initial_weight - lost_weight = 66 :=
by sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l1753_175310


namespace NUMINAMATH_CALUDE_sqrt_square_abs_l1753_175335

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_l1753_175335


namespace NUMINAMATH_CALUDE_triangle_side_length_l1753_175307

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  C = 4 * A ∧  -- Given condition
  a = 30 ∧  -- Given side length
  c = 48 ∧  -- Given side length
  a / Real.sin A = c / Real.sin C ∧  -- Law of Sines
  b / Real.sin B = a / Real.sin A ∧  -- Law of Sines
  ∃ x : ℝ, 4 * x^3 - 4 * x - 8 / 5 = 0 ∧ x = Real.cos A  -- Equation for cosA
  →
  b = 30 * (5 - 20 * Real.sin A ^ 2 + 16 * Real.sin A ^ 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1753_175307


namespace NUMINAMATH_CALUDE_martha_children_count_l1753_175337

theorem martha_children_count (total_cakes : ℕ) (cakes_per_child : ℕ) (h1 : total_cakes = 18) (h2 : cakes_per_child = 6) : 
  total_cakes / cakes_per_child = 3 :=
by sorry

end NUMINAMATH_CALUDE_martha_children_count_l1753_175337


namespace NUMINAMATH_CALUDE_circular_ring_area_l1753_175316

/-- The area of a circular ring enclosed between two concentric circles -/
theorem circular_ring_area (C₁ C₂ : ℝ) (h : C₁ > C₂) :
  let S := (C₁^2 - C₂^2) / (4 * Real.pi)
  ∃ (R₁ R₂ : ℝ), R₁ > R₂ ∧ 
    C₁ = 2 * Real.pi * R₁ ∧ 
    C₂ = 2 * Real.pi * R₂ ∧
    S = Real.pi * R₁^2 - Real.pi * R₂^2 :=
by sorry

end NUMINAMATH_CALUDE_circular_ring_area_l1753_175316


namespace NUMINAMATH_CALUDE_function_composition_identity_l1753_175329

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x + b
  else if x < 3 then 2 * x - 1
  else 10 - 4 * x

theorem function_composition_identity (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_identity_l1753_175329


namespace NUMINAMATH_CALUDE_tan_cot_45_simplification_l1753_175305

theorem tan_cot_45_simplification :
  let tan_45 : ℝ := 1
  let cot_45 : ℝ := 1
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_cot_45_simplification_l1753_175305


namespace NUMINAMATH_CALUDE_concert_attendance_difference_l1753_175328

theorem concert_attendance_difference (first_concert : ℕ) (second_concert : ℕ)
  (h1 : first_concert = 65899)
  (h2 : second_concert = 66018) :
  second_concert - first_concert = 119 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_difference_l1753_175328


namespace NUMINAMATH_CALUDE_unique_maintaining_value_interval_for_square_maintaining_value_intervals_for_square_plus_constant_l1753_175393

/-- Definition of a "maintaining value" interval for a function f on [a,b] --/
def is_maintaining_value_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ 
  Monotone f ∧
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc a b, f x = y)

/-- The square function --/
def f (x : ℝ) : ℝ := x^2

/-- The square function with constant --/
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + m

/-- Theorem: [0,1] is the only "maintaining value" interval for f(x) = x^2 --/
theorem unique_maintaining_value_interval_for_square :
  ∀ a b : ℝ, is_maintaining_value_interval f a b ↔ a = 0 ∧ b = 1 :=
sorry

/-- Theorem: Characterization of "maintaining value" intervals for g(x) = x^2 + m --/
theorem maintaining_value_intervals_for_square_plus_constant :
  ∀ m : ℝ, m ≠ 0 →
  (∃ a b : ℝ, is_maintaining_value_interval (g m) a b) ↔ 
  (m ∈ Set.Icc (-1) (-3/4) ∪ Set.Ioc 0 (1/4)) :=
sorry

end NUMINAMATH_CALUDE_unique_maintaining_value_interval_for_square_maintaining_value_intervals_for_square_plus_constant_l1753_175393


namespace NUMINAMATH_CALUDE_product_of_fractions_l1753_175355

theorem product_of_fractions :
  (3 : ℚ) / 7 * (5 : ℚ) / 13 * (11 : ℚ) / 17 * (19 : ℚ) / 23 = 3135 / 35581 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1753_175355


namespace NUMINAMATH_CALUDE_equation_value_l1753_175341

theorem equation_value (x y : ℝ) (h : x^2 - 3*y - 5 = 0) : 2*x^2 - 6*y - 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l1753_175341


namespace NUMINAMATH_CALUDE_units_digit_of_150_factorial_l1753_175336

theorem units_digit_of_150_factorial (n : ℕ) : n = 150 → n.factorial % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_150_factorial_l1753_175336


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1753_175303

/-- Given a line L1 with equation 3x + 4y + 5 = 0 and a point P (0, -3),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 4y - 3x - 3 = 0 -/
theorem perpendicular_line_equation 
  (L1 : Set (ℝ × ℝ)) 
  (P : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ L1 ↔ 3 * x + 4 * y + 5 = 0) →
  P = (0, -3) →
  ∃ (L2 : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ L2 ↔ 4 * y - 3 * x - 3 = 0) ∧
    P ∈ L2 ∧
    (∀ (v w : ℝ × ℝ), v ∈ L1 → w ∈ L1 → v ≠ w →
      ∀ (p q : ℝ × ℝ), p ∈ L2 → q ∈ L2 → p ≠ q →
        ((v.1 - w.1) * (p.1 - q.1) + (v.2 - w.2) * (p.2 - q.2) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1753_175303


namespace NUMINAMATH_CALUDE_tens_digit_of_6_pow_22_l1753_175367

/-- The tens digit of 6^n -/
def tens_digit_of_6_pow (n : ℕ) : ℕ :=
  match n % 5 with
  | 0 => 6
  | 1 => 3
  | 2 => 1
  | 3 => 9
  | 4 => 7
  | _ => 0  -- This case should never occur

theorem tens_digit_of_6_pow_22 : tens_digit_of_6_pow 22 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_pow_22_l1753_175367


namespace NUMINAMATH_CALUDE_y_coordinate_of_first_point_l1753_175343

/-- Given a line with equation x = 2y + 5 passing through points (m, n) and (m + 5, n + 2.5),
    prove that the y-coordinate of the first point (n) is equal to (m - 5)/2. -/
theorem y_coordinate_of_first_point 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  n = (m - 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_of_first_point_l1753_175343


namespace NUMINAMATH_CALUDE_fraction_without_finite_decimal_l1753_175345

def has_finite_decimal_expansion (n d : ℕ) : Prop :=
  ∃ k : ℕ, d * (10 ^ k) % n = 0

theorem fraction_without_finite_decimal : 
  has_finite_decimal_expansion 9 10 ∧ 
  has_finite_decimal_expansion 3 5 ∧ 
  ¬ has_finite_decimal_expansion 3 7 ∧ 
  has_finite_decimal_expansion 7 8 :=
sorry

end NUMINAMATH_CALUDE_fraction_without_finite_decimal_l1753_175345


namespace NUMINAMATH_CALUDE_log_expression_eval_l1753_175349

-- Define lg as base 10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the fourth root
noncomputable def fourthRoot (x : ℝ) := Real.rpow x (1/4)

theorem log_expression_eval :
  Real.log (fourthRoot 27 / 3) / Real.log 3 + lg 25 + lg 4 + 7^(Real.log 2 / Real.log 7) = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_eval_l1753_175349


namespace NUMINAMATH_CALUDE_remaining_funds_is_38817_l1753_175319

/-- Represents the family's financial situation and tax obligations -/
structure FamilyFinances where
  father_income : ℕ
  mother_income : ℕ
  grandmother_pension : ℕ
  mikhail_scholarship : ℕ
  tax_deduction_per_child : ℕ
  num_children : ℕ
  income_tax_rate : ℚ
  monthly_savings : ℕ
  monthly_household_expenses : ℕ
  apartment_area : ℕ
  apartment_cadastral_value : ℕ
  car1_horsepower : ℕ
  car1_months_owned : ℕ
  car2_horsepower : ℕ
  car2_months_registered : ℕ
  land_area : ℕ
  land_cadastral_value : ℕ
  tour_cost_per_person : ℕ
  num_people_for_tour : ℕ

/-- Calculates the remaining funds for additional expenses -/
def calculate_remaining_funds (f : FamilyFinances) : ℕ :=
  sorry

/-- Theorem stating that the remaining funds for additional expenses is 38817 rubles -/
theorem remaining_funds_is_38817 (f : FamilyFinances) 
  (h1 : f.father_income = 50000)
  (h2 : f.mother_income = 28000)
  (h3 : f.grandmother_pension = 15000)
  (h4 : f.mikhail_scholarship = 3000)
  (h5 : f.tax_deduction_per_child = 1400)
  (h6 : f.num_children = 2)
  (h7 : f.income_tax_rate = 13 / 100)
  (h8 : f.monthly_savings = 10000)
  (h9 : f.monthly_household_expenses = 65000)
  (h10 : f.apartment_area = 78)
  (h11 : f.apartment_cadastral_value = 6240000)
  (h12 : f.car1_horsepower = 106)
  (h13 : f.car1_months_owned = 3)
  (h14 : f.car2_horsepower = 122)
  (h15 : f.car2_months_registered = 8)
  (h16 : f.land_area = 10)
  (h17 : f.land_cadastral_value = 420300)
  (h18 : f.tour_cost_per_person = 17900)
  (h19 : f.num_people_for_tour = 5) :
  calculate_remaining_funds f = 38817 :=
by sorry

end NUMINAMATH_CALUDE_remaining_funds_is_38817_l1753_175319


namespace NUMINAMATH_CALUDE_cube_sum_problem_l1753_175395

theorem cube_sum_problem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) :
  x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l1753_175395


namespace NUMINAMATH_CALUDE_racecourse_length_l1753_175365

/-- The length of a racecourse where two runners A and B finish simultaneously,
    given that A runs twice as fast as B and B starts 42 meters ahead. -/
theorem racecourse_length : ℝ := by
  /- Let v be B's speed -/
  let v : ℝ := 1
  /- A's speed is twice B's speed -/
  let speed_A : ℝ := 2 * v
  /- B starts 42 meters ahead -/
  let head_start : ℝ := 42
  /- d is the length of the racecourse -/
  let d : ℝ := 84
  /- Time for A to finish the race -/
  let time_A : ℝ := d / speed_A
  /- Time for B to finish the race -/
  let time_B : ℝ := (d - head_start) / v
  /- A and B reach the finish line simultaneously -/
  have h : time_A = time_B := by sorry
  /- The racecourse length is 84 meters -/
  exact d

end NUMINAMATH_CALUDE_racecourse_length_l1753_175365


namespace NUMINAMATH_CALUDE_not_right_triangle_2_3_4_l1753_175360

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that the set {2, 3, 4} cannot form a right triangle -/
theorem not_right_triangle_2_3_4 : ¬ is_right_triangle 2 3 4 := by
  sorry

#check not_right_triangle_2_3_4

end NUMINAMATH_CALUDE_not_right_triangle_2_3_4_l1753_175360


namespace NUMINAMATH_CALUDE_cats_and_dogs_owners_l1753_175376

/-- The number of people owning only cats and dogs -/
def catsDogs : ℕ := sorry

/-- The total number of people owning pets -/
def totalPeople : ℕ := 59

/-- The number of people owning only dogs -/
def onlyDogs : ℕ := 15

/-- The number of people owning only cats -/
def onlyCats : ℕ := 10

/-- The number of people owning cats, dogs, and snakes -/
def catsDogSnakes : ℕ := 3

/-- The total number of snakes -/
def totalSnakes : ℕ := 29

theorem cats_and_dogs_owners :
  catsDogs = totalPeople - (onlyDogs + onlyCats + catsDogSnakes + (totalSnakes - catsDogSnakes)) :=
by sorry

end NUMINAMATH_CALUDE_cats_and_dogs_owners_l1753_175376


namespace NUMINAMATH_CALUDE_icosahedron_edges_l1753_175333

/-- A regular icosahedron is a convex polyhedron with 20 congruent equilateral triangular faces -/
def RegularIcosahedron : Type := sorry

/-- The number of edges in a polyhedron -/
def num_edges (p : RegularIcosahedron) : ℕ := sorry

/-- Theorem: A regular icosahedron has 30 edges -/
theorem icosahedron_edges :
  ∀ (i : RegularIcosahedron), num_edges i = 30 := by sorry

end NUMINAMATH_CALUDE_icosahedron_edges_l1753_175333


namespace NUMINAMATH_CALUDE_greatest_a_divisible_by_three_l1753_175396

theorem greatest_a_divisible_by_three : 
  ∀ a : ℕ, 
    a < 10 → 
    (168 * 10000 + a * 100 + 26) % 3 = 0 → 
    a ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_a_divisible_by_three_l1753_175396


namespace NUMINAMATH_CALUDE_smallest_n_for_monochromatic_isosceles_trapezoid_l1753_175339

/-- A coloring of vertices using three colors -/
def Coloring (n : ℕ) := Fin n → Fin 3

/-- Checks if four vertices form an isosceles trapezoid in a regular n-gon -/
def IsIsoscelesTrapezoid (n : ℕ) (v1 v2 v3 v4 : Fin n) : Prop :=
  sorry

/-- Checks if a coloring contains four vertices of the same color forming an isosceles trapezoid -/
def HasMonochromaticIsoscelesTrapezoid (n : ℕ) (c : Coloring n) : Prop :=
  ∃ (v1 v2 v3 v4 : Fin n),
    c v1 = c v2 ∧ c v2 = c v3 ∧ c v3 = c v4 ∧
    IsIsoscelesTrapezoid n v1 v2 v3 v4

theorem smallest_n_for_monochromatic_isosceles_trapezoid :
  (∀ (c : Coloring 17), HasMonochromaticIsoscelesTrapezoid 17 c) ∧
  (∀ (n : ℕ), n < 17 → ∃ (c : Coloring n), ¬HasMonochromaticIsoscelesTrapezoid n c) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_monochromatic_isosceles_trapezoid_l1753_175339


namespace NUMINAMATH_CALUDE_sphere_volume_surface_area_ratio_l1753_175398

theorem sphere_volume_surface_area_ratio : 
  ∀ (r₁ r₂ : ℝ), r₁ > 0 → r₂ > 0 →
  (4/3 * π * r₁^3) / (4/3 * π * r₂^3) = 8 →
  (4 * π * r₁^2) / (4 * π * r₂^2) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_surface_area_ratio_l1753_175398


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1753_175340

theorem sufficient_condition_for_inequality (x : ℝ) :
  0 < x ∧ x < 2 → x^2 - 3*x < 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1753_175340


namespace NUMINAMATH_CALUDE_coefficient_x5y_in_expansion_l1753_175397

/-- The coefficient of x^5y in the expansion of (x-2y)^5(x+y) -/
def coefficient_x5y : ℤ := -9

/-- The expansion of (x-2y)^5(x+y) -/
def expansion (x y : ℚ) : ℚ := (x - 2*y)^5 * (x + y)

theorem coefficient_x5y_in_expansion :
  coefficient_x5y = (
    -- Extract the coefficient of x^5y from the expansion
    -- This part is left unimplemented as it requires complex polynomial manipulation
    sorry
  ) := by sorry

end NUMINAMATH_CALUDE_coefficient_x5y_in_expansion_l1753_175397


namespace NUMINAMATH_CALUDE_complex_sum_equality_l1753_175354

theorem complex_sum_equality : 
  let B : ℂ := 3 + 2*I
  let Q : ℂ := -5
  let R : ℂ := 1 - I
  let T : ℂ := 3 + 5*I
  B - Q + R + T = 2 + 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l1753_175354


namespace NUMINAMATH_CALUDE_monotone_decreasing_iff_b_positive_l1753_175342

/-- The function f(x) = (ax + b) / x is monotonically decreasing on (0, +∞) if and only if b > 0 -/
theorem monotone_decreasing_iff_b_positive (a b : ℝ) :
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → (a * x + b) / x > (a * y + b) / y) ↔ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_iff_b_positive_l1753_175342


namespace NUMINAMATH_CALUDE_mangoes_per_jar_l1753_175352

/-- Given the following conditions about Jordan's mangoes:
    - Jordan picked 54 mangoes
    - One-third of the mangoes were ripe
    - Two-thirds of the mangoes were unripe
    - Jordan kept 16 unripe mangoes
    - The remainder was given to his sister
    - His sister made 5 jars of pickled mangoes
    Prove that it takes 4 mangoes to fill one jar. -/
theorem mangoes_per_jar :
  ∀ (total_mangoes : ℕ)
    (ripe_fraction : ℚ)
    (unripe_fraction : ℚ)
    (kept_unripe : ℕ)
    (num_jars : ℕ),
  total_mangoes = 54 →
  ripe_fraction = 1/3 →
  unripe_fraction = 2/3 →
  kept_unripe = 16 →
  num_jars = 5 →
  (total_mangoes : ℚ) * unripe_fraction - kept_unripe = num_jars * 4 :=
by sorry

end NUMINAMATH_CALUDE_mangoes_per_jar_l1753_175352


namespace NUMINAMATH_CALUDE_baker_remaining_cakes_l1753_175369

theorem baker_remaining_cakes 
  (initial_cakes : ℝ) 
  (additional_cakes : ℝ) 
  (sold_cakes : ℝ) 
  (h1 : initial_cakes = 62.5)
  (h2 : additional_cakes = 149.25)
  (h3 : sold_cakes = 144.75) :
  initial_cakes + additional_cakes - sold_cakes = 67 := by
sorry

end NUMINAMATH_CALUDE_baker_remaining_cakes_l1753_175369


namespace NUMINAMATH_CALUDE_comparison_of_powers_l1753_175315

theorem comparison_of_powers : 
  let a : ℝ := Real.rpow 0.6 0.6
  let b : ℝ := Real.rpow 0.6 1.2
  let c : ℝ := Real.rpow 1.2 0.6
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_comparison_of_powers_l1753_175315


namespace NUMINAMATH_CALUDE_age_multiple_problem_l1753_175390

theorem age_multiple_problem (P_age Q_age : ℝ) (M : ℝ) : 
  P_age + Q_age = 100 →
  Q_age = 37.5 →
  37.5 = M * (Q_age - (P_age - Q_age)) →
  M = 3 := by
sorry

end NUMINAMATH_CALUDE_age_multiple_problem_l1753_175390


namespace NUMINAMATH_CALUDE_show_attendance_ratio_l1753_175323

/-- The ratio of attendees at the second showing to the debut show is 4 -/
theorem show_attendance_ratio : 
  ∀ (debut_attendance second_attendance ticket_price total_revenue : ℕ),
    debut_attendance = 200 →
    ticket_price = 25 →
    total_revenue = 20000 →
    second_attendance = total_revenue / ticket_price →
    second_attendance / debut_attendance = 4 := by
  sorry

end NUMINAMATH_CALUDE_show_attendance_ratio_l1753_175323


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1753_175311

theorem arithmetic_mean_problem (x : ℚ) : 
  (((x + 10) + 18 + 3*x + 16 + (x + 5) + (3*x + 6)) / 6 = 25) → x = 95/8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1753_175311


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1753_175366

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2*x + 1/(3*y)) * (2*x + 1/(3*y) - 2023) + (3*y + 1/(2*x)) * (3*y + 1/(2*x) - 2023) ≥ -2050529.5 :=
by sorry

theorem min_value_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧
  (2*x + 1/(3*y)) * (2*x + 1/(3*y) - 2023) + (3*y + 1/(2*x)) * (3*y + 1/(2*x) - 2023) = -2050529.5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1753_175366


namespace NUMINAMATH_CALUDE_carbonic_acid_molecular_weight_l1753_175364

/-- The molecular weight of carbonic acid in grams per mole. -/
def molecular_weight_carbonic_acid : ℝ := 62

/-- The number of moles of carbonic acid in the given sample. -/
def moles_carbonic_acid : ℝ := 8

/-- The total weight of the given sample of carbonic acid in grams. -/
def total_weight_carbonic_acid : ℝ := 496

/-- Theorem stating that the molecular weight of carbonic acid is 62 grams/mole,
    given that 8 moles of carbonic acid weigh 496 grams. -/
theorem carbonic_acid_molecular_weight :
  molecular_weight_carbonic_acid = total_weight_carbonic_acid / moles_carbonic_acid :=
by sorry

end NUMINAMATH_CALUDE_carbonic_acid_molecular_weight_l1753_175364


namespace NUMINAMATH_CALUDE_box_max_volume_l1753_175338

/-- The volume of the box as a function of the side length of the cut squares -/
def boxVolume (x : ℝ) : ℝ := (10 - 2*x) * (16 - 2*x) * x

/-- The maximum volume of the box -/
def maxVolume : ℝ := 144

theorem box_max_volume :
  ∃ (x : ℝ), 0 < x ∧ x < 5 ∧ 
  (∀ (y : ℝ), 0 < y ∧ y < 5 → boxVolume y ≤ boxVolume x) ∧
  boxVolume x = maxVolume :=
sorry

end NUMINAMATH_CALUDE_box_max_volume_l1753_175338


namespace NUMINAMATH_CALUDE_tom_tim_ratio_l1753_175346

structure TypingSpeed where
  tim : ℝ
  tom : ℝ

def combined_speed (s : TypingSpeed) : ℝ := s.tim + s.tom

def increased_speed (s : TypingSpeed) : ℝ := s.tim + 1.4 * s.tom

theorem tom_tim_ratio (s : TypingSpeed) 
  (h1 : combined_speed s = 20)
  (h2 : increased_speed s = 24) : 
  s.tom / s.tim = 1 := by
  sorry

end NUMINAMATH_CALUDE_tom_tim_ratio_l1753_175346


namespace NUMINAMATH_CALUDE_weights_division_impossibility_l1753_175318

theorem weights_division_impossibility : 
  let weights : List Nat := List.range 23
  let total_sum : Nat := (weights.sum + 23) - 21
  ¬ ∃ (half : Nat), 2 * half = total_sum
  := by sorry

end NUMINAMATH_CALUDE_weights_division_impossibility_l1753_175318


namespace NUMINAMATH_CALUDE_xiaoming_scoring_frequency_l1753_175325

/-- The frequency of scoring given total shots and goals -/
def scoring_frequency (total_shots : ℕ) (goals : ℕ) : ℚ :=
  (goals : ℚ) / (total_shots : ℚ)

/-- Theorem stating that given 80 total shots and 50 goals, the frequency of scoring is 0.625 -/
theorem xiaoming_scoring_frequency :
  scoring_frequency 80 50 = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_scoring_frequency_l1753_175325


namespace NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l1753_175317

def number : Nat := 18191

theorem greatest_prime_divisor_digit_sum (p : Nat) : 
  Nat.Prime p ∧ 
  p ∣ number ∧ 
  (∀ q : Nat, Nat.Prime q → q ∣ number → q ≤ p) →
  (p / 10 + p % 10) = 16 := by
sorry

end NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l1753_175317


namespace NUMINAMATH_CALUDE_train_length_l1753_175359

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 5 → speed * time * (1000 / 3600) = 125 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1753_175359


namespace NUMINAMATH_CALUDE_integer_difference_l1753_175385

theorem integer_difference (x y : ℕ+) : 
  x > y → x + y = 5 → x^3 - y^3 = 63 → x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_difference_l1753_175385


namespace NUMINAMATH_CALUDE_tessa_initial_apples_l1753_175357

/-- The number of apples needed to make a pie -/
def apples_for_pie : ℕ := 10

/-- The number of apples Anita gave to Tessa -/
def apples_from_anita : ℕ := 5

/-- The number of additional apples Tessa needs after receiving apples from Anita -/
def additional_apples_needed : ℕ := 1

/-- Tessa's initial number of apples -/
def initial_apples : ℕ := apples_for_pie - apples_from_anita - additional_apples_needed

theorem tessa_initial_apples :
  initial_apples = 4 := by sorry

end NUMINAMATH_CALUDE_tessa_initial_apples_l1753_175357


namespace NUMINAMATH_CALUDE_unique_positive_zero_implies_a_range_l1753_175302

/-- The function f(x) = ax^3 - 3x^2 + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem unique_positive_zero_implies_a_range (a : ℝ) :
  (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ > 0) → a < -2 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_zero_implies_a_range_l1753_175302


namespace NUMINAMATH_CALUDE_unique_three_digit_numbers_l1753_175380

theorem unique_three_digit_numbers : ∃! (x y : ℕ), 
  100 ≤ x ∧ x ≤ 999 ∧ 
  100 ≤ y ∧ y ≤ 999 ∧ 
  1000 * x + y = 7 * x * y ∧
  x = 143 ∧ y = 143 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_numbers_l1753_175380


namespace NUMINAMATH_CALUDE_simplify_expression_l1753_175358

theorem simplify_expression : 
  let x := 2 / (Real.sqrt 2 + Real.sqrt 3)
  let y := Real.sqrt 2 / (4 * Real.sqrt (97 + 56 * Real.sqrt 3))
  x + y = (3 * Real.sqrt 2) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1753_175358


namespace NUMINAMATH_CALUDE_supplementary_angles_problem_l1753_175382

theorem supplementary_angles_problem (x y : ℝ) :
  x + y = 180 ∧ y = x + 18 → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_supplementary_angles_problem_l1753_175382


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1753_175362

/-- Two angles are supplementary if their measures sum to 180 degrees -/
def Supplementary (a b : ℝ) : Prop := a + b = 180

theorem angle_measure_proof (A B : ℝ) 
  (h1 : Supplementary A B) 
  (h2 : A = 8 * B) : 
  A = 160 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1753_175362


namespace NUMINAMATH_CALUDE_art_club_officer_selection_l1753_175375

/-- Represents the Art Club with its members and officer selection rules -/
structure ArtClub where
  totalMembers : ℕ
  officerCount : ℕ
  andyIndex : ℕ
  breeIndex : ℕ
  carlosIndex : ℕ
  danaIndex : ℕ

/-- Calculates the number of ways to choose officers in the Art Club -/
def chooseOfficers (club : ArtClub) : ℕ :=
  sorry

/-- Theorem stating the correct number of ways to choose officers -/
theorem art_club_officer_selection (club : ArtClub) :
  club.totalMembers = 30 ∧
  club.officerCount = 4 ∧
  club.andyIndex ≠ club.breeIndex ∧
  club.carlosIndex ≠ club.danaIndex ∧
  club.andyIndex ≤ club.totalMembers ∧
  club.breeIndex ≤ club.totalMembers ∧
  club.carlosIndex ≤ club.totalMembers ∧
  club.danaIndex ≤ club.totalMembers →
  chooseOfficers club = 369208 :=
sorry

end NUMINAMATH_CALUDE_art_club_officer_selection_l1753_175375


namespace NUMINAMATH_CALUDE_trapezoid_area_l1753_175314

/-- The area of a trapezoid given the areas of triangles formed by its diagonals -/
theorem trapezoid_area (S₁ S₂ : ℝ) (h₁ : S₁ > 0) (h₂ : S₂ > 0) : 
  ∃ (A : ℝ), A = (Real.sqrt S₁ + Real.sqrt S₂)^2 ∧ A > 0 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1753_175314


namespace NUMINAMATH_CALUDE_chess_club_committees_l1753_175331

/-- The number of teams in the chess club -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 6

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 3

/-- The total number of members in the organizing committee -/
def committee_size : ℕ := 16

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem chess_club_committees :
  (num_teams * choose team_size host_selection * (choose team_size non_host_selection) ^ (num_teams - 1)) = 12000000 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_committees_l1753_175331


namespace NUMINAMATH_CALUDE_special_function_1988_l1753_175389

/-- A function from positive integers to positive integers satisfying the given property -/
def special_function (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, f (f m + f n) = m + n

/-- The main theorem stating that any function satisfying the special property maps 1988 to 1988 -/
theorem special_function_1988 (f : ℕ+ → ℕ+) (h : special_function f) : f 1988 = 1988 := by
  sorry

end NUMINAMATH_CALUDE_special_function_1988_l1753_175389


namespace NUMINAMATH_CALUDE_problem_solution_l1753_175378

-- Define the function f
def f (m : ℕ) (x : ℝ) : ℝ := |x - m| + |x|

-- State the theorem
theorem problem_solution (m : ℕ) (α β : ℝ) 
  (h1 : m > 0)
  (h2 : ∃ x : ℝ, f m x < 2)
  (h3 : α > 1)
  (h4 : β > 1)
  (h5 : f m α + f m β = 6) :
  (m = 1) ∧ ((4 / α) + (1 / β) ≥ 9 / 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1753_175378


namespace NUMINAMATH_CALUDE_pirate_game_l1753_175312

/-- Represents the number of coins each pirate has -/
structure PirateCoins where
  first : ℕ
  second : ℕ

/-- Simulates one round of the game where the first pirate loses half their coins -/
def firstLosesHalf (coins : PirateCoins) : PirateCoins :=
  { first := coins.first / 2,
    second := coins.second + coins.first / 2 }

/-- Simulates one round of the game where the second pirate loses half their coins -/
def secondLosesHalf (coins : PirateCoins) : PirateCoins :=
  { first := coins.first + coins.second / 2,
    second := coins.second / 2 }

/-- The main theorem to prove -/
theorem pirate_game (initial : ℕ) :
  (firstLosesHalf (secondLosesHalf (firstLosesHalf { first := initial, second := 0 })))
  = { first := 15, second := 33 } →
  initial = 24 := by
  sorry

end NUMINAMATH_CALUDE_pirate_game_l1753_175312
