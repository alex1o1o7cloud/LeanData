import Mathlib

namespace NUMINAMATH_CALUDE_sheepdog_catch_time_l1563_156396

theorem sheepdog_catch_time (sheep_speed dog_speed initial_distance : ℝ) 
  (h1 : sheep_speed = 16)
  (h2 : dog_speed = 28)
  (h3 : initial_distance = 240) : 
  initial_distance / (dog_speed - sheep_speed) = 20 := by
  sorry

#check sheepdog_catch_time

end NUMINAMATH_CALUDE_sheepdog_catch_time_l1563_156396


namespace NUMINAMATH_CALUDE_sandy_safe_moon_tokens_l1563_156306

theorem sandy_safe_moon_tokens :
  ∀ (T : ℕ),
    (T / 2 = T / 8 + 375000) →
    T = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_sandy_safe_moon_tokens_l1563_156306


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l1563_156309

theorem difference_of_squares_special_case : (527 : ℕ) * 527 - 526 * 528 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l1563_156309


namespace NUMINAMATH_CALUDE_min_sum_of_product_72_l1563_156327

theorem min_sum_of_product_72 (a b : ℤ) (h : a * b = 72) :
  ∀ x y : ℤ, x * y = 72 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 72 ∧ a₀ + b₀ = -73 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_72_l1563_156327


namespace NUMINAMATH_CALUDE_first_number_problem_l1563_156375

theorem first_number_problem (x y : ℤ) : y = 11 → x + (y + 3) = 19 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_number_problem_l1563_156375


namespace NUMINAMATH_CALUDE_candy_distribution_l1563_156358

theorem candy_distribution (total_candy : ℕ) (num_friends : ℕ) (candy_per_friend : ℕ) 
  (h1 : total_candy = 420)
  (h2 : num_friends = 35)
  (h3 : total_candy = num_friends * candy_per_friend) :
  candy_per_friend = 12 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1563_156358


namespace NUMINAMATH_CALUDE_intersection_determinant_l1563_156355

theorem intersection_determinant (a : ℝ) :
  (∃! p : ℝ × ℝ, a * p.1 + p.2 + 3 = 0 ∧ p.1 + p.2 + 2 = 0 ∧ 2 * p.1 - p.2 + 1 = 0) →
  Matrix.det !![a, 1, 3; 1, 1, 2; 2, -1, 1] = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_determinant_l1563_156355


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l1563_156378

-- Define a perfect square trinomial
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (x + k)^2

theorem perfect_square_trinomial_m_value (m : ℝ) :
  is_perfect_square_trinomial 1 (2*m) 9 → m = 3 ∨ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l1563_156378


namespace NUMINAMATH_CALUDE_product_of_roots_l1563_156372

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 18 → ∃ (x₁ x₂ : ℝ), x₁ * x₂ = -30 ∧ (x₁ + 3) * (x₁ - 4) = 18 ∧ (x₂ + 3) * (x₂ - 4) = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1563_156372


namespace NUMINAMATH_CALUDE_xy_positive_necessary_not_sufficient_l1563_156338

theorem xy_positive_necessary_not_sufficient (x y : ℝ) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → x * y > 0) ∧
  (∃ x y : ℝ, x * y > 0 ∧ ¬(x > 0 ∧ y > 0)) :=
by sorry

end NUMINAMATH_CALUDE_xy_positive_necessary_not_sufficient_l1563_156338


namespace NUMINAMATH_CALUDE_all_propositions_true_l1563_156391

def S (m n : ℝ) := {x : ℝ | m ≤ x ∧ x ≤ n}

theorem all_propositions_true 
  (m n : ℝ) 
  (h_nonempty : (S m n).Nonempty) 
  (h_closure : ∀ x ∈ S m n, x^2 ∈ S m n) :
  (m = 1 → S m n = {1}) ∧ 
  (m = -1/2 → 1/4 ≤ n ∧ n ≤ 1) ∧ 
  (n = 1/2 → -Real.sqrt 2/2 ≤ m ∧ m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_all_propositions_true_l1563_156391


namespace NUMINAMATH_CALUDE_water_segment_length_l1563_156365

/-- Represents the problem of finding the length of a water segment in a journey --/
theorem water_segment_length 
  (total_distance : ℝ) 
  (find_probability : ℝ) 
  (h1 : total_distance = 2500) 
  (h2 : find_probability = 7/10) : 
  ∃ water_length : ℝ, 
    water_length / total_distance = 1 - find_probability ∧ 
    water_length = 750 := by
  sorry

end NUMINAMATH_CALUDE_water_segment_length_l1563_156365


namespace NUMINAMATH_CALUDE_position_of_b_l1563_156322

theorem position_of_b (a b c : ℚ) (h : |a| + |b - c| = |a - c|) :
  (∃ a b c : ℚ, a < b ∧ c < b ∧ |a| + |b - c| = |a - c|) ∧
  (∃ a b c : ℚ, b < a ∧ b < c ∧ |a| + |b - c| = |a - c|) ∧
  (∃ a b c : ℚ, a < b ∧ b < c ∧ |a| + |b - c| = |a - c|) :=
sorry

end NUMINAMATH_CALUDE_position_of_b_l1563_156322


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1563_156345

theorem inequality_equivalence (x : ℝ) : 3 * x^2 - 2 * x - 1 > 4 * x + 5 ↔ x < 1 - Real.sqrt 3 ∨ x > 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1563_156345


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1563_156340

/-- Given two vectors a and b in ℝ², where a is perpendicular to a + b, prove that the second component of b is -6. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a.1 = 4 ∧ a.2 = 2 ∧ b.1 = -2) 
  (perp : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0) : 
  b.2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1563_156340


namespace NUMINAMATH_CALUDE_unique_reciprocal_function_l1563_156307

/-- Given a function f(x) = x / (ax + b) where a and b are constants, a ≠ 0,
    f(2) = 1, and f(x) = x has a unique solution, prove that f(x) = 2x / (x + 2) -/
theorem unique_reciprocal_function (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, x ≠ -b/a → (x / (a * x + b) = x → ∀ y, y ≠ -b/a → y / (a * y + b) = y → x = y)) →
  (2 / (2 * a + b) = 1) →
  (∀ x, x ≠ -b/a → x / (a * x + b) = 2 * x / (x + 2)) :=
by sorry

end NUMINAMATH_CALUDE_unique_reciprocal_function_l1563_156307


namespace NUMINAMATH_CALUDE_safari_count_l1563_156302

theorem safari_count (antelopes rabbits hyenas wild_dogs leopards : ℕ) : 
  antelopes = 80 →
  rabbits > antelopes →
  hyenas = antelopes + rabbits - 42 →
  wild_dogs = hyenas + 50 →
  leopards * 2 = rabbits →
  antelopes + rabbits + hyenas + wild_dogs + leopards = 605 →
  rabbits - antelopes = 70 := by
sorry

end NUMINAMATH_CALUDE_safari_count_l1563_156302


namespace NUMINAMATH_CALUDE_cannot_cover_naturals_with_disjoint_sets_l1563_156319

def S (α : ℝ) : Set ℕ := {n : ℕ | ∃ m : ℕ, n = ⌊m * α⌋}

theorem cannot_cover_naturals_with_disjoint_sets :
  ∀ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 →
  ¬(Disjoint (S α) (S β) ∧ Disjoint (S α) (S γ) ∧ Disjoint (S β) (S γ) ∧
    (S α ∪ S β ∪ S γ) = Set.univ) :=
sorry

end NUMINAMATH_CALUDE_cannot_cover_naturals_with_disjoint_sets_l1563_156319


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1563_156337

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 3 = 2 →
  a 4 * a 6 = 64 →
  (a 5 + a 6) / (a 1 + a 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1563_156337


namespace NUMINAMATH_CALUDE_right_triangle_area_l1563_156326

theorem right_triangle_area (h : ℝ) (angle : ℝ) :
  h = 8 * Real.sqrt 3 →
  angle = 30 * π / 180 →
  let a := h / 2
  let b := a * Real.sqrt 3
  (1 / 2) * a * b = 24 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1563_156326


namespace NUMINAMATH_CALUDE_triangular_array_coin_sum_l1563_156308

def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem triangular_array_coin_sum :
  ∃ N : ℕ, triangular_sum N = 3780 ∧ sum_of_digits N = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_coin_sum_l1563_156308


namespace NUMINAMATH_CALUDE_second_sum_proof_l1563_156352

/-- Given a total sum and interest conditions, prove the second sum -/
theorem second_sum_proof (total : ℝ) (first : ℝ) (second : ℝ) : 
  total = 2743 →
  first + second = total →
  (first * 3 / 100 * 8) = (second * 5 / 100 * 3) →
  second = 1688 := by
  sorry

end NUMINAMATH_CALUDE_second_sum_proof_l1563_156352


namespace NUMINAMATH_CALUDE_complex_number_real_twice_imaginary_l1563_156333

theorem complex_number_real_twice_imaginary (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) / (4 - 3 * Complex.I) + m / 25
  (z.re = 2 * z.im) → m = -1/5 := by
sorry

end NUMINAMATH_CALUDE_complex_number_real_twice_imaginary_l1563_156333


namespace NUMINAMATH_CALUDE_equation_solution_l1563_156366

theorem equation_solution :
  ∃! x : ℚ, (x ≠ 3 ∧ x ≠ -2) ∧ (3 - x) / (x + 2) + (3*x - 9) / (3 - x) = 2 :=
by
  use (-7/6)
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1563_156366


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1563_156361

/-- The statement "at least one of x and y is greater than 1" is neither a sufficient nor a necessary condition for x^2 + y^2 > 2 -/
theorem not_sufficient_not_necessary (x y : ℝ) : 
  ¬(((x > 1 ∨ y > 1) → x^2 + y^2 > 2) ∧ (x^2 + y^2 > 2 → (x > 1 ∨ y > 1))) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1563_156361


namespace NUMINAMATH_CALUDE_obtuse_triangle_properties_l1563_156321

/-- Properties of an obtuse triangle ABC -/
structure ObtuseTriangleABC where
  -- Side lengths
  a : ℝ
  b : ℝ
  -- Angle A in radians
  A : ℝ
  -- Triangle ABC is obtuse
  is_obtuse : Bool
  -- Given conditions
  ha : a = 7
  hb : b = 8
  hA : A = π / 3
  h_obtuse : is_obtuse = true

/-- Main theorem about the obtuse triangle ABC -/
theorem obtuse_triangle_properties (t : ObtuseTriangleABC) :
  -- 1. sin B = (4√3) / 7
  Real.sin (Real.arcsin ((t.b * Real.sin t.A) / t.a)) = (4 * Real.sqrt 3) / 7 ∧
  -- 2. Height on side BC = (12√3) / 7
  ∃ (h : ℝ), h = (12 * Real.sqrt 3) / 7 ∧ h = t.b * Real.sin (π - t.A - Real.arcsin ((t.b * Real.sin t.A) / t.a)) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_properties_l1563_156321


namespace NUMINAMATH_CALUDE_fourth_roll_max_probability_l1563_156370

-- Define the dice
structure Die :=
  (sides : ℕ)
  (max_prob : ℚ)
  (other_prob : ℚ)

-- Define the three dice
def six_sided_die : Die := ⟨6, 1/6, 1/6⟩
def eight_sided_die : Die := ⟨8, 3/4, 1/28⟩
def ten_sided_die : Die := ⟨10, 4/5, 1/45⟩

-- Define the probability of choosing each die
def choose_prob : ℚ := 1/3

-- Define the event of rolling maximum value three times for a given die
def max_three_times (d : Die) : ℚ := d.max_prob^3

-- Define the total probability of rolling maximum value three times
def total_max_three_times : ℚ :=
  choose_prob * (max_three_times six_sided_die + 
                 max_three_times eight_sided_die + 
                 max_three_times ten_sided_die)

-- Define the conditional probability of using each die given three max rolls
def cond_prob (d : Die) : ℚ :=
  (choose_prob * max_three_times d) / total_max_three_times

-- Define the probability of fourth roll being max given three max rolls
def fourth_max_prob : ℚ :=
  cond_prob six_sided_die * six_sided_die.max_prob +
  cond_prob eight_sided_die * eight_sided_die.max_prob +
  cond_prob ten_sided_die * ten_sided_die.max_prob

-- The theorem to prove
theorem fourth_roll_max_probability : 
  fourth_max_prob = 1443 / 2943 := by
  sorry

end NUMINAMATH_CALUDE_fourth_roll_max_probability_l1563_156370


namespace NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_l1563_156335

theorem tan_alpha_two_implies_fraction (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_l1563_156335


namespace NUMINAMATH_CALUDE_convex_polygon_diagonals_l1563_156329

theorem convex_polygon_diagonals (n : ℕ) (h : n = 49) : 
  (n * (n - 3)) / 2 = 23 * n := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_diagonals_l1563_156329


namespace NUMINAMATH_CALUDE_min_dot_product_l1563_156320

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

-- Define the fixed point E
def E : ℝ × ℝ := (3, 0)

-- Define a point on the ellipse
def point_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define perpendicularity condition
def perpendicular (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define dot product of vectors
def dot_product (A B C D : ℝ × ℝ) : ℝ :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2)

-- Theorem statement
theorem min_dot_product :
  ∀ P Q : ℝ × ℝ, 
  point_on_ellipse P → 
  point_on_ellipse Q → 
  perpendicular E P Q → 
  ∃ m : ℝ, 
  (∀ P' Q' : ℝ × ℝ, 
    point_on_ellipse P' → 
    point_on_ellipse Q' → 
    perpendicular E P' Q' → 
    m ≤ dot_product E P P Q) ∧ 
  m = 6 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_l1563_156320


namespace NUMINAMATH_CALUDE_davids_english_marks_l1563_156385

def davidsMaths : ℕ := 89
def davidsPhysics : ℕ := 82
def davidsChemistry : ℕ := 87
def davidsBiology : ℕ := 81
def averageMarks : ℕ := 85
def numberOfSubjects : ℕ := 5

theorem davids_english_marks :
  ∃ (englishMarks : ℕ),
    (englishMarks + davidsMaths + davidsPhysics + davidsChemistry + davidsBiology) / numberOfSubjects = averageMarks ∧
    englishMarks = 86 := by
  sorry

end NUMINAMATH_CALUDE_davids_english_marks_l1563_156385


namespace NUMINAMATH_CALUDE_number_ordering_l1563_156334

theorem number_ordering : 
  (3 : ℚ) / 8 < (3 : ℚ) / 4 ∧ 
  (3 : ℚ) / 4 < (7 : ℚ) / 5 ∧ 
  (7 : ℚ) / 5 < (143 : ℚ) / 100 ∧ 
  (143 : ℚ) / 100 < (13 : ℚ) / 8 := by
sorry

end NUMINAMATH_CALUDE_number_ordering_l1563_156334


namespace NUMINAMATH_CALUDE_inequality_range_l1563_156363

theorem inequality_range (x y a : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  (∀ x y, x > 0 → y > 0 → x + y = 1 → (1/x) + (16/y) > a^2 + 24*a) ↔ -25 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l1563_156363


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1563_156371

theorem right_triangle_side_length 
  (P Q R : ℝ × ℝ) 
  (right_angle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0) 
  (cos_R : Real.cos (Real.arccos ((3 * Real.sqrt 65) / 65)) = (3 * Real.sqrt 65) / 65) 
  (hypotenuse : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = Real.sqrt 169) :
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = (3 * Real.sqrt 65) / 5 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_side_length_l1563_156371


namespace NUMINAMATH_CALUDE_manuscript_pages_count_l1563_156382

/-- Represents the cost structure and revision information for a manuscript --/
structure ManuscriptInfo where
  firstTypingCost : ℕ
  revisionCost : ℕ
  pagesRevisedOnce : ℕ
  pagesRevisedTwice : ℕ
  totalCost : ℕ

/-- Calculates the total number of pages in a manuscript given its cost information --/
def calculateTotalPages (info : ManuscriptInfo) : ℕ :=
  sorry

/-- Theorem stating that for the given manuscript information, the total number of pages is 100 --/
theorem manuscript_pages_count (info : ManuscriptInfo) 
  (h1 : info.firstTypingCost = 10)
  (h2 : info.revisionCost = 5)
  (h3 : info.pagesRevisedOnce = 30)
  (h4 : info.pagesRevisedTwice = 20)
  (h5 : info.totalCost = 1350) :
  calculateTotalPages info = 100 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_pages_count_l1563_156382


namespace NUMINAMATH_CALUDE_lottery_jackpot_probability_l1563_156328

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def bonusBallCount : ℕ := 15
def winnerBallsDrawn : ℕ := 5

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def megaBallProb : ℚ := 1 / megaBallCount
def winnerBallsProb : ℚ := 1 / (binomial winnerBallCount winnerBallsDrawn)
def bonusBallProb : ℚ := 1 / bonusBallCount

theorem lottery_jackpot_probability : 
  megaBallProb * winnerBallsProb * bonusBallProb = 1 / 954594900 := by
  sorry

end NUMINAMATH_CALUDE_lottery_jackpot_probability_l1563_156328


namespace NUMINAMATH_CALUDE_find_second_number_l1563_156356

theorem find_second_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + 19 + x) / 3) + 7 →
  x = 70 := by
sorry

end NUMINAMATH_CALUDE_find_second_number_l1563_156356


namespace NUMINAMATH_CALUDE_train_speed_problem_l1563_156301

/-- Proves that Train B's speed is 80 mph given the problem conditions --/
theorem train_speed_problem (speed_a : ℝ) (time_difference : ℝ) (overtake_time : ℝ) :
  speed_a = 60 →
  time_difference = 40 / 60 →
  overtake_time = 120 / 60 →
  ∃ (speed_b : ℝ),
    speed_b * overtake_time = speed_a * (time_difference + overtake_time) ∧
    speed_b = 80 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_problem_l1563_156301


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l1563_156353

theorem completing_square_quadratic :
  ∀ x : ℝ, x^2 + 4*x - 1 = 0 ↔ (x + 2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l1563_156353


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1563_156394

/-- Given a geometric sequence starting with 25, -50, 100, -200, prove that its common ratio is -2 -/
theorem geometric_sequence_common_ratio :
  let a₁ : ℝ := 25
  let a₂ : ℝ := -50
  let a₃ : ℝ := 100
  let a₄ : ℝ := -200
  ∀ r : ℝ, (a₂ = r * a₁ ∧ a₃ = r * a₂ ∧ a₄ = r * a₃) → r = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1563_156394


namespace NUMINAMATH_CALUDE_xy_sum_when_equation_zero_l1563_156386

theorem xy_sum_when_equation_zero (x y : ℝ) :
  (3 * x - y + 5)^2 + |2 * x - y + 3| = 0 → x + y = -3 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_when_equation_zero_l1563_156386


namespace NUMINAMATH_CALUDE_remainder_problem_l1563_156323

theorem remainder_problem : (5^7 + 9^6 + 3^5) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1563_156323


namespace NUMINAMATH_CALUDE_ln_exp_relationship_l1563_156376

theorem ln_exp_relationship :
  (∀ x : ℝ, (Real.log x > 0) → (Real.exp x > 1)) ∧
  (∃ x : ℝ, Real.exp x > 1 ∧ Real.log x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ln_exp_relationship_l1563_156376


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1563_156348

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < π →
  B > 0 ∧ B < π →
  C > 0 ∧ C < π →
  A + B + C = π →
  a * Real.sin A = b * Real.sin B + (c - b) * Real.sin C →
  A = π / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1563_156348


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l1563_156312

theorem repeating_decimal_subtraction (x : ℚ) : x = 1/3 → 5 - 7 * x = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l1563_156312


namespace NUMINAMATH_CALUDE_min_fence_length_is_28_l1563_156300

/-- Represents a rectangular flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Calculates the minimum fence length required for a flower bed with one side against a wall -/
def minFenceLength (fb : FlowerBed) : ℝ :=
  2 * fb.width + fb.length

/-- The specific flower bed in the problem -/
def problemFlowerBed : FlowerBed :=
  { length := 12, width := 8 }

theorem min_fence_length_is_28 :
  minFenceLength problemFlowerBed = 28 := by
  sorry

#eval minFenceLength problemFlowerBed

end NUMINAMATH_CALUDE_min_fence_length_is_28_l1563_156300


namespace NUMINAMATH_CALUDE_min_m_for_24m_eq_n4_l1563_156367

theorem min_m_for_24m_eq_n4 (m n : ℕ+) (h : 24 * m = n ^ 4) :
  ∀ k : ℕ+, 24 * k = (k : ℕ+) ^ 4 → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_min_m_for_24m_eq_n4_l1563_156367


namespace NUMINAMATH_CALUDE_least_positive_solution_congruence_l1563_156316

theorem least_positive_solution_congruence :
  ∃! x : ℕ+, x.val + 7813 ≡ 2500 [ZMOD 15] ∧
  ∀ y : ℕ+, y.val + 7813 ≡ 2500 [ZMOD 15] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_solution_congruence_l1563_156316


namespace NUMINAMATH_CALUDE_triangle_angle_c_l1563_156350

theorem triangle_angle_c (A B C : ℝ) (m n : ℝ × ℝ) : 
  0 < C ∧ C < π →
  A + B + C = π →
  m = (Real.sqrt 3 * Real.sin A, Real.sin B) →
  n = (Real.cos B, Real.sqrt 3 * Real.cos A) →
  m.1 * n.1 + m.2 * n.2 = 1 + Real.cos (A + B) →
  C = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_c_l1563_156350


namespace NUMINAMATH_CALUDE_f_properties_l1563_156364

def f (x : ℕ) : ℕ := x % 2

def g (x : ℕ) : ℕ := x % 3

theorem f_properties :
  (∀ x : ℕ, f (2 * x) = 0) ∧
  (∀ x : ℕ, f x + f (x + 3) = 1) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1563_156364


namespace NUMINAMATH_CALUDE_brady_hours_june_l1563_156373

def hours_per_day_april : ℝ := 6
def hours_per_day_september : ℝ := 8
def average_hours_per_month : ℝ := 190
def days_per_month : ℕ := 30

theorem brady_hours_june :
  ∃ (hours_per_day_june : ℝ),
    hours_per_day_june * days_per_month +
    hours_per_day_april * days_per_month +
    hours_per_day_september * days_per_month =
    average_hours_per_month * 3 ∧
    hours_per_day_june = 5 := by
  sorry

end NUMINAMATH_CALUDE_brady_hours_june_l1563_156373


namespace NUMINAMATH_CALUDE_range_of_x_l1563_156398

theorem range_of_x (x : ℝ) : 4 * x - 12 ≥ 0 → x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1563_156398


namespace NUMINAMATH_CALUDE_plane_parallel_from_skew_lines_l1563_156393

-- Define the types for planes and lines
variable (α β : Plane) (L m : Line)

-- Define the parallel relation between lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the parallel relation between planes
def parallel_plane (p1 p2 : Plane) : Prop := sorry

-- Define skew lines
def skew_lines (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem plane_parallel_from_skew_lines 
  (h_skew : skew_lines L m) 
  (h_L_alpha : parallel_line_plane L α)
  (h_m_alpha : parallel_line_plane m α)
  (h_L_beta : parallel_line_plane L β)
  (h_m_beta : parallel_line_plane m β) :
  parallel_plane α β := sorry

end NUMINAMATH_CALUDE_plane_parallel_from_skew_lines_l1563_156393


namespace NUMINAMATH_CALUDE_range_of_a_for_two_roots_l1563_156389

theorem range_of_a_for_two_roots (a : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (∃ x y : ℝ, x ≠ y ∧ a^x = x ∧ a^y = y) → 
  1 < a ∧ a < Real.exp (1 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_roots_l1563_156389


namespace NUMINAMATH_CALUDE_scatter_plot_placement_l1563_156349

/-- Represents a variable in a scatter plot --/
inductive Variable
| Explanatory
| Forecast

/-- Represents an axis in a scatter plot --/
inductive Axis
| X
| Y

/-- Defines the relationship between variables and their roles in regression analysis --/
def is_independent (v : Variable) : Prop :=
  match v with
  | Variable.Explanatory => true
  | Variable.Forecast => false

/-- Defines the correct placement of variables on axes in a scatter plot --/
def correct_placement (v : Variable) (a : Axis) : Prop :=
  (v = Variable.Explanatory ∧ a = Axis.X) ∨ (v = Variable.Forecast ∧ a = Axis.Y)

/-- Theorem stating the correct placement of variables in a scatter plot for regression analysis --/
theorem scatter_plot_placement :
  ∀ (v : Variable) (a : Axis),
    is_independent v ↔ correct_placement v Axis.X :=
by sorry

end NUMINAMATH_CALUDE_scatter_plot_placement_l1563_156349


namespace NUMINAMATH_CALUDE_modular_inverse_12_mod_997_l1563_156315

theorem modular_inverse_12_mod_997 : ∃ x : ℤ, 12 * x ≡ 1 [ZMOD 997] :=
by
  use 914
  sorry

end NUMINAMATH_CALUDE_modular_inverse_12_mod_997_l1563_156315


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l1563_156311

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 200) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + a * c) = 1875 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l1563_156311


namespace NUMINAMATH_CALUDE_solve_pretzel_problem_l1563_156368

def pretzel_problem (barry_pretzels : ℕ) : Prop :=
  let shelly_pretzels : ℕ := barry_pretzels / 2
  let angie_pretzels : ℕ := 3 * shelly_pretzels
  let dave_pretzels : ℕ := (angie_pretzels + shelly_pretzels) / 4
  let total_pretzels : ℕ := barry_pretzels + shelly_pretzels + angie_pretzels + dave_pretzels
  let price_per_pretzel : ℕ := 1
  let total_cost : ℕ := total_pretzels * price_per_pretzel
  (barry_pretzels = 12) →
  (total_cost = 42)

theorem solve_pretzel_problem :
  pretzel_problem 12 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_pretzel_problem_l1563_156368


namespace NUMINAMATH_CALUDE_expression_evaluation_l1563_156395

theorem expression_evaluation (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (3 * x + y / 3 + 3 * z)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹ + (3 * z)⁻¹) = (9 * x * y * z)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1563_156395


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l1563_156351

open Set

-- Define sets A and B
def A : Set ℝ := {x | -2 < x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < -1} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | x ≤ 3 ∨ x > 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l1563_156351


namespace NUMINAMATH_CALUDE_inequality_solution_l1563_156381

theorem inequality_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (x + 1) / (x - 2) + (x - 3) / (3 * x) ≥ 2 ↔ x ≤ -1/3 ∨ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1563_156381


namespace NUMINAMATH_CALUDE_game_probability_l1563_156380

/-- A game with 8 rounds where one person wins each round -/
structure Game where
  rounds : Nat
  alex_prob : ℚ
  mel_prob : ℚ
  chelsea_prob : ℚ

/-- The probability of a specific outcome in the game -/
def outcome_probability (g : Game) (alex_wins mel_wins chelsea_wins : Nat) : ℚ :=
  (g.alex_prob ^ alex_wins) * (g.mel_prob ^ mel_wins) * (g.chelsea_prob ^ chelsea_wins) *
  (Nat.choose g.rounds alex_wins).choose mel_wins

/-- The theorem to be proved -/
theorem game_probability (g : Game) :
  g.rounds = 8 →
  g.alex_prob = 1/2 →
  g.mel_prob = g.chelsea_prob →
  g.alex_prob + g.mel_prob + g.chelsea_prob = 1 →
  outcome_probability g 4 3 1 = 35/512 := by
  sorry

end NUMINAMATH_CALUDE_game_probability_l1563_156380


namespace NUMINAMATH_CALUDE_margie_change_theorem_l1563_156342

-- Define the problem parameters
def num_apples : ℕ := 5
def cost_per_apple : ℚ := 30 / 100  -- 30 cents in dollars
def discount_rate : ℚ := 10 / 100   -- 10% discount
def paid_amount : ℚ := 10           -- 10-dollar bill

-- Define the theorem
theorem margie_change_theorem :
  let total_cost := num_apples * cost_per_apple
  let discounted_cost := total_cost * (1 - discount_rate)
  let change := paid_amount - discounted_cost
  change = 865 / 100 := by
sorry


end NUMINAMATH_CALUDE_margie_change_theorem_l1563_156342


namespace NUMINAMATH_CALUDE_fewer_noodles_than_pirates_l1563_156377

theorem fewer_noodles_than_pirates (noodles pirates : ℕ) : 
  noodles < pirates →
  pirates = 45 →
  noodles + pirates = 83 →
  pirates - noodles = 7 := by
sorry

end NUMINAMATH_CALUDE_fewer_noodles_than_pirates_l1563_156377


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1563_156357

theorem quadratic_equation_solution :
  ∃ (x₁ x₂ : ℝ),
    (x₁ * (5 * x₁ - 11) = 2) ∧
    (x₂ * (5 * x₂ - 11) = 2) ∧
    (x₁ = (11 + Real.sqrt 161) / 10) ∧
    (x₂ = (11 - Real.sqrt 161) / 10) ∧
    (Nat.gcd 11 (Nat.gcd 161 10) = 1) ∧
    (11 + 161 + 10 = 182) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1563_156357


namespace NUMINAMATH_CALUDE_tylers_age_l1563_156392

/-- Given the ages of Tyler (T), his brother (B), and their sister (S),
    prove that Tyler's age is 5 years old. -/
theorem tylers_age (T B S : ℕ) : 
  T = B - 3 → 
  S = B + 2 → 
  S = 2 * T → 
  T + B + S = 30 → 
  T = 5 := by
  sorry

end NUMINAMATH_CALUDE_tylers_age_l1563_156392


namespace NUMINAMATH_CALUDE_problem_statement_l1563_156313

theorem problem_statement (x y : ℝ) (h : x + 2*y - 1 = 0) : 3 + 2*x + 4*y = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1563_156313


namespace NUMINAMATH_CALUDE_cab_delay_l1563_156341

theorem cab_delay (S : ℝ) (h : S > 0) : 
  let reduced_speed := (5 / 6) * S
  let usual_time := 30
  let new_time := usual_time * (S / reduced_speed)
  new_time - usual_time = 6 := by
sorry

end NUMINAMATH_CALUDE_cab_delay_l1563_156341


namespace NUMINAMATH_CALUDE_opposite_solutions_imply_a_l1563_156318

theorem opposite_solutions_imply_a (a : ℝ) : 
  (∃ x y : ℝ, 2 * (x - 1) - 6 = 0 ∧ 1 - (3 * a - x) / 3 = 0 ∧ x = -y) → 
  a = -1/3 := by
sorry

end NUMINAMATH_CALUDE_opposite_solutions_imply_a_l1563_156318


namespace NUMINAMATH_CALUDE_four_mutually_tangent_circles_exist_l1563_156310

-- Define a circle with a center point and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles being externally tangent
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Theorem statement
theorem four_mutually_tangent_circles_exist : 
  ∃ (c1 c2 c3 c4 : Circle),
    are_externally_tangent c1 c2 ∧
    are_externally_tangent c1 c3 ∧
    are_externally_tangent c1 c4 ∧
    are_externally_tangent c2 c3 ∧
    are_externally_tangent c2 c4 ∧
    are_externally_tangent c3 c4 :=
sorry

end NUMINAMATH_CALUDE_four_mutually_tangent_circles_exist_l1563_156310


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l1563_156304

theorem wire_cutting_problem (piece_length : ℕ) : 
  piece_length > 0 ∧
  9 * piece_length ≤ 1000 ∧
  9 * piece_length ≤ 1100 ∧
  10 * piece_length > 1100 →
  piece_length = 111 :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l1563_156304


namespace NUMINAMATH_CALUDE_min_value_theorem_l1563_156360

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  3*x + 4*y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 5*x₀*y₀ ∧ 3*x₀ + 4*y₀ = 5 :=
by sorry


end NUMINAMATH_CALUDE_min_value_theorem_l1563_156360


namespace NUMINAMATH_CALUDE_f_properties_l1563_156314

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -a * x + x + a

-- Define the open interval (0,1]
def openUnitInterval : Set ℝ := { x | 0 < x ∧ x ≤ 1 }

-- Theorem statement
theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∀ x ∈ openUnitInterval, ∀ y ∈ openUnitInterval, x < y → f a x < f a y) ↔ (0 < a ∧ a ≤ 1) ∧
  (∃ M : ℝ, M = 1 ∧ ∀ x ∈ openUnitInterval, f a x ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1563_156314


namespace NUMINAMATH_CALUDE_new_person_age_l1563_156362

theorem new_person_age (T : ℕ) : 
  (T / 10 : ℚ) - 3 = ((T - 40 + 10) / 10 : ℚ) → 10 = 10 := by
sorry

end NUMINAMATH_CALUDE_new_person_age_l1563_156362


namespace NUMINAMATH_CALUDE_count_primes_with_squares_between_5000_and_8000_eq_5_l1563_156305

/-- The count of prime numbers whose squares are between 5000 and 8000 -/
def count_primes_with_squares_between_5000_and_8000 : Nat :=
  (Finset.filter (fun p => 5000 < p * p ∧ p * p < 8000) (Finset.filter Nat.Prime (Finset.range 90))).card

/-- Theorem stating that the count of prime numbers with squares between 5000 and 8000 is 5 -/
theorem count_primes_with_squares_between_5000_and_8000_eq_5 :
  count_primes_with_squares_between_5000_and_8000 = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_primes_with_squares_between_5000_and_8000_eq_5_l1563_156305


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l1563_156317

theorem triangle_perimeter_bound (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_angle_B : B = π / 3) (h_side_b : b = 2 * Real.sqrt 3) :
  a + b + c ≤ 6 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l1563_156317


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l1563_156399

/-- The ratio of the area of the inscribed circle to the area of a right triangle -/
theorem inscribed_circle_area_ratio (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let a := (3 / 5) * h
  let b := (4 / 5) * h
  let triangle_area := (1 / 2) * a * b
  let circle_area := π * r^2
  circle_area / triangle_area = (5 * π * r) / (12 * h) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l1563_156399


namespace NUMINAMATH_CALUDE_teacher_age_l1563_156330

/-- Given a class of students and their teacher, proves the teacher's age based on average ages. -/
theorem teacher_age (num_students : ℕ) (student_avg_age teacher_age : ℝ) (total_avg_age : ℝ) :
  num_students = 20 →
  student_avg_age = 15 →
  total_avg_age = 16 →
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = total_avg_age →
  teacher_age = 36 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l1563_156330


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l1563_156339

-- Define the circles O₁ and O₂
def O₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def O₂ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define the center of circle C
structure CircleCenter where
  x : ℝ
  y : ℝ

-- Define the property of being externally tangent to O₁ and internally tangent to O₂
def is_tangent_to_O₁_and_O₂ (c : CircleCenter) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    ((c.x - 1)^2 + c.y^2 = (r + 1)^2) ∧
    ((c.x + 1)^2 + c.y^2 = (4 - r)^2)

-- Define the locus of centers of circle C
def locus : Set CircleCenter :=
  {c : CircleCenter | is_tangent_to_O₁_and_O₂ c}

-- Theorem stating that the locus is an ellipse
theorem locus_is_ellipse :
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧
    ∀ c : CircleCenter, c ∈ locus ↔
      (c.x - h)^2 / a^2 + (c.y - k)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l1563_156339


namespace NUMINAMATH_CALUDE_jesse_remaining_money_l1563_156331

/-- Represents the currency exchange rates -/
structure ExchangeRates where
  usd_to_gbp : ℝ
  gbp_to_eur : ℝ

/-- Represents Jesse's shopping expenses -/
structure ShoppingExpenses where
  novel_price : ℝ
  novel_count : ℕ
  novel_discount : ℝ
  lunch_multiplier : ℕ
  lunch_tax : ℝ
  lunch_tip : ℝ
  jacket_price : ℝ
  jacket_discount : ℝ

/-- Calculates Jesse's remaining money after shopping -/
def remaining_money (initial_amount : ℝ) (rates : ExchangeRates) (expenses : ShoppingExpenses) : ℝ :=
  sorry

/-- Theorem stating that Jesse's remaining money is $174.66 -/
theorem jesse_remaining_money :
  let rates := ExchangeRates.mk (1/0.7) 1.15
  let expenses := ShoppingExpenses.mk 13 10 0.2 3 0.12 0.18 120 0.3
  remaining_money 500 rates expenses = 174.66 := by sorry

end NUMINAMATH_CALUDE_jesse_remaining_money_l1563_156331


namespace NUMINAMATH_CALUDE_gold_bars_calculation_l1563_156346

theorem gold_bars_calculation (initial_bars : ℕ) (tax_rate : ℚ) (divorce_loss_fraction : ℚ) : 
  initial_bars = 60 →
  tax_rate = 1/10 →
  divorce_loss_fraction = 1/2 →
  initial_bars * (1 - tax_rate) * (1 - divorce_loss_fraction) = 27 := by
  sorry

end NUMINAMATH_CALUDE_gold_bars_calculation_l1563_156346


namespace NUMINAMATH_CALUDE_intersects_implies_a_in_range_l1563_156325

/-- A function f(x) that always intersects the x-axis -/
def f (m a x : ℝ) : ℝ := m * (x^2 - 1) + x - a

/-- The property that f(x) always intersects the x-axis for all m -/
def always_intersects (a : ℝ) : Prop :=
  ∀ m : ℝ, ∃ x : ℝ, f m a x = 0

/-- Theorem: If f(x) always intersects the x-axis for all m, then a is in [-1, 1] -/
theorem intersects_implies_a_in_range (a : ℝ) :
  always_intersects a → a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_intersects_implies_a_in_range_l1563_156325


namespace NUMINAMATH_CALUDE_initial_leaves_count_l1563_156347

/-- The number of leaves that blew away -/
def leaves_blown_away : ℕ := 244

/-- The number of leaves left -/
def leaves_left : ℕ := 112

/-- The initial number of leaves -/
def initial_leaves : ℕ := leaves_blown_away + leaves_left

theorem initial_leaves_count : initial_leaves = 356 := by
  sorry

end NUMINAMATH_CALUDE_initial_leaves_count_l1563_156347


namespace NUMINAMATH_CALUDE_williams_hot_dogs_left_l1563_156390

/-- Calculates the number of hot dogs left after selling in two periods -/
def hot_dogs_left (initial : ℕ) (sold_first : ℕ) (sold_second : ℕ) : ℕ :=
  initial - (sold_first + sold_second)

/-- Theorem stating that for William's hot dog sales, 45 hot dogs were left -/
theorem williams_hot_dogs_left : hot_dogs_left 91 19 27 = 45 := by
  sorry

end NUMINAMATH_CALUDE_williams_hot_dogs_left_l1563_156390


namespace NUMINAMATH_CALUDE_lilly_fish_count_l1563_156343

/-- Given that Rosy has 12 fish and the total number of fish is 22,
    prove that Lilly has 10 fish. -/
theorem lilly_fish_count (rosy_fish : ℕ) (total_fish : ℕ) (h1 : rosy_fish = 12) (h2 : total_fish = 22) :
  total_fish - rosy_fish = 10 := by
  sorry

end NUMINAMATH_CALUDE_lilly_fish_count_l1563_156343


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1563_156344

theorem expansion_coefficient (n : ℕ) : 
  (Nat.choose n 2) * 9 = 54 → n = 4 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1563_156344


namespace NUMINAMATH_CALUDE_soccer_team_selection_l1563_156379

theorem soccer_team_selection (n m k : ℕ) (h1 : n = 18) (h2 : m = 7) (h3 : k = 2) :
  (Nat.choose (n - k) (m - k)) = (Nat.choose 16 5) :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_selection_l1563_156379


namespace NUMINAMATH_CALUDE_sally_has_six_cards_l1563_156332

/-- The number of baseball cards Sally has after selling some to Sara -/
def sallys_remaining_cards (initial_cards torn_cards cards_sold : ℕ) : ℕ :=
  initial_cards - torn_cards - cards_sold

/-- Theorem stating that Sally has 6 cards remaining -/
theorem sally_has_six_cards :
  sallys_remaining_cards 39 9 24 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_has_six_cards_l1563_156332


namespace NUMINAMATH_CALUDE_min_distinct_values_l1563_156369

theorem min_distinct_values (list_size : ℕ) (mode_count : ℕ) (min_distinct : ℕ) : 
  list_size = 3045 →
  mode_count = 15 →
  min_distinct = 218 →
  (∀ n : ℕ, n < min_distinct → 
    n * (mode_count - 1) + mode_count < list_size) ∧
  min_distinct * (mode_count - 1) + mode_count ≥ list_size :=
by sorry

end NUMINAMATH_CALUDE_min_distinct_values_l1563_156369


namespace NUMINAMATH_CALUDE_liar_count_theorem_l1563_156388

/-- Represents the type of islander: knight or liar -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents a group of islanders making a statement -/
structure IslanderGroup where
  size : Nat
  statement : Nat

/-- The problem setup -/
def islanderProblem : List IslanderGroup :=
  [⟨2, 2⟩, ⟨4, 4⟩, ⟨8, 8⟩, ⟨14, 14⟩]

/-- The total number of islanders -/
def totalIslanders : Nat := 28

/-- Function to determine if a statement is true given the actual number of liars -/
def isStatementTrue (group : IslanderGroup) (actualLiars : Nat) : Bool :=
  group.statement == actualLiars

/-- Function to determine the type of an islander based on their statement and the actual number of liars -/
def determineType (group : IslanderGroup) (actualLiars : Nat) : IslanderType :=
  if isStatementTrue group actualLiars then IslanderType.Knight else IslanderType.Liar

/-- Theorem stating that the number of liars is either 14 or 28 -/
theorem liar_count_theorem :
  ∃ (liarCount : Nat), (liarCount = 14 ∨ liarCount = 28) ∧
  (∀ (group : IslanderGroup), group ∈ islanderProblem →
    (determineType group liarCount = IslanderType.Liar) = (group.size ≤ liarCount)) ∧
  (liarCount ≤ totalIslanders) := by
  sorry

end NUMINAMATH_CALUDE_liar_count_theorem_l1563_156388


namespace NUMINAMATH_CALUDE_constant_function_from_square_plus_k_l1563_156383

/-- A continuous function satisfying f(x) = f(x² + k) for non-negative k is constant. -/
theorem constant_function_from_square_plus_k 
  (f : ℝ → ℝ) (hf : Continuous f) (k : ℝ) (hk : k ≥ 0) 
  (h : ∀ x, f x = f (x^2 + k)) : 
  ∃ C, ∀ x, f x = C :=
sorry

end NUMINAMATH_CALUDE_constant_function_from_square_plus_k_l1563_156383


namespace NUMINAMATH_CALUDE_parallel_lines_equation_l1563_156359

/-- A line in 2D space represented by its slope-intercept form -/
structure Line where
  slope : ℚ
  yIntercept : ℚ

/-- Distance between two parallel lines -/
def distanceBetweenParallelLines (l1 l2 : Line) : ℚ :=
  sorry

/-- Checks if two lines are parallel -/
def areParallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem parallel_lines_equation (l : Line) (P : ℚ × ℚ) (m : Line) :
  l.slope = -3/4 →
  P = (-2, 5) →
  areParallel l m →
  distanceBetweenParallelLines l m = 3 →
  (∃ (c : ℚ), (3 * P.1 + 4 * P.2 + c = 0 ∧ (c = 1 ∨ c = -29))) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_equation_l1563_156359


namespace NUMINAMATH_CALUDE_ben_spending_l1563_156354

-- Define the prices and quantities
def apple_price : ℚ := 2
def apple_quantity : ℕ := 7
def milk_price : ℚ := 4
def milk_quantity : ℕ := 4
def bread_price : ℚ := 3
def bread_quantity : ℕ := 3
def sugar_price : ℚ := 6
def sugar_quantity : ℕ := 3

-- Define the discounts
def dairy_discount : ℚ := 0.25
def coupon_discount : ℚ := 10
def coupon_threshold : ℚ := 50

-- Define the total spending function
def total_spending : ℚ :=
  let apple_cost := apple_price * apple_quantity
  let milk_cost := milk_price * milk_quantity * (1 - dairy_discount)
  let bread_cost := bread_price * bread_quantity
  let sugar_cost := sugar_price * sugar_quantity
  let subtotal := apple_cost + milk_cost + bread_cost + sugar_cost
  if subtotal ≥ coupon_threshold then subtotal - coupon_discount else subtotal

-- Theorem to prove
theorem ben_spending :
  total_spending = 43 :=
sorry

end NUMINAMATH_CALUDE_ben_spending_l1563_156354


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l1563_156384

/-- An ellipse with center at the origin, foci at (±√2, 0), intersected by the line y = x + 1
    such that the x-coordinate of the midpoint of the chord is -2/3 -/
def special_ellipse (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (a^2 = b^2 + 2) ∧
  (∃ (x₁ x₂ y₁ y₂ : ℝ),
    (x₁^2 / a^2 + y₁^2 / b^2 = 1) ∧
    (x₂^2 / a^2 + y₂^2 / b^2 = 1) ∧
    (y₁ = x₁ + 1) ∧ (y₂ = x₂ + 1) ∧
    ((x₁ + x₂) / 2 = -2/3))

/-- The equation of the special ellipse is x²/4 + y²/2 = 1 -/
theorem special_ellipse_equation :
  ∀ x y : ℝ, special_ellipse x y ↔ x^2/4 + y^2/2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l1563_156384


namespace NUMINAMATH_CALUDE_baseball_game_earnings_l1563_156336

theorem baseball_game_earnings (total : ℝ) (difference : ℝ) (wednesday : ℝ) (sunday : ℝ)
  (h1 : total = 4994.50)
  (h2 : difference = 1330.50)
  (h3 : wednesday + sunday = total)
  (h4 : wednesday = sunday - difference) :
  wednesday = 1832 := by
sorry

end NUMINAMATH_CALUDE_baseball_game_earnings_l1563_156336


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1563_156374

open Set

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {1, 2, 3, 4, 5}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {1, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1563_156374


namespace NUMINAMATH_CALUDE_two_boys_three_girls_probability_l1563_156324

-- Define the number of children
def n : ℕ := 5

-- Define the number of boys
def k : ℕ := 2

-- Define the probability of having a boy
def p : ℚ := 1/2

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := sorry

-- Define the probability function
def probability (n k : ℕ) (p : ℚ) : ℚ := sorry

-- Theorem statement
theorem two_boys_three_girls_probability :
  probability n k p = 0.3125 := by sorry

end NUMINAMATH_CALUDE_two_boys_three_girls_probability_l1563_156324


namespace NUMINAMATH_CALUDE_selection_schemes_count_l1563_156387

/-- The number of boys in the selection pool -/
def num_boys : ℕ := 4

/-- The number of girls in the selection pool -/
def num_girls : ℕ := 3

/-- The total number of volunteers to be selected -/
def num_volunteers : ℕ := 4

/-- Function to calculate the number of ways to select volunteers -/
def selection_schemes : ℕ := sorry

/-- Theorem stating that the number of selection schemes is 25 -/
theorem selection_schemes_count : selection_schemes = 25 := by sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l1563_156387


namespace NUMINAMATH_CALUDE_dice_stack_top_bottom_sum_l1563_156397

/-- Represents a standard die -/
structure StandardDie :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ (f : Fin 6), faces f + faces (5 - f) = 7)

/-- Represents a stack of two standard dice -/
structure DiceStack :=
  (top : StandardDie)
  (bottom : StandardDie)
  (touching_sum : ∃ (f1 f2 : Fin 6), top.faces f1 + bottom.faces f2 = 5)

/-- Theorem: The sum of pips on the top and bottom faces of a dice stack is 9 -/
theorem dice_stack_top_bottom_sum (stack : DiceStack) : 
  ∃ (f1 f2 : Fin 6), stack.top.faces f1 + stack.bottom.faces f2 = 9 :=
sorry

end NUMINAMATH_CALUDE_dice_stack_top_bottom_sum_l1563_156397


namespace NUMINAMATH_CALUDE_karen_pickup_cases_l1563_156303

/-- The number of boxes Karen sold -/
def boxes_sold : ℕ := 36

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 12

/-- The number of cases Karen needs to pick up -/
def cases_to_pickup : ℕ := boxes_sold / boxes_per_case

theorem karen_pickup_cases : cases_to_pickup = 3 := by
  sorry

end NUMINAMATH_CALUDE_karen_pickup_cases_l1563_156303
