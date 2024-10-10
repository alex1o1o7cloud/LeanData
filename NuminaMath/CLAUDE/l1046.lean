import Mathlib

namespace octahedron_side_length_l1046_104699

/-- A unit cube in 3D space -/
structure UnitCube where
  A : ℝ × ℝ × ℝ := (0, 0, 0)
  A' : ℝ × ℝ × ℝ := (1, 1, 1)

/-- A regular octahedron inscribed in a unit cube -/
structure InscribedOctahedron (cube : UnitCube) where
  vertices : List (ℝ × ℝ × ℝ)

/-- The side length of an inscribed octahedron -/
def sideLength (octahedron : InscribedOctahedron cube) : ℝ :=
  sorry

/-- Theorem: The side length of the inscribed octahedron is √2/3 -/
theorem octahedron_side_length (cube : UnitCube) 
  (octahedron : InscribedOctahedron cube) : 
  sideLength octahedron = Real.sqrt 2 / 3 := by
  sorry

end octahedron_side_length_l1046_104699


namespace v3_equals_55_l1046_104622

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 3x^5 + 8x^4 - 3x^3 + 5x^2 + 12x - 6 -/
def f : List ℤ := [3, 8, -3, 5, 12, -6]

/-- Theorem: V_3 equals 55 when x = 2 for the given polynomial using Horner's method -/
theorem v3_equals_55 : horner f 2 = 55 := by
  sorry

end v3_equals_55_l1046_104622


namespace restaurant_students_l1046_104621

theorem restaurant_students (burgers hot_dogs pizza_slices sandwiches : ℕ) : 
  burgers = 30 ∧ 
  burgers = 2 * hot_dogs ∧ 
  pizza_slices = hot_dogs + 5 ∧ 
  sandwiches = 3 * pizza_slices → 
  burgers + hot_dogs + pizza_slices + sandwiches = 125 := by
sorry

end restaurant_students_l1046_104621


namespace x_squared_mod_20_l1046_104682

theorem x_squared_mod_20 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 7 * x ≡ 14 [ZMOD 20]) : 
  x^2 ≡ 4 [ZMOD 20] := by
  sorry

end x_squared_mod_20_l1046_104682


namespace cos_105_degrees_l1046_104641

theorem cos_105_degrees : Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_degrees_l1046_104641


namespace marble_bag_problem_l1046_104600

theorem marble_bag_problem (r b : ℕ) : 
  (r - 1 : ℚ) / (r + b - 2 : ℚ) = 1/8 →
  (r : ℚ) / (r + b - 3 : ℚ) = 1/4 →
  r + b = 9 := by
  sorry

end marble_bag_problem_l1046_104600


namespace find_point_c_l1046_104633

/-- Given two points A and B in a 2D plane, and a point C such that 
    vector BC is half of vector BA, find the coordinates of point C. -/
theorem find_point_c (A B : ℝ × ℝ) (C : ℝ × ℝ) : 
  A = (1, 1) → 
  B = (-1, 2) → 
  C - B = (1/2) • (A - B) → 
  C = (0, 3/2) := by
sorry

end find_point_c_l1046_104633


namespace intersection_sum_l1046_104601

/-- Given two lines that intersect at (3,3), prove that a + b = 4 -/
theorem intersection_sum (a b : ℝ) : 
  (3 = (1/3) * 3 + a) → -- First line passes through (3,3)
  (3 = (1/3) * 3 + b) → -- Second line passes through (3,3)
  a + b = 4 := by
sorry

end intersection_sum_l1046_104601


namespace parabola_fixed_y_coordinate_l1046_104679

/-- A parabola that intersects the x-axis at only one point and passes through two specific points has a fixed y-coordinate for those points. -/
theorem parabola_fixed_y_coordinate (b c m n : ℝ) : 
  (∃ x, x^2 + b*x + c = 0 ∧ ∀ y, y ≠ x → y^2 + b*y + c ≠ 0) →  -- Parabola intersects x-axis at only one point
  (m^2 + b*m + c = n) →                                       -- Point (m, n) is on the parabola
  ((m-8)^2 + b*(m-8) + c = n) →                               -- Point (m-8, n) is on the parabola
  n = 16 := by
sorry

end parabola_fixed_y_coordinate_l1046_104679


namespace distance_for_equilateral_hyperbola_locus_l1046_104636

/-- Two circles C1 and C2 with variable tangent t to C1 intersecting C2 at A and B.
    Tangents to C2 through A and B intersect at P. -/
structure TwoCirclesConfig where
  r1 : ℝ  -- radius of C1
  r2 : ℝ  -- radius of C2
  d : ℝ   -- distance between centers of C1 and C2

/-- The locus of P is contained in an equilateral hyperbola -/
def isEquilateralHyperbolaLocus (config : TwoCirclesConfig) : Prop :=
  config.d = config.r1 * Real.sqrt 2

/-- Theorem: The distance between centers for equilateral hyperbola locus -/
theorem distance_for_equilateral_hyperbola_locus 
  (config : TwoCirclesConfig) (h1 : config.r1 > 0) (h2 : config.r2 > 0) :
  isEquilateralHyperbolaLocus config ↔ config.d = config.r1 * Real.sqrt 2 :=
sorry

end distance_for_equilateral_hyperbola_locus_l1046_104636


namespace hyperbola_circumradius_l1046_104674

/-- The hyperbola in the xy-plane -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- A point on the hyperbola -/
def P : ℝ × ℝ := sorry

/-- The centroid of triangle F₁PF₂ -/
def G : ℝ × ℝ := sorry

/-- The incenter of triangle F₁PF₂ -/
def I : ℝ × ℝ := sorry

/-- The circumradius of triangle F₁PF₂ -/
def R : ℝ := sorry

theorem hyperbola_circumradius :
  hyperbola P.1 P.2 ∧ 
  (G.2 = I.2) →  -- GI is parallel to x-axis
  R = 5 := by sorry

end hyperbola_circumradius_l1046_104674


namespace unique_solution_k_values_l1046_104604

/-- The set of values for k that satisfy the given conditions -/
def k_values : Set ℝ := {1 + Real.sqrt 2, (1 - Real.sqrt 5) / 2}

/-- The system of inequalities -/
def system (k x : ℝ) : Prop :=
  1 ≤ k * x^2 + 2 ∧ x + k ≤ 2

/-- The main theorem stating that k_values is the correct set of values for k -/
theorem unique_solution_k_values :
  ∀ k : ℝ, (∃! x : ℝ, system k x) ↔ k ∈ k_values := by
  sorry

#check unique_solution_k_values

end unique_solution_k_values_l1046_104604


namespace wire_cutting_problem_l1046_104619

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) : 
  total_length = 140 → ratio = 2/5 → 
  ∃ (shorter_piece longer_piece : ℝ), 
    shorter_piece + longer_piece = total_length ∧ 
    shorter_piece = ratio * longer_piece ∧
    shorter_piece = 40 := by
  sorry

end wire_cutting_problem_l1046_104619


namespace red_then_blue_probability_l1046_104625

/-- The probability of drawing a red marble first and a blue marble second from a jar -/
theorem red_then_blue_probability (red green white blue : ℕ) :
  red = 4 →
  green = 3 →
  white = 10 →
  blue = 2 →
  let total := red + green + white + blue
  let prob_red := red / total
  let prob_blue_after_red := blue / (total - 1)
  prob_red * prob_blue_after_red = 4 / 171 := by
  sorry

end red_then_blue_probability_l1046_104625


namespace p_iff_q_l1046_104673

-- Define the propositions
def p (a : ℝ) : Prop := a = -1

def q (a : ℝ) : Prop := ∀ (x y : ℝ), (a * x + y + 1 = 0) ↔ (x + a * y + 2 * a - 1 = 0)

-- State the theorem
theorem p_iff_q : ∀ (a : ℝ), p a ↔ q a := by sorry

end p_iff_q_l1046_104673


namespace distance_to_plane_l1046_104647

/-- The distance from a point to a plane defined by three points -/
def distancePointToPlane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₀, y₀, z₀) := M₀
  let (x₁, y₁, z₁) := M₁
  let (x₂, y₂, z₂) := M₂
  let (x₃, y₃, z₃) := M₃
  -- Implementation details omitted
  sorry

theorem distance_to_plane :
  let M₀ : ℝ × ℝ × ℝ := (1, -6, -5)
  let M₁ : ℝ × ℝ × ℝ := (-1, 2, -3)
  let M₂ : ℝ × ℝ × ℝ := (4, -1, 0)
  let M₃ : ℝ × ℝ × ℝ := (2, 1, -2)
  distancePointToPlane M₀ M₁ M₂ M₃ = 5 * Real.sqrt 2 := by
  sorry

end distance_to_plane_l1046_104647


namespace professors_age_l1046_104651

/-- Represents a four-digit number abac --/
def FourDigitNumber (a b c : Nat) : Nat :=
  1000 * a + 100 * b + 10 * a + c

/-- Represents a two-digit number ab --/
def TwoDigitNumber (a b : Nat) : Nat :=
  10 * a + b

theorem professors_age (a b c : Nat) (x : Nat) 
  (h1 : x^2 = FourDigitNumber a b c)
  (h2 : x = TwoDigitNumber a b + TwoDigitNumber a c) :
  x = 45 := by
sorry

end professors_age_l1046_104651


namespace unique_m_value_l1046_104664

def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

theorem unique_m_value (a m : ℝ) 
  (h1 : 0 < m) (h2 : m < 1)
  (h3 : ∀ x : ℝ, f x a m ≥ 2)
  (h4 : a ≤ -5 ∨ a ≥ 5) :
  m = 1/5 := by
sorry

end unique_m_value_l1046_104664


namespace fraction_inequality_l1046_104661

theorem fraction_inequality (a b : ℚ) (h : a / b = 2 / 3) :
  ¬(∀ (x y : ℚ), x / y = 2 / 3 → x / y = (x + 2) / (y + 2)) := by
  sorry

end fraction_inequality_l1046_104661


namespace transaction_result_l1046_104650

theorem transaction_result : 
  ∀ (house_cost store_cost : ℕ),
  (house_cost * 3 / 4 = 15000) →
  (store_cost * 5 / 4 = 10000) →
  (house_cost + store_cost) - (15000 + 10000) = 3000 :=
by
  sorry

end transaction_result_l1046_104650


namespace quadratic_inequality_solution_set_l1046_104655

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, -x^2 + 2*x + 3 ≥ 0 ↔ x ∈ Set.Icc (-1) 3 := by
sorry

end quadratic_inequality_solution_set_l1046_104655


namespace three_color_theorem_l1046_104654

/-- Represents a country on the island -/
structure Country where
  vertices : Fin 3 → ℝ × ℝ

/-- Represents the entire island -/
structure Island where
  countries : List Country
  adjacent : Country → Country → Bool

/-- A coloring of the island -/
def Coloring := Country → Fin 3

theorem three_color_theorem (I : Island) :
  ∃ (c : Coloring), ∀ (x y : Country),
    I.adjacent x y → c x ≠ c y :=
sorry

end three_color_theorem_l1046_104654


namespace linear_function_max_min_sum_l1046_104681

theorem linear_function_max_min_sum (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, a * x ≤ max (a * 0) (a * 1) ∧ min (a * 0) (a * 1) ≤ a * x) →
  max (a * 0) (a * 1) + min (a * 0) (a * 1) = 3 →
  a = 3 := by sorry

end linear_function_max_min_sum_l1046_104681


namespace bart_mixtape_length_l1046_104605

/-- Calculates the total length of a mixtape in minutes -/
def mixtape_length (first_side_songs : ℕ) (second_side_songs : ℕ) (song_length : ℕ) : ℕ :=
  (first_side_songs + second_side_songs) * song_length

/-- Proves that the total length of Bart's mixtape is 40 minutes -/
theorem bart_mixtape_length :
  mixtape_length 6 4 4 = 40 := by
  sorry

end bart_mixtape_length_l1046_104605


namespace expression_value_l1046_104606

theorem expression_value (x y : ℝ) (h : x = 2*y + 1) : x^2 - 4*x*y + 4*y^2 = 1 := by
  sorry

end expression_value_l1046_104606


namespace bird_count_l1046_104698

theorem bird_count (crows : ℕ) (hawk_percentage : ℚ) : 
  crows = 30 → 
  hawk_percentage = 60 / 100 → 
  crows + (crows + hawk_percentage * crows) = 78 := by
  sorry

end bird_count_l1046_104698


namespace infinite_coprime_pairs_l1046_104629

theorem infinite_coprime_pairs (m : ℕ+) :
  ∃ (seq : ℕ → ℕ × ℕ), ∀ n : ℕ,
    let (x, y) := seq n
    Int.gcd x y = 1 ∧
    x > 0 ∧ y > 0 ∧
    (y^2 + m.val) % x = 0 ∧
    (x^2 + m.val) % y = 0 ∧
    (∀ k < n, seq k ≠ seq n) :=
sorry

end infinite_coprime_pairs_l1046_104629


namespace exists_all_accessible_l1046_104611

-- Define the type for cities
variable {City : Type}

-- Define the accessibility relation
variable (accessible : City → City → Prop)

-- Define the property that a city can access itself
variable (self_accessible : ∀ c : City, accessible c c)

-- Define the property that for any two cities, there's a third city that can access both
variable (exists_common_accessible : ∀ p q : City, ∃ r : City, accessible p r ∧ accessible q r)

-- The theorem to prove
theorem exists_all_accessible :
  ∃ c : City, ∀ other : City, accessible other c :=
sorry

end exists_all_accessible_l1046_104611


namespace adam_first_half_correct_l1046_104612

/-- Represents the trivia game scenario -/
structure TriviaGame where
  pointsPerQuestion : ℕ
  secondHalfCorrect : ℕ
  finalScore : ℕ

/-- Calculates the number of correctly answered questions in the first half -/
def firstHalfCorrect (game : TriviaGame) : ℕ :=
  (game.finalScore - game.secondHalfCorrect * game.pointsPerQuestion) / game.pointsPerQuestion

/-- Theorem stating that Adam answered 8 questions correctly in the first half -/
theorem adam_first_half_correct :
  let game : TriviaGame := {
    pointsPerQuestion := 8,
    secondHalfCorrect := 2,
    finalScore := 80
  }
  firstHalfCorrect game = 8 := by sorry

end adam_first_half_correct_l1046_104612


namespace common_factor_proof_l1046_104671

theorem common_factor_proof (x y : ℝ) (m n : ℕ) :
  ∃ (k : ℝ), 8 * x^m * y^(n-1) - 12 * x^(3*m) * y^n = k * (4 * x^m * y^(n-1)) ∧
              k ≠ 0 :=
by
  sorry

end common_factor_proof_l1046_104671


namespace min_value_theorem_l1046_104603

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 24 / 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 5 * x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 24 / 5 :=
by sorry

end min_value_theorem_l1046_104603


namespace necessary_not_sufficient_l1046_104692

theorem necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, (a > 2 ∧ b > 2) → a + b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ ¬(a > 2 ∧ b > 2)) := by
  sorry

end necessary_not_sufficient_l1046_104692


namespace remainder_n_plus_2023_l1046_104646

theorem remainder_n_plus_2023 (n : ℤ) (h : n % 7 = 3) : (n + 2023) % 7 = 3 := by
  sorry

end remainder_n_plus_2023_l1046_104646


namespace nine_by_n_grid_rectangles_l1046_104637

theorem nine_by_n_grid_rectangles (n : ℕ) : 
  (9 : ℕ) > 1 → n > 1 → (Nat.choose 9 2 * Nat.choose n 2 = 756) → n = 7 := by
  sorry

end nine_by_n_grid_rectangles_l1046_104637


namespace trajectory_and_constant_slope_l1046_104630

noncomputable section

-- Define the points A and P
def A : ℝ × ℝ := (3, -6)
def P : ℝ × ℝ := (1, -2)

-- Define the curve
def on_curve (Q : ℝ × ℝ) : Prop :=
  let (x, y) := Q
  (x^2 + y^2) / ((x - 3)^2 + (y + 6)^2) = 1/4

-- Define complementary angles
def complementary_angles (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

-- Define the theorem
theorem trajectory_and_constant_slope :
  -- Part 1: Equation of the curve
  (∀ Q : ℝ × ℝ, on_curve Q ↔ (Q.1 + 1)^2 + (Q.2 - 2)^2 = 20) ∧
  -- Part 2: Constant slope of BC
  (∀ B C : ℝ × ℝ, 
    on_curve B ∧ on_curve C ∧ 
    (∃ m1 m2 : ℝ, 
      complementary_angles m1 m2 ∧
      (B.2 - P.2) = m1 * (B.1 - P.1) ∧
      (C.2 - P.2) = m2 * (C.1 - P.1)) →
    (C.2 - B.2) / (C.1 - B.1) = -1/2) :=
sorry

end

end trajectory_and_constant_slope_l1046_104630


namespace solve_quadratic_sets_l1046_104623

-- Define the sets A and B
def A (p : ℝ) : Set ℝ := {x | x^2 + p*x - 8 = 0}
def B (q r : ℝ) : Set ℝ := {x | x^2 - q*x + r = 0}

-- State the theorem
theorem solve_quadratic_sets :
  ∃ (p q r : ℝ),
    A p ≠ B q r ∧
    A p ∪ B q r = {-2, 4} ∧
    A p ∩ B q r = {-2} ∧
    p = -2 ∧ q = -4 ∧ r = 4 :=
by sorry

end solve_quadratic_sets_l1046_104623


namespace prob_select_one_from_couple_l1046_104644

/-- The probability of selecting exactly one person from a couple, given their individual selection probabilities -/
theorem prob_select_one_from_couple (p_husband p_wife : ℝ) 
  (h_husband : p_husband = 1/7)
  (h_wife : p_wife = 1/5) :
  p_husband * (1 - p_wife) + p_wife * (1 - p_husband) = 2/7 := by
  sorry

end prob_select_one_from_couple_l1046_104644


namespace contradiction_proof_l1046_104691

theorem contradiction_proof (x : ℝ) : (x^2 - 1 = 0) → (x = -1 ∨ x = 1) := by
  contrapose
  intro h
  have h1 : x ≠ -1 ∧ x ≠ 1 := by
    push_neg at h
    exact h
  sorry

end contradiction_proof_l1046_104691


namespace coat_price_reduction_l1046_104652

/-- Given a coat with an original price and a price reduction, 
    calculate the percent reduction. -/
theorem coat_price_reduction 
  (original_price : ℝ) 
  (price_reduction : ℝ) 
  (h1 : original_price = 500) 
  (h2 : price_reduction = 150) : 
  (price_reduction / original_price) * 100 = 30 := by
  sorry

end coat_price_reduction_l1046_104652


namespace melt_to_spend_ratio_is_80_l1046_104609

/-- The ratio of the value of melted quarters to spent quarters -/
def meltToSpendRatio : ℚ :=
  let quarterWeight : ℚ := 1 / 5
  let meltedValuePerOunce : ℚ := 100
  let spendingValuePerQuarter : ℚ := 1 / 4
  let quartersPerOunce : ℚ := 1 / quarterWeight
  let meltedValuePerQuarter : ℚ := meltedValuePerOunce / quartersPerOunce
  meltedValuePerQuarter / spendingValuePerQuarter

/-- The ratio of the value of melted quarters to spent quarters is 80 -/
theorem melt_to_spend_ratio_is_80 : meltToSpendRatio = 80 := by
  sorry

end melt_to_spend_ratio_is_80_l1046_104609


namespace fraction_before_lunch_l1046_104613

/-- Proves that the fraction of distance driven before lunch is 1/4 given the problem conditions --/
theorem fraction_before_lunch (total_distance : ℝ) (total_time : ℝ) (lunch_time : ℝ) 
  (h1 : total_distance = 200)
  (h2 : total_time = 5)
  (h3 : lunch_time = 1)
  (h4 : total_time ≥ lunch_time) :
  let f := (total_time - lunch_time) / 4 / (total_time - lunch_time)
  f = 1 / 4 := by
  sorry

end fraction_before_lunch_l1046_104613


namespace negation_equivalence_l1046_104631

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 > Real.exp x) ↔ (∀ x : ℝ, x^2 ≤ Real.exp x) := by
  sorry

end negation_equivalence_l1046_104631


namespace equal_numbers_product_l1046_104662

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 15 →
  a = 10 →
  b = 18 →
  c = d →
  c * d = 256 := by
sorry

end equal_numbers_product_l1046_104662


namespace inequality_proof_l1046_104640

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a + b + c) + Real.sqrt a) / (b + c) +
  (Real.sqrt (a + b + c) + Real.sqrt b) / (c + a) +
  (Real.sqrt (a + b + c) + Real.sqrt c) / (a + b) ≥
  (9 + 3 * Real.sqrt 3) / (2 * Real.sqrt (a + b + c)) :=
by sorry

end inequality_proof_l1046_104640


namespace divisibility_condition_l1046_104696

theorem divisibility_condition (M : ℕ) : 
  M > 0 ∧ M < 10 → (5 ∣ 1989^M + M^1989 ↔ M = 1 ∨ M = 4) := by
sorry

end divisibility_condition_l1046_104696


namespace lazy_kingdom_date_l1046_104610

-- Define the days of the week in the Lazy Kingdom
inductive LazyDay
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Saturday

-- Define a function to calculate the next day
def nextDay (d : LazyDay) : LazyDay :=
  match d with
  | LazyDay.Sunday => LazyDay.Monday
  | LazyDay.Monday => LazyDay.Tuesday
  | LazyDay.Tuesday => LazyDay.Wednesday
  | LazyDay.Wednesday => LazyDay.Thursday
  | LazyDay.Thursday => LazyDay.Saturday
  | LazyDay.Saturday => LazyDay.Sunday

-- Define a function to calculate the day after n days
def dayAfter (start : LazyDay) (n : Nat) : LazyDay :=
  match n with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

-- Theorem statement
theorem lazy_kingdom_date : 
  dayAfter LazyDay.Sunday 374 = LazyDay.Tuesday := by
  sorry


end lazy_kingdom_date_l1046_104610


namespace horner_method_v3_l1046_104615

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 3*x^2 + x + 1

def horner_v3 (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * x + 0
  let v2 := v1 * x + 2
  v2 * x + 3

theorem horner_method_v3 :
  horner_v3 f 3 = 36 := by sorry

end horner_method_v3_l1046_104615


namespace fruits_remaining_l1046_104608

theorem fruits_remaining (initial_apples : ℕ) (plum_ratio : ℚ) (picked_ratio : ℚ) : 
  initial_apples = 180 → 
  plum_ratio = 1 / 3 → 
  picked_ratio = 3 / 5 → 
  (initial_apples + (↑initial_apples * plum_ratio)) * (1 - picked_ratio) = 96 := by
sorry

end fruits_remaining_l1046_104608


namespace sequence_problem_l1046_104639

theorem sequence_problem (b : ℕ → ℝ) 
  (h1 : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h2 : b 1 = 3 + Real.sqrt 11)
  (h3 : b 1987 = 17 + Real.sqrt 11) :
  b 2015 = (3 - Real.sqrt 11) / 8 := by
sorry

end sequence_problem_l1046_104639


namespace millet_in_brand_b_l1046_104684

/-- Represents the composition of a bird seed brand -/
structure BirdSeed where
  millet : ℝ
  other : ℝ
  composition_sum : millet + other = 1

/-- Represents a mix of two bird seed brands -/
structure BirdSeedMix where
  brandA : BirdSeed
  brandB : BirdSeed
  proportionA : ℝ
  proportionB : ℝ
  mix_sum : proportionA + proportionB = 1

/-- Theorem stating the millet percentage in Brand B given the conditions -/
theorem millet_in_brand_b 
  (mix : BirdSeedMix)
  (brandA_millet : mix.brandA.millet = 0.6)
  (mix_proportionA : mix.proportionA = 0.6)
  (mix_millet : mix.proportionA * mix.brandA.millet + mix.proportionB * mix.brandB.millet = 0.5) :
  mix.brandB.millet = 0.35 := by
  sorry


end millet_in_brand_b_l1046_104684


namespace math_club_composition_l1046_104672

theorem math_club_composition :
  ∀ (initial_males initial_females : ℕ),
    initial_males = initial_females →
    (3 * (initial_males + initial_females - 1) = 4 * (initial_females - 1)) →
    initial_males = 2 ∧ initial_females = 3 := by
  sorry

end math_club_composition_l1046_104672


namespace tan_difference_implies_ratio_l1046_104648

theorem tan_difference_implies_ratio (α : Real) 
  (h : Real.tan (α - π/4) = 1/2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 := by
  sorry

end tan_difference_implies_ratio_l1046_104648


namespace decimal_89_to_base5_l1046_104618

/-- Converts a natural number to its base-5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Checks if a list of digits is a valid base-5 representation --/
def isValidBase5 (l : List ℕ) : Prop :=
  l.all (· < 5)

theorem decimal_89_to_base5 :
  let base5_representation := toBase5 89
  isValidBase5 base5_representation ∧ base5_representation = [4, 2, 3] :=
by sorry

end decimal_89_to_base5_l1046_104618


namespace visitor_increase_l1046_104669

theorem visitor_increase (original_fee : ℝ) (fee_reduction : ℝ) (sale_increase : ℝ) :
  original_fee = 1 →
  fee_reduction = 0.25 →
  sale_increase = 0.20 →
  let new_fee := original_fee * (1 - fee_reduction)
  let visitor_increase := (1 + sale_increase) / (1 - fee_reduction) - 1
  visitor_increase = 0.60 := by sorry

end visitor_increase_l1046_104669


namespace batch_composition_l1046_104667

/-- Represents the characteristics of a product type -/
structure ProductType where
  volume : ℝ  -- Volume per unit in m³
  mass : ℝ    -- Mass per unit in tons

/-- Represents a batch of products -/
structure Batch where
  typeA : ProductType
  typeB : ProductType
  totalVolume : ℝ
  totalMass : ℝ

/-- Theorem: Given the specific product characteristics and total volume and mass,
    prove that the batch consists of 5 units of type A and 8 units of type B -/
theorem batch_composition (b : Batch)
    (h1 : b.typeA.volume = 0.8)
    (h2 : b.typeA.mass = 0.5)
    (h3 : b.typeB.volume = 2)
    (h4 : b.typeB.mass = 1)
    (h5 : b.totalVolume = 20)
    (h6 : b.totalMass = 10.5) :
    ∃ (x y : ℝ), x = 5 ∧ y = 8 ∧
    x * b.typeA.volume + y * b.typeB.volume = b.totalVolume ∧
    x * b.typeA.mass + y * b.typeB.mass = b.totalMass :=
  sorry


end batch_composition_l1046_104667


namespace inequality_proof_l1046_104627

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 1/2 := by
  sorry

end inequality_proof_l1046_104627


namespace find_a_l1046_104620

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- State the theorem
theorem find_a : 
  ∃ (a : ℝ), (∀ x : ℝ, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) ∧ a = 2 := by
  sorry

end find_a_l1046_104620


namespace park_fencing_cost_l1046_104695

/-- Represents the dimensions and fencing costs of a park with a flower bed -/
structure ParkWithFlowerBed where
  park_ratio : Rat -- Ratio of park's length to width
  park_area : ℝ -- Area of the park in square meters
  park_fence_cost : ℝ -- Cost of fencing the park per meter
  flowerbed_fence_cost : ℝ -- Cost of fencing the flower bed per meter

/-- Calculates the total fencing cost for a park with a flower bed -/
def total_fencing_cost (p : ParkWithFlowerBed) : ℝ :=
  sorry

/-- Theorem stating the total fencing cost for the given park configuration -/
theorem park_fencing_cost :
  let p : ParkWithFlowerBed := {
    park_ratio := 3/2,
    park_area := 3750,
    park_fence_cost := 0.70,
    flowerbed_fence_cost := 0.90
  }
  total_fencing_cost p = 245.65 := by
  sorry

end park_fencing_cost_l1046_104695


namespace circle_equation_with_given_endpoints_l1046_104660

/-- The standard equation of a circle with diameter endpoints M(2,0) and N(0,4) -/
theorem circle_equation_with_given_endpoints :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 5 ↔ 
  ∃ (t : ℝ), x = 2 * (1 - t) + 0 * t ∧ y = 0 * (1 - t) + 4 * t ∧ 0 ≤ t ∧ t ≤ 1 :=
by sorry

end circle_equation_with_given_endpoints_l1046_104660


namespace larger_solution_of_quadratic_l1046_104632

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 42 = 0 ∧ x ≠ 6 → x = 7 := by
  sorry

end larger_solution_of_quadratic_l1046_104632


namespace floor_equation_solution_l1046_104663

theorem floor_equation_solution (n : ℤ) :
  (⌊n^2 / 3⌋ : ℤ) - (⌊n / 2⌋ : ℤ)^2 = 3 ↔ n = 6 :=
by sorry

end floor_equation_solution_l1046_104663


namespace visitor_increase_percentage_l1046_104686

/-- Represents the percentage increase in visitors after implementing discounts -/
def overallPercentageIncrease (initialChildren : ℕ) (initialSeniors : ℕ) (initialAdults : ℕ)
  (childrenIncrease : ℚ) (seniorsIncrease : ℚ) : ℚ :=
  let totalInitial := initialChildren + initialSeniors + initialAdults
  let totalAfter := 
    (initialChildren * (1 + childrenIncrease)) + 
    (initialSeniors * (1 + seniorsIncrease)) + 
    initialAdults
  (totalAfter - totalInitial) / totalInitial * 100

/-- Theorem stating that the overall percentage increase in visitors is approximately 13.33% -/
theorem visitor_increase_percentage : 
  ∀ (initialChildren initialSeniors initialAdults : ℕ),
  initialChildren > 0 → initialSeniors > 0 → initialAdults > 0 →
  let childrenIncrease : ℚ := 25 / 100
  let seniorsIncrease : ℚ := 15 / 100
  abs (overallPercentageIncrease initialChildren initialSeniors initialAdults childrenIncrease seniorsIncrease - 40 / 3) < 1 / 100 :=
by
  sorry

#eval overallPercentageIncrease 100 100 100 (25 / 100) (15 / 100)

end visitor_increase_percentage_l1046_104686


namespace first_nonzero_digit_of_1_137_l1046_104676

theorem first_nonzero_digit_of_1_137 :
  ∃ (n : ℕ) (k : ℕ), 
    10^n > 137 ∧ 
    (1000 : ℚ) / 137 = k + (1000 - k * 137 : ℚ) / 137 ∧ 
    k = 7 := by sorry

end first_nonzero_digit_of_1_137_l1046_104676


namespace uncool_parents_count_l1046_104665

/-- Proves the number of students with uncool parents in a music class -/
theorem uncool_parents_count (total : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ) 
  (h1 : total = 40)
  (h2 : cool_dads = 25)
  (h3 : cool_moms = 19)
  (h4 : both_cool = 8) :
  total - (cool_dads + cool_moms - both_cool) = 4 := by
  sorry

end uncool_parents_count_l1046_104665


namespace translation_problem_l1046_104694

def complex_translation (z w : ℂ) : ℂ := z + w

theorem translation_problem (t : ℂ → ℂ) :
  (t (1 + 3*I) = -2 + 4*I) →
  (∃ w : ℂ, ∀ z : ℂ, t z = complex_translation z w) →
  (t (3 + 7*I) = 8*I) :=
by
  sorry

end translation_problem_l1046_104694


namespace max_ab_l1046_104670

theorem max_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4*a + b = 1) :
  ab ≤ 1/16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4*a₀ + b₀ = 1 ∧ a₀*b₀ = 1/16 :=
sorry


end max_ab_l1046_104670


namespace ballot_marking_combinations_l1046_104687

theorem ballot_marking_combinations : 
  ∀ n : ℕ, n = 10 → n.factorial = 3628800 :=
by
  sorry

end ballot_marking_combinations_l1046_104687


namespace female_democrats_count_l1046_104635

def meeting_participants (total_participants : ℕ) 
  (female_participants : ℕ) (male_participants : ℕ) : Prop :=
  female_participants + male_participants = total_participants

def democrat_ratio (female_democrats : ℕ) (male_democrats : ℕ) 
  (female_participants : ℕ) (male_participants : ℕ) : Prop :=
  female_democrats = female_participants / 2 ∧ 
  male_democrats = male_participants / 4

def total_democrats (female_democrats : ℕ) (male_democrats : ℕ) 
  (total_participants : ℕ) : Prop :=
  female_democrats + male_democrats = total_participants / 3

theorem female_democrats_count : 
  ∀ (total_participants female_participants male_participants 
     female_democrats male_democrats : ℕ),
  total_participants = 990 →
  meeting_participants total_participants female_participants male_participants →
  democrat_ratio female_democrats male_democrats female_participants male_participants →
  total_democrats female_democrats male_democrats total_participants →
  female_democrats = 165 := by
  sorry

end female_democrats_count_l1046_104635


namespace bread_baking_time_l1046_104649

theorem bread_baking_time (rise_time bake_time : ℕ) (num_balls : ℕ) : 
  rise_time = 3 → 
  bake_time = 2 → 
  num_balls = 4 → 
  (rise_time * num_balls) + (bake_time * num_balls) = 20 :=
by sorry

end bread_baking_time_l1046_104649


namespace gcd_of_polynomial_and_multiple_l1046_104634

theorem gcd_of_polynomial_and_multiple (x : ℤ) : 
  (∃ k : ℤ, x = 11739 * k) → 
  Nat.gcd ((3*x + 4)*(5*x + 3)*(11*x + 5)*(x + 11)).natAbs x.natAbs = 3 := by
  sorry

end gcd_of_polynomial_and_multiple_l1046_104634


namespace intersection_values_l1046_104677

-- Define the function f(x) = ax² + (3-a)x + 1
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (3 - a) * x + 1

-- Define the condition for intersection with x-axis at only one point
def intersects_once (a : ℝ) : Prop :=
  ∃! x, f a x = 0

-- State the theorem
theorem intersection_values : 
  ∀ a : ℝ, intersects_once a ↔ a = 0 ∨ a = 1 ∨ a = 9 := by sorry

end intersection_values_l1046_104677


namespace larger_root_of_quadratic_l1046_104624

theorem larger_root_of_quadratic (x : ℝ) : 
  (x - 5/8) * (x - 5/8) + (x - 5/8) * (x - 1/3) = 0 →
  x = 5/8 ∨ x = 23/48 →
  (5/8 : ℝ) > (23/48 : ℝ) →
  x = 5/8 := by sorry

end larger_root_of_quadratic_l1046_104624


namespace price_per_book_is_two_l1046_104659

/-- Represents the sale of books with given conditions -/
def BookSale (total_books : ℕ) (price_per_book : ℚ) : Prop :=
  (2 : ℚ) / 3 * total_books + 36 = total_books ∧
  (2 : ℚ) / 3 * total_books * price_per_book = 144

/-- Theorem stating that the price per book is $2 given the conditions -/
theorem price_per_book_is_two :
  ∃ (total_books : ℕ), BookSale total_books 2 := by
  sorry

end price_per_book_is_two_l1046_104659


namespace coin_game_winning_strategy_l1046_104666

/-- Represents the state of the coin game -/
structure GameState where
  piles : List Nat
  deriving Repr

/-- Checks if a player has a winning strategy given the current game state -/
def hasWinningStrategy (state : GameState) : Prop :=
  let n1 := (state.piles.filter (· = 1)).length
  let evenPiles := state.piles.filter (· % 2 = 0)
  let sumEvenPiles := (evenPiles.map (λ x => x / 2)).sum
  Odd n1 ∨ Odd sumEvenPiles

/-- The main theorem stating the winning condition for the coin game -/
theorem coin_game_winning_strategy (state : GameState) :
  hasWinningStrategy state ↔ 
  Odd (state.piles.filter (· = 1)).length ∨ 
  Odd ((state.piles.filter (· % 2 = 0)).map (λ x => x / 2)).sum :=
by sorry


end coin_game_winning_strategy_l1046_104666


namespace parabola_focus_line_intersection_l1046_104656

/-- Parabola struct representing x^2 = 2py -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (par : Parabola) where
  x : ℝ
  y : ℝ
  h : x^2 = 2 * par.p * y

/-- Line passing through the focus of a parabola with slope √3 -/
structure FocusLine (par : Parabola) where
  slope : ℝ
  hslope : slope = Real.sqrt 3
  pass_focus : ℝ → ℝ
  hpass : pass_focus 0 = par.p / 2

theorem parabola_focus_line_intersection (par : Parabola) 
  (l : FocusLine par) (M N : ParabolaPoint par) 
  (hM : M.x > 0) (hMN : M.x ≠ N.x) : 
  (par.p = 2 → M.x * N.x = -4) ∧ 
  (M.y * N.y = 1 → par.p = 2) ∧ 
  (par.p = 2 → Real.sqrt ((M.x - 0)^2 + (M.y - par.p/2)^2) = 8 + 4 * Real.sqrt 3) :=
sorry

end parabola_focus_line_intersection_l1046_104656


namespace stating_greatest_N_with_property_l1046_104690

/-- 
Given a positive integer N, this function type represents the existence of 
N integers x_1, ..., x_N such that x_i^2 - x_i x_j is not divisible by 1111 for any i ≠ j.
-/
def HasProperty (N : ℕ+) : Prop :=
  ∃ (x : Fin N → ℤ), ∀ (i j : Fin N), i ≠ j → ¬(1111 ∣ (x i)^2 - (x i) * (x j))

/-- 
Theorem stating that 1000 is the greatest positive integer satisfying the property.
-/
theorem greatest_N_with_property : 
  HasProperty 1000 ∧ ∀ (N : ℕ+), N > 1000 → ¬HasProperty N :=
sorry

end stating_greatest_N_with_property_l1046_104690


namespace statue_weight_proof_l1046_104675

def original_weight : ℝ := 190

def week1_reduction : ℝ := 0.25
def week2_reduction : ℝ := 0.15
def week3_reduction : ℝ := 0.10

def final_weight : ℝ := original_weight * (1 - week1_reduction) * (1 - week2_reduction) * (1 - week3_reduction)

theorem statue_weight_proof : final_weight = 108.9125 := by
  sorry

end statue_weight_proof_l1046_104675


namespace perfect_power_l1046_104657

theorem perfect_power (M a b r : ℕ) (f : ℤ → ℤ) 
  (h_a : a ≥ 2) 
  (h_r : r ≥ 2) 
  (h_comp : ∀ n : ℤ, (f^[r]) n = a * n + b) 
  (h_nonneg : ∀ n : ℤ, n ≥ M → f n ≥ 0) 
  (h_div : ∀ n m : ℤ, n > m → m > M → (n - m) ∣ (f n - f m)) :
  ∃ k : ℕ, a = k^r := by
sorry

end perfect_power_l1046_104657


namespace chimney_bricks_count_l1046_104614

-- Define the number of bricks in the chimney
def chimney_bricks : ℕ := 148

-- Define Brenda's time to build the chimney alone
def brenda_time : ℕ := 7

-- Define Brandon's time to build the chimney alone
def brandon_time : ℕ := 8

-- Define the time they take to build the chimney together
def combined_time : ℕ := 6

-- Define the productivity drop when working together
def productivity_drop : ℕ := 15

-- Theorem statement
theorem chimney_bricks_count :
  -- Individual rates
  let brenda_rate := chimney_bricks / brenda_time
  let brandon_rate := chimney_bricks / brandon_time
  -- Combined rate without drop
  let combined_rate := brenda_rate + brandon_rate
  -- Actual combined rate with productivity drop
  let actual_combined_rate := combined_rate - productivity_drop
  -- The work completed matches the number of bricks
  actual_combined_rate * combined_time = chimney_bricks := by
  sorry

end chimney_bricks_count_l1046_104614


namespace lincoln_high_school_groups_l1046_104689

/-- Represents the number of students in various groups at Lincoln High School -/
structure SchoolGroups where
  total : ℕ
  band : ℕ
  chorus : ℕ
  drama : ℕ
  band_chorus_drama : ℕ

/-- Calculates the number of students in both band and chorus but not in drama -/
def students_in_band_and_chorus_not_drama (g : SchoolGroups) : ℕ :=
  g.band + g.chorus - (g.band_chorus_drama - g.drama)

/-- Theorem stating the number of students in both band and chorus but not in drama -/
theorem lincoln_high_school_groups (g : SchoolGroups) 
  (h1 : g.total = 300)
  (h2 : g.band = 80)
  (h3 : g.chorus = 120)
  (h4 : g.drama = 50)
  (h5 : g.band_chorus_drama = 200) :
  students_in_band_and_chorus_not_drama g = 50 := by
  sorry

end lincoln_high_school_groups_l1046_104689


namespace prob_exactly_one_hit_prob_distribution_X_expected_value_X_l1046_104607

-- Define the probabilities and scores
def prob_A_hit : ℝ := 0.8
def prob_B_hit : ℝ := 0.5
def score_A_hit : ℕ := 5
def score_B_hit : ℕ := 10

-- Define the random variable X for the total score
def X : ℕ → ℝ
| 0 => (1 - prob_A_hit)^2 * (1 - prob_B_hit)
| 5 => 2 * prob_A_hit * (1 - prob_A_hit) * (1 - prob_B_hit)
| 10 => prob_A_hit^2 * (1 - prob_B_hit) + (1 - prob_A_hit)^2 * prob_B_hit
| 15 => 2 * prob_A_hit * (1 - prob_A_hit) * prob_B_hit
| 20 => prob_A_hit^2 * prob_B_hit
| _ => 0

-- Theorem for the probability of exactly one hit
theorem prob_exactly_one_hit : 
  2 * prob_A_hit * (1 - prob_A_hit) * (1 - prob_B_hit) + (1 - prob_A_hit)^2 * prob_B_hit = 0.18 := 
by sorry

-- Theorem for the probability distribution of X
theorem prob_distribution_X : 
  X 0 = 0.02 ∧ X 5 = 0.16 ∧ X 10 = 0.34 ∧ X 15 = 0.16 ∧ X 20 = 0.32 := 
by sorry

-- Theorem for the expected value of X
theorem expected_value_X : 
  0 * X 0 + 5 * X 5 + 10 * X 10 + 15 * X 15 + 20 * X 20 = 13.0 := 
by sorry

end prob_exactly_one_hit_prob_distribution_X_expected_value_X_l1046_104607


namespace skaters_meeting_distance_l1046_104653

/-- Represents the meeting point of two skaters --/
structure MeetingPoint where
  time : ℝ
  distance_allie : ℝ
  distance_billie : ℝ

/-- Calculates the meeting point of two skaters --/
def calculate_meeting_point (speed_allie speed_billie distance_ab angle : ℝ) : MeetingPoint :=
  sorry

/-- The theorem to be proved --/
theorem skaters_meeting_distance 
  (speed_allie : ℝ)
  (speed_billie : ℝ)
  (distance_ab : ℝ)
  (angle : ℝ)
  (h1 : speed_allie = 8)
  (h2 : speed_billie = 7)
  (h3 : distance_ab = 100)
  (h4 : angle = π / 3) -- 60 degrees in radians
  : 
  let meeting := calculate_meeting_point speed_allie speed_billie distance_ab angle
  meeting.distance_allie = 160 :=
by
  sorry

end skaters_meeting_distance_l1046_104653


namespace g_value_proof_l1046_104602

def nabla (g h : ℝ) : ℝ := g^2 - h^2

theorem g_value_proof (g : ℝ) (h_pos : g > 0) (h_eq : nabla g 6 = 45) : g = 9 := by
  sorry

end g_value_proof_l1046_104602


namespace compound_interest_problem_l1046_104616

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Total amount calculation -/
def total_amount (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

theorem compound_interest_problem (P : ℝ) :
  compound_interest P 0.06 2 = 370.80 →
  total_amount P 370.80 = 3370.80 := by
  sorry

end compound_interest_problem_l1046_104616


namespace rectangle_y_value_l1046_104688

/-- Given a rectangle with vertices at (-2, y), (6, y), (-2, 2), and (6, 2),
    where y is positive, and an area of 64 square units, y must equal 10. -/
theorem rectangle_y_value (y : ℝ) (h1 : y > 0) : 
  (6 - (-2)) * (y - 2) = 64 → y = 10 := by
  sorry

end rectangle_y_value_l1046_104688


namespace arithmetic_sequence_property_l1046_104683

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  d : ℤ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h1 : seq.a 3 = S seq 5)
  (h2 : seq.a 2 * seq.a 4 = S seq 4) :
  (∀ n, seq.a n = 2 * n - 6) ∧
  (∀ n < 7, S seq n ≤ seq.a n) ∧
  (S seq 7 > seq.a 7) :=
sorry

end arithmetic_sequence_property_l1046_104683


namespace prime_triple_uniqueness_l1046_104668

theorem prime_triple_uniqueness : 
  ∀ p : ℕ, p > 0 → Prime p → Prime (p + 2) → Prime (p + 4) → p = 3 :=
by sorry

end prime_triple_uniqueness_l1046_104668


namespace cabbages_on_plot_l1046_104697

/-- Calculates the total number of cabbages that can be planted on a rectangular plot. -/
def total_cabbages (length width density : ℕ) : ℕ :=
  length * width * density

/-- Theorem stating the total number of cabbages on the given plot. -/
theorem cabbages_on_plot :
  total_cabbages 16 12 9 = 1728 := by
  sorry

#eval total_cabbages 16 12 9

end cabbages_on_plot_l1046_104697


namespace cheerleaders_size2_l1046_104658

/-- The number of cheerleaders needing size 2 uniforms -/
def size2 (total size6 : ℕ) : ℕ :=
  total - (size6 + size6 / 2)

/-- Theorem stating that 4 cheerleaders need size 2 uniforms -/
theorem cheerleaders_size2 :
  size2 19 10 = 4 := by
  sorry

end cheerleaders_size2_l1046_104658


namespace shaded_area_calculation_l1046_104626

theorem shaded_area_calculation (carpet_side : ℝ) (S T : ℝ) 
  (h1 : carpet_side = 12)
  (h2 : carpet_side / S = 4)
  (h3 : S / T = 4)
  (h4 : carpet_side > 0) : 
  S^2 + 16 * T^2 = 18 := by
  sorry

end shaded_area_calculation_l1046_104626


namespace triangle_side_relation_l1046_104638

theorem triangle_side_relation (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (angle_B : Real.cos (2 * Real.pi / 3) = -1/2) :
  a^2 + a*c + c^2 - b^2 = 0 := by
sorry

end triangle_side_relation_l1046_104638


namespace inequality_proof_l1046_104617

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end inequality_proof_l1046_104617


namespace line_segment_polar_equation_l1046_104680

/-- The polar coordinate equation of the line segment y = 1 - x (0 ≤ x ≤ 1) -/
theorem line_segment_polar_equation :
  ∀ (x y ρ θ : ℝ),
  (0 ≤ x) ∧ (x ≤ 1) ∧ (y = 1 - x) →
  (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  (ρ = 1 / (Real.cos θ + Real.sin θ)) ∧ (0 ≤ θ) ∧ (θ ≤ Real.pi / 2) :=
by sorry

end line_segment_polar_equation_l1046_104680


namespace f_equals_g_l1046_104643

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 1
def g (t : ℝ) : ℝ := 2 * t - 1

-- Theorem stating that f and g are the same function
theorem f_equals_g : f = g := by sorry

end f_equals_g_l1046_104643


namespace shopping_tax_calculation_l1046_104693

theorem shopping_tax_calculation (total_amount : ℝ) (total_amount_pos : total_amount > 0) :
  let clothing_percent : ℝ := 0.40
  let food_percent : ℝ := 0.20
  let electronics_percent : ℝ := 0.15
  let other_percent : ℝ := 0.25
  let clothing_tax : ℝ := 0.12
  let food_tax : ℝ := 0
  let electronics_tax : ℝ := 0.05
  let other_tax : ℝ := 0.20
  let total_tax := 
    clothing_percent * total_amount * clothing_tax +
    food_percent * total_amount * food_tax +
    electronics_percent * total_amount * electronics_tax +
    other_percent * total_amount * other_tax
  (total_tax / total_amount) * 100 = 10.55 := by
sorry

end shopping_tax_calculation_l1046_104693


namespace twitch_income_per_subscriber_l1046_104645

/-- Calculates the income per subscriber for a Twitch streamer --/
theorem twitch_income_per_subscriber
  (initial_subscribers : ℕ)
  (gifted_subscribers : ℕ)
  (total_monthly_income : ℕ)
  (h1 : initial_subscribers = 150)
  (h2 : gifted_subscribers = 50)
  (h3 : total_monthly_income = 1800) :
  total_monthly_income / (initial_subscribers + gifted_subscribers) = 9 := by
sorry

end twitch_income_per_subscriber_l1046_104645


namespace inequality_solution_set_l1046_104678

theorem inequality_solution_set (x : ℝ) :
  x ≠ 3/2 →
  ((x - 4) / (3 - 2*x) < 0) ↔ (x < 3/2 ∨ x > 4) := by
sorry

end inequality_solution_set_l1046_104678


namespace walking_speed_is_4_l1046_104642

/-- The speed at which Jack and Jill walked -/
def walking_speed : ℝ → ℝ := λ x => x^3 - 5*x^2 - 14*x + 104

/-- The distance Jill walked -/
def jill_distance : ℝ → ℝ := λ x => x^2 - 7*x - 60

/-- The time Jill walked -/
def jill_time : ℝ → ℝ := λ x => x + 7

theorem walking_speed_is_4 :
  ∃ x : ℝ, x ≠ -7 ∧ walking_speed x = (jill_distance x) / (jill_time x) ∧ walking_speed x = 4 := by
  sorry

end walking_speed_is_4_l1046_104642


namespace max_value_of_f_l1046_104628

def f (x : ℝ) := x^3 - 3*x

theorem max_value_of_f :
  ∃ (m : ℝ), m = 18 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ m :=
by sorry

end max_value_of_f_l1046_104628


namespace complex_fraction_simplification_l1046_104685

theorem complex_fraction_simplification : 
  (((3875/1000) * (1/5) + (155/4) * (9/100) - (155/400)) / 
   ((13/6) + (((108/25) - (42/25) - (33/25)) * (5/11) - (2/7)) / (44/35) + (35/24))) = 1 := by
  sorry

end complex_fraction_simplification_l1046_104685
