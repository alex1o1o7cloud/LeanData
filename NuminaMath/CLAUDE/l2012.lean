import Mathlib

namespace next_adjacent_natural_number_l2012_201241

theorem next_adjacent_natural_number (n a : ℕ) (h : n = a^2) : 
  n + 1 = a^2 + 1 := by sorry

end next_adjacent_natural_number_l2012_201241


namespace total_students_l2012_201249

/-- The total number of students in five classes given specific conditions -/
theorem total_students (finley johnson garcia smith patel : ℕ) : 
  finley = 24 →
  johnson = finley / 2 + 10 →
  garcia = 2 * johnson →
  smith = finley / 3 →
  patel = (3 * (finley + johnson + garcia)) / 4 →
  finley + johnson + garcia + smith + patel = 166 := by
sorry

end total_students_l2012_201249


namespace linear_function_slope_l2012_201294

-- Define the linear function
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem linear_function_slope (k b : ℝ) :
  (∀ x : ℝ, linear_function k b (x + 3) = linear_function k b x - 2) →
  k = -2/3 := by
  sorry

end linear_function_slope_l2012_201294


namespace congruence_problem_l2012_201205

theorem congruence_problem (a b : ℤ) (h1 : a ≡ 16 [ZMOD 44]) (h2 : b ≡ 77 [ZMOD 44]) :
  (a - b ≡ 159 [ZMOD 44]) ∧
  (∀ n : ℤ, 120 ≤ n ∧ n ≤ 161 → (a - b ≡ n [ZMOD 44] ↔ n = 159)) :=
by sorry

end congruence_problem_l2012_201205


namespace hyperbola_eccentricity_l2012_201211

/-- Given a hyperbola with center at the origin, foci on the y-axis,
    and an asymptote passing through (-2, 4), its eccentricity is √5/2 -/
theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ),
  (∀ x y : ℝ, y = a/b * x → (x = -2 ∧ y = 4)) →  -- asymptote passes through (-2, 4)
  a^2 = c^2 - b^2 →                              -- hyperbola equation
  c^2 / a^2 = (5:ℝ)/4 :=                         -- eccentricity squared
by sorry

end hyperbola_eccentricity_l2012_201211


namespace inscribed_circle_area_ratio_l2012_201213

/-- Theorem: Ratio of inscribed circle area to square area is π/4 -/
theorem inscribed_circle_area_ratio (a b : ℤ) (h : b ≠ 0) :
  let r : ℚ := a / b
  let circle_area := π * r^2
  let square_side := 2 * r
  let square_area := square_side^2
  circle_area / square_area = π / 4 := by
  sorry

end inscribed_circle_area_ratio_l2012_201213


namespace triangle_perimeter_l2012_201216

theorem triangle_perimeter (a b c : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Positive side lengths
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  ((a - 6) * (a - 3) = 0 ∨ (b - 6) * (b - 3) = 0 ∨ (c - 6) * (c - 3) = 0) →  -- At least one side satisfies the equation
  (a + b + c = 9 ∨ a + b + c = 15 ∨ a + b + c = 18) :=
by sorry


end triangle_perimeter_l2012_201216


namespace average_marks_of_combined_classes_l2012_201297

theorem average_marks_of_combined_classes 
  (class1_size : ℕ) (class1_avg : ℝ) 
  (class2_size : ℕ) (class2_avg : ℝ) : 
  let total_students := class1_size + class2_size
  let total_marks := class1_size * class1_avg + class2_size * class2_avg
  total_marks / total_students = (35 * 45 + 55 * 65) / (35 + 55) :=
by
  sorry

#eval (35 * 45 + 55 * 65) / (35 + 55)

end average_marks_of_combined_classes_l2012_201297


namespace number_division_problem_l2012_201210

theorem number_division_problem (x y : ℝ) 
  (h1 : (x - 5) / 7 = 7)
  (h2 : (x - 14) / y = 4) : 
  y = 10 := by
sorry

end number_division_problem_l2012_201210


namespace train_length_l2012_201225

/-- The length of a train given its speed and the time it takes to cross a platform -/
theorem train_length 
  (train_speed : ℝ) 
  (platform_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_speed = 72) -- speed in kmph
  (h2 : platform_length = 250) -- length in meters
  (h3 : crossing_time = 36) -- time in seconds
  : ∃ (train_length : ℝ), train_length = 470 :=
by
  sorry

end train_length_l2012_201225


namespace problem_solution_l2012_201285

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (h1 : x/2 = y^2) (h2 : x/4 = 4*y) : x = 128 := by
  sorry

end problem_solution_l2012_201285


namespace max_rabbits_with_traits_l2012_201295

theorem max_rabbits_with_traits :
  ∃ (N : ℕ), N = 27 ∧
  (∀ (n : ℕ), n > N →
    ¬(∃ (long_ears jump_far both : Finset (Fin n)),
      long_ears.card = 13 ∧
      jump_far.card = 17 ∧
      (long_ears ∩ jump_far).card ≥ 3)) ∧
  (∃ (long_ears jump_far both : Finset (Fin N)),
    long_ears.card = 13 ∧
    jump_far.card = 17 ∧
    (long_ears ∩ jump_far).card ≥ 3) :=
by sorry

end max_rabbits_with_traits_l2012_201295


namespace unique_x_value_l2012_201238

theorem unique_x_value : ∃! x : ℕ, 
  (∃ k : ℕ, x = 9 * k) ∧ 
  (x^2 > 120) ∧ 
  (x < 25) ∧ 
  (x = 18) := by
sorry

end unique_x_value_l2012_201238


namespace smallest_c_value_l2012_201231

theorem smallest_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x : ℝ, a * Real.cos (b * x + c) ≤ a * Real.cos (b * (-π/4) + c)) →
  c ≥ π/4 :=
sorry

end smallest_c_value_l2012_201231


namespace parabola_passes_through_origin_l2012_201276

/-- A parabola defined by y = 3x^2 passes through the point (0, 0) -/
theorem parabola_passes_through_origin :
  let f : ℝ → ℝ := λ x ↦ 3 * x^2
  f 0 = 0 := by
  sorry

end parabola_passes_through_origin_l2012_201276


namespace expression_value_l2012_201261

theorem expression_value (x y z w : ℝ) 
  (eq1 : 4 * x * z + y * w = 4) 
  (eq2 : x * w + y * z = 8) : 
  (2 * x + y) * (2 * z + w) = 20 := by
sorry

end expression_value_l2012_201261


namespace instantaneous_velocity_at_3s_l2012_201208

/-- The position function of a particle -/
def S (t : ℝ) : ℝ := 2 * t^3 + t

/-- The velocity function of a particle -/
def V (t : ℝ) : ℝ := 6 * t^2 + 1

theorem instantaneous_velocity_at_3s :
  V 3 = 55 := by sorry

end instantaneous_velocity_at_3s_l2012_201208


namespace exactly_three_valid_sets_l2012_201290

/-- A set of consecutive positive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  length_ge_3 : length ≥ 3

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a valid set (sum equals 150) -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  sum_consecutive s = 150

theorem exactly_three_valid_sets :
  ∃! (sets : Finset ConsecutiveSet), 
    (∀ s ∈ sets, is_valid_set s) ∧ 
    Finset.card sets = 3 := by sorry

end exactly_three_valid_sets_l2012_201290


namespace quadratic_solution_implies_sum_l2012_201281

/-- Given that x = -1 is a solution of the quadratic equation ax^2 + bx + 23 = 0,
    prove that -a + b + 2000 = 2023 -/
theorem quadratic_solution_implies_sum (a b : ℝ) 
  (h : a * (-1)^2 + b * (-1) + 23 = 0) : 
  -a + b + 2000 = 2023 := by
  sorry

end quadratic_solution_implies_sum_l2012_201281


namespace hyperbola_asymptote_slopes_l2012_201265

/-- The slopes of the asymptotes for the hyperbola (y^2/16) - (x^2/9) = 1 are ±4/3 -/
theorem hyperbola_asymptote_slopes :
  let f (x y : ℝ) := y^2 / 16 - x^2 / 9
  ∃ (m : ℝ), m = 4/3 ∧ 
    (∀ ε > 0, ∃ M > 0, ∀ x y, |x| > M → |y| > M → f x y = 1 → 
      (|y - m*x| < ε*|x| ∨ |y + m*x| < ε*|x|)) :=
by sorry

end hyperbola_asymptote_slopes_l2012_201265


namespace initial_insurance_premium_l2012_201209

/-- Proves that the initial insurance premium is $50 given the specified conditions --/
theorem initial_insurance_premium (P : ℝ) : 
  (1.1 * P + 3 * 5 = 70) → P = 50 := by
  sorry

end initial_insurance_premium_l2012_201209


namespace cars_meet_time_l2012_201262

/-- Represents a rectangle ABCD -/
structure Rectangle where
  BC : ℝ
  CD : ℝ

/-- Represents a car with a constant speed -/
structure Car where
  speed : ℝ

/-- Time for cars to meet on diagonal BD -/
def meetingTime (rect : Rectangle) (car1 car2 : Car) : ℝ :=
  40 -- in minutes

/-- Theorem stating that cars meet after 40 minutes -/
theorem cars_meet_time (rect : Rectangle) (car1 car2 : Car) :
  meetingTime rect car1 car2 = 40 := by
  sorry

end cars_meet_time_l2012_201262


namespace cosine_angle_vectors_l2012_201266

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_angle_vectors (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : 2 * ‖a‖ = 3 * ‖b‖) (h2 : ‖a - 2•b‖ = ‖a + b‖) :
  inner a b / (‖a‖ * ‖b‖) = 1/3 := by sorry

end cosine_angle_vectors_l2012_201266


namespace binomial_expectation_five_l2012_201252

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Probability mass function for a binomial distribution -/
def pmf (ξ : BinomialRV) (k : ℕ) : ℝ :=
  (ξ.n.choose k) * (ξ.p ^ k) * ((1 - ξ.p) ^ (ξ.n - k))

/-- Expected value of a binomial distribution -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

theorem binomial_expectation_five (ξ : BinomialRV) 
    (h_p : ξ.p = 1/2) 
    (h_pmf : pmf ξ 2 = 45 / 2^10) : 
  expectation ξ = 5 := by
  sorry

end binomial_expectation_five_l2012_201252


namespace curve_point_coordinates_l2012_201272

theorem curve_point_coordinates (θ : Real) (x y : Real) :
  0 ≤ θ ∧ θ ≤ π →
  x = 3 * Real.cos θ →
  y = 4 * Real.sin θ →
  y = x →
  x = 12/5 ∧ y = 12/5 := by
sorry

end curve_point_coordinates_l2012_201272


namespace quadratic_monotonicity_l2012_201240

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∨
  (∀ x y, a < x ∧ x < y ∧ y < b → f x > f y)

theorem quadratic_monotonicity (a : ℝ) :
  monotonic_on (f a) 2 3 ↔ a ≤ 2 ∨ a ≥ 3 :=
sorry

end quadratic_monotonicity_l2012_201240


namespace blue_pens_count_l2012_201274

/-- Given the prices of red and blue pens, the total amount spent, and the total number of pens,
    prove that the number of blue pens bought is 11. -/
theorem blue_pens_count (red_price blue_price total_spent total_pens : ℕ) 
    (h1 : red_price = 5)
    (h2 : blue_price = 7)
    (h3 : total_spent = 102)
    (h4 : total_pens = 16) : 
  ∃ (red_count blue_count : ℕ),
    red_count + blue_count = total_pens ∧
    red_count * red_price + blue_count * blue_price = total_spent ∧
    blue_count = 11 := by
  sorry

end blue_pens_count_l2012_201274


namespace bobby_pancakes_l2012_201256

theorem bobby_pancakes (total : ℕ) (dog_ate : ℕ) (left : ℕ) (bobby_ate : ℕ) : 
  total = 21 → dog_ate = 7 → left = 9 → bobby_ate = total - dog_ate - left → bobby_ate = 5 := by
  sorry

end bobby_pancakes_l2012_201256


namespace quadratic_expression_rewrite_l2012_201271

theorem quadratic_expression_rewrite (i j : ℂ) : 
  let expression := 8 * j^2 + (6 * i) * j + 16
  ∃ (c p q : ℂ), 
    expression = c * (j + p)^2 + q ∧ 
    q / p = -137 * I / 3 := by
  sorry

end quadratic_expression_rewrite_l2012_201271


namespace quadratic_two_distinct_roots_l2012_201234

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + m*x₁ - 3 = 0) ∧ 
  (x₂^2 + m*x₂ - 3 = 0) := by
  sorry

end quadratic_two_distinct_roots_l2012_201234


namespace vector_problem_l2012_201224

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (3, 4)

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  dot_product v w = 0

theorem vector_problem :
  (∃ k : ℝ, parallel (3 • a - b) (a + k • b) ∧ k = -1/3) ∧
  (∃ m : ℝ, perpendicular a (m • a - b) ∧ m = -1) := by
  sorry

end vector_problem_l2012_201224


namespace vector_problem_l2012_201270

def a : Fin 2 → ℝ := ![- 3, 1]
def b : Fin 2 → ℝ := ![1, -2]
def c : Fin 2 → ℝ := ![1, -1]

def m (k : ℝ) : Fin 2 → ℝ := fun i ↦ a i + k * b i

theorem vector_problem :
  (∃ k : ℝ, (∀ i : Fin 2, m k i * (2 * a i - b i) = 0) ∧ k = 5 / 3) ∧
  (∃ k : ℝ, (∀ i : Fin 2, ∃ t : ℝ, m k i = t * (k * b i + c i)) ∧ k = -1 / 3) := by
  sorry

end vector_problem_l2012_201270


namespace toothpick_20th_stage_l2012_201258

def toothpick_sequence (n : ℕ) : ℕ :=
  3 + 3 * (n - 1)

theorem toothpick_20th_stage :
  toothpick_sequence 20 = 60 := by
  sorry

end toothpick_20th_stage_l2012_201258


namespace intersection_distance_l2012_201226

theorem intersection_distance : 
  ∃ (p1 p2 : ℝ × ℝ),
    (p1.1^2 + p1.2^2 = 13) ∧ 
    (p1.1 + p1.2 = 4) ∧
    (p2.1^2 + p2.2^2 = 13) ∧ 
    (p2.1 + p2.2 = 4) ∧
    (p1 ≠ p2) ∧
    ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = 80) :=
by sorry

end intersection_distance_l2012_201226


namespace sum_smallest_largest_prime_1_to_50_l2012_201293

theorem sum_smallest_largest_prime_1_to_50 : 
  ∃ (p q : ℕ), 
    (1 < p) ∧ (p ≤ 50) ∧ Nat.Prime p ∧
    (1 < q) ∧ (q ≤ 50) ∧ Nat.Prime q ∧
    (∀ r : ℕ, (1 < r) ∧ (r ≤ 50) ∧ Nat.Prime r → p ≤ r ∧ r ≤ q) ∧
    p + q = 49 :=
by sorry

end sum_smallest_largest_prime_1_to_50_l2012_201293


namespace susie_savings_account_l2012_201214

/-- The compound interest formula for yearly compounding -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- The problem statement -/
theorem susie_savings_account :
  let principal : ℝ := 2500
  let rate : ℝ := 0.06
  let years : ℕ := 21
  let result := compound_interest principal rate years
  ∃ ε > 0, |result - 8017.84| < ε :=
sorry

end susie_savings_account_l2012_201214


namespace oldest_sibling_age_is_44_l2012_201246

def kay_age : ℕ := 32

def youngest_sibling_age : ℕ := kay_age / 2 - 5

def oldest_sibling_age : ℕ := 4 * youngest_sibling_age

theorem oldest_sibling_age_is_44 : oldest_sibling_age = 44 := by
  sorry

end oldest_sibling_age_is_44_l2012_201246


namespace square_plot_with_path_l2012_201218

theorem square_plot_with_path (path_area : ℝ) (edge_diff : ℝ) (total_area : ℝ) :
  path_area = 464 →
  edge_diff = 32 →
  (∃ x y : ℝ,
    x > 0 ∧
    y > 0 ∧
    x^2 - y^2 = path_area ∧
    4 * (x - y) = edge_diff ∧
    total_area = x^2) →
  total_area = 1089 := by
  sorry

end square_plot_with_path_l2012_201218


namespace least_satisfying_number_l2012_201206

def satisfies_conditions (n : ℕ) : Prop :=
  n % 10 = 9 ∧ n % 11 = 10 ∧ n % 12 = 11 ∧ n % 13 = 12

theorem least_satisfying_number : 
  satisfies_conditions 8579 ∧ 
  ∀ m : ℕ, m < 8579 → ¬(satisfies_conditions m) :=
by sorry

end least_satisfying_number_l2012_201206


namespace study_group_probability_l2012_201201

/-- Represents the gender distribution in the study group -/
def gender_distribution : Fin 2 → ℝ
  | 0 => 0.55  -- women
  | 1 => 0.45  -- men

/-- Represents the age distribution for each gender -/
def age_distribution : Fin 2 → Fin 3 → ℝ
  | 0, 0 => 0.20  -- women below 35
  | 0, 1 => 0.35  -- women 35-50
  | 0, 2 => 0.45  -- women above 50
  | 1, 0 => 0.30  -- men below 35
  | 1, 1 => 0.40  -- men 35-50
  | 1, 2 => 0.30  -- men above 50

/-- Represents the profession distribution for each gender and age group -/
def profession_distribution : Fin 2 → Fin 3 → Fin 3 → ℝ
  | 0, 0, 0 => 0.35  -- women below 35, lawyers
  | 0, 0, 1 => 0.45  -- women below 35, doctors
  | 0, 0, 2 => 0.20  -- women below 35, engineers
  | 0, 1, 0 => 0.25  -- women 35-50, lawyers
  | 0, 1, 1 => 0.50  -- women 35-50, doctors
  | 0, 1, 2 => 0.25  -- women 35-50, engineers
  | 0, 2, 0 => 0.20  -- women above 50, lawyers
  | 0, 2, 1 => 0.30  -- women above 50, doctors
  | 0, 2, 2 => 0.50  -- women above 50, engineers
  | 1, 0, 0 => 0.40  -- men below 35, lawyers
  | 1, 0, 1 => 0.30  -- men below 35, doctors
  | 1, 0, 2 => 0.30  -- men below 35, engineers
  | 1, 1, 0 => 0.45  -- men 35-50, lawyers
  | 1, 1, 1 => 0.25  -- men 35-50, doctors
  | 1, 1, 2 => 0.30  -- men 35-50, engineers
  | 1, 2, 0 => 0.30  -- men above 50, lawyers
  | 1, 2, 1 => 0.40  -- men above 50, doctors
  | 1, 2, 2 => 0.30  -- men above 50, engineers

theorem study_group_probability : 
  gender_distribution 0 * age_distribution 0 0 * profession_distribution 0 0 0 +
  gender_distribution 1 * age_distribution 1 2 * profession_distribution 1 2 2 +
  gender_distribution 0 * age_distribution 0 1 * profession_distribution 0 1 1 +
  gender_distribution 1 * age_distribution 1 1 * profession_distribution 1 1 1 = 0.22025 := by
  sorry

end study_group_probability_l2012_201201


namespace geometric_sequence_product_l2012_201248

/-- Given a geometric sequence {aₙ} where a₁ and a₁₃ are the roots of x² - 8x + 1 = 0,
    the product a₅ · a₇ · a₉ equals 1. -/
theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  (a 1)^2 - 8*(a 1) + 1 = 0 →           -- a₁ is a root
  (a 13)^2 - 8*(a 13) + 1 = 0 →         -- a₁₃ is a root
  a 5 * a 7 * a 9 = 1 := by
sorry

end geometric_sequence_product_l2012_201248


namespace jade_tower_levels_l2012_201273

/-- Calculates the number of complete levels in a Lego tower. -/
def towerLevels (totalPieces piecesPerLevel unusedPieces : ℕ) : ℕ :=
  (totalPieces - unusedPieces) / piecesPerLevel

/-- Proves that given the specific conditions, the tower has 11 levels. -/
theorem jade_tower_levels :
  towerLevels 100 7 23 = 11 := by
  sorry

end jade_tower_levels_l2012_201273


namespace number_value_l2012_201200

theorem number_value (N : ℝ) (h : (1/2) * N = 1) : N = 2 := by
  sorry

end number_value_l2012_201200


namespace hens_and_cows_problem_l2012_201283

theorem hens_and_cows_problem (total_animals : ℕ) (total_feet : ℕ) (hens : ℕ) (cows : ℕ) :
  total_animals = 46 →
  total_feet = 136 →
  total_animals = hens + cows →
  total_feet = 2 * hens + 4 * cows →
  hens = 24 := by
  sorry

end hens_and_cows_problem_l2012_201283


namespace line_through_point_l2012_201223

/-- Given a line ax - y - 1 = 0 passing through the point (1, 3), prove that a = 4 -/
theorem line_through_point (a : ℝ) : (a * 1 - 3 - 1 = 0) → a = 4 := by
  sorry

end line_through_point_l2012_201223


namespace inheritance_calculation_l2012_201264

theorem inheritance_calculation (x : ℝ) : 
  (0.2 * x + 0.1 * (0.8 * x) = 10500) → x = 37500 := by
  sorry

end inheritance_calculation_l2012_201264


namespace dilation_matrix_determinant_l2012_201229

theorem dilation_matrix_determinant :
  ∀ (E : Matrix (Fin 2) (Fin 2) ℝ),
  (∀ (i j : Fin 2), E i j = if i = j then 9 else 0) →
  Matrix.det E = 81 := by
sorry

end dilation_matrix_determinant_l2012_201229


namespace cyclic_quadrilateral_max_product_l2012_201220

/-- A cyclic quadrilateral with sides a, b, c, d inscribed in a circle of radius R -/
structure CyclicQuadrilateral where
  R : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  inscribed : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ R > 0

/-- The product of sums of opposite sides pairs -/
def sideProduct (q : CyclicQuadrilateral) : ℝ :=
  (q.a * q.b + q.c * q.d) * (q.a * q.c + q.b * q.d) * (q.a * q.d + q.b * q.c)

/-- Predicate to check if a cyclic quadrilateral is a square -/
def isSquare (q : CyclicQuadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

theorem cyclic_quadrilateral_max_product (q : CyclicQuadrilateral) :
  ∀ q' : CyclicQuadrilateral, q'.R = q.R → sideProduct q ≤ sideProduct q' ↔ isSquare q' :=
sorry

end cyclic_quadrilateral_max_product_l2012_201220


namespace triangle_side_length_l2012_201212

theorem triangle_side_length (b c : ℝ) (C : ℝ) (h1 : b = 6 * Real.sqrt 3) (h2 : c = 6) (h3 : C = 30 * π / 180) :
  ∃ (a : ℝ), (a = 6 ∨ a = 12) ∧ c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) :=
sorry

end triangle_side_length_l2012_201212


namespace lattice_point_probability_l2012_201202

theorem lattice_point_probability (d : ℝ) : 
  (d > 0) → 
  (π * d^2 = 3/4) → 
  (d = Real.sqrt (3 / (4 * π))) :=
sorry

end lattice_point_probability_l2012_201202


namespace unique_solution_quadratic_l2012_201257

theorem unique_solution_quadratic (j : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) * (x - 6) = -51 + j * x) ↔ (j = 0 ∨ j = -36) := by
  sorry

end unique_solution_quadratic_l2012_201257


namespace furniture_uncountable_l2012_201296

-- Define what an uncountable noun is
def UncountableNoun (word : String) : Prop := sorry

-- Define the property of an uncountable noun not changing form
def DoesNotChangeForm (word : String) : Prop := 
  UncountableNoun word → word = word

-- Theorem statement
theorem furniture_uncountable : 
  UncountableNoun "furniture" → DoesNotChangeForm "furniture" := by
  sorry

end furniture_uncountable_l2012_201296


namespace jerrys_breakfast_calories_l2012_201217

/-- Calculates the total calories in Jerry's breakfast. -/
theorem jerrys_breakfast_calories :
  let pancake_calories : ℕ := 7 * 120
  let bacon_calories : ℕ := 3 * 100
  let orange_juice_calories : ℕ := 2 * 300
  let cereal_calories : ℕ := 200
  let muffin_calories : ℕ := 350
  pancake_calories + bacon_calories + orange_juice_calories + cereal_calories + muffin_calories = 2290 :=
by sorry

end jerrys_breakfast_calories_l2012_201217


namespace right_triangle_leg_length_l2012_201268

theorem right_triangle_leg_length 
  (a b c : ℝ) 
  (right_angle : a^2 + b^2 = c^2) 
  (hypotenuse : c = 25) 
  (known_leg : a = 24) : 
  b = 7 := by
sorry

end right_triangle_leg_length_l2012_201268


namespace oliver_final_amount_l2012_201239

-- Define the currencies
structure Currency where
  usd : ℚ
  quarters : ℚ
  dimes : ℚ
  eur : ℚ
  gbp : ℚ
  chf : ℚ
  jpy : ℚ
  cad : ℚ
  aud : ℚ

-- Define the exchange rates
structure ExchangeRates where
  usd_to_gbp : ℚ
  eur_to_gbp : ℚ
  usd_to_chf : ℚ
  eur_to_chf : ℚ
  jpy_to_cad : ℚ
  eur_to_aud : ℚ

-- Define the initial amounts and exchanges
def initial_amount : Currency := {
  usd := 40,
  quarters := 200,
  dimes := 100,
  eur := 15,
  gbp := 0,
  chf := 0,
  jpy := 3000,
  cad := 0,
  aud := 0
}

def exchange_rates : ExchangeRates := {
  usd_to_gbp := 3/4,
  eur_to_gbp := 17/20,
  usd_to_chf := 9/10,
  eur_to_chf := 21/20,
  jpy_to_cad := 3/250,
  eur_to_aud := 3/2
}

def exchanged_amount : Currency := {
  usd := 10,
  quarters := 0,
  dimes := 0,
  eur := 13,
  gbp := 0,
  chf := 0,
  jpy := 2000,
  cad := 0,
  aud := 0
}

def given_to_sister : Currency := {
  usd := 5,
  quarters := 120,
  dimes := 50,
  eur := 0,
  gbp := 7/2,
  chf := 2,
  jpy := 500,
  cad := 0,
  aud := 7
}

-- Theorem to prove
theorem oliver_final_amount (initial : Currency) (rates : ExchangeRates) 
  (exchanged : Currency) (given : Currency) : 
  ∃ (final : Currency),
    final.usd = 20 ∧
    final.quarters = 0 ∧
    final.dimes = 0 ∧
    final.eur = 2 ∧
    final.gbp = 33/4 ∧
    final.chf = 49/4 ∧
    final.jpy = 0 ∧
    final.cad = 24 ∧
    final.aud = 5 :=
by sorry

end oliver_final_amount_l2012_201239


namespace min_students_theorem_l2012_201289

/-- Given a class of students, returns the minimum number of students
    who have brown eyes, a lunch box, and do not wear glasses. -/
def min_students_with_characteristics (total : ℕ) (brown_eyes : ℕ) (lunch_box : ℕ) (glasses : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of students with the given characteristics -/
theorem min_students_theorem :
  min_students_with_characteristics 40 18 25 16 = 3 :=
by sorry

end min_students_theorem_l2012_201289


namespace actual_distance_traveled_l2012_201204

/-- Proves that the actual distance traveled is 100 km given the conditions of the problem -/
theorem actual_distance_traveled (speed_slow speed_fast distance_diff : ℝ) 
  (h1 : speed_slow = 10)
  (h2 : speed_fast = 12)
  (h3 : distance_diff = 20)
  (h4 : speed_slow > 0)
  (h5 : speed_fast > speed_slow) :
  ∃ (actual_distance : ℝ),
    actual_distance / speed_slow = (actual_distance + distance_diff) / speed_fast ∧
    actual_distance = 100 :=
by sorry

end actual_distance_traveled_l2012_201204


namespace parabola_intersection_difference_l2012_201287

/-- The difference between the larger and smaller x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_difference : ∃ (a c : ℝ),
  (∀ x : ℝ, 3 * x^2 - 6 * x + 6 = -x^2 - 4 * x + 6 → (x = a ∨ x = c)) ∧
  c ≥ a ∧
  c - a = (1 : ℝ) / 2 :=
by sorry

end parabola_intersection_difference_l2012_201287


namespace chess_game_probability_l2012_201227

theorem chess_game_probability (p_win p_not_lose : ℝ) 
  (h_win : p_win = 0.3)
  (h_not_lose : p_not_lose = 0.8) :
  p_not_lose - p_win = 0.5 := by
  sorry

end chess_game_probability_l2012_201227


namespace thirty_fifth_digit_of_sum_one_ninth_one_fifth_l2012_201251

/-- The decimal representation of a rational number -/
def decimal_rep (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sum_decimal_rep (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- Theorem: The 35th digit after the decimal point of the sum of 1/9 and 1/5 is 3 -/
theorem thirty_fifth_digit_of_sum_one_ninth_one_fifth : 
  sum_decimal_rep (1/9) (1/5) 35 = 3 := by sorry

end thirty_fifth_digit_of_sum_one_ninth_one_fifth_l2012_201251


namespace a_less_than_b_less_than_c_l2012_201207

theorem a_less_than_b_less_than_c : ∀ a b c : ℝ,
  a = Real.log (1/2) →
  b = Real.sin (1/2) →
  c = 2^(-1/2 : ℝ) →
  a < b ∧ b < c := by sorry

end a_less_than_b_less_than_c_l2012_201207


namespace square_area_on_parabola_l2012_201237

theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 7) ∧
  (x₂^2 + 4*x₂ + 3 = 7) ∧
  (x₂ > x₁) ∧
  ((x₂ - x₁)^2 = 32) := by
  sorry

end square_area_on_parabola_l2012_201237


namespace small_circle_radius_l2012_201288

/-- Given a configuration of circles where:
    - There is one large circle with radius 10 meters
    - There are six congruent smaller circles
    - The smaller circles are aligned in a straight line
    - The smaller circles touch each other and the perimeter of the larger circle
    This theorem proves that the radius of each smaller circle is 5/3 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) (h1 : R = 10) (h2 : 6 * (2 * r) = 2 * R) :
  r = 5 / 3 := by
  sorry

end small_circle_radius_l2012_201288


namespace largest_integer_with_gcd_six_largest_integer_is_138_l2012_201219

theorem largest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 138 :=
sorry

theorem largest_integer_is_138 : ∃ n : ℕ, n = 138 ∧ n < 150 ∧ Nat.gcd n 18 = 6 :=
sorry

end largest_integer_with_gcd_six_largest_integer_is_138_l2012_201219


namespace chord_equation_l2012_201277

/-- Given a circle with equation x² + y² = 9 and a chord PQ with midpoint (1,2),
    the equation of line PQ is x + 2y - 5 = 0 -/
theorem chord_equation (P Q : ℝ × ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 9 → (x - 1)^2 + (y - 2)^2 = ((P.1 - 1)^2 + (P.2 - 2)^2)) →
  ((P.1 + Q.1) / 2 = 1 ∧ (P.2 + Q.2) / 2 = 2) →
  ∃ (k : ℝ), ∀ (x y : ℝ), (x = P.1 ∧ y = P.2) ∨ (x = Q.1 ∧ y = Q.2) → x + 2*y - 5 = k := by
  sorry

end chord_equation_l2012_201277


namespace college_students_count_l2012_201233

theorem college_students_count : ℕ :=
  let students_to_professors_ratio : ℕ := 15
  let total_people : ℕ := 40000
  let students : ℕ := 37500

  have h1 : students = students_to_professors_ratio * (total_people - students) := by sorry
  have h2 : students + (total_people - students) = total_people := by sorry

  students

/- Proof
sorry
-/

end college_students_count_l2012_201233


namespace lunchroom_total_people_l2012_201235

def num_tables : ℕ := 34
def first_table_students : ℕ := 6
def teacher_count : ℕ := 5

def arithmetic_sum (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem lunchroom_total_people :
  arithmetic_sum num_tables first_table_students 1 + teacher_count = 770 := by
  sorry

end lunchroom_total_people_l2012_201235


namespace simplify_algebraic_expression_l2012_201269

theorem simplify_algebraic_expression (a : ℝ) : 
  3*a + 6*a + 9*a + 6 + 12*a + 15 + 18*a = 48*a + 21 := by
  sorry

end simplify_algebraic_expression_l2012_201269


namespace town_population_growth_l2012_201230

/-- The final population after compound growth --/
def final_population (initial_population : ℕ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_population * (1 + growth_rate) ^ years

/-- Theorem stating the approximate final population after a decade --/
theorem town_population_growth : 
  ∃ (result : ℕ), 
    344251 ≤ result ∧ 
    result ≤ 344252 ∧ 
    result = ⌊final_population 175000 0.07 10⌋ := by
  sorry

end town_population_growth_l2012_201230


namespace clara_final_stickers_l2012_201244

/-- Calculates the number of stickers Clara has left after a series of operations --/
def clara_stickers : ℕ :=
  let initial := 100
  let after_boy := initial - 10
  let after_teacher := after_boy + 50
  let after_classmates := after_teacher - (after_teacher * 20 / 100)
  let exchange_amount := after_classmates / 3
  let after_exchange := after_classmates - exchange_amount + (2 * exchange_amount)
  let give_to_friends := after_exchange / 4
  let remaining := after_exchange - (give_to_friends / 3 * 3)
  remaining

/-- Theorem stating that Clara ends up with 114 stickers --/
theorem clara_final_stickers : clara_stickers = 114 := by
  sorry


end clara_final_stickers_l2012_201244


namespace function_properties_l2012_201222

-- Define the function f(x) and its derivative
def f (x : ℝ) (c : ℝ) : ℝ := x^3 - 3*x^2 + c
def f_derivative (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem function_properties :
  -- f'(x) passes through (0,0) and (2,0)
  f_derivative 0 = 0 ∧ f_derivative 2 = 0 ∧
  -- f(x) attains its minimum at x = 2
  (∀ x : ℝ, f x (-1) ≥ f 2 (-1)) ∧
  -- The minimum value is -5
  f 2 (-1) = -5 :=
sorry

end function_properties_l2012_201222


namespace consecutive_odd_numbers_sum_l2012_201284

theorem consecutive_odd_numbers_sum (n k : ℕ) : n > 0 ∧ k > 0 → 
  (∃ (seq : List ℕ), 
    (∀ i ∈ seq, ∃ j, i = n + 2 * j ∧ j ≤ k) ∧ 
    (seq.length = k + 1) ∧
    (seq.sum = 20 * (n + 2 * k)) ∧
    (seq.sum = 60 * n)) →
  n = 29 ∧ k = 29 := by
sorry

end consecutive_odd_numbers_sum_l2012_201284


namespace first_five_terms_sequence_1_l2012_201278

def a (n : ℕ+) : ℚ := 1 / (4 * n - 1)

theorem first_five_terms_sequence_1 :
  [a 1, a 2, a 3, a 4, a 5] = [1/3, 1/7, 1/11, 1/15, 1/19] := by sorry

end first_five_terms_sequence_1_l2012_201278


namespace volume_of_one_gram_volume_of_one_gram_substance_l2012_201259

-- Define the constants from the problem
def mass_per_cubic_meter : ℝ := 300
def grams_per_kilogram : ℝ := 1000
def cubic_cm_per_cubic_meter : ℝ := 1000000

-- Define the theorem
theorem volume_of_one_gram (mass_per_cubic_meter : ℝ) (grams_per_kilogram : ℝ) (cubic_cm_per_cubic_meter : ℝ) :
  mass_per_cubic_meter * grams_per_kilogram > 0 →
  cubic_cm_per_cubic_meter / (mass_per_cubic_meter * grams_per_kilogram) = 10 / 3 := by
  sorry

-- Apply the theorem to our specific values
theorem volume_of_one_gram_substance :
  cubic_cm_per_cubic_meter / (mass_per_cubic_meter * grams_per_kilogram) = 10 / 3 := by
  apply volume_of_one_gram mass_per_cubic_meter grams_per_kilogram cubic_cm_per_cubic_meter
  -- Prove that mass_per_cubic_meter * grams_per_kilogram > 0
  sorry

end volume_of_one_gram_volume_of_one_gram_substance_l2012_201259


namespace max_value_of_expression_l2012_201247

theorem max_value_of_expression (x y : ℝ) : 
  |x + 1| - |x - 1| - |y - 4| - |y| ≤ -2 := by sorry

end max_value_of_expression_l2012_201247


namespace quadratic_inequality_integer_solution_l2012_201267

theorem quadratic_inequality_integer_solution (z : ℕ) :
  z^2 - 50*z + 550 ≤ 10 ↔ 20 ≤ z ∧ z ≤ 30 :=
by sorry

end quadratic_inequality_integer_solution_l2012_201267


namespace necessary_and_sufficient_condition_l2012_201282

theorem necessary_and_sufficient_condition : 
  (∀ x : ℝ, x^2 - 2*x + 1 = 0 ↔ x = 1) := by sorry

end necessary_and_sufficient_condition_l2012_201282


namespace same_remainder_divisor_l2012_201215

theorem same_remainder_divisor : ∃ (r : ℕ), 
  1108 % 23 = r ∧ 
  1453 % 23 = r ∧ 
  1844 % 23 = r ∧ 
  2281 % 23 = r :=
by sorry

end same_remainder_divisor_l2012_201215


namespace perfect_square_expression_l2012_201279

theorem perfect_square_expression (x : ℝ) : ∃ y : ℝ, x^2 - x + (1/4 : ℝ) = y^2 := by
  sorry

end perfect_square_expression_l2012_201279


namespace probability_is_one_seventh_l2012_201260

/-- The number of students in the group -/
def total_students : ℕ := 7

/-- The number of students who can speak a foreign language -/
def foreign_language_speakers : ℕ := 3

/-- The number of students being selected -/
def selected_students : ℕ := 2

/-- The probability of selecting two students who both speak a foreign language -/
def probability_both_speak_foreign : ℚ :=
  (foreign_language_speakers.choose selected_students) / (total_students.choose selected_students)

theorem probability_is_one_seventh :
  probability_both_speak_foreign = 1 / 7 := by
  sorry

end probability_is_one_seventh_l2012_201260


namespace average_sales_per_month_l2012_201254

def sales_data : List ℝ := [120, 90, 50, 110, 80, 100]

theorem average_sales_per_month :
  (List.sum sales_data) / (List.length sales_data) = 91.67 := by
  sorry

end average_sales_per_month_l2012_201254


namespace point_vector_relations_l2012_201232

/-- Given points A, B, C in ℝ², and points M, N such that CM = 3CA and CN = 2CB,
    prove that M and N have specific coordinates and MN has a specific value. -/
theorem point_vector_relations (A B C M N : ℝ × ℝ) :
  A = (-2, 4) →
  B = (3, -1) →
  C = (-3, -4) →
  M - C = 3 • (A - C) →
  N - C = 2 • (B - C) →
  M = (0, 20) ∧
  N = (9, 2) ∧
  M - N = (9, -18) := by
  sorry

end point_vector_relations_l2012_201232


namespace olympic_medal_awards_l2012_201250

/-- The number of ways to award medals in the Olympic 100-meter finals --/
def medal_award_ways (total_sprinters : ℕ) (american_sprinters : ℕ) (medals : ℕ) : ℕ :=
  let non_american_sprinters := total_sprinters - american_sprinters
  let no_american_medal := Nat.descFactorial non_american_sprinters medals
  let one_american_medal := american_sprinters * medals * (Nat.descFactorial non_american_sprinters (medals - 1))
  no_american_medal + one_american_medal

/-- Theorem stating the number of ways to award medals in the given scenario --/
theorem olympic_medal_awards : 
  medal_award_ways 10 4 3 = 480 := by
  sorry

end olympic_medal_awards_l2012_201250


namespace unique_solution_for_prime_equation_l2012_201245

theorem unique_solution_for_prime_equation :
  ∃! n : ℕ, ∃ p : ℕ, Nat.Prime p ∧ n^2 = p^2 + 3*p + 9 :=
by
  -- The proof would go here
  sorry

end unique_solution_for_prime_equation_l2012_201245


namespace provisions_after_reinforcement_provisions_last_20_days_l2012_201298

theorem provisions_after_reinforcement 
  (initial_garrison : ℕ) 
  (initial_provisions : ℕ) 
  (days_before_reinforcement : ℕ) 
  (reinforcement : ℕ) : ℕ :=
  let remaining_provisions := initial_garrison * (initial_provisions - days_before_reinforcement)
  let total_men := initial_garrison + reinforcement
  remaining_provisions / total_men

theorem provisions_last_20_days 
  (initial_garrison : ℕ) 
  (initial_provisions : ℕ) 
  (days_before_reinforcement : ℕ) 
  (reinforcement : ℕ) :
  initial_garrison = 1000 →
  initial_provisions = 60 →
  days_before_reinforcement = 15 →
  reinforcement = 1250 →
  provisions_after_reinforcement initial_garrison initial_provisions days_before_reinforcement reinforcement = 20 :=
by sorry

end provisions_after_reinforcement_provisions_last_20_days_l2012_201298


namespace chair_difference_l2012_201291

theorem chair_difference (initial_chairs left_chairs : ℕ) : 
  initial_chairs = 15 → left_chairs = 3 → initial_chairs - left_chairs = 12 := by
  sorry

end chair_difference_l2012_201291


namespace problem_solution_l2012_201253

theorem problem_solution (a b c d e : ℕ+) 
  (eq1 : a * b + a + b = 182)
  (eq2 : b * c + b + c = 306)
  (eq3 : c * d + c + d = 210)
  (eq4 : d * e + d + e = 156)
  (prod : a * b * c * d * e = Nat.factorial 10) :
  (a : ℤ) - (e : ℤ) = -154 := by
  sorry

end problem_solution_l2012_201253


namespace total_diagonals_two_polygons_l2012_201299

/-- Number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The first polygon has 100 sides -/
def polygon1_sides : ℕ := 100

/-- The second polygon has 150 sides -/
def polygon2_sides : ℕ := 150

/-- Theorem: The total number of diagonals in a 100-sided polygon and a 150-sided polygon is 15875 -/
theorem total_diagonals_two_polygons : 
  diagonals polygon1_sides + diagonals polygon2_sides = 15875 := by
  sorry

end total_diagonals_two_polygons_l2012_201299


namespace hyperbola_asymptotes_l2012_201286

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and eccentricity e = √5/2, prove that its asymptotes are y = ±(1/2)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (he : Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 / 2) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = (1/2) * x ∨ f x = -(1/2) * x) ∧
  (∀ ε > 0, ∃ M > 0, ∀ x y, x^2/a^2 - y^2/b^2 = 1 → abs x > M →
    abs (y - f x) < ε * abs x) :=
sorry

end hyperbola_asymptotes_l2012_201286


namespace trailing_zeros_310_factorial_l2012_201243

/-- The number of trailing zeros in n! --/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 310! is 76 --/
theorem trailing_zeros_310_factorial :
  trailingZeros 310 = 76 := by
  sorry

end trailing_zeros_310_factorial_l2012_201243


namespace gcd_24_36_l2012_201255

theorem gcd_24_36 : Nat.gcd 24 36 = 12 := by sorry

end gcd_24_36_l2012_201255


namespace interior_edges_sum_for_specific_frame_l2012_201242

/-- Represents a rectangular picture frame -/
structure Frame where
  outerLength : ℝ
  outerWidth : ℝ
  frameWidth : ℝ

/-- Calculates the area of the frame -/
def frameArea (f : Frame) : ℝ :=
  f.outerLength * f.outerWidth - (f.outerLength - 2 * f.frameWidth) * (f.outerWidth - 2 * f.frameWidth)

/-- Calculates the sum of the lengths of the four interior edges -/
def interiorEdgesSum (f : Frame) : ℝ :=
  2 * (f.outerLength - 2 * f.frameWidth) + 2 * (f.outerWidth - 2 * f.frameWidth)

/-- Theorem stating the sum of interior edges for a specific frame -/
theorem interior_edges_sum_for_specific_frame :
  ∃ (f : Frame),
    f.outerLength = 7 ∧
    f.frameWidth = 2 ∧
    frameArea f = 30 ∧
    interiorEdgesSum f = 7 := by
  sorry

end interior_edges_sum_for_specific_frame_l2012_201242


namespace election_result_l2012_201275

/-- Represents an election with three candidates -/
structure Election where
  total_votes : ℕ
  votes_a : ℕ
  votes_b : ℕ
  votes_c : ℕ

/-- Conditions for the specific election scenario -/
def election_conditions (e : Election) : Prop :=
  e.votes_a = (32 * e.total_votes) / 100 ∧
  e.votes_b = (42 * e.total_votes) / 100 ∧
  e.votes_c = e.votes_b - 1908 ∧
  e.total_votes = e.votes_a + e.votes_b + e.votes_c

/-- The theorem to be proved -/
theorem election_result (e : Election) (h : election_conditions e) :
  e.votes_c = (26 * e.total_votes) / 100 ∧ e.total_votes = 11925 := by
  sorry

#check election_result

end election_result_l2012_201275


namespace box_volume_and_area_l2012_201280

/-- A rectangular box with given dimensions -/
structure RectangularBox where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the volume of a rectangular box -/
def volume (box : RectangularBox) : ℝ :=
  box.length * box.width * box.height

/-- Calculate the maximum ground area of a rectangular box -/
def maxGroundArea (box : RectangularBox) : ℝ :=
  box.length * box.width

/-- Theorem about the volume and maximum ground area of a specific rectangular box -/
theorem box_volume_and_area (box : RectangularBox)
    (h1 : box.length = 20)
    (h2 : box.width = 15)
    (h3 : box.height = 5) :
    volume box = 1500 ∧ maxGroundArea box = 300 := by
  sorry

end box_volume_and_area_l2012_201280


namespace first_team_cups_l2012_201292

theorem first_team_cups (total : ℕ) (second : ℕ) (third : ℕ) 
  (h1 : total = 280)
  (h2 : second = 120)
  (h3 : third = 70) :
  ∃ first : ℕ, first + second + third = total ∧ first = 90 := by
  sorry

end first_team_cups_l2012_201292


namespace grasshopper_jump_distance_l2012_201221

theorem grasshopper_jump_distance 
  (frog_jump : ℕ) 
  (grasshopper_extra : ℕ) 
  (h1 : frog_jump = 11) 
  (h2 : grasshopper_extra = 2) : 
  frog_jump + grasshopper_extra = 13 := by
  sorry

end grasshopper_jump_distance_l2012_201221


namespace complement_determines_set_l2012_201236

def U : Set Nat := {0, 1, 2, 4}

theorem complement_determines_set 
  (h : Set.compl {1, 2} = {0, 4}) : 
  ∃ A : Set Nat, A ⊆ U ∧ Set.compl A = {1, 2} ∧ A = {0, 4} := by
  sorry

#check complement_determines_set

end complement_determines_set_l2012_201236


namespace intersection_equality_implies_a_range_l2012_201263

/-- Given sets A and B, prove the range of a when A ∩ B = B -/
theorem intersection_equality_implies_a_range
  (A : Set ℝ)
  (B : Set ℝ)
  (a : ℝ)
  (h_A : A = {x : ℝ | -2 ≤ x ∧ x ≤ 2})
  (h_B : B = {x : ℝ | a < x ∧ x < a + 1})
  (h_int : A ∩ B = B) :
  -2 ≤ a ∧ a ≤ 1 := by
  sorry

end intersection_equality_implies_a_range_l2012_201263


namespace u_floor_formula_l2012_201228

def u : ℕ → ℚ
  | 0 => 2
  | 1 => 5/2
  | (n+2) => u (n+1) * (u n ^ 2 - 2) - u 1

theorem u_floor_formula (n : ℕ) (h : n ≥ 1) :
  ⌊u n⌋ = (2 * (2^n - (-1)^n)) / 3 :=
sorry

end u_floor_formula_l2012_201228


namespace time_to_paint_one_room_l2012_201203

theorem time_to_paint_one_room 
  (total_rooms : ℕ) 
  (painted_rooms : ℕ) 
  (time_for_remaining : ℕ) : 
  total_rooms = 10 → 
  painted_rooms = 8 → 
  time_for_remaining = 16 → 
  (time_for_remaining / (total_rooms - painted_rooms) : ℚ) = 8 := by
  sorry

end time_to_paint_one_room_l2012_201203
